from dataclasses import dataclass,replace
from .events import compile_events
from .query import initial_snapshot
from .features import compile_features
from .kernel.analytic import compile_geometry
from .schema import ResourceSet
from .errors import invalid,ModelError


@dataclass(frozen=True)
class CompiledModel:
    numeric: object
    kernel: object
    plan: object
    snapshots: dict
    final_stage_id: str
    relation_results: tuple = ()


def compile_model(numeric,capabilities=None,*,request=None):
    from .schema import thaw
    from .relations import execute_relation,certify_relation
    d=thaw(numeric.data);resources=capabilities if isinstance(capabilities,ResourceSet) else ResourceSet.empty();kernel=compile_geometry(numeric,resources,request);plan=compile_events(numeric);initial=replace(initial_snapshot(numeric,kernel),tolerance=kernel.request.tolerance);current=compile_features(numeric,initial,kernel);snapshots={};results=[];executed=set()
    stages={'initial'}|{e.get('output_stage',e['id']) for e in d['events']}
    for feature in d['features']:
        if feature['kind'] in {'fold','intrusion'}:
            event=next((e for e in d['events'] if e['id']==feature['event']),None)
            if event is None or event['kind']!=feature['kind']:invalid(feature['id'],'feature requires its matching executable event')
        stage=feature.get('at_stage','initial')
        if stage not in stages:invalid(feature['id'],'unknown feature creation stage')
        if stage!='initial' and not any(e['kind']=='create_features' and e.get('output_stage',e['id'])==stage and feature['id'] in e.get('features',[]) for e in d['events']):invalid(feature['id'],'late feature requires an explicit creation event')
    for body in d['bodies']:
        if 'overlap_policy' in body:raise ModelError(body['id'],'unsupported','initial body overlap policy requires an explicit intrusion/fault event')
        for oid in body.get('fabrics',[]):
            feature=next((f for f in d['features'] if f['id']==oid),None)
            if feature is None or feature['kind'] not in {'bedding','foliation'} or body['id'] not in feature['hosts']:invalid(body['id'],'body fabric must be an actual hosted bedding/foliation feature')
    def relations_at(state,stage):
        for r in sorted(d['relations'],key=lambda x:x['id']):
            if r.get('at_stage','final')!=stage:continue
            for expanded in expand_relation(state,r):
                execution=execute_relation(state,expanded,kernel);state=execution.snapshot;result=execution.result;results.append(result)
                if result.status!='pass' and r.get('required',True):raise ModelError(r['id'],'constraint_failed','required relation failed',{'relation_id':r['id'],'evidence':result.evidence.reason})
            executed.add(r['id'])
        return state
    current=relations_at(current,'initial');snapshots['initial']=current
    for frozen in plan.events:
        from .schema import thaw
        event=thaw(frozen);kind=event["kind"]
        if 'frame' in event:
            import numpy as np
            frame=event['frame'].get('matrix') if isinstance(event['frame'],dict) else event['frame']
            if not np.array_equal(frame,np.eye(4)):raise ModelError(event['id'],'unsupported','event frame currently requires ENU identity; use geometry/map frame explicitly')
        if "input_stage" in event:
            source=snapshots[event["input_stage"]]
            for oid in event.get("targets",[]):
                if current.bodies[oid].version!=source.bodies[oid].version:invalid(event["id"],"input snapshot conflicts with current target version")
        if kind=="intrusion":
            from .intrusions import apply_intrusion
            current=apply_intrusion(current,event,kernel)
        elif kind in {"fold","deformation"}:
            from .deformations import apply_deformation
            current=apply_deformation(current,event,kernel)
        elif kind=='field_deformation':
            from .deformations import apply_field_deformation
            current=apply_field_deformation(current,event,kernel)
        elif kind=='create_features':
            ids=event.get('features',[])
            if not ids:invalid(event['id'],'explicit feature IDs required')
            for oid in ids:
                record=next((f for f in d['features'] if f['id']==oid),None)
                if record is None or record.get('at_stage')!=event.get('output_stage',event['id']):invalid(event['id'],'late feature must declare matching creation stage')
            current=compile_features(numeric,replace(current,id=event.get('output_stage',event['id'])),kernel,set(ids))
        elif kind=="fault":
            from .faults import apply_fault
            current=apply_fault(current,event,kernel)
        elif kind in {'cuts','truncates'}:
            relation={'id':event['id'],'kind':kind,'targets':event['targets']}
            if kind=='cuts':relation['source']=event['geometry']
            else:relation.update({k:event[k] for k in ('cutter','remove_side') if k in event})
            execution=execute_relation(current,relation,kernel)
            if execution.result.status!='pass':raise ModelError(event['id'],'constraint_failed','event relation failed')
            current=replace(execution.snapshot,id=event.get('output_stage',event['id']))
        else:raise ModelError(event["id"],"unsupported",f"event backend not implemented for {kind}")
        current=relations_at(current,current.id);snapshots[current.id]=current
    current=relations_at(current,'final');snapshots[current.id]=current
    if len(executed)!=len(d['relations']):invalid('relations','relation references an unavailable stage')
    for r in d['relations']:
        if not r.get('final_check',False) or r.get('at_stage','final')=='final':continue
        for expanded in expand_relation(current,r):
            if r['kind']=='terminates_at':
                from .termination import verify_final_termination
                result=verify_final_termination(current,expanded,kernel)
            elif r['kind'] in {'contains','contacts','offsets','cuts','truncates'}:result=certify_relation(current,expanded,kernel)
            else:raise ModelError(r['id'],'unsupported','final recheck not implemented for this relation kind')
            results.append(result)
            if result.status!='pass' and r.get('required',True):raise ModelError(r['id'],'constraint_failed','required final relation check failed',{'relation_id':r['id'],'evidence':result.evidence.reason})
    for constraint in d['constraints']:
        if constraint['kind']!='relation':continue
        matches=[r for r in results if r.id==constraint['relation'] or r.id.startswith(constraint['relation']+':')]
        if not matches or any(r.status!='pass' for r in matches):raise ModelError(constraint['id'],'constraint_failed','relation constraint failed',{'constraint_id':constraint['id']})
    return CompiledModel(numeric,kernel,plan,snapshots,current.id,tuple(results))


def expand_relation(snapshot,relation):
    source=relation.get('source')
    if source in snapshot.interfaces or not isinstance(source,str):return (relation,)
    candidates=sorted(oid for oid,s in snapshot.interfaces.items() if s.feature_id==source or oid.startswith(source+':joint:'))
    if not candidates:return (relation,)
    selector=relation.get('selector',{})
    if set(selector)-{'ids'}:invalid(relation['id'],'random selectors must be resolved before event compilation')
    ids=selector.get('ids',candidates)
    if len(set(ids))!=len(ids) or set(ids)-set(candidates):invalid(relation['id'],'selector IDs outside stable feature set')
    return tuple({**relation,'id':relation['id']+':'+oid,'source':oid} for oid in sorted(ids))
