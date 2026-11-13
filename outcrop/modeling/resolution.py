"""Deterministic primitive dependency groups and revalidated resolved instances."""
from dataclasses import dataclass,asdict,replace
from collections.abc import Mapping
import hashlib
import json
import platform
import numpy as np
from .schema import thaw,freeze,NumericModelV2,ResourceSet
from .decode import decode_model,encode_model,read_model_json
from .parameters import FieldRule
from .sampling import SamplingContext,ResolutionRecord,RandomKey
from .resolved import ResolvedModelV2,materialize_parameters
from .compiler import compile_model
from .errors import ModelError,invalid
from . import MODEL_VERSION,BACKEND_VERSION


@dataclass(frozen=True)
class PrimitiveNode:
    id: str
    parameter: object
    rule: FieldRule
    object_id: str
    local_path: str
    assignment_path: str
    @property
    def random(self):return isinstance(self.parameter,Mapping) and self.parameter.get("kind") in {"range","distribution"}


@dataclass(frozen=True)
class RetryGroup:
    id: str
    primitive_paths: tuple
    derived_paths: tuple
    constraint_ids: tuple
    max_attempts: int


def _rule(name,p):
    key=name.split(".")[-1].split("[")[0]
    choice=isinstance(p,Mapping) and p.get("name")=="weighted_choice" or isinstance(p,str)
    return FieldRule(unit="deg" if key.endswith("_deg") else "m" if key.endswith("_m") else None,
        value_type="choice" if choice else "integer" if key in {"count","max_objects","max_attempts"} else "number",
        positive=key in {"radius_m","radii_m","spacing_m","increments_m","extent_m","width_m","depth_m"},
        minimum=0 if key in {"dip_deg","count","fraction","slip_m"} else None,maximum=90 if key=="dip_deg" else 1 if key=="fraction" else None)


def _collect(spec):
    d=thaw(spec.data);nodes={};derived={}
    for i,p in enumerate(d["parameters"]):
        if "derived" in p:derived[p["id"]]=p;continue
        if "value" not in p:invalid(p["id"],"parameter value or derived rule required")
        try:rule=FieldRule(**p["rule"]) if "rule" in p else _rule("value",p["value"])
        except TypeError as e:invalid(p["id"],str(e))
        nodes[p["id"]]=PrimitiveNode(p["id"],p["value"],rule,p["id"],"value",f"parameters[{i}].value")
    def walk(value,assignment,semantic,object_id,local):
        if isinstance(value,dict):
            if value.get("kind") in {"fixed","range","distribution"}:
                oid=semantic;nodes[oid]=PrimitiveNode(oid,value,_rule(local,value),object_id,local,assignment);return
            for key,v in value.items():
                if key not in {'metadata','source'}:walk(v,f"{assignment}.{key}",f"{semantic}.{key}",object_id,f"{local}.{key}" if local else key)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,f"{assignment}[{i}]",f"{semantic}[{i}]",object_id,f"{local}[{i}]")
    for table in ("geometry","bodies","interfaces","features","events","exposures","appearances"):
        for i,obj in enumerate(d.get(table,[])):walk(obj,f"{table}[{i}]",f"{table}.{obj['id']}",obj["id"],"")
    def inherit_units(value,path):
        if isinstance(value,dict):
            if value.get("kind")=="ref":
                consumer=_rule(path,value);oid=value["id"]
                if consumer.unit is not None and oid in nodes:
                    node=nodes[oid]
                    if node.rule.unit is not None and node.rule.unit!=consumer.unit:invalid(path,"incompatible parameter consumer units")
                    nodes[oid]=replace(node,rule=replace(node.rule,unit=consumer.unit))
            for key,v in value.items():
                if key not in {'metadata','source'}:inherit_units(v,f"{path}.{key}" if path else key)
        elif isinstance(value,list):
            for i,v in enumerate(value):inherit_units(v,f"{path}[{i}]")
    inherit_units(d,"")
    for oid,p in derived.items():
        kind=p['derived']['kind'];unit='deg' if kind=='orientation_rhr' else 'm' if kind=='positive_spacing_cumsum' else None
        if unit is not None:
            for dependency in p.get('dependencies',[]):
                if dependency in nodes:
                    node=nodes[dependency]
                    if node.rule.unit not in {None,unit}:invalid(oid,'derived dependency unit mismatch')
                    nodes[dependency]=replace(node,rule=replace(node.rule,unit=unit))
    return nodes,derived


def _primitive_dependencies(name,nodes,derived,active=None):
    if name in nodes:return {name}
    if name not in derived:invalid(name,"unknown constraint/derived parameter")
    active=set() if active is None else active
    if name in active:invalid(name,"derived parameter cycle")
    return set().union(*(_primitive_dependencies(x,nodes,derived,active|{name}) for x in derived[name].get("dependencies",[])))


def compile_retry_groups(spec,dependency_graph=None):
    nodes,derived=_collect(spec);random=sorted(k for k,n in nodes.items() if n.random);parent={k:k for k in random};constraints=thaw(spec.data).get("constraints",[]);limit=spec.get("sampling",{}).get("max_attempts",32)
    if type(limit) is not int or limit<=0:invalid("sampling.max_attempts","positive integer required")
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    def union(names):
        names=sorted(names)
        if names:
            r=min(root(x) for x in names)
            for x in names:parent[root(x)]=r
    deps={}
    for c in constraints:
        dependencies=set().union(*(_primitive_dependencies(x,nodes,derived) for x in c.get("parameters",[])))
        deps[c["id"]]=dependencies&set(random)
        if c.get("generative",False):union(deps[c["id"]])
    explicit=thaw(spec.data).get("sampling",{}).get("groups",[])
    for group in explicit:
        names=set(group["parameters"])
        if not names<=set(random):invalid(group["id"],"sampling group must name random primitives")
        for dep in deps.values():
            if dep&names and not dep<=names:invalid(group["id"],"sampling group omits shared constraint dependencies")
        union(names)
    groups={}
    for name in random:groups.setdefault(root(name),[]).append(name)
    result=[]
    for names in groups.values():
        key=hashlib.sha256(json.dumps(names,separators=(",",":")).encode()).hexdigest()[:16]
        derived_paths=tuple(sorted(x for x in derived if _primitive_dependencies(x,nodes,derived)&set(names)))
        constraint_ids=tuple(sorted(k for k,v in deps.items() if v&set(names)))
        result.append(RetryGroup("group:"+key,tuple(names),derived_paths,constraint_ids,limit))
    return tuple(sorted(result,key=lambda g:g.primitive_paths))


def _derive(values,derived):
    pending=dict(derived)
    while pending:
        progress=False
        for oid,p in list(pending.items()):
            deps=p.get("dependencies",[])
            if not all(x in values for x in deps):continue
            rule=p["derived"];kind=rule["kind"];args=[values[x] for x in deps]
            if kind=="orientation_rhr":
                from .kinematics import resolve_orientation
                result=resolve_orientation({"dip_direction_deg":args[0],"dip_deg":args[1]})
            elif kind=="positive_spacing_cumsum":
                if any(x<=0 for x in args):invalid(oid,"positive spacing increments required")
                result=np.r_[float(rule.get("start_m",0)),float(rule.get("start_m",0))+np.cumsum(args)].tolist()
            elif kind=="fault_motion":
                from .kinematics import resolve_fault_motion
                def replace_refs(v):
                    if isinstance(v,dict):return values[v["id"]] if v.get("kind")=="ref" else {k:replace_refs(x) for k,x in v.items()}
                    if isinstance(v,list):return [replace_refs(x) for x in v]
                    return v
                result=asdict(resolve_fault_motion(replace_refs(rule["orientation"]),replace_refs(rule["motion"]),moving_side=rule["moving_side"]))
            else:invalid(oid,"unknown derived rule")
            if "component" in rule:result=result[rule["component"]]
            if "index" in rule:result=result[rule["index"]]
            values[oid]=result;del pending[oid];progress=True
        if not progress:invalid("parameters","unresolved derived dependency cycle/reference")


def _check_arithmetic(numeric,values):
    for c in numeric.get("constraints",()):
        if c['kind']=='relation':continue
        if c['kind']!='linear':invalid(c['id'],'unhandled constraint kind')
        names=c["parameters"];coeff=c["coefficients"]
        if len(names)!=len(coeff):invalid(c["id"],"one coefficient per parameter required")
        lhs=sum(float(a)*float(values[b]) for a,b in zip(coeff,names));rhs=float(c["value"]);op=c["operator"]
        passed={"le":lhs<=rhs,"lt":lhs<rhs,"ge":lhs>=rhs,"gt":lhs>rhs,"eq":lhs==rhs}.get(op)
        if passed is None:invalid(c["id"],"known comparison operator required")
        if not passed:raise ModelError(c["id"],"constraint_failed","linear constraint failed",{"constraint_id":c["id"],"lhs":lhs,"rhs":rhs})


def _check_relations(compiled):
    # Stage mutations and final checks are performed in event chronology by
    # the compiler, rather than replaying them on already-finished snapshots.
    return compiled,compiled.relation_results


def _motions(compiled):
    from .faults import compile_fault_map
    result={}
    for frozen in compiled.plan.events:
        e=thaw(frozen)
        if e["kind"]=="fault":
            s=dict(e["fault"]) if isinstance(e["fault"],dict) else {"geometry":e["fault"]};s.update({k:e[k] for k in ("motion","moving_side","support","edge_policy") if k in e});s["id"]=e["id"]
            result[e["id"]]=asdict(compile_fault_map(s,compiled.kernel).motion)
    return result


def resolve_model(spec,context,*,range_policy=None,capabilities=None):
    from . import sampling
    _verified_resources(spec,capabilities)
    nodes,derived=_collect(spec);groups=compile_retry_groups(spec);group_by_node={x:g for g in groups for x in g.primitive_paths};attempts={g.id:0 for g in groups};values={};records={};history=[]
    policy=range_policy or context.range_policy or spec.get("sampling",{}).get("range_policy")
    def sample_nodes(names):
        for name in sorted(names):
            node=nodes[name];group=group_by_node.get(name);attempt=attempts[group.id] if group else 0
            sampled=sampling.sample_parameter(node.parameter,node.rule,replace(context,object_id=node.object_id,parameter_path=node.local_path,range_policy=policy),attempt=attempt,element_id="0")
            values[name]=sampled.value;records[name]=replace(sampled.record,path=node.assignment_path)
    sample_nodes(nodes)
    while True:
        current=dict(values);_derive(current,derived);assign=dict(current)
        for name,node in nodes.items():assign[node.assignment_path]=current[name]
        for i,p in enumerate(thaw(spec.data)["parameters"]):
            if p["id"] in derived:assign[f"parameters[{i}].value"]=current[p["id"]]
        numeric=materialize_parameters(spec,assign)
        # Store the derived value explicitly; no re-evaluation is needed in replay.
        raw=encode_model(numeric)
        for p in raw["parameters"]:
            if p["id"] in derived:p["value"]=current[p["id"]]
        numeric=NumericModelV2(decode_model(raw).data);numeric,selectors=_resolve_selectors(numeric,context,capabilities)
        try:
            _check_arithmetic(numeric,current);compiled=compile_model(numeric,capabilities);compiled,evidence=_check_relations(compiled);break
        except ModelError as error:
            if error.code not in {"constraint_failed","unresolved"}:raise
            constraints=thaw(spec.data)["constraints"];cid=error.details.get("constraint_id");relation=error.details.get("relation_id")
            generating=[c for c in constraints if c.get("generative",False) and (c["id"]==cid or c.get("relation")==relation and relation is not None)]
            affected={group_by_node[x].id for c in generating for p in c.get("parameters",[]) for x in _primitive_dependencies(p,nodes,derived) if x in group_by_node}
            if not affected:raise
            for group in groups:
                if group.id not in affected:continue
                history.append({"group":group.id,"attempt":attempts[group.id],"parameters":list(group.primitive_paths),"reason":str(error)})
                if attempts[group.id]+1>=group.max_attempts:raise ModelError(error.field,"constraint_failed","joint sampling attempt budget exhausted",{"attempts":history}) from error
                attempts[group.id]+=1;sample_nodes(group.primitive_paths)
    resource_hashes={r["id"]:r["sha256"] for r in numeric["resources"]}
    versions={"model":MODEL_VERSION,"backend":BACKEND_VERSION,"python":platform.python_version(),"numpy":np.__version__,"rng":context.rng_version,"sampler":context.sampler_version}
    return ResolvedModelV2(numeric,context.root_seed,context.sample_id,tuple(records[k] for k in sorted(records)),_motions(compiled),evidence,resource_hashes,versions,selectors,tuple(history),spec)


def json_value(v):
    from dataclasses import is_dataclass
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Mapping):return {k:json_value(x) for k,x in v.items()}
    if is_dataclass(v):return {k:json_value(x) for k,x in asdict(v).items()}
    if isinstance(v,(tuple,list)):return [json_value(x) for x in v]
    return v


def resolved_to_dict(resolved):
    return {"schema_version":2,"kind":"resolved_model","root_seed":resolved.root_seed,"sample_id":resolved.sample_id,"source_spec":encode_model(resolved.source_spec),"numeric_model":encode_model(resolved.numeric),"records":json_value(resolved.records),"motions":json_value(resolved.motions),"evidence":json_value(resolved.evidence),"resource_hashes":dict(resolved.resource_hashes),"versions":dict(resolved.versions),"selectors":dict(resolved.selectors),"attempt_history":json_value(resolved.attempt_history)}


def read_resolved(path,*,capabilities=None):
    d=read_model_json(path);allowed={"schema_version","kind","root_seed","sample_id","source_spec","numeric_model","records","motions","evidence","resource_hashes","versions","selectors","attempt_history"}
    if not isinstance(d,dict) or set(d)-allowed or d.get("schema_version")!=2 or d.get("kind")!="resolved_model":invalid("resolved_model","versioned resolved contract required")
    from .validation import required
    required(d,allowed-{'selectors','attempt_history'},'resolved_model')
    SamplingContext(d['root_seed'],d['sample_id'])
    for key in ('versions','motions','resource_hashes','selectors'):
        if not isinstance(d.get(key,{}),dict):invalid(key,'object required')
    if not isinstance(d['records'],list):invalid('records','resolution record array required')
    expected={'model':MODEL_VERSION,'backend':BACKEND_VERSION,'rng':'rng-v2','sampler':'sampler-v2'}
    if any(d['versions'].get(k)!=v for k,v in expected.items()):invalid('versions','unsupported resolved model/backend/RNG/sampler version')
    numeric=NumericModelV2(decode_model(d["numeric_model"]).data)
    _verified_resources(numeric,capabilities)
    nodes,derived=_collect(numeric)
    for n in nodes.values():
        if isinstance(n.parameter,Mapping):invalid(n.assignment_path,"resolved values cannot retain parameter tags")
        n.rule.validate(n.parameter,n.assignment_path)
    values=_semantic_values(numeric);_check_arithmetic(numeric,values)
    expected_derived=dict(values)
    for oid in derived:expected_derived.pop(oid,None)
    _derive(expected_derived,derived)
    if any(json_value(expected_derived[oid])!=json_value(values.get(oid)) for oid in derived):invalid('parameters.derived','stored derived value differs from declared rule')
    selectors=d.get('selectors',{})
    for relation in numeric['relations']:
        if 'selector' in relation:
            if relation['id'] not in selectors or list(relation['selector'].get('ids',()))!=selectors[relation['id']]['ids']:invalid('selectors','resolved selected joint identity mismatch')
    compiled=compile_model(numeric,capabilities);compiled,evidence=_check_relations(compiled);motions=_motions(compiled)
    if json_value(motions)!=d["motions"]:raise ModelError("motions","constraint_failed","canonical motion mismatch in resolved replay")
    hashes={r["id"]:r["sha256"] for r in numeric["resources"]}
    if hashes!=d["resource_hashes"]:raise ModelError("resources","resource_mismatch","resolved resource identity mismatch")
    records=[]
    for r in d["records"]:
        if not isinstance(r,dict) or set(r)-{'path','value','key','attempt','sampler','draws'}:invalid('records','strict resolution record required')
        required(r,('path','value','key','attempt','sampler'),'records')
        if type(r['attempt']) is not int or r['attempt']<0 or type(r.get('draws',0)) is not int or r.get('draws',0)<0:invalid('records','nonnegative attempt/draw count required')
        if r['sampler'] not in {'fixed','midpoint','uniform','integer_uniform','circular_uniform','weighted_choice','truncated_normal','lognormal'}:invalid('records.sampler','supported sampler required')
        if json_value(_value_at_path(encode_model(numeric),r['path']))!=r['value']:invalid('records.value','record does not match resolved numeric value')
        if r['key'] is not None and (not isinstance(r['key'],dict) or set(r['key'])!=set(RandomKey.__dataclass_fields__)):invalid('records.key','strict random key record required')
        key=RandomKey(**r["key"]) if r.get("key") is not None else None;records.append(ResolutionRecord(r["path"],r["value"],key,r["attempt"],r["sampler"],r.get("draws",0)))
        if key is not None and (key.root_seed!=d['root_seed'] or key.sample_id!=d['sample_id'] or key.group_attempt!=r['attempt'] or key.rng_version!='rng-v2' or key.sampler_version!='sampler-v2'):invalid('records.key','random identity mismatch')
    source=decode_model(d['source_spec']);_check_source_assignment(source,numeric,records,selectors)
    return ResolvedModelV2(numeric,d["root_seed"],d["sample_id"],tuple(records),motions,evidence,hashes,d["versions"],d.get("selectors",{}),tuple(d.get("attempt_history",[])),source)


def _check_source_assignment(source,numeric,records,selectors):
    nodes,derived=_collect(source);by_path={r.path:r for r in records}
    if len(by_path)!=len(records) or set(by_path)!={n.assignment_path for n in nodes.values()}:invalid('records','one resolution record per original primitive required')
    values={}
    for oid,n in nodes.items():
        value=by_path[n.assignment_path].value;n.rule.validate(value,n.assignment_path);decl=n.parameter
        if isinstance(decl,Mapping):
            kind=decl['kind']
            if kind=='fixed' and value!=decl['value']:invalid(n.assignment_path,'fixed source value mismatch')
            if kind=='distribution' and decl.get('name')=='circular_uniform':
                if not decl.get('full_circle',False) and (value-decl['min'])%360>(decl['max']-decl['min'])%360:invalid(n.assignment_path,'resolved value outside declared circular support')
            elif kind=='range' or kind=='distribution' and decl.get('name') in {'uniform','integer_uniform','truncated_normal'}:
                if not decl.get('full_circle',False) and not (decl['min']<=value<=decl['max']):invalid(n.assignment_path,'resolved value outside declared support')
            if kind=='distribution' and decl.get('name')=='weighted_choice' and value not in decl['values']:invalid(n.assignment_path,'resolved choice outside declared support')
            if kind=='distribution' and decl.get('name')=='lognormal':
                if value<=0 or value<decl.get('min',0) or value>decl.get('max',float('inf')):invalid(n.assignment_path,'resolved lognormal value outside declared support')
        elif value!=decl:invalid(n.assignment_path,'fixed primitive source mismatch')
        values[oid]=value
    _derive(values,derived);assign=dict(values);assign.update({n.assignment_path:values[oid] for oid,n in nodes.items()});raw=encode_model(materialize_parameters(source,assign))
    for p in raw['parameters']:
        if p['id'] in derived:p['value']=values[p['id']]
    for relation in raw['relations']:
        if 'selector' in relation:
            record=selectors.get(relation['id'])
            if not isinstance(record,dict) or record.get('declared')!=relation['selector'] or not isinstance(record.get('ids'),list):invalid('selectors','declared selector association mismatch')
            declaration=record['declared'];count=declaration.get('count')
            if count is not None and len(record['ids'])!=count:invalid('selectors','selected count mismatch')
            relation['selector']={'ids':record['ids']}
    if encode_model(decode_model(raw))!=encode_model(numeric):invalid('source_spec','original source assignments do not reproduce numeric model')


def _verified_resources(spec,capabilities):
    declarations=spec['resources'];available=capabilities.by_id if isinstance(capabilities,ResourceSet) else {}
    for ref in declarations:
        resource=available.get(ref['id'])
        if resource is None or resource.ref.sha256!=ref['sha256'] or hashlib.sha256(resource.bytes).hexdigest()!=ref['sha256']:raise ModelError(ref['id'],'resource_mismatch','declared resource needs matching verified bytes')


def _value_at_path(raw,path):
    import re
    if not isinstance(path,str) or not path:invalid('records.path','assignment path required')
    tokens=re.findall(r'([^.[\]]+)|\[(\d+)\]',path);value=raw
    try:
        for key,index in tokens:value=value[int(index)] if index else value[key]
    except (KeyError,TypeError,IndexError,ValueError):invalid('records.path','invalid numeric assignment path')
    return value


def _semantic_values(numeric):
    d=encode_model(numeric);values={p['id']:p['value'] for p in d['parameters'] if 'value' in p}
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():
                if k not in {'metadata','source'}:walk(v,path+'.'+k)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,path+f'[{i}]')
        else:values[path]=value
    for table in ('geometry','bodies','interfaces','features','events','exposures','appearances'):
        for obj in d[table]:walk(obj,table+'.'+obj['id'])
    return values


def _resolve_selectors(numeric,context,capabilities):
    from .sampling import rng_for_key
    from .joints import expand_joint_set
    from .kernel.analytic import compile_geometry
    raw=encode_model(numeric);features={f['id']:f for f in raw['features']};selectors={};kernel=None
    for relation in raw['relations']:
        if 'selector' not in relation:continue
        selector=relation['selector'];feature=features.get(relation.get('source'))
        if feature is None or feature['kind']!='joint_set':invalid(relation['id'],'selector requires a stable joint set')
        if kernel is None:kernel=compile_geometry(numeric,capabilities)
        candidates=sorted(j.interface.id for j in expand_joint_set(feature,kernel));count=selector.get('count')
        if 'ids' in selector:
            chosen=sorted(selector['ids'])
            if set(chosen)-set(candidates):invalid(relation['id'],'selected IDs outside joint set')
            keys=[]
        else:
            if count is None:
                fraction=selector.get('fraction')
                if type(fraction) not in (int,float) or not 0<=fraction<=1:invalid(relation['id'],'fraction in [0,1] required')
                count=int(np.floor(fraction*len(candidates)))
            if type(count) is not int or not 0<=count<=len(candidates):invalid(relation['id'],'selected count within joint set required')
            priorities=[];keys=[]
            for oid in candidates:
                key=RandomKey(context.rng_version,context.root_seed,context.sample_id,relation['id'],'selector',context.sampler_version,0,oid);keys.append(asdict(key));priorities.append((float(rng_for_key(key).random()),oid))
            chosen=sorted(oid for priority,oid in sorted(priorities)[:count])
        selectors[relation['id']]={'ids':chosen,'declared':selector,'keys':keys,'policy':'stable-joint-priority-v2; fraction count is floor(n*fraction)'};relation['selector']={'ids':chosen}
    return NumericModelV2(decode_model(raw).data),selectors
