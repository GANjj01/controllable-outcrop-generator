from dataclasses import dataclass,replace
import numpy as np
from .constraints import ConstraintEvidence
from .kernel.types import GeometryRequest,AABB
from .kernel.csg import CSGRegion
from .kernel.sets import certify_empty
from .kernel.boundary import common_region_facets,clip_surface_to_region,region_boundary_cells
from .kernel.mesh import MeshSurface
from .kernel.intersections import IntersectionResult,intersect_surfaces
from .kernel.surface_arrangement import SurfaceCells,cells_from_polygons
from .query import InterfaceState
from .errors import ModelError,invalid,unresolved


@dataclass(frozen=True)
class RelationResult:
    id: str
    kind: str
    stage_id: str
    status: str
    inputs: tuple
    outputs: tuple
    intersection: object | None
    retained_components: tuple
    incident_regions: tuple
    evidence: ConstraintEvidence


@dataclass(frozen=True)
class RelationExecution:
    snapshot: object
    result: RelationResult


def _region(snapshot,kernel,oid):
    if oid in snapshot.bodies:return snapshot.bodies[oid].region
    if oid in kernel.regions:return kernel.regions[oid]
    invalid(oid,"region/body reference required")


def _surface(snapshot,kernel,oid):
    if oid in snapshot.interfaces:return snapshot.interfaces[oid].geometry
    if oid in kernel.surfaces:return kernel.surfaces[oid]
    invalid(oid,"actual interface/surface reference required")


def supported_surface(snapshot,kernel,oid,request,geometry_override=None):
    surface=_surface(snapshot,kernel,geometry_override or oid)
    if oid in snapshot.interfaces:
        state=snapshot.interfaces[oid];support=state.support
        if state.hosts:support=CSGRegion(oid+':actual-hosts','intersection',(support,CSGRegion(oid+':host-union','union',tuple(snapshot.bodies[h].region for h in state.hosts))))
        cells=clip_surface_to_region(surface,support,request)
        return MeshSurface(oid,cells.vertices,cells.faces,getattr(cells,'exact_vertices',None),cells.position_error_m)
    return surface


def _cutter(snapshot,kernel,relation):
    oid=relation['cutter'];side=relation.get('remove_side')
    if oid in kernel.surfaces:
        from .kernel.analytic import PlaneSurface,FieldRegion,OffsetField
        surface=kernel.surfaces[oid]
        if not isinstance(surface,PlaneSurface) or surface.extent_m is not None or side not in {'positive','negative'}:invalid('remove_side','infinite plane cutter requires explicit positive/negative side')
        return FieldRegion(relation['id']+':cutter',OffsetField(surface.field,0,-1 if side=='positive' else 1),snapshot.domain)
    if side not in {None,'inside'}:invalid('remove_side','region cutter removes its inside')
    return _region(snapshot,kernel,oid)


def _result(snapshot,relation,status,*,intersection=None,inputs=(),outputs=(),method="exact arrangement",witness=None,reason=""):
    evidence=ConstraintEvidence(status,np.empty((0,3)) if witness is None else np.asarray(witness).reshape(-1,3),0.,method,snapshot.tolerance.length_m,reason)
    return RelationResult(relation["id"],relation["kind"],snapshot.id,status,tuple(inputs),tuple(outputs),intersection,(),(),evidence)


def certify_relation(snapshot,relation,kernel):
    kind=relation["kind"];request=replace(kernel.request,tolerance=snapshot.tolerance)
    if kind=='cuts':return execute_relation(snapshot,relation,kernel).result
    if kind=='truncates':
        cutter=_cutter(snapshot,kernel,relation)
        for target in relation['targets']:
            proof=certify_empty(CSGRegion(relation['id']+':remaining','intersection',(_region(snapshot,kernel,target),cutter)),request)
            if proof.status!='pass':return _result(snapshot,relation,proof.status,witness=proof.witness_points,reason='matter remains in removed region')
        return _result(snapshot,relation,'pass',method='empty-interior proof of removed matter')
    if kind=="contains":
        outer=_region(snapshot,kernel,relation["source"]);inner=_region(snapshot,kernel,relation["target"])
        try:
            cells=region_boundary_cells(inner,request);probes=cells.vertices
        except ModelError as error:
            if error.code!="unsupported":raise
            if relation.get('boundary_inclusion','include')=='exclude':unresolved(relation['id'],'strict curved containment needs a boundary certificate')
            probes=np.empty((0,3))
        if len(probes):
            result=outer.classify(probes,snapshot.tolerance)
            bad=result.outside|(result.boundary if relation.get("boundary_inclusion","include")=="exclude" else False)
            if bad.any():return _result(snapshot,relation,"fail",witness=probes[bad],method="input-boundary counterexample",inputs=(relation["source"],relation["target"]))
        difference=CSGRegion(relation["id"]+":violation","difference",(inner,outer));proof=certify_empty(difference,request)
        return _result(snapshot,relation,proof.status,witness=proof.witness_points,method=proof.method,inputs=(relation["source"],relation["target"]))
    if kind=="contacts":
        a,b=relation["source"],relation["target"];regions={a:_region(snapshot,kernel,a),b:_region(snapshot,kernel,b)}
        box=AABB(tuple(np.min([r.bounds().minimum for r in regions.values()],axis=0)),tuple(np.max([r.bounds().maximum for r in regions.values()],axis=0)))
        facets=common_region_facets(regions,box,request);polygons=[f.polygon for f in facets if set(f.incident)=={a,b}]
        if polygons:
            cells=cells_from_polygons(polygons,[relation["id"]]*len(polygons));intersection=IntersectionResult("area",np.empty((0,2,3)),np.empty((0,3)),cells,{"method":"shared occupancy facets"})
            return _result(snapshot,relation,"pass",intersection=intersection,inputs=(a,b))
        ca,cb=(region_boundary_cells(r,request) for r in regions.values());intersection=intersect_surfaces(MeshSurface(a,ca.vertices,ca.faces),MeshSurface(b,cb.vertices,cb.faces),request)
        return _result(snapshot,relation,"fail",intersection=intersection,inputs=(a,b),reason="no positive-area material adjacency")
    if kind=="offsets":
        eid=relation["event"]
        branches=[b for b in snapshot.branches if eid in b.lineage];affected={b.body_id for b in branches}
        requested=set(relation.get('targets',[])) or ({relation['target']} if 'target' in relation else affected)
        if 'source' in relation and relation['source'] in snapshot.bodies:requested.add(relation['source'])
        present=bool(requested) and requested<=affected
        for oid in requested:
            pair=[b for b in branches if b.body_id==oid]
            moving=[b for b in pair if b.map_chain];reference=[b for b in pair if not b.map_chain]
            if len(moving)!=1 or len(reference)!=1:present=False;continue
            moved_matter=certify_empty(moving[0].source_region,request);reference_matter=certify_empty(reference[0].source_region,request)
            if moved_matter.status!='fail' or reference_matter.status!='fail':present=False;continue
            q=moved_matter.witness_points;p=q.copy()
            for mapping in moving[0].map_chain:p=mapping.forward(p)
            if np.linalg.norm(p-q)<=snapshot.tolerance.length_m:present=False
        return _result(snapshot,relation,"pass" if present else "fail",inputs=(eid,*sorted(requested)),method="paired predecessor branches and nonzero executed map",reason='' if present else 'requested bodies lack executed nonzero offset branches')
    raise ModelError(relation["id"],"unsupported",f"relation certificate not implemented for {kind}")


def execute_relation(snapshot,relation,kernel):
    request=replace(kernel.request,tolerance=snapshot.tolerance);kind=relation["kind"]
    if kind in {"contains","contacts","offsets"}:return RelationExecution(snapshot,certify_relation(snapshot,relation,kernel))
    if kind=="cuts":
        surface=supported_surface(snapshot,kernel,relation["source"],request);targets=tuple(relation["targets"]);support=CSGRegion(relation["id"]+":targets","union",tuple(_region(snapshot,kernel,t) for t in targets));cells=clip_surface_to_region(surface,support,request)
        intersection=IntersectionResult("area" if len(cells.faces) else "empty",np.empty((0,2,3)),np.empty((0,3)),cells,{"method":"actual surface clipped to targets"})
        status="pass" if len(cells.faces) or not relation.get("required_intersection",True) else "fail";interfaces=dict(snapshot.interfaces);oid=relation["id"]+":cut"
        if len(cells.faces):interfaces[oid]=InterfaceState(oid,MeshSurface(oid,cells.vertices,cells.faces,getattr(cells,'exact_vertices',None),cells.position_error_m),support,targets,None,snapshot.id,(relation["id"],))
        result=_result(snapshot,relation,status,intersection=intersection,inputs=(relation["source"],*targets),outputs=(oid,) if len(cells.faces) else ())
        return RelationExecution(replace(snapshot,interfaces=interfaces,relations=snapshot.relations+(result,)),result)
    if kind=="truncates":
        cutter=_cutter(snapshot,kernel,relation);bodies=dict(snapshot.bodies)
        for target in relation["targets"]:
            body=bodies[target];bodies[target]=replace(body,region=CSGRegion(target+"@"+relation["id"],"difference",(body.region,cutter)),version=relation['id'],lineage=body.lineage+(relation["id"],))
        result=_result(snapshot,relation,"pass",inputs=(relation["cutter"],*relation["targets"]),outputs=relation["targets"],method="explicit region difference")
        return RelationExecution(replace(snapshot,bodies=bodies,relations=snapshot.relations+(result,)),result)
    if kind=="terminates_at":
        from .termination import apply_termination
        return apply_termination(snapshot,relation,kernel)
    raise ModelError(relation["id"],"unsupported",f"unknown relation {kind}")
