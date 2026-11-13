from dataclasses import dataclass
import numpy as np
from .schema import thaw
from .body_fields import compile_material
from .kernel.types import AABB,Tolerance
from .kernel.csg import CSGRegion
from .errors import ModelError,invalid,unresolved


@dataclass(frozen=True)
class BodyState:
    id: str
    region: object
    material: object
    unit_id: str | None = None
    version: str = "initial"
    lineage: tuple = ()


@dataclass(frozen=True)
class InterfaceState:
    id: str
    geometry: object
    support: object
    hosts: tuple = ()
    feature_id: str | None = None
    version: str = "initial"
    lineage: tuple = ()
    terminations: tuple = ()


@dataclass(frozen=True)
class BoundaryPatch:
    id: str
    interface_id: str
    surface: object
    negative_body: str | None
    positive_body: str | None
    incident_regions: tuple
    lineage: tuple = ()


@dataclass(frozen=True)
class StageSnapshot:
    id: str
    domain: AABB
    bodies: dict
    interfaces: dict
    fabrics: dict
    branches: tuple = ()
    relations: tuple = ()
    tolerance: Tolerance = Tolerance()


@dataclass(frozen=True)
class QueryBatch:
    status: np.ndarray
    on_interface: np.ndarray
    body_ids: np.ndarray
    lithology_ids: np.ndarray
    candidate_offsets: np.ndarray
    candidate_body_ids: tuple
    unit_ids: np.ndarray
    unit_candidate_offsets: np.ndarray
    candidate_unit_ids: tuple


def initial_snapshot(numeric,kernel):
    d=thaw(numeric.data);tol=kernel.request.tolerance;lithologies={s["id"]:s for s in d["lithologies"]}
    bodies={s["id"]:BodyState(s["id"],kernel.regions[s["geometry"]],compile_material(s["material"],lithologies,kernel,tol,s.get('unit_id')),s.get("unit_id")) for s in d["bodies"]}
    kernel.body_templates=dict(bodies)
    created={e["body"] for e in d.get("events",[]) if e["kind"]=="intrusion"}
    bodies={oid:b for oid,b in bodies.items() if oid not in created}
    interfaces={}
    for s in d["interfaces"]:
        if s["geometry"] not in kernel.surfaces:invalid(s["id"],"actual surface geometry required")
        hosts=tuple(s.get("hosts",[]));support=kernel.regions[s["support"]] if "support" in s else CSGRegion(s["id"]+":support","union",tuple(bodies[h].region for h in hosts)) if hosts else box_domain(d)
        interfaces[s["id"]]=InterfaceState(s["id"],kernel.surfaces[s["geometry"]],support,hosts,s.get("feature_id"))
    return StageSnapshot("initial",AABB(tuple(d["domain"]["min_m"]),tuple(d["domain"]["max_m"])),bodies,interfaces,{},tolerance=tol)


def box_domain(d):
    from .kernel.analytic import box_region
    return box_region("domain",d["domain"]["min_m"],d["domain"]["max_m"])


def query_stage(snapshot,points,kernel=None,*,side_hint=None):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():invalid("query","finite Nx3 required")
    n=len(p);active=snapshot.domain.contains(p);status=np.where(active,"void","outside_domain").astype(object);body_ids=np.full(n,None,object);lithology=np.full(n,None,object);units=np.full(n,None,object);unit_candidates=[() for _ in range(n)];on=np.zeros(n,bool);candidates=[[] for _ in range(n)];inside_count=np.zeros(n,int)
    def material_at(body,indices):
        sample=body.material.sample(p[indices]);lithology[indices]=sample.lithology_ids
        units[indices]=sample.unit_ids if sample.unit_ids is not None else body.unit_id
        for j,i in enumerate(indices):
            unit_candidates[i]=sample.unit_candidates[j] if sample.unit_candidates else (units[i],) if units[i] is not None else ()
            if sample.on_interface is not None:on[i]|=sample.on_interface[j]
    for oid,body in sorted(snapshot.bodies.items()):
        c=body.region.classify(p,snapshot.tolerance)
        if np.any(c.unresolved&active):unresolved(oid,"body query unresolved")
        inside=c.inside&active;boundary=c.boundary&active;inside_count+=inside
        body_ids[inside]=oid;status[inside]="occupied";on|=boundary
        if inside.any():material_at(body,np.where(inside)[0])
        for i in np.where(boundary)[0]:candidates[i].append(oid)
    if np.any(inside_count>1):raise ModelError(snapshot.id,"constraint_failed","unresolved body overlap",{"points":p[inside_count>1].tolist()})
    for i in np.where(on&active)[0]:
        try:incident=_local_incident(snapshot,p[i])
        except ModelError as error:
            if error.code!='unsupported':raise
            incident=None
        if incident is not None:
            candidates[i]=list(incident)
            if len(incident)==1 and incident[0] is not None:
                oid=incident[0];body_ids[i]=oid;status[i]='occupied';on[i]=False;candidates[i]=[];material_at(snapshot.bodies[oid],np.array([i]))
            elif side_hint is not None:
                if side_hint not in incident:invalid('side_hint','hint must be a locally incident body')
                body_ids[i]=side_hint;status[i]='occupied';material_at(snapshot.bodies[side_hint],np.array([i]))
        elif body_ids[i] is None:candidates[i].append(None)
    offsets=[0];flat=[]
    for i in range(n):
        if candidates[i]:
            if body_ids[i] is not None:candidates[i].append(body_ids[i])
            flat.extend(sorted(set(candidates[i]),key=lambda x:"" if x is None else x))
        offsets.append(len(flat))
    unit_offsets=[0];unit_flat=[]
    for row in unit_candidates:unit_flat.extend(row);unit_offsets.append(len(unit_flat))
    return QueryBatch(status,on,body_ids,lithology,np.array(offsets,np.int64),tuple(flat),units,np.array(unit_offsets,np.int64),tuple(unit_flat))


def _local_incident(snapshot,p):
    from .kernel.boundary import region_planes
    from .kernel.polytopes import ConvexCell
    from .kernel.arrangement import arrange_cell
    from .kernel.partition import interior
    from .kernel.types import GeometryRequest
    size=snapshot.tolerance.length_m*8;box=AABB(tuple(p-size),tuple(p+size));planes={q.source_id:q for b in snapshot.bodies.values() for q in region_planes(b.region,box)}
    cells=arrange_cell(ConvexCell.box(box.minimum,box.maximum,'query-local'),planes.values(),GeometryRequest(tolerance=snapshot.tolerance)).cells;owners=set()
    for cell in cells:
        vertices=cell.exact_vertices;center=tuple(sum(v[j] for v in vertices)/len(vertices) for j in range(3));labels=[oid for oid,b in snapshot.bodies.items() if interior(b.region,center,planes)]
        if len(labels)>1:raise ModelError(snapshot.id,'constraint_failed','body overlap in local incidence')
        owners.add(labels[0] if labels else None)
    return tuple(sorted(owners,key=lambda x:'' if x is None else x))


@dataclass(frozen=True)
class InterfaceQuery:
    interface_id: str
    negative: QueryBatch
    positive: QueryBatch
    incident_regions: tuple


def interface_query(snapshot,interface_id,points):
    state=snapshot.interfaces[interface_id];p=np.asarray(points,float);normal=state.geometry.normal(p);epsilon=snapshot.tolerance.length_m*4
    minus=query_stage(snapshot,p-normal*epsilon);plus=query_stage(snapshot,p+normal*epsilon)
    incident=tuple(tuple(sorted({a,b},key=lambda x:"" if x is None else x)) for a,b in zip(minus.body_ids,plus.body_ids))
    return InterfaceQuery(interface_id,minus,plus,incident)
