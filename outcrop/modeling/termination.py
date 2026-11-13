from dataclasses import dataclass,replace
from collections import deque
import numpy as np
from .kernel.types import GeometryRequest
from .kernel.mesh import MeshSurface,triangle_distances
from .kernel.intersections import intersect_surfaces
from .kernel.surface_arrangement import arrange_surface_cells,SurfaceCells
from .query import InterfaceState
from .errors import ModelError,invalid,unresolved


@dataclass(frozen=True)
class TrimResult:
    surface: SurfaceCells
    termination_edges: np.ndarray
    retained_component_ids: tuple
    target_id: str
    relation_id: str
    evidence: dict


@dataclass(frozen=True)
class TerminationBinding:
    relation_id: str
    target_id: str
    edges: np.ndarray


def _on_segment(point,segment,tol):
    a,b=segment;d=b-a;length=np.linalg.norm(d)
    if length==0:return np.linalg.norm(point-a)<=tol
    u=(point-a)@d/(d@d);return -tol/length<=u<=1+tol/length and np.linalg.norm(point-a-u*d)<=tol


def _components(adjacency):
    labels=np.full(len(adjacency),-1,int);number=0
    for start in range(len(adjacency)):
        if labels[start]>=0:continue
        todo=[start];labels[start]=number
        while todo:
            face=todo.pop()
            for other in adjacency[face]:
                if labels[other]<0:labels[other]=number;todo.append(other)
        number+=1
    return labels,number


def trim_surface_at_target(source,target,rule,kernel):
    request=kernel if isinstance(kernel,GeometryRequest) else kernel.request;tol=request.tolerance.length_m;relation_id=rule["id"]
    if "trim_region" in rule:
        if isinstance(kernel,GeometryRequest):invalid(relation_id,"trim_region requires kernel region registry")
        from .kernel.boundary import clip_surface_to_region
        clipped=clip_surface_to_region(source,kernel.regions[rule["trim_region"]],request);source=MeshSurface(source.id,clipped.vertices,clipped.faces)
    intersection=intersect_surfaces(source,target,request)
    if len(intersection.coplanar_cells.faces):raise ModelError(relation_id,"constraint_failed","coplanar termination requires explicit trim interpretation")
    if not len(intersection.segments):
        if rule.get("required_intersection",True):raise ModelError(relation_id,"constraint_failed","termination has no separating intersection")
        return TrimResult(source.cells(request),np.empty((0,2,3)),(),target.id,relation_id,{"method":"optional empty intersection"})
    arranged=arrange_surface_cells(source.cells(request),getattr(intersection,'exact_segments',intersection.segments),request);cells=arranged.cells;edge_faces={}
    for face_id,face in enumerate(cells.faces):
        for a,b in zip(face,np.roll(face,-1)):edge_faces.setdefault(tuple(sorted((int(a),int(b)))),[]).append(face_id)
    all_graph=[set() for _ in cells.faces];cut_graph=[set() for _ in cells.faces];barriers={}
    for edge,faces in edge_faces.items():
        if len(faces)>2:unresolved(relation_id,"nonmanifold source arrangement")
        from .kernel.predicates import sub,cross,dot
        def exact_member(p,segment):
            a,b=segment;v=sub(b,a);w=sub(p,a)
            return not any(cross(v,w)) and 0<=dot(w,v)<=dot(v,v)
        a,b=(cells.exact_vertices[i] for i in edge)
        barrier=any(exact_member(a,s) and exact_member(b,s) for s in intersection.exact_segments)
        if barrier:barriers[edge]=faces
        if len(faces)==2:
            x,y=faces;all_graph[x].add(y);all_graph[y].add(x)
            if not barrier:cut_graph[x].add(y);cut_graph[y].add(x)
    original_labels,original_count=_components(all_graph);labels,count=_components(cut_graph)
    if count<=original_count:raise ModelError(relation_id,"constraint_failed","finite intersection does not separate source; explicit trim region or connecting boundary required")
    if "seed_m" in rule:
        seed=np.asarray(rule["seed_m"],float)
        if seed.shape!=(3,) or any(_on_segment(seed,s,tol) for s in intersection.segments):invalid(relation_id,"retain seed must lie on source away from termination barrier")
        distance=triangle_distances(seed[None,:],cells.vertices[cells.faces])[0];matches=np.where(distance<=tol)[0]
        if not len(matches):invalid(relation_id,"retain seed outside source surface")
        selected=set(labels[matches].tolist())
        if len(selected)!=1:invalid(relation_id,"retain seed selects ambiguous components")
    elif "retain_side" in rule:
        side=rule["retain_side"]
        if side not in {"negative","positive"}:invalid(relation_id,"negative/positive retain side required")
        tc=target.cells(request);normal=target.normal(tc.vertices[:1])[0];anchor=tc.vertices[0]
        if np.max(np.abs((tc.vertices-anchor)@normal))>tol:invalid(relation_id,"curved target needs explicit retain seed")
        selected=set()
        for component in range(count):
            probes=cells.vertices[cells.faces[labels==component]].mean(1);values=(probes-anchor)@normal
            if np.all(values<-tol) if side=="negative" else np.all(values>tol):selected.add(component)
        if not selected:invalid(relation_id,"retain side selects no component")
    else:invalid(relation_id,"explicit retain seed/side required")
    keep=np.array([label in selected for label in labels]);faces=cells.faces[keep];sources=tuple(s for s,k in zip(cells.source_ids,keep) if k);edges=[]
    for edge,incident in barriers.items():
        if len(incident)==2 and keep[incident[0]]!=keep[incident[1]]:edges.append(cells.vertices[list(edge)])
    if not edges:raise ModelError(relation_id,"constraint_failed","selected component has no target termination boundary")
    used=np.unique(faces);remap=np.full(len(cells.vertices),-1,int);remap[used]=np.arange(len(used))
    kept=SurfaceCells(cells.vertices[used],remap[faces],sources)
    if hasattr(cells,'exact_vertices'):object.__setattr__(kept,'exact_vertices',tuple(cells.exact_vertices[i] for i in used))
    return TrimResult(kept,np.asarray(edges).reshape(-1,2,3),tuple(sorted(selected)),target.id,relation_id,{"method":"actual finite-segment constrained arrangement","source_components":original_count,"cut_components":count,"target_support_verified":True})


def apply_termination(snapshot,relation,kernel):
    from .relations import RelationExecution,_result,supported_surface
    source_id=relation["source"];target_id=relation["target"];request=replace(kernel.request,tolerance=snapshot.tolerance)
    source=supported_surface(snapshot,kernel,source_id,request);target=supported_surface(snapshot,kernel,target_id,request,relation.get('target_geometry'))
    trim=trim_surface_at_target(source,target,relation,kernel);interfaces=dict(snapshot.interfaces)
    old=interfaces.get(source_id)
    if old is None:invalid(source_id,"termination source must be an interface object")
    interfaces[source_id]=replace(old,geometry=MeshSurface(source_id,trim.surface.vertices,trim.surface.faces,getattr(trim.surface,'exact_vertices',None)),lineage=old.lineage+(relation["id"],),terminations=old.terminations+(TerminationBinding(relation['id'],target_id,trim.termination_edges),))
    result=_result(snapshot,relation,"pass",intersection=trim,inputs=(source_id,target_id),outputs=(source_id,),method=trim.evidence["method"])
    return RelationExecution(replace(snapshot,interfaces=interfaces,relations=snapshot.relations+(result,)),result)


def verify_final_termination(snapshot,relation,kernel):
    from .relations import supported_surface,_result
    from .kernel.predicates import point,polygon_normal,dot,sub,cross,plane_value
    from fractions import Fraction
    source=snapshot.interfaces[relation['source']];binding=next((t for t in source.terminations if t.relation_id==relation['id']),None)
    if binding is None:return _result(snapshot,relation,'fail',reason='missing final termination binding')
    request=replace(kernel.request,tolerance=snapshot.tolerance);target=supported_surface(snapshot,kernel,binding.target_id,request,relation.get('target_geometry'));cells=target.cells(request);triangles=cells.vertices[cells.faces]
    for edge in binding.edges:
        a,b=map(point,edge);direction=sub(b,a);intervals=[]
        for tri in triangles:
            tri=tuple(point(p) for p in tri);normal=polygon_normal(tri);offset=dot(normal,tri[0])
            if plane_value(normal,offset,a)!=0 or plane_value(normal,offset,b)!=0:continue
            lo,hi=Fraction(0),Fraction(1)
            for i,p in enumerate(tri):
                n=cross(sub(tri[(i+1)%3],p),normal);v=dot(n,sub(a,p));slope=dot(n,direction)
                if slope>0:hi=min(hi,-v/slope)
                elif slope<0:lo=max(lo,-v/slope)
                elif v>0:lo=Fraction(2);break
            if lo<=hi:intervals.append((lo,hi))
        covered=Fraction(0)
        for lo,hi in sorted(intervals):
            if lo>covered:break
            covered=max(covered,hi)
        if covered<1:return _result(snapshot,relation,'fail',witness=np.asarray(edge),reason='final termination edge left actual target patch')
    return _result(snapshot,relation,'pass',method='exact segment coverage by final target triangles')
