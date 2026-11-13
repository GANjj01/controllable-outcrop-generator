from dataclasses import dataclass
import numpy as np
from .predicates import point,sub,cross,dot,scale,facet_key
from .polytopes import Plane,clip_polygon
from .surface_arrangement import SurfaceCells,cells_from_polygons
from .types import AABB
from ..errors import unresolved


@dataclass(frozen=True)
class IntersectionResult:
    kind: str
    segments: np.ndarray
    points: np.ndarray
    coplanar_cells: SurfaceCells
    evidence: dict


def slice_triangle(tri,n,offset):
    values=[dot(n,p)-offset for p in tri];out=[]
    for i,a in enumerate(tri):
        b=tri[(i+1)%3];u,v=values[i],values[(i+1)%3]
        if u==0:out.append(a)
        if (u<0<v) or (v<0<u):out.append(tuple((u*y-v*x)/(u-v) for x,y in zip(a,b)))
    return tuple(sorted(set(out)))


def triangle_intersection(a,b):
    a=tuple(point(p) for p in a);b=tuple(point(p) for p in b)
    na=cross(sub(a[1],a[0]),sub(a[2],a[0]));nb=cross(sub(b[1],b[0]),sub(b[2],b[0]))
    oa=dot(na,a[0]);ob=dot(nb,b[0]);line=cross(na,nb)
    if not any(line):
        if any(dot(na,p)!=oa for p in b):return "empty",()
        poly=a
        for i,p in enumerate(b):
            edge=sub(b[(i+1)%3],p);n=cross(edge,nb);poly,_=clip_polygon(poly,Plane(n,dot(n,p),"coplanar"))
            if not poly:break
        pts=tuple(sorted(set(poly)))
        if not pts:return "empty",()
        if len(pts)==1:return "point",pts
        if len(pts)==2 or not any(cross(sub(pts[1],pts[0]),sub(pts[-1],pts[0]))):return "line",(pts[0],pts[-1])
        return "area",poly
    sa=slice_triangle(a,nb,ob);sb=slice_triangle(b,na,oa)
    if not sa or not sb:return "empty",()
    axis=max(range(3),key=lambda i:abs(line[i]));lo=max(min(p[axis] for p in sa),min(p[axis] for p in sb));hi=min(max(p[axis] for p in sa),max(p[axis] for p in sb))
    if lo>hi:return "empty",()
    pair=sa if len(sa)>=2 else sb
    if len(pair)==1:return "point",pair
    pair=sorted(pair,key=lambda p:p[axis]);p,q=pair[0],pair[-1]
    def at(t):return tuple(p[j]+(t-p[axis])*(q[j]-p[j])/(q[axis]-p[axis]) for j in range(3))
    return ("point",(at(lo),)) if lo==hi else ("line",tuple(sorted((at(lo),at(hi)))))


def intersect_surfaces(a,b,request):
    from .spatial import AABBTree
    ac=a.cells(request);bc=b.cells(request);bt=bc.vertices[bc.faces]
    if any(c.position_error_m and not hasattr(c,'exact_vertices') for c in (ac,bc)):unresolved('intersection','curved surface intersection needs a transversality/topology certificate')
    tree=AABBTree(bt.min(1),bt.max(1)) if len(bt) else AABBTree(np.empty((0,3)),np.empty((0,3)))
    segments=set();pts=set();areas=[];pairs=0;exact_b=bc.exact_triangles
    for triangle in ac.exact_triangles:
        triangle_array=np.asarray(triangle,float)
        for j in tree.query(AABB(tuple(triangle_array.min(0)),tuple(triangle_array.max(0)))):
            pairs+=1
            if pairs>request.budget.max_predicate_pairs:unresolved("intersection","predicate pair budget exhausted")
            kind,data=triangle_intersection(triangle,exact_b[j])
            if kind=="line":segments.add(tuple(sorted(data)))
            elif kind=="point":pts.update(data)
            elif kind=="area":areas.append(data)
    kinds=sum(bool(x) for x in (segments,pts,areas));kind="mixed" if kinds>1 else "area" if areas else "line" if segments else "point" if pts else "empty"
    result=IntersectionResult(kind,np.asarray(sorted(segments),float).reshape(-1,2,3),np.asarray(sorted(pts),float).reshape(-1,3),cells_from_polygons(areas,["intersection"]*len(areas)),{"method":"exact_binary_rational_PL","pairs":pairs});object.__setattr__(result,'exact_segments',tuple(sorted(segments)));return result
