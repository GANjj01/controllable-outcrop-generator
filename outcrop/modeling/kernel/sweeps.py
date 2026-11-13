import numpy as np
from .types import AABB
from .analytic import FieldRegion,QuadraticField,LinearField
from .csg import CSGRegion
from .mesh import MeshRegion
from ..errors import invalid


def triangulate_section(section):
    from .predicates import orient2d
    p=np.asarray(section,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():invalid("section","finite polygon required")
    area=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
    if area==0:invalid("section","nonzero area required")
    if area<0:p=p[::-1].copy()
    ids=list(range(len(p)));triangles=[]
    while len(ids)>3:
        found=False
        for index,b in enumerate(ids):
            a=ids[index-1];c=ids[(index+1)%len(ids)]
            if orient2d(p[a],p[b],p[c])<=0:continue
            def inside(q):return all(orient2d(p[x],p[y],q)>=0 for x,y in ((a,b),(b,c),(c,a)))
            if any(inside(p[j]) for j in ids if j not in {a,b,c}):continue
            triangles.append((a,b,c));ids.pop(index);found=True;break
        if not found:invalid("section","self-intersecting or degenerate polygon")
    triangles.append(tuple(ids));return p,triangles


def build_sweep(spec,request):
    path=np.asarray(spec["path_m"],float);oid=spec["id"]
    if path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or not np.isfinite(path).all() or np.any(np.linalg.norm(np.diff(path,axis=0),axis=1)==0):invalid(oid,"finite nonrepeating polyline required")
    if "radius_m" in spec:
        radius=float(spec["radius_m"])
        if radius<=0:invalid(oid,"positive radius required")
        parts=[]
        for i,p in enumerate(path):parts.append(FieldRegion(f"{oid}:cap:{i}",QuadraticField(p,np.eye(3),radius**2),AABB(tuple(p-radius),tuple(p+radius))))
        for i,(p,q) in enumerate(zip(path,path[1:])):
            n=(q-p)/np.linalg.norm(q-p);extent=radius*np.sqrt(np.maximum(0,1-n*n));bounds=AABB(tuple(np.minimum(p,q)-extent),tuple(np.maximum(p,q)+extent))
            parts.append(CSGRegion(f"{oid}:segment:{i}","intersection",(
                FieldRegion(f"{oid}:side:{i}",QuadraticField(p,np.eye(3)-np.outer(n,n),radius**2),bounds),
                FieldRegion(f"{oid}:start:{i}",LinearField(-n,-n@p),bounds),FieldRegion(f"{oid}:end:{i}",LinearField(n,n@q),bounds)),bounds))
        return CSGRegion(oid,"union",tuple(parts))
    section,cap=triangulate_section(spec["section"]);count=len(section);tangents=np.diff(path,axis=0);tangents/=np.linalg.norm(tangents,axis=1)[:,None]
    frames=[];previous=None
    for i in range(len(path)):
        tangent=tangents[0] if i==0 else tangents[-1] if i==len(path)-1 else tangents[i-1]+tangents[i]
        norm=np.linalg.norm(tangent)
        if norm==0:invalid(oid,"180-degree sweep turn")
        tangent=tangent/norm;axis=np.eye(3)[np.argmin(np.abs(tangent))] if previous is None else previous
        u=axis-tangent*(axis@tangent);u/=np.linalg.norm(u);v=np.cross(tangent,u);previous=u;frames.append((u,v))
    vertices=np.array([p+x*u+y*v for p,(u,v) in zip(path,frames) for x,y in section]);faces=[]
    faces.extend(tuple(reversed(t)) for t in cap);faces.extend(tuple(x+(len(path)-1)*count for x in t) for t in cap)
    for k in range(len(path)-1):
        for i in range(count):
            j=(i+1)%count;a,b,c,d=k*count+i,k*count+j,(k+1)*count+j,(k+1)*count+i;faces.extend(((a,b,c),(a,c,d)))
    return MeshRegion(oid,vertices,np.array(faces,np.int64),request)
