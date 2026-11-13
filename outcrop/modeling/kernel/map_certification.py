import numpy as np
from .maps import MapCertificate
from .spatial import AABBTree
from .types import AABB
from .predicates import orient3d,point,sub,cross,dot,rational,affine_coefficients
from .certified_arithmetic import inverse3,sqrt_interval,float_lower,float_upper
from .intersections import triangle_intersection
from ..errors import invalid,unresolved,ModelError


def _inside_tetra(p,t):
    total=orient3d(*t)
    if total==0:return False
    signs=[orient3d(p,t[1],t[2],t[3]),orient3d(t[0],p,t[2],t[3]),orient3d(t[0],t[1],p,t[3]),orient3d(t[0],t[1],t[2],p)]
    return all(s==total for s in signs)


def certify_grid_map(mapping,request):
    vertices=tuple(tuple(x+y for x,y in zip(point(p),point(d))) for p,d in zip(mapping.complex.vertices,mapping.data.reshape(-1,3)))
    targets=tuple(tuple(vertices[i] for i in ids) for ids in mapping.complex.cells);jacobians=[];dets=[];rho2=0;condition=1
    for ids,t in zip(mapping.complex.cells,targets):
        source=tuple(point(mapping.complex.vertices[i]) for i in ids);J=tuple(affine_coefficients(source,tuple(p[j] for p in t))[0] for j in range(3));det=dot(J[0],cross(J[1],J[2]))
        if det<=0:invalid(mapping.id,'singular or flipped exact map cell')
        condition=max(condition,_norm_upper(J)*_norm_upper(inverse3(J)))
        if condition>rational(request.tolerance.max_condition):unresolved(mapping.id,'map condition budget exceeded')
        dets.append(det);jacobians.append(J);rho2=max(rho2,sum((J[i][j]-(i==j))**2 for i in range(3) for j in range(3)))
    minimum=float_lower(min(dets));rho=sqrt_interval(rho2).upper
    d=mapping.data;boundary=np.zeros(d.shape[:3],bool);boundary[[0,-1],:,:]=True;boundary[:,[0,-1],:]=True;boundary[:,:,[0,-1]]=True
    zero_boundary=np.array_equal(d[boundary],np.zeros_like(d[boundary]))
    if mapping.extension=="zero" and not zero_boundary:invalid(mapping.id,"zero extension requires zero boundary displacement")
    if mapping.extension=="zero" and rho2<1:
        return MapCertificate("pass",minimum,True,{"method":"exact continuous zero-extension contraction","rho_upper":float_upper(rho),'condition_upper':float_upper(condition)})
    # Complete pair checks in the finite image. Local determinants alone never
    # establish injectivity; source shared simplices are the only allowed contact.
    minima=np.array([[float_lower(min(p[j] for p in t)) for j in range(3)] for t in targets]);maxima=np.array([[float_upper(max(p[j] for p in t)) for j in range(3)] for t in targets]);tree=AABBTree(minima,maxima);pairs=0
    for i,t in enumerate(targets):
        for j in tree.query(AABB(tuple(minima[i]),tuple(maxima[i]))):
            if j<=i:continue
            pairs+=1
            if pairs>request.budget.max_predicate_pairs:unresolved(mapping.id,"global map predicate budget exhausted")
            u=targets[j];shared_ids=set(mapping.complex.cells[i])&set(mapping.complex.cells[j]);shared=[vertices[x] for x in shared_ids]
            if any(_inside_tetra(p,u) for p in t) or any(_inside_tetra(p,t) for p in u):raise ModelError(mapping.id,"constraint_failed","global map overlap")
            for ai in range(4):
                a=tuple(p for k,p in enumerate(t) if k!=ai)
                for bi in range(4):
                    b=tuple(p for k,p in enumerate(u) if k!=bi);kind,data=triangle_intersection(a,b)
                    if kind=="empty":continue
                    for p in data:
                        if not _on_shared(p,shared,request.tolerance.length_m):raise ModelError(mapping.id,"constraint_failed","non-shared image intersection/overlap")
    return MapCertificate("pass",minimum,True,{"method":"complete target tetra nonoverlap","pairs":pairs,"extension":mapping.extension,'condition_upper':float_upper(condition)})


def _on_shared(p,shared,tol):
    p=point(p);shared=tuple(point(q) for q in shared)
    if not shared:return False
    if len(shared)==1:return p==shared[0]
    if len(shared)==2:
        a,b=shared;d=sub(b,a);v=sub(p,a);s=dot(v,d)/dot(d,d);return not any(cross(v,d)) and 0<=s<=1
    a,b,c=shared;n=cross(sub(b,a),sub(c,a))
    return dot(n,sub(p,a))==0 and all(dot(n,cross(sub(v,u),sub(p,u)))>=0 for u,v in ((a,b),(b,c),(c,a)))


def _norm_upper(M):
    return sqrt_interval(max(sum(abs(M[i][j]) for i in range(3)) for j in range(3))*max(sum(abs(x) for x in row) for row in M)).upper
