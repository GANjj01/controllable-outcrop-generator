import numpy as np
from .types import AABB,Classification,IntervalEvidence,BoundaryLeaf
from .surface_arrangement import SurfaceCells
from .spatial import AABBTree
from ..errors import invalid,unresolved,ModelError


def triangle_distances(points,triangles):
    p=np.asarray(points,float);t=np.asarray(triangles,float);result=np.empty((len(p),len(t)))
    for i,tri in enumerate(t):
        a,b,c=tri;ab=b-a;ac=c-a;n=np.cross(ab,ac);normal=n/np.linalg.norm(n)
        height=(p-a)@normal;projected=p-height[:,None]*normal
        dots=np.array([ab@ab,ab@ac,ac@ac]);q=projected-a;den=dots[0]*dots[2]-dots[1]**2
        u=(q@ab*dots[2]-q@ac*dots[1])/den;v=(q@ac*dots[0]-q@ab*dots[1])/den
        inside=(u>=0)&(v>=0)&(u+v<=1);distance=np.where(inside,np.abs(height),np.inf)
        for x,y in ((a,b),(b,c),(c,a)):
            edge=y-x;alpha=np.clip((p-x)@edge/(edge@edge),0,1);d=np.linalg.norm(p-x-alpha[:,None]*edge,axis=1);distance=np.minimum(distance,d)
        result[:,i]=distance
    return result


def winding(points,triangles):
    result=np.zeros(len(points),float)
    for tri in triangles:
        a,b,c=(v-points for v in tri);la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
        numerator=np.einsum("ni,ni->n",a,np.cross(b,c));den=la*lb*lc+np.einsum("ni,ni->n",a,b)*lc+np.einsum("ni,ni->n",b,c)*la+np.einsum("ni,ni->n",c,a)*lb
        result+=2*np.arctan2(numerator,den)
    return result/(4*np.pi)


class MeshSurface:
    def __init__(self,oid,vertices,faces,exact_vertices=None,position_error_m=0.):
        self.id=oid;self.vertices=np.asarray(vertices,float);raw=np.asarray(faces)
        self.exact_vertices=exact_vertices
        self.position_error_m=position_error_m
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or not np.isfinite(self.vertices).all():invalid(oid,"finite Nx3 vertices required")
        if raw.ndim!=2 or raw.shape[1]!=3 or raw.dtype.kind not in "iu" or np.any(raw<0) or np.any(raw>=len(self.vertices)):invalid(oid,"valid integer triangle indices required")
        self.faces=raw.astype(np.int64);self.triangles=self.vertices[self.faces]
        normals=np.cross(self.triangles[:,1]-self.triangles[:,0],self.triangles[:,2]-self.triangles[:,0]);length=np.linalg.norm(normals,axis=1)
        # A short edge can disappear in rounded coordinates or when two long
        # edges are subtracted. Prefer supplied exact geometry for the fallback;
        # scale before converting to float so area underflow is harmless.
        from .predicates import point,cross,sub
        for i in np.flatnonzero((length==0)|~np.isfinite(length)):
            source=self.triangles[i] if self.exact_vertices is None else (self.exact_vertices[int(j)] for j in self.faces[i])
            tri=tuple(point(p) for p in source);n=cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]));scale=max(map(abs,n))
            if not scale:invalid(oid,"degenerate triangles forbidden")
            normals[i]=np.asarray(tuple(x/scale for x in n),float);length[i]=np.linalg.norm(normals[i])
        self.normals=normals/length[:,None];self.tree=AABBTree(self.triangles.min(1),self.triangles.max(1))
    def bounds(self):return AABB(tuple(self.vertices.min(0)),tuple(self.vertices.max(0)))
    def cells(self,request):
        result=SurfaceCells(self.vertices,self.faces,tuple(f"{self.id}:triangle:{i}" for i in range(len(self.faces))),self.position_error_m)
        if self.exact_vertices is not None:object.__setattr__(result,'exact_vertices',self.exact_vertices)
        return result
    def normal(self,points):return self.normals[np.argmin(triangle_distances(np.asarray(points,float),self.triangles),axis=1)]
    def restrict(self,cells,request):
        # Actual triangle/cell clipping, never the extended supporting plane.
        from .polytopes import clip_polygon_by_cell
        from .predicates import point
        from .surface_arrangement import cells_from_polygons
        polygons=[];sources=[]
        for cell in cells:
            for i in self.tree.query(AABB(tuple(cell.vertices.min(0)),tuple(cell.vertices.max(0)))):
                poly=tuple(point(p) for p in self.triangles[i])
                poly=clip_polygon_by_cell(poly,cell,'cell-face')
                if len(poly)>=3:polygons.append(poly);sources.append(f"{self.id}:triangle:{i}")
        return cells_from_polygons(polygons,sources)


class MeshRegion(MeshSurface):
    def __init__(self,oid,vertices,faces,request):
        super().__init__(oid,vertices,faces);edges={}
        for i,face in enumerate(self.faces):
            for a,b in zip(face,np.roll(face,-1)):edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append((i,int(a),int(b)))
        if any(len(v)!=2 or v[0][1:]!=v[1][1:][::-1] for v in edges.values()):invalid(oid,"closed oriented manifold required")
        from .intersections import triangle_intersection
        pairs=0
        for i,tri in enumerate(self.triangles):
            for j in self.tree.query(AABB(tuple(tri.min(0)),tuple(tri.max(0)))):
                if j<=i:continue
                pairs+=1
                if pairs>request.budget.max_predicate_pairs:unresolved(oid,"mesh self-intersection budget exhausted")
                kind,data=triangle_intersection(tri,self.triangles[j]);shared=set(self.faces[i])&set(self.faces[j])
                if kind=="empty":continue
                expected={tuple(self.vertices[x]) for x in shared}
                if kind=="area" or any(tuple(map(float,p)) not in expected for p in data):invalid(oid,"self-intersection or intersecting shells")
        # Probe oriented facets on both sides, with a local clearance-limited
        # displacement. A valid material shell changes winding 1 -> 0.
        for i,tri in enumerate(self.triangles):
            center=tri.mean(0);dist=triangle_distances(center[None,:],self.triangles)[0]
            # The queried facet is the support, not an obstacle: a rounding
            # residual to itself must not become the available clearance.
            dist[i]=np.inf
            clearance=min(dist.min(),np.linalg.norm(tri[1]-tri[0]),np.linalg.norm(tri[2]-tri[0]))
            eps=clearance*1e-5
            if eps<=np.finfo(float).eps*np.max(np.abs(center)):unresolved(oid,"shell orientation clearance unresolved")
            q=np.array([center-self.normals[i]*eps,center+self.normals[i]*eps]);w=winding(q,self.triangles)
            if not np.allclose(w,[1,0],atol=1e-5):invalid(oid,"invalid nested-shell orientation or winding")
    def classify(self,points,tol):
        p=np.asarray(points,float);distance=triangle_distances(p,self.triangles).min(1);on=distance<=tol.length_m
        w=winding(p,self.triangles);inside=np.abs(w-1)<1e-7;outside=np.abs(w)<1e-7;unknown=~(inside|outside|on)
        return Classification(inside&~on,on,outside&~on,unknown)
    def values(self,points):
        p=np.asarray(points,float);d=triangle_distances(p,self.triangles).min(1);w=winding(p,self.triangles)
        return np.where(np.abs(w-1)<1e-7,-d,d)
    def interval(self,box):
        if len(self.tree.query(box)):return IntervalEvidence(-np.inf,np.inf,True,"sign")
        w=winding(np.asarray([np.mean([box.minimum,box.maximum],axis=0)]),self.triangles)[0]
        if abs(w-1)<1e-7:return IntervalEvidence(-1,-1,True,"sign")
        if abs(w)<1e-7:return IntervalEvidence(1,1,True,"sign")
        return IntervalEvidence(-np.inf,np.inf,False,"sign")
    def boundary_leaves(self,box):return (BoundaryLeaf(self.id,None,self,self,"mesh"),)
