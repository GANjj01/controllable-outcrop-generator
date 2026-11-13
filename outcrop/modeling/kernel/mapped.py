import numpy as np
from .types import Classification,BoundaryLeaf,IntervalEvidence
from .analytic import FieldRegion,LinearField,OffsetField
from .maps import AffineMap
from ..errors import unresolved


class MappedField:
    def __init__(self,field,mapping):self.field=field;self.mapping=mapping;self.unit=field.unit
    def _source(self,p):
        from .types import Tolerance
        candidates=self.mapping.inverse_candidates(p,Tolerance());out=np.zeros_like(np.asarray(p,float));found=np.zeros(len(p),bool)
        for c in candidates:out[c.valid]=c.points[c.valid];found|=c.valid
        if not found.all():unresolved("mapped_field","no valid inverse field branch")
        return out
    def values(self,p):return self.field.values(self._source(p))
    def gradient(self,p):
        q=self._source(p);g=self.field.gradient(q);J=self.mapping.jacobian(q);return np.linalg.solve(J.transpose(0,2,1),g[...,None])[...,0]
    def interval(self,box):return self.field.interval(self.mapping.preimage_bounds(box))
    def hessian_bound(self,box):
        if isinstance(self.mapping,AffineMap) and hasattr(self.field,"hessian_bound"):
            b=self.field.hessian_bound(self.mapping.preimage_bounds(box))
            return None if b is None else b*np.linalg.norm(self.mapping.inverse_matrix[:3,:3],2)**2
        return None


class MappedRegion:
    def __init__(self,oid,region,mapping):self.id=oid;self.region=region;self.mapping=mapping
    def classify(self,p,tol):
        p=np.asarray(p,float);inside=np.zeros(len(p),bool);boundary=inside.copy();unknown=inside.copy();found=inside.copy()
        for c in self.mapping.inverse_candidates(p,tol):
            if not c.valid.any():continue
            q=c.points[c.valid];result=self.region.classify(q,tol);ids=np.where(c.valid)[0];inside[ids]|=result.inside;boundary[ids]|=result.boundary;unknown[ids]|=result.unresolved;found[ids]=True
        return Classification(inside&~unknown,boundary&~inside&~unknown,~(inside|boundary|unknown),unknown)
    def values(self,p):
        from .types import Tolerance
        values=np.full(len(p),np.inf)
        for c in self.mapping.inverse_candidates(p,Tolerance()):values[c.valid]=np.minimum(values[c.valid],self.region.values(c.points[c.valid]))
        return values
    def bounds(self):return self.mapping.image_bounds(self.region.bounds())
    def interval(self,box):return self.region.interval(self.mapping.preimage_bounds(box))
    def boundary_leaves(self,box):
        leaves=[]
        for leaf in self.region.boundary_leaves(self.mapping.preimage_bounds(box)):
            if leaf.field is not None:
                field=MappedField(leaf.field,self.mapping)
                leaves.append(BoundaryLeaf(self.id+":"+leaf.source_id,field,None,self,"mapped"))
            else:
                from .mesh import MeshSurface
                source=leaf.surface;surface=MeshSurface(self.id+":"+source.id,self.mapping.forward(source.vertices),source.faces)
                leaves.append(BoundaryLeaf(surface.id,None,surface,self,"mapped_mesh"))
        return tuple(leaves)
