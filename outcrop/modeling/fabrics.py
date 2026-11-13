from dataclasses import dataclass
import numpy as np
from .kernel.types import Tolerance
from .kernel.mapped import MappedField,MappedRegion
from .errors import unresolved


@dataclass(frozen=True)
class FabricSamples:
    valid: np.ndarray
    normals: np.ndarray
    scalar_coordinate: np.ndarray | None


class FabricField:
    def __init__(self,oid,field,support,tolerance=None):self.id=oid;self.field=field;self.support=support;self.tolerance=tolerance or Tolerance()
    def sample(self,points):
        p=np.asarray(points,float);c=self.support.classify(p,self.tolerance);valid=c.inside|c.boundary;normals=np.zeros_like(p);values=np.full(len(p),np.nan)
        if c.unresolved.any():unresolved(self.id,"fabric support unresolved")
        if valid.any():
            g=self.field.gradient(p[valid]);length=np.linalg.norm(g,axis=1)
            if np.any(length==0):unresolved(self.id,"zero-gradient fabric has no normal")
            normals[valid]=g/length[:,None];values[valid]=self.field.values(p[valid])
        return FabricSamples(valid,normals,values)
    def mapped(self,mapping):return FabricField(self.id,MappedField(self.field,mapping),MappedRegion(self.id+":support",self.support,mapping),self.tolerance)


@dataclass(frozen=True)
class FabricState:
    id: str
    field: FabricField
    hosts: tuple
    version: str = "initial"
