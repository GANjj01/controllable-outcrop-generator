from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from ..errors import invalid
from ..schema import ResourceSet

Points = np.ndarray
Indices = np.ndarray


@dataclass(frozen=True)
class AABB:
    minimum: tuple
    maximum: tuple
    def __post_init__(self):
        lo, hi = np.asarray(self.minimum,float), np.asarray(self.maximum,float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo,hi]).all() or np.any(lo>hi):
            invalid("bounds", "finite three-dimensional ordered bounds required")
    def contains(self, points):
        p=np.asarray(points,float); return np.all((p>=self.minimum)&(p<=self.maximum),axis=-1)
    @property
    def diameter(self): return float(np.linalg.norm(np.subtract(self.maximum,self.minimum)))
    @property
    def corners(self):
        from itertools import product
        return np.array(list(product(*zip(self.minimum,self.maximum))),float)


@dataclass(frozen=True)
class Tolerance:
    length_m: float = 1e-8
    angle_deg: float = 1e-7
    scalar_by_unit: dict = field(default_factory=dict)
    inverse_residual_m: float = 1e-9
    max_condition: float = 1e8
    def __post_init__(self):
        for k in ("length_m","angle_deg","inverse_residual_m","max_condition"):
            if not np.isfinite(getattr(self,k)) or getattr(self,k)<=0: invalid(k,"positive finite tolerance required")
        for k,v in self.scalar_by_unit.items():
            if not np.isfinite(v) or v<=0: invalid(f"scalar_by_unit.{k}","positive finite tolerance required")


@dataclass(frozen=True)
class Budget:
    max_cells: int = 100000
    max_planes_per_cell: int = 128
    max_fragments_per_cell: int = 4096
    max_faces: int = 1000000
    max_bytes: int = 536870912
    max_predicate_pairs: int = 1000000
    max_iterations: int = 100
    def __post_init__(self):
        for k,v in vars(self).items():
            if type(v) is not int or v<=0: invalid(k,"positive integer budget required")


@dataclass(frozen=True)
class GeometryRequest:
    tolerance: Tolerance = field(default_factory=Tolerance)
    budget: Budget = field(default_factory=Budget)
    max_cell_size_m: float = 1.
    min_feature_m: float = .01
    max_position_error_m: float = .01
    max_refinement: int = 8
    grid_origin_enu_m: tuple = (0.,0.,0.)
    def __post_init__(self):
        for k in ("max_cell_size_m","min_feature_m","max_position_error_m"):
            if isinstance(getattr(self,k),(bool,np.bool_)) or not np.isfinite(getattr(self,k)) or getattr(self,k)<=0: invalid(k,"positive finite resolution required")
        if type(self.max_refinement) is not int or self.max_refinement<0: invalid("max_refinement","nonnegative integer required")
        if len(self.grid_origin_enu_m)!=3 or any(isinstance(v,(bool,np.bool_)) or not np.isfinite(v) for v in self.grid_origin_enu_m):invalid('grid_origin_enu_m','three finite numeric coordinates required')
        object.__setattr__(self,'grid_origin_enu_m',tuple(float(v) for v in self.grid_origin_enu_m))


@dataclass(frozen=True)
class Classification:
    inside: np.ndarray
    boundary: np.ndarray
    outside: np.ndarray
    unresolved: np.ndarray


@dataclass(frozen=True)
class IntervalEvidence:
    lower: float
    upper: float
    certified: bool = True
    unit: str = "1"


@dataclass(frozen=True)
class BoundaryLeaf:
    source_id: str
    field: object | None
    surface: object | None
    support: object
    representation: str


@dataclass
class KernelContext:
    regions: dict = field(default_factory=dict)
    surfaces: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    maps: dict = field(default_factory=dict)
    resources: ResourceSet = field(default_factory=ResourceSet.empty)
    body_templates: dict = field(default_factory=dict)
    request: GeometryRequest = field(default_factory=GeometryRequest)


def model_request(numeric):
    from ..schema import thaw
    values=thaw(numeric.data).get('meshing',{}).copy()
    if 'tolerance' in values:values['tolerance']=Tolerance(**values['tolerance'])
    if 'budget' in values:values['budget']=Budget(**values['budget'])
    try:return GeometryRequest(**values)
    except (TypeError,ValueError):invalid('meshing','strict geometry request required')
