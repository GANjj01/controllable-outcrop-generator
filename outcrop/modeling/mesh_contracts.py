from dataclasses import dataclass,field
import numpy as np
from .kernel.types import AABB
from .kernel.surface_arrangement import SurfaceCells


@dataclass(frozen=True)
class SurfaceMeshV2:
    vertices: np.ndarray
    faces: np.ndarray
    body_ids: tuple
    lithology_ids: tuple
    source_ids: tuple
    incident_regions: tuple = ()
    unit_ids: tuple = ()
    position_error_m: float = 0.
    exact_vertices: tuple | None = field(default=None,repr=False,compare=False)
    @property
    def bounds(self):
        return AABB(tuple(self.vertices.min(0)),tuple(self.vertices.max(0))) if len(self.vertices) else None
    def cells(self):
        result=SurfaceCells(self.vertices,self.faces,self.source_ids,self.position_error_m)
        if self.exact_vertices is not None:object.__setattr__(result,'exact_vertices',self.exact_vertices)
        return result
    @classmethod
    def empty(cls):return cls(np.empty((0,3),float),np.empty((0,3),np.int64),(),(),(),())


@dataclass(frozen=True)
class InterfaceMeshV2:
    id: str
    cells: SurfaceCells
    feature_id: str | None
    hosts: tuple
    lineage: tuple


@dataclass(frozen=True)
class TraceMeshV2:
    segments: np.ndarray
    points: np.ndarray
    coplanar_cells: SurfaceCells
    interface_ids: tuple
    relation_ids: tuple = ()
    @classmethod
    def empty(cls):return cls(np.empty((0,2,3),float),np.empty((0,3),float),SurfaceCells.empty(),())


@dataclass(frozen=True)
class ExtractionResult:
    surface: SurfaceMeshV2
    interfaces: tuple
    traces: TraceMeshV2
    contacts: tuple
    partition: object
    error_report: object
