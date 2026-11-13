from dataclasses import dataclass
import numpy as np
from .predicates import facet_key,point


@dataclass(frozen=True)
class EntityTable:
    keys: tuple
    facets: tuple
    @classmethod
    def from_cells(cls,cells):
        vertices=tuple(sorted({p for c in cells for p in c.exact_vertices}))
        facets=tuple(sorted({facet_key(poly) for c in cells for poly in c.polygons}))
        return cls(vertices,facets)


def triangulate_polygon(poly):
    """Canonical center fan retains collinear subdivision vertices."""
    if len(poly)<3: return ()
    if len(poly)==3:return (tuple(poly),)
    center=tuple(sum(p[j] for p in poly)/len(poly) for j in range(3))
    return tuple((center,a,poly[(i+1)%len(poly)]) for i,a in enumerate(poly))
