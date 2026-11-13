from dataclasses import dataclass
import numpy as np
from .polytopes import Plane,split_polygon
from .predicates import point,cross,sub,dot,facet_key
from .entities import triangulate_polygon,EntityTable
from ..errors import invalid,unresolved


@dataclass(frozen=True)
class SurfaceCells:
    vertices: np.ndarray
    faces: np.ndarray
    source_ids: tuple
    position_error_m: float = 0.
    @property
    def exact_triangles(self):
        vertices=getattr(self,'exact_vertices',None)
        return tuple(tuple(vertices[i] if vertices is not None else point(self.vertices[i]) for i in face) for face in self.faces)
    @classmethod
    def empty(cls): return cls(np.empty((0,3),float),np.empty((0,3),np.int64),())


@dataclass(frozen=True)
class SurfaceArrangement:
    cells: SurfaceCells
    entities: EntityTable
    segments: np.ndarray
    barriers: tuple


def cells_from_polygons(polygons,source_ids):
    tris=[];sources=[]
    for poly,source in zip(polygons,source_ids):
        for tri in triangulate_polygon(poly):
            if any(cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]))): tris.append(tri);sources.append(source)
    vertices=tuple(sorted({p for tri in tris for p in tri}));lookup={p:i for i,p in enumerate(vertices)}
    from .mapped_surfaces import _rounded
    from .certified_arithmetic import float_upper
    values,error=_rounded(vertices)
    result=SurfaceCells(values,np.array([[lookup[p] for p in tri] for tri in tris],np.int64).reshape(-1,3),tuple(sources),float_upper(error));object.__setattr__(result,'exact_vertices',vertices);return result


def arrange_surface_cells(cells,segments,request):
    seg=np.asarray(segments,float).reshape(-1,2,3);polygons=[];sources=[];barriers=[]
    exact_segments=tuple((point(a),point(b)) for a,b in segments)
    # Include endpoint cross-lines for real finite endpoints, but only the
    # original segment is a barrier. Infinite extension merely overpartitions.
    for triangle,source in zip(cells.exact_triangles,cells.source_ids):
        tri=tuple(point(p) for p in triangle);normal=cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]));parts=[tri]
        if not any(normal): invalid("surface","degenerate triangle")
        for a,b in exact_segments:
            a,b=point(a),point(b);direction=sub(b,a)
            if not any(direction): continue
            if dot(normal,sub(a,tri[0])) or dot(normal,sub(b,tri[0])): continue
            cutnormal=cross(normal,direction)
            for n,p in ((cutnormal,a),(direction,a),(direction,b)):
                plane=Plane(n,dot(n,p),"surface-cut");next_parts=[]
                for poly in parts:
                    left,right=split_polygon(poly,plane)
                    if facet_key(left)==facet_key(right):next_parts.append(left)
                    else:
                        if len(left)>=3 and any(cross(sub(left[1],left[0]),sub(left[2],left[0]))):next_parts.append(left)
                        if len(right)>=3 and any(cross(sub(right[1],right[0]),sub(right[2],right[0]))):next_parts.append(right)
                parts=next_parts
        polygons.extend(parts);sources.extend([source]*len(parts))
        if len(polygons)>request.budget.max_faces: unresolved("surface","surface arrangement budget exhausted")
    result=cells_from_polygons(polygons,sources)
    # Barriers are source segments; downstream graph code clips them to actual
    # source cells and never treats auxiliary triangulation lines as barriers.
    barriers=tuple(seg)
    keys=tuple(point(p) for p in result.vertices)
    return SurfaceArrangement(result,EntityTable(keys,()),seg,barriers)
