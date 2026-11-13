import numpy as np
from .kernel.mesh import MeshSurface
from .kernel.intersections import intersect_surfaces
from .kernel.surface_arrangement import cells_from_polygons,SurfaceCells
from .kernel.predicates import point
from .mesh_contracts import TraceMeshV2


def extract_traces(interface_cells,exposure_cells,relations,request,groove_certificate=None):
    if not len(exposure_cells.faces) and groove_certificate is None:return TraceMeshV2.empty()
    exposure=MeshSurface('exposure-traces',exposure_cells.vertices,exposure_cells.faces,getattr(exposure_cells,'exact_vertices',None),exposure_cells.position_error_m)
    segments=[];points=[];polys=[];sources=[];ids=[];rounding=0.
    from .groove_traces import _TraceWork,certified_groove_intersection
    work=_TraceWork(request)
    for oid,cells in sorted(interface_cells.items()):
        if not len(cells.faces):continue
        if groove_certificate is not None and (not len(exposure_cells.faces) or exposure_cells.position_error_m and not hasattr(exposure_cells,'exact_vertices')):
            result=certified_groove_intersection(cells,exposure_cells,groove_certificate,request,work)
            rounding=max(rounding,result.evidence['rounding_error_m'])
        else:result=intersect_surfaces(MeshSurface(oid,cells.vertices,cells.faces,getattr(cells,'exact_vertices',None),cells.position_error_m),exposure,request)
        segments.extend(result.segments);ids.extend([oid]*len(result.segments));points.extend(result.points)
        polys.extend(tuple(point(p) for p in tri) for tri in result.coplanar_cells.vertices[result.coplanar_cells.faces]);sources.extend([oid]*len(result.coplanar_cells.faces))
    result=TraceMeshV2(np.asarray(segments,float).reshape(-1,2,3),np.asarray(points,float).reshape(-1,3),cells_from_polygons(polys,sources) if polys else SurfaceCells.empty(),tuple(ids),tuple(r.id for r in relations))
    object.__setattr__(result,'_position_error_m',rounding)
    return result
