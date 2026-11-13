"""Exact PL solid decomposition before nonlinear maps; no chord surrogate."""
from copy import copy
import numpy as np
from .mesh import MeshRegion
from .csg import CSGRegion
from .mapped import MappedRegion
from .analytic import FieldRegion
from .exact_fields import ExactLinearField
from .boundary import region_planes
from .arrangement import arrange_cell
from .polytopes import ConvexCell
from .predicates import polygon_normal,dot
from ..errors import unresolved


def decompose_mesh(region,request):
    source=arrange_cell(ConvexCell.box(region.bounds().minimum,region.bounds().maximum,region.id+':source-roi'),region_planes(region,region.bounds()),request);parts=[]
    for i,cell in enumerate(source.cells):
        value=region.values(cell.centroid[None,:])[0]
        if not np.isfinite(value) or value==0:unresolved(region.id,'source PL solid fragment sign unresolved')
        if value>0:continue
        halfspaces=[]
        for j,poly in enumerate(cell.polygons):
            n=polygon_normal(poly);oid=f'{region.id}:solid:{i}:face:{j}';halfspaces.append(FieldRegion(oid,ExactLinearField(n,dot(n,poly[0])),region.bounds()))
        parts.append(CSGRegion(f'{region.id}:solid:{i}','intersection',tuple(halfspaces),region.bounds()))
    return CSGRegion(region.id,'union',tuple(parts),region.bounds())


def normalize_region(region,request,cache=None,mapped=False):
    cache={} if cache is None else cache
    if isinstance(region,MappedRegion):return MappedRegion(region.id,normalize_region(region.region,request,cache,True),region.mapping)
    if isinstance(region,CSGRegion):
        result=copy(region);result.operands=tuple(normalize_region(r,request,cache,mapped) for r in region.operands);return result
    if mapped and isinstance(region,MeshRegion):
        if region.id not in cache:cache[region.id]=decompose_mesh(region,request)
        return cache[region.id]
    return region
