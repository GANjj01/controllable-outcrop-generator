"""Conservative empty-interior proof; uncertainty cannot become success."""
from dataclasses import dataclass
import numpy as np
from .types import AABB
from ..errors import unresolved,ModelError


@dataclass(frozen=True)
class SetEvidence:
    status: str
    witness_points: np.ndarray
    method: str
    cells: int


def certify_empty(region,request,*,boundary_included=False):
    bounds=region.bounds();lo=np.asarray(bounds.minimum);hi=np.asarray(bounds.maximum)
    if np.any(hi<=lo) and not boundary_included:return SetEvidence("pass",np.empty((0,3)),"zero-volume bounds",0)
    if not boundary_included:
        try:return _exact_empty_interior(region,bounds,request)
        except ModelError as error:
            if error.code!='unsupported':raise
    stack=[(bounds,0)];count=0
    while stack:
        box,depth=stack.pop();count+=1
        if count>request.budget.max_cells:unresolved(region.id,"set predicate cell budget exhausted")
        interval=region.interval(box)
        if interval.certified and (interval.lower>0 if boundary_included else interval.lower>=0):continue
        center=np.mean([box.minimum,box.maximum],axis=0);classification=region.classify(np.asarray([center]),request.tolerance)
        if classification.inside[0] or (boundary_included and classification.boundary[0]):return SetEvidence("fail",np.asarray([center]),"occupied witness",count)
        if depth>=request.max_refinement:unresolved(region.id,"set emptiness not certified within refinement budget")
        minimum=np.asarray(box.minimum);maximum=np.asarray(box.maximum);axis=int(np.argmax(maximum-minimum));mid=(minimum[axis]+maximum[axis])/2
        a=maximum.copy();a[axis]=mid;b=minimum.copy();b[axis]=mid
        stack.extend(((AABB(tuple(minimum),tuple(a)),depth+1),(AABB(tuple(b),tuple(maximum)),depth+1)))
    return SetEvidence("pass",np.empty((0,3)),"conservative interval empty proof",count)


def _exact_empty_interior(region,bounds,request):
    from .analytic import LinearField
    from .proxies import normalize_field,grid_proxy,constraints
    from .polytopes import Plane,ConvexCell
    from .arrangement import arrange_cell
    from .partition import interior
    from .solid_decomposition import normalize_region
    region=normalize_region(region,request,{})
    planes={}
    for leaf in region.boundary_leaves(bounds):
        if leaf.field is None:raise ModelError(region.id,'unsupported','exact empty proof requires normalized field leaves')
        field=normalize_field(leaf.field);proxy=grid_proxy(field,leaf.source_id)
        if proxy is not None:planes[leaf.source_id]=proxy
        elif isinstance(field,LinearField):planes[leaf.source_id]=Plane(tuple(field.normal),field.offset,leaf.source_id)
        else:raise ModelError(region.id,'unsupported','nonlinear empty proof uses conservative intervals')
    arrangement=arrange_cell(ConvexCell.box(bounds.minimum,bounds.maximum,region.id+':empty-proof'),constraints(planes),request)
    if len(arrangement.cells)>request.budget.max_cells:unresolved(region.id,'set predicate cell budget exhausted')
    for cell in arrangement.cells:
        vertices=cell.exact_vertices;p=tuple(sum(v[j] for v in vertices)/len(vertices) for j in range(3))
        if interior(region,p,planes):return SetEvidence('fail',np.asarray([p],float),'exact positive-volume arrangement witness',len(arrangement.cells))
    return SetEvidence('pass',np.empty((0,3)),'exact Boolean arrangement empty-interior proof',len(arrangement.cells))
