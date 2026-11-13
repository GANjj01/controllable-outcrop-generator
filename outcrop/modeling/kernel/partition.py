"""One shared spatial arrangement; rational facets are the topology authority."""
from dataclasses import dataclass
import numpy as np
from .analytic import FieldRegion,EmptyRegion
from .csg import CSGRegion
from .mapped import MappedRegion
from .polytopes import ConvexCell
from .predicates import plane_sign
from .arrangement import arrange_cell
from ..errors import ModelError,unresolved


@dataclass(frozen=True)
class PartitionCell:
    cell: ConvexCell
    owner: str | None
    lithology: str | None
    source_tet_id: str
    unit_id: str | None = None


@dataclass(frozen=True)
class PartitionFacet:
    polygon: tuple
    incident: tuple
    incident_lithologies: tuple
    source_ids: tuple
    domain_boundary: bool
    cell_ids: tuple


@dataclass(frozen=True)
class ErrorReport:
    position_error_m: float = 0.
    value_error: float = 0.
    input_pl_error_m: float | None = None
    unresolved_items: tuple = ()
    fault_crossing_interpolation_count: int = 0


@dataclass(frozen=True)
class SpatialPartition:
    cells: tuple
    entities: object
    facets: tuple
    boundaries: tuple
    roi: object
    domain: object
    error_report: ErrorReport
    compiled: object = None


@dataclass(frozen=True)
class TopologyEvidence:
    shared_face_mismatch_count: int
    hanging_node_count: int
    duplicate_facet_count: int
    method: str = "exact rational facet incidence"


def interior(region,p,planes,prefix="",closed=False):
    """Evaluate the full Boolean expression using its arranged leaf signs."""
    if isinstance(region,EmptyRegion):return False
    if isinstance(region,CSGRegion):
        if region.kind=='difference':return interior(region.operands[0],p,planes,prefix,closed) and not interior(region.operands[1],p,planes,prefix,not closed)
        signs=[interior(r,p,planes,prefix,closed) for r in region.operands]
        return any(signs) if region.kind=="union" else all(signs)
    if isinstance(region,MappedRegion):return interior(region.region,p,planes,prefix+region.id+":",closed)
    if isinstance(region,FieldRegion):
        key=prefix+region.id
        if key not in planes:unresolved(key,"missing field plane in common arrangement")
        plane=planes[key]
        if hasattr(plane,"negative_at"):return plane.negative_at(p,closed=closed)
        s=plane_sign(plane.normal,plane.offset,p);return s<=0 if closed else s<0
    # Closed input PL surfaces are classified after all of their supporting
    # planes have split the cell. No extended plane is itself a boundary.
    v=region.values(np.asarray([p],float))[0]
    if not np.isfinite(v) or v==0 and not closed:unresolved(region.id,"PL fragment sign unresolved")
    return v<=0 if closed else v<0


def partition_cell(cell,planes,snapshot,request,prepared=None):
    from .proxies import constraints
    arrangement=arrange_cell(cell,prepared if prepared is not None else constraints(planes),request);parts=[]
    for fragment in arrangement.cells:
        vertices=fragment.exact_vertices
        p=tuple(sum(v[j] for v in vertices)/len(vertices) for j in range(3))
        owners=[oid for oid,b in sorted(snapshot.bodies.items()) if interior(b.region,p,planes)]
        if len(owners)>1:raise ModelError(snapshot.id,"constraint_failed","body overlap in common partition",{"bodies":owners,"point":list(map(float,p))})
        owner=owners[0] if owners else None
        from ..body_fields import fragment_material
        material=fragment_material(snapshot.bodies[owner].material,p,planes) if owner else None;lid=material.lithology_ids[0] if material is not None else None;unit=material.unit_ids[0] if material is not None and material.unit_ids is not None else None
        parts.append(PartitionCell(fragment,owner,lid,cell.source_id,unit))
    return tuple(parts)
