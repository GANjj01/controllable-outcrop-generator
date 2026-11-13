from dataclasses import dataclass
from fractions import Fraction
from .polytopes import ConvexCell,split_polytope
from .entities import EntityTable
from .predicates import plane_rows,filtered_signs,plane_sign
from .native import exact_cut_native,filter_statistics
from ..errors import unresolved


@dataclass(frozen=True)
class CellArrangement:
    cells: tuple
    entities: EntityTable
    facets: tuple


@dataclass(frozen=True)
class PreparedPlanes:
    planes: tuple
    normals: object
    offsets: object


def prepare_constraints(constraints):
    """Keep the original canonical order and duplicate-plane source winner."""
    planes={p.key:p for p in sorted(constraints,key=lambda p:p.source_id)}
    ordered=tuple(planes[key] for key in sorted(planes))
    return PreparedPlanes(ordered,*plane_rows(ordered))


def arrange_cell(cell,constraints,request):
    prepared=constraints if isinstance(constraints,PreparedPlanes) else prepare_constraints(constraints)
    planes=prepared.planes
    if len(planes)>request.budget.max_planes_per_cell: unresolved(cell.source_id,"plane budget exhausted")
    vertices=cell.exact_vertices
    signs=filtered_signs(prepared.normals,prepared.offsets,vertices)
    crossing=[]
    for index,(plane,row) in enumerate(zip(planes,signs)):
        exact=[int(s) or plane_sign(plane.normal,plane.offset,v) for s,v in zip(row,vertices)]
        # No descendant of this convex source cell can cross a plane that
        # does not strictly separate its vertices. Resolve filter zeros exactly,
        # including tangent/coincident vertices and cuts below float precision.
        if not exact or not min(exact)<0<max(exact):continue
        crossing.append(index)
    native=exact_cut_native()
    # A single cut has no intermediate conversion to eliminate.
    if len(crossing)>1 and native is not None and all(isinstance(x,Fraction)
            for face in cell.polygons for p in face for x in p):
        cached=getattr(prepared,'_native_planes',None)
        if cached is None or cached[0] is not native:
            cached=(native,native.prepare_planes(tuple((p.normal,p.offset) for p in planes)))
            object.__setattr__(prepared,'_native_planes',cached)
        rows=native.arrange_polytope(cell.polygons,cached[1],crossing,
                                   request.budget.max_fragments_per_cell)
        filter_statistics()['cpp_arrange_calls']+=1
        if rows is None:unresolved(cell.source_id,"fragment budget exhausted")
        cells=tuple(ConvexCell(polygons,cell.source_id,
                    cell.constraints|frozenset(planes[i].source_id for i in lineage))
                    if lineage else cell for polygons,lineage in rows)
    else:
        cells=(cell,)
        for index in crossing:
            plane=planes[index]
            cells=tuple(part for c in cells for part in split_polytope(c,plane,request.tolerance).cells)
            if len(cells)>request.budget.max_fragments_per_cell: unresolved(cell.source_id,"fragment budget exhausted")
    entities=EntityTable.from_cells(cells)
    if sum(len(c.polygons) for c in cells)>request.budget.max_faces: unresolved(cell.source_id,"face budget exhausted")
    return CellArrangement(cells,entities,entities.facets)
