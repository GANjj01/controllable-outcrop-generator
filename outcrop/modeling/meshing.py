"""Deterministic, budgeted common partition and boundary extraction."""
import numpy as np
from .kernel.types import AABB
from .kernel.grid import freudenthal_axes,anchored_axes
from .kernel.polytopes import ConvexCell
from .kernel.predicates import facet_key,point,plane_sign,plane_rows,possibly_on_planes
from .kernel.entities import EntityTable
from .kernel.boundary import region_planes
from .kernel.partition import SpatialPartition,PartitionFacet,ErrorReport,TopologyEvidence,partition_cell
from .kernel.surface_arrangement import cells_from_polygons,SurfaceCells
from .errors import unresolved
from .kernel.proxies import normalize_field,grid_proxy,constraints,field_evidence,proxy_plane
from .kernel.analytic import LinearField
from .kernel.polytopes import Plane
from .kernel.predicates import polygon_normal,dot
from .kernel.arrangement import prepare_constraints
from .kernel.native import facet_batch_enabled


def _roi(snapshot):
    if not snapshot.bodies:return None
    boxes=[b.region.bounds() for b in snapshot.bodies.values()]
    lo=np.maximum(snapshot.domain.minimum,np.min([b.minimum for b in boxes],axis=0))
    hi=np.minimum(snapshot.domain.maximum,np.max([b.maximum for b in boxes],axis=0))
    return None if np.any(lo>=hi) else AABB(tuple(lo),tuple(hi))


def _box_face(poly,box):
    return any(all(p[j]==value for p in poly) for j in range(3) for value in (point(box.minimum)[j],point(box.maximum)[j]))


def _assemble(parts,roi,domain,planes,request,compiled,error_report=None,plane_sets=None):
    faces={};face_count=0
    for i,c in enumerate(parts):
        for poly in c.cell.polygons:
            faces.setdefault(facet_key(poly),[]).append((i,poly));face_count+=1
        if face_count>request.budget.max_faces:unresolved("partition","global face budget exhausted")
        # Conservative reservation for rational coordinates and incidence.
        if face_count*1024>request.budget.max_bytes:unresolved("partition","memory reservation budget exhausted")
    facets=[];boundaries=[];candidate_rows={};ordered_faces=sorted(faces.items())
    batched=plane_sets is None and facet_batch_enabled(len(ordered_faces))
    if batched:
        from .kernel.facet_filter import fixed_facet_sources
        sources=fixed_facet_sources(ordered_faces,constraints(planes),request)
    for key,entries in ordered_faces:
        if len(entries)>2:unresolved("partition","nonmanifold facet")
        poly=entries[0][1]
        if len(entries)==1 and not _box_face(poly,roi):unresolved("partition","unpaired internal facet; nonconforming partition",source_tet=parts[entries[0][0]].source_tet_id,polygon=np.asarray(poly,float).tolist())
        ids=tuple(i for i,_ in entries);owners=tuple(parts[i].owner for i in ids);lithos=tuple(parts[i].lithology for i in ids)
        if len(ids)==1:owners+=(None,);lithos+=(None,)
        if batched:active=next(sources)
        else:
            group=None if plane_sets is None else tuple(parts[i].source_tet_id for i in ids)
            if group not in candidate_rows:
                candidates=constraints(planes) if plane_sets is None else tuple(p for i in ids for p in constraints(plane_sets[parts[i].source_tet_id]))
                candidate_rows[group]=(candidates,)+plane_rows(candidates)
            candidates,N,O=candidate_rows[group]
            # Vectorised float filter first; exact signs only for planes the facet may lie on.
            maybe=[candidates[j] for j in np.flatnonzero(possibly_on_planes(N,O,poly))]
            active=tuple(sorted({p.source_id for p in maybe if all(plane_sign(p.normal,p.offset,v)==0 for v in poly)}))
        facet=PartitionFacet(poly,owners,lithos,active,_box_face(poly,domain),ids);facets.append(facet)
        if owners[0]!=owners[1] or lithos[0]!=lithos[1]:boundaries.append(facet)
    return SpatialPartition(tuple(parts),EntityTable.from_cells([p.cell for p in parts]),tuple(facets),tuple(boundaries),roi,domain,error_report or ErrorReport(),compiled)


def build_partition(compiled,request):
    from dataclasses import replace
    from .kernel.solid_decomposition import normalize_region
    snapshot=compiled.snapshots[compiled.final_stage_id];cache={}
    snapshot=replace(snapshot,bodies={oid:replace(b,region=normalize_region(b.region,request,cache)) for oid,b in snapshot.bodies.items()});roi=_roi(snapshot)
    if roi is None:return SpatialPartition((),EntityTable((),()),(),(),None,snapshot.domain,ErrorReport(),compiled)
    # finite triangle AABB alone would produce hanging subdivisions.
    from .body_fields import material_regions
    regions=tuple(r for b in snapshot.bodies.values() for r in (b.region,*material_regions(b.material)))
    leaves={leaf.source_id:leaf for region in regions for leaf in region.boundary_leaves(roi)}
    fixed={};nonlinear={}
    for oid,leaf in sorted(leaves.items()):
        if leaf.field is not None:
            field=normalize_field(leaf.field);grid=grid_proxy(field,oid)
            if grid is not None:fixed[oid]=grid
            elif isinstance(field,LinearField):fixed[oid]=Plane(tuple(field.normal),field.offset,oid)
            else:nonlinear[oid]=field
        else:
            if leaf.representation=="mapped_mesh":unresolved(oid,"mapped mesh chord approximation needs correspondence certificate")
            for i,tri in enumerate(leaf.surface.vertices[leaf.surface.faces]):
                tri=tuple(point(p) for p in tri);n=polygon_normal(tri);fixed[f'{oid}:{i}']=Plane(n,dot(n,tri[0]),f'{oid}:{i}')
    prepared=prepare_constraints(constraints(fixed)) if not nonlinear else None
    for refinement in range(request.max_refinement+1):
        axes=anchored_axes(roi,request.grid_origin_enu_m,request.max_cell_size_m/(2**refinement),request.budget)
        complex=freudenthal_axes(axes);source=[];max_error=max_value=0.;need_refine=False;crossings={oid:False for oid in nonlinear}
        for ids,sid in zip(complex.cells,complex.source_ids):
            cell=ConvexCell.tetra(complex.vertices[ids],sid);local=dict(fixed)
            for oid,field in nonlinear.items():
                certificate=field_evidence(field,cell,oid)
                if certificate.possible_zero:
                    if not certificate.certified or certificate.position_error_m is None or certificate.position_error_m>request.max_position_error_m:need_refine=True
                    else:max_error=max(max_error,certificate.position_error_m);max_value=max(max_value,certificate.value_error)
                values=field.values(cell.vertices);crossings[oid]|=bool(values.min()<=0<=values.max())
                local[oid]=proxy_plane(field,cell,oid)
            source.append((cell,local))
        # Ellipsoid/shear zero-witness certificates plus a continuous global
        # proxy are required. No isolated component may have no nodal crossing.
        if nonlinear and not all(crossings.values()):need_refine=True
        if need_refine:
            if refinement==request.max_refinement:unresolved("partition","nonlinear zero/error certification exhausted refinement")
            continue
        parts=[];all_planes={}
        for cell,local in source:
            parts.extend(partition_cell(cell,local,snapshot,request,prepared))
            for oid,p in local.items():all_planes[oid+":"+cell.source_id if oid in nonlinear else oid]=p
            if len(parts)>request.budget.max_cells:unresolved("partition","fragment cell budget exhausted")
        local_planes=dict((c.source_id,p) for c,p in source)
        if not nonlinear and facet_batch_enabled(len(parts)):local_planes=None
        return _assemble(parts,roi,snapshot.domain,all_planes,request,compiled,ErrorReport(max_error,max_value),local_planes)


def validate_partition(partition,request):
    keys=[facet_key(f.polygon) for f in partition.facets]
    duplicate=len(keys)-len(set(keys));unpaired=sum(len(f.cell_ids)==1 and not _box_face(f.polygon,partition.roi) for f in partition.facets)
    if duplicate or unpaired:unresolved("partition","topology validation failed")
    return TopologyEvidence(unpaired,unpaired,duplicate)


def extract_region_boundary(region,request):
    from .query import BodyState,StageSnapshot
    from .body_fields import MaterialField
    from .compiler import CompiledModel
    snapshot=StageSnapshot("boundary",region.bounds(),{region.id:BodyState(region.id,region,MaterialField("boundary"))},{},{})
    compiled=CompiledModel(None,None,None,{snapshot.id:snapshot},snapshot.id)
    partition=build_partition(compiled,request)
    from dataclasses import replace
    cells=boundary_cells(partition)
    from .kernel.certified_arithmetic import float_upper
    from .kernel.predicates import rational
    result=replace(cells,position_error_m=float_upper(rational(cells.position_error_m)+rational(partition.error_report.position_error_m)))
    if not partition.error_report.position_error_m:object.__setattr__(result,'exact_vertices',cells.exact_vertices)
    return result


def boundary_cells(partition):
    return cells_from_polygons([f.polygon for f in partition.boundaries],["/".join(str(x) for x in f.incident) for f in partition.boundaries]) if partition.boundaries else SurfaceCells.empty()
