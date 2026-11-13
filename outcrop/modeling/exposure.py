"""Exposure is an independent surface/retained-region operation."""
from dataclasses import replace
import numpy as np
from .meshing import build_partition
from .kernel.polytopes import clip_polygon_by_cell
from .kernel.predicates import point,polygon_normal,facet_key
from .kernel.surface_arrangement import cells_from_polygons,SurfaceCells
from .kernel.spatial import AABBTree
from .kernel.types import AABB
from .kernel.csg import CSGRegion
from .mesh_contracts import SurfaceMeshV2,InterfaceMeshV2,ExtractionResult
from .traces import extract_traces
from .errors import invalid,unresolved


def _mesh(polygons,records):
    if not polygons:return SurfaceMeshV2.empty()
    # Unique source token joins deterministic triangulation back to face data.
    tokens=[str(i) for i in range(len(polygons))];cells=cells_from_polygons(polygons,tokens)
    labels=[records[int(i)] for i in cells.source_ids]
    return SurfaceMeshV2(cells.vertices,cells.faces,tuple(r[0] for r in labels),tuple(r[1] for r in labels),tuple(r[2] for r in labels),tuple(r[3] for r in labels),tuple(r[4] if len(r)>4 else None for r in labels),cells.position_error_m,exact_vertices=getattr(cells,'exact_vertices',None))


def section_cells(source,partition,request,hosts=()):
    if not len(source.faces) or not partition.cells:return SurfaceMeshV2.empty()
    minima=np.array([c.cell.vertices.min(0) for c in partition.cells]);maxima=np.array([c.cell.vertices.max(0) for c in partition.cells]);tree=AABBTree(minima,maxima);pieces={};pairs=0
    originals=source.exact_triangles
    for source_index,(tri,sid) in enumerate(zip(originals,source.source_ids)):
        array=np.asarray(tri,float)
        for index in tree.query(AABB(tuple(array.min(0)),tuple(array.max(0)))):
            c=partition.cells[index]
            if c.owner is None or hosts and c.owner not in hosts:continue
            pairs+=1
            if pairs>request.budget.max_predicate_pairs:unresolved('exposure','surface/cell pair budget exhausted')
            poly=tuple(point(p) for p in tri)
            poly=clip_polygon_by_cell(poly,c.cell)
            if len(poly)<3 or not any(polygon_normal(poly)):continue
            key=(source_index,facet_key(poly))
            if key not in pieces:pieces[key]=[poly,set(),set(),set()]
            pieces[key][1].add(c.owner);pieces[key][2].add(c.lithology)
            pieces[key][3].add(c.unit_id)
            if len(pieces)>request.budget.max_faces:unresolved('exposure','surface face budget exhausted')
    polys=[];records=[];groups={}
    for (index,key),(poly,bodies,lithos,units) in sorted(pieces.items()):
        record=(next(iter(bodies)) if len(bodies)==1 else None,next(iter(lithos)) if len(lithos)==1 else None,source.source_ids[index],tuple(sorted(bodies)),next(iter(units)) if len(units)==1 else None)
        groups.setdefault(index,[]).append((poly,record))
    for index,parts in groups.items():
        original=originals[index];axis=max(range(3),key=lambda j:abs(polygon_normal(original)[j]))
        def area(poly):
            from .kernel.predicates import cross
            return abs(sum(cross(p,poly[(i+1)%len(poly)])[axis] for i,p in enumerate(poly)))
        if len({record for poly,record in parts})==1 and sum(area(poly) for poly,record in parts)==area(original):parts=[(original,parts[0][1])]
        else:parts=_merge_convex_parts(parts,axis,request)
        for poly,record in parts:polys.append(poly);records.append(record)
    from .kernel.certified_arithmetic import float_upper
    from .kernel.predicates import rational
    result=_mesh(polys,records)
    result=replace(result,position_error_m=float_upper(rational(result.position_error_m)+rational(source.position_error_m)))
    if source.position_error_m and not hasattr(source,'exact_vertices'):result=replace(result,exact_vertices=None)
    return result


def _merge_convex_parts(parts,drop,request):
    from .kernel.predicates import cross,sub
    axes=[i for i in range(3) if i!=drop];pairs=0;result=[]
    def turn(a,b,c):
        return (b[axes[0]]-a[axes[0]])*(c[axes[1]]-a[axes[1]])-(b[axes[1]]-a[axes[1]])*(c[axes[0]]-a[axes[0]])
    def hull(points):
        points=sorted(set(points),key=lambda p:(p[axes[0]],p[axes[1]]));chains=[]
        for order in (points,list(reversed(points))):
            chain=[]
            for p in order:
                while len(chain)>1 and turn(chain[-2],chain[-1],p)<=0:chain.pop()
                chain.append(p)
            chains.append(chain[:-1])
        return tuple(chains[0]+chains[1])
    def signed_area(poly):return sum(cross(p,poly[(i+1)%len(poly)])[drop] for i,p in enumerate(poly))
    for poly,record in parts:
        changed=True
        while changed:
            changed=False
            for i,(other,label) in enumerate(result):
                if label!=record:continue
                pairs+=1
                if pairs>request.budget.max_predicate_pairs:unresolved('exposure','coplanar merge predicate budget exhausted')
                joined=hull((*poly,*other))
                if abs(signed_area(joined))!=abs(signed_area(poly))+abs(signed_area(other)):continue
                if signed_area(joined)*signed_area(poly)<0:joined=tuple(reversed(joined))
                poly=joined;result.pop(i);changed=True;break
        result.append((poly,record))
    return result


def _interface_cells(state,partition,request):
    from .kernel.boundary import clip_surface_to_region
    from .errors import ModelError
    source=state.geometry.cells(request)
    class SourceSurface:
        id=state.id
        def cells(self,request):return source
        def bounds(self):return AABB(tuple(source.vertices.min(0)),tuple(source.vertices.max(0)))
    if not len(source.faces):return SurfaceCells.empty()
    try:
        clipped=clip_surface_to_region(SourceSurface(),state.support,request)
        return section_cells(clipped,partition,request,state.hosts).cells()
    except ModelError as error:
        if error.code!='unsupported':raise
    # Combine support and host occupancy before clipping. A float round-trip
    # between two arrangements would create spurious positive sliver polygons.
    from .compiler import CompiledModel
    snapshot=partition.compiled.snapshots[partition.compiled.final_stage_id]
    bodies={oid:replace(body,region=CSGRegion(oid+':interface-support','intersection',(body.region,state.support))) for oid,body in snapshot.bodies.items() if not state.hosts or oid in state.hosts}
    snapshot=replace(snapshot,bodies=bodies)
    compiled=CompiledModel(None,None,None,{snapshot.id:snapshot},snapshot.id)
    common=build_partition(compiled,request)
    return section_cells(state.geometry.cells(request),common,request,state.hosts).cells()


def extract(compiled,exposure,request):
    from .decode import decode_model,encode_model
    from .kernel.maps import AffineMap
    from .kernel.mapped import MappedRegion
    from .deformations import MappedSurface
    from .decode import _keys,_object,COMMON,FIELDS
    from .validation import validate_record
    from .schema import thaw,freeze
    exposure=thaw(freeze(exposure));_keys(exposure,COMMON|FIELDS['exposures'],'exposure');validate_record('exposures',exposure,'exposure');_object(exposure,'exposure')
    stage=exposure.get('at_stage','final');stage=compiled.final_stage_id if stage=='final' else stage
    if stage not in compiled.snapshots:invalid(exposure['id'],'unknown exposure stage')
    compiled=replace(compiled,final_stage_id=stage);snapshot=compiled.snapshots[stage];kind=exposure['kind'];frame=None
    if set(exposure.get('bodies',[]))-set(snapshot.bodies):invalid(exposure['id'],'exposure body unavailable at selected stage')
    if 'frame' in exposure:
        frame=AffineMap(exposure['id']+':frame',exposure['frame']);frame.certify(request)
    support=compiled.kernel.regions.get(exposure.get('support'))
    if 'support' in exposure and support is None:invalid(exposure['id'],'actual exposure support region required')
    if support is not None and frame is not None:support=MappedRegion(exposure['id']+':support',support,frame)
    if frame is not None and kind=='body_boundary' and support is None and not np.array_equal(frame.matrix,np.eye(4)):invalid(exposure['id'],'body boundary frame requires a support selector')
    if kind=='retained_region_boundary':
        retained=compiled.kernel.regions[exposure['geometry']]
        if frame is not None:retained=MappedRegion(exposure['id']+':retained-frame',retained,frame)
        bodies={oid:replace(b,region=CSGRegion(oid+':retained','intersection',(b.region,retained))) for oid,b in snapshot.bodies.items()}
        snapshot=replace(snapshot,bodies=bodies);compiled=replace(compiled,snapshots={**compiled.snapshots,snapshot.id:snapshot})
    partition=build_partition(compiled,request);selected=tuple(exposure.get('bodies',()));contacts=tuple(f for f in partition.boundaries if all(x is not None for x in f.incident));surface_prior=0.
    if kind=='section':
        if exposure['geometry'] not in compiled.kernel.surfaces:invalid(exposure['id'],'section requires an actual surface')
        source=compiled.kernel.surfaces[exposure['geometry']]
        if frame is not None:source=MappedSurface(source,frame)
        surface=section_cells(source.cells(request),partition,request,selected)
    elif kind in {'body_boundary','retained_region_boundary'}:
        polygons=[];records=[]
        for f in partition.boundaries:
            if all(x is not None for x in f.incident):continue
            owner=f.incident[0] or f.incident[1]
            if selected and owner not in selected:continue
            if f.domain_boundary and not exposure.get('include_domain_boundary',False):continue
            side=0 if f.incident[0] is not None else 1
            polygons.append(f.polygon if side==0 else tuple(reversed(f.polygon)));records.append((owner,f.incident_lithologies[side],';'.join(f.source_ids),f.incident,partition.cells[f.cell_ids[side]].unit_id))
        surface=_mesh(polygons,records)
        if partition.error_report.position_error_m:
            from .kernel.certified_arithmetic import float_upper
            from .kernel.predicates import rational
            surface_prior=partition.error_report.position_error_m
            surface=replace(surface,position_error_m=float_upper(rational(surface.position_error_m)+rational(surface_prior)),exact_vertices=None)
    else:invalid(exposure['id'],'supported independent exposure kind required')
    if support is not None and len(surface.faces):
        if surface.position_error_m and surface.exact_vertices is None:unresolved(exposure['id'],'curved support clipping needs a transversality/topology certificate')
        from .kernel.boundary import clip_surface_to_region
        from .kernel.mesh import MeshSurface
        polygons=[];records=[]
        for i,tri in enumerate(surface.cells().exact_triangles):
            cells=clip_surface_to_region(MeshSurface('patch',np.asarray(tri,float),np.array([[0,1,2]],np.int64),tuple(tri),surface.position_error_m),support,request)
            polygons.extend(cells.exact_triangles);records.extend([(surface.body_ids[i],surface.lithology_ids[i],surface.source_ids[i],surface.incident_regions[i],surface.unit_ids[i])]*len(cells.faces))
        prior=surface.position_error_m;surface=_mesh(polygons,records)
        from .kernel.certified_arithmetic import float_upper
        from .kernel.predicates import rational
        surface=replace(surface,position_error_m=float_upper(rational(prior)+rational(surface.position_error_m)))
    groove_certificate=None
    if exposure.get('grooves'):
        from .fractures import build_surface_groove
        groove=build_surface_groove(surface,snapshot.interfaces,exposure['grooves'],compiled.kernel,request)
        surface=groove.surface;groove_certificate=groove._trace_certificate
        surface=section_cells(surface.cells(),partition,request,selected)
    interfaces=tuple(InterfaceMeshV2(oid,_interface_cells(state,partition,request),state.feature_id,state.hosts,state.lineage) for oid,state in sorted(snapshot.interfaces.items()))
    traces=extract_traces({s.id:s.cells for s in interfaces},surface.cells(),snapshot.relations,request,groove_certificate)
    from .kernel.certified_arithmetic import float_upper
    from .kernel.predicates import rational
    extra=max(rational(surface.position_error_m)-rational(surface_prior)+rational(getattr(traces,'_position_error_m',0.)),*[rational(i.cells.position_error_m) for i in interfaces],0)
    error=replace(partition.error_report,position_error_m=float_upper(rational(partition.error_report.position_error_m)+extra))
    if error.position_error_m>request.max_position_error_m:unresolved('exposure','combined surface/region position budget exhausted')
    return ExtractionResult(surface,interfaces,traces,contacts,partition,error)
