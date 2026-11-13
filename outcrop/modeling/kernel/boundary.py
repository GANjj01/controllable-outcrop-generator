"""Common exact linear/PL arrangements used by relations and extraction."""
from dataclasses import dataclass
import numpy as np
from .analytic import LinearField,OffsetField
from .polytopes import ConvexCell,Plane,split_polygon
from .predicates import point,polygon_normal,dot,facet_key,plane_sign,plane_rows,filtered_signs
from .arrangement import arrange_cell
from .surface_arrangement import cells_from_polygons
from .types import AABB
from ..errors import ModelError,unresolved


def linear_plane(field,source):
    from .proxies import normalize_field
    field=normalize_field(field)
    if isinstance(field,LinearField):return Plane(tuple(field.normal),field.offset,source)
    if isinstance(field,OffsetField):
        plane=linear_plane(field.field,source)
        return Plane(tuple(field.sign*x for x in plane.normal),field.sign*(plane.offset+field.level),source)
    raise ModelError(source,"unsupported","nonlinear boundary requires certified partition backend")


def region_planes(region,box):
    planes=[]
    for leaf in region.boundary_leaves(box):
        if leaf.field is not None:planes.append(linear_plane(leaf.field,leaf.source_id))
        else:
            surface=leaf.surface
            for i,tri in enumerate(surface.vertices[surface.faces]):
                tri=tuple(point(p) for p in tri);n=polygon_normal(tri);planes.append(Plane(n,dot(n,tri[0]),f"{leaf.source_id}:{i}"))
    return planes


@dataclass(frozen=True)
class RegionFacet:
    polygon: tuple
    incident: tuple
    domain_boundary: bool


def common_region_facets(regions,box,request):
    constraints=[p for r in regions.values() for p in region_planes(r,box)];arranged=arrange_cell(ConvexCell.box(box.minimum,box.maximum,"region-roi"),constraints,request)
    facets={}
    for cell in arranged.cells:
        p=cell.centroid[None,:];labels=[]
        for oid,region in sorted(regions.items()):
            c=region.classify(p,request.tolerance)
            if c.unresolved[0] or c.boundary[0]:unresolved(oid,"fragment ownership not interior to classified arrangement")
            if c.inside[0]:labels.append(oid)
        if len(labels)>1:raise ModelError("partition","constraint_failed","unresolved body overlap")
        label=labels[0] if labels else None
        for polygon in cell.polygons:facets.setdefault(facet_key(polygon),[]).append((polygon,label))
    result=[]
    for key,entries in sorted(facets.items()):
        labels=tuple(e[1] for e in entries)
        if len(entries)>2:unresolved("partition","nonmanifold facet adjacency")
        if len(entries)==2 and labels[0]==labels[1]:continue
        if len(entries)==1:labels=labels+(None,)
        result.append(RegionFacet(entries[0][0],labels,len(entries)==1))
    return tuple(result)


def region_boundary_cells(region,request):
    facets=common_region_facets({region.id:region},region.bounds(),request)
    selected=[f.polygon for f in facets if region.id in f.incident]
    return cells_from_polygons(selected,[region.id]*len(selected))


def clip_surface_to_region(surface,region,request):
    source=surface.cells(request)
    if source.position_error_m and not hasattr(source,'exact_vertices'):unresolved(surface.id,'curved clipping relation needs a transversality/topology certificate')
    from .partition import interior
    from .proxies import normalize_field,grid_proxy,constraints
    box=surface.bounds();by_source={};polygons=[];sources=[]
    # planes. Keep the proxy under the original leaf ID for exact Boolean
    # classification; extended local planes alone are not the field boundary.
    for leaf in region.boundary_leaves(box):
        if leaf.field is not None:
            field=normalize_field(leaf.field);proxy=grid_proxy(field,leaf.source_id)
            by_source[leaf.source_id]=proxy if proxy is not None else linear_plane(field,leaf.source_id)
        else:
            mesh=leaf.surface
            for i,triangle in enumerate(mesh.vertices[mesh.faces]):
                triangle=tuple(point(p) for p in triangle);n=polygon_normal(triangle);oid=f'{leaf.source_id}:{i}'
                by_source[oid]=Plane(n,dot(n,triangle[0]),oid)
    planes=constraints(by_source)
    ordered=sorted({p.key:p for p in planes}.values(),key=lambda p:p.key);N,O=plane_rows(ordered)
    for triangle,source_id in zip(source.exact_triangles,source.source_ids):
        tri=tuple(point(p) for p in triangle);parts=[tri]
        # Fast path for the untouched source triangle: signs against every plane at once; a
        # plane with all vertices strictly on one side leaves it as it is (clip_polygon returns
        # the triangle itself on that side and nothing on the other).
        clean=[]
        for p in tri:
            if not clean or clean[-1]!=p:clean.append(p)
        if len(clean)>1 and clean[0]==clean[-1]:clean.pop()
        fast=tuple(clean)==tri;S=filtered_signs(N,O,tri) if fast else None
        keep=fast and len(tri)>=3 and any(polygon_normal(tri))
        for j,plane in enumerate(ordered):
            next_parts=[]
            for poly in parts:
                if poly is tri and fast:
                    s=[int(S[j,i]) or plane_sign(plane.normal,plane.offset,p) for i,p in enumerate(poly)]
                    if all(x<0 for x in s) or all(x>0 for x in s):
                        if keep:next_parts.append(tri)
                        continue
                else:s=[plane_sign(plane.normal,plane.offset,p) for p in poly]
                a,b=split_polygon(poly,plane,signs=s)
                if facet_key(a)==facet_key(b):next_parts.append(a)
                else:
                    for q in (a,b):
                        if len(q)>=3 and any(polygon_normal(q)):next_parts.append(q)
            parts=next_parts
            if len(parts)>request.budget.max_fragments_per_cell:unresolved(region.id,"surface fragment budget exhausted")
        kept=[]
        for poly in parts:
            p=tuple(sum(v[j] for v in poly)/len(poly) for j in range(3))
            if interior(region,p,by_source,closed=True):kept.append(poly)
        # Auxiliary supporting planes do not change an entirely retained
        # source triangle and must not persist as artificial subdivisions.
        selected=[tuple(triangle)] if len(kept)==len(parts) else kept
        polygons.extend(selected);sources.extend([source_id]*len(selected))
    return cells_from_polygons(polygons,sources)
