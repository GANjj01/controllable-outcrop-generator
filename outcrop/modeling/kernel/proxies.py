"""Exact source-plane proxies and explicit nonlinear zero-witness bounds."""
from dataclasses import dataclass,replace
from functools import cached_property
from itertools import product
import numpy as np
from .analytic import LinearField,QuadraticField,OffsetField
from .grid import GridField
from .mapped import MappedField
from .maps import AffineMap,PeriodicShearMap,ComposedMap
from .polytopes import Plane,ConvexCell
from .predicates import point,affine_coefficients,plane_value,polygon_normal,dot,rational
from .types import BoundaryLeaf
from .error_bounds import linearization_evidence,ApproximationEvidence
from ..errors import unresolved
from .native import get_native


def normalize_field(field):
    if isinstance(field,OffsetField):
        base=normalize_field(field.field)
        if isinstance(base,LinearField):
            from .exact_fields import ExactLinearField
            return ExactLinearField(tuple(rational(field.sign)*rational(x) for x in base.normal),rational(field.sign)*(rational(base.offset)+rational(field.level)),base.unit)
        if isinstance(base,QuadraticField) and field.sign==1:return QuadraticField(base.center,base.matrix,base.level+field.level,base.unit)
        return OffsetField(base,field.level,field.sign)
    if isinstance(field,MappedField):
        from copy import copy
        from .exact_fields import pullback_affine
        base,maps=_field_chain(field);base=normalize_field(base);simplified=[]
        for mapping,inverse in maps:
            if simplified and simplified[-1][0] is mapping and simplified[-1][1]!=inverse:simplified.pop()
            else:simplified.append((mapping,inverse))
        for mapping,inverse in simplified:
            from .maps import DisplacementGridMap
            if isinstance(mapping,DisplacementGridMap) and isinstance(base,LinearField) and all(dot(point(base.normal),point(d))==0 for d in mapping.data.reshape(-1,3)):continue
            if isinstance(mapping,AffineMap):
                transformed=pullback_affine(base,mapping,inverse)
                if transformed is not None:base=transformed;continue
            if inverse and isinstance(mapping,PeriodicShearMap):mapping=copy(mapping);mapping.A=-mapping.A
            elif inverse:return MappedField(base,field.mapping)
            if isinstance(base,LinearField) and isinstance(mapping,PeriodicShearMap):
                if dot(point(base.normal),point(mapping.frame.matrix[:3,2]))==0:continue
            base=MappedField(base,mapping)
        return base
    return field


def _map_chain(mapping,inverse=False):
    if type(mapping).__name__=='InverseMap':return _map_chain(mapping.mapping,not inverse)
    if isinstance(mapping,ComposedMap):return tuple(pair for m in (reversed(mapping.maps) if inverse else mapping.maps) for pair in _map_chain(m,inverse))
    return ((mapping,inverse),)


def _field_chain(field):
    if isinstance(field,OffsetField):
        base,maps=_field_chain(field.field);return OffsetField(base,field.level,field.sign),maps
    if isinstance(field,MappedField):
        base,maps=_field_chain(field.field);return base,maps+_map_chain(field.mapping)
    return field,()


@dataclass(frozen=True)
class GridProxy:
    field: GridField
    level: float
    sign: float
    planes: tuple
    cell_planes: tuple
    tets: tuple
    constants: tuple
    exterior_plane: object = None
    @cached_property
    def _native_grid(self):
        # This prepared state belongs to this immutable proxy. dataclasses.replace
        # (including affine transport) creates a new cache with its new supports.
        supports=[]
        for tet in self.tets:
            faces=[]
            for face in tet.polygons:
                normal=polygon_normal(face);faces.append((normal,dot(normal,face[0])))
            supports.append(tuple(faces))
        fields=tuple((p.normal,p.offset) if p is not None else None for p in self.cell_planes)
        exterior=(self.exterior_plane.normal,self.exterior_plane.offset) if self.exterior_plane is not None else None
        return get_native().prepare_grid(tuple(supports),fields,self.constants,exterior)
    def negative_at(self,p,closed=False):
        native=get_native()
        if native is not None:return native.grid_negative(self._native_grid,p,closed)
        for tet,plane,constant in zip(self.tets,self.cell_planes,self.constants):
            if all(plane_value(polygon_normal(face),dot(polygon_normal(face),face[0]),p)<=0 for face in tet.polygons):
                value=plane_value(plane.normal,plane.offset,p) if plane is not None else constant;return value<=0 if closed else value<0
        if self.exterior_plane is None:return False
        value=plane_value(self.exterior_plane.normal,self.exterior_plane.offset,p);return value<=0 if closed else value<0


def grid_proxy(field,oid):
    base,maps=_field_chain(field)
    if maps and all(isinstance(mapping,AffineMap) for mapping,_ in maps):
        proxy=grid_proxy(base,oid)
        if proxy is None:return None
        for mapping,inverted in maps:proxy=affine_grid_proxy(proxy,mapping,inverted)
        return proxy
    # Scalar signs/offsets commute with spatial pullback. Keep their arithmetic
    # rational so nested band endpoints do not round before source fitting.
    sign=rational(1);shift=rational(0)
    while isinstance(field,OffsetField):
        factor=rational(field.sign);shift-=sign*factor*rational(field.level);sign*=factor;field=field.field
    level=-shift/sign if sign else rational(0)
    from .maps import DisplacementGridMap
    if isinstance(field,MappedField) and isinstance(field.mapping,DisplacementGridMap) and isinstance(field.field,LinearField):return mapped_grid_proxy(field,oid,level,sign)
    if not isinstance(field,GridField):return None
    source=field.cell_complex();planes=[];cell_planes=[];tets=[];constants=[];data=field.data.ravel()
    for ids,sid in zip(source.cells,source.source_ids):
        tet=ConvexCell.tetra(source.vertices[ids],oid+":"+sid);tets.append(tet)
        for i,face in enumerate(tet.polygons):
            n=polygon_normal(face);planes.append(Plane(n,dot(n,face[0]),oid+":"+sid+":support:"+str(i)))
        n,offset=affine_coefficients(source.vertices[ids],tuple(sign*rational(data[i])+shift for i in ids))
        p=Plane(n,offset,oid+":"+sid+":field") if any(n) else None
        constants.append(-offset)
        cell_planes.append(p)
        if p is not None:planes.append(p)
    return GridProxy(field,level,sign,tuple(planes),tuple(cell_planes),tuple(tets),tuple(constants))


def affine_grid_proxy(proxy,mapping,inverted=False):
    from .certified_arithmetic import inverse3
    from .exact_fields import ExactLinearField,pullback_affine
    full=mapping.exact_matrix;matrix=tuple(row[:3] for row in full[:3]);translation=tuple(row[3] for row in full[:3])
    if inverted:
        matrix=inverse3(matrix);translation=tuple(-dot(row,translation) for row in matrix)
    def position(p):return tuple(dot(row,p)+shift for row,shift in zip(matrix,translation))
    transformed={}
    for plane in proxy.planes:
        field=pullback_affine(ExactLinearField(plane.normal,plane.offset),mapping,inverted)
        transformed[id(plane)]=Plane(field.normal,field.offset,plane.source_id)
    tets=tuple(ConvexCell.from_polygons(tuple(tuple(position(p) for p in face) for face in tet.polygons),tet.source_id,tet.constraints) for tet in proxy.tets)
    return replace(proxy,planes=tuple(transformed[id(p)] for p in proxy.planes),
                   cell_planes=tuple(transformed[id(p)] if p is not None else None for p in proxy.cell_planes),
                   tets=tets,exterior_plane=transformed[id(proxy.exterior_plane)] if proxy.exterior_plane is not None else None)


def mapped_grid_proxy(field,oid,level=0.,sign=1.):
    from types import SimpleNamespace
    from .predicates import rational
    level=rational(level);sign=rational(sign);mapping=field.mapping;source=mapping.complex;displacements=mapping.data.reshape(-1,3)
    target=tuple(tuple(x+rational(d) for x,d in zip(point(p),u)) for p,u in zip(source.vertices,displacements));planes=[];cell_planes=[];tets=[];constants=[]
    values=tuple(sign*(dot(point(field.field.normal),point(p))-rational(field.field.offset)-rational(level)) for p in source.vertices)
    for ids,sid in zip(source.cells,source.source_ids):
        vertices=tuple(target[i] for i in ids);tet=ConvexCell.tetra(vertices,oid+':'+sid);tets.append(tet)
        for i,face in enumerate(tet.polygons):
            n=polygon_normal(face);planes.append(Plane(n,dot(n,face[0]),oid+':'+sid+':support:'+str(i)))
        n,offset=affine_coefficients(vertices,tuple(values[i] for i in ids));p=Plane(n,offset,oid+':'+sid+':field') if any(n) else None;cell_planes.append(p);constants.append(-offset)
        if p is not None:planes.append(p)
    exterior=Plane(tuple(sign*rational(x) for x in field.field.normal),sign*(rational(field.field.offset)+rational(level)),oid+':exterior') if mapping.extension=='zero' else None
    if exterior is not None:planes.append(exterior)
    return GridProxy(SimpleNamespace(id=oid),level,sign,tuple(planes),tuple(cell_planes),tuple(tets),tuple(constants),exterior)


def constraints(planes):
    return tuple(p for value in planes.values() for p in (value.planes if isinstance(value,GridProxy) else (value,)))


def field_evidence(field,cell,oid):
    field=normalize_field(field)
    from .certified_arithmetic import quadratic_bounds,float_upper,float_lower,periodic_linear_nodes
    if isinstance(field,QuadraticField):
        delta,position,lower=quadratic_bounds(field,cell.vertices)
        return ApproximationEvidence(True,float_upper(delta),float_upper(position) if position is not None else None,float_lower(lower) if lower is not None else None,"exact rational quadratic radial zero witness",True)
    if isinstance(field,MappedField) and isinstance(field.mapping,PeriodicShearMap) and isinstance(field.field,LinearField):
        samples,delta,position,lower=periodic_linear_nodes(field.field,field.mapping,cell.vertices)
        return ApproximationEvidence(True,float_upper(delta),float_upper(position),float_lower(lower),"rational periodic directional zero witness and sine enclosure",True)
    return linearization_evidence(BoundaryLeaf(oid,field,None,None,"analytic"),cell)


def proxy_plane(field,cell,oid):
    field=normalize_field(field)
    if isinstance(field,LinearField):return Plane(tuple(field.normal),field.offset,oid)
    from .certified_arithmetic import exact_quadratic_nodes,periodic_linear_nodes
    if isinstance(field,QuadraticField):values=exact_quadratic_nodes(field,cell.vertices)
    elif isinstance(field,MappedField) and isinstance(field.mapping,PeriodicShearMap) and isinstance(field.field,LinearField):values=periodic_linear_nodes(field.field,field.mapping,cell.vertices)[0]
    else:unresolved(oid,"no certified nodal oracle")
    n,offset=affine_coefficients(cell.vertices,values)
    if not any(n):
        # A constant field has no separating plane. Give it an arbitrary
        # exterior plane retaining its exact constant sign on this cell.
        x=cell.vertices[:,0];return Plane((1,0,0),float(x.max()+1) if values[0]<0 else float(x.min()-1),oid)
    return Plane(n,offset,oid)
