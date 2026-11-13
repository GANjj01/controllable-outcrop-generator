from dataclasses import dataclass
import numpy as np
from .errors import invalid,ModelError


@dataclass(frozen=True)
class MaterialSamples:
    lithology_ids: np.ndarray
    composition: tuple
    properties: tuple
    unit_ids: np.ndarray | None = None
    unit_candidates: tuple = ()
    on_interface: np.ndarray | None = None


class MaterialField:
    def __init__(self,lithology_id,composition=None,properties=None,unit_id=None):self.lithology_id=lithology_id;self.composition=composition or {};self.properties=properties or {};self.unit_id=unit_id
    def sample(self,points):
        return MaterialSamples(np.full(len(points),self.lithology_id,dtype=object),tuple(self.composition for _ in points),tuple(self.properties for _ in points),np.full(len(points),self.unit_id,object))
    def mapped(self,mapping):return MappedMaterialField(self,mapping)


class PiecewiseMaterialField:
    def __init__(self,pieces,default,tolerance):self.pieces=tuple(pieces);self.default=default;self.tolerance=tolerance
    def sample(self,points):
        result=self.default.sample(points);ids=result.lithology_ids.copy();composition=list(result.composition);properties=list(result.properties);units=result.unit_ids.copy() if result.unit_ids is not None else np.full(len(points),None,object);claimed=np.zeros(len(points),bool);boundary=np.zeros(len(points),bool);candidates=[set() for p in points];lithos=[set() for p in points]
        for region,field in self.pieces:
            c=region.classify(points,self.tolerance);mask=c.inside|c.boundary
            if np.any(c.inside&claimed):raise ModelError("material","constraint_failed","ambiguous material field interior partition")
            m=field.sample(np.asarray(points)[mask]);ids[mask]=m.lithology_ids
            if m.unit_ids is not None:units[mask]=m.unit_ids
            for index,comp,prop in zip(np.where(mask)[0],m.composition,m.properties):composition[index]=comp;properties[index]=prop
            for index,lid,uid in zip(np.where(mask)[0],m.lithology_ids,m.unit_ids if m.unit_ids is not None else [None]*sum(mask)):
                candidates[index].add(uid);lithos[index].add(lid)
            claimed|=c.inside;boundary|=c.boundary
        for i in np.where(boundary)[0]:
            if len(candidates[i])>1:units[i]=None
            if len(lithos[i])>1:ids[i]=None;composition[i]={};properties[i]={}
        return MaterialSamples(ids,tuple(composition),tuple(properties),units,tuple(tuple(sorted(c,key=lambda x:'' if x is None else x)) for c in candidates),boundary)
    def mapped(self,mapping):return MappedMaterialField(self,mapping)


class MappedMaterialField:
    def __init__(self,field,mapping):self.field=field;self.mapping=mapping
    def sample(self,points):
        from .kernel.types import Tolerance
        candidates=self.mapping.inverse_candidates(points,Tolerance());q=np.zeros_like(np.asarray(points,float));found=np.zeros(len(points),bool)
        for c in candidates:q[c.valid]=c.points[c.valid];found|=c.valid
        if not found.all():raise ModelError("material","unresolved","no inverse material coordinate")
        return self.field.sample(q)
    def mapped(self,mapping):return MappedMaterialField(self,mapping)


def compile_material(spec,lithologies,kernel,tolerance,unit_id=None):
    kind=spec.get("kind","homogeneous")
    def homogeneous(lid,piece_unit_id=unit_id):
        if lid not in lithologies:invalid("material.lithology","unknown lithology")
        record=lithologies[lid];return MaterialField(lid,record.get("composition"),record.get("properties"),piece_unit_id)
    if kind=="homogeneous":
        field=homogeneous(spec['lithology']);field.composition=spec.get('composition',field.composition);field.properties={**field.properties,**spec.get('properties',{})};return field
    if kind=="piecewise":return PiecewiseMaterialField(tuple((kernel.regions[p["geometry"]],homogeneous(p["lithology"],p.get('unit_id',unit_id))) for p in spec["pieces"]),homogeneous(spec["default"]),tolerance)
    invalid("material.kind","unsupported material field")


def material_regions(field):
    """Cut material transitions in the same arrangement as body boundaries."""
    from .kernel.mapped import MappedRegion
    if isinstance(field,PiecewiseMaterialField):return tuple(r for r,f in field.pieces)
    if isinstance(field,MappedMaterialField):return tuple(MappedRegion('material:'+field.mapping.id+':'+r.id,r,field.mapping) for r in material_regions(field.field))
    if hasattr(field,'reference_field'):
        return (field.reference,field.moving,*material_regions(field.reference_field),*material_regions(field.moving_field))
    return ()


def fragment_material(field,p,planes,maps=()):
    """Material authority uses arranged interior signs, never query tolerance."""
    from .kernel.partition import interior
    from .kernel.mapped import MappedRegion
    def occupied(region):
        for mapping in reversed(maps):region=MappedRegion('material:'+mapping.id+':'+region.id,region,mapping)
        return interior(region,p,planes)
    if isinstance(field,MaterialField):return field.sample(np.asarray([p],float))
    if isinstance(field,MappedMaterialField):return fragment_material(field.field,p,planes,(*maps,field.mapping))
    if isinstance(field,PiecewiseMaterialField):
        matches=[f for region,f in field.pieces if occupied(region)]
        if len(matches)>1:raise ModelError('material','constraint_failed','ambiguous exact material fragment')
        return fragment_material(matches[0] if matches else field.default,p,planes,maps)
    if hasattr(field,'reference_field'):
        a,b=occupied(field.reference),occupied(field.moving)
        if a and b and field.policy not in {'keep_reference','keep_moving'}:raise ModelError('material','constraint_failed','ambiguous exact predecessor material fragment')
        if b and (not a or field.policy=='keep_moving'):return fragment_material(field.moving_field,p,planes,maps)
        if a:return fragment_material(field.reference_field,p,planes,maps)
        raise ModelError('material','unresolved','no exact predecessor material fragment')
    raise ModelError('material','unsupported','material field lacks fragment authority')
