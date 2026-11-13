from dataclasses import dataclass,asdict
import hashlib
import json
from .schema import thaw
from .resolution import json_value
from . import BACKEND_VERSION


def canonical_bytes(value):return json.dumps(json_value(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ModelDigests:
    geology: str
    geometry: str
    appearance: str


def _canonical_model(numeric):
    d=thaw(numeric.data)
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='metadata' and not (k=='source' and isinstance(x,dict))}
        if isinstance(v,list):return [clean(x) for x in sorted(v,key=lambda x:x['id'])] if v and all(isinstance(x,dict) and 'id' in x for x in v) else [clean(x) for x in v]
        return v
    return clean(d)


def geology_payload(numeric):
    d=_canonical_model(numeric);geometry={g['id']:g for g in d['geometry']};roots=set()
    def references(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in {'geometry','support','map','field','fault','cutter','source','target'} and isinstance(x,str) and x in geometry:roots.add(x)
                references(x)
        elif isinstance(v,list):
            for x in v:references(x)
    keys=('coordinates','domain','lithologies','bodies','interfaces','features','events','relations')
    payload={k:d[k] for k in keys};references(payload);pending=list(roots)
    while pending:
        oid=pending.pop();before=set(roots);references(geometry[oid]);roots.update(geometry[oid].get('operands',[]));pending.extend(sorted(roots-before))
    payload['geometry']=[geometry[k] for k in sorted(roots)];payload['resources']=[{'id':r['id'],'sha256':r['sha256']} for r in d['resources']]
    return payload


def model_digests(numeric,exposure,request,extraction,appearance):
    geology=digest(geology_payload(numeric));d=_canonical_model(numeric);geometry={g['id']:g for g in d['geometry']};view=geometry.get(exposure.get('geometry'));mesh=extraction.surface
    roots={exposure[k] for k in ('geometry','support') if exposure.get(k) in geometry};pending=list(roots)
    def refs(value):
        if isinstance(value,dict):
            for key,v in value.items():
                if key in {'geometry','support','field','map'} and isinstance(v,str) and v in geometry:yield v
                elif key=='operands':yield from (oid for oid in v if oid in geometry)
                else:yield from refs(v)
        elif isinstance(value,list):
            for v in value:yield from refs(v)
    while pending:
        oid=pending.pop()
        for ref in refs(geometry[oid]):
            if ref not in roots:roots.add(ref);pending.append(ref)
    shape=digest({'geology':geology,'backend':BACKEND_VERSION,'exposure':exposure,'exposure_source':[geometry[k] for k in sorted(roots)],'request':asdict(request),'vertices':mesh.vertices,'faces':mesh.faces,'body_ids':mesh.body_ids,'lithology_ids':mesh.lithology_ids,'unit_ids':mesh.unit_ids,'source_ids':mesh.source_ids,'incident_regions':mesh.incident_regions,'surface_position_error_m':mesh.position_error_m,'interfaces':extraction.interfaces,'traces':extraction.traces,'contacts':contact_payload(extraction)})
    appearance_digest=digest({'geometry':shape,'appearances':d['appearances'],'bindings':d['bindings'],'bound':appearance})
    return ModelDigests(geology,shape,appearance_digest)


def contact_payload(extraction):
    import numpy as np
    return [{'polygon':np.asarray(f.polygon,float).tolist(),'incident':list(f.incident),'incident_lithologies':list(f.incident_lithologies),'source_ids':list(f.source_ids),'domain_boundary':f.domain_boundary} for f in getattr(extraction,'contacts',())]
