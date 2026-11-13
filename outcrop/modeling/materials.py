from dataclasses import dataclass
import numpy as np
from .schema import thaw
from .errors import invalid,ModelError


@dataclass(frozen=True)
class AppearanceBinding:
    face_appearance_ids: tuple
    appearances: dict
    uv_m: np.ndarray


def bind_appearances(numeric,extraction):
    d=thaw(numeric.data);appearances={a['id']:a for a in d['appearances']};mesh=extraction.surface;bound=[]
    for oid,a in appearances.items():
        color=np.asarray(a.get('color',[.65,.65,.65,1]),float)
        if color.shape not in {(3,),(4,)} or not np.isfinite(color).all() or np.any((color<0)|(color>1)):invalid(oid+'.color','finite RGB/RGBA in [0,1] required')
        for key in ('roughness','metallic'):
            v=a.get(key,.8 if key=='roughness' else 0)
            if type(v) not in (int,float) or not 0<=v<=1:invalid(oid+'.'+key,'scalar in [0,1] required')
    for i,(body,lithology) in enumerate(zip(mesh.body_ids,mesh.lithology_ids)):
        candidates=[]
        for rule in d['bindings']:
            selector=rule['selector'];ids=set(mesh.incident_regions[i]) if mesh.incident_regions else {body}
            if 'bodies' in selector and not ids.intersection(selector['bodies']):continue
            if 'lithologies' in selector and lithology not in selector['lithologies']:continue
            if set(selector)-{'bodies','lithologies','ids'}:invalid(rule['id'],'appearance selector backend supports body/lithology/face IDs')
            if 'ids' in selector and mesh.source_ids[i] not in selector['ids']:continue
            priority=rule.get('priority',0)
            if type(priority) is not int:invalid(rule['id']+'.priority','integer priority required')
            candidates.append((priority,rule['appearance']))
        if candidates:
            highest=max(p for p,a in candidates);choices={a for p,a in candidates if p==highest}
            if len(choices)>1:raise ModelError('bindings','constraint_failed','equal-priority appearance conflict')
            bound.append(next(iter(choices)))
        else:bound.append(None)
    uv=[]
    for tri in mesh.vertices[mesh.faces]:
        normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);axes=[j for j in range(3) if j!=int(np.argmax(np.abs(normal)))];uv.append(tri[:,axes])
    return AppearanceBinding(tuple(bound),appearances,np.asarray(uv,float).reshape(-1,3,2))
