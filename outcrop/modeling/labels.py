from dataclasses import dataclass
import hashlib
import numpy as np
from .schema import thaw
from .digests import canonical_bytes
from .kernel.types import Budget
from outcrop.dsgen.lithology import order_key   # catalogue rows by the order field, other ids by text


@dataclass(frozen=True)
class FaceLabelsV2:
    face_ids: tuple
    body_ids: tuple
    face_body_indices: np.ndarray
    lithology_ids: tuple
    lithology_codes: np.ndarray
    incident_offsets: np.ndarray
    incident_body_indices: np.ndarray
    source_ids: tuple
    interface_ids: tuple
    feature_ids: tuple
    interface_feature_ids: tuple
    unit_ids: tuple
    face_unit_indices: np.ndarray
    face_feature_offsets: np.ndarray
    face_feature_indices: np.ndarray


def build_labels(numeric,extraction,request=None):
    pair_limit=(request.budget if request is not None else Budget()).max_predicate_pairs
    d=thaw(numeric.data);mesh=extraction.surface;bodies=tuple(sorted(b['id'] for b in d['bodies']));index={oid:i for i,oid in enumerate(bodies)};codes={l['id']:l['code'] for l in d['lithologies']};offsets=[0];flat=[];face_ids=[]
    for i,tri in enumerate(mesh.vertices[mesh.faces]):
        incident=mesh.incident_regions[i] if mesh.incident_regions else (mesh.body_ids[i],)
        flat.extend(-1 if oid is None else index[oid] for oid in sorted(set(incident),key=lambda x:'' if x is None else x));offsets.append(len(flat))
        face_ids.append(hashlib.sha256(canonical_bytes({'triangle':tri,'source':mesh.source_ids[i],'incident':incident})).hexdigest())
    units=tuple(sorted({b['unit_id'] for b in d['bodies'] if b.get('unit_id')}|{p['unit_id'] for b in d['bodies'] for p in b['material'].get('pieces',[]) if p.get('unit_id')}|{u['id'] for f in d['features'] for u in f.get('units',[])}));unit_index={u:i for i,u in enumerate(units)}
    features=tuple(sorted({f['id'] for f in d['features']}|{i.feature_id for i in extraction.interfaces if i.feature_id is not None}));feature_index={f:i for i,f in enumerate(features)};members=[set() for f in mesh.faces]
    from .kernel.spatial import AABBTree
    from .kernel.types import AABB
    from .kernel.intersections import triangle_intersection
    from .errors import unresolved
    triangles=mesh.vertices[mesh.faces];tree=AABBTree(triangles.min(1),triangles.max(1)) if len(triangles) else None;pairs=0
    for state in extraction.interfaces:
        if state.feature_id is None:continue
        for tri in state.cells.vertices[state.cells.faces]:
            for i in tree.query(AABB(tuple(tri.min(0)),tuple(tri.max(0)))) if tree else ():
                pairs+=1
                if pairs>pair_limit:unresolved('labels','feature membership pair budget exhausted')
                if triangle_intersection(tri,triangles[i])[0]!='empty':members[i].add(feature_index[state.feature_id])
    member_offsets=[0];member_flat=[]
    for row in members:member_flat.extend(sorted(row));member_offsets.append(len(member_flat))
    return FaceLabelsV2(tuple(face_ids),bodies,np.array([index.get(b,-1) for b in mesh.body_ids],np.int64),tuple(sorted(codes, key=order_key)),np.array([codes.get(l,0) for l in mesh.lithology_ids],np.uint32),np.array(offsets,np.int64),np.array(flat,np.int64),mesh.source_ids,tuple(i.id for i in extraction.interfaces),features,tuple(i.feature_id for i in extraction.interfaces),units,np.array([unit_index.get(u,-1) for u in mesh.unit_ids or (None,)*len(mesh.faces)],np.int64),np.array(member_offsets,np.int64),np.array(member_flat,np.int64))
