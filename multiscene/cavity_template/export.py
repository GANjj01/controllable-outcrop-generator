"""Export native Boolean provenance and actual host/cavity interface candidates."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent))
from prepare import read,save
from source_tables import load_sources
from frgb.morphology import flat_display
from outcrop.dsgen.declared import _native_preview
from outcrop.dsgen.pipeline import _preview


def export(name,glb_previews=False):
    folder=ROOT/'cases'/name
    report=read(folder/'build.json')
    data=np.load(folder/'difference.npz')
    vertices=data['vertices'];faces=data['faces'].astype(np.int64)
    origin=data['original_id'];source_faces=data['source_face_id']
    host=origin==report['host_original_id']
    n=len(faces)
    roles=np.full(n,'original_host',dtype='<U32')
    source_ids=np.full(n,'original_host',dtype='<U96')
    records=None
    for row in report['paths']:
        records=load_sources(folder,row['index'])['face_sources']
        for j in np.flatnonzero(origin==row['original_id']):
            record=records[str(int(source_faces[j]))]
            roles[j]=record['role']
            source_ids[j]=f'path{row["index"]:02d}:segment{record["source_segment_id"]}'
    # the display shares the label arrays of native instead of copying them; every output is unchanged.
    del records
    tri=vertices[faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area2=np.linalg.norm(cross,axis=1)
    if np.any(area2<=0):raise ValueError('Degenerate Boolean output triangle')
    area2_min=area2.min();del tri,area2
    edges=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    undirected=np.sort(edges,axis=1)
    unique,inverse,counts=np.unique(undirected,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts!=2):raise ValueError('Boolean output is not closed index-manifold')
    order=np.argsort(inverse,kind='stable').reshape(-1,2)
    pairs=order//3
    if np.any(np.sum(np.where(edges[:,0]<edges[:,1],1,-1)[order],axis=1)!=0):
        raise ValueError('Boolean output edge winding mismatch')
    mouth=np.flatnonzero(host[pairs[:,0]]!=host[pairs[:,1]])
    rows=[]
    for k in mouth:
        pair=pairs[k]
        cavity=int(pair[0] if not host[pair[0]] else pair[1])
        role=str(roles[cavity])
        rows.append(dict(source_vertex_indices=unique[k].tolist(),source_face_indices=pair.tolist(),
            segment_enu_m=vertices[unique[k]].tolist(),length_m=float(np.linalg.norm(vertices[unique[k,1]]-vertices[unique[k,0]])),
            cavity_role=role,source_id=str(source_ids[cavity]),
            candidate=role in ('left_wall','right_wall'),
            status=('SIDE_WALL_CANDIDATE_VISIBILITY_PENDING' if role in ('left_wall','right_wall') else
                    'JUNCTION_BRIDGE_CLASSIFICATION_PENDING' if role=='junction_bridge' else 'CAP_OR_FLOOR_EXCLUDED_PENDING_REVIEW')))
    del edges,undirected,unique,inverse,counts,order,pairs,mouth
    normals=np.zeros_like(vertices)
    for i in range(3):np.add.at(normals,faces[:,i],cross)
    del cross
    norms=np.linalg.norm(normals,axis=1)
    if np.any(norms<=0):raise ValueError('Undefined native vertex normal')
    normals/=norms[:,None]
    del norms
    native=dict(vertices=vertices,faces=faces,normals=normals,codes=np.ones(n,np.int32),
        body_ids=np.full(n,'rock-body'),unit_ids=np.full(n,''),lithology_ids=np.full(n,'diagnostic-rock'),
        body_valid=np.ones(n,bool),unit_valid=np.zeros(n,bool),lithology_valid=np.ones(n,bool),
        source_ids=source_ids,face_roles=roles)
    geo,corners=flat_display(dict(vertices=vertices,faces=faces))
    display={k:geo.get(k,v) for k,v in native.items()};del geo
    output=folder/'geometry'
    output.mkdir()
    book=[dict(code=1,body='rock-body',unit=None,lithology='diagnostic-rock',appearance='uniform-gray',
        color_srgb=[137,137,137],color_linear=[.25,.25,.25,1.],roughness=.8,metallic=0)]
    np.savez_compressed(output/'native_surface.npz',**native)
    np.savez_compressed(output/'surface.npz',**display)
    np.savez_compressed(output/'display_provenance.npz',source_vertex_indices=corners,source_face_indices=np.arange(n))
    save(output/'rim_candidates.json',dict(status='NOT_ACCEPTED_TRAINING_LABELS',
        construction='shared native edge with exactly one original-host face and one cavity face',segments=rows))
    save(output/'codebook.json',book)
    if glb_previews:
        _native_preview(native,native['codes'],book,output/'native_preview.glb')
        _native_preview(display,display['codes'],book,output/'preview.glb')
        _preview(display,display['codes'],book,output/'gray.glb',gray=True)
    boundary=dict(contract='closed_rock',closed=True,physical_boundary_edges=0,index_boundary_edges=0,
        t_junction_boundary_edges=0,nonmanifold_edges=0,positive_triangle_areas=True)
    provenance=dict(source='native_surface.npz',display='surface.npz',correspondence='display_provenance.npz',
        operation='triangle corner split with flat normals',positions_winding_face_order_face_labels_preserved=True,
        geometry_origin='Manifold native host difference; source original IDs and face IDs in ../difference.npz')
    save(output/'display_provenance.json',provenance)
    checks=dict(status='CLOSED_DERIVED_GEOMETRY_ONLY',vertices=len(vertices),faces=n,
        boundary_edges=0,nonmanifold_edges=0,minimum_triangle_area_m2=float(area2_min/2),
        original_host_faces=int(host.sum()),cavity_faces=int((~host).sum()),mouth_edges=len(rows),
        side_wall_candidates=sum(row['candidate'] for row in rows),
        junction_bridge_edges=sum(row['cavity_role']=='junction_bridge' for row in rows),
        excluded_cap_or_floor_edges=sum(row['cavity_role'] not in ('left_wall','right_wall','junction_bridge') for row in rows),
        geological_label_transfer=False,visibility_checked=False,training_qualified=False)
    save(output/'geometry_checks.json',checks)
    save(output/'geometry.json',dict(vertices=len(display['vertices']),faces=n,pipeline_version=2,
        geometry_kind='closed_rock',boundary=boundary,validation=dict(success=True,boundary=boundary,
        geological_labels='Single gray diagnostic code only; existing geological source files preserved'),
        geometry_family='original-ui-network-aperture-r1',case_id=name,parameters=report['parameters'],
        bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],source_vertices=len(vertices),source_faces=n,
        source_surface='native_surface.npz',surface='surface.npz',codebook='codebook.json',preview_glb='preview.glb',
        gray_glb='gray.glb',geometry_checks='geometry_checks.json',display_derivation=provenance,
        geometry_frozen=True,boundary_measured_on='native exact index topology',trace_count=0,trace_interfaces=[],layer_count=0,exposed_units=[]))
    print(checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--glb-previews',action='store_true')
    args=parser.parse_args();export(args.name,glb_previews=args.glb_previews)
