"""Bind the existing main-UI t02 paths to local inward frames; never regenerate paths."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent
P1=ROOT.parent
sys.path.insert(0,str(P1))
from frgb.network_profiles import load_frozen_network,classify_path_endpoints
from frgb import fault_charts
from frgb.tracechains import build_chains
from outcrop.modeling.kernel.spatial import AABBTree
from outcrop.modeling.kernel.types import AABB
from outcrop.modeling.kernel.mesh import triangle_distances

# supplies SOURCE; missing or empty source.txt fails closed.
SOURCE=(ROOT/'source.txt').read_text(encoding='utf8').strip()
if not SOURCE:raise ValueError('source.txt must contain a source geometry path')
SOURCE=Path(SOURCE)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,v:p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def normals_on(points, vertices, faces):
    tri=vertices[faces]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    tree=AABBTree(tri.min(1),tri.max(1))
    output=[];residual=[]
    for p in points:
        ids=tree.query(AABB(tuple(p-1e-6),tuple(p+1e-6)))
        if not len(ids):raise ValueError('Source path point has no source surface face')
        dist=triangle_distances(p[None],tri[ids])[0]
        chosen=ids[dist<=max(1e-8,dist.min()+1e-10)]
        if dist.min()>1e-7:raise ValueError('Source path is not on the original surface')
        n=normals[chosen].sum(0)
        if np.linalg.norm(n)<1e-8:raise ValueError('Ambiguous surface normal')
        output.append(n/np.linalg.norm(n));residual.append(float(dist.min()))
    return np.asarray(output),max(residual)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    network=load_frozen_network(SOURCE)
    # is classified against its trace with the kernel's translation undone; no fault event -> 'no_faults', nothing changes
    frames=fault_charts.fault_frames(SOURCE)
    if frames['status']=='unsupported':raise ValueError('fault frames unsupported: '+'; '.join(frames['reasons']))
    source=np.load(SOURCE/'source_surface.npz')
    chains=build_chains(network.actual_segments_3d_m,network.interface_ids)
    combined=np.vstack([np.vstack([c.segments[:,0],c.segments[-1,1]]) for c in chains])
    outward,residual=normals_on(combined,source['vertices'],source['faces'])
    interfaces=read(SOURCE/'package/interfaces.json')
    interface_arrays=np.load(SOURCE/'package/interfaces.npz')
    rows=[];offset=0
    excluded=[];index=-1
    for c in chains:
        points=np.vstack([c.segments[:,0],c.segments[-1,1]])
        no=outward[offset:offset+len(points)];offset+=len(points)
        # is not a fracture path: recorded in excluded_contacts, never bound; path indices count fracture chains only
        if network.interface_trace_ids.get(c.interface_id) is None:
            excluded.append(dict(interface_id=c.interface_id,chain_id=c.chain_id,segments=len(points)-1));continue
        index+=1
        iface_index=next(i for i,x in enumerate(interfaces) if x['id']==c.interface_id)
        transverse,_=normals_on(points,interface_arrays[f'vertices_{iface_index}'],interface_arrays[f'faces_{iface_index}'])
        edges=np.diff(points,axis=0);lengths=np.linalg.norm(edges,axis=1)
        unit=edges/lengths[:,None]
        tangent=np.vstack([unit[0],unit[:-1]+unit[1:],unit[-1]])
        tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        transverse-=np.sum(transverse*tangent,axis=1)[:,None]*tangent
        transverse/=np.linalg.norm(transverse,axis=1)[:,None]
        for j in range(1,len(transverse)):
            if transverse[j]@transverse[j-1]<0:transverse[j]*=-1
        inward=np.cross(tangent,transverse)
        sign=np.sum(inward*no,axis=1)>0
        inward[sign]*=-1
        trace=network.trace(network.interface_trace_ids[c.interface_id])
        shift,frame_records=fault_charts.path_frame(frames,c.interface_id,points)
        free,endpoint_evidence=classify_path_endpoints(network,trace,points if shift is None else points-shift,closed=c.closed)
        arc=np.r_[0.,np.cumsum(lengths)]
        path=ROOT/f'path_{index:02d}.npz'
        np.savez_compressed(path,points=points,inward=inward,normal=transverse,outward=no,
                            arc=arc,segment_ids=np.asarray(c.segment_ids))
        rows.append(dict(index=index,interface_id=c.interface_id,chain_id=c.chain_id,
            source_trace_id=trace.id,phase_rad=trace.phase_rad,segments=len(lengths),length_m=float(arc[-1]),
            free_ends=free,endpoint_evidence=endpoint_evidence,file=path.name,
            minimum_segment_length_m=float(lengths.min()),maximum_inward_outward_dot=float(np.max(np.sum(inward*no,axis=1))),
            **({} if shift is None else dict(fault_frame=fault_charts.frame_record(frames,shift,frame_records)))))
        print(json.dumps(rows[-1]),flush=True)
    assert sum(x['segments'] for x in rows)+sum(x['segments'] for x in excluded)==len(network.actual_segments_3d_m)
    network.assert_source_unchanged()
    save(ROOT/'source_binding.json',dict(source=str(SOURCE),source_network_traces=len(network.traces),
        source_network_topology=dict(network.node_counts),source_network_components=network.connected_components,
        exposure_paths=rows,exposure_segments=len(network.actual_segments_3d_m),
        **({} if not excluded else dict(excluded_contacts=excluded)),
        source_surface_maximum_path_distance_m=residual,
        note='Every actual exposure segment is retained; topology of XY geological network differs from exposure intersections.'))


if __name__=='__main__':main()
