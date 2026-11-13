"""Build a derivative of the original UI rock with its own original paths."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent))
from frgb import runconfig as rc
sys.path.insert(0,rc.path('manifold_site').as_posix())
import manifold3d as md
from frgb.network_profiles import (evaluate_discrete_arc_profile,refine_path_stations,
    load_frozen_network,decode_genome,genome_parameter_names)
from frgb.network_cutters import build_network_cutter,union_cutters,clip_cutter_by_halfspaces
from frgb.network_profiles import end_edge_halfspace,CLOSURE_INWARD_ERROR
from frgb import fault_charts
from prepare import SOURCE,save,read
from source_tables import save_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--path',type=int,action='append')
    parser.add_argument('--width',type=float,default=.06)
    parser.add_argument('--variation',type=float,default=.3)
    parser.add_argument('--depth',type=float,default=.18)
    parser.add_argument('--roughness',type=float,default=.004)
    parser.add_argument('--cutter-simplify-m',type=float)
    parser.add_argument('--joint',choices=['hull','mitre_wedge'],default='hull')
    parser.add_argument('--construction',choices=['segment_union','panel'],default='segment_union')
    parser.add_argument('--panel-refine-m',type=float,default=.02)
    parser.add_argument('--monte-carlo',action='store_true')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--wall-relief',type=Path)
    parser.add_argument('--wall-relief-tip-m',type=float)
    # VISIBLE stations (in the frame, not occluded), widened by --view-clip-margin-m + --view-clip-margin-px x footprint (sigma
    # half-planes, the mechanism of the finite-support end clips); a path with no visible station is dropped.
    parser.add_argument('--view-clip',type=Path)
    parser.add_argument('--view-clip-margin-px',type=float,default=256.)
    parser.add_argument('--view-clip-margin-m',type=float,default=.3)
    # dilated) are refined to --wall-mouth-fine-m; elsewhere --panel-refine-m (network_panels wall_refine)
    parser.add_argument('--wall-mouth-band',type=Path)
    parser.add_argument('--wall-mouth-fine-m',type=float)
    # json = the earlier format, byte for byte. build.json and every other output are unchanged.
    parser.add_argument('--sources-format',choices=['npz','json'],default='npz')
    # a fault acted on, the cavity before and after fault_charts.replay_cavity as DIR/<trace>_<block>_{source,world}.npz
    parser.add_argument('--save-fault-cavities',type=Path)
    args=parser.parse_args()
    if (args.wall_mouth_band is None)!=(args.wall_mouth_fine_m is None):parser.error('--wall-mouth-band and --wall-mouth-fine-m go together')
    output=ROOT/'cases'/args.name
    if output.exists():raise FileExistsError(output)
    if args.save_fault_cavities is not None and (args.save_fault_cavities.resolve()==output.resolve() or output.resolve() in args.save_fault_cavities.resolve().parents):
        parser.error('--save-fault-cavities must lie outside the case output directory')
    output.mkdir(parents=True)
    source_dir=args.source or SOURCE
    network=load_frozen_network(source_dir)
    # 'no_faults' and every step below is the unchanged one
    frames=fault_charts.fault_frames(source_dir)
    if frames['status']=='unsupported':raise ValueError('fault frames unsupported: '+'; '.join(frames['reasons']))
    if frames['status']=='ok' and args.construction!='panel':raise ValueError('fault slip needs --construction panel')
    binding=read(ROOT/'source_binding.json')
    selected=[x for x in binding['exposure_paths'] if args.path is None or x['index'] in args.path]
    if not selected:raise ValueError('No source paths selected')
    has_tips=any(any(row['free_ends']) for row in selected)
    genome=[args.width,args.variation,args.depth,args.roughness]+([.18] if has_tips else [])
    params=decode_genome(genome,has_true_free_ends=has_tips)
    host_arrays=np.load(source_dir/'source_surface.npz')
    host_id=md.Manifold.reserve_ids(1)
    host=md.Manifold(md.Mesh64(host_arrays['vertices'],host_arrays['faces'].astype(np.uint64),
        run_index=np.asarray([0,host_arrays['faces'].size],np.uint64),
        run_original_id=np.asarray([host_id],np.uint32),face_id=np.arange(len(host_arrays['faces']),dtype=np.uint64)))
    assert host.status()==md.Error.NoError and host.volume()>0
    cutters=[];paths=[];panel_rows=[]
    for row in selected:
        arrays=np.load(ROOT/row['file'])
        refined=refine_path_stations(arrays['points'],arrays['segment_ids'],row['length_m']/64)
        points=refined['points']
        si=refined['segment_index']; sf=refined['source_fraction'][:,None]
        inward=(1-sf)*arrays['inward'][si]+sf*arrays['inward'][si+1]
        normal=(1-sf)*arrays['normal'][si]+sf*arrays['normal'][si+1]
        inward/=np.linalg.norm(inward,axis=1)[:,None]
        normal-=np.sum(normal*inward,axis=1)[:,None]*inward
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        fields=evaluate_discrete_arc_profile(params,refined['arc_m'],row['length_m'],free_ends=row['free_ends'],
            phase=row['phase_rad'],depth_fraction=np.linspace(0,1,5))
        width=np.broadcast_to(fields['mouth_width_m'][:,None],fields['common_displacement_m'].shape)
        depth=fields['depth_m'][:,None]*fields['depth_fraction'][None,:]
        print(json.dumps(dict(event='path_start',index=row['index'],segments=row['segments'])),flush=True)
        if args.construction=='panel':
            assert set(map(int,refined['original_segment_id']))==set(map(int,arrays['segment_ids']))
            assert np.array_equal(points[refined['source_fraction']==0],arrays['points'][:-1])
            assert np.array_equal(points[-1],arrays['points'][-1])
            shift,frame_records=fault_charts.path_frame(frames,row['interface_id'],points)
            # the shift must be prepare.py's (source_binding fault_frame), exactly; otherwise fail closed
            if (shift is None)!=('fault_frame' not in row) or (shift is not None and not np.array_equal(shift,np.asarray(row['fault_frame']['shift_m'],float))):
                raise ValueError('path %d: fault frame differs from source_binding.json'%row['index'])
            panel_rows.append(dict(row=row,arrays=arrays,refined=refined,inward=inward,normal=normal,fields=fields,
                                   width=width,depth=depth,shift=shift,frame_records=frame_records))
            continue
        cutter=build_network_cutter(points,inward,normal,width,depth,
            fields['common_displacement_m'],fields['differential_displacement_m'],outside_margin_m=.12,
            original_id=md.Manifold.reserve_ids(1),source_segment_ids=refined['original_segment_id'][:-1],mode='segment_union',
            simplify_m=args.cutter_simplify_m,joint=args.joint)
        trace=network.trace(row['source_trace_id'])
        # Both exposed ends may lie on the same source end edge (paths 04/05): one plane per source endpoint.
        planes={}
        for k,ev in enumerate(row['endpoint_evidence']):
            if ev['footprint_distance_m']<=1e-7 and ev['source_endpoint'] not in planes:
                g,c=end_edge_halfspace(network,trace,ev['source_endpoint'])
                assert np.max(arrays['points']@g-c)<=1e-7, 'original path is not behind its source end face'
                planes[ev['source_endpoint']]=(g,c,int(arrays['segment_ids'][0 if k==0 else -1]))
        planes=list(planes.values())
        if planes:
            cutter=clip_cutter_by_halfspaces(cutter,planes)
        assert np.array_equal(cutter.source_points,points)
        assert set(map(int,refined['original_segment_id']))==set(map(int,arrays['segment_ids']))
        assert np.array_equal(points[refined['source_fraction']==0],arrays['points'][:-1])
        assert np.array_equal(points[-1],arrays['points'][-1])
        cutters.append(cutter)
        save_sources(output,row['index'],dict(original_id=cutter.original_id,face_sources=cutter.face_sources,
            junction_bridges=cutter.junction_bridges,construction_checks=cutter.construction_checks),args.sources_format,save)
        np.savez_compressed(output/f'path_{row["index"]:02d}_stations.npz',**refined,
            inward=inward,normal=normal,width=width,depth=depth,
            common=fields['common_displacement_m'],differential=fields['differential_displacement_m'])
        paths.append(dict(**row,original_id=cutter.original_id,faces=cutter.solid.num_tri(),volume_m3=cutter.solid.volume(),
            refined_segments=len(points)-1,continuous_to_discrete_factor=fields['continuous_to_discrete_factor'],
            actual_pl_mean_opening_m=fields['actual_pl_mean_opening_m'],construction_checks=cutter.construction_checks))
        print(json.dumps(dict(event='path_complete',index=row['index'],faces=cutter.solid.num_tri())),flush=True)
    if args.construction=='panel':
        solids,paths=build_panels(panel_rows,network,output,args,frames)
        # after the replay; a failure raises (fail closed), the records go into build.json only (the sources are already saved)
        if frames['status']=='ok' and FAULT_STRUCTURE:fault_charts.structure_checks(host,host_id,solids,FAULT_STRUCTURE,frames,(host_arrays['vertices'],host_arrays['faces']))
        void=md.Manifold.batch_boolean(solids,md.OpType.Add) if len(solids)>1 else solids[0]
        if void.status()!=md.Error.NoError or void.volume()<=0:
            raise ValueError('network union failed: '+str(void.status()))
    else:
        void=union_cutters(cutters)
    components=void.decompose()
    print(json.dumps(dict(event='network_union_complete',components=len(components),
        volumes_m3=[p.volume() for p in components])),flush=True)
    result=host-void
    if result.status()!=md.Error.NoError or not 0<result.volume()<host.volume():
        raise ValueError('Native-host difference failed')
    mesh=result.to_mesh64()
    vertices=np.asarray(mesh.vert_properties)[:,:3];faces=np.asarray(mesh.tri_verts)
    origin=np.empty(len(faces),np.uint32)
    for start,end,oid in zip(mesh.run_index[:-1],mesh.run_index[1:],mesh.run_original_id):origin[start//3:end//3]=oid
    np.savez_compressed(output/'difference.npz',vertices=vertices,faces=faces,original_id=origin,
                        source_face_id=mesh.face_id)
    save(output/'build.json',dict(source=str(source_dir),**({} if frames['status']=='no_faults' else dict(fault_frames=[dict(id=f['id'],
        t_m=[float(x) for x in f['t']],t_exact=f['t_exact'],moving_side=f['moving_side'],moved_joints=sorted(f['moved'])) for f in frames['faults']])),
        wall_relief=None if args.wall_relief is None else dict(path=str(args.wall_relief),sha256=__import__('hashlib').sha256(Path(args.wall_relief).read_bytes()).hexdigest()),**({} if args.wall_relief_tip_m is None else dict(wall_relief_tip_m=args.wall_relief_tip_m)),
        **({} if args.wall_mouth_band is None else dict(wall_mouth_band=dict(path=str(args.wall_mouth_band),sha256=__import__('hashlib').sha256(Path(args.wall_mouth_band).read_bytes()).hexdigest(),fine_m=args.wall_mouth_fine_m,coarse_m=args.panel_refine_m))),parameters=asdict(params),host_original_id=host_id,
        genome=genome,active_parameter_names=genome_parameter_names(has_true_free_ends=has_tips),
        cutter_simplify_m=args.cutter_simplify_m,joint=args.joint,construction=args.construction,
        source_host_faces=len(host_arrays['faces']),result_faces=len(faces),
        host_volume_m3=host.volume(),remaining_volume_m3=result.volume(),removed_volume_m3=host.volume()-result.volume(),
        cutter_components=len(components),cutter_component_volumes_m3=[p.volume() for p in components],
        original_network_geometry_connectivity_verified=False,
        status='DERIVED_GEOMETRY_ONLY_QUALIFICATION_PENDING',paths=paths,
        paths_preserved=True,full_exposure_coverage=args.path is None,model_scoring=False))
    network.assert_source_unchanged()
    print(json.dumps(dict(event='difference_complete',output=str(output),faces=len(faces),removed_m3=host.volume()-result.volume())),flush=True)


def wall_relief_of(spec_path):
    """rho(P) = the host's self-affine D1 relief restricted to the spec's groups (same plane waves as the host: the octave
    list keeps its order, unused groups get rms 0), or None."""
    if spec_path is None:
        return None
    from outcrop.dsgen.morph import self_affine_relief
    sp=json.loads(Path(spec_path).read_text(encoding='utf8'))
    octs=[[o[0],o[1],o[2] if k in sp['groups'] else 0.,o[3]] for k,o in enumerate(sp['octaves'])]
    return lambda P:self_affine_relief(np.asarray(P,float),octs,int(sp['seed']))


def mouth_band_of(band_path):
    if band_path is None:
        return None
    from frgb.raster import camera
    z=np.load(band_path);band=z['band'];meta=json.loads(str(z['meta']))
    dmin,dmax=(z['dmin'],z['dmax']) if 'dmin' in z.files else (None,None)
    cam=camera(meta['shot'],meta['resolution']);hh,ww=band.shape
    def fine_of(P):
        rel=np.asarray(P,float)-cam['pos'];zz=rel@cam['fwd'];ok=zz>cam['near'];zs=np.where(ok,zz,1.)
        x=ww/2+cam['fpx']*(rel@cam['right'])/zs-.5;y=hh/2-cam['fpx']*(rel@cam['up'])/zs-.5
        xi,yi=np.round(x).astype(np.int64),np.round(y).astype(np.int64)
        ok&=(xi>=0)&(xi<ww)&(yi>=0)&(yi<hh)
        out=np.zeros(len(rel),bool);out[ok]=band[yi[ok],xi[ok]]
        if dmin is not None:out[ok]&=(zz[ok]>=dmin[yi[ok],xi[ok]])&(zz[ok]<=dmax[yi[ok],xi[ok]])
        return out
    return fine_of


_HOST_DEPTH={}
FAULT_STRUCTURE=[]


def view_clip_planes(chart,points,segment_ids,args,shift=None):
    """sigma half-planes around the stations that are VISIBLE at the camera: inside the frame and not occluded; extended on
    both sides by margin_m + margin_px x the largest pixel footprint of the visible stations. None if no station is
    visible."""
    from frgb.raster import camera,rasterize
    req=json.loads(Path(args.view_clip).read_text(encoding='utf8'))
    key=str(args.view_clip)
    if key not in _HOST_DEPTH:
        h=np.load(Path(args.source or SOURCE)/'source_surface.npz');cam0=camera(req['shots'][0],(512,512))
        _HOST_DEPTH[key]=(cam0,rasterize(h['vertices'],h['faces'].astype(np.int64),cam0)['depth'])
    cam0,depth=_HOST_DEPTH[key]
    rel=np.asarray(points,float)-cam0['pos'];z=rel@cam0['fwd'];zz=np.where(z>cam0['near'],z,np.inf)
    x=(rel@cam0['right'])/zz*cam0['fpx']+256;y=-(rel@cam0['up'])/zz*cam0['fpx']+256
    inf=(z>cam0['near'])&(x>=0)&(x<512)&(y>=0)&(y<512)
    xi,yi=np.clip(x.astype(int),0,511),np.clip(y.astype(int),0,511)
    vis=inf&(z<=depth[yi,xi]+.25)
    if not vis.any():return None
    shot=req['shots'][0];foot=float(z[vis].max())*shot['filmback_mm'][0]/shot['focal_length_mm']/req['resolution'][0]
    ext=args.view_clip_margin_m+args.view_clip_margin_px*foot
    sig=chart.locate(points if shift is None else np.asarray(points,float)-shift)[0][:,0];lo,hi=sig[vis].min()-ext,sig[vis].max()+ext
    seg=lambda sg:int(segment_ids[min(int(np.argmin(np.abs(sig-sg))),len(segment_ids)-2)])
    planes={}
    if lo>sig.min():planes['view_lo']=(float(lo),'ge',seg(lo))
    if hi<sig.max():planes['view_hi']=(float(hi),'le',seg(hi))
    return dict(rule='r2 (occlusion-tested)',planes=planes,visible_stations=int(vis.sum()),in_frame_stations=int(inf.sum()),stations=int(len(vis)),
                extension_m=float(ext),sigma_path=[float(sig.min()),float(sig.max())],sigma_kept=[float(max(lo,sig.min())),float(min(hi,sig.max()))])


def closure_frame(item,source_dir):
    """Closure fallback: re-frame the closure node of a closed path like an interior node
    (frgb.network_profiles.closed_path_closure_frame; transverse = the node's raw interface-mesh normal by
    prepare.normals_on, as prepare.py), then interpolate inward / normal to the stations exactly as main() does."""
    from frgb.network_profiles import closed_path_closure_frame
    from prepare import normals_on
    row,arrays,refined=item['row'],item['arrays'],item['refined']
    k=next(i for i,x in enumerate(read(source_dir/'package/interfaces.json')) if x['id']==row['interface_id'])
    with np.load(source_dir/'package/interfaces.npz') as ia:
        raw,_=normals_on(arrays['points'][[0]],ia[f'vertices_{k}'],ia[f'faces_{k}'])
    nodes_in,nodes_n,record=closed_path_closure_frame(arrays['points'],arrays['inward'],arrays['normal'],arrays['outward'],raw[0])
    si=refined['segment_index']; sf=refined['source_fraction'][:,None]
    inward=(1-sf)*nodes_in[si]+sf*nodes_in[si+1]
    normal=(1-sf)*nodes_n[si]+sf*nodes_n[si+1]
    inward/=np.linalg.norm(inward,axis=1)[:,None]
    normal-=np.sum(normal*inward,axis=1)[:,None]*inward
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    print(json.dumps(dict(event='closure_frame_fallback',index=row['index'],**record)),flush=True)
    item.update(inward=inward,normal=normal,closure_frame_fallback=record)


def overlap_blend_end_rho(chart,pieces,regular,kept,relief,tip,ends_rec,group_inputs,tol=1e-12,n_lambda=41,n_a=21):
    """The checks ran with rho_q, a lazy closure over the group's piece list while it was being extended and before any flag:
    for an end of the j-th path, pieces[:j] (extended) + regular[j:] with every flag None, which is rebuilt here
    (rho_q itself now sees the final list). The build uses the final, flagged pieces. The largest |difference| must be
    <= tol (m), else ValueError (fail closed). Recorded in the end's record ('overlap_blend_rho'): that difference and
    its two parts, C1's (the final pieces flagged against the same pieces unflagged) and the extension's (the
    check-time pieces against the final pieces unflagged). No relief: recorded, nothing checked."""
    if relief is None:
        for r in ends_rec:r['overlap_blend_rho']=dict(relief=None)
        return
    order=[it['row']['index'] for it in kept];objs=list({id(p_):p_ for p_ in list(pieces)+list(regular)}.values())
    flags=[(p_,p_.overlap_blend) for p_ in objs]
    def rho(group,q,flagged):
        for p_ in objs:p_.overlap_blend=None
        try:
            if flagged:
                for p_,fl in flags:p_.overlap_blend=fl
            return np.asarray(fault_charts.relief_on_chart(chart,relief,tip,None,None,pieces=list(group))(q),float)
        finally:
            for p_,fl in flags:p_.overlap_blend=fl
    for r,g in zip(ends_rec,group_inputs):
        j=order.index(r['path']);s_=g['strip']
        lam,a=np.meshgrid(np.linspace(0.,s_['L'],n_lambda),np.linspace(-s_['margin'],s_['depth'],n_a))
        q=s_['p_end']+np.outer(lam.ravel(),s_['t2'])+np.outer(a.ravel(),s_['nu2'])
        checked=rho(list(pieces[:j])+list(regular[j:]),q,False);plain=rho(pieces,q,False);flagged=rho(pieces,q,True)
        rec=dict(points=int(len(q)),max_difference_m=float(np.max(np.abs(checked-flagged))),c1_part_m=float(np.max(np.abs(plain-flagged))),
                 extension_part_m=float(np.max(np.abs(checked-plain))),tol_m=tol)
        r['overlap_blend_rho']=rec
        if rec['max_difference_m']>tol:
            raise ValueError('overlap blend (sec. 11.23): the relief on fault %s end %d of path %d differs from the one its checks used'
                             ' (%.3e m > %.0e m; C1 part %.3e m, extension part %.3e m) (fail closed)'
                             %(r['fault'],r['end'],r['path'],rec['max_difference_m'],tol,rec['c1_part_m'],rec['extension_part_m']))


def build_panels(rows,network,output,args,frames=None):
    """R3: one panel cavity per source surface (all its selected paths together); per-path provenance files. Fault slip: one
    cavity per (source surface, block). A moved group is built in its source frame (q = p - t: PathPieces, the
    end-face asserts and probes, the view-clip sigma; the world-frame fields rho and fine_of evaluated at P + t), each
    fault end is extended by L along the trace (fault_charts.end_extension / extend_path), and
    fault_charts.replay_cavity translates the cavity and trims it to its own block. Without fault events every step
    below is the unchanged one (same arrays, same calls)."""
    from frgb.network_panels import PathPieces,SurfaceChart,build_surface_cavity,monte_carlo_volume,overlap_gate,OVERLAP_BLEND
    frames=frames or dict(status='no_faults',faults=[])
    FAULT_STRUCTURE.clear()
    meta=network.network_metadata
    zref=.5*(network.domain_z_m[0]+network.domain_z_m[1])
    groups={}
    for item in rows:
        block=None if item.get('shift') is None else tuple(r['fault'] for r in item['frame_records'] if r['moved'])
        groups.setdefault((item['row']['source_trace_id'],block),[]).append(item)
    solids,paths=[],[]
    for (trace_id,block),items in groups.items():
        trace=network.trace(trace_id)
        chart=SurfaceChart(trace.points_2d_m,zref,meta['extrusion_direction'])
        shift=items[0].get('shift')
        frame=(lambda x:x) if shift is None else (lambda x,s=shift:np.asarray(x,float)-s)
        pieces=[];kept=[]
        for item in items:
            row,arrays,refined,fields=item['row'],item['arrays'],item['refined'],item['fields']
            A=frame(arrays['points'])
            planes={}
            for k,ev in enumerate(row['endpoint_evidence']):
                if ev['footprint_distance_m']<=1e-7 and ev['source_endpoint'] not in planes:
                    g,c=end_edge_halfspace(network,trace,ev['source_endpoint'])
                    assert np.max(A@g-c)<=1e-7,'original path is not behind its source end face'
                    sigma=0. if ev['source_endpoint']==0 else float(chart.sigma[-1])
                    keep='ge' if ev['source_endpoint']==0 else 'le'
                    # The sigma line is that end face's trace on the surface: on it g.x=c, inside it g.x<c.
                    probe=np.array([[sigma,A[0]@chart.dh],[sigma+(.01 if keep=='ge' else -.01),A[0]@chart.dh]])
                    on,inside=chart.surface(probe)@g-c
                    assert abs(on)<=1e-9 and inside<0,'sigma half-plane disagrees with the source end face'
                    planes[ev['source_endpoint']]=(sigma,keep,int(arrays['segment_ids'][0 if k==0 else -1]))
            if args.view_clip is not None:
                clip=view_clip_planes(chart,refined['points'],refined['original_segment_id'],args,shift)
                item['view_clip']=clip
                if clip is None:
                    print(json.dumps(dict(event='view_clip_drop',index=row['index'])),flush=True);continue
                for key,plane in clip['planes'].items():planes[key]=plane

            def make(ends=(),item=item,row=row,refined=refined,fields=fields,planes=planes):
                pts,seg=frame(refined['points']),refined['original_segment_id'][:-1]
                per=dict(inward=item['inward'],normal=item['normal'],width=item['width'],depth=item['depth'],
                         common=fields['common_displacement_m'],differential=fields['differential_displacement_m'])
                for end,point in ends:pts,per,seg=fault_charts.extend_path(pts,per,seg,end,point)
                return PathPieces(chart,pts,per['inward'],per['normal'],seg,per['width'],per['depth'],per['common'],per['differential'],
                    fields['depth_fraction'],.12,row['index'],md.Manifold.reserve_ids(1),list(planes.values()))
            try:
                piece=make()
            except ValueError as error:
                # its closure node framed like an interior node; every path that built before never reaches this branch
                if str(error)!=CLOSURE_INWARD_ERROR or not np.array_equal(arrays['points'][0],arrays['points'][-1]):raise
                closure_frame(item,Path(args.source or SOURCE))
                piece=make()
            item['make']=make
            pieces.append(piece);kept.append(item)
        # merge too; the regenerated source gave a tip whose refine_to_length wall split missed the exact twin by rounding.
        # footprint mesh split along the chart folds; a cavity that passes unsplit is unchanged (byte-identical geometry).
        if not pieces:continue
        relief=fault_charts.world_field(wall_relief_of(args.wall_relief),shift)
        ends_rec=[];group_inputs=[]
        if frames['status']=='ok':
            rho_q=fault_charts.relief_on_chart(chart,relief,args.wall_relief_tip_m,np.concatenate([p.p[:,0] for p in pieces]),
                                               np.concatenate([(p.oR-p.oL)[:,0] for p in pieces]),pieces=pieces)   # taper_r2
            regular=list(pieces)
            for j,(item,piece) in enumerate(zip(kept,pieces)):
                add=[];trim_j=[]
                for f,k,moved in fault_charts.fault_ends(frames,item['frame_records'],len(item['refined']['points'])):
                    if item['row']['free_ends'][0 if k==0 else 1]:
                        raise ValueError('path %d end %d lies on fault %s but is classified as a free end (fail closed)'%(item['row']['index'],k,f['id']))
                    i=0 if k==0 else -1;t2=-piece.t[0] if k==0 else piece.t[-1];nu2=piece.nu[0] if k==0 else piece.nu[-1]
                    e=fault_charts.end_extension(chart,f,moved,piece.p[i],t2,nu2,float(piece.depth_last[i]),piece.margin,piece.oL[i],piece.oR[i],
                                                 piece.fr,rho_q=rho_q,defer_chart=True)
                    ends_rec.append(dict(path=item['row']['index'],end=k,fault=f['id'],moved=moved,L_m=e['L_m'],max_c_m=e['max_c_m'],at=e['at'],
                                         delta_m=e['delta_m']))
                    group_inputs.append(fault_charts.check_inputs(chart,piece,k,e['L_m'],rho_q,shift,f,moved,ends_rec[-1],
                                                                     joint=frames['joint_triangles'].get(item['row']['interface_id']),
                                                                     chart_overrun=e.get('chart_overrun')))
                    pl=group_inputs[-1].get('joint_edge_plan')
                    add.append((k,chart.surface((piece.p[i]+(e['L_m'] if pl is None else pl.get('L_build_m',e['L_m']))*t2)[None])[0]))
                    if pl is not None:trim_j.append(pl)
                if add:pieces[j]=item['make'](tuple(add))
                for pl in trim_j:
                    pl['owner']=(pieces[j].original_id,np.asarray(item['refined']['points'],float)-(0. if shift is None else np.asarray(shift,float)),
                                 np.asarray(item['refined']['original_segment_id']))
            if group_inputs:
                classify=fault_charts.mouth_classifier(chart,regular,[g['strip'] for g in group_inputs],shift)
                for n_,g in enumerate(group_inputs):g.update(classify=classify,strip_index=n_)
        trims=[g['joint_edge_plan'] for g in group_inputs if g.get('joint_edge_plan') is not None]
        # cross-checks the two); G false: nothing set, nothing recorded.
        # then, on this branch only, each end's relief is checked against the one its checks used (overlap_blend_end_rho, fail closed)
        gate,gate_rec=overlap_gate(pieces);blend=None
        if gate:
            blend=dict(OVERLAP_BLEND)
            for p_ in pieces:p_.overlap_blend=blend
            if ends_rec or trims:overlap_blend_end_rho(chart,pieces,regular,kept,relief,args.wall_relief_tip_m,ends_rec,group_inputs)
        # keeps the same solid. The module's bytes are pinned here, so the label check's md5 of this file covers them.
        here_=Path(__file__).resolve().parent;assert __import__('hashlib').sha256((here_/'floor_relief.py').read_bytes()).hexdigest()=='be4a826b0ab6e5e811e1f5422fd45bf68afd3b8b5a24ee18cfc93ad6e1f14d91','floor_relief.py is not the deployed module'
        sys.path.insert(0,str(here_));import floor_relief
        solid,tables,checks=build_surface_cavity(chart,pieces,h=args.panel_refine_m,tip_merge_tol=1e-7,fold_split='fallback',
                                                 wall_relief=relief,
                                                 end_trim=floor_relief.hook(None if not trims else fault_charts.make_end_trim(trims),relief,args.panel_refine_m,None if relief is None else Path(args.source or SOURCE),shift),
                                                 **({} if args.wall_relief_tip_m is None else dict(wall_relief_tip_m=args.wall_relief_tip_m)),
                                                 **({} if args.wall_mouth_band is None else dict(wall_refine=(fault_charts.world_field(mouth_band_of(args.wall_mouth_band),shift),args.wall_mouth_fine_m))))
        rec_=checks.pop('end_trim');checks.update(floor_relief=rec_['floor_relief'],**({} if rec_['trim'] is None else dict(end_trim=rec_['trim'])))  # the hook's two records apart
        if blend is not None:checks['overlap_blend']=dict(gate=True,**blend,**gate_rec)
        if frames['status']=='ok':
            owners=[(p.original_id,item['refined']['points'],item['refined']['original_segment_id']) for item,p in zip(kept,pieces)]
            before=solid
            solid,tables,replay=fault_charts.replay_cavity(solid,tables,frames,items[0]['row']['interface_id'],block,owners)
            if replay is not None and getattr(args,'save_fault_cavities',None) is not None:
                tag='%s_%s'%(trace_id,'-'.join(block or ('reference',)))
                for kind,m_ in (('source',before),('world',solid)):fault_charts.save_cavity(args.save_fault_cavities/('%s_%s.npz'%(tag,kind)),m_,tables)
            if replay is not None or ends_rec:
                checks['fault']=dict(block=list(block or ()),shift_m=None if shift is None else [float(x) for x in shift],ends=ends_rec,replay=replay)
        if args.monte_carlo:
            estimate,stderr,samples=monte_carlo_volume(pieces)
            checks['monte_carlo']=dict(volume_m3=estimate,stderr_m3=stderr,samples=samples,
                relative_difference=abs(solid.volume()-estimate)/estimate)
        solids.append(solid)
        if group_inputs:FAULT_STRUCTURE.append((len(solids)-1,group_inputs))
        for item,path in zip(kept,pieces):
            row,refined,fields=item['row'],item['refined'],item['fields']
            table=tables[path.original_id]
            save_sources(output,row['index'],dict(original_id=path.original_id,face_sources=table,
                junction_bridges=[],construction_checks=dict(checks,source_trace_id=trace_id,surface_paths=[i['row']['index'] for i in items],
                **({} if 'closure_frame_fallback' not in item else dict(closure_frame_fallback=item['closure_frame_fallback'])))),
                args.sources_format,save)
            np.savez_compressed(output/f'path_{row["index"]:02d}_stations.npz',**refined,
                inward=item['inward'],normal=item['normal'],width=item['width'],depth=item['depth'],
                common=fields['common_displacement_m'],differential=fields['differential_displacement_m'])
            paths.append(dict(**row,original_id=path.original_id,faces=len(table),surface_volume_m3=solid.volume(),
                refined_segments=len(refined['points'])-1,continuous_to_discrete_factor=fields['continuous_to_discrete_factor'],
                actual_pl_mean_opening_m=fields['actual_pl_mean_opening_m'],construction_checks=checks,
                **({} if 'view_clip' not in item else dict(view_clip=item['view_clip'])),
                **({} if 'closure_frame_fallback' not in item else dict(closure_frame_fallback=item['closure_frame_fallback']))))
        print(json.dumps(dict(event='surface_complete',trace=trace_id,paths=[i['row']['index'] for i in items],
            faces=checks['cavity_triangles'],volume_m3=solid.volume(),monte_carlo=checks.get('monte_carlo'))),flush=True)
    paths.sort(key=lambda p:p['index'])
    return solids,paths


if __name__=='__main__':main()
