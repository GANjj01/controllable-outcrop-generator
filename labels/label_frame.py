"""python labels/label_frame.py --case CASE_DIR --request REQUEST_JSON --frame FRAME_NPZ --png FRAME_PNG --out OUT
[--source SOURCE_DIR] [--masks MASK_NPY] [--cpu-raster NPZ] [--keep-scratch]

Inputs --case     the repaired cut case (net/cases/<name>; its sibling <name>_raw, the build.py of net/, build.json)
--request  {shots: [shot], resolution: [w, h]} of this frame; also the view-clip request the case was built with
--frame    render_point's TAG.npz: depth (m, view axis), mask (blender_client.region_mask: 0 sky, 1 rock, 2 ground,
255 undetermined) --png      the 8-bit sRGB frame (U uses its Rec.709 luma) --source   optional; must equal build.json
'source' (the uncut host) --masks    optional mask .npy overriding the frame's mask --cpu-raster optional npz (face,
depth) of the cut native mesh at this camera; computed with rasterize_native otherwise Steps: rebuild the per-fracture
cut solids with the case's own build.py, verify (volumes, construction checks, stations, difference mesh bitwise vs
<name>_raw), export the mid-surface; (B) project every station line onto the flattened host (all branches); refine
every label point onto the exact mid-surface; (D) visibility, exemption, gates G1 / G2, U, scene masks (ground, sky,
undetermined and rock within 3 px of ground are not scored), labels. Outputs in OUT: target.npy, valid.npy,
u_line.npy, ignore_reason.npy, occluded_line.npy, windows.json (rows with gid), band_win.npy, ring_win.npy (int32, gid
or -1; first window where a pixel serves several), window_pixels.npz (band, ring: all (linear index, gid) memberships;
reproduces every window contrast), report.json (gates, records, timings). Exit codes: 0 all gates pass; 3 a gate stops
(exemption > 1%, G1, G2; report.json 'stop'); 4 the cavity rebuild does not verify; 2 bad input."""
import argparse
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(P1)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P1 / 'frgb' / 'chain'))
import run_frames_r2 as R2  # noqa: E402
import run_frames_r3 as R3  # noqa: E402
from frgb import trace_labels as tl  # noqa: E402
from frgb.metrics import graph_node_weights, skeletonize  # noqa: E402
from frgb.network_training_labels import _projected_boxes, rasterize_native  # noqa: E402
from frgb.raster import camera  # noqa: E402
from procmem import machine_available_gb, process_gb  # noqa: E402
import fault_support as FS  # noqa: E402
import occlusion_rule as OR  # noqa: E402

from frgb import runconfig as frgb_rc  # noqa: E402
PHASE0 = frgb_rc.path('env.geometry_dir').as_posix()
P0_PYTHONPATH = ';'.join([str(P1), frgb_rc.path('outcrop_src').as_posix(), frgb_rc.path('manifold_site').as_posix()])


def phase0(args, log):
    env = dict(os.environ, PYTHONPATH=P0_PYTHONPATH,
               PATH=os.pathsep.join([PHASE0, PHASE0 + '/Library/bin', os.environ.get('PATH', '')]))
    with open(log, 'w', encoding='utf8') as fh:
        return subprocess.run([PHASE0 + '/python.exe', str(HERE / 'rebuild_cavities.py')] + [str(a) for a in args],
                              stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def phase0_start(args, log):
    env = dict(os.environ, PYTHONPATH=P0_PYTHONPATH,
               PATH=os.pathsep.join([PHASE0, PHASE0 + '/Library/bin', os.environ.get('PATH', '')]))
    fh = open(log, 'w', encoding='utf8')
    return subprocess.Popen([PHASE0 + '/python.exe', str(HERE / 'rebuild_cavities.py')] + [str(a) for a in args],
                            stdout=fh, stderr=subprocess.STDOUT, env=env), fh


def _min_available(stop, box):
    """speed-up s2 record: the machine's minimum available memory (GB) while stage B runs"""
    while not stop.is_set():
        box[0] = min(box[0], machine_available_gb()[0])
        stop.wait(.5)


def aligned_mask(cpu, depth_m, rock):
    hit = cpu['face'] >= 0
    both = hit & rock
    residual = np.full(hit.shape, np.inf)
    residual[both] = np.abs(depth_m[both] - cpu['depth'][both])
    return both & np.isfinite(depth_m) & (residual <= np.maximum(.01, .002 * cpu['depth']))


def fault_frame_records(case, samples, nat, cpu, labels, cam, frames, FC, groups, rock):
    """Record only, faulted scenes: fault_cap faces and visible pixels; visible joint label length whose first hit is a
    fault_cap face; per side, visible cavity-face pixels in the near-fault distance band with no joint label within 2
    px."""
    from scipy import ndimage as ndi
    roles = nat['roles']
    cap_faces = roles == 'fault_cap'
    face = cpu['face']
    hit = face >= 0
    vis_cap = np.zeros(face.shape, bool); vis_cap[hit] = cap_faces[face[hit]]
    joint_on_cap = 0.
    for r in samples.values():
        sel = r['visible'] & (r['hit_face'] >= 0)
        if sel.any():
            w = tl._sample_weights(r)
            joint_on_cap += float(w[sel][cap_faces[r['hit_face'][sel]]].sum())
    rec = dict(fault_cap=dict(faces=int(cap_faces.sum()), visible_px=int((vis_cap & rock).sum()),
                              visible_joint_label_px_first_hit_fault_cap=joint_on_cap,
                              fault_label_px_first_hit_fault_cap='not recorded here (fault labels are merged by the r4 entry)'))
    cav = np.zeros(face.shape, bool); cav[hit] = nat['path_of_face'][face[hit]] >= 0
    cav &= rock
    far = ndi.distance_transform_edt(~labels['target']) > 2
    vv, uu = np.nonzero(cav & far)
    seam = {}
    if len(vv):
        P = cam['pos'] + cpu['depth'][vv, uu][:, None] * tl.rays(cam, uu, vv)
        pth = nat['path_of_face'][face[vv, uu]]
        foot = cpu['depth'][vv, uu] / cam['fpx']
        iface = {i: p['row']['interface_id'] for i, p in case['paths'].items()}
        for f in frames['faults']:
            for name, flag in (('moving', True), ('fixed', False)):
                on = np.array([int(p) in groups and iface[int(p)] in f['moved'] and
                               (bool(groups[int(p)]['block']) and f['id'] in groups[int(p)]['block']) == flag for p in pth], bool)
                if not on.any():
                    seam['%s:%s' % (f['id'], name)] = 0
                    continue
                d = np.abs(FC.signed_other(f, flag, P[on]))
                seam['%s:%s' % (f['id'], name)] = int((d < np.maximum(.02, 3. * foot[on])).sum())
    rec['seam_px_near_fault_without_joint_label'] = seam
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=Path, required=True)
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--frame', type=Path, required=True)
    ap.add_argument('--png', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--masks', type=Path)
    ap.add_argument('--cpu-raster', type=Path)
    ap.add_argument('--keep-scratch', action='store_true')
    args = ap.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    timing, t0 = {}, time.time()
    report = dict(rule='trace_labels r3 (+ amendments a, b, c, d; T1; speed-ups s1, s2)', case=str(args.case), request=str(args.request),
                  frame=str(args.frame), png=str(args.png), status='TRACE_LABEL_CANDIDATE_NOT_QUALIFIED')

    def finish(code, **extra):
        report.update(extra, timing_s=timing, exit_code=code)
        (out / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=float) + '\n', encoding='utf8')
        sys.exit(code)
    req = json.loads(args.request.read_text(encoding='utf8'))
    cam = camera(req['shots'][0], req['resolution'])
    try:
        case = tl.load_case(args.case, known_roles=FS.KNOWN_ROLES)
    except ValueError as e:
        if 'face roles outside the known set' not in str(e):
            raise
        finish(3, error=str(e), stop=dict(unknown_face_roles=True))
    if args.source is not None and Path(args.source).resolve() != Path(case['build']['source']).resolve():
        finish(2, error='--source differs from build.json source: %s' % case['build']['source'])
    cav = out / 'cavities'
    refine_in = out / 'refine_in.npz'
    signal = {s: refine_in.with_name(refine_in.name + s) for s in ('.go', '.stop', '.ready.json')}
    for f in signal.values():
        f.unlink(missing_ok=True)
    proc, fh = phase0_start(['--case-dir', args.case, '--request', args.request, '--dest', cav, '--scratch', out / 'scratch',
                             '--clean', '--midsurface', '--serve-refine', refine_in], out / 'stage_a.log')
    try:
        while not signal['.ready.json'].exists() and proc.poll() is None:
            time.sleep(.2)
        timing['a_rebuild_export'] = time.time() - t0
        if not signal['.ready.json'].exists():
            rc_a = proc.wait()
            mf = cav / 'manifest_current.json'
            fault_rec = json.loads(mf.read_text(encoding='utf8')).get('fault') if mf.is_file() else None
            man_a = json.loads(mf.read_text(encoding='utf8')) if mf.is_file() else {}
            if (man_a.get('network_panels') or {}).get('ok') is False:
                finish(3, error='live frgb/network_panels.py differs from the pinned one (stage A F1)', fault=fault_rec,
                       stop=dict(network_panels_pin=True))
            if (man_a.get('wall_fidelity') or {}).get('ok') is False:
                finish(3, error='captured cavity walls differ from the label reference (stage A F2)', fault=fault_rec,
                       stop=dict(wall_fidelity=True))
            if rc_a == 6:
                finish(3, status='unsupported', error='fault frames unsupported (stage_a.log)', fault=fault_rec,
                       stop=dict(fault_unsupported=True))
            finish(4, error='cavity rebuild did not verify (stage_a.log, cavities/manifest_current.json)', fault=fault_rec)
        served = json.loads(signal['.ready.json'].read_text(encoding='utf8'))
        fault_rec = json.loads((cav / 'manifest_current.json').read_text(encoding='utf8')).get('fault')
        faulted = fault_rec is not None and fault_rec.get('status') == 'ok'
        if faulted:
            mids = json.loads((cav / 'manifest_current.json').read_text(encoding='utf8')).get('midsurface') or {}
            clips = {t_: r_['clip'] for t_, r_ in mids.items() if isinstance(r_, dict) and r_.get('clip')}
            nz = [dict(trace=t_, end=k_, **v_) for t_, c_ in clips.items() for k_, v_ in c_['per_end'].items()
                  if v_['rejected_not_converged'] or v_['rejected_not_covered']]
            report['clip'] = dict(per_trace=clips, clip_rejected_nonzero=nz)
        if faulted:
            groups = FS.path_groups(case['build'])
            case['cutter_key'] = {i: g['key'] for i, g in groups.items()}
            case['collect_g1'] = True
            FCx = FS.load_fault_charts()
            case['ext_trim'] = FS.ext_trim_fn(case, FCx.fault_frames(case['build']['source']), FCx)
        # (B) projection
        t = time.time()
        stop_watch, low = threading.Event(), [machine_available_gb()[0]]
        watch = threading.Thread(target=_min_available, args=(stop_watch, low), daemon=True); watch.start()
        vol = R2.r1.signed_volume(case['host']['vertices'].astype(float), case['host']['faces'])
        case['front_sign'] = 1. if vol > 0 else -1.
        samples, unresolved, info, subs, folds = R3.r3_samples(case, cam, cav)
        if faulted:
            for i in [i for i in case['paths'] if FS.is_pseudo(i)]:
                groups[i] = groups[FS.base_of(i)[0]]
            report['extension'] = {str(i): dict(base=FS.base_of(i)[0], end=FS.base_of(i)[1], info=info.get(i),
                                                source=case['paths'][i]['ext_meta']['source'], L_m=case['paths'][i]['ext_meta']['L'],
                                                samples=int(sum(len(s['u']) for k, s in samples.items() if k[0] == i)))
                                   for i in case['paths'] if FS.is_pseudo(i)}
        order = sorted(samples)
        trace_of = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
        np.savez(refine_in, point=np.concatenate([samples[k]['point'] for k in order]) if order else np.empty((0, 3)),
                 face=np.concatenate([samples[k]['host_face'] for k in order]) if order else np.empty(0, np.int64),
                 trace=np.concatenate([np.full(len(samples[k]['u']), int(trace_of[samples[k]['path']].split('-')[1])) for k in order])
                 if order else np.empty(0, np.int64),
                 cap=R3.refine_caps(samples, order) if order else np.empty(0),
                 **({} if not faulted else dict(path=np.concatenate([np.full(len(samples[k]['u']), FS.base_of(samples[k]['path'])[0], np.int64)
                                                                     for k in order]) if order else np.empty(0, np.int64))))
        stop_watch.set(); watch.join()
        memory = dict(phase0_retained=served['memory']['retained'], cnn_after_b=process_gb(), machine_min_available_during_b_gb=low[0],
                      machine_total_gb=machine_available_gb()[1])
        timing['b_project'] = time.time() - t
        # (C) refinement, in stage A's process on its verified context
        t = time.time()
        signal['.go' if order else '.stop'].write_text('\n', encoding='utf8')
        rc = proc.wait()
        if rc != 0:
            finish(4, error='refinement on the stage-A context failed (stage_a.log)')
        if order:
            refinement = R3.apply_refinement(samples, order, out / 'refine_in_refined.npz')
            case['refined_npz'] = out / 'refine_in_refined.npz'
            jump = R3.jump_check(samples, order, cam)
            memory['phase0_after_refine'] = json.loads((cav / 'manifest_refine.json').read_text(encoding='utf8'))['memory']['after_refine']
        else:
            refinement = jump = None
        timing['c_refine'] = time.time() - t
    finally:
        if proc.poll() is None:
            proc.kill()
        fh.close()
    report['speedup_s2_memory'] = memory
    # (D) visibility, gates, labels
    t = time.time()
    grid = tl.HostGrid(case['host']['vertices'], case['host']['faces'])
    case['host_normals'] = grid.normals
    del grid
    cutters, manifest = R2.load_cutters(cav, cam)
    nat = case['native']
    boxes = _projected_boxes(nat['vertices'], nat['faces'], cam)[:3]
    host_boxes = _projected_boxes(case['host']['vertices'], case['host']['faces'], cam)[:3]
    tl.classify_visibility(case, samples, cam, boxes=boxes)
    ex = tl.exemption_check(case, samples, cam, cutters, host_boxes=host_boxes)
    if faulted and any(FS.is_pseudo(k[0]) for k in samples):
        ext_slot = FS.extension_slot_check(case, samples)
        ext_cover = FS.exposure_crosscheck(case, samples, cam, tl.project, tl.in_frame)
        report['extension_checks'] = dict(slot_opening=ext_slot, exposure=ext_cover)
        if ext_slot['not_at_slot_opening']:
            finish(3, error='visible extension samples not at the slot opening', stop=dict(extension_not_at_slot_opening=True))
    elif faulted:
        report['extension_checks'] = dict(slot_opening=None, exposure=FS.exposure_crosscheck(case, samples, cam, tl.project, tl.in_frame))
    case['collect_mismatch'] = True
    g = R3.gate1(case, samples, cam, cutters, folds)
    mm_samples = g.pop('_mismatch')
    if faulted:
        import fault_sides as FSD
        FC = FS.load_fault_charts()
        frames = FC.fault_frames(case['build']['source'])
        if frames['status'] != 'ok':
            finish(3, status='unsupported', error='module status %s in stage D' % frames['status'], stop=dict(fault_unsupported=True))
        side_rec, side_arrays = FSD.side_check(g.pop('_samples'), groups, frames, FC, case, samples)
    timing['d_visibility_gates'] = time.time() - t
    t = time.time()
    with np.load(args.frame) as a:
        depth_m = np.asarray(a['depth'], float)
        mask = np.asarray(a['mask']) if 'mask' in a.files else None
    if args.masks is not None:
        mask = np.load(args.masks)
    if mask is None:
        finish(2, error='no mask: the frame npz has none and --masks is not given')
    shape = (req['resolution'][1], req['resolution'][0])
    if mask.shape != shape or depth_m.shape != shape:
        finish(2, error='frame / mask shape %s / %s differs from the request %s' % (depth_m.shape, mask.shape, shape))
    rgb = np.asarray(Image.open(args.png).convert('RGB'))
    if args.cpu_raster is not None:
        with np.load(args.cpu_raster) as a:
            cpu = dict(face=a['face'], depth=a['depth'])
    else:
        cpu = rasterize_native(nat['vertices'], nat['faces'], cam)
    timing['d_cpu_raster'] = time.time() - t
    t = time.time()
    rock = mask == 1
    al = aligned_mask(cpu, depth_m, rock)
    try:
        scal = OR.scene_scalars(case['build']['source'])
    except ValueError as e:
        finish(3, error=str(e), stop=dict(occlusion_scalars_missing=True))
    owner = case.get('cutter_key') or {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    shift_of = {i: g['shift'] for i, g in groups.items()} if faulted else {}
    region_j, occ_rec = OR.joint_regions(samples, cam, cpu['depth'], (cpu['face'] >= 0) & rock, tl.rays, scal, case['charts'], owner,
                                         shift_of)
    case['occlusion'] = dict(scalars=scal, rock=(cpu['face'] >= 0) & rock)
    import mismatch as MM
    mm_band, mm_rec = MM.mismatch(mm_samples, MM.load_coverage(case.get('refined_npz')), cam, shape, tl.project)
    opening_base = {i: (FS.base_of(i)[0] if FS.is_pseudo(i) else i) for i in case['paths']}
    labels = tl.build_trace_labels(case, samples, cam, rgb, cpu_raster=cpu, render_rock=rock, aligned=al,
                                   ground=mask == 2, sky=(mask == 0) | (mask == 255), unresolved_points=unresolved,
                                   export_windows=True, occlusion_regions=[(tl.IGNORE_OWN_GROOVE, region_j)],
                                   mismatch_region=mm_band, opening_base=opening_base)
    mm_rec['bit128_positive_px'] = labels['mismatch_removed_positive_px']
    report['mismatch'] = mm_rec
    report['opening'] = dict(rule='criterion_trace_labels_mismatch_opening_r1 (793c9155) O6', rock_px=int(labels['rock'].sum()),
                             class1_px=int((labels['opening_class'] == 1).sum()), class2_px=int((labels['opening_class'] == 2).sum()),
                             by_path=labels['opening_by_path'])
    if faulted:
        ic = FS.indep_check(case, labels, cam, tl.first_hits, tl.project, tl.in_frame, tl.EPS_FLOOR_M, tl.EPS_PIXEL_FRACTION)
        report.setdefault('extension_checks', {})['independent'] = ic
        if ic['uncovered']:
            finish(3, error='visible extension trace not covered by the labels', stop=dict(extension_trace_uncovered=True))
        if ic['mismatch']:
            finish(3, error='extension trace outside the r3 exposure record', stop=dict(extension_record_mismatch=True))
        if ic['missing']:
            finish(3, error='extension trace without an r3 exposure record', stop=dict(extension_record_missing=True))
    for k in ('target', 'valid', 'ignore_reason', 'u_line', 'occluded_line', 'band_win', 'ring_win'):
        np.save(out / (k + '.npy'), labels[k])
    np.save(out / 'opening_class.npy', labels['opening_class'])     # mismatch_opening_r1 O1
    if faulted:
        ext = {k: r for k, r in samples.items() if FS.is_pseudo(k[0])}
        emask = np.zeros(labels['target'].shape, bool)
        for r in ext.values():
            if 'uv' in r and len(r['uv']):
                emask |= OR._disc_band(np.asarray(r['uv'])[np.asarray(r['in_frame'], bool)], emask.shape, radius=FS.EXT_MASK_PX)
        if ext:
            emask |= OR.joint_regions(ext, cam, cpu['depth'], (cpu['face'] >= 0) & rock, tl.rays, scal, case['charts'], owner, shift_of)[0]
        np.save(out / 'extension_mask.npy', emask)
    np.savez_compressed(out / 'window_pixels.npz', band=labels['window_pixels']['band_pixels'],
                        ring=labels['window_pixels']['ring_pixels'])
    (out / 'windows.json').write_text(json.dumps(labels['windows'], allow_nan=False) + '\n', encoding='utf8')
    timing['d_labels'] = time.time() - t
    wins = labels['windows']
    rock_px = int(labels['rock'].sum())
    stop = dict(exemption=bool(ex['stop_triggered']), g1_fail=not g['g1_pass'], g2=bool(g['g2_stop']),
                **({'g1_excluded_off_midsurface': True} if g.get('g1r1_off_midsurface') else {}))
    d_own = np.concatenate([r['occluder_distance_m'][r['own_occluded']] for r in samples.values()]) if samples else np.empty(0)
    d_far = np.concatenate([r['occluder_distance_m'][r['far_occluded']] for r in samples.values()]) if samples else np.empty(0)
    qd = lambda d: [float(x) for x in np.quantile(d, [.5, .99, 1.])] if len(d) else None
    report['occlusion'] = dict(rule='criterion_trace_labels_occlusion_r1 (c037e180)', scalars=scal, joint=dict(
        occ_rec, own_occluded_px=float(sum(tl._sample_weights(r)[r['own_occluded']].sum() for r in samples.values())),
        far_occluded_px=float(sum(tl._sample_weights(r)[r['far_occluded']].sum() for r in samples.values())),
        own_d_m_p50_p99_max=qd(d_own), far_d_m_p50_p99_max=qd(d_far),
        bit32_px=int(((labels['ignore_reason'] & tl.IGNORE_OWN_GROOVE) > 0).sum()),
        bit32_share_of_rock=float(((labels['ignore_reason'] & tl.IGNORE_OWN_GROOVE) > 0)[labels['rock']].sum() / max(int(labels['rock'].sum()), 1))))
    if faulted:
        stop['fault_closure_b'] = bool(side_rec['closure_b']['violated'])
        report['fault_sides'] = FSD.write(out, side_rec, side_arrays, fault_frame_records(case, samples, nat, cpu, labels, cam, frames, FC,
                                                                                         groups, rock))
        report['fault'] = fault_rec
    report.update(visible_length_px=R2.weights_sum(samples, 'visible'), exemption=ex, gates=g, refinement=refinement,
                  jump_check=jump,
                  flattening=subs, projection={str(k): {kk: vv for kk, vv in v.items()} for k, v in info.items()},
                  positive_graph_length_px=labels['positive_graph_length_px'], positive_pixels=int(labels['target'].sum()),
                  ignore_fraction_of_rock=float((labels['ignore_reason'] > 0)[labels['rock']].sum() / max(rock_px, 1)),
                  u_windows=dict(total=len(wins), u=sum(w['u'] for w in wins),
                                 ring_below_min=sum(w['u'] and w['ring_pixels'] < tl.U_MIN_RING for w in wins)),
                  u_line_px=float(graph_node_weights(skeletonize(labels['u_line'])).sum()),
                  band_win_multi_pixels=labels['band_win_multi_pixels'], ring_win_multi_pixels=labels['ring_win_multi_pixels'],
                  parallel=tl.parallel_record(samples), technical_failure_all_u=labels['all_visible_u'],
                  per_path=labels['per_path'], stop=stop, cavities_manifest=str(cav / 'manifest_current.json'))
    if not args.keep_scratch:
        for name in ('refine_in.npz',):
            (out / name).unlink(missing_ok=True)
    report['speedup_s2_memory']['cnn_end'] = process_gb()
    finish(3 if any(stop.values()) else 0)


if __name__ == '__main__':
    main()
