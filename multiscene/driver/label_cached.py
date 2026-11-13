"""A copy of labels/label_frame.py main() PINNED to its sha256 below; the only change: --state-out FILE  after step C,
pickle the state that step D reads with the sha256 of every .py of labels and frgb/chain and of every frgb module
loaded by then; --state-in FILE   skip steps A-C and run D on this frame with that state (render-noise repeats: A-C
depend on geometry and camera only; D reads the frame's depth, mask and luma). Refused (exit 5) when any sha256
differs (s4: also when the refined-points file is missing or changed). Exit 5 also when label_frame.py itself differs
from the pinned copy: the caller then runs label_frame.py instead. Exit codes otherwise as label_frame.py (0, 3, 4,
2)."""
import hashlib
import json
import pickle
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent; P1 = HERE.parents[1]; W = P1/'labels'
sys.path.insert(0, str(W))
import label_frame as LF  # noqa: E402  (its imports: run_frames_r2 / r3, frgb.trace_labels, ...)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PINNED = '22431c04ced9c9decbe8844e40bf19f19f27ace41c79d2ee271fa265048c2a8d'
PINNED_R4 = '4cc934ea35595e513c536874f4e2ace819ee0c8712c683e2f62e4ce3f72db3ca'


ROOTS = (W.resolve(), (P1/'frgb'/'chain').resolve(), (P1/'frgb').resolve(), P1.resolve())


def _local(name):
    """a module name -> its file in one of the project roots, or None (standard library / site packages)"""
    parts = name.split('.')
    for r in ROOTS:
        for cand in (r.joinpath(*parts).with_suffix('.py'), r.joinpath(*parts)/'__init__.py'):
            if cand.is_file(): return cand.resolve()
    return None


def static_closure(path, seen=None):
    """project modules a script imports, recursively"""
    import ast
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    if path in seen: return seen
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                ([node.module] + ['%s.%s' % (node.module, a.name) for a in node.names] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
        for n in names:
            f = _local(n)
            if f is not None: static_closure(f, seen)
    return seen


def code_files():
    """The cache is valid only while all are unchanged; a change of any refuses it and the caller falls back"""
    loaded = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)}
    mine = {p for p in loaded if p.parent in ROOTS}
    return sorted(mine | static_closure(W/'rebuild_cavities.py'))


def shas(files=None):
    return {str(Path(p).resolve().relative_to(P1.resolve())).replace('\\', '/'): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (files if files is not None else code_files())}


def main():
    import argparse
    R2, R3, tl, camera, FS, OR = LF.R2, LF.R3, LF.tl, LF.camera, LF.FS, LF.OR
    ap = argparse.ArgumentParser()
    for k in ('--case', '--request', '--frame', '--png', '--out'): ap.add_argument(k, type=Path, required=True)
    for k in ('--source', '--masks', '--cpu-raster', '--state-out', '--state-in'): ap.add_argument(k, type=Path)
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
    lf = hashlib.sha256((W/'label_frame.py').read_bytes()).hexdigest()
    if lf != PINNED:
        finish(5, error='label_frame.py differs from the pinned copy (%s); run label_frame.py instead' % lf)
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
    if FS.faulted_build(case['build']):                          # faulted cases do not use the state cache
        finish(5, error='s6a: faulted case (build.json fault_frames): the cache serves fault-free frames only; run label_frame_r4.py')
    groups = None
    if args.state_in is not None:
        st = pickle.loads(args.state_in.read_bytes())
        try:
            now = shas([P1/k for k in st['shas']])
        except (OSError, ValueError):
            now = None
        if now != st['shas']:
            changed = sorted(k for k in st['shas'] if (now or {}).get(k) != st['shas'][k])
            finish(5, error='state cache built with other code (changed or missing: %s)' % changed[:20])
        if st['case'] != str(Path(args.case).resolve()) or st['request'] != json.loads(args.request.read_text(encoding='utf8')):
            finish(2, error='state cache belongs to another case / request')
        samples, unresolved, info, subs, folds, refinement, jump, cav = (st[k] for k in ('samples', 'unresolved', 'info', 'subs', 'folds',
                                                                                         'refinement', 'jump', 'cavities'))
        faulted, fault_rec = st['faulted'], st['fault_rec']
        cav = Path(cav)
        rp = st['case_added'].get('refined_npz')
        if rp is not None and (not Path(rp).is_file() or hashlib.sha256(Path(rp).read_bytes()).hexdigest() != st['refined_npz_sha256']):
            finish(5, error='state cache: the refined-points file %s is missing or changed' % rp)
        if faulted:                                               # as stage B: samples follow their builder group (trace, block)
            groups = FS.path_groups(case['build'])
            case['cutter_key'] = {i: g['key'] for i, g in groups.items()}
            case['collect_g1'] = True
        vol = R2.r1.signed_volume(case['host']['vertices'].astype(float), case['host']['faces'])
        case['front_sign'] = 1. if vol > 0 else -1.
        case.update(st['case_added'])                             # s3 fix: what stages A-C added to case (charts for the occlusion rule)
        case['paths'].update(st['paths_added'])
        if faulted:
            for i in [i for i in case['paths'] if FS.is_pseudo(i)]:
                groups[i] = groups[FS.base_of(i)[0]]
        report.update(st['report_ac'])                            # s4: the A-C report entries (h clip, g extension)
        timing['state_in'] = time.time() - t0; report['state_in'] = str(args.state_in)
        report['speedup_s2_memory'] = {}                          # D only: cnn_end below is this process's own record
    else:
        keys0 = set(case)                                         # s3 fix: the case keys stages A-C add go into the state
        cav = out / 'cavities'
        refine_in = out / 'refine_in.npz'
        signal = {s: refine_in.with_name(refine_in.name + s) for s in ('.go', '.stop', '.ready.json')}
        for f in signal.values():
            f.unlink(missing_ok=True)
        proc, fh = LF.phase0_start(['--case-dir', args.case, '--request', args.request, '--dest', cav, '--scratch', out / 'scratch',
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
            stop_watch, low = threading.Event(), [LF.machine_available_gb()[0]]
            watch = threading.Thread(target=LF._min_available, args=(stop_watch, low), daemon=True); watch.start()
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
            memory = dict(phase0_retained=served['memory']['retained'], cnn_after_b=LF.process_gb(), machine_min_available_during_b_gb=low[0],
                          machine_total_gb=LF.machine_available_gb()[1])
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
        if args.state_out is not None:
            t = time.time()
            args.state_out.write_bytes(pickle.dumps(dict(samples=samples, unresolved=unresolved, info=info, subs=subs, folds=folds,
                                                         refinement=refinement, jump=jump, cavities=str(cav.resolve()), faulted=faulted,
                                                         fault_rec=fault_rec, case_added={k: case[k] for k in case if k not in keys0 and k != 'ext_trim'},
                                                         paths_added={i: case['paths'][i] for i in case['paths'] if FS.is_pseudo(i)},
                                                         report_ac={k: report[k] for k in ('clip', 'extension') if k in report},
                                                         refined_npz_sha256=hashlib.sha256(Path(case['refined_npz']).read_bytes()).hexdigest()
                                                         if case.get('refined_npz') is not None else None,
                                                         shas=shas(), case=str(Path(args.case).resolve()), request=req),
                                                    protocol=4))
            timing['state_out'] = time.time() - t; report['state_out'] = str(args.state_out)
    # (D) visibility, gates, labels -- label_frame.py (666fc5e6...) main() stage D, unchanged
    t = time.time()
    grid = tl.HostGrid(case['host']['vertices'], case['host']['faces'])
    case['host_normals'] = grid.normals
    del grid
    cutters, manifest = R2.load_cutters(cav, cam)
    nat = case['native']
    boxes = LF._projected_boxes(nat['vertices'], nat['faces'], cam)[:3]
    host_boxes = LF._projected_boxes(case['host']['vertices'], case['host']['faces'], cam)[:3]
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
        cpu = LF.rasterize_native(nat['vertices'], nat['faces'], cam)
    timing['d_cpu_raster'] = time.time() - t
    t = time.time()
    rock = mask == 1
    al = LF.aligned_mask(cpu, depth_m, rock)
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
    qd = lambda d: [float(x) for x in np.quantile(d, [.5, .99, 1.])] if len(d) else None  # noqa: E731
    report['occlusion'] = dict(rule='criterion_trace_labels_occlusion_r1 (c037e180)', scalars=scal, joint=dict(
        occ_rec, own_occluded_px=float(sum(tl._sample_weights(r)[r['own_occluded']].sum() for r in samples.values())),
        far_occluded_px=float(sum(tl._sample_weights(r)[r['far_occluded']].sum() for r in samples.values())),
        own_d_m_p50_p99_max=qd(d_own), far_d_m_p50_p99_max=qd(d_far),
        bit32_px=int(((labels['ignore_reason'] & tl.IGNORE_OWN_GROOVE) > 0).sum()),
        bit32_share_of_rock=float(((labels['ignore_reason'] & tl.IGNORE_OWN_GROOVE) > 0)[labels['rock']].sum() / max(int(labels['rock'].sum()), 1))))
    if faulted:
        stop['fault_closure_b'] = bool(side_rec['closure_b']['violated'])
        report['fault_sides'] = FSD.write(out, side_rec, side_arrays, LF.fault_frame_records(case, samples, nat, cpu, labels, cam, frames, FC,
                                                                                            groups, rock))
        report['fault'] = fault_rec
    report.update(visible_length_px=R2.weights_sum(samples, 'visible'), exemption=ex, gates=g, refinement=refinement,
                  jump_check=jump,
                  flattening=subs, projection={str(k): {kk: vv for kk, vv in v.items()} for k, v in info.items()},
                  positive_graph_length_px=labels['positive_graph_length_px'], positive_pixels=int(labels['target'].sum()),
                  ignore_fraction_of_rock=float((labels['ignore_reason'] > 0)[labels['rock']].sum() / max(rock_px, 1)),
                  u_windows=dict(total=len(wins), u=sum(w['u'] for w in wins),
                                 ring_below_min=sum(w['u'] and w['ring_pixels'] < tl.U_MIN_RING for w in wins)),
                  u_line_px=float(LF.graph_node_weights(LF.skeletonize(labels['u_line'])).sum()),
                  band_win_multi_pixels=labels['band_win_multi_pixels'], ring_win_multi_pixels=labels['ring_win_multi_pixels'],
                  parallel=tl.parallel_record(samples), technical_failure_all_u=labels['all_visible_u'],
                  per_path=labels['per_path'], stop=stop, cavities_manifest=str(cav / 'manifest_current.json'))
    if not args.keep_scratch:
        for name in ('refine_in.npz',):
            (out / name).unlink(missing_ok=True)
    report['speedup_s2_memory']['cnn_end'] = LF.process_gb()
    finish(3 if any(stop.values()) else 0)


def main_r4():
    argv = [a for a in sys.argv if a != '--r4']; sys.argv[:] = argv
    out = Path(argv[argv.index('--out') + 1]) if '--out' in argv else None
    r4 = hashlib.sha256((W/'label_frame_r4.py').read_bytes()).hexdigest()
    if r4 != PINNED_R4:
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out/'report.json').write_text(json.dumps(dict(exit_code=5, error='label_frame_r4.py differs from the pinned copy (%s)' % r4)) + '\n', encoding='utf8')
        sys.exit(5)
    import label_frame_r4 as R4                                  # replaces frgb.trace_labels.build_trace_labels
    try:
        main(); code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    rp = out/'report.json' if out is not None else None
    if code != 5 and rp is not None and rp.is_file():           # label_frame_r4.py __main__, unchanged
        rep = json.loads(rp.read_text(encoding='utf8'))
        rep['rule'] = R4.RULE
        rep['r4_criterion_sha256'] = (R4.HERE/'criterion_trace_labels_r4_faults.sha256.txt').read_text(encoding='utf8').split()[0]
        if 'source_type' in R4.STATE:
            np.save(out/'source_type.npy', R4.STATE['source_type'])
            rep['faults'] = R4.STATE['faults']
            rep['per_type'] = R4.STATE['per_type']
            rep['source_type_codes'] = {'0': 'none', '1': 'joint', '2': 'fault', '3': 'joint and fault'}
            rep.setdefault('timing_s', {})['d_faults'] = R4.STATE['seconds']
            if R4.STATE['faults']['status'] == 'unsupported':
                rep.setdefault('stop', {})['fault_unsupported'] = True
                code = 3 if code == 0 else code
            rep['exit_code'] = code
        rp.write_text(json.dumps(rep, indent=1, allow_nan=False, default=float) + '\n', encoding='utf8')
    sys.exit(code)


if __name__ == '__main__':
    if '--r4' in sys.argv: main_r4()
    else: main()
