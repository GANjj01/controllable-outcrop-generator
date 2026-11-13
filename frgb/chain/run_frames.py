"""Trace labels on existing frames: no rendering, no scoring, no model.

Groups share one cut mesh and one camera: geometry (projection, visibility, contradiction check, mouth record) runs
once per group, the RGB-dependent U rule once per frame."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1]
R1DIR = P1 / 'trace_labels_r1'
sys.path.insert(0, str(P1))
from frgb.metrics import graph_node_weights, skeletonize  # noqa: E402
from frgb.network_training_labels import _projected_boxes, raster_targets, material_u_from_readback  # noqa: E402
from frgb.raster import camera  # noqa: E402
from frgb import trace_labels as tl  # noqa: E402

from frgb import runconfig as frgb_rc  # noqa: E402
try:
    REVIEW = Path(frgb_rc.root('repo')) / 'docs/review_images/fracture_trace_labels_20261030'
except frgb_rc.ConfigError:
    REVIEW = None
CRITERION = P1 / 'trace_labels_r1' / 'criterion_trace_labels_r1.json'
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
CAMERA_KEYS = ('position_enu_m', 'target_enu_m', 'focal_length_mm', 'filmback_mm', 'near_clip_cm')


def capture_frame(name, capture):
    req = read(capture / 'project/ReviewData/request.json')
    cam_name = req['shots'][0]['camera_name']
    result = read(capture / 'result.json')
    return dict(name=name, capture=capture, request=req,
                rgb=capture / result['images'][0],
                depth=capture / ('images/labels/' + cam_name + '_depth_cm.npy'),
                rock=capture / ('images/labels/' + cam_name + '_labels.npy'),
                readiness=capture / 'project/ReviewData/readiness.json')


_HISTORY = None


def _history():
    global _HISTORY
    if _HISTORY is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('run_frames_r1_history', R1DIR / 'run_frames.py')
        _HISTORY = importlib.util.module_from_spec(spec); spec.loader.exec_module(_HISTORY)
    return _HISTORY


def __getattr__(name):
    if name in ('GROUPS', 'T', 'N2'):
        return getattr(_history(), name)
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def load_cpu_cache(path, request):
    a = np.load(path, allow_pickle=False)
    meta = json.loads(str(a['metadata']))
    same = all(meta['shot'][k] == request['shots'][0][k] for k in CAMERA_KEYS)
    if not same or meta['resolution'] != request['resolution'] or meta['source'] != request['source_native_surface']:
        raise ValueError('CPU cache has another camera/source: ' + str(path))
    return {k: a[k] for k in a.files if k != 'metadata'}


def aligned_mask(cpu, depth_m, rock):
    hit = cpu['face'] >= 0
    both = hit & rock
    residual = np.full(hit.shape, np.inf)
    residual[both] = np.abs(depth_m[both] - cpu['depth'][both])
    return both & np.isfinite(depth_m) & (residual <= np.maximum(.01, .002 * cpu['depth']))


def signed_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)


def visible_under(res, factor, cam):
    eps = factor * np.maximum(tl.EPS_FLOOR_M, tl.EPS_PIXEL_FRACTION * res['z'] / cam['fpx'])
    depth_ok = res['in_frame'] & (res['hit_z'] >= res['z'] - eps)
    return depth_ok | res['visible_own_cavity_only']


def old_edge_reproduction(case, rec, cpu, rock, aligned, frame, roles):
    from scipy import ndimage as ndi
    cam_near = None
    hit = cpu['face'] >= 0
    visible_roles = roles[np.maximum(cpu['face'], 0)]
    geometric_u = hit & ((cpu['view_cos'] < .1) | (cpu['depth'] <= frame['cam']['near'] + 1e-7))
    unresolved_face = hit & ~np.isin(visible_roles, ['original_host', 'left_wall', 'right_wall'])
    pending = rec['pending'] | geometric_u | unresolved_face
    material, _ = material_u_from_readback(read(frame['readiness']), hit.shape)
    protection = hit & (visible_roles != 'original_host')
    protection |= ndi.binary_dilation(rec['target'] | pending, iterations=2)
    out = raster_targets(rec['target'], rock, aligned, visible_roles, pending_u=pending, material_u=material,
                         protect=protection)
    return out


def overlay(rgb, labels, edge_target=None):
    img = rgb.astype(float).copy()
    ign = labels['ignore_reason'] > 0
    img[ign] = .6 * img[ign] + .4 * np.array([220, 40, 210])
    if edge_target is not None:
        img[edge_target] = [0, 200, 255]
    img[labels['occluded_line']] = [255, 40, 40]
    img[labels['u_line']] = [255, 220, 0]
    img[labels['target']] = [0, 255, 90]
    return img.astype(np.uint8)


def thick(mask, r=1):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(mask, iterations=r) if mask.any() else mask


def save_overlays(frame_name, rgb, labels, edge_target, crops, full=True):
    REVIEW.mkdir(parents=True, exist_ok=True)
    paths = []
    if full:
        big = dict(labels)
        for k in ('target', 'u_line', 'occluded_line'):
            big[k] = thick(labels[k], 2)
        im = Image.fromarray(overlay(rgb, big, thick(edge_target, 1) if edge_target is not None else None))
        im = im.resize((rgb.shape[1] // 2, rgb.shape[0] // 2), Image.LANCZOS)
        p = REVIEW / (frame_name + '_full.png'); im.save(p); paths.append(p)
    for i, (x0, y0, size) in enumerate(crops):
        sub = {k: v[y0:y0 + size, x0:x0 + size] for k, v in labels.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        e = edge_target[y0:y0 + size, x0:x0 + size] if edge_target is not None else None
        left = Image.fromarray(rgb[y0:y0 + size, x0:x0 + size])
        right = Image.fromarray(overlay(rgb[y0:y0 + size, x0:x0 + size], sub, e))
        sheet = Image.new('RGB', (2 * size + 8, size), (255, 255, 255))
        sheet.paste(left, (0, 0)); sheet.paste(right, (size + 8, 0))
        scale = 2 if size <= 400 else 1
        sheet = sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST)
        p = REVIEW / ('%s_crop%d.png' % (frame_name, i)); sheet.save(p); paths.append(p)
    return [str(p) for p in paths]


def pick_crops(labels, shape, size=320, count=3):
    """Deterministic crops: centres of the longest positive, U and occluded clusters (geometry/labels only)."""
    from scipy import ndimage as ndi
    out = []
    for key in ('target', 'u_line', 'occluded_line'):
        m = labels[key]
        if not m.any():
            continue
        lab, n = ndi.label(ndi.binary_dilation(m, iterations=3))
        sizes = np.bincount(lab.ravel())[1:]
        k = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(lab == k)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        x0 = int(np.clip(cx - size // 2, 0, shape[1] - size)); y0 = int(np.clip(cy - size // 2, 0, shape[0] - size))
        out.append((x0, y0, size))
    return out[:count]


def run_group(gname, spec, args):
    started = time.time()
    first = capture_frame(*spec['frames'][0])
    req = first['request']
    cam = camera(req['shots'][0], req['resolution'])
    case = tl.load_case(spec['case'])
    if Path(req['source_native_surface']) != spec['case'] / 'geometry/native_surface.npz':
        raise ValueError('capture not bound to the case')
    log = dict(group=gname, case=str(spec['case']), source=str(case['source']), resolution=req['resolution'],
               native_faces=int(len(case['native']['faces'])), host_faces=int(len(case['host']['faces'])))
    print(gname, 'loaded', round(time.time() - started, 1), flush=True)
    grid = tl.HostGrid(case['host']['vertices'], case['host']['faces'])
    case['host_normals'] = grid.normals
    vol = signed_volume(case['host']['vertices'].astype(float), case['host']['faces'])
    case['front_sign'] = 1. if vol > 0 else -1.
    log['host_signed_volume_m3'] = vol
    samples = {}
    for index, p in sorted(case['paths'].items()):
        res = tl.project_path(p['stations'], grid, cam)
        if res is not None:
            samples[index] = res
    print(gname, 'projected', {i: len(r['u']) for i, r in samples.items()}, round(time.time() - started, 1), flush=True)
    residual = tl.fracture_surface_residual(case['source'], samples, case['paths'])
    proj = {}
    for i, r in samples.items():
        ok = np.isfinite(r['t'])
        proj[i] = dict(samples=int(len(r['u'])), unresolved=int((~ok).sum()), multi_hit=int((r['host_hits'] > 1).sum()),
                       other_intersection_within_bound=int((r['other_intersections'] > 0).sum()),
                       abs_t_m=dict(p50=float(np.median(np.abs(r['t'][ok]))) if ok.any() else None,
                                    p99=float(np.quantile(np.abs(r['t'][ok]), .99)) if ok.any() else None,
                                    max=float(np.abs(r['t'][ok]).max()) if ok.any() else None),
                       fracture_surface_residual_max_m=residual[i], gaps_after_refinement=r['refine_rounds_left_gaps'])
    log['projection'] = proj
    nat = case['native']
    boxes = _projected_boxes(nat['vertices'], nat['faces'], cam)[:3]
    tl.classify_visibility(case, samples, cam, boxes=boxes)
    print(gname, 'visibility', round(time.time() - started, 1), flush=True)
    host_boxes = _projected_boxes(case['host']['vertices'], case['host']['faces'], cam)[:3]
    contra = tl.contradiction_check(case, samples, cam, host_boxes=host_boxes)
    log['contradiction'] = contra
    sens = {}
    for factor in (.5, 1., 2.):
        tot = 0.
        for r in samples.values():
            w = tl._sample_weights(r)
            tot += float(w[visible_under(r, factor, cam)].sum())
        sens[str(factor)] = tot
    log['eps_sensitivity_visible_px'] = sens
    print(gname, 'contradiction', json.dumps(contra), round(time.time() - started, 1), flush=True)
    cpu = load_cpu_cache(spec['cpu_cache'], req)
    depth_m = np.load(first['depth'], allow_pickle=False).astype(float) / 100.
    rock = np.load(first['rock'], allow_pickle=False) > 0
    aligned = aligned_mask(cpu, depth_m, rock)
    rec = tl.mouth_edge_record(case, cam, depth_m, rock, aligned, boxes=boxes)
    first['cam'] = cam
    old = old_edge_reproduction(case, rec, cpu, rock, aligned, first, nat['roles'])
    stored = np.load(_history().T / ('labels_%s_target.npy' % spec['old_labels']), allow_pickle=False)
    stored_valid = np.load(_history().T / ('labels_%s_valid.npy' % spec['old_labels']), allow_pickle=False)
    log['mouth_record_vs_stored_r3'] = dict(target_identical=bool(np.array_equal(old['target'], stored)),
                                            valid_identical=bool(np.array_equal(old['valid'], stored_valid)),
                                            target_pixels=int(old['target'].sum()), stored_pixels=int(stored.sum()))
    print(gname, 'mouth record', log['mouth_record_vs_stored_r3'], round(time.time() - started, 1), flush=True)
    # record-only cross-check: visible trace samples whose nearby same-path mouth samples are all hidden
    from scipy.spatial import cKDTree
    q = rec['qualification']; owner = rec['samples']['owner']
    mpath = rec['edges']['path_index'][owner]
    cross = 0.; checked = 0.; cross_w = 0.; checked_w = 0.
    for i, r in samples.items():
        sel = np.flatnonzero(r['visible'])
        m = np.flatnonzero(mpath == i)
        if not len(sel) or not len(m):
            continue
        tree = cKDTree(rec['samples']['uv'][m])
        near = tree.query_ball_point(r['uv'][sel], 2.)
        w = tl._sample_weights(r)[sel]
        for j, nb in enumerate(near):
            if nb:
                checked += w[j]
                if q['hidden'][m[nb]].all():
                    cross += w[j]
        radius = 2. + r['width'][sel] * cam['fpx'] / r['z'][sel]
        for j in range(len(sel)):
            nb = tree.query_ball_point(r['uv'][sel[j]], radius[j])
            if nb:
                checked_w += w[j]
                if q['hidden'][m[nb]].all():
                    cross_w += w[j]
    log['record_cross_check_all_mouth_hidden_px'] = dict(visible_with_mouth_samples_px=checked, all_hidden_px=cross,
                                                         posthoc_width_radius=dict(visible_with_mouth_samples_px=checked_w,
                                                                                   all_hidden_px=cross_w))
    out_group = R1DIR / 'frames'
    reports = []
    group_crops = None
    for fname, capture in spec['frames']:
        frame = capture_frame(fname, capture)
        if frame['request']['shots'][0]['position_enu_m'] != req['shots'][0]['position_enu_m'] or \
                frame['request']['source_native_surface'] != req['source_native_surface']:
            raise ValueError('frame not in this group: ' + fname)
        rgb = np.asarray(Image.open(frame['rgb']).convert('RGB'))
        d = np.load(frame['depth'], allow_pickle=False).astype(float) / 100.
        rk = np.load(frame['rock'], allow_pickle=False) > 0
        al = aligned_mask(cpu, d, rk)
        smp = {i: dict(r) for i, r in samples.items()}
        labels = tl.build_trace_labels(case, smp, cam, rgb, cpu_raster=cpu, render_rock=rk, aligned=al)
        od = out_group / fname
        od.mkdir(parents=True, exist_ok=True)
        for k in ('target', 'valid', 'ignore_reason', 'u_line', 'u_line_old_threshold', 'occluded_line'):
            np.save(od / (k + '.npy'), labels[k])
        np.save(od / 'mouth_edge_record_target.npy', rec['target'])
        rock_px = int(labels['rock'].sum())
        wins = labels['windows']
        report = dict(criterion=str(CRITERION), criterion_sha256=args.criterion_sha, frame=fname, group=gname,
                      rgb=str(frame['rgb']), status='TRACE_LABEL_CANDIDATE_NOT_QUALIFIED',
                      positive_graph_length_px=labels['positive_graph_length_px'],
                      positive_pixels=int(labels['target'].sum()),
                      ignore_fraction_of_rock=float((labels['ignore_reason'] > 0)[labels['rock']].sum() / max(rock_px, 1)),
                      ignore_reason_pixels={name: int(((labels['ignore_reason'] & bit) > 0).sum()) for name, bit in
                                            [('u', tl.IGNORE_U), ('transition', tl.IGNORE_TRANSITION),
                                             ('unresolved', tl.IGNORE_UNRESOLVED), ('scene', tl.IGNORE_SCENE),
                                             ('registration', tl.IGNORE_REGISTRATION)]},
                      u_windows=dict(total=len(wins), u=sum(w['u'] for w in wins),
                                     contrast_levels_quantiles=[float(x) for x in np.nanquantile(
                                         [w['contrast'] if w['contrast'] is not None else np.nan for w in wins],
                                         [.05, .25, .5, .75, .95])] if wins else None),
                      u_reason_windows=dict(
                          ring_below_min=sum(w['u'] and w['ring_pixels'] < tl.U_MIN_RING for w in wins),
                          band_empty=sum(w['u'] and w['band_pixels'] == 0 for w in wins),
                          contrast_below_T=sum(w['u'] and w['ring_pixels'] >= tl.U_MIN_RING and w['band_pixels'] > 0
                                               for w in wins)),
                      u_line_px=float(graph_node_weights(skeletonize(labels['u_line'])).sum()),
                      u_line_old_threshold_px=float(graph_node_weights(skeletonize(labels['u_line_old_threshold'])).sum()),
                      technical_failure_all_u=labels['all_visible_u'], per_path=labels['per_path'],
                      mouth_edge_record_pixels=int(rec['target'].sum()))
        if group_crops is None:
            group_crops = pick_crops(labels, rgb.shape[:2])
        report['crops_xy_size'] = group_crops
        report['overlays'] = save_overlays(fname, rgb, labels, rec['target'], group_crops, full=fname in args.full)
        (od / 'windows.json').write_text(json.dumps(wins, allow_nan=False) + '\n', encoding='utf8')
        (od / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n', encoding='utf8')
        reports.append(report)
        print(fname, 'pos', round(report['positive_graph_length_px'], 1), 'U windows', report['u_windows']['u'], '/',
              report['u_windows']['total'], 'ignore', round(report['ignore_fraction_of_rock'], 4), flush=True)
    log['frames'] = [dict(frame=r['frame'], positive_graph_length_px=r['positive_graph_length_px'],
                          u_windows=r['u_windows'], u_line_px=r['u_line_px'], u_line_old_threshold_px=r['u_line_old_threshold_px'],
                          ignore_fraction_of_rock=r['ignore_fraction_of_rock']) for r in reports]
    log['elapsed_s'] = time.time() - started
    (R1DIR / ('group_%s.json' % gname)).write_text(json.dumps(log, indent=1, allow_nan=False) + '\n', encoding='utf8')
    return log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='+', default=list(_history().GROUPS))
    ap.add_argument('--full', nargs='*', default=['path08_day_b1', 'path04_day_b1', 'shape_ctl_b1_e0_a0',
                                                 'shape_ctl_b1_e3_a1'])
    args = ap.parse_args()
    args.criterion_sha = (R1DIR / 'criterion_trace_labels_r1.sha256.txt').read_text(encoding='utf8').split()[0]
    for g in args.groups:
        log = run_group(g, _history().GROUPS[g], args)
        if log['contradiction']['stop_triggered']:
            print('STOP: visibility contradiction above the frozen 1% in group', g, flush=True)
            sys.exit(3)
