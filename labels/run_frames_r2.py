"""The a255 group has no render: geometry-only labels (build_trace_geometry) on a CPU shading image. Outputs:
labels/frames/<frame>/, labels/group_<name>.json. Exit 3 on a stop."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(P1)); sys.path.insert(0, str(P1 / 'frgb' / 'chain'))
import run_frames as r1  # noqa: E402
from frgb.metrics import graph_node_weights, skeletonize  # noqa: E402
from frgb.network_training_labels import _projected_boxes  # noqa: E402
from frgb.raster import camera  # noqa: E402
from frgb import trace_labels as tl  # noqa: E402

MS = P1 / 'multiscene'
from frgb import runconfig as frgb_rc  # noqa: E402
try:
    REVIEW = Path(frgb_rc.root('repo')) / 'docs/review_images/fracture_trace_labels_20261030/r2'
except frgb_rc.ConfigError:
    REVIEW = None
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))


def a255_group(scene, cavities):
    d = MS / 'feas_r2' / scene
    return dict(case=d / 'net/cases/c0', cavities=HERE / 'cavities' / cavities, request=d / 'levels/request_0.json',
                frames=[(scene + '_c0_cam0', None)], geometry_only=True)


GROUPS = {}


def load_cutters(path, cam):
    manifest = read(path / 'manifest_current.json')
    if not manifest['verified']:
        raise ValueError('cavity rebuild not verified: ' + str(path))
    out = {}
    if 'fault_groups' in manifest:
        import hashlib
        listed = {g['file']: g for g in manifest['fault_groups']}
        present = {f.name for f in path.glob('trace-*.npz')}
        if present != set(listed):
            raise ValueError('cavity files differ from the stage-A list: extra %s, missing %s'
                             % (sorted(present - set(listed)), sorted(set(listed) - present)))
        for name, g in sorted(listed.items()):
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != g['sha256']:
                raise ValueError('cavity file changed after stage A: ' + name)
            with np.load(path / name, allow_pickle=False) as a:
                v, fc = a['vertices'], a['faces']
            out[g['key']] = dict(vertices=v, faces=fc, sign=tl.mesh_sign(v, fc), boxes=_projected_boxes(v, fc, cam)[:3],
                                 trace=g['trace'], tag=g['tag'])
        return out, manifest
    for f in sorted(path.glob('trace-*.npz')):
        with np.load(f, allow_pickle=False) as a:
            v, fc, tid = a['vertices'], a['faces'], str(a['source_trace_id'])
        out[tid] = dict(vertices=v, faces=fc, sign=tl.mesh_sign(v, fc), boxes=_projected_boxes(v, fc, cam)[:3])
    return out, manifest


def weights_sum(samples, key):
    return float(sum(tl._sample_weights(r)[r[key]].sum() for r in samples.values()))


def eps_sensitivity(samples, cam):
    out = {}
    for factor in (.5, 1., 2.):
        out[str(factor)] = float(sum(tl._sample_weights(r)[r1.visible_under(r, factor, cam)].sum() for r in samples.values()))
    return out


def r1_checks(case, samples, cam, host_boxes):
    s = tl.contradiction_check(case, samples, cam, host_boxes=host_boxes)
    return dict(visible_length_px=s['visible_length_px'], a_length_px=s['contradiction_a_length_px'],
                b_length_px=s['contradiction_b_length_px'], fraction=s['contradiction_fraction'])


def branch_samples(case, grid, cam):
    samples, unresolved, info = {}, {}, {}
    for index, p in sorted(case['paths'].items()):
        branches, inf = tl.project_path_branches(p['stations'], grid, cam, case['front_sign'])
        if branches is None:
            continue
        unresolved[index] = inf.pop('unresolved_station_points')
        info[index] = inf
        for b in branches:
            b['path'] = index
            samples[(index, b['branch'])] = b
    return samples, unresolved, info


def projection_record(samples, info, residual, cam):
    out = {}
    for index, inf in info.items():
        rows = [r for k, r in samples.items() if k[0] == index]
        t = np.concatenate([r['t'] for r in rows]) if rows else np.empty(0)
        gaps = 0
        for r in rows:
            uv, _ = tl.project(r['point'], cam)
            step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
            gaps += int(np.sum(np.isfinite(step) & (step > tl.RUN_BREAK_PX)))
        out[str(index)] = dict(inf, abs_t_m=dict(p50=float(np.median(np.abs(t))), p99=float(np.quantile(np.abs(t), .99)),
                                                 max=float(np.abs(t).max())) if len(t) else None,
                               steps_over_2px=gaps, fracture_surface_residual_max_m=max(
                                   [residual[k] for k in residual if k[0] == index and residual[k] is not None], default=None))
    return out


def shading_image(case, cam_full, size=1024):
    """CPU headlight shading of the cut mesh at size x size (first hits at pixel centres), for geometry-only overlays."""
    shot = case['_shot']
    cam = camera(shot, (size, size))
    nat = case['native']
    yy, xx = np.mgrid[0:size, 0:size]
    uv = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    z, face = tl.first_hits(nat['vertices'], nat['faces'], cam, uv)
    v, f = nat['vertices'], nat['faces']
    ok = face >= 0
    tri = v[f[face[ok]]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1)[:, None]
    d = tl.rays(cam, uv[ok, 0], uv[ok, 1]); d /= np.linalg.norm(d, axis=1)[:, None]
    shade = np.zeros(len(uv)); shade[ok] = .15 + .85 * np.abs(np.einsum('ij,ij->i', n, d))
    roles = nat['roles'][np.maximum(face, 0)]
    img = np.repeat((shade * 220)[:, None], 3, 1)
    cav = ok & (roles != 'original_host')
    img[cav] = img[cav] * np.array([.75, .8, 1.])
    img[~ok] = [235, 240, 248]
    return img.reshape(size, size, 3).astype(np.uint8)


def draw(img, mask, colour, r=0):
    m = r1.thick(mask, r) if r else mask
    img[m] = colour


def marks(samples, flag, shape):
    out = np.zeros(shape, bool)
    for r in samples.values():
        if flag in r:
            out |= tl.rasterise_samples(r, r[flag], shape)
    return out


def parallel_far_marks(samples, shape):
    """Visible samples whose nearest same-u sibling is > 2 px away (for the overlay only)."""
    out = np.zeros(shape, bool)
    by = {}
    for k, r in samples.items():
        ok = r['visible']
        by.setdefault(r['path'], []).append((r['sample_index'][ok], r['uv'][ok]))
    for parts in by.values():
        s = np.concatenate([p[0] for p in parts]); uv = np.concatenate([p[1] for p in parts])
        order = np.argsort(s, kind='stable')
        bounds = np.flatnonzero(np.r_[True, s[order][1:] != s[order][:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                continue
            idx = order[a:b]
            dd = np.linalg.norm(uv[idx][:, None] - uv[idx][None], axis=2); np.fill_diagonal(dd, np.inf)
            far = idx[dd.min(1) > tl.PARALLEL_PX]
            p = np.floor(uv[far] + .5).astype(int)
            ok = (p[:, 0] >= 0) & (p[:, 0] < shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < shape[0])
            out[p[ok, 1], p[ok, 0]] = True
    return out


def overlay_r2(base, labels, contra, far, scale=1):
    img = base.astype(float).copy()
    ign = labels['ignore_reason'] > 0
    if scale == 1:
        img[ign] = .6 * img[ign] + .4 * np.array([220, 40, 210])
    sub = (lambda m: m) if scale == 1 else (lambda m: m.reshape(m.shape[0] // scale, scale, m.shape[1] // scale, scale).any((1, 3)))
    img = img.astype(np.uint8)
    draw(img, sub(labels['occluded_line']), [255, 40, 40])
    if 'u_line' in labels:
        draw(img, sub(labels['u_line']), [255, 220, 0])
    draw(img, sub(labels['target']), [0, 255, 90])
    draw(img, sub(far), [40, 120, 255])
    draw(img, sub(contra), [255, 128, 0])
    return img


def save_r2_overlays(name, base, labels, contra, far, crops, full=True, scale=1):
    REVIEW.mkdir(parents=True, exist_ok=True)
    paths = []
    if full:
        big = dict(labels)
        for k in ('target', 'u_line', 'occluded_line'):
            if k in labels:
                big[k] = r1.thick(labels[k], 2)
        im = Image.fromarray(overlay_r2(base, big, r1.thick(contra, 3), r1.thick(far, 2), scale))
        if scale == 1:
            im = im.resize((base.shape[1] // 2, base.shape[0] // 2), Image.LANCZOS)
        p = REVIEW / (name + '_full.png'); im.save(p); paths.append(str(p))
    for i, (x0, y0, size) in enumerate(crops):
        sl = (slice(y0, y0 + size), slice(x0, x0 + size))
        sub = {k: v[sl] for k, v in labels.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        if scale == 1:
            left = base[sl]
        else:
            left = np.asarray(Image.fromarray(base).resize((base.shape[1] * scale, base.shape[0] * scale), Image.NEAREST))[sl]
        right = overlay_r2(left, sub, contra[sl], far[sl])
        sheet = Image.new('RGB', (2 * size + 8, size), (255, 255, 255))
        sheet.paste(Image.fromarray(left), (0, 0)); sheet.paste(Image.fromarray(right), (size + 8, 0))
        if size <= 400:
            sheet = sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST)
        p = REVIEW / ('%s_crop%d.png' % (name, i)); sheet.save(p); paths.append(str(p))
    return paths


def pick_crops(labels, contra, far, shape, size=320):
    from scipy import ndimage as ndi
    out = []
    for m in (labels['target'], far, contra, labels.get('u_line', np.zeros(shape, bool)), labels['occluded_line']):
        if not m.any() or len(out) == 3:
            continue
        lab, _ = ndi.label(ndi.binary_dilation(m, iterations=3))
        k = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
        ys, xs = np.nonzero(lab == k)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        out.append((int(np.clip(cx - size // 2, 0, shape[1] - size)), int(np.clip(cy - size // 2, 0, shape[0] - size)), size))
    return out


def run_group(gname, spec, args):
    started = time.time()
    geometry_only = spec.get('geometry_only', False)
    if geometry_only:
        req = read(spec['request'])
        first = None
    else:
        first = r1.capture_frame(*spec['frames'][0])
        req = first['request']
    cam = camera(req['shots'][0], req['resolution'])
    case = tl.load_case(spec['case'])
    case['_shot'] = req['shots'][0]
    if not geometry_only and Path(req['source_native_surface']) != spec['case'] / 'geometry/native_surface.npz':
        raise ValueError('capture not bound to the case')
    log = dict(group=gname, criterion_sha256=args.criterion_sha, amendment_sha256=args.amendment_sha, case=str(spec['case']),
               source=str(case['source']), resolution=req['resolution'], native_faces=int(len(case['native']['faces'])),
               host_faces=int(len(case['host']['faces'])))
    grid = tl.HostGrid(case['host']['vertices'], case['host']['faces'])
    case['host_normals'] = grid.normals
    vol = r1.signed_volume(case['host']['vertices'].astype(float), case['host']['faces'])
    case['front_sign'] = 1. if vol > 0 else -1.
    log['host_signed_volume_m3'] = vol
    log['host_volume_m3_build'] = case['build']['host_volume_m3']
    log['host_volume_relative_difference'] = abs(abs(vol) - case['build']['host_volume_m3']) / case['build']['host_volume_m3']
    cutters, manifest = load_cutters(spec['cavities'], cam)
    log['cavities'] = dict(path=str(spec['cavities']), verified=manifest['verified'],
                           fractures={t: dict(volume_m3=f['volume_m3'], relative=f['relative']) for t, f in manifest['fractures'].items()})
    nat = case['native']
    boxes = _projected_boxes(nat['vertices'], nat['faces'], cam)[:3]
    host_boxes = _projected_boxes(case['host']['vertices'], case['host']['faces'], cam)[:3]
    print(gname, 'loaded', round(time.time() - started, 1), flush=True)
    ctrl = {}
    for index, p in sorted(case['paths'].items()):
        res = tl.project_path(p['stations'], grid, cam)
        if res is not None:
            ctrl[index] = res
    tl.classify_visibility(case, ctrl, cam, boxes=boxes)
    control = dict(visible_length_px=weights_sum(ctrl, 'visible'), r1_checks=r1_checks(case, ctrl, cam, host_boxes),
                   exemption=tl.exemption_check(case, ctrl, cam, cutters, host_boxes=host_boxes))
    print(gname, 'control', json.dumps(control['exemption']), round(time.time() - started, 1), flush=True)
    samples, unresolved, info = branch_samples(case, grid, cam)
    residual = tl.fracture_surface_residual(case['source'], samples, case['paths'])
    log['projection_r2'] = projection_record(samples, info, residual, cam)
    tl.classify_visibility(case, samples, cam, boxes=boxes)
    ex = tl.exemption_check(case, samples, cam, cutters, host_boxes=host_boxes)
    log['r2'] = dict(visible_length_px=weights_sum(samples, 'visible'), exemption=ex,
                     r1_checks_on_r2_samples=r1_checks(case, samples, cam, host_boxes),
                     eps_sensitivity_visible_px=eps_sensitivity(samples, cam),
                     own_cavity_clause_only_px=weights_sum(samples, 'visible_own_cavity_only'),
                     literal_extra_px=weights_sum(samples, 'literal_extra'))
    log['control_r1_projection'] = control
    log['stop_triggered'] = bool(ex['stop_triggered'])
    print(gname, 'r2 exemption', json.dumps(ex), round(time.time() - started, 1), flush=True)
    out_group = HERE / 'frames'
    reports = []
    frames = spec['frames'][:1] if log['stop_triggered'] else spec['frames']
    shape = (req['resolution'][1], req['resolution'][0])
    contra = marks(samples, 'ex_contradiction', shape)
    if geometry_only:
        labels = tl.build_trace_geometry(case, samples, cam, shape, unresolved_points=unresolved)
        far = parallel_far_marks(samples, shape)
        name = spec['frames'][0][0]
        od = out_group / name; od.mkdir(parents=True, exist_ok=True)
        for k in ('target', 'valid', 'ignore_reason', 'occluded_line'):
            np.save(od / (k + '.npy'), labels[k])
        base = shading_image(case, cam, size=shape[0] // 2)
        crops = pick_crops(labels, contra, far, shape)
        report = dict(frame=name, group=gname, status='TRACE_LABEL_GEOMETRY_ONLY_NOT_QUALIFIED',
                      positive_graph_length_px=labels['positive_graph_length_px'], positive_pixels=int(labels['target'].sum()),
                      parallel=tl.parallel_record(samples), per_path=labels['per_path'], crops_xy_size=crops,
                      overlays=save_r2_overlays(name, base, labels, contra, far, crops, scale=2))
        (od / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n', encoding='utf8')
        reports.append(report)
    else:
        cpu = r1.load_cpu_cache(spec['cpu_cache'], req)
        group_crops = None
        for fi, (fname, capture) in enumerate(frames):
            frame = r1.capture_frame(fname, capture)
            if frame['request']['shots'][0]['position_enu_m'] != req['shots'][0]['position_enu_m'] or \
                    frame['request']['source_native_surface'] != req['source_native_surface']:
                raise ValueError('frame not in this group: ' + fname)
            rgb = np.asarray(Image.open(frame['rgb']).convert('RGB'))
            dm = np.load(frame['depth'], allow_pickle=False).astype(float) / 100.
            rk = np.load(frame['rock'], allow_pickle=False) > 0
            al = r1.aligned_mask(cpu, dm, rk)
            if fi == 0:
                c_labels = tl.build_trace_labels(case, {i: dict(r) for i, r in ctrl.items()}, cam, rgb, cpu_raster=cpu,
                                                 render_rock=rk, aligned=al)
                old = HERE.parent / 'trace_labels_r1/frames' / fname
                log['control_reproduces_r1'] = {k: bool(np.array_equal(c_labels[k], np.load(old / (k + '.npy'))))
                                                for k in ('target', 'valid', 'ignore_reason', 'u_line', 'occluded_line')}
                print(gname, 'control reproduces r1', log['control_reproduces_r1'], flush=True)
            smp = {k: dict(r) for k, r in samples.items()}
            labels = tl.build_trace_labels(case, smp, cam, rgb, cpu_raster=cpu, render_rock=rk, aligned=al,
                                           unresolved_points=unresolved)
            far = parallel_far_marks(smp, shape)
            od = out_group / fname; od.mkdir(parents=True, exist_ok=True)
            for k in ('target', 'valid', 'ignore_reason', 'u_line', 'u_line_old_threshold', 'occluded_line'):
                np.save(od / (k + '.npy'), labels[k])
            wins = labels['windows']
            rock_px = int(labels['rock'].sum())
            report = dict(frame=fname, group=gname, rgb=str(frame['rgb']),
                          status='TRACE_LABEL_CANDIDATE_NOT_QUALIFIED' + ('_STOPPED_GROUP' if log['stop_triggered'] else ''),
                          positive_graph_length_px=labels['positive_graph_length_px'],
                          positive_pixels=int(labels['target'].sum()),
                          ignore_fraction_of_rock=float((labels['ignore_reason'] > 0)[labels['rock']].sum() / max(rock_px, 1)),
                          u_windows=dict(total=len(wins), u=sum(w['u'] for w in wins),
                                         ring_below_min=sum(w['u'] and w['ring_pixels'] < tl.U_MIN_RING for w in wins),
                                         ring_excluded_pixels=sum(w.get('ring_excluded', 0) for w in wins)),
                          u_line_px=float(graph_node_weights(skeletonize(labels['u_line'])).sum()),
                          parallel=tl.parallel_record(smp), parallel_visible_only=tl.parallel_record(smp, use_u=False),
                          technical_failure_all_u=labels['all_visible_u'], per_path=labels['per_path'])
            if group_crops is None:
                group_crops = pick_crops(labels, contra, far, shape)
            report['crops_xy_size'] = group_crops
            report['overlays'] = save_r2_overlays(fname, rgb, labels, contra, far, group_crops, full=fname in args.full)
            (od / 'windows.json').write_text(json.dumps(wins, allow_nan=False) + '\n', encoding='utf8')
            (od / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n', encoding='utf8')
            reports.append(report)
            print(fname, 'pos', round(report['positive_graph_length_px'], 1), 'U', report['u_windows']['u'], '/',
                  report['u_windows']['total'], 'parallel far', report['parallel']['far_fraction'], flush=True)
    log['frames'] = [dict(frame=r['frame'], positive_graph_length_px=r['positive_graph_length_px'],
                          u_windows=r.get('u_windows'), parallel=r['parallel'],
                          parallel_visible_only=r.get('parallel_visible_only')) for r in reports]
    log['elapsed_s'] = time.time() - started
    (HERE / ('group_%s.json' % gname)).write_text(json.dumps(log, indent=1, allow_nan=False) + '\n', encoding='utf8')
    return log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='+', default=['sa08_path08cam', 'sa04_path04cam', 'a255_H100_c0'])
    ap.add_argument('--full', nargs='*', default=['path08_day_b1', 'path04_day_b1', 'shape_ctl_b1_e0_a0', 'shape_ctl_b1_e3_a1'])
    args = ap.parse_args()
    args.criterion_sha = (HERE / 'criterion_trace_labels_r2.sha256.txt').read_text(encoding='utf8').split()[0]
    args.amendment_sha = (HERE / 'criterion_trace_labels_r2_amendment_a.sha256.txt').read_text(encoding='utf8').split()[0]
    for g in args.groups:
        log = run_group(g, GROUPS[g], args)
        if log['stop_triggered']:
            print('STOP: r2 visibility contradiction above the frozen 1% in group', g, flush=True)
            sys.exit(3)
