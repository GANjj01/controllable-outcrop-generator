"""Dark-band width against the geometric opening, and the share of the visible length that is not unclear against the
opening in pixels (main text Section 5.3, first paragraph; supplement S4, apparent width). The manuscript calls the opening
"aperture" and the share that is not unclear the "brightness-confirmed" fraction.

Definitions:
  geometric opening (px)  = mean opening / nominal pixel size
  dark-band width (px)    = per-image median of the dark-band width (records/fine_width_formal.json, rows[].fine.median: every
                            pixel of target.npy, brightness profile across the trace at a step of 0.25 px, half-depth crossings
                            linearly interpolated; samples with a depth below 2 grey levels have no dark band and do not enter
                            the median; procedure in supplement S3)
  clear fraction          = 1 - unclear (U) length / visible length, both from report.json per_type.joint (visible_u_px,
                            visible_px)
30 cases = the 6 viewpoints of the dark-band width table x the mean openings 3, 4, 5.5, 6.5 and 8 cm (cases w030, w040, b055,
w065, w080 of records/frames_index.json, blocks B and R); sun elevation 40 degrees, one image per case.

At the two coarse flat-outcrop viewpoints target.npy also holds fault-trace pixels, so the dark-band median of the record mixes
joint and fault pixels there. The width measurement is therefore run again here (checked against the record on all pixels) and
its median is also taken over the joint pixels only (key fault_pixels).

    python reproduce/paper1/analysis/width_curve.py --frames FRAMES

Reads records/fine_width_formal.json, records/frames_index.json and, for the 30 frames, FRAMES/<frame id>/q0/{light.json,
render_spec.json, render/frame.png, trace_labels/{report.json,target.npy,source_type.npy}}; writes records/round3/width_curve.json.
The input checks stop the script (status input_check_failed) when a frame does not match the record. Needs NumPy, SciPy and
Pillow (the labels environment)."""
import os
import sys
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
# lets NumPy find the BLAS libraries when the interpreter of a conda environment is started by its path
_lib = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.isdir(_lib):
    os.environ['PATH'] = _lib + os.pathsep + os.environ.get('PATH', '')
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402

_ap = argparse.ArgumentParser(description='Dark-band width and share of the visible length that is not unclear, against the geometric opening.')
_ap.add_argument('--frames', required=True, help='folder with one sub-folder per frame (a0001 ...)')
_ap.add_argument('--index', default=str(fio.INDEX))
_ap.add_argument('--fine-width', default=str(fio.PAPER1 / 'records' / 'fine_width_formal.json'))
_ap.add_argument('--out', default=str(fio.RECORDS / 'width_curve.json'))
ARGS = _ap.parse_args()
FRAMES = ARGS.frames
FW = Path(ARGS.fine_width)
# the 6 viewpoints of the dark-band width table (flat outcrop coarse to fine, standalone outcrop, then the 2 mm/px pair)
VIEWS = [('a26120005_c2', 'flat', 'flat 19.39'), ('a26120005_c3', 'flat', 'flat 13.54'), ('a26120005_c0', 'flat', 'flat 6.77'),
         ('a26130005_c1', 'standalone', 'standalone 7.96'), ('n26120003_c1', 'flat', 'flat 2.00'), ('n26130007_c0', 'standalone', 'standalone 2.00')]
TAGS = [('w030', 30.), ('w040', 40.), ('b055', 55.), ('w065', 65.), ('w080', 80.)]      # mean opening, mm
SEG_LOW, SEG_HIGH = 3., 10.        # px, strict inequalities (below 3 pixels, above 10 pixels)
SUN_ELEVATION = 40.


def J(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ranks(a):
    """average ranks (ties share the mean of their positions), 1-based"""
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2. + 1.
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else None


def rng(v):
    return [float(min(v)), float(max(v))] if len(v) else None


out = dict(
    description='Dark-band width and share of the visible length that is not unclear of the 30 cases (6 viewpoints x 5 mean openings; one image per '
                'case, clear sky, sun elevation 40 degrees). points: one row per case. width_ratio = dark-band width / geometric opening, both in '
                'pixels; discernible_share = 1 - U length / visible length of the joints.',
    paper_items=['main text Section 5.3, first paragraph', 'supplement S4: apparent width, gaps and the trace length retained in the labels, first paragraph'],
    script='reproduce/paper1/analysis/width_curve.py',
    units=dict(opening_mm='mean opening, mm', pixel_mm='nominal pixel size, mm', geometric_width_px='opening_mm / pixel_mm',
               image_width_px='per-image median dark-band width, pixels', width_ratio='image_width_px / geometric_width_px',
               u_share='visible_u_px / visible_px of the joints', discernible_share='1 - u_share',
               segment='lt3: geometric opening below 3 px; gt10: above 10 px; mid: between'))

fw = J(FW)
idx = fio.index(ARGS.index)
rows = {r['id']: r for r in fw['rows']}

# ---------------------------------------------------------------- input checks
points, problems = [], dict(g1=[], g2=[], g3=[], g4=[], g5=[])
g1_head = dict(label_set=fw.get('label_set'), old_ruler_rows=fw['old_ruler_check']['rows'], old_ruler_equal=fw['old_ruler_check']['equal'],
               old_ruler_mismatches=fw['old_ruler_check']['mismatches'])
if g1_head['label_set'] != 'formal':
    problems['g1'].append('label_set is not formal')
if g1_head['old_ruler_mismatches'] or g1_head['old_ruler_rows'] != g1_head['old_ruler_equal']:
    problems['g1'].append('old ruler self-check of the record has mismatches')

for cam, outcrop, view in VIEWS:
    for tag, mm in TAGS:
        rid = 'b_%s_%s' % (tag, cam)
        p = dict(id=rid, camera=cam, view=view, outcrop=outcrop, tag=tag, opening_mm=mm)
        points.append(p)
        r = rows.get(rid)
        if r is None or r.get('status') != 'measured' or (r.get('fine') or {}).get('median') is None:
            problems['g1'].append('%s: no measured fine width' % rid)
            continue
        key = r['label_dir'].replace(chr(92), '/').split('/')[-1]
        p.update(key=key, image_width_px=float(r['fine']['median']), image_width_iqr_px=[float(x) for x in r['fine']['iqr']],
                 width_samples=int(r['fine']['samples']), no_band_share=float(r['fine']['no_band_share']),
                 width_end_reached=int(r['fine']['end_reached']), role=r['role'])
        v = idx.get(key)
        if v is None:
            problems['g2'].append('%s: frame %s is not in the index' % (rid, key))
            continue
        d = fio.label_dir(FRAMES, key)
        point = fio.point_dir(FRAMES, key)
        p.update(block=v['block'], pixel_mm=float(v['pixel_mm']))
        bad = []
        if v['sid'] != rid or v['pid'] != 'sun0' or v['block'] not in ('B', 'R'):
            bad.append('index sid/pid/block = %s/%s/%s' % (v['sid'], v['pid'], v['block']))
        if float(v['pixel_mm']) != float(r['pixel_mm']):
            bad.append('pixel_mm index %r != fine width record %r' % (v['pixel_mm'], r['pixel_mm']))
        p['target_sha256_equal'] = sha(d / 'target.npy') == r['target_sha256']
        p['png_sha256_equal'] = sha(point / 'render' / 'frame.png') == r['png_sha256']
        if not p['target_sha256_equal']:
            bad.append('target.npy changed since the width was measured')
        if not p['png_sha256_equal']:
            bad.append('frame.png differs from the image the width was measured on')
        problems['g2'] += ['%s: %s' % (rid, b) for b in bad]

        # label record
        rep = J(d / 'report.json')
        jt = (rep.get('per_type') or {}).get('joint') or {}
        pp = rep.get('per_path') or []
        vis, vu = jt.get('visible_px'), jt.get('visible_u_px')
        s_vis = float(sum(float(x.get('visible_px') or 0.) for x in pp))
        s_vu = float(sum(float(x.get('visible_u_px') or 0.) for x in pp))
        p.update(report_sha256=sha(d / 'report.json'), exit_code=rep.get('exit_code'),
                 stop_flags=sorted(k for k, x in (rep.get('stop') or {}).items() if x),
                 visible_px=vis, visible_u_px=vu, per_path_rows=len(pp),
                 fault_visible_px=float(((rep.get('per_type') or {}).get('fault') or {}).get('visible_px') or 0.),
                 u_windows=rep.get('u_windows'), u_line_px_raster=rep.get('u_line_px'), positive_pixels=rep.get('positive_pixels'))
        bad = []
        if rep.get('exit_code') != 0:
            bad.append('exit_code %r' % rep.get('exit_code'))
        if vis is None or vu is None or not vis > 0 or not 0 <= vu <= vis:
            bad.append('per_type.joint visible_px / visible_u_px = %r / %r' % (vis, vu))
        elif abs(vis - s_vis) > 1e-6 or abs(vu - s_vu) > 1e-6:
            bad.append('per_type.joint differs from the per_path sums (%r, %r against %r, %r)' % (vis, vu, s_vis, s_vu))
        problems['g3'] += ['%s: %s' % (rid, b) for b in bad]

        # case and light of the frame
        light = J(point / 'light.json')
        spec = J(point / 'render_spec.json')['frames'][0]
        phys = light.get('physical') or {}
        cov = light.get('light_covariates') or {}
        p.update(mean_opening_m=phys.get('mean_opening_m'), opening_variation=phys.get('opening_variation'), depth_m=phys.get('depth_m'),
                 roughness_m=phys.get('roughness_m'), sky=light.get('sky'), sun_elevation_deg=(light.get('light') or {}).get('elevation_deg'),
                 sun_azimuth_deg=(light.get('light') or {}).get('azimuth_deg'),
                 light_trace_angle_in_surface_deg=cov.get('in_surface_light_trace_angle_deg'),
                 spec_sun_elevation_deg=(spec.get('light') or {}).get('elevation_deg'), spp=spec.get('spp'),
                 stain_codes=(spec.get('mesh') or {}).get('stain_codes'), shot=spec.get('shot'), resolution=spec.get('resolution'))
        bad = []
        if phys.get('mean_opening_m') is None or abs(phys['mean_opening_m'] * 1000. - mm) > 1e-9:
            bad.append('mean_opening_m %r against tag %s' % (phys.get('mean_opening_m'), tag))
        if light.get('sky') != 'sun' or p['sun_elevation_deg'] != SUN_ELEVATION or p['spec_sun_elevation_deg'] != SUN_ELEVATION:
            bad.append('sky %r, sun elevation %r (light.json) %r (render_spec.json)' % (light.get('sky'), p['sun_elevation_deg'],
                                                                                    p['spec_sun_elevation_deg']))
        if spec.get('resolution') != [1024, 1024]:
            bad.append('resolution %r' % spec.get('resolution'))
        problems['g4'] += ['%s: %s' % (rid, b) for b in bad]

complete = all('shot' in p for p in points)
per_view_same = {}
if complete:
    for cam, _, view in VIEWS:
        ps = [p for p in points if p['camera'] == cam]
        same_shot = all(p['shot'] == ps[0]['shot'] for p in ps)
        same_az = all(p['sun_azimuth_deg'] == ps[0]['sun_azimuth_deg'] for p in ps)
        same_px = all(p['pixel_mm'] == ps[0]['pixel_mm'] for p in ps)
        same_other = all((p['opening_variation'], p['depth_m']) == (ps[0]['opening_variation'], ps[0]['depth_m']) for p in ps)
        per_view_same[cam] = dict(view=view, same_shot=same_shot, same_sun_azimuth=same_az, same_pixel_mm=same_px,
                                  same_variation_and_depth=same_other, sun_azimuth_deg=ps[0]['sun_azimuth_deg'],
                                  light_trace_angle_in_surface_deg=[p['light_trace_angle_in_surface_deg'] for p in ps],
                                  roughness_over_opening=[p['roughness_m'] / p['mean_opening_m'] for p in ps])
        if not (same_shot and same_az and same_px and same_other):
            problems['g5'].append('%s: shot %s, sun azimuth %s, pixel size %s, variation and depth %s' % (cam, same_shot, same_az, same_px, same_other))
else:
    problems['g5'].append('not evaluated: some of the 30 points are incomplete')

gates = [
    dict(name='I1 dark-band width record usable', passed=not problems['g1'],
         detail=dict(g1_head, points_with_fine_width=sum('image_width_px' in p for p in points), expected=30, problems=problems['g1'])),
    dict(name='I2 the 30 cases are frames of the formal label set', passed=not problems['g1'] and not problems['g2'],
         detail=dict(keys=sum('block' in p for p in points), target_sha256_equal=sum(bool(p.get('target_sha256_equal')) for p in points),
                     png_sha256_equal=sum(bool(p.get('png_sha256_equal')) for p in points),
                     blocks={b: sum(p.get('block') == b for p in points) for b in ('B', 'R')}, problems=problems['g2'])),
    dict(name='I3 label reports usable', passed=complete and not problems['g3'],
         detail=dict(exit_code_0=sum(p.get('exit_code') == 0 for p in points),
                     frames_with_stop_flag=sum(bool(p.get('stop_flags')) for p in points),
                     frames_with_fault_visible_length=sum((p.get('fault_visible_px') or 0.) > 0 for p in points), problems=problems['g3'])),
    dict(name='I4 opening and illumination as stated', passed=complete and not problems['g4'],
         detail=dict(mean_opening_equals_tag=sum(p.get('mean_opening_m') is not None and abs(p['mean_opening_m'] * 1000. - p['opening_mm']) <= 1e-9
                                                 for p in points),
                     sky_sun=sum(p.get('sky') == 'sun' for p in points),
                     sun_elevation_40=sum(p.get('sun_elevation_deg') == SUN_ELEVATION for p in points),
                     stain_codes_empty=sum(p.get('stain_codes') == [] for p in points),
                     spp=sorted({p.get('spp') for p in points if p.get('spp') is not None}), problems=problems['g4'])),
    dict(name='I5 the 5 openings of a viewpoint share the camera, the sun azimuth and the other shape parameters', passed=complete and not problems['g5'],
         detail=dict(per_view=per_view_same, problems=problems['g5'])),
]
out['input_checks'] = gates
passed = all(g['passed'] for g in gates)
out['status'] = 'done' if passed else 'input_check_failed'

if passed:
    for p in points:
        p['geometric_width_px'] = p['opening_mm'] / p['pixel_mm']
        p['width_ratio'] = p['image_width_px'] / p['geometric_width_px']
        p['image_width_mm'] = p['image_width_px'] * p['pixel_mm']
        p['u_share'] = p['visible_u_px'] / p['visible_px']
        p['discernible_share'] = 1. - p['u_share']
        p['segment'] = 'lt3' if p['geometric_width_px'] < SEG_LOW else ('gt10' if p['geometric_width_px'] > SEG_HIGH else 'mid')
    g = np.array([p['geometric_width_px'] for p in points])
    ratio = np.array([p['width_ratio'] for p in points])
    disc = np.array([p['discernible_share'] for p in points])

    def seg(name):
        ps = [p for p in points if p['segment'] == name]
        rs = [p['width_ratio'] for p in ps]
        return dict(points=len(ps), ratio_range=rng(rs), all_above_1=bool(ps) and all(r > 1. for r in rs), all_below_1=bool(ps) and all(r < 1. for r in rs),
                    above_1=sum(r > 1. for r in rs), below_1=sum(r < 1. for r in rs),
                    geometric_width_range_px=rng([p['geometric_width_px'] for p in ps]),
                    discernible_share_range=rng([p['discernible_share'] for p in ps]),
                    members=[dict(id=p['id'], view=p['view'], opening_mm=p['opening_mm'], geometric_width_px=p['geometric_width_px'],
                                  image_width_px=p['image_width_px'], width_ratio=p['width_ratio']) for p in ps])

    lo, hi, mid = seg('lt3'), seg('gt10'), seg('mid')
    rho = spearman(g, disc)
    try:                                    # cross-check of the rank correlation only; the p value is stored but not used
        from scipy import stats
        sp = stats.spearmanr(g, disc)
        scipy_check = dict(rho=float(sp[0]), p_two_sided=float(sp[1]), equal_to_own=bool(abs(float(sp[0]) - rho) < 1e-12))
    except Exception as e:                  # noqa: BLE001
        scipy_check = dict(error=repr(e))
    imin, imax = int(np.argmin(ratio)), int(np.argmax(ratio))
    dmin, dmax = int(np.argmin(disc)), int(np.argmax(disc))

    out['report'] = dict(
        points=30,
        geometric_width_range_px=rng(g),
        image_width_range_px=rng([p['image_width_px'] for p in points]),
        width_ratio=dict(range=rng(ratio), median=float(np.median(ratio)), min_at=points[imin]['id'], max_at=points[imax]['id'],
                         above_1=int((ratio > 1.).sum()), below_1=int((ratio < 1.).sum())),
        segment_below_3px=lo, segment_above_10px=hi, segment_3_to_10px_record_only=mid,
        discernible_share=dict(range=rng(disc), median=float(np.median(disc)), min_at=points[dmin]['id'], max_at=points[dmax]['id'],
                               u_share_range=rng(1. - disc)),
        spearman_geometric_width_vs_discernible_share=dict(rho=rho, n=30, ties='average ranks', scipy=scipy_check))

    # not used in the manuscript
    per_view = []
    for cam, outcrop, view in VIEWS:
        ps = [p for p in points if p['camera'] == cam]
        gg = [p['geometric_width_px'] for p in ps]
        per_view.append(dict(
            camera=cam, view=view, outcrop=outcrop, pixel_mm=ps[0]['pixel_mm'],
            geometric_width_px=gg, image_width_px=[p['image_width_px'] for p in ps], width_ratio=[p['width_ratio'] for p in ps],
            width_ratio_range=rng([p['width_ratio'] for p in ps]),
            discernible_share=[p['discernible_share'] for p in ps], discernible_share_range=rng([p['discernible_share'] for p in ps]),
            image_width_slope_px_per_px=float(np.polyfit(gg, [p['image_width_px'] for p in ps], 1)[0]),
            spearman_width_vs_discernible_n5=spearman(gg, [p['discernible_share'] for p in ps]),
            sun_azimuth_deg=ps[0]['sun_azimuth_deg'], light_trace_angle_in_surface_deg=ps[0]['light_trace_angle_in_surface_deg']))
    out['record_only'] = dict(
        note='Not used in the manuscript: values per viewpoint, Spearman correlation of the 5 cases of a viewpoint, least-squares slope of '
             'the dark-band width against the geometric opening, Spearman correlation for flat and standalone outcrops separately, and '
             'the fraction of unclear windows.',
        per_view=per_view,
        spearman_flat_n20=spearman([p['geometric_width_px'] for p in points if p['outcrop'] == 'flat'],
                                   [p['discernible_share'] for p in points if p['outcrop'] == 'flat']),
        spearman_standalone_n10=spearman([p['geometric_width_px'] for p in points if p['outcrop'] == 'standalone'],
                                       [p['discernible_share'] for p in points if p['outcrop'] == 'standalone']),
        spearman_geometric_width_vs_u_window_share_n30=spearman(g, [p['u_windows']['u'] / p['u_windows']['total'] for p in points]),
        spearman_geometric_width_vs_width_ratio_n30=spearman(g, ratio))

    # ------------------------------------------------------------ fault pixels in the centreline
    # target.npy holds joint and fault positives together (source_type.npy: 1 joint, 2 fault, 3 both). The dark-band width of the
    # record is the median over ALL target pixels; the clear fraction above is the joints' alone. Here the width measurement
    # (step 0.25 px, every target pixel, normals from all target pixels) is run again, checked against the record on all pixels,
    # and its median is taken again over the joint-only samples.
    import math
    from PIL import Image
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    def fine_ruler(Y, pts, pixel_mm, step=.25):
        mm = pixel_mm / 1000.
        Rr = min(220, int(math.ceil(.40 / mm)))
        centre_reach = max(2., .04 / mm)
        s = np.arange(-Rr, Rr + step / 2., step)
        ringm = np.abs(s) >= .75 * Rr
        inner = np.abs(s) <= centre_reach
        inner_idx = np.flatnonzero(inner)
        tree = cKDTree(pts)
        idx_, wid, end = [], [], []
        for i in range(len(pts)):
            y, x = pts[i]
            near = pts[tree.query_ball_point((y, x), 6.)]
            if len(near) < 3:
                continue
            c = near - near.mean(axis=0)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            ny, nx = -vt[0][1], vt[0][0]
            prof = ndi.map_coordinates(Y, [y + s * ny, x + s * nx], order=1, mode='nearest')
            ring = float(np.median(prof[ringm]))
            i0 = inner_idx[int(np.argmin(prof[inner]))]
            depth = ring - float(prof[i0])
            idx_.append(i)
            if depth < 2.:
                wid.append(np.nan)
                end.append(False)
                continue
            half = ring - depth / 2.
            lo_ = i0
            while lo_ > 0 and prof[lo_ - 1] < half:
                lo_ -= 1
            hi_ = i0
            while hi_ < len(s) - 1 and prof[hi_ + 1] < half:
                hi_ += 1
            s_lo = s[lo_] - step * (half - prof[lo_]) / (prof[lo_ - 1] - prof[lo_]) if lo_ > 0 else s[0]
            s_hi = s[hi_] + step * (half - prof[hi_]) / (prof[hi_ + 1] - prof[hi_]) if hi_ < len(s) - 1 else s[-1]
            wid.append(float(s_hi - s_lo))
            end.append(lo_ == 0 or hi_ == len(s) - 1)
        return np.array(idx_, int), np.array(wid, float), np.array(end, bool)

    def stat(w, e):
        ok = np.isfinite(w)
        total = len(w)
        if not ok.any():
            return dict(samples=total, median=None, iqr=None, no_band_share=1. if total else None, end_reached=int(e.sum()))
        q = np.quantile(w[ok], [.25, .5, .75])
        return dict(samples=total, median=float(q[1]), iqr=[float(q[0]), float(q[2])], no_band_share=float((~ok).sum() / total),
                    end_reached=int(e.sum()))

    recheck_max_diff = 0.
    for p in points:
        d = fio.label_dir(FRAMES, p['key'])
        target = np.load(d / 'target.npy').astype(bool)
        st = np.load(d / 'source_type.npy')
        pts = np.argwhere(target)
        Y = np.asarray(Image.open(fio.point_dir(FRAMES, p['key']) / 'render' / 'frame.png').convert('RGB')).astype(float).mean(axis=2)
        i_, w_, e_ = fine_ruler(Y, pts, p['pixel_mm'])
        typ = st[pts[i_, 0], pts[i_, 1]]
        a = stat(w_, e_)
        ref = rows[p['id']]['fine']
        diff = abs(a['median'] - ref['median'])
        recheck_max_diff = max(recheck_max_diff, diff)
        c = np.bincount(st[target].ravel(), minlength=4)
        p['fault_check'] = dict(
            target_pixels=int(target.sum()), joint_only_pixels=int(c[1]), fault_only_pixels=int(c[2]), crossing_pixels=int(c[3]),
            untyped_pixels=int(c[0]), fault_pixel_share_of_target=float((c[2] + c[3]) / target.sum()),
            all_pixels=a, all_pixels_median_abs_diff_from_record=diff,
            all_pixels_equal_record=bool(diff <= 1e-9 and a['samples'] == ref['samples'] and a['no_band_share'] == ref['no_band_share']
                                         and a['end_reached'] == ref['end_reached']),
            all_pixels_exactly_equal_record=bool(a == {k: ref[k] for k in a}),
            joint_only=stat(w_[typ == 1], e_[typ == 1]), fault_only=stat(w_[typ == 2], e_[typ == 2]),
            u_share_joint_and_fault=(p['visible_u_px'] + float(J(d / 'report.json')['per_type']['fault']['visible_u_px']))
            / (p['visible_px'] + p['fault_visible_px']))
        p['image_width_joint_only_px'] = p['fault_check']['joint_only']['median']
        p['width_ratio_joint_only'] = p['image_width_joint_only_px'] / p['geometric_width_px']
    recheck_ok = all(p['fault_check']['all_pixels_equal_record'] for p in points)
    with_fault = [p for p in points if p['fault_check']['fault_only_pixels'] + p['fault_check']['crossing_pixels'] > 0]

    def seg_j(name):
        rs = [p['width_ratio_joint_only'] for p in points if p['segment'] == name]
        return dict(points=len(rs), ratio_range=rng(rs), above_1=sum(r > 1. for r in rs), below_1=sum(r < 1. for r in rs))

    rj = np.array([p['width_ratio_joint_only'] for p in points])
    out['fault_pixels'] = dict(
        note='At the two coarse flat-outcrop viewpoints (scene 26120005, plan cameras c2 and c3) the final centreline holds fault-trace '
             'pixels. The dark-band width record is the median over all centreline pixels, so the dark-band width of these two viewpoints '
             'is the median over joint and fault pixels together, and the fault trace does not change with the opening of the joints. '
             'The clear fraction is taken from per_type.joint and holds joints only. The same width measurement is run again here: '
             'its result on all pixels is checked against the record, then the median is taken over the joint pixels (source_type 1).',
        ruler_recheck=dict(frames=len(points), equal_to_record=sum(p['fault_check']['all_pixels_equal_record'] for p in points),
                           exactly_equal=sum(p['fault_check']['all_pixels_exactly_equal_record'] for p in points),
                           max_abs_median_diff_px=recheck_max_diff, passed=recheck_ok,
                           tolerance='medians within 1e-9 px; same number of samples, same share without dark band, same number of samples at the profile end'),
        frames_with_fault_pixels=[dict(id=p['id'], view=p['view'], opening_mm=p['opening_mm'], key=p['key'],
                                       geometric_width_px=p['geometric_width_px'],
                                       fault_pixel_share_of_target=p['fault_check']['fault_pixel_share_of_target'],
                                       image_width_all_px=p['image_width_px'], image_width_joint_only_px=p['image_width_joint_only_px'],
                                       image_width_fault_only_px=p['fault_check']['fault_only']['median'],
                                       width_ratio_all=p['width_ratio'], width_ratio_joint_only=p['width_ratio_joint_only'],
                                       joint_only_samples=p['fault_check']['joint_only']['samples'],
                                       joint_only_no_band_share=p['fault_check']['joint_only']['no_band_share'],
                                       u_share_joint=p['u_share'], u_share_joint_and_fault=p['fault_check']['u_share_joint_and_fault'])
                                  for p in with_fault],
        views_with_fault_pixels=sorted({p['view'] for p in with_fault}),
        fault_pixel_share_range=rng([p['fault_check']['fault_pixel_share_of_target'] for p in with_fault]),
        joint_only_variant=dict(
            valid=recheck_ok,
            width_ratio=dict(range=rng(rj), median=float(np.median(rj)), above_1=int((rj > 1.).sum()), below_1=int((rj < 1.).sum())),
            segment_below_3px=seg_j('lt3'), segment_above_10px=seg_j('gt10'), segment_3_to_10px=seg_j('mid'),
            rule_width_would_hold=bool(all(p['width_ratio_joint_only'] > 1. for p in points if p['segment'] == 'lt3')
                                       and all(p['width_ratio_joint_only'] < 1. for p in points if p['segment'] == 'gt10')),
            per_view=[dict(view=view, image_width_joint_only_px=[p['image_width_joint_only_px'] for p in points if p['camera'] == cam],
                           width_ratio_joint_only=[p['width_ratio_joint_only'] for p in points if p['camera'] == cam])
                      for cam, _, view in VIEWS]))

    fp = out['fault_pixels']
    lt3_views = sorted({p['view'] for p in points if p['segment'] == 'lt3'})
    out['notes'] = [
        'All values come from one fixed illumination per viewpoint (clear sky, sun elevation 40 degrees) and the untextured material, one image per case.',
        'Viewpoints with fault pixels in the final centreline: %s (%.1f%%-%.1f%% of its pixels); the %d cases with a geometric opening below '
        '3 px all come from %s.' % (', '.join(fp['views_with_fault_pixels']), 100. * fp['fault_pixel_share_range'][0], 100. * fp['fault_pixel_share_range'][1],
                                    lo['points'], ', '.join(lt3_views)) if with_fault else 'No viewpoint has fault pixels in the final centreline.']
    out['points'] = points
else:
    out['points_checked'] = points

fio.write_json(ARGS.out, out)
for x in gates:
    print('[%s] %s %s' % ('PASS' if x['passed'] else 'FAIL', x['name'], x['detail'].get('problems') or ''))
if 'report' in out:
    rep_ = out['report']
    print('width ratio: range %.2f-%.2f, median %.2f, above 1: %d, below 1: %d' % (
        rep_['width_ratio']['range'][0], rep_['width_ratio']['range'][1], rep_['width_ratio']['median'], rep_['width_ratio']['above_1'], rep_['width_ratio']['below_1']))
    for k in ('segment_below_3px', 'segment_above_10px'):
        print('%s: %d cases, ratio %.2f-%.2f, above 1: %d, below 1: %d' % (k, rep_[k]['points'], rep_[k]['ratio_range'][0], rep_[k]['ratio_range'][1],
                                                                        rep_[k]['above_1'], rep_[k]['below_1']))
    print('clear fraction %.1f%%-%.1f%%, Spearman with the geometric opening %.2f' % (
        100 * rep_['discernible_share']['range'][0], 100 * rep_['discernible_share']['range'][1], rep_['spearman_geometric_width_vs_discernible_share']['rho']))
    jv = out['fault_pixels']['joint_only_variant']
    print('joint pixels only: ratio %.2f-%.2f; width measurement equal to the record: %s' % (jv['width_ratio']['range'][0], jv['width_ratio']['range'][1], jv['valid']))

