"""Position and facing direction of the occluded segments, and the trace length retained in the labels
(main text Section 4.2, paragraph on occluded traces; supplement S4, "Position and facing direction of the occluded segments"
with its distance table, and the last paragraph of "Apparent width, gaps and the trace length retained in the labels").

One frame for each of 11 viewpoints (frames_io.VIEWPOINTS); the retained length uses the 10 frames of every viewpoint of the
multi-illumination aerial set that meets the four conditions below.

1. Image distance d from every occluded-line pixel to the nearest pixel of the visible evidence M (fracture-mouth pixels
   opening_class > 0, final centreline target, unclear line u_line): fractions with d <= 2 px, 2 < d <= 10 px and d > 10 px,
   and the median. At viewpoints with a fault the occluded line also holds fault traces; their share of the occluded length is
   taken from the lengths in report.json.

2. Facing direction. Every trace sample of the label program lies on a triangle of the uncut rock surface (source_surface.npz
   of the cut case). A sample is in frame when it projects into the image, lies beyond the near clip plane and is not below the
   ground plane; it is occluded when the rendered depth at its pixel (floor(uv + 0.5)) is smaller than its view-axis depth by
   more than max(0.1 mm, half a pixel footprint). It faces the camera when the dot product of the triangle normal (vertex
   winding) and the direction from the sample to the camera is positive. Lengths are the image lengths of the label program:
   two successive rows of the same trace that are both in frame and at most 2 px apart share their distance half and half.
   The division by facing direction is used at a viewpoint only when four conditions are met:
     C1  at least 99 % of the samples lie within 0.1 mm of the plane of their triangle;
     C2  at least 95 % of the visible length faces the camera;
     C3  the occluded fraction reproduced from the depth map differs from that of report.json by at most 2 percentage points;
     C4  at least 99 % of the in-frame length that faces away from the camera is occluded.
   Reported there: b = share of the occluded length on rock facing away from the camera; o_f = occluded share of the length
   facing the camera (also with equal weight per sample).

3. Retained length, per frame of a viewpoint that meets the conditions: L_f = visible length of report.json + occluded length
   facing the camera; L_m = visible length - unclear (U) length; retained_share = L_m / L_f; 1 - retained_share = contribution
   of occlusion + contribution of the unclear segments. The same per group of traces of similar strike.

    python reproduce/paper1/analysis/occluded_segments.py --frames FRAMES
    python reproduce/paper1/analysis/occluded_segments.py --selftest      checks the pure functions; reads and writes nothing

Reads records/frames_index.json and, per frame, FRAMES/<frame id>/plan.json, q0/render_spec.json, q0/geometry.json,
q0/render/frame.npz (depth), q0/trace_labels/{report.json, target.npy, u_line.npy, occluded_line.npy, opening_class.npy,
refine_in_refined.npz} and q0/trace_labels_state.pkl; from the cut case named in geometry.json, build.json and the
source_surface.npz of its source folder. Writes records/round3/occluded_segments.json. Needs NumPy and SciPy (the labels
environment)."""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402

VIEWS = fio.VIEWPOINTS
MULTI_LIGHT = [k for k, v in VIEWS.items() if v[1] == 'multi-illumination aerial']
MESH_NAME = 'source_surface.npz'
C1_DIST_M, C1_MIN_FRACTION = 1e-4, .99
C2_MIN_FRONT = .95
C3_MAX_DIFF_PP = 2.
C4_MIN_BACK_OCCLUDED = .99
EPS_FLOOR_M, EPS_PIXEL_FRACTION = 1e-4, .5      # epsilon = max(0.1 mm, 0.5 z / f_px)
NEAR_PX, FAR_PX = 2., 10.
RUN_BREAK_PX = 2.                               # as frgb/trace_labels.py
VERTICAL_UP_SWITCH = 0.99                       # as frgb/raster.py


# ------------------------------------------------------------------ camera (as frgb/raster.py and frgb/trace_labels.py)

def camera(shot, resolution):
    pos = np.asarray(shot['position_enu_m'], float)
    fwd = np.asarray(shot['target_enu_m'], float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0., 0., 1.] if abs(fwd[2]) <= VERTICAL_UP_SWITCH else [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    w, h = resolution
    fw, fh = shot['filmback_mm']
    if abs(fw / fh - w / h) > 1e-6:
        raise ValueError('filmback aspect differs from resolution aspect')
    near = float(shot.get('near_clip_cm', 10.)) / 100.
    return dict(pos=pos, fwd=fwd, right=right, up=up, fpx=shot['focal_length_mm'] / fw * w, w=int(w), h=int(h), near=near)


def project(points, cam):
    r = np.asarray(points, float) - cam['pos']
    z = r @ cam['fwd']
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.column_stack([cam['w'] / 2 + cam['fpx'] * (r @ cam['right']) / z - .5,
                              cam['h'] / 2 - cam['fpx'] * (r @ cam['up']) / z - .5])
    return uv, z


def in_frame(uv, cam):
    return ((uv[:, 0] >= -.5) & (uv[:, 0] < cam['w'] - .5) & (uv[:, 1] >= -.5) & (uv[:, 1] < cam['h'] - .5))


# ------------------------------------------------------------------ pure functions

def face_plane_distance(P, face, V, F):
    """distance (m) of every sample to the plane of its triangle and the unit normal of that triangle by vertex winding; inf / nan
    for an index out of range or a degenerate triangle"""
    P = np.asarray(P, float)
    dist = np.full(len(P), np.inf)
    normal = np.full((len(P), 3), np.nan)
    centroid_dist = np.full(len(P), np.inf)
    ok = (face >= 0) & (face < len(F)) & np.isfinite(P).all(1)
    tri = V[F[face[ok]]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    good = length > 0
    n[good] /= length[good, None]
    d = np.abs(np.einsum('ij,ij->i', P[ok] - tri[:, 0], n))
    d[~good] = np.inf
    n[~good] = np.nan
    dist[ok] = d
    normal[ok] = n
    centroid_dist[ok] = np.linalg.norm(P[ok] - tri.mean(1), axis=1)
    return dist, normal, centroid_dist


def distance_to_evidence(target, u_line, occluded_line, opening_class, pixel_mm):
    """every occluded-line pixel: Euclidean distance (pixel centres, in pixels) to the nearest pixel of the visible evidence M"""
    M = (opening_class > 0) | (target > 0) | (u_line > 0)
    O = occluded_line > 0
    n_o, n_m = int(O.sum()), int(M.sum())
    out = dict(occluded_line_px=n_o, evidence_px=n_m, occluded_in_evidence_px=int((O & M).sum()), pixel_mm=float(pixel_mm),
               threshold_2px_mm=NEAR_PX * float(pixel_mm), threshold_10px_mm=FAR_PX * float(pixel_mm))
    if n_o == 0:
        out.update(frac_d_le_2=None, frac_d_2_10=None, frac_d_gt_10=None, median_d_px=None, median_d_mm=None,
                   note='no occluded-line pixels')
        return out, None
    d = ndi.distance_transform_edt(~M)[O] if n_m > 0 else np.full(n_o, np.inf)
    med = float(np.median(d))
    out.update(frac_d_le_2=float((d <= NEAR_PX).mean()),
               frac_d_2_10=float(((d > NEAR_PX) & (d <= FAR_PX)).mean()),
               frac_d_gt_10=float((d > FAR_PX).mean()),
               median_d_px=med if np.isfinite(med) else None,
               median_d_mm=med * float(pixel_mm) if np.isfinite(med) else None)
    return out, d


def pcts(x, q=(50, 99, 100)):
    x = np.asarray(x, float)
    return [float(v) for v in np.percentile(x, q)] if len(x) else None


def label_weights(uv, trace, member):
    """image length per sample as in the label program: two successive rows of the same trace that both belong to member and are a
    finite distance of at most 2 px apart share that distance half and half"""
    with np.errstate(invalid='ignore'):
        step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
        ok = np.isfinite(step) & (step <= RUN_BREAK_PX) & member[1:] & member[:-1] & (trace[1:] == trace[:-1])
    step = np.where(ok, step, 0.)
    return np.r_[step, 0.] / 2 + np.r_[0., step] / 2


def strike_azimuth_deg(P):
    """azimuth (clockwise from north, folded into [0, 180)) of the first principal axis of the horizontal sample coordinates"""
    xy = P[:, :2] - P[:, :2].mean(0)
    sxx, syy, sxy = float((xy[:, 0] ** 2).sum()), float((xy[:, 1] ** 2).sum()), float((xy[:, 0] * xy[:, 1]).sum())
    ang = .5 * np.arctan2(2. * sxy, sxx - syy)             # angle of the principal axis of the 2 x 2 covariance to east
    return float((90. - np.degrees(ang)) % 180.)


def two_strike_groups(az):
    """az: {trace: azimuth (degrees, period 180)}. The traces sorted by azimuth are cut at the two largest circular gaps into at
    most two groups."""
    keys = sorted(az, key=lambda k: (az[k], k))
    if len(keys) <= 1:
        return [keys] if keys else []
    a = np.array([az[k] for k in keys])
    gaps = np.r_[np.diff(a), a[0] + 180. - a[-1]]           # gaps[i]: gap after the i-th trace (the last one wraps around)
    cut = sorted(np.argsort(-gaps, kind='stable')[:2].tolist())
    g1 = keys[cut[0] + 1:cut[1] + 1]
    g2 = keys[cut[1] + 1:] + keys[:cut[0] + 1]
    return [g for g in (g1, g2) if g]


def circ_mean_180(vals):
    t = np.radians(2. * np.asarray(vals, float))
    return float((np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())) / 2.) % 180.)


def split_numbers(front, back, occluded, in_view, w):
    visible = in_view & ~occluded
    s = lambda m: float(w[m].sum())  # noqa: E731
    occ_w, in_w, vis_w = s(occluded), s(in_view), s(visible)
    fo, bo = s(occluded & front), s(occluded & back)
    n_fo, n_bo, n_occ, n_vis = int((occluded & front).sum()), int((occluded & back).sum()), int(occluded.sum()), int(visible.sum())
    return dict(b=bo / occ_w if occ_w > 0 else None,
                o_f=fo / (fo + vis_w) if fo + vis_w > 0 else None,
                front_occluded_share_of_in_frame=fo / in_w if in_w > 0 else None,
                b_equal_weight=n_bo / n_occ if n_occ else None, o_f_equal_weight=n_fo / (n_fo + n_vis) if n_fo + n_vis else None,
                in_frame_weight_px=in_w, occluded_weight_px=occ_w, visible_weight_px=vis_w,
                front_occluded_weight_px=fo, back_occluded_weight_px=bo,
                undefined_normal_occluded_weight_px=occ_w - fo - bo,
                front_visible_weight_px=s(visible & front), back_visible_weight_px=s(visible & back),
                occluded_ratio=occ_w / in_w if in_w > 0 else None)


# ------------------------------------------------------------------ one viewpoint

def run_view(frames, key, index):
    outcrop, image_set, sid, pid = VIEWS[key]
    e = index[key]
    lab = fio.label_dir(frames, key)
    spec = fio.render_spec(frames, key)
    cam = camera(spec['shot'], spec['resolution'])
    ground_z = float(spec['ground']['z_m'])
    report = fio.read_json(lab / 'report.json')
    rec = dict(outcrop=outcrop, image_set=image_set, block=e['block'], sid=sid, pid=pid, pixel_mm=float(e['pixel_mm']), ground_z_m=ground_z,
               **fio.view_angle(frames, key))

    # ---- 1 distance of the occluded line to the visible evidence
    a2, _ = distance_to_evidence(np.load(lab / 'target.npy'), np.load(lab / 'u_line.npy'), np.load(lab / 'occluded_line.npy'),
                                 np.load(lab / 'opening_class.npy'), e['pixel_mm'])
    fl = report.get('faults') or {}
    fault_occ = float(sum(f.get('occluded_px', 0.) for f in fl.get('faults', [])))
    joint_occ = float(sum(p_['occluded_px'] for p_ in report['per_path']))
    a2.update(report_faults_status=fl.get('status'), report_joint_occluded_px=joint_occ, report_fault_occluded_px=fault_occ,
              occluded_line_includes_fault_trace=bool(fault_occ > 0),
              fault_share_of_report_occluded_px=fault_occ / (fault_occ + joint_occ) if fault_occ + joint_occ > 0 else None)
    rec['distance'] = a2

    # ---- 2 samples in the row order of the label program, and the uncut rock surface
    sm = fio.samples(frames, key)
    P, face, trace = sm['point'], sm['face'], sm['trace']
    rec['samples'] = dict(rows=int(len(P)), rows_using_pre_refinement_point=sm['rows_using_pre_refinement_point'],
                          traces=[int(t) for t in np.unique(trace)], read_from=sm['source'])
    src = fio.read_json(fio.case_dir(frames, key) / 'build.json')['source']
    with np.load(Path(src) / MESH_NAME, allow_pickle=False) as a:
        V, F = a['vertices'], a['faces']
    with np.load(fio.point_dir(frames, key) / 'render' / 'frame.npz', allow_pickle=False) as a:
        D = np.asarray(a['depth'], float)
    if D.shape != (cam['h'], cam['w']):
        raise ValueError('%s: depth shape %s differs from the resolution' % (key, D.shape))

    # C1 face index
    dist, normal, centroid_dist = face_plane_distance(P, face, V, F)
    c1_frac = float((dist <= C1_DIST_M).mean())
    C1 = dict(mesh_faces=int(len(F)), mesh_vertices=int(len(V)), samples=int(len(P)),
              face_index_out_of_range=int(((face < 0) | (face >= len(F))).sum()),
              fraction_within_0p1mm=c1_frac, threshold=C1_MIN_FRACTION, passed=bool(c1_frac >= C1_MIN_FRACTION),
              plane_distance_m_p50_p99_max=pcts(dist[np.isfinite(dist)]),
              centroid_distance_m_p50=float(np.median(centroid_dist[np.isfinite(centroid_dist)])))

    # per sample: in frame, occluded (from the depth map)
    uv, z = project(P, cam)
    finite = np.isfinite(P).all(1) & (z > cam['near'])
    uvs = np.where(np.isfinite(uv), uv, -1e9)
    below = finite & (P[:, 2] < ground_z)
    in_img = finite & in_frame(uvs, cam)
    in_view = in_img & ~below
    ix = np.clip(np.floor(uvs[:, 0] + .5), 0, cam['w'] - 1).astype(np.int64)
    iy = np.clip(np.floor(uvs[:, 1] + .5), 0, cam['h'] - 1).astype(np.int64)
    d_px = D[iy, ix]
    eps = np.maximum(EPS_FLOOR_M, EPS_PIXEL_FRACTION * z / cam['fpx'])
    occluded = in_view & (d_px < z - eps)
    visible = in_view & ~occluded

    # lengths, and C3 against the label report
    w = label_weights(uv, trace, in_view)
    w_below = label_weights(uv, trace, in_img & below)
    pp = report['per_path']
    rep = dict(projected_in_frame_px=float(sum(p['projected_in_frame_px'] for p in pp)),
               occluded_px=float(sum(p['occluded_px'] for p in pp)),
               visible_px=float(sum(p['visible_px'] for p in pp)),
               visible_u_px=float(sum(p['visible_u_px'] for p in pp)),
               below_ground_px=float(sum((p.get('below_ground_px') or 0.) for p in pp)),
               own_occluded_px=float(report['occlusion']['joint']['own_occluded_px']),
               far_occluded_px=float(report['occlusion']['joint']['far_occluded_px']),
               faults_status=(report.get('faults') or {}).get('status'), per_path_rows=int(len(pp)))
    rep['occluded_ratio'] = rep['occluded_px'] / rep['projected_in_frame_px'] if rep['projected_in_frame_px'] > 0 else None
    in_w, occ_w, vis_w = float(w[in_view].sum()), float(w[occluded].sum()), float(w[visible].sum())
    ratio = occ_w / in_w if in_w > 0 else None
    diff = None if ratio is None or rep['occluded_ratio'] is None else 100. * abs(ratio - rep['occluded_ratio'])
    C3 = dict(in_frame_weight_px=in_w, occluded_weight_px=occ_w, visible_weight_px=vis_w, occluded_ratio=ratio,
              report_in_frame_px=rep['projected_in_frame_px'], report_occluded_px=rep['occluded_px'],
              report_occluded_ratio=rep['occluded_ratio'],
              in_frame_weight_rel_diff=in_w / rep['projected_in_frame_px'] - 1. if rep['projected_in_frame_px'] > 0 else None,
              occluded_ratio_minus_report_pp=None if diff is None else 100. * (ratio - rep['occluded_ratio']),
              diff_pp=diff, threshold_pp=C3_MAX_DIFF_PP, passed=None if diff is None else bool(diff <= C3_MAX_DIFF_PP),
              below_ground_weight_px=float(w_below[in_img & below].sum()), report_below_ground_px=rep['below_ground_px'])

    # facing direction by vertex winding; whether all viewpoints are reversed together is decided afterwards
    v = cam['pos'] - P
    v /= np.linalg.norm(v, axis=1)[:, None]
    ndotv = np.einsum('ij,ij->i', normal, v)
    defined = np.isfinite(ndotv)
    raw_front = defined & (ndotv > 0)
    raw_back = defined & ~raw_front
    c2_raw = float(w[visible & raw_front].sum()) / vis_w if vis_w > 0 else None
    c2_raw_flipped = float(w[visible & raw_back].sum()) / vis_w if vis_w > 0 else None

    az = {}
    for t in np.unique(trace):
        m = (trace == t) & np.isfinite(P).all(1)
        if m.sum() >= 2:
            az[int(t)] = strike_azimuth_deg(P[m])
    rec.update(C1=C1, C3=C3, report=rep, trace_strike_azimuth_deg={str(k): a for k, a in az.items()},
               C2_raw=dict(visible_front_fraction_raw_winding=c2_raw, visible_back_fraction_raw_winding=c2_raw_flipped,
                           undefined_normal_rows=int((~defined).sum())))
    rec['_arrays'] = dict(w=w, in_view=in_view, occluded=occluded, raw_front=raw_front, raw_back=raw_back, trace=trace, az=az,
                          sets=fio.joint_sets(key))
    return rec


def finish_view(rec, flip):
    """once the orientation is fixed: C2, C4 and the reported quantities; returns the record without arrays"""
    a = rec.pop('_arrays')
    w, in_view, occluded = a['w'], a['in_view'], a['occluded']
    front, back = (a['raw_back'], a['raw_front']) if flip else (a['raw_front'], a['raw_back'])
    visible = in_view & ~occluded
    c2r = rec.pop('C2_raw')
    frac = c2r['visible_back_fraction_raw_winding'] if flip else c2r['visible_front_fraction_raw_winding']
    rec['C2'] = dict(orientation_flipped=bool(flip), visible_front_fraction=frac, threshold=C2_MIN_FRONT,
                     passed=None if frac is None else bool(frac >= C2_MIN_FRONT), **c2r)
    back_in = float(w[back & in_view].sum())
    back_occ = float(w[back & occluded].sum())
    c4 = back_occ / back_in if back_in > 0 else None
    rec['C4'] = dict(back_in_frame_weight_px=back_in, back_occluded_weight_px=back_occ,
                     back_visible_weight_px=back_in - back_occ, back_in_frame_samples=int((back & in_view).sum()),
                     back_visible_samples=int((back & visible).sum()),
                     fraction_occluded=c4, threshold=C4_MIN_BACK_OCCLUDED,
                     passed=True if c4 is None else bool(c4 >= C4_MIN_BACK_OCCLUDED),
                     note='no in-frame sample faces away from the camera: the condition has no object and counts as met' if c4 is None else None)
    sp = split_numbers(front, back, occluded, in_view, w)
    s = lambda m: float(w[m].sum())  # noqa: E731
    per_trace = {}
    for t in np.unique(a['trace']):
        m = a['trace'] == t
        per_trace[str(int(t))] = dict(
            strike_azimuth_deg=a['az'].get(int(t)), in_frame_weight_px=s(m & in_view), visible_weight_px=s(m & visible),
            occluded_weight_px=s(m & occluded), front_occluded_weight_px=s(m & occluded & front),
            back_occluded_weight_px=s(m & occluded & back))
    groups = two_strike_groups(a['az'])
    rec['strike_groups'] = [dict(traces=[int(t) for t in g], mean_strike_azimuth_deg=circ_mean_180([a['az'][t] for t in g]))
                            for g in groups]
    for g in rec['strike_groups']:      # joint set of the traces of the group in the scene configuration
        g['joint_set_direction_deg'] = sorted({a['sets'][t]['mean_direction_deg'] for t in g['traces']})
    gates = [rec['C1']['passed'], rec['C2']['passed'], rec['C3']['passed'], rec['C4']['passed']]
    rec['all_conditions_met'] = bool(all(g is True for g in gates))
    rec['first_condition_not_met'] = next((n for n, g in zip(('C1', 'C2', 'C3', 'C4'), gates) if g is not True), None)
    rec['split'] = sp if rec['all_conditions_met'] else None
    rec['_split'], rec['_per_trace'] = sp, per_trace
    return rec


def decide_flip(views):
    raw = {k: r['C2_raw']['visible_front_fraction_raw_winding'] for k, r in views.items()}
    opp = {k: r['C2_raw']['visible_back_fraction_raw_winding'] for k, r in views.items()}
    if all(x is not None and x >= C2_MIN_FRONT for x in raw.values()):
        return False, 'consistent_raw'
    if all(x is not None and x >= C2_MIN_FRONT for x in opp.values()):
        return True, 'consistent_opposite_flipped_once'
    n_raw = sum(x is not None and x >= C2_MIN_FRONT for x in raw.values())
    n_opp = sum(x is not None and x >= C2_MIN_FRONT for x in opp.values())
    if n_opp == 0:
        return False, 'raw_some_views_fail'
    if n_raw == 0:
        return True, 'opposite_some_views_fail'
    return False, 'mixed'


# ------------------------------------------------------------------ retained length, per frame

def retained_length(frames, index, views):
    view_of = {(VIEWS[k][2], VIEWS[k][3].split('_')[1]): k for k in MULTI_LIGHT if k in views}
    rows, by_view = {}, {}
    for key in fio.aerial_frames(index):
        e = index[key]
        vk = view_of.get((e['sid'], e['pid'].split('_')[1]))
        if vk is None or not views[vk]['all_conditions_met']:
            continue
        vr = views[vk]
        rp = fio.read_json(fio.label_dir(frames, key) / 'report.json')
        pp = rp['per_path']
        Vis = float(sum(p['visible_px'] for p in pp))
        U = float(sum(p['visible_u_px'] for p in pp))
        geo = dict(projected_in_frame_px=float(sum(p['projected_in_frame_px'] for p in pp)),
                   occluded_px=float(sum(p['occluded_px'] for p in pp)), visible_px=Vis,
                   own_occluded_px=float(rp['occlusion']['joint']['own_occluded_px']),
                   far_occluded_px=float(rp['occlusion']['joint']['far_occluded_px']))
        dev = max(abs(geo[n] - vr['report'][n]) for n in geo)
        Fo = vr['_split']['front_occluded_weight_px']
        L_f, L_m = Vis + Fo, Vis - U
        row = dict(view=vk, sid=e['sid'], pid=e['pid'], outcrop=vr['outcrop'], report_visible_px=Vis, report_visible_u_px=U,
                   report_fault_visible_px=rp['per_type']['fault']['visible_px'],
                   front_occluded_px=Fo, L_f_px=L_f, L_m_px=L_m,
                   retained_share=L_m / L_f if L_f > 0 else None,
                   contribution_occlusion=Fo / L_f if L_f > 0 else None,
                   contribution_unclear=U / L_f if L_f > 0 else None,
                   geometry_max_abs_dev_from_view_frame_px=dev)
        gl = []
        for g in vr['strike_groups']:
            ids = set(g['traces'])
            rws = [p for p in pp if int(p['source_trace_id'].split('-')[1]) in ids]
            Vg = float(sum(p['visible_px'] for p in rws))
            Ug = float(sum(p['visible_u_px'] for p in rws))
            Fg = float(sum(vr['_per_trace'][str(t)]['front_occluded_weight_px'] for t in g['traces']))
            gl.append(dict(traces=g['traces'], mean_strike_azimuth_deg=g['mean_strike_azimuth_deg'],
                           joint_set_direction_deg=g['joint_set_direction_deg'], report_visible_px=Vg,
                           report_visible_u_px=Ug, front_occluded_px=Fg, L_f_px=Vg + Fg, L_m_px=Vg - Ug,
                           retained_share=(Vg - Ug) / (Vg + Fg) if Vg + Fg > 0 else None))
        row['by_strike_group'] = gl
        rows[key] = row
        by_view.setdefault(vk, []).append(key)
    rng = lambda x: [float(min(x)), float(np.median(x)), float(max(x))] if x else None  # noqa: E731
    summ = {}
    for vk, keys in by_view.items():
        rws = [rows[k] for k in keys]
        ng = len(views[vk]['strike_groups'])
        summ[vk] = dict(
            outcrop=views[vk]['outcrop'], frames=keys, n_frames=len(keys),
            L_f_px=rws[0]['L_f_px'], front_occluded_px=rws[0]['front_occluded_px'], report_visible_px=rws[0]['report_visible_px'],
            retained_share_min_median_max=rng([r['retained_share'] for r in rws]),
            contribution_occlusion=rws[0]['contribution_occlusion'],
            contribution_unclear_min_median_max=rng([r['contribution_unclear'] for r in rws]),
            geometry_max_abs_dev_px=float(max(r['geometry_max_abs_dev_from_view_frame_px'] for r in rws)),
            by_strike_group=[dict(traces=views[vk]['strike_groups'][i]['traces'],
                                  mean_strike_azimuth_deg=views[vk]['strike_groups'][i]['mean_strike_azimuth_deg'],
                                  joint_set_direction_deg=views[vk]['strike_groups'][i]['joint_set_direction_deg'],
                                  L_f_px=rws[0]['by_strike_group'][i]['L_f_px'],
                                  front_occluded_px=rws[0]['by_strike_group'][i]['front_occluded_px'],
                                  retained_share_min_median_max=rng([r['by_strike_group'][i]['retained_share'] for r in rws
                                                                     if r['by_strike_group'][i]['retained_share'] is not None]))
                             for i in range(ng)])
    return dict(definition='L_f = visible length of report.json + occluded length facing the camera (taken once per viewpoint); L_m = visible '
                           'length - unclear length (sum of per_path.visible_u_px); retained_share = L_m / L_f; 1 - retained_share = '
                           'contribution_occlusion + contribution_unclear',
                per_view=summ, per_frame=rows, n_frames=len(rows))


def selftest():
    # two triangles as a ridge: the left slope has a normal towards -x and up, the right slope towards +x and up (by winding)
    V = np.array([[0., 0., 1.], [0., 1., 1.], [-1., 0., 0.], [-1., 1., 0.], [1., 0., 0.], [1., 1., 0.]])
    F = np.array([[0, 1, 3], [0, 3, 2], [0, 4, 5], [0, 5, 1]])
    P = np.array([[-.5, .6, .5], [-.5, .3, .5], [.5, .6, .5], [.5, .3, .5], [.5, .3, .6]])
    face = np.array([0, 1, 2, 3, 3])
    dist, n, _ = face_plane_distance(P, face, V, F)
    assert np.allclose(dist[:4], 0, atol=1e-12) and abs(dist[4] - .1 / np.sqrt(2)) < 1e-12, dist
    assert (n[:2, 0] < 0).all() and (n[2:, 0] > 0).all() and (n[:, 2] > 0).all(), n
    # distance: a vertical evidence line at x = 10, occluded lines at x = 11, 15 and 30
    t = np.zeros((40, 40), bool); t[:, 10] = True
    o = np.zeros((40, 40), bool); o[:, 11] = True; o[:, 15] = True; o[:, 30] = True
    r, d = distance_to_evidence(t, np.zeros_like(t), o, np.zeros((40, 40), np.uint8), 4.)
    assert abs(r['frac_d_le_2'] - 1 / 3) < 1e-12 and abs(r['frac_d_2_10'] - 1 / 3) < 1e-12 and abs(r['frac_d_gt_10'] - 1 / 3) < 1e-12, r
    assert r['median_d_px'] == 5. and r['median_d_mm'] == 20., r
    # camera looking straight down at the origin
    cam = camera(dict(position_enu_m=[0, 0, 10], target_enu_m=[0, 0, 0], focal_length_mm=35., filmback_mm=[36., 36.]), [1024, 1024])
    uv, z = project(np.array([[0., 0., 0.]]), cam)
    assert np.allclose(uv, [[511.5, 511.5]]) and np.allclose(z, [10.]), (uv, z)
    uv = np.array([[0., 0.], [1., 0.], [2., 0.], [10., 0.], [11., 0.], [12., 0.], [np.nan, np.nan]])
    tr = np.array([1, 1, 1, 1, 1, 2, 2])
    mem = np.array([True, True, True, True, True, True, False])
    w = label_weights(uv, tr, mem)
    assert np.allclose(w, [.5, 1., .5, .5, .5, 0., 0.]), w
    mem2 = mem.copy(); mem2[1] = False
    assert np.allclose(label_weights(uv, tr, mem2), [0., 0., 0., .5, .5, 0., 0.])
    assert abs(strike_azimuth_deg(np.array([[0., 0., 0.], [1., 1., 0.], [2., 2., 5.]])) - 45.) < 1e-9
    g = two_strike_groups({1: 10., 2: 14., 3: 100., 4: 95., 5: 178.})
    assert sorted(map(sorted, g)) == [[1, 2, 5], [3, 4]], g
    assert two_strike_groups({7: 33.}) == [[7]]
    assert abs(circ_mean_180([178., 2.]) - 0.) < 1e-9 or abs(circ_mean_180([178., 2.]) - 180.) < 1e-9
    f = np.array([True, False, False]); b = np.array([False, True, False])
    s = split_numbers(f, b, np.array([True, True, False]), np.array([True, True, True]), np.array([1., 3., 4.]))
    assert abs(s['b'] - .75) < 1e-12 and abs(s['o_f'] - .2) < 1e-12 and abs(s['front_occluded_share_of_in_frame'] - .125) < 1e-12, s
    assert abs(s['o_f_equal_weight'] - .5) < 1e-12 and abs(s['b_equal_weight'] - .5) < 1e-12, s
    print('selftest PASS')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', help='folder with one sub-folder per frame (a0044 ...)')
    ap.add_argument('--index', default=str(fio.INDEX))
    ap.add_argument('--out', default=str(fio.RECORDS / 'occluded_segments.json'))
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if not args.frames:
        ap.error('--frames is required')
    t0 = time.time()
    index = fio.index(args.index)
    fio.check_viewpoints(index)
    views = {}
    for k in VIEWS:
        views[k] = run_view(args.frames, k, index)
    flip, flip_state = decide_flip(views)
    num = lambda x: float('nan') if x is None else x  # noqa: E731
    for k in VIEWS:
        r = finish_view(views[k], flip)
        print('%s C1 %.5f %s | C2 %.4f %s | C3 diff %.2f pp %s | C4 %.4f %s | b %.4f o_f %.4f | d<=2 %.3f d>10 %.3f median %.1f px' % (
            k, r['C1']['fraction_within_0p1mm'], r['C1']['passed'], num(r['C2']['visible_front_fraction']), r['C2']['passed'],
            num(r['C3']['diff_pp']), r['C3']['passed'], num(r['C4']['fraction_occluded']), r['C4']['passed'], num(r['_split']['b']),
            num(r['_split']['o_f']), num(r['distance']['frac_d_le_2']), num(r['distance']['frac_d_gt_10']), num(r['distance']['median_d_px'])),
            flush=True)
    kept = retained_length(args.frames, index, views)
    for vk, s in kept['per_view'].items():
        print('retained length %s (%s, %d frames): min/median/max %s; occlusion %.4f; unclear %s' % (
            vk, s['outcrop'], s['n_frames'], ['%.4f' % x for x in s['retained_share_min_median_max']], s['contribution_occlusion'],
            ['%.4f' % x for x in s['contribution_unclear_min_median_max']]))
    for r in views.values():
        r.pop('_split'), r.pop('_per_trace')
    with_fault = {k: dict(report_joint_occluded_px=v['distance']['report_joint_occluded_px'],
                          report_fault_occluded_px=v['distance']['report_fault_occluded_px'],
                          fault_share_of_report_occluded_px=v['distance']['fault_share_of_report_occluded_px'])
                  for k, v in views.items() if v['distance']['occluded_line_includes_fault_trace']}
    out = dict(
        description='Occluded segments at 11 viewpoints, one image each. distance: image distance from the occluded-line pixels to the nearest '
                    'visible evidence. facing: the four conditions for dividing the occluded length by the facing direction of the rock surface, '
                    'per viewpoint, and where they are all met the division itself (split). retained_length: share of the trace length facing the '
                    'camera that is visible and not unclear, for the 10 images of every viewpoint of the multi-illumination aerial set '
                    'that meets the conditions.',
        paper_items=['main text Section 4.2: 28.6% of the in-frame length occluded, 85.7% of it on rock facing away from the camera, 5.4% of the '
                     'length facing the camera occluded (viewpoint a0062, about 3 degrees between line of sight and rock-surface normal)',
                     'supplement S4: position and facing direction of the occluded segments (distance table; fault-trace shares; conditions; '
                     'b, o_f and o_f with equal weight)',
                     'supplement S4: apparent width, gaps and the trace length retained in the labels, last paragraph'],
        script='reproduce/paper1/analysis/occluded_segments.py',
        units=dict(d='image distance, pixels', pixel_mm='nominal pixel size, mm', weight_px='image length, pixels (lengths of the label program)',
                   b='occluded length on rock facing away from the camera / occluded length', o_f='occluded length facing the camera / (that '
                   'length + visible length)', theta_deg='angle between the line of sight and the rock-surface normal at the camera target, degrees',
                   strike_azimuth_deg='degrees clockwise from north, in [0, 180)',
                   joint_set_direction_deg='mean direction of the joint set(s) of the traces as written in the scene configuration (30 or 120 '
                                           'degrees, anticlockwise from east)'),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        conditions=dict(C1='at least 99% of the samples lie within 0.1 mm of the plane of their triangle of the uncut rock surface',
                        C2='at least 95% of the visible length faces the camera (dot product of triangle normal and direction to the camera > 0)',
                        C3='occluded / in-frame length reproduced from the depth map within 2 percentage points of report.json',
                        C4='at least 99% of the in-frame length facing away from the camera is occluded'),
        notes=['The samples lie on the centreline of the groove mouth; where the triangle normals inside a joint groove vary strongly, '
               'per-triangle normals do not represent the facing direction at the scale of the mouth. This is why C2 or C4 is not met at most '
               'viewpoints, and why the division is reported only where all four conditions are met.',
               'The occluded-line layer of a frame with a fault also holds the occluded fault traces (distance.fault_trace_in_occluded_line).',
               'The visible evidence leaves 2-4 px empty where visible and occluded parts of a trace meet, because the label layers remove '
               'the centreline pixels around such transitions.',
               'The groups of similar strike are formed per viewpoint by cutting the traces, sorted by strike azimuth, at the two largest '
               'circular gaps; a viewpoint with two traces therefore always has two groups.'],
        orientation_flip_state=flip_state,
        distance=dict(per_view={k: v.pop('distance') for k, v in views.items()}, fault_trace_in_occluded_line=with_fault),
        facing=dict(views_meeting_all_conditions=[k for k, v in views.items() if v['all_conditions_met']],
                    first_condition_not_met={k: v['first_condition_not_met'] for k, v in views.items() if not v['all_conditions_met']},
                    per_view=views),
        retained_length=kept)
    fio.write_json(args.out, out)
    print('wrote %s  %.1f s' % (Path(args.out).name, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
