"""Illumination sweep: brightness difference of the windows under every sun direction, and its statistics
(main text Section 5.2 and Fig. 6; supplement S4, "Illumination sweep and the shadow ratio" with its three tables).

Six cases (frames of block light_sweep in reproduce/paper1/sweeps.csv): at viewpoint V1 (flat outcrop 26120001, camera slot 0)
the baseline ls_v1b055 and four shape variants ls_v1w030, ls_v1w080, ls_v1d010, ls_v1d025; at viewpoint V2 (standalone outcrop
26130000, camera slot 1) the baseline ls_v2b055. Per case 72 sun directions (azimuth in steps of 15 degrees, elevations 20, 40
and 60 degrees); the two baselines also have 6 repeats with render seed 1.

The windows of the brightness check depend only on the geometry. For every sun direction the windows of the base frame of the
case (trace_labels/windows.json and the pixel memberships of window_pixels.npz) are reused and the brightness difference of each
window is recalculated by the rule of the label program: Rec. 709 luma of the 8-bit PNG; mean over the in-fracture pixels (band)
and over the adjacent-rock ring; a window is valid when it has band pixels and at least 4 ring pixels; it has a low brightness
difference when |band mean - ring mean| is below 2 grey levels.
  brightness difference (signed, "darkness") = ring mean - band mean; relative contrast = darkness / ring mean.
Geometry of a window, from the trace samples of the base frame that project onto its band pixels and are not hidden in the
rendered depth: local strike (principal direction of the sample points, horizontal azimuth), fracture-plane normal n (mean
mid-surface normal), trace tangent tau, a = n x tau turned to the outward side of the rock surface, image length.
  phi = angle between the sun azimuth and the local strike of the window (0-90 degrees).
  shadow ratio S = (d / w) |s.n| / (s.a), infinite when s.a <= 0; s is the unit vector towards the sun, d and w the nominal depth
  and mean opening of the case.
Statistics use the frames with render seed 0 whose mean rock brightness is at least 0.0764; bins with fewer than 30 windows are
flagged.
Checks: (W1) recalculating the base frame's own image reproduces its windows.json; (W2) three sun directions per viewpoint that
were generated independently with the whole chain (frames w2_v1_0 ... w2_v2_2) have the same windows as the base frame, the same
brightness differences on their own image and the same classification on the sweep image of the same sun direction.

    python reproduce/paper1/analysis/light_sweep_stats.py --frames FRAMES --sweep SWEEP

FRAMES holds the 6 base frames and the 6 whole-chain frames (written by `python -m fracgen frame`, see frames_io.py); SWEEP holds
the images written by light_sweep.py (SWEEP/<case>/<aAAAeEEsS>/frame.png and frame.npz). A sun direction that is missing in
SWEEP is looked up as the whole-chain frame FRAMES/<case>_<aAAAeEEsS>. Writes records/round3/light_sweep.json (per case: per-image
rows and the binned statistics) and records/round3/light_sweep_windows.npz (per case: brightness difference of every window under
every sun direction and the geometry of the windows, from which the binned statistics can be recomputed).
Run from the labels environment with the repository root as working directory or on PYTHONPATH (it imports frgb); needs NumPy,
SciPy and Pillow."""
import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import frames_io as fio  # noqa: E402

CASES = {            # case -> viewpoint, nominal mean opening w, nominal depth d, seed repeats
    'ls_v2b055': dict(view='V2', w=.055, d=.175, repeats=True),
    'ls_v1b055': dict(view='V1', w=.055, d=.175, repeats=True),
    'ls_v1w030': dict(view='V1', w=.030, d=.175, repeats=False),
    'ls_v1w080': dict(view='V1', w=.080, d=.175, repeats=False),
    'ls_v1d010': dict(view='V1', w=.055, d=.100, repeats=False),
    'ls_v1d025': dict(view='V1', w=.055, d=.250, repeats=False),
}
ORDER = ('ls_v2b055', 'ls_v1b055', 'ls_v1w030', 'ls_v1w080', 'ls_v1d010', 'ls_v1d025')
AZIMUTHS = tuple(range(0, 360, 15))
ELEVATIONS = (20, 40, 60)
REPEATS = ((0, 40), (90, 40), (180, 40), (270, 40), (45, 20), (45, 60))      # repeated with render seed 1
W2_LIGHTS = ((90, 20), (210, 40), (330, 60))                                 # generated with the whole chain as well
LUMA_MIN = 0.0764                    # clear-sky lower limit of the mean rock brightness
U_T, U_MIN_RING = 2., 4
THRESHOLDS = (2., 5., 10., 20.)
PHI_EDGES = (0, 15, 30, 45, 60, 75, 90)
S_EDGES = (0., .25, .5, .75, 1., 2., np.inf)
S_LABELS = ('0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1', '1-2', '>=2 (with infinite S)')
MIN_BIN = 30
GROUP_CENTRES_AZ = (60., 150.)       # strike azimuths of the two joint sets of the scene configuration (directions 30 and 120
#                                      degrees anticlockwise from east: azimuth = 90 - direction)
FRAMES = SWEEP = None                # set by main()


def tag_of(az, el, seed):
    return 'a%03de%02ds%d' % (int(az), int(el), int(seed))


def lights_of(case):
    L = [(az, el, 0) for el in ELEVATIONS for az in AZIMUTHS]
    if CASES[case]['repeats']:
        L += [(az, el, 1) for az, el in REPEATS]
    return L


def png(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert('RGB'))


def pkg():
    from frgb import trace_labels as tl
    from frgb.lighting_rules import CLIP_HIGH, CLIP_LOW, luma
    from frgb import raster
    return dict(tl=tl, luma=luma, CLIP_LOW=CLIP_LOW, CLIP_HIGH=CLIP_HIGH, raster=raster)


def image_of(case, az, el, seed):
    """folder that holds frame.png and frame.npz of one sun direction of a case, or None"""
    t = tag_of(az, el, seed)
    d = Path(SWEEP) / case / t
    if (d / 'frame.png').is_file() and (d / 'frame.npz').is_file():
        return d
    try:
        d = fio.point_dir(FRAMES, '%s_%s' % (case, t)) / 'render'
    except FileNotFoundError:
        return None
    return d if (d / 'frame.png').is_file() else None


def load_windows(labels):
    """windows.json and window_pixels.npz of a labelled frame"""
    labels = Path(labels)
    wins = fio.read_json(labels / 'windows.json')
    assert all(w['gid'] == i for i, w in enumerate(wins))
    with np.load(labels / 'window_pixels.npz') as z:
        band, ring = z['band'].copy(), z['ring'].copy()
    n = len(wins)
    bn = np.bincount(band[:, 1], minlength=n)
    rn = np.bincount(ring[:, 1], minlength=n)
    assert [int(x) for x in bn] == [w['band_pixels'] for w in wins] and [int(x) for x in rn] == [w['ring_pixels'] for w in wins], \
        'window_pixels.npz and windows.json differ in their pixel counts'
    return dict(wins=wins, band=band, ring=ring, n=n, bn=bn, rn=rn, valid=(bn > 0) & (rn >= U_MIN_RING))


def measure(png_path, W, P):
    """band mean, ring mean, brightness difference of the label rule and its decision for every window on one image"""
    rgb = png(png_path)
    L = (P['luma'](rgb) * 255.).ravel()
    with np.errstate(invalid='ignore', divide='ignore'):
        bm = np.bincount(W['band'][:, 1], weights=L[W['band'][:, 0]], minlength=W['n']) / W['bn']
        rm = np.bincount(W['ring'][:, 1], weights=L[W['ring'][:, 0]], minlength=W['n']) / W['rn']
        c = np.abs(bm - rm)
        u = (W['rn'] < U_MIN_RING) | (W['bn'] == 0) | ~(c >= U_T)
    return dict(band_mean=bm, ring_mean=rm, contrast=c, u=u, rgb=rgb)


def exposure(rgb, mask, P):
    rock = mask == 1
    Lm = P['luma'](rgb)[rock]
    px = rgb[rock]
    return dict(mean_luma=float(Lm.mean()), clipped=float(((px <= P['CLIP_LOW']) | (px >= P['CLIP_HIGH'])).any(1).mean()), rock_pixels=int(rock.sum()))


def compare_windows(W, m, wins_ref, tol=1e-9):
    """recalculated windows against the rows of a windows.json"""
    if len(wins_ref) != W['n']:
        return dict(same_count=False, n=W['n'], n_ref=len(wins_ref), passed=False)
    cref = np.array([np.nan if w['contrast'] is None else w['contrast'] for w in wins_ref], float)
    uref = np.array([w['u'] for w in wins_ref], bool)
    both = np.isfinite(cref) & np.isfinite(m['contrast'])
    nan_same = bool((np.isfinite(cref) == np.isfinite(m['contrast'])).all())
    d = np.abs(cref - m['contrast'])[both]
    return dict(same_count=True, n=W['n'], nan_pattern_same=nan_same, max_abs_contrast_diff=float(d.max()) if len(d) else 0., windows_over_tol=int((d > tol).sum()),
                decisions_differ=int((uref != m['u']).sum()), tol=tol,
                passed=bool(nan_same and (not len(d) or d.max() <= tol) and (uref == m['u']).all()))


def compare_images(a_png, b_png, a_npz, b_npz):
    """count of differing 8-bit PNG values; whether depth, material index and region mask are bit-identical"""
    a, b = png(a_png).astype(np.int16), png(b_png).astype(np.int16)
    d = np.abs(a - b)
    out = dict(png_shape=list(a.shape), channel_values=int(d.size), diff1=int((d == 1).sum()), diff2plus=int((d >= 2).sum()), max_diff=int(d.max()),
               diff1_share=float((d == 1).mean()), png_identical=bool((d == 0).all()))
    with np.load(a_npz) as za, np.load(b_npz) as zb:
        for k in ('depth', 'index', 'mask'):
            out[k + '_bitwise_equal'] = bool(za[k].dtype == zb[k].dtype and za[k].shape == zb[k].shape and za[k].tobytes() == zb[k].tobytes())
        out['rgb_linear_max_abs_diff'] = float(np.abs(za['rgb'].astype(np.float64) - zb['rgb'].astype(np.float64)).max())
    return out


# ---------------------------------------------------------------- geometry of the windows
def circ_dist180(a, b):
    d = np.abs((np.asarray(a, float) - b) % 180.)
    return np.minimum(d, 180. - d)


def window_geometry(case, W, P):
    """Per window: strike, fracture-plane normal n, trace tangent tau, in-plane direction a towards the outside of the rock body,
    outward rock-surface normal N, image length. The samples are those of the label program for the base frame
    (trace_labels_state.pkl: refined 3-D points, slot_normal, host_face, inward). A sample belongs to a window when the pixel it
    projects to is a band pixel of a window of the same path and branch and the rendered depth there is not smaller than the
    view depth of the sample minus a tolerance (not hidden). Windows that receive fewer than 3 samples have no geometry and do
    not enter the statistics by phi and S (they are counted)."""
    tl = P['tl']
    q0 = fio.point_dir(FRAMES, case)
    st = pickle.loads((q0 / 'trace_labels_state.pkl').read_bytes())
    req = st['request']
    cam = P['raster'].camera(req['shots'][0], tuple(req['resolution']))
    S = st['samples']
    front = float(st['case_added']['front_sign'])
    geo = fio.read_json(q0 / 'geometry.json')
    src = Path(fio.read_json(Path(geo['case']) / 'build.json')['source']) / 'source_surface.npz'
    with np.load(src) as z:
        V, F = z['vertices'], z['faces']
    with np.load(q0 / 'render/frame.npz') as z:
        depth = np.asarray(z['depth'], float)
    h, w = depth.shape
    n = W['n']
    key_of = {}
    for g, row in enumerate(W['wins']):
        key_of.setdefault((row['path'], row.get('branch', 0)), []).append(g)
    band_lin, band_gid = W['band'][:, 0], W['band'][:, 1]
    acc = dict(pts=[[] for _ in range(n)], sn=np.zeros((n, 3)), inw=np.zeros((n, 3)), N=np.zeros((n, 3)), L=np.zeros(n), cnt=np.zeros(n, int))
    g1_dist, g2_dot = [], []
    reach, dip = [], []
    for key in sorted(S):
        r = S[key]
        uv, zc = tl.project(r['point'], cam)
        inside = tl.in_frame(uv, cam) & (zc > cam['near'])
        # straight, parallel-walled fracture: while the line of sight crosses one opening w along the plane normal n it advances
        # w |v.inward| / |v.n| along the depth axis; it reaches the floor when this is at least the depth d
        vv = r['point'][inside] - cam['pos']
        vv /= np.linalg.norm(vv, axis=1, keepdims=True)
        a_ = np.abs((vv * r['slot_normal'][inside]).sum(1))
        b_ = (vv * r['inward'][inside]).sum(1)
        reach.append(CASES[case]['w'] * np.abs(b_) / np.maximum(a_, 1e-9))
        dip.append(np.degrees(np.arccos(np.abs(r['slot_normal'][inside][:, 2]))))
        gids = key_of.get((r['path'], r['branch']))
        if not gids:
            continue
        sel = np.isin(band_gid, gids)
        lut = np.full(h * w, -1, np.int64)
        lut[band_lin[sel]] = band_gid[sel]
        px = np.floor(uv + .5).astype(np.int64)
        pxc = np.clip(px, 0, [w - 1, h - 1])
        lin = pxc[:, 1] * w + pxc[:, 0]
        eps = np.maximum(1e-3, 1.5 * zc / cam['fpx'])
        seen = inside & (depth[pxc[:, 1], pxc[:, 0]] >= zc - eps)
        gid = np.where(seen, lut[lin], -1)
        step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
        okstep = np.isfinite(step) & (step <= 2.) & inside[1:] & inside[:-1]          # lengths as in the label program (steps of at most 2 px)
        step = np.where(okstep, step, 0.)
        wt = np.r_[step, 0.] / 2 + np.r_[0., step] / 2
        tri = V[F[r['host_face']]].astype(float)
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-300
        fn *= front
        g1_dist.append(np.abs(((r['point'] - tri[:, 0]) * fn).sum(1)))
        v = cam['pos'] - r['point']
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        g2_dot.append(((fn * v).sum(1))[seen])
        for g in np.unique(gid[gid >= 0]):
            m = gid == g
            acc['pts'][g].append(r['point'][m])
            acc['sn'][g] += r['slot_normal'][m].sum(0)
            acc['inw'][g] += r['inward'][m].sum(0)
            acc['N'][g] += fn[m].sum(0)
            acc['L'][g] += wt[m].sum()
            acc['cnt'][g] += int(m.sum())
    out = dict(strike=np.full(n, np.nan), plane_strike=np.full(n, np.nan), n=np.full((n, 3), np.nan), tau=np.full((n, 3), np.nan), a=np.full((n, 3), np.nan),
               N=np.full((n, 3), np.nan), length_px=acc['L'], samples=acc['cnt'], a_vs_inward_cos=np.full(n, np.nan), tau_dot_n_raw=np.full(n, np.nan),
               surface_tilt_deg=np.full(n, np.nan), a_inward=np.full((n, 3), np.nan))
    # joint set of the source trace of every path (scene declaration: fracture_network.traces[].direction_group_id)
    rep = fio.read_json(q0 / 'trace_labels/report.json')
    trace_of_path = {int(r_['path']): r_['source_trace_id'] for r_ in rep['per_path']}
    fj = fio.read_json(fio.frame_dir(FRAMES, case) / 'frame.json')
    dec = fio.read_json(Path(fj['scene_root']) / 'scene/geom/declaration.json')
    net_tr = {t_['id']: t_ for t_ in dec['metadata']['paper_methods']['fracture_network']['traces']}
    del dec
    out['paths'] = {str(p_): dict(source_trace_id=t_, direction_group_id=net_tr.get(t_, {}).get('direction_group_id'),
                                  windows=int(sum(1 for row in W['wins'] if row['path'] == p_)))
                    for p_, t_ in sorted(trace_of_path.items()) if any(row['path'] == p_ for row in W['wins'])}
    out['report_visible_length_px'] = float(rep.get('visible_length_px') or 0.)
    for g in range(n):
        if acc['cnt'][g] < 3:
            continue
        Pn = np.concatenate(acc['pts'][g])
        nv = acc['sn'][g] / np.linalg.norm(acc['sn'][g])
        c = Pn - Pn.mean(0)
        _, sv, vt = np.linalg.svd(c, full_matrices=False)
        if sv[0] < 1e-6:
            continue
        t = vt[0]
        out['tau_dot_n_raw'][g] = abs(float(t @ nv))
        t = t - (t @ nv) * nv
        if np.linalg.norm(t) < 1e-6:
            continue
        t /= np.linalg.norm(t)
        Nv = acc['N'][g] / np.linalg.norm(acc['N'][g])
        a = np.cross(nv, t)
        a /= np.linalg.norm(a)
        if a @ Nv < 0:
            a = -a
        iw = acc['inw'][g] / np.linalg.norm(acc['inw'][g])
        out['n'][g], out['tau'][g], out['a'][g], out['N'][g] = nv, t, a, Nv
        out['a_inward'][g] = -iw
        out['a_vs_inward_cos'][g] = float(a @ (-iw))
        out['surface_tilt_deg'][g] = math.degrees(math.acos(max(-1., min(1., float(Nv[2])))))
        if math.hypot(t[0], t[1]) > 1e-6:
            out['strike'][g] = math.degrees(math.atan2(t[0], t[1])) % 180.
        ps = np.cross([0., 0., 1.], nv)
        if np.linalg.norm(ps) > 1e-6:
            out['plane_strike'][g] = math.degrees(math.atan2(ps[0], ps[1])) % 180.
    g1 = np.concatenate(g1_dist) if g1_dist else np.empty(0)
    g2 = np.concatenate(g2_dot) if g2_dot else np.empty(0)
    reach, dip = np.concatenate(reach), np.concatenate(dip)
    out['floor_visibility'] = dict(
        samples=int(len(reach)), share_floor_reachable=float((reach >= CASES[case]['d']).mean()),
        reach_m_quantiles_10_50_90=[float(x) for x in np.quantile(reach, [.1, .5, .9])],
        fracture_dip_deg_quantiles_10_50_90=[float(x) for x in np.quantile(dip, [.1, .5, .9])])
    out['checks'] = dict(samples_total=int(sum(len(S[k]['u']) for k in S)), windows=n, windows_with_geometry=int(np.isfinite(out['strike']).sum()),
                         valid_windows_without_geometry=int((W['valid'] & ~np.isfinite(out['strike'])).sum()),
                         sample_to_face_plane_le_0p1mm_share=float((g1 <= 1e-4).mean()) if len(g1) else None,
                         seen_samples_facing_camera_share=float((g2 > 0).mean()) if len(g2) else None,
                         window_length_px_quantiles=[float(x) for x in np.quantile(acc['L'][acc['cnt'] >= 3], [0., .1, .5, .9, 1.])] if (acc['cnt'] >= 3).any() else None,
                         total_window_length_px=float(acc['L'].sum()), report_visible_length_px=out['report_visible_length_px'],
                         a_vs_minus_inward_cos_quantiles=[float(x) for x in np.nanquantile(out['a_vs_inward_cos'], [0., .05, .5])] if np.isfinite(out['a_vs_inward_cos']).any() else None,
                         tau_dot_n_before_projection_p50_p95=[float(x) for x in np.nanquantile(out['tau_dot_n_raw'], [.5, .95])] if np.isfinite(out['tau_dot_n_raw']).any() else None,
                         surface_tilt_deg_quantiles=[float(x) for x in np.nanquantile(out['surface_tilt_deg'], [.05, .5, .95])] if np.isfinite(out['surface_tilt_deg']).any() else None)
    return out


def sun_vec(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def phi_of(strike, sun_az):
    return circ_dist180(strike, sun_az % 180.)                   # 0-90: 0 light along strike, 90 light across strike


def S_of(G, az, el, d, w, akey='a'):
    s = sun_vec(az, el)
    sn = np.abs(G['n'] @ s)
    sa = G[akey] @ s
    with np.errstate(invalid='ignore', divide='ignore'):
        S = np.where(sa > 0, (d / w) * sn / sa, np.inf)
    S[~np.isfinite(G['a'][:, 0])] = np.nan
    return S


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return None
    r = spearmanr(x, y)
    return float(r[0]) if np.isfinite(r[0]) else None


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


# ---------------------------------------------------------------- checks W1, W2
def check_w1(case, W, P):
    m = measure(fio.point_dir(FRAMES, case) / 'render/frame.png', W, P)
    return compare_windows(W, m, W['wins'])


def check_w2(view, base_case, W, P):
    """the 3 whole-chain frames of a viewpoint (geometry rebuilt independently, sweep light) against the windows of the base frame"""
    out = dict(frames={})
    for k, (az, el) in enumerate(W2_LIGHTS):
        name = 'w2_%s_%d' % (view.lower(), k)
        rec = dict(light=dict(azimuth_deg=az, elevation_deg=el))
        try:
            lab = fio.label_dir(FRAMES, name)
        except FileNotFoundError:
            lab = None
        if lab is None or not (lab / 'windows.json').is_file():
            rec.update(available=False, passed=False)
            out['frames'][name] = rec
            continue
        W2 = load_windows(lab)
        fj = fio.read_json(fio.frame_dir(FRAMES, name) / 'frame.json')
        assert fj['light'] == dict(azimuth_deg=float(az), elevation_deg=float(el)), fj['light']
        strip = lambda rows: [{k_: v for k_, v in r.items() if k_ not in ('contrast', 'u')} for r in rows]  # noqa: E731
        same_rows = strip(W2['wins']) == strip(W['wins'])
        same_px = bool(same_rows and all(W2[k_].shape == W[k_].shape and (W2[k_] == W[k_]).all() for k_ in ('band', 'ring')))
        rec['window_set'] = dict(count_base=W['n'], count_w2=W2['n'], rows_equal_without_contrast=bool(same_rows), pixel_memberships_equal=same_px)
        own_png = fio.point_dir(FRAMES, name) / 'render/frame.png'
        # (a) the windows of the base frame recalculated on the image of the whole-chain frame itself
        rec['on_w2_image'] = compare_windows(W, measure(own_png, W, P), W2['wins'])
        # (b) the same windows recalculated on the sweep image of the same sun direction
        d = image_of(base_case, az, el, 0)
        if d is not None:
            mine = d / 'frame.png'
            rec['on_sweep_image'] = compare_windows(W, measure(mine, W, P), W2['wins'])
            rec['sweep_image_vs_w2_image'] = compare_images(mine, own_png, mine.with_suffix('.npz'), own_png.with_suffix('.npz'))
        else:
            rec['on_sweep_image'] = None
        rec['passed_same_image'] = bool(same_px and rec['on_w2_image']['passed'])
        rec['passed_sweep_image_strict'] = bool(same_px and rec['on_sweep_image'] and rec['on_sweep_image']['passed'])
        rec['decisions_equal_on_sweep_image'] = bool(rec['on_sweep_image'] and rec['on_sweep_image'].get('decisions_differ') == 0)
        out['frames'][name] = rec
    fr = out['frames'].values()
    out['window_sets_equal'] = bool(all(r.get('window_set', {}).get('pixel_memberships_equal') for r in fr))
    out['pass_same_image'] = bool(all(r.get('passed_same_image') for r in fr))
    out['pass_sweep_image_strict'] = bool(all(r.get('passed_sweep_image_strict') for r in fr))
    out['decisions_equal_on_sweep_image'] = bool(all(r.get('decisions_equal_on_sweep_image') for r in fr))
    return out


# ---------------------------------------------------------------- statistics
def frac(mask, sel):
    n = int(sel.sum())
    return (float((mask & sel).sum() / n) if n else None), n


def phi_bins(T, v, low, key):
    pb = {}
    for el in ELEVATIONS:
        pb[str(el)] = []
        for i in range(6):
            hi = T[key] <= PHI_EDGES[i + 1] if i == 5 else T[key] < PHI_EDGES[i + 1]
            sel = v & (T['el'] == el) & np.isfinite(T[key]) & (T[key] >= PHI_EDGES[i]) & hi
            f_, n_ = frac(low, sel)
            pb[str(el)].append(dict(phi_bin='%d-%d' % (PHI_EDGES[i], PHI_EDGES[i + 1]), windows=n_, low_contrast_share=f_, median_darkness=med(T['darkness'][sel]),
                                    median_relative_contrast=med(T['rel'][sel])))
    r1 = {}
    for el in ELEVATIONS:
        a_, b_ = pb[str(el)][0], pb[str(el)][5]
        okn = a_['windows'] >= MIN_BIN and b_['windows'] >= MIN_BIN
        r1[str(el)] = dict(along_phi_lt15=a_['low_contrast_share'], across_phi_ge75=b_['low_contrast_share'], windows=[a_['windows'], b_['windows']],
                           diff_along_minus_across=(a_['low_contrast_share'] - b_['low_contrast_share']) if okn else None, enough_windows=bool(okn),
                           median_darkness_along=a_['median_darkness'], median_darkness_across=b_['median_darkness'])
    return dict(bins=pb, by_elevation=r1)


def S_bins(T, v, low, key):
    sv = v & ~np.isnan(T[key])
    bins = []
    for i in range(6):
        sel = sv & (T[key] >= S_EDGES[i]) & ((T[key] < S_EDGES[i + 1]) if i < 5 else True)
        f_, n_ = frac(low, sel)
        bins.append(dict(S_bin=S_LABELS[i], windows=n_, median_darkness=med(T['darkness'][sel]), median_abs_contrast=med(T['contrast'][sel]),
                         median_relative_contrast=med(T['rel'][sel]), low_contrast_share=f_, infinite_S=int((sel & np.isinf(T[key])).sum()), enough_windows=bool(n_ >= MIN_BIN),
                         median_band_mean=med(T['band_mean'][sel]), median_ring_mean=med(T['ring_mean'][sel])))
    hi, lo = sv & (T[key] >= 1.), sv & (T[key] < .25)
    mh, ml = med(T['darkness'][hi]), med(T['darkness'][lo])
    enough = int(hi.sum()) >= MIN_BIN and int(lo.sum()) >= MIN_BIN
    lt1 = sv & (T[key] < 1.)
    rho = spearman(T[key][lt1], T['darkness'][lt1]) if int(lt1.sum()) >= MIN_BIN else None
    fin = hi & np.isfinite(T[key])
    return dict(bins=bins, windows_with_S=int(sv.sum()), valid_windows_without_S=int((v & np.isnan(T[key])).sum()),
                S_ge1=dict(windows=int(hi.sum()), median_darkness=mh, median_relative_contrast=med(T['rel'][hi]), low_contrast_share=frac(low, hi)[0]),
                S_lt025=dict(windows=int(lo.sum()), median_darkness=ml, median_relative_contrast=med(T['rel'][lo]), low_contrast_share=frac(low, lo)[0]),
                ratio_ge1_over_lt025=(mh / ml) if (enough and ml is not None and ml > 0) else None, enough_windows=bool(enough),
                S_lt1=dict(windows=int(lt1.sum()), spearman_rho_darkness_vs_S=rho,
                           spearman_rho_abs_contrast_vs_S=spearman(T[key][lt1], T['contrast'][lt1]) if int(lt1.sum()) >= MIN_BIN else None,
                           spearman_rho_relative_contrast_vs_S=spearman(T[key][lt1], T['rel'][lt1]) if int(lt1.sum()) >= MIN_BIN else None),
                S_ge1_finite_spearman_rho_darkness_vs_S=spearman(T[key][fin], T['darkness'][fin]) if int(fin.sum()) >= MIN_BIN else None,
                by_elevation={str(el): dict(S_ge1=dict(windows=int((hi & (T['el'] == el)).sum()), median_darkness=med(T['darkness'][hi & (T['el'] == el)])),
                                            S_lt025=dict(windows=int((lo & (T['el'] == el)).sum()), median_darkness=med(T['darkness'][lo & (T['el'] == el)]))) for el in ELEVATIONS})


def strike_class_difference(frames, sizes):
    gfr = [f for f in frames if f['seed'] == 0 and not f['below_luma_limit'] and f['strike_classes']['length_share_diff'] is not None]
    if not gfr or min(sizes) < MIN_BIN:
        return dict(available=False, windows_g0=int(sizes[0]), windows_g1=int(sizes[1]), reason='a class has fewer than %d windows' % MIN_BIN)
    dl = [abs(f['strike_classes']['length_share_diff']) for f in gfr]
    dw = [abs(f['strike_classes']['window_share_diff']) for f in gfr]
    k, kw = int(np.argmax(dl)), int(np.argmax(dw))
    return dict(available=True, windows_g0=int(sizes[0]), windows_g1=int(sizes[1]), max_abs_length_share_diff=dl[k], at=gfr[k]['tag'],
                signed_at_max_g0_minus_g1=gfr[k]['strike_classes']['length_share_diff'], shares_at_max=gfr[k]['strike_classes'],
                max_abs_window_share_diff=dw[kw], window_at=gfr[kw]['tag'], median_abs_length_share_diff=med(dl), min_abs_length_share_diff=float(min(dl)))


def analyze_case(case, P):
    info = CASES[case]
    W = load_windows(fio.label_dir(FRAMES, case))
    G = window_geometry(case, W, P)
    q0 = fio.point_dir(FRAMES, case)
    with np.load(q0 / 'render/frame.npz') as z:
        base_mask = z['mask'].copy()
    n, valid = W['n'], W['valid']
    geom = np.isfinite(G['strike'])
    c0, c1 = GROUP_CENTRES_AZ
    d0, d1 = circ_dist180(G['strike'], c0), circ_dist180(G['strike'], c1)
    group = np.full(n, -1)
    group[geom] = np.where(d0[geom] <= d1[geom], 0, 1)
    sizes = (int((group == 0).sum()), int((group == 1).sum()))
    Lw = G['length_px']
    rec = dict(case=case, view=info['view'], width_m=info['w'], depth_m=info['d'], windows=n, valid_windows=int(valid.sum()),
               undersampled_windows=int((~valid).sum()), geometry=G['checks'], floor_visibility_straight_slot=G['floor_visibility'],
               strike_groups=dict(centres_azimuth_deg=[c0, c1], config_directions_deg=[30., 120.],
                                  by_window_strike=dict(windows_g0=sizes[0], windows_g1=sizes[1], length_px_g0=float(Lw[group == 0].sum()),
                                                        length_px_g1=float(Lw[group == 1].sum())),
                                  visible_paths=G['paths'],
                                  window_strike_hist_15deg=[int(x) for x in np.histogram(G['strike'][geom], bins=np.arange(0, 181, 15))[0]]))
    rec['check_W1'] = check_w1(case, W, P)
    frames, rows, missing = [], [], []
    for az, el, seed in lights_of(case):
        d = image_of(case, az, el, seed)
        if d is None:
            missing.append(tag_of(az, el, seed))
            continue
        m = measure(d / 'frame.png', W, P)
        with np.load(d / 'frame.npz') as z:
            mask = z['mask']
            mask_same = bool((mask == base_mask).all())
        ex = exposure(m['rgb'], mask, P)
        dark = m['ring_mean'] - m['band_mean']
        with np.errstate(invalid='ignore', divide='ignore'):
            relc = dark / m['ring_mean']
        low = valid & ~(m['contrast'] >= U_T)
        nv = max(int(valid.sum()), 1)
        fr = dict(tag=tag_of(az, el, seed), azimuth_deg=az, elevation_deg=el, seed=seed, mean_luma=ex['mean_luma'], clipped=ex['clipped'],
                  below_luma_limit=bool(ex['mean_luma'] < LUMA_MIN), mask_equal_to_base=mask_same, windows=n, u_windows=int(m['u'].sum()),
                  u_ratio=float(m['u'].mean()), low_contrast_windows=int(low.sum()), undersampled_windows=int((~valid).sum()),
                  low_contrast_share_of_valid=float(low.sum() / nv), low_contrast_share_of_all=float(low.sum() / n), undersampled_share_of_all=float((~valid).sum() / n),
                  median_darkness_levels=med(dark[valid]), median_relative_contrast=med(relc[valid]),
                  below_threshold_share_abs={str(int(t)): float((valid & ~(m['contrast'] >= t)).sum() / nv) for t in THRESHOLDS},
                  negative_darkness_share=float((valid & (dark < 0)).sum() / nv))
        sh = {}
        for gi in (0, 1):
            sel = group == gi
            tot = float(Lw[sel].sum())
            sh['g%d_length_share' % gi] = float(Lw[sel & ~m['u']].sum() / tot) if tot > 0 else None
            sh['g%d_window_share' % gi] = float((sel & ~m['u']).sum() / int(sel.sum())) if sel.any() else None
        sh['length_share_diff'] = None if sh['g0_length_share'] is None or sh['g1_length_share'] is None else float(sh['g0_length_share'] - sh['g1_length_share'])
        sh['window_share_diff'] = None if sh['g0_window_share'] is None or sh['g1_window_share'] is None else float(sh['g0_window_share'] - sh['g1_window_share'])
        fr['strike_classes'] = sh
        frames.append(fr)
        sv_ = sun_vec(az, el)
        rows.append(dict(sn_up=(G['n'] * np.where(G['n'][:, 2:3] >= 0, 1., -1.)) @ sv_,
                         az=np.full(n, az), el=np.full(n, el), seed=np.full(n, seed), gid=np.arange(n), contrast=m['contrast'], darkness=dark, rel=relc,
                         valid=valid.copy(), u=m['u'], phi=phi_of(G['strike'], az), S=S_of(G, az, el, info['d'], info['w']),
                         below=np.full(n, fr['below_luma_limit']), band_mean=m['band_mean'], ring_mean=m['ring_mean']))
    rec['frames_missing'] = missing
    rec['frames_rendered'] = len(frames)
    rec['frames'] = frames
    if not frames:
        return rec, None, None
    T = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    rec['frames_below_luma_limit'] = [f['tag'] for f in frames if f['below_luma_limit']]
    rec['mask_equal_to_base_all'] = bool(all(f['mask_equal_to_base'] for f in frames))
    use = (T['seed'] == 0) & ~T['below']                            # statistics: render seed 0, bright enough
    v = use & T['valid']
    low = ~(T['contrast'] >= U_T)
    rec['rule_phi'] = phi_bins(T, v, low, 'phi')
    # sun elevation: fraction of valid windows with a low brightness difference per image (median over the 24 azimuths)
    fsel = lambda el: [f for f in frames if f['seed'] == 0 and f['elevation_deg'] == el and not f['below_luma_limit']]  # noqa: E731
    el_med = {str(el): med([f['low_contrast_share_of_valid'] for f in fsel(el)]) for el in ELEVATIONS}
    el_rng = {str(el): ([float(min(f['low_contrast_share_of_valid'] for f in fsel(el))), float(max(f['low_contrast_share_of_valid'] for f in fsel(el)))] if fsel(el) else None) for el in ELEVATIONS}
    rec['rule_elevation'] = dict(median_low_contrast_share_of_valid=el_med, range_over_azimuths=el_rng, frames_used={str(el): len(fsel(el)) for el in ELEVATIONS},
                                 median_low_contrast_share_of_all={str(el): med([f['low_contrast_share_of_all'] for f in fsel(el)]) for el in ELEVATIONS},
                                 median_u_ratio_all_windows={str(el): med([f['u_ratio'] for f in fsel(el)]) for el in ELEVATIONS},
                                 median_darkness_levels={str(el): med([f['median_darkness_levels'] for f in fsel(el)]) for el in ELEVATIONS})
    rec['rule_groups'] = strike_class_difference(frames, sizes)
    f0 = [f for f in frames if f['seed'] == 0 and not f['below_luma_limit']]
    rec['thresholds'] = dict(
        abs_rule={str(int(t)): dict(overall=frac(~(T['contrast'] >= t), v)[0],
                                    by_elevation={str(el): frac(~(T['contrast'] >= t), v & (T['el'] == el))[0] for el in ELEVATIONS},
                                    per_frame_range=[float(min(f['below_threshold_share_abs'][str(int(t))] for f in f0)), float(max(f['below_threshold_share_abs'][str(int(t))] for f in f0))] if f0 else None)
                  for t in THRESHOLDS},
        valid_window_instances=int(v.sum()))
    rec['continuous'] = dict(median_darkness_levels=med(T['darkness'][v]), median_relative_contrast=med(T['rel'][v]),
                             darkness_quantiles_5_25_50_75_95=[float(x) for x in np.quantile(T['darkness'][v], [.05, .25, .5, .75, .95])] if v.any() else None,
                             negative_darkness_share=frac(T['darkness'] < 0, v)[0],
                             by_elevation={str(el): dict(median_darkness=med(T['darkness'][v & (T['el'] == el)]), median_relative_contrast=med(T['rel'][v & (T['el'] == el)])) for el in ELEVATIONS})
    rec['S'] = S_bins(T, v, low, 'S')
    # by the side of the fracture plane the sunlight comes from: s . n_up, the component of the sun direction along the upward
    # fracture-plane normal (positive: the sun shines on the upward-facing wall)
    edges = (-1.0001, -.5, -.25, 0., .25, .5, 1.0001)
    side = []
    for i in range(6):
        sel = v & np.isfinite(T['sn_up']) & (T['sn_up'] >= edges[i]) & (T['sn_up'] < edges[i + 1])
        side.append(dict(s_dot_n_up='%g..%g' % (round(edges[i], 2), round(edges[i + 1], 2)), windows=int(sel.sum()), median_band_mean=med(T['band_mean'][sel]),
                         median_ring_mean=med(T['ring_mean'][sel]), median_darkness=med(T['darkness'][sel]), median_relative_contrast=med(T['rel'][sel]),
                         low_contrast_share=frac(low, sel)[0]))
    sel = v & np.isfinite(T['sn_up'])
    rec['sun_side'] = dict(
        bins=side, spearman_rho_band_mean_vs_s_dot_n_up=spearman(T['sn_up'][sel], T['band_mean'][sel]) if int(sel.sum()) >= MIN_BIN else None,
        spearman_rho_darkness_vs_s_dot_n_up=spearman(T['sn_up'][sel], T['darkness'][sel]) if int(sel.sum()) >= MIN_BIN else None,
        fracture_normal_up_mean=[float(x) for x in np.nanmean(G['n'] * np.where(G['n'][:, 2:3] >= 0, 1., -1.), axis=0)])
    if info['repeats']:
        pairs = []
        for az, el in REPEATS:
            a = (T['az'] == az) & (T['el'] == el) & (T['seed'] == 0)
            b = (T['az'] == az) & (T['el'] == el) & (T['seed'] == 1)
            if a.sum() == n and b.sum() == n:
                ua, ub = T['u'][a], T['u'][b]
                dc = np.abs(T['contrast'][a] - T['contrast'][b])
                pairs.append(dict(azimuth_deg=az, elevation_deg=el, flipped_windows=int((ua != ub).sum()), flipped_share_of_all=float((ua != ub).mean()),
                                  flipped_share_of_valid=float((ua != ub)[valid].sum() / max(int(valid.sum()), 1)),
                                  max_abs_contrast_diff=float(np.nanmax(dc[valid])) if valid.any() else None, median_abs_contrast_diff=med(dc[valid])))
        rec['seed_repeats'] = dict(pairs=pairs, pairs_done=len(pairs), max_flipped_share_of_all=max([p['flipped_share_of_all'] for p in pairs], default=None),
                                   total_flipped=int(sum(p['flipped_windows'] for p in pairs)), total_windows=int(n * len(pairs)),
                                   pooled_flipped_share=float(sum(p['flipped_windows'] for p in pairs) / max(n * len(pairs), 1)) if pairs else None)
    rec['per_frame_summary'] = dict(frames_used=len(f0), u_ratio_range=[min(f['u_ratio'] for f in f0), max(f['u_ratio'] for f in f0)] if f0 else None,
                                    u_ratio_median=med([f['u_ratio'] for f in f0]),
                                    low_contrast_share_of_all_range=[min(f['low_contrast_share_of_all'] for f in f0), max(f['low_contrast_share_of_all'] for f in f0)] if f0 else None,
                                    low_contrast_share_of_valid_range=[min(f['low_contrast_share_of_valid'] for f in f0), max(f['low_contrast_share_of_valid'] for f in f0)] if f0 else None,
                                    median_darkness_levels_range=[min(f['median_darkness_levels'] for f in f0), max(f['median_darkness_levels'] for f in f0)] if f0 else None,
                                    undersampled_share_of_all=float((~valid).sum() / n),
                                    mean_luma_range=[min(f['mean_luma'] for f in frames), max(f['mean_luma'] for f in frames)],
                                    clipped_range=[min(f['clipped'] for f in frames), max(f['clipped'] for f in frames)])
    nf = len(frames)
    table = {                                                    # compact window table of the case, see WINDOW_TABLE
        'darkness': T['darkness'].reshape(nf, n).astype(np.float32), 'ring_mean': T['ring_mean'].reshape(nf, n).astype(np.float32),
        'azimuth_deg': np.array([f['azimuth_deg'] for f in frames], np.int16), 'elevation_deg': np.array([f['elevation_deg'] for f in frames], np.int16),
        'seed': np.array([f['seed'] for f in frames], np.int8), 'below_luma_limit': np.array([f['below_luma_limit'] for f in frames], bool),
        'valid': valid.copy(), 'strike_deg': G['strike'].copy(), 'n': G['n'].copy(), 'a': G['a'].copy(), 'length_px': Lw.copy(),
        'path': np.array([w_['path'] for w_ in W['wins']], np.int16)}
    return rec, T, table


WINDOW_TABLE = dict(
    file='light_sweep_windows.npz',
    arrays={'<case>__darkness': 'float32 (images, windows): ring mean - band mean of every window on every image, grey levels; NaN for a window '
                                'without band or ring pixels. |darkness| is the brightness difference of the label rule.',
            '<case>__ring_mean': 'float32 (images, windows): mean luma of the adjacent-rock ring, grey levels (stored for the two baseline cases; '
                                 'band mean = ring mean - darkness)',
            '<case>__azimuth_deg, __elevation_deg, __seed, __below_luma_limit': 'per image, in the order of cases.<case>.frames: sun azimuth and '
                                                                                'elevation, render seed, mean rock brightness below the limit',
            '<case>__valid': 'bool (windows): the window has band pixels and at least 4 ring pixels',
            '<case>__strike_deg': 'float64 (windows): local strike azimuth, degrees clockwise from north in [0, 180); NaN without geometry',
            '<case>__n, __a': 'float64 (windows, 3): unit normal of the fracture plane and in-plane direction a (east, north, up); NaN without geometry',
            '<case>__length_px': 'float64 (windows): image length of the samples of the window',
            '<case>__path': 'int16 (windows): path number of the window (cases.<case>.strike_groups.visible_paths gives its source trace)'},
    recompute='phi = angle between (azimuth mod 180) and strike_deg folded to 0-90; S = (depth_m / width_m) |s.n| / (s.a) with '
              's = (sin az cos el, cos az cos el, sin el), infinite where s.a <= 0; low brightness difference = valid and not (|darkness| >= 2); '
              'statistics use the images with seed 0 that are not below the brightness limit and the valid windows.',
    precision='float32: the values differ from the float64 values of the analysis by less than 2e-5 grey levels; the script checks that the '
              'counts of every bin and threshold recomputed from the stored values equal those of the analysis.')


def check_table(case, rec, T, table):
    """the stored float32 values reproduce the counts of the analysis (thresholds, S bins) exactly and the medians closely"""
    info = CASES[case]
    dk = table['darkness'].astype(np.float64)
    use = (table['seed'] == 0) & ~table['below_luma_limit']
    v = use[:, None] & table['valid'][None, :]
    out = dict(max_abs_darkness_diff=float(np.nanmax(np.abs(dk.ravel() - T['darkness']))), threshold_counts_equal=True, S_bin_counts_equal=True,
               S_bin_median_max_abs_diff=0.)
    for t in THRESHOLDS:
        a = int((v & ~(np.abs(dk) >= t)).sum())
        b = int((((T['seed'] == 0) & ~T['below'] & T['valid']) & ~(T['contrast'] >= t)).sum())
        out['threshold_counts_equal'] &= bool(a == b)
    G = dict(n=table['n'], a=table['a'])
    Sm = np.stack([S_of(G, az, el, info['d'], info['w']) for az, el in zip(table['azimuth_deg'], table['elevation_deg'])])
    sv = v & ~np.isnan(Sm)
    for i, b in enumerate(rec['S']['bins']):
        sel = sv & (Sm >= S_EDGES[i]) & ((Sm < S_EDGES[i + 1]) if i < 5 else True)
        out['S_bin_counts_equal'] &= bool(int(sel.sum()) == b['windows'] and int((sel & ~(np.abs(dk) >= U_T)).sum()) == round(b['low_contrast_share'] * b['windows']))
        if b['median_darkness'] is not None:
            out['S_bin_median_max_abs_diff'] = max(out['S_bin_median_max_abs_diff'], abs(float(np.median(dk[sel])) - b['median_darkness']))
    out['passed'] = bool(out['threshold_counts_equal'] and out['S_bin_counts_equal'] and out['S_bin_median_max_abs_diff'] < 1e-4)
    return out


def analyze(out_json, out_npz):
    P = pkg()
    res = dict(
        description='Illumination sweep: per case, one row per image (frames) and the statistics of the windows of the brightness check binned by '
                    'the shadow ratio S (S), by the angle phi between sun azimuth and local strike (rule_phi), by sun elevation (rule_elevation), by '
                    'the side of the fracture plane the sunlight comes from (sun_side) and by threshold (thresholds). The window values '
                    'themselves are in light_sweep_windows.npz (window_table).',
        paper_items=['main text Section 5.2 and Fig. 6', 'supplement S4: illumination sweep and the shadow ratio; tables of the readings binned '
                     'by S, of the numbers of windows per bin and of the fractions binned by phi'],
        script='reproduce/paper1/analysis/light_sweep_stats.py (images rendered with light_sweep.py)',
        units=dict(darkness='brightness difference, adjacent-rock ring minus in-fracture band, grey levels of the 8-bit image (Rec. 709 luma)',
                   low_contrast_share='fraction of valid windows with |band mean - ring mean| below 2 grey levels',
                   S_bin='shadow ratio S = (d / w) |s.n| / (s.a); the last bin holds the windows with infinite S (s.a <= 0)',
                   phi_bin='angle between sun azimuth and local strike of the window, degrees',
                   s_dot_n_up='component of the unit vector towards the sun along the upward normal of the fracture plane',
                   mean_luma='mean Rec. 709 luma of the rock pixels of the 8-bit image, 0-1', width_m='nominal mean opening w, m',
                   depth_m='nominal cavity depth d, m', length_share='image length of the windows that are not unclear / image length of the windows of a class',
                   tag='aAAAeEEsS: sun azimuth AAA and elevation EE in degrees, render seed S (frame <case>_<tag> of sweeps.csv)'),
        notes=['Statistics use the images with render seed 0 whose mean rock brightness is at least luma_limit, and the valid windows. Bins with '
               'fewer than 30 windows carry enough_windows = false.',
               'strike classes: the windows are assigned by their local strike to the nearer of the strike azimuths 60 and 150 degrees (the two '
               'joint sets of the scene configuration). These classes are not the joint sets: at both viewpoints the traces in frame belong '
               'almost entirely to one set (strike_groups.visible_paths); the local strike of the windows is widely scattered.',
               'floor_visibility_straight_slot: for a straight, parallel-walled fracture the line of sight through a sample reaches the floor '
               'when w |v.inward| / |v.n| >= d (v: viewing direction, n: fracture-plane normal, inward: depth axis); share over the in-frame '
               'samples of the base frame, with the 10th, 50th and 90th percentile of the fracture dip.',
               'checks.W2: the brightness differences recalculated on the sweep image differ from those of the independently generated '
               'whole-chain frame by at most W2_cross_render_max_abs_contrast_diff grey levels (two renders of the same scene differ by one '
               '8-bit level in a few tens of values) and every window is classified identically.'],
        luma_limit=LUMA_MIN, cases={}, checks={}, window_table=dict(WINDOW_TABLE, check={}))
    tables = {}
    for case in ORDER:
        try:
            ok = (fio.label_dir(FRAMES, case) / 'windows.json').is_file()
        except FileNotFoundError:
            ok = False
        if not ok:
            res['cases'][case] = dict(case=case, not_done='no base frame')
            continue
        rec, T, table = analyze_case(case, P)
        res['cases'][case] = rec
        if T is not None:
            res['window_table']['check'][case] = check_table(case, rec, T, table)
            if not CASES[case]['repeats']:
                table.pop('ring_mean')
            tables.update({'%s__%s' % (case, k): a for k, a in table.items()})
        print(case, 'images', rec['frames_rendered'], 'missing', len(rec['frames_missing']), 'windows', rec['windows'], flush=True)
    for view, case in (('V1', 'ls_v1b055'), ('V2', 'ls_v2b055')):
        if 'windows' in res['cases'].get(case, {}):
            res['checks']['W2_' + view] = check_w2(view, case, load_windows(fio.label_dir(FRAMES, case)), P)
    res['checks']['W1'] = {c: r.get('check_W1') for c, r in res['cases'].items()}
    w2 = [res['checks'][k] for k in ('W2_V1', 'W2_V2') if k in res['checks']]
    res['checks']['summary'] = dict(
        W1_all_pass=bool(all((g or {}).get('passed') for g in res['checks']['W1'].values())),
        W2_window_sets_equal=bool(w2 and all(g['window_sets_equal'] for g in w2)), W2_pass_same_image=bool(w2 and all(g['pass_same_image'] for g in w2)),
        W2_decisions_equal_cross_render=bool(w2 and all(g['decisions_equal_on_sweep_image'] for g in w2)),
        W2_cross_render_max_abs_contrast_diff=max([(f.get('on_sweep_image') or {}).get('max_abs_contrast_diff', 0.) for g in w2 for f in g['frames'].values()], default=None),
        W2_cross_render_png_diff1_channel_values=[(f.get('sweep_image_vs_w2_image') or {}).get('diff1') for g in w2 for f in g['frames'].values()])
    fio.write_json(out_json, res, depth=4)
    if tables:
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, cases=np.array([c for c in ORDER if c + '__darkness' in tables]), **tables)
    print('checks', res['checks']['summary'])
    print('window table', {c: v['passed'] for c, v in res['window_table']['check'].items()})
    return res


def main(argv=None):
    global FRAMES, SWEEP
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', required=True, help='folder with the base frames (ls_v1b055 ...) and the whole-chain frames (w2_v1_0 ...)')
    ap.add_argument('--sweep', required=True, help='folder with the sweep images written by light_sweep.py')
    ap.add_argument('--out', default=str(fio.RECORDS / 'light_sweep.json'))
    ap.add_argument('--windows-out', default=None, help='default: light_sweep_windows.npz next to --out')
    a = ap.parse_args(argv)
    FRAMES, SWEEP = a.frames, a.sweep
    analyze(a.out, a.windows_out or str(Path(a.out).with_name('light_sweep_windows.npz')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
