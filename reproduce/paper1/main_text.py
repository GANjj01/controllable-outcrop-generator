"""Recompute the numbers printed in the main text of the paper from the records in records/ and records/round3/.

main_text_values.json lists the statements of the manuscript that carry numbers (abstract, Sections 4 to 7, Table 2 and the
captions of Figs 4 to 6): where each is printed, the printed values with their rounding, and the record and keys behind them.
This script computes every item again from the records, rounds it as printed and compares. It prints one line per item and ends
with "all main-text values recomputed"; when an item differs it lists the differences and exits with status 1.

    python reproduce/paper1/main_text.py [--records DIR] [--values FILE] [--out FILE]

--records  folder that holds the records (default: records next to this script; the records of Sections 4 and 5 are in its
           sub-folder round3)
--values   the printed values (default: main_text_values.json next to this script)
--out      also write the recomputed values to this JSON file (default: not written)

The "basis" of an item says how far down the recomputation starts:
  rows     aggregated here from the rows of the record: one row per image, per viewpoint or per case
  windows  recomputed from the window table of the illumination sweep (round3/light_sweep_windows.npz): the brightness
           difference of every window under every sun direction, binned here by sun elevation, shadow ratio and sun side
  counts   computed from stored counts: sample points of the occlusion analysis on the real outcrops, histograms of the
           measurement points
  stored   a statistic read from the record as stored (quantiles over the measurement points of the real images, readings of
           single frames); computing it again needs the frames or the public dataset and the script named in the record

Comparison ("compare" of an item): "equal" formats every recomputed number with the decimals of the printed one and requires
the same text; "below" and "at_most" check a bound stated in the text ("less than 1%", "did not exceed 3%"); "about" allows the
stated tolerance around a rounded statement ("about a quarter").

Terms: the manuscript writes "aperture" for the opening of a fracture and "unconfirmed" for the unclear (U) layer of the labels;
the records m_summary_formal.json and m2_summary_formal.json name standalone outcrops "isolated".

Needs Python 3.8+ and NumPy."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINNISH = ('OG1', 'KB3')
V1_CASES = ('ls_v1b055', 'ls_v1w030', 'ls_v1w080', 'ls_v1d010', 'ls_v1d025')     # the five cases of the flat-outcrop viewpoint
NEAR_NADIR = '26120001_c0'                                                       # the flat-outcrop viewpoint of the aerial set


class Records:
    """lazy access to the JSON records and to the window table"""

    def __init__(self, root):
        self.root, self.cache = Path(root), {}

    def __call__(self, name):
        if name not in self.cache:
            p = self.root / name
            self.cache[name] = np.load(p) if p.suffix == '.npz' else json.loads(p.read_text(encoding='utf-8'))
        return self.cache[name]


def median(x):
    return float(np.median(np.asarray(x, float)))


def lo_hi(x):
    return [float(min(x)), float(max(x))]


def ranks(x):
    """ranks from 1, ties get their average rank"""
    x = np.asarray(x, float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    return ((last - counts + 1 + last) / 2.)[inverse]


def spearman(x, y):
    dx, dy = ranks(x), ranks(y)
    dx, dy = dx - dx.mean(), dy - dy.mean()
    return float((dx * dy).sum() / np.sqrt((dx * dx).sum() * (dy * dy).sum()))


def dots(u, v):
    """scalar products of every row of u with every row of v. Written with einsum, which does not call the linear-algebra
    library: on Windows an interpreter of a conda environment that was not activated cannot load it and ends without a message."""
    return np.einsum('ik,jk->ij', u, v)


def overlap(a, b):
    """overlapping coefficient of two histograms on the same bins"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.minimum(a / a.sum(), b / b.sum()).sum())


# ---------------------------------------------------------------- Section 4.1


def image_set(R):
    ab = R('round3/ablation.json')
    frames, outcrop = ab['frames'], {v: x['outcrop'] for v, x in ab['occlusion_test']['by_view'].items()}
    views = sorted({f['view'] for f in frames})
    per_view = {v: [f['pid'] for f in frames if f['view'] == v] for v in views}
    light = {v: {re.sub(r'_r\d+$', '', p).rsplit('_', 1)[-1] for p in ps} for v, ps in per_view.items()}
    seeds = {v: {(re.search(r'_r(\d+)$', p) or [0, '0'])[1] for p in ps} for v, ps in per_view.items()}
    assert len({len(x) for x in light.values()}) == 1 and len({len(x) for x in seeds.values()}) == 1
    return [len(frames), len(views), sum(outcrop[v] == 'flat' for v in views), sum(outcrop[v] == 'standalone' for v in views),
            len(light[views[0]]), len(seeds[views[0]])]


def pixel_sizes_set(R):
    frames = R('round3/ablation.json')['frames']
    px = [f['pixel_mm'] for f in frames]
    return [[f['pixel_mm'] for f in frames if f['view'] == NEAR_NADIR][0], min(px), max(px)]


def shape_set(R):
    rows = R('b_summary_formal.json')['rows']
    role = {r['id']: r['role'] for r in rows}
    views = [v for v in R('m_summary_formal.json')['views'] if v['group'] == 'B']
    other_scale = [v for v in views if v['view'] not in role]
    assert all(abs(v['pixel_mm'] - 3.) < 1e-9 for v in other_scale)
    openings = {}
    for r in rows:
        if r['role'] == 'baseline' or r['group'] == 'A_opening':
            openings.setdefault(r['cam'], set()).add(round(r['physical']['mean_opening_m'], 6))
    values = {len(v) for v in openings.values()}
    assert len(values) == 1
    viewpoints = {('_'.join(v['view'].split('_')[-2:]), round(v['pixel_mm'], 3)) for v in views}
    px = [v['pixel_mm'] for v in views]
    return [len(views), sum(role.get(v['view']) == 'variant' for v in views), sum(role.get(v['view']) == 'baseline' for v in views),
            len(other_scale), values.pop(), len(openings), len(viewpoints), min(px), max(px)]


def aerial_checked(R):
    frames = [f for f in R('m_summary_formal.json')['frames'] if f['group'] in ('C', 'D')]
    failed = {x['key'] for x in R('values_relief_formal.json')['formal_labels']['not_pass']}
    passed = [f for f in frames if f['exit_code'] == 0 and f['key'] not in failed]
    scenes = {re.search(r'26\d{6}', f['view']).group(0) for f in frames}
    return [len(passed), sum(f['group'] == 'C' for f in frames), sum(f['group'] == 'D' for f in frames), len(scenes),
            len({f['view'] for f in frames})]


def centre_deviation(R):
    return [R('values_relief_formal.json')['numeric_detail_C']['nontip_centre_px_p99']['max']]


def centre_deviation_mm(R):
    d = R('values_relief_formal.json')['numeric_detail_C']
    return [d['nontip_centre_px_p99']['max'] * d['pixel_mm']['max']]


def locality(R):
    rows = [r for r in R('b_summary_formal.json')['rows'] if r['role'] == 'variant']
    out = [r['local']['T_variant'] for r in rows]
    inside = [r['local']['T_inside_band'] for r in rows]
    return [len(rows), sum(r['local']['geometry_changed_share'] > 0 for r in rows), min(out), max(out), max(r['local']['T_null'] for r in rows),
            min(inside), max(inside)]


def checked_pixel_sizes(R):
    frames = [f for f in R('m_summary_formal.json')['frames'] if f['pixel_mm'] >= 2. - 1e-9]
    assert all(f['exit_code'] == 0 for f in frames)
    flat = [f['pixel_mm'] for f in frames if f['outcrop'] == 'flat']
    standalone = [f['pixel_mm'] for f in frames if f['outcrop'] == 'isolated']
    return lo_hi(flat) + lo_hi(standalone)


def ridge_auc(R):
    frames = R('round3/ridge_check.json')['frames']

    def values(k):
        return [f['auc'][k] for f in frames if f['auc'].get(k) is not None]
    c = values('C')
    return [len(c), median(c), min(c), max(c), median(values('O_near')), median(values('U')), median(values('O_far'))]


# ---------------------------------------------------------------- Section 4.2 and Table 2


def table2_row1(R):
    views = {}
    for f in R('m2_summary_formal.json')['frames']:
        if f['pixel_mm'] >= 2. - 1e-9 and f.get('S') is not None:
            views.setdefault((f['view'], f['outcrop']), []).append((f['S'], f['mid']))
    s = {k: median([x[0] for x in v]) for k, v in views.items()}
    standalone = [x for (_, o), x in s.items() if o == 'isolated']
    flat = [x for (_, o), x in s.items() if o == 'flat']
    mid = min(x[1] for v in views.values() for x in v)
    return [len(views)] + [100. * x for x in lo_hi(standalone) + lo_hi(flat) + [mid]]


def by_view(frames, key):
    out = {}
    for f in frames:
        out.setdefault(f['view'], []).append(key(f))
    return out


def table2_row4(R):
    ab = R('round3/ablation.json')
    share = {v: median(x) for v, x in by_view(ab['frames'], lambda f: f['occluded_px'] / f['projected_in_frame_px']).items()}
    standalone = [x for v, x in share.items() if ab['occlusion_test']['by_view'][v]['outcrop'] == 'standalone']
    return [100. * share[NEAR_NADIR]] + [100. * x for x in lo_hi(standalone)]


def table2_row5(R):
    x = [f['exempt_with_occluder_length_px'] / f['visible_length_px'] for f in R('round3/ablation.json')['frames']]
    return [100. * min(x), 100. * max(x), 100. * median(x)]


def table2_row6(R):
    frames = R('round3/ablation.json')['frames']
    px = [[1000. * s / f['pixel_mm'] for s in f['shift_m_p50_p99_max']] for f in frames]
    return lo_hi([p[1] for p in px]) + lo_hi([p[2] for p in px])


def unclear_windows_share(R, t):
    return [(f['sampling_insufficient'] + f['low_contrast_windows_t%d' % t]) / f['n'] for f in R('round3/u_threshold.json')['frames']]


def table2_row7(R):
    return [100. * x for x in lo_hi(unclear_windows_share(R, 2))]


def facing(R):
    v = R('round3/occluded_segments.json')['facing']['per_view']['a0062']
    assert v['all_conditions_met']
    s = v['split']
    return [v['theta_deg'], 100. * v['report']['occluded_px'] / v['report']['projected_in_frame_px'],
            100. * s['back_occluded_weight_px'] / s['occluded_weight_px'],
            100. * s['front_occluded_weight_px'] / (s['front_occluded_weight_px'] + s['front_visible_weight_px'])]


def unclear_length(R):
    return [100. * x for x in lo_hi([f['visible_u_px'] / f['visible_px'] for f in R('round3/ablation.json')['frames']])]


def unclear_windows(R):
    frames = R('round3/u_threshold.json')['frames']
    unclear = sum(f['sampling_insufficient'] + f['low_contrast_windows_t2'] for f in frames)
    return [unclear, 100. * sum(f['sampling_insufficient'] for f in frames) / unclear]


def low_difference_windows(R):
    return [100. * x for x in lo_hi([f['low_contrast_windows_t2'] / f['n'] for f in R('round3/u_threshold.json')['frames']])]


def threshold_sensitivity(R):
    return [100. * x for t in (1, 2, 4) for x in lo_hi(unclear_windows_share(R, t))]


def skeleton(R):
    views = R('round3/mask_skeleton.json')['views']
    close = [v for v in views if abs(v['pixel_mm'] - 2.) < 1e-9]
    x = [v['skeleton_to_centreline_frac_gt_2px'] for v in views]
    return [len(views) - len(close), len(close), 100. * min(x), 100. * max(x), 100. * median(x)]


# ---------------------------------------------------------------- Section 4.3


def geocrack_images(R):
    return [len(R('round3/geocrack_comparison.json')['per_image']['real'])]


def geocrack_contrast(R):
    s = R('round3/geocrack_comparison.json')['summary']['relative_contrast']
    return [s['real']['all']['median'], s['synthetic_aerial']['all']['median']]


def geocrack_overlap(R, quantity, real_subset):
    h = R('round3/geocrack_comparison.json')['histograms'][quantity]
    return overlap(h['real'][real_subset], h['synthetic_aerial']['all'])


def geocrack_contrast_overlap(R):
    return [geocrack_overlap(R, 'relative_contrast', 'all')]


def geocrack_width(R):
    s = R('round3/geocrack_comparison.json')['summary']['log10_width_over_spacing']
    return [s['real']['all']['ratio']['median'], s['synthetic_aerial']['all']['ratio']['median']]


def geocrack_width_overlap(R):
    return [geocrack_overlap(R, 'log10_width_over_spacing', 'all')]


def geocrack_comparable_overlap(R):
    return [R('round3/geocrack_comparison.json')['parameters']['comparable_min_width_px'],
            geocrack_overlap(R, 'relative_contrast', 'comparable'), geocrack_overlap(R, 'log10_width_over_spacing', 'comparable')]


def geocrack_centre(R):
    s = R('round3/geocrack_comparison.json')['summary']['centre_brightness']
    return [s['synthetic_aerial']['all']['median'], s['real']['all']['median']]


def geocrack_background(R):
    p = R('round3/geocrack_comparison.json')['per_image']
    return [median([x['background_variation'] for x in p['synthetic_aerial']]), median([x['background_variation'] for x in p['real']])]


def geocrack_background_ratio(R):
    synthetic, real = geocrack_background(R)
    return [synthetic / real]


def fraxet_pixel_sizes(R):
    d = R('round3/fraxet_sites.json')
    assert d['sites']['OG1']['pixel_mm'] == d['sites']['KB3']['pixel_mm']
    return d['sites']['OG1']['pixel_mm'] + d['sites']['EN']['pixel_mm'] + d['synthetic_reference']['pixel_mm_range']


def fraxet_profiles(R, site):
    return R('round3/fraxet_sites.json')['sites'][site]['profiles']


def fraxet_width(R):
    ref = R('round3/fraxet_sites.json')['synthetic_reference']['dark_band_width_mm']
    med = [x['median'] for s in FINNISH for x in fraxet_profiles(R, s)['dark_band_width_mm'].values()]
    return lo_hi(med) + [ref['median'], ref['p5'], ref['p95']]


def fraxet_width_coverage(R):
    w = [x for s in FINNISH for x in fraxet_profiles(R, s)['vs_synthetic']['width_mm'].values()]
    return [100. * x for x in lo_hi([x['inside_synthetic_p5_p95'] for x in w]) + lo_hi([x['below_synthetic_p5'] for x in w])]


def fraxet_contrast(R):
    return (lo_hi([fraxet_profiles(R, s)['relative_contrast']['median'] for s in FINNISH])
            + lo_hi([fraxet_profiles(R, s)['vs_synthetic']['relative_contrast_overlap'] for s in FINNISH]))


# ---------------------------------------------------------------- Section 5.1


def sweep_rows(R):
    return {(r['outcrop'], r['azimuth_kind'], int(r['beta_deg'])): r for r in R('round3/view_sweep.json')['rows']}


def occluded(row):
    return 100. * row['occluded_px'] / row['in_frame_px']


def sweep_settings(R):
    betas = [b for (_, _, b) in sweep_rows(R)]
    cases = R('round3/light_sweep.json')['cases']
    margin = R('round3/fraxet_sites.json')['parameters']['occlusion_margin']          # '0.5 pixel size'
    assert margin.endswith(' pixel size')
    return [min(betas), max(betas), sum(c['view'] == 'V1' for c in cases.values()), float(margin.split()[0])]


def view_flat_nadir(R):
    return [occluded(sweep_rows(R)[('flat', 'nadir', 0)])]


def view_flat_15(R):
    rows = sweep_rows(R)
    return [occluded(rows[('flat', k, 15)]) for k in ('across strike', 'along strike')]


def view_flat(R):
    rows = sweep_rows(R)
    return [occluded(rows[('flat', 'across strike', b)]) for b in (30, 45, 60)] + [occluded(rows[('flat', 'along strike', 60)])]


def view_standalone(R):
    rows = sweep_rows(R)
    return [occluded(rows[('standalone', k, b)]) for b in (15, 60) for k in ('along strike', 'across strike')]


def fig4_caption(R):
    d = R('round3/view_sweep.json')
    first = min(b for (o, _, b), r in sweep_rows(R).items() if o == 'standalone' and r['state'] == 'done')
    return [d['scenes']['flat']['pixel_mm'], d['scenes']['standalone']['pixel_mm'], 100. * d['shape_parameters']['width_m'],
            d['shape_parameters']['depth_m'], first]


def site_rows(R, site, beta=None):
    """rows of the occlusion analysis of a site at the main clearance height: one per pixel size (and per off-nadir angle)"""
    d = R('round3/fraxet_sites.json')
    main = d['parameters']['clearance_main_m']
    return [r for rows in d['sites'][site]['occlusion']['by_pixel_mm'].values() for r in rows
            if r['clearance_m'] == main and (beta is None or r['beta_deg'] == beta)]


def shares(row, direction):
    """occluded fraction with the undetermined sample points left out of the denominator, and with them counted as visible"""
    x = row[direction]
    return [x['occluded'] / x['determined'], x['occluded'] / (x['determined'] + x['undetermined'])]


def real_15(R):
    return [100. * max(s for site in FINNISH for r in site_rows(R, site, 15.) for k in ('across', 'along') for s in shares(r, k))]


def real_60(R, sites):
    rows = [r for site in sites for r in site_rows(R, site, 60.)]
    return [100. * x for k in ('across', 'along') for x in lo_hi([s for r in rows for s in shares(r, k)])]


def real_60_og1(R):
    return real_60(R, ('OG1',))


def real_60_kb3(R):
    return real_60(R, ('KB3',))


def real_60_both(R):
    return real_60(R, FINNISH)


def real_ratio(R):
    out = []
    for site in FINNISH:
        out += lo_hi([a / b for r in site_rows(R, site, 60.) for a, b in zip(shares(r, 'across'), shares(r, 'along'))])
    return out


def synthetic_ratio(R):
    rows = sweep_rows(R)
    return [occluded(rows[('flat', 'across strike', 60)]) / occluded(rows[('flat', 'along strike', 60)])]


def real_en(R):
    return [100. * max(s for r in site_rows(R, 'EN') for k in ('across', 'along') for s in shares(r, k))]


def fig5_undetermined(R):
    x = [r['across']['undetermined'] / (r['across']['determined'] + r['across']['undetermined'])
         for site in FINNISH for r in site_rows(R, site, 60.)]
    return [100. * v for v in lo_hi(x)]


def fig5_dem_spacing(R):
    s = R('round3/fraxet_sites.json')['sites']['KB3']
    return [s['elevation_grid']['identical_run_median_px'] * p for p in s['pixel_mm']]


# ---------------------------------------------------------------- Section 5.2


def light_case(R, case):
    """window table of one case of the illumination sweep: brightness difference (images x windows), the images and windows that
    enter the statistics, shadow ratio S and the component of the sun direction along the upward normal of the fracture plane"""
    z, c = R('round3/light_sweep_windows.npz'), R('round3/light_sweep.json')['cases'][case]
    dark = z[case + '__darkness'].astype(float)
    use = (z[case + '__seed'] == 0) & ~z[case + '__below_luma_limit']
    az, el = np.radians(z[case + '__azimuth_deg'].astype(float)), np.radians(z[case + '__elevation_deg'].astype(float))
    sun = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=1)        # east, north, up
    n, a = z[case + '__n'], z[case + '__a']
    geometry = np.isfinite(a).all(axis=1) & np.isfinite(n).all(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sa, sn = dots(sun, a), dots(sun, n)
        S = np.where(sa > 0, (c['depth_m'] / c['width_m']) * np.abs(sn) / sa, np.inf)
        side = dots(sun, n * np.sign(n[:, 2:3]))
    ok = use[:, None] & z[case + '__valid'][None, :]
    return dict(dark=dark, ok=ok, ok_geometry=ok & geometry[None, :], S=S, side=side, use=use, valid=z[case + '__valid'],
                elevation=z[case + '__elevation_deg'], azimuth=z[case + '__azimuth_deg'])


def light_low_difference(R):
    t = light_case(R, 'ls_v1b055')
    low = (t['ok'] & ~(np.abs(t['dark']) >= 2.)).sum(axis=1) / float(t['valid'].sum())
    out, n = [], set()
    for e in (20, 40, 60):
        sel = t['use'] & (t['elevation'] == e)
        n.add(int(sel.sum()))
        out.append(100. * median(low[sel]))
    assert len(n) == 1
    return [n.pop()] + out


def light_low_difference_range(R):
    return lo_hi(light_low_difference(R)[1:])


def s_medians(R, case):
    t = light_case(R, case)
    return (median(t['dark'][t['ok_geometry'] & (t['S'] < .25)]), median(t['dark'][t['ok_geometry'] & (t['S'] >= 1.)]),
            spearman(t['S'][t['ok_geometry'] & (t['S'] < 1.)], t['dark'][t['ok_geometry'] & (t['S'] < 1.)]))


def light_S_cases(R):
    m = [s_medians(R, c) for c in V1_CASES]
    return [len(m), sum(deep < shallow for shallow, deep, _ in m)] + lo_hi([rho for _, _, rho in m])


def light_S_baseline(R):
    return list(s_medians(R, 'ls_v1b055')[:2])


def light_dip(R):
    v = R('round3/light_sweep.json')['cases']['ls_v1b055']['floor_visibility_straight_slot']
    return [v['fracture_dip_deg_quantiles_10_50_90'][0], v['fracture_dip_deg_quantiles_10_50_90'][2], 100. * v['share_floor_reachable']]


def light_sun_side(R):
    t = light_case(R, 'ls_v1b055')
    edges = (-1., -.5, -.25, 0., .25, .5, 1.)
    med = []
    for i in range(6):
        sel = t['ok_geometry'] & (t['side'] >= edges[i]) & ((t['side'] < edges[i + 1]) if i < 5 else (t['side'] <= edges[i + 1]))
        med.append(median(t['dark'][sel]))
    wall_or_behind, near_plane = [med[0], med[1], med[5]], [med[2], med[3]]
    return lo_hi(wall_or_behind) + lo_hi(near_plane) + [max(med) / min(med)]


def fig6_caption(R):
    t, cases = light_case(R, 'ls_v1b055'), R('round3/light_sweep.json')['cases']
    az = sorted(set(int(x) for x in t['azimuth'][t['use']]))
    step = set(np.diff(az).tolist())
    assert len(step) == 1
    others = [cases[c] for c in V1_CASES[1:]]
    base = cases['ls_v1b055']
    w = sorted({100. * c['width_m'] for c in others} - {100. * base['width_m']})
    d = sorted({c['depth_m'] for c in others} - {base['depth_m']})
    return [step.pop()] + sorted(set(int(x) for x in t['elevation'][t['use']])) + [int(t['use'].sum()), 100. * base['width_m'],
                                                                                     base['depth_m']] + w + d


def fraxet_phi(R):
    sites = R('round3/fraxet_sites.json')['sites']
    return lo_hi([s['illumination']['spearman_phi_brightness_difference'] for s in sites.values()])


# ---------------------------------------------------------------- Section 5.3


def width_points(R):
    pts = R('round3/width_curve.json')['points']
    return [dict(p, geometric=p['opening_mm'] / p['pixel_mm'], ratio=p['image_width_px'] / (p['opening_mm'] / p['pixel_mm'])) for p in pts]


def width_ratio(R):
    pts = width_points(R)
    elevation = {p['sun_elevation_deg'] for p in pts}
    assert len(elevation) == 1
    r = [p['ratio'] for p in pts]
    return [len(pts), len({p['camera'] for p in pts}), len({p['opening_mm'] for p in pts}), elevation.pop(), min(r), max(r), median(r)]


def width_segments(R):
    pts = width_points(R)
    small, large = [p for p in pts if p['geometric'] < 3.], [p for p in pts if p['geometric'] > 10.]
    assert all(p['outcrop'] == 'flat' and p['fault_visible_px'] > 0 for p in small)
    return [len(small), len({p['camera'] for p in small}), sum(p['ratio'] > 1. for p in small), len(large), sum(p['ratio'] < 1. for p in large)]


def width_standalone(R):
    pts = [p for p in width_points(R) if p['camera'] == 'a26130005_c1']
    assert all(p['outcrop'] == 'standalone' for p in pts)
    r = [p['ratio'] for p in pts]
    return [pts[0]['pixel_mm'], sum(x > 1. for x in r), len(r), min(r), max(r)]


def near_nadir_frames(R, record):
    return [f for f in R(record)['frames'] if f['view'] == NEAR_NADIR]


def retained_near_nadir(R):
    frames = near_nadir_frames(R, 'round3/ablation.json')
    visible = {f['visible_px'] / f['projected_in_frame_px'] for f in frames}
    assert len(visible) == 1
    clear = [(f['visible_px'] - f['visible_u_px']) / f['projected_in_frame_px'] for f in frames]
    return [100. * visible.pop()] + [100. * x for x in lo_hi(clear)]


def retained_shortfall(R):
    _, lo, hi = retained_near_nadir(R)
    return [100. - hi, 100. - lo]


def windows_near_nadir(R):
    frames = near_nadir_frames(R, 'round3/u_threshold.json')
    insufficient = {f['sampling_insufficient'] / f['n'] for f in frames}
    assert len(insufficient) == 1
    return [100. * insufficient.pop()] + [100. * x for x in lo_hi([f['low_contrast_windows_t2'] / f['n'] for f in frames])]


CALC = {f.__name__: f for f in (
    image_set, pixel_sizes_set, shape_set, aerial_checked, centre_deviation, centre_deviation_mm, locality, checked_pixel_sizes, ridge_auc,
    table2_row1, table2_row4, table2_row5, table2_row6, table2_row7, facing, unclear_length, unclear_windows,
    low_difference_windows, threshold_sensitivity, skeleton, geocrack_images, geocrack_contrast, geocrack_contrast_overlap,
    geocrack_width, geocrack_width_overlap, geocrack_comparable_overlap, geocrack_centre, geocrack_background,
    geocrack_background_ratio, fraxet_pixel_sizes, fraxet_width, fraxet_width_coverage, fraxet_contrast, sweep_settings,
    view_flat_nadir, view_flat_15, view_flat, view_standalone, fig4_caption, real_15, real_60_og1, real_60_kb3, real_60_both,
    real_ratio, synthetic_ratio, real_en, fig5_undetermined, fig5_dem_spacing, light_low_difference, light_low_difference_range,
    light_S_cases, light_S_baseline, light_dip, light_sun_side, fig6_caption, fraxet_phi, width_ratio, width_segments,
    width_standalone, retained_near_nadir, retained_shortfall, windows_near_nadir)}


def compare(item, got):
    """(agrees, recomputed values as text)"""
    printed, mode = item['printed'], item.get('compare', 'equal')
    if mode == 'equal':
        dec = item['decimals'] if isinstance(item['decimals'], list) else [item['decimals']] * len(printed)
        text = ['%.*f' % (d, v) for d, v in zip(dec, got)]
        return len(got) == len(printed) == len(dec) and text == printed, text
    text = ['%.3g' % v for v in got]
    bound = [float(p) for p in printed]
    if mode == 'below':
        return all(v < bound[0] for v in got), text
    if mode == 'at_most':
        return all(v <= bound[0] for v in got), text
    if mode == 'about':
        target = bound if len(bound) == len(got) else bound * len(got)
        return all(abs(v - t) <= item['tolerance'] for v, t in zip(got, target)), text
    raise ValueError('unknown comparison %r' % mode)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Recompute the numbers of the main text from the records and compare them with the printed values.')
    ap.add_argument('--records', default=str(HERE / 'records'), help='folder that holds the records (with the sub-folder round3)')
    ap.add_argument('--values', default=str(HERE / 'main_text_values.json'), help='the printed values')
    ap.add_argument('--out', help='also write the recomputed values to this JSON file')
    a = ap.parse_args(argv)
    values = json.loads(Path(a.values).read_text(encoding='utf-8'))
    items, R = values['items'], Records(a.records)
    ids = [i['id'] for i in items]
    unknown, unused = sorted(set(ids) - set(CALC)), sorted(set(CALC) - set(ids))
    if unknown or unused or len(set(ids)) != len(ids):
        raise SystemExit('main_text_values.json and this script list different items: %s' % (unknown + unused or 'an id is repeated'))
    bad, done, basis, numbers = [], [], {}, 0
    for item in items:
        got = [float(v) for v in CALC[item['id']](R)]
        ok, text = compare(item, got)
        mode = item.get('compare', 'equal')
        shown = ' '.join(item['printed']) if mode == 'equal' else '%s %s%s' % (mode, ' '.join(item['printed']),
                                                                             ' +-%g' % item['tolerance'] if mode == 'about' else '')
        print('%-4s %-28s %-8s printed %-46s recomputed %-46s %s' % ('ok' if ok else 'DIFF', item['id'], item['basis'], shown, ' '.join(text),
                                                                item['where']))
        if not ok:
            bad.append('%s (%s): printed %s, recomputed %s' % (item['id'], item['where'], shown, ' '.join(text)))
        basis[item['basis']] = basis.get(item['basis'], 0) + 1
        numbers += len(item['printed'])
        done.append(dict(id=item['id'], where=item['where'], printed=item['printed'], recomputed=text, values=got, agrees=bool(ok)))
    if a.out:
        Path(a.out).write_text(json.dumps(done, indent=1) + '\n', encoding='utf-8')
    print('\n%d items with %d printed values; basis: %s' % (len(items), numbers, ', '.join('%s %d' % kv for kv in sorted(basis.items()))))
    if bad:
        print('\nDIFFERENCES FROM THE PRINTED VALUES:\n  ' + '\n  '.join(bad))
        return 1
    print('all main-text values recomputed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
