"""Bookkeeping for faulted cases in the trace labels. No fault arithmetic here: translations, sides and plane distances
come from frgb/fault_charts.py or from the builder's own records in build.json. Both environments.

Group of a path = the builder's group (source trace, block): block = construction_checks.fault.block of its build.json
row; tag = '-'.join(block), 'reference' when the block is empty; a path without a fault record has no tag and keeps
the fault-free name (its trace id)."""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parents[1]
sys.path.append(str(P1))
from frgb import runconfig as frgb_rc  # noqa: E402
SRC = frgb_rc.path('outcrop_src').as_posix()

DEPLOYED_MD5 = {'build.py': '612dd2317a3e8c4f1bae05204b19ee30', 'prepare.py': '7fecbde7bb2e7fcf20c96f5555e32852',
                'frgb/fault_charts.py': '61ee4c25f208cf65c0095ec816a9f145'}
DEPLOYED_NETWORK_PANELS_SHA256 = '68ec33412bbb2e1a9be254d4d53c83017008a31866f721fc379b371721c7ef6e'
INTERNAL_MD5 = {'build.py': '1745da821ccedd6673a0dc8e193b0890', 'prepare.py': 'ff2b552571015c9467c51e62aab6d0a4', 'frgb/fault_charts.py': 'a7f0838fedb04554731895597f06356b'}
INTERNAL_NETWORK_PANELS_SHA256 = '6e0272773da0f78acec6dda9b1ce437507bfe0b5de975102da72c74d19bd7148'

# cases, the role literals frgb/network_panels.py and frgb/fault_charts.py can assign, and fault_cap
KNOWN_ROLES = frozenset({'original_host', 'left_wall', 'right_wall', 'floor', 'internal_cap', 'internal', 'finite_support_bound',
                         'junction_bridge', 'outside_cap', 'station_start', 'station_end', 'start_cap', 'end_cap', 'fault_cap'})


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag_of_block(block):
    return '-'.join(block) if block else 'reference'


def path_groups(build):
    """{path index: dict(trace, key, tag, block, shift)} from build.json; tag / shift None for a path no fault acted on"""
    out = {}
    for p in build['paths']:
        f = (p.get('construction_checks') or {}).get('fault')
        if f is None or f.get('replay') is None:
            out[p['index']] = dict(trace=p['source_trace_id'], key=p['source_trace_id'], tag=None, block=None, shift=None)
            continue
        tag = tag_of_block(f['block'])
        shift = None if f.get('shift_m') is None else np.asarray(f['shift_m'], float)
        out[p['index']] = dict(trace=p['source_trace_id'], key='%s_%s' % (p['source_trace_id'], tag), tag=tag,
                               block=list(f['block']), shift=shift)
    return out


def faulted_build(build):
    return 'fault_frames' in build


def load_fault_charts():
    """Called only for faulted scenes, after stage A has started."""
    env = Path(sys.executable).parent
    parts = os.environ.get('PATH', '').split(os.pathsep)
    for d in (str(env / 'Library' / 'bin'), str(env)):
        if d not in parts:
            os.environ['PATH'] = os.pathsep.join([d, os.environ.get('PATH', '')])
    for p in (str(P1), SRC):
        if p not in sys.path:
            sys.path.append(p)
    from frgb import fault_charts
    return fault_charts


EXT_BASE = 100000
EXT_EXCLUDE_M = 1e-3       # I4: pseudo samples within 1 mm (chart chord) of the shared end station are dropped
EXT_TRIM_M = 1e-3          # I5: pseudo samples must lie in the own block at least 1 mm from the fault plane
EXT_COVER_STEP_M = 2e-3    # R1.4: the recorded exposure interval is checked every 2 mm, covered within 2 mm
EXT_MASK_PX = 4.           # R1.6: the label band of the pseudo samples for the regression


def pseudo_index(path, end):
    return EXT_BASE + 10 * int(path) + (0 if int(end) == 0 else 1)


def is_pseudo(index):
    return int(index) >= EXT_BASE


def base_of(index):
    """(base path, end) of a pseudo path; (index, None) for a real path"""
    i = int(index)
    if i < EXT_BASE:
        return i, None
    k = i - EXT_BASE
    return k // 10, (0 if k % 10 == 0 else -1)


def fault_ends_of(build):
    """{path: [dict(end, fault, moved, L_m, exposure)]} from build.json paths[i].construction_checks.fault.ends"""
    out = {}
    for p in build['paths']:
        for e in ((p.get('construction_checks') or {}).get('fault') or {}).get('ends', []):
            out.setdefault(int(e['path']), []).append(dict(end=int(e['end']), fault=e['fault'], moved=bool(e.get('moved')),
                                                           L_m=float(e['L_m']),
                                                           exposure=((e.get('structure') or {}).get('mouth') or {}).get('exposure'),
                                                           structure=e.get('structure')))
    return out


def filter_samples(b, keep):
    """a branch dict keeping only the samples where keep (arrays whose first axis is the sample axis)"""
    n = len(b['u'])
    keep = np.asarray(keep, bool)
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in b.items()}


def ext_trim_fn(case, frames, FC):
    """keep-mask of a pseudo path's samples: I4 chart chord distance from the end station > EXT_EXCLUDE_M; I5
    fault_charts.signed_other <= -EXT_TRIM_M for the end's fault (the plane is invariant under the in-plane slip); R1.2 the chord
    parameter t (along nu2) within [-margin, depth] of the end (when the path carries its ext_meta)"""
    ends = fault_ends_of(case['build'])
    fby = {f['id']: f for f in frames['faults']}

    def trim(pid, b):
        base, end = base_of(pid)
        e = next(x for x in ends[base] if x['end'] == end)
        st = case['paths'][pid]['stations']
        Lq = float(np.linalg.norm(np.asarray(st['points'][1], float) - np.asarray(st['points'][0], float)))
        keep = np.asarray(b['u'], float) * Lq > EXT_EXCLUDE_M
        keep &= np.asarray(FC.signed_other(fby[e['fault']], e['moved'], np.asarray(b['point'], float))) <= -EXT_TRIM_M
        meta = case['paths'][pid].get('ext_meta')
        if meta is not None and 't' in b:
            t = np.asarray(b['t'], float)
            keep &= (t >= -float(meta['margin'])) & (t <= float(meta['depth']))
        return keep
    return trim


def extension_slot_check(case, samples):
    """R1.3: every visible pseudo sample's first hit on the cut native mesh is a cavity face of a path of its own group"""
    nat = case['native']; key = case.get('cutter_key') or {}
    out, bad = {}, 0
    for k, r in samples.items():
        pid = k[0]
        if not is_pseudo(pid) or 'visible' not in r:
            continue
        group_paths = {i for i, g in key.items() if g == key.get(pid)}
        vis = np.asarray(r['visible'], bool); hf = np.asarray(r['hit_face'], np.int64)[vis]
        roles = nat['roles'][hf] if len(hf) else np.empty(0, str)
        pof = nat['path_of_face'][hf] if len(hf) else np.empty(0, np.int64)
        ok = (hf >= 0) & (roles != 'original_host') & np.isin(pof, list(group_paths))
        rec = out.setdefault(str(pid), dict(visible=0, at_slot_opening=0, roles={}))
        rec['visible'] += int(vis.sum()); rec['at_slot_opening'] += int(ok.sum())
        for role, c in zip(*np.unique(roles, return_counts=True)):
            rec['roles'][str(role)] = rec['roles'].get(str(role), 0) + int(c)
        bad += int((~ok).sum())
    return dict(per_path=out, not_at_slot_opening=bad)


def exposure_crosscheck(case, samples, cam, project, in_frame):
    out, uncovered = {}, 0
    for pid, p in case['paths'].items():
        if not is_pseudo(pid) or p.get('ext_meta') is None:
            continue
        base, end = base_of(pid)
        e = next((x for x in fault_ends_of(case['build']).get(base, ()) if x['end'] == end), None)
        if e is None or not e['exposure']:
            out[str(pid)] = dict(record=None)
            continue
        meta = p['ext_meta']; lo, hi = (float(x) for x in e['exposure']['lambda_m'])
        lam = np.arange(lo, hi + EXT_COVER_STEP_M / 2, EXT_COVER_STEP_M) if hi > lo else np.array([lo])
        X = np.asarray(p['stations']['points'], float)
        P = X[0] + np.outer(lam / max(float(meta['L']), 1e-12), X[1] - X[0])
        uv, z = project(P, cam)
        inside = np.isfinite(uv).all(1) & (z > cam['near']) & in_frame(np.where(np.isfinite(uv), uv, -1e9), cam)
        ls = np.concatenate([(np.asarray(r['hit_q'], float) - meta['p_end']) @ meta['t2'] for k, r in samples.items() if k[0] == pid]
                            or [np.empty(0)])
        cov = np.array([bool(len(ls)) and float(np.min(np.abs(ls - x))) <= EXT_COVER_STEP_M for x in lam[inside]], bool)
        n_unc = int((~cov).sum()) if len(cov) else 0
        uncovered += n_unc
        out[str(pid)] = dict(record=e['exposure'], points=int(len(lam)), in_frame=int(inside.sum()), uncovered=n_unc,
                             status='out_of_frame' if not inside.any() else ('covered' if not n_unc else 'uncovered'))
    return dict(per_path=out, uncovered=uncovered)


def clip_straddling(Fv, cov, so, tol):
    """Fv (k, 3) host vertex ids of candidate faces, cov (k, 3) covered flags, so (k, 3) fault_charts.signed_other at those
    vertices (> tol: the other block's side). A face straddles when it has a vertex on the other side and every
    own-side vertex is covered. Its own-side part: one own vertex A (cyclic order A, B, C) -> (A, P_AB, P_AC); two own
    vertices (far vertex C, cyclic order A, B, C) -> (A, B, P_BC), (A, P_BC, P_CA); orientation kept. New points sit
    on host edges (own vertex a, other-side vertex b) at t = -so(a) / (so(b) - so(a)) (exact for the plane), one per
    host edge. Returns dict(rows = straddling input rows, edges (n, 2) [a, b], t (n,), tris (m, 3) with host vertex
    ids >= 0 and new point j as -1 - j, tri_row (m,) input row of each triangle)."""
    Fv = np.asarray(Fv, np.int64).reshape(-1, 3); cov = np.asarray(cov, bool).reshape(-1, 3)
    so = np.asarray(so, float).reshape(-1, 3)
    far = so > tol; own = ~far
    rows = np.flatnonzero(far.any(1) & own.any(1) & (far | cov).all(1))
    keys, ts, tris, trow = {}, [], [], []

    def new(a_, sa, b_, sb):
        k_ = (int(a_), int(b_))
        if k_ not in keys:
            keys[k_] = len(ts); ts.append(min(max(-sa / (sb - sa), 0.), 1.))
        return -1 - keys[k_]
    for r in rows:
        v, s_, o = Fv[r], so[r], own[r]
        if o.sum() == 1:
            i = int(np.argmax(o)); A, B, C = (i, (i + 1) % 3, (i + 2) % 3)
            tris.append((v[A], new(v[A], s_[A], v[B], s_[B]), new(v[A], s_[A], v[C], s_[C]))); trow.append(r)
        else:
            j = int(np.argmax(~o)); C, A, B = (j, (j + 1) % 3, (j + 2) % 3)
            pbc, pca = new(v[B], s_[B], v[C], s_[C]), new(v[A], s_[A], v[C], s_[C])
            tris += [(v[A], v[B], pbc), (v[A], pbc, pca)]; trow += [r, r]
    edges = np.array(sorted(keys, key=keys.get), np.int64).reshape(-1, 2)
    return dict(rows=rows, edges=edges, t=np.asarray(ts, float), tris=np.asarray(tris, np.int64).reshape(-1, 3),
                tri_row=np.asarray(trow, np.int64))


INDEP_STEP_M = 2e-3          # R2.1: clipped host faces split to edges <= 2 mm
INDEP_SPACING_M = 1e-3       # R2.1: points every <= 1 mm along the phi = 0 segments
INDEP_BBOX_TOL_M = 2e-3      # R2.4: record bbox widened by 2 mm
INDEP_CLASS_B_SHARE = .2     # R2.3: report condition (not a stop)
INDEP_CLASS_B_MIN = 10


def _clip_polygon(P, d):
    """Sutherland-Hodgman of one polygon P (k, 3) against d <= 0 (d (k,) the signed values at its corners, linear on it)"""
    out = []
    for i in range(len(P)):
        a, b, da, db = P[i], P[(i + 1) % len(P)], d[i], d[(i + 1) % len(P)]
        if da <= 0:
            out.append(a)
        if (da <= 0) != (db <= 0):
            out.append(a + (da / (da - db)) * (b - a))
    return np.asarray(out, float).reshape(-1, 3)


def indep_trace(V, F, so_fn, phi_fn, step=INDEP_STEP_M, spacing=INDEP_SPACING_M):
    """R2.1 (pure numpy apart from the two callables): host faces F (k, 3) of vertices V clipped to so_fn <= 0 (polygon clipping per
    face, fan triangles), candidates by phi at their corners (sign change, or min |phi| <= the longest edge), each candidate split
    uniformly to edges <= step, phi on every split vertex, the phi = 0 segment of every split triangle (linear), points every <= spacing.
    phi_fn(P) -> (phi, q) with NaN where the mid-surface is undefined. Returns the points (n, 3)."""
    F = np.asarray(F, np.int64).reshape(-1, 3)
    if not len(F):
        return np.zeros((0, 3))
    T = V[F]
    d = np.asarray(so_fn(T.reshape(-1, 3)), float).reshape(-1, 3)
    tris = []
    for t_, d_ in zip(T, d):
        if (d_ > 0).all():
            continue
        poly = t_ if (d_ <= 0).all() else _clip_polygon(t_, d_)
        for j in range(1, len(poly) - 1):
            tris.append((poly[0], poly[j], poly[j + 1]))
    if not tris:
        return np.zeros((0, 3))
    tris = np.asarray(tris, float)
    ph, _ = phi_fn(tris.reshape(-1, 3))
    ph = np.asarray(ph, float).reshape(-1, 3)
    edge = np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max(1)
    cand = np.isfinite(ph).all(1) & ((np.sign(ph).min(1) != np.sign(ph).max(1)) | (np.abs(ph).min(1) <= edge))
    grids, subs, base = [], [], 0
    for t_, e_ in zip(tris[cand], edge[cand]):
        n = max(1, int(np.ceil(e_ / step)))
        ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
        idx = {k: base + m for m, k in enumerate(ij)}
        A, B, C = t_
        grids.append(np.array([A + (i / n) * (B - A) + (j / n) * (C - A) for i, j in ij]))
        for i in range(n):
            for j in range(n - i):
                subs.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
                if i + j <= n - 2:
                    subs.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
        base += len(ij)
    if not grids:
        return np.zeros((0, 3))
    G = np.concatenate(grids); S = np.asarray(subs, np.int64)
    pg, _ = phi_fn(G)
    pg = np.asarray(pg, float)
    out = []
    for s_ in S:
        f = pg[s_]
        if not np.isfinite(f).all():
            continue
        f = np.where(f == 0, 1e-300, f)
        cross = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (f[a] < 0) != (f[b] < 0):
                cross.append(G[s_[a]] + (f[a] / (f[a] - f[b])) * (G[s_[b]] - G[s_[a]]))
        if len(cross) == 2:
            p, q = cross
            k = max(1, int(np.ceil(np.linalg.norm(q - p) / spacing)))
            out.append(p + np.linspace(0., 1., k + 1)[:, None] * (q - p))
    return np.concatenate(out) if out else np.zeros((0, 3))


def indep_check(case, labels, cam, first_hits, project, in_frame, eps_floor, eps_fraction):
    """R2.2-R2.4: per pseudo path with independent points (case['paths'][pid]['indep']): visibility by the label rule,
    classes on the label image (A: a positive in the 3 x 3 neighbourhood;"""
    nat = case['native']
    target = np.asarray(labels['target'], bool); reason = np.asarray(labels['ignore_reason'])
    occl = np.asarray(labels['occluded_line'], bool)
    H, W = target.shape
    ends = fault_ends_of(case['build'])
    out, tot = {}, dict(visible=0, A=0, B=0, uncovered=0, mismatch=0, missing=0, mouth_excluded=0)
    for pid, p in sorted(case['paths'].items()):
        if not is_pseudo(pid) or not p.get('indep'):
            continue
        base, end = base_of(pid)
        e = next(x for x in ends[base] if x['end'] == end)
        P = np.asarray(p['indep'].get('points', np.zeros((0, 3))), float).reshape(-1, 3)
        rec = dict(points=int(len(P)), visible=0, A=0, B=0, uncovered=0)
        if len(P):
            uv, z = project(P, cam)
            inside = np.isfinite(uv).all(1) & (z > cam['near']) & in_frame(np.where(np.isfinite(uv), uv, -1e9), cam)
            vis = np.zeros(len(P), bool)
            if inside.any():
                hz, _hf = first_hits(nat['vertices'], nat['faces'], cam, uv[inside])
                eps = np.maximum(eps_floor, eps_fraction * z[inside] / cam['fpx'])
                vis[inside] = np.isfinite(hz) & ~(hz < z[inside] - eps)
            if vis.any():
                ij = np.floor(uv[vis] + .5).astype(np.int64)
                a_ = np.zeros(len(ij), bool); b_ = np.zeros(len(ij), bool)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        x, y = ij[:, 0] + dx, ij[:, 1] + dy
                        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                        xs, ys = np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)
                        a_ |= ok & target[ys, xs]
                        b_ |= ok & ((reason[ys, xs] > 0) | occl[ys, xs])
                rec.update(visible=int(vis.sum()), A=int(a_.sum()), B=int((~a_ & b_).sum()), uncovered=int((~a_ & ~b_).sum()))
        # R2.4
        mouth = p['indep'].get('mouth')
        if e.get('structure') is None:
            rec['record_check'] = 'no_record_r2'
        else:
            chk = P if mouth is None else P[~np.asarray(mouth, bool)]
            rec['mouth_excluded'] = 0 if mouth is None else int(np.asarray(mouth, bool).sum())
            tot['mouth_excluded'] += rec['mouth_excluded']
            if e['exposure']:
                lo = np.asarray(e['exposure']['bbox'][0], float) - INDEP_BBOX_TOL_M
                hi = np.asarray(e['exposure']['bbox'][1], float) + INDEP_BBOX_TOL_M
                outside = ~((chk >= lo) & (chk <= hi)).all(1) if len(chk) else np.zeros(0, bool)
                rec.update(record_check='bbox', outside_bbox=int(outside.sum()))
                tot['mismatch'] += int(outside.sum())
            else:
                rec.update(record_check='no_record', points_without_record=int(len(chk)))
                tot['missing'] += int(len(chk))
        for k in ('visible', 'A', 'B', 'uncovered'):
            tot[k] += rec[k]
        out[str(pid)] = rec
    tot['class_b_share_high'] = bool(tot['visible'] >= INDEP_CLASS_B_MIN and tot['B'] > INDEP_CLASS_B_SHARE * tot['visible'])
    return dict(per_path=out, **tot)
