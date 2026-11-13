"""Labels of occluded trace segments. Both envs. An occluded sample (visibility rule unchanged) with 3D distance d from
the label point to the first hit on its own ray d <= jd + jw (joint) or d <= fault_recess + damage + |t| (fault) is
occluded by its own groove / trench or scarp: its run and the groove around it become ignore (bit 32 joints, 64
faults); far-occluded samples stay negatives. The region of a run is the set of rock pixels whose CPU first-hit point
lies within the groove band of the fracture (joint: <= 2 x 1.3 x jw from its source surface; fault: <= 2 x damage from
its plane) and, along the trace, within the run's sigma interval extended by the same half-band, plus the 2 px band
around the run's samples. Positives and U pixels keep precedence (the caller applies the bits to non-positive, non-U
pixels only)."""
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from frgb.fault_traces import _point_segment_distance

JOINT_BAND = 2. * 1.3        # x jw
FAULT_BAND = 2.              # x damage
DISC_PX = 2.
SCALARS = ('jw', 'jd', 'damage', 'fault_recess')


def declaration_of(source):
    """the scene declaration of a case source, following refinement.json 'source' (as frgb.fault_traces.fault_planes)"""
    d, hops = Path(source) / 'declaration.json', 0
    while not d.is_file() and (d.parent / 'refinement.json').is_file() and hops < 4:
        d, hops = Path(json.loads((d.parent / 'refinement.json').read_text(encoding='utf8'))['source']) / 'declaration.json', hops + 1
    return d if d.is_file() else None


def scene_scalars(source):
    """jw, jd, damage, fault_recess from the scene generator record <scene>/logs/fields.log (declaration at
    <scene>/scene/geom). Every value of a key must agree where the log repeats it. Raises ValueError when the record
    or a key is missing: the label entry stops."""
    decl = declaration_of(source)
    if decl is None:
        raise ValueError('occlusion rule: no scene declaration for %s' % source)
    log = decl.parents[2] / 'logs/fields.log'
    if not log.is_file():
        raise ValueError('occlusion rule: no generator record %s' % log)
    text = log.read_text(encoding='utf8', errors='replace')
    out = {}
    for k in SCALARS:
        vals = {float(v) for v in re.findall(r'"%s":\s*([-0-9.eE+]+)' % k, text)}
        if len(vals) != 1:
            raise ValueError('occlusion rule: %s %s in %s' % (k, 'missing' if not vals else 'not unique %s' % sorted(vals), log))
        out[k] = vals.pop()
    out['record'] = str(log)
    return out


def occluder_distance(res, cam, rays):
    """3D distance from each sample P' to the first hit on its own camera ray (nan where there is none)"""
    uv = np.where(np.isfinite(res['uv']), res['uv'], 0.)
    d = rays(cam, uv[:, 0], uv[:, 1])                       # d . fwd = 1: H = pos + hz d
    hz = np.where(np.isfinite(res['hit_z']), res['hit_z'], np.nan)
    return np.linalg.norm(res['point'] - (cam['pos'] + hz[:, None] * d), axis=1)


def runs(flag):
    f = np.r_[False, np.asarray(flag, bool), False].astype(np.int8)
    e = np.flatnonzero(np.diff(f))
    return list(zip(e[0::2], e[1::2]))


def strip_distance(X, chart):
    """distance of X (chart frame) to a chart's extruded source polyline (infinite along dh) and the sigma of the nearest point"""
    Q, dh, sig = np.asarray(chart['Q'], float), np.asarray(chart['dh'], float), np.asarray(chart['sigma'], float)
    dh = dh / np.linalg.norm(dh)
    best = np.full(len(X), np.inf); bsig = np.full(len(X), np.nan)
    for i in range(len(Q) - 1):
        e = Q[i + 1] - Q[i]; e = e - (e @ dh) * dh
        ee = float(e @ e)
        if ee <= 0:
            continue
        R = X - Q[i]; R = R - (R @ dh)[:, None] * dh
        a = np.clip((R @ e) / ee, 0., 1.)
        dist = np.linalg.norm(R - a[:, None] * e, axis=1)
        better = dist < best
        best[better] = dist[better]; bsig[better] = sig[i] + a[better] * (sig[i + 1] - sig[i])
    return best, bsig


def _disc_band(uv, shape, radius=DISC_PX):
    m = np.zeros(shape, bool)
    ok = np.isfinite(uv).all(1)
    if not ok.any():
        return m
    ij = np.floor(uv[ok] + .5).astype(np.int64)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    m[ij[inside, 1], ij[inside, 0]] = True
    return ndi.distance_transform_edt(~m) <= radius if m.any() else m


def hit_points(cam, cpu_depth, rays, mask):
    vv, uu = np.nonzero(mask)
    return vv, uu, cam['pos'] + cpu_depth[vv, uu][:, None] * rays(cam, uu, vv)


def near_run(X, P, reach):
    """X is pre-filtered by the box of P widened by reach"""
    P = P[np.isfinite(P).all(1)]
    if not len(P) or not len(X):
        return np.empty(0, np.int64), np.empty(0)
    lo, hi = P.min(0) - reach, P.max(0) + reach
    idx = np.flatnonzero(np.all((X >= lo) & (X <= hi), axis=1))
    if not len(idx):
        return idx, np.empty(0)
    A, B = (P[:-1], P[1:]) if len(P) > 1 else (P, P)
    d = _point_segment_distance(X[idx], A, B)
    keep = d <= reach
    return idx[keep], d[keep]


def joint_regions(samples, cam, cpu_depth, rock, rays, scal, charts, chart_key, shift_of):
    """bit-32 region, per-sample kinds and records for the joint samples. charts[key] = dict(Q, u, dh, sigma); chart_key[path] = key;
    shift_of[path] = None or (3,) world - chart translation of the path's group."""
    shape = rock.shape
    region = np.zeros(shape, bool)
    reach, half = scal['jd'] + scal['jw'], JOINT_BAND * scal['jw']
    rec = dict(own_threshold_m=reach, band_m=half, runs=0, extent_m=reach, amendment='a (c): extent <= jd + jw, full-frame 3D candidates')
    pts = None
    for key, res in samples.items():
        d = occluder_distance(res, cam, rays)
        occ = res['occluded'] & res['in_frame']
        res['occluder_distance_m'] = np.where(occ, d, np.nan)
        res['own_occluded'] = occ & np.isfinite(d) & (d <= reach)
        res['far_occluded'] = occ & ~res['own_occluded']
        path = res.get('path', key)
        for a, b in runs(res['own_occluded']):
            rec['runs'] += 1
            sig = res['sigma'][a:b]
            s0, s1 = float(np.nanmin(sig)) - half, float(np.nanmax(sig)) + half
            band = _disc_band(res['uv'][a:b], shape)
            if pts is None:
                pts = hit_points(cam, cpu_depth, rays, rock)
            vv, uu, X = pts
            idx, _ = near_run(X, res['point'][a:b], reach)                       # (c)
            if len(idx):
                s = shift_of.get(path)
                dist, sg = strip_distance(X[idx] - (0. if s is None else s), charts[chart_key[path]])
                inside = idx[(dist <= half) & (sg >= s0) & (sg <= s1)]          # (a), (b)
                region[vv[inside], uu[inside]] = True
            region |= band & rock
    return region, rec


def fault_regions(samples, cam, cpu_depth, rock, rays, faults):
    """bit-64 region for the fault samples. faults[id] = dict(normal, point, reach=D_F, half=2 damage)"""
    shape = rock.shape
    region = np.zeros(shape, bool)
    rec = dict(runs=0, amendment='a (c): extent <= D_F, full-frame 3D candidates')
    pts = None
    for key, res in samples.items():
        f = faults[res['fault']]
        d = occluder_distance(res, cam, rays)
        occ = res['occluded'] & res['in_frame']
        res['occluder_distance_m'] = np.where(occ, d, np.nan)
        res['own_occluded'] = occ & np.isfinite(d) & (d <= f['reach'])
        res['far_occluded'] = occ & ~res['own_occluded']
        P = res['point']
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        n = np.asarray(f['normal'], float); n = n / np.linalg.norm(n); p0 = np.asarray(f['point'], float)
        for a, b in runs(res['own_occluded']):
            rec['runs'] += 1
            band = _disc_band(res['uv'][a:b], shape)
            if pts is None:
                pts = hit_points(cam, cpu_depth, rays, rock)
            vv, uu, Xall = pts
            idx, _ = near_run(Xall, P[a:b], f['reach'])                          # (c)
            if len(idx):
                X = Xall[idx]
                near = np.abs((X - p0) @ n) <= f['half']
                # along the trace: the nearest branch sample's arclength within the run's interval +- half
                fin_p = np.isfinite(P).all(1)
                j = np.flatnonzero(fin_p)[cKDTree(P[fin_p]).query(X[near])[1]] if near.any() else np.empty(0, int)
                ok = np.zeros(len(X), bool)
                ok[np.flatnonzero(near)] = (arc[j] >= arc[a] - f['half']) & (arc[j] <= arc[b - 1] + f['half'])
                region[vv[idx[ok]], uu[idx[ok]]] = True
            region |= band & rock
    return region, rec


def per_kind(res, w):
    return dict(own_occluded_px=float(w[res.get('own_occluded', np.zeros(len(w), bool))].sum()),
                far_occluded_px=float(w[res.get('far_occluded', np.zeros(len(w), bool))].sum()))
