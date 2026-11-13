"""Whole-wall scene: the B construction and the scan-detail transfer over the whole visible rock wall of an exposure
instead of one 3 m sample. Opt-in: sampling.blockface.mode == 'wall' (processes.erode); the B sample path is
unchanged.

Regions. On the surface before B, a vertex is wall rock where it is excavated (bare_w > .5), not faded under cover
(fade > .5, tile ramp < .5), occupied v2 rock, not steep to the B direction (n . -m >= .35, as blocks-3) and outside
the fault trenches (fault groove profile <= .05). Faces with three such vertices in one fault block, connected, >=
min_area, form the regions; each has a taper (smoothstep of the distance to its boundary, 0 within .15 m, 1 beyond .6
m) shared by the spalling and the scan detail. One B direction m for the whole wall (area-weighted mean inward normal
of the candidate faces), so the donor frame (a1 along the bedding trace, a2 = -m x a1) is one frame.

Scan detail. Square footprints (side W, blend b) on a brick layout in the plane normal to m: pitch P = W - b along a1
and a2, odd rows shifted by P / 2; the window weights of neighbours form a partition of unity; colour c = (k + 2 (r
mod 2)) mod 3 never repeats between overlapping footprints. Each footprint takes a square window of the donor atlas
(scanmesh.DonorWindow, translation only) assigned so that footprints within ``neighbour_m`` take windows at least
``min_source_sep_m`` apart in the atlas. The footprints of one colour are solved in parallel on their own sub-meshes
(refine_displace_mesh with the region mask, the taper as vertex weight, the imaging-scale tolerance) and merged; the
colours follow one another.

B. One region_setup / build_patch_parallel per region on the detailed mesh, zoned weld (1 um where the detail weight
is > 0, 0.5 mm elsewhere), triangulate; rule-2 records across regions."""
import math
import time
from pathlib import Path

import numpy as np

SENSOR_W_MM = 36.
TOL_MIN, TOL_MAX = 5e-4, 6e-3
CONSTRAIN_BELOW = 1e-3


# ---------------------------------------------------------------- imaging-scale tolerance
class Cameras:
    """Formal cameras: positions, unit forward / right / up, focal (mm), resolution (w, h)."""

    def __init__(self, cams, tol_min=TOL_MIN, tol_max=TOL_MAX):
        self.cams = [{k: (np.asarray(v, float) if k in ('position', 'forward', 'right', 'up') else v) for k, v in c.items()} for c in cams]
        self.tol_min, self.tol_max = float(tol_min), float(tol_max)

    def tolerance(self, P):
        """0.5 x pixel footprint of the nearest camera whose frustum (10 % enlarged) contains P, clamped."""
        P = np.atleast_2d(np.asarray(P, float))
        best = np.full(len(P), np.inf)
        for c in self.cams:
            d = P - c['position']
            z = d @ c['forward']
            w, h = c['resolution']
            half_w = .5 * SENSOR_W_MM / c['focal_mm'] * 1.1
            half_h = half_w * h / w
            ok = (z > .05) & (np.abs(d @ c['right']) <= half_w * z) & (np.abs(d @ c['up']) <= half_h * z)
            px = z * SENSOR_W_MM / (c['focal_mm'] * w)
            best = np.where(ok, np.minimum(best, .5 * px), best)
        return np.clip(np.where(np.isfinite(best), best, self.tol_max), self.tol_min, self.tol_max)


class FootTol:
    """tol_of / min_split_of / constrain_of of one footprint (2-D footprint coordinates -> 3-D on the surface
    plane of the footprint), picklable."""

    def __init__(self, cams, cf, a1, a2, constrain_below=CONSTRAIN_BELOW, min_split=1e-3):
        self.cams, self.cf, self.a1, self.a2 = cams, np.asarray(cf, float), np.asarray(a1, float), np.asarray(a2, float)
        self.constrain_below, self.min_split = float(constrain_below), float(min_split)

    def _p3(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return self.cf + X[:, :1] * self.a1 + X[:, 1:2] * self.a2

    def tol(self, X):
        return self.cams.tolerance(self._p3(X))

    def min_split_of(self, X):
        return np.maximum(self.min_split, self.tol(X))

    def constrain(self, X):
        return self.tol(X) <= self.constrain_below


# ---------------------------------------------------------------- regions
def _faults(truth):
    from .truth import _plane_groups
    return [(n, o) for iid, iv, if_ in truth.interfaces if iid.startswith('fault') and len(if_) for n, o, _ in _plane_groups(iv, if_)]


def block_ids(P, faults):
    return sum(((P @ n - o) > 0).astype(np.int64) << i for i, (n, o) in enumerate(faults)) if faults else np.zeros(len(P), np.int64)


def _components(F, faces):
    """Edge-connected components of a face subset: label per face of ``faces`` (0..k-1)."""
    faces = np.asarray(faces)
    E = np.concatenate([F[faces][:, [0, 1]], F[faces][:, [1, 2]], F[faces][:, [2, 0]]])
    E.sort(1)
    owner = np.tile(np.arange(len(faces)), 3)
    o = np.lexsort(E.T[::-1])
    E, owner = E[o], owner[o]
    same = (E[1:] == E[:-1]).all(1)
    parent = np.arange(len(faces))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(owner[:-1][same], owner[1:][same]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(len(faces))])
    _, lab = np.unique(root, return_inverse=True)
    return lab


def _dist_taper(V, F, fk, t0, t1):
    """smoothstep taper of the vertices of faces fk by their distance to the boundary of that face set; and stats."""
    from .display import segment_distance
    E = np.concatenate([F[fk][:, [0, 1]], F[fk][:, [1, 2]], F[fk][:, [2, 0]]])
    E.sort(1)
    u, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = u[cnt == 1]
    vs = np.unique(F[fk])
    d, _ = segment_distance(V[vs], np.stack([V[bnd[:, 0]], V[bnd[:, 1]]], 1))
    x = np.clip((np.asarray(d, float) - t0) / (t1 - t0), 0, 1)
    return vs, x * x * (3 - 2 * x), {'boundary_edges': int(len(bnd)), 'boundary_length_m': float(np.linalg.norm(V[bnd[:, 1]] - V[bnd[:, 0]], axis=1).sum())}


def fold_faces(V, F, faces, m, cell=.02, eps=1e-12):
    """Faces of ``faces`` whose projections along m overlap another face of the set with positive area (the B prisms
    of overlapping tops would overlap: the region is not a height field along m there). Projected 2-D triangles are
    binned in ``cell`` squares; pairs sharing a vertex are skipped (consistently wound neighbours do not overlap);
    separating-axis test with a small tolerance. Returns a bool mask over ``faces``."""
    faces = np.asarray(faces)
    m = np.asarray(m, float)
    g1 = np.cross(m, [0., 0., 1.])
    g1 /= np.linalg.norm(g1)
    g2 = np.cross(m, g1)
    T = np.stack([V[F[faces]] @ g1, V[F[faces]] @ g2], -1)          # (n, 3, 2)
    lo = np.floor(T.min(1) / cell).astype(np.int64)
    hi = np.floor(T.max(1) / cell).astype(np.int64)
    nx, ny = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    cnt = nx * ny
    fi = np.repeat(np.arange(len(faces)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    key = (lo[fi, 0] + off // ny[fi]) * (1 << 32) + (lo[fi, 1] + off % ny[fi])
    o = np.argsort(key, kind='stable')
    key, fi = key[o], fi[o]
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    end = np.r_[start[1:], len(key)]
    pa, pb = [], []
    for k in range(1, int((end - start).max()) if len(start) else 1):
        sel = np.flatnonzero(end - start > k)
        if not len(sel):
            break
        i0 = np.repeat(start[sel], end[sel] - start[sel] - k)
        r_ = np.arange(len(i0)) - np.repeat(np.cumsum(end[sel] - start[sel] - k) - (end[sel] - start[sel] - k), end[sel] - start[sel] - k)
        a_ = fi[i0 + r_]
        b_ = fi[i0 + r_ + k]
        pa.append(a_)
        pb.append(b_)
    out = np.zeros(len(faces), bool)
    if not pa:
        return out
    a, b = np.concatenate(pa), np.concatenate(pb)
    pr = np.unique(np.sort(np.c_[a, b], 1), axis=0)
    a, b = pr[:, 0], pr[:, 1]
    share = (F[faces[a]][:, :, None] == F[faces[b]][:, None, :]).any((1, 2))
    a, b = a[~share], b[~share]
    if not len(a):
        return out
    Ta, Tb = T[a], T[b]
    sep = np.zeros(len(a), bool)
    for X, Y in ((Ta, Tb), (Tb, Ta)):
        for e in range(3):
            d = X[:, (e + 1) % 3] - X[:, e]
            nrm = np.c_[-d[:, 1], d[:, 0]]
            px = np.einsum('nkd,nd->nk', X, nrm)
            py = np.einsum('nkd,nd->nk', Y, nrm)
            sep |= (px.max(1) <= py.min(1) + eps) | (py.max(1) <= px.min(1) + eps)
    hit = ~sep
    out[a[hit]] = True
    out[b[hit]] = True
    return out


def wall_regions(V, F, outward_v, fields, truth, opts):
    """Regions of the wall (see module doc). fields: bare_w, fade, fault_zone (per vertex), tile_ramp (callable or None).
    Two face sets: the detail region (scan detail; excavated occupied rock, fade > fade_min, tile ramp < tile_max, not
    steep to m) and inside it the B regions (also outside the fault-trench cores, fault_zone <= fault_zone_max, and in
    one fault block: one region per block component). Tapers: distance to the boundary of the detail / own B region,
    times ``fade`` (the cover fade of the procedural relief, so both fade out with it towards the soil).
    Returns dict(m, regions=[{'faces', 'block', 'area_m2', ...}], taper (B), taper_detail, detail_faces,
    vertex_region (-1 none), stats)."""
    from .display import vertex_normals
    T = V[F]
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    if (fn * outward_v[F].mean(1)).sum() < 0:
        fn = -fn
    area = .5 * np.linalg.norm(fn, axis=1)
    fnu = fn / np.maximum(2 * area, 1e-30)[:, None]
    st, _, _, _ = truth.query(V)
    occ = st == 'occupied'
    tile = fields['tile_ramp'](V) if fields.get('tile_ramp') is not None else np.zeros(len(V))
    fade = np.asarray(fields['fade'], float)
    fz_max = float(opts.get('fault_zone_max', .5))
    base_d = (fields['bare_w'] > .5) & (fade > float(opts.get('fade_min', .05))) & (tile < float(opts.get('tile_max', .95))) & occ
    cand = (base_d & (fields['fault_zone'] <= fz_max))[F].all(1)
    mvec = -(fnu[cand] * area[cand, None]).sum(0)
    mvec /= np.linalg.norm(mvec)
    N = vertex_normals(V, F, np.array([0., 0., 1.]), consistent=True)
    if (N * outward_v).sum() < 0:
        N = -N
    steep_ok = ((N @ -mvec) >= float(opts.get('min_cos_m', .35)))
    face_ok = (fnu @ -mvec) > .05
    faults = _faults(truth)
    blk = block_ids(V, faults)
    # fault clearance (B only): no B where a fault plane passes within fault_clear_m of the surface along m, so no
    # spalled cell (depth <= B depth) and no displaced vertex reaches across the fault
    clear = float(opts.get('fault_clear_m', .4))
    near_fault = np.zeros(len(V), bool)
    for n_, o_ in faults:
        mn = float(np.asarray(n_) @ mvec)
        dist = np.abs(V @ n_ - o_) / max(abs(mn), 1e-6)
        near_fault |= dist < clear
    t0, t1 = float(opts.get('taper_zero_m', .15)), float(opts.get('taper_full_m', .6))
    min_a = float(opts.get('min_region_m2', 1.))
    # detail region: components >= min area (blocks irrelevant)
    win_ok = np.ones(len(V), bool)
    if opts.get('window'):
        win_ok = np.linalg.norm(V - np.asarray(opts['window']['center'], float), axis=1) <= float(opts['window']['radius_m'])
    df = np.flatnonzero((base_d & steep_ok & win_ok)[F].all(1) & face_ok)
    lab = _components(F, df) if len(df) else np.zeros(0, int)
    keep = np.zeros(len(df), bool)
    for k in range(lab.max() + 1 if len(lab) else 0):
        if area[df[lab == k]].sum() >= min_a:
            keep[lab == k] = True
    df = df[keep]
    taper_d = np.zeros(len(V))
    dstat = {}
    if len(df):
        vs, tv, dstat = _dist_taper(V, F, df, t0, t1)
        taper_d[vs] = tv
    # B regions: inside the detail region, outside the fault cores, one fault block per region
    in_d = np.zeros(len(F), bool)
    in_d[df] = True
    rf = in_d & (fields['fault_zone'] <= fz_max)[F].all(1) & ~near_fault[F].any(1) & (blk[F] == blk[F][:, :1]).all(1)
    faces = np.flatnonzero(rf)
    folded = 0
    if opts.get('heightfield'):
        # surface along m, e.g. the cut toe) and one ring of their neighbours, before regions and tapers are formed
        fm = fold_faces(V, F, faces, mvec)
        if fm.any():
            bad_v = np.zeros(len(V), bool)
            bad_v[F[faces[fm]].reshape(-1)] = True
            ring = bad_v[F[faces]].any(1)
            folded = int(ring.sum())
            faces = faces[~ring]
    regions = []
    vreg = np.full(len(V), -1, np.int64)
    conflicts = 0
    taper = np.zeros(len(V))
    for b in np.unique(blk[F[faces, 0]]):
        fb = faces[blk[F[faces, 0]] == b]
        lab = _components(F, fb)
        for k in range(lab.max() + 1):
            fk = fb[lab == k]
            a = float(area[fk].sum())
            if a < min_a:
                continue
            r = len(regions)
            vs = np.unique(F[fk])
            conflicts += int((vreg[vs] >= 0).sum())
            vreg[vs] = r
            vs2, tv, st_ = _dist_taper(V, F, fk, t0, t1)
            taper[vs2] = np.maximum(taper[vs2], tv)
            regions.append({'faces': fk, 'block': int(b), 'area_m2': a, **st_})
    taper *= np.clip(fade, 0, 1)
    taper_d *= np.clip(fade, 0, 1)
    exc = (fields['bare_w'] > .5)[F].all(1) & occ[F].all(1)
    rules = {'fade<=fade_min': ~(fade > float(opts.get('fade_min', .05)))[F].all(1), 'tile_ramp>=tile_max': ~(tile < float(opts.get('tile_max', .95)))[F].all(1),
             'fault_core (B only)': ~(fields['fault_zone'] <= fz_max)[F].all(1), 'steep_to_m': ~steep_ok[F].all(1) | ~face_ok,
             'crossing_blocks (B only)': ~(blk[F] == blk[F][:, :1]).all(1), 'fault_clearance (B only)': near_fault[F].any(1)}
    stats = {'m': mvec.tolist(), 'excavated_rock_area_m2': float(area[exc].sum()),
             'excluded_area_m2_by_rule': {k: float(area[exc & v].sum()) for k, v in rules.items()},
             'detail_area_m2': float(area[df].sum()), 'detail_boundary': dstat, 'region_area_m2': float(sum(R['area_m2'] for R in regions)),
             'regions': [{k: v for k, v in R.items() if k != 'faces'} for R in regions], 'vertex_region_conflicts': conflicts,
             'blocks_on_wall': sorted(set(int(R['block']) for R in regions)), 'rules': {k: opts.get(k) for k in ('fault_zone_max', 'fade_min', 'tile_max', 'min_cos_m', 'fault_clear_m', 'heightfield')},
             'heightfield_faces_removed': folded}
    return {'m': mvec, 'regions': regions, 'taper': taper, 'taper_detail': taper_d, 'detail_faces': df, 'vertex_region': vreg,
            'stats': stats, 'block_of_vertex': blk}


# ---------------------------------------------------------------- footprint layout and donor windows
def axes(truth, P, m):
    """a1 along the bedding trace on the plane normal to m, a2 = -m x a1 up (scandetail.trace_axes), from the
    bedding normal at the region points (must be one direction: no folds)."""
    from .scandetail import trace_axes
    st, body, _, _ = truth.query(P)
    occ = st == 'occupied'
    ok, nbs, _ = truth.fabric(P[occ], body[occ], 'bedding')
    nb = nbs[ok][0]
    if np.abs(nbs[ok] @ nb).min() < 1 - 1e-9:
        raise ValueError('bedding normal varies over the wall: the single donor frame needs planar beds')
    return trace_axes(nb, m), nb


def layout(V, F, detail_faces, taper, a1, a2, m, W=1.3, blend=.15, margin=.1):
    """Brick layout over the detail-region vertices with taper > 0 (see module doc). Returns origin o and a list of
    footprints {id, row, col, colour, xy, centre (3-D, on the surface along m), region_vertices}."""
    vs = np.unique(F[np.asarray(detail_faces)].reshape(-1)) if len(detail_faces) else np.zeros(0, int)
    vs = vs[taper[vs] > 0] if len(vs) else vs
    P = V[vs]
    o = P.mean(0)
    x, y = (P - o) @ a1, (P - o) @ a2
    pitch = W - blend
    r0, r1 = int(np.floor((y.min() - W / 2) / pitch)) - 1, int(np.ceil((y.max() + W / 2) / pitch)) + 1
    feet = []
    for r in range(r0, r1 + 1):
        sh = .5 * pitch if r % 2 else 0.
        k0, k1 = int(np.floor((x.min() - W / 2 - sh) / pitch)) - 1, int(np.ceil((x.max() + W / 2 - sh) / pitch)) + 1
        yc = r * pitch
        rowsel = np.abs(y - yc) < W / 2
        if not rowsel.any():
            continue
        for k in range(k0, k1 + 1):
            xc = k * pitch + sh
            inn = rowsel & (np.abs(x - xc) < W / 2)
            if not inn.any():
                continue
            # the footprint's plane passes through the surface at its centre: median offset along m of its vertices
            dm = float(np.median((P[inn] - o) @ m))
            feet.append({'id': len(feet), 'row': r, 'col': k, 'colour': int((k + 2 * (r % 2)) % 3), 'xy': [xc, yc],
                         'centre': (o + xc * a1 + yc * a2 + dm * m).tolist(), 'region_vertices': int(inn.sum())})
    return o, feet, {'pitch_m': pitch, 'W_m': W, 'blend_m': blend, 'rows': [r0, r1], 'footprints': len(feet)}


def assign_windows(feet, usable, seed, min_sep=.5, neighbour=3.):
    """One atlas window centre per footprint: random among the usable centres at least min_sep (atlas metres) from
    the windows of the already assigned footprints within ``neighbour`` (wall metres); if none qualifies, the
    candidate farthest from them. Deterministic for a seed."""
    U = np.asarray(usable, float)
    g = np.random.default_rng(seed)
    out = []
    XY = np.array([f['xy'] for f in feet], float)
    for i, f in enumerate(feet):
        near = [j for j in range(i) if np.linalg.norm(XY[j] - XY[i]) < neighbour]
        if near:
            S = np.array([out[j] for j in near])
            dmin = np.sqrt(((U[:, None, :] - S[None]) ** 2).sum(-1)).min(1)
            ok = np.flatnonzero(dmin >= min_sep)
            pick = int(g.choice(ok)) if len(ok) else int(np.argmax(dmin))
        else:
            pick = int(g.integers(len(U)))
        out.append(U[pick].tolist())
    for f, w in zip(feet, out):
        f['window'] = w
    seps = []
    for i in range(len(feet)):
        for j in range(i):
            if np.linalg.norm(XY[j] - XY[i]) < neighbour:
                seps.append(float(np.linalg.norm(np.asarray(out[i]) - np.asarray(out[j]))))
    return {'min_source_sep_m': min_sep, 'neighbour_m': neighbour, 'pairs_within_neighbour': len(seps),
            'source_sep_m_min_p5_p50': [float(x) for x in np.percentile(seps, [0, 5, 50])] if seps else None}


# ---------------------------------------------------------------- per-footprint refinement (worker)
_ATLAS = {}


def _atlas(path):
    if path not in _ATLAS:
        from . import scanmesh
        _ATLAS[path] = scanmesh.load(Path(path) / 'mesh_donor.npz')
    return _ATLAS[path]


def _refine_job(job):
    from .scandetail import refine_displace_mesh
    from .scanmesh import DonorWindow
    t0 = time.time()
    don = DonorWindow(_atlas(job['atlas']), job['window'], job['W'])
    ft = job['tol']
    opts = dict(job['opts'])
    opts.update(face_ok=job['face_ok'], vertex_weight=job['vweight'], tol_of=ft.tol, min_split_of=ft.min_split_of, constrain_of=ft.constrain)
    res = refine_displace_mesh(job['V'], job['F'], job['cf'], job['a1'], job['a2'], job['m'], don, opts)
    loc = res['local']
    out = {'id': job['id'], 'faces_replaced': res['faces_replaced'], 'faces': res['faces'], 'new_positions': res['new_positions'],
           'moved': np.array(sorted(res['moved_original'])), 'parents': res['parents'], 'stats': res['stats'], 'n_sub': len(job['V']),
           'gidx': loc['gidx'], 'disp': loc['disp'], 'seconds': time.time() - t0}
    out['moved_pos'] = np.array([res['moved_original'][k] for k in out['moved']]) if len(out['moved']) else np.zeros((0, 3))
    return out


def _grow(fields, nV, parents):
    """Append the parent-interpolated values of new vertices nV.. to every per-vertex array in ``fields`` (dict of
    arrays, 1-D or 2-D); parents: list of (idx tuple, weight tuple) in creation order (indices < own index)."""
    n = len(parents)
    if not n:
        return fields
    k = max(len(p[0]) for p in parents)
    I = np.zeros((n, k), np.int64)
    Wt = np.zeros((n, k))
    lev = np.zeros(n, np.int64)
    for j, (idx, wts) in enumerate(parents):
        I[j, :len(idx)] = idx
        Wt[j, :len(wts)] = wts
        dep = [lev[i - nV] for i in idx if i >= nV]
        if dep:
            lev[j] = 1 + max(dep)
    out = {}
    for name, a in fields.items():
        ext = np.concatenate([a, np.zeros((n,) + a.shape[1:], a.dtype)])
        for L in range(int(lev.max()) + 1):
            sel = np.flatnonzero(lev == L)
            if not len(sel):
                continue
            vals = ext[I[sel]]
            wt = Wt[sel]
            ext[nV + sel] = (vals * (wt[:, :, None] if a.ndim == 2 else wt)).sum(1)
        out[name] = ext
    return out


def apply_detail(V, F, fields, wr, feet, cams, opts, atlas_path, workers=1, log=None):
    """Scan detail of all footprints (see module doc). fields: dict of per-vertex arrays (float) carried through
    (grown by parent interpolation), must include 'taper' (B) and 'taper_d' (detail). Returns V, F, fields, dvec (accumulated scan displacement,
    per vertex), face_region (per face, -1 outside), stats."""
    from concurrent.futures import ProcessPoolExecutor
    m = wr['m']
    a1, a2 = opts['a1'], opts['a2']
    W, blend, margin = float(opts.get('W_m', 1.3)), float(opts.get('blend_m', .15)), float(opts.get('margin_m', .1))
    face_region = np.full(len(F), -1, np.int64)
    for r, R in enumerate(wr['regions']):
        face_region[R['faces']] = r
    vreg = wr['vertex_region'].copy()
    face_det = np.zeros(len(F), bool)
    face_det[wr['detail_faces']] = True
    dvec = np.zeros((len(V), 3))
    base_opts = {k: v for k, v in opts['detail'].items() if k in ('blend_m', 'margin_m', 'snap_m', 'amplitude', 'min_edge_m', 'min_height_m',
                                                                    'steep_flip', 'max_lin_err_m', 'min_split_m')}
    base_opts['weight_error'] = True
    stats = {'footprints': [], 'colours': {}}
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for colour in (0, 1, 2):
            tc = time.time()
            group = [f for f in feet if f['colour'] == colour]
            jobs = []
            for f in group:
                cf = np.asarray(f['centre'], float)
                Q = np.c_[(V - cf) @ a1, (V - cf) @ a2]
                near_v = (np.abs(Q) <= W / 2 + margin + 1e-6).all(1)
                sub_f = np.flatnonzero(near_v[F].any(1) & face_det)
                if not len(sub_f):
                    continue
                used, loc = np.unique(F[sub_f], return_inverse=True)
                jobs.append({'id': f['id'], 'atlas': str(atlas_path), 'window': f['window'], 'W': W, 'V': V[used], 'F': loc.reshape(-1, 3),
                             'face_ok': np.ones(len(sub_f), bool), 'vweight': fields['taper_d'][used], 'cf': cf, 'a1': a1, 'a2': a2, 'm': m,
                             'opts': base_opts, 'tol': FootTol(cams, cf, a1, a2, float(opts.get('constrain_below', CONSTRAIN_BELOW)),
                                                                 float(opts.get('min_split_floor_m', 1e-3))),
                             '_sub_f': sub_f, '_used': used})
            payload = [{k: v for k, v in j.items() if not k.startswith('_')} for j in jobs]
            results = list(ex.map(_refine_job, payload)) if ex is not None and len(jobs) > 1 else [_refine_job(p) for p in payload]
            # merge the colour at once (its footprints are disjoint): new vertices of footprint after footprint
            nV = len(V)
            off = nV
            parents, newpos, moved_i, moved_p, rm_f, add_f, disp_g, disp_v = [], [], [], [], [], [], [], []
            for j, res in zip(jobs, results):
                used, sub_f = j['_used'], j['_sub_f']
                nn = len(res['new_positions'])
                gmap = np.concatenate([used, off + np.arange(nn)])
                parents += [(tuple(int(gmap[i]) for i in idx), wts) for idx, wts in res['parents']]
                newpos.append(res['new_positions'])
                if len(res['moved']):
                    moved_i.append(used[res['moved']])
                    moved_p.append(res['moved_pos'])
                rm_f.append(sub_f[np.asarray(res['faces_replaced'], np.int64)])
                add_f.append(gmap[np.asarray(res['faces'], np.int64)])
                disp_g.append(gmap[np.asarray(res['gidx'], np.int64)])
                disp_v.append(np.asarray(res['disp'], float))
                off += nn
                st = {k: res['stats'].get(k) for k in ('region_faces_in', 'region_faces_out', 'vertices_added', 'constraints_inserted',
                                                       'constraints_failed', 'adaptive_splits', 'adaptive_passes', 'vertices_without_donor_surface',
                                                       'projected_orientation_consistent', 'steep_flip', 'short_edges_left', 'flat_triangles_left')}
                stats['footprints'].append({'id': j['id'], 'colour': colour, 'seconds': res['seconds'], **st})
                if log:
                    log('footprint %d (colour %d): %d -> %d faces, %.1f s' % (j['id'], colour, st['region_faces_in'] or 0, st['region_faces_out'] or 0, res['seconds']))
            if not jobs:
                stats['colours'][str(colour)] = {'footprints': 0, 'seconds': time.time() - tc}
                continue
            fields = _grow({**fields, '_dvec': dvec}, nV, parents)
            dvec = fields.pop('_dvec')
            # region id of new vertices: their parents' common id (parents may be earlier new vertices)
            vreg = np.concatenate([vreg, np.full(len(parents), -1, np.int64)])
            for j, (idx, _) in enumerate(parents):
                ids = {int(vreg[i]) for i in idx}
                if len(ids) == 1:
                    vreg[nV + j] = ids.pop()
            V = np.concatenate([V] + newpos)
            if moved_i:
                V[np.concatenate(moved_i)] = np.concatenate(moved_p)
            # the scan displacement of the colour's footprints, along -m (local disp d: P - d m)
            dvec[np.concatenate(disp_g)] += -np.concatenate(disp_v)[:, None] * m[None]
            keep = np.ones(len(F), bool)
            keep[np.concatenate(rm_f)] = False
            newF = np.concatenate(add_f)
            rv = vreg[newF]
            fr_new = np.where((rv == rv[:, :1]).all(1), rv[:, 0], -1)
            F = np.concatenate([F[keep], newF])
            face_region = np.concatenate([face_region[keep], fr_new])
            face_det = np.concatenate([face_det[keep], np.ones(len(newF), bool)])
            stats.setdefault('new_faces_unlabelled', 0)
            stats['new_faces_unlabelled'] += int((fr_new < 0).sum())
            stats['colours'][str(colour)] = {'footprints': len(jobs), 'seconds': time.time() - tc}
    finally:
        if ex is not None:
            ex.shutdown()
    if opts.get('dump_dir'):
        dump = Path(opts['dump_dir'])
        dump.mkdir(parents=True, exist_ok=True)
        with (dump / 'preb_detail.npz').open('xb') as stream:
            np.savez(stream, V=V, F=F, dvec=dvec, taper_d=fields['taper_d'],
                     face_detail=face_det, m=m, a1=a1, a2=a2)
    return V, F, fields, dvec, face_region, vreg, stats


# ---------------------------------------------------------------- B per region
def wall_weld_tolerance(V, detail_w, m, cut, coarse, fine):
    """Position-based scan-footprint tolerance, protected against crossing cut planes.

    The fine zone follows the footprint perpendicular to m through the block
    depth, as in the accepted B sample. Cut-plane clearance prevents a floor
    vertex from being pulled onto a neighbouring cell's plane. The roundoff
    floor joins numerically coincident cut corners without moving them off-plane.
    """
    m = np.asarray(m, float)
    g1 = np.cross(m, [0., 0., 1.])
    g1 /= np.linalg.norm(g1)
    g2 = np.cross(m, g1)
    basis = np.stack([g1, g2], 1)
    zc, Bk = .15, 1 << 20
    code = lambda K: (K[:, 0] + Bk) * (2 * Bk) + K[:, 1] + Bk
    Kd = np.unique(np.floor((V[detail_w > 0] @ basis) / zc).astype(np.int64), axis=0)
    kd = np.sort(code(Kd))

    def tolerance(P):
        P = np.asarray(P, float)
        K = np.floor((P @ basis) / zc).astype(np.int64)
        hit = np.zeros(len(P), bool)
        if len(kd):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    c = code(K + np.array([dx, dy]))
                    pos = np.minimum(np.searchsorted(kd, c), len(kd) - 1)
                    hit |= kd[pos] == c
        t = np.where(hit, fine, coarse)
        if cut:
            clearance = np.full(len(P), np.inf)
            for _, _, n, d in cut:
                np.minimum(clearance, np.abs(P @ n - d), out=clearance)
            t = np.minimum(t, .49 * np.maximum(clearance, 2e-13))
        return t
    return tolerance


def apply_blocks(V, F, face_region, taper, detail_w, truth, m, regions, opts, workers=1, log=None):
    """B construction of every region on the detailed mesh (region_setup, build_patch_parallel, zoned weld,
    triangulate). detail_w: per-vertex scan-detail weight (> 0: the 1 um weld zone). Returns dict with the
    concatenated new geometry (bv, bfaces (into V + bv), bprism, bbary, bkind, code (rule-2 record + 1 per new
    face), nout per new face), the faces removed, the rule-2 records and per-region info."""
    from .blockface import region_setup, build_patch_parallel, weld_slivers, triangulate, _area_normal
    from .wall_evidence import save_polygons
    wtol = float(opts.get('weld_tol', 5e-4))
    fine = float(opts.get('weld_fine_m', 1e-6))
    out = {'bv': [], 'bfaces': [], 'bprism': [], 'bbary': [], 'bkind': [], 'code': [], 'nout': [], 'regions': [], 'breg': [], 'bcell': []}
    dump = Path(opts['dump_dir']) if opts.get('dump_dir') else None
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
    cop_records, rec_of = [], {}
    removed = []
    nb_total = 0
    for r in range(len(regions)):
        t0 = time.time()
        inside = np.flatnonzero(face_region == r)
        if not len(inside):
            continue
        (inside, _, _, cut, cell_of, depth_of, info), cells = region_setup(
            V, F, truth, inside, taper, m, depth=float(opts['depth']), parting=float(opts['parting']), seed=int(opts['seed']), region=r,
            spall=opts.get('spall'))
        t1 = time.time()
        fast_out = None
        bottom = float(opts['bottom_depth_m']) if opts.get('bottom_depth_m') else 1.
        if opts.get('fast'):
            # only for the prisms a cut plane passes through, with prisms just deeper than the deepest floor
            from .blockface import build_region_fast
            fmask, fkeys, fR, _ = build_region_fast(V, F, inside, taper, m, cut, cells, float(opts['depth']), intact={'intact': True, 'uniform': 'uniform'}.get(opts.get('fast'), False))
            bottom = float(opts.get('bottom_depth_m') or (float(opts['depth']) * 1.02 + 1e-3))
            fast_faces = inside[fmask]
            inside_slow = inside[~fmask]
            fF = F[fast_faces]
            vk = np.c_[fF.reshape(-1), np.repeat(fkeys, 3)]
            uq, inv = np.unique(vk, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            Rv = np.repeat(fR, 3)[np.unique(inv, return_index=True)[1]]
            first = np.unique(inv, return_index=True)[1]          # first fast-face corner of each new vertex
            vid = uq[:, 0]
            fv = V[vid] + (m[None, :] * Rv[:, None]) * taper[vid][:, None]
            ff = inv.reshape(-1, 3)
            # outward winding (-m), as the clipping path's floor polygons
            Tn = np.cross(fv[ff[:, 1]] - fv[ff[:, 0]], fv[ff[:, 2]] - fv[ff[:, 0]])
            flip = (Tn @ (-m)) < 0
            ff[flip] = ff[flip][:, ::-1]
            bary_f = np.zeros((len(uq), 3))
            bary_f[np.arange(len(uq)), first % 3] = 1.
            fast_out = {'bv': fv, 'bfaces': ff, 'bprism': fast_faces[first // 3], 'bbary': bary_f, 'keys': fkeys,
                        'nout': Tn / np.maximum(np.linalg.norm(Tn, axis=1, keepdims=True), 1e-300) * np.where(flip, -1, 1)[:, None]}
        else:
            inside_slow = inside
        if dump is not None:
            # geometry acceptance inputs (wallcheck.py): the region's prism tops (detailed surface), taper, cells, planes
            used_, loc_ = np.unique(F[inside], return_inverse=True)
            np.savez_compressed(dump / ('region_%d.npz' % r), V=V[used_], F=loc_.reshape(-1, 3), inside=inside, used=used_, taper=taper[used_], m=m,
                                cut_n=np.array([c_[2] for c_ in cut], float).reshape(-1, 3), cut_d=np.array([c_[3] for c_ in cut], float),
                                cut_fam=np.array([c_[0] for c_ in cut]), cut_lev=np.array([c_[1] for c_ in cut]),
                                levels=cells.levels, C=cells.C, nb=cells.nb, jsets=np.array(cells.jsets),
                                joint_n=np.array([cells.joint_offs[k_][0] for k_ in cells.jsets], float).reshape(-1, 3),
                                joint_offs=np.array([cells.joint_offs[k_][1] for k_ in cells.jsets], dtype=object) if cells.jsets else np.zeros(0),
                                seed=cells.seed, depth=cells.depth, region=r,
                                **({'spall': True, 'cross_n': np.array([c_[0] for c_ in cells.cross], float),
                                    'cross_offs': np.array([c_[1] for c_ in cells.cross], dtype=object),
                                    'table_keys': np.array(sorted(cells.table), np.int64),
                                    'table_R': np.array([cells.table[k_] for k_ in sorted(cells.table)], float)}
                                   if hasattr(cells, 'table') else {}), allow_pickle=True)
        # scan route keeps the historical 1e-7 / rounded keys
        polys, bst = build_patch_parallel(V, F, inside_slow, taper, m, cut, cells, exact=True, workers=workers,
                                          floor_cap_tol=float(opts.get('floor_cap_tol', 1e-7)), bottom_depth=bottom)
        if fast_out is not None:
            bst['fast'] = {'fast_prisms': int(len(inside) - len(inside_slow)), 'clipped_prisms': int(len(inside_slow)),
                           'fast_vertices': int(len(fast_out['bv'])), 'fast_spalled_prisms': int((np.asarray(fR) > 0).sum()),
                           'bottom_depth_m': bottom}
        fast_extra = None
        if fast_out is not None:
            fast_extra = (fast_out['bv'][fast_out['bfaces']], np.asarray(fast_out['keys'], np.int64), fast_faces)
        if dump is not None:
            save_polygons(dump / ('raw_region_%d.npz' % r), polys, extra=fast_extra)
        t2 = time.time()
        tol_of_pts = wall_weld_tolerance(V, detail_w, m, cut, wtol, fine)
        polys, bst['weld'] = weld_slivers(polys, cut, tol=wtol, max_move=wtol, tol_of=tol_of_pts, fine_min_area=0., fast=True, fine_min_width=float(opts.get('fine_min_width_m', 1e-6)))
        if opts.get('nano_merge_m'):
            from .blockface import merge_nano
            polys, bst['nano_merge'] = merge_nano(polys, cut, float(opts['nano_merge_m']))
            if not bst['nano_merge']['plane_membership_preserved']:
                raise RuntimeError('merge_nano moved a corner off a cut plane')
        if dump is not None:
            save_polygons(dump / ('welded_region_%d.npz' % r), polys, extra=fast_extra)
        t3 = time.time()
        bv, bfaces, bprism, bbary, bkind, bfpoly = triangulate(polys, V, F, m, exact_floor_vertices=bool(opts.get('exact_floor_vertices', False)))
        if opts.get('thin_triangle_width_m'):
            # (blockface._deletable_slivers: delete only if the left hole closes within the threshold) is applied to the
            # triangles of width (2 area / perimeter) below thin_triangle_width_m.
            from .blockface import _deletable_slivers
            Tt = bv[bfaces]
            ar = .5 * np.linalg.norm(np.cross(Tt[:, 1] - Tt[:, 0], Tt[:, 2] - Tt[:, 0]), axis=1)
            pr = np.linalg.norm(Tt - Tt[:, [1, 2, 0]], axis=2).sum(1)
            thin = np.flatnonzero(2 * ar / np.maximum(pr, 1e-300) < float(opts['thin_triangle_width_m']))
            drop = set()
            if len(thin):
                tri_polys = {int(i): (list(Tt[i]), bkind[i], {}) for i in thin}
                drop = _deletable_slivers(tri_polys, [int(i) for i in thin], float(opts['thin_triangle_width_m']))
            keep_t = np.ones(len(bfaces), bool)
            keep_t[list(drop)] = False
            bfaces, bfpoly = bfaces[keep_t], bfpoly[keep_t]
            bkind = [k_ for k_, kk in zip(bkind, keep_t) if kk]
            bst['thin_triangles'] = {'width_m': float(opts['thin_triangle_width_m']), 'candidates': int(len(thin)), 'deleted': int(len(drop)),
                                     'kept': int(len(thin) - len(drop))}
        t4 = time.time()
        code = np.zeros(len(polys), np.int64)
        nout = np.zeros((len(polys), 3))
        for i, (pp, k_, inf_) in enumerate(polys):
            nn = _area_normal(pp)
            nout[i] = nn / max(np.linalg.norm(nn), 1e-15)
            key = ('%s|%d' % inf_['plane']) if k_ == 'riser' else None
            rec = info['coplanar'].get(key) if key else None
            if rec is not None:
                if rec['id'] not in rec_of:
                    rec_of[rec['id']] = len(cop_records)
                    cop_records.append(dict(rec))
                code[i] = rec_of[rec['id']] + 1
        out['bv'].append(bv)
        out['bfaces'].append(bfaces + nb_total)
        out['bprism'].append(bprism)
        out['bbary'].append(bbary)
        out['bkind'] += list(bkind)
        out['code'].append(code[bfpoly])
        out['nout'].append(nout[bfpoly])
        out['breg'].append(np.full(len(bfaces), r, np.int64))
        out['bcell'].append(np.array([int(polys[i][2]['cell']) for i in bfpoly], np.int64))
        nb_total += len(bv)
        if fast_out is not None:
            out['bv'].append(fast_out['bv'])
            out['bfaces'].append(fast_out['bfaces'] + nb_total)
            out['bprism'].append(fast_out['bprism'])
            out['bbary'].append(fast_out['bbary'])
            out['bkind'] += ['floor'] * len(fast_out['bfaces'])
            out['code'].append(np.zeros(len(fast_out['bfaces']), np.int64))
            out['nout'].append(fast_out['nout'])
            out['breg'].append(np.full(len(fast_out['bfaces']), r, np.int64))
            out['bcell'].append(np.asarray(fast_out['keys'], np.int64))
            nb_total += len(fast_out['bv'])
        removed.append(inside)
        risers = [i for i, (_, k_, _) in enumerate(polys) if k_ == 'riser']
        n_over = sum(1 for i in risers if _area_normal(polys[i][0]) @ (-m) <= 0)
        rinfo = {**info, **bst, 'polygons': len(polys), 'risers': len(risers), 'overhanging_risers': n_over,
                 'faces_added': int(len(bfaces)) + (len(fast_out['bfaces']) if fast_out is not None else 0),
                 'faces_removed': int(len(inside)), 'cut_planes': len(cut),
                 'seconds': {'setup': t1 - t0, 'build': t2 - t1, 'weld': t3 - t2, 'triangulate': t4 - t3}}
        out['regions'].append(rinfo)
        if log:
            log('B region %d: %d prisms, %d polygons, %d planes, %.0f s' % (r, len(inside), len(polys), len(cut), time.time() - t0))
        del polys

    def cat(a, w):
        return np.concatenate(a) if a else np.zeros((0, w))
    return {'bv': cat(out['bv'], 3), 'bfaces': np.concatenate(out['bfaces']).astype(np.int64) if out['bfaces'] else np.zeros((0, 3), np.int64),
            'bprism': np.concatenate(out['bprism']).astype(np.int64) if out['bprism'] else np.zeros(0, np.int64),
            'bbary': cat(out['bbary'], 3), 'bkind': out['bkind'],
            'code': np.concatenate(out['code']).astype(np.int64) if out['code'] else np.zeros(0, np.int64),
            'nout': cat(out['nout'], 3), 'removed': np.concatenate(removed) if removed else np.zeros(0, np.int64),
            'records': cop_records, 'regions': out['regions'],
            'b_region': np.concatenate(out['breg']) if out['breg'] else np.zeros(0, np.int64),
            'b_cell': np.concatenate(out['bcell']) if out['bcell'] else np.zeros(0, np.int64)}


# ---------------------------------------------------------------- per-vertex donor attributes (3 colour classes)
class Layout:
    """Footprint lookup of the brick layout: for points in layout coordinates, the footprint of each colour whose
    square contains them (same-colour squares never overlap)."""

    def __init__(self, o, a1, a2, m, feet, W, blend, atlas_bounds):
        self.o, self.a1, self.a2, self.m = (np.asarray(v, float) for v in (o, a1, a2, m))
        self.W, self.blend, self.P = float(W), float(blend), float(W) - float(blend)
        self.by_rc = {(f['row'], f['col']): f for f in feet}
        self.ab = atlas_bounds

    def xy(self, P):
        P = np.atleast_2d(P)
        return np.c_[(P - self.o) @ self.a1, (P - self.o) @ self.a2]

    def classes(self, XY):
        """per point and colour: footprint id (-1 none), local coordinates (x - xc, y - yc) and window weight."""
        from .scandetail import window_weight
        n = len(XY)
        fid = np.full((n, 3), -1, np.int64)
        loc = np.zeros((n, 3, 2))
        w = np.zeros((n, 3))
        keys = np.array(sorted(self.by_rc), np.int64).reshape(-1, 2)
        kid = {tuple(k): i for i, k in enumerate(keys.tolist())}
        fids = np.array([self.by_rc[tuple(k)]['id'] for k in keys.tolist()], np.int64)
        cols = np.array([self.by_rc[tuple(k)]['colour'] for k in keys.tolist()], np.int64)
        r0 = np.round(XY[:, 1] / self.P).astype(np.int64)
        for dr in (-1, 0, 1):
            r = r0 + dr
            sh = np.where(r % 2 == 1, .5 * self.P, 0.)
            k0 = np.round((XY[:, 0] - sh) / self.P).astype(np.int64)
            for dk in (-1, 0, 1):
                k = k0 + dk
                xc, yc = k * self.P + sh, r * self.P
                lx, ly = XY[:, 0] - xc, XY[:, 1] - yc
                inn = np.flatnonzero((np.abs(lx) < self.W / 2) & (np.abs(ly) < self.W / 2))
                if not len(inn):
                    continue
                idx = np.array([kid.get((int(a), int(b)), -1) for a, b in zip(r[inn], k[inn])], np.int64)
                ok = idx >= 0
                inn, idx = inn[ok], idx[ok]
                c = cols[idx]
                ww = window_weight(lx[inn], ly[inn], self.W, self.blend)
                better = (fid[inn, c] < 0) | (ww > w[inn, c])
                inn, idx, c, ww = inn[better], idx[better], c[better], ww[better]
                fid[inn, c] = fids[idx]
                loc[inn, c, 0], loc[inn, c, 1] = lx[inn], ly[inn]
                w[inn, c] = ww
        return fid, loc, w

    def atlas_uv(self, fid, loc, feet):
        """atlas texture coordinates (U along d1, V along -d2) of window centre + local offset."""
        x0, x1, y0, y1 = self.ab
        win = np.array([f['window'] for f in feet], float)
        c = np.where(fid[..., None] >= 0, win[np.maximum(fid, 0)], 0.)
        X = c + loc
        return np.stack([(X[..., 0] - x0) / (x1 - x0), (y1 - X[..., 1]) / (y1 - y0)], -1)


def xfer_attributes_wall(V, F, riser_f, nout_f, rock_f, taper, lay, feet):
    """Wall version of scandetail.xfer_attributes: per vertex the atlas UV of each colour class (floors: projection
    on the layout plane; risers: their own plane basis about the footprint centre, as the B sample), the class
    window weights, the total weight w = taper x sum of class weights; faces: rock faces with any vertex at w > 0."""
    XY = lay.xy(V)
    fid, loc, wc = lay.classes(XY)
    rs = np.flatnonzero(riser_f)
    riser_v = np.zeros(len(V), bool)
    if len(rs):
        cen = np.array([f['centre'] for f in feet], float)
        for k in range(3):
            v_ = F[rs, k]
            n_ = nout_f[rs]
            u = np.cross(np.array([0., 0., 1.])[None], n_)
            hz = np.linalg.norm(u, axis=1) < 1e-6
            u[hz] = np.cross(n_[hz], np.array([0., 1., 0.])[None])
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            v = np.cross(n_, u)
            for c in range(3):
                ok = fid[v_, c] >= 0
                if not ok.any():
                    continue
                vv = v_[ok]
                d = V[vv] - cen[fid[vv, c]]
                loc[vv, c, 0] = (d * u[ok]).sum(1)
                loc[vv, c, 1] = (d * v[ok]).sum(1)
            riser_v[v_] = True
    uv3 = lay.atlas_uv(fid, loc, feet)
    wsum = wc.sum(1)
    w = np.clip(taper * np.minimum(wsum, 1.), 0, 1)
    faces = rock_f & (w[F] > 0).any(1)
    rel = np.where(wsum[:, None] > 0, wc / np.maximum(wsum, 1e-30)[:, None], 0.)
    core = (taper >= 1) & (wsum > 0)
    return {'uv3': uv3, 'class_w': rel, 'w': w, 'faces': faces, 'fid': fid, 'riser_vertices': int(riser_v.sum()),
            'pou_core_max_dev': float(np.abs(wsum[core] - 1).max()) if core.any() else None,
            'core_vertices_without_footprint': int(((taper >= 1) & (wsum <= 0)).sum())}


def fill_codes_wall(V, F, xfer_faces, riser_f, outward, lay, feet, atlas_path, slope_deg=60.):
    """Wall version of xfermat.fill_codes: per footprint face the source class (envelope / synthetic wall / riser)
    from the donor window of the dominant colour class at the face centroid."""
    from .scanmesh import DonorWindow
    xi = np.flatnonzero(xfer_faces)
    T = V[F[xi]]
    cen = T.mean(1)
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = .5 * np.linalg.norm(fn, axis=1)
    slope = np.degrees(np.arccos(np.clip(np.abs((fn / np.maximum(2 * area, 1e-300)[:, None]) @ np.asarray(outward, float)), 0, 1)))
    fid, loc, wc = lay.classes(lay.xy(cen))
    dom = np.argmax(wc, 1)
    ar = np.arange(len(xi))
    f_dom, l_dom = fid[ar, dom], loc[ar, dom]
    he = np.full(len(xi), -np.inf)
    hw = np.full(len(xi), -np.inf)
    atlas = _atlas(str(atlas_path))
    for f in np.unique(f_dom[f_dom >= 0]):
        sel = np.flatnonzero(f_dom == f)
        dw = DonorWindow(atlas, feet[f]['window'], lay.W)
        ins = (np.abs(l_dom[sel]) < lay.W / 2).all(1)
        if ins.any():
            _, _, he[sel[ins]], hw[sel[ins]] = dw.height(l_dom[sel[ins]])
    cls = np.zeros(len(xi), np.int8)
    cls[np.isfinite(he) & (he >= hw)] = 1
    cls[hw > he] = 2
    cls[riser_f[xi]] = 3
    fy = np.zeros(len(xi), np.float32)
    fy[cls == 3] = 2.
    fy[(cls == 2) | ((cls == 1) & (slope > slope_deg))] = 1.
    out = np.zeros(len(F), np.float32)
    out[xi] = fy
    names = {0: 'none', 1: 'envelope', 2: 'synthetic_wall', 3: 'riser'}
    stats = {'slope_deg': slope_deg, 'area_m2_by_source': {names[c]: float(area[cls == c].sum()) for c in range(4)},
             'area_m2_by_fill_code': {str(int(c)): float(area[fy == c].sum()) for c in (0, 1, 2)}}
    return out, stats
