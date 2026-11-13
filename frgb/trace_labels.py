"""Synthetic trace labels: fracture source surface ∩ uncut host, as 1 px image centrelines.

* Geometry: the case's own path_XX_stations.npz (what the cutter used) are interpolated between stations and every
sample moves along its inward vector (in the fracture surface, normal to the trace) onto the UNCUT host of build.json
'source'; the smallest |t| <= 0.12 m (cutter outside margin) wins. Station points sit on the generator's coarse
surface, which differs from a refined host by centimetres, so they are not used unprojected. * Visibility: subpixel
camera ray through P' on the CUT native mesh; visible when the first hit is not more than eps in front of P' (eps =
max(0.1 mm, half the pixel footprint)), or the hit is a nearby cavity face of the same source fracture. No RGB enters
the geometry; RGB is only read for U. No model, scoring or rendering here.

A case without a source binding has no declared ground height: no sample is classed as below the ground."""
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from frgb.lighting_rules import luma
from frgb.metrics import graph_node_weights, skeletonize
from frgb.network_training_labels import (_projected_boxes, extract_mouth_edges, qualify_mouth_samples, ray_pair_hits,
                                          sample_edges, vertex_face_csr)
from frgb.raster import rays

CRITERION_ID = 'trace_labels_r1'
T_BOUND_M = .12
SCREEN_SPACING_PX = .25
REFINE_ROUNDS = 6
EPS_FLOOR_M = 1e-4
EPS_PIXEL_FRACTION = .5
OWN_CAVITY_SLACK_M = .01
TRANSITION_RADIUS_PX = 2
RUN_BREAK_PX = 2.
U_K_PX = 1
U_RING_PX = 2
U_CAVITY_CAP_PX = 64
U_WINDOW_PX = 8.
U_T_LEVELS = 2.
U_T_OLD_LEVELS = 10.
U_MIN_RING = 4
U_IGNORE_RADIUS_PX = 2
GROUND_BOUNDARY_PX = 3
CONTRADICTION_CAVITY_DISTANCE_M = .10
CONTRADICTION_STOP_FRACTION = .01
GRID_CELL_M = .1
REFINE_ROUNDS_BRANCHES = 10
HIT_MERGE_T_M = 1e-7
CROSSING_MERGE_Z_M = 1e-9
INTERVAL_TOL_M = 1e-7
PARALLEL_PX = 2.

# ignore reason bits
IGNORE_U, IGNORE_TRANSITION, IGNORE_UNRESOLVED, IGNORE_SCENE, IGNORE_REGISTRATION = 1, 2, 4, 8, 16
IGNORE_OWN_GROOVE, IGNORE_OWN_FAULT = 32, 64
IGNORE_LABEL_RENDER_MISMATCH = 128

read_json = lambda p: json.loads(Path(p).read_text(encoding='utf8'))


# ---------------------------------------------------------------- camera helpers

def project(points, cam):
    """(N,3) world -> (uv (N,2) in the pixel-centre convention of sample_edges, view depth z (N,))."""
    r = np.asarray(points, float) - cam['pos']
    z = r @ cam['fwd']
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.column_stack([cam['w'] / 2 + cam['fpx'] * (r @ cam['right']) / z - .5,
                              cam['h'] / 2 - cam['fpx'] * (r @ cam['up']) / z - .5])
    return uv, z


def in_frame(uv, cam, pad=0.):
    return ((uv[:, 0] >= -.5 - pad) & (uv[:, 0] < cam['w'] - .5 + pad)
            & (uv[:, 1] >= -.5 - pad) & (uv[:, 1] < cam['h'] - .5 + pad))


def first_hits(vertices, faces, cam, uv, *, batch_pairs=4_000_000, boxes=None, native_index_compat=False):
    """Nearest two-sided hit along the subpixel ray through each uv: same candidates, exact test and tie rule (smallest face
    id) as network_training_labels.NativeRayIndex.nearest, vectorised (no per-point Python loop).

    NativeRayIndex buckets a point by floor(uv/16), so a point in [-0.5, 0) of pixel 0 finds no bucket and no hit.
    native_index_compat=True keeps that behaviour; the default looks such points up in the first cell, so every point
    inside the frame gets its true nearest hit."""
    uv = np.asarray(uv, float)
    z = np.full(len(uv), np.inf); face = np.full(len(uv), -1, np.int64)
    for q, fid, hits in _ray_pairs(vertices, faces, cam, uv, batch_pairs=batch_pairs, boxes=boxes,
                                   native_index_compat=native_index_compat):
        # per query: minimal depth, ties -> smallest face id (pairs are sorted by (q, fid))
        best = np.lexsort((fid, hits, q))
        first = np.r_[True, q[best][1:] != q[best][:-1]]
        sel = best[first]
        z[q[sel]] = hits[sel]
        face[q[sel]] = np.where(np.isfinite(hits[sel]), fid[sel], -1)
    return z, face


def all_hits(vertices, faces, cam, uv, *, batch_pairs=4_000_000, boxes=None):
    """Every finite two-sided hit (same candidates and exact test as first_hits): query index, face id, view depth."""
    parts = [(q[np.isfinite(h)], fid[np.isfinite(h)], h[np.isfinite(h)])
             for q, fid, h in _ray_pairs(vertices, faces, cam, np.asarray(uv, float), batch_pairs=batch_pairs, boxes=boxes)]
    if not parts:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    return tuple(np.concatenate(x) for x in zip(*parts))


def _ray_pairs(vertices, faces, cam, uv, *, batch_pairs, boxes, native_index_compat=False):
    """Candidate (query, face) pairs by padded projected AABB (4 px cells) and their exact hit depths, in query chunks."""
    v, f = np.asarray(vertices, float), np.asarray(faces, np.int64)
    if not len(uv):
        return
    lo, hi, ids = boxes if boxes is not None else _projected_boxes(v, f, cam)[:3]
    edge = 0. if native_index_compat else .5  # frame is [-0.5, w-0.5) in pixel-centre coordinates
    keep = ((hi[:, 0] >= -edge) & (lo[:, 0] <= cam['w'] - 1 + edge)
            & (hi[:, 1] >= -edge) & (lo[:, 1] <= cam['h'] - 1 + edge))
    lo, hi, ids = lo[keep], hi[keep], ids[keep]
    cell = 4
    ncx = int(np.ceil(cam['w'] / cell)) + 1
    clip = np.clip(uv, 0, [cam['w'] - 1, cam['h'] - 1])
    qc = np.floor(clip / cell).astype(np.int64)
    qkey = qc[:, 1] * ncx + qc[:, 0]
    if native_index_compat:
        qkey = np.where(np.any(uv < 0, axis=1), -1, qkey)
    wanted = np.unique(qkey)
    top = [cam['w'] - 1, cam['h'] - 1]
    c0 = np.floor(np.clip(lo, 0, top) / cell).astype(np.int64)
    c1 = np.floor(np.clip(hi, 0, top) / cell).astype(np.int64)
    nx, ny = c1[:, 0] - c0[:, 0] + 1, c1[:, 1] - c0[:, 1] + 1
    rows_parts, key_parts = [], []
    order = np.arange(len(ids))
    for chunk in np.array_split(order, max(1, len(order) // 500_000)):
        n = nx[chunk] * ny[chunk]
        r = np.repeat(chunk, n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        key = (c0[r, 1] + k // nx[r]) * ncx + c0[r, 0] + k % nx[r]
        hit = np.isin(key, wanted)
        rows_parts.append(r[hit]); key_parts.append(key[hit])
    rows, keys = np.concatenate(rows_parts), np.concatenate(key_parts)
    srt = np.argsort(keys, kind='stable'); rows, keys = rows[srt], keys[srt]
    start = np.searchsorted(keys, qkey, 'left'); stop = np.searchsorted(keys, qkey, 'right')
    counts = stop - start
    queries = np.arange(len(uv))
    for qchunk in np.array_split(queries, max(1, int(counts.sum() // batch_pairs) + 1)):
        cnt = counts[qchunk]
        q = np.repeat(qchunk, cnt)
        k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        row = rows[start[q] + k]
        inside = np.all((uv[q] >= lo[row]) & (uv[q] <= hi[row]), axis=1)
        q, fid = q[inside], ids[row[inside]]
        pair = np.unique(np.column_stack([q, fid]), axis=0)
        if not len(pair):
            continue
        q, fid = pair[:, 0], pair[:, 1]
        direction = rays(cam, uv[q, 0], uv[q, 1])
        hits, _ = ray_pair_hits(v[f[fid]], cam['pos'], direction, cam['fwd'], cam['near'])
        yield q, fid, hits


# ---------------------------------------------------------------- projection onto the uncut host

class HostGrid:
    """Uniform 3-D grid of host triangles by AABB (cell GRID_CELL_M) for segment queries."""

    def __init__(self, vertices, faces, cell=GRID_CELL_M):
        self.v = np.asarray(vertices, float); self.f = np.asarray(faces, np.int64); self.cell = float(cell)
        tri = self.v[self.f]
        self.origin = tri.min((0, 1)) - cell
        lo = np.floor((tri.min(1) - self.origin) / cell).astype(np.int64)
        hi = np.floor((tri.max(1) - self.origin) / cell).astype(np.int64)
        self.dims = hi.max(0) + 2
        n = np.prod(hi - lo + 1, axis=1)
        parts_f, parts_k = [], []
        for chunk in np.array_split(np.arange(len(tri)), max(1, len(tri) // 400_000)):
            nn = n[chunk]
            r = np.repeat(chunk, nn)
            k = np.arange(nn.sum()) - np.repeat(np.cumsum(nn) - nn, nn)
            ext = hi[r] - lo[r] + 1
            ix = lo[r, 0] + k % ext[:, 0]
            iy = lo[r, 1] + (k // ext[:, 0]) % ext[:, 1]
            iz = lo[r, 2] + k // (ext[:, 0] * ext[:, 1])
            parts_f.append(r); parts_k.append(self._key(ix, iy, iz))
        faces_, keys = np.concatenate(parts_f), np.concatenate(parts_k)
        srt = np.argsort(keys, kind='stable')
        self.cell_faces, self.cell_keys = faces_[srt], keys[srt]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        nrm = np.cross(e1, e2)
        self.normals = nrm / np.linalg.norm(nrm, axis=1)[:, None]

    def _key(self, ix, iy, iz):
        return (iz * self.dims[1] + iy) * self.dims[0] + ix

    def candidates(self, lo, hi):
        a = np.floor((np.asarray(lo) - self.origin) / self.cell).astype(np.int64)
        b = np.floor((np.asarray(hi) - self.origin) / self.cell).astype(np.int64)
        a, b = np.maximum(a, 0), np.minimum(b, self.dims - 1)
        if np.any(b < a):
            return np.empty(0, np.int64)
        grid = np.stack(np.meshgrid(*[np.arange(a[i], b[i] + 1) for i in range(3)], indexing='ij'), -1).reshape(-1, 3)
        keys = self._key(grid[:, 0], grid[:, 1], grid[:, 2])
        s, e = np.searchsorted(self.cell_keys, keys, 'left'), np.searchsorted(self.cell_keys, keys, 'right')
        if not (e - s).sum():
            return np.empty(0, np.int64)
        return np.unique(np.concatenate([self.cell_faces[i:j] for i, j in zip(s, e) if j > i]))


def _segment_hits(origin, direction, tri):
    """Pairwise Moller-Trumbore, line X = origin + u*direction, u in [0,1]; returns u (nan = miss)."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(direction, e2); det = np.einsum('ij,ij->i', e1, h)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(direction, axis=1)
    ok = np.abs(det) > 1e-14 * scale
    inv = np.divide(1., det, out=np.zeros_like(det), where=ok)
    s = origin - tri[:, 0]
    a = np.einsum('ij,ij->i', s, h) * inv
    q = np.cross(s, e1)
    b = np.einsum('ij,ij->i', direction, q) * inv
    u = np.einsum('ij,ij->i', e2, q) * inv
    tol = 1e-10
    hit = ok & (a >= -tol) & (b >= -tol) & (a + b <= 1 + tol) & (u >= -tol) & (u <= 1 + tol)
    return np.where(hit, u, np.nan)


def _interp(st, u):
    n = len(st['points'])
    i = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
    s = (u - i)[:, None]
    p = (1 - s) * st['points'][i] + s * st['points'][i + 1]
    w = (1 - s) * st['inward'][i] + s * st['inward'][i + 1]
    w /= np.linalg.norm(w, axis=1)[:, None]
    width = (1 - s) * st['width'][i].max(1, keepdims=True) + s * st['width'][i + 1].max(1, keepdims=True)
    depth = (1 - s) * st['depth'][i, -1:] + s * st['depth'][i + 1, -1:]
    return i, p, w, width[:, 0], depth[:, 0]


class _ChordCache:
    def __init__(self, st, grid, t_bound):
        self.st, self.grid, self.t_bound, self.cache = st, grid, t_bound, {}

    def get(self, i):
        if i not in self.cache:
            st, tb = self.st, self.t_bound
            p0, p1 = st['points'][i], st['points'][i + 1]
            w0, w1 = st['inward'][i], st['inward'][i + 1]
            corners = np.array([p0 - tb * w0, p0 + tb * w0, p1 - tb * w1, p1 + tb * w1])
            pad = .05 * tb + 1e-6
            cand = self.grid.candidates(corners.min(0) - pad, corners.max(0) + pad)
            chord = p1 - p0
            nrm = np.cross(chord, w0)
            if np.linalg.norm(nrm) > 1e-12 and len(cand):
                nrm /= np.linalg.norm(nrm)
                dist = (self.grid.v[self.grid.f[cand]] - p0) @ nrm
                tol = 1e-9 + 1.05 * tb * np.linalg.norm(w1 - w0) + 1e-9 * np.linalg.norm(chord)
                cand = cand[(dist.min(1) <= tol) & (dist.max(1) >= -tol)]
            self.cache[i] = cand
        return self.cache[i]


def _project_params(st, grid, chords, u, t_bound):
    i, p, w, width, depth = _interp(st, u)
    out_p = np.full_like(p, np.nan); out_t = np.full(len(u), np.nan)
    out_face = np.full(len(u), -1, np.int64); nhit = np.zeros(len(u), np.int64); distinct = np.zeros(len(u), np.int64)
    for c in np.unique(i):
        sel = np.flatnonzero(i == c)
        cand = chords.get(int(c))
        if not len(cand):
            continue
        tri = grid.v[grid.f[cand]]
        for part in np.array_split(sel, max(1, len(sel) * len(cand) // 2_000_000 + 1)):
            q = np.repeat(part, len(cand)); fc = np.tile(np.arange(len(cand)), len(part))
            origin = p[q] - t_bound * w[q]; direction = 2 * t_bound * w[q]
            uu = _segment_hits(origin, direction, tri[fc])
            hit = np.isfinite(uu)
            q, fc, tt = q[hit], fc[hit], (uu[hit] - .5) * 2 * t_bound
            if not len(q):
                continue
            best = np.lexsort((cand[fc], np.abs(tt), q))
            first = np.r_[True, q[best][1:] != q[best][:-1]]
            b = best[first]
            chosen = np.full(len(u), np.nan); chosen[q[b]] = tt[b]
            np.add.at(nhit, q, 1)
            np.add.at(distinct, q, np.abs(tt - chosen[q]) > 1e-9)
            out_t[q[b]] = tt[b]; out_face[q[b]] = cand[fc[b]]
            out_p[q[b]] = p[q[b]] + tt[b, None] * w[q[b]]
    return dict(u=u, station_point=p, inward=w, width=width, depth=depth, point=out_p, t=out_t,
                host_face=out_face, host_hits=nhit, other_intersections=distinct)


def _initial_params(st, cam, t_bound, spacing):
    """Station parameters u of the chords whose +-t_bound quad touches the frame, <= spacing px apart along the stations."""
    pts, inward = st['points'], st['inward']
    n = len(pts)
    corners = np.concatenate([pts, pts - t_bound * inward, pts + t_bound * inward])
    uv, z = project(corners, cam)
    uv = uv.reshape(3, n, 2); z = z.reshape(3, n)
    front = (z > cam['near']).all(0)
    lo = np.minimum(uv[:, :-1].min(0), uv[:, 1:].min(0)); hi = np.maximum(uv[:, :-1].max(0), uv[:, 1:].max(0))
    relevant = (front[:-1] & front[1:] & (hi[:, 0] >= -2) & (lo[:, 0] <= cam['w'] + 1)
                & (hi[:, 1] >= -2) & (lo[:, 1] <= cam['h'] + 1))
    params = []
    for c in np.flatnonzero(relevant):
        length = float(np.linalg.norm(uv[0, c + 1] - uv[0, c]))
        k = max(1, int(np.ceil(length / spacing)))
        params.append(c + np.arange(k) / k)
        if c == n - 2 or not relevant[c + 1]:
            params.append(np.array([c + 1.]))
    return np.unique(np.concatenate(params)) if params else None


def project_path(st, grid, cam, *, t_bound=T_BOUND_M, spacing=SCREEN_SPACING_PX, rounds=REFINE_ROUNDS):
    """Dense samples of one path projected onto the uncut host, ordered by the station parameter u."""
    u = _initial_params(st, cam, t_bound, spacing)
    if u is None:
        return None
    chords = _ChordCache(st, grid, t_bound)
    res = _project_params(st, grid, chords, u, t_bound)
    for _ in range(rounds):
        suv, _z = project(res['point'], cam)
        step = np.linalg.norm(np.diff(suv, axis=0), axis=1)
        same = np.floor(u[1:] - 1e-12) <= np.floor(u[:-1]) + 1  # consecutive along the path
        need = np.isfinite(step) & (step > spacing) & same & ((u[1:] - u[:-1]) > 1e-9)
        if not need.any():
            break
        mid = .5 * (u[:-1][need] + u[1:][need])
        new = _project_params(st, grid, chords, mid, t_bound)
        u = np.r_[u, mid]
        order = np.argsort(u, kind='stable')
        u = u[order]
        res = {k: np.concatenate([res[k], new[k]])[order] for k in res}
    res['refine_rounds_left_gaps'] = int(_gap_count(res, cam, spacing))
    return res


def _gap_count(res, cam, spacing):
    suv, _ = project(res['point'], cam)
    step = np.linalg.norm(np.diff(suv, axis=0), axis=1)
    return int(np.sum(np.isfinite(step) & (step > spacing)))



def _fine_pairs(fine, origin, direction, batch=1000, pad=1e-6):
    """(sample, face) pairs from a fine HostGrid (speed-up s1): every cell met by the padded AABBs of one-cell-long pieces of
    the segment origin + [0, 1] direction. Conservative (a face intersecting the segment overlaps one of those cells)."""
    cell = fine.cell
    out_q, out_f = [], []
    for part in np.array_split(np.arange(len(origin)), max(1, len(origin) // batch + 1)):
        if not len(part):
            continue
        o, d = origin[part], direction[part]
        K = max(1, int(np.ceil(np.linalg.norm(d, axis=1).max() / cell)))
        pts = o[:, None] + (np.arange(K + 1) / K)[None, :, None] * d[:, None]
        lo = np.floor((np.minimum(pts[:, :-1], pts[:, 1:]) - pad - fine.origin) / cell).astype(np.int64)
        hi = np.floor((np.maximum(pts[:, :-1], pts[:, 1:]) + pad - fine.origin) / cell).astype(np.int64)
        lo = np.maximum(lo, 0).reshape(-1, 3); hi = np.minimum(hi, fine.dims - 1).reshape(-1, 3)
        owner = np.repeat(np.arange(len(part)), K)
        ext = np.maximum(hi - lo + 1, 0)
        n = ext.prod(1)
        r = np.repeat(np.arange(len(lo)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        e = ext[r]
        key = fine._key(lo[r, 0] + k % e[:, 0], lo[r, 1] + (k // e[:, 0]) % e[:, 1], lo[r, 2] + k // (e[:, 0] * e[:, 1]))
        sk = np.unique(owner[r] * (int(fine.dims.prod()) + 1) + key)
        s_own, s_key = sk // (int(fine.dims.prod()) + 1), sk % (int(fine.dims.prod()) + 1)
        a0 = np.searchsorted(fine.cell_keys, s_key, 'left'); a1 = np.searchsorted(fine.cell_keys, s_key, 'right')
        cnt = a1 - a0
        q = np.repeat(s_own, cnt)
        kk = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        f = fine.cell_faces[np.repeat(a0, cnt) + kk]
        pair = np.unique(q * (len(fine.f) + 1) + f)
        out_q.append(part[pair // (len(fine.f) + 1)]); out_f.append(pair % (len(fine.f) + 1))
    return (np.concatenate(out_q), np.concatenate(out_f)) if out_q else (np.empty(0, np.int64), np.empty(0, np.int64))


def _all_params(st, grid, chords, u, t_bound, host_sign, fine=None):
    """Every host intersection of the lines P(u) + t w(u), |t| <= t_bound, as flat hits sorted by (sample, t). o = +1 where
    moving along +w leaves the host, -1 where it enters; same-o hits within HIT_MERGE_T_M merge (smallest face id kept).
    fine (speed-up s1): a fine HostGrid on the same mesh; candidate pairs then come from the cells each line crosses."""
    i, p, w, width, depth = _interp(st, u)
    qs, ts, fs = [], [], []
    if fine is not None:
        q, fc = _fine_pairs(fine, p - t_bound * w, 2 * t_bound * w)
        for part in np.array_split(np.arange(len(q)), max(1, len(q) // 2_000_000 + 1)):
            qq, ff = q[part], fc[part]
            uu = _segment_hits(p[qq] - t_bound * w[qq], 2 * t_bound * w[qq], grid.v[grid.f[ff]])
            hit = np.isfinite(uu)
            qs.append(qq[hit]); ts.append((uu[hit] - .5) * 2 * t_bound); fs.append(ff[hit])
    for c in (np.unique(i) if fine is None else []):
        sel = np.flatnonzero(i == c)
        cand = chords.get(int(c))
        if not len(cand):
            continue
        tri = grid.v[grid.f[cand]]
        for q, fc in _chord_pairs(st, int(c), u[sel] - c, sel, tri, t_bound):
            uu = _segment_hits(p[q] - t_bound * w[q], 2 * t_bound * w[q], tri[fc])
            hit = np.isfinite(uu)
            qs.append(q[hit]); ts.append((uu[hit] - .5) * 2 * t_bound); fs.append(cand[fc[hit]])
    q = np.concatenate(qs) if qs else np.empty(0, np.int64)
    t = np.concatenate(ts) if ts else np.empty(0)
    f = np.concatenate(fs) if fs else np.empty(0, np.int64)
    o = np.sign(host_sign * np.einsum('ij,ij->i', grid.normals[f], w[q])).astype(np.int64)
    order = np.lexsort((f, t, q)); q, t, f, o = q[order], t[order], f[order], o[order]
    new = np.r_[True, (q[1:] != q[:-1]) | (o[1:] != o[:-1]) | (t[1:] - t[:-1] > HIT_MERGE_T_M)][:len(q)]
    cluster = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    fmin = np.full(len(first), np.iinfo(np.int64).max); np.minimum.at(fmin, cluster, f)
    return dict(u=u, station_point=p, inward=w, width=width, depth=depth, hq=q[first], t=t[first], face=fmin, o=o[first],
                merged=int(len(q) - len(first)))


def _chord_pairs(st, c, s, sel, tri, t_bound, chunk=2_000_000):
    """(sample, triangle) pairs to test on one station chord.

    The lines of chord c are P(s) + t w(s), P(s) = p0 + s. Conservative: the pairs dropped cannot hit; the segment test
    itself is unchanged. Degenerate chords keep all pairs."""
    p0, p1 = st['points'][c], st['points'][c + 1]
    w0, w1 = st['inward'][c], st['inward'][c + 1]
    chord = p1 - p0
    a = chord - (chord @ w0) * w0
    la = float(np.linalg.norm(a))
    if la <= 1e-12:
        for part in np.array_split(sel, max(1, len(sel) * len(tri) // chunk + 1)):
            yield np.repeat(part, len(tri)), np.tile(np.arange(len(tri)), len(part))
        return
    a /= la
    L = float(chord @ a)
    xi = np.einsum('kij,j->ki', tri - p0, a) / L
    delta = (t_bound * float(np.linalg.norm(w1 - w0)) + 1e-9) / L * 1.05 + 1e-9
    lo, hi = xi.min(1) - delta, xi.max(1) + delta
    order = np.argsort(s, kind='stable')
    ss = s[order]
    a0 = np.searchsorted(ss, lo, 'left'); a1 = np.searchsorted(ss, hi, 'right')
    n = np.maximum(a1 - a0, 0)
    tris = np.flatnonzero(n)
    if not len(tris):
        return
    for part in np.array_split(tris, max(1, int(n[tris].sum()) // chunk + 1)):
        cnt = n[part]
        fc = np.repeat(part, cnt)
        k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        yield sel[order[a0[fc] + k]], fc


def unroll_host(host_vertices, host_faces, mid):
    V = np.asarray(host_vertices, float); F = np.asarray(host_faces, np.int64)
    ids = np.asarray(mid['vertex_ids'], np.int64)
    covered = np.zeros(len(V), bool); covered[ids] = True
    cov_f = covered[F]
    if 'sub_U' in mid:
        grid = HostGrid(np.asarray(mid['sub_U'], float), np.asarray(mid['sub_faces'], np.int64))
        original, normal = np.asarray(mid['sub_X'], float), np.asarray(mid['sub_n'], float)
        sub_f, faces, sigma = grid.f, np.asarray(mid['sub_parent'], np.int64), np.asarray(mid['sub_sigma'], float)
    else:
        faces = np.flatnonzero(cov_f.all(1))
        used = np.unique(F[faces])
        local = np.full(len(V), -1, np.int64); local[used] = np.arange(len(used))
        row = np.full(len(V), -1, np.int64); row[ids] = np.arange(len(ids))
        r = row[used]
        U = np.column_stack([mid['sigma'][r], mid['tau'][r], mid['s'][r] - mid['m'][r]])
        sub_f = local[F[faces]]
        grid = HostGrid(U, sub_f)
        original = V[used]
        normal = np.asarray(mid['n'], float)[r]
        sigma = np.asarray(mid['sigma'], float)[r]
    t = original[sub_f]
    n_orig = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flipped = np.sign(grid.normals[:, 2]) != np.sign(np.einsum('ij,ij->i', n_orig, normal[sub_f].sum(1)))
    return dict(grid=grid, face_ids=faces, original=original, normal=normal, sigma=sigma,
                flipped=flipped, partial_faces=int((cov_f.any(1) & ~cov_f.all(1)).sum()), sub_faces=int(len(faces)))


def chart_surface(chart, q):
    """frgb.network_panels.SurfaceChart.surface from the exported arrays (Q, u, dh, sigma): the source-surface point of q."""
    q = np.atleast_2d(q)
    i = np.clip(np.searchsorted(chart['sigma'], q[:, 0], side='right') - 1, 0, len(chart['u']) - 1)
    e = chart['Q'][i + 1] - chart['Q'][i]
    alpha = (q[:, 0] - chart['sigma'][i]) / (chart['sigma'][i + 1] - chart['sigma'][i])
    base = chart['Q'][i] + alpha[:, None] * e
    return base + (q[:, 1] - base @ chart['dh'])[:, None] * chart['dh']


def _barycentric(X, tri):
    e1, e2, r = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], X - tri[:, 0]
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    d20, d21 = (r * e1).sum(1), (r * e2).sum(1)
    den = d00 * d11 - d01 * d01
    b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den
    return np.column_stack([1 - b1 - b2, b1, b2])


def _unrolled_params(st, ur, u, t_bound, host_sign):
    """_all_params in the unrolled chart: line q(u) + t dir(u), |t| <= t_bound, in the plane z = 0 of U = (sigma, tau, s -
    m), cut with the unrolled sub-mesh; hits carry hit_q (chart point), bary and hit_point (uncut host)."""
    i, p, w, width, depth = _interp(st, u)
    s_ = (u - i)[:, None]
    q2 = (1 - s_) * ur['q'][i] + s_ * ur['q'][i + 1]
    d2 = (1 - s_) * ur['dir'][i] + s_ * ur['dir'][i + 1]
    d2 /= np.linalg.norm(d2, axis=1)[:, None]
    O = np.column_stack([q2 - t_bound * d2, np.zeros(len(u))])
    D = np.column_stack([2 * t_bound * d2, np.zeros(len(u))])
    grid = ur['grid']
    qq, ff = _fine_pairs(ur['fine'], O, D)
    qs, ts, fs = [], [], []
    for part in np.array_split(np.arange(len(qq)), max(1, len(qq) // 2_000_000 + 1)):
        a, b = qq[part], ff[part]
        uu = _segment_hits(O[a], D[a], grid.v[grid.f[b]])
        hit = np.isfinite(uu)
        qs.append(a[hit]); ts.append((uu[hit] - .5) * 2 * t_bound); fs.append(b[hit])
    q = np.concatenate(qs) if qs else np.empty(0, np.int64)
    t = np.concatenate(ts) if ts else np.empty(0)
    f = np.concatenate(fs) if fs else np.empty(0, np.int64)
    o = np.sign(host_sign * np.einsum('ij,ij->i', grid.normals[f][:, :2], d2[q])).astype(np.int64)
    order = np.lexsort((f, t, q)); q, t, f, o = q[order], t[order], f[order], o[order]
    new = np.r_[True, (q[1:] != q[:-1]) | (o[1:] != o[:-1]) | (t[1:] - t[:-1] > HIT_MERGE_T_M)][:len(q)]
    cluster = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    fmin = np.full(len(first), np.iinfo(np.int64).max); np.minimum.at(fmin, cluster, f)
    hq, ht = q[first], t[first]
    hit_q = q2[hq] + ht[:, None] * d2[hq]
    bary = _barycentric(np.column_stack([hit_q, np.zeros(len(hq))]), grid.v[grid.f[fmin]])
    hit_point = (bary[:, :, None] * ur['original'][grid.f[fmin]]).sum(1)
    return dict(u=u, station_point=p, inward=w, width=width, depth=depth, hq=hq, t=ht, face=fmin, o=o[first],
                merged=int(len(q) - len(first)), hit_q=hit_q, bary=bary, hit_point=hit_point)


def _merge_params(a, b):
    n = len(a['u'])
    order = np.argsort(np.r_[a['u'], b['u']], kind='stable')
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    out = {k: np.concatenate([a[k], b[k]])[order] for k in ('u', 'station_point', 'inward', 'width', 'depth')}
    hq = rank[np.r_[a['hq'], b['hq'] + n]]
    t, f, o = np.r_[a['t'], b['t']], np.r_[a['face'], b['face']], np.r_[a['o'], b['o']]
    s = np.lexsort((t, hq))
    out.update(hq=hq[s], t=t[s], face=f[s], o=o[s], merged=a['merged'] + b['merged'])
    for k in ('hit_point', 'bary', 'hit_q'):
        if k in a:
            out[k] = np.concatenate([a[k], b[k]])[s]
    return out


def match_hits(ta, oa, tb, ob):
    """Order- and orientation-preserving matching of two sorted hit lists: most pairs, then smallest sum |dt|, then the
    first in dynamic-programming order (match before skip). Returns matched indices into a and b."""
    m, n = len(ta), len(tb)
    if m == n and np.array_equal(oa, ob):
        return np.arange(m), np.arange(n)
    dp = [[None] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = (0, 0.)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == j == 0:
                continue
            best = arg = None
            if i and j and oa[i - 1] == ob[j - 1]:
                c = dp[i - 1][j - 1]
                best, arg = (c[0] - 1, c[1] + abs(ta[i - 1] - tb[j - 1])), 'm'
            if i and (best is None or dp[i - 1][j] < best):
                best, arg = dp[i - 1][j], 'a'
            if j and (best is None or dp[i][j - 1] < best):
                best, arg = dp[i][j - 1], 'b'
            dp[i][j], back[i][j] = best, arg
    ia, ib = [], []
    i, j = m, n
    while i or j:
        if back[i][j] == 'm':
            ia.append(i - 1); ib.append(j - 1); i -= 1; j -= 1
        elif back[i][j] == 'a':
            i -= 1
        else:
            j -= 1
    return np.asarray(ia[::-1], np.int64), np.asarray(ib[::-1], np.int64)


def _consecutive(u):
    return (np.floor(u[1:] - 1e-12) <= np.floor(u[:-1]) + 1) & ((u[1:] - u[:-1]) > 1e-9)


def _link(h, cam, spacing):
    n = len(h['u'])
    start = np.searchsorted(h['hq'], np.arange(n)); stop = np.searchsorted(h['hq'], np.arange(n), 'right')
    point = h['hit_point'] if 'hit_point' in h else h['station_point'][h['hq']] + h['t'][:, None] * h['inward'][h['hq']]
    uv, _ = project(point, cam)
    fwd = np.full(len(h['hq']), -1, np.int64); bwd = fwd.copy()
    cons = _consecutive(h['u'])
    unmatched = np.zeros(max(n - 1, 0), bool); far = np.zeros(max(n - 1, 0), bool)
    for i in np.flatnonzero(cons):
        a0, a1, b0, b1 = start[i], stop[i], start[i + 1], stop[i + 1]
        if a1 == a0 and b1 == b0:
            continue
        ia, ib = match_hits(h['t'][a0:a1], h['o'][a0:a1], h['t'][b0:b1], h['o'][b0:b1])
        fwd[a0 + ia] = b0 + ib; bwd[b0 + ib] = a0 + ia
        unmatched[i] = len(ia) < a1 - a0 or len(ib) < b1 - b0
        if len(ia):
            far[i] = bool(np.any(np.linalg.norm(uv[a0 + ia] - uv[b0 + ib], axis=1) > spacing))
    return dict(fwd=fwd, bwd=bwd, start=start, stop=stop, cons=cons, unmatched=unmatched, far=far, uv=uv, point=point)


def _extent(st, u0, u1, cam, t_bound):
    """Screen extent of the station line between u0 and u1: max over t in {-t_bound, 0, t_bound}."""
    _, p0, w0, _, _ = _interp(st, u0); _, p1, w1, _, _ = _interp(st, u1)
    ext = np.zeros(len(u0))
    for t in (-t_bound, 0., t_bound):
        a, _ = project(p0 + t * w0, cam); b, _ = project(p1 + t * w1, cam)
        ext = np.maximum(ext, np.linalg.norm(a - b, axis=1))
    return ext


def _needs_refinement(st, h, L, cam, t_bound, spacing):
    need = L['far'].copy()
    cand = np.flatnonzero(L['unmatched'] & ~need)
    if len(cand):
        need[cand] = _extent(st, h['u'][cand], h['u'][cand + 1], cam, t_bound) > spacing
    return need


def project_path_branches(st, grid, cam, host_sign, *, t_bound=T_BOUND_M, spacing=SCREEN_SPACING_PX,
                          rounds=REFINE_ROUNDS_BRANCHES, fine=None, unrolled=None):
    u = _initial_params(st, cam, t_bound, spacing)
    if u is None:
        return None, None
    chords = _ChordCache(st, grid, t_bound)
    if unrolled is not None:
        params = lambda uu: _unrolled_params(st, unrolled, uu, t_bound, host_sign)
    else:
        params = lambda uu: _all_params(st, grid, chords, uu, t_bound, host_sign, fine=fine)
    h = params(u)
    used = 0
    for _ in range(rounds):
        L = _link(h, cam, spacing)
        need = _needs_refinement(st, h, L, cam, t_bound, spacing)
        if not need.any():
            break
        used += 1
        mid = .5 * (h['u'][:-1][need] + h['u'][1:][need])
        h = _merge_params(h, params(mid))
    L = _link(h, cam, spacing)
    left = int(_needs_refinement(st, h, L, cam, t_bound, spacing).sum())
    branches, folds = _chains(h, L)
    counts = L['stop'] - L['start']
    info = dict(samples=int(len(h['u'])), hits=int(len(h['hq'])), merged_duplicate_hits=int(h['merged']),
                zero_hit_samples=int((counts == 0).sum()), max_hits_per_sample=int(counts.max()) if len(counts) else 0,
                samples_with_several_hits=int((counts > 1).sum()), branches=len(branches),
                closed_loops=int(sum(b['closed'] for b in branches)), fold_joins=folds, rounds_used=used,
                intervals_still_needing_refinement=left,
                unresolved_station_points=h['station_point'][counts == 0])
    return branches, info



def flatten_host(host_vertices, host_faces, mid):
    """Returns the flattened HostGrid with the maps back to the uncut host."""
    V = np.asarray(host_vertices, float); F = np.asarray(host_faces, np.int64)
    ids = np.asarray(mid['vertex_ids'], np.int64)
    covered = np.zeros(len(V), bool); covered[ids] = True
    cov_f = covered[F]
    faces = np.flatnonzero(cov_f.all(1))
    used = np.unique(F[faces])
    local = np.full(len(V), -1, np.int64); local[used] = np.arange(len(used))
    row = np.full(len(V), -1, np.int64); row[ids] = np.arange(len(ids))
    original = V[used]
    flat = original - np.asarray(mid['D'], float)[row[used]]
    sub_f = local[F[faces]]
    grid = HostGrid(flat, sub_f)
    t = original[sub_f]
    n_orig = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flipped = np.einsum('ij,ij->i', grid.normals, n_orig) < 0
    return dict(grid=grid, face_ids=faces, original=original, normal=np.asarray(mid['n'], float)[row[used]],
                sigma=np.asarray(mid['sigma'], float)[row[used]], flipped=flipped,
                partial_faces=int((cov_f.any(1) & ~cov_f.all(1)).sum()), sub_faces=int(len(faces)))


def unflatten(res, sub):
    """Map the hits of project_path_branches on the flattened sub-mesh back to the uncut host (barycentric, same face)."""
    g = sub['grid']
    f = res['host_face']
    tri = g.v[g.f[f]]
    X = res['point']
    e1, e2, r = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], X - tri[:, 0]
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    d20, d21 = (r * e1).sum(1), (r * e2).sum(1)
    den = d00 * d11 - d01 * d01
    b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den
    b = np.column_stack([1 - b1 - b2, b1, b2])
    idx = g.f[f]
    res['flat_point'] = X
    res['point'] = (b[:, :, None] * sub['original'][idx]).sum(1)
    n = (b[:, :, None] * sub['normal'][idx]).sum(1)
    res['slot_normal'] = n / np.linalg.norm(n, axis=1)[:, None]
    res['sigma'] = (b * sub['sigma'][idx]).sum(1)
    res['on_flipped_face'] = sub['flipped'][f]
    res['host_face'] = sub['face_ids'][f]
    res['barycentric_min'] = b.min(1)
    return res


def slot_sections(grid, sign, points, normals, *, reach=.2, sub=.02):
    """Interior interval of a closed mesh on P + s n (|s| <= reach) containing or nearest s = 0: (lo, hi) per point."""
    lo = np.full(len(points), np.nan); hi = np.full(len(points), np.nan)
    for k, (p, n) in enumerate(zip(points, normals)):
        a, b = p - reach * n, p + reach * n
        knots = a + np.linspace(0, 1, int(round(2 * reach / sub)) + 1)[:, None] * (b - a)
        parts = [grid.candidates(np.minimum(x, y) - 1e-6, np.maximum(x, y) + 1e-6) for x, y in zip(knots[:-1], knots[1:])]
        cand = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
        if not len(cand):
            continue
        uu = _segment_hits(np.tile(a, (len(cand), 1)), np.tile(b - a, (len(cand), 1)), grid.v[grid.f[cand]])
        ok = np.isfinite(uu)
        if not ok.any():
            continue
        s = (uu[ok] - .5) * 2 * reach
        o = np.sign(sign * grid.normals[cand[ok]] @ n)
        order = np.lexsort((o, s)); s, o = s[order], o[order]
        keep = np.r_[True, (o[1:] != o[:-1]) | (s[1:] - s[:-1] > CROSSING_MERGE_Z_M)]
        s, o = s[keep], o[keep]
        pairs = [(s[j], s[j + 1]) for j in range(len(s) - 1) if o[j] < 0 and o[j + 1] > 0]
        if pairs:
            dist = [0. if x <= 0 <= y else min(abs(x), abs(y)) for x, y in pairs]
            lo[k], hi[k] = pairs[int(np.argmin(dist))]
    return lo, hi


def _chains(h, L):
    """Link + fold graph (degree <= 2) -> chains: open ones from their (u, t)-first end, loops closed by their first node."""
    nh = len(h['hq'])
    fwd, bwd = L['fwd'], L['bwd']
    fold_f = np.full(nh, -1, np.int64); fold_b = np.full(nh, -1, np.int64)
    has_next = np.r_[L['cons'], False]; has_prev = np.r_[False, L['cons']]
    folds = 0
    lonely = np.flatnonzero(((fwd < 0) | (bwd < 0))[:-1] & ((fwd < 0) | (bwd < 0))[1:]
                            & (h['hq'][1:] == h['hq'][:-1]) & (h['o'][1:] != h['o'][:-1])) if nh > 1 else []
    for k in lonely:
        i = h['hq'][k]
        if has_next[i] and fwd[k] < 0 and fwd[k + 1] < 0 and fold_f[k] < 0 and fold_f[k + 1] < 0:
            fold_f[k], fold_f[k + 1] = k + 1, k; folds += 1
        if has_prev[i] and bwd[k] < 0 and bwd[k + 1] < 0 and fold_b[k] < 0 and fold_b[k + 1] < 0:
            fold_b[k], fold_b[k + 1] = k + 1, k; folds += 1
    nbrs = np.stack([fwd, bwd, fold_f, fold_b], 1)
    deg = (nbrs >= 0).sum(1)
    seen = np.zeros(nh, bool)

    def walk(s):
        seq, prev, cur = [s], -1, s
        seen[s] = True
        while True:
            nxt = [int(x) for x in nbrs[cur] if x >= 0 and x != prev and not seen[x]]
            if not nxt:
                return seq
            prev, cur = cur, nxt[0]
            seen[cur] = True; seq.append(cur)
    chains = [(walk(s), False) for s in range(nh) if not seen[s] and deg[s] <= 1]
    for s in range(nh):
        if not seen[s]:
            seq = walk(s)
            chains.append((seq + [seq[0]], True))
    counts = L['stop'] - L['start']
    out = []
    for b, (seq, closed) in enumerate(chains):
        seq = np.asarray(seq, np.int64)
        s = h['hq'][seq]
        out.append(dict(u=h['u'][s], station_point=h['station_point'][s], inward=h['inward'][s], width=h['width'][s],
                        depth=h['depth'][s], point=L['point'][seq], t=h['t'][seq], host_face=h['face'][seq],
                        host_hits=counts[s], other_intersections=counts[s] - 1, orientation=h['o'][seq],
                        hit_index=seq, sample_index=s, closed=bool(closed), branch=b))
        for k in ('bary', 'hit_q'):
            if k in h:
                out[-1][k] = h[k][seq]
    return out, folds


# ---------------------------------------------------------------- case loading

GROUND_Z_RE = re.compile(r'"ground_z_m":\s*([-0-9.eE+]+)')


def case_ground_z(case_dir):
    """A shape case sits in <net>/cases/<name>; the net's source_binding.json names the scene geom directory, whose
    package/model_spec.json declares ground_z_m. None when the case has no source binding (no ground rule is applied then); a
    model spec without exactly one ground height stops the entry (ValueError)."""
    binding = Path(case_dir).parent.parent / 'source_binding.json'
    if not binding.is_file():
        return None
    spec = Path(read_json(binding)['source']) / 'package' / 'model_spec.json'
    values = sorted({float(v) for v in GROUND_Z_RE.findall(spec.read_text(encoding='utf8'))})
    if len(values) != 1:
        raise ValueError('ground rule r1: %s declares ground_z_m %s, not one value' % (spec, values))
    return values[0]


def load_case(case_dir, known_roles=None):
    """Cut native mesh, roles, per-face path index, build.json paths, stations and the uncut host of build.json 'source'.
    known_roles: a face role outside this set stops the entry."""
    case = Path(case_dir)
    build = read_json(case / 'build.json')
    with np.load(case / 'geometry/native_surface.npz', allow_pickle=False) as a:
        native = dict(vertices=a['vertices'], faces=a['faces'], roles=a['face_roles'].astype(str))
        source_ids = a['source_ids']
    if known_roles is not None:
        unknown = sorted(set(np.unique(native['roles']).tolist()) - set(known_roles))
        if unknown:
            raise ValueError('face roles outside the known set: %s' % unknown)
    path_of_face = np.full(len(native['faces']), -1, np.int32)
    cavity = np.flatnonzero(native['roles'] != 'original_host')
    for j, sid in zip(cavity, source_ids[cavity]):
        m = re.match(r'path(\d+):', str(sid))
        if m is None:
            raise ValueError('cavity face without path identity: ' + str(sid))
        path_of_face[j] = int(m[1])
    del source_ids
    native['path_of_face'] = path_of_face
    paths = {}
    for row in build['paths']:
        with np.load(case / ('path_%02d_stations.npz' % row['index']), allow_pickle=False) as a:
            st = {k: a[k] for k in a.files}
        paths[row['index']] = dict(row=row, stations=st)
    source = Path(build['source'])
    with np.load(source / 'source_surface.npz', allow_pickle=False) as a:
        host = dict(vertices=a['vertices'], faces=a['faces'])
    return dict(case=case, build=build, native=native, paths=paths, source=source, host=host, ground_z_m=case_ground_z(case))


def fracture_surface_residual(source, samples_by_path, paths):
    """Distance (m) of projected points to their source fracture surface via the extrusion map (growth_footprint_xy)."""
    from frgb.network_profiles import load_frozen_network
    network = load_frozen_network(source)
    out = {}
    for index, res in samples_by_path.items():
        ok = np.isfinite(res['point']).all(1)
        if not ok.any():
            out[index] = None; continue
        trace = network.trace(paths[res.get('path', index)]['row']['source_trace_id'])
        xy = network.growth_footprint_xy(res['point'][ok])
        out[index] = float(trace.locate_2d(xy)['distance_m'].max())
    return out


# ---------------------------------------------------------------- visibility

def classify_visibility(case, samples_by_path, cam, *, eps_factor=1., literal=False, boxes=None):
    """Adds z, uv, hit depth/face, eps and the visible flag to every in-frame projected sample."""
    nat = case['native']
    trace_of_path = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    all_uv, owner = [], []
    for index, res in samples_by_path.items():
        uv, z = project(res['point'], cam)
        ok = np.isfinite(res['point']).all(1) & (z > cam['near'])
        res['uv'], res['z'] = uv, z
        res['in_frame'] = ok & in_frame(np.where(np.isfinite(uv), uv, -1e9), cam)
        all_uv.append(uv[res['in_frame']]); owner.append((index, np.flatnonzero(res['in_frame'])))
    if 'hit_z' not in next(iter(samples_by_path.values()), {}):
        uv = np.concatenate(all_uv) if all_uv else np.empty((0, 2))
        hz, hf = first_hits(nat['vertices'], nat['faces'], cam, uv, boxes=boxes)
        k = 0
        for index, sel in owner:
            res = samples_by_path[index]
            res['hit_z'] = np.full(len(res['u']), np.nan); res['hit_face'] = np.full(len(res['u']), -1, np.int64)
            res['hit_z'][sel] = hz[k:k + len(sel)]; res['hit_face'][sel] = hf[k:k + len(sel)]; k += len(sel)
    for index, res in samples_by_path.items():
        eps = eps_factor * np.maximum(EPS_FLOOR_M, EPS_PIXEL_FRACTION * res['z'] / cam['fpx'])
        hz, hf = res['hit_z'], res['hit_face']
        depth_ok = res['in_frame'] & (hz >= res['z'] - eps)
        own_any = res['in_frame'] & (hf >= 0)
        own_any[own_any] = nat['path_of_face'][hf[own_any]] >= 0
        same = np.zeros(len(hz), bool)
        same[own_any] = np.array([trace_of_path.get(int(p)) == trace_of_path[res.get('path', index)]
                                  for p in nat['path_of_face'][hf[own_any]]], bool)
        hit_point = cam['pos'] + np.where(np.isfinite(hz), hz, 0.)[:, None] * _ray_directions(res, cam)
        near = np.linalg.norm(hit_point - res['point'], axis=1) <= (res['depth'] + res['width'] + T_BOUND_M
                                                                     + OWN_CAVITY_SLACK_M)
        own_local = same & near
        res['eps'] = eps
        res['visible_depth_rule'] = depth_ok
        res['visible_own_cavity_only'] = own_local & ~depth_ok
        res['literal_extra'] = (same & ~near & ~depth_ok)
        res['visible'] = depth_ok | own_local | (res['literal_extra'] if literal else False)
        res['occluded'] = res['in_frame'] & ~res['visible']
    gz = case.get('ground_z_m')
    if gz is not None:
        for res in samples_by_path.values():
            p = res['point']
            below = np.isfinite(p).all(1)
            below[below] = p[below, 2] < gz
            res['below_ground'] = res['in_frame'] & below
            for k in ('in_frame', 'visible', 'visible_depth_rule', 'visible_own_cavity_only', 'literal_extra'):
                res[k] = res[k] & ~below
            res['occluded'] = res['in_frame'] & ~res['visible']
    return samples_by_path


def _ray_directions(res, cam):
    """Direction (not unit, d.fwd = 1) of the ray through each sample; hit point = pos + hit_depth * direction."""
    uv = np.where(np.isfinite(res['uv']), res['uv'], 0.)
    return rays(cam, uv[:, 0], uv[:, 1])


def contradiction_check(case, samples_by_path, cam, *, host_boxes=None):
    """Stop-condition evidence: visible on the cut mesh but occluded on the uncut host by rock that no cut removed, or a
    back-facing uncut host face at P'. Orientation of 'front' comes from the cut mesh's own host pixels (majority)."""
    from scipy.spatial import cKDTree
    host = case['host']
    nat = case['native']
    cav_vertices = np.unique(nat['faces'][nat['roles'] != 'original_host'])
    tree = cKDTree(nat['vertices'][cav_vertices])
    sign = case.get('front_sign', 1.)
    uv_all, owner = [], []
    for index, res in samples_by_path.items():
        sel = np.flatnonzero(res['visible'])
        uv_all.append(res['uv'][sel]); owner.append((index, sel))
    uv = np.concatenate(uv_all) if uv_all else np.empty((0, 2))
    hz, hf = first_hits(host['vertices'], host['faces'], cam, uv, boxes=host_boxes)
    k = 0
    summary = dict(visible_length_px=0., contradiction_a_length_px=0., contradiction_b_length_px=0.,
                   uncut_occluded_cut_visible_removed_occluder_length_px=0.)
    for index, sel in owner:
        res = samples_by_path[index]
        a = np.zeros(len(res['u']), bool); b = np.zeros(len(res['u']), bool); removed = np.zeros(len(res['u']), bool)
        z_u = hz[k:k + len(sel)]; k += len(sel)
        in_front = z_u < res['z'][sel] - res['eps'][sel]
        if in_front.any():
            d = rays(cam, res['uv'][sel, 0], res['uv'][sel, 1])
            point = cam['pos'] + z_u[:, None] * d
            dist, _ = tree.query(np.where(np.isfinite(point), point, 0.))
            a[sel[in_front & (dist > CONTRADICTION_CAVITY_DISTANCE_M)]] = True
            removed[sel[in_front & (dist <= CONTRADICTION_CAVITY_DISTANCE_M)]] = True
        hfaces = res['host_face'][sel]
        n = sign * case['host_normals'][hfaces]
        b[sel] = np.einsum('ij,ij->i', n, cam['pos'] - res['point'][sel]) < 0
        res['contradiction_a'], res['contradiction_b'], res['occluder_removed_by_cut'] = a, b, removed
        w = _sample_weights(res)
        summary['visible_length_px'] += float(w[res['visible']].sum())
        summary['contradiction_a_length_px'] += float(w[a].sum())
        summary['contradiction_b_length_px'] += float(w[b].sum())
        summary['uncut_occluded_cut_visible_removed_occluder_length_px'] += float(w[removed].sum())
    total = summary['visible_length_px']
    summary['contradiction_fraction'] = ((summary['contradiction_a_length_px'] + summary['contradiction_b_length_px'])
                                         / total) if total else None
    summary['stop_triggered'] = bool(summary['contradiction_fraction'] is not None
                                     and summary['contradiction_fraction'] > CONTRADICTION_STOP_FRACTION)
    return summary


def mesh_sign(vertices, faces):
    v, f = np.asarray(vertices, float), np.asarray(faces, np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 1. if np.einsum('ij,ij->i', a, np.cross(b, c)).sum() > 0 else -1.


def interior_intervals(vertices, faces, sign, cam, uv, *, boxes=None):
    """Interior intervals of a closed mesh along the subpixel rays through uv (view depth). Crossings: all two-sided hits,
    o = sign(sign . n . d) (+1 exit, -1 entry), same-o crossings within CROSSING_MERGE_Z_M merged. Returns (owner, lo, hi)
    and a per-query anomaly flag (camera inside, or entries/exits not alternating)."""
    n = len(uv)
    q, fid, z = all_hits(vertices, faces, cam, uv, boxes=boxes)
    v, f = np.asarray(vertices, float), np.asarray(faces, np.int64)
    tri = v[f[fid]]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    d = rays(cam, uv[q, 0], uv[q, 1]) if len(q) else np.empty((0, 3))
    o = np.sign(sign * np.einsum('ij,ij->i', nrm, d)).astype(np.int64)
    order = np.lexsort((o, z, q)); q, z, o = q[order], z[order], o[order]
    new = np.r_[True, (q[1:] != q[:-1]) | (o[1:] != o[:-1]) | (z[1:] - z[:-1] > CROSSING_MERGE_Z_M)][:len(q)]
    q, z, o = q[new], z[new], o[new]
    start = np.searchsorted(q, np.arange(n)); stop = np.searchsorted(q, np.arange(n), 'right')
    pos = np.arange(len(q)) - start[q]
    anomaly = np.zeros(n, bool)
    np.logical_or.at(anomaly, q, o != np.where(pos % 2 == 0, -1, 1))
    anomaly |= (stop - start) % 2 == 1
    entry = np.flatnonzero((pos % 2 == 0) & ~anomaly[q])
    return q[entry], z[entry], z[entry + 1], anomaly


def _group(owner, lo, hi, n):
    order = np.lexsort((lo, owner))
    owner, lo, hi = owner[order], lo[order], hi[order]
    return owner, lo, hi, np.searchsorted(owner, np.arange(n)), np.searchsorted(owner, np.arange(n), 'right')


def _contained(a, b, lo, hi, tol=INTERVAL_TOL_M):
    """Is [a, b] inside the union of the intervals (lo, hi)? (lo sorted)"""
    if not len(lo):
        return False
    merged = [[lo[0], hi[0]]]
    for x, y in zip(lo[1:], hi[1:]):
        if x <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], y)
        else:
            merged.append([x, y])
    return any(x <= a + tol and y >= b - tol for x, y in merged)


def exemption_check(case, samples_by_path, cam, cutters, *, host_boxes=None):
    host = case['host']
    trace_of_path = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    keys, sel_of, uv_all = [], {}, []
    for key, res in samples_by_path.items():
        sel = np.flatnonzero(res['visible'])
        sel_of[key] = (sum(len(x) for x in uv_all), sel); uv_all.append(res['uv'][sel]); keys.append(key)
    uv = np.concatenate(uv_all) if uv_all else np.empty((0, 2))
    n = len(uv)
    z_p = np.concatenate([samples_by_path[k]['z'][sel_of[k][1]] for k in keys]) if n else np.empty(0)
    eps = np.concatenate([samples_by_path[k]['eps'][sel_of[k][1]] for k in keys]) if n else np.empty(0)
    hit = np.concatenate([samples_by_path[k]['hit_z'][sel_of[k][1]] for k in keys]) if n else np.empty(0)
    trace = np.concatenate([np.full(len(sel_of[k][1]), trace_of_path[samples_by_path[k].get('path', k)], object)
                            for k in keys]) if n else np.empty(0, object)
    z_end = np.minimum(z_p - eps, hit)
    hown, hlo, hhi, h_anom = interior_intervals(host['vertices'], host['faces'], case['front_sign'], cam, uv, boxes=host_boxes)
    hlo, hhi = np.maximum(hlo, cam['near']), np.minimum(hhi, z_end[hown])
    keep = hhi - hlo > CROSSING_MERGE_Z_M
    oown, olo, ohi, ostart, ostop = _group(hown[keep], hlo[keep], hhi[keep], n)
    has_occluder = ostop > ostart
    own_parts, all_parts = [], []
    c_anom = np.zeros(n, bool)
    for tid, c in cutters.items():
        own = c.get('trace', tid)
        co, clo, chi, ca = interior_intervals(c['vertices'], c['faces'], c['sign'], cam, uv, boxes=c.get('boxes'))
        mine = trace[co] == own
        own_parts.append((co[mine], clo[mine], chi[mine])); all_parts.append((co, clo, chi))
        c_anom |= ca & (trace == own)
    cat = lambda parts: tuple(np.concatenate([p[i] for p in parts]) if parts else np.empty(0) for i in range(3))
    wo, wlo, whi, wstart, wstop = _group(*[x.astype(np.int64) if i == 0 else x for i, x in enumerate(cat(own_parts))], n)
    ao, alo, ahi, astart, astop = _group(*[x.astype(np.int64) if i == 0 else x for i, x in enumerate(cat(all_parts))], n)
    exempt = np.ones(n, bool); other_cut = np.zeros(n, bool); no_cut = np.zeros(n, bool)
    for s in np.flatnonzero(has_occluder):
        occ = list(zip(olo[ostart[s]:ostop[s]], ohi[ostart[s]:ostop[s]]))
        mine = (wlo[wstart[s]:wstop[s]], whi[wstart[s]:wstop[s]])
        if all(_contained(a, b, *mine) for a, b in occ):
            continue
        exempt[s] = False
        if all(_contained(a, b, alo[astart[s]:astop[s]], ahi[astart[s]:astop[s]]) for a, b in occ):
            other_cut[s] = True
        else:
            no_cut[s] = True
    anomaly = h_anom | c_anom
    p_inside = np.zeros(n, bool)
    m = (wlo <= z_p[wo] + INTERVAL_TOL_M) & (whi >= z_p[wo] - INTERVAL_TOL_M)
    p_inside[wo[m]] = True
    contradiction = (~exempt) | anomaly
    occluder_len = np.zeros(n); np.add.at(occluder_len, oown, ohi - olo)
    summary = dict(visible_length_px=0., contradiction_length_px=0., exempt_with_occluder_length_px=0.,
                   contradiction_other_cut_length_px=0., contradiction_no_cut_length_px=0., parity_anomaly_length_px=0.,
                   backfacing_occluder_within_eps_length_px=0., p_outside_own_cut_length_px=0.)
    for key in keys:
        res = samples_by_path[key]
        k0, sel = sel_of[key]
        sl = slice(k0, k0 + len(sel))
        flags = dict(contradiction=contradiction[sl], exempt_with_occluder=exempt[sl] & has_occluder[sl] & ~anomaly[sl],
                     other_cut=other_cut[sl], no_cut=no_cut[sl], anomaly=anomaly[sl], p_outside_own_cut=~p_inside[sl])
        for name, val in flags.items():
            full = np.zeros(len(res['u']), bool); full[sel] = val; res['ex_' + name] = full
        res['ex_occluder_m'] = np.zeros(len(res['u'])); res['ex_occluder_m'][sel] = occluder_len[sl]
        back = np.zeros(len(res['u']), bool)
        nrm = case['front_sign'] * case['host_normals'][res['host_face'][sel]]
        back[sel] = (np.einsum('ij,ij->i', nrm, cam['pos'] - res['point'][sel]) < 0) & ~has_occluder[sl]
        res['ex_backfacing_within_eps'] = back
        w = _sample_weights(res)
        summary['visible_length_px'] += float(w[res['visible']].sum())
        summary['contradiction_length_px'] += float(w[res['ex_contradiction']].sum())
        summary['exempt_with_occluder_length_px'] += float(w[res['ex_exempt_with_occluder']].sum())
        summary['contradiction_other_cut_length_px'] += float(w[res['ex_other_cut']].sum())
        summary['contradiction_no_cut_length_px'] += float(w[res['ex_no_cut']].sum())
        summary['parity_anomaly_length_px'] += float(w[res['ex_anomaly']].sum())
        summary['backfacing_occluder_within_eps_length_px'] += float(w[back].sum())
        summary['p_outside_own_cut_length_px'] += float(w[res['ex_p_outside_own_cut']].sum())
    total = summary['visible_length_px']
    summary['contradiction_fraction'] = summary['contradiction_length_px'] / total if total else None
    summary['stop_triggered'] = bool(total and summary['contradiction_fraction'] > CONTRADICTION_STOP_FRACTION)
    return summary


def parallel_record(samples_by_path, *, use_u=True):
    """Lengths by class: no sibling, <= 2 px, > 2 px."""
    by_path = {}
    for key, res in samples_by_path.items():
        ok = res['visible'] & ~(res['sample_u'] if use_u and 'sample_u' in res else np.zeros(len(res['u']), bool))
        w = _sample_weights(res)
        sample = res.get('sample_index', np.arange(len(res['u'])))
        by_path.setdefault(res.get('path', key), []).append((sample[ok], res['uv'][ok], w[ok], res['t'][ok]))
    out = dict(total_px=0., none_px=0., near_px=0., far_px=0., far_abs_dt_m=[], per_path={})
    for path, parts in sorted(by_path.items()):
        s = np.concatenate([p[0] for p in parts]); uv = np.concatenate([p[1] for p in parts])
        w = np.concatenate([p[2] for p in parts]); t = np.concatenate([p[3] for p in parts])
        d = np.full(len(s), np.inf); dt = np.full(len(s), np.nan)
        order = np.argsort(s, kind='stable')
        bounds = np.flatnonzero(np.r_[True, s[order][1:] != s[order][:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                continue
            idx = order[a:b]
            dd = np.linalg.norm(uv[idx][:, None] - uv[idx][None], axis=2)
            np.fill_diagonal(dd, np.inf)
            j = dd.argmin(1)
            d[idx] = dd[np.arange(len(idx)), j]; dt[idx] = np.abs(t[idx] - t[idx][j])
        row = dict(total_px=float(w.sum()), none_px=float(w[~np.isfinite(d)].sum()),
                   near_px=float(w[np.isfinite(d) & (d <= PARALLEL_PX)].sum()),
                   far_px=float(w[np.isfinite(d) & (d > PARALLEL_PX)].sum()))
        out['per_path'][int(path)] = row
        for k in ('total_px', 'none_px', 'near_px', 'far_px'):
            out[k] += row[k]
        out['far_abs_dt_m'].append(dt[np.isfinite(d) & (d > PARALLEL_PX)])
    far_dt = np.concatenate(out['far_abs_dt_m']) if out['far_abs_dt_m'] else np.empty(0)
    out['far_abs_dt_m'] = [float(x) for x in np.quantile(far_dt, [.1, .5, .9])] if len(far_dt) else None
    for k in ('none', 'near', 'far'):
        out[k + '_fraction'] = out[k + '_px'] / out['total_px'] if out['total_px'] else None
    return out


def _sample_weights(res):
    """Screen length represented by each sample: half of each adjacent step within a run (step <= RUN_BREAK_PX)."""
    uv = res['uv']
    step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    ok = np.isfinite(step) & (step <= RUN_BREAK_PX) & res['in_frame'][1:] & res['in_frame'][:-1]
    step = np.where(ok, step, 0.)
    return np.r_[step, 0.] / 2 + np.r_[0., step] / 2


def _below_ground_px(res):
    """None without a declared ground."""
    m = res.get('below_ground')
    if m is None:
        return None
    step = np.linalg.norm(np.diff(res['uv'], axis=0), axis=1)
    ok = np.isfinite(step) & (step <= RUN_BREAK_PX) & m[1:] & m[:-1]
    step = np.where(ok, step, 0.)
    return float((np.r_[step, 0.] / 2 + np.r_[0., step] / 2)[m].sum())


def runs_of(mask, res):
    """Contiguous runs of mask along the sample order; a screen jump > RUN_BREAK_PX also breaks a run."""
    uv = res['uv']
    step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    cont = mask[1:] & mask[:-1] & np.isfinite(step) & (step <= RUN_BREAK_PX)
    run = np.full(len(mask), -1, np.int64)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return run
    new = np.r_[True, ~cont[idx[1:] - 1]] if len(idx) > 1 else np.array([True])
    run[idx] = np.cumsum(new) - 1
    return run


# ---------------------------------------------------------------- scene masks

def scene_nonscored(shape, ground=None, sky=None, *, boundary_px=GROUND_BOUNDARY_PX):
    """Ground and sky pixels, plus rock pixels within boundary_px (EDT) of ground."""
    out = np.zeros(shape, bool)
    if sky is not None:
        out |= np.asarray(sky, bool)
    if ground is not None:
        g = np.asarray(ground, bool)
        out |= g
        if g.any():
            out |= ndi.distance_transform_edt(~g) <= boundary_px
    return out


# ---------------------------------------------------------------- rasterisation, U and assembly

def _pixels(uv):
    return np.floor(uv + .5).astype(np.int64)


def rasterise_samples(res, mask, shape):
    img = np.zeros(shape, bool)
    p = _pixels(res['uv'][mask])
    ok = (p[:, 0] >= 0) & (p[:, 0] < shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < shape[0])
    img[p[ok, 1], p[ok, 0]] = True
    return img


def _disc(points_xy, shape, radius):
    img = np.zeros(shape, bool)
    p = _pixels(np.asarray(points_xy, float).reshape(-1, 2))
    ok = (p[:, 0] >= 0) & (p[:, 0] < shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < shape[0])
    img[p[ok, 1], p[ok, 0]] = True
    if not img.any():
        return img
    return ndi.distance_transform_edt(~img) <= radius


def path_arc(res):
    """A step that is not finite or is longer than RUN_BREAK_PX (where runs break) adds nothing. It does not read in_frame or
    visible, so a change of visibility does not move the grid."""
    step = np.linalg.norm(np.diff(res['uv'], axis=0), axis=1)
    step = np.where(np.isfinite(step) & (step <= RUN_BREAK_PX), step, 0.)
    return np.r_[0., np.cumsum(step)]


def windows_of(res):
    """Window id per visible sample: within each visible run (runs_of), the samples of one grid cell floor(path_arc /
    U_WINDOW_PX) form a window; an end piece of a run shorter than half a window joins the run's neighbouring cell
    (the first end the next cell, the last end the previous one). Ids are numbered run by run, cell by cell."""
    vis = res['visible']
    run = runs_of(vis, res)
    window = np.full(len(vis), -1, np.int64)
    if not (run >= 0).any():
        return window
    arc = path_arc(res)
    cell = np.floor(arc / U_WINDOW_PX).astype(np.int64)
    next_id = 0
    for r in range(run.max() + 1):
        idx = np.flatnonzero(run == r)
        s, c = arc[idx], cell[idx].copy()
        c0, c1 = int(c[0]), int(c[-1])
        if c1 > c0 and (c0 + 1) * U_WINDOW_PX - s[0] < U_WINDOW_PX / 2:
            c[c == c0] = c0 + 1
            c0 += 1
        if c1 > c0 and s[-1] - c1 * U_WINDOW_PX < U_WINDOW_PX / 2:
            c[c == c1] = c1 - 1
        ids, inv = np.unique(c, return_inverse=True)
        window[idx] = next_id + inv
        next_id += len(ids)
    return window


def visible_centre(res, shape):
    """Skeleton of the in-frame visible samples (the centreline u_windows works on)."""
    p = _pixels(res['uv'])
    inside = res['visible'] & (p[:, 0] >= 0) & (p[:, 0] < shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < shape[0])
    raw = np.zeros(shape, bool); raw[p[inside, 1], p[inside, 0]] = True
    return skeletonize(raw) if raw.any() else raw


def u_windows(res, rgb_luma255, cpu_face, roles, path_of_face, same_trace_paths, host_valid, shape, ring_exclude=None,
              export=False):
    """Per-window band/ring contrast (8-bit luma). Returns window table and the visible centreline with window ids.
    ring_exclude: pixels removed from the ring (within 1 px of another branch of the same fracture), counted. export
    (output only, U unchanged): also return the band and ring pixels used, as (linear index, local window id)."""
    vis = res['visible']
    window = windows_of(res)
    p = _pixels(res['uv'])
    inside = vis & (p[:, 0] >= 0) & (p[:, 0] < shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < shape[0])
    idx = np.flatnonzero(inside)
    if not len(idx):
        return dict(window=window, centre=np.zeros(shape, bool), table=None)
    lin = p[idx, 1] * shape[1] + p[idx, 0]
    first = np.unique(lin, return_index=True)[1]
    lin, win_of_pixel = lin[first], window[idx[first]]
    raw = np.zeros(shape, bool); raw.flat[lin] = True
    centre = skeletonize(raw)
    if not centre.any():
        # branch without visible pixels (the per-path union skeleton drops the component alike)
        return dict(window=window, centre=centre, table=None)
    win_img = np.full(shape, -1, np.int64); win_img.flat[lin] = win_of_pixel
    # crop to the centreline box + reach
    ys, xs = np.nonzero(centre)
    reach = U_CAVITY_CAP_PX + U_RING_PX + 2
    y0, y1 = max(0, ys.min() - reach), min(shape[0], ys.max() + reach + 1)
    x0, x1 = max(0, xs.min() - reach), min(shape[1], xs.max() + reach + 1)
    c = centre[y0:y1, x0:x1]
    dist, (iy, ix) = ndi.distance_transform_edt(~c, return_indices=True)
    nearest_win = win_img[y0:y1, x0:x1][iy, ix]
    face = cpu_face[y0:y1, x0:x1]
    cav = face >= 0
    cav[cav] = (roles[face[cav]] != 'original_host') & np.isin(path_of_face[face[cav]], list(same_trace_paths))
    hv = host_valid[y0:y1, x0:x1]
    band = ((dist <= U_K_PX) | (cav & (dist <= U_CAVITY_CAP_PX)))
    band_use = band & hv
    host = hv & (face >= 0)
    host[host] = roles[face[host]] == 'original_host'
    to_band = ndi.distance_transform_edt(~band)
    ring = host & ~band & (to_band <= U_RING_PX) & (dist <= U_CAVITY_CAP_PX + U_RING_PX)
    excluded = np.zeros_like(ring)
    if ring_exclude is not None:
        excluded = ring & ring_exclude[y0:y1, x0:x1]
        ring &= ~excluded
    L = rgb_luma255[y0:y1, x0:x1]
    nwin = int(window.max()) + 1 if (window >= 0).any() else 0
    def sums(mask):
        w = nearest_win[mask]; keep = w >= 0
        return (np.bincount(w[keep], weights=L[mask][keep], minlength=nwin), np.bincount(w[keep], minlength=nwin))
    bs, bn = sums(band_use); rs, rn = sums(ring)
    ring_excluded = sums(excluded)[1]
    with np.errstate(invalid='ignore', divide='ignore'):
        contrast = np.abs(bs / bn - rs / rn)
    present = np.bincount(window[window >= 0], minlength=nwin) > 0
    table = dict(band_pixels=bn, ring_pixels=rn, band_mean=bs / np.maximum(bn, 1), ring_mean=rs / np.maximum(rn, 1),
                 contrast=contrast, present=present, ring_excluded=ring_excluded,
                 u=present & ((rn < U_MIN_RING) | (bn == 0) | ~(contrast >= U_T_LEVELS)),
                 u_old=present & ((rn < U_MIN_RING) | (bn == 0) | ~(contrast >= U_T_OLD_LEVELS)))
    full_win = np.full(shape, -1, np.int64)
    full_win[centre] = win_img[centre]
    out = dict(window=window, centre=centre, centre_window=full_win, table=table)
    if export:
        for name, mask in (('band_pixels', band_use), ('ring_pixels', ring)):
            yy, xx = np.nonzero(mask)
            w_ = nearest_win[yy, xx]
            keep = w_ >= 0
            out[name] = ((yy[keep] + y0) * shape[1] + xx[keep] + x0, w_[keep])
    return out


def _other_branch_exclusions(samples_by_path, trace_of_path, shape):
    centres = {k: visible_centre(r, shape) for k, r in samples_by_path.items()}
    out = {}
    for k, r in samples_by_path.items():
        others = np.zeros(shape, bool)
        for j, c in centres.items():
            if j != k and trace_of_path[samples_by_path[j]['path']] == trace_of_path[r['path']]:
                others |= c
        out[k] = ndi.distance_transform_edt(~others) <= U_K_PX if others.any() else None
    return out


def _transitions_and_unresolved(res, cam, shape):
    both = res['in_frame'][1:] & res['in_frame'][:-1]
    flip = both & (res['visible'][1:] != res['visible'][:-1])
    tpts = np.concatenate([res['uv'][:-1][flip], res['uv'][1:][flip]])
    transition = _disc(tpts, shape, TRANSITION_RADIUS_PX) if len(tpts) else np.zeros(shape, bool)
    bad = ~np.isfinite(res['point']).all(1)
    return transition, _station_discs(res['station_point'][bad], cam, shape), bad


def _station_discs(points, cam, shape):
    out = np.zeros(shape, bool)
    if len(points):
        puv, pz = project(points, cam)
        ok = (pz > cam['near']) & in_frame(np.where(np.isfinite(puv), puv, -1e9), cam)
        if ok.any():
            out |= _disc(puv[ok], shape, U_IGNORE_RADIUS_PX)
    return out


def opening_classes(face, roles, path_of_face, rock, centre_of, band_px=2.):
    """Domain: rock pixels whose first hit is a cavity face (role not original_host); path = the face's path; centre_of:
    {base path: visible centreline before U / ignore}. Components: 8-connectivity per path."""
    shape = face.shape
    cls = np.zeros(shape, np.uint8)
    hit = face >= 0
    fidx = np.where(hit, face, 0)
    cav = rock & hit & (roles[fidx] != 'original_host')
    pid = np.where(cav, path_of_face[fidx], -1)
    allc = np.zeros(shape, bool)
    for m in centre_of.values():
        allc |= m
    band = ndi.distance_transform_edt(~allc) <= band_px if allc.any() else np.zeros(shape, bool)
    by_path = {}
    for p in np.unique(pid[pid >= 0]):
        own = pid == p
        cp = centre_of.get(int(p))
        near = ndi.distance_transform_edt(~cp) <= band_px if cp is not None and cp.any() else np.zeros(shape, bool)
        lab, n = ndi.label(own, structure=np.ones((3, 3), bool))
        has = np.zeros(n + 1, bool)
        has[np.unique(lab[own & near])] = True
        has[0] = False
        out_ = own & ~band
        cls[out_ & has[lab]] = 1
        cls[out_ & ~has[lab]] = 2
        by_path[str(int(p))] = {'1': int((out_ & has[lab]).sum()), '2': int((out_ & ~has[lab]).sum())}
    return cls, by_path


def build_trace_labels(case, samples_by_path, cam, rgb, *, cpu_raster, render_rock, aligned, ground=None, sky=None,
                       unresolved_points=None, export_windows=False, occlusion_regions=None, mismatch_region=None,
                       opening_base=None):
    """Assemble target / valid / ignore for one frame (samples already projected and classified).

    export_windows (output only; U, labels and windows unchanged otherwise): every windows row gets a global id 'gid' (its
    index in the list) and the result holds band_win / ring_win (int32 frame images: the gid of the band / ring pixels
    the U contrast used, -1 elsewhere; where a pixel serves several windows the first one is kept) and window_pixels
    (all (linear index, gid) memberships of band and ring, which reproduce every window's contrast exactly)."""
    shape = render_rock.shape
    nat = case['native']
    roles, path_of_face = nat['roles'], nat['path_of_face']
    hit = cpu_raster['face'] >= 0
    rock = hit & render_rock
    scene = scene_nonscored(shape, ground, sky)
    registration = render_rock & ~aligned
    host_valid = rock & aligned & ~scene
    L255 = luma(rgb) * 255.
    trace_of_path = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    branch_mode = any('path' in r for r in samples_by_path.values())
    path_of = {k: r.get('path', k) for k, r in samples_by_path.items()}
    exclusions = _other_branch_exclusions(samples_by_path, trace_of_path, shape) if branch_mode else {}
    target = np.zeros(shape, bool); u_line = np.zeros(shape, bool); u_line_old = np.zeros(shape, bool)
    occluded_line = np.zeros(shape, bool)
    transition = np.zeros(shape, bool); unresolved = np.zeros(shape, bool)
    per_path, windows, positive_of_path = [], [], {}
    member = dict(band_pixels=[], ring_pixels=[])
    centre_of = {}      # mismatch_opening_r1 O2: the visible centreline per base path, before U and ignore
    for index, res in samples_by_path.items():
        path = path_of[index]
        same = [p for p, t in trace_of_path.items() if t == trace_of_path[path]]
        uw = u_windows(res, L255, cpu_raster['face'], roles, path_of_face, same, host_valid, shape,
                       ring_exclude=exclusions.get(index), export=export_windows)
        res['window'] = uw['window']
        tab = uw['table']
        centre = uw['centre']
        if opening_base is not None:
            b_ = opening_base.get(path, path)
            centre_of[b_] = (centre_of[b_] | centre) if b_ in centre_of else centre.copy()
        if tab is not None:
            cw = uw['centre_window']
            is_u = np.zeros(shape, bool); is_u[centre] = tab['u'][cw[centre]]
            is_u_old = np.zeros(shape, bool); is_u_old[centre] = tab['u_old'][cw[centre]]
            u_line |= is_u; u_line_old |= is_u_old
            if not branch_mode:
                target |= centre & ~is_u
            positive_of_path[index] = centre & ~is_u
            res['sample_u'] = np.zeros(len(res['u']), bool)
            ok = res['window'] >= 0
            res['sample_u'][ok] = tab['u'][res['window'][ok]]
            gid_of = np.full(len(tab['present']), -1, np.int64)
            for w in np.flatnonzero(tab['present']):
                gid_of[w] = len(windows)
                row = dict(path=int(path), window=int(w), band_pixels=int(tab['band_pixels'][w]),
                           ring_pixels=int(tab['ring_pixels'][w]), contrast=float(tab['contrast'][w])
                           if np.isfinite(tab['contrast'][w]) else None, u=bool(tab['u'][w]))
                if branch_mode:
                    row.update(branch=int(res['branch']), ring_excluded=int(tab['ring_excluded'][w]))
                if export_windows:
                    row['gid'] = int(gid_of[w])
                windows.append(row)
            if export_windows:
                for name in ('band_pixels', 'ring_pixels'):
                    lin, lw = uw[name]
                    member[name].append((lin, gid_of[lw]))
        else:
            res['sample_u'] = np.zeros(len(res['u']), bool)
            positive_of_path[index] = np.zeros(shape, bool)
        occluded_line |= rasterise_samples(res, res['occluded'], shape)
        # transitions between visible and occluded consecutive in-frame samples
        tr, unr, bad = _transitions_and_unresolved(res, cam, shape)
        transition |= tr; unresolved |= unr
        w = _sample_weights(res)
        row = dict(path=int(path), source_trace_id=trace_of_path[path],
                   projected_in_frame_px=float(w[res['in_frame']].sum()),
                   visible_px=float(w[res['visible']].sum()), occluded_px=float(w[res['occluded']].sum()),
                   visible_u_px=float(w[res['visible'] & res['sample_u']].sum()),
                   unresolved_projection_samples=int(bad.sum()),
                   own_cavity_clause_only_px=float(w[res['visible_own_cavity_only']].sum()),
                   literal_extra_px=float(w[res['literal_extra']].sum()), below_ground_px=_below_ground_px(res), _key=index)
        if branch_mode:
            row['branch'] = int(res['branch'])
        per_path.append(row)
    if branch_mode:
        for path in sorted(set(path_of.values())):
            union = np.zeros(shape, bool)
            for k in samples_by_path:
                if path_of[k] == path:
                    union |= positive_of_path[k]
            target |= skeletonize(union) if union.any() else union
    for points in (unresolved_points or {}).values():
        unresolved |= _station_discs(points, cam, shape)
    u_ignore = ndi.distance_transform_edt(~u_line) <= U_IGNORE_RADIUS_PX if u_line.any() else np.zeros(shape, bool)
    reason = np.zeros(shape, np.uint8)
    reason[u_ignore] |= IGNORE_U
    reason[transition] |= IGNORE_TRANSITION
    reason[unresolved] |= IGNORE_UNRESOLVED
    reason[scene] |= IGNORE_SCENE
    reason[registration] |= IGNORE_REGISTRATION
    for bit, region in occlusion_regions or ():
        reason[region & ~target & ~u_ignore] |= bit
    if mismatch_region is not None:      # mismatch_opening_r1 M3: every pixel of the band, positives and U pixels included
        reason[mismatch_region] |= IGNORE_LABEL_RENDER_MISMATCH
    ignore = reason > 0
    valid = rock & ~ignore
    final = target & valid
    weights = graph_node_weights(final)
    for row in per_path:
        own = final & (positive_of_path[row.pop('_key')])
        row['positive_graph_length_px'] = float(graph_node_weights(own).sum())
    out = dict(target=final, valid=valid, ignore_reason=reason, u_line=u_line, u_line_old_threshold=u_line_old,
               occluded_line=occluded_line, rock=rock, per_path=per_path, windows=windows,
               positive_graph_length_px=float(weights.sum()),
               all_visible_u=bool(u_line.any() and not final.any()))
    if mismatch_region is not None:
        out['mismatch_removed_positive_px'] = int((target & rock & mismatch_region
                                                   & ((reason & np.uint8(255 - IGNORE_LABEL_RENDER_MISMATCH)) == 0)).sum())
    if opening_base is not None:
        out['opening_class'], out['opening_by_path'] = opening_classes(cpu_raster['face'], roles, path_of_face, rock, centre_of)
    if export_windows:
        pairs = {}
        for name, key in (('band_pixels', 'band_win'), ('ring_pixels', 'ring_win')):
            lin = np.concatenate([a for a, _ in member[name]]) if member[name] else np.empty(0, np.int64)
            gid = np.concatenate([b for _, b in member[name]]) if member[name] else np.empty(0, np.int64)
            img = np.full(shape[0] * shape[1], -1, np.int32)
            first = np.unique(lin, return_index=True)[1]
            img[lin[first]] = gid[first]
            out[key] = img.reshape(shape)
            pairs[name] = np.column_stack([lin, gid]).astype(np.int64)
            out[key + '_multi_pixels'] = int(len(lin) - len(first))
        out['window_pixels'] = pairs
    return out


def window_contrast_from_pixels(window_pixels, luma255, n_windows):
    """|mean band - mean ring| per global window id from export_windows' window_pixels (8-bit luma image); nan if empty."""
    L = np.asarray(luma255, float).ravel()
    sums = {}
    for name in ('band_pixels', 'ring_pixels'):
        lin, gid = window_pixels[name][:, 0], window_pixels[name][:, 1]
        sums[name] = (np.bincount(gid, weights=L[lin], minlength=n_windows), np.bincount(gid, minlength=n_windows))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.abs(sums['band_pixels'][0] / sums['band_pixels'][1] - sums['ring_pixels'][0] / sums['ring_pixels'][1])


def build_trace_geometry(case, samples_by_path, cam, shape, *, unresolved_points=None):
    """Geometry-only frame: target = per path the skeleton of the union of its branches' visible centrelines; transition and
    zero-hit discs are ignore."""
    trace_of_path = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    target = np.zeros(shape, bool); occluded_line = np.zeros(shape, bool)
    transition = np.zeros(shape, bool); unresolved = np.zeros(shape, bool)
    unions, per_path = {}, []
    for index, res in samples_by_path.items():
        res['sample_u'] = np.zeros(len(res['u']), bool)
        path = res.get('path', index)
        unions[path] = unions.get(path, np.zeros(shape, bool)) | visible_centre(res, shape)
        occluded_line |= rasterise_samples(res, res['occluded'], shape)
        tr, unr, bad = _transitions_and_unresolved(res, cam, shape)
        transition |= tr; unresolved |= unr
        w = _sample_weights(res)
        per_path.append(dict(path=int(path), branch=int(res.get('branch', 0)), source_trace_id=trace_of_path[path],
                             projected_in_frame_px=float(w[res['in_frame']].sum()), visible_px=float(w[res['visible']].sum()),
                             occluded_px=float(w[res['occluded']].sum()),
                             own_cavity_clause_only_px=float(w[res['visible_own_cavity_only']].sum()),
                             below_ground_px=_below_ground_px(res)))
    for u in unions.values():
        target |= skeletonize(u) if u.any() else u
    for points in (unresolved_points or {}).values():
        unresolved |= _station_discs(points, cam, shape)
    reason = np.zeros(shape, np.uint8)
    reason[transition] |= IGNORE_TRANSITION
    reason[unresolved] |= IGNORE_UNRESOLVED
    valid = reason == 0
    final = target & valid
    return dict(target=final, valid=valid, ignore_reason=reason, occluded_line=occluded_line, per_path=per_path,
                positive_graph_length_px=float(graph_node_weights(final).sum()))



def mouth_edge_record(case, cam, render_depth_m, render_rock, aligned, *, boxes=None, trace_ids=None):
    """Nearest hits use first_hits (same candidates and ties as NativeRayIndex)."""
    nat = case['native']
    with np.load(case['case'] / 'geometry/native_surface.npz', allow_pickle=False) as a:
        source_ids = a['source_ids']
    if trace_ids is None:
        trace_ids = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    edges = extract_mouth_edges(nat['vertices'], nat['faces'], nat['roles'], source_ids, trace_ids)
    del source_ids
    samples = sample_edges(nat['vertices'], edges, cam)
    first, first_face = first_hits(nat['vertices'], nat['faces'], cam, samples['uv'], boxes=boxes,
                                   native_index_compat=True)
    q = qualify_mouth_samples(samples, edges, first, first_face, render_depth_m, render_rock, aligned,
                              vertex_faces=vertex_face_csr(nat['faces'], len(nat['vertices'])))
    shape = render_rock.shape
    pending = np.zeros(shape, bool)
    x, y = q['pixel_xy'].T
    uncertain = q['pending'] | q['uncertain']
    pending[y[uncertain], x[uncertain]] = True
    pending = ndi.binary_dilation(pending, iterations=2)
    target = np.zeros(shape, bool); path_image = np.full(shape, -1, np.int16); side_image = np.zeros(shape, np.int8)
    owner = samples['owner']; candidate = q['candidate']
    for path in np.unique(edges['path_index'][owner[candidate]]):
        for side, code in [('left_wall', 1), ('right_wall', 2)]:
            take = candidate & (edges['path_index'][owner] == path) & (edges['cavity_role'][owner] == side)
            raw = np.zeros(shape, bool); raw[y[take], x[take]] = True
            line = skeletonize(raw)
            collision = line & target & ((path_image != path) | (side_image != code))
            pending |= ndi.binary_dilation(collision, iterations=1)
            target |= line; path_image[line] = path; side_image[line] = code
    return dict(target=target, pending=pending, samples=samples, qualification=q, edges=edges,
                nearest_depth_m=first, nearest_face=first_face)
