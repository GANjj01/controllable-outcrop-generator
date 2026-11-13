"""Geometry acceptance of the whole-wall scene, on a staged directory: display.npz (b_region / b_cell per face, risers,
xfer_ref / relief_dvec), request.json (display_stats.blockface) and the per-region dumps wall_debug/region_<r>.npz
(the surface the prisms of the region were cut from, taper, planes, cells).

- G-a: weld moves below each point's tolerance, global max < 0.5 mm (blockface.blocks[r].weld); - G-b: every floor
face's centroid lies in its own cell (cell_of of the region == the face's recorded cell); - G-c / G-c2: outer boundary
of every B face (failures of the coarse offsets that the B sample classed as weld deformation within the 0.5 mm
tolerance -- pulled probes at their own deviation + 0.1 mm pass -- are counted apart): centroid and 4 points pulled
1/4 to it, offsets min(1e-4, .05 sqrt(a)), 1e-5, 1e-6 m tried in order, pass iff every inner point is rock and every
outer point air (rock locator: projection along m on the region's prism tops, rock iff below the floor depth(cell) x
taper); the failures re-tested at the offsets {1e-7, 1e-8, 1e-9} <= 0.1 x width with the reversed normal failing at
each (G-c2); < 2 usable tiers or a failing tier = not closed. The probes use the EXPORTED winding of each face
(display.npz faces, the order written to the GLB); the checker never flips a normal; - winding: the exported floor
winding of every B region faces out of the rock (-m) -- the same convention as G-c; - X1: fold guard final 0; for the
scan-detail route also constraint failures 0, vertices without donor surface 0; - X3: displacement of the detail (scan
transfer: final - xfer_ref; relief route: relief_dvec) along m only; - X4: every riser face within 1e-12 m of one of
its region's cut planes (max reported); the faces labelled with a joint record lie in that joint piece's v2 plane; -
X5: orientation: every interior edge of the B faces / of the whole mesh is used in opposite directions by its two
faces, no edge has more than two faces; - X6: closure at the retained raw and welded polygon boundaries, then the
final mesh: samples every 0.5 mm must lie within 1 um of an edge of another polygon/face, including adjacent non-B
surfaces; - partition of unity in the detail core (blockface.detail.pou_core_max_dev; scan-detail route only).

Every item has a status PASS / FAIL / NOT_APPLICABLE. Applicability follows the route the request declares (blockface
mode, detail.method, B on/off), never the absence of data: an applicable item without its evidence is FAIL. ``pass``
is True only for PASS (None for NOT_APPLICABLE); the overall status is FAIL if any item fails."""
import json
import time
from pathlib import Path

import numpy as np

NONE, AIR, ROCK = -1, 0, 1
M_HASH = 2147483647


class RegionDump:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.V, self.F, self.taper, self.m = z['V'], z['F'], z['taper'], z['m']
        self.cut_n, self.cut_d = z['cut_n'], z['cut_d']
        self.cut_fam, self.cut_lev = z['cut_fam'], z['cut_lev']
        self.levels, self.C, self.nb = z['levels'], float(z['C']), z['nb']
        self.jsets = list(z['jsets'])
        self.joint_n = z['joint_n']
        self.joint_offs = list(z['joint_offs']) if len(self.jsets) else []
        self.seed, self.depth, self.region = int(z['seed']), float(z['depth']), int(z['region'])
        self.spall = 'spall' in z.files
        if self.spall:
            self.cross_n, self.cross_offs = z['cross_n'], list(z['cross_offs'])
            self.table_keys, self.table_R = z['table_keys'], z['table_R']

    def cell_of(self, P):
        """vectorised RegionCells.cell_of / SpallCells.cell_of (same integer arithmetic)."""
        i = np.searchsorted(self.levels, P @ self.nb + self.C).astype(np.int64)
        k = i % M_HASH
        for j in range(len(self.jsets)):
            k = (k * 48271 + np.searchsorted(np.asarray(self.joint_offs[j], float), P @ self.joint_n[j]).astype(np.int64) + 7919 * (j + 1)) % M_HASH
        if self.spall:
            c = np.zeros(len(P), np.int64)
            for li in np.unique(i):
                sel = i == li
                c[sel] = np.searchsorted(np.asarray(self.cross_offs[int(li)], float), P[sel] @ self.cross_n[int(li)])
            k = (k * 48271 + c + 15485863) % M_HASH
        return (k * 48271 + 104729 * (self.region + 1)) % M_HASH

    def depth_of(self, cell):
        if self.spall:
            cell = np.asarray(cell, np.int64)
            pos = np.minimum(np.searchsorted(self.table_keys, cell), len(self.table_keys) - 1)
            if not (self.table_keys[pos] == cell).all():
                raise KeyError('cell key not in the region depth table')
            return self.table_R[pos]
        h = ((cell * 16807 + self.seed) % M_HASH * 48271 % M_HASH) / M_HASH
        return self.depth * h ** 1.5


# Upper bounds of the work expanded at once by Locator: (face, grid cell) pairs when building the grid, (query point,
# candidate face) pairs per query batch. A coarse mesh expands each triangle into more cells of the fine grid, so the
CELL_PAIR_BUDGET = 30_000_000
QUERY_PAIR_BUDGET = 8_000_000


class Locator:
    """rock / air / none of points against one region: the FIRST prism top (faces in ascending order within a grid cell along
    m) whose barycentric test passes decides; s < 0 air, else rock iff s >= depth(cell(p)) x taper(barycentric). The
    grid cell starts at ``cell`` and doubles until the expanded (face, cell) pairs fit CELL_PAIR_BUDGET; the cell size
    changes only the candidate lists, not the answer."""

    def __init__(self, R, cell=.005, pair_budget=None):
        V, F, m = R.V, R.F, np.asarray(R.m, float)
        budget = CELL_PAIR_BUDGET if pair_budget is None else int(pair_budget)
        self.R, self.m = R, m
        g1 = np.cross(m, [0., 0., 1.])
        g1 /= np.linalg.norm(g1)
        g2 = np.cross(m, g1)
        self.g1, self.g2, self.c = g1, g2, V.mean(0)
        uvV = np.c_[(V - self.c) @ g1, (V - self.c) @ g2]
        tri2 = uvV[F]
        tmin, tmax = tri2.min(1), tri2.max(1)
        while True:
            lo = np.floor(tmin / cell).astype(np.int64)
            hi = np.floor(tmax / cell).astype(np.int64)
            nx, ny = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
            cnt = nx * ny
            if int(cnt.sum()) <= budget:
                break
            cell *= 2.
        self.cell = cell
        self.pairs = int(cnt.sum())
        fidx = np.repeat(np.arange(len(F)), cnt)
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ix = lo[fidx, 0] + off // ny[fidx]
        iy = lo[fidx, 1] + off % ny[fidx]
        self.B = 1 << 21
        key = (ix + self.B) * (2 * self.B) + (iy + self.B)
        del ix, iy, off
        order = np.lexsort((fidx, key))
        key, self.cand = key[order], fidx[order]
        self.ukey, self.start = np.unique(key, return_index=True)
        self.end = np.r_[self.start[1:], len(key)]
        T = V[F]
        self.t0, self.e0, self.e1 = T[:, 0], T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        self.nrm = np.cross(self.e0, self.e1)
        self.mn = self.nrm @ m
        self.d00 = (self.e0 * self.e0).sum(1)
        self.d01 = (self.e0 * self.e1).sum(1)
        self.d11 = (self.e1 * self.e1).sum(1)
        self.den = self.d00 * self.d11 - self.d01 * self.d01
        self.taper_f = R.taper[F]

    def floor_gap(self, P):
        """s - floor depth (m along m) at each point from its first prism-top hit (nan: no hit): 0 on the exact floor."""
        out = np.full(len(P), np.nan)
        for i, p in enumerate(np.atleast_2d(P)):
            k2 = np.floor(np.r_[(p - self.c) @ self.g1, (p - self.c) @ self.g2] / self.cell).astype(np.int64)
            key = (k2[0] + self.B) * (2 * self.B) + (k2[1] + self.B)
            pos = int(np.searchsorted(self.ukey, key))
            if pos >= len(self.ukey) or self.ukey[pos] != key:
                continue
            for ci in self.cand[self.start[pos]:self.end[pos]]:
                sv = ((p - self.t0[ci]) @ self.nrm[ci]) / self.mn[ci]
                w = p - sv * self.m - self.t0[ci]
                d20, d21 = w @ self.e0[ci], w @ self.e1[ci]
                b1 = (self.d11[ci] * d20 - self.d01[ci] * d21) / self.den[ci]
                b2 = (self.d00[ci] * d21 - self.d01[ci] * d20) / self.den[ci]
                if b1 >= -1e-12 and b2 >= -1e-12 and b1 + b2 <= 1 + 1e-12:
                    tw = (np.array([1 - b1 - b2, b1, b2]) * self.taper_f[ci]).sum()
                    out[i] = sv - self.R.depth_of(self.R.cell_of(p[None]))[0] * tw
                    break
        return out

    def _lookup(self, P):
        k2 = np.floor(np.c_[(P - self.c) @ self.g1, (P - self.c) @ self.g2] / self.cell).astype(np.int64)
        key = (k2[:, 0] + self.B) * (2 * self.B) + (k2[:, 1] + self.B)
        pos = np.minimum(np.searchsorted(self.ukey, key), len(self.ukey) - 1)
        found = self.ukey[pos] == key
        n = np.where(found, self.end[pos] - self.start[pos], 0)
        return pos, n

    def query(self, P, chunk=200000, pair_budget=None):
        """Batches of query points whose candidate pairs fit QUERY_PAIR_BUDGET (one point per batch at least)."""
        budget = QUERY_PAIR_BUDGET if pair_budget is None else int(pair_budget)
        P = np.asarray(P, float)
        out = np.full(len(P), NONE, np.int8)
        for a in range(0, len(P), chunk):
            Pc = P[a:a + chunk]
            pos, n = self._lookup(Pc)
            c = np.cumsum(n)
            s0 = 0
            while s0 < len(Pc):
                base = c[s0 - 1] if s0 else 0
                s1 = max(int(np.searchsorted(c, base + budget, side='right')), s0 + 1)
                out[a + s0:a + s1] = self._query(Pc[s0:s1], pos[s0:s1], n[s0:s1])
                s0 = s1
        return out

    def _query(self, P, pos, n):
        res = np.full(len(P), NONE, np.int8)
        qi = np.flatnonzero(n > 0)
        if not len(qi):
            return res
        s0 = self.start[pos[qi]]
        nn = n[qi]
        q = np.repeat(qi, nn)
        ci = self.cand[np.repeat(s0, nn) + (np.arange(nn.sum()) - np.repeat(np.cumsum(nn) - nn, nn))]
        p = P[q]
        s = ((p - self.t0[ci]) * self.nrm[ci]).sum(1) / self.mn[ci]
        w = p - s[:, None] * self.m - self.t0[ci]
        d20 = (w * self.e0[ci]).sum(1)
        d21 = (w * self.e1[ci]).sum(1)
        b1 = (self.d11[ci] * d20 - self.d01[ci] * d21) / self.den[ci]
        b2 = (self.d00[ci] * d21 - self.d01[ci] * d20) / self.den[ci]
        hit = (b1 >= -1e-12) & (b2 >= -1e-12) & (b1 + b2 <= 1 + 1e-12)
        hp = np.nonzero(hit)[0]
        if not len(hp):
            return res
        uq, first = np.unique(q[hp], return_index=True)
        j = hp[first]
        f, sj = ci[j], s[j]
        r = np.where(sj < 0, AIR, ROCK).astype(np.int8)
        inn = sj >= 0
        if inn.any():
            tw = (np.c_[1 - b1[j] - b2[j], b1[j], b2[j]] * self.taper_f[f]).sum(1)
            dep = self.R.depth_of(self.R.cell_of(P[uq[inn]]))
            r[inn] = np.where(sj[inn] >= dep * tw[inn], ROCK, AIR)
        res[uq] = r
        return res


def _outer(loc, T, nrm, area, offsets):
    """per face (T: (n, 3, 3)) the first offset index (column of offsets) at which all probes pass, -1 none; also the
    per-face result of the last tried offset."""
    cc = T.mean(1)
    pts = np.concatenate([cc[:, None], .75 * cc[:, None] + .25 * T], 1)      # (n, 4, 3)
    passed = np.full(len(T), -1, np.int64)
    todo = np.ones(len(T), bool)
    for e in range(offsets.shape[1]):
        sel = np.flatnonzero(todo)
        if not len(sel):
            break
        eps = offsets[sel, e][:, None, None]
        n_ = nrm[sel][:, None, :]
        ri = loc.query((pts[sel] - eps * n_).reshape(-1, 3)).reshape(-1, 4)
        ro = loc.query((pts[sel] + eps * n_).reshape(-1, 3)).reshape(-1, 4)
        ok = ((ri == ROCK) & (ro == AIR)).all(1)
        passed[sel[ok]] = e
        todo[sel[ok]] = False
    return passed


def _closure(V, F, faces, tol=1e-6, step=5e-4, gsz=.004):
    """samples along the topologically open edges of ``faces`` farther than tol from every open edge of another face
    of the whole mesh (the B region's outer edges are matched by the neighbouring non-B faces)."""
    allf = np.arange(len(F))
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    own = np.tile(allf, 3)
    Es = np.sort(E, 1)
    u, inv, cnt = np.unique(Es, axis=0, return_inverse=True, return_counts=True)
    op = cnt[inv.reshape(-1)] == 1
    isb = np.zeros(len(F), bool)
    isb[faces] = True
    A, B, O = V[E[op, 0]], V[E[op, 1]], own[op]
    from_b = isb[O]
    return _match_edges(A, B, O, from_b, tol=tol, step=step, gsz=gsz)


def _match_edges(A, B, O, from_b, tol=1e-6, step=5e-4, gsz=.004, max_points=200):
    """Match sampled source edges against edges of another polygon/face owner."""
    if not len(A):
        return {'open_edges': 0, 'samples': 0, 'gap': 0, 'gap_max_m': 0.}
    L = np.linalg.norm(B - A, axis=1)
    # grid of segments: cells of gsz touched by points every gsz/2 along each segment
    ns = np.maximum(np.ceil(L / (gsz / 2)).astype(np.int64), 1) + 1
    sid = np.repeat(np.arange(len(A)), ns)
    t = (np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns)) / np.repeat(ns - 1, ns).clip(1)
    Q = A[sid] + (B - A)[sid] * t[:, None]
    K = np.floor(Q / gsz).astype(np.int64)
    Bk = 1 << 20
    key = ((K[:, 0] + Bk) * (2 * Bk) + (K[:, 1] + Bk)) * (2 * Bk) + (K[:, 2] + Bk)
    pair = np.unique(np.c_[key, sid], axis=0)
    ukey, start = np.unique(pair[:, 0], return_index=True)
    end = np.r_[start[1:], len(pair)]
    segs = pair[:, 1]
    # samples: on the open edges of the B faces only
    k_s = np.where(from_b, np.maximum(np.ceil(L / step).astype(np.int64), 1), 0)
    ssid = np.repeat(np.arange(len(A)), k_s)
    ts = (np.arange(k_s.sum()) - np.repeat(np.cumsum(k_s) - k_s, k_s) + .5) / np.repeat(k_s, k_s)
    S = A[ssid] + (B - A)[ssid] * ts[:, None]
    KS = np.floor(S / gsz).astype(np.int64)
    dmin = np.full(len(S), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kk = ((KS[:, 0] + dx + Bk) * (2 * Bk) + (KS[:, 1] + dy + Bk)) * (2 * Bk) + (KS[:, 2] + dz + Bk)
                pos = np.minimum(np.searchsorted(ukey, kk), len(ukey) - 1)
                hit = np.flatnonzero(ukey[pos] == kk)
                for c0 in range(0, len(hit), 2000000):
                    h = hit[c0:c0 + 2000000]
                    n = end[pos[h]] - start[pos[h]]
                    qi = np.repeat(h, n)
                    cj = segs[np.repeat(start[pos[h]], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))]
                    keep = O[cj] != O[ssid[qi]]
                    qi, cj = qi[keep], cj[keep]
                    a, ab = A[cj], (B - A)[cj]
                    tt = np.clip(((S[qi] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30), 0, 1)
                    d = np.linalg.norm(S[qi] - (a + tt[:, None] * ab), axis=1)
                    np.minimum.at(dmin, qi, d)
    gap = dmin > tol
    return {'open_edges_b': int(from_b.sum()), 'open_edges_all': int(len(A)), 'samples': int(len(S)), 'gap': int(gap.sum()),
            'gap_max_m': float(dmin[np.isfinite(dmin)].max()) if np.isfinite(dmin).any() else None,
            'gap_samples_unmatched_any': int((~np.isfinite(dmin)).sum()), 'gap_points': S[gap][:max_points].tolist(),
            'gap_values_m': dmin[gap][:max_points].tolist()}


def _weld_acceptance(welds):
    """Require measured individual tolerance compliance, including plane caps."""
    result = {k: max((w.get(k, 0.) for w in welds), default=0.)
              for k in ('max_move_m', 'max_move_fine_m', 'max_move_coarse_m')}
    counts = [w.get('own_tolerance_violations') for w in welds]
    result['own_tolerance_violations'] = sum(counts) if counts and all(v is not None for v in counts) else None
    result['pass'] = result['max_move_m'] < 5e-4 and result['own_tolerance_violations'] == 0
    return result


PASS, FAIL, NA = 'PASS', 'FAIL', 'NOT_APPLICABLE'


def _st(ok, **kw):
    """an applicable item: PASS iff ok is True (None / missing evidence = FAIL)."""
    status = PASS if ok is True else FAIL
    return {**kw, 'status': status, 'pass': status == PASS}


def _na(reason, **kw):
    return {**kw, 'status': NA, 'pass': None, 'reason': reason}


def route_of(req):
    """Applicability of the items from the declared route (request display_stats.blockface), not from data presence.
    Returns dict(b: B residual blocks declared, detail: 'mesh' (scan-detail transfer) / 'relief' (material-height
    relief) / None)."""
    bf = (req.get('display_stats') or {}).get('blockface')
    if not bf or bf.get('mode') != 'wall':
        raise ValueError('wallcheck: request does not declare a whole-wall blockface (display_stats.blockface.mode wall)')
    det = bf.get('detail') or {}
    method = det.get('method')
    if method not in (None, 'mesh', 'relief'):
        raise ValueError('wallcheck: unknown detail method %r' % method)
    return {'b': bool(bf.get('b_enabled', True)), 'detail': method}


def _orientation(F):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    Es = np.sort(E, 1)
    u, inv, cnt = np.unique(Es, axis=0, return_inverse=True, return_counts=True)
    fwd = (E[:, 0] < E[:, 1]).astype(np.int64)
    s_f = np.bincount(inv.reshape(-1), weights=fwd, minlength=len(u))
    inter = cnt == 2
    return {'interior_edges': int(inter.sum()), 'same_direction_edges': int((inter & (s_f != 1)).sum()), 'non_manifold_edges': int((cnt > 2).sum())}


def check_wall(staged, out_json=None, log=print):
    t0 = time.time()
    staged = Path(staged)
    z = np.load(staged / 'display.npz')
    req = json.loads((staged / 'request.json').read_text(encoding='utf8'))
    route = route_of(req)
    bf = req['display_stats']['blockface']
    det = bf.get('detail') or {}
    V, F = z['vertices'], z['faces']
    res = {'staged': str(staged), 'route': route, 'faces': int(len(F)), 'vertices': int(len(V))}
    breg = z['b_region'] if 'b_region' in z.files else None
    res['b_faces'] = int((breg >= 0).sum()) if breg is not None else None
    m = np.asarray(bf['m'], float)
    # X1 (always): fold guard; donor constraints only for the scan transfer
    fg = req['display_stats'].get('fold_guard_final_flipped')
    res['X1'] = _st(fg == 0, fold_guard_final_flipped=fg)
    if route['detail'] == 'mesh':
        fps = det.get('footprints') or []
        cf = [f.get('constraints_failed') for f in fps]
        nd = [f.get('vertices_without_donor_surface') for f in fps]
        ev = bool(fps) and all(v is not None for v in cf + nd)
        res['X1_donor'] = _st(ev and sum(cf) == 0 and sum(nd) == 0, footprints=len(fps), constraints_failed=sum(v or 0 for v in cf),
                              vertices_without_donor_surface=sum(v or 0 for v in nd), evidence_complete=ev)
    else:
        res['X1_donor'] = _na('no scan-detail donor in this route (detail.method %r)' % route['detail'])
    # X3: detail displacement along m only
    if route['detail'] in ('mesh', 'relief'):
        key = 'xfer_ref' if route['detail'] == 'mesh' else 'relief_dvec'
        if key not in z.files or 'rock_vertices' not in z.files:
            res['X3'] = _st(None, missing=key)
        else:
            nr = int(z['rock_vertices'])
            d = ((V - z['xfer_ref']) if key == 'xfer_ref' else z['relief_dvec'])[:nr]
            off = np.linalg.norm(d - (d @ m)[:, None] * m, axis=1)
            res['X3'] = _st(bool(off.max() <= 1e-9) if len(off) else None, source=key, max_offaxis_m=float(off.max()) if len(off) else None,
                            disp_max_m=float(np.abs(d @ m).max()) if len(off) else None)
    else:
        res['X3'] = _na('no detail displacement in this route')
    # partition of unity (scan footprints only)
    if route['detail'] == 'mesh':
        dev, nofp = det.get('pou_core_max_dev'), det.get('core_vertices_without_footprint')
        res['partition_of_unity'] = _st(dev is not None and nofp is not None and dev <= 1e-9 and nofp == 0, core_max_dev=dev,
                                        core_vertices_without_footprint=nofp)
    else:
        res['partition_of_unity'] = _na('no footprint layout in this route')
    cmst = (req.get('display_stats') or {}).get('cm')
    if cmst is not None:
        sc = cmst.get('shape_check')
        se = cmst.get('shape_error')
        if sc is not None:
            res['shape_error'] = _st(bool(sc.get('pass')), **{k: v for k, v in sc.items() if k not in ('rule', 'pass')})
        else:
            ok = se is not None and se['ratio_p50_p99_max'][1] <= 1. and se['ratio_p50_p99_max'][2] <= 2.
            res['shape_error'] = _st(ok if se is not None else None, **({k: v for k, v in se.items() if k != 'rule'} if se else {'missing': 'display_stats.cm.shape_error'}))
    else:
        res['shape_error'] = _na('no cm refinement in this route')
    # X5 over the whole exported mesh (always)
    res['X5_all'] = _st(None, **_orientation(F))
    res['X5_all'] = _st(res['X5_all']['same_direction_edges'] == 0 and res['X5_all']['non_manifold_edges'] == 0,
                        **{k: v for k, v in res['X5_all'].items() if k not in ('status', 'pass')})
    log('X5_all done %.0f s' % (time.time() - t0))
    b_items = ('G-a', 'G-b', 'G-c2', 'winding', 'X4', 'X5', 'X6', 'X6_raw', 'X6_welded')
    if not route['b']:
        for k in b_items:
            res[k] = _na('route declares no B residual blocks')
    else:
        blocks = bf.get('blocks')
        if breg is None or 'b_cell' not in z.files or 'riser_faces' not in z.files or not blocks:
            for k in b_items:
                res[k] = _st(None, missing='b_region / b_cell / riser_faces / blockface.blocks')
        else:
            _check_b(res, staged, z, req, bf, V, F, m, log, t0)
    res['seconds'] = time.time() - t0
    items = {k: v for k, v in res.items() if isinstance(v, dict) and 'status' in v}
    res['failed_items'] = sorted(k for k, v in items.items() if v['status'] == FAIL)
    res['not_applicable_items'] = sorted(k for k, v in items.items() if v['status'] == NA)
    res['status'] = FAIL if res['failed_items'] else PASS
    res['pass'] = res['status'] == PASS
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf8')
    return res


def _check_b(res, staged, z, req, bf, V, F, m, log, t0):
    breg, bcell, riser = z['b_region'], z['b_cell'], z['riser_faces']
    # G-a
    welds = [b.get('weld') for b in bf['blocks']]
    if any(w is None for w in welds):
        res['G-a'] = _st(None, missing='weld record of a block region')
    else:
        ga = _weld_acceptance(welds)
        res['G-a'] = _st(bool(ga.pop('pass')), **ga)
    bf_idx = np.flatnonzero(breg >= 0)
    if not len(bf_idx):
        for k in ('G-b', 'G-c2', 'winding', 'X4', 'X5', 'X6', 'X6_raw', 'X6_welded'):
            res[k] = _st(None, missing='no B faces in the export although the route declares B')
        return
    o5 = _orientation(F[bf_idx])
    res['X5'] = _st(o5['same_direction_edges'] == 0 and o5['non_manifold_edges'] == 0, **o5)
    log('X5 done %.0f s' % (time.time() - t0))
    regions = sorted(int(r) for r in np.unique(breg[breg >= 0]))
    res['regions'] = {}
    gb_fail, x4_max, gc_fail_total, gc2_open = 0, 0., 0, 0
    wind = {}
    for r in regions:
        R = RegionDump(staged / 'wall_debug' / ('region_%d.npz' % r))
        fr = np.flatnonzero(breg == r)
        T = V[F[fr]]
        cen = T.mean(1)
        fl = fr[~riser[fr]]
        cfl = V[F[fl]].mean(1)
        own = R.cell_of(cfl)
        gb = int((own != bcell[fl]).sum())
        gb_fail += gb
        rs = fr[riser[fr]]
        x4 = 0.
        if len(rs):
            Tr = V[F[rs]]
            dist = np.abs(np.einsum('nkd,pd->nkp', Tr, R.cut_n) - R.cut_d[None, None, :]).max(1).min(1)
            x4 = float(dist.max())
            x4_max = max(x4_max, x4)
        # exported winding (display.npz face order = GLB order): no checker flip
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        a = .5 * np.linalg.norm(nrm, axis=1)
        nrm /= np.maximum(2 * a, 1e-300)[:, None]
        fl_local = ~riser[fr]
        out_area = float((a[fl_local] * ((nrm[fl_local] @ -m) > 0)).sum())
        tot_area = float(a[fl_local].sum())
        wind[str(r)] = {'floor_area_m2': tot_area, 'floor_area_facing_out_m2': out_area, 'floor_area_facing_in_m2': tot_area - out_area}
        loc = Locator(R)
        offs = np.c_[np.minimum(1e-4, .05 * np.sqrt(a)), np.full(len(fr), 1e-5), np.full(len(fr), 1e-6)]
        passed = _outer(loc, T, nrm, a, offs)
        fail = np.flatnonzero(passed < 0)
        rows = []
        for i in fail:
            L = np.linalg.norm(T[i] - T[i][[1, 2, 0]], axis=1)
            w = 2 * a[i] / L.sum()
            tiers = [e for e in (1e-7, 1e-8, 1e-9) if e <= .1 * w]
            tr = {}
            for e in tiers:
                p1 = _outer(loc, T[i:i + 1], nrm[i:i + 1], a[i:i + 1], np.array([[e]]))[0] == 0
                p2 = _outer(loc, T[i:i + 1], -nrm[i:i + 1], a[i:i + 1], np.array([[e]]))[0] == 0
                tr[str(e)] = {'pass': bool(p1), 'reversed_pass': bool(p2)}
            closed = len(tiers) >= 2 and all(v['pass'] and not v['reversed_pass'] for v in tr.values())
            if riser[fr[i]]:
                off = float(np.abs(np.einsum('kd,pd->kp', T[i], R.cut_n) - R.cut_d[None]).max(0).min())
            else:
                g_ = loc.floor_gap(T[i])
                off = float(np.nanmax(np.abs(g_))) if np.isfinite(g_).any() else np.inf
            weld_def = False
            if np.isfinite(off) and off <= 5e-4:
                cc_ = T[i].mean(0)
                q_ = np.vstack([cc_, .5 * cc_ + .5 * T[i]])
                e2 = off + 1e-4
                weld_def = bool((loc.query(q_ - e2 * nrm[i]) == ROCK).all() and (loc.query(q_ + e2 * nrm[i]) == AIR).all())
            rows.append({'face': int(fr[i]), 'riser': bool(riser[fr[i]]), 'width_um': float(w * 1e6), 'centroid': cen[i].tolist(), 'tiers': tr,
                         'gc2_closed': bool(closed), 'max_off_own_surface_mm': off * 1e3, 'weld_deformation_within_tol': weld_def})
        n_open = sum(not x['gc2_closed'] and not x['weld_deformation_within_tol'] for x in rows)
        n_weld = sum(not x['gc2_closed'] and x['weld_deformation_within_tol'] for x in rows)
        gc_fail_total += len(rows)
        gc2_open += n_open
        res['regions'][str(r)] = {'faces': int(len(fr)), 'floors': int(len(fl)), 'risers': int(len(rs)), 'G-b_not_own_cell': gb,
                                  'X4_riser_plane_dist_max_m': x4, 'G-c_fail': int(len(rows)), 'G-c2_not_closed': n_open, 'weld_deformation_within_tol': n_weld,
                                  'locator_cell_m': loc.cell, 'locator_pairs': loc.pairs, 'G-c_fail_rows': rows[:500]}
        log('region %d: %d faces, G-b %d, X4 %.2e, G-c fail %d, G-c2 open %d (%.0f s)' % (r, len(fr), gb, x4, len(rows), n_open, time.time() - t0))
        del loc
    res['G-b'] = _st(gb_fail == 0, not_own_cell=gb_fail)
    res['G-c2'] = _st(gc2_open == 0, G_c_fail=gc_fail_total, G_c2_not_closed=gc2_open)
    res['winding'] = _st(all(v['floor_area_facing_in_m2'] == 0. for v in wind.values()), regions=wind,
                         rule='every exported B floor triangle faces out of the rock (n . -m > 0); no checker flip')
    # X4 joint attribution: faces labelled with a joint record lie in that piece's v2 plane
    x4 = {'riser_plane_dist_max_m': x4_max}
    if 'coplanar_faces' not in z.files or not (staged / 'display_coplanar.json').exists():
        res['X4'] = _st(None, missing='coplanar_faces / display_coplanar.json', **x4)
    else:
        recs = json.loads((staged / 'display_coplanar.json').read_text(encoding='utf8'))
        cop = z['coplanar_faces']
        from .truth import TruthModel, _plane_groups, _inside_triangles
        truth = TruthModel(req['package'])
        planes, pieces = {}, {}
        for iid, iv, if_ in truth.interfaces:
            if len(if_):
                pieces[iid] = _plane_groups(iv, if_)
                planes[iid] = [(n, o) for n, o, _ in pieces[iid]]
        att, missing_plane, outside_area = {}, [], {}
        for k, rec in enumerate(recs, 1):
            fk = np.flatnonzero(cop == k)
            if not len(fk) or rec['type'] != 'joint':
                continue
            if rec['id'] not in planes:
                missing_plane.append(rec['id'])
                continue
            P = V[F[fk]].reshape(-1, 3)
            att[rec['id']] = min(float(np.abs(P @ n - o).max()) for n, o in planes[rec['id']])
            # as an infinite plane inside its fault block); area of riser faces whose centroid projects outside the piece
            Tj = V[F[fk]]
            cj = Tj.mean(1)
            aj = .5 * np.linalg.norm(np.cross(Tj[:, 1] - Tj[:, 0], Tj[:, 2] - Tj[:, 0]), axis=1)
            inside = np.zeros(len(fk), bool)
            for n, o, tri in pieces[rec['id']]:
                dj = cj @ n - o
                on = np.abs(dj) < 1e-6
                if on.any():
                    inside[on] |= _inside_triangles(cj[on] - dj[on, None] * n, tri, n, eps=1e-3)
            outside_area[rec['id']] = float(aj[~inside].sum())
        jmax = max(att.values()) if att else None
        out_tot = float(sum(outside_area.values()))
        ok = x4_max <= 1e-12 and not missing_plane and (jmax is None or jmax <= 1e-9) and out_tot == 0.
        res['X4'] = _st(ok, **x4, joint_records=len(att), joint_plane_dist_max_m=jmax, joint_records_without_v2_plane=missing_plane,
                        joint_riser_area_outside_v2_piece_m2=out_tot,
                        note='joint attribution has no joint-labelled faces' if not att else None)
    log('X4 joints done %.0f s' % (time.time() - t0))
    x6 = _closure(V, F, bf_idx)
    res['X6'] = _st(x6['gap'] == 0, **x6)
    log('X6 final done %.0f s' % (time.time() - t0))
    from .wall_evidence import check_polygon_stages
    try:
        stages = check_polygon_stages(staged)
        for k in ('raw', 'welded'):
            st_ = stages[k]
            st_.pop('pass', None)
            res['X6_' + k] = _st(st_['gap'] == 0, **st_)
    except FileNotFoundError as e:
        res['X6_raw'] = _st(None, missing=str(e))
        res['X6_welded'] = _st(None, missing=str(e))
