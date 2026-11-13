"""Exact geological truth on an arbitrary display surface.

The v2 model is the single source of truth. Region labels are v2 point queries at the actual
3D positions of the displayed surface; line labels are the intersections of the displayed
surface with the v2 fault/joint/contact interface surfaces. Every label is therefore what the
geological model says at the visible point, and carries the body's event lineage.
"""
import json
from pathlib import Path

import numpy as np


class TruthModel:
    def __init__(self, package):
        from ..modeling.compiler import compile_model
        from ..modeling.decode import decode_model
        package = Path(package)
        resolved = package / 'resolved_model.json'
        spec = None
        if resolved.exists():
            spec = json.loads(resolved.read_text(encoding='utf8')).get('numeric_model')
        if spec is None:
            spec = json.loads((package / 'model_spec.json').read_text(encoding='utf8'))
        self.package_path = package
        compiled = compile_model(decode_model(spec))
        self.snapshot = compiled.snapshots[compiled.final_stage_id]
        self.lineage = {oid: list(body.lineage) for oid, body in self.snapshot.bodies.items()}
        self.units = {oid: body.unit_id for oid, body in self.snapshot.bodies.items()}
        interfaces = json.loads((package / 'interfaces.json').read_text(encoding='utf8'))
        meshes = np.load(package / 'interfaces.npz')
        self.interfaces = [(rec['id'], meshes['vertices_%d' % k], meshes['faces_%d' % k])
                           for k, rec in enumerate(interfaces)]
        self.feature_kind = {f['id']: f['kind'] for f in spec.get('features', [])}
        intrusion_events = {e['id'] for e in spec.get('events', []) if e['kind'] == 'intrusion'}
        self.interfaces = [(iid, v, self._true_contact(v, f) if iid.split(':', 1)[0] in intrusion_events else f)
                           for iid, v, f in self.interfaces]

    def _true_contact(self, vertices, faces, eps=2e-3):
        """An intrusion's contact interface is the boundary of (intrusion ∩ footprint): parts of it lie
        on the domain box or inside later intrusions. Keep only triangles with different bodies on
        their two sides (a real geological contact)."""
        if not len(faces):
            return faces
        tri = vertices[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1)
        ok = length > 1e-14
        n = n / np.maximum(length, 1e-30)[:, None]
        c = tri.mean(1)
        sa, ba, _, _ = self.query(c + eps * n)
        sb, bb, _, _ = self.query(c - eps * n)
        keep = ok & (sa == 'occupied') & (sb == 'occupied') & (ba != bb)
        return faces[keep]

    def fabric(self, points, bodies, kind):
        """Per-point (valid, normal, scalar coordinate) of the ``kind`` fabric ('bedding' or
        'foliation') hosting each point's body; fold limbs and faulted blocks carry their own."""
        points = np.asarray(points, float)
        valid = np.zeros(len(points), bool)
        normals = np.zeros((len(points), 3))
        values = np.full(len(points), np.nan)
        bodies = np.asarray(bodies, object)
        for fid, state in self.snapshot.fabrics.items():
            if self.feature_kind.get(fid) != kind:
                continue
            mask = np.isin(bodies, list(state.hosts)) & ~valid
            if not mask.any():
                continue
            s = state.field.sample(points[mask])
            idx = np.nonzero(mask)[0][s.valid]
            valid[idx] = True
            normals[idx] = s.normals[s.valid]
            if s.scalar_coordinate is not None:
                values[idx] = s.scalar_coordinate[s.valid]
        return valid, normals, values

    # same snapshot from the package and answers its contiguous chunk, so the result is the serial result
    workers = 1
    _pool = None
    _package = None

    def _parallel(self, points, chunk):
        from concurrent.futures import ProcessPoolExecutor
        if TruthModel._pool is None or TruthModel._package != str(self.package_path):
            if TruthModel._pool is not None:
                TruthModel._pool.shutdown()
            TruthModel._pool = ProcessPoolExecutor(self.workers, initializer=_worker_init, initargs=(str(self.package_path),))
            TruthModel._package = str(self.package_path)
        n = len(points)
        step = max(5000, -(-n // (4 * self.workers)))
        parts = list(TruthModel._pool.map(_worker_query, [np.asarray(points[a:a + step], float) for a in range(0, n, step)]))
        return [np.concatenate([p_[k] for p_ in parts]) for k in range(4)]

    def query(self, points, chunk=200000, min_parallel=50000):
        """(status, body, unit, lithology) arrays for ENU points. ``min_parallel``: smallest batch sent to the worker processes."""
        from ..modeling.query import query_stage
        if self.workers > 1 and len(points) >= min_parallel and getattr(self, 'package_path', None) is not None:
            return self._parallel(points, chunk)
        out = [[], [], [], []]
        for s in range(0, len(points), chunk):
            q = query_stage(self.snapshot, np.asarray(points[s:s + chunk], float))
            for lst, arr in zip(out, (q.status, q.body_ids, q.unit_ids, q.lithology_ids)):
                lst.append(np.asarray(arr, object))
        return [np.concatenate(x) if x else np.zeros(0, object) for x in out]


_WORKER = {}


def _worker_init(package):
    _WORKER['truth'] = TruthModel(package)


def _worker_query(points):
    t = _WORKER['truth']
    t.workers = 1
    return t.query(points)


_TRI_GRID = {}


def inside_triangles_grid(points, tri, normal, eps=1e-6):
    """_inside_triangles with a uniform 2D grid over the triangles: each point is tested only against the triangles whose
    bounding box, grown by 2 eps x their longest edge, overlaps its cell; the per-triangle predicate and its
    arithmetic are those of _inside_triangles, so the answer is the same. The grid is cached per triangle array."""
    axis = int(np.argmax(np.abs(normal)))
    keep = [k for k in range(3) if k != axis]
    key = (id(tri), len(tri), float(eps), axis)
    g = _TRI_GRID.get(key)
    if g is None or g[0] is not tri:
        t = tri[:, :, keep]
        edge = np.linalg.norm(t - t[:, [1, 2, 0]], axis=2).max(1)
        lo = t.min(1) - (2 * eps * edge + 1e-12)[:, None]
        hi = t.max(1) + (2 * eps * edge + 1e-12)[:, None]
        g0, g1 = lo.min(0), hi.max(0)
        G = int(np.clip(np.sqrt(len(t)) * 2, 1, 256))
        cs = np.maximum((g1 - g0) / G, 1e-12)
        i0 = np.clip(((lo - g0) / cs).astype(np.int64), 0, G - 1)
        i1 = np.clip(((hi - g0) / cs).astype(np.int64), 0, G - 1)
        nx, ny = i1[:, 0] - i0[:, 0] + 1, i1[:, 1] - i0[:, 1] + 1
        n = nx * ny
        r = np.repeat(np.arange(len(t)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        cell = (i0[r, 0] + k % nx[r]) * G + (i0[r, 1] + k // nx[r])
        o = np.argsort(cell, kind='stable')
        start = np.searchsorted(cell[o], np.arange(G * G + 1))
        g = (tri, t, g0, cs, G, r[o], start)
        _TRI_GRID[key] = g
        if len(_TRI_GRID) > 4096:
            _TRI_GRID.clear()
            _TRI_GRID[key] = g
    _, t, g0, cs, G, cand, start = g
    p = points[:, keep]
    inside = np.zeros(len(points), bool)
    c = np.floor((p - g0) / cs).astype(np.int64)
    inb = (c >= 0).all(1) & (c < G).all(1)
    pi = np.flatnonzero(inb)
    if not len(pi):
        return inside
    cid = c[pi, 0] * G + c[pi, 1]
    cnt = start[cid + 1] - start[cid]
    rp = np.repeat(np.arange(len(pi)), cnt)
    kk = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ti = cand[start[cid][rp] + kk]
    a, b, cc = t[ti, 0], t[ti, 1], t[ti, 2]
    v0, v1 = b - a, cc - a
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(den) > 1e-18
    w = p[pi[rp]] - a
    l1 = (w[:, 0] * v1[:, 1] - w[:, 1] * v1[:, 0]) / np.where(ok, den, 1)
    l2 = (v0[:, 0] * w[:, 1] - v0[:, 1] * w[:, 0]) / np.where(ok, den, 1)
    hit = (l1 >= -eps) & (l2 >= -eps) & (l1 + l2 <= 1 + eps) & ok
    np.logical_or.at(inside, pi[rp[hit]], True)
    return inside


def _plane_groups(vertices, faces, tol=1e-7):
    """Split an interface mesh into coplanar triangle groups: [(normal, offset, triangles)]."""
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    keep = length > 1e-14
    tri, n = tri[keep], n[keep] / length[keep, None]
    groups = []
    used = np.zeros(len(tri), bool)
    for i in range(len(tri)):
        if used[i]:
            continue
        ni = n[i]
        di = float(ni @ tri[i, 0])
        same = (~used) & (np.abs(np.abs(n @ ni) - 1) < 1e-6) & (np.abs(tri[:, 0] @ ni - di) < 1e-5)
        used |= same
        groups.append((ni, di, tri[same]))
    return groups


def _inside_triangles(points, tri, normal, eps=1e-6):
    """Points (already on the plane) inside the union of coplanar triangles (2D barycentric test)."""
    axis = int(np.argmax(np.abs(normal)))
    keep = [k for k in range(3) if k != axis]
    p = points[:, keep]
    t = tri[:, :, keep]
    inside = np.zeros(len(points), bool)
    for s in range(0, len(t), 256):
        a, b, c = t[s:s + 256, 0], t[s:s + 256, 1], t[s:s + 256, 2]
        v0, v1 = b - a, c - a
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        ok = np.abs(den) > 1e-18
        w = p[:, None, :] - a[None]
        l1 = (w[..., 0] * v1[None, :, 1] - w[..., 1] * v1[None, :, 0]) / np.where(ok, den, 1)[None]
        l2 = (v0[None, :, 0] * w[..., 1] - v0[None, :, 1] * w[..., 0]) / np.where(ok, den, 1)[None]
        hit = (l1 >= -eps) & (l2 >= -eps) & (l1 + l2 <= 1 + eps) & ok[None]
        inside |= hit.any(1)
    return inside


def surface_intersections(V, F, interfaces, min_points=2, with_normals=False):
    """Polylines where the display mesh (V, F) crosses each interface surface.

    Returns [(interface_id, (n, 3) points)]. Points lie on the display mesh and
    interface plane, at display edges or finite-interface boundaries. Segments
    are clipped to the interface's actual extent.
    """
    result = []
    lo, hi = V.min(0), V.max(0)
    for iid, iv, if_ in interfaces:
        if not len(if_):
            continue
        if (iv.max(0) < lo).any() or (iv.min(0) > hi).any():
            continue
        for normal, offset, tri in _plane_groups(iv, if_):
            d = V @ normal - offset
            sgn = d > 0
            fs = sgn[F]
            crossing = fs.any(1) & ~fs.all(1)
            if not crossing.any():
                continue
            cf = F[crossing]
            # Each crossing triangle has exactly two sign-changing edges.
            edges = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                ea, eb = cf[:, a], cf[:, b]
                ch = sgn[ea] != sgn[eb]
                edges.append((ch, np.minimum(ea, eb), np.maximum(ea, eb)))
            pick = np.stack([e[0] for e in edges], 1)
            lo_i = np.stack([e[1] for e in edges], 1)
            hi_i = np.stack([e[2] for e in edges], 1)
            rows = np.nonzero(pick.sum(1) == 2)[0]
            if not len(rows):
                continue
            pairs = []
            for r in rows:
                ks = np.nonzero(pick[r])[0]
                pairs.append(((lo_i[r, ks[0]], hi_i[r, ks[0]]), (lo_i[r, ks[1]], hi_i[r, ks[1]])))

            def point(edge):
                a, b = edge
                t = d[a] / (d[a] - d[b])
                return V[a] + t * (V[b] - V[a])

            # Chain triangle segments through shared edges into polylines.
            adj = {}
            for e1, e2 in pairs:
                adj.setdefault(e1, []).append(e2)
                adj.setdefault(e2, []).append(e1)
            seen = set()
            for start in list(adj):
                if start in seen or len(adj[start]) == 2:
                    continue
                chain = _walk(start, adj, seen)
                result.extend((i, pts, normal) for i, pts in _clip(iid, chain, point, tri, normal, min_points))
            for start in list(adj):  # closed loops
                if start not in seen:
                    chain = _walk(start, adj, seen)
                    result.extend((i, pts, normal) for i, pts in _clip(iid, chain, point, tri, normal, min_points))
    return result if with_normals else [(i, pts) for i, pts, _ in result]


def _walk(start, adj, seen):
    chain, prev, cur = [start], None, start
    seen.add(start)
    while True:
        nxt = [e for e in adj[cur] if e != prev and e not in seen]
        if not nxt:
            # A closed surface intersection also contains the final segment
            # back to its start; repeat that edge so _clip consumes it.
            if start in adj[cur] and start != prev:
                chain.append(start)
            return chain
        prev, cur = cur, nxt[0]
        seen.add(cur)
        chain.append(cur)


def _clip(iid, chain, point, tri, normal, min_points):
    """Clip every crossing segment to the union of the actual finite triangles.

    Testing only the existing chain vertices loses a short finite intersection
    whose ends lie inside one surface triangle, and can bridge an unsupported gap.
    Barycentric coordinates vary linearly along each segment, so their three
    nonnegative intervals give its complete intersection with each triangle.
    """
    pts = np.array([point(e) for e in chain])
    axes = [k for k in range(3) if k != int(np.argmax(np.abs(normal)))]
    t = tri[:, :, axes]
    a, u, v = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    den = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(den) > 1e-18
    a, u, v, den, t = a[valid], u[valid], v[valid], den[valid], t[valid]
    bounds_lo, bounds_hi = t.min(1), t.max(1)
    out, cur = [], []

    def finish():
        if len(cur) >= min_points:
            out.append((iid, np.array(cur)))
        cur.clear()

    for p, q in zip(pts[:-1], pts[1:]):
        ends = np.array([p[axes], q[axes]])
        candidates = ((bounds_hi >= ends.min(0)).all(1) & (bounds_lo <= ends.max(0)).all(1))
        w = ends[:, None] - a[candidates]
        l1 = (w[..., 0] * v[candidates, 1] - w[..., 1] * v[candidates, 0]) / den[candidates]
        l2 = (u[candidates, 0] * w[..., 1] - u[candidates, 1] * w[..., 0]) / den[candidates]
        bary = np.stack([1 - l1 - l2, l1, l2], -1)
        # Rotation can leave an exactly coincident edge a few arithmetic ulps
        # outside (both ends ~-1e-17). Snap only that rounding scale to zero;
        # do not expand the finite triangles by a geometric epsilon.
        roundoff = 8 * np.finfo(float).eps * (1 + np.abs(l1) + np.abs(l2))
        bary[np.abs(bary) <= roundoff[..., None]] = 0.
        start, delta = bary[0], bary[1] - bary[0]
        cross = np.divide(-start, delta, out=np.zeros_like(start), where=delta != 0)
        lo = np.maximum(0., np.where(delta > 0, cross, -np.inf).max(1))
        hi = np.minimum(1., np.where(delta < 0, cross, np.inf).min(1))
        hit = (hi > lo) & ((delta != 0) | (start >= 0)).all(1)
        intervals = []
        for lower, upper in sorted(zip(lo[hit], hi[hit])):
            # Adjacent triangles share a boundary; absorb only roundoff in the
            # segment parameter, preserving actual gaps between finite pieces.
            if intervals and lower <= intervals[-1][1] + 32 * np.finfo(float).eps:
                intervals[-1][1] = max(intervals[-1][1], upper)
            else:
                intervals.append([lower, upper])
        if not intervals:
            finish()
        for lower, upper in intervals:
            first = p if lower == 0 else p + lower * (q - p)
            last = q if upper == 1 else p + upper * (q - p)
            if cur and not np.array_equal(cur[-1], first):
                finish()
            if not cur:
                cur.append(first)
            if not np.array_equal(cur[-1], last):
                cur.append(last)
            if upper < 1:
                finish()
    finish()
    return out


def contact_segments(V, F, truth, iterations=7, known=None):
    """Smooth unit-contact centre lines on a mesh: on every edge joining vertices of different v2
    units the contact point is located by bisection with v2 queries (8 cm / 2^7 ~ 0.6 mm), and in
    each triangle with exactly two such edges the two points are joined. Returns (m, 2, 3).
    ``known``: an earlier truth.query(V) result, reused instead of querying V again."""
    def key_of(st, b, u):
        return np.array([str(uu if uu is not None else bb) if s == 'occupied' else '' for s, bb, uu in zip(st, b, u)], object)

    def key(points):
        st, b, u, _ = truth.query(points)
        return key_of(st, b, u)
    kv = key_of(*known[:3]) if known is not None else key(V)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    cross = (kv[ue[:, 0]] != kv[ue[:, 1]]) & (kv[ue[:, 0]] != '') & (kv[ue[:, 1]] != '')
    if not cross.any():
        return np.zeros((0, 2, 3))
    a, b = V[ue[cross, 0]].copy(), V[ue[cross, 1]].copy()
    ka = kv[ue[cross, 0]]
    for _ in range(iterations):
        m = (a + b) / 2
        same = key(m) == ka
        a = np.where(same[:, None], m, a)
        b = np.where(same[:, None], b, m)
    point = np.full((len(ue), 3), np.nan)
    point[cross] = (a + b) / 2
    per_face = inv.reshape(3, -1).T
    hit = cross[per_face]
    rows = np.nonzero(hit.sum(1) == 2)[0]
    first = np.where(hit[rows, 0], per_face[rows, 0], per_face[rows, 1])
    second = np.where(hit[rows, 2], per_face[rows, 2], per_face[rows, 1])
    return np.stack([point[first], point[second]], 1)
