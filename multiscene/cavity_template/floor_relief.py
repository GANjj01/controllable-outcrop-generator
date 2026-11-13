"""Floor relief, trial copy only. Runs in the source frame on the cavity build_surface_cavity returned. v6. Only faces
with role floor move. After each split round the floor gets Delaunay flips. The longest edge of a floor triangle,
while longer than max_edge = 2 h, is bisected at its midpoint. A cap face sharing such an edge gets the same straight
split, so there are no T junctions and its shape stays; wall feet are at most about h long and stay. New vertices move
along the floor normal by rho(P) - base(P): rho is the wall relief field the build already uses, base is rho linearly
interpolated over the original floor triangles, which span the two wall feet. So: no new amplitude or length; the wall
feet stay; waves much longer than the aperture cancel; the relief shrinks with the aperture. Next to a cap, whose edge
also stays, the displacement fades over the local aperture. Runs as build_surface_cavity's end_trim (hook below), so
the solid it returns is the one the label rebuild keeps."""
import itertools
import os
from pathlib import Path

import numpy as np

CHUNK = 200000
WALLS = ('left_wall', 'right_wall')
CREASE_DEG = 5.    # classifies the flat floor's edges; no amplitude or length


def _eval(field, P):
    out = np.empty(len(P))
    for i in range(0, len(P), CHUNK):
        out[i:i+CHUNK] = np.asarray(field(P[i:i+CHUNK]), float)
    return out


def _edges(tri):
    """(3n, 2) sorted pairs; row 3i + j is the edge tri[i, j] to tri[i, (j + 1) % 3]"""
    return np.sort(np.stack([tri, np.roll(tri, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)


def flip_delaunay(V, tri, flags, group, max_rounds=64):
    """Delaunay edge flips between coplanar floor triangles of one owner, in place. A round flips an independent set: each triangle takes at most its worst edge. Returns the number of flips."""
    rows = np.flatnonzero(flags)
    total = 0
    for _ in range(max_rounds):
        t = tri[rows]
        e, nv = _edges(t), len(V)
        key = e[:, 0]*nv+e[:, 1]
        order = np.argsort(key, kind='stable')
        same = key[order[:-1]] == key[order[1:]]
        r1, r2 = order[:-1][same], order[1:][same]
        t1, j1, t2, j2 = r1//3, r1 % 3, r2//3, r2 % 3
        a, b, c, d = t[t1, j1], t[t1, (j1+1) % 3], t[t1, (j1+2) % 3], t[t2, (j2+2) % 3]

        def angle(o, x, y):
            u, v = V[x]-V[o], V[y]-V[o]
            return np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.einsum('ij,ij->i', u, v))
        excess = angle(c, a, b)+angle(d, a, b)-np.pi
        n1, n2 = np.cross(V[b]-V[a], V[c]-V[a]), np.cross(V[a]-V[b], V[d]-V[b])
        flat = np.einsum('ij,ij->i', n1, n2) >= (1.-1e-8)*np.linalg.norm(n1, axis=1)*np.linalg.norm(n2, axis=1)
        cand = np.flatnonzero((excess > 1e-6) & flat & (group[rows[t1]] == group[rows[t2]]))
        if not len(cand):
            break
        pri = excess[cand]+1e-12*np.arange(len(cand))/len(cand)
        best = np.zeros(len(t))
        np.maximum.at(best, t1[cand], pri)
        np.maximum.at(best, t2[cand], pri)
        keep = cand[(pri == best[t1[cand]]) & (pri == best[t2[cand]])]
        tri[rows[t1[keep]]] = np.c_[a[keep], d[keep], c[keep]]
        tri[rows[t2[keep]]] = np.c_[d[keep], b[keep], c[keep]]
        total += len(keep)
    return total


def refine(V, base, tri, floor, parent, max_edge, group=None, max_rounds=40):
    """Conforming midpoint bisection of every floor edge longer than max_edge. floor: per triangle of tri; other triangles only follow. Returns V, base, tri, parent, rounds."""
    for rounds in range(max_rounds):
        e, nv = _edges(tri), len(V)
        uk, inv = np.unique(e[:, 0]*nv+e[:, 1], return_inverse=True)
        inv = inv.ravel()
        a, b = uk//nv, uk % nv
        length = np.linalg.norm(V[a]-V[b], axis=1)
        # v2: only the longest edge of a floor triangle, which keeps the triangles' shape; a neighbour follows with the same straight split
        per = length[inv].reshape(-1, 3)
        rows = np.flatnonzero(floor[parent])
        mark = np.zeros(len(uk), bool)
        mark[inv.reshape(-1, 3)[rows, per[rows].argmax(1)]] = True
        mark &= length > max_edge
        if not mark.any():
            return V, base, tri, parent, rounds
        idx = np.flatnonzero(mark)
        mid = np.full(len(uk), -1, np.int64)
        mid[idx] = nv+np.arange(len(idx))
        V = np.vstack([V, .5*(V[a[idx]]+V[b[idx]])])
        base = np.r_[base, .5*(base[a[idx]]+base[b[idx]])]
        m = mid[inv].reshape(-1, 3)
        count = (m >= 0).sum(1)
        kids, par = [tri[count == 0]], [parent[count == 0]]
        i = np.flatnonzero(count == 1)
        j = (m[i] >= 0).argmax(1)
        v0, v1, v2, m0 = tri[i, j], tri[i, (j+1) % 3], tri[i, (j+2) % 3], m[i, j]
        kids += [np.c_[v0, m0, v2], np.c_[m0, v1, v2]]
        par += [parent[i]]*2
        i = np.flatnonzero(count == 2)
        s = ((m[i] < 0).argmax(1)+1) % 3
        v0, v1, v2, m0, m1 = tri[i, s], tri[i, (s+1) % 3], tri[i, (s+2) % 3], m[i, s], m[i, (s+1) % 3]
        short = (np.linalg.norm(V[v0]-V[m1], axis=1) <= np.linalg.norm(V[m0]-V[v2], axis=1))[:, None]
        kids += [np.c_[m0, v1, m1], np.where(short, np.c_[v0, m0, m1], np.c_[v0, m0, v2]), np.where(short, np.c_[v0, m1, v2], np.c_[m0, m1, v2])]
        par += [parent[i]]*3
        i = np.flatnonzero(count == 3)
        v, mm = tri[i], m[i]
        kids += [np.c_[v[:, 0], mm[:, 0], mm[:, 2]], np.c_[mm[:, 0], v[:, 1], mm[:, 1]], np.c_[mm[:, 2], mm[:, 1], v[:, 2]], mm.copy()]
        par += [parent[i]]*4
        tri, parent = np.vstack(kids), np.concatenate(par)
        if group is not None:     # v3: without this the fans of the two long triangles of each floor quad end as slivers
            flip_delaunay(V, tri, floor[parent], group[parent])
    raise ValueError('floor refinement did not finish in %d rounds' % max_rounds)


def fade(V, tri, cap_vertices, aperture):
    """1 far from a cap, smoothstep(path distance over floor edges / local aperture) near one. Bellman-Ford rounds on the floor's edges."""
    w = np.ones(len(V))
    if not len(cap_vertices):
        return w
    e = np.unique(_edges(tri), axis=0)
    a, b = e[:, 0], e[:, 1]
    L = np.linalg.norm(V[a]-V[b], axis=1)
    d = np.full(len(V), np.inf)
    d[cap_vertices] = 0.
    reach = float(aperture.max())
    for _ in range(400):
        keep = (d[a] < reach) | (d[b] < reach)
        if not keep.any():
            break
        n = d.copy()
        np.minimum.at(n, b[keep], d[a[keep]]+L[keep])
        np.minimum.at(n, a[keep], d[b[keep]]+L[keep])
        if np.array_equal(n, d):
            break
        d = n
    x = np.clip(d/np.maximum(aperture, 1e-9), 0., 1.)
    return x*x*(3.-2.*x)


def creases(V, tri, degrees=CREASE_DEG):
    """Vertices of interior floor edges whose two faces differ by more than degrees"""
    c = np.cross(V[tri[:, 1]]-V[tri[:, 0]], V[tri[:, 2]]-V[tri[:, 0]])
    n = c/np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-30)
    e, nv = _edges(tri), len(V)
    key = e[:, 0]*nv+e[:, 1]
    order = np.argsort(key, kind='stable')
    same = np.flatnonzero(key[order[:-1]] == key[order[1:]])
    r1, r2 = order[same], order[same+1]
    bent = np.einsum('ij,ij->i', n[r1//3], n[r2//3]) < np.cos(np.radians(degrees))
    return np.unique(e[r1[bent]])


def limit_folds(V, tri, c, interior, delta, vn, max_rounds=40):
    """Halves the displacement of the vertices of reversed floor triangles until none is left; zero is the flat floor, which is not reversed. No parameter. Returns delta, vertices changed, reversed at the start."""
    delta = delta.copy()
    d = np.zeros(len(V))
    d[interior] = delta
    limited = np.zeros(len(V), bool)
    first = None
    for k in range(max_rounds):
        P = V+d[:, None]*vn
        c2 = np.cross(P[tri[:, 1]]-P[tri[:, 0]], P[tri[:, 2]]-P[tri[:, 0]])
        bad = np.einsum('ij,ij->i', c, c2) <= 0
        first = int(bad.sum()) if first is None else first
        if not bad.any():
            break
        hit = np.unique(tri[bad])
        limited[hit] = True
        d[hit] *= 0. if k >= max_rounds-12 else .5
    return d[interior], int(limited[interior].sum()), first


def _keys(c, extent):
    return (c[..., 0]*extent[1]+c[..., 1])*extent[2]+c[..., 2]


def nearest_within(P, S, r):
    out = np.full(len(P), float(r))
    if not len(P) or not len(S):
        return out
    origin = np.minimum(P.min(0), S.min(0))-2.*r
    extent = np.floor((np.maximum(P.max(0), S.max(0))-origin)/r).astype(np.int64)+3
    ks = _keys(np.floor((S-origin)/r).astype(np.int64), extent)
    order = np.argsort(ks, kind='stable')
    ks, S = ks[order], S[order]
    cp = np.floor((P-origin)/r).astype(np.int64)
    uk, first, inv = np.unique(_keys(cp, extent), return_index=True, return_inverse=True)
    inv = inv.ravel()
    members = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[members], np.arange(len(uk)+1))
    off = np.array(list(itertools.product((-1, 0, 1), repeat=3)), np.int64)
    nb = _keys(cp[first][:, None, :]+off, extent)
    L, R = np.searchsorted(ks, nb.ravel(), 'left').reshape(nb.shape), np.searchsorted(ks, nb.ravel(), 'right').reshape(nb.shape)
    for i in np.flatnonzero((R > L).any(1)):
        Q = np.concatenate([S[a:b] for a, b in zip(L[i], R[i]) if b > a])
        rows = members[bounds[i]:bounds[i+1]]
        for k in range(0, len(rows), 512):
            part = rows[k:k+512]
            out[part] = np.minimum(out[part], np.sqrt(((P[part, None, :]-Q[None, :, :])**2).sum(2).min(1)))
    return out


def host_samples(P, hv, hf, reach, spacing):
    """Points on the host triangles within about reach of P, at most spacing apart."""
    tri = hv[hf]
    near = ((tri.max(1) >= P.min(0)-reach) & (tri.min(1) <= P.max(0)+reach)).all(1)
    tri = tri[near]
    if not len(tri):
        return np.zeros((0, 3))
    origin = P.min(0)-3.*reach
    extent = np.floor((P.max(0)-origin)/reach).astype(np.int64)+7
    cells = np.unique(np.floor((P-origin)/reach).astype(np.int64), axis=0)
    off1 = np.array(list(itertools.product((-1, 0, 1), repeat=3)), np.int64)
    off2 = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), np.int64)
    within1 = np.unique(_keys(cells[:, None, :]+off1, extent))
    within2 = np.unique(_keys(cells[:, None, :]+off2, extent))

    def inside(X, keys):
        c = np.floor((X-origin)/reach).astype(np.int64)
        ok = ((c >= 0) & (c < extent)).all(1)
        ok[ok] = np.isin(_keys(c[ok], extent), keys)
        return ok
    longest = np.max([np.linalg.norm(tri[:, a]-tri[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    small = longest <= reach
    keep = ~small
    keep[small] = inside(tri[small].mean(1), within2)   # a point of a small triangle lies within reach of its centroid
    tri, longest = tri[keep], longest[keep]
    k = np.ceil(longest/spacing).astype(np.int64).clip(1)
    out = []
    for n in np.unique(k):
        i, j = np.meshgrid(np.arange(n+1), np.arange(n+1), indexing='ij')
        use = i+j <= n
        w = np.c_[i[use], j[use], n-i[use]-j[use]]/float(n)
        t = tri[k == n]
        step = max(1, 2000000//len(w))
        for a in range(0, len(t), step):
            pts = np.einsum('mi,tij->tmj', w, t[a:a+step]).reshape(-1, 3)
            out.append(pts[inside(pts, within1)])
    return np.vstack(out) if out else np.zeros((0, 3))


def clearance_limit(P, delta, hv, hf, d0, spacing):
    """delta limited to |delta| <= (d - d0) / 2, d the distance of P to the host surface. So a moved floor cannot reach the surface; within d0 it does not move. The slope 1/2 is the crease fade's. Returns delta, record."""
    need = d0+2.*np.abs(delta)+spacing     # farther than this the limit cannot act
    if not len(P):
        return delta, dict(d0_m=d0, spacing_m=spacing, host_samples=0, vertices_limited=0, vertices_zero=0)
    S = host_samples(P, hv, hf, float(need.max()), spacing)
    d = np.full(len(P), np.inf)
    r = d0+spacing
    todo = np.ones(len(P), bool)
    while todo.any():
        idx = np.flatnonzero(todo & (need <= r))
        if len(idx):
            d[idx] = nearest_within(P[idx], S, r)
            todo[idx] = False
        r *= 2.
    allowed = np.maximum(d-d0, 0.)/2.
    out = np.clip(delta, -allowed, allowed)
    return out, dict(d0_m=d0, spacing_m=spacing, host_samples=int(len(S)), vertices_limited=int((np.abs(out) < np.abs(delta)).sum()), vertices_zero=int((allowed == 0.).sum()))


def apply(solid, tables, relief, h, make_solid=None, host=None):
    """(solid, tables, record). h: the cavity's refine length; floor edges stay at or below 2 h, except edges shared with a wall or a cap. host: (vertices, faces) of the rock surface in this cavity's frame, or None."""
    if relief is None:
        return solid, tables, dict(applied=False, reason='no wall relief')
    if make_solid is None:
        import manifold3d as md

        def make_solid(X, T, run_index, run_ids, ids):
            return md.Manifold(md.Mesh64(np.ascontiguousarray(X), np.ascontiguousarray(T.astype(np.uint64)), run_index=run_index, run_original_id=run_ids, face_id=ids))
    mesh = solid.to_mesh64()
    X = np.array(np.asarray(mesh.vert_properties)[:, :3], dtype=np.float64)
    T = np.array(mesh.tri_verts, dtype=np.int64)
    fid = np.array(mesh.face_id, dtype=np.int64)
    oid = np.empty(len(T), np.int64)
    for a, b, o in zip(mesh.run_index[:-1], mesh.run_index[1:], mesh.run_original_id):
        oid[int(a)//3:int(b)//3] = int(o)
    role = np.empty(len(T), dtype=object)
    role[:] = ''
    for o in np.unique(oid):
        table = tables.get(int(o))
        if table is None:
            continue
        idx = np.flatnonzero(oid == o)
        role[idx] = [table[int(f)]['role'] if int(f) in table else '' for f in fid[idx]]
    is_floor = role == 'floor'
    record = dict(applied=True, max_edge_m=2.*h, original_floor_triangles=int(is_floor.sum()), cavity_triangles_before=int(len(T)), volume_before_m3=float(solid.volume()), genus_before=int(solid.genus()))
    if not is_floor.any():
        record.update(applied=False, reason='no floor faces')
        return solid, tables, record
    # the floor and every triangle touching it: the only ones refine can split
    on_floor = np.zeros(len(X), bool)
    on_floor[T[is_floor]] = True
    sub = np.flatnonzero(on_floor[T].any(1))
    tri0, flag0 = T[sub], is_floor[sub]
    base = np.full(len(X), np.nan)
    u0 = np.unique(tri0[flag0])
    base[u0] = _eval(relief, X[u0])
    V, base, all_tri, parent, rounds = refine(X, base, tri0, flag0, np.arange(len(sub)), 2.*h, group=oid[sub])
    flag = flag0[parent]
    tri = all_tri[flag]
    # boundary of the refined floor: the edges of one floor face
    e, nv = _edges(tri), len(V)
    uk, cnt = np.unique(e[:, 0]*nv+e[:, 1], return_counts=True)
    boundary = np.zeros(nv, bool)
    bk = uk[cnt == 1]
    boundary[bk//nv] = boundary[bk % nv] = True
    used = np.zeros(nv, bool)
    used[tri] = True
    interior = np.flatnonzero(used & ~boundary)
    # the roles across the boundary: an edge against a non wall face is a cap edge
    e2 = _edges(all_tri[~flag])
    k2 = e2[:, 0]*nv+e2[:, 1]
    is_wall = np.repeat(np.isin(role[sub][parent[~flag]].astype(str), WALLS), 3)
    order = np.argsort(k2)
    k2, is_wall = k2[order], is_wall[order]
    at = np.searchsorted(k2, bk)
    found = at < len(k2)
    found[found] &= k2[at[found]] == bk[found]
    if not found.all():
        raise ValueError('floor boundary edge without a neighbour: %d' % int((~found).sum()))
    cap_keys = bk[~is_wall[at]]
    cap_vertices = np.unique(np.r_[cap_keys//nv, cap_keys % nv]) if len(cap_keys) else np.zeros(0, np.int64)
    # local aperture: the longest edge of the original floor triangle
    t0 = X[tri0]
    span = np.max([np.linalg.norm(t0[:, a]-t0[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    aperture = np.zeros(nv)
    np.maximum.at(aperture, tri.ravel(), np.repeat(span[parent[flag]], 3))
    # v5: at a crease of the flat floor (path bend, fold) two planes meet and no single direction is normal to both: reversed triangles sat there. Creases stay fixed like caps, and on each plane the move is along its exact normal, so it cannot fold.
    crease_vertices = creases(V, tri)
    w = fade(V, tri, cap_vertices, aperture)
    # normals of the flat floor, area weighted
    c = np.cross(V[tri[:, 1]]-V[tri[:, 0]], V[tri[:, 2]]-V[tri[:, 0]])
    vn = np.zeros((nv, 3))
    for j in range(3):
        np.add.at(vn, tri[:, j], c)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    delta = (_eval(relief, V[interior])-base[interior])*w[interior]
    # the fade from a crease: twice the rms displacement, so its slope is about 1/2, the field's own median tilt (22 deg on 26120001). No new length.
    rms_before = float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.
    delta = delta*fade(V, tri, crease_vertices, np.full(nv, max(2.*rms_before, 1e-9)))[interior]
    near_host = None
    if host is not None:
        delta, near_host = clearance_limit(V[interior], delta, host[0], host[1], 4.*h, 2.*h)
    delta, limited, flipped_before = limit_folds(V, tri, c, interior, delta, vn)
    V = V.copy()
    V[interior] += delta[:, None]*vn[interior]
    c2 = np.cross(V[tri[:, 1]]-V[tri[:, 0]], V[tri[:, 2]]-V[tri[:, 0]])
    flipped = int((np.einsum('ij,ij->i', c, c2) <= 0).sum())
    if flipped:
        raise ValueError('floor relief left %d reversed floor triangles' % flipped)
    if os.environ.get('FLOOR_RELIEF_DEBUG'):
        np.savez(Path(os.environ['FLOOR_RELIEF_DEBUG'])/('surface_%d_%d.npz' % (int(oid[0]), len(tri))), V=V, tri=tri, c0=c, c1=c2, interior=interior, delta=delta, vn=vn, aperture=aperture, w=w, boundary=boundary, base=base)
    # new face ids and tables: a triangle that split gets one id per child, an untouched one keeps its id
    tables = {int(k): dict(v) for k, v in tables.items()}
    start = 1+max((int(k) for v in tables.values() for k in v), default=-1)
    split = np.bincount(parent, minlength=len(sub))[parent] > 1
    moved = np.zeros(nv, bool)
    moved[interior] = delta != 0.
    split &= ~(flag & ~moved[all_tri].any(1))     # v6: a floor triangle that did not move keeps its parent's id and merges back
    child_oid, child_fid = oid[sub][parent], fid[sub][parent].copy()
    new = np.arange(start, start+int(split.sum()))
    for f, o, p in zip(new.tolist(), child_oid[split].tolist(), child_fid[split].tolist()):
        tables[o][f] = tables[o][p]
    child_fid[split] = new
    far = np.ones(len(T), bool)
    far[sub] = False
    Tn, owner, ids = np.vstack([T[far], all_tri]), np.r_[oid[far], child_oid], np.r_[fid[far], child_fid]
    order = np.argsort(owner, kind='stable')
    Tn, owner, ids = Tn[order], owner[order], ids[order]
    run_ids, starts = np.unique(owner, return_index=True)
    out = make_solid(V, Tn, (np.r_[starts, len(Tn)]*3).astype(np.uint64), run_ids.astype(np.uint32), ids.astype(np.uint64))
    out = out.simplify(out.get_tolerance())     # v6: only triangles of one face id on one plane merge: the floor that did not move
    record.update(rounds=int(rounds), floor_triangles=int(len(tri)), near_triangles_split=int((np.bincount(parent, minlength=len(sub)) > 1)[~flag0].sum()), cavity_triangles_after=int(out.num_tri()), triangles_before_merge=int(len(Tn)), interior_vertices=int(len(interior)), cap_edges=int(len(cap_keys)),
                  wall_edges=int(len(bk)-len(cap_keys)), flipped_floor_triangles=flipped, reversed_before_limit=flipped_before, vertices_limited=limited, crease_vertices=int(len(crease_vertices)), rms_before_crease_fade_m=rms_before, near_host=near_host,
                  displacement_m=dict(rms=float(np.sqrt(np.mean(delta**2))), p99_abs=float(np.quantile(np.abs(delta), .99)), max_abs=float(np.abs(delta).max()), mean=float(delta.mean())) if len(delta) else None,
                  volume_after_m3=float(out.volume()), genus_after=int(out.genus()), status=str(out.status()))
    if out.status().name != 'NoError' or out.volume() <= 0:
        raise ValueError('floor relief made an invalid manifold: %s' % out.status())
    if int(out.genus()) != record['genus_before']:
        raise ValueError('floor relief changed the genus: %d to %d' % (record['genus_before'], int(out.genus())))
    return out, tables, record


_HOST = {}


def host_mesh(source_dir):
    """The rock surface, read once per process"""
    key = str(source_dir)
    if key not in _HOST:
        with np.load(Path(source_dir)/'source_surface.npz') as a:
            _HOST[key] = (np.array(a['vertices'], dtype=np.float64), np.array(a['faces'], dtype=np.int64))
    return _HOST[key]


def hook(trim, relief, h, source_dir=None, shift=None):
    """The callable for build_surface_cavity(end_trim=...): it runs before that function returns, so the label tool's rebuild keeps the solid with the relief. The joint-edge trim of fault ends, if any, runs first, as before. Returns (solid, tables, record) with the record's two parts apart."""
    def run(solid, tables):
        trimmed = None
        if trim is not None:
            solid, tables, trimmed = trim(solid, tables)
        host = None
        if source_dir is not None:
            hv, hf = host_mesh(source_dir)
            host = (hv if shift is None else hv-np.asarray(shift, float), hf)     # a moved group is built in its source frame: q = p - t
        solid, tables, record = apply(solid, tables, relief, h, host=host)
        return solid, tables, dict(trim=trimmed, floor_relief=record)
    return run
