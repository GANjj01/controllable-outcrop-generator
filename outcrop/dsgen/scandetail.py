"""Route B scan-detail transfer: the in-bed relief of a scanned rock face (donor, a detail band: the donor height minus
its 25 cm low-pass) is added to the patch surface *before* the residual blocks are cut, so v2 keeps the macro form,
the bedding / contact / joint planes and the block boundaries; every riser still lies exactly in its plane and the
floor-riser edges follow the relief (no edge is pinned, no ramp to a fixed boundary). Only a square footprint (side =
donor window, blend band inside its edge) changes.

Frames. Patch: centre c, outward normal n0, m = -n0 (B stretch direction). Footprint: centre at (u0, v0) in the patch
(e1, e2) frame, axes a1 along the v2 bedding trace on the patch plane, a2 = -m x a1 (up the face), (a1, a2, -m)
right-handed like the donor's (d1 along its strata, d2, n), so the transfer is a rotation (no mirror). Donor grid cell
(row, col): x1 = -W/2 + (col + .5) step, x2 = W/2 - (row + .5) step; texture U = (x1 + W/2) / W, V = (W/2 - x2) / W
(donor_lib conventions)."""
import math
from pathlib import Path

import numpy as np

from .relief import Sheet


def load_donor(path):
    d = np.load(Path(path) / 'donor.npz')
    return {k: d[k] for k in d.files}


def footprint_axes(frame, centre_uv, truth):
    c, n0, e1, e2 = (np.asarray(frame[k], float) for k in ('centre', 'normal', 'e1', 'e2'))
    m = -n0
    cf = c + centre_uv[0] * e1 + centre_uv[1] * e2
    st, body, _, _ = truth.query(cf[None])
    ok, nbs, _ = truth.fabric(cf[None], body, 'bedding')
    if not ok[0]:
        raise ValueError('no bedding fabric at the footprint centre')
    a1, a2 = trace_axes(nbs[0], m)
    return cf, a1, a2, m


def trace_axes(nb, m):
    """a1 along the trace of the bedding (normal nb) on the plane normal to m, a2 = -m x a1 pointing
    up: (a1, a2, -m) right-handed."""
    a1 = np.cross(nb, m)
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(-m, a1)
    if a2[2] < 0:
        a1, a2 = -a1, -a2
    return a1, a2


def window_weight(x1, x2, W, blend):
    """1 inside the footprint core, smoothstep to 0 over ``blend`` inside the footprint edge."""
    s = lambda t: np.where(t <= 0, 0., np.where(t >= 1, 1., t * t * (3 - 2 * t)))
    return s((W / 2 - np.abs(x1)) / blend) * s((W / 2 - np.abs(x2)) / blend)


def sample_grid(G, step, W, x1, x2):
    col = np.clip((x1 + W / 2) / step - .5, 0, G.shape[1] - 1.000001)
    row = np.clip((W / 2 - x2) / step - .5, 0, G.shape[0] - 1.000001)
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    return (G[r0, c0] * (1 - fc) + G[r0, c0 + 1] * fc) * (1 - fr) + (G[r0 + 1, c0] * (1 - fc) + G[r0 + 1, c0 + 1] * fc) * fr


def refine_where(sh, target_of):
    """Longest-edge bisection as Sheet.refine, with a per-edge target (target_of(midpoint 2D) -> m,
    inf = leave as is)."""
    prot_len = {k: float(np.linalg.norm(sh.P3[k[0]] - sh.P3[k[1]])) for k in sh.protected}
    elen = lambda k: float(np.linalg.norm(sh.P3[k[0]] - sh.P3[k[1]]))
    passes = 0
    while True:
        passes += 1
        longest = {}
        for fi, f in enumerate(sh.F):
            ks = [(min(x, y), max(x, y)) for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))]
            Ls = [elen(k) for k in ks]
            longest[fi] = (ks[int(np.argmax(Ls))], max([.75 * prot_len[k] for k in ks if k in prot_len] or [0.]))
        cand = []
        for k, fs in sh.E.items():
            if k in sh.protected or not all(longest[fi][0] == k for fi in fs):
                continue
            tgt = target_of(.5 * (sh.P2[k[0]] + sh.P2[k[1]]))
            if not np.isfinite(tgt):
                continue
            L = elen(k)
            if L > max([tgt] + [longest[fi][1] for fi in fs]):
                cand.append((L, k))
        if not cand:
            return passes
        cand.sort(reverse=True)
        used = set()
        for L, k in cand:
            fs = sh.E.get(k)
            if not fs or any(fi in used for fi in fs):
                continue
            used.update(fs)
            nb = len(sh.F)
            sh.split_edge(k[0], k[1], .5)
            used.update(range(nb, len(sh.F)))


def apply_top(V, F, frame, opts, truth):
    """Refine the surface around the footprint to ~target_m and displace it along -m (outward) by
    amplitude x w x donor detail. Returns faces (replacing ``faces``), parents of new vertices,
    moved original vertices, footprint frame and stats."""
    if opts.get('method', 'raster') == 'mesh':
        return apply_top_mesh(V, F, frame, opts, truth)
    cf, a1, a2, m = footprint_axes(frame, opts.get('centre_uv', (0., 0.)), truth)
    return refine_displace(V, F, cf, a1, a2, m, load_donor(opts['donor']), opts)


def refine_displace(V, F, cf, a1, a2, m, donor, opts):
    """apply_top for a given footprint frame and donor (dict: detail grid, step, W)."""
    W, step = float(donor['W']), float(donor['step'])
    blend, margin = float(opts.get('blend_m', .15)), float(opts.get('margin_m', .1))
    target = float(opts.get('target_m', .01))
    amp = float(opts.get('amplitude', 1.))
    to_fp = lambda X: np.c_[(X - cf) @ a1, (X - cf) @ a2]
    # region: faces with a vertex inside the footprint + margin
    Q = to_fp(V)
    near_v = (np.abs(Q) <= W / 2 + margin).all(1)
    faces = np.nonzero(near_v[F].any(1))[0]
    vids = np.unique(F[faces])
    loc = {int(v): i for i, v in enumerate(vids)}
    P3 = V[vids]
    P2 = to_fp(P3)
    Fl = np.vectorize(loc.get)(F[faces])
    a2d = (P2[Fl[:, 1], 0] - P2[Fl[:, 0], 0]) * (P2[Fl[:, 2], 1] - P2[Fl[:, 0], 1]) - \
          (P2[Fl[:, 1], 1] - P2[Fl[:, 0], 1]) * (P2[Fl[:, 2], 0] - P2[Fl[:, 0], 0])
    flipped = False
    if (a2d < 0).all():
        Fl, flipped = Fl[:, ::-1], True
    elif not (a2d > 0).all():
        raise RuntimeError('detail region folds in projection on the footprint plane')
    ecount = {}
    for f in Fl:
        for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(x, y), max(x, y))
            ecount[k] = ecount.get(k, 0) + 1
    sh = Sheet(P2, P3, Fl, [k for k, cnt in ecount.items() if cnt == 1])
    inner = W / 2 + .5 * margin
    passes = refine_where(sh, lambda p: target if (abs(p[0]) <= inner and abs(p[1]) <= inner) else np.inf)
    Pf3, Pf2, Ff = np.array(sh.P3), np.array(sh.P2), np.array(sh.F)
    # displacement along -m (outward): x1, x2 from the 3D position projected on the footprint plane
    X = to_fp(Pf3)
    w = window_weight(X[:, 0], X[:, 1], W, blend)
    inside = (np.abs(X) < W / 2).all(1)
    h = np.zeros(len(Pf3))
    h[inside] = sample_grid(donor['detail'], step, W, X[inside, 0], X[inside, 1])
    d = amp * w * h
    Pf3 = Pf3 - d[:, None] * m[None]
    nV, nloc = len(V), len(Pf3)
    gidx = np.empty(nloc, np.int64)
    gidx[:len(vids)] = vids
    gidx[len(vids):] = nV + np.arange(nloc - len(vids))
    parents = []
    for i in range(len(vids), nloc):
        idx, wts = sh.parents[i]
        parents.append((tuple(int(gidx[j]) for j in idx), tuple(float(x) for x in wts)))
    a2n = (Pf2[Ff[:, 1], 0] - Pf2[Ff[:, 0], 0]) * (Pf2[Ff[:, 2], 1] - Pf2[Ff[:, 0], 1]) - \
          (Pf2[Ff[:, 1], 1] - Pf2[Ff[:, 0], 1]) * (Pf2[Ff[:, 2], 0] - Pf2[Ff[:, 0], 0])
    out_f = gidx[Ff][:, ::-1] if flipped else gidx[Ff]
    eL = np.linalg.norm(Pf3[Ff] - Pf3[Ff[:, [1, 2, 0]]], axis=2)
    core = (np.abs(to_fp(Pf3[Ff].mean(1))) < W / 2 - blend).all(1)
    stats = {'footprint_centre': cf.tolist(), 'a1': a1.tolist(), 'a2': a2.tolist(), 'W_m': W, 'blend_m': blend, 'amplitude': amp,
             'region_faces_in': int(len(faces)), 'region_faces_out': int(len(Ff)), 'vertices_added': int(nloc - len(vids)),
             'refine_passes': passes, 'core_edge_mm_p50_p99_max': [float(x) for x in np.percentile(eL[core] * 1e3, [50, 99, 100])] if core.any() else None,
             'disp_mm_p0_p1_p50_p99_p100': [float(x) for x in np.percentile(d[inside] * 1e3, [0, 1, 50, 99, 100])] if inside.any() else None,
             'projected_orientation_consistent': bool((a2n > 0).all()), 'min_projected_area_m2': float(np.abs(a2n).min() / 2)}
    return {'faces_replaced': faces, 'faces': out_f, 'parent_face': faces[np.array(sh.parent_face)],
            'new_positions': Pf3[len(vids):], 'moved_original': {int(vids[i]): Pf3[i] for i in range(len(vids)) if d[i] != 0},
            'parents': parents, 'frame': (cf, a1, a2, m, W, blend), 'stats': stats,
            'local': {'gidx': gidx, 'P2': Pf2, 'disp': d, 'F': Ff}}


def xfer_attributes(V, F, riser_f, nout_f, rock_faces, frame):
    """Per-vertex donor texture coordinates and blend weight, and the faces that use them.
    Floors / unspalled faces: projection on the footprint plane (U along a1, V along -a2). Risers:
    their own plane (plane_basis axes, (u, v, n_out) right-handed), offset so each plane's part sits
    around the texture centre. Faces: rock faces with any vertex at w > 0."""
    from .blockface import plane_basis
    cf, a1, a2, m, W, blend = frame
    X = np.c_[(V - cf) @ a1, (V - cf) @ a2]
    w = window_weight(X[:, 0], X[:, 1], W, blend)
    uv = np.c_[(X[:, 0] + W / 2) / W, (W / 2 - X[:, 1]) / W]
    rs = np.flatnonzero(riser_f)
    groups = {}
    for fi in rs:
        key = tuple(np.round(nout_f[fi], 4)) + (round(float(V[F[fi, 0]] @ nout_f[fi]), 3),)
        groups.setdefault(key, []).append(fi)
    riser_v = np.zeros(len(V), bool)
    for key, fis in groups.items():
        n = np.asarray(key[:3], float)
        u, v = plane_basis(n / np.linalg.norm(n))
        vs = np.unique(F[fis])
        P = V[vs]
        pu, pv = P @ u, P @ v
        uv[vs] = np.c_[.5 + (pu - .5 * (pu.min() + pu.max())) / W, .5 - (pv - .5 * (pv.min() + pv.max())) / W]
        riser_v[vs] = True
    faces = rock_faces & (w[F] > 0).any(1)
    return {'uv': uv, 'w': w, 'faces': faces, 'riser_planes': len(groups), 'riser_vertices': int(riser_v.sum())}


def collapse_short(P2, F, n_fixed, min_len, rounds=50, min_height=None):
    """Collapse projected edges shorter than min_len (and, with min_height, the edge from the middle
    vertex of a flat triangle to the nearer end of its longest edge): a vertex >= n_fixed (inserted) is
    moved onto the other end -- either end may move; faces using both ends are removed; a collapse is
    skipped if any other face of the moving vertex would get a non-positive (CCW) area. Returns faces
    and the number collapsed."""
    F = np.asarray(F).copy()
    total = 0
    for _ in range(rounds):
        e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
        L = np.linalg.norm(P2[e[:, 0]] - P2[e[:, 1]], axis=1)
        short = np.nonzero(L < min_len)[0]
        if min_height is not None:
            T = P2[F]
            A2 = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 1, 1] - T[:, 0, 1]) * (T[:, 2, 0] - T[:, 0, 0])
            EL = np.linalg.norm(T - T[:, [1, 2, 0]], axis=2)
            flat = np.nonzero(A2 / np.maximum(EL.max(1), 1e-30) < min_height)[0]
            extra = []
            for fi in flat:
                j = int(np.argmax(EL[fi]))
                a_, b_, c_ = F[fi, j], F[fi, (j + 1) % 3], F[fi, (j + 2) % 3]
                t_ = b_ if np.linalg.norm(P2[c_] - P2[b_]) < np.linalg.norm(P2[c_] - P2[a_]) else a_
                extra.append((c_, t_))
            if extra:
                e = np.r_[e, np.array(extra)]
                L = np.r_[L, np.zeros(len(extra))]
                short = np.r_[short, np.arange(len(L) - len(extra), len(L))]
        if not len(short):
            break
        vf = {}
        for fi, f in enumerate(F):
            for v in f:
                vf.setdefault(int(v), []).append(fi)
        locked = set()
        alive = np.ones(len(F), bool)
        done = 0
        for k in short[np.argsort(L[short])]:
            a, b = int(e[k, 0]), int(e[k, 1])
            if a in locked or b in locked:
                continue
            choice = None
            for mv, to in ((a, b), (b, a)):
                if mv < n_fixed:
                    continue
                fs = [fi for fi in vf.get(mv, []) if alive[fi]]
                ok = True
                for fi in fs:
                    f = F[fi]
                    if to in f:
                        continue
                    q = np.array([P2[to] if v == mv else P2[v] for v in f])
                    if (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0]) <= 0:
                        ok = False
                        break
                if ok:
                    choice = (mv, to, fs)
                    break
            if choice is None:
                continue
            mv, to, fs = choice
            for fi in fs:
                if to in F[fi]:
                    alive[fi] = False
                else:
                    F[fi][F[fi] == mv] = to
            for v in set(int(x) for fi in fs for x in F[fi]) | {mv, to}:
                locked.add(v)
            done += 1
        F = F[alive]
        total += done
        if not done:
            break
    return F, total


def flip_flat(P2, F, min_height, rounds=50):
    """Flat triangles (height over the longest edge < min_height: three almost collinear vertices) get
    their longest edge flipped with the neighbouring triangle when both new triangles are CCW and not
    flat. Returns faces and the number of flips."""
    F = np.asarray(F).copy()
    area2 = lambda q: (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0])
    total = 0
    for _ in range(rounds):
        T = P2[F]
        A2 = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 1, 1] - T[:, 0, 1]) * (T[:, 2, 0] - T[:, 0, 0])
        Lm = np.linalg.norm(T - T[:, [1, 2, 0]], axis=2).max(1)
        flat = np.nonzero(A2 / np.maximum(Lm, 1e-30) < min_height)[0]
        if not len(flat):
            break
        ed = {}
        for fi, f in enumerate(F):
            for j in range(3):
                ed.setdefault((min(f[j], f[(j + 1) % 3]), max(f[j], f[(j + 1) % 3])), []).append(fi)
        used = set()
        done = 0
        for fi in flat:
            if fi in used:
                continue
            f = F[fi]
            L = [np.linalg.norm(P2[f[j]] - P2[f[(j + 1) % 3]]) for j in range(3)]
            j = int(np.argmax(L))
            a, b, c = f[j], f[(j + 1) % 3], f[(j + 2) % 3]
            nb = [g for g in ed.get((min(a, b), max(a, b)), []) if g != fi]
            if len(nb) != 1 or nb[0] in used:
                continue
            g = nb[0]
            d = [v for v in F[g] if v not in (a, b)][0]
            t1, t2 = np.array([c, a, d]), np.array([c, d, b])
            q1, q2 = P2[t1], P2[t2]
            h1 = area2(q1) / max(np.linalg.norm(q1 - q1[[1, 2, 0]], axis=1).max(), 1e-30)
            h2 = area2(q2) / max(np.linalg.norm(q2 - q2[[1, 2, 0]], axis=1).max(), 1e-30)
            if h1 <= min_height or h2 <= min_height:
                continue
            F[fi], F[g] = t1, t2
            used.update((fi, g))
            done += 1
        total += done
        if not done:
            break
    return F, total


def flip_steep(P2, X3, F, m, cos_min, min_height, segs=(), snap=1e-5, rounds=50):
    """Knife facets: a sliver whose apex lies a few um off its longest edge gets the exact height at all three corners, but
    the chord of the long edge misses the curved exact surface by up to the refinement tolerance -- the facet then
    stands almost along m (steeper than any donor front face) and the surface carries um-thin fins and slots. Such a
    face (|n.m| < cos_min on the positions X3) has its longest edge flipped with the neighbouring triangle when both
    new triangles are CCW, not flat (min_height) and less steep than the steeper of the pair. Vertices and heights do
    not change. Returns faces and stats: flips, flipped edges lying on an inserted constraint segment (within snap)
    and the largest offset of such a constraint from its new path (the apex distance), steep faces before and after."""
    F = np.asarray(F).copy()

    def cosv(T):
        n = np.cross(X3[T[..., 1]] - X3[T[..., 0]], X3[T[..., 2]] - X3[T[..., 0]])
        return np.abs(n @ m) / np.maximum(np.linalg.norm(n, axis=-1), 1e-300)

    def hgt(t):
        q = P2[t]
        a2 = (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0])
        return a2 / max(np.linalg.norm(q - q[[1, 2, 0]], axis=1).max(), 1e-30)
    SA = np.array([a for a, _ in segs]).reshape(-1, 2)
    SB = np.array([b for _, b in segs]).reshape(-1, 2)

    def on_constraint(a, b):
        if not len(SA):
            return False
        ab = SB - SA
        den = np.maximum((ab * ab).sum(1), 1e-30)
        ok = np.ones(len(SA), bool)
        for p in (P2[a], P2[b]):
            t = np.clip(((p - SA) * ab).sum(1) / den, 0, 1)
            ok &= np.linalg.norm(p - (SA + t[:, None] * ab), axis=1) <= snap
        return bool(ok.any())
    before = int((cosv(F) < cos_min).sum())
    flips = on_c = 0
    off_max = 0.
    todo = np.nonzero(cosv(F) < cos_min)[0]
    for _ in range(rounds):
        ed = {}
        for fi, f in enumerate(F):
            for j in range(3):
                ed.setdefault((min(f[j], f[(j + 1) % 3]), max(f[j], f[(j + 1) % 3])), []).append(fi)
        used, done = set(), 0
        for fi in todo:
            if fi in used or cosv(F[fi]) >= cos_min:
                continue
            f = F[fi]
            L = [np.linalg.norm(P2[f[j]] - P2[f[(j + 1) % 3]]) for j in range(3)]
            j = int(np.argmax(L))
            a, b, c = f[j], f[(j + 1) % 3], f[(j + 2) % 3]
            nb = [g for g in ed.get((min(a, b), max(a, b)), []) if g != fi]
            if len(nb) != 1 or nb[0] in used:
                continue
            g = nb[0]
            d = [v for v in F[g] if v not in (a, b)][0]
            t1, t2 = np.array([c, a, d]), np.array([c, d, b])
            if hgt(t1) <= min_height or hgt(t2) <= min_height:
                continue
            if min(cosv(t1), cosv(t2)) <= min(cosv(F[fi]), cosv(F[g])):
                continue
            if on_constraint(a, b):
                on_c += 1
                ab = P2[b] - P2[a]
                off_max = max(off_max, min(abs(ab[0] * (P2[v] - P2[a])[1] - ab[1] * (P2[v] - P2[a])[0]) / np.linalg.norm(ab) for v in (c, d)))
            F[fi], F[g] = t1, t2
            used.update((fi, g))
            done += 1
        flips += done
        if not done:
            break
        todo = np.array(sorted(used | {int(x) for x in todo}))
    return F, {'flips': flips, 'flipped_on_constraint': on_c, 'constraint_offset_max_m': off_max,
               'steep_before': before, 'steep_after': int((cosv(F) < cos_min).sum()), 'cos_min': cos_min}


def insert_plane_lines(V, F, frame, cut, cell_of, depth_of, taper, region_of, snap=1e-5):
    """Before the B blocks are cut: insert into the patch surface, inside the detail region, the lines where every cutting
    plane meets the floor of each cell it bounds -- floor point x = P + R t m (P on the surface, R the cell depth, t
    the taper), so n.P + R t (n.m) = d is a level line of a function linear on each triangle. The planes then cut
    every prism exactly through existing vertices and edges: no sliver pieces, no sub-mm seams between pieces. A line
    piece is kept where the cell on one side of the plane at its midpoint has depth R. ``region_of``: callable(face
    centroids) -> bool. Returns (V, F, parents of new vertices, stats)."""
    c, n0 = (np.asarray(frame[k], float) for k in ('centre', 'normal'))
    m = -n0
    e1 = np.cross(m, [0., 0., 1.]) if abs(m[2]) < .9 else np.cross(m, [1., 0., 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(m, e1)
    faces = np.nonzero(region_of(V[F].mean(1)))[0]
    vids = np.unique(F[faces])
    loc = {int(v): i for i, v in enumerate(vids)}
    P3 = V[vids]
    P2 = np.c_[(P3 - c) @ e1, (P3 - c) @ e2]
    Fl = np.vectorize(loc.get)(F[faces])
    a2d = (P2[Fl[:, 1], 0] - P2[Fl[:, 0], 0]) * (P2[Fl[:, 2], 1] - P2[Fl[:, 0], 1]) - \
          (P2[Fl[:, 1], 1] - P2[Fl[:, 0], 1]) * (P2[Fl[:, 2], 0] - P2[Fl[:, 0], 0])
    flipped = False
    if (a2d < 0).all():
        Fl, flipped = Fl[:, ::-1], True
    elif not (a2d > 0).all():
        raise RuntimeError('plane-line region folds in projection along m')
    ecount = {}
    for f in Fl:
        for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(x, y), max(x, y))
            ecount[k] = ecount.get(k, 0) + 1
    sh = GridSheet(P2, P3, Fl, [k for k, cnt in ecount.items() if cnt == 1], cell=.005)
    tv = taper[vids]
    T3, Tt = P3[Fl], tv[Fl]
    depths = {}
    segs = []
    for fam, lev, nn, d in cut:
        nm = float(nn @ m)
        g0 = T3 @ nn - d                       # (faces, 3): plane value of the surface vertices
        # candidate depths: cells touching this plane inside the region (probe both sides at many points)
        Rmax = .3                              # >= the B depth (0.254 m)
        near = np.nonzero((np.minimum(g0, g0 + Rmax * Tt * nm).min(1) <= 0) & (np.maximum(g0, g0 + Rmax * Tt * nm).max(1) >= 0))[0]
        if not len(near):
            continue
        cand_R = set()
        for fi in near[:: max(1, len(near) // 400)]:
            cen = T3[fi].mean(0)
            for R in np.linspace(0, Rmax, 9):
                q = cen + R * Tt[fi].mean() * m
                q = q - ((q @ nn - d) / 1.) * nn
                for s_ in (1e-6, -1e-6):
                    cand_R.add(float(depth_of(cell_of(q + s_ * nn))))
        for R in sorted(cand_R):
            g = g0[near] + R * Tt[near] * nm
            sgn = np.sign(g)
            cross = (sgn.min(1) < 0) & (sgn.max(1) > 0)
            for k in np.nonzero(cross)[0]:
                fi = near[k]
                gv, f = g[k], Fl[fi]
                pts = []
                for i in range(3):
                    a_, b_ = i, (i + 1) % 3
                    if (gv[a_] < 0) != (gv[b_] < 0):
                        s = gv[a_] / (gv[a_] - gv[b_])
                        pts.append((P2[f[a_]] + s * (P2[f[b_]] - P2[f[a_]]), P3[f[a_]] + s * (P3[f[b_]] - P3[f[a_]]),
                                    tv[f[a_]] + s * (tv[f[b_]] - tv[f[a_]])))
                if len(pts) != 2:
                    continue
                mid3 = .5 * (pts[0][1] + pts[1][1])
                tm = .5 * (pts[0][2] + pts[1][2])
                q = mid3 + R * tm * m
                if any(float(depth_of(cell_of(q + s_ * nn))) == R for s_ in (1e-6, -1e-6)):
                    segs.append((pts[0][0], pts[1][0]))
                    depths[(fam, lev)] = depths.get((fam, lev), 0) + 1
    n_ok = n_fail = 0
    for a, b in segs:
        va = sh.add_point(a, snap)
        vb = sh.add_point(b, snap) if va is not None else None
        if va is not None and vb is not None and (va == vb or sh.insert_segment(va, vb, tol=snap)):
            n_ok += 1
        else:
            n_fail += 1
    Pf3, Ff = np.array(sh.P3), np.array(sh.F)
    nV, nloc = len(V), len(Pf3)
    gidx = np.empty(nloc, np.int64)
    gidx[:len(vids)] = vids
    gidx[len(vids):] = nV + np.arange(nloc - len(vids))
    parents = []
    for i in range(len(vids), nloc):
        idx, wts = sh.parents[i]
        parents.append((tuple(int(gidx[j]) for j in idx), tuple(float(x) for x in wts)))
    out_f = gidx[Ff][:, ::-1] if flipped else gidx[Ff]
    keep = np.ones(len(F), bool)
    keep[faces] = False
    eL = np.linalg.norm(np.array(sh.P2)[Ff] - np.array(sh.P2)[Ff][:, [1, 2, 0]], axis=2)
    stats = {'region_faces_in': int(len(faces)), 'region_faces_out': int(len(Ff)), 'segments': len(segs), 'inserted': n_ok,
             'failed': n_fail, 'planes_with_lines': len(depths), 'vertices_added': int(nloc - len(vids)),
             'min_projected_edge_m': float(eL.min())}
    return np.concatenate([V, Pf3[len(vids):]]), np.concatenate([F[keep], out_f]), parents, stats


class GridSheet(Sheet):
    """Sheet with a spatial hash of its faces (2D bounding boxes, cells of ``cell`` m), so a point is
    located among the faces of its cell instead of all faces."""

    def __init__(self, P2, P3, faces, protected, cell=.01):
        self.cell = cell
        self.fgrid = {}
        super().__init__(P2, P3, faces, protected)

    def _cells(self, fi):
        q = np.array([self.P2[v] for v in self.F[fi]])
        lo, hi = np.floor(q.min(0) / self.cell).astype(int), np.floor(q.max(0) / self.cell).astype(int)
        return [(i, j) for i in range(lo[0], hi[0] + 1) for j in range(lo[1], hi[1] + 1)]

    def _add_face_links(self, fi):
        super()._add_face_links(fi)
        for k in self._cells(fi):
            self.fgrid.setdefault(k, set()).add(fi)

    def _drop_face_links(self, fi):
        for k in self._cells(fi):
            self.fgrid[k].discard(fi)
        super()._drop_face_links(fi)

    def add_point(self, p, tol):
        p = np.asarray(p, float)
        return self.insert_point(p, sorted(self.fgrid.get((int(np.floor(p[0] / self.cell)), int(np.floor(p[1] / self.cell))), ())), tol=tol)


def apply_top_mesh(V, F, frame, opts, truth):
    """Method 'mesh': as apply_top, but the target is the donor's own surface (scanmesh: front-triangle envelope + lip
    walls). Every donor front edge inside the footprint and every wall-foot crease is inserted as a mesh edge (snap
    ``snap_m``) and each vertex's height is evaluated on the donor triangles; detail = height - 25 cm low-pass;
    displacement along -m by amplitude x w x detail. No height raster is sampled for the detail (the low-pass is
    smooth)."""
    from . import scanmesh
    cf, a1, a2, m = footprint_axes(frame, opts.get('centre_uv', (0., 0.)), truth)
    don = scanmesh.load(Path(opts['donor_mesh']) / 'mesh_donor.npz')
    return refine_displace_mesh(V, F, cf, a1, a2, m, don, opts)


def refine_displace_mesh(V, F, cf, a1, a2, m, don, opts):
    """Opt-in keys of the whole-wall scene, all absent for the B sample: ``face_ok`` (bool per face of F): only those faces
    form the region; ``vertex_weight`` (per vertex of V, in [0, 1]): multiplies the footprint window weight (region
    taper), new vertices take it from their parents; ``tol_of`` (callable, footprint 2-D points -> linear-error limit
    m) and ``min_split_of`` (same, -> shortest edge to split) replace max_lin_err_m / min_split_m per face (evaluated
    at the face centroid); ``constrain_of`` (callable, 2-D points -> bool): a donor edge / crease is inserted only
    when both of its ends are True. ``weight_error``: the refinement error is weighted by window weight x vertex
    weight. ``don`` may be a scanmesh.DonorWindow (atlas window)."""
    from . import scanmesh
    W = don.W
    blend, margin = float(opts.get('blend_m', .15)), float(opts.get('margin_m', .1))
    snap = float(opts.get('snap_m', 1e-5))
    amp = float(opts.get('amplitude', 1.))
    to_fp = lambda X: np.c_[(X - cf) @ a1, (X - cf) @ a2]
    Q = to_fp(V)
    near_v = (np.abs(Q) <= W / 2 + margin).all(1)
    sel_f = near_v[F].any(1)
    if opts.get('face_ok') is not None:
        sel_f &= np.asarray(opts['face_ok'], bool)
    faces = np.nonzero(sel_f)[0]
    vids = np.unique(F[faces])
    loc = {int(v): i for i, v in enumerate(vids)}
    P3 = V[vids]
    P2 = to_fp(P3)
    Fl = np.vectorize(loc.get)(F[faces])
    a2d = (P2[Fl[:, 1], 0] - P2[Fl[:, 0], 0]) * (P2[Fl[:, 2], 1] - P2[Fl[:, 0], 1]) - \
          (P2[Fl[:, 1], 1] - P2[Fl[:, 0], 1]) * (P2[Fl[:, 2], 0] - P2[Fl[:, 0], 0])
    flipped = False
    if (a2d < 0).all():
        Fl, flipped = Fl[:, ::-1], True
    elif not (a2d > 0).all():
        raise RuntimeError('detail region folds in projection on the footprint plane')
    ecount = {}
    for f in Fl:
        for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(x, y), max(x, y))
            ecount[k] = ecount.get(k, 0) + 1
    sh = GridSheet(P2, P3, Fl, [k for k, cnt in ecount.items() if cnt == 1])
    segs = []
    for a, b in don.edges:
        c_ = scanmesh.clip_segment(np.asarray(a), np.asarray(b), W / 2)
        if c_ is not None:
            segs.append(c_)
    n_edges = len(segs)
    for pl in don.creases:
        for a, b in zip(pl[:-1], pl[1:]):
            c_ = scanmesh.clip_segment(a, b, W / 2)
            if c_ is not None:
                segs.append(c_)
    if opts.get('constrain_of') is not None and segs:
        # whole-wall scene: constraints only where the imaging-scale tolerance is fine (both ends)
        A_ = np.array([a for a, _ in segs])
        B_ = np.array([b for _, b in segs])
        keep_ = np.asarray(opts['constrain_of'](A_), bool) & np.asarray(opts['constrain_of'](B_), bool)
        n_edges = int(keep_[:n_edges].sum())
        segs = [sg for sg, k_ in zip(segs, keep_) if k_]
    n_ok = n_fail = 0
    failed = []
    for a, b in segs:
        va = sh.add_point(a, snap)
        vb = sh.add_point(b, snap) if va is not None else None
        if va is not None and vb is not None and (va == vb or sh.insert_segment(va, vb, tol=snap)):
            n_ok += 1
        else:
            n_fail += 1
            failed.append((a, b))
    # adaptive refinement against the exact donor surface: where the height at an edge midpoint differs
    # from the linear interpolation of its ends by more than max_lin_err_m (cone caps at lip ends, crease
    # pieces not inserted), the edge is bisected (both faces: conforming), down to min_split_m
    err_max, min_split = float(opts.get('max_lin_err_m', 2e-4)), float(opts.get('min_split_m', 1e-3))
    tol_of, msplit_of = opts.get('tol_of'), opts.get('min_split_of')
    hcache = {}
    # exact heights at each face's centroid and edge midpoints, kept while the face (ordered vertex triple) is
    fcache = {}

    def hexact(ids):
        need = [i for i in ids if i not in hcache]
        if need:
            Pn = np.array([sh.P2[i] for i in need])
            ins = (np.abs(Pn) < W / 2).all(1)
            hv_ = np.zeros(len(need))
            if ins.any():
                hv_[ins] = don.height(Pn[ins])[0]
            for i, v in zip(need, hv_):
                hcache[i] = v if np.isfinite(v) else 0.
        return np.array([hcache[i] for i in ids])
    vwl = []

    def vwf(tri):
        # vertex weight (opts vertex_weight) of sheet vertices, through the parents of new vertices (which always have
        # lower indices: filled in index order); mean per face
        if opts.get('vertex_weight') is None:
            return np.ones(len(tri))
        if not vwl:
            vwl.extend(np.asarray(opts['vertex_weight'], float)[vids].tolist())
        need = int(np.max(tri)) + 1 if len(tri) else 0
        for i in range(len(vwl), need):
            idx_, wts_ = sh.parents[i]
            vwl.append(sum(wk * vwl[ik] for ik, wk in zip(idx_, wts_)))
        a_ = np.asarray(vwl)
        return a_[np.asarray(tri)].mean(1)
    n_split, passes_ad = 0, 0
    elen = lambda k: float(np.linalg.norm(sh.P2[k[0]] - sh.P2[k[1]]))
    marked = set()
    for passes_ad in range(1, 61):
        # faces whose linear interpolation misses the exact surface at an edge midpoint or the centroid
        Fa = np.array(sh.F)
        Pa = np.array(sh.P2)
        T_ = Pa[Fa]
        cen = T_.mean(1)
        mids = .5 * (T_ + T_[:, [1, 2, 0]])
        EL = np.linalg.norm(T_ - T_[:, [1, 2, 0]], axis=2)
        ms_f = min_split if msplit_of is None else np.asarray(msplit_of(cen), float)
        cand = np.nonzero((np.abs(cen) < W / 2).all(1) & (EL.max(1) > ms_f))[0]
        hv3 = hexact(list(Fa[cand].reshape(-1))).reshape(-1, 3)
        keys_ = [tuple(int(v) for v in Fa[f]) for f in cand]
        new_ = [j for j, k_ in enumerate(keys_) if k_ not in fcache]
        if new_:
            nj = np.array(new_)
            Q_ = np.concatenate([cen[cand[nj]], mids[cand[nj]].reshape(-1, 2)])
            okq = (np.abs(Q_) < W / 2).all(1)
            hq = np.zeros(len(Q_))
            hq[okq] = don.height(Q_[okq])[0]
            hq = np.where(np.isfinite(hq), hq, 0.)
            hc_n, hm_n = hq[:len(nj)], hq[len(nj):].reshape(-1, 3)
            for j, a_, b_ in zip(new_, hc_n, hm_n):
                fcache[keys_[j]] = (a_, b_)
        hc = np.array([fcache[k_][0] for k_ in keys_])
        hmid = np.array([fcache[k_][1] for k_ in keys_]).reshape(-1, 3)
        err = np.maximum(np.abs(hc - hv3.mean(1)), np.abs(hmid - .5 * (hv3 + hv3[:, [1, 2, 0]])).max(1))
        lim = err_max if tol_of is None else np.asarray(tol_of(cen[cand]), float)
        if opts.get('weight_error') and len(cand):
            # whole-wall scene: the displacement is w x detail, so its interpolation error is w x that of the detail
            # (w = window weight x vertex weight, at the centroid); faces of the blend / taper bands are not refined to
            # the full-amplitude tolerance
            err = err * window_weight(cen[cand, 0], cen[cand, 1], W, blend) * vwf(Fa[cand])
        marked = set(int(f) for f in cand[err > lim]) | {f for f in marked if f < len(sh.F)}
        if not marked:
            break
        # Rivara: split the longest edge of a marked face when it is the longest edge of every face
        # sharing it; otherwise mark the neighbour (its longest edge goes first)
        split_now, nxt = [], set()
        for fi in marked:
            f = sh.F[fi]
            ks = [(min(x, y), max(x, y)) for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))]
            k = max(ks, key=elen)
            ms_ = min_split if msplit_of is None else (float(ms_f[fi]) if fi < len(ms_f) else
                                                       float(msplit_of(np.mean([sh.P2[v] for v in f], axis=0)[None])[0]))
            if k in sh.protected or elen(k) <= ms_:
                continue
            fs = sh.E.get(k, [])
            terminal = True
            for g in fs:
                fg = sh.F[g]
                kg = max([(min(x, y), max(x, y)) for x, y in ((fg[0], fg[1]), (fg[1], fg[2]), (fg[2], fg[0]))], key=elen)
                if kg != k:
                    terminal = False
                    nxt.add(g)
            if terminal:
                split_now.append(k)
            else:
                nxt.add(fi)
        used = set()
        for k in sorted(set(split_now), key=elen, reverse=True):
            fs = sh.E.get(k)
            if not fs or any(g in used for g in fs):
                continue
            used.update(fs)
            nb = len(sh.F)
            sh.split_edge(k[0], k[1], .5)
            used.update(range(nb, len(sh.F)))
            n_split += 1
        marked = {f for f in nxt if f not in used}
        if not split_now and not nxt:
            break
    Pf3, Pf2, Ff = np.array(sh.P3), np.array(sh.P2), np.array(sh.F)
    nloc = len(Pf3)
    # near-collinear constraints cross at almost the same point: collapse projected edges shorter than
    # min_edge_m by moving a new vertex onto its neighbour (never an original vertex), skipping any
    # collapse that would invert a triangle; heights are evaluated afterwards at the final positions
    min_edge, min_h = float(opts.get('min_edge_m', 1e-4)), float(opts.get('min_height_m', 1e-6))
    n_col = n_flip = 0
    for _ in range(10):
        Ff, f1 = flip_flat(Pf2, Ff, min_h)
        Ff, c1 = collapse_short(Pf2, Ff, len(vids), min_edge, min_height=min_h)
        n_col, n_flip = n_col + c1, n_flip + f1
        if not c1 and not f1:
            break
    X = to_fp(Pf3)
    inside = (np.abs(X) < W / 2).all(1)
    h = np.zeros(nloc)
    hv, own, he, hw = don.height(X[inside])
    undefined = ~np.isfinite(hv)
    lp = sample_grid(don.lowpass, don.lp_step, W, X[inside, 0], X[inside, 1])
    h[inside] = np.where(undefined, 0., hv - lp)
    n_undef = int(undefined.sum())
    if n_undef:
        # vertices without donor surface: harmonic fill from their neighbours
        und = np.nonzero(inside)[0][undefined]
        nb = {int(v): set() for v in und}
        for f in Ff:
            for x in f:
                if int(x) in nb:
                    nb[int(x)].update(int(y) for y in f if y != x)
        for _ in range(200):
            for v in und:
                h[v] = np.mean([h[u] for u in nb[int(v)]])
    w = window_weight(X[:, 0], X[:, 1], W, blend)
    if opts.get('vertex_weight') is not None:
        # whole-wall scene: region taper, interpolated to the new vertices from their parents (creation order)
        vw = np.zeros(nloc)
        vw[:len(vids)] = np.asarray(opts['vertex_weight'], float)[vids]
        for i in range(len(vids), nloc):
            idx_, wts_ = sh.parents[i]
            vw[i] = sum(wk * vw[ik] for ik, wk in zip(idx_, wts_))
        w = w * vw
    steep = None
    if opts.get('steep_flip'):
        # zero and detail groups get the same faces
        Ff, steep = flip_steep(Pf2, Pf3 - (w * h)[:, None] * m[None], Ff, m, scanmesh.FRONT_MIN, min_h, segs, snap)
    d = amp * w * h
    Pf3 = Pf3 - d[:, None] * m[None]
    nV = len(V)
    gidx = np.empty(nloc, np.int64)
    gidx[:len(vids)] = vids
    gidx[len(vids):] = nV + np.arange(nloc - len(vids))
    parents = []
    for i in range(len(vids), nloc):
        idx, wts = sh.parents[i]
        parents.append((tuple(int(gidx[j]) for j in idx), tuple(float(x) for x in wts)))
    a2n = (Pf2[Ff[:, 1], 0] - Pf2[Ff[:, 0], 0]) * (Pf2[Ff[:, 2], 1] - Pf2[Ff[:, 0], 1]) - \
          (Pf2[Ff[:, 1], 1] - Pf2[Ff[:, 0], 1]) * (Pf2[Ff[:, 2], 0] - Pf2[Ff[:, 0], 0])
    out_f = gidx[Ff][:, ::-1] if flipped else gidx[Ff]
    eL2 = np.linalg.norm(Pf2[Ff] - Pf2[Ff[:, [1, 2, 0]]], axis=2)
    stats = {'method': 'mesh', 'footprint_centre': cf.tolist(), 'a1': a1.tolist(), 'a2': a2.tolist(), 'W_m': W, 'blend_m': blend,
             'amplitude': amp, 'region_faces_in': int(len(faces)), 'region_faces_out': int(len(Ff)), 'vertices_added': int(nloc - len(vids)),
             'constraint_edges': n_edges, 'constraint_creases': len(segs) - n_edges, 'constraints_inserted': n_ok,
             'constraints_failed': n_fail, 'snap_m': snap, 'adaptive_splits': n_split, 'adaptive_passes': passes_ad,
             'max_lin_err_m': err_max, 'short_edges_collapsed': n_col, 'flat_triangles_flipped': n_flip,
             'vertices_without_donor_surface': n_undef,
             'vertices_on_wall': int((hw > he).sum()), 'min_projected_edge_m': float(eL2.min()),
             'min_projected_area_m2': float(np.abs(a2n).min() / 2), 'projected_orientation_consistent': bool((a2n > 0).all()),
             'disp_mm_p0_p1_p50_p99_p100': [float(x) for x in np.percentile(d[inside] * 1e3, [0, 1, 50, 99, 100])] if inside.any() else None}
    if steep is not None:
        stats['steep_flip'] = steep
    stats['short_edges_left'] = int((eL2 < min_edge).sum())
    flat_left = np.abs(a2n) / np.maximum(eL2.max(1), 1e-30) < min_h
    stats['flat_triangles_left'] = int(flat_left.sum())
    se = np.nonzero(eL2 < min_edge)
    kinds = {}
    for fi, j in zip(*se):
        a_, b_ = Ff[fi, j], Ff[fi, (j + 1) % 3]
        k_ = 'orig-orig' if max(a_, b_) < len(vids) else ('orig-new' if min(a_, b_) < len(vids) else 'new-new')
        kinds[k_] = kinds.get(k_, 0) + 1
    stats['short_edges_left_by_kind'] = kinds
    stats['flat_left_with_all_original_vertices'] = int((Ff[flat_left] < len(vids)).all(1).sum())
    stats['short_edge_left_len_um_min_p50'] = [float(eL2[se].min() * 1e6), float(np.median(eL2[se]) * 1e6)] if len(se[0]) else None
    return {'faces_replaced': faces, 'faces': out_f, 'parent_face': None,
            'new_positions': Pf3[len(vids):], 'moved_original': {int(vids[i]): Pf3[i] for i in range(len(vids)) if d[i] != 0},
            'parents': parents, 'frame': (cf, a1, a2, m, W, blend), 'stats': stats, 'failed_constraints': failed,
            'local': {'gidx': gidx, 'P2': Pf2, 'disp': d, 'F': Ff}}
