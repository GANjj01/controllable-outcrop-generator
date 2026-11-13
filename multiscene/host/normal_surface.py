"""Normal-offset route of the bluff envelope.

The ray route samples the envelope (outcrop.dsgen.morph.envelope_distance) along rays from one interior origin and
re-solves every offset along the ray, so a relief offset o moves the point by about o * tan(theta) along the surface
(theta = ray vs normal); on a wide flat top theta approaches 90 deg. Here the envelope is meshed directly on its
surface and the patch is an erode SHEET (apply = None): positions = base - outward * offset with outward = the
analytic unit normal at the base point. The relief is evaluated at the base point and applied along the normal at any
height.

- envelope_value_grad: envelope_distance with the same arithmetic (checked) and its analytic gradient (rounded-box
distance and the polynomial smooth minimum of morph, rotation included). - project: Newton steps along the gradient
onto envelope_distance = 0 (one step is exact for a single rounded box, whose field is a Euclidean distance outside
the inner box). - box_grid_mesh (one mass): a conforming node grid on the OUTER box surface (six faces share their
edge and corner nodes: one closed patch, no seams), node q -> clamp(q, inner box) + r * unit(q - clamp(q)), which lies
on the rounded box; node spacing uniform (spacing <= h, an odd number of intervals so no node sits on the box's centre
planes) on the flat parts and e + r tan(phi) with equal phi steps on the rounded strips, so edges and corners are
sampled by equal arcs. - normal_patch: the erode Patch (id 'bluff', outward = normal, weights = morph.surface_weights
as for the ray patch). Several masses: surface nets on the smooth union (nets_mesh)."""
import math

import numpy as np


def _rotation(env):
    a = math.radians(env.get('rotation_deg', 0.))
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def envelope_value_grad(p, env):
    """(envelope_distance(p, env), its gradient) for points p (n, 3)."""
    p = np.asarray(p, float)
    rot = bool(env.get('rotation_deg', 0.))
    if rot:
        R = _rotation(env)
        centre = np.asarray(env['origin_m'][:2])
        p = p.copy()
        p[:, :2] = (p[:, :2] - centre) @ R + centre
    result = np.full(len(p), 1e4)
    grad = np.zeros((len(p), 3))
    blend = env['blend_m']
    for mass in env['masses']:
        r = mass['rounding_m']
        dp = p - np.asarray(mass['center_m'])
        q = np.abs(dp) - np.asarray(mass['half_size_m']) + r
        qp = np.maximum(q, 0)
        length = np.linalg.norm(qp, axis=1)
        v = length + np.minimum(np.max(q, axis=1), 0) - r
        sign = np.where(dp >= 0, 1., -1.)
        gv = sign * qp / np.maximum(length, 1e-300)[:, None]
        inner = length <= 0
        if inner.any():
            k = np.argmax(q[inner], axis=1)
            gi = np.zeros((int(inner.sum()), 3))
            gi[np.arange(len(k)), k] = sign[inner][np.arange(len(k)), k]
            gv[inner] = gi
        d = result - v
        h = np.maximum(blend - np.abs(d), 0) / blend
        gmin = np.where((result < v)[:, None], grad, gv)
        grad = gmin + (.5 * h * np.sign(d))[:, None] * (grad - gv)
        result = np.minimum(result, v) - h * h * blend * .25
    if rot:
        grad[:, :2] = grad[:, :2] @ R.T
    return result, grad


def project(p, env, iterations=8, tol=1e-12):
    """Points on envelope_distance = 0 reached by Newton steps along the gradient from p."""
    p = np.array(p, float)
    todo = np.arange(len(p))
    for _ in range(iterations):
        f, g = envelope_value_grad(p[todo], env)
        move = np.abs(f) > tol
        if not move.any():
            break
        todo, f, g = todo[move], f[move], g[move]
        p[todo] -= (f / np.maximum((g * g).sum(1), 1e-300))[:, None] * g
    return p


def normal(p, env):
    g = envelope_value_grad(p, env)[1]
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def axis_nodes(half, r, h):
    """Node coordinates on [-half, half] (relative to the mass centre): rounded strips of width r at both ends, flat part between."""
    e = half - r
    m = max(1, int(math.ceil(2 * e / h)))
    if m % 2 == 0:
        m += 1
    flat = -e + 2 * e * np.arange(m + 1) / m
    k = max(1, int(math.ceil((math.pi / 4) * r / h)))
    strip = e + r * np.tan((math.pi / 4) * np.arange(1, k + 1) / k)
    strip[-1] = half
    return np.concatenate([-strip[::-1], flat, strip])


# box faces: (fixed axis, fixed side, u axis, v axis) with u x v = outward
_FACES = ((0, 1, 1, 2), (0, 0, 2, 1), (1, 1, 2, 0), (1, 0, 0, 2), (2, 1, 0, 1), (2, 0, 1, 0))


def box_grid_mesh(mass, h):
    """(vertices in the mass frame relative to its centre, faces) of the closed conforming box grid on the rounded box."""
    half = np.asarray(mass['half_size_m'], float)
    r = float(mass['rounding_m'])
    axes = [axis_nodes(half[a], r, h) for a in range(3)]
    n = [len(a) for a in axes]
    index = -np.ones(n, np.int64)
    I, J, K = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing='ij')
    surf = (I == 0) | (I == n[0] - 1) | (J == 0) | (J == n[1] - 1) | (K == 0) | (K == n[2] - 1)
    index[surf] = np.arange(int(surf.sum()))
    q = np.stack([axes[0][I[surf]], axes[1][J[surf]], axes[2][K[surf]]], 1)
    inner = half - r
    cl = np.clip(q, -inner, inner)
    w = q - cl
    V = cl + r * w / np.linalg.norm(w, axis=1, keepdims=True)
    faces = []
    for fixed, side, ua, va in _FACES:
        sl = [slice(None)] * 3
        sl[fixed] = n[fixed] - 1 if side else 0
        grid = index[tuple(sl)]                       # 2D array over the two free axes in increasing axis order
        free = [a for a in range(3) if a != fixed]
        if [ua, va] != free:
            grid = grid.T                             # rows = u, columns = v
        a, b = grid[:-1, :-1].ravel(), grid[1:, :-1].ravel()
        c, d = grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()
        # split each quad (a, b, c, d) along its shorter mapped diagonal
        ac = np.linalg.norm(V[a] - V[c], axis=1) <= np.linalg.norm(V[b] - V[d], axis=1)
        faces.append(np.where(ac[:, None], np.c_[a, b, c], np.c_[a, b, d]))
        faces.append(np.where(ac[:, None], np.c_[a, c, d], np.c_[b, c, d]))
    return V, np.vstack(faces), dict(nodes_per_axis=n, vertices=int(len(V)))


def check_closed(V, F):
    """closed 2-manifold with consistent winding, Euler characteristic, zero-area faces, signed volume."""
    e = F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    key = np.sort(e, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    closed = bool((cnt == 2).all())
    winding = False
    if closed:
        order = np.argsort(inv.ravel(), kind='stable').reshape(-1, 2)
        winding = bool((np.where(e[:, 0] < e[:, 1], 1, -1)[order].sum(1) == 0).all())
    T = V[F]
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    vol = float(np.einsum('ij,ij->i', T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6.)
    used = np.unique(F)
    return dict(closed_manifold=closed, winding_consistent=winding, euler=int(len(used) - len(cnt) + len(F)),
                zero_area_faces=int((area <= 0).sum()), min_area_m2=float(area.min()), signed_volume_m3=vol,
                unused_vertices=int(len(V) - len(used)))


NETS_EDGE_RATIO = (.160446 / .139) / 1.0827     # box-grid rms edge per h over nets rms edge per spacing (0.1505 / 0.139)


def normal_mesh(env, h):
    """(V world, F, n, info) of the normal-route base mesh of the envelope."""
    if len(env['masses']) == 1:
        mass = env['masses'][0]
        Vl, F, info = box_grid_mesh(mass, h)
        V = Vl + np.asarray(mass['center_m'], float)
        if env.get('rotation_deg', 0.):
            centre = np.asarray(env['origin_m'][:2])
            V[:, :2] = (V[:, :2] - centre) @ _rotation(env).T + centre
        info['mesher'] = 'box_grid'
    else:
        # surface nets on the smooth union; grid spacing scaled so the rms edge follows the edge rule (rms edge of the box
        # grid's flat parts = 1.155 h; surface nets after relaxation measured 1.083 x its spacing on two 4-mass bluffs)
        V, F, info = nets_mesh(env, h * NETS_EDGE_RATIO)
        info['mesher'] = 'surface_nets'
    f0 = envelope_value_grad(V, env)[0]
    V = project(V, env)
    n = normal(V, env)
    f1 = envelope_value_grad(V, env)[0]
    T = V[F]
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    cosn = np.einsum('ij,ij->i', fn, n[F].mean(1))
    e = np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2)
    info.update(edge_m=dict(min=float(e.min()), p50=float(np.median(e)), max=float(e.max()), rms=float(np.sqrt((e ** 2).mean()))),
                grid_spacing_m=float(h), rounding_m=[float(m_['rounding_m']) for m_ in env['masses']], faces=int(len(F)),
                abs_f_before_projection_max=float(np.abs(f0).max()), abs_f_max=float(np.abs(f1).max()),
                face_vs_normal_cos_min=float(cosn.min()), closed=check_closed(V, F))
    ok = info['closed']
    if not (ok['closed_manifold'] and ok['winding_consistent'] and ok['euler'] == 2 and ok['zero_area_faces'] == 0
            and ok['signed_volume_m3'] > 0 and info['abs_f_max'] <= 1e-9 and info['face_vs_normal_cos_min'] > 0):
        raise RuntimeError('normal route: base mesh self-check failed: %s' % info)
    return V, F, n, info


def normal_patch(env, h, crest_cap=False, focal_curvature=False):
    """crest_cap: positions = base - n * cap_offset(offset, k(base)) through the patch's own apply (erode then treats the
    patch like the re-solved bluff patch: no sheet fold guard is needed, the cap keeps 1 - o k >= CAP_EPS)"""
    from outcrop.dsgen.morph import surface_weights
    from outcrop.dsgen.surface import Patch
    V, F, n, info = normal_mesh(env, h)
    apply = None
    if crest_cap:
        k = principal_curvatures(V, env); Vb = V.copy()
        apply = lambda o: Vb - n * cap_offset(o, k)[:, None]
        info['curvature_per_m'] = dict(k1_max=float(k[:, 0].max()), k2_min=float(k[:, 1].min()))
    if focal_curvature and not crest_cap:
        k = principal_curvatures(V, env)
        info['curvature_per_m'] = dict(k1_max=float(k[:, 0].max()), k2_min=float(k[:, 1].min()))
    patch = Patch('bluff', V, F, n, apply=apply, weights=surface_weights(V, env),
                  provenance={'construction': 'normal-offset sheet on the rounded-box envelope: '
                                              'conforming box grid mapped onto the rounded box, outward = analytic normal'
                                              + ('; offsets capped at the local focal distance (r2c)' if crest_cap else ''),
                              'grid_spacing_m': float(h)})
    if focal_curvature and not crest_cap:
        patch.curvature = k
    return patch, info


# Offsets along the normal fold where 1 - o k_i <= 0 (o the offset, k_i a principal curvature, convex positive): an inward
# offset reaching the radius of a convex edge, or an outward one reaching that of a concave blend. A3 (r = 1.0 m crest, total up
# to 0.99 r) folded there between coarse vertices. The cap keeps 1 - o_eff k >= CAP_EPS everywhere: with L = (1 - CAP_EPS) / k
# (k the largest curvature of the offset's sign), x = o / L, o_eff = L x for x <= CAP_KNEE, else L (a + (1 - a) tanh((x - a) /
# (1 - a))) with a = CAP_KNEE (C1, < L). Where k <= 0 for that sign the offset is unchanged. Physically: a retreat larger than the
# local radius makes a sharp edge (the level set f = -o), which the normal parametrisation cannot reach; the edge keeps a radius
# >= CAP_EPS x the original and the retreat there is reduced (recorded as a limitation).
CAP_EPS, CAP_KNEE, CURV_H = .1, .7, .005


def principal_curvatures(P, env, h=CURV_H):
    """(k1 >= k2) of envelope_distance's level set through P: tangent-projected Hessian (central differences of the analytic
    gradient, step h) / |grad f|."""
    g = envelope_value_grad(P, env)[1]; gn = np.linalg.norm(g, axis=1); n = g / gn[:, None]
    H = np.stack([(envelope_value_grad(P + h * e, env)[1] - envelope_value_grad(P - h * e, env)[1]) / (2 * h) for e in np.eye(3)], 1)
    H = (H + np.transpose(H, (0, 2, 1))) / 2
    Pt = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
    S = (Pt @ H @ Pt) / gn[:, None, None]            # f < 0 inside, outward normal: a convex surface has a positive tangent Hessian
    vals, vecs = np.linalg.eigh(S)
    # the projection leaves one ~zero eigenvalue along n: drop the eigenvalue whose eigenvector is closest to n
    drop = np.argmax(np.abs(np.einsum('nij,ni->nj', vecs, n)), 1)
    keep = np.ones_like(vals, bool); keep[np.arange(len(vals)), drop] = False
    return np.sort(vals[keep].reshape(-1, 2), 1)[:, ::-1]


def cap_offset(off, k):
    """offsets off (n,) with principal curvatures k (n, 2) (convex positive) -> capped offsets"""
    off = np.asarray(off, float); out = off.copy()
    for sign, kk in ((1., k[:, 0]), (-1., -k[:, 1])):            # inward vs largest convex k; outward vs largest concave |k|
        m = (sign * off > 0) & (kk > 1e-12)
        if not m.any(): continue
        L = (1 - CAP_EPS) / kk[m]; x = sign * off[m] / L; a = CAP_KNEE
        y = np.where(x <= a, x, a + (1 - a) * np.tanh((x - a) / (1 - a)))
        out[m] = sign * y * L
    return out


# ---------------------------------------------------------------- several masses: surface nets on the smooth union
def _unrotated(env):
    e = dict(env); e.pop('rotation_deg', None); return e


def nets_mesh(env, h, relax=4, offset=.3819660113):
    """(V world, F) of a closed mesh of envelope_distance = 0 for any number of masses: naive surface nets on a grid of spacing h in
    the masses' own (unrotated) frame, nodes shifted by offset x h (no node exactly on a mass face plane); one vertex per sign-changing
    cell at the mean of its edge crossings; one quad per sign-changing grid edge (wound so its normal points to +f), split along the
    shorter diagonal; vertices projected onto f = 0 and relaxed tangentially (relax rounds of half a Laplacian step without the
    normal component, each followed by the projection)."""
    from outcrop.dsgen.morph import envelope_distance
    eu = _unrotated(env); blend = float(env['blend_m'])
    c = np.array([m['center_m'] for m in env['masses']], float); hs = np.array([m['half_size_m'] for m in env['masses']], float)
    pad = blend + 3 * h
    lo, hi = (c - hs).min(0) - pad, (c + hs).max(0) + pad
    axes = [lo[a] + (np.arange(int(np.ceil((hi[a] - lo[a]) / h)) + 1) + offset) * h for a in range(3)]
    n = [len(a) for a in axes]
    Fv = np.empty(n)
    Y, Z = np.meshgrid(axes[1], axes[2], indexing='ij')
    for i in range(n[0]):
        P = np.c_[np.full(Y.size, axes[0][i]), Y.ravel(), Z.ravel()]
        Fv[i] = envelope_distance(P, eu).reshape(n[1], n[2])
    if (Fv[0] < 0).any() or (Fv[-1] < 0).any() or (Fv[:, 0] < 0).any() or (Fv[:, -1] < 0).any() or (Fv[:, :, 0] < 0).any() or (Fv[:, :, -1] < 0).any():
        raise RuntimeError('nets: the grid does not enclose the envelope')
    inside = Fv < 0
    cell_id = -np.ones((n[0] - 1, n[1] - 1, n[2] - 1), np.int64)
    acc = np.zeros(cell_id.shape + (3,)); cnt = np.zeros(cell_id.shape)
    quads = []
    for ax in range(3):
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3; sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        f0, f1 = Fv[tuple(sl0)], Fv[tuple(sl1)]
        ch = (f0 < 0) != (f1 < 0)
        idx = np.argwhere(ch)
        if not len(idx): continue
        a0, a1 = f0[ch], f1[ch]; t = a0 / (a0 - a1)
        pt = np.stack([axes[k][idx[:, k]] for k in range(3)], 1); pt[:, ax] += t * h
        o1, o2 = [k for k in range(3) if k != ax]
        # the 4 cells sharing this edge: offsets in (o1, o2) of 0/-1
        cells = []
        for d1, d2 in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            cc = idx.copy(); cc[:, o1] += d1; cc[:, o2] += d2; cells.append(cc)
        ok = np.ones(len(idx), bool)
        for cc in cells:
            ok &= (cc >= 0).all(1) & (cc[:, 0] < n[0] - 1) & (cc[:, 1] < n[1] - 1) & (cc[:, 2] < n[2] - 1)
        if not ok.all(): raise RuntimeError('nets: a crossing edge on the grid boundary')
        for cc in cells:
            np.add.at(acc, (cc[:, 0], cc[:, 1], cc[:, 2]), pt); np.add.at(cnt, (cc[:, 0], cc[:, 1], cc[:, 2]), 1)
        # orientation: CCW about +ax in the (o1, o2) frame when (o1, o2, ax) is right-handed; flip if f grows against +ax
        rh = (o1, o2, ax) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        outward_plus = a0 < 0                                           # inside at the start -> outward normal +ax
        flip = outward_plus != rh
        quads.append((cells, flip))
    act = cnt > 0
    cell_id[act] = np.arange(int(act.sum()))
    V = acc[act] / cnt[act][:, None]
    Fl = []
    for cells, flip in quads:
        q = np.stack([cell_id[cc[:, 0], cc[:, 1], cc[:, 2]] for cc in cells], 1)
        q[flip] = q[flip][:, ::-1]
        Fl.append(q)
    Q = np.vstack(Fl)
    # world frame and projection
    def to_world(X):
        X = X.copy()
        if env.get('rotation_deg', 0.):
            centre = np.asarray(env['origin_m'][:2]); X[:, :2] = (X[:, :2] - centre) @ _rotation(env).T + centre
        return X
    V = project(to_world(V), env, iterations=12)
    # triangulate quads along the shorter diagonal
    a, b, cq, d = Q.T
    ac = np.linalg.norm(V[a] - V[cq], axis=1) <= np.linalg.norm(V[b] - V[d], axis=1)
    F = np.vstack([np.where(ac[:, None], np.c_[a, b, cq], np.c_[a, b, d]), np.where(ac[:, None], np.c_[a, cq, d], np.c_[b, cq, d])])
    # tangential relaxation
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    for _ in range(relax):
        s = np.zeros_like(V); k = np.zeros(len(V))
        np.add.at(s, E[:, 0], V[E[:, 1]]); np.add.at(k, E[:, 0], 1); np.add.at(s, E[:, 1], V[E[:, 0]]); np.add.at(k, E[:, 1], 1)
        dv = s / k[:, None] - V; nn = normal(V, env); dv -= (dv * nn).sum(1)[:, None] * nn
        V = project(V + .5 * dv, env, iterations=12)
    return V, F, dict(grid_nodes=n, grid_spacing_m=float(h), relax=relax, cells_active=int(act.sum()), quads=int(len(Q)))
