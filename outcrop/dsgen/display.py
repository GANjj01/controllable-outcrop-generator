"""Display layer: subdivide the exposure, displace and colour it so labelled features are visible.

Region labels stay attached to the original faces (each sub-triangle inherits its parent's code).
All displacement is a continuous field over a welded mesh, so contacts do not crack. Trace
polylines are displaced with the same field, so projected line labels stay on the visible surface.
"""
import math

import numpy as np

from .lithology import get as lith_params, order_key


# ---------------------------------------------------------------- noise
def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def value_noise(p, seed):
    """Trilinear value noise in [0,1] at points p (N,3), unit lattice."""
    f = np.floor(p)
    i = f.astype(np.int64)
    t = p - f
    t = t * t * (3 - 2 * t)
    out = np.zeros(len(p))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                out += w * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out


def fbm(p, seed, octaves=5, lacunarity=2.03, gain=.5):
    total, amp, norm = np.zeros(len(p)), 1., 0.
    for o in range(octaves):
        total += amp * (value_noise(p * lacunarity ** o, seed + 101 * o) - .5)
        norm += amp
        amp *= gain
    return total / norm  # roughly in [-0.5, 0.5]


# ---------------------------------------------------------------- mesh
def subdivide(vertices, faces, max_edge):
    """Uniform barycentric subdivision; returns welded vertices, faces and parent face ids."""
    pts, tris, parents = [], [], []
    offset = 0
    for fi, (a, b, c) in enumerate(vertices[faces]):
        k = max(1, int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)) / max_edge)))
        index = {}
        local = []
        for i in range(k + 1):
            for j in range(k + 1 - i):
                index[(i, j)] = len(local)
                local.append(a + (b - a) * (i / k) + (c - a) * (j / k))
        for i in range(k):
            for j in range(k - i):
                tris.append([index[(i, j)] + offset, index[(i + 1, j)] + offset, index[(i, j + 1)] + offset])
                parents.append(fi)
                if i + j < k - 1:
                    tris.append([index[(i + 1, j)] + offset, index[(i + 1, j + 1)] + offset, index[(i, j + 1)] + offset])
                    parents.append(fi)
        pts.extend(local)
        offset += len(local)
    pts = np.asarray(pts)
    tris = np.asarray(tris, np.int64)
    # Weld coincident points (shared original edges/vertices) at 0.1 mm.
    key = np.round(pts / 1e-4).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return pts[first], inverse[tris], np.asarray(parents, np.int64)


def vertex_normals(vertices, faces, reference, consistent=False):
    """Area-weighted vertex normals. With consistent winding (the parametric grid and its
    refinements) only one global sign is chosen from the reference, so genuinely downward-facing
    faces (overhangs, ledge roofs, the foot of the cliff) keep their true orientation. Without it,
    each face is flipped towards the reference (inconsistent v2 section meshes)."""
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if consistent:
        n *= -1. if (n @ reference).sum() < 0 else 1.
    else:
        n *= np.where((n @ reference) < 0, -1., 1.)[:, None]
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], n)
    length = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(length, 1e-12)


def smooth(values, faces, n_vertices, iterations):
    """Graph Laplacian smoothing over mesh edges (vectorised Jacobi iterations)."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    degree = np.bincount(edges[:, 0], minlength=n_vertices).astype(float)
    v = values.copy()
    for _ in range(iterations):
        acc = np.zeros_like(v)
        np.add.at(acc, edges[:, 0], v[edges[:, 1]])
        v = .5 * v + .5 * acc / np.maximum(degree, 1)[..., None] if v.ndim > 1 else .5 * v + .5 * acc / np.maximum(degree, 1)
    return v


def taubin(vertices, faces, iterations, lam=.5, mu=-.53):
    """Volume-preserving Taubin smoothing; open-boundary vertices stay fixed."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(edges, 1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    boundary = np.zeros(len(vertices), bool)
    boundary[uniq[counts == 1].ravel()] = True
    both = np.concatenate([edges, edges[:, ::-1]])
    degree = np.maximum(np.bincount(both[:, 0], minlength=len(vertices)), 1).astype(float)[:, None]
    v = vertices.copy()
    for k in range(2 * iterations):
        acc = np.stack([np.bincount(both[:, 0], weights=v[both[:, 1], c], minlength=len(v)) for c in range(3)], 1)
        step = (lam if k % 2 == 0 else mu) * (acc / degree - v)
        step[boundary] = 0
        v = v + step
    return v


_SEG_KERNEL = r"""
extern "C" __global__ void seg_min_dist(const float* p, const float* a, const float* ab, const float* inv,
                                        int n, int m, float* out, int* arg) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float px = p[3*i], py = p[3*i+1], pz = p[3*i+2];
    float best = 3.4e38f;
    int besti = 0;
    for (int k = 0; k < m; ++k) {
        float wx = px - a[3*k], wy = py - a[3*k+1], wz = pz - a[3*k+2];
        float t = (wx*ab[3*k] + wy*ab[3*k+1] + wz*ab[3*k+2]) * inv[k];
        t = fminf(fmaxf(t, 0.f), 1.f);
        float dx = wx - t*ab[3*k], dy = wy - t*ab[3*k+1], dz = wz - t*ab[3*k+2];
        float dd = dx*dx + dy*dy + dz*dz;
        if (dd < best) { best = dd; besti = k; }
    }
    out[i] = sqrtf(best);
    arg[i] = besti;
}
"""
_GPU = {}


def _gpu_kernel():
    if 'kernel' not in _GPU:
        from ..gpu_runtime import configure_gpu_cache
        configure_gpu_cache()
        try:
            import cupy as cp
            _GPU['cp'] = cp
            _GPU['kernel'] = cp.RawKernel(_SEG_KERNEL, 'seg_min_dist')
        except Exception:  # no usable CUDA/cupy: CPU fallback
            _GPU['kernel'] = None
    return _GPU['kernel']


def segment_distance(points, segments, chunk=4096):
    """Min distance from each point to a set of 3D segments (GPU when available, else NumPy)."""
    kernel = _gpu_kernel()
    if kernel is not None and len(segments):
        cp = _GPU['cp']
        origin = points.mean(0)  # float32 relative coordinates keep sub-millimetre precision
        a = (segments[:, 0] - origin).astype(np.float32)
        ab = (segments[:, 1] - segments[:, 0]).astype(np.float32)
        inv = (1. / np.maximum((ab.astype(np.float64) ** 2).sum(1), 1e-18)).astype(np.float32)
        p = cp.asarray((points - origin).astype(np.float32))
        out = cp.empty(len(points), dtype=cp.float32)
        arg = cp.empty(len(points), dtype=cp.int32)
        threads = 256
        kernel(((len(points) + threads - 1) // threads,), (threads,),
               (p, cp.asarray(a), cp.asarray(ab), cp.asarray(inv), np.int32(len(points)), np.int32(len(a)), out, arg))
        return cp.asnumpy(out).astype(np.float64), cp.asnumpy(arg).astype(np.int64)
    a, b = segments[:, 0], segments[:, 1]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-18)
    best = np.full(len(points), np.inf)
    arg = np.zeros(len(points), np.int64)
    for s in range(0, len(points), chunk):
        q = points[s:s + chunk, None, :]
        t = np.clip(((q - a) * ab).sum(-1) / denom, 0, 1)
        dist = np.linalg.norm(q - (a + t[..., None] * ab), axis=-1)
        best[s:s + chunk] = dist.min(1)
        arg[s:s + chunk] = dist.argmin(1)
    return best, arg


def densify(segments, step):
    out = []
    for a, b in segments:
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        t = np.linspace(0, 1, n + 1)
        out.append(a + (b - a) * t[:, None])
    return out


def nearest_vertex(points, vertices, cell):
    """Nearest vertex via a uniform hash grid (points lie on or near the mesh)."""
    keys = np.floor(vertices / cell).astype(np.int64)
    grid = {}
    for i, k in enumerate(map(tuple, keys)):
        grid.setdefault(k, []).append(i)
    idx = np.zeros(len(points), np.int64)
    for n, p in enumerate(points):
        c = np.floor(p / cell).astype(np.int64)
        cand = []
        r = 1
        while not cand:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        cand.extend(grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ()))
            r += 1
        cand = np.asarray(cand)
        idx[n] = cand[((vertices[cand] - p) ** 2).sum(1).argmin()]
    return idx


# ---------------------------------------------------------------- parametric smooth surface
def _catmull_weights(t):
    t2, t3 = t * t, t * t * t
    return np.stack([-.5 * t3 + t2 - .5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + .5 * t, .5 * t3 - .5 * t2], -1)


def catmull_surface(G, u, v):
    """Tensor Catmull-Rom surface through node grid G (ny, nx, 3) at parameters u (cols), v (rows)."""
    ny, nx, _ = G.shape
    iu = np.clip(np.floor(u).astype(np.int64), 0, nx - 2)
    iv = np.clip(np.floor(v).astype(np.int64), 0, ny - 2)
    wu = _catmull_weights(u - iu)
    wv = _catmull_weights(v - iv)
    out = np.zeros((len(u), 3))
    for a in range(4):
        cols = np.clip(iu + a - 1, 0, nx - 1)
        for b in range(4):
            rows = np.clip(iv + b - 1, 0, ny - 1)
            out += (wu[:, a] * wv[:, b])[:, None] * G[rows, cols]
    # No overshoot: keep each point inside the bounding box of its cell's four corner nodes (a
    # Catmull-Rom spline otherwise dips below the plain before the cliff foot, leaving a trench).
    corners = np.stack([G[iv, iu], G[iv, iu + 1], G[iv + 1, iu], G[iv + 1, iu + 1]])
    return np.clip(out, corners.min(0), corners.max(0))


def _coarse_triangle(G, u, v):
    """Coarse PL triangle (scene face order) containing parameter (u, v) and the PL point there."""
    ny, nx, _ = G.shape
    i = np.clip(np.floor(u).astype(np.int64), 0, nx - 2)
    j = np.clip(np.floor(v).astype(np.int64), 0, ny - 2)
    fu, fv = u - i, v - j
    a, b, cc, d = G[j, i], G[j, i + 1], G[j + 1, i + 1], G[j + 1, i]
    first = fu >= fv
    p1 = a + (fu - fv)[:, None] * (b - a) + fv[:, None] * (cc - a)
    p2 = a + fu[:, None] * (cc - a) + (fv - fu)[:, None] * (d - a)
    k = 2 * (j * (nx - 1) + i) + np.where(first, 0, 1)
    return k, np.where(first[:, None], p1, p2)


def _barycentric(P, T):
    """Barycentric coordinates of points P (m,3) in triangles T (f,3,3) -> (m,f,3) and plane distance."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    v0, v1 = b - a, c - a
    n = np.cross(v0, v1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    w = P[:, None, :] - a[None]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (w * v0[None]).sum(-1), (w * v1[None]).sum(-1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - bv - bw, bv, bw], -1), np.abs((w * n[None]).sum(-1))


def classify_points(pkg, k, P):
    """Index of the v2 section face (same source cliff triangle) containing each PL point, or -1."""
    out = np.full(len(P), -1, np.int64)
    by_source = {}
    for f, sid in enumerate(pkg['source_ids']):
        if sid.startswith('cliff:triangle:'):
            by_source.setdefault(int(sid.rsplit(':', 1)[1]), []).append(f)
    tri = pkg['vertices'][pkg['faces']]
    order = np.argsort(k, kind='stable')
    ks, starts = np.unique(k[order], return_index=True)
    bounds = np.r_[starts, len(order)]
    for n, kk in enumerate(ks):
        idx = order[bounds[n]:bounds[n + 1]]
        faces = by_source.get(int(kk))
        if not faces:
            continue
        bary, _ = _barycentric(P[idx], tri[faces])
        best = bary.min(-1).argmax(1)
        out[idx] = np.asarray(faces)[best]
    return out


def map_to_parameters(G, points, chunk=2048):
    """Parameters (u, v) of points lying on the coarse PL node-grid surface."""
    ny, nx, _ = G.shape
    tris, cells = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, cc, d = G[j, i], G[j, i + 1], G[j + 1, i + 1], G[j + 1, i]
            tris += [[a, b, cc], [a, cc, d]]
            cells += [(i, j, 0), (i, j, 1)]
    tris = np.asarray(tris)
    cells = np.asarray(cells)
    uv = np.zeros((len(points), 2))
    for s in range(0, len(points), chunk):
        bary, dist = _barycentric(points[s:s + chunk], tris)
        score = dist + 10 * np.clip(-bary.min(-1), 0, None)
        t = score.argmin(1)
        bb = bary[np.arange(len(t)), t]
        i, j, half = cells[t, 0], cells[t, 1], cells[t, 2]
        # tri0: a + alpha(b-a) + beta(cc-a): fu = alpha + beta, fv = beta
        # tri1: a + alpha(cc-a) + beta(d-a): fu = alpha, fv = alpha + beta
        alpha, beta = bb[:, 1], bb[:, 2]
        fu = np.where(half == 0, alpha + beta, alpha)
        fv = np.where(half == 0, beta, alpha + beta)
        uv[s:s + chunk] = np.c_[i + fu, j + fv]
    return uv


def _samples(lengths, edges):
    return [max(1, int(math.ceil(length / edge))) for length, edge in zip(lengths, edges)]


def parametric_base(pkg, grid, max_edge, ext=None, offset=(0, 0), outer_edge=.6, classify=None):
    """Fine smooth surface over the (optionally extended) node grid; inner faces classified by v2."""
    ny, nx = grid
    G = pkg['model_cliff_vertices'].reshape(ny, nx, 3)
    E = G if ext is None else np.asarray(ext, float)
    r0, c0 = offset
    my, mx, _ = E.shape
    inner_c = lambda i: c0 <= i < c0 + nx - 1
    inner_r = lambda j: r0 <= j < r0 + ny - 1
    ku = _samples([np.linalg.norm(E[:, i + 1] - E[:, i], axis=1)[r0:r0 + ny].max() if inner_c(i) else
                   np.linalg.norm(E[:, i + 1] - E[:, i], axis=1).max() for i in range(mx - 1)],
                  [max_edge if inner_c(i) else outer_edge for i in range(mx - 1)])
    kv = _samples([np.linalg.norm(E[j + 1] - E[j], axis=1)[c0:c0 + nx].max() if inner_r(j) else
                   np.linalg.norm(E[j + 1] - E[j], axis=1).max() for j in range(my - 1)],
                  [max_edge if inner_r(j) else outer_edge for j in range(my - 1)])
    us = np.concatenate([i + np.arange(k) / k for i, k in enumerate(ku)] + [[mx - 1.]])
    vs = np.concatenate([j + np.arange(k) / k for j, k in enumerate(kv)] + [[my - 1.]])
    UU, VV = np.meshgrid(us, vs)
    u, v = UU.ravel(), VV.ravel()
    V = catmull_surface(E, u, v)
    m, n = len(vs), len(us)
    r, c = np.meshgrid(np.arange(m - 1), np.arange(n - 1), indexing='ij')
    a = (r * n + c).ravel()
    b, cc, d = a + 1, a + n + 1, a + n
    F = np.concatenate([np.c_[a, b, cc], np.c_[a, cc, d]])
    # Inner faces: classify the centroid parameter on the exact v2 section; outer faces have no rock.
    cu, cv = u[F].mean(1) - c0, v[F].mean(1) - r0
    inside = (cu >= 0) & (cu <= nx - 1) & (cv >= 0) & (cv <= ny - 1)
    parent = np.full(len(F), -1, np.int64)
    if inside.any():
        k, P = _coarse_triangle(G, cu[inside], cv[inside])
        # Sandbox tiles have no v2 cliff section: classify the PL points by v2 queries instead.
        parent[inside] = classify(P) if classify is not None else classify_points(pkg, k, P)
    # Distance (m, along strike) from each vertex to the lateral rock-domain edge, for edge cover.
    col_x = G[0, :, 0]
    uu = np.clip(u - c0, 0, nx - 1)
    x_here = np.interp(uu, np.arange(nx), col_x)
    lateral = np.where((u - c0 >= 0) & (u - c0 <= nx - 1), np.minimum(x_here - col_x[0], col_x[-1] - x_here), 0.)
    # Map trace points from the coarse PL surface onto the smooth surface.
    dense = densify(pkg['segments'], max_edge / 2)
    mapped = []
    if dense:
        flat = np.concatenate(dense)
        uv = map_to_parameters(G, flat)
        moved = catmull_surface(E, uv[:, 0] + c0, uv[:, 1] + r0)
        start = 0
        for pts in dense:
            mapped.append(moved[start:start + len(pts)])
            start += len(pts)
    return V, F, parent, mapped, lateral


# ---------------------------------------------------------------- build
def build(pkg, face_codes, line_classes, outward, rng, *, max_edge=.08, grid=None, ext=None, ext_offset=(0, 0), truth=None,
          classify=None):
    seed = int(rng.integers(1, 2 ** 30))
    lateral = None
    if grid is not None:
        V, F, parent, traces, lateral = parametric_base(pkg, grid, max_edge, ext=ext, offset=ext_offset, classify=classify)
    else:
        V, F, parent = subdivide(pkg['vertices'], pkg['faces'], max_edge)
        traces = densify(pkg['segments'], max_edge / 2)
    nv = len(V)
    consistent = grid is not None
    N = vertex_normals(V, F, outward, consistent)
    rock = parent >= 0
    safe_parent = np.where(rock, parent, 0)
    lith_names = pkg['lithology']
    face_lith = [lith_names[p] for p in safe_parent]
    params = {name: lith_params(name) for name in set(lith_names)}

    # Per-vertex lithology properties (mean over incident faces), then smoothed so contacts round off.
    recess_scale = float(rng.uniform(.12, .6))
    face_recess = np.array([params[l]['recess'] for l in face_lith]) * recess_scale
    face_rough = np.array([params[l]['rough'] for l in face_lith]) * float(rng.uniform(.6, 1.6))
    # Bed-specific colour: each body draws its own tint so equal lithologies differ between beds/scenes.
    body_color = {}

    def tint(body, lith):
        if body not in body_color:
            choices = lith_params(lith)['base']
            # Stronger per-bed brightness and hue jitter: neighbouring beds of similar lithology
            # still differ, as weathered beds do in the field.
            c = np.array(choices[rng.integers(len(choices))]) * rng.uniform(.7, 1.25) + rng.normal(0, .045, 3)
            body_color[body] = np.clip(c, .05, .95)
        return body_color[body]

    for body, lith in sorted(set(zip(pkg['body'], pkg['lithology'])), key=lambda k: (k[0], order_key(k[1]))):   # tint draws in the order field
        tint(body, lith)
    face_color = np.array([body_color[pkg['body'][p]] for p in safe_parent])
    smooth_iters = int(rng.integers(2, 6))
    attr = np.zeros((nv, 5))
    count = np.zeros(nv)
    for k in range(3):
        np.add.at(attr, F[:, k], np.c_[face_recess, face_rough, face_color])
        np.add.at(count, F[:, k], 1)
    attr /= np.maximum(count, 1)[:, None]
    attr = smooth(attr, F, nv, smooth_iters)
    recess, rough, color = attr[:, 0], attr[:, 1], attr[:, 2:]

    # Structural grooves from traces: joints narrow and shallow, faults wider and deeper.
    groove = np.zeros(nv)
    stain = np.zeros(nv)
    fault_zone = np.zeros(nv)
    seg_ids = list(pkg['segment_ids'])
    if truth is not None:
        # Grooves follow the exact intersections of the (undisplaced) display surface with the v2
        # fault/joint interfaces - the same computation that later yields the line labels on the
        # displaced surface - rather than the parametric mapping of the section traces.
        from .truth import surface_intersections
        exact = surface_intersections(V, F, truth.interfaces)
        traces = [pts for _, pts in exact]
        seg_ids = [iid for iid, _ in exact]
    seg_cls = np.array([line_classes.get(i, 'contact') for i in seg_ids])
    # Grooves follow the traces on the display surface (consecutive point pairs).
    pair_segments = [np.stack([t[:-1], t[1:]], 1) for t in traces]
    groove_centres = traces
    jw, jd = float(rng.uniform(.03, .10)), float(rng.uniform(.03, .12))
    fw, fd = float(rng.uniform(.08, .35)), float(rng.uniform(.04, .18))
    for cls, width, depth in (('joint', jw, jd), ('fault', fw, fd), ('contact', jw * .7, jd * .5)):
        mask = seg_cls == cls
        if not mask.any():
            continue
        d, _ = segment_distance(V, np.concatenate([pair_segments[q] for q in np.nonzero(mask)[0]]))
        # Width varies along the feature (weathering is not uniform).
        local = width * (.6 + .8 * value_noise(V * 1.3, seed + 7))
        profile = np.exp(-.5 * (d / local) ** 2)
        groove = np.maximum(groove, depth * profile)
        stain = np.maximum(stain, profile * (1. if cls == 'fault' else .6))
        if cls == 'fault':
            fault_zone = np.maximum(fault_zone, np.exp(-.5 * (d / (2.5 * local)) ** 2))

    # Bedding partings: a narrow recessed, darkened seam along every contact between different
    # units, so stratification is visible even between similar lithologies.
    unit_key = np.array([pkg['unit'][p] or pkg['body'][p] if p >= 0 else '' for p in parent], object)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    owner = np.tile(np.arange(len(F)), 3)
    key = np.sort(edges, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same = (key[1:] == key[:-1]).all(1)
    fa, fb = owner[:-1][same], owner[1:][same]
    contact = (unit_key[fa] != unit_key[fb]) & (parent[fa] >= 0) & (parent[fb] >= 0)
    parting = np.zeros(nv)
    d_contact = np.full(nv, np.inf)
    if truth is not None:
        from .truth import contact_segments
        cs = contact_segments(V, F, truth)
        if len(cs):
            d_contact, _ = segment_distance(V, cs)
    elif contact.any():
        ce = key[:-1][same][contact]
        d_contact, _ = segment_distance(V, np.stack([V[ce[:, 0]], V[ce[:, 1]]], 1))
    if np.isfinite(d_contact).any():
        pw, pd = float(rng.uniform(.03, .08)), float(rng.uniform(.03, .08))
        parting = pd * np.exp(-.5 * (d_contact / (pw * (.6 + .8 * value_noise(V * 2.1, seed + 19)))) ** 2)
        groove = np.maximum(groove, parting)
    # Weak-bed recess as a continuous function of distance to the nearest bed contact: zero on the
    # contact and deepening into the weak bed over a ramp proportional to the recess depth. The
    # per-face (8 cm staircase) bed membership then only decides the side, never a vertical step
    # whose shadowed riser would show as a black sawtooth.
    if truth is not None:
        # Each vertex takes the recess of the v2 unit at its own position (not the average of its
        # 8 cm faces): vertices on the resistant side of the exact contact line get exactly zero.
        st_v, body_v, _, lith_v = truth.query(V)
        r_v = np.array([lith_params(l)['recess'] * recess_scale if st == 'occupied' else 0.
                        for st, l in zip(st_v, lith_v)])
    else:
        r_v = np.zeros(nv)
        n_v = np.zeros(nv)
        for k in range(3):
            np.add.at(r_v, F[:, k], face_recess)
            np.add.at(n_v, F[:, k], 1)
        r_v /= np.maximum(n_v, 1)
    ramp_k = float(rng.uniform(1.5, 2.8))
    t = np.clip(d_contact / np.maximum(ramp_k * np.abs(r_v), .05), 0, 1)
    recess = r_v * t * t * (3 - 2 * t)
    macro = fbm(V * float(rng.uniform(.25, .6)), seed + 1, 4) * float(rng.uniform(.03, .12))
    micro = fbm(V * float(rng.uniform(5, 12)), seed + 2, 5) * rough * 2
    # Metamorphic banding: compositional layering / schistosity drawn along the v2 foliation
    # coordinate (fold limbs and faulted blocks carry their own fabric), as fine relief and tint.
    band_tint = np.zeros(nv)
    if truth is not None:
        amp = np.array([lith_params(l)['banding'] if st == 'occupied' else 0. for st, l in zip(st_v, lith_v)])
        if amp.any():
            ok, _, fol = truth.fabric(V, body_v, 'foliation')
            amp = np.where(ok, amp, 0.)
            fol = np.where(ok, fol, 0.)
            # Wavelengths the 8 cm display mesh can carry (cm-scale schistosity would alias).
            fine, coarse = float(rng.uniform(.25, .6)), float(rng.uniform(.4, 1.2))
            lam = np.array([coarse if l == 'group_c_04' else fine for l in lith_v])
            wob = 2. * fbm(V * .7, seed + 21, 3)
            stripes = np.sin(2 * np.pi * fol / lam + wob) + .5 * np.sin(2 * np.pi * fol / (2.7 * lam) + 1.3 * wob)
            band_tint = .22 * amp * stripes / 1.5
            micro = micro + .004 * amp * stripes
    # Fault zones are crushed: extra rough relief inside the damage zone.
    crush = fbm(V * 18., seed + 3, 3) * .03 * fault_zone
    disp = -(recess + groove) + macro + micro + crush
    Vs, Ns = V, N

    # Soil/vegetation cover on gentle slopes: rock is exposed only where it is steep enough.
    slope = np.degrees(np.arccos(np.clip(Ns @ (outward / np.linalg.norm(outward)), -1, 1)))
    threshold = float(rng.uniform(28, 46)) + 14 * (value_noise(V * .35, seed + 11) - .5)
    cover = np.clip((threshold - slope) / 6 + .5, 0, 1)
    # Points outside any rock body (if any) are always covered.
    no_rock = np.zeros(nv)
    for q in range(3):
        np.maximum.at(no_rock, F[~rock, q], 1.)
    cover = np.maximum(cover, no_rock)
    if lateral is not None:
        # Rock gives way to soil towards the lateral domain edges (irregular, not a straight seam).
        ramp = float(rng.uniform(2, 6)) * (.6 + .8 * value_noise(V * .5, seed + 18))
        cover = np.maximum(cover, np.clip(1 - lateral / ramp, 0, 1))
    cover = smooth(cover, F, nv, 10)
    # Soil thickens away from the rock edge (cover squared), avoiding a raised lip with a dark gap.
    soil = float(rng.uniform(.05, .25)) * cover ** 4 * (.6 + .8 * value_noise(V * 1.7, seed + 12))
    # Rock relief (including weak-bed recess) fades quadratically into the cover, and soil only
    # thickens where cover is complete: no raised soil lip over a recessed notch.
    disp = disp * (1 - cover) ** 2 + soil + fbm(V * 4., seed + 13, 3) * .04 * cover ** 2
    inplane = None
    if truth is not None:
        # Normal displacement without grooves; grooves/partings are carved afterwards, in-plane.
        disp = disp + groove * (1 - cover) ** 2
    V2 = Vs + Ns * disp[:, None]
    # Fold guard: where displacement flips a face (its normal turns against the base normal),
    # halve the displacement of that face's vertices until no face is inverted.
    base_n = np.cross(Vs[F[:, 1]] - Vs[F[:, 0]], Vs[F[:, 2]] - Vs[F[:, 0]])
    fold_passes = []
    for _ in range(8):
        n2 = np.cross(V2[F[:, 1]] - V2[F[:, 0]], V2[F[:, 2]] - V2[F[:, 0]])
        flipped = (n2 * base_n).sum(1) <= 0
        fold_passes.append(int(flipped.sum()))
        if not flipped.any():
            break
        bad = np.unique(F[flipped])
        disp[bad] *= .5
        V2 = Vs + Ns * disp[:, None]
    if truth is not None:
        # Fracture grooves and bedding partings are centred on the exact intersections of THIS
        # surface with the v2 interfaces and carved along a direction lying in the interface plane:
        # points on the plane stay on it, so the final line labels run along the groove floor.
        from .truth import surface_intersections
        N1 = vertex_normals(V2, F, outward, consistent)
        fade = (1 - cover) ** 2
        carve = np.zeros((nv, 3))
        groove = np.zeros(nv)
        stain = np.zeros(nv)
        frac_n = np.zeros((nv, 3))
        frac_w = np.zeros(nv)
        d_fault = np.full(nv, np.inf)
        exact = surface_intersections(V2, F, truth.interfaces, with_normals=True)
        groove_centres = [pts for _, pts, _ in exact]
        segs, seg_n, seg_c = [], [], []
        for iid, pts, normal in exact:
            segs.append(np.stack([pts[:-1], pts[1:]], 1))
            seg_n.append(np.repeat(normal[None], len(pts) - 1, 0))
            seg_c.extend([line_classes.get(iid, 'contact')] * (len(pts) - 1))
        if segs:
            segs, seg_n, seg_c = np.concatenate(segs), np.concatenate(seg_n), np.array(seg_c)
            for cls, width, depth in (('joint', jw, jd), ('fault', fw, fd), ('contact', jw * .7, jd * .5)):
                mask = seg_c == cls
                if not mask.any():
                    continue
                d, k = segment_distance(V2, segs[mask])
                if cls == 'fault':
                    d_fault = np.minimum(d_fault, d)
                local = width * (.6 + .8 * value_noise(V * 1.3, seed + 7))
                profile = np.exp(-.5 * (d / local) ** 2)
                g = depth * profile * fade
                upd = g > groove
                n = seg_n[mask][k]
                tdir = -(N1 - (N1 * n).sum(1, keepdims=True) * n)
                tl = np.linalg.norm(tdir, axis=1, keepdims=True)
                tdir = np.where(tl > .2, tdir / np.maximum(tl, 1e-9), -N1)
                carve[upd] = tdir[upd] * g[upd, None]
                near = profile > frac_w
                frac_n[near] = n[near]
                frac_w = np.maximum(frac_w, profile)
                groove = np.maximum(groove, g)
                stain = np.maximum(stain, profile * (1. if cls == 'fault' else .6))
        # Bedding partings on the contacts between v2 units of this surface, carved in the bedding plane.
        from .truth import contact_segments
        cs = contact_segments(V2, F, truth)
        if len(cs):
            # Unit changes across a fault are fault contacts, and across an intrusion contact they are
            # intrusive contacts (carved above along their own plane): no bedding parting there.
            for cls, width in (('fault', fw), ('contact', jw * .7)):
                if len(cs) and len(segs) and (seg_c == cls).any():
                    dd, _ = segment_distance(cs.mean(1), segs[seg_c == cls])
                    cs = cs[dd > max(width, .05)]
        ce = cs
        parting = np.zeros(nv)
        if len(ce):
            dc, _ = segment_distance(V2, ce)
            pw, pd = float(rng.uniform(.03, .08)), float(rng.uniform(.03, .08))
            parting = pd * np.exp(-.5 * (dc / (pw * (.6 + .8 * value_noise(V * 2.1, seed + 19)))) ** 2) * fade
            # Carve in the local bedding plane: the v2 bedding fabric of each vertex's body (each
            # chevron limb has its own); vertices without a bedding host carve along -normal.
            _, body_2, _, _ = truth.query(V2)
            has_bed, bed_n, _ = truth.fabric(V2, body_2, 'bedding')
            tb = -(N1 - (N1 * bed_n).sum(1, keepdims=True) * bed_n)
            tl = np.linalg.norm(tb, axis=1, keepdims=True)
            tb = np.where((tl > .2) & has_bed[:, None], tb / np.maximum(tl, 1e-9), -N1)
            bed_n = np.where(has_bed[:, None], bed_n, 0.)
            upd = parting > groove
            carve[upd] = tb[upd] * parting[upd, None]
            groove = np.maximum(groove, parting)
        else:
            bed_n = None
        # Where a fracture and a bedding contact both act, carve along their common line so points
        # stay on both planes (keeps fracture labels on the groove floor across partings).
        mag = np.linalg.norm(carve, axis=1)
        dirs = carve / np.maximum(mag, 1e-12)[:, None]
        both = (mag > 0) & (frac_w > .05) & (parting > .05 * max(float(parting.max()), 1e-9))
        if both.any():
            dd = dirs[both] - (dirs[both] * frac_n[both]).sum(1, keepdims=True) * frac_n[both]
            if bed_n is not None:
                bn = bed_n[both]
                dd = dd - (dd * bn).sum(1, keepdims=True) * bn
            dl = np.linalg.norm(dd, axis=1, keepdims=True)
            dirs[both] = np.where(dl > .2, dd / np.maximum(dl, 1e-9), dirs[both])
        V_pre_carve, carve_final = V2, dirs * mag[:, None]
        # Fold guard for the carve itself: halve the carve on vertices of any face it inverts.
        pre_n = np.cross(V2[F[:, 1]] - V2[F[:, 0]], V2[F[:, 2]] - V2[F[:, 0]])
        carve_flips = []
        for _ in range(10):
            Vc = V_pre_carve + carve_final
            nc = np.cross(Vc[F[:, 1]] - Vc[F[:, 0]], Vc[F[:, 2]] - Vc[F[:, 0]])
            flip = (nc * pre_n).sum(1) <= 0
            carve_flips.append(int(flip.sum()))
            if not flip.any():
                break
            carve_final[np.unique(F[flip])] *= .5
        V2 = V_pre_carve + carve_final
        inplane = {'fracture_lines': len(exact), 'contact_segments': int(len(ce)), 'carve_flips_per_pass': carve_flips}
    folded_faces = int(flipped.sum()) if 'flipped' in dir() else 0
    offset = V2 - V

    # Exact truth on the displayed surface: query v2 at every final face centroid, then colour
    # (and later texture) faces from that answer so image and labels share one source.
    face_keys = None
    if truth is not None:
        status, qbody, qunit, qlith = truth.query(V2[F].mean(1))
        rock_q = status == 'occupied'
        face_keys = [(b, u, l) if ok else None for ok, b, u, l in zip(rock_q, qbody, qunit, qlith)]
        fc = np.array([tint(b, l) if ok else (.3, .25, .2) for ok, b, l in zip(rock_q, qbody, qlith)])
        acc = np.zeros((nv, 3))
        cnt = np.zeros(nv)
        for k in range(3):
            np.add.at(acc, F[rock_q, k], fc[rock_q])
            np.add.at(cnt, F[rock_q, k], 1)
        color = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], color)
        color = smooth(color, F, nv, max(1, smooth_iters - 1))
        not_rock = np.zeros(nv)
        for q in range(3):
            np.maximum.at(not_rock, F[~rock_q, q], 1.)
        cover = np.maximum(cover, not_rock)
        rock = rock & rock_q

    # Colour: bed tint, weathering mottling, darkening in recesses, iron staining near fractures,
    # vertical runoff streaks.
    mott = fbm(V * float(rng.uniform(.6, 1.5)), seed + 4, 4)
    streak_p = V * np.array([2.5, 2.5, .15])
    streak = np.clip(fbm(streak_p, seed + 5, 3) * 2.2, -1, 1)
    # Darken fracture grooves strongly and weak-bed recesses only mildly (relative to the local
    # maximum recess), so deep recess does not turn whole beds black.
    cavity = (np.clip(groove / .08, 0, 1) + .35 * np.clip(recess / max(float(recess.max()), 1e-6), 0, 1)
              + .6 * np.clip(parting / .04, 0, 1))
    cavity = np.clip(cavity, 0, 1)
    rgb = color * (1 + .35 * mott + band_tint)[:, None]
    rgb *= (1 - .45 * cavity)[:, None]
    iron = np.array([.55, .30, .14])
    # Staining is patchy along fractures (only part of each trace carries oxide), and the
    # stain colour itself varies from yellow-brown to dark grey-brown between scenes.
    patch = np.clip((value_noise(V * float(rng.uniform(.3, .9)), seed + 9) - .45) * 3, 0, 1)
    stain = stain * patch
    iron = np.array([.55, .32, .16]) * rng.uniform(.6, 1.1) + rng.normal(0, .04, 3)
    stain_strength = float(rng.uniform(.1, .45))
    rgb = rgb * (1 - stain_strength * stain[:, None]) + np.clip(iron, .05, .9) * (stain_strength * stain[:, None]) * (.8 + .4 * mott[:, None])
    rgb *= (1 + float(rng.uniform(.05, .25)) * streak)[:, None]
    # Cover colour: soil with patchy grass/shrub, blended at the cover edge.
    soil_rgb = np.array([[.30, .24, .17], [.38, .32, .24], [.26, .23, .19], [.42, .36, .27]])[rng.integers(4)] * rng.uniform(.8, 1.15)
    veg_rgb = np.array([[.16, .22, .08], [.24, .26, .12], [.30, .28, .16]])[rng.integers(3)] * rng.uniform(.8, 1.2)
    veg_amount = float(rng.uniform(0, .8))
    veg = np.clip((value_noise(V * 1.1, seed + 14) + .3 * value_noise(V * 6, seed + 15) - (1.1 - veg_amount)) * 4, 0, 1)
    cover_rgb = soil_rgb * (1 + .4 * fbm(V * 3, seed + 16, 4))[:, None]
    cover_rgb = cover_rgb * (1 - veg[:, None]) + veg_rgb * veg[:, None] * (1 + .5 * fbm(V * 9, seed + 17, 3))[:, None]
    rgb = rgb * (1 - cover[:, None]) + cover_rgb * cover[:, None]
    rgb = np.clip(rgb, .02, .95)
    cover_faces = (cover[F].mean(1) > .5) | ~rock

    refine_stats = None
    if truth is not None:
        # Faces whose vertices lie in different units straddle a contact: subdivide them until the
        # straddling band is ~1 cm, then label every face from v2 (vertex agreement + centroid).
        V2, F, (rgb, cover), (cover_faces, parent), face_keys, refine_stats = refine_contacts(
            V2, F, [rgb, cover], [cover_faces, parent], truth, levels=3)
        nv = len(V2)

    # Displace trace polylines with the same field (nearest welded vertex) for label projection.
    polylines = []
    dense = traces
    if truth is not None:
        from .truth import surface_intersections
        polylines = surface_intersections(V2, F, truth.interfaces)
        dense = []
    if dense:
        flat = np.concatenate(dense)
        idx = nearest_vertex(flat, V, max_edge)
        moved = flat + offset[idx]
        start = 0
        for pts, iid in zip(dense, pkg['segment_ids']):
            polylines.append((iid, moved[start:start + len(pts)]))
            start += len(pts)
    if truth is not None:
        # Groove/label alignment: distance from the final label lines to the groove centre lines.
        # Compare with the groove FLOOR centre line: the centre line carried by the same in-plane
        # carve that deepened the groove (labels are expected to lie on it, not on the rim line).
        centres = [c for c in groove_centres if len(c) > 1]
        if centres:
            flat = np.concatenate(centres)
            moved = flat + carve_final[nearest_vertex(flat, V_pre_carve, max_edge)]
            cut = np.cumsum([len(c) for c in centres])[:-1]
            centres = np.split(moved, cut)
        lab = np.concatenate([p for _, p in polylines]) if polylines else np.zeros((0, 3))
        if centres and len(lab):
            cs = np.concatenate([np.stack([c[:-1], c[1:]], 1) for c in centres])
            dist, _ = segment_distance(lab, cs)
            cls = np.concatenate([[line_classes.get(i, 'contact')] * len(q) for i, q in polylines])
            # Points within 10 cm of a polyline end (interface extent / clipping) reported separately.
            endish = np.concatenate([np.minimum(np.cumsum(np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1)]),
                                                np.cumsum(np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1)])[::-1]) < .1
                                     for _, q in polylines])
            summary = lambda m: {'n': int(m.sum()), 'median': float(np.median(dist[m])) if m.any() else None,
                                 'p95': float(np.percentile(dist[m], 95)) if m.any() else None}
            refine_stats = dict(refine_stats or {}, inplane=inplane, groove_label_offset_m={
                'median': float(np.median(dist)), 'p95': float(np.percentile(dist, 95)), 'max': float(dist.max()),
                'fault': summary((cls == 'fault') & ~endish), 'joint': summary((cls == 'joint') & ~endish),
                'near_ends': summary(endish)})
    stats = dict(vertices=int(nv), faces=int(len(F)), cover_fraction=float(cover_faces.mean()), remaining_folded_faces=folded_faces,
                 contact_refinement=refine_stats, fold_passes=fold_passes,
                 max_recess_m=float(recess.max()), joint_width_m=jw, joint_depth_m=jd,
                 fault_width_m=fw, fault_depth_m=fd, displacement_range_m=[float(disp.min()), float(disp.max())])
    area = np.linalg.norm(np.cross(V2[F[:, 1]] - V2[F[:, 0]], V2[F[:, 2]] - V2[F[:, 0]]), axis=1)
    keep = area > 1e-9
    if not keep.all():
        F, parent, cover_faces = F[keep], parent[keep], cover_faces[keep]
        if face_keys is not None:
            face_keys = [k for k, ok in zip(face_keys, keep) if ok]
        stats['dropped_degenerate_faces'] = int((~keep).sum())
        stats['faces'] = int(len(F))
    return dict(vertices=V2, faces=F, parent=parent, normals=vertex_normals(V2, F, outward, consistent), colors=rgb,
                polylines=polylines, cover_faces=cover_faces, cover=cover, face_keys=face_keys, stats=stats)


def split_contacts(V, F, vattrs, fattrs, truth, known=None, skip=None, tol_m=1e-5, end_tol=None, max_iters=40):
    """Exact unit boundaries on the display mesh: every edge whose end points belong to different v2 units (in a face with
    exactly two units) is split at the v2 boundary, bisected along the edge to an interval below ``tol_m`` (10 um; the
    v2 query takes an exact rational path for points within nanometres of an interface, so the bisection stops well
    before); the face is cut along the segment between its two crossing points into one triangle and a quadrilateral
    (shorter diagonal). The surface geometry is unchanged (new points on the edges) and a planar contact is followed
    exactly. Crossings within ``end_tol`` of an edge end (a vertex on the contact) are not split. Faces with three
    units, wedges (vertices of one unit, centroid of another), skipped faces (``skip``, e.g. B risers in a contact
    plane) and faces with an unsplit crossing take their centroid's unit. New faces are checked at their centroids.
    Returns (V, F, vattrs, fattrs, face_keys, stats) as refine_contacts."""
    # a crossing within the bisection precision of an edge end is the end vertex itself (lying on the boundary)
    end_tol = tol_m if end_tol is None else end_tol
    ids, rev = {}, []

    def kid_of(points, answer=None):
        status, b, u, l = answer if answer is not None else (truth.query(points, min_parallel=8000) if getattr(truth, 'workers', 1) > 1
                                                              else truth.query(points))
        out = np.empty(len(points), np.int64)
        for i, (st, bb, uu, ll) in enumerate(zip(status, b, u, l)):
            key = (bb, uu, ll) if st == 'occupied' else None
            if key not in ids:
                ids[key] = len(rev)
                rev.append(key)
            out[i] = ids[key]
        return out
    kid = kid_of(V, known)
    k0, k1, k2 = kid[F[:, 0]], kid[F[:, 1]], kid[F[:, 2]]
    same = (k0 == k1) & (k1 == k2)
    three = (k0 != k1) & (k1 != k2) & (k0 != k2)
    two = ~same & ~three
    if skip is not None:
        two &= ~skip
    ti = np.flatnonzero(two)
    T = F[ti]
    kt = kid[T]
    # rotate each face so the lone vertex (the one whose unit differs) is first
    lone = np.where(kt[:, 1] == kt[:, 2], 0, np.where(kt[:, 0] == kt[:, 2], 1, 2))
    r = np.arange(len(ti))
    R = T[r[:, None], (lone[:, None] + np.arange(3)[None]) % 3]
    ea = np.sort(np.c_[R[:, 0], R[:, 1]], 1)
    eb = np.sort(np.c_[R[:, 0], R[:, 2]], 1)
    ue, inv = np.unique(np.r_[ea, eb], axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    A, B = V[ue[:, 0]], V[ue[:, 1]]
    ka = kid[ue[:, 0]]
    lo, hi = np.zeros(len(ue)), np.ones(len(ue))
    L = np.linalg.norm(B - A, axis=1)
    for _ in range(max_iters):
        act = np.flatnonzero((hi - lo) * L > tol_m)
        if not len(act):
            break
        mid = .5 * (lo[act] + hi[act])
        km = kid_of(A[act] + mid[:, None] * (B[act] - A[act]))
        on_a = km == ka[act]
        lo[act] = np.where(on_a, mid, lo[act])
        hi[act] = np.where(on_a, hi[act], mid)
    sp = .5 * (lo + hi)
    valid = (sp * L > end_tol) & ((1 - sp) * L > end_tol)
    nv = len(V)
    new_id = np.full(len(ue), -1, np.int64)
    new_id[valid] = nv + np.arange(int(valid.sum()))
    V = np.vstack([V, (A + sp[:, None] * (B - A))[valid]])
    vattrs = [np.concatenate([x, (x[ue[valid, 0]] * (1 - sp[valid]).reshape((-1,) + (1,) * (x.ndim - 1))
                                  + x[ue[valid, 1]] * sp[valid].reshape((-1,) + (1,) * (x.ndim - 1)))]) for x in vattrs]
    pa, pb = new_id[inv[:len(ti)]], new_id[inv[len(ti):]]
    ia, ib = inv[:len(ti)], inv[len(ti):]
    ta = np.where(ue[ia, 0] == R[:, 0], sp[ia], 1 - sp[ia])
    tb = np.where(ue[ib, 0] == R[:, 0], sp[ib], 1 - sp[ib])
    va, vb = pa >= 0, pb >= 0
    at_r1 = ~va & (ta > .5)
    at_r2 = ~vb & (tb > .5)
    both = va & vb
    via1 = at_r1 & vb
    via2 = at_r2 & va
    touch = ~va & ~vb & (ta < .5) & (tb < .5)   # the lone vertex itself lies on the boundary: the face does not straddle
    r0, r1, r2 = R[:, 0], R[:, 1], R[:, 2]
    d1 = np.linalg.norm(V[np.maximum(pa, 0)] - V[r2], axis=1)
    d2 = np.linalg.norm(V[r1] - V[np.maximum(pb, 0)], axis=1)
    use1 = d1 <= d2

    def _width(a_, b_, c_):
        A_, B_, C_ = V[a_], V[b_], V[c_]
        ar = .5 * np.linalg.norm(np.cross(B_ - A_, C_ - A_), axis=1)
        pr = np.linalg.norm(B_ - A_, axis=1) + np.linalg.norm(C_ - B_, axis=1) + np.linalg.norm(A_ - C_, axis=1)
        return 2 * ar / np.maximum(pr, 1e-300)
    pa_, pb_ = np.maximum(pa, 0), np.maximum(pb, 0)
    w3 = np.minimum.reduce([_width(r0, pa_, pb_), np.where(use1, _width(pa_, r1, r2), _width(pa_, r1, pb_)),
                            np.where(use1, _width(pa_, r2, pb_), _width(r1, r2, pb_))])
    w2a = np.minimum(_width(r0, r1, pb_), _width(r1, r2, pb_))
    w2b = np.minimum(_width(r0, pa_, r2), _width(pa_, r1, r2))
    # no child thinner than 1 um: such a face stays whole and takes its centroid's unit (the neighbour's split point
    # on the shared edge is then a collinear T-junction, no gap)
    sliver = (both & (w3 < 1e-6)) | (via1 & (w2a < 1e-6)) | (via2 & (w2b < 1e-6))
    both, via1, via2 = both & ~sliver, via1 & ~sliver, via2 & ~sliver
    ok = both | via1 | via2
    # three children (both crossings inside their edges): lone triangle + quadrilateral (shorter diagonal)
    c3 = [np.c_[r0, pa, pb], np.where(use1[:, None], np.c_[pa, r1, r2], np.c_[pa, r1, pb]),
          np.where(use1[:, None], np.c_[pa, r2, pb], np.c_[r1, r2, pb])]
    kids_lone = [kid[r0], kid[r1], kid[r1]]
    parts, pkid, pidx = [], [], []
    for c_, k_ in zip(c3, kids_lone):
        parts.append(c_[both])
        pkid.append(k_[both])
        pidx.append(ti[both])
    # two children: the cut runs from a vertex on the boundary to the other edge's crossing
    parts += [np.c_[r0, r1, pb][via1], np.c_[r1, r2, pb][via1], np.c_[r0, pa, r2][via2], np.c_[pa, r1, r2][via2]]
    pkid += [kid[r0][via1], kid[r2][via1], kid[r0][via2], kid[r1][via2]]
    pidx += [ti[via1], ti[via1], ti[via2], ti[via2]]
    split_f = ti[ok]
    keep = np.ones(len(F), bool)
    keep[split_f] = False
    F2 = np.vstack([F[keep]] + parts)
    src = np.concatenate(pidx)
    fattrs = [np.concatenate([f[keep], f[src]]) for f in fattrs]
    fk = np.concatenate([np.where(same[keep], k0[keep], -1)] + pkid)
    # wedges and unsplit faces: centroid unit; every new face is checked at its centroid
    check = np.zeros(len(F2), bool)
    check[:int(keep.sum())] = same[keep]
    need_c = (fk < 0) | check
    nk = int(keep.sum())
    newf = np.arange(nk, len(F2))
    cen_q = np.r_[np.flatnonzero(need_c), newf]
    kc = kid_of(V[F2[cen_q]].mean(1))
    n_need = int(need_c.sum())
    wedge = (fk[cen_q[:n_need]] >= 0) & (kc[:n_need] != fk[cen_q[:n_need]])
    fk[cen_q[:n_need]] = kc[:n_need]
    child_bad = kc[n_need:] != fk[newf]
    fk[newf[child_bad]] = kc[n_need:][child_bad]
    face_keys = [rev[k] for k in fk]
    stats = {'method': 'exact split (bisected unit boundary on edges)', 'two_unit_faces': int(len(ti)), 'split_faces': int(len(split_f)),
             'new_vertices': int(valid.sum()), 'crossings_at_edge_end': int((~valid).sum()), 'three_unit_faces': int(three.sum()),
             'wedge_faces': int(wedge.sum()), 'child_centroid_mismatch': int(child_bad.sum()),
             'split_at_boundary_vertex': int((via1 | via2).sum()), 'lone_vertex_on_boundary': int(touch.sum()),
             'kept_whole_sliver_children': int(sliver.sum()),
             'final_straddling_faces': int((~ok & ~touch).sum() + three.sum() + wedge.sum() + child_bad.sum())}
    return V, F2, vattrs, fattrs, face_keys, stats


def refine_contacts(V, F, vattrs, fattrs, truth, levels=3, known=None, min_edge=None):
    """Subdivide faces that straddle a v2 contact, query the new vertices, repeat.

    A face straddles a contact if its vertices belong to different v2 units, or if its centroid does: a unit cut into
    pieces by a later fault is not convex, and a thin wedge of the neighbouring unit along the fault can cross a face
    whose three vertices all lie in the same unit. Midpoints lie on the parent edges, so the surface geometry is
    unchanged. Faces still straddling after ``levels`` subdivisions (~1 cm) take their centroid's unit. Returns
    refined (V, F, vattrs, fattrs, face_keys, stats). ``min_edge``: faces whose longest edge is shorter are not
    subdivided (micrometre faces of the B construction would become sub-micrometre slivers); they take their
    centroid's unit."""
    ids, rev = {}, []

    def kid_of(points, answer=None):
        status, b, u, l = answer if answer is not None else truth.query(points)
        out = np.empty(len(points), np.int64)
        for i, (st, bb, uu, ll) in enumerate(zip(status, b, u, l)):
            key = (bb, uu, ll) if st == 'occupied' else None
            if key not in ids:
                ids[key] = len(rev)
                rev.append(key)
            out[i] = ids[key]
        return out

    kid = kid_of(V, known)  # ``known``: an earlier truth.query(V) result, reused
    history, wedges = [], []
    centre_ok = np.zeros(len(F), bool)  # centroid already found in the vertices' unit

    def check_centres(F, kid, centre_ok):
        same = (kid[F[:, 0]] == kid[F[:, 1]]) & (kid[F[:, 1]] == kid[F[:, 2]])
        need = same & ~centre_ok
        if need.any():
            centre_ok[need] = kid_of(V[F[need]].mean(1)) == kid[F[need, 0]]
        return same

    for _ in range(levels):
        same = check_centres(F, kid, centre_ok)
        bnd = ~same | ~centre_ok
        if min_edge is not None:
            Tm = V[F]
            bnd &= np.linalg.norm(Tm - Tm[:, [1, 2, 0]], axis=2).max(1) >= min_edge
        history.append(int(bnd.sum()))
        wedges.append(int((same & ~centre_ok).sum()))
        if not bnd.any():
            break
        BF = F[bnd]
        e = np.sort(np.concatenate([BF[:, [0, 1]], BF[:, [1, 2]], BF[:, [2, 0]]]), 1)
        ue, inv = np.unique(e, axis=0, return_inverse=True)
        mid = len(V) + np.arange(len(ue))
        V = np.vstack([V, (V[ue[:, 0]] + V[ue[:, 1]]) / 2])
        vattrs = [np.concatenate([x, (x[ue[:, 0]] + x[ue[:, 1]]) / 2]) for x in vattrs]
        kid = np.concatenate([kid, kid_of(V[mid])])
        m01, m12, m20 = mid[inv.reshape(-1)].reshape(3, -1)
        v0, v1, v2 = BF[:, 0], BF[:, 1], BF[:, 2]
        children = np.stack([np.c_[v0, m01, m20], np.c_[m01, v1, m12], np.c_[m20, m12, v2], np.c_[m01, m12, m20]], 1)
        F = np.vstack([F[~bnd], children.reshape(-1, 3)])
        fattrs = [np.concatenate([f[~bnd], np.repeat(f[bnd], 4, axis=0)]) for f in fattrs]
        centre_ok = np.concatenate([centre_ok[~bnd], np.zeros(4 * len(BF), bool)])
    same = check_centres(F, kid, centre_ok)
    agree = same & centre_ok
    fk = kid[F[:, 0]].copy()
    if (~agree).any():
        fk[~agree] = kid_of(V[F[~agree]].mean(1))
    face_keys = [rev[k] for k in fk]
    stats = {'straddling_faces_per_level': history, 'wedge_faces_per_level': wedges, 'final_straddling_faces': int((~agree).sum())}
    return V, F, vattrs, fattrs, face_keys, stats
