"""Bluff morphology for the Virtual Outcrop Generator, ported from the seminar demo.

Method of ``outcrop.seminar.envelope`` / ``hero`` (copied, not imported, so the demo stays frozen):
a closed outcrop is the zero level of a smooth union of rounded boxes, sampled along rays from one
interior origin (star-shaped); multi-scale facet noise gives rock-like planar facets; recession is
added as an offset of the level surface and the surface is re-solved along the same rays.

Difference from the demo: every geological input is the v2 model of the scene. Retreat follows the
v2 lithology at each point (ramped across the exact contacts), undercuts sit on the weaker side of
v2 unit contacts, the fault damage zone and joint grooves follow the v2 fault/joint planes, and
labels are v2 queries at the FINAL surface positions (the demo labelled pre-erosion coordinates).
"""
import math

import numpy as np

from .display import fbm


def noise(x, z, seed=47):
    x, z = np.broadcast_arrays(x, z)
    i, j = np.floor(x), np.floor(z)
    u, v = x - i, z - j
    u = u * u * (3 - 2 * u)
    v = v * v * (3 - 2 * v)

    def h(a, b):
        q = np.sin(a * 127.1 + b * 311.7 + seed * 19.3) * 43758.5453
        return (q - np.floor(q)) * 2 - 1
    return (1 - v) * ((1 - u) * h(i, j) + u * h(i + 1, j)) + v * ((1 - u) * h(i, j + 1) + u * h(i + 1, j + 1))


def noise_c2(x, z, seed=47):
    """noise() with the quintic fade 6t^5 - 15t^4 + 10t^3: the smoothstep fade of noise() leaves curvature jumps on the
    lattice lines, which the refinement cannot resolve cheaply."""
    x, z = np.broadcast_arrays(x, z)
    i, j = np.floor(x), np.floor(z)
    u, v = x - i, z - j
    u = u * u * u * (u * (6 * u - 15) + 10)
    v = v * v * v * (v * (6 * v - 15) + 10)

    def h(a, b):
        q = np.sin(a * 127.1 + b * 311.7 + seed * 19.3) * 43758.5453
        return (q - np.floor(q)) * 2 - 1
    return (1 - v) * ((1 - u) * h(i, j) + u * h(i + 1, j)) + v * ((1 - u) * h(i, j + 1) + u * h(i + 1, j + 1))


def facet_noise(x, z, seed):
    """Piecewise planar relief (rock-like facets); callers domain-warp it to avoid a grid."""
    x, z = np.broadcast_arrays(x, z)
    i, j = np.floor(x), np.floor(z)
    u, v = x - i, z - j

    def h(a, b):
        q = np.sin(a * 127.1 + b * 311.7 + seed * 19.3) * 43758.5453
        return (q - np.floor(q)) * 2 - 1
    a, b, d, e = h(i, j), h(i + 1, j), h(i, j + 1), h(i + 1, j + 1)
    return np.where(u + v < 1, a + (b - a) * u + (d - a) * v, e + (d - e) * (1 - u) + (b - e) * (1 - v))


def facet_noise_c1(x, z, seed, w=.12):
    """facet_noise with rounded creases: C1 across cell edges (each cell coordinate is remapped with zero slope at the cell
    edges, linear in between) and across the cell diagonal (the max(0, u + v - 1) crease term is replaced by a
    quadratic soft-plus of half-width w inside the cell, blended back to the exact term within w / 2 of the cell edges
    so the values on the edges equal the neighbours'; a weak partial crease remains near the corners). Planar facets
    away from the creases."""
    x, z = np.broadcast_arrays(x, z)
    i, j = np.floor(x), np.floor(z)
    u, v = x - i, z - j
    sl = 1. / (1. - w)

    def g(t):
        return np.where(t < w, sl * t * t / (2 * w), np.where(t > 1 - w, 1 - sl * (1 - t) ** 2 / (2 * w), sl * (t - w / 2)))

    def psi(t):
        # back to the exact crease within w / 2 of the cell edges (w: residual corner crease max 1.03, w / 2: 0.46)
        a_ = np.clip(t / (.5 * w), 0, 1)
        b_ = np.clip((1 - t) / (.5 * w), 0, 1)
        return a_ * a_ * (3 - 2 * a_) * b_ * b_ * (3 - 2 * b_)

    def h(a_, b_):
        q = np.sin(a_ * 127.1 + b_ * 311.7 + seed * 19.3) * 43758.5453
        return (q - np.floor(q)) * 2 - 1
    a, b, d, e = h(i, j), h(i + 1, j), h(i, j + 1), h(i + 1, j + 1)
    U, Vv = g(u), g(v)
    xd = U + Vv - 1
    hard = np.maximum(xd, 0.)
    soft = np.where(xd >= w, xd, np.where(xd <= -w, 0., (xd + w) ** 2 / (4 * w)))
    return a + (b - a) * U + (d - a) * Vv + (e - b - d + a) * (hard + psi(u) * psi(v) * (soft - hard))


def soft_clip(x, lo=0., hi=1., w=.1):
    """C1 clamp to [lo, hi]: quadratic blends of half-width w at both ends."""
    x = np.asarray(x, float)
    y = np.clip(x, lo, hi)
    y = np.where(np.abs(x - lo) < w, lo + (x - lo + w) ** 2 / (4 * w), y)
    return np.where(np.abs(x - hi) < w, hi - (hi - x + w) ** 2 / (4 * w), y)


def pnorm_max(a, b, p=8.):
    """C1 union of non-negative fields (p-norm; equals max where one of them is 0, at most 2^(1/p) x max)."""
    a, b = np.maximum(a, 0.), np.maximum(b, 0.)
    mx = np.maximum(a, b)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(mx > 0, mx * ((a / np.where(mx > 0, mx, 1)) ** p + (b / np.where(mx > 0, mx, 1)) ** p) ** (1. / p), 0.)
    return r


def self_affine_waves(octaves, seed):
    """the random plane waves of self_affine_relief: per octave k, n wavenumbers log-uniform in [1 / lam_hi, 1 / lam_lo]
    (cycle/m), directions uniform on the sphere, phases uniform, each of amplitude rms sqrt(2 / n); from (seed, k) only."""
    K, ph, amp = [], [], []
    for k, (lam_lo, lam_hi, rms, n) in enumerate(octaves):
        n = int(n)
        g = np.random.default_rng([int(seed), 7757, k])
        kap = np.exp(np.log(1. / lam_hi) + g.uniform(size=n) * (np.log(1. / lam_lo) - np.log(1. / lam_hi)))
        z = g.uniform(-1., 1., n)
        phi = g.uniform(0., 2 * np.pi, n)
        d = np.stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z], 1)
        K.append(2 * np.pi * kap[:, None] * d)
        ph.append(g.uniform(0., 2 * np.pi, n))
        amp.append(np.full(n, float(rms) * np.sqrt(2. / n)))
    return np.concatenate(K), np.concatenate(ph), np.concatenate(amp)


def self_affine_relief(P, octaves, seed, chunk=32768):
    """Self-affine relief: spectral synthesis with random plane waves in 3D, sum a cos(2 pi kappa d . P + phase). A 3D field
    (independent of camera and surface parametrisation), C-infinity, evaluable exactly at any point; on any plane each
    wave keeps its amplitude and its wavelength can only grow (projection), so an octave's power stays in its band or
    the next longer one. octaves = [[lam_lo_m, lam_hi_m, rms_m, n_waves], ...] from the calibration (rock_face_02 scan
    spectrum in its band)."""
    P = np.asarray(P, float)
    K, ph, amp = self_affine_waves(octaves, seed)
    keep = amp != 0
    K, ph, amp = K[keep], ph[keep], amp[keep]
    out = np.zeros(len(P))
    for a in range(0, len(P), chunk):
        out[a:a + chunk] = np.cos(P[a:a + chunk] @ K.T + ph) @ amp
    return out


def compact(rho):
    r = np.clip(np.asarray(rho, float), 0, 1)
    return (1 - r) ** 4 * (4 * r + 1)


def envelope_distance(p, env):
    """Signed distance-like field of the smooth union of rounded boxes (negative inside)."""
    p = np.asarray(p, float)
    if env.get('rotation_deg', 0.):
        angle = math.radians(env['rotation_deg'])
        c, s = math.cos(angle), math.sin(angle)
        centre = np.asarray(env['origin_m'][:2])
        p = p.copy()
        p[:, :2] = (p[:, :2] - centre) @ np.array([[c, -s], [s, c]]) + centre
    result = np.full(len(p), 1e4)
    blend = env['blend_m']
    for mass in env['masses']:
        r = mass['rounding_m']
        q = np.abs(p - np.asarray(mass['center_m'])) - np.asarray(mass['half_size_m']) + r
        v = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0) - r
        h = np.maximum(blend - np.abs(result - v), 0) / blend
        result = np.minimum(result, v) - h * h * blend * .25
    return result


def radial_surface(direction, offset, env, iterations=20):
    """Points where envelope_distance + offset = 0 along rays from the interior origin."""
    origin = np.asarray(env['origin_m'])
    lo = np.zeros(len(direction))
    hi = np.full(len(direction), env['max_radius_m'])
    for _ in range(iterations):
        r = (lo + hi) * .5
        outside = envelope_distance(origin + direction * r[:, None], env) + offset > 0
        hi = np.where(outside, r, hi)
        lo = np.where(outside, lo, r)
    radius = (lo + hi) * .5
    return origin + direction * radius[:, None], radius


def surface_weights(p, env):
    """Tri-planar weights from the envelope gradient (the ray is not the normal on broad faces)."""
    g = []
    for axis in range(3):
        d = np.zeros(3)
        d[axis] = .025
        g.append(envelope_distance(p + d, env) - envelope_distance(p - d, env))
    w = np.abs(np.column_stack(g)) ** 4
    return w / np.maximum(w.sum(1, keepdims=True), 1e-20)


def ray_grid(n_az, n_lat, stretch):
    t = np.arange(n_az) * 2 * np.pi / n_az
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]
    sx, sy, sz = stretch
    d = np.stack((np.cos(t[:, None]) * np.cos(lat)[None] * sx, np.sin(t[:, None]) * np.cos(lat)[None] * sy,
                  np.broadcast_to(np.sin(lat)[None] * sz, (n_az, n_lat))), -1).reshape(-1, 3)
    d = np.vstack((d, [[0, 0, 1], [0, 0, -1]]))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ids = np.arange(n_az * n_lat).reshape(n_az, n_lat)
    a = ids[:, :-1].ravel()
    b = np.roll(ids, -1, axis=0)[:, :-1].ravel()
    faces = np.vstack((np.column_stack((a, b, a + 1)), np.column_stack((b, b + 1, a + 1)),
                       np.column_stack((ids[:, -1], np.roll(ids[:, -1], -1), np.full(n_az, n_az * n_lat))),
                       np.column_stack((np.roll(ids[:, 0], -1), ids[:, 0], np.full(n_az, n_az * n_lat + 1)))))
    return d, faces


# ---------------------------------------------------------------- build
def build_bluff(pkg, truth, rng, *, env=None, samples=(1800, 520), clay=False, debug=None):
    """Bluff display mesh (legacy entry point): the bluff exposure patch through the shared surface processes
    (processes.erode)."""
    from .exposure import build_surface
    from .processes import erode
    env = env or pkg['model']['metadata']['bluff']
    surf = build_surface({'kind': 'bluff', 'view_direction': [0., -1., 0.], **env}, samples=samples)
    return erode(surf, pkg['model'], truth, rng, clay=clay, debug=debug)


def ground_mesh(env, rng, seed, half=60., spacing=.75, far=3000.):
    """Soil plain around the bluff: gentle long-wave relief, slightly raised against the foot.
    Dense within +-``half`` m, then geometrically coarser out to +-``far`` m (no visible edge)."""
    origin = np.asarray(env['origin_m'])
    inner = np.linspace(-half, half, int(2 * half / spacing) + 1)
    outer = half * 1.15 ** np.arange(1, int(np.log(far / half) / np.log(1.15)) + 2)
    g = np.concatenate([-outer[::-1], inner, outer])
    n = len(g)
    X, Y = np.meshgrid(origin[0] + g, origin[1] + g)
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    zg = env['ground_z_m'] + .5 * fbm(P * .02, seed + 40, 4) + .12 * fbm(P * .15, seed + 41, 3)
    # Apron: ground rises gently against the rock (wash and soil banked at the foot).
    apron = float(rng.uniform(.2, .6))
    if env['masses']:
        dist = envelope_distance(np.c_[P[:, :2], np.full(len(P), env['ground_z_m'] + .5)], env)
        zg += apron * np.exp(-np.maximum(dist, 0) / 3.)
    P[:, 2] = zg
    ids = np.arange(n * n).reshape(n, n)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, 1:].ravel(), ids[1:, :-1].ravel()
    F = np.vstack([np.c_[a, b, c], np.c_[a, c, d]])
    return P, F
