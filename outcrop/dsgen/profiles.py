"""Cross-section profiles parameterised by arc length, for sheet exposures (escarpment, roadcut).

A profile is a chain of segments in the (w, z) plane of one column (w across the exposure, z up):
straight lines at a given angle and circular arcs that turn the angle to a new value, so the
curve is C1 everywhere and sampled at even arc-length spacing on steep faces as well as on flats.
Every column has the same segment chain with its own lengths, radii and angles (smooth functions
of the along-strike coordinate), so one structured (row, column) grid covers the whole sheet.
"""
import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])


def line(length, angle, edge, name=''):
    return {'kind': 'line', 'length': np.asarray(length, float), 'angle': np.asarray(angle, float), 'edge': float(edge), 'name': name}


def arc(radius, to, edge, name=''):
    return {'kind': 'arc', 'radius': np.asarray(radius, float), 'to': np.asarray(to, float), 'edge': float(edge), 'name': name}


def end_state(segments, w0, z0, phi0):
    """(w, z, angle) at the end of a chain; used to solve segment lengths (e.g. face height)."""
    w, z, phi = np.asarray(w0, float), np.asarray(z0, float), np.asarray(phi0, float)
    for seg in segments:
        w, z, phi = _advance(seg, w, z, phi, 1.)
    return w, z, phi


def _length(seg, phi):
    if seg['kind'] == 'line':
        return seg['length'] + 0 * phi
    return seg['radius'] * np.abs(seg['to'] - phi)


def _advance(seg, w, z, phi, t):
    """Point at fraction t (scalar or array) of a segment that starts at (w, z, phi)."""
    if seg['kind'] == 'line':
        a = seg['angle'] + 0 * phi
        s = seg['length'] * t
        return w + s * np.cos(a), z + s * np.sin(a), a + 0 * s
    b = phi + (seg['to'] - phi) * t
    sign = np.where(seg['to'] >= phi, 1., -1.)
    r = seg['radius'] * sign  # 1 / curvature
    return w + r * (np.sin(b) - np.sin(phi)), z + r * (np.cos(phi) - np.cos(b)), b


def trace(segments, w0, z0, phi0, *, smooth_m=1.):
    """Sample a chain for every column. Each segment gets ceil(max length / edge) rows, so the
    spacing is at most its edge in every column. Returns dict with arrays (rows, columns):
    w, z, angle (exact tangent angle), s (arc length), smooth_angle (tangent angle smoothed with a
    Gaussian of ``smooth_m`` along the arc, for the offset field), plus ``rows`` (segment name per
    row) and ``segment_rows`` (start row of every segment)."""
    w, z, phi = (np.asarray(v, float) for v in (w0, z0, phi0))
    s = np.zeros_like(w)
    W, Z, A, S, names, starts = [], [], [], [], [], []
    breaks = []  # (arc length of the break, change of curvature) per column
    for seg in segments:
        L = _length(seg, phi)
        if (L <= 1e-6).any():
            raise ValueError('profile segment %r has zero length in some column' % seg['name'])
        k = max(1, int(math.ceil(float(L.max()) / seg['edge'])))
        t = (np.arange(k) / k)[:, None]
        wr, zr, ar = _advance(seg, w[None], z[None], phi[None], t)
        starts.append(len(names))
        W.append(wr)
        Z.append(zr)
        A.append(ar)
        S.append(s[None] + L[None] * t)
        names += [seg['name']] * k
        if seg['kind'] == 'arc':
            kappa = (seg['to'] - phi) / L
            breaks += [(s.copy(), kappa), (s + L, -kappa)]
        w, z, phi = _advance(seg, w, z, phi, 1.)
        s = s + L
    W.append(w[None])
    Z.append(z[None])
    A.append(phi[None])
    S.append(s[None])
    names.append(segments[-1]['name'])
    out = {'w': np.vstack(W), 'z': np.vstack(Z), 'angle': np.vstack(A), 's': np.vstack(S),
           'rows': np.array(names), 'segment_rows': dict(zip([g['name'] for g in segments], starts))}
    # Gaussian-smoothed angle in closed form: the angle is piecewise linear in s, i.e. a sum of
    # ramps at the curvature breaks; a smoothed ramp is x * Phi(x / sigma) + sigma * phi(x / sigma).
    sm = np.zeros_like(out['s']) + np.broadcast_to(np.asarray(phi0, float), out['s'].shape[1:])[None]
    for sk, dk in breaks:
        x = (out['s'] - sk[None]) / smooth_m
        ramp = smooth_m * (x * .5 * (1 + _erf(x / math.sqrt(2))) + np.exp(-.5 * x * x) / math.sqrt(2 * math.pi))
        sm += dk[None] * ramp
    out['smooth_angle'] = sm
    return out


def grid_faces(rows, cols):
    """Two triangles per cell of a (rows, cols) vertex grid, wound so that the normal is
    d/dcol x d/drow (outward for a profile running from the front to the back)."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
    a = (r * cols + c).ravel()
    b, cc, d = a + 1, a + cols + 1, a + cols
    return np.concatenate([np.c_[a, b, cc], np.c_[a, cc, d]])


def graded(start, end, fine, coarse, growth=1.15):
    """Coordinates from start to end (end > start) beginning at spacing ``fine`` and growing
    geometrically up to ``coarse``."""
    xs, x, h = [start], start, fine
    while x < end - 1e-9:
        h = min(h * growth, coarse)
        x = min(x + h, end)
        xs.append(x)
    return np.array(xs)
