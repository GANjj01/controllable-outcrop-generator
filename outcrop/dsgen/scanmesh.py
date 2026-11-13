"""Donor surface from the scan's own triangles.

Window frame (x1, x2, h): x1 along the donor strata, x2 up the face, h along the window normal (outward). The donor's
front-facing triangles (normal . n > FRONT_MIN) give the visible envelope h_env(p) = max of their planes over the
triangles containing p (exact, piecewise planar). Where the envelope drops across a front triangle's edge (a ledge lip
/ overhang silhouette: the next surface is lower, or none), the overhang cannot stay (the B blocks need a height field
along m): it is cut back to a steep wall descending from the lip at WALL_DEG, h_wall(p) = h_lip(foot) - tan(WALL_DEG)
d for points p outside the lip segment (d = distance to the segment, foot = nearest point on it), and the surface is h
= max(h_env, h_wall). The lip line is a donor edge; the wall's foot on the lower surface is computed per (lip segment,
lower triangle) as the intersection line of two planes (the wall is planar where its foot lies inside the segment).
Regions with neither (no front triangle and no wall) are reported and filled harmonically on the target mesh."""
import numpy as np

FRONT_MIN = .02
WALL_DEG = 75.


class Donor:
    def __init__(self, P, F, frame, W, wall_deg=WALL_DEG, front_min=FRONT_MIN, bounds=None):
        """P, F: scan mesh (z-up frame); frame: (c, d1, d2, n) of the window. ``bounds``: (x0, x1, y0, y1) of a rectangular
        window in (d1, d2) instead of the square of side W."""
        c, d1, d2, n = (np.asarray(x, float) for x in frame)
        Q = np.c_[(P - c) @ d1, (P - c) @ d2, (P - c) @ n]
        T = Q[F]
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        a2 = np.linalg.norm(fn, axis=1)
        fz = fn[:, 2] / np.maximum(a2, 1e-30)
        lo, hi = T[:, :, :2].min(1), T[:, :, :2].max(1)
        if bounds is None:
            win = (hi > -W / 2 - .05).all(1) & (lo < W / 2 + .05).all(1)
        else:
            x0, x1, y0, y1 = bounds
            win = (hi > np.array([x0, y0]) - .05).all(1) & (lo < np.array([x1, y1]) + .05).all(1)
            self.bounds = tuple(float(v) for v in bounds)
        self.W, self.tan = W, np.tan(np.radians(wall_deg))
        keep = np.nonzero(win & (fz > front_min))[0]
        self.tri_ids = keep
        self.T = T[keep]
        self.lo, self.hi = lo[keep], hi[keep]
        # plane h = A x1 + B x2 + C per front triangle
        t = self.T
        M = np.stack([t[:, :, 0], t[:, :, 1], np.ones((len(t), 3))], -1)
        self.abc = np.linalg.solve(M, t[:, :, 2:3])[:, :, 0]
        self.cell = .01
        self._grid()
        self.lips = self._lips()

    def _grid(self):
        g = {}
        k0 = np.floor(self.lo / self.cell).astype(int)
        k1 = np.floor(self.hi / self.cell).astype(int)
        for i, (a, b) in enumerate(zip(k0, k1)):
            for ix in range(a[0], b[0] + 1):
                for iy in range(a[1], b[1] + 1):
                    g.setdefault((ix, iy), []).append(i)
        self.grid = {k: np.array(v) for k, v in g.items()}

    def candidates(self, p):
        return self.grid.get((int(np.floor(p[0] / self.cell)), int(np.floor(p[1] / self.cell))), np.zeros(0, int))

    def env(self, X, eps=1e-10):
        """Envelope height and owning front triangle (-1 / -inf where no front triangle)."""
        X = np.atleast_2d(X)
        h = np.full(len(X), -np.inf)
        own = np.full(len(X), -1)
        keys = np.floor(X / self.cell).astype(int)
        order = np.lexsort(keys.T)
        ks = keys[order]
        starts = np.r_[0, np.nonzero((np.diff(ks, axis=0) != 0).any(1))[0] + 1, len(ks)]
        for g0, g1 in zip(starts[:-1], starts[1:]):
            cand = self.grid.get(tuple(ks[g0]))
            if cand is None:
                continue
            idx = order[g0:g1]
            p = X[idx]
            t = self.T[cand]
            ax, ay, bx, by, qx, qy = t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1], t[:, 2, 0], t[:, 2, 1]
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            l1 = ((by - qy)[None] * (p[:, :1] - qx[None]) + (qx - bx)[None] * (p[:, 1:] - qy[None])) / den[None]
            l2 = ((qy - ay)[None] * (p[:, :1] - qx[None]) + (ax - qx)[None] * (p[:, 1:] - qy[None])) / den[None]
            inside = (l1 >= -eps) & (l2 >= -eps) & (1 - l1 - l2 >= -eps)
            hv = self.abc[cand, 0][None] * p[:, :1] + self.abc[cand, 1][None] * p[:, 1:] + self.abc[cand, 2][None]
            hv = np.where(inside, hv, -np.inf)
            j = hv.argmax(1)
            hb = hv[np.arange(len(idx)), j]
            better = hb > h[idx]
            h[idx[better]] = hb[better]
            own[idx[better]] = cand[j[better]]
        return h, own

    def _lips(self, probe=2e-4, drop=1e-3):
        """Front-triangle edges across which the envelope drops by > drop (or ends): (a, b, ha, hb,
        outward normal, triangle)."""
        lips = []
        T = self.T
        A, Bp, Cn, owner = [], [], [], []
        for i in range(len(T)):
            for j in range(3):
                a, b, o = T[i, j], T[i, (j + 1) % 3], T[i, (j + 2) % 3]
                e = b[:2] - a[:2]
                L = np.linalg.norm(e)
                if L < 1e-9:
                    continue
                nrm = np.array([e[1], -e[0]]) / L
                if nrm @ (o[:2] - a[:2]) > 0:
                    nrm = -nrm                      # outward of triangle i
                A.append(a)
                Bp.append(b)
                Cn.append(nrm)
                owner.append(i)
        A, Bp, Cn, owner = np.array(A), np.array(Bp), np.array(Cn), np.array(owner)
        S = []
        for s in (.25, .5, .75):
            S.append(A[:, :2] + s * (Bp[:, :2] - A[:, :2]) + probe * Cn)
        hs = [self.env(x)[0] for x in S]
        hup = [A[:, 2] + s * (Bp[:, 2] - A[:, 2]) for s in (.25, .5, .75)]
        is_lip = np.zeros(len(A), bool)
        for x, he, hu in zip(S, hs, hup):
            hpl = (self.abc[owner, 0] * x[:, 0] + self.abc[owner, 1] * x[:, 1] + self.abc[owner, 2])
            is_lip |= he < hpl - drop
        k = np.nonzero(is_lip)[0]
        # one record per geometric edge (both triangles may report a shared edge only if both drop)
        self.lip_A, self.lip_B, self.lip_n, self.lip_owner = A[k], Bp[k], Cn[k], owner[k]
        return len(k)

    def wall(self, X):
        """max over lip segments of h_lip(foot) - tan * d, for X outside the segment (d > 0)."""
        X = np.atleast_2d(X)
        out = np.full(len(X), -np.inf)
        if not len(self.lip_A):
            return out
        if not hasattr(self, 'lip_grid'):
            # a wall can only rise above the envelope within reach = (highest lip - lowest donor
            # point) / tan of its lip; lips are indexed in cells of 2 cm dilated by that reach
            hi_ = max(self.lip_A[:, 2].max(), self.lip_B[:, 2].max())
            self.wall_reach = float(hi_ - self.T[:, :, 2].min()) / self.tan + 1e-3
            g = {}
            c = .02
            lo_ = np.floor((np.minimum(self.lip_A[:, :2], self.lip_B[:, :2]) - self.wall_reach) / c).astype(int)
            hi2 = np.floor((np.maximum(self.lip_A[:, :2], self.lip_B[:, :2]) + self.wall_reach) / c).astype(int)
            for i, (l_, h_) in enumerate(zip(lo_, hi2)):
                for ix in range(l_[0], h_[0] + 1):
                    for iy in range(l_[1], h_[1] + 1):
                        g.setdefault((ix, iy), []).append(i)
            self.lip_grid = {k: np.array(v) for k, v in g.items()}
            self.lip_cell = c
        if not hasattr(self, 'lip_xy'):
            # component instead of (points, candidates, 2) arrays reduced over the last axis; the element-wise
            # operations and their order are those of the vector form, so the result is bitwise the same
            ax, ay, az = self.lip_A[:, 0], self.lip_A[:, 1], self.lip_A[:, 2]
            ex, ey = self.lip_B[:, 0] - ax, self.lip_B[:, 1] - ay
            self.lip_xy = (ax, ay, az, ex, ey, self.lip_B[:, 2] - az, ex * ex + ey * ey, self.lip_n[:, 0], self.lip_n[:, 1])
        ax, ay, az, ex, ey, ez, L2, nx, ny = self.lip_xy
        keys = np.floor(X / self.lip_cell).astype(int)
        order = np.lexsort(keys.T)
        ks = keys[order]
        starts = np.r_[0, np.nonzero((np.diff(ks, axis=0) != 0).any(1))[0] + 1, len(ks)]
        for g0, g1 in zip(starts[:-1], starts[1:]):
            cand = self.lip_grid.get(tuple(ks[g0]))
            if cand is None:
                continue
            idx = order[g0:g1]
            px, py = X[idx, 0:1], X[idx, 1:2]
            cx, cy, ux, uy = ax[cand], ay[cand], ex[cand], ey[cand]
            dx, dy = px - cx, py - cy
            u = np.clip((dx * ux + dy * uy) / L2[cand], 0, 1)
            fx, fy = px - (cx + u * ux), py - (cy + u * uy)
            d = np.sqrt(fx * fx + fy * fy)
            side = dx * nx[cand] + dy * ny[cand] > 0
            hw = np.where(side, az[cand] + u * ez[cand] - self.tan * d, -np.inf)
            out[idx] = hw.max(1)
        return out

    def height(self, X):
        he, own = self.env(X)
        hw = self.wall(X)
        return np.maximum(he, hw), own, he, hw


def simplify(P, tol):
    """Douglas-Peucker: drop polyline vertices within tol of the chord."""
    if len(P) <= 2:
        return P
    a, b = P[0], P[-1]
    e = b - a
    L = np.linalg.norm(e)
    d = np.abs(np.cross(e, P - a)) / L if L > 0 else np.linalg.norm(P - a, axis=1)
    k = int(np.argmax(d))
    if d[k] <= tol:
        return np.array([a, b])
    return np.vstack([simplify(P[:k + 1], tol)[:-1], simplify(P[k:], tol)])


def wall_foot_creases(don, step=1e-3, dmax=.06, segments=None):
    """Lines where a lip wall meets the lower envelope (h_wall = h_env), as 2D polylines: per (lip
    segment s, front triangle t) the intersection of the wall plane (foot inside the segment) or cone
    (foot at an end point, sampled every 3 deg) with t's plane, kept where wall s is the winning wall,
    t owns the envelope and the two heights agree (1e-6 m)."""
    out = []
    a3, b3, nn = don.lip_A, don.lip_B, don.lip_n
    tanv = don.tan

    def valid(P, s, t):
        if not len(P):
            return np.zeros(0, bool)
        he, own = don.env(P)
        hw = don.wall(P)
        e = b3[s, :2] - a3[s, :2]
        L2 = e @ e
        u = np.clip(((P - a3[s, :2]) @ e) / L2, 0, 1)
        foot = a3[s, :2] + u[:, None] * e
        hs = a3[s, 2] + u * (b3[s, 2] - a3[s, 2]) - tanv * np.linalg.norm(P - foot, axis=1)
        return (own == t) & (np.abs(he - hw) < 1e-6) & (np.abs(hs - hw) < 1e-6)

    def runs(P, ok):
        i = 0
        while i < len(P):
            if not ok[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(P) and ok[j + 1]:
                j += 1
            if j > i:
                out.append(simplify(P[i:j + 1], 1e-5))
            i = j + 1

    for s in (range(len(a3)) if segments is None else segments):
        a, b, n = a3[s], b3[s], nn[s]
        e = b[:2] - a[:2]
        L = np.linalg.norm(e)
        eu = e / L
        lo = np.minimum(a[:2], b[:2]) - dmax
        hi = np.maximum(a[:2], b[:2]) + dmax
        cand = np.nonzero((don.hi >= lo).all(1) & (don.lo <= hi).all(1))[0]
        hmax = max(a[2], b[2])
        for t in cand:
            A, B, C = don.abc[t]
            # planar part: p = a + x eu + y n (0 <= x <= L, y > 0): wall = ha + x (hb - ha) / L - tan y
            # g(x, y) = wall - plane = g0 + gx x + gy y
            p0 = a[:2]
            g0 = a[2] - (A * p0[0] + B * p0[1] + C)
            gx = (b[2] - a[2]) / L - (A * eu[0] + B * eu[1])
            gy = -tanv - (A * n[0] + B * n[1])
            if abs(gy) > 1e-12:
                xs = np.linspace(0, L, max(int(L / step), 1) + 1)
                ys = -(g0 + gx * xs) / gy
                ok = (ys > 0) & (ys < dmax)
                if ok.any():
                    P = p0[None] + xs[:, None] * eu[None] + ys[:, None] * n[None]
                    runs(P, ok & valid(P, s, t))
            # caps at both ends: cone h = h_end - tan r
            for end, sgn in ((a, -1.), (b, 1.)):
                ang0 = np.arctan2(n[1], n[0])
                phis = np.radians(np.arange(0, 91, 3)) * (-sgn if np.cross(eu, n) > 0 else sgn)
                dirs = np.stack([np.cos(ang0 + phis), np.sin(ang0 + phis)], 1)
                ge = end[2] - (A * end[0] + B * end[1] + C)
                den = tanv + dirs @ np.array([A, B])
                r = np.where(den > 1e-9, ge / np.maximum(den, 1e-12), -1)
                ok = (r > 0) & (r < dmax)
                if ok.any():
                    P = end[:2][None] + np.maximum(r, 0)[:, None] * dirs
                    runs(P, ok & valid(P, s, t))
    return out


def gaussian_lowpass(h, step, sigma):
    pad = int(np.ceil(4 * sigma / step))
    hp = np.pad(h, pad, mode='reflect')
    ky = np.fft.fftfreq(hp.shape[0], step)
    kx = np.fft.rfftfreq(hp.shape[1], step)
    G = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (ky[:, None] ** 2 + kx[None, :] ** 2))
    return np.fft.irfft2(np.fft.rfft2(hp) * G, s=hp.shape)[pad:-pad, pad:-pad]


def load(path):
    z = np.load(path)
    d = Donor.__new__(Donor)
    d.W, d.tan = float(z['W']), float(np.tan(np.radians(float(z['wall_deg']))))
    if 'bounds' in z.files:
        d.bounds = tuple(float(v) for v in z['bounds'])
    d.T, d.abc = z['T'], z['abc']
    d.lo, d.hi = d.T[:, :, :2].min(1), d.T[:, :, :2].max(1)
    d.cell = .01
    d._grid()
    d.lip_A, d.lip_B, d.lip_n, d.lip_owner = z['lip_A'], z['lip_B'], z['lip_n'], z['lip_owner']
    d.lips = len(d.lip_A)
    pts, lens = z['crease_points'], z['crease_lengths']
    d.creases = np.split(pts, np.cumsum(lens)[:-1]) if len(lens) else []
    d.edges = z['edges']
    d.lowpass, d.lp_step = z['lowpass'], float(z['step'])
    return d


def clip_segment(a, b, half):
    """Part of segment a-b inside the square |x|, |y| <= half (Liang-Barsky), or None."""
    t0, t1 = 0., 1.
    d = b - a
    for p_, q_ in ((-d[0], a[0] + half), (d[0], half - a[0]), (-d[1], a[1] + half), (d[1], half - a[1])):
        if abs(p_) < 1e-18:
            if q_ < 0:
                return None
            continue
        r = q_ / p_
        if p_ < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    if t1 - t0 < 1e-12:
        return None
    return a + t0 * d, a + t1 * d


class DonorWindow:
    """Translation only: the window keeps the atlas axes (no rotation, no mirror). The low-pass grid is the atlas low-pass
    cropped to the window (offset - W/2 on the atlas grid), so the detail is the atlas detail band, not a re-filtered
    window."""

    def __init__(self, atlas, offset, W):
        self.atlas, self.W = atlas, float(W)
        self.offset = np.asarray(offset, float)
        x0, x1, y0, y1 = atlas.bounds
        s = atlas.lp_step
        c0 = (self.offset[0] - self.W / 2 - x0) / s
        r0 = (y1 - (self.offset[1] + self.W / 2)) / s
        n = self.W / s
        if max(abs(c0 - round(c0)), abs(r0 - round(r0)), abs(n - round(n))) > 1e-6:
            raise ValueError('donor window not aligned to the atlas low-pass grid')
        c0, r0, n = int(round(c0)), int(round(r0)), int(round(n))
        if c0 < 0 or r0 < 0 or c0 + n > atlas.lowpass.shape[1] or r0 + n > atlas.lowpass.shape[0]:
            raise ValueError('donor window outside the atlas')
        self.lowpass, self.lp_step = atlas.lowpass[r0:r0 + n, c0:c0 + n], s
        reach = self.W / 2 + .2
        E = np.asarray(atlas.edges, float).reshape(-1, 2, 2) - self.offset
        near = (np.abs(E).max(1) <= reach).any(1) if len(E) else np.zeros(0, bool)
        self.edges = E[near]
        self.creases = [c - self.offset for c in atlas.creases if len(c) and (np.abs(c - self.offset).max(1) <= reach).any()]
        self.lips = atlas.lips

    def height(self, X):
        return self.atlas.height(np.atleast_2d(X) + self.offset)
