"""Panel-based fracture cavities on the source fracture surface.

A source fracture is an XY polyline at zref extruded along d. That cylinder is developable: with d^ = d/|d|, sigma
(arc length of the polyline projected onto the plane normal to d^) and tau = x.d^ are exact isometric coordinates, and
every fold is a line sigma = const. Each exposure path lies on its surface (residual ~1e-14 m).

The cavity of all paths on one surface is ONE closed mesh built without any 3D Boolean: * pieces in the unrolled
plane: one quad per refined span (outside margin to depth, the old loft's footprint) and one chord wedge per turning
node on its gap side (the R2 wedge); * footprint F = 2D union of the pieces (Clipper2 via CrossSection),
finite-support clips as sigma half-planes; * wall offsets along the panel normal use union semantics: L = min, R = max
over the pieces covering a point, each piece reproducing the old loft's piecewise-linear wall (its triangle diagonals
included); * a thin prism over F, refined to edge length h, maps bottom -> left wall, top -> right wall, sides -> the
footprint boundary, by X = S(sigma, tau) + o N(sigma), N rotating linearly about d^ within +-e_f of a fold (e_f =
max|o| phi_f keeps |o| dalpha/dsigma <= 1/2, so the offset sheets cannot fold over)."""
import json

import manifold3d as md
import numpy as np

TOL = 2e-6            # coverage tolerance: >= footprint simplification (1e-6) + Clipper2 grid (~1e-8)
EDGE_TOL = 1e-5       # boundary-edge to piece-edge match
SIMPLIFY_M = 1e-6
MIN_TURN_RAD = 1e-6


class SurfaceChart:
    """Exact unrolled coordinates of one extruded source polyline."""

    def __init__(self, points_2d, zref, direction):
        d = np.asarray(direction, float)
        self.dh = d / np.linalg.norm(d)
        self.d = d
        self.zref = float(zref)
        q = np.asarray(points_2d, float)
        self.xy = q
        self.Q = np.c_[q, np.full(len(q), self.zref)]
        self.e = np.diff(self.Q, axis=0)
        perp = self.e - (self.e @ self.dh)[:, None] * self.dh
        self.length = np.linalg.norm(perp, axis=1)
        if np.any(self.length <= 0):
            raise ValueError('source polyline segment parallel to the extrusion direction')
        self.u = perp / self.length[:, None]
        self.sigma = np.r_[0., np.cumsum(self.length)]
        n = np.cross(self.u, self.dh)
        self.n = n / np.linalg.norm(n, axis=1)[:, None]
        cos = np.clip(np.einsum('ij,ij->i', self.n[:-1], self.n[1:]), -1., 1.)
        self.fold_rad = np.arccos(cos)

    def locate(self, points):
        """3D surface points -> (sigma, tau), panel index and footprint residual."""
        p = np.atleast_2d(np.asarray(points, float))
        xy = p[:, :2] - ((p[:, 2] - self.zref) / self.d[2])[:, None] * self.d[:2]
        start, edge = self.xy[:-1], np.diff(self.xy, axis=0)
        f = np.clip(np.einsum('nsi,si->ns', xy[:, None] - start, edge) / np.einsum('si,si->s', edge, edge), 0., 1.)
        dist = np.linalg.norm(xy[:, None] - (start + f[:, :, None] * edge), axis=2)
        i = np.argmin(dist, axis=1)
        frac = f[np.arange(len(p)), i]
        sigma = self.sigma[i] + frac * self.length[i]
        return np.c_[sigma, p @ self.dh], i, dist[np.arange(len(p)), i]

    def panel(self, sigma):
        return np.clip(np.searchsorted(self.sigma, sigma, side='right') - 1, 0, len(self.length) - 1)

    def surface(self, st):
        st = np.atleast_2d(st)
        i = self.panel(st[:, 0])
        alpha = (st[:, 0] - self.sigma[i]) / self.length[i]
        base = self.Q[i] + alpha[:, None] * self.e[i]
        return base + (st[:, 1] - base @ self.dh)[:, None] * self.dh

    def vector_2d(self, vectors, panels):
        return np.c_[np.einsum('ij,ij->i', vectors, self.u[panels]), vectors @ self.dh]

    def fold_zones(self, max_offset):
        """Half-widths e_f = max|o| phi_f; adjacent zones must stay inside half the neighbouring panels.

        Angle fallback: arccos of a normal dot product that rounds to 1 - 1 ulp gives 1.49e-8 rad, the smallest nonzero
        arccos, for a collinear source vertex (atan2 ~1e-16), and the rotation by that angle misses the next normal by
        1.5e-8 > 1e-9. Only where the rotation check fails (the old code raised there), that fold's angle is
        recomputed as atan2(|axis.n_f+1|, n_f.n_f+1): <= 1e-12 -> no fold (the threshold above), else the zone uses it
        and both checks run again (fail-closed). Folds passing with arccos are unchanged; self.fold_angle_fallback
        (set only when used) records the folds of the latest call.

        Terminal-panel zones: the half-panel limit keeps the zones of two neighbouring folds apart. A trace truncated at
        another fracture can end with a very short panel (1.8 mm - 1.8 cm on flat outcrops) behind a fold; no other
        fold lies beyond it. Only where the old limit fails: if exactly one of the two panels is a terminal panel
        (first or last) and the zone fits in half of the other, interior, panel, the zone is accepted and may run past
        the trace end (no source surface there). Every other case still raises. self.fold_zone_terminal (set only when
        used) records those folds."""
        zones, fallback, terminal = [], [], []
        last = len(self.length) - 1

        def fits(f, e):
            if e < .5 * min(self.length[f], self.length[f + 1]):
                return True, None
            ends = [k for k in (f, f + 1) if k in (0, last)]
            inner = [k for k in (f, f + 1) if k not in (0, last)]
            if len(ends) != 1 or len(inner) != 1 or not e < .5 * self.length[inner[0]]:
                return False, None
            k, L = ends[0], float(self.length[ends[0]])
            return True, dict(fold=f, sigma=float(self.sigma[f + 1]), half_width_m=float(e), terminal_panel=int(k), terminal_length_m=L,
                              interior_length_m=float(self.length[inner[0]]), beyond_trace_end_m=float(max(0., e - L)))
        for f, phi in enumerate(self.fold_rad):
            if phi <= 1e-12:
                continue
            e = max_offset * phi
            ok, term = fits(f, e)
            if not ok:
                raise ValueError('fold transition zone exceeds half an adjacent source panel')
            axis = np.cross(self.dh, self.n[f])
            sign = 1. if axis @ self.n[f + 1] >= 0 else -1.
            end = np.cos(phi) * self.n[f] + sign * np.sin(phi) * axis
            if np.linalg.norm(end - self.n[f + 1]) > 1e-9:
                exact = float(np.arctan2(abs(axis @ self.n[f + 1]), self.n[f] @ self.n[f + 1]))
                fallback.append(dict(fold=f, sigma=float(self.sigma[f + 1]), phi_arccos_rad=float(phi), phi_atan2_rad=exact,
                                     skipped=bool(exact <= 1e-12)))
                if exact <= 1e-12:
                    continue
                phi = exact
                e = max_offset * phi
                ok, term = fits(f, e)
                if not ok:
                    raise ValueError('fold transition zone exceeds half an adjacent source panel')
                end = np.cos(phi) * self.n[f] + sign * np.sin(phi) * axis
                if np.linalg.norm(end - self.n[f + 1]) > 1e-9:
                    raise ValueError('fold normal rotation does not reach the next panel normal')
            zones.append(dict(fold=f, sigma=float(self.sigma[f + 1]), phi_rad=float(phi), half_width_m=float(e),
                              axis=axis, sign=sign))
            if term is not None:
                terminal.append(term)
        if fallback:
            self.fold_angle_fallback = fallback
        if terminal:
            self.fold_zone_terminal = terminal
        return zones

    def normals(self, sigma, zones):
        N = self.n[self.panel(sigma)].copy()
        for z in zones:
            m = np.abs(sigma - z['sigma']) < z['half_width_m']
            if m.any():
                alpha = z['phi_rad'] * (sigma[m] - (z['sigma'] - z['half_width_m'])) / (2 * z['half_width_m'])
                n0 = self.n[z['fold']]
                N[m] = np.cos(alpha)[:, None] * n0 + (z['sign'] * np.sin(alpha))[:, None] * z['axis']
        return N


def _ccw(poly):
    x, y = poly[:, 0], poly[:, 1]
    return poly if (x @ np.roll(y, -1) - y @ np.roll(x, -1)) > 0 else poly[::-1]


class PathPieces:
    """Footprint pieces and their wall-offset fields for one path (all arrays in the unrolled plane)."""
    overlap_blend = None

    def __init__(self, chart, points, inward, normal, segment_ids, width, depth, common, differential,
                 fractions, margin, path_index, original_id, planes=()):
        self.path_index, self.original_id, self.margin = path_index, int(original_id), float(margin)
        # finite-support clips of THIS path only: (sigma, 'le'|'ge'[, end segment id])
        self.planes = [tuple(pl) if len(pl) == 3 else (pl[0], pl[1], int(segment_ids[0 if pl[1] == 'ge' else -1])) for pl in planes]
        st, panels, resid = chart.locate(points)
        if resid.max() > 1e-9:
            raise ValueError('path does not lie on its source surface: %.3e m' % resid.max())
        if np.min(np.einsum('ij,ij->i', normal, chart.n[panels])) <= 0:
            raise ValueError('supplied fracture normal disagrees with the source panel orientation')
        self.p = st
        self.depth_last = np.asarray(depth, float)[:, -1]
        self.fr = np.asarray(fractions, float)
        self.oL = np.asarray(common - width / 2 - differential, float)
        self.oR = np.asarray(common + width / 2 + differential, float)
        self.segment_ids = np.asarray(segment_ids)
        inward2 = chart.vector_2d(np.asarray(inward, float), panels)
        groups, start = [], 0
        for i in range(1, len(st)):
            if i == len(st) - 1 or segment_ids[i] != segment_ids[i - 1]:
                groups.append((start, i)); start = i
        self.groups, self.t, self.nu = groups, [], []
        for a, b in groups:
            t = st[b] - st[a]; t /= np.linalg.norm(t)
            nu = np.array([-t[1], t[0]])
            if nu @ inward2[a:b + 1].sum(0) < 0:
                nu = -nu
            if np.min(inward2[a:b + 1] @ nu) <= 0:
                raise ValueError('segment inward disagrees with its in-surface frame')
            self.t.append(t); self.nu.append(nu)
        self.spans, self.wedges = [], []
        last = len(st) - 2
        for g, (a, b) in enumerate(groups):
            for k in range(a, b):
                self.spans.append(dict(k=k, g=g, segment=int(segment_ids[k]),
                                       start_cap='station_start' if k == 0 else 'internal_cap',
                                       end_cap='station_end' if k == last else 'internal_cap'))
        self.skipped_small_turns = 0
        for g in range(1, len(groups)):
            ta, tb, na, nb = self.t[g - 1], self.t[g], self.nu[g - 1], self.nu[g]
            theta = float(np.arccos(np.clip(ta @ tb, -1., 1.)))
            if theta < MIN_TURN_RAD:
                self.skipped_small_turns += 1
                continue
            i = groups[g][0]
            self.wedges.append(dict(i=i, g=g, theta=theta, depth_side=bool(nb @ ta > 0),
                                    segment=int(segment_ids[i - 1]), pair=[int(segment_ids[i - 1]), int(segment_ids[i])]))

    # --- geometry ---------------------------------------------------------------------------------
    def span_polygon(self, s):
        """: the path nodes are explicit vertices of the end edges, so a node wedge shares whole edges (identical vertices, also
        after Clipper2's grid snap) instead of sub-edges."""
        k, nu, m = s['k'], self.nu[s['g']], self.margin
        p0, p1 = self.p[k], self.p[k + 1]
        return np.array([p0 - m * nu, p1 - m * nu, p1, p1 + self.depth_last[k + 1] * nu, p0 + self.depth_last[k] * nu, p0])

    def wedge_polygon(self, w):
        i, na, nb = w['i'], self.nu[w['g'] - 1], self.nu[w['g']]
        r = self.depth_last[i] if w['depth_side'] else -self.margin
        return np.array([self.p[i], self.p[i] + r * na, self.p[i] + r * nb])

    def pieces(self):
        out = []
        for s in self.spans:
            poly = self.span_polygon(s)
            out.append(dict(kind='span', ref=s, poly=poly, segment=s['segment'],
                            edges=[(poly[0], poly[1], 'outside_cap'), (poly[1], poly[2], s['end_cap']),
                                   (poly[2], poly[3], s['end_cap']), (poly[3], poly[4], 'floor'),
                                   (poly[4], poly[5], s['start_cap']), (poly[5], poly[0], s['start_cap'])]))
        for w in self.wedges:
            poly = self.wedge_polygon(w)
            chord = 'floor' if w['depth_side'] else 'outside_cap'
            out.append(dict(kind='wedge', ref=w, poly=poly, segment=w['segment'],
                            edges=[(poly[0], poly[1], 'internal'), (poly[1], poly[2], chord), (poly[2], poly[0], 'internal')]))
        for piece in out:
            piece.update(path=self, lo=piece['poly'].min(0) - TOL, hi=piece['poly'].max(0) + TOL)
        return out

    def inside_clips(self, q):
        keep = np.ones(len(q), bool)
        for sigma, side, _ in self.planes:
            keep &= (q[:, 0] <= sigma + TOL) if side == 'le' else (q[:, 0] >= sigma - TOL)
        return keep

    # --- fields -----------------------------------------------------------------------------------
    def span_field(self, s, q, side):
        """Old loft wall on one span: returns (covered, offset), exact PL with the loft's own diagonals.

        The ring lists the left wall by increasing depth and the right wall by decreasing depth, so the loft's
        (a,c,b)/(a,d,c) split uses diagonal (k,j)-(k+1,j+1) on the left and (k,j+1)-(k+1,j) on the right."""
        o = self.oL if side == 'L' else self.oR
        k, t, nu = s['k'], self.t[s['g']], self.nu[s['g']]
        rel = q - self.p[k]
        span = (self.p[k + 1] - self.p[k]) @ t
        a_raw, s_raw = rel @ t, rel @ nu
        a = np.clip(a_raw / span, 0., 1.)
        dep = (1 - a) * self.depth_last[k] + a * self.depth_last[k + 1]
        covered = (a_raw >= -TOL) & (a_raw <= span + TOL) & (s_raw >= -self.margin - TOL) & (s_raw <= dep + TOL)
        value = (1 - a) * o[k, 0] + a * o[k + 1, 0]
        deep = s_raw > 0
        if deep.any():
            qq = q[deep]
            G0 = self.p[k] + np.outer(self.depth_last[k] * self.fr, nu)
            G1 = self.p[k + 1] + np.outer(self.depth_last[k + 1] * self.fr, nu)
            f = np.clip(s_raw[deep] / np.maximum(dep[deep], 1e-300), 0., 1.)
            j = np.clip(np.searchsorted(self.fr, f, side='right') - 1, 0, len(self.fr) - 2)
            if side == 'L':   # a=(k,j) b=(k,j+1) c=(k+1,j+1) d=(k+1,j)
                pa, pb, pc, pd = G0[j], G0[j + 1], G1[j + 1], G1[j]
                va, vb, vc, vd = o[k, j], o[k, j + 1], o[k + 1, j + 1], o[k + 1, j]
            else:             # a=(k,j+1) b=(k,j) c=(k+1,j) d=(k+1,j+1)
                pa, pb, pc, pd = G0[j + 1], G0[j], G1[j], G1[j + 1]
                va, vb, vc, vd = o[k, j + 1], o[k, j], o[k + 1, j], o[k + 1, j + 1]
            ok1, w1 = _barycentric(qq, pa, pc, pb)
            ok2, w2 = _barycentric(qq, pa, pd, pc)
            use1 = ok1 & ((np.min(w1, axis=1) >= np.min(w2, axis=1)) | ~ok2)
            val1 = w1[:, 0] * va + w1[:, 1] * vc + w1[:, 2] * vb
            val2 = w2[:, 0] * va + w2[:, 1] * vd + w2[:, 2] * vc
            value[deep] = np.where(use1, val1, val2)
        return covered, value

    def wedge_field(self, w, q, side):
        o = self.oL if side == 'L' else self.oR
        i = w['i']
        poly = self.wedge_polygon(w)
        _, bc = _barycentric(q, poly[0], poly[1], poly[2])
        scale = np.linalg.norm(poly[1] - poly[0])
        covered = np.min(bc, axis=1) * scale >= -TOL
        if not w['depth_side']:
            return covered, np.full(len(q), o[i, 0])
        b = self.nu[w['g'] - 1] + self.nu[w['g']]
        b /= np.linalg.norm(b)
        f = np.clip(((q - self.p[i]) @ b) / (self.depth_last[i] * np.cos(w['theta'] / 2)), 0., 1.)
        return covered, np.interp(f, self.fr, o[i])


def taper_aperture(paths, q, cell=.05):
    """taper_r2: the aperture the relief's tip taper uses at footprint points q (n, 2): over every piece covering q inside
    its path's clips (as envelope: buckets, the same coverage test; spans: the mouth aperture oR - oL interpolated
    linearly between the span's two stations in the span's own coordinate; wedges: their station's aperture), the
    largest one. Continuous along a path (span ends and wedges carry the station values), so neither the sigma-sorted
    mixing of distant stations (paths whose sigma is not monotone, several paths on one surface) nor a jump at a piece
    join enters the taper. A point no piece covers takes the aperture of the nearest station of any path."""
    q = np.atleast_2d(np.asarray(q, float)); best = np.full(len(q), -np.inf); buckets = _Buckets(q, cell)
    for piece in [pc for P_ in paths for pc in P_.pieces()]:
        idx = buckets.query(piece['lo'], piece['hi'])
        if not len(idx):
            continue
        P_, qq = piece['path'], q[idx]; ap = (P_.oR - P_.oL)[:, 0]
        if piece['kind'] == 'span':
            s_ = piece['ref']; k, t, nu = s_['k'], P_.t[s_['g']], P_.nu[s_['g']]; rel = qq - P_.p[k]; span = (P_.p[k + 1] - P_.p[k]) @ t
            a_raw, s_raw = rel @ t, rel @ nu; a = np.clip(a_raw / span, 0., 1.)
            dep = (1 - a) * P_.depth_last[k] + a * P_.depth_last[k + 1]
            cov = (a_raw >= -TOL) & (a_raw <= span + TOL) & (s_raw >= -P_.margin - TOL) & (s_raw <= dep + TOL)
            val = (1 - a) * ap[k] + a * ap[k + 1]
        else:
            w = piece['ref']; poly = P_.wedge_polygon(w); _, bc = _barycentric(qq, poly[0], poly[1], poly[2])
            cov = np.min(bc, axis=1) * np.linalg.norm(poly[1] - poly[0]) >= -TOL; val = np.full(len(qq), ap[w['i']])
        cov &= P_.inside_clips(qq); i_ = idx[cov]; best[i_] = np.maximum(best[i_], val[cov])
    ob = paths[0].overlap_blend if len(paths) else None
    if ob is not None:
        best = _soft_aperture(paths, q, best, ob, cell)
    miss = ~np.isfinite(best)
    if miss.any():
        P = np.concatenate([P_.p for P_ in paths]); A = np.concatenate([(P_.oR - P_.oL)[:, 0] for P_ in paths])
        for j in np.flatnonzero(miss):
            best[j] = A[int(np.argmin(np.linalg.norm(P - q[j], axis=1)))]
    return best


def taper_factor(paths, q, tip):
    """taper_r2: the relief's tip-taper factor g(q) = smoothstep(x), x = clip(taper_aperture(paths, q) / tip, 0, 1), the ONE
    implementation _surface_cavity, fault_charts.relief_on_chart and the trace labels' reference use. q: chart coordinates (the
    unrolled plane of PathPieces.p); paths: the PathPieces build_surface_cavity receives (source frame for a moved group)."""
    x = np.clip(taper_aperture(paths, q) / tip, 0., 1.)
    return x * x * (3. - 2. * x)


def _barycentric(q, A, B, C):
    """Barycentric coordinates of q in triangles (A,B,C) (broadcast); valid flag where the triangle is not degenerate."""
    v0, v1, v2 = B - A, C - A, q - A
    d00 = np.sum(v0 * v0, -1); d01 = np.sum(v0 * v1, -1); d11 = np.sum(v1 * v1, -1)
    d20 = np.sum(v2 * v0, -1); d21 = np.sum(v2 * v1, -1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-30 * np.maximum(d00 * d11, 1e-300)
    den = np.where(ok, den, 1.)
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    return np.broadcast_to(ok, np.shape(b1)), np.stack((1 - b1 - b2, b1, b2), axis=-1)


class _Buckets:
    def __init__(self, q, cell):
        self.q, self.cell = q, cell
        self.origin = q.min(0)
        keys = np.floor((q - self.origin) / cell).astype(np.int64)
        self.shape = keys.max(0) + 1
        flat = keys[:, 0] * self.shape[1] + keys[:, 1]
        order = np.argsort(flat, kind='stable')
        self.order, self.sorted = order, flat[order]

    def query(self, lo, hi):
        a = np.maximum(np.floor((lo - self.origin) / self.cell).astype(np.int64), 0)
        b = np.minimum(np.floor((hi - self.origin) / self.cell).astype(np.int64), self.shape - 1)
        if np.any(b < a):
            return np.zeros(0, np.int64)
        out = []
        for x in range(a[0], b[0] + 1):
            l = np.searchsorted(self.sorted, x * self.shape[1] + a[1], 'left')
            r = np.searchsorted(self.sorted, x * self.shape[1] + b[1], 'right')
            out.append(self.order[l:r])
        idx = np.concatenate(out) if out else np.zeros(0, np.int64)
        p = self.q[idx]
        return idx[np.all((p >= lo) & (p <= hi), axis=1)]


def envelope(pieces, q, side, cell=.05):
    """Union-semantics offset at q: min over covering pieces for 'L', max for 'R'; also the defining piece."""
    best = np.full(len(q), np.inf if side == 'L' else -np.inf)
    owner = np.full(len(q), -1)
    count = np.zeros(len(q), np.int64)
    buckets = _Buckets(q, cell)
    for index, piece in enumerate(pieces):
        idx = buckets.query(piece['lo'], piece['hi'])
        if not len(idx):
            continue
        path = piece['path']
        field = path.span_field if piece['kind'] == 'span' else path.wedge_field
        covered, value = field(piece['ref'], q[idx], side)
        covered &= path.inside_clips(q[idx])
        idx, value = idx[covered], value[covered]
        count[idx] += 1
        better = value < best[idx] if side == 'L' else value > best[idx]
        best[idx[better]] = value[better]
        owner[idx[better]] = index
    ob = pieces[0]['path'].overlap_blend if len(pieces) else None
    if ob is not None:
        best, owner = _soft_envelope(pieces, q, side, best, owner, count, ob, cell)
    return best, owner, count


# Opt-in per surface: the builder sets PathPieces.overlap_blend (the same dict it records in construction_checks['overlap_blend'])
# With it, at points covered by >= 2 pieces of which at least one is FAR -- its exposure (a span's station segment, a wedge's
# station) is more than near_m farther from the point than the nearest covering exposure -- the far pieces enter the union with a
# weight w = 1 - smoothstep((e - near_m) / fade_m): L = o_near + min(0, min_far w (o_j - o_near)), R likewise with max; the
# taper aperture likewise with max. count is the strict cover count; owner is the piece that gives the final value. Points whose
# own end cap, a clip plane) inside a far band, excluding the far piece would open a new jump at that edge; so the exclusion is lifted
# towards the near group's true boundary: w_eff = max(w, 1 - smoothstep(D / edge_m)), D = the largest w-weighted distance of the
OVERLAP_BLEND = dict(near_m=.005, fade_m=.03, edge_m=.02)


def overlap_gate(paths, min_area=1e-8):
    closed = [int(P_.path_index) for P_ in paths if float(np.linalg.norm(P_.p[0] - P_.p[-1])) < 1e-9]
    flat = [pc for P_ in paths for pc in P_.pieces()]
    pts = np.concatenate([pc['poly'] for pc in flat]); lo, hi = pts.min(0), pts.max(0); span = 10. + float(np.max(hi - lo))
    boxes = {}
    for P_ in paths:
        cs = None
        for sigma, keep, _ in P_.planes:
            x0, x1 = (lo[0] - span, sigma) if keep == 'le' else (sigma, hi[0] + span)
            box = md.CrossSection.square((x1 - x0, 2 * span + hi[1] - lo[1])).translate((x0, lo[1] - span))
            cs = box if cs is None else md.CrossSection.batch_boolean([cs, box], md.OpType.Intersect)
        boxes[id(P_)] = cs
    sec = []
    for pc in flat:
        c = md.CrossSection([_ccw(pc['poly'])]); bx = boxes[id(pc['path'])]
        sec.append(c if bx is None else md.CrossSection.batch_boolean([c, bx], md.OpType.Intersect))
    LO = np.array([pc['lo'] for pc in flat]); HI = np.array([pc['hi'] for pc in flat]); order = np.argsort(LO[:, 0], kind='stable')
    pairs, area = 0, 0.
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if LO[j, 0] > HI[i, 0]:
                break
            if flat[i]['path'] is flat[j]['path'] or LO[j, 1] > HI[i, 1] or LO[i, 1] > HI[j, 1]:
                continue
            a = md.CrossSection.batch_boolean([sec[i], sec[j]], md.OpType.Intersect).area()
            if a > min_area:
                pairs += 1; area += a
    return bool(closed or pairs), dict(closed_paths=closed, other_path_overlap_pairs=int(pairs), other_path_overlap_area_m2=float(area))


def _exposure_distance(piece, qq):
    """chart distance from qq to the piece's exposure: a span's station segment p_k p_k+1, a wedge's station p_i"""
    P_ = piece['path']
    if piece['kind'] == 'span':
        k = piece['ref']['k']; a, b = P_.p[k], P_.p[k + 1]; ab = b - a; L2 = float(ab @ ab)
        t = np.clip(((qq - a) @ ab) / L2, 0., 1.) if L2 > 0 else np.zeros(len(qq))
        return np.linalg.norm(qq - (a + t[:, None] * ab), axis=1)
    return np.linalg.norm(qq - P_.p[piece['ref']['i']], axis=1)


def _true_edge_distance(piece, qq):
    """distance from qq to the piece's TRUE edges (its edges whose role is not internal_cap / internal: the margin edge, the floor, a
    path's own end caps, a wedge's chord) and to its path's clip lines. A CLOSED path has no wedge at its closure node: there its
    station_start / station_end caps are internal on the side where the first and last spans overlap and true on the side of the
    gap (the side a wedge would take, as PathPieces' depth_side rule: the depth side if nu_first . t_last > 0)"""
    P_ = piece['path']; closed = bool(np.linalg.norm(P_.p[0] - P_.p[-1]) < 1e-9)
    gap_depth = bool(np.asarray(P_.nu[0], float) @ np.asarray(P_.t[-1], float) > 0) if closed else None
    d = np.full(len(qq), np.inf)
    for n_, (a, b, role) in enumerate(piece['edges']):
        if role in ('internal_cap', 'internal'):
            continue
        if closed and piece['kind'] == 'span' and role in ('station_start', 'station_end'):
            depth_side = n_ in (2, 4)      # span edges: 1 / 2 the end cap's margin / depth half, 4 / 5 the start cap's depth / margin half
            if depth_side != gap_depth:
                continue
        ab = b - a; L2 = float(ab @ ab); t = np.clip(((qq - a) @ ab) / L2, 0., 1.) if L2 > 0 else np.zeros(len(qq))
        d = np.minimum(d, np.linalg.norm(qq - (a + t[:, None] * ab), axis=1))
    for sigma, _, _ in piece['path'].planes:
        d = np.minimum(d, np.abs(qq[:, 0] - sigma))
    return d


def _soft_union(rec, n, sgn, ob):
    """rec: [(piece index, idx, value, exposure distance, true-edge distance)] over n points, sgn = +1 (min) / -1 (max)
    -> (value, owner, far_any)"""
    dhome = np.full(n, np.inf)
    for _, idx, _, x, _ in rec:
        np.minimum.at(dhome, idx, x)
    near_m, fade_m, edge_m = float(ob['near_m']), float(ob['fade_m']), float(ob['edge_m'])
    ws, D = [], np.zeros(n)
    for _, idx, _, x, dt in rec:
        t = np.clip((x - dhome[idx] - near_m) / fade_m, 0., 1.); w = 1. - t * t * (3. - 2. * t); ws.append(w)
        np.maximum.at(D, idx, w * dt)
    t = np.clip(D / edge_m, 0., 1.); lift = 1. - t * t * (3. - 2. * t)
    onear = np.full(n, np.inf); own = np.full(n, -1); far_any = np.zeros(n, bool)
    for index, idx, value, x, _ in rec:
        nr = (x - dhome[idx]) <= near_m; i_ = idx[nr]; v = sgn * value[nr]; better = v < onear[i_]
        onear[i_[better]] = v[better]; own[i_[better]] = index; far_any[idx[~nr]] = True
    adj = np.zeros(n)
    for (index, idx, value, x, _), w in zip(rec, ws):
        fr = (x - dhome[idx]) > near_m
        if not fr.any():
            continue
        i_ = idx[fr]; we = np.maximum(w[fr], lift[i_])
        a = we * (sgn * value[fr] - onear[i_]); better = a < adj[i_]
        adj[i_[better]] = a[better]; own[i_[better]] = index
    return sgn * (onear + adj), own, far_any


def _soft_envelope(pieces, q, side, best, owner, count, ob, cell):
    multi = np.flatnonzero(count >= 2)
    if not len(multi):
        return best, owner
    qm = q[multi]; buckets = _Buckets(qm, cell); rec = []
    for index, piece in enumerate(pieces):
        idx = buckets.query(piece['lo'], piece['hi'])
        if not len(idx):
            continue
        path = piece['path']
        field = path.span_field if piece['kind'] == 'span' else path.wedge_field
        covered, value = field(piece['ref'], qm[idx], side)
        covered &= path.inside_clips(qm[idx])
        idx, value = idx[covered], value[covered]
        if len(idx):
            rec.append((index, idx, value, _exposure_distance(piece, qm[idx]), _true_edge_distance(piece, qm[idx])))
    val, own, far_any = _soft_union(rec, len(qm), 1. if side == 'L' else -1., ob)
    best, owner = best.copy(), owner.copy(); ch = multi[far_any]
    best[ch] = val[far_any]; owner[ch] = own[far_any]
    return best, owner


def _soft_aperture(paths, q, best, ob, cell):
    buckets = _Buckets(q, cell); rec = []; count = np.zeros(len(q), np.int64)
    for index, piece in enumerate([pc for P_ in paths for pc in P_.pieces()]):
        idx = buckets.query(piece['lo'], piece['hi'])
        if not len(idx):
            continue
        P_, qq = piece['path'], q[idx]; ap = (P_.oR - P_.oL)[:, 0]
        if piece['kind'] == 'span':
            s_ = piece['ref']; k, t, nu = s_['k'], P_.t[s_['g']], P_.nu[s_['g']]; rel = qq - P_.p[k]; span = (P_.p[k + 1] - P_.p[k]) @ t
            a_raw, s_raw = rel @ t, rel @ nu; a = np.clip(a_raw / span, 0., 1.)
            dep = (1 - a) * P_.depth_last[k] + a * P_.depth_last[k + 1]
            cov = (a_raw >= -TOL) & (a_raw <= span + TOL) & (s_raw >= -P_.margin - TOL) & (s_raw <= dep + TOL)
            val = (1 - a) * ap[k] + a * ap[k + 1]
        else:
            w = piece['ref']; poly = P_.wedge_polygon(w); _, bc = _barycentric(qq, poly[0], poly[1], poly[2])
            cov = np.min(bc, axis=1) * np.linalg.norm(poly[1] - poly[0]) >= -TOL; val = np.full(len(qq), ap[w['i']])
        cov &= P_.inside_clips(qq); idx, val = idx[cov], val[cov]
        if len(idx):
            count[idx] += 1; rec.append((index, idx, val, _exposure_distance(piece, q[idx]), _true_edge_distance(piece, q[idx])))
    multi = count >= 2
    if not multi.any():
        return best
    v, _, far_any = _soft_union(rec, len(q), -1., ob)
    best = best.copy(); ch = multi & far_any; best[ch] = v[ch]
    return best


def footprint(paths_pieces, paths):
    """2D union of all pieces of a surface; each path's union is first clipped by its own sigma half-planes."""
    sections = []
    for pieces, path in zip(paths_pieces, paths):
        cs = md.CrossSection.batch_boolean([md.CrossSection([_ccw(pc['poly'])]) for pc in pieces], md.OpType.Add)
        pts = np.concatenate([pc['poly'] for pc in pieces])
        lo, hi = pts.min(0), pts.max(0)
        span = 10. + float(np.max(hi - lo))
        for sigma, keep, _ in path.planes:
            x0, x1 = (lo[0] - span, sigma) if keep == 'le' else (sigma, hi[0] + span)
            box = md.CrossSection.square((x1 - x0, 2 * span + hi[1] - lo[1])).translate((x0, lo[1] - span))
            cs = md.CrossSection.batch_boolean([cs, box], md.OpType.Intersect)
        sections.append(cs)
    union = md.CrossSection.batch_boolean(sections, md.OpType.Add) if len(sections) > 1 else sections[0]
    return union.simplify(SIMPLIFY_M)


FOLD_SPLIT_TOL_M = 1e-7          # a footprint vertex this close to a fold line counts as on it (network position tolerance)


def split_at_sigma(V, T, cuts, tol=FOLD_SPLIT_TOL_M):
    """Exact split of a closed triangle mesh along the planes V[:, 0] = c (footprint sigma): every edge crossing a plane gets
    one shared point on it, every crossing triangle is re-triangulated keeping its orientation (a vertex within tol of the
    plane counts as on it, so no triangle spans more than tol across). Returns V, T and a record."""
    V = np.asarray(V, float).copy(); T = np.asarray(T, np.int64).copy()
    added, crossed = 0, 0
    for c in cuts:
        d = V[:, 0] - c
        s = np.where(np.abs(d) <= tol, 0, np.sign(d)).astype(int)
        st = s[T]
        cross = (st > 0).any(1) & (st < 0).any(1)
        if not cross.any():
            continue
        new, edge, rows = [], {}, []

        def point(a, b):
            k = (a, b) if a < b else (b, a)
            if k not in edge:
                t = d[a] / (d[a] - d[b])
                p = V[a] + t * (V[b] - V[a]); p[0] = c
                edge[k] = len(V) + len(new); new.append(p)
            return edge[k]
        for tri in T[cross]:
            ss = s[tri]
            zero = np.flatnonzero(ss == 0)
            if len(zero) == 1:
                r = int(zero[0]); a, b, e = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
                p = point(b, e); rows += [(a, b, p), (a, p, e)]
            else:
                r = next(i for i in range(3) if ss[i] != ss[(i + 1) % 3] and ss[i] != ss[(i + 2) % 3])
                a, b, e = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
                p, q = point(a, b), point(a, e)
                rows += [(a, p, q), (p, b, e), (p, e, q)]
        V = np.vstack([V, np.asarray(new)]); T = np.vstack([T[~cross], np.asarray(rows, np.int64)])
        added += len(new); crossed += int(cross.sum())
    return V, T, dict(cuts=[float(c) for c in cuts], added_vertices=added, split_triangles=crossed, tol_m=tol)


def build_surface_cavity(chart, paths, *, h=.02, first_face_id=0, tip_merge_tol=0., fold_split=None, wall_relief=None,
                         wall_relief_tip_m=None, wall_refine=None, end_trim=None):
    """One closed cavity mesh for every path on one source surface: _build_surface_cavity. end_trim: a callable (solid,
    tables) -> (solid, tables, record) applied in the source frame before the cavity is returned
    (fault_charts.make_end_trim: the joint-edge trim of fault ends); the record goes to checks['end_trim']."""
    out = _build_surface_cavity(chart, paths, h=h, first_face_id=first_face_id, tip_merge_tol=tip_merge_tol, fold_split=fold_split,
                                wall_relief=wall_relief, wall_relief_tip_m=wall_relief_tip_m, wall_refine=wall_refine)
    if end_trim is None:
        return out
    solid, tables, checks = out
    solid, tables, record = end_trim(solid, tables)
    checks['end_trim'] = record
    return solid, tables, checks


def _build_surface_cavity(chart, paths, *, h=.02, first_face_id=0, tip_merge_tol=0., fold_split=None, wall_relief=None,
                          wall_relief_tip_m=None, wall_refine=None):
    """One closed cavity mesh for every path on one source surface (each path carries its own clips).

    tip_merge_tol: a zero-width vertex inserted by refine_to_length whose footprint point differs from its bottom twin by
    <= tol (float rounding of the wall split) also merges onto that twin, instead of leaving a zero-area wall
    triangle. fold_split: 'always' splits the extruded footprint mesh along every chart fold line before the envelopes
    are evaluated, so no cavity triangle straddles a C0 kink of the chart surface (a near- collinear boundary sliver
    across a fold can flip there); 'fallback' builds unsplit first and rebuilds split only when the unsplit cavity
    fails its own 'flipped' / 'degenerate' check. Every check stays fail-closed. wall_relief: a callable rho(P) of 3D
    points (m). Both walls are displaced by the SAME rho evaluated at the chart surface point S(q) (mated walls: the
    aperture R - L is unchanged, the walls cannot cross). rho enters the offset before the fold zones / fold-split
    cuts are sized, so e_f = max|o| phi still bounds the offset sheets. Reported: the relief's linear-interpolation
    error on the wall triangles (random barycentric samples). wall_relief_tip_m: rho is multiplied by g(a) = x^2 (3 -
    2x), x = min(1, a / wall_relief_tip_m), a = the mouth aperture of the path stations interpolated linearly in sigma
    (a function of sigma only). Near a free tip the cavity is only millimetres deep while rho shifts it ~1 cm sideways
    onto host relief of a different height, so the shallow floor and the host meet at grazing angles and the Boolean
    leaves folded host fragments; with the taper the tip region reverts to the validated planar tip. The tapered rho
    is used everywhere rho is (offsets, fold zones, mouth report, interpolation report). fold zones on a flagged
    surface: e_f = M phi_f took M from the C1 offsets, so C1 moved the walls near every fold of a G surface (26130007
    trace-0011: up to 17 um). Where the paths carry overlap_blend and the chart has a fold, M = max(M0, M_C1): M0 is
    the offset of the same surface without C1 (same mode, same prism mesh, wall_refine re-run without the flags), so
    the zones and split lines equal the C1-free ones unless C1 raises the offsets past M0 (then M_C1, as before: e >=
    |o| phi). Runs before the fold-split cuts. Reported in checks['wall_refine']. Returns (manifold, {original_id:
    face_sources}, checks). Fails closed on any invalid construction."""
    extra = {} if wall_relief is None else dict(wall_relief=wall_relief)     # default: the unchanged call
    if wall_relief_tip_m is not None:
        if wall_relief is None or not wall_relief_tip_m > 0:
            raise ValueError('wall_relief_tip_m needs wall_relief and a positive length')
        extra['wall_relief_tip_m'] = float(wall_relief_tip_m)
    if wall_refine is not None:
        fine_of, h_fine = wall_refine
        if not (callable(fine_of) and 0 < float(h_fine) < h):
            raise ValueError('wall_refine needs (callable, 0 < h_fine < h)')
        extra['wall_refine'] = (fine_of, float(h_fine))
    if fold_split is None:
        return _surface_cavity(chart, paths, h, first_face_id, tip_merge_tol, False, **extra)
    if fold_split == 'always':
        solid, tables, checks = _surface_cavity(chart, paths, h, first_face_id, tip_merge_tol, True, **extra)
        checks['fold_split'] = dict(checks['fold_split'], mode='always', used=True)
        return solid, tables, checks
    if fold_split != 'fallback':
        raise ValueError('fold_split must be None, always or fallback')
    try:
        solid, tables, checks = _surface_cavity(chart, paths, h, first_face_id, tip_merge_tol, False, **extra)
        checks['fold_split'] = dict(mode='fallback', used=False)
        return solid, tables, checks
    except ValueError as error:
        if not str(error).startswith(('panel cavity wall triangles flipped', 'degenerate panel cavity triangle')):
            raise
        try:
            solid, tables, checks = _surface_cavity(chart, paths, h, first_face_id, tip_merge_tol, True, **extra)
        except ValueError as error2:
            # cavity that built before is unchanged): edge-flip repair of the flipped wall slivers (_flip_slivers)
            if not str(error2).startswith('panel cavity wall triangles flipped'):
                raise
            solid, tables, checks = _surface_cavity(chart, paths, h, first_face_id, tip_merge_tol, True, sliver_flip=True, **extra)
            checks.setdefault('sliver_flip', {})['split_error'] = str(error2)
        checks['fold_split'] = dict(checks['fold_split'], mode='fallback', used=True, unsplit_error=str(error))
        return solid, tables, checks


def _paired_prism(V, T, hz):
    """F6: refine_to_length triangulates the bottom and the top of the extruded footprint differently (~10 % of the top
    triangles have no bottom twin), so in thin zones the left and right walls, tens of micrometres apart, cross
    between vertices. Rebuild the prism from the refined BOTTOM triangulation only: the top is its mirror (same
    chart-plane vertices and triangles, reversed), the side walls join every boundary edge to its top copy (two
    triangles per edge). Vertex positions stay a function of (q, z), so a vertex present before keeps its position."""
    z = V[:, 2] / hz
    Tb = T[(z[T] <= 1e-9).all(1)]
    used = np.unique(Tb); nb = len(used); idx = np.full(len(V), -1, np.int64); idx[used] = np.arange(nb)
    Vb = V[used].copy(); Vb[:, 2] = 0.; Vt = Vb.copy(); Vt[:, 2] = hz
    B = idx[Tb]; Tt = B[:, ::-1] + nb
    E = np.r_[B[:, [0, 1]], B[:, [1, 2]], B[:, [2, 0]]]
    _, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    bnd = E[cnt[inv.ravel()] == 1]
    a, b = bnd[:, 0], bnd[:, 1]
    S1 = np.c_[b, a, a + nb]; S2 = np.c_[b, a + nb, b + nb]
    return np.r_[Vb, Vt], np.r_[B, Tt, S1, S2]


def _flip_slivers(T, kind, q, X, chart, zones, max_rounds=8):
    """Edge-flip repair of flipped cavity wall triangles.

    Cause: the flipped wall triangles are near-collinear footprint slivers (q-area 1-4 % of a regular triangle of the same
    edges) inside a fold's normal-rotation zone. On a flat panel X = S(q) + o n is affine in q along n, so a
    triangle's normal component is exactly its q-determinant and no sliver can flip, whatever the offset gradient.
    Inside a zone n turns with sigma, the term o dn/dsigma is not affine on the triangle, and on a sliver
    (q-determinant ~ 0, apparent offset gradient 10-36) it can outweigh the determinant: the linear sliver comes out
    reversed. Repair: each flipped wall triangle t = (a, b, c), its longest q-edge (a, b), the other triangle u = (b,
    a, d) of the same wall sharing it; the edge (a, b) is replaced by (c, d): (a, d, c), (d, b, c) (winding kept),
    only if (c, d) is not an edge yet and both new triangles face correctly with a nonzero q-determinant of the wall's
    sign. No vertex moves (the exact wall map is unchanged; only the piecewise-linear interpolation of the slivers
    changes). Repeated up to max_rounds; whatever stays flipped is left for the caller's fail-closed check."""
    T = T.copy()
    sgn = {'R': 1., 'L': -1.}

    def facing(idx):
        tri = X[T[idx]]; cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); a2 = np.linalg.norm(cr, axis=1)
        c2 = q[T[idx]].mean(1)
        return np.einsum('ij,ij->i', cr, chart.normals(c2[:, 0], zones)) / np.where(a2 > 0, a2, 1.), a2

    def bad(idx):
        f, a2 = facing(idx); k = kind[idx]
        return ((k == 'R') & (f <= 0)) | ((k == 'L') & (f >= 0)) | (a2 <= 0)

    def det_q(tri):
        p = q[np.asarray(tri)]
        return (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])

    wall = np.flatnonzero(kind != 'S')
    rec = dict(rounds=0, edge_flips=0, flipped_before=None, flipped_after=None, repaired=[])
    for _ in range(max_rounds):
        b = wall[bad(wall)]
        if rec['flipped_before'] is None: rec['flipped_before'] = int(len(b))
        if not len(b): break
        rec['rounds'] += 1; done = 0
        for t in b:
            if not bad(np.array([t]))[0]: continue
            v = T[t]; p = q[v]
            ln = [np.linalg.norm(p[(i + 1) % 3] - p[i]) for i in range(3)]
            i = int(np.argmax(ln)); a, bb, c = v[i], v[(i + 1) % 3], v[(i + 2) % 3]
            cand = np.flatnonzero((T == a).any(1) & (T == bb).any(1) & (kind == kind[t]))
            cand = cand[cand != t]
            if len(cand) != 1: continue
            u = int(cand[0]); w = list(T[u]); j = w.index(bb)
            if w[(j + 1) % 3] != a: continue                         # u must run (b, a, d): consistent winding
            d = w[(j + 2) % 3]
            if ((T == c).any(1) & (T == d).any(1)).any(): continue    # (c, d) already an edge
            n1, n2 = [a, d, c], [d, bb, c]
            s = sgn[str(kind[t])]
            if not (s * det_q(n1) > 0 and s * det_q(n2) > 0): continue
            old_t, old_u = T[t].copy(), T[u].copy()
            T[t], T[u] = n1, n2
            if bad(np.array([t, u])).any():
                T[t], T[u] = old_t, old_u; continue
            done += 1; rec['repaired'].append(dict(sigma=float(p[:, 0].mean()), s=float(p[:, 1].mean()), kind=str(kind[t]),
                                                    q_det_ratio=float(abs(det_q(old_t)) / max(ln) ** 2)))
        rec['edge_flips'] += done
        if not done: break
    rec['flipped_after'] = int(bad(wall).sum())
    return T, rec


def _lexless(P, Q):
    return (P[:, 0] < Q[:, 0]) | ((P[:, 0] == Q[:, 0]) & (P[:, 1] < Q[:, 1]))


def _split_marked(V, T, marked_edge):
    """One conformal longest-edge (Rivara) pass: the marked edges are closed under 'a triangle with a marked edge also has its
    longest edge marked' (longest in the unrolled plane; exact ties broken by the edges' lexicographically smaller end point,
    a purely geometric rule, so twins with reversed winding are treated identically); every marked edge is split at its
    midpoint (prism coordinates); each split triangle is bisected through its longest edge's midpoint first, then the other
    marked edges from that midpoint: 1 / 2 / 3 marked edges -> 2 / 3 / 4 triangles, orientation kept. Longest-edge
    bisection terminates and keeps the angles bounded below (plain bisection of the marked edge cycled on thin triangles)."""
    n = len(V); q = V[:, :2]
    E = np.sort(T[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uk, inv = np.unique(E[:, 0] * n + E[:, 1], return_inverse=True)
    ue = np.c_[uk // n, uk % n]; inv = inv.ravel().reshape(-1, 3)
    ln = np.linalg.norm(q[ue[:, 0]] - q[ue[:, 1]], axis=1)
    first = _lexless(q[ue[:, 0]], q[ue[:, 1]])[:, None]
    lo = np.where(first, q[ue[:, 0]], q[ue[:, 1]]); hi = np.where(first, q[ue[:, 1]], q[ue[:, 0]])
    # longest edge of each triangle (local index 0..2): the largest length; exact ties -> the lexicographically smallest
    # (smaller end point x, y, larger end point x, y), which no two distinct edges share
    cand = ln[inv] == ln[inv].max(1, keepdims=True)
    for key in (lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1]):
        k3 = np.where(cand, key[inv], np.inf); cand &= k3 == k3.min(1, keepdims=True)
    longest = np.argmax(cand, axis=1)
    mark = np.asarray(marked_edge(ue), bool).copy()
    lid = inv[np.arange(len(T)), longest]
    while True:
        need = mark[inv].any(1) & ~mark[lid]
        if not need.any():
            break
        mark[lid[need]] = True
    mid = np.full(len(ue), -1, np.int64); mid[mark] = n + np.arange(int(mark.sum()))
    V = np.vstack([V, .5 * (V[ue[mark, 0]] + V[ue[mark, 1]])])
    m = mid[inv]                                   # midpoint on edge (0,1), (1,2), (2,0) of each triangle, -1 = none
    split = m[np.arange(len(T)), longest] >= 0
    out = [T[~split]]
    sel = np.flatnonzero(split)
    if len(sel):
        idx = (np.arange(3)[None, :] + longest[sel, None]) % 3
        t = np.take_along_axis(T[sel], idx, 1); mm = np.take_along_axis(m[sel], idx, 1)
        a, b, cc, p0, p1, p2 = t[:, 0], t[:, 1], t[:, 2], mm[:, 0], mm[:, 1], mm[:, 2]
        h1, h2 = p1 >= 0, p2 >= 0
        out += [np.c_[a, p0, cc][~h2], np.c_[a, p0, p2][h2], np.c_[p0, cc, p2][h2],
                np.c_[p0, b, cc][~h1], np.c_[p0, b, p1][h1], np.c_[p0, p1, cc][h1]]
    return V, np.vstack(out).astype(np.int64), int(mark.sum())


def _refine_marked(V, T, fine_q, h_fine, max_passes=16):
    """Split until no edge with a fine end point is longer than h_fine in the unrolled plane (q = V[:, :2])."""
    fine = np.asarray(fine_q(V[:, :2]), bool); passes, added = 0, 0
    while True:
        def marked_edge(ue):
            ln = np.linalg.norm(V[ue[:, 0], :2] - V[ue[:, 1], :2], axis=1)
            return (ln > h_fine * (1 + 1e-9)) & (fine[ue[:, 0]] | fine[ue[:, 1]])
        n0 = len(V)
        V, T, k = _split_marked(V, T, marked_edge)
        if not k:
            return V, T, dict(passes=passes, added_vertices=added, fine_vertices=int(fine.sum()))
        fine = np.r_[fine, np.asarray(fine_q(V[n0:, :2]), bool)]; passes += 1; added += k
        if passes > max_passes:
            raise ValueError('wall refine did not settle in %d passes' % max_passes)


def _split_zone_offset(chart, flat, rho_of, V, T, floor=0.):
    """The fold-split iteration of _surface_cavity: M is iterated over the split vertices until the cuts and the zones
    coincide. floor: every m is raised to at least floor. Returns the split (V, T), the split record, M and the last
    unraised m (max over the final split vertices)."""
    V0, T0, split_record = V, T, None
    zone_offset = 0.
    for _ in range(6):
        Le, _, _ = envelope(flat, V0[:, :2], 'L')
        Re, _, _ = envelope(flat, V0[:, :2], 'R')
        if rho_of is not None:
            r0_ = rho_of(V0[:, :2]); Le, Re = Le + r0_, Re + r0_
        m = raw = float(max(np.max(np.abs(Le)), np.max(np.abs(Re))))
        if m < floor:
            m = floor
        if m <= zone_offset:
            break
        zone_offset = m
        cuts = sorted({c for z in chart.fold_zones(zone_offset) for c in (z['sigma'] - z['half_width_m'], z['sigma'], z['sigma'] + z['half_width_m'])})
        V0, T0, split_record = split_at_sigma(V, T, cuts)
    else:
        raise ValueError('fold split: zone offset did not settle')
    return V0, T0, split_record, zone_offset, raw


def _plain_zone_applies(chart, paths):
    """Whether a zone exists does not depend on M (a fold phi > 1e-12 that the angle fallback keeps), so fold_zones(0.)
    answers it; with e = 0 no terminal-panel zone is taken and the fallback record it may set is the one the build's
    own call sets again"""
    return (bool(len(paths)) and paths[0].overlap_blend is not None and bool(np.any(np.asarray(chart.fold_rad) > 1e-12))
            and bool(chart.fold_zones(0.)))


def _plain_zone_offset(chart, paths, flat, rho_of, V, T, hz, split, refine):
    """Every path's flag is None meanwhile and restored after (envelope, taper_aperture and the closures read it at call
    time). numpy only: no manifold is created, so no original id is consumed."""
    saved = [p.overlap_blend for p in paths]
    for p in paths:
        p.overlap_blend = None
    try:
        if refine is not None:
            V, T, _ = _refine_marked(V, T, refine[0], refine[1])
        if split:
            return _split_zone_offset(chart, flat, rho_of, V, T)[3]
        z = np.clip(V[:, 2] / hz, 0., 1.)
        q = V[:, :2]
        L, _, _ = envelope(flat, q, 'L')
        R, _, _ = envelope(flat, q, 'R')
        o = L + (R - L) * z
        if rho_of is not None:
            o = o + rho_of(q)
        return float(np.max(np.abs(o)))
    finally:
        for p, f in zip(paths, saved):
            p.overlap_blend = f


def _surface_cavity(chart, paths, h, first_face_id, tip_merge_tol, split, wall_relief=None, wall_relief_tip_m=None,
                    wall_refine=None, sliver_flip=False):
    rho_of = (lambda q_: np.asarray(wall_relief(chart.surface(q_)), float)) if wall_relief is not None else None
    all_pieces = [p.pieces() for p in paths]
    F = footprint(all_pieces, paths)
    flat = [pc for pieces in all_pieces for pc in pieces]
    if wall_relief_tip_m is not None:
        # continuous', but on paths whose sigma is not monotone and on surfaces with several paths it mixed distant stations (a free
        # tip's zero aperture beside a wide mouth): near-discontinuous, up to 36 mm of relief error on b395 trace-0009
        rho_full = rho_of

        def rho_of(q_):
            return rho_full(q_) * taper_factor(paths, q_, wall_relief_tip_m)
    hz = h / 2
    mesh = md.Manifold.extrude(F, hz).refine_to_length(h).to_mesh64()
    V = np.asarray(mesh.vert_properties)[:, :3].copy()
    T = np.asarray(mesh.tri_verts, dtype=np.int64).copy()
    V, T = _paired_prism(V, T, hz)          # F6: paired top / bottom (and side) triangulation
    V_prism, T_prism = V, T
    refine_record = None
    if wall_refine is not None:
        fine_of, h_fine = wall_refine

        def fine_q(q_):
            L_, _, cL_ = envelope(flat, q_, 'L'); R_, _, cR_ = envelope(flat, q_, 'R')
            if rho_of is not None:
                r_ = rho_of(q_); L_, R_ = L_ + r_, R_ + r_
            ok_ = (cL_ > 0) & (cR_ > 0); res_ = np.zeros(len(q_), bool)
            if ok_.any():
                S_ = chart.surface(q_[ok_]); n_ = chart.n[chart.panel(q_[ok_, 0])]
                res_[ok_] = np.asarray(fine_of(S_ + L_[ok_, None] * n_), bool) | np.asarray(fine_of(S_ + R_[ok_, None] * n_), bool)
            return res_
        n_coarse = len(T)
        V, T, refine_record = _refine_marked(V, T, fine_q, h_fine)
        refine_record.update(h_fine_m=h_fine, h_coarse_m=h, prism_triangles_coarse=int(n_coarse), prism_triangles=int(len(T)))
    zone_floor, zone_record = 0., None
    if _plain_zone_applies(chart, paths):
        zone_floor = _plain_zone_offset(chart, paths, flat, rho_of, V_prism, T_prism, hz, split,
                                        None if wall_refine is None else (fine_q, h_fine))
        zone_record = dict(rule='sec. 11.23.10', mode='split' if split else 'unsplit', unflagged_m=zone_floor)
    split_record, zone_offset = None, None
    if split:
        # The wall map S(q) + o n(sigma) is only C0 on three lines per fold: the chart kink sigma_f and the two edges
        # sigma_f +- e of the normal-rotation zone (n' jumps there). Cut on all three; e = M phi depends on the maximum
        # offset M, so M is iterated over the split vertices until the cuts and the zones used below coincide.
        V, T, split_record, zone_offset, zone_raw = _split_zone_offset(chart, flat, rho_of, V, T, zone_floor)
        split_record = dict(split_record or dict(cuts=[], added_vertices=0, split_triangles=0), zone_offset_m=zone_offset)
    # refine_to_length may add vertices inside the side walls; they map linearly between the two walls.
    z = np.clip(V[:, 2] / hz, 0., 1.)
    bottom, top = z <= 1e-9, z >= 1 - 1e-9
    q = V[:, :2]
    kind = np.where(top[T].all(1), 'R', np.where(bottom[T].all(1), 'L', 'S'))
    L, ownL, cL = envelope(flat, q, 'L')
    R, ownR, cR = envelope(flat, q, 'R')
    if np.any(cL == 0) or np.any(cR == 0):
        raise ValueError('footprint vertex outside every piece (%d)' % int(np.sum((cL == 0) | (cR == 0))))
    offset = L + (R - L) * z
    rho = None
    if rho_of is not None:
        rho = rho_of(q)
        offset = offset + rho
    vowner = np.where(top, ownR, ownL)
    # Tip lines (zero width) close: every vertex over a zero-thickness point merges onto its bottom twin.
    bottom_index = np.flatnonzero(bottom)
    key = {tuple(x): bottom_index[i] for i, x in enumerate(q[bottom])}
    remap = np.arange(len(V))
    for i in np.flatnonzero(~bottom & (R - L <= 1e-12)):
        j = key.get(tuple(q[i]))
        if j is not None:
            remap[i] = j
    tol_merged, tol_max = 0, 0.
    if tip_merge_tol > 0:
        zb = bottom_index[(R - L)[bottom_index] <= 1e-12]
        for i in np.flatnonzero(~bottom & (R - L <= 1e-12) & (remap == np.arange(len(V)))):
            if not len(zb):
                break
            d = np.linalg.norm(q[zb] - q[i], axis=1)
            k = int(d.argmin())
            if d[k] <= tip_merge_tol:
                remap[i] = zb[k]
                tol_merged += 1
                tol_max = max(tol_max, float(d[k]))
    merged = int(np.sum(remap != np.arange(len(V))))
    T = remap[T]
    nondegenerate = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    T, kind = T[nondegenerate], kind[nondegenerate]
    mouth = _mouth_offset_error(paths, flat, q, T, kind, offset, rho_of)
    # split path: the zones the cuts were placed on (zone_offset >= max |offset| over every vertex, so e >= |o| phi holds)
    zone_m = float(np.max(np.abs(offset))) if zone_offset is None else zone_offset
    if zone_record is not None:
        if zone_offset is None:
            zone_raw = zone_m
            if zone_m < zone_floor:
                zone_m = zone_floor
        zone_record.update(flagged_max_m=zone_raw, used_m=zone_m, flagged_exceeds=bool(zone_m > zone_floor))
    zones = chart.fold_zones(zone_m)
    X = chart.surface(q) + offset[:, None] * chart.normals(q[:, 0], zones)
    tri = X[T]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if np.any(area2 <= 0):
        raise ValueError('degenerate panel cavity triangle: %d' % int(np.sum(area2 <= 0)))
    centroid2 = q[T].mean(1)
    facing = np.einsum('ij,ij->i', cross, chart.normals(centroid2[:, 0], zones)) / area2
    flips = int(np.sum((kind == 'R') & (facing <= 0)) + np.sum((kind == 'L') & (facing >= 0)))
    sliver_record = None
    if flips and sliver_flip:
        T, sliver_record = _flip_slivers(T, kind, q, X, chart, zones)
        tri = X[T]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        if np.any(area2 <= 0):
            raise ValueError('degenerate panel cavity triangle: %d' % int(np.sum(area2 <= 0)))
        centroid2 = q[T].mean(1)
        facing = np.einsum('ij,ij->i', cross, chart.normals(centroid2[:, 0], zones)) / area2
        flips = int(np.sum((kind == 'R') & (facing <= 0)) + np.sum((kind == 'L') & (facing >= 0)))
    if flips:
        raise ValueError('panel cavity wall triangles flipped against the panel normal: %d' % flips)
    # Provenance: walls from the piece defining the envelope at the face centroid; sides from the piece edge.
    roles = np.empty(len(T), dtype=object)
    segs = np.zeros(len(T), np.int64)
    owner_path = np.zeros(len(T), np.int64)
    for side in ('L', 'R'):
        m = np.flatnonzero(kind == side)
        if not len(m):
            continue
        _, own, cnt = envelope(flat, centroid2[m], side)
        own = np.where(cnt > 0, own, vowner[T[m, 0]])
        roles[m] = 'left_wall' if side == 'L' else 'right_wall'
        segs[m] = [flat[i]['segment'] for i in own]
        owner_path[m] = [flat[i]['path'].original_id for i in own]
    sides = np.flatnonzero(kind == 'S')
    side_record = {}
    edge_role, edge_seg, edge_path, internal = _classify_sides(flat, q, T[sides], paths, side_record)
    roles[sides], segs[sides], owner_path[sides] = edge_role, edge_seg, edge_path
    # One Mesh64 run per owning path (its original ID), faces grouped by owner.
    order = np.argsort(owner_path, kind='stable')
    T, roles, segs, owner_path = T[order], roles[order], segs[order], owner_path[order]
    ids = np.arange(first_face_id, first_face_id + len(T), dtype=np.uint64)
    run_ids, starts = np.unique(owner_path, return_index=True)
    run_index = (np.r_[starts, len(T)] * 3).astype(np.uint64)
    used = np.unique(T)
    compact = np.full(len(X), -1, np.int64)
    compact[used] = np.arange(len(used))
    solid = md.Manifold(md.Mesh64(np.ascontiguousarray(X[used]), np.ascontiguousarray(compact[T].astype(np.uint64)),
        run_index=run_index, run_original_id=run_ids.astype(np.uint32), face_id=ids))
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError('panel cavity is not a valid oriented manifold: ' + str(solid.status()))
    tables = {int(p.original_id): {} for p in paths}
    for fid, role, seg, oid in zip(ids, roles, segs, owner_path):
        tables[int(oid)][int(fid)] = dict(source_segment_id=int(seg), role=str(role), construction='panel_r3')
    checks = dict(construction='panel_r3', refine_length_m=h, footprint_area_m2=F.area(),
                  footprint_simplify_m=SIMPLIFY_M, cavity_triangles=int(len(T)), tip_vertices_merged=merged,
                  fold_zones=[{k: v for k, v in z.items() if k != 'axis'} for z in zones],
                  max_abs_offset_m=float(np.max(np.abs(offset))), components=len(solid.decompose()),
                  exposed_internal_cap_edges=internal, mesh_volume_m3=solid.volume(), mouth_offset_error_m=mouth,
                  skipped_small_turns=int(sum(p.skipped_small_turns for p in paths)),
                  pieces=dict(spans=int(sum(len(p.spans) for p in paths)), wedges=int(sum(len(p.wedges) for p in paths))))
    if tip_merge_tol > 0:
        checks.update(tip_merge_tol_m=float(tip_merge_tol), tip_vertices_merged_by_tol=tol_merged, tip_merge_max_distance_m=tol_max)
    if split_record is not None:
        checks['fold_split'] = split_record
    if refine_record is not None:
        checks['wall_refine'] = refine_record
    if zone_record is not None:
        checks['overlap_blend_zone'] = zone_record
    if sliver_record is not None:
        checks['sliver_flip'] = sliver_record
    if getattr(chart, 'fold_angle_fallback', None):      # only when SurfaceChart.fold_zones used its angle fallback
        checks['fold_angle_fallback'] = chart.fold_angle_fallback
    if side_record:                                       # only when _classify_sides used its off-line fallback
        checks.update(side_record)
    if getattr(chart, 'fold_zone_terminal', None):       # only when SurfaceChart.fold_zones accepted a terminal-panel zone
        checks['fold_zone_terminal'] = chart.fold_zone_terminal
    if rho is not None:
        # relief interpolation error on the wall triangles (exact rho at random barycentric points vs the linear mesh)
        wt = np.flatnonzero(kind != 'S')
        g_ = np.random.default_rng(25102901)
        fi = g_.choice(wt, min(len(wt), 200000), replace=False) if len(wt) else wt
        b_ = g_.random((len(fi), 2)); fl_ = b_.sum(1) > 1; b_[fl_] = 1 - b_[fl_]; bc_ = np.c_[1 - b_.sum(1), b_]
        qs_ = (q[T[fi]] * bc_[:, :, None]).sum(1)
        err_ = np.abs(rho_of(qs_) - (rho[T[fi]] * bc_).sum(1))
        checks['wall_relief'] = dict(rho_m=dict(rms=float(np.sqrt(np.mean(rho ** 2))), max_abs=float(np.max(np.abs(rho)))),
                                     interp_error_m=(dict(p99=float(np.quantile(err_, .99)), max=float(err_.max()), samples=int(len(fi)))
                                                     if len(fi) else None))
        if wall_relief_tip_m is not None:
            checks['wall_relief']['tip_taper_m'] = wall_relief_tip_m
    return solid, tables, checks


def _bucket_candidates(b, lo, hi):
    """Indices (ascending) of the query boxes [lo, hi] whose _Buckets cell range a..z (the same clipped floor rule as
    _Buckets.query) holds at least one point: a superset of the boxes whose query is non-empty (summed-area table of the
    per-cell point counts)."""
    keys = np.floor((b.q - b.origin) / b.cell).astype(np.int64)
    grid = np.zeros(b.shape, np.int64); np.add.at(grid, (keys[:, 0], keys[:, 1]), 1)
    S = np.zeros((b.shape[0] + 1, b.shape[1] + 1), np.int64); S[1:, 1:] = grid.cumsum(0).cumsum(1)
    a = np.maximum(np.floor((lo - b.origin) / b.cell).astype(np.int64), 0)
    z = np.minimum(np.floor((hi - b.origin) / b.cell).astype(np.int64), b.shape - 1)
    ok = np.all(z >= a, axis=1)
    ai = np.minimum(a, b.shape); zi = np.clip(z + 1, 0, b.shape)
    cnt = S[zi[:, 0], zi[:, 1]] - S[ai[:, 0], zi[:, 1]] - S[zi[:, 0], ai[:, 1]] + S[ai[:, 0], ai[:, 1]]
    return np.flatnonzero(ok & (cnt > 0))


def _mouth_offset_error(paths, pieces, q, T, kind, offset, rho_of=None):
    """Report only: at every refined station (on the path, s = 0) the mesh-interpolated wall offset vs the exact
    envelope. The mesh does not follow the path line, so the s = 0 crease is smoothed over about one cell."""
    stations = np.concatenate([p.p for p in paths])
    out = {}
    for side, name in (('L', 'left'), ('R', 'right')):
        exact, _, cnt = envelope(pieces, stations, side)
        if rho_of is not None:
            exact = exact + rho_of(stations)
        tris = T[kind == side]
        A, B, C = q[tris[:, 0]], q[tris[:, 1]], q[tris[:, 2]]
        buckets = _Buckets(stations, .05)
        mesh = np.full(len(stations), np.nan)
        lo, hi = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
        for t in _bucket_candidates(buckets, lo, hi):
            idx = buckets.query(lo[t], hi[t])
            idx = idx[np.isnan(mesh[idx])]
            if not len(idx):
                continue
            ok, w = _barycentric(stations[idx], A[t], B[t], C[t])
            inside = ok & (np.min(w, axis=1) >= -1e-9)
            if inside.any():
                mesh[idx[inside]] = w[inside] @ offset[tris[t]]
        error = np.abs(mesh - exact)[np.isfinite(mesh) & (cnt > 0)]
        out[name] = dict(max=float(error.max()), p95=float(np.quantile(error, .95)), median=float(np.median(error)),
                         stations=int(len(error)))
    return out


def _classify_sides(pieces, q, side_tris, paths, record=None):
    """Role/segment/path of each side triangle from the piece edge carrying its footprint boundary edge.
    record (dict, optional): receives 'clip_side_fallback' when the off-line fallback below is used."""
    priority = {'finite_support_bound': 0, 'station_start': 1, 'station_end': 1, 'floor': 2, 'outside_cap': 3,
                'internal_cap': 4, 'internal': 5}
    edges = [(a, b, role, pc['segment'], pc['path'].original_id) for pc in pieces for a, b, role in pc['edges']]
    # A clip line belongs to no single path: paths of one surface ending on the same source end share it. Its side
    # faces take the path/segment of the piece that covers the boundary just inside the line (sigma side, owner).
    clip_lines = sorted({(sigma, keep) for path in paths for sigma, keep, _ in path.planes})
    for sigma, keep in clip_lines:
        edges.append((np.array([sigma, -1e9]), np.array([sigma, 1e9]), 'finite_support_bound', None, keep))
    A = np.array([e[0] for e in edges]); B = np.array([e[1] for e in edges])
    lo, hi = np.minimum(A, B) - EDGE_TOL, np.maximum(A, B) + EDGE_TOL
    roles_out, segs_out, paths_out, clip, clip_src = [], [], [], [], []
    unmatched = internal = 0
    for t in side_tris:
        mid = np.unique(q[t], axis=0).mean(0)
        cand = np.flatnonzero(np.all((mid >= lo) & (mid <= hi), axis=1))
        best = None
        for c in cand:
            ab = B[c] - A[c]
            f = np.clip((mid - A[c]) @ ab / (ab @ ab), 0., 1.)
            if np.linalg.norm(mid - (A[c] + f * ab)) <= EDGE_TOL and (
                    best is None or priority[edges[c][2]] < priority[edges[best][2]]):
                best = c
        if best is None:
            unmatched += 1
            continue
        role = edges[best][2]
        if role in ('internal_cap', 'internal'):
            internal += 1
            role = 'internal_cap'
        if role == 'finite_support_bound':
            clip.append((len(roles_out), mid + np.array([10 * TOL if edges[best][4] == 'ge' else -10 * TOL, 0.])))
            clip_src.append((t, best))
            roles_out.append(role); segs_out.append(-1); paths_out.append(-1)
            continue
        roles_out.append(role); segs_out.append(int(edges[best][3])); paths_out.append(int(edges[best][4]))
    if unmatched:
        raise ValueError('footprint boundary edges match no piece edge: %d' % unmatched)
    if clip:
        _, own, cnt = envelope(pieces, np.array([p for _, p in clip]), 'L')
        if np.any(cnt == 0):
            # above raised, so every surface that built before is unchanged. A clip line is matched by the MIDPOINT of a
            # footprint edge; on a surface with several paths, another path's piece edge crossing the line can leave a tiny
            # boundary edge whose midpoint is within EDGE_TOL of the line although the edge does not lie on it (normal-route
            # a255s4 camera 3: a 1.2e-5 m edge of path 10's station end, midpoint 6e-6 m from path 9's end line). Such an
            # edge is re-matched among the piece edges (same priority, same EDGE_TOL); an edge whose end points both lie on
            # the line, or one that no piece edge carries, still raises.
            n_piece_edges = len(edges) - len(clip_lines)
            fallback = []
            for k in np.flatnonzero(cnt == 0):
                i, (t, line) = clip[k][0], clip_src[k]
                ends = np.unique(q[t], axis=0)
                off = np.abs(ends[:, 0] - A[line][0])
                if np.all(off <= EDGE_TOL):
                    raise ValueError('finite-support side face has no covering piece just inside its clip line')
                mid = ends.mean(0)
                best = None
                for c in np.flatnonzero(np.all((mid >= lo[:n_piece_edges]) & (mid <= hi[:n_piece_edges]), axis=1)):
                    ab = B[c] - A[c]
                    f = np.clip((mid - A[c]) @ ab / (ab @ ab), 0., 1.)
                    d = np.linalg.norm(mid - (A[c] + f * ab))
                    if d <= EDGE_TOL and (best is None or priority[edges[c][2]] < priority[edges[best][2]]):
                        best, best_d = c, d
                if best is None:
                    raise ValueError('finite-support side face has no covering piece just inside its clip line')
                role = edges[best][2]
                if role in ('internal_cap', 'internal'):
                    internal += 1
                    role = 'internal_cap'
                roles_out[i], segs_out[i], paths_out[i] = role, int(edges[best][3]), int(edges[best][4])
                fallback.append(dict(clip_sigma=float(A[line][0]), clip_keep=str(edges[line][4]), end_offsets_m=off.tolist(),
                                     role=role, path_original_id=int(edges[best][4]), segment=int(edges[best][3]), edge_distance_m=float(best_d)))
            if record is not None:
                record['clip_side_fallback'] = dict(count=len(fallback), faces=fallback[:50])
        for k, ((i, _), o) in enumerate(zip(clip, own)):
            if cnt[k] == 0:
                continue                          # re-matched by the off-line fallback above
            segs_out[i], paths_out[i] = int(pieces[o]['segment']), int(pieces[o]['path'].original_id)
    return roles_out, segs_out, paths_out, internal


def monte_carlo_volume(paths, n_per_m2=2e5, seed=25102701):
    """Independent stratified estimate of integral_F (R - L) dA: each piece triangle is a stratum sampled uniformly,
    a sample keeps its own path's clips and is weighted by 1 / (number of valid covering pieces)."""
    rng = np.random.default_rng(seed)
    pieces = [pc for p in paths for pc in p.pieces()]
    samples, strata = [], []
    for pc in pieces:
        poly = pc['poly']
        for a, b, c in [(poly[0], poly[j], poly[j + 1]) for j in range(1, len(poly) - 1)]:  # convex fan
            u, v = b - a, c - a
            area = abs(u[0] * v[1] - u[1] * v[0]) / 2
            if area <= 0:
                continue
            n = max(2, int(round(area * n_per_m2)))
            r1, r2 = rng.random(n), rng.random(n)
            s = np.sqrt(r1)
            x = (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c
            samples.append((x, pc['path'].inside_clips(x)))
            strata.append(area)
    q = np.concatenate([x for x, _ in samples])
    L, _, cnt = envelope(pieces, q, 'L')
    R, _, _ = envelope(pieces, q, 'R')
    estimate = variance = 0.
    start = 0
    for (x, keep), area in zip(samples, strata):
        sl = slice(start, start + len(x)); start += len(x)
        v = np.where(keep & (cnt[sl] > 0), (R[sl] - L[sl]) / np.maximum(cnt[sl], 1), 0.)
        estimate += area * v.mean()
        variance += area ** 2 * v.var(ddof=1) / len(v)
    return float(estimate), float(np.sqrt(variance)), int(len(q))
