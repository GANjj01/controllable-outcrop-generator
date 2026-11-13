"""Route B floor relief: continuous piecewise-planar chip facets on the floors of selected cells (first version: v2
group_a_05), as a procedural weathering appearance proxy -- not a validated model of group_a_05 fracture. Opt-in
(blockface 'relief'); nothing else changes.

Facet field: an irregular triangulation (Delaunay of variable-radius Poisson-disk nodes, 2D in the plane normal to the
B stretch direction m) with random node depths in [0, 1]; a share of neighbouring triangles is merged into planar
quadrilaterals so facets are not all triangles. Depth is continuous; the normal jumps across facet edges (creases). No
open cracks, no step faces.

Floor sheet of one cell (its floor triangles share vertices, triangulate()): the edges used by one sheet triangle are
protected -- risers, other cells, the patch boundary -- and are never split and never moved. The crease lines are
inserted as mesh edges (edge splits along each crease, stopping edge_m before a protected edge), then interior edges
are bisected to ~target_m (an edge of a triangle with a protected edge only down to 0.75 x that protected edge, so the
refinement ends). Displacement: along m (into the rock), amplitude x max_m x node depth (piecewise linear) x a
smoothstep ramp over ramp_m from the protected edges. Projection and displacement share m, so the sheet stays a height
field over its projection (no fold); every displaced vertex is checked to stay in its own cell (cells are convex:
bedding slab x own-block joint slabs), hence every displaced triangle stays inside its own residual block."""
import math

import numpy as np


def poisson_disk(rng, lo, hi, radius, tries=60000):
    """Dart throwing with a variable radius r(p) (accept if farther than r(p) and r(q) from each
    accepted q). lo, hi: 2D box."""
    pts, rad = [], []
    cand = rng.uniform(lo, hi, (tries, 2))
    rc = radius(cand)
    cell = float(rc.min()) / math.sqrt(2)
    grid = {}
    rmax = float(rc.max())
    reach = int(math.ceil(rmax / cell))
    for p, r in zip(cand, rc):
        gx, gy = int((p[0] - lo[0]) // cell), int((p[1] - lo[1]) // cell)
        ok = True
        for ix in range(gx - reach, gx + reach + 1):
            for iy in range(gy - reach, gy + reach + 1):
                for j in grid.get((ix, iy), ()):
                    if np.hypot(*(p - pts[j])) < max(r, rad[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((gx, gy), []).append(len(pts))
            pts.append(p)
            rad.append(r)
    return np.array(pts)


def delaunay(P):
    """Bowyer-Watson triangulation of 2D points (CCW triangles, index triples)."""
    lo, hi = P.min(0), P.max(0)
    span = float((hi - lo).max()) * 20
    c = (lo + hi) / 2
    S = np.vstack([P, [c[0] - span, c[1] - span], [c[0] + span, c[1] - span], [c[0], c[1] + span]])
    n = len(P)
    tris = [(n, n + 1, n + 2)]

    def circ(t):
        (ax, ay), (bx, by), (cx, cy) = S[list(t)]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2
    cc = [circ(tris[0])]
    for i in range(n):
        p = S[i]
        C = np.array(cc)
        bad = np.nonzero((C[:, 0] - p[0]) ** 2 + (C[:, 1] - p[1]) ** 2 < C[:, 2])[0]
        edges = {}
        for k in bad:
            a, b, c_ = tris[k]
            for e in ((a, b), (b, c_), (c_, a)):
                key = (min(e), max(e))
                edges[key] = None if key in edges else e
        keep = np.ones(len(tris), bool)
        keep[bad] = False
        tris = [t for t, k in zip(tris, keep) if k]
        cc = [x for x, k in zip(cc, keep) if k]
        for e in edges.values():
            if e is None:
                continue
            t = (e[0], e[1], i)
            (ax, ay), (bx, by), (cx, cy) = S[list(t)]
            if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
                t = (e[1], e[0], i)
            tris.append(t)
            cc.append(circ(t))
    return np.array([t for t in tris if max(t) < n], np.int64)


def facet_field(rng, lo, hi, r_lo, r_hi, merge_share, noise_scale=.6):
    """Irregular facet field: nodes, triangles, node depth in [0, 1], facet id per triangle.
    Poisson radius varies smoothly between r_lo and r_hi (random Fourier field, no grid)."""
    k = rng.normal(0, 1 / noise_scale, (6, 2))
    ph = rng.uniform(0, 2 * np.pi, 6)

    def radius(Q):
        s = np.cos(Q @ k.T * 2 * np.pi + ph).sum(1) / math.sqrt(3)
        return r_lo + (r_hi - r_lo) * np.clip(.5 + .35 * s, 0, 1)
    P = poisson_disk(rng, lo, hi, radius)
    T = delaunay(P)
    h = rng.uniform(0, 1, len(P))
    fid = np.arange(len(T))
    # merge: across a random share of interior edges, put the far node of the second triangle on the
    # first one's plane (both triangles then form one planar quadrilateral facet)
    emap = {}
    for ti, t in enumerate(T):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            emap.setdefault((min(a, b), max(a, b)), []).append(ti)
    inner = [(e, ts) for e, ts in emap.items() if len(ts) == 2]
    order = rng.permutation(len(inner))
    locked = np.zeros(len(P), bool)
    merged = 0
    target = int(merge_share * len(inner))
    for oi in order:
        if merged >= target:
            break
        (a, b), (t1, t2) = inner[oi]
        if fid[t1] != t1 or fid[t2] != t2:
            continue
        s = [v for v in T[t2] if v not in (a, b)][0]
        r = [v for v in T[t1] if v not in (a, b)][0]
        if locked[[a, b, r, s]].any():
            continue
        A = np.c_[P[[a, b, r]], np.ones(3)]
        coef = np.linalg.solve(A, h[[a, b, r]])
        hs = float(coef[:2] @ P[s] + coef[2])
        if not 0 <= hs <= 1:
            continue
        h[s] = hs
        locked[[a, b, r, s]] = True
        fid[t2] = t1
        merged += 1
    return {'P': P, 'T': T, 'h': h, 'fid': fid, 'merged_pairs': merged}


def field_eval(field, Q, chunk=1500):
    """Piecewise-linear depth at 2D points Q (NaN outside the triangulation) and the facet id."""
    P, T, h = field['P'], field['T'], field['h']
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    den = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    out = np.full(len(Q), np.nan)
    tid = np.full(len(Q), -1)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        l1 = ((B[:, 1] - C[:, 1])[None] * (q[:, :1] - C[:, 0][None]) + (C[:, 0] - B[:, 0])[None] * (q[:, 1:] - C[:, 1][None])) / den[None]
        l2 = ((C[:, 1] - A[:, 1])[None] * (q[:, :1] - C[:, 0][None]) + (A[:, 0] - C[:, 0])[None] * (q[:, 1:] - C[:, 1][None])) / den[None]
        l3 = 1 - l1 - l2
        ins = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        has = ins.any(1)
        j = ins.argmax(1)
        r = np.arange(len(q))
        val = l1[r, j] * h[T[j, 0]] + l2[r, j] * h[T[j, 1]] + l3[r, j] * h[T[j, 2]]
        out[s:s + chunk] = np.where(has, val, np.nan)
        tid[s:s + chunk] = np.where(has, j, -1)
    return out, tid


def _seg_dist2d(Q, S):
    """Min distance from 2D points Q to 2D segments S (k, 2, 2)."""
    best = np.full(len(Q), np.inf)
    for s in range(0, len(S), 512):
        a, b = S[s:s + 512, 0], S[s:s + 512, 1]
        ab = b - a
        L = np.maximum((ab * ab).sum(1), 1e-30)
        t = np.clip(((Q[:, None] - a[None]) * ab[None]).sum(2) / L[None], 0, 1)
        d = np.linalg.norm(Q[:, None] - (a[None] + t[..., None] * ab[None]), axis=2)
        best = np.minimum(best, d.min(1))
    return best


def _clip_half(poly, g, c):
    """Part of the convex 2D polygon (list of points) where c + g.x >= 0 (Sutherland-Hodgman)."""
    out = []
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        fp, fq = c + g @ p, c + g @ q
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            out.append(p + (q - p) * (fp / (fp - fq)))
    return out


def _poly_area(P):
    if len(P) < 3:
        return 0.
    P = np.asarray(P)
    return .5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def _tent_support(apex, planes, half):
    sq = [apex + np.array(v) * half for v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    poly = sq
    for g, c in planes:
        poly = _clip_half(poly, g, c)
        if len(poly) < 3:
            return []
    return poly


def chip_tents(rng, P2, Fl, PS, opts):
    """Chip scars of one floor sheet: non-overlapping 'tents', each the minimum of 4-7 planes, deepest (A) at an off-centre
    apex and falling to 0 at its polygonal rim (one short steep side, the others long and gentle); most of the floor
    stays at 0. Protected segments near a tent add an edge plane s_b x (distance to the segment line - margin), so the
    tent reaches 0 margin inside every protected edge with a planar wall (no ramp, no lip). Returns a list of tents
    {apex, planes [(g, c)]: L(x) = c + g.x, support, radius}."""
    max_m = float(opts.get('max_m', .015))
    margin = float(opts.get('edge_margin_m', .003))
    a2 = (P2[Fl[:, 1], 0] - P2[Fl[:, 0], 0]) * (P2[Fl[:, 2], 1] - P2[Fl[:, 0], 1]) - (P2[Fl[:, 1], 1] - P2[Fl[:, 0], 1]) * (P2[Fl[:, 2], 0] - P2[Fl[:, 0], 0])
    area = np.abs(a2) / 2
    n_cand = max(200, int(area.sum() / float(opts.get('candidate_spacing_m', .02)) ** 2))
    tri = rng.choice(len(Fl), n_cand, p=area / area.sum())
    r1, r2 = rng.uniform(0, 1, n_cand), rng.uniform(0, 1, n_cand)
    s = np.sqrt(r1)
    cand = (1 - s)[:, None] * P2[Fl[tri, 0]] + (s * (1 - r2))[:, None] * P2[Fl[tri, 1]] + (s * r2)[:, None] * P2[Fl[tri, 2]]
    centres, radii, tents = np.zeros((0, 2)), np.zeros(0), []
    lo_r, hi_r = opts.get('run_clip_m', (.04, .20))
    for ci in range(n_cand):
        a = cand[ci]
        dseg = _seg_dist2d(a[None], PS)[0] if len(PS) else np.inf
        R = float(np.clip(rng.lognormal(math.log(float(opts.get('median_run_m', .10))), .35), lo_r, hi_r))
        A = max_m * float(rng.uniform(.4, 1.))
        k = int(rng.integers(4, 8))
        for _ in range(30):
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
            if gaps.min() > .35 and gaps.max() < 2.2:
                break
        else:
            continue
        run = R * rng.uniform(.6, 1.4, k)
        steep = int(rng.integers(k))
        run[steep] = float(rng.uniform(*opts.get('steep_run_m', (.015, .035))))
        if dseg < margin + .004:
            continue
        planes = []
        for t, r in zip(ang, run):
            u = np.array([math.cos(t), math.sin(t)])
            sig = A / r
            planes.append((-sig * u, A + sig * (u @ a)))
        poly = _tent_support(a, planes, 3 * float(run.max()) + .05)
        if len(poly) < 3:
            continue
        rad = float(np.max(np.linalg.norm(np.asarray(poly) - a, axis=1)))
        if rad > float(opts.get('max_radius_m', .25)):
            continue
        if len(PS):
            # edge planes of the protected segments that can reach the support
            ab = PS[:, 1] - PS[:, 0]
            L2 = np.maximum((ab * ab).sum(1), 1e-30)
            t = np.clip(((a - PS[:, 0]) * ab).sum(1) / L2, 0, 1)
            dd = np.linalg.norm(a - (PS[:, 0] + t[:, None] * ab), axis=1)
            sb = math.tan(math.radians(float(rng.uniform(*opts.get('edge_slope_deg', (25., 40.))))))
            for si in np.nonzero(dd < rad + .02)[0]:
                p, q = PS[si]
                nrm = np.array([-(q - p)[1], (q - p)[0]])
                nrm /= max(np.linalg.norm(nrm), 1e-30)
                if nrm @ (a - p) < 0:
                    nrm = -nrm
                planes.append((sb * nrm, -sb * (nrm @ p + margin)))
            poly = _tent_support(a, planes, rad + .01)
            if len(poly) < 3:
                continue
            rad = float(np.max(np.linalg.norm(np.asarray(poly) - a, axis=1)))
        if min(c + g @ a for g, c in planes) < .001:
            continue  # too shallow after the edge planes
        # scars may overlap (union = max), but an apex keeps >= overlap x (r_i + r_j) from the others
        if len(centres) and (np.linalg.norm(centres - a, axis=1) < float(opts.get('overlap', .45)) * (radii + rad)).any():
            continue
        centres = np.vstack([centres, a])
        radii = np.r_[radii, rad]
        tents.append({'apex': a, 'planes': planes, 'support': poly, 'radius': rad, 'depth': A, 'n_dirs': k,
                      'steep_deg': math.degrees(math.atan(A / run[steep]))})
    return tents


def tent_facets(tent):
    """Convex planar facets of a tent: where plane j is the minimum, inside the support."""
    out = []
    pl = tent['planes']
    for j, (gj, cj) in enumerate(pl):
        poly = tent['support']
        for k_, (gk, ck) in enumerate(pl):
            if k_ != j:
                poly = _clip_half(poly, gk - gj, ck - cj)
                if len(poly) < 3:
                    break
        # keep thin facets: many near-parallel edge planes (short protected segments) meet in
        # slivers whose edges are pieces of the zero rim; dropping them left the rim open (B1)
        if len(poly) >= 3 and abs(_poly_area(poly)) > 1e-14:
            out.append((j, poly))
    return out


def _clip_convex(P, Q):
    """Convex polygon P clipped by the convex CCW polygon Q."""
    for i in range(len(Q)):
        p, q = np.asarray(Q[i]), np.asarray(Q[(i + 1) % len(Q)])
        e = q - p
        g = np.array([-e[1], e[0]])
        P = _clip_half(P, g, -(g @ p))
        if len(P) < 3:
            return []
    return P


def _line_in_poly(P, g, c):
    """Segment of the line c + g.x = 0 inside the convex polygon P (None if shorter than 1 mm)."""
    pts = []
    for i in range(len(P)):
        p, q = np.asarray(P[i]), np.asarray(P[(i + 1) % len(P)])
        fp, fq = c + g @ p, c + g @ q
        if (fp >= 0) != (fq >= 0):
            pts.append(p + (q - p) * (fp / (fp - fq)))
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    i, j = np.unravel_index(d.argmax(), d.shape)
    return (pts[i], pts[j]) if d[i, j] >= 1e-3 else None


def chip_creases(tents):
    """Crease segments of the chip field max(0, max_i min_j L_ij): every tent's facet edges (rim and
    hip lines) and, for overlapping tents, the lines where facet planes of two tents are equal
    inside both facets (extra edges where a third tent dominates carry no crease and do no harm)."""
    facets = [tent_facets(t) for t in tents]
    segs = {}

    def add(p, q):
        if np.linalg.norm(q - p) >= 1e-6:
            segs[tuple(sorted((tuple(np.round(p, 6)), tuple(np.round(q, 6)))))] = (p, q)
    for fs in facets:
        for _, poly in fs:
            for i in range(len(poly)):
                add(np.asarray(poly[i]), np.asarray(poly[(i + 1) % len(poly)]))
    for i in range(len(tents)):
        for j in range(i + 1, len(tents)):
            if np.linalg.norm(tents[i]['apex'] - tents[j]['apex']) >= tents[i]['radius'] + tents[j]['radius']:
                continue
            for ki, Pi in facets[i]:
                gi, ci = tents[i]['planes'][ki]
                for kj, Pj in facets[j]:
                    both = _clip_convex(Pi, Pj)
                    if len(both) < 3:
                        continue
                    gj, cj = tents[j]['planes'][kj]
                    seg = _line_in_poly(both, gi - gj, ci - cj)
                    if seg is not None:
                        add(*seg)
    return list(segs.values())


def tents_eval(tents, Q):
    """Depth max(0, max_i min_j L_ij) at 2D points Q."""
    d = np.zeros(len(Q))
    for t in tents:
        a, r = t['apex'], t['radius']
        idx = np.nonzero((np.abs(Q - a) <= r + 1e-9).all(1))[0]
        if not len(idx):
            continue
        v = np.min(np.stack([c + Q[idx] @ g for g, c in t['planes']]), axis=0)
        d[idx] = np.maximum(d[idx], np.maximum(v, 0.))
    d[d < 1e-9] = 0.  # sub-nanometre values on a zero rim are rounding, not relief
    return d


class Sheet:
    """Triangle sheet with 2D (projected) and 3D coordinates; edge splits keep orientation."""

    def __init__(self, P2, P3, faces, protected):
        self.P2 = [p for p in P2]
        self.P3 = [p for p in P3]
        self.F = [list(f) for f in faces]
        self.alive = [True] * len(self.F)
        self.E = {}
        self.VF = {}
        for fi, f in enumerate(self.F):
            self._add_face_links(fi)
        self.protected = set(protected)
        self.parents = {}  # new vertex -> (indices, weights)
        self.parent_face = list(range(len(self.F)))  # face -> original face index

    def _add_face_links(self, fi):
        a, b, c = self.F[fi]
        for x, y in ((a, b), (b, c), (c, a)):
            self.E.setdefault((min(x, y), max(x, y)), []).append(fi)
        for v in (a, b, c):
            self.VF.setdefault(v, set()).add(fi)

    def _drop_face_links(self, fi):
        a, b, c = self.F[fi]
        for x, y in ((a, b), (b, c), (c, a)):
            k = (min(x, y), max(x, y))
            self.E[k].remove(fi)
            if not self.E[k]:
                del self.E[k]
        for v in (a, b, c):
            self.VF[v].discard(fi)

    def _new_vertex(self, p2, p3, parents):
        self.P2.append(p2)
        self.P3.append(p3)
        self.parents[len(self.P2) - 1] = parents
        return len(self.P2) - 1

    def _replace(self, fi, faces):
        """Replace face fi by ``faces`` (first one reuses the index)."""
        src = self.parent_face[fi]
        self._drop_face_links(fi)
        self.F[fi] = list(faces[0])
        self._add_face_links(fi)
        out = [fi]
        for f in faces[1:]:
            self.F.append(list(f))
            self.alive.append(True)
            self.parent_face.append(src)
            self._add_face_links(len(self.F) - 1)
            out.append(len(self.F) - 1)
        return out

    def split_edge(self, a, b, s):
        k = (min(a, b), max(a, b))
        if k in self.protected:
            raise RuntimeError('protected edge split')
        w = self._new_vertex(self.P2[a] + s * (self.P2[b] - self.P2[a]), self.P3[a] + s * (self.P3[b] - self.P3[a]),
                             ((a, b), (1 - s, s)))
        for fi in list(self.E[k]):
            f = self.F[fi]
            i = [j for j in range(3) if f[j] in (a, b) and f[(j + 1) % 3] in (a, b)][0]
            p, q, o = f[i], f[(i + 1) % 3], f[(i + 2) % 3]
            self._replace(fi, [(p, w, o), (w, q, o)])
        return w

    def split_face(self, fi, l):
        a, b, c = self.F[fi]
        w = self._new_vertex(l[0] * self.P2[a] + l[1] * self.P2[b] + l[2] * self.P2[c],
                             l[0] * self.P3[a] + l[1] * self.P3[b] + l[2] * self.P3[c], ((a, b, c), tuple(l)))
        self._replace(fi, [(a, b, w), (b, c, w), (c, a, w)])
        return w

    def locate(self, p, cand):
        for fi in cand:
            a, b, c = (self.P2[v] for v in self.F[fi])
            den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / den
            l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / den
            l3 = 1 - l1 - l2
            if min(l1, l2, l3) >= -1e-10:
                return fi, (l1, l2, l3)
        return None, None

    def insert_point(self, p, cand, tol=1e-4):
        fi, l = self.locate(p, cand)
        if fi is None:
            return None
        f = self.F[fi]
        for v in f:
            if np.hypot(*(self.P2[v] - p)) < tol:
                return v
        for j in range(3):
            if l[j] < 1e-9:  # on the edge opposite vertex j
                a, b = f[(j + 1) % 3], f[(j + 2) % 3]
                if (min(a, b), max(a, b)) in self.protected:
                    return None
                ab = self.P2[b] - self.P2[a]
                s = float(((p - self.P2[a]) @ ab) / (ab @ ab))
                return self.split_edge(a, b, s)
        return self.split_face(fi, l)

    def insert_segment(self, va, vb, tol=1e-4):
        """Walk from va to vb splitting the crossed edges; True if the chain of edges va..vb exists."""
        cur = va
        pb = self.P2[vb]
        for _ in range(100000):
            if cur == vb or (min(cur, vb), max(cur, vb)) in self.E:
                return True
            pc = self.P2[cur]
            d = pb - pc
            step = None
            for fi in self.VF[cur]:
                f = self.F[fi]
                i = f.index(cur)
                b, c = f[(i + 1) % 3], f[(i + 2) % 3]
                xb, xc = self.P2[b] - pc, self.P2[c] - pc
                # d inside the wedge of the CCW triangle (cur, b, c): cross(xb, d) >= 0, cross(d, xc) >= 0
                if xb[0] * d[1] - xb[1] * d[0] >= -1e-14 and d[0] * xc[1] - d[1] * xc[0] >= -1e-14:
                    step = (b, c)
                    break
            if step is None:
                return False
            b, c = step
            pB, pC = self.P2[b], self.P2[c]
            e = pC - pB
            den = d[0] * e[1] - d[1] * e[0]
            if abs(den) < 1e-18:
                return False
            s = ((pB[0] - pc[0]) * d[1] - (pB[1] - pc[1]) * d[0]) / den   # param on b->c
            t = ((pB[0] - pc[0]) * e[1] - (pB[1] - pc[1]) * e[0]) / den   # param on cur->vb
            if t > 1 + 1e-9:
                return False
            q = pB + s * e
            if np.hypot(*(q - pB)) < tol:
                cur = b
                continue
            if np.hypot(*(q - pC)) < tol:
                cur = c
                continue
            if (min(b, c), max(b, c)) in self.protected:
                return False
            cur = self.split_edge(b, c, float(s))
        return False

    def refine(self, target):
        """Longest-edge bisection (Rivara): an unprotected edge is split when it is the longest edge of
        every triangle it borders and longer than max(target, 0.75 x the longest protected edge of an
        adjacent triangle). Always terminates and never degrades the triangles; edges next to long
        protected edges may stay above target (reported)."""
        prot_len = {k: float(np.linalg.norm(self.P3[k[0]] - self.P3[k[1]])) for k in self.protected}
        elen = lambda k: float(np.linalg.norm(self.P3[k[0]] - self.P3[k[1]]))
        passes = 0
        while True:
            passes += 1
            longest = {}
            for fi, f in enumerate(self.F):
                ks = [(min(x, y), max(x, y)) for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))]
                Ls = [elen(k) for k in ks]
                longest[fi] = (ks[int(np.argmax(Ls))], max(Ls), max([.75 * prot_len[k] for k in ks if k in prot_len] or [0.]))
            cand = []
            for k, fs in self.E.items():
                if k in self.protected:
                    continue
                if not all(longest[fi][0] == k for fi in fs):
                    continue
                L = elen(k)
                if L > max([target] + [longest[fi][2] for fi in fs]):
                    cand.append((L, k))
            if not cand:
                return passes
            cand.sort(reverse=True)
            used = set()
            for L, k in cand:
                fs = self.E.get(k)
                if not fs or any(fi in used for fi in fs):
                    continue
                used.update(fs)
                nb = len(self.F)
                self.split_edge(k[0], k[1], .5)
                used.update(range(nb, len(self.F)))


def floor_relief(V, F, sheet_faces, cell_keys, m, frame_centre, opts, cell_of, segment_distance):
    """Relief on the floor sheets. V, F: full mesh; sheet_faces: indices of the selected floor faces;
    cell_keys: their cell. Returns dict with new faces (replacing sheet_faces), new vertex positions,
    parents (for per-vertex fields), parent face per new face, displacement per vertex and stats."""
    m = np.asarray(m, float)
    rng = np.random.default_rng(int(opts.get('seed', 41)) + 811)
    b1 = np.cross(m, [0., 0., 1.]) if abs(m[2]) < .9 else np.cross(m, [1., 0., 0.])
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(m, b1)
    to2 = lambda X: np.c_[(X - frame_centre) @ b1, (X - frame_centre) @ b2]
    allv = np.unique(F[sheet_faces])
    Q = to2(V[allv])
    lo, hi = Q.min(0) - .35, Q.max(0) + .35
    chips = opts.get('kind', 'facets') == 'chips'
    field = None
    if not chips:
        r_lo, r_hi = opts.get('node_radius_m', (.08, .17))
        field = facet_field(rng, lo, hi, r_lo, r_hi, float(opts.get('merge_share', .3)))
    amp = float(opts.get('amplitude', 1.)) * float(opts.get('max_m', .015))
    target, ramp_m, edge_m = float(opts.get('target_m', .01)), float(opts.get('ramp_m', .02)), float(opts.get('edge_m', .01))
    nV = len(V)
    out_faces, out_parent_face, new_pos, parents = [], [], [], []
    disp = {}
    orig_moves = {}  # original interior sheet vertices: new position
    stats = {'cells': {}}
    if field is not None:
        stats['field'] = {'nodes': int(len(field['P'])), 'triangles': int(len(field['T'])), 'merged_pairs': int(field['merged_pairs'])}
    all_tents = []
    cell_arr = np.asarray(cell_keys)
    for cell in sorted(set(cell_arr.tolist())):
        fsel = np.asarray(sheet_faces)[cell_arr == cell]
        Fs = F[fsel]
        vids = np.unique(Fs)
        loc = {int(v): i for i, v in enumerate(vids)}
        P3 = V[vids]
        P2 = to2(P3)
        Fl = np.vectorize(loc.get)(Fs)
        a2 = (P2[Fl[:, 1], 0] - P2[Fl[:, 0], 0]) * (P2[Fl[:, 2], 1] - P2[Fl[:, 0], 1]) - \
             (P2[Fl[:, 1], 1] - P2[Fl[:, 0], 1]) * (P2[Fl[:, 2], 0] - P2[Fl[:, 0], 0])
        flipped = False
        if (a2 > 0).all():
            pass
        elif (a2 < 0).all():
            Fl = Fl[:, ::-1]  # CCW in the projection for the mesh operations; restored on output
            flipped = True
        else:
            raise RuntimeError('floor sheet of cell %s folds in projection along m' % cell)
        ecount = {}
        for f in Fl:
            for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                k = (min(x, y), max(x, y))
                ecount[k] = ecount.get(k, 0) + 1
        prot = [k for k, c in ecount.items() if c == 1]
        sh = Sheet(P2, P3, Fl, prot)
        PS = np.array([[P2[a], P2[b]] for a, b in prot])
        n_runs = n_ok = 0
        tents = []
        if chips:
            # facet edge lies >= edge_margin inside the protected edges (edge planes).
            tents = chip_tents(rng, P2, Fl, PS, opts)
            snap = float(opts.get('snap_m', 1e-6))  # crease vertices stay on their lines (v1: 0.1 mm)
            for p_, q_ in chip_creases(tents):
                n_runs += 1
                va = sh.insert_point(p_, list(range(len(sh.F))), tol=snap)
                vb = sh.insert_point(q_, list(range(len(sh.F))), tol=snap) if va is not None else None
                if va is not None and vb is not None and (va == vb or sh.insert_segment(va, vb, tol=snap)):
                    n_ok += 1
            all_tents.extend(tents)
        creases = []
        if not chips:
            # crease lines: field edges between different facets, inside the sheet and >= edge_m from
            # protected edges (sampled every 2 mm)
            T, fid = field['T'], field['fid']
            ed = {}
            for ti, t in enumerate(T):
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    ed.setdefault((min(a, b), max(a, b)), []).append(ti)
            creases = [k for k, ts in ed.items() if len(ts) == 1 or fid[ts[0]] != fid[ts[1]]]
        bmin, bmax = P2.min(0) - .01, P2.max(0) + .01
        for a, b in creases:
            A, B = field['P'][a], field['P'][b]
            if (np.maximum(A, B) < bmin).any() or (np.minimum(A, B) > bmax).any():
                continue
            L = float(np.linalg.norm(B - A))
            ns = max(2, int(math.ceil(L / .002)) + 1)
            S = A[None] + np.linspace(0, 1, ns)[:, None] * (B - A)[None]
            far = _seg_dist2d(S, PS) >= edge_m
            # inside the sheet: inside some sheet triangle (vectorised barycentric over the original faces)
            inside = np.zeros(ns, bool)
            fa, fb, fc = P2[Fl[:, 0]], P2[Fl[:, 1]], P2[Fl[:, 2]]
            near = np.nonzero(((np.minimum(np.minimum(fa, fb), fc) <= np.maximum(A, B) + .01).all(1)) &
                              ((np.maximum(np.maximum(fa, fb), fc) >= np.minimum(A, B) - .01).all(1)))[0]
            if len(near):
                a_, b_, c_ = fa[near], fb[near], fc[near]
                den = (b_[:, 1] - c_[:, 1]) * (a_[:, 0] - c_[:, 0]) + (c_[:, 0] - b_[:, 0]) * (a_[:, 1] - c_[:, 1])
                l1 = ((b_[:, 1] - c_[:, 1])[None] * (S[:, :1] - c_[:, 0][None]) + (c_[:, 0] - b_[:, 0])[None] * (S[:, 1:] - c_[:, 1][None])) / den[None]
                l2 = ((c_[:, 1] - a_[:, 1])[None] * (S[:, :1] - c_[:, 0][None]) + (a_[:, 0] - c_[:, 0])[None] * (S[:, 1:] - c_[:, 1][None])) / den[None]
                inside = ((l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)).any(1)
            ok = far & inside
            # runs of ok samples
            i = 0
            while i < ns:
                if not ok[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < ns and ok[j + 1]:
                    j += 1
                if j > i:
                    n_runs += 1
                    cand = list(range(len(sh.F)))
                    va = sh.insert_point(S[i], cand)
                    vb = sh.insert_point(S[j], list(range(len(sh.F)))) if va is not None else None
                    if va is not None and vb is not None and va != vb and sh.insert_segment(va, vb):
                        n_ok += 1
                i = j + 1
        passes = sh.refine(target)
        # displacement
        nloc = len(sh.P2)
        Pf3 = np.array(sh.P3)
        Pf2 = np.array(sh.P2)
        Ff = np.array(sh.F)
        is_prot_v = np.zeros(nloc, bool)
        for k in sh.protected:
            is_prot_v[list(k)] = True
        if chips:
            # tent depths are absolute (A <= max_m); amplitude scales them (0: same mesh, no relief)
            d = float(opts.get('amplitude', 1.)) * tents_eval(tents, Pf2)
        else:
            segs3 = np.array([[sh.P3[a], sh.P3[b]] for a, b in sh.protected])
            dist, _ = segment_distance(Pf3, segs3)
            x = np.clip(dist / ramp_m, 0, 1)
            ramp = x * x * (3 - 2 * x)
            hval, _ = field_eval(field, Pf2)
            if np.isnan(hval).any():
                raise RuntimeError('facet field does not cover the sheet')
            d = amp * ramp * hval
        if chips and (d[is_prot_v] != 0).any():
            raise RuntimeError('chip field nonzero on a protected vertex')
        d[is_prot_v] = 0.
        Pf3 = Pf3 + d[:, None] * m[None]
        # stays in its own cell (convex): every displaced vertex and every face centroid with a
        # displaced vertex
        moved = np.nonzero(d > 0)[0]
        bad_v = sum(1 for i in moved if cell_of(Pf3[i]) != cell)
        fm = np.nonzero((d[Ff] > 0).any(1))[0]
        cen = Pf3[Ff[fm]].mean(1)
        bad_c = sum(1 for p in cen if cell_of(p) != cell)
        a2n = (Pf2[Ff[:, 1], 0] - Pf2[Ff[:, 0], 0]) * (Pf2[Ff[:, 2], 1] - Pf2[Ff[:, 0], 1]) - \
              (Pf2[Ff[:, 1], 1] - Pf2[Ff[:, 0], 1]) * (Pf2[Ff[:, 2], 0] - Pf2[Ff[:, 0], 0])
        # map local -> global indices
        gidx = np.empty(nloc, np.int64)
        gidx[:len(vids)] = vids
        for i in range(len(vids), nloc):
            gidx[i] = nV + len(new_pos)
            new_pos.append(Pf3[i])
            idx, wts = sh.parents[i]
            parents.append((tuple(int(gidx[j]) for j in idx), tuple(float(w) for w in wts)))
        moved_orig = {int(vids[i]): Pf3[i] for i in range(len(vids)) if d[i] != 0}
        orig_moves.update(moved_orig)
        for i in range(nloc):
            disp[int(gidx[i])] = float(d[i])
        out_faces.append(gidx[Ff][:, ::-1] if flipped else gidx[Ff])
        out_parent_face.append(fsel[np.array(sh.parent_face)])
        stats['cells'][str(cell)] = {'faces_in': int(len(fsel)), 'faces_out': int(len(Ff)), 'vertices_added': int(nloc - len(vids)),
                                     'protected_edges': int(len(prot)), 'crease_runs': n_runs, 'crease_runs_inserted': n_ok,
                                     'refine_passes': passes, 'displaced_vertices': int(len(moved)),
                                     'left_cell_vertices': int(bad_v), 'left_cell_centroids': int(bad_c),
                                     'min_projected_area_m2': float(np.abs(a2n).min() / 2),
                                     'orientation_consistent': bool((a2n > 0).all())}
        if chips:
            sheet_area = float(np.abs(a2n).sum() / 2)
            stats['cells'][str(cell)].update(tents=len(tents), displaced_area_share=float((np.abs(a2n)[(d[Ff] > 0).all(1)]).sum() / 2 / max(sheet_area, 1e-12)))
            # self-check at relief time: vertices within the edge margin of a protected segment
            # (projected distance, as the edge planes) must not be displaced
            band = _seg_dist2d(Pf2, PS) < float(opts.get('edge_margin_m', .003)) - 1e-9 if len(PS) else np.zeros(nloc, bool)
            stats['cells'][str(cell)].update(band_vertices=int(band.sum()), band_displaced=int((d[band] > 0).sum()),
                                             band_max_d_m=float(d[band].max()) if band.any() else 0.)
    if chips and all_tents:
        dia = np.array([2 * math.sqrt(abs(_poly_area(t['support'])) / math.pi) for t in all_tents])
        stats['tents'] = {'count': len(all_tents),
                          'support_diameter_cm_p10_p50_p90': [float(x) for x in np.percentile(dia * 100, [10, 50, 90])],
                          'depth_mm_p10_p50_p90': [float(x) for x in np.percentile([t['depth'] * 1e3 for t in all_tents], [10, 50, 90])],
                          'steep_facet_deg_p10_p50_p90': [float(x) for x in np.percentile([t['steep_deg'] for t in all_tents], [10, 50, 90])],
                          'edge_plane_tents': int(sum(1 for t in all_tents if len(t['planes']) > t['n_dirs']))}
    return {'faces': np.concatenate(out_faces), 'parent_face': np.concatenate(out_parent_face),
            'new_positions': np.array(new_pos).reshape(-1, 3), 'moved_original': orig_moves, 'parents': parents, 'disp': disp, 'field': field, 'tents': all_tents,
            'basis': (b1, b2), 'stats': stats}
