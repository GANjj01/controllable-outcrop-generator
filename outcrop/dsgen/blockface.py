"""Route B sample: explicit residual blocks on a small patch.

Every surface triangle of the patch is the top of a prism along the patch's inward normal m. The prisms are cut into
convex pieces by geological planes -- bedding partings and v2 bedding contacts (levels of the bedding coordinate) and
the v2 joint pieces of the patch's own fault block. Each piece belongs to one cell (bedding level x joint slab); the
cell's spalling depth R (hash, as in fracture.py) is applied as a floor: the piece keeps the part behind its floor
plane, which is its top triangle moved along m by R x t (t = taper to 0 at the patch boundary, linear per triangle).
Emitted surface: each piece's floor cap, and the part of each piece face on a cutting plane that the neighbouring
piece's deeper floor exposes (a riser lying exactly in the geological plane). No hidden faces, no prism side walls; at
the boundary t = 0, so the floors are the original triangles."""
import math

import numpy as np

EPS = 1e-9


def _cross(a, b):
    """np.cross of two 3-vectors with the same operations (bit-identical), without its axis handling."""
    a0, a1, a2 = a
    b0, b1, b2 = b
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _clip_d(poly, ds):
    """_clip with the signed distances ds[i] = poly[i].n - d given (computed once per polygon)."""
    out = []
    k = len(poly)
    for i in range(k):
        da, db = ds[i], ds[(i + 1) % k]
        if da <= EPS:
            out.append(poly[i])
        if (da < -EPS and db > EPS) or (da > EPS and db < -EPS):
            a = poly[i]
            out.append(a + (poly[(i + 1) % k] - a) * (da / (da - db)))
    return out


def _clip(poly, n, d):
    """Keep the part of a convex polygon with n.x <= d."""
    return _clip_d(poly, [a @ n - d for a in poly])


class Piece:
    """Convex polyhedron: faces = list of (polygon, tag); tag = None (prism side / top / bottom) or
    (family, level index, side) for faces on a cutting plane."""

    def __init__(self, faces):
        self.faces = faces

    def split(self, n, d, tag, cap_tol=1e-7, project=False):
        """Two pieces (n.x <= d, n.x >= d); the new cap faces get tags (tag, -1) / (tag, +1). cap_tol: existing corners within
        this distance of the plane join the cap as they are (not projected). The default 1e-7 is the historical rule;
        a corner farther than EPS (1e-9) is also cut, so its edge intersections join the cap too and the cap leaves
        its plane by up to 1e-7. The whole-wall scene passes 1e-12 for floor caps (opt-in blockface floor_cap_tol):
        only line-plane intersections and corners really on the plane."""
        lo, hi, pts = [], [], []
        for poly, t in self.faces:
            # one signed distance per vertex; the far side uses its negation, (-x).(-n) - (-d) is
            # bit-identical to -(x.n - d), so the result equals clipping with (n, d) and (-n, -d)
            ds = [p @ n - d for p in poly]
            a = _clip_d(poly, ds)
            b = _clip_d(poly, [-x for x in ds])
            if len(a) >= 3:
                lo.append((a, t))
            if len(b) >= 3:
                hi.append((b, t))
            for p, dp in zip(poly, ds):
                if abs(dp) <= cap_tol:
                    # face in a cut plane has no corner up to cap_tol off it
                    pts.append(p - (dp / (n @ n)) * n if project else p)
            k = len(poly)
            for i in range(k):
                dp, dq = ds[i], ds[(i + 1) % k]
                if (dp < -EPS and dq > EPS) or (dp > EPS and dq < -EPS):
                    p = poly[i]
                    pts.append(p + (poly[(i + 1) % k] - p) * (dp / (dp - dq)))
        if len(lo) == 0 or len(hi) == 0 or len(pts) < 3:
            return (self, None) if len(lo) else (None, self)
        cap = _order(np.array(pts), n)
        if len(cap) < 3:
            return (self, None) if len(lo) else (None, self)
        lo.append((cap, (tag, -1)))          # on the low side the cap's outward normal is +n
        hi.append((cap[::-1], (tag, +1)))
        return Piece(lo), Piece(hi)

    def centroid(self):
        return np.concatenate([np.array(p) for p, _ in self.faces]).mean(0)


def _order(pts, n):
    """Unique coplanar points ordered counter-clockwise around n."""
    keep = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < 1e-8 for q in keep):
            keep.append(p)
    if len(keep) < 3:
        return keep
    P = np.array(keep)
    c = P.mean(0)
    u = P[0] - c
    u -= n * (u @ n)
    u /= max(np.linalg.norm(u), 1e-15)
    v = _cross(n, u)
    ang = np.arctan2((P - c) @ v, (P - c) @ u)
    return [P[i] for i in np.argsort(ang)]


def _area_normal(poly):
    P = np.array(poly)
    s = np.zeros(3)
    for i in range(len(P)):
        s += _cross(P[i], P[(i + 1) % len(P)])
    return .5 * s


def build_patch(V, F, inside_faces, taper, m, cut_planes, cell_of, depth_of, exact=False, planes_of=None, floor_cap_tol=1e-7,
                bottom_depth=1., project_caps=False):
    """Residual-block surface of the patch triangles ``inside_faces`` (indices into F).

    taper: per-vertex t in [0, 1] (0 at the patch boundary); m: unit inward direction; cut_planes: list of (family, level
    index, normal, offset) -- the planes that may cross the patch; cell_of(point) -> cell key (hashable);
    depth_of(cell) -> R. ``exact``: the riser footprint is clipped with unit edge normals (otherwise |normal| = |edge|
    and the 1e-9 EPS of _clip is a distance of 1e-9 / |edge|: 100 um on a 10 um edge) and every riser of positive area
    is kept (otherwise risers under 1e-8 m^2 are dropped: up to 43 um wide on a sub-mm mesh). Each dropped riser
    leaves an open slit of its width in its plane, which the 0.5 mm weld used to close; with exact the raw output
    closes without welding. False: unchanged. ``planes_of`` (opt-in; build_patch_parallel): per prism (in the order of
    inside_faces) the indices of the cut planes it crosses (crossing_planes, the same test), so the others are not
    tested again. ``floor_cap_tol`` (opt-in, whole-wall scene): Piece.split cap_tol of the floor caps (default: the
    historical 1e-7). ``bottom_depth``: prism depth along m; every floor and riser lies within the deepest floor
    (depth x taper <= depth), so a prism slightly deeper than that gives the same surface with fewer pieces. Returns
    polygons [(points, kind, info)] with kind 'floor' or 'riser', and stats."""
    polys = []
    stats = {'prisms': 0, 'pieces': 0}
    for pi_, fi in enumerate(inside_faces):
        tri = V[F[fi]]
        tv = taper[F[fi]]
        top = [tri[0], tri[1], tri[2]]
        bot = [p + m * bottom_depth for p in tri]
        nrm = _cross(tri[1] - tri[0], tri[2] - tri[0])
        if nrm @ (-m) < 0:  # top faces outward (-m)
            top = top[::-1]
            bot = bot[::-1]
        faces = [(top, 'top'), (bot[::-1], 'bottom')]
        for i in range(3):
            j = (i + 1) % 3
            faces.append(([top[i], bot[i], bot[j], top[j]], 'side'))
        pieces = [Piece(faces)]
        stats['prisms'] += 1
        lo_t, hi_t = None, None
        for fam, lev, n, d in (cut_planes if planes_of is None else [cut_planes[j] for j in planes_of[pi_]]):
            vals = np.array([p @ n for p in top + bot])
            if vals.min() >= d - EPS or vals.max() <= d + EPS:
                continue
            new = []
            for pc in pieces:
                a, b = pc.split(n, d, (fam, lev), project=project_caps)
                for x in (a, b):
                    if x is not None:
                        new.append(x)
            pieces = new
        stats['pieces'] += len(pieces)
        # floor plane of each cell in this prism: top triangle moved by R x t along m
        recs = []
        for pc in pieces:
            c = pc.centroid()
            cell = cell_of(c)
            R = depth_of(cell)
            fl = [tri[k] + m * R * tv[k] for k in range(3)]
            fn = _cross(fl[1] - fl[0], fl[2] - fl[0])
            fn /= max(np.linalg.norm(fn), 1e-15)
            if fn @ m < 0:
                fn = -fn                      # points into the rock
            fd = fn @ fl[0]
            recs.append((pc, cell, R, fn, fd))
        # floor caps: the piece cut by its floor plane (keep the rock side, fn.x >= fd)
        for pc, cell, R, fn, fd in recs:
            keep, _ = pc.split(-fn, -fd, ('floor', 0), cap_tol=floor_cap_tol)
            if keep is None:
                continue
            # the floor cap; where R x t = 0 the floor is the top itself and the kept top is exposed
            # (only when no cap was cut: the two coincide there)
            caps = [poly for poly, t in keep.faces if t == (('floor', 0), -1)]
            if not caps:
                caps = [poly for poly, t in keep.faces if t == 'top' and np.linalg.norm(_area_normal(poly)) > 1e-10
                        and all(abs(p @ fn - fd) < 1e-7 for p in poly)]
            for poly in caps:
                polys.append((poly, 'floor', {'cell': cell, 'R': R, 'prism': int(fi)}))
        # risers: a piece face on a cutting plane, behind its own floor, in front of the facing
        # piece's floor (that piece is removed there), restricted to their common footprint
        for ia, (pa, ca, Ra, fna, fda) in enumerate(recs):
            for poly, t in pa.faces:
                if not isinstance(t, tuple) or t[0] == ('floor', 0):
                    continue
                plane_key, side = t
                face = _clip(poly, -fna, -fda)  # behind own floor
                if len(face) < 3:
                    continue
                for ib, (pb, cb, Rb, fnb, fdb) in enumerate(recs):
                    if ib == ia or Rb <= Ra + 1e-6:
                        continue
                    for polyb, tb in pb.faces:
                        if tb != (plane_key, -side):
                            continue
                        # common footprint: clip A's face by B's face edges (coplanar, opposite winding)
                        exp = face
                        nb_ = _area_normal(polyb)
                        nb_ /= max(np.linalg.norm(nb_), 1e-15)
                        Pb = np.array(polyb)
                        for k in range(len(Pb)):
                            e = Pb[(k + 1) % len(Pb)] - Pb[k]
                            inward = _cross(nb_, e)  # B's interior side in its own winding
                            if exact:
                                inward /= max(np.linalg.norm(inward), 1e-300)
                            exp = _clip(exp, -inward, -(inward @ Pb[k]))
                            if len(exp) < 3:
                                break
                        if len(exp) < 3:
                            continue
                        exp = _clip(exp, fnb, fdb)   # in front of B's floor (B removed)
                        if len(exp) >= 3 and np.linalg.norm(_area_normal(exp)) > (0. if exact else 1e-8):
                            polys.append((exp, 'riser', {'plane': plane_key, 'cell': ca, 'R': Ra, 'R_open': Rb, 'prism': int(fi)}))
    return polys, stats


def patch_setup(V, F, truth, frame, *, depth, parting, seed):
    """Planes and cells of a B patch: the triangles fully inside the L x L square (frame: centre, outward normal, e1, e2, L),
    the boundary taper (0 within 0.15 m of the square edge, 1 beyond 0.6 m), the bedding levels (irregular partings +
    v2 contacts bisected to 1e-7 m) and the own-block v2 joint pieces crossing the patch."""
    from .truth import _plane_groups
    c, n0, e1, e2 = (np.asarray(frame[k], float) for k in ('centre', 'normal', 'e1', 'e2'))
    L = float(frame['L'])
    m = -n0
    uv = np.c_[(V - c) @ e1, (V - c) @ e2]
    vin = (np.abs(uv) <= L / 2).all(1)
    inside = np.nonzero(vin[F].all(1))[0]
    dist = L / 2 - np.abs(uv).max(1)
    x = np.clip((dist - .15) / .45, 0, 1)
    taper = x * x * (3 - 2 * x)
    faults = [(n, o) for iid, iv, if_ in truth.interfaces if iid.startswith('fault') and len(if_) for n, o, _ in _plane_groups(iv, if_)]
    block_of = lambda q: sum(((q @ n - o) > 0).astype(np.int64) << i for i, (n, o) in enumerate(faults))
    PV = V[np.unique(F[inside])]
    blocks = np.unique(block_of(PV))
    if len(blocks) != 1:
        raise ValueError('B patch spans %d fault blocks' % len(blocks))
    blk = int(blocks[0])
    corners = np.concatenate([PV, PV + m])
    joint_planes = {}
    for iid, iv, if_ in truth.interfaces:
        if not iid.startswith('joints') or not len(if_):
            continue
        for n, o, tri in _plane_groups(iv, if_):
            if int(block_of(tri.reshape(-1, 3).mean(0)[None])[0]) != blk:
                continue
            dd = corners @ n - o
            if dd.min() < 0 < dd.max():
                joint_planes.setdefault(iid.split(':', 1)[0], []).append((n, o, tri, iid))
    st, body, _, _ = truth.query(c[None])
    ok, nbs, tbs = truth.fabric(c[None], body, 'bedding')
    nb, C = nbs[0], float(tbs[0] - nbs[0] @ c)
    tvals = corners @ nb + C
    tlo, thi = tvals.min() - .01, tvals.max() + .01
    line_t = np.arange(tlo, thi, .002)
    _, _, uq, _ = truth.query(c[None] + (line_t - (c @ nb + C))[:, None] * nb[None])
    contacts = []
    for i in range(len(line_t) - 1):
        if uq[i] == uq[i + 1]:
            continue
        lo_, hi_ = line_t[i], line_t[i + 1]
        for _ in range(25):
            mid = .5 * (lo_ + hi_)
            _, _, um, _ = truth.query((c + (mid - (c @ nb + C)) * nb)[None])
            lo_, hi_ = (mid, hi_) if um[0] == uq[i] else (lo_, mid)
        contacts.append(float(.5 * (lo_ + hi_)))
    g = np.random.default_rng(seed + 307)
    lv, partings = tlo - parting * g.uniform(0, 1), []
    while lv < thi:
        partings.append(lv)
        lv += parting * g.uniform(.4, 1.6)
    levels = sorted(set(np.round(partings + contacts, 6)))
    cut = [('bed', i, nb, lv - C) for i, lv in enumerate(levels)]
    jsets = sorted(joint_planes)
    joint_offs = {}
    # cutting planes that are v2 interfaces (rule 2: their risers are labelled as interface faces)
    coplanar = {}
    contact_r = set(np.round(contacts, 6))
    for i, lv in enumerate(levels):
        if lv in contact_r:
            _, _, ua, _ = truth.query(np.array([c + (lv - 1e-3 - (c @ nb + C)) * nb, c + (lv + 1e-3 - (c @ nb + C)) * nb]))
            coplanar['bed|%d' % i] = {'type': 'contact', 'id': 'contact:%s|%s' % (ua[0], ua[1])}
    for sid in jsets:
        ref_n = joint_planes[sid][0][0]
        signed = sorted(((o if n @ ref_n > 0 else -o), iid) for n, o, _, iid in joint_planes[sid])
        offs = [o for o, _ in signed]
        joint_offs[sid] = (ref_n, np.array(offs))
        for i, (o, iid) in enumerate(signed):
            cut.append(('joint:' + sid, i, ref_n, o))
            coplanar['joint:%s|%d' % (sid, i)] = {'type': 'joint', 'id': iid}
    M = 2147483647

    def cell_of(p):
        k = int(np.searchsorted(levels, p @ nb + C)) % M
        for j, sid in enumerate(jsets):
            n, offs = joint_offs[sid]
            k = (k * 48271 + int(np.searchsorted(offs, p @ n)) + 7919 * (j + 1)) % M
        return k

    def depth_of(cell):
        h = ((cell * 16807 + seed) % M * 48271 % M) / M
        return depth * h ** 1.5

    info = {'block': blk, 'triangles': int(len(inside)), 'contacts_t': contacts, 'partings': len(partings),
            'joints': {s: [x[3] for x in joint_planes[s]] for s in jsets}, 'coplanar': coplanar}
    return inside, taper, m, cut, cell_of, depth_of, info


def region_setup(V, F, truth, inside, taper, m, *, depth, parting, seed, region, spall=None):
    """patch_setup for one region of the whole-wall scene: the faces ``inside`` (all in one fault block) with the per-vertex
    taper and B direction m given by the caller. Bedding levels: partings (own random stream per region) and the v2
    contacts of this block, each bisected to 1e-7 m on a short segment along the bedding normal next to a pair of
    region vertices of the two units (inside the rock, 5 cm along m), not on one line through the patch centre (a long
    line leaves the block or the rock). Joint planes: the block's own pieces crossing the region's prisms (1 m deep),
    as patch_setup. Cell keys carry the region index. Returns as patch_setup (inside, taper, m, cut, cell_of,
    depth_of, info) and the RegionCells object."""
    from .truth import _plane_groups
    m = np.asarray(m, float)
    faults = [(n, o) for iid, iv, if_ in truth.interfaces if iid.startswith('fault') and len(if_) for n, o, _ in _plane_groups(iv, if_)]
    block_of = lambda q: sum(((q @ n - o) > 0).astype(np.int64) << i for i, (n, o) in enumerate(faults))
    vid = np.unique(F[inside])
    PV = V[vid]
    blocks = np.unique(block_of(PV))
    if len(blocks) != 1:
        raise ValueError('B region spans %d fault blocks' % len(blocks))
    blk = int(blocks[0])
    corners = np.concatenate([PV, PV + m])
    joint_planes = {}
    for iid, iv, if_ in truth.interfaces:
        if not iid.startswith('joints') or not len(if_):
            continue
        for n, o, tri in _plane_groups(iv, if_):
            if int(block_of(tri.reshape(-1, 3).mean(0)[None])[0]) != blk:
                continue
            dd = corners @ n - o
            if dd.min() < 0 < dd.max():
                joint_planes.setdefault(iid.split(':', 1)[0], []).append((n, o, tri, iid))
    probe = PV + .05 * m
    st, body, unit, lith_p = truth.query(probe)
    occ = st == 'occupied'
    ok, nbs, tbs = truth.fabric(probe[occ], body[occ], 'bedding')
    if not ok.all():
        raise ValueError('region points without bedding fabric')
    nb = nbs[0]
    if np.abs(nbs @ nb).min() < 1 - 1e-9:
        raise ValueError('bedding normal not constant in the region (fold?): region_setup needs planar beds')
    P_o, U_o, T_o = probe[occ], unit[occ], tbs
    Cs = T_o - P_o @ nb
    C = float(np.median(Cs))
    if np.abs(Cs - C).max() > 1e-6:
        raise ValueError('bedding coordinate offset not constant in the region (faulted block?)')
    tvals = corners @ nb + C
    tlo, thi = tvals.min() - .01, tvals.max() + .01
    contacts = []
    units_sorted = []
    for u_ in np.unique(U_o):
        tt = T_o[U_o == u_]
        units_sorted.append((float(np.median(tt)), u_, float(tt.min()), float(tt.max())))
    units_sorted.sort()
    if spall is not None:
        # per contact and step; each contact's sequence of brackets is unchanged)
        brk = []
        for (_, ua, amin, amax), (_, ub, bmin, bmax) in zip(units_sorted[:-1], units_sorted[1:]):
            ia = np.flatnonzero(U_o == ua)
            ib = np.flatnonzero(U_o == ub)
            ja = ia[np.argmax(T_o[ia])]
            jb = ib[np.argmin(T_o[ib])]
            lo_, hi_ = float(T_o[ja]), float(T_o[jb])
            if not lo_ < hi_:
                raise ValueError('units %s / %s overlap in bedding coordinate' % (ua, ub))
            brk.append([lo_, hi_, ua, ub, P_o[jb] - T_o[jb] * nb])
        for _ in range(60):
            act = [b_ for b_ in brk if not b_[1] - b_[0] <= 1e-7]
            if not act:
                break
            mids = [.5 * (b_[0] + b_[1]) for b_ in act]
            _, _, um, _ = truth.query(np.array([b_[4] + mid * nb for b_, mid in zip(act, mids)]))
            for b_, mid, u_ in zip(act, mids, um):
                b_[0], b_[1] = (mid, b_[1]) if u_ == b_[2] else (b_[0], mid)
        if brk:
            _, _, chk, _ = truth.query(np.array([q_ for b_ in brk for q_ in (b_[4] + (b_[0] - 1e-4) * nb, b_[4] + (b_[1] + 1e-4) * nb)]))
            for k_, b_ in enumerate(brk):
                if chk[2 * k_] != b_[2] or chk[2 * k_ + 1] != b_[3]:
                    raise ValueError('contact %s|%s not bracketed by its probe line' % (b_[2], b_[3]))
                contacts.append((float(.5 * (b_[0] + b_[1])), b_[2], b_[3]))
    for (_, ua, amin, amax), (_, ub, bmin, bmax) in (zip(units_sorted[:-1], units_sorted[1:]) if spall is None else []):
        # the closest pair (in t) of region points of the two units brackets their contact
        ia = np.flatnonzero(U_o == ua)
        ib = np.flatnonzero(U_o == ub)
        ja = ia[np.argmax(T_o[ia])]
        jb = ib[np.argmin(T_o[ib])]
        lo_, hi_ = float(T_o[ja]), float(T_o[jb])
        if not lo_ < hi_:
            raise ValueError('units %s / %s overlap in bedding coordinate' % (ua, ub))
        base = P_o[jb] - T_o[jb] * nb      # line through the upper point along nb, parametrised by t
        for _ in range(60):
            if hi_ - lo_ <= 1e-7:
                break
            mid = .5 * (lo_ + hi_)
            _, _, um, _ = truth.query((base + mid * nb)[None])
            lo_, hi_ = (mid, hi_) if um[0] == ua else (lo_, mid)
        _, _, chk, _ = truth.query(np.array([base + (lo_ - 1e-4) * nb, base + (hi_ + 1e-4) * nb]))
        if chk[0] != ua or chk[1] != ub:
            raise ValueError('contact %s|%s not bracketed by its probe line' % (ua, ub))
        contacts.append((float(.5 * (lo_ + hi_)), ua, ub))
    g = np.random.default_rng(seed + 307 + 7919 * (region + 1))
    lv, partings = tlo - parting * g.uniform(0, 1), []
    while lv < thi:
        partings.append(lv)
        lv += parting * g.uniform(.4, 1.6)
    levels = sorted(set(np.round(partings + [c_ for c_, _, _ in contacts], 6)))
    cut = [('bed', i, nb, lv - C) for i, lv in enumerate(levels)]
    jsets = sorted(joint_planes)
    joint_offs = {}
    coplanar = {}
    contact_r = {round(c_, 6): (ua, ub) for c_, ua, ub in contacts}
    for i, lv in enumerate(levels):
        if lv in contact_r:
            ua, ub = contact_r[lv]
            coplanar['bed|%d' % i] = {'type': 'contact', 'id': 'contact:%s|%s' % (ua, ub)}
    for sid in jsets:
        ref_n = joint_planes[sid][0][0]
        signed = sorted(((o if n @ ref_n > 0 else -o), iid) for n, o, _, iid in joint_planes[sid])
        offs = [o for o, _ in signed]
        joint_offs[sid] = (ref_n, np.array(offs))
        for i, (o, iid) in enumerate(signed):
            cut.append(('joint:' + sid, i, ref_n, o))
            coplanar['joint:%s|%d' % (sid, i)] = {'type': 'joint', 'id': iid}
    if spall is not None:
        lith_of_unit = {}
        for u_, l_ in zip(unit[occ], lith_p[occ]):
            lith_of_unit.setdefault(u_, l_)
        cells, cross_cut, sp_info = spall_cells(levels, C, nb, jsets, joint_offs, seed, depth, region, m, corners, contacts,
                                                units_sorted, lith_of_unit, spall)
        cut += cross_cut
        info_spall = sp_info
    else:
        cells = RegionCells(levels, C, nb, jsets, joint_offs, seed, depth, region)
        info_spall = None
    info = {'region': int(region), 'block': blk, 'triangles': int(len(inside)), 'contacts_t': [c_ for c_, _, _ in contacts],
            'contacts': ['%s|%s' % (ua, ub) for _, ua, ub in contacts], 'partings': len(partings), 'bedding_C': C,
            'bedding_normal': nb.tolist(), 'joints': {s_: [x[3] for x in joint_planes[s_]] for s_ in jsets}, 'coplanar': coplanar}
    if info_spall is not None:
        info['spall'] = info_spall
    return (inside, taper, m, cut, cells.cell_of, cells.depth_of, info), cells


class RegionCells:
    """Cell keys and spalling depth of a region_setup region (picklable, for the parallel build_patch): the
    patch_setup hash with the region index mixed in."""
    M = 2147483647

    def __init__(self, levels, C, nb, jsets, joint_offs, seed, depth, region):
        self.levels, self.C, self.nb = np.asarray(levels, float), float(C), np.asarray(nb, float)
        self.jsets, self.joint_offs = list(jsets), {k: (np.asarray(n, float), np.asarray(o, float)) for k, (n, o) in joint_offs.items()}
        self.seed, self.depth, self.region = int(seed), float(depth), int(region)

    def cell_of(self, p):
        M = self.M
        k = int(np.searchsorted(self.levels, p @ self.nb + self.C)) % M
        for j, sid in enumerate(self.jsets):
            n, offs = self.joint_offs[sid]
            k = (k * 48271 + int(np.searchsorted(offs, p @ n)) + 7919 * (j + 1)) % M
        return (k * 48271 + 104729 * (self.region + 1)) % M

    def depth_of(self, cell):
        M = self.M
        h = ((cell * 16807 + self.seed) % M * 48271 % M) / M
        return self.depth * h ** 1.5


_HM = 2147483647


def _blocky(lith):
    """Tendency of a lithology to break into blocks (1 resistant, 0.2 weak), from its differential-erosion recess."""
    from .lithology import get
    return float(np.clip(1. - get(lith)['recess'], .2, 1.))


class SpallCells:
    """Cells and spalling depth of a B region, picklable.

    Cell = (bedding level slab i, joint slab per v2 joint set, cross slab of level i). Cross fractures are bed-confined:
    each level slab has its own family of parallel planes perpendicular to bedding, its own strike jitter and its own
    irregular spacing, so cross breaks do not line up from bed to bed. Depth: a cell is spalled when a low-frequency
    value-noise score at its centre (in the bedding / strike frame, anisotropic scales), lowered for weak lithologies,
    exceeds the region's quantile for the target spalled fraction; spalled depth grows from amp_min x depth at the
    cluster margin to depth in the cluster core, with a per-cell jitter. The depth of every cell key is precomputed
    once (one implementation; the fast path and the clipping path read the same table)."""

    def __init__(self, levels, C, nb, jsets, joint_offs, cross, region, table):
        self.levels, self.C, self.nb = np.asarray(levels, float), float(C), np.asarray(nb, float)
        self.jsets = list(jsets)
        self.joint_offs = {k: (np.asarray(n, float), np.asarray(o, float)) for k, (n, o) in joint_offs.items()}
        self.cross = [(np.asarray(n, float), np.asarray(o, float)) for n, o in cross]
        self.region, self.table = int(region), dict(table)

    def key_of(self, i, js, c):
        k = int(i) % _HM
        for j, ji in enumerate(js):
            k = (k * 48271 + int(ji) + 7919 * (j + 1)) % _HM
        k = (k * 48271 + int(c) + 15485863) % _HM
        return (k * 48271 + 104729 * (self.region + 1)) % _HM

    def cell_of(self, p):
        i = int(np.searchsorted(self.levels, p @ self.nb + self.C))
        js = [int(np.searchsorted(self.joint_offs[sid][1], p @ self.joint_offs[sid][0])) for sid in self.jsets]
        n, offs = self.cross[i]
        return self.key_of(i, js, int(np.searchsorted(offs, p @ n)))

    def cells_of(self, P):
        """Vectorised cell_of (same integer arithmetic)."""
        i = np.searchsorted(self.levels, P @ self.nb + self.C).astype(np.int64)
        k = i % _HM
        for j, sid in enumerate(self.jsets):
            n, offs = self.joint_offs[sid]
            k = (k * 48271 + np.searchsorted(offs, P @ n) + 7919 * (j + 1)) % _HM
        c = np.zeros(len(P), np.int64)
        for li in np.unique(i):
            sel = i == li
            n, offs = self.cross[int(li)]
            c[sel] = np.searchsorted(offs, P[sel] @ n)
        k = (k * 48271 + c + 15485863) % _HM
        return (k * 48271 + 104729 * (self.region + 1)) % _HM

    def depth_of(self, cell):
        return self.table[cell]

    def _sat(self):
        """Per level slab: summed-area table of 'spalled' over (joint slab per set..., cross slab)."""
        if getattr(self, '_sat_cache', None) is None:
            import itertools
            sats = []
            jn = [len(self.joint_offs[sid][1]) + 1 for sid in self.jsets]
            for i in range(len(self.levels) + 1):
                nc = len(self.cross[i][1]) + 1
                A = np.zeros(jn + [nc], np.int64)
                for js in itertools.product(*[range(n) for n in jn]):
                    for c in range(nc):
                        A[js + (c,)] = self.table[self.key_of(i, js, c)] > 0
                S = A
                for ax in range(S.ndim):
                    S = np.cumsum(S, axis=ax)
                sats.append(np.pad(S, [(1, 0)] * S.ndim))
            self._sat_cache = sats
        return self._sat_cache

    def uniform_depth(self, C6):
        """Per prism slab (corners C6): (uniform, R) -- True where every cell the slab may meet has one and the same
        depth R (superset test over index ranges per level; several levels: uniform only if all intact)."""
        if getattr(self, '_vals', None) is None:
            self._vals = np.unique(np.asarray(list(self.table.values()), float))
            self._sats_v = {}
        n = len(C6)
        uni = np.zeros(n, bool)
        Rv = np.zeros(n)
        t = C6 @ self.nb + self.C
        lo_i = np.searchsorted(self.levels, t.min(1))
        hi_i = np.searchsorted(self.levels, t.max(1))
        intact = ~self.any_spalled(C6)
        uni |= intact
        one = (lo_i == hi_i) & ~intact
        jr = []
        for sid in self.jsets:
            nj, offs = self.joint_offs[sid]
            pj = C6 @ nj
            jr.append((np.searchsorted(offs, pj.min(1)), np.searchsorted(offs, pj.max(1))))
        import itertools
        jn = [len(self.joint_offs[sid][1]) + 1 for sid in self.jsets]
        for li in np.unique(lo_i[one]):
            sel = np.flatnonzero(one & (lo_i == li))
            cn, coffs = self.cross[int(li)]
            nc = len(coffs) + 1
            if int(li) not in self._sats_v:
                A = np.zeros(jn + [nc])
                for js in itertools.product(*[range(k) for k in jn]):
                    for c in range(nc):
                        A[js + (c,)] = self.table[self.key_of(int(li), js, c)]
                cnt = {}
                for v in np.unique(A):
                    S = (A == v).astype(np.int64)
                    for ax in range(S.ndim):
                        S = np.cumsum(S, axis=ax)
                    cnt[float(v)] = np.pad(S, [(1, 0)] * S.ndim)
                self._sats_v[int(li)] = cnt
            pc = C6[sel] @ cn
            ranges = [(a[sel], b[sel]) for a, b in jr] + [(np.searchsorted(coffs, pc.min(1)), np.searchsorted(coffs, pc.max(1)))]
            d = len(ranges)
            vol = np.ones(len(sel), np.int64)
            for a, b in ranges:
                vol *= (b - a + 1)
            for v, S in self._sats_v[int(li)].items():
                tot = np.zeros(len(sel), np.int64)
                for corner in range(1 << d):
                    idx, sign = [], 1
                    for ax in range(d):
                        if corner >> ax & 1:
                            idx.append(ranges[ax][1] + 1)
                        else:
                            idx.append(ranges[ax][0])
                            sign = -sign
                    tot += sign * S[tuple(idx)]
                full = tot == vol
                uni[sel[full]] = True
                Rv[sel[full]] = v
        return uni, Rv

    def any_spalled(self, C6):
        """Per prism (corner points C6 (n, k, 3) of its slab): True if a spalled cell may meet the slab (a superset:
        the product of the index ranges of its corners per level, joint set and cross family)."""
        sats = self._sat()
        n = len(C6)
        out = np.zeros(n, bool)
        t = C6 @ self.nb + self.C
        lo_i = np.searchsorted(self.levels, t.min(1))
        hi_i = np.searchsorted(self.levels, t.max(1))
        jr = []
        for sid in self.jsets:
            nj, offs = self.joint_offs[sid]
            pj = C6 @ nj
            jr.append((np.searchsorted(offs, pj.min(1)), np.searchsorted(offs, pj.max(1))))
        for li in range(len(self.levels) + 1):
            sel = np.flatnonzero((lo_i <= li) & (hi_i >= li) & ~out)
            if not len(sel):
                continue
            cn, coffs = self.cross[li]
            pc = C6[sel] @ cn
            ranges = [(a[sel], b[sel]) for a, b in jr] + [(np.searchsorted(coffs, pc.min(1)), np.searchsorted(coffs, pc.max(1)))]
            S = sats[li]
            d = len(ranges)
            tot = np.zeros(len(sel), np.int64)
            for corner in range(1 << d):
                idx, sign = [], 1
                for ax in range(d):
                    if corner >> ax & 1:
                        idx.append(ranges[ax][1] + 1)
                    else:
                        idx.append(ranges[ax][0])
                        sign = -sign
                tot += sign * S[tuple(idx)]
            out[sel] |= tot > 0
        return out

    def relevant_plane(self, fam, C6):
        """Per prism: can the plane family matter (cross fractures only within their own level slab)?"""
        if not fam.startswith('cross:'):
            return np.ones(len(C6), bool)
        i = int(fam.split(':', 1)[1])
        t = C6 @ self.nb + self.C
        lo_i = np.searchsorted(self.levels, t.min(1))
        hi_i = np.searchsorted(self.levels, t.max(1))
        return (lo_i <= i) & (hi_i >= i)


def spall_cells(levels, C, nb, jsets, joint_offs, seed, depth, region, m, corners, contacts, units_sorted, lith_of_unit, opts):
    """SpallCells of a region (see class doc) and its cross-fracture cut planes ('cross:<level>', index, normal, offset).
    opts: fraction (target spalled share of the cells' slab area, default 0.3), scale_t_m / scale_s_m (noise scales
    across beds / along strike, 0.45 / 1.6 m), cross_spacing_m ([0.35, 1.4]), cross_jitter_deg (25), amp_min (0.35),
    weak_penalty (0.6: score lowered by weak_penalty x (1 - blockiness))."""
    from .morph import noise
    g = np.random.default_rng(seed + 523 + 7919 * (region + 1))
    s_dir = np.cross(nb, m)
    s_dir /= np.linalg.norm(s_dir)
    d_dir = np.cross(nb, s_dir)
    lv = np.asarray(levels, float)
    tv = corners @ nb + C
    tmin, tmax = float(tv.min()), float(tv.max())
    edges_t = np.r_[min(tmin, lv[0]) - .01, lv, max(tmax, lv[-1]) + .01] if len(lv) else np.array([tmin - .01, tmax + .01])
    sp_lo, sp_hi = (float(x) for x in opts.get('cross_spacing_m', [.35, 1.4]))
    jit = np.deg2rad(float(opts.get('cross_jitter_deg', 25.)))
    cross, cross_cut = [], []
    for i in range(len(lv) + 1):
        phi = g.uniform(-jit, jit)
        n = s_dir * np.cos(phi) + d_dir * np.sin(phi)
        pr = corners @ n
        o, offs = float(pr.min()) - sp_hi * g.uniform(0, 1), []
        while o < float(pr.max()) + .01:
            offs.append(o)
            o += g.uniform(sp_lo, sp_hi)
        offs = np.round(np.asarray(offs), 9)
        cross.append((n, offs))
        for k, o_ in enumerate(offs):
            cross_cut.append(('cross:%d' % i, k, n, float(o_)))
    # unit (and lithology) of each level slab from the bisected contacts
    ct = sorted(c_ for c_, _, _ in contacts)
    units = [u for _, u, _, _ in units_sorted]
    blocky_of_level = []
    for i in range(len(lv) + 1):
        tm = .5 * (edges_t[i] + edges_t[i + 1])
        ui = int(np.searchsorted(ct, tm))
        u_ = units[min(ui, len(units) - 1)] if units else None
        blocky_of_level.append(_blocky(lith_of_unit.get(u_, '')) if u_ is not None else .6)
    # every cell: key, centre (t, s), slab area weight
    proto = SpallCells(levels, C, nb, jsets, joint_offs, cross, region, {})
    jranges = [range(len(joint_offs[sid][1]) + 1) for sid in jsets]
    import itertools
    keys, tc, sc, wt, bl, jh, lvl_idx = [], [], [], [], [], [], []
    for i in range(len(lv) + 1):
        n, offs = cross[i]
        se = np.r_[offs[0] - sp_hi, offs, offs[-1] + sp_hi] if len(offs) else np.array([-1., 1.])
        th = edges_t[i + 1] - edges_t[i]
        for js in itertools.product(*jranges):
            for c in range(len(offs) + 1):
                keys.append(proto.key_of(i, js, c))
                tc.append(.5 * (edges_t[i] + edges_t[i + 1]))
                sc.append(.5 * (se[c] + se[c + 1]))
                wt.append(th * (se[c + 1] - se[c]))
                bl.append(blocky_of_level[i])
                lvl_idx.append(i)
    keys, tc, sc, wt, bl = (np.asarray(a) for a in (keys, tc, sc, wt, bl))
    nseed = int(seed % 100000) + 61 * (region + 1)
    score = noise(tc / float(opts.get('scale_t_m', .45)), sc / float(opts.get('scale_s_m', 1.6)), nseed)
    score = score - float(opts.get('weak_penalty', .6)) * (1. - bl)
    frac = float(opts.get('fraction', .3))
    o_ = np.argsort(score)
    cw = np.cumsum(wt[o_]) / wt.sum()
    theta = float(score[o_][min(np.searchsorted(cw, 1. - frac), len(o_) - 1)])
    h = ((keys * 16807 + seed) % _HM * 48271 % _HM) / _HM
    ramp = np.clip((score - theta) / .35, 0, 1)
    amp_min = float(opts.get('amp_min', .35))
    tiers = opts.get('tiers')
    if tiers:
        # slab, so neighbouring cells of a bed at the same tier share their floor (no step at the cross fracture between
        # them) and steps remain at bed partings and tier changes
        tiers = np.asarray(tiers, float)
        lvl = np.asarray(lvl_idx)
        ljit = (.8 + .4 * g.random(len(lv) + 1))[lvl]
        tier = np.minimum((ramp * len(tiers)).astype(int), len(tiers) - 1)
        R = np.where(score > theta, depth * np.minimum(1., tiers[tier] * ljit), 0.)
    else:
        R = np.where(score > theta, depth * np.minimum(1., (amp_min + (1 - amp_min) * ramp * ramp * (3 - 2 * ramp)) * (.8 + .4 * h)), 0.)
    table = {int(k): float(r) for k, r in zip(keys, R)}
    cells = SpallCells(levels, C, nb, jsets, joint_offs, cross, region, table)
    cells.seed, cells.depth = int(seed), float(depth)
    info = {'cells': int(len(keys)), 'spalled_area_fraction_of_slabs': float(wt[R > 0].sum() / wt.sum()), 'theta': theta,
            'cross_planes': len(cross_cut), 'opts': {k: opts[k] for k in opts}, 'blockiness_per_level': [float(x) for x in blocky_of_level],
            'key_collisions': int(len(keys) - len(table))}
    return cells, cross_cut, info


def _plane_families(cut_planes):
    """Cut planes grouped by family (one normal each): [(family, normal, offsets sorted, plane indices in that order)]."""
    fam = {}
    for j, (f, lev, n, d) in enumerate(cut_planes):
        fam.setdefault(f, (np.asarray(n, float), []))[1].append((float(d), j))
    out = []
    for f, (n, lst) in fam.items():
        lst.sort()
        out.append((f, n, np.array([d for d, _ in lst]), np.array([j for _, j in lst], np.int64)))
    return out


def _level_range(cells, C6):
    t = C6 @ cells.nb + cells.C
    return np.searchsorted(cells.levels, t.min(1)), np.searchsorted(cells.levels, t.max(1))


def build_region_fast(V, F, inside, taper, m, cut_planes, cells, depth, margin=1e-7, intact=False):
    """Fast path of the B construction. A prism whose slab from its top down to the deepest possible floor (depth x max taper
    of its corners, + 0.1 mm) has every cut plane strictly on one side (beyond ``margin``) meets one cell only: its
    surface is the top triangle translated along m by the cell depth x taper per corner (the floor build_patch would
    emit), no riser. Returns (fast mask over inside, cell key per fast prism, R per fast prism, slab depth per prism).
    A plane through a fast slab would cross the shared side of its neighbour too, so a fast prism's edges carry no
    split point of a neighbouring clipped prism (no T-junction)."""
    T = V[F[inside]]
    tmax = taper[F[inside]].max(1)
    h = depth * tmax + 1e-4
    hit = np.zeros(len(inside), bool)
    if hasattr(cells, 'cross'):
        # below, per plane), found by binary search; a cross fracture counts only for prisms meeting its level slab
        C6 = np.concatenate([T, T + m * h[:, None, None]], axis=1)
        li0, li1 = _level_range(cells, C6)
        for fam, n, ds, _ in _plane_families(cut_planes):
            pr = C6 @ n
            lo, hi = pr.min(1), pr.max(1)
            h_ = np.searchsorted(ds, lo - margin, 'left') < np.searchsorted(ds, hi + margin, 'right')
            if fam.startswith('cross:'):
                L = int(fam.split(':', 1)[1])
                h_ &= (li0 <= L) & (li1 >= L)
            hit |= h_
    else:
        for fam, lev, n, d in cut_planes:
            n = np.asarray(n, float)
            vt = T @ n                                   # (n, 3)
            mn = float(m @ n)
            vb = vt + (h * mn)[:, None]
            lo = np.minimum(vt.min(1), vb.min(1))
            hi = np.maximum(vt.max(1), vb.max(1))
            hit |= (lo <= d + margin) & (hi >= d - margin)
    if intact and hasattr(cells, 'any_spalled'):
        # A clipped neighbour may split the shared edge at a plane between two intact cells: a collinear T-junction (no gap)
        C6 = np.concatenate([T, T + m * h[:, None, None]], axis=1)
        if intact == 'uniform':
            # also slabs whose cells all share one depth (one tier of one level): the floor is one plane
            uni, _ = cells.uniform_depth(C6)
            hit = hit & ~uni
        else:
            hit = hit & cells.any_spalled(C6)
    fast = ~hit
    # the cell is taken at the emitted face's centroid (the top centroid; a single-cell slab contains it, and an
    # intact-rule face is the top itself), as the G-b check (centroid in its own cell) reads it
    cen = T[fast].mean(1)
    keys = cells.cells_of(cen)
    R = np.array([cells.depth_of(int(k)) for k in keys], float)
    if hasattr(cells, 'cross'):
        # the recorded cell is the one containing the emitted face's centroid (top + m R taper); a uniform slab gives every
        # one of its cells the same depth R, so R is unchanged
        tf = taper[F[inside[fast]]]
        cen_out = (T[fast] + (m[None, None, :] * R[:, None, None]) * tf[:, :, None]).mean(1)
        keys = cells.cells_of(cen_out)

    return fast, keys, R, h


def crossing_planes_grouped(V, F, inside, m, cut_planes, bottom_depth, cells):
    """crossing_planes for SpallCells regions: per family one projection and a binary search (the same strict test as
    crossing_planes, with its 1e-12 slack), cross fractures only for prisms meeting their level slab."""
    tri = V[F[inside]]
    C6 = np.concatenate([tri, tri + m * bottom_depth], axis=1)
    li0, li1 = _level_range(cells, C6)
    per = [[] for _ in range(len(inside))]
    for fam, n, ds, js in _plane_families(cut_planes):
        pr = C6 @ n
        a = np.searchsorted(ds, pr.min(1) + EPS - 1e-12, 'right')     # first d with vals.min < d - EPS + 1e-12
        b = np.searchsorted(ds, pr.max(1) - EPS + 1e-12, 'left')      # d with vals.max > d + EPS - 1e-12
        ok = a < b
        if fam.startswith('cross:'):
            L = int(fam.split(':', 1)[1])
            ok &= (li0 <= L) & (li1 >= L)
        for i in np.flatnonzero(ok):
            per[i].extend(js[a[i]:b[i]].tolist())
    return [sorted(x) for x in per]


def crossing_planes(V, F, inside, m, cut_planes, bottom_depth=1.):
    """Per prism of ``inside`` the indices of the cut planes it crosses, by the same test as build_patch
    (the 6 prism corners, bottom ``bottom_depth`` along m; strictly on both sides beyond EPS), vectorised, as a superset."""
    tri = V[F[inside]]
    corners = np.concatenate([tri, tri + m * bottom_depth], axis=1)       # (n, 6, 3)
    out = [[] for _ in range(len(inside))]
    for j, (fam, lev, n, d) in enumerate(cut_planes):
        vals = corners @ np.asarray(n, float)
        # a superset (1e-12 m slack against the rounding of a vectorised dot product); build_patch re-tests exactly
        cross = ~((vals.min(1) >= d - EPS + 1e-12) | (vals.max(1) <= d + EPS - 1e-12))
        for i in np.flatnonzero(cross):
            out[i].append(j)
    return out


def _build_chunk(args):
    V, F, local, taper, m, cut_planes, cells, exact, planes = args[:9]
    fct = args[9] if len(args) > 9 else 1e-7
    bd = args[10] if len(args) > 10 else 1.
    return build_patch(V, F, local, taper, m, cut_planes, cells.cell_of, cells.depth_of, exact=exact, planes_of=planes, floor_cap_tol=fct,
                       bottom_depth=bd, project_caps=hasattr(cells, 'cross'))


def build_patch_parallel(V, F, inside, taper, m, cut_planes, cells, exact=False, workers=1, chunk=20000, floor_cap_tol=1e-7, bottom_depth=1.):
    """build_patch over chunks of prisms in worker processes; the result is the concatenation in prism order (that
    of build_patch over ``inside``); each prism tests only the planes crossing_planes found for it. ``cells``: a
    RegionCells. Each chunk carries only its prisms' vertices; info['prism'] is mapped back to the global face."""
    from concurrent.futures import ProcessPoolExecutor
    inside = np.asarray(inside)
    if hasattr(cells, 'cross'):
        planes = crossing_planes_grouped(V, F, inside, m, cut_planes, bottom_depth, cells)
    else:
        planes = crossing_planes(V, F, inside, m, cut_planes, bottom_depth)
    if hasattr(cells, 'table') and workers > 1:
        # values are those of build_patch over ``inside`` regardless of the chunking
        chunk = max(500, -(-len(inside) // (3 * workers)))
    if hasattr(cells, 'relevant_plane') and len(inside) and not hasattr(cells, 'cross'):
        tri = V[F[inside]]
        C6 = np.concatenate([tri, tri + m * bottom_depth], axis=1)
        rel = {}
        for j, (fam, lev, n_, d_) in enumerate(cut_planes):
            if fam.startswith('cross:'):
                rel[j] = cells.relevant_plane(fam, C6)
        planes = [[j for j in pl if j not in rel or rel[j][i]] for i, pl in enumerate(planes)]
    jobs, idxs = [], []
    for s0 in range(0, len(inside), chunk):
        idx = inside[s0:s0 + chunk]
        used, loc = np.unique(F[idx], return_inverse=True)
        jobs.append((V[used], loc.reshape(-1, 3), np.arange(len(idx)), taper[used], m, cut_planes, cells, exact, planes[s0:s0 + chunk], floor_cap_tol,
                     bottom_depth))
        idxs.append(idx)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            res = list(ex.map(_build_chunk, jobs))
    else:
        res = [_build_chunk(j) for j in jobs]
    polys, stats = [], {'prisms': 0, 'pieces': 0}
    for idx, (pp, st) in zip(idxs, res):
        for poly, kind, info in pp:
            info['prism'] = int(idx[info['prism']])
            polys.append((poly, kind, info))
        stats['prisms'] += st['prisms']
        stats['pieces'] += st['pieces']
    return polys, stats


def _ear_clip(P):
    """Triangles (index triples) of a planar simple polygon, all wound like the polygon. Welding can
    leave a reflex corner in a former convex polygon; a fan from vertex 0 then folds a triangle."""
    n = _area_normal(P)
    n /= max(np.linalg.norm(n), 1e-30)
    o = np.mean(P, axis=0)
    u = P[1] - P[0]
    u = u - n * (u @ n)
    u /= max(np.linalg.norm(u), 1e-30)
    v = np.cross(n, u)
    q = np.array([((p - o) @ u, (p - o) @ v) for p in P])
    cross = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    idx = list(range(len(P)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10 * len(P):
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(q[i0], q[i1], q[i2]) <= 1e-18:
                continue                      # reflex or flat corner: not an ear
            if any(cross(q[i0], q[i1], q[j]) >= 0 and cross(q[i1], q[i2], q[j]) >= 0 and cross(q[i2], q[i0], q[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue                      # another vertex inside: not an ear
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def triangulate(polys, V, F, m, exact_floor_vertices=False):
    """Fan triangles of the emitted polygons with their own vertices; per new vertex the prism top triangle and barycentric
    weights (projection along m) for interpolating per-vertex fields. exact_floor_vertices (opt-in, whole-wall scene):
    floor vertices of one cell are shared only at bit-identical coordinates; the historical key rounds to 1e-7 m,
    which merges distinct corners of a sub-0.1 um thin floor and collapses it to zero area."""
    verts, faces, prism, bary, kind, fpoly = [], [], [], [], [], []
    shared = {}  # floor vertices of one cell at one position are shared: smooth shading across the
    # floor; floors and risers never share a vertex, so the riser edges stay creased
    for pi, (poly, k, info) in enumerate(polys):
        idx = []
        tri = V[F[info['prism']]]
        e0, e1_ = tri[1] - tri[0], tri[2] - tri[0]
        nrm = _cross(e0, e1_)
        for p in poly:
            key = (info['cell'], tuple(np.asarray(p, float).tolist()) if exact_floor_vertices else tuple(np.round(p, 7))) if k == 'floor' else None
            if key is not None and key in shared:
                idx.append(shared[key])
                continue
            # project p along m onto the top triangle's plane, then barycentric coordinates
            t = ((tri[0] - p) @ nrm) / (m @ nrm)
            q = p + t * m - tri[0]
            d00, d01, d11 = e0 @ e0, e0 @ e1_, e1_ @ e1_
            d20, d21 = q @ e0, q @ e1_
            den = d00 * d11 - d01 * d01
            b1 = (d11 * d20 - d01 * d21) / den
            b2 = (d00 * d21 - d01 * d20) / den
            if key is not None:
                shared[key] = len(verts)
            idx.append(len(verts))
            verts.append(p)
            prism.append(info['prism'])
            bary.append((1 - b1 - b2, b1, b2))
        P_ = [np.asarray(p) for p in poly]
        for a, b, c in _ear_clip(P_):
            if np.linalg.norm(_cross(P_[b] - P_[a], P_[c] - P_[a])) < 2e-12:
                continue                      # zero-area triangle of collinear corners: no surface
            faces.append((idx[a], idx[b], idx[c]))
            kind.append(k)
            fpoly.append(pi)
    return (np.asarray(verts, float), np.asarray(faces, np.int64), np.asarray(prism, np.int64),
            np.clip(np.asarray(bary, float), 0, 1), kind, np.asarray(fpoly, np.int64))


def plane_basis(n):
    """In-plane texture axes of a riser with outward normal n: u horizontal (along strike), v up the
    plane, (u, v, n) right-handed, so the image is not mirrored seen from outside; any horizontal
    axis for a horizontal plane."""
    n = np.asarray(n, float)
    u = np.cross([0., 0., 1.], n)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [0., 1., 0.])  # horizontal plane
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def riser_plane_uv(V, F, riser_f, nout_f):
    """Per-vertex texture coordinates of the riser vertices in metres along their plane's axes
    (plane_basis of the polygon's outward normal, world origin: coplanar risers stay continuous);
    NaN for every other vertex. Risers never share a vertex with a floor or another riser
    (triangulate); a vertex used by a riser and a non-riser face would get two mappings: fail."""
    use_r = np.zeros(len(V), bool)
    use_o = np.zeros(len(V), bool)
    use_r[F[riser_f].reshape(-1)] = True
    use_o[F[~riser_f].reshape(-1)] = True
    if (use_r & use_o).any():
        raise RuntimeError('%d riser vertices are shared with non-riser faces' % int((use_r & use_o).sum()))
    uv = np.full((len(V), 2), np.nan)
    for fi in np.flatnonzero(riser_f):
        u, v = plane_basis(nout_f[fi])
        P = V[F[fi]]
        new = np.c_[P @ u, P @ v]
        old = uv[F[fi]]
        if (np.isfinite(old) & (np.abs(old - new) > 1e-9)).any():
            raise RuntimeError('riser vertex with two plane mappings (face %d)' % fi)
        uv[F[fi]] = new
    return uv


def _weld_roots(pts, tp, tol, chunk=4000000):
    """Cluster roots of weld_slivers, vectorised: every pair (i < j) in neighbouring cells of the tol grid closer than
    min(tp_i, tp_j) is linked (the same pairs as the per-point loop) and each point takes the smallest index of its
    connected component (the root the loop's union rule leaves: the larger root is always attached to the smaller)."""
    n = len(pts)
    keys = np.floor(pts / tol).astype(np.int64)
    Bk = 1 << 20
    code = lambda K: ((K[:, 0] + Bk) * (2 * Bk) + (K[:, 1] + Bk)) * (2 * Bk) + (K[:, 2] + Bk)
    kc = code(keys)
    order = np.argsort(kc, kind='stable')
    ks = kc[order]
    uk, st = np.unique(ks, return_index=True)
    en = np.r_[st[1:], len(ks)]
    I, J = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kk = code(keys + np.array([dx, dy, dz]))
                pos = np.minimum(np.searchsorted(uk, kk), len(uk) - 1)
                hit = np.flatnonzero(uk[pos] == kk)
                for c0 in range(0, len(hit), chunk):
                    h = hit[c0:c0 + chunk]
                    cnt = en[pos[h]] - st[pos[h]]
                    qi = np.repeat(h, cnt)
                    cj = order[np.repeat(st[pos[h]], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
                    keep = cj > qi
                    qi, cj = qi[keep], cj[keep]
                    d = np.linalg.norm(pts[qi] - pts[cj], axis=1)
                    ok = d < np.minimum(tp[qi], tp[cj])
                    I.append(qi[ok])
                    J.append(cj[ok])
    I = np.concatenate(I) if I else np.zeros(0, np.int64)
    J = np.concatenate(J) if J else np.zeros(0, np.int64)
    lab = np.arange(n)
    while len(I):
        m_ = np.minimum(lab[I], lab[J])
        new = lab.copy()
        np.minimum.at(new, I, m_)
        np.minimum.at(new, J, m_)
        new = new[new]                      # pointer jumping
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _deletable_slivers(polys, candidates, tol, step=5e-4):
    """Delete connected thin candidates only if the hole boundary matches within tol.

    The inradius used to select candidates is not a bound on slit width, and
    adjacent individually thin polygons can leave a wider hole together.
    """
    parent = {i: i for i in candidates}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    at_vertex = {}
    keys = {}
    for i in candidates:
        # Shared prism corners differ by floating-point roundoff. This only
        # groups deletion candidates; all distance checks use original points.
        keys[i] = [tuple(np.round(p, 11)) for p in polys[i][0]]
        for k in keys[i]:
            if k in at_vertex:
                a, b = root(i), root(at_vertex[k])
                parent[max(a, b)] = min(a, b)
            else:
                at_vertex[k] = i
    # A T-junction or partially overlapping shared edge need not have a
    # common corner. Join owners before checking their combined hole boundary.
    points = np.concatenate([np.asarray(polys[i][0]) for i in candidates])
    owners = np.repeat(candidates, [len(polys[i][0]) for i in candidates])
    gsz = .004
    cells = {}
    for k, key in enumerate(map(tuple, np.floor(points / gsz).astype(np.int64))):
        cells.setdefault(key, []).append(k)
    offsets = np.asarray(list(np.ndindex(3, 3, 3))) - 1
    for i in candidates:
        P = np.asarray(polys[i][0])
        for j in range(len(P)):
            a, b = P[j], P[(j + 1) % len(P)]
            e = b - a
            L2 = float(e @ e)
            count = max(int(np.ceil(np.sqrt(L2) / gsz)), 1)
            along = a + np.linspace(0., 1., count + 1)[:, None] * e
            grid = np.floor(along / gsz).astype(np.int64)
            neighbours = np.unique((grid[:, None] + offsets).reshape(-1, 3), axis=0)
            near = np.unique([k for key in map(tuple, neighbours) for k in cells.get(key, [])])
            near = near.astype(np.int64)
            t = np.clip(((points[near] - a) @ e) / max(L2, 1e-300), 0., 1.)
            touches = near[np.linalg.norm(points[near] - (a + t[:, None] * e), axis=1) <= 1e-12]
            for k in np.unique(owners[touches]):
                ra, rb = root(i), root(int(k))
                parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in candidates:
        groups.setdefault(root(i), []).append(i)
    drop = set()
    for group in groups.values():
        vertices = {}
        for i in group:
            vertices.update(zip(keys[i], polys[i][0]))
        vkeys, VP = list(vertices), np.asarray(list(vertices.values()))
        edges = {}
        for i in group:
            P = polys[i][0]
            for j in range(len(P)):
                a, b = P[j], P[(j + 1) % len(P)]
                e = b - a
                L2 = float(e @ e)
                # Split a long edge wherever a candidate corner touches it.
                # Then an opposing pair of short edges cancels the long edge
                # at a T-junction rather than falsely narrowing the hole.
                t = ((VP - a) @ e) / max(L2, 1e-300)
                foot = a + t[:, None] * e
                on = np.flatnonzero((t > 0.) & (t < 1.) &
                                    (np.linalg.norm(VP - foot, axis=1) <= 1e-12))
                breaks = [(0., keys[i][j]), (1., keys[i][(j + 1) % len(P)])]
                breaks += [(float(t[k]), vkeys[k]) for k in on]
                breaks.sort()
                for (u, ka), (v, kb) in zip(breaks, breaks[1:]):
                    if ka == kb:
                        continue
                    key = tuple(sorted((ka, kb)))
                    edges.setdefault(key, []).append((a + u * e, a + v * e))
        if any(len(e) > 2 for e in edges.values()):
            continue
        boundary = [e[0] for e in edges.values() if len(e) == 1]
        if not boundary:
            drop.update(group)
            continue
        E = np.asarray(boundary)
        A, ab = E[:, 0], E[:, 1] - E[:, 0]
        ab2 = (ab * ab).sum(1)
        ok = True
        for j in range(len(E)):
            count = max(int(np.ceil(np.sqrt(ab2[j]) / step)), 1)
            for start in range(0, count, 4096):
                t = (np.arange(start, min(start + 4096, count)) + .5) / count
                S = A[j] + t[:, None] * ab[j]
                u = np.clip(((S[:, None] - A) * ab).sum(2) / np.maximum(ab2, 1e-300), 0., 1.)
                dist = np.linalg.norm(S[:, None] - (A + u[:, :, None] * ab), axis=2)
                dist[:, j] = np.inf
                if (dist.min(1) > tol).any():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            drop.update(group)
    return drop


def merge_nano(polys, cut, tol=1e-7, plane_tol=1e-9):
    """Plane-preserving merge of polygon corners closer than ``tol``. B depth tapers to zero at a region boundary, so a riser
    there is a wedge whose tip edge can be nanometres long; the weld keeps such corners apart (points on cut planes
    have a ~1e-13 tolerance so no riser leaves its plane), and every triangulation of that polygon contains a sliver
    too thin for the outer-boundary check. Here a corner is merged into a representative only if the representative
    lies on EVERY cut plane the corner lies on (within plane_tol), so no face leaves a plane. Consecutive duplicates
    are removed; polygons left with < 3 corners or zero area are dropped (they had no width). Returns polygons and
    stats (moves, merged corners, dropped polygons, plane-membership check)."""
    pts = np.array([p for poly, _, _ in polys for p in poly])
    owner = np.repeat(np.arange(len(polys)), [len(poly) for poly, _, _ in polys])
    N = np.array([n for _, _, n, _ in cut]).reshape(-1, 3)
    D = np.array([d for _, _, _, d in cut])
    on = np.abs(pts @ N.T - D[None]) < plane_tol if len(cut) else np.zeros((len(pts), 0), bool)
    # unique positions, then candidate pairs closer than tol from a hashed grid of cell tol (27 neighbour cells)
    U, inv = np.unique(pts, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    onU = np.zeros((len(U), on.shape[1]), bool)
    np.logical_or.at(onU, inv, on)
    K = np.floor(U / tol).astype(np.int64)
    Bk = 1 << 20
    code = lambda k: ((k[:, 0] + Bk) * (2 * Bk) + (k[:, 1] + Bk)) * (2 * Bk) + (k[:, 2] + Bk)
    c0 = code(K)
    o = np.argsort(c0, kind='stable')
    cs = c0[o]
    pa, pb = [], []
    for off in np.asarray(list(np.ndindex(3, 3, 3))) - 1:
        cq = code(K + off)
        lo = np.searchsorted(cs, cq, 'left')
        hi = np.searchsorted(cs, cq, 'right')
        n = hi - lo
        i = np.repeat(np.arange(len(U)), n)
        j = o[np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))]
        keep = i < j
        pa.append(i[keep])
        pb.append(j[keep])
    pa, pb = np.concatenate(pa), np.concatenate(pb)
    dd = np.linalg.norm(U[pa] - U[pb], axis=1)
    sel = (dd > 0.) & (dd < tol)
    pa, pb, dd = pa[sel], pb[sel], dd[sel]
    repU = np.arange(len(U))
    nplanes = onU.sum(1)
    moved, max_move = 0, 0.
    # representatives: more planes first; a member joins only if its planes are a subset of the representative's
    for k in np.argsort(-np.maximum(nplanes[pa], nplanes[pb]), kind='stable'):
        i_, j_ = int(pa[k]), int(pb[k])
        if nplanes[j_] > nplanes[i_]:
            i_, j_ = j_, i_
        if repU[i_] != i_ or repU[j_] != j_:
            continue
        if (onU[j_] & ~onU[i_]).any():
            continue
        repU[j_] = i_
        moved += 1
        max_move = max(max_move, float(dd[k]))
    for _ in range(8):
        repU = repU[repU]   # chains (subset planes are transitive)
    rep = repU[inv]
    pts = U
    new = pts[rep]
    on = onU[inv]
    out, dropped, k0 = [], 0, 0
    for poly, kind, info in polys:
        q = new[k0:k0 + len(poly)]
        k0 += len(poly)
        keep = [q[0]]
        for x in q[1:]:
            if not np.array_equal(x, keep[-1]):
                keep.append(x)
        if len(keep) > 1 and np.array_equal(keep[0], keep[-1]):
            keep.pop()
        if len(keep) < 3 or np.linalg.norm(_area_normal(keep)) == 0.:
            dropped += 1
            continue
        out.append((list(keep), kind, info))
    still = bool((np.abs(new @ N.T - D[None]) < plane_tol)[on].all()) if len(cut) else True
    return out, {'tol_m': tol, 'corners_merged': int(moved), 'max_move_m': max_move, 'polygons_dropped': dropped,
                 'plane_membership_preserved': still}


def weld_slivers(polys, cut, tol=5e-4, max_move=None, tol_of=None, fine_min_area=None, fast=False, fine_min_width=None):
    """Merge emitted vertices closer than ``tol`` and drop polygons that collapse (sliver risers in the boundary taper,
    slivers from planes passing next to a prism corner). The representative of each cluster is the vertex lying on the
    most cutting planes (v2 contacts, joints, partings), so faces on those planes stay exactly in them. Returns the
    new polygons and stats (moves, dropped). ``max_move``: clusters are single-linkage, so a chain of close points can
    span more than ``tol`` and move a vertex further (1.04 mm on a 1 cm mesh); with max_move, a cluster is split
    greedily -- representative by the same priority, members closer than max_move to it, the rest re-clustered -- so
    no vertex moves max_move or more. None: unchanged. ``tol_of``: callable(points) -> per-point tolerance <= tol; two
    points merge only when closer than both tolerances (a fine mesh keeps its sub-mm detail while the coarse rest of
    the patch keeps tol); with max_move, each point's cap is its own tolerance. ``fine_min_area``: a polygon whose
    vertices all have a tolerance below tol is dropped only when its area is <= fine_min_area (0: only collapsed
    polygons), not under the 1e-10 m^2 used elsewhere -- a dropped polygon of positive area opens a slit of its width.
    ``fast``: the clusters from _weld_roots (vectorised, same roots). ``fine_min_width`` (opt-in, with tol_of;
    whole-wall scene): select fine polygons with inradius proxy 2 area / perimeter below this threshold, then delete a
    connected group only when its remaining hole boundary matches within this distance at 0.5 mm sampling. The proxy
    alone does not bound the resulting hole."""
    pts = np.array([p for poly, _, _ in polys for p in poly])
    owner = np.repeat(np.arange(len(polys)), [len(poly) for poly, _, _ in polys])
    on_planes = np.zeros(len(pts), int)
    for fam, lev, n, d in cut:
        on_planes += np.abs(pts @ n - d) < 1e-9
    tp = np.full(len(pts), tol) if tol_of is None else np.minimum(np.asarray(tol_of(pts), float), tol)
    if fast:
        root = _weld_roots(pts, tp, tol)
        # representative: the most planes, ties the lowest index (as the loop below)
        o_ = np.lexsort((np.arange(len(pts)), -on_planes, root))
        first = np.r_[True, root[o_][1:] != root[o_][:-1]]
        rep_of_root = np.empty(len(pts), np.int64)
        rep_of_root[root[o_][first]] = o_[first]
        new_pts = pts[rep_of_root[root]]
    else:
        parent = np.arange(len(pts))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        cells = {}
        keys = np.floor(pts / tol).astype(np.int64)
        for i, k in enumerate(map(tuple, keys)):
            cells.setdefault(k, []).append(i)
        for i, k in enumerate(map(tuple, keys)):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((k[0] + dx, k[1] + dy, k[2] + dz), []):
                            if j > i and np.linalg.norm(pts[i] - pts[j]) < min(tp[i], tp[j]):
                                ri, rj = find(i), find(j)
                                if ri != rj:
                                    parent[max(ri, rj)] = min(ri, rj)
        root = np.array([find(i) for i in range(len(pts))])
        rep = {}
        for i in range(len(pts)):
            r = root[i]
            if r not in rep or on_planes[i] > on_planes[rep[r]]:
                rep[r] = i
        new_pts = np.array([pts[rep[root[i]]] for i in range(len(pts))])
    moved = np.linalg.norm(new_pts - pts, axis=1)
    n_split = 0
    cap = None if max_move is None else (np.full(len(pts), max_move) if tol_of is None else np.minimum(tp, max_move))
    if max_move is not None and (moved >= cap).any():
        members = {}
        for i in range(len(pts)):
            members.setdefault(root[i], []).append(i)
        for r in sorted({root[i] for i in np.nonzero(moved >= cap)[0]}):
            n_split += 1
            left = list(members[r])        # ascending index: ties keep the first, as above
            while left:
                best = left[0]
                for i in left:
                    if on_planes[i] > on_planes[best]:
                        best = i
                take = [i for i in left if np.linalg.norm(pts[i] - pts[best]) < cap[i]]
                for i in take:
                    new_pts[i] = pts[best]
                left = [i for i in left if i not in set(take)]
        moved = np.linalg.norm(new_pts - pts, axis=1)
    out, dropped, k = [], {'floor': 0, 'riser': 0}, 0
    narrow = []
    fine_pt = tp < tol
    for poly, kind, info in polys:
        min_a = fine_min_area if fine_min_area is not None and fine_pt[k:k + len(poly)].all() else 1e-10
        q = []
        for _ in poly:
            p = new_pts[k]
            k += 1
            if not q or np.linalg.norm(p - q[-1]) > 1e-12:
                q.append(p)
        if len(q) > 1 and np.linalg.norm(q[0] - q[-1]) <= 1e-12:
            q.pop()
        a_ = np.linalg.norm(_area_normal(q)) if len(q) >= 3 else 0.
        if len(q) < 3 or a_ < min_a or a_ == 0.:
            dropped[kind] += 1
            continue
        if fine_min_width is not None and fine_pt[k - len(poly):k].all():
            per = sum(float(np.linalg.norm(q[(i + 1) % len(q)] - q[i])) for i in range(len(q)))
            if 2 * a_ / max(per, 1e-300) < fine_min_width:
                narrow.append(len(out))
        out.append((q, kind, info))
    if narrow:
        remove = _deletable_slivers(out, narrow, fine_min_width)
        for i in remove:
            dropped[out[i][1]] += 1
        if remove:
            dropped['narrow'] = len(remove)
            out = [p for i, p in enumerate(out) if i not in remove]
    st = {'tol_m': tol, 'vertices_moved': int((moved > 0).sum()), 'max_move_m': float(moved.max()) if len(moved) else 0.,
          'polygons_dropped': dropped}
    if max_move is not None:
        st.update(max_move_cap_m=max_move, clusters_split=n_split)
    if tol_of is not None:
        st.update(fine_points=int((tp < tol).sum()), fine_tol_m=float(tp.min()), max_move_fine_m=float(moved[tp < tol].max()) if (tp < tol).any() else 0.,
                  max_move_coarse_m=float(moved[tp >= tol].max()) if (tp >= tol).any() else 0.)
    if fine_min_width is not None:
        st['own_tolerance_violations'] = int((moved >= tp).sum())
    return out, st
