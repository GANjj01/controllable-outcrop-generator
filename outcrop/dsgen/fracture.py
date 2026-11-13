"""Plane-bounded block spalling with exact edges (sampling ``fracture: blocks-3``).

blocks-1/2 (processes.fracture_blocks) sample the spalled surface on the sheet's vertex grid: a riser
on a bedding parting or a v2 joint plane falls between vertices, aliases into a zip-like stair and was
smoothed away (blocks-2). blocks-3 keeps the same cells and spalling depths but reconstructs the
surface along the planes:

1. ``Fracture.evaluate`` gives the exact spalled offset of any start point (no 1 cm quantisation)
   and the *piece* it lies on -- a cell floor (offset R) or a riser in one plane (offset linear in the
   start point) -- with that piece's linear form ``offset = a + c . p0``.
2. ``insert_edges`` splits every mesh edge whose two ends lie on different pieces at the exact
   crossing of the two linear forms (the top / bottom edge of a riser), re-evaluates the new vertex
   and repeats, so the piecewise-linear mesh reproduces the planar riser and its sharp edges.
3. ``crease_normals`` gives the two sides of every edge folded by more than ``CREASE_DEG`` their own
   vertex copies and normals (shared positions; labels are per face and unchanged).
"""
import math

import numpy as np

from .truth import _plane_groups

CREASE_DEG = 30.
MIN_SLOPE = .05  # |plane normal . offset direction| floor: a plane parallel to the offsets is a step


class Fracture:
    """Cells: irregular bedding partings (levels of the v2 bedding coordinate) x v2 joint planes (one of each set within 20
    deg of another skipped); R(cell) = depth * h^1.5, h a hash of the cell."""

    def __init__(self, ref, body, truth, joint_ids, seed, *, depth, spacing):
        ok, nb, tb = truth.fabric(ref, body, 'bedding')
        self.ok, self.nb = ok, nb
        # bedding coordinate as an affine function of position, per reference vertex: t(p) = nb.p + C
        self.C = np.where(ok, tb - (nb * ref).sum(1), np.nan)
        # Fault blocks: the side of every fault plane piece. v2 joints are cut and offset by the faults;
        # a joint piece bounds cells only inside its own block (extending every piece as an infinite
        # joint-like steps with no v2 joint and no label -- found in the blocks-3 sample).
        self.faults = [(nrm, off) for iid, iv, if_ in truth.interfaces if iid.startswith('fault') and len(if_)
                       for nrm, off, _ in _plane_groups(iv, if_)]
        planes = []
        for set_id in sorted(joint_ids):
            normal, by_block = None, {}
            for iid, iv, if_ in truth.interfaces:
                if iid.split(':', 1)[0] != set_id or not len(if_):
                    continue
                for nrm, off, tri in _plane_groups(iv, if_):
                    if normal is None:
                        normal = nrm
                    key = int(self.block_of(tri.reshape(-1, 3).mean(0)[None])[0])
                    by_block.setdefault(key, []).append((1. if nrm @ normal >= 0 else -1.) * off)
            if normal is not None and not any(abs(normal @ n2) > math.cos(math.radians(20)) for n2, _ in planes):
                planes.append((normal, {k: np.unique(np.round(v, 4)) for k, v in by_block.items()}))
        self.planes = planes
        self.depth, self.seed = depth, seed
        phase = (seed % 9973) / 9973. * spacing
        t_all = tb[ok]
        lo, hi = t_all.min() - depth - 1., t_all.max() + depth + 1.
        g = np.random.default_rng(seed + 307)
        th = spacing * g.uniform(.4, 1.6, int((hi - lo) / (.4 * spacing)) + 2)
        self.levels = lo + phase + np.concatenate([[0.], np.cumsum(th)])

    def block_of(self, q):
        """Fault-block key of points: one bit per fault plane piece (side)."""
        key = np.zeros(len(q), np.int64)
        for i, (nrm, off) in enumerate(self.faults):
            key |= ((q @ nrm - off) > 0).astype(np.int64) << i
        return key

    def _joint_index(self, q, normal, by_block):
        blk = self.block_of(q)
        idx = np.zeros(len(q), np.int64)
        d = q @ normal
        for k in np.unique(blk):
            sel = blk == k
            offs = by_block.get(int(k))
            idx[sel] = np.searchsorted(offs, d[sel]) if offs is not None else 0
        return idx + blk * 1000

    def _cells(self, t, q):
        """Cell index per family (bedding first) and the hashed spalling depth."""
        idx = [np.searchsorted(self.levels, t)] + [self._joint_index(q, n, by_block) for n, by_block in self.planes]
        M = 2147483647
        cell = idx[0].astype(np.int64) % M
        for j, k in enumerate(idx[1:]):
            cell = (cell * 48271 + k + 7919 * (j + 1)) % M
        h = ((cell * 16807 + self.seed % M) % M * 48271 % M) / M
        return np.stack(idx, 1), cell, self.depth * h ** 1.5

    def evaluate(self, p0, dirv, nb, C, step=.01):
        """Exact offset of start points p0 (direction dirv; bedding nb.p + C) and the piece each lies
        on: piece id (int64: floor = cell hash, riser = -(1 + family * 2^40 + level index)),
        linear form a + c . p0."""
        n = len(p0)
        out = np.full(n, np.nan)
        pid = np.zeros(n, np.int64)
        a = np.zeros(n)
        c = np.zeros((n, 3))
        kb = (nb * dirv).sum(1)
        normals = [nb] + [np.broadcast_to(nrm, p0.shape) for nrm, _ in self.planes]
        rate = [kb] + [dirv @ nrm for nrm, _ in self.planes]
        t_of = lambda q, sel: (nb[sel] * q).sum(1) + C[sel]
        todo = np.nonzero(np.isfinite(C))[0]
        prev_idx = None
        for k in range(int(math.ceil(self.depth / step)) + 2):
            if not len(todo):
                break
            s = k * step
            q = p0[todo] + s * dirv[todo]
            idx, cell, R = self._cells(t_of(q, todo), q)
            if k == 0:
                done = R <= 0.
                floor = done
                prev_idx = idx
            else:
                done = s >= R
                changed = (idx != prev_idx).any(1)
                floor = done & ~changed
            # floors: offset = R of the cell
            f = np.nonzero(floor)[0]
            g = todo[f]
            out[g], pid[g], a[g], c[g] = R[f], cell[f], R[f], 0.
            # crossings: the family whose index changed in (s - step, s]; solve the plane exactly
            x = np.nonzero(done & ~floor)[0]
            if len(x):
                gx = todo[x]
                best = np.full(len(x), np.inf)
                fam = np.zeros(len(x), np.int64)
                lev = np.zeros(len(x), np.int64)
                for j in range(len(normals)):
                    ch = idx[x, j] != prev_idx[x, j]
                    if not ch.any():
                        continue
                    L_i = np.minimum(idx[x, j], prev_idx[x, j])
                    if j == 0:
                        level = self.levels[np.clip(L_i, 0, len(self.levels) - 1)]
                    else:
                        # joints: index = block * 1000 + position in the block's levels; a change of
                        # block (crossing a fault) is not a joint riser
                        blk = L_i // 1000
                        same_blk = (idx[x, j] // 1000) == (prev_idx[x, j] // 1000)
                        ch = ch & same_blk
                        level = np.zeros(len(x))
                        for kb in np.unique(blk):
                            offs = self.planes[j - 1][1].get(int(kb))
                            sel = blk == kb
                            if offs is not None:
                                level[sel] = offs[np.clip(L_i[sel] % 1000, 0, len(offs) - 1)]
                    nj = normals[j][gx]
                    phi0 = (nj * p0[gx]).sum(1) + (C[gx] if j == 0 else 0.)
                    rj = rate[j][gx]
                    rj = np.where(np.abs(rj) < MIN_SLOPE, np.copysign(MIN_SLOPE, rj + 1e-12), rj)
                    sc = (level - phi0) / rj
                    use = ch & (sc < best)
                    best[use], fam[use], lev[use] = sc[use], j, L_i[use]
                found = np.isfinite(best)  # no family crossed: a fault (block change) -> floor of the new cell
                best = np.clip(np.where(found, best, s), s - step, s)
                riser = (R[x] <= best) & found
                # riser: the plane itself; else the floor of the new cell (R in (crossing, s])
                r = np.nonzero(riser)[0]
                gr = gx[r]
                out[gr] = best[r]
                pid[gr] = -(1 + fam[r] * (1 << 40) + lev[r])
                for rr, gg in zip(r, gr):
                    j = fam[rr]
                    rj = rate[j][gg]
                    rj = rj if abs(rj) >= MIN_SLOPE else math.copysign(MIN_SLOPE, rj + 1e-12)
                    if j == 0:
                        level = self.levels[min(max(lev[rr], 0), len(self.levels) - 1)]
                    else:
                        offs = self.planes[j - 1][1][int(lev[rr] // 1000)]
                        level = offs[min(max(int(lev[rr] % 1000), 0), len(offs) - 1)]
                    a[gg] = (level - (C[gg] if j == 0 else 0.)) / rj
                    c[gg] = -normals[j][gg] / rj
                fl = np.nonzero(~riser)[0]
                gf = gx[fl]
                val = np.maximum(R[x][fl], np.where(found[fl], 0., s - step))
                out[gf], pid[gf], a[gf], c[gf] = val, cell[x][fl], val, 0.
            keep = ~done
            todo, prev_idx = todo[keep], idx[keep]
        out[np.isnan(out)] = 0.
        return out, pid, a, c


def _min_angle(V, tris):
    worst = np.pi
    for t in tris:
        P = V[list(t)]
        for k in range(3):
            u, w = P[(k + 1) % 3] - P[k], P[(k + 2) % 3] - P[k]
            c = u @ w / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-30)
            worst = min(worst, math.acos(max(-1., min(1., c))))
    return worst


def _split(V, F, arrays, edges, lam):
    """Split each edge (i, j) at parameter lam (new vertex = lerp); faces with 1-3 split edges are
    re-triangulated. ``arrays``: per-vertex arrays interpolated the same way."""
    nv = len(V)
    key = edges[:, 0] * (nv + 1) + edges[:, 1]
    order = np.argsort(key)
    key, edges, lam = key[order], edges[order], lam[order]
    newV = V[edges[:, 0]] * (1 - lam)[:, None] + V[edges[:, 1]] * lam[:, None]
    new_arrays = {}
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):  # ids: the nearer end's value (re-evaluated anyway)
            A = np.where((lam < .5).reshape((-1,) + (1,) * (arr.ndim - 1)), arr[edges[:, 0]], arr[edges[:, 1]])
        else:
            w1 = lam.reshape((-1,) + (1,) * (arr.ndim - 1))
            A = arr[edges[:, 0]] * (1 - w1) + arr[edges[:, 1]] * w1
        new_arrays[name] = np.concatenate([arr, A])
    V2 = np.concatenate([V, newV])

    def mid(u, w):
        lo, hi = np.minimum(u, w), np.maximum(u, w)
        k = lo * (nv + 1) + hi
        pos = np.searchsorted(key, k)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[pos] == k
        return np.where(hit, nv + pos, -1)
    m01, m12, m20 = mid(F[:, 0], F[:, 1]), mid(F[:, 1], F[:, 2]), mid(F[:, 2], F[:, 0])
    out, parent = [], []
    cnt = (m01 >= 0).astype(int) + (m12 >= 0) + (m20 >= 0)
    keep = cnt == 0
    out.append(F[keep])
    parent.append(np.nonzero(keep)[0])
    # rotate each split face so the pattern starts at a fixed corner
    for fi in np.nonzero(~keep)[0]:
        a, b, c = F[fi]
        ab, bc, ca = m01[fi], m12[fi], m20[fi]
        corners = [(a, b, c, ab, bc, ca), (b, c, a, bc, ca, ab), (c, a, b, ca, ab, bc)]
        for x, y, z, xy, yz, zx in corners:
            if cnt[fi] == 1 and xy >= 0:
                tris = [(x, xy, z), (xy, y, z)]
                break
            if cnt[fi] == 2 and xy >= 0 and yz >= 0:
                # quad (x, xy, yz, z): the diagonal giving the larger minimum angle (no slivers
                # from consecutive points of one riser edge)
                q1 = [(x, xy, z), (xy, yz, z)]
                q2 = [(x, xy, yz), (x, yz, z)]
                tris = (q1 if _min_angle(V2, q1) >= _min_angle(V2, q2) else q2) + [(xy, y, yz)]
                break
            if cnt[fi] == 3:
                tris = [(x, xy, zx), (xy, y, yz), (zx, yz, z), (xy, yz, zx)]
                break
        out.append(np.array(tris, np.int64))
        parent.append(np.full(len(tris), fi))
    return V2, np.concatenate(out), new_arrays, np.concatenate(parent)


def insert_edges(F, arrays, piece_of, passes=4, tol=1e-4):
    """Insert the riser edges. ``piece_of(arrays_subset) -> (offset, pid, a, c)`` evaluates new
    points; arrays must hold 'p0' (start point), 'pid', 'a', 'c' and every per-vertex array to carry.
    Returns F, arrays, parent face of every final face (index into the input F), inserted count."""
    parent = np.arange(len(F))
    inserted = 0
    for _ in range(passes):
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e = np.unique(np.sort(e, 1), axis=0)
        i, j = e[:, 0], e[:, 1]
        pid = arrays['pid']
        diff = pid[i] != pid[j]
        if 'sheetf' in arrays:  # only edges between two sheet vertices (positions = base + direction x offset)
            diff &= (arrays['sheetf'][i] > .5) & (arrays['sheetf'][j] > .5)
        i, j = i[diff], j[diff]
        if not len(i):
            break
        p_i, p_j = arrays['p0'][i], arrays['p0'][j]
        # e_A(l) = a_A + c_A . p(l), p(l) = p_i + l (p_j - p_i); solve e_A(l) = e_B(l)
        d = p_j - p_i
        A0 = arrays['a'][i] + (arrays['c'][i] * p_i).sum(1)
        A1 = (arrays['c'][i] * d).sum(1)
        B0 = arrays['a'][j] + (arrays['c'][j] * p_i).sum(1)
        B1 = (arrays['c'][j] * d).sum(1)
        den = A1 - B1
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = (B0 - A0) / den
        # a crossing within 10 % of an edge end is left to that vertex (<= ~6 mm): inserting it would
        # cut sliver triangles that the smallest offset difference folds
        ok = np.isfinite(lam) & (lam > .1) & (lam < .9)
        if not ok.any():
            break
        edges, lam = np.c_[i[ok], j[ok]], lam[ok]
        V_dummy = arrays['p0']
        _, F, arrays, par = _split(V_dummy, F, {k: v for k, v in arrays.items()}, edges, lam)
        parent = parent[par]
        new = np.arange(len(arrays['p0']) - len(edges), len(arrays['p0']))
        off, pid_n, a_n, c_n = piece_of({k: v[new] for k, v in arrays.items()})
        arrays['frac'][new], arrays['pid'][new], arrays['a'][new], arrays['c'][new] = off, pid_n, a_n, c_n
        inserted += len(edges)
    return F, arrays, parent, inserted


def crease_normals(V, F, outward_hint, arrays, angle=CREASE_DEG):
    """Vertex copies across creases: corners of faces meeting at an edge folded by less than
    ``angle`` share a vertex (and an area-weighted normal), others get their own copy. Returns
    V, F, normals, arrays (per-vertex arrays copied), and the original index of every vertex."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    un = fn / np.maximum(area, 1e-20)[:, None]
    nf = len(F)
    corner = np.arange(3 * nf).reshape(nf, 3)       # corner id = 3 f + k
    # interior edges: (u, w) -> faces
    e = np.concatenate([np.c_[F[:, 0], F[:, 1], np.arange(nf), np.zeros(nf, int), np.ones(nf, int)],
                        np.c_[F[:, 1], F[:, 2], np.arange(nf), np.ones(nf, int), np.full(nf, 2)],
                        np.c_[F[:, 2], F[:, 0], np.arange(nf), np.full(nf, 2), np.zeros(nf, int)]])
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    swap = e[:, 0] > e[:, 1]
    k_lo = np.where(swap, e[:, 4], e[:, 3])
    k_hi = np.where(swap, e[:, 3], e[:, 4])
    order = np.lexsort((hi, lo))
    lo, hi, fa, kl, kh = lo[order], hi[order], e[order, 2], k_lo[order], k_hi[order]
    same = (lo[1:] == lo[:-1]) & (hi[1:] == hi[:-1])
    a_, b_ = np.nonzero(same)[0], np.nonzero(same)[0] + 1
    smooth = (un[fa[a_]] * un[fa[b_]]).sum(1) > math.cos(math.radians(angle))
    a_, b_ = a_[smooth], b_[smooth]
    # union corners at both ends of each smooth edge
    pairs = np.r_[np.c_[corner[fa[a_], kl[a_]], corner[fa[b_], kl[b_]]], np.c_[corner[fa[a_], kh[a_]], corner[fa[b_], kh[b_]]]]
    lab = np.arange(3 * nf)
    for _ in range(64):
        m = np.minimum(lab[pairs[:, 0]], lab[pairs[:, 1]])
        changed = (lab[pairs[:, 0]] != m) | (lab[pairs[:, 1]] != m)
        np.minimum.at(lab, pairs[:, 0], m)
        np.minimum.at(lab, pairs[:, 1], m)
        lab = lab[lab]
        if not changed.any():
            break
    groups, inv = np.unique(lab, return_inverse=True)
    vert_of_corner = F.reshape(-1)
    orig = np.zeros(len(groups), np.int64)
    orig[inv] = vert_of_corner
    nrm = np.zeros((len(groups), 3))
    np.add.at(nrm, inv, np.repeat(fn, 3, axis=0))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    if outward_hint is not None and (nrm * outward_hint[orig]).sum() < 0:
        nrm = -nrm
    F2 = inv.reshape(nf, 3)
    return V[orig], F2, nrm, {k: v[orig] for k, v in arrays.items()}, orig
