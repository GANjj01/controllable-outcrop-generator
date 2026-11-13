"""Centimetre-scale geometry of the whole-wall scene.

The exposure sheet is sampled coarsely (sheet_edge_m, 12-15 cm candidates) and refined locally where the surface needs
it, instead of a uniform fine sheet or a millimetre scan transfer:

- structure bands: triangles crossing or near the finite v2 plane pieces of faults, joints and intrusive contacts
(selected per triangle against the plane, not per vertex, on the surface the grooves are evaluated on) down to
``struct_edge_m`` -- a groove is sampled across its width, not aliased into a zig-zag; - shape error: an edge is
bisected when the exact surface at its midpoint deviates from the linear interpolation by more than tol(x) = clamp; -
conforming longest-edge bisection (a face with any marked edge has its longest edge marked; 1, 2 or 3 marked edges
split into 2, 3 or 4 children, longest edge first), so no T-junctions and bounded triangle quality.

New vertices are created on the coarse reference sheet (midpoint of the parents' reference positions) and their final
positions come from the exact offset evaluator of the surface processes (processes.erode, cm branch): noise relief,
facets and grooves are re-evaluated at the new point; only fields that are smooth at >= 0.3 m (recession, cover fade,
fold-guard factor, tri-planar weights) are interpolated from the parents. Mesh smoothing is never run on the refined
mesh with the coarse iteration counts: physical scale = iterations x (0.07 / sheet_edge_m)^2 on the uniform coarse
sheet (the historical counts were set on the 7 cm sheet)."""
import time

import numpy as np

HIST_EDGE_M = .07


def smoothing_iterations(n, sheet_edge_m):
    """Iteration count of a graph smoothing set on the historical 7 cm sheet, for the same physical scale on a uniform
    sheet of edge ``sheet_edge_m`` (diffusion length ~ edge x sqrt(iterations))."""
    return max(1, int(round(n * (HIST_EDGE_M / float(sheet_edge_m)) ** 2)))


def edges_of(F):
    """unique undirected edges (sorted pairs) and per face the edge id of its local edge k = (v_k, v_k+1)."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    Es = np.sort(E, 1)
    u, inv = np.unique(Es, axis=0, return_inverse=True)
    return u, inv.reshape(3, len(F)).T


def close_longest(F, fe, L, marked):
    """Longest-edge closure: every face with a marked edge has its longest edge marked (repeat to a fixed point)."""
    longest = fe[np.arange(len(F)), np.argmax(L[fe], 1)]
    while True:
        hit = marked[fe].any(1)
        need = longest[hit]
        new = need[~marked[need]]
        if not len(new):
            return marked
        marked[new] = True


def bisect(F, fe, L, marked, mid_index):
    """Split the faces with marked edges (closure applied): longest edge first, then the other marked edges of the two
    children. mid_index[e] = vertex id of edge e's midpoint. Returns new F (children replace parents, winding kept)."""
    keep = ~marked[fe].any(1)
    out = [F[keep]]
    sel = np.flatnonzero(~keep)
    if not len(sel):
        return F
    f, e = F[sel], fe[sel]
    kL = np.argmax(L[e], 1)
    # rotate so the longest edge is local edge 0 = (a, b)
    r = np.arange(len(sel))
    idx = (kL[:, None] + np.arange(3)[None]) % 3
    f = f[r[:, None], idx]
    e = e[r[:, None], idx]
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    m = mid_index[e[:, 0]]
    mbc_on = marked[e[:, 1]]
    mca_on = marked[e[:, 2]]
    mbc = np.where(mbc_on, mid_index[e[:, 1]], -1)
    mca = np.where(mca_on, mid_index[e[:, 2]], -1)
    # child (m, b, c), optionally split at bc
    s = ~mbc_on
    out.append(np.c_[m[s], b[s], c[s]])
    s = mbc_on
    out.append(np.c_[m[s], b[s], mbc[s]])
    out.append(np.c_[m[s], mbc[s], c[s]])
    # child (a, m, c), optionally split at ca
    s = ~mca_on
    out.append(np.c_[a[s], m[s], c[s]])
    s = mca_on
    out.append(np.c_[a[s], m[s], mca[s]])
    out.append(np.c_[m[s], c[s], mca[s]])
    return np.concatenate(out).astype(F.dtype)


class Refiner:
    """Adaptive refinement driver.

    carry: dict name -> per-vertex array (interpolated fields; must include the evaluator inputs).
    evaluate(sub_carry) -> dict name -> per-point array, must include 'V' (final positions); every key it returns is
    also kept per vertex (exact values for new vertices).
    tol_of(P) -> per-point tolerance (m); struct_need(V, F, fe_edges, L) -> bool per unique edge that must be split
    (structure band rule, already restricted to edges longer than its target).
    """

    def __init__(self, evaluate, tol_of, struct_need=None, min_edge=.005, max_vertices=1_500_000, max_rounds=12, log=None,
                 skip_vertex=None, centroids=False, interior=None):
        self.evaluate, self.tol_of, self.struct_need = evaluate, tol_of, struct_need
        self.centroids = bool(centroids)
        # longest edge. Faces whose interior points all passed are cached (as the known-ok edges).
        self.interior = None if interior is None else np.asarray(interior, float)
        self.min_edge, self.max_vertices, self.max_rounds = float(min_edge), int(max_vertices), int(max_rounds)
        self.log = log or (lambda m: None)
        self.skip_vertex = skip_vertex

    def run(self, F, carry, exact):
        """F: faces; carry: interpolated fields per vertex; exact: evaluated outputs per vertex (incl. 'V').
        Returns F, carry, exact, stats."""
        t0 = time.time()
        stats = {'rounds': [], 'stopped': None}
        # edges whose midpoint was evaluated and found within tolerance: unchanged edges are not evaluated again (an edge is
        # its two vertex ids; vertices never move, so the same edge has the same midpoint error)
        ok_keys = np.zeros(0, np.int64)
        ok_faces = np.zeros(0, np.int64)
        for rnd in range(self.max_rounds):
            V = exact['V']
            u, fe = edges_of(F)
            L = np.linalg.norm(V[u[:, 1]] - V[u[:, 0]], axis=1)
            cand = L > 2 * self.min_edge
            if self.skip_vertex is not None:
                cand &= ~(self.skip_vertex[u[:, 0]] & self.skip_vertex[u[:, 1]])
            keys = u[:, 0].astype(np.int64) * (1 << 31) + u[:, 1]
            if len(ok_keys):
                pos = np.minimum(np.searchsorted(ok_keys, keys), len(ok_keys) - 1)
                known_ok = ok_keys[pos] == keys
            else:
                known_ok = np.zeros(len(u), bool)
            ci = np.flatnonzero(cand & ~known_ok)
            mid_carry = {k: .5 * (a[u[ci, 0]] + a[u[ci, 1]]) for k, a in carry.items()}
            mid_exact = self.evaluate(mid_carry)
            lin = .5 * (V[u[ci, 0]] + V[u[ci, 1]])
            err = np.linalg.norm(mid_exact['V'] - lin, axis=1)
            tol = self.tol_of(lin)
            marked = np.zeros(len(u), bool)
            marked[ci[err > tol]] = True
            ok_keys = np.union1d(ok_keys, keys[ci[err <= tol]])
            if self.interior is not None:
                fl = L[fe].max(1)
                fkey = np.sort(F, 1)
                fkey = (fkey[:, 0].astype(np.int64) * (1 << 21) + fkey[:, 1]) * (1 << 21) + fkey[:, 2]
                if len(ok_faces):
                    pos_ = np.minimum(np.searchsorted(ok_faces, fkey), len(ok_faces) - 1)
                    fknown = ok_faces[pos_] == fkey
                else:
                    fknown = np.zeros(len(F), bool)
                fc = np.flatnonzero(~marked[fe].any(1) & (fl > 2 * self.min_edge) & ~fknown)
                if self.skip_vertex is not None:
                    fc = fc[~self.skip_vertex[F[fc]].all(1)]
                if len(fc):
                    nb_ = len(self.interior)
                    ff = np.repeat(fc, nb_)
                    bb = np.tile(self.interior, (len(fc), 1))
                    cc = {k: (a_[F[ff]] * (bb[:, :, None] if a_.ndim == 2 else bb)).sum(1) for k, a_ in carry.items()}
                    ce = self.evaluate(cc)
                    lin_c = (V[F[ff]] * bb[:, :, None]).sum(1)
                    badp = np.linalg.norm(ce['V'] - lin_c, axis=1) > self.tol_of(lin_c)
                    badf = badp.reshape(-1, nb_).any(1)
                    marked[fe[fc[badf], np.argmax(L[fe[fc[badf]]], 1)]] = True
                    ok_faces = np.union1d(ok_faces, fkey[fc[~badf]])
            if self.centroids:
                # all passed); a failing centroid marks the face's longest edge
                fl = L[fe].max(1)
                fkey = np.sort(F, 1)
                fkey = (fkey[:, 0].astype(np.int64) * (1 << 21) + fkey[:, 1]) * (1 << 21) + fkey[:, 2]
                if len(ok_faces):
                    pos_ = np.minimum(np.searchsorted(ok_faces, fkey), len(ok_faces) - 1)
                    fknown = ok_faces[pos_] == fkey
                else:
                    fknown = np.zeros(len(F), bool)
                fc = np.flatnonzero(~marked[fe].any(1) & (fl > 2 * self.min_edge) & ~fknown)
                if self.skip_vertex is not None:
                    fc = fc[~self.skip_vertex[F[fc]].all(1)]
                if len(fc):
                    cc = {k: a_[F[fc]].mean(1) for k, a_ in carry.items()}
                    ce = self.evaluate(cc)
                    lin_c = V[F[fc]].mean(1)
                    errc = np.linalg.norm(ce['V'] - lin_c, axis=1)
                    badc = errc > self.tol_of(lin_c)
                    marked[fe[fc[badc], np.argmax(L[fe[fc[badc]]], 1)]] = True
                    ok_faces = np.union1d(ok_faces, fkey[fc[~badc]])
            n_err = int(marked.sum())
            n_struct = 0
            if self.struct_need is not None:
                sn = self.struct_need(V, u, L, exact) & cand
                n_struct = int((sn & ~marked).sum())
                marked |= sn
            if not marked.any():
                stats['stopped'] = 'converged'
                break
            marked = close_longest(F, fe, L, marked)
            if self.struct_need is not None:
                # struct-marked edges among the known-ok ones have no midpoint evaluation yet (handled as 'extra' below)
                pass
            # closure can mark edges that were not evaluated (short, held or known ok): evaluate them too
            ev = np.zeros(len(u), bool)
            ev[ci] = True
            extra = np.flatnonzero(marked & ~ev)
            mi = np.flatnonzero(marked)
            nV = len(V)
            if nV + len(mi) > self.max_vertices:
                stats['stopped'] = 'max_vertices'
                self.log('cm refine: vertex budget %d reached at round %d (would add %d)' % (self.max_vertices, rnd, len(mi)))
                break
            pos_in_ci = np.full(len(u), -1, np.int64)
            pos_in_ci[ci] = np.arange(len(ci))
            new_carry = {k: .5 * (a[u[mi, 0]] + a[u[mi, 1]]) for k, a in carry.items()}
            take = pos_in_ci[mi]
            new_exact = {k: np.zeros((len(mi),) + v.shape[1:], v.dtype) for k, v in mid_exact.items()}
            have = take >= 0
            for k in new_exact:
                new_exact[k][have] = mid_exact[k][take[have]]
            if (~have).any():
                ex = self.evaluate({k: v[~have] for k, v in new_carry.items()})
                for k in new_exact:
                    new_exact[k][~have] = ex[k]
            mid_index = np.full(len(u), -1, np.int64)
            mid_index[mi] = nV + np.arange(len(mi))
            F = bisect(F, fe, L, marked, mid_index)
            carry = {k: np.concatenate([a, new_carry[k]]) for k, a in carry.items()}
            exact = {k: np.concatenate([a, new_exact[k]]) for k, a in exact.items()}
            if self.skip_vertex is not None:
                self.skip_vertex = np.concatenate([self.skip_vertex, self.skip_vertex[u[mi, 0]] & self.skip_vertex[u[mi, 1]]])
            st = {'round': rnd, 'edges': int(len(u)), 'evaluated': int(len(ci) + len(extra)), 'known_ok': int(known_ok.sum()), 'marked_error': n_err,
                  'marked_structure_only': n_struct, 'split': int(len(mi)), 'vertices': int(len(exact['V'])), 'faces': int(len(F)),
                  'seconds': time.time() - t0}
            stats['rounds'].append(st)
            self.log('cm refine round %d: %d edges, %d error + %d structure -> %d split, %d vertices, %d faces (%.1f s)'
                     % (rnd, len(u), n_err, n_struct, len(mi), len(exact['V']), len(F), time.time() - t0))
        else:
            stats['stopped'] = 'max_rounds'
        stats['seconds'] = time.time() - t0
        return F, carry, exact, stats


def shape_error(V, F, evaluate_bary, tol_of, samples=3, seed=0):
    """Deviation of the mesh from the exact surface at random barycentric points (samples per face). evaluate_bary(fi,
    bary) -> exact positions. Returns per-sample error and tolerance."""
    g = np.random.default_rng(seed)
    fi = np.repeat(np.arange(len(F)), samples)
    r = g.random((len(fi), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    bary = np.c_[1 - r.sum(1), r]
    lin = (V[F[fi]] * bary[:, :, None]).sum(1)
    ex = evaluate_bary(fi, bary)
    return np.linalg.norm(ex - lin, axis=1), tol_of(lin), fi
