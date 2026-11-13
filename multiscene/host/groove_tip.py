"""Continuous joint grooves. Shared by the overlay erode and NormalExact, so both evaluate one definition.

erode digs a joint groove per planar piece of a fracture interface only where the surface point's projection onto the
piece's plane lies inside the piece (truth._inside_triangles). That makes the groove depth jump to 0 at two kinds of
piece boundary: - the fracture's own outline (trace tips): the groove ends as a cliff; - an internal edge between two
planar pieces of one trace (a polyline kink): on the kink's convex side a point projects outside both pieces, a
wedge-shaped notch with no groove. Here, per fracture interface (all its pieces), with the groove profile itself
unchanged: - outline = triangle edges used once in the whole interface (endpoints matched at 1e-9 m, as erode's cm
route); the depth is multiplied by smoothstep(distance(projection, outline) / taper_m): the groove fades to 0 at the
tips (the cm route's groove_end_taper, but measured to the fracture outline, not to each piece's boundary, so kinks
are not notched); - ownership at kinks: every internal edge between pieces A and B defines the plane through the edge
that bisects the two pieces (normal m = u_A - u_B, u = unit in-plane direction from the edge into the piece); a point
uses piece A only on A's side of all of A's bisectors. On a bisector the distances to A's and B's planes are equal, so
the groove centre term is continuous there (up to the small dig-direction term fa * moved . n, which differs between
the two normals); - a point owned by A whose projection lies outside A's triangles but is nearer to one of A's
internal edges than to A's outline edges still uses A (the groove continues across the kink instead of stopping at the
piece edge). Pieces of a single-plane fracture have no internal edges: only the tip taper applies."""
import nogpu  # noqa: F401,E402  (CPU only)
from collections import Counter

import numpy as np

from outcrop.dsgen.truth import _inside_triangles, _plane_groups


def segment_distance(points, segments, chunk=4096):
    """min distance from each point to 3D segments: the NumPy branch of outcrop.dsgen.display.segment_distance, always"""
    a, b = segments[:, 0], segments[:, 1]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-18)
    best = np.full(len(points), np.inf)
    for s in range(0, len(points), chunk):
        q = points[s:s + chunk, None, :]
        t = np.clip(((q - a) * ab).sum(-1) / denom, 0, 1)
        best[s:s + chunk] = np.linalg.norm(q - (a + t[..., None] * ab), axis=-1).min(1)
    return best, None


def _edge_keys(tri):
    E = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    opp = np.concatenate([tri[:, 2], tri[:, 0], tri[:, 1]])
    K = np.round(E.reshape(-1, 6), 9).tolist()
    return E, opp, [tuple(sorted((tuple(k[:3]), tuple(k[3:])))) for k in K]


def _inplane_dir(a, b, opp, normal):
    L = (b - a) / np.linalg.norm(b - a)
    u = opp - a - ((opp - a) @ L) * L - ((opp - a) @ normal) * normal
    return u / np.linalg.norm(u)


class Fracture:
    def __init__(self, iv, if_):
        self.groups = _plane_groups(np.asarray(iv), np.asarray(if_))
        allk = []
        per = []
        for normal, offset, tri in self.groups:
            E, opp, keys = _edge_keys(tri); per.append((E, opp, keys)); allk += keys
        count = Counter(allk)
        owner = {}
        for g, (E, opp, keys) in enumerate(per):
            c = Counter(keys)
            for i, k in enumerate(keys):
                if c[k] == 1: owner.setdefault(k, []).append((g, i))
        self.outline = np.array([E[i] for g, (E, opp, keys) in enumerate(per) for i, k in enumerate(keys) if count[k] == 1], float).reshape(-1, 2, 3)
        self.piece = []
        for g, (normal, offset, tri) in enumerate(self.groups):
            E, opp, keys = per[g]; c = Counter(keys); half, internal, out = [], [], []
            for i, k in enumerate(keys):
                if c[k] != 1: continue                       # interior edge of the piece itself
                if count[k] == 1:
                    out.append(E[i]); continue
                others = [(h, j) for h, j in owner.get(k, []) if h != g]
                if not others:
                    out.append(E[i]); continue
                h, j = others[0]; a, b = E[i]
                uA = _inplane_dir(a, b, opp[i], normal); uB = _inplane_dir(a, b, per[h][1][j], self.groups[h][0])
                m = uA - uB; nm = np.linalg.norm(m)
                if nm < 1e-9: continue                         # coplanar neighbour (not expected: groups are coplanar sets)
                half.append((0.5 * (a + b), m / nm)); internal.append(E[i])
            self.piece.append(dict(half=half, internal=np.array(internal, float).reshape(-1, 2, 3), outline=np.array(out, float).reshape(-1, 2, 3)))

    def allow(self, g, surface, proj, normal, tri):
        """bool mask: points (surface, their projection proj onto piece g) that use piece g"""
        pc = self.piece[g]
        own = np.ones(len(surface), bool)
        for k, m in pc['half']:
            own &= (surface - k) @ m >= 0
        ins = _inside_triangles(proj, tri, normal, eps=1e-4)
        out = own & ~ins
        if len(pc['internal']) and out.any():
            di = segment_distance(proj[out], pc['internal'])[0]
            do = segment_distance(proj[out], pc['outline'])[0] if len(pc['outline']) else np.full(len(di), np.inf)
            ext = np.zeros(len(surface), bool); ext[np.flatnonzero(out)] = np.asarray(di) < np.asarray(do)
            return own & (ins | ext)
        return own & ins

    def taper(self, proj, taper_m):
        if not len(self.outline): return np.ones(len(proj))
        t = np.clip(np.asarray(segment_distance(proj, self.outline)[0], float) / taper_m, 0, 1)
        return t * t * (3 - 2 * t)


def fractures(interfaces):
    """{iid: Fracture} for every interface (built lazily by the callers)."""
    return {iid: Fracture(iv, if_) for iid, iv, if_ in interfaces if len(if_)}
