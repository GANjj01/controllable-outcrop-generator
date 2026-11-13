"""Exposure-surface contract of the sandbox.

An exposure is a set of patches, not one continuous surface: parametric grids (escarpment sheet, bluff ray grid,
roadcut faces, quarry benches) or arbitrary triangle meshes (overhangs, caves). Seams declare which patch vertices are
the same point and must stay watertight; surface processes run on the welded union, so seam offsets agree by
construction. A declared seam pair is NOT welded where the v2 volume is not occupied between its two points (an open
joint or other real void): geological discontinuities are kept, never smoothed over by the mesh."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Patch:
    """One piece of the exposed surface.

    vertices   reference positions (n, 3), before surface processes
    faces      triangles (m, 3), consistently wound
    outward    unit outward offset direction per vertex (n, 3); positive process offsets move a
               point against it (recession into the rock)
    apply      optional offsets (n,) -> positions (n, 3); default ``vertices - outward * offset``
               (the bluff re-solves its envelope along rays instead)
    weights    tri-planar weights (n, 3) for facet relief; default from the outward field
    role       'rock' (surface processes and v2 labels apply) or a non-rock semantic class
    hold       optional bool (n,): vertices the surface processes leave in place (rock under a
               road or lining item, which would otherwise poke through it)
    bare       optional bool (n,): freshly excavated surface (cut faces, bench treads): bare rock
               whatever its slope, no soil cover from the slope rule
    """
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    outward: np.ndarray
    apply: object = None
    weights: np.ndarray = None
    role: str = 'rock'
    provenance: dict = field(default_factory=dict)
    hold: np.ndarray = None
    bare: np.ndarray = None

    def positions(self, offset):
        if self.apply is not None:
            return self.apply(offset)
        return self.vertices - self.outward * np.asarray(offset, float)[:, None]

    def tri_weights(self):
        if self.weights is not None:
            return self.weights
        w = np.abs(self.outward) ** 4
        return w / np.maximum(w.sum(1, keepdims=True), 1e-20)


@dataclass
class Item:
    """Non-rock scene element added by the exposure or by surface processes (ground, road, water,
    debris). It carries its own semantic class and provenance; it never claims v2 geology."""
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    semantic: str
    colors: np.ndarray = None
    normals: np.ndarray = None
    provenance: dict = field(default_factory=dict)


@dataclass
class ExposureSurface:
    kind: str
    patches: list
    seams: list = field(default_factory=list)   # (patch_a, idx_a, patch_b, idx_b) vertex index arrays
    items: list = field(default_factory=list)   # Item, or callables (rng, seed, soil_rgb) -> Item
    camera: dict = field(default_factory=dict)  # view_direction, focus rule, min camera height
    profile: dict = field(default_factory=dict)  # surface-process profile (natural / fresh / ...)
    ground_z_m: float = 0.
    origin_m: object = None                      # interior point for outward orientation checks


def weld(surface, truth=None, eps=.01):
    """Merge rock patches into one mesh. Returns (V, F, outward, weights, patch_of_vertex,
    representative index per original vertex, seam report). Seam pairs are welded unless v2 says
    the midpoint between them is not occupied rock (a real opening)."""
    rock = [p for p in surface.patches if p.role == 'rock']
    offsets = np.cumsum([0] + [len(p.vertices) for p in rock])
    index = {p.id: k for k, p in enumerate(rock)}
    V = np.vstack([p.vertices for p in rock])
    F = np.vstack([p.faces + offsets[k] for k, p in enumerate(rock)])
    N = np.vstack([p.outward for p in rock])
    W = np.vstack([p.tri_weights() for p in rock])
    owner = np.concatenate([np.full(len(p.vertices), k) for k, p in enumerate(rock)])
    hold = np.concatenate([np.zeros(len(p.vertices), bool) if p.hold is None else np.asarray(p.hold, bool) for p in rock])
    bare = np.concatenate([np.zeros(len(p.vertices), bool) if p.bare is None else np.asarray(p.bare, bool) for p in rock])
    rep = np.arange(len(V))
    kept, opened = 0, 0
    for a, ia, b, ib in surface.seams:
        if a not in index or b not in index:
            continue
        ga = np.asarray(ia) + offsets[index[a]]
        gb = np.asarray(ib) + offsets[index[b]]
        ok = np.ones(len(ga), bool)
        if truth is not None and len(ga):
            st, _, _, _ = truth.query(.5 * (V[ga] + V[gb]))
            ok = st == 'occupied'
        rep[gb[ok]] = ga[ok]
        kept += int(ok.sum())
        opened += int((~ok).sum())
    # Resolve chains (b -> a -> ...) and renumber the surviving vertices.
    for _ in range(4):
        rep = rep[rep]
    survivors, new_index = np.unique(rep, return_inverse=True)
    Fw = new_index[F]
    return dict(V=V[survivors], F=Fw, outward=N[survivors], weights=W[survivors], owner=owner[survivors], hold=hold[survivors], bare=bare[survivors],
                rock=rock, map=new_index, offsets=offsets, seams={'welded_pairs': kept, 'kept_open_pairs': opened})


def positions(welded, offset):
    """Apply per-vertex offsets on the welded mesh through each patch's own ``apply``."""
    out = np.empty_like(welded['V'])
    for k, patch in enumerate(welded['rock']):
        lo, hi = welded['offsets'][k], welded['offsets'][k + 1]
        idx = welded['map'][lo:hi]                    # welded index of every original vertex
        out[idx] = patch.positions(offset[idx])
    return out
