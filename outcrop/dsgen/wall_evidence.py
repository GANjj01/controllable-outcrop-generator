"""Retained raw/welded whole-wall polygons and their original boundary checks."""
import json
from pathlib import Path

import numpy as np


def save_polygons(path, polys, extra=None):
    """Save actual polygon vertices and provenance without replacing evidence. ``extra``: triangles of the B fast path
    appended as floor polygons, (points (n, 3, 3), cell (n,), prism (n,))."""
    lengths = np.fromiter((len(p) for p, _, _ in polys), dtype=np.int64, count=len(polys))
    points = np.concatenate([np.asarray(p, float) for p, _, _ in polys]) if polys else np.zeros((0, 3))
    kind = np.asarray([k for _, k, _ in polys], dtype='U5')
    cell = np.asarray([info['cell'] for _, _, info in polys], dtype=np.int64)
    prism = np.asarray([info['prism'] for _, _, info in polys], dtype=np.int64)
    if extra is not None and len(extra[0]):
        lengths = np.r_[lengths, np.full(len(extra[0]), 3, np.int64)]
        points = np.concatenate([points, np.asarray(extra[0], float).reshape(-1, 3)])
        kind = np.r_[kind, np.full(len(extra[0]), 'floor', dtype='U5')]
        cell = np.r_[cell, np.asarray(extra[1], np.int64)]
        prism = np.r_[prism, np.asarray(extra[2], np.int64)]
    offsets = np.r_[0, np.cumsum(lengths)]
    with Path(path).open('xb') as stream:
        np.savez(stream, points=points, offsets=offsets, kind=kind, cell=cell, prism=prism)


def _polygon_edges(path, owner_start):
    with np.load(path, allow_pickle=False) as z:
        points, offsets = z['points'], z['offsets']
        n = len(offsets) - 1
        # Every original polygon edge, including its closing edge. No triangulation.
        next_point = np.arange(len(points)) + 1
        next_point[offsets[1:] - 1] = offsets[:-1]
        owners = np.repeat(owner_start + np.arange(n), np.diff(offsets))
        return points, points[next_point], owners, n


def _non_b_edges(V, F, region):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    owner = np.tile(np.arange(len(F)), 3)
    _, inv, count = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    keep = (count[inv] == 1) & (region[owner] < 0)
    return V[E[keep, 0]], V[E[keep, 1]], owner[keep]


def _exact_edge_certificates(A, B, O):
    """Certify distance zero along an entire segment to another owner's edge.

    Keys use the unchanged floating point coordinates, with endpoints in
    lexicographic order. Neither tolerance bins nor rounded positions are used.
    """
    if not len(A):
        return np.zeros(0, bool)
    reverse = ((A[:, 0] > B[:, 0]) |
               ((A[:, 0] == B[:, 0]) & (A[:, 1] > B[:, 1])) |
               ((A[:, :2] == B[:, :2]).all(1) & (A[:, 2] > B[:, 2])))
    keys = np.empty((len(A), 6), dtype=float)
    keys[:, :3], keys[:, 3:] = A, B
    keys[reverse, :3], keys[reverse, 3:] = B[reverse], A[reverse]
    order = np.lexsort(tuple(keys[:, k] for k in range(5, -1, -1)))
    changed = np.zeros(max(len(order) - 1, 0), bool)
    for k in range(6):
        column = keys[order, k]
        changed |= column[1:] != column[:-1]
    starts = np.r_[0, np.flatnonzero(changed) + 1]
    owners = O[order]
    distinct = np.minimum.reduceat(owners, starts) != np.maximum.reduceat(owners, starts)
    certified = np.empty(len(A), bool)
    certified[order] = np.repeat(distinct, np.diff(np.r_[starts, len(A)]))
    return certified


def _check_edges(A, B, O, from_b):
    from .wallcheck import _match_edges

    certified = from_b & _exact_edge_certificates(A, B, O)
    residual = from_b & ~certified
    if residual.any():
        # Certified edges remain candidates: an unpaired short T-junction edge
        # can lie along a certified longer edge of another owner.
        result = _match_edges(A, B, O, residual)
    else:
        result = {'samples': 0, 'gap': 0, 'gap_max_m': 0.,
                  'gap_samples_unmatched_any': 0, 'gap_points': []}
    result['open_edges_b'] = int(from_b.sum())
    result['open_edges_all'] = len(A)
    result['exact_certified_edges'] = int(certified.sum())
    return result


def check_polygon_stages(staged):
    """Check all regions together against other polygons and final non-B edges."""
    staged = Path(staged)
    request = json.loads((staged / 'request.json').read_text(encoding='utf8'))
    regions = [int(b['region']) for b in request['display_stats']['blockface']['blocks']]
    with np.load(staged / 'display.npz', allow_pickle=False) as z:
        na, nb, no = _non_b_edges(z['vertices'], z['faces'], z['b_region'])
    results = {}
    for stage in ('raw', 'welded'):
        starts, ends, owners, counts = [], [], [], {}
        owner_start = 0
        for r in regions:
            a, b, o, n = _polygon_edges(staged / 'wall_debug' / f'{stage}_region_{r}.npz', owner_start)
            starts.append(a)
            ends.append(b)
            owners.append(o)
            counts[str(r)] = n
            owner_start += n
        # Non-B owners are disjoint from every polygon owner in every region.
        starts.append(na)
        ends.append(nb)
        owners.append(no + owner_start)
        A, B, O = np.concatenate(starts), np.concatenate(ends), np.concatenate(owners)
        from_b = np.arange(len(A)) < len(A) - len(na)
        result = _check_edges(A, B, O, from_b)
        result['polygon_counts'] = counts
        result['pass'] = result['gap'] == 0
        results[stage] = result
    return results
