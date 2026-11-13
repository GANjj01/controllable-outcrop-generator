"""Exact shared-endpoint trace chains; source segment order has no spatial meaning.

Chains stop at endpoints and branches, and never bridge a coordinate gap. No
nearest-neighbour tolerance or new geological qualification rule is introduced.
Coordinates, original segment identities and per-segment orientation are kept.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TraceChain:
    interface_id: str
    chain_id: int
    segments: np.ndarray
    segment_ids: tuple
    flipped: tuple
    closed: bool
    endpoint_degrees: tuple

    @property
    def length_m(self):
        return float(np.linalg.norm(self.segments[:, 1] - self.segments[:, 0], axis=1).sum())


@dataclass(frozen=True)
class TraceStation:
    interface_id: str
    chain_id: int
    arc_m: float
    segment_id: int
    source_fraction: float
    point: np.ndarray
    tangent: np.ndarray


def build_chains(segments, interface_ids, *, segment_ids=None):
    """Return maximal nonbranching chains, canonically oriented and sorted.

    Only exactly equal 3-D endpoints belonging to the same interface connect.
    Optional segment_ids preserve external IDs when the input is permuted; by
    default they are the input array indices. Cycles start at their smallest
    coordinate and take the lexicographically smaller traversal direction.
    """
    segments = np.asarray(segments, dtype=float)
    labels = np.asarray(interface_ids)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError('segments must have shape (N, 2, 3)')
    n = len(segments)
    if labels.shape != (n,) or not np.isfinite(segments).all():
        raise ValueError('interface IDs must align with finite segment coordinates')
    if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) == 0):
        raise ValueError('zero-length segments do not define trace tangents')
    source_ids = np.arange(n, dtype=int) if segment_ids is None else np.asarray(segment_ids)
    if source_ids.shape != (n,) or not np.issubdtype(source_ids.dtype, np.integer) or len(set(map(int, source_ids))) != n:
        raise ValueError('segment IDs must be unique integers aligned with segments')
    labels = labels.astype(str)
    result = []
    for interface_id in sorted(set(labels)):
        indices = np.flatnonzero(labels == interface_id)
        endpoints = {int(i): tuple(map(tuple, segments[i])) for i in indices}
        adjacency = {}
        for i, (a, b) in endpoints.items():
            adjacency.setdefault(a, []).append(i)
            adjacency.setdefault(b, []).append(i)

        def edge_key(i):
            return (*sorted(endpoints[i]), int(source_ids[i]))

        for edges in adjacency.values():
            edges.sort(key=edge_key)
        unused = set(endpoints)
        paths = []

        def walk(start, first):
            nodes, edges, node, edge = [start], [], start, first
            while edge in unused:
                unused.remove(edge)
                a, b = endpoints[edge]
                next_node = b if node == a else a
                edges.append(edge)
                nodes.append(next_node)
                node = next_node
                if node == start or len(adjacency[node]) != 2:
                    break
                remaining = [i for i in adjacency[node] if i in unused]
                if not remaining:
                    break
                edge = remaining[0]
            if tuple(reversed(nodes)) < tuple(nodes):
                nodes.reverse()
                edges.reverse()
            paths.append((tuple(nodes), edges))

        for node in sorted(adjacency):
            if len(adjacency[node]) != 2:
                for edge in adjacency[node]:
                    if edge in unused:
                        walk(node, edge)
        while unused:
            edge = min(unused, key=edge_key)
            walk(min(endpoints[edge]), edge)
        for chain_id, (nodes, edges) in enumerate(sorted(paths, key=lambda item: item[0])):
            ordered = np.asarray(list(zip(nodes[:-1], nodes[1:])), dtype=float)
            result.append(TraceChain(
                str(interface_id), chain_id, ordered,
                tuple(int(source_ids[i]) for i in edges),
                tuple(endpoints[i][0] != nodes[j] for j, i in enumerate(edges)),
                nodes[0] == nodes[-1],
                (len(adjacency[nodes[0]]), len(adjacency[nodes[-1]])),
            ))
    return result


def sample_chain(chain, spacing_m):
    """Sample spacing, 2*spacing, ... along one oriented chain, in metres.

    Open-chain endpoints are included when exactly on the grid; a cycle's
    repeated terminal point is omitted. A sample exactly at a vertex uses the
    preceding segment's tangent. No tangent is inferred across a branch/gap.
    """
    spacing_m = float(spacing_m)
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('spacing must be positive and finite')
    lengths = np.linalg.norm(chain.segments[:, 1] - chain.segments[:, 0], axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    total = float(cumulative[-1])
    count = int(np.floor(np.nextafter(total / spacing_m, np.inf)))
    samples = []
    for k in range(1, count + 1):
        arc = min(k * spacing_m, total)
        if chain.closed and arc == total:
            continue
        j = min(max(int(np.searchsorted(cumulative, arc, side='left')) - 1, 0), len(lengths) - 1)
        fraction = float(np.clip((arc - cumulative[j]) / lengths[j], 0., 1.))
        a, b = chain.segments[j]
        samples.append(TraceStation(
            chain.interface_id, chain.chain_id, arc, chain.segment_ids[j],
            1. - fraction if chain.flipped[j] else fraction,
            a + fraction * (b - a), (b - a) / lengths[j],
        ))
    return samples
