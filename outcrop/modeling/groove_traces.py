"""Query-local exact traces for one finite normal-displacement groove graph.

The cap remains a curved, bounded-error surface. Only a target whose entire
exact patch projects into the constant-depth centerline can use this proof.
"""
from dataclasses import dataclass

import numpy as np

from .errors import unresolved
from .kernel.intersections import IntersectionResult, triangle_intersection
from .kernel.predicates import point, sub, cross, dot, rational
from .kernel.surface_arrangement import SurfaceCells


@dataclass(frozen=True)
class _NormalGraphTraceCertificate:
    domain: SurfaceCells
    normal: tuple
    segment: tuple
    depth: object


class _TraceWork:
    """One work allowance for all proof and intersection passes in extraction."""
    def __init__(self, request):
        self.request = request
        self.pairs = 0
        self.retained_bytes = 0
        self.workspace_bytes = 0

    def consume(self, count=1):
        self.pairs += count
        if self.pairs > self.request.budget.max_predicate_pairs:
            unresolved('groove.trace', 'combined certificate predicate budget exhausted')

    def begin(self, inputs):
        self.workspace_bytes = self.retained_bytes
        self.reserve(inputs)

    def reserve(self, count):
        self.workspace_bytes += count
        if self.workspace_bytes > self.request.budget.max_bytes:
            unresolved('groove.trace', 'certificate workspace/output memory budget exhausted')

    def retain(self, segments, points):
        # Float results survive this call in extract_traces' accumulation lists.
        self.retained_bytes += segments * 2048 + points * 1024


def _intersections(first, second, work):
    # This evaluates a PL proxy only after the graph theorem has been checked;
    # it does not pass a curved surface through the generic exact-surface gate.
    from .kernel.spatial import AABBTree
    from .kernel.types import AABB
    arrays = np.asarray(second, float).reshape(-1, 3, 3)
    low = np.nextafter(arrays.min(1), -np.inf) if len(arrays) else np.empty((0, 3))
    high = np.nextafter(arrays.max(1), np.inf) if len(arrays) else np.empty((0, 3))
    tree = AABBTree(low, high)
    segments = set()
    points = set()
    for tri in first:
        array = np.asarray(tri, float)
        bounds = AABB(tuple(np.nextafter(array.min(0), -np.inf)),
                      tuple(np.nextafter(array.max(0), np.inf)))
        for index in tree.query(bounds):
            work.consume()
            kind, data = triangle_intersection(tri, second[index])
            if kind == 'line':
                segment = tuple(sorted(data))
                if segment not in segments:
                    work.reserve(4096)
                    segments.add(segment)
            elif kind == 'point':
                for p in data:
                    if p not in points:
                        work.reserve(2048)
                        points.add(p)
            elif kind == 'area':
                unresolved('groove.trace', 'constant-depth trace unexpectedly has coplanar area')
    return segments, points


def _coverage(segments, points, anchor, edge, work):
    work.reserve((len(segments) + len(points)) * 256)
    axis = max(range(3), key=lambda j: abs(edge[j]))
    intervals = []
    for segment in segments:
        work.consume()
        if any(any(cross(edge, sub(p, anchor))) for p in segment):
            unresolved('groove.trace', 'proxy has a branch outside the certified centerline')
        intervals.append(tuple(sorted(p[axis] for p in segment)))
    merged = []
    for lo, hi in sorted(intervals):
        work.consume()
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    isolated = []
    for p in sorted(points):
        work.consume()
        if any(cross(edge, sub(p, anchor))):
            unresolved('groove.trace', 'proxy has a point outside the certified centerline')
        covered = False
        for lo, hi in merged:
            work.consume()
            if lo <= p[axis] <= hi:
                covered = True
                break
        if not covered:
            isolated.append(p[axis])
    return tuple(merged), tuple(sorted(set(isolated))), axis


def certified_groove_intersection(target, proxy, certificate, request, work):
    if target.position_error_m and not hasattr(target, 'exact_vertices'):
        unresolved('groove.trace', 'target needs an exact surface certificate')
    domain = certificate.domain
    if domain.position_error_m and not hasattr(domain, 'exact_vertices'):
        unresolved('groove.trace', 'base domain needs an exact surface certificate')
    faces = len(domain.faces) + len(target.faces) + len(proxy.faces)
    storage = sum(len(c.vertices) * 512 + len(c.faces) * 128 for c in (domain, target, proxy))
    if faces > request.budget.max_faces or storage > request.budget.max_bytes:
        unresolved('groove.trace', 'certificate face/memory budget exhausted')
    work.begin(storage)
    # exact_triangles can construct a fresh Fraction point at each proxy face
    # corner. Reserve those tuples/arithmetic and the AABB arrays/tree before
    # materializing them, then retain both passes' sets until coverage agrees.
    work.reserve(faces * (3 * 1024 + 512))
    original = domain.exact_triangles
    anchor = original[0][0]
    base_n = cross(sub(original[0][1], anchor), sub(original[0][2], anchor))
    normal = certificate.normal
    a, b = certificate.segment
    edge = sub(b, a)
    length2 = dot(edge, edge)
    denominator = dot(base_n, normal)
    if not length2 or not denominator or any(cross(base_n, normal)):
        unresolved('groove.trace', 'displacement must be exactly normal to the base plane')
    if dot(base_n, sub(a, anchor)) or dot(base_n, sub(b, anchor)):
        unresolved('groove.trace', 'profile must lie in the exact base plane')
    target_triangles = target.exact_triangles
    target_anchor = target_triangles[0][0]
    target_n = cross(sub(target_triangles[0][1], target_anchor),
                     sub(target_triangles[0][2], target_anchor))
    if not any(target_n) or dot(target_n, normal):
        unresolved('groove.trace', 'target plane must contain the displacement direction')
    # Linear constraints at every vertex certify every point of every target
    # triangle, including disconnected patches and support holes.
    for tri in target_triangles:
        for p in tri:
            work.consume()
            if dot(target_n, sub(p, target_anchor)):
                unresolved('groove.trace', 'target patch must be exactly coplanar')
            height = dot(base_n, sub(p, anchor)) / denominator
            projected = tuple(x - height * n for x, n in zip(p, normal))
            alpha = dot(sub(projected, a), edge) / length2
            if any(cross(edge, sub(projected, a))) or not 0 <= alpha <= 1:
                unresolved('groove.trace', 'full target projection leaves the finite centerline')
    translated = tuple(tuple(tuple(x - certificate.depth * n for x, n in zip(p, normal))
                             for p in tri) for tri in original)
    true_segments, true_points = _intersections(target_triangles, translated, work)
    proxy_segments, proxy_points = _intersections(target_triangles, proxy.exact_triangles, work)
    lowered_anchor = tuple(x - certificate.depth * n for x, n in zip(a, normal))
    truth = _coverage(true_segments, true_points, lowered_anchor, edge, work)
    actual = _coverage(proxy_segments, proxy_points, lowered_anchor, edge, work)
    if truth != actual:
        unresolved('groove.trace', 'sectioned proxy and true graph have different complete trace coverage')
    intervals, isolated, axis = truth
    work.reserve((2 * len(intervals) + len(isolated)) * 1024)
    def at(value):
        alpha = (value - lowered_anchor[axis]) / edge[axis]
        return tuple(x + alpha * d for x, d in zip(lowered_anchor, edge))
    exact_segments = tuple((at(lo), at(hi)) for lo, hi in intervals)
    exact_points = tuple(at(value) for value in isolated)
    from .kernel.mapped_surfaces import _rounded
    from .kernel.certified_arithmetic import float_upper
    values, rounding = _rounded([p for s in exact_segments for p in s] + list(exact_points))
    if rational(proxy.position_error_m) + rational(target.position_error_m) + rounding > rational(request.max_position_error_m):
        unresolved('groove.trace', 'combined surface/trace rounding position budget exhausted')
    kind = 'mixed' if exact_segments and exact_points else 'line' if exact_segments else 'point' if exact_points else 'empty'
    count = len(exact_segments) * 2
    result = IntersectionResult(kind, values[:count].reshape(-1, 2, 3), values[count:].reshape(-1, 3),
                                SurfaceCells.empty(), {'method': 'exact_constant_depth_normal_graph',
                                                       'pairs': work.pairs, 'rounding_error_m': float_upper(rounding)})
    work.retain(len(exact_segments), len(exact_points))
    return result
