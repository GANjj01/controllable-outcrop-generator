"""Physical mesh boundaries, including geometrically matching T subdivisions."""
import numpy as np

from outcrop.modeling.kernel.spatial import AABBTree
from outcrop.modeling.kernel.types import AABB


def _indexed_boundary(vertices, faces):
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, first, incidence = np.unique(np.sort(directed, axis=1), axis=0,
                                    return_index=True, return_counts=True)
    return vertices[directed[first[incidence == 1]]], int((incidence > 2).sum())


def _remaining(segment, candidates, tolerance, opposed):
    delta = segment[1] - segment[0]
    length = float(np.linalg.norm(delta))
    unit = delta / length
    if not len(candidates):
        return [segment]
    distance = np.linalg.norm(np.cross(candidates - segment[0], unit), axis=2).max(1)
    valid = distance <= tolerance
    if opposed:
        valid &= ((candidates[:, 1] - candidates[:, 0]) * unit).sum(1) < 0
    intervals = np.sort(((candidates[valid] - segment[0]) * unit).sum(2), axis=1)
    intervals[:, 0] = np.maximum(intervals[:, 0], 0.)
    intervals[:, 1] = np.minimum(intervals[:, 1], length)
    intervals = intervals[intervals[:, 1] > intervals[:, 0]]
    intervals = intervals[np.argsort(intervals[:, 0])]
    cursor, result = 0., []
    for lower, upper in intervals:
        if lower > cursor + tolerance:
            result.append(np.asarray([segment[0] + unit * cursor, segment[0] + unit * lower]))
        cursor = max(cursor, float(upper))
    if cursor < length - tolerance:
        result.append(np.asarray([segment[0] + unit * cursor, segment[1]]))
    return result


def _physical_boundary(vertices, faces, tolerance):
    indexed, nonmanifold = _indexed_boundary(vertices, faces)
    if not len(indexed):
        return indexed, indexed, nonmanifold
    tree = AABBTree(indexed.min(1) - tolerance, indexed.max(1) + tolerance)
    remaining = []
    for segment in indexed:
        candidate = tree.query(AABB(tuple(segment.min(0) - tolerance), tuple(segment.max(0) + tolerance)))
        remaining.extend(_remaining(segment, indexed[candidate], tolerance, opposed=True))
    return indexed, np.asarray(remaining, float).reshape(-1, 2, 3), nonmanifold


def validate_boundary(vertices, faces, *, contract, reference=None, tolerance=1e-8):
    """Reject true holes; an open terrain may retain only its source outer edge.

    ``reference`` is the original complete exposure patch, before surface
    processes. Indexed single edges with opposite collinear partners are
    matching subdivisions, rather than openings. No vertices are welded or
    moved by this check.
    """
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    if contract not in ('closed_rock', 'open_terrain'):
        raise ValueError('Unknown physical boundary contract')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('A nonempty three-dimensional mesh is required')
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid physical mesh coordinates or indices')
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError('Degenerate physical mesh triangle')
    indexed, physical, nonmanifold = _physical_boundary(vertices, faces, tolerance)
    if nonmanifold:
        raise ValueError('Nonmanifold physical boundary')
    source_count = 0
    if contract == 'closed_rock':
        if len(physical):
            raise ValueError('Closed rock has %d physical boundary segments' % len(physical))
        volume = float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.)
        if volume <= 0:
            raise ValueError('Closed rock must have positive oriented volume')
    else:
        if reference is None:
            raise ValueError('Open terrain requires its original outer boundary')
        _, original, original_nonmanifold = _physical_boundary(np.asarray(reference[0]), np.asarray(reference[1]), tolerance)
        source_count = len(original)
        if original_nonmanifold or not len(original):
            raise ValueError('Open terrain source boundary is invalid')
        if len(physical):
            tree = AABBTree(original.min(1) - tolerance, original.max(1) + tolerance)
            for segment in physical:
                selected = tree.query(AABB(tuple(segment.min(0) - tolerance), tuple(segment.max(0) + tolerance)))
                if _remaining(segment, original[selected], tolerance, opposed=False):
                    raise ValueError('Terrain has a physical hole or a geology-domain truncation boundary')
        if not len(physical):
            raise ValueError('Open terrain lost its declared outer boundary')
    return dict(contract=contract, closed=not len(physical), physical_boundary_edges=len(physical),
        index_boundary_edges=len(indexed), t_junction_boundary_edges=max(0, len(indexed) - len(physical)),
        nonmanifold_edges=nonmanifold, positive_triangle_areas=True, source_boundary_edges=source_count)


def verify_source_coverage(source_vertices, source_faces, vertices, faces, source_ids):
    """Fail when exact geological extraction truncates any visible source face."""
    def areas(v, f):
        tri = np.asarray(v)[np.asarray(f)]
        return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    expected, actual = areas(source_vertices, source_faces), areas(vertices, faces)
    indices = np.asarray([int(token.rsplit(':', 1)[1]) for token in source_ids], np.int64)
    if len(indices) != len(actual) or len(indices) and (indices.min() < 0 or indices.max() >= len(expected)):
        raise ValueError('Clipped source-face identities do not close')
    measured = np.bincount(indices, weights=actual, minlength=len(expected))
    error = np.abs(measured - expected)
    bad = error > np.maximum(1e-12, expected * 1e-9)
    if bad.any():
        raise ValueError('Geology domain truncates or overlaps %d visible bare-rock source faces; full terrain must be retained' % int(bad.sum()))
    return dict(all_source_faces_preserved=True, source_faces=len(expected),
                source_area_m2=float(expected.sum()), extracted_area_m2=float(actual.sum()),
                maximum_face_area_error_m2=float(error.max(initial=0.)))
