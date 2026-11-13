"""Bounded six-parameter engineering fracture family, in ENU metres.

This is an explicit ordered-section loft, not a fracture-mechanics simulation or
an estimate of a natural shape distribution. Each depth strip uses matching
diagonals on both walls: linear interpolation preserves their positive gap.
Sections have increasing z, depth nodes increasing y, and the cavity stays
inside the rock. Together these construction properties exclude intersections.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MorphologyParameters:
    bend_m: float
    mean_opening_m: float
    opening_variation: float
    depth_m: float
    roughness_m: float
    tip_fraction: float


@dataclass
class MorphologyResult:
    surface: dict
    sections: dict
    rims: list
    checks: dict


PARAMETER_BOUNDS = {
    'bend_m': (0., .12), 'mean_opening_m': (.03, .08),
    'opening_variation': (0., .45), 'depth_m': (.10, .25),
    'roughness_m': (0., .006), 'tip_fraction': (.10, .25),
}


def _sections(parameters):
    for name, value in asdict(parameters).items():
        lo, hi = PARAMETER_BOUNDS[name]
        if not np.isscalar(value) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{name} must be finite and in [{lo}, {hi}], got {value}')
    p = parameters
    # Eight intervals in each tapered tip and 32 in the body. The taper joins
    # are exact nodes; linear rather than quadratic tip onset avoids tiny cells.
    t = np.concatenate((np.linspace(0, p.tip_fraction, 9)[:-1],
                        np.linspace(p.tip_fraction, 1 - p.tip_fraction, 33)[:-1],
                        np.linspace(1 - p.tip_fraction, 1, 9)))
    z = .2 + 1.6 * t
    z[[0, -1]] = [.2, 1.8]
    center = p.bend_m * np.sin(2 * np.pi * t)
    center[[0, -1]] = 0.
    tip_coordinate = np.minimum(t, 1 - t) / p.tip_fraction
    taper = np.sin(.5 * np.pi * np.minimum(tip_coordinate, 1.))
    taper[[0, -1]] = 0.
    raw_width = taper * (1 + p.opening_variation * np.sin(2 * np.pi * t + .3))
    ds = np.hypot(np.diff(center), np.diff(z))
    width = raw_width * p.mean_opening_m / (np.dot((raw_width[:-1] + raw_width[1:]) / 2, ds) / ds.sum())
    depth_fraction = np.linspace(0, 1, 5)
    d, u = depth_fraction[None, :], t[:, None]
    # Fixed basis/phase family; no independent vertex or wall RNG. Common
    # amplitude <= roughness_m; each differential amplitude <= 0.1 of it.
    common = p.roughness_m * taper[:, None] * (
        .7 * np.sin(4 * np.pi * u + .6) * np.sin(.5 * np.pi * d)
        + .3 * np.sin(6 * np.pi * u - .2) * np.sin(np.pi * d))
    differential = .1 * p.roughness_m * taper[:, None] * np.sin(4 * np.pi * u + 1.) * np.sin(np.pi * d)
    y = p.depth_m * taper[:, None] * np.ones_like(d) * d
    zz = np.broadcast_to(z[:, None], y.shape)
    left = np.stack((center[:, None] - width[:, None] / 2 + common - differential, y, zz), axis=-1)
    right = np.stack((center[:, None] + width[:, None] / 2 + common + differential, y, zz), axis=-1)
    # Both tips are the same exact vertex, not tiny rectangular end caps.
    for end, height in ((0, .2), (-1, 1.8)):
        left[end, :, :] = right[end, :, :] = [0., 0., height]
    if not np.all(right[1:-1, :, 0] > left[1:-1, :, 0]):
        raise ValueError('wall ordering failed')
    if not np.all(np.diff(y[1:-1], axis=1) > 0):
        raise ValueError('depth grid ordering failed')
    if min(left[:, :, 0].min() + 1, 1 - right[:, :, 0].max(), .5 - y.max()) <= 0:
        raise ValueError('cavity penetrates outer rock boundary')
    return dict(z_m=z, normalized_z=t, depth_fraction=depth_fraction,
                center_x_m=center, taper=taper, mouth_width_m=width,
                left_wall_enu_m=left, right_wall_enu_m=right,
                common_displacement_m=common, differential_displacement_m=differential)


def flat_display(source):
    """Split only triangle corners, keeping oriented geometry and face metadata."""
    origin = source['faces'].reshape(-1)
    tri = source['vertices'][source['faces']]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    if np.any(length <= 0):
        raise ValueError('Degenerate source triangle')
    display = {name: value.copy() for name, value in source.items()}
    display.update(vertices=source['vertices'][origin].copy(),
                   faces=np.arange(len(origin), dtype=np.int64).reshape(-1, 3),
                   normals=np.repeat(cross / length[:, None], 3, axis=0))
    return display, origin


def _topology_and_rims(vertices, faces, roles, source_ids):
    adjacent = defaultdict(list)
    direction = defaultdict(int)
    for index, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(a), int(b))))
            adjacent[edge].append(index)
            direction[edge] += 1 if a < b else -1
    if any(len(neighbors) != 2 for neighbors in adjacent.values()):
        raise ValueError('Mesh is not a closed two-manifold')
    if any(direction.values()):
        raise ValueError('Adjacent triangle winding is inconsistent')
    if len(vertices) - len(adjacent) + len(faces) != 2:
        raise ValueError('Loft must be a single genus-zero rock boundary')
    rims = []
    for edge, neighbors in sorted(adjacent.items()):
        pair = set(roles[neighbors])
        for side in ('left', 'right'):
            if pair == {'front', f'{side}_wall'}:
                segment = vertices[list(edge)]
                rims.append(dict(kind=f'{side}_rim', segment_enu_m=segment.tolist(),
                    length_m=float(np.linalg.norm(segment[1] - segment[0])),
                    source_vertex_indices=list(edge), source_face_indices=neighbors,
                    source_ids=source_ids[neighbors].tolist()))
    return rims, len(adjacent)


def build_morphology(parameters):
    """Build and validate a deterministic ~1,400-triangle closed rock surface."""
    sections = _sections(parameters)
    left, right, z = (sections[key] for key in ('left_wall_enu_m', 'right_wall_enu_m', 'z_m'))
    # Ring travels front-left, lip, wall, floor, opposite wall/lip,
    # front-right, back-right, back-left; it is counterclockwise in XY.
    rings = []
    for index, height in enumerate(z):
        rings.append(np.vstack(([-1., 0., height], left[index], right[index, ::-1],
                                [1., 0., height], [1., .5, height], [-1., .5, height])))
    for end, height in ((0, 0.), (-1, 2.)):
        ring = rings[end].copy()
        ring[:, 2] = height
        if end == 0:
            rings.insert(0, ring)
        else:
            rings.append(ring)
    rings = np.asarray(rings)
    vertices, inverse = np.unique(rings.reshape(-1, 3), axis=0, return_inverse=True)
    index_rings = inverse.reshape(rings.shape[:2])
    vertices = vertices.tolist()
    faces, roles = [], []
    collapsed = 0
    segment_roles = ['front'] + ['left_wall'] * 4 + ['cavity_floor'] + ['right_wall'] * 4 + ['front', 'outer_right', 'back', 'outer_left']

    def append(face, role, allow_collapse=False):
        nonlocal collapsed
        if len(set(face)) < 3:
            if not allow_collapse:
                raise ValueError('Unexpected collapsed triangle outside the tips')
            collapsed += 1
            return
        triangle = np.asarray([vertices[k] for k in face])
        if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) <= 1e-14:
            raise ValueError('Unexpected zero-area triangle')
        faces.append(face)
        roles.append(role)

    for i in range(len(index_rings) - 1):
        at_collapsed_tip = i in (0, 1, len(index_rings) - 3, len(index_rings) - 2)
        for j, role in enumerate(segment_roles):
            following = (j + 1) % len(segment_roles)
            a, b = index_rings[i, [j, following]]
            d, c = index_rings[i + 1, [j, following]]
            # Opposite traversal of the right wall needs the opposite index
            # diagonal to give both walls the same physical (z, depth) grid.
            pair = ((a, b, d), (b, c, d)) if role == 'right_wall' else ((a, b, c), (a, c, d))
            for face in pair:
                append(face, role, at_collapsed_tip)
    for ring_index, height, role in ((0, 0., 'outer_bottom'), (-1, 2., 'outer_top')):
        center = len(vertices)
        vertices.append([0., .25, height])
        ring = list(dict.fromkeys(index_rings[ring_index].tolist()))
        for a, b in zip(ring, ring[1:] + ring[:1]):
            append((center, b, a) if height == 0 else (center, a, b), role)
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    roles = np.asarray(roles, str)
    source_ids = np.asarray([f'morphology-r1:{role}:{i}' for i, role in enumerate(roles)])
    rims, edge_count = _topology_and_rims(vertices, faces, roles, source_ids)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    volume = float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
    if not 0 < volume < 2:
        raise ValueError('Rock signed volume must be positive and below intact box volume')
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms <= 0):
        raise ValueError('Undefined source vertex normal')
    normals /= norms[:, None]
    n = len(faces)
    surface = dict(vertices=vertices, faces=faces, normals=normals,
        codes=np.ones(n, np.int32), body_ids=np.full(n, 'rock-body'),
        unit_ids=np.full(n, ''), lithology_ids=np.full(n, 'diagnostic-rock'),
        body_valid=np.ones(n, bool), unit_valid=np.zeros(n, bool),
        lithology_valid=np.ones(n, bool), source_ids=source_ids, face_roles=roles)
    widths = right[:, :, 0] - left[:, :, 0]
    mid = (left[:, 0] + right[:, 0]) / 2
    ds = np.linalg.norm(np.diff(mid, axis=0), axis=1)
    mean_width = float(np.dot((widths[:-1, 0] + widths[1:, 0]) / 2, ds) / ds.sum())
    common, differential = sections['common_displacement_m'], sections['differential_displacement_m']
    correlation = float(np.corrcoef((common - differential).ravel(), (common + differential).ravel())[0, 1]) if parameters.roughness_m else None
    edges = np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]), axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    angles = np.arccos(np.clip(-np.sum(edges * np.roll(edges, 1, axis=1), axis=2) / (lengths * np.roll(lengths, 1, axis=1)), -1, 1))
    checks = dict(status='PASS', parameters=asdict(parameters),
        construction='ordered z loft; matched depth grids/diagonals; shared correlated displacement; exact welded tips',
        natural_distribution_or_fracture_mechanics_claimed=False,
        vertices=len(vertices), faces=n, edges=edge_count, boundary_edges=0, nonmanifold_edges=0,
        winding_consistent=True, euler_characteristic=2, signed_volume_m3=volume,
        void_volume_m3=2 - volume, minimum_triangle_area_m2=float(area2.min() / 2),
        minimum_triangle_angle_deg=float(np.degrees(angles.min())),
        maximum_triangle_aspect_ratio=float((lengths.max(1) ** 2 / area2).max()),
        expected_collapsed_tip_triangles_removed=collapsed,
        mean_transverse_opening_m=mean_width, mean_width_metric='mouth transverse x width; trapezoid integrated over piecewise-linear mouth centerline arc length',
        minimum_sampled_internal_opening_m=float(widths[1:-1].min()),
        maximum_mouth_opening_m=float(widths[:, 0].max()),
        measured_bend_m=float(np.max(np.abs(sections['center_x_m']))),
        measured_depth_m=float(left[:, :, 1].max()),
        minimum_back_thickness_m=float(.5 - left[:, :, 1].max()),
        minimum_side_thickness_m=float(min(left[:, :, 0].min() + 1, 1 - right[:, :, 0].max())),
        maximum_centerline_slope=float(np.max(np.abs(np.diff(mid[:, 0]) / np.diff(z)))),
        maximum_wall_z_slope=float(max(np.max(np.abs(np.diff(left[:, :, 0], axis=0) / np.diff(z)[:, None])), np.max(np.abs(np.diff(right[:, :, 0], axis=0) / np.diff(z)[:, None])))),
        wall_displacement_correlation=correlation,
        maximum_common_displacement_m=float(np.abs(common).max()),
        maximum_differential_displacement_m=float(np.abs(differential).max()),
        rim_segments=len(rims), rim_length_m={f'{side}_rim': sum(row['length_m'] for row in rims if row['kind'] == f'{side}_rim') for side in ('left', 'right')},
        source_triangle_role_counts={role: int(np.count_nonzero(roles == role)) for role in sorted(set(roles))},
        continuous_clearance='positive on open tip-to-tip interval by matching piecewise-linear wall interpolation; exactly zero at the two shared endpoints',
        basis='fixed analytic phases: center sin(2pi*t); opening sin(2pi*t+.3); common 4pi/6pi and differential 4pi; no RNG',
        section_stations=len(z), depth_stations=5)
    return MorphologyResult(surface, sections, rims, checks)
