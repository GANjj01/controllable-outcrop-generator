"""Finite aperture cutters on caller-owned, unchanged 3D network paths.

The caller classifies real tips and supplies physical fields and inward frames.
No geological path is generated here. Refined stations of one original segment
share a closed PL loft; Boolean union removes caps at original network joins.
This module neither subtracts a host nor labels visible mouth edges.
"""
from dataclasses import dataclass
import json

import manifold3d as md
import numpy as np


@dataclass(frozen=True)
class NetworkCutter:
    solid: object
    left_wall: np.ndarray
    right_wall: np.ndarray
    source_points: np.ndarray
    original_id: int
    face_sources: dict
    construction_mode: str = 'strict_loft'
    segment_left_wall: np.ndarray | None = None
    segment_right_wall: np.ndarray | None = None
    junction_bridges: tuple = ()
    construction_checks: dict | None = None


def _owned(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _self_intersections(vertices, faces):
    """Exact PL intersection check on one small cell, including adjacent folds."""
    from outcrop.modeling.kernel.intersections import triangle_intersection
    triangles = vertices[faces]
    lo, hi = triangles.min(1), triangles.max(1)
    for i, triangle in enumerate(triangles):
        candidates = np.flatnonzero(np.all(lo <= hi[i], axis=1) & np.all(hi >= lo[i], axis=1))
        for j in candidates[candidates > i]:
            kind, points = triangle_intersection(triangle, triangles[j])
            if kind == 'empty':
                continue
            shared = set(map(int, faces[i])) & set(map(int, faces[j]))
            expected = {tuple(vertices[k]) for k in shared}
            if kind == 'area' or any(tuple(map(float, point)) not in expected for point in points):
                raise ValueError('self-intersecting or folded cutter cell')


def _loft_geometry(rings, source_segment_id):
    """Join section rings with shared vertices and only the two outer caps.

    source_segment_id is one original ID or one ID per span (len(rings)-1),
    so an exactly collinear run of several original segments keeps each
    span's identity while sharing a single frame and mesh."""
    vertices, inverse = np.unique(rings.reshape(-1, 3), axis=0, return_inverse=True)
    indices = inverse.reshape(rings.shape[:2])
    count = rings.shape[1]
    levels = count // 2 - 1
    roles = ['left_wall'] * levels + ['floor'] + ['right_wall'] * levels + ['outside_cap']
    span_ids = np.broadcast_to(np.asarray(source_segment_id), (len(rings) - 1,))
    faces, records = [], []

    def append(face, role, segment_id):
        if len(set(map(int, face))) < 3:
            return  # Exact collapsed edge at a zero-width, zero-depth tip.
        triangle = vertices[list(face)]
        if not np.any(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])):
            raise ValueError('non-tip degenerate cutter triangle')
        faces.append(face)
        records.append(dict(source_segment_id=int(segment_id), role=role))

    for station in range(len(rings) - 1):
        for j, role in enumerate(roles):
            following = (j + 1) % count
            a, b = indices[station, [j, following]]
            d, c = indices[station + 1, [j, following]]
            append((a, c, b), role, span_ids[station])
            append((a, d, c), role, span_ids[station])
    # Cross-sections are ordered depth strips, so no fan crosses a rough wall.
    for end, role, segment_id in ((0, 'station_start', span_ids[0]), (len(rings) - 1, 'station_end', span_ids[-1])):
        for face in _cap_strips(indices[end], levels):
            append(face if end == 0 else face[::-1], role, segment_id)
    faces = np.asarray(faces, dtype=np.uint64)
    if len(faces) < 4:
        raise ValueError('adjacent closed tips do not define a cutter volume')
    triangles = vertices[faces]
    # Translate for robust volume at a large world origin.
    q = triangles - vertices[0]
    volume6 = np.einsum('ij,ij->i', q[:, 0], np.cross(q[:, 1], q[:, 2])).sum()
    if volume6 == 0 or not np.isfinite(volume6):
        raise ValueError('cutter cell has zero or invalid volume')
    if volume6 < 0:
        faces = np.ascontiguousarray(faces[:, ::-1])
    return vertices, faces, records


def _cap_strips(ring_indices, levels, strips=None):
    """Cap triangles of one section as depth strips; strip 0 is the outside part (margin to mouth)."""
    count = len(ring_indices)
    out = []
    for j in (range(levels) if strips is None else strips):
        a, b, c, d = ring_indices[[j, j + 1, count - 2 - j, count - 1 - j]]
        out += [(a, b, c), (a, c, d)]
    return out


def _cell(rings, original_id, first_face_id, source_segment_id, *, station_axis=None):
    """Close a loft; validate bounded windows instead of a whole-run N² check."""
    vertices, faces, records = _loft_geometry(rings, source_segment_id)
    if len(rings) == 2:
        _self_intersections(vertices, faces)
    else:
        if station_axis is None:
            raise ValueError('multi-station loft requires its fixed source-segment axis')
        projected = rings @ station_axis
        # Non-neighbour spans occupy disjoint longitudinal slabs. Test the
        # actual ring vertices, including their floating-point residuals.
        if np.any(projected[:-1].max(1) >= projected[1:].min(1)):
            raise ValueError('source-segment sections must advance in disjoint ordered slabs')
        # Two-ring windows retain individual cell checks. Three-ring windows
        # also cover adjacent spans, with no duplicated cap at their join.
        for count in (2, 3):
            for start in range(len(rings) - count + 1):
                ids = np.broadcast_to(np.asarray(source_segment_id), (len(rings) - 1,))
                v, f, _ = _loft_geometry(rings[start:start + count], ids[start:start + count - 1])
                _self_intersections(v, f)
    face_ids = np.arange(first_face_id, first_face_id + len(faces), dtype=np.uint64)
    mesh = md.Mesh64(np.ascontiguousarray(vertices), np.ascontiguousarray(faces),
        run_index=np.array([0, faces.size], dtype=np.uint64),
        run_original_id=np.array([original_id], dtype=np.uint32), face_id=face_ids)
    solid = md.Manifold(mesh)
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError('cutter cell is not a valid oriented volume: ' + str(solid.status()))
    return solid, dict(zip(map(int, face_ids), records))


def _local_union(solids):
    """Evaluate a Boolean near its own bounding-box centre before translating."""
    if len(solids) == 1:
        return solids[0]
    bounds = np.asarray([solid.bounding_box() for solid in solids])
    origin = (bounds[:, :3].min(0) + bounds[:, 3:].max(0)) / 2
    result = md.Manifold.batch_boolean([solid.translate(-origin) for solid in solids], md.OpType.Add)
    # Force evaluation before applying the world transform; otherwise a lazy
    # transform may be pushed down through the Boolean back into large coords.
    if result.status() != md.Error.NoError or result.volume() <= 0:
        raise ValueError('local union failed: ' + str(result.status()))
    return result.translate(origin)


def _sheet_face(row, caps):
    if caps:
        return row.get('role') in ('station_start', 'station_end') or (
            row.get('role') == 'junction_bridge' and row.get('side', 'cap') == 'cap')
    return row.get('role') == 'junction_bridge'


def _drop_bridge_sheets(union, sources, tolerance, caps=False):
    """Detach-and-drop only junction-bridge sheets thinner than the position tolerance.

    A bridge hull whose wedge is thinner than the Boolean's epsilon can leave a detached zero-thickness
    sheet (mean thickness 2|V|/A). It encloses no cutter volume. It is dropped and returned only when
    exactly one other component remains; otherwise the union is returned unchanged for the caller's gate.
    caps=True (mitre_wedge joints) also accepts sheets made only of run caps and wedge end faces: exactly
    coincident cap pairs that the Boolean left uncancelled where a further solid crosses them. A sheet
    with any wall, floor, outside-cap or wedge side face is never dropped."""
    components = union.decompose()
    if union.status() != md.Error.NoError or len(components) < 2:
        return union, components, []
    sheets, kept = [], []
    for part in components:
        ids = np.unique(np.asarray(part.to_mesh64().face_id))
        area = part.surface_area()
        thickness = 2 * abs(part.volume()) / area if area > 0 else 0.
        if (thickness <= tolerance and len(ids)
                and all(_sheet_face(sources.get(int(f), {}), caps) for f in ids)):
            sheets.append(dict(volume_m3=part.volume(), surface_area_m2=area, mean_thickness_m=thickness,
                triangles=part.num_tri(), face_sources=[dict(face_id=int(f), **sources[int(f)]) for f in ids]))
        else:
            kept.append(part)
    if not sheets or len(kept) != 1:
        return union, components, []
    return kept[0], kept, sheets


def _simplify_cutter(union, sources, simplify_m):
    """Cutter-only Manifold.simplify, in the segment's local frame and before any host Boolean.

    simplify keeps the manifold's own tolerance (set_tolerance would carry it into host - void). Fails closed
    unless the result is valid, keeps only original vertices and known face IDs, and moves volume by at most
    simplify_m times the surface area."""
    if not 0 < simplify_m <= 1e-7:
        raise ValueError('cutter simplification is bounded by the 1e-7 m source position tolerance')
    before = union.to_mesh64()
    result = union.simplify(simplify_m)
    after = result.to_mesh64()
    old = {tuple(v) for v in np.asarray(before.vert_properties)[:, :3]}
    new = np.asarray(after.vert_properties)[:, :3]
    unknown = set(map(int, np.asarray(after.face_id))) - set(sources)
    bound = simplify_m * union.surface_area()
    record = dict(tolerance_m=simplify_m, triangles_before=union.num_tri(), triangles_after=result.num_tri(),
                  volume_change_m3=result.volume() - union.volume(), volume_change_bound_m3=bound,
                  manifold_tolerance_after=result.get_tolerance())
    if (result.status() != md.Error.NoError or unknown or any(tuple(v) not in old for v in new)
            or abs(record['volume_change_m3']) > bound or result.get_tolerance() != union.get_tolerance()):
        raise ValueError('cutter simplification left an invalid or non-subset cutter: ' + json.dumps(
            dict(record, status=str(result.status()), unknown_face_ids=len(unknown))))
    return result, record


MITRE_MAX_TURN_DEG = 15.


def _ring_roles(count):
    levels = count // 2 - 1
    return ['left_wall'] * levels + ['floor'] + ['right_wall'] * levels + ['outside_cap']


def _mitre_ring(ring_a, ring_b, node, tangent_a, tangent_b):
    """Each run's node section extended along its own axis onto the bisector plane at the node."""
    bisector = tangent_a + tangent_b
    bisector /= np.linalg.norm(bisector)
    a = ring_a + (-((ring_a - node) @ bisector) / (tangent_a @ bisector))[:, None] * tangent_a
    b = ring_b + (-((ring_b - node) @ bisector) / (tangent_b @ bisector))[:, None] * tangent_b
    return a, b


def _wedge_cell(ring_a, ring_b, depth_side, original_id, first_face_id, record):
    """Gap-side rotational wedge between two node sections that share their two mouth vertices.

    Its two end faces are exactly the matching cap strips of ring_a (end of the previous run) and ring_b
    (start of the next), triangulated as _cap_strips does, so the Boolean cancels them; side faces are
    chords between corresponding vertices. The other half of the node is covered by the overlapping runs."""
    count = len(ring_a)
    levels = count // 2 - 1
    roles = _ring_roles(count)
    if depth_side:
        strips, chain = range(1, levels), list(range(1, count - 1))
    else:
        strips, chain = [0], [count - 2, count - 1, 0, 1]
    rings = np.stack((ring_a, ring_b))
    vertices, inverse = np.unique(rings.reshape(-1, 3), axis=0, return_inverse=True)
    indices = inverse.reshape(2, count)
    faces, records = [], []

    def append(face, side):
        if len(set(map(int, face))) < 3:
            return  # shared mouth vertex
        triangle = vertices[list(face)]
        if not np.any(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])):
            raise ValueError('degenerate wedge joint triangle')
        faces.append(face)
        records.append(dict(record, role='junction_bridge', side=side))

    for face in _cap_strips(indices[0], levels, strips):
        append(face, 'cap')
    for face in _cap_strips(indices[1], levels, strips):
        append(face[::-1], 'cap')
    for j, k in zip(chain[:-1], chain[1:]):
        a, b, c, d = indices[0, j], indices[0, k], indices[1, k], indices[1, j]
        side = roles[j].replace('_wall', '').replace('outside_cap', 'outside')
        append((a, c, b), side)
        append((a, d, c), side)
    faces = np.asarray(faces, dtype=np.uint64)
    q = vertices[faces] - vertices[0]
    volume6 = np.einsum('ij,ij->i', q[:, 0], np.cross(q[:, 1], q[:, 2])).sum()
    if volume6 == 0 or not np.isfinite(volume6):
        raise ValueError('wedge joint has zero or invalid volume')
    if volume6 < 0:
        faces = np.ascontiguousarray(faces[:, ::-1])
    _self_intersections(vertices, faces)
    face_ids = np.arange(first_face_id, first_face_id + len(faces), dtype=np.uint64)
    solid = md.Manifold(md.Mesh64(np.ascontiguousarray(vertices), faces,
        run_index=np.array([0, faces.size], dtype=np.uint64),
        run_original_id=np.array([original_id], dtype=np.uint32), face_id=face_ids))
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError('wedge joint is not a valid oriented volume: ' + str(solid.status()))
    return solid, dict(zip(map(int, face_ids), records))


def _hull_joint(previous, following, node, width, differential, lengths, original_id, first_face_id, record):
    """The original convex node bridge (as in hull mode), used only as the recorded per-node fallback."""
    collar = np.minimum(.25 * np.min(width + 2 * differential), .1 * lengths)
    if np.any(collar <= 0):
        raise ValueError('internal junction must have positive opening')
    before = previous[1] + collar[0] / lengths[0] * (previous[0] - previous[1])
    after = following[0] + collar[1] / lengths[1] * (following[1] - following[0])
    hull = md.Manifold.hull_points(np.vstack((before, previous[1], following[0], after)) - node)
    mesh = hull.to_mesh64()
    faces = np.asarray(mesh.tri_verts, dtype=np.uint64)
    if hull.status() != md.Error.NoError or hull.volume() <= 0 or not len(faces):
        raise ValueError('junction bridge has no finite volume')
    face_ids = np.arange(first_face_id, first_face_id + len(faces), dtype=np.uint64)
    bridge = md.Manifold(md.Mesh64(np.array(mesh.vert_properties[:, :3], dtype=np.float64, copy=True),
        np.array(faces, dtype=np.uint64, copy=True), run_index=np.array([0, faces.size], dtype=np.uint64),
        run_original_id=np.array([original_id], dtype=np.uint32), face_id=face_ids))
    return bridge.translate(node), {int(f): dict(record, role='junction_bridge') for f in face_ids}, collar


def _mitre_wedge_union(runs, frame, sections, points, inward, width, differential, lengths, segment_ids,
                       original_id, sources, tolerance):
    """Every section is computed in path-local coordinates and shared values are written into BOTH runs' own section arrays
    before either cell is built, so coincident caps are bitwise identical. One Boolean over all cells without
    re-centring keeps that identity (a per-group translate round trip need not)."""
    zero = np.zeros(3)
    frames = [frame(a, b) for a, b in runs]
    rings, left_sides, right_sides = [], [], []
    for (a, b), (tangent, lateral, inside) in zip(runs, frames):
        if np.min(inward[a:b + 1] @ inside) <= 0:
            raise ValueError('segment inward disagrees with a supplied endpoint frame')
        left, right, ring = sections(slice(a, b + 1), zero, lateral, inside, mouth=True)
        rings.append(np.array(ring, copy=True))
        for j in range(b - a):
            left_sides.append(left[j:j + 2])
            right_sides.append(right[j:j + 2])
    count = rings[0].shape[1]
    mouth = [1, count - 2]
    nodes = []
    for r in range(1, len(runs)):
        i = runs[r][0]
        (ta, _, na), (tb, _, nb) = frames[r - 1], frames[r]
        ring_a, ring_b = rings[r - 1], rings[r]
        record = dict(station_index=i, adjacent_source_segment_ids=list(map(int, segment_ids[i - 1:i + 1])),
                      theta_deg=float(np.degrees(np.arccos(np.clip(ta @ tb, -1., 1.)))))
        mitre_a, mitre_b = _mitre_ring(ring_a[-1], ring_b[0], points[i], ta, tb)
        mismatch = float(np.max(np.linalg.norm(mitre_a - mitre_b, axis=1)))
        # outer side a mitre corner lies d/cos(theta/2) from the mouth instead of the chord's ~d, so it is
        # used only up to MITRE_MAX_TURN_DEG (corner depth excess <= 0.86%); larger turns get the wedge.
        if (record['theta_deg'] <= MITRE_MAX_TURN_DEG and mismatch <= tolerance
                and (mitre_a @ ta).min() > (ring_a[-2] @ ta).max()
                and (mitre_a @ tb).max() < (ring_b[1] @ tb).min()):
            ring_a[-1] = mitre_a
            ring_b[0] = mitre_a
            nodes.append(dict(record, kind='mitre', mitre_mismatch_m=mismatch))
            continue
        gap = float(np.max(np.linalg.norm(ring_a[-1][mouth] - ring_b[0][mouth], axis=1)))
        if gap <= 1e-12:
            ring_b[0][mouth] = ring_a[-1][mouth]
            nodes.append(dict(record, kind='wedge', depth_side=bool(nb @ ta > 0), mouth_snap_m=gap))
        else:
            nodes.append(dict(record, kind='hull', reason='mouth vertices differ by %.3e m' % gap))
    solids = []
    for (a, b), (tangent, _, _), ring in zip(runs, frames, rings):
        cell, records = _cell(ring, original_id, len(sources), segment_ids[a:b], station_axis=tangent)
        solids.append(cell)
        sources.update(records)
    bridges = []
    for r, node in enumerate(nodes, start=1):
        i = node['station_index']
        record = dict(source_segment_id=node['adjacent_source_segment_ids'][0],
                      adjacent_source_segment_ids=node['adjacent_source_segment_ids'], station_index=i)
        if node['kind'] == 'wedge':
            try:
                solid, records = _wedge_cell(rings[r - 1][-1], rings[r][0], node['depth_side'], original_id,
                                             len(sources), record)
            except ValueError as error:
                node.update(kind='hull', reason='wedge rejected: ' + str(error))
        if node['kind'] == 'hull':
            solid, records, collar = _hull_joint(rings[r - 1][-2:], rings[r][:2], points[i], width[i], differential[i],
                                                 lengths[i - 1:i + 1], original_id, len(sources), record)
            overlaps = [(solid ^ solids[j]).volume() for j in (r - 1, r)]
            if solid.status() != md.Error.NoError or min(overlaps) <= 0:
                raise ValueError('junction bridge must overlap both segment volumes')
            node.update(collar_length_m=collar.tolist(), overlap_volume_m3=overlaps)
        if node['kind'] in ('wedge', 'hull'):
            solids.append(solid)
            sources.update(records)
            bridges.append(dict(node, volume_m3=solid.volume()))
    union = md.Manifold.batch_boolean(solids, md.OpType.Add)
    if union.status() != md.Error.NoError or union.volume() <= 0:
        raise ValueError('cutter union failed: ' + str(union.status()))
    return union, tuple(bridges), left_sides, right_sides, nodes


def _segment_union(points, inward, normal, width, depth, common, differential,
                   margin, original_id, segment_ids, simplify_m=None, joint='hull'):
    """Straight original segments plus bounded, overlapping internal-node joins.

    Consecutive original segments share one frame and mesh only when that frame
    moves every actual section vertex of each of them (all depth/opening
    stations, outside margin included) by at most the source position
    tolerance relative to the segment's own frame; their span IDs are kept."""
    origin = points[len(points) // 2].copy()
    relative = points - origin
    plane_normal = normal[0] / np.linalg.norm(normal[0])
    residual = float(np.max(np.abs(relative @ plane_normal)))
    extent = float(depth.max() + margin + width.max() / 2 + np.abs(common).max() + np.abs(differential).max())
    normal_error = float(np.max(np.linalg.norm(normal - plane_normal, axis=1)) * extent)
    planar = residual <= 1e-7 and normal_error <= 1e-7
    axes = np.eye(3)
    if planar:
        along = points[1] - points[0]
        along -= (along @ plane_normal) * plane_normal
        along /= np.linalg.norm(along)
        axes = np.stack((plane_normal, np.cross(along, plane_normal), along))
        points = relative @ axes.T
        inward = inward @ axes.T
        normal = np.tile([1., 0., 0.], (len(points), 1))
    else:
        points = relative
    checks = dict(planar_frame=planar, source_plane_max_residual_m=residual,
        common_normal_adjustment_bound_m=normal_error if planar else 0.,
        source_points_projected=False, source_position_tolerance_m=1e-7,
        computation='stitched original-segment lofts; straight originals share one only if the actual section '
                    'moves <= the position tolerance; common verified plane where applicable; local grouped Boolean')
    linear_depth_walls = (np.all(common == 0) and np.all(differential == 0)
                          and np.all(width == width[:, :1]))
    checks['exact_linear_depth_walls'] = bool(linear_depth_walls)
    solids, left_sides, right_sides, sources = [], [], [], {}
    span_rings = [None] * (len(points) - 1)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    tolerance = checks['source_position_tolerance_m']

    def frame(run_start, run_end):
        tangent = points[run_end] - points[run_start]
        tangent /= np.linalg.norm(tangent)
        lateral = normal[run_start] + normal[run_end]
        if not planar:
            lateral -= np.dot(lateral, tangent) * tangent
        magnitude = np.linalg.norm(lateral)
        if magnitude <= 1e-10:
            raise ValueError('segment frame cannot preserve the supplied lateral normal')
        lateral /= magnitude
        inside = np.cross(tangent, lateral)
        inside /= np.linalg.norm(inside)
        if np.dot(inside, inward[run_start] + inward[run_end]) < 0:
            inside = -inside
        return tangent, lateral, inside

    def sections(k, run_origin, lateral, inside, mouth=False):
        base = (points[k] - run_origin)[:, None] + depth[k, :, None] * inside
        left = base + (common[k] - width[k] / 2 - differential[k])[..., None] * lateral
        right = base + (common[k] + width[k] / 2 + differential[k])[..., None] * lateral
        if linear_depth_walls and mouth:
            # collinear mouth vertices stay in the planar-wall ring (triangulation only).
            ring = np.stack((left[:, 0] - margin * inside, left[:, 0], left[:, -1],
                             right[:, -1], right[:, 0], right[:, 0] - margin * inside), axis=1)
        elif linear_depth_walls:
            # These fields define exact planar walls in depth. Interior depth
            # samples are redundant, and evaluating many near-coplanar faces
            # creates numerical slivers when an adjacent join is unioned.
            # Keep every ALONG-PATH station; only the analytic linear wall's
            # interior depth samples disappear from its triangulation.
            ring = np.stack((left[:, 0] - margin * inside, left[:, -1],
                             right[:, -1], right[:, 0] - margin * inside), axis=1)
        else:
            ring = np.concatenate(((left[:, 0] - margin * inside)[:, None], left,
                                   right[:, ::-1], (right[:, 0] - margin * inside)[:, None]), axis=1)
        return left, right, ring

    # Spans of one original segment share its frame (tangent through its own
    # endpoints). Consecutive original segments may also share one run frame,
    # but only if, for each of them, that frame moves every actual section
    # vertex of its stations (all depth/opening levels and the outside margin)
    # by at most the source position tolerance from the segment's own frame.
    # A point-to-chord test alone is insufficient: a micrometre segment's
    # angle error is multiplied by the section extent. Such nodes otherwise get
    # two nearly coplanar caps plus a flat hull, whose Boolean left slivers.
    originals, start = [], 0
    for i in range(1, len(points)):
        if i == len(points) - 1 or segment_ids[i] != segment_ids[i - 1]:
            originals.append((start, i))
            start = i
    own = []
    for a_, b_ in originals:
        tangent, lateral, inside = frame(a_, b_)
        own.append(sections(slice(a_, b_ + 1), points[a_], lateral, inside)[2])

    def frame_offset(first, last):
        run_start, run_end = originals[first][0], originals[last][1]
        tangent, lateral, inside = frame(run_start, run_end)
        bound = 0.
        for g in range(first, last + 1):
            a_, b_ = originals[g]
            ring = sections(slice(a_, b_ + 1), points[a_], lateral, inside)[2]
            bound = max(bound, float(np.max(np.linalg.norm(ring - own[g], axis=-1))))
        return bound

    runs, merged, first = [], [], 0
    for g in range(1, len(originals)):
        bound = frame_offset(first, g)
        if bound <= tolerance:
            i = originals[g][0]
            merged.append(dict(station_index=i, adjacent_source_segment_ids=list(map(int, segment_ids[i - 1:i + 1])),
                               actual_section_offset_bound_m=bound))
            continue
        runs.append((originals[first][0], originals[g - 1][1]))
        first = g
    runs.append((originals[first][0], originals[-1][1]))
    if joint == 'mitre_wedge':
        union, bridges, left_sides, right_sides, nodes = _mitre_wedge_union(
            runs, frame, sections, points, inward, width, differential, lengths, segment_ids,
            original_id, sources, tolerance)
        checks['joint'] = 'mitre_wedge'
        checks['joint_counts'] = {kind: sum(node['kind'] == kind for node in nodes) for kind in ('mitre', 'wedge', 'hull')}
        checks['joint_nodes'] = nodes
        return _finish_segment_union(union, sources, checks, merged, bridges, len(runs), len(points) - 1,
                                     tolerance, simplify_m, axes, origin,
                                     [x @ axes + origin for x in left_sides], [x @ axes + origin for x in right_sides])
    for run_start, run_end in runs:
        tangent, lateral, inside = frame(run_start, run_end)
        if np.min(inward[run_start:run_end + 1] @ inside) <= 0:
            raise ValueError('segment inward disagrees with a supplied endpoint frame')
        run_origin = points[run_start]
        left, right, ring = sections(slice(run_start, run_end + 1), run_origin, lateral, inside)
        cell, records = _cell(ring, original_id, len(sources), segment_ids[run_start:run_end], station_axis=tangent)
        solids.append(cell.translate(run_origin))
        sources.update(records)
        for j in range(run_end - run_start):
            span_rings[run_start + j] = ring[j:j + 2] + run_origin
            left_sides.append((left[j:j + 2] + run_origin) @ axes + origin)
            right_sides.append((right[j:j + 2] + run_origin) @ axes + origin)
    bridges, bridge_solids = [], {}
    for r in range(1, len(runs)):
        i = runs[r][0]
        pair = list(map(int, segment_ids[i - 1:i + 1]))
        previous, following = span_rings[i - 1], span_rings[i]
        # Convex bridge on both node sections and two short collars ON the
        # existing source segments, no endpoint extension. Collars are bounded
        # by 1/4 opening and 1/10 segment.
        collar = np.minimum(.25 * np.min(width[i] + 2 * differential[i]),
                            .1 * lengths[i - 1:i + 1])
        if np.any(collar <= 0):
            raise ValueError('internal junction must have positive opening')
        before = previous[1] + collar[0] / lengths[i - 1] * (previous[0] - previous[1])
        after = following[0] + collar[1] / lengths[i] * (following[1] - following[0])
        hull = md.Manifold.hull_points(np.vstack((before, previous[1], following[0], after)) - points[i])
        mesh = hull.to_mesh64()
        faces = np.asarray(mesh.tri_verts, dtype=np.uint64)
        if hull.status() != md.Error.NoError or hull.volume() <= 0 or not len(faces):
            raise ValueError('junction bridge has no finite volume')
        face_ids = np.arange(len(sources), len(sources) + len(faces), dtype=np.uint64)
        # to_mesh64 exposes read-only native buffers; the input constructor
        # requires owned writable arrays even when dtype and strides match.
        bridge = md.Manifold(md.Mesh64(np.array(mesh.vert_properties[:, :3], dtype=np.float64, copy=True),
            np.array(faces, dtype=np.uint64, copy=True), run_index=np.array([0, faces.size], dtype=np.uint64),
            run_original_id=np.array([original_id], dtype=np.uint32), face_id=face_ids))
        overlaps = [(bridge ^ solids[j].translate(-points[i])).volume() for j in (r - 1, r)]
        if bridge.status() != md.Error.NoError or min(overlaps) <= 0:
            raise ValueError('junction bridge must overlap both segment volumes')
        sources.update({int(fid): dict(source_segment_id=pair[0], adjacent_source_segment_ids=pair,
            station_index=i, role='junction_bridge') for fid in face_ids})
        bridges.append(dict(station_index=i, adjacent_source_segment_ids=pair,
            collar_length_m=collar.tolist(), volume_m3=bridge.volume(), overlap_volume_m3=overlaps))
        bridge_solids[r] = bridge.translate(points[i])
    # Contiguous blocks keep near-coplanar junctions together. Balanced merges
    # limit intermediate complexity and re-centre each actual Boolean.
    ordered = []
    for r, solid in enumerate(solids):
        if r in bridge_solids:
            ordered.append(bridge_solids[r])
        ordered.append(solid)
    groups = [_local_union(ordered[i:i + 16]) for i in range(0, len(ordered), 16)]
    while len(groups) > 1:
        groups = [_local_union(groups[i:i + 2]) for i in range(0, len(groups), 2)]
    union = groups[0]
    return _finish_segment_union(union, sources, checks, merged, bridges, len(solids), len(points) - 1,
                                 tolerance, simplify_m, axes, origin, left_sides, right_sides)


def _finish_segment_union(union, sources, checks, merged, bridges, cell_count, span_count,
                          tolerance, simplify_m, axes, origin, left_sides, right_sides):
    checks['shared_frame_original_nodes'] = merged
    checks['node_join_count'] = len(bridges)
    checks['original_segment_mesh_count'] = cell_count
    checks['refined_span_count'] = span_count
    if simplify_m is not None:
        union, checks['cutter_simplify'] = _simplify_cutter(union, sources, simplify_m)
    union, components, sheets = _drop_bridge_sheets(union, sources, tolerance, caps=checks.get('joint') == 'mitre_wedge')
    if sheets:
        checks['discarded_zero_thickness_bridge_sheets'] = sheets
    if union.status() != md.Error.NoError or union.volume() <= 0 or len(components) != 1:
        details = dict(status=str(union.status()), volume_m3=union.volume(),
            component_count=len(components), components=[dict(status=str(part.status()),
                volume_m3=part.volume(), bounding_box=part.bounding_box(), triangles=part.num_tri(),
                face_sources=[dict(face_id=int(f), **sources.get(int(f), {}))
                              for f in np.unique(np.asarray(part.to_mesh64().face_id))] if part.num_tri() <= 16 else None)
                for part in components])
        raise ValueError('segment union must be one connected valid cutter volume: ' + json.dumps(details))
    transform = np.column_stack((axes.T, origin))
    return union.transform(transform), sources, _owned(left_sides), _owned(right_sides), tuple(bridges), checks


def build_network_cutter(points, inward, normal, width, depth, common, differential, *,
                         outside_margin_m, original_id, source_segment_ids=None,
                         mode='strict_loft', simplify_m=None, joint='hull'):
    """Return one fixed-path cutter and source identity for its original faces.

    points, inward and normal are (N,3); inward and normal must be orthonormal.
    All four fields are (N,D), D>=2. ``depth`` contains actual nonnegative
    distances, not fractions. Width is full nominal width; the two walls are
    offset by ``common +/- (width/2 + differential)``. The first depth station
    is the mouth, whose depth/common/differential are exactly zero. Only entire
    first/last sections may have zero width and depth, after caller tip checks.

    original_id must be reserved by Manifold.reserve_ids; face_sources is keyed
    by Mesh64.face_id. The caller must verify inward against its actual host:
    this cutter-only function cannot distinguish an inward frame from its
    globally reversed outward counterpart.

    strict_loft uses the supplied station frames unchanged and rejects folds.
    segment_union fixes one orthogonal frame on each ORIGINAL segment, using
    its tangent and supplied lateral normal, with inward sign from the caller.
    Consecutive original segments share one frame only where that moves every
    actual section vertex by <= 1e-7 m from each segment's own frame (recorded
    in construction_checks['shared_frame_original_nodes']; span IDs kept).
    Its refined sections share mesh vertices; only run endpoints are capped. Bounded two/three-section checks and ordered longitudinal slabs
    validate these longer lofts without a quadratic whole-loft triangle scan.
    Adjacent different frames get a convex node bridge with positive overlap
    into both segments. The same externally computed fields are used without
    rephasing. In this mode left_wall/right_wall retain the nominal caller-frame
    sections for reference; segment_left_wall/segment_right_wall are the actual
    (N-1,2,D,3) segment walls, and junction_bridges records the extra join volume.
    """
    if mode not in ('strict_loft', 'segment_union'):
        raise ValueError('mode must be strict_loft or segment_union')
    # component gate (construction_checks['cutter_simplify']); never applied to a host.
    if simplify_m is not None and mode != 'segment_union':
        raise ValueError('cutter simplification is only defined for segment_union')
    # convex node bridge by a shared mitre section or a gap-side wedge per node; 'hull' is the original.
    if joint not in ('hull', 'mitre_wedge') or (joint != 'hull' and mode != 'segment_union'):
        raise ValueError("joint must be 'hull', or 'mitre_wedge' with segment_union")
    points, inward, normal = (_owned(value) for value in (points, inward, normal))
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('points must be finite (N,3), N>=2')
    n = len(points)
    if any(value.shape != (n, 3) or not np.isfinite(value).all() for value in (inward, normal)):
        raise ValueError('frame arrays must be finite (N,3)')
    if (not np.allclose(np.linalg.norm(inward, axis=1), 1., rtol=0, atol=1e-8)
            or not np.allclose(np.linalg.norm(normal, axis=1), 1., rtol=0, atol=1e-8)
            or np.any(np.abs(np.einsum('ij,ij->i', inward, normal)) > 1e-8)):
        raise ValueError('frame must have orthonormal inward and normal')
    if np.any(np.einsum('ij,ij->i', inward[:-1], inward[1:]) <= 0) or np.any(np.einsum('ij,ij->i', normal[:-1], normal[1:]) <= 0):
        raise ValueError('frame has an abrupt reversal')
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0):
        raise ValueError('path has zero length')
    if n > 2:
        turn_dot = np.einsum('ij,ij->i', edges[:-1], edges[1:])
        cross_length = np.linalg.norm(np.cross(edges[:-1], edges[1:]), axis=1)
        true_reversal = (turn_dot < 0) & (cross_length <= 32 * np.finfo(float).eps * lengths[:-1] * lengths[1:])
        # The strict loft's acute-turn sufficient condition does not constrain
        # separate valid segment solids. Preserve nonzero segments and actual
        # reversal rejection, then test the cells, overlaps and union directly.
        if np.any(true_reversal) or (mode == 'strict_loft' and np.any(turn_dot <= 0)):
            raise ValueError('path has an abrupt reversal')
    transverse = np.cross(normal, inward)
    advance = np.concatenate((np.einsum('ij,ij->i', edges, transverse[:-1]),
                              np.einsum('ij,ij->i', edges, transverse[1:])))
    if not (np.all(advance > 0) or np.all(advance < 0)):
        raise ValueError('frame does not advance consistently along the source path')
    width, depth, common, differential = (_owned(value) for value in (width, depth, common, differential))
    if width.ndim != 2 or width.shape[0] != n or width.shape[1] < 2:
        raise ValueError('width must have shape (N,D), D>=2')
    if any(value.shape != width.shape or not np.isfinite(value).all() for value in (width, depth, common, differential)):
        raise ValueError('width/depth/common/differential must be aligned finite (N,D) fields')
    if np.any(width < 0):
        raise ValueError('width must be nonnegative')
    if np.any(depth < 0) or np.any(depth[:, 0] != 0):
        raise ValueError('depth must start at zero and be nonnegative')
    if np.any(common[:, 0] != 0) or np.any(differential[:, 0] != 0):
        raise ValueError('mouth common and differential must be zero')
    tips = np.all(width == 0, axis=1) & np.all(depth == 0, axis=1)
    if np.any(tips[1:-1]) or np.any(common[tips] != 0) or np.any(differential[tips] != 0):
        raise ValueError('only caller-classified endpoint tips may collapse')
    if np.any(np.diff(depth[~tips], axis=1) <= 0):
        raise ValueError('depth must strictly increase inside non-tip sections')
    if np.any(width[~tips] <= 0) or np.any((width + 2 * differential)[~tips] <= 0):
        raise ValueError('width and actual wall gap must remain positive')
    if not np.isscalar(outside_margin_m) or not np.isfinite(outside_margin_m) or outside_margin_m <= 0:
        raise ValueError('outside margin must be finite and positive')
    if isinstance(original_id, (bool, np.bool_)) or not isinstance(original_id, (int, np.integer)) or not 0 <= original_id <= np.iinfo(np.uint32).max:
        raise ValueError('original_id must be a reserved uint32 ID')
    segment_ids = np.arange(n - 1) if source_segment_ids is None else np.asarray(source_segment_ids)
    if segment_ids.shape != (n - 1,) or segment_ids.dtype.kind not in 'iu':
        raise ValueError('source_segment_ids must be N-1 integer identities')
    base = points[:, None, :] + depth[..., None] * inward[:, None, :]
    left = base + (common - width / 2 - differential)[..., None] * normal[:, None, :]
    right = base + (common + width / 2 + differential)[..., None] * normal[:, None, :]
    if mode == 'segment_union':
        union, sources, segment_left, segment_right, bridges, checks = _segment_union(
            points, inward, normal, width, depth, common, differential,
            outside_margin_m, int(original_id), segment_ids, simplify_m, joint)
        return NetworkCutter(union, _owned(left), _owned(right), points, int(original_id), sources,
                             mode, segment_left, segment_right, bridges, checks)
    outside_left = left[:, 0] - outside_margin_m * inward
    outside_right = right[:, 0] - outside_margin_m * inward
    rings = np.concatenate((outside_left[:, None], left, right[:, ::-1], outside_right[:, None]), axis=1)
    solids, sources = [], {}
    for index in range(n - 1):
        solid, records = _cell(rings[index:index + 2], int(original_id), len(sources), segment_ids[index])
        solids.append(solid)
        sources.update(records)
    union = md.Manifold.batch_boolean(solids, md.OpType.Add)
    if union.status() != md.Error.NoError or union.volume() <= 0:
        raise ValueError('cutter union failed: ' + str(union.status()))
    return NetworkCutter(union, _owned(left), _owned(right), points, int(original_id), sources)


def union_cutters(cutters):
    """Return the Boolean union of paths, removing caps inside T/X junctions."""
    cutters = list(cutters)
    if not cutters:
        raise ValueError('at least one cutter is required')
    if len({c.original_id for c in cutters}) != len(cutters):
        raise ValueError('each cutter needs a distinct reserved original_id')
    solid = md.Manifold.batch_boolean([c.solid for c in cutters], md.OpType.Add)
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError('network union failed: ' + str(solid.status()))
    return solid


def clip_cutter_by_halfspaces(cutter, planes, *, extent_m=100.):
    """Intersect a built cutter with caller-supplied exact half-spaces (g, c, source_segment_id): g.p <= c.

    Used to keep a cavity inside the finite support of its source interface (e.g. beyond a real free edge).
    Each half-space is a large box whose only face near the cutter is the plane g.p = c; its faces carry the
    cutter's own original_id and new face IDs with role 'finite_support_bound', so provenance stays complete.
    The source path and all other faces are unchanged; the result must remain one valid positive volume."""
    import dataclasses
    solid, sources, clips = cutter.solid, dict(cutter.face_sources), []
    for g, c, segment_id in planes:
        g = np.asarray(g, float)
        norm = np.linalg.norm(g)
        if g.shape != (3,) or not np.isfinite(g).all() or norm == 0 or not np.isfinite(c):
            raise ValueError('half-space needs a finite nonzero normal and offset')
        n, c = g / norm, float(c) / norm
        helper = np.array([1., 0., 0.]) if abs(n[0]) < .9 else np.array([0., 1., 0.])
        e1 = np.cross(n, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        box = md.Manifold.cube([2 * extent_m, 2 * extent_m, extent_m]).translate([-extent_m, -extent_m, -extent_m])
        mesh = box.to_mesh64()
        local = np.asarray(mesh.vert_properties)[:, :3]
        # Box anchored on the plane at the cutter's bounding-box centre projection: its z=0 face IS the plane.
        bb = np.asarray(solid.bounding_box())
        centre = (bb[:3] + bb[3:]) / 2
        anchor = centre - (n @ centre - c) * n
        world = anchor + local @ np.stack((e1, e2, n))
        faces = np.array(mesh.tri_verts, dtype=np.uint64, copy=True)
        first = max(sources) + 1
        face_ids = np.arange(first, first + len(faces), dtype=np.uint64)
        half = md.Manifold(md.Mesh64(np.ascontiguousarray(world), faces,
            run_index=np.array([0, faces.size], dtype=np.uint64),
            run_original_id=np.array([cutter.original_id], dtype=np.uint32), face_id=face_ids))
        if half.status() != md.Error.NoError or half.volume() <= 0:
            raise ValueError('finite-support half-space is not a valid volume: ' + str(half.status()))
        before = solid.volume()
        solid = solid ^ half
        if solid.status() != md.Error.NoError or solid.volume() <= 0:
            raise ValueError('finite-support clip left no valid cutter volume: ' + str(solid.status()))
        sources.update({int(f): dict(source_segment_id=int(segment_id), role='finite_support_bound') for f in face_ids})
        clips.append(dict(normal=n.tolist(), offset_m=c, end_source_segment_id=int(segment_id),
                          removed_volume_m3=before - solid.volume()))
    parts = solid.decompose()
    if len(parts) != 1:
        raise ValueError('finite-support clip split the cutter: ' + json.dumps(
            [dict(volume_m3=p.volume(), triangles=p.num_tri()) for p in parts]))
    checks = dict(cutter.construction_checks or {}, finite_support_clips=clips)
    return dataclasses.replace(cutter, solid=solid, face_sources=sources, construction_checks=checks)
