"""Explicit finite 2.5D fracture growth and observable 2D network statistics.

The sequential, long-first growth, neighbor attraction, and intersection termination are inspired by Bonneau et al.
(2013), doi:10.1016/j.cageo.2013.02.004. This is a heuristic geometry generator, not a stress solver or a reproduction
of that implementation. Parameters (including the convenience defaults) are analyst choices, not calibrated
natural-rock distributions.

All observation coordinates and lengths are metres. Directions are axial: 0 degrees is +u, 90 is +v, and 180 is
equivalent to 0. The observation/growth window is ``[[u_min, v_min], [u_max, v_max]]``. Seeds are uniform in this
rectangle, with either a fixed count or a Poisson count of mean density*area. Both ends of each seed grow up to half
its sampled lognormal target length. Only already grown, longer fractures influence a later fracture.

The finite-strip conversion extrudes the entire 2D trace linearly by an explicit height. It makes no inference about
fracture height, 3D size, aperture, or volumetric voids. Metrics are window observations; boundary-censored lengths
are kept separate, following the measurement concerns discussed by Casiraghi et al."""
from __future__ import annotations

import math

import numpy as np


def _bounds(value):
    b = np.asarray(value, dtype=float)
    if b.shape != (2, 2) or not np.isfinite(b).all() or np.any(b[1] <= b[0]):
        raise ValueError('bounds_2d must be finite [[u_min,v_min],[u_max,v_max]] with positive area')
    return b


def _number(value, name, *, positive=False, probability=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite number')
    x = float(value)
    if not math.isfinite(x) or (x <= 0 if positive else x < 0) or (probability and x > 1):
        raise ValueError(name + (' must be in [0,1]' if probability else ' is outside its valid range'))
    return x


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _intersections(a, b, c, d, tolerance):
    """Parameters on two closed segments, including collinear overlap limits."""
    r, s = b - a, d - c
    lr, ls = float(np.linalg.norm(r)), float(np.linalg.norm(s))
    if lr <= tolerance or ls <= tolerance:
        return []
    denom = _cross(r, s)
    delta = c - a
    et, eu = tolerance / lr, tolerance / ls
    if abs(denom) > 1e-12 * lr * ls:
        t, u = _cross(delta, s) / denom, _cross(delta, r) / denom
        if -et <= t <= 1 + et and -eu <= u <= 1 + eu:
            return [(float(np.clip(t, 0, 1)), float(np.clip(u, 0, 1)))]
        return []
    if abs(_cross(delta, r)) > tolerance * lr:
        return []
    rr, ss = float(r @ r), float(s @ s)
    lo, hi = sorted((float((c - a) @ r / rr), float((d - a) @ r / rr)))
    lo, hi = max(0., lo), min(1., hi)
    if lo > hi + et:
        return []
    ts = [float(np.clip(lo, 0, 1))]
    if hi - lo > et:
        ts.append(float(np.clip(hi, 0, 1)))
    return [(t, float(np.clip((a + t * r - c) @ s / ss, 0, 1))) for t in ts]


def _boundary(point, bounds, tolerance):
    return bool(np.any(np.abs(point - bounds[0]) <= tolerance)
                or np.any(np.abs(point - bounds[1]) <= tolerance))


def _neighbor_direction(point, direction, segments, radius, weight, tolerance):
    if not segments or radius == 0 or weight == 0:
        return direction
    influence = np.zeros(2)
    for a, b in segments:
        edge = b - a
        t = float(np.clip((point - a) @ edge / (edge @ edge), 0, 1))
        delta = a + t * edge - point
        distance = float(np.linalg.norm(delta))
        if tolerance < distance <= radius and delta @ direction > tolerance:
            # Inverse-distance attraction toward actual closest points ahead.
            influence += delta / (distance * distance)
    norm = float(np.linalg.norm(influence))
    if norm <= tolerance:
        return direction
    out = (1 - weight) * direction + weight * influence / norm
    return out / np.linalg.norm(out)


def _grow_side(seed, direction, length, segments, bounds, rng, *, step,
               radius, weight, stop_probability, tolerance):
    points = [seed.copy()]
    remaining = length
    termination = 'target'
    while remaining > tolerance:
        direction = _neighbor_direction(points[-1], direction, segments, radius, weight, tolerance)
        travel = min(step, remaining)
        start = points[-1]
        edge_fraction = 1.
        for axis in range(2):
            if direction[axis] > 0:
                edge_fraction = min(edge_fraction, (bounds[1, axis] - start[axis]) / (travel * direction[axis]))
            elif direction[axis] < 0:
                edge_fraction = min(edge_fraction, (bounds[0, axis] - start[axis]) / (travel * direction[axis]))
        end = np.clip(start + max(0., edge_fraction) * travel * direction, bounds[0], bounds[1])
        stop_at = None
        hits = []
        for a, b in segments:
            for t, _ in _intersections(start, end, a, b, tolerance):
                if t * np.linalg.norm(end - start) > tolerance:
                    hits.append(t)
        # One trial per contact point, irrespective of the older trace's subdivision.
        previous = -math.inf
        for t in sorted(hits):
            if (t - previous) * np.linalg.norm(end - start) <= tolerance:
                continue
            previous = t
            if stop_probability == 1 or (stop_probability > 0 and rng.random() < stop_probability):
                stop_at = t
                break
        if stop_at is not None:
            end = start + stop_at * (end - start)
            termination = 'intersection'
        distance = float(np.linalg.norm(end - start))
        if distance > tolerance:
            points.append(end)
            remaining = max(0., remaining - distance)
        if _boundary(end, bounds, tolerance):
            termination = 'boundary'
        if termination != 'target' or distance <= tolerance:
            break
    return points, termination


def validate_direction_groups(value, count):
    """Validate fixed trace-index quotas for one shared growth network."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError('direction_groups must be a nonempty list')
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError('direction_groups require a fixed integer count')
    groups, identifiers = [], set()
    keys = {'id', 'count', 'mean_direction_deg', 'direction_std_deg'}
    for row in value:
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError('direction_groups require id, count, mean_direction_deg and direction_std_deg')
        identifier, quota = row['id'], row['count']
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError('direction_group id must be nonempty and unique')
        if isinstance(quota, (bool, np.bool_)) or not isinstance(quota, (int, np.integer)) or quota <= 0:
            raise ValueError('direction_group count must be a positive integer')
        numbers = [row['mean_direction_deg'], row['direction_std_deg']]
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating))
               or not math.isfinite(x) for x in numbers) or numbers[1] < 0:
            raise ValueError('direction_group angles must be finite numbers with nonnegative standard deviation')
        identifiers.add(identifier)
        groups.append(dict(id=identifier, count=int(quota), mean_direction_deg=float(numbers[0]),
                           direction_std_deg=float(numbers[1])))
    if sum(row['count'] for row in groups) != count:
        raise ValueError('direction_group counts must sum to the network count')
    return groups


def sample_network(seed, bounds_2d, *, count=None, density_m2=None,
                   mean_direction_deg=0., direction_std_deg=15.,
                   length_median_m=5., length_log_sigma=.5, step_m=.25,
                   interaction_radius_m=1., direction_weight=.25,
                   stop_probability=1., direction_groups=None):
    """Sample a serializable finite network; supply exactly one of count/density.

    ``direction_std_deg`` is the standard deviation of a wrapped Gaussian axial
    orientation. ``length_log_sigma`` is the dimensionless standard deviation of
    log(length/metre). ``direction_weight`` blends current tip direction with
    inverse-distance attraction to already grown traces within the stated radius
    and ahead of the tip. Crossing/stop trials occur only at actual intersections.
    Returned traces are in decreasing target-length growth order. Endpoints and
    termination arrays follow polyline order (negative side, positive side).
    Optional direction groups allocate fixed original trace-index quotas before
    the same long-first growth loop; all groups interact with previous traces.
    """
    bounds = _bounds(bounds_2d)
    if (count is None) == (density_m2 is None):
        raise ValueError('supply exactly one of count or density_m2')
    if count is not None and (isinstance(count, (bool, np.bool_))
                              or not isinstance(count, (int, np.integer)) or count < 0):
        raise ValueError('count must be a nonnegative integer')
    density = None if density_m2 is None else _number(density_m2, 'density_m2')
    mean = float(mean_direction_deg)
    if not math.isfinite(mean):
        raise ValueError('mean_direction_deg must be finite')
    scatter = _number(direction_std_deg, 'direction_std_deg')
    median = _number(length_median_m, 'length_median_m', positive=True)
    sigma = _number(length_log_sigma, 'length_log_sigma')
    step = _number(step_m, 'step_m', positive=True)
    radius = _number(interaction_radius_m, 'interaction_radius_m')
    weight = _number(direction_weight, 'direction_weight', probability=True)
    stop = _number(stop_probability, 'stop_probability', probability=True)
    groups = validate_direction_groups(direction_groups, count)
    rng = np.random.default_rng(seed)
    area = float(np.prod(bounds[1] - bounds[0]))
    n = int(count) if count is not None else int(rng.poisson(density * area))
    seeds = rng.uniform(bounds[0], bounds[1], (n, 2))
    if groups is None:
        angles = rng.normal(mean, scatter, n) % 180.
    else:
        means = np.concatenate([np.full(row['count'], row['mean_direction_deg']) for row in groups])
        scatters = np.concatenate([np.full(row['count'], row['direction_std_deg']) for row in groups])
        group_ids = [row['id'] for row in groups for _ in range(row['count'])]
        angles = rng.normal(means, scatters, n) % 180.
    targets = rng.lognormal(math.log(median), sigma, n)
    if not np.isfinite(targets).all() or np.any(targets <= 0):
        raise ValueError('sampled target lengths must be finite and positive')
    tolerance = max(1e-10, float(np.linalg.norm(bounds[1] - bounds[0])) * 1e-12)
    traces, segments = [], []
    for index in sorted(range(n), key=lambda i: (-targets[i], i)):
        angle = math.radians(float(angles[index]))
        direction = np.array([math.cos(angle), math.sin(angle)])
        sides, terminations = [], []
        for sign in (-1., 1.):
            points, term = _grow_side(seeds[index], sign * direction, float(targets[index]) / 2,
                                      segments, bounds, rng, step=step, radius=radius,
                                      weight=weight, stop_probability=stop, tolerance=tolerance)
            sides.append(points)
            terminations.append(term)
        points = np.array(list(reversed(sides[0])) + sides[1][1:])
        realized = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if len(points) < 2:
            raise ValueError('growth step or target length is below coordinate resolution')
        traces.append({'id': f'trace-{index:04d}', 'seed_2d_m': seeds[index].tolist(),
                       'initial_direction_deg': float(angles[index]),
                       'target_length_m': float(targets[index]),
                       'realized_length_m': realized, 'points_2d_m': points.tolist(),
                       'endpoint_termination': terminations,
                       'endpoint_boundary_censored': [_boundary(points[0], bounds, tolerance),
                                                      _boundary(points[-1], bounds, tolerance)]})
        if groups is not None:
            traces[-1]['direction_group_id'] = group_ids[index]
        segments.extend((a, b) for a, b in zip(points[:-1], points[1:]))
    network = {'method': 'sequential_pseudo_genetic_2_5d', 'seed': int(seed),
               'bounds_2d_m': bounds.tolist(),
               'parameters': {'count': None if count is None else int(count), 'density_m2': density,
                              'mean_direction_deg': mean % 180., 'direction_std_deg': scatter,
                              'length_median_m': median, 'length_log_sigma': sigma, 'step_m': step,
                              'interaction_radius_m': radius, 'direction_weight': weight,
                              'stop_probability': stop}, 'traces': traces}
    if groups is not None:
        network['parameters'].update(mean_direction_deg=None, direction_std_deg=None,
                                     direction_groups=groups)
    network['metrics'] = network_metrics(network)
    return network


def _observed_traces(network):
    bounds = _bounds(network['bounds_2d_m'])
    points = []
    for trace in network['traces']:
        p = np.asarray(trace['points_2d_m'], dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError('each trace must have at least two finite 2D points')
        if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) == 0):
            raise ValueError('trace segments must have positive length')
        if np.any(p < bounds[0]) or np.any(p > bounds[1]):
            raise ValueError('trace points must lie in the declared observation window')
        points.append(p)
    return bounds, points


def _topology(polylines, bounds, tolerance):
    segments = [(a, b) for p in polylines for a, b in zip(p[:-1], p[1:])]
    splits = [[0., 1.] for _ in segments]
    for i, (a, b) in enumerate(segments):
        for j in range(i):
            c, d = segments[j]
            if np.any(np.maximum(a, b) + tolerance < np.minimum(c, d)) or np.any(np.maximum(c, d) + tolerance < np.minimum(a, b)):
                continue
            for t, u in _intersections(a, b, c, d, tolerance):
                splits[i].append(t)
                splits[j].append(u)
    nodes, bins, adjacency = [], {}, []

    def node(point):
        key = tuple(np.floor(point / tolerance).astype(np.int64))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in bins.get((key[0] + dx, key[1] + dy), ()):
                    if np.linalg.norm(nodes[index] - point) <= tolerance:
                        return index
        index = len(nodes)
        nodes.append(point)
        adjacency.append(set())
        bins.setdefault(key, []).append(index)
        return index

    for (a, b), ts in zip(segments, splits):
        ids = [node(a + t * (b - a)) for t in sorted(ts)]
        for first, second in zip(ids[:-1], ids[1:]):
            if first != second:
                adjacency[first].add(second)
                adjacency[second].add(first)
    counts = dict(I=0, Y=0, X=0, B=0, other=0)
    records = []
    for index, (point, adjacent) in enumerate(zip(nodes, adjacency)):
        degree = len(adjacent)
        kind = 'B' if _boundary(point, bounds, tolerance) else {1: 'I', 2: None, 3: 'Y', 4: 'X'}.get(degree, 'other')
        if kind is not None:
            counts[kind] += 1
            records.append({'point_2d_m': point.tolist(), 'degree': degree, 'kind': kind})
    seen, components = set(), 0
    for index in range(len(nodes)):
        if index in seen:
            continue
        components += 1
        todo = [index]
        while todo:
            current = todo.pop()
            if current not in seen:
                seen.add(current)
                todo.extend(adjacency[current] - seen)
    return {'node_counts': counts, 'nodes': records, 'connected_components': components,
            'node_tolerance_m': tolerance}


def _length_statistics(values):
    return {'count': len(values), 'mean': float(np.mean(values)) if values else None,
            'median': float(np.median(values)) if values else None,
            'min': float(min(values)) if values else None, 'max': float(max(values)) if values else None}


def network_metrics(network, *, node_tolerance_m=1e-7):
    """Recompute 2D observations, without estimating an unobserved 3D population.

    P21 is sum of observed trace lengths divided by rectangle area (1/m).
    Orientation is the segment-length-weighted double-angle axial mean and
    resultant concentration in [0,1]. I/Y/X are graph degrees 1/3/4; boundary
    contacts are B regardless of degree. Degree-2 polyline subdivision vertices
    are omitted. Intersections are split before graph assembly, and repeated
    geometric edges are merged for topology. Higher degrees are reported as
    ``other`` rather than interpreted as X. No connectivity index is inferred.
    """
    bounds, polylines = _observed_traces(network)
    tolerance = _number(node_tolerance_m, 'node_tolerance_m', positive=True)
    total, c, s = 0., 0., 0.
    lengths, targets, censored, uncensored = [], [], [], []
    for trace, points in zip(network['traces'], polylines):
        edges = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(edges, axis=1)
        length = float(segment_lengths.sum())
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        c += float(np.sum(segment_lengths * np.cos(2 * angles)))
        s += float(np.sum(segment_lengths * np.sin(2 * angles)))
        total += length
        lengths.append(length)
        if 'target_length_m' in trace:
            targets.append(_number(trace['target_length_m'], 'target_length_m', positive=True))
        is_censored = _boundary(points[0], bounds, tolerance) or _boundary(points[-1], bounds, tolerance)
        (censored if is_censored else uncensored).append(length)
    area = float(np.prod(bounds[1] - bounds[0]))
    concentration = min(1., math.hypot(c, s) / total) if total else 0.
    mean = (math.degrees(math.atan2(s, c)) / 2) % 180. if concentration > 1e-12 else None
    if mean is not None and (mean < 1e-10 or 180. - mean < 1e-10):
        mean = 0.
    return {'area_m2': area, 'trace_count': len(polylines), 'total_observed_length_m': total,
            'p21_per_m': total / area, 'axial_mean_direction_deg': mean,
            'axial_concentration': concentration,
            'length_statistics_m': {'target': _length_statistics(targets),
                                    'observed': _length_statistics(lengths),
                                    'boundary_censored': _length_statistics(censored),
                                    'uncensored': _length_statistics(uncensored)},
            'topology': _topology(polylines, bounds, tolerance)}


def network_geometry(network, *, prefix, origin_m, axis_u, axis_v,
                     extrusion_direction, height_m):
    """Return (open mesh records, joint features) for finite centered strips.

    For each observed (u,v), the two strip edges are
    ``origin + u*axis_u + v*axis_v +/- height/2*unit(extrusion_direction)``.
    Growth axes must be orthonormal, so input metre lengths are preserved.
    Extrusion must be transverse to the growth plane; its finite height is an
    explicit analyst assumption. Feature hosts/support and creation events
    are assigned by the scene caller. No aperture or volumetric cutter is added.
    """
    _, polylines = _observed_traces(network)
    if not isinstance(prefix, str) or not prefix:
        raise ValueError('prefix must be a nonempty identifier')
    vectors = [np.asarray(v, dtype=float) for v in (origin_m, axis_u, axis_v, extrusion_direction)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError('origin and basis vectors must be finite 3D vectors')
    origin, u, v, extrusion = vectors
    if not np.isclose(np.linalg.norm(u), 1., atol=1e-9, rtol=0) or not np.isclose(np.linalg.norm(v), 1., atol=1e-9, rtol=0) or abs(float(u @ v)) > 1e-9:
        raise ValueError('axis_u and axis_v must be orthonormal')
    norm = float(np.linalg.norm(extrusion))
    if norm == 0 or abs(float(np.cross(u, v) @ (extrusion / norm))) <= 1e-9:
        raise ValueError('extrusion_direction must be nonzero and transverse to growth plane')
    height = _number(height_m, 'height_m', positive=True)
    extrusion = extrusion / norm * height / 2
    geometries, features = [], []
    for trace, points in zip(network['traces'], polylines):
        center = origin + points[:, :1] * u + points[:, 1:] * v
        vertices = np.stack((center - extrusion, center + extrusion), axis=1).reshape(-1, 3)
        faces = []
        for i in range(len(points) - 1):
            a, b, c, d = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
            faces.extend(([a, b, c], [a, c, d]))
        fid = prefix + '-' + str(trace['id'])
        gid = fid + '-surface'
        # Validate the actual placed coordinates: a positive requested height
        # can round to zero at a large origin. Reuse the kernel's exact fallback
        # for short edges rather than rejecting valid underflow-scale areas.
        from outcrop.modeling.kernel.mesh import MeshSurface
        MeshSurface(gid, vertices, np.asarray(faces, np.int64))
        geometries.append({'id': gid, 'kind': 'mesh', 'closed': False,
                           'vertices': vertices.tolist(), 'faces': faces})
        features.append({'id': fid, 'kind': 'joint', 'geometry': gid})
        if 'direction_group_id' in trace:
            features[-1]['metadata'] = {'direction_group_id': trace['direction_group_id']}
    return geometries, features
