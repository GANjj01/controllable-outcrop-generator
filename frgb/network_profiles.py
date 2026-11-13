"""Read-only main-UI path binding and five post-path morphology parameters.

No path growth, deformation, or cutter is performed here. Optional stations
are inserted only on unchanged original segments. XY source arclength belongs
to the growth footprint; it is not a 3D exposure length.
Actual exposure chains must supply their own arclength to evaluate_arc_profile.
"""
from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np


PARAMETER_BOUNDS = MappingProxyType({
    'mean_opening_m': (.03, .08), 'opening_variation': (0., .45),
    'depth_m': (.10, .25), 'roughness_m': (0., .006), 'tip_fraction': (.10, .25),
})
PARAMETER_NAMES = tuple(PARAMETER_BOUNDS)


@dataclass(frozen=True)
class ProfileParameters:
    mean_opening_m: float
    opening_variation: float
    depth_m: float
    roughness_m: float
    tip_fraction: float


def _validate_parameters(parameters):
    if not isinstance(parameters, ProfileParameters):
        raise TypeError('five ProfileParameters are required')
    for name, value in asdict(parameters).items():
        lo, hi = PARAMETER_BOUNDS[name]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{name} must be finite in [{lo}, {hi}]')


def genome_parameter_names(*, has_true_free_ends=True):
    """Only proven real free ends authorize an active tip gene."""
    if not isinstance(has_true_free_ends, (bool, np.bool_)):
        raise ValueError('has_true_free_ends must be an explicit boolean')
    return PARAMETER_NAMES if has_true_free_ends else PARAMETER_NAMES[:-1]


def genome_bounds(*, has_true_free_ends=True):
    return tuple(PARAMETER_BOUNDS[name]
                 for name in genome_parameter_names(has_true_free_ends=has_true_free_ends))


def decode_genome(values, *, has_true_free_ends=True, fixed_tip_fraction=.18):
    """Decode a bounded 5D/4D real vector; reject invalid values without clipping.

    An unproved/cropped-only path uses four genes. Its fixed tip_fraction is a
    stored inactive parameter and has no effect when free_ends=(False, False).
    Network coordinates, bend, seeds, and other growth controls have no slots.
    """
    names = genome_parameter_names(has_true_free_ends=has_true_free_ends)
    vector = np.asarray(values, dtype=object)
    if vector.ndim != 1 or vector.shape != (len(names),):
        raise ValueError(f'genome requires a {len(names)}-dimensional real vector')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in vector):
        raise ValueError('genome must be a real numeric vector')
    fields = dict(zip(names, vector.astype(float)))
    if not has_true_free_ends:
        fields['tip_fraction'] = fixed_tip_fraction
    parameters = ProfileParameters(**fields)
    _validate_parameters(parameters)
    return parameters


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=float)
    # Immutable bytes backing prevents callers from re-enabling WRITEABLE.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _arcs(value, total):
    arc = np.atleast_1d(np.asarray(value, dtype=float))
    if not np.isfinite(total) or total <= 0 or arc.ndim != 1 or not np.isfinite(arc).all():
        raise ValueError('arc and total_length_m must be finite with positive total length')
    if np.any(arc < 0) or np.any(arc > total):
        raise ValueError('arc lies outside the original path length')
    return arc


@dataclass(frozen=True)
class FrozenTrace:
    id: str
    points_2d_m: np.ndarray
    arclength_m: np.ndarray
    endpoint_termination: tuple
    endpoint_boundary_censored: tuple
    endpoint_degrees: tuple
    free_ends: tuple
    phase_rad: float

    @property
    def length_m(self):
        return float(self.arclength_m[-1])

    def points_at(self, s_m):
        """Interpolate on the unchanged source footprint, returning (N,2)."""
        arc = _arcs(s_m, self.length_m)
        index = np.clip(np.searchsorted(self.arclength_m, arc, side='right') - 1,
                        0, len(self.points_2d_m) - 2)
        fraction = (arc - self.arclength_m[index]) / np.diff(self.arclength_m)[index]
        result = self.points_2d_m[index] + fraction[:, None] * np.diff(self.points_2d_m, axis=0)[index]
        # Preserve stored vertices exactly, including the final endpoint.
        for i, station in enumerate(self.arclength_m):
            result[arc == station] = self.points_2d_m[i]
        return result

    def locate_2d(self, points):
        """Nearest original segment and its source arclength; ties use first segment."""
        queries = np.atleast_2d(np.asarray(points, dtype=float))
        if queries.ndim != 2 or queries.shape[1] != 2 or not np.isfinite(queries).all():
            raise ValueError('footprint query must contain finite 2D points')
        start, edge = self.points_2d_m[:-1], np.diff(self.points_2d_m, axis=0)
        fraction = np.clip(np.einsum('nsi,si->ns', queries[:, None] - start, edge)
                           / np.einsum('si,si->s', edge, edge), 0., 1.)
        projected = start + fraction[:, :, None] * edge
        distance = np.linalg.norm(queries[:, None] - projected, axis=2)
        index = np.argmin(distance, axis=1)
        rows = np.arange(len(queries))
        f = fraction[rows, index]
        return dict(point_2d_m=projected[rows, index],
                    s_m=self.arclength_m[index] + f * np.diff(self.arclength_m)[index],
                    distance_m=distance[rows, index], segment_index=index, fraction=f)


@dataclass(frozen=True)
class FrozenNetwork:
    geometry_directory: Path
    traces: tuple
    network_metadata: object
    topology_nodes: tuple
    node_counts: object
    connected_components: int
    actual_segments_3d_m: np.ndarray
    interface_ids: tuple
    source_trace_ids: tuple
    interface_trace_ids: object
    domain_z_m: tuple | None
    _source_bytes: tuple = field(repr=False)

    def trace(self, trace_id):
        for trace in self.traces:
            if trace.id == trace_id:
                return trace
        raise KeyError(trace_id)

    def growth_footprint_xy(self, points):
        """Inverse of the generator's growth-plane extrusion: 3D points on a strip -> source (N,2) footprint.

        apply_fracture_network places each strip at origin z = domain mid-height and offsets it along
        unit(extrusion_direction); for the horizontal ENU XY growth plane a strip point p therefore has
        source_xy = p[:2] - ((p.z - zref) / d.z) * d[:2]. Other growth planes are rejected, not approximated."""
        meta = self.network_metadata
        if meta.get('growth_plane') != 'horizontal ENU XY':
            raise ValueError('only the horizontal ENU XY growth plane has an implemented inverse')
        d = np.asarray(meta.get('extrusion_direction'), float)
        if d.shape != (3,) or not np.isfinite(d).all() or d[2] == 0:
            raise ValueError('extrusion direction must be finite and transverse to the XY growth plane')
        if self.domain_z_m is None:
            raise ValueError('source domain z range is required for the growth-plane origin')
        zref = .5 * (self.domain_z_m[0] + self.domain_z_m[1])
        q = np.atleast_2d(np.asarray(points, float))
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError('footprint inverse needs finite (N,3) points')
        return q[:, :2] - ((q[:, 2] - zref) / d[2])[:, None] * d[:2]

    def assert_source_unchanged(self):
        """Read only: compare the four input files to their loaded bytes."""
        for path, original in self._source_bytes:
            if not path.is_file() or path.read_bytes() != original:
                raise ValueError('source changed: ' + str(path))


def _checked_topology(network):
    from outcrop.dsgen.fracture_network import network_metrics
    try:
        saved = network['metrics']['topology']
        actual = network_metrics(network, node_tolerance_m=saved['node_tolerance_m'])['topology']
        if saved['node_counts'] != actual['node_counts'] or saved['connected_components'] != actual['connected_components']:
            raise ValueError('source topology statistics disagree with original polylines')
        remaining = list(actual['nodes'])
        tolerance = float(saved['node_tolerance_m'])
        for node in saved['nodes']:
            matches = [i for i, other in enumerate(remaining)
                       if np.linalg.norm(np.asarray(node['point_2d_m']) - other['point_2d_m']) <= tolerance
                       and node['degree'] == other['degree'] and node['kind'] == other['kind']]
            if len(matches) != 1:
                raise ValueError('source topology node disagrees with original polylines')
            remaining.pop(matches[0])
        if remaining:
            raise ValueError('source topology nodes are incomplete')
        return saved, actual, tolerance
    except (KeyError, TypeError) as exc:
        raise ValueError('source topology truth is incomplete') from exc


def load_frozen_network(geometry_directory):
    """Load current UI package fields without regenerating or changing inputs."""
    root = Path(geometry_directory)
    package = root / 'package'
    paths = [package / name for name in ('model_spec.json', 'traces.npz', 'traces.json', 'interfaces.json')]
    originals = tuple((path, path.read_bytes()) for path in paths)
    raw = json.loads(originals[0][1])
    try:
        network = raw['metadata']['paper_methods']['fracture_network']
    except (KeyError, TypeError) as exc:
        raise ValueError('main-UI source fracture_network is required') from exc
    saved, actual, tolerance = _checked_topology(network)
    bounds = np.asarray(network['bounds_2d_m'], float)
    traces, identifiers = [], set()
    for record in network['traces']:
        identifier = record['id']
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('source trace IDs must be unique nonempty strings')
        identifiers.add(identifier)
        points = np.asarray(record['points_2d_m'], float)
        termination = record.get('endpoint_termination')
        censored = record.get('endpoint_boundary_censored')
        if not isinstance(termination, list) or len(termination) != 2 or any(t not in ('target', 'intersection', 'boundary') for t in termination):
            raise ValueError('source endpoint termination truth is required')
        if not isinstance(censored, list) or len(censored) != 2 or any(type(c) is not bool for c in censored):
            raise ValueError('source endpoint boundary truth is required')
        degrees, free = [], []
        for point, term, flag in zip(points[[0, -1]], termination, censored):
            boundary = bool(np.any(np.abs(point - bounds[0]) <= tolerance)
                            or np.any(np.abs(point - bounds[1]) <= tolerance))
            if flag != boundary or (term == 'boundary') != boundary:
                raise ValueError('source endpoint boundary evidence conflicts')
            matches = [node for node in actual['nodes']
                       if np.linalg.norm(point - node['point_2d_m']) <= tolerance]
            if len(matches) != 1:
                raise ValueError('source endpoint topology degree is unresolved')
            node = matches[0]
            degrees.append(node['degree'])
            free.append(term == 'target' and not boundary and node['kind'] == 'I' and node['degree'] == 1)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        # Fixed trace identity phase only; no path/random seed enters morphology.
        phase = 2 * np.pi * (sum((i + 1) * b for i, b in enumerate(identifier.encode('utf8'))) % 997) / 997.
        traces.append(FrozenTrace(identifier, _readonly(points), _readonly(arc), tuple(termination),
                                  tuple(censored), tuple(degrees), tuple(free), float(phase)))
    declared = {f['id'] for f in raw.get('features', ()) if f['kind'] == 'joint'}
    features = {'network-joint-' + trace.id: trace.id for trace in traces}
    if not set(features) <= declared:
        raise ValueError('source trace lacks its declared network-joint feature')
    interfaces = json.loads(originals[3][1])
    mapping = {row['id']: features.get(row.get('feature_id')) for row in interfaces}
    interface_ids = tuple(json.loads(originals[2][1])['interface_ids'])
    if any(oid not in mapping for oid in interface_ids):
        raise ValueError('actual trace has unknown source interface')
    with np.load(BytesIO(originals[1][1]), allow_pickle=False) as arrays:
        segments = np.asarray(arrays['segments'], float)
    if segments.shape != (len(interface_ids), 2, 3) or not np.isfinite(segments).all():
        raise ValueError('actual trace segments and interface IDs must correspond')
    if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 0):
        raise ValueError('actual trace segments must have positive length')
    domain = raw.get('domain')
    domain_z = None
    if domain is not None:
        domain_z = (float(domain['min_m'][2]), float(domain['max_m'][2]))
        if not np.isfinite(domain_z).all() or domain_z[1] <= domain_z[0]:
            raise ValueError('source domain z range must be finite and increasing')
    return FrozenNetwork(root, tuple(traces), _freeze(network), _freeze(saved['nodes']),
                         _freeze(saved['node_counts']), saved['connected_components'],
                         _readonly(segments), interface_ids, tuple(mapping[oid] for oid in interface_ids),
                         MappingProxyType(mapping), domain_z, originals)


def end_edge_halfspace(network, trace, endpoint_index, *, tolerance_m=1e-7):
    """Exact 3D half-space g.p <= c bounding the finite source strip at one source trace end.

    The source strip is the XY polyline at z = zref extruded along unit d by +/- height/2; its end edge at source
    endpoint e is the line E + s*d, E = (e, zref). The end face is the plane containing that edge and the fracture
    normal of the end panel (T x d, T = outward XY tangent of the last source segment as a world vector), so its
    normal is g = T - (T.d)d / (d.d): points displaced along the fracture normal (finite opening) keep their side.
    The whole source strip (every lifted polyline vertex, both strip edges) must lie on the kept side within the
    source tolerance; a source polyline folding back past its own end face is rejected rather than approximated.
    Applies to any exposed end mapped to a source end (free, intersection or boundary termination)."""
    pts = np.asarray(trace.points_2d_m, float)
    end, inner = (pts[0], pts[1]) if endpoint_index == 0 else (pts[-1], pts[-2])
    meta = network.network_metadata
    network.growth_footprint_xy([[0., 0., 0.]])  # validates growth plane, direction and domain
    d = np.asarray(meta['extrusion_direction'], float)
    zref = .5 * (network.domain_z_m[0] + network.domain_z_m[1])
    T = np.r_[end - inner, 0.]
    T /= np.linalg.norm(T)
    g = T - (T @ d) / (d @ d) * d
    if np.linalg.norm(g) <= 1e-12:
        raise ValueError('end segment is parallel to the extrusion; end face undefined')
    g /= np.linalg.norm(g)
    E = np.r_[end, zref]
    c = float(g @ E)
    half = .5 * float(meta['height_m']) * d / np.linalg.norm(d)
    strip = np.vstack([np.c_[pts, np.full(len(pts), zref)] + sign * half for sign in (-1., 1.)])
    if np.max(strip @ g - c) > tolerance_m:
        raise ValueError('source strip crosses its own end face; no single half-space supports this end')
    return g, c


def classify_path_endpoints(network, trace, points, *, closed, tolerance_m=1e-7):
    """Real free-end evidence for the two ends of one exposed 3D path of ``trace``.

    Each end is mapped back through the generator's extrusion to the source footprint; it is a true tip only if
    that footprint coincides (<= the source position tolerance) with a source endpoint that is itself a proven
    free end (target, not boundary, I-node degree 1), and the path is open. Crops and junctions map elsewhere."""
    ends = np.asarray(points, float)[[0, -1]]
    footprint = network.growth_footprint_xy(ends)
    free, evidence = [], []
    for point, xy in zip(ends, footprint):
        d = np.linalg.norm(trace.points_2d_m[[0, -1]] - xy, axis=1)
        j = int(d.argmin())
        eligible = bool(d[j] <= tolerance_m and trace.free_ends[j] and not closed)
        free.append(eligible)
        evidence.append(dict(source_endpoint=j, footprint_xy_m=xy.tolist(), footprint_distance_m=float(d[j]),
                             source_endpoint_free=bool(trace.free_ends[j]), free=eligible,
                             mapping='inverse growth-plane extrusion (model_spec fracture_network, domain mid z)',
                             reason='confirmed source free end' if eligible else 'not a confirmed free end; no taper'))
    return free, evidence


def _taper(u, fraction, free_ends):
    value = np.ones_like(u)
    for enabled, distance in zip(free_ends, (u, 1. - u)):
        if enabled:
            t = np.clip(distance / fraction, 0., 1.)
            value *= t * t * (3. - 2. * t)
    return value


def evaluate_arc_profile(parameters, arc_m, total_length_m, *, free_ends=(False, False),
                         phase=0., depth_fraction=(0.,)):
    """Profile fields on a supplied path's own arc; cropped ends default open.

    mean_opening_m is the full-path arclength mean, including eligible tapered
    ends. Fields have N stations and wall offsets have (N,D) depth samples.
    Caller supplies the unchanged 2D or 3D path and its transverse/depth frame.
    """
    _validate_parameters(parameters)
    if len(free_ends) != 2 or any(not isinstance(v, (bool, np.bool_)) for v in free_ends):
        raise ValueError('free_ends requires two resolved booleans')
    if not np.isscalar(phase) or not np.isfinite(phase):
        raise ValueError('phase must be finite')
    arc = _arcs(arc_m, total_length_m)
    depths = np.atleast_1d(np.asarray(depth_fraction, float))
    if depths.ndim != 1 or not np.isfinite(depths).all() or np.any(depths < 0) or np.any(depths > 1):
        raise ValueError('depth_fraction must lie in [0,1]')
    p, u = parameters, arc / total_length_m
    taper = _taper(u, p.tip_fraction, free_ends)
    # Integrate the scalar width field, independent of the caller's station
    # spacing. These quadrature nodes never replace or resample a source path.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    breaks = [0., p.tip_fraction, 1. - p.tip_fraction, 1.]
    mean_raw = 0.
    for a, b in zip(breaks[:-1], breaks[1:]):
        q = a + (nodes + 1.) * (b - a) / 2.
        mean_raw += (b - a) / 2. * np.dot(weights, _taper(q, p.tip_fraction, free_ends)
                                          * (1. + p.opening_variation * np.sin(2 * np.pi * q + phase)))
    width = p.mean_opening_m / mean_raw * taper * (1. + p.opening_variation * np.sin(2 * np.pi * u + phase))
    d, x = depths[None, :], u[:, None]
    common = p.roughness_m * taper[:, None] * (
        .7 * np.sin(4 * np.pi * x + phase) * np.sin(.5 * np.pi * d)
        + .3 * np.sin(6 * np.pi * x - phase) * np.sin(np.pi * d))
    differential = .1 * p.roughness_m * taper[:, None] * np.sin(4 * np.pi * x + phase + .4) * np.sin(np.pi * d)
    return dict(arc_m=arc, taper=taper, mouth_width_m=width, depth_m=p.depth_m * taper,
                depth_fraction=depths, common_displacement_m=common, differential_displacement_m=differential,
                left_offset_m=-width[:, None] / 2. + common - differential,
                right_offset_m=width[:, None] / 2. + common + differential)


def evaluate_profile(trace, parameters, s_m, *, depth_fraction=(0.,)):
    """Evaluate on the unchanged source XY footprint, never on a cropped chain."""
    result = evaluate_arc_profile(parameters, s_m, trace.length_m, free_ends=trace.free_ends,
                                  phase=trace.phase_rad, depth_fraction=depth_fraction)
    result['center_2d_m'] = trace.points_at(s_m)
    return result


def refine_path_stations(points, source_segment_ids, max_arc_step_m):
    """Insert stations on original segments, preserving every original vertex.

    Mapping arrays are per station. An interior source vertex belongs to the
    next original segment at fraction zero; the final vertex belongs to the
    last segment at fraction one. Thus segment_index[:-1] also maps each new
    edge to its original edge. No coordinates are smoothed or removed.
    """
    original = np.asarray(points, dtype=float)
    identities = np.asarray(source_segment_ids)
    if original.ndim != 2 or original.shape[1] not in (2, 3) or len(original) < 2 or not np.isfinite(original).all():
        raise ValueError('source path requires at least two finite 2D or 3D points')
    if identities.shape != (len(original) - 1,):
        raise ValueError('one source segment identity is required per original edge')
    if isinstance(max_arc_step_m, (bool, np.bool_)) or not isinstance(max_arc_step_m, Real) or not np.isfinite(max_arc_step_m) or max_arc_step_m <= 0:
        raise ValueError('max_arc_step_m must be positive and finite')
    edges = np.diff(original, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0):
        raise ValueError('source path segments must have positive length')
    original_arc = np.r_[0., np.cumsum(lengths)]
    refined, arcs, indices, fractions = [], [], [], []
    for index, (start, edge, length) in enumerate(zip(original[:-1], edges, lengths)):
        count = int(np.ceil(length / max_arc_step_m))
        f = np.arange(count, dtype=float) / count
        part = start + f[:, None] * edge
        part[0] = original[index]
        refined.append(part)
        arcs.append(original_arc[index] + f * length)
        indices.extend([index] * count)
        fractions.extend(f)
    indices = np.asarray(indices + [len(edges) - 1], dtype=np.int64)
    result = np.vstack([*refined, original[-1:]])
    if np.any(np.linalg.norm(np.diff(result, axis=0), axis=1) <= 0):
        raise ValueError('refined stations collapse at source coordinate resolution')
    return dict(points=result, arc_m=np.concatenate([*arcs, original_arc[-1:]]),
                segment_index=indices, original_segment_id=identities[indices],
                source_fraction=np.asarray(fractions + [1.]))


def evaluate_discrete_arc_profile(parameters, arc_m, total_length_m, *, free_ends=(False, False),
                                  phase=0., depth_fraction=(0.,)):
    """Normalize nominal opening on the actual piecewise-linear stations.

    Complete ordered stations are required, including both path ends. Only
    the nominal half-opening terms change; roughness retains its metre scale.
    """
    arc = _arcs(arc_m, total_length_m)
    if len(arc) < 2 or arc[0] != 0. or arc[-1] != total_length_m or np.any(np.diff(arc) <= 0):
        raise ValueError('discrete profile requires complete strictly ordered path stations')
    result = evaluate_arc_profile(parameters, arc, total_length_m, free_ends=free_ends,
                                  phase=phase, depth_fraction=depth_fraction)
    widths = result['mouth_width_m']
    ds = np.diff(arc)
    mean = float(np.dot((widths[:-1] + widths[1:]) / 2., ds) / total_length_m)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('discrete path needs positive mean opening; add interior stations at closed tips')
    factor = parameters.mean_opening_m / mean
    widths = widths * factor
    common, differential = result['common_displacement_m'], result['differential_displacement_m']
    result.update(mouth_width_m=widths,
                  left_offset_m=-widths[:, None] / 2. + common - differential,
                  right_offset_m=widths[:, None] / 2. + common + differential,
                  continuous_to_discrete_factor=factor,
                  continuous_sampled_mean_opening_m=mean,
                  actual_pl_mean_opening_m=float(np.dot((widths[:-1] + widths[1:]) / 2., ds) / total_length_m))
    return result


CLOSURE_INWARD_ERROR = 'segment inward disagrees with its in-surface frame'


def closed_path_closure_frame(points, inward, normal, outward, transverse_raw):
    """Node frames of a CLOSED exposure path with its closure node framed like an interior node.

    The binding (cavity_template/prepare.py) gives every interior node the bisector tangent of its two edges but node 0
    and node -1 the one-sided tangents unit[0] / unit[-1], also when the chain is closed and both are the same point.
    With a sharp closing turn (108-174 deg in every failing case) the one-sided tangent is far from the host's
    in-surface boundary direction and the outward-sign rule puts that node's inward on the other side of the loop;
    panel PathPieces then rejects the segment. Here the closure node gets prepare.py's interior rule with tangent t =
    unit[-1] + unit[0]: transverse = transverse_raw (the node's raw interface-mesh normal, prepare.normals_on)
    projected perpendicular to t, its sign continuing the stored normal of each of the two node slots; inward = t x
    transverse, negated where inward.outward > 0. Every other node is returned unchanged. Returns (inward, normal,
    record). Fails closed on an open path or a closing turn that folds back (|t| <= 1e-9)."""
    P = np.asarray(points, float)
    if len(P) < 4 or not np.array_equal(P[0], P[-1]):
        raise ValueError('closure frame needs a closed path (first point == last point)')
    u = np.diff(P, axis=0)
    u = u / np.linalg.norm(u, axis=1)[:, None]
    t = u[-1] + u[0]
    size = float(np.linalg.norm(t))
    if size <= 1e-9:
        raise ValueError('closing turn of a closed path folds back on itself')
    t = t / size
    inward_out, normal_out = np.array(inward, float), np.array(normal, float)
    base = np.asarray(transverse_raw, float)
    base = base - (base @ t) * t
    base = base / np.linalg.norm(base)
    record = dict(rule='closure node framed as an interior node (bisector tangent)',
                  closing_turn_deg=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u[-1], u[0])), u[-1] @ u[0]))),
                  bisector_norm=size, nodes=[])
    for k in (0, len(P) - 1):
        side = base @ normal_out[k]
        if side == 0:
            raise ValueError('closure transverse is perpendicular to the stored node normal')
        tr = base if side > 0 else -base
        w = np.cross(t, tr)
        o = np.asarray(outward[k], float)
        if w @ o > 0:
            w = -w
        record['nodes'].append(dict(node=int(k), inward_outward_before=float(inward_out[k] @ o), inward_outward_after=float(w @ o),
                                    inward_sign_changed=bool(w @ inward_out[k] < 0),
                                    normal_change_deg=float(np.degrees(np.arccos(np.clip(tr @ normal_out[k], -1., 1.))))))
        inward_out[k], normal_out[k] = w, tr
    return inward_out, normal_out, record
