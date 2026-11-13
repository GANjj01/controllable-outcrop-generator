"""Optional geological declarations (fracture networks and related features) built from the existing model primitives."""
import math

import numpy as np

from .lithology import LITHOLOGY, ORDER, lithology_records


def _options(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError('%s must be a parameter object' % name)
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError('unknown %s parameters: %s' % (name, ', '.join(sorted(extra))))
    return dict(value)


def _normal(dip, direction):
    dip, direction = float(dip), float(direction)
    if not np.isfinite([dip, direction]).all() or not 0 <= dip < 90:
        raise ValueError('finite dip in [0,90) and dip direction required')
    a, b = math.radians(dip), math.radians(direction)
    return np.array([math.sin(a) * math.sin(b), math.sin(a) * math.cos(b), math.cos(a)])


def apply_stratigraphy(spec, options):
    """Share one bed coordinate, then optionally truncate two successions.

    Scene bands span the coordinate's nodal range to cover the domain. Thus
    layer fractions are relative proportions across the declared tile; the
    physical B/H helper is separately available in ``stratigraphy.py``.
    """
    from .stratigraphy import nonparallel_field
    cfg = _options(options, {'kind', 'layer_count', 'base_m', 'total_thickness_m', 'relief_m',
        'thickness_variation', 'wavelengths_m', 'phase_rad', 'shape', 'dip_deg',
        'dip_direction_deg', 'unconformity'}, 'stratigraphy')
    kind = cfg.pop('kind', 'nonparallel')
    cfg.pop('layer_count', None)
    erosion = cfg.pop('unconformity', None)
    if kind not in {'nonparallel', 'planar'}:
        raise ValueError('stratigraphy kind must be nonparallel or planar')
    meta = spec['metadata']
    record = {'kind': kind, 'layer_count': len(meta['beds']),
              'source': 'RGM-inspired shared stratigraphic coordinate'}
    if kind == 'planar' and cfg:
        raise ValueError('planar mode accepts only an optional unconformity')
    if kind == 'nonparallel':
        lo, hi = np.array(spec['domain']['min_m']), np.array(spec['domain']['max_m'])
        span = hi - lo
        defaults = dict(base_m=float(lo[2]), total_thickness_m=float(span[2]),
            relief_m=.6, thickness_variation=.15,
            wavelengths_m=[float(1.7 * span[0]), float(1.3 * span[1])],
            phase_rad=np.random.default_rng([int(meta['seed']), 901]).uniform(0, 2 * math.pi, 2).tolist(),
            shape=[2, 2, 2], dip_deg=meta['bed_dip_deg'],
            dip_direction_deg=meta['bed_dip_direction_deg'])
        defaults.update(cfg)
        field = nonparallel_field('bed-coordinate', (lo, hi), **defaults)
        values = np.asarray(field['values'])
        lower, upper = float(values.min()), float(values.max())
        bands = [g for g in spec['geometry'] if g['kind'] == 'band' and g.get('field') == 'bed-coordinate']
        pile = [g for g in bands if g['id'].startswith('band')]
        a, b = pile[0]['min'], pile[-1]['max']
        for band in bands:
            band['min'] = lower + (band['min'] - a) / (b - a) * (upper - lower)
            band['max'] = lower + (band['max'] - a) / (b - a) * (upper - lower)
            band['unit'] = '1'
        spec['geometry'] = [field if g['id'] == 'bed-coordinate' else g for g in spec['geometry']]
        record.update(parameters=defaults, coordinate_range=[lower, upper],
                      representation='shared Freudenthal P1 field; vertical thickness')
        meta['bed_dip_deg'] = float(defaults['dip_deg'])
        meta['bed_dip_direction_deg'] = float(defaults['dip_direction_deg'])
    if erosion is not None:
        record['unconformity'] = _unconformity(spec, erosion)
    meta.setdefault('paper_methods', {})['stratigraphy'] = record


def _unconformity(spec, options):
    from .stratigraphy import unconformity_regions
    cfg = _options(options, {'elevation_m', 'dip_deg', 'dip_direction_deg',
        'young_lithologies', 'young_fractions', 'young_dip_deg', 'young_dip_direction_deg'}, 'unconformity')
    if 'elevation_m' not in cfg or not np.isfinite(float(cfg['elevation_m'])):
        raise ValueError('unconformity requires finite elevation_m at domain XY centre')
    lo, hi = np.array(spec['domain']['min_m']), np.array(spec['domain']['max_m'])
    point = (lo + hi) / 2
    point[2] = float(cfg['elevation_m'])
    n = _normal(cfg.get('dip_deg', 0.), cfg.get('dip_direction_deg', 0.))
    selectors = unconformity_regions('erosion', n, float(n @ point))
    spec['geometry'].extend(selectors)
    for body in spec['bodies']:
        if body['id'].startswith('bed-'):
            oid = body['geometry'] + ':eroded'
            spec['geometry'].append({'id': oid, 'kind': 'intersection',
                                     'operands': [body['geometry'], 'erosion:old']})
            body['geometry'] = oid
    liths = cfg.get('young_lithologies', ['group_a_01', 'group_a_02'])
    if not isinstance(liths, (list, tuple)) or not liths or any(l not in LITHOLOGY for l in liths):
        raise ValueError('young_lithologies must name catalog lithologies')
    fractions = np.asarray(cfg.get('young_fractions', [1.] * len(liths)), float)
    if fractions.shape != (len(liths),) or not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError('one finite positive young fraction per lithology required')
    ny = _normal(cfg.get('young_dip_deg', 0.), cfg.get('young_dip_direction_deg', 0.))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    q = corners @ ny
    levels = float(q.min()) - 1 + np.r_[0., np.cumsum(fractions) / fractions.sum()] * (float(np.ptp(q)) + 2.)
    spec['geometry'].append({'id': 'young-coordinate', 'kind': 'scalar_plane',
                             'normal': ny.tolist(), 'offset_m': 0., 'unit': 'm'})
    young = []
    for i, lith in enumerate(liths):
        bid, band, region = 'young-bed-%02d' % i, 'young-band-%02d' % i, 'young-rock-%02d' % i
        spec['geometry'].extend([
            {'id': band, 'kind': 'band', 'field': 'young-coordinate', 'min': float(levels[i]),
             'max': float(levels[i + 1]), 'unit': 'm'},
            {'id': region, 'kind': 'intersection', 'operands': ['footprint', 'erosion:young', band]}])
        spec['bodies'].append({'id': bid, 'geometry': region, 'material': {'kind': 'homogeneous', 'lithology': lith},
                               'unit_id': 'young-unit-%02d' % i})
        young.append(bid)
    spec['features'].append({'id': 'young-bedding', 'kind': 'bedding', 'hosts': young,
                              'fabric': {'kind': 'scalar', 'field': 'young-coordinate'}})
    for feature in spec['features']:
        if feature['kind'] in {'joint', 'joint_set'}:
            feature['hosts'].extend(young)
    for event in spec['events']:
        if event['kind'] in {'fault', 'intrusion'}:
            event['targets'].extend(young)
    spec['lithologies'] = lithology_records(sorted({b['material']['lithology'] for b in spec['bodies']}, key=ORDER.__getitem__))
    return {**cfg, 'plane_normal': n.tolist(), 'plane_offset_m': float(n @ point),
            'old_selector': 'erosion:old', 'young_selector': 'erosion:young',
            'construction': 'complementary static successions, before existing intrusion/fault events'}


def apply_fracture_network(spec, options):
    from .fracture_network import sample_network, network_geometry
    cfg = dict(options) if isinstance(options, dict) else _options(options, (), 'fracture_network')
    height = float(cfg.pop('height_m', spec['domain']['max_m'][2]))
    extrusion = cfg.pop('extrusion_direction', [0., 0., 1.])
    if not np.isfinite(height) or height <= 0:
        raise ValueError('finite positive fracture height_m required')
    seed = int(cfg.pop('seed', int(spec['metadata']['seed']) + 3001))
    lo, hi = spec['domain']['min_m'], spec['domain']['max_m']
    network = sample_network(seed, [[lo[0], lo[1]], [hi[0], hi[1]]], **cfg)
    geometry, features = network_geometry(network, prefix='network-joint',
        origin_m=[0., 0., .5 * (lo[2] + hi[2])], axis_u=[1., 0., 0.], axis_v=[0., 1., 0.],
        extrusion_direction=extrusion, height_m=height)
    old = {f['id'] for f in spec['features'] if f['kind'] in {'joint', 'joint_set'}}
    spec['features'] = [f for f in spec['features'] if f['id'] not in old]
    hosts = [b['id'] for b in spec['bodies']]
    for feature in features:
        feature.update(hosts=list(hosts), support='footprint')
    creation = next((e for e in spec['events'] if e['kind'] == 'create_features' and set(e['features']) <= old), None)
    if creation is not None:
        creation['features'] = [f['id'] for f in features]
        for feature in features:
            feature['at_stage'] = creation.get('output_stage', creation['id'])
        if not features:
            spec['events'].remove(creation)
            for event in spec['events']:
                event['after'] = [p for dep in event.get('after', [])
                                  for p in (creation.get('after', []) if dep == creation['id'] else [dep])]
    spec['geometry'].extend(geometry)
    spec['features'].extend(features)
    spec['metadata'].setdefault('paper_methods', {})['fracture_network'] = {
        **network, 'height_m': height, 'growth_plane': 'horizontal ENU XY',
        'extrusion_direction': np.asarray(extrusion, float).tolist(),
        'extrusion': 'finite linear strips; zero opening',
        'source': 'Bonneau-inspired sequential 2.5D growth; observable 2D trace statistics'}
