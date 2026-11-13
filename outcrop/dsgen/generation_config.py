"""Shared local generation settings and independent deterministic random groups."""
from copy import deepcopy
import math

import numpy as np

from outcrop.dsgen.exposure import EXPOSURES
from outcrop.dsgen.lithology import LITHOLOGY, SETTING_OF_KIND, display_catalog

GROUPS = ('geometry', 'appearance', 'scene', 'lighting', 'cameras')
PRESETS = ('dawn', 'noon', 'dusk', 'moon', 'custom')
# Roadcut/canyon source boundaries fail the ordinary-process check in the
# current coarse probes. Keep their manual editors, but do not draw them.
RANDOM_EXPOSURES = ('escarpment', 'bluff', 'escarpment_ridged', 'coast')


def default_config():
    return dict(schema_version=1, seed=42, mode='stepwise', count=1, retry_limit=0,
        geometry=dict(family='group_a', group_b_setting='setting_1', exposure='bluff',
            tile_m=40., layer_count=5, fault_count=0, fracture_count=8, unconformity=True,
            faults=None, stratigraphy=None, fracture_network=None, meshing=None, shape=None, model_spec=None, model_exposure=None, model_sample_id=None, resource_root=None,
            max_edge_m=.14, samples=[720, 240], lithologies=['group_a_01', 'group_a_02', 'group_a_04']),
        appearance=dict(provider='manual', model='', quality='high', guide_resolution=[768, 512],
            grain_mm=1., notes='', unit_overrides={}, guide_views=None),
        scene=dict(kind='bare', ground_color=[155, 139, 110], ground=None, environment=None, weather=None,
                   scanned_preset=None),
        lighting=dict(presets=['noon'], azimuth_deg=135., elevation_deg=60., sun_lux=50000.,
            moon_lux=.2, exposure_bias=-14.5),
        cameras=dict(count=4, azimuth_deg=[-35., 0., 35., 180.], elevation_deg=15.,
            distance_scale=1.8, focal_mm=35., closeups=[]),
        output=dict(width=1280, height=720, format='png'))


def capabilities():
    from .scene_config import ground_capabilities, scene_catalogue
    from .scanned_scenes import catalogue as scanned_catalogue
    return dict(schema_version=1, defaults=default_config(), exposures=list(EXPOSURES),
        families=['group_a', 'group_b', 'group_c'],
        group_b_settings=['setting_1', 'setting_2'], lithologies=list(LITHOLOGY), lithology_catalog=display_catalog(),
        scene_kinds=['bare', 'sand'], scene_catalogue=scene_catalogue(),
        ground_controls={kind: ground_capabilities(kind) for kind in EXPOSURES},
        environments=__import__('outcrop.dsgen.env.config', fromlist=['x']).environment_capabilities(),
        scanned_scenes=scanned_catalogue(),
        lighting_presets=list(PRESETS), random_groups=list(GROUPS),
        ranges=dict(seed=[0, 2**32 - 1], count=[1, 100], retry_limit=[0, 10],
            tile_m=[30., 60.], layer_count=[1, 8], fault_count=[0, None],
            fracture_count=[0, 20], max_edge_m=[.07, 5.],
            sample_azimuth=[8, 1800], sample_latitude=[4, 520],
            samples_min=[8, 4], samples_max=[1800, 520], grain_mm=[.01, 1000.],
            azimuth_deg=[-360., 360.], elevation_deg=[-90., 90.],
            sun_lux=[0., 200000.], moon_lux=[0., 10.], exposure_bias=[-25., 25.],
            camera_count=[1, 32], camera_elevation=[-30., 85.],
            distance_scale=[1., 10.], focal_mm=[12., 200.],
            width=[64, 8192], height=[64, 8192]))


def _merge(default, value, prefix=''):
    if not isinstance(value, dict):
        raise ValueError((prefix or 'config') + ' must be an object')
    extra = set(value) - set(default)
    if extra:
        raise ValueError('unknown settings: ' + ', '.join(prefix + key for key in sorted(extra)))
    result = deepcopy(default)
    for key, item in value.items():
        if isinstance(default[key], dict) and key != 'unit_overrides':
            result[key] = _merge(default[key], item, prefix + key + '.')
        else:
            result[key] = deepcopy(item)
    return result


def _number(value, name, lower, upper, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be numeric')
    if integer and type(value) is not int:
        raise ValueError(name + ' must be an integer')
    if not math.isfinite(value) or value < lower or upper is not None and value > upper:
        raise ValueError('%s must be finite in [%s, %s]' % (name, lower, upper))


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(name + ' must be one of ' + ', '.join(choices))


def _vector(value, size, name, lower, upper, integer=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(name + ' must contain %d values' % size)
    for item in value:
        _number(item, name, lower, upper, integer)


def _validate_fracture_network(value, count):
    if value is None:
        return None
    name = 'geometry.fracture_network'
    allowed = {'mean_direction_deg', 'direction_std_deg', 'direction_groups', 'extrusion_direction'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(name + ' must contain only direction controls')
    result = deepcopy(value)
    if 'direction_groups' in result:
        if result['direction_groups'] is None or {'mean_direction_deg', 'direction_std_deg'} & set(result):
            raise ValueError(name + '.direction_groups requires groups without scalar angle overrides')
        from .fracture_network import validate_direction_groups
        result['direction_groups'] = validate_direction_groups(result['direction_groups'], count)
    for key, lower in [('mean_direction_deg', -math.inf), ('direction_std_deg', 0.)]:
        if key in result:
            _number(result[key], name + '.' + key, lower, None)
    if 'extrusion_direction' in result:
        vector = result['extrusion_direction']
        _vector(vector, 3, name + '.extrusion_direction', -math.inf, None)
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0 or abs(vector[2] / norm) <= 1e-9:
            raise ValueError(name + '.extrusion_direction must be nonzero and transverse to horizontal XY')
    return result


def _validate(cfg):
    _number(cfg['schema_version'], 'schema_version', 1, 1, True)
    _number(cfg['seed'], 'seed', 0, 2**32 - 1, True)
    _choice(cfg['mode'], ['stepwise', 'automatic'], 'mode')
    for name, maximum in [('count', 100), ('retry_limit', 10)]:
        _number(cfg[name], name, 0 if name == 'retry_limit' else 1, maximum, True)
    if cfg['seed'] + cfg['count'] - 1 > 2**32 - 1:
        raise ValueError('batch seed range exceeds the supported 32-bit seeds')
    geo = cfg['geometry']
    from .scanned_scenes import validate_preset
    scanned = validate_preset(cfg['scene']['scanned_preset'])
    if scanned is not None:
        if cfg['scene']['environment'] is not None:
            raise ValueError('scene.scanned_preset and scene.environment are mutually exclusive')
        if geo['model_spec'] is not None:
            raise ValueError('scene.scanned_preset does not support ModelSpec; use a native bluff exposure')
        if geo['exposure'] != 'bluff':
            raise ValueError('scene.scanned_preset requires geometry.exposure=bluff for the 32m contact area')
    for key in ('model_exposure', 'model_sample_id', 'resource_root'):
        if geo[key] is not None and (not isinstance(geo[key], str) or not geo[key].strip()):
            raise ValueError('geometry.' + key + ' must be null or nonempty text')
    if geo['model_spec'] is not None:
        if geo['fracture_network'] is not None and geo['fracture_network'] != {}:
            raise ValueError('geometry.fracture_network and geometry.model_spec are mutually exclusive')
        from outcrop.modeling.decode import decode_model
        decode_model(geo['model_spec'])
    else:
        from .faults import validate_faults
        if geo['stratigraphy'] is not None and not isinstance(geo['stratigraphy'], dict):
            raise ValueError('geometry.stratigraphy must be null or native stratigraphy options')
        if geo['meshing'] is not None and not isinstance(geo['meshing'], dict):
            raise ValueError('geometry.meshing must be null or a native meshing request object')
        validate_faults(geo['faults'])
        if geo['faults'] is not None:
            geo['fault_count'] = len(geo['faults'])
        _choice(geo['family'], ['group_a', 'group_b', 'group_c'], 'geometry.family')
        _choice(geo['group_b_setting'], ['setting_1', 'setting_2'], 'geometry.group_b_setting')
        _choice(geo['exposure'], EXPOSURES, 'geometry.exposure')
        from .shape_config import validate_shape
        geo['shape'] = validate_shape(geo['exposure'], geo['shape'])
        for key, lower, upper, integer in [('tile_m', 30., 60., False), ('layer_count', 1, 8, True),
                ('fault_count', 0, None, True), ('fracture_count', 0, 20, True), ('max_edge_m', .07, 5., False)]:
            _number(geo[key], 'geometry.' + key, lower, upper, integer)
        geo['fracture_network'] = _validate_fracture_network(geo['fracture_network'], geo['fracture_count'])
        if type(geo['unconformity']) is not bool:
            raise ValueError('geometry.unconformity must be boolean')
        if geo['unconformity'] and geo['layer_count'] < 2:
            raise ValueError('unconformity requires at least two layers')
        _vector(geo['samples'], 2, 'geometry.samples', 4, 1800, True)
        if not 8 <= geo['samples'][0] <= 1800 or not 4 <= geo['samples'][1] <= 520:
            raise ValueError('geometry.samples requires azimuth 8..1800 and latitude 4..520')
        if not isinstance(geo['lithologies'], list) or not geo['lithologies']:
            raise ValueError('geometry.lithologies must be a nonempty list')
        for lith in geo['lithologies']:
            _choice(lith, LITHOLOGY, 'geometry.lithologies')
    app = cfg['appearance']
    _choice(app['provider'], ['manual'], 'appearance.provider')
    _choice(app['quality'], ['low', 'medium', 'high', 'auto'], 'appearance.quality')
    for key in ('model', 'notes'):
        if not isinstance(app[key], str):
            raise ValueError('appearance.' + key + ' must be text')
    _vector(app['guide_resolution'], 2, 'appearance.guide_resolution', 64, 4096, True)
    _number(app['grain_mm'], 'appearance.grain_mm', .01, 1000.)
    if app['guide_views'] is not None:
        from outcrop.workbench.ue import GUIDE_VIEWS
        names = [row[0] for row in GUIDE_VIEWS]
        if (not isinstance(app['guide_views'], list) or len(set(app['guide_views'])) != len(app['guide_views'])
                or any(name not in names for name in app['guide_views']) or 'front' not in app['guide_views']):
            raise ValueError('appearance.guide_views must be null or distinct names from %s including the '
                             'common front reference' % ', '.join(names))
    if not isinstance(app['unit_overrides'], dict):
        raise ValueError('appearance.unit_overrides must be an object')
    for unit, override in app['unit_overrides'].items():
        if not isinstance(unit, str) or not unit or not isinstance(override, dict):
            raise ValueError('unit overrides require nonempty unit IDs and objects')
        if set(override) - {'color_srgb', 'description', 'grain_mm'}:
            raise ValueError('unknown unit appearance settings')
        if 'color_srgb' in override:
            _vector(override['color_srgb'], 3, 'color_srgb', 0, 255, True)
        if 'description' in override and not isinstance(override['description'], str):
            raise ValueError('unit description must be text')
        if 'grain_mm' in override:
            _number(override['grain_mm'], 'unit grain_mm', .01, 1000.)
    _choice(cfg['scene']['kind'], ['bare', 'sand'], 'scene.kind')
    _vector(cfg['scene']['ground_color'], 3, 'scene.ground_color', 0, 255, True)
    from .scene_config import validate_ground, check_ground_budget
    cfg['scene']['ground'] = validate_ground(cfg['scene']['ground'])
    if cfg['scene']['ground'] is not None:
        if geo['model_spec'] is not None:
            raise ValueError('scene.ground requires a template exposure with native ground Items; '
                             'a complete ModelSpec supplies its own geometry')
        check_ground_budget(geo['exposure'], cfg['scene']['ground'])
    from .env.config import validate_environment, validate_weather
    for key in ('environment', 'weather'):
        if cfg['scene'][key] is not None and geo['model_spec'] is not None:
            raise ValueError('scene.%s requires a template exposure; a complete ModelSpec supplies its own scene' % key)
    cfg['scene']['environment'] = validate_environment(cfg['scene']['environment'], cfg['seed'])
    from .env.config import GEOMETRY_REQUIRED
    if cfg['scene']['environment'] is not None:
        required = GEOMETRY_REQUIRED.get(cfg['scene']['environment']['preset'])
        if required is not None and geo['exposure'] != required:
            raise ValueError('environment %s changes the target exposure: set geometry.exposure=%r (a new geometry '
                             'revision with its own geological cut)' % (cfg['scene']['environment']['preset'], required))
    if cfg['scene']['environment'] is not None and cfg['scene']['ground'] is None:
        # Landforms need a denser near ground than the native default; materialized explicitly.
        cfg['scene']['ground'] = validate_ground(dict(half_m=160., spacing_m=1., far_m=3000.))
        check_ground_budget(geo['exposure'], cfg['scene']['ground'])
    cfg['scene']['weather'] = validate_weather(cfg['scene']['weather'], cfg['seed'])
    if cfg['scene']['weather'] is not None:
        cfg['scene']['weather'] = dict(variants=cfg['scene']['weather'])
    lights = cfg['lighting']
    if not isinstance(lights['presets'], list) or not lights['presets']:
        raise ValueError('lighting.presets must be a nonempty list')
    for preset in lights['presets']:
        _choice(preset, PRESETS + ('sunset', 'moonlight'), 'lighting.presets')
    for key, lower, upper in [('azimuth_deg', -360., 360.), ('elevation_deg', -90., 90.),
            ('sun_lux', 0., 200000.), ('moon_lux', 0., 10.), ('exposure_bias', -25., 25.)]:
        _number(lights[key], 'lighting.' + key, lower, upper)
    cams = cfg['cameras']
    _number(cams['count'], 'cameras.count', 1, 32, True)
    _vector(cams['azimuth_deg'], cams['count'], 'cameras.azimuth_deg', -360., 360.)
    for key, lower, upper in [('elevation_deg', -30., 85.), ('distance_scale', 1., 10.), ('focal_mm', 12., 200.)]:
        _number(cams[key], 'cameras.' + key, lower, upper)
    if not isinstance(cams['closeups'], list) or len(cams['closeups']) > 8:
        raise ValueError('cameras.closeups must be a list of at most 8 contact close-ups')
    for i, row in enumerate(cams['closeups']):
        name = 'cameras.closeups[%d]' % i
        if not isinstance(row, dict) or {'azimuth_deg', 'elevation_deg', 'distance_m'} - set(row)                 or set(row) - {'azimuth_deg', 'elevation_deg', 'distance_m', 'height_fraction', 'inset'}:
            raise ValueError(name + ' requires azimuth_deg, elevation_deg, distance_m '
                             '(optional height_fraction, inset)')
        for key, lower, upper in [('azimuth_deg', -360., 360.), ('elevation_deg', -10., 60.), ('distance_m', 2., 80.),
                                  ('height_fraction', 0., 1.), ('inset', 0., 1.)]:
            if key in row:
                _number(row[key], name + '.' + key, lower, upper)
    for key in ('width', 'height'):
        _number(cfg['output'][key], 'output.' + key, 64, 8192, True)
    _choice(cfg['output']['format'], ['png'], 'output.format')


def _random_group(group, current, seed, locked=(), *, scanned_scene=False):
    g = np.random.default_rng([seed, 7001, GROUPS.index(group)])
    value = deepcopy(current)
    if group in locked or all(group + '.' + key in locked for key in current):
        return value
    if group == 'geometry':
        if current['model_spec'] is not None:
            return value
        shape_locked = any(path == 'geometry.shape' or path.startswith('geometry.shape.') for path in locked)
        if (shape_locked or 'geometry.exposure' in locked) and current['exposure'] not in RANDOM_EXPOSURES:
            raise ValueError('This shape is outside the constrained random range; unlock the shape and retry, or edit it manually.')
        family = str(g.choice(['group_a', 'group_b', 'group_c']))
        count = int(g.integers(2, 5))
        setting = str(g.choice(['setting_1', 'setting_2']))
        fault_locked = 'geometry.faults' in locked and current['faults'] is not None
        network_locked = any(path == 'geometry.fracture_network' or path.startswith('geometry.fracture_network.')
                             for path in locked)
        strata = current['stratigraphy'] or {}
        strata_count_locked = ('layer_count' in strata and (
            'geometry.stratigraphy' in locked or 'geometry.stratigraphy.layer_count' in locked))
        if fault_locked or 'geometry.family' in locked:
            family = current['family']
        if fault_locked or 'geometry.group_b_setting' in locked:
            setting = current['group_b_setting']
        if fault_locked or strata_count_locked or 'geometry.layer_count' in locked:
            count = current['layer_count']
        names = [name for name, item in LITHOLOGY.items() if item['family'] == family
                 and (family != 'group_b' or SETTING_OF_KIND.get(item['subclass']) == setting)]
        value.update(family=family, group_b_setting=setting,
            exposure=str(g.choice(RANDOM_EXPOSURES)), tile_m=float(g.uniform(30., 32. if scanned_scene else 60.)), layer_count=count,
            fault_count=int(g.integers(0, 2)), fracture_count=int(g.integers(0, 7)),
            unconformity=bool(g.integers(0, 2)), lithologies=g.choice(names, count).tolist(),
            shape=None, faults=None, stratigraphy={'kind': 'planar'},
            fracture_network=deepcopy(current['fracture_network']) if network_locked else None)
        if network_locked and (current['fracture_network'] or {}).get('direction_groups'):
            value['fracture_count'] = current['fracture_count']
        # One native rigid fault duplicates at most the planar body boundaries.
        # With four layers, UC, four roof facets and three dykes this is <=47
        # occupancy planes, within the unchanged 128-plane native budget.
        if fault_locked or strata_count_locked:
            value['unconformity'] = current['unconformity']
        # These declarations reference the old layer/body layout. Exact locks
        # are restored by resolve_config; unlocked declarations must be redrawn.
        if any(path.startswith('geometry.stratigraphy.') and path not in (
                'geometry.stratigraphy.kind', 'geometry.stratigraphy.layer_count') for path in locked):
            value['stratigraphy']['kind'] = strata.get('kind', 'nonparallel')
        if shape_locked:
            # A locked morphology belongs to its exposure family. Keep that
            # family and its coordinate frame while geology is redrawn.
            value['exposure'] = current['exposure']
            value['tile_m'] = current['tile_m']
            value['shape'] = {}
    elif group == 'appearance':
        value['grain_mm'] = float(g.uniform(.1, 5.))
    elif group == 'scene':
        value['kind'] = str(g.choice(['bare', 'sand']))
        palette = [[155, 139, 110], [178, 157, 119], [194, 174, 138], [143, 130, 106]]
        base = np.asarray(palette[int(g.integers(len(palette)))])
        value['ground_color'] = np.clip(base + int(g.integers(-10, 11)), 0, 255).tolist()
    elif group == 'lighting':
        value.update(presets=[str(g.choice(PRESETS))], azimuth_deg=float(g.uniform(0., 360.)),
            elevation_deg=float(g.uniform(8., 75.)), sun_lux=float(g.uniform(20000., 100000.)))
    else:
        count = int(g.integers(2, 7))
        if 'cameras.count' in locked:
            count = current['count']
        elif 'cameras.azimuth_deg' in locked:
            count = len(current['azimuth_deg'])
        value.update(count=count, azimuth_deg=g.uniform(-180., 180., count).tolist(),
            elevation_deg=float(g.uniform(5., 40.)), distance_scale=float(g.uniform(1.5, 2.5)),
            focal_mm=float(g.uniform(24., 60.)))
    return value


def resolve_config(config, randomize=(), locked=()):
    """Merge defaults, validate, then draw only explicitly selected groups.

    Locks name a group or a dotted field. Calling without ``randomize`` keeps
    the materialized values; later camera/light changes do not redraw geology.
    """
    from .legacy import translate_config as _legacy_translate_config
    actual = _merge(default_config(), _legacy_translate_config(config))   # a configuration with the former names reads as labels
    _validate(actual)
    if isinstance(randomize, str) or isinstance(locked, str):
        raise ValueError('randomize and locked must be lists of names')
    locked = [{'geometry.igneous_setting': 'geometry.group_b_setting'}.get(p, p) for p in locked]   # a lock saved before the labels
    for group in randomize:
        if group not in GROUPS:
            raise ValueError('unknown random group: ' + str(group))
    originals = {}
    for path in locked:
        if not isinstance(path, str):
            raise ValueError('lock paths must be text')
        steps = path.split('.')
        if steps[0] not in GROUPS:
            raise ValueError('unknown lock path: ' + path)
        node = actual
        for step in steps:
            if not isinstance(node, dict) or step not in node:
                raise ValueError('unknown lock path: ' + path)
            node = node[step]
        originals[path] = deepcopy(node)
    for group in set(randomize):
        actual[group] = _random_group(group, actual[group], actual['seed'], locked,
                                     scanned_scene=actual['scene']['scanned_preset'] is not None)
    if 'geometry' in randomize and actual['scene']['scanned_preset'] is not None:
        actual['geometry']['exposure'] = 'bluff'
    for path, value in originals.items():
        steps, node = path.split('.'), actual
        for step in steps[:-1]:
            node = node.setdefault(step, {})
        node[steps[-1]] = value
    if 'geometry' in randomize and actual['geometry']['layer_count'] == 1:
        actual['geometry']['unconformity'] = False
    _validate(actual)
    return actual


def materialize_scenes(config):
    """Resolve one batch into the same actual scene settings for every caller."""
    resolved = resolve_config(config)
    scenes = []
    for index in range(resolved['count']):
        actual = deepcopy(resolved)
        actual.update(seed=resolved['seed'] + index, count=1)
        model = actual['geometry']['model_spec']
        if resolved['count'] > 1 and model is not None and 'root_seed' in model.get('sampling', {}):
            base = model['sampling']['root_seed']
            _number(base, 'sampling.root_seed', 0, None, True)
            model['sampling']['root_seed'] = base + index
        scenes.append(actual)
    return scenes
