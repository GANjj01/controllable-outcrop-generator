"""Editable source-exterior controls, separate from geological orientations.

Angles exposed here use ENU north-clockwise bearings. The older profile
builders use a counter-clockwise rotation from +X internally.
"""
from copy import deepcopy
import math


_ORIENTATION = ('strike_deg', 'dip_direction_deg')
_SCARP = ('centre_xy_m', 'height_m', 'face_dip_deg', 'toe_radius_m', 'crest_radius_m',
          'apron_slope_deg', 'upland_slope_deg', 'back_slope_deg', 'foot_w_m',
          'relax_start_m', 'relax_m', 'repose_deg', 'taper_start_m', 'reach_m',
          'waves', 'relief', 'variation')
_ROAD = ('centre_xy_m', 'cut_height_m', 'benches', 'lift_1_fraction', 'bench_width_m',
         'bench_tilt_deg', 'face_dip_deg', 'toe_radius_m', 'edge_radius_m',
         'hill_slope_deg', 'hill_rise_m', 'upland_slope_deg', 'back_slope_deg',
         'road_width_m', 'crossfall', 'shoulder_m', 'verge_m', 'toe_w_m',
         'curve_radius_m', 'curve_sign', 'relax_start_m', 'relax_m', 'taper_start_m',
         'reach_m', 'waves', 'relief', 'variation')
_FIELDS = {
    'bluff': ('width_m', 'depth_m', 'height_m', 'azimuth_deg', 'centre_xy_m',
              'base_depth_m', 'mass_count', 'blend_m', 'rounding_fraction'),
    'escarpment': _ORIENTATION + _SCARP,
    'escarpment_ridged': _ORIENTATION + _SCARP + ('ridges', 'ridge_scale'),
    'roadcut': _ORIENTATION + _ROAD,
    'coast': _ORIENTATION + ('length_m', 'cliff_height_m', 'face_dip_deg',
                            'centre_xy_m', 'sea_level_m'),
    'canyon': _ORIENTATION + ('centre_xy_m', 'depth_m', 'floor_width_m', 'wall_dip_deg', 'wall_asymmetry_deg',
                             'benches', 'bench_fraction', 'bench_width_m', 'crest_radius_m', 'toe_radius_m',
                             'meander_m', 'meander_wavelength_m', 'meander_phase_rad', 'repose_deg', 'reach_m',
                             'end_depth_fraction', 'waves', 'relief', 'variation'),
}
_LABELS = {
    'strike_deg': 'face strike (right-hand rule)', 'dip_direction_deg': 'face dip direction',
    'face_dip_deg': 'face profile dip', 'azimuth_deg': 'long-axis azimuth of the standalone outcrop',
    'width_m': 'design width of the rock body', 'depth_m': 'design depth of the rock body', 'height_m': 'design height',
    'length_m': 'cliff length (including end slopes)', 'cliff_height_m': 'cliff height',
    'cut_height_m': 'cut height', 'center_xy_m': 'plan centre (east, north)',
    'centre_xy_m': 'plan centre (east, north)', 'base_depth_m': 'buried base depth',
    'mass_count': 'number of rounded masses', 'blend_m': 'blend width of the rounded masses',
    'rounding_fraction': 'rounding radius / smallest half size', 'waves': 'waves along strike',
    'relief': 'gentle slope relief', 'ridges': 'ridges and gullies', 'variation': 'wave amplitude factor',
    'ridge_scale': 'ridge amplitude factor', 'sea_level_m': 'sea level above the ground',
    'toe_radius_m': 'toe rounding radius', 'crest_radius_m': 'crest rounding radius',
    'edge_radius_m': 'bench and crest rounding radius', 'apron_slope_deg': 'apron slope',
    'upland_slope_deg': 'upland slope', 'back_slope_deg': 'back slope',
    'foot_w_m': 'toe offset', 'toe_w_m': 'cut toe offset',
    'relax_start_m': 'start of the end relaxation', 'relax_m': 'length of the end relaxation',
    'repose_deg': 'repose angle at the ends', 'taper_start_m': 'start of the end height taper',
    'reach_m': 'length of the end height taper', 'benches': 'number of benches',
    'lift_1_fraction': 'height fraction of the first lift', 'bench_width_m': 'bench width',
    'bench_tilt_deg': 'bench drainage tilt', 'hill_slope_deg': 'slope of the hill above',
    'hill_rise_m': 'rise of the hill above', 'road_width_m': 'road width',
    'crossfall': 'road crossfall', 'shoulder_m': 'shoulder width', 'verge_m': 'verge width',
    'curve_radius_m': 'road curve radius', 'curve_sign': 'road curve direction',
    'depth_m': 'canyon depth', 'floor_width_m': 'floor width', 'wall_dip_deg': 'wall dip',
    'wall_asymmetry_deg': 'half the wall dip difference (positive when the right wall is steeper)', 'bench_fraction': 'wall fraction above the bench',
    'meander_m': 'meander amplitude', 'meander_wavelength_m': 'meander wavelength', 'meander_phase_rad': 'meander phase',
    'end_depth_fraction': 'residual depth fraction at the ends',
}
_FRACTIONS = {'variation', 'ridge_scale', 'lift_1_fraction', 'bench_fraction', 'end_depth_fraction'}
_POSITIVE = {'width_m', 'depth_m', 'height_m', 'base_depth_m', 'blend_m', 'length_m',
             'cliff_height_m', 'cut_height_m', 'toe_radius_m', 'crest_radius_m',
             'edge_radius_m', 'bench_width_m', 'hill_rise_m', 'road_width_m',
             'shoulder_m', 'verge_m', 'curve_radius_m', 'relax_m', 'reach_m', 'floor_width_m',
             'meander_wavelength_m'}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('shape.%s must be a finite number' % name)
    return value


def _object(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError('shape.%s must be an object' % name)
    if any(not isinstance(key, str) for key in value):
        raise ValueError('shape.%s requires text field names' % name)
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError('unknown shape.%s fields: %s' % (name, ', '.join(sorted(unknown))))


def _wave(value, name, height=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError('shape.%s requires [amplitude, wavelength_m, phase_rad]' % name)
    for component in value:
        _number(component, name)
    if value[1] <= 0 or height and abs(value[0]) >= 1:
        raise ValueError('shape.%s requires positive wavelength and height amplitude magnitude <1' % name)


def validate_shape(kind, shape):
    """Return a validated independent copy; ``None`` retains legacy sampling."""
    if kind not in _FIELDS:
        raise ValueError('unknown exposure kind %r' % kind)
    if shape is None:
        return None
    _object(shape, _FIELDS[kind], kind)
    result = deepcopy(shape)
    for key, value in result.items():
        if key in ('center_xy_m', 'centre_xy_m'):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError('shape.%s requires [east_m, north_m]' % key)
            for component in value:
                _number(component, key)
        elif key == 'waves':
            allowed = ('height',) if kind == 'roadcut' else ('depth', 'dip') if kind == 'canyon' else ('foot', 'height', 'dip')
            _object(value, allowed, key)
            for name, wave in value.items():
                if name == 'foot' and isinstance(wave, list) and wave and isinstance(wave[0], list):
                    for item in wave:
                        _wave(item, key + '.' + name)
                else:
                    _wave(wave, key + '.' + name, height=name in ('height', 'depth'))
        elif key == 'relief':
            _object(value, ('seed', 'lowland_m', 'upland_m'), key)
            for name, component in value.items():
                _number(component, key + '.' + name)
                if component < 0 or name == 'seed' and (type(component) is not int or component > 2**32 - 2):
                    raise ValueError('shape.relief requires nonnegative amplitudes and an integer seed')
        elif key == 'ridges':
            if not isinstance(value, list):
                raise ValueError('shape.ridges must be a list')
            for ridge in value:
                _object(ridge, ('u_m', 'sigma_m', 'amp_m', 'height_frac'), key)
                if set(ridge) != {'u_m', 'sigma_m', 'amp_m', 'height_frac'}:
                    raise ValueError('shape.ridges requires u_m, sigma_m, amp_m and height_frac')
                for component in ridge.values():
                    _number(component, key)
                if ridge['sigma_m'] <= 0 or ridge['height_frac'] <= -1:
                    raise ValueError('shape.ridges requires sigma_m>0 and height_frac>-1')
        else:
            _number(value, key)
            if key in _ORIENTATION or key == 'azimuth_deg':
                result[key] = value % 360.
            elif key in _POSITIVE and value <= 0:
                raise ValueError('shape.%s must be positive' % key)
            elif key == 'mass_count' and (type(value) is not int or value < 1):
                raise ValueError('shape.mass_count must be a positive integer')
            elif key == 'benches' and (type(value) is not int or value not in (0, 1)):
                raise ValueError('shape.benches must be 0 or 1')
            elif key == 'curve_sign' and value not in (-1, 1):
                raise ValueError('shape.curve_sign must be -1 or 1')
            elif key == 'rounding_fraction' and not 0 < value <= .45:
                raise ValueError('shape.rounding_fraction must be in (0,0.45]')
            elif key in _FRACTIONS and not 0 <= value <= 1:
                raise ValueError('shape.%s must be in [0,1]' % key)
            elif key == 'wall_asymmetry_deg':
                if not -20 <= value <= 20:
                    raise ValueError('shape.wall_asymmetry_deg must be in [-20,20]')
            elif key.endswith('_deg') and not 0 <= value < 90:
                raise ValueError('shape.%s must be in [0,90)' % key)
            elif key == 'crossfall' and not -1 < value < 1:
                raise ValueError('shape.crossfall must be in (-1,1)')
    if all(key in result for key in _ORIENTATION):
        error = (result['dip_direction_deg'] - result['strike_deg'] - 90. + 180.) % 360. - 180.
        if abs(error) > 1e-8:
            raise ValueError('shape strike/dip direction violate the right-hand rule')
    for key in ('face_dip_deg', 'back_slope_deg'):
        if result.get(key) == 0:
            raise ValueError('shape.%s must be greater than zero' % key)
    return result


def apply_profile_shape(parameters, shape):
    """Merge validated native controls, converting only the public bearing."""
    if not shape:
        return
    for key, value in shape.items():
        if key in _ORIENTATION or key == 'length_m':
            continue
        if key in ('waves', 'relief'):
            parameters[key].update(deepcopy(value))
        else:
            parameters[key] = deepcopy(value)
    if 'strike_deg' in shape or 'dip_direction_deg' in shape:
        bearing = shape.get('strike_deg', (shape.get('dip_direction_deg', 90.) - 90.) % 360.)
        parameters['strike_deg'] = 90. - bearing


def effective_shape(exposure):
    """Editable source values in public conventions, excluding derived fields."""
    kind = exposure['kind']
    if kind == 'bluff':
        masses = exposure['masses']
        result = {key: deepcopy(exposure[key]) for key in ('width_m', 'depth_m', 'height_m', 'blend_m')}
        result.update(azimuth_deg=(90. - exposure.get('rotation_deg', 0.)) % 360.,
                      centre_xy_m=list(exposure['origin_m'][:2]), mass_count=len(masses),
                      base_depth_m=exposure['ground_z_m'] - min(m['center_m'][2] - m['half_size_m'][2] for m in masses))
        if 'rounding_fraction' in exposure:
            result['rounding_fraction'] = exposure['rounding_fraction']
        return result
    editable = set(_FIELDS[kind]) - set(_ORIENTATION)
    if kind == 'coast':
        # The accepted adapter owns actual length and dip, not this old helper.
        editable -= {'length_m', 'face_dip_deg'}
    result = {key: deepcopy(exposure[key]) for key in _FIELDS[kind] if key in editable and key in exposure}
    bearing = (90. - exposure['strike_deg']) % 360.
    result.update(strike_deg=bearing, dip_direction_deg=(bearing + 90.) % 360.)
    return result


def shape_capabilities(kind):
    if kind not in _FIELDS:
        raise ValueError('unknown exposure kind %r' % kind)
    fields = []
    for key in _FIELDS[kind]:
        unit = 'deg' if key.endswith('_deg') else 'm' if key.endswith('_m') else ''
        field = dict(key=key, label=_LABELS.get(key, key), unit=unit)
        if kind == 'coast' and key == 'face_dip_deg':
            field['label'] = 'nominal face dip (toe to crest)'
        if key in ('waves', 'relief', 'ridges'):
            field['type'] = 'json'
        elif key in ('center_xy_m', 'centre_xy_m'):
            field['type'] = 'vector2'
        else:
            field.update(type='integer' if key in ('mass_count', 'benches') else 'number', step=1 if key in ('mass_count', 'benches') else .1)
            if key in _POSITIVE:
                field['min'] = .001
            if key in _FRACTIONS or key == 'rounding_fraction':
                field.update(min=0., max=.45 if key == 'rounding_fraction' else 1., step=.01)
            if key.endswith('_deg'):
                field.update(min=0., max=360. if key in _ORIENTATION or key == 'azimuth_deg' else 89.9)
            if key == 'mass_count':
                field['min'] = 1
            if key == 'benches':
                field.update(min=0, max=1)
            if key == 'curve_sign':
                field.update(min=-1, max=1, step=2)
        fields.append(field)
    notes = ['Angles are ENU azimuths: north is 0 deg, clockwise; face dip direction = right-hand-rule strike + 90 deg.',
             'Sizes act when the source surface is generated; the final extent is limited by the geological domain and the geometry checks.']
    if kind == 'bluff':
        notes.append('A long-axis azimuth of 90 deg is east-west; the design width, depth and height set the rounded-mass construction, not an exact bounding box after weathering.')
    elif kind == 'coast':
        notes.append('The length includes the end slopes; the nominal dip sets the mean horizontal distance from toe to crest, and the local slope varies along the surface. An explicit length or dip is not rescaled.')
    else:
        notes.append('The face dip is the cross-slope profile angle; the local attitude of curved and rough surfaces varies with position.')
    return dict(fields=fields, notes=notes)
