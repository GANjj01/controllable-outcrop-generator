"""Validation of scene.environment and scene.weather; None keeps every existing output."""
from copy import deepcopy
import math

from .catalog import CATEGORIES, IMPLEMENTED, SCHEMAS, WEATHER_DEFAULTS, WEATHER_FIELDS, WEATHER_PRESETS, preset_category

GROUPS = ('terrain', 'ecology', 'water', 'surface')


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    if value < low or value > high:
        raise ValueError('%s must be in [%s, %s]' % (name, low, high))
    return float(value)


def _seed(value, name):
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ValueError(name + ' must be an integer seed in [0, 2^32-1]')
    return value


GEOMETRY_REQUIRED = {'arid_canyon': 'canyon'}


def validate_environment(settings, default_seed):
    """Return the complete actual environment (all parameters materialized) or None."""
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError('scene.environment must be null or an object')
    extra = set(settings) - {'category', 'preset', 'seed', *GROUPS}
    if extra:
        raise ValueError('unknown scene.environment settings: ' + ', '.join(sorted(extra)))
    preset = settings.get('preset')
    category = preset_category(preset) if isinstance(preset, str) else None
    if category is None:
        raise ValueError('scene.environment.preset is required')
    if settings.get('category', category) != category:
        raise ValueError('scene.environment.category %r does not contain preset %r' % (settings.get('category'), preset))
    if category not in dict(CATEGORIES):
        raise ValueError('unknown environment category')
    if preset not in IMPLEMENTED:
        raise ValueError('environment preset %r is not implemented yet; it belongs to a later batch' % preset)
    actual = dict(category=category, preset=preset,
                  seed=_seed(settings.get('seed', (default_seed * 2654435761 + 4101) % 2**32), 'scene.environment.seed'))
    schema = SCHEMAS[preset]
    for group in GROUPS:
        given = settings.get(group, {})
        if not isinstance(given, dict):
            raise ValueError('scene.environment.%s must be an object' % group)
        fields = {row['key']: row for row in schema.get(group, [])}
        unknown = set(given) - set(fields)
        if unknown:
            raise ValueError('preset %s has no %s parameters: %s' % (preset, group, ', '.join(sorted(unknown))))
        actual[group] = {key: _number(given.get(key, row['default']), 'scene.environment.%s.%s' % (group, key), row['min'], row['max'])
                         for key, row in fields.items()}
    terrain = actual['terrain']
    if 'contact_band_m' in terrain and terrain['contact_band_m'] <= 0:
        raise ValueError('contact band must be positive')
    return actual


def validate_weather(settings, default_seed):
    """Return actual weather variants (each with every field) or None."""
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) - {'variants'}:
        raise ValueError('scene.weather must be null or {"variants": [...]}')
    variants = settings.get('variants')
    if not isinstance(variants, list) or not variants:
        raise ValueError('scene.weather.variants must be a nonempty list')
    fields = {row['key']: row for row in WEATHER_FIELDS}
    result, seen = [], set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError('weather variants must be objects')
        extra = set(variant) - {'id', 'preset', 'seed', *fields}
        if extra:
            raise ValueError('unknown weather settings: ' + ', '.join(sorted(extra)))
        preset = variant.get('preset')
        if preset not in WEATHER_PRESETS:
            raise ValueError('weather preset must be one of ' + ', '.join(WEATHER_PRESETS))
        identifier = variant.get('id', preset)
        if not isinstance(identifier, str) or not identifier.replace('_', '').isalnum() or identifier in seen:
            raise ValueError('weather variant ids must be unique alphanumeric/underscore names')
        seen.add(identifier)
        actual = dict(id=identifier, preset=preset,
                      seed=_seed(variant.get('seed', (default_seed + 5101 + index) % 2**32), 'weather seed'))
        defaults = WEATHER_DEFAULTS[preset]
        for key, row in fields.items():
            actual[key] = _number(variant.get(key, defaults[key]), 'scene.weather.%s.%s' % (identifier, key), row['min'], row['max'])
        if preset in ('clear', 'overcast') and actual['precip_density'] > 0:
            raise ValueError('weather %s: precipitation requires the rain or snow preset' % identifier)
        result.append(actual)
    return result


def default_weather():
    return dict(variants=[dict(id=p, preset=p) for p in WEATHER_PRESETS])


def environment_capabilities():
    from .catalog import catalogue
    return deepcopy(catalogue())
