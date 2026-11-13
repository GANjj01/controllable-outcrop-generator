from copy import deepcopy
import math

from outcrop.modeling.kernel.types import Budget


_DEFAULTS = {
    'bluff': dict(half_m=60., spacing_m=.75, far_m=3000.),
    'escarpment': dict(half_m=40., spacing_m=4., far_m=3000.),
    'escarpment_ridged': dict(half_m=40., spacing_m=4., far_m=3000.),
    'roadcut': dict(half_m=40., spacing_m=4., far_m=3000.),
    'coast': dict(half_m=150., spacing_m=5., far_m=3000.),
    'canyon': dict(half_m=40., spacing_m=4., far_m=3000.),
}


def validate_ground(settings):
    """Validate a complete explicit ground setting; None keeps native defaults."""
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) != {'half_m', 'spacing_m', 'far_m'}:
        raise ValueError('scene.ground must be null or contain half_m, spacing_m and far_m')
    result = {}
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('scene.ground.' + name + ' must be positive and finite')
        result[name] = float(value)
    if result['far_m'] <= result['half_m']:
        raise ValueError('scene.ground.far_m must exceed half_m')
    if result['spacing_m'] > result['half_m']:
        raise ValueError('scene.ground.spacing_m must not exceed half_m; retain at least three near-grid samples')
    return result


def check_ground_budget(kind, settings):
    """Use the existing native geometry face budget before allocating a grid."""
    budget = Budget()
    half, spacing, far = (settings[name] for name in ('half_m', 'spacing_m', 'far_m'))
    ratio = half / spacing
    if not math.isfinite(ratio) or ratio > budget.max_faces:
        raise ValueError('scene.ground exceeds native geometry max_faces budget (%d)' % budget.max_faces)
    if kind == 'coast':
        if half <= 5.:
            raise ValueError('scene.ground.half_m must exceed the native coastal 5m shore transition')
        columns = math.ceil(2 * ratio) + 3
        rows = math.ceil((half - 5.) / spacing) + 2
    else:
        outer = int((math.log(far) - math.log(half)) / math.log(1.15)) + 1
        columns = rows = int(2 * ratio) + 1 + 2 * outer
    faces = 2 * (rows - 1) * (columns - 1)
    if faces > budget.max_faces:
        raise ValueError('scene.ground requires %d faces, exceeding native geometry max_faces budget (%d)'
                         % (faces, budget.max_faces))


def ground_capabilities(kind):
    if kind not in _DEFAULTS:
        raise ValueError('No native ground controls for exposure ' + str(kind))
    fields = [dict(key='half_m', label='near-ground half width', unit='m', type='number', min=.001, step=1.),
              dict(key='spacing_m', label='near-ground sample spacing', unit='m', type='number', min=.001, step=.1),
              dict(key='far_m', label='far ground limit', unit='m', type='number', min=.001, step=10.)]
    notes = ['These parameters set the extent and sampling of the native ground; they do not change the outcrop surface or the geological labels and do not create any vegetation or environment.',
             'The far limit must exceed the near half width and the spacing must not exceed it. The mesh uses the native face budget.',
             'null keeps the full default arrays of each landform.']
    if kind == 'coast':
        fields[0]['min'] = 5.001
        notes.append('On coasts the near half width must exceed the native 5 m shore transition and the far limit must lie beyond the near ground; the static sea stays level.')
    else:
        notes.append('Beyond the near ground the native coarsening mesh and soil relief are kept; standalone outcrops keep the native toe transition to the ground.')
        notes.append('The far ground keeps the native 1.15 coarsening sequence, so the mesh edge may extend slightly beyond the target distance.')
    return dict(defaults=deepcopy(_DEFAULTS[kind]), fields=fields, notes=notes)


def scene_catalogue():
    current = [('native_ground', 'native soil ground with relief'), ('native_scarp', 'native cliff slope and far ground'),
               ('native_roadcut', 'native road cut'), ('native_coast', 'native coast with a static sea')]
    result = [dict(id=identifier, label=label, available=True) for identifier, label in current]
    # The 44 environment presets: availability means a real recipe consumes the parameters.
    from .env.catalog import catalogue
    for row in catalogue()['presets']:
        entry = dict(id=row['id'], label=row['label'], category=row['category'], available=row['available'])
        if not row['available']:
            entry['reason'] = row['reason']
        else:
            entry['stage'] = 'available through scene.environment in a full JSON configuration only'
        result.append(entry)
    return result
