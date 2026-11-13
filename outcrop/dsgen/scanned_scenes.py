"""Explicit local catalogue of the three accepted scanned scene templates."""
from copy import deepcopy
from pathlib import Path


PROJECT = ''
PRESETS = {
    'river_valley': dict(label='river valley', project=PROJECT,
        lit='',
        labels='',
        source_offset_enu_m=[-24., -24., -6.369671168578553], max_footprint_m=32.),
    'gobi': dict(label='gobi yardang', project=PROJECT,
        lit='',
        labels='',
        source_offset_enu_m=[-24., -24., -6.369671168578553], max_footprint_m=32.),
    'sea_cliff': dict(label='sea cliff', project=PROJECT,
        lit='',
        labels='',
        source_offset_enu_m=[-24., -24., -6.369671168578553], max_footprint_m=32.),
}


def validate_preset(value):
    """Validate a selection without requiring this computer's asset installation."""
    if value is not None and (not isinstance(value, str) or value not in PRESETS):
        raise ValueError('scene.scanned_preset must be null or one of ' + ', '.join(PRESETS))
    return value


def catalogue():
    """Report installation from the named project and maps only; never scan Content."""
    result = []
    for identifier, preset in PRESETS.items():
        project = Path(preset['project'])
        maps = [project.parent / 'Content' / (preset[key].removeprefix('/Game/') + '.umap')
                for key in ('lit', 'labels')]
        result.append(dict(id=identifier, **deepcopy(preset),
                           available=project.is_file() and all(path.is_file() for path in maps)))
    return result
