"""Translate chronological recipe fault records into native ModelSpec records."""
from copy import deepcopy


def validate_faults(faults):
    if faults is None:
        return
    if not isinstance(faults, list):
        raise ValueError('geometry.faults must be null or an ordered list')
    identifiers = set()
    for index, fault in enumerate(faults):
        path = 'geometry.faults[%d]' % index
        if not isinstance(fault, dict):
            raise ValueError(path + ' must be an object')
        identifier = fault.get('id')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError(path + ' requires a unique nonempty fault id')
        identifiers.add(identifier)
        if not isinstance(fault.get('plane'), dict) or not fault['plane']:
            raise ValueError(path + '.plane must be a native plane object')
        if not isinstance(fault.get('motion'), dict) or not fault['motion']:
            raise ValueError(path + '.motion must be a native fault motion object')
        if fault.get('moving_side') not in ('positive', 'negative'):
            raise ValueError(path + '.moving_side must be positive or negative')


def apply_faults(raw, faults):
    """Append exact native faults after the recipe events, in declared list order.

    Native decoding/compilation remains responsible for field validity,
    kinematics, references, finite support, collisions and resource budgets.
    """
    validate_faults(faults)
    if faults is None:
        return
    existing = {obj['id'] for table in ('geometry', 'events') for obj in raw[table]}
    previous = [event['id'] for event in raw['events']]
    bodies = [body['id'] for body in raw['bodies']]
    for record in faults:
        event = deepcopy(record)
        plane = event.pop('plane')
        plane.setdefault('id', event['id'] + '-plane')
        plane.setdefault('kind', 'plane')
        if event['id'] in existing or plane['id'] in existing or event['id'] == plane['id']:
            raise ValueError('Fault event/plane ID collides with the generated model: ' + event['id'])
        if plane['kind'] != 'plane' or event.get('kind', 'fault') != 'fault':
            raise ValueError('Recipe faults require plane geometry and fault events')
        existing.update((event['id'], plane['id']))
        event.setdefault('kind', 'fault')
        event.setdefault('targets', list(bodies))
        event.setdefault('fault', plane['id'])
        event.setdefault('interfaces', 'hosted')
        event.setdefault('fabrics', 'hosted')
        dependencies = event.get('after', [])
        if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
            raise ValueError('Fault after must be a list of event IDs')
        event['after'] = list(dict.fromkeys(dependencies + previous))
        # Use the same existing proof as the random recipe only when its
        if (set(event['targets']) == set(bodies) and 'support' not in event
                and 'extent_m' not in plane and 'overlap_policy' not in event
                and event['motion'].get('opening_m', 0) == 0):
            event.setdefault('collision_proof', 'rigid_partition')
        raw['geometry'].append(plane)
        raw['events'].append(event)
        previous = [event['id']]
