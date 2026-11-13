"""Geological volume of a sandbox tile (v2 ModelSpec) for the three rock classes.

The volume is independent of exposure: exposure.py decides how it is laid bare (bluff, escarpment, roadcut, ...). Rock
classes: group_a, group_b, group_c.

``family`` is one of the three classes only; subclasses are template variants inside a class: group_a  layered
group_a_kind_1/group_a_kind_2 succession (optionally chevron-folded, rare group_b_09 dyke) group_b      setting
``setting_1``: lava flows (group_b_01/group_b_02/group_b_03) with pyroclastic group_b_04/breccia, feeder dykes, sills;
setting ``setting_2``: plutonic body (group_b_06/group_b_07/group_b_08, faceted roof) intruding a group_a or group_c
host, cut by hypabyssal dykes (group_b_10/group_b_09) group_c  group_c_kind_1
(group_c_01/group_c_02/group_c_03/group_c_04/group_c_07) and group_c_kind_2 (group_c_05/group_c_06) units with a
foliation fabric, often chevron-folded, late dykes All share joint sets and faults. Default boundaries are planar
(beds, limbs, dykes, pluton roof facets, faults, joints)."""
import json
import math
from pathlib import Path

import numpy as np

from .lithology import ORDER, lithology_records
from .legacy import TABLE as _legacy_table, name as _legacy_name, translate_stratigraphy as _legacy_stratigraphy

GROUP_A_ROCKS = ['group_a_01', 'group_a_02', 'group_a_03', 'group_a_04', 'group_a_05', 'group_a_06', 'group_a_07']
# Rough relative frequencies in group_a_kind_1/group_a_kind_2 successions.
WEIGHTS = np.array([.26, .22, .14, .16, .10, .05, .07])
GROUP_C_ROCKS = ['group_c_01', 'group_c_02', 'group_c_03', 'group_c_04', 'group_c_05', 'group_c_06', 'group_c_07']
GROUP_C_WEIGHTS = np.array([.12, .12, .22, .20, .12, .12, .10])
FLOWS = ['group_b_01', 'group_b_02', 'group_b_03']
FLOW_WEIGHTS = np.array([.5, .3, .2])
GROUP_B_KIND_2_ROCKS = ['group_b_04', 'group_b_05']
PLUTONS = ['group_b_06', 'group_b_07', 'group_b_08']
PLUTON_WEIGHTS = np.array([.6, .25, .15])
# Dyke rocks per template (setting_1 = group_b/setting_1 scene, setting_2 = group_b/setting_2 scene).
DYKES = {'group_a': (['group_b_09', 'group_b_01'], [.7, .3]),
         'setting_1': (['group_b_01', 'group_b_09', 'group_b_02'], [.4, .4, .2]),
         'setting_2': (['group_b_10', 'group_b_09', 'group_b_06'], [.45, .35, .2]),
         'group_c': (['group_b_10', 'group_b_09', 'group_b_06'], [.4, .4, .2])}
FAMILIES = ('group_a', 'group_b', 'group_c')
FAMILY_WEIGHTS = {'group_a': 1 / 3, 'group_b': 1 / 3, 'group_c': 1 / 3}
GROUP_B_SETTINGS = ('setting_1', 'setting_2')
# Bed levels on neighbouring fold limbs are offset by this much so that bed contacts of two limbs
# never meet exactly on the axial plane (three planes through one line give degenerate cells).
LIMB_EPS_M = .003
# Sandbox tile: the geology fills a fixed tile (default 40 m) plus margins, independent of how it
# is exposed. Ground level sits GROUND_Z_M above the domain floor (buried rock below it); exposed
# relief reaches at most MAX_RELIEF_M above ground. The margins leave room for outward surface
# relief and for rigid fault blocks (throw-high empty band at the top, heave-wide at the sides).
TILE_M = 40.
TILE_MARGIN_M = 4.
GROUND_Z_M = 6.
MAX_RELIEF_M = 15.
TOP_MARGIN_M = 6.


def _unit(dip_deg, dip_dir_deg):
    d, a = math.radians(dip_deg), math.radians(dip_dir_deg)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


def _strike_normal(strike_deg, dip_deg):
    # RHR: dip direction = strike + 90; returns the pole (upward hemisphere).
    return _unit(dip_deg, (strike_deg + 90) % 360)


def _choose(rng, names, weights):
    w = np.asarray(weights, float)
    return names[rng.choice(len(names), p=w / w.sum())]


def _succession(rng, family, span, mean_t, t_max=4.):
    """Unit lithologies and thicknesses filling ``span`` metres of stratigraphy."""
    thick, beds, prev = [], [], None
    if family == 'group_a':
        while sum(thick) < span:
            thick.append(float(np.clip(rng.lognormal(math.log(mean_t), .45), .25, t_max)))
        for _ in thick:
            while True:
                lith = GROUP_A_ROCKS[rng.choice(len(GROUP_A_ROCKS), p=WEIGHTS)]
                if lith != prev or rng.random() < .15:
                    break
            beds.append(lith)
            prev = lith
        return beds, thick
    if family == 'group_b_setting_1':
        # A dominant magma type with occasional other flows; pyroclastic beds between flows.
        main = _choose(rng, FLOWS, FLOW_WEIGHTS)
        while sum(thick) < span:
            if prev in FLOWS and rng.random() < .45:
                lith = _choose(rng, GROUP_B_KIND_2_ROCKS, [.7, .3])
                t = rng.lognormal(math.log(.6 * mean_t), .5)
            else:
                lith = main if rng.random() < .7 else _choose(rng, FLOWS, FLOW_WEIGHTS)
                t = rng.lognormal(math.log(1.4 * mean_t), .4)
            beds.append(lith)
            thick.append(float(np.clip(t, .3, max(5., t_max))))
            prev = lith
        return beds, thick
    if family == 'group_c':
        while sum(thick) < span:
            while True:
                lith = _choose(rng, GROUP_C_ROCKS, GROUP_C_WEIGHTS)
                if lith != prev or rng.random() < .1:
                    break
            beds.append(lith)
            thick.append(float(np.clip(rng.lognormal(math.log(mean_t), .5), .25, t_max)))
            prev = lith
        return beds, thick
    raise ValueError('no layered succession for family ' + family)


def _chevron(rng, corners):
    """Chevron (kink) fold: limb panels separated by vertical axial planes, mirrored planar limbs.
    Returns (axis strike deg, profile unit vector u, panel edges along u, limb normals, offsets)."""
    axis = float(rng.uniform(0, 180))
    # Profile direction: horizontal, normal to the (vertical) axial planes, azimuth axis + 90.
    u = np.array([math.sin(math.radians(axis + 90)), math.cos(math.radians(axis + 90)), 0.])
    z = np.array([0., 0., 1.])
    delta = math.radians(float(rng.uniform(25, 60)))  # limb dip
    pu = corners @ u
    lo, hi = float(pu.min()), float(pu.max())
    wavelength = float(rng.uniform(.5, 1.2)) * (hi - lo)
    first = lo + float(rng.uniform(.25, .6)) * wavelength / 2
    # At most two hinges (three limbs): each limb multiplies the number of v2 bodies.
    hinges = [h for h in first + wavelength / 2 * np.arange(4) if lo + 2 < h < hi - 2][:2]
    edges = [lo - 1.] + hinges + [hi + 1.]
    sign = 1 if rng.random() < .5 else -1
    normals, offsets, off = [], [], 0.
    for k in range(len(edges) - 1):
        n = sign * math.sin(delta) * u + math.cos(delta) * z
        if k > 0:
            prev = -sign * math.sin(delta) * u + math.cos(delta) * z
            # Bed coordinate continuous across the axial plane u.p = edge.
            off += float((n - prev) @ (edges[k] * u))
        normals.append(n)
        offsets.append(off + LIMB_EPS_M * k)
        sign = -sign
    return axis, u, edges, normals, offsets, math.degrees(delta)


def _dyke_strike(rng):
    # Cliff face runs E-W: keep dykes at least 25 deg from it so they cut the exposure.
    return float(rng.choice([rng.uniform(-65, 65), rng.uniform(115, 245)]) % 360)


def sample_spec(seed, *, max_beds=14, family=None, family_weights=None, group_b_setting=None, tile_m=TILE_M,
                stratigraphy=None, fracture_network=None, fault_count=None, igneous_setting=None):
    """Geological volume of one sandbox tile (v2 ModelSpec). Exposure is chosen separately
    (exposure.sample_exposure), so the same volume can be exposed in different ways."""
    # the former names of frozen callers read through the legacy table: family, the group_b setting (also under its
    # former keyword igneous_setting), the family_weights keys and the stratigraphy's young_lithologies
    if igneous_setting is not None:
        if group_b_setting is not None:
            raise ValueError('give group_b_setting or its former keyword igneous_setting, not both')
        group_b_setting = igneous_setting
    family = _legacy_name(family)
    group_b_setting = _legacy_table['settings'].get(group_b_setting, group_b_setting)
    if family_weights is not None:
        family_weights = {_legacy_name(k): v for k, v in family_weights.items()}
    stratigraphy = _legacy_stratigraphy(stratigraphy)
    rng = np.random.default_rng(seed)
    if family is None:
        weights = family_weights or FAMILY_WEIGHTS
        names = [f for f in FAMILIES if weights.get(f, 0) > 0]
        family = _choose(np.random.default_rng([seed, 1]), names, [weights[f] for f in names])
    if family not in FAMILIES:
        raise ValueError('unknown family %r (choose from %s)' % (family, ', '.join(FAMILIES)))
    geo = np.random.default_rng([seed, 2])  # family-specific choices; group_a draws stay as before
    if not 30. <= tile_m <= 60.:
        raise ValueError('tile size must be 30-60 m')
    m, g0 = TILE_MARGIN_M, GROUND_Z_M
    W = D = tile_m + 2 * m
    H = g0 + MAX_RELIEF_M + TOP_MARGIN_M
    lo, hi = np.array([0., 0., 0.]), np.array([W, D, H])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    dip, dip_dir = float(rng.uniform(0, 30)), float(rng.uniform(0, 360))
    setting = None
    if family == 'group_b':
        setting = group_b_setting or GROUP_B_SETTINGS[int(geo.random() < .5)]
        if setting not in GROUP_B_SETTINGS:
            raise ValueError('group_b setting must be setting_1 or setting_2')
    elif group_b_setting is not None:
        raise ValueError('group_b_setting applies to the group_b family only')
    template = setting or family
    # Layered host: the succession itself (setting_1 pile) or the country rock of an intrusion.
    host = {'setting_1': 'group_b_setting_1', 'setting_2': 'group_a' if geo.random() < .5 else 'group_c'}.get(template, family)
    if host == 'group_b_setting_1':
        dip = float(geo.uniform(0, 20))
    elif host == 'group_c':
        dip = float(geo.uniform(10, 65))
    # Folded country rock under a pluton multiplies the v2 arrangement (limbs x beds x roof facets):
    # setting_2 scenes keep a planar host.
    folded = template != 'setting_2' and geo.random() < {'group_a': .15, 'group_b_setting_1': 0., 'group_c': .55}[host]
    if stratigraphy is not None:
        # The shared coordinate is an alternative to the chevron template.
        folded = False
    geometry = [{'id': 'footprint', 'kind': 'box', 'min_m': lo.tolist(), 'max_m': hi.tolist()}]
    features, meta = [], {}
    if folded:
        axis, u, edges, normals, offsets, limb_dip = _chevron(geo, corners)
        s = np.concatenate([corners @ n - o for n, o in zip(normals, offsets)])
        meta.update(fold={'kind': 'chevron', 'axis_strike_deg': axis, 'limb_dip_deg': limb_dip,
                          'axial_planes_u': edges[1:-1], 'profile_direction': u.tolist()})
    else:
        normal = _unit(dip, dip_dir)
        s = corners @ normal
    span = s.max() - s.min() + 2.
    t_max = 4.
    if folded:
        # Bodies = beds x limbs and every bed adds two planes per limb to the v2 arrangement:
        # fewer, thicker units in folded scenes.
        max_beds = min(max_beds, 7)
        t_max = max(4., 1.6 * span / max_beds)
    mean_t = max(span / max_beds, float(rng.uniform(.5, 1.6)))
    beds, thick = _succession(rng, host, span, mean_t, t_max)
    if isinstance(stratigraphy, dict) and ('layer_count' in stratigraphy or stratigraphy.get('kind', 'nonparallel') == 'nonparallel'):
        # opt-in mode with two units; callers can explicitly request more.
        count = stratigraphy.get('layer_count', 2)
        if type(count) is not int or count < 1:
            raise ValueError('layer_count must be a positive integer')
        selected = np.arange(count) * len(beds) // count
        beds = [beds[i] for i in selected]
        original = np.asarray(thick, float)
        chosen = original[selected]
        thick = chosen / chosen.sum() * original.sum()
    levels = s.min() - 1. + np.r_[0, np.cumsum(thick)]
    bodies, unit_ids = [], []
    if folded:
        for k, (n, o) in enumerate(zip(normals, offsets)):
            geometry += [{'id': 'bed-coordinate-l%d' % k, 'kind': 'scalar_plane', 'normal': n.tolist(), 'offset_m': float(o), 'unit': 'm'},
                         {'id': 'limb-lo-%d' % k, 'kind': 'halfspace', 'normal': (-u).tolist(), 'offset_m': float(-edges[k])},
                         {'id': 'limb-hi-%d' % k, 'kind': 'halfspace', 'normal': u.tolist(), 'offset_m': float(edges[k + 1])}]
            limb = []
            for i, lith in enumerate(beds):
                bid = 'bed-%02d-l%d' % (i, k)
                geometry += [{'id': 'band%d-l%d' % (i, k), 'kind': 'band', 'field': 'bed-coordinate-l%d' % k,
                              'min': float(levels[i]), 'max': float(levels[i + 1]), 'unit': 'm'},
                             {'id': 'rock%d-l%d' % (i, k), 'kind': 'intersection',
                              'operands': ['footprint', 'limb-lo-%d' % k, 'limb-hi-%d' % k, 'band%d-l%d' % (i, k)]}]
                bodies.append({'id': bid, 'geometry': 'rock%d-l%d' % (i, k),
                               'material': {'kind': 'homogeneous', 'lithology': lith}, 'unit_id': 'unit-%02d' % i})
                limb.append(bid)
            features.append({'id': 'bedding-l%d' % k, 'kind': 'bedding', 'hosts': limb,
                             'fabric': {'kind': 'scalar', 'field': 'bed-coordinate-l%d' % k}})
            unit_ids += limb
    else:
        geometry.append({'id': 'bed-coordinate', 'kind': 'scalar_plane', 'normal': normal.tolist(), 'offset_m': 0., 'unit': 'm'})
        for i, lith in enumerate(beds):
            geometry.append({'id': 'band%d' % i, 'kind': 'band', 'field': 'bed-coordinate',
                             'min': float(levels[i]), 'max': float(levels[i + 1]), 'unit': 'm'})
            geometry.append({'id': 'rock%d' % i, 'kind': 'intersection', 'operands': ['footprint', 'band%d' % i]})
            bodies.append({'id': 'bed-%02d' % i, 'geometry': 'rock%d' % i,
                           'material': {'kind': 'homogeneous', 'lithology': lith}, 'unit_id': 'unit-%02d' % i})
            unit_ids.append('bed-%02d' % i)
        features.append({'id': 'bedding', 'kind': 'bedding', 'hosts': unit_ids, 'fabric': {'kind': 'scalar', 'field': 'bed-coordinate'}})
    if host == 'group_c':
        # Foliation: near-parallel to compositional layering, or axial-planar in folded rocks.
        if folded:
            fn = _unit(float(geo.uniform(75, 90)), (axis + 90 + (180 if geo.random() < .5 else 0)) % 360)
        else:
            fn = normal + geo.normal(0, .08, 3)
        fn = fn / np.linalg.norm(fn)
        geometry.append({'id': 'foliation-coordinate', 'kind': 'scalar_plane', 'normal': fn.tolist(), 'offset_m': 0., 'unit': 'm'})
        features.append({'id': 'foliation', 'kind': 'foliation', 'hosts': list(unit_ids),
                         'fabric': {'kind': 'scalar', 'field': 'foliation-coordinate'}})
    # Visible surfaces come from the exposure layer and are labelled by v2 point queries; v2 only
    # needs a small exposure to complete its package: one vertical section through the domain.
    y = D / 2
    geometry.append({'id': 'section', 'kind': 'mesh', 'closed': False,
                     'vertices': [[.5, y, .5], [W - .5, y, .5], [W - .5, y, H - .5], [.5, y, H - .5]],
                     'faces': [[0, 1, 2], [0, 2, 3]]})
    meta['tile'] = {'size_m': tile_m, 'min_xy_m': [m, m], 'ground_z_m': g0, 'max_relief_m': MAX_RELIEF_M}

    # Intrusions: pluton first, then dykes and sills; each intrudes every body present at its stage.
    events, intrusions = [], []
    rock = list(unit_ids)

    def intrude(eid, body):
        event = {'id': eid, 'kind': 'intrusion', 'body': body['id'], 'targets': list(rock)}
        if events:
            event['after'] = [events[-1]['id']]
        events.append(event)
        bodies.append(body)
        rock.append(body['id'])
        intrusions.append({'id': body['id'], 'lithology': body['material']['lithology'], 'event': eid})

    if template == 'setting_2':
        lith = _choose(geo, PLUTONS, PLUTON_WEIGHTS)
        # Apex below the crest in most scenes, so the roof contact with the country rock crosses the cliff.
        apex = np.array([m + float(geo.uniform(.2, .8)) * tile_m, m + float(geo.uniform(.3, .9)) * tile_m,
                         g0 + float(geo.uniform(0., .75)) * MAX_RELIEF_M])
        ops = ['footprint']
        for k in range(int(geo.integers(2, 5))):
            facet = _unit(float(geo.uniform(8, 50)), float(geo.uniform(0, 360)))
            geometry.append({'id': 'pluton-roof-%d' % k, 'kind': 'halfspace', 'normal': facet.tolist(),
                             'offset_m': float(facet @ (apex + geo.normal(0, .8, 3)))})
            ops.append('pluton-roof-%d' % k)
        geometry.append({'id': 'pluton-0-region', 'kind': 'intersection', 'operands': ops})
        intrude('intrude-pluton-0', {'id': 'pluton-0', 'geometry': 'pluton-0-region',
                                     'material': {'kind': 'homogeneous', 'lithology': lith}, 'unit_id': 'pluton-0'})
    n_dykes = int(geo.choice({'group_a': [0, 0, 0, 0, 0, 1], 'setting_1': [0, 1, 1, 2], 'setting_2': [0, 1, 1, 2, 3],
                              'group_c': [0, 0, 1, 1, 2]}[template]))
    names, weights = DYKES[template]
    for k in range(n_dykes):
        dn = _strike_normal(_dyke_strike(geo), float(geo.uniform(65, 90)))
        centre = np.array([m + float(geo.uniform(.15, .85)) * tile_m, m + float(geo.uniform(.3, .7)) * tile_m,
                           g0 + float(geo.uniform(.2, .8)) * MAX_RELIEF_M])
        t = float(geo.lognormal(math.log(.9), .5))
        c = float(dn @ centre)
        geometry += [{'id': 'dyke-%d-coordinate' % k, 'kind': 'scalar_plane', 'normal': dn.tolist(), 'offset_m': 0., 'unit': 'm'},
                     {'id': 'dyke-%d-band' % k, 'kind': 'band', 'field': 'dyke-%d-coordinate' % k,
                      'min': c - t / 2, 'max': c + t / 2, 'unit': 'm'},
                     {'id': 'dyke-%d-region' % k, 'kind': 'intersection', 'operands': ['footprint', 'dyke-%d-band' % k]}]
        intrude('intrude-dyke-%d' % k, {'id': 'dyke-%d' % k, 'geometry': 'dyke-%d-region' % k,
                                        'material': {'kind': 'homogeneous', 'lithology': _choose(geo, names, weights)},
                                        'unit_id': 'dyke-%d' % k})
    if template == 'setting_1' and not folded and geo.random() < .3 and len(beds) > 2:
        # Sill: concordant sheet inside the pile, between existing bed contacts.
        i = int(geo.integers(1, len(beds) - 1))
        t = float(geo.uniform(.4, 1.5))
        mid = float(levels[i] + geo.uniform(.3, .7) * thick[i])
        geometry += [{'id': 'sill-0-band', 'kind': 'band', 'field': 'bed-coordinate', 'min': mid - t / 2, 'max': mid + t / 2, 'unit': 'm'},
                     {'id': 'sill-0-region', 'kind': 'intersection', 'operands': ['footprint', 'sill-0-band']}]
        intrude('intrude-sill-0', {'id': 'sill-0', 'geometry': 'sill-0-region',
                                   'material': {'kind': 'homogeneous', 'lithology': _choose(geo, ['group_b_09', 'group_b_01'], [.6, .4])},
                                   'unit_id': 'sill-0'})

    for j in range(int(rng.integers(1, 3))):
        strike = float(rng.uniform(0, 360))
        if folded:
            # Fractures nearly parallel to the axial planes give sliver cells against the limb
            # boundaries; keep them at least 20 deg away.
            while min(abs((strike - axis + 90) % 180 - 90), 90) < 20:
                strike = float(geo.uniform(0, 360))
        jdip = float(rng.uniform(70, 89))
        spacing = float(rng.uniform(1.2, 4.))
        pole = _strike_normal(strike, jdip)
        proj = corners @ pole
        count = int(min(14, max(1, (proj.max() - proj.min()) // spacing)))
        # Joints cut every rock body, intrusions included (jointed group_b_06, cooling joints in dykes).
        features.append({'id': 'joints-%d' % j, 'kind': 'joint_set', 'hosts': list(rock), 'support': 'footprint',
                         'orientation': {'strike_deg': strike, 'dip_deg': jdip},
                         'placement': {'kind': 'periodic', 'count': count, 'spacing_m': spacing,
                                       'start_m': float(proj.min() + spacing * rng.uniform(.3, .9))},
                         'extent_m': [200, 200]})
    if intrusions:
        # Intrusion bodies do not exist at the initial stage: the joint sets are created after the
        # last intrusion (and before faulting, so faults still offset them).
        joints = [f for f in features if f['kind'] == 'joint_set']
        for f in joints:
            f['at_stage'] = 'jointing'
        events.append({'id': 'jointing', 'kind': 'create_features', 'features': [f['id'] for f in joints],
                       'output_stage': 'jointing', 'after': [events[-1]['id']]})
    n_faults = int(rng.choice([0, 1, 1, 2]))
    if folded or intrusions:
        # Two faults across chevron limbs exceed the v2 per-cell plane budget (limbs x beds x fault
        # branches; one fault already takes 3-8 minutes), and two faults through a dyke-cut pile
        # ran past the 20-minute generation timeout: one fault in these scenes.
        n_faults = min(n_faults, 1)
    if fault_count is not None:
        if type(fault_count) is not int or fault_count < 0:
            raise ValueError('fault_count must be a nonnegative integer')
        n_faults = fault_count
    for k in range(n_faults):
        fid = 'fault-%d' % k
        point = [float(rng.uniform(m + .25 * tile_m, m + .75 * tile_m)), D / 2, float(rng.uniform(g0 + 1, g0 + MAX_RELIEF_M - 1))]
        geometry.append({'id': fid + '-plane', 'kind': 'plane', 'strike_deg': float(rng.uniform(0, 360)),
                         'dip_deg': float(rng.uniform(50, 85)), 'point_m': point})
        event = {'id': fid, 'kind': 'fault', 'targets': list(rock), 'fault': fid + '-plane',
                 'motion': {'kind': 'throw', 'throw_down_m': float(rng.uniform(.6, 3.)), 'pure_dip_slip': True},
                 'moving_side': str(rng.choice(['positive', 'negative'])), 'interfaces': 'hosted', 'fabrics': 'hosted',
                 # All bodies partition the footprint and every body is a target of an unbounded dip-slip fault.
                 'collision_proof': 'rigid_partition'}
        if events:
            event['after'] = [events[-1]['id']]
        events.append(event)
    lithologies = sorted({b['material']['lithology'] for b in bodies}, key=ORDER.__getitem__)   # the catalogue's order field
    spec = {
        'schema_version': 2, 'model_id': 'vog-%s-%d' % (template, seed),
        'coordinates': {'frame': 'ENU', 'length_unit': 'm', 'angle_unit': 'deg'},
        'domain': {'min_m': [0, 0, 0], 'max_m': [W, D, H]},
        'geometry': geometry, 'bodies': bodies,
        'lithologies': lithology_records(lithologies),
        'features': features, 'interfaces': [], 'events': events, 'relations': [], 'parameters': [],
        'exposures': [{'id': 'tile-section', 'kind': 'section', 'geometry': 'section'}],
        'meshing': {'max_cell_size_m': 20, 'max_position_error_m': .1},
        'sampling': {'root_seed': int(seed), 'sample_ids': ['a'], 'range_policy': 'uniform'},
        'metadata': {'generator': 'outcrop.dsgen.scene', 'schema': 'vog-geology-tile-1', 'seed': int(seed), 'family': family, 'group_b_setting': setting,
                     'host_family': {'group_b_setting_1': 'group_b'}.get(host, host),
                     'bed_dip_deg': None if folded else dip, 'bed_dip_direction_deg': None if folded else dip_dir,
                     'beds': beds,
                     'intrusions': intrusions, **meta},
    }
    if stratigraphy is not None:
        from .paper_geology import apply_stratigraphy
        apply_stratigraphy(spec, stratigraphy)
    if fracture_network is not None:
        from .paper_geology import apply_fracture_network
        apply_fracture_network(spec, fracture_network)
    return spec


def write_spec(seed, path, **kwargs):
    spec = sample_spec(seed, **kwargs)
    Path(path).write_text(json.dumps(spec, indent=1), encoding='utf8')
    return spec
