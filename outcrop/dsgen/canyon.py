"""Canyon exposure: two facing walls, an optional bench on each wall, a winding floor and plateaus.

A per-column profile chain (profiles.py) runs across the valley: left plateau, crest, wall
(upper lift, bench, lower lift), toe, floor, toe, right wall, crest, right plateau. The plateau
stays at the nominal height; the canyon depth tapers beyond the geology tile so the valley
shallows into the plateau instead of ending at a cut. Both walls are ordinary bare surfaces of
the same source sheet and pass through the native geological cut. This is an editable form;
it makes no claim to model fluvial incision.
"""
import math

import numpy as np

from .exposure import EDGE_ROOM_M, FACE_CHECK_SLOPE_DEG, FACE_MIN_RADIUS_M, _waves

WALL_ROWS = ('crest-left', 'wall-left-upper', 'bench-left-out', 'bench-left', 'bench-left-in', 'wall-left-lower',
             'toe-left', 'toe-right', 'wall-right-lower', 'bench-right-in', 'bench-right', 'bench-right-out',
             'wall-right-upper', 'crest-right')
# Redraw cap for sample(): a 30 m tile (the smallest scene.sample_spec allows) accepts ~0.15 % of
# draws, so 10^4 draws miss with probability ~4e-7.
MAX_DRAWS = 10000


def sample(g, tile):
    """Draw the canyon from the fixed ranges, redrawing whole sets whose rims leave the tile.

    The first draw that fits is returned unchanged, so seeds that fitted before keep their
    parameters; small tiles accept only the narrow, shallow part of the same ranges. The rim
    limit itself is _fit's, unchanged; after MAX_DRAWS misses it fails with that error.
    """
    T, (mx, my), g0 = tile['size_m'], tile['min_xy_m'], tile['ground_z_m']
    relief = tile['max_relief_m']
    wave = lambda amp, wl=(18., 50.): [float(amp), float(g.uniform(*wl)), float(g.uniform(0., 2 * math.pi))]
    for _ in range(MAX_DRAWS):
        depth = float(g.uniform(.6, .85)) * (relief - 2.)
        p = {'strike_deg': float(g.uniform(-30., 30.)), 'centre_xy_m': [mx + T / 2, my + T / 2],
             'domain_half_m': T / 2 + mx, 'plateau_m': depth + 1., 'depth_m': depth,
             'floor_width_m': float(g.uniform(4., 9.)), 'wall_dip_deg': float(g.uniform(65., 80.)),
             'wall_asymmetry_deg': float(g.uniform(-6., 6.)), 'bench_fraction': float(g.uniform(.4, .6)),
             'bench_width_m': float(g.uniform(1., 2.5)), 'benches': 1,
             'crest_radius_m': float(g.uniform(1.5, 3.)), 'toe_radius_m': float(g.uniform(.8, 1.6)),
             'meander_m': float(g.uniform(1., 3.)), 'meander_wavelength_m': float(g.uniform(35., 60.)),
             'meander_phase_rad': float(g.uniform(0., 2 * math.pi)),
             'relax_start_m': T / 2 - 3., 'relax_m': 30., 'repose_deg': float(g.uniform(32., 37.)),
             'taper_start_m': T / 2 + 4., 'reach_m': float(g.uniform(30., 50.)), 'end_depth_fraction': .15,
             'waves': {'depth': wave(g.uniform(.04, .1)), 'dip': wave(g.uniform(2., 5.))},
             'relief': {'seed': int(g.integers(1, 2 ** 30)), 'lowland_m': float(g.uniform(.1, .3)),
                        'upland_m': float(g.uniform(.1, .3))},
             'variation': 1.}
        if _half_width(p) <= _rim_limit(T):
            break
    _fit(p, T)
    c, s = math.cos(math.radians(p['strike_deg'])), math.sin(math.radians(p['strike_deg']))
    p['view_direction'] = [s, -c, 0.]
    p['rock_xy'] = [[mx, my], [mx + T, my + T]]
    p['ground_z_m'] = g0
    p['edge_z_m'] = p['plateau_m'] - .5
    return p


def configure(p, shape, tile):
    """Apply explicit canyon controls, then re-check the tile fit and the fold limit unchanged."""
    from copy import deepcopy
    for key, value in shape.items():
        if key in ('strike_deg', 'dip_direction_deg'):
            continue
        if key in ('waves', 'relief'):
            p[key].update(deepcopy(value))
        else:
            p[key] = deepcopy(value)
    if 'strike_deg' in shape or 'dip_direction_deg' in shape:
        bearing = shape.get('strike_deg', (shape.get('dip_direction_deg', 90.) - 90.) % 360.)
        p['strike_deg'] = 90. - bearing
    relief = tile['max_relief_m']
    if not 0 < p['depth_m'] <= relief - 2.:
        raise ValueError('canyon depth must be in (0, %.1f] m (geological relief with 2 m room)' % (relief - 2.))
    low, high = p['wall_dip_deg'] - abs(p['wall_asymmetry_deg']), p['wall_dip_deg'] + abs(p['wall_asymmetry_deg'])
    if low < 30. or high > 86.:
        raise ValueError('canyon wall dips must stay within [30, 86] degrees')
    p['plateau_m'] = p['depth_m'] + 1.
    p['edge_z_m'] = p['plateau_m'] - .5
    p['variation'] = float(p.get('variation', 1.))
    _fit(p, tile['size_m'])
    angle = math.radians(p['strike_deg'])
    p['view_direction'] = [math.sin(angle), -math.cos(angle), 0.]


def _half_width(p):
    """Plan half width from the valley axis to the outer edge of a rim crest."""
    return (p['floor_width_m'] / 2 + p['toe_radius_m'] + p['crest_radius_m']
            + p['depth_m'] / math.tan(math.radians(p['wall_dip_deg'] - abs(p['wall_asymmetry_deg'])))
            + p['bench_width_m'] * p['benches'] + p['meander_m'])


def _rim_limit(T):
    return T / 2 - EDGE_ROOM_M - 1.


def _fit(p, T):
    """Keep both rims inside the tile and the plan curvature of the walls above the fold limit."""
    if _half_width(p) > _rim_limit(T):
        raise ValueError('canyon rims (half width %.1fm) exceed the geology tile; increase tile_m or reduce '
                         'floor width, depth, bench width or meander' % _half_width(p))
    c, s = math.cos(math.radians(p['strike_deg'])), math.sin(math.radians(p['strike_deg']))
    U = p['domain_half_m'] * (abs(c) + abs(s))
    u = np.linspace(-U, U, int(2 * U / .25) + 1)
    for _ in range(40):
        if min_radius(p, u) >= FACE_MIN_RADIUS_M:
            return
        p['variation'] *= .85
    raise ValueError('canyon plan curvature does not converge above the fold limit')


def chain(p, u, coarse=1.):
    from .profiles import arc, end_state, line
    v = p['variation']
    e = .07 * coarse
    t = np.clip((np.abs(u) - p['taper_start_m']) / p['reach_m'], 0, 1)
    taper = 1 - t * t * (3 - 2 * t)
    full = p['depth_m'] * (1 + _waves(p['waves']['depth'], u, v))
    D = full * (p['end_depth_fraction'] + (1 - p['end_depth_fraction']) * taper)
    q = np.clip(D / p['depth_m'], .05, 1.)
    r = np.clip((np.abs(u) - p['relax_start_m']) / p['relax_m'], 0, 1)
    r = r * r * (3 - 2 * r)
    rad = math.radians
    dip = p['wall_dip_deg'] + _waves(p['waves']['dip'], u, v)
    walls = []
    for side in (-1., 1.):
        alpha = np.minimum(dip + side * p['wall_asymmetry_deg'], 86.)
        walls.append(np.radians(alpha + (p['repose_deg'] - alpha) * r))
    aL, aR = walls
    Rc, Rt = p['crest_radius_m'] * q, p['toe_radius_m'] * q
    Rb = .5 * q
    nb = p['benches']
    bw = p['bench_width_m'] * q * (1 - r)
    f = p['bench_fraction']

    def lifts(alpha):
        drop = D - (Rc + Rt + 2 * Rb * nb) * (1 - np.cos(alpha))
        F = drop / np.sin(alpha)
        if (F <= .1).any():
            raise ValueError('canyon fillets leave no wall face')
        return F * f, F * (1 - f)

    upL, lowL = lifts(aL)
    upR, lowR = lifts(aR)
    bench = lambda name, alpha, width, up: (
        [arc(Rb, 0., e, name + '-out'), line(np.maximum(width, .05), 0., e, name), arc(Rb, alpha, e, name + '-in')]
        if nb else [])
    left = [line(70., 0., 1. * coarse, 'plateau-left-far'), line(4., 0., e, 'plateau-left-near'),
            arc(Rc, -aL, e, 'crest-left'), line(upL, -aL, e, 'wall-left-upper'),
            *bench('bench-left', -aL, bw, False), line(lowL, -aL, e, 'wall-left-lower'), arc(Rt, 0., e, 'toe-left')]
    floor = [line(p['floor_width_m'] * (1 - .3 * r), 0., .3 * coarse, 'floor')]
    right = [arc(Rt, aR, e, 'toe-right'), line(lowR, aR, e, 'wall-right-lower'),
             *bench('bench-right', aR, bw, True), line(upR, aR, e, 'wall-right-upper'), arc(Rc, 0., e, 'crest-right'),
             line(4., 0., e, 'plateau-right-near'), line(70., 0., 1. * coarse, 'plateau-right-far')]
    z0 = np.full_like(u, p['plateau_m'])
    w_floor_start, _, _ = end_state(left, np.zeros_like(u), z0, np.zeros_like(u))
    axis = p['meander_m'] * v * np.sin(2 * math.pi * u / p['meander_wavelength_m'] + p['meander_phase_rad'])
    shift = axis - (w_floor_start + p['floor_width_m'] * (1 - .3 * r) / 2)
    return left + floor + right, shift, p['plateau_m']


def min_radius(p, u):
    from .profiles import trace
    segments, shift, z0 = chain(p, u, coarse=4.)
    tr = trace(segments, np.zeros_like(u), np.full_like(u, z0), np.zeros_like(u))
    rows = np.isin(tr['rows'], WALL_ROWS)
    w = tr['w'][rows] + shift[None]
    steep = np.abs(tr['angle'][rows]) > math.radians(FACE_CHECK_SLOPE_DEG)
    du = u[1] - u[0]
    w1 = np.gradient(w, du, axis=1)
    w2 = np.gradient(w1, du, axis=1)
    radius = (1 + w1 ** 2) ** 1.5 / np.maximum(np.abs(w2), 1e-12)
    return float(np.where(steep, radius, np.inf)[:, 2:-2].min())


def sheet(p, edge=.07, smooth_m=1.):
    from .exposure import chain_sheet
    rows_of = lambda tr, rows: ((tr['s'][rows['crest-left']][None] - 3., 4.),
                                (tr['s'][rows['plateau-right-near']][None] + 2., tr['s'][-1:] - 8.))
    built = chain_sheet(p, lambda q, u, coarse=1.: chain(q, u, coarse), edge, smooth_m, relief=rows_of)
    return built
