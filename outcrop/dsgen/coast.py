"""Minimal static coast exposure: a sea cliff with a wave-cut platform, a pocket beach and the sea.

Local frame as for the escarpment (u along the shore, w inland, z above the tile ground; the cliff
looks towards -w, the sea). Per column the profile chain runs from the sea floor: seabed -> rise ->
platform edge -> wave-cut platform (bare rock, gently rising inland across mean sea level) -> cliff
foot -> cliff face -> crest -> upland (soil) -> back to a low coastal plain above the sea. Headlands
and coves are long plan waves under the same fold guard as the escarpment.

Visible semantics added here (never geology): the sea surface (``water``) and a pocket beach
(``sediment``) on the platform in the deepest cove. Their pixels get geological truth from the rock
surface vertically below them (labels.bed_lookup), not from the water or sand surface. The sea is a
static opaque plane at mean sea level: no waves, tide, foam or transparency (minimal template).

Not done (recorded): an undercut wave-cut notch (an overhang would break the per-column profile
chain), sea stacks, tidal pools.
"""
import math

import numpy as np

from .exposure import EDGE_ROOM_M, FACE_MIN_RADIUS_M, _waves

# Wave-washed rows: bare rock at any slope (no soil); they also carry the sea floor / beach bed.
BED_ROWS = ('seabed-far', 'seabed', 'seabed-rise', 'platform-front', 'platform-edge', 'platform')
BARE_ROWS = ('platform-front', 'platform-edge', 'platform', 'foot', 'face')
FACE_ROWS = ('foot', 'face', 'crest', 'upland-near')
CHECK_SLOPE_DEG = 40.


def sample(g, tile):
    T, (mx, my), g0 = tile['size_m'], tile['min_xy_m'], tile['ground_z_m']
    strike = float(g.uniform(-25., 25.))
    wave = lambda amp, wl=(18., 60.): [float(amp), float(g.uniform(*wl)), float(g.uniform(0., 2 * math.pi))]
    platform_w = float(g.uniform(6., 12.))
    p = {'strike_deg': strike, 'centre_xy_m': [mx + T / 2, my + T / 2], 'domain_half_m': T / 2 + mx, 'ground_z_m': g0,
         'sea_level_m': 0.,  # local z of mean sea level (tile ground)
         'cliff_height_m': float(g.uniform(8., tile['max_relief_m'] - 2.)),
         'face_dip_deg': float(g.uniform(65., 84.)), 'foot_radius_m': float(g.uniform(.5, 1.)),
         'crest_radius_m': float(g.uniform(2., 4.)),
         'platform_width_m': platform_w, 'platform_slope_deg': float(g.uniform(2., 4.)),
         'platform_edge_z_m': float(g.uniform(-.4, -.15)),  # seaward edge under water, inland part emerges
         'seabed_depth_m': float(g.uniform(2., 4.)), 'seabed_slope_deg': float(g.uniform(6., 12.)),
         'upland_slope_deg': float(g.uniform(1., 5.)), 'back_slope_deg': float(g.uniform(8., 14.)), 'back_z_m': 1.,
         # cliff foot line: the platform and some sea inside the tile in front of it
         'foot_w_m': -T / 2 + EDGE_ROOM_M + platform_w + float(g.uniform(.05, .2)) * T,
         'relax_start_m': T / 2 - 3., 'relax_m': 30., 'repose_deg': float(g.uniform(32., 37.)),
         'taper_start_m': T / 2 + 6., 'reach_m': float(g.uniform(25., 45.)),
         'waves': {'foot': [wave(g.uniform(.6, 2.)) for _ in range(2)], 'height': wave(g.uniform(.05, .12)),
                   'dip': wave(g.uniform(2., 5.))},
         'relief': {'seed': int(g.integers(1, 2 ** 30)), 'lowland_m': 0., 'upland_m': float(g.uniform(.2, .6))},
         'beach': {'length_m': float(g.uniform(8., 16.)), 'width_m': float(g.uniform(3., min(6., platform_w - 1.))),
                   'thickness_m': float(g.uniform(.25, .5))},
         'variation': 1.}
    c, s = math.cos(math.radians(strike)), math.sin(math.radians(strike))
    U = p['domain_half_m'] * (abs(c) + abs(s))
    u = np.linspace(-U, U, int(2 * U / .25) + 1)
    for _ in range(40):
        if min_radius(p, u) >= FACE_MIN_RADIUS_M:
            break
        p['variation'] *= .85
    else:
        raise ValueError('coast plan curvature does not converge')
    # Pocket beach in the deepest cove inside the tile (largest landward foot offset).
    inner = np.abs(u) < T / 2 - EDGE_ROOM_M - p['beach']['length_m'] / 2
    p['beach']['u_m'] = float(u[inner][np.argmax(_waves(p['waves']['foot'], u[inner], p['variation']))])
    p['view_direction'] = [s, -c, 0.]
    p['rock_xy'] = [[mx, my], [mx + T, my + T]]
    return p


def chain(p, u, coarse=1.):
    """Profile chain per column (from the sea floor, local z) and the w shift."""
    from .profiles import arc, end_state, line
    rad = math.radians
    v = p['variation']
    e = .07 * coarse
    t = np.clip((np.abs(u) - p['taper_start_m']) / p['reach_m'], 0, 1)
    taper = 1 - t * t * (3 - 2 * t)
    full = p['cliff_height_m'] * (1 + _waves(p['waves']['height'], u, v))
    H = .3 + (full - .3) * taper
    q = np.clip(H / p['cliff_height_m'], .05, 1.)
    alpha = np.minimum(p['face_dip_deg'] + _waves(p['waves']['dip'], u, v), 86.)
    r = np.clip((np.abs(u) - p['relax_start_m']) / p['relax_m'], 0, 1)
    r = r * r * (3 - 2 * r)
    alpha = np.radians(alpha + (p['repose_deg'] - alpha) * r)
    sb, ps = rad(p['seabed_slope_deg']), rad(p['platform_slope_deg'])
    up, back = rad(p['upland_slope_deg']) * taper + 1e-3, rad(p['back_slope_deg'])
    z_low = p['sea_level_m'] - p['seabed_depth_m']
    # straight front slope so that the platform starts at platform_edge_z_m (the two fillets rise
    # R (1 - cos) and R |cos ps - cos sb|)
    climb = p['platform_edge_z_m'] - z_low - 8. * (1 - math.cos(sb)) - 6. * abs(math.cos(ps) - math.cos(sb))
    front = [line(70., 0., 1. * coarse, 'seabed-far'), line(10., 0., .3 * coarse, 'seabed'),
             arc(8., sb, .2 * coarse, 'seabed-rise'), line(max(climb / math.sin(sb), .5), sb, .2 * coarse, 'platform-front'),
             arc(6., ps, e, 'platform-edge'), line(p['platform_width_m'], ps, e, 'platform')]
    w_foot, z_foot, _ = end_state(front, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    Rf, Rc = p['foot_radius_m'] * q, p['crest_radius_m'] * q
    face_len = (H - Rf * (math.cos(ps) - np.cos(alpha)) - Rc * (np.cos(up) - np.cos(alpha))) / np.sin(alpha)
    if (face_len <= .05).any():
        raise ValueError('coast fillets leave no face')
    mid = [arc(Rf, alpha, e, 'foot'), line(face_len, alpha, e, 'face'), arc(Rc, up, e, 'crest'),
           line(3., up, e, 'upland-near'), line(10., up, .3 * coarse, 'upland-mid'),
           line(20., up, 1. * coarse, 'upland-far'), arc(10., -back, 1. * coarse, 'brow')]
    _, z_brow, _ = end_state(front + mid, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    drop = np.maximum(z_brow - p['back_z_m'], .5)
    Rb = np.minimum(40., .5 * drop / (1 - math.cos(back)))
    tail = [line(np.maximum((drop - Rb * (1 - math.cos(back))) / math.sin(back), .1), -back, 1. * coarse, 'descent'),
            arc(Rb, 0., 1. * coarse, 'base'), line(15., 0., 1. * coarse, 'back-flat')]
    shift = p['foot_w_m'] + _waves(p['waves']['foot'], u, v) - w_foot
    return front + mid + tail, shift, z_low


def min_radius(p, u):
    from .profiles import trace
    segs, shift, z_low = chain(p, u, coarse=4.)
    tr = trace(segs, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    rows = np.isin(tr['rows'], FACE_ROWS)
    w = tr['w'][rows] + shift[None]
    steep = tr['angle'][rows] > math.radians(CHECK_SLOPE_DEG)
    du = u[1] - u[0]
    w1 = np.gradient(w, du, axis=1)
    w2 = np.gradient(w1, du, axis=1)
    radius = (1 + w1 ** 2) ** 1.5 / np.maximum(np.abs(w2), 1e-12)
    return float(np.where(steep, radius, np.inf)[:, 2:-2].min())


def sheet(p, edge=.07, smooth_m=1.):
    from .exposure import chain_sheet
    return chain_sheet(p, chain, edge, smooth_m,
                       relief=lambda tr, rows: ((tr['s'][rows['seabed-far']][None], 1.),
                                                (tr['s'][rows['upland-near']][None] + 2., tr['s'][rows['brow']][None] - 8.)))


def far_exact(V, centre, beyond=500.):
    """Moves them < 0.5 m, far out."""
    far = np.abs(V[:, :2] - np.asarray(centre)).max(1) > beyond
    V = V.copy()
    V[far, :2] = np.round(V[far, :2])
    return V


def water_item(sh, p, rng, seed):
    """Static sea surface at mean sea level: a large flat plane seaward of the cliff foot, out to
    the horizon (no visible tile edge). Opaque, coloured by depth below it (shallow -> turquoise)."""
    from .surface import Item
    c, s = sh['frame']
    names = sh['trace']['rows']
    Vg = sh['V_grid']
    foot = np.nonzero(names == 'foot')[0][0]
    # landward limit: the cliff foot, where the platform is above the sea anyway
    w_lim = float(((Vg[foot, :, 0] - p['centre_xy_m'][0]) * -s + (Vg[foot, :, 1] - p['centre_xy_m'][1]) * c).max())
    z = p['ground_z_m'] + p['sea_level_m']
    us = np.r_[-3000., np.linspace(-120., 120., 97), 3000.]
    ws = np.r_[-3000., -600., np.linspace(-150., w_lim, 120)]
    U, W = np.meshgrid(us, ws)
    X = p['centre_xy_m'][0] + U * c - W * s
    Y = p['centre_xy_m'][1] + U * s + W * c
    V = far_exact(np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3), p['centre_xy_m'])
    from .profiles import grid_faces
    F = grid_faces(len(ws), len(us))
    # depth-tinted colour: the bed height under each vertex from the sheet (nearest column/row)
    deep, shallow = np.array([.05, .16, .2]), np.array([.16, .36, .36])
    bed = np.interp(W.ravel(), np.r_[-3000., -150., 0.], [-30., p['sea_level_m'] - p['seabed_depth_m'], p['platform_edge_z_m']])
    k = np.clip((p['sea_level_m'] - bed) / 4., 0, 1)[:, None]
    from .display import fbm
    colors = (shallow * (1 - k) + deep * k) * (1 + .08 * fbm(V * .05, seed + 90, 3))[:, None]
    return Item('sea', V, F, 'water', colors=np.clip(colors, .01, .9),
                provenance={'process': 'static opaque sea surface at mean sea level (minimal coast template)', 'seed': seed,
                            'sea_level_z_m': z})


def beach_item(sh, p, rng, seed):
    """Pocket beach: a sand lens on the platform at the cliff foot in the deepest cove, up to
    ``thickness_m`` thick at the foot, thinning seaward and along the shore, tucked under its rims."""
    from .surface import Item
    from .profiles import grid_faces
    names = sh['trace']['rows']
    Vg, u, s_arc = sh['V_grid'], sh['u'], sh['trace']['s']
    b = p['beach']
    plat = np.nonzero(names == 'platform')[0]
    rows = np.r_[plat, np.nonzero(names == 'foot')[0]]  # on up against the cliff foot (no cavity)
    cols = np.nonzero(np.abs(u - b['u_m']) <= b['length_m'] / 2 + .5)[0]
    rock = Vg[rows][:, cols]
    s_end = s_arc[plat[-1]][cols]  # arc length where the platform meets the cliff foot, per column
    d = np.maximum(s_end[None] - s_arc[rows][:, cols], 0.)  # distance seaward from the foot
    st = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
    along = st(1 - 2 * np.abs(u[cols] - b['u_m']) / b['length_m'])[None]
    across = st(1 - d / b['width_m'])
    # sand surface: on the platform the rock plus the lens thickness (smoothly thinning to the
    # rims); up the cliff foot the level of the lens at the foot line; where the rock is higher,
    # 3 cm under the rock (hidden)
    shape = b['thickness_m'] * across * along
    on_platform = np.isin(names[rows], ('platform',))[:, None]
    top = np.where(on_platform, rock[..., 2] + shape, Vg[plat[-1]][cols][None, :, 2] + shape)
    lens = (top > rock[..., 2] + 1e-4) & (shape > 1e-4)
    sub = rock.copy()
    sub[..., 2] = np.where(lens, top, rock[..., 2] - .03)
    keep_r = np.nonzero(lens.any(1))[0]
    sub = sub[max(keep_r[0] - 1, 0):keep_r[-1] + 2] if len(keep_r) else sub[:0]
    V = sub.reshape(-1, 3)
    F = grid_faces(sub.shape[0], sub.shape[1]) if len(sub) > 1 else np.zeros((0, 3), np.int64)
    from .display import fbm
    sand = np.array([.62, .55, .42]) * rng.uniform(.85, 1.1)
    colors = sand * (1 + .2 * fbm(V * 3., seed + 91, 3))[:, None]
    return Item('beach', V, F, 'sediment', colors=np.clip(colors, .02, .95),
                provenance={'process': 'pocket beach sand lens on the wave-cut platform at the cliff foot', 'seed': seed, **b})
