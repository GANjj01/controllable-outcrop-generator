"""Exposure layer of the sandbox: how a tile's geological volume is laid bare.

``sample_exposure(kind, model, seed)`` returns the parameters of one exposure of a v2 geology tile
(scene.sample_spec). They depend only on (seed, kind) and the tile frame, never on the geology, so one v2 volume can
be exposed in several ways with identical geological truth.

Kinds: ``bluff``, ``escarpment`` (scarp across the tile), ``roadcut`` (fresh cut slope along a curved road,
roadcut.py). Quarry, valley and coast follow."""
import math

import numpy as np

# Append only: the RNG stream of a kind is keyed by its index here.
EXPOSURES = ('escarpment', 'bluff', 'roadcut', 'escarpment_ridged', 'coast', 'canyon')
# Room around the exposed rock inside the tile for outward relief (~1.5 m) and fault heave.
EDGE_ROOM_M = 4.


def sample_exposure(kind, model, seed=None, *, shape=None):
    meta = model['metadata']
    if 'tile' not in meta:
        raise ValueError('model has no sandbox tile (generated before the geology/exposure split)')
    if kind not in EXPOSURES:
        raise ValueError('unknown exposure %r (choose from %s)' % (kind, ', '.join(EXPOSURES)))
    from .shape_config import validate_shape
    shape = validate_shape(kind, shape)
    seed = int(meta['seed'] if seed is None else seed)
    rng = np.random.default_rng([seed, 11, EXPOSURES.index(kind)])
    tile = meta['tile']
    if kind == 'bluff':
        params = _bluff(rng, tile, shape=shape)
    elif kind == 'roadcut':
        from .roadcut import sample
        params = sample(rng, tile)
    elif kind == 'escarpment_ridged':
        params = _escarpment_ridged(rng, tile)
    elif kind == 'coast':
        from .coast import sample as coast_sample
        params = coast_sample(rng, tile)
    elif kind == 'canyon':
        from .canyon import sample as canyon_sample
        params = canyon_sample(rng, tile)
    else:
        params = _escarpment(rng, tile)
    if shape and kind == 'canyon':
        from .canyon import configure
        configure(params, shape, tile)
    elif shape and kind != 'bluff':
        _configure_profile(kind, params, shape)
    return {'kind': kind, 'seed': seed, 'view_direction': [0., -1., 0.], 'ground_z_m': tile['ground_z_m'], **params}


def _configure_profile(kind, parameters, shape):
    """Keep the safe sampled template, then validate explicit controls unchanged.

    The template's normal fold guard runs first, as it always did."""
    from .shape_config import apply_profile_shape
    if kind == 'coast':
        # The accepted cliff adapter supplies its own length/height/dip. These
        # old profile helpers only provide the aligned water and far plain.
        helper = {key: value for key, value in shape.items()
                  if key in ('strike_deg', 'dip_direction_deg', 'centre_xy_m', 'sea_level_m')}
        apply_profile_shape(parameters, helper)
    else:
        apply_profile_shape(parameters, shape)
        if kind in ('escarpment', 'escarpment_ridged'):
            dip_wave = parameters['waves']['dip']
            amplitude = parameters['variation'] * abs(dip_wave[0])
            if (parameters['face_dip_deg'] - amplitude <= 0.
                    or parameters['face_dip_deg'] + amplitude > 86.):
                raise ValueError('Explicit shape face dip and dip wave must stay in (0,86] degrees without clipping')
        c, s = math.cos(math.radians(parameters['strike_deg'])), math.sin(math.radians(parameters['strike_deg']))
        U = parameters['domain_half_m'] * (abs(c) + abs(s))
        u = np.linspace(-U, U, int(2 * U / .25) + 1)
        if kind == 'roadcut':
            from .roadcut import CUT_MIN_RADIUS_M, min_radius
            minimum, limit = min_radius(parameters, u), CUT_MIN_RADIUS_M
        else:
            minimum, limit = _scarp_min_radius(parameters, u), FACE_MIN_RADIUS_M
        if not np.isfinite(minimum) and minimum != np.inf or minimum < limit:
            raise ValueError('Explicit %s shape plan curvature is below the %.1fm fold limit' % (kind, limit))
    angle = math.radians(parameters['strike_deg'])
    parameters['view_direction'] = [math.sin(angle), -math.cos(angle), 0.]


def _escarpment(g, tile):
    """Scarp across the tile: lowland in front, a steep face, a rounded crest and a gentle upland
    that falls back to the plain far behind. Local frame: u along strike, w into the upland
    (face looks towards -w), z above ground; the strike is rotated against the tile axes.

    Height, face dip and foot line vary smoothly along strike; their amplitudes are scaled down
    until the steep part of the face (slope > FACE_CHECK_SLOPE_DEG, where recession acts) keeps a
    plan-view radius of curvature of at least FACE_MIN_RADIUS_M inside the geology domain (inward
    recession at a convex bend tighter than that folds the sheet)."""
    T, (mx, my), g0 = tile['size_m'], tile['min_xy_m'], tile['ground_z_m']
    strike = float(g.uniform(-25., 25.))
    wave = lambda amp, wl=(14., 45.): [float(amp), float(g.uniform(*wl)), float(g.uniform(0., 2 * math.pi))]
    # Height along strike is height_m * (1 + wave); the crest (toe rise < 1 m) stays below max relief.
    h_wave = wave(g.uniform(.05, .15))
    p = {'strike_deg': strike, 'centre_xy_m': [mx + T / 2, my + T / 2],
         'domain_half_m': T / 2 + mx,  # tile + margin (the geology domain starts at 0)
         'height_m': float(g.uniform(8., (tile['max_relief_m'] - 1.) / (1 + h_wave[0]))),
         'face_dip_deg': float(g.uniform(60., 82.)),
         'toe_radius_m': float(g.uniform(.8, 2.)),
         'crest_radius_m': float(g.uniform(3., 5.)),
         'apron_slope_deg': float(g.uniform(.5, 4.)),
         'upland_slope_deg': float(g.uniform(1., 7.)),
         'back_slope_deg': float(g.uniform(8., 14.)),
         # Toe line at -T/2 + room + 10..30 % of the tile (the crest stays well inside the tile).
         'foot_w_m': -T / 2 + EDGE_ROOM_M + float(g.uniform(.1, .3)) * T,
         # Beyond the tile's rock the face relaxes to a soil/debris slope at the angle of repose
         # (over relax_m from just inside the tile edge), and the height dies out over reach_m.
         'relax_start_m': T / 2 - 3., 'relax_m': 30., 'repose_deg': float(g.uniform(32., 37.)),
         'taper_start_m': T / 2 + 6., 'reach_m': float(g.uniform(25., 45.)),
         'waves': {'foot': [wave(g.uniform(.5, 3.), (18., 60.)) for _ in range(3)],
                   'height': h_wave, 'dip': wave(g.uniform(2., 6.))},
         'relief': {'seed': int(g.integers(1, 2 ** 30)), 'lowland_m': float(g.uniform(.05, .25)),
                    'upland_m': float(g.uniform(.2, .8))},
         'variation': 1.}
    c, s = math.cos(math.radians(strike)), math.sin(math.radians(strike))
    U = p['domain_half_m'] * (abs(c) + abs(s))
    u = np.linspace(-U, U, int(2 * U / .25) + 1)
    for _ in range(40):
        if _scarp_min_radius(p, u) >= FACE_MIN_RADIUS_M:
            break
        p['variation'] *= .85
    else:
        raise ValueError('scarp plan curvature does not converge')
    p['view_direction'] = [s, -c, 0.]  # -w: from the face towards the viewer
    p['rock_xy'] = [[mx, my], [mx + T, my + T]]  # exposed rock stays inside the tile
    return p


def _escarpment_ridged(g, tile):
    """Scarp with a spur and shallow gullies.

    The base scarp is drawn exactly as ``_escarpment`` (same draws, same order); its long foot waves are damped to 30 % to
    leave the curvature budget to the relief. The same fold guard as the plain scarp (steep rows: plan radius >=
    FACE_MIN_RADIUS_M) scales waves and relief down together if needed."""
    T = tile['size_m']
    p = _escarpment(g, tile)
    for w in p['waves']['foot']:
        w[0] *= .3
    sig_s = float(g.uniform(6., 9.))
    # the spur's toe keeps EDGE_ROOM_M + 1 m from the front edge of the tile
    room = p['foot_w_m'] + T / 2 - EDGE_ROOM_M - 1.
    spur = {'u_m': float(g.uniform(-5., 5.)), 'sigma_m': sig_s,
            'amp_m': -min(.9 * sig_s ** 2 / FACE_MIN_RADIUS_M, room), 'height_frac': 0.}
    ridges = [spur]
    for side in (-1., 1.):
        sig_g = float(g.uniform(4., 6.))
        gap = float(g.uniform(1., 4.))
        ridges.append({'u_m': spur['u_m'] + side * (sig_s + sig_g + gap), 'sigma_m': sig_g,
                       'amp_m': float(g.uniform(.5, .8)) * sig_g ** 2 / FACE_MIN_RADIUS_M,
                       'height_frac': -float(g.uniform(.15, .3))})
    p['ridges'] = ridges
    p['variation'], p['ridge_scale'] = 1., 1.
    c, s = math.cos(math.radians(p['strike_deg'])), math.sin(math.radians(p['strike_deg']))
    U = p['domain_half_m'] * (abs(c) + abs(s))
    u = np.linspace(-U, U, int(2 * U / .25) + 1)
    for _ in range(60):
        if _scarp_min_radius(p, u) >= FACE_MIN_RADIUS_M:
            break
        p['variation'] *= .9
        p['ridge_scale'] *= .9
    else:
        raise ValueError('ridged scarp plan curvature does not converge')
    return p


def _ridges(p, u):
    """Plan offset (m along w, + into the scarp) and relative height change of the spur/gullies."""
    k = p.get('ridge_scale', 1.)
    shift, height = np.zeros_like(u), np.zeros_like(u)
    for r in p.get('ridges', ()):
        b = np.exp(-.5 * ((u - r['u_m']) / r['sigma_m']) ** 2)
        shift += k * r['amp_m'] * b
        height += k * r['height_frac'] * b
    return shift, height


# Minimum plan-view radius of the face rows inside the geology domain (> worst inward offset).
FACE_MIN_RADIUS_M = 8.
# Gentler slopes are soil covered (cover threshold 28-46 deg): no recession, no fold risk.
FACE_CHECK_SLOPE_DEG = 40.
FACE_ROWS = ('apron-toe', 'toe', 'face', 'crest', 'upland-near')


def _waves(spec, u, scale):
    spec = [spec] if np.ndim(spec[0]) == 0 else spec
    return scale * sum(a * np.sin(2 * math.pi * u / wl + ph) for a, wl, ph in spec)


def _scarp_chain(p, u, coarse=1.):
    """Profile chain per column u (local frame, toe start at w = foot line) and the w shift."""
    from .profiles import arc, end_state, line
    v = p['variation']
    e = .07 * coarse
    t = np.clip((np.abs(u) - p['taper_start_m']) / p['reach_m'], 0, 1)
    taper = 1 - t * t * (3 - 2 * t)
    ridge_w, ridge_h = _ridges(p, u)  # zero for the plain escarpment
    full = p['height_m'] * (1 + _waves(p['waves']['height'], u, v)) * (1 + ridge_h)
    H = .3 + (full - .3) * taper
    q = np.clip(H / p['height_m'], .05, 1.)
    alpha = np.minimum(p['face_dip_deg'] + _waves(p['waves']['dip'], u, v), 86.)
    r = np.clip((np.abs(u) - p['relax_start_m']) / p['relax_m'], 0, 1)
    r = r * r * (3 - 2 * r)
    alpha = np.radians(alpha + (p['repose_deg'] - alpha) * r)
    rad = math.radians
    # The upland rises only behind the full scarp, so the far ends of the sheet stay low.
    lo, up, back = rad(p['apron_slope_deg']), rad(p['upland_slope_deg']) * taper, rad(p['back_slope_deg'])
    z_low = .1
    front = [line(80., 0., 1. * coarse, 'front-far'), arc(60., lo, .3 * coarse, 'apron-bend'),
             line(8., lo, .3 * coarse, 'apron'), line(2.5, lo, e, 'apron-toe')]
    w_toe, z_toe, _ = end_state(front, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    Rt, Rc = p['toe_radius_m'] * q, p['crest_radius_m'] * q
    face_len = (H - Rt * (math.cos(lo) - np.cos(alpha)) - Rc * (np.cos(up) - np.cos(alpha))) / np.sin(alpha)
    if (face_len <= .05).any():
        raise ValueError('scarp fillets leave no face')
    mid = [arc(Rt, alpha, e, 'toe'), line(face_len, alpha, e, 'face'), arc(Rc, up, e, 'crest'),
           line(3., up, e, 'upland-near'), line(10., up, .3 * coarse, 'upland-mid'),
           line(20., up, 1. * coarse, 'upland-far'), arc(10., -back, 1. * coarse, 'brow')]
    _, z_brow, _ = end_state(front + mid, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    drop = z_brow - z_low
    Rb = np.minimum(40., .5 * drop / (1 - math.cos(back)))
    tail = [line((drop - Rb * (1 - math.cos(back))) / math.sin(back), -back, 1. * coarse, 'descent'),
            arc(Rb, 0., 1. * coarse, 'base'), line(15., 0., 1. * coarse, 'back-flat')]
    shift = p['foot_w_m'] + _waves(p['waves']['foot'], u, v) - w_toe
    if p.get('ridges'):
        shift = shift + ridge_w
    return front + mid + tail, shift, z_low


def _scarp_min_radius(p, u):
    from .profiles import trace
    chain, shift, z_low = _scarp_chain(p, u, coarse=4.)
    tr = trace(chain, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    rows = np.isin(tr['rows'], FACE_ROWS)
    w = tr['w'][rows] + shift[None]
    steep = tr['angle'][rows] > math.radians(FACE_CHECK_SLOPE_DEG)
    du = u[1] - u[0]
    w1 = np.gradient(w, du, axis=1)
    w2 = np.gradient(w1, du, axis=1)
    radius = (1 + w1 ** 2) ** 1.5 / np.maximum(np.abs(w2), 1e-12)
    return float(np.where(steep, radius, np.inf)[:, 2:-2].min())


def scarp_sheet(p, edge=.07, smooth_m=1.):
    """Escarpment sheet in world coordinates: vertices, faces, smooth outward field, row names."""
    sheet = chain_sheet(p, _scarp_chain, edge, smooth_m,
                        relief=lambda tr, rows: ((tr['s'][rows['toe']][None] - 6., 4.),
                                                 (tr['s'][rows['upland-near']][None] + 2., tr['s'][rows['brow']][None] - 8.)))
    return sheet['V'], sheet['F'], sheet['n'], sheet['rows']


def strike_columns(p, edge):
    """Along-strike column coordinates: ``edge`` over the tile (rotated), graded to 1 m outside."""
    from .profiles import graded
    c, s = math.cos(math.radians(p['strike_deg'])), math.sin(math.radians(p['strike_deg']))
    fine = (p['domain_half_m'] - EDGE_ROOM_M) * (abs(c) + abs(s)) + 1.
    end = p['taper_start_m'] + p['reach_m'] + 40.
    outer = graded(fine, end, edge, 1.)[1:]
    return np.r_[-outer[::-1], np.linspace(-fine, fine, int(math.ceil(2 * fine / edge)) + 1), outer]


def chain_sheet(p, chain_fn, edge, smooth_m, relief):
    """Sheet exposure from a per-column profile chain (escarpment, roadcut).

    ``chain_fn(p, u, coarse) -> (segments, shift, z_low)``; ``relief(tr, rows)`` gives
    ((lowland end s, fade m), (upland start s, upland end s)): gentle fbm relief only on the
    lowland and the upland, never on the faces. Returns a dict with world vertices V, faces F,
    smooth outward field n, row names per vertex, the trace, the local frame and grid shape."""
    from .display import fbm
    from .profiles import grid_faces, trace
    c, s = math.cos(math.radians(p['strike_deg'])), math.sin(math.radians(p['strike_deg']))
    u = strike_columns(p, edge)
    chain, shift, z_low = chain_fn(p, u, coarse=edge / .07)
    tr = trace(chain, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u), smooth_m=smooth_m)
    w = tr['w'] + shift[None]
    z = tr['z'].copy()
    uu = np.broadcast_to(u[None], w.shape)
    cx, cy = p['centre_xy_m']
    X, Y = cx + uu * c - w * s, cy + uu * s + w * c
    rows = tr['segment_rows']
    (low_end, low_fade), (up_start, up_end) = relief(tr, rows)
    st = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
    w_low = 1 - st((tr['s'] - low_end) / low_fade)
    w_up = st((tr['s'] - up_start) / 8.) * (1 - st((tr['s'] - up_end) / 8.))
    P2 = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    rl = p['relief']
    z += (rl['lowland_m'] * fbm(P2 / 12., rl['seed'], 3)).reshape(z.shape) * w_low
    z += (rl['upland_m'] * fbm(P2 / 25., rl['seed'] + 1, 3)).reshape(z.shape) * w_up
    # The outer 12 m of the sheet blends down to below the far ground plane (no visible edge).
    edge_d = np.minimum.reduce([uu - u[0], u[-1] - uu, tr['s'], tr['s'][-1:] - tr['s']])
    dive = 1 - st(edge_d / 12.)
    z += (p.get('edge_z_m', -.5) - z) * dive
    V = np.stack([X, Y, p['ground_z_m'] + z], -1)
    # Outward field: d/du x d/ds with the tangent angle smoothed over ~smooth_m along the arc.
    tu = np.gradient(V, u, axis=1)
    a = tr['smooth_angle']
    ts = np.stack([-s * np.cos(a), c * np.cos(a), np.sin(a)], -1)
    n = np.cross(tu, ts)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    R, C = w.shape
    return {'V': V.reshape(-1, 3), 'F': grid_faces(R, C), 'n': n.reshape(-1, 3), 'rows': np.repeat(tr['rows'], C),
            'grid': (R, C), 'trace': tr, 'u': u, 'frame': (c, s), 'dive': dive.reshape(-1), 'V_grid': V}


def _bluff(g, tile, shape=None):
    """2-4 rounded boxes joined by a smooth union; every mass contains the ray origin, so the union is star-shaped about it
    and radial sampling gives a closed surface. The base is buried 2.5 m below the ground; the bluff keeps EDGE_ROOM_M
    from the tile edge (the geology domain adds its own margin for rigid fault blocks)."""
    T, (mx, my), g0 = tile['size_m'], tile['min_xy_m'], tile['ground_z_m']
    room = T - 2 * EDGE_ROOM_M
    Wb, Db = float(g.uniform(.65, 1.) * room), float(g.uniform(.4, .75) * room)
    Hb = float(g.uniform(8., tile['max_relief_m']))
    base = -2.5
    k = int(g.integers(2, 5))
    if shape:
        Wb, Db, Hb = (shape.get(key, value) for key, value in
                      (('width_m', Wb), ('depth_m', Db), ('height_m', Hb)))
        base, k = -shape.get('base_depth_m', 2.5), shape.get('mass_count', k)
    masses = []
    for i in range(k):
        fraction = (i - (k - 1) / 2) / ((k - 1) / 2) if k > 1 else 0.
        cx = fraction * Wb / 4 * float(g.uniform(.7, 1.))
        hx = max((Wb / 2 - abs(cx)) * float(g.uniform(.85, 1.)), abs(cx) + 1.2)
        cy = float(g.uniform(-1.5, 1.5))
        hy = abs(cy) + float(g.uniform(.36, .5)) * Db
        top = Hb * float(g.uniform(.72, 1.05))
        cz, hz = (top + base) / 2, (top - base) / 2
        r = min(float(g.uniform(1.1, 2.4)), .45 * min(hx, hy, hz))
        if shape and 'rounding_fraction' in shape:
            r = shape['rounding_fraction'] * min(hx, hy, hz)
        masses.append({'center_m': [cx, cy, cz], 'half_size_m': [hx, hy, hz], 'rounding_m': r})
    # Tile frame: centred in the tile, ground at g0 (local z = 0).
    shift = np.array([mx + T / 2, my + T / 2, g0])
    if shape and 'centre_xy_m' in shape:
        shift[:2] = shape['centre_xy_m']
    c = np.array([m['center_m'] for m in masses])
    h = np.array([m['half_size_m'] for m in masses])
    ext_x = float(np.max(np.abs(c[:, 0]) + h[:, 0]))
    ext_y = float(np.max(np.abs(c[:, 1]) + h[:, 1]))
    top = float(np.max(c[:, 2] + h[:, 2]))
    for m in masses:
        m['center_m'] = (np.asarray(m['center_m']) + shift).tolist()
    origin = (np.array([0., 0., .35 * Hb]) + shift).tolist()
    result = {'origin_m': origin, 'masses': masses, 'blend_m': float(g.uniform(.8, 1.4)),
            'stretch': [max(1., ext_x / max(top, 1.)), max(.6, ext_y / max(top, 1.)), 1.],
            'max_radius_m': float(np.linalg.norm([ext_x, ext_y, top - base]) + 6.),
            'width_m': Wb, 'depth_m': Db, 'height_m': Hb}
    if shape:
        result['blend_m'] = shape.get('blend_m', result['blend_m'])
        # Radial sampling requires an interior origin in every mass. Extreme
        # requested aspect ratios can move it outside a rounded corner.
        for mass in masses:
            radius = mass['rounding_m']
            q = np.abs(np.asarray(origin) - mass['center_m']) - mass['half_size_m'] + radius
            distance = np.linalg.norm(np.maximum(q, 0)) + min(float(q.max()), 0.) - radius
            if distance >= 0:
                raise ValueError('Bluff dimensions/rounding must retain the ray origin inside every mass')
        result['max_radius_m'] = float(np.linalg.norm([ext_x, ext_y, top - base])
            + max(6., .25 * result['blend_m'] * max(k - 1, 0) + 1.))
        if 'rounding_fraction' in shape:
            result['rounding_fraction'] = shape['rounding_fraction']
        if 'azimuth_deg' in shape:
            result['rotation_deg'] = 90. - shape['azimuth_deg']
            angle = math.radians(result['rotation_deg'])
            result['view_direction'] = [math.sin(angle), -math.cos(angle), 0.]
    return result


# ---------------------------------------------------------------- patch sets (surface.py contract)
def build_surface(expo, *, samples=(1800, 520), max_edge=.07, ground_options=None):
    """ExposureSurface (patches + items) for one sampled exposure."""
    from .scene_config import check_ground_budget, validate_ground
    from .surface import ExposureSurface, Item, Patch
    ground_options = validate_ground(ground_options)
    if ground_options is not None:
        check_ground_budget(expo['kind'], ground_options)
    ground_args = {} if ground_options is None else dict(half=ground_options['half_m'],
        spacing=ground_options['spacing_m'], far=ground_options['far_m'])
    if expo['kind'] == 'bluff':
        from .morph import envelope_distance, radial_surface, ray_grid, surface_weights
        env = expo
        dirs, F = ray_grid(samples[0], samples[1], env['stretch'])
        if env.get('rotation_deg', 0.):
            angle = math.radians(env['rotation_deg'])
            c, s = math.cos(angle), math.sin(angle)
            dirs[:, :2] = dirs[:, :2] @ np.array([[c, s], [-s, c]])
        base, _ = radial_surface(dirs, 0., env)
        patch = Patch('bluff', base, F, dirs, apply=lambda o: radial_surface(dirs, o, env)[0],
                      weights=surface_weights(base, env),
                      provenance={'construction': 'smooth union of rounded boxes, radial level-set sampling (seminar demo)',
                                  'rays': list(samples)})

        def ground(rng, seed, soil_rgb):
            gV, gF = _ground_mesh(env, rng, seed, **ground_args)
            colors = soil_rgb * (1 + .35 * _fbm(gV * .4, seed + 30, 4) + .15 * _fbm(gV * 3, seed + 31, 3))[:, None]
            return Item('ground', gV, gF, 'soil', colors=colors,
                        provenance={'process': 'soil plain with apron against the rock foot', 'seed': seed,
                            **({'ground_parameters': dict(ground_options)} if ground_options is not None else {})})
        return ExposureSurface('bluff', [patch], items=[ground], camera={'view_direction': expo['view_direction'], 'facing_only': True},
                               profile={}, ground_z_m=env['ground_z_m'], origin_m=np.asarray(env['origin_m']))
    if expo['kind'] in ('escarpment', 'escarpment_ridged'):
        V, F, n, rows = scarp_sheet(expo, edge=max_edge)
        patch = Patch(expo['kind'], V, F, n, provenance={'construction': 'arc-length profile chain per strike column (profiles.py)'
                                                                         + ('; spur and gullies in plan (exposure._ridges)' if expo.get('ridges') else ''),
                                                           'max_edge_m': max_edge, 'outward': 'profile normal smoothed over 1 m of arc'})
        tile_centre = np.r_[np.mean(expo['rock_xy'], axis=0), expo['ground_z_m']]

        def far_ground(rng, seed, soil_rgb):
            # Plain beyond the sheet's outer ring (+-70 m), a little lower so the sheet stays on top.
            env = {'origin_m': tile_centre.tolist(), 'ground_z_m': expo['ground_z_m'] - .25, 'masses': []}
            gV, gF = _ground_mesh(env, rng, seed, **dict(dict(half=40., spacing=4.), **ground_args))
            colors = soil_rgb * (1 + .35 * _fbm(gV * .4, seed + 30, 4))[:, None]
            return Item('far-ground', gV, gF, 'soil', colors=colors,
                        provenance={'process': 'distant soil plain beyond the scarp sheet', 'seed': seed,
                            **({'ground_parameters': dict(ground_options)} if ground_options is not None else {})})
        return ExposureSurface(expo['kind'], [patch], items=[far_ground],
                               camera={'view_direction': expo['view_direction'], 'rock_xy': expo['rock_xy']},
                               profile={'fade_on_cover': True, 'cover_ramp': 25.}, ground_z_m=expo['ground_z_m'])
    if expo['kind'] == 'roadcut':
        from . import roadcut
        sh = roadcut.sheet(expo, edge=max_edge)
        hold = np.isin(sh['rows'], roadcut.PAVED)
        patch = Patch('roadcut', sh['V'], sh['F'], sh['n'], hold=hold, bare=np.isin(sh['rows'], roadcut.CUT_ROWS),
                      provenance={'construction': 'roadcut profile chain per column along a curved road (roadcut.py)',
                                  'max_edge_m': max_edge, 'held_rows': list(roadcut.PAVED)})
        tile_centre = np.r_[np.mean(expo['rock_xy'], axis=0), expo['ground_z_m']]

        def far_ground(rng, seed, soil_rgb):
            env = {'origin_m': tile_centre.tolist(), 'ground_z_m': expo['ground_z_m'] - .25, 'masses': []}
            gV, gF = _ground_mesh(env, rng, seed, **dict(dict(half=40., spacing=4.), **ground_args))
            colors = soil_rgb * (1 + .35 * _fbm(gV * .4, seed + 30, 4))[:, None]
            return Item('far-ground', gV, gF, 'soil', colors=colors,
                        provenance={'process': 'distant soil plain beyond the roadcut sheet', 'seed': seed,
                            **({'ground_parameters': dict(ground_options)} if ground_options is not None else {})})
        items = [far_ground, lambda rng, seed, soil_rgb: roadcut.road_item(sh, rng, seed)]
        if expo['guardrail']:
            items.append(lambda rng, seed, soil_rgb: roadcut.guardrail_item(sh, expo, rng, seed))
        return ExposureSurface('roadcut', [patch], items=items,
                               camera={'view_direction': expo['view_direction'], 'rock_xy': expo['rock_xy']},
                               profile={'process': 'fresh'}, ground_z_m=expo['ground_z_m'])
    if expo['kind'] == 'coast':
        from . import coast
        sh = coast.sheet(expo, edge=max_edge)
        patch = Patch('coast', sh['V'], sh['F'], sh['n'], bare=np.isin(sh['rows'], coast.BARE_ROWS),
                      provenance={'construction': 'coast profile chain per shore column: seabed, wave-cut platform, sea cliff (coast.py)',
                                  'max_edge_m': max_edge, 'bare_rows': list(coast.BARE_ROWS)})
        c, s = sh['frame']
        foot = np.nonzero(sh['trace']['rows'] == 'foot')[0][0]
        Vf = sh['V_grid'][foot]
        w_lim = float(((Vf[:, 0] - expo['centre_xy_m'][0]) * -s + (Vf[:, 1] - expo['centre_xy_m'][1]) * c).max())

        def land(rng, seed, soil_rgb):
            # Low coastal plain behind the cliff only (seaward it would rise above the sea).
            from .profiles import grid_faces
            if ground_options is None:
                us = np.r_[-3000., np.linspace(-150., 150., 61), 3000.]
                ws = np.r_[np.linspace(w_lim + 5., w_lim + 150., 30), 3000.]
            else:
                half, spacing, far = (ground_options[name] for name in ('half_m', 'spacing_m', 'far_m'))
                if w_lim + half >= far:
                    raise ValueError('scene.ground.far_m must extend beyond the coastal near ground and foot offset')
                us = np.r_[-far, np.linspace(-half, half, math.ceil(2 * half / spacing) + 1), far]
                ws = np.r_[np.linspace(w_lim + 5., w_lim + half, math.ceil((half - 5.) / spacing) + 1), far]
            U, W = np.meshgrid(us, ws)
            X = expo['centre_xy_m'][0] + U * c - W * s
            Y = expo['centre_xy_m'][1] + U * s + W * c
            gV = coast.far_exact(np.stack([X, Y, np.full_like(X, expo['ground_z_m'] + expo['back_z_m'] - .3)], -1).reshape(-1, 3),
                                 expo['centre_xy_m'])
            colors = soil_rgb * (1 + .35 * _fbm(gV * .4, seed + 30, 4))[:, None]
            return Item('far-ground', gV, grid_faces(len(ws), len(us)), 'soil', colors=colors,
                        provenance={'process': 'coastal plain behind the cliff', 'seed': seed,
                            **({'ground_parameters': dict(ground_options)} if ground_options is not None else {})})
        items = [land, lambda rng, seed, soil_rgb: coast.water_item(sh, expo, rng, seed),
                 lambda rng, seed, soil_rgb: coast.beach_item(sh, expo, rng, seed)]
        return ExposureSurface('coast', [patch], items=items,
                               camera={'view_direction': expo['view_direction'], 'rock_xy': expo['rock_xy']},
                               profile={'fade_on_cover': True, 'cover_ramp': 25.}, ground_z_m=expo['ground_z_m'])
    if expo['kind'] == 'canyon':
        from . import canyon
        sh = canyon.sheet(expo, edge=max_edge)
        patch = Patch('canyon', sh['V'], sh['F'], sh['n'],
                      provenance={'construction': 'two-wall canyon profile chain per column (canyon.py): plateau, '
                                                  'walls with benches, winding floor', 'max_edge_m': max_edge})
        tile_centre = np.r_[np.mean(expo['rock_xy'], axis=0), expo['ground_z_m']]

        def far_ground(rng, seed, soil_rgb):
            # The plateau continues beyond the sheet at its own height, a little lower so the sheet stays on top.
            env = {'origin_m': tile_centre.tolist(), 'ground_z_m': expo['ground_z_m'] + expo['plateau_m'] - .25, 'masses': []}
            gV, gF = _ground_mesh(env, rng, seed, **dict(dict(half=40., spacing=4.), **ground_args))
            # Open the plateau where the canyon sheet itself carries plateau, walls and floor.
            c, s_ = sh['frame']
            centre = np.asarray(expo['centre_xy_m'])
            local = lambda P: np.c_[(P[:, 0] - centre[0]) * c + (P[:, 1] - centre[1]) * s_,
                                    -(P[:, 0] - centre[0]) * s_ + (P[:, 1] - centre[1]) * c]
            sheet_uw = local(sh['V'])
            lo, hi = sheet_uw.min(0) + 14., sheet_uw.max(0) - 14.
            inside = np.all((local(gV) > lo) & (local(gV) < hi), axis=1)
            gF = gF[~np.all(inside[gF], axis=1)]
            colors = soil_rgb * (1 + .35 * _fbm(gV * .4, seed + 30, 4))[:, None]
            return Item('far-ground', gV, gF, 'soil', colors=colors,
                        provenance={'process': 'plateau continuing beyond the canyon sheet (open over the sheet)', 'seed': seed,
                            **({'ground_parameters': dict(ground_options)} if ground_options is not None else {})})
        return ExposureSurface('canyon', [patch], items=[far_ground],
                               camera={'view_direction': expo['view_direction'], 'rock_xy': expo['rock_xy']},
                               profile={'fade_on_cover': True, 'cover_ramp': 25.}, ground_z_m=expo['ground_z_m'])
    raise ValueError('no surface builder for exposure ' + expo['kind'])


def _fbm(p, seed, octaves):
    from .display import fbm
    return fbm(p, seed, octaves)


def _ground_mesh(env, rng, seed, half=60., spacing=.75, far=3000.):
    from .morph import ground_mesh
    return ground_mesh(env, rng, seed, half=half, spacing=spacing, far=far)
