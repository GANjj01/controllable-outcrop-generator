"""Roadcut exposure: a road along the tile, cut into a hillside, with a fresh rock cut slope (60-80 deg, one bench when
the cut is higher than 8 m).

Local frame as for the escarpment (u along the road, w into the hill, z above ground). Per column the profile chain
runs: front ground -> verge -> road (2 % crossfall towards the cut) -> shoulder -> ditch -> cut toe -> face [-> bench
edge -> bench -> bench toe -> face] -> cut top -> natural hillside (soil) -> hill top -> upland -> back to the plain.
The road is gently curved in plan (radius 60-200 m): a straight planar cut meets every planar contact in a straight
line, from which no orientation can be measured; a curved cut gives curved traces.

The road, shoulder and ditch lining are non-rock items (semantic ``manmade``) laid over the excavated rock; the rock
under them is held still by the surface processes, and its geology stays available as bedrock under cover. Labels of
the cut faces come, as for every exposure, from v2 at the final surface (processes.erode) -- nothing here reads
geology.

Not done (recorded): presplit half-barrel marks (boreholes of 4-5 cm radius are below the 7 cm mesh resolution)."""
import math

import numpy as np

from .exposure import EDGE_ROOM_M, _waves

# Rows under the road item: held still by the surface processes.
PAVED = ('road-in', 'road', 'shoulder', 'ditch-a', 'ditch-in', 'ditch-b', 'ditch-out', 'ditch-c')
CUT_ROWS = ('toe-flat', 'toe', 'face-1', 'bench-edge', 'bench', 'bench-toe', 'face-2', 'cut-top')
# Fresh faces recede little: a far smaller fold limit than the natural scarp (FACE_MIN_RADIUS_M).
CUT_MIN_RADIUS_M = 3.
CUT_CHECK_SLOPE_DEG = 40.
REPOSE_DEG = 35.


def sample(g, tile):
    T, (mx, my), g0 = tile['size_m'], tile['min_xy_m'], tile['ground_z_m']
    strike = float(g.uniform(-25., 25.))
    relief_max = tile['max_relief_m']
    H = float(g.uniform(6., relief_max - 4.))
    two = H > 8.
    hill = float(g.uniform(1.5, max(2., relief_max - 3. - H)))  # crest, hill and upland stay below max relief
    p = {'strike_deg': strike, 'centre_xy_m': [mx + T / 2, my + T / 2], 'domain_half_m': T / 2 + mx,
         'ground_z_m': g0,
         'cut_height_m': H, 'benches': int(two), 'lift_1_fraction': float(g.uniform(.45, .6)) if two else 1.,
         'bench_width_m': float(g.uniform(2., 4.)), 'bench_tilt_deg': float(g.uniform(2., 5.)),
         'face_dip_deg': float(g.uniform(60., 80.)),
         'toe_radius_m': .3, 'edge_radius_m': float(g.uniform(1.2, 2.)),
         'hill_slope_deg': float(g.uniform(18., 30.)), 'hill_rise_m': hill,
         'upland_slope_deg': float(g.uniform(0., 3.)), 'back_slope_deg': float(g.uniform(8., 14.)),
         'road_width_m': float(g.uniform(6.5, 8.)), 'crossfall': .02, 'shoulder_m': 1., 'verge_m': float(g.uniform(1.5, 3.)),
         'guardrail': bool(g.random() < .6),
         # cut toe line: room in front of it for the road inside the tile
         'toe_w_m': -T / 2 + EDGE_ROOM_M + 11. + float(g.uniform(0., .12)) * T,
         'curve_radius_m': float(g.uniform(60., 200.)), 'curve_sign': float(g.choice([-1., 1.])),
         'relax_start_m': T / 2 - 3., 'relax_m': 30.,
         'taper_start_m': T / 2 + 6., 'reach_m': float(g.uniform(25., 45.)),
         'waves': {'height': [float(g.uniform(.03, .08)), float(g.uniform(20., 50.)), float(g.uniform(0., 2 * math.pi))]},
         'relief': {'seed': int(g.integers(1, 2 ** 30)), 'lowland_m': float(g.uniform(.05, .2)), 'upland_m': float(g.uniform(.2, .6))},
         'variation': 1.}
    c, s = math.cos(math.radians(strike)), math.sin(math.radians(strike))
    U = p['domain_half_m'] * (abs(c) + abs(s))
    u = np.linspace(-U, U, int(2 * U / .25) + 1)
    if min_radius(p, u) < CUT_MIN_RADIUS_M:
        raise ValueError('roadcut plan curvature below the fold limit')
    p['view_direction'] = [s, -c, 0.]
    p['rock_xy'] = [[mx, my], [mx + T, my + T]]
    return p


def chain(p, u, coarse=1.):
    """Profile chain per column (w = 0 at the start of the front ground) and the w shift."""
    from .profiles import arc, end_state, line
    rad = math.radians
    e = .07 * coarse
    t = np.clip((np.abs(u) - p['taper_start_m']) / p['reach_m'], 0, 1)
    taper = 1 - t * t * (3 - 2 * t)
    full = p['cut_height_m'] * (1 + _waves(p['waves']['height'], u, p['variation']))
    H = .4 + (full - .4) * taper
    q = np.clip(H / p['cut_height_m'], .05, 1.)
    r = np.clip((np.abs(u) - p['relax_start_m']) / p['relax_m'], 0, 1)
    r = r * r * (3 - 2 * r)
    alpha = np.radians(p['face_dip_deg'] + (REPOSE_DEG - p['face_dip_deg']) * r)
    beta = rad(p['hill_slope_deg']) * np.maximum(taper, .2)
    up, back, tilt = rad(p['upland_slope_deg']) * taper + 1e-3, rad(p['back_slope_deg']), rad(p['bench_tilt_deg'])
    cf = math.atan(p['crossfall'])
    rt, rc = p['toe_radius_m'] * q, p['edge_radius_m'] * q
    z_low = 0.
    front = [line(70., 0., 1. * coarse, 'front-far'), line(p['verge_m'], 0., .2 * coarse, 'verge'),
             arc(50., -cf, .2 * coarse, 'road-in'), line(p['road_width_m'] - 50. * cf, -cf, .2 * coarse, 'road'),
             line(p['shoulder_m'], -cf, .1 * coarse, 'shoulder'),
             arc(.5, -.6, e, 'ditch-a'), line(.45, -.6, e, 'ditch-in'), arc(.4, .6, e, 'ditch-b'),
             line(.45, .6, e, 'ditch-out'), arc(.5, 0., e, 'ditch-c'), line(.3, 0., e, 'toe-flat')]
    # Lift heights: the cut rises H above the toe in one or two lifts.
    h1 = H * p['lift_1_fraction']
    cos = np.cos
    if p['benches']:
        f1 = (h1 - rt * (1 - cos(alpha)) - rc * (math.cos(tilt) - cos(alpha))) / np.sin(alpha)
        bw = p['bench_width_m'] * np.maximum(q, .3)
        h2 = H - h1 + bw * math.sin(tilt)
        f2 = (h2 - rt * (math.cos(tilt) - cos(alpha)) - rc * (cos(beta) - cos(alpha))) / np.sin(alpha)
        cut = [arc(rt, alpha, e, 'toe'), line(f1, alpha, e, 'face-1'), arc(rc, -tilt, e, 'bench-edge'),
               line(bw, -tilt, e, 'bench'), arc(rt, alpha, e, 'bench-toe'), line(f2, alpha, e, 'face-2')]
        faces = (f1, f2)
    else:
        f1 = (H - rt * (1 - cos(alpha)) - rc * (cos(beta) - cos(alpha))) / np.sin(alpha)
        cut = [arc(rt, alpha, e, 'toe'), line(f1, alpha, e, 'face-1')]
        faces = (f1,)
    if any((f <= .05).any() for f in faces):
        raise ValueError('roadcut fillets leave no face')
    hill_len = np.maximum(p['hill_rise_m'] * taper, .3) / np.sin(beta)
    mid = [arc(rc, beta, e, 'cut-top'), line(hill_len, beta, .2 * coarse, 'hillside'), arc(15., up, .5 * coarse, 'hill-top'),
           line(12., up, 1. * coarse, 'upland'), arc(10., -back, 1. * coarse, 'brow')]
    w_toe, _, _ = end_state(front, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    _, z_brow, _ = end_state(front + cut + mid, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    drop = z_brow - z_low
    Rb = np.minimum(40., .5 * drop / (1 - math.cos(back)))
    tail = [line((drop - Rb * (1 - math.cos(back))) / math.sin(back), -back, 1. * coarse, 'descent'),
            arc(Rb, 0., 1. * coarse, 'base'), line(15., 0., 1. * coarse, 'back-flat')]
    # Curved road: the whole profile follows a circular-ish plan curve.
    shift = p['toe_w_m'] + p['curve_sign'] * u ** 2 / (2 * p['curve_radius_m']) - w_toe
    return front + cut + mid + tail, shift, z_low


def min_radius(p, u):
    """Smallest plan radius of the steep cut rows inside the geology domain."""
    from .profiles import trace
    segs, shift, z_low = chain(p, u, coarse=4.)
    tr = trace(segs, np.zeros_like(u), np.full_like(u, z_low), np.zeros_like(u))
    rows = np.isin(tr['rows'], CUT_ROWS)
    w = tr['w'][rows] + shift[None]
    steep = tr['angle'][rows] > math.radians(CUT_CHECK_SLOPE_DEG)
    du = u[1] - u[0]
    w1 = np.gradient(w, du, axis=1)
    w2 = np.gradient(w1, du, axis=1)
    radius = (1 + w1 ** 2) ** 1.5 / np.maximum(np.abs(w2), 1e-12)
    return float(np.where(steep, radius, np.inf)[:, 2:-2].min())


def sheet(p, edge=.07, smooth_m=1.):
    from .exposure import chain_sheet
    return chain_sheet(p, chain, edge, smooth_m,
                       relief=lambda tr, rows: ((tr['s'][rows['verge']][None] - 3., 3.),
                                                (tr['s'][rows['hillside']][None] + 2., tr['s'][rows['brow']][None] - 8.)))


def road_item(sh, rng, seed):
    """Pavement over the road/shoulder rows, lining over the ditch, a skirt at the front edge;
    lane edge lines. Same columns as the rock sheet, laid a little above it."""
    from .display import fbm
    from .surface import Item
    R, C = sh['grid']
    names = sh['trace']['rows']
    keep = np.nonzero(np.isin(names, PAVED))[0]
    Vg = sh['V_grid']
    sub = Vg[keep].copy()
    lift = np.where(np.isin(names[keep], ('road-in', 'road')), .15,
                    np.where(names[keep] == 'shoulder', .1, .05))[:, None]
    # shoulder: ramp from road height to lining height across its rows
    sh_rows = np.nonzero(names[keep] == 'shoulder')[0]
    if len(sh_rows):
        lift[sh_rows, 0] = np.linspace(.15, .05, len(sh_rows) + 2)[1:-1]
    sub[..., 2] += lift
    skirt = sub[:1].copy()
    skirt[..., 2] = Vg[keep[0] - 1:keep[0], :, 2] - .15  # front edge dips under the verge
    grid = np.concatenate([skirt, sub])
    from .profiles import grid_faces
    rows_all = np.r_[[-1], keep]
    Fi = grid_faces(len(grid), C)
    V = grid.reshape(-1, 3)
    # colours: asphalt with lane edge lines, gravel shoulder, concrete ditch lining
    across = np.repeat(np.r_[0., sh['trace']['s'][keep, :].mean(1)], C)
    kind = np.repeat(np.r_[['skirt'], names[keep]], C)
    asphalt = np.array([.09, .09, .095]) * rng.uniform(.8, 1.25)
    base = np.where(np.isin(kind, ('road-in', 'road', 'skirt'))[:, None], asphalt,
                    np.where((kind == 'shoulder')[:, None], np.array([.32, .3, .27]), np.array([.48, .47, .45])))
    s_road = sh['trace']['s'][[np.nonzero(names == 'road-in')[0][0], np.nonzero(names == 'shoulder')[0][0]], :].mean(1)
    line_at = np.abs(across - (s_road[0] + .35)) < .08
    line_at |= np.abs(across - (s_road[1] - .35)) < .08
    base = np.where((line_at & np.isin(kind, ('road-in', 'road')))[:, None], np.array([.8, .8, .76]), base)
    colors = base * (1 + .25 * fbm(V * 2.5, seed + 70, 3))[:, None]
    return Item('road', V, Fi, 'manmade', colors=np.clip(colors, .02, .95),
                provenance={'process': 'pavement, gravel shoulder and concrete ditch lining over the excavated rock', 'seed': seed})


def guardrail_item(sh, p, rng, seed):
    """Posts every 2 m along the verge and a W-beam rail (boxes, non-degenerate triangles)."""
    from .surface import Item
    names = sh['trace']['rows']
    r = np.nonzero(names == 'verge')[0]
    row = int(r[len(r) // 2])
    Vg, u = sh['V_grid'], sh['u']
    inside = np.abs(u) < p['taper_start_m'] + p['reach_m']
    cols = np.nonzero(inside)[0]
    base_pts = Vg[row, cols]
    uu = u[cols]
    posts = np.arange(math.ceil(uu.min() / 2.) * 2., uu.max(), 2.)
    verts, faces = [], []

    def box(centre, half, along):
        along = along / np.linalg.norm(along)
        across = np.array([-along[1], along[0], 0.])
        o = len(verts)
        for dz in (-1, 1):
            for dy in (-1, 1):
                for dx in (-1, 1):
                    verts.append(centre + dx * half[0] * along + dy * half[1] * across + np.array([0, 0, dz * half[2]]))
        for a, b, cc, d in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            faces.extend([[o + a, o + b, o + cc], [o + a, o + cc, o + d]])
    tangent = np.gradient(base_pts, axis=0) * [1, 1, 0]
    for x in posts:
        k = int(np.argmin(np.abs(uu - x)))
        box(base_pts[k] + np.array([0, 0, .42]), (.05, .05, .42), tangent[k])
    # rail: a thin box per span of 4 columns, following the curve
    for k in range(0, len(cols) - 1, 4):
        a_pt, b_pt = base_pts[k], base_pts[min(k + 4, len(cols) - 1)]
        t = (b_pt - a_pt) * [1, 1, 0]
        side = np.array([-t[1], t[0], 0.]) / np.linalg.norm(t)  # +w: the traffic side of the posts
        mid = .5 * (a_pt + b_pt) + np.array([0, 0, .62]) + .07 * side
        box(mid, (.5 * np.linalg.norm(t) + .01, .02, .15), t)
    V = np.asarray(verts)
    colors = np.tile(np.array([.62, .63, .64]) * rng.uniform(.85, 1.1), (len(V), 1))
    return Item('guardrail', V, np.asarray(faces, np.int64), 'manmade', colors=colors,
                provenance={'process': 'steel W-beam guardrail on 2 m posts along the verge', 'seed': seed})
