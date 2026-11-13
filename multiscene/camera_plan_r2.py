"""Frame: 1024 px, 35 mm lens, 36 mm filmback (fixed), near clip 10 cm; pixel size at the target p = d * 36/35/1024
(mm/px, view axis footprint, = camera_plan.footprint_at_target_mm). Cells: pixel level fine [4, 11] / coarse [11, 32]
mm/px (log-uniform) x angle frontal / oblique. Target as camera_plan (shape: length-uniform on the exposed paths;
stain: area-uniform on the stainable host above ground + 0.5 m); normal n = area-weighted normal within 0.5 m. Face
type: horizontal if angle(n, z) <= 20 deg else side. Angle theta (target->camera vs n): horizontal frontal [0, 25]
oblique [25, 45]; side frontal [0, 25] oblique [35, 60] deg. w = cos(theta) n + sin(theta) (cos(phi) u + sin(phi) v),
v = up-slope unit vector in the tangent plane (+y projected when n is vertical), u = v x n; phi uniform [0, 360) on
horizontal faces, [0, 180] on side faces (offset toward up-slope). Draw order per candidate: target (as camera_plan),
log p, theta, phi. Acceptance (CPU raster at 1024 px of surface.npz + the ground square z = declaration ground_z_m,
half 2000 m about the tile centre; all must hold, first failing reported in this order): a camera z >= target z and >=
ground + 0.5 m; b every pixel is rock (front face), ground inside the tile [0, tile_m]^2 or sky (no hit, ray z >= 0);
blank = ground outside the tile, a downward ray with no hit, rock seen from behind (counterfactual 'unbounded ground':
only the back-face part); c the 2x2 centre pixels hit rock, |mean depth - target view depth| <= 0.15 m + 2 dt / fpx; d
rock pixels >= 70 %; e shape: visible length of the target path >= 256 px (station visible: in frame, |depth - z| <=
0.10 m + 2 z / fpx); f stain: stainable host >= 80 % of the central half, >= 90 % of the faces within 0.5 m. g top
fold: rock pixels (every 2nd row / column) on coarse faces with n_z >= cos 20 deg ('top'); tan(theta) between the
bluff ray from expo.origin_m and the envelope normal at the hit point (central differences of
outcrop.dsgen.morph.envelope_distance, h = 1 cm); pass when there is no top pixel or the pixel-weighted p95 <= 0.5.
Side pixels: the same statistic, recorded only. Scenes without erode_fields.pkl, and normal-route scenes: not
applicable (passes). Plan: 4 cells x K cameras, stream default_rng([scene seed, 4212, cell, k]), first accepted of at
most 400 tries. Outcrop type: config.json geometry.outcrop_type, else --outcrop, else 'isolated'. Usage: python
camera_plan_r2.py plan shape|stain SCENE_DIR NET_ROOT|- OUT_JSON [--per-cell K] [--outcrop flat|isolated] python
camera_plan_r2.py capacity shape|stain SCENE_DIR NET_ROOT|- OUT_JSON [--draws N]"""
import json, math, pickle, re, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; P1 = HERE.parent
sys.path[:0] = [str(P1), str(HERE)]
from frgb import raster, scene_light  # noqa: E402
from frgb import lith_records  # noqa: E402
from frgb import runconfig as frgb_rc  # noqa: E402
STAINABLE_HOST = json.loads(frgb_rc.path('labels.lithology_aliases_r1').read_text(encoding='utf8'))['stainable_host']
import camera_plan as cp  # noqa: E402  (local_normal, unchanged)

RES = 1024
LENS = dict(focal_length_mm=35., filmback_mm=[36., 36.], near_clip_cm=10.)
FPX = LENS['focal_length_mm'] / LENS['filmback_mm'][0] * RES
LEVELS = dict(fine=(4., 11.), coarse=(11., 32.))
ANGLE = dict(horizontal=dict(frontal=(0., 25.), oblique=(25., 45.)), side=dict(frontal=(0., 25.), oblique=(35., 60.)))
CELLS = [dict(level='fine', angle='frontal'), dict(level='fine', angle='oblique'),
         dict(level='coarse', angle='frontal'), dict(level='coarse', angle='oblique')]
RULES = dict(horizontal_face_max_deg=20., ground_clearance_m=.5, centre_tol_m=.15, centre_tol_px=2., rock_frac=.7,
             shape_visible_px=256., station_tol_m=.10, station_tol_px=2., stain_centre_sand=.8, stain_disc_sand=.9,
             normal_radius_m=.5, tangent_half_arc_m=.25, ground_half_size_m=2000., max_tries=400, capacity_draws=100,
             top_fold_tan_p95_max=.5, top_fold_stride_px=2, envelope_normal_step_m=.01, coarse_outcrops=('flat',))
ORDER = ('a_camera', 'b_boundary', 'c_centre', 'd_rock', 'e_shape', 'f_stain', 'g_top_fold')
COS_TOP = math.cos(math.radians(20.))


def distance_of(pixel_mm):
    return pixel_mm / 1000. * RES * LENS['focal_length_mm'] / LENS['filmback_mm'][0]


def pixel_of(distance_m):
    return distance_m * LENS['filmback_mm'][0] / LENS['focal_length_mm'] / RES * 1000.


def load(scene):
    scene = Path(scene).resolve(); geom = scene/'geom'
    cfg = json.loads((scene/'config.json').read_text(encoding='utf8'))
    with open(geom/'declaration.json', encoding='utf8') as fh:
        m = re.search(r'"ground_z_m":\s*([-0-9.eE+]+)', fh.read())
    if not m: raise ValueError('declaration.json has no ground_z_m')
    gz, tile = float(m.group(1)), float(cfg['geometry']['tile_m'])
    s = np.load(geom/'surface.npz'); V, F, codes = s['vertices'], s['faces'].astype(np.int64), s['codes']
    book = json.loads((geom/'codebook.json').read_text(encoding='utf8')); book = book if isinstance(book, list) else book.get('codebook', book)
    sand = [int(b['code']) for b in book if lith_records.is_host(b.get('lithology'))]   # a bare read: either name of the host
    c, h = tile / 2, RULES['ground_half_size_m']
    G = np.array([[c - h, c - h, gz], [c + h, c - h, gz], [c + h, c + h, gz], [c - h, c + h, gz]])
    tri = V[F]; cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    expo = pickle.load(open(scene/'erode_fields.pkl', 'rb'))['expo'] if (scene/'erode_fields.pkl').is_file() else None
    return dict(scene=scene, geom=geom, seed=int(cfg['seed']), tile=tile, ground_z=gz, V=V, F=F, codes=codes, sand=sand, nfaces=len(F), expo=expo,
                outcrop_type_config=cfg['geometry'].get('outcrop_type'), normal_route=(scene/'normal_route.json').is_file(),
                VG=np.r_[V, G], FG=np.r_[F, np.array([[0, 1, 2], [0, 2, 3]]) + len(V)], cent=tri.mean(1), cross=cr,
                area=np.linalg.norm(cr, axis=1) / 2, n=cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-300)[:, None],
                ground=dict(z_m=gz, center_xy_m=[c, c], half_size_m=h), xx=np.mgrid[0:RES, 0:RES][1], yy=np.mgrid[0:RES, 0:RES][0])


def load_paths(net):
    b = json.loads((Path(net)/'source_binding.json').read_text(encoding='utf8'))
    rows = b['exposure_paths']; out = []
    for r in rows:
        z = np.load(Path(net)/r['file']); out.append(dict(row=r, points=z['points'], arc=z['arc'], normal=z['normal']))
    return out


def target_shape(paths, rng):
    """camera_plan.targets_shape (same draws), plus the segment index, the trace tangent and the fracture-plane normal"""
    L = np.array([p['row']['length_m'] for p in paths]); i = int(rng.choice(len(paths), p=L / L.sum())); P = paths[i]
    pts, arc = P['points'], P['arc']; s = rng.uniform(0, arc[-1]); k = min(int(np.searchsorted(arc, s, 'right')) - 1, len(arc) - 2)
    t = (s - arc[k]) / (arc[k + 1] - arc[k]); p = pts[k] * (1 - t) + pts[k + 1] * t
    h = RULES['tangent_half_arc_m']
    a, b = np.interp(max(s - h, 0.), arc, np.arange(len(arc))), np.interp(min(s + h, arc[-1]), arc, np.arange(len(arc)))
    at = lambda q: pts[int(q)] * (1 - (q - int(q))) + pts[min(int(q) + 1, len(pts) - 1)] * (q - int(q))  # noqa: E731
    tang = at(b) - at(a)
    if np.linalg.norm(tang) < 1e-9: tang = pts[k + 1] - pts[k]          # a closed path shorter than the chord window: local segment
    tang = tang / np.linalg.norm(tang)
    m = P['normal'][k] + P['normal'][k + 1]
    return p, dict(path_index=int(P['row']['index']), trace_id=P['row']['source_trace_id'], arc_m=float(s), path_length_m=float(arc[-1]),
                   segment=k), pts, tang, m / np.linalg.norm(m)


def target_stain(sc, rng):
    ok = np.isin(sc['codes'], sc['sand']) & (sc['cent'][:, 2] > sc['ground_z'] + RULES['ground_clearance_m'])
    idx = np.flatnonzero(ok); f = int(rng.choice(idx, p=sc['area'][idx] / sc['area'][idx].sum()))
    w = rng.dirichlet(np.ones(3)); p = (sc['V'][sc['F'][f]] * w[:, None]).sum(0)
    return p, dict(face=f), None, None, None


def envelope_tan(expo, P):
    """tan(theta) between the bluff ray unit(P - origin) and the envelope normal at P (central differences, h = 1 cm)"""
    from outcrop.dsgen.morph import envelope_distance
    h = RULES['envelope_normal_step_m']
    g = np.stack([envelope_distance(P + e, expo) - envelope_distance(P - e, expo) for e in np.eye(3) * h], 1)
    n = g / np.linalg.norm(g, axis=1, keepdims=True); u = P - np.asarray(expo['origin_m']); u /= np.linalg.norm(u, axis=1, keepdims=True)
    c = (u * n).sum(1)
    return np.linalg.norm(np.cross(u, n), axis=1) / np.maximum(np.abs(c), 1e-12)


def top_fold(sc, rock, f, P):
    """gate g: (pass, record)"""
    if sc.get('expo') is None: return True, dict(applicable=False)
    if sc.get('normal_route'): return True, dict(applicable=False, reason='normal-route scene (scene/normal_route.json): gate g measures the ray route only')
    s = RULES['top_fold_stride_px']; sub = np.zeros(rock.shape, bool); sub[::s, ::s] = True
    sel = rock & sub; top = sel & (sc['n'][np.clip(f, 0, sc['nfaces'] - 1), 2] >= COS_TOP); side = sel & ~top
    rec = dict(applicable=True, top_share_of_rock=float(top.sum() / max(sel.sum(), 1)), top_pixels_sampled=int(top.sum()))
    for name, m in (('top', top), ('side', side)):
        if m.any():
            t = envelope_tan(sc['expo'], P[m]); rec[name + '_tan_p50'] = float(np.percentile(t, 50)); rec[name + '_tan_p95'] = float(np.percentile(t, 95))
    return (not top.any()) or rec['top_tan_p95'] <= RULES['top_fold_tan_p95_max'], rec


def basis(n):
    """v = up-slope unit vector in the tangent plane (+y projected when n is vertical), u = v x n"""
    v = np.array([0., 0., 1.]) - n[2] * n
    if np.linalg.norm(v) < 1e-6: v = np.array([0., 1., 0.]) - n[1] * n
    v /= np.linalg.norm(v)
    return np.cross(v, n), v


def draw(sc, mech, paths, cell, rng):
    p, info, pts, tang, m = target_shape(paths, rng) if mech == 'shape' else target_stain(sc, rng)
    n, _ = cp.local_normal(sc, p)
    ftype = 'horizontal' if math.degrees(math.acos(max(-1., min(1., n[2])))) <= RULES['horizontal_face_max_deg'] else 'side'
    lo, hi = LEVELS[cell['level']]; pix = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    th = math.radians(rng.uniform(*ANGLE[ftype][cell['angle']]))
    ph = rng.uniform(0, 2 * math.pi) if ftype == 'horizontal' else rng.uniform(0, math.pi)
    u, v = basis(n); w = math.cos(th) * n + math.sin(th) * (math.cos(ph) * u + math.sin(ph) * v)
    d = distance_of(pix)
    return dict(target=p, info=info, points=pts, tangent=tang, plane_normal=m, normal=n, face_type=ftype, pixel_mm=pix, distance_m=d,
                theta=th, phi=ph, w=w, pos=p + d * w)


def evaluate(sc, mech, c):
    """conditions a-f (None = not evaluated: only when a fails); b-f are all evaluated even when an earlier one fails"""
    pos, tgt = c['pos'], c['target']; cond = dict.fromkeys(ORDER); rep = {}
    cond['a_camera'] = bool(pos[2] >= tgt[2] and pos[2] >= sc['ground_z'] + RULES['ground_clearance_m'])
    if not cond['a_camera']:
        return cond, rep, None
    shot = dict(position_enu_m=[float(x) for x in pos], target_enu_m=[float(x) for x in tgt], **LENS)
    cam = raster.camera(shot, (RES, RES)); r = raster.rasterize(sc['VG'], sc['FG'], cam); f = r['face']; nf = sc['nfaces']
    d = raster.rays(cam, sc['xx'], sc['yy'])
    rock = (f >= 0) & (f < nf); gnd = f >= nf; miss = f < 0
    back = rock & ((sc['n'][np.clip(f, 0, nf - 1)] * d).sum(-1) > 0)
    P = cam['pos'] + d / (d @ cam['fwd'])[..., None] * np.where(np.isfinite(r['depth']), r['depth'], 0.)[..., None]
    inside = (P[..., 0] >= 0) & (P[..., 0] <= sc['tile']) & (P[..., 1] >= 0) & (P[..., 1] <= sc['tile'])
    down_miss = miss & (d[..., 2] < 0)
    frac = dict(rock=float((rock & ~back).mean()), rock_back=float(back.mean()), ground_in=float((gnd & inside).mean()),
                ground_out=float((gnd & ~inside).mean()), downward_miss=float(down_miss.mean()), sky=float((miss & ~down_miss).mean()))
    rep['fractions'] = frac
    cond['b_boundary'] = bool(frac['rock_back'] == 0 and frac['ground_out'] == 0 and frac['downward_miss'] == 0)
    rep['b_unbounded_ground'] = bool(frac['rock_back'] == 0)
    k = RES // 2; hc = f[k - 1:k + 1, k - 1:k + 1]; dt = float((tgt - cam['pos']) @ cam['fwd'])
    if ((hc < 0) | (hc >= nf)).any():
        cond['c_centre'] = False; rep['centre'] = 'centre misses the rock'
    else:
        dc = float(r['depth'][k - 1:k + 1, k - 1:k + 1].mean()); rep['centre_depth_diff_m'] = dt - dc
        cond['c_centre'] = bool(abs(dc - dt) <= RULES['centre_tol_m'] + RULES['centre_tol_px'] * dt / FPX)
    cond['d_rock'] = bool(frac['rock'] >= RULES['rock_frac'])
    if mech == 'shape':
        Pp = c['points']; rel = Pp - cam['pos']; z = rel @ cam['fwd']; zz = np.where(z > cam['near'], z, np.inf)
        x = (rel @ cam['right']) / zz * FPX + RES / 2; y = -(rel @ cam['up']) / zz * FPX + RES / 2
        ins = (z > cam['near']) & (x >= 0) & (x < RES) & (y >= 0) & (y < RES)
        xi, yi = np.clip(x.astype(int), 0, RES - 1), np.clip(y.astype(int), 0, RES - 1)
        vis = ins & (np.abs(r['depth'][yi, xi] - z) <= RULES['station_tol_m'] + RULES['station_tol_px'] * np.where(ins, z, 0.) / FPX)
        seg = vis[:-1] & vis[1:]
        rep['visible_px'] = float(np.hypot(np.diff(x), np.diff(y))[seg].sum()) if seg.any() else 0.
        cond['e_shape'] = bool(rep['visible_px'] >= RULES['shape_visible_px'])
    else:
        q = RES // 4; cen = f[q:3 * q, q:3 * q]; okc = (cen >= 0) & (cen < nf)
        rep['centre_sand'] = float(np.isin(sc['codes'][cen[okc]], sc['sand']).sum() / cen.size)
        _, near = cp.local_normal(sc, np.asarray(tgt))
        rep['disc_sand'] = float(np.isin(sc['codes'][near], sc['sand']).mean())
        cond['f_stain'] = bool(rep['centre_sand'] >= RULES['stain_centre_sand'] and rep['disc_sand'] >= RULES['stain_disc_sand'])
    cond['g_top_fold'], rep['top_fold'] = top_fold(sc, rock & ~back, f, P)
    rp = r['depth'][rock & ~back] / FPX * 1e3
    if rp.size: rep['rock_pixel_mm'] = dict(p10=float(np.percentile(rp, 10)), p50=float(np.median(rp)), p90=float(np.percentile(rp, 90)))
    return cond, rep, (shot, cam)


def passed(cond, mech):
    keys = ORDER[:4] + (('e_shape',) if mech == 'shape' else ('f_stain',)) + ('g_top_fold',)
    return all(cond[k] for k in keys), next((k for k in keys if not cond[k]), None)


def covariates(c, cam):
    fwd = cam['fwd']; cov = dict(camera_elevation_deg=math.degrees(math.asin(max(-1., min(1., c['w'][2])))),
                                 view_azimuth_deg=scene_light.compass_deg(fwd) if math.hypot(fwd[0], fwd[1]) > 1e-9 else None)
    if c['tangent'] is not None:
        t = c['tangent']; cov['trace_tangent'] = t.tolist()
        cov['view_trace_angle_deg'] = math.degrees(math.acos(min(1., abs(float(fwd @ t)))))
        cov['image_trace_angle_deg'] = math.degrees(math.atan2(float(t @ cam['up']), float(t @ cam['right']))) % 180.
        cov['fracture_plane_normal'] = c['plane_normal'].tolist()
    return cov


def record(sc, c, cell_index, cell, k, cond, rep, shot, cam, tries=None, rejections=None):
    ref = scene_light.strike_reference(c['plane_normal'], c['tangent']) if c['plane_normal'] is not None else None
    cov = covariates(c, cam); cov.update({'fold_' + k_: v for k_, v in (rep.get('top_fold') or {}).items()})
    return dict(cell=cell_index, k=k, **cell, face_type=c['face_type'], pixel_mm=c['pixel_mm'], distance_m=c['distance_m'],
                angle_deg=math.degrees(c['theta']), azimuth_about_normal_deg=math.degrees(c['phi']), target=c['info'],
                target_enu_m=[float(x) for x in c['target']], normal=c['normal'].tolist(), shot=shot, resolution=[RES, RES],
                footprint_at_target_mm=c['pixel_mm'], checks=rep, tries=tries, rejections=rejections, covariates=cov,
                light_reference=ref, light_reference_note=None if ref else 'stain: strike of the band plane, scene_light.strike_reference(band normal) at staging')


def header(sc, mech):
    return dict(mechanism=mech, geometry=str(sc['geom']).replace('\\', '/'), scene=str(sc['scene']).replace('\\', '/'), seed=sc['seed'],
                tile_m=sc['tile'], ground=sc['ground'], resolution=[RES, RES], lens=LENS, levels_mm=LEVELS, angle_deg=ANGLE, cells=CELLS,
                rules=RULES, criteria='camera-plan rule r1 with revisions r1a (gate g) and r1cd '
                '(r1d coarse on flat outcrops only)', method=__doc__)


def plan(mech, scene, net, per_cell=1, outcrop=None, near=None):
    sc = load(scene); paths = load_paths(net) if mech == 'shape' else None; out = []; failed = []; skipped = []
    otype = sc['outcrop_type_config'] or outcrop or 'isolated'
    osrc = 'config.json geometry.outcrop_type' if sc['outcrop_type_config'] else ('--outcrop' if outcrop else 'default (isolated)')
    for ci, cell in enumerate(CELLS):
        if cell['level'] == 'coarse' and otype not in RULES['coarse_outcrops']:
            skipped.append(dict(cell=ci, **cell, reason='r1d: coarse cells only on flat outcrops')); continue
        for k in range(per_cell):
            rng = np.random.default_rng([sc['seed'], 4212, ci, k]); reasons = []
            for t in range(RULES['max_tries']):
                c = draw(sc, mech, paths, cell, rng); cond, rep, sc_cam = evaluate(sc, mech, c)
                ok, why = passed(cond, mech)
                reasons.append(why or 'ok')
                if ok:
                    out.append(record(sc, c, ci, cell, k, cond, rep, *sc_cam, tries=t + 1,
                                      rejections={x: reasons.count(x) for x in set(reasons) if x != 'ok'}))
                    break
            else:
                failed.append(dict(cell=ci, k=k, rejections={x: reasons.count(x) for x in set(reasons)}))
    res = dict(header(sc, mech), outcrop_type=otype, outcrop_type_source=osrc, qualified=not failed, failed_cells=failed,
               skipped_cells=skipped, cameras=out)
    if near is not None:
        import camera_plan_near
        camera_plan_near.add_near(res, sc, mech, paths, **near)
    return res


def capacity(mech, scene, net, draws=None):
    draws = draws or RULES['capacity_draws']; sc = load(scene); paths = load_paths(net) if mech == 'shape' else None; cells = []; t0 = time.time()
    for ci, cell in enumerate(CELLS):
        rng = np.random.default_rng([sc['seed'], 4213, ci]); rows = []
        lo, hi = LEVELS[cell['level']]; edges = np.exp(np.linspace(math.log(lo), math.log(hi), 5))
        for _ in range(draws):
            c = draw(sc, mech, paths, cell, rng); cond, rep, _ = evaluate(sc, mech, c)
            ok, why = passed(cond, mech)
            cf = dict(cond, b_boundary=rep.get('b_unbounded_ground', False)); ok_u, _ = passed(cf, mech) if cond['a_camera'] else (False, None)
            rows.append(dict(face_type=c['face_type'], pixel_mm=c['pixel_mm'], angle_deg=math.degrees(c['theta']), accepted=ok, first_fail=why,
                             accepted_unbounded_ground=ok_u, conditions=cond, fractions=rep.get('fractions'), visible_px=rep.get('visible_px')))
        acc = [r_ for r_ in rows if r_['accepted']]; accu = [r_ for r_ in rows if r_['accepted_unbounded_ground']]
        hist = lambda rs: np.histogram([r_['pixel_mm'] for r_ in rs], bins=edges)[0].tolist()  # noqa: E731
        fr = [r_['fractions'] for r_ in rows if r_['fractions']]
        cells.append(dict(cell=ci, **cell, draws=draws, accepted=len(acc), accepted_unbounded_ground=len(accu), cannot_fit=len(acc) == 0,
                          first_fail={k: sum(r_['first_fail'] == k for r_ in rows) for k in ORDER if any(r_['first_fail'] == k for r_ in rows)},
                          pixel_subbin_edges_mm=edges.tolist(), drawn_pixel_hist=hist(rows), accepted_pixel_hist=hist(acc),
                          accepted_unbounded_pixel_hist=hist(accu),
                          face_types=dict(drawn={t: sum(r_['face_type'] == t for r_ in rows) for t in ('horizontal', 'side')},
                                          accepted={t: sum(r_['face_type'] == t for r_ in acc) for t in ('horizontal', 'side')}),
                          mean_fractions={k: float(np.mean([x[k] for x in fr])) for k in fr[0]} if fr else None, rows=rows))
    return dict(header(sc, mech), capacity_stream='default_rng([seed, 4213, cell])', seconds=time.time() - t0, cells=cells)


if __name__ == '__main__':
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index('--' + k) + 1] if '--' + k in a else d  # noqa: E731
    mode, mech, scene, net, outp = a[:5]
    net = None if net == '-' else net
    near_mm = g('near-footprint-mm')
    res = plan(mech, scene, net, int(g('per-cell', 1)), g('outcrop'), near=dict(footprint_mm=float(near_mm)) if near_mm else None) if mode == 'plan' else capacity(mech, scene, net, int(g('draws', 0)) or None)
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    Path(outp).write_text(json.dumps(res, indent=1) + '\n', encoding='utf8')
    if mode == 'plan':
        print(json.dumps(dict(qualified=res['qualified'], failed=res['failed_cells'],
                              cameras=[dict(cell=c['cell'], level=c['level'], angle=c['angle'], face=c['face_type'], px_mm=round(c['pixel_mm'], 2),
                                            theta=round(c['angle_deg'], 1), tries=c['tries'], rock=round(c['checks']['fractions']['rock'], 3),
                                            sky=round(c['checks']['fractions']['sky'], 3)) for c in res['cameras']]), indent=0))
    else:
        print(json.dumps(dict(seconds=round(res['seconds']), cells=[{k: v for k, v in c.items() if k != 'rows'} for c in res['cells']])))
