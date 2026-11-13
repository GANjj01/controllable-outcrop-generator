"""CPU exposure prediction per camera and light direction. Predicts the two exposure-gate quantities of a frame without
rendering: pixel rgb (linear) = rho_rgb / pi * (E_sun_rgb * max(0, n.s) * lit + c_sky * E_sky_h_rgb * (1 + n_z) / 2) 8
bit = round(OETF(pixel * k * white-balance gains of that sun) * 255)  (blender_client.to_srgb8) -> mean luma of rock
pixels and clipped fraction (frgb.lighting_rules: luma, CLIP_LOW, CLIP_HIGH) - geometry: the scene's coarse
surface.npz (lithology codes) + the scanned ground plane, rasterized from the camera at PRED_RES px (pixel-centre
rays); rho = the renderer's uniform base colour worker.GRAY (0.22 linear, every rock code; read from worker.py); n =
face normal of the coarse mesh; a ground-reflection term c_gnd * (E_sun sin(el) + E_sky_h) (1 - n_z) / 2 for faces
looking down; - lit: shadow map along the sun (a camera 2 km away looking along -s with a field just covering the
view's points, SHADOW_RES px), a point is lit when its depth along the sun is within BIAS_M of the map; - sun / sky
irradiance: the frozen sky calibration table (blender_client.sun_rgb, interp_rgb 'sky'); - c_sun, c_sky, c_gnd are
fitted once on the 32 S7 frames (calibrate), stored in light_predict_calibration.json. Usage: python light_predict.py
calibrate OUT_JSON | grid PLAN_JSON CAMERA_INDEX OUT_NPZ | grid_r2 PLAN_R2_JSON CAMERA_INDEX OUT_NPZ [REF_AZ]"""
import json, math, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; P1 = HERE.parent; M = P1/'frgb'/'chain'
sys.path[:0] = [str(M), str(P1), str(HERE)]
import blender_client as bc  # noqa: E402
from frgb import raster  # noqa: E402
from frgb.lighting_rules import CLIP_HIGH, CLIP_LOW, luma  # noqa: E402
from frgb import runconfig as rc  # noqa: E402

PRED_RES, SHADOW_RES, BIAS_M = 256, 1024, .05
GRAY = float(re.search(r'^GRAY = ([0-9.]+)', (M/'worker.py').read_text(encoding='utf8'), re.M).group(1))   # renderer base colour
CFG = json.loads(rc.path('render.backend_config_r1').read_text(encoding='utf8'))
TABLE = bc.load_table(rc.path('render.sky_calibration_r1'))   # the table CFG['sky_table'] names (byte copy)
K = bc.exposure_k(TABLE)
REL_AZ = np.arange(7.5, 360., 15.)             # 24 relative azimuth cells (deg, 0 = sun in front of the face)
ELEV = np.arange(17.5, 70., 5.)                # 11 elevation cells inside the search range 15-70 deg


def srgb_to_lin(c):
    c = np.asarray(c, float) / 255.
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def view_points(V, F, codes, colours, shot, res):
    """pixel hits of the scene from the camera: 3D point, face normal, albedo (linear rgb), rock mask"""
    cam = raster.camera(shot, (res, res)); r = raster.rasterize(V, F, cam)
    face, depth = r['face'], r['depth']
    y, x = np.mgrid[0:res, 0:res]; ray = raster.rays(cam, x, y)
    ray_n = ray / (ray @ cam['fwd'])[..., None]                 # unit view-axis depth
    P = cam['pos'] + ray_n * depth[..., None]
    hit = face >= 0
    tri = V[F[np.maximum(face, 0)]]; n = np.cross(tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :])
    n /= np.maximum(np.linalg.norm(n, axis=-1), 1e-30)[..., None]
    n = np.where(((n * ray).sum(-1) > 0)[..., None], -n, n)     # face the camera
    code = np.where(hit, codes[np.maximum(face, 0)], 0)
    rho = colours[code]
    return dict(P=P, n=n, rho=rho, rock=hit & (code > 0), hit=hit)


def lit_mask(V, F, P, mask, s):
    """shadow map along the sun direction s (unit, toward the sun)"""
    pts = P[mask]; c = pts.mean(0); rad = float(np.linalg.norm(pts - c, axis=1).max()) + .5
    dist = 2000.; pos = c + s * dist
    fov = 2 * math.atan(rad / dist); film = 36.; focal = film / (2 * math.tan(fov / 2))
    shot = dict(position_enu_m=pos.tolist(), target_enu_m=c.tolist(), focal_length_mm=focal, filmback_mm=[film, film], near_clip_cm=100.)
    cam = raster.camera(shot, (SHADOW_RES, SHADOW_RES)); r = raster.rasterize(V, F, cam)
    rel = pts - cam['pos']; z = rel @ cam['fwd']
    xs = (rel @ cam['right']) / z * cam['fpx'] + SHADOW_RES / 2; ys = -(rel @ cam['up']) / z * cam['fpx'] + SHADOW_RES / 2
    xi, yi = np.clip(xs.astype(int), 0, SHADOW_RES - 1), np.clip(ys.astype(int), 0, SHADOW_RES - 1)
    d = r['depth'][yi, xi]
    out = np.zeros(mask.shape, bool); out[mask] = ~np.isfinite(d) | (z <= d + BIAS_M)
    return out


def terms(vp, V, F, az, el):
    """per-pixel linear rgb of the sun and the sky term (before the calibration factors), the white-balance gains, lit"""
    s = np.asarray(bc.toward_sun_enu(az, el), float)
    gains = np.asarray(bc.white_balance_gains(bc.light(az, el, TABLE)))
    En = bc.sun_rgb(TABLE, el); Eh = bc.interp_rgb(TABLE, 'sky', el)
    lit = lit_mask(V, F, vp['P'], vp['hit'], s)
    cos = np.clip((vp['n'] * s).sum(-1), 0, None)
    sun = vp['rho'] * En[None, None, :] * (cos * lit)[..., None] / math.pi
    sky = vp['rho'] * Eh[None, None, :] * ((1 + vp['n'][..., 2]) / 2)[..., None] / math.pi
    gnd = vp['rho'] * (En * math.sin(math.radians(el)) + Eh)[None, None, :] * ((1 - vp['n'][..., 2]) / 2)[..., None] / math.pi
    return dict(sun=sun, sky=sky, gnd=gnd, gains=gains, lit=lit)


def combine(tm, rock, c_sun=1., c_sky=1., c_gnd=0.):
    a = np.round(bc.srgb_oetf((c_sun * tm['sun'] + c_sky * tm['sky'] + c_gnd * tm['gnd']) * K * tm['gains']) * 255.).astype(np.uint8)
    L = luma(a)
    return dict(mean_luma=float(L[rock].mean()), clipped=float(((a[rock] <= CLIP_LOW) | (a[rock] >= CLIP_HIGH)).any(1).mean()),
                lit_fraction=float(tm['lit'][rock].mean()))


def predict(vp, V, F, az, el, c_sun=1., c_sky=1., c_gnd=0.):
    return combine(terms(vp, V, F, az, el), vp['rock'], c_sun, c_sky, c_gnd)


def scene_arrays(geom):
    from camera_plan import load_scene
    sc = load_scene(geom)
    book = json.loads((Path(geom)/'codebook.json').read_text(encoding='utf8'))
    book = book if isinstance(book, list) else book.get('codebook', book)
    colours = np.zeros((max(int(b['code']) for b in book) + 1, 3))
    colours[1:] = GRAY                                  # the renderer's uniform base colour
    codes = np.r_[sc['codes'], [0, 0]]
    return sc['VG'], sc['FG'], codes, colours


def calibrate(out):
    """fit c_sun, c_sky on the S7 frames"""
    import s7_controls as s7
    info = json.loads(rc.path('exposure.info_r2').read_text(encoding='utf8'))
    cams = dict(shape=json.loads(Path(CFG['shape']['camera']).read_text(encoding='utf8')), stain=CFG['stain']['camera'])
    geoms = dict(shape=rc.path('geometry.dev_t02_sa_c2'), stain=rc.path('geometry.fx_n03_sa_c2'))
    rows = []
    for mech in ('shape', 'stain'):
        V, F, codes, colours = scene_arrays(geoms[mech])
        vp = view_points(V, F, codes, colours, cams[mech], PRED_RES)
        for cell in s7.CRIT['cells']:
            lt = s7.absolute(cell, mech); nat = info['cells'][mech][cell['cell']]['native']
            rows.append(dict(mech=mech, cell=cell['cell'], light=lt, real_luma=nat['mean_luma'], real_clipped=nat['clipped'],
                             tm=terms(vp, V, F, lt['azimuth_deg'], lt['elevation_deg']), rock=vp['rock']))
    best = None
    for cs in np.arange(.6, 1.61, .05):
        for ck in np.arange(.5, 2.01, .1):
            for cg in np.arange(0., .61, .05):
                sse = float(sum((combine(r_['tm'], r_['rock'], cs, ck, cg)['mean_luma'] - r_['real_luma']) ** 2 for r_ in rows))
                if best is None or sse < best[0]: best = (sse, float(cs), float(ck), float(cg))
    _, cs, ck, cg = best
    res = []
    for r_ in rows:
        p = combine(r_['tm'], r_['rock'], cs, ck, cg)
        res.append(dict(mech=r_['mech'], cell=r_['cell'], light=r_['light'], real_luma=r_['real_luma'], pred_luma=p['mean_luma'],
                        real_clipped=r_['real_clipped'], pred_clipped=p['clipped'], lit_fraction=p['lit_fraction']))
    e = np.array([x['pred_luma'] - x['real_luma'] for x in res])
    rep = dict(method=__doc__, c_sun=cs, c_sky=ck, c_gnd=cg, gray=GRAY, pred_res=PRED_RES, shadow_res=SHADOW_RES, rows=res,
               luma_error=dict(mean=float(e.mean()), rms=float(np.sqrt((e ** 2).mean())), max_abs=float(np.abs(e).max())))
    Path(out).write_text(json.dumps(rep, indent=1) + '\n', encoding='utf8')
    print('c_sun %.2f c_sky %.2f c_gnd %.2f' % (cs, ck, cg), rep['luma_error'])
    for x in res: print('%s %s real %.3f pred %.3f clip %.3f/%.3f lit %.2f' % (x['mech'], x['cell'], x['real_luma'], x['pred_luma'], x['real_clipped'], x['pred_clipped'], x['lit_fraction']))


if __name__ == '__main__':
    if sys.argv[1] == 'calibrate':
        calibrate(sys.argv[2])


def facing_azimuth(vp):
    """compass azimuth (0 = N = +y, 90 = E = +x) of the mean outward normal of the rock pixels (the S5 rule, per camera)"""
    n = vp['n'][vp['rock']].mean(0)
    return float(math.degrees(math.atan2(n[0], n[1])) % 360.)


def grid(plan_path, cam_index, out):
    """predicted gate quantities for every (relative azimuth, elevation) cell of one camera; accept = predicted mean luma >=
    gate - DELTA and clipped <= the clip gate (DELTA = 2 x calibration rms: the predictor is allowed to err toward keeping a
    direction; a kept direction that fails is caught by the exposure gate of the rendered frame, qualification G)"""
    plan = json.loads(Path(plan_path).read_text(encoding='utf8')); c = plan['cameras'][int(cam_index)]
    cal = json.loads((HERE/'light_predict_calibration.json').read_text(encoding='utf8'))
    gate = json.loads(rc.path('exposure.gate_r14').read_text(encoding='utf8'))
    g = gate.get('gate', gate); lmin = float(g['mean_luma_min']); cmax = float(g['clipped_fraction_max'])
    delta = 2 * cal['luma_error']['rms']
    V, F, codes, colours = scene_arrays(plan['geometry']); vp = view_points(V, F, codes, colours, c['shot'], PRED_RES)
    fa = facing_azimuth(vp)
    L = np.zeros((len(REL_AZ), len(ELEV))); C = np.zeros_like(L); LIT = np.zeros_like(L)
    for i, a in enumerate(REL_AZ):
        for j, e in enumerate(ELEV):
            p = combine(terms(vp, V, F, (fa + a) % 360., e), vp['rock'], cal['c_sun'], cal['c_sky'], cal['c_gnd'])
            L[i, j], C[i, j], LIT[i, j] = p['mean_luma'], p['clipped'], p['lit_fraction']
    acc = (L >= lmin - delta) & (C <= cmax)
    np.savez_compressed(out, rel_az_centres_deg=REL_AZ, elev_centres_deg=ELEV, cell_deg=np.array([15., 5.]), mean_luma=L, clipped=C,
                        lit_fraction=LIT, accept=acc, facing_azimuth_deg=fa, delta=delta, gate=np.array([lmin, cmax]),
                        meta=json.dumps(dict(plan=str(plan_path), camera=int(cam_index), calibration_c=[cal['c_sun'], cal['c_sky'], cal['c_gnd']])))
    print(json.dumps(dict(camera=int(cam_index), facing=fa, accepted=int(acc.sum()), cells=int(acc.size), luma=[float(L.min()), float(L.max())])))


if __name__ == '__main__' and sys.argv[1] == 'grid':
    grid(sys.argv[2], sys.argv[3], sys.argv[4])


def grid_r2(plan_path, cam_index, out, ref_az=None):
    plan = json.loads(Path(plan_path).read_text(encoding='utf8')); c = plan['cameras'][int(cam_index)]
    ref = dict(azimuth_deg=float(ref_az), kind='given') if ref_az is not None else c.get('light_reference')
    if not ref: raise ValueError('camera has no light_reference (stain: pass the band-plane strike azimuth)')
    cal = json.loads((HERE/'light_predict_calibration.json').read_text(encoding='utf8'))
    gate = json.loads(rc.path('exposure.gate_r14').read_text(encoding='utf8'))
    g = gate.get('gate', gate); lmin = float(g['mean_luma_min']); cmax = float(g['clipped_fraction_max'])
    delta = 2 * cal['luma_error']['rms']
    V, F, codes, colours = scene_arrays(plan['geometry']); vp = view_points(V, F, codes, colours, c['shot'], PRED_RES)
    L = np.zeros((len(REL_AZ), len(ELEV))); C = np.zeros_like(L); LIT = np.zeros_like(L)
    for i, a in enumerate(REL_AZ):
        for j, e in enumerate(ELEV):
            p = combine(terms(vp, V, F, (ref['azimuth_deg'] + a) % 360., e), vp['rock'], cal['c_sun'], cal['c_sky'], cal['c_gnd'])
            L[i, j], C[i, j], LIT[i, j] = p['mean_luma'], p['clipped'], p['lit_fraction']
    acc = (L >= lmin - delta) & (C <= cmax)
    np.savez_compressed(out, rel_az_centres_deg=REL_AZ, elev_centres_deg=ELEV, cell_deg=np.array([15., 5.]), mean_luma=L, clipped=C,
                        lit_fraction=LIT, accept=acc, reference_azimuth_deg=ref['azimuth_deg'], facing_azimuth_deg=facing_azimuth(vp), delta=delta,
                        gate=np.array([lmin, cmax]),
                        meta=json.dumps(dict(plan=str(plan_path), camera=int(cam_index), reference=ref, rule='light domain r2 (strike reference)',
                                             calibration_c=[cal['c_sun'], cal['c_sky'], cal['c_gnd']])))
    print(json.dumps(dict(camera=int(cam_index), reference=ref['azimuth_deg'], kind=ref['kind'], accepted=int(acc.sum()), cells=int(acc.size),
                          luma=[float(L.min()), float(L.max())])))


if __name__ == '__main__' and sys.argv[1] == 'grid_r2':
    grid_r2(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else None)
