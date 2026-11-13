"""Multi-scene camera plan.

Cells (full factorial within a scene): distance near [1.5, 2.5] m / far [3.0, 5.0] m (camera to target) x angle
between the view axis and the target's outward normal frontal [0, 25] deg / oblique [35, 60] deg; inside a cell both
are uniform. Target: shape = a uniform point (by length) on the scene's exposed fracture paths (network root
source_binding.json, path_XX.npz); stain = an area-uniform point on sandstone faces (surface.npz codes, codebook
lithology 'sandstone'). Normal: area-weighted normal of the surface faces within 0.5 m of the target (not the
micro-relief of one face). Camera: target + d (cos t n + sin t (cos f u + sin f v)), f uniform; looks at the target
(target at the image centre), 35 mm lens, 36 mm filmback, 2048 px, near clip 10 cm. Acceptance
(CPU raster of surface.npz + the scanned ground plane z = GROUND_Z, 512 px; all must hold): - camera at least 0.5 m
above the ground plane; |view axis z| <= 0.8 (the raster's up vector is world z); - unobstructed: the centre pixel
hits the surface within 0.15 m of the target; - rock (non-ground) pixels >= 70 % of the frame; - shape: visible length
of the target's path in the frame >= 512 px at 2048 (stations inside the frame whose depth agrees with the raster
within 0.1 m); stain: sandstone >= 80 % of the central half of the frame and >= 90 % of the faces within 0.5 m of the
target. Usage: python camera_plan.py shape|stain GEOM_DIR NET_ROOT|- SCENE_SEED OUT_JSON"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; P1 = HERE.parent
sys.path.insert(0, str(P1))
from frgb import raster  # noqa: E402

GROUND_Z = 6.
CELLS = [dict(distance='near', angle='frontal'), dict(distance='near', angle='oblique'),
         dict(distance='far', angle='frontal'), dict(distance='far', angle='oblique')]
DIST = dict(near=(1.5, 2.5), far=(3.0, 5.0))
ANGLE = dict(frontal=(0., 25.), oblique=(35., 60.))
LENS = dict(focal_length_mm=35., filmback_mm=[36., 36.], near_clip_cm=10.)
RES, TEST_RES = 2048, 512
RULES = dict(ground_clearance_m=.5, max_abs_axis_z=.8, centre_hit_m=.15, rock_frac=.7, shape_visible_px=512.,
             stain_centre_sand=.8, stain_disc_sand=.9, normal_radius_m=.5, max_tries=400)


def load_scene(geom):
    s = np.load(Path(geom)/'surface.npz')
    V, F, codes = s['vertices'], s['faces'].astype(np.int64), s['codes']
    book = json.loads((Path(geom)/'codebook.json').read_text(encoding='utf8'))
    book = book if isinstance(book, list) else book.get('codebook', book)
    sand = [int(b['code']) for b in book if b.get('lithology') == 'sandstone']
    lo, hi = V.min(0) - 200., V.max(0) + 200.
    G = np.array([[lo[0], lo[1], GROUND_Z], [hi[0], lo[1], GROUND_Z], [hi[0], hi[1], GROUND_Z], [lo[0], hi[1], GROUND_Z]])
    VG, FG = np.r_[V, G], np.r_[F, np.array([[0, 1, 2], [0, 2, 3]]) + len(V)]
    tri = V[F]; cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return dict(V=V, F=F, codes=codes, sand=sand, VG=VG, FG=FG, nfaces=len(F), cent=tri.mean(1), cross=cr, area=np.linalg.norm(cr, axis=1) / 2)


def local_normal(sc, p):
    near = np.linalg.norm(sc['cent'] - p, axis=1) <= RULES['normal_radius_m']
    n = sc['cross'][near].sum(0)
    if not near.any() or np.linalg.norm(n) == 0: raise ValueError('no surface near target')
    return n / np.linalg.norm(n), near


def shot_of(pos, target):
    return dict(position_enu_m=[float(x) for x in pos], target_enu_m=[float(x) for x in target], **LENS)


def test_view(sc, shot):
    cam = raster.camera(shot, (TEST_RES, TEST_RES))
    r = raster.rasterize(sc['VG'], sc['FG'], cam)
    return cam, r


def targets_shape(net, rng):
    b = json.loads((Path(net)/'source_binding.json').read_text(encoding='utf8'))
    rows = b['exposure_paths']; L = np.array([r['length_m'] for r in rows])
    i = int(rng.choice(len(rows), p=L / L.sum())); z = np.load(Path(net)/rows[i]['file'])
    pts, arc = z['points'], z['arc']; s = rng.uniform(0, arc[-1]); k = min(int(np.searchsorted(arc, s, 'right')) - 1, len(arc) - 2)
    t = (s - arc[k]) / (arc[k + 1] - arc[k]); p = pts[k] * (1 - t) + pts[k + 1] * t
    return p, dict(path_index=int(rows[i]['index']), trace_id=rows[i]['source_trace_id'], arc_m=float(s), path_length_m=float(arc[-1]), points=pts)


def targets_stain(sc, rng):
    ok = np.isin(sc['codes'], sc['sand']) & (sc['cent'][:, 2] > GROUND_Z + RULES['ground_clearance_m'])
    idx = np.flatnonzero(ok); f = int(rng.choice(idx, p=sc['area'][idx] / sc['area'][idx].sum()))
    w = rng.dirichlet(np.ones(3)); p = (sc['V'][sc['F'][f]] * w[:, None]).sum(0)
    return p, dict(face=f)


def accept(sc, mech, shot, target, info, cam_pos):
    rep = {}
    if cam_pos[2] < GROUND_Z + RULES['ground_clearance_m']: return False, dict(reason='below ground clearance')
    fwd = np.asarray(target) - cam_pos; fwd /= np.linalg.norm(fwd)
    if abs(fwd[2]) > RULES['max_abs_axis_z']: return False, dict(reason='view axis too steep')
    cam, r = test_view(sc, shot); face = r['face']; c = TEST_RES // 2
    hitc = face[c - 1:c + 1, c - 1:c + 1]
    if (hitc < 0).any() or (hitc >= sc['nfaces']).any(): return False, dict(reason='centre misses the rock')
    dc = r['depth'][c - 1:c + 1, c - 1:c + 1].mean(); dt = float((np.asarray(target) - cam['pos']) @ cam['fwd'])
    if abs(dc - dt) > RULES['centre_hit_m']: return False, dict(reason='target occluded (%.2f m)' % (dt - dc))
    rock = (face >= 0) & (face < sc['nfaces']); rep['rock_frac'] = float(rock.mean())
    if rep['rock_frac'] < RULES['rock_frac']: return False, dict(reason='rock fraction %.2f' % rep['rock_frac'])
    if mech == 'shape':
        P = info['points']; rel = P - cam['pos']; z = rel @ cam['fwd']; zz = np.where(z > cam['near'], z, np.inf)
        x = (rel @ cam['right']) / zz * cam['fpx'] + TEST_RES / 2; y = -(rel @ cam['up']) / zz * cam['fpx'] + TEST_RES / 2
        ins = (z > cam['near']) & (x >= 0) & (x < TEST_RES) & (y >= 0) & (y < TEST_RES)
        xi, yi = np.clip(x.astype(int), 0, TEST_RES - 1), np.clip(y.astype(int), 0, TEST_RES - 1)
        vis = ins & (np.abs(r['depth'][yi, xi] - z) <= .1)
        seg = vis[:-1] & vis[1:]
        rep['visible_px_2048'] = float(np.hypot(np.diff(x), np.diff(y))[seg].sum() * RES / TEST_RES)
        if rep['visible_px_2048'] < RULES['shape_visible_px']: return False, dict(reason='visible path %.0f px' % rep['visible_px_2048'])
    else:
        q = TEST_RES // 4; cen = face[q:3 * q, q:3 * q]; okc = (cen >= 0) & (cen < sc['nfaces'])
        rep['centre_sand'] = float(np.isin(sc['codes'][cen[okc]], sc['sand']).sum() / cen.size)
        _, near = local_normal(sc, np.asarray(target))
        rep['disc_sand'] = float(np.isin(sc['codes'][near], sc['sand']).mean())
        if rep['centre_sand'] < RULES['stain_centre_sand'] or rep['disc_sand'] < RULES['stain_disc_sand']:
            return False, dict(reason='sandstone %.2f / %.2f' % (rep['centre_sand'], rep['disc_sand']))
    return True, rep


def plan(mech, geom, net, seed):
    sc = load_scene(geom); out = []
    for ci, cell in enumerate(CELLS):
        rng = np.random.default_rng([int(seed), 4211, ci]); tries = []
        for k in range(RULES['max_tries']):
            p, info = targets_shape(net, rng) if mech == 'shape' else targets_stain(sc, rng)
            d = rng.uniform(*DIST[cell['distance']]); th = math.radians(rng.uniform(*ANGLE[cell['angle']])); ph = rng.uniform(0, 2 * math.pi)
            n, _ = local_normal(sc, p)
            u = np.cross(n, [0., 0., 1.]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1., 0., 0.]); v = np.cross(n, u)
            w = math.cos(th) * n + math.sin(th) * (math.cos(ph) * u + math.sin(ph) * v); pos = p + d * w
            shot = shot_of(pos, p); ok, rep = accept(sc, mech, shot, p, info, pos)
            tries.append(rep.get('reason', 'ok'))
            if ok:
                info = {kk: vv for kk, vv in info.items() if kk != 'points'}
                fp = d * LENS['filmback_mm'][0] / LENS['focal_length_mm'] / RES
                out.append(dict(cell=ci, **cell, distance_m=float(d), angle_deg=math.degrees(th), azimuth_about_normal_deg=math.degrees(ph),
                                target=info, normal=n.tolist(), shot=shot, resolution=[RES, RES], footprint_at_target_mm=fp * 1e3,
                                checks=rep, tries=k + 1, rejections={r_: tries.count(r_) for r_ in set(tries) if r_ != 'ok'}))
                break
        else:
            return dict(mechanism=mech, geometry=str(geom), seed=int(seed), qualified=False, failed_cell=ci, rules=RULES, cameras=out,
                        rejections={r_: tries.count(r_) for r_ in set(tries)})
    return dict(mechanism=mech, geometry=str(geom), seed=int(seed), qualified=True, rules=RULES, cells=CELLS, distance_m=DIST,
                angle_deg=ANGLE, cameras=out, method=__doc__)


if __name__ == '__main__':
    mech, geom, net, seed, outp = sys.argv[1:6]
    res = plan(mech, geom, None if net == '-' else net, int(seed))
    Path(outp).write_text(json.dumps(res, indent=1) + '\n', encoding='utf8')
    print(json.dumps(dict(qualified=res['qualified'], cameras=[dict(cell=c['cell'], d=round(c['distance_m'], 2), angle=round(c['angle_deg'], 1),
                                                                   tries=c['tries'], target=c['target'], checks=c['checks']) for c in res['cameras']]), indent=0))
