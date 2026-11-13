"""Per-zone refinement-level selection.

Zones: zone_dist.py (half octaves of camera distance of the visible coarse faces). Search: start L_z = clip(6 -
round(log2(median d_z / 2 m)) - 1, 1, 6) (one below auto_levels' start, from below to keep the memory peak low); per
round one refinement + one check: failing zones +1 (a zone failing at 6 -> the camera is not qualified, exit 2),
passing zones -1 unless L - 1 already failed for that zone; stop when nothing changes (every zone passes at L_z and
L_z = 1 or L_z - 1 failed): the last round's mesh is the result. Memory guard: predicted faces
(zone_refine.predicted_faces) > GUARD -> no refinement, exit 3 ("does not fit under zoning"). Seam checks:
zone_refine's geometric checks (seam_pass) on every round; zone_raster_check.py on the result (optionally against a
single-level mesh)."""
import sys as _s; _s.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import nogpu  # noqa: F401,E402
import json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; MS = HERE.parent
M = MS.parent/'frgb'/'chain'
PY = sys.executable
sys.path.append(str(MS.parent))
from frgb import runconfig as rc  # noqa: E402
CNN = rc.path('python.labels').as_posix()
GUARD = 40_000_000
sys.path.insert(0, str(HERE))


def run(args, py=PY, env=None):
    r = subprocess.run([py] + [str(a) for a in args], capture_output=True, text=True, env=env)
    if r.returncode: raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def cnn_env():
    e = dict(os.environ); b = rc.path('env.labels_dir').as_posix()
    e['PATH'] = b + os.pathsep + b + '/Library/bin' + os.pathsep + e.get('PATH', ''); return e


def main(scene, plan_path, cam, out, workers=1, single=None, raster=None):
    import n_zone_refine as zr
    scene, out, cam = Path(scene), Path(out), int(cam); out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(plan_path).read_text(encoding='utf8')); c = plan['cameras'][cam]
    req = out/('request_%d.json' % cam); req.write_text(json.dumps(dict(shots=[c['shot']], resolution=c['resolution'])) + '\n', encoding='utf8')
    ras = out/('cpu_raster_%d.npz' % cam)
    if not ras.is_file():
        if raster: shutil.copyfile(raster, ras)
        else: run([M/'make_source_raster.py', scene/'geom/source_surface.npz', req, ras])
    zn = out/('zones_%d.npz' % cam); t0 = time.time()
    if not zn.is_file(): run([MS/'zone_dist.py', scene/'geom', ras, req, zn], py=CNN, env=cnn_env())
    Z = np.load(zn); info = json.loads(str(Z['info'])); nz = len(info)
    L = [min(6, max(1, 6 - round(math.log2(z['d_median_m'] / 2.)) - 1)) for z in info]
    failed = [set() for _ in range(nz)]; rounds = []; status = None; last = None
    while True:
        banded, _ = zr.banded_regions(Z['zone_dist'], L); pred = zr.predicted_faces(banded)
        rec = dict(levels=list(L), predicted_faces=pred)
        if pred > GUARD:
            rec['skipped'] = 'memory guard'; rounds.append(rec); status = 'does not fit under zoning'; break
        d = out/('round_%d_%d' % (cam, len(rounds)))
        if d.exists(): shutil.rmtree(d)
        t1 = time.time()
        run([HERE/'n_zone_refine.py', scene/'erode_fields.pkl', zn, ','.join(map(str, L)), d, '--geom', scene/'geom', '--workers', workers, '--buffer', 'fallback'])
        run([HERE/'n_zone_shape_check.py', d, scene/'erode_fields.pkl', scene/'geom', req, zn, '--workers', workers])
        ref = json.loads((d/'refinement.json').read_text()); sc = json.loads((d/'zone_shape_check.json').read_text())
        rec.update(faces=ref['faces'], buffer_used=ref['buffer']['used'], seam_pass=ref['seam_pass'], checks=ref['checks'],
                   frame=sc['frame'], zones=[dict(zone=z['zone'], level=L[z['zone']], p99=z['p99'], max=z['max'], pass_=z['pass_']) for z in sc['zones']],
                   PASS=sc['PASS'], seconds=time.time() - t1)
        rounds.append(rec); print(json.dumps({k: v for k, v in rec.items() if k != 'checks'}), flush=True)
        if last is not None: shutil.rmtree(last, ignore_errors=True)
        last = d
        if not ref['seam_pass']: status = 'seam check failed'; break
        new = list(L); stuck = False
        for z in sc['zones']:
            k = z['zone']
            if not z['pass_']:
                failed[k].add(L[k])
                if L[k] >= 6: stuck = True
                else: new[k] = L[k] + 1
            elif L[k] > 1 and (L[k] - 1) not in failed[k]:
                new[k] = L[k] - 1
        if not sc['frame']['pass_'] and sc['PASS'] is False and all(z['pass_'] for z in sc['zones']):
            new = [x + 1 if x < 6 else x for x in L]          # frame-wide fails while every zone passes: refine all (not expected)
        if stuck: status = 'a zone fails at level 6: camera not qualified'; break
        if new == L:
            status = 'settled' if sc['PASS'] else 'no progress'; break
        L = new
    result = dict(camera=cam, distance_m=c['distance_m'], footprint_mm=c['footprint_at_target_mm'], zones=info, rounds=rounds, status=status,
                  chosen_levels=L if status == 'settled' else None, faces=rounds[-1].get('faces') if status == 'settled' else None,
                  seconds=time.time() - t0, rule=__doc__.split('\n')[0])
    if status == 'settled':
        dst = out/('source_%d' % cam)
        if dst.exists(): shutil.rmtree(dst)
        last.rename(dst)
        rc = out/('zone_raster_check_%d.json' % cam)
        run([MS/'zone_raster_check.py', dst, req, rc] + ([single] if single else []))
        result['raster_check'] = json.loads(rc.read_text())
    (out/('zone_levels_%d.json' % cam)).write_text(json.dumps(result, indent=1) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rule', 'rounds')}))
    if status != 'settled': sys.exit(3 if status == 'does not fit under zoning' else 2)


if __name__ == '__main__':
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index('--' + k) + 1] if '--' + k in a else d
    main(a[0], a[1], a[2], a[3], int(g('workers', 1)), g('single'), g('raster'))
