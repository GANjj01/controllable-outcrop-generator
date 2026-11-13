"""One labelled frame of a generated scene: camera-adaptive refinement of the host, exact field normals, fracture cavities cut into
the host, Blender Cycles rendering and geometry-based trace labels with their checks.

Runs in the labels environment; the geometry steps run as subprocesses in the geometry environment, rendering in Blender.
The stage functions are those of the multi-scene driver (multiscene/driver/run_rq1.py); this module only replaces its memory
scheduling (a shared reservation pool on the original workstation) by a plain sequential run.

Outputs under OUT/<name>/: point/render/frame.npz and frame.png (RGB, depth, material index, region mask), point/trace_labels/
(label layers and report.json), point/labels.npz, checks.json and frame.json (summary)."""
import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fracgen.env import MS, ROOT, child_env, load_json, python, save_json  # noqa: E402

GENES = ('mean_opening_m', 'opening_variation', 'depth_m', 'roughness_m')


def _driver():
    for p in (str(ROOT), str(MS / 'driver')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_rq1 as rq

    @contextlib.contextmanager
    def reserve(kind, quantity, unit, label):
        yield dict(kind=kind, quantity=quantity, unit=unit, reserve_rule='sequential (no pool)')

    rq.reserve = reserve
    if platform.system() != 'Windows':
        class NoJob:
            def assign(self, handle):
                return False

            def peak_gb(self):
                return None

            def close(self):
                pass
        rq.JobPeak = NoJob
    return rq


def _shot(shot):
    return dict(position_enu_m=[float(x) for x in shot['position_enu_m']], target_enu_m=[float(x) for x in shot['target_enu_m']],
                focal_length_mm=float(shot['focal_length_mm']), filmback_mm=[float(x) for x in shot.get('filmback_mm', (36., 36.))],
                near_clip_cm=float(shot.get('near_clip_cm', 10.)))


def point_of(genes, light, spp, camera, sky='sun'):
    return dict(mechanism='shape', unit=None, sky=sky, physical={k: float(genes[k]) for k in GENES},
                light=dict(azimuth_deg=float(light['azimuth_deg']), elevation_deg=float(light['elevation_deg'])),
                spp=int(spp), index='q0', camera=int(camera), slot=None, source='fracgen frame')


def render(rq, st, ctx, cam, spp, device, threads, out, render_seed=0):
    frames = rq.frames_of(ctx['s'], ctx['point'], cam, ctx['geo'], ctx['plan'], ctx['point']['light'], int(render_seed))
    spec = st.dir / 'render_spec.json'
    save_json(spec, dict(out=str(st.dir / 'render'), ledger=None, ledger_purpose='fracgen frame', frames=frames))
    if not (st.dir / 'render/render_point.json').is_file():
        cmd = [python('geometry'), '-B', MS / 'driver/render_server.py', '--device', device, '--threads', threads,
               '--queue', out / 'render_queue']
        command = '\n'.join(json.dumps(x) for x in [dict(op='render', spec=str(spec)), dict(op='quit')]) + '\n'
        env = child_env('geometry')
        with (out / 'logs/render.log').open('a', encoding='utf-8') as log:
            r = subprocess.run([str(c) for c in cmd], input=command, text=True, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError('render failed, see %s' % (out / 'logs/render.log'))
    st.save('render', load_json(st.dir / 'render/render_point.json'), 0.)


def frame(scene_root, out, camera, genes, light, resolution=1024, spp=1024, device='OPTIX', threads=8, name='frame', sky='sun',
          render_seed=0, shot=None):
    """shot: optional camera that replaces the plan camera of the slot (position_enu_m, target_enu_m, focal_length_mm[,
    filmback_mm, near_clip_cm]); sky: 'sun' or 'overcast'; render_seed: the Cycles seed of the frame"""
    scene_root, out = Path(scene_root).resolve(), Path(out).resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    req = load_json(scene_root / 'scene_request.json')
    rq = _driver()
    # refinement, cavity view clip, rendering and labels all use the camera resolution of the plan: a frame-local copy of
    # the plan carries the requested resolution (the plan itself is drawn at 1024 px)
    plan = load_json(scene_root / 'plan.json')
    for c in plan['cameras']:
        if c:
            c['resolution'] = [int(resolution), int(resolution)]
    outcrop = 'flat' if req['outcrop'] == 'flat' else 'isolated'
    if shot is not None:
        c = rq.rm.camera_for_slot(plan['cameras'], outcrop, camera)
        if c is None:
            raise SystemExit('camera slot %d has no plan camera to replace' % camera)
        c['shot'] = _shot(shot)
        d = float(np.linalg.norm(np.subtract(c['shot']['position_enu_m'], c['shot']['target_enu_m'])))
        c['distance_m'] = d
        c['pixel_mm'] = c['footprint_at_target_mm'] = d * 1000. * c['shot']['filmback_mm'][0] / c['shot']['focal_length_mm'] / 1024.
        c['user_shot'] = True
    plan_path = out / ('%s_plan_%dpx.json' % (name, int(resolution)))
    save_json(plan_path, plan)
    s = dict(mechanism='shape', outcrop=outcrop, scene_dir=str(scene_root / 'scene'),
             net=str(scene_root / 'net'), scene_index=0, plan=str(plan_path), qual=None, normal_route_dir='host',
             seed=int(req['seed']), set='custom')
    man = dict(scenes={name: s})
    st = rq.rm.PointState(out, name, 'q0')
    if not st.done('light'):
        st.save('light', point_of(genes, light, spp, camera, sky), 0.)
    t0 = time.time()
    ctx = rq.prepare_point(man, out, name, 'q0', unit=[.5] * 6, camera=camera, sky=sky)
    if ctx.get('final'):
        raise SystemExit('frame stopped before rendering: %s' % json.dumps(ctx, default=str))
    t_prepare = time.time() - t0
    cam = ctx['cam']
    assert list(cam['resolution']) == [int(resolution), int(resolution)], cam['resolution']
    t1 = time.time()
    render(rq, st, ctx, cam, spp, device, threads, out, render_seed)
    t_render = time.time() - t1
    t2 = time.time()
    if not st.done('labels'):
        lab = rq.trace_labels(st, ctx['s'], Path(ctx['sd']), ctx['ci'], ctx['geo'], st.load('refine')['source'],
                              st.load('render')['frames']['frame'])
        st.save('labels', lab, 0.)
    t_labels = time.time() - t2
    from fracgen.checks import frame_checks
    checks = frame_checks(st.dir / 'trace_labels', cam)
    save_json(out / name / 'checks.json', checks)
    summary = dict(scene_root=str(scene_root), camera_slot=int(camera), plan_camera_index=int(ctx['ci']), resolution=cam['resolution'],
                   spp=int(spp), device=device, genes={k: float(genes[k]) for k in GENES}, light=light, sky=sky,
                   render_seed=int(render_seed), shot=cam['shot'], user_shot=shot is not None,
                   pixel_mm=float(ctx['cam'].get('pixel_mm') or 0) * 1024. / float(resolution),
                   seconds=dict(prepare=t_prepare, render=t_render, labels=t_labels),
                   labels={k: v for k, v in st.load('labels').items() if k not in ('out', 'timing_s')}, checks=checks,
                   files=dict(rgb_png=str(st.dir / 'render/frame.png'), render_npz=str(st.dir / 'render/frame.npz'),
                              labels_dir=str(st.dir / 'trace_labels'), labels_npz=str(st.dir / 'labels.npz')))
    save_json(out / name / 'frame.json', summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description='Cut, render and label one frame of a generated scene.')
    ap.add_argument('scene_root')
    ap.add_argument('--out', required=True)
    ap.add_argument('--camera', type=int, default=0, help='camera slot 0-3 (fine frontal / fine oblique, then the second draw of each '
                                                          'on standalone outcrops; the four cells on flat outcrops)')
    ap.add_argument('--width-m', type=float, default=.055, help='mean opening w (validated 0.03-0.08 m)')
    ap.add_argument('--variation', type=float, default=.225, help='relative opening variation (0-0.45)')
    ap.add_argument('--depth-m', type=float, default=.175, help='cavity depth (aspect depth / w within [0.10/w, 0.25/w] in the paper)')
    ap.add_argument('--roughness-m', type=float, help='wall roughness (default 0.0375 w)')
    ap.add_argument('--sun-azimuth', type=float, default=135., help='degrees clockwise from north')
    ap.add_argument('--sun-elevation', type=float, default=35.)
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--spp', type=int, default=1024)
    ap.add_argument('--device', default='OPTIX', choices=['OPTIX', 'CUDA', 'CPU'])
    ap.add_argument('--threads', type=int, default=8, help='CPU render threads')
    ap.add_argument('--name', default='frame')
    ap.add_argument('--sky', default='sun', choices=['sun', 'overcast'], help='sunny (sun from --sun-azimuth / --sun-elevation) or overcast')
    ap.add_argument('--render-seed', type=int, default=0, help='Cycles seed of the rendering (render-noise repeats use 1, 2, ...)')
    ap.add_argument('--position', type=float, nargs=3, metavar=('E', 'N', 'U'), help='camera position (m); with --target and '
                    '--focal-mm it replaces the plan camera of --camera')
    ap.add_argument('--target', type=float, nargs=3, metavar=('E', 'N', 'U'), help='point the camera looks at (m)')
    ap.add_argument('--focal-mm', type=float, help='focal length (mm) of the given camera; film back 36 x 36 mm')
    a = ap.parse_args(argv)
    genes = dict(mean_opening_m=a.width_m, opening_variation=a.variation, depth_m=a.depth_m,
                 roughness_m=a.roughness_m if a.roughness_m is not None else .0375 * a.width_m)
    given = [a.position is not None, a.target is not None, a.focal_mm is not None]
    if any(given) and not all(given):
        ap.error('--position, --target and --focal-mm go together')
    shot = dict(position_enu_m=a.position, target_enu_m=a.target, focal_length_mm=a.focal_mm) if all(given) else None
    s = frame(a.scene_root, a.out, a.camera, genes, dict(azimuth_deg=a.sun_azimuth, elevation_deg=a.sun_elevation), a.resolution,
              a.spp, a.device, a.threads, a.name, a.sky, a.render_seed, shot)
    print(json.dumps(dict(checks=s['checks']['overall'], visible_px=s['labels'].get('visible_px'), files=s['files']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
