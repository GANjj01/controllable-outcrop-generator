"""Outputs per frame: OUT/<case>/<variant>/<light>_s<seed>.{exr,png} and _aux.npz (Blender depth and material index for
later alignment checks). Usage: python sliver_render.py OUT_ROOT VARIANT CASE... (VARIANT = baseline or a name under
OUT_ROOT/<case>/)"""
import json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1]
sys.path[:0] = [str(HERE), str(P1)]
import blender_client as bc  # noqa: E402
from frgb import budget  # noqa: E402
from frgb import runconfig as rc  # noqa: E402

LEDGER = P1/'render_equivalence_ledger_r3.json'
TABLE = rc.path('render.sky_calibration_r1')
SAMPLES = 1024
SEEDS = (0, 1)
LIGHTS = {'L1': (40.7171618035, 20.), 'L2': (315., 45.)}
CAMERA_KEYS = ('position_enu_m', 'target_enu_m', 'focal_length_mm', 'filmback_mm', 'near_clip_cm', 'camera_name')


def shot_of(case):
    request = json.loads((case/'capture/project/ReviewData/request.json').read_text(encoding='utf8'))
    return {k: request['shots'][0][k] for k in CAMERA_KEYS}, request['resolution']


def main(root, variant, cases):
    root = Path(root).resolve(); table = bc.load_table(TABLE); k = bc.exposure_k(table)
    worker = bc.Worker(root/('queue_' + variant), SAMPLES)
    worker.start()
    log = []
    try:
        for c in cases:
            case = Path(c)
            ladder = {}
            for revision in sorted(rc.path('render.sliver_ladders').glob('sliver_ladder_r*.json')):
                ladder.update(json.loads(revision.read_text(encoding='utf8'))['variants'])
            spec = ladder.get(variant, dict(display=variant, shadow_terminator=None))
            display = case/'shading/globaldisplay_surface.npz' if spec['display'] == 'baseline' else root/case.name/spec['display']/'globaldisplay_surface.npz'
            shot, resolution = shot_of(case)
            out = root/case.name/('render_' + variant); out.mkdir(parents=True, exist_ok=True)
            for lname, (az, el) in LIGHTS.items():
                light = bc.light(az, el, table); gains = bc.white_balance_gains(light)
                for seed in SEEDS:
                    stem = '%s_s%d' % (lname, seed)
                    if (out/(stem + '.png')).is_file(): continue
                    budget.charge(LEDGER, 'rgb', 1, 'sliver criterion %s %s %s (Blender %d spp)' % (case.name, variant, stem, SAMPLES))
                    job = dict(id='%s_%s_%s' % (case.name, variant, stem), seed=seed, shot=shot, resolution=resolution, light=light,
                               mesh=dict(key=case.name + '/' + variant, native_surface=str(case/'geometry/native_surface.npz'),
                                         display_surface=str(display), stain_codes=[], shadow_terminator=spec['shadow_terminator']),
                               stain={}, output_exr=str(out/(stem + '.exr')))
                    t0 = time.time(); result = worker.run(job)
                    d = bc.decode(result['output_exr'])
                    rgb8 = bc.to_srgb8(d['rgb'], k, gains)
                    Image.fromarray(rgb8).save(out/(stem + '.png'))
                    np.savez_compressed(out/(stem + '_aux.npz'), depth=d['depth'].astype(np.float32), index=d['index'].astype(np.int16))
                    (out/(stem + '_result.json')).write_text(json.dumps(dict(result, exposure_k=k, wb_gains=gains), indent=1) + '\n')
                    log.append(dict(stem=stem, case=case.name, seconds=time.time() - t0, timing=result['timing']))
                    print('RENDER', case.name, variant, stem, json.dumps(result['timing']), flush=True)
    finally:
        worker.stop()
    (root/('render_log_%s_%d.json' % (variant, int(time.time())))).write_text(json.dumps(log, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
