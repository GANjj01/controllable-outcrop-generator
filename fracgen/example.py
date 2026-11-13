"""End-to-end example: configuration -> 3-D scene -> camera plan -> cut, render (CPU) and label one frame -> checks.

The example is one standalone outcrop of the paper's design (seed 26130000) with the paper's mid-range fracture genes, rendered at
256 x 256 px with 32 samples on the CPU so that it runs on an ordinary workstation (about 16 minutes, 8 GB of memory, 0.8 GB of
disk). examples/minimal/expected.json holds the reference results; `--check-only` compares an existing output directory."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fracgen.env import ROOT, child_env, load_json, python, save_json  # noqa: E402

EX = ROOT / 'examples' / 'minimal'


def run(out):
    from fracgen import cameras, scene
    p = load_json(EX / 'example.json')
    out = Path(out).resolve()
    sroot = out / 'scene'
    scene.generate(EX / 'scene_config.json', p['seed'], sroot, p['amplitude'], p['hurst'], p['outcrop'])
    if not (sroot / 'plan.json').is_file():
        cameras.plan(sroot, p['outcrop'], p['cameras_per_cell'])
    g, light, r = p['genes'], p['light'], p['render']
    cmd = [python('labels'), '-m', 'fracgen.frame', sroot, '--out', out / 'frames', '--camera', p['camera_slot'],
           '--width-m', g['mean_opening_m'], '--variation', g['opening_variation'], '--depth-m', g['depth_m'],
           '--roughness-m', g['roughness_m'], '--sun-azimuth', light['azimuth_deg'], '--sun-elevation', light['elevation_deg'],
           '--resolution', r['resolution'], '--spp', r['spp'], '--device', r['device'], '--threads', r['threads'], '--name', 'example']
    rc = subprocess.call([str(c) for c in cmd], env=child_env('labels'), cwd=str(ROOT))
    if rc:
        raise SystemExit('frame step failed (exit %d)' % rc)
    return out


def digest(path, names=None):
    import numpy as np
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(z.files):
            if names is None or k in names:
                out[k] = hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()
    return out


def summarise(out, surface=None, plan=None, point=None):
    """example statistics; the three paths default to the layout of `fracgen example --out OUT`"""
    import numpy as np
    from fracgen.checks import frame_checks
    out = Path(out)
    surface = Path(surface or out / 'scene/scene/geom/surface.npz')
    plan = Path(plan or out / 'scene/plan.json')
    point = Path(point or out / 'frames/example/q0')
    checks = frame_checks(point / 'trace_labels')
    rgb = np.load(point / 'render/frame.npz')
    lab = {k: np.load(point / 'trace_labels' / (k + '.npy')) for k in ('target', 'valid', 'u_line', 'occluded_line', 'ignore_reason')}
    surf = np.load(surface)
    return dict(
        scene=dict(surface_vertices=int(surf['vertices'].shape[0]), surface_faces=int(surf['faces'].shape[0]),
                   surface_sha256=digest(surface, ('vertices', 'faces'))),
        cameras=len([c for c in load_json(plan)['cameras'] if c]),
        render=dict(shape=list(rgb['rgb'].shape), rock_fraction=float((rgb['mask'] == 1).mean()),
                    mean_linear_rgb=float(rgb['rgb'][rgb['mask'] == 1].mean()), arrays_sha256=digest(point / 'render/frame.npz')),
        labels=dict(positive_pixels=int(lab['target'].sum()), valid_pixels=int(lab['valid'].sum()), u_line_pixels=int(lab['u_line'].sum()),
                    occluded_pixels=int(lab['occluded_line'].sum()), marked_pixels=int((lab['ignore_reason'] > 0).sum()),
                    visible_length_px=checks['visible_length_px'],
                    arrays_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in lab.items()}),
        checks=dict(overall=checks['overall'], label_status=checks['label_status'], items=checks['label_items']))


def compare(got, ref):
    rows = []

    def row(name, a, b, ok):
        rows.append(dict(item=name, got=a, expected=b, ok=bool(ok)))
    row('checks overall', got['checks']['overall'], ref['checks']['overall'], got['checks']['overall'] == ref['checks']['overall'])
    row('label status', got['checks']['label_status'], ref['checks']['label_status'], got['checks']['label_status'] == ref['checks']['label_status'])
    for k in ('surface_vertices', 'surface_faces'):
        row('scene ' + k, got['scene'][k], ref['scene'][k], abs(got['scene'][k] - ref['scene'][k]) <= .001 * ref['scene'][k])
    row('cameras in plan', got['cameras'], ref['cameras'], got['cameras'] == ref['cameras'])
    row('render rock fraction', got['render']['rock_fraction'], ref['render']['rock_fraction'],
        abs(got['render']['rock_fraction'] - ref['render']['rock_fraction']) <= .01)
    row('render mean linear RGB on rock', got['render']['mean_linear_rgb'], ref['render']['mean_linear_rgb'],
        abs(got['render']['mean_linear_rgb'] / ref['render']['mean_linear_rgb'] - 1) <= .02)
    for k in ('positive_pixels', 'u_line_pixels', 'occluded_pixels', 'marked_pixels'):
        a, b = got['labels'][k], ref['labels'][k]
        row('labels ' + k, a, b, abs(a - b) <= max(3, .05 * b))
    row('labels visible length (px)', got['labels']['visible_length_px'], ref['labels']['visible_length_px'],
        abs(got['labels']['visible_length_px'] / ref['labels']['visible_length_px'] - 1) <= .02)
    bitwise = dict(scene=got['scene']['surface_sha256'] == ref['scene']['surface_sha256'],
                   render=got['render']['arrays_sha256'] == ref['render']['arrays_sha256'],
                   labels=got['labels']['arrays_sha256'] == ref['labels']['arrays_sha256'])
    return dict(passed=all(r['ok'] for r in rows), rows=rows, bitwise_identical_to_reference=bitwise)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(ROOT / 'example_output'))
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--write-expected', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if not a.check_only:
        run(out)
    cmd = [python('labels'), '-c', 'import json,sys; from fracgen.example import summarise; print(json.dumps(summarise(sys.argv[1])))', out]
    got = json.loads(subprocess.check_output([str(c) for c in cmd], env=child_env('labels'), cwd=str(ROOT), text=True))
    save_json(out / 'example_summary.json', got)
    if a.write_expected:
        save_json(EX / 'expected.json', got)
        print('expected values written')
        return 0
    res = compare(got, load_json(EX / 'expected.json'))
    save_json(out / 'example_check.json', res)
    for r in res['rows']:
        print('%-36s %-6s got %-22s expected %s' % (r['item'], 'ok' if r['ok'] else 'FAIL', r['got'], r['expected']))
    print('bitwise identical to the reference run:', res['bitwise_identical_to_reference'])
    print('EXAMPLE', 'PASS' if res['passed'] else 'FAIL')
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
