"""Scene generation: one outcrop with its joint network (and optional faults) from a configuration, a seed and the two
background-relief parameters (amplitude a, spectral slope H).

Steps (each in the geometry environment, CPU only, single-threaded maths libraries):
  1. effective configuration: the requested number of randomly placed faults and the meshing budget of the cavity builder;
  2. host generation in four phases (spec, generate, fields, wall): layered rock body, outcrop surface on the normal-offset
     route, calibrated self-affine relief, eroded joint grooves, exact surface fields and the cavity-wall relief;
  3. cavity preparation (net/): the exposed fracture paths and the source bindings the cavity builder uses.
Outputs OUT/scene (host), OUT/net (cavity inputs), OUT/config_effective.json, OUT/times.json and OUT/logs/."""
import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fracgen.env import MS, load_json, python, run, save_json  # noqa: E402

GROOVE_TIP_M, FOCAL_GUARD = '0.2', '0.1'


def fault_count(cfg):
    geo = cfg.get('geometry', {})
    if geo.get('faults') is not None:
        return 0
    return int(geo.get('fault_count', 0))


def generate(config, seed, out, amplitude, hurst, outcrop, mechanism='shape', faults=None):
    """outcrop: 'standalone' (isolated bluff) or 'flat' (tiled slab); faults: number of randomly placed faults (default: the
    configuration's fault_count; explicit faults listed in geometry.faults are always kept)"""
    out = Path(out).resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    py = python('geometry')
    times = load_json(out / 'times.json') if (out / 'times.json').is_file() else {}
    cfg = load_json(config)
    k = fault_count(cfg) if faults is None else int(faults)
    eff = out / 'config_effective.json'
    if not eff.is_file() and cfg.get('geometry', {}).get('faults') is not None:
        # explicit faults: the effective configuration is written directly (the random-fault consistency check of
        # fs_effective_config.py compares event orders that differ by design when user faults are listed)
        e = json.loads(json.dumps(cfg))
        e['seed'] = int(seed)
        e.get('geometry', {}).pop('outcrop_type', None)
        e['geometry']['fault_count'] = k
        if mechanism == 'shape':
            e['geometry']['meshing'] = dict(budget=dict(max_planes_per_cell=1024))
        save_json(eff, e)
        save_json(out / 'nfaults.json', dict(seed=int(seed), random_faults=k, explicit_faults=[f['id'] for f in cfg['geometry']['faults']]))
    if not eff.is_file():
        cmd = [py, MS / 'scene_sets/fs_effective_config.py', Path(config).resolve(), seed, eff, out / 'nfaults.json', '--set', k]
        if mechanism == 'shape':
            cmd += ['--planes', 1024]
        times['effective_config'] = run(cmd, 'geometry', out / 'logs/effective_config.log')
    edge = load_json(MS / 'host/edge_ref.json')['grid_spacing_m']
    for phase in ('spec', 'generate', 'fields', 'wall'):
        if phase in times:
            continue
        cmd = [py, MS / 'host/gen_normal4.py', out / 'scene', '--config', eff, '--seed', seed, '--amplitude', repr(float(amplitude)),
               '--hurst', repr(float(hurst)), '--phase', phase, '--edge-m', edge]
        if outcrop == 'flat':
            cmd.append('--exact-slab')
        cmd += ['--groove-tip-m', GROOVE_TIP_M, '--focal-guard', FOCAL_GUARD, '--focal-local']
        times[phase] = run(cmd, 'geometry', out / 'logs' / (phase + '.log'))
        save_json(out / 'times.json', times)
    if mechanism == 'shape' and 'prepare' not in times:
        net = out / 'net'
        net.mkdir(exist_ok=True)
        for f in sorted((MS / 'cavity_template').glob('*.py')):
            if not (net / f.name).exists():
                shutil.copyfile(f, net / f.name)
        (net / 'source.txt').write_text((out / 'scene/geom').as_posix() + '\n', encoding='utf-8')
        times['prepare'] = run([py, 'prepare.py'], 'geometry', out / 'logs/prepare.log', cwd=net)
        save_json(out / 'times.json', times)
    save_json(out / 'scene_request.json', dict(config=str(Path(config).resolve()), seed=int(seed), amplitude=float(amplitude),
                                               hurst=float(hurst), outcrop=outcrop, mechanism=mechanism, random_faults=k))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Generate one outcrop scene (host rock, joints, faults, cavity inputs).')
    ap.add_argument('config', help='scene configuration JSON (see configs/scenes)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--amplitude', type=float, default=1.0, help='background relief amplitude a (paper range 0.85-2.55)')
    ap.add_argument('--hurst', type=float, default=0.718, help='background relief spectral slope H (paper range 0.50-1.25)')
    ap.add_argument('--outcrop', choices=['standalone', 'flat'], default='standalone')
    ap.add_argument('--faults', type=int, help='number of randomly placed faults (default: geometry.fault_count)')
    a = ap.parse_args(argv)
    generate(a.config, a.seed, a.out, a.amplitude, a.hurst, a.outcrop, faults=a.faults)
    print('scene ready:', Path(a.out).resolve())
    return 0


if __name__ == '__main__':
    sys.exit(main())
