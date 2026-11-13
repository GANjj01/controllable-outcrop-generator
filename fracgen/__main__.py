"""Command line of the controllable outcrop generator.

  python -m fracgen check                       tools and inputs of the run configuration
  python -m fracgen preview CONFIG --seed N --out DIR
  python -m fracgen scene CONFIG --seed N --out DIR [--amplitude A --hurst H --outcrop standalone|flat --faults K]
  python -m fracgen cameras SCENE_DIR
  python -m fracgen frame SCENE_DIR --out DIR [--camera SLOT --width-m W ... --resolution PX --spp N --device CPU|OPTIX|CUDA]
  python -m fracgen example [--out DIR] [--check-only]

Each command prints its own help with -h."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMMANDS = {'preview': ('geometry', 'fracgen.preview'), 'scene': (None, 'fracgen.scene'), 'cameras': (None, 'fracgen.cameras'),
            'frame': ('labels', 'fracgen.frame'), 'example': (None, 'fracgen.example')}


REQUIRED = ('python.geometry', 'python.labels', 'env.geometry_dir', 'env.labels_dir', 'blender', 'manifold_site', 'outcrop_src',
            'relief.spec_C_r2', 'relief.wall_C_r2', 'render.sky_calibration_r1', 'exposure.gate_r14',
            'labels.lithology_aliases_r1')


def check():
    """the tools and inputs the generator needs, and the third-party packages of both environments"""
    from frgb import runconfig as rc
    bad = 0
    print('run configuration:', rc.load()['_path'])
    for name in REQUIRED:
        try:
            p = rc.path(name)
            ok = p.exists()
        except rc.ConfigError as e:
            p, ok = str(e).split(' (')[0], False
        bad += not ok
        print('  %-28s %-8s %s' % (name, 'ok' if ok else 'MISSING', p))
    if not bad:
        from fracgen.env import child_env, python
        probes = dict(geometry='import numpy, OpenEXR, trimesh, PIL, manifold3d; print(numpy.__version__)',
                      labels='import numpy, scipy, shapely, networkx, PIL; print(numpy.__version__, scipy.__version__)')
        for kind, code in probes.items():
            r = subprocess.run([str(python(kind)), '-c', code], env=child_env(kind), capture_output=True, text=True)
            bad += r.returncode != 0
            print('  %-28s %-8s %s' % ('packages (%s)' % kind, 'ok' if r.returncode == 0 else 'MISSING', (r.stdout or r.stderr).strip()[-200:]))
        r = subprocess.run([str(rc.path('blender')), '--version'], capture_output=True, text=True)
        print('  %-28s %-8s %s' % ('blender', 'ok' if r.returncode == 0 else 'FAILED', (r.stdout or '').splitlines()[0] if r.stdout else ''))
        bad += r.returncode != 0
    print('ready' if not bad else '%d problem(s)' % bad)
    return 1 if bad else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'check':
        return check()
    if cmd not in COMMANDS:
        print('unknown command %r\n%s' % (cmd, __doc__))
        return 2
    env_kind, module = COMMANDS[cmd]
    if env_kind is None:
        import importlib
        return importlib.import_module(module).main(rest)
    from fracgen.env import child_env, python
    exe = python(env_kind)
    return subprocess.call([str(exe), '-m', module] + rest, env=child_env(env_kind), cwd=str(ROOT))


if __name__ == '__main__':
    sys.exit(main())
