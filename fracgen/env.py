"""Locations, interpreters and subprocess environments of the generator.

Every tool and input is looked up by logical name in the run configuration (frgb/runconfig.py): inputs/runconfig_default.json,
overridden by inputs/runconfig_local.json or by the file named in FRGB_RUNCONFIG."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / 'multiscene'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frgb import runconfig as rc  # noqa: E402

SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def python(kind):
    """interpreter of an environment: 'geometry' (OpenEXR, trimesh, manifold3d) or 'labels' (SciPy, Shapely)"""
    return rc.path('python.' + kind)


def child_env(kind):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', **SINGLE_THREAD)
    e['PYTHONPATH'] = rc.pythonpath()
    e['PATH'] = rc.env_dir_path(rc.path('env.%s_dir' % kind))
    return e


def run(cmd, kind, log, cwd=None):
    """run one step in an environment; the log keeps stdout and stderr; a failure raises with the end of the log"""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(c) for c in cmd]
    t = time.time()
    with log.open('w', encoding='utf-8') as fh:
        fh.write('$ ' + ' '.join(cmd) + '\n')
        fh.flush()
        r = subprocess.run(cmd, cwd=cwd, env=child_env(kind), stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = log.read_text(encoding='utf-8', errors='replace')[-1500:]
        raise RuntimeError('%s failed (exit %d), log %s:\n%s' % (Path(cmd[1]).name, r.returncode, log, tail))
    return time.time() - t


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save_json(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=1, default=str) + '\n', encoding='utf-8')
