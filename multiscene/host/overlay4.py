"""Loads dsgen_overlay/processes_gt4.py as outcrop.dsgen.processes_gt4 and installs its erode, with groove_tip and
focal_guard set, in place of outcrop.dsgen.processes.erode. outcrop/dsgen/processes.py itself is not modified."""
import nogpu  # noqa: F401,E402  (CPU only)
import functools
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load():
    name = 'outcrop.dsgen.processes_gt4'
    if name in sys.modules:
        return sys.modules[name]
    import outcrop.dsgen  # noqa: F401
    spec = importlib.util.spec_from_file_location(name, HERE/'dsgen_overlay/processes_gt4.py')
    mod = importlib.util.module_from_spec(spec); mod.__package__ = 'outcrop.dsgen'
    sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def install(groove_tip, focal_guard=None):
    import outcrop.dsgen.processes as proc
    mod = load(); original = proc.erode

    @functools.wraps(mod.erode)
    def erode(*a, **k):
        if 'groove_tip' in k or 'focal_guard' in k: raise ValueError('caller already sets groove_tip / focal_guard')
        return mod.erode(*a, **k, groove_tip=groove_tip, focal_guard=focal_guard)
    proc.erode = erode
    return proc, original
