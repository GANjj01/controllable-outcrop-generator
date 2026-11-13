"""B6: the cut chain's export step without the rim-edge record.

Runs the network's own export.py (imported from the net, not copied or edited) with its save() skipping only
geometry/rim_candidates.json (status NOT_ACCEPTED_TRAINING_LABELS; no gate, label or score reads it). Usage: python
export_norim.py NAME [--timing JSON]"""
import importlib.util, inspect, json, sys, time
from pathlib import Path

SKIP = 'rim_candidates.json'


def load(net):
    sys.path.insert(0, str(net))
    spec = importlib.util.spec_from_file_location('export', net/'export.py'); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


def main(argv):
    name, timing = argv[0], (argv[argv.index('--timing') + 1] if '--timing' in argv else None)
    glb = '--glb-previews' in argv
    mod = load(Path.cwd()); _save = mod.save; seconds = {}

    def timed(key, fn):
        def run(*a, **k):
            t = time.time(); r = fn(*a, **k); seconds[key] = seconds.get(key, 0.) + time.time() - t; return r
        return run if timing else fn

    def save(path, obj):
        if Path(path).name == SKIP: return None
        return _save(path, obj)
    mod.save = timed('save', save)
    if not glb:
        mod._native_preview = mod._preview = lambda *a, **k: None
    elif timing:
        mod._native_preview, mod._preview = timed('native_preview_glb', mod._native_preview), timed('gray_preview_glb', mod._preview)
    keep = dict(glb_previews=True) if glb and 'glb_previews' in inspect.signature(mod.export).parameters else {}
    t0 = time.time(); mod.export(name, **keep)
    if timing: Path(timing).write_text(json.dumps(dict(seconds, total=time.time() - t0)) + '\n', encoding='utf8')


if __name__ == '__main__':
    main(sys.argv[1:])
