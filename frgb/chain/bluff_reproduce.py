"""Saves the exported fields for the evaluator. With --self-affine SPEC erode runs in the opt-in self-affine relief mode;
the equality checks are then reported, not expected, and the processed/source vertices of the new surface are saved
next to the fields. Usage: python bluff_reproduce.py OUT_PKL [--self-affine SPEC_JSON] [--config C --original G]"""
import json, pickle, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from frgb import runconfig as rc  # noqa: E402
SRC = rc.path('geometry.dev_t02_directions')
GEOM = SRC/'geometry/dev_t02_v2_attempt3'
CONFIG = SRC/'configs/dev_t02_v2_attempt3.json'


def main(out, self_affine=None, config=None, original=None):
    global GEOM, CONFIG
    GEOM, CONFIG = Path(original or GEOM), Path(config or CONFIG)
    t = {}; t0 = time.time()
    from outcrop.dsgen.pipeline import resolve_config, build_declaration
    from outcrop.dsgen.processes import erode
    from outcrop.dsgen.truth import TruthModel
    cfg = resolve_config(json.loads(CONFIG.read_text(encoding='utf8')))
    raw, source = build_declaration(cfg); t['declaration_s'] = time.time() - t0
    ref = np.load(GEOM/'reference_surface.npz')
    rep = dict(reference_vertices_equal=bool(np.array_equal(source['vertices'], ref['vertices'])),
               reference_faces_equal=bool(np.array_equal(source['faces'], ref['faces'])))
    # same preliminary model as generate_scene (the source package itself is reused from the original run)
    preliminary = deepcopy(raw)
    lo, hi = np.asarray(raw['domain']['min_m']), np.asarray(raw['domain']['max_m'])
    y = float((lo[1] + hi[1]) / 2)
    section = next(item for item in preliminary['geometry'] if item['id'] == 'section')
    section.update(vertices=[[lo[0] + .5, y, lo[2] + .5], [hi[0] - .5, y, lo[2] + .5], [hi[0] - .5, y, hi[2] - .5], [lo[0] + .5, y, hi[2] - .5]],
                   faces=[[0, 1, 2], [0, 2, 3]])
    preliminary['model_id'] += '-source-geology'
    truth = TruthModel(GEOM/'source_package')
    surface = source['surface']; items, surface.items = surface.items, []
    debug = dict(export_fields=True)
    t1 = time.time()
    processed = erode(surface, preliminary, truth, np.random.default_rng(cfg['seed'] + 7919), clay=cfg['geometry']['exposure'] == 'bluff',
                      tint_jitter='luma', debug=debug, self_affine=self_affine)
    t['erode_s'] = time.time() - t1
    surface.items = items
    pv = processed['vertices'][:len(ref['vertices'])]
    old_p = np.load(GEOM/'processed_surface.npz'); old_s = np.load(GEOM/'source_surface.npz')
    rep.update(processed_equal=bool(np.array_equal(processed['vertices'], old_p['vertices'])),
               processed_max_abs_m=float(np.abs(processed['vertices'] - old_p['vertices']).max()) if processed['vertices'].shape == old_p['vertices'].shape else None,
               source_vertices_equal=bool(np.array_equal(pv, old_s['vertices'])), source_max_abs_m=float(np.abs(pv - old_s['vertices']).max()),
               fields_V_vs_source_max_abs_m=float(np.abs(debug['fields']['V'] - old_s['vertices']).max()),
               sheet_any=bool(debug['fields']['sheet'].any()), scalars=debug['scalars'], sets=debug['sets'], timing=t)
    patch = source['surface'].patches[0]
    env = dict(source['surface'].patches and {})  # placeholder; the bluff env is the exposure record below
    with open(out, 'wb') as fh:
        pickle.dump(dict(fields=debug['fields'], scalars=debug['scalars'], sets=debug['sets'], F=processed['faces'][:0],
                         reference_faces=ref['faces'], patch_outward=np.asarray(patch.outward), expo=source.get('exposure'),
                         profile=dict(getattr(source['surface'], 'profile', {})), cfg_seed=cfg['seed'], report=rep,
                         self_affine=self_affine, processed_vertices=processed['vertices'], processed_faces=processed['faces'],
                         source_vertices=pv), fh)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('scalars',)}, indent=1, default=str))
    print('scalars', json.dumps(debug['scalars'], default=str))


if __name__ == '__main__':
    a = sys.argv[1:]
    kw = {k: a[a.index('--' + k) + 1] for k in ('config', 'original') if '--' + k in a}
    main(a[0], a[a.index('--self-affine') + 1] if '--self-affine' in a else None, **kw)
