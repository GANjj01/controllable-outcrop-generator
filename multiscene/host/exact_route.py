"""Route-aware exact evaluator for the refinement and check scripts: BluffExact for the ray route, NormalExact
(normal_exact.py) when the exported erode fields are offset sheets. Same interface (coarse_carry, evaluate); worker
hooks init / eval_V mirror bluff_refine._init / _eval. precise=True: field_normals' 52-iteration ray bisection for the
ray route (the normal route has no bisection)."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parent)) if str(_p.Path(__file__).resolve().parent) not in _s.path else None
import nogpu  # noqa: F401,E402  (CPU only)
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M = next(p for p in HERE.parents if (p/'frgb'/'chain').is_dir())/'frgb'/'chain'
for p in (str(M), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
CARRY = ('meso', 'w', 'fade', 'recess', 'bare_w', 'gscale')
_E = None


def route_of(fields_pkl):
    d = pickle.load(open(fields_pkl, 'rb'))
    normal = np.asarray(d['fields']['sheet']).all() or bool((d.get('expo') or {}).get('normal_route'))
    if not normal and d['scalars'].get('groove_tip') is not None:
        raise ValueError('groove_tip fields on the ray route: BluffExact does not mirror it')
    return ('normal' if normal else 'ray'), d


def make_exact(fields_pkl, geom=None, precise=False, iterations=52):
    route, d = route_of(fields_pkl)
    if route == 'normal':
        from normal_exact import NormalExact
        return NormalExact(d, geom)
    from bluff_exact import BluffExact
    if not precise:
        return BluffExact(fields_pkl, geom)
    from outcrop.dsgen.morph import radial_surface

    class Precise(BluffExact):
        def solve(self, d_, off):
            return radial_surface(d_, np.asarray(off, float), self.env, iterations=iterations)[0]
    return Precise(fields_pkl, geom)


def init(fields_pkl, geom=None, precise=False):
    global _E
    _E = make_exact(fields_pkl, geom, precise)


def eval_V(chunk):
    return _E.evaluate(chunk)['V']
