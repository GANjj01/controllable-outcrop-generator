"""Field normals of a refined host.

The display normal at a refined vertex is taken from the EXACT eroded surface, not from the mesh, so it does not
depend on the refinement level. - Parameterisation (exactly that of bluff_refine): a refined vertex lies in one coarse
source face (its ancestor); its carried state (ray direction d and the smooth erode fields meso, w, fade, recess,
bare_w, gscale) is the barycentric combination of the ancestor's corner states (red-green midpoints average the state
linearly). The barycentric coordinates are recovered from the stored (unnormalised) direction by a 3 x 3 solve; the
reconstructed state re-evaluates the vertex (reported). - Normal: central differences on the exact surface along two
barycentric directions, +-eps with eps = DELTA_M / (coarse edge length), the fields interpolated at the perturbed
points as well (so the gradients of recess / gscale count); n = (S(b+e1) - S(b-e1)) x (S(b+e2) - S(b-e2)), oriented
like the mesh vertex normal. This is a finite difference of the exact field at a step far below the shortest relief
wavelength (2.4 cm), not a closed-form derivative; the ray bisection is run with ITERATIONS = 52 here (the generator's
20 resolve 22 um, too coarse for a 0.5 mm difference). Convergence is reported (DELTA_M vs DELTA_M / 2 on a random
sample). - Scope: vertices of faces whose ancestor lies within DISTANCES[0] = 1.5 m of the faces seen at the camera
(the refined region); every other vertex keeps its mesh normal (counted)."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parent)) if str(_p.Path(__file__).resolve().parent) not in _s.path else None
import nogpu  # noqa: F401,E402  (CPU only)
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; M = HERE.parents[1]/'frgb'/'chain'
sys.path[:0] = [str(M), str(HERE.parents[1]), str(HERE)]
import exact_route as er  # noqa: E402
ITERATIONS = 52
DELTA_M = 5e-4
SCOPE_M = 1.5
CARRY = ('meso', 'w', 'fade', 'recess', 'bare_w', 'gscale')
_E = None


def _exact(fields, geom):
    return er.make_exact(fields, geom, precise=True, iterations=ITERATIONS)


def _init(fields, geom):
    global _E
    _E = _exact(fields, geom)


def _state(c0, corners, b):
    out = {'d': np.einsum('ni,nij->nj', b, c0['d'][corners])}
    for k in CARRY:
        a = c0[k][corners]
        out[k] = np.einsum('ni,nij->nj', b, a) if a.ndim == 3 else np.einsum('ni,ni->n', b, a)
    return out


def _normals(job):
    c0, corners, b, eps, vn = job
    pts = []
    for e in (np.array([-1., 1, 0]), np.array([-1., 0, 1])):
        for s in (1., -1.):
            pts.append(_E.evaluate(_state(c0, corners, b + s * eps[:, None] * e))['V'])
    n = np.cross(pts[0] - pts[1], pts[2] - pts[3]); n /= np.linalg.norm(n, axis=1)[:, None]
    return np.where((np.einsum('ij,ij->i', n, vn) < 0)[:, None], -n, n)


def _reeval(job):
    c0, corners, b = job
    return _E.evaluate(_state(c0, corners, b))['V']


def main(src, fields, geom, out, workers=6, delta=DELTA_M):
    t0 = time.time(); src, geom = Path(src), Path(geom)
    S = np.load(src/'source_surface.npz'); R = np.load(src/'refinement.npz')
    V, F, vn = S['vertices'], S['faces'].astype(np.int64), S['normals']; anc, D = R['ancestor_face'], R['direction']
    dist = R['distance_to_visible_m']
    C = np.load(geom/'source_surface.npz'); sv, sf = C['vertices'], C['faces'].astype(np.int64)
    ex = _exact(fields, geom); c0 = ex.coarse_carry()
    vf = np.full(len(V), -1, np.int64); vf[F.ravel()] = np.repeat(np.arange(len(F)), 3)
    a = anc[vf]; corners = sf[a]
    b = np.linalg.solve(np.transpose(c0['d'][corners], (0, 2, 1)), D[:, :, None])[:, :, 0]
    in_scope = np.zeros(len(V), bool); in_scope[F[dist[anc] <= SCOPE_M].ravel()] = True
    idx = np.flatnonzero(in_scope)
    L = np.stack([np.linalg.norm(sv[corners[:, 1]] - sv[corners[:, 0]], axis=1), np.linalg.norm(sv[corners[:, 2]] - sv[corners[:, 0]], axis=1)], 1).min(1)
    eps = delta / L
    rng = np.random.default_rng(26102911)
    samp = rng.choice(idx, min(len(idx), 4000), replace=False)
    chunk = 20000
    jobs = [(c0, corners[idx[i:i + chunk]], b[idx[i:i + chunk]], eps[idx[i:i + chunk]], vn[idx[i:i + chunk]]) for i in range(0, len(idx), chunk)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(fields), str(geom))) as pool:
        parts = list(pool.map(_normals, jobs))
        half = pool.submit(_normals, (c0, corners[samp], b[samp], eps[samp] / 2, vn[samp])).result()
        rev = pool.submit(_reeval, (c0, corners[samp], b[samp])).result()
    N = vn.astype(np.float64).copy()
    if parts: N[idx] = np.concatenate(parts)
    ang = lambda x, y: np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', x, y), -1, 1)))
    q = lambda x: {('p%g' % p): float(np.percentile(x, p)) for p in (50, 90, 99, 99.9)} | dict(max=float(x.max()))
    rep = dict(method=__doc__, source=str(src), fields=str(fields), geom=str(geom), iterations=ITERATIONS, delta_m=delta, scope_m=SCOPE_M,
               vertices=int(len(V)), computed=int(len(idx)), kept_mesh_normal=int(len(V) - len(idx)),
               barycentric=dict(min=float(b[idx].min()), max=float(b[idx].max()), sum_dev_max=float(np.abs(b[idx].sum(1) - 1).max())),
               reevaluation_vs_vertex_m=q(np.linalg.norm(rev - V[samp], axis=1)),
               convergence_delta_vs_half_deg=q(ang(N[samp], half)),
               field_vs_mesh_normal_deg=q(ang(N[idx], vn[idx] / np.linalg.norm(vn[idx], axis=1)[:, None])),
               seconds=time.time() - t0, workers=workers, route=getattr(ex, 'route', 'ray'))
    np.savez_compressed(out, normals=N, computed=in_scope)
    Path(str(out) + '.json').write_text(json.dumps(rep, indent=1) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'method'}))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], int(a[a.index('--workers') + 1]) if '--workers' in a else 6,
         float(a[a.index('--delta-m') + 1]) if '--delta-m' in a else DELTA_M)
