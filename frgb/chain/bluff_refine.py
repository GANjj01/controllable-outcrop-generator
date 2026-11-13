"""Bluff centimetre route: view-adaptive refinement of the source surface with EXACT positions at every new
vertex.

Graded red-green refinement (adaptive_refine.refine_level, no T-junctions) of the source topology around the source
faces seen at the camera. Linear interpolation applies only to the ray direction and to the smooth erode fields (meso,
w, fade, recess, bare_w, gscale) and to the soil cover weight; every new vertex position comes from
BluffExact.evaluate (re-solved along its ray), evaluated in parallel worker processes. Coarse vertices keep their
exact source positions (checked bit-for-bit by the evaluator self-test). Output: a derived source directory for
build.py --source. Usage: python bluff_refine.py FIELDS_PKL CPU_RASTER_NPZ OUT_DIR [--workers N] [--geom DIR]
[--levels N]"""
import hashlib, json, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adaptive_refine import refine_level  # noqa: E402

DISTANCES = (1.5, 1.0, .65, .4, .25, .1)
CARRY = ('meso', 'w', 'fade', 'recess', 'bare_w', 'gscale')
_E = None


def _init(fields_pkl, geom=None):
    global _E
    from bluff_exact import BluffExact
    _E = BluffExact(fields_pkl, geom)


def _eval(chunk):
    return _E.evaluate(chunk)['V']


def main(fields_pkl, raster, out_dir, workers=12, geom=None, n_levels=None):
    t0 = time.time(); out_dir = Path(out_dir)
    if out_dir.exists(): raise FileExistsError(out_dir)
    from bluff_exact import BluffExact, GEOM
    GEOM = Path(geom) if geom else GEOM
    ex = BluffExact(fields_pkl, geom)
    src = np.load(GEOM/'source_surface.npz')
    sv, sf, cover, sem = src['vertices'], src['faces'].astype(np.int64), src['cover'].astype(float), src['semantic']
    c = ex.coarse_carry()
    cols = [('d', 3)] + [(k, c[k].shape[1] if c[k].ndim == 2 else 1) for k in CARRY] + [('cover', 1)]
    X = np.hstack([c['d']] + [c[k].reshape(len(sv), -1) for k in CARRY] + [cover[:, None]])
    face = np.load(raster)['face']; vis = np.unique(face[face >= 0])
    cent = sv[sf].mean(1); dist = np.full(len(sf), np.inf)
    for a in range(0, len(vis), 64):
        dist = np.minimum(dist, np.min(np.linalg.norm(cent[:, None, :] - cent[None, vis[a:a + 64], :], axis=2), axis=1))
    F, anc, green = sf.copy(), np.arange(len(sf)), np.zeros(len(sf), bool)
    levels = []
    dists = DISTANCES[:n_levels] if n_levels else DISTANCES          # --levels N: the first N bands (equal pixel-relative edge at a wider view)
    for k, d in enumerate(dists, 1):
        F, anc, green, X, rep = refine_level(F, anc, green, X, dist <= d)
        rep.update(level=k, distance_m=d, faces=len(F), vertices=len(X)); levels.append(rep)
    t_refine = time.time() - t0
    new = np.arange(len(sv), len(X))
    carry = {}; o = 0
    for name, width in cols:
        carry[name] = X[:, o:o + width] if width > 1 else X[:, o]; o += width
    V = np.zeros((len(X), 3)); V[:len(sv)] = sv
    t1 = time.time()
    chunks = [{k: v[new[a:a + 50000]] for k, v in carry.items() if k != 'cover'} for a in range(0, len(new), 50000)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(fields_pkl), str(GEOM))) as pool:
        parts = list(pool.map(_eval, chunks))
    if parts: V[new] = np.concatenate(parts)
    t_eval = time.time() - t1
    # checks: closed manifold, consistent winding, no degenerate face, no face flipped against its ancestor
    e = F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inv, cnt = np.unique(np.sort(e, 1), axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable').reshape(-1, 2) if (cnt == 2).all() else None
    tri = V[F]; cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(cross, axis=1) / 2
    st = sv[sf]; scross = np.cross(st[:, 1] - st[:, 0], st[:, 2] - st[:, 0])
    flips = int(((cross * scross[anc]).sum(1) <= 0).sum())
    vn = np.zeros_like(V)
    for i in range(3): np.add.at(vn, F[:, i], cross)
    vn /= np.linalg.norm(vn, axis=1)[:, None]
    cov = carry['cover']; cover_faces = cov[F].mean(1) > .5
    out_dir.mkdir(parents=True)
    np.savez_compressed(out_dir/'source_surface.npz', vertices=V, faces=F, normals=vn, cover=cov, cover_faces=cover_faces,
                        semantic=np.where(cover_faces, 'soil', 'bedrock'))
    np.savez_compressed(out_dir/'refinement.npz', ancestor_face=anc, green=green, visible_source_faces=vis, distance_to_visible_m=dist,
                        direction=carry['d'])
    shutil.copytree(GEOM/'package', out_dir/'package')
    edge = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    finest = np.isin(anc, vis)
    rep = dict(method=__doc__, fields=str(fields_pkl), fields_sha256=hashlib.sha256(Path(fields_pkl).read_bytes()).hexdigest(),
               source=str(GEOM), distances_m=dists, levels=levels, faces=len(F), vertices=len(V), new_vertices=int(len(new)),
               checks=dict(closed_manifold=bool((cnt == 2).all()), winding_consistent=bool(order is not None and (np.where(e[:, 0] < e[:, 1], 1, -1)[order].sum(1) == 0).all()),
                           degenerate_faces=int((area <= 0).sum()), min_face_area_m2=float(area.min()), flips_vs_ancestor=flips,
                           semantic_changed_faces_vs_ancestor=int((np.where(cover_faces, 'soil', 'bedrock') != sem[anc]).sum())),
               visible_edge_m=dict(median=float(np.median(edge[finest])), p90=float(np.percentile(edge[finest], 90))),
               seconds=dict(refine=t_refine, evaluate=t_eval, total=time.time() - t0), workers=workers)
    (out_dir/'refinement.json').write_text(json.dumps(rep, indent=1) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('method', 'levels')}, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]; wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 12
    main(a[0], a[1], a[2], wk, a[a.index('--geom') + 1] if '--geom' in a else None, int(a[a.index('--levels') + 1]) if '--levels' in a else None)
