"""Per-zone view-adaptive refinement.

Here every visible face v carries the level L_v of its zone (zone_dist.py) and level k is refined on the coarse faces
within DISTANCES[k-1] of a visible face with L_v >= k (dist_k = min over the zones with L_z >= k of zone_dist[z]). The
regions stay nested (dist_{k+1} >= dist_k, DISTANCES decreasing). Added: refinement.json['zones'] (levels per zone,
predicted / actual faces) and the seam check 'duplicate_vertex_positions'."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parent)) if str(_p.Path(__file__).resolve().parent) not in _s.path else None
import nogpu  # noqa: F401,E402  (CPU only)
import hashlib, json, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; M = HERE.parents[1]/'frgb'/'chain'
sys.path[:0] = [str(M), str(HERE)]
import bluff_refine as br  # noqa: E402  (DISTANCES, CARRY: unchanged)
import exact_route as er  # noqa: E402
from adaptive_refine import refine_level, level_regions, BUFFER_RINGS  # noqa: E402

MAX_LEVEL = len(br.DISTANCES)


def banded_regions(zone_dist, levels):
    """[region_1 .. region_K] (K = max level) and the distances used; region_k = dist_k <= DISTANCES[k-1]."""
    levels = [int(x) for x in levels]
    if not levels or min(levels) < 1 or max(levels) > MAX_LEVEL or len(levels) != len(zone_dist):
        raise ValueError('one level in 1..%d per zone' % MAX_LEVEL)
    K = max(levels); dists = br.DISTANCES[:K]
    banded = [zone_dist[[z for z, L in enumerate(levels) if L >= k]].min(0) <= dists[k - 1] for k in range(1, K + 1)]
    return banded, dists


def predicted_faces(banded):
    """sum over coarse faces of 4^(number of bands containing the face): red refinement only (green closures and buffer rings
    add a little)."""
    req = np.sum(np.stack(banded), axis=0) if banded else 0
    return int(np.sum(np.power(4, req, dtype=np.int64)))


def refine_regions(sf, X, banded, dists, buffer):
    def run(regions):
        F, anc, green, Y, levels = sf.copy(), np.arange(len(sf)), np.zeros(len(sf), bool), X, []
        for k, (d, region) in enumerate(zip(dists, regions), 1):
            F, anc, green, Y, rep = refine_level(F, anc, green, Y, region)
            rep.update(level=k, distance_m=d, faces=len(F), vertices=len(Y), region_faces=int(region.sum())); levels.append(rep)
        return F, anc, green, Y, levels
    rec = dict(mode=buffer, used=False, error=None, buffer_rings=BUFFER_RINGS)
    if buffer != 'always':
        try:
            return (*run(banded), rec)
        except ValueError as e:
            if buffer == 'off': raise
            rec['error'] = str(e)
    required = np.sum(np.stack(banded), axis=0)                     # bands are nested: the number of bands containing the face
    reg = level_regions(sf, required, len(dists))
    rec.update(used=True, extra_faces_per_level=[int((reg[k] & ~banded[k - 1]).sum()) for k in range(1, len(dists) + 1)])
    return (*run([reg[k] for k in range(1, len(dists) + 1)]), rec)


def main(fields_pkl, zones_npz, levels, out_dir, workers=1, geom=None, buffer='fallback'):
    t0 = time.time(); out_dir = Path(out_dir)
    if out_dir.exists(): raise FileExistsError(out_dir)
    GEOM = Path(geom)
    ex = er.make_exact(fields_pkl, geom)
    src = np.load(GEOM/'source_surface.npz')
    sv, sf, cover, sem = src['vertices'], src['faces'].astype(np.int64), src['cover'].astype(float), src['semantic']
    c = ex.coarse_carry()
    cols = [('d', 3)] + [(k, c[k].shape[1] if c[k].ndim == 2 else 1) for k in br.CARRY] + [('cover', 1)]
    X = np.hstack([c['d']] + [c[k].reshape(len(sv), -1) for k in br.CARRY] + [cover[:, None]])
    Z = np.load(zones_npz)
    vis, zone_dist, dist = Z['visible_source_faces'], Z['zone_dist'], Z['dist_all']
    if zone_dist.shape[1] != len(sf): raise ValueError('zones do not belong to this source')
    banded, dists = banded_regions(zone_dist, levels)
    pred = predicted_faces(banded)
    F, anc, green, X, levels_rep, brec = refine_regions(sf, X, banded, dists, buffer)
    t_refine = time.time() - t0
    new = np.arange(len(sv), len(X))
    carry = {}; o = 0
    for name, width in cols:
        carry[name] = X[:, o:o + width] if width > 1 else X[:, o]; o += width
    V = np.zeros((len(X), 3)); V[:len(sv)] = sv
    t1 = time.time()
    chunks = [{k: v[new[a:a + 50000]] for k, v in carry.items() if k != 'cover'} for a in range(0, len(new), 50000)]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=er.init, initargs=(str(fields_pkl), str(GEOM))) as pool:
            parts = list(pool.map(er.eval_V, chunks))
    else:
        er.init(str(fields_pkl), str(GEOM)); parts = [er.eval_V(ch) for ch in chunks]
    if parts: V[new] = np.concatenate(parts)
    t_eval = time.time() - t1
    e = F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inv, cnt = np.unique(np.sort(e, 1), axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable').reshape(-1, 2) if (cnt == 2).all() else None
    tri = V[F]; cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(cross, axis=1) / 2
    st = sv[sf]; scross = np.cross(st[:, 1] - st[:, 0], st[:, 2] - st[:, 0])
    flips = int(((cross * scross[anc]).sum(1) <= 0).sum())
    dup = int(len(V) - len(np.unique(V, axis=0)))
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
    rep = dict(method=__doc__, base='frgb/chain/bluff_refine.py', fields=str(fields_pkl),
               fields_sha256=hashlib.sha256(Path(fields_pkl).read_bytes()).hexdigest(),
               source=str(GEOM), distances_m=dists, levels=levels_rep, faces=len(F), vertices=len(V), new_vertices=int(len(new)),
               checks=dict(closed_manifold=bool((cnt == 2).all()), winding_consistent=bool(order is not None and (np.where(e[:, 0] < e[:, 1], 1, -1)[order].sum(1) == 0).all()),
                           degenerate_faces=int((area <= 0).sum()), min_face_area_m2=float(area.min()), flips_vs_ancestor=flips,
                           semantic_changed_faces_vs_ancestor=int((np.where(cover_faces, 'soil', 'bedrock') != sem[anc]).sum()),
                           duplicate_vertex_positions=dup),
               visible_edge_m=dict(median=float(np.median(edge[finest])), p90=float(np.percentile(edge[finest], 90))),
               seconds=dict(refine=t_refine, evaluate=t_eval, total=time.time() - t0), workers=workers, buffer=brec, route=getattr(ex, 'route', 'ray'),
               zones=dict(file=str(zones_npz), levels=[int(x) for x in levels], predicted_faces_red_only=pred,
                          info=json.loads(str(Z['info']))))
    rep['seam_pass'] = bool(rep['checks']['closed_manifold'] and rep['checks']['winding_consistent'] and rep['checks']['degenerate_faces'] == 0
                            and dup == 0)
    (out_dir/'refinement.json').write_text(json.dumps(rep, indent=1) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('method', 'levels')}, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index('--' + k) + 1] if '--' + k in a else d
    main(a[0], a[1], [int(x) for x in a[2].split(',')], a[3], int(g('workers', 1)), g('geom'), g('buffer', 'fallback'))
