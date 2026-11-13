"""Zones of the per-zone refinement-level selection.

Visible coarse faces v get d_v = |centroid - camera| (the pixel footprint is d_v x the pixel angle, so distance zones
are footprint zones); zone z = floor(2 log2(d_v / 1 m)) (half octaves); zones with fewer than MIN_FACES visible faces
merge into the adjacent nearer zone (the farther one when there is none), repeated until every zone has MIN_FACES or
one zone is left. Zones are renumbered 0..Z-1 from near to far. For every zone the distance of every coarse centroid
to the nearest visible centroid of that zone -> zone_dist (Z, n_faces); dist_all = the same to every visible face.
Usage: python zone_dist.py GEOM_DIR RASTER_NPZ REQUEST_JSON OUT_NPZ"""
import json, sys
from pathlib import Path
import numpy as np

MIN_FACES = 20


def zones_of(d, min_faces=MIN_FACES):
    z = np.floor(2 * np.log2(d)).astype(np.int64)
    keys = sorted(set(z.tolist()))
    while len(keys) > 1:
        counts = {k: int((z == k).sum()) for k in keys}
        small = [k for k in keys if counts[k] < min_faces]
        if not small: break
        k = small[0]; i = keys.index(k)
        tgt = keys[i - 1] if i > 0 else keys[i + 1]
        z[z == k] = tgt; keys.remove(k)
    lut = {k: i for i, k in enumerate(keys)}
    return np.array([lut[x] for x in z.tolist()], np.int64), keys


def main(geom, raster, request, out):
    from scipy.spatial import cKDTree
    src = np.load(Path(geom)/'source_surface.npz'); sv, sf = src['vertices'], src['faces'].astype(np.int64)
    face = np.load(raster)['face']; vis = np.unique(face[face >= 0])
    shot = json.loads(Path(request).read_text(encoding='utf8'))['shots'][0]
    cam = np.asarray(shot['position_enu_m'], float)
    cent = sv[sf].mean(1)
    d = np.linalg.norm(cent[vis] - cam, axis=1)
    zone, keys = zones_of(d)
    zd = np.stack([cKDTree(cent[vis[zone == z]]).query(cent, k=1)[0] for z in range(len(keys))])
    dist_all = cKDTree(cent[vis]).query(cent, k=1)[0]
    if not np.array_equal(zd.min(0), dist_all): raise RuntimeError('per-zone distances do not reproduce the all-visible distance')
    info = [dict(zone=z, half_octave_key=int(keys[z]), d_range_m=[float(d[zone == z].min()), float(d[zone == z].max())],
                 d_median_m=float(np.median(d[zone == z])), visible_faces=int((zone == z).sum())) for z in range(len(keys))]
    np.savez_compressed(out, visible_source_faces=vis, d_visible_m=d, zone_of_visible=zone, zone_dist=zd, dist_all=dist_all,
                        info=json.dumps(info), camera_m=cam)
    print(json.dumps(dict(visible=int(len(vis)), faces=int(len(sf)), zones=info)))


if __name__ == '__main__':
    main(*sys.argv[1:5])
