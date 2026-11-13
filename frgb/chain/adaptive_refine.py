"""View-adaptive Phong refinement of the source rock surface.

Graded red-green refinement driven by the camera: original faces whose centroid lies within d_k of a source face
visible in the camera frame (CPU pixel-centre raster) are red-split (1 -> 4) at level k; the nested regions shrink
with k (DISTANCES, levels 1..6), so the visible faces reach 2^6 = 64 times finer edges and shadow casters around them
are refined progressively. Faces with one split edge get a green closure (1 -> 2); a face with two or three split
edges is promoted to red; green faces are never split again (asserted), so the mesh stays conforming (no T-junctions).
All new vertices are first placed on the original flat triangles (linear subdivision), then every vertex is moved onto
the Phong tessellation surface of its original triangle (Boubekeur & Alexa 2008, shape factor ALPHA) built from the
source's unique per-vertex normals. A Phong edge curve depends only on that edge's two end points and normals, so the
two original triangles sharing an edge place an edge vertex identically: the refined surface is watertight. Original
vertices and unrefined faces do not move. Writes OUT_DIR/source_surface.npz (vertices, faces, normals = area-weighted
vertex normals of the refined mesh, cover, cover_faces copied, semantic inherited), OUT_DIR/refinement.npz (ancestor
face, level, green flag, flat positions) and OUT_DIR/refinement.json; copies the source package/ (frozen network)
unchanged."""
BUFFER_RINGS = 2
import hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np

DISTANCES = (1.5, 1.0, .65, .4, .25, .1)     # metres from a visible source face (centroid to centroid), levels 1..6
ALPHA = .75                              # Phong shape factor (Boubekeur & Alexa 2008 recommendation)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def refine_level(F, anc, green, V, region):
    """One red-green level on the current mesh; region: bool over original faces."""
    n = len(F)
    fe = np.sort(F[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    edges, eid = np.unique(fe, axis=0, return_inverse=True); eid = eid.reshape(n, 3)
    red = ~green & region[anc]
    split = np.zeros(len(edges), bool)
    while True:
        split[eid[red].ravel()] = True
        cnt = split[eid].sum(1)
        promote = ~red & (cnt >= 2)
        if not promote.any(): break
        if (promote & green).any(): raise ValueError('a green closure would be refined again; enlarge the region spacing')
        red |= promote
    cnt = split[eid].sum(1)
    grn = ~red & (cnt == 1)
    if (grn & green).any(): raise ValueError('a green closure would be split again; enlarge the region spacing')
    mid = np.full(len(edges), -1, np.int64); s = np.flatnonzero(split)
    mid[s] = len(V) + np.arange(len(s))
    V = np.vstack([V, V[edges[s]].mean(1)])
    keep = ~red & ~grn
    out_F, out_anc, out_green = [F[keep]], [anc[keep]], [green[keep]]
    a, b, c = F[red].T; mab, mbc, mca = mid[eid[red, 0]], mid[eid[red, 1]], mid[eid[red, 2]]
    for tri in ((a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)):
        out_F.append(np.stack(tri, 1)); out_anc.append(anc[red]); out_green.append(np.zeros(red.sum(), bool))
    gi = np.flatnonzero(grn); k = np.argmax(split[eid[gi]], axis=1)
    v0, v1, v2 = F[gi, k], F[gi, (k + 1) % 3], F[gi, (k + 2) % 3]; m = mid[eid[gi, k]]
    for tri in ((v0, m, v2), (m, v1, v2)):
        out_F.append(np.stack(tri, 1)); out_anc.append(anc[gi]); out_green.append(np.ones(len(gi), bool))
    return np.concatenate(out_F), np.concatenate(out_anc), np.concatenate(out_green), V, dict(red=int(red.sum()), green=int(grn.sum()), new_vertices=int(len(s)))


def phong(V_flat, F, anc, sv, sf, sn):
    """Move every vertex onto the Phong surface of the original triangle of (one of) its faces."""
    owner = np.full(len(V_flat), -1, np.int64)
    owner[F.ravel()] = np.repeat(anc, 3)
    tri = sv[sf[owner]]; nrm = sn[sf[owner]]
    # barycentrics of the flat point in its original triangle (it lies in the triangle's plane)
    e1, e2, q = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], V_flat - tri[:, 0]
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1); d20, d21 = (q * e1).sum(1), (q * e2).sum(1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den; u = 1 - v - w
    bary = np.stack([u, v, w], 1)
    proj = V_flat[:, None, :] - ((V_flat[:, None, :] - tri) * nrm).sum(2)[:, :, None] * nrm      # projections on the 3 tangent planes
    ph = (bary[:, :, None] * proj).sum(1)
    out = (1 - ALPHA) * V_flat + ALPHA * ph
    return out, bary


def dilate_faces(F, mask, rings):
    for _ in range(rings):
        mv = np.zeros(F.max() + 1, bool); mv[F[mask].ravel()] = True; mask = mv[F].any(1)
    return mask


def level_regions(F, required, maxlevel):
    regions = {maxlevel: required >= maxlevel}
    for k in range(maxlevel - 1, 0, -1):
        regions[k] = (required >= k) | dilate_faces(F, regions[k + 1], BUFFER_RINGS)
    return regions


def main(source_dir, raster_path, out_dir, levels_path=None, penumbra_fraction=1.):
    t0 = time.time()
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if out_dir.exists(): raise FileExistsError(out_dir)
    s = np.load(source_dir/'source_surface.npz')
    sv, sf, sn, sem = s['vertices'].astype(np.float64), s['faces'].astype(np.int64), s['normals'].astype(np.float64), s['semantic']
    sn = sn / np.linalg.norm(sn, axis=1)[:, None]
    from scipy.spatial import cKDTree
    face = np.load(raster_path)['face']; vis = np.unique(face[face >= 0])
    cent = sv[sf].mean(1); dist, _ = cKDTree(cent[vis]).query(cent)
    F, anc, green, V = sf.copy(), np.arange(len(sf)), np.zeros(len(sf), bool), sv.copy()
    levels = []
    if levels_path is None:
        schedule = [(k, dist <= d, dict(distance_m=d)) for k, d in enumerate(DISTANCES, 1)]
    else:
        c = np.load(levels_path); tmin, visible = c['caster_min_distance_m'], c['visible']
        tri0 = sv[sf]; emax = np.linalg.norm(tri0 - np.roll(tri0, 1, axis=1), axis=2).max(1)
        target = np.where(visible, .0036, np.where(np.isfinite(tmin), np.maximum(.0036, penumbra_fraction * tmin * np.tan(np.radians(.5357))), np.inf))
        required = np.where(np.isfinite(target), np.clip(np.ceil(np.log2(np.maximum(emax / target, 1.))), 0, 6), 0).astype(int)
        regions = level_regions(sf, required, int(required.max()))
        schedule = [(k, regions[k], dict(required_faces=int((required >= k).sum()), region_faces=int(regions[k].sum()))) for k in sorted(regions)]
    for k, region, info in schedule:
        F, anc, green, V, rep = refine_level(F, anc, green, V, region)
        rep.update(level=k, faces=len(F), vertices=len(V), **info); levels.append(rep); print('level', json.dumps(rep), flush=True)
    V_flat = V.copy()
    V, bary = phong(V_flat, F, anc, sv, sf, sn)
    orig = np.arange(len(sv))
    if not np.array_equal(V[orig], sv) and np.abs(V[orig] - sv).max() > 1e-12: raise ValueError('original vertices moved')
    if bary.min() < -1e-9 or bary.max() > 1 + 1e-9: raise ValueError('a refined vertex left its original triangle')
    # checks: closed index-manifold with consistent winding, no degenerate face
    e = F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2); und = np.sort(e, 1)
    _, inv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable').reshape(-1, 2) if (cnt == 2).all() else None
    tri = V[F]; cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(cross, axis=1) / 2
    src_tri = sv[sf]; src_area = np.linalg.norm(np.cross(src_tri[:, 1] - src_tri[:, 0], src_tri[:, 2] - src_tri[:, 0]), axis=1) / 2
    checks = dict(edge_use_counts=np.unique(cnt).tolist(), closed_manifold=bool((cnt == 2).all()),
                  winding_consistent=bool(order is not None and (np.where(e[:, 0] < e[:, 1], 1, -1)[order].sum(1) == 0).all()),
                  min_face_area_m2=float(area.min()), degenerate_faces=int((area <= 0).sum()),
                  face_normal_flips_vs_ancestor=int(((cross * np.cross(src_tri[anc, 1] - src_tri[anc, 0], src_tri[anc, 2] - src_tri[anc, 0])).sum(1) <= 0).sum()),
                  displacement_m=dict(max=float(np.linalg.norm(V - V_flat, axis=1).max()), p99=float(np.percentile(np.linalg.norm(V - V_flat, axis=1), 99))),
                  area_ratio=float(area.sum() / src_area.sum()))
    vn = np.zeros_like(V)
    for i in range(3): np.add.at(vn, F[:, i], cross)
    vn /= np.linalg.norm(vn, axis=1)[:, None]
    out_dir.mkdir(parents=True)
    np.savez_compressed(out_dir/'source_surface.npz', vertices=V, faces=F, normals=vn, cover=s['cover'], cover_faces=s['cover_faces'], semantic=sem[anc])
    np.savez_compressed(out_dir/'refinement.npz', ancestor_face=anc, green=green, flat_vertices=V_flat, visible_source_faces=vis, distance_to_visible_m=dist)
    shutil.copytree(source_dir/'package', out_dir/'package')
    edges_len = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    finest = np.isin(anc, vis)
    report = dict(method=__doc__, source=str(source_dir), source_sha256=sha(source_dir/'source_surface.npz'), raster=str(raster_path),
                  distances_m=DISTANCES if levels_path is None else None, levels_file=str(levels_path) if levels_path else None,
                  penumbra_fraction=penumbra_fraction if levels_path else None, buffer_rings=BUFFER_RINGS if levels_path else None,
                  alpha=ALPHA, levels=levels, faces=len(F), vertices=len(V), checks=checks,
                  visible_source_faces=int(len(vis)), edge_length_on_visible_faces_m=dict(median=float(np.median(edges_len[finest])),
                  p90=float(np.percentile(edges_len[finest], 90))), seconds=time.time() - t0,
                  note='semantic inherited from the ancestor face; cover/cover_faces copied unchanged; package/ copied unchanged')
    (out_dir/'refinement.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'method'}, indent=1))


if __name__ == '__main__':
    argv = sys.argv[1:]
    lv = argv[argv.index('--levels') + 1] if '--levels' in argv else None
    pf = float(argv[argv.index('--penumbra-fraction') + 1]) if '--penumbra-fraction' in argv else 1.
    main(*argv[:3], levels_path=lv, penumbra_fraction=pf)
