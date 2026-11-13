"""- per zone: N_ZONE = 50000 points, faces drawn WITH replacement uniformly among the refined faces whose ancestor is a
visible face of the zone, uniform barycentric, rng seed = zone index. Uniform-by-face sampling over a whole frame is
dominated by the finest zones, so the frame-wide p99 alone could hide a failing coarse zone. PASS iff every zone p99
<= 0.5 px AND the frame-wide p99 <= 0.5 px; maxima reported only."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parent)) if str(_p.Path(__file__).resolve().parent) not in _s.path else None
import nogpu  # noqa: F401,E402  (CPU only)
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parents[1]/'frgb'/'chain'; sys.path[:0] = [str(M), str(HERE)]
import exact_route as er  # noqa: E402
from exact_route import CARRY  # noqa: E402

P99_PX = .5
N_FRAME, N_ZONE = 200000, 50000


def stats(e):
    return dict(n=int(len(e)), p50=float(np.median(e)), p99=float(np.percentile(e, 99)), max=float(e.max()))


def main(out, fields, geom, request, zones_npz, workers=1):
    out, geom = Path(out), Path(geom)
    s = np.load(out/'source_surface.npz'); r = np.load(out/'refinement.npz')
    V, F, anc, vis, d_all = s['vertices'], s['faces'].astype(np.int64), r['ancestor_face'], r['visible_source_faces'], r['direction']
    src = np.load(geom/'source_surface.npz'); sv, sf = src['vertices'], src['faces'].astype(np.int64)
    Z = np.load(zones_npz)
    if not np.array_equal(Z['visible_source_faces'], vis): raise ValueError('zones and refinement disagree on the visible faces')
    zone_of = np.full(len(sf), -1, np.int64); zone_of[vis] = Z['zone_of_visible']; nz = int(Z['zone_of_visible'].max()) + 1
    shot = json.loads(Path(request).read_text(encoding='utf8'))
    res = shot['resolution']; shot = shot['shots'][0]
    pix = shot['filmback_mm'][0] / shot['focal_length_mm'] / res[0]
    cam = np.asarray(shot['position_enu_m'], float)
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(np.isin(anc, vis)); fi0 = rng.choice(cand, min(N_FRAME, len(cand)), replace=False)
    b = rng.random((len(fi0), 2)); fl = b.sum(1) > 1; b[fl] = 1 - b[fl]; bcs = [np.c_[1 - b.sum(1), b]]; fis = [fi0]; tags = [np.full(len(fi0), -1)]
    fz = zone_of[anc]
    for z in range(nz):
        rz = np.random.default_rng(z); cz = np.flatnonzero(fz == z)
        fi = rz.choice(cz, N_ZONE, replace=True)
        b = rz.random((N_ZONE, 2)); fl = b.sum(1) > 1; b[fl] = 1 - b[fl]
        fis.append(fi); bcs.append(np.c_[1 - b.sum(1), b]); tags.append(np.full(N_ZONE, z))
    fi, bc, tag = np.concatenate(fis), np.concatenate(bcs), np.concatenate(tags)
    ex = er.make_exact(fields, geom); c0 = ex.coarse_carry()
    Aa = np.transpose(c0['d'][sf[anc[fi]]], (0, 2, 1))
    bcorner = np.stack([np.linalg.solve(Aa, d_all[F[fi, j]][:, :, None])[:, :, 0] for j in range(3)], 1)
    vb = (bcorner * bc[:, :, None]).sum(1)
    corners = sf[anc[fi]]
    cs = {k: (np.einsum('ij,ij...->i...', vb, c0[k][corners]) if c0[k].ndim == 2 else (vb * c0[k][corners]).sum(1)) for k in ['d'] + list(CARRY)}
    chunks = [{k: v[a:a + 25000] for k, v in cs.items()} for a in range(0, len(fi), 25000)]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=er.init, initargs=(str(fields), str(geom))) as pool:
            exact = np.concatenate(list(pool.map(er.eval_V, chunks)))
    else:
        er.init(str(fields), str(geom)); exact = np.concatenate([er.eval_V(c) for c in chunks])
    lin = (V[F[fi]] * bc[:, :, None]).sum(1)
    e_ref = np.linalg.norm(exact - lin, axis=1)
    e_px = e_ref / (np.linalg.norm(exact - cam, axis=1) * pix)
    frame = stats(e_px[tag == -1]); frame['err_mm'] = stats(e_ref[tag == -1] * 1e3)
    zones = []
    for z in range(nz):
        st = stats(e_px[tag == z]); st.update(zone=z, refined_faces=int((fz == z).sum()), pass_=bool(st['p99'] <= P99_PX)); zones.append(st)
    rep = dict(frame=dict(frame, pass_=bool(frame['p99'] <= P99_PX)), zones=zones, criterion=dict(p99_px=P99_PX, n_frame=N_FRAME, n_zone=N_ZONE),
               workers=workers, pixel_angle_rad=pix)
    rep['PASS'] = bool(rep['frame']['pass_'] and all(z['pass_'] for z in zones))
    (out/'zone_shape_check.json').write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps(rep))
    return rep


if __name__ == '__main__':
    a = sys.argv[1:]; g = lambda k, d: type(d)(a[a.index('--' + k) + 1]) if '--' + k in a else d
    main(a[0], a[1], a[2], a[3], a[4], g('workers', 1))
