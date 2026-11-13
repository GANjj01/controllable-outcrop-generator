"""Skeleton of the fracture-mouth mask against the visible centreline of the labels (main text Section 4.2, last paragraph;
supplement S4, "Role of each step and the mask skeleton" with its table).

One frame for each of 11 viewpoints (frames_io.VIEWPOINTS); the mask and the visible centreline depend only on geometry and
viewpoint.

Mouth mask: the pixels that the render classes as rock (region mask 1) and at which the ray through the pixel centre first hits a
cavity surface. The rendered material index does not separate cavity faces from the rock surface (both carry index 1), so the
first hit is computed here pixel by pixel from the cut mesh (geometry/native_surface.npz of the case); the cavity faces are the
faces whose original_id in difference.npz differs from host_original_id of build.json. Two checks per viewpoint: the first-hit
depth agrees with the rendered depth (within max(0.01 m, 0.002 x depth)) on at least 99 % of the rock pixels, and at least
99 % of the fracture-mouth pixels of the label program (opening_class > 0) lie inside the mask.
Skeleton: Zhang-Suen thinning of the mask (the algorithm the labels use). Visible centreline: target or u_line.
Reported per viewpoint: skeleton and centreline pixels; fraction of skeleton pixels more than 2 px from the centreline; fraction
of centreline pixels with no skeleton pixel within 2 px; median distances; number of 8-connected components of the skeleton over
that of the centreline; occluded length / in-frame length of the joints (report.json), which no skeleton can give; and the
angle theta between the line of sight and the rock-surface normal (camera plan).

    python reproduce/paper1/analysis/mask_skeleton.py --frames FRAMES
    python reproduce/paper1/analysis/mask_skeleton.py --selftest      checks the functions on synthetic shapes; reads and writes nothing

Reads records/frames_index.json and, per frame, FRAMES/<frame id>/plan.json, q0/render_spec.json, q0/geometry.json,
q0/render/frame.npz (depth, region mask), q0/trace_labels/{report.json, target.npy, u_line.npy, opening_class.npy}; from the cut
case named in geometry.json: build.json, difference.npz (original_id, faces), geometry/native_surface.npz (vertices, faces) and
geometry/geometry_checks.json. Writes records/round3/mask_skeleton.json. Needs NumPy and SciPy (the labels environment)."""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402

FAR_PX = 2.                                     # "more than 2 pixels"
VERTICAL_UP_SWITCH = 0.99                       # as frgb/raster.py
ALIGN_ABS_M, ALIGN_REL = .01, .002              # tolerance of labels/label_frame.py::aligned_mask
MIN_ALIGNED, MIN_MOUTH_INSIDE = .99, .99        # the two checks per viewpoint
EIGHT = np.ones((3, 3), bool)


# ------------------------------------------------------------------ Zhang-Suen thinning

def _neighbours(img):
    """img is padded by one ring of background; returns the centre view and the views P2..P9 (clockwise from above)"""
    return (img[1:-1, 1:-1],
            [img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]])


def zhang_suen(mask):
    """Zhang-Suen thinning (Zhang and Suen, 1984): the two sub-iterations with parallel deletion, 8-neighbourhood clockwise from P2
    (above), background outside the image, repeated until neither sub-iteration deletes a pixel. Returns (skeleton, rounds).
    Works inside the bounding box of the mask (grown by 1 pixel)."""
    mask = np.asarray(mask, bool)
    out = np.zeros(mask.shape, bool)
    if not mask.any():
        return out, 0
    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    img = np.pad(mask[y0:y1, x0:x1], 1)
    rounds = 0
    while True:
        changed = False
        for step in (0, 1):
            C, (P2, P3, P4, P5, P6, P7, P8, P9) = _neighbours(img)
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = np.zeros(C.shape, np.uint8)
            A = np.zeros(C.shape, np.uint8)
            for i in range(8):
                B += nb[i]
                A += ~nb[i] & nb[(i + 1) % 8]
            if step == 0:
                cond = ~(P2 & P4 & P6) & ~(P4 & P6 & P8)
            else:
                cond = ~(P2 & P4 & P8) & ~(P2 & P6 & P8)
            delete = C & (B >= 2) & (B <= 6) & (A == 1) & cond       # a new array: all deletions of a sub-iteration at once
            if delete.any():
                img[1:-1, 1:-1] &= ~delete
                changed = True
        rounds += 1
        if not changed:
            break
    out[y0:y1, x0:x1] = img[1:-1, 1:-1]
    return out, rounds


def zhang_suen_reference(mask):
    """pixel-by-pixel reference implementation (self-test only)"""
    img = np.pad(np.asarray(mask, bool), 1).copy()
    h, w = img.shape
    while True:
        changed = False
        for step in (0, 1):
            kill = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if not img[y, x]:
                        continue
                    p = [img[y - 1, x], img[y - 1, x + 1], img[y, x + 1], img[y + 1, x + 1], img[y + 1, x], img[y + 1, x - 1],
                         img[y, x - 1], img[y - 1, x - 1]]
                    b = sum(p)
                    a = sum((not p[i]) and p[(i + 1) % 8] for i in range(8))
                    if step == 0:
                        c = (not (p[0] and p[2] and p[4])) and (not (p[2] and p[4] and p[6]))
                    else:
                        c = (not (p[0] and p[2] and p[6])) and (not (p[0] and p[4] and p[6]))
                    if 2 <= b <= 6 and a == 1 and c:
                        kill.append((y, x))
            for y, x in kill:
                img[y, x] = False
            changed |= bool(kill)
        if not changed:
            break
    return img[1:-1, 1:-1]


# ------------------------------------------------------------------ comparison

def components(mask):
    return int(ndi.label(mask, structure=EIGHT)[1])


def evaluate(skeleton, centreline):
    """skeleton against visible centreline (two boolean images)"""
    skeleton, centreline = np.asarray(skeleton, bool), np.asarray(centreline, bool)
    n_s, n_c = int(skeleton.sum()), int(centreline.sum())
    out = dict(skeleton_px=n_s, centreline_px=n_c, skeleton_components=components(skeleton), centreline_components=components(centreline))
    if n_s and n_c:
        d_sc = ndi.distance_transform_edt(~centreline)[skeleton]
        d_cs = ndi.distance_transform_edt(~skeleton)[centreline]
        out.update(skeleton_to_centreline_median_px=float(np.median(d_sc)),
                   skeleton_to_centreline_frac_gt_2px=float((d_sc > FAR_PX).mean()),
                   centreline_to_skeleton_frac_gt_2px=float((d_cs > FAR_PX).mean()),
                   centreline_to_skeleton_median_px=float(np.median(d_cs)))
    else:
        out.update(skeleton_to_centreline_median_px=None, skeleton_to_centreline_frac_gt_2px=None if not n_s else 1.,
                   centreline_to_skeleton_frac_gt_2px=None if not n_c else 1., centreline_to_skeleton_median_px=None)
    out['component_ratio'] = out['skeleton_components'] / out['centreline_components'] if out['centreline_components'] else None
    return out


# ------------------------------------------------------------------ camera and rasterisation (conventions of frgb/raster.py)

def camera(shot, resolution):
    pos = np.asarray(shot['position_enu_m'], float)
    fwd = np.asarray(shot['target_enu_m'], float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0., 0., 1.] if abs(fwd[2]) <= VERTICAL_UP_SWITCH else [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    w, h = resolution
    fw, fh = shot['filmback_mm']
    if abs(fw / fh - w / h) > 1e-6:
        raise ValueError('filmback aspect differs from resolution aspect')
    near = float(shot.get('near_clip_cm', 10.)) / 100.
    return dict(pos=pos, fwd=fwd, right=right, up=up, fpx=shot['focal_length_mm'] / fw * w, w=int(w), h=int(h), near=near)


def rasterize_first_hit(V, F, cam, chunk_faces=1_000_000, max_pairs=4_000_000):
    """First hit of the rays through the pixel centres: per pixel the nearest face and its view-axis depth (-1 / inf without a hit).
    Faces that cross the near clip plane are not rasterised, only counted. Equal depths: the face with the smaller number.
    Perspective-correct depth interpolation; barycentric tolerance 1e-9; the faces are processed in chunks to bound the memory."""
    w, h = cam['w'], cam['h']
    rel = np.asarray(V, float) - cam['pos']
    zc = rel @ cam['fwd']
    with np.errstate(divide='ignore', invalid='ignore'):
        X = w / 2 + cam['fpx'] * (rel @ cam['right']) / zc - .5
        Y = h / 2 - cam['fpx'] * (rel @ cam['up']) / zc - .5
    del rel
    depth = np.full(h * w, np.inf)
    face = np.full(h * w, -1, np.int64)
    stats = dict(faces=int(len(F)), near_crossing_faces=0, behind_faces=0, faces_covering_a_pixel_centre=0, pairs=0)
    for s in range(0, len(F), chunk_faces):
        f = np.asarray(F[s:s + chunk_faces], np.int64)
        z3 = zc[f]
        front = (z3 >= cam['near']).all(1)
        some = (z3 >= cam['near']).any(1)
        stats['near_crossing_faces'] += int((some & ~front).sum())
        stats['behind_faces'] += int((~some).sum())
        idx = np.flatnonzero(front)
        if not len(idx):
            continue
        f, z3 = f[idx], z3[idx]
        x3, y3 = X[f], Y[f]
        bx0 = np.clip(np.ceil(x3.min(1)), 0, w).astype(np.int64)
        bx1 = np.clip(np.floor(x3.max(1)), -1, w - 1).astype(np.int64)
        by0 = np.clip(np.ceil(y3.min(1)), 0, h).astype(np.int64)
        by1 = np.clip(np.floor(y3.max(1)), -1, h - 1).astype(np.int64)
        keep = np.flatnonzero((bx1 >= bx0) & (by1 >= by0))
        if not len(keep):
            continue
        stats['faces_covering_a_pixel_centre'] += int(len(keep))
        bw, bh = (bx1 - bx0 + 1)[keep], (by1 - by0 + 1)[keep]
        size = bw * bh
        csum = np.cumsum(size)
        a = 0
        while a < len(keep):
            base = csum[a - 1] if a else 0
            b = max(int(np.searchsorted(csum, base + max_pairs, 'right')), a + 1)
            sel = np.arange(a, b)
            n = size[sel]
            t = np.repeat(sel, n)
            k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
            g = keep[t]
            xx = bx0[g] + k % bw[t]
            yy = by0[g] + k // bw[t]
            ax, ay, bx, by, cx, cy = x3[g, 0], y3[g, 0], x3[g, 1], y3[g, 1], x3[g, 2], y3[g, 2]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            with np.errstate(divide='ignore', invalid='ignore'):
                w0 = ((bx - xx) * (cy - yy) - (by - yy) * (cx - xx)) / area
                w1 = ((cx - xx) * (ay - yy) - (cy - yy) * (ax - xx)) / area
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (area != 0)
            g, xx, yy = g[inside], xx[inside], yy[inside]
            iz = w0[inside] / z3[g, 0] + w1[inside] / z3[g, 1] + w2[inside] / z3[g, 2]
            zz = 1. / iz
            pix = yy * w + xx
            fid = s + idx[g]
            stats['pairs'] += int(len(pix))
            order = np.lexsort((fid, zz, pix))
            ps = pix[order]
            first = np.r_[True, ps[1:] != ps[:-1]] if len(ps) else np.zeros(0, bool)
            p1, z1, f1 = ps[first], zz[order][first], fid[order][first]
            upd = z1 < depth[p1]
            depth[p1[upd]] = z1[upd]
            face[p1[upd]] = f1[upd]
            a = b
    return face.reshape(h, w), depth.reshape(h, w), stats


def cavity_faces(case):
    """per-face cavity flag: original_id of difference.npz differs from host_original_id of build.json"""
    build = fio.read_json(Path(case) / 'build.json')
    with np.load(Path(case) / 'difference.npz', allow_pickle=False) as a:
        origin = np.asarray(a['original_id'])
        faces_diff = np.asarray(a['faces'])
    return origin != build['host_original_id'], faces_diff, build


def mouth_mask(frames, key):
    """the mask (rock in the render, first hit on a cavity face) and its two checks against the render and the label program"""
    case = fio.case_dir(frames, key)
    cav, faces_diff, build = cavity_faces(case)
    with np.load(case / 'geometry' / 'native_surface.npz', allow_pickle=False) as a:
        V, F = a['vertices'], a['faces']
    same_faces = bool(F.shape == faces_diff.shape and np.array_equal(F, faces_diff))
    del faces_diff
    if not same_faces or len(cav) != len(F):
        raise ValueError('%s: the faces of difference.npz differ from those of native_surface.npz' % key)
    spec = fio.render_spec(frames, key)
    cam = camera(spec['shot'], spec['resolution'])
    face, depth, stats = rasterize_first_hit(V, F, cam)
    del V, F
    with np.load(fio.point_dir(frames, key) / 'render' / 'frame.npz', allow_pickle=False) as a:
        D = np.asarray(a['depth'], float)
        rmask = np.asarray(a['mask'])
    rock = rmask == 1
    hit = face >= 0
    cav_px = np.zeros(face.shape, bool)
    cav_px[hit] = cav[face[hit]]
    mask = rock & cav_px
    both = rock & hit
    resid = np.abs(D[both] - depth[both])
    aligned = resid <= np.maximum(ALIGN_ABS_M, ALIGN_REL * depth[both])
    gc = fio.read_json(case / 'geometry' / 'geometry_checks.json')
    rec = dict(
        faces=int(len(cav)), cavity_faces=int(cav.sum()), host_faces=int((~cav).sum()),
        cavity_face_count_matches_geometry_checks=bool(int(cav.sum()) == gc['cavity_faces'] and int((~cav).sum()) == gc['original_host_faces']),
        rock_px=int(rock.sum()), raster_hit_px=int(hit.sum()),
        rock_px_with_raster_hit_fraction=float(both.sum() / max(int(rock.sum()), 1)),
        depth_abs_diff_m_p50_p99_max=[float(x) for x in np.quantile(resid, [.5, .99, 1.])] if len(resid) else None,
        depth_aligned_fraction_of_rock=float(aligned.sum() / max(int(rock.sum()), 1)),
        mask_px=int(mask.sum()))
    oc = np.load(fio.label_dir(frames, key) / 'opening_class.npy') > 0
    rec['opening_class_px'] = int(oc.sum())
    rec['opening_class_inside_mask_fraction'] = float((oc & mask).sum() / max(int(oc.sum()), 1))
    return mask, rec


def occluded_share(frames, key):
    pp = fio.read_json(fio.label_dir(frames, key) / 'report.json')['per_path']
    a = float(sum(p['projected_in_frame_px'] for p in pp))
    b = float(sum(p['occluded_px'] for p in pp))
    return b / a if a > 0 else None


# ------------------------------------------------------------------ self-test on synthetic shapes

def _shapes():
    yy, xx = np.mgrid[0:40, 0:60]
    s = []
    a = np.zeros((40, 60), bool); a[10:15, 5:45] = True; s.append(('bar', a))
    a = np.zeros((40, 60), bool); a[5:35, 20:27] = True; a[17:24, 5:55] = True; s.append(('cross', a))
    s.append(('disc', (yy - 20) ** 2 + (xx - 30) ** 2 <= 144))
    s.append(('oblique band', np.abs(yy - (.5 * xx + 3)) <= 3))
    s.append(('sine band', np.abs(yy - (20 + 8 * np.sin(xx / 6.))) <= 2.5))
    a = np.zeros((40, 60), bool); a[8:30, 10:14] = True; a[26:30, 10:50] = True; s.append(('L', a))
    s.append(('ring', ((yy - 20) ** 2 + (xx - 30) ** 2 <= 196) & ((yy - 20) ** 2 + (xx - 30) ** 2 >= 81)))
    a = np.zeros((40, 60), bool); a[3:5, 3:5] = True; a[20, 10:50] = True; a[30:33, 30:33] = True; s.append(('blocks and line', a))
    return s


def selftest():
    # 1 thinning: equal to the pixel-by-pixel reference; a subset of the mask; idempotent; solid shapes keep one component
    for name, m in _shapes():
        sk, rounds = zhang_suen(m)
        ref = zhang_suen_reference(m)
        assert np.array_equal(sk, ref), name
        assert not (sk & ~m).any(), name
        assert np.array_equal(zhang_suen(sk)[0], sk), name
        if name in ('bar', 'cross', 'oblique band', 'sine band', 'L', 'ring'):
            assert components(sk) == 1 and sk.any(), (name, components(sk))
    sk, _ = zhang_suen(_shapes()[0][1])
    cols = np.flatnonzero(sk.any(0))
    assert (sk.sum(0)[cols] == 1).all() and len(cols) >= 30, 'the skeleton of the bar is a line 1 pixel wide'
    assert not zhang_suen(np.zeros((8, 8), bool))[0].any()
    edge = np.zeros((12, 12), bool); edge[0:4, :] = True               # a mask on the image border
    assert np.array_equal(zhang_suen(edge)[0], zhang_suen_reference(edge))
    # 2 comparison
    s = np.zeros((40, 60), bool); s[10, 0:50] = True
    c = np.zeros((40, 60), bool); c[13, 0:30] = True
    e = evaluate(s, c)
    assert e['skeleton_to_centreline_frac_gt_2px'] == 1. and e['centreline_to_skeleton_frac_gt_2px'] == 1., e
    assert abs(e['skeleton_to_centreline_median_px'] - 3.) < 1e-12 and e['component_ratio'] == 1., e
    c2 = np.zeros((40, 60), bool); c2[11, 0:50] = True; c2[30, 0:5] = True
    e = evaluate(s, c2)
    assert e['skeleton_to_centreline_frac_gt_2px'] == 0. and abs(e['centreline_to_skeleton_frac_gt_2px'] - 5 / 55) < 1e-12, e
    assert e['skeleton_to_centreline_median_px'] == 1. and e['component_ratio'] == .5, e
    s3 = s.copy(); s3[10, 20:23] = False
    assert evaluate(s3, c2)['component_ratio'] == 1.
    e = evaluate(np.zeros((5, 5), bool), c2[:5, :5])
    assert e['skeleton_to_centreline_median_px'] is None
    # 3 rasterisation: an oblique camera on a square in the plane z = 0, depth against the analytic ray-plane intersection
    cam = camera(dict(position_enu_m=[0., -3., 4.], target_enu_m=[0., 0., 0.], focal_length_mm=35., filmback_mm=[36., 36.]), [64, 64])
    V = np.array([[-4., -4., 0.], [4., -4., 0.], [4., 4., 0.], [-4., 4., 0.], [-.5, -.5, 1.], [.5, -.5, 1.], [0., .5, 1.]])
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    face, depth, st = rasterize_first_hit(V, F, cam, chunk_faces=2, max_pairs=500)
    yy, xx = np.mgrid[0:64, 0:64]
    d = cam['fwd'] + ((xx + .5 - 32) / cam['fpx'])[..., None] * cam['right'] + (-(yy + .5 - 32) / cam['fpx'])[..., None] * cam['up']
    t0 = -cam['pos'][2] / d[..., 2]                               # view-axis depth to z = 0 (d . fwd = 1)
    p0 = cam['pos'] + t0[..., None] * d
    on_plane = (np.abs(p0[..., 0]) < 4) & (np.abs(p0[..., 1]) < 4) & (t0 > 0)
    t1 = (1. - cam['pos'][2]) / d[..., 2]
    p1 = cam['pos'] + t1[..., None] * d
    e1, e2 = V[5] - V[4], V[6] - V[4]
    rr = p1 - V[4]
    den = e1[0] * e2[1] - e1[1] * e2[0]
    u = (rr[..., 0] * e2[1] - rr[..., 1] * e2[0]) / den
    vv = (e1[0] * rr[..., 1] - e1[1] * rr[..., 0]) / den
    in_tri = (u > 1e-6) & (vv > 1e-6) & (u + vv < 1 - 1e-6)
    assert (face[in_tri] == 2).all() and np.allclose(depth[in_tri], t1[in_tri], atol=1e-9)
    only_plane = on_plane & ~((u > -1e-6) & (vv > -1e-6) & (u + vv < 1 + 1e-6))
    assert np.isin(face[only_plane], [0, 1]).all() and np.allclose(depth[only_plane], t0[only_plane], atol=1e-9)
    assert st['near_crossing_faces'] == 0 and in_tri.sum() > 5 and only_plane.sum() > 500
    f_big, d_big, _ = rasterize_first_hit(V, F, cam)               # the same result with and without chunks
    assert np.array_equal(face, f_big) and np.array_equal(depth, d_big)
    # 4 the whole chain: a slab with a vertical groove (floor and walls flagged as cavity), camera straight above; the mask is a
    #   vertical band and its skeleton lies within 1 pixel of the groove centreline
    hw, dz = .12, .3
    quads, flags = [], []

    def quad(a, b, c, d_, flag):
        n = len(quads) * 4
        quads.append([a, b, c, d_]); flags.extend([flag, flag])
        return [[n, n + 1, n + 2], [n, n + 2, n + 3]]
    Fs = []
    Fs += quad([-3, -3, 0], [-hw, -3, 0], [-hw, 3, 0], [-3, 3, 0], False)
    Fs += quad([hw, -3, 0], [3, -3, 0], [3, 3, 0], [hw, 3, 0], False)
    Fs += quad([-hw, -3, -dz], [hw, -3, -dz], [hw, 3, -dz], [-hw, 3, -dz], True)
    Fs += quad([-hw, -3, 0], [-hw, -3, -dz], [-hw, 3, -dz], [-hw, 3, 0], True)
    Fs += quad([hw, -3, -dz], [hw, -3, 0], [hw, 3, 0], [hw, 3, -dz], True)
    V2 = np.array(quads, float).reshape(-1, 3)
    F2 = np.array(Fs)
    flag = np.array(flags)
    cam2 = camera(dict(position_enu_m=[.37, .11, 5.], target_enu_m=[.37, .11, 0.], focal_length_mm=35., filmback_mm=[36., 36.]), [96, 96])
    face2, depth2, _ = rasterize_first_hit(V2, F2, cam2)
    hit = face2 >= 0
    m = np.zeros(face2.shape, bool); m[hit] = flag[face2[hit]]
    width_px = 2 * hw * cam2['fpx'] / 5.
    assert abs(m.sum(1).max() - width_px) <= 1.5 and m.any(), (m.sum(1).max(), width_px)
    sk, _ = zhang_suen(m)
    centre = np.zeros(face2.shape, bool)
    xc = cam2['w'] / 2 + cam2['fpx'] * (0. - .37) / 5. - .5        # column of the groove centreline x = 0
    centre[:, int(round(xc))] = m.any(1)
    e = evaluate(sk, centre)
    assert e['skeleton_to_centreline_frac_gt_2px'] == 0. and e['skeleton_to_centreline_median_px'] <= 1., e
    assert e['skeleton_components'] == 1, e
    print('selftest PASS')


# ------------------------------------------------------------------ the 11 viewpoints

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', help='folder with one sub-folder per frame (a0044 ...)')
    ap.add_argument('--index', default=str(fio.INDEX))
    ap.add_argument('--out', default=str(fio.RECORDS / 'mask_skeleton.json'))
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if not args.frames:
        ap.error('--frames is required')
    index = fio.index(args.index)
    fio.check_viewpoints(index)
    out = []
    for key, (outcrop, image_set, sid, pid) in fio.VIEWPOINTS.items():
        t = time.time()
        mask, rec = mouth_mask(args.frames, key)
        check = bool(rec['depth_aligned_fraction_of_rock'] >= MIN_ALIGNED and rec['opening_class_inside_mask_fraction'] >= MIN_MOUTH_INSIDE)
        skel, rounds = zhang_suen(mask)
        lab = fio.label_dir(args.frames, key)
        centre = (np.load(lab / 'target.npy') > 0) | (np.load(lab / 'u_line.npy') > 0)
        e = evaluate(skel, centre)
        e.update(key=key, view=fio.view_of(index[key]) if index[key]['block'] == 'C' else sid, outcrop=outcrop, image_set=image_set,
                 pixel_mm=float(index[key]['pixel_mm']), checks_passed=check,
                 depth_aligned_fraction_of_rock=rec['depth_aligned_fraction_of_rock'],
                 opening_class_inside_mask_fraction=rec['opening_class_inside_mask_fraction'], mask_px=int(mask.sum()),
                 cavity_face_count_matches=rec['cavity_face_count_matches_geometry_checks'], occluded_share=occluded_share(args.frames, key),
                 **fio.view_angle(args.frames, key))
        out.append(e)
        print('%s %-22s checks %s  aligned %.4f  mouth in mask %.4f  skeleton > 2 px from the line %.3f  line with no skeleton nearby %.3f  '
              'components %.3f  theta %.2f  [%.0f s]' % (
                  key, e['view'], check, rec['depth_aligned_fraction_of_rock'], rec['opening_class_inside_mask_fraction'],
                  e['skeleton_to_centreline_frac_gt_2px'] or 0, e['centreline_to_skeleton_frac_gt_2px'] or 0, e['component_ratio'],
                  e['theta_deg'], time.time() - t), flush=True)
    ok = [e for e in out if e['checks_passed'] and e['skeleton_to_centreline_frac_gt_2px'] is not None]
    rng = lambda k: [min(e[k] for e in ok), statistics.median(e[k] for e in ok), max(e[k] for e in ok)] if ok else None  # noqa: E731
    res = dict(
        description='Skeleton (Zhang-Suen) of the fracture-mouth mask against the visible centreline of the labels (final centreline plus '
                    'unclear line), one image for each of 11 viewpoints (views: one row per viewpoint).',
        paper_items=['main text Section 4.2, last paragraph (7.0%-53.8% of the skeleton pixels more than 2 px from the visible centreline, median '
                     '33.9%; 9 aerial viewpoints and 2 close-range viewpoints at 2 mm/px)',
                     'supplement S4: role of each step and the mask skeleton, second paragraph and the skeleton table (with theta)'],
        script='reproduce/paper1/analysis/mask_skeleton.py',
        units=dict(skeleton_to_centreline_frac_gt_2px='fraction of skeleton pixels more than 2 px from the nearest visible-centreline pixel',
                   centreline_to_skeleton_frac_gt_2px='fraction of visible-centreline pixels more than 2 px from the nearest skeleton pixel',
                   component_ratio='8-connected components of the skeleton / of the visible centreline',
                   occluded_share='occluded length / in-frame length of the joints (report.json, per_path sums)',
                   theta_deg='angle between the line of sight and the rock-surface normal at the camera target, degrees (camera plan)',
                   depth_aligned_fraction_of_rock='rock pixels whose first-hit depth agrees with the rendered depth / rock pixels',
                   opening_class_inside_mask_fraction='fracture-mouth pixels of the label program inside the mask / all of them',
                   pixel_mm='nominal pixel size, mm', distances='pixels (Euclidean, between pixel centres)'),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        notes=['The skeleton is the medial axis of the visible mouth: it changes with the viewing direction and the shape of the mouth and '
               'holds nothing of the occluded part of a trace (occluded_share).',
               'At the viewpoints with a fault (a0001, a0002, d0001, a0018) occluded_share counts joints only.'],
        views=out, views_evaluated=len(ok),
        skeleton_to_centreline_frac_gt_2px_min_med_max=rng('skeleton_to_centreline_frac_gt_2px'),
        centreline_to_skeleton_frac_gt_2px_min_med_max=rng('centreline_to_skeleton_frac_gt_2px'),
        skeleton_to_centreline_median_px_min_med_max=rng('skeleton_to_centreline_median_px'),
        component_ratio_min_med_max=rng('component_ratio'))
    fio.write_json(args.out, res)
    print({k: v for k, v in res.items() if k.endswith('min_med_max')})
    return 0


if __name__ == '__main__':
    sys.exit(main())
