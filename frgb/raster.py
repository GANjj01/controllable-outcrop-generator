"""Pixel -> final-mesh face by CPU ray casting (pixel-centre rays, z-buffer over all faces).

Camera frame: the request's world ENU (source ENU + scanned translation), shot position/target, focal and filmback;
pixel (x, y) centre ray = forward + sx * right + sy * up. Triangles crossing the near plane are clipped
(Sutherland-Hodgman) and keep their face id. For every pixel the nearest hit face, its view-axis depth and the nearest
hit of any OTHER face (id and depth) are kept, so front/back surfaces are detectable."""
import numpy as np

VERTICAL_UP_SWITCH = 0.99


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
        raise ValueError('filmback aspect differs from resolution aspect (no crop rule implemented)')
    near = float(shot.get('near_clip_cm', 10.)) / 100.
    return dict(pos=pos, fwd=fwd, right=right, up=up, fpx=shot['focal_length_mm'] / fw * w, w=int(w), h=int(h), near=near)


def rays(cam, x, y):
    sx = (x + .5 - cam['w'] / 2) / cam['fpx']
    sy = -(y + .5 - cam['h'] / 2) / cam['fpx']
    return cam['fwd'] + sx[..., None] * cam['right'] + sy[..., None] * cam['up']


def _clip_near(tri, near):
    """Clip one camera-space triangle (3x3, z forward) to z >= near; fan triangles of the kept polygon."""
    poly = []
    for a, b in zip(tri, np.roll(tri, -1, 0)):
        if a[2] >= near:
            poly.append(a)
        if (a[2] >= near) != (b[2] >= near):
            poly.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    return [np.array([poly[0], poly[i], poly[i + 1]]) for i in range(1, len(poly) - 1)]


def camera_triangles(vertices, faces, cam):
    rel = vertices - cam['pos']
    c = np.stack([rel @ cam['right'], rel @ cam['up'], rel @ cam['fwd']], 1)
    tri = c[faces]
    z = tri[..., 2]
    front = (z >= cam['near']).all(1)
    cross = (z >= cam['near']).any(1) & ~front
    ids = [np.flatnonzero(front)]
    tris = [tri[front]]
    extra_t, extra_i = [], []
    for f in np.flatnonzero(cross):
        for t in _clip_near(tri[f], cam['near']):
            extra_t.append(t)
            extra_i.append(f)
    if extra_t:
        tris.append(np.array(extra_t))
        ids.append(np.array(extra_i, np.int64))
    return np.concatenate(tris), np.concatenate(ids), int(cross.sum())


def rasterize(vertices, faces, cam, *, max_candidates=30_000_000):
    """Return dict(depth, face, depth2, face2) per pixel (inf / -1 = none) and the near-crossing count."""
    w, h = cam['w'], cam['h']
    tri, fid, crossing = camera_triangles(vertices, faces, cam)
    X = w / 2 + cam['fpx'] * tri[..., 0] / tri[..., 2] - .5
    Y = h / 2 - cam['fpx'] * tri[..., 1] / tri[..., 2] - .5
    x0 = np.clip(np.ceil(X.min(1)), 0, w).astype(np.int64)
    x1 = np.clip(np.floor(X.max(1)), -1, w - 1).astype(np.int64)
    y0 = np.clip(np.ceil(Y.min(1)), 0, h).astype(np.int64)
    y1 = np.clip(np.floor(Y.max(1)), -1, h - 1).astype(np.int64)
    keep = np.flatnonzero((x1 >= x0) & (y1 >= y0))
    invz = 1. / tri[..., 2]
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    depth = np.full(h * w, np.inf)
    cand = []

    def candidates(sel):
        n = bw[sel] * bh[sel]
        t = np.repeat(sel, n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        xx = x0[t] + k % bw[t]
        yy = y0[t] + k // bw[t]
        ax, ay, bx, by, cx, cy = X[t, 0], Y[t, 0], X[t, 1], Y[t, 1], X[t, 2], Y[t, 2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        with np.errstate(divide='ignore', invalid='ignore'):
            w0 = ((bx - xx) * (cy - yy) - (by - yy) * (cx - xx)) / area
            w1 = ((cx - xx) * (ay - yy) - (cy - yy) * (ax - xx)) / area
        w2 = 1 - w0 - w1
        eps = 1e-9  # pixel centres on a shared edge hit both faces (kept as a second hit)
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps) & (area != 0)
        t, xx, yy = t[inside], xx[inside], yy[inside]
        iz = w0[inside] * invz[t, 0] + w1[inside] * invz[t, 1] + w2[inside] * invz[t, 2]
        return yy * w + xx, 1. / iz, fid[t]

    order = keep[np.argsort((bw * bh)[keep])]
    sizes = np.cumsum((bw * bh)[order])
    start, base = 0, 0
    while start < len(order):
        stop = int(np.searchsorted(sizes, base + max_candidates, 'right'))
        stop = max(stop, start + 1)
        pix, z, f = candidates(order[start:stop])
        cand.append((pix, z, f))
        np.minimum.at(depth, pix, z)
        base = sizes[stop - 1]
        start = stop
    face = np.full(h * w, -1, np.int64)
    for pix, z, f in cand:
        win = z == depth[pix]
        face[pix[win]] = f[win]
    depth2 = np.full(h * w, np.inf)
    for pix, z, f in cand:
        other = f != face[pix]
        np.minimum.at(depth2, pix[other], z[other])
    face2 = np.full(h * w, -1, np.int64)
    for pix, z, f in cand:
        win = (f != face[pix]) & (z == depth2[pix])
        face2[pix[win]] = f[win]
    shape = (h, w)
    return dict(depth=depth.reshape(shape), face=face.reshape(shape), depth2=depth2.reshape(shape),
                face2=face2.reshape(shape), near_crossing_faces=crossing)


def face_normals(vertices, faces):
    n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def dilate(mask, r):
    """Binary dilation with a disc of radius r px."""
    h, w = mask.shape
    pad = np.pad(mask, r)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= pad[r + dy:r + dy + h, r + dx:r + dx + w]
    return out


def classify(r, prot_face, normals, cam, *, face_code=None, front_back_m=.05, grazing_cos=.1, edge_jump=.02,
             ue_rock=None, ue_code=None, ue_depth_m=None, depth_tol_rel=.002, depth_tol_abs=.01, dilate_px=2):
    """Per pixel: 'F' protected, 'U' untrusted/ambiguous, 'N' candidate negative, 'X' not outcrop. F = protected first or
    ambiguous-second face, dilated by dilate_px. Precedence F > U > N;"""
    depth, face, depth2, face2 = r['depth'], r['face'], r['depth2'], r['face2']
    h, w = depth.shape
    hit = face >= 0
    f0 = np.maximum(face, 0)
    f2 = np.maximum(face2, 0)
    with np.errstate(invalid='ignore'):
        close2 = hit & (face2 >= 0) & (depth2 - depth < front_back_m)
    differs = prot_face[f0] != prot_face[f2]
    if face_code is not None:
        differs |= face_code[f0] != face_code[f2]
    U = close2 & differs
    yy, xx = np.mgrid[0:h, 0:w]
    d = rays(cam, xx, yy)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    cosang = np.abs((normals[f0] * d).sum(-1))
    U |= hit & (cosang < grazing_cos)
    z = np.where(hit, depth, np.inf)
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(z, shift, axis)
            with np.errstate(invalid='ignore'):
                U |= hit & ~(np.abs(nb - z) <= edge_jump * np.minimum(nb, z))
    stats = dict(cpu_hit=int(hit.sum()), near_crossing_faces=r['near_crossing_faces'])
    rock = hit.copy()
    X_ue = np.zeros((h, w), bool)
    if ue_rock is not None:
        stats.update(both_rock=int((hit & ue_rock).sum()), cpu_hit_ue_not_rock=int((hit & ~ue_rock).sum()),
                     ue_rock_no_cpu_hit=int((~hit & ue_rock).sum()))
        U |= ue_rock & ~hit
        rock = ue_rock.copy()
        X_ue = ~ue_rock
    if ue_code is not None and face_code is not None:
        mism = hit & rock & (face_code[f0] != ue_code)
        stats['code_mismatch'] = int(mism.sum())
        U |= mism
    if ue_depth_m is not None:
        both = hit & rock & np.isfinite(ue_depth_m)
        res = np.abs(depth - ue_depth_m)
        bad = both & ~(res <= np.maximum(depth_tol_abs, depth_tol_rel * depth))
        stats['depth_residual_m'] = dict(p50=float(np.median(res[both])), p99=float(np.percentile(res[both], 99)),
                                         max=float(res[both].max())) if both.any() else None
        stats['depth_mismatch'] = int(bad.sum())
        U |= bad
    F = np.zeros((h, w), bool)
    F[hit] = prot_face[f0[hit]]
    F |= close2 & prot_face[f2]
    F = dilate(F, dilate_px) & rock
    label = np.full((h, w), 'X', '<U1')
    label[rock] = 'N'
    label[rock & U] = 'U'
    label[F] = 'F'
    label[X_ue] = 'X'
    stats.update({k: int((label == k).sum()) for k in 'FUNX'})
    return label, stats
