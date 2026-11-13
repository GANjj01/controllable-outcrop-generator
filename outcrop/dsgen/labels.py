"""Decode material-index and depth passes and project geological traces into per-shot labels."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..dataset.export import read_exr_channel
from .stage import load_package, classify_interfaces
from .legacy import name as _label   # old codebooks carry the former names

LINE_CLASSES = {'fault': 1, 'joint': 2, 'contact': 3}
LINE_COLORS = {'fault': (255, 40, 40), 'joint': (40, 220, 255), 'contact': (255, 230, 0)}


def camera_matrices(cam, resolution):
    m = np.array([1., -1., 1.])
    position = np.asarray(cam['position_cm']) * .01 * m
    right, up, forward = (np.asarray(cam[k]) * m for k in ('right', 'up', 'forward'))
    rot = np.array([right, -up, forward])
    w, h = resolution
    f = cam['focal_length_mm']
    fb = cam['filmback_mm']
    K = np.array([[w * f / fb[0], 0., (w - 1) / 2], [0., h * f / fb[1], (h - 1) / 2], [0., 0., 1.]])
    return rot, position, K


def project(points, rot, position, K):
    cam = (points - position) @ rot.T
    z = cam[:, 2]
    px = cam @ K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = px[:, :2] / z[:, None]
    return uv, z


def check_import(mesh_vertices, mesh_faces, report):
    """Hash lookup on a 0.01 cm lattice (checking the 27 neighbouring cells) instead of all-pairs search."""
    expected = mesh_vertices[np.unique(mesh_faces)] * np.array([100., -100., 100.])
    actual = np.asarray(report['imported_vertices_cm'])
    q = .01
    keys = {tuple(k) for k in np.floor(expected / q).astype(np.int64)}
    offsets = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    cells = np.floor(actual / q).astype(np.int64)
    found = np.array([any(tuple(c + o) in keys for o in offsets) for c in cells])
    return {'sampled_vertices': int(len(actual)), 'unmatched_sampled_vertices': int((~found).sum()),
            'tolerance_cm': 2 * q, 'triangles': report['triangles'], 'expected_triangles': int(len(mesh_faces)),
            'pass': bool(found.all() and abs(report['triangles'] - len(mesh_faces)) <= 1e-3 * len(mesh_faces))}


def sampled_agreement(vertices, faces, face_codes, codes, depth, rot, position, K, samples=40000, seed=0, front_face=None, stats=None):
    """``front_face``: the nearest face index at each pixel centre of an independent z-buffer of the whole display mesh
    (nearest_face_image, no depth window). Visibility is then the identity of the front-most face: a sample counts
    only where the sampled face itself is front-most at its pixel; the depth window can only reject (a face a
    centimetre behind a ledge lip is within the window at a few metres but is not visible)."""
    h, w = codes.shape
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), min(samples, len(faces)), replace=False)
    c = vertices[faces[pick]].mean(1)
    uv, z = project(c, rot, position, K)
    x, y = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    ok = (z > .1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y, z, fc = x[ok], y[ok], z[ok], face_codes[pick][ok]
    seen = (codes[y, x] > 0) & (np.abs(depth[y, x] - z) < .005 + .002 * z)
    if front_face is not None:
        own = front_face[y, x] == pick[ok]
        if stats is not None:
            stats.update({'visibility': 'front-most face identity (independent z-buffer); depth window rejects only',
                          'in_depth_window': int(seen.sum()), 'in_window_not_front_most': int((seen & ~own).sum())})
        seen = seen & own
    # Ignore samples whose 3x3 neighbourhood mixes codes (contact pixels).
    pad = np.pad(codes, 1, mode='edge')
    uniform = np.ones(len(x), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            uniform &= pad[y + dy, x + dx] == codes[y, x]
    use = seen & uniform
    # Rock-to-rock samples check the camera convention and ID decoding of the geology labels; a
    # rock face against a soil pixel is usually a face buried a few mm under the ground item or a
    # grazing neighbour within the depth tolerance (reported, not a geology label error).
    from .semantics import is_rock
    rock = use & is_rock(fc) & is_rock(codes[y, x])
    rate = lambda m: float((fc[m] == codes[y[m], x[m]]).mean()) if m.any() else float('nan')
    return rate(use), int(use.sum()), rate(rock), int(rock.sum())


def requery_contact_band(codes, depth, rot, position, K, truth, code_of, face_size):
    """Exact per-pixel truth near contacts: unproject every pixel within a band around boundaries
    between two different rock codes (band >= twice the projected size of the finest faces) and
    take its region code from a v2 point query at that 3D position."""
    from .semantics import is_rock
    h, w = codes.shape
    rock = is_rock(codes)
    edge = np.zeros_like(rock)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = codes[:h - dy, :w - dx], codes[dy:, dx:]
        # Only bedrock-bedrock boundaries: soil on a rock patch is v2 rock underneath, and a
        # requery there would repaint visible soil as bedrock.
        diff = (a != b) & is_rock(a) & is_rock(b)
        edge[:h - dy, :w - dx] |= diff
        edge[dy:, dx:] |= diff
    if not edge.any():
        return codes, {'band_px': 0, 'queried': 0, 'changed': 0, 'unmapped': 0}
    near = float(np.percentile(depth[rock], 1))
    band = int(np.clip(np.ceil(2 * face_size * K[0, 0] / max(near, .3)), 2, 30))
    grown = edge.copy()
    for _ in range(band):
        g = grown.copy()
        g[1:] |= grown[:-1]; g[:-1] |= grown[1:]; g[:, 1:] |= grown[:, :-1]; g[:, :-1] |= grown[:, 1:]
        grown = g
    ys, xs = np.nonzero(grown & rock)
    rays = np.c_[xs, ys, np.ones(len(xs))] @ np.linalg.inv(K).T
    pts = (rays * depth[ys, xs][:, None]) @ rot + position
    status, b, u, l = truth.query(pts)
    new = codes.copy()
    unmapped = 0
    from .semantics import code_of as semantic_code
    soil = semantic_code('soil')
    no_rock = 0
    for i, (st, bb, uu, ll) in enumerate(zip(status, b, u, l)):
        if st != 'occupied':
            # v2 has no rock here (e.g. a fault void): never label it bedrock.
            c = soil
            no_rock += 1
        else:
            c = code_of.get((bb, uu, ll))
        if c is None:
            unmapped += 1
            continue
        new[ys[i], xs[i]] = c
    changed = int((new != codes).sum())
    return new, {'band_px': band, 'queried': int(len(ys)), 'changed': changed, 'unmapped': unmapped, 'not_rock_in_v2': no_rock}


# Same continuous surface: a depth predicted from a pixel along the image plane with the robust
# local gradient must match within SURFACE_TOL_ABS_M + SURFACE_TOL_REL * depth. Calibrated on the
# continuous surfaces stay below ~0.07 % of depth for 99.5 % of pairs, occlusion steps form a flat
# tail above ~0.3 %. Fixed here; never tuned against v2 distances.
SURFACE_TOL_ABS_M = .005
SURFACE_TOL_REL = .002


def surface_tol(z):
    return SURFACE_TOL_ABS_M + SURFACE_TOL_REL * np.asarray(z)


def surface_gradient(depth, valid):
    """Robust depth gradient per pixel (d/dx, d/dy): per axis the smaller-magnitude one-sided
    difference between valid pixels (an occlusion step affects one side only); 0 if none."""
    out = []
    for axis in (1, 0):
        d = np.moveaxis(depth, axis, 0).astype(np.float64)
        v = np.moveaxis(valid, axis, 0)
        fwd, bwd = np.full(d.shape, np.nan), np.full(d.shape, np.nan)
        ok = v[1:] & v[:-1]
        diff = np.where(ok, d[1:] - d[:-1], np.nan)
        fwd[:-1], bwd[1:] = diff, diff
        g = np.where(np.isnan(fwd), bwd, np.where(np.isnan(bwd), fwd, np.where(np.abs(fwd) < np.abs(bwd), fwd, bwd)))
        out.append(np.moveaxis(np.nan_to_num(g), 0, axis))
    return out[0], out[1]


def same_surface(depth, valid, grad, x, y, u, v, z):
    """Pixel (x, y) is visible bedrock on the continuous surface through the point at sub-pixel
    (u, v) with depth z: its depth, carried to (u, v) with the robust gradient, matches z."""
    gx, gy = grad
    pred = depth[y, x] + gx[y, x] * (u - x) + gy[y, x] * (v - y)
    return valid[y, x] & (np.abs(z - pred) <= surface_tol(z))


def visible_runs(seg, rot, position, K, depth, valid, step_px=.5, grad=None, with_depth=False):
    """Split a 3D segment into image polylines whose samples lie on the visible surface."""
    h, w = depth.shape
    uv, z = project(seg, rot, position, K)
    if (z <= .1).any() or not np.isfinite(uv).all():
        return ([], []) if with_depth else []
    steps = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(steps)]
    n = int(max(2, np.ceil(cum[-1] / step_px) + 1))
    # Resample uniformly in image space along the 3D segment/polyline.
    t = np.linspace(0, cum[-1], n)
    pts = np.stack([np.interp(t, cum, seg[:, k]) for k in range(3)], 1) if cum[-1] > 0 else seg[[0, -1]]
    n = len(pts)
    puv, pz = project(pts, rot, position, K)
    x = np.rint(puv[:, 0]).astype(int)
    y = np.rint(puv[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    vis = np.zeros(n, bool)
    # A sample is visible only where the pixel it is drawn on shows it: that pixel is visible
    # bedrock on the sample's continuous surface. (A 3x3 min..max depth window let samples hidden
    grad = grad if grad is not None else surface_gradient(depth, valid)
    k = np.nonzero(ok)[0]
    vis[k] = same_surface(depth, valid, grad, x[k], y[k], puv[k, 0], puv[k, 1], pz[k])
    runs, depths, current, cz = [], [], [], []
    for k in range(n):
        if vis[k]:
            current.append([float(puv[k, 0]), float(puv[k, 1])])
            cz.append(float(pz[k]))
        elif current:
            runs.append(current)
            depths.append(cz)
            current, cz = [], []
    if current:
        runs.append(current)
        depths.append(cz)
    keep = [i for i, r in enumerate(runs) if len(r) >= 2]
    runs, depths = [runs[i] for i in keep], [depths[i] for i in keep]
    return (runs, depths) if with_depth else runs


def clip_line_raster(cls_img, ins_img, samples, depth, valid, grad, reach=2.5):
    """Keep only drawn line pixels on the continuous surface of their nearest centre-line sample
    (a 3 px line otherwise spills 1 px across an occlusion step onto another surface, or onto
    non-bedrock). samples: (u, v, z) of the visible centre-line samples."""
    h, w = depth.shape
    drawn = (cls_img > 0) | (ins_img > 0)
    keep = np.zeros_like(drawn)
    if len(samples) and drawn.any():
        su, sv, sz = samples[:, 0], samples[:, 1], samples[:, 2]
        best = np.full((h, w), np.inf)
        bu, bv, bz = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
        r = int(np.ceil(reach))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px, py = np.rint(su).astype(int) + dx, np.rint(sv).astype(int) + dy
                d = np.hypot(su - px, sv - py)
                m = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (d <= reach)
                px, py, d, u, v, z = px[m], py[m], d[m], su[m], sv[m], sz[m]
                order = np.argsort(-d)  # nearest written last
                px, py, d, u, v, z = px[order], py[order], d[order], u[order], v[order], z[order]
                better = d < best[py, px]
                px, py = px[better], py[better]
                best[py, px], bu[py, px], bv[py, px], bz[py, px] = d[better], u[better], v[better], z[better]
        ys, xs = np.nonzero(drawn & np.isfinite(best))
        keep[ys, xs] = same_surface(depth, valid, grad, xs, ys, bu[ys, xs], bv[ys, xs], bz[ys, xs])
    return np.where(keep, cls_img, 0), np.where(keep, ins_img, 0), int(drawn.sum()), int(keep.sum())


def region_boundaries(codes, lookup_key):
    """Pixels where the chosen region attribute changes between two foreground pixels."""
    from .semantics import is_rock
    key = lookup_key[codes]
    fg = is_rock(codes)
    edge = np.zeros(codes.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = key[: key.shape[0] - dy, : key.shape[1] - dx], key[dy:, dx:]
        fa, fb = fg[: key.shape[0] - dy, : key.shape[1] - dx], fg[dy:, dx:]
        diff = (a != b) & fa & fb
        edge[: key.shape[0] - dy, : key.shape[1] - dx] |= diff
        edge[dy:, dx:] |= diff
    return edge


def _unproject(xs, ys, depth, rot, position, K):
    rays = np.c_[xs, ys, np.ones(len(xs))] @ np.linalg.inv(K).T
    return (rays * depth[ys, xs][:, None]) @ rot + position


def pixel_truth_audit(codes, depth, rot, position, K, truth, code_of, near_lines, seed, random_px=20000, line_px=20000, exclude=None):
    """Independent check of the final region image against v2: bedrock pixels whose 5x5
    neighbourhood carries one code (away from any drawn boundary) are unprojected and queried.
    Samples: random interior pixels plus pixels near the visible line labels (fault, joint and
    intrusive-contact traces, i.e. where thin dykes and junctions are). A pixel whose code is not
    the v2 answer at its 3D point is a label error the contact-band requery could not see (e.g. a
    thin dyke missing from the image)."""
    from .semantics import is_rock
    h, w = codes.shape
    pad = np.pad(codes, 2, mode='edge')
    uniform = is_rock(codes).copy()
    for dy in range(5):
        for dx in range(5):
            uniform &= pad[dy:dy + h, dx:dx + w] == codes
    if exclude is not None:
        # interface faces (B patch): their surface points lie on the v2 contact itself; audited by
        # interface_truth_audit at a point inside the kept rock instead
        uniform &= ~exclude
    rng = np.random.default_rng(seed)
    picks = []
    for pool, n in ((np.flatnonzero(uniform), random_px), (np.flatnonzero(uniform & near_lines), line_px)):
        if len(pool):
            picks.append(rng.choice(pool, min(n, len(pool)), replace=False))
    if not picks:
        return {'sampled': 0, 'near_lines': 0, 'mismatched': 0, 'not_rock_in_v2': 0, 'examples': []}
    flat = np.unique(np.concatenate(picks))
    ys, xs = np.unravel_index(flat, codes.shape)
    st, b, u, l = truth.query(_unproject(xs, ys, depth, rot, position, K))
    want = np.array([code_of.get((bb, uu, ll), -1) if s == 'occupied' else 0 for s, bb, uu, ll in zip(st, b, u, l)])
    bad = want != codes[ys, xs]
    return {'sampled': int(len(flat)), 'near_lines': min(line_px, int((uniform & near_lines).sum())),
            'mismatched': int(bad.sum()), 'not_rock_in_v2': int((bad & (want == 0)).sum()),
            'examples': [[int(x), int(y), int(c), int(t)] for x, y, c, t in zip(xs[bad][:5], ys[bad][:5], codes[ys, xs][bad][:5], want[bad][:5])]}


def bed_lookup(disp, sea_z, reach=1.):
    """Rock surface under water and beach sand (coast): the display-mesh rock vertices lower than
    ``sea_z + reach`` (sea floor, wave-cut platform, cliff foot), hashed in plan."""
    from .semantics import VertexHash
    nr = int(disp['rock_vertices'])
    V = disp['vertices'][:nr]
    V = V[V[:, 2] < sea_z + reach]
    flat = V * [1., 1., 0.]
    return {'V': V, 'flat': flat, 'index': VertexHash(flat, .1)}


def bedrock_truth(codes, depth, rot, position, K, truth, lith_of, fam_of, probe, bed=None):
    """Geological truth for every foreground pixel, including where bedrock is hidden by soil,
    debris, water or a road: the v2 answer at the visible point, or ``probe`` metres below it.
    Water and sediment pixels with a ``bed`` (coast): the visible point is the water / sand surface,
    above the rock; their truth is v2 2 cm under the rock surface vertically below (nearest bed
    vertex in plan within 15 cm), none if there is no rock surface below.
    Visible bedrock pixels keep their (contact-requeried) region code; images carry global
    lithology and rock-class codes."""
    from .lithology import LITHOLOGY, FAMILY_CODES
    from .semantics import code_of, is_rock, nearest_vertex_fast
    lith = lith_of[codes].astype(np.int64)
    fam = fam_of[codes].astype(np.int64)
    ys, xs = np.nonzero((codes > 0) & ~is_rock(codes))
    stats = {'queried': int(len(ys)), 'bedrock_found': 0, 'outside_rock': 0}
    if len(ys):
        p = _unproject(xs, ys, depth, rot, position, K)
        if bed is not None:
            over = np.isin(codes[ys, xs], [code_of('water'), code_of('sediment')])
            if over.any():
                k, dist = nearest_vertex_fast(p[over] * [1., 1., 0.], bed['flat'], cell=.1, index=bed['index'])
                hit = (k >= 0) & (dist < .15)
                q = np.full((int(over.sum()), 3), np.nan)
                q[hit] = bed['V'][k[hit]] - [0., 0., .02]
                q[~hit] = [-1e6, -1e6, -1e6]
                p[over] = q
                stats['bed_lookup'] = {'pixels': int(over.sum()), 'rock_surface_found': int(hit.sum())}
        # v2 has no rock outside its domain: points whose surface point and probe point both lie
        # and far-ground pixels).
        lo, hi = np.asarray(truth.snapshot.domain.minimum, float), np.asarray(truth.snapshot.domain.maximum, float)
        inside = lambda q: ((q >= lo) & (q <= hi)).all(1)
        ask = inside(p) | inside(p - [0., 0., probe])
        st = np.full(len(p), 'outside', object)
        l = np.full(len(p), None, object)
        if ask.any():
            st_a, _, _, l_a = truth.query(p[ask])
            st[ask], l[ask] = st_a, l_a
        miss = (st != 'occupied') & ask
        if miss.any():
            st2, _, _, l2 = truth.query(p[miss] - [0., 0., probe])
            st[miss] = st2
            l[miss] = l2
        ok = st == 'occupied'
        lith[ys, xs] = [LITHOLOGY.get(_label(x), {}).get('code', 0) if o else 0 for x, o in zip(l, ok)]
        fam[ys, xs] = [FAMILY_CODES.get(LITHOLOGY.get(_label(x), {}).get('family'), 0) if o else 0 for x, o in zip(l, ok)]
        stats.update(bedrock_found=int(ok.sum()), outside_rock=int((~ok).sum()))
    return {'images': {'lithology': lith.astype(np.uint8), 'family': fam.astype(np.uint8)}, 'stats': stats}


def surface_state(disp, codes, depth, rot, position, K, index=None):
    """Per-pixel applied recession (m) of the rock surface (bedrock and soil-on-rock pixels);
    NaN elsewhere. Nearest rock vertex of the display mesh to the unprojected pixel. ``index``:
    VertexHash of the rock vertices (built once per capture)."""
    from .semantics import nearest_vertex_fast
    if 'state' not in disp:
        return None
    nr = int(disp['rock_vertices'])
    ys, xs = np.nonzero(codes > 0)
    out = np.full(codes.shape, np.nan, np.float32)
    if not len(ys):
        return out
    p = _unproject(xs, ys, depth, rot, position, K)
    idx, dist = nearest_vertex_fast(p, disp['vertices'][:nr], cell=.1, index=index)
    ok = (idx >= 0) & (dist < .15)
    out[ys[ok], xs[ok]] = disp['state'][idx[ok], 0]
    return out


def _counts(image):
    values, counts = np.unique(image[image > 0], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def label_tables(book, size=None):
    """Region code -> (lithology code, rock-class code, unit instance) lookup arrays.

    Lithology and rock-class pixels carry the GLOBAL catalog codes (stable across scenes);
    lithologies outside the catalog get codes from 200 upward. Unit instances are scene-local."""
    from .lithology import LITHOLOGY, FAMILY_CODES
    extra = sorted({e['lithology'] for e in book if _label(e['lithology']) not in LITHOLOGY})
    lith_code = {**{n: r['code'] for n, r in LITHOLOGY.items()},
                 **{n: LITHOLOGY[_label(n)]['code'] for n in {e['lithology'] for e in book} if _label(n) in LITHOLOGY},
                 **{n: 200 + i for i, n in enumerate(extra)}}
    lith_names = {lith_code[n]: n for n in sorted({e['lithology'] for e in book})}
    unit_names = sorted({e['unit'] or e['body'] for e in book})
    size = max(size or 0, max(e['code'] for e in book) + 1)
    lith_of, fam_of, unit_of = (np.zeros(size, np.int64) for _ in range(3))
    for e in book:
        lith_of[e['code']] = lith_code[e['lithology']]
        fam_of[e['code']] = FAMILY_CODES.get(LITHOLOGY.get(_label(e['lithology']), {}).get('family'), 0)
        unit_of[e['code']] = unit_names.index(e['unit'] or e['body']) + 1
    return lith_of, fam_of, unit_of, lith_names, unit_names


def interface_face_image(V, F, codes_f, depth, rot, position, K, tol_abs=.002, tol_rel=.002):
    """Rule 2 (B patch): pixels showing a face that lies in a v2 interface plane. Each face is rasterised with
    perspective-correct depth and kept where it is the visible surface (|z - depth| within 2 mm or 0.2 %). Returns a
    uint16 image of record index + 1 (0 = none).

    Not used by export any more: the depth tolerance alone (~8 mm at 4 m) also accepts a riser lying a few mm behind the
    plain rock face that is actually seen; export uses nearest_face_image over the whole display mesh. Kept for the
    site-2 check scripts."""
    h, w = depth.shape
    out = np.zeros((h, w), np.uint16)
    for tri, code in zip(F, codes_f):
        uv, z = project(V[tri], rot, position, K)
        if (z <= 1e-3).any() or not np.isfinite(uv).all():
            continue
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x0 > x1 or y0 > y1:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        (ax, ay), (bx, by), (cx, cy) = uv
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        l3 = 1 - l1 - l2
        ins = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not ins.any():
            continue
        zi = 1. / (l1 / z[0] + l2 / z[1] + l3 / z[2])
        dpx = depth[ys, xs]
        vis = ins & (dpx > 0) & (np.abs(zi - dpx) <= np.maximum(tol_abs, tol_rel * dpx))
        out[ys[vis], xs[vis]] = code
    return out


# Upper bound of (face, bounding-box pixel) pairs expanded at once by the rasterisers below. Chunking by face count
# alone does not bound memory: a close-up camera can put a whole screen (2 M pixels at 1920 x 1080) into the bounding
PIXEL_PAIR_BUDGET = 4_000_000


def bbox_chunks(npix, budget=PIXEL_PAIR_BUDGET):
    """Consecutive index ranges [a, b) of faces whose bounding-box pixel counts ``npix`` sum to at most ``budget``
    (a single face over the budget forms its own range)."""
    npix = np.asarray(npix, np.int64)
    out, a, n = [], 0, len(npix)
    c = np.cumsum(npix)
    while a < n:
        base = c[a - 1] if a else 0
        b = int(np.searchsorted(c, base + budget, side='right'))
        b = max(b, a + 1)
        out.append((a, min(b, n)))
        a = b
    return out


def _raster_boxes(u, w, h):
    """Clipped pixel bounding boxes of projected triangles u (n, 3, 2): x0, y0, nx, ny (nx, ny <= 0: empty)."""
    x0 = np.maximum(np.ceil(u[..., 0].min(1)), 0).astype(np.int64)
    x1 = np.minimum(np.floor(u[..., 0].max(1)), w - 1).astype(np.int64)
    y0 = np.maximum(np.ceil(u[..., 1].min(1)), 0).astype(np.int64)
    y1 = np.minimum(np.floor(u[..., 1].max(1)), h - 1).astype(np.int64)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def nearest_face_image(V, F, shape, rot, position, K, window=None, chunk=200000, pair_budget=None):
    """Z-buffer of a triangle mesh at the pixel centres: the nearest face index under each pixel
    (-1 = none) and its perspective-correct depth (inf = none). Faces with a vertex behind the
    camera are skipped, as in interface_face_image. ``window = (depth, tol_abs, tol_rel)`` keeps
    only fragments within max(tol_abs, tol_rel * depth) of a depth image. Faces are processed in groups whose expanded
    (face, bounding-box pixel) pairs stay within ``pair_budget`` (default PIXEL_PAIR_BUDGET); ``chunk`` only bounds the
    projection step. The result does not depend on the grouping."""
    h, w = shape
    budget = PIXEL_PAIR_BUDGET if pair_budget is None else int(pair_budget)
    if window is not None:
        wdepth, tol_abs, tol_rel = window
        wdepth = wdepth.ravel()
        wtol = np.maximum(tol_abs, tol_rel * wdepth)
    uv, z = project(V, rot, position, K)
    zbuf = np.full(h * w, np.inf)
    fbuf = np.full(h * w, -1, np.int64)
    for s in range(0, len(F), chunk):
        idx_all = np.arange(s, min(s + chunk, len(F)))
        zt_all = z[F[idx_all]]
        u_all = uv[F[idx_all]]
        ok = (zt_all > 1e-3).all(1) & np.isfinite(u_all).all((1, 2))
        idx_all, zt_all, u_all = idx_all[ok], zt_all[ok], u_all[ok]
        x0_all, y0_all, nx_all, ny_all = _raster_boxes(u_all, w, h)
        ok = (nx_all > 0) & (ny_all > 0)
        idx_all, zt_all, u_all, x0_all, y0_all, nx_all, ny_all = (a[ok] for a in (idx_all, zt_all, u_all, x0_all, y0_all, nx_all, ny_all))
        for a0, a1 in bbox_chunks(nx_all * ny_all, budget):
            idx, zt, u, x0, y0, nx, ny = (a[a0:a1] for a in (idx_all, zt_all, u_all, x0_all, y0_all, nx_all, ny_all))
            # every (face, pixel centre in its bounding box) pair
            n = nx * ny
            r = np.repeat(np.arange(len(idx)), n)
            k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            xs, ys = x0[r] + k % nx[r], y0[r] + k // nx[r]
            a, b, c = u[r, 0], u[r, 1], u[r, 2]
            den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
            with np.errstate(divide='ignore', invalid='ignore'):
                l1 = ((b[:, 1] - c[:, 1]) * (xs - c[:, 0]) + (c[:, 0] - b[:, 0]) * (ys - c[:, 1])) / den
                l2 = ((c[:, 1] - a[:, 1]) * (xs - c[:, 0]) + (a[:, 0] - c[:, 0]) * (ys - c[:, 1])) / den
                l3 = 1 - l1 - l2
                zr = zt[r]
                zi = 1. / (l1 / zr[:, 0] + l2 / zr[:, 1] + l3 / zr[:, 2])
            ins = (np.abs(den) >= 1e-12) & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            pix, zi, f = (ys * w + xs)[ins], zi[ins], idx[r][ins]
            if window is not None:
                keep = (wdepth[pix] > 0) & (np.abs(zi - wdepth[pix]) <= wtol[pix])
                pix, zi, f = pix[keep], zi[keep], f[keep]
            if not len(pix):
                continue
            o = np.lexsort((zi, pix))
            pix, zi, f = pix[o], zi[o], f[o]
            first = np.r_[True, pix[1:] != pix[:-1]]
            pix, zi, f = pix[first], zi[first], f[first]
            nearer = zi < zbuf[pix]
            zbuf[pix[nearer]], fbuf[pix[nearer]] = zi[nearer], f[nearer]
    return fbuf.reshape(h, w), zbuf.reshape(h, w)


def interface_face_images(V, F, face_iface, face_codes, depth, rot, position, K, tol_abs=.002, tol_rel=.002):
    """Returns the record index + 1 (uint16, 0 = none) and the mesh region code of that face (0 = none)."""
    fidx, _ = nearest_face_image(V, F, depth.shape, rot, position, K, window=(depth, tol_abs, tol_rel))
    f = np.maximum(fidx, 0)
    iface = np.where(fidx >= 0, face_iface[f], 0)
    return iface.astype(np.uint16), np.where(iface > 0, face_codes[f], 0).astype(np.int64)


def interface_truth_audit(codes, iface, depth, rot, position, K, truth, code_of, seed, n=20000, step=1e-4):
    """Interface-face pixels (B patch, rule 2): the region code must be v2's answer just inside the
    kept rock, ``step`` further along the viewing ray (a visible face faces the camera). As in
    pixel_truth_audit, only pixels whose 5x5 neighbourhood carries one code: next to a unit boundary
    on the face the oblique step crosses into the neighbouring unit (found in the B sample: the
    mismatches sat 0-2 px from boundaries and grew with the step)."""
    from .semantics import is_rock
    h, w = codes.shape
    pad = np.pad(codes, 2, mode='edge')
    uniform = (iface > 0) & is_rock(codes)
    for dy in range(5):
        for dx in range(5):
            uniform &= pad[dy:dy + h, dx:dx + w] == codes
    pool = np.flatnonzero(uniform)
    pool_rule = '5x5 uniform code'
    if not len(pool):
        # bedrock pixels (the stats.acceptance gate requires an audit wherever interface pixels are visible)
        pad1 = np.pad(codes, 1, mode='edge')
        u3 = (iface > 0) & is_rock(codes)
        for dy in range(3):
            for dx in range(3):
                u3 &= pad1[dy:dy + h, dx:dx + w] == codes
        pool, pool_rule = np.flatnonzero(u3), '3x3 uniform code'
        if not len(pool):
            pool, pool_rule = np.flatnonzero((iface > 0) & is_rock(codes)), 'all interface bedrock pixels'
    if not len(pool):
        return {'sampled': 0, 'mismatched': 0}
    flat = np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False)
    ys, xs = np.unravel_index(flat, codes.shape)
    pts = _unproject(xs, ys, depth, rot, position, K)
    ray = pts - position
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    st, b, u, l = truth.query(pts + step * ray)
    want = np.array([code_of.get((bb, uu, ll), -1) if s_ == 'occupied' else 0 for s_, bb, uu, ll in zip(st, b, u, l)])
    bad = want != codes[ys, xs]
    return {'sampled': int(len(flat)), 'mismatched': int(bad.sum()), 'pool': pool_rule,
            'examples': [[int(x), int(y), int(c), int(t)] for x, y, c, t in zip(xs[bad][:5], ys[bad][:5], codes[ys, xs][bad][:5], want[bad][:5])]}


def export(request_path, *, line_width=3):
    request = json.loads(Path(request_path).read_text(encoding='utf8'))
    if request.get('mode', 'final') != 'final':
        raise RuntimeError('%s capture: labels are exported only from final captures' % request.get('mode'))
    out = Path(request['directory'])
    report = json.loads((out / 'ue_report.json').read_text(encoding='utf8'))
    if not report['success']:
        raise RuntimeError('UE capture did not succeed: %s' % report['errors'][:3])
    pkg = load_package(request['package'])
    book = request['codebook']
    from .semantics import BEDROCK_PROBE_M, CLASSES, is_rock, semantic_image
    codes_set = {e['code'] for e in book} | {int(c) for c in request.get('semantic_codes', {})}
    from .lithology import LITHOLOGY, FAMILY_CODES
    # Reserved semantic codes map to 0 in every geology image (they are not bedrock).
    lith_of, fam_of, unit_of, lith_names, unit_names = label_tables(book, size=2048)
    classes = request['line_classes']
    if request.get('display'):
        disp = np.load(out / 'display.npz')
        mesh_v, mesh_f, mesh_codes = disp['vertices'], disp['faces'], disp['codes']
        ids = json.loads((out / 'display_polylines.json').read_text(encoding='utf8'))
        pts, lengths = disp['polyline_points'], disp['polyline_lengths']
        starts = np.r_[0, np.cumsum(lengths)]
        lines = [(iid, pts[starts[i]:starts[i + 1]]) for i, iid in enumerate(ids)]
        coplanar = None
        if 'coplanar_faces' in disp:
            # B patch, rule 2: faces in a v2 interface plane are labelled as faces, not as lines
            cop = disp['coplanar_faces']
            coplanar = {'all': cop, 'records': json.loads((out / 'display_coplanar.json').read_text(encoding='utf8'))}
    else:
        index = {(e['body'], e['unit'] or '', e['lithology']): e['code'] for e in book}
        face_codes = np.array([index[(b, u if u is not None else '', l)] for b, u, l in zip(pkg['body'], pkg['unit'], pkg['lithology'])])
        mesh_v, mesh_f, mesh_codes = pkg['vertices'], pkg['faces'], face_codes
        lines = list(zip(pkg['segment_ids'], pkg['segments']))
    instances = sorted({iid for iid, _ in lines})
    truth, code_of, max_edge = None, None, request.get('max_edge_m', .08) / 8
    bed = None
    if request.get('display'):
        from .truth import TruthModel
        truth = TruthModel(request['package'])
        cm_ = (request.get('display_stats') or {}).get('cm')
        if cm_ is not None:
            truth.workers = 12
        code_of = {(e['body'], e['unit'], e['lithology']): e['code'] for e in book}
        expo = (request.get('provenance') or {}).get('exposure') or {}
        if 'sea_level_m' in expo and 'rock_vertices' in disp:
            # coast: water and beach pixels take their bedrock from the rock surface below them
            bed = bed_lookup(disp, expo['ground_z_m'] + expo['sea_level_m'])
    summary = {'schema': 'outcrop-dsgen-labels-1', 'import_check': check_import(mesh_v, mesh_f, report),
               'lithology_classes': {str(k): v for k, v in sorted(lith_names.items())},
               'lithology_catalog': {str(r['code']): {'name': n, 'family': r['family']} for n, r in LITHOLOGY.items()},
               'family_classes': {str(v): k for k, v in FAMILY_CODES.items()}, 'unit_instances': unit_names,
               'semantic_classes': {str(i): c for i, c in enumerate(CLASSES)},
               'provenance': request.get('provenance'),
               # groove floors vs v2 planes on the final surface (processes.groove_alignment)
               'groove_alignment': (request.get('display_stats') or {}).get('groove_alignment'),
               # surface-process fold guard: counts before each halving, and faces left inverted on
               # the final surface (stats.acceptance fails on any)
               'fold_guard': {'passes': (request.get('display_stats') or {}).get('fold_guard_passes'),
                              'final_flipped': (request.get('display_stats') or {}).get('fold_guard_final_flipped')},
               'line_classes': LINE_CLASSES, 'line_instances': [{'id': i + 1, 'interface': iid, 'class': classes[iid]} for i, iid in enumerate(instances)],
               'shots': []}
    if request.get('display') and coplanar is not None:
        summary['interface_face_instances'] = [{'id': i + 1, **r} for i, r in enumerate(coplanar['records'])]
        summary['interface_face_rule'] = ('interface_face.png: visible faces lying in a v2 interface plane (id -> '
                                          'interface_face_instances); lines_not_applicable.png: line and strata-boundary '
                                          'labels do not apply there (neither positive nor negative samples)')
    renders = out / 'renders'
    w, h = request['resolution']
    rock_hash = None
    for frame, (cam_id, cam_rgb, shot) in enumerate(zip(report['cameras']['id'], report['cameras']['rgb'], request['shots'])):
        for key in ('position_cm', 'forward', 'right', 'up', 'focal_length_mm', 'filmback_mm'):
            if np.max(np.abs(np.asarray(cam_id[key]) - np.asarray(cam_rgb[key]))) > 1e-4:
                raise RuntimeError('RGB/ID camera mismatch for ' + shot['id'])
        raw = read_exr_channel(renders / ('id.FinalImage.%04d.exr' % frame))
        codes = np.rint(raw).astype(np.int64)
        off = np.abs(raw - codes)
        # The ID pass carries a ~0.2% gain plus half-float rounding (52 reads back as 52.094): the
        # tolerance grows with the code; any code below the ceiling still rounds exactly.
        bad_integer = float((off > .01 + .004 * codes).mean())
        unknown = sorted(set(np.unique(codes).tolist()) - codes_set - {0})
        if unknown:
            raise RuntimeError('unknown region codes %s in %s' % (unknown, shot['id']))
        depth = read_exr_channel(renders / ('id.FinalImageDepth.%04d.exr' % frame)) / 100.
        fg = codes > 0
        rot, position, K = camera_matrices(cam_id, (w, h))
        pixel_truth = None
        iface_id = iface_code = None
        if request.get('display') and coplanar is not None:
            # rule 2: visible interface faces (nearest face of the whole mesh), with their record id
            # and their mesh region code
            iface_id, iface_code = interface_face_images(mesh_v, mesh_f, coplanar['all'], mesh_codes, depth, rot, position, K)
        raw_codes = codes
        if truth is not None and is_rock(codes).any():
            codes, pixel_truth = requery_contact_band(codes, depth, rot, position, K, truth, code_of, max_edge)
        if iface_code is not None:
            # A riser lying in a v2 contact plane: its surface points are on the contact, where the
            # band requery may answer either unit (or no rock). Where the ID pass shows the face, keep
            # its mesh code -- the unit of the kept rock behind it (processes.erode).
            keep = (iface_code > 0) & (raw_codes == iface_code)
            if pixel_truth is not None:
                pixel_truth['interface_pixels_restored'] = int((keep & (codes != iface_code)).sum())
            codes = np.where(keep, iface_code, codes)
        valid = is_rock(codes)  # visible bedrock: visible line labels are drawn over these pixels only
        depth = np.where(fg, depth, 0.).astype(np.float32)
        vis_stats = None
        front_face = None
        if (request.get('label_options') or {}).get('visibility') == 'front_face':
            front_face, _ = nearest_face_image(mesh_v, mesh_f, depth.shape, rot, position, K)
            vis_stats = {}
        agreement, agreement_samples, rock_agreement, rock_samples = sampled_agreement(mesh_v, mesh_f, mesh_codes, codes, depth, rot, position, K, seed=frame,
                                                                                       front_face=front_face, stats=vis_stats)
        shot_dir = out / 'shots' / shot['id']
        shot_dir.mkdir(parents=True, exist_ok=True)
        rgb = Image.open(renders / ('rgb.FinalImage.%04d.png' % frame)).convert('RGB')
        rgb.save(shot_dir / 'rgb.png')
        Image.fromarray(codes.astype(np.uint16)).save(shot_dir / 'region.png')
        Image.fromarray(lith_of[codes].astype(np.uint8)).save(shot_dir / 'lithology.png')
        Image.fromarray(fam_of[codes].astype(np.uint8)).save(shot_dir / 'family.png')
        sem = semantic_image(codes)
        Image.fromarray(sem).save(shot_dir / 'semantic.png')
        bedrock = bedrock_truth(codes, depth, rot, position, K, truth, lith_of, fam_of, BEDROCK_PROBE_M, bed=bed) if truth is not None else None
        if bedrock is not None:
            # Geological truth under cover: v2 at the pixel's surface point, or BEDROCK_PROBE_M below it.
            for name, img in bedrock['images'].items():
                Image.fromarray(img).save(shot_dir / (name + '_bedrock.png'))
        if request.get('display') and 'state' in disp and rock_hash is None:
            from .semantics import VertexHash
            rock_hash = VertexHash(disp['vertices'][:int(disp['rock_vertices'])], .1)
        state_px = surface_state(disp, codes, depth, rot, position, K, index=rock_hash) if request.get('display') else None
        if state_px is not None:
            np.save(shot_dir / 'recession_m.npy', state_px)
        Image.fromarray(unit_of[codes].astype(np.uint16)).save(shot_dir / 'unit.png')
        np.save(shot_dir / 'depth.npy', depth)
        strata_edge = region_boundaries(codes, unit_of)
        lith_edge = region_boundaries(codes, lith_of)
        line_cls = Image.new('L', (w, h), 0)
        line_ins = Image.new('I', (w, h), 0)
        dc, di = ImageDraw.Draw(line_cls), ImageDraw.Draw(line_ins)
        polylines = []
        grad = surface_gradient(depth, valid)
        samples = []
        for iid, seg in lines:
            runs, run_z = visible_runs(seg, rot, position, K, depth, valid, grad=grad, with_depth=True)
            for run, rz in zip(runs, run_z):
                pts = [(p[0] + 0., p[1] + 0.) for p in run]
                dc.line(pts, fill=LINE_CLASSES[classes[iid]], width=line_width)
                di.line(pts, fill=instances.index(iid) + 1, width=line_width)
                polylines.append({'instance': instances.index(iid) + 1, 'class': classes[iid], 'points_px': run})
                samples.append(np.c_[np.asarray(run), rz])
        # The drawn width stays on the centre line's own continuous surface (visible bedrock).
        cls_arr, ins_arr, drawn_px, kept_px = clip_line_raster(
            np.asarray(line_cls), np.asarray(line_ins, np.int64), np.concatenate(samples) if samples else np.zeros((0, 3)),
            depth, valid, grad)
        line_cls = Image.fromarray(cls_arr.astype(np.uint8))
        line_cls.save(shot_dir / 'lines_class.png')
        audit = None
        if truth is not None:
            # Pixels within ~6 px of a drawn line label (any class).
            lc = np.asarray(line_cls) > 0
            near = lc.copy()
            for _ in range(2):
                grown = near.copy()
                grown[1:] |= near[:-1]
                grown[:-1] |= near[1:]
                grown[:, 1:] |= near[:, :-1]
                grown[:, :-1] |= near[:, 1:]
                near = grown
            audit = pixel_truth_audit(codes, depth, rot, position, K, truth, code_of, near, seed=frame,
                                      exclude=(iface_id > 0) if iface_id is not None else None)
        Image.fromarray(ins_arr.astype(np.uint16)).save(shot_dir / 'lines_instance.png')
        Image.fromarray((strata_edge * 255).astype(np.uint8)).save(shot_dir / 'strata_boundary.png')
        (shot_dir / 'lines.json').write_text(json.dumps(polylines), encoding='utf8')
        iface_stats = None
        if iface_id is not None:
            # visible interface faces: bedrock pixels only (a riser under soil cover shows soil)
            iface = np.where(is_rock(codes) & (codes == iface_code), iface_id, 0).astype(np.uint16)
            na = iface > 0
            for _ in range(line_width + 1):
                grown = na.copy()
                grown[1:] |= na[:-1]
                grown[:-1] |= na[1:]
                grown[:, 1:] |= na[:, :-1]
                grown[:, :-1] |= na[:, 1:]
                na = grown
            Image.fromarray(iface).save(shot_dir / 'interface_face.png')
            Image.fromarray((na * 255).astype(np.uint8)).save(shot_dir / 'lines_not_applicable.png')
            iface_stats = {'pixels': {str(k): int((iface == k).sum()) for k in np.unique(iface) if k},
                           'truth_audit': interface_truth_audit(codes, iface, depth, rot, position, K, truth, code_of, seed=frame),
                           'lines_not_applicable_px': int(na.sum()),
                           'line_px_in_not_applicable': int(((np.asarray(line_cls) > 0) & na).sum()),
                           'strata_px_in_not_applicable': int((strata_edge & na).sum())}
        cam_record = {'K': K.tolist(), 'world_to_camera_rotation': rot.tolist(), 'position_enu_m': position.tolist(),
                      'camera_axes': 'right, down, forward', 'resolution': [w, h], 'shot': shot}
        (shot_dir / 'camera.json').write_text(json.dumps(cam_record, indent=1), encoding='utf8')
        # Visual QA overlay: lithology/strata edges plus projected traces on the photograph.
        over = np.asarray(rgb).copy()
        over[lith_edge] = (255, 255, 255)
        over[strata_edge & ~lith_edge] = (255, 140, 0)
        overlay = Image.fromarray(over)
        do = ImageDraw.Draw(overlay)
        for p in polylines:
            do.line([tuple(q) for q in p['points_px']], fill=LINE_COLORS[p['class']], width=line_width)
        overlay.save(shot_dir / 'overlay.png')
        light = shot.get('lighting') or {}
        summary['shots'].append({'id': shot['id'], 'camera_index': shot.get('camera_index'),
                                 'time_local': light.get('time_local'), 'period': light.get('period'),
                                 'sun_elevation_deg': light.get('sun_elevation_deg'), 'foreground_fraction': float(valid.mean()),
                                 'non_integer_id_fraction': bad_integer, 'reprojection_interior_agreement': agreement, 'reprojection_samples': agreement_samples,
                                 'reprojection_rock_agreement': rock_agreement, 'reprojection_rock_samples': rock_samples,
                                 **({'reprojection_visibility': vis_stats} if vis_stats is not None else {}),
                                 'contact_pixel_truth': pixel_truth, 'pixel_truth_audit': audit,
                                 'visible_polylines': len(polylines),
                                 'line_raster_clip': {'drawn_px': drawn_px, 'kept_px': kept_px},
                                 'line_pixels': {c: int((np.asarray(line_cls) == v).sum()) for c, v in LINE_CLASSES.items()},
                                 # Per-class pixel counts for the batch-level dataset statistics (stats.py).
                                 'lithology_pixels': _counts(lith_of[codes]), 'family_pixels': _counts(fam_of[codes]),
                                 'semantic_pixels': {CLASSES[int(k)]: v for k, v in _counts(sem).items()},
                                 'bedrock_under_cover': bedrock['stats'] if bedrock is not None else None,
                                 **({'interface_faces': iface_stats} if iface_stats is not None else {})})
    (out / 'labels_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    return summary
