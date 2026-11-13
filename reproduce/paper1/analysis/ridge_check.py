"""An independent ridge filter on the three label layers (main text Section 4.1, last paragraph; supplement S4, ridge filter).

A multi-scale Hessian-based filter that responds to dark lines and uses no geometry is applied to the rendered image; its
response on the pixels of each label layer is compared with its response on background rock (AUC).

Filter: Rec. 709 luma of the 8-bit sRGB values of frame.png; for sigma = 1, 2 and 4 pixels the Hessian of the Gaussian-smoothed
luma times sigma^2; positive part of its larger eigenvalue; maximum over the three scales (R); the reading of a pixel is the
maximum of R in its 3 x 3 neighbourhood.
Pixel sets: C final centreline (target); U unclear line on rock (u_line with region mask 1); O occluded line; O_near and O_far
the occluded-line pixels at an image distance d <= 2 px and d > 10 px from the visible evidence M (fracture-mouth pixels,
final centreline and unclear line). Background: valid pixels more than 20 px from M and from the occluded line, taken at a
fixed stride to at most 200,000 pixels. AUC = P(reading of a set pixel > reading of a background pixel) + 0.5 P(equal); a set
with fewer than 30 pixels in an image is skipped. Reported: median and range over the images.

    python reproduce/paper1/analysis/ridge_check.py --frames FRAMES

Reads, for the 50 frames of the multi-illumination aerial set (block C of records/frames_index.json),
FRAMES/<frame id>/q0/render/frame.png, frame.npz (region mask) and q0/trace_labels/{target,u_line,occluded_line,opening_class,valid}.npy;
writes records/round3/ridge_check.json. Needs NumPy, SciPy and Pillow (the labels environment).
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402


SIGMAS = (1.0, 2.0, 4.0)
REC709 = (0.2126, 0.7152, 0.0722)
NEAR_PX, FAR_PX = 2.0, 10.0          # d <= 2: next to the visible evidence; d > 10: far from it
BG_DIST_PX = 20.0                    # background: more than 20 px from M and from the occluded line
BG_MAX = 200000                      # at most 200,000 background pixels
MIN_SET = 30                         # an image with fewer pixels in a set is skipped for that set
SHAPE = (1024, 1024)
# the 5 viewpoints of the multi-illumination aerial set (representative frame -> outcrop type)
VIEW_REPS = (('a0077', 'flat'), ('a0074', 'standalone'), ('a0062', 'standalone'), ('a0099', 'standalone'), ('a0044', 'standalone'))
SETS = ('C', 'U', 'O', 'O_near', 'O_far')
ROCK_MASK_VALUE = 1                  # region mask of frame.npz: 1 = rock
GAUSS_MODE, GAUSS_TRUNCATE = 'reflect', 4.0      # defaults of scipy.ndimage.gaussian_filter


def luminance(png):
    with Image.open(png) as im:
        assert im.mode == 'RGB', (str(png), im.mode)
        a = np.asarray(im)
    assert a.dtype == np.uint8 and a.shape == SHAPE + (3,), (str(png), a.dtype, a.shape)
    a = a.astype(np.float64)
    return REC709[0] * a[..., 0] + REC709[1] * a[..., 1] + REC709[2] * a[..., 2]


def ridge_reading(lum):
    """R = maximum over the three scales of the positive part of sigma^2 x (larger Hessian eigenvalue); reading = maximum of R in
    the 3 x 3 neighbourhood."""
    best = np.zeros(lum.shape, np.float64)
    for s in SIGMAS:
        hyy = ndi.gaussian_filter(lum, s, order=(2, 0), mode=GAUSS_MODE, truncate=GAUSS_TRUNCATE)   # second derivative along axis 0 (rows)
        hxx = ndi.gaussian_filter(lum, s, order=(0, 2), mode=GAUSS_MODE, truncate=GAUSS_TRUNCATE)   # second derivative along axis 1 (columns)
        hxy = ndi.gaussian_filter(lum, s, order=(1, 1), mode=GAUSS_MODE, truncate=GAUSS_TRUNCATE)
        big = .5 * (hxx + hyy) + np.sqrt(.25 * (hxx - hyy) ** 2 + hxy ** 2)       # larger eigenvalue of the symmetric 2 x 2 matrix
        np.maximum(best, np.maximum(s * s * big, 0.), out=best)
    return ndi.maximum_filter(best, size=3, mode='reflect')


def auc(values, bg_sorted):
    """P(reading of a set pixel > reading of a background pixel) + 0.5 P(equal)."""
    lo = np.searchsorted(bg_sorted, values, 'left').astype(np.int64)
    hi = np.searchsorted(bg_sorted, values, 'right').astype(np.int64)
    return float((lo + hi).sum() / (2. * values.size * bg_sorted.size))


def med(a):
    return float(np.median(a)) if len(a) else None


def summary(vals):
    vals = [v for v in vals if v is not None]
    return dict(n_images=len(vals), median=med(vals), min=float(min(vals)) if vals else None, max=float(max(vals)) if vals else None)


def one_frame(frames, key, row):
    lab = fio.label_dir(frames, key)
    png = fio.point_dir(frames, key) / 'render' / 'frame.png'
    npz = fio.point_dir(frames, key) / 'render' / 'frame.npz'
    arr = {n: np.load(lab / (n + '.npy')) for n in ('target', 'u_line', 'occluded_line', 'opening_class', 'valid')}
    for n, a in arr.items():
        assert a.shape == SHAPE, (key, n, a.shape)
    target, u_line, occ, valid = (arr[n].astype(bool) for n in ('target', 'u_line', 'occluded_line', 'valid'))
    opening = arr['opening_class'] > 0
    with np.load(npz) as z:
        mask = z['mask']
    assert mask.shape == SHAPE, (key, mask.shape)
    rock = mask == ROCK_MASK_VALUE

    lum = luminance(png)
    reading = ridge_reading(lum)

    # distance d: visible evidence M = fracture-mouth pixels, final centreline and unclear line; Euclidean distance of the
    # occluded-line pixels to the nearest pixel of M
    M = opening | target | u_line
    d_M = ndi.distance_transform_edt(~M) if M.any() else np.full(SHAPE, np.inf)
    d_occ = d_M[occ]
    near = occ & (d_M <= NEAR_PX)
    far = occ & (d_M > FAR_PX)
    mid = occ & ~near & ~far

    # background: valid pixels more than 20 px from M and from the occluded line; at most 200,000 of them at a fixed stride in
    # row-major order
    lines = M | occ
    d_L = ndi.distance_transform_edt(~lines) if lines.any() else np.full(SHAPE, np.inf)
    bg_idx = np.flatnonzero((valid & (d_L > BG_DIST_PX)).ravel())
    step = max(1, -(-bg_idx.size // BG_MAX))           # ceil(n / 200000)
    bg_take = bg_idx[::step]
    assert bg_take.size <= BG_MAX
    bg = np.sort(reading.ravel()[bg_take])

    sets = dict(C=target, U=u_line & rock, O=occ, O_near=near, O_far=far)
    out = dict(key=key, sid=row['sid'], pid=row['pid'], view=fio.view_of(row), pixel_mm=float(row['pixel_mm']),
               luminance_min=float(lum.min()), luminance_max=float(lum.max()),
               n=dict(C=int(target.sum()), U_all=int(u_line.sum()), U=int(sets['U'].sum()), U_off_rock=int((u_line & ~rock).sum()),
                      O=int(occ.sum()), O_off_rock=int((occ & ~rock).sum()), O_near=int(near.sum()), O_mid=int(mid.sum()),
                      O_far=int(far.sum()), M=int(M.sum()), opening=int(opening.sum()), valid=int(valid.sum()), rock=int(rock.sum()),
                      O_and_U=int((occ & u_line).sum()), O_and_C=int((occ & target).sum()), C_not_valid=int((target & ~valid).sum())),
               a2=dict(share_near=float(near.sum() / occ.sum()) if occ.any() else None,
                       share_mid=float(mid.sum() / occ.sum()) if occ.any() else None,
                       share_far=float(far.sum() / occ.sum()) if occ.any() else None,
                       d_median_px=med(d_occ), d_median_mm=(med(d_occ) * float(row['pixel_mm'])) if d_occ.size else None),
               background=dict(candidates=int(bg_idx.size), step=int(step), sampled=int(bg_take.size),
                               reading_median=med(bg), reading_p90=float(np.quantile(bg, .9)) if bg.size else None,
                               reading_zero_share=float((bg == 0).mean()) if bg.size else None),
               auc={}, reading_median={}, skipped={})
    for name in SETS:
        v = reading[sets[name]]
        if v.size < MIN_SET or bg.size < MIN_SET:
            out['auc'][name] = None
            out['reading_median'][name] = med(v)
            out['skipped'][name] = 'fewer than 30 pixels in the set' if v.size < MIN_SET else 'fewer than 30 background pixels'
            continue
        out['auc'][name] = auc(v, bg)
        out['reading_median'][name] = med(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', required=True, help='folder with one sub-folder per frame (a0044 ...)')
    ap.add_argument('--index', default=str(fio.INDEX))
    ap.add_argument('--out', default=str(fio.RECORDS / 'ridge_check.json'))
    args = ap.parse_args(argv)
    t0 = time.time()
    idx = fio.index(args.index)
    keys = sorted(k for k, v in idx.items() if v['block'] == 'C')
    assert len(keys) == 50, len(keys)
    rows = []
    for k in keys:
        rows.append(one_frame(args.frames, k, idx[k]))
        r = rows[-1]
        print(k, r['view'], ' '.join('%s=%s' % (s, 'skip' if r['auc'][s] is None else '%.3f' % r['auc'][s]) for s in SETS), flush=True)

    views = []
    for rep, kind in VIEW_REPS:
        v = fio.view_of(idx[rep])
        mine = [r for r in rows if r['view'] == v]
        assert len(mine) == 10, (rep, v, len(mine))
        views.append(dict(view=v, representative=rep, outcrop=kind, pixel_mm=float(idx[rep]['pixel_mm']), frames=[r['key'] for r in mine],
                          auc={s: dict(summary([r['auc'][s] for r in mine]), skipped=sum(r['auc'][s] is None for r in mine)) for s in SETS},
                          a2_representative=next(r['a2'] for r in mine if r['key'] == rep),
                          n_representative=next(r['n'] for r in mine if r['key'] == rep)))
    assert sum(len(v['frames']) for v in views) == 50 and len({k for v in views for k in v['frames']}) == 50

    overall = {}
    for s in SETS:
        sm = summary([r['auc'][s] for r in rows])
        sm['skipped'] = sum(r['auc'][s] is None for r in rows)
        overall[s] = sm

    res = dict(
        description='Independent ridge filter (multi-scale Hessian, dark lines, no geometry) on the 50 images of the multi-illumination aerial '
                    'set: AUC of the filter reading on the pixels of each label layer against background rock, per image, per viewpoint and '
                    'overall (median, minimum and maximum over the images in which the set has at least 30 pixels).',
        paper_items=['main text Section 4.1, last paragraph', 'supplement S4: an independent ridge filter, AUC table'],
        script='reproduce/paper1/analysis/ridge_check.py',
        units=dict(auc='probability that the reading of a set pixel exceeds that of a background pixel (ties count one half)',
                   d='image distance in pixels to the nearest pixel of the visible evidence', pixel_mm='nominal pixel size, mm'),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, pillow=Image.__version__),
        detector=dict(luminance='8-bit sRGB values of frame.png, Rec. 709 weights %s' % (REC709,), sigmas_px=SIGMAS,
                      response='maximum over sigma of max(sigma^2 x larger Hessian eigenvalue, 0)', reading='maximum of the response in the 3 x 3 neighbourhood',
                      gaussian_mode=GAUSS_MODE, gaussian_truncate=GAUSS_TRUNCATE),
        sets=dict(C='final centreline (target)', U='unclear line on rock (u_line and region mask = 1)', O='occluded line',
                  O_near='occluded line with d <= 2 px', O_far='occluded line with d > 10 px',
                  d='Euclidean distance to the nearest pixel of M = (opening_class > 0) or target or u_line',
                  B='background: valid pixels more than 20 px from M and from the occluded line, at most 200000 at a fixed stride'),
        notes=['The filter is set by hand; it is not a trained detector and does not stand for a human interpreter. It answers only whether '
               'a dark linear feature is present at the labelled position.',
               'The reading is an absolute brightness curvature: in images partly in shadow it is affected by the division into lit and '
               'shadowed areas.',
               'Background stride = ceil(candidates / 200000) in row-major order, see frames[*].background.step. O, O_near and O_far keep the '
               'pixels that coincide with the unclear line (n.O_and_U) and are not restricted to rock (n.O_off_rock).'],
        frames_total=len(rows), min_set_pixels=MIN_SET, overall=overall, per_view=views, frames=rows)
    fio.write_json(args.out, res)
    print(json.dumps(overall, indent=1))
    for v in views:
        print(v['representative'], v['outcrop'], v['view'], ' '.join('%s=%s' % (s, 'NA' if v['auc'][s]['median'] is None else '%.3f' % v['auc'][s]['median']) for s in SETS))
    print('runtime_s', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()

