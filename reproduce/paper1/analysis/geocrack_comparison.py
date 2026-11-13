"""Comparison with real photographs (main text Section 4.3, first paragraph; supplement S4, "Comparison with real photographs").

The dark band along fracture lines is measured with one code path in real photographs and in the generator's frames:

  real       49 photographs of the GeoCrack dataset with their edge masks; the line is the skeleton of the manual annotation mask
  synthetic  the 50 frames of the multi-illumination aerial set (block C of records/frames_index.json; the "multi-light aerial
             set" of tables.py) and, separately, the two sunlit close-range frames at 2 mm/px (a0018 on a flat outcrop, a0020 on
             a standalone outcrop); the line is the target layer (target.npy, the final centreline of the paper)

    python reproduce/paper1/analysis/geocrack_comparison.py --geocrack GEOCRACK --frames FRAMES --out geocrack_comparison.json

GEOCRACK  folder of the dataset with its two folders "Original Images" and "Edge Masks" (other names: --image-dir, --mask-dir),
          each holding one PNG of the same name per photograph. README_real_data.md says where the dataset is published.
FRAMES    folder with one sub-folder per frame, as described in frames_io.py: a0018, a0020 and the 50 frames of block C
          (column "block" of reproduce/paper1/frames.csv, which also holds the options that generate each frame; the stored
          record lists them under "synthetic").

Either input can be left out:
  --geocrack only   the photographs are measured; the synthetic side of the comparison (histograms, percentiles, per-frame
                    background values) is taken from the stored record (--record, default records/round3/geocrack_comparison.json).
                    The overlapping coefficient with equal weight per image is then left out.
  --frames only     the two synthetic groups are measured; nothing is compared.

--images ID ...     photographs to use (default: the 49 of the paper, listed in IMAGES)
--reference RECORD  lists the differences between the result and a stored record, for the parts that this run has measured
                    (relative tolerance 1e-9, --tolerance)
--check-ruler       also recomputes six rows of records/fine_width_formal.json with the parameters of that record (needs the
                    frames a0001, a0002, a0017, a0018, a0020 and a0061)

The output has the layout of records/round3/geocrack_comparison.json. No file of the dataset is written anywhere: the record holds
statistics only. Needs Python 3.8+, NumPy, SciPy and Pillow (the labels environment); about 3 minutes and 2 GB of memory for the
whole set. No random numbers are used.

Measurement (one function, measure_image, for real and synthetic images)
- Line: real, the Zhang-Suen skeleton of the annotation mask (mask converted to 8-bit grey, values > 0); the thinning is the
  algorithm of the label program (skeletonize in frgb/chain/evaluate_lines.py), evaluated on the foreground pixels only so that
  50-megapixel images can be processed, and compared with that function on a 512 x 512 crop of every mask. Synthetic, target.npy.
- Measurement points: every 8th line pixel in row-major order (not equally spaced along the lines); a point with fewer than 3
  line pixels within 6 px is skipped.
- Brightness: Rec. 709 weights (0.2126, 0.7152, 0.0722) on the 8-bit sRGB values.
- Profile: along the local normal (perpendicular to the principal direction of the line pixels within 6 px), +-63 px in steps of
  0.25 px, bilinear interpolation, nearest pixel outside the image. Surrounding brightness = median of the part with
  |s| >= 47.25 px (0.75 R); centre brightness = minimum over |s| <= 6.3 px (0.1 R); brightness difference = surrounding - centre.
  A difference below 2 grey levels is recorded as "no dark band". Otherwise the dark-band width is the length of the continuous
  interval around the centre minimum in which the brightness is below the midpoint of the two, with the crossings interpolated
  linearly between neighbouring profile samples.
- Relative contrast = brightness difference / surrounding brightness.
- Spacing of adjacent lines: from every measurement point the two normal directions are followed (from 8 px, in steps of 0.5 px,
  up to 512 px or the image border) and the first distance at which the 3 x 3-dilated line raster is met is recorded; the spacing
  of an image is the median of all its hits, and an image with fewer than 30 hits has no spacing (its points do not enter the
  distribution of width / spacing). The relative width of the paper is dark-band width / spacing.
- Fine-scale brightness variation of the background: standard deviation of the brightness minus its Gaussian smoothing
  (sigma = 2 px), one number per image. Real: all pixels more than 20 px from the annotation mask (the dataset has no rock mask).
  Synthetic: valid pixels more than 20 px from the target, unclear (U) and occluded lines and from the opening-class pixels.
- Comparable subset: the measurement points with a dark-band width of at least 3 px.

Statistics
- Distributions are pooled over the measurement points (every point has the same weight). The overlapping coefficient is the sum
  over the bins of the smaller of the two normalised histograms; values outside the range go into the end bins. The coefficient
  with every image given the same total weight is recorded as well.
- Real against synthetic: all real points against all synthetic points (all_points); the real comparable subset against all
  synthetic points (comparable_subset); both sides restricted to their comparable subsets (both_sides_comparable).
- Points without a dark band have no width: the distributions of width and of width / spacing hold the points with a dark band;
  the other quantities hold all points.
- Self-overlap: overlapping coefficient of the two halves of one group (even and odd point index), a reference for the largest
  coefficient that the sample sizes allow.
- Coverage: where the 5th, 25th, 75th and 95th percentile of a synthetic group lie in the real distribution, and the share of the
  real points inside, below and above the synthetic 5th-95th percentile range.

The profile length is fixed in pixels here, and the brightness is Rec. 709. The dark-band width table of the paper
(records/fine_width_formal.json) uses a half length of 0.40 m, a centre part of max(2 px, 0.04 m), the mean of R, G and B and every
line pixel; widths of the two rulers are not comparable with each other."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames_io as fio  # noqa: E402  (index of the paper's frames and the folder of a frame)

Image.MAX_IMAGE_PIXELS = None      # the photographs have up to 50 megapixels
PAPER1 = HERE.parent
REPO = HERE.parents[2]
RECORD = PAPER1 / 'records' / 'round3' / 'geocrack_comparison.json'
FINE_WIDTH = PAPER1 / 'records' / 'fine_width_formal.json'

IMAGES = ('0V5A3666', '0V5A3736', '0V5A4035', 'CZBJ2377', 'DJI_0011', 'DJI_0022', 'DJI_0034', 'DJI_0040', 'DJI_0050', 'DJI_0051',
          'DJI_0052', 'DJI_0059', 'DJI_0065', 'DJI_0067', 'DJI_0076', 'DJI_0081', 'DJI_0107', 'DJI_0115', 'DJI_0135', 'DJI_0138',
          'DJI_0144', 'DJI_0183', 'DJI_0194', 'DJI_0225', 'DJI_0226', 'DJI_0234', 'DJI_0275', 'DJI_0276', 'DJI_0284', 'DJI_0302',
          'DJI_0366', 'DJI_0404', 'DJI_0413', 'DJI_0420', 'DJI_0443', 'DJI_0489', 'DJI_0491', 'DJI_0552', 'DJI_0796', 'DJI_0948',
          'DJI_0963', 'DJI_0997', 'DSC00659', 'DSC00888', 'DSC_5049', 'DSC_5053', 'DSC_5065', 'DSC_5082', 'TRRW7180')
IMAGE_DIR, MASK_DIR = 'Original Images', 'Edge Masks'
AERIAL_BLOCK = 'C'
CLOSE_RANGE = ('a0018', 'a0020')
RULER_ROWS = ('b_b055_a26120005_c2', 'b_b055_a26120005_c3', 'b_b055_a26120005_c0', 'b_b055_a26130005_c1', 'b_b055_n26120003_c1',
              'b_b055_n26130007_c0')

STRIDE = 8
R_PROF = 63
STEP = .25
CENTRE_REACH = .1 * R_PROF
NO_BAND = 2.
COMP_MIN_WIDTH = 3.
BG_DIST = 20.
SIGMA = 2.
S_MIN, S_MAX, S_STEP, MIN_HITS = 8., 512., .5, 30

# quantity -> name, unit, histogram range and bin width
QUANTITIES = dict(
    surrounding_brightness=dict(name='surrounding brightness', unit='grey level', range=[0., 256.], bin_width=8.),
    centre_brightness=dict(name='centre brightness', unit='grey level', range=[0., 256.], bin_width=8.),
    brightness_difference=dict(name='brightness difference (surrounding - centre)', unit='grey level', range=[-64., 256.], bin_width=4.),
    relative_contrast=dict(name='relative contrast (brightness difference / surrounding brightness)', unit='dimensionless',
                           range=[-.25, 1.], bin_width=.05),
    dark_band_width_px=dict(name='dark-band width', unit='px', range=[0., 126.], bin_width=1.),
    log10_width_over_spacing=dict(name='log10 of the relative width (dark-band width / median spacing of adjacent lines of the image)',
                                  unit='dex', range=[-3., 1.], bin_width=.1),
    background_variation=dict(name='fine-scale brightness variation of the background, one value per image', unit='grey level',
                              range=[0., 32.], bin_width=1.))
POINT_Q = ('surrounding_brightness', 'centre_brightness', 'brightness_difference', 'relative_contrast', 'dark_band_width_px',
           'log10_width_over_spacing')
SYNTHETIC = ('synthetic_aerial', 'synthetic_close_range')
GROUPS = dict(real='GeoCrack photographs, skeleton of the manual annotation mask',
              synthetic_aerial='frames of the multi-illumination aerial set (block C of the frame list), target layer',
              synthetic_close_range='sunlit close-range frames at 2 mm/px (a0018, a0020), target layer')
RESULTS = ('groups', 'summary', 'histograms', 'self_overlap', 'overlap', 'coverage', 'per_image')
DATASET = dict(
    name='GeoCrack', version='1.0', doi='10.7910/DVN/E4OXHQ', licence='CC0 1.0',
    references=[
        'Ansari, Mohammed Yaqoob, 2024. GeoCrack: A high-resolution dataset of fracture edges in geological outcrops (version 1.0) '
        '[data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/E4OXHQ',
        'Yaqoob, M., Ishaq, M., Ansari, M.Y., Konagandla, V.R.S., Al Tamimi, T., Tavani, S., Corradetti, A., Seers, T.D., 2024. '
        'GeoCrack: A high-resolution dataset for segmentation of fracture edges in geological outcrops. Scientific Data 11, 1318. '
        'https://doi.org/10.1038/s41597-024-04107-0'],
    used='49 photographs with their edge masks (listed in per_image.real); no file of the dataset is redistributed, this record holds '
         'statistics only')
NOTES = [
    'The photographs carry no pixel size: real and synthetic images are compared in pixels and in dimensionless quantities only.',
    'Annotation: mask converted to 8-bit grey, values > 0 (the weak values of anti-aliased edges included); the line is its '
    'Zhang-Suen skeleton, which need not lie at the centre of the dark band.',
    'Measurement points: every 8th line pixel in row-major order, not equally spaced along the lines; points with fewer than 3 line '
    'pixels within 6 px are skipped.',
    'Profile: half length 63 px whatever the pixel size, step 0.25 px; surrounding part |s| >= 47.25 px, centre part |s| <= 6.3 px; '
    'brightness = Rec. 709-weighted 8-bit sRGB values. The widths are not readings of the ruler of records/fine_width_formal.json '
    '(half length 0.40 m, mean of R, G and B, every line pixel).',
    'The two synthetic groups are reported separately and never pooled. The 50 frames of the aerial group come from 5 viewpoints, so '
    'their measurement points are not independent. The overcast frames of the close-range viewpoints (a0019, a0021) are not used.',
    'Spacing of adjacent lines: first hit of the 3 x 3-dilated line raster along the two normal directions (8 to 512 px, step '
    '0.5 px); directions without a hit are not counted, so the spacing is biased low; an image with fewer than 30 hits has no spacing. '
    'summary.log10_width_over_spacing gives the quantiles of the logarithm and, under "ratio", those of the ratio itself (the relative '
    'width of the paper).',
    'Background: real, all pixels more than 20 px from the annotation mask (vegetation, sky, shadow and unmapped fractures may be '
    'included); synthetic, valid pixels more than 20 px from the target, unclear (U) and occluded lines and the opening-class pixels.',
    'Histograms are counts of measurement points (background_variation: of images); values outside the range are counted in the end '
    'bins. overlap = sum over the bins of min(count_a / total_a, count_b / total_b): the overlapping coefficients of the pooled points '
    'can be recomputed from the histograms; the quantiles (median, iqr, p5, p95), the coverage and overlap_image_equal_weight cannot.',
    'Comparable subset: measurement points with a dark-band width of at least 3 px. overlap.*.*.comparable_subset compares the real '
    'subset with all synthetic points; both_sides_comparable restricts both sides.',
    'Points without a dark band (brightness difference below 2 grey levels) have no width. Along a manual annotation they include '
    'offsets of the annotation, closed fractures and fills brighter than the surrounding rock, which this measurement cannot tell apart.',
    'checks.skeleton: the thinning used here equals skeletonize of the label program on a 512 x 512 crop of every mask. checks.ruler: '
    'band_measure with the parameters of records/fine_width_formal.json reproduces six rows of that record exactly.']


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=True, indent=1, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


# ---------------------------------------------------------------- inputs
def frame_image(frames, key):
    """8-bit sRGB image of a frame, H x W x 3"""
    with Image.open(fio.point_dir(frames, key) / 'render' / 'frame.png') as im:
        return np.asarray(im.convert('RGB'))


def frame_layer(frames, key, name):
    """one label layer of a frame (target, u_line, occluded_line, opening_class, valid)"""
    return np.load(fio.point_dir(frames, key) / 'trace_labels' / (name + '.npy'))


def geocrack_pair(root, image_id, image_dir=IMAGE_DIR, mask_dir=MASK_DIR):
    """paths of a photograph and of its edge mask"""
    pair = [Path(root) / d / (image_id + '.png') for d in (image_dir, mask_dir)]
    for p in pair:
        if not p.is_file():
            raise SystemExit('not found: %s (folders of the photographs and of the masks: --image-dir, --mask-dir)' % p)
    return pair


# ---------------------------------------------------------------- skeleton
def zs_sparse(mask):
    """Zhang-Suen thinning in two sub-iterations, evaluated on the foreground pixels only (same simultaneous deletion per phase
    as skeletonize in frgb/chain/evaluate_lines.py)"""
    p = np.pad(np.asarray(mask, bool), 1).astype(np.uint8)
    wp = p.shape[1]
    flat = p.reshape(-1)
    idx = np.flatnonzero(flat)
    offs = (-wp, -wp + 1, 1, wp + 1, wp, wp - 1, -1, -wp - 1)
    while True:
        deleted = 0
        for phase in (0, 1):
            n = [flat[idx + o] for o in offs]
            count = n[0] + n[1] + n[2] + n[3] + n[4] + n[5] + n[6] + n[7]
            trans = np.zeros(len(idx), np.uint8)
            for i in range(8):
                trans += (1 - n[i]) & n[(i + 1) % 8]
            if phase == 0:
                triples = (n[0] & n[2] & n[4]) | (n[2] & n[4] & n[6])
            else:
                triples = (n[0] & n[2] & n[6]) | (n[0] & n[4] & n[6])
            remove = (count >= 2) & (count <= 6) & (trans == 1) & (triples == 0)
            deleted += int(remove.sum())
            flat[idx[remove]] = 0
            idx = idx[~remove]
        if not deleted:
            return p[1:-1, 1:-1].astype(bool)


def skeleton_crop_check(mask):
    """True when zs_sparse equals the label program's skeletonize on the 512 x 512 crop around the middle foreground pixel"""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from frgb.chain.evaluate_lines import skeletonize
    fy, fx = np.unravel_index(np.flatnonzero(mask)[int(mask.sum()) // 2], mask.shape)
    y0 = int(min(max(0, fy - 256), max(0, mask.shape[0] - 512)))
    x0 = int(min(max(0, fx - 256), max(0, mask.shape[1] - 512)))
    crop = mask[y0:y0 + 512, x0:x0 + 512]
    return bool((zs_sparse(crop) == skeletonize(crop)).all())


# ---------------------------------------------------------------- measurement (one code path for real and synthetic images)
def luminance709(rgb):
    y = rgb[..., 0].astype(np.float64)
    y *= .2126
    t = rgb[..., 1].astype(np.float64)
    t *= .7152
    y += t
    t = rgb[..., 2].astype(np.float64)
    t *= .0722
    y += t
    return y


def band_measure(Y, pts, R, centre_reach, step, stride, chunk=4000):
    """Brightness profile across the line at every stride-th line pixel (index i of np.argwhere order) that has at least 3 line
    pixels within 6 px: surrounding brightness (ring), centre brightness, dark-band width (nan without a dark band), whether the
    dark band reaches an end of the profile, and the unit normal (ny, nx)."""
    tree = cKDTree(pts)
    s = np.arange(-R, R + step / 2., step)
    ringm = np.abs(s) >= .75 * R
    inner = np.abs(s) <= centre_reach
    inner_idx = np.flatnonzero(inner)
    I, NY, NX = [], [], []
    for i in range(0, len(pts), stride):
        y, x = pts[i]
        near = pts[tree.query_ball_point((y, x), 6.)]
        if len(near) < 3:
            continue
        c = near - near.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        I.append(i)
        NY.append(-vt[0][1])
        NX.append(vt[0][0])
    I = np.array(I, int)
    ny, nx = np.array(NY, float), np.array(NX, float)
    n = len(I)
    ring, centre = np.empty(n), np.empty(n)
    width = np.full(n, np.nan)
    end = np.zeros(n, bool)
    last = len(s) - 1
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        yy = pts[I[a:b], 0][:, None] + s[None, :] * ny[a:b, None]
        xx = pts[I[a:b], 1][:, None] + s[None, :] * nx[a:b, None]
        P = ndi.map_coordinates(Y, [yy.ravel(), xx.ravel()], order=1, mode='nearest').reshape(b - a, len(s))
        rg = np.median(P[:, ringm], axis=1)
        i0s = inner_idx[np.argmin(P[:, inner], axis=1)]
        for k in range(b - a):
            prof = P[k]
            r_ = float(rg[k])
            i0 = int(i0s[k])
            ring[a + k] = r_
            centre[a + k] = float(prof[i0])
            depth = r_ - float(prof[i0])
            if depth < NO_BAND:
                continue
            half = r_ - depth / 2.
            lo = i0
            while lo > 0 and prof[lo - 1] < half:
                lo -= 1
            hi = i0
            while hi < last and prof[hi + 1] < half:
                hi += 1
            s_lo = s[lo] - step * (half - prof[lo]) / (prof[lo - 1] - prof[lo]) if lo > 0 else s[0]
            s_hi = s[hi] + step * (half - prof[hi]) / (prof[hi + 1] - prof[hi]) if hi < last else s[-1]
            width[a + k] = float(s_hi - s_lo)
            end[a + k] = lo == 0 or hi == last
    return dict(i=I, ny=ny, nx=nx, ring=ring, centre=centre, width=width, end=end)


def spacing_hits(line, y, x, ny, nx):
    """first line pixel (3 x 3-dilated raster) met along +normal and -normal, from S_MIN to S_MAX; nan when none"""
    H, W = line.shape
    dil = ndi.binary_dilation(line, structure=np.ones((3, 3), bool))
    out = np.full((len(y), 2), np.nan)
    for j, sign in enumerate((1., -1.)):
        active = np.arange(len(y))
        s = S_MIN
        while len(active) and s <= S_MAX:
            yy = np.rint(y[active] + sign * s * ny[active]).astype(np.int64)
            xx = np.rint(x[active] + sign * s * nx[active]).astype(np.int64)
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            hit = np.zeros(len(active), bool)
            hit[inside] = dil[yy[inside], xx[inside]]
            out[active[hit], j] = s
            active = active[inside & ~hit]
            s += S_STEP
    return out


def far_mask(lines, dist, strip=2048):
    """True where the Euclidean distance to the nearest True pixel of `lines` exceeds dist (exact; strips with a halo)"""
    H = lines.shape[0]
    halo = int(math.ceil(dist)) + 1
    out = np.empty(lines.shape, bool)
    for r0 in range(0, H, strip):
        r1 = min(H, r0 + strip)
        a0, a1 = max(0, r0 - halo), min(H, r1 + halo)
        sub = lines[a0:a1]
        if not sub.any():
            out[r0:r1] = True
        else:
            out[r0:r1] = ndi.distance_transform_edt(~sub)[r0 - a0:r1 - a0] > dist
    return out


def texture_std(Y, rock, lines_any):
    far = far_mask(lines_any, BG_DIST)
    if rock is not None:
        far &= rock
    n = int(far.sum())
    if n < 1000:
        return None, n
    g = ndi.gaussian_filter(Y, SIGMA, mode='nearest')
    np.subtract(Y, g, out=g)
    return float(g[far].std()), n


def measure_image(Y, line, rock, lines_any):
    """dark band at every 8th line pixel, spacing along the normals, fine-scale brightness variation of the background"""
    pts = np.argwhere(line)
    m = band_measure(Y, pts, R_PROF, CENTRE_REACH, STEP, STRIDE)
    y, x = pts[m['i'], 0].astype(float), pts[m['i'], 1].astype(float)
    hits = spacing_hits(line, y, x, m['ny'], m['nx'])
    tex, nbg = texture_std(Y, rock, lines_any)
    found = hits[np.isfinite(hits)]
    spacing = float(np.median(found)) if len(found) >= MIN_HITS else None
    depth = m['ring'] - m['centre']
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(m['ring'] > 0, depth / m['ring'], np.nan)
    return dict(line_px=int(len(pts)), ring=m['ring'], centre=m['centre'], depth=depth, rel=rel, width=m['width'], end=m['end'],
                spacing_px=spacing, hits=int(len(found)), hit_share=float(len(found) / (2 * len(y))) if len(y) else None,
                texture=tex, bg_px=nbg)


def measure_frame(frames, key):
    """measure_image on a frame: the line is the target layer; valid pixels and all line and opening-class pixels bound the background"""
    line = frame_layer(frames, key, 'target').astype(bool)
    any_line = line | frame_layer(frames, key, 'u_line').astype(bool) | frame_layer(frames, key, 'occluded_line').astype(bool) \
        | (frame_layer(frames, key, 'opening_class') > 0)
    valid = frame_layer(frames, key, 'valid').astype(bool)
    rgb = frame_image(frames, key)
    if rgb.shape[:2] != line.shape:
        raise SystemExit('frame %s: image and label layers differ in size' % key)
    return int(valid.sum()), measure_image(luminance709(rgb), line, valid, any_line)


# ---------------------------------------------------------------- statistics
def q3(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=None, iqr=None, p5=None, p95=None)
    q = np.quantile(v, [.05, .25, .5, .75, .95])
    return dict(n=int(len(v)), median=float(q[2]), iqr=[float(q[1]), float(q[3])], p5=float(q[0]), p95=float(q[4]))


def edges(name):
    (lo, hi), w = QUANTITIES[name]['range'], QUANTITIES[name]['bin_width']
    nb = int(round((hi - lo) / w))
    return lo + w * np.arange(nb + 1)


def hist(name, v, weights=None):
    e = edges(name)
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    v = np.clip(v[ok], e[0], np.nextafter(e[-1], e[0]))
    h, _ = np.histogram(v, e, weights=None if weights is None else np.asarray(weights, float)[ok])
    return h.astype(float)


def ovl(h1, h2):
    """overlapping coefficient of two histograms"""
    if h1.sum() <= 0 or h2.sum() <= 0:
        return None
    return float(np.minimum(h1 / h1.sum(), h2 / h2.sum()).sum())


def image_weights(img):
    """the points of every image share a total weight of 1"""
    w = np.zeros(len(img))
    if len(img):
        cnt = np.bincount(img)
        w = 1. / cnt[img]
    return w


class Group:
    """measurement points of one group of images"""

    def __init__(self, name):
        self.name, self.rows, self.parts = name, [], []

    def add(self, row, m):
        k = len(self.rows)
        band = np.isfinite(m['width'])
        comp = band & (m['width'] >= COMP_MIN_WIDTH)
        n = len(m['ring'])
        wos = m['width'] / m['spacing_px'] if m['spacing_px'] else np.full(n, np.nan)
        row.update(line_px=m['line_px'], points=n, band_points=int(band.sum()), no_band_share=float((~band).sum() / n) if n else None,
                   comparable_points=int(comp.sum()), comparable_share=float(comp.sum() / n) if n else None,
                   end_reached=int(m['end'].sum()),
                   median_surrounding_brightness=q3(m['ring'])['median'], median_centre_brightness=q3(m['centre'])['median'],
                   median_brightness_difference=q3(m['depth'])['median'], median_relative_contrast=q3(m['rel'])['median'],
                   median_dark_band_width_px=q3(m['width'])['median'],
                   median_dark_band_width_px_comparable=q3(m['width'][comp])['median'],
                   spacing_px=m['spacing_px'], spacing_hits=m['hits'], spacing_hit_share=m['hit_share'],
                   median_width_over_spacing=q3(wos)['median'], background_variation=m['texture'], background_px=m['bg_px'])
        self.rows.append(row)
        with np.errstate(divide='ignore', invalid='ignore'):
            lw = np.log10(wos)
        self.parts.append(dict(img=np.full(n, k, int), surrounding_brightness=m['ring'], centre_brightness=m['centre'],
                               brightness_difference=m['depth'], relative_contrast=m['rel'], dark_band_width_px=m['width'],
                               log10_width_over_spacing=lw, end=m['end']))

    def close(self):
        self.d = {k: np.concatenate([p[k] for p in self.parts]) if self.parts else np.zeros(0) for k in ('img', 'end') + POINT_Q}
        self.d['img'] = self.d['img'].astype(int)
        self.band = np.isfinite(self.d['dark_band_width_px'])
        self.comp = self.band & (self.d['dark_band_width_px'] >= COMP_MIN_WIDTH)
        self.parts = None

    def values(self, q, subset):
        sel = self.comp if subset == 'comparable' else np.ones(len(self.band), bool)
        v = self.d[q]
        ok = sel & np.isfinite(v)
        return v[ok], self.d['img'][ok]

    def count(self, q, subset):
        return int(len(self.values(q, subset)[0]))

    def hist(self, q, subset):
        return hist(q, self.values(q, subset)[0])

    def hist_images(self, q, subset):
        v, img = self.values(q, subset)
        return hist(q, v, image_weights(img)) if len(v) else None

    def percentiles(self, q):
        """5th, 25th, 75th and 95th percentile of all points"""
        return np.quantile(self.values(q, 'all')[0], [.05, .25, .75, .95])

    def variation(self):
        return [r['background_variation'] for r in self.rows if r['background_variation'] is not None]


class StoredGroup:
    """a synthetic group taken from a stored record instead of from frames: its histograms, percentiles and per-image values"""

    def __init__(self, name, record):
        self.name, self.rec, self.rows = name, record, record['per_image'][name]
        for q in POINT_Q + ('background_variation',):
            if not np.array_equal(np.array(record['histograms'][q]['edges'], float), edges(q)):
                raise SystemExit('the stored record has other histogram bins for %s' % q)

    def count(self, q, subset):
        return int(self.rec['summary'][q][self.name][subset]['n'])

    def hist(self, q, subset):
        return np.array(self.rec['histograms'][q][self.name][subset], float)

    def percentiles(self, q):
        s = self.rec['summary'][q][self.name]['all']
        return np.array([s['p5'], s['iqr'][0], s['iqr'][1], s['p95']], float)

    def variation(self):
        return [r['background_variation'] for r in self.rows if r['background_variation'] is not None]


def compare(real, synth, q, rs, ss):
    out = dict(real_subset=rs, synthetic_subset=ss, n_real=real.count(q, rs), n_synthetic=synth.count(q, ss),
               overlap=ovl(real.hist(q, rs), synth.hist(q, ss)))
    if isinstance(synth, Group):        # needs the measurement points of both sides
        wa, wb = real.hist_images(q, rs), synth.hist_images(q, ss)
        out['overlap_image_equal_weight'] = None if wa is None or wb is None else ovl(wa, wb)
    return out


def coverage(real, synth, q, rs):
    a, _ = real.values(q, rs)
    if len(a) < 30 or synth.count(q, 'all') < 30:
        return None
    qs = synth.percentiles(q)
    rank = [float((a <= v).mean()) for v in qs]
    return dict(real_subset=rs, synthetic_p5_p25_p75_p95=[float(v) for v in qs], real_percentile_rank_of_those=rank,
                real_share_inside_synthetic_p5_p95=float(((a >= qs[0]) & (a <= qs[3])).mean()),
                real_share_below_synthetic_p5=float((a < qs[0]).mean()), real_share_above_synthetic_p95=float((a > qs[3]).mean()))


def self_overlap(g, q, subset):
    v, _ = g.values(q, subset)
    if len(v) < 60:
        return None
    return dict(n=int(len(v)), overlap_even_odd=ovl(hist(q, v[0::2]), hist(q, v[1::2])))


def group_counts(g):
    n = len(g.band)
    nb = [r['no_band_share'] for r in g.rows if r['no_band_share'] is not None]
    cs = [r['comparable_share'] for r in g.rows if r['comparable_share'] is not None]
    sp = [r['spacing_px'] for r in g.rows if r['spacing_px'] is not None]
    return dict(
        description=GROUPS[g.name], images=len(g.rows), points=int(n), band_points=int(g.band.sum()),
        no_band_points=int((~g.band).sum()), no_band_share=float((~g.band).sum() / n),
        no_band_share_per_image=dict(q3(nb), min=float(min(nb)), max=float(max(nb))),
        comparable_points=int(g.comp.sum()), comparable_share_of_all_points=float(g.comp.sum() / n),
        comparable_share_of_band_points=float(g.comp.sum() / g.band.sum()) if g.band.any() else None,
        comparable_share_per_image=dict(q3(cs), min=float(min(cs)), max=float(max(cs))),
        end_reached_points=int(g.d['end'].sum()),
        end_reached_share_of_band_points=float(g.d['end'].sum() / g.band.sum()) if g.band.any() else None,
        relative_contrast_undefined_points=int((~np.isfinite(g.d['relative_contrast'])).sum()),
        spacing=dict(images_with_spacing=len(sp), images_without_spacing=len(g.rows) - len(sp),
                     per_image_median_spacing_px=dict(q3(sp), min=float(min(sp)) if sp else None, max=float(max(sp)) if sp else None),
                     points_with_width_over_spacing=int(np.isfinite(g.d['log10_width_over_spacing']).sum())))


def report(measured, synthetic):
    """Statistics of the measured groups (name -> Group) and, when the real group is among them, its comparison with the synthetic
    groups (name -> Group or StoredGroup)."""
    out = dict(groups={name: group_counts(g) for name, g in measured.items()}, summary={}, histograms={})
    for q in POINT_Q:
        out['summary'][q], out['histograms'][q] = {}, dict(edges=[float(v) for v in edges(q)])
        for name, g in measured.items():
            out['summary'][q][name], out['histograms'][q][name] = {}, {}
            for sub in ('all', 'comparable'):
                v, _ = g.values(q, sub)
                s = q3(v)
                if q == 'log10_width_over_spacing':     # also the plain ratio
                    s['ratio'] = q3(10 ** v)
                out['summary'][q][name][sub] = s
                out['histograms'][q][name][sub] = [int(c) for c in hist(q, v)]
    tex = {name: g.variation() for name, g in measured.items()}
    out['summary']['background_variation'] = {name: dict(q3(v), min=float(min(v)), max=float(max(v))) for name, v in tex.items()}
    out['histograms']['background_variation'] = dict(edges=[float(v) for v in edges('background_variation')],
                                                     **{name: [int(c) for c in hist('background_variation', v)] for name, v in tex.items()})
    out['self_overlap'] = {q: {name: {sub: self_overlap(g, q, sub) for sub in ('all', 'comparable')} for name, g in measured.items()}
                           for q in POINT_Q}
    if 'real' in measured and synthetic:
        real = measured['real']
        out['overlap'], out['coverage'] = {}, {}
        for q in POINT_Q:
            out['overlap'][q] = {s: dict(all_points=compare(real, g, q, 'all', 'all'),
                                         comparable_subset=compare(real, g, q, 'comparable', 'all'),
                                         both_sides_comparable=compare(real, g, q, 'comparable', 'comparable')) for s, g in synthetic.items()}
            out['coverage'][q] = {s: dict(all_points=coverage(real, g, q, 'all'), comparable_subset=coverage(real, g, q, 'comparable'))
                                  for s, g in synthetic.items()}
        out['overlap']['background_variation'] = {}
        for s, g in synthetic.items():
            ts = g.variation()
            out['overlap']['background_variation'][s] = dict(
                n_real=len(tex['real']), n_synthetic=len(ts),
                overlap=ovl(hist('background_variation', tex['real']), hist('background_variation', ts)),
                real_images_below_synthetic_max=int(sum(v <= max(ts) for v in tex['real'])),
                real_images_above_synthetic_max=int(sum(v > max(ts) for v in tex['real'])))
    out['per_image'] = {name: g.rows for name, g in measured.items()}
    return out


# ---------------------------------------------------------------- the ruler of the dark-band width table
def stat_fine(w, e):
    ok = np.isfinite(w)
    total = len(w)
    if not ok.any():
        return dict(samples=total, median=None, iqr=None, no_band_share=1. if total else None, end_reached=int(e.sum()))
    q = np.quantile(w[ok], [.25, .5, .75])
    return dict(samples=total, median=float(q[1]), iqr=[float(q[0]), float(q[2])], no_band_share=float((~ok).sum() / total),
                end_reached=int(e.sum()))


def ruler_check(frames):
    """band_measure with the parameters of records/fine_width_formal.json (half length 0.40 m up to 220 px, centre part
    max(2 px, 0.04 m), mean of R, G and B, every line pixel) on six of its rows; `equal` when the statistics are identical"""
    ref = {r['id']: r for r in read_json(FINE_WIDTH)['rows']}
    rows = []
    for rid in RULER_ROWS:
        r = ref[rid]
        key = Path(r['label_dir']).name
        target = frame_layer(frames, key, 'target')
        Y = frame_image(frames, key).astype(float).mean(axis=2)
        mm = r['pixel_mm'] / 1000.
        m = band_measure(Y, np.argwhere(target), min(220, int(math.ceil(.40 / mm))), max(2., .04 / mm), .25, 1)
        got = stat_fine(m['width'], m['end'])
        rows.append(dict(frame=key, case=rid, pixel_mm=r['pixel_mm'], recomputed=got, equal=got == r['fine']))
    return rows


# ---------------------------------------------------------------- comparison of two records
def differences(new, ref, tol=1e-9, path=''):
    """Leaves of `new` that differ from `ref` (numbers: by more than tol, relative above 1 and absolute below) or that `ref` lacks.
    Rows with an image or frame identifier are matched by it. What only `ref` has is not listed."""
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    ident = lambda v: isinstance(v, dict) and (v.get('image_id') or v.get('frame'))
    out = []
    if isinstance(new, dict) and isinstance(ref, dict):
        for k in new:
            p = '%s.%s' % (path, k) if path else str(k)
            out += differences(new[k], ref[k], tol, p) if k in ref else ['%s: not in the reference' % p]
    elif isinstance(new, list) and isinstance(ref, list) and new and all(ident(v) for v in new):
        byid = {ident(v): v for v in ref if ident(v)}
        for v in new:
            p = '%s[%s]' % (path, ident(v))
            out += differences(v, byid[ident(v)], tol, p) if ident(v) in byid else ['%s: not in the reference' % p]
    elif isinstance(new, list) and isinstance(ref, list):
        if len(new) != len(ref):
            out.append('%s: %d against %d entries' % (path, len(new), len(ref)))
        for i, (a, b) in enumerate(zip(new, ref)):
            out += differences(a, b, tol, '%s[%d]' % (path, i))
    elif number(new) and number(ref):
        if abs(new - ref) > tol * max(1., abs(ref)):
            out.append('%s: %r against %r' % (path, new, ref))
    elif new != ref:
        out.append('%s: %r against %r' % (path, new, ref))
    return out


def print_differences(new, ref_path, sections, tol=1e-9):
    """compare the result sections of a record with a stored record (after the round trip through JSON that the stored one made)"""
    ref = read_json(ref_path)
    new = json.loads(json.dumps({k: new[k] for k in sections if k in new}))
    diff = differences(new, ref, tol)
    leaves = lambda v: sum(leaves(x) for x in (v.values() if isinstance(v, dict) else v)) if isinstance(v, (dict, list)) else 1
    print('compared with %s: %d values, %d differences (relative tolerance %g)' % (ref_path, leaves(new), len(diff), tol))
    for d in diff[:60]:
        print('  ' + d)
    if len(diff) > 60:
        print('  ... and %d more' % (len(diff) - 60))
    return diff


# ---------------------------------------------------------------- main
def header(measured, checks):
    """the descriptive part of the record"""
    idx = fio.index()
    aerial = sorted(k for k, v in idx.items() if v.get('block') == AERIAL_BLOCK)
    px = lambda keys: [min(idx[k]['pixel_mm'] for k in keys), max(idx[k]['pixel_mm'] for k in keys)]
    return dict(record='geocrack_comparison',
                description='Dark band along fracture lines in the GeoCrack photographs and in synthetic frames, measured with one code path',
                paper='main text Section 4.3 (first paragraph); supplement S4, comparison with real photographs (two tables)',
                script='reproduce/paper1/analysis/geocrack_comparison.py',
                dataset=DATASET, measured=list(measured),
                synthetic=dict(aerial_frames=aerial, aerial_pixel_mm_range=px(aerial), close_range_frames=list(CLOSE_RANGE),
                               close_range_pixel_mm_range=px(CLOSE_RANGE)),
                parameters=dict(stride_px=STRIDE, profile_half_length_px=R_PROF, profile_step_px=STEP, surrounding_from_px=.75 * R_PROF,
                                centre_reach_px=CENTRE_REACH, no_band_below_levels=NO_BAND, comparable_min_width_px=COMP_MIN_WIDTH,
                                background_distance_px=BG_DIST, background_sigma_px=SIGMA,
                                spacing=dict(start_px=S_MIN, max_px=S_MAX, step_px=S_STEP, min_hits=MIN_HITS),
                                brightness='Rec. 709 weights on 8-bit sRGB values'),
                quantities=QUANTITIES, notes=NOTES, checks=checks)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Dark band along fracture lines: GeoCrack photographs against the generator\'s frames.')
    ap.add_argument('--geocrack', help='folder of the GeoCrack dataset; omit to measure the frames only')
    ap.add_argument('--frames', help='folder with one sub-folder per frame (see frames_io.py); omit to take the synthetic side from --record')
    ap.add_argument('--out', default='geocrack_comparison.json', help='record to write (default: geocrack_comparison.json in the current folder)')
    ap.add_argument('--images', nargs='+', default=list(IMAGES), help='photographs to use (default: the 49 of the paper)')
    ap.add_argument('--image-dir', default=IMAGE_DIR, help='folder of the photographs inside GEOCRACK (default: "%s")' % IMAGE_DIR)
    ap.add_argument('--mask-dir', default=MASK_DIR, help='folder of the edge masks inside GEOCRACK (default: "%s")' % MASK_DIR)
    ap.add_argument('--record', default=str(RECORD), help='stored record that supplies the synthetic side when --frames is not given')
    ap.add_argument('--reference', help='stored record to compare the result with')
    ap.add_argument('--tolerance', type=float, default=1e-9, help='relative tolerance of --reference (default 1e-9)')
    ap.add_argument('--check-ruler', action='store_true', help='recompute six rows of records/fine_width_formal.json (needs --frames)')
    a = ap.parse_args(argv)
    if not a.geocrack and not a.frames:
        ap.error('give --geocrack, --frames or both')
    if a.check_ruler and not a.frames:
        ap.error('--check-ruler needs --frames')

    idx = fio.index()
    aerial = sorted(k for k, v in idx.items() if v.get('block') == AERIAL_BLOCK)
    measured, synthetic, checks = {}, {}, {}
    if a.geocrack:
        real = measured['real'] = Group('real')
        same = 0
        for image_id in sorted(a.images):
            pi, pm = geocrack_pair(a.geocrack, image_id, a.image_dir, a.mask_dir)
            with Image.open(pi) as im, Image.open(pm) as mk:
                if im.size != mk.size:
                    raise SystemExit('%s: photograph and mask differ in size' % image_id)
                rgb = np.asarray(im.convert('RGB'))
                mask = np.asarray(mk.convert('L')) > 0
            Y = luminance709(rgb)
            del rgb
            skel = zs_sparse(mask)
            same += skeleton_crop_check(mask)
            row = dict(image_id=image_id, shape=[int(mask.shape[0]), int(mask.shape[1])], mask_px=int(mask.sum()))
            real.add(row, measure_image(Y, skel, None, mask))
            print('real %-9s %5.1f MP  points %d' % (image_id, mask.size / 1e6, row['points']), flush=True)
            del Y, mask, skel
        checks['skeleton'] = dict(images=len(real.rows), identical_to_label_program_on_crop=int(same))
    if a.frames:
        for name, keys in zip(SYNTHETIC, (aerial, list(CLOSE_RANGE))):
            g = measured[name] = synthetic[name] = Group(name)
            for k in keys:
                valid_px, m = measure_frame(a.frames, k)
                g.add(dict(frame=k, pixel_mm=idx[k]['pixel_mm'], valid_px=valid_px), m)
            print(name, len(keys), 'frames', flush=True)
        if a.check_ruler:
            checks['ruler'] = ruler_check(a.frames)
            print('ruler check: %d of %d rows equal to records/fine_width_formal.json' % (sum(r['equal'] for r in checks['ruler']), len(checks['ruler'])))
    else:
        stored = read_json(a.record)
        synthetic = {name: StoredGroup(name, stored) for name in SYNTHETIC}
        print('synthetic side taken from', a.record, flush=True)
    for g in measured.values():
        g.close()

    out = header(measured, checks)
    out.update(report(measured, synthetic))
    write_json(a.out, out)
    print('written', a.out)
    if a.reference:
        subset = a.geocrack and set(a.images) != set(IMAGES)
        if subset:
            print('not all photographs of the paper were measured: only the rows of per_image are compared')
        print_differences(out, a.reference, ('per_image',) if subset else RESULTS, a.tolerance)
    return 0


if __name__ == '__main__':
    sys.exit(main())
