"""Mapped traces on real outcrops: three sites of the FraXet dataset (main text Section 4.3, second paragraph; Section 5.1, last
paragraph and Fig. 5; Section 5.2, last paragraph; supplement S4, "Comparison on real-outcrop data").

Three analyses on UAV orthophotos with manual trace maps and digital elevation models (DEM):

  occlusion     line of sight from every trace pixel across the DEM, by off-nadir angle beta and by the azimuth of the line of
                sight relative to the local trace strike (across strike, along strike)
  profiles      dark-band width and contrast along the traces, with the ruler of geocrack_comparison.py, in pixels and in
                millimetres, next to the frames of the multi-illumination aerial set (block C of records/frames_index.json)
  illumination  brightness difference of the traces against the angle phi between the illumination azimuth and the trace strike;
                the azimuth is estimated by regressing image brightness on the DEM gradient

    python reproduce/paper1/analysis/fraxet_sites.py --fraxet FRAXET --out fraxet_sites.json

FRAXET  the folder of the dataset that holds "ovaskainen23_" and "samsu19" (FraXet_v0.1 in the published archive);
        README_real_data.md says where the dataset is published. The script reads three of its folders, each with image/*.png
        (orthophoto), gt/*.tif (trace annotation) and dem/*.tif (elevation in metres) in patches of 256 x 256 px:
            OG1  ovaskainen23_/train  Loviisa coast, Finland  5-6 mm/px
            KB3  ovaskainen23_/valid  Loviisa coast, Finland  5-6 mm/px
            EN   samsu19/train        Australia               29-32 mm/px
--frames FRAMES     folder with one sub-folder per frame, as described in frames_io.py (the 50 frames of block C): the synthetic
                    reference of the profile readings is measured again. Without it, the synthetic reference is taken from the
                    stored record (--record, default records/round3/fraxet_sites.json). With --frames and without --fraxet only
                    the synthetic reference is written.
--sites SITE ...    sites to analyse (default: OG1 KB3 EN)
--device cuda|cpu   device of the line-of-sight computation (default: cuda when available). The stored record was computed
                    with cuda in single precision; rounding can move single sample points between devices.
--reference RECORD  lists the differences between the result and a stored record, for the parts that this run has computed
                    (relative tolerance 1e-9, --tolerance; the regression of the illumination azimuth can differ in its last
                    digits between runs, and the Spearman correlation of the illumination analysis then differs by up to
                    about 0.0001, see Illumination below)

The output has the layout of records/round3/fraxet_sites.json. No file of the dataset is written anywhere: the record holds
statistics only. Needs Python 3.8+, NumPy, SciPy, Pillow (the labels environment) and PyTorch, which is not part of the two
environments of the generator. The stored record took about 3 minutes with a GPU on a machine with 64 GB of memory; the largest
site is held in memory as one mosaic of 12,800 x 35,584 px (its peak memory use was not recorded).

Data and stitching
- The patches are not georeferenced. The pixel size is taken at the two ends of the range given in the literature, and every
  quantity that uses it is computed once for each end.
- Valid pixels: alpha > 0 in four-channel images; in three-channel images, regions of pure white connected to the patch border
  are invalid. Pixels whose elevation is not finite or below -1000 (the no-data value is -32767) are invalid.
- The two numbers of a file name are read as (row, column) and as (column, row). The reading with the smaller median elevation
  step across the shared edges of adjacent patches is used; it is accepted when that median is at most half of the other
  reading's and at most 20 mm. The patches are then assembled into one image, annotation and DEM per site.

Occlusion
- Sample points: the annotation inside the valid domain is thinned to a skeleton (Zhang-Suen); every skeleton pixel with at
  least 3 skeleton pixels within 6 px (itself included) is one sample point. The local strike is the principal direction of the
  skeleton pixels within 6 px. Points are counted, not weighted by arc length.
- Line of sight: off-nadir angle beta of 15, 30, 45 and 60 degrees; horizontal azimuth across strike (both sides) and along
  strike (both directions). From the elevation z0 of the sample point the horizontal distance s grows from 1 px in steps of
  0.5 px, and the line of sight is at z0 + s / tan(beta). The point is occluded when the DEM (bilinear) exceeds the line of sight
  by more than half a pixel size anywhere on the way.
- The line of sight is followed until it is the clearance height above its start (1.0 m; 0.5 and 2.0 m as a sensitivity check).
  A point whose line of sight leaves the valid DEM before that, without having been occluded, is undetermined.
- occluded_share = occluded / determined, both sides or directions pooled (null with fewer than 1000 determined points);
  occluded_share_undetermined_as_visible = occluded / (determined + undetermined), a lower bound.
- elevation_grid: runs of identical values and of zero second difference along 400 sampled rows of the DEM show whether the DEM
  was upsampled to the image grid; joint grooves narrower than the native DEM spacing are absent from the DEM.

Profiles
- geocrack_comparison.band_measure with the same parameters: every 8th skeleton pixel, profile of +-63 px along the local normal,
  Rec. 709 brightness, surrounding brightness = median of |s| >= 47.25 px, centre brightness = minimum within +-6.3 px, no dark
  band below a difference of 2 grey levels. A measurement point whose profile passes through invalid pixels is not used.
- Synthetic reference: the same readings along the target layer of the 50 frames of block C; widths in millimetres by the
  nominal pixel size of each frame. In the overlapping coefficients every frame has a total weight of 1 over its measurement
  points and the real points of a site have equal weights; the synthetic 5th and 95th percentiles are taken over the points
  without weights. Bins: 0.05 for the relative contrast (0 to 1.5), 5 mm for the width (0 to 500 mm). The widths of EN are not
  compared (its pixel size is 4 to 8 times that of the frames).

Illumination
- Over the valid pixels, brightness smoothed with a Gaussian (sigma = 4 px) is regressed linearly on the two horizontal gradient
  components of the DEM smoothed in the same way (every 4th pixel in both directions). The azimuth is the aspect towards which
  the brightness increases fastest (opposite to the coefficient vector), in the image frame: degrees from +x (right) towards +y
  (down). The regression cannot separate direct light from albedo that varies with aspect.
- The estimate is accepted when the azimuths of the patches with odd and with even row index differ by at most 20 degrees and
  the gradient terms of both regressions have a joint t value above 10.
- phi = angle between the azimuth and the local strike at a measurement point of the profiles (0 to 90 degrees); medians of the
  brightness difference (surrounding - centre) in bins of 15 degrees and the Spearman correlation of the two. trend:
  "increases with phi" when rho >= 0.1, at least 4 of the 5 steps between bins go up and the last bin is above the first;
  "decreases with phi" for the opposite; otherwise "no consistent change".
- The Spearman correlation is reproducible to within about 0.0001. Many measurement points share a local strike, and values of
  phi that are equal in exact arithmetic can differ in their last bits; the rank order of such values follows the last bits of
  the estimated azimuth, which can differ between runs (the regression is solved with multi-threaded linear algebra)."""
import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames_io as fio  # noqa: E402  (index of the paper's frames)
import geocrack_comparison as gc  # noqa: E402  (zs_sparse, luminance709, band_measure, image_weights and the ruler constants)

RECORD = HERE.parent / 'records' / 'round3' / 'fraxet_sites.json'
SITES = {
    'OG1': dict(folder='ovaskainen23_/train', prefix='og1', pixel_mm=(5., 6.), location='Loviisa coast, Finland'),
    'KB3': dict(folder='ovaskainen23_/valid', prefix='kb3', pixel_mm=(5., 6.), location='Loviisa coast, Finland'),
    'EN': dict(folder='samsu19/train', prefix='en', pixel_mm=(29., 32.), location='Australia'),
}
WIDTH_COMPARED = ('OG1', 'KB3')       # sites whose pixel size is close to that of the synthetic frames
AERIAL_BLOCK = 'C'
RESULTS = ('synthetic_reference', 'sites')
DATASET = dict(
    name='FraXet', version='0.1', doi='10.5281/zenodo.17069947', licence='CC BY 4.0',
    references=[
        'Fatihi, A., Caldeira, J., Beucler, T., Thiele, S., Samsu, A., 2025. FraXet (version 0.1) [data set]. Zenodo. '
        'https://doi.org/10.5281/zenodo.17069947',
        'Fatihi, A., Caldeira, J., Beucler, T., Thiele, S.T., Samsu, A., 2026. Towards robust fracture mapping: benchmarking automatic '
        'fracture mapping in 2D outcrop imagery. Solid Earth 17, 1087–1115. https://doi.org/10.5194/se-17-1087-2026'],
    original_sources={
        'OG1 and KB3 (manual traces; orthomosaics and DEMs)': [
            'Ovaskainen, N., Nordbäck, N., 2022. Manually mapped traces from UAV-acquired images of Loviisa shoreline outcrops '
            '(version 1.1.0) [data set]. Zenodo. https://doi.org/10.5281/zenodo.7077846',
            'Nordbäck, N., Ovaskainen, N., 2025. UAV-acquired orthomosaics of Loviisa shoreline outcrops [data set]. Zenodo. '
            'https://doi.org/10.5281/zenodo.17878870'],
        'EN': [
            'Samsu, A., Cruden, A., Vollgger, S., 2019. Scale matters: the influence of structural inheritance on fracture patterns – '
            'Supplementary material [data set]. Monash University. https://doi.org/10.26180/5cdcad0a73fe0',
            'Samsu, A., Cruden, A.R., Micklethwaite, S., Grose, L., Vollgger, S.A., 2020. Scale matters: The influence of structural '
            'inheritance on fracture patterns. Journal of Structural Geology 130, 103896. https://doi.org/10.1016/j.jsg.2019.103896']},
    used='orthophoto, trace annotation and DEM patches of the sites OG1, KB3 and EN; no file of the dataset is redistributed, this '
         'record holds statistics only')

T = 256                       # patch size (px)
BETAS = (15., 30., 45., 60.)
HC = (0.5, 1.0, 2.0)          # clearance heights (m); 1.0 is the main value
HC_MAIN = 1.0
STEP_PX = 0.5
S0_PX = 1.0
MIN_DET = 1000
NB_R = 6.                     # neighbourhood radius for the local strike (px)
SIGMA_SUN = 4.
PHI_EDGES = (0., 15., 30., 45., 60., 75., 90.)
W_BIN_MM = 5.                 # histogram bin of the dark-band width in mm
W_MAX_MM = 500.
REL_BIN = 0.05
REL_MAX = 1.5
DEV = None                    # torch device, set in main

NOTES = [
    'The patches are not georeferenced: the pixel size is taken at the two ends of the range given in the literature (keys of '
    'by_pixel_mm and dark_band_width_mm), and every quantity that uses it is given for both.',
    'Valid pixels: alpha > 0, or not pure white connected to the patch border in three-channel images; elevation finite and above '
    '-1000 (no-data value -32767). Annotation pixels outside the valid domain are not sample points: '
    'mosaic.annotation_px_outside_valid counts them and mosaic.annotation_px counts those inside.',
    'stitching: median and 90th percentile of the absolute elevation step across the shared edges of adjacent patches, for the two '
    'readings of the file names; the patches were assembled with the reading named in "chosen".',
    'Sample points of the occlusion analysis are skeleton pixels, counted without arc-length weights. Rows of by_pixel_mm: one per '
    'off-nadir angle beta_deg and clearance height clearance_m; "across" and "along" pool the two sides or directions '
    '(side_shares gives each), so their counts are twice the number of sample points.',
    'occluded_share excludes the undetermined points (line of sight left the valid DEM before reaching the clearance height) from '
    'the denominator; occluded_share_undetermined_as_visible counts them all as not occluded and is a lower bound; '
    'undetermined_share is their share of all sample points.',
    'elevation_grid: OG1 has few runs of identical or linearly interpolated values (DEM on the image grid); in KB3 the DEM changes '
    'every 4 image pixels and in EN every 2.',
    'Profiles use the ruler of geocrack_comparison.py (every 8th skeleton pixel, +-63 px, Rec. 709 brightness); points whose '
    'profile passes through invalid pixels are dropped. Quantiles are over the measurement points of a site.',
    'synthetic_reference: the same readings along the target layer of the 50 frames of the multi-illumination aerial set. Its histograms '
    'are weighted so that every frame has a total weight of 1 over its measurement points; they are what the overlapping '
    'coefficients of vs_synthetic are computed with, and its dark_band_width_mm.p5 and p95 bound the shares below, inside and above.',
    'vs_synthetic.width_mm.*.median_position: "narrower" when the median width of the site is below the synthetic 5th percentile, '
    '"wider" when it is above the 95th, otherwise "inside". The widths of EN are not compared.',
    'illumination.azimuth_image_deg is in the image frame (degrees from +x towards +y, y down), not a compass bearing; it is taken '
    'as the horizontal azimuth of the sun. The regression cannot separate direct light from albedo that varies with aspect. '
    'illumination.spearman_phi_brightness_difference is reproducible to within about 0.0001: values of phi that are equal in exact '
    'arithmetic can differ in their last bits, and their rank order follows the last bits of the estimated azimuth, which can '
    'differ between runs.',
    'The line-of-sight computation of the stored record ran in single precision on a GPU ("device").']


def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- loading and stitching
def load_site(root, key):
    cfg = SITES[key]
    d = Path(root) / cfg['folder']
    if not (d / 'image').is_dir():
        raise SystemExit('%s: folder %s not found under %s' % (key, cfg['folder'] + '/image', root))
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(d / 'image'))
    pat = re.compile(cfg['prefix'] + r'_(\d+)_(\d+)')
    tiles = {}
    for n in names:
        m = pat.fullmatch(n)
        assert m, n
        a, b = int(m.group(1)), int(m.group(2))
        with Image.open(d / 'image' / (n + '.png')) as im:
            arr = np.asarray(im)
        with Image.open(d / 'dem' / (n + '.tif')) as im:
            dem = np.asarray(im, np.float32)
        with Image.open(d / 'gt' / (n + '.tif')) as im:
            lab = np.asarray(im) > 0
        assert arr.shape[:2] == dem.shape == lab.shape == (T, T), n
        rgb = arr[..., :3]
        if arr.shape[2] == 4:
            valid = arr[..., 3] > 0
        else:  # white fill connected to the patch border is not rock
            white = (rgb == 255).all(axis=2)
            lbl, nlab = ndi.label(white)
            edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
            valid = ~np.isin(lbl, edge[edge > 0])
        valid &= np.isfinite(dem) & (dem > -1000.)
        tiles[(a, b)] = dict(Y=gc.luminance709(rgb).astype(np.float32), dem=dem, lab=lab, valid=valid)
    return cfg, tiles, dict(patches=len(names), patch_names_sha256=sha_text('\n'.join(names)))


def seam_stats(tiles, reading):
    """median absolute elevation step across shared patch edges for one reading of the two indices"""
    pos = {(k if reading == 'row_col' else (k[1], k[0])): v for k, v in tiles.items()}
    diffs = []
    for (r, c), t in pos.items():
        o = pos.get((r, c + 1))
        if o is not None:
            ok = t['valid'][:, -1] & o['valid'][:, 0]
            if ok.any():
                diffs.append(np.abs(t['dem'][:, -1] - o['dem'][:, 0])[ok])
        o = pos.get((r + 1, c))
        if o is not None:
            ok = t['valid'][-1, :] & o['valid'][0, :]
            if ok.any():
                diffs.append(np.abs(t['dem'][-1, :] - o['dem'][0, :])[ok])
    d = np.concatenate(diffs) if diffs else np.zeros(0)
    return dict(pairs=len(diffs), median_m=float(np.median(d)) if len(d) else None, p90_m=float(np.quantile(d, .9)) if len(d) else None)


def stitch(tiles, reading):
    pos = {(k if reading == 'row_col' else (k[1], k[0])): v for k, v in tiles.items()}
    r0, c0 = min(k[0] for k in pos), min(k[1] for k in pos)
    R = max(k[0] for k in pos) - r0 + 1
    C = max(k[1] for k in pos) - c0 + 1
    Y = np.zeros((R * T, C * T), np.float32)
    dem = np.full((R * T, C * T), np.nan, np.float32)
    lab = np.zeros((R * T, C * T), bool)
    valid = np.zeros((R * T, C * T), bool)
    row_of = np.full((R * T,), -1, np.int32)
    outside = 0
    for (r, c), t in pos.items():
        ys, xs = slice((r - r0) * T, (r - r0 + 1) * T), slice((c - c0) * T, (c - c0 + 1) * T)
        Y[ys, xs] = t['Y']
        dem[ys, xs] = np.where(t['valid'], t['dem'], np.nan)
        lab[ys, xs] = t['lab'] & t['valid']
        valid[ys, xs] = t['valid']
        row_of[ys] = r
        outside += int((t['lab'] & ~t['valid']).sum())
    return dict(Y=Y, dem=dem, lab=lab, valid=valid, tile_rows=R, tile_cols=C, row_of=row_of, lab_outside=outside)


def dem_resolution(dem, valid):
    """two estimates of the native elevation pixel in image pixels: mean run of identical values along rows (nearest upsampling)
    and mean run of zero second differences along rows + 2 (bilinear upsampling); on a sample of rows"""
    rows = np.linspace(0, dem.shape[0] - 1, 400).astype(int)
    run_eq, run_lin, n_valid = [], [], 0
    for r in rows:
        v = valid[r]
        z = dem[r]
        if v.sum() < 64:
            continue
        n_valid += int(v.sum())
        same = (z[1:] == z[:-1]) & v[1:] & v[:-1]
        lin = (np.abs(z[2:] - 2 * z[1:-1] + z[:-2]) < 1e-6) & v[2:] & v[1:-1] & v[:-2]
        for flag, store, base in ((same, run_eq, 1), (lin, run_lin, 2)):
            f = np.concatenate([[False], flag, [False]])
            starts = np.flatnonzero(~f[:-1] & f[1:])
            ends = np.flatnonzero(f[:-1] & ~f[1:])
            if len(starts):
                store.append((ends - starts).astype(float) + base)
    eq = np.concatenate(run_eq) if run_eq else np.zeros(0)
    li = np.concatenate(run_lin) if run_lin else np.zeros(0)
    return dict(method='400 rows sampled: runs of identical values (+1) and runs of zero second difference (+2), lengths in image pixels; '
                       'share = pixels covered by such runs / valid pixels of the sampled rows',
                identical_run_median_px=float(np.median(eq)) if len(eq) else None,
                identical_run_share=float(eq.sum() / n_valid) if n_valid else None,
                linear_run_median_px=float(np.median(li)) if len(li) else None,
                linear_run_share=float(li.sum() / n_valid) if n_valid else None)


# ---------------------------------------------------------------- sample points and local strike
def strike_field(pts):
    """principal direction of the skeleton pixels within NB_R of every skeleton pixel: unit vector (ty, tx), number of pixels"""
    tree = cKDTree(pts)
    coo = tree.sparse_distance_matrix(tree, NB_R, output_type='coo_matrix')
    off = coo.row != coo.col                                   # drop self pairs if stored; the point itself is added below
    i, j = coo.row[off], coo.col[off]
    n = np.bincount(i, minlength=len(pts)).astype(float)
    dy = (pts[j, 0] - pts[i, 0]).astype(float)
    dx = (pts[j, 1] - pts[i, 1]).astype(float)
    n1 = n + 1.
    my = np.bincount(i, dy, len(pts)) / n1
    mx = np.bincount(i, dx, len(pts)) / n1
    cyy = np.bincount(i, dy * dy, len(pts)) / n1 - my * my
    cxx = np.bincount(i, dx * dx, len(pts)) / n1 - mx * mx
    cxy = np.bincount(i, dx * dy, len(pts)) / n1 - mx * my
    ang = .5 * np.arctan2(2. * cxy, cxx - cyy)                 # angle of the major axis from +x
    return np.sin(ang), np.cos(ang), n1


# ---------------------------------------------------------------- occlusion: lines of sight across the DEM
def bilinear(dem_t, y, x):
    import torch
    H, W = dem_t.shape
    y0 = torch.floor(y)
    x0 = torch.floor(x)
    inside = (y0 >= 0) & (x0 >= 0) & (y0 < H - 1) & (x0 < W - 1)
    yi = y0.clamp(0, H - 2).long()
    xi = x0.clamp(0, W - 2).long()
    fy, fx = y - y0, x - x0
    z = (dem_t[yi, xi] * (1 - fy) * (1 - fx) + dem_t[yi, xi + 1] * (1 - fy) * fx
         + dem_t[yi + 1, xi] * fy * (1 - fx) + dem_t[yi + 1, xi + 1] * fy * fx)
    return torch.where(inside, z, torch.full_like(z, float('nan')))


def march(dem_t, y, x, z0, dy, dx, gsd_m, beta_deg):
    """first-occlusion distance and exit distance (pixels) of the lines of sight from (y, x, z0) along (dy, dx), up to the largest
    clearance height"""
    import torch
    tb = math.tan(math.radians(beta_deg))
    smax = max(HC) * tb / gsd_m
    eps = .5 * gsd_m
    n = len(y)
    d_occ = torch.full((n,), float('inf'), device=DEV)
    d_exit = torch.full((n,), float('inf'), device=DEV)
    active = torch.arange(n, device=DEV)
    s = S0_PX
    while s <= smax + 1e-9 and len(active):
        z = bilinear(dem_t, y[active] + s * dy[active], x[active] + s * dx[active])
        zr = z0[active] + s * gsd_m / tb
        gone = torch.isnan(z)
        occ = ~gone & (z - zr > eps)
        d_exit[active[gone]] = s
        d_occ[active[occ]] = s
        active = active[~(gone | occ)]
        s += STEP_PX
    return d_occ.cpu().numpy(), d_exit.cpu().numpy()


def occlusion_site(m, pts, ty, tx, nb, pixel_mm):
    import torch
    dem_t = torch.from_numpy(m['dem']).to(DEV)
    y = torch.from_numpy(pts[:, 0].astype(np.float32)).to(DEV)
    x = torch.from_numpy(pts[:, 1].astype(np.float32)).to(DEV)
    z0 = dem_t[y.long(), x.long()]
    keep = (~torch.isnan(z0)).cpu().numpy() & (nb >= 3)
    sel = torch.from_numpy(np.flatnonzero(keep)).to(DEV)
    y, x, z0 = y[sel], x[sel], z0[sel]
    tyt = torch.from_numpy(ty[keep].astype(np.float32)).to(DEV)
    txt = torch.from_numpy(tx[keep].astype(np.float32)).to(DEV)
    dirs = dict(across_a=(txt, -tyt), across_b=(-txt, tyt), along_a=(tyt, txt), along_b=(-tyt, -txt))   # (dy, dx)
    out = dict(samples=int(keep.sum()), by_pixel_mm={})
    for g in pixel_mm:
        gm = g / 1000.
        rows = []
        for b in BETAS:
            res = {k: march(dem_t, y, x, z0, dy, dx, gm, b) for k, (dy, dx) in dirs.items()}
            tb = math.tan(math.radians(b))
            for hc in HC:
                smax = hc * tb / gm
                row = dict(beta_deg=b, clearance_m=hc)
                for kind in ('across', 'along'):
                    parts = []
                    for side in ('a', 'b'):
                        d_occ, d_exit = res['%s_%s' % (kind, side)]
                        occ = d_occ <= smax
                        und = ~occ & (d_exit < smax)
                        det = ~und
                        parts.append((int(occ.sum()), int(det.sum()), int(und.sum())))
                    o, dn, u = (sum(p[i] for p in parts) for i in range(3))
                    sides = [p[0] / p[1] if p[1] else None for p in parts]
                    row[kind] = dict(occluded=o, determined=dn, undetermined=u,
                                     occluded_share=(o / dn) if dn >= MIN_DET else None,
                                     occluded_share_undetermined_as_visible=(o / (dn + u)) if dn >= MIN_DET else None,
                                     undetermined_share=u / (dn + u) if dn + u else None,
                                     side_shares=sides)
                rows.append(row)
        out['by_pixel_mm']['%g' % g] = rows
    del dem_t
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- profiles
def profile_valid(valid, pts, m):
    s = np.arange(-gc.R_PROF, gc.R_PROF + gc.STEP / 2., gc.STEP)
    ok = np.ones(len(m['i']), bool)
    v = valid.astype(np.uint8)
    for a in range(0, len(ok), 4000):
        b = min(len(ok), a + 4000)
        yy = pts[m['i'][a:b], 0][:, None] + s[None, :] * m['ny'][a:b, None]
        xx = pts[m['i'][a:b], 1][:, None] + s[None, :] * m['nx'][a:b, None]
        P = ndi.map_coordinates(v, [yy.ravel(), xx.ravel()], order=0, mode='constant', cval=0).reshape(b - a, len(s))
        ok[a:b] = P.all(axis=1)
    return ok


def q5(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0)
    q = np.quantile(v, [.05, .25, .5, .75, .95])
    return dict(n=int(len(v)), p5=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p95=float(q[4]))


def weighted_hist(v, w, lo, hi, width):
    """histogram of v on [lo, hi] in bins of `width`, values outside counted in the end bins; w: weights, or None for counts"""
    e = np.arange(lo, hi + width / 2., width)
    h, _ = np.histogram(np.clip(v, lo, hi - 1e-9), e, weights=w)
    return h


def hist_ovl(ha, hb):
    """overlapping coefficient of two histograms"""
    ha, hb = ha / ha.sum(), hb / hb.sum()
    return float(np.minimum(ha, hb).sum())


def synthetic_reference(frames):
    """Dark-band readings along the target layer of the multi-illumination aerial set (block C, 50 frames), same ruler; width in
    mm by the nominal pixel size of the frame. The histograms give every frame a total weight of 1 over its measurement points."""
    idx = fio.index()
    keys = sorted(k for k, v in idx.items() if v.get('block') == AERIAL_BLOCK)
    assert len(keys) == 50
    rel, wpx, wmm, img = [], [], [], []
    for n, k in enumerate(keys):
        Y = gc.luminance709(gc.frame_image(frames, k))
        line = gc.frame_layer(frames, k, 'target') > 0
        pts = np.argwhere(line)
        m = gc.band_measure(Y, pts, gc.R_PROF, gc.CENTRE_REACH, gc.STEP, gc.STRIDE)
        depth = m['ring'] - m['centre']
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(m['ring'] > 0, depth / m['ring'], np.nan)
        rel.append(r)
        wpx.append(m['width'])
        wmm.append(m['width'] * idx[k]['pixel_mm'])
        img.append(np.full(len(r), n))
    rel, wpx, wmm, img = (np.concatenate(v) for v in (rel, wpx, wmm, img))
    w = gc.image_weights(img.astype(int))
    fs, ws = np.isfinite(rel), np.isfinite(wmm)
    return dict(frames=len(keys), pixel_mm_range=[min(idx[k]['pixel_mm'] for k in keys), max(idx[k]['pixel_mm'] for k in keys)],
                points=int(len(rel)), relative_contrast=q5(rel), dark_band_width_px=q5(wpx), dark_band_width_mm=q5(wmm),
                histograms=dict(
                    relative_contrast=dict(range=[0., REL_MAX], bin_width=REL_BIN,
                                           weighted_counts=[float(v) for v in weighted_hist(rel[fs], w[fs], 0., REL_MAX, REL_BIN)]),
                    dark_band_width_mm=dict(range=[0., W_MAX_MM], bin_width=W_BIN_MM,
                                            weighted_counts=[float(v) for v in weighted_hist(wmm[ws], w[ws], 0., W_MAX_MM, W_BIN_MM)])))


def profiles_site(m, pts, pixel_mm, syn, compare_width):
    mm_ = gc.band_measure(m['Y'].astype(np.float64), pts, gc.R_PROF, gc.CENTRE_REACH, gc.STEP, gc.STRIDE)
    ok = profile_valid(m['valid'], pts, mm_)
    ring, centre, width = mm_['ring'][ok], mm_['centre'][ok], mm_['width'][ok]
    depth = ring - centre
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(ring > 0, depth / ring, np.nan)
    band = np.isfinite(width)
    out = dict(points=int(ok.sum()), dropped_invalid_profile=int((~ok).sum()), no_band_share=float((~band).mean()) if len(band) else None,
               end_reached_share=float(mm_['end'][ok].mean()) if ok.any() else None,
               surrounding_brightness=q5(ring), centre_brightness=q5(centre), brightness_difference=q5(depth), relative_contrast=q5(rel),
               dark_band_width_px=q5(width), dark_band_width_mm={('%g' % g): q5(width * g) for g in pixel_mm})
    if syn is not None:
        fr = np.isfinite(rel)
        hs = {k: np.array(v['weighted_counts'], float) for k, v in syn['histograms'].items()}
        out['vs_synthetic'] = dict(relative_contrast_overlap=hist_ovl(weighted_hist(rel[fr], None, 0., REL_MAX, REL_BIN), hs['relative_contrast']))
        if compare_width:
            sp = [syn['dark_band_width_mm']['p5'], syn['dark_band_width_mm']['p95']]
            cmpw = {}
            for g in pixel_mm:
                w = width[band] * g
                cmpw['%g' % g] = dict(overlap=hist_ovl(weighted_hist(w, None, 0., W_MAX_MM, W_BIN_MM), hs['dark_band_width_mm']),
                                      below_synthetic_p5=float((w < sp[0]).mean()),
                                      inside_synthetic_p5_p95=float(((w >= sp[0]) & (w <= sp[1])).mean()),
                                      above_synthetic_p95=float((w > sp[1]).mean()),
                                      median_position='narrower' if np.median(w) < sp[0] else ('wider' if np.median(w) > sp[1] else 'inside'))
            out['vs_synthetic']['width_mm'] = cmpw
    samples = dict(i=mm_['i'][ok], ty=mm_['nx'][ok], tx=-mm_['ny'][ok], depth=depth, rel=rel)   # tangent (ty, tx) from the normal (ny, nx) = (-tx, ty)
    return out, samples


# ---------------------------------------------------------------- illumination azimuth from shading
def masked_gauss(a, valid, sigma):
    w = ndi.gaussian_filter(valid.astype(np.float32), sigma, mode='constant')
    g = ndi.gaussian_filter(np.where(valid, a, 0.).astype(np.float32), sigma, mode='constant')
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(w > .5, g / w, np.nan)


def sun_fit(Yg, gy, gx, sel):
    yv, a, b = Yg[sel].astype(np.float64), gx[sel].astype(np.float64), gy[sel].astype(np.float64)
    X = np.stack([np.ones_like(a), a, b], axis=1)
    coef, res, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    n = len(yv)
    resid = yv - X @ coef
    s2 = float(resid @ resid) / max(1, n - 3)
    cov = s2 * np.linalg.inv(X.T @ X)
    c = coef[1:]
    wald = float(np.sqrt(c @ np.linalg.inv(cov[1:, 1:]) @ c))
    r2 = 1. - float(resid @ resid) / float(((yv - yv.mean()) ** 2).sum())
    az = math.degrees(math.atan2(-coef[2], -coef[1])) % 360.      # direction (image frame, x right, y down) that the lit slopes face
    return dict(n=n, coef=[float(v) for v in coef], azimuth_image_deg=az, wald_t=wald, r2=r2)


def illumination_site(m, samples, pixel_mm):
    g = float(np.mean(pixel_mm)) / 1000.
    valid = m['valid']
    Yg = masked_gauss(m['Y'], valid, SIGMA_SUN)
    zg = masked_gauss(np.nan_to_num(m['dem']), valid, SIGMA_SUN)
    gy, gx = np.gradient(zg, g)
    ok = np.isfinite(Yg) & np.isfinite(gx) & np.isfinite(gy)
    sub = np.zeros_like(ok)
    sub[::4, ::4] = True
    odd = (m['row_of'] % 2 == 1)[:, None]
    full = sun_fit(Yg, gy, gx, ok & sub)
    h1 = sun_fit(Yg, gy, gx, ok & sub & odd)
    h2 = sun_fit(Yg, gy, gx, ok & sub & ~odd)
    dif = abs((h1['azimuth_image_deg'] - h2['azimuth_image_deg'] + 180.) % 360. - 180.)
    passed = bool(dif <= 20. and h1['wald_t'] > 10. and h2['wald_t'] > 10.)
    out = dict(fit_all=full, fit_odd_patch_rows=h1, fit_even_patch_rows=h2, halves_azimuth_difference_deg=dif, estimate_accepted=passed)
    if not passed:
        out['trend'] = 'not testable (azimuth estimate not accepted)'
        return out
    az = math.radians(full['azimuth_image_deg'])
    sx, sy = math.cos(az), math.sin(az)
    cosphi = np.abs(samples['tx'] * sx + samples['ty'] * sy)
    phi = np.degrees(np.arccos(np.clip(cosphi, 0., 1.)))
    d = samples['depth']
    fin = np.isfinite(phi) & np.isfinite(d)
    bins = []
    for lo, hi in zip(PHI_EDGES[:-1], PHI_EDGES[1:]):
        s = fin & (phi >= lo) & ((phi < hi) | (hi == 90.))
        bins.append(dict(phi_from=lo, phi_to=hi, n=int(s.sum()), median_brightness_difference=float(np.median(d[s])) if s.any() else None,
                         median_relative_contrast=float(np.nanmedian(samples['rel'][s])) if s.any() else None))
    rho = float(stats.spearmanr(phi[fin], d[fin]).statistic)
    med = [b['median_brightness_difference'] for b in bins]
    ups = sum(med[i + 1] > med[i] for i in range(5))
    downs = sum(med[i + 1] < med[i] for i in range(5))
    if rho >= .1 and ups >= 4 and med[-1] > med[0]:
        trend = 'increases with phi'
    elif rho <= -.1 and downs >= 4 and med[-1] < med[0]:
        trend = 'decreases with phi'
    else:
        trend = 'no consistent change'
    out.update(samples=int(fin.sum()), bins=bins, spearman_phi_brightness_difference=rho, trend=trend)
    return out


# ---------------------------------------------------------------- main
def header():
    """the descriptive part of the record"""
    return dict(record='fraxet_sites',
                description='Occlusion on the DEM, dark band along the traces and illumination azimuth at three sites of the FraXet dataset',
                paper='main text Section 4.3 (second paragraph), Section 5.1 (last paragraph, Fig. 5), Section 5.2 (last paragraph); '
                      'supplement S4, comparison on real-outcrop data (four tables)',
                script='reproduce/paper1/analysis/fraxet_sites.py',
                dataset=DATASET,
                parameters=dict(betas_deg=BETAS, clearance_m=HC, clearance_main_m=HC_MAIN, step_px=STEP_PX, start_px=S0_PX,
                                occlusion_margin='0.5 pixel size', strike_radius_px=NB_R, min_determined=MIN_DET,
                                profile=dict(stride_px=gc.STRIDE, half_length_px=gc.R_PROF, step_px=gc.STEP, centre_reach_px=gc.CENTRE_REACH,
                                             no_band_below_levels=gc.NO_BAND),
                                width_bin_mm=W_BIN_MM, width_max_mm=W_MAX_MM, relative_contrast_bin=REL_BIN, relative_contrast_max=REL_MAX,
                                sun_sigma_px=SIGMA_SUN, phi_edges_deg=PHI_EDGES),
                notes=NOTES)


def analyse_site(root, key, syn):
    """stitching, occlusion, profiles and illumination of one site"""
    cfg, tiles, closure = load_site(root, key)
    sa, sb = seam_stats(tiles, 'row_col'), seam_stats(tiles, 'col_row')
    big = float('inf')
    best = 'row_col' if (big if sa['median_m'] is None else sa['median_m']) <= (big if sb['median_m'] is None else sb['median_m']) else 'col_row'
    mb, mo = (sa, sb) if best == 'row_col' else (sb, sa)
    accepted = bool(mb['median_m'] is not None and mo['median_m'] is not None and mb['median_m'] <= .5 * mo['median_m'] and mb['median_m'] <= .020)
    site = dict(location=cfg['location'], folder=cfg['folder'], pixel_mm=cfg['pixel_mm'], **closure,
                stitching=dict(row_col=sa, col_row=sb, chosen=best, accepted=accepted))
    print(key, 'patches', len(tiles), 'edge steps', sa, sb, 'accepted', accepted, flush=True)
    if not accepted:
        site['status'] = 'stitching not accepted: the patches were not assembled and the site was not analysed'
        return site
    m = stitch(tiles, best)
    del tiles
    site['mosaic'] = dict(shape=list(m['dem'].shape), patch_rows=m['tile_rows'], patch_cols=m['tile_cols'],
                          valid_px=int(m['valid'].sum()), annotation_px=int(m['lab'].sum()), annotation_px_outside_valid=m['lab_outside'],
                          elevation_range_m=[float(np.nanmin(m['dem'])), float(np.nanmax(m['dem']))])
    site['elevation_grid'] = dem_resolution(m['dem'], m['valid'])
    skel = gc.zs_sparse(m['lab'])
    pts = np.argwhere(skel)
    ty, tx, nb = strike_field(pts)
    site['skeleton_px'] = int(len(pts))
    print(key, 'mosaic', m['dem'].shape, 'skeleton', len(pts), flush=True)
    site['occlusion'] = occlusion_site(m, pts, ty, tx, nb, cfg['pixel_mm'])
    print(key, 'occlusion done', flush=True)
    site['profiles'], samples = profiles_site(m, pts, cfg['pixel_mm'], syn, compare_width=key in WIDTH_COMPARED)
    print(key, 'profiles done', site['profiles']['points'], flush=True)
    site['illumination'] = illumination_site(m, samples, cfg['pixel_mm'])
    print(key, 'illumination done', site['illumination'].get('trend'), flush=True)
    return site


def main(argv=None):
    global DEV
    ap = argparse.ArgumentParser(description='Occlusion, dark band and illumination azimuth of mapped traces at three FraXet sites.')
    ap.add_argument('--fraxet', help='folder of the FraXet dataset that holds ovaskainen23_ and samsu19; omit to measure the synthetic reference only')
    ap.add_argument('--frames', help='folder with one sub-folder per frame (see frames_io.py); omit to take the synthetic reference from --record')
    ap.add_argument('--out', default='fraxet_sites.json', help='record to write (default: fraxet_sites.json in the current folder)')
    ap.add_argument('--sites', nargs='+', default=list(SITES), choices=list(SITES), help='sites to analyse (default: all three)')
    ap.add_argument('--device', choices=('cuda', 'cpu'), help='device of the line-of-sight computation (default: cuda when available)')
    ap.add_argument('--record', default=str(RECORD), help='stored record that supplies the synthetic reference when --frames is not given')
    ap.add_argument('--reference', help='stored record to compare the result with')
    ap.add_argument('--tolerance', type=float, default=1e-9, help='relative tolerance of --reference (default 1e-9)')
    a = ap.parse_args(argv)
    if not a.fraxet and not a.frames:
        ap.error('give --fraxet, --frames or both')

    out = header()
    if a.frames:
        print('synthetic reference ...', flush=True)
        syn = synthetic_reference(a.frames)
    else:
        syn = gc.read_json(a.record)['synthetic_reference']
        print('synthetic reference taken from', a.record, flush=True)
    out['synthetic_reference'] = syn
    if a.fraxet:
        import torch
        DEV = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        out['device'] = str(DEV)
        out['sites'] = {key: analyse_site(a.fraxet, key, syn) for key in a.sites}
    gc.write_json(a.out, out)
    print('written', a.out)
    if a.reference:
        gc.print_differences(out, a.reference, RESULTS if a.frames else ('sites',), a.tolerance)
    return 0


if __name__ == '__main__':
    sys.exit(main())
