"""Footprint material 'lum-2' inputs for the B scan-detail transfer. Everything here only derives texture maps and scalar
parameters; no geometry.

- Fill code per footprint face (``fill_codes``): 0 donor-corresponding (donor front-triangle envelope, slope <=
slope_deg to the patch outward normal), 1 bedding-frame triplanar fill (synthetic lip wall, i.e. the donor wall wins
at the face centroid, or envelope steeper than slope_deg), 2 riser fill (B risers: the C#### own-plane mapping). The
GLB carries max over a vertex's footprint faces in TEXCOORD_2.y. - Donor lightness detail (``donor_detail``): linear
luminance of the donor albedo over its Gaussian low-pass at the transfer's detail band (sigma = donor.json
sigma_lowpass_m), times k so that its clamped mean equals the bed texture's clamped mean as used by rock_material;
clamp [0.5, 1.6], 8-bit D = 0.5 + 1.1 t. - Scale matching: keep the mean and the fine part (f - G_sc f) of the log
modulation f = log(1 + ds (D - 1)) and of the decoded normal slope (-x/z, -y/z); scale the coarse part (G_sc f - mean)
by one gain each; clamped mean kept; normals re-encoded with the same DirectX convention. - Fill texture luminance
normaliser (``fill_lum_mean``): own linear mean / k, k matching the same clamped mean.

Recipe 'lum-3': one complementary split in the log-modulation domain f = log(1 + ds (D - 1)) (D = clamp(L / mu, .5,
1.6) as rock_material): low = G_sL f, high = f - G_sL f, with the Gaussian sigma sL (46.8 mm; a Gaussian sigma, not a
wavelength cut) for the bed, the donor and the fill alike (wrap padding for the tiling bed and fill textures, reflect
for the donor window). The bed's low part is kept everywhere (sampled with the bed's own UV0); the donor (footprint
core) and the fill (synthetic walls) supply only their high part, whose coarse band G_sc f - G_sL f (sc = 4 mm) gets
one gain matched to the same band of the current bed (std of the telescoped DoG; lightness and decoded slope
separately, the bed slope after the normal_strength lerp); the fine band f - G_sc f is unchanged. soft' = exp(high')
is scaled so that mean(soft_low) mean(soft') = mean(soft_bed) on the bed tile. Maps are 8-bit linear with fixed
ranges."""
import json
from pathlib import Path

import numpy as np

LW32 = np.array([.2126, .7152, .0722], np.float32)


def _lin(s):
    return np.where(s <= .04045, s / 12.92, ((s + .055) / 1.055) ** 2.4)


def _lum(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255
    return (_lin(a) @ LW32).astype(np.float64)


def _gblur(L, sig):
    p = int(3 * sig)
    X = np.pad(L, p, mode='reflect')
    f0 = np.fft.fftfreq(X.shape[0])[:, None]
    f1 = np.fft.rfftfreq(X.shape[1])[None, :]
    H = np.exp(-2 * (np.pi * sig) ** 2 * (f0 ** 2 + f1 ** 2))
    return np.fft.irfft2(np.fft.rfft2(X) * H, s=X.shape)[p:-p, p:-p]


def _match_k(values, target, lo=.5, hi=4., fn=None):
    """k with mean(clip(k * values, .5, 1.6)) = target (bisection, 60 steps)."""
    for _ in range(60):
        k = .5 * (lo + hi)
        m = (np.clip(k * values, .5, 1.6) if fn is None else fn(k)).mean()
        lo, hi = (k, hi) if m < target else (lo, k)
    return .5 * (lo + hi)


def bed_clamped_mean(record):
    """Mean of clamp(L / mu, .5, 1.6) of a scene texture record as rock_material uses it (recorded mean)."""
    L = _lum(record['albedo'])
    mu = float(np.array(record['mean_linear']) @ LW32)
    return float(np.clip(L / mu, .5, 1.6).mean())


def donor_detail(donor_dir, target):
    """8-bit donor lightness detail (uint8 array) and stats; donor_dir holds albedo.png and donor.json."""
    donor_dir = Path(donor_dir)
    info = json.loads((donor_dir / 'donor.json').read_text(encoding='utf8'))
    L = _lum(donor_dir / 'albedo.png')
    px = info['W_m'] / L.shape[0]
    r = L / np.maximum(_gblur(L, info['sigma_lowpass_m'] / px), 1e-6)
    k = _match_k(r, target)
    D = np.clip(k * r, .5, 1.6)
    t = np.round((D - .5) / 1.1 * 255).astype(np.uint8)
    return t, {'lum_mean_linear': float(L.mean()), 'sigma_m': info['sigma_lowpass_m'], 'k': k, 'clamped_mean': float(D.mean())}


def scale_match(detail8, normal_rgb8, W, ds, target, sigma_c_mm, gain_lightness, gain_slope):
    """Coarse-band gains on the donor lightness detail and normal map (both 8-bit arrays); returns new 8-bit
    arrays and stats."""
    D = .5 + 1.1 * detail8.astype(np.float64) / 255
    px = W / D.shape[0] * 1e3
    sig = sigma_c_mm / px
    f = np.log(1 + ds * (D - 1))
    lp = _gblur(f, sig)
    soft = np.exp(f.mean() + (f - lp) + gain_lightness * (lp - f.mean()))
    k = _match_k(None, target, .5, 2., fn=lambda k_: np.clip((soft * k_ - 1) / ds + 1, .5, 1.6))
    D2 = np.clip((soft * k - 1) / ds + 1, .5, 1.6)
    t = np.round((D2 - .5) / 1.1 * 255).astype(np.uint8)
    n = normal_rgb8.astype(np.float64) / 255 * 2 - 1
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    z = np.maximum(n[..., 2], .05)
    s2 = []
    for c in (-n[..., 0] / z, -n[..., 1] / z):
        lpc = _gblur(c, sig)
        s2.append(c.mean() + (c - lpc) + gain_slope * (lpc - c.mean()))
    n2 = np.stack([-s2[0], -s2[1], np.ones_like(s2[0])], -1)
    n2 /= np.linalg.norm(n2, axis=-1, keepdims=True)
    tilt = lambda a: float(np.median(np.degrees(np.arccos(np.clip(a[..., 2], -1, 1)))))
    return t, np.round((n2 * .5 + .5) * 255).clip(0, 255).astype(np.uint8), {
        'sigma_c_mm': sigma_c_mm, 'gain_lightness': gain_lightness, 'gain_slope': gain_slope, 'k_mean': k,
        'clamp_low_share': float((D2 <= .5).mean()), 'clamp_high_share': float((D2 >= 1.6).mean()),
        'normal_tilt_median_deg': [tilt(n), tilt(n2)]}


def fill_lum_mean(albedo, target):
    """(lum_mean for the shader, raw linear mean, k): k matches the clamped mean to target."""
    L = _lum(albedo)
    mu = float(L.mean())
    k = _match_k(None, target, fn=lambda k_: np.clip(k_ * L / mu, .5, 1.6))
    return mu / k, mu, k


def fill_codes(V, F, xfer_faces, riser_f, frame, donor_mesh, outward, slope_deg=60.):
    """Per-face fill code (float32; 0 outside the footprint faces too) and per-class areas."""
    from . import scanmesh
    cf, a1, a2, _, W, _ = frame
    don = scanmesh.load(Path(donor_mesh) / 'mesh_donor.npz')
    xi = np.flatnonzero(xfer_faces)
    T = V[F[xi]]
    cen = T.mean(1)
    X2 = np.c_[(cen - cf) @ a1, (cen - cf) @ a2]
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = .5 * np.linalg.norm(fn, axis=1)
    nout = np.asarray(outward, float)
    slope = np.degrees(np.arccos(np.clip(np.abs((fn / np.maximum(2 * area, 1e-300)[:, None]) @ nout), 0, 1)))
    inside = (np.abs(X2) < W / 2).all(1)
    he = np.full(len(xi), -np.inf)
    hw = np.full(len(xi), -np.inf)
    if inside.any():
        _, _, he[inside], hw[inside] = don.height(X2[inside])
    cls = np.zeros(len(xi), np.int8)
    cls[np.isfinite(he) & (he >= hw)] = 1
    cls[hw > he] = 2
    cls[riser_f[xi]] = 3
    fy = np.zeros(len(xi), np.float32)
    fy[cls == 3] = 2.
    fy[(cls == 2) | ((cls == 1) & (slope > slope_deg))] = 1.
    out = np.zeros(len(F), np.float32)
    out[xi] = fy
    names = {0: 'none', 1: 'envelope', 2: 'synthetic_wall', 3: 'riser'}
    stats = {'slope_deg': slope_deg, 'area_m2_by_source': {names[c]: float(area[cls == c].sum()) for c in range(4)},
             'area_m2_envelope_gt_slope': float(area[(cls == 1) & (slope > slope_deg)].sum()),
             'area_m2_by_fill_code': {str(int(c)): float(area[fy == c].sum()) for c in (0, 1, 2)}}
    return out, stats


def write_maps(out_dir, donor_dir, bed_record, ds, W, match):
    """Generate the lum-2 donor maps into out_dir; returns (paths, stats)."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = bed_clamped_mean(bed_record)
    d8, st = donor_detail(donor_dir, target)
    paths = {'detail': out_dir / 'donor_detail.png', 'normal': Path(donor_dir) / 'normal_dx.png'}
    stats = {'bed_clamped_mean': target, 'donor_detail': st}
    if match:
        nrm = np.asarray(Image.open(Path(donor_dir) / 'normal_dx.png').convert('RGB'))
        d8, n8, st2 = scale_match(d8, nrm, W, ds, target, float(match['sigma_mm']), float(match['gain_lightness']), float(match['gain_slope']))
        Image.fromarray(n8, 'RGB').save(out_dir / 'normal_dx.png')
        paths['normal'] = out_dir / 'normal_dx.png'
        stats['scale_match'] = st2
    Image.fromarray(d8, 'L').save(paths['detail'])
    return {k: str(v) for k, v in paths.items()}, stats


LUM3_SOFT_RANGE = (.25, 2.5)   # 8-bit linear encoding of the donor / fill high-part maps: soft = lo + (hi - lo) t


def _gblur_wrap(L, sig):
    """Gaussian of a tiling texture (periodic: FFT without padding)."""
    f0 = np.fft.fftfreq(L.shape[0])[:, None]
    f1 = np.fft.rfftfreq(L.shape[1])[None, :]
    H = np.exp(-2 * (np.pi * sig) ** 2 * (f0 ** 2 + f1 ** 2))
    return np.fft.irfft2(np.fft.rfft2(L) * H, s=L.shape)


def _rgb(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path).convert('RGB'), np.float64) / 255


def lum3_split(f, px_mm, sigma_c_mm, sigma_low_mm, blur):
    """(low, coarse, fine) = (G_sL f, G_sc f - G_sL f, f - G_sc f); the three sum to f exactly."""
    lc, ll = blur(f, sigma_c_mm / px_mm), blur(f, sigma_low_mm / px_mm)
    return ll, lc - ll, f - lc


def _slopes(n):
    z = np.maximum(n[..., 2], .05)
    return [n[..., 0] / z, n[..., 1] / z]


def _std2(parts):
    return float(np.sqrt(sum(float(np.var(x)) for x in parts)))


def _enc_soft(x, rng=LUM3_SOFT_RANGE):
    t = np.round((x - rng[0]) / (rng[1] - rng[0]) * 255)
    return np.clip(t, 0, 255).astype(np.uint8), {'clamp_low_share': float((t < 0).mean()), 'clamp_high_share': float((t > 255).mean())}


def _enc_normal(sx, sy):
    n = np.stack([sx, sy, np.ones_like(sx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.round((n * .5 + .5) * 255).clip(0, 255).astype(np.uint8)


def lum3_maps(out_dir, bed, donor_dir, donor_detail8, fill_albedo, fill_normal, fill_tile_m, ds, ns, uv_scale, W,
              sigma_low_mm=46.8, sigma_c_mm=4.0, neutral=False, mid_weight=0.):
    """'lum-3' maps into out_dir; returns (paths, stats). bed_low.png: D of soft_low = exp(G_sL f_bed) (8-bit D = .5 + 1.1 t;
    the bed tile, sampled with UV0); bed_low_normal.png: the bed normal map's slopes (x/z, y/z) low-passed, same
    encoding as the bed map; donor_soft.png / fill_soft.png: soft' = k exp(fine + g coarse) (LUM3_SOFT_RANGE), g
    matching the bed's coarse band, k the mean rule; donor_normal_dx.png / fill_normal_dx.png: slopes fine + g coarse
    (no low part), the bed target taken after the normal_strength lerp (the fill map is also drawn at normal_strength
    by the shader). donor_detail8: the donor's lum-2 lightness detail against this bed (donor_detail). neutral
    (implementation probe): soft' constant at the mean factor and flat donor / fill normals. mid_weight a > 0: the
    bed's own coarse band is kept at weight a and the donor / fill coarse band (gained as above) at the complementary
    b = sqrt(1 - a^2) (equal-variance split for independent patterns); extra maps bed_lowmid.png (D of exp(low + a
    coarse_bed), UV0, footprint core) and bed_mid_soft.png (soft of exp(a coarse_bed), LUM3_SOFT_RANGE, drawn in the
    bedding-frame triplanar on the walls); the mean rule then uses mean(exp(low + a coarse)) for the core and
    mean(soft_low) mean(bed_mid_soft) for the walls. Normals are unchanged."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LW = np.array([.2126, .7152, .0722])

    def logsoft(D):
        return np.log(1 + ds * (D - 1))
    Lb = _lin(_rgb(bed['albedo'])) @ LW
    fb = logsoft(np.clip(Lb / (np.asarray(bed['mean_linear']) @ LW), .5, 1.6))
    px_b = bed['tile_m'] / uv_scale / fb.shape[0] * 1e3
    low_b, coarse_b, fine_b = lum3_split(fb, px_b, sigma_c_mm, sigma_low_mm, _gblur_wrap)
    soft_bed, soft_low = np.exp(fb), np.exp(low_b)
    D_low = 1 + (soft_low - 1) / ds
    Image.fromarray(np.round((np.clip(D_low, .5, 1.6) - .5) / 1.1 * 255).astype(np.uint8), 'L').save(out_dir / 'bed_low.png')
    nb_raw = _rgb(bed['normal']) * 2 - 1
    Image.fromarray(_enc_normal(*[_gblur_wrap(c, sigma_low_mm / px_b) for c in _slopes(nb_raw)]), 'RGB').save(out_dir / 'bed_low_normal.png')
    nb_eff = (1 - ns) * np.array([0., 0., 1.]) + ns * nb_raw
    nb_eff /= np.linalg.norm(nb_eff, axis=-1, keepdims=True)
    target_L = float(coarse_b.std())
    target_N = _std2([lum3_split(c, px_b, sigma_c_mm, sigma_low_mm, _gblur_wrap)[1] for c in _slopes(nb_eff)])
    mean_factor = float(soft_bed.mean() / soft_low.mean())
    stats = {'sigma_low_mm': sigma_low_mm, 'sigma_c_mm': sigma_c_mm, 'bed': bed.get('id'), 'bed_texel_mm': px_b, 'neutral': bool(neutral),
             'bed_coarse_std_lightness': target_L, 'bed_coarse_std_slope': target_N, 'bed_fine_std_lightness': float(fine_b.std()),
             'bed_low_std_lightness': float(low_b.std()), 'mean_soft_bed': float(soft_bed.mean()), 'mean_soft_low': float(soft_low.mean()),
             'mean_factor': mean_factor, 'bed_low_D_clamp_share': float(((D_low < .5) | (D_low > 1.6)).mean()), 'soft_range': list(LUM3_SOFT_RANGE)}

    beta = float(np.sqrt(1 - mid_weight ** 2))
    base_mean = {'donor': float(soft_low.mean()), 'fill': float(soft_low.mean())}
    if mid_weight > 0:
        soft_lowmid, soft_mid = np.exp(low_b + mid_weight * coarse_b), np.exp(mid_weight * coarse_b)
        D_lm = 1 + (soft_lowmid - 1) / ds
        Image.fromarray(np.round((np.clip(D_lm, .5, 1.6) - .5) / 1.1 * 255).astype(np.uint8), 'L').save(out_dir / 'bed_lowmid.png')
        t_mid, cl_mid = _enc_soft(soft_mid)
        Image.fromarray(t_mid, 'L').save(out_dir / 'bed_mid_soft.png')
        base_mean = {'donor': float(soft_lowmid.mean()), 'fill': float(soft_low.mean() * soft_mid.mean())}
        stats.update(mid_weight=float(mid_weight), complementary_weight=beta, bed_lowmid_D_clamp_share=float(((D_lm < .5) | (D_lm > 1.6)).mean()),
                     bed_mid_soft_clamp=cl_mid, mean_soft_lowmid=float(soft_lowmid.mean()), mean_soft_mid=float(soft_mid.mean()))

    def lightness(f, px, blur, name):
        _, c, fi = lum3_split(f, px, sigma_c_mm, sigma_low_mm, blur)
        g = target_L / float(c.std())
        if mid_weight > 0:
            sp = np.exp(fi + beta * g * c)
            k = float(soft_bed.mean()) / (base_mean[name] * float(sp.mean()))
        else:
            sp = np.exp(fi + g * c)
            k = mean_factor / float(sp.mean())
        sp = np.full_like(sp, mean_factor) if neutral else sp * k
        t, cl = _enc_soft(sp)
        Image.fromarray(t, 'L').save(out_dir / ('%s_soft.png' % name))
        stats[name + '_lightness'] = {'texel_mm': px, 'coarse_std_before': float(c.std()), 'gain': g, 'fine_std': float(fi.std()), 'k': k, **cl}

    def slopes(n_raw, px, blur, name, strength):
        parts = [lum3_split(c, px, sigma_c_mm, sigma_low_mm, blur) for c in _slopes(n_raw)]
        cstd = _std2([p[1] for p in parts]) * strength
        g = target_N / cstd
        hs = [np.zeros_like(p[2]) if neutral else p[2] + g * p[1] for p in parts]
        Image.fromarray(_enc_normal(*hs), 'RGB').save(out_dir / ('%s_normal_dx.png' % name))
        stats[name + '_slope'] = {'coarse_std_before': cstd, 'gain': g, 'fine_std': _std2([p[2] for p in parts]) * strength}

    Dd = .5 + 1.1 * donor_detail8.astype(np.float64) / 255
    px_d = W / Dd.shape[0] * 1e3
    lightness(logsoft(Dd), px_d, _gblur, 'donor')
    slopes(_rgb(Path(donor_dir) / 'normal_dx.png') * 2 - 1, px_d, _gblur, 'donor', 1.)
    Lf = _lin(_rgb(fill_albedo)) @ LW
    ff = logsoft(np.clip(Lf / Lf.mean(), .5, 1.6))
    px_f = fill_tile_m / uv_scale / ff.shape[0] * 1e3
    lightness(ff, px_f, _gblur_wrap, 'fill')
    slopes(_rgb(fill_normal) * 2 - 1, px_f, _gblur_wrap, 'fill', ns)
    names = (('bed_low', 'bed_low.png'), ('bed_low_normal', 'bed_low_normal.png'), ('detail', 'donor_soft.png'),
             ('normal', 'donor_normal_dx.png'), ('fill_detail', 'fill_soft.png'), ('fill_normal', 'fill_normal_dx.png'))
    if mid_weight > 0:
        names += (('bed_lowmid', 'bed_lowmid.png'), ('bed_mid', 'bed_mid_soft.png'))
    return {k: str(out_dir / f) for k, f in names}, stats


def donor_fine_soft(donor_detail8, ds, W, bed_fine_std, ratio=.5, sigma_c_mm=4.0):
    """'lum-6' donor fine-band perturbation over the full bed lightness of the footprint core (lum-5): soft' =
    k exp(c fine_donor), fine_donor = f - G_sc f of the donor's lightness detail f = log(1 + ds (D - 1)) (reflect
    padding), c such that std(c fine_donor) = ratio x the bed's own fine-band std (lum3_maps
    'bed_fine_std_lightness'), k making mean(soft') = 1. 8-bit with LUM3_SOFT_RANGE; returns (uint8, stats)."""
    D = .5 + 1.1 * donor_detail8.astype(np.float64) / 255
    f = np.log(1 + ds * (D - 1))
    px = W / f.shape[0] * 1e3
    fine = f - _gblur(f, sigma_c_mm / px)
    c = ratio * float(bed_fine_std) / float(fine.std())
    sp = np.exp(c * fine)
    k = 1. / float(sp.mean())
    t, cl = _enc_soft(sp * k)
    return t, {'sigma_c_mm': sigma_c_mm, 'ratio': ratio, 'bed_fine_std': float(bed_fine_std), 'donor_fine_std': float(fine.std()),
               'gain': c, 'k': k, 'texel_mm': px, **cl}


def atlas_normal_split(out_dir, atlas_normal, texel_mm, sigma_low_mm=46.8, sigma_c_mm=4.0):
    """The donor atlas normal map split as lum3_maps' donor slopes: fine = f - G_sc f and coarse = G_sc f - G_sL f of
    each slope component (x/z, y/z), reflect padding. Encoded as two normal maps (_enc_normal of scale x slopes, the
    scales chosen so that the 99.9th percentile of |scale x slope| is 1.5; the shader divides them back). Returns
    (paths, stats with the coarse-band std at strength 1, the quantity lum3_maps' donor gain divides by)."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_raw = _rgb(atlas_normal) * 2 - 1
    parts = [lum3_split(c, texel_mm, sigma_c_mm, sigma_low_mm, _gblur) for c in _slopes(n_raw)]
    fine = [p[2] for p in parts]
    coarse = [p[1] for p in parts]
    del n_raw, parts
    sc = {}
    for name, (sx, sy) in (('fine', fine), ('coarse', coarse)):
        q = float(np.percentile(np.abs(np.concatenate([sx.ravel()[::7], sy.ravel()[::7]])), 99.9))
        sc[name] = 1.5 / max(q, 1e-6)
        Image.fromarray(_enc_normal(sx * sc[name], sy * sc[name]), 'RGB').save(out_dir / ('atlas_%s_normal.png' % name))
    stats = {'texel_mm': texel_mm, 'sigma_low_mm': sigma_low_mm, 'sigma_c_mm': sigma_c_mm, 'scale': sc,
             'coarse_std_slope': _std2(coarse), 'fine_std_slope': _std2(fine)}
    return {'fine': str(out_dir / 'atlas_fine_normal.png'), 'coarse': str(out_dir / 'atlas_coarse_normal.png')}, stats


def bed_wall_maps(out_dir, bed, fill_albedo, fill_normal, fill_tile_m, ds, ns, uv_scale, donor_coarse_std, sigma_low_mm=46.8,
                  sigma_c_mm=4.0, mid_weight=.7071067811865476):
    """lum3_maps for one bed texture without its donor part (the donor normals come from the atlas split; the
    donor coarse gain g = target_N / donor_coarse_std becomes a material constant): bed_low.png, bed_low_normal.png,
    bed_mid_soft.png (and bed_lowmid.png), fill_soft.png, fill_normal_dx.png -- the same formulas and encodings as
    lum3_maps with mid_weight (recipe 'lum-4' walls; 'lum-5' core uses the bed itself). Returns (paths, stats)."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LW = np.array([.2126, .7152, .0722])

    def logsoft(D):
        return np.log(1 + ds * (D - 1))
    Lb = _lin(_rgb(bed['albedo'])) @ LW
    fb = logsoft(np.clip(Lb / (np.asarray(bed['mean_linear']) @ LW), .5, 1.6))
    px_b = bed['tile_m'] / uv_scale / fb.shape[0] * 1e3
    low_b, coarse_b, fine_b = lum3_split(fb, px_b, sigma_c_mm, sigma_low_mm, _gblur_wrap)
    soft_bed, soft_low = np.exp(fb), np.exp(low_b)
    D_low = 1 + (soft_low - 1) / ds
    Image.fromarray(np.round((np.clip(D_low, .5, 1.6) - .5) / 1.1 * 255).astype(np.uint8), 'L').save(out_dir / 'bed_low.png')
    nb_raw = _rgb(bed['normal']) * 2 - 1
    Image.fromarray(_enc_normal(*[_gblur_wrap(c, sigma_low_mm / px_b) for c in _slopes(nb_raw)]), 'RGB').save(out_dir / 'bed_low_normal.png')
    nb_eff = (1 - ns) * np.array([0., 0., 1.]) + ns * nb_raw
    nb_eff /= np.linalg.norm(nb_eff, axis=-1, keepdims=True)
    target_L = float(coarse_b.std())
    target_N = _std2([lum3_split(c, px_b, sigma_c_mm, sigma_low_mm, _gblur_wrap)[1] for c in _slopes(nb_eff)])
    beta = float(np.sqrt(1 - mid_weight ** 2))
    soft_lowmid, soft_mid = np.exp(low_b + mid_weight * coarse_b), np.exp(mid_weight * coarse_b)
    D_lm = 1 + (soft_lowmid - 1) / ds
    Image.fromarray(np.round((np.clip(D_lm, .5, 1.6) - .5) / 1.1 * 255).astype(np.uint8), 'L').save(out_dir / 'bed_lowmid.png')
    t_mid, cl_mid = _enc_soft(soft_mid)
    Image.fromarray(t_mid, 'L').save(out_dir / 'bed_mid_soft.png')
    stats = {'bed': bed.get('id'), 'bed_texel_mm': px_b, 'bed_coarse_std_lightness': target_L, 'bed_coarse_std_slope': target_N,
             'mean_soft_bed': float(soft_bed.mean()), 'mean_soft_low': float(soft_low.mean()), 'mid_weight': float(mid_weight),
             'complementary_weight': beta, 'bed_mid_soft_clamp': cl_mid, 'soft_range': list(LUM3_SOFT_RANGE),
             'donor_coarse_std_slope': float(donor_coarse_std), 'donor_slope_gain': float(target_N / donor_coarse_std)}
    Lf = _lin(_rgb(fill_albedo)) @ LW
    ff = logsoft(np.clip(Lf / Lf.mean(), .5, 1.6))
    px_f = fill_tile_m / uv_scale / ff.shape[0] * 1e3
    _, c, fi = lum3_split(ff, px_f, sigma_c_mm, sigma_low_mm, _gblur_wrap)
    g = target_L / float(c.std())
    sp = np.exp(fi + beta * g * c)
    k = float(soft_bed.mean()) / (float(soft_low.mean() * soft_mid.mean()) * float(sp.mean()))
    t, cl = _enc_soft(sp * k)
    Image.fromarray(t, 'L').save(out_dir / 'fill_soft.png')
    stats['fill_lightness'] = {'texel_mm': px_f, 'coarse_std_before': float(c.std()), 'gain': g, 'fine_std': float(fi.std()), 'k': k, **cl}
    parts = [lum3_split(cc, px_f, sigma_c_mm, sigma_low_mm, _gblur_wrap) for cc in _slopes(_rgb(fill_normal) * 2 - 1)]
    cstd = _std2([p[1] for p in parts]) * ns
    gs = target_N / cstd
    Image.fromarray(_enc_normal(*[p[2] + gs * p[1] for p in parts]), 'RGB').save(out_dir / 'fill_normal_dx.png')
    stats['fill_slope'] = {'coarse_std_before': cstd, 'gain': gs, 'fine_std': _std2([p[2] for p in parts]) * ns}
    names = (('bed_low', 'bed_low.png'), ('bed_low_normal', 'bed_low_normal.png'), ('bed_lowmid', 'bed_lowmid.png'),
             ('bed_mid', 'bed_mid_soft.png'), ('fill_detail', 'fill_soft.png'), ('fill_normal', 'fill_normal_dx.png'))
    return {k_: str(out_dir / f) for k_, f in names}, stats
