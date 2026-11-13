"""Formal lighting-search technical rules.

Model-free: only RGB, labels and the rock mask enter. - exposure_gate: on every rendered frame, rock pixels with any
channel <= 2 or >= 253 must be <= 5% and the mean rock luma (Rec.709 weights on 8-bit sRGB / 255) must lie in [0.15,
0.85]; otherwise the candidate is a technical failure. - visibility_u (shape positives): at each positive pixel the
local label normal comes from the structure tensor of the binary target (gradient sigma 1, integration sigma 3); luma
is sampled bilinearly at +-3 px along it; a positive with |dL| < 10/255 (or a sample outside the image/rock) becomes U
(ignored, not negative). - sun_source_record: the Sun's angular diameter is fixed at 0.5357 deg (real Sun), not
searched; an explicit request must be read back on the saved lit map."""
import numpy as np
import scipy.ndimage as ndi

CLIP_LOW, CLIP_HIGH = 2, 253
CLIP_FRACTION_MAX = 0.05
LUMA_RANGE = (0.15, 0.85)
U_OFFSET_PX = 3.0
U_DELTA_MIN = 10 / 255
GRAD_SIGMA, TENSOR_SIGMA = 1.0, 3.0
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
SUN_ANGLE_TOLERANCE_DEG = 1e-6  # the engine stores the angle as float32


def sun_source_record(request, readiness):
    """Explicit Sun angular diameter vs. the saved lit map's readback (readiness 'sun_source' rows)."""
    requested = request.get('sun_source_angle_deg')
    saved = [row for row in readiness.get('sun_source', []) if row.get('mode') == 'Lit']
    value = saved[0]['source_angle_deg'] if len(saved) == 1 else None
    soft = saved[0].get('soft_angle_deg') if len(saved) == 1 else None
    ok = requested is None or (value is not None and abs(value - requested) <= SUN_ANGLE_TOLERANCE_DEG and soft == 0.)
    return dict(requested_deg=requested, saved_deg=value, saved_soft_deg=soft, explicit=requested is not None, ok=bool(ok))


def luma(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('uint8 HWC RGB required')
    return (rgb.astype(np.float64) / 255.) @ LUMA_WEIGHTS


def exposure_gate(rgb, rock):
    rgb = np.asarray(rgb); rock = np.asarray(rock).astype(bool)
    if rock.shape != rgb.shape[:2] or not rock.any():
        raise ValueError('non-empty rock mask on the RGB grid required')
    px = rgb[rock]
    clipped = float(((px <= CLIP_LOW) | (px >= CLIP_HIGH)).any(1).mean())
    mean = float(luma(rgb)[rock].mean())
    ok = clipped <= CLIP_FRACTION_MAX and LUMA_RANGE[0] <= mean <= LUMA_RANGE[1]
    return dict(passed=bool(ok), clipped_fraction=clipped, mean_luma=mean, rock_pixels=int(rock.sum()),
                rule=dict(clip_low=CLIP_LOW, clip_high=CLIP_HIGH, clip_fraction_max=CLIP_FRACTION_MAX, luma_range=list(LUMA_RANGE)))


def label_normals(target):
    """Unit normals (ny, nx) at every pixel from the structure tensor of the binary label."""
    t = np.asarray(target).astype(np.float64)
    gy = ndi.gaussian_filter(t, GRAD_SIGMA, order=(1, 0))
    gx = ndi.gaussian_filter(t, GRAD_SIGMA, order=(0, 1))
    jyy, jxx, jxy = (ndi.gaussian_filter(a, TENSOR_SIGMA) for a in (gy * gy, gx * gx, gx * gy))
    # Dominant eigenvector of [[jxx, jxy], [jxy, jyy]] (x, y order): angle = 0.5 * atan2(2 jxy, jxx - jyy).
    angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return np.sin(angle), np.cos(angle)


def visibility_u(rgb, target, rock=None):
    target = np.asarray(target).astype(bool)
    L = luma(rgb)
    if target.shape != L.shape:
        raise ValueError('target must lie on the RGB grid')
    rock = np.ones_like(target) if rock is None else np.asarray(rock).astype(bool)
    ys, xs = np.nonzero(target)
    ny, nx = label_normals(target)
    ny, nx = ny[ys, xs], nx[ys, xs]
    h, w = L.shape
    deltas = []
    inside = np.ones(len(ys), bool)
    for s in (1, -1):
        py, px = ys + s * U_OFFSET_PX * ny, xs + s * U_OFFSET_PX * nx
        ok = (py >= 0) & (py <= h - 1) & (px >= 0) & (px <= w - 1)
        ok[ok] &= rock[np.rint(py[ok]).astype(int), np.rint(px[ok]).astype(int)]
        inside &= ok
        deltas.append(ndi.map_coordinates(L, [np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)], order=1, mode='nearest'))
    delta = np.abs(deltas[0] - deltas[1])
    u_pos = (~inside) | (delta < U_DELTA_MIN)
    u = np.zeros_like(target)
    u[ys[u_pos], xs[u_pos]] = True
    return dict(u=u, positives=int(len(ys)), u_pixels=int(u_pos.sum()), outside_or_off_rock=int((~inside).sum()),
                u_fraction=float(u_pos.mean()) if len(ys) else None, delta_255=delta[inside] * 255,
                rule=dict(offset_px=U_OFFSET_PX, delta_min_255=U_DELTA_MIN * 255, grad_sigma=GRAD_SIGMA, tensor_sigma=TENSOR_SIGMA))
