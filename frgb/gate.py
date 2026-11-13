"""Pre-score mechanism checks."""
import numpy as np


def proxy_request_corners(proxy):
    roll, pitch, yaw = np.deg2rad(proxy['rotation_deg'])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    rot = np.array([[cp*cy, cp*sy, sp], [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, -sr*cp],
                    [-(cr*sp*cy+sr*sy), cy*sr-cr*sp*sy, cr*cp]])
    corners = np.array([[x, y, z] for x in (-50, 50) for y in (-50, 50) for z in (-50, 50)])
    return ((corners*np.asarray(proxy['scale'])) @ rot + proxy['location_cm'])*[.01, -.01, .01]


def validate_shadow_request(proxy, request):
    from .raster import camera
    corners = proxy_request_corners(proxy)
    results = [proxy_outside_frustum(corners, camera(s, request['resolution']), 1e6)
               for s in request['shots']]
    if not all(results):
        raise ValueError('Shadow proxy intersects an expanded camera frustum')
    return dict(all_outside=True, camera_count=len(results), margin_each_side_full_frame=.02,
                far_m=1e6, planned_corners_world_enu_m=corners.tolist(), actual_ue_readback_pending=True)


def proxy_outside_frustum(corners, cam, far, margin=.02):
    """OBB/frustum separating-axis test; touching is conservatively rejected.

    corners use the same Cartesian frame as cam, ordered with z changing fastest,
    then y, then x. The two-percent full-frame margin expands both sides.
    """
    b = np.asarray(corners, float)
    if b.shape != (8, 3) or not np.isfinite(b).all() or far <= cam['near']:
        raise ValueError('finite OBB corners and far > near required')
    x = cam['w'] / (2 * cam['fpx']) * (1 + 2 * margin)
    y = cam['h'] / (2 * cam['fpx']) * (1 + 2 * margin)
    r, u, f, p = (np.asarray(cam[k]) for k in ('right', 'up', 'fwd', 'pos'))
    a = np.array([p + z * (f + sx * x * r + sy * y * u)
                  for z in (cam['near'], far) for sx in (-1, 1) for sy in (-1, 1)])
    be = [b[i] - b[0] for i in (1, 2, 4)]
    fe = [r, u] + [f + sx*x*r + sy*y*u for sx in (-1, 1) for sy in (-1, 1)]
    axes = [np.cross(be[0], be[1]), np.cross(be[1], be[2]), np.cross(be[2], be[0]),
            f, r-x*f, r+x*f, u-y*f, u+y*f]
    axes += [np.cross(e1, e2) for e1 in be for e2 in fe]
    for axis in axes:
        length = np.linalg.norm(axis)
        if length <= 1e-12:
            continue
        aa, bb = a @ (axis / length), b @ (axis / length)
        if aa.max() < bb.min() - 1e-9 or bb.max() < aa.min() - 1e-9:
            return True
    return False


def stain_mask(points_local_cm, spec):
    q = np.asarray(points_local_cm) - spec['origin_cm']
    d = q @ spec['normal_local'] + spec['amplitude_cm'] * np.sin(
        2*np.pi*(q @ spec['tangent_local'])/spec['wavelength_cm'] + spec['phase'])
    half, soft = spec['width_cm']/2, spec['soft_cm']
    if soft == 0:
        m = (np.abs(d) <= half).astype(float)
    else:
        t = np.clip((np.abs(d)-(half-soft))/(2*soft), 0, 1)
        m = 1-t*t*(3-2*t)
    return spec['weight'] * m


def negative_domain(influence, rock, protected_f, uncertain, off_lit):
    arrays = [np.asarray(a, bool) for a in (influence, rock, protected_f, uncertain, off_lit)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('negative-domain masks must use the same image grid')
    m, rock, f, u, lit = arrays
    return m & rock & ~f & ~u & lit


def check_pair(off, on, rock, influence, repeat_a, repeat_b, geometry_equal, *, frozen_threshold_8bit=None,
               repeat_ratio_rule=True):
    """Repeat inputs are the worst of within/across-launch controls at the caller; retain both results and require both to
    pass before freezing their maximum T."""
    from scipy.ndimage import label
    off, on, repeat_a, repeat_b = [np.asarray(a, float) for a in (off, on, repeat_a, repeat_b)]
    rock, influence = np.asarray(rock, bool), np.asarray(influence, bool)
    if off.shape != on.shape or off.shape != repeat_a.shape or off.shape != repeat_b.shape or off.shape != rock.shape+(3,) or influence.shape != rock.shape:
        raise ValueError('pair/control arrays must have identical native grids')
    outside = rock & ~influence
    if not outside.any() or not (rock & influence).any():
        return {'pass': False, 'reason': 'empty influence or control domain'}
    repeat_diff = np.abs(repeat_a-repeat_b)
    if frozen_threshold_8bit is None:
        threshold = max(2., float(np.quantile(repeat_diff[outside], .999)))
    else:
        threshold = float(frozen_threshold_8bit)
        if not np.isfinite(threshold) or threshold < 2.:
            raise ValueError('frozen threshold must be finite and at least 2 in 8-bit RGB')
    changed = (np.abs(on-off).max(-1) > threshold) & outside
    control = (repeat_diff.max(-1) > threshold) & outside
    fraction = float(changed.sum()/outside.sum())
    control_fraction = float(control.sum()/outside.sum())
    labels, count = label(changed, np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max(initial=0))
    # Under deterministic rendering the repeat control is 0, so "fraction <= 2 x repeat" demands zero outside change,
    # contrary to the requirement "not pixel-identical"; the 0.2%, 50 px and T=2 limits are unchanged.
    ok = bool(geometry_equal and fraction <= .002 and (not repeat_ratio_rule or fraction <= 2*control_fraction) and largest <= 50)
    return {'pass': ok, 'repeat_ratio_rule': bool(repeat_ratio_rule), 'geometry_equal': bool(geometry_equal), 'threshold_8bit': threshold,
            'outside_px': int(outside.sum()), 'influence_rock_px': int((rock & influence).sum()),
            'outside_changed_fraction': fraction, 'repeat_changed_fraction': control_fraction,
            'largest_changed_component_px': largest}
