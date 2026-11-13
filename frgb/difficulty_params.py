"""Each mechanism keeps 4 unit genes + 2 light genes in [0,1]; this module maps them to physically readable, nearly
separable coordinates and back (bijective on the kept region), and to the builder's physical parameters (the keys the
builders and renderers read): - shape: mean opening w (3-8 cm, linear),
aspect ratio a = depth / w uniform on its opening-dependent range [0.10 / w, 0.25 / w] (depth stays 10-25 cm), opening
variation (0-0.45), relative roughness rho = roughness / w in [0, 0.075] (roughness <= 6 mm for every w, the builder's
validated range); - stain: width (1 mm - 2 cm, log-uniform), contrast c = 1 - darkening factor (0.2-0.5), bend slope s
= 2 pi A / lambda in [0, 0.25] (amplitude A <= 0.2 m for every lambda), wavelength lambda (0.5-5 m, log-uniform); -
light: azimuth RELATIVE to the rock-face facing direction at the feature (compass azimuth of the horizontal outward
normal; 0 = sun in front of the face, +-90 = along the face strike), 0-360, and elevation 15-70 deg; each light gene
maps linearly onto its kept range after the exposure-rule exclusions (as cd_search.lighting). The relative azimuth
replaces "relative to the local trace strike": the stain carrier's trace is near vertical (trend undefined); the
in-surface angle between the light and the trace is reported per sample as a derived covariate instead."""
import math

import numpy as np

PARAMETRIZATION_R2 = dict(
    shape=dict(mean_opening_m=(.03, .08), depth_m=(.10, .25), opening_variation=(0., .45), relative_roughness=(0., .075)),
    stain=dict(width_m=(.001, .02), contrast=(.2, .5), bend_slope=(0., .25), wavelength_m=(.5, 5.)),
    light=dict(relative_azimuth_deg=(0., 360.), elevation_deg=(15., 70.)),
    log_uniform=('width_m', 'wavelength_m'),
    status='parameter ranges of the design (D2/D3)')
SHAPE_GENES = ('mean_opening_m', 'aspect_ratio', 'opening_variation', 'relative_roughness')
STAIN_GENES = ('width_m', 'contrast', 'bend_slope', 'wavelength_m')


def _unit(u):
    u = np.asarray(u, float)
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('unit genes must lie in [0,1]')
    return u


def _lin(u, lo, hi):
    return lo + (hi - lo) * u


def _ilin(x, lo, hi):
    return (x - lo) / (hi - lo)


def _log(u, lo, hi):
    return math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)


def _ilog(x, lo, hi):
    return (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))


def shape_forward(unit):
    u = _unit(unit)
    if u.shape != (4,): raise ValueError('shape needs 4 unit genes')
    b = PARAMETRIZATION_R2['shape']
    w = _lin(u[0], *b['mean_opening_m'])
    a_lo, a_hi = b['depth_m'][0] / w, b['depth_m'][1] / w
    a = _lin(u[1], a_lo, a_hi)
    rho = _lin(u[3], *b['relative_roughness'])
    return dict(physical=dict(mean_opening_m=w, opening_variation=_lin(u[2], *b['opening_variation']), depth_m=a * w, roughness_m=rho * w),
                analysis=dict(mean_opening_m=w, aspect_ratio=a, aspect_range=[a_lo, a_hi], opening_variation=_lin(u[2], *b['opening_variation']),
                              relative_roughness=rho))


def shape_inverse(physical):
    b = PARAMETRIZATION_R2['shape']
    w = float(physical['mean_opening_m'])
    a = float(physical['depth_m']) / w
    return np.array([_ilin(w, *b['mean_opening_m']), _ilin(a, b['depth_m'][0] / w, b['depth_m'][1] / w),
                     _ilin(float(physical['opening_variation']), *b['opening_variation']),
                     _ilin(float(physical['roughness_m']) / w, *b['relative_roughness'])])


def stain_forward(unit):
    u = _unit(unit)
    if u.shape != (4,): raise ValueError('stain needs 4 unit genes')
    b = PARAMETRIZATION_R2['stain']
    width, c = _log(u[0], *b['width_m']), _lin(u[1], *b['contrast'])
    s, lam = _lin(u[2], *b['bend_slope']), _log(u[3], *b['wavelength_m'])
    return dict(physical=dict(width_m=width, factor=1. - c, amplitude_m=s * lam / (2 * math.pi), wavelength_m=lam),
                analysis=dict(width_m=width, contrast=c, bend_slope=s, wavelength_m=lam))


def stain_inverse(physical):
    b = PARAMETRIZATION_R2['stain']
    lam = float(physical['wavelength_m'])
    return np.array([_ilog(float(physical['width_m']), *b['width_m']), _ilin(1. - float(physical['factor']), *b['contrast']),
                     _ilin(2 * math.pi * float(physical['amplitude_m']) / lam, *b['bend_slope']), _ilog(lam, *b['wavelength_m'])])


def kept(unit, bounds, excluded):
    """unit in [0,1] -> the kept part of [lo, hi] after removing the excluded intervals, linearly in increasing order
    (boundary points stay on the kept side), as cd_search._kept."""
    value = bounds[0] + (bounds[1] - bounds[0] - sum(hi - lo for lo, hi in excluded)) * float(unit)
    for lo, hi in sorted(excluded):
        if value > lo:
            value += hi - lo
    return value


def kept_inverse(value, bounds, excluded):
    removed = 0.
    for lo, hi in sorted(excluded):
        if lo < value < hi:
            raise ValueError('value %.6g lies in an excluded interval' % value)
        if value >= hi:
            removed += hi - lo
    return (value - removed - bounds[0]) / (bounds[1] - bounds[0] - sum(hi - lo for lo, hi in excluded))


def facing_azimuth_deg(outward_normal_enu):
    """compass azimuth (0 = north, 90 = east) of the horizontal projection of the outward face normal"""
    n = np.asarray(outward_normal_enu, float)
    h = math.hypot(n[0], n[1])
    if h < .2:
        raise ValueError('face too close to horizontal for a facing azimuth')
    return math.degrees(math.atan2(n[0], n[1])) % 360.


def light_forward(unit2, facing_az_deg, excluded_rel_az=(), excluded_el=()):
    u = _unit(unit2)
    b = PARAMETRIZATION_R2['light']
    rel = kept(u[0], b['relative_azimuth_deg'], excluded_rel_az)
    el = kept(u[1], b['elevation_deg'], excluded_el)
    return dict(azimuth_deg=(facing_az_deg + rel) % 360., elevation_deg=el, relative_azimuth_deg=rel)


def light_inverse(azimuth_deg, elevation_deg, facing_az_deg, excluded_rel_az=(), excluded_el=()):
    b = PARAMETRIZATION_R2['light']
    rel = (float(azimuth_deg) - facing_az_deg) % 360.
    return np.array([kept_inverse(rel, b['relative_azimuth_deg'], excluded_rel_az), kept_inverse(float(elevation_deg), b['elevation_deg'], excluded_el)])


def sun_vector(azimuth_deg, elevation_deg):
    """unit vector towards the sun, ENU, compass azimuth"""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])


def in_surface_light_angle_deg(sun, surface_normal, trace_tangent):
    """derived covariate: angle (0-90 deg) between the trace and the sun direction projected into the surface plane"""
    n = np.asarray(surface_normal, float); n = n / np.linalg.norm(n)
    t = np.asarray(trace_tangent, float); t = t - (t @ n) * n; t /= np.linalg.norm(t)
    p = np.asarray(sun, float) - (np.asarray(sun, float) @ n) * n
    if np.linalg.norm(p) < 1e-9:
        return None
    return math.degrees(math.acos(min(1., abs(float(p @ t)) / float(np.linalg.norm(p)))))
