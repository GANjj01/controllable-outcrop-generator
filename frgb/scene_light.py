"""Per-camera light domain of the multi-scene design.

The prior is uniform (in degrees) over the accepted cells. The two light genes (u_az, u_el) in [0, 1]^2 map onto it by
the Rosenblatt transform: u_el -> elevation by the inverse CDF of the elevation marginal (each elevation cell weighted
by its number of accepted azimuth cells, uniform inside the cell), then u_az -> relative azimuth by the inverse CDF of
the azimuths accepted in that elevation cell (uniform inside each accepted cell). The map is piecewise linear, measure
preserving and monotone in each coordinate, so a stratified (Sobol) design stays stratified over the accepted set.
density() gives the prior density for importance weights (1 / accepted area inside, 0 outside)."""
import numpy as np

AZ_EDGES = np.arange(0., 360.01, 15.)
EL_EDGES = np.arange(15., 70.01, 5.)


def _check(accept):
    a = np.asarray(accept, bool)
    if a.shape != (len(AZ_EDGES) - 1, len(EL_EDGES) - 1):
        raise ValueError('accept must be (24, 11) over (relative azimuth, elevation)')
    if not a.any():
        raise ValueError('camera has no accepted light cell')
    return a


def map_light(u_az, u_el, accept):
    """unit genes -> (relative azimuth deg, elevation deg) uniform over the accepted cells"""
    a = _check(accept); u_az, u_el = np.atleast_1d(np.asarray(u_az, float)), np.atleast_1d(np.asarray(u_el, float))
    if np.any((u_az < 0) | (u_az > 1) | (u_el < 0) | (u_el > 1)): raise ValueError('genes must lie in [0, 1]')
    w_el = a.sum(0).astype(float); cdf = np.r_[0., np.cumsum(w_el)] / w_el.sum()
    j = np.clip(np.searchsorted(cdf, u_el, side='right') - 1, 0, len(w_el) - 1)
    j = np.where(w_el[j] > 0, j, np.array([np.flatnonzero(w_el > 0)[np.argmin(np.abs(np.flatnonzero(w_el > 0) - jj))] for jj in j]))
    t = np.clip((u_el - cdf[j]) / np.maximum(cdf[j + 1] - cdf[j], 1e-300), 0., 1.)
    el = EL_EDGES[j] + t * (EL_EDGES[j + 1] - EL_EDGES[j])
    az = np.empty_like(el)
    for k, (jj, uu) in enumerate(zip(j, u_az)):
        cells = np.flatnonzero(a[:, jj]); x = uu * len(cells); i = min(int(x), len(cells) - 1)
        az[k] = AZ_EDGES[cells[i]] + (x - i) * 15.
    return az, el


def density(rel_az, el, accept):
    """prior density (per deg^2) of the camera's light domain at the given points"""
    a = _check(accept); rel_az, el = np.atleast_1d(rel_az) % 360., np.atleast_1d(el)
    i = np.clip((rel_az // 15.).astype(int), 0, 23); j = np.clip(((el - 15.) // 5.).astype(int), 0, 10)
    inside = (el >= 15.) & (el <= 70.) & a[i, j]
    return np.where(inside, 1. / (a.sum() * 75.), 0.)


FALLBACK_SIN = float(np.sin(np.radians(10.)))


def compass_deg(v):
    return float(np.degrees(np.arctan2(v[0], v[1])) % 360.)


def strike_reference(plane_normal, trace_tangent=None):
    """m (m_z >= 0) -> s = normalize(z x m) (right-hand rule: the plane dips to the right of s), alpha = compass azimuth of s;
    m_z == 0: the side with alpha in [0, 180). |m_h| < sin 10 deg: horizontal projection of the trace tangent, alpha in [0, 180)."""
    m = np.asarray(plane_normal, float); m = m / np.linalg.norm(m)
    if m[2] < 0: m = -m
    mh = float(np.hypot(m[0], m[1]))
    if mh >= FALLBACK_SIN:
        az = compass_deg(np.array([-m[1], m[0]]) / mh)
        if m[2] == 0. and az >= 180.: az -= 180.
        kind = 'plane_strike'
    else:
        if trace_tangent is None: raise ValueError('plane near horizontal and no trace tangent for the fallback')
        t = np.asarray(trace_tangent, float)
        if np.hypot(t[0], t[1]) < 1e-3 * np.linalg.norm(t): raise ValueError('plane near horizontal and trace near vertical: no reference')
        az = compass_deg(t) % 180.
        kind = 'trace_horizontal'
    return dict(azimuth_deg=float(az), kind=kind, plane_normal_up=m.tolist(), plane_dip_deg=float(np.degrees(np.arccos(np.clip(m[2], -1, 1)))))


def toward_sun(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])


def light_covariates(camera, az_deg, el_deg):
    """Per-frame light covariates of a camera_plan_r2 camera record under a Sun at (compass az, el)."""
    s = toward_sun(az_deg, el_deg)
    pos, tgt = np.asarray(camera['shot']['position_enu_m'], float), np.asarray(camera['shot']['target_enu_m'], float)
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos); n = np.asarray(camera['normal'], float); n = n / np.linalg.norm(n)
    ref = camera.get('light_reference')
    out = dict(sun_azimuth_deg=float(az_deg) % 360., sun_elevation_deg=float(el_deg),
               relative_azimuth_deg=None if not ref else float((az_deg - ref['azimuth_deg']) % 360.),
               reference_kind=None if not ref else ref['kind'], incidence_cos=float(n @ s),
               phase_angle_deg=float(np.degrees(np.arccos(np.clip(-fwd @ s, -1, 1)))))
    fh = np.hypot(fwd[0], fwd[1])
    out['sun_minus_view_azimuth_deg'] = float((az_deg - compass_deg(fwd)) % 360.) if fh > np.sin(np.radians(1.)) else None
    nh = np.hypot(n[0], n[1])
    out['facing_relative_azimuth_deg'] = float((az_deg - compass_deg(n)) % 360.) if nh > 1e-6 else None
    t = (camera.get('covariates') or {}).get('trace_tangent')
    if t is not None:
        t = np.asarray(t, float); ps, pt = s - (s @ n) * n, t - (t @ n) * n
        if np.linalg.norm(ps) > 1e-9 and np.linalg.norm(pt) > 1e-9:
            c = abs(ps @ pt) / np.linalg.norm(ps) / np.linalg.norm(pt)
            out['in_surface_light_trace_angle_deg'] = float(np.degrees(np.arccos(np.clip(c, 0, 1))))
    return out
