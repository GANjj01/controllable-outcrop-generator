"""Astronomical lighting plans: where the Sun and Moon are for every shot, and how bright.

Three modes, all deterministic for a given seed:

The outcrop's geology (strike/dip) is fixed in the scene frame; only the sky moves."""
from datetime import datetime, timedelta
import math

from .sky import (horizontal_illuminance, moon_illuminance, moon_position, parse_local, scene_direction,
                  sun_illuminance, sun_position)

SUN_TOA_LUX = 120000.
MOON_TOA_FULL_LUX = .32

DEFAULT_LIGHTING = {
    'mode': 'random',
    'latitude_deg': [-55., 55.],
    'longitude_deg': [-180., 180.],
    'year': 2026,
    'north_azimuth_deg': [0., 360.],
    'day_fraction': .75,
    'twilight_fraction': .12,
    'turbidity': [1., 2.],
    # auto: camera-like histogram metering over a wide EV100 range (night shots come out bright,
    #       like a long exposure); auto_clamped: metering cannot brighten below the floor
    #       illuminance, so night stays dark; fixed: one manual bias for every shot.
    'exposure_mode': 'auto_clamped',
    'exposure_floor_lux': 3.,
    'exposure_offset_ev': 0.,
    'exposure_jitter_ev': [-.5, .5],
    'fixed_exposure_bias': -14.6,       # fixed: one bias for every shot (true day/night contrast)
    'cameras': None,                    # timeseries/explicit: number of fixed viewpoints
}


def _pick(rng, value):
    return float(rng.uniform(*value)) if isinstance(value, (list, tuple)) else float(value)


def _day_window(date, lat, lon, utc_offset):
    """Local sunrise/sunset (datetimes) by scanning elevation; None for polar day/night."""
    start = parse_local(date + 'T00:00', utc_offset)
    els = [sun_position(start + timedelta(minutes=m), lat, lon)[1] for m in range(0, 1441, 10)]
    up = [e > -.833 for e in els]
    if all(up) or not any(up):
        return None, all(up)
    rise = next(i for i in range(1, len(up)) if up[i] and not up[i - 1]) if not up[0] else 0
    set_ = next((i for i in range(rise + 1, len(up)) if not up[i] and up[i - 1]), len(up) - 1)
    return (start + timedelta(minutes=10 * rise), start + timedelta(minutes=10 * set_)), True


def _state(dt, lat, lon, north, turbidity):
    s_az, s_el = sun_position(dt, lat, lon)
    m_az, m_el, frac = moon_position(dt, lat, lon)
    s_lux = sun_illuminance(s_el, turbidity)
    m_lux = moon_illuminance(m_el, frac, turbidity)
    # Sun keeps lighting the sky through civil/nautical twilight; it is switched off below -12 deg.
    sun_ue = SUN_TOA_LUX * min(1., max(0., (s_el + 12.) / 6.))
    moon_ue = MOON_TOA_FULL_LUX * (m_lux / max(moon_illuminance(90, 1.), 1e-9)) if m_el > -1 else 0.
    return {
        'time_local': dt.isoformat(), 'latitude_deg': lat, 'longitude_deg': lon, 'north_azimuth_deg': north,
        'sun_azimuth_deg': s_az, 'sun_elevation_deg': s_el, 'sun_direct_lux': s_lux,
        'moon_azimuth_deg': m_az, 'moon_elevation_deg': m_el, 'moon_illuminated_fraction': frac, 'moon_direct_lux': m_lux,
        'horizontal_lux': horizontal_illuminance(s_el, s_lux, m_el, m_lux),
        'to_sun_scene': scene_direction(s_az, s_el, north), 'to_moon_scene': scene_direction(m_az, m_el, north),
        'sun_ue_lux': sun_ue, 'moon_ue_lux': moon_ue,
        'period': 'day' if s_el > 0 else 'twilight' if s_el > -12 else 'night',
    }


def _exposure(state, cfg, rng):
    """(bias, EV100 metering range or None for manual)."""
    jitter = _pick(rng, cfg['exposure_jitter_ev'])
    if cfg['exposure_mode'] == 'fixed':
        return float(cfg['fixed_exposure_bias']) + jitter, None
    # Incident-light EV100 = log2(E / 2.5) (calibration constant C = 250).
    low = -6. if cfg['exposure_mode'] == 'auto' else math.log2(float(cfg['exposure_floor_lux']) / 2.5)
    return float(cfg['exposure_offset_ev']) + jitter, [low, 20.]


def plan(cfg, rng, shots):
    """Return (lighting states, camera index per shot, number of cameras)."""
    cfg = {**DEFAULT_LIGHTING, **(cfg or {})}
    mode = cfg['mode']
    lat, lon = _pick(rng, cfg['latitude_deg']), _pick(rng, cfg['longitude_deg'])
    utc = float(cfg['utc_offset_h']) if 'utc_offset_h' in cfg else round(lon / 15.)
    north = _pick(rng, cfg['north_azimuth_deg']) % 360
    turbidity = _pick(rng, cfg['turbidity'])
    if mode == 'random':
        day = int(rng.integers(0, 365))
        date = (datetime(int(cfg['year']), 1, 1) + timedelta(days=day)).strftime('%Y-%m-%d')
        window, has_day = _day_window(date, lat, lon, utc)
        midnight = parse_local(date + 'T00:00', utc)
        times = []
        for _ in range(shots):
            r = rng.random()
            if window is None:
                t = midnight + timedelta(minutes=float(rng.uniform(0, 1440)))
            elif r < cfg['day_fraction']:
                rise, set_ = window
                span = max((set_ - rise).total_seconds() / 60 - 60, 1)
                t = rise + timedelta(minutes=30 + float(rng.uniform(0, span)))
            elif r < cfg['day_fraction'] + cfg['twilight_fraction']:
                edge = window[0] if rng.random() < .5 else window[1]
                t = edge + timedelta(minutes=float(rng.uniform(-50, 30)) * (1 if edge == window[0] else -1))
            else:
                # Night: between sunset+90 min and next sunrise-90 min (wrapped onto the same date).
                rise, set_ = window
                night = 1440 - (set_ - rise).total_seconds() / 60 - 180
                t = set_ + timedelta(minutes=90 + float(rng.uniform(0, max(night, 1))))
                t -= timedelta(days=(t - midnight).days)
            times.append(t)
        cameras = shots
        cam_index = list(range(shots))
    else:
        if mode == 'timeseries':
            start = parse_local(cfg['date'] + 'T' + cfg['start'], utc)
            end = parse_local(cfg['date'] + 'T' + cfg['end'], utc)
            if end <= start:
                end += timedelta(days=1)
            step = timedelta(minutes=float(cfg['step_min']))
            times = []
            t = start
            while t <= end:
                times.append(t)
                t += step
        elif mode == 'explicit':
            times = [parse_local(x, utc) for x in cfg['times']]
        else:
            raise ValueError('lighting mode must be random, timeseries or explicit')
        cameras = int(cfg['cameras'] or 1)
        cam_index = [c for c in range(cameras) for _ in times]
        times = [t for _ in range(cameras) for t in times]
    states = []
    for t in times:
        st = _state(t, lat, lon, north, turbidity)
        st['exposure_bias'], st['auto_exposure_ev100'] = _exposure(st, cfg, rng)
        st['exposure_mode'] = cfg['exposure_mode']
        st['utc_offset_h'] = utc
        states.append(st)
    return states, cam_index, cameras
