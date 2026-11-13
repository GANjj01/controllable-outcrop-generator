"""Sun and Moon positions, Moon phase and clear-sky illuminance (no external dependencies).

Positions follow the low-precision algorithms of the Astronomical Almanac (Sun ~0.01 deg,
Moon ~0.3 deg), adequate for rendering. Azimuth is measured clockwise from true north,
elevation above the horizon, both in degrees, for a geographic latitude/longitude.
"""
from datetime import datetime, timedelta, timezone
import math

DEG = math.pi / 180


def julian_day(dt):
    """Julian day of an aware datetime (converted to UTC)."""
    dt = dt.astimezone(timezone.utc)
    return dt.timestamp() / 86400. + 2440587.5


def _horizontal(ra, dec, jd, lat, lon):
    d = jd - 2451545.0
    gmst = (280.46061837 + 360.98564736629 * d) % 360
    ha = (gmst + lon - ra / DEG) * DEG
    phi = lat * DEG
    sin_el = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha)
    el = math.asin(max(-1., min(1., sin_el)))
    az = math.atan2(-math.cos(dec) * math.sin(ha),
                    math.sin(dec) * math.cos(phi) - math.cos(dec) * math.sin(phi) * math.cos(ha))
    return (az / DEG) % 360, el / DEG


def _refraction(el):
    """Bennett atmospheric refraction (deg) added to the geometric elevation."""
    if el < -2:
        return 0.
    return 1.02 / math.tan((el + 10.3 / (el + 5.11)) * DEG) / 60.


def sun_ecliptic(jd):
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = (357.528 + 0.9856003 * n) % 360 * DEG
    lam = (L + 1.915 * math.sin(g) + .020 * math.sin(2 * g)) * DEG
    eps = (23.439 - 4e-7 * n) * DEG
    return lam, eps


def sun_position(dt, lat, lon):
    jd = julian_day(dt)
    lam, eps = sun_ecliptic(jd)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    az, el = _horizontal(ra, dec, jd, lat, lon)
    return az, el + _refraction(el)


def moon_position(dt, lat, lon):
    """(azimuth, elevation, illuminated fraction 0..1)."""
    jd = julian_day(dt)
    d = jd - 2451545.0
    L = (218.316 + 13.176396 * d) * DEG
    M = (134.963 + 13.064993 * d) * DEG
    F = (93.272 + 13.229350 * d) * DEG
    lam = L + 6.289 * DEG * math.sin(M)
    beta = 5.128 * DEG * math.sin(F)
    dist_km = 385001 - 20905 * math.cos(M)
    _, eps = sun_ecliptic(jd)
    ra = math.atan2(math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps), math.cos(lam))
    dec = math.asin(math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam))
    az, el = _horizontal(ra, dec, jd, lat, lon)
    # Topocentric parallax lowers the Moon by up to ~1 deg near the horizon.
    el -= math.degrees(math.asin(6378.14 / dist_km)) * math.cos(el * DEG)
    sun_lam, _ = sun_ecliptic(jd)
    elong = math.acos(max(-1., min(1., math.cos(beta) * math.cos(lam - sun_lam))))
    fraction = (1 - math.cos(elong)) / 2
    return az, el + _refraction(el), fraction


def airmass(el):
    """Kasten & Young (1989) relative air mass."""
    if el <= -.5:
        return float('inf')
    return 1 / (math.sin(max(el, -.5) * DEG) + .50572 * (max(el, -.5) + 6.07995) ** -1.6364)


def sun_illuminance(el, turbidity=1.):
    """Direct-normal illuminance (lux) of the Sun; fades through civil twilight."""
    if el < -.8:
        return 0.
    return 127500. * math.exp(-.21 * turbidity * airmass(max(el, .5))) * min(1., (el + .8) / 1.5)


def moon_illuminance(el, fraction, turbidity=1.):
    """Direct-normal moonlight (lux); ~0.27 lux for a full Moon at zenith."""
    if el < -.8 or fraction <= 0:
        return 0.
    phase_angle = math.degrees(math.acos(max(-1., min(1., 2 * fraction - 1))))  # 0 deg at full Moon
    # Lunar phase law (Allen): brightness falls much faster than the illuminated fraction.
    mag = -12.73 + .026 * phase_angle + 4e-9 * phase_angle ** 4
    zenith_full = .27
    rel = 10 ** (-.4 * (mag + 12.73))
    return zenith_full * rel * math.exp(-.21 * turbidity * (airmass(max(el, .5)) - 1)) * min(1., (el + .8) / 1.5)


def horizontal_illuminance(sun_el, sun_lux, moon_el, moon_lux):
    """Rough global horizontal illuminance (direct + sky) used for exposure compensation."""
    # Diffuse skylight: ~15 klux clear day, ~400 lux at sunset, ~3 lux at civil dusk (-6 deg),
    # ~0.01 lux at nautical dusk (-12 deg).
    if sun_el >= 0:
        sky = 15000. * math.sin(sun_el * DEG) ** .7 + 400.
    elif sun_el > -18:
        sky = 400. * math.exp(sun_el / 1.25) if sun_el > -6 else 3. * math.exp((sun_el + 6) / 1.1)
    else:
        sky = 0.
    direct = sun_lux * max(math.sin(sun_el * DEG), 0) + moon_lux * max(math.sin(moon_el * DEG), 0)
    return direct + sky + moon_lux * .2 + 1e-3  # starlight/airglow floor


def parse_local(text, utc_offset_h):
    """'YYYY-MM-DDTHH:MM' local time -> aware datetime."""
    return datetime.fromisoformat(text).replace(tzinfo=timezone(timedelta(hours=utc_offset_h)))


def scene_direction(azimuth_deg, elevation_deg, north_azimuth_deg):
    """Unit vector towards a body, in the scene ENU frame whose +y points to true azimuth
    `north_azimuth_deg` (clockwise from north)."""
    a = (azimuth_deg - north_azimuth_deg) * DEG
    e = elevation_deg * DEG
    return [math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)]
