"""Client side of the resident Blender worker.

Pure numpy except exr_channels. Pieces: - light(az, el, table): Sun lamp strength/colour from the frozen Nishita
calibration table (same atmosphere model as the sky), toward-Sun ENU vector, Nishita sky parameters; sun:sky ratio is
the model's own at that elevation. - to_srgb8(linear, k, gains): fixed exposure k, daylight white-balance gains, clip,
sRGB OETF, round half to even. - Worker: start / submit / wait / stop of `blender -b --python worker.py -- QUEUE
--samples N`."""
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(Path(__file__).resolve().parents[2]))
from frgb import runconfig as rc  # noqa: E402
BLENDER = rc.path('blender')
WORKER = HERE/'worker.py'
SUN_ANGLE_DEG = 0.5357
LUMA = np.array([.2126, .7152, .0722])
SKY = dict(altitude_m=0., air_density=1., dust_density=1., ozone_density=1., strength=1.)


def toward_sun_enu(azimuth_deg, elevation_deg):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return [math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)]


def ue_ray(azimuth_deg, elevation_deg):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return [-math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), -math.sin(el)]


def load_table(path):
    data = json.loads(Path(path).read_text(encoding='utf8'))
    rows = sorted(data['table'], key=lambda r: r['elevation_deg'])
    return dict(meta=data['meta'], elevation=np.array([r['elevation_deg'] for r in rows]),
                sun=np.array([r['sun_normal_irradiance_rgb'] for r in rows]),
                sky=np.array([r['sky_horizontal_irradiance_rgb'] for r in rows]), path=str(path))


def interp_rgb(table, key, elevation_deg):
    e = table['elevation']
    if not e[0] <= elevation_deg <= e[-1]:
        raise ValueError('elevation %.3f outside the calibration table [%g, %g]' % (elevation_deg, e[0], e[-1]))
    return np.array([np.interp(elevation_deg, e, table[key][:, c]) for c in range(3)])


def sun_rgb(table, elevation_deg):
    """Transmitted solar disc normal irradiance (rgb, Blender units) at this elevation."""
    return interp_rgb(table, 'sun', elevation_deg)


def light(azimuth_deg, elevation_deg, table, sun_rotation_offset_rad=0.):
    e_n = sun_rgb(table, elevation_deg)
    y = float(e_n @ LUMA)
    to_sun = toward_sun_enu(azimuth_deg, elevation_deg)
    # Nishita sun_rotation: angle about +Z of the sky's own Sun; measured convention (worker acceptance test) is
    # rotation = atan2(x, y) of the toward-Sun vector plus a fixed offset.
    rotation = math.atan2(to_sun[0], to_sun[1]) + sun_rotation_offset_rad
    return dict(azimuth_deg=float(azimuth_deg), elevation_deg=float(elevation_deg), toward_sun_enu=to_sun,
                angle_deg=SUN_ANGLE_DEG, strength=y, color=(e_n / y).tolist(), sun_irradiance_rgb=e_n.tolist(),
                sky=dict(SKY, sun_rotation_rad=rotation), table=table['path'])


def white_balance_gains(light_record):
    """Daylight white balance keyed to the actual Sun colour: a gray surface in direct sun renders neutral."""
    c = np.asarray(light_record['color'], float)
    return (1. / c).tolist()


def exposure_k(table, reference_elevation_deg=45.):
    """Incident-light exposure: a white horizontal Lambertian surface at the reference elevation (sun + sky, after the
    daylight white balance at that elevation) has luminance 1.0 in linear output; one k for the whole study."""
    e_n = sun_rgb(table, reference_elevation_deg); y = float(e_n @ LUMA); c = e_n / y
    e_h = e_n * math.sin(math.radians(reference_elevation_deg)) + interp_rgb(table, 'sky', reference_elevation_deg)
    radiance = e_h / math.pi
    return float(1. / ((radiance / c) @ LUMA))


def srgb_oetf(x):
    x = np.clip(x, 0., 1.)
    return np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - .055)


def to_srgb8(linear_rgb, k, gains):
    lin = np.asarray(linear_rgb, np.float64)[..., :3] * float(k) * np.asarray(gains, float)
    return np.round(srgb_oetf(lin) * 255.).astype(np.uint8)


def exr_channels(path):
    import OpenEXR
    with OpenEXR.File(str(path)) as f:
        return {name: np.asarray(ch.pixels) for part in f.parts for name, ch in part.channels.items()}


def decode(path):
    """Combined linear RGB (H,W,3 float64), Depth (H,W), IndexMA (H,W int), optional AOV stain_m; rows top-down."""
    ch = exr_channels(path)
    def find(suffix):
        names = [n for n in ch if n.endswith(suffix)]
        return ch[names[0]] if len(names) == 1 else None
    combined = find('Combined')
    if combined is None:
        rgba = [find('Combined.' + c) for c in 'RGBA']
        combined = np.stack(rgba, -1) if all(x is not None for x in rgba) else None
    if combined is None:
        raise ValueError('no Combined pass in %s: %s' % (path, sorted(ch)))
    depth = next((x for x in (find('Depth.Z'), find('Depth.V'), find('Depth')) if x is not None), None)
    index = find('IndexMA.X') if find('IndexMA.X') is not None else find('IndexMA')
    aov = next((x for x in (find('stain_m.X'), find('stain_m.V'), find('stain_m')) if x is not None), None)
    out = dict(rgb=np.asarray(combined, np.float64)[..., :3])
    if depth is not None: out['depth'] = np.asarray(depth, np.float64).reshape(out['rgb'].shape[:2])
    if index is not None: out['index'] = np.rint(np.asarray(index, np.float64)).astype(np.int64).reshape(out['rgb'].shape[:2])
    if aov is not None: out['stain_m'] = np.asarray(aov, np.float64).reshape(out['rgb'].shape[:2])
    out['channels'] = sorted(ch)
    return out


GROUND_PASS_INDEX = int(re.search(r'^GROUND_PASS_INDEX = (\d+)', WORKER.read_text(encoding='utf8'), re.M).group(1))
MASK = dict(sky=0, rock=1, ground=2, undetermined=255)


def overcast_light(table):
    """CIE standard overcast, neutral, L_z = 9 / (7 k): a white horizontal Lambertian surface (E_h = 7 pi L_z / 9) reads 1.0
    after the study's exposure k, as the 45 deg sun + sky reference does; white-balance gains (1, 1, 1)."""
    k = exposure_k(table)
    return dict(mode='overcast', zenith_radiance=9. / (7. * k), color=[1., 1., 1.], distribution='CIE standard overcast (1 + 2 sin e) / 3',
                exposure_k=k, table=table['path'])


def overcast_gains():
    return [1., 1., 1.]


def ground_spec(plan, color_srgb8=None):
    """the plan's ground square (camera_plan_r2: z_m, center_xy_m, half_size_m) as the worker job field"""
    g = dict(plan['ground'])
    if color_srgb8 is not None: g['color_srgb8'] = list(color_srgb8)
    return g


def region_mask(decoded, sky_depth=1e9):
    """uint8 per pixel: 0 sky (IndexMA 0, no depth), 1 rock, 2 ground (IndexMA GROUND_PASS_INDEX), 255 undetermined (IndexMA 0
    with a finite depth). IndexMA / Depth are not anti-aliased (one value per pixel)."""
    idx, depth = decoded['index'], decoded['depth']
    m = np.full(idx.shape, MASK['rock'], np.uint8)
    m[idx == GROUND_PASS_INDEX] = MASK['ground']
    m[(idx == 0) & (depth >= sky_depth)] = MASK['sky']
    m[(idx == 0) & (depth < sky_depth)] = MASK['undetermined']
    return m


class Worker:
    """One resident Blender process; one sample count for its lifetime. worker: another worker script (regression checks only);
    threads: Blender -t for CPU-device sessions (None keeps the command line unchanged); filter_width: worker --filter W, the
    Cycles pixel filter width for diagnostics (None keeps the command line unchanged)."""

    def __init__(self, queue, samples, device='OPTIX', aov=False, log=None, worker=None, threads=None, filter_width=None):
        self.queue, self.samples, self.device, self.aov = Path(queue), int(samples), device, aov
        self.log = Path(log) if log else self.queue/'worker.log'
        self.process = None
        self.worker = Path(worker) if worker else WORKER
        self.threads = threads                      # Blender -t (CPU render threads); None = Blender's default
        self.filter_width = filter_width            # worker --filter (pixel filter width, px); None = the study's 1.5

    def start(self, timeout=180):
        for sub in ('pending', 'running', 'done', 'failed'):
            (self.queue/sub).mkdir(parents=True, exist_ok=True)
        stop = self.queue/'stop'
        if stop.exists(): stop.unlink()
        if (self.queue/'worker.json').exists(): (self.queue/'worker.json').unlink()
        args = [str(BLENDER), '-b'] + (['-t', str(int(self.threads))] if self.threads else []) + ['--factory-startup', '--python', str(self.worker), '--', str(self.queue), '--samples', str(self.samples),
                '--device', self.device] + (['--aov'] if self.aov else []) + (['--filter', repr(float(self.filter_width))] if self.filter_width is not None else [])
        self.logfile = open(self.log, 'a', encoding='utf8')
        self.process = subprocess.Popen(args, stdout=self.logfile, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            beat = self.queue/'worker.json'
            if beat.exists():
                try:
                    state = json.loads(beat.read_text(encoding='utf8'))
                    if state['pid'] == self.process.pid and state['state'] == 'ready': return state
                except (ValueError, KeyError):
                    pass
            if self.process.poll() is not None: raise RuntimeError('Blender worker exited with %s; see %s' % (self.process.returncode, self.log))
            time.sleep(.1)
        raise TimeoutError('Blender worker not ready')

    def submit(self, job):
        job = dict(job, samples=self.samples)
        name = job['id'] + '.json'
        for sub in ('done', 'failed'):
            if (self.queue/sub/name).exists(): raise FileExistsError('job id reused: ' + job['id'])
        tmp = self.queue/(name + '.tmp')
        tmp.write_text(json.dumps(job, indent=1) + '\n', encoding='utf8'); tmp.replace(self.queue/'pending'/name)
        return name

    def wait(self, name, timeout=2700):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for sub in ('done', 'failed'):
                p = self.queue/sub/name
                if p.exists():
                    return json.loads(p.read_text(encoding='utf8'))
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError('Blender worker exited with %s during %s; see %s' % (self.process.returncode, name, self.log))
            time.sleep(.05)
        raise TimeoutError('job %s not finished in %d s (render stuck > 45 min rule)' % (name, timeout))

    def run(self, job):
        result = self.wait(self.submit(job))
        if not result.get('success'):
            raise RuntimeError('Blender job %s failed: %s' % (job['id'], result.get('error')))
        return result

    def stop(self, timeout=120):
        if self.process is None: return
        (self.queue/'stop').write_text('stop\n')
        try:
            self.process.wait(timeout)
        finally:
            self.logfile.close()
            self.process = None
