"""Top-view preview of a scene configuration.

Runs only the geological sampling of the host generator (layers, joint network, faults, outcrop masses); no mesh is built, so a
preview takes well under a second. Use it to choose a seed and to place faults before running the full scene generation.

Writes OUT/preview.png (map view: tile, outcrop masses, joint traces and fault strike lines at height Z) and OUT/preview.json (the
same objects as numbers: joint trace end points, fault planes, outcrop masses)."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def sample(config, seed=None):
    """(spec, exposure): the sampled geological model and the outcrop description, stopped before meshing"""
    sys.modules.setdefault('cupy', None)
    from outcrop.dsgen import pipeline
    cfg = json.loads(json.dumps(config))
    if seed is not None:
        cfg['seed'] = int(seed)
    cfg.get('geometry', {}).pop('outcrop_type', None)
    got = {}

    class _Stop(Exception):
        pass

    original_spec, original_surface = pipeline.sample_spec, pipeline.build_surface

    def spec(s, **kw):
        got['spec'] = original_spec(s, **kw)
        return got['spec']

    def surface(expo, **kw):
        got['exposure'] = expo
        raise _Stop()

    pipeline.sample_spec, pipeline.build_surface = spec, surface
    try:
        pipeline.build_declaration(cfg)
    except _Stop:
        pass
    finally:
        pipeline.sample_spec, pipeline.build_surface = original_spec, original_surface
    return got['spec'], got['exposure']


def joint_traces(spec, z):
    """joint id -> (direction group, polyline (N, 2) of the joint surface at height z)"""
    meshes = {g['id']: g for g in spec['geometry'] if g.get('kind') == 'mesh'}
    out = {}
    for f in spec['features']:
        if f.get('kind') != 'joint' or f.get('geometry') not in meshes:
            continue
        v = np.asarray(meshes[f['geometry']]['vertices'], float)
        bottom, top = v[0::2], v[1::2]
        t = (z - bottom[:, 2]) / np.where(np.abs(top[:, 2] - bottom[:, 2]) > 1e-12, top[:, 2] - bottom[:, 2], np.nan)
        p = bottom + t[:, None] * (top - bottom)
        out[f['id']] = ((f.get('metadata') or {}).get('direction_group_id'), p[:, :2])
    return out


def fault_planes(spec):
    planes = {g['id']: g for g in spec['geometry'] if g.get('kind') == 'plane'}
    rows = []
    for e in spec['events']:
        if e.get('kind') != 'fault':
            continue
        p = planes.get(e.get('fault')) or {}
        rows.append(dict(id=e['id'], strike_deg=p.get('strike_deg'), dip_deg=p.get('dip_deg'), point_m=p.get('point_m'),
                         motion=e.get('motion'), moving_side=e.get('moving_side')))
    return rows


def fault_line(row, z, half_length):
    a, d = math.radians(row['strike_deg']), math.radians(row['dip_deg'])
    s = np.array([math.sin(a), math.cos(a)])
    h = np.array([math.cos(a), -math.sin(a)])
    p = np.asarray(row['point_m'], float)
    shift = (p[2] - z) / math.tan(d) if d > 1e-9 else 0.
    c = p[:2] + shift * h
    return np.array([c - half_length * s, c + half_length * s])


def draw(spec, expo, z, path, size=900):
    from PIL import Image, ImageDraw
    lo, hi = np.asarray(spec['domain']['min_m'][:2], float), np.asarray(spec['domain']['max_m'][:2], float)
    pad = 40
    scale = (size - 2 * pad) / float(max(hi - lo))

    def px(q):
        q = np.atleast_2d(q)
        return [(pad + (x - lo[0]) * scale, size - pad - (y - lo[1]) * scale) for x, y in q]

    img = Image.new('RGB', (size, size + 60), 'white')
    dr = ImageDraw.Draw(img)
    dr.rectangle([*px([lo[0], hi[1]])[0], *px([hi[0], lo[1]])[0]], outline=(120, 120, 120), width=2)
    for m in expo.get('masses') or []:
        c, h = np.asarray(m['center_m'][:2]), np.asarray(m['half_size_m'][:2])
        a, b = px(c + [-h[0], h[1]])[0], px(c + [h[0], -h[1]])[0]
        dr.rounded_rectangle([a, b], radius=float(m.get('rounding_m', 0.)) * scale, outline=(160, 110, 60), width=3)
    if expo.get('origin_m'):
        o = px(expo['origin_m'][:2])[0]
        dr.ellipse([o[0] - 4, o[1] - 4, o[0] + 4, o[1] + 4], fill=(160, 110, 60))
    colours = [(30, 90, 200), (0, 150, 120), (140, 60, 180), (200, 120, 0)]
    groups = {}
    for jid, (group, line) in sorted(joint_traces(spec, z).items()):
        k = groups.setdefault(group, len(groups))
        ok = np.isfinite(line).all(1)
        pts = px(line[ok])
        if len(pts) > 1:
            dr.line(pts, fill=colours[k % len(colours)], width=2)
            dr.text(pts[0], jid.rsplit('-', 1)[-1], fill=colours[k % len(colours)])
    for row in fault_planes(spec):
        pts = px(fault_line(row, z, float(max(hi - lo))))
        dr.line(pts, fill=(210, 30, 30), width=3)
        dr.text(pts[0], row['id'], fill=(210, 30, 30))
    x = pad
    for text, colour in [('map view at z = %.2f m, domain %s m' % (z, (hi - lo).round(2).tolist()), (0, 0, 0)),
                         ('outcrop masses', (160, 110, 60)), ('faults', (210, 30, 30))] +             [('joint set %s' % g, colours[i % len(colours)]) for g, i in groups.items()]:
        dr.text((x, size + 10), text, fill=colour)
        x += 8 * len(text) + 24
    dr.text((pad, size + 30), 'x = east, y = north (metres); joint numbers at the start of each trace', fill=(0, 0, 0))
    img.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('config')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--z', type=float, help='map height in metres (default: the outcrop ground plus 1 m)')
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
    spec, expo = sample(cfg, a.seed)
    z = a.z if a.z is not None else float(expo.get('ground_z_m', 0.)) + 1.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    draw(spec, expo, z, out / 'preview.png')
    record = dict(seed=int(a.seed if a.seed is not None else cfg.get('seed')), z_m=z, domain=spec['domain'],
                  exposure={k: v for k, v in expo.items() if isinstance(v, (int, float, str, list, dict)) and k != 'model'},
                  joints={k: dict(group=g, trace_xy_m=[[round(float(x), 4), round(float(y), 4)] for x, y in line if np.isfinite(x)])
                          for k, (g, line) in sorted(joint_traces(spec, z).items())},
                  faults=fault_planes(spec))
    (out / 'preview.json').write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    print('joints', len(record['joints']), 'faults', len(record['faults']), '->', out / 'preview.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
