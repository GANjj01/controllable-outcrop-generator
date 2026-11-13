"""View-angle sweep (main text Section 5.1 and Fig. 4; supplement S4, view-angle sweep): viewpoints and readings.

One target point per scene (the camera target of a paper viewpoint), constant imaging distance, off-nadir angle beta of 0, 15,
30, 45 and 60 degrees, and for beta > 0 two camera azimuths: across (perpendicular to) and along the strike of the target fracture.

    python reproduce/paper1/analysis/view_sweep.py plan --scene flat=FLAT_SCENE_DIR --scene standalone=STANDALONE_SCENE_DIR --out view_sweep_plan.json
    python reproduce/paper1/analysis/view_sweep.py collect --plan view_sweep_plan.json --frames FRAMES --out reproduce/paper1/records/round3/view_sweep.json

plan     reads SCENE_DIR/plan.json and SCENE_DIR/scene/geom/surface.npz of the two scenes (python -m fracgen scene, then
         python -m fracgen cameras; flat outcrop: seed 26120001, standalone outcrop: seed 26130000, see frames.csv rows a0077 and
         a0044) and writes, for every viewpoint, the camera position, the sun direction, the options of `python -m fracgen frame`
         and geometric readings of the camera position (height above the ground, rock hits above the camera, first hit along the
         optical axis, angle between the line of sight and the rock-surface normal at the target).
collect  reads FRAMES/<viewpoint id>/frame.json and q0/trace_labels/report.json of every viewpoint that was generated and writes
         the record with one row per viewpoint.

Camera position = target + distance x (sin(beta) sin(A), sin(beta) cos(A), cos(beta)) (east, north, up); A is the horizontal azimuth
of the camera seen from the target, clockwise from north. The strike is the reference azimuth of the plan camera (the strike of
the plane of the target fracture). Of the two candidates of each azimuth (A and A + 180 degrees) the one nearer to the camera
azimuth of the paper viewpoint is used. Sun elevation 40 degrees, sun azimuth A + 30 degrees; baseline shape parameters; 1024 px,
1024 samples.

Readings (report.json of the label program): in-frame length = sum of per_path.projected_in_frame_px; occluded length = sum of
per_path.occluded_px = occlusion.joint.own_occluded_px (occluded by the groove of the fracture's own joint) +
occlusion.joint.far_occluded_px (occluded by other rock); below-ground fraction = sum of below_ground_px / (in-frame + below-ground
length); unclear (U) fraction = per_type.joint.visible_u_px / per_type.joint.visible_px.

Needs Python 3.8+ and NumPy."""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402

SCENES = dict(flat=dict(tag='f', seed=26120001, base_frame='a0077'), standalone=dict(tag='s', seed=26130000, base_frame='a0044'))
BETAS = (0, 15, 30, 45, 60)
SHAPE = dict(width_m=.055, variation=.225, depth_m=.175, roughness_m=.0020625)
SUN_ELEVATION, SUN_OFFSET = 40., 30.
KIND = dict(z='nadir', p='across strike', a='along strike')


def frames_row(name):
    with open(fio.PAPER1 / 'frames.csv', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['frame'] == name:
                return r
    raise KeyError(name)


def ray_hits(v, f, o, d):
    """Moeller-Trumbore, one ray against all triangles: (t, front face towards the ray origin) in order of increasing t"""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1. / np.where(ok, det, 1.), 0.)
    s = o - a
    u = np.einsum('ij,ij->i', s, p) * inv
    q = np.cross(s, e1)
    w = (q @ d) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-9)
    idx = np.flatnonzero(hit)
    nrm = np.cross(e1[idx], e2[idx])
    front = (nrm @ d) < 0
    o_ = np.argsort(t[idx])
    return t[idx][o_], front[o_]


def position(target, dist, beta, az):
    b, a = math.radians(beta), math.radians(az)
    return [target[0] + dist * math.sin(b) * math.sin(a), target[1] + dist * math.sin(b) * math.cos(a), target[2] + dist * math.cos(b)]


def clearance(v, f, pos, ground):
    """clearance (m) below the camera down to the rock surface or the ground plane, and the number of rock hits above and below it"""
    td, _ = ray_hits(v, f, np.asarray(pos, float), np.array([0., 0., -1.]))
    tu, _ = ray_hits(v, f, np.asarray(pos, float), np.array([0., 0., 1.]))
    below = min([pos[2] - ground] + ([float(td[0])] if len(td) else []))
    return float(below), int(len(tu)), int(len(td))


def plan_scene(name, root):
    sc = SCENES[name]
    row = frames_row(sc['base_frame'])
    plan = fio.read_json(Path(root) / 'plan.json')
    tgt6 = [float(x) for x in row['camera_target_m'].split()]
    cam = min((c for c in plan['cameras'] if c), key=lambda c: float(np.linalg.norm(np.subtract(c['target_enu_m'], tgt6))))
    P, T = cam['shot']['position_enu_m'], cam['shot']['target_enu_m']
    assert float(np.abs(np.subtract(T, tgt6)).max()) <= 1e-5, 'no plan camera at the target of frame %s' % sc['base_frame']
    d = [P[i] - T[i] for i in range(3)]
    dist = math.sqrt(sum(x * x for x in d))
    a0 = math.degrees(math.atan2(d[0], d[1])) % 360.
    strike = float(cam['light_reference']['azimuth_deg']) % 360.

    def closer(c1, c2):
        f = lambda a: abs((a - a0 + 180.) % 360. - 180.)  # noqa: E731
        return c1 % 360. if f(c1) <= f(c2) else c2 % 360.

    az = dict(p=closer(strike + 90., strike - 90.), a=closer(strike, strike + 180.))
    ground = float(plan['ground']['z_m'])
    focal = float(cam['shot']['focal_length_mm'])
    slot = int(row['camera_slot'])
    n = np.asarray(cam['normal'], float)
    n = n / np.linalg.norm(n)
    with np.load(Path(root) / 'scene/geom/surface.npz', allow_pickle=False) as z:
        v, f = z['vertices'], z['faces']
    views = []
    for beta in BETAS:
        for kind in (('z',) if beta == 0 else ('p', 'a')):
            A = az['p' if kind == 'z' else kind]
            pos = position(T, dist, float(beta), A)
            cl, up, dn = clearance(v, f, pos, ground)
            dvec = np.subtract(T, pos) / dist
            t, front = ray_hits(v, f, np.asarray(pos, float), dvec)
            sun = (A + SUN_OFFSET) % 360.
            vid = 'vs_%s_b%02d%s' % (sc['tag'], beta, kind)
            opts = ['--camera', str(slot), '--width-m', repr(SHAPE['width_m']), '--variation', repr(SHAPE['variation']), '--depth-m', repr(SHAPE['depth_m']),
                    '--roughness-m', repr(SHAPE['roughness_m']), '--sun-azimuth', repr(sun), '--sun-elevation', repr(SUN_ELEVATION),
                    '--resolution', '1024', '--spp', '1024', '--position'] + [repr(float(x)) for x in pos] + ['--target'] + [repr(float(x)) for x in T] + \
                   ['--focal-mm', repr(focal)]
            views.append(dict(
                id=vid, outcrop=name, scene_tag=sc['tag'], seed=sc['seed'], beta_deg=beta, azimuth_kind=KIND[kind], camera_azimuth_deg=A,
                position_enu_m=pos, target_enu_m=list(T), distance_m=dist, focal_mm=focal, pixel_mm=dist * 1000. * 36. / focal / 1024.,
                sun_azimuth_deg=sun, sun_elevation_deg=SUN_ELEVATION, camera_slot=slot, frame_options=' '.join(opts),
                camera_above_ground_m=pos[2] - ground, clearance_below_m=cl, rock_hits_above_camera=up, rock_hits_below_camera=dn,
                view_axis_first_hit_m=float(t[0]) if len(t) else None, view_axis_first_hit_front=bool(front[0]) if len(t) else None,
                view_axis_hits=int(len(t)),
                angle_to_surface_normal_deg=float(math.degrees(math.acos(max(-1., min(1., float(-dvec @ n))))))))
    return dict(outcrop=name, seed=sc['seed'], base_frame=sc['base_frame'], camera_slot=slot, target_enu_m=list(T), base_position_enu_m=list(P),
                distance_m=dist, pixel_mm=dist * 1000. * 36. / focal / 1024., base_beta_deg=math.degrees(math.acos(d[2] / dist)),
                base_camera_azimuth_deg=a0, strike_azimuth_deg=strike, azimuth_across_deg=az['p'], azimuth_along_deg=az['a'], ground_z_m=ground,
                target_surface_normal=n.tolist(), target_normal_tilt_deg=float(math.degrees(math.acos(n[2]))),
                plan_camera=dict(cell=cam['cell'], k=cam.get('k', 0), level=cam['level'], angle=cam['angle'], face_type=cam.get('face_type'),
                                 angle_deg=cam['angle_deg']), views=views)


def cmd_plan(args):
    scenes = {}
    for item in args.scene:
        name, root = item.split('=', 1)
        scenes[name] = plan_scene(name, root)
        for v in scenes[name]['views']:
            print('%-10s beta %2d  %-13s A %7.2f  above ground %5.2f m  hits above %d  axis first hit front %s  angle to normal %5.1f' % (
                v['id'], v['beta_deg'], v['azimuth_kind'], v['camera_azimuth_deg'], v['camera_above_ground_m'], v['rock_hits_above_camera'],
                v['view_axis_first_hit_front'], v['angle_to_surface_normal_deg']))
    fio.write_json(args.out, dict(shape_parameters=SHAPE, scenes=scenes))


def read_view(fd):
    rep = fio.read_json(fd / 'q0/trace_labels/report.json')
    fr = fio.read_json(fd / 'frame.json')
    pp = rep['per_path']
    s = lambda k: float(sum((r.get(k) or 0.) for r in pp))  # noqa: E731
    L_in, L_occ, L_vis, L_below = s('projected_in_frame_px'), s('occluded_px'), s('visible_px'), s('below_ground_px')
    oj = rep['occlusion']['joint']
    own, far = float(oj['own_occluded_px']), float(oj['far_occluded_px'])
    assert abs(own + far - L_occ) <= 1e-6 * max(1., L_occ), (str(fd), own, far, L_occ)
    jt = rep['per_type']['joint']
    return dict(label_exit_code=rep.get('exit_code'), checks=fr['checks']['overall'], pixel_mm=fr['pixel_mm'], in_frame_px=L_in, visible_px=L_vis,
                occluded_px=L_occ, own_groove_occluded_px=own, other_rock_occluded_px=far, below_ground_px=L_below,
                joint_visible_px=float(jt['visible_px']), joint_unclear_px=float(jt['visible_u_px']),
                occluded_share=L_occ / L_in if L_in else None, own_groove_share=own / L_in if L_in else None,
                other_rock_share=far / L_in if L_in else None, below_ground_share=L_below / (L_in + L_below) if (L_in + L_below) else None,
                occluded_runs=oj.get('runs'), unclear_share_of_visible=jt['visible_u_px'] / jt['visible_px'] if jt['visible_px'] else None,
                fault_visible_px=rep['per_type']['fault']['visible_px'])


def cmd_collect(args):
    plan = fio.read_json(args.plan)
    rows, scenes = [], {}
    for name, sp in plan['scenes'].items():
        scenes[name] = {k: v for k, v in sp.items() if k != 'views'}
        for v in sp['views']:
            row = {k: v[k] for k in ('id', 'outcrop', 'scene_tag', 'seed', 'beta_deg', 'azimuth_kind', 'camera_azimuth_deg', 'sun_azimuth_deg',
                                     'sun_elevation_deg', 'position_enu_m', 'camera_above_ground_m', 'rock_hits_above_camera',
                                     'rock_hits_below_camera', 'view_axis_first_hit_front', 'angle_to_surface_normal_deg')}
            try:
                fd = fio.frame_dir(args.frames, v['id'])
                ok = (fd / 'frame.json').is_file() and (fd / 'q0/trace_labels/report.json').is_file()
            except FileNotFoundError:
                ok = False
            if ok:
                row.update(state='done', **read_view(fd))
            elif v['rock_hits_above_camera'] % 2 == 1:
                row.update(state='not_available', reason='camera position inside the rock body (odd number of rock-surface hits above the '
                                                         'camera); no frame can be generated from it')
            else:
                row.update(state='not_available', reason='no labelled frame for this viewpoint')
            rows.append(row)
    rows.sort(key=lambda r: (r['scene_tag'], r['azimuth_kind'], r['beta_deg']))
    out = dict(
        description='View-angle sweep: one row per viewpoint. Occluded length as a fraction of the in-frame length, divided into occlusion by '
                    "the groove of the fracture's own joint and occlusion by other rock, against the off-nadir angle beta, with the camera azimuth "
                    'across or along the strike of the target fracture. Target point and imaging distance are fixed per scene; shape '
                    'parameters at their baseline values; sun elevation 40 degrees, sun azimuth = camera azimuth + 30 degrees; 1024 px, 1024 samples.',
        paper_items=['main text Section 5.1', 'main text Fig. 4', 'supplement S4: view-angle sweep, Table of the readings at each viewpoint'],
        script='reproduce/paper1/analysis/view_sweep.py',
        units=dict(beta_deg='degrees from the zenith to the target-to-camera direction', camera_azimuth_deg='degrees clockwise from north, camera seen from the target',
                   in_frame_px='image pixels of trace length', occluded_share='occluded_px / in_frame_px', own_groove_share='own_groove_occluded_px / in_frame_px',
                   other_rock_share='other_rock_occluded_px / in_frame_px', below_ground_share='below_ground_px / (in_frame_px + below_ground_px)',
                   unclear_share_of_visible='joint_unclear_px / joint_visible_px (U length over visible length)', pixel_mm='nominal pixel size at the target, mm',
                   angle_to_surface_normal_deg='angle between the line of sight and the rock-surface normal at the target, degrees',
                   position_enu_m='metres, east-north-up'),
        shape_parameters=plan.get('shape_parameters'), scenes=scenes, rows=rows)
    fio.write_json(args.out, out)
    pct = lambda x: '   -  ' if x is None else '%5.1f%%' % (100 * x)  # noqa: E731
    print('id          beta  azimuth        in-frame  occluded  own    other  below  U/vis  checks')
    for r in rows:
        if r['state'] != 'done':
            print('%-11s %3d   %-13s %s' % (r['id'], r['beta_deg'], r['azimuth_kind'], r['reason']))
            continue
        print('%-11s %3d   %-13s %7.0f   %s   %s %s %s %s  %s' % (r['id'], r['beta_deg'], r['azimuth_kind'], r['in_frame_px'], pct(r['occluded_share']),
                                                                 pct(r['own_groove_share']), pct(r['other_rock_share']), pct(r['below_ground_share']),
                                                                 pct(r['unclear_share_of_visible']), r['checks']))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--scene', action='append', required=True, metavar='NAME=SCENE_DIR', help='flat=... and standalone=...')
    p.add_argument('--out', default='view_sweep_plan.json')
    p.set_defaults(func=cmd_plan)
    c = sub.add_parser('collect')
    c.add_argument('--plan', default='view_sweep_plan.json')
    c.add_argument('--frames', required=True, help='folder with one sub-folder per viewpoint (vs_f_b00z ...)')
    c.add_argument('--out', default=str(fio.RECORDS / 'view_sweep.json'))
    c.set_defaults(func=cmd_collect)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
