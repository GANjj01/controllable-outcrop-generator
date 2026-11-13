"""Illumination sweep: renders the sun directions of a case on the cut geometry of its base frame
(main text Section 5.2; supplement S4, "Illumination sweep and the shadow ratio"; frames of block light_sweep in
reproduce/paper1/sweeps.csv, roles sweep and repeat).

The base frame of a case (ls_v1b055, ls_v1w030, ls_v1w080, ls_v1d010, ls_v1d025, ls_v2b055; role base in sweeps.csv) is
generated first with the whole chain by `python -m fracgen frame`. This script then changes only the sun direction and the
render seed of the base frame's render specification and renders the 72 sun directions of the case (azimuth 0-345 degrees in
steps of 15 degrees at elevations 20, 40 and 60 degrees, render seed 0; the two baseline cases also 6 repeats with render seed 1)
in one session of the generator's render server, which loads the mesh once (about 6 s per image on the tested GPU). Geometry
and labels are not rebuilt: the windows of the brightness check depend only on the geometry, and light_sweep_stats.py
recalculates their brightness difference on every image.

    python reproduce/paper1/analysis/light_sweep.py CASE [CASE ...] --frames FRAMES --sweep SWEEP [--device OPTIX] [--dry-run]

Before the sweep of a case the driver is checked: (1) the render specification it builds for the base frame's own light must
equal the base frame's render_spec.json item by item; (2) one image rendered from it must have depth, material index and region
mask bit-identical to the base frame's render, with at most one 8-bit level of difference in at most 0.01 % of the PNG values.
--dry-run does (1) only and lists the images that would be rendered; nothing is rendered.

Reads FRAMES/<case>/{frame.json, plan.json, q0/camera.json, q0/light.json, q0/geometry.json, q0/render_spec.json} and the meshes
they name. Writes SWEEP/<case>/<aAAAeEEsS>/{render_spec.json, frame.exr, frame.npz, frame.png, frame_result.json} (AAA sun
azimuth, EE sun elevation, S render seed), SWEEP/<case>/selftest.json and SWEEP/<case>/sweep_done.json; images that exist are
not rendered again. Run from the repository root in the labels environment, with the machine configuration of the generator in
place (inputs/runconfig_local.json: interpreters and Blender)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import frames_io as fio  # noqa: E402
import light_sweep_stats as ST  # noqa: E402

CASES, tag_of, lights_of = ST.CASES, ST.tag_of, ST.lights_of


def release():
    """the modules of the generator that build the render specification and start the render server"""
    from fracgen import frame as FR
    from fracgen.env import MS, child_env, python
    rq = FR._driver()
    return dict(FR=FR, rq=rq, MS=MS, child_env=child_env, python=python)


def base_ctx(frames, case, rel):
    """what fracgen frame passes to the render step, read from the files the base frame stored (read only)"""
    sd = fio.frame_dir(frames, case)
    q0 = sd / 'q0'
    fj = fio.read_json(sd / 'frame.json')
    req = fio.read_json(Path(fj['scene_root']) / 'scene_request.json')
    outcrop = 'flat' if req['outcrop'] == 'flat' else 'isolated'
    plan = fio.read_json(sd / 'plan.json')                       # the frame's copy of the camera plan, with its resolution
    camrec = fio.read_json(q0 / 'camera.json')
    cam = rel['rq'].camera_of(plan, int(camrec['camera_index']), outcrop)
    assert cam is not None and plan['cameras'].index(cam) == int(fj['plan_camera_index'])
    point = fio.read_json(q0 / 'light.json')
    geo = fio.read_json(q0 / 'geometry.json')
    assert not geo.get('geometry_rejected')
    s = dict(mechanism='shape', outcrop=outcrop, id=case)
    return dict(s=s, point=point, cam=cam, geo=geo, plan=plan, frame_json=fj, q0=q0, base_spec=fio.read_json(q0 / 'render_spec.json')['frames'][0])


def spec_frames(ctx, rel, az, el, seed):
    return rel['rq'].frames_of(ctx['s'], ctx['point'], ctx['cam'], ctx['geo'], ctx['plan'], dict(azimuth_deg=float(az), elevation_deg=float(el)), int(seed))


def norm(o):
    return json.loads(json.dumps(o, default=str))


def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=1) + '\n', encoding='utf-8')


class Server:
    """the generator's render server (multiscene/driver/render_server.py) as a resident process, started with the command and the
    environment of fracgen frame; queue and working directory are in the sweep folder of the case"""

    def __init__(self, rel, workdir, device='OPTIX', threads=8):
        self.workdir = Path(workdir)
        (self.workdir / 'logs').mkdir(parents=True, exist_ok=True)
        cmd = [rel['python']('geometry'), '-B', rel['MS'] / 'driver/render_server.py', '--device', device, '--threads', threads,
               '--queue', self.workdir / 'render_queue']
        self.log = (self.workdir / 'logs/render_server.log').open('a', encoding='utf-8')
        self.proc = subprocess.Popen([str(c) for c in cmd], env=rel['child_env']('geometry'), cwd=str(self.workdir), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True)

    def ask(self, cmd):
        self.proc.stdin.write(json.dumps(cmd) + '\n')
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError('the render server ended (log %s)' % (self.workdir / 'logs/render_server.log'))
            if line.startswith('{'):
                ans = json.loads(line)
                if 'ok' in ans:
                    return ans

    def render(self, spec_path):
        ans = self.ask(dict(op='render', spec=str(spec_path)))
        if not ans['ok']:
            raise RuntimeError('render server: %s\n%s' % (ans.get('error'), ans.get('traceback')))
        return ans

    def quit(self):
        try:
            self.ask(dict(op='quit'))
        except Exception:
            pass
        try:
            self.proc.wait(timeout=180)
        finally:
            self.log.close()


def render_list(root, case, items, purpose, rel, ctx, device, threads):
    """items: [(tag, azimuth, elevation, seed)]; an item whose frame_result.json exists is skipped. Returns the number rendered."""
    todo = [it for it in items if not (root / it[0] / 'frame_result.json').is_file()]
    if not todo:
        return 0
    srv = Server(rel, root, device, threads)
    n = 0
    try:
        for tag, az, el, seed in todo:
            d = root / tag
            frames = spec_frames(ctx, rel, az, el, seed)
            assert len(frames) == 1 and frames[0]['tag'] == 'frame' and frames[0]['seed'] == int(seed)
            spec = d / 'render_spec.json'
            d.mkdir(parents=True, exist_ok=True)
            spec.write_text(json.dumps(dict(out=str(d), ledger=None, ledger_purpose='illumination sweep (%s)' % purpose, base_frame=case, frames=frames),
                                       indent=1, default=str) + '\n', encoding='utf-8')
            t0 = time.time()
            srv.render(spec)
            assert (d / 'frame.png').is_file() and (d / 'frame.npz').is_file()
            print('%s %s  %.1f s' % (case, tag, time.time() - t0), flush=True)
            n += 1
    finally:
        srv.quit()
    return n


def selftest(frames, sweep, case, rel, device, threads, dry_run=False):
    """(1) the specification built for the base frame's own light equals its render_spec.json; (2) one image rendered from it is
    compared with the base frame's image"""
    ctx = base_ctx(frames, case, rel)
    bl = ctx['point']['light']
    fr = spec_frames(ctx, rel, bl['azimuth_deg'], bl['elevation_deg'], 0)
    same = norm(fr[0]) == norm(ctx['base_spec'])
    diff = [k for k in set(fr[0]) | set(ctx['base_spec']) if norm(fr[0]).get(k) != norm(ctx['base_spec']).get(k)]
    out = dict(case=case, spec_equal_to_base=bool(same), spec_keys_differ=diff, base_light=bl)
    if same and not dry_run:
        root = Path(sweep) / case
        d = root / 'base_s0'
        if not (d / 'frame_result.json').is_file():
            srv = Server(rel, root, device, threads)
            try:
                d.mkdir(parents=True, exist_ok=True)
                (d / 'render_spec.json').write_text(json.dumps(dict(out=str(d), ledger=None, ledger_purpose='illumination sweep, driver self-test',
                                                                    base_frame=case, frames=fr), indent=1, default=str) + '\n', encoding='utf-8')
                srv.render(d / 'render_spec.json')
            finally:
                srv.quit()
        r = out['render_vs_base'] = ST.compare_images(d / 'frame.png', ctx['q0'] / 'render/frame.png', d / 'frame.npz', ctx['q0'] / 'render/frame.npz')
        out['pass'] = bool(r['depth_bitwise_equal'] and r['index_bitwise_equal'] and r['mask_bitwise_equal'] and r['diff2plus'] == 0 and r['diff1_share'] <= 1e-4)
    else:
        out['pass'] = bool(same) if dry_run else False
    if not dry_run:
        save(Path(sweep) / case / 'selftest.json', out)
    print(json.dumps(out, indent=1))
    return out, ctx


def render_case(frames, sweep, case, rel, device, threads, dry_run=False):
    root = Path(sweep) / case
    st = root / 'selftest.json'
    if not dry_run and st.is_file() and fio.read_json(st).get('pass'):
        ctx = base_ctx(frames, case, rel)
    else:
        s, ctx = selftest(frames, sweep, case, rel, device, threads, dry_run)
        if not s['pass']:
            raise SystemExit('driver self-test failed, %s is not rendered' % case)
    items = [(tag_of(az, el, seed), az, el, seed) for az, el, seed in lights_of(case)]
    if dry_run:
        todo = [it[0] for it in items if not (root / it[0] / 'frame_result.json').is_file()]
        print('%s: %d images, %d to render: %s ...' % (case, len(items), len(todo), ' '.join(todo[:4])))
        return
    n = render_list(root, case, items, 'sweep', rel, ctx, device, threads)
    done = sum((root / it[0] / 'frame_result.json').is_file() for it in items)
    save(root / 'sweep_done.json', dict(case=case, requested=len(items), present=done, rendered_this_run=n, complete=bool(done == len(items))))
    print(json.dumps(dict(case=case, rendered=n, present=done, of=len(items))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cases', nargs='+', choices=sorted(CASES), metavar='CASE', help='one or more of ' + ', '.join(ST.ORDER))
    ap.add_argument('--frames', required=True, help='folder with the base frames (ls_v1b055 ...)')
    ap.add_argument('--sweep', required=True, help='output folder of the sweep images')
    ap.add_argument('--device', default='OPTIX', choices=['OPTIX', 'CUDA', 'CPU'])
    ap.add_argument('--threads', type=int, default=8, help='CPU render threads')
    ap.add_argument('--dry-run', action='store_true', help='check the render specification against the base frame and list the images; render nothing')
    a = ap.parse_args(argv)
    rel = release()
    for case in a.cases:
        render_case(a.frames, a.sweep, case, rel, a.device, a.threads, a.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
