"""Usage: python render_server.py [--device OPTIX|CPU] [--threads N] [--queue DIR]"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; MS = HERE.parent; P1 = MS.parent; M = P1/'frgb'/'chain'
sys.path[:0] = [str(M), str(P1)]
import blender_client as bc  # noqa: E402
from frgb import budget  # noqa: E402
from sliver_render import TABLE, LEDGER  # noqa: E402
FORMAL_LEDGER = P1/'formal_trace_budget.json'
from PIL import Image  # noqa: E402


class Server:
    def __init__(self, device='OPTIX', threads=None, queue=None):
        self.device, self.threads = device, threads
        self.queue = Path(queue) if queue else HERE/'render_server_queue'
        self.sessions = {}; self.table = bc.load_table(TABLE); self.k = bc.exposure_k(self.table)

    def session(self, spp):
        if spp not in self.sessions:
            w = bc.Worker(self.queue/('s%d_%d' % (spp, int(time.time() * 1000))), spp, device=self.device, threads=self.threads); w.start()
            self.sessions[spp] = w
        return self.sessions[spp]

    def release(self):
        for w in self.sessions.values(): w.stop()
        self.sessions = {}

    def render(self, spec_path):
        spec = json.loads(Path(spec_path).read_text(encoding='utf8')); out = Path(spec['out']); out.mkdir(parents=True, exist_ok=True)
        todo = [f for f in spec['frames'] if not (out/(f['tag'] + '_result.json')).is_file()]
        led = None if ('ledger' in spec and spec['ledger'] is None) else Path(spec.get('ledger') or LEDGER)
        if led is not None and led.resolve() not in (Path(LEDGER).resolve(), FORMAL_LEDGER.resolve()): raise ValueError('ledger not accepted: %s' % led)
        for f in todo:
            w = self.session(f['spp'])
            led is None or budget.charge(led, 'rgb', 1, '%s %s (Blender %d spp, %s)' % (spec.get('ledger_purpose', 'RQ1 driver'), f['tag'], f['spp'],
                                                                          'GPU' if self.device != 'CPU' else 'CPU'))
            t0 = time.time()
            job = dict(id='rq1_%s_%d' % (f['tag'], int(t0 * 1000)), seed=f['seed'], shot=f['shot'], resolution=f['resolution'], light=f['light'],
                       stain=f.get('stain') or {}, mesh=f['mesh'], output_exr=str(out/(f['tag'] + '.exr')))
            if f.get('ground'): job['ground'] = f['ground']
            r = w.run(job); d = bc.decode(r['output_exr'])
            np.savez_compressed(out/(f['tag'] + '.npz'), rgb=d['rgb'].astype(np.float32), depth=d['depth'].astype(np.float32),
                                index=d['index'].astype(np.int32), mask=bc.region_mask(d))
            Image.fromarray(bc.to_srgb8(d['rgb'], self.k, f['gains'])).save(out/(f['tag'] + '.png'))
            r['wall_s'] = time.time() - t0; r['shared_session'] = dict(spp=f['spp'], jobs_in_session=r.get('session_jobs'))
            (out/(f['tag'] + '_result.json')).write_text(json.dumps(r, indent=1) + '\n', encoding='utf8')
        results = {}
        for f in spec['frames']:
            r = json.loads((out/(f['tag'] + '_result.json')).read_text(encoding='utf8'))
            results[f['tag']] = dict(npz=str(out/(f['tag'] + '.npz')), png=str(out/(f['tag'] + '.png')), timing=r['timing'], wall_s=r.get('wall_s'),
                                     readback_closed=r['readback'].get('aerial', {}).get('closed', r['readback'].get('closed')))
        (out/'render_point.json').write_text(json.dumps(dict(exposure_k=self.k, frames=results), indent=1) + '\n', encoding='utf8')
        return dict(frames=list(results), rendered=len(todo))


def main(argv):
    g = lambda k, d=None: argv[argv.index('--' + k) + 1] if '--' + k in argv else d  # noqa: E731
    s = Server(g('device', 'OPTIX'), int(g('threads')) if g('threads') else None, g('queue'))
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            cmd = json.loads(line)
            try:
                if cmd['op'] == 'render': ans = dict(ok=True, **s.render(cmd['spec']))
                elif cmd['op'] == 'release': s.release(); ans = dict(ok=True)
                elif cmd['op'] == 'quit': s.release(); print(json.dumps(dict(ok=True)), flush=True); break
                else: ans = dict(ok=False, error='unknown op')
            except Exception as e:                                 # noqa: BLE001
                s.release(); ans = dict(ok=False, error=repr(e), traceback=traceback.format_exc()[-3000:])
            print(json.dumps(ans), flush=True)
    finally:
        s.release()


if __name__ == '__main__':
    main(sys.argv[1:])
