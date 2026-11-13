"""Resumable: every step of a point is one JSON in OUT/<scene>/pNN/; per-camera work (plan, light domain, refinement,
field normals, stain frame, stain host case) is cached in OUT/<scene>/.

Every subprocess runs in a Windows job object; its peak committed memory is recorded and feeds the table. - three
stages: prepare (CPU threads), render + score (GPU, serial), labels + qualify + difficulty + record (CPU threads); -
the scoring model is loaded once and moved to the GPU only inside a GPU segment; - a resident render process
(render_server.py): consecutive stain points of one scene, camera slot and sample count form one GPU segment; -
render-noise repeats SCENE:INDEX:rR (geometry of pNN; seeds shape R, stain off / on 2R, off' 2R + 1): their trace
labels read pNN's cached state of the entry's steps A-C (label_cached.py, pinned to label_frame.py's sha256) and run
step D only; - light-domain grids of all sunny cameras are computed first on --grid-workers subprocesses."""
import hashlib, json, math, os, shutil, subprocess, sys, threading, time
from contextlib import ExitStack, contextmanager
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; MS = HERE.parent; P1 = MS.parent; M = P1/'frgb'/'chain'; A = M
sys.path[:0] = [str(P1), str(M), str(MS), str(A)]
from frgb import rq1_multiscene as rm, scene_light as sl, budget, raster  # noqa: E402

from frgb import runconfig as rc  # noqa: E402
PH0 = rc.path('python.geometry').as_posix()
REPO_SRC = rc.path('outcrop_src').as_posix(); MANIFOLD = rc.path('manifold_site').as_posix()
LEDGER = P1/'render_equivalence_ledger_r3.json'
NOGPU = "import sys; sys.modules['cupy']=None; sys.argv=sys.argv[1:]; import runpy; runpy.run_path(sys.argv[0], run_name='__main__')"
DRY = 'RQ1 driver dry run (stand-in model, no scientific use)'
SHAPE_BUILD = dict(construction='panel', panel_refine_m=.002, wall_relief_tip_m=.003)
DRY_PANEL_REFINE_M = .005
EXPORT_NORIM = HERE/'export_norim.py'
VISIBLE_PX_MIN = 256.
EXPOSURE_GATE = rc.path('exposure.gate_r14')
DISPLAY = 'shading_field'
FIELD_REEVAL_MAX_M = 1e-6                                   # S2_R3 gate on the field normals' re-evaluation residual
LABEL_FRAME = P1/'labels/label_frame.py'
LABEL_FRAME_R4 = P1/'labels/label_frame_r4.py'
LABEL_CACHED = HERE/'label_cached.py'
RATES = HERE/'memory_rates.json'
MODEL = HERE/'memory_model.json'
_LOCKS = {}; _LOCKS_GUARD = threading.Lock()
_RES = dict(outstanding=0.); _RES_LOCK = threading.Lock(); _RATES_LOCK = threading.Lock()
_LANES = [threading.BoundedSemaphore(1)]
EXCLUSIVE_RESERVE_GB = 25.                                  # an own exclusive step counts this much in the shared reservations
HEAVY_MIN_FREE_GB = 25.
POOL = True
POOL_BRANCH = 'camera-render'
POOL_MAX_GB = 20.
POOL_TIMEOUT_S = 600.
POOL_ROOT = None
POOL_PROBE = None                                           # None = heavy_budget.Probe() (tests: an injected probe)
WAIT_LONG_S = 1800.
HEAVY_BUDGET = MS/'heavy_budget.py'
_MODE = dict(legacy=False, lanes=1, formal=False, g6=False, halt=None)
FORMAL_LEDGER = P1/'formal_trace_budget.json'
FORMAL_NOTE = 'RQ1 formal run (trace task; ledger formal_trace_budget.json)'
G6_RULE, G6_RULE_SHA = HERE/'criterion_rq1_formal_r1f.json', 'efed7eed7b9705fedbf23db93449851899663b3a1ee847a9a275371ce6f9dae8'
G6_VERIFIED = HERE/'measure_v4/g6_verify/g6_verify.json'
G6_SEED, G6_PER = 20261101, 16


def env0():
    e = dict(os.environ); e['PYTHONPATH'] = os.pathsep.join([str(P1), REPO_SRC, MANIFOLD])
    b = Path(PH0).parent; e['PATH'] = os.pathsep.join([str(b), str(b/'Library/bin'), e.get('PATH', '')]); return e


class JobPeak:
    """a Windows job object: the peak committed memory of a process and all its children"""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes; self.k32 = ctypes.windll.kernel32
        self.k32.CreateJobObjectW.restype = wintypes.HANDLE; self.k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.h = self.k32.CreateJobObjectW(None, None)

    def assign(self, process_handle):
        return bool(self.h) and bool(self.k32.AssignProcessToJobObject(self.h, int(process_handle)))

    def peak_gb(self):
        ct, wt = self.ct, self.wt

        class Basic(ct.Structure):
            _fields_ = [('a', ct.c_int64), ('b', ct.c_int64), ('LimitFlags', wt.DWORD), ('c', ct.c_size_t), ('d', ct.c_size_t), ('e', wt.DWORD),
                        ('f', ct.c_size_t), ('g', wt.DWORD), ('h', wt.DWORD)]

        class Ext(ct.Structure):
            _fields_ = [('basic', Basic), ('io', ct.c_ulonglong * 6), ('ProcessMemoryLimit', ct.c_size_t), ('JobMemoryLimit', ct.c_size_t),
                        ('PeakProcessMemoryUsed', ct.c_size_t), ('PeakJobMemoryUsed', ct.c_size_t)]
        x = Ext()
        if not self.h or not self.k32.QueryInformationJobObject(self.h, 9, ct.byref(x), ct.sizeof(x), None): return None
        return x.PeakJobMemoryUsed / 2 ** 30

    def close(self):
        if self.h: self.k32.CloseHandle(self.h); self.h = None


def run_measured(cmd, cwd=None, env=None, measure=None):
    p = subprocess.Popen([str(a) for a in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    job = JobPeak(); ok = job.assign(p._handle)
    out, err = p.communicate(); pk = job.peak_gb() if ok else None; job.close()
    if measure is not None and pk is not None: measure['peak_job_gb'] = max(measure.get('peak_job_gb') or 0., pk)
    return p.returncode, out, err


def run0(args, cwd=None, log=None, nogpu=False, measure=None):
    cmd = [PH0] + (['-c', NOGPU] if nogpu else []) + [str(a) for a in args]
    rc, out, err = run_measured(cmd, cwd=cwd, env=env0(), measure=measure)
    if log: Path(log).write_text(out[-30000:] + '\n--- stderr ---\n' + err[-30000:], encoding='utf8')
    if rc: raise RuntimeError('%s failed (%d): %s' % (Path(str(args[0])).name, rc, (err or out)[-800:]))
    return out


class D9CodeError(RuntimeError):
    pass


def halted(stage):
    if _MODE.get('halt'): raise RuntimeError('driver halted before %s: %s' % (stage, _MODE['halt']))


def ledger_path():
    return FORMAL_LEDGER if _MODE['formal'] else LEDGER


def note():
    return FORMAL_NOTE if _MODE['formal'] else DRY


def free_gb():
    r = subprocess.run(['powershell', '-NoProfile', '-Command', '[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)'],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def acquire_lock(name, label, min_free_gb=0):
    path = MS/name
    while True:
        if path.exists(): time.sleep(2); continue
        if min_free_gb and free_gb() < min_free_gb: time.sleep(10); continue
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, ('RQ1 driver (%s) start %s\n' % (label, time.strftime('%Y-%m-%d %H:%M:%S'))).encode()); os.close(fd); return
        except FileExistsError:
            time.sleep(2)


def release_lock(name):
    (MS/name).unlink(missing_ok=True)


@contextmanager
def lock(name, label, min_free_gb=0):
    acquire_lock(name, label, min_free_gb)
    try:
        yield
    finally:
        release_lock(name)


def load_model():
    return json.loads(MODEL.read_text(encoding='utf8')) if MODEL.is_file() else {}


def load_rates():
    return json.loads(RATES.read_text(encoding='utf8')) if RATES.is_file() else {}


def usable_rates(rates):
    return {k: v for k, v in (rates or {}).items() if not (isinstance(v, dict) and v.get('disabled'))}


def rate_pred(rates, kind, q):
    rates = usable_rates(rates); e = rates.get(kind)
    if kind == 'trace_labels_d' and e is not None and q is not None and not float(q) <= float(e.get('max_quantity', -1.)): return None
    return rm.predicted_peak_gb(rates, kind, q)


def _log_line(name, rec):
    with open(RATES.with_name(name), 'a', encoding='utf8') as f: f.write(json.dumps(rec) + '\n')


def _exclusive_need(P, label, m):
    """Above the physical memory it could never be met: RuntimeError at once (mode 'infeasible') instead of waiting for ever"""
    need = HEAVY_MIN_FREE_GB if P is None else max(HEAVY_MIN_FREE_GB, P); m['exclusive_free_gb'] = need
    if need > HEAVY_MIN_FREE_GB:
        total = (POOL_PROBE or _hb().Probe()).machine()['phys_total_gb']
        if need > total:
            m.update(mode='infeasible', phys_total_gb=total)
            raise RuntimeError('reservation infeasible: exclusive needs free >= %.2f GB > physical memory %.2f GB (%s)' % (need, total, label))
    return need


def _wait_watch(m, label):
    t0 = time.time()

    def fire():
        ev = dict(event='wait_long', time=time.strftime('%Y-%m-%d %H:%M:%S'), label=label, kind=m.get('kind'), waited_s=time.time() - t0,
                  **{k: m.get(k) for k in ('predicted_gb', 'P_pool_gb', 'exclusive_free_gb', 'reserve_rule')})
        try:
            _log_line('WAIT_LONG.json', ev); m['wait_long'] = ev
        except Exception as e:
            m['wait_long_error'] = repr(e)
        print('wait_long: %s has waited %.0f s for admission' % (label, ev['waited_s']), file=sys.stderr, flush=True)
    t = threading.Timer(WAIT_LONG_S, fire); t.daemon = True; t.start()
    return t


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sidecar(state):
    return Path(state).with_name(Path(state).name + '.sha256.json')


def state_cache_check(state):
    """The file exists, its sha256 equals the sidecar the driver wrote after its state-out step, every code file the state
    records is unchanged, and label_frame.py / label_frame_r4.py equal label_cached.py's pinned sha256s (its exit-5
    conditions). Returns (ok, reason)."""
    import hashlib, pickle, re
    state = Path(state); side = state_sidecar(state)
    if not state.is_file(): return False, 'state file missing'
    if not side.is_file(): return False, 'no sha256 record of the state (%s missing)' % side.name
    data = state.read_bytes()
    if hashlib.sha256(data).hexdigest() != json.loads(side.read_text(encoding='utf8')).get('sha256'): return False, 'state sha256 differs from its record'
    try:
        recorded = pickle.loads(data)['shas']
    except Exception as e:
        return False, 'state unreadable: %r' % e
    changed = sorted(k for k, v in recorded.items() if not (P1/k).is_file() or _sha256(P1/k) != v)
    if changed: return False, 'code changed since the state was written: %s' % changed[:10]
    pins = dict(re.findall(r"^(PINNED(?:_R4)?) = '([0-9a-f]{64})'", Path(LABEL_CACHED).read_text(encoding='utf8'), re.M))
    for name, f in (('PINNED', 'labels/label_frame.py'), ('PINNED_R4', 'labels/label_frame_r4.py')):
        if name not in pins or not (P1/f).is_file() or _sha256(P1/f) != pins[name]: return False, '%s differs from label_cached.py %s' % (f, name)
    return True, 'ok'


def _feed_rates(m, kind, quantity, unit, label=None):
    q = m.get('actual_quantity', quantity); now = time.strftime('%Y-%m-%d %H:%M:%S')
    fallback = kind == 'trace_labels_d' and bool(m.get('entry_fallback'))
    if fallback:
        ev = dict(event='reservation_shortfall', time=now, label=label, mode=m.get('mode'), predicted_gb=m.get('predicted_gb'),
                  full_entry_gb=label_peak_gb(usable_rates(load_rates()), q), peak_gb=m.get('peak_job_gb'), quantity=q)
        _log_line('RESERVE_SHORTFALL.json', ev); m['reserve_shortfall'] = ev
        print('reservation shortfall: the full entry ran under a D-only reservation (%s); recorded in RESERVE_SHORTFALL.json' % label,
              file=sys.stderr, flush=True)
    if m.get('peak_job_gb') is not None and q:
        with _RATES_LOCK:
            old = load_rates(); e = old.get(kind) or {}; check = None
            if kind == 'trace_labels_d' and e.get('disabled'):
                check = dict(skipped='entry already disabled')
            elif kind == 'trace_labels_d' and e.get('gb_per_unit') is not None:     # a fallback to the full entry (exit 5) is checked too
                R = float(e['gb_per_unit']); pred = rm.predicted_peak_gb(old, kind, q); bare = R * float(q) + .5; peak = float(m['peak_job_gb'])
                check = dict(R_D_before=R, pred_gb=pred, bare_gb=bare, peak_gb=peak, quantity=float(q), exceeded=bool(peak > pred),
                             margin_eaten=bool(peak > bare), in_range=bool(float(q) <= float(e.get('max_quantity', -1.))), entry_fallback=fallback)
            fed = 'trace_labels' if fallback else kind
            r = rm.update_rate(old, fed, m['peak_job_gb'], q, unit)
            if (old.get(fed) or {}).get('gb_per_unit') != r[fed].get('gb_per_unit'): r[fed].pop('fed_from', None)
            if kind == 'trace_labels_d' and not fallback: r[kind]['max_quantity'] = max(float(r[kind].get('max_quantity', 0.)), float(q))
            if check and check.get('exceeded'):
                r[kind] = dict(r[kind], disabled=dict(time=now, label=label, peak_gb=check['peak_gb'], quantity=check['quantity'], pred_gb=check['pred_gb']))
                _log_line('TRACE_LABELS_D_REVERTED.json', dict(r[kind]['disabled'], kind=kind))
                print('trace_labels_d disabled: peak %.2f GB > prediction %.2f GB (%s); recorded in TRACE_LABELS_D_REVERTED.json'
                      % (check['peak_gb'], check['pred_gb'], label), file=sys.stderr, flush=True)
            if check and check.get('margin_eaten'):
                _log_line('TRACE_LABELS_D_MARGIN.json', dict(event='margin_eaten', time=now, label=label, kind=kind, **{k: check[k] for k in
                          ('R_D_before', 'bare_gb', 'pred_gb', 'peak_gb', 'quantity', 'exceeded', 'entry_fallback')}))
            tmp = RATES.with_suffix('.json.tmp'); tmp.write_text(json.dumps(r, indent=1) + '\n', encoding='utf8'); tmp.replace(RATES)
        if check is not None: m['d_class_check'] = dict(check, runs_after=r[kind]['runs'], fed_to=fed)


def _hb():
    m = sys.modules.get('heavy_budget')
    if m is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('heavy_budget', HEAVY_BUDGET); m = importlib.util.module_from_spec(spec)
        sys.modules['heavy_budget'] = m; spec.loader.exec_module(m)
    return m


def label_peak_gb(rates, q):
    """D-only runs take this full-entry value"""
    r = (rates or {}).get('trace_labels')
    if r is None or q is None: return None
    R = float(r['gb_per_unit']); q = float(q); F = max(R * q + 1., 4.5 + .85 * q)
    return max(1.1 * F + .5, 1.5 * R * q + .5)


def step_peak(kind, quantity):
    rates = usable_rates(load_rates())
    if kind in ('trace_labels', 'trace_labels_d'):
        return label_peak_gb(rates, quantity), dict(prediction_source="labels: max(1.1 F' + 0.5, 1.5 R q + 0.5)")
    model = load_model()
    if kind in model:
        extra = {}
        if kind == 'shape_chain' and quantity is not None:
            extra = dict(q_gene_factor=quantity * model['gene_factor'], q_2x=2 * quantity); q = max(extra.values())
        else:
            q = quantity
        env, rate = rm.envelope_predict(model[kind], q), rm.predicted_peak_gb(rates, kind, q)
        known = [(v, s) for v, s in ((env, 'memory_model.json'), (rate, 'memory_rates.json')) if v is not None]
        p, s = max(known) if known else (None, None)
        return p, dict(prediction_source='max(envelope, rate formula): %s' % s, envelope_gb=env, rate_formula_gb=rate, model_quantity=q, **extra)
    q = 2 * quantity if (kind == 'shape_chain' and quantity is not None) else quantity
    return rate_pred(rates, kind, q), dict(prediction_source='memory_rates.json')


def _pool_events(m, root, label, t0):
    try:
        ev, me = [], os.getpid()
        with open(Path(root)/'HEAVY_JOB.pool'/'events.log', encoding='utf8') as f:
            for line in f:
                if str(me) not in line: continue
                e = json.loads(line)
                if e.get('pid') == me and e.get('tag') == label and e.get('t', 0) >= t0: ev.append(e)
        m['pool_events'] = ev
    except Exception as e:
        m['pool_events_error'] = repr(e)


@contextmanager
def reserve(kind, quantity, unit, label):
    """Memory reservation of one step: the pool rule when the pool is on, with one lane and without the legacy mode;
    otherwise reserve_legacy. Yields the step's memory record."""
    if not POOL or _MODE['legacy'] or _MODE['lanes'] != 1:
        why = '--legacy-reserve' if _MODE['legacy'] else 'POOL off' if not POOL else '--lanes %d' % _MODE['lanes']
        with reserve_legacy(kind, quantity, unit, label) as m:
            m['reserve_rule'] = 'legacy (%s)' % why
            yield m
        return
    P, how = step_peak(kind, quantity)
    m = dict(kind=kind, quantity=quantity, unit=unit, predicted_gb=P, reserve_rule='pool', **how); t0 = time.time()
    lane = _LANES[0]; lane.acquire()
    watch = _wait_watch(m, label)
    try:
        with ExitStack() as stack:
            admitted = None
            if P is None:
                m['exclusive_reason'] = 'P unknown'
            else:
                hb = _hb(); root = Path(POOL_ROOT or MS); probe = POOL_PROBE or hb.Probe()
                D = probe.tree_gb(os.getpid()); pp = P + D; m.update(D_gb=D, P_pool_gb=pp)
                if pp > POOL_MAX_GB:
                    m['exclusive_reason'] = 'P_pool %.2f GB > %g GB' % (pp, POOL_MAX_GB)
                else:
                    stack.callback(_pool_events, m, root, label, time.time())      # LIFO: runs after the leave
                    try:
                        admitted = stack.enter_context(hb.slot(POOL_BRANCH, label, pp, root=root, probe=POOL_PROBE, timeout_s=POOL_TIMEOUT_S))
                    except TimeoutError:
                        pool = hb.Pool(root, POOL_PROBE)
                        with pool.mutex(): pool.log('slot_timeout', tag=label, P=pp, timeout_s=POOL_TIMEOUT_S)
                        m.update(slot_timeout=True, exclusive_reason='slot_timeout after %g s' % POOL_TIMEOUT_S)
            if admitted is not None:
                m.update(mode='pool', pool_record=Path(admitted).name)
            else:
                need = _exclusive_need(P, label, m)
                m['mode'] = 'exclusive'; stack.enter_context(lock('HEAVY_JOB.lock', label, min_free_gb=need))
            watch.cancel(); m['wait_s'] = time.time() - t0
            yield m
    finally:
        watch.cancel(); lane.release(); _feed_rates(m, kind, quantity, unit, label)


@contextmanager
def reserve_legacy(kind, quantity, unit, label):
    model = load_model()
    if kind in model:
        q = quantity * model['gene_factor'] if (kind == 'shape_chain' and quantity is not None) else quantity
        pred = rm.envelope_predict(model[kind], q); source = 'memory_model.json'
    else:
        q = 2 * quantity if (kind == 'shape_chain' and quantity is not None) else quantity
        pred = rate_pred(load_rates(), kind, q); source = 'memory_rates.json'
    m = dict(kind=kind, quantity=quantity, unit=unit, predicted_gb=pred, prediction_source=source); t0 = time.time(); shared = held = False
    lane = _LANES[0]; lane.acquire()
    watch = _wait_watch(m, label)
    try:
        while True:
            with _RES_LOCK:
                d = rm.memory_decision(pred, free_gb(), _RES['outstanding'])
                if d['mode'] == 'shared' and d['go']: _RES['outstanding'] += pred; shared = True
            if d['mode'] == 'exclusive' or shared: break
            time.sleep(10)
        m['mode'] = d['mode']
        if d['mode'] == 'exclusive':
            need = _exclusive_need(pred, label, m)
            with lock('HEAVY_JOB.lock', label, min_free_gb=need):
                with _RES_LOCK: _RES['outstanding'] += EXCLUSIVE_RESERVE_GB; held = True
                watch.cancel(); m['wait_s'] = time.time() - t0; yield m
        else:
            watch.cancel(); m['wait_s'] = time.time() - t0; yield m
    finally:
        watch.cancel(); lane.release()
        with _RES_LOCK:
            if shared: _RES['outstanding'] -= pred
            if held: _RES['outstanding'] -= EXCLUSIVE_RESERVE_GB
        _feed_rates(m, kind, quantity, unit, label)


def cache_lock(path):
    """one thread at a time computes a per-camera cache entry (prepares run on several threads)"""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(Path(path).resolve()), threading.Lock())


def set_aside(path):
    """a partial output directory is renamed, never deleted"""
    path = Path(path)
    if path.exists(): path.rename(path.with_name(path.name + '.partial_' + time.strftime('%Y%m%d_%H%M%S')))


def qual_quantity(s, gate, ci, key):
    """the scene set's qualification record (manifest 'qual'): gate Q5 / Q7, camera ci, field -> millions, else None"""
    if not s.get('qual'): return None
    try:
        q = json.loads(Path(s['qual']).read_text(encoding='utf8'))['gates'][gate]['cameras']
        row = next(c for c in q if int(c['camera']) == int(ci)); return float(row[key]) / 1e6
    except (KeyError, StopIteration, OSError, ValueError):
        return None


def npz_rows(path, key):
    with np.load(path) as z: return z[key].shape[0] / 1e6


# ---------------------------------------------------------------- per scene / camera (cached)
def scene_plan(sd, s):
    """the scene set's Q4 plan (manifest 'plan', copied with its sha256) or camera_plan_r2 (K = 2 per cell on isolated outcrops)"""
    import hashlib, shutil
    p = sd/'plan.json'
    with cache_lock(p):
        if not p.is_file():
            if s.get('plan'):
                shutil.copyfile(s['plan'], p)
                src = dict(source=str(s['plan']), sha256=hashlib.sha256(Path(s['plan']).read_bytes()).hexdigest(), copied=time.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                k = 2 if s['outcrop'] == 'isolated' else 1
                args = [MS/'camera_plan_r2.py', 'plan', s['mechanism'], s['scene_dir'], s.get('net') or '-', p, '--outcrop', s['outcrop'], '--per-cell', k]
                run0(args, log=sd/'plan.log'); src = dict(source='camera_plan_r2.py plan --per-cell %d' % k)
            (sd/'plan_source.json').write_text(json.dumps(src, indent=1) + '\n', encoding='utf8')
    return json.loads(p.read_text(encoding='utf8'))


def camera_of(plan, cam_index, outcrop):
    return rm.camera_for_slot(plan['cameras'], outcrop, cam_index)


def scene_structure_of(sd, s):
    p = Path(sd)/'scene_structure.json'
    with cache_lock(p):
        if not p.is_file():
            cfg = json.loads((Path(s['scene_dir'])/'config.json').read_text(encoding='utf8'))
            dp_ = Path(s['scene_dir'])/'geom/declaration.json'
            dec = json.loads(dp_.read_text(encoding='utf8')) if dp_.is_file() else None
            p.write_text(json.dumps(rm.scene_structure(cfg, dec), indent=1) + '\n', encoding='utf8')
    return json.loads(p.read_text(encoding='utf8'))


def normal_route(s):
    return (Path(s['scene_dir'])/'normal_route.json').is_file()


def normal_route_dir(s):
    return MS/s.get('normal_route_dir', 'host')


def grid_subprocess(s, plan_path, ci, g, ref_az):
    code = ('import sys; sys.path[:0] = %r; import light_predict as lp; lp.grid_r2(%r, %d, %r, %r)'
            % ([str(P1), str(M), str(MS), str(A)], str(plan_path), int(ci), str(g), None if ref_az is None else float(ref_az)))
    qty = npz_rows(Path(json.loads(Path(plan_path).read_text(encoding='utf8'))['geometry'])/'surface.npz', 'faces')
    with reserve('light_grid', qty, 'M coarse faces', 'light grid %s %s' % (s['id'], Path(g).name)) as m:
        rc, out, err = run_measured([sys.executable, '-c', code], env=dict(os.environ), measure=m)
    if rc: raise RuntimeError('light grid failed: ' + (err or out)[-800:])


def light_domain(sd, plan_path, ci, cam_index, ref_az):
    import light_predict as lp
    g = sd/('grid_r2_c%d.npz' % cam_index)
    with cache_lock(g):
        if not g.is_file(): lp.grid_r2(str(plan_path), ci, str(g), ref_az)
    z = np.load(g); return z['accept'], float(z['reference_azimuth_deg'])


def refine(sd, s, plan_path, ci):
    out = sd/'refine'; src = out/('source_%d' % ci)
    with cache_lock(src):
        if src.is_dir(): return src
        out.mkdir(parents=True, exist_ok=True)
        tool = (normal_route_dir(s)/'n_zone_levels.py') if normal_route(s) else (A/'zone_split.py')      # normal route: flat and isolated outcrops
        with reserve('refine', qual_quantity(s, 'Q5', ci, 'faces'), 'M refined faces', 'refine %s cam %d' % (s['id'], ci)) as m:
            run0([tool, s['scene_dir'], plan_path, ci, out, '--workers', 1], log=out/('refine_%d.log' % ci), nogpu=True, measure=m)
            m['actual_quantity'] = npz_rows(src/'source_surface.npz', 'faces')
        (out/('memory_%d.json' % ci)).write_text(json.dumps(m) + '\n', encoding='utf8')
    return src


def field_normals(sd, s, ci, source):
    out = sd/'field'/('field_%d.npz' % ci); rep_path = Path(str(out) + '.json')
    with cache_lock(out):
        if not (out.is_file() and rep_path.is_file()):
            out.parent.mkdir(parents=True, exist_ok=True)
            tool = (normal_route_dir(s)/'n_field_normals.py') if normal_route(s) else (MS/'field_normals.py')
            with reserve('field_normals', npz_rows(Path(source)/'source_surface.npz', 'vertices'), 'M source vertices',
                         'field normals %s cam %d' % (s['id'], ci)) as m:
                run0([tool, source, Path(s['scene_dir'])/'erode_fields.pkl', Path(s['scene_dir'])/'geom', out, '--workers', 1],
                     log=out.parent/('field_%d.log' % ci), nogpu=True, measure=m)
            Path(str(out) + '.memory.json').write_text(json.dumps(m) + '\n', encoding='utf8')
    rep = json.loads(rep_path.read_text(encoding='utf8'))
    r = rep['reevaluation_vs_vertex_m']['max']
    if not r <= FIELD_REEVAL_MAX_M: raise RuntimeError('field normal gate: re-evaluation residual %.3g m > %g m (%s)' % (r, FIELD_REEVAL_MAX_M, out))
    return out, {k: rep[k] for k in ('route', 'computed', 'kept_mesh_normal', 'reevaluation_vs_vertex_m', 'convergence_delta_vs_half_deg',
                                     'field_vs_mesh_normal_deg', 'seconds') if k in rep}


def field_display(case, field_npz, log, measure=None):
    case = Path(case)
    if (case/DISPLAY/'globaldisplay_surface.npz').is_file(): return
    set_aside(case/DISPLAY)
    run0([MS/'field_display.py', case, field_npz], log=log, nogpu=True, measure=measure)


def stain_frame(sd, s, ci, cam_index):
    """the receiver frame of a camera, computed once and read by the light and the geometry steps"""
    p = sd/('frame_%d.json' % cam_index)
    with cache_lock(p):
        if not p.is_file(): run0([HERE/'stain_host.py', 'frame', s['scene_dir'], sd/'plan.json', ci, p], log=sd/('frame_%d.log' % cam_index), nogpu=True)
    return json.loads(p.read_text(encoding='utf8'))


def stain_host_case(sd, s, ci, source, field_npz):
    case = sd/('host_case_%d' % ci)
    with cache_lock(case):
        if not (case/DISPLAY/'globaldisplay_surface.npz').is_file() or not (case/'codes_report.json').is_file():
            with reserve('stain_host', npz_rows(Path(source)/'source_surface.npz', 'faces'), 'M source faces', 'stain host %s cam %d' % (s['id'], ci)) as m:
                if not (case/'codes_report.json').is_file():
                    set_aside(case)
                    run0([HERE/'stain_host.py', 'case', s['scene_dir'], source, case], log=sd/('host_case_%d.log' % ci), nogpu=True, measure=m)
                field_display(case, field_npz, sd/('host_display_%d.log' % ci), measure=m)
            (sd/('host_case_%d.memory.json' % ci)).write_text(json.dumps(m) + '\n', encoding='utf8')
    return case


def stain_domain(case, frame, stain, cam, codes_ok):
    """faces inside the band (|d| < w/2 + soft, the shader's support) with a stainable code whose centroid projects into the
    frame: count and cover (semantic soil) share - recorded only"""
    build = json.loads((Path(case)/'build.json').read_text(encoding='utf8'))
    src = np.load(Path(build['source'])/'source_surface.npz'); nat = np.load(Path(case)/'geometry/native_surface.npz', allow_pickle=True)
    c = src['vertices'][src['faces']].mean(1); q = c - np.asarray(frame['origin_m'], float)
    n = np.asarray(stain['normal'], float); n /= np.linalg.norm(n); t = np.asarray(stain['tangent'], float); t = t - n * (t @ n); t /= np.linalg.norm(t)
    d = q @ n + stain['amplitude_m'] * np.sin(2 * math.pi * (q @ t) / stain['wavelength_m'] + stain['phase'])
    band = (np.abs(d) < .5 * stain['width_m'] + stain['soft_m']) & np.isin(nat['codes'], codes_ok)
    k = raster.camera(cam['shot'], tuple(cam['resolution'])); rel = c[band] - k['pos']; z = rel @ k['fwd']; w, h = cam['resolution']
    x = (rel @ k['right']) / np.where(z > 0, z, np.inf) * k['fpx'] + w / 2; y = -(rel @ k['up']) / np.where(z > 0, z, np.inf) * k['fpx'] + h / 2
    view = (z > k['near']) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    cover = (src['semantic'][band][view] == 'soil') if 'semantic' in src.files else np.zeros(int(view.sum()), bool)
    return dict(band_faces_in_view=int(view.sum()), cover_faces=int(cover.sum()), cover_share=float(cover.mean()) if len(cover) else None,
                note='3D support of the band on stainable codes, centroid in the frustum; occlusion not tested; recorded only')


# ---------------------------------------------------------------- steps
def shape_chain_cmds(s, sd, ci, source, point, name):
    """the four chain commands of a shape case and their log names, shared by geometry and the G6 rebuild; the view-clip
    request sd/request_<ci>.json is written by geometry"""
    req = sd/('request_%d.json' % ci)
    ph = point['physical']; wr = Path(s['scene_dir'])/'wall_relief.json'
    panel = float(s.get('panel_refine_m', SHAPE_BUILD['panel_refine_m']))
    build = ['build.py', name + '_raw', '--width', repr(ph['mean_opening_m']), '--variation', repr(ph['opening_variation']), '--depth', repr(ph['depth_m']),
             '--roughness', repr(ph['roughness_m']), '--construction', SHAPE_BUILD['construction'], '--panel-refine-m', panel,
             '--view-clip', req, '--source', source]
    if wr.is_file(): build += ['--wall-relief', wr, '--wall-relief-tip-m', SHAPE_BUILD['wall_relief_tip_m']]
    return ([(build, 'build.log'), (['repair_host_needles.py', name + '_raw', name], 'repair.log'), (['check_host_provenance.py', name], 'provenance.log'),
             ([EXPORT_NORIM, name], 'export.log')], dict(wall_relief=wr.is_file(), panel_refine_m=panel))


def geometry(st, s, sd, cam, ci, source, point, field_npz, frame=None):
    idx = point['index']; name = 'rq1_%s_%s' % (s['id'], ('p%02d' % idx) if isinstance(idx, int) else str(idx))
    if s['mechanism'] == 'shape':
        net = Path(s['net']); case = net/'cases'/name
        logs = st.dir/'logs'; logs.mkdir(exist_ok=True)
        if not (case/'chain_done.json').is_file():
            req = sd/('request_%d.json' % ci); req.write_text(json.dumps(dict(shots=[cam['shot']], resolution=cam['resolution'])) + '\n', encoding='utf8')
            cmds, info = shape_chain_cmds(s, sd, ci, source, point, name)
            q7 = qual_quantity(s, 'Q7', ci, 'result_faces')
            try:
                with reserve('shape_chain', q7, 'M cut result faces (prediction: Q7 faces)', 'shape chain ' + name) as m:
                    for args, lg in cmds: run0(args, cwd=net, log=logs/lg, nogpu=True, measure=m)
                    m['actual_quantity'] = npz_rows(case/'geometry/native_surface.npz', 'faces')
            except RuntimeError as e:
                if 'ValueError' in str(e) or 'AssertionError' in str(e): return dict(case=str(case), geometry_rejected=True, evidence=str(e)[-600:])
                raise
            (case/'chain_done.json').write_text(json.dumps(dict(steps=['build', 'repair', 'provenance', 'export (export_norim.py: no rim_candidates.json)'],
                                                                wall_relief=info['wall_relief'], panel_refine_m=info['panel_refine_m'],
                                                                saved=time.strftime('%Y-%m-%d %H:%M:%S'))) + '\n', encoding='utf8')
            (logs/'chain_memory.json').write_text(json.dumps(m) + '\n', encoding='utf8')
        if not (case/DISPLAY/'globaldisplay_surface.npz').is_file():
            with reserve('field_display', npz_rows(case/'geometry/native_surface.npz', 'faces'), 'M case faces', 'field display ' + name) as m:
                field_display(case, field_npz, logs/'field_display.log', measure=m)
            (logs/'display_memory.json').write_text(json.dumps(m) + '\n', encoding='utf8')
        return dict(case=str(case), display=DISPLAY, mesh=dict(key=name, native_surface=str(case/'geometry/native_surface.npz'), stain_codes=[],
                                                               display_surface=str(case/DISPLAY/'globaldisplay_surface.npz'), shadow_terminator=dict(geometry_offset=0., shading_offset=0.)))
    import blender_stage as bs
    from outcrop.workbench.scanned import _stain
    case = stain_host_case(sd, s, ci, source, field_npz)
    book = json.loads((Path(s['scene_dir'])/'geom/codebook.json').read_text(encoding='utf8'))
    book = book if isinstance(book, list) else book.get('codebook', book)
    crep = json.loads((case/'codes_report.json').read_text(encoding='utf8'))
    present = sorted(int(c) for c in crep['codes'])
    raw_on, raw_off = bs.raw_stain(point['physical'], frame, False), bs.raw_stain(point['physical'], frame, True)
    eligible = [c for c in present if any(int(r['code']) == c and r['lithology'] in raw_on['lithologies'] for r in book)]
    if not eligible: return dict(case=str(case), geometry_rejected=True, evidence='no stainable code in the host')
    on, off = _stain(raw_on), _stain(raw_off)
    params = lambda sp: {str(c): bs.stain_scalars(sp, True) for c in eligible}  # noqa: E731
    return dict(case=str(case), display=DISPLAY, frame=frame, stain_on=params(on), stain_off=params(off), stain_codes=eligible,
                codes_report={k: crep[k] for k in ('codes', 'lithologies', 'fast_path', 'fast_path_reason', 'truth_queried', 'sample_check',
                                                   'cover_faces', 'peak_working_set_gb', 'seconds') if k in crep},
                stain_domain=stain_domain(case, frame, raw_on, cam, eligible),
                mesh=dict(key='host_%s_%d' % (s['id'], ci), native_surface=str(case/'geometry/native_surface.npz'), stain_codes=eligible,
                          display_surface=str(case/DISPLAY/'globaldisplay_surface.npz'), shadow_terminator=dict(geometry_offset=0., shading_offset=0.)))


def frames_of(s, point, cam, geo, plan, light, repeat=0):
    """repeat R: seeds shape R; stain off / on 2R, off' 2R + 1 (R = 0: the design frames)"""
    import blender_client as bc
    from sliver_render import TABLE
    table = bc.load_table(TABLE)
    if point['sky'] == 'sun':
        lr = bc.light(light['azimuth_deg'], light['elevation_deg'], table); gains = bc.white_balance_gains(lr)
    else:
        lr = bc.overcast_light(table); gains = bc.overcast_gains()
    base = dict(shot=cam['shot'], resolution=cam['resolution'], light=lr, gains=gains, spp=point['spp'], mesh=geo['mesh'], ground=plan['ground'])
    if s['mechanism'] == 'shape': return [dict(base, tag='frame', seed=int(repeat))]
    return [dict(base, tag='off_s0', seed=2 * repeat, stain=geo['stain_off']), dict(base, tag='on_s0', seed=2 * repeat, stain=geo['stain_on']),
            dict(base, tag='off_s1', seed=2 * repeat + 1, stain=geo['stain_off'])]


def standin_labels(s, cam, frame_npz):
    from PIL import Image, ImageDraw
    d = np.load(frame_npz); depth, mask = d['depth'], d['mask']; h, w = depth.shape
    c = raster.camera(cam['shot'], tuple(cam['resolution']))
    img = Image.new('L', (w, h), 0); dr = ImageDraw.Draw(img); length = 0.
    b = json.loads((Path(s['net'])/'source_binding.json').read_text(encoding='utf8'))
    for row in b['exposure_paths']:
        P = np.load(Path(s['net'])/row['file'])['points']; rel = P - c['pos']; z = rel @ c['fwd']; zz = np.where(z > c['near'], z, np.inf)
        x = (rel @ c['right']) / zz * c['fpx'] + w / 2; y = -(rel @ c['up']) / zz * c['fpx'] + h / 2
        ins = (z > c['near']) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        xi, yi = np.clip(x.astype(int), 0, w - 1), np.clip(y.astype(int), 0, h - 1)
        vis = ins & (np.abs(depth[yi, xi] - z) <= .3) & (mask[yi, xi] == 1)
        for i in range(len(P) - 1):
            if vis[i] and vis[i + 1]:
                dr.line([(float(x[i]), float(y[i])), (float(x[i + 1]), float(y[i + 1]))], fill=255, width=1); length += float(np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]))
    target = np.asarray(img) > 0
    return dict(target=target, u=np.zeros_like(target), visible_px=length, kind='stand-in (geometry-only, no U; not trace-labels)')


def lfair(rep):
    """L_fair = visible minus visible-U, the length G gates (B2); S_U, S_occ recorded only (no cap)"""
    rows = rep.get('per_path') or []
    L = {k: float(sum(r.get(f, 0.) or 0. for r in rows)) for k, f in (('L_in', 'projected_in_frame_px'), ('L_vis', 'visible_px'),
                                                                        ('L_occ', 'occluded_px'), ('L_U', 'visible_u_px'))}
    L['L_fair'] = L['L_vis'] - L['L_U']
    L['S_U'] = L['L_U'] / L['L_vis'] if L['L_vis'] > 0 else None
    L['S_occ'] = L['L_occ'] / L['L_in'] if L['L_in'] > 0 else None
    L['visible_length_px_report'] = rep.get('visible_length_px')
    return L


def visible_fair(lb):
    if lb.get('L_fair') is not None: return float(lb['L_fair']), lb
    if lb.get('out') and (Path(lb['out'])/'report.json').is_file():
        lb = dict(lb, **lfair(json.loads((Path(lb['out'])/'report.json').read_text(encoding='utf8')))); return float(lb['L_fair']), lb
    return float(lb.get('visible_px') or 0.), lb


def trace_labels(st, s, sd, ci, geo, source, frame, base_state=None):
    """The pinned copy label_cached.py runs (it saves the A-C state; a noise repeat reads its base point's state and runs D
    only); exit 5 (the entry changed) falls back to label_frame.py. Exit 0 / 3 recorded (3 = a gate stopped: not
    scorable), 2 / 4 raise."""
    out = st.dir/'trace_labels'; ex = out/'exit.json'; state = st.dir/'trace_labels_state.pkl'
    if not ex.is_file():
        set_aside(out)
        common = ['--case', geo['case'], '--request', sd/('request_%d.json' % ci), '--frame', frame['npz'], '--png', frame['png'], '--out', out, '--source', source]
        t0 = time.time(); qty = npz_rows(Path(geo['case'])/'geometry/native_surface.npz', 'faces')
        guard = cache_lock(base_state if base_state is not None else state)        # a repeat waits for its base's state
        name = 'trace labels ' + Path(geo['case']).name
        with guard:
            cache = None
            if base_state is not None:
                ok, why = state_cache_check(base_state); cache = dict(state=str(base_state), ok=ok, reason=why)
            d_only = bool(cache and cache['ok']); kind = 'trace_labels_d' if d_only else 'trace_labels'      # D only: own rate
            with reserve(kind, qty, 'M cut result faces', name) as m:
                if cache is not None: m['state_cache_check'] = cache
                if d_only:
                    cmd, entry = [sys.executable, LABEL_CACHED, '--r4'] + common + ['--state-in', base_state], 'label_cached.py --r4 --state-in'
                else:
                    cmd, entry = [sys.executable, LABEL_CACHED, '--r4'] + common + ['--state-out', state], 'label_cached.py --r4 --state-out'
                rc, so, se = run_measured(cmd, env=dict(os.environ), measure=m)
                if rc == 5:                                        # the pinned copy is stale: the entry itself, no cache
                    m['entry_fallback'] = True
                    set_aside(out); cmd, entry = [sys.executable, LABEL_FRAME_R4] + common, 'label_frame_r4.py (pinned copy stale)'
                    rc, so, se = run_measured(cmd, env=dict(os.environ), measure=m)
                m['actual_quantity'] = npz_rows(Path(geo['case'])/'geometry/native_surface.npz', 'faces')
            if not d_only and not m.get('entry_fallback') and rc in (0, 3) and state.is_file():
                state_sidecar(state).write_text(json.dumps(dict(sha256=_sha256(state), written=time.strftime('%Y-%m-%d %H:%M:%S'), step=name)) + '\n',
                                                encoding='utf8')
        out.mkdir(parents=True, exist_ok=True)
        (out/'label_frame.log').write_text(so[-30000:] + '\n--- stderr ---\n' + se[-30000:], encoding='utf8')
        if rc not in (0, 3):
            raise RuntimeError('%s exit %d: %s' % (entry, rc, (se or so)[-800:]))
        (out/'entry.json').write_text(json.dumps(dict(entry=entry, memory=m)) + '\n', encoding='utf8')
        ex.write_text(json.dumps(dict(exit_code=rc, wall_s=time.time() - t0)) + '\n', encoding='utf8')
    e = json.loads(ex.read_text(encoding='utf8')); rep = json.loads((out/'report.json').read_text(encoding='utf8'))
    fc = scene_structure_of(sd, s).get('fault_count') or 0
    summ = dict(kind='trace-labels r3 (label_frame.py)', exit_code=e['exit_code'], wall_s=e['wall_s'], status=rep.get('status'), stop=rep.get('stop'),
                visible_px_by_type=rm.visible_by_type(rep.get('per_path'), fc, rep.get('per_type')), source_type_map=(out/'source_type.npy').is_file(),
                faults=(rep.get('faults') or {}).get('status'),
                visible_px=rep.get('visible_length_px'), positive_graph_length_px=rep.get('positive_graph_length_px'),
                positive_pixels=rep.get('positive_pixels'), u_line_px=rep.get('u_line_px'), u_windows=rep.get('u_windows'),
                technical_failure_all_u=rep.get('technical_failure_all_u'), ignore_fraction_of_rock=rep.get('ignore_fraction_of_rock'),
                timing_s=rep.get('timing_s'), out=str(out), **lfair(rep))
    if e['exit_code'] == 0:
        extra = dict(ttype=np.load(out/'source_type.npy')) if (out/'source_type.npy').is_file() else {}
        np.savez_compressed(st.dir/'labels.npz', target=np.load(out/'target.npy'), u=np.load(out/'u_line.npy'), valid=np.load(out/'valid.npy'), **extra)
    return summ


def exposure(png, mask):
    from PIL import Image
    from frgb.lighting_rules import CLIP_HIGH, CLIP_LOW, luma
    a = np.asarray(Image.open(png).convert('RGB')); rock = mask == 1
    g = json.loads(EXPOSURE_GATE.read_text(encoding='utf8')); g = g.get('gate', g)
    L = luma(a)[rock]; r8 = a[rock]; clip = float(((r8 <= CLIP_LOW) | (r8 >= CLIP_HIGH)).any(1).mean())
    return dict(mean_luma=float(L.mean()), clipped=clip, gate=[float(g['mean_luma_min']), float(g['clipped_fraction_max'])],
                pass_=bool(L.mean() >= float(g['mean_luma_min']) and clip <= float(g['clipped_fraction_max'])))


class StandInScorer:
    def __init__(self, ckpt):
        self.ckpt, self.model, self.on_gpu = Path(ckpt), None, False

    def _load(self):
        import torch
        from frgb.model import UNetResNet34
        from frgb.train import strict_fp32
        strict_fp32(); ck = torch.load(self.ckpt, map_location='cpu', weights_only=False)
        self.model = UNetResNet34(encoder_weights=None); self.model.load_state_dict(ck['model'], strict=True); self.model.eval()

    def to_gpu(self):
        if self.model is None: self._load()
        if not self.on_gpu: self.model.to('cuda'); self.on_gpu = True

    def offload(self):
        import torch
        if self.model is not None and self.on_gpu:
            self.model.to('cpu'); self.on_gpu = False
            if torch.cuda.is_initialized(): torch.cuda.empty_cache()

    def probability(self, png, purpose):
        from PIL import Image
        from frgb.inference import sliding_window_probability
        self.to_gpu()
        budget.charge(ledger_path(), 'score', 1, purpose)
        rgb = np.asarray(Image.open(png).convert('RGB')).copy()
        return sliding_window_probability(self.model, rgb, device='cuda')['probability']


class GPU:
    def __init__(self, scorer, log_dir):
        self.scorer, self.proc, self.held, self.log_dir = scorer, None, False, Path(log_dir)

    def open(self, label):
        if not self.held: acquire_lock('GPU_JOB.lock', label); self.held = True
        self.scorer.to_gpu()

    def _ask(self, cmd):
        self.proc.stdin.write(json.dumps(cmd) + '\n'); self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line: raise RuntimeError('render server ended (log %s)' % (self.log_dir/'render_server.log'))
            if line.startswith('{'):
                ans = json.loads(line)
                if 'ok' in ans: return ans

    def render(self, spec_path):
        if self.proc is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen([PH0, str(HERE/'render_server.py'), '--queue', str(self.log_dir/'render_server_queue')], env=env0(),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.log_dir/'render_server.log', 'a'), text=True)
        ans = self._ask(dict(op='render', spec=str(spec_path)))
        if not ans['ok']: raise RuntimeError('render server: %s' % ans.get('error'))
        return ans

    def close(self):
        if self.proc is not None: self._ask(dict(op='release'))
        self.scorer.offload()
        if self.held: release_lock('GPU_JOB.lock'); self.held = False

    def quit(self):
        try:
            self.close()
        finally:
            if self.proc is not None:
                try:
                    self._ask(dict(op='quit'))
                except Exception:                                  # noqa: BLE001
                    pass
                self.proc.wait(timeout=120); self.proc = None


def miss_loss(prob, target, u):
    """B6: along the visible non-U centreline pixels, 1 - max probability within 2 px; mean (pixel-length weighted); and the
    thresholded share (max probability < 0.5)"""
    from scipy import ndimage
    yy, xx = np.mgrid[-2:3, -2:3]; disc = (yy ** 2 + xx ** 2) <= 4
    mx = ndimage.maximum_filter(prob, footprint=disc, mode='constant', cval=0.)
    sel = target & ~u
    if not sel.any(): return dict(loss=None, thresholded=None, pixels=0)
    return dict(loss=float((1. - mx[sel]).mean()), thresholded=float((mx[sel] < .5).mean()), pixels=int(sel.sum()))


def _EMPTY_NPZ():
    """an empty npz (labels stopped by a trace-label gate: no target)"""
    import io
    b = io.BytesIO(); np.savez(b); b.seek(0); return b


def point_name(index):
    return ('p%02d' % index) if isinstance(index, (int, np.integer)) else str(index)


def _prepare(man, out, scene_id, index, unit=None, camera=None, sky=None):
    """Returns a context for the GPU and post stages, or a final record (dict with 'final': True) when the point stops here."""
    s = dict(man['scenes'][scene_id], id=scene_id); sd = Path(out)/scene_id; sd.mkdir(parents=True, exist_ok=True)
    st = rm.PointState(out, scene_id, index)
    if unit is None: cam_index, slot, sky = rm.slot_of(index)
    else: cam_index, slot, sky = int(camera), None, sky or 'sun'
    tick = time.time
    # camera
    t = tick(); plan = scene_plan(sd, s); cam = camera_of(plan, cam_index, s['outcrop']); cell, k = rm.camera_slot(s['outcrop'], cam_index)
    if not st.done('camera'):
        st.save('camera', dict(camera_index=cam_index, slot=slot, sky=sky, available=cam is not None, plan_cell=cell, plan_k=k,
                               level=cam['level'] if cam else None, angle_class=cam['angle'] if cam else None,
                               reason=None if cam else 'no plan camera for cell %d, k %d (skipped / failed in the plan)' % (cell, k)), tick() - t)
    if cam is None: return dict(st.load('camera'), final=True)
    ci = plan['cameras'].index(cam)
    frame = stain_frame(sd, s, ci, cam_index) if s['mechanism'] == 'stain' else None
    if frame is not None and frame.get('rejected'):
        if not st.done('geometry'): st.save('geometry', dict(geometry_rejected=True, evidence='stain frame: ' + frame['reason']), 0.)
        return dict(scene=scene_id, index=index, score=None, reason='stain frame rejected: ' + frame['reason'], final=True)
    # light domain and design point
    t = tick()
    if not st.done('light'):
        ref_az = sl.strike_reference(frame['normal_enu'])['azimuth_deg'] if frame is not None else None
        cams = []
        for c4 in range(rm.CAMERAS):
            cc = camera_of(plan, c4, s['outcrop'])
            if cc is None or c4 != cam_index: cams.append(dict(accept=np.ones((24, 11), bool), reference_azimuth_deg=0.)); continue
            acc, ref = light_domain(sd, sd/'plan.json', plan['cameras'].index(cc), c4, ref_az) if sky == 'sun' else (None, None)
            cams.append(dict(accept=acc, reference_azimuth_deg=ref))
        if unit is None:
            units = rm.scene_units(s['mechanism'], s['scene_index'])
            point = rm.map_point(s['mechanism'], units[index], sky, cams[cam_index]['accept'], cams[cam_index]['reference_azimuth_deg'])
            point.update(index=index, camera=cam_index, slot=slot)
        else:
            point = rm.cd_candidate(s['mechanism'], unit, cam_index, cams[cam_index]['accept'], cams[cam_index]['reference_azimuth_deg'], index, sky)
        if sky == 'sun':
            cam_ref = dict(cam, light_reference=sl.strike_reference(frame['normal_enu'])) if frame is not None else cam
            point['light_covariates'] = sl.light_covariates(cam_ref, point['light']['azimuth_deg'], point['light']['elevation_deg'])
        st.save('light', point, tick() - t)
    point = st.load('light')
    # refine (+ field normals of the refined source, per camera)
    t = tick()
    if not st.done('refine'):
        src = refine(sd, s, sd/'plan.json', ci); fn, frep = field_normals(sd, s, ci, src)
        st.save('refine', dict(source=str(src), field_normals=str(fn), field_report=frep), tick() - t)
    rf = st.load('refine'); source = rf['source']
    field_npz = rf.get('field_normals') or str(field_normals(sd, s, ci, source)[0])
    # geometry
    t = tick()
    if not st.done('geometry'): st.save('geometry', geometry(st, s, sd, cam, ci, source, point, field_npz, frame), tick() - t)
    geo = st.load('geometry')
    if geo.get('geometry_rejected'): return dict(scene=scene_id, index=index, score=None, reason='geometry rejected', final=True)
    return dict(s=s, sd=str(sd), st=st, plan=plan, cam=cam, ci=ci, sky=sky, point=point, geo=geo, frame=frame, final=False, repeat=0)


def prepare_point(man, out, scene_id, index, unit=None, camera=None, sky=None, repeat=0):
    """stage 1; repeat R > 0: the base point pNN is prepared (or resumed), then pNN_rR copies its camera, light, refine and
    geometry steps"""
    halted('prepare')
    base_dir = rm.PointState(out, scene_id, index).dir
    with cache_lock(base_dir):
        ctx = _prepare(man, out, scene_id, index, unit, camera, sky)
    if not repeat or ctx.get('final'): return ctx
    st0 = ctx['st']; rname = '%s_r%d' % (point_name(index), int(repeat)); st = rm.PointState(out, scene_id, rname)
    for step in ('camera', 'light', 'refine', 'geometry'):
        if not st.done(step): st.save(step, dict(st0.load(step), repeat_of=point_name(index), repeat=int(repeat)), 0.)
    return dict(ctx, st=st, repeat=int(repeat), base_state=str(st0.dir/'trace_labels_state.pkl'))


def segment_key(item, ctx):
    if ctx.get('final') or ctx['s']['mechanism'] != 'stain': return None
    return (ctx['s']['id'], int(ctx['point']['camera']), int(ctx['point']['spp']))


def gpu_point(ctx, keep, G):
    """stage 2: render (resident render process) and score (resident model), one GPU segment"""
    if ctx.get('final'):
        G.close(); return None
    halted('render')
    s, st, plan, cam, point, geo = (ctx[k] for k in ('s', 'st', 'plan', 'cam', 'point', 'geo'))
    scene_id, pn = s['id'], st.dir.name; tick = time.time
    try:
        if not st.done('render'):
            t = tick(); G.open('RQ1 GPU segment %s %s' % (scene_id, pn))
            spec = dict(out=str(st.dir/'render'), ledger_purpose='%s %s %s' % (note(), scene_id, pn), ledger=str(ledger_path()),
                        frames=frames_of(s, point, cam, geo, plan, point['light'], ctx.get('repeat', 0)))
            (st.dir/'render_spec.json').write_text(json.dumps(spec, indent=1, default=rm._plain) + '\n', encoding='utf8')
            G.render(st.dir/'render_spec.json')
            st.save('render', json.loads((st.dir/'render/render_point.json').read_text(encoding='utf8')), tick() - t)
        rend = st.load('render')['frames']
        if not st.done('score'):
            t = tick(); G.open('RQ1 GPU segment %s %s' % (scene_id, pn)); probs = {}
            for k, v in rend.items():
                probs[k] = G.scorer.probability(v['png'], '%s score %s %s %s' % (note(), scene_id, pn, k))
            np.savez_compressed(st.dir/'probability.npz', **probs)
            st.save('score', dict(model=str(G.scorer.ckpt), frames=list(probs), note=note()), tick() - t)
    except BaseException:
        G.close(); raise
    if not keep(): G.close()
    return dict(rendered=True)


def mixed_pixels(a, b, rock, infl, stain_codes):
    code = b['index']; stainable = rock & np.isin(code, stain_codes); bad = infl & ~np.isin(code, stain_codes)
    h, w = bad.shape; pad = np.pad(stainable, 1); nb = np.zeros_like(stainable)
    for dy in range(3):
        for dx in range(3): nb |= pad[dy:dy + h, dx:dx + w]
    mixed, err = bad & nb, bad & ~nb
    rel = np.abs(b['rgb'] - a['rgb']).max(-1) / np.maximum(a['rgb'].max(-1), 1e-6)
    return dict(mixed_pixels=int(mixed.sum()), mixed_codes={str(c): int((code[mixed] == c).sum()) for c in np.unique(code[mixed])},
                mixed_max_rel_change=float(rel[mixed].max()) if mixed.any() else None, code_error_pixels=int(err.sum()),
                d9_rule='r1f: mixed if a stainable rock pixel is in the 3 x 3 neighbourhood, else a code error')


def post_point(ctx):
    """stage 3: labels, qualify, difficulty, record (CPU)"""
    from frgb.search_objectives import false_positive_increment
    if ctx.get('final'): return {k: v for k, v in ctx.items() if k != 'final'}
    halted('labels')
    s, st, plan, cam, sky, point, geo = (ctx[k] for k in ('s', 'st', 'plan', 'cam', 'sky', 'point', 'geo'))
    scene_id, index = s['id'], point['index']; tick = time.time
    rend = st.load('render')['frames']
    # labels
    t = tick()
    if not st.done('labels'):
        if s['mechanism'] == 'shape' and s.get('labels') == 'standin':
            lab = standin_labels(s, cam, rend['frame']['npz'])
            np.savez_compressed(st.dir/'labels.npz', target=lab['target'], u=lab['u'])
            st.save('labels', dict(kind=lab['kind'], visible_px=lab['visible_px'], centreline_pixels=int(lab['target'].sum())), tick() - t)
        elif s['mechanism'] == 'shape':
            st.save('labels', trace_labels(st, s, Path(ctx['sd']), ctx['ci'], geo, st.load('refine')['source'], rend['frame'], ctx.get('base_state')), tick() - t)
        else:
            a, b = np.load(rend['off_s0']['npz']), np.load(rend['on_s0']['npz'])
            rock = (a['mask'] == 1) & (b['mask'] == 1)
            infl = rock & (np.abs(b['rgb'] - a['rgb']).max(-1) > .01 * np.maximum(a['rgb'].max(-1), 1e-6))
            np.savez_compressed(st.dir/'labels.npz', rock=rock, influence=infl)
            idx = b['index'][infl]; ok = np.isin(idx, geo['stain_codes'])
            d9 = mixed_pixels(a, b, rock, infl, geo['stain_codes'])
            lab = dict(kind='stain M_N = rock & influence (on vs off0 differ > 1 % linear)', rock=int(rock.sum()), influence=int(infl.sum()),
                       influence_codes={str(c): int((idx == c).sum()) for c in np.unique(idx)}, influence_on_stainable_codes=int(ok.sum()),
                       influence_off_stainable_codes=int((~ok).sum()), **d9)
            if d9['code_error_pixels']:
                (st.dir/'labels_code_error.json').write_text(json.dumps(lab, default=rm._plain) + '\n', encoding='utf8')
                _MODE['halt'] = 'D9 (r1f): %d non-stainable influence pixels without a stainable neighbour at %s: code error' % (d9['code_error_pixels'], st.dir)
                raise D9CodeError(_MODE['halt'])
            st.save('labels', lab, tick() - t)
    # qualify
    t = tick()
    if not st.done('qualify'):
        exp = {k: exposure(v['png'], np.load(v['npz'])['mask']) for k, v in rend.items()}
        q = dict(exposure=exp, exposure_rule='sunny: r14 gate; overcast: recorded only (the r14 gate is not applied to overcast frames)')
        if s['mechanism'] == 'shape':
            lb = st.load('labels'); q['visible_px'] = lb.get('visible_px') or 0.
            vis, lbf = visible_fair(lb); q['visible_fair_px'] = vis; q['visible_ok'] = vis >= VISIBLE_PX_MIN    # B2: unoccluded, non-U length
            q['trace_shares'] = {k: lbf.get(k) for k in ('L_in', 'L_vis', 'L_occ', 'L_U', 'S_U', 'S_occ')}
            q['labels_ok'] = lb.get('exit_code', 0) == 0 and (lb.get('positive_pixels') is None or lb['positive_pixels'] > 0)
            if lb.get('exit_code') == 3: q['labels_stop'] = lb.get('stop')
            q['pass_'] = q['visible_ok'] and q['labels_ok'] and (sky == 'overcast' or exp['frame']['pass_'])
        else:
            from PIL import Image
            from frgb.gate import check_pair
            L = np.load(st.dir/'labels.npz'); im = {k: np.asarray(Image.open(v['png']).convert('RGB')).astype(float) for k, v in rend.items()}
            cp = check_pair(im['off_s0'], im['on_s0'], L['rock'], L['influence'], im['off_s0'], im['off_s1'], True, frozen_threshold_8bit=2.)
            q['check_pair'] = {k: (v if isinstance(v, (bool, int, float, str, type(None))) else str(v)) for k, v in cp.items()}
            q['pass_'] = bool(cp.get('pass')) and (sky == 'overcast' or exp['on_s0']['pass_'])
        st.save('qualify', q, tick() - t)
    q = st.load('qualify')
    P = np.load(st.dir/'probability.npz')
    # difficulty
    t = tick()
    if not st.done('difficulty'):
        L = np.load(st.dir/'labels.npz') if (st.dir/'labels.npz').is_file() else np.load(_EMPTY_NPZ())
        if s['mechanism'] == 'shape':
            fc = scene_structure_of(ctx['sd'], s).get('fault_count') or 0
            d = (rm.typed_miss_loss(P['frame'], L['target'], L['u'], L['ttype'] if 'ttype' in L.files else None, fc) if 'target' in L.files
                 else dict(loss=None, thresholded=None, pixels=0, reason='no labels (gate stop)'))
        else:
            dom = L['influence'].astype(np.uint8)
            if dom.any():
                on, rep = false_positive_increment(P['on_s0'], P['off_s0'], dom), false_positive_increment(P['off_s1'], P['off_s0'], dom)
                d = dict(delta_h=on['score'], tolerance=abs(rep['score']), domain_pixels=on['domain_pixels'], exceeds_tolerance=bool(on['score'] > abs(rep['score'])))
            else:
                d = dict(delta_h=None, tolerance=None, domain_pixels=0, reason='empty influence domain')
        d['scorable'] = bool(q['pass_'])
        st.save('difficulty', dict(d, note=note()), tick() - t)
    # record
    t = tick()
    if not st.done('record'):
        steps = {k: st.load(k).get('seconds') for k in rm.STEPS if st.done(k)}
        rf = st.load('refine')
        extra = dict(display=geo.get('display'), field_report=rf.get('field_report'))
        if s['mechanism'] == 'stain':
            fr = geo['frame']
            extra.update(stain_frame={k: fr.get(k) for k in ('origin_lithology', 'layer_normal_enu', 'normal_enu', 'tangent_enu', 'surface_to_bedding_deg',
                                                             'degenerate', 'trilinear_bedding')},
                         codes_report=geo.get('codes_report'), stain_domain=geo.get('stain_domain'))
        cs = st.load('camera')
        extra['scene_structure'] = scene_structure_of(ctx['sd'], s)
        extra.update(plan_cell=cs.get('plan_cell'), plan_k=cs.get('plan_k'), level=cam['level'], angle_class=cam['angle'],
                     cell=rm.cell_label(s['outcrop'], cam['level']))
        if ctx.get('repeat'): extra.update(repeat=ctx['repeat'], repeat_of=point_name(index))
        st.save('record', dict(scene=scene_id, mechanism=s['mechanism'], outcrop=s['outcrop'], index=index, camera=point['camera'], sky=sky,
                               unit=point['unit'], camera_covariates=cam['covariates'], pixel_mm=cam['pixel_mm'], angle_deg=cam['angle_deg'],
                               face_type=cam['face_type'], light=point['light'], light_covariates=point.get('light_covariates'), physical=point['physical'],
                               charges=rm.point_charges(s['mechanism'], models=1), seconds_by_step=steps, note=note(), **extra), tick() - t)
    return st.load('record')


def prestage_grids(man, out, items, workers, log=print):
    from concurrent.futures import ThreadPoolExecutor
    tasks = {}
    for scene_id, index, repeat in items:
        if repeat or not isinstance(index, (int, np.integer)): continue
        cam_index, _, sky = rm.slot_of(index)
        if sky != 'sun': continue
        s = dict(man['scenes'][scene_id], id=scene_id); sd = Path(out)/scene_id; sd.mkdir(parents=True, exist_ok=True)
        plan = scene_plan(sd, s); cam = camera_of(plan, cam_index, s['outcrop'])
        if cam is None: continue
        tasks[(scene_id, cam_index)] = (s, sd, plan['cameras'].index(cam))

    def task(key):
        (scene_id, cam_index), (s, sd, ci) = key, tasks[key]; g = sd/('grid_r2_c%d.npz' % cam_index)
        with cache_lock(g):
            if g.is_file(): return 'cached'
            ref_az = None
            if s['mechanism'] == 'stain':
                fr = stain_frame(sd, s, ci, cam_index)
                if fr.get('rejected'): return 'frame rejected'
                ref_az = sl.strike_reference(fr['normal_enu'])['azimuth_deg']
            t0 = time.time(); grid_subprocess(s, sd/'plan.json', ci, g, ref_az); return 'computed %.0f s' % (time.time() - t0)
    with ThreadPoolExecutor(max(1, int(workers))) as ex:
        res = dict(zip(tasks, ex.map(task, list(tasks))))
    log(dict(event='grids', result={'%s:%d' % k: v for k, v in res.items()}))
    return res


def array_hashes(path):
    out = {}
    with np.load(path, allow_pickle=True) as z:
        for k in sorted(z.files):
            a = z[k]; h = hashlib.sha256()
            h.update(k.encode('utf8')); h.update(str(a.dtype).encode('utf8')); h.update(repr(a.shape).encode('utf8'))
            h.update(repr(a.tolist()).encode('utf8') if a.dtype == object else np.ascontiguousarray(a).tobytes())
            out[k] = h.hexdigest()
    return out


def g6_targets(case):
    case = Path(case); raw = case.with_name(case.name + '_raw')
    return [case/'difference.npz', raw/'difference.npz'] + sorted((case/'geometry').glob('*.npz'))


def g6_level(outcrop, cam_index):
    return 0 if rm.camera_slot(outcrop, cam_index)[0] in (0, 1) else 1


def g6_audit(man, out):
    rows = {}
    for sid, s in sorted(man['scenes'].items()):
        if s['mechanism'] != 'shape': continue
        strata = {}
        for i in range(rm.POINTS_PER_SCENE): strata.setdefault(g6_level(s['outcrop'], rm.slot_of(i)[0]), []).append(i)
        keep = []
        for lv, idx in sorted(strata.items()):
            rng = np.random.default_rng([G6_SEED, int(rm.MECHANISMS[s['mechanism']]), int(s['scene_index']), int(lv)])
            keep += [int(x) for x in rng.choice(idx, len(idx) // G6_PER, replace=False)]
        rows[sid] = sorted(keep)
    fp = Path(out)/'g6_audit_points.json'
    if fp.is_file():
        old = json.loads(fp.read_text(encoding='utf8'))['points']
        if any(old[k] != v for k, v in rows.items() if k in old): raise SystemExit('G6 audit list differs from the frozen one (%s)' % fp)
        rows = dict(old, **rows)
    fp.write_text(json.dumps(dict(rule=str(G6_RULE), rule_sha256=G6_RULE_SHA, seed=G6_SEED, per=G6_PER, points=rows), indent=1) + '\n', encoding='utf8')
    return rows


def g6_code_state(net):
    import render_batch_fault as rbf
    tpl = Path(net)/'template_sha256.txt'
    return dict(driver_sha256=_sha256(Path(__file__)), builder={k: v for k, v in rbf.builder_record(Path(net)).items() if k != 'at'},
                template_sha256=tpl.read_text(encoding='utf8') if tpl.is_file() else None, python=PH0)


def g6_point(man, out, sid, pn):
    """the pieces of a recorded shape point the chain commands need"""
    s = dict(man['scenes'][sid], id=sid); sd = Path(out)/sid; st = rm.PointState(out, sid, pn)
    plan = scene_plan(sd, s); cam = camera_of(plan, st.load('camera')['camera_index'], s['outcrop'])
    return s, sd, st, plan['cameras'].index(cam), st.load('refine')['source'], st.load('light'), st.load('geometry')


def g6_slim(man, out):
    audit = g6_audit(man, out); done = []
    for sid, s in sorted(man['scenes'].items()):
        if s['mechanism'] != 'shape': continue
        noise = {i for i, _ in rm.noise_points(s['mechanism'], s['scene_index'])['items']}
        for i in range(rm.POINTS_PER_SCENE):
            d0 = Path(out)/sid/point_name(i)
            if i in audit.get(sid, []) or not (d0/'record.json').is_file() or (d0/'slim.json').is_file(): continue
            if i in noise and not all((Path(out)/sid/('%s_r%d' % (point_name(i), r))/'record.json').is_file() for r in rm.NOISE_REPEATS): continue
            s_, sd, st, ci, source, point, geo = g6_point(man, out, sid, point_name(i))
            if geo.get('geometry_rejected') or not geo.get('case'): continue
            case = Path(geo['case']); files = [f for f in g6_targets(case) if f.is_file()]
            cmds, _ = shape_chain_cmds(s_, sd, ci, source, point, case.name)
            rec = dict(rule=str(G6_RULE), rule_sha256=G6_RULE_SHA, case=str(case), at=time.strftime('%Y-%m-%d %H:%M:%S'), code=g6_code_state(s_['net']),
                       commands=[[str(a) for a in c] for c, _ in cmds],
                       files=[dict(path=str(f), size=f.stat().st_size, file_sha256=_sha256(f), arrays=array_hashes(f)) for f in files])
            tmp = d0/'slim.json.tmp'; tmp.write_text(json.dumps(rec, indent=1) + '\n', encoding='utf8'); tmp.replace(d0/'slim.json')
            for f in files: f.unlink()
            done.append(dict(point='%s/%s' % (sid, point_name(i)), files=len(files), bytes=sum(r['size'] for r in rec['files'])))
    return done


def g6_check(man, out, batch):
    pool = sorted(p.parent for p in Path(out).glob('*/*/slim.json'))
    if not pool: return None
    d0 = pool[int(np.random.default_rng([G6_SEED, int(batch)]).integers(len(pool)))]
    rec = json.loads((d0/'slim.json').read_text(encoding='utf8')); sid, pn = d0.parent.name, d0.name
    s, sd, st, ci, source, point, geo = g6_point(man, out, sid, pn)
    now, was = g6_code_state(s['net']), rec['code']
    res = dict(batch=int(batch), point='%s/%s' % (sid, pn), code_same=now == was)
    if now != was: res['result'] = 'FAIL (code changed since the slim: not rebuilt)'; return res
    case = Path(rec['case']); name = case.name + '_g6check'; net = Path(s['net']); scratch = net/'cases'/name; sraw = net/'cases'/(name + '_raw')
    for p_ in (scratch, sraw):
        if p_.exists(): shutil.rmtree(p_)
    cmds, _ = shape_chain_cmds(s, sd, ci, source, point, name); logs = st.dir/'logs'; logs.mkdir(exist_ok=True)
    try:
        with reserve('shape_chain', qual_quantity(s, 'Q7', ci, 'result_faces'), 'M cut result faces (prediction: Q7 faces)', 'G6 check ' + name) as m:
            for args, lg in cmds: run0(args, cwd=net, log=logs/('g6check_' + lg), nogpu=True, measure=m)
    except RuntimeError as e:
        res.update(result='FAIL (rebuild failed)', evidence=str(e)[-600:]); return res
    raw = case.with_name(case.name + '_raw'); cmp_ = []
    for r in rec['files']:
        f = Path(r['path']); g = sraw/f.relative_to(raw) if raw in f.parents else scratch/f.relative_to(case)
        a = array_hashes(g) if g.is_file() else {}
        cmp_.append(dict(path=r['path'], rebuilt=str(g), differ=sorted(k for k in set(a) | set(r['arrays']) if a.get(k) != r['arrays'].get(k))))
    res.update(compare=cmp_, result='PASS' if all(not c['differ'] for c in cmp_) else 'FAIL (per-array hashes differ)')
    for p_ in (scratch, sraw): shutil.rmtree(p_)
    return res


def run_points(man, out, items, scorer, cpu_workers=1, post_workers=1, grid_workers=1, log=print):
    if grid_workers: prestage_grids(man, out, items, grid_workers, log)
    G = GPU(scorer, Path(out)/'_gpu')
    try:
        return rm.pipeline3(items, lambda p: prepare_point(man, out, p[0], p[1], repeat=p[2]), lambda p, ctx, keep: gpu_point(ctx, keep, G),
                            lambda p, ctx, g: post_point(ctx), cpu_workers=cpu_workers, post_workers=post_workers, segment_key=segment_key, log=log)
    finally:
        G.quit()


def parse_item(a, man=None):
    """SCENE:INDEX, SCENE:INDEX:rR, or SCENE:noise"""
    parts = a.split(':')
    if parts[1] == 'noise':
        s = man['scenes'][parts[0]]
        return [(parts[0], i, r) for i, r in rm.noise_points(s['mechanism'], s['scene_index'])['items']]
    return [(parts[0], int(parts[1]), int(parts[2][1:]) if len(parts) > 2 else 0)]


def check_absolute(argv, man):
    rel = [x for x in argv[:2] if not Path(x).is_absolute()]
    rel += ['%s.%s' % (sid, k) for sid, sc in man.get('scenes', {}).items() for k in ('scene_dir', 'net', 'plan', 'qual')
            if sc.get(k) and not Path(sc[k]).is_absolute()]
    if rel: raise SystemExit('relative paths are not allowed: %s' % rel)


def main(argv):
    if not Path(argv[0]).is_absolute(): raise SystemExit('relative paths are not allowed: %s' % argv[0])
    man = json.loads(Path(argv[0]).read_text(encoding='utf8')); check_absolute(argv, man); out = Path(argv[1]); out.mkdir(parents=True, exist_ok=True)
    opts = {}; rest = []; i = 2
    while i < len(argv):
        if argv[i] in ('--legacy-reserve', '--formal', '--g6'): opts[argv[i][2:]] = True; i += 1
        elif argv[i].startswith('--'): opts[argv[i][2:]] = argv[i + 1]; i += 2
        else: rest.append(argv[i]); i += 1
    items = [it for a in rest if ':' in a for it in parse_item(a, man)]
    cw = int(opts.get('cpu-workers', 1)); pw = int(opts.get('post-workers', cw)); gw = int(opts.get('grid-workers', 1))
    _LANES[0] = threading.BoundedSemaphore(int(opts.get('lanes', 1)))
    _MODE.update(legacy=bool(opts.get('legacy-reserve')), lanes=int(opts.get('lanes', 1)), formal=bool(opts.get('formal')), g6=bool(opts.get('g6')))
    g6log = out/'g6_batches.jsonl'
    if _MODE['g6']:
        if not _MODE['formal']: raise SystemExit('--g6 needs --formal (r1f)')
        if _sha256(G6_RULE) != G6_RULE_SHA: raise SystemExit('r1f rule changed')
        if not (G6_VERIFIED.is_file() and json.loads(G6_VERIFIED.read_text(encoding='utf8')).get('result') == 'PASS'): raise SystemExit('G6 pre-formal check not passed (r1f)')
        last = g6log.read_text(encoding='utf8').splitlines()[-1:] if g6log.is_file() else []
        if last and (json.loads(last[0]).get('check') or {}).get('result', 'PASS') != 'PASS': raise SystemExit('the last G6 batch check failed (r1f): stopped')
    scorer = StandInScorer(opts['model']); t0 = time.time()

    def log(ev):
        print('EVENT', json.dumps(ev, default=str), flush=True)
    res = run_points(man, out, items, scorer, cw, pw, gw, log)
    if _MODE['g6'] and not _MODE['halt']:
        batch = (len(g6log.read_text(encoding='utf8').splitlines()) if g6log.is_file() else 0) + 1
        slim = g6_slim(man, out); chk = g6_check(man, out, batch)
        with open(g6log, 'a', encoding='utf8') as f: f.write(json.dumps(dict(batch=batch, at=time.strftime('%Y-%m-%d %H:%M:%S'), slimmed=slim, check=chk), default=str) + '\n')
        if chk and chk['result'] != 'PASS': raise SystemExit('G6 batch check failed (r1f): %s' % chk['result'])
    for r in res:
        print('POINT', r['item'], json.dumps({k: (r.get('result') or {}).get(k) for k in ('seconds_by_step', 'charges')}), r.get('error'), flush=True)
    (out/'dryrun_summary.json').write_text(json.dumps(dict(points=res, cpu_workers=cw, post_workers=pw, grid_workers=gw, wall_s=time.time() - t0, note=note(),
                                                           halt=_MODE['halt']), indent=1, default=rm._plain) + '\n', encoding='utf8')
    if _MODE['halt']: raise SystemExit('driver stopped for investigation: ' + _MODE['halt'])


if __name__ == '__main__':
    main(sys.argv[1:])
