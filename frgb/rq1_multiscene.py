"""- Design: one independently Owen-scrambled Sobol set of 32 points x 6 genes per scene; point i belongs to camera i //
8 (each 8-point block of the 2^5 net is itself a balanced 2^3 net); slots 6 and 7 of every block are overcast (6 sunny
+ 2 overcast per camera), whose two light genes are recorded but not used. - Samples per frame: sunny
1024, overcast 2048. - Estimation over scenes: per-scene estimates, mean over R scenes, variance = between-scene
sample variance / R, interval with t_0.975(R - 1). - PointState: one directory per point, one JSON per finished step
(resumable). - pipeline: point-level CPU parallelism - prepares on cpu_workers threads, finishes serially in
submission order. - camera_slot / camera_for_slot: slot -> plan camera (isolated outcrops: 4 fine cameras, K = 2);
cell_estimates / equal_weight_estimate: per-cell estimates and the equal-weight overall estimate. - pipeline3: prepare
/ GPU / post stages; predicted_peak_gb, memory_decision, update_rate."""
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from frgb import difficulty_params as dp
from frgb import scene_light as sl
from frgb.cd_search import MECHANISMS

SCHEMA = 'rq1-multiscene-r1'
DEFAULT_SEED = 20261030
SCENE_STREAM = 11
BASE_CAMERAS = 4
CAMERAS = BASE_CAMERAS                                       # + near cameras, only through configure_design
POINTS_PER_CAMERA = 8
POINTS_PER_SCENE = CAMERAS * POINTS_PER_CAMERA
NEAR_CAMERAS = 0
NEAR_STREAM = 12
OVERCAST_SLOTS = (6, 7)
ISOLATED_SLOTS = ((0, 0), (1, 0), (0, 1), (1, 1))
SPP = dict(sun=1024, overcast=2048)
FRAMES = dict(shape=1, stain=3)
STEPS = ('camera', 'refine', 'geometry', 'light', 'render', 'labels', 'qualify', 'score', 'difficulty', 'record')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069,
        24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def scene_seed(mechanism, scene_index, seed=DEFAULT_SEED):
    return np.random.SeedSequence([int(seed), MECHANISMS[mechanism], SCENE_STREAM, int(scene_index)])


def configure_design(near_cameras=0, spp_sun=1024):
    """The near level as a parameter. near_cameras 1 or 2 adds that many camera slots after the 4 existing ones, each with
    its 8 points; the existing 32 points do not move, the near points come from their own scrambled net (a power of
    two is what owen_sobol takes). spp_sun: the sunny samples per frame. Returns the settings. The driver reads these
    module values when it runs, so call this before it starts."""
    global CAMERAS, POINTS_PER_SCENE, NEAR_CAMERAS
    if near_cameras not in (0, 1, 2): raise ValueError('near_cameras must be 0, 1 or 2')
    NEAR_CAMERAS = int(near_cameras)
    CAMERAS = BASE_CAMERAS + NEAR_CAMERAS
    POINTS_PER_SCENE = CAMERAS * POINTS_PER_CAMERA
    SPP['sun'] = int(spp_sun)
    return dict(cameras=CAMERAS, points_per_scene=POINTS_PER_SCENE, near_cameras=NEAR_CAMERAS, spp=dict(SPP))


def scene_units(mechanism, scene_index, seed=DEFAULT_SEED, n=None):
    from frgb.rq1_design import owen_sobol, DIMENSION
    if n is None and NEAR_CAMERAS:
        base = owen_sobol(BASE_CAMERAS * POINTS_PER_CAMERA, DIMENSION, scene_seed(mechanism, scene_index, seed))
        near = owen_sobol(NEAR_CAMERAS * POINTS_PER_CAMERA, DIMENSION, np.random.SeedSequence([int(seed), MECHANISMS[mechanism], NEAR_STREAM, int(scene_index)]))
        return np.vstack([base, near])
    return owen_sobol(POINTS_PER_SCENE if n is None else n, DIMENSION, scene_seed(mechanism, scene_index, seed))


def slot_of(index):
    """point index in the scene -> (camera, slot, sky)"""
    index = int(index)
    if not 0 <= index < POINTS_PER_SCENE: raise ValueError('point index out of the scene design')
    camera, slot = divmod(index, POINTS_PER_CAMERA)
    return camera, slot, 'overcast' if slot in OVERCAST_SLOTS else 'sun'


def map_point(mechanism, unit, sky, accept=None, reference_az=None):
    """unit genes (6) -> physical parameters, analysis coordinates and the light"""
    u = np.asarray(unit, float)
    if u.shape != (6,) or np.any((u < 0) | (u > 1)): raise ValueError('6 unit genes in [0, 1] required')
    f = (dp.shape_forward if mechanism == 'shape' else dp.stain_forward)(u[:4])
    out = dict(mechanism=mechanism, unit=u.tolist(), sky=sky, physical=f['physical'], analysis=dict(f['analysis']), spp=SPP[sky])
    if sky == 'sun':
        if accept is None or reference_az is None: raise ValueError('a sunny point needs the light domain and the reference azimuth')
        rel, el = sl.map_light(u[4], u[5], accept)
        out['light'] = dict(relative_azimuth_deg=float(rel[0]), elevation_deg=float(el[0]), reference_azimuth_deg=float(reference_az),
                            azimuth_deg=float((reference_az + rel[0]) % 360.), prior_density=float(sl.density(rel, el, accept)[0]))
        out['analysis'].update(relative_azimuth_deg=out['light']['relative_azimuth_deg'], elevation_deg=out['light']['elevation_deg'])
    elif sky == 'overcast':
        out['light'] = dict(mode='overcast', unused_light_genes=u[4:].tolist())
    else:
        raise ValueError('sky must be sun or overcast')
    return out


def cd_candidate(mechanism, unit, camera, accept, reference_az, point_id, sky='sun'):
    if not str(point_id): raise ValueError('a candidate needs an id')
    p = map_point(mechanism, unit, sky, accept, reference_az)
    return dict(p, index=str(point_id), camera=int(camera), slot=None, source='cd_candidate')


def train_round(model_path, pool, review_pool, round_index, *, real_cache=None, **kw):
    """Training is not part of this release: the module that does the work, frgb.trace_train, is not included, and the
    real data slots need a training cache (real_cache). Without real_cache this raises NotImplementedError; with it the
    import of frgb.trace_train fails."""
    if real_cache is None:
        raise NotImplementedError('training is not part of this release (frgb.trace_train is not included)')
    from frgb import trace_train
    return trace_train.train_round(model_path, pool, review_pool, round_index, real_cache=real_cache, **kw)


def scene_design(mechanism, scene_index, cameras, seed=DEFAULT_SEED):
    """cameras: 4 dicts with 'accept' (24 x 11 bool) and 'reference_azimuth_deg' -> the scene's 32 mapped points"""
    if len(cameras) != CAMERAS: raise ValueError('%d cameras per scene' % CAMERAS)
    units = scene_units(mechanism, scene_index, seed)
    pts = []
    for i, u in enumerate(units):
        cam, slot, sky = slot_of(i)
        c = cameras[cam]
        p = map_point(mechanism, u, sky, c.get('accept'), c.get('reference_azimuth_deg'))
        pts.append(dict(p, scene=int(scene_index), index=i, camera=cam, slot=slot))
    return pts


def t975(df):
    df = int(df)
    if df < 1: raise ValueError('at least 2 scenes')
    return T975.get(df, 1.959964 + 2.37 / df)


def replicate_estimate(per_scene):
    """per-scene estimates (NaN = scene without an estimate, excluded and counted) -> mean, between-scene variance / R, t interval"""
    x = np.asarray(per_scene, float); ok = np.isfinite(x); r = int(ok.sum())
    out = dict(scenes=int(len(x)), scenes_used=r, mean=float(x[ok].mean()) if r else None)
    if r >= 2:
        var = float(x[ok].var(ddof=1) / r); half = t975(r - 1) * math.sqrt(var)
        out.update(variance=var, se=math.sqrt(var), ci95=[out['mean'] - half, out['mean'] + half], t=t975(r - 1))
    return out


def hard_fraction_by_scene(scores_by_scene, tau):
    """joint fraction P(scorable and s > tau) per scene (NaN = unscorable, counted in the denominator), then replicate_estimate"""
    per = []
    for s in scores_by_scene:
        s = np.asarray(s, float)
        per.append(float(np.mean(np.isfinite(s) & (np.nan_to_num(s, nan=-np.inf) > tau))) if len(s) else np.nan)
    return dict(replicate_estimate(per), per_scene=per, tau=float(tau))


def ledger_caps(n_scenes=16, n_noise=64, n_ce=256, m=1.25, k=3, mechanisms=('shape', 'stain')):
    n = n_scenes * POINTS_PER_SCENE
    rgb = sum((n + n_noise + n_ce) * FRAMES[mech] * m for mech in mechanisms)
    return dict(N=n, N_noise=n_noise, N_CE=n_ce, m=m, K=k, frames=dict(FRAMES), rq1_rgb=int(round(rgb)), rq1_score=int(round(rgb * k)))


def point_charges(mechanism, models=1, retries=0):
    """rgb and score counts of one point (frames x (1 + retries); every frame scored by every model)"""
    f = FRAMES[mechanism] * (1 + int(retries))
    return dict(rgb=f, score=f * int(models))


class PointState:
    """resumable per-point state: OUT/<scene>/<point>/<step>.json"""

    def __init__(self, root, scene, index):
        name = ('p%02d' % index) if isinstance(index, (int, np.integer)) else str(index)
        self.dir = Path(root)/str(scene)/name; self.dir.mkdir(parents=True, exist_ok=True)

    def done(self, step):
        return (self.dir/(step + '.json')).is_file()

    def load(self, step):
        return json.loads((self.dir/(step + '.json')).read_text(encoding='utf8'))

    def save(self, step, data, seconds=None):
        if step not in STEPS: raise ValueError('unknown step ' + step)
        rec = dict(data, step=step, seconds=seconds, saved=time.strftime('%Y-%m-%d %H:%M:%S'))
        tmp = self.dir/(step + '.json.tmp'); tmp.write_text(json.dumps(rec, indent=1, default=_plain) + '\n', encoding='utf8'); tmp.replace(self.dir/(step + '.json'))
        return rec

    def next_step(self):
        return next((s for s in STEPS if not self.done(s)), None)


def _plain(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def pipeline(items, prepare, finish, cpu_workers=1, log=None):
    """prepare(item) on cpu_workers threads (all submitted in order), finish(item, prepared) on the calling thread, one at a
    time, in submission order: with one worker the next point's prepare overlaps this point's finish. An exception is recorded
    in that item's result (stage, error, traceback) and the run continues; finish is not called after a failed prepare."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()

    def ev(kind, i):
        if log: log(dict(event=kind, item=i, t=time.time() - t0))

    def prep(i, item):
        ev('prepare_start', i)
        try:
            return prepare(item)
        finally:
            ev('prepare_end', i)
    out = []
    with ThreadPoolExecutor(max(1, int(cpu_workers))) as ex:
        futures = [ex.submit(prep, i, it) for i, it in enumerate(items)]
        for i, (it, fu) in enumerate(zip(items, futures)):
            try:
                ctx = fu.result()
            except Exception as e:                               # noqa: BLE001
                out.append(dict(item=it, stage='prepare', error=repr(e), traceback=traceback.format_exc()[-3000:])); ev('error', i); continue
            ev('finish_start', i)
            try:
                out.append(dict(item=it, result=finish(it, ctx)))
            except Exception as e:                               # noqa: BLE001
                out.append(dict(item=it, stage='finish', error=repr(e), traceback=traceback.format_exc()[-3000:])); ev('error', i)
            ev('finish_end', i)
    return out


def _by_scene(scene_ids, *arrays):
    ids = list(scene_ids); order = list(dict.fromkeys(ids))
    groups = [np.flatnonzero(np.array([x == sid for x in ids])) for sid in order]
    return order, [[np.asarray(a)[g] for a in arrays] for g in groups]


def scene_estimates(scene_ids, scores, taus):
    """per scene (NaN score = unscorable): scorable fraction, mean of the scorable scores, joint P(scorable and s > tau),
    conditional P(s > tau | scorable); each combined over scenes by replicate_estimate (a scene without an estimate - no
    scorable point for the mean and the conditional fractions - is excluded and counted)"""
    order, groups = _by_scene(scene_ids, np.asarray(scores, float))
    taus = [float(t) for t in taus]; per = dict(scorable=[], mean=[], joint={t: [] for t in taus}, conditional={t: [] for t in taus}, n=[])
    for (s,) in groups:
        ok = np.isfinite(s); per['n'].append(int(len(s))); per['scorable'].append(float(ok.mean()) if len(s) else np.nan)
        per['mean'].append(float(s[ok].mean()) if ok.any() else np.nan)
        for t in taus:
            hit = ok & (np.where(ok, s, -np.inf) > t)
            per['joint'][t].append(float(hit.mean()) if len(s) else np.nan)
            per['conditional'][t].append(float(hit.sum() / ok.sum()) if ok.any() else np.nan)

    def est(x):
        x = np.asarray(x, float)
        return dict(replicate_estimate(x), per_scene=x.tolist(), scenes_dropped=int(np.sum(~np.isfinite(x))))
    return dict(scenes=order, points_per_scene=per['n'], scorable=est(per['scorable']), mean_scorable=est(per['mean']),
                hard=[dict(tau=t, joint=est(per['joint'][t]), conditional=est(per['conditional'][t])) for t in taus],
                method='per-scene estimates, mean over scenes, between-scene variance / R, t_0.975(R - 1) (replicate_estimate)')


def scene_icc(scene_ids, scores):
    """one-way random-effects ANOVA on the scorable scores grouped by scene: ICC(1) = (MSB - MSW) / (MSB + (n0 - 1) MSW),
    n0 = (N - sum n_i^2 / N) / (k - 1) for unequal groups"""
    _, groups = _by_scene(scene_ids, np.asarray(scores, float))
    g = [s[np.isfinite(s)] for (s,) in groups]; g = [x for x in g if len(x)]
    k = len(g); n = np.array([len(x) for x in g]); N = int(n.sum())
    if k < 2 or N - k < 1: return dict(icc=None, k=k, N=N, reason='needs >= 2 scenes with scores and N > k')
    grand = np.concatenate(g).mean(); msb = float(sum(len(x) * (x.mean() - grand) ** 2 for x in g) / (k - 1))
    msw = float(sum(((x - x.mean()) ** 2).sum() for x in g) / (N - k)); n0 = float((N - (n ** 2).sum() / N) / (k - 1))
    den = msb + (n0 - 1) * msw
    return dict(icc=float((msb - msw) / den) if den > 0 else None, msb=msb, msw=msw, n0=n0, k=k, N=N)


def camera_slot(outcrop, slot_camera):
    c = int(slot_camera)
    if not 0 <= c < CAMERAS: raise ValueError('camera slot out of range')
    if c >= BASE_CAMERAS: return c, 0              # near slots: the plan's near cells 4, 5, on both outcrops (camera_plan_near)
    if outcrop == 'flat': return c, 0
    if outcrop == 'isolated': return ISOLATED_SLOTS[c]
    raise ValueError('outcrop must be flat or isolated')


def camera_for_slot(cameras, outcrop, slot_camera):
    """the plan camera of a slot, or None (the point is then recorded as unavailable)"""
    cell, k = camera_slot(outcrop, slot_camera)
    return next((c for c in cameras if int(c['cell']) == cell and int(c.get('k', 0)) == k), None)


def cell_label(outcrop, level):
    return '%s/%s' % (outcrop, level)


def _t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(.975, df))
    except ImportError:                                          # conservative: the next smaller integer df
        return t975(max(1, int(math.floor(df))))


def _stat_names(taus):
    return ['scorable', 'mean_scorable'] + ['joint_%g' % t for t in taus] + ['conditional_%g' % t for t in taus]


def _scene_stats(s, taus):
    """one scene's (or one scene-cell's) estimates, NaN where undefined"""
    s = np.asarray(s, float); ok = np.isfinite(s); n = len(s)
    out = [float(ok.mean()) if n else np.nan, float(s[ok].mean()) if ok.any() else np.nan]
    hits = [ok & (np.where(ok, s, -np.inf) > t) for t in taus]
    out += [float(h.mean()) if n else np.nan for h in hits]
    out += [float(h.sum() / ok.sum()) if ok.any() else np.nan for h in hits]
    return np.array(out)


def cell_estimates(scene_ids, cells, scores, taus):
    cells = np.asarray(cells, object); ids = np.asarray(list(scene_ids), object); s = np.asarray(scores, float)
    return {str(c): dict(scene_estimates(ids[cells == c], s[cells == c], taus), points=int((cells == c).sum()))
            for c in dict.fromkeys(cells.tolist())}


def equal_weight_estimate(scene_ids, cells, groups, scores, taus, weights=None):
    ids = np.asarray(list(scene_ids), object); cells = np.asarray(cells, object); groups = np.asarray(groups, object); s = np.asarray(scores, float)
    clist = list(dict.fromkeys(cells.tolist())); w = dict(weights or {c: 1. / len(clist) for c in clist})
    gof = {}
    for c in clist:
        g = set(groups[cells == c].tolist())
        if len(g) != 1: raise ValueError('cell %s spans several groups' % c)
        gof[c] = g.pop()
    stats = {}                                                   # (scene, cell) -> the scene-cell estimates
    for sid in dict.fromkeys(ids.tolist()):
        for c in clist:
            m = (ids == sid) & (cells == c)
            if m.any(): stats[sid, c] = _scene_stats(s[m], taus)
    names = _stat_names(taus); out = dict(cells=[str(c) for c in clist], weights={str(c): float(w[c]) for c in clist},
                                          groups={str(c): str(gof[c]) for c in clist}, estimands={})
    for j, name in enumerate(names):
        est = 0.; var = 0.; dfden = 0.; ok = True; per = {}
        for g in dict.fromkeys(gof[c] for c in clist):
            cg = [c for c in clist if gof[c] == g]; Wg = sum(w[c] for c in cg); z = []; dropped = 0
            for sid in dict.fromkeys(ids[groups == g].tolist()):
                vals = [stats[sid, c][j] if (sid, c) in stats else np.nan for c in cg]
                if not all(np.isfinite(vals)): dropped += 1; continue
                z.append(sum(w[c] * v for c, v in zip(cg, vals)) / Wg)
            R = len(z); per[str(g)] = dict(weight=Wg, scenes_used=R, scenes_dropped=dropped, mean=float(np.mean(z)) if R else None)
            if R < 2: ok = False; continue
            vg = Wg ** 2 * float(np.var(z, ddof=1)) / R; est += Wg * float(np.mean(z)); var += vg; dfden += vg ** 2 / (R - 1)
        rec = dict(groups=per)
        if ok:
            df = var ** 2 / dfden if dfden > 0 else float('inf'); tq = _t975(df) if math.isfinite(df) else 1.959964
            rec.update(estimate=est, se=math.sqrt(var), df=df, t=tq, ci95=[est - tq * math.sqrt(var), est + tq * math.sqrt(var)])
        else:
            rec.update(estimate=None, reason='a group has fewer than 2 scenes with this estimand')
        out['estimands'][name] = rec
    return out


def analyse_scenes(mechanism, scene_ids, units, scores, taus, physical_light=None, pooled=True, cells=None, groups=None, **pooled_kw):
    from frgb import rq1_design as rd
    s = np.asarray(scores, float); u = np.asarray(units, float)
    out = dict(mechanism=mechanism, n=int(len(s)), scene_level=scene_estimates(scene_ids, s, taus), scene_icc=scene_icc(scene_ids, s))
    if cells is not None:
        out['by_cell'] = cell_estimates(scene_ids, cells, s, taus)
        out['equal_weight'] = equal_weight_estimate(scene_ids, cells, groups, s, taus)
        out['scene_level_note'] = 'all points of a scene pooled: weighted by points per cell (isolated fine cells hold half the points)'
    if physical_light is not None:
        az, el = (np.asarray(x, float) for x in physical_light); sun = np.isfinite(az) & np.isfinite(el)
        coords = np.stack([np.mod(az[sun], 360.) / 360., np.clip((el[sun] - 15.) / 55., 0., 1. - 1e-12)], 1)
        out['physical_light'] = dict(sunny_points=int(sun.sum()), axes=['relative_azimuth_deg / 360', '(elevation_deg - 15) / 55'],
                                     relative_azimuth=rd.binned_means(coords, s[sun], (0,), n_boot=200),
                                     elevation=rd.binned_means(coords, s[sun], (1,), n_boot=200),
                                     note='bins of the recorded physical light; overcast points excluded; bootstrap over points (iid)')
    if pooled:
        out['pooled'] = rd.analyse(mechanism, u, s, taus, **pooled_kw)
        out['pooled_note'] = ('points pooled over scenes; intervals treat points as iid (scene clustering ignored), the scene-level '
                              'estimates are primary; light-gene effects are on UNIT genes only (per-camera light domains), importance '
                              'weights of light priors share that caveat')
    return _plain_tree(out)


def _plain_tree(x):
    if isinstance(x, dict): return {str(k): _plain_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_plain_tree(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.ndarray): return _plain_tree(x.tolist())
    return x


def pipeline3(items, prepare, gpu, post, cpu_workers=1, post_workers=1, segment_key=None, log=None):
    """An exception is recorded in that item's result (stage, error, traceback) and the run continues."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()

    def ev(kind, i):
        if log: log(dict(event=kind, item=i, t=time.time() - t0))

    def staged(kind, i, f, *a):
        ev(kind + '_start', i)
        try:
            return f(*a)
        finally:
            ev(kind + '_end', i)

    def fail(it, stage):
        return dict(item=it, stage=stage, error=repr(sys.exc_info()[1]), traceback=traceback.format_exc()[-3000:])
    out = [None] * len(items); posts = {}
    with ThreadPoolExecutor(max(1, int(cpu_workers))) as pre, ThreadPoolExecutor(max(1, int(post_workers))) as po:
        futures = [pre.submit(staged, 'prepare', i, prepare, it) for i, it in enumerate(items)]
        for i, (it, fu) in enumerate(zip(items, futures)):
            try:
                ctx = fu.result()
            except Exception:                                     # noqa: BLE001
                out[i] = fail(it, 'prepare'); ev('error', i); continue
            def keep(i=i, it=it, ctx=ctx):
                if segment_key is None or i + 1 >= len(items): return False
                nx = futures[i + 1]
                if not nx.done() or nx.exception() is not None: return False
                k1, k2 = segment_key(it, ctx), segment_key(items[i + 1], nx.result())
                return k1 is not None and k1 == k2
            try:
                g = staged('gpu', i, gpu, it, ctx, keep)
            except Exception:                                     # noqa: BLE001
                out[i] = fail(it, 'gpu'); ev('error', i); continue
            posts[i] = po.submit(staged, 'post', i, post, it, ctx, g)
        for i, f in posts.items():
            try:
                out[i] = dict(item=items[i], result=f.result())
            except Exception:                                     # noqa: BLE001
                out[i] = fail(items[i], 'post'); ev('error', i)
    return out


def predicted_peak_gb(rates, kind, quantity, factor=1.5, base=.5):
    """None when unknown"""
    r = (rates or {}).get(kind)
    if r is None or quantity is None: return None
    return factor * float(r['gb_per_unit']) * float(quantity) + base


def memory_decision(predicted, free_gb, outstanding_gb=0., small_max=10., margin=10., heavy_min_free=25.):
    if predicted is None or predicted > small_max:
        need = heavy_min_free; mode = 'exclusive'
    else:
        need = predicted + margin + outstanding_gb; mode = 'shared'
    return dict(mode=mode, need_free_gb=need, go=bool(free_gb >= need))


def update_rate(rates, kind, peak_gb, quantity, unit):
    """the rate table only grows: gb_per_unit = max over runs of peak / quantity"""
    if not quantity or peak_gb is None: return rates
    r = dict((rates or {}).get(kind) or {}); v = float(peak_gb) / float(quantity)
    if v > r.get('gb_per_unit', -1.):
        r.update(gb_per_unit=v, unit=unit, from_peak_gb=float(peak_gb), from_quantity=float(quantity), updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    r['runs'] = int(r.get('runs', 0)) + 1
    return dict(rates or {}, **{kind: r})


TYPE_NONE, TYPE_JOINT, TYPE_FAULT, TYPE_BOTH = 0, 1, 2, 3


def scene_structure(config, declaration=None):
    g = (config or {}).get('geometry', {}); meta = (declaration or {}).get('metadata', {}) or {}
    fc = g.get('fault_count'); out = dict(fault_count=None if fc is None else int(fc), fault_source='config.json geometry.fault_count' if fc is not None else None)
    if g.get('dyke_count') is not None:
        out.update(dyke_count=int(g['dyke_count']), dyke_source='config.json geometry.dyke_count')
    elif declaration is not None:
        out.update(dyke_count=sum(1 for x in meta.get('intrusions') or [] if str(x.get('id', '')).startswith('dyke-')),
                   dyke_source="geom/declaration.json metadata.intrusions (id 'dyke-*')")
    else:
        out.update(dyke_count=None, dyke_source=None)
    if declaration is not None:
        ev = sum(1 for e in declaration.get('events') or [] if e.get('kind') == 'fault')
        out['faults_in_declaration'] = ev
        out['fault_count_mismatch'] = bool(out['fault_count'] is not None and ev != out['fault_count'])
    return out


def _miss(mx, sel):
    if not sel.any(): return dict(loss=None, thresholded=None, pixels=0)
    return dict(loss=float((1. - mx[sel]).mean()), thresholded=float((mx[sel] < .5).mean()), pixels=int(sel.sum()))


def typed_miss_loss(prob, target, u, ttype=None, fault_count=0):
    """No type map: with no fault in the scene every positive is a joint; with faults the
    joint loss is not defined (None, reason)."""
    from scipy import ndimage
    yy, xx = np.mgrid[-2:3, -2:3]; disc = (yy ** 2 + xx ** 2) <= 4
    mx = ndimage.maximum_filter(prob, footprint=disc, mode='constant', cval=0.)
    sel = target & ~u
    if ttype is None:
        if fault_count:
            return dict(loss=None, thresholded=None, pixels=0, reason='faults in the scene and no source-type map: joints cannot be separated',
                        by_type=dict(all=_miss(mx, sel)))
        d = _miss(mx, sel); return dict(d, by_type=dict(joint=dict(d), fault=_miss(mx, np.zeros_like(sel))), type_source='no fault in the scene')
    j = _miss(mx, sel & np.isin(ttype, [TYPE_JOINT, TYPE_BOTH])); f = _miss(mx, sel & np.isin(ttype, [TYPE_FAULT, TYPE_BOTH]))   # crossings count for both
    other = int((sel & ~np.isin(ttype, [TYPE_JOINT, TYPE_FAULT, TYPE_BOTH])).sum())
    return dict(j, by_type=dict(joint=j, fault=f), positives_without_type=other, type_source='source-type map')


def visible_by_type(per_path, fault_count=0, per_type=None):
    if per_type: return {k: float(v.get('visible_px') or 0.) for k, v in per_type.items() if isinstance(v, dict)}
    rows = per_path or []
    if rows and all('source_type' in r for r in rows):
        out = {}
        for r in rows: out[str(r['source_type'])] = out.get(str(r['source_type']), 0.) + float(r.get('visible_px') or 0.)
        return out
    if not fault_count: return dict(joint=float(sum(float(r.get('visible_px') or 0.) for r in rows)), fault=0.)
    return None


def fault_summary(scene_ids, fault_counts, joint_scores, fault_losses, fault_pixels):
    ids = np.asarray(list(scene_ids), object); fc = np.asarray([np.nan if v is None else v for v in fault_counts], float)
    js = np.asarray(joint_scores, float); fl = np.asarray([np.nan if v is None else v for v in fault_losses], float)
    fp = np.asarray([0 if v is None else v for v in fault_pixels], float); has = fp > 0
    per = []
    for sid in dict.fromkeys(ids.tolist()):
        m = (ids == sid) & has & np.isfinite(fl)
        if m.any(): per.append(float(fl[m].mean()))
    out = dict(points=int(len(ids)), points_with_fault_pixels=int(has.sum()), scenes_with_fault_points=len(per),
               fault_miss_loss=dict(replicate_estimate(per), per_scene=per) if per else None, by_scene_fault_count={})
    for name, m in (('0', fc == 0), ('>=1', fc >= 1), ('unknown', ~np.isfinite(fc))):
        if m.any(): out['by_scene_fault_count'][name] = dict(points=int(m.sum()), mean_scorable=scene_estimates(ids[m], js[m], [])['mean_scorable'])
    out['note'] = 'descriptive: the shape difficulty is defined on the joints; faults are other positives in the frame'
    return _plain_tree(out)


NOISE_SEED_BASE = 20261030
NOISE_REPEATS = (1, 2)


def noise_points(mechanism, scene_index, seed_base=NOISE_SEED_BASE):
    rng = np.random.default_rng([int(seed_base) + int(MECHANISMS[mechanism]), int(scene_index)])
    sunny = [i for i in range(POINTS_PER_SCENE) if slot_of(i)[2] == 'sun']; overcast = [i for i in range(POINTS_PER_SCENE) if slot_of(i)[2] == 'overcast']
    s = int(sunny[rng.integers(len(sunny))]); o = int(overcast[rng.integers(len(overcast))])
    return dict(sunny=s, overcast=o, items=[(s, r) for r in NOISE_REPEATS] + [(o, r) for r in NOISE_REPEATS],
                rule='uniform draw per scene, rng [%d + mechanism number, scene index]' % seed_base)


def envelope_fit(q, y):
    from scipy.optimize import linprog
    q, y = np.asarray(q, float), np.asarray(y, float)
    if len(q) < 2: raise ValueError('at least two points')
    res = linprog(c=[len(q), float(q.sum())], A_ub=np.stack([-np.ones_like(q), -q], 1), b_ub=-y, bounds=[(0, None), (0, None)], method='highs')
    if not res.success: raise ValueError('envelope fit failed: ' + res.message)
    a, b = (float(v) for v in res.x)
    return dict(a=a, b=b, n=int(len(q)), max_slack=float((a + b * q - y).max()), min_slack=float((a + b * q - y).min()),
                rule='upper envelope (LP), predict = 1.1 x (a + b q) + 0.5 GB')


def envelope_predict(model, q, factor=1.1, base=.5):
    if model is None or q is None: return None
    return factor * (float(model['a']) + float(model['b']) * float(q)) + base
