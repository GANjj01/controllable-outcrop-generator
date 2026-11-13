"""Everything works on the 6 independent unit genes of difficulty_params (4 mechanism genes, relative azimuth,
elevation), whose primary prior (D3) is the uniform distribution on [0,1]^6: - design: Owen (nested uniform) scrambled
Sobol points -> physical / analysis coordinates, frozen as a JSON manifest with a sha256 over its canonical content;
the first 32 points (a scrambled 2^5 net) get repeated renders with other seeds; - priors: per-gene linear /
log-uniform specs; other priors by exact importance weights (unnormalized, unbiased) + Kish ESS; - estimation: score
distribution, hard fractions P(s > tau) (joint with scorability and conditional on it) with bootstrap, Wilson and
Clopper-Pearson intervals; unscorable points are NaN and never dropped silently; a surrogate (ARD Gaussian process
with cos/sin azimuth features vs. Legendre chaos, chosen by K-fold Q2) gives 1D/2D partial dependence and first/total
Sobol indices on the unit genes; render noise from the repeats is deducted from the variance shares; - rare hard
examples: one cross-entropy round (product-histogram proposal + defensive prior component) with an unbiased
importance-sampling estimate; the rule is written down here and not run. Gene 1 of shape: under the mapping of
difficulty_params it is linear in depth (10-25 cm) and independent of the opening, so its effects are depth effects at
fixed opening; the derived aspect ratio depth / w depends on w."""
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from frgb import difficulty_params as dp
from frgb.cd_search import MECHANISMS

DESIGN_SCHEMA = 'rq1-design-r1'
STATUS = 'design draft; not a formal sample'
DIMENSION = 6
N_DESIGN = 512
N_NOISE = 32
NOISE_REPEATS = 2
DEFAULT_SEED = 20261028
THRESHOLD_QUANTILES = (.5, .9, .99)
AZIMUTH, ELEVATION = 4, 5
LIGHT_GENES = ('relative_azimuth_deg', 'elevation_deg')
GENES = {'shape': dp.SHAPE_GENES + LIGHT_GENES, 'stain': dp.STAIN_GENES + LIGHT_GENES}
# the physical axis each unit gene is linear (or log-linear) in; shape gene 1 is the depth at fixed opening
AXES = {'shape': ('mean_opening_m', 'depth_m', 'opening_variation', 'relative_roughness') + LIGHT_GENES,
        'stain': dp.STAIN_GENES + LIGHT_GENES}
STREAMS = dict(scramble=0, render=1, ce_sample=2, ce_render=3)
SOBOL_MC_STREAM = 97
Q2_RELATIVE_GATE = .7
SCORABLE_GATE = .8         # below this overall scorable fraction, landscape effects are labelled exploratory
CE_RULE = dict(
    trigger_fraction=.05, trigger_on='point',  # joint P(scorable and s > tau) point estimate below 5%, each tau judged
    target='smallest_triggered_tau', rho=.1, bins=8, pseudo_count=1., defensive_fraction=.25, n_stage2=256,
    stage1_weight=.2, stage2_sampler='iid numpy default_rng, SeedSequence([seed, mechanism, 2])',
    status='design choice')
PRIORS = dict(
    primary='difficulty_params mapping: uniform in each analysis coordinate, width and wavelength log-uniform (D3)',
    all_linear='every mechanism gene linear-uniform on the same range (for shape identical to primary: a null check)',
    log_positive='every mechanism gene with a positive lower bound log-uniform')


# ---------------------------------------------------------------- design

def owen_sobol(n, d, seed):
    """n = 2^m points of the d-dimensional Sobol sequence (scipy's Joe-Kuo direction numbers, unscrambled, first point
    kept) with Owen's nested uniform scrambling. For the first 2^m points each coordinate has distinct top-m bits and
    zero lower bits, so random flips per input prefix on levels 1..m (drawn level by level in prefix order), followed by
    independent uniform lower digits per point, equal the full nested scramble."""
    from scipy.stats import qmc
    m = int(round(math.log2(n))) if n >= 1 else -1
    if m < 0 or 2 ** m != n:
        raise ValueError('n must be a power of two')
    x = qmc.Sobol(d, scramble=False).random_base2(m)
    t = x * 2. ** m
    if not np.array_equal(t, np.floor(t)):
        raise RuntimeError('unscrambled Sobol points have nonzero bits below level m')
    t = t.astype(np.int64)
    rng = np.random.default_rng(seed)
    out = np.empty((n, d))
    for j in range(d):
        flip = np.zeros(n, np.int64)
        for k in range(m):
            flip |= rng.integers(0, 2, size=2 ** k)[t[:, j] >> (m - k)] << (m - 1 - k)
        out[:, j] = ((t[:, j] ^ flip) + rng.random(n)) / 2. ** m
    return out


def seed_sequence(seed, mechanism, stream, *extra):
    return np.random.SeedSequence([int(seed), MECHANISMS[mechanism], STREAMS[stream]] + [int(e) for e in extra])


def design_units(mechanism, n=N_DESIGN, seed=DEFAULT_SEED):
    return owen_sobol(n, DIMENSION, seed_sequence(seed, mechanism, 'scramble'))


def render_seed(seed, mechanism, index, repeat=0, stream='render'):
    state = seed_sequence(seed, mechanism, stream, index, repeat).generate_state(1)[0]
    return int(state % (2 ** 31 - 2)) + 1


def noise_subset(n_noise=N_NOISE):
    """fixed rule: the first n_noise points in Sobol order (a scrambled 2^5 net for 32), not sorted or selected"""
    return list(range(int(n_noise)))


def kept_vec(u, bounds, excluded=()):
    """vectorized difficulty_params.kept (same operation order, so bit-identical)"""
    v = bounds[0] + (bounds[1] - bounds[0] - sum(hi - lo for lo, hi in excluded)) * np.asarray(u, float)
    for lo, hi in sorted(excluded):
        v = np.where(v > lo, v + (hi - lo), v)
    return v


def _forward(mechanism):
    if mechanism == 'shape':
        return dp.shape_forward
    if mechanism == 'stain':
        return dp.stain_forward
    raise ValueError('mechanism must be shape or stain')


def map_point(mechanism, unit, excluded_rel_az=(), excluded_el=()):
    """unit genes -> builder physical parameters and analysis coordinates; the light stays relative to the rock-face facing
    azimuth (the compass azimuth needs the face normal: dp.light_forward at render time)"""
    u = np.asarray(unit, float)
    if u.shape != (DIMENSION,):
        raise ValueError('6 unit genes required')
    f = _forward(mechanism)(u[:4])
    light = dp.light_forward(u[4:], 0., excluded_rel_az, excluded_el)
    return dict(physical=f['physical'], analysis=dict(f['analysis'], relative_azimuth_deg=light['relative_azimuth_deg'],
                                                      elevation_deg=light['elevation_deg']))


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


def canonical_sha256(manifest):
    """sha256 of the canonical JSON (sorted keys, fixed separators, plain floats) without 'sha256' and 'environment'"""
    body = {k: v for k, v in manifest.items() if k not in ('sha256', 'environment')}
    text = json.dumps(_plain(body), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def environment():
    import scipy
    files = dict(rq1_design=Path(__file__), difficulty_params=Path(dp.__file__))
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                source_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()})


def build_manifest(mechanism, seed=DEFAULT_SEED, n=N_DESIGN, n_noise=N_NOISE, repeats=NOISE_REPEATS,
                   excluded_rel_az=(), excluded_el=(), lighting_rule=None):
    """freezable design manifest for one mechanism; the light exclusions enter the hash"""
    if not 0 < n_noise <= n:
        raise ValueError('noise subset must be a nonempty part of the design')
    code = MECHANISMS[mechanism]
    units = design_units(mechanism, n, seed)
    points = [dict(index=i, unit=u, render_seed=render_seed(seed, mechanism, i), **map_point(mechanism, u, excluded_rel_az, excluded_el))
              for i, u in enumerate(units)]
    noise = noise_subset(n_noise)
    rep = [dict(index=i, repeat=r, render_seed=render_seed(seed, mechanism, i, r)) for i in noise for r in range(1, repeats + 1)]
    m = dict(schema=DESIGN_SCHEMA, status=STATUS, mechanism=mechanism, genes=GENES[mechanism], axes=AXES[mechanism], n=n,
             prior=dict(name='primary', description=PRIORS['primary']),
             seed=dict(base=int(seed), scramble_entropy=[int(seed), code, STREAMS['scramble']],
                       render='SeedSequence([base, %d, %d, index, repeat]).generate_state(1)[0] %% (2^31 - 2) + 1' % (code, STREAMS['render'])),
             sampler=dict(kind='Sobol (scipy.stats.qmc, unscrambled Joe-Kuo direction numbers, no point skipped) + Owen nested '
                               'uniform scrambling (rq1_design.owen_sobol)', dimension=DIMENSION),
             parametrization=dp.PARAMETRIZATION_R2,
             light=dict(reference='relative to the rock-face facing azimuth (difficulty_params.light_forward)',
                        excluded_relative_azimuth_deg=[list(e) for e in excluded_rel_az],
                        excluded_elevation_deg=[list(e) for e in excluded_el],
                        rule=lighting_rule or 'NOT FROZEN: exclusions pending rule r4; regenerate after r4'),
             noise=dict(rule='first %d points in Sobol order' % n_noise, indices=noise, repeats_per_point=int(repeats),
                        seeds='different render seeds per repeat; what the seed drives is fixed by the builder (pipeline r11)'),
             points=points, repeats=rep)
    m = _plain(m)
    m['environment'] = environment()
    m['sha256'] = canonical_sha256(m)
    return m


def write_manifest(manifest, path):
    if manifest.get('sha256') != canonical_sha256(manifest):
        raise ValueError('manifest sha256 does not match its content')
    text = json.dumps(_plain(manifest), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + '\n'
    Path(path).write_text(text, encoding='utf-8', newline='\n')


def load_manifest(path):
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    if m.get('sha256') != canonical_sha256(m):
        raise ValueError('manifest sha256 mismatch: content changed after freezing')
    return m


def verify_manifest(manifest):
    """rebuild from the recorded seed, sizes and exclusions; fail closed if anything differs"""
    light = manifest['light']
    rule = light['rule'] if not str(light['rule']).startswith('NOT FROZEN') else None
    again = build_manifest(manifest['mechanism'], manifest['seed']['base'], manifest['n'], len(manifest['noise']['indices']),
                           manifest['noise']['repeats_per_point'], [tuple(e) for e in light['excluded_relative_azimuth_deg']],
                           [tuple(e) for e in light['excluded_elevation_deg']], rule)
    if again['sha256'] != manifest['sha256'] or canonical_sha256(manifest) != manifest['sha256']:
        raise ValueError('manifest does not reproduce from its recorded seed and parameters')
    return True


def manifest_units(manifest):
    return np.array([p['unit'] for p in manifest['points']], float)


# ---------------------------------------------------------------- priors and importance weights

def gene_specs(mechanism):
    """primary marginal of each unit gene as (kind, lo, hi) of the coordinate it maps to linearly / log-linearly; the
    aspect gene's range [0.10/w, 0.25/w] depends on w, but the prior transforms only need hi/lo = 2.5"""
    b = dp.PARAMETRIZATION_R2
    if mechanism == 'shape':
        s, d = b['shape'], b['shape']['depth_m']
        specs = [('linear',) + tuple(s['mean_opening_m']), ('linear', 1., d[1] / d[0]),
                 ('linear',) + tuple(s['opening_variation']), ('linear',) + tuple(s['relative_roughness'])]
    elif mechanism == 'stain':
        specs = [('log' if k in b['log_uniform'] else 'linear',) + tuple(b['stain'][k]) for k in dp.STAIN_GENES]
    else:
        raise ValueError('mechanism must be shape or stain')
    return specs + [('light', 0., 1.), ('light', 0., 1.)]


def prior_kinds(mechanism, prior):
    specs = gene_specs(mechanism)
    if isinstance(prior, (list, tuple)):
        kinds = list(prior)
    elif prior == 'primary':
        kinds = [s[0] for s in specs]
    elif prior == 'all_linear':
        kinds = ['light' if s[0] == 'light' else 'linear' for s in specs]
    elif prior == 'log_positive':
        kinds = ['light' if s[0] == 'light' else ('log' if s[1] > 0 else 'linear') for s in specs]
    else:
        raise ValueError('unknown prior %r' % (prior,))
    for k, s in zip(kinds, specs):
        if (k == 'light') != (s[0] == 'light') or k not in ('linear', 'log', 'light') or (k == 'log' and s[1] <= 0):
            raise ValueError('prior kind %r not available for this gene' % k)
    return kinds


def _gene_ratio(u, spec, kind):
    """density of the unit gene under `kind`, relative to its primary (uniform) density"""
    base, lo, hi = spec
    if kind == base:
        return np.ones_like(u)
    r = hi / lo
    if base == 'log' and kind == 'linear':
        return lo * r ** u * math.log(r) / (hi - lo)
    if base == 'linear' and kind == 'log':
        return (hi - lo) / ((lo + (hi - lo) * u) * math.log(r))
    raise ValueError('unsupported prior change %s -> %s' % (base, kind))


def _gene_inverse(v, spec, kind):
    """uniform v -> unit gene distributed as `kind`"""
    base, lo, hi = spec
    if kind == base:
        return v
    r = hi / lo
    if base == 'log' and kind == 'linear':
        return np.log((lo + (hi - lo) * v) / lo) / math.log(r)
    if base == 'linear' and kind == 'log':
        return (lo * r ** v - lo) / (hi - lo)
    raise ValueError('unsupported prior change %s -> %s' % (base, kind))


def prior_density(units, mechanism, prior):
    """density of the prior on the unit cube (the primary prior has density 1); mechanism None = uniform only"""
    u = np.atleast_2d(np.asarray(units, float))
    if mechanism is None:
        if prior != 'primary':
            raise ValueError('generic genes only have the uniform prior')
        return np.ones(len(u))
    out = np.ones(len(u))
    for j, (spec, kind) in enumerate(zip(gene_specs(mechanism), prior_kinds(mechanism, prior))):
        out = out * _gene_ratio(u[:, j], spec, kind)
    return out


def sample_prior(v, mechanism, prior='primary'):
    """uniform points v on the unit cube -> unit genes distributed as `prior` (column-wise inverse CDF)"""
    v = np.asarray(v, float)
    if mechanism is None or prior == 'primary':
        if prior != 'primary':
            raise ValueError('generic genes only have the uniform prior')
        return v.copy()
    return np.column_stack([_gene_inverse(v[:, j], s, k) for j, (s, k) in
                            enumerate(zip(gene_specs(mechanism), prior_kinds(mechanism, prior)))])


def importance_weights(units, mechanism, target, source='primary'):
    """exact density ratio target / source; use with unnormalized means (unbiased), self-normalized only as auxiliary"""
    return prior_density(units, mechanism, target) / prior_density(units, mechanism, source)


def kish_ess(weights):
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


# ---------------------------------------------------------------- distribution and hard fractions

def _weights(weights, n):
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('weights must be finite, nonnegative, one per sample')
    return w


def weighted_quantile(x, q, w=None):
    """inverse of the weighted empirical CDF: smallest x with F(x) >= q (unweighted: Hyndman-Fan type 1)"""
    x = np.asarray(x, float)
    w = _weights(w, len(x))
    o = np.argsort(x, kind='stable')
    c = np.cumsum(w[o])
    k = np.searchsorted(c, np.asarray(q, float) * c[-1], side='left')
    return x[o][np.minimum(k, len(x) - 1)]


def thresholds_from_reference(reference_losses, quantiles=THRESHOLD_QUANTILES):
    r = np.asarray(reference_losses, float).ravel()
    if r.size < 2 or not np.isfinite(r).all():
        raise ValueError('reference losses must be finite (no unscorable entries) and at least two')
    return dict(quantiles=list(quantiles), taus=[float(np.quantile(r, q)) for q in quantiles], n_reference=int(r.size),
                method='numpy.quantile linear (type 7)')


def aggregate_models(score_table):
    """primary score = mean over the fixed models; scorability is geometric, so the NaN pattern must agree"""
    arrays = [np.asarray(v, float) for v in score_table.values()]
    nan = np.isnan(arrays[0])
    if any(a.shape != arrays[0].shape or not np.array_equal(np.isnan(a), nan) for a in arrays):
        raise ValueError('models disagree on which samples are scorable')
    return np.mean(arrays, axis=0)


def _z(level):
    from scipy.stats import norm
    return float(norm.ppf(.5 + level / 2))


def wilson(k, n, level=.95):
    if n == 0:
        return [float('nan'), float('nan')]
    z = _z(level)
    c = (k + z * z / 2) / (n + z * z)
    h = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [max(0., c - h), min(1., c + h)]


def clopper_pearson(k, n, level=.95):
    from scipy.stats import beta
    if n == 0:
        return [float('nan'), float('nan')]
    a = 1 - level
    return [float(beta.ppf(a / 2, k, n - k + 1)) if k > 0 else 0., float(beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.]


def _percentile_ci(samples, level, axis=0):
    a = (1 - level) / 2
    return np.nanquantile(samples, [a, 1 - a], axis=axis)


def score_distribution(scores, weights=None, quantiles=(.01, .05, .1, .25, .5, .75, .9, .95, .99), n_boot=2000, seed=0,
                       level=.95, bins=24):
    """distribution of the scorable scores (self-normalized weights), bootstrap CIs resampling all points"""
    s = np.asarray(scores, float)
    n = len(s)
    w = _weights(weights, n)
    ok = np.isfinite(s)
    if not ok.any():
        raise ValueError('no scorable sample')

    def stats(si, wi):
        m = np.isfinite(si)
        if not m.any() or wi[m].sum() <= 0:
            return np.full(2 + len(quantiles), np.nan)
        x, ww = si[m], wi[m]
        mu = np.sum(ww * x) / ww.sum()
        return np.concatenate([[mu, math.sqrt(np.sum(ww * (x - mu) ** 2) / ww.sum())], weighted_quantile(x, quantiles, ww)])

    est = stats(s, w)
    rng = np.random.default_rng(seed)
    boot = np.array([stats(s[i], w[i]) for i in rng.integers(0, n, size=(n_boot, n))]) if n_boot else None
    ci = _percentile_ci(boot, level) if n_boot else np.full((2, len(est)), np.nan)
    hist, edges = np.histogram(s[ok], bins=bins, weights=w[ok], density=True)
    return dict(n=n, n_scorable=int(ok.sum()), mean=float(est[0]), mean_ci=ci[:, 0].tolist(), sd=float(est[1]),
                quantiles={str(q): dict(estimate=float(est[2 + i]), ci=ci[:, 2 + i].tolist()) for i, q in enumerate(quantiles)},
                histogram=dict(edges=edges.tolist(), density=hist.tolist()),
                note='self-normalized over scorable samples; bootstrap resamples all points (iid, conservative for scrambled Sobol)')


def hard_fractions(scores, taus, weights=None, n_boot=2000, seed=0, level=.95):
    """per tau: joint P(scorable and s > tau) (unnormalized importance mean: unbiased) and conditional P(s > tau |
    scorable) (ratio), plus the scorable fraction; percentile bootstrap, and Wilson / Clopper-Pearson when unweighted"""
    s = np.asarray(scores, float)
    n = len(s)
    w = _weights(weights, n)
    ok = np.isfinite(s)
    taus = [float(t) for t in taus]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n)) if n_boot else None
    unweighted = weights is None

    def interval(values, denom=None):
        """bootstrap CI of mean(values) or of sum(values) / sum(denom)"""
        if not n_boot:
            return [float('nan')] * 2
        b = values[idx].mean(1) if denom is None else values[idx].sum(1) / np.where(denom[idx].sum(1) > 0, denom[idx].sum(1), np.nan)
        return _percentile_ci(b, level).tolist()

    wok = w * ok
    out = dict(n=n, n_scorable=int(ok.sum()), weighted=not unweighted,
               scorable=dict(estimate=float(wok.mean()), ci=interval(wok), count=int(ok.sum())), thresholds=[])
    if unweighted:
        out['scorable'].update(wilson=wilson(int(ok.sum()), n, level), clopper_pearson=clopper_pearson(int(ok.sum()), n, level))
    else:
        out['ess'] = kish_ess(w)
    for t in taus:
        hit = ok & (np.where(ok, s, -np.inf) > t)
        wh = w * hit
        k = int(hit.sum())
        joint = dict(estimate=float(wh.mean()), ci=interval(wh))
        cond = dict(estimate=float(wh.sum() / wok.sum()) if wok.sum() > 0 else float('nan'), ci=interval(wh, wok))
        if unweighted:
            joint.update(wilson=wilson(k, n, level), clopper_pearson=clopper_pearson(k, n, level))
            cond.update(wilson=wilson(k, int(ok.sum()), level), clopper_pearson=clopper_pearson(k, int(ok.sum()), level))
        out['thresholds'].append(dict(tau=t, hits=k, joint=joint, conditional=cond))
    return out


# ---------------------------------------------------------------- surrogates (fit on scorable points only)

def kfold_ids(n, k, seed):
    ids = np.empty(n, int)
    ids[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return ids


def _chunked(f, x, size=4096):
    return np.concatenate([f(x[i:i + size]) for i in range(0, len(x), size)], axis=0)


def _legendre(u, pmax):
    """orthonormal Legendre polynomials on U(0,1): array (pmax + 1, *u.shape)"""
    x = 2 * np.asarray(u, float) - 1
    p = [np.ones_like(x), x]
    for k in range(1, pmax):
        p.append(((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1))
    return np.stack([p[k] * math.sqrt(2 * k + 1) for k in range(pmax + 1)])


def multi_indices(d, p, q=.75):
    """nonzero multi-indices of total degree <= p inside the hyperbolic q-norm ball of radius p"""
    out = []

    def rec(prefix, left):
        if len(prefix) == d:
            a = np.array(prefix, float)
            if a.sum() > 0 and np.sum(a ** q) ** (1 / q) <= p + 1e-9:
                out.append(prefix)
            return
        for k in range(left + 1):
            rec(prefix + (k,), left - k)
    rec((), p)
    return sorted(out, key=lambda a: (sum(a), a[::-1]))


class PCESurrogate:
    """Legendre polynomial chaos on the unit genes (orthonormal under the primary prior), ridge with unpenalized mean;
    degree and ridge chosen by inner K-fold CV. Analytic Sobol indices from the coefficients (primary prior)."""
    name = 'pce_legendre'

    def __init__(self, degrees=tuple(range(1, 9)), q=.75, ridges=tuple(10. ** np.arange(-8, 2)), max_terms=.5, folds=5, seed=0):
        self.degrees, self.q, self.ridges, self.max_terms, self.folds, self.seed = degrees, q, ridges, max_terms, folds, seed

    def _phi(self, units, index):
        u = np.asarray(units, float)
        pmax = max(max(a) for a in index)
        leg = _legendre(u, pmax)
        cols = []
        for a in index:
            c = np.ones(len(u))
            for j, k in enumerate(a):
                if k:
                    c = c * leg[k, :, j]
            cols.append(c)
        return np.column_stack(cols)

    def basis(self, units):
        return self._phi(units, self.index)

    @staticmethod
    def _ridge(phi, y, lams):
        mp, my = phi.mean(0), y.mean()
        U, S, Vt = np.linalg.svd(phi - mp, full_matrices=False)
        uy = U.T @ (y - my)
        out = []
        for lam in lams:
            b = Vt.T @ (S / (S ** 2 + lam * len(y)) * uy)
            out.append((my - mp @ b, b))
        return out

    def fit(self, units, y):
        u, y = np.asarray(units, float), np.asarray(y, float)
        d = u.shape[1]
        ids = kfold_ids(len(y), self.folds, self.seed)
        best = None
        for p in self.degrees:
            index = multi_indices(d, p, self.q)
            if len(index) > self.max_terms * len(y) * (1 - 1 / self.folds):
                break
            phi = self._phi(u, index)
            sse = np.zeros(len(self.ridges))
            for f in range(self.folds):
                tr, te = ids != f, ids == f
                for i, (c0, b) in enumerate(self._ridge(phi[tr], y[tr], self.ridges)):
                    sse[i] += np.sum((y[te] - c0 - phi[te] @ b) ** 2)
            i = int(np.argmin(sse))
            if best is None or sse[i] < best[0]:
                best = (sse[i], p, self.ridges[i], index)
        _, self.degree, self.ridge, self.index = best
        self.units, self.y = u, y
        self.intercept, self.coef = self._ridge(self._phi(u, self.index), y, [self.ridge])[0]
        return self

    def refit_idx(self, idx):
        phi = self._phi(self.units[idx], self.index)
        return self._ridge(phi, self.y[idx], [self.ridge])[0]

    def predict(self, units):
        return self.intercept + _chunked(self.basis, np.asarray(units, float)) @ self.coef

    def sobol_analytic(self):
        a = np.array(self.index) > 0
        c2 = self.coef ** 2
        v = c2.sum()
        only = a.sum(1) == 1
        return dict(first=[float(c2[only & a[:, j]].sum() / v) for j in range(a.shape[1])],
                    total=[float(c2[a[:, j]].sum() / v) for j in range(a.shape[1])], variance=float(v))


class GPSurrogate:
    """Gaussian-process / kernel-ridge surrogate with an ARD squared-exponential kernel, one length scale per gene.
    Features: unit genes, except the light genes, which use their physical kept values (azimuth as cos/sin of the
    relative azimuth, elevation scaled to [0,1] over 15-70 deg), so exclusion gaps are not glued together.
    Hyperparameters by maximum marginal likelihood (L-BFGS-B, analytic gradient)."""
    name = 'gp_ard'
    BOUNDS = dict(ell=(.02, 50.), signal=(1e-2, 1e2), noise=(1e-6, 2.))
    JITTER = 1e-10

    def __init__(self, excluded_rel_az=(), excluded_el=(), light=True, restarts=2, seed=0, theta0=None):
        self.excluded_rel_az, self.excluded_el, self.light = tuple(excluded_rel_az), tuple(excluded_el), light
        self.restarts, self.seed, self.theta0 = restarts, seed, theta0

    def features(self, units):
        u = np.asarray(units, float)
        cols, groups = [], []
        for j in range(u.shape[1]):
            if self.light and j == AZIMUTH:
                th = np.radians(kept_vec(u[:, j], dp.PARAMETRIZATION_R2['light']['relative_azimuth_deg'], self.excluded_rel_az))
                groups.append([len(cols), len(cols) + 1])
                cols += [np.cos(th) / 2, np.sin(th) / 2]
            elif self.light and j == ELEVATION:
                b = dp.PARAMETRIZATION_R2['light']['elevation_deg']
                groups.append([len(cols)])
                cols.append((kept_vec(u[:, j], b, self.excluded_el) - b[0]) / (b[1] - b[0]))
            else:
                groups.append([len(cols)])
                cols.append(u[:, j])
        return np.column_stack(cols), groups

    @staticmethod
    def _sqdist(x1, x2, groups):
        return np.stack([sum((x1[:, c, None] - x2[None, :, c]) ** 2 for c in g) for g in groups])

    def _nll(self, theta, D, ys):
        from scipy.linalg import cho_factor, cho_solve
        G, n = D.shape[0], len(ys)
        ell2 = np.exp(2 * theta[:G])
        s2, n2 = math.exp(theta[G]), math.exp(theta[G + 1])
        kf = s2 * np.exp(-.5 * np.tensordot(1 / ell2, D, axes=1))
        try:
            c = cho_factor(kf + (n2 + self.JITTER) * np.eye(n), lower=True)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros_like(theta)
        alpha = cho_solve(c, ys)
        nll = .5 * ys @ alpha + np.log(np.diag(c[0])).sum() + .5 * n * math.log(2 * math.pi)
        W = cho_solve(c, np.eye(n)) - np.outer(alpha, alpha)
        grad = [.5 * np.sum(W * kf * D[g] / ell2[g]) for g in range(G)] + [.5 * np.sum(W * kf), .5 * n2 * np.trace(W)]
        return nll, np.array(grad)

    def _theta_bounds(self, G):
        b = self.BOUNDS
        return [tuple(np.log(b['ell']))] * G + [tuple(np.log(b['signal'])), tuple(np.log(b['noise']))]

    def fit(self, units, y):
        from scipy.optimize import minimize
        self.units, self.y = np.asarray(units, float), np.asarray(y, float)
        self.x, self.groups = self.features(self.units)
        G = len(self.groups)
        D = self._sqdist(self.x, self.x, self.groups)
        self.mu, self.sd = float(self.y.mean()), float(self.y.std()) or 1.
        ys = (self.y - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        starts = [np.array(self.theta0, float)] if self.theta0 is not None else [np.r_[np.full(G, math.log(.5)), 0., math.log(.1)]]
        starts += [np.r_[rng.uniform(math.log(.1), math.log(2.), G), 0., rng.uniform(math.log(1e-3), math.log(.5))]
                   for _ in range(self.restarts)]
        bounds = self._theta_bounds(G)
        best = None
        for t0 in starts:
            t0 = np.clip(t0, [b[0] for b in bounds], [b[1] for b in bounds])
            r = minimize(self._nll, t0, args=(D, ys), jac=True, method='L-BFGS-B', bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        self.kf = math.exp(self.theta[G]) * np.exp(-.5 * np.tensordot(np.exp(-2 * self.theta[:G]), D, axes=1))
        self.intercept, self.coef = self.mu, self._alpha(np.arange(len(self.y)))
        return self

    def _alpha(self, idx):
        from scipy.linalg import cho_factor, cho_solve
        G = len(self.groups)
        k = self.kf[np.ix_(idx, idx)] + (math.exp(self.theta[G + 1]) + self.JITTER) * np.eye(len(idx))
        a = cho_solve(cho_factor(k, lower=True), (self.y[idx] - self.mu) / self.sd)
        return self.sd * np.bincount(idx, weights=a, minlength=len(self.y))

    def refit_idx(self, idx):
        """bootstrap refit with fixed hyperparameters; duplicated points fold back onto the full basis"""
        return self.mu, self._alpha(np.asarray(idx))

    def basis(self, units):
        G = len(self.groups)
        x, _ = self.features(units)
        return math.exp(self.theta[G]) * np.exp(-.5 * np.tensordot(np.exp(-2 * self.theta[:G]), self._sqdist(x, self.x, self.groups), axes=1))

    def predict(self, units):
        return self.intercept + _chunked(self.basis, np.asarray(units, float), 2048) @ self.coef

    @property
    def noise_variance(self):
        return math.exp(self.theta[len(self.groups) + 1]) * self.sd ** 2

    @property
    def length_scales(self):
        return np.exp(self.theta[:len(self.groups)]).tolist()


def cross_validate(make, units, y, folds=8, seed=0):
    u, y = np.asarray(units, float), np.asarray(y, float)
    ids = kfold_ids(len(y), folds, seed)
    pred = np.empty(len(y))
    for f in range(folds):
        tr, te = ids != f, ids == f
        pred[te] = make().fit(u[tr], y[tr]).predict(u[te])
    sse = float(np.sum((y - pred) ** 2))
    return dict(q2=1 - sse / float(np.sum((y - y.mean()) ** 2)), rmse=math.sqrt(sse / len(y)), folds=folds, seed=seed)


def fit_surrogates(units, scores, excluded_rel_az=(), excluded_el=(), folds=8, seed=0, light=True, candidates=('gp', 'pce')):
    """fit each candidate on the scorable points and choose the higher K-fold Q2. Every fold repeats the full procedure on
    its training part only (GP: the same data-independent starts and restarts as the full fit, never the full-data
    optimum; PCE: the inner degree / ridge selection), so no fold sees its test points."""
    s = np.asarray(scores, float)
    ok = np.isfinite(s)
    u, y = np.asarray(units, float)[ok], s[ok]
    models, cv = {}, {}
    if 'gp' in candidates:
        gp = GPSurrogate(excluded_rel_az, excluded_el, light, seed=seed).fit(u, y)
        models['gp'] = gp
        cv['gp'] = cross_validate(lambda: GPSurrogate(excluded_rel_az, excluded_el, light, seed=seed), u, y, folds, seed)
    if 'pce' in candidates:
        models['pce'] = PCESurrogate(seed=seed).fit(u, y)
        cv['pce'] = cross_validate(lambda: PCESurrogate(seed=seed), u, y, folds, seed)
    chosen = max(cv, key=lambda k: (cv[k]['q2'], k == 'gp'))
    return dict(chosen=chosen, model=models[chosen], models=models, cv=cv, n_fit=int(ok.sum()), units=u, y=y,
                rule='higher %d-fold CV Q2 on scorable points (fold seed %d); tie -> gp' % (folds, seed))


# ---------------------------------------------------------------- Sobol indices and partial dependence

def _evaluator(model, coefs=None, intercepts=None):
    """predictions as (m, K) for K coefficient vectors (and intercepts; default the model's own) on the model basis,
    or for a plain callable (K = 1)"""
    if hasattr(model, 'basis'):
        c = np.asarray(model.coef, float)[:, None] if coefs is None else np.asarray(coefs, float)
        b = np.broadcast_to(np.asarray(model.intercept if intercepts is None else intercepts, float), (c.shape[1],))
        return lambda x: b + _chunked(model.basis, x, 2048) @ c
    return lambda x: np.asarray(model(x), float).reshape(len(x), 1)


def sobol_indices(model, mechanism=None, prior='primary', n=4096, seed=0, coefs=None, d=DIMENSION):
    """first-order (Saltelli 2010) and total (Jansen) indices on the unit genes by pick-freeze over the prior;
    A and B from one Owen-scrambled Sobol set in 2d dimensions; the same matrices for every coefficient vector"""
    v = owen_sobol(n, 2 * d, np.random.SeedSequence([int(seed), SOBOL_MC_STREAM]))
    a, b = sample_prior(v[:, :d], mechanism, prior), sample_prior(v[:, d:], mechanism, prior)
    f = _evaluator(model, coefs)
    fa, fb = f(a), f(b)
    f0 = np.mean(np.vstack([fa, fb]), axis=0)  # centring makes the Saltelli estimator shift-invariant in finite samples
    fa, fb = fa - f0, fb - f0
    var = np.var(np.vstack([fa, fb]), axis=0)
    v1, vt = [], []
    for i in range(d):
        ab = a.copy()
        ab[:, i] = b[:, i]
        fab = f(ab) - f0
        v1.append(np.mean(fb * (fab - fa), axis=0))
        vt.append(.5 * np.mean((fa - fab) ** 2, axis=0))
    v1, vt = np.array(v1), np.array(vt)
    return dict(first=v1 / var, total=vt / var, variance=var, v_first=v1, v_total=vt, n=n, prior=prior)


def sobol_bootstrap(fit, mechanism, prior='primary', n_boot=200, seed=0, n=4096, level=.95):
    """indices of the chosen surrogate + percentile CIs from refits on bootstrap resamples (hyperparameters fixed)"""
    model, m = fit['model'], len(fit['y'])
    rng = np.random.default_rng([int(seed), 11])
    coefs = [model.coef] + [model.refit_idx(rng.integers(0, m, m))[1] for _ in range(n_boot)]
    r = sobol_indices(model, mechanism, prior, n, seed, np.column_stack(coefs))
    out = dict(prior=prior, n_mc=n, n_boot=n_boot, surrogate=fit['chosen'],
               first=r['first'][:, 0].tolist(), total=r['total'][:, 0].tolist(), variance=float(r['variance'][0]),
               v_first=r['v_first'][:, 0].tolist(), v_total=r['v_total'][:, 0].tolist())
    if n_boot:
        out.update(first_ci=_percentile_ci(r['first'][:, 1:], level, axis=1).T.tolist(),
                   total_ci=_percentile_ci(r['total'][:, 1:], level, axis=1).T.tolist())
    return out


def gene_axis(mechanism, j, u, excluded_rel_az=(), excluded_el=()):
    """physical axis value of unit gene j (shape gene 1 -> depth at fixed opening)"""
    u = np.asarray(u, float)
    b = dp.PARAMETRIZATION_R2
    if j == AZIMUTH:
        return kept_vec(u, b['light']['relative_azimuth_deg'], excluded_rel_az)
    if j == ELEVATION:
        return kept_vec(u, b['light']['elevation_deg'], excluded_el)
    if mechanism is None:
        return u
    if mechanism == 'shape':
        lo, hi = (b['shape']['depth_m'] if j == 1 else b['shape'][AXES['shape'][j]])
        return lo + (hi - lo) * u
    kind, lo, hi = gene_specs(mechanism)[j]
    return lo * (hi / lo) ** u if kind == 'log' else lo + (hi - lo) * u


def partial_dependence(model, mechanism, genes, grid=None, prior='primary', n=1024, seed=0, coefs=None,
                       excluded_rel_az=(), excluded_el=(), intercepts=None):
    """Friedman partial dependence on one or two unit genes, averaging the surrogate over a prior background sample;
    with independent genes this equals E[f | gene values]. Bootstrap bands: pass coefs (p, K) with their intercepts."""
    genes = tuple(int(g) for g in genes)
    if len(genes) not in (1, 2):
        raise ValueError('one or two genes')
    if grid is None:
        grid = np.linspace(0, 1, 21 if len(genes) == 1 else 13)
    grid = np.asarray(grid, float)
    bg = sample_prior(owen_sobol(n, DIMENSION, np.random.SeedSequence([int(seed), SOBOL_MC_STREAM + 1])), mechanism, prior)
    f = _evaluator(model, coefs, intercepts)
    combos = [(g,) for g in grid] if len(genes) == 1 else [(g, h) for g in grid for h in grid]
    vals = []
    for c in combos:
        x = bg.copy()
        x[:, list(genes)] = c
        vals.append(f(x).mean(0))
    vals = np.array(vals)
    shape = (len(grid),) * len(genes) + (vals.shape[1],)
    names = [AXES[mechanism][g] if mechanism else 'gene%d' % g for g in genes]
    return dict(genes=list(genes), axes=names, unit_grid=grid.tolist(),
                axis_values=[gene_axis(mechanism, g, grid, excluded_rel_az, excluded_el).tolist() for g in genes],
                values=vals.reshape(shape)[..., 0] if vals.shape[1] == 1 else vals.reshape(shape), prior=prior)


def binned_means(units, values, genes, bins=None, weights=None, n_boot=1000, seed=0, level=.95):
    """model-free check of the partial dependence: weighted means of `values` (NaN = unscorable, excluded) in equal-mass
    cells of one or two unit genes, with bootstrap CIs over all points"""
    u, v = np.asarray(units, float), np.asarray(values, float)
    n, genes = len(v), tuple(int(g) for g in genes)
    bins = bins or (8 if len(genes) == 1 else 4)
    w = _weights(weights, n)
    cell = np.zeros(n, int)
    for g in genes:
        cell = cell * bins + np.minimum((u[:, g] * bins).astype(int), bins - 1)
    ncell = bins ** len(genes)
    fin = np.isfinite(v)
    v0 = np.where(fin, v, 0.)

    def means(i):
        num = np.bincount(cell[i], weights=(w * v0)[i], minlength=ncell)
        den = np.bincount(cell[i], weights=(w * fin)[i], minlength=ncell)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

    est = means(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = np.array([means(rng.integers(0, n, n)) for _ in range(n_boot)]) if n_boot else None
    shape = (bins,) * len(genes)
    out = dict(genes=list(genes), bins=bins, mean=est.reshape(shape).tolist(),
               count=np.bincount(cell[fin], minlength=ncell).reshape(shape).tolist())
    if n_boot:
        ci = _percentile_ci(boot, level)
        out.update(lower=ci[0].reshape(shape).tolist(), upper=ci[1].reshape(shape).tolist())
    return out


# ---------------------------------------------------------------- render noise

def render_noise(repeat_scores, weights=None, level=.95):
    """pooled within-point variance of the repeated renders (rows = noise points in manifest order, column 0 = the main
    design render); weights (e.g. prior importance weights of the points) give the prior-average noise variance"""
    from scipy.stats import chi2, spearmanr
    r = np.asarray(repeat_scores, float)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError('need at least two renders per noise point')
    fin = np.isfinite(r)
    nf = fin.sum(1)
    use = nf >= 2
    w = _weights(weights, len(r))
    if not use.any():
        raise ValueError('no noise point scorable in two renders')
    rr = np.where(fin, r, np.nan)
    var = np.full(len(r), np.nan)
    var[use] = np.nanvar(rr[use], axis=1, ddof=1)
    df = nf - 1
    sigma2 = float(np.sum((w * df * np.nan_to_num(var))[use]) / np.sum((w * df)[use]))
    dft = int(df[use].sum())
    a = 1 - level
    ci = [dft * sigma2 / chi2.ppf(1 - a / 2, dft), dft * sigma2 / chi2.ppf(a / 2, dft)]
    rho = spearmanr(np.nanmean(rr[use], axis=1), np.sqrt(var[use])) if use.sum() >= 3 else None
    return dict(sigma2=sigma2, sd=math.sqrt(sigma2), df=dft, ci=ci if weights is None else None, n_points=int(len(r)),
                n_used=int(use.sum()), n_mixed_scorability=int(((nf > 0) & (nf < r.shape[1])).sum()),
                heteroscedasticity_spearman=None if rho is None else dict(rho=float(rho.statistic), p=float(rho.pvalue)),
                note='chi-square CI assumes homoscedastic Gaussian render noise; unweighted only')


def hard_label_flips(repeat_scores, taus):
    """noise band of the hard label: share of noise points whose renders disagree on s > tau, and the pairwise
    disagreement rate (points scorable in every render)"""
    r = np.asarray(repeat_scores, float)
    full = np.isfinite(r).all(1)
    k = r.shape[1]
    out = []
    for t in taus:
        h = (r[full] > t).sum(1)
        out.append(dict(tau=float(t), n_points=int(full.sum()), flip_share=float(np.mean((h > 0) & (h < k))) if full.any() else float('nan'),
                        pairwise_disagreement=float(np.mean(2 * h * (k - h) / (k * (k - 1)))) if full.any() else float('nan')))
    return out


def noise_adjusted(scores, sigma2, sobol, cv_q2=None, weights=None):
    """deduct render noise: signal variance = observed - noise; indices relative to it (the surrogate-relative ones are
    in `sobol`); CV Q2 relative to its noise ceiling 1 - noise share"""
    s = np.asarray(scores, float)
    ok = np.isfinite(s)
    w = _weights(weights, len(s))[ok]
    x = s[ok]
    mu = np.sum(w * x) / w.sum()
    v_obs = float(np.sum(w * (x - mu) ** 2) / w.sum())
    v_sig = max(v_obs - sigma2, 0.)
    share = sigma2 / v_obs if v_obs > 0 else float('nan')
    out = dict(v_observed=v_obs, sigma2_render=float(sigma2), v_signal=v_sig, noise_share=share, q2_ceiling=1 - share,
               surrogate_variance_over_signal=float(sobol['variance'] / v_sig) if v_sig > 0 else float('nan'))
    if v_sig > 0:
        out.update(first_signal=(np.asarray(sobol['v_first']) / v_sig).tolist(), total_signal=(np.asarray(sobol['v_total']) / v_sig).tolist())
    if cv_q2 is not None:
        out['q2'] = float(cv_q2)
        out['q2_relative'] = float(cv_q2 / (1 - share)) if share < 1 else float('nan')
        out['surrogate_adequate'] = bool(out['q2_relative'] >= Q2_RELATIVE_GATE)
    return out


# ---------------------------------------------------------------- cross-entropy round for rare hard examples (not run)

def ce_trigger(hard, rule=CE_RULE):
    key = 0 if rule['trigger_on'] == 'point' else 1

    def value(h):
        return h['joint']['estimate'] if key == 0 else h['joint']['ci'][1]
    triggered = [h['tau'] for h in hard['thresholds'] if value(h) < rule['trigger_fraction']]
    return dict(triggered=triggered, target=min(triggered) if triggered else None, rule=rule)


def ce_fit_proposal(units, scores, tau, weights=None, rho=CE_RULE['rho'], bins=CE_RULE['bins'], pseudo=CE_RULE['pseudo_count']):
    """one cross-entropy update from the stage-1 sample: level gamma = min(tau, weighted (1 - rho) quantile of the
    scorable scores); elites = scorable with s >= gamma; product of per-gene piecewise-constant densities on `bins`
    equal cells with the weighted elite frequencies (+ pseudo counts)"""
    u, s = np.asarray(units, float), np.asarray(scores, float)
    w = _weights(weights, len(s))
    ok = np.isfinite(s)
    gamma = float(min(tau, weighted_quantile(s[ok], 1 - rho, w[ok])))
    elite = ok & (np.where(ok, s, -np.inf) >= gamma)
    cells = np.minimum((u[elite] * bins).astype(int), bins - 1)
    probs = np.array([np.bincount(cells[:, j], weights=w[elite], minlength=bins) for j in range(u.shape[1])]) + pseudo
    probs = probs / probs.sum(1, keepdims=True)
    return dict(tau=float(tau), gamma=gamma, n_elite=int(elite.sum()), bins=int(bins), rho=rho, pseudo_count=pseudo,
                probs=probs.tolist())


def _hist_density(units, proposal):
    p, k = np.asarray(proposal['probs']), proposal['bins']
    u = np.atleast_2d(np.asarray(units, float))
    cells = np.minimum((u * k).astype(int), k - 1)
    return np.prod(p[np.arange(u.shape[1]), cells] * k, axis=1)


def _hist_inverse(v, proposal):
    p, k = np.asarray(proposal['probs']), proposal['bins']
    out = np.empty_like(v)
    for j in range(v.shape[1]):
        c = np.concatenate([[0.], np.cumsum(p[j])])
        c[-1] = 1.
        cell = np.clip(np.searchsorted(c, v[:, j], side='right') - 1, 0, k - 1)
        out[:, j] = (cell + (v[:, j] - c[cell]) / p[j][cell]) / k
    return np.clip(out, 0., 1.)


def ce_mixture_density(units, proposal, defensive_fraction):
    """stage-2 sampling density on the unit cube: lambda * primary prior (density 1) + (1 - lambda) * histogram product"""
    return defensive_fraction + (1 - defensive_fraction) * _hist_density(units, proposal)


def ce_draw(proposal, mechanism, seed=DEFAULT_SEED, n=CE_RULE['n_stage2'], defensive_fraction=CE_RULE['defensive_fraction']):
    """stage-2 sample with deterministic component counts: round(lambda n) iid prior points, the rest iid from the
    histogram product; weights must use the realized lambda = n_defensive / n"""
    rng = np.random.default_rng(seed_sequence(seed, mechanism, 'ce_sample'))
    nd = int(round(defensive_fraction * n))
    u = np.vstack([rng.random((nd, DIMENSION)), _hist_inverse(rng.random((n - nd, DIMENSION)), proposal)])
    return dict(units=u, component=[0] * nd + [1] * (n - nd), defensive_fraction=nd / n)


def ce_weights(units, proposal, defensive_fraction, mechanism, target='primary'):
    """importance weights target / mixture; bounded by max(target density) / lambda"""
    return prior_density(units, mechanism, target) / ce_mixture_density(units, proposal, defensive_fraction)


def ce_estimate(stage1_units, stage1_scores, stage2_units, stage2_scores, proposal, defensive_fraction, taus, mechanism,
                target='primary', stage1_weight=CE_RULE['stage1_weight'], level=.95):
    """unbiased joint hard fractions: stage 1 = prior design (importance weights target / primary), stage 2 = the CE
    sample (target / mixture, conditionally unbiased given stage 1), combined with the fixed weight alpha; normal CIs.
    The conditional fraction (joint / scorable) is a ratio: consistent, not unbiased."""
    s1, s2 = np.asarray(stage1_scores, float), np.asarray(stage2_scores, float)
    w1 = prior_density(stage1_units, mechanism, target)
    w2 = ce_weights(stage2_units, proposal, defensive_fraction, mechanism, target)
    z = _z(level)
    a = float(stage1_weight)

    def est(values1, values2):
        p1, p2 = values1.mean(), values2.mean()
        e1, e2 = values1.std(ddof=1) / math.sqrt(len(values1)), values2.std(ddof=1) / math.sqrt(len(values2))
        pc, ec = a * p1 + (1 - a) * p2, math.sqrt(a * a * e1 * e1 + (1 - a) ** 2 * e2 * e2)
        return dict(stage1=float(p1), stage1_se=float(e1), stage2=float(p2), stage2_se=float(e2), combined=float(pc),
                    combined_se=float(ec), ci=[max(0., pc - z * ec), pc + z * ec])

    ok1, ok2 = np.isfinite(s1), np.isfinite(s2)
    scor = est(w1 * ok1, w2 * ok2)
    out = dict(target_prior=target, stage1_weight=a, n_stage1=len(s1), n_stage2=len(s2), scorable=scor,
               stage2_max_weight=float(w2.max()), stage2_ess=kish_ess(w2), thresholds=[])
    for t in taus:
        h1 = ok1 & (np.where(ok1, s1, -np.inf) > t)
        h2 = ok2 & (np.where(ok2, s2, -np.inf) > t)
        j = est(w1 * h1, w2 * h2)
        out['thresholds'].append(dict(tau=float(t), hits_stage1=int(h1.sum()), hits_stage2=int(h2.sum()), joint=j,
                                      conditional_ratio=j['combined'] / scor['combined'] if scor['combined'] > 0 else float('nan')))
    return out


def build_ce_manifest(parent, proposal, draw, seed=DEFAULT_SEED, excluded_rel_az=(), excluded_el=(), rule=CE_RULE):
    """stage-2 design manifest; render seeds from a separate stream"""
    mech = parent['mechanism']
    points = [dict(index=i, component=c, unit=u, render_seed=render_seed(seed, mech, i, 0, 'ce_render'),
                   **map_point(mech, u, excluded_rel_az, excluded_el)) for i, (u, c) in enumerate(zip(draw['units'], draw['component']))]
    m = _plain(dict(schema=DESIGN_SCHEMA + '-ce', status=STATUS, mechanism=mech, parent_sha256=parent['sha256'], rule=rule,
                    proposal=proposal, defensive_fraction=draw['defensive_fraction'], seed=dict(base=int(seed)),
                    light=dict(excluded_relative_azimuth_deg=[list(e) for e in excluded_rel_az],
                               excluded_elevation_deg=[list(e) for e in excluded_el]), points=points))
    m['environment'] = environment()
    m['sha256'] = canonical_sha256(m)
    return m


# ---------------------------------------------------------------- one-call analysis (checked on synthetic data only)

def analyse(mechanism, units, scores, taus, repeat_scores=None, excluded_rel_az=(), excluded_el=(),
            priors=('primary', 'all_linear', 'log_positive'), n_boot=2000, n_sobol=4096, sobol_boot=200, seed=0,
            pd_grid=21, pd_grid2=13, pd_background=1024, folds=8):
    u, s = np.asarray(units, float), np.asarray(scores, float)
    ok = np.isfinite(s)
    rep = dict(status=STATUS, mechanism=mechanism, genes=list(GENES[mechanism]), axes=list(AXES[mechanism]), n=len(s),
               n_scorable=int(ok.sum()), taus=[float(t) for t in taus])
    rep['scorable'] = dict(fraction=float(ok.mean()), gate=SCORABLE_GATE, exploratory=bool(ok.mean() < SCORABLE_GATE),
                           by_gene=[binned_means(u, ok.astype(float), (j,), n_boot=n_boot // 4, seed=seed) for j in range(DIMENSION)],
                           light=binned_means(u, ok.astype(float), (AZIMUTH, ELEVATION), n_boot=n_boot // 4, seed=seed))
    rep['priors'] = {}
    for p in priors:
        w = importance_weights(u, mechanism, p)
        rep['priors'][p] = dict(description=PRIORS[p], ess=kish_ess(w),
                                distribution=score_distribution(s, None if p == 'primary' else w, n_boot=n_boot, seed=seed),
                                hard=hard_fractions(s, taus, None if p == 'primary' else w, n_boot=n_boot, seed=seed))
    fit = fit_surrogates(u, s, excluded_rel_az, excluded_el, folds, seed)
    rep['surrogate'] = dict(chosen=fit['chosen'], cv=fit['cv'], rule=fit['rule'], n_fit=fit['n_fit'],
                            gp_length_scales=fit['models']['gp'].length_scales, gp_noise_variance=fit['models']['gp'].noise_variance,
                            pce_degree=fit['models']['pce'].degree, pce_terms=len(fit['models']['pce'].index),
                            pce_analytic_sobol=fit['models']['pce'].sobol_analytic(),
                            note='fitted on scorable points; indices / partial dependence extrapolate into unscorable regions')
    rep['sobol'] = {p: sobol_bootstrap(fit, mechanism, p, sobol_boot, seed, n_sobol) for p in priors}
    model = fit['model']
    rep['partial_dependence'] = dict(
        one=[partial_dependence(model, mechanism, (j,), np.linspace(0, 1, pd_grid), n=pd_background, seed=seed,
                                excluded_rel_az=excluded_rel_az, excluded_el=excluded_el) for j in range(DIMENSION)],
        two=[partial_dependence(model, mechanism, (i, j), np.linspace(0, 1, pd_grid2), n=pd_background, seed=seed,
                                excluded_rel_az=excluded_rel_az, excluded_el=excluded_el)
             for i in range(DIMENSION) for j in range(i + 1, DIMENSION)],
        binned_one=[binned_means(u, s, (j,), n_boot=n_boot // 4, seed=seed) for j in range(DIMENSION)],
        binned_light=binned_means(u, s, (AZIMUTH, ELEVATION), n_boot=n_boot // 4, seed=seed))
    if repeat_scores is not None:
        noise = render_noise(repeat_scores)
        rep['render_noise'] = dict(noise, flips=hard_label_flips(repeat_scores, taus),
                                   adjusted=noise_adjusted(s, noise['sigma2'], rep['sobol']['primary'], fit['cv'][fit['chosen']]['q2']))
    rep['ce'] = ce_trigger(rep['priors']['primary']['hard'])
    return _plain(rep)
