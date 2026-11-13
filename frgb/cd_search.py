"""Each proposal is 6 normalized genes in [0,1]: the mechanism's 4 frozen genes followed by sun azimuth (0-360 deg) and
elevation (15-70 deg), point light. - Both groups start from the common initial distribution (mean 0.5, sigma 0.25,
unit correlation; pycma 4.5.0, BoundTransform on [0,1]); generation 0 is drawn from one shared stream, so C and D
share their first batch. - D: a single CMA-ES engine per cell (the shared generation 0 is its first ask) updated by
tell each generation. - C: every later generation is an independent ask from a fresh engine at the initial
distribution (no update). - Streams: SeedSequence([seed, round, mechanism, cell, stream]); stream 0 = shared
generation 0, 1 + g = C generation g."""
import numpy as np

POPULATION = 8
GENERATIONS = 4
MEAN, SIGMA = .5, .25
MECHANISMS = {'shape': 1, 'stain': 2}
DIMENSION = 6
LIGHT_AZIMUTH = (0., 360.)
LIGHT_ELEVATION = (15., 70.)


def stream_seed(seed, round_index, mechanism, cell, stream):
    state = np.random.SeedSequence([int(seed), int(round_index), MECHANISMS[mechanism], int(cell), int(stream)]).generate_state(1)[0]
    return int(state % (2 ** 31 - 2)) + 1


def _engine(seed):
    import cma
    if cma.__version__ != '4.5.0':
        raise ValueError('Frozen installed pycma 4.5.0 required')
    return cma.CMAEvolutionStrategy([MEAN] * DIMENSION, SIGMA, dict(popsize=POPULATION, seed=seed, bounds=[0, 1],
                                    BoundaryHandler=cma.BoundTransform, verbose=-9, verb_log=0, verb_disp=0, maxiter=GENERATIONS))


def state(es):
    return dict(mean=np.asarray(es.mean).tolist(), sigma=float(es.sigma), covariance=np.asarray(es.sm.C).tolist())


EXCLUDED_AZIMUTH = {'stain': ((90., 180.),), 'shape': ((90., 180.), (180., 270.))}
EXCLUDED_ELEVATION = {'shape': ((56.25, 70.),)}


def _kept(unit, bounds, excluded):
    value = bounds[0] + (bounds[1] - bounds[0] - sum(hi - lo for lo, hi in excluded)) * unit
    for lo, hi in sorted(excluded):
        if value > lo:
            value += hi - lo
    return value


def lighting(unit, mechanism=None):
    return dict(azimuth_deg=_kept(float(unit[4]), LIGHT_AZIMUTH, EXCLUDED_AZIMUTH.get(mechanism, ())),
                elevation_deg=_kept(float(unit[5]), LIGHT_ELEVATION, EXCLUDED_ELEVATION.get(mechanism, ())))


def losses(scores, qualified):
    valid = [-s for s, q in zip(scores, qualified) if q]
    penalty = max(valid) + max(1., max(abs(x) for x in valid)) if valid else 1.
    return [-s if q else penalty for s, q in zip(scores, qualified)]


class CellSearch:
    def __init__(self, group, seed, round_index, mechanism, cell):
        if group not in ('C', 'D'):
            raise ValueError('group must be C or D')
        self.group, self.key = group, (int(seed), int(round_index), mechanism, int(cell))
        # pycma seeds and samples the GLOBAL numpy RNG; each search keeps its own state so interleaved
        outer = np.random.get_state()
        self.engine = _engine(stream_seed(*self.key, 0))
        self.np_state = np.random.get_state()
        np.random.set_state(outer)
        self.generation = 0
        self.history = []

    def ask(self):
        if self.generation >= GENERATIONS:
            raise ValueError('all %d generations already asked' % GENERATIONS)
        outer = np.random.get_state()
        try:
            if self.generation == 0 or self.group == 'D':
                np.random.set_state(self.np_state)
                before = state(self.engine)
                x = self.engine.ask()
                self.np_state = np.random.get_state()
            else:
                fresh = _engine(stream_seed(*self.key, 1 + self.generation))
                before = state(fresh)
                x = fresh.ask()
        finally:
            np.random.set_state(outer)
        x = [np.asarray(v, float) for v in x]
        if any(v.shape != (DIMENSION,) or np.any((v < 0) | (v > 1)) for v in x):
            raise ValueError('proposal outside [0,1]^6')
        self.pending = dict(generation=self.generation, state_before_ask=before, parameters=[v.tolist() for v in x])
        return [v.copy() for v in x]

    def tell(self, scores, qualified):
        p = self.pending
        if len(scores) != POPULATION or len(qualified) != POPULATION:
            raise ValueError('one score and one qualification flag per proposal')
        loss = losses(scores, qualified)
        if self.group == 'D':
            outer = np.random.get_state()
            np.random.set_state(self.np_state)
            try:
                self.engine.tell([np.asarray(v) for v in p['parameters']], loss)
                self.np_state = np.random.get_state()
            finally:
                np.random.set_state(outer)
        self.history.append(dict(p, losses=loss, qualified_count=int(sum(bool(q) for q in qualified)),
                                 state_after_tell=state(self.engine) if self.group == 'D' else None))
        del self.pending
        self.generation += 1


GROUPS = ('C', 'D')
SELECT_PER_CELL = 8
BINS = 4


def select_binned(rows, limit=SELECT_PER_CELL):
    out, used = [], set()
    for row in sorted((r for r in rows if r['qualified'] and r['eligible_hard']), key=lambda r: (-r['score'], r['generation'], r['index'])):
        key = tuple(np.minimum((np.asarray(row['parameters']) * BINS).astype(int), BINS - 1))
        if key in used:
            continue
        used.add(key)
        out.append(row)
        if len(out) == limit:
            break
    return out


def _new_row(mech, cell, group, gen, index, x, seed, round_index):
    return dict(mechanism=mech, cell=cell, group=group, generation=gen, index=index, parameters=np.asarray(x).tolist(),
                name='fcd_s%d_r%d_%s%d_%s_g%d_i%d' % (seed, round_index, mech, cell, group, gen, index),
                candidate=None, gate=None, qualified=False, scores={})


def run_round_search(cells, *, seed, round_index, render, qualify, score, charge_proposals, same_models, state_path, log=print,
                     render_batch=None, qualify_batch=None):
    """Resumable (pickle at state_path).

    render(mechanism, cell, name, unit) -> candidate dict; qualify(mechanism, candidate) -> gate with bool 'qualified';
    score(group, mechanism, candidate, gate) -> dict(score=float, eligible_hard=bool, ...); charge_proposals(n,
    purpose). Proposals are charged per group before each ask (16 for a shared generation 0, 8 otherwise). Failed G
    keeps its slot and is never scored. Render speedup: in generations >= 1 both groups ask before either tells
    (independent searches with their own RNG states, so proposals are unchanged); the whole cell-generation batch is
    then rendered/qualified together through the optional render_batch([(mech, cell, name, unit)]) -> {name:
    candidate} and qualify_batch([(mech, name, candidate)]) -> {name: gate}; rows they leave out fall back to
    render/qualify per row. Scoring and tells keep the original per-row order. Returns dict(status,
    selected={'mechanism:cell:group': [rows]}, rows=[...])."""
    import json, os, pickle
    from pathlib import Path
    path = Path(state_path)
    contract = dict(cells=[list(c) for c in cells], seed=seed, round=round_index, same_models=bool(same_models))
    if path.exists():
        with path.open('rb') as f:
            st = pickle.load(f)
        if st['contract'] != contract:
            raise ValueError('search state belongs to another contract')
        if st.get('in_flight'):
            raise RuntimeError('unresolved in-flight proposal charge/ask needs inspection; no automatic retry: %s' % (st['in_flight'],))
    else:
        st = dict(contract=contract, rows=[], searches={}, told=set(), status='running')

    def save():
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        js = dict(contract=contract, status=st['status'], rows=st['rows'], told=sorted(':'.join(map(str, t)) for t in st['told']),
                  selected=st.get('selected'))
        tmpj = path.with_suffix('.json.tmp')
        tmpj.write_text(json.dumps(js, indent=1, default=str) + '\n', encoding='utf8')
        os.replace(tmpj, path.with_suffix('.json'))

    def rows_of(mech, cell, group, gen):
        return [r for r in st['rows'] if r['mechanism'] == mech and r['cell'] == cell and r['group'] == group and r['generation'] == gen]

    def process(row, groups):
        if row.get('candidate') is None:
            row['candidate'] = render(row['mechanism'], row['cell'], row['name'], np.asarray(row['parameters']))
            save()
        if row.get('gate') is None:
            row['gate'] = qualify(row['mechanism'], row['candidate'])
            row['qualified'] = row['gate'].get('qualified') is True
            save()
        for g in groups:
            if row['qualified'] and g not in row['scores']:
                if same_models and g == 'D' and 'C' in row['scores']:
                    row['scores']['D'] = dict(row['scores']['C'], copied_from='C (same model)')
                else:
                    s = score(g, row['mechanism'], row['candidate'], row['gate'])
                    if not np.isfinite(s['score']):
                        raise ValueError('qualified objective is not finite')
                    row['scores'][g] = s
                save()

    def prepare(batch):
        pending = [r for r in batch if r.get('candidate') is None]
        if pending and render_batch is not None:
            done = render_batch([(r['mechanism'], r['cell'], r['name'], np.asarray(r['parameters'])) for r in pending])
            for r in pending:
                if r['name'] in done:
                    r['candidate'] = done[r['name']]
            save()
        pending = [r for r in batch if r.get('candidate') is not None and r.get('gate') is None]
        if pending and qualify_batch is not None:
            gates = qualify_batch([(r['mechanism'], r['name'], r['candidate']) for r in pending])
            for r in pending:
                if r['name'] in gates:
                    r['gate'] = gates[r['name']]
                    r['qualified'] = r['gate'].get('qualified') is True
            save()

    def view(row, g):
        s = row['scores'].get(g, {})
        return dict(row, score=float(s.get('score', 0.)), eligible_hard=bool(s.get('eligible_hard')) and row['qualified'])

    def tell(mech, cell, g, gen, batch):
        if (mech, cell, g, gen) not in st['told']:
            st['searches'][(mech, cell, g)].tell([view(r, g)['score'] for r in batch], [r['qualified'] for r in batch])
            st['told'].add((mech, cell, g, gen))
            save()

    try:
        for mech, cell in cells:
            for g in GROUPS:
                st['searches'].setdefault((mech, cell, g), CellSearch(g, seed, round_index, mech, cell))
            for gen in range(GENERATIONS):
                if gen == 0:
                    if not rows_of(mech, cell, 'S', 0):
                        st['in_flight'] = ('ask', mech, cell, 'S', 0); save()
                        charge_proposals(2 * POPULATION, 'formal C+D shared gen0 s%d r%d %s%d' % (seed, round_index, mech, cell))
                        xc = st['searches'][(mech, cell, 'C')].ask()
                        xd = st['searches'][(mech, cell, 'D')].ask()
                        if not all(np.array_equal(a, b) for a, b in zip(xc, xd)):
                            raise RuntimeError('C and D generation 0 differ')
                        st['rows'] += [_new_row(mech, cell, 'S', 0, i, x, seed, round_index) for i, x in enumerate(xc)]
                        st['in_flight'] = None; save()
                    batch = rows_of(mech, cell, 'S', 0)
                    prepare(batch)
                    for row in batch:
                        process(row, GROUPS)
                    for g in GROUPS:
                        tell(mech, cell, g, 0, batch)
                    continue
                for g in GROUPS:
                    if not rows_of(mech, cell, g, gen):
                        st['in_flight'] = ('ask', mech, cell, g, gen); save()
                        charge_proposals(POPULATION, 'formal %s gen%d s%d r%d %s%d' % (g, gen, seed, round_index, mech, cell))
                        x = st['searches'][(mech, cell, g)].ask()
                        st['rows'] += [_new_row(mech, cell, g, gen, i, v, seed, round_index) for i, v in enumerate(x)]
                        st['in_flight'] = None; save()
                batches = {g: rows_of(mech, cell, g, gen) for g in GROUPS}
                prepare([row for g in GROUPS for row in batches[g]])
                for g in GROUPS:
                    for row in batches[g]:
                        process(row, (g,))
                    tell(mech, cell, g, gen, batches[g])
            log('cell %s%d done' % (mech, cell))
        selected = {}
        for mech, cell in cells:
            for g in GROUPS:
                pool = [view(r, g) for r in st['rows'] if r['mechanism'] == mech and r['cell'] == cell and r['group'] in ('S', g)]
                selected['%s:%d:%s' % (mech, cell, g)] = [dict(name=r['name'], score=r['score'], generation=r['generation'],
                                                              index=r['index'], parameters=r['parameters']) for r in select_binned(pool)]
        st['selected'] = selected
        st['status'] = 'awaiting_batch_review'
        save()
        return dict(status=st['status'], selected=selected, rows=st['rows'])
    except BaseException as error:
        st['status'] = 'paused: ' + repr(error)
        save()
        raise
