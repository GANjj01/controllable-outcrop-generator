"""Opt-in fracture density per unit area.

count = round(D x domain plan area) (deterministic, not the Poisson draw of sample_network's own density_m2, so that
the direction groups keep fixed quotas); the direction-group quotas are scaled to that count by the largest-remainder
rule (ties: group order), each group keeping >= 1. The config's fracture_count (validated 0..20) is not used on this
path, so the cap of 20 does not apply. With D = 16 / 1600 m^2 = 0.01 on the calibrated 40 m domain, count and quotas
equal the default config (checked by selftest). PROVISIONAL_RANGE_PER_M2: x 0.5 .. x 2 of the calibrated 0.01. Usage:
python fracture_density.py selftest CONFIG_JSON"""
import functools, json, math, sys
from copy import deepcopy

CALIBRATED_PER_M2 = 16 / 1600.
PROVISIONAL_RANGE_PER_M2 = (.005, .02)


def scaled_groups(n, groups):
    q = [int(g['count']) for g in groups]; tot = sum(q)
    if n < len(groups): raise ValueError('density gives %d fractures for %d direction groups' % (n, len(groups)))
    raw = [n * c / tot for c in q]; base = [max(1, int(math.floor(x))) for x in raw]
    left = n - sum(base)
    order = sorted(range(len(q)), key=lambda i: (-(raw[i] - math.floor(raw[i])), i))
    k = 0
    while left > 0:
        base[order[k % len(order)]] += 1; left -= 1; k += 1
    while left < 0:                                   # only when the >= 1 floor overshot
        i = max(range(len(base)), key=lambda j: (base[j], -j)); base[i] -= 1; left += 1
    out = deepcopy(groups)
    for g, c in zip(out, base):
        g['count'] = int(c)
    return out


def install(density, record):
    import outcrop.dsgen.paper_geology as pg
    if not (isinstance(density, float) and math.isfinite(density) and density > 0): raise ValueError('density must be > 0')
    original = pg.apply_fracture_network

    @functools.wraps(original)
    def apply_fracture_network(spec, options):
        cfg = dict(options)
        lo, hi = spec['domain']['min_m'], spec['domain']['max_m']; area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        n = int(round(density * area)); old = cfg.get('count')
        cfg['count'] = n
        if cfg.get('direction_groups'):
            cfg['direction_groups'] = scaled_groups(n, cfg['direction_groups'])
        record.append(dict(density_per_m2=density, domain_plan_m2=area, count=n, config_count=old,
                           groups=[(g['id'], g['count']) for g in cfg.get('direction_groups') or []],
                           provisional_range_per_m2=list(PROVISIONAL_RANGE_PER_M2), in_range=bool(PROVISIONAL_RANGE_PER_M2[0] <= density <= PROVISIONAL_RANGE_PER_M2[1])))
        return original(spec, cfg)
    pg.apply_fracture_network = apply_fracture_network
    return pg, original


def selftest(config):
    from outcrop.dsgen.pipeline import resolve_config, build_declaration
    cfg = resolve_config(json.loads(open(config, encoding='utf8').read()))
    fr = lambda raw: [g for g in raw['geometry'] if g['id'].startswith('network-joint')]
    a = fr(build_declaration(cfg)[0]); rec = []
    pg, orig = install(CALIBRATED_PER_M2, rec)
    try:
        b = fr(build_declaration(cfg)[0])
    finally:
        pg.apply_fracture_network = orig
    rep = dict(default_count=len(a), density_count=len(b), identical=json.dumps(a) == json.dumps(b), record=rec,
               groups_check=[scaled_groups(n, [dict(id='g30', count=8), dict(id='g120', count=8)]) for n in (5, 16, 27, 33)])
    print(json.dumps(rep, indent=1)); return rep


if __name__ == '__main__':
    if sys.argv[1] == 'selftest': selftest(sys.argv[2])
