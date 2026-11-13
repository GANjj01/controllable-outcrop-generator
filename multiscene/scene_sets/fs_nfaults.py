"""This script runs the generator's own build_declaration up to sample_spec twice: with pipeline.sample_spec called with
fault_count=None (the natural draw) and with the config's explicit value; it reports the natural n_faults and checks
that the spec drawn with fault_count = natural equals the natural spec (json-equal). Usage: python fs_nfaults.py
CONFIG_JSON SEED OUT_JSON [--write CONFIG_OUT]"""
import json, sys
from pathlib import Path
from outcrop.dsgen import pipeline


class _Stop(Exception):
    pass


def raw_of(cfg, natural):
    got = {}; orig_ss, orig_se = pipeline.sample_spec, pipeline.sample_exposure

    def ss(seed, **kw):
        if natural: kw['fault_count'] = None
        got['raw'] = orig_ss(seed, **kw); return got['raw']

    def stop(*a, **k):
        raise _Stop()
    pipeline.sample_spec, pipeline.sample_exposure = ss, stop
    try:
        pipeline.build_declaration(cfg)
    except _Stop:
        pass
    finally:
        pipeline.sample_spec, pipeline.sample_exposure = orig_ss, orig_se
    return got['raw']


def faults(raw):
    return sorted(e['id'] for e in raw['events'] if e.get('kind') == 'fault')


if __name__ == '__main__':
    a = sys.argv[1:]
    cfg = json.loads(Path(a[0]).read_text(encoding='utf8')); cfg['seed'] = int(a[1]); cfg.get('geometry', {}).pop('outcrop_type', None)
    nat = raw_of(json.loads(json.dumps(cfg)), True); n = len(faults(nat))
    eff = json.loads(json.dumps(cfg)); eff['geometry']['fault_count'] = n
    same = json.dumps(raw_of(eff, False), sort_keys=True) == json.dumps(nat, sort_keys=True)
    cur = raw_of(json.loads(json.dumps(cfg)), False)
    rep = dict(seed=int(a[1]), config=a[0], config_fault_count=cfg['geometry'].get('fault_count'), natural_faults=n, natural_fault_ids=faults(nat),
               spec_with_natural_count_equals_natural=same, spec_as_configured_equals_natural=json.dumps(cur, sort_keys=True) == json.dumps(nat, sort_keys=True),
               intrusions=sorted(e['id'] for e in nat['events'] if e.get('kind') == 'intrusion'))
    if '--write' in a:
        out = Path(a[a.index('--write') + 1]); out.write_text(json.dumps(eff, indent=1) + '\n', encoding='utf8'); rep['effective_config'] = str(out)
    Path(a[2]).write_text(json.dumps(rep, indent=1) + '\n', encoding='utf8'); print(json.dumps(rep))
