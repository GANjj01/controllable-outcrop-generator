"""Writes CONFIG_OUT (template + effective fault_count [+ meshing budget]) and OUT_JSON with both counts (the covariates)
and the check that the spec drawn with the effective count differs from the natural spec ONLY by faults (added /
dropped fault events and their planes) and that the faults common to both are identical. Usage: python
fs_effective_config.py CONFIG_JSON SEED CONFIG_OUT OUT_JSON [--cap N | --set K [--planes 1024]]"""
import json, sys
from pathlib import Path
import fs_nfaults as nf

a = sys.argv[1:]
cap = int(a[a.index('--cap') + 1]) if '--cap' in a else None
setk = int(a[a.index('--set') + 1]) if '--set' in a else None
planes = int(a[a.index('--planes') + 1]) if '--planes' in a else None
cfg = json.loads(Path(a[0]).read_text(encoding='utf8')); cfg['seed'] = int(a[1]); cfg.get('geometry', {}).pop('outcrop_type', None)
nat = nf.raw_of(json.loads(json.dumps(cfg)), True); n = len(nf.faults(nat))
eff_n = setk if setk is not None else (n if cap is None else min(n, cap))
eff = json.loads(json.dumps(cfg)); eff['geometry']['fault_count'] = eff_n
if planes is not None: eff['geometry']['meshing'] = dict(budget=dict(max_planes_per_cell=planes))
check_cfg = json.loads(json.dumps(eff)); check_cfg['geometry'].pop('meshing', None)     # the meshing budget does not enter sample_spec
got = nf.raw_of(check_cfg, False)
fa, fb = set(nf.faults(nat)), set(nf.faults(got))
is_fault = lambda x: any(f in str(x.get('id', '')) for f in fa | fb)  # noqa: E731
nonf = lambda r: {k: ([x for x in v if not is_fault(x)] if k in ('events', 'geometry') else v) for k, v in r.items()}  # noqa: E731
common = lambda r: [x for k in ('events', 'geometry') for x in r[k] if any(f in str(x.get('id', '')) for f in fa & fb)]  # noqa: E731
ok = (json.dumps(nonf(nat), sort_keys=True, default=str) == json.dumps(nonf(got), sort_keys=True, default=str)
      and json.dumps(common(nat), sort_keys=True, default=str) == json.dumps(common(got), sort_keys=True, default=str))
rep = dict(seed=int(a[1]), config=a[0], n_faults_natural=n, n_faults_effective=eff_n, cap=cap, set=setk, max_planes_per_cell=planes,
           kept_fault_ids=sorted(fa & fb), dropped_fault_ids=sorted(fa - fb), added_fault_ids=sorted(fb - fa), differs_only_by_faults=ok,
           intrusions=sorted(e['id'] for e in nat['events'] if e.get('kind') == 'intrusion'),
           rule='r1n set %s' % setk if setk is not None else ('r1j' if cap is None else 'r1m cap %d' % cap))
Path(a[2]).write_text(json.dumps(eff, indent=1) + '\n', encoding='utf8'); Path(a[3]).write_text(json.dumps(rep, indent=1) + '\n', encoding='utf8')
print(json.dumps(rep)); sys.exit(0 if ok else 3)
