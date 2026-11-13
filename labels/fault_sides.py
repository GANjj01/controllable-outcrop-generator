"""Per-side joint-label check of a faulted frame. Sides, plane distances and the other-block test all come from
frgb.fault_charts (signed_other on the module's compiled frames); groups from build.json (fault_support).

Per fault F that acted on a sample's joint, the sample is on (F, moving) when F is in its group's block, else on (F,
fixed). Near-fault band (excluded from the verdict, reported): |distance to F's plane| < max(0.02 m, 3 px footprint)
UNION the cap-reach set (the G1 section P' +- (w + 1 mm) n_slot touches or crosses F's plane; w = the sample's
aperture field). Per side outside the band, non-tip: >= 200 px -> PASS if length-weighted p99 |c| <= 1 mm and inside
>= 99 %, else FAIL; 0 < px < 200 -> NOT JUDGED."""
import json
from pathlib import Path

import numpy as np

JUDGE_MIN_PX = 200.
P99_M = 1e-3
INSIDE_MIN = .99
BAND_FLOOR_M = .02
BAND_PX = 3.
CAP_REACH_PAD_M = 1e-3
CLOSURE_TOL_M = 1e-6


def _q(x, w, m, qs=(.5, .99)):
    ok = m & np.isfinite(x)
    if not ok.any():
        return None
    o = np.argsort(x[ok]); cw = np.cumsum(w[ok][o]) / w[ok].sum()
    return [float(x[ok][o][min(np.searchsorted(cw, qq), ok.sum() - 1)]) for qq in qs]


def _side_stats(s, m):
    w = s['w']
    absc = np.abs(s['c'])
    px = float(w[m].sum())
    rec = dict(nontip_px_outside_band=px,
               abs_centre_m_p50_p99=_q(absc, w, m),
               abs_centre_max_m=float(absc[m & np.isfinite(absc)].max()) if (m & np.isfinite(absc)).any() else None,
               centre_px_p50_p99=_q(s['px'], w, m),
               inside_fraction=float(w[m & s['inside'].astype(bool)].sum() / px) if px else None,
               no_slot_found_px=float(w[m & ~np.isfinite(s['c'])].sum()),
               mesh_wall_deviation_m_p99=(_q(s['dev'], w, m, (.99,)) or [None])[0],
               mesh_wall_deviation_over_1mm_px=float(w[m & (s['dev'] > 1e-3)].sum()),
               abs_phi_m_p99=(_q(s['phi'], w, m, (.99,)) or [None])[0])
    if px <= 0:
        rec['verdict'] = 'none visible'
    elif px < JUDGE_MIN_PX:
        rec['verdict'] = 'NOT JUDGED (<200 px)'
    else:
        p99 = rec['abs_centre_m_p50_p99'][1] if rec['abs_centre_m_p50_p99'] else None
        rec['verdict'] = 'PASS' if (p99 is not None and p99 <= P99_M and rec['inside_fraction'] >= INSIDE_MIN) else 'FAIL'
    return rec


def side_check(g1, groups, frames, FC, case, samples):
    """g1: run_frames_r3.gate1(..)['_samples']; groups: fault_support.path_groups(build). Returns (record, arrays)."""
    faults = frames['faults']
    iface = {i: p['row']['interface_id'] for i, p in case['paths'].items()}
    n = len(g1['w'])
    nontip = ~g1['tip'].astype(bool)
    arrays = dict(weight=g1['w'], c=g1['c'], inside=g1['inside'].astype(bool), tip=g1['tip'].astype(bool), path=g1['path'],
                  branch=g1['branch'], sample=g1['sample'], point=g1['point'])
    sides, closure = {}, {}
    for f in faults:
        acted = np.array([iface[int(p)] in f['moved'] for p in g1['path']], bool) if n else np.zeros(0, bool)
        if not acted.any():
            continue
        moved = np.array([bool(groups[int(p)]['block']) and f['id'] in groups[int(p)]['block'] for p in g1['path']], bool)
        d = np.full(n, np.nan); reach = np.zeros(n, bool)
        for flag in (True, False):
            sel = acted & (moved == flag)
            if not sel.any():
                continue
            d[sel] = FC.signed_other(f, flag, g1['point'][sel])
            ends = [g1['point'][sel] + sgn * (g1['aperture'][sel] + CAP_REACH_PAD_M)[:, None] * g1['slot_normal'][sel] for sgn in (1., -1.)]
            reach[sel] = np.maximum(FC.signed_other(f, flag, ends[0]), FC.signed_other(f, flag, ends[1])) >= 0
        dist_band = acted & (np.abs(d) < np.maximum(BAND_FLOOR_M, BAND_PX * g1['foot']))
        band = dist_band | (acted & reach)
        arrays['side_%s' % f['id']] = np.where(acted, np.where(moved, 1, 0), -1).astype(np.int8)
        arrays['signed_other_%s' % f['id']] = d
        arrays['band_%s' % f['id']] = band
        w = g1['w']
        for name, flag in (('moving', True), ('fixed', False)):
            on = acted & (moved == flag)
            rec = _side_stats(g1, on & nontip & ~band)
            rec.update(band_px=dict(distance=float(w[on & dist_band].sum()), cap_reach=float(w[on & reach].sum()),
                                    union=float(w[on & band].sum())),
                       band_abs_centre_m_p50_p99=_q(np.abs(g1['c']), w, on & nontip & band),
                       band_inside_fraction=float(w[on & nontip & band & g1['inside'].astype(bool)].sum() / w[on & nontip & band].sum())
                       if (on & nontip & band).any() else None)
            sides['%s:%s' % (f['id'], name)] = rec
        mv, fx = sides['%s:moving' % f['id']], sides['%s:fixed' % f['id']]
        ratio = lambda k: (mv['abs_centre_m_p50_p99'][k] / fx['abs_centre_m_p50_p99'][k]
                           if mv['abs_centre_m_p50_p99'] and fx['abs_centre_m_p50_p99'] and fx['abs_centre_m_p50_p99'][k] else None)
        sides['%s:moving_over_fixed' % f['id']] = dict(p50=ratio(0), p99=ratio(1))
    for key, r in samples.items():
        ok = np.isfinite(r['point']).all(1)
        g = groups[int(r['path'])]
        for f in faults:
            if iface[int(r['path'])] not in f['moved'] or not ok.any():
                continue
            mv = bool(g['block']) and f['id'] in g['block']
            worst = float(np.max(FC.signed_other(f, mv, r['point'][ok])))
            k = '%s:%s' % (g['key'], f['id'])
            closure[k] = max(closure.get(k, -np.inf), worst)
    violated = {k: v for k, v in closure.items() if v > CLOSURE_TOL_M}
    fails = [k for k, v in sides.items() if isinstance(v, dict) and v.get('verdict') == 'FAIL']
    record = dict(rule='r4 faults amendments b (item 5), d, e', judge_min_px=JUDGE_MIN_PX, p99_limit_m=P99_M, inside_min=INSIDE_MIN,
                  band='|d| < max(%.2f m, %g px footprint) UNION cap reach (w + %g m)' % (BAND_FLOOR_M, BAND_PX, CAP_REACH_PAD_M),
                  sides=sides, failing_sides=fails, usable=not fails, closure_b=dict(max_other_block_m=closure, violated=violated))
    return record, arrays


def write(out, record, arrays, extra_records):
    """fault_sides.json, g1_samples.npz, the report field and UNUSABLE_FAULT_SIDES.txt"""
    out = Path(out)
    record = dict(record, **extra_records)
    (out / 'fault_sides.json').write_text(json.dumps(record, indent=1, allow_nan=False, default=float) + '\n', encoding='utf8')
    np.savez_compressed(out / 'g1_samples.npz', **arrays)
    if not record['usable']:
        lines = []
        for k in record['failing_sides']:
            s = record['sides'][k]
            lines.append('%s: p99 |c| %s m, inside %s, %.1f px' % (k, None if not s['abs_centre_m_p50_p99'] else s['abs_centre_m_p50_p99'][1],
                                                                  s['inside_fraction'], s['nontip_px_outside_band']))
        (out / 'UNUSABLE_FAULT_SIDES.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return dict(usable=record['usable'], sides={k: v['verdict'] for k, v in record['sides'].items() if 'verdict' in v},
                file='fault_sides.json')
