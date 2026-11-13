"""Per-frame checks of a labelled frame.

Label checks (the five numeric items applied to every aerial frame of the paper):
  1_gates         the label program finished (exit 0) and no stop gate fired (centre deviation, in-cavity coverage, contradictions);
  2_jumps         no trace pixel jumped to another branch during position refinement;
  3_displacement  share of the visible trace refined by more than 2 px <= 0.5 %;
  4_breaks        new breaks in the trace runs after refinement <= max(1, floor(0.005 * visible length / 6));
  5_capped        share of the in-frame trace whose refinement hit the step limit and was reverted <= 2 %.
Geometry checks recorded by the chain: the field-normal re-evaluation residual (<= 1e-6 m) and the refinement shape error
(per-zone and frame p99 <= 0.5 px)."""
import json
import math
from pathlib import Path


def label_items(rep):
    """the five items from a label report (report.json of the label program)"""
    jc = rep.get('jump_check') or {}
    vis = float(rep['visible_length_px'])
    inf = jc.get('in_frame_length_px')
    dbreak = None if not jc else jc['run_breaks_after'] - jc['run_breaks_before']
    break_limit = max(1, math.floor(.005 * vis / 6))
    return {
        '1_gates': dict(exit_code=rep.get('exit_code'), stop=rep.get('stop'),
                        ok=rep.get('exit_code') == 0 and not any((rep.get('stop') or {}).values())),
        '2_jumps': dict(jumped_px=jc.get('jumped_px'), ok=bool(jc) and jc['jumped_px'] == 0),
        '3_displacement': dict(fraction=jc.get('displaced_over_2px_fraction'), limit=.005,
                               ok=bool(jc) and (jc['displaced_over_2px_fraction'] or 0.) <= .005),
        '4_breaks': dict(before=jc.get('run_breaks_before'), after=jc.get('run_breaks_after'), increase=dbreak, limit=break_limit,
                         ok=bool(jc) and dbreak <= break_limit),
        '5_capped': dict(capped_px=jc.get('capped_px'), in_frame_px=inf, fraction=(jc['capped_px'] / inf) if jc and inf else None,
                         limit=.02, ok=bool(jc) and (jc['capped_px'] <= .02 * inf if inf else jc['capped_px'] == 0)),
    }


def frame_checks(label_dir, cam=None, refine=None):
    rep = json.loads((Path(label_dir) / 'report.json').read_text(encoding='utf-8'))
    items = label_items(rep)
    out = dict(label_status=rep.get('status'), label_items={k: v['ok'] for k, v in items.items()}, label_details=items,
               visible_length_px=rep.get('visible_length_px'), positive_pixels=rep.get('positive_pixels'),
               u_line_px=rep.get('u_line_px'), ignore_fraction_of_rock=rep.get('ignore_fraction_of_rock'))
    if refine:
        fr = refine.get('field_report') or {}
        r = (fr.get('reevaluation_vs_vertex_m') or {}).get('max')
        out['field_normal_residual_m'] = dict(max=r, limit=1e-6, ok=r is not None and r <= 1e-6)
    out['overall'] = 'PASS' if all(v['ok'] for v in items.values()) and all(
        v.get('ok', True) for v in out.values() if isinstance(v, dict) and 'limit' in v) else 'FAIL'
    return out
