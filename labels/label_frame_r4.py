"""label_frame.py + fault traces.

Same arguments, outputs and exit codes as label_frame.py, plus: source_type.npy   uint8, shape of target: 0 none, 1
joint, 2 fault, 3 both (a joint-fault crossing) report.json       'faults' (status, planes, checks, per-fault
lengths), 'per_type' (joint / fault visible, U, positives) Joint labels are computed by label_frame.py unchanged
(frgb.trace_labels.build_trace_labels is wrapped, not edited); fault labels are merged by
frgb.fault_traces.merge_labels, which never removes a joint positive. Use absolute paths for --out (as for
label_frame.py)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(P1)); sys.path.insert(0, str(HERE))
from frgb import fault_traces as FT  # noqa: E402
from frgb import trace_labels as tl  # noqa: E402
from frgb.metrics import graph_node_weights  # noqa: E402

RULE = 'trace_labels r3 (+ amendments a, b, c, d, e; T1; speed-up s1) + r4 faults'
STATE = {}
_build = tl.build_trace_labels


def _jsonable(f):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.items()}


def build_r4(case, samples_by_path, cam, rgb, *, cpu_raster, render_rock, aligned, ground=None, sky=None, unresolved_points=None,
             export_windows=False, occlusion_regions=None, mismatch_region=None, opening_base=None):
    J = _build(case, samples_by_path, cam, rgb, cpu_raster=cpu_raster, render_rock=render_rock, aligned=aligned, ground=ground,
               sky=sky, unresolved_points=unresolved_points, export_windows=export_windows, occlusion_regions=occlusion_regions,
               mismatch_region=mismatch_region, opening_base=opening_base)     # mismatch_opening_r1
    t = time.time()
    source = case['build']['source']
    planes = FT.fault_planes(source)
    rec = dict(status=planes['status'], reasons=planes['reasons'], source=str(source), declaration=planes.get('declaration'),
               faults=[_jsonable(f) for f in planes['faults']])
    F = None
    if planes['status'] == 'ok' and planes['faults']:
        fs, info = FT.fault_samples(case, cam, planes['faults'])
        if fs:
            nat = case['native']
            boxes = tl._projected_boxes(nat['vertices'], nat['faces'], cam)[:3]
            FT.classify_fault_visibility(case, fs, cam, boxes=boxes)
            F = FT.fault_labels(case, fs, cam, rgb, cpu_raster=cpu_raster, render_rock=render_rock, aligned=aligned, ground=ground,
                                sky=sky, export_windows=export_windows, gid0=len(J['windows']))
        for f in rec['faults']:
            fd = next(x for x in planes['faults'] if x['id'] == f['id'])
            rows = [r for r in (F['rows'] if F else []) if r['fault'] == f['id']]
            f.update(info.get(f['id'], {}), checks=FT.plane_and_generator_checks(source, fd, fs),
                     **{k: float(sum(r[k] for r in rows)) for k in ('projected_in_frame_px', 'visible_px', 'occluded_px', 'in_cavity_px',
                                                                     'visible_u_px')},
                     branches=len(rows), windows=sum(1 for w in (F['windows'] if F else []) if w['fault'] == f['id']),
                     u_windows=sum(1 for w in (F['windows'] if F else []) if w['fault'] == f['id'] and w['u']))
        rec['branch_rows'] = F['rows'] if F else []
    merged, st = FT.merge_labels(J, F, export_windows=export_windows)
    if F is not None and case.get('occlusion') is not None:
        import occlusion_rule as OR
        import fault_support as FS
        frames = FS.load_fault_charts().fault_frames(source)
        tmag = {f['id']: float(np.linalg.norm(f['t'])) for f in frames['faults']}
        sc = case['occlusion']['scalars']
        fd = {f['id']: dict(normal=f['normal'], point=f['point'], reach=sc['fault_recess'] + sc['damage'] + tmag[f['id']],
                            half=OR.FAULT_BAND * sc['damage']) for f in planes['faults']}
        region_f, frec = OR.fault_regions(fs, cam, cpu_raster['depth'], case['occlusion']['rock'], tl.rays, fd)
        u_px = (merged['ignore_reason'] & tl.IGNORE_U) > 0
        merged['ignore_reason'][region_f & ~merged['target'] & ~u_px] |= tl.IGNORE_OWN_FAULT
        merged['valid'] = merged['valid'] & (merged['ignore_reason'] == 0)
        rec['occlusion'] = dict(frec, reach_m={k: v['reach'] for k, v in fd.items()}, band_m={k: v['half'] for k, v in fd.items()},
                                own_occluded_px=float(sum(tl._sample_weights(r)[r['own_occluded']].sum() for r in fs.values())),
                                far_occluded_px=float(sum(tl._sample_weights(r)[r['far_occluded']].sum() for r in fs.values())),
                                bit64_px=int(((merged['ignore_reason'] & tl.IGNORE_OWN_FAULT) > 0).sum()))
    w = graph_node_weights(merged['target'])
    jrows = J['per_path']
    STATE.update(source_type=st, faults=rec, seconds=time.time() - t,
                 per_type=dict(joint=dict(visible_px=float(sum(r['visible_px'] for r in jrows)),
                                          visible_u_px=float(sum(r['visible_u_px'] for r in jrows)),
                                          positive_pixels=int((st & 1).astype(bool).sum()),
                                          positive_graph_length_px=float(w[(st & 1).astype(bool)].sum())),
                               fault=dict(visible_px=float(sum(f.get('visible_px', 0.) for f in rec['faults'])),
                                          visible_u_px=float(sum(f.get('visible_u_px', 0.) for f in rec['faults'])),
                                          positive_pixels=int((st & 2).astype(bool).sum()),
                                          positive_graph_length_px=float(w[(st & 2).astype(bool)].sum())),
                               crossing_pixels=int((st == 3).sum())))
    return merged


tl.build_trace_labels = build_r4
import label_frame as LF  # noqa: E402

if __name__ == '__main__':
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else None
    try:
        LF.main()
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    rp = out / 'report.json' if out is not None else None
    if rp is not None and rp.is_file():
        rep = json.loads(rp.read_text(encoding='utf8'))
        rep['rule'] = RULE
        rep['r4_criterion_sha256'] = (HERE / 'criterion_trace_labels_r4_faults.sha256.txt').read_text(encoding='utf8').split()[0]
        if 'source_type' in STATE:
            np.save(out / 'source_type.npy', STATE['source_type'])
            rep['faults'] = STATE['faults']
            rep['per_type'] = STATE['per_type']
            rep['source_type_codes'] = {'0': 'none', '1': 'joint', '2': 'fault', '3': 'joint and fault'}
            rep.setdefault('timing_s', {})['d_faults'] = STATE['seconds']
            if STATE['faults']['status'] == 'unsupported':
                rep.setdefault('stop', {})['fault_unsupported'] = True
                code = 3 if code == 0 else code
            rep['exit_code'] = code
        rp.write_text(json.dumps(rep, indent=1, allow_nan=False, default=float) + '\n', encoding='utf8')
    sys.exit(code)
