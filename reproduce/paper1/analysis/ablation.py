"""Role of each step: rows "Occlusion test", "Removed-volume check" and "Position refinement" of main-text Table 2 (supplement S4,
role of each step), and the trace length retained in the labels at each viewpoint (main text Section 5.3, second paragraph).

Multi-illumination aerial set (the 50 frames of block C in records/frames_index.json); everything is read from the report of the
label program, report.json.

Occlusion test: occluded length / in-frame length, both summed over per_path (occluded_px, projected_in_frame_px); the length
that the geometric intersection lines would add to the labels if they were used directly. Retained length: visible length /
in-frame length, unclear (U) length / visible length (visible_u_px / visible_px) and their combination
(visible - unclear) / in-frame, which does not subtract the 2 px that the label layers also remove at both ends of an
unclear segment and where occluded and visible parts alternate.

Removed-volume check: exemption.exempt_with_occluder_length_px / visible_length_px. Of the trace length that is visible on the
cut mesh, this is the part for which rock lay in front of the sample point before cutting and all of that rock is inside the
volume removed for this fracture: the part of the visible length that would be classed as occluded if the occlusion test were
run on the uncut rock surface. (The label program tests occlusion on the cut mesh; the check itself changes no label.)

Position refinement: refinement.shift_m_p50_p99_max / nominal pixel size. The 3-D distance by which the refinement moves a sample
point along the rock surface onto the exact mid-surface of the fracture (median, 99th percentile and maximum over all sample
points of the frame, in-frame or not), expressed in nominal pixels.

    python reproduce/paper1/analysis/ablation.py --frames FRAMES

Reads FRAMES/<frame id>/q0/trace_labels/report.json of the 50 frames; writes records/round3/ablation.json.
Needs Python 3.8+ and NumPy."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402

AERIAL_VIEWS, AERIAL_KIND = fio.AERIAL_VIEWS, fio.AERIAL_OUTCROP


def rng(x):
    x = [float(v) for v in x]
    return dict(n=len(x), min=min(x), median=float(np.median(x)), max=max(x))


def frame_row(frames, key, v):
    rep = fio.read_json(fio.label_dir(frames, key) / 'report.json')
    ex, ref = rep['exemption'], rep['refinement']
    vis = float(rep['visible_length_px'])
    pm = float(v['pixel_mm'])
    shift = [float(x) for x in ref['shift_m_p50_p99_max']]
    pp = rep['per_path']
    in_px, occ_px = float(sum(p['projected_in_frame_px'] for p in pp)), float(sum(p['occluded_px'] for p in pp))
    vis_px, u_px = float(sum(p['visible_px'] for p in pp)), float(sum(p['visible_u_px'] for p in pp))
    return dict(
        key=key, sid=v['sid'], pid=v['pid'], pixel_mm=pm,
        projected_in_frame_px=in_px, occluded_px=occ_px, visible_px=vis_px, visible_u_px=u_px,
        occluded_share=occ_px / in_px if in_px > 0 else None, visible_share=vis_px / in_px if in_px > 0 else None,
        unclear_share_of_visible=u_px / vis_px if vis_px > 0 else None,
        visible_and_clear_share=(vis_px - u_px) / in_px if in_px > 0 else None,
        visible_length_px=vis, exemption_visible_length_px=float(ex['visible_length_px']),
        exempt_with_occluder_length_px=float(ex['exempt_with_occluder_length_px']),
        occluded_on_uncut_surface_share=float(ex['exempt_with_occluder_length_px']) / vis if vis > 0 else None,
        shift_m_p50_p99_max=shift, shift_px_p50_p99_max=[1000. * s / pm for s in shift],
        refinement_points=int(ref['points']), exit_code=rep.get('exit_code'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', required=True, help='folder with one sub-folder per frame (a0044 ...)')
    ap.add_argument('--index', default=str(fio.INDEX))
    ap.add_argument('--out', default=str(fio.RECORDS / 'ablation.json'))
    args = ap.parse_args(argv)
    index = fio.index(args.index)
    frames = []
    for key in fio.aerial_frames(index):
        row = frame_row(args.frames, key, index[key])
        row['view'] = fio.view_of(index[key])
        frames.append(row)
    views = sorted({f['view'] for f in frames}, key=AERIAL_VIEWS.index)
    if views != AERIAL_VIEWS:
        raise SystemExit('unexpected viewpoints: %s' % views)

    # occlusion test and retained length, per viewpoint
    occlusion, retained = {}, {}
    for vw in AERIAL_VIEWS:
        fs = [f for f in frames if f['view'] == vw]
        x = [f['occluded_share'] for f in fs]
        occlusion[vw] = dict(outcrop=AERIAL_KIND[vw], frames=len(fs), occluded_share=float(np.median(x)), spread=max(x) - min(x))
        retained[vw] = dict(outcrop=AERIAL_KIND[vw], frames=len(fs), visible_share=float(np.median([f['visible_share'] for f in fs])),
                            unclear_share_of_visible=rng([f['unclear_share_of_visible'] for f in fs]),
                            visible_and_clear_share=rng([f['visible_and_clear_share'] for f in fs]))
    iso4 = [occlusion[vw]['occluded_share'] for vw in AERIAL_VIEWS if AERIAL_KIND[vw] == 'standalone']
    occlusion_test = dict(by_view=occlusion, flat_view=occlusion['26120001_c0']['occluded_share'], standalone_views_range=[min(iso4), max(iso4)],
                          within_view_spread_max=max(v['spread'] for v in occlusion.values()))
    retained_length = dict(by_view=retained, unclear_share_of_visible_50_frames=rng([f['unclear_share_of_visible'] for f in frames]))

    # removed-volume check
    r5 = [f['occluded_on_uncut_surface_share'] for f in frames]
    by_view5 = {}
    for vw in AERIAL_VIEWS:
        x = [f['occluded_on_uncut_surface_share'] for f in frames if f['view'] == vw]
        by_view5[vw] = dict(outcrop=AERIAL_KIND[vw], frames=len(x), value=float(np.median(x)), min=min(x), max=max(x), spread=max(x) - min(x))
    iso5 = [by_view5[vw]['value'] for vw in AERIAL_VIEWS if AERIAL_KIND[vw] == 'standalone']
    removed = dict(frames_50=rng(r5), by_view=by_view5, standalone_views_range=[min(iso5), max(iso5)], flat_view=by_view5['26120001_c0']['value'])

    # position refinement
    names = ['p50', 'p99', 'max']
    r6 = {n: [f['shift_px_p50_p99_max'][i] for f in frames] for i, n in enumerate(names)}
    r6m = {n: [1000. * f['shift_m_p50_p99_max'][i] for f in frames] for i, n in enumerate(names)}
    by_view6 = {}
    for vw in AERIAL_VIEWS:
        fs = [f for f in frames if f['view'] == vw]
        by_view6[vw] = dict(
            outcrop=AERIAL_KIND[vw], frames=len(fs), pixel_mm=fs[0]['pixel_mm'],
            shift_px={n: float(np.median([f['shift_px_p50_p99_max'][i] for f in fs])) for i, n in enumerate(names)},
            shift_mm={n: float(np.median([1000. * f['shift_m_p50_p99_max'][i] for f in fs])) for i, n in enumerate(names)})
    refinement = dict(frames_50_px={n: rng(r6[n]) for n in names}, frames_50_mm={n: rng(r6m[n]) for n in names}, by_view=by_view6)

    checks = dict(
        frames_exemption_visible_equals_top_level=int(sum(abs(f['exemption_visible_length_px'] - f['visible_length_px'])
                                                          <= 1e-9 * max(1., f['visible_length_px']) for f in frames)),
        share_within_0_1=bool(all(0. <= x <= 1. for x in r5)), within_view_spread_of_share_max=max(v['spread'] for v in by_view5.values()),
        frames_exit_code_0=int(sum(f['exit_code'] == 0 for f in frames)))
    out = dict(
        description='Rows "Occlusion test", "Removed-volume check" and "Position refinement" of main-text Table 2 and the trace length retained in '
                    'the labels, from the label reports of the 50 images of the multi-illumination aerial set (frames: one row per image). '
                    'occluded_share: occluded length / in-frame length. occluded_on_uncut_surface_share: visible length that would be '
                    'classed as occluded if the occlusion test were run on the uncut rock surface, as a fraction of the visible length. '
                    'shift_px_p50_p99_max: 3-D displacement of the sample points by the position refinement (median, 99th percentile, maximum '
                    'over the sample points of a frame) divided by the nominal pixel size.',
        paper_items=['main text Table 2, rows "Occlusion test", "Removed-volume check" and "Position refinement"',
                     'main text Section 4.2, first paragraph, and the unclear length per image (2.6%-15.7% of the visible length)',
                     'main text Section 5.3, second paragraph (near-nadir viewpoint 26120001_c0: visible 99.6%, visible and not unclear 84%-93%)',
                     'supplement S4: role of each step and the mask skeleton, first paragraph'],
        script='reproduce/paper1/analysis/ablation.py',
        units=dict(occluded_share='occluded_px / projected_in_frame_px (sums over per_path; image pixels of trace length)',
                   visible_share='visible_px / projected_in_frame_px', unclear_share_of_visible='visible_u_px / visible_px',
                   visible_and_clear_share='(visible_px - visible_u_px) / projected_in_frame_px',
                   occluded_on_uncut_surface_share='exempt_with_occluder_length_px / visible_length_px (image pixels of trace length)',
                   shift_m_p50_p99_max='metres', shift_px_p50_p99_max='nominal pixels = 1000 x shift_m / pixel_mm', pixel_mm='nominal pixel size, mm'),
        notes=['The denominator of occluded_on_uncut_surface_share is the length that is visible on the cut mesh, not the in-frame length.',
               'The percentiles of the displacement weight all sample points equally, including points outside the frame and occluded points; '
               'the displacement is a 3-D distance converted to nominal pixels, not an image displacement.',
               'The occluded share, the removed-volume share and the displacement depend only on geometry and viewpoint: the 10 frames of a '
               'viewpoint (5 illumination conditions x 2 render seeds) have the same value. The unclear length changes with illumination '
               'and render seed.'],
        occlusion_test=occlusion_test, removed_volume_check=removed, position_refinement=refinement, retained_length=retained_length,
        checks=checks, frames=frames)
    fio.write_json(args.out, out)
    print('occlusion test, occluded / in-frame length by viewpoint: %s' % {k: round(100 * v['occluded_share'], 2) for k, v in occlusion.items()})
    print('retained length by viewpoint (visible; visible and not unclear min-max): %s' % {
        k: (round(100 * v['visible_share'], 2), round(100 * v['visible_and_clear_share']['min'], 1), round(100 * v['visible_and_clear_share']['max'], 1))
        for k, v in retained.items()})
    print('removed-volume check, 50 frames: %.1f%%-%.1f%% (median %.1f%%); by viewpoint: %s' % (
        100 * min(r5), 100 * max(r5), 100 * float(np.median(r5)), {k: round(100 * v['value'], 2) for k, v in by_view5.items()}))
    print('position refinement, 50 frames (nominal px): median %.4f-%.4f, 99th percentile %.3f-%.3f, maximum %.2f-%.2f' % (
        min(r6['p50']), max(r6['p50']), min(r6['p99']), max(r6['p99']), min(r6['max']), max(r6['max'])))


if __name__ == '__main__':
    main()
