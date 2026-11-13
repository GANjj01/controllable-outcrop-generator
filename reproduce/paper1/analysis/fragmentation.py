"""Number of connected components of the final centreline at several gap tolerances (supplement S4, table of connected
components; "19 centreline branches ... 61 connected components").

Multi-illumination aerial set (the 50 frames of block C in records/frames_index.json).
  visible centreline = target or u_line; final centreline = target.
  Gap tolerance c = 0, 0.10, 0.25 and 0.50 m: the line pixels are dilated by a disc of radius c/2 (converted to pixels with the
  nominal pixel size; pixels whose centre is within that Euclidean distance of a line pixel) and the 8-connected components are
  counted. c = 0 is the layer itself.
  Centreline branches = entries of per_path in report.json with an in-frame length (projected_in_frame_px) of at least 8 pixels.
  Reported per image, per viewpoint (median and range over its 10 images) and over the 50 images: number of components, and
  number of components / number of centreline branches.

    python reproduce/paper1/analysis/fragmentation.py --frames FRAMES
    python reproduce/paper1/analysis/fragmentation.py --selftest      checks the pure functions; reads and writes nothing

Reads FRAMES/<frame id>/q0/trace_labels/{report.json,target.npy,u_line.npy}; writes records/round3/fragmentation.json.
Needs NumPy and SciPy (the labels environment).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
_lib = Path(sys.executable).resolve().parent / 'Library' / 'bin'      # lets NumPy find the BLAS libraries in a conda environment that is not activated
if _lib.is_dir():
    os.environ['PATH'] = str(_lib) + os.pathsep + os.environ.get('PATH', '')

import argparse
import json
import re
import time

import numpy as np
import scipy
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402

TOLERANCES_M = (0., .10, .25, .50)          # gap tolerance c
MIN_BRANCH_PX = 8.                          # smallest in-frame length of a centreline branch
SMALL_COMPONENT_PX = 3                      # components smaller than this are counted separately (record only)
EIGHT = np.ones((3, 3), bool)

# viewpoint <scene seed>_<plan camera> -> (its frame with render seed 0 and the first clear-sky illumination, outcrop type)
VIEWS = {
    '26120001_c0': ('a0077', 'flat'),
    '26130002_c1': ('a0074', 'standalone'),
    '26130002_c0': ('a0062', 'standalone'),
    '26130002_c3': ('a0099', 'standalone'),
    '26130000_c1': ('a0044', 'standalone'),
}


def radius_px(c_m, pixel_mm):
    """dilation radius (pixels) of the gap tolerance c (metres): c/2 over the nominal pixel size"""
    return .5 * float(c_m) * 1000. / float(pixel_mm)


def components_after_dilation(mask, radii):
    """For every radius r (pixels): dilate the line pixels to the pixels whose centre is within the Euclidean distance r of a line
    pixel, and count the 8-connected components. r = 0 is the layer itself."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return [0 for _ in radii]
    dist = ndi.distance_transform_edt(~mask)
    return [int(ndi.label(dist <= float(r), structure=EIGHT)[1]) for r in radii]


def component_sizes(mask):
    """pixel counts of the 8-connected components (no dilation)"""
    lab, n = ndi.label(np.asarray(mask, bool), structure=EIGHT)
    return np.bincount(lab.ravel(), minlength=n + 1)[1:]


def stat(values):
    v = [float(x) for x in values if x is not None]
    if not v:
        return dict(n=0, median=None, min=None, max=None)
    return dict(n=len(v), median=float(np.median(v)), min=float(min(v)), max=float(max(v)))


def run_frame(frames, key, e):
    m = re.match(r'r2b_(c\d+)_(q\d+|oc)(_r1)?$', e['pid'])
    if not m:
        raise ValueError('%s: unexpected point name %s' % (key, e['pid']))
    view = e['sid'].split('_')[1] + '_' + m.group(1)
    if view not in VIEWS:
        raise ValueError('%s: viewpoint %s is not one of the 5 viewpoints of the multi-illumination aerial set' % (key, view))
    d = fio.label_dir(frames, key)
    rep = fio.read_json(d / 'report.json')
    target = np.load(d / 'target.npy')
    u_line = np.load(d / 'u_line.npy')
    if target.dtype != bool or u_line.dtype != bool or target.shape != u_line.shape:
        raise ValueError('%s: unexpected type or shape of target / u_line' % key)
    visible = target | u_line
    pp = rep['per_path']
    n_in8 = int(sum(p['projected_in_frame_px'] >= MIN_BRANCH_PX for p in pp))
    n_vis8 = int(sum(p['visible_px'] >= MIN_BRANCH_PX for p in pp))
    pixel_mm = float(e['pixel_mm'])
    radii = [radius_px(c, pixel_mm) for c in TOLERANCES_M]
    n_vis = components_after_dilation(visible, radii)
    n_fin = components_after_dilation(target, radii)
    sz_vis, sz_fin = component_sizes(visible), component_sizes(target)
    fl = rep.get('faults') or {}
    row = dict(
        key=key, view=view, pid=e['pid'], light=m.group(2), render_seed=1 if m.group(3) else 0, pixel_mm=pixel_mm,
        per_path_rows=len(pp), true_branches=n_in8, branches_visible_ge_8px=n_vis8,
        in_frame_px=float(sum(p['projected_in_frame_px'] for p in pp)), visible_px=float(sum(p['visible_px'] for p in pp)),
        target_px=int(target.sum()), u_line_px=int(u_line.sum()), visible_centreline_px=int(visible.sum()),
        target_u_overlap_px=int((target & u_line).sum()),
        report_positive_pixels=int(rep['positive_pixels']),
        faults_status=fl.get('status'), fault_rows=len(fl.get('faults') or []),
        fault_visible_px=float(((rep.get('per_type') or {}).get('fault') or {}).get('visible_px') or 0.),
        tolerances=[dict(c_m=float(c), radius_px=float(r),
                         components_visible=int(a), components_final=int(b),
                         ratio_visible=a / n_in8 if n_in8 else None, ratio_final=b / n_in8 if n_in8 else None,
                         # record only: the same ratios over the number of branches with a visible length of at least 8 px
                         ratio_visible_over_visible_branches=a / n_vis8 if n_vis8 else None,
                         ratio_final_over_visible_branches=b / n_vis8 if n_vis8 else None)
                    for c, r, a, b in zip(TOLERANCES_M, radii, n_vis, n_fin)],
        # record only: sizes of the components without dilation
        c0_components=dict(
            visible=dict(count=int(len(sz_vis)), below_3px=int((sz_vis < SMALL_COMPONENT_PX).sum()),
                         median_px=float(np.median(sz_vis)) if len(sz_vis) else None,
                         max_px=int(sz_vis.max()) if len(sz_vis) else None),
            final=dict(count=int(len(sz_fin)), below_3px=int((sz_fin < SMALL_COMPONENT_PX).sum()),
                       median_px=float(np.median(sz_fin)) if len(sz_fin) else None,
                       max_px=int(sz_fin.max()) if len(sz_fin) else None)))
    return row


def summarise(rows):
    """median and range, at every tolerance, of the number of components and of components / centreline branches, for both lines"""
    out = []
    for i, c in enumerate(TOLERANCES_M):
        t = [r['tolerances'][i] for r in rows]
        out.append(dict(
            c_m=float(c), radius_px=stat(x['radius_px'] for x in t),
            ratio_visible=stat(x['ratio_visible'] for x in t), ratio_final=stat(x['ratio_final'] for x in t),
            components_visible=stat(x['components_visible'] for x in t), components_final=stat(x['components_final'] for x in t),
            supplementary_ratio_visible_over_visible_branches=stat(x['ratio_visible_over_visible_branches'] for x in t),
            supplementary_ratio_final_over_visible_branches=stat(x['ratio_final_over_visible_branches'] for x in t)))
    return out


NOTES = [
    'At c = 0 the gaps of the visible centreline (target or u_line) are not all places where the trace cannot be seen: the label program '
    'removes from target the centreline pixels within 2 px of an unclear (U) line, within 2 px of a transition between visible and '
    'occluded parts and inside the other marked areas, and u_line returns only the pixels of the unclear windows themselves, so that a '
    'few pixels stay empty between the two layers. From c = 0.10 m (dilation radius of 6 px or more) these gaps are bridged.',
    'Dilation also joins neighbouring traces, so a ratio of components to branches below 1 at a large tolerance does not mean that the '
    'traces have no gaps.',
    'The 10 frames of a viewpoint differ only in illumination and render seed: the number of centreline branches is the same, target and '
    'u_line change with the brightness check.',
    'At the standalone-outcrop viewpoints the pixel size is not uniform within the frame; the tolerance in metres holds near the target point.',
]


def selftest():
    # three collinear horizontal segments with gaps of 5 and 13 empty pixels, and a parallel line 30 pixels away
    m = np.zeros((80, 120), bool)
    m[20, 10:30] = True
    m[20, 35:60] = True
    m[20, 73:100] = True
    m[50, 10:100] = True
    n = components_after_dilation(m, [0., 2., 3., 7., 15.])
    # r = 0: 4; r = 2: 29+2 = 31 and 35-2 = 33 leave pixel 32 empty: 4; r = 3: they meet: 3; r = 7: 59+7 = 66 = 73-7: 2;
    # r = 15: the two lines 30 px apart meet (20+15 = 35 = 50-15): 1
    assert n == [4, 4, 3, 2, 1], n
    assert components_after_dilation(np.zeros((5, 5), bool), [0., 3.]) == [0, 0]
    assert abs(radius_px(.10, 5.) - 10.) < 1e-12 and radius_px(0., 5.) == 0.
    s = component_sizes(m)
    assert sorted(s.tolist()) == [20, 25, 27, 90], s
    d = np.zeros((6, 6), bool); d[1, 1] = d[2, 2] = True; d[4, 4] = True      # diagonal neighbours are connected
    assert components_after_dilation(d, [0.]) == [2]
    st = stat([1., 3., 2.])
    assert st == dict(n=3, median=2., min=1., max=3.), st
    print('selftest PASS')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', help='folder with one sub-folder per frame (a0044 ...)')
    ap.add_argument('--index', default=str(fio.INDEX))
    ap.add_argument('--out', default=str(fio.RECORDS / 'fragmentation.json'))
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if not args.frames:
        ap.error('--frames is required')
    t0 = time.time()
    frames = fio.index(args.index)
    rows = []
    for key in sorted(frames):
        if frames[key].get('block') != 'C':
            continue
        rows.append(run_frame(args.frames, key, frames[key]))
        r = rows[-1]
        print('%s %s %-13s branches %2d | components of the visible centreline %s | of the final centreline %s' % (
            key, r['view'], r['pid'], r['true_branches'],
            [t['components_visible'] for t in r['tolerances']], [t['components_final'] for t in r['tolerances']]), flush=True)

    # input checks (record only)
    per_view_n = {v: sum(r['view'] == v for r in rows) for v in VIEWS}
    branch_const = {v: sorted({r['true_branches'] for r in rows if r['view'] == v}) for v in VIEWS}
    checks = dict(
        frames=len(rows), frames_expected=50, frames_ok=bool(len(rows) == 50),
        frames_per_view=per_view_n, ten_per_view=bool(all(n == 10 for n in per_view_n.values())),
        frames_with_fault_traces=int(sum(r['fault_rows'] > 0 or r['fault_visible_px'] > 0 for r in rows)),
        frames_with_target_u_overlap=int(sum(r['target_u_overlap_px'] > 0 for r in rows)),
        frames_target_px_equals_report_positive_pixels=int(sum(r['target_px'] == r['report_positive_pixels'] for r in rows)),
        frames_without_true_branch=int(sum(r['true_branches'] == 0 for r in rows)),
        true_branches_by_view=branch_const,
        true_branches_constant_within_view=bool(all(len(v) == 1 for v in branch_const.values())))
    checks['all_ok'] = bool(checks['frames_ok'] and checks['ten_per_view'] and checks['frames_with_fault_traces'] == 0
                            and checks['frames_with_target_u_overlap'] == 0
                            and checks['frames_target_px_equals_report_positive_pixels'] == len(rows)
                            and checks['frames_without_true_branch'] == 0 and checks['true_branches_constant_within_view'])

    all50 = summarise(rows)
    by_view = {}
    for v, (rep_frame, kind) in VIEWS.items():
        vr = [r for r in rows if r['view'] == v]
        by_view[v] = dict(representative_frame=rep_frame, outcrop=kind, frames=[r['key'] for r in vr],
                          pixel_mm=vr[0]['pixel_mm'], per_path_rows=vr[0]['per_path_rows'],
                          true_branches=vr[0]['true_branches'],
                          branches_visible_ge_8px=sorted({r['branches_visible_ge_8px'] for r in vr}),
                          in_frame_px=vr[0]['in_frame_px'], visible_px=vr[0]['visible_px'],
                          target_px=stat(r['target_px'] for r in vr), u_line_px=stat(r['u_line_px'] for r in vr),
                          c0_small_components_final=stat(r['c0_components']['final']['below_3px'] for r in vr),
                          c0_small_components_visible=stat(r['c0_components']['visible']['below_3px'] for r in vr),
                          tolerances=summarise(vr))
    standalone = [r for r in rows if VIEWS[r['view']][1] == 'standalone']
    flat = [r for r in rows if VIEWS[r['view']][1] == 'flat']
    by_outcrop = {'flat': dict(views=[v for v in VIEWS if VIEWS[v][1] == 'flat'], frames=len(flat), tolerances=summarise(flat)),
                  'standalone': dict(views=[v for v in VIEWS if VIEWS[v][1] == 'standalone'], frames=len(standalone),
                                     tolerances=summarise(standalone))}
    print('input checks:', json.dumps({k: v for k, v in checks.items() if k not in ('frames_per_view', 'true_branches_by_view')}))

    out = dict(
        description='Number of 8-connected components of the final centreline (target) and of the visible centreline (target or u_line) after '
                    'dilation by half the gap tolerance c, and their ratio to the number of centreline branches, for the 50 images of the '
                    'multi-illumination aerial set: per image (frames), per viewpoint (by_view: median and range over its 10 images), per '
                    'outcrop type and over all 50 images.',
        paper_items=['supplement S4: table of the number of connected components of the final centreline',
                     'supplement S4, text: 19 centreline branches at the flat-outcrop viewpoint, median of 61 components at a gap tolerance of 0'],
        script='reproduce/paper1/analysis/fragmentation.py',
        units=dict(c_m='gap tolerance, metres', radius_px='dilation radius = (c/2) x 1000 / pixel_mm, pixels', pixel_mm='nominal pixel size, mm',
                   true_branches='entries of per_path with projected_in_frame_px >= 8', ratio_final='components_final / true_branches',
                   ratio_visible='components_visible / true_branches'),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        definitions=dict(
            visible_centreline='target or u_line', final_centreline='target',
            tolerances_m=list(TOLERANCES_M), dilation='Euclidean disc of radius r = (c/2) / nominal pixel size (pixels): distance transform <= r',
            connectivity='8-connected', true_branches='entries of per_path with projected_in_frame_px >= 8',
            reported='number of components, and number of components / number of centreline branches'),
        checks=checks, all_50_frames=all50, by_view=by_view, by_outcrop=by_outcrop, notes=NOTES, frames=rows)
    fio.write_json(args.out, out)
    print('wrote %s  %.1f s' % (Path(args.out).name, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())

