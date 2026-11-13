"""Sensitivity of the fraction of unclear (U) windows to the brightness-difference threshold (main text Section 4.2 and
Table 2, row "Brightness check"; main text Section 5.3; supplement S4, sensitivity table). The manuscript calls the unclear
windows and segments "unconfirmed".

Multi-illumination aerial set (the 50 frames of block C in records/frames_index.json), counted by window. A window is valid when
band_pixels > 0 and ring_pixels >= 4 (otherwise its sampling is insufficient); a window is unclear at threshold t when it is not
valid, or valid with a brightness difference below t grey levels. The labels use t = 2.

    python reproduce/paper1/analysis/u_threshold.py --frames FRAMES

Reads FRAMES/<frame id>/q0/trace_labels/windows.json of the 50 frames; writes records/round3/u_threshold.json.
Standard library only."""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames_io as fio  # noqa: E402

TS = (1., 2., 4.)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', required=True, help='folder with one sub-folder per frame (a0044 ...)')
    ap.add_argument('--index', default=str(fio.INDEX))
    ap.add_argument('--out', default=str(fio.RECORDS / 'u_threshold.json'))
    args = ap.parse_args(argv)
    idx = fio.index(args.index)
    frames = []
    for key, v in sorted(idx.items()):
        if v.get('block') != 'C':
            continue
        ws = fio.read_json(fio.label_dir(args.frames, key) / 'windows.json')
        valid = [(w['band_pixels'] > 0) and (w['ring_pixels'] >= 4) for w in ws]
        n = len(ws)
        struct = sum(1 for ok in valid if not ok)
        rec = dict(key=key, view=fio.view_of(v), n=n, sampling_insufficient=struct)
        for t in TS:
            low = sum(1 for w, ok in zip(ws, valid) if ok and w['contrast'] < t)
            rec['unclear_share_t%g' % t] = (struct + low) / n
            rec['low_contrast_share_t%g' % t] = low / n
            rec['low_contrast_windows_t%g' % t] = low
        assert sum(1 for w in ws if w['u']) == struct + sum(1 for w, ok in zip(ws, valid) if ok and w['contrast'] < 2.), key
        frames.append(rec)
    assert len(frames) == 50
    out = dict(
        description='Fraction of unclear (U) windows per image at brightness-difference thresholds of 1, 2 and 4 grey levels, the 50 images of '
                    'the multi-illumination aerial set, counted by window. unclear = windows with insufficient sampling + valid windows with a '
                    'brightness difference below the threshold; low_contrast = the second part only.',
        paper_items=['main text Section 4.2 (unclear windows per image at 1, 2 and 4 levels; 1025 unclear windows, 59.5% with insufficient '
                     'sampling; windows with low brightness difference 0.6%-10.7%)', 'main text Table 2, row "Brightness check"',
                     'main text Section 5.3 (near-nadir viewpoint 26120001_c0: insufficient sampling 6.0%, low brightness difference 0.6%-9.9%)',
                     'supplement S4: sensitivity to the brightness-difference threshold'],
        script='reproduce/paper1/analysis/u_threshold.py',
        units=dict(n='windows in the image', sampling_insufficient='windows with band_pixels = 0 or ring_pixels < 4',
                   unclear_share_tT='(sampling_insufficient + low_contrast_windows_tT) / n', low_contrast_share_tT='low_contrast_windows_tT / n',
                   threshold='grey levels of the 8-bit image (Rec. 709 luma)'),
        frames=frames, summary={})
    for t in TS:
        for kind in ('unclear_share', 'low_contrast_share'):
            xs = [f['%s_t%g' % (kind, t)] for f in frames]
            out['summary']['%s_t%g' % (kind, t)] = dict(min=min(xs), median=statistics.median(xs), max=max(xs))
    out['sampling_insufficient_share_of_windows'] = dict(min=min(f['sampling_insufficient'] / f['n'] for f in frames),
                                                         max=max(f['sampling_insufficient'] / f['n'] for f in frames))
    u2 = sum(f['sampling_insufficient'] + f['low_contrast_windows_t2'] for f in frames)
    out['unclear_windows_t2'] = dict(windows=sum(f['n'] for f in frames), unclear=u2,
                                     sampling_insufficient=sum(f['sampling_insufficient'] for f in frames),
                                     sampling_insufficient_share_of_unclear=sum(f['sampling_insufficient'] for f in frames) / u2)
    fio.write_json(args.out, out)
    for k, v in out['summary'].items():
        print('%-24s min %.4f  median %.4f  max %.4f' % (k, v['min'], v['median'], v['max']))
    print('insufficient sampling, share of windows per image: %.4f-%.4f' % tuple(out['sampling_insufficient_share_of_windows'].values()))
    print('unclear windows at 2 levels: %(unclear)d of %(windows)d, of which insufficient sampling %(sampling_insufficient)d' % out['unclear_windows_t2'])


if __name__ == '__main__':
    main()
