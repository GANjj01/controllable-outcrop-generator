"""Figure 6 of the paper: illumination sweep at the flat-outcrop viewpoint (72 sun directions per case; the frames are listed
in ../sweeps.csv, block light_sweep, viewpoint V1).
  (a) Brightness difference between the adjacent-rock ring and the in-fracture band of the contrast windows (median over the
      windows, grey levels) against the shadow ratio S, for the baseline case (mean opening 5.5 cm, depth 0.175 m) and the
      other 4 cases (mean opening 3 or 8 cm, or depth 0.10 or 0.25 m).
  (b) Fraction of valid windows with a brightness difference below 2 grey levels, the threshold of the unclear (U) test, in
      each S bin.
  (c) The same fraction in the baseline case against the angle between the sun azimuth and the trace strike, for the three
      sun elevations.

    python reproduce/paper1/figures/fig6_illumination_sweep.py [--records DIR] [--out DIR]

--records  folder that holds light_sweep.json (default: ../records/round3 next to this script)
--out      output folder (default: the current folder)

S is the reach of the sunward wall's shadow at floor depth divided by the opening, for a straight fracture with parallel
walls: S = (depth / opening) |s.n| / (s.u), with s the unit vector towards the sun, n the normal of the fracture plane and u
the unit vector in that plane perpendicular to the trace and pointing out of the rock; S is infinite when s.u <= 0, and the
last bin (S >= 2) includes those windows.

Keys read from light_sweep.json:
  cases.<case>.S.bins             the six S bins of a case (0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1, 1-2, >= 2), each with
                                  median_darkness (ring minus band, grey levels), low_contrast_share and windows
  cases.ls_v1b055.rule_phi.bins   per sun elevation ("20", "40", "60"), the six 15-degree bins of the angle between sun
                                  azimuth and strike, each with low_contrast_share and windows
The cases are ls_v1b055 (baseline), ls_v1w030, ls_v1w080, ls_v1d010 and ls_v1d025, named as in ../sweeps.csv.
Writes fig6_illumination_sweep.pdf and fig6_illumination_sweep.png and prints the plotted values.

Needs Python 3.8+ and Matplotlib (3.10.8 was used), which environment/geometry.yml and environment/labels.yml do not install
(pip install matplotlib).

Colours: the baseline case blue, the other four cases neutral grey (their values are printed, and listed in the supplement);
the three sun elevations are one blue hue from light to dark."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = 'fig6_illumination_sweep'
INK, MUTED, BASE, GREY = '#1a1a1a', '#5a5a5a', '#0072B2', '#9a9a9a'
ELEVATIONS = (('20', '#9ecae1'), ('40', '#4292c6'), ('60', '#08519c'))
BASELINE = 'ls_v1b055'
CASES = (BASELINE, 'ls_v1w030', 'ls_v1w080', 'ls_v1d010', 'ls_v1d025')      # the five cases of the flat-outcrop viewpoint (V1)
S_BINS = ('0\u20130.25', '0.25\u20130.5', '0.5\u20130.75', '0.75\u20131', '1\u20132', '\u22652')
PHI_BINS = ('0\u201315', '15\u201330', '30\u201345', '45\u201360', '60\u201375', '75\u201390')
Y_DIFF = 'Brightness difference, adjacent\nrock minus fracture (grey levels)'
Y_LOW = 'Valid windows with brightness\ndifference < 2 grey levels (%)'


def s_bins(record, case):
    """the six S bins of a case: (median brightness difference in grey levels, fraction of low-difference windows, windows)"""
    bins = record['cases'][case]['S']['bins']
    if len(bins) != len(S_BINS):
        raise SystemExit('light_sweep.json: expected %d S bins for %s, found %d' % (len(S_BINS), case, len(bins)))
    return [(float(b['median_darkness']), float(b['low_contrast_share']), int(b['windows'])) for b in bins]


def phi_bins(record, elevation):
    """the six bins of the angle between sun azimuth and strike, baseline case: (fraction of low-difference windows, windows)"""
    bins = record['cases'][BASELINE]['rule_phi']['bins'][elevation]
    if len(bins) != len(PHI_BINS):
        raise SystemExit('light_sweep.json: expected %d bins of the angle to the strike at elevation %s, found %d'
                         % (len(PHI_BINS), elevation, len(bins)))
    return [(float(b['low_contrast_share']), int(b['windows'])) for b in bins]


def main(argv=None):
    ap = argparse.ArgumentParser(description='Draw Figure 6 (illumination sweep) from records/round3/light_sweep.json.')
    ap.add_argument('--records', default=str(HERE.parent / 'records' / 'round3'), help='folder that holds light_sweep.json')
    ap.add_argument('--out', default='.', help='output folder')
    a = ap.parse_args(argv)
    record = json.loads((Path(a.records) / 'light_sweep.json').read_text(encoding='utf-8'))

    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    data = {c: s_bins(record, c) for c in CASES}
    panels = ((axes[0], 0, 1., Y_DIFF, '(a) Brightness difference against S', 'lower left'),
              (axes[1], 1, 100., Y_LOW, '(b) Windows with low brightness difference', 'upper left'))
    print('S bins: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1, 1-2, >= 2')
    for ax, k, scale, ylabel, title, legend in panels:
        print(title + (' (grey levels)' if k == 0 else ' (%)'))
        for c in CASES:
            y = [b[k] * scale for b in data[c]]
            base = c == BASELINE
            ax.plot(range(6), y, '-o', color=BASE if base else GREY, linewidth=2 if base else 1, markersize=4.5 if base else 3,
                    zorder=3 if base else 2, label='Baseline case' if base else ('Other 4 cases' if c == CASES[1] else None))
            print('  %-10s ' % c + ' '.join('%7.2f' % v for v in y) + '   windows ' + ' '.join('%d' % b[2] for b in data[c]))
        ax.set_xticks(range(6))
        ax.set_xticklabels(S_BINS, fontsize=6.6, rotation=30)
        ax.set_xlabel('Shadow ratio S', fontsize=8.4, color=INK)
        ax.set_ylabel(ylabel, fontsize=8.4, color=INK)
        ax.set_title(title, fontsize=8.0, loc='left', pad=4, color=INK)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, fontsize=7.6, loc=legend)
    ax = axes[2]
    print('(c) Windows with low brightness difference (%) against the angle between sun azimuth and strike, baseline case, bins of 15 degrees')
    for el, col in ELEVATIONS:
        bins = phi_bins(record, el)
        y = [100. * b[0] for b in bins]
        ax.plot(range(6), y, '-o', color=col, linewidth=1.8, markersize=4, label='Sun elevation %s\u00b0' % el)
        print('  elevation %s ' % el + ' '.join('%7.2f' % v for v in y) + '   windows ' + ' '.join('%d' % b[1] for b in bins))
    ax.set_xticks(range(6))
    ax.set_xticklabels(PHI_BINS, fontsize=6.6, rotation=30)
    ax.set_xlabel('Sun azimuth relative to strike, \u03c6 (\u00b0)', fontsize=8.4, color=INK)
    ax.set_ylabel(Y_LOW, fontsize=8.4, color=INK)
    ax.set_title('(c) Variation with \u03c6 (baseline case)', fontsize=8.0, loc='left', pad=4, color=INK)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7.6, loc='lower left')
    for ax in axes:
        ax.grid(axis='y', color='#e3e3e3', linewidth=.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.6, colors=MUTED)
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / ('%s.%s' % (NAME, ext)), dpi=220, facecolor='white')
    print('written %s.pdf and %s.png in %s' % (NAME, NAME, out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
