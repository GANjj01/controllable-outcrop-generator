"""Figure 4 of the paper: view-angle sweep. Occluded length as a fraction of the in-frame length against the off-nadir angle,
each bar divided into occlusion by the fracture's own joint groove and occlusion by other rock; one panel per outcrop type
(flat, standalone) and camera azimuth (across strike, along strike).

    python reproduce/paper1/figures/fig4_view_sweep.py [--records DIR] [--out DIR]

--records  folder that holds view_sweep.json (default: ../records/round3 next to this script)
--out      output folder (default: the current folder)

Reads view_sweep.json, one row per viewpoint (the frames are listed in ../sweeps.csv, block view_sweep). Keys read from each
row: outcrop ("flat" or "standalone"), azimuth_kind ("nadir", "across strike" or "along strike"), beta_deg (off-nadir angle),
own_groove_share and other_rock_share (fractions of the in-frame length; absent or null for a viewpoint that has no frame: the
nadir view of the standalone outcrop, whose camera position lies inside the rock body). The nadir view of an outcrop is drawn
in both of its panels. Writes fig4_view_sweep.pdf and fig4_view_sweep.png and prints the plotted values.

Needs Python 3.8+ and Matplotlib (3.10.8 was used), which environment/geometry.yml and environment/labels.yml do not install
(pip install matplotlib).

Colours: orange and blue of the Okabe-Ito palette (distinguishable with colour-vision deficiency); the orange has a low
contrast against white, so every bar carries its total as a number."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = 'fig4_view_sweep'
OWN, OTHER, INK, MUTED = '#E69F00', '#0072B2', '#1a1a1a', '#5a5a5a'
BETAS = (0, 15, 30, 45, 60)
PANELS = (('flat', 'across strike', '(a) Flat outcrop, camera azimuth across strike'),
          ('flat', 'along strike', '(b) Flat outcrop, camera azimuth along strike'),
          ('standalone', 'across strike', '(c) Standalone outcrop, camera azimuth across strike'),
          ('standalone', 'along strike', '(d) Standalone outcrop, camera azimuth along strike'))


def viewpoints(record):
    """rows of the record as (outcrop, camera azimuth kind, off-nadir angle, own-groove share, other-rock share); the two shares
    are None for a viewpoint that has no frame"""
    out = []
    for r in record['rows']:
        own, other = r.get('own_groove_share'), r.get('other_rock_share')
        done = own is not None and other is not None
        out.append((r['outcrop'], r['azimuth_kind'], float(r['beta_deg']), float(own) if done else None, float(other) if done else None))
    return out


def panel_rows(rows, outcrop, direction):
    """the viewpoints of one panel in order of the off-nadir angle; the nadir view belongs to both camera azimuths of its outcrop"""
    return sorted((r[2:] for r in rows if r[0] == outcrop and r[1] in (direction, 'nadir')), key=lambda r: r[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description='Draw Figure 4 (view-angle sweep) from records/round3/view_sweep.json.')
    ap.add_argument('--records', default=str(HERE.parent / 'records' / 'round3'), help='folder that holds view_sweep.json')
    ap.add_argument('--out', default='.', help='output folder')
    a = ap.parse_args(argv)
    rows = viewpoints(json.loads((Path(a.records) / 'view_sweep.json').read_text(encoding='utf-8')))

    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=True)
    print('%-54s %5s %8s %8s %8s' % ('panel', 'beta', 'own %', 'other %', 'total %'))
    for ax, (outcrop, direction, title) in zip(axes.ravel(), PANELS):
        sel = panel_rows(rows, outcrop, direction)
        if not sel:
            raise SystemExit('view_sweep.json has no row for %s outcrop, %s' % (outcrop, direction))
        for beta, own, other in sel:
            if own is None:
                ax.text(beta, 3, 'no such viewpoint', ha='center', va='bottom', fontsize=7.5, color=MUTED, rotation=90)
                print('%-54s %5.0f %8s %8s %8s' % (title, beta, '-', '-', '-'))
                continue
            own, other = 100. * own, 100. * other
            ax.bar(beta, own, width=9.5, color=OWN, edgecolor='white', linewidth=.8)
            ax.bar(beta, other, width=9.5, bottom=own, color=OTHER, edgecolor='white', linewidth=.8)
            ax.text(beta, own + other + 1.5, '%.1f' % (own + other), ha='center', va='bottom', fontsize=7.8, color=INK)
            print('%-54s %5.0f %8.2f %8.2f %8.2f' % (title, beta, own, other, own + other))
        ax.set_title(title, fontsize=8.6, loc='left', pad=4, color=INK)
        ax.set_xticks(BETAS)
        ax.set_xlim(-8, 68)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', color='#e3e3e3', linewidth=.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
    for ax in axes[1]:
        ax.set_xlabel('Off-nadir angle \u03b2 (\u00b0)', fontsize=8.6, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel('Occluded length /\nin-frame length (%)', fontsize=8.6, color=INK)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=OTHER), plt.Rectangle((0, 0), 1, 1, color=OWN)],
               labels=['Occluded by other rock', 'Occluded by its own joint groove'], loc='upper center', ncol=2, frameon=False,
               fontsize=8.6, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .955))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / ('%s.%s' % (NAME, ext)), dpi=220, facecolor='white')
    print('written %s.pdf and %s.png in %s' % (NAME, NAME, out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
