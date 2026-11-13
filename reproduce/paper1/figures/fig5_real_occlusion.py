"""Figure 5 of the paper: fraction of trace pixels occluded on the digital elevation models (DEMs) of real outcrops against the
off-nadir angle, for lines of sight across and along the local trace strike, next to the view-angle sweep of the synthetic flat
outcrop.

    python reproduce/paper1/figures/fig5_real_occlusion.py [--records DIR] [--out DIR]

--records  folder that holds fraxet_sites.json and view_sweep.json (default: ../records/round3 next to this script)
--out      output folder (default: the current folder)

The real outcrops are the sites OG1, KB3 and EN of the FraXet dataset. fraxet_sites.json holds only statistics derived from
the dataset; its references, licence and DOI are given under the key "dataset" of that record. Keys read:
  fraxet_sites.json  parameters.clearance_main_m (clearance height of the line of sight used in the figure, 1.0 m);
                     sites.<site>.occlusion.by_pixel_mm.<pixel size>: rows with beta_deg, clearance_m and, under "across" and
                     "along", occluded_share (undetermined sample points left out of the denominator) and
                     occluded_share_undetermined_as_visible (undetermined sample points counted as not occluded)
  view_sweep.json    rows with outcrop, azimuth_kind, beta_deg, occluded_share and pixel_mm (see fig4_view_sweep.py)
A sample point is undetermined when its line of sight leaves the valid DEM before it reaches the clearance height. The shaded
areas span the values obtained with the pixel size at either end of the published range (5 and 6 mm at the two Finnish sites,
29 and 32 mm at EN) and with the two treatments of the undetermined sample points; the solid lines mark the midpoints.
Writes fig5_real_occlusion.pdf and fig5_real_occlusion.png and prints the plotted values.

Needs Python 3.8+ and Matplotlib (3.10.8 was used), which environment/geometry.yml and environment/labels.yml do not install
(pip install matplotlib).

Colours: the synthetic series black and dashed, the three sites blue, orange and green of the Okabe-Ito palette."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = 'fig5_real_occlusion'
INK, MUTED = '#1a1a1a', '#5a5a5a'
SITES = (('OG1', '#0072B2', 'OG1 (Finland, 5\u20136 mm/px)'), ('KB3', '#E69F00', 'KB3 (Finland, 5\u20136 mm/px)'),
         ('EN', '#009E73', 'EN (Australia, 29\u201332 mm/px)'))
PANELS = (('across', 'across strike', '(a) Line of sight across the trace strike'),
          ('along', 'along strike', '(b) Line of sight along the trace strike'))
SHARES = ('occluded_share', 'occluded_share_undetermined_as_visible')


def synthetic(record, direction):
    """view-angle sweep of the synthetic flat outcrop: [(off-nadir angle, occluded fraction)], nadir view included"""
    return sorted((float(r['beta_deg']), float(r['occluded_share'])) for r in record['rows']
                  if r['outcrop'] == 'flat' and r['azimuth_kind'] in (direction, 'nadir') and r.get('occluded_share') is not None)


def site_band(record, site, direction):
    """[(off-nadir angle, lowest, highest occluded fraction)] over the pixel sizes and the two treatments of the undetermined
    sample points, at the main clearance height"""
    clearance = float(record['parameters']['clearance_main_m'])
    by_pixel = record['sites'][site]['occlusion']['by_pixel_mm']
    values = {}
    for rows in by_pixel.values():
        for r in rows:
            if float(r['clearance_m']) == clearance:
                values.setdefault(float(r['beta_deg']), []).extend(float(r[direction][k]) for k in SHARES if r[direction][k] is not None)
    if not values or any(len(v) != len(SHARES) * len(by_pixel) for v in values.values()):
        raise SystemExit('fraxet_sites.json: incomplete occlusion readings for site %s' % site)
    return [(b, min(v), max(v)) for b, v in sorted(values.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(description='Draw Figure 5 (occlusion on real-outcrop DEMs) from records/round3.')
    ap.add_argument('--records', default=str(HERE.parent / 'records' / 'round3'), help='folder that holds fraxet_sites.json and view_sweep.json')
    ap.add_argument('--out', default='.', help='output folder')
    a = ap.parse_args(argv)
    real = json.loads((Path(a.records) / 'fraxet_sites.json').read_text(encoding='utf-8'))
    sweep = json.loads((Path(a.records) / 'view_sweep.json').read_text(encoding='utf-8'))
    pixel = [float(r['pixel_mm']) for r in sweep['rows'] if r['outcrop'] == 'flat' and r.get('pixel_mm') is not None]
    syn_label = 'Synthetic flat outcrop (%.2f mm/px)' % (sum(pixel) / len(pixel))

    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    print('%-44s %-10s %5s %8s %8s %8s' % ('panel', 'series', 'beta', 'low %', 'mid %', 'high %'))
    for ax, (key, direction, title) in zip(axes, PANELS):
        syn = synthetic(sweep, direction)
        ax.plot([b for b, _ in syn], [100. * v for _, v in syn], color=INK, linestyle=(0, (4, 2)), linewidth=1.6, marker='s', markersize=4,
                label=syn_label)
        for b, v in syn:
            print('%-44s %-10s %5.0f %8s %8.2f %8s' % (title, 'synthetic', b, '', 100. * v, ''))
        for site, colour, label in SITES:
            band = site_band(real, site, key)
            betas = [b for b, _, _ in band]
            lo, hi = [100. * x for _, x, _ in band], [100. * x for _, _, x in band]
            mid = [(x + y) / 2. for x, y in zip(lo, hi)]
            ax.fill_between(betas, lo, hi, color=colour, alpha=.22, linewidth=0)
            ax.plot(betas, mid, color=colour, linewidth=2, marker='o', markersize=4.5, label=label)
            for b, x, m, y in zip(betas, lo, mid, hi):
                print('%-44s %-10s %5.0f %8.2f %8.2f %8.2f' % (title, site, b, x, m, y))
        ax.set_title(title, fontsize=8.8, loc='left', pad=4, color=INK)
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.set_xlim(-3, 63)
        ax.set_ylim(0, 50)
        ax.grid(axis='y', color='#e3e3e3', linewidth=.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.set_xlabel('Off-nadir angle \u03b2 (\u00b0)', fontsize=8.6, color=INK)
    axes[0].set_ylabel('Occluded trace length (%)', fontsize=8.6, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=8.2, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .86))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / ('%s.%s' % (NAME, ext)), dpi=220, facecolor='white')
    print('written %s.pdf and %s.png in %s' % (NAME, NAME, out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
