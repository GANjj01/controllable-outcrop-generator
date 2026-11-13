"""Recompute the paper's main tables from the per-frame and per-case records in records/.

Each table is aggregated from the rows of one record (one row per frame or per case) and compared with the summary that the
manuscript quotes, which is stored next to the rows. Prints the tables and writes tables.json; exit status 1 when a
recomputed value differs from the stored summary. Needs Python 3.8+ and NumPy.

    python reproduce/paper1/tables.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = HERE / 'records'
OUT = {}
BAD = []


def load(name):
    return json.loads((REC / name).read_text(encoding='utf-8'))


def same(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol * max(1., abs(float(b)))


def dataset_stats():
    """multi-light aerial set (50 frames): median and range of the per-frame label statistics"""
    d = load('stats_e_formal.json')
    rows = d['main_rows']
    table = {}
    for key, ref in d['main'].items():
        vals = np.array([r[key] for r in rows if r.get(key) is not None], float)
        got = dict(n=int(len(vals)), median=float(np.median(vals)), min=float(vals.min()), max=float(vals.max()))
        table[key] = got
        for k in ('n', 'median', 'min', 'max'):
            if not same(got[k], ref[k]):
                BAD.append('dataset_stats %s %s: %r vs %r' % (key, k, got[k], ref[k]))
    w = d['width_px_per_view']
    med = [r['ratio_px_p10_p50_p90'][1] for r in w['rows']]
    table['geometric_opening_px_per_view_median'] = dict(median=float(np.median(med)), min=float(min(med)), max=float(max(med)))
    for k, v in (('median', np.median(med)), ('min', min(med)), ('max', max(med))):
        if not same(v, w[k]):
            BAD.append('dataset_stats opening width %s' % k)
    OUT['dataset_stats'] = table
    print('\nTable: per-frame statistics of the multi-light aerial set (%d frames)' % len(rows))
    names = [('visible_length_px', 'visible trace length (px)', 1), ('u_share_of_visible', 'unclear share of visible length', 100),
             ('occluded_share_of_in_frame', 'occluded share of in-frame length', 100), ('any_mark_share_of_rock', 'any mark, share of rock', 100),
             ('opening_1_share_of_rock', 'opening class 1, share of rock', 100), ('opening_2_share_of_rock', 'opening class 2, share of rock', 100)]
    for key, label, f in names:
        t = table[key]
        unit = '' if f == 1 else ' %'
        print('  %-38s median %10.4g%s   range %.4g-%.4g%s' % (label, t['median'] * f, unit, t['min'] * f, t['max'] * f, unit))
    t = table['geometric_opening_px_per_view_median']
    print('  %-38s median %10.4g     range %.4g-%.4g (5 views)' % ('geometric opening (px)', t['median'], t['min'], t['max']))


def shape_control():
    """shape-parameter variants: locality and label checks (passes / cases) and the image change inside the trace band"""
    rows = load('b_summary_formal.json')['rows']
    groups = {}
    for r in rows:
        if r['role'] != 'variant':
            continue
        base = [b for b in rows if b['id'] == r['baseline']][0]['physical']
        changed = tuple(sorted(k for k in r['physical'] if abs(r['physical'][k] - base[k]) > 1e-12 and k != 'roughness_m'
                               or (k == 'roughness_m' and abs(r['physical'][k] / r['physical']['mean_opening_m']
                                                              - base[k] / base['mean_opening_m']) > 1e-9)))
        key = r['group'] if r['group'] != 'B_other_genes' else 'B_' + '+'.join(changed)
        g = groups.setdefault(key, dict(n=0, locality=0, labels=0, T=[], cameras=set(), values=set()))
        g['n'] += 1
        loc = r.get('local') or {}
        g['locality'] += loc.get('geometry_gate') == 'PASS' and loc.get('image_gate') == 'PASS'
        g['labels'] += (r.get('labels') or {}).get('numeric_overall') == 'PASS'
        if loc.get('T_inside_band') is not None:
            g['T'].append(loc['T_inside_band'])
        g['cameras'].add(r['cam'])
        g['values'].add(json.dumps({k: round(v, 6) for k, v in r['physical'].items()}, sort_keys=True))
    print('\nTable: locality and label checks of the shape-parameter variants (passes / cases)')
    out = {}
    for key, g in sorted(groups.items()):
        out[key] = dict(cases=g['n'], cameras=len(g['cameras']), locality_pass=int(g['locality']), label_pass=int(g['labels']),
                        T_inside_band_levels=[min(g['T']), max(g['T'])] if g['T'] else None)
        print('  %-34s views %d   locality %2d/%-2d   labels %2d/%-2d   band T %s' % (key, len(g['cameras']), g['locality'], g['n'],
                                                                                    g['labels'], g['n'], out[key]['T_inside_band_levels']))
    OUT['shape_control'] = out


def dark_band():
    """median apparent dark-band width (px) per view and mean opening (fine ruler)"""
    d = load('fine_width_formal.json')
    tags = {'w030': '3 cm', 'w040': '4 cm', 'b055': '5.5 cm', 'w065': '6.5 cm', 'w080': '8 cm'}
    cams = {}
    for r in d['rows']:
        if r.get('tag') in tags and r.get('status') == 'measured' and r.get('fine'):
            cams.setdefault((r['pixel_mm'], r['cam']), {})[r['tag']] = r['fine']['median']
    print('\nTable: dark-band width (median px) per view and mean opening')
    print('  %-30s ' % 'view (mm/px, camera)' + ' '.join('%7s' % t for t in tags.values()))
    out = {}
    for (mm, cam), v in sorted(cams.items(), key=lambda kv: -kv[0][0]):
        out[cam] = dict(pixel_mm=mm, **{tags[t]: v.get(t) for t in tags})
        print('  %-30s ' % ('%.2f %s' % (mm, cam)) + ' '.join('%7s' % ('%.2f' % v[t] if t in v else '-') for t in tags))
    OUT['dark_band'] = out


def frame_counts():
    """labelled frames of the formal label set per experiment block"""
    idx = load('frames_index.json')['frames']
    blocks = {}
    for k, f in idx.items():
        blocks[f['block']] = blocks.get(f['block'], 0) + 1
    OUT['formal_frames_per_block'] = blocks
    print('\nFormal label set: %d frames, per block %s' % (len(idx), json.dumps(blocks, sort_keys=True)))


if __name__ == '__main__':
    dataset_stats()
    shape_control()
    dark_band()
    frame_counts()
    (HERE / 'tables.json').write_text(json.dumps(OUT, indent=1) + '\n', encoding='utf-8')
    if BAD:
        print('\nDIFFERENCES FROM THE STORED SUMMARIES:\n  ' + '\n  '.join(BAD))
        sys.exit(1)
    print('\nall recomputed summaries equal the stored ones')
