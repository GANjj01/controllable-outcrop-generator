"""Same frames and steps as run_frames_r2.py, except the projection: per fracture the stage-A export
(cavities/<case>/mid_*.npz, rebuild_cavities.py --midsurface) flattens the uncut host, project_path_branches runs on
the flattened sub-mesh and every hit maps back to the uncut host. Exit 3 if any gate stops.

--stage finish applies the refined points and runs visibility, gates and labels."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(P1)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P1 / 'trace_labels_r1'))
import run_frames_r2 as R2  # noqa: E402
from frgb.metrics import graph_node_weights, skeletonize  # noqa: E402
from frgb.network_training_labels import _projected_boxes  # noqa: E402
from frgb.raster import camera  # noqa: E402
from frgb import trace_labels as tl  # noqa: E402

r1 = R2.r1
OUT = HERE / 'r3'
R2.REVIEW = R2.REVIEW and R2.REVIEW.parent / 'r3'
T_OUT_M = .12
G1_P99_M = 1e-3
G1_INSIDE = .99
G2_OUTSIDE = .01
TIP_WIDTH_M = 3e-3
CORNER_DEG = 1.
FINE_CELL_M = .01
UNROLL = True


def unrolled_finish(res, sub, chart):
    g = sub['grid']
    f = res['host_face']
    b = res['bary']
    idx = g.f[f]
    n = (b[:, :, None] * sub['normal'][idx]).sum(1)
    res['slot_normal'] = n / np.linalg.norm(n, axis=1)[:, None]
    res['sigma'] = (b * sub['sigma'][idx]).sum(1)
    res['on_flipped_face'] = sub['flipped'][f]
    res['host_face'] = sub['face_ids'][f]
    res['barycentric_min'] = b.min(1)
    res['flat_point'] = tl.chart_surface(chart, res['hit_q'])
    if chart.get('shift') is not None:
        res['flat_point'] = res['flat_point'] + chart['shift']
    return res


def refine_caps(samples, order):
    pts = np.concatenate([samples[k]['point'] for k in order])
    grp = np.concatenate([samples[k]['path'] * 10 ** 9 + samples[k]['sample_index'] for k in order])
    cap = np.full(len(pts), np.inf)
    o = np.argsort(grp, kind='stable')
    bounds = np.flatnonzero(np.r_[True, grp[o][1:] != grp[o][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        idx = o[a:b]
        d = np.linalg.norm(pts[idx][:, None] - pts[idx][None], axis=2)
        np.fill_diagonal(d, np.inf)
        cap[idx] = .5 * d.min(1)
    return cap


def jump_check(samples, order, cam):
    """Jumped / capped length (screen weights), displacement over sibling distance, run breaks before / after."""
    pre = np.concatenate([samples[k]['pre_point'] for k in order])
    post = np.concatenate([samples[k]['point'] for k in order])
    grp = np.concatenate([samples[k]['path'] * 10 ** 9 + samples[k]['sample_index'] for k in order])
    capped = np.concatenate([samples[k].get('refine_capped', np.zeros(len(samples[k]['u']), bool)) for k in order])
    w_all, inf_all, br_pre, br_post = [], [], 0, 0
    for k in order:
        r = samples[k]
        uv0, z0 = tl.project(r['pre_point'], cam); uv1, z1 = tl.project(r['point'], cam)
        inf = tl.in_frame(np.where(np.isfinite(uv1), uv1, -1e9), cam) & (z1 > cam['near'])
        w_all.append(tl._sample_weights(dict(uv=uv1, in_frame=inf))); inf_all.append(inf)
        for uv, which in ((uv0, 0), (uv1, 1)):
            step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
            n = int(np.sum((step > tl.RUN_BREAK_PX) & inf[1:] & inf[:-1]))
            if which:
                br_post += n
            else:
                br_pre += n
    w, inf = np.concatenate(w_all), np.concatenate(inf_all)
    uv0, _ = tl.project(pre, cam); uv1, _ = tl.project(post, cam)
    disp_px = np.linalg.norm(uv1 - uv0, axis=1)
    jumped = np.zeros(len(pre), bool); ratio = np.full(len(pre), np.nan)
    o = np.argsort(grp, kind='stable')
    bounds = np.flatnonzero(np.r_[True, grp[o][1:] != grp[o][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        idx = o[a:b]
        d = np.linalg.norm(post[idx][:, None] - pre[idx][None], axis=2)
        jumped[idx] = d[np.arange(len(idx)), np.arange(len(idx))] > d.min(1) + 1e-12     # ties (coincident hits) are not jumps
        sib = np.linalg.norm(pre[idx][:, None] - pre[idx][None], axis=2); np.fill_diagonal(sib, np.inf)
        ratio[idx] = np.linalg.norm(post[idx] - pre[idx], axis=1) / sib.min(1)
    tot = float(w[inf].sum())
    ok = np.isfinite(ratio) & inf
    return dict(in_frame_length_px=tot, jumped_px=float(w[inf & jumped].sum()),
                jumped_fraction=float(w[inf & jumped].sum() / tot) if tot else None,
                capped_px=float(w[inf & capped].sum()), displaced_over_2px_px=float(w[inf & (disp_px > 2)].sum()),
                displaced_over_2px_fraction=float(w[inf & (disp_px > 2)].sum() / tot) if tot else None,
                displacement_over_sibling_p50_p99_max=[float(x) for x in np.quantile(ratio[ok], [.5, .99, 1.])] if ok.any() else None,
                run_breaks_before=br_pre, run_breaks_after=br_post)


def FS_filter(b, keep):
    """fault_support.filter_samples (imported lazily: run_frames_r3 is also used by fault-free batch drivers)"""
    import fault_support
    return fault_support.filter_samples(b, keep)


def _is_pseudo(index):
    import fault_support
    return fault_support.is_pseudo(index)


def r3_samples(case, cam, cavities):
    trace_of = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    owner = case.get('cutter_key', trace_of)
    samples, unresolved, info, subs, folds = {}, {}, {}, {}, {}
    for f in sorted(cavities.glob('mid_*.npz')):
        trace = f.stem[4:]
        with np.load(f, allow_pickle=False) as a:
            mid = {k: a[k] for k in a.files}
        folds[trace] = (mid['fold_sigma'], mid['fold_half_width'])
        make = tl.unroll_host if UNROLL else tl.flatten_host
        sub = make(case['host']['vertices'], case['host']['faces'], mid) if len(mid['vertex_ids']) else None
        chart = dict(Q=mid['chart_Q'], u=mid['chart_u'], dh=mid['chart_dh'], sigma=mid['chart_sigma']) if UNROLL else None
        case.setdefault('charts', {})[trace] = dict(Q=mid['chart_Q'], u=mid['chart_u'], dh=mid['chart_dh'], sigma=mid['chart_sigma'])
        if 'ext_index' in mid and case.get('ext_trim') is not None:
            for pid, base, end in np.asarray(mid['ext_index'], np.int64).reshape(-1, 3):
                pid, base, pre = int(pid), int(base), 'ext_%d_' % int(pid)
                dml = mid['extm_%d_depth_margin_L' % pid]
                case['paths'][pid] = dict(row=dict(case['paths'][base]['row'], index=pid, extension_of=base, extension_end=int(end)),
                                          stations={k[len(pre):]: mid[k] for k in mid if k.startswith(pre)},
                                          ext_meta=dict(p_end=mid['extm_%d_p_end' % pid], t2=mid['extm_%d_t2' % pid],
                                                        nu2=mid['extm_%d_nu2' % pid], depth=float(dml[0]), margin=float(dml[1]),
                                                        L=float(dml[2]), source=str(mid['extm_%d_source' % pid])))
                trace_of[pid] = trace_of[base]
                owner[pid] = owner[base]
                ind = {k[len('indep_%d_' % pid):]: mid[k] for k in mid if k.startswith('indep_%d_' % pid)}
                case['paths'][pid]['indep'] = ind or None
        if chart is not None and 'group_shift' in mid and np.any(mid['group_shift'] != 0):
            chart['shift'] = np.asarray(mid['group_shift'], float)
        subs[trace] = dict(covered_vertices=int(len(mid['vertex_ids'])), sub_faces=sub['sub_faces'] if sub else 0,
                           partial_faces=sub['partial_faces'] if sub else None,
                           flipped_faces=int(sub['flipped'].sum()) if sub else 0)
        for index, p in sorted(case['paths'].items()):
            if owner[index] != trace:
                continue
            st = p['stations']
            T = T_OUT_M + float(np.max(st['depth'][:, -1]))
            pseudo = _is_pseudo(index)
            if sub is None or sub['sub_faces'] == 0:
                u = tl._initial_params(st, cam, T, tl.SCREEN_SPACING_PX)
                if u is not None and not pseudo:
                    unresolved[index] = tl._interp(st, u)[1]
                    info[index] = dict(samples=int(len(u)), no_cavity_faces=True, t_bound_m=T)
                continue
            if FINE_CELL_M and 'fine' not in sub:      # speed-up s1: fine candidate grid on the flattened / unrolled sub-mesh
                sub['fine'] = tl.HostGrid(sub['grid'].v, sub['grid'].f, cell=FINE_CELL_M)
            ur = None
            if UNROLL:
                ur = dict(q=mid['path_%d_q' % index], dir=mid['path_%d_dir' % index], grid=sub['grid'], fine=sub['fine'],
                          original=sub['original'])
            branches, inf = tl.project_path_branches(st, sub['grid'], cam, case['front_sign'], t_bound=T, fine=sub.get('fine'),
                                                     unrolled=ur)
            if branches is None:
                continue
            unresolved_ = inf.pop('unresolved_station_points')
            if not pseudo:
                unresolved[index] = unresolved_
            info[index] = dict(inf, t_bound_m=T)
            if UNROLL:
                info[index]['station_locate_residual_max_m'] = float(mid['path_%d_locate_residual' % index].max())
            for b in branches:
                if UNROLL:
                    unrolled_finish(b, sub, chart)
                else:
                    tl.unflatten(b, sub)
                b['path'] = index; b['t_bound'] = T
                if pseudo:
                    n0 = len(b['u'])
                    b = FS_filter(b, case['ext_trim'](index, b))
                    info[index]['kept_samples'] = info[index].get('kept_samples', 0) + len(b['u'])
                    info[index]['trimmed_samples'] = info[index].get('trimmed_samples', 0) + n0 - len(b['u'])
                    if not len(b['u']):
                        continue
                samples[(index, b['branch'])] = b
        del sub
    return samples, unresolved, info, subs, folds


G1R1_REACH_M = .2
G1R1_PLANE_M = 1e-6       # G1r1.1 (a): an interval end on the fault plane (the trimmed cavity's cap)
G1R1_PHI_M = 1e-3         # G1r1.2
G1R1_PHI_AGREE_M = 1e-9
G1R1_SHARE = .2           # G1r1.3 report condition


def g1r1_group_planes(case):
    """{cutter group key: [(fault frame, moved)]} from the fault ends of build.json (pseudo paths share their base's group)"""
    owner = case.get('cutter_key')
    if not owner:
        return {}
    import fault_support as FS_
    FC = FS_.load_fault_charts()
    fby = {f['id']: f for f in FC.fault_frames(case['build']['source'])['faults']}
    out = {}
    for p in case['build']['paths']:
        for e in ((p.get('construction_checks') or {}).get('fault') or {}).get('ends', []):
            k = owner.get(int(p['index']))
            if k is not None and e.get('fault') in fby:
                pl = (fby[e['fault']], bool(e.get('moved')))
                if all(pl[0]['id'] != x[0]['id'] or pl[1] != x[1] for x in out.get(k, [])):
                    out.setdefault(k, []).append(pl)
    return dict(planes=out, so=FC.signed_other)


def g1r1_exclude(P, n, lo, hi, planes, so, reach=G1R1_REACH_M, tol=G1R1_PLANE_M):
    """G1r1.1 (a) an interval end on a plane;"""
    P, n = np.asarray(P, float).reshape(-1, 3), np.asarray(n, float).reshape(-1, 3)
    a = np.zeros(len(P), bool); b = np.zeros(len(P), bool); d = np.full(len(P), np.inf)
    for f, moved in planes:
        d = np.minimum(d, np.abs(so(f, moved, P)))
        for s_ in (lo, hi):
            e = so(f, moved, P + np.asarray(s_, float)[:, None] * n)
            a |= np.isfinite(e) & (np.abs(e) <= tol)
        s0, s1 = so(f, moved, P), so(f, moved, P + n)
        with np.errstate(divide='ignore', invalid='ignore'):
            sstar = -s0 / (s1 - s0)                          # the plane is linear along the line
        b |= np.isfinite(sstar) & (sstar >= np.asarray(lo, float) - tol) & (sstar <= np.asarray(hi, float) + tol)
    return a, b, d


def g1r1_on_midsurface(point, ref_point, phi_check, phi, tol=G1R1_PHI_M, agree=G1R1_PHI_AGREE_M):
    """G1r1.2 per sample: stored point bitwise the refined point, |phi_check| <= tol, |phi_check - phi| <= agree"""
    same = (np.asarray(point) == np.asarray(ref_point)).all(1)
    pc, ph = np.asarray(phi_check, float), np.asarray(phi, float)
    near = np.isfinite(pc) & (np.abs(pc) <= tol)
    agree_ = np.isfinite(pc) & np.isfinite(ph) & (np.abs(pc - ph) <= agree)
    return same & near & agree_, dict(point_differs=int((~same).sum()), off_midsurface=int((~near).sum()), phi_disagrees=int((~agree_).sum()))


def gate1(case, samples, cam, cutters, folds):
    """G1 and the G2 breakdown: slot centre at P' along the interpolated vertex normal, against the rebuilt cut solid."""
    trace_of = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
    owner = case.get('cutter_key', trace_of)
    mcol = bool(case.get('collect_mismatch'))    # mismatch_opening_r1 M1: per-sample arrays of the band (label_frame pops them)
    collect = bool(case.get('collect_g1'))
    grids = {}
    rows = []
    g1r1 = g1r1_group_planes(case)
    for key, r in samples.items():
        sel = np.flatnonzero(r['visible'])
        if not len(sel):
            continue
        tid = owner[r['path']]
        if tid not in grids:
            c = cutters[tid]
            grids[tid] = (tl.HostGrid(c['vertices'], c['faces'], cell=.02), c['sign'])
        g, sign = grids[tid]
        lo, hi = tl.slot_sections(g, sign, r['point'][sel], r['slot_normal'][sel])
        st = case['paths'][r['path']]['stations']
        n_st = st['normal']
        i = np.clip(np.floor(r['u'][sel]).astype(int), 0, len(n_st) - 2)
        ang = lambda a, b: np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', a, b), -1, 1)))
        corner = (ang(n_st[i], n_st[i + 1]) > CORNER_DEG) | (ang(n_st[np.maximum(i - 1, 0)], n_st[i]) > CORNER_DEG) | \
                 (ang(n_st[i + 1], n_st[np.minimum(i + 2, len(n_st) - 1)]) > CORNER_DEG)
        fs, fw = folds[tid]
        fold = (np.abs(r['sigma'][sel][:, None] - fs[None]) < fw[None]).any(1) if len(fs) else np.zeros(len(sel), bool)
        width = hi - lo
        foot = r['z'][sel] / cam['fpx']
        tip = (r['width'][sel] < TIP_WIDTH_M) | ~(width >= foot)
        dev = np.maximum(np.abs(lo - r['wall_lo'][sel]), np.abs(hi - r['wall_hi'][sel])) if 'wall_lo' in r else np.full(len(sel), np.nan)
        pl = (g1r1 or {}).get('planes', {}).get(tid, [])
        if pl:
            ex_a, ex_b, d_pl = g1r1_exclude(r['point'][sel], r['slot_normal'][sel], lo, hi, pl, g1r1['so'])
        else:
            ex_a = ex_b = np.zeros(len(sel), bool); d_pl = np.full(len(sel), np.inf)
        rows.append(dict(w=tl._sample_weights(r)[sel], c=.5 * (lo + hi), inside=(lo <= 0) & (hi >= 0), tip=tip, fold=fold,
                         corner=corner, width=width, px=np.abs(.5 * (lo + hi)) / foot,
                         ex_outside=r['ex_p_outside_own_cut'][sel], flipped=r['on_flipped_face'][sel],
                         phi=np.abs(r['refine_phi'][sel]) if 'refine_phi' in r else np.full(len(sel), np.nan), dev=dev,
                         ex_a=ex_a, ex_b=ex_b, d_plane=d_pl, pt=r['point'][sel],
                         row=r['refine_row'][sel] if 'refine_row' in r else np.full(len(sel), -1, np.int64)))
        if collect:
            rows[-1].update(path=np.full(len(sel), r['path'], np.int64), branch=np.full(len(sel), r.get('branch', 0), np.int64),
                            sample=sel.astype(np.int64), point=r['point'][sel], slot_normal=r['slot_normal'][sel],
                            aperture=r['width'][sel], foot=foot)
        if mcol:
            rows[-1].update(mm_uv=r['uv'][sel], mm_point=r['point'][sel], mm_normal=r['slot_normal'][sel],
                            mm_aperture=r['width'][sel], mm_foot=foot, mm_path=np.full(len(sel), r['path'], np.int64))
    cat = {k: np.concatenate([x[k] for x in rows]) if rows else np.empty(0) for k in
           ('w', 'c', 'inside', 'tip', 'fold', 'corner', 'width', 'px', 'ex_outside', 'flipped', 'phi', 'dev', 'ex_a', 'ex_b', 'd_plane', 'row')}
    cat['pt'] = np.concatenate([x['pt'] for x in rows]) if rows else np.zeros((0, 3))
    w = cat['w']
    nt = ~cat['tip'].astype(bool)

    def q(x, m, qs=(.5, .99)):
        ok = m & np.isfinite(x)
        if not ok.any():
            return None
        o = np.argsort(x[ok]); cw = np.cumsum(w[ok][o]) / w[ok].sum()
        return [float(x[ok][o][min(np.searchsorted(cw, qq), ok.sum() - 1)]) for qq in qs]
    absc = np.abs(cat['c'])
    out = dict(visible_px=float(w.sum()), tip_px=float(w[~nt].sum()),
               nontip_abs_centre_m_p50_p99=q(absc, nt), nontip_abs_centre_max_m=float(np.nanmax(absc[nt])) if nt.any() else None,
               nontip_centre_px_p50_p99=q(cat['px'], nt),
               nontip_inside_fraction=float(w[nt & cat['inside'].astype(bool)].sum() / w[nt].sum()) if nt.any() else None,
               nontip_no_slot_found_px=float(w[nt & ~np.isfinite(cat['c'])].sum()),
               fold_zone=dict(px=float(w[nt & cat['fold'].astype(bool)].sum()),
                              abs_centre_m_p50_p99=q(absc, nt & cat['fold'].astype(bool))),
               station_corner=dict(px=float(w[nt & cat['corner'].astype(bool)].sum()),
                                   abs_centre_m_p50_p99=q(absc, nt & cat['corner'].astype(bool))),
               elsewhere_abs_centre_m_p50_p99=q(absc, nt & ~cat['fold'].astype(bool) & ~cat['corner'].astype(bool)),
               slot_width_m_p50=q(cat['width'], nt, (.5,)),
               g2_outside_own_cut_nontip_px=float(w[nt & cat['ex_outside'].astype(bool)].sum()),
               visible_on_flipped_faces_px=float(w[cat['flipped'].astype(bool)].sum()),
               nontip_label_abs_phi_m_p50_p99=q(cat['phi'], nt),
               nontip_mesh_wall_deviation_m_p50_p99=q(cat['dev'], nt),
               nontip_mesh_wall_deviation_over_1mm_px=float(w[nt & (cat['dev'] > 1e-3)].sum()),
               nontip_abs_centre_m_p50_p99_where_mesh_within_half_mm=q(absc, nt & (cat['dev'] <= 5e-4)))
    p99 = out['nontip_abs_centre_m_p50_p99'][1] if out['nontip_abs_centre_m_p50_p99'] else None
    out['g1_all'] = dict(p99_m=p99, inside_fraction=out['nontip_inside_fraction'],
                         pass_=bool(p99 is not None and p99 <= G1_P99_M and out['nontip_inside_fraction'] >= G1_INSIDE))
    excl = nt & (cat['ex_a'].astype(bool) | cat['ex_b'].astype(bool))
    keep = nt & ~excl
    absc_ = np.abs(cat['c'])
    p99r = q(absc_, keep)
    ins = float(w[keep & cat['inside'].astype(bool)].sum() / w[keep].sum()) if keep.any() else None
    rec = dict(rule='criterion_trace_labels_r4_faults_g1_r1 (9cfa3297) + g1_r2 (c5497ebb)', excluded_px=float(w[excl].sum()), excluded=int(excl.sum()),
               excluded_a_cap=int((nt & cat['ex_a'].astype(bool)).sum()), excluded_b_within_interval=int((nt & cat['ex_b'].astype(bool)).sum()),
               nontip_px=float(w[nt].sum()), p99_m=(p99r[1] if p99r else None), inside_fraction=ins)
    rec['excluded_share'] = rec['excluded_px'] / rec['nontip_px'] if rec['nontip_px'] else 0.
    out['g1_excluded_share_high'] = bool(rec['excluded_share'] > G1R1_SHARE)
    bins = [(0., .01), (.01, .03), (.03, .10), (.10, np.inf)]
    over = nt & (absc_ > G1_P99_M)
    dist = lambda m_: {'%g-%g m' % b_: float(w[m_ & (cat['d_plane'] >= b_[0]) & (cat['d_plane'] < b_[1]) & np.isfinite(cat['d_plane'])].sum())
                       for b_ in bins[:-1]} | {'>0.1 m': float(w[m_ & (cat['d_plane'] >= .1) & np.isfinite(cat['d_plane'])].sum()),
                                               'no fault': float(w[m_ & ~np.isfinite(cat['d_plane'])].sum())}
    rec['over_1mm_px_by_plane_distance_before'] = dist(over)
    rec['over_1mm_px_by_plane_distance_after'] = dist(over & ~excl)
    off = False
    if excl.any():
        rp = case.get('refined_npz')
        if rp is None or not Path(rp).is_file():
            off, rec['midsurface_check'] = True, dict(error='no refined file')
        else:
            with np.load(rp) as a_:
                ref_pt, ref_phi = a_['point'], a_['phi']
                ref_chk = a_['phi_check'] if 'phi_check' in a_.files else np.full(len(ref_phi), np.nan)
            row = cat['row'].astype(np.int64)
            ok_row = (row >= 0) & (row < len(ref_pt))
            ix = np.flatnonzero(excl)
            rr = np.clip(row[ix], 0, max(len(ref_pt) - 1, 0))
            ok, detail = g1r1_on_midsurface(cat['pt'][ix], ref_pt[rr], ref_chk[rr], ref_phi[rr])
            ok &= ok_row[ix]
            detail.update(checked=int(len(ix)), passed=int(ok.sum()), rows_missing=int((~ok_row[ix]).sum()),
                          abs_phi_check_max_m=float(np.nanmax(np.abs(ref_chk[rr]))) if len(ix) and np.isfinite(ref_chk[rr]).any() else None)
            rec['midsurface_check'] = detail
            off = not bool(ok.all())
    out['g1r1_off_midsurface'] = off
    out['g1_r1'] = rec
    out['g1_pass'] = bool(p99r is not None and p99r[1] <= G1_P99_M and ins is not None and ins >= G1_INSIDE)
    out['g2_fraction'] = out['g2_outside_own_cut_nontip_px'] / w[nt].sum() if nt.any() else None
    out['g2_stop'] = bool(out['g2_fraction'] is not None and out['g2_fraction'] > G2_OUTSIDE)
    if collect:
        extra = ('path', 'branch', 'sample', 'point', 'slot_normal', 'aperture', 'foot')
        out['_samples'] = {k: np.concatenate([x[k] for x in rows]) if rows else np.empty(0) for k in
                           ('w', 'c', 'inside', 'tip', 'px', 'phi', 'dev') + extra}
    if mcol:
        cat_ = lambda k, e: np.concatenate([x[k] for x in rows]) if rows else e  # noqa: E731
        out['_mismatch'] = dict(w=cat['w'], c=cat['c'], px=cat['px'], tip=cat['tip'].astype(bool), ex_a=cat['ex_a'].astype(bool),
                                ex_b=cat['ex_b'].astype(bool), row=cat['row'].astype(np.int64), uv=cat_('mm_uv', np.zeros((0, 2))),
                                point=cat_('mm_point', np.zeros((0, 3))), slot_normal=cat_('mm_normal', np.zeros((0, 3))),
                                aperture=cat_('mm_aperture', np.empty(0)), foot=cat_('mm_foot', np.empty(0)),
                                path=cat_('mm_path', np.empty(0, np.int64)))
    return out


def bound_exits(samples):
    px = 0
    for r in samples.values():
        for end in (0, -1):
            if r['visible'][end] and abs(r['t'][end]) >= r['t_bound'] - 1e-3:
                px += 1
    return px


def shift_vs_r2(target, r2_target):
    if not target.any() or not r2_target.any():
        return None
    d = ndi.distance_transform_edt(~r2_target)[target]
    w = graph_node_weights(target)[target]
    o = np.argsort(d); cw = np.cumsum(w[o]) / w.sum()
    pick = lambda qq: float(d[o][min(np.searchsorted(cw, qq), len(d) - 1)])
    return dict(p50=pick(.5), p90=pick(.9), p99=pick(.99), over_2px_fraction=float(w[d > 2].sum() / w.sum()))


def apply_refinement(samples, order, path):
    with np.load(path) as a:
        ref = {k: a[k] for k in a.files}
    k0 = 0
    for key in order:
        r = samples[key]
        n = len(r['u'])
        sl = slice(k0, k0 + n); k0 += n
        ok = np.isfinite(ref['normal'][sl]).all(1)
        r['pre_point'] = r['point'].copy()
        r['refine_capped'] = ref['capped'][sl] if 'capped' in ref else np.zeros(n, bool)
        r['point'] = np.where(ok[:, None], ref['point'][sl], r['point'])
        r['host_face'] = np.where(ok, ref['face'][sl], r['host_face'])
        r['slot_normal'] = np.where(ok[:, None], ref['normal'][sl], r['slot_normal'])
        r['sigma'] = np.where(ok, ref['sigma'][sl], r['sigma'])
        r['refine_phi'], r['refine_shift'], r['refine_steps'] = ref['phi'][sl], ref['shift'][sl], ref['steps'][sl]
        r['wall_lo'], r['wall_hi'] = ref['wall_lo'][sl], ref['wall_hi'][sl]
        r['refine_row'] = np.arange(sl.start, sl.stop)
    assert k0 == len(ref['point'])
    a = np.abs(ref['phi'][np.isfinite(ref['phi'])])
    return dict(points=int(k0), finite_phi=int(len(a)), converged=int((a <= 1e-5).sum()),
                capped=int(ref['capped'].sum()) if 'capped' in ref else None,
                abs_phi_m_p50_p99_max=[float(x) for x in np.quantile(a, [.5, .99, 1.])] if len(a) else None,
                shift_m_p50_p99_max=[float(x) for x in np.quantile(ref['shift'], [.5, .99, 1.])])


def run_group(gname, spec, args):
    started = time.time()
    geometry_only = spec.get('geometry_only', False)
    req = R2.read(spec['request']) if geometry_only else r1.capture_frame(*spec['frames'][0])['request']
    first = None if geometry_only else r1.capture_frame(*spec['frames'][0])
    cam = camera(req['shots'][0], req['resolution'])
    case = tl.load_case(spec['case'])
    case['_shot'] = req['shots'][0]
    grid = tl.HostGrid(case['host']['vertices'], case['host']['faces'])
    case['host_normals'] = grid.normals
    del grid
    vol = r1.signed_volume(case['host']['vertices'].astype(float), case['host']['faces'])
    case['front_sign'] = 1. if vol > 0 else -1.
    cutters, manifest = R2.load_cutters(spec['cavities'], cam)
    log = dict(group=gname, criterion_sha256=args.criterion_sha, amendment_a_sha256=args.amendment_sha, amendment_b_sha256=args.amendment_b_sha,
               amendment_c_sha256=args.amendment_c_sha, case=str(spec['case']), resolution=req['resolution'],
               midsurface_export=manifest.get('midsurface'))
    pkl = OUT / ('samples_%s.pkl' % gname)
    if args.stage == 'finish':
        with open(pkl, 'rb') as fh:
            samples, unresolved, info, subs, folds, order = pickle.load(fh)
        log['refinement'] = apply_refinement(samples, order, OUT / ('refine_in_%s_refined.npz' % gname))
        log['jump_check'] = jump_check(samples, order, cam)
    else:
        samples, unresolved, info, subs, folds = r3_samples(case, cam, spec['cavities'])
        order = sorted(samples)
        if args.stage == 'project':
            OUT.mkdir(parents=True, exist_ok=True)
            with open(pkl, 'wb') as fh:
                pickle.dump((samples, unresolved, info, subs, folds, order), fh, protocol=4)
            trace_of = {i: p['row']['source_trace_id'] for i, p in case['paths'].items()}
            np.savez(OUT / ('refine_in_%s.npz' % gname), point=np.concatenate([samples[k]['point'] for k in order]),
                     face=np.concatenate([samples[k]['host_face'] for k in order]),
                     trace=np.concatenate([np.full(len(samples[k]['u']), int(trace_of[samples[k]['path']].split('-')[1]))
                                           for k in order]),
                     cap=refine_caps(samples, order))
            print(gname, 'projected and saved', len(order), 'branches', round(time.time() - started, 1), flush=True)
            return dict(stop=dict(), stage='project')
    log['flattening'] = subs
    print(gname, 'projected', {k: v.get('branches') for k, v in info.items()}, round(time.time() - started, 1), flush=True)
    flat_view = {k: dict(r, point=r['flat_point']) for k, r in samples.items()}
    residual = tl.fracture_surface_residual(case['source'], flat_view, case['paths'])
    log['projection_r3'] = R2.projection_record(samples, info, residual, cam)
    for k, v in info.items():
        log['projection_r3'].setdefault(str(k), {}).update(t_bound_m=v.get('t_bound_m'), no_cavity_faces=v.get('no_cavity_faces', False))
    nat = case['native']
    boxes = _projected_boxes(nat['vertices'], nat['faces'], cam)[:3]
    host_boxes = _projected_boxes(case['host']['vertices'], case['host']['faces'], cam)[:3]
    tl.classify_visibility(case, samples, cam, boxes=boxes)
    ex = tl.exemption_check(case, samples, cam, cutters, host_boxes=host_boxes)
    g = gate1(case, samples, cam, cutters, folds)
    log['r3'] = dict(visible_length_px=R2.weights_sum(samples, 'visible'), exemption=ex, gates=g,
                     eps_sensitivity_visible_px=R2.eps_sensitivity(samples, cam),
                     own_cavity_clause_only_px=R2.weights_sum(samples, 'visible_own_cavity_only'),
                     visible_branch_ends_at_t_bound=bound_exits(samples),
                     in_frame_zero_hit_station_samples=int(sum(len(v) for v in unresolved.values())))
    stop = dict(exemption=bool(ex['stop_triggered']), g1_fail=not g['g1_pass'], g2=bool(g['g2_stop']))
    log['stop'] = stop
    print(gname, 'r3 exemption', round(ex['contradiction_fraction'] or 0, 5), 'gates', json.dumps(g), round(time.time() - started, 1),
          flush=True)
    shape = (req['resolution'][1], req['resolution'][0])
    contra = R2.marks(samples, 'ex_contradiction', shape)
    out_group = OUT / 'frames'
    reports = []
    if geometry_only:
        labels = tl.build_trace_geometry(case, samples, cam, shape, unresolved_points=unresolved)
        far = R2.parallel_far_marks(samples, shape)
        name = spec['frames'][0][0]
        od = out_group / name; od.mkdir(parents=True, exist_ok=True)
        for k in ('target', 'valid', 'ignore_reason', 'occluded_line'):
            np.save(od / (k + '.npy'), labels[k])
        base = R2.shading_image(case, cam, size=shape[0] // 2)
        crops = R2.pick_crops(labels, contra, far, shape)
        r2_target = np.load(HERE / 'frames' / name / 'target.npy')
        report = dict(frame=name, group=gname, status='TRACE_LABEL_GEOMETRY_ONLY_NOT_QUALIFIED',
                      positive_graph_length_px=labels['positive_graph_length_px'], positive_pixels=int(labels['target'].sum()),
                      parallel=tl.parallel_record(samples), shift_vs_r2=shift_vs_r2(labels['target'], r2_target),
                      per_path=labels['per_path'], crops_xy_size=crops,
                      overlays=R2.save_r2_overlays(name, base, labels, contra, far, crops, scale=2))
        r2_labels = dict(target=r2_target, ignore_reason=np.load(HERE / 'frames' / name / 'ignore_reason.npy'),
                         occluded_line=np.load(HERE / 'frames' / name / 'occluded_line.npy'))
        R2.save_r2_overlays(name + '_r2_same_crops', base, r2_labels, np.zeros(shape, bool), np.zeros(shape, bool), crops,
                            full=False, scale=2)
        (od / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n', encoding='utf8')
        reports.append(report)
    else:
        cpu = r1.load_cpu_cache(spec['cpu_cache'], req)
        group_crops = None
        for fname, capture in (spec['frames'][:1] if any(stop.values()) else spec['frames']):
            frame = r1.capture_frame(fname, capture)
            rgb = np.asarray(Image.open(frame['rgb']).convert('RGB'))
            dm = np.load(frame['depth'], allow_pickle=False).astype(float) / 100.
            rk = np.load(frame['rock'], allow_pickle=False) > 0
            al = r1.aligned_mask(cpu, dm, rk)
            smp = {k: dict(r) for k, r in samples.items()}
            labels = tl.build_trace_labels(case, smp, cam, rgb, cpu_raster=cpu, render_rock=rk, aligned=al,
                                           unresolved_points=unresolved)
            far = R2.parallel_far_marks(smp, shape)
            od = out_group / fname; od.mkdir(parents=True, exist_ok=True)
            for k in ('target', 'valid', 'ignore_reason', 'u_line', 'u_line_old_threshold', 'occluded_line'):
                np.save(od / (k + '.npy'), labels[k])
            wins = labels['windows']
            rock_px = int(labels['rock'].sum())
            report = dict(frame=fname, group=gname, rgb=str(frame['rgb']),
                          status='TRACE_LABEL_CANDIDATE_NOT_QUALIFIED' + ('_STOPPED_GROUP' if any(stop.values()) else ''),
                          positive_graph_length_px=labels['positive_graph_length_px'],
                          positive_pixels=int(labels['target'].sum()),
                          ignore_fraction_of_rock=float((labels['ignore_reason'] > 0)[labels['rock']].sum() / max(rock_px, 1)),
                          u_windows=dict(total=len(wins), u=sum(w['u'] for w in wins),
                                         ring_below_min=sum(w['u'] and w['ring_pixels'] < tl.U_MIN_RING for w in wins),
                                         ring_excluded_pixels=sum(w.get('ring_excluded', 0) for w in wins)),
                          u_line_px=float(graph_node_weights(skeletonize(labels['u_line'])).sum()),
                          parallel=tl.parallel_record(smp), parallel_visible_only=tl.parallel_record(smp, use_u=False),
                          shift_vs_r2=shift_vs_r2(labels['target'], np.load(HERE / 'frames' / fname / 'target.npy')),
                          technical_failure_all_u=labels['all_visible_u'], per_path=labels['per_path'])
            if group_crops is None:
                group_crops = R2.pick_crops(labels, contra, far, shape)
            report['crops_xy_size'] = group_crops
            report['overlays'] = R2.save_r2_overlays(fname, rgb, labels, contra, far, group_crops, full=fname in args.full)
            (od / 'windows.json').write_text(json.dumps(wins, allow_nan=False) + '\n', encoding='utf8')
            (od / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n', encoding='utf8')
            reports.append(report)
            print(fname, 'pos', round(report['positive_graph_length_px'], 1), 'U', report['u_windows']['u'], '/',
                  report['u_windows']['total'], 'parallel far', report['parallel']['far_fraction'], 'shift', report['shift_vs_r2'],
                  flush=True)
    log['frames'] = [dict(frame=r['frame'], positive_graph_length_px=r['positive_graph_length_px'], u_windows=r.get('u_windows'),
                          parallel=r['parallel'], parallel_visible_only=r.get('parallel_visible_only'),
                          shift_vs_r2=r['shift_vs_r2']) for r in reports]
    log['elapsed_s'] = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('group_%s.json' % gname)).write_text(json.dumps(log, indent=1, allow_nan=False) + '\n', encoding='utf8')
    return log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='+', default=['sa08_path08cam', 'sa04_path04cam', 'a255_H100_c0'])
    ap.add_argument('--full', nargs='*', default=['path08_day_b1', 'path04_day_b1', 'shape_ctl_b1_e0_a0', 'shape_ctl_b1_e3_a1'])
    ap.add_argument('--stage', choices=['all', 'project', 'finish'], default='all')
    args = ap.parse_args()
    args.criterion_sha = (HERE / 'criterion_trace_labels_r3.sha256.txt').read_text(encoding='utf8').split()[0]
    args.amendment_sha = (HERE / 'criterion_trace_labels_r3_amendment_a.sha256.txt').read_text(encoding='utf8').split()[0]
    args.amendment_b_sha = (HERE / 'criterion_trace_labels_r3_amendment_b.sha256.txt').read_text(encoding='utf8').split()[0]
    args.amendment_c_sha = (HERE / 'criterion_trace_labels_r3_amendment_c.sha256.txt').read_text(encoding='utf8').split()[0]
    stopped = []
    for gname in args.groups:
        log = run_group(gname, R2.GROUPS[gname], args)
        if any(log['stop'].values()):
            stopped.append((gname, log['stop']))
            print('STOP in group', gname, log['stop'], flush=True)
    sys.exit(3 if stopped else 0)
