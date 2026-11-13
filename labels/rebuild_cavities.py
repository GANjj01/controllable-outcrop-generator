"""Rebuild the per-fracture cut solids of a case.

Runs the case's own build.py main() with the recorded arguments into a scratch directory; frgb.network_panels.
build_surface_cavity is wrapped only to keep the solids it returns. Verification (all, else exit 3, no adjustment):
per-fracture volume and triangle count against build.json, remaining volume, and bitwise stations. Exports every solid
as cavities/<case>/<trace>.npz (vertices float64, faces int64) and cavities/<case>/manifest.json. Usage:
rebuild_cavities.py CASE [--code current|backup] [--midsurface] rebuild_cavities.py --case-dir CASE_DIR --request
REQUEST_JSON --dest DIR [--scratch DIR] [--clean] [...]

--midsurface: after the verification, for every fracture the cavity mid-surface offset m = (L + R) / 2 + rho
(envelopes of its pieces, wall relief with the tip taper as _surface_cavity, fold zones of the cavity) is evaluated at
the uncut-host vertices inside the cutter AABB + 0.50 m whose chart point is covered by both envelopes; exported as
cavities/<case>/mid_<trace>.npz (vertex ids, m, n, D = m n)."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
MS = P1 / 'multiscene'
sys.path.append(str(P1))
from frgb import runconfig as frgb_rc  # noqa: E402
TOOLS = frgb_rc.path('manifold_site').as_posix()
SCRATCH = Path(tempfile.gettempdir())
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))


def scene_case(scene):
    d = MS / 'feas_r2' / scene
    return dict(case=d / 'net/cases/c0', raw=d / 'net/cases/c0_raw', module=d / 'net', scene=d)


CASES = {
    'sa04_G1_p0_r1': dict(case=P1 / 'network_morphology_r2/cases/sa04_G1_p0_r1', module=P1 / 'network_morphology_r2'),
    'sa08_G1_p0_r2': dict(case=P1 / 'network_morphology_r2/cases/sa08_G1_p0_r2', module=P1 / 'network_morphology_r2'),
    'a255_H100_c0': scene_case('r2_sh4_a255_H100'),
    'a255_H125_c0': scene_case('r2_sh4_a255_H125'),
}


def arguments(spec, build, out):
    g = build['genome']
    argv = [str(out), '--width', repr(g[0]), '--variation', repr(g[1]), '--depth', repr(g[2]), '--roughness', repr(g[3]),
            '--construction', 'panel']
    h = {p['construction_checks']['refine_length_m'] for p in build['paths']}
    assert len(h) == 1, h
    if 'scene' not in spec:
        return argv + ['--panel-refine-m', repr(h.pop()), '--source', build['source']]
    assert h.pop() == .002
    d = spec['scene']
    relief = d / 'scene/wall_relief.json'
    assert hashlib.sha256(relief.read_bytes()).hexdigest() == build['wall_relief']['sha256']
    assert build['wall_relief_tip_m'] == .003 and Path(build['source']) == d / 'levels/source_0'
    return argv + ['--panel-refine-m', '0.002', '--wall-relief', str(relief).replace('\\', '/'), '--wall-relief-tip-m', '0.003',
                   '--view-clip', str(d / 'levels/request_0.json').replace('\\', '/'),
                   '--source', str(d / 'levels/source_0').replace('\\', '/')]


def generic_arguments(build, out, request):
    """build.py argv recorded in build.json."""
    g = build['genome']
    argv = [str(out), '--width', repr(g[0]), '--variation', repr(g[1]), '--depth', repr(g[2]), '--roughness', repr(g[3]),
            '--construction', build['construction']]
    h = {p['construction_checks']['refine_length_m'] for p in build['paths']}
    assert len(h) == 1, h
    argv += ['--panel-refine-m', repr(h.pop()), '--source', str(build['source']).replace(chr(92), '/')]
    if build.get('joint', 'hull') != 'hull':
        argv += ['--joint', build['joint']]
    if build.get('cutter_simplify_m') is not None:
        argv += ['--cutter-simplify-m', repr(build['cutter_simplify_m'])]
    if build.get('wall_relief'):
        wr = Path(build['wall_relief']['path'])
        assert hashlib.sha256(wr.read_bytes()).hexdigest() == build['wall_relief']['sha256'], 'wall relief file changed'
        argv += ['--wall-relief', str(wr).replace(chr(92), '/')]
    if build.get('wall_relief_tip_m') is not None:
        argv += ['--wall-relief-tip-m', repr(build['wall_relief_tip_m'])]
    if build.get('wall_mouth_band'):
        mb = build['wall_mouth_band']
        assert hashlib.sha256(Path(mb['path']).read_bytes()).hexdigest() == mb['sha256'], 'mouth band file changed'
        argv += ['--wall-mouth-band', str(mb['path']).replace(chr(92), '/'), '--wall-mouth-fine-m', repr(mb['fine_m'])]
    if any('view_clip' in p for p in build['paths']):
        if request is None:
            raise ValueError('the case was view-clipped: --request is needed')
        argv += ['--view-clip', str(request).replace(chr(92), '/')]
    if not build.get('full_exposure_coverage', True):
        for p in build['paths']:
            argv += ['--path', str(p['index'])]
    return argv


def backup_package():
    root = Path(tempfile.mkdtemp(prefix='frgb_bak_', dir=SCRATCH))
    shutil.copytree(P1 / 'frgb', root / 'frgb', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('network_panels', 'network_cutters', 'network_profiles'):
        shutil.copyfile(MS / 'cutter_fix_r1/bak' / (name + '.py.bak'), root / 'frgb' / (name + '.py'))
    return root


FAULT_POS_TOL_M = 1e-9
FAULT_VOL_REL = 1e-9
STATION_SHIFT_TOL_M = 1e-12
STATION_RESIDUAL_M = 1e-9
UNSUPPORTED_EXIT = 6


def fault_status(raw, spec, FC, FS, code):
    """Returns a record with status no_faults / ok / unsupported / stop."""
    src = Path(raw['source'])
    if not (src / 'package/model_spec.json').is_file():
        if FS.faulted_build(raw):
            return dict(status='stop', reason='build.json carries fault_frames but the source has no package/model_spec.json')
        return dict(status='no_faults', legacy_source_without_model_spec=True)
    frames = FC.fault_frames(src)
    if frames['status'] == 'no_faults':
        if FS.faulted_build(raw):
            return dict(status='stop', reason='module: no_faults, build.json: fault_frames')
        return dict(status='no_faults')
    if frames['status'] == 'unsupported':
        return dict(status='unsupported', reasons=list(frames['reasons']))
    rec = dict(status='ok', faults=[dict(id=f['id'], t_m=[float(x) for x in f['t']], moving_side=f['moving_side'],
                                         moved_joints=sorted(f['moved'])) for f in frames['faults']],
               multiple=sorted(frames['multiple']), tol_m=float(frames['tol'].length_m))
    joints = {p['interface_id'] for p in raw['paths']}
    if joints & set(frames['multiple']):
        return dict(rec, status='unsupported', reasons=['joints moved by >= 2 faults: %s' % sorted(joints & set(frames['multiple']))])
    if code != 'current':
        return dict(rec, status='stop', reason='--code %s on a faulted case: the backup package is never used for faulted scenes' % code)
    if not FS.faulted_build(raw):
        return dict(rec, status='stop', reason='module: faults, build.json: no fault_frames (a builder without fault support)')
    recorded = [dict(id=f['id'], t_m=[float(x) for x in f['t_m']], moving_side=f['moving_side'], moved_joints=sorted(f['moved_joints']))
                for f in raw['fault_frames']]
    if recorded != rec['faults']:
        return dict(rec, status='stop', reason='build.json fault_frames differ from the module', recorded=recorded)
    md5s = {'build.py': FS.md5(spec['module'] / 'build.py'), 'prepare.py': FS.md5(spec['module'] / 'prepare.py'),
            'frgb/fault_charts.py': FS.md5(P1 / 'frgb/fault_charts.py')}
    rec['builder_md5'] = md5s
    if any(md5s[k] not in (FS.DEPLOYED_MD5[k], FS.INTERNAL_MD5[k]) for k in md5s):
        return dict(rec, status='stop', reason='builder files differ from the fault-support deployment', expected=FS.DEPLOYED_MD5)
    rec['frames'] = frames
    return rec


def _set_distance(X, Y, tol=FAULT_POS_TOL_M, max_rest=20000):
    """distance of every row of X to the nearest row of Y, matched as a set: exact row matches first (translate keeps a
    vertex as v + t bit for bit), then a sorted-x window search for the rest; inf beyond tol or when more than
    max_rest rows are unmatched"""
    X, Y = np.ascontiguousarray(X, float), np.ascontiguousarray(Y, float)
    view = lambda a: a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    out = np.zeros(len(X))
    rest = np.flatnonzero(~np.isin(view(X), view(Y)))
    if len(rest) > max_rest:
        out[rest] = np.inf
        return out
    o = np.argsort(Y[:, 0], kind='stable'); ys = Y[o, 0]
    for i in rest:
        a, b = np.searchsorted(ys, X[i, 0] - tol, 'left'), np.searchsorted(ys, X[i, 0] + tol, 'right')
        out[i] = float(np.linalg.norm(Y[o[a:b]] - X[i], axis=1).min()) if b > a else np.inf
    return out


def verify_faulted(captured, raw, new, fault, fcdir, FC, FS):
    """Group-wise stage-A verification of a faulted case: per builder group (trace, block) the captured (pre-replay) solid
    against the replay record, and the --save-fault-cavities export against gates i-v. Returns (ok, record, groups)
    with groups[k] = dict(key, trace, tag, shift, acted, paths, world=(vertices, faces) or None)."""
    import manifold3d as md
    frames = fault['frames']
    pg = FS.path_groups(raw)
    if {i: (g['key'], g['tag']) for i, g in FS.path_groups(new).items()} != {i: (g['key'], g['tag']) for i, g in pg.items()}:
        return False, dict(error='rebuilt groups differ from the recorded ones'), []
    rows = {p['index']: p for p in raw['paths']}
    rebuilt_checks = {p['index']: p['construction_checks'] for p in new['paths']}
    rec, groups, ok = {}, [], True
    files = sorted(p.name for p in Path(fcdir).glob('*')) if Path(fcdir).is_dir() else []
    expected_files = set()
    for indices, solid in captured:
        keys = {pg[i]['key'] for i in indices}
        if len(keys) != 1:
            return False, dict(error='captured solid spans several groups: %s' % sorted(keys)), []
        g = pg[indices[0]]
        key, row = g['key'], rows[indices[0]]
        fr = row['construction_checks'].get('fault')
        acted = fr is not None and fr.get('replay') is not None
        vol = solid.volume()
        target = fr['replay']['volume_before_m3'] if acted else row['surface_volume_m3']
        rel = abs(vol - target) / target
        differ, extra = set(), set()
        for i in indices:
            o, n = rows[i]['construction_checks'], rebuilt_checks[i]
            differ |= {k for k in o if o[k] != n.get(k)}
            extra |= set(n) - set(o)
        same = rel <= 1e-9 and not differ and not extra
        r = dict(paths=indices, trace=g['trace'], tag=g['tag'], acted=acted, captured_volume_m3=vol, compared_with=target,
                 relative=rel, differing_check_keys=sorted(differ), added_check_keys=sorted(extra))
        world = None
        if acted:
            stem = key
            expected_files |= {stem + '_source.npz', stem + '_world.npz'}
            ws, ss = Path(fcdir) / (stem + '_world.npz'), Path(fcdir) / (stem + '_source.npz')
            if not (ws.is_file() and ss.is_file()):
                same = False; r['gate_i_files'] = False
            else:
                with np.load(ws) as a:
                    W = {k: a[k] for k in a.files}
                with np.load(ss) as a:
                    S = {k: a[k] for k in a.files}
                m = solid.to_mesh64()
                oid = np.empty(len(m.tri_verts), np.int64)
                for a_, b_, o_ in zip(m.run_index[:-1], m.run_index[1:], m.run_original_id):
                    oid[int(a_) // 3:int(b_) // 3] = int(o_)
                g2 = bool(np.array_equal(np.asarray(m.vert_properties)[:, :3], S['vertices'])
                          and np.array_equal(np.asarray(m.tri_verts, np.int64), S['faces']) and np.array_equal(oid, S['original_id']))
                shift = np.zeros(3) if g['shift'] is None else np.asarray(g['shift'], float)
                Xw = np.asarray(W['vertices'], float)
                off = np.ones(len(Xw), bool)
                for f in frames['faults']:
                    if row['interface_id'] in f['moved']:
                        off &= np.abs(FC.signed_other(f, f['id'] in (g['block'] or []), Xw)) > FAULT_POS_TOL_M
                dist = _set_distance(Xw[off], np.asarray(S['vertices'], float) + shift)
                fin = np.isfinite(dist)
                g3 = bool(len(dist) > 0 and fin.all() and float(dist.max()) <= FAULT_POS_TOL_M)
                mw = md.Manifold(md.Mesh64(np.ascontiguousarray(Xw), np.ascontiguousarray(np.asarray(W['faces']).astype(np.uint64))))
                vw = mw.volume()
                after = fr['replay']['volume_after_m3']
                g4 = abs(vw - after) / after <= FAULT_VOL_REL and abs(vw - row['surface_volume_m3']) / row['surface_volume_m3'] <= FAULT_VOL_REL
                roles = set(np.asarray(W['role']).astype(str).tolist())
                g5 = '' not in roles and roles <= FS.KNOWN_ROLES
                same &= g2 and g3 and g4 and g5
                r.update(gate_ii_capture_equals_source=g2, gate_iii_position=dict(ok=g3, off_plane_vertices=int(off.sum()),
                         unmatched_vertices=int((~fin).sum()), max_m=float(dist[fin].max()) if fin.any() else None,
                         shift_m=shift.tolist()),
                         gate_iv_volume=dict(ok=bool(g4), world_m3=vw, volume_after_m3=after, surface_volume_m3=row['surface_volume_m3']),
                         gate_v_roles=dict(ok=bool(g5), roles=sorted(roles)), world_sha256=FS.sha256(ws), source_sha256=FS.sha256(ss))
                world = (Xw, np.asarray(W['faces'], np.int64))
        ok &= same
        r['match'] = bool(same)
        rec[key] = r
        groups.append(dict(key=key, trace=g['trace'], tag=g['tag'], block=g['block'], shift=g['shift'], acted=acted, paths=indices,
                           interface_id=row['interface_id'], world=world))
    g1 = set(files) == expected_files
    ok &= g1
    rec['_gate_i_file_set'] = dict(ok=bool(g1), files=files, expected=sorted(expected_files))
    recorded_groups = {g['key'] for g in pg.values()}
    rec['_groups'] = dict(captured=len(captured), recorded=len(recorded_groups))
    ok &= len(captured) == len(recorded_groups)
    return bool(ok), rec, groups


MID_AABB_PAD_M = .50
FOOT_ITER = 30
FOOT_TOL_M = 1e-6


def reference_zone_offset(cav):
    """Split and recorded: both values must agree (fail closed)."""
    fs = cav.get('fold_split') or {}
    old = fs['zone_offset_m'] if fs.get('used') else cav['max_abs_offset_m']
    rec = cav.get('overlap_blend_zone')
    if rec is None:
        return old
    if fs.get('used') and rec['used_m'] != fs['zone_offset_m']:
        raise ValueError('overlap_blend_zone used_m %r differs from fold_split zone_offset_m %r' % (rec['used_m'], fs['zone_offset_m']))
    return rec['used_m']


def normal_foot(chart, zones, P):
    """(n(sigma) x dh) = 0 (Newton from the orthogonal foot on the locate panel, central differences). Returns q and |g| (m)
    after the iterations."""
    tau = P @ chart.dh
    q0, panel, _ = chart.locate(P)
    sig = chart.sigma[panel] + np.einsum('ij,ij->i', P - chart.Q[panel], chart.u[panel])
    top = float(chart.sigma[-1])

    def g(s_, idx):
        n_ = chart.normals(s_, zones)
        return np.einsum('ij,ij->i', P[idx] - chart.surface(np.c_[s_, tau[idx]]), np.cross(n_, chart.dh))
    h = 1e-5
    active = np.arange(len(P))
    for _ in range(FOOT_ITER):
        if not len(active):
            break
        s0 = np.clip(sig[active], 0., top)
        gv = g(s0, active)
        lo_, hi_ = np.clip(s0 - h, 0., top), np.clip(s0 + h, 0., top)
        dg = (g(hi_, active) - g(lo_, active)) / (hi_ - lo_)
        step = np.where(np.abs(dg) > 1e-12, gv / np.where(np.abs(dg) > 1e-12, dg, 1.), 0.)
        sig[active] = np.clip(s0 - step, 0., top)
        active = active[np.abs(step) >= 1e-12]
    return np.c_[sig, tau], np.abs(g(sig, np.arange(len(P))))


SPLIT_TOL_M = 5e-5
SPLIT_PASSES = 8


def _split_faces(F, parent, mid_of_edge, inv):
    """Conforming split: per face the marked local edges (v0v1, v1v2, v2v0) -> 2 / 3 / 4 triangles, orientation kept."""
    m = mid_of_edge[inv.reshape(-1, 3)]                      # (nf, 3), -1 = not split
    k = (m >= 0).sum(1)
    keep = k == 0
    outF, outP = [F[keep]], [parent[keep]]
    rows = np.arange(len(F))
    for kk in (1, 2, 3):
        sel = rows[k == kk]
        if not len(sel):
            continue
        mk = m[sel] >= 0
        if kk == 1:
            r = np.argmax(mk, 1)                              # the marked edge becomes e0
        elif kk == 2:
            r = (np.argmin(mk, 1) + 1) % 3                    # the unmarked edge becomes e2
        else:
            r = np.zeros(len(sel), np.int64)
        idx = (r[:, None] + np.arange(3)) % 3
        a, b, c = (F[sel[:, None], idx][:, j] for j in range(3))
        m0, m1, m2 = (m[sel[:, None], idx][:, j] for j in range(3))
        if kk == 1:
            tris = [np.c_[a, m0, c], np.c_[m0, b, c]]
        elif kk == 2:
            tris = [np.c_[m0, b, m1], np.c_[a, m0, m1], np.c_[a, m1, c]]
        else:
            tris = [np.c_[a, m0, m2], np.c_[m0, b, m1], np.c_[m2, m1, c], np.c_[m0, m1, m2]]
        outF += tris; outP += [parent[sel]] * len(tris)
    return np.concatenate(outF), np.concatenate(outP)


def refine_submesh(X, Q, S, M, N, F, parent, exact, tol=SPLIT_TOL_M, passes=SPLIT_PASSES):
    record = []
    for _ in range(passes):
        phi = S - M
        e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.ravel()
        a, b = uniq[:, 0], uniq[:, 1]
        L = np.linalg.norm(X[a] - X[b], axis=1)
        cand = np.flatnonzero((np.sign(phi[a]) != np.sign(phi[b])) | (np.minimum(np.abs(phi[a]), np.abs(phi[b])) <= L))
        if not len(cand):
            record.append(dict(candidates=0, split=0)); break
        Xm = .5 * (X[a[cand]] + X[b[cand]])
        qm, sm, mm, nm, ok = exact(Xm)
        err_phi = np.abs((sm - mm) - .5 * (phi[a[cand]] + phi[b[cand]]))
        err_q = np.abs(qm - .5 * (Q[a[cand]] + Q[b[cand]])).max(1)
        mark = ok & ((err_phi > tol) | (err_q > tol))
        record.append(dict(candidates=int(len(cand)), split=int(mark.sum()), foot_failed=int((~ok).sum()),
                           err_phi_p99=float(np.quantile(err_phi[ok], .99)) if ok.any() else None))
        if not mark.any():
            break
        sel = cand[mark]
        mid_of_edge = np.full(len(uniq), -1, np.int64)
        mid_of_edge[sel] = len(X) + np.arange(len(sel))
        X = np.concatenate([X, Xm[mark]]); Q = np.concatenate([Q, qm[mark]]); S = np.concatenate([S, sm[mark]])
        M = np.concatenate([M, mm[mark]]); N = np.concatenate([N, nm[mark]])
        F, parent = _split_faces(F, parent, mid_of_edge, inv)
    return X, Q, S, M, N, F, parent, record


def _rho_full(relief, chart):
    """wall relief at chart points exactly as frgb.network_panels._surface_cavity: the captured callable"""
    return lambda q_: np.asarray(relief(chart.surface(q_)), float)


def extension_stations(pp, st_all, ends, chart, to_chart, shift, qs_, res_, FS, strips=None, L_rec=None):
    """L must equal L_rec[end] bit for bit); without a strip the captured extended piece pp gives them (virtual row, nu,
    depth_last, margin). Returns ({name: array} for the mid file, rows [pseudo, path, end]). Fail closed: piece rows =
    stations + ends; end station chart point = piece row (and = strip p_end); piece virtual row = p_end + L t2."""
    out, rows = {}, []
    n_st = len(st_all['points'])
    ends = sorted(int(e) for e in ends)
    if len(pp.p) != n_st + len(ends):
        raise ValueError('path %d: extended piece has %d rows, stations %d + ends %d' % (pp.path_index, len(pp.p), n_st, len(ends)))
    off0 = 1 if 0 in ends else 0
    for end in ends:
        er, vr, si = (off0, 0, 0) if end == 0 else (off0 + n_st - 1, off0 + n_st, n_st - 1)
        if float(np.linalg.norm(np.asarray(qs_[si], float) - np.asarray(pp.p[er], float))) > 1e-9:
            raise ValueError('path %d end %d: end station chart point differs from the piece row' % (pp.path_index, end))
        s = (strips or {}).get(end)
        if s is not None:
            if L_rec is None or float(s['L']) != float(L_rec[end]):
                raise ValueError('path %d end %d: strip L %r differs from build.json L_m %r' % (pp.path_index, end, s['L'],
                                                                                               None if L_rec is None else L_rec.get(end)))
            p_end, t2, nu2 = (np.asarray(s[k], float) for k in ('p_end', 't2', 'nu2'))
            if float(np.linalg.norm(p_end - np.asarray(qs_[si], float))) > 1e-9:
                raise ValueError('path %d end %d: strip p_end differs from the end station' % (pp.path_index, end))
            qv = p_end + float(s['L']) * t2
            if float(np.linalg.norm(qv - np.asarray(pp.p[vr], float))) > 1e-9:
                raise ValueError('path %d end %d: piece virtual row differs from p_end + L t2' % (pp.path_index, end))
            depth, margin, L, source = float(s['depth']), float(s['margin']), float(s['L']), 'strip'
        else:
            p_end = np.asarray(pp.p[er], float); qv = np.asarray(pp.p[vr], float)
            nu2 = np.asarray(pp.nu[0 if end == 0 else -1], float)
            L = float(np.linalg.norm(qv - p_end)); t2 = (qv - p_end) / L
            depth, margin, source = float(pp.depth_last[er]), float(pp.margin), 'piece'
        Xv = chart.surface(qv[None])[0] + (0. if shift is None else np.asarray(shift, float))
        _, _, res_v = chart.locate(to_chart(Xv[None]))
        pid = FS.pseudo_index(pp.path_index, end)
        out['path_%d_q' % pid] = np.array([np.asarray(qs_[si], float), qv])
        out['path_%d_dir' % pid] = np.array([nu2, nu2])
        out['path_%d_locate_residual' % pid] = np.array([float(res_[si]), float(res_v[0])])
        for k, v in st_all.items():
            v = np.asarray(v)
            out['ext_%d_%s' % (pid, k)] = np.concatenate([v[si:si + 1], v[si:si + 1]])
        X_e = np.asarray(st_all['points'][si], float)
        out['ext_%d_points' % pid] = np.array([X_e, Xv])
        out['ext_%d_arc_m' % pid] = np.array([0., float(np.linalg.norm(Xv - X_e))])
        # metadata under another prefix ('ext_<pid>_' arrays are the pseudo stations)
        out['extm_%d_p_end' % pid] = p_end; out['extm_%d_t2' % pid] = t2; out['extm_%d_nu2' % pid] = nu2
        out['extm_%d_depth_margin_L' % pid] = np.array([depth, margin, L])
        out['extm_%d_source' % pid] = np.array(source)
        rows.append([pid, int(pp.path_index), end])
    return out, rows


def independent_traces(V, source, ext_rows, ext_ends, stations, classifiers, fault, FC, FS, chart, zones, to_chart, m_of):
    """R1.2: -margin <= a <= depth;"""
    with np.load(source / 'source_surface.npz') as a_:
        Fh = np.asarray(a_['faces'], np.int64)
    fby = {f['id']: f for f in fault['frames']['faults']}
    ends_by = {(int(p_), int(e['end'])): e for p_, es in (ext_ends or {}).items() for e in es}
    rec = {}
    for pid, path, end in ext_rows:
        e = ends_by[(int(path), int(end))]
        so_fn = lambda X, e=e: np.asarray(FC.signed_other(fby[e['fault']], e['moved'], np.asarray(X, float)), float)

        def phi_fn(P_):
            P_ = np.asarray(P_, float).reshape(-1, 3)
            if not len(P_):
                return np.zeros(0), np.zeros((0, 2))
            q_, g_ = normal_foot(chart, zones, to_chart(P_))
            n_ = chart.normals(q_[:, 0], zones)
            ph = np.einsum('ij,ij->i', to_chart(P_) - chart.surface(q_), n_) - m_of(q_)
            return np.where(g_ <= FOOT_TOL_M, ph, np.nan), q_
        Xe = np.asarray(stations['ext_%d_points' % pid][0], float)
        depth, margin, L = (float(x) for x in stations['extm_%d_depth_margin_L' % pid])
        half = L + float(np.max(stations['ext_%d_width' % pid][0])) + .1
        fsel = np.flatnonzero((np.abs(V[Fh] - Xe) <= half).all(2).any(1))
        pts = FS.indep_trace(V, Fh[fsel], so_fn, phi_fn)
        ph, q = phi_fn(pts)
        p_end, t2, nu2 = (np.asarray(stations['extm_%d_%s' % (pid, k)], float) for k in ('p_end', 't2', 'nu2'))
        lam, a = (q - p_end) @ t2, (q - p_end) @ nu2
        keep = (np.isfinite(ph) & (lam > FS.EXT_EXCLUDE_M) & (lam <= L) & (a >= -margin) & (a <= depth)
                & (so_fn(pts) <= -FS.EXT_TRIM_M)) if len(pts) else np.zeros(0, bool)
        stations['indep_%d_points' % pid] = pts[keep]
        stations['indep_%d_lambda_a' % pid] = np.c_[lam, a][keep] if len(pts) else np.zeros((0, 2))
        cls = classifiers.get((int(path), int(end)))
        r_ = dict(faces_in_box=int(len(fsel)), contour_points=int(len(pts)), points=int(keep.sum()))
        if cls is not None:
            mouth = (cls(pts[keep])[1] == 0) if keep.any() else np.zeros(0, bool)
            stations['indep_%d_mouth' % pid] = mouth
            r_['mouth'] = int(mouth.sum())
        rec[str(pid)] = r_
    return rec


def clip_own_side(V, Fh, row_, ids, act_f, tol, chart, zones, to_chart, m_ex, FC, FS, base, stations, ext_rows):
    """Returns the sub-mesh inputs extended by the clipped triangles and the record (with '_parents')."""
    if len(act_f) != 1:
        raise ValueError('amendment h: %d faults act on one group (unsupported cases stop before stage A)' % len(act_f))
    f_, side = act_f[0]
    X0, Q0, S0, M0, N0, F0, P0 = base
    cov_v = row_[Fh] >= 0
    cand = np.flatnonzero(cov_v.any(1) & ~cov_v.all(1))
    Fv = Fh[cand]
    so = np.asarray(FC.signed_other(f_, side, V[Fv.reshape(-1)]), float).reshape(-1, 3)
    cl = FS.clip_straddling(Fv, cov_v[cand], so, tol)
    E, T = cl['edges'], cl['t']
    P = V[E[:, 0]] + T[:, None] * (V[E[:, 1]] - V[E[:, 0]]) if len(E) else np.zeros((0, 3))
    if len(P):
        q, g = normal_foot(chart, zones, to_chart(P))
        nrm = chart.normals(q[:, 0], zones)
        sv = np.einsum('ij,ij->i', to_chart(P) - chart.surface(q), nrm)
        mv = m_ex(q)
        conv, covd = g <= FOOT_TOL_M, np.isfinite(mv)
    else:
        q, nrm, sv, mv = np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), np.zeros(0)
        conv = covd = np.zeros(0, bool)
    tris, trow = cl['tris'], cl['tri_row']
    newp = -1 - tris                                  # >= 0 where the entry is a new point
    bad_nc = np.zeros(len(Fv), bool); bad_cv = np.zeros(len(Fv), bool)
    for c in range(3):
        isn = newp[:, c] >= 0
        j = newp[isn, c]
        np.logical_or.at(bad_nc, trow[isn], ~conv[j])
        np.logical_or.at(bad_cv, trow[isn], conv[j] & ~covd[j])
    reject = bad_nc | bad_cv
    keep = ~reject[trow]
    used = np.unique(newp[keep][newp[keep] >= 0])
    remap = np.full(len(P), -1, np.int64); remap[used] = len(X0) + np.arange(len(used))
    Fadd = np.where(tris[keep] >= 0, row_[np.maximum(tris[keep], 0)], remap[np.maximum(newp[keep], 0)])
    if len(Fadd) and (Fadd < 0).any():
        raise ValueError('amendment h: a clipped triangle refers to an uncovered vertex')
    parents = cand[trow[keep]]
    # per fault end (H5): the end's check box = end station +- (L + w + 0.1 m)
    edge = np.linalg.norm(V[Fv[cl['rows']]] - V[np.roll(Fv[cl['rows']], 1, axis=1)], axis=2) if len(cl['rows']) else np.zeros((0, 3))
    cen = V[Fv[cl['rows']]].mean(1) if len(cl['rows']) else np.zeros((0, 3))
    owner_end = np.full(len(cl['rows']), -1, np.int64)
    for pid, _path, _end in ext_rows:
        Xe = np.asarray(stations['ext_%d_points' % pid][0], float)
        half = float(stations['extm_%d_depth_margin_L' % pid][2]) + float(np.max(stations['ext_%d_width' % pid][0])) + .1
        inbox = (np.abs(cen - Xe) <= half).all(1) & (owner_end < 0)
        owner_end[inbox] = pid
    per_end = {}
    for key in sorted(set(owner_end.tolist())):
        sel = owner_end == key; r_ = cl['rows'][sel]
        per_end['unassigned' if key < 0 else str(key)] = dict(clipped=int((~reject[r_]).sum()), rejected_not_converged=int(bad_nc[r_].sum()),
                                                               rejected_not_covered=int((bad_cv[r_] & ~bad_nc[r_]).sum()))
    rej_trace = None
    if reject.any() and hasattr(FS, 'indep_trace'):
        so_fn = lambda X: np.asarray(FC.signed_other(f_, side, np.asarray(X, float)), float)

        def phi_fn(P_):
            P_ = np.asarray(P_, float).reshape(-1, 3)
            if not len(P_):
                return np.zeros(0), np.zeros((0, 2))
            q_, g_ = normal_foot(chart, zones, to_chart(P_))
            ph = np.einsum('ij,ij->i', to_chart(P_) - chart.surface(q_), chart.normals(q_[:, 0], zones)) - m_ex(q_)
            return np.where(g_ <= FOOT_TOL_M, ph, np.nan), q_
        rej_trace = int(len(FS.indep_trace(V, Fv[reject], so_fn, phi_fn)))
    rec = dict(rule='criterion_trace_labels_r4_faults_amendment_h (49668831)', fault=f_['id'], straddling_faces=int(len(cl['rows'])),
               rejected_faces=[int(x) for x in cand[np.flatnonzero(reject)]], rejected_faces_trace_points=rej_trace,
               triangles_added=int(keep.sum()), new_vertices=int(len(used)), rejected_not_converged=int(bad_nc.sum()),
               rejected_not_covered=int((bad_cv & ~bad_nc).sum()), max_edge_m=float(edge.max()) if edge.size else None,
               per_end=per_end, _parents=np.unique(parents))
    out = (np.concatenate([X0, P[used]]), np.concatenate([Q0, q[used]]), np.concatenate([S0, sv[used]]),
           np.concatenate([M0, mv[used]]), np.concatenate([N0, nrm[used]]), np.concatenate([F0, Fadd.reshape(-1, 3)]),
           np.concatenate([P0, parents]))
    return out + (rec,)


def export_midsurface(captured, context, source, trace_of, dest, chunk=200_000, case_dir=None, groups=None, fault=None, FC=None,
                      ext_ends=None, FS=None, structure=None):
    from frgb.network_panels import envelope
    with np.load(source / 'source_surface.npz') as a:
        V = np.asarray(a['vertices'], float)
    record = {}
    for gi, ((indices, solid), ctx) in enumerate(zip(captured, context)):
        trace = trace_of[indices[0]]
        grp = None if groups is None else groups[gi]
        shift = None if grp is None or grp['shift'] is None else np.asarray(grp['shift'], float)
        to_chart = (lambda X_: X_) if shift is None else (lambda X_, s_=shift: np.asarray(X_, float) - s_)
        if grp is not None:
            trace = grp['key']
        chart, pieces, kw, cav = ctx['chart'], ctx['pieces'], ctx['kw'], ctx['checks']
        flat = [pc for p in pieces for pc in p.pieces()]
        rho_of = None
        if kw.get('wall_relief') is not None:      # as frgb.network_panels._surface_cavity
            relief = kw['wall_relief']
            rho_full = _rho_full(relief, chart)
            rho_of = rho_full
            if kw.get('wall_relief_tip_m') is not None:
                tip = kw['wall_relief_tip_m']
                from frgb.network_panels import taper_factor

                def rho_of(q_, rho_full=rho_full, pieces=pieces, tip=tip):
                    return rho_full(q_) * taper_factor(pieces, np.atleast_2d(q_), tip)
        zone_offset = reference_zone_offset(cav)
        zones = chart.fold_zones(zone_offset)
        if grp is not None and grp.get('world') is not None:
            cv = grp['world'][0]                          # the post-replay (world) cavity
        else:
            mesh = solid.to_mesh64()
            cv = np.asarray(mesh.vert_properties)[:, :3]
            if shift is not None:
                cv = cv + shift
        lo, hi = cv.min(0) - MID_AABB_PAD_M, cv.max(0) + MID_AABB_PAD_M
        cand = np.flatnonzero(np.all((V >= lo) & (V <= hi), axis=1))
        side_dropped = 0
        act_f = []
        if grp is not None and grp['acted']:
            act_f = [(f, f['id'] in (grp['block'] or [])) for f in fault['frames']['faults'] if grp['interface_id'] in f['moved']]
        if grp is not None and grp['acted']:
            keep_ = np.ones(len(cand), bool)
            for f in fault['frames']['faults']:
                if grp['interface_id'] in f['moved']:
                    keep_ &= FC.signed_other(f, f['id'] in (grp['block'] or []), V[cand]) <= fault['tol_m']
            side_dropped = int((~keep_).sum())
            cand = cand[keep_]
        ids, ms, ns, ss, sg, tg = [], [], [], [], [], []
        nonconv = 0
        for part in np.array_split(cand, max(1, len(cand) // chunk + 1)):
            if not len(part):
                continue
            q, gres = normal_foot(chart, zones, to_chart(V[part]))
            ok = gres <= FOOT_TOL_M
            nonconv += int((~ok).sum())
            part, q = part[ok], q[ok]
            if not len(part):
                continue
            s = np.einsum('ij,ij->i', to_chart(V[part]) - chart.surface(q), chart.normals(q[:, 0], zones))
            L, _, cL = envelope(flat, q, 'L'); R, _, cR = envelope(flat, q, 'R')
            cov = (cL > 0) & (cR > 0)
            part, q, s, L, R = part[cov], q[cov], s[cov], L[cov], R[cov]
            m = .5 * (L + R) + (rho_of(q) if rho_of is not None else 0.)
            ids.append(part); ms.append(m); ns.append(chart.normals(q[:, 0], zones)); ss.append(s); sg.append(q[:, 0]); tg.append(q[:, 1])
        cat = lambda x, shape: np.concatenate(x) if x else np.zeros(shape)
        ids, m, n = cat(ids, (0,)).astype(np.int64), cat(ms, (0,)), cat(ns, (0, 3))
        s, sigma, tau = cat(ss, (0,)), cat(sg, (0,)), cat(tg, (0,))
        # inward vectors as chart directions, for the unrolled projection
        stations = {}
        station_gate = {}
        ext_rows = []
        strips = {}
        classifiers = {}
        for _gi, inputs in (structure or ()):
            for inp in inputs:
                if inp.get('strip') is not None:
                    strips[(int(inp['record']['path']), int(inp['record']['end']))] = inp['strip']
                if inp.get('classify') is not None:
                    classifiers[(int(inp['record']['path']), int(inp['record']['end']))] = inp['classify']
        if case_dir is not None:
            for pp in pieces:
                with np.load(Path(case_dir) / ('path_%02d_stations.npz' % pp.path_index)) as a:
                    pts, inw = np.asarray(a['points'], float), np.asarray(a['inward'], float)
                if grp is not None and grp['acted']:
                    mshift, _ = FC.path_frame(fault['frames'], grp['interface_id'], pts)
                    agree = (mshift is None and shift is None) or (mshift is not None and shift is not None
                                                                   and float(np.abs(mshift - shift).max()) <= STATION_SHIFT_TOL_M)
                    if not agree:
                        raise ValueError('path %d: module shift %s differs from build.json %s' % (pp.path_index, mshift, shift))
                qs_, pan_, res_ = chart.locate(to_chart(pts))
                if grp is not None:
                    station_gate[int(pp.path_index)] = float(res_.max())
                    if float(res_.max()) > STATION_RESIDUAL_M:
                        raise ValueError('path %d: station locate residual %.3e m > %.0e m after the shift'
                                         % (pp.path_index, float(res_.max()), STATION_RESIDUAL_M))
                d2 = chart.vector_2d(inw, pan_)
                stations['path_%d_q' % pp.path_index] = qs_
                stations['path_%d_dir' % pp.path_index] = d2 / np.linalg.norm(d2, axis=1)[:, None]
                stations['path_%d_locate_residual' % pp.path_index] = res_
                ends_ = {e['end']: e['L_m'] for e in (ext_ends or {}).get(int(pp.path_index), ())}
                if ends_:
                    with np.load(Path(case_dir) / ('path_%02d_stations.npz' % pp.path_index)) as a:
                        st_all = {k: np.asarray(a[k]) for k in a.files}
                    ext, rows_ = extension_stations(pp, st_all, list(ends_), chart, to_chart, shift, qs_, res_, FS,
                                                    strips={e: strips[(int(pp.path_index), e)] for e in ends_
                                                            if (int(pp.path_index), e) in strips} or None, L_rec=ends_)
                    stations.update(ext); ext_rows += rows_
        if ext_rows:
            stations['ext_index'] = np.array(ext_rows, np.int64)
        sub = {}
        clip_rec = None
        if len(ids):
            with np.load(source / 'source_surface.npz') as a_:
                Fh = np.asarray(a_['faces'], np.int64)
            row_ = np.full(len(V), -1, np.int64); row_[ids] = np.arange(len(ids))
            cov_f = np.flatnonzero((row_[Fh] >= 0).all(1))
            _, _, m_ex = _mid_context(solid, ctx)

            def exact(P_):
                q_, g_ = normal_foot(chart, zones, to_chart(P_))
                n_ = chart.normals(q_[:, 0], zones)
                s_ = np.einsum('ij,ij->i', to_chart(P_) - chart.surface(q_), n_)
                m_ = m_ex(q_)
                return q_, s_, m_, n_, (g_ <= FOOT_TOL_M) & np.isfinite(m_)
            X0, Q0, S0, M0, N0, F0, P0 = V[ids], np.c_[sigma, tau], s, m, n, row_[Fh[cov_f]], cov_f
            if act_f:
                X0, Q0, S0, M0, N0, F0, P0, clip_rec = clip_own_side(V, Fh, row_, ids, act_f, fault['tol_m'], chart, zones, to_chart,
                                                                      m_ex, FC, FS, (X0, Q0, S0, M0, N0, F0, P0), stations, ext_rows)
                sub['clip_parent_faces'] = np.asarray(clip_rec.pop('_parents'), np.int64)
            Xs, Qs, Ss, Ms, Ns, Fs, Ps, rec = refine_submesh(X0, Q0, S0, M0, N0, F0, P0, exact)
            sub.update(sub_X=Xs, sub_U=np.c_[Qs, Ss - Ms], sub_n=Ns, sub_sigma=Qs[:, 0], sub_faces=Fs, sub_parent=Ps)
            split_record = dict(passes=rec, vertices=int(len(Xs)), faces=int(len(Fs)), covered_faces=int(len(cov_f)))
        else:
            split_record = None
        indep_rec = None
        if ext_rows and act_f:
            indep_rec = independent_traces(V, source, ext_rows, ext_ends, stations, classifiers, fault, FC, FS, chart, zones,
                                           to_chart, _mid_context(solid, ctx)[2])
        extra = {} if grp is None else dict(group_key=np.array(grp['key']), group_tag=np.array(grp['tag'] or ''),
                                            group_shift=np.zeros(3) if shift is None else shift, group_trace=np.array(grp['trace']))
        np.savez_compressed(dest / ('mid_%s.npz' % trace), vertex_ids=ids, m=m, n=n, D=m[:, None] * n, s=s, sigma=sigma, tau=tau, **sub,
                            fold_sigma=np.array([z['sigma'] for z in zones]), fold_half_width=np.array([z['half_width_m'] for z in zones]),
                            chart_Q=chart.Q, chart_u=chart.u, chart_dh=chart.dh, chart_sigma=chart.sigma, **stations, **extra)
        record[trace] = dict(paths=indices, aabb_vertices=int(len(cand)), covered_vertices=int(len(ids)), foot_not_converged=nonconv, abs_s_max_covered_m=float(np.abs(s).max()) if len(s) else None,
                             zone_offset_m=zone_offset, fold_zones=len(zones), wall_relief=kw.get('wall_relief') is not None,
                             tip_taper_m=kw.get('wall_relief_tip_m'),
                             m_abs_p50_p99_max=[float(x) for x in np.quantile(np.abs(m), [.5, .99, 1.])] if len(m) else None,
                             split=split_record)
        if clip_rec is not None:
            record[trace]['clip'] = clip_rec
        if indep_rec is not None:
            record[trace]['independent_trace'] = indep_rec
        if grp is not None:
            if ext_rows:
                record[trace]['extension'] = [dict(pseudo=r[0], path=r[1], end=r[2], source=str(stations['extm_%d_source' % r[0]]))
                                              for r in ext_rows]
            record[trace].update(group=dict(trace=grp['trace'], tag=grp['tag'], shift_m=None if shift is None else shift.tolist(),
                                            acted=grp['acted'], other_side_vertices_dropped=side_dropped,
                                            station_locate_residual_max_m=station_gate))
    return record


WALL_FID_TOL_M = 1e-6
WALL_FID_FOOT_M = 1e-9      # F2: foot residual |g| of an evaluable vertex
WALL_FID_EXCL_CAP = .05     # F2: share of wall vertices excluded for touching a face of another construction
WALL_FID_NONEVAL_CAP = .001 # F2: share of checked vertices that cannot be evaluated
WALL_FID_EPS_M = 1e-9       # F2: the four neighbour points (sigma, tau +- eps) of a coverage-boundary vertex
WALL_FID_BEYOND_STOP_M = .5


def network_panels_pin(code, FS, path=None):
    if code != 'current':
        return None
    sha = FS.sha256(P1 / 'frgb/network_panels.py' if path is None else path)
    return dict(ok=sha in (FS.DEPLOYED_NETWORK_PANELS_SHA256, FS.INTERNAL_NETWORK_PANELS_SHA256), sha256=sha, expected=FS.DEPLOYED_NETWORK_PANELS_SHA256)


def _lookup(keys, vals, x, default):
    o = np.argsort(keys); k, v = keys[o], vals[o]
    i = np.clip(np.searchsorted(k, x), 0, max(len(k) - 1, 0))
    hit = (len(k) > 0) & (k[i] == x) if len(k) else np.zeros(len(x), bool)
    return np.where(hit, v[i] if len(k) else default, default)


def overlap_blend_check(captured, context, new):
    """Any difference: ok false (stage A stops)"""
    rec_of = {int(p_['index']): (p_.get('construction_checks') or {}).get('overlap_blend') for p_ in new['paths']}
    out, ok = dict(groups=[], rule='criterion_trace_labels_overlap_blend_c1_r1 R2'), True
    for (indices, _solid), ctx in zip(captured, context):
        flags = [getattr(pp, 'overlap_blend', None) for pp in ctx['pieces']]
        flag = flags[0] if flags else None
        same = all(f == flag for f in flags)
        bad = []
        for i in indices:
            r = rec_of.get(int(i))
            good = (r is None) if flag is None else (isinstance(r, dict) and r.get('gate') is True
                                                     and all(k in r and r[k] == v for k, v in flag.items()))
            if not good:
                bad.append(int(i))
        g_ok = bool(same and not bad)
        out['groups'].append(dict(paths=[int(i) for i in indices], flag=None if flag is None else dict(flag), same_flag=bool(same),
                                  record_mismatch_paths=bad, ok=g_ok))
        ok &= g_ok
    out['ok'] = bool(ok)
    return out


def foot_f2(chart, zones, X):
    """Every other point keeps normal_foot's q, |g| bit for bit. Returns q, |g| and the F2E2 record (beyond_ends,
    beyond_ends_converged, beyond_ends_max_overshoot_m)."""
    q, g = normal_foot(chart, zones, X)
    top = float(chart.sigma[-1])
    sel = np.flatnonzero((g > WALL_FID_FOOT_M) & ((q[:, 0] <= 1e-12) | (q[:, 0] >= top - 1e-12)))
    rec = dict(beyond_ends=int(len(sel)), beyond_ends_converged=0, beyond_ends_max_overshoot_m=0.)
    if not len(sel):
        return q, g, rec
    P, tau, sig = X[sel], q[sel, 1], q[sel, 0].copy()

    def g_(s_, idx):
        n_ = chart.normals(s_, zones)
        return np.einsum('ij,ij->i', P[idx] - chart.surface(np.c_[s_, tau[idx]]), np.cross(n_, chart.dh))
    h = 1e-5
    active = np.arange(len(sel))
    for _ in range(FOOT_ITER):
        if not len(active):
            break
        s0 = sig[active]
        gv = g_(s0, active)
        dg = (g_(s0 + h, active) - g_(s0 - h, active)) / (2 * h)
        step = np.where(np.abs(dg) > 1e-12, gv / np.where(np.abs(dg) > 1e-12, dg, 1.), 0.)
        sig[active] = s0 - step
        active = active[np.abs(step) >= 1e-12]
    g2 = np.abs(g_(sig, np.arange(len(sel))))
    q, g = q.copy(), g.copy()
    q[sel, 0], g[sel] = sig, g2
    rec.update(beyond_ends_converged=int((g2 <= WALL_FID_FOOT_M).sum()),
               beyond_ends_max_overshoot_m=float(np.max(np.maximum(-sig, sig - top))))
    return q, g, rec


def wall_fidelity(captured, context):
    """Excluded and counted: vertices touching faces of another role, and vertices touching faces of another construction
    (share <= WALL_FID_EXCL_CAP). Not evaluable (|g| > WALL_FID_FOOT_M, or no coverage at q nor at a neighbour point):
    share <= WALL_FID_NONEVAL_CAP. A vertex whose envelope (values, owners, counts) differs between q and its four
    neighbour points is a boundary vertex: it passes if any of the five points does. One failing vertex stops."""
    from frgb.network_panels import envelope
    out, ok = dict(groups=[], rule='criterion_trace_labels_r4_faults_stage_a_fidelity_r1 F2'), True
    for (indices, solid), ctx in zip(captured, context):
        t0 = time.time()
        m = solid.to_mesh64()
        V = np.asarray(m.vert_properties)[:, :3].astype(float); F = np.asarray(m.tri_verts, np.int64)
        fid = np.asarray(m.face_id, np.int64)
        keys, roles, cons = [], [], []
        for tab in (ctx.get('tables') or {}).values():
            for k_, r_ in tab.items():
                keys.append(int(k_)); roles.append(str(r_.get('role', ''))); cons.append(str(r_.get('construction', '')))
        keys = np.asarray(keys, np.int64)
        role = _lookup(keys, np.asarray(roles, dtype=object), fid, '')
        con = _lookup(keys, np.asarray(cons, dtype=object), fid, '')
        nv = len(V)

        def mark(sel):
            a = np.zeros(nv, bool); a[F[sel].ravel()] = True
            return a
        isL, isR = role == 'left_wall', role == 'right_wall'
        tL, tR = mark(isL), mark(isR)
        t_role = mark(~(isL | isR)); t_cons = mark(con != 'panel_r3')
        wall_v = tL | tR
        checked = wall_v & ~t_role & ~t_cons
        n_wall, n_chk = int(wall_v.sum()), int(checked.sum())
        excl_cons = int((wall_v & t_cons).sum())
        rec = dict(paths=list(indices), faces=int(len(F)), unmapped_faces=int((role == '').sum()), wall_vertices=n_wall,
                   checked=n_chk, excluded_other_role=int((wall_v & t_role & ~t_cons).sum()), excluded_other_construction=excl_cons,
                   excluded_other_construction_share=excl_cons / max(n_wall, 1))
        idx = np.flatnonzero(checked)
        chart, zones, m_of = _mid_context(solid, ctx)
        flat = [pc for p in ctx['pieces'] for pc in p.pieces()]
        X = V[idx]
        q, g, beyond = foot_f2(chart, zones, X)

        def dev_at(q_):
            n_ = chart.normals(q_[:, 0], zones)
            s_ = np.einsum('ij,ij->i', X_ - chart.surface(q_), n_)
            m_, Lw, Rw = m_of(q_, walls=True)
            d_ = np.where(L_, np.abs(s_ - Lw), 0.)
            d_ = np.maximum(d_, np.where(R_, np.abs(s_ - Rw), 0.))
            return np.where(np.isfinite(m_), d_, np.nan)

        def env_at(q_):
            L, oL, cL = envelope(flat, q_, 'L'); R, oR, cR = envelope(flat, q_, 'R')
            return np.c_[L, R, oL, oR, cL, cR]
        X_, L_, R_ = X, tL[idx], tR[idx]
        foot_ok = g <= WALL_FID_FOOT_M
        d0 = dev_at(q)
        bad = foot_ok & ~(d0 <= WALL_FID_TOL_M)          # failing or without coverage at q
        boundary_pass, boundary = 0, 0
        if bad.any():
            b = np.flatnonzero(bad)
            X_, L_, R_ = X[b], tL[idx][b], tR[idx][b]
            e0 = env_at(q[b])
            passed = np.zeros(len(b), bool); is_bd = np.zeros(len(b), bool)
            for dq in ((WALL_FID_EPS_M, 0.), (-WALL_FID_EPS_M, 0.), (0., WALL_FID_EPS_M), (0., -WALL_FID_EPS_M)):
                qn = q[b] + np.asarray(dq)
                en = env_at(qn)
                is_bd |= ~np.all(np.isclose(en, e0, rtol=0., atol=1e-12) | (np.isnan(en) & np.isnan(e0)), axis=1)
                passed |= dev_at(qn) <= WALL_FID_TOL_M
            ok_b = is_bd & passed
            boundary, boundary_pass = int(is_bd.sum()), int(ok_b.sum())
            bad[b[ok_b]] = False
            no_cov = int((~np.isfinite(d0[b]) & ~ok_b).sum())     # no coverage at q nor a passing neighbour: not evaluable
        else:
            no_cov = 0
        noneval = int((~foot_ok).sum()) + no_cov
        fails = bad & np.isfinite(d0)                    # a finite deviation above the tolerance, not passed as a boundary vertex
        fin = np.isfinite(d0) & foot_ok
        worst = int(np.nanargmax(np.where(fin, d0, -1.))) if fin.any() else None
        rec.update(evaluable=int(n_chk - noneval), not_evaluable=noneval, not_evaluable_share=noneval / max(n_chk, 1),
                   foot_not_converged=int((~foot_ok).sum()), boundary_vertices=boundary, boundary_passed=boundary_pass,
                   failing=int(fails.sum()), max_dev_m=float(np.nanmax(np.where(fin, d0, np.nan))) if fin.any() else None,
                   max_dev_q=[float(x) for x in q[worst]] if worst is not None else None, seconds=round(time.time() - t0, 1))
        rec.update(beyond, beyond_ends_stop=bool(beyond['beyond_ends_max_overshoot_m'] > WALL_FID_BEYOND_STOP_M))   # F2E2
        rec['ok'] = bool(rec['failing'] == 0 and rec['excluded_other_construction_share'] <= WALL_FID_EXCL_CAP
                         and not rec['beyond_ends_stop']
                         and rec['not_evaluable_share'] <= WALL_FID_NONEVAL_CAP and rec['unmapped_faces'] == 0)
        ok &= rec['ok']
        out['groups'].append(rec)
    out['ok'] = bool(ok)
    return out


def _mid_context(solid, ctx):
    """(chart, zones, m_of) of one fracture exactly as export_midsurface builds them."""
    from frgb.network_panels import envelope
    chart, pieces, kw, cav = ctx['chart'], ctx['pieces'], ctx['kw'], ctx['checks']
    flat = [pc for p in pieces for pc in p.pieces()]
    rho_of = None
    if kw.get('wall_relief') is not None:
        relief = kw['wall_relief']
        rho_full = _rho_full(relief, chart)
        rho_of = rho_full
        if kw.get('wall_relief_tip_m') is not None:
            tip = kw['wall_relief_tip_m']
            from frgb.network_panels import taper_factor

            def rho_of(q_, rho_full=rho_full, pieces=pieces, tip=tip):
                return rho_full(q_) * taper_factor(pieces, np.atleast_2d(q_), tip)
    zones = chart.fold_zones(reference_zone_offset(cav))

    def m_of(q_, walls=False):
        L, _, cL = envelope(flat, q_, 'L'); R, _, cR = envelope(flat, q_, 'R')
        r_ = rho_of(q_) if rho_of is not None else 0.
        m_ = .5 * (L + R) + r_
        if walls:
            return np.where((cL > 0) & (cR > 0), m_, np.nan), L + r_, R + r_
        return np.where((cL > 0) & (cR > 0), m_, np.nan)
    return chart, zones, m_of


def _group_points(gi, indices, trace_of, tr, pa, groups):
    """(selection, to_chart) of the points of captured solid gi: by trace id (fault-free), by path index (faulted groups)"""
    if groups is None:
        return np.flatnonzero(tr == int(trace_of[indices[0]].split('-')[1])), (lambda X_: X_)
    if pa is None:
        raise ValueError('a faulted case needs the path index of every point')
    shift = groups[gi]['shift']
    return np.flatnonzero(np.isin(pa, indices)), ((lambda X_: X_) if shift is None else (lambda X_, s_=np.asarray(shift, float): X_ - s_))


def evaluate_points(captured, context, trace_of, path, groups=None):
    """Diagnosis helper: exact phi = s - m (signed distance of a point from the cavity mid-surface along n) at given points."""
    with np.load(path) as a:
        P, tr = np.asarray(a['point'], float), np.asarray(a['trace'], np.int64)
        pa = np.asarray(a['path'], np.int64) if 'path' in a.files else None
    phi = np.full(len(P), np.nan); lo_w = np.full(len(P), np.nan); hi_w = np.full(len(P), np.nan)
    for gi, ((indices, solid), ctx) in enumerate(zip(captured, context)):
        sel, to_chart = _group_points(gi, indices, trace_of, tr, pa, groups)
        if not len(sel):
            continue
        chart, zones, m_of = _mid_context(solid, ctx)
        q, g = normal_foot(chart, zones, to_chart(P[sel]))
        s = np.einsum('ij,ij->i', to_chart(P[sel]) - chart.surface(q), chart.normals(q[:, 0], zones))
        m_, Lw, Rw = m_of(q, walls=True)
        phi[sel] = np.where(g <= FOOT_TOL_M, s - m_, np.nan)
        lo_w[sel] = Lw - s; hi_w[sel] = Rw - s          # analytic wall offsets along n, relative to the point
    out = path.with_name(path.stem + '_phi.npz')
    np.savez_compressed(out, phi=phi, wall_lo=lo_w, wall_hi=hi_w)
    return dict(points=int(len(P)), evaluated=int(np.isfinite(phi).sum()), output=str(out))


REFINE_H = 2e-4
REFINE_TOL = 1e-5
REFINE_STEPS = 12
REFINE_CAP = .01


def _vertex_faces(F, nv):
    order = np.argsort(F.ravel(), kind='stable')
    counts = np.bincount(F.ravel(), minlength=nv)
    return order // 3, np.r_[0, np.cumsum(counts)]


def _snap(P, f, V, F, vf, start):
    """Nearest of the faces sharing a vertex with f (plane projection, barycentric clamp): (point, face)."""
    out_p, out_f = P.copy(), f.copy()
    for k in range(len(P)):
        cand = np.unique(np.concatenate([vf[start[v]:start[v + 1]] for v in F[f[k]]]))
        tri = V[F[cand]]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1)[:, None]
        r = P[k] - tri[:, 0]
        r -= (r * n).sum(1)[:, None] * n
        d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
        d20, d21 = (r * e1).sum(1), (r * e2).sum(1)
        den = d00 * d11 - d01 * d01
        b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den
        b = np.clip(np.column_stack([1 - b1 - b2, b1, b2]), 0., None); b /= b.sum(1)[:, None]
        X = (b[:, :, None] * tri).sum(1)
        j = int(np.argmin(np.linalg.norm(X - P[k], axis=1)))
        out_p[k], out_f[k] = X[j], cand[j]
    return out_p, out_f


COVER_RING_M = (5e-4, 1e-3, 2e-3)     # mismatch_opening_r1 M4: probe rings in the chart plane
COVER_RING_DIRS = 8


def label_coverage(pieces, q):
    from frgb.network_panels import envelope
    q = np.asarray(q, float)
    flat = [pc for p in pieces for pc in p.pieces()]

    def counts(q_):
        return envelope(flat, q_, 'L')[2], envelope(flat, q_, 'R')[2]
    cL, cR = counts(q)
    paths = np.zeros(len(q), np.int64)
    for p in pieces:
        pc = p.pieces()
        paths += (envelope(pc, q, 'L')[2] > 0) | (envelope(pc, q, 'R')[2] > 0)
    ring = np.zeros(len(q), bool)
    for r in COVER_RING_M:
        for k in range(COVER_RING_DIRS):
            a = 2. * np.pi * k / COVER_RING_DIRS
            l_, r_ = counts(q + r * np.array([np.cos(a), np.sin(a)]))
            ring |= (l_ != cL) | (r_ != cR)
    return dict(cover_L=cL, cover_R=cR, cover_paths=paths, ring_change=ring)


def refine_points(captured, context, trace_of, path, source, groups=None):
    with np.load(source / 'source_surface.npz') as a:
        V, F = np.asarray(a['vertices'], float), np.asarray(a['faces'], np.int64)
    vf, start = _vertex_faces(F, len(V))
    with np.load(path) as a:
        P0, f0, tr = np.asarray(a['point'], float), np.asarray(a['face'], np.int64), np.asarray(a['trace'], np.int64)
        cap = np.asarray(a['cap'], float) if 'cap' in a.files else np.full(len(P0), np.inf)
        pa = np.asarray(a['path'], np.int64) if 'path' in a.files else None
    P, f = P0.copy(), f0.copy()
    phi0 = np.full(len(P), np.nan)
    phi = np.full(len(P), np.nan); steps = np.zeros(len(P), np.int64); normal = np.full((len(P), 3), np.nan)
    sigma = np.full(len(P), np.nan); wall_lo = np.full(len(P), np.nan); wall_hi = np.full(len(P), np.nan)
    capped = np.zeros(len(P), bool)
    phi_check = np.full(len(P), np.nan)
    cover = dict(cover_L=np.full(len(P), -1, np.int64), cover_R=np.full(len(P), -1, np.int64),     # mismatch_opening_r1 M4
                 cover_paths=np.full(len(P), -1, np.int64), ring_change=np.zeros(len(P), bool))
    for gi, ((indices, solid), ctx) in enumerate(zip(captured, context)):
        sel, to_chart = _group_points(gi, indices, trace_of, tr, pa, groups)
        if not len(sel):
            continue
        chart, zones, m_of = _mid_context(solid, ctx)

        def phi_of(X):
            out = np.full(len(X), np.nan)
            for part in np.array_split(np.arange(len(X)), max(1, len(X) // 100_000 + 1)):
                if not len(part):
                    continue
                q, g = normal_foot(chart, zones, to_chart(X[part]))
                s = np.einsum('ij,ij->i', to_chart(X[part]) - chart.surface(q), chart.normals(q[:, 0], zones))
                out[part] = np.where(g <= FOOT_TOL_M, s - m_of(q), np.nan)
            return out
        live = sel.copy()
        cur = phi_of(P[live])
        phi[live] = cur; phi0[live] = cur
        for it in range(REFINE_STEPS):
            go = np.isfinite(cur) & (np.abs(cur) > REFINE_TOL)
            live, cur = live[go], cur[go]
            if not len(live):
                break
            tri = V[F[f[live]]]
            e1 = tri[:, 1] - tri[:, 0]; e1 /= np.linalg.norm(e1, axis=1)[:, None]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1)[:, None]
            e2 = np.cross(n, e1)
            X = P[live]
            d = phi_of(np.concatenate([X + REFINE_H * e1, X - REFINE_H * e1, X + REFINE_H * e2, X - REFINE_H * e2])).reshape(4, -1)
            g = ((d[0] - d[1]) / (2 * REFINE_H))[:, None] * e1 + ((d[2] - d[3]) / (2 * REFINE_H))[:, None] * e2
            gg = (g * g).sum(1)
            ok = np.isfinite(gg) & (gg > 1e-12)
            step = np.where(ok[:, None], -(cur / np.where(ok, gg, 1.))[:, None] * g, 0.)
            ln = np.linalg.norm(step, axis=1)
            step *= np.minimum(1., REFINE_CAP / np.maximum(ln, 1e-300))[:, None]
            Pn, fn = _snap(X + step, f[live], V, F, vf, start)
            P[live], f[live] = Pn, fn
            steps[live] += 1
            cur = phi_of(Pn)
            phi[live] = cur
        # the same station sample) is undone
        undo = sel[np.linalg.norm(P[sel] - P0[sel], axis=1) > cap[sel]]
        P[undo], f[undo], phi[undo] = P0[undo], f0[undo], phi0[undo]
        capped[undo] = True
        q, g = normal_foot(chart, zones, to_chart(P[sel]))
        normal[sel] = chart.normals(q[:, 0], zones); sigma[sel] = q[:, 0]
        for k_, v_ in label_coverage(ctx['pieces'], q).items():      # mismatch_opening_r1 M4 (q via to_chart)
            cover[k_][sel] = v_
        s = np.einsum('ij,ij->i', to_chart(P[sel]) - chart.surface(q), normal[sel])
        _, Lw, Rw = m_of(q, walls=True)
        wall_lo[sel], wall_hi[sel] = Lw - s, Rw - s          # analytic walls along n relative to the point (record)
        q2, g2 = normal_foot(chart, zones, to_chart(P[sel]))
        s2 = np.einsum('ij,ij->i', to_chart(P[sel]) - chart.surface(q2), chart.normals(q2[:, 0], zones))
        phi_check[sel] = np.where(g2 <= FOOT_TOL_M, s2 - m_of(q2), np.nan)
    out = path.with_name(path.stem + '_refined.npz')
    np.savez_compressed(out, point=P, face=f, phi=phi, steps=steps, normal=normal, sigma=sigma, shift=np.linalg.norm(P - P0, axis=1),
                        wall_lo=wall_lo, wall_hi=wall_hi, capped=capped,
                        **({} if groups is None else dict(phi_check=phi_check)), **cover)
    a = np.abs(phi[np.isfinite(phi)])
    return dict(points=int(len(P)), output=str(out), finite_phi=int(np.isfinite(phi).sum()),
                converged=int((a <= REFINE_TOL).sum()), abs_phi_p50_p99_max=[float(x) for x in np.quantile(a, [.5, .99, 1.])] if len(a) else None,
                shift_m_p50_p99_max=[float(x) for x in np.quantile(np.linalg.norm(P - P0, axis=1), [.5, .99, 1.])],
                face_changed=int((f != f0).sum()), capped=int(capped.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', nargs='?', choices=list(CASES))
    ap.add_argument('--case-dir', type=Path)
    ap.add_argument('--request', type=Path)
    ap.add_argument('--dest', type=Path)
    ap.add_argument('--scratch', type=Path, default=SCRATCH)
    ap.add_argument('--clean', action='store_true', help='delete the scratch build after the verification')
    ap.add_argument('--manifest', default=None, help='manifest file name in the destination (default manifest_<code>.json)')
    ap.add_argument('--code', choices=['current', 'backup'], default='current')
    ap.add_argument('--midsurface', action='store_true')
    ap.add_argument('--evaluate', type=Path, help='npz with point (n,3), trace (n,) int: exact s - m at those points')
    ap.add_argument('--refine', type=Path, help='npz with point, face, trace: r3 amendment c Newton refinement onto the mid-surface')
    # and run the refinement of PATH in this process when PATH.go appears (PATH.stop: exit without refining)
    ap.add_argument('--serve-refine', type=Path)
    ap.add_argument('--serve-manifest', default='manifest_refine.json')
    ap.add_argument('--serve-timeout-s', type=float, default=6 * 3600.)
    args = ap.parse_args()
    if args.case_dir is not None:
        cd = args.case_dir.resolve()
        raw_dir = cd if cd.name.endswith('_raw') else cd.with_name(cd.name + '_raw')
        spec = dict(case=cd, raw=raw_dir if raw_dir.is_dir() else cd, module=cd.parent.parent, generic=True, request=args.request)
        args.case = cd.name
    else:
        spec = CASES[args.case]
    build = read(spec['case'] / 'build.json')
    raw = read(spec['raw'] / 'build.json') if 'raw' in spec else build
    pkg = backup_package() if args.code == 'backup' else P1
    sys.path[:0] = [str(spec['module']), str(pkg), TOOLS]
    import frgb.network_panels as NP
    assert Path(NP.__file__).resolve().parent == (Path(pkg) / 'frgb').resolve(), NP.__file__
    # below as before
    from frgb import fault_charts as FC
    import fault_support as FS
    fault = fault_status(raw, spec, FC, FS, args.code)
    np_rec = network_panels_pin(args.code, FS)
    if np_rec is not None:
        if not np_rec['ok']:
            dest_ = args.dest if args.dest is not None else HERE / 'cavities' / args.case
            dest_.mkdir(parents=True, exist_ok=True)
            (dest_ / (args.manifest or 'manifest_%s.json' % args.code)).write_text(
                json.dumps(dict(verified=False, network_panels=np_rec), indent=1, allow_nan=False) + '\n', encoding='utf8')
            print(json.dumps(dict(case=args.case, network_panels_pin=False)), flush=True)
            sys.exit(3)
    dest = args.dest if args.dest is not None else HERE / 'cavities' / args.case
    if fault['status'] in ('stop', 'unsupported'):
        dest.mkdir(parents=True, exist_ok=True)
        rec = {k: v for k, v in fault.items() if k != 'frames'}
        (dest / (args.manifest or 'manifest_%s.json' % args.code)).write_text(
            json.dumps(dict(verified=False, fault=rec), indent=1, allow_nan=False) + '\n', encoding='utf8')
        print(json.dumps(dict(case=args.case, fault_status=fault['status'], reason=fault.get('reason') or fault.get('reasons'))), flush=True)
        sys.exit(UNSUPPORTED_EXIT if fault['status'] == 'unsupported' else 3)
    faulted = fault['status'] == 'ok'
    captured, context = [], []
    original = NP.build_surface_cavity

    def keep(chart, paths, **kw):
        out = original(chart, paths, **kw)
        captured.append(([int(p.path_index) for p in paths], out[0]))
        context.append(dict(chart=chart, pieces=list(paths), kw=kw, checks=out[2], tables=out[1]))
        return out
    NP.build_surface_cavity = keep
    module = importlib.util.spec_from_file_location('case_build', spec['module'] / 'build.py')
    B = importlib.util.module_from_spec(module); module.loader.exec_module(B)
    args.scratch.mkdir(parents=True, exist_ok=True)
    out = args.scratch / ('rebuild_%s_%s_%d' % (args.case, args.code, int(time.time())))
    sys.argv = ['build.py'] + (generic_arguments(raw, out, spec['request']) if spec.get('generic') else arguments(spec, raw, out))
    fcdir = None
    if faulted:
        fcdir = args.scratch / ('faultcav_%s_%s_%d' % (args.case, args.code, int(time.time())))
        sys.argv += ['--save-fault-cavities', str(fcdir)]
    started = time.time()
    B.main()
    elapsed = time.time() - started
    new = read(out / 'build.json')
    trace_of = {p['index']: p['source_trace_id'] for p in raw['paths']}
    checks = dict(code=args.code, elapsed_s=elapsed, argv=sys.argv[1:], fractures={}, stations={},
                  verification='criterion r2 cavity_rebuild with amendment a')
    if np_rec is not None:
        checks['network_panels'] = np_rec
    ok = True
    groups = None
    if faulted:
        checks['fault'] = {k: v for k, v in fault.items() if k != 'frames'}
        checks['verification'] += '; r4 faults amendments d, f (group-wise, gates i-v)'
        ok, checks['fractures'], groups = verify_faulted(captured, raw, new, fault, fcdir, FC, FS)
    if not faulted:
        by_trace_old = {p['source_trace_id']: p for p in raw['paths']}
        rebuilt_checks = {p['index']: p['construction_checks'] for p in new['paths']}
        for indices, solid in captured:
            traces = {trace_of[i] for i in indices if i in trace_of}
            assert len(traces) == 1, (indices, traces)
            trace = traces.pop()
            old = by_trace_old[trace]
            vol = solid.volume()
            rel = abs(vol - old['surface_volume_m3']) / old['surface_volume_m3']
            # later fold_split record with used = false); the solid's num_tri is counted after manifold and only recorded
            extra, differ = set(), set()
            for p in raw['paths']:
                if p['source_trace_id'] != trace:
                    continue
                o, n = p['construction_checks'], rebuilt_checks[p['index']]
                differ |= {k for k in o if o[k] != n.get(k)}
                extra |= set(n) - set(o)
            extra_ok = extra <= {'fold_split'} and all(rebuilt_checks[p['index']].get('fold_split', {}).get('used') is False
                                                       for p in raw['paths'] if p['source_trace_id'] == trace and 'fold_split' in extra)
            same = rel <= 1e-9 and not differ and extra_ok
            ok &= same
            checks['fractures'][trace] = dict(paths=indices, volume_m3=vol, recorded_m3=old['surface_volume_m3'], relative=rel,
                                             solid_num_tri=int(solid.num_tri()),
                                             recorded_cavity_triangles=old['construction_checks']['cavity_triangles'],
                                             differing_check_keys=sorted(differ), added_check_keys=sorted(extra), match=bool(same))
    raw_case = spec['raw'] if 'raw' in spec else spec['case'].with_name(spec['case'].name + '_raw')
    with np.load(out / 'difference.npz') as x, np.load(raw_case / 'difference.npz') as y:
        same = set(x.files) == set(y.files) and all(x[k].shape == y[k].shape and np.array_equal(x[k], y[k]) for k in y.files)
    checks['difference_bitwise_vs_raw'] = dict(raw=str(raw_case), match=bool(same))
    ok &= same
    rel = abs(new['remaining_volume_m3'] - raw['remaining_volume_m3']) / raw['remaining_volume_m3']
    checks['remaining_volume'] = dict(rebuilt=new['remaining_volume_m3'], recorded=raw['remaining_volume_m3'], relative=rel,
                                      match=bool(rel <= 1e-9))
    ok &= rel <= 1e-9
    if not faulted:
        checks['captured_fractures'] = len(captured)
        checks['recorded_fractures'] = len(by_trace_old)
        ok &= len(captured) == len(by_trace_old)
    for p in raw['paths']:
        name = 'path_%02d_stations.npz' % p['index']
        with np.load(spec['raw'] / name if 'raw' in spec else spec['case'] / name) as a, np.load(out / name) as b:
            same = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        if 'raw' in spec:  # the repaired case keeps the raw stations
            with np.load(spec['raw'] / name) as a, np.load(spec['case'] / name) as c:
                same &= all(np.array_equal(a[k], c[k]) for k in a.files)
        checks['stations'][name] = bool(same)
        ok &= same
    checks['wall_fidelity'] = wall_fidelity(captured, context)
    ok &= checks['wall_fidelity']['ok']
    checks['overlap_blend'] = overlap_blend_check(captured, context, new)  # overlap_blend_c1_r1 R2 (fail closed)
    ok &= checks['overlap_blend']['ok']
    checks['verified'] = bool(ok)
    dest = args.dest if args.dest is not None else HERE / 'cavities' / args.case
    dest.mkdir(parents=True, exist_ok=True)
    if ok and faulted:
        listed = []
        for (indices, solid), grp in zip(captured, groups):
            if grp['world'] is not None:
                v, f = grp['world']; src_kind = 'export (post-replay, world)'
            else:
                mesh = solid.to_mesh64()
                v = np.asarray(mesh.vert_properties)[:, :3].astype(float); f = np.asarray(mesh.tri_verts).astype(np.int64)
                src_kind = 'capture (no fault acted on this group)'
            name = grp['key'] + '.npz'
            np.savez_compressed(dest / name, vertices=v, faces=f, source_trace_id=grp['trace'], paths=np.asarray(indices, np.int64),
                                group_key=grp['key'], group_tag=grp['tag'] or '')
            listed.append(dict(file=name, sha256=FS.sha256(dest / name), key=grp['key'], trace=grp['trace'], tag=grp['tag'],
                               paths=list(indices), source=src_kind,
                               shift_m=None if grp['shift'] is None else [float(x) for x in grp['shift']]))
        checks['fault_groups'] = listed
    if ok and not faulted:
        for indices, solid in captured:
            trace = trace_of[indices[0]]
            mesh = solid.to_mesh64()
            v = np.asarray(mesh.vert_properties)[:, :3].astype(float)
            f = np.asarray(mesh.tri_verts).astype(np.int64)
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            signed = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)
            checks['fractures'][trace]['mesh_signed_volume_m3'] = signed
            np.savez_compressed(dest / (trace + '.npz'), vertices=v, faces=f, source_trace_id=trace,
                                paths=np.asarray(indices, np.int64))
    if ok and args.evaluate is not None:
        checks['evaluate'] = evaluate_points(captured, context, trace_of, args.evaluate, groups=groups)
    if ok and args.refine is not None:
        checks['refine'] = refine_points(captured, context, trace_of, args.refine, Path(raw['source']), groups=groups)
    if ok and args.midsurface:
        if faulted:
            checks['midsurface'] = export_midsurface(captured, context, Path(raw['source']), trace_of, dest, case_dir=spec['case'],
                                                     groups=groups, fault=fault, FC=FC, ext_ends=FS.fault_ends_of(raw), FS=FS,
                                                     structure=getattr(B, 'FAULT_STRUCTURE', None))
        else:
            checks['midsurface'] = export_midsurface(captured, context, Path(raw['source']), trace_of, dest, case_dir=spec['case'])
    checks['scratch_build'] = str(out)
    if args.clean:
        shutil.rmtree(out, ignore_errors=True)
        checks['scratch_build_removed'] = True
        if fcdir is not None:
            shutil.rmtree(fcdir, ignore_errors=True)
            checks['fault_cavities_removed'] = True
    (dest / (args.manifest or 'manifest_%s.json' % args.code)).write_text(json.dumps(checks, indent=1, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(case=args.case, code=args.code, verified=ok, elapsed_s=round(elapsed, 1)), indent=None), flush=True)
    if ok and args.serve_refine is not None:
        sys.exit(serve_refine(args, captured, context, trace_of, Path(raw['source']), dest, checks, groups=groups))
    sys.exit(0 if ok else 3)


def serve_refine(args, captured, context, trace_of, source, dest, stage_a, groups=None):
    """Speed-up s2: the verified context stays; the solids are dropped (refine_points reads the context only). Waits for
    PATH.go (refine PATH with refine_points, write the manifest) or PATH.stop; writes PATH.ready.json first."""
    import gc
    from procmem import process_gb
    path = args.serve_refine
    go, stop, ready = (path.with_name(path.name + s) for s in ('.go', '.stop', '.ready.json'))
    captured[:] = [(indices, None) for indices, _ in captured]
    for g in groups or []:
        g['world'] = None
    gc.collect()
    mem = dict(retained=process_gb())
    tmp = ready.with_name(ready.name + '.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(), memory=mem)) + '\n', encoding='utf8')
    os.replace(tmp, ready)                                  # atomic: the client never reads a partial file
    print(json.dumps(dict(event='serve_ready', memory=mem)), flush=True)
    t0 = time.time()
    while not go.exists():
        if stop.exists():
            print(json.dumps(dict(event='serve_stop')), flush=True)
            return 0
        if time.time() - t0 > args.serve_timeout_s:
            print(json.dumps(dict(event='serve_timeout')), flush=True)
            return 5
        time.sleep(.2)
    waited = time.time() - t0
    t1 = time.time()
    rec = refine_points(captured, context, trace_of, path, source, groups=groups)
    mem['after_refine'] = process_gb()
    out = dict(code=args.code, speedup='s2: refinement on the stage-A context (verified there; no second build)',
               stage_a_manifest=str(dest / (args.manifest or 'manifest_%s.json' % args.code)), stage_a_verified=stage_a['verified'],
               refine=rec, refine_s=time.time() - t1, waited_s=waited, memory=mem, verified=True)
    (dest / args.serve_manifest).write_text(json.dumps(out, indent=1, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(event='serve_refined', refine_s=round(out['refine_s'], 1), memory=mem)), flush=True)
    return 0


if __name__ == '__main__':
    main()
