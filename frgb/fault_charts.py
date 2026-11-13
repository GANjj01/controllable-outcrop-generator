"""Fault-slip frames of a frozen network's joint interfaces.

The kernel's apply_fault moves the part of a fault-cut joint interface that lies in the fault's moving half-space by
the fault's exact rigid translation t (BranchedSurface); the network's trace definitions (SurfaceChart) are the ones
before the slip. Downstream code maps a path on a moving block back with q = p - t, builds its cavity there and
replays the slip on the cavity. This module is the single implementation of that frame: prepare.py, build.py and the
trace labels only import it."""
import json
from pathlib import Path

import numpy as np

L_DELTA_M = .02
L_MAX_M = 5.
MOUTH_H_MIN_M = .002
JOINT_TOL_M = 1e-9
STRIP_TOL_M = 1e-3
_NO_JOINT = object()       # check_inputs default: no joint check (unit tests of the other checks); builds pass the triangles or None
CONTACT_TOL_M3 = 1e-9
BOX_MARGIN_M = .1
LAMBDA_STEP_M = 1e-3
LAMBDA_TOL_M = 1e-9
DEPTH_SAMPLES = 21
WALL_FRACTIONS = (0., .5, 1.)
MIN_OWN_SIDE_SAMPLES = 21
DENSE_SAMPLES = 201
FOOT_MARGIN_M = 1e-3
JOINT_EDGE_ANGLE_DEG = .1
JOINT_EDGE_LINE_M = 1e-6   # ... and 1e-6 m of the first piece's line
CAP_DEPTH_MIN_M = 1e-3


def _no_faults():
    return dict(status='no_faults', faults=[], multiple=frozenset(), reasons=[])


def fault_frames(geometry_directory):
    """the fault frames of a scene geometry (the directory load_frozen_network reads); no fault event -> 'no_faults' without
    reading anything else (callers keep their old code path)"""
    gd = Path(geometry_directory)
    spec = json.loads((gd/'package/model_spec.json').read_text(encoding='utf8'))
    if not any(e.get('kind') == 'fault' for e in spec.get('events', [])):
        return _no_faults()
    return frames_from(spec, json.loads((gd/'package/interfaces.json').read_text(encoding='utf8')))


def frames_from(spec, interface_rows):
    """spec = model_spec.json, interface_rows = the kernel's interface records (id, lineage); the kernel's own objects throughout:
    t = compile_fault_map(...).motion.vector_enu_m (exact rationals -> float), the moving half-space = apply_fault's
    FieldRegion(OffsetField(side_classifier, 0, sign)), on-plane points by the kernel's length tolerance"""
    from outcrop.modeling.decode import decode_model
    from outcrop.modeling.events import compile_events
    from outcrop.modeling.faults import compile_fault_map
    from outcrop.modeling.kernel.analytic import FieldRegion, OffsetField, compile_geometry
    from outcrop.modeling.kernel.types import AABB
    from outcrop.modeling.schema import thaw
    if not any(e.get('kind') == 'fault' for e in spec.get('events', [])):
        return _no_faults()
    numeric = decode_model(spec); kernel = compile_geometry(numeric); plan = [thaw(e) for e in compile_events(numeric).events]
    order = [e['id'] for e in plan]; kinds = {e['id']: e['kind'] for e in plan}
    joints = {f['id']: f for f in spec['features'] if f['kind'] == 'joint'}
    rows = {r['id']: r for r in interface_rows}
    bodies = {b['id'] for b in spec['bodies']}
    lo, hi = spec['domain']['min_m'], spec['domain']['max_m']
    tol = kernel.request.tolerance
    faults, reasons = [], []
    for o in sorted(joints):
        extra = [x for x in rows.get(o, {}).get('lineage', []) if kinds.get(x) not in ('fault', 'create_features')]
        if extra:
            reasons.append('%s: lineage has events other than fault / create_features %s' % (o, extra))
    for e in [e for e in plan if e['kind'] == 'fault']:
        f = dict(e['fault']) if isinstance(e['fault'], dict) else {'geometry': e['fault']}
        f.update({k: e[k] for k in ('motion', 'moving_side', 'support', 'edge_policy') if k in e}); f['id'] = e['id']
        why = [w for w, bad in (('finite support', 'support' in f), ('curved map', 'map' in f), ('overlap policy', 'overlap_policy' in e),
                                ('vector motion', e['motion'].get('kind') == 'vector'),
                                ('nonzero opening', float(e['motion'].get('opening_m', 0) or 0) != 0),
                                ('not every body a target', set(e.get('targets', [])) != bodies)) if bad]
        if why:
            reasons.append('%s: unsupported (%s)' % (e['id'], ', '.join(why))); continue
        fm = compile_fault_map(f, kernel)
        sign = -1. if e['moving_side'] == 'positive' else 1.
        action = FieldRegion(e['id'] + ':moving-side', OffsetField(fm.side_classifier, 0., sign), AABB(tuple(lo), tuple(hi)))
        t = np.asarray(fm.motion.vector_enu_m, float)
        n = np.asarray(fm.side_classifier.normal, float)
        if abs(n @ t) / np.linalg.norm(n) > tol.length_m:
            reasons.append('%s: slip not tangential to the fault plane (%.3e m)' % (e['id'], abs(n @ t) / np.linalg.norm(n))); continue
        fi = order.index(e['id'])
        sel = e.get('interfaces', [])
        cand = {o for o, j in joints.items() if set(j['hosts']) & set(e['targets'])} if sel == 'hosted' else set(sel) & set(joints)
        before = lambda o: joints[o].get('at_stage', 'initial') == 'initial' or any(  # noqa: E731
            x['kind'] == 'create_features' and o in x.get('features', []) and order.index(x['id']) < fi for x in plan)
        declared = {o for o in cand if before(o)}
        recorded = {o for o in joints if o in rows and e['id'] in rows[o].get('lineage', [])}
        if declared != recorded:
            reasons.append('%s: moved joints declared %d vs kernel lineage %d' % (e['id'], len(declared), len(recorded))); continue
        faults.append(dict(id=e['id'], t=t, t_exact=fm.motion.residuals['exact_vector_enu_m'],
                           tangent_correction_m=fm.motion.residuals['tangent_realization_correction_m'], moving_side=e['moving_side'],
                           sign=sign, action=action, field=fm.side_classifier, moved=frozenset(declared)))
    count = {}
    for f in faults:
        for o in f['moved']:
            count[o] = count.get(o, 0) + 1
    multiple = frozenset(o for o, k in count.items() if k > 1)
    status = 'unsupported' if reasons else 'ok'
    joint_triangles = {o: np.array(kernel.surfaces[j['geometry']].triangles, float) for o, j in joints.items() if j.get('geometry') in kernel.surfaces}
    return dict(status=status, faults=faults, multiple=multiple, reasons=reasons, tol=tol, joint_triangles=joint_triangles)


def path_frame(frames, interface_id, points, *, allow_multiple=False):
    """(shift, records): shift = the total translation to subtract from the path's points (None: no fault moved this path).
    Faults are undone in reverse event order; per fault every station is classified with that fault's moving half-space (the
    zero-opening slip maps that half-space onto itself, so final coordinates classify correctly): stations strictly inside and
    none strictly outside -> moved (all stations, on-plane ones included, shifted by -t); strictly on both sides or only on the
    plane -> error."""
    if frames['status'] == 'no_faults':
        return None, []
    if frames['status'] != 'ok':
        raise ValueError('fault frames unsupported: ' + '; '.join(frames['reasons']))
    if interface_id in frames['multiple'] and not allow_multiple:
        raise ValueError('joint %s moved by more than one fault: not supported (fail closed)' % interface_id)
    q = np.asarray(points, float); total = np.zeros(3); rec = []
    for f in reversed(frames['faults']):
        if interface_id not in f['moved']:
            continue
        c = f['action'].classify(q, frames['tol'])
        n_in, n_on, n_out = int(c.inside.sum()), int(c.boundary.sum()), int(c.outside.sum())
        r = dict(fault=f['id'], inside=n_in, on_plane=n_on, outside=n_out, on_plane_stations=np.flatnonzero(c.boundary).tolist())
        if n_in and n_out:
            raise ValueError('path on both sides of %s' % f['id'])
        if not n_in and not n_out:
            raise ValueError('path lies in the plane of %s' % f['id'])
        r['moved'] = bool(n_in)
        if n_in:
            q = q - f['t']; total = total + f['t']
        rec.append(r)
    return (total if any(r['moved'] for r in rec) else None), rec


def fault_ends(frames, records, n_stations):
    """the path ends (0 / -1) on the plane of a fault that acted on the path's joint: [(fault, end, moved)]"""
    by_id = {f['id']: f for f in frames['faults']}
    out = []
    for r in records:
        for k, st in ((0, 0), (-1, n_stations - 1)):
            if st in r['on_plane_stations']:
                out.append((by_id[r['fault']], k, r['moved']))
    return out


def world_field(fn, shift):
    """P -> fn(P + shift) for a world-frame field (wall relief rho, the mouth band fine_of) evaluated on a moved group's source
    frame; unchanged when fn or shift is None"""
    if fn is None or shift is None:
        return fn
    s = np.asarray(shift, float)
    return lambda P: fn(np.asarray(P, float) + s)


def own_halfspace(fault, moved):
    """(unit normal, offset) of the closed half-space of the path's own block, for Manifold.trim_by_plane (keeps n.x >= offset):
    moving block sign*(n.x - d) <= 0, the other block >= 0"""
    n = np.asarray(fault['field'].normal, float); d = float(fault['field'].offset); k = (-1. if moved else 1.) * fault['sign']
    s = np.linalg.norm(n)
    return k * n / s, k * d / s


def signed_other(fault, moved, X):
    """> 0: X is strictly on the other block's side of the plane (m)"""
    n, d = own_halfspace(fault, moved)
    return d - np.asarray(X, float) @ n


def relief_on_chart(chart, wall_relief, wall_relief_tip_m, sigma_stations, aperture_stations, pieces=None):
    if wall_relief is None:
        return None
    if wall_relief_tip_m is None:
        return lambda q: np.asarray(wall_relief(chart.surface(q)), float)
    if pieces is not None:
        from frgb.network_panels import taper_factor

        def rho2(q):
            return np.asarray(wall_relief(chart.surface(q)), float) * taper_factor(pieces, q, wall_relief_tip_m)
        return rho2
    o = np.argsort(np.asarray(sigma_stations, float), kind='stable')
    sig, ap = np.asarray(sigma_stations, float)[o], np.asarray(aperture_stations, float)[o]

    def rho(q):
        x = np.clip(np.interp(np.atleast_2d(q)[:, 0], sig, ap) / wall_relief_tip_m, 0., 1.)
        return np.asarray(wall_relief(chart.surface(q)), float) * x * x * (3. - 2. * x)
    return rho


def end_extension(chart, fault, moved, st_end, t2, nu2, depth_end, margin, oL_row, oR_row, fractions, *, normals_of=None, rho_q=None,
                  defer_chart=False):
    """extension length L of one fault end, in the path's source frame. st_end, t2 (unit, leaving the path) and nu2 (unit,
    inward) in the chart's unrolled plane; oL_row / oR_row: the end station's wall offsets over the depth fractions;
    normals_of(sigma) -> 3D unit normals (default: panel normals); rho_q(q) -> relief added to both walls (None:
    none). Grid: 21 depths a in [-margin, depth_end] x walls / centre line f in {0, .5, 1}; per grid point the first
    lambda >= 0 at which S(st_end + lambda t2 + a nu2) + o N crosses into the other block (1 mm bracket, bisection to
    1e-9 m; already across -> 0)."""
    st_end, t2, nu2 = (np.asarray(v, float) for v in (st_end, t2, nu2))
    fr = np.asarray(fractions, float)
    a = np.linspace(-margin, depth_end, DEPTH_SAMPLES)

    def wall(row):
        row = np.asarray(row, float)
        return np.where(a <= 0, row[0], np.interp(np.clip(a / depth_end, 0., 1.), fr, row))
    oL, oR = wall(oL_row), wall(oR_row)

    def other(lam, ai, oi):
        q = st_end + np.outer(lam, t2) + np.outer(ai, nu2)
        N = normals_of(q[:, 0]) if normals_of is not None else chart.n[chart.panel(q[:, 0])]
        o = oi + (np.asarray(rho_q(q), float) if rho_q is not None else 0.)
        return signed_other(fault, moved, chart.surface(q) + o[:, None] * N)
    lam = np.arange(0., L_MAX_M + LAMBDA_STEP_M / 2, LAMBDA_STEP_M)
    grid = []
    for f in WALL_FRACTIONS:
        o = (1 - f) * oL + f * oR
        s = other(np.tile(lam, len(a)), np.repeat(a, len(lam)), np.repeat(o, len(lam))).reshape(len(a), len(lam))
        for i in range(len(a)):
            if s[i, 0] > 0:
                grid.append(dict(a_m=float(a[i]), f=f, c_m=0., already_across=True)); continue
            j = np.flatnonzero(s[i] > 0)
            if not len(j):
                raise ValueError('fault %s end: no crossing within %.1f m (a %.4f m, wall fraction %.1f)' % (fault['id'], L_MAX_M, a[i], f))
            lo_, hi_ = lam[j[0] - 1], lam[j[0]]
            while hi_ - lo_ > LAMBDA_TOL_M:
                mid = .5 * (lo_ + hi_)
                if other(np.array([mid]), np.array([a[i]]), np.array([o[i]]))[0] > 0:
                    hi_ = mid
                else:
                    lo_ = mid
            grid.append(dict(a_m=float(a[i]), f=f, c_m=float(hi_), already_across=False))
    worst = max(grid, key=lambda g: g['c_m'])
    L = worst['c_m'] + L_DELTA_M
    if L > L_MAX_M:
        raise ValueError('fault %s end: extension %.3f m exceeds %.1f m' % (fault['id'], L, L_MAX_M))
    sig = float(st_end[0] + L * t2[0]); lo, hi = float(chart.sigma[0]), float(chart.sigma[-1])
    out = dict(L_m=float(L), max_c_m=worst['c_m'], at=dict(a_m=worst['a_m'], f=worst['f']), delta_m=L_DELTA_M, grid=grid)
    if not lo - 1e-9 <= sig <= hi + 1e-9:
        message = 'fault %s end: the extension leaves the trace chart (sigma %.4f outside [%.4f, %.4f], L %.3f m)' % (fault['id'], sig, lo, hi, L)
        if not defer_chart:
            raise ValueError(message)
        out['chart_overrun'] = dict(sigma=sig, range=[lo, hi], L_m=float(L), message=message)
    return out


def _on_triangles(P, tri):
    """per point: the distance to the nearest of the triangles tri (n, 3, 3) and whether that nearest point is interior (the foot on
    the triangle's plane has barycentrics >= -1e-12); exact for interior feet, an upper bound otherwise (only used to fail)"""
    best = np.full(len(P), np.inf); inside = np.zeros(len(P), bool)
    for a, b, c in tri:
        e0, e1 = b - a, c - a; n = np.cross(e0, e1); n = n / np.linalg.norm(n); d = (P - a) @ n; F = P - np.outer(d, n)
        g = np.array([[e0 @ e0, e0 @ e1], [e0 @ e1, e1 @ e1]]); uv = np.linalg.solve(g, np.c_[(F - a) @ e0, (F - a) @ e1].T).T
        bary = np.c_[1 - uv.sum(1), uv]; ins = (bary >= -1e-12).all(1)
        cl = np.clip(bary, 0, None); cl = cl / cl.sum(1)[:, None]; Q = cl[:, :1] * a + cl[:, 1:2] * b + cl[:, 2:] * c
        dist = np.where(ins, np.abs(d), np.linalg.norm(P - Q, axis=1)); upd = dist < best; best[upd] = dist[upd]; inside[upd] = ins[upd]
    return best, inside


def strip_samples(chart, strip, n=DEPTH_SAMPLES):
    """the n x n (21 x 21) extension strip of one fault end on the mid-surface (source frame): S(p_end + lambda t2 + a nu2),
    lambda in [0, L], a in [-margin, depth]"""
    lam, a = np.meshgrid(np.linspace(0., strip['L'], n), np.linspace(-strip['margin'], strip['depth'], n))
    return chart.surface(strip['p_end'] + np.outer(lam.ravel(), strip['t2']) + np.outer(a.ravel(), strip['nu2']))


def extension_on_joint(chart, strip, triangles, fault_id, side=None):
    """Layers 2 / 3 are recorded under 'own_side_dense' (layer, grid, its samples and own-side numbers; layer 3
    full_strip_fallback)."""
    if triangles is None:
        raise ValueError('fault %s end: the extension leaves its joint surface (no kernel joint surface for this path)' % fault_id)
    X = strip_samples(chart, strip)
    d, inside = _on_triangles(X, np.asarray(triangles, float))
    rec = dict(samples=int(len(X)), max_distance_m=float(d.max()), outside=int((~inside).sum()), tol_m=JOINT_TOL_M)
    if side is None:
        if rec['outside'] or rec['max_distance_m'] > JOINT_TOL_M:
            raise ValueError('fault %s end: the extension leaves its joint surface (max %.3e m, %d sample(s) outside it)' % (fault_id, rec['max_distance_m'], rec['outside']))
        return rec
    own = np.asarray(side(X), float) <= JOINT_TOL_M
    rec.update(own_side_samples=int(own.sum()), own_side_max_distance_m=float(d[own].max()) if own.any() else None,
               own_side_outside=int((~inside[own]).sum()), min_own_side_samples=MIN_OWN_SIDE_SAMPLES)
    n_, dmax, out_ = rec['own_side_samples'], rec['own_side_max_distance_m'], rec['own_side_outside']; where = 'own side'
    if n_ < MIN_OWN_SIDE_SAMPLES:                                   # layer 2: the dense own-side resampling
        X2 = strip_samples(chart, strip, DENSE_SAMPLES); own2 = np.asarray(side(X2), float) <= JOINT_TOL_M
        d2, in2 = _on_triangles(X2[own2], np.asarray(triangles, float)) if own2.any() else (np.zeros(0), np.zeros(0, bool))
        n_, dmax, out_ = int(own2.sum()), (float(d2.max()) if len(d2) else None), int((~in2).sum()); where = 'own side, dense grid'
        rec['own_side_dense'] = dict(layer=2, grid=DENSE_SAMPLES, samples=int(len(X2)), own_side_samples=n_, own_side_max_distance_m=dmax,
                                     own_side_outside=out_)
        if n_ < MIN_OWN_SIDE_SAMPLES:
            rec['own_side_dense'].update(layer=3, full_strip_fallback=True)
            n_, dmax, out_ = rec['samples'], rec['max_distance_m'], rec['outside']; where = 'whole strip, r3\' fallback'
    if out_ or dmax > JOINT_TOL_M:
        raise ValueError('fault %s end: the extension leaves its joint surface (max %.3e m, %d sample(s) outside it, %s)' % (fault_id, dmax, out_, where))
    return rec


def _boundary_segments(tri):
    """the boundary edges (used by one triangle) of the triangle soup tri (k, 3, 3), vertices matched at 1e-9 m: [(a, b, c)], c = the
    triangle's third vertex (the joint's interior side)"""
    cnt, seg = {}, {}
    for t in np.asarray(tri, float):
        for i in range(3):
            a, b = tuple(np.round(t[i], 9)), tuple(np.round(t[(i + 1) % 3], 9)); k = (a, b) if a <= b else (b, a)
            cnt[k] = cnt.get(k, 0) + 1; seg[k] = (t[i], t[(i + 1) % 3], t[(i + 2) % 3])
    return [seg[k] for k, c in cnt.items() if c == 1]


def _nearest_segment(P, segs):
    best = np.full(len(P), np.inf); who = np.full(len(P), -1)
    for j, (a, b, _) in enumerate(segs):
        ab = b - a; t = np.clip(((P - a) @ ab) / (ab @ ab), 0., 1.); d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        m = d < best; best[m] = d[m]; who[m] = j
    return who, best


def _box_clip(a, u, lo, hi):
    """the parameter interval of the line a + s u inside the axis-aligned box [lo, hi] (None: misses it)"""
    s0, s1 = -np.inf, np.inf
    for k in range(3):
        if abs(u[k]) < 1e-15:
            if not lo[k] <= a[k] <= hi[k]:
                return None
            continue
        t0, t1 = sorted(((lo[k] - a[k]) / u[k], (hi[k] - a[k]) / u[k])); s0, s1 = max(s0, t0), min(s1, t1)
    return (s0, s1) if s0 <= s1 else None


def joint_edge_plan(chart, strip, triangles, side, box=None):
    """Returns ('none', None), ('corner', n) (off-joint samples nearest to n > 1 non-collinear boundary segments:
    check_inputs fails closed) or ('trim', plan). Trigger, all three: (1) own-side samples off the joint (nearest
    point not interior to a joint triangle; inside but too far, e.g. a bend, does not count), on the grid the joint
    check would gate (21 x 21, or 201 x 201 when the former has < MIN_OWN_SIDE_SAMPLES own-side samples; fewer on both
    -> none); (2) all their nearest boundary segments on one line (JOINT_EDGE_ANGLE_DEG, JOINT_EDGE_LINE_M); (3) every
    own-side sample of the lambda = 0 column on the joint (interior, <= JOINT_TOL_M). plan: p_e, u_e (the edge), m_e
    (in the joint plane of the first segment's triangle, normal to the edge, away from the joint), the segments, the
    grid, the off-joint own-side counts on both grids, and with box = K in the source frame the largest signed_other
    of the edge part inside K. Never raises: any error -> ('none', None)."""
    try:
        return _joint_edge_plan(chart, strip, triangles, side, box)
    except Exception:  # noqa: BLE001  the trigger only computes; the r3p4 joint check then judges the end
        return 'none', None


def _joint_edge_plan(chart, strip, triangles, side, box):
    if triangles is None or side is None:
        return 'none', None
    tri = np.asarray(triangles, float); counts = {}
    for n in (DEPTH_SAMPLES, DENSE_SAMPLES):
        X = strip_samples(chart, strip, n); own = np.asarray(side(X), float) <= JOINT_TOL_M
        d, ins = _on_triangles(X, tri); counts[n] = int((own & ~ins).sum())
        if own.sum() >= MIN_OWN_SIDE_SAMPLES:
            break
    else:
        return 'none', None
    if n == DEPTH_SAMPLES:  # the record holds both grids
        X2 = strip_samples(chart, strip, DENSE_SAMPLES); own2 = np.asarray(side(X2), float) <= JOINT_TOL_M
        counts[DENSE_SAMPLES] = int((own2 & ~_on_triangles(X2, tri)[1]).sum())
    off = own & ~ins
    if not off.any():
        return 'none', None
    col = own & (np.arange(n * n) % n == 0)
    if not col.any() or not (ins[col].all() and (d[col] <= JOINT_TOL_M).all()):
        return 'none', None
    segs = _boundary_segments(tri); who, _ = _nearest_segment(X[off], segs); used = sorted(set(int(x) for x in who))
    a0, b0, c0 = segs[used[0]]; u0 = (b0 - a0) / np.linalg.norm(b0 - a0)
    for j in used[1:]:
        a, b, _ = segs[j]; u = (b - a) / np.linalg.norm(b - a)
        ang = float(np.degrees(np.arccos(min(1., abs(float(u @ u0))))))
        if ang > JOINT_EDGE_ANGLE_DEG or max(float(np.linalg.norm(np.cross(x - a0, u0))) for x in (a, b)) > JOINT_EDGE_LINE_M:
            return 'corner', len(used)
    nj = np.cross(b0 - a0, c0 - a0); m = np.cross(u0, nj); m = m / np.linalg.norm(m)
    if (c0 - a0) @ m > 0:
        m = -m
    plan = dict(p_e=[float(x) for x in a0], u_e=[float(x) for x in u0], m_e=[float(x) for x in m], grid=int(n),
                segments=[[[float(x) for x in segs[j][0]], [float(x) for x in segs[j][1]]] for j in used],
                off_joint_samples={'21x21': counts.get(DEPTH_SAMPLES), '201x201': counts.get(DENSE_SAMPLES)})
    if box is not None:
        lo, hi = (np.asarray(x, float) for x in box); vals = []
        for j in used:
            a, b, _ = segs[j]; ab = b - a; ln = float(np.linalg.norm(ab)); iv = _box_clip(a, ab / ln, lo, hi)
            if iv is not None and iv[1] >= 0 and iv[0] <= ln:
                s0, s1 = max(iv[0], 0.), min(iv[1], ln)
                vals += [float(v) for v in np.asarray(side(np.array([a + s0 * ab / ln, a + s1 * ab / ln])), float)]
        plan['edge_in_K_max_signed_other_m'] = max(vals) if vals else None
    return 'trim', plan


def extension_on_joint_kept(chart, strip, triangles, fault_id, side, plan):
    p, m = np.asarray(plan['p_e'], float), np.asarray(plan['m_e'], float); tri = np.asarray(triangles, float)
    X = strip_samples(chart, strip); kept = (X - p) @ m <= 0.; d, inside = _on_triangles(X, tri)
    own = (np.asarray(side(X), float) <= JOINT_TOL_M) & kept
    rec = dict(samples=int(len(X)), max_distance_m=float(d.max()), outside=int((~inside).sum()), tol_m=JOINT_TOL_M,
               kept_samples=int(kept.sum()), joint_edge_trim=True,
               own_side_samples=int(own.sum()), own_side_max_distance_m=float(d[own].max()) if own.any() else None,
               own_side_outside=int((~inside[own]).sum()), min_own_side_samples=MIN_OWN_SIDE_SAMPLES)
    n_, dmax, out_ = rec['own_side_samples'], rec['own_side_max_distance_m'], rec['own_side_outside']; where = 'own side'
    if n_ < MIN_OWN_SIDE_SAMPLES:
        X2 = strip_samples(chart, strip, DENSE_SAMPLES); own2 = (np.asarray(side(X2), float) <= JOINT_TOL_M) & ((X2 - p) @ m <= 0.)
        d2, in2 = _on_triangles(X2[own2], tri) if own2.any() else (np.zeros(0), np.zeros(0, bool))
        n_, dmax, out_ = int(own2.sum()), (float(d2.max()) if len(d2) else None), int((~in2).sum()); where = 'own side, dense grid'
        rec['own_side_dense'] = dict(layer=2, grid=DENSE_SAMPLES, samples=int(len(X2)), own_side_samples=n_, own_side_max_distance_m=dmax,
                                     own_side_outside=out_)
        if n_ < MIN_OWN_SIDE_SAMPLES:
            rec['own_side_dense'].update(layer=3, full_strip_fallback=True)
            n_ = int(kept.sum()); dmax = float(d[kept].max()) if kept.any() else None; out_ = int((~inside[kept]).sum())
            where = 'kept strip, r3\' fallback'
    if dmax is None or out_ or dmax > JOINT_TOL_M:
        raise ValueError('fault %s end: the extension leaves its joint surface after the joint-edge trim (max %s m, %d sample(s) outside it, %s)'
                         % (fault_id, 'n/a' if dmax is None else '%.3e' % dmax, out_, where))
    return rec

def mouth_classifier(chart, regular, strips, shift):
    from .network_panels import envelope
    flat = [pc for P_ in regular for pc in P_.pieces()]; sh = np.zeros(3) if shift is None else np.asarray(shift, float)

    def label(q):
        _, _, cnt = envelope(flat, q, 'L'); lab = np.where(cnt > 0, 0, -1)
        for k, s_ in enumerate(strips):
            dq = q - s_['p_end']; lam = dq @ s_['t2']; a = dq @ s_['nu2']
            m = (lab == -1) & (lam > -STRIP_TOL_M) & (lam <= s_['L'] + STRIP_TOL_M) & (a >= -s_['margin'] - STRIP_TOL_M) & (a <= s_['depth'] + STRIP_TOL_M)
            lab[m] = k + 1
        return lab

    def classify(P):
        q = chart.locate(np.asarray(P, float) - sh)[0]
        return q, label(q)

    def by_foot(P):
        Ps = np.atleast_2d(np.asarray(P, float)) - sh; tau = Ps @ chart.dh
        best, q = np.full(len(Ps), np.inf), np.zeros((len(Ps), 2))
        for i in range(len(chart.length)):
            s_ = chart.sigma[i] + np.clip((Ps - chart.Q[i]) @ chart.u[i], 0., chart.length[i]); qi = np.c_[s_, tau]
            di = np.linalg.norm(Ps - chart.surface(qi), axis=1); upd = di < best; best[upd] = di[upd]; q[upd] = qi[upd]
        return q, best, label(q)
    classify.by_foot = by_foot
    return classify


def check_inputs(chart, piece, k, L, rho_q, shift, fault, moved, record, joint=_NO_JOINT, chart_overrun=None):
    """'corner' -> fail closed; 'trim' -> extension_on_joint_kept, and 'joint_edge_plan' (with K in the source frame, the
    fault id) is added to the inputs for build.py / structure_checks; Without a joint (unit tests): an overrun raises
    the chart message."""
    i = 0 if k == 0 else -1; t2 = -piece.t[0] if k == 0 else piece.t[-1]; nu2 = piece.nu[0] if k == 0 else piece.nu[-1]
    pan = chart.panel(np.array([piece.p[i][0]]))[0]
    T3 = chart.u[pan] * t2[0] + chart.dh * t2[1]; T3 = T3 / np.linalg.norm(T3)
    X = chart.surface(piece.p[i][None])[0] + (0. if shift is None else np.asarray(shift, float))
    w = max(abs(float(piece.oL[i][0])), abs(float(piece.oR[i][0])))
    if rho_q is not None:
        lam, a = np.meshgrid(np.linspace(0., L, 21), np.linspace(-piece.margin, float(piece.depth_last[i]), 21))
        q = piece.p[i] + np.outer(lam.ravel(), t2) + np.outer(a.ravel(), nu2)
        w += float(np.max(np.abs(np.asarray(rho_q(q), float))))
    strip = dict(p_end=np.array(piece.p[i], float), t2=np.asarray(t2, float), nu2=np.asarray(nu2, float), L=float(L),
                 depth=float(piece.depth_last[i]), margin=float(piece.margin))
    sh = np.zeros(3) if shift is None else np.asarray(shift, float)
    side = (lambda Xs: signed_other(fault, moved, Xs + sh)) if 'field' in fault else None
    Xw = strip_samples(chart, strip) + sh; grow = w + BOX_MARGIN_M; K = (Xw.min(0) - grow, Xw.max(0) + grow)
    plan = None
    if joint is _NO_JOINT:
        if chart_overrun is not None:
            raise ValueError(chart_overrun['message'])
        joint_rec = None
    else:
        status, plan = joint_edge_plan(chart, strip, joint, side, box=(K[0] - sh, K[1] - sh))
        if status != 'trim' and chart_overrun is not None:
            raise ValueError(chart_overrun['message'])
        if status == 'corner':
            raise ValueError('fault %s end: joint-edge trim: the off-joint samples lie on more than one joint boundary edge (%d boundary segments)' % (fault['id'], plan))
        if status == 'trim':
            joint_rec = extension_on_joint_kept(chart, strip, joint, fault['id'], side, plan)
            plan.update(fault=fault['id'], K_src=[[float(x) for x in K[0] - sh], [float(x) for x in K[1] - sh]],
                        chart_overrun=None if chart_overrun is None else {k_: v for k_, v in chart_overrun.items() if k_ != 'message'})
            if chart_overrun is not None:
                lam_end = ((float(chart.sigma[-1]) if t2[0] > 0 else float(chart.sigma[0])) - float(piece.p[i][0])) / float(t2[0])
                Xs = strip_samples(chart, strip); lam = np.tile(np.linspace(0., float(L), DEPTH_SAMPLES), DEPTH_SAMPLES)
                keep = ((Xs - np.asarray(plan['p_e'], float)) @ np.asarray(plan['m_e'], float) <= 0.) & (np.asarray(side(Xs), float) <= JOINT_TOL_M)
                if (lam[keep] > lam_end + 1e-9).any():
                    raise ValueError('fault %s end: joint-edge trim: the kept extension leaves the trace chart (lambda %.4f beyond the chart end %.4f m)'
                                     % (fault['id'], float(lam[keep].max()), lam_end))
                plan['L_build_m'] = float(lam_end)
        else:
            plan = None; joint_rec = extension_on_joint(chart, strip, joint, fault['id'], side=side)
    out = dict(fault=fault['id'], moved=bool(moved), X_end=X, T3=T3, w_m=w, L_m=float(L), record=record, strip=strip,
               shift=None if shift is None else np.asarray(shift, float), joint=joint_rec,
               contact_box=K)
    if plan is not None:
        out['joint_edge_plan'] = plan
    return out


def _mesh(m, md):
    mesh = m.to_mesh64(); V = np.asarray(mesh.vert_properties)[:, :3]; T = np.asarray(mesh.tri_verts).astype(np.int64)
    oid = np.empty(len(T), np.int64)
    for s, e, o in zip(mesh.run_index[:-1], mesh.run_index[1:], mesh.run_original_id): oid[s // 3:e // 3] = o
    return V, T, oid


def _components(T):
    """connected components of a triangle set through shared vertices (union-find)"""
    parent = np.arange(len(T))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    first = {}
    for t, tri in enumerate(T):
        for v in tri:
            v = int(v)
            if v in first:
                a, b = find(t), find(first[v])
                if a != b: parent[a] = b
            else:
                first[v] = t
    return np.array([find(t) for t in range(len(T))])


def _mouth_r2(V, F, X0, T3, dv, w, host_mesh):
    if not len(F):
        return dict(h_m=MOUTH_H_MIN_M, beyond_faces=0, gate_A=True, gate_B=True, max_plane_distance_beyond_m=0.)
    lam = (V[F].mean(1) - X0) @ T3; beyond = lam > 0
    h = MOUTH_H_MIN_M
    if beyond.any():
        Pb = V[F[beyond]].reshape(-1, 3); blo, bhi = Pb.min(0) - 1e-6, Pb.max(0) + 1e-6
        HV, HF = np.asarray(host_mesh[0], float), np.asarray(host_mesh[1], np.int64); tri = HV[HF]
        near = ((tri.max(1) >= blo) & (tri.min(1) <= bhi)).all(1)
        if near.any(): h = max(h, float(np.linalg.norm(tri[near] - np.roll(tri[near], 1, 1), axis=2).max()))
    comp = _components(F); ok_comp = set(comp[~beyond])
    loose = sorted(set(comp[beyond]) - ok_comp)
    slivers = [c for c in loose if dv[np.unique(F[comp == c])].max() <= h]; bad = [c for c in loose if c not in slivers]
    vb = np.unique(F[beyond]); dist = dv[vb] if len(vb) else np.zeros(0)
    return dict(h_m=h, beyond_faces=int(beyond.sum()), components=int(len(set(comp))), end_face_slivers=len(slivers), components_beyond_without_mouth=len(bad),
                max_plane_distance_beyond_m=float(dist.max()) if len(dist) else 0., gate_A=not bad, gate_B=bool(not len(dist) or dist.max() <= w + h))


def mouth_check(host, host_id, cavity, fault, moved, inp, host_mesh):
    """Its vertices are classified in the group's chart coordinates (inp['classify'], mouth_classifier): the group's own
    mouth, the extension of one of the group's fault ends, or neither. The extension is the true joint
    (extension_on_joint), so a host surface cutting it is a real exposure: this end's part is RECORDED (record
    'exposure'), not failed. A vertex in neither fails closed. Returns the record; raises ValueError (the record on
    the error) on a vertex in neither."""
    import manifold3d as md
    X0, T3, R = np.asarray(inp['X_end'], float), np.asarray(inp['T3'], float), inp['L_m'] + inp['w_m'] + BOX_MARGIN_M
    box = md.Manifold.cube((2 * R, 2 * R, 2 * R)).translate(tuple(X0 - R))
    V, T, oid = _mesh((host ^ box) ^ (cavity ^ box), md)
    F = T[oid == host_id]; w = float(inp['w_m'])
    n, d = own_halfspace(fault, moved); dv = np.abs(V @ n - d)
    rec = dict(box_half_m=float(R), w_m=w, opened_faces=int(len(F)), r2=_mouth_r2(V, F, X0, T3, dv, w, host_mesh))
    rec['h_m'] = rec['r2']['h_m']
    vid = np.unique(F); own = inp['strip_index'] + 1
    if len(vid):
        q, lab = inp['classify'](V[vid])
    else:
        q, lab = np.zeros((0, 2)), np.zeros(0, np.int64)
    foot = None
    redo = np.flatnonzero(lab < 0)
    if len(redo) and hasattr(inp['classify'], 'by_foot'):
        q2, d2, lab2 = inp['classify'].by_foot(V[vid[redo]]); ok = (d2 <= w + FOOT_MARGIN_M) & (lab2 >= 0)
        if ok.any():
            q, lab = np.array(q, float), np.array(lab); q[redo[ok]] = q2[ok]; lab[redo[ok]] = lab2[ok]
            foot = dict(vertices=int(ok.sum()), to_mouth=int((lab2[ok] == 0).sum()), to_own_extension=int((lab2[ok] == own).sum()),
                        to_other_extension=int(((lab2[ok] > 0) & (lab2[ok] != own)).sum()), max_distance_m=float(d2[ok].max()), limit_m=w + FOOT_MARGIN_M)
    rec.update(mouth_vertices=int((lab == 0).sum()), extension_vertices=int((lab > 0).sum()), own_extension_vertices=int((lab == own).sum()),
               neither_vertices=int((lab < 0).sum()))
    m = lab == own
    if m.any():
        P = V[vid[m]]; s_ = inp['strip']; dq = q[m] - s_['p_end']
        fm = np.isin(F, vid[m]).all(1)
        rec['exposure'] = dict(faces=int(fm.sum()), vertices=int(m.sum()), sigma=[float(q[m, 0].min()), float(q[m, 0].max())],
                               lambda_m=[float((dq @ s_['t2']).min()), float((dq @ s_['t2']).max())],
                               plane_distance_m=[float(dv[vid[m]].min()), float(dv[vid[m]].max())],
                               bbox=[P.min(0).round(4).tolist(), P.max(0).round(4).tolist()])
    else:
        rec['exposure'] = None
    if foot is not None: rec['neither_by_foot'] = foot
    if rec['neither_vertices']:
        bad = V[vid[lab < 0]]
        rec['neither'] = dict(bbox=[bad.min(0).round(4).tolist(), bad.max(0).round(4).tolist()], plane_distance_m=[float(dv[vid[lab < 0]].min()), float(dv[vid[lab < 0]].max())])
        err = ValueError('fault %s end: the extension opens the host surface outside both the mouth and the extension (%d vertices)'
                         % (fault['id'], rec['neither_vertices']))
        err.record = rec; raise err
    return rec


def contact_check(cavity, others, inp):
    """No box: fail closed."""
    import manifold3d as md
    X0, T3 = np.asarray(inp['X_end'], float), np.asarray(inp['T3'], float); off = float(T3 @ X0)
    if inp.get('contact_box') is None:
        raise ValueError('fault %s end: no contact box (check_inputs r3\'\'\')' % inp['fault'])
    lo, hi = (np.asarray(x, float) for x in inp['contact_box'])
    K = md.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
    E, B = cavity.trim_by_plane(tuple(T3), off), cavity.trim_by_plane(tuple(-T3), -off)

    def judge(E_, B_):
        rows_ = []
        for j, C in others:
            vE = float((E_ ^ C).volume()); vB = float((B_ ^ C).volume()) if vE > CONTACT_TOL_M3 else None
            rows_.append(dict(other=int(j), beyond_m3=vE, behind_m3=vB, new=bool(vE > CONTACT_TOL_M3 and not (vB or 0.) > CONTACT_TOL_M3)))
        return rows_
    rows, rows0 = judge(E ^ K, B ^ K), judge(E, B)
    new = [r for r in rows if r['new']]
    rec = dict(others=len(rows), contacts_beyond=sum(r['beyond_m3'] > CONTACT_TOL_M3 for r in rows), new_contacts=len(new),
               rows=[r for r in rows if r['beyond_m3'] > CONTACT_TOL_M3], box_m=[lo.tolist(), hi.tolist()],
               unboxed=dict(contacts_beyond=sum(r['beyond_m3'] > CONTACT_TOL_M3 for r in rows0), new_contacts=sum(r['new'] for r in rows0),
                            rows=[r for r in rows0 if r['beyond_m3'] > CONTACT_TOL_M3]))
    if new:
        raise ValueError('fault %s end: the extension reaches another cavity (%d new contact(s), %.3g m3)' % (inp['fault'], len(new), new[0]['beyond_m3']))
    return rec


def structure_checks(host, host_id, solids, groups, frames, host_mesh):
    by_id = {f['id']: f for f in frames['faults']}
    for gi, ends in groups:
        others = [(j, s) for j, s in enumerate(solids) if j != gi]
        for inp in ends:
            f = by_id[inp['fault']]; rec = inp['record'].setdefault('structure', {})
            if inp.get('joint') is not None: rec['joint'] = inp['joint']
            if inp.get('joint_edge_plan') is not None: rec['joint_edge'] = joint_edge_record(solids[gi], inp, host_mesh)
            rec['mouth'] = mouth_check(host, host_id, solids[gi], f, inp['moved'], inp, host_mesh)
            rec['contact'] = contact_check(solids[gi], others, inp)


def _point_triangle_distance(P, A, B, C, chunk=256):
    """exact distance of each point P (m, 3) to the nearest triangle (A[k], B[k], C[k]) and that triangle's signed plane distance
    ((P - A).n, n the unit normal of the triangle's winding); brute force in chunks of triangles"""
    best = np.full(len(P), np.inf); sgn = np.zeros(len(P)); p = P[:, None]
    dot = lambda u, v: np.einsum('...i,...i', u, v)  # noqa: E731

    def seg(a, b):
        ab = b - a; t = np.clip(dot(p - a, ab) / dot(ab, ab), 0., 1.)
        return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)
    for lo in range(0, len(A), chunk):
        a, b, c = A[None, lo:lo + chunk], B[None, lo:lo + chunk], C[None, lo:lo + chunk]
        n = np.cross(b - a, c - a); n = n / np.linalg.norm(n, axis=-1)[..., None]; h = dot(p - a, n); F = p - h[..., None] * n
        inside = (dot(np.cross(b - a, F - a), n) >= 0) & (dot(np.cross(c - b, F - b), n) >= 0) & (dot(np.cross(a - c, F - c), n) >= 0)
        dist = np.where(inside, np.abs(h), np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a)))
        j = dist.argmin(1); dm = dist[np.arange(len(P)), j]; upd = dm < best; best[upd] = dm[upd]; sgn[upd] = h[np.arange(len(P)), j][upd]
    return best, sgn


def joint_edge_record(solid, inp, host_mesh):
    """Recorded: cap triangles, area, the smallest host distance."""
    plan = inp['joint_edge_plan']; sh = np.zeros(3) if inp['shift'] is None else np.asarray(inp['shift'], float)
    p, m = np.asarray(plan['p_e'], float) + sh, np.asarray(plan['m_e'], float); lo, hi = (np.asarray(x, float) for x in inp['contact_box'])
    mesh = solid.to_mesh64(); X = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64); T = np.asarray(mesh.tri_verts, np.int64)
    on = (np.abs((X - p) @ m) <= 1e-9) & ((X >= lo) & (X <= hi)).all(1); cap = on[T].all(1)
    rec = {k: v for k, v in plan.items() if k not in ('fault', 'owner')}
    tri = X[T[cap]]
    rec.update(cap_triangles=int(cap.sum()), cap_area_m2=float(.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()))
    if not cap.any():
        rec['cap_min_host_distance_m'] = None
        return rec
    P = np.vstack([X[np.unique(T[cap])], tri.mean(1)]); HV, HT = (np.asarray(x) for x in host_mesh); H = HV[HT.astype(np.int64)]
    near = ((H.max(1) >= P.min(0) - .05) & (H.min(1) <= P.max(0) + .05)).all(1)
    # An empty surface neighbourhood does not classify a point as outside:
    # a cap deeper than 5 cm can be inside the finite host. Keep the local
    # query unchanged when nonempty; otherwise use one finite, chunked pass.
    if near.any(): H = H[near]
    if not len(H):
        raise ValueError('fault %s end: joint-edge cap outside the host (no host triangle within 5 cm)' % inp['fault'])
    d, s = _point_triangle_distance(P, H[:, 0], H[:, 1], H[:, 2])
    rec['cap_min_host_distance_m'] = float(d.min())
    if (s >= 0).any() or d.min() < CAP_DEPTH_MIN_M:
        raise ValueError('fault %s end: joint-edge cap reaches the host surface (min distance %.3e m, %d point(s) outside)' % (inp['fault'], d.min(), int((s >= 0).sum())))
    return rec


def make_end_trim(plans):
    """Per plan: guard -- no vertex of another path's faces inside the cutter = K_src cut to {(x - p_e).m_e >= 0} (else
    ValueError); solid - cutter; every new triangle must lie on the trim plane within 1e-9 m (else the cavity reaches
    K: ValueError); the new triangles go to the trimmed path (as replay_cavity's caps): role 'station_end',
    construction 'joint_edge_trim', the segment of the nearest station of that path, face ids after the largest.
    Returns (solid, tables, record)."""
    def trim(solid, tables):
        import manifold3d as md
        tables = {int(k): dict(v) for k, v in tables.items()}; recs = []
        for plan in plans:
            lo, hi = (np.asarray(x, float) for x in plan['K_src']); p, m = np.asarray(plan['p_e'], float), np.asarray(plan['m_e'], float)
            o_id, S, sid = plan['owner']; S = np.asarray(S, float); sid = np.asarray(sid)
            mesh = solid.to_mesh64(); X = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64); T = np.asarray(mesh.tri_verts, np.int64)
            oid = np.empty(len(T), np.int64)
            for a, b, o in zip(mesh.run_index[:-1], mesh.run_index[1:], mesh.run_original_id):
                oid[int(a) // 3:int(b) // 3] = int(o)
            Vo = np.unique(T[oid != int(o_id)])
            if len(Vo) and (((X[Vo] >= lo) & (X[Vo] <= hi)).all(1) & ((X[Vo] - p) @ m > 1e-9)).any():
                raise ValueError('fault %s end: joint-edge trim would cut another path\'s cavity' % plan['fault'])
            cutter = md.Manifold.cube(tuple(float(x) for x in hi - lo)).translate(tuple(float(x) for x in lo)).trim_by_plane(
                tuple(float(x) for x in m), float(m @ p))
            v0 = solid.volume(); out = solid - cutter
            mesh = out.to_mesh64(); X = np.array(np.asarray(mesh.vert_properties)[:, :3], dtype=np.float64); T = np.array(mesh.tri_verts, dtype=np.int64)
            fid = np.array(mesh.face_id, dtype=np.int64); oid = np.empty(len(T), np.int64)
            for a, b, o in zip(mesh.run_index[:-1], mesh.run_index[1:], mesh.run_original_id):
                oid[int(a) // 3:int(b) // 3] = int(o)
            new = ~np.isin(oid, list(tables)); area = 0.
            if new.any():
                if float(np.abs((X[T[new]] - p) @ m).max()) > 1e-9:
                    raise ValueError('fault %s end: joint-edge trim: the cavity reaches the trim box K' % plan['fault'])
                C = X[T[new]].mean(1); j = np.linalg.norm(C[:, None] - S[None], axis=2).argmin(1)
                start = 1 + max((int(k) for v in tables.values() for k in v), default=-1); ids = np.arange(start, start + len(C))
                fid[new] = ids; oid[new] = int(o_id)
                for f_, sg in zip(ids, sid[j]):
                    tables[int(o_id)][int(f_)] = dict(source_segment_id=int(sg), role='station_end', construction='joint_edge_trim')
                tri = X[T[new]]; area = float(.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
            order = np.argsort(oid, kind='stable'); T, fid, oid = T[order], fid[order], oid[order]; ids_, first = np.unique(oid, return_index=True)
            solid = md.Manifold(md.Mesh64(X, np.ascontiguousarray(T.astype(np.uint64)), run_index=np.r_[first * 3, len(T) * 3].astype(np.uint64),
                                          run_original_id=ids_.astype(np.uint32), face_id=np.ascontiguousarray(fid.astype(np.uint64))))
            if solid.status() != md.Error.NoError or solid.volume() <= 0:
                raise ValueError('fault %s end: joint-edge trim: cavity is not a valid oriented manifold: %s' % (plan['fault'], solid.status()))
            recs.append(dict(fault=plan['fault'], path_original_id=int(o_id), trimmed_volume_m3=float(v0 - solid.volume()),
                             cap_triangles=int(new.sum()), cap_area_m2=area))
        return solid, tables, dict(trims=recs)
    return trim

def frame_record(frames, shift, records):
    """the build.json / prepare row record of a path's fault frame (None: not moved)"""
    if shift is None:
        return None
    moved = [r['fault'] for r in records if r['moved']]
    by_id = {f['id']: f for f in frames['faults']}
    return dict(shift_m=[float(x) for x in shift], faults=moved, exact_vector_enu=[by_id[i]['t_exact'] for i in moved])


def replay_cavity(solid, tables, frames, joint_id, block, owners):
    """the slip replayed on a cavity built in its source frame: for every fault that acted on the joint, in event order,
    translate by t when the group lies on that fault's moving block, then keep the group's own closed half-space
    (Manifold.trim_by_plane). Returns (solid, tables, record); a cavity no fault acted on comes back unchanged with
    record None."""
    acted = [f for f in frames['faults'] if joint_id in f['moved']]
    if not acted:
        return solid, tables, None
    import manifold3d as md
    block = set(block or ())
    v0, ops = solid.volume(), []
    for f in acted:
        moved = f['id'] in block
        if moved:
            solid = solid.translate(tuple(float(x) for x in f['t']))
        n, d = own_halfspace(f, moved); before = solid.volume()
        solid = solid.trim_by_plane(tuple(float(x) for x in n), float(d))
        ops.append(dict(fault=f['id'], moved=moved, trimmed_volume_m3=float(before - solid.volume())))
    mesh = solid.to_mesh64()
    # copies: the to_mesh64 arrays are read-only views and Mesh64 needs writable arrays
    X = np.array(np.asarray(mesh.vert_properties)[:, :3], dtype=np.float64); T = np.array(mesh.tri_verts, dtype=np.int64)
    fid = np.array(mesh.face_id, dtype=np.int64)
    oid = np.empty(len(T), np.int64)
    for a, b, o in zip(mesh.run_index[:-1], mesh.run_index[1:], mesh.run_original_id):
        oid[int(a) // 3:int(b) // 3] = int(o)
    tables = {int(k): dict(v) for k, v in tables.items()}
    cap = ~np.isin(oid, list(tables)); caps = {}
    if cap.any():
        C = X[T[cap]].mean(1)
        best = np.full(len(C), np.inf); who = np.zeros(len(C), np.int64); seg = np.zeros(len(C), np.int64)
        for o, S, sid in owners:
            S = np.asarray(S, float); sid = np.asarray(sid)
            for lo in range(0, len(C), 4096):
                dist = np.linalg.norm(C[lo:lo + 4096, None] - S[None], axis=2); j = dist.argmin(1); dm = dist[np.arange(len(j)), j]
                better = dm < best[lo:lo + 4096]; idx = np.flatnonzero(better) + lo
                best[idx] = dm[better]; who[idx] = int(o); seg[idx] = sid[j[better]]
        start = 1 + max((int(k) for v in tables.values() for k in v), default=-1)
        new = np.arange(start, start + len(C)); fid[cap] = new; oid[cap] = who
        for f_, o, sg in zip(new, who, seg):
            tables[int(o)][int(f_)] = dict(source_segment_id=int(sg), role='fault_cap', construction='fault_trim')
            caps[int(o)] = caps.get(int(o), 0) + 1
    order = np.argsort(oid, kind='stable'); T, fid, oid = T[order], fid[order], oid[order]
    ids, first = np.unique(oid, return_index=True)
    out = md.Manifold(md.Mesh64(X, np.ascontiguousarray(T.astype(np.uint64)),
                                run_index=np.r_[first * 3, len(T) * 3].astype(np.uint64), run_original_id=ids.astype(np.uint32),
                                face_id=np.ascontiguousarray(fid.astype(np.uint64))))
    if out.status() != md.Error.NoError or out.volume() <= 0:
        raise ValueError('fault replay: cavity is not a valid oriented manifold: ' + str(out.status()))
    for f, op in zip(acted, ops):
        n, d = own_halfspace(f, f['id'] in block)
        op['other_block_volume_m3'] = float(out.trim_by_plane(tuple(float(-x) for x in n), float(-d + 1e-6)).volume())
    return out, tables, dict(operations=ops, volume_before_m3=float(v0), volume_after_m3=float(out.volume()),
                             cap_triangles={str(k): v for k, v in sorted(caps.items())})


def save_cavity(path, solid, tables):
    """verification output (build.py --save-fault-cavities): vertices, triangles, per-triangle original id, face id and role"""
    m = solid.to_mesh64()
    oid = np.empty(len(m.tri_verts), np.int64)
    for a, b, o in zip(m.run_index[:-1], m.run_index[1:], m.run_original_id):
        oid[int(a) // 3:int(b) // 3] = int(o)
    fid = np.asarray(m.face_id, np.int64)
    role = np.array([tables.get(int(o), {}).get(int(f), {}).get('role', '') for o, f in zip(oid, fid)], dtype='<U32')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vertices=np.asarray(m.vert_properties)[:, :3], faces=np.asarray(m.tri_verts, np.int64), original_id=oid,
                        face_id=fid, role=role)


def extend_path(points, per_station, span_segment_ids, end, point):
    """the path arrays with one virtual station appended (end -1) or prepended (end 0) at `point`: every per-station array
    repeats its end row, the new span takes the end span's segment id. per_station: {name: (n, ...) array};
    span_segment_ids: (n - 1,). Returns (points, per_station, span_segment_ids)."""
    p = np.asarray(point, float)[None]
    if end == 0:
        return (np.r_[p, points], {k: np.r_[a[:1], a] for k, a in per_station.items()},
                np.r_[span_segment_ids[:1], span_segment_ids])
    return (np.r_[points, p], {k: np.r_[a, a[-1:]] for k, a in per_station.items()},
            np.r_[span_segment_ids, span_segment_ids[-1:]])
