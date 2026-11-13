"""Fault trace labels.

Fault trace = declared fault plane ∩ uncut host surface; visibility by a two-sided depth test on the rendered (cut)
mesh; U by the joints' u_windows with the same constants and no cavity; positives per fault = skeleton of the union of
its branches' visible non-U centrelines. merge_labels adds them to the joint labels without changing any joint
positive. The joint code (frgb/trace_labels.py) is used, never edited."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import trace_labels as tl
from .metrics import graph_node_weights, skeletonize

PLANE_TOL_M = 1e-6          # resolved interface on the declared plane; label-to-plane and generator cross-check
ON_PLANE_NUDGE_M = 1e-13    # host vertices exactly on the plane count as the positive side (consistent per vertex)
FAULT_PATH_BASE = 10 ** 6   # sample keys of fault branches (never a joint path index)


def fault_planes(source):
    """Fault planes of the case source (build.json 'source'): dict(status, faults, reasons). status: 'no_declaration' |
    'no_faults' | 'ok' | 'unsupported'."""
    source = Path(source)
    decl, hops = source / 'declaration.json', 0
    while not decl.is_file() and (decl.parent / 'refinement.json').is_file() and hops < 4:
        # a derived host (zone / level refinement) names its scene in refinement.json 'source'
        decl, hops = Path(json.loads((decl.parent / 'refinement.json').read_text(encoding='utf8'))['source']) / 'declaration.json', hops + 1
    if not decl.is_file():
        return dict(status='no_declaration', faults=[], reasons=[])
    if not (source / 'package' / 'interfaces.json').is_file():
        source = decl.parent
    d = json.loads(decl.read_text(encoding='utf8'))
    geom = {g['id']: g for g in d.get('geometry', [])}
    events = [e for e in d.get('events', []) if e.get('kind') == 'fault']
    if not events:
        return dict(status='no_faults', faults=[], reasons=[], declaration=str(decl))
    from outcrop.modeling.kinematics import orientation_basis
    reasons, faults = [], []
    ij, inpz = source / 'package' / 'interfaces.json', source / 'package' / 'interfaces.npz'
    if not (ij.is_file() and inpz.is_file()):
        return dict(status='unsupported', faults=[], reasons=['declaration lists faults but package/interfaces.* is missing'],
                    declaration=str(decl))
    interfaces = json.loads(ij.read_text(encoding='utf8'))
    with np.load(inpz) as arrays:
        for e in events:
            pl = geom.get(e.get('fault'))
            if pl is None or pl.get('kind') != 'plane':
                reasons.append('%s: fault geometry is not a declared plane' % e['id']); continue
            if 'extent_m' in pl or not all(k in pl for k in ('strike_deg', 'dip_deg', 'point_m')):
                reasons.append('%s: plane with extent_m or without strike_deg / dip_deg / point_m' % e['id']); continue
            if float(e.get('motion', {}).get('opening_m', 0) or 0) != 0:
                reasons.append('%s: opening_m != 0' % e['id']); continue
            n = np.asarray(orientation_basis(pl['strike_deg'], pl['dip_deg']).normal, float)
            n /= np.linalg.norm(n)
            p0 = np.asarray(pl['point_m'], float)
            rows = [k for k, x in enumerate(interfaces) if x.get('feature_id') == e['id']]
            if not rows:
                reasons.append('%s: no resolved interface in the package' % e['id']); continue
            dist = max(float(np.abs((arrays['vertices_%d' % k] - p0) @ n).max()) for k in rows)
            if dist > PLANE_TOL_M:
                reasons.append('%s: resolved interface %.3g m off the declared plane' % (e['id'], dist)); continue
            faults.append(dict(id=e['id'], normal=n, point=p0, strike_deg=float(pl['strike_deg']), dip_deg=float(pl['dip_deg']),
                               motion=e.get('motion'), moving_side=e.get('moving_side'),
                               interface_ids=[interfaces[k]['id'] for k in rows], interface_max_plane_distance_m=dist))
    return dict(status='unsupported' if reasons else 'ok', faults=faults, reasons=reasons, declaration=str(decl))


def plane_section(vertices, faces, normal, point):
    """Polylines of plane ∩ triangle mesh: list of (points (k,3), host face of each step (k-1,)). Crossing points by linear
    interpolation on the edges with a sign change; chained through shared edges; a node of degree > 2 (non-manifold edge)
    ends its chains."""
    V = np.asarray(vertices, float); F = np.asarray(faces, np.int64)
    d = (V - point) @ normal
    d = np.where(np.abs(d) < ON_PLANE_NUDGE_M, ON_PLANE_NUDGE_M, d)
    pos = d > 0
    fs = pos[F]
    cf = np.flatnonzero(fs.any(1) & ~fs.all(1))
    if not len(cf):
        return []
    tri = F[cf]
    ea = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 1)          # (m, 3, 2)
    change = pos[ea[..., 0]] != pos[ea[..., 1]]                                   # exactly two per crossing face
    keys = np.sort(ea[change].reshape(-1, 2), axis=1)                            # (2m, 2) in face order
    uniq, node = np.unique(keys, axis=0, return_inverse=True)
    node = node.ravel().reshape(-1, 2)                                            # per face: its two crossing edges
    a, b = uniq[:, 0], uniq[:, 1]
    t = d[a] / (d[a] - d[b])
    P = V[a] + t[:, None] * (V[b] - V[a])
    deg = np.bincount(node.ravel(), minlength=len(uniq))
    adj = [[] for _ in range(len(uniq))]
    for s, (u, v) in enumerate(node):
        adj[u].append((v, s)); adj[v].append((u, s))
    used = np.zeros(len(node), bool)
    lines = []

    def walk(start):
        seq, steps, cur = [start], [], start
        while True:
            nxt = [(v, s) for v, s in adj[cur] if not used[s]]
            if not nxt or (deg[cur] > 2 and cur != start):
                break
            v, s = nxt[0]
            used[s] = True; seq.append(v); steps.append(s); cur = v
            if deg[cur] > 2 or v == start:
                break
        return seq, steps
    for start in np.r_[np.flatnonzero(deg != 2), np.arange(len(uniq))]:       # open chains first, then loops
        while any(not used[s] for _, s in adj[start]):
            seq, steps = walk(int(start))
            if len(steps):
                lines.append((P[seq], cf[np.asarray(steps)]))
    return lines


def subdivide(points, faces, cam, spacing=tl.SCREEN_SPACING_PX):
    """Insert points so that consecutive samples are <= spacing px apart on screen (where both ends project in front of the
    camera and one is near the frame); inserted points lie on the same face ∩ plane segment."""
    uv, z = tl.project(points, cam)
    ok = (z > cam['near']) & np.isfinite(uv).all(1)
    near = ok & tl.in_frame(np.where(ok[:, None], uv, -1e9), cam, pad=8.)
    L = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    k = np.where(ok[1:] & ok[:-1] & (near[1:] | near[:-1]) & np.isfinite(L), np.ceil(L / spacing), 1).astype(np.int64)
    k = np.maximum(k, 1)
    out, face = [points[:1]], []
    for i in range(len(points) - 1):
        f = (np.arange(1, k[i] + 1) / k[i])[:, None]
        out.append(points[i] + f * (points[i + 1] - points[i])); face.append(np.full(k[i], faces[i]))
    return np.concatenate(out), np.concatenate(face) if face else np.empty(0, np.int64)


def fault_samples(case, cam, planes):
    """Sample dicts of the fault branches (keys FAULT_PATH_BASE + running index): point, host_face (per step), fault id,
    branch, u (sample index), station_point."""
    V, F = case['host']['vertices'], case['host']['faces']
    out, info = {}, {}
    for fi, f in enumerate(planes):
        lines = plane_section(V, F, f['normal'], f['point'])
        length = 0.
        for pts, faces in lines:
            P, hf = subdivide(pts, faces, cam)
            key = FAULT_PATH_BASE + len(out)
            out[key] = dict(point=P, step_face=hf, fault=f['id'], fault_index=fi, path=key, branch=len(out),
                            u=np.arange(len(P), dtype=float), station_point=P.copy())
            length += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        info[f['id']] = dict(polylines=len(lines), trace_length_m=length,
                             plane_distance_max_m=max((float(np.abs((p - f['point']) @ f['normal']).max()) for p, _ in lines), default=0.))
    return out, info


def classify_fault_visibility(case, samples, cam, *, boxes=None):
    """Two-sided depth rule on the native (cut) mesh with the joints' eps: visible |hz - z| <= eps; occluded hz < z - eps;
    in_cavity hz > z + eps or no hit (the point is not on the rendered surface)."""
    nat = case['native']
    all_uv, owner = [], []
    for k, res in samples.items():
        uv, z = tl.project(res['point'], cam)
        ok = np.isfinite(res['point']).all(1) & (z > cam['near'])
        res['uv'], res['z'] = uv, z
        res['in_frame'] = ok & tl.in_frame(np.where(np.isfinite(uv), uv, -1e9), cam)
        all_uv.append(uv[res['in_frame']]); owner.append((k, np.flatnonzero(res['in_frame'])))
    uv = np.concatenate(all_uv) if all_uv else np.empty((0, 2))
    hz, hf = tl.first_hits(nat['vertices'], nat['faces'], cam, uv, boxes=boxes) if len(uv) else (np.empty(0), np.empty(0, np.int64))
    j = 0
    for k, sel in owner:
        res = samples[k]
        n = len(res['u'])
        res['hit_z'] = np.full(n, np.nan); res['hit_face'] = np.full(n, -1, np.int64)
        res['hit_z'][sel] = hz[j:j + len(sel)]; res['hit_face'][sel] = hf[j:j + len(sel)]; j += len(sel)
        eps = np.maximum(tl.EPS_FLOOR_M, tl.EPS_PIXEL_FRACTION * res['z'] / cam['fpx'])
        h, z = res['hit_z'], res['z']
        fin = np.isfinite(h)
        res['eps'] = eps
        res['visible'] = res['in_frame'] & fin & (np.abs(h - z) <= eps)
        res['occluded'] = res['in_frame'] & fin & (h < z - eps)
        res['in_cavity'] = res['in_frame'] & ~res['visible'] & ~res['occluded']
        res['visible_own_cavity_only'] = np.zeros(n, bool); res['literal_extra'] = np.zeros(n, bool)
    gz = case.get('ground_z_m')
    if gz is not None:
        for res in samples.values():
            p = res['point']
            below = np.isfinite(p).all(1)
            below[below] = p[below, 2] < gz
            res['below_ground'] = res['in_frame'] & below
            for k in ('in_frame', 'visible', 'occluded', 'in_cavity'):
                res[k] = res[k] & ~below
    return samples


def fault_labels(case, samples, cam, rgb, *, cpu_raster, render_rock, aligned, ground=None, sky=None, export_windows=False, gid0=0):
    shape = render_rock.shape
    nat = case['native']
    roles, path_of_face = nat['roles'], nat['path_of_face']
    hit = cpu_raster['face'] >= 0
    rock = hit & render_rock
    scene = tl.scene_nonscored(shape, ground, sky)
    host_valid = rock & aligned & ~scene
    L255 = tl.luma(rgb) * 255.
    centres = {k: tl.visible_centre(r, shape) for k, r in samples.items()}
    u_line = np.zeros(shape, bool); occluded_line = np.zeros(shape, bool); transition = np.zeros(shape, bool)
    positive_of, windows, rows = {}, [], []
    member = dict(band_pixels=[], ring_pixels=[])
    for k, res in samples.items():
        others = np.zeros(shape, bool)
        for j, c in centres.items():
            if j != k and samples[j]['fault'] == res['fault']:
                others |= c
        excl = ndi.distance_transform_edt(~others) <= tl.U_K_PX if others.any() else None
        uw = tl.u_windows(res, L255, cpu_raster['face'], roles, path_of_face, [], host_valid, shape, ring_exclude=excl,
                          export=export_windows)
        res['window'] = uw['window']
        tab, centre = uw['table'], uw['centre']
        res['sample_u'] = np.zeros(len(res['u']), bool)
        if tab is not None:
            cw = uw['centre_window']
            is_u = np.zeros(shape, bool); is_u[centre] = tab['u'][cw[centre]]
            u_line |= is_u
            positive_of[k] = centre & ~is_u
            ok = res['window'] >= 0
            res['sample_u'][ok] = tab['u'][res['window'][ok]]
            gid_of = np.full(len(tab['present']), -1, np.int64)
            for w in np.flatnonzero(tab['present']):
                gid_of[w] = gid0 + len(windows)
                row = dict(path=int(k), window=int(w), band_pixels=int(tab['band_pixels'][w]), ring_pixels=int(tab['ring_pixels'][w]),
                           contrast=float(tab['contrast'][w]) if np.isfinite(tab['contrast'][w]) else None, u=bool(tab['u'][w]),
                           branch=int(res['branch']), ring_excluded=int(tab['ring_excluded'][w]), source_type='fault', fault=res['fault'])
                if export_windows:
                    row['gid'] = int(gid_of[w])
                windows.append(row)
            if export_windows:
                for name in ('band_pixels', 'ring_pixels'):
                    lin, lw = uw[name]
                    member[name].append((lin, gid_of[lw]))
        else:
            positive_of[k] = np.zeros(shape, bool)
        occluded_line |= tl.rasterise_samples(res, res['occluded'], shape)
        tr, _, _ = tl._transitions_and_unresolved(res, cam, shape)
        transition |= tr
        cav = res['in_cavity'] & res['in_frame']
        if cav.any():
            transition |= tl._disc(res['uv'][cav], shape, tl.TRANSITION_RADIUS_PX)
        w = tl._sample_weights(res)
        rows.append(dict(fault=res['fault'], branch=int(res['branch']), below_ground_px=tl._below_ground_px(res), projected_in_frame_px=float(w[res['in_frame']].sum()),
                         visible_px=float(w[res['visible']].sum()), occluded_px=float(w[res['occluded']].sum()),
                         in_cavity_px=float(w[cav].sum()), visible_u_px=float(w[res['visible'] & res['sample_u']].sum())))
    positive = np.zeros(shape, bool)
    for fid in sorted({r['fault'] for r in samples.values()}):
        union = np.zeros(shape, bool)
        for k, r in samples.items():
            if r['fault'] == fid:
                union |= positive_of[k]
        positive |= skeletonize(union) if union.any() else union
    u_ignore = ndi.distance_transform_edt(~u_line) <= tl.U_IGNORE_RADIUS_PX if u_line.any() else np.zeros(shape, bool)
    reason = np.zeros(shape, np.uint8)
    reason[u_ignore] |= tl.IGNORE_U
    reason[transition] |= tl.IGNORE_TRANSITION
    return dict(positive=positive, reason=reason, u_line=u_line, occluded_line=occluded_line, windows=windows, member=member,
                rows=rows)


def merge_labels(J, F, export_windows=False):
    """Returns (merged labels with J's keys, source_type uint8). F None: J unchanged, source_type = J.target."""
    st = J['target'].astype(np.uint8)
    if F is None:
        return J, st
    out = dict(J)
    reason = np.where(J['target'], J['ignore_reason'], J['ignore_reason'] | F['reason']).astype(J['ignore_reason'].dtype)
    valid = J['rock'] & (reason == 0)
    fpos = F['positive'] & valid
    target = J['target'] | fpos
    st = (J['target'].astype(np.uint8) + 2 * fpos.astype(np.uint8)).astype(np.uint8)
    out.update(target=target, valid=valid, ignore_reason=reason, u_line=J['u_line'] | F['u_line'],
               occluded_line=J['occluded_line'] | F['occluded_line'], windows=list(J['windows']) + F['windows'],
               positive_graph_length_px=float(graph_node_weights(target).sum()),
               all_visible_u=bool((J['u_line'] | F['u_line']).any() and not target.any()))
    if export_windows:
        pairs = {}
        for name, key in (('band_pixels', 'band_win'), ('ring_pixels', 'ring_win')):
            lin = np.concatenate([a for a, _ in F['member'][name]]) if F['member'][name] else np.empty(0, np.int64)
            gid = np.concatenate([b for _, b in F['member'][name]]) if F['member'][name] else np.empty(0, np.int64)
            img = J[key].copy().ravel()
            first = np.unique(lin, return_index=True)[1]
            free = img[lin[first]] < 0
            img[lin[first][free]] = gid[first][free]
            out[key] = img.reshape(J[key].shape)
            pairs[name] = np.concatenate([J['window_pixels'][name], np.column_stack([lin, gid]).astype(np.int64)])
            out[key + '_multi_pixels'] = int(J[key + '_multi_pixels'] + len(lin) - int(free.sum()))
        out['window_pixels'] = pairs
    return out, st


def _point_segment_distance(P, A, B, chunk=2000):
    """min over segments AB of the distance of each point of P (N,3); segments (M,3) each."""
    out = np.full(len(P), np.inf)
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-300)
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk, None, :]
        t = np.clip(((p - A[None]) * AB[None]).sum(2) / L2[None], 0., 1.)
        q = A[None] + t[..., None] * AB[None]
        out[i:i + chunk] = np.linalg.norm(p - q, axis=2).min(1)
    return out


def plane_and_generator_checks(source, fault, samples):
    """None where the package has no such segment."""
    source = Path(source)
    P = np.concatenate([r['point'] for r in samples.values() if r['fault'] == fault['id']] or [np.empty((0, 3))])
    out = dict(samples=int(len(P)))
    if len(P):
        dp = np.abs((P - fault['point']) @ fault['normal'])
        out['label_to_plane_m_p99_max'] = [float(np.quantile(dp, .99)), float(dp.max())]
    tj, tn = source / 'package' / 'traces.json', source / 'package' / 'traces.npz'
    if not (tj.is_file() and tn.is_file()) or not len(P):
        out['generator'] = None
        return out
    ids = json.loads(tj.read_text(encoding='utf8'))['interface_ids']
    with np.load(tn) as a:
        seg = np.asarray(a['segments'], float)
    sel = np.isin(np.asarray(ids), fault['interface_ids'])
    if not sel.any():
        out['generator'] = dict(segments=0)
        return out
    S = seg[sel]
    d1 = _point_segment_distance(P, S[:, 0], S[:, 1])
    ours = [(r['point'][:-1], r['point'][1:]) for r in samples.values() if r['fault'] == fault['id'] and len(r['point']) > 1]
    A = np.concatenate([a for a, _ in ours]); B = np.concatenate([b for _, b in ours])
    E = np.unique(S.reshape(-1, 3), axis=0)
    d2 = _point_segment_distance(E, A, B)
    out['generator'] = dict(segments=int(sel.sum()), generator_length_m=float(np.linalg.norm(S[:, 1] - S[:, 0], axis=1).sum()),
                            samples_to_generator_m_p99_max=[float(np.quantile(d1, .99)), float(d1.max())],
                            generator_to_samples_m_p99_max=[float(np.quantile(d2, .99)), float(d2.max())])
    return out
