"""Surface processes of the sandbox, shared by every exposure.

Input: an ExposureSurface (patches of reference surface) and the v2 truth of the tile. Output: the final display mesh
with exact labels. Geological inputs all come from v2: lithology retreat (ramped across the exact contacts), undercuts
on the weaker side of contacts, scars, the fault damage zone and joint grooves centred on the v2 planes at the solved
surface. The method is the seminar-demo morphology (facet relief, re-solved recession), generalised from one ray grid
to any set of patches.

Continuity rule (interface contract): every offset term must be continuous across geological contacts; a step aliases
into a stair-stepped crack along the sampling grid.

Outputs keep the three kinds of information apart: geological truth (v2 face keys of the rock surface, also under thin
cover), visible semantics (bedrock / soil / item classes) and surface state (applied recession, weathering) as
per-vertex fields."""
import math

import numpy as np

from .display import fbm, refine_contacts, segment_distance, smooth, value_noise, vertex_normals
from .lithology import get as lith_params
from .morph import compact, facet_noise, facet_noise_c1, noise, noise_c2, pnorm_max, soft_clip
from .truth import _inside_triangles, _plane_groups, inside_triangles_grid
from .surface import positions, weld

PROFILES = {
    # natural: fully weathered exposure (bluff, escarpment, canyon walls)
    # cover_ramp: cover per unit of (normal z - cos threshold); 6 lets flat ground stay partly bare
    # (the bluff has its own ground item), sheets with ground in the patch use a ~5 deg transition.
    'natural': dict(retreat=1., undercut=1., scars=True, fault_recess=1., joint_depth=1., fade_on_cover=False, cover_ramp=6.,
                    relief=1., edge=1.),
    # fresh: excavated face (roadcut, quarry): little recession, no undercut or scars, shallow
    # fracture-controlled grooves, sharp contact edges, subdued facet relief.
    'fresh': dict(retreat=.12, undercut=0., scars=False, fault_recess=.3, joint_depth=.45, fade_on_cover=True, cover_ramp=25.,
                  relief=.55, edge=.2),
}

# Maximum spalling depth of fracture_blocks per process (kept out of PROFILES, which is recorded in
# the provenance of every scene: scenes without 'fracture' stay byte-identical).
FRACTURE_DEPTH = {'natural': .3, 'fresh': .22}


# A groove centre may sit at most this fraction of its half width off its v2 plane (labels draw
# the plane's trace; the visible groove bottom must coincide with it).
GROOVE_OFFSET_TOLERANCE = .25


def groove_alignment(V, parts, truth, owners, widths, min_vertices=10):
    """Offset of every fault / joint / intrusive-contact groove floor from its v2 plane on the
    final surface (the line label draws that plane's trace).

    ``parts[cls] = (depth, amplitude)`` per vertex. Floor vertices are those dug to >= 90 % of
    their local amplitude (amplitude >= 2 cm); for each plane piece (one groove) the mean signed
    distance of its floor vertices is the groove's centre offset. Returns per class:

    - ``centre_offset_m``: median / p95 / max over GROOVES of |centre offset| -- one value per
      groove, not a pointwise error; ``ok`` compares its p95 with GROOVE_OFFSET_TOLERANCE x half width;
    - ``floor_abs_distance_m``: distribution over floor VERTICES of |distance to the plane| (report
      only: it includes the normal width of the groove floor band, so the centre tolerance does not
      apply to it)."""
    from .truth import _inside_triangles, _plane_groups
    out = {}
    for cls, own in owners.items():
        (g, amp), width = parts[cls], widths[cls]
        floor = np.nonzero((amp >= .02) & (g >= .9 * amp))[0]
        # Each floor vertex belongs to its nearest plane piece only (parallel joints can be closer
        # than the search window).
        best, piece, signed = np.full(len(floor), np.inf), np.full(len(floor), -1), np.zeros(len(floor))
        k = 0
        for iid, iv, if_ in truth.interfaces:
            if iid.split(':', 1)[0] not in own or not len(if_) or not len(floor):
                continue
            for normal, offset, tri in _plane_groups(iv, if_):
                k += 1
                d = V[floor] @ normal - offset
                idx = np.nonzero((np.abs(d) < 3 * width) & (np.abs(d) < best))[0]
                idx = idx[_inside_triangles(V[floor[idx]] - d[idx, None] * normal, tri, normal, eps=1e-4)]
                best[idx], piece[idx], signed[idx] = np.abs(d[idx]), k, d[idx]
        measured = [j for j in np.unique(piece[piece >= 0]) if (piece == j).sum() >= min_vertices]
        a = np.asarray([float(abs(signed[piece == j].mean())) for j in measured])
        pts = np.abs(signed[np.isin(piece, measured)]) if measured else np.zeros(0)
        out[cls] = {'grooves': len(a), 'tolerance_m': GROOVE_OFFSET_TOLERANCE * width,
                    'centre_offset_m': ({'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)), 'max': float(a.max())}
                                        if len(a) else {}),
                    'floor_vertices': int(len(pts)),
                    'floor_abs_distance_m': ({q: float(np.percentile(pts, p)) for q, p in (('median', 50), ('p90', 90), ('p95', 95), ('p99', 99))}
                                             | {'max': float(pts.max())} if len(pts) else {}),
                    # the "unmeasured" rule (no groove measured on >= 1 m of grooved trace) is applied by erode()
                    'ok': bool(not len(a) or np.percentile(a, 95) <= GROOVE_OFFSET_TOLERANCE * width)}
    return out


def bed_tint(choices, rng, mode='rgb'):
    """Weathered colour of one bed: a base tint of its lithology, brightness x0.7-1.25 and a jitter.
    ``rgb`` (original): independent per-channel jitter, which can push brown/grey beds to green or
    purple hues. ``luma``: the same random draws (stream unchanged) applied as a luminance shift
    only, so the hue and saturation of the base tint are kept."""
    c = np.array(choices[rng.integers(len(choices))]) * rng.uniform(.7, 1.25)
    jitter = rng.normal(0, .045, 3)
    if mode == 'luma':
        lum = float(c @ (.2126, .7152, .0722))
        c = c * (1 + float(jitter @ (.2126, .7152, .0722)) / max(lum, .05))
    elif mode == 'rgb':
        c = c + jitter
    else:
        raise ValueError('tint_jitter must be rgb or luma')
    return np.clip(c, .05, .95)


def fracture_blocks(W, start, ref, body, truth, joint_ids, seed, *, depth, spacing, step=.01, version=1):
    """Block spalling bounded by geological planes (sampling ``fracture: blocks-1``).

    The rock behind the surface is cut into cells by bedding partings (levels of the v2 bedding coordinate, metres along
    the bedding normal, every ``spacing`` m) and the v2 joint planes (every plane of each joint set, as infinite
    planes: v2 joints run through the tile and are only offset by faults). Each cell is spalled to a random depth R in
    [0, depth]. Marching from the surface along each patch's offset direction, the new surface is the first depth s
    with s >= R(cell at s): risers then lie exactly in the bedding / joint planes and meet the cell floors at sharp
    edges, and the offset stays continuous (the continuity rule) wherever a plane is not parallel to the offset
    direction. The cell is a function of 3D position only (bedding coordinate extrapolated linearly from the reference
    surface, exact for planar beds), so neighbouring vertices agree. Partings carry no label (they are not v2
    interfaces). CAUTION: every joint piece is extended as an infinite plane; v2 joints are offset by faults, so each
    fault block also gets the other blocks' copies of its joints (2-27 cm away) -- joint-like steps with no v2 joint
    and no label. blocks-1/2 are kept for reproducibility of the recorded samples only; fracture.py (blocks-3) bounds
    cells by the joint pieces of each fault block.

    ``start``: offset (along the patch direction) of the surface to cut; returns the extra offset."""
    p0 = positions(W, start)
    dirv = positions(W, start + 1.) - p0
    ok, nb, tb = truth.fabric(ref, body, 'bedding')
    t0 = np.where(ok, tb + (nb * (p0 - ref)).sum(1), np.nan)
    planes = []
    for set_id in sorted(joint_ids):
        normal, offs = None, []
        for iid, iv, if_ in truth.interfaces:
            if iid.split(':', 1)[0] != set_id or not len(if_):
                continue
            for nrm, off, _ in _plane_groups(iv, if_):
                if normal is None:
                    normal = nrm
                sgn = 1. if nrm @ normal >= 0 else -1.
                offs.append(sgn * off)
        if normal is not None:
            if version >= 2 and any(abs(normal @ n2) > math.cos(math.radians(20)) for n2, _ in planes):
                continue
            planes.append((normal, np.unique(np.round(offs, 4))))
    phase = (seed % 9973) / 9973. * spacing
    levels = None
    if version >= 2:
        lo, hi = np.nanmin(t0) - depth - 1., np.nanmax(t0) + depth + 1.
        g = np.random.default_rng(seed + 307)
        th = spacing * g.uniform(.4, 1.6, int((hi - lo) / (.4 * spacing)) + 2)
        levels = lo + phase + np.concatenate([[0.], np.cumsum(th)])
    extra = np.zeros(len(p0))
    todo = np.nonzero(ok)[0]
    for k in range(int(math.ceil(depth / step)) + 1):
        if not len(todo):
            break
        s = k * step
        q = p0[todo] + s * dirv[todo]
        tq = t0[todo] + s * (nb[todo] * dirv[todo]).sum(1)
        if levels is None:
            cell = np.floor((tq - phase) / spacing).astype(np.int64)
        else:
            cell = np.searchsorted(levels, tq).astype(np.int64)
        M = 2147483647  # keep the hash in int64 range at every step
        cell = cell % M
        for j, (normal, offs) in enumerate(planes):
            cell = (cell * 48271 + np.searchsorted(offs, q @ normal) + 7919 * (j + 1)) % M
        h = ((cell * 16807 + seed % M) % M * 48271 % M) / M
        R = depth * h ** 1.5
        done = s >= R
        extra[todo[done]] = s
        todo = todo[~done]
    if version >= 2:
        F = W['F']
        jump = np.zeros(len(extra))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            dd = np.abs(extra[F[:, a]] - extra[F[:, b]])
            np.maximum.at(jump, F[:, a], dd)
            np.maximum.at(jump, F[:, b], dd)
        band = smooth((jump > .03).astype(float), F, len(extra), 1) > 0
        extra = np.where(band, smooth(extra, F, len(extra), 2), extra)
    return extra


def _softmin_segment_distance(P, seg, sigma, chunk=20000):
    """Soft minimum (sigma) of the distances from each point to every segment: C1 inside a polygon where the nearest segment
    switches (medial axis), at most sigma x log(n) below the true minimum."""
    a, b = seg[:, 0], seg[:, 1]
    ab = b - a
    den = np.maximum((ab * ab).sum(1), 1e-18)
    out = np.empty(len(P))
    for s0 in range(0, len(P), chunk):
        q = P[s0:s0 + chunk]
        t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(2) / den[None], 0, 1)
        d = np.linalg.norm(q[:, None, :] - (a[None] + t[:, :, None] * ab[None]), axis=2)
        dm = d.min(1)
        out[s0:s0 + chunk] = dm - sigma * np.log(np.exp(-(d - dm[:, None]) / sigma).sum(1))
    return np.maximum(out, 0.)


def _shape_check(opts, F, V, carry, exact_v, g_extra, evaluate, tol_of, faces=None):
    """Sets: 'random_<seed>' (per_face random barycentric points per face, fixed seed) and 'deterministic' (centroid + the
    three rotations of (2/3, 1/6, 1/6)). err = |exact - linear| (m), tol = imaging tolerance (m); each set passes iff
    p99(err/tol) <= 1 and max(err/tol) <= 2 (sampled maximum, not a proof over the surface). Returns (summary,
    exceedances): exceedances are the samples with err/tol > 1 per set (face, barycentric, reference position, exact,
    linear, err, tol, per-term error)."""
    seeds = [int(x) for x in opts.get('random_seeds', [0, 1, 2])]
    per_face = int(opts.get('per_face', 3))
    chunk = int(opts.get('chunk', 300000))
    sets = []
    fids = np.arange(len(F)) if faces is None else np.asarray(faces)
    for sd in seeds:
        g = np.random.default_rng(sd)
        fi = np.repeat(fids, per_face)
        r = g.random((len(fi), 2))
        fl = r.sum(1) > 1
        r[fl] = 1 - r[fl]
        sets.append(('random_%d' % sd, fi, np.c_[1 - r.sum(1), r]))
    if opts.get('deterministic', True):
        pts = np.array([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        sets.append(('deterministic', np.repeat(fids, 4), np.tile(pts, (len(fids), 1))))
    terms = ('broad', 'meso', 'facet', 'relief', 'fault', 'joint', 'contact')
    summary, exceed = {}, {}
    for name, fi, bc in sets:
        err, tol = np.zeros(len(fi)), np.zeros(len(fi))
        gprof = np.zeros(len(fi))
        keep = []
        for a0 in range(0, len(fi), chunk):
            sl = slice(a0, a0 + chunk)
            ff, bb = F[fi[sl]], bc[sl]
            cs = {k: (v[ff] * (bb[:, :, None] if v.ndim == 2 else bb)).sum(1) for k, v in carry.items()}
            ex = evaluate(cs)
            ge = (g_extra[ff] * bb).sum(1)
            Ve = cs['b'] - ex['outward'] * (ex['total'] * ge)[:, None]
            Vl = (V[ff] * bb[:, :, None]).sum(1)
            err[sl] = np.linalg.norm(Ve - Vl, axis=1)
            tol[sl] = tol_of(Vl)
            gprof[sl] = np.maximum(ex['groove_profile'], ex['fault_zone'])
            bad = np.flatnonzero(err[sl] > tol[sl])
            if len(bad):
                rec = {'face': fi[sl][bad], 'bary': bb[bad], 'b': cs['b'][bad], 'exact': Ve[bad], 'linear': Vl[bad],
                       'err': err[sl][bad], 'tol': tol[sl][bad]}
                # per-term error along the offset (m): exact term - linear interpolation of the term's vertex values
                for t in terms:
                    tv = exact_v['term_' + t]
                    rec['term_' + t] = ex['term_' + t][bad] - (tv[ff[bad]] * bb[bad]).sum(1)
                keep.append(rec)
        ratio = err / tol
        ok = bool(np.percentile(ratio, 99) <= 1. and ratio.max() <= 2.)
        summary[name] = {'samples': int(len(fi)), 'p50_p99_max_ratio': [float(np.percentile(ratio, 50)), float(np.percentile(ratio, 99)), float(ratio.max())],
                         'err_mm_p99_max': [float(np.percentile(err, 99) * 1e3), float(err.max() * 1e3)], 'over_1': int((ratio > 1).sum()),
                         'over_2': int((ratio > 2).sum()), 'pass': ok}
        band = gprof > .1
        if band.any():
            # groove / fault-zone band (profile > 0.1): reported to compare structure-band refinement rules
            summary[name]['groove_band'] = {'samples': int(band.sum()), 'p99_max_ratio': [float(np.percentile(ratio[band], 99)), float(ratio[band].max())]}
        if keep:
            exceed[name] = {k: np.concatenate([r_[k] for r_ in keep]) for k in keep[0]}
    summary['pass'] = all(v['pass'] for k, v in summary.items() if isinstance(v, dict))
    summary['rule'] = ('every set: p99(err/tol) <= 1 and max(err/tol) <= 2; err = |exact - linear| (m) on the refined cm surface before B; '
                       'tol = clamp(0.5 x pixel footprint of the nearest formal camera containing the point, tol_min, tol_max) (m); '
                       'sampled maximum, not a proof over the surface')
    return summary, exceed


def erode(surface, model, truth, rng, *, clay=False, debug=None, tint_jitter='rgb', fracture=None, blockface=None, cm=None,
          self_affine=None):
    """cm: coarse sheet + adaptive refinement. Mesh smoothing iterations are converted to the physical scale of the
    historical 7 cm sheet (x (0.07 / cm['sheet_edge_m'])^2); after the fold guard the surface is refined (structure
    bands, shape error vs the imaging tolerance) with the offsets re-evaluated exactly at new points; then the
    whole-wall B (blockface mode wall, no scan detail) is built on the refined surface."""
    profile = {**PROFILES[surface.profile.get('process', 'natural')], **surface.profile}
    _t0 = __import__('time').time()
    _marks = []
    _mark = lambda name: _marks.append((name, __import__('time').time() - _t0))
    if cm is not None:
        from .cmgeom import smoothing_iterations
        if fracture is not None:
            raise ValueError('cm route: fracture options are not supported (B is the block construction)')
        _it = lambda n_: smoothing_iterations(n_, cm['sheet_edge_m'])
    else:
        _it = lambda n_: n_
    smooth_f = bool(cm is not None and cm.get('smooth_fields'))
    _facet = facet_noise_c1 if smooth_f else facet_noise
    _clip01 = (lambda x_: soft_clip(x_)) if smooth_f else (lambda x_: np.clip(x_, 0, 1))
    _union = pnorm_max if smooth_f else np.maximum
    seed = int(rng.integers(1, 2 ** 30))
    W = weld(surface, truth)
    V0, F = W['V'], W['F']
    w = W['weights']
    ground_z = surface.ground_z_m
    zero = np.zeros(len(V0))

    # 1. reference surface: broad roughness + multi-scale facets on the patch reference (seminar macro).
    base = positions(W, zero)
    x, y, z = base.T
    broad = float(rng.uniform(.22, .4))
    facets = [[3.8, .75], [1.7, .28], [.72, .10], [.28, .032]]
    fscale = float(rng.uniform(.75, 1.2))
    rough = broad * noise(x / 4. + y / 7., z / 3., seed + 12)
    st0, body0, _, lith0 = truth.query(base)
    meso = np.array([1.4 * lith_params(l)['rough'] if s == 'occupied' else .015 for s, l in zip(st0, lith0)])
    # Amplitude per lithology jumps at contacts; smooth it over the mesh (~0.3 m).
    meso = smooth(meso, F, len(meso), _it(15))
    rough += meso * (w[:, 0] * noise(y / .3, z / .2, seed + 15) + w[:, 1] * noise(x / .3, z / .2, seed + 16)
                     + w[:, 2] * noise(x / .3, y / .2, seed + 17))
    # amplitude amp_of_meso x the lithology roughness field; finer detail is left to the material normal map
    rc1 = (cm or {}).get('relief_c1')
    if rc1 and (cm or {}).get('relief'):
        raise ValueError('cm.relief and cm.relief_c1 are exclusive')
    rc1_oct = [([float(v_) for v_ in o_['scale_m']], float(o_['amp_of_meso'])) for o_ in (rc1 or {}).get('octaves', [])] or         [([float(v_) for v_ in (rc1 or {}).get('scale_m', [.11, .08])], float((rc1 or {}).get('amp_of_meso', .35)))]

    def relief_c1_of(x_, y_, z_, w_, meso_):
        out_ = 0.
        for k_, (sc_, am_) in enumerate(rc1_oct):
            sd_ = seed + 181 + 7 * k_
            nz_ = noise_c2 if (rc1 or {}).get('c2', False) else noise
            out_ = out_ + am_ * meso_ * (w_[:, 0] * nz_(y_ / sc_[0], z_ / sc_[1], sd_) + w_[:, 1] * nz_(x_ / sc_[0], z_ / sc_[1], sd_ + 1)
                                         + w_[:, 2] * nz_(x_ / sc_[0], y_ / sc_[1], sd_ + 2))
        return out_
    if rc1:
        rough += relief_c1_of(x, y, z, w, meso)
    # keep_facets are replaced by the scan-calibrated self-affine relief (morph.self_affine_relief) at the reference
    # points; kept octaves use facet_noise_c1 when facets_c1. Every rng draw is unchanged; the relief seed derives from seed.
    sa = None
    if self_affine is not None:
        if cm is not None:
            raise ValueError('self_affine relief is not supported on the cm route')
        import hashlib
        import json
        from pathlib import Path
        from .morph import self_affine_relief
        _sa_bytes = Path(self_affine).read_bytes()
        sa = json.loads(_sa_bytes.decode('utf8'))
        sa.update(sha256=hashlib.sha256(_sa_bytes).hexdigest(), seed=seed + 1301, path=str(self_affine))
    for i, (scale, amp) in enumerate(facets):
        if sa is not None and i not in sa['keep_facets']:
            continue
        _fi = facet_noise_c1 if sa is not None and sa.get('facets_c1') else _facet
        a = np.deg2rad(23 + i * 47)
        cs, sn = math.cos(a), math.sin(a)
        for axis, (v, u) in enumerate(((y, z), (x, z), (x, y))):
            warp = .3 * noise(v / (2 * scale), u / (2 * scale), seed + 105 + i)
            rough += w[:, axis] * amp * fscale * _fi((v * cs + u * sn) / scale + warp,
                                                          (u * cs - v * sn) / scale - warp, seed + 125 + i + axis)
    if sa is not None:
        rough += self_affine_relief(base, sa['octaves'], sa['seed'])
    tile_ramp = None
    if surface.camera.get('rock_xy') is not None:
        # Rock only inside the tile: soil takes over over a few metres towards the tile edge. The
        # same ramp fades the rock relief, so the relief ends under the soil, not at the domain edge.
        (x0, y0), (x1, y1) = surface.camera['rock_xy']
        ramp_scale = float(rng.uniform(2, 5))

        def tile_ramp(P):
            dxy = np.minimum.reduce([P[:, 0] - x0, x1 - P[:, 0], P[:, 1] - y0, y1 - P[:, 1]])
            return np.clip(1 - dxy / (ramp_scale * (.6 + .8 * value_noise(P * .5, seed + 18))), 0, 1)
    rough = rough * profile['relief']
    # Excavated surfaces stay bare rock at any slope (bench treads); ~0.7 m transition.
    bare_w = smooth(W['bare'].astype(float), F, len(V0), _it(10)) if W['bare'].any() else np.zeros(len(V0))
    fade = np.ones(len(V0))
    if profile['fade_on_cover']:
        # Gentle slopes are soil-covered: rock relief fades out there (cover estimated on the base).
        nb = vertex_normals(base, F, np.array([0., 0., 1.]), consistent=True)
        thr = math.cos(math.radians(float(rng.uniform(28, 46))))
        c0 = np.clip((nb[:, 2] - thr) * 6 + .5, 0, 1) * (1 - bare_w)
        c0 = np.maximum(c0, (st0 != 'occupied').astype(float))
        if tile_ramp is not None:
            c0 = np.maximum(c0, tile_ramp(base))
        if W['hold'].any():
            c0 = np.maximum(c0, W['hold'].astype(float))
        fade = (1 - smooth(c0, F, len(c0), _it(10))) ** 2
        fade[W['hold']] = 0.  # rock under road / lining items stays exactly in place
        rough = rough * fade
    ref = positions(W, -rough)

    # 2. geology at the reference surface (v2): lithology retreat ramped across exact contacts.
    status, body, unit, lith = truth.query(ref)
    occ = status == 'occupied'
    retreat_scale = float(rng.uniform(.75, 1.25)) * profile['retreat']

    def retreat_of(names, sts):
        return np.array([max(.06, .38 + .9 * lith_params(l)['recess']) * retreat_scale if s == 'occupied' else .3 * profile['retreat']
                         for s, l in zip(sts, names)])
    r_own = retreat_of(lith, status)
    from .truth import contact_segments
    cs = contact_segments(ref, F, truth, known=(status, body, unit, lith))
    d_contact = np.full(len(ref), np.inf)
    r_other = r_own.copy()
    if len(cs):
        d_contact, k = segment_distance(ref, cs)
        a, b = cs[k, 0], cs[k, 1]
        ab = b - a
        t = np.clip(((ref - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-18), 0, 1)
        foot = a + t[:, None] * ab
        near = d_contact < 1.5
        if near.any():
            away = foot[near] - ref[near]
            away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
            s2, _, _, l2 = truth.query(foot[near] + .04 * away)
            r_other[near] = retreat_of(l2, s2)
    ramp = .4
    mid = .5 * (r_own + r_other)
    tt = np.clip(d_contact / ramp, 0, 1)
    retreat = mid + (r_own - mid) * tt * tt * (3 - 2 * tt)
    packet = .5 + .5 * noise(ref[:, 2] / 2.8 + ref[:, 0] / 9., ref[:, 0] / 6. + ref[:, 1] / 7., seed + 61)
    exposure = np.clip((noise(ref[:, 0] / 4.2 + ref[:, 1] / 11., ref[:, 2] / 2.7, seed + 5) + .05) / .7, 0, 1)
    exposure = .06 + .94 * exposure * exposure * (3 - 2 * exposure)
    recess = retreat * (.25 + .75 * packet) * (.35 + .65 * exposure)
    recess_base = recess.copy()
    # Undercut on the weaker side only, zero on the contact; where the nearest contact switches the
    # raw term jumps, mesh smoothing (~0.3 m) turns that into a ramp.
    weaker = r_own > r_other + .05
    basal = float(rng.uniform(.35, .7)) * profile['undercut']
    on = np.clip(d_contact / .15, 0, 1)
    raw = basal * weaker * on * on * (3 - 2 * on) * np.exp(-((d_contact - .42) / .6) ** 2) \
        * np.clip(noise(ref[:, 0] / 3.5 + ref[:, 1] / 9., ref[:, 2] / 4., seed + 66) + .35, 0, 1)
    recess += smooth(raw, F, len(raw), _it(25))
    edge = .26 * profile['edge'] * compact(d_contact / .42) * (.6 + .4 * noise(ref[:, 0] / 1.2, ref[:, 2] / 1.5, seed + 7))
    recess += edge * np.clip((noise(ref[:, 0] / 5. + ref[:, 1] / 8., ref[:, 2] / 3., seed + 70) + .05) * 1.8, 0, 1)
    # Scars: open undercuts below a few contacts on the weaker side.
    cand = np.nonzero(weaker & (d_contact < .4) & occ & (ref[:, 2] > ground_z + .5))[0]
    scar = np.zeros(len(ref))
    picks = ref[rng.choice(cand, min(len(cand), int(rng.integers(2, 6))), replace=False)] if len(cand) else []
    for c in (picks if profile['scars'] else []):
        wd, ht, dp = float(rng.uniform(2.5, 4.2)), float(rng.uniform(1.2, 2.1)), float(rng.uniform(.6, 1.1))
        horiz = np.linalg.norm((ref - c)[:, :2], axis=1) / wd
        below = c[2] - ref[:, 2]
        roof = .5 + .5 * np.tanh((below - .15) / .25)
        scar += dp * roof * np.exp(-np.maximum(below, 0) / ht) * np.exp(-horiz ** 4)
    recess += scar
    recess *= fade
    frac = np.zeros(len(ref))
    fracture_info = None
    exact = None
    if fracture in ('blocks-1', 'blocks-2', 'blocks-3'):
        # Opt-in (sampling 'fracture'): spalled blocks bounded by bedding partings and the v2 joint
        # planes. Own random stream: the main rng is not consumed, everything else stays the same.
        frng = np.random.default_rng(seed + 211)
        fdepth_max = FRACTURE_DEPTH[surface.profile.get('process', 'natural')] * float(frng.uniform(.8, 1.2))
        parting = float(frng.uniform(.18, .45))
        joint_sets = {f['id'] for f in model.get('features', []) if f['kind'] in ('joint_set', 'joint')}
        if fracture == 'blocks-3':
            # Same cells as blocks-2, exact pieces; risers and their edges are reconstructed on the
            # final mesh below (fracture.insert_edges), crease normals at the end.
            from .fracture import Fracture
            fx = Fracture(ref, body, truth, joint_sets, seed, depth=fdepth_max, spacing=parting)
            # Cut and displace along the reference-surface vertex normal, not the patch offset
            # direction: roadcut offsets are horizontal while the face leans 20-30 deg, and risers
            # cut along them overhang the face (the fold check flags them). The cut is evaluated on
            # the final unspalled surface (after grooves and the fold guard, below), so risers lie
            # exactly in their planes and the spalled surface is a height field over that surface;
            # the fold guard never halves the spalling.
            n_out = vertex_normals(base, F, np.array([0., 0., 1.]), consistent=True)
            if (n_out * W['outward']).sum() < 0:
                n_out = -n_out
            # No spalling where the reference itself bends (bench edges, crest roll-over): there the
            # vertex normal leans off its faces' normals and a deep riser shears thin triangles.
            fnb = np.cross(base[F[:, 1]] - base[F[:, 0]], base[F[:, 2]] - base[F[:, 0]])
            fnb /= np.maximum(np.linalg.norm(fnb, axis=1, keepdims=True), 1e-20)
            fnb *= np.sign((fnb * n_out[F].mean(1)).sum(1))[:, None]
            lean = np.zeros(len(base))
            for k in range(3):
                np.maximum.at(lean, F[:, k], np.degrees(np.arccos(np.clip((fnb * n_out[F[:, k]]).sum(1), -1, 1))))
            straight = np.clip((4. - lean) / 2., 0, 1)
            straight = smooth(straight * straight * (3 - 2 * straight), F, len(base), 2)
            exact = dict(fx=fx, m=-n_out, weight=fade * straight, bend_free=float((straight > .99).mean()))
        else:
            frac = fracture_blocks(W, -rough + recess, ref, body, truth, joint_sets, seed, depth=fdepth_max, spacing=parting,
                                   version=int(fracture[-1])) * fade
            recess = recess + frac
        fracture_info = {'method': fracture, 'max_depth_m': fdepth_max, 'mean_parting_m': parting,
                         'note': 'surface spalling proxy: in-bed parting steps are not modelled laminae and carry no '
                                 'line label; stratigraphic contacts and joints are labelled from v2 only'}
    elif fracture is not None:
        raise ValueError('fracture must be None, blocks-1, blocks-2 or blocks-3')

    # 3. faults and joints from the v2 planes, centred on the plane at the solved surface.
    kinds = {f['id']: f['kind'] for f in model.get('features', [])}
    faults = {e['id'] for e in model.get('events', []) if e['kind'] == 'fault'}
    joints = {k for k, v in kinds.items() if v in ('joint_set', 'joint')}
    intrusions = {e['id'] for e in model.get('events', []) if e['kind'] == 'intrusion'}
    fault_recess = float(rng.uniform(.9, 1.6)) * profile['fault_recess']
    damage = float(rng.uniform(.4, .8))
    jw, jd = float(rng.uniform(.06, .14)), float(rng.uniform(.15, .35)) * profile['joint_depth']
    # Fault trenches belong on walls: on near-horizontal tops they stretch into stripes.
    wall = 1. - .8 * w[:, 2]

    # Grooves are dug like every other offset (along the patch's offset direction), so a groove
    # whose plane meets the surface obliquely would shift sideways by depth * cos. Each profile is
    # therefore centred where the floor lands back on the v2 plane after the full-depth dig: the
    # line label (plane x final surface) runs along the groove floor. Distances are measured on the
    # surface without grooves; the shift uses the actual displacement of the patch (exact for
    # re-solved patches too, not a linear estimate).
    pre = -rough + recess
    relief_cm = None
    if cm is not None and cm.get('relief'):
        # scan in the 5-30 cm band), planar mapping on the vertical plane normal to the exposure view direction, weight
        # fade x bare_w, displacement along the sheet offset (outward bump = negative recession); exact at any point
        _rz = np.load(cm['relief'])
        _rh, _rt = np.asarray(_rz['height_m'], float), float(_rz['texel_m'])
        _vd = np.asarray(surface.camera.get('view_direction') or [0., -1., 0.], float)
        _eh = np.cross([0., 0., 1.], _vd)
        _eh /= np.linalg.norm(_eh)
        _rn = _rh.shape[0]

        def relief_cm(P):
            u_ = (P @ _eh) / _rt
            v_ = P[:, 2] / _rt
            i0, j0 = np.floor(v_).astype(np.int64), np.floor(u_).astype(np.int64)
            fy, fx = v_ - i0, u_ - j0
            i0, j0 = i0 % _rn, j0 % _rn
            i1, j1 = (i0 + 1) % _rn, (j0 + 1) % _rn
            return ((1 - fy) * ((1 - fx) * _rh[i0, j0] + fx * _rh[i0, j1]) + fy * ((1 - fx) * _rh[i1, j0] + fx * _rh[i1, j1]))
        pre = pre - relief_cm(base) * fade * bare_w
    V_ng = positions(W, pre)
    x, y, z = V_ng.T

    _pgroups, _pbound = {}, {}
    groove_end_taper = float(cm.get('groove_end_taper_m', .2)) if cm is not None else None

    def grooves(owners, base, surface, amp, width, reach, varied=False, pos=None):
        """Groove depth = max over every owned plane piece k of a_k exp(-((d_k + s_k) / w_k)^2):
        continuous where pieces meet (no nearest-piece switch). s_k moves the profile so the floor
        (d_k + s_k = 0) lands back on the plane after the full-depth dig; the dig of the reference
        amplitude is solved once by the patch and scaled to each piece's amplitude. ``varied``:
        per-piece depth and width (joints). Returns depth, winning amplitude, profile max."""
        moved = (positions(W, base + amp) if pos is None else pos(base + amp)) - surface
        depth, win, prof = np.zeros(len(surface)), np.zeros(len(surface)), np.zeros(len(surface))
        for k, (iid, iv, if_) in enumerate(truth.interfaces):
            if iid.split(':', 1)[0] not in owners or not len(if_):
                continue
            per = .5 + .5 * math.sin(k * 12.9898 + seed) if varied else 1.
            fa, fw = (.4 + .6 * per, .7 + .6 * per) if varied else (1., 1.)
            if iid not in _pgroups:
                _pgroups[iid] = _plane_groups(iv, if_)
            for gi_, (normal, offset, tri) in enumerate(_pgroups[iid]):
                d = surface @ normal - offset
                idx = np.nonzero(np.abs(d) < reach)[0]
                if not len(idx):
                    continue
                proj = surface[idx] - d[idx, None] * normal
                ins_ = (inside_triangles_grid if cm is not None else _inside_triangles)(proj, tri, normal, eps=1e-4)
                idx, proj = idx[ins_], proj[ins_]
                c = d[idx] + fa * (moved[idx] @ normal)
                env = np.exp(-(c / (width * fw)) ** 2)
                penv = np.exp(-(c / width) ** 2)
                if groove_end_taper is not None and len(idx):
                    # cm route: the groove fades out towards the end of its finite plane piece (smoothstep of the in-plane
                    # distance to the piece boundary over groove_end_taper m), so the offset field stays continuous there
                    key_ = (iid, gi_)
                    if key_ not in _pbound:
                        E_ = np.concatenate([np.stack([tri[:, 0], tri[:, 1]], 1), np.stack([tri[:, 1], tri[:, 2]], 1), np.stack([tri[:, 2], tri[:, 0]], 1)])
                        # boundary of the piece = triangle edges used once (corners matched at 1e-9 m)
                        Ek = np.round(np.concatenate([E_[:, 0], E_[:, 1]], 1), 9)
                        key_e = [tuple(sorted((tuple(r_[:3]), tuple(r_[3:])))) for r_ in Ek.tolist()]
                        cnt_ = {}
                        for kk in key_e:
                            cnt_[kk] = cnt_.get(kk, 0) + 1
                        bnd = np.array([[kk[0], kk[1]] for kk, cc in cnt_.items() if cc == 1], float).reshape(-1, 2, 3)
                        _pbound[key_] = bnd
                    bnd = _pbound[key_]
                    if len(bnd):
                        if smooth_f:
                            # soft minimum only where it can fall below the taper length: it is at most sigma ln(n)
                            # below the hard minimum, so elsewhere the taper is 1 either way (same result, less work)
                            db_ = np.asarray(segment_distance(proj, bnd)[0], float)
                            near_ = np.flatnonzero(db_ < groove_end_taper + .03 * np.log(max(len(bnd), 2)) + 1e-3)
                            if len(near_):
                                db_[near_] = _softmin_segment_distance(proj[near_], bnd, .03)
                        else:
                            db_ = segment_distance(proj, bnd)[0]
                        t_ = np.clip(np.asarray(db_, float) / groove_end_taper, 0, 1)
                        t_ = t_ * t_ * (3 - 2 * t_)
                        env = env * t_
                        penv = penv * t_
                g = fa * amp[idx] * env
                better = g > depth[idx]
                win[idx[better]] = fa * amp[idx[better]]
                if smooth_f:
                    depth[idx] = pnorm_max(depth[idx], g)
                    prof[idx] = pnorm_max(prof[idx], penv)
                else:
                    depth[idx[better]] = g[better]
                    prof[idx] = np.maximum(prof[idx], penv)
        return depth, win, prof

    noise_f = .72 + .28 * noise(z / 3.5, y / 3., seed + 80)
    a_fault = fault_recess * wall * fade * noise_f
    along = _clip01(noise(x / 2. + y / 3., z / 2.5, seed + 90) + .55)
    fdepth, _, gf = grooves(faults, pre, V_ng, a_fault, damage, 3 * damage + 1.)
    # Joints and contacts are placed on the surface after the fault trench is dug (a trench can
    # cut through the fault plane, where joint pieces are offset: the visible piece decides) and
    # fade out inside the damage zone, where joint pieces end at the fault.
    with_fault = pre + fdepth
    V_f = positions(W, with_fault)
    inside_fault = _clip01(1 - gf)
    jg, a_joint, gj = grooves(joints, with_fault, V_f, jd * along * fade * inside_fault, jw, 4 * jw + .5, varied=True)
    ig, a_intr, gi = grooves(intrusions, with_fault, V_f, .5 * jd * fade * inside_fault, jw, 4 * jw + .5)
    s_fix = fdepth + _union(jg, ig)
    fault_zone = gf * noise_f
    groove_profile = _union(gj, gi)
    groove_parts = {'fault': (fdepth, a_fault), 'joint': (jg, a_joint), 'contact': (ig, a_intr)}
    # Depth a groove of each class could reach here (smallest per-piece factor for joints): a
    # visible trace where this is >= 2 cm must yield a measured groove.
    groove_potential = {'fault': a_fault, 'joint': .4 * jd * along * fade * inside_fault, 'contact': .5 * jd * fade * inside_fault}
    if debug is not None:
        debug.update(recess_base=recess_base, ref=ref, rough=rough, retreat=retreat, recess=recess.copy(), scar=scar,
                     edge=edge, d_contact=d_contact, r_own=r_own, r_other=r_other, structures=s_fix)
    total = pre + s_fix
    V = positions(W, total)
    # Fold guard for sheet patches (offsets along a fixed field, not re-solved): where the offset
    # inverts a face relative to the reference, halve that face's vertex offsets until none does.
    sheet = np.array([W['rock'][k].apply is None for k in W['owner']])
    folds = []
    flip = np.zeros(len(F), bool)
    gscale = np.ones(len(total))  # cumulative fold-guard factor per vertex (blocks-3 re-evaluates the fracture with it)
    if sheet.any():
        ref_n = np.cross(base[F[:, 1]] - base[F[:, 0]], base[F[:, 2]] - base[F[:, 0]])
        flipped = lambda V: ((np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]) * ref_n).sum(1) <= 0) & sheet[F].all(1)
        for _ in range(12):
            flip = flipped(V)
            folds.append(int(flip.sum()))
            if not flip.any():
                break
            # Halve the offsets at the inverted faces and fade the reduction out over a few rings
            # around them: halving only the flagged vertices leaves an offset step to their
            # neighbours, which inverts the next faces (it never converged at the roadcut toe).
            mark = np.zeros(len(total))
            mark[np.unique(F[flip])] = 1.
            reach = np.clip(np.maximum(4 * smooth(mark, F, len(mark), 6), mark), 0, 1)
            total *= 1 - .5 * reach
            gscale *= 1 - .5 * reach
            V = positions(W, total)
        else:
            # the 8th halving was never checked: count what is left on the surface actually used
            flip = flipped(V)
    # Faces still inverted on the final surface (acceptance fails on any; stats.acceptance).
    folds_final = int(flip.sum())
    if debug is not None and debug.get('export_fields'):
        # at this point, for an external evaluator of re-solved (bluff) patches at new ray directions. Outputs are unchanged.
        debug['fields'] = dict(base=base.copy(), outward=W['outward'].copy(), w=w.copy(), meso=meso.copy(), fade=fade.copy(),
                               recess=recess.copy(), gscale=gscale.copy(), bare_w=bare_w.copy(), hold=W['hold'].astype(float),
                               total=total.copy(), V=V.copy(), sheet=sheet.copy())
        debug['scalars'] = dict(seed=seed, broad=broad, facets=[list(map(float, f_)) for f_ in facets], fscale=fscale,
                                relief=float(profile['relief']), fade_on_cover=bool(profile['fade_on_cover']), fault_recess=fault_recess,
                                damage=damage, jw=jw, jd=jd, smooth_fields=smooth_f, relief_c1=bool(rc1), relief_cm=relief_cm is not None,
                                groove_end_taper=groove_end_taper,
                                self_affine=None if sa is None else {k_: sa[k_] for k_ in ('path', 'sha256', 'seed', 'keep_facets', 'facets_c1', 'octaves')})
        debug['sets'] = dict(faults=sorted(faults), joints=sorted(joints), intrusions=sorted(intrusions))
    cm_info = None
    shape_exceed = None
    _mark('coarse_surface')
    if cm is not None:
        from . import cmgeom
        from .wallscene import Cameras
        if not sheet.all():
            raise ValueError('cm route: every rock patch must be an offset sheet (no re-solved patch)')
        t_cm = __import__('time').time()
        on = np.linalg.norm(W['outward'], axis=1, keepdims=True)

        def _evaluate(c):
            # exact offsets at points of the coarse reference sheet (mirrors steps 1-3 above); smooth fields from c
            b_ = c['b']
            o_ = c['o'] / np.maximum(np.linalg.norm(c['o'], axis=1, keepdims=True), 1e-30)
            w_ = c['w']
            pos_ = lambda off: b_ - o_ * np.asarray(off, float)[:, None]
            x_, y_, z_ = b_.T
            r_ = broad * noise(x_ / 4. + y_ / 7., z_ / 3., seed + 12)
            t_broad_ = r_
            r_ = r_ + c['meso'] * (w_[:, 0] * noise(y_ / .3, z_ / .2, seed + 15) + w_[:, 1] * noise(x_ / .3, z_ / .2, seed + 16)
                                   + w_[:, 2] * noise(x_ / .3, y_ / .2, seed + 17))
            t_meso_ = r_ - t_broad_
            t_rc1_ = relief_c1_of(x_, y_, z_, w_, c['meso']) if rc1 else np.zeros(len(b_))
            r_ = r_ + t_rc1_
            for i_, (scale_, amp_) in enumerate(facets):
                a_ = np.deg2rad(23 + i_ * 47)
                cs_, sn_ = math.cos(a_), math.sin(a_)
                for axis_, (v_, u_) in enumerate(((y_, z_), (x_, z_), (x_, y_))):
                    warp_ = .3 * noise(v_ / (2 * scale_), u_ / (2 * scale_), seed + 105 + i_)
                    r_ = r_ + w_[:, axis_] * amp_ * fscale * _facet((v_ * cs_ + u_ * sn_) / scale_ + warp_,
                                                                         (u_ * cs_ - v_ * sn_) / scale_ - warp_, seed + 125 + i_ + axis_)
            t_facet_ = r_ - t_broad_ - t_meso_ - t_rc1_
            k_r_ = profile['relief'] * (c['fade'] if profile['fade_on_cover'] else 1.)
            r_ = r_ * profile['relief']
            if profile['fade_on_cover']:
                r_ = r_ * c['fade']
            pre_ = -r_ + c['recess']
            if relief_cm is not None:
                pre_ = pre_ - relief_cm(b_) * c['fade'] * c['bare_w']
            vng_ = pos_(pre_)
            xx_, yy_, zz_ = vng_.T
            nf_ = .72 + .28 * noise(zz_ / 3.5, yy_ / 3., seed + 80)
            af_ = fault_recess * (1. - .8 * w_[:, 2]) * c['fade'] * nf_
            al_ = _clip01(noise(xx_ / 2. + yy_ / 3., zz_ / 2.5, seed + 90) + .55)
            fd_, _, gf_ = grooves(faults, pre_, vng_, af_, damage, 3 * damage + 1., pos=pos_)
            wf_ = pre_ + fd_
            vf_ = pos_(wf_)
            inf_ = _clip01(1 - gf_)
            jg_, aj_, gj_ = grooves(joints, wf_, vf_, jd * al_ * c['fade'] * inf_, jw, 4 * jw + .5, varied=True, pos=pos_)
            ig_, ai_, gi_ = grooves(intrusions, wf_, vf_, .5 * jd * c['fade'] * inf_, jw, 4 * jw + .5, pos=pos_)
            tot_ = (pre_ + fd_ + _union(jg_, ig_)) * c['gscale']
            return {'V': pos_(tot_), 'ref': pos_(-r_), 'outward': o_, 'total': tot_, 'fault_zone': gf_ * nf_, 'groove_profile': _union(gj_, gi_),
                    'gp_fault_0': fd_, 'gp_fault_1': af_, 'gp_joint_0': jg_, 'gp_joint_1': aj_, 'gp_contact_0': ig_, 'gp_contact_1': ai_,
                    'pot_fault': af_, 'pot_joint': .4 * jd * al_ * c['fade'] * inf_, 'pot_contact': .5 * jd * c['fade'] * inf_,
                    # per-term offsets (shape-check attribution; total = (sum) x gscale, up to the max of joint / contact)
                    'term_broad': -t_broad_ * k_r_, 'term_meso': -t_meso_ * k_r_, 'term_facet': -t_facet_ * k_r_,
                    'term_relief': (-relief_cm(b_) * c['fade'] * c['bare_w'] if relief_cm is not None else -t_rc1_ * k_r_),
                    'term_fault': fd_, 'term_joint': jg_, 'term_contact': ig_}
        eval_workers = int(cm.get('eval_workers', 1))
        _outside_window = lambda P_: (np.linalg.norm(P_ - np.asarray(cm['window']['center'], float), axis=1) > float(cm['window']['radius_m'])
                                      if cm.get('window') else np.zeros(len(P_), bool))

        def _evaluate_par(c, chunk=60000):
            n_ = len(c['b'])
            if eval_workers <= 1 or n_ <= 2 * chunk:
                return _evaluate(c)
            from concurrent.futures import ThreadPoolExecutor
            parts = [(a0, min(n_, a0 + chunk)) for a0 in range(0, n_, chunk)]
            with ThreadPoolExecutor(eval_workers) as ex_pool:
                outs = list(ex_pool.map(lambda ab: _evaluate({k: v[ab[0]:ab[1]] for k, v in c.items()}), parts))
            return {k: np.concatenate([o_[k] for o_ in outs]) for k in outs[0]}
        carry = {'b': base.copy(), 'o': W['outward'] / np.maximum(on, 1e-30), 'w': w.copy(), 'meso': meso.copy(), 'fade': fade.copy(),
                 'recess': recess.copy(), 'gscale': gscale.copy(), 'bare_w': bare_w.copy(), 'hold': W['hold'].astype(float)}
        exact0 = _evaluate(carry)
        # the evaluator must reproduce this function's own coarse result
        self_dev = float(np.abs(exact0['V'] - V).max())
        if self_dev > 1e-9:
            raise RuntimeError('cm evaluator differs from erode on the coarse sheet by %.3g m' % self_dev)
        cams = Cameras(cm.get('cameras') or [], float(cm.get('tol_min_m', 1e-3)), float(cm.get('tol_max_m', 8e-3)))
        tol_of = cams.tolerance if cm.get('cameras') else (lambda P: np.full(len(P), float(cm.get('tol_max_m', 8e-3))))
        _rtf = float(cm.get('refine_tol_factor', 1.))
        tol_refine = (lambda P_: _rtf * tol_of(P_)) if _rtf != 1. else tol_of
        piece_list = []
        for cls_, own_, width_, fw_ in (('fault', faults, damage, 1.), ('joint', joints, jw, 1.3), ('contact', intrusions, jw, 1.)):
            for iid_, iv_, if_ in truth.interfaces:
                if iid_.split(':', 1)[0] in own_ and len(if_):
                    for n_, o_, tri_ in _plane_groups(iv_, if_):
                        piece_list.append((cls_, n_, o_, tri_, 2.2 * width_ * fw_, width_ / float(cm.get('struct_samples_per_width', 4.))))
        amp_min = float(cm.get('struct_min_amp_m', .005))

        def _struct(Vc, u_, L_, ex_):
            need = np.zeros(len(u_), bool)
            A_, B_ = Vc[u_[:, 0]], Vc[u_[:, 1]]
            # only where a groove of the class is actually dug (planned amplitude >= struct_min_amp_m at an end)
            dug = {k: np.maximum(ex_['pot_' + k][u_[:, 0]], ex_['pot_' + k][u_[:, 1]]) >= amp_min for k in ('fault', 'joint', 'contact')}
            for cls_, n_, o_, tri_, reach_, target_ in piece_list:
                da, db = A_ @ n_ - o_, B_ @ n_ - o_
                band = (np.minimum(da, db) <= reach_) & (np.maximum(da, db) >= -reach_) & (L_ > target_) & ~need & dug[cls_]
                if not band.any():
                    continue
                ix = np.flatnonzero(band)
                M_ = .5 * (A_[ix] + B_[ix])
                ins = np.zeros(len(ix), bool)
                # finite piece (1 cm margin on the plane): either end or the midpoint projected inside
                for P_ in (A_[ix], B_[ix], M_):
                    dd = P_ @ n_ - o_
                    ins |= inside_triangles_grid(P_ - dd[:, None] * n_, tri_, n_, eps=1e-2 / max(float(np.sqrt(np.abs(np.cross(tri_[:, 1] - tri_[:, 0], tri_[:, 2] - tri_[:, 0])).sum(1).max())), 1e-6))
                need[ix[ins]] = True
            return need
        ref_ = cmgeom.Refiner(_evaluate_par, tol_refine, _struct, min_edge=float(cm.get('min_edge_m', .005)),
                              max_vertices=int(cm.get('max_vertices', 1_500_000)), max_rounds=int(cm.get('max_rounds', 12)),
                              log=lambda msg: print('[cm %6.1f s] %s' % (__import__('time').time() - t_cm, msg), flush=True),
                              skip_vertex=(W['hold'] | _outside_window(base)) if cm.get('window') else W['hold'].copy(),
                              centroids=bool(cm.get('centroid_test', False)), interior=cm.get('interior_points'))
        F, carry, exact_c, rstats = ref_.run(F, carry, exact0)
        _mark('cm_refine')
        # final fine mesh: fold check against the reference sheet, halving the offsets locally as above
        base = carry['b']
        V = exact_c['V']
        total = exact_c['total']
        ref_n = np.cross(base[F[:, 1]] - base[F[:, 0]], base[F[:, 2]] - base[F[:, 0]])
        o_v = exact_c['outward']
        flipped_c = lambda V_: (np.cross(V_[F[:, 1]] - V_[F[:, 0]], V_[F[:, 2]] - V_[F[:, 0]]) * ref_n).sum(1) <= 0
        folds_cm = []
        guard_rounds = []
        for g_round in range(3 if cm.get('guard_refine') else 1):
            g_extra = np.ones(len(V))
            for _ in range(12):
                flip = flipped_c(V)
                folds_cm.append(int(flip.sum()))
                if not flip.any():
                    break
                mark = np.zeros(len(V))
                mark[np.unique(F[flip])] = 1.
                reach = np.clip(np.maximum(4 * smooth(mark, F, len(mark), 6), mark), 0, 1)
                g_extra *= 1 - .5 * reach
                V = base - o_v * (total * g_extra)[:, None]
            else:
                flip = flipped_c(V)
            if not cm.get('guard_refine') or (g_extra == 1.).all():
                break
            # surface (it is per vertex, so inside the coarse mesh it is linear) and the mesh is refined again around the
            near = g_extra < 1.
            for _ in range(3):
                nb_ = np.zeros(len(V), bool)
                nb_[F[near[F].any(1)].reshape(-1)] = True
                near = nb_
            carry['gscale'] = carry['gscale'] * g_extra
            exact_c = _evaluate_par(carry)
            ref2 = cmgeom.Refiner(_evaluate_par, tol_refine, _struct, min_edge=float(cm.get('min_edge_m', .005)),
                                  max_vertices=int(cm.get('max_vertices', 1_500_000)), max_rounds=int(cm.get('max_rounds', 12)),
                                  log=lambda msg: print('[cm guard %d] %s' % (g_round, msg), flush=True),
                                  skip_vertex=(carry['hold'] >= 1.) | ~near,
                                  interior=cm.get('interior_points'))
            F, carry, exact_c, rs2 = ref2.run(F, carry, exact_c)
            guard_rounds.append({'guarded_vertices': int((g_extra < 1).sum()), 'refine': rs2})
            base, V, total, o_v = carry['b'], exact_c['V'], exact_c['total'], exact_c['outward']
            ref_n = np.cross(base[F[:, 1]] - base[F[:, 0]], base[F[:, 2]] - base[F[:, 0]])
            flipped_c = lambda V_: (np.cross(V_[F[:, 1]] - V_[F[:, 0]], V_[F[:, 2]] - V_[F[:, 0]]) * ref_n).sum(1) <= 0
            folds_cm.append('guard_refine')
        folds_final = int(flip.sum())
        folds = folds + ['cm'] + folds_cm
        g_sh = np.random.default_rng(int(cm.get('shape_error_seed', 0)))
        ns_ = 0 if cm.get('shape_check') else int(cm.get('shape_error_samples', 1))
        fi_ = np.repeat(np.arange(len(F)), ns_)
        r_ = g_sh.random((len(fi_), 2))
        fl_ = r_.sum(1) > 1
        r_[fl_] = 1 - r_[fl_]
        bc_ = np.c_[1 - r_.sum(1), r_]
        sh_err, sh_tol = np.zeros(len(fi_)), np.zeros(len(fi_))
        for a0 in range(0, len(fi_), 400000):
            sl = slice(a0, a0 + 400000)
            ff, bb = F[fi_[sl]], bc_[sl]
            cs_ = {k: (v[ff] * (bb[:, :, None] if v.ndim == 2 else bb)).sum(1) for k, v in carry.items()}
            ex_ = _evaluate(cs_)
            ge_ = (g_extra[ff] * bb).sum(1)
            Ve = cs_['b'] - ex_['outward'] * (ex_['total'] * ge_)[:, None]
            Vl = (V[ff] * bb[:, :, None]).sum(1)
            sh_err[sl] = np.linalg.norm(Ve - Vl, axis=1)
            sh_tol[sl] = tol_of(Vl)
        ratio = sh_err / sh_tol if len(sh_err) else np.zeros(1)
        t_sc = __import__('time').time()
        shape_check, shape_exceed = None, None
        if cm.get('shape_check'):
            shape_check, shape_exceed = _shape_check(cm['shape_check'], F, V, carry, exact_c, g_extra, _evaluate_par, tol_of,
                                                     faces=(np.flatnonzero(~_outside_window(V[F].mean(1))) if cm.get('window') else None))
        shape_check_s = __import__('time').time() - t_sc
        _mark('cm_shape_check')
        shape_error = {'samples': int(len(fi_)), 'per_face': ns_, 'seed': int(cm.get('shape_error_seed', 0)),
                       'ratio_p50_p99_max': [float(np.percentile(ratio, 50)), float(np.percentile(ratio, 99)), float(ratio.max())],
                       'err_mm_p50_p99_max': ([float(np.percentile(sh_err, 50) * 1e3), float(np.percentile(sh_err, 99) * 1e3), float(sh_err.max() * 1e3)]
                                              if len(sh_err) else None),
                       'by_tolerance_mm': {('%.1f-%.1f' % (lo * 1e3, hi * 1e3)): ([float(np.percentile(ratio[(sh_tol >= lo) & (sh_tol < hi)], 99)), int(((sh_tol >= lo) & (sh_tol < hi)).sum())]
                                                                                 if len(sh_tol) and ((sh_tol >= lo) & (sh_tol < hi)).any() else None)
                                           for lo, hi in ((1e-3, 2e-3), (2e-3, 4e-3), (4e-3, 7.9e-3), (7.9e-3, 1.))},
                       'rule': 'p99(err / tol) <= 1 and max(err / tol) <= 2; err = |exact - linear| at random barycentric points'}
        if not ns_:
            shape_error = None
        total = total * g_extra
        ref, outward_v, bare_w, fade = exact_c['ref'], o_v, carry['bare_w'], carry['fade']
        fault_zone, groove_profile = exact_c['fault_zone'], exact_c['groove_profile']
        # exact groove parts / potentials of the evaluator; the fold-guard factor scales the dug depth like the offsets
        groove_parts = {k: (exact_c['gp_%s_0' % k] * g_extra * carry['gscale'], exact_c['gp_%s_1' % k]) for k in ('fault', 'joint', 'contact')}
        groove_potential = {k: exact_c['pot_' + k] for k in ('fault', 'joint', 'contact')}
        W['outward'] = o_v
        sheet = np.ones(len(V), bool)
        cm_info = {'sheet_edge_m': float(cm['sheet_edge_m']), 'smoothing_scale': (cmgeom.HIST_EDGE_M / float(cm['sheet_edge_m'])) ** 2,
                   'evaluator_self_deviation_m': self_dev, 'refine': rstats, 'fold_guard_fine': folds_cm, 'vertices': int(len(V)), 'faces': int(len(F)),
                   'tolerance': {'min_m': float(cm.get('tol_min_m', 1e-3)), 'max_m': float(cm.get('tol_max_m', 8e-3)), 'cameras': len(cm.get('cameras') or [])},
                   'structure_pieces': len(piece_list), 'shape_error': shape_error, 'seconds': __import__('time').time() - t_cm,
                   'shape_check': shape_check, 'shape_check_seconds': shape_check_s, 'eval_workers': eval_workers,
                   'guard_refine': guard_rounds,
                   'relief': ({'file': str(cm['relief']), 'rms_mm': float(_rh.std() * 1e3), 'texel_m': _rt, 'tile_m': _rt * _rn,
                               'mapping': 'vertical plane normal to the exposure view direction', 'weight': 'fade x bare_w'} if relief_cm is not None else None)}
    if debug is not None:
        debug.update(V_final=V, base=base, sheet=sheet, total=total, fade=fade, F=F, groove_parts=groove_parts,
                     fold_flip_final=np.nonzero(flip)[0])
    # Line labels are the v2 planes cut by the final surface: the grooves must be centred there.
    alignment = groove_alignment(V, groove_parts, truth, {'fault': faults, 'joint': joints, 'contact': intrusions},
                                 {'fault': damage, 'joint': jw, 'contact': jw})
    outward_v = W['outward']
    if exact is not None:
        # blocks-3: split every edge between two pieces at their exact crossing (riser edges), with
        # every per-vertex field used below interpolated; the spalling is then added along the
        # reference normal: V = V (fold-guarded, unspalled) + m x offset x fade.
        from .fracture import insert_edges
        fval, fpid, fa, fc = exact['fx'].evaluate(V, exact['m'], exact['fx'].nb, exact['fx'].C)
        # No spalling where the unspalled surface is already steep to the spalling direction (groove
        # and trench walls, steep relief sides): a height field cannot be added there without
        # folding faces. This limits where spalling applies; it does not smooth the risers.
        # Per face (vertex normals average single steep faces away): faces of the unspalled surface
        # at less than cos 0.35 to the spalling direction, their vertices and two rings around.
        fnv = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        fnv /= np.maximum(np.linalg.norm(fnv, axis=1, keepdims=True), 1e-20)
        if (fnv * W['outward'][F].mean(1)).sum() < 0:
            fnv = -fnv
        steep = (fnv * -exact['m'][F].mean(1)).sum(1) < .35
        mark = np.zeros(len(V))
        mark[np.unique(F[steep])] = 1.
        keep_out = np.clip(4 * smooth(mark, F, len(V), 2), 0, 1)
        exact['weight'] = exact['weight'] * (1 - keep_out)
        arrays = dict(p0=V, m=exact['m'], nb=exact['fx'].nb, C=exact['fx'].C, frac=fval,
                      pid=fpid, a=fa, c=fc, weight=exact['weight'], Vn=V, base0=base,
                      ref=ref, outward=W['outward'], bare_w=bare_w, fault_zone=fault_zone,
                      groove_profile=groove_profile, sheetf=sheet.astype(float),
                      **{'pot_' + k: v for k, v in groove_potential.items()},
                      **{'gp_%s_%d' % (k, i): v[i] for k, v in groove_parts.items() for i in (0, 1)})
        nv0 = len(V)
        F, arrays, _, n_ins = insert_edges(F, arrays, lambda sub: exact['fx'].evaluate(sub['p0'], sub['m'], sub['nb'], sub['C']))
        spall = arrays['frac'] * arrays['weight']
        V = arrays['Vn'] + arrays['m'] * spall[:, None]
        ref, outward_v, bare_w = arrays['ref'], arrays['outward'], arrays['bare_w']
        fault_zone, groove_profile = arrays['fault_zone'], arrays['groove_profile']
        groove_potential = {k: arrays['pot_' + k] for k in groove_potential}
        groove_parts = {k: (arrays['gp_%s_0' % k], arrays['gp_%s_1' % k]) for k in groove_parts}
        b = arrays['base0']
        rn = np.cross(b[F[:, 1]] - b[F[:, 0]], b[F[:, 2]] - b[F[:, 0]])
        fn_ = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        flip_mask = ((fn_ * rn).sum(1) <= 0) & (arrays['sheetf'][F] > .5).all(1)
        flip_after = int(flip_mask.sum())
        if debug is not None:
            debug.update(post_insert=dict(V=V, F=F, arrays=arrays, flipped=np.nonzero(flip_mask)[0], nv0=nv0))
        folds_final = max(folds_final, flip_after)
        # the groove check on the final (spalled) surface
        alignment = groove_alignment(V, groove_parts, truth, {'fault': faults, 'joint': joints, 'contact': intrusions},
                                     {'fault': damage, 'joint': jw, 'contact': jw})
        fracture_info.update(edges_inserted=int(n_ins), vertices_added=int(len(V) - nv0), flipped_after_insert=flip_after,
                             spall_m={'p50': float(np.median(spall)), 'p95': float(np.percentile(spall, 95)), 'max': float(spall.max())})
    blockface_info = None
    cop_f = nout_f = cop_records = None
    detail_frame = detail_info = None
    # through the B construction and the contact refinement, so the footprint gets a reference position (final
    # position minus that displacement only: B spalling and every other deformation kept) for the bed texture's UV
    dvec = None
    want_ref = blockface is not None and ((blockface.get('detail') or {}).get('material') or {}).get('recipe') == 'lum-5'
    wall = None
    wall_cm = None
    if blockface is not None and blockface.get('mode') == 'wall' and (blockface.get('detail') or {}).get('method') != 'mesh':
        # blocks per region with the strict floor caps and exact floor-vertex sharing; no scan detail. The 1 um weld
        # tolerance applies everywhere (the exact construction closes without welding; only coincident points merge).
        from . import wallscene as wsc
        import time as _time
        t_w = _time.time()
        log_w = (lambda msg: print('[wall %6.0f s] %s' % (_time.time() - t_w, msg), flush=True))
        if blockface.get('detail'):
            raise ValueError('cm wall: detail %r is not implemented yet' % blockface['detail'].get('method'))
        workers = int(blockface.get('workers', 1))
        wr = wsc.wall_regions(V, F, outward_v, {'bare_w': bare_w, 'fade': fade, 'fault_zone': fault_zone, 'tile_ramp': tile_ramp},
                              truth, {**blockface.get('region', {}), **({'window': cm['window']} if (cm or {}).get('window') else {})})
        log_w('regions: %d B regions %.1f m2' % (len(wr['regions']), wr['stats']['region_area_m2']))
        m_w = wr['m']
        rv_ = np.unique(np.concatenate([F[R['faces']].reshape(-1) for R in wr['regions']]))
        (a1_w, a2_w), nb_w = wsc.axes(truth, V[rv_], m_w)
        dvec = np.zeros((len(V), 3))
        face_region = np.full(len(F), -1, np.int64)
        for r_, R_ in enumerate(wr['regions']):
            face_region[R_['faces']] = r_
        bres = wsc.apply_blocks(V, F, face_region, wr['taper'], np.ones(len(V)), truth, m_w, wr['regions'],
                                {'depth': blockface['depth'], 'parting': blockface['parting'], 'seed': blockface['seed'],
                                 'weld_tol': blockface.get('weld_tol', 5e-4), 'weld_fine_m': float(blockface.get('weld_fine_m', 1e-6)),
                                 'floor_cap_tol': float(blockface.get('floor_cap_tol', 1e-9)),
                                 'exact_floor_vertices': bool(blockface.get('exact_floor_vertices', True)),
                                 'nano_merge_m': blockface.get('nano_merge_m'),
                                 'thin_triangle_width_m': blockface.get('thin_triangle_width_m'),
                                 'spall': blockface.get('spall'), 'fast': blockface.get('fast', False),
                                 'bottom_depth_m': blockface.get('bottom_depth_m'),
                                 'dump_dir': blockface.get('dump_dir')}, workers, log_w)
        corner = F[bres['bprism']]
        bbary = bres['bbary']

        def _interp_c(arr):
            return (arr[corner] * (bbary[:, :, None] if arr.ndim == 2 else bbary)).sum(1)
        nv0 = len(V)
        keep_f = np.ones(len(F), bool)
        keep_f[bres['removed']] = False
        taper_w = wr['taper']
        ref, outward_v, bare_w, fault_zone, groove_profile, dvec, taper_w = (np.concatenate([a_, _interp_c(a_)]) for a_ in
                                                                             (ref, outward_v, bare_w, fault_zone, groove_profile, dvec, taper_w))
        V = np.concatenate([V, bres['bv']])
        nk = int(keep_f.sum())
        cop_f = np.concatenate([np.zeros(nk, np.int64), bres['code']])
        nout_f = np.concatenate([np.zeros((nk, 3)), bres['nout']])
        riser_f = np.concatenate([np.zeros(nk, bool), np.asarray(bres['bkind']) == 'riser'])
        relief_f = np.zeros(nk + len(bres['bfaces']), bool)
        breg_f = np.concatenate([np.full(nk, -1, np.int64), bres['b_region']])
        bcell_f = np.concatenate([np.full(nk, -1, np.int64), bres['b_cell']])
        F = np.concatenate([F[keep_f], bres['bfaces'] + nv0])
        cop_records = bres['records']
        used = np.zeros(len(V), bool)
        used[F.reshape(-1)] = True
        zero_new = lambda a_: np.where(used, np.concatenate([a_, np.zeros(len(bres['bv']))]), 0.)
        groove_potential = {k: zero_new(v) for k, v in groove_potential.items()}
        groove_parts = {k: (zero_new(a_), zero_new(b_)) for k, (a_, b_) in groove_parts.items()}
        alignment = groove_alignment(V, groove_parts, truth, {'fault': faults, 'joint': joints, 'contact': intrusions},
                                     {'fault': damage, 'joint': jw, 'contact': jw})
        bm_ = np.asarray(bres['bkind']) == 'riser'
        blockface_info = {'mode': 'wall', 'b_enabled': True, 'm': m_w.tolist(), 'a1': a1_w.tolist(), 'a2': a2_w.tolist(), 'bedding_normal': nb_w.tolist(),
                          'depth_m': blockface['depth'], 'parting_m': blockface['parting'], 'seed': blockface['seed'],
                          'regions': wr['stats'], 'detail': None, 'blocks': bres['regions'],
                          'faces_added': int(len(bres['bfaces'])), 'faces_removed': int(len(bres['removed'])), 'risers_faces': int(bm_.sum()),
                          'floor_cap_tol': float(blockface.get('floor_cap_tol', 1e-9)), 'exact_floor_vertices': bool(blockface.get('exact_floor_vertices', True)),
                          'weld_fine_everywhere_m': float(blockface.get('weld_fine_m', 1e-6)), 'seconds': _time.time() - t_w,
                          'note': 'whole-wall scene, centimetre route: explicit residual blocks per region (fault block) on the refined '
                                  'surface; in-bed parting steps carry no line label; groove check excluded inside the regions (joint '
                                  'attribution of risers instead); no scan detail'}
        wall_cm = {'taper': taper_w}
        log_w('B done: %d vertices, %d faces' % (len(V), len(F)))
        _mark('blocks')
    elif blockface is not None and blockface.get('mode') == 'wall':
        # multi-footprint scan detail from the donor atlas at imaging-scale tolerance, B per region. Produces the same
        # variables as the B sample branch below (cop_f, nout_f, riser_f, cop_records, dvec, blockface_info, ...).
        from . import wallscene as wsc
        from .blockface import _area_normal
        import json as _json
        import time as _time
        from pathlib import Path as _Path
        t_w = _time.time()
        log_w = (lambda msg: print('[wall %6.0f s] %s' % (_time.time() - t_w, msg), flush=True))
        det = blockface['detail']
        workers = int(blockface.get('workers', 1))
        wr = wsc.wall_regions(V, F, outward_v, {'bare_w': bare_w, 'fade': fade, 'fault_zone': fault_zone, 'tile_ramp': tile_ramp},
                              truth, blockface.get('region', {}))
        log_w('regions: %d B regions %.1f m2, detail %.1f m2' % (len(wr['regions']), wr['stats']['region_area_m2'], wr['stats']['detail_area_m2']))
        m_w = wr['m']
        rv_ = np.unique(np.concatenate([F[R['faces']].reshape(-1) for R in wr['regions']]))
        (a1_w, a2_w), nb_w = wsc.axes(truth, V[rv_], m_w)
        atlas_dir = _Path(det['donor_atlas'])
        atlas_info = _json.loads((atlas_dir / 'atlas.json').read_text(encoding='utf8'))
        W_f, blend_f = float(det.get('W_m', 1.3)), float(det.get('blend_m', .15))
        o_w, feet, lay_stats = wsc.layout(V, F, wr['detail_faces'], wr['taper_detail'], a1_w, a2_w, m_w, W_f, blend_f, float(det.get('margin_m', .1)))
        win_stats = wsc.assign_windows(feet, atlas_info['usable_windows']['centres'], int(det.get('window_seed', blockface['seed'])),
                                       float(det.get('min_source_sep_m', .5)), float(det.get('neighbour_m', 3.)))
        cams = wsc.Cameras(blockface['cameras'], float(det.get('tol_min_m', wsc.TOL_MIN)), float(det.get('tol_max_m', wsc.TOL_MAX)))
        log_w('layout: %d footprints' % len(feet))
        carried = {'ref': ref, 'outward_v': outward_v, 'bare_w': bare_w, 'fault_zone': fault_zone, 'groove_profile': groove_profile,
                   'taper': wr['taper'], 'taper_d': wr['taper_detail'], **{'pot_' + k: v for k, v in groove_potential.items()},
                   **{'gp_%s_%d' % (k, i): v[i] for k, v in groove_parts.items() for i in (0, 1)}}
        dopts = {'a1': a1_w, 'a2': a2_w, 'W_m': W_f, 'blend_m': blend_f, 'margin_m': float(det.get('margin_m', .1)), 'detail': det,
                 'constrain_below': float(det.get('constrain_below_m', wsc.CONSTRAIN_BELOW)), 'min_split_floor_m': float(det.get('min_split_m', 1e-3)),
                 'dump_dir': blockface.get('dump_dir')}
        V, F, carried, dvec, face_region, vreg_w, dstats = wsc.apply_detail(V, F, carried, wr, feet, cams, dopts, atlas_dir / 'mesh', workers, log_w)
        ref, outward_v, bare_w, fault_zone, groove_profile = (carried[k] for k in ('ref', 'outward_v', 'bare_w', 'fault_zone', 'groove_profile'))
        taper_w, taper_d = carried['taper'], carried['taper_d']
        groove_potential = {k: carried['pot_' + k] for k in groove_potential}
        groove_parts = {k: (carried['gp_%s_0' % k], carried['gp_%s_1' % k]) for k in groove_parts}
        lay = wsc.Layout(o_w, a1_w, a2_w, m_w, feet, W_f, blend_f, atlas_info['bounds_m'])
        _, _, wc_ = lay.classes(lay.xy(V))
        detail_w = taper_d * wc_.sum(1)
        log_w('detail done: %d vertices, %d faces' % (len(V), len(F)))
        bres = wsc.apply_blocks(V, F, face_region, taper_w, detail_w, truth, m_w, wr['regions'],
                                {'depth': blockface['depth'], 'parting': blockface['parting'], 'seed': blockface['seed'],
                                 'weld_tol': blockface.get('weld_tol', 5e-4), 'weld_fine_m': float(det.get('weld_fine_m', 1e-6)),
                                 'dump_dir': blockface.get('dump_dir')}, workers, log_w)
        corner = F[bres['bprism']]
        bbary = bres['bbary']

        def _interp_w(arr):
            return (arr[corner] * (bbary[:, :, None] if arr.ndim == 2 else bbary)).sum(1)
        nv0 = len(V)
        keep_f = np.ones(len(F), bool)
        keep_f[bres['removed']] = False
        ref, outward_v, bare_w, fault_zone, groove_profile, dvec, taper_w, taper_d = (np.concatenate([a_, _interp_w(a_)]) for a_ in
                                                                                       (ref, outward_v, bare_w, fault_zone, groove_profile, dvec, taper_w, taper_d))
        V = np.concatenate([V, bres['bv']])
        nk = int(keep_f.sum())
        cop_f = np.concatenate([np.zeros(nk, np.int64), bres['code']])
        nout_f = np.concatenate([np.zeros((nk, 3)), bres['nout']])
        riser_f = np.concatenate([np.zeros(nk, bool), np.asarray(bres['bkind']) == 'riser'])
        relief_f = np.zeros(nk + len(bres['bfaces']), bool)
        # per face: B region and cell of the new faces (-1 elsewhere), for the geometry checks (wallcheck.py)
        breg_f = np.concatenate([np.full(nk, -1, np.int64), bres['b_region']])
        bcell_f = np.concatenate([np.full(nk, -1, np.int64), bres['b_cell']])
        F = np.concatenate([F[keep_f], bres['bfaces'] + nv0])
        cop_records = bres['records']
        used = np.zeros(len(V), bool)
        used[F.reshape(-1)] = True
        zero_new = lambda a_: np.where(used, np.concatenate([a_, np.zeros(len(bres['bv']))]), 0.)
        groove_potential = {k: zero_new(v) for k, v in groove_potential.items()}
        groove_parts = {k: (zero_new(a_), zero_new(b_)) for k, (a_, b_) in groove_parts.items()}
        alignment = groove_alignment(V, groove_parts, truth, {'fault': faults, 'joint': joints, 'contact': intrusions},
                                     {'fault': damage, 'joint': jw, 'contact': jw})
        bm_ = np.asarray(bres['bkind']) == 'riser'
        blockface_info = {'mode': 'wall', 'm': m_w.tolist(), 'a1': a1_w.tolist(), 'a2': a2_w.tolist(), 'bedding_normal': nb_w.tolist(),
                          'depth_m': blockface['depth'], 'parting_m': blockface['parting'], 'seed': blockface['seed'],
                          'regions': wr['stats'], 'layout': {**lay_stats, 'origin': o_w.tolist(), 'windows': win_stats, 'footprints': feet},
                          'detail': {**{k: v for k, v in det.items()}, **dstats}, 'blocks': bres['regions'],
                          'faces_added': int(len(bres['bfaces'])), 'faces_removed': int(len(bres['removed'])), 'risers_faces': int(bm_.sum()),
                          'cameras': blockface['cameras'], 'seconds': _time.time() - t_w,
                          'note': 'whole-wall scene: explicit residual blocks per region (fault block); in-bed parting steps carry no line '
                                  'label; groove check excluded inside the regions (joint attribution of risers instead); scan detail from '
                                  'the donor atlas at imaging-scale tolerance'}
        wall = {'lay': lay, 'feet': feet, 'atlas_dir': atlas_dir, 'atlas_info': atlas_info, 'taper': taper_d}
        log_w('B done: %d vertices, %d faces' % (len(V), len(F)))
    elif blockface is not None:
        # Route B sample (blockface.py): the patch triangles are replaced by explicit residual blocks
        # (floor caps and risers in the bedding / contact / own-block joint planes) with their own
        # vertices; per-vertex fields are interpolated from each prism's top triangle. Inside the patch
        # no groove is dug any more: groove potential and parts are zero there and the groove check
        # runs on the final mesh (outside the patch); the joint attribution of the risers replaces it.
        from .blockface import patch_setup, build_patch, triangulate, weld_slivers
        if blockface.get('detail'):
            # added to the patch surface around a footprint before the blocks are cut, so the risers
            # stay in their planes and the floor-riser edges follow the relief.
            from .scandetail import apply_top
            dres = apply_top(V, F, blockface['frame'], blockface['detail'], truth)
            nV_ = len(V)
            npar = len(dres['parents'])

            def grow_d(a_):
                out_ = np.concatenate([a_, np.zeros((npar,) + a_.shape[1:])])
                for j, (idx_, w_) in enumerate(dres['parents']):
                    out_[nV_ + j] = sum(wk * out_[ik] for ik, wk in zip(idx_, w_))
                return out_
            ref, outward_v, bare_w, fault_zone, groove_profile = (grow_d(a_) for a_ in (ref, outward_v, bare_w, fault_zone, groove_profile))
            groove_potential = {k: grow_d(v) for k, v in groove_potential.items()}
            groove_parts = {k: (grow_d(a_), grow_d(b_)) for k, (a_, b_) in groove_parts.items()}
            V = np.concatenate([V, dres['new_positions']])
            for v_, p_ in dres['moved_original'].items():
                V[v_] = p_
            if want_ref:
                loc_ = dres['local']
                dvec = np.zeros((len(V), 3))
                dvec[loc_['gidx']] = -np.asarray(loc_['disp'], float)[:, None] * np.asarray(dres['frame'][3], float)[None]
            keep_d = np.ones(len(F), bool)
            keep_d[dres['faces_replaced']] = False
            F = np.concatenate([F[keep_d], dres['faces']])
            detail_frame, detail_info = dres['frame'], dres['stats']
        bf_inside, bf_taper, bf_m, bf_cut, bf_cell, bf_depth, binfo = patch_setup(
            V, F, truth, blockface['frame'], depth=blockface['depth'], parting=blockface['parting'], seed=blockface['seed'])
        bf_polys, bstats = build_patch(V, F, bf_inside, bf_taper, bf_m, bf_cut, bf_cell, bf_depth, exact=blockface.get('exact', False))
        if blockface.get('weld', True):
            wtol = blockface.get('weld_tol', 5e-4)
            if detail_frame is not None and blockface['detail'].get('weld') == 'zoned':
                # points (fine_tol) merge within the footprint + 15 cm; the rest of the patch keeps wtol;
                # no vertex moves its tolerance or more
                cf_, a1_, a2_, _, W_, _ = detail_frame
                fine = float(blockface['detail'].get('weld_fine_m', 1e-6))
                tol_of = lambda P: np.where((np.abs(np.c_[(P - cf_) @ a1_, (P - cf_) @ a2_]) <= W_ / 2 + .15).all(1), fine, wtol)
                bf_polys, bstats['weld'] = weld_slivers(bf_polys, bf_cut, tol=wtol, max_move=wtol, tol_of=tol_of,
                                                        fine_min_area=0. if blockface.get('exact', False) else None)
            else:
                bf_polys, bstats['weld'] = weld_slivers(bf_polys, bf_cut, tol=wtol)
        bv, bfaces, bprism, bbary, bkind, bfpoly = triangulate(bf_polys, V, F, bf_m)
        corner = F[bprism]
        # labelled as interface faces; per face: record index + 1 (0 = none) and the outward normal of
        # its polygon (to look up its unit on the kept-rock side).
        cop_records, rec_of, poly_code, poly_nout = [], {}, np.zeros(len(bf_polys), np.int64), np.zeros((len(bf_polys), 3))
        from .blockface import _area_normal
        for i, (pp, k_, inf_) in enumerate(bf_polys):
            nn = _area_normal(pp)
            poly_nout[i] = nn / max(np.linalg.norm(nn), 1e-15)
            key = '%s|%d' % inf_['plane'] if k_ == 'riser' else None
            rec = binfo['coplanar'].get(key) if key else None
            if rec is not None:
                if rec['id'] not in rec_of:
                    rec_of[rec['id']] = len(cop_records)
                    cop_records.append(dict(rec))
                poly_code[i] = rec_of[rec['id']] + 1

        def _interp(arr):
            return (arr[corner] * (bbary[:, :, None] if arr.ndim == 2 else bbary)).sum(1)
        nv0 = len(V)
        keep_f = np.ones(len(F), bool)
        keep_f[bf_inside] = False
        ref, outward_v, bare_w, fault_zone, groove_profile = (np.concatenate([a_, _interp(a_)]) for a_ in
                                                              (ref, outward_v, bare_w, fault_zone, groove_profile))
        if dvec is not None:
            dvec = np.concatenate([dvec, _interp(dvec)])
        V = np.concatenate([V, bv])
        cop_f = np.concatenate([np.zeros(int(keep_f.sum()), np.int64), poly_code[bfpoly]])
        nout_f = np.concatenate([np.zeros((int(keep_f.sum()), 3)), poly_nout[bfpoly]])
        riser_f = np.concatenate([np.zeros(int(keep_f.sum()), bool), np.asarray(bkind) == 'riser'])
        F = np.concatenate([F[keep_f], bfaces + nv0])
        used = np.zeros(len(V), bool)
        used[F.reshape(-1)] = True
        zero_new = lambda a_: np.where(used, np.concatenate([a_, np.zeros(len(bv))]), 0.)
        groove_potential = {k: zero_new(v) for k, v in groove_potential.items()}
        groove_parts = {k: (zero_new(a_), zero_new(b_)) for k, (a_, b_) in groove_parts.items()}
        alignment = groove_alignment(V, groove_parts, truth, {'fault': faults, 'joint': joints, 'contact': intrusions},
                                     {'fault': damage, 'joint': jw, 'contact': jw})
        risers = [i for i, (_, k_, _) in enumerate(bf_polys) if k_ == 'riser']
        from .blockface import _area_normal
        n_over = sum(1 for i in risers if _area_normal(bf_polys[i][0]) @ (-bf_m) <= 0)
        blockface_info = {'frame': blockface['frame'], 'depth_m': blockface['depth'], 'parting_m': blockface['parting'],
                          'seed': blockface['seed'], **binfo, **bstats, 'polygons': len(bf_polys), 'risers': len(risers),
                          'overhanging_risers': n_over, 'faces_added': int(len(bfaces)), 'faces_removed': int((~keep_f).sum()),
                          'note': 'route B sample: explicit residual blocks; in-bed parting steps carry no line label; '
                                  'groove check excluded inside the patch (joint attribution of risers instead)'}
        if detail_info is not None:
            blockface_info['detail'] = {**{k: v for k, v in blockface['detail'].items()}, **detail_info}
        relief_f = np.zeros(len(F), bool)
        if blockface.get('relief'):
            # floors of the cells of one lithology, displaced along m; a procedural weathering
            # appearance proxy. Risers and protected floor edges stay exactly in place.
            from .relief import floor_relief
            rel = {'seed': blockface['seed'], **blockface['relief']}
            nkeep = int(keep_f.sum())
            floor_idx = np.array([nkeep + i for i, k_ in enumerate(bkind) if k_ == 'floor'], np.int64)
            fcell = [bf_polys[bfpoly[i - nkeep]][2]['cell'] for i in floor_idx]
            probe = {}
            for i, c_ in zip(floor_idx, fcell):
                probe.setdefault(c_, V[F[i]].mean(0) + 1e-3 * bf_m)
            pk = list(probe)
            st_p, _, _, l_p = truth.query(np.array([probe[c_] for c_ in pk]))
            want = {c_ for c_, s_, l_ in zip(pk, st_p, l_p) if s_ == 'occupied' and l_ == rel.get('lithology', 'group_a_05')}
            sel = np.array([i for i, c_ in zip(floor_idx, fcell) if c_ in want], np.int64)
            res = floor_relief(V, F, sel, [c_ for c_ in fcell if c_ in want], bf_m, np.asarray(blockface['frame']['centre'], float),
                               rel, bf_cell, segment_distance)
            nV_ = len(V)
            npar = len(res['parents'])

            def grow(a_):
                out_ = np.concatenate([a_, np.zeros((npar,) + a_.shape[1:])])
                for j, (idx_, w_) in enumerate(res['parents']):
                    out_[nV_ + j] = sum(wk * out_[ik] for ik, wk in zip(idx_, w_))
                return out_
            ref, outward_v, bare_w, fault_zone, groove_profile = (grow(a_) for a_ in (ref, outward_v, bare_w, fault_zone, groove_profile))
            if dvec is not None:
                dvec = grow(dvec)
            groove_potential = {k: np.concatenate([v, np.zeros(npar)]) for k, v in groove_potential.items()}
            groove_parts = {k: (np.concatenate([a_, np.zeros(npar)]), np.concatenate([b_, np.zeros(npar)])) for k, (a_, b_) in groove_parts.items()}
            V = np.concatenate([V, res['new_positions']])
            for v_, p_ in res['moved_original'].items():
                V[v_] = p_
            keep2 = np.ones(len(F), bool)
            keep2[sel] = False
            pf = res['parent_face']
            F = np.concatenate([F[keep2], res['faces']])
            cop_f = np.concatenate([cop_f[keep2], cop_f[pf]])
            nout_f = np.concatenate([nout_f[keep2], nout_f[pf]])
            riser_f = np.concatenate([riser_f[keep2], riser_f[pf]])
            relief_f = np.concatenate([np.zeros(int(keep2.sum()), bool), np.ones(len(pf), bool)])
            dvals = np.array(list(res['disp'].values()))
            blockface_info['relief'] = {**{k: v for k, v in rel.items()}, **res['stats'], 'faces_replaced': int(len(sel)),
                                        'faces_added': int(len(pf)), 'vertices_added': npar,
                                        'disp_mm_quantiles_p0_p1_p5_p25_p50_p75_p95_p99_p100':
                                            [float(x) for x in np.percentile(dvals * 1e3, [0, 1, 5, 25, 50, 75, 95, 99, 100])],
                                        'cells_selected': len(want)}
            if debug is not None:
                debug['relief_field'] = res['field']
                debug['relief_basis'] = res['basis']
    applied = np.maximum(((ref - V) * outward_v).sum(1), 0) if surface.origin_m is None else \
        np.maximum(np.linalg.norm(ref - surface.origin_m, axis=1) - np.linalg.norm(V - surface.origin_m, axis=1), 0)

    # 4. normals: one global outward sign, never per face.
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if (n * outward_v[F].mean(1)).sum() < 0:
        F = F[:, ::-1]

    def outward_normals(V, F):
        N = vertex_normals(V, F, np.array([0., 0., 1.]), consistent=True)
        ref_dir = outward_v if len(V) == len(outward_v) else None
        if ref_dir is not None and (N * ref_dir).sum() < 0:
            N = -N
        return N
    N = outward_normals(V, F)

    # 5. colour and cover (cover = visible soil on gentle rock surfaces; geology stays underneath).
    nv = len(V)
    body_color = {}

    def tint(b, l):
        if b not in body_color:
            body_color[b] = bed_tint(lith_params(l)['base'], rng, tint_jitter)
        return body_color[b]
    q_v = truth.query(V)
    st_v, b_v, _, l_v = q_v
    color = np.array([tint(b, l) if s == 'occupied' else (.35, .3, .25) for s, b, l in zip(st_v, b_v, l_v)])
    color = smooth(color, F, nv, _it(2))
    mott = fbm(V * float(rng.uniform(.6, 1.5)), seed + 4, 4)
    cavity = np.clip(applied / max(float(np.percentile(applied, 98)), 1e-6), 0, 1)
    rgb = color * (1 + .3 * mott)[:, None] * (1 - .35 * cavity)[:, None]
    iron = np.clip(np.array([.55, .32, .16]) * rng.uniform(.6, 1.1) + rng.normal(0, .04, 3), .05, .9)
    stain = np.clip(np.maximum(fault_zone, .6 * groove_profile) * np.clip((value_noise(V * .6, seed + 9) - .45) * 3, 0, 1), 0, 1)
    k_stain = float(rng.uniform(.1, .4))
    rgb = rgb * (1 - k_stain * stain[:, None]) + iron * (k_stain * stain[:, None])
    streak = np.clip(fbm(V * np.array([2.5, 2.5, .15]), seed + 5, 3) * 2.2, -1, 1)
    rgb *= (1 + float(rng.uniform(.05, .2)) * streak)[:, None]
    up = N[:, 2]
    cover = np.clip((up - math.cos(math.radians(float(rng.uniform(22, 34))))) * profile['cover_ramp'] + .3 * (value_noise(V * .4, seed + 11) - .5), 0, 1) * (1 - bare_w)
    cover = np.maximum(cover, np.clip((ground_z + .05 - V[:, 2]) / .1, 0, 1))
    cover = np.maximum(cover, (st_v != 'occupied').astype(float))
    if tile_ramp is not None:
        cover = np.maximum(cover, tile_ramp(V))
    cover = smooth(cover, F, nv, _it(4))
    soil_rgb = np.array([[.36, .29, .21], [.42, .35, .26], [.33, .28, .22]])[rng.integers(3)] * rng.uniform(.85, 1.15)
    rgb = rgb * (1 - cover[:, None]) + soil_rgb * (1 + .3 * fbm(V * 3, seed + 16, 3))[:, None] * cover[:, None]
    if clay:
        rgb = np.full_like(rgb, .35)
        cover = np.where(V[:, 2] < ground_z, cover, 0.)
    rgb = np.clip(rgb, .02, .95)
    cover_faces = cover[F].mean(1) > .5

    # 6. exact geological truth on the final surface: refine contact-straddling faces, query v2.
    _mark('colour_query')
    _nv_before_contacts = len(V)
    V_pre = V
    state = np.c_[applied, cover]  # surface state carried through refinement (recession m, cover)
    if (cm or {}).get('debug_pre_contacts'):
        np.savez(cm['debug_pre_contacts'], V=V, F=F, cop_f=cop_f if cop_f is not None else np.zeros(0))
    if cop_f is None:
        V, F, (rgb, cover, state), (cover_faces,), face_keys, refine_stats = refine_contacts(V, F, [rgb, cover, state], [cover_faces], truth, levels=3, known=q_v)
    elif wall_cm is not None and blockface.get('contact_split') == 'exact':
        # B risers lying in a contact plane are not split (their unit is queried inside the kept rock below)
        from .display import split_contacts
        V, F, (rgb, cover, state, dvec, taper_w), (cover_faces, cop_f, nout_f, riser_f, relief_f, breg_f, bcell_f), face_keys, refine_stats = split_contacts(
            V, F, [rgb, cover, state, dvec, wall_cm['taper']], [cover_faces, cop_f, nout_f, riser_f, relief_f, breg_f, bcell_f], truth, known=q_v,
            skip=cop_f > 0)
        wall_cm['taper'] = taper_w
        sel = np.nonzero(cop_f > 0)[0]
        if len(sel):
            st_c, b_c, u_c, l_c = truth.query(V[F[sel]].mean(1) - 1e-3 * nout_f[sel])
            for j, fi in enumerate(sel):
                face_keys[fi] = (b_c[j], u_c[j], l_c[j]) if st_c[j] == 'occupied' else None
    elif wall_cm is not None:
        V, F, (rgb, cover, state, dvec, taper_w), (cover_faces, cop_f, nout_f, riser_f, relief_f, breg_f, bcell_f), face_keys, refine_stats = refine_contacts(
            V, F, [rgb, cover, state, dvec, wall_cm['taper']], [cover_faces, cop_f, nout_f, riser_f, relief_f, breg_f, bcell_f], truth, levels=3, known=q_v,
            min_edge=float(blockface.get('refine_min_edge_m', 1e-5)))
        wall_cm['taper'] = taper_w
        sel = np.nonzero(cop_f > 0)[0]
        if len(sel):
            st_c, b_c, u_c, l_c = truth.query(V[F[sel]].mean(1) - 1e-3 * nout_f[sel])
            for j, fi in enumerate(sel):
                face_keys[fi] = (b_c[j], u_c[j], l_c[j]) if st_c[j] == 'occupied' else None
    elif wall is not None:
        V, F, (rgb, cover, state, dvec, taper_w), (cover_faces, cop_f, nout_f, riser_f, relief_f, breg_f, bcell_f), face_keys, refine_stats = refine_contacts(
            V, F, [rgb, cover, state, dvec, wall['taper']], [cover_faces, cop_f, nout_f, riser_f, relief_f, breg_f, bcell_f], truth, levels=3, known=q_v,
            min_edge=float(blockface.get('refine_min_edge_m', 1e-5)))
        wall['taper'] = taper_w
        sel = np.nonzero(cop_f > 0)[0]
        if len(sel):
            st_c, b_c, u_c, l_c = truth.query(V[F[sel]].mean(1) - 1e-3 * nout_f[sel])
            for j, fi in enumerate(sel):
                face_keys[fi] = (b_c[j], u_c[j], l_c[j]) if st_c[j] == 'occupied' else None
    else:
        if dvec is not None:
            V, F, (rgb, cover, state, dvec), (cover_faces, cop_f, nout_f, riser_f, relief_f), face_keys, refine_stats = refine_contacts(
                V, F, [rgb, cover, state, dvec], [cover_faces, cop_f, nout_f, riser_f, relief_f], truth, levels=3, known=q_v)
        else:
            V, F, (rgb, cover, state), (cover_faces, cop_f, nout_f, riser_f, relief_f), face_keys, refine_stats = refine_contacts(
                V, F, [rgb, cover, state], [cover_faces, cop_f, nout_f, riser_f, relief_f], truth, levels=3, known=q_v)
        # A riser in a contact plane has its vertices (and centroid) on the contact: its unit is the
        # kept rock behind it, 1 mm inside along the polygon's outward normal.
        sel = np.nonzero(cop_f > 0)[0]
        if len(sel):
            st_c, b_c, u_c, l_c = truth.query(V[F[sel]].mean(1) - 1e-3 * nout_f[sel])
            for j, fi in enumerate(sel):
                face_keys[fi] = (b_c[j], u_c[j], l_c[j]) if st_c[j] == 'occupied' else None
    _mark('contact_refine')
    riser_uv = None
    if cop_f is not None and blockface.get('riser_uv', 'world') == 'plane':
        from .blockface import riser_plane_uv
        riser_uv = riser_plane_uv(V, F, riser_f, nout_f)
        blockface_info['riser_uv'] = {'mode': 'plane', 'riser_faces': int(riser_f.sum()),
                                      'riser_vertices': int(np.isfinite(riser_uv[:, 0]).sum())}
    xfer = None
    if wall is not None:
        # whole-wall scene: atlas UV of the three colour classes, class weights, total weight, reference positions, fill codes
        from . import wallscene as wsc
        rock_f = np.array([k is not None for k in face_keys]) & ~cover_faces
        xa = wsc.xfer_attributes_wall(V, F, riser_f, nout_f, rock_f, wall['taper'], wall['lay'], wall['feet'])
        xfer = {'faces': xa['faces'], 'uv': xa['uv3'][:, 0], 'uv3': xa['uv3'], 'class_w': xa['class_w'], 'w': xa['w'], 'ref': V - dvec}
        dn_ = np.linalg.norm(dvec[np.unique(F[xa['faces']])], axis=1) if xa['faces'].any() else np.zeros(1)
        mat_opts = blockface['detail'].get('material') or {}
        xfer['fill'], fstats = wsc.fill_codes_wall(V, F, xa['faces'], riser_f, -wall['lay'].m, wall['lay'], wall['feet'], wall['atlas_dir'] / 'mesh',
                                                   float(mat_opts.get('slope_deg', 60.)))
        blockface_info['detail'].update(xfer_faces=int(xa['faces'].sum()), xfer_riser_vertices=xa['riser_vertices'], pou_core_max_dev=xa['pou_core_max_dev'],
                                        core_vertices_without_footprint=xa['core_vertices_without_footprint'], material_fill=fstats,
                                        reference_uv={'source': 'final position minus the scan-detail displacement of all footprints (B spalling kept)',
                                                      'footprint_vertex_disp_mm_p50_p99_max': [float(x) for x in np.percentile(dn_ * 1e3, [50, 99, 100])]})
        blockface_info['detail']['material_fill']['bedding_normal'] = blockface_info['bedding_normal']
    if cop_f is not None and detail_frame is not None and blockface['detail'].get('textures', True):
        # donor texture coordinates and blend weight for the footprint faces (stage.write_display_glb)
        from .scandetail import xfer_attributes
        rock_f = np.array([k is not None for k in face_keys]) & ~cover_faces
        xfer = xfer_attributes(V, F, riser_f, nout_f, rock_f, detail_frame)
        blockface_info['detail'].update(xfer_faces=int(xfer['faces'].sum()), xfer_riser_planes=xfer['riser_planes'],
                                        xfer_riser_vertices=xfer['riser_vertices'])
        if dvec is not None:
            xfer['ref'] = V - dvec
            dn_ = np.linalg.norm(dvec[np.unique(F[xfer['faces']])], axis=1)
            blockface_info['detail']['reference_uv'] = {'source': 'final position minus the scan-detail displacement (B spalling kept)',
                                                        'footprint_vertex_disp_mm_p50_p99_max': [float(x) for x in np.percentile(dn_ * 1e3, [50, 99, 100])]}
        mat_opts = blockface['detail'].get('material')
        if mat_opts:
            from .xfermat import fill_codes
            xfer['fill'], fstats = fill_codes(V, F, xfer['faces'], riser_f, detail_frame, blockface['detail']['donor_mesh'],
                                              blockface['frame']['normal'], float(mat_opts.get('slope_deg', 60.)))
            beds = [c for c in bf_cut if c[0] == 'bed']
            nb_ = np.asarray(beds[0][2], float) if beds else np.array([0., 0., 1.])
            blockface_info['detail']['material_fill'] = {**fstats, 'bedding_normal': (nb_ if nb_[2] >= 0 else -nb_).tolist()}
    N = vertex_normals(V, F, np.array([0., 0., 1.]), consistent=True)
    if (N[:len(outward_v)] * outward_v).sum() < 0:
        N = -N
    unoccupied_rock = sum(1 for k, cf in zip(face_keys, cover_faces) if k is None and not cf)
    from .truth import surface_intersections
    # Rule 2: no line trace on faces lying in their interface plane (the intersection degenerates
    # into the face itself; those faces are labelled as interface faces instead).
    polylines = surface_intersections(V, F if cop_f is None else F[cop_f == 0], truth.interfaces)
    # A class whose traces run >= 1 m where grooves are expected but with no groove measured
    # cannot pass the alignment check vacuously. Scope: the 3D traces of the class on the final
    # surface (all of them, independent of any camera) where the planned groove amplitude of the
    # nearest pre-refinement vertex (within 15 cm) is >= 2 cm.
    from .semantics import nearest_vertex_fast
    owners = {'fault': faults, 'joint': joints, 'contact': intrusions}
    for cls, own in owners.items():
        segs = [pts for iid, pts in polylines if iid.split(':', 1)[0] in own and len(pts) > 1]
        length = 0.
        if segs:
            pts = np.concatenate(segs)
            k, dist = nearest_vertex_fast(pts, V_pre, cell=.1)
            pot = np.where((k >= 0) & (dist < .15), groove_potential[cls][np.maximum(k, 0)], 0.)
            o = 0
            for sgm in segs:
                ok = pot[o:o + len(sgm)] >= .02
                step = np.linalg.norm(np.diff(sgm, axis=0), axis=1)
                length += float(step[ok[1:] & ok[:-1]].sum())
                o += len(sgm)
        alignment[cls]['grooved_trace_m'] = length
        if alignment[cls]['grooves'] == 0 and length >= 1.:
            alignment[cls]['ok'] = False
            alignment[cls]['unmeasured'] = True

    if exact is not None:
        # blocks-3: own vertex copies and normals on both sides of every creased edge (sharp shading).
        from .fracture import crease_normals, CREASE_DEG
        n_before = len(V)
        V, F, N, ca, _ = crease_normals(V, F, N, {'rgb': rgb, 'cover': cover, 'state': state})
        rgb, cover, state = ca['rgb'], ca['cover'], ca['state']
        fracture_info.update(crease_deg=CREASE_DEG, crease_vertex_copies=int(len(V) - n_before))

    # 7. non-rock items (ground, ...), each with its semantic class and provenance.
    items = [it(rng, seed, soil_rgb) if callable(it) else it for it in surface.items]
    stats = {'exposure': surface.kind, 'patches': [p.id for p in surface.patches], 'seams': W['seams'],
             'vertices': int(len(V)), 'faces': int(len(F)), 'items': {it.id: int(len(it.vertices)) for it in items},
             'contact_refinement': refine_stats,
             'applied_recession_m': {'median': float(np.median(applied)), 'p95': float(np.percentile(applied, 95)),
                                     'max': float(applied.max())},
             'outward_relief_m': float(np.percentile(-rough, 99)),
             'above_ground_unoccupied_rock_faces': int(unoccupied_rock), 'fold_guard_passes': folds,
             'fold_guard_final_flipped': folds_final, 'clay': bool(clay),
             'groove_alignment': alignment}
    if fracture_info:
        stats['fracture'] = fracture_info
    if blockface_info:
        stats['blockface'] = blockface_info
    _mark('polylines_end')
    if cm_info is not None:
        cm_info['timing_s'] = _marks
        cm_info['vertices_before_contact_refine'] = int(_nv_before_contacts)
        cm_info['rock_vertices_final'] = int(len(V))
        stats['cm'] = cm_info
    Vs, Fs, Ns, Cs, cov, cf, keys, sem = [V], [F], [N], [rgb], [cover], [cover_faces], list(face_keys), \
        [np.where(cover_faces, 'soil', 'bedrock').astype(object)]
    offset = len(V)
    for it in items:
        Vs.append(it.vertices)
        Fs.append(it.faces + offset)
        Ns.append(it.normals if it.normals is not None else np.tile([0., 0., 1.], (len(it.vertices), 1)))
        Cs.append(it.colors if it.colors is not None else np.tile(soil_rgb, (len(it.vertices), 1)))
        cov.append(np.ones(len(it.vertices)))
        cf.append(np.ones(len(it.faces), bool))
        keys += [None] * len(it.faces)
        sem.append(np.full(len(it.faces), it.semantic, object))
        offset += len(it.vertices)
    extra_out = {}
    if cop_f is not None:
        extra_out = {'coplanar_faces': np.concatenate([cop_f] + [np.zeros(len(it.faces), np.int64) for it in items]),
                     'coplanar_records': cop_records}
    if riser_uv is not None:
        # B riser texture coordinates in their own plane (stage.write_display_glb); NaN = default mapping
        extra_out['riser_faces'] = np.concatenate([riser_f] + [np.zeros(len(it.faces), bool) for it in items])
        # (was written only together with relief_faces, which dropped the riser mapping of a B
        extra_out['riser_uv'] = np.concatenate([riser_uv] + [np.full((len(it.vertices), 2), np.nan) for it in items])
    if wall_cm is not None:
        # whole-wall cm route: B region / cell per face (wallcheck), detail displacement per vertex (X3; zero without relief)
        extra_out['b_region'] = np.concatenate([breg_f] + [np.full(len(it.faces), -1, np.int64) for it in items])
        extra_out['b_cell'] = np.concatenate([bcell_f] + [np.full(len(it.faces), -1, np.int64) for it in items])
        extra_out['relief_dvec'] = np.concatenate([dvec] + [np.zeros((len(it.vertices), 3)) for it in items])
    if cop_f is not None and relief_f.any():
        # B floor relief faces (relief.py), for the geometry checks
        extra_out['relief_faces'] = np.concatenate([relief_f] + [np.zeros(len(it.faces), bool) for it in items])
    if xfer is not None:
        # B scan-detail footprint (scandetail.py): faces using the donor textures, per-vertex donor UV and weight
        extra_out['xfer_faces'] = np.concatenate([xfer['faces']] + [np.zeros(len(it.faces), bool) for it in items])
        extra_out['xfer_uv'] = np.concatenate([xfer['uv']] + [np.zeros((len(it.vertices), 2)) for it in items])
        extra_out['xfer_w'] = np.concatenate([xfer['w']] + [np.zeros(len(it.vertices)) for it in items])
        if 'ref' in xfer:
            extra_out['xfer_ref'] = np.concatenate([xfer['ref']] + [np.asarray(it.vertices, float) for it in items])
        if 'fill' in xfer:
            extra_out['xfer_fill'] = np.concatenate([xfer['fill']] + [np.zeros(len(it.faces), np.float32) for it in items])
        if wall is not None:
            extra_out['b_region'] = np.concatenate([breg_f] + [np.full(len(it.faces), -1, np.int64) for it in items])
            extra_out['b_cell'] = np.concatenate([bcell_f] + [np.full(len(it.faces), -1, np.int64) for it in items])
        if 'uv3' in xfer:
            # whole-wall scene: atlas UV per colour class and the classes' relative weights (stage.write_display_glb)
            extra_out['xfer_uv3'] = np.concatenate([xfer['uv3']] + [np.zeros((len(it.vertices), 3, 2)) for it in items])
            extra_out['xfer_cw'] = np.concatenate([xfer['class_w']] + [np.zeros((len(it.vertices), 3)) for it in items])
    if shape_exceed:
        extra_out['cm_shape_exceed'] = shape_exceed
    return dict(**extra_out, vertices=np.vstack(Vs), faces=np.vstack(Fs), normals=np.vstack(Ns), colors=np.clip(np.vstack(Cs), .02, .95),
                cover=np.concatenate(cov), cover_faces=np.concatenate(cf), face_keys=keys, semantic=np.concatenate(sem),
                rock_vertices=len(V), state=state, polylines=polylines, stats=stats,
                provenance={'items': {it.id: it.provenance for it in items}, 'profile': profile,
                            **({'fracture': fracture_info} if fracture_info else {})})
