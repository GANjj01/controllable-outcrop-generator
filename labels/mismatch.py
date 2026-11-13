"""Input: run_frames_r3.gate1(..)['_mismatch'] (case['collect_mismatch'] set) and the stage-A coverage keys of the
refined file. The band enters frgb.trace_labels.build_trace_labels(mismatch_region=); opening_class is built there."""
import numpy as np

BASE_PX = 2.             # M3: the soft band
FLAG_PX = 1.             # M2 a: |c| > 1 px at the frame's own scale (G1's px = |c| / (z / fpx))
RADIUS_STEP_PX = .125    # radii are rounded up to this step before the union of discs (only ever widens the band)
LIST_SHARE = .05         # M5: frames whose flagged length exceeds 5 % of the visible non-tip length are listed
SOURCES = ('near_fault', 'no_slot', 'junction', 'self_overlap', 'coverage_boundary', 'other')
COVER_KEYS = ('cover_L', 'cover_R', 'cover_paths', 'ring_change')


def flags(m):
    """M2 on G1's visible samples: (flagged, a = measured |c| > 1 px, b = no slot); tip samples never"""
    nt = ~np.asarray(m['tip'], bool)
    fin = np.isfinite(m['c'])
    a = nt & fin & (np.asarray(m['px'], float) > FLAG_PX)
    b = nt & ~fin
    return a | b, a, b


def offsets(m, cam, project):
    """d per sample: max(px, the screen offset of P + c n from the sample's uv) where c is finite, else nan"""
    c = np.asarray(m['c'], float)
    fin = np.isfinite(c)
    d = np.full(len(c), np.nan)
    if fin.any():
        uv_c, _ = project(m['point'][fin] + c[fin, None] * m['slot_normal'][fin], cam)
        d[fin] = np.maximum(np.asarray(m['px'], float)[fin], np.linalg.norm(uv_c - m['uv'][fin], axis=1))
    return d


def radii(m, a, b, d):
    """M3: R = 2 + d (a); R = 2 + 1/2 aperture / foot (b); nan for samples not flagged"""
    R = np.full(len(a), np.nan)
    R[a] = BASE_PX + d[a]
    R[b] = BASE_PX + .5 * np.asarray(m['aperture'], float)[b] / np.asarray(m['foot'], float)[b]
    return R


def band(uv, R, shape):
    """union over samples of the pixels within R of the sample's pixel (pixel-centre rounding and EDT as the other discs)"""
    from scipy import ndimage as ndi
    out = np.zeros(shape, bool)
    ok = np.isfinite(uv).all(1) & np.isfinite(R)
    if not ok.any():
        return out
    ij = np.floor(uv[ok] + .5).astype(np.int64)
    Rq = np.ceil(R[ok] / RADIUS_STEP_PX) * RADIUS_STEP_PX
    for r in np.unique(Rq):
        pad = int(np.ceil(r))            # a disc centred just outside the frame still reaches in: seed on a padded image
        big = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad), bool)
        q = ij[Rq == r] + pad
        okq = (q[:, 0] >= 0) & (q[:, 0] < big.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < big.shape[0])
        if not okq.any():
            continue
        big[q[okq, 1], q[okq, 0]] = True
        out |= (ndi.distance_transform_edt(~big) <= r)[pad:pad + shape[0], pad:pad + shape[1]]
    return out


def load_coverage(refined_npz):
    """the stage-A coverage arrays (refine rows), or None when the refined file has none"""
    if refined_npz is None:
        return None
    with np.load(refined_npz) as z:
        if not all(k in z.files for k in COVER_KEYS):
            return None
        return {k: z[k] for k in COVER_KEYS}


def sources(m, cov):
    """M4 (report only): per sample the exclusive source (first true in SOURCES order) and the non-exclusive matrix"""
    n = len(m['c'])
    row = np.asarray(m['row'], np.int64)
    known = np.zeros(n, bool)
    cl = cr = cp = np.full(n, -1, np.int64); ring = np.zeros(n, bool)
    if cov is not None and n:
        known = (row >= 0) & (row < len(cov['cover_L']))
        r_ = np.where(known, row, 0)
        cl, cr, cp = (np.where(known, np.asarray(cov[k])[r_], -1) for k in ('cover_L', 'cover_R', 'cover_paths'))
        ring = known & np.asarray(cov['ring_change'], bool)[r_]
        known &= (cl >= 0) & (cr >= 0) & (cp >= 0)
    many = np.maximum(cl, cr) >= 2
    mat = np.stack([np.asarray(m['ex_a'], bool) | np.asarray(m['ex_b'], bool), ~np.isfinite(m['c']), known & (cp >= 2),
                    known & (cp == 1) & many, known & ring, np.zeros(n, bool)], axis=1)
    mat[:, -1] = ~mat[:, :-1].any(1)
    first = np.argmax(mat, axis=1)
    return first, mat, known


def mismatch(m, cov, cam, shape, project):
    """(band M, report record) for one frame"""
    flagged, a, b = flags(m)
    d = offsets(m, cam, project)
    R = radii(m, a, b, d)
    M = band(m['uv'][flagged], R[flagged], shape)
    w = np.asarray(m['w'], float); nt = ~np.asarray(m['tip'], bool)
    first, mat, known = sources(m, cov)
    vis_nt = float(w[nt].sum())
    px_f = float(w[flagged].sum())
    q = lambda x: [float(v) for v in np.quantile(x, [.5, .99, 1.])] if len(x) else None
    paths = np.asarray(m['path'], np.int64)
    rec = dict(rule='criterion_trace_labels_mismatch_opening_r1 (793c9155) M5', visible_nontip_px=vis_nt, flagged_px=px_f,
               flagged_measured_px=float(w[a].sum()), flagged_no_slot_px=float(w[b].sum()),
               flagged_share=px_f / vis_nt if vis_nt else 0., listed_over_5pct=bool(vis_nt and px_f / vis_nt > LIST_SHARE),
               flagged_samples=int(flagged.sum()),
               by_source_exclusive_px={s: float(w[flagged & (first == k)].sum()) for k, s in enumerate(SOURCES)},
               by_source_nonexclusive_px={s: float(w[flagged & mat[:, k]].sum()) for k, s in enumerate(SOURCES)},
               coverage_unknown_flagged_px=float(w[flagged & ~known].sum()),
               by_path_px={str(int(p)): float(w[flagged & (paths == p)].sum()) for p in np.unique(paths[flagged])},
               px_p50_p99_max_measured=q(np.asarray(m['px'], float)[a]), d_p50_p99_max_measured=q(d[a]),
               radius_p50_p99_max=q(R[flagged]), band_px=int(M.sum()))
    return M, rec
