"""Scene-level background relief parameters of the multi-scene design.

The calibrated D1 spec is NOT refitted; only the per-group rms of its self-affine octaves is rescaled, so the random
waves (they depend on (seed, k) only, morph.self_affine_waves) are the same realization:

rms_k' = a * c(H) * rms_k * (lam_k / lam_ref) ** (H - H0)        lam_k = sqrt(lam_lo * lam_hi)

- a (amplitude): overall factor of every group; - H (Hurst): multiplicative tilt relative to the calibrated values
(the calibrated rms do not lie exactly on the H0 slope, so the scan residuals are kept); group 4 (extrapolated with
H0) is thereby re-extrapolated with H; - c(H): normalisation that keeps the total rms of the scan-measured groups
NORM_GROUPS (2.4-38.4 cm) unchanged, so that a alone sets the amplitude in the scan band and H alone the distribution
over scales (lam_ref then cancels). At (a, H) = (1, H0) the octaves are returned bit-for-bit (tilt = x ** 0.0 = 1.0, c
= sqrt(S / S) = 1.0, 1.0 * r = r). The macro facets (keep_facets, facets_c1) and the lithology term meso are
untouched: the flattest scene is not a plane. The cavity walls carry the same field: wall_spec() rescales the same
groups 0-3 with the same (a, H)."""
import hashlib
import json
from pathlib import Path

import numpy as np

H0 = 0.718
NORM_GROUPS = (0, 1, 2, 3)    # scan-measured groups (2.4-38.4 cm); group 4 (38.4-76.8 cm) is extrapolated
WALL_GROUPS = (0, 1, 2, 3)
LAM_REF = 0.096               # any value: cancelled by the normalisation (kept for the report only)


def scale_octaves(octaves, amplitude=1.0, hurst=H0, h0=H0, norm_groups=NORM_GROUPS):
    """octaves = [[lam_lo_m, lam_hi_m, rms_m, n_waves], ...] -> rescaled copy (same lam, same n)."""
    if not (amplitude > 0 and np.isfinite(amplitude)): raise ValueError('amplitude must be > 0')
    if not (0 < hurst < 1.5): raise ValueError('hurst out of range')
    lam = [float(np.sqrt(o[0] * o[1])) for o in octaves]
    tilt = [(l / LAM_REF) ** (float(hurst) - float(h0)) for l in lam]
    r1 = [float(o[2]) * t for o, t in zip(octaves, tilt)]
    s0 = sum(float(octaves[k][2]) ** 2 for k in norm_groups)
    s1 = sum(r1[k] ** 2 for k in norm_groups)
    c = float(np.sqrt(s0 / s1))
    return [[o[0], o[1], float(amplitude) * (c * r), o[3]] for o, r in zip(octaves, r1)]


def total_rms(octaves, groups=None):
    groups = range(len(octaves)) if groups is None else groups
    return float(np.sqrt(sum(float(octaves[k][2]) ** 2 for k in groups)))


def scaled_spec(base_spec_path, amplitude, hurst, out_path):
    """write a scene spec (same schema as self_affine_spec_C_r2.json) with the rescaled octaves."""
    base_spec_path = Path(base_spec_path)
    base = json.loads(base_spec_path.read_text(encoding='utf8'))
    spec = dict(base)
    spec['octaves'] = scale_octaves(base['octaves'], amplitude, hurst)
    spec.update(variant='%s_a%.4g_H%.4g' % (base.get('variant', 'base'), amplitude, hurst),
                status='multi-scene background relief (scene_relief.py)',
                scene_relief=dict(amplitude=float(amplitude), hurst=float(hurst), h0=H0, norm_groups=list(NORM_GROUPS),
                                  base_spec=str(base_spec_path), base_spec_sha256=hashlib.sha256(base_spec_path.read_bytes()).hexdigest(),
                                  total_rms_scan_band_m=[total_rms(base['octaves'], NORM_GROUPS), total_rms(spec['octaves'], NORM_GROUPS)],
                                  total_rms_all_groups_m=[total_rms(base['octaves']), total_rms(spec['octaves'])]))
    Path(out_path).write_text(json.dumps(spec, indent=1) + '\n', encoding='utf8')
    return spec


def wall_spec(scene_spec_path, field_seed, out_path, base_wall=None):
    """wall relief config from a scene spec: same octaves (already rescaled), groups 0-3, seed = the host field seed (erode
    scalars self_affine.seed of THIS scene)."""
    scene_spec_path = Path(scene_spec_path)
    spec = json.loads(scene_spec_path.read_text(encoding='utf8'))
    w = dict(json.loads(Path(base_wall).read_text(encoding='utf8'))) if base_wall else dict(schema='fracture-wall-relief-1')
    w.update(status='multi-scene: walls carry the scene host relief',
             source_spec=str(scene_spec_path), source_spec_sha256=hashlib.sha256(scene_spec_path.read_bytes()).hexdigest(),
             seed=int(field_seed), seed_source='erode scalars self_affine.seed of the scene', octaves=spec['octaves'],
             groups=list(WALL_GROUPS))
    Path(out_path).write_text(json.dumps(w, indent=1) + '\n', encoding='utf8')
    return w
