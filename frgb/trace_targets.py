"""Trace training targets from a 1 px centreline raster.

Softness depends only on the distance to the centreline, never on image content. Shape and radii are fixed here before
any training: d = Euclidean distance (px) to the nearest centreline pixel; synthetic: s = 1 - d/2 for d < 2, else 0
(negative); real:      s = 1 - d/2 for d < 2, ignore for 2 <= d <= 4, negative for d > 4. On the integer grid the
nonzero values are 1, 0.5 and 1 - sqrt(2)/2. Ignore masks of the frame (U, transitions, scene) are applied after the
distance transform, so they never move the centreline. The 2-4 px ring is training-only; evaluation uses the frame's
own valid mask (frgb.metrics.evaluate_trace)."""
import numpy as np
from scipy import ndimage as ndi

SOFT_RADIUS_PX = 2.
REAL_IGNORE_OUTER_PX = 4.


def soft_value(distance):
    d = np.asarray(distance, float)
    return np.where(d < SOFT_RADIUS_PX, 1. - d / SOFT_RADIUS_PX, 0.)


def trace_training_target(centreline, valid=None, *, real):
    """centreline: 2-D bool/0-1 raster (1 px lines). valid: frame valid mask (default all). real: bool, required.

    Returns float32 target in [0,1], bool valid (frame valid minus the real-data ring), distance and the ring mask.
    """
    if not isinstance(real, bool):
        raise ValueError('real must be given explicitly as True or False')
    c = np.asarray(centreline)
    if c.ndim != 2 or not np.isin(c, (0, 1)).all():
        raise ValueError('centreline must be a 2D binary raster')
    c = c.astype(bool)
    v = np.ones(c.shape, bool) if valid is None else np.asarray(valid)
    if v.shape != c.shape or v.dtype != bool:
        raise ValueError('valid must be a bool mask on the centreline grid')
    distance = ndi.distance_transform_edt(~c) if c.any() else np.full(c.shape, np.inf)
    target = soft_value(distance).astype(np.float32)
    ring = np.zeros(c.shape, bool)
    if real:
        ring = (distance >= SOFT_RADIUS_PX) & (distance <= REAL_IGNORE_OUTER_PX)
    out_valid = v & ~ring
    target = np.where(out_valid, target, np.float32(0))
    return dict(target=target.astype(np.float32), valid=out_valid, distance=distance, ignore_ring=ring & v,
                soft_pixels=int((out_valid & (target > 0)).sum()),
                rule=dict(shape='linear 1-d/2', soft_radius_px=SOFT_RADIUS_PX,
                          real_ignore_ring_px=[SOFT_RADIUS_PX, REAL_IGNORE_OUTER_PX] if real else None))
