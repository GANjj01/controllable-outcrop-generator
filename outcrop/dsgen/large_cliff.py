"""A complete three-dimensional cliff terrain, independent of geological labels.

The returned open, upward-oriented PL surface is intended to be cut against
the final geological volume. Its computational rectangle ends in level ground;
the cliff itself fades to natural slopes before reaching either end.
"""
import math

import numpy as np


def _smooth(value):
    t = np.clip(value, 0., 1.)
    return t ** 3 * (10. + t * (-15. + 6. * t))


def cliff_surface(length_m=150., height_m=25., spacing_m=1., *, face_dip_deg=None):
    """Return ``vertices`` (ENU metres) and ``faces`` for the complete terrain.

    ``length_m`` is the span of the cliff, including its smooth end slopes.
    Foreground, plateau and back slope share one connected height-field mesh.
    ``spacing_m`` controls nominal sampling on the face and along strike;
    level ground and the gentle back slope use coarser rows. The mesh itself
    defines the PL terrain; no geological boundaries or textures are painted.
    """
    values = []
    for name, value in (('length_m', length_m), ('height_m', height_m),
                        ('spacing_m', spacing_m)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + ' must be positive and finite')
        value = float(value)
        if not np.isfinite(value) or value <= 0.:
            raise ValueError(name + ' must be positive and finite')
        values.append(value)
    length, height, spacing = values
    sx, sy = length / 150., height / 25.
    # Nominal dip is the mean toe-to-crest chord. Local slopes vary on the
    # rounded, irregular face. Keep vertical relief independent of this run.
    profile_scale = sy
    if face_dip_deg is not None:
        if isinstance(face_dip_deg, (bool, np.bool_)):
            raise ValueError('face_dip_deg must be finite and strictly between 0 and 90')
        face_dip_deg = float(face_dip_deg)
        if not np.isfinite(face_dip_deg) or not 0. < face_dip_deg < 90.:
            raise ValueError('face_dip_deg must be finite and strictly between 0 and 90')
        profile_scale = height / (14. * math.tan(math.radians(face_dip_deg)))
    half_domain = length * 2. / 3.
    x = np.linspace(-half_domain, half_domain,
                    math.ceil(2. * half_domain / spacing) + 1)
    # Keep most rows on the actual face, rather than oversampling flat ground.
    y_parts = []
    for start, end, multiplier in ((-55., -14., 3.), (-14., 16., 1.),
                                   (16., 28., 2.), (28., 50., 1.5),
                                   (50., 55., 3.)):
        count = math.ceil((end - start) * profile_scale / (spacing * multiplier))
        y_parts.append(np.linspace(start * profile_scale, end * profile_scale, count + 1)[:-1])
    y = np.r_[np.concatenate(y_parts), 55. * profile_scale]
    X, Y = np.meshgrid(x, y)
    u, w = X / sx, Y / profile_scale
    toe = -6. + 1.65 * np.sin(2. * np.pi * u / 43.) + .55 * np.sin(2. * np.pi * u / 18. + .4)
    crest = 8. + .9 * np.sin(2. * np.pi * u / 59. + .7) + .5 * np.sin(2. * np.pi * u / 24.)
    across_face = np.clip((w - toe) / (crest - toe), 0., 1.)
    rise = _smooth(across_face)
    end_fade = 1. - _smooth((np.abs(u) - 51.) / 24.)
    back_fade = 1. - _smooth((w - 28.) / 22.)
    local_height = height * (.98 + .02 * np.cos(2. * np.pi * u / 73. + .2))
    # Bounded metre-scale irregularity, zero on the toe and crest, avoids a
    # planar extrusion without imposing erosion scars or a fine noise field.
    face_relief = (.35 * sy * np.sin(2. * np.pi * u / 17.)
                   * np.sin(2. * np.pi * w / 13.)
                   * 4. * rise * (1. - rise))
    Z = 6. + end_fade * back_fade * (local_height * rise + face_relief)
    vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    rows, columns = X.shape
    origins = (np.arange(rows - 1)[:, None] * columns
               + np.arange(columns - 1)[None, :]).ravel()
    faces = np.stack([np.column_stack([origins, origins + 1, origins + columns]),
                      np.column_stack([origins + 1, origins + columns + 1,
                                       origins + columns])], axis=1).reshape(-1, 3)
    return {'vertices': vertices, 'faces': faces,
            'dimensions': {'cliff_length_m': length, 'target_relief_m': height,
                           'ground_z_m': 6., 'nominal_face_spacing_m': spacing,
                           'bounds_enu_m': [vertices.min(0).tolist(), vertices.max(0).tolist()]},
            'profile': {'construction': 'continuous 3D height surface with varying toe, crest and relief',
                        'end_transition_m': 24. * sx,
                        'toe_mean_y_m': -6. * profile_scale, 'crest_mean_y_m': 8. * profile_scale,
                        'nominal_face_dip_deg': (math.degrees(math.atan2(height, 14. * profile_scale))
                                                 if face_dip_deg is None else face_dip_deg),
                        'dip_definition': 'mean toe-to-crest chord; local curved-face slopes vary',
                        'back_slope_start_y_m': 28. * profile_scale,
                        'flat_back_y_m': 50. * profile_scale}}
