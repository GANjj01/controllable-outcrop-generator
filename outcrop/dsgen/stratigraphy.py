"""Shared depositional coordinates and erosion selectors for v2 declarations.

These helpers use existing grid/band/CSG geometry. One coordinate is shared by every band in a succession, so variable
thickness does not create mismatched contacts between independently generated units."""
import math

import numpy as np

from outcrop.modeling.errors import invalid
from outcrop.modeling.kernel.grid import GridField


def _finite(value, shape, field):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        invalid(field, 'finite numeric values required')
    if result.shape != shape or not np.isfinite(result).all():
        invalid(field, 'finite numeric values of shape %s required' % (shape,))
    return result


def nonparallel_field(oid, bounds, *, base_m, total_thickness_m, relief_m,
                      thickness_variation, wavelengths_m, phase_rad=(0., 0.),
                      shape=(3, 3, 2), dip_deg=0., dip_direction_deg=0.):
    """Return one dimensionless grid declaration for a complete succession.

    ``bounds`` is ``(minimum_xyz, maximum_xyz)`` (or a kernel AABB), and ``shape`` gives node counts. In coordinates
    relative to the southwest bounds corner, let ``u=2*pi*x/wavelength_x+phase_x`` and similarly ``v``. Sample
    ``phi=(z-B)/H``, where::

    B = base - tan(dip)*(x*sin(azimuth) + y*cos(azimuth)) + relief*(sin(u) + sin(v))/2 H = total_thickness*(1 +
    thickness_variation*sin(u)*cos(v))

    Dip is in [0,90) degrees, with ENU azimuth clockwise from north. Thickness is vertical, and
    ``abs(thickness_variation)<1`` makes analytic H positive. Bands use this same field with unit ``'1'`` and shared
    level endpoints."""
    if not isinstance(oid, str) or not oid:
        invalid('stratigraphy.id', 'nonempty field ID required')
    if hasattr(bounds, 'minimum') and hasattr(bounds, 'maximum'):
        bounds = (bounds.minimum, bounds.maximum)
    box = _finite(bounds, (2, 3), oid + '.bounds')
    if np.any(box[0] >= box[1]):
        invalid(oid + '.bounds', 'positive extents required')
    counts = np.asarray(shape)
    if counts.shape != (3,) or counts.dtype.kind not in 'iu' or np.any(counts < 2):
        invalid(oid + '.shape', 'three integer node counts of at least two required')
    base, total, relief, variation, dip, azimuth = (
        float(_finite(value, (), oid + '.' + name))
        for name, value in (('base_m', base_m), ('total_thickness_m', total_thickness_m),
                            ('relief_m', relief_m), ('thickness_variation', thickness_variation),
                            ('dip_deg', dip_deg), ('dip_direction_deg', dip_direction_deg)))
    if total <= 0 or relief < 0 or abs(variation) >= 1:
        invalid(oid, 'positive thickness, nonnegative relief and abs(thickness_variation)<1 required')
    if not 0 <= dip < 90:
        invalid(oid + '.dip_deg', 'dip in [0,90) degrees required')
    wavelengths = _finite(wavelengths_m, (2,), oid + '.wavelengths_m')
    phase = _finite(phase_rad, (2,), oid + '.phase_rad')
    if np.any(wavelengths <= 0):
        invalid(oid + '.wavelengths_m', 'positive wavelengths required')

    # Use exactly the origin/spacing arithmetic used by GridField, including
    # its represented node positions, rather than a separately rounded grid.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        spacing = (box[1] - box[0]) / (counts - 1)
        axes = [box[0, j] + np.arange(int(counts[j])) * spacing[j] for j in range(3)]
        if not np.isfinite(spacing).all() or any(not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in axes):
            invalid(oid, 'finite representable grid spacing required')
        x, y = np.meshgrid(axes[0] - box[0, 0], axes[1] - box[0, 1], indexing='ij')
        u = 2 * math.pi * x / wavelengths[0] + phase[0]
        v = 2 * math.pi * y / wavelengths[1] + phase[1]
        angle = math.radians(azimuth % 360)
        base_surface = (base - math.tan(math.radians(dip)) * (x * math.sin(angle) + y * math.cos(angle))
                        + relief * (np.sin(u) + np.sin(v)) / 2)
        thickness = total * (1 + variation * np.sin(u) * np.cos(v))
        if not np.isfinite(thickness).all() or np.any(thickness <= 0):
            invalid(oid, 'finite positive local thickness required')
        values = (axes[2][None, None, :] - base_surface[:, :, None]) / thickness[:, :, None]
    if not np.isfinite(values).all():
        invalid(oid, 'finite stratigraphic field values required')

    field = GridField(oid, box[0], spacing, values, '1')
    complex_ = field.cell_complex()
    vertices = complex_.vertices[complex_.cells]
    node_values = values.ravel()[complex_.cells]
    # Solving the affine gradient on every source tetra checks the declaration
    # the compiler actually consumes, including cancellation at large offsets.
    try:
        gradients = np.linalg.solve(vertices[:, 1:] - vertices[:, :1],
                                    (node_values[:, 1:] - node_values[:, :1])[..., None])[..., 0]
    except np.linalg.LinAlgError:
        invalid(oid, 'nondegenerate P1 tetrahedra required')
    if not np.isfinite(gradients).all() or np.any(gradients[:, 2] <= 0):
        invalid(oid, 'finite positive P1 vertical gradient required')
    return {'id': oid, 'kind': 'grid', 'origin_m': box[0].tolist(),
            'spacing_m': spacing.tolist(), 'shape': counts.tolist(),
            'values': values.tolist(), 'unit': '1'}


def unconformity_regions(oid, normal, offset_m):
    """Return ``oid:old`` / ``oid:young`` complementary erosion halfspaces.

    The supplied equation is ``normal.dot(point)-offset_m=0``. Old rock is
    below/on its negative side and younger rock is on its positive side.
    Normalize that equation once and negate the same coefficients, giving
    disjoint interiors and one shared boundary without independently fitted
    erosion surfaces. Intersect each selector with its own succession/domain.
    """
    if not isinstance(oid, str) or not oid:
        invalid('unconformity.id', 'nonempty selector ID required')
    vector = _finite(normal, (3,), oid + '.normal')
    offset = float(_finite(offset_m, (), oid + '.offset_m'))
    norm = math.hypot(*vector)
    if norm == 0 or not math.isfinite(norm):
        invalid(oid + '.normal', 'nonzero finite normal magnitude required')
    coefficients = np.r_[vector / norm, offset / norm]
    if not np.isfinite(coefficients).all():
        invalid(oid, 'finite normalized plane required')
    return [{'id': oid + suffix, 'kind': 'halfspace',
             'normal': (sign * coefficients[:3]).tolist(),
             'offset_m': float(sign * coefficients[3])}
            for suffix, sign in ((':old', 1.), (':young', -1.))]
