"""Check an existing difference and write its provenance report; no Boolean work.

Usage: python check_host_provenance.py CASE_NAME
Only cases/CASE_NAME/host_provenance_check.json is written. These numerical
checks certify retained host-face provenance, not network or visible-rim
qualification. Cutter roles are looked up, never inferred from shading.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from source_tables import load_sources  # noqa: E402  path_XX_sources as json or npz
TOLERANCES = dict(plane_distance_m=1e-8, barycentric=1e-8,
                  normal_cosine_error=1e-10, source_face_area_gain_m2=1e-10)


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def mesh_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    vertices, faces = arrays['vertices'], arrays['faces']
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError(f'{path.name}: vertices must be finite nonempty (N,3)')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0
            or faces.max() >= len(vertices)):
        raise ValueError(f'{path.name}: invalid triangle vertex indices')
    return arrays


def check_case(directory):
    build = read_json(directory / 'build.json')
    source_path = Path(build['source']) / 'source_surface.npz'
    source, result = mesh_arrays(source_path), mesh_arrays(directory / 'difference.npz')
    vertices, faces = result['vertices'], result['faces']
    original_ids, source_ids = result['original_id'], result['source_face_id']
    for name, values in (('original_id', original_ids), ('source_face_id', source_ids)):
        if values.shape != (len(faces),) or values.dtype.kind not in 'iu' or values.min() < 0:
            raise ValueError(f'{name}: nonnegative integer identity required for every output face')
    host = original_ids == build['host_original_id']
    if not host.any() or source_ids[host].max() >= len(source['faces']):
        raise ValueError('host source-face IDs are empty or outside the original host')

    errors, role_coverage = [], []
    known_ids = {int(build['host_original_id'])}
    for path in build['paths']:
        record = load_sources(directory, path['index'])
        oid = int(path['original_id'])
        if oid in known_ids or int(record['original_id']) != oid:
            raise ValueError('cutter original_id is duplicated or disagrees with its source table')
        known_ids.add(oid)
        selected_ids = source_ids[original_ids == oid]
        table = record['face_sources']
        missing = sorted(set(map(int, selected_ids)) - set(map(int, table)))
        counts = {}
        for fid in selected_ids:
            row = table.get(str(int(fid)))
            if row is not None:
                role = row.get('role')
                if not isinstance(role, str) or not role or 'source_segment_id' not in row:
                    raise ValueError('cutter source row lacks a role or source segment identity')
                counts[role] = counts.get(role, 0) + 1
        role_coverage.append(dict(path_index=path['index'], original_id=oid,
            output_faces=len(selected_ids), distinct_source_faces=len(np.unique(selected_ids)),
            missing_source_face_ids=missing, output_face_roles=counts))
        if missing:
            errors.append(f'cutter {oid} has unmapped source faces')
    unknown = sorted(set(map(int, original_ids)) - known_ids)
    if unknown:
        errors.append('unknown original mesh IDs')

    source_triangles = source['vertices'][source['faces']]
    tri = vertices[faces[host]]
    reference = source_triangles[source_ids[host]]
    e0, e1 = reference[:, 1] - reference[:, 0], reference[:, 2] - reference[:, 0]
    normal = np.cross(e0, e1)
    normal_length = np.linalg.norm(normal, axis=1)
    actual_cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    actual_area = np.linalg.norm(actual_cross, axis=1) / 2
    if np.any(normal_length <= 0) or np.any(actual_area <= 0):
        raise ValueError('degenerate host source or retained triangle')
    relative = tri - reference[:, 0, None]
    plane = np.abs(np.einsum('fvi,fi->fv', relative, normal) / normal_length[:, None])
    d00, d01, d11 = (np.einsum('fi,fi->f', a, b) for a, b in ((e0, e0), (e0, e1), (e1, e1)))
    d20 = np.einsum('fvi,fi->fv', relative, e0)
    d21 = np.einsum('fvi,fi->fv', relative, e1)
    denominator = d00 * d11 - d01 * d01
    if np.any(denominator <= 0):
        raise ValueError('host source triangle has no stable barycentric denominator')
    u = (d11[:, None] * d20 - d01[:, None] * d21) / denominator[:, None]
    v = (d00[:, None] * d21 - d01[:, None] * d20) / denominator[:, None]
    bary = np.stack((1 - u - v, u, v), axis=2)
    cosine = np.einsum('fi,fi->f', actual_cross, normal) / (2 * actual_area * normal_length)
    source_area = np.linalg.norm(np.cross(source_triangles[:, 1] - source_triangles[:, 0],
                                         source_triangles[:, 2] - source_triangles[:, 0]), axis=1) / 2
    retained_area = np.bincount(source_ids[host], weights=actual_area, minlength=len(source_area))
    exact = np.zeros(len(tri), bool)
    for shift in range(3):
        exact |= np.all(tri == np.roll(reference, shift, axis=1), axis=(1, 2))
    gates = dict(host_vertices_coplanar=bool(plane.max() <= TOLERANCES['plane_distance_m']),
        host_vertices_inside_source_face=bool(bary.min() >= -TOLERANCES['barycentric']
                                             and bary.max() <= 1 + TOLERANCES['barycentric']),
        host_face_orientation_preserved=bool(cosine.min() >= 1 - TOLERANCES['normal_cosine_error']),
        per_source_face_area_not_increased=bool((retained_area - source_area).max()
                                               <= TOLERANCES['source_face_area_gain_m2']))
    errors.extend(name for name, passed in gates.items() if not passed)
    return dict(status='PASS' if not errors else 'FAIL', errors=errors, source=str(source_path),
        scope='retained host-face provenance and cutter role lookup; not topology, visible rims or E_syn',
        numerical_tolerances=TOLERANCES, gates=gates,
        input_vertices_finite_and_indices_in_range=True, host_source_face_ids_in_range=True,
        unknown_original_ids=unknown, source_host_faces=len(source['faces']),
        output_faces=len(faces), output_host_faces=int(host.sum()),
        exact_unchanged_oriented_host_triangles=int(exact.sum()),
        host_fragments_not_original_full_triangle=int((~exact).sum()),
        source_faces_with_fragments=int(len(np.unique(source_ids[host][~exact]))),
        max_plane_distance_m=float(plane.max()), min_barycentric=float(bary.min()),
        max_barycentric=float(bary.max()), minimum_normal_cosine=float(cosine.min()),
        max_area_gain_per_source_m2=float((retained_area - source_area).max()),
        retained_host_surface_area_loss_m2=float((source_area - retained_area).sum()),
        cutter_role_coverage=role_coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='existing case directory name')
    args = parser.parse_args()
    directory = ROOT / 'cases' / args.name
    try:
        report = check_case(directory)
    except (ValueError, KeyError) as error:
        report = dict(status='FAIL', errors=[str(error)], numerical_tolerances=TOLERANCES)
    target = directory / 'host_provenance_check.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(status=report['status'], report=str(target), errors=report['errors'])))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
