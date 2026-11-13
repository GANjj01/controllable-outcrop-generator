"""Native ModelSpec resolution and scene artifacts, without recipe overrides."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import struct
import time

import numpy as np

from outcrop.modeling.decode import decode_model, encode_model
from outcrop.modeling.kernel.mesh import MeshSurface
from outcrop.modeling.kernel.types import model_request
from outcrop.modeling.package import build_products, write_package
from outcrop.modeling.resolution import json_value, resolve_model
from outcrop.modeling.resources import load_resources
from outcrop.modeling.sampling import SamplingContext


def _resolve(config):
    geometry = config['geometry']
    raw = geometry.get('model_spec')
    if raw is None:
        raise ValueError('geometry.model_spec is required for native model validation')
    spec = decode_model(raw)
    if spec['resources'] and not geometry.get('resource_root'):
        raise ValueError('geometry.resource_root is required for declared resources')
    if spec['resources'] and not Path(geometry['resource_root']).is_absolute():
        raise ValueError('geometry.resource_root must be an absolute local resource base')
    resources = load_resources(spec, geometry.get('resource_root') or '.')
    sampling = spec.get('sampling', {})
    seed = sampling.get('root_seed', config['seed'])
    sample_id = geometry.get('model_sample_id')
    if sample_id is None:
        sample_id = next(iter(sampling.get('sample_ids', ())), 'sample')
    context = SamplingContext(seed, sample_id,
        range_policy=sampling.get('range_policy'))
    resolved = resolve_model(spec, context, capabilities=resources)
    requested = geometry.get('model_exposure')
    choices = [value for value in resolved.numeric['exposures']
               if requested is None or value['id'] == requested]
    if len(choices) != 1:
        raise ValueError('Select exactly one declared exposure with geometry.model_exposure')
    exposure = dict(choices[0])
    return spec, resolved, resources, exposure, model_request(resolved.numeric)


def _summary(resolved, exposure, request):
    return dict(success=True, model_id=resolved.numeric['model_id'],
        root_seed=resolved.root_seed, sample_id=resolved.sample_id,
        exposure=json_value(exposure), motions=json_value(resolved.motions),
        request=asdict(request), evidence=json_value(resolved.evidence),
        resources=sorted(resolved.resource_hashes))


def validate_declared_model(config):
    """Resolve resources, sampling, events and constraints without extraction."""
    _, resolved, _, exposure, request = _resolve(config)
    return dict(_summary(resolved, exposure, request),
                numeric_model=encode_model(resolved.numeric))


def _normals(surface):
    mesh = MeshSurface('declared-exposure', surface.vertices, surface.faces,
                       surface.exact_vertices, surface.position_error_m)
    vertices, faces = surface.vertices, surface.faces
    triangles = vertices[faces]
    area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], area)
    length = np.linalg.norm(normals, axis=1)
    # Opposing incident faces can cancel at a vertex. Choose an actual
    # incident face normal for shading without modifying native topology.
    for vertex in np.flatnonzero(length == 0):
        incident = np.flatnonzero(np.any(faces == vertex, axis=1))
        if not len(incident):
            raise ValueError('Declared exposure has an unused vertex')
        normals[vertex] = mesh.normals[incident[0]]
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def _boundary(surface, request):
    from .boundary import _physical_boundary, validate_boundary
    indexed, physical, nonmanifold = _physical_boundary(
        surface.vertices, surface.faces, request.tolerance.length_m)
    if nonmanifold:
        raise ValueError('Nonmanifold declared exposure boundary')
    if not len(physical):
        return validate_boundary(surface.vertices, surface.faces, contract='closed_rock',
                                 tolerance=request.tolerance.length_m)
    return dict(contract='declared_exposure', closed=False,
        physical_boundary_edges=len(physical), index_boundary_edges=len(indexed),
        t_junction_boundary_edges=max(0, len(indexed) - len(physical)),
        nonmanifold_edges=0, positive_triangle_areas=True,
        boundary_source='native declared exposure and occupied-region intersection',
        full_source_preservation_claimed=False)


def _book(products):
    surface, appearance = products.extraction.surface, products.appearance
    keys = list(zip(surface.body_ids, surface.unit_ids, surface.lithology_ids,
                    appearance.face_appearance_ids))
    ordered = sorted(set(keys), key=lambda key: tuple('' if value is None else value for value in key))
    book = []
    for code, (body, unit, lithology, appearance_id) in enumerate(ordered, 1):
        record = appearance.appearances.get(appearance_id, {})
        linear = np.asarray(record.get('color', [.65, .65, .65, 1.]), float)
        if len(linear) == 3:
            linear = np.r_[linear, 1.]
        srgb = np.where(linear[:3] <= .0031308, linear[:3] * 12.92,
                        1.055 * linear[:3] ** (1. / 2.4) - .055)
        book.append(dict(code=code, body=body, unit=unit, lithology=lithology,
            appearance=appearance_id, color_srgb=np.rint(255 * srgb).astype(int).tolist(),
            color_linear=linear.tolist(), roughness=record.get('roughness', .8),
            metallic=record.get('metallic', 0.)))
    lookup = {key: code for code, key in enumerate(ordered, 1)}
    return book, np.asarray([lookup[key] for key in keys], np.int32)


def _native_preview(mesh, codes, book, path):
    from .pipeline import _preview
    _preview(mesh, codes, book, path)
    data = path.read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    document = json.loads(data[20:20 + length])
    rows = {row['code']: row for row in book}
    for material in document['materials']:
        row = rows[int(material['name'][1:])]
        material['pbrMetallicRoughness'].update(baseColorFactor=row['color_linear'],
            roughnessFactor=row['roughness'], metallicFactor=row['metallic'])
    text = json.dumps(document, separators=(',', ':')).encode('utf8')
    text += b' ' * (-len(text) % 4)
    tail = data[20 + length:]
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, 20 + len(text) + len(tail))
                     + struct.pack('<II', len(text), 0x4E4F534A) + text + tail)


def generate_declared_scene(config, output):
    from .environment import build_scene
    from .pipeline import _json, _preview, _structure_preview
    started = time.monotonic()
    spec, resolved, resources, exposure, request = _resolve(config)
    products = build_products(resolved, exposure, request, resources)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_package(spec, resolved, products, output / 'package')
    surface = products.extraction.surface
    if not len(surface.faces):
        raise ValueError('Declared exposure has no visible occupied surface; native empty package retained')
    normals = _normals(surface)
    boundary = _boundary(surface, request)
    book, codes = _book(products)
    vertices, faces = surface.vertices, surface.faces
    encoded = lambda values: np.asarray(['' if value is None else value for value in values], str)
    np.savez_compressed(output / 'surface.npz', vertices=vertices, faces=faces,
        normals=normals, codes=codes, body_ids=encoded(surface.body_ids),
        unit_ids=encoded(surface.unit_ids), lithology_ids=encoded(surface.lithology_ids),
        body_valid=np.asarray([value is not None for value in surface.body_ids]),
        unit_valid=np.asarray([value is not None for value in surface.unit_ids]),
        lithology_valid=np.asarray([value is not None for value in surface.lithology_ids]),
        source_ids=np.asarray(surface.source_ids, str))
    for name in ('reference_surface', 'processed_surface', 'source_surface'):
        np.savez_compressed(output / (name + '.npz'), vertices=vertices, faces=faces, normals=normals)
    _json(output / 'declaration.json', encode_model(spec))
    _json(output / 'codebook.json', book)
    mesh = dict(vertices=vertices, faces=faces, normals=normals)
    _native_preview(mesh, codes, book, output / 'preview.glb')
    _preview(mesh, codes, book, output / 'gray.glb', gray=True)
    scene = build_scene(config, output, SimpleNamespace(items=[]), None, mesh, dict(kind='declared'))
    traces = products.extraction.traces
    np.savez_compressed(output / 'traces.npz', segments=traces.segments,
                        interface_ids=np.asarray(traces.interface_ids, str))
    _structure_preview(output / 'scene_preview.glb', output / 'structure_preview.glb',
                       traces.segments, traces.interface_ids)
    _structure_preview(output / 'preview.glb', output / 'structure_rock.glb',
                       traces.segments, traces.interface_ids)
    counts = {oid: traces.interface_ids.count(oid) for oid in sorted(set(traces.interface_ids))}
    features = {item['id']: item['kind'] for item in resolved.numeric['features'] + resolved.numeric['events']}
    trace_interfaces = [dict(id=item.id, feature_id=item.feature_id,
        kind=features.get(item.feature_id, 'interface'), segment_count=counts[item.id])
        for item in products.extraction.interfaces if item.id in counts]
    validation = dict(success=True, boundary=boundary,
        native_error_report=asdict(products.extraction.error_report),
        geological_labels='native ModelSpec extraction; authoritative nullable labels retained in package',
        full_source_preservation_claimed=False)
    _json(output / 'boundary_validation.json', validation)
    result = dict(vertices=len(vertices), faces=len(faces), pipeline_version=2,
        geometry_kind=boundary['contract'], boundary=boundary, validation=validation,
        boundary_validation='boundary_validation.json', resource_budget=asdict(request.budget),
        native_model=_summary(resolved, exposure, request),
        contact_tessellation='native declared extraction; no recipe or surface-process overrides',
        bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()],
        preview_glb='preview.glb', gray_glb='gray.glb', scene_glb='scene.glb',
        scene_preview_glb='scene_preview.glb', scene_gray_glb='scene_gray.glb',
        structure_preview_glb='structure_preview.glb', structure_rock_glb='structure_rock.glb',
        surface='surface.npz', codebook='codebook.json', declaration='declaration.json', package='package',
        trace_count=len(traces.segments), trace_interfaces=trace_interfaces,
        trace_legend=[dict(label='true geological interface traces', color_srgb=[255, 92, 32],
            source='package/traces.npz', interface_ids=sorted(counts))],
        exposure=exposure['kind'], source_vertices=len(vertices), source_faces=len(faces),
        layer_count=len({value for value in surface.unit_ids if value is not None}),
        exposed_units=sorted({value for value in surface.unit_ids if value is not None}),
        error_report=asdict(products.extraction.error_report), elapsed_s=time.monotonic() - started,
        scene=scene, geometry_frozen=True)
    return _json(output / 'geometry.json', result)
