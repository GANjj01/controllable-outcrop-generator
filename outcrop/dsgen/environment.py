"""Complete local environment artifacts and their unchanged near-view subsets."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import struct

import numpy as np


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def _read_glb(path):
    data = Path(path).read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    return json.loads(data[20:20 + size]), bytearray(data[28 + size:])


def _save_glb(path, doc, blob):
    blob.extend(b'\0' * (-len(blob) % 4))
    doc['buffers'] = [{'byteLength': len(blob)}]
    text = json.dumps(doc, separators=(',', ':')).encode('utf8')
    text += b' ' * (-len(text) % 4)
    Path(path).write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(text) + len(blob))
        + struct.pack('<II', len(text), 0x4E4F534A) + text
        + struct.pack('<II', len(blob), 0x004E4942) + blob)


def combine_glbs(rock, scene, target, *, gray=False):
    """Append scene payloads; every original rock accessor and binary byte stays fixed."""
    doc, blob = _read_glb(rock)
    extra, extra_blob = _read_glb(scene)
    offset = len(blob)
    view_offset, accessor_offset = len(doc['bufferViews']), len(doc['accessors'])
    material_offset, mesh_offset, node_offset = len(doc['materials']), len(doc['meshes']), len(doc['nodes'])
    for view in extra['bufferViews']:
        view['byteOffset'] = view.get('byteOffset', 0) + offset
    for accessor in extra['accessors']:
        accessor['bufferView'] += view_offset
    for mesh in extra['meshes']:
        for primitive in mesh['primitives']:
            primitive['attributes'] = {key: value + accessor_offset for key, value in primitive['attributes'].items()}
            primitive['indices'] += accessor_offset
            primitive['material'] += material_offset
    for node in extra['nodes']:
        if 'mesh' in node:
            node['mesh'] += mesh_offset
        if 'children' in node:
            node['children'] = [value + node_offset for value in node['children']]
    if gray:
        for material in extra['materials']:
            material['pbrMetallicRoughness']['baseColorFactor'] = [.25, .25, .25, 1.]
        for mesh in extra['meshes']:
            for primitive in mesh['primitives']:
                primitive['attributes'].pop('COLOR_0', None)
    for key in ('bufferViews', 'accessors', 'materials', 'meshes', 'nodes'):
        doc[key].extend(extra[key])
    doc['scenes'][doc.get('scene', 0)]['nodes'].extend(
        value + node_offset for value in extra['scenes'][extra.get('scene', 0)]['nodes'])
    blob.extend(extra_blob)
    _save_glb(target, doc, blob)


def _normals(vertices, faces):
    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if not np.isfinite(area).all() or np.any(np.linalg.norm(area, axis=1) == 0):
        raise ValueError('Scene terrain contains degenerate or nonfinite triangles')
    for k in range(3):
        np.add.at(normals, faces[:, k], area)
    length = np.linalg.norm(normals, axis=1)
    normals[length == 0] = [0., 0., 1.]
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def _part(identifier, semantic, vertices, faces, normals=None, colors=None, provenance=None):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    if not len(faces):
        return None
    used, inverse = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[used], inverse.reshape(-1, 3)
    normals = _normals(vertices, faces) if normals is None else np.asarray(normals, float)[used]
    colors = np.tile([.23, .18, .1], (len(vertices), 1)) if colors is None else np.asarray(colors, float)[used, :3]
    if not np.isfinite(vertices).all() or not np.isfinite(normals).all() or not np.isfinite(colors).all():
        raise ValueError('Scene arrays must be finite')
    return dict(id=identifier, semantic=str(semantic), vertices=vertices, faces=faces, normals=normals,
                colors=np.clip(colors, 0., 1.), provenance=provenance or {})


def _export(parts, path):
    doc = dict(asset=dict(version='2.0', generator='outcrop-workbench-scene'), scene=0,
        scenes=[dict(nodes=[])], nodes=[], meshes=[], materials=[], accessors=[], bufferViews=[])
    blob = bytearray()
    def add(array, kind, component, target):
        blob.extend(b'\0' * (-len(blob) % 4))
        view = dict(buffer=0, byteOffset=len(blob), byteLength=array.nbytes, target=target)
        blob.extend(array.tobytes())
        doc['bufferViews'].append(view)
        accessor = dict(bufferView=len(doc['bufferViews']) - 1, componentType=component, count=len(array), type=kind)
        if kind == 'VEC3':
            accessor.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        doc['accessors'].append(accessor)
        return len(doc['accessors']) - 1
    for index, part in enumerate(parts):
        convert = lambda a: (a[:, [0, 2, 1]] * [1., 1., -1.]).astype('<f4')
        vertices = part['vertices']
        name = 'Scene_' + re.sub('[^A-Za-z0-9_]', '_', part['id'])
        roughness = .18 if part['semantic'] == 'water' else .95
        doc['materials'].append(dict(name='E%04d' % (index + 1), doubleSided=True,
            pbrMetallicRoughness=dict(baseColorFactor=[1., 1., 1., 1.], roughnessFactor=roughness, metallicFactor=0.)))
        attributes = dict(POSITION=add(convert(vertices), 'VEC3', 5126, 34962),
            NORMAL=add(convert(part['normals']), 'VEC3', 5126, 34962),
            TEXCOORD_0=add(vertices[:, :2].astype('<f4'), 'VEC2', 5126, 34962),
            COLOR_0=add(np.c_[part['colors'], np.ones(len(vertices))].astype('<f4'), 'VEC4', 5126, 34962))
        doc['meshes'].append(dict(name=name, primitives=[dict(attributes=attributes,
            indices=add(part['faces'].astype('<u4').ravel(), 'SCALAR', 5125, 34963), material=index)]))
        doc['nodes'].append(dict(name=name, mesh=index, extras=dict(item_id=part['id'],
            semantic=part['semantic'], geological_truth=False)))
        doc['scenes'][0]['nodes'].append(index)
    _save_glb(path, doc, blob)


def _shore_edge(part, along, inland, *, land):
    faces = part['faces']
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    edges = unique[counts == 1]
    uv = np.c_[part['vertices'] @ along, part['vertices'] @ inland]
    delta = uv[edges[:, 1]] - uv[edges[:, 0]]
    aligned = np.abs(delta[:, 0]) > 2 * np.abs(delta[:, 1])
    edges = edges[aligned]
    if not len(edges):
        return None
    mean_w = uv[edges, 1].mean(1)
    target = mean_w.min() if land else mean_w.max()
    ids = np.unique(edges[np.abs(mean_w - target) < 1.])
    ids = ids[np.argsort(uv[ids, 0])]
    return uv[ids, 0], part['vertices'][ids], part['colors'][ids]


def _sea_continuation(parts, exposure):
    """Close the native five-metre water/plain gap with the same mean sea plane."""
    if 'strike_deg' not in exposure:
        return None
    sea = next((part for part in parts if part['id'] == 'sea'), None)
    land = next((part for part in parts if part['id'] == 'far-ground'), None)
    if sea is None or land is None:
        return None
    angle = math.radians(exposure['strike_deg'])
    along = np.array([math.cos(angle), math.sin(angle), 0.])
    inland = np.array([-math.sin(angle), math.cos(angle), 0.])
    water_edge, land_edge = _shore_edge(sea, along, inland, land=False), _shore_edge(land, along, inland, land=True)
    if water_edge is None or land_edge is None:
        return None
    wu, wv, wc = water_edge
    lu, lv, _ = land_edge
    lower, upper = max(wu.min(), lu.min()), min(wu.max(), lu.max())
    coordinates = np.unique(np.r_[wu[(wu >= lower) & (wu <= upper)], lu[(lu >= lower) & (lu <= upper)], lower, upper])
    # Projecting the two rotated source grids can produce numerically duplicate
    # knots. Merge sub-micrometre duplicates before creating new float32 faces.
    coordinates = coordinates[np.r_[True, np.diff(coordinates) > 1e-6]]
    water = np.stack([np.interp(coordinates, wu, wv[:, i]) for i in range(3)], axis=1)
    shore = np.stack([np.interp(coordinates, lu, lv[:, i]) for i in range(3)], axis=1)
    gap = (shore - water) @ inland
    if gap.min() <= 0 or upper <= lower:
        return None
    shore[:, 2] = water[:, 2]
    shore += inland * 5.  # continue below the raised plain so oblique rays cannot see under its free edge
    vertices = np.vstack([water, shore])
    n = len(coordinates)
    a = np.arange(n - 1)
    faces = np.vstack([np.c_[a, a + 1, a + n + 1], np.c_[a, a + n + 1, a + n]])
    colors = np.stack([np.interp(coordinates, wu, wc[:, i]) for i in range(3)], axis=1)
    return _part('sea_continuation', 'water', vertices, faces, np.tile([0., 0., 1.], (len(vertices), 1)),
        np.vstack([colors, colors]), {'construction': 'native mean sea surface continued to and 5m under the existing raised coastal plain edge',
        'original_water_and_land_unchanged': True, 'minimum_gap_m': float(gap.min()), 'maximum_gap_m': float(gap.max()),
        'geological_truth': False})


def build_scene(config, output, exposure_surface, environment, rock_mesh, exposure):
    """Keep processed terrain plus native intrinsic Items; never alter the rock."""
    output = Path(output)
    parts, omitted = [], []
    color = np.asarray(config.get('scene', {}).get('ground_color', [110, 99, 74]), float) / 255.
    soil_rgb = np.where(color <= .04045, color / 12.92, ((color + .055) / 1.055) ** 2.4)
    if environment is not None and len(environment['faces']):
        semantic = np.asarray(environment.get('semantic', ['soil'] * len(environment['faces'])), str)
        identifiers = np.asarray(environment.get('item_ids', ['processed_terrain'] * len(semantic)), str)
        for identifier, label in sorted(set(zip(identifiers, semantic))):
            mask = (identifiers == identifier) & (semantic == label)
            part = _part(identifier + '_' + label, label, environment['vertices'], np.asarray(environment['faces'])[mask],
                environment.get('normals'), environment.get('colors'),
                {'construction': 'exact complementary faces from the same processed exposure surface', 'source_item': identifier})
            if part:
                if label in ('soil', 'sediment'):
                    brightness = part['colors'].mean(1)
                    variation = brightness / max(float(brightness.mean()), 1e-8)
                    part['colors'] = np.clip(soil_rgb * variation[:, None], 0., 1.)
                parts.append(part)
    rng = np.random.default_rng([int(config.get('seed', 0)), 983])
    configured = config.get('scene', {}).get('environment')
    if exposure['kind'] == 'bluff' and config.get('scene', {}).get('kind', 'bare') == 'bare' and configured is None:
        factories = []
    else:
        factories = exposure_surface.items
    for factory in factories:
        item = factory(rng, int(config.get('seed', 0)), soil_rgb) if callable(factory) else factory
        if item.id == 'beach':
            omitted.append(dict(id=item.id, reason='Native pre-process beach lens does not follow the final processed platform; retain the real coastal terrain and water instead.'))
            continue
        if item.semantic in ('vegetation', 'metal') or 'guardrail' in item.id.lower() or 'tree' in item.id.lower():
            omitted.append(dict(id=item.id, reason='Decoration is outside the current shared terrain scene.'))
            continue
        part = _part(item.id, item.semantic, item.vertices, item.faces, item.normals, item.colors, item.provenance)
        if part:
            parts.append(part)
    if exposure['kind'] == 'coast':
        continuation = _sea_continuation(parts, exposure)
        if continuation:
            parts.append(continuation)
    environment_record = None
    if configured is not None:
        from .env.scene import apply as apply_environment
        extras, environment_record = apply_environment(config, parts, rock_mesh, output)
        for extra in extras:
            part = _part(extra['id'], extra['semantic'], extra['vertices'], extra['faces'], None, extra.get('colors'),
                         extra.get('provenance'))
            if part:
                part['material'] = extra.get('material')
                parts.append(part)
    vertices, faces, normals, colors, semantics, identifiers, codes, offset = [], [], [], [], [], [], [], 0
    items = []
    for index, part in enumerate(parts):
        vertices.append(part['vertices']); faces.append(part['faces'] + offset)
        normals.append(part['normals']); colors.append(part['colors'])
        semantics.extend([part['semantic']] * len(part['faces']))
        identifiers.extend([part['id']] * len(part['faces'])); codes.extend([index + 1] * len(part['faces']))
        offset += len(part['vertices'])
        items.append(dict(id=part['id'], semantic=part['semantic'], vertices=len(part['vertices']), faces=len(part['faces']),
            code=index + 1, material_slot='E%04d' % (index + 1), geological_truth=False,
            provenance=part['provenance']))
        if part.get('material'):
            items[-1]['material'] = part['material']
    concatenate = lambda values, width: np.concatenate(values) if values else np.zeros((0, width))
    np.savez_compressed(output / 'scene.npz', vertices=concatenate(vertices, 3), faces=concatenate(faces, 3).astype(np.int64),
        normals=concatenate(normals, 3), colors=concatenate(colors, 3), semantic=np.asarray(semantics, str),
        item_ids=np.asarray(identifiers, str), codes=np.asarray(codes, np.int32))
    _export(parts, output / 'scene.glb')
    rock_vertices = np.asarray(rock_mesh['vertices'], float)
    low, high = rock_vertices.min(0), rock_vertices.max(0)
    margin = max(float(np.max(high - low)), 1.) * 3.
    near_low, near_high = low - margin, high + margin
    browser_parts = []
    for part in parts:
        vertices_inside = np.all((part['vertices'] >= near_low) & (part['vertices'] <= near_high), axis=1)
        keep = np.all(vertices_inside[part['faces']], axis=1)
        if keep.any():
            browser_parts.append(_part(part['id'], part['semantic'], part['vertices'], part['faces'][keep],
                part['normals'], part['colors'], part['provenance']))
    _export(browser_parts, output / 'scene_browser.glb')
    combine_glbs(output / 'preview.glb', output / 'scene_browser.glb', output / 'scene_preview.glb')
    combine_glbs(output / 'gray.glb', output / 'scene_browser.glb', output / 'scene_gray.glb', gray=True)
    result = dict(scene_npz='scene.npz', scene_json='scene.json', scene_glb='scene.glb',
        scene_preview_glb='scene_preview.glb', scene_gray_glb='scene_gray.glb', items=items,
        vertices=offset, faces=len(semantics), omitted_items=omitted, geological_truth=False,
        source='same final processed terrain complement and exposure Items for browser and UE',
        rock_geometry_changed=False, flat_plane_fallback=False)
    ground_parameters = {item['id']: item['provenance']['ground_parameters'] for item in items
                         if 'ground_parameters' in item['provenance']}
    if ground_parameters:
        result['ground_parameters'] = ground_parameters
    preview_vertices = np.vstack([part['vertices'] for part in browser_parts]) if browser_parts else np.zeros((0, 3))
    result.update(scene_browser_glb='scene_browser.glb',
        preview_extent=[preview_vertices.min(0).tolist(), preview_vertices.max(0).tolist()] if len(preview_vertices) else None,
        preview_limit=[near_low.tolist(), near_high.tolist()],
        omitted_far_faces=len(semantics) - sum(len(part['faces']) for part in browser_parts),
        preview_label='near-scene preview',
        preview_contract='bounded subset of original scene triangles; no replacement or geology edits')
    if environment_record is not None:
        result['environment'] = 'environment/environment.json'
    if exposure['kind'] == 'coast':
        result['coast_sediment_role'] = 'processed terrain complement; no unsupported elevated sand lens'
    _write(output / 'scene.json', result)
    return result
