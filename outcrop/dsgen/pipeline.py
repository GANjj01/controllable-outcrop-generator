"""Complete local geometry and scene generation shared by the CLI and web tasks."""
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np

from outcrop.dsgen.exposure import build_surface, sample_exposure
from outcrop.dsgen.lithology import get as lithology, lithology_records, order_key
from outcrop.dsgen.scene import sample_spec
from outcrop.dsgen.stage import write_display_glb
from outcrop.modeling.decode import decode_model
from outcrop.modeling.kernel.types import model_request
from outcrop.modeling.package import build_products, write_package
from outcrop.modeling.resolution import resolve_model
from outcrop.modeling.sampling import SamplingContext

from .generation_config import materialize_scenes, resolve_config


def describe_shape(config):
    """Describe seeded exterior parameters without geological extraction."""
    from types import SimpleNamespace
    from .scene import TILE_MARGIN_M, GROUND_Z_M, MAX_RELIEF_M
    from .shape_config import effective_shape, shape_capabilities

    cfg = resolve_config(config)
    geo = cfg['geometry']
    if geo['model_spec'] is not None:
        return dict(shape=None, overrides=None, fields=[], notes=['The shape is defined by the complete ModelSpec geometry.'])
    tile = dict(size_m=geo['tile_m'], min_xy_m=[TILE_MARGIN_M] * 2,
                ground_z_m=GROUND_Z_M, max_relief_m=MAX_RELIEF_M)
    expo = sample_exposure(geo['exposure'], {'metadata': dict(seed=cfg['seed'], tile=tile)},
                           cfg['seed'], shape=geo['shape'])
    result = shape_capabilities(geo['exposure'])
    result.update(shape=effective_shape(expo), overrides=deepcopy(geo['shape']))
    if expo['kind'] == 'coast':
        accepted = _accepted_coast(cfg, expo, SimpleNamespace(patches=[], camera={}))
        result['shape'].update(length_m=accepted['fit']['actual_length_m'],
                               face_dip_deg=accepted['fit']['nominal_face_dip_deg'],
                               cliff_height_m=accepted['fit']['sampled_height_m'])
        result['fit'] = accepted['fit']
    return result


def _accepted_coast(config, exposure, native_surface):
    """Fit the accepted complete cliff terrain inside this geological tile."""
    from outcrop.dsgen.large_cliff import cliff_surface
    from outcrop.dsgen.surface import Patch
    from .environment import _normals

    geo = config['geometry']
    angle = np.deg2rad(exposure['strike_deg'])
    c, s = np.cos(angle), np.sin(angle)
    frame = np.array([[c, s], [-s, c]])
    shape = geo.get('shape') or {}
    centre = np.asarray(shape.get('centre_xy_m', np.mean(exposure['rock_xy'], axis=0)), float)
    low, high = np.asarray(exposure['rock_xy'])
    sea = exposure['ground_z_m'] + exposure['sea_level_m']
    exact_profile = 'length_m' in shape or 'face_dip_deg' in shape
    length = shape.get('length_m', geo['tile_m'])
    height = shape.get('cliff_height_m', exposure['cliff_height_m'])
    dip = shape.get('face_dip_deg')
    spacing = geo['max_edge_m']
    if exact_profile and not {'length_m', 'face_dip_deg'} <= shape.keys():
        # A partial edit inherits the seeded *fitted* exterior shown by the
        # editor, rather than the wider, pre-fit computational leaf.
        from types import SimpleNamespace
        baseline_config = deepcopy(config)
        baseline_config['geometry']['shape'] = {key: value for key, value in shape.items()
                                                if key not in ('length_m', 'face_dip_deg')}
        baseline = _accepted_coast(baseline_config, exposure, SimpleNamespace(patches=[], camera={}))['fit']
        length = shape.get('length_m', baseline['actual_length_m'])
        dip = shape.get('face_dip_deg', baseline['nominal_face_dip_deg'])
        spacing = baseline['spacing_m']
    leaf = cliff_surface(length_m=length, height_m=height, spacing_m=spacing,
                         face_dip_deg=dip)
    horizontal_scale = 1.
    history = []
    for _ in range(40):
        vertices, faces = leaf['vertices'].copy(), leaf['faces']
        vertices[:, :2] = (vertices[:, :2] * horizontal_scale) @ frame
        vertices[:, 2] += sea - .3 - leaf['dimensions']['ground_z_m']
        triangles = vertices[faces]
        area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area /= np.linalg.norm(area, axis=1, keepdims=True)
        steep = ((triangles[:, :, 2].max(1) > sea + 2.)
                 & (area[:, 2] < np.cos(np.deg2rad(40.))))
        if not steep.any():
            raise ValueError('Coastal cliff sampling has no resolved high steep faces')
        footprint = vertices[np.unique(faces[steep]), :2]
        shift = centre - (footprint.min(0) + footprint.max(0)) / 2
        vertices[:, :2] += shift
        normals = _normals(vertices, faces)
        # Physical slope cover on the accepted terrain; no square tile mask.
        cover = np.clip((normals[:, 2] - np.cos(np.deg2rad(30.)))
                        / (1. - np.cos(np.deg2rad(30.))), 0., 1.)
        bare = cover[faces].mean(1) <= .5
        visible = bare & (vertices[faces, 2].max(1) > sea)

        def clearance(mask):
            xy = vertices[np.unique(faces[mask]), :2]
            return float(np.minimum(xy - low, high - xy).min())

        high_clearance, bare_clearance = clearance(steep), clearance(visible)
        history.append(dict(length_m=length * horizontal_scale, horizontal_scale=horizontal_scale,
            high_steep_faces=int(steep.sum()),
            visible_bare_faces=int(visible.sum()), high_steep_minimum_clearance_m=high_clearance,
            visible_bare_minimum_clearance_m=bare_clearance))
        if high_clearance >= 7. and bare_clearance >= 4.:
            break
        if exact_profile:
            raise ValueError('Requested coastal shape does not fit geological tile: '
                'length=%.3fm, height=%.3fm, nominal dip=%.3fdeg; '
                'steep/bare clearance %.3f/%.3fm requires 7/4m. '
                'Increase tile_m, reduce the requested shape, or move its centre.'
                % (length, height, leaf['profile']['nominal_face_dip_deg'], high_clearance, bare_clearance))
        half = geo['tile_m'] / 2
        ratio = min((half - 7.) / (half - high_clearance),
                    (half - 4.) / (half - bare_clearance))
        # A positive uniform plan transform fits both strike and cross-profile
        # dimensions. Height and the accepted leaf's resolved topology stay fixed.
        horizontal_scale *= min(.98, ratio * .99)
    else:
        raise ValueError('Accepted coastal cliff does not fit the geological tile')
    fit = dict(processor='accepted_cliff_surface', sampled_height_m=height,
        actual_length_m=length * horizontal_scale, horizontal_scale=horizontal_scale,
        source_leaf_length_m=length, source_leaf_spacing_m=spacing,
        nominal_face_dip_deg=float(np.degrees(np.arctan(
            np.tan(np.deg2rad(leaf['profile']['nominal_face_dip_deg'])) / horizontal_scale))),
        exact_profile_requested=exact_profile,
        spacing_m=spacing * horizontal_scale, mean_sea_level_m=sea,
        base_below_mean_sea_m=.3, strike_deg=exposure['strike_deg'], xy_shift_m=shift.tolist(),
        high_steep_minimum_clearance_m=high_clearance,
        visible_bare_minimum_clearance_m=bare_clearance,
        high_steep_definition='all corners of every slope >40deg face with any corner >2m above sea',
        visible_bare_definition='all corners of every slope-cover bare face with any corner above sea',
        physical_cover='source vertex slope field with 30deg reference; no tile mask',
        tile_mask_used=False, ordinary_erode_used=False, wave_cut_platform=False,
        fit_history=history, profile=leaf['profile'])
    native_surface.patches = [Patch('coast', vertices, faces, normals,
        provenance=dict(construction='accepted dsgen.large_cliff.cliff_surface', fit=fit))]
    native_surface.camera.pop('rock_xy', None)
    return dict(vertices=vertices, faces=faces, normals=normals, cover=cover,
        colors=np.full_like(vertices, .3), processor='accepted_cliff_surface', fit=fit,
        description='accepted complete cliff height field; physical slope cover; no ordinary erode or tile mask')


def build_declaration(config):
    """Return a concrete ModelSpec and its original, uncut exposure surface.

    All requested layers are geological bodies; unconformity divides that
    succession into old and young layers. The caller applies the ordinary
    accepted surface processes; frozen studies and scan-texture routes are
    not invoked.
    """
    cfg = resolve_config(config)
    geo = cfg['geometry']
    layers = geo['layer_count']
    young_count = min(2, layers - 1) if geo['unconformity'] else 0
    old_count = layers - young_count
    selected = [geo['lithologies'][i % len(geo['lithologies'])] for i in range(layers)]
    strata = dict(kind='nonparallel', layer_count=old_count, shape=[3, 2, 2],
                  relief_m=.6, thickness_variation=.15, phase_rad=[.2, .7])
    if young_count:
        strata['unconformity'] = dict(elevation_m=14., dip_deg=8., dip_direction_deg=120.,
            young_lithologies=selected[old_count:], young_fractions=[1.] * young_count,
            young_dip_deg=3., young_dip_direction_deg=35.)
    if geo.get('stratigraphy') is not None:
        supplied = deepcopy(geo['stratigraphy'])
        if supplied.get('layer_count', old_count) != old_count:
            raise ValueError('stratigraphy.layer_count must match the requested old-layer count')
        if supplied.get('kind') == 'planar':
            strata = {key: value for key, value in strata.items()
                      if key in ('layer_count', 'unconformity')}
        strata.update(supplied)
    network = dict(count=geo['fracture_count'], length_median_m=12., length_log_sigma=.35,
        step_m=2., interaction_radius_m=3., mean_direction_deg=30., direction_std_deg=25.,
        direction_weight=.35, stop_probability=.8, height_m=20.)
    if geo.get('fracture_network'):
        network.update(deepcopy(geo['fracture_network']))
    options = dict(family=geo['family'], tile_m=geo['tile_m'], stratigraphy=strata,
                   fracture_network=network,
                   fault_count=0 if geo.get('faults') is not None else geo['fault_count'])
    if geo['family'] == 'group_b':
        options['group_b_setting'] = geo['group_b_setting']
    raw = sample_spec(cfg['seed'], **options)
    old_bodies = [body for body in raw['bodies'] if body['id'].startswith('bed-')]
    for body, name in zip(old_bodies, selected[:old_count]):
        body['material']['lithology'] = name
    raw['lithologies'] = lithology_records([body['material']['lithology'] for body in raw['bodies']])
    raw['metadata']['beds'] = selected[:old_count]
    if geo.get('faults') is not None:
        from .faults import apply_faults
        apply_faults(raw, geo['faults'])
    expo = sample_exposure(geo['exposure'], raw, cfg['seed'], shape=geo['shape'])
    if expo['kind'] == 'roadcut':
        expo['guardrail'] = False
    # The exposure patch contains the geological surface. Road/guardrail and
    # other Item factories are deliberately not called by this adapter.
    surf = build_surface(expo, samples=tuple(geo['samples']), max_edge=geo['max_edge_m'],
                         ground_options=cfg['scene']['ground'])
    accepted = _accepted_coast(cfg, expo, surf) if expo['kind'] == 'coast' else None
    if expo['kind'] == 'bluff':
        # sin/cos at mathematical axes must be exactly zero. Leaving a
        # 1e-16 component puts the same radial axis one ULP beside an exact
        for patch in surf.patches:
            directions = patch.outward
            directions[np.abs(directions) < 8 * np.finfo(float).eps] = 0.
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            patch.vertices = patch.positions(np.zeros(len(directions)))
    vertices, faces, offset = [], [], 0
    for patch in surf.patches:
        vertices.append(np.asarray(patch.vertices, float))
        faces.append(np.asarray(patch.faces, np.int64) + offset)
        offset += len(patch.vertices)
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    if geo['shape'] is not None and accepted is not None:
        z = vertices[:, 2]
        bare = accepted['cover'] <= .5
        visible = bare & (z > accepted['fit']['mean_sea_level_m'])
        low_z, high_z = raw['domain']['min_m'][2] + 2., raw['domain']['max_m'][2] - 2.
        if not visible.any() or z[visible].min() < low_z or z[visible].max() > high_z:
            raise ValueError('Requested coastal height/sea level exceeds geological vertical bounds '
                             'with 2m surface-process room (%.3f..%.3fm)' % (low_z, high_z))
    if geo['shape'] is not None and expo['kind'] != 'coast':
        # Extended terrain is deliberately outside the geological volume.
        # Check the central exposed face, rather than its distant soil apron.
        lo, hi = np.asarray(raw['domain']['min_m']), np.asarray(raw['domain']['max_m'])
        if expo['kind'] == 'bluff':
            intended = vertices
        else:
            centre = np.asarray(expo['centre_xy_m'])
            angle = np.deg2rad(expo['strike_deg'])
            along = np.array([np.cos(angle), np.sin(angle)])
            u = (vertices[:, :2] - centre) @ along
            normals = np.concatenate([patch.outward for patch in surf.patches])
            central_face = (np.abs(u) <= max(1., 2 * geo['max_edge_m'])) & (normals[:, 2] < np.cos(np.deg2rad(40.)))
            intended = vertices[central_face]
        if not len(intended) or np.any(intended.min(0) < lo + 2.) or np.any(intended.max(0) > hi - 2.):
            raise ValueError('Requested exterior exceeds geological volume with 2m surface-process room; '
                'increase tile_m or reduce/move the shape. Exterior bounds: %s; geological bounds: %s'
                % ([intended.min(0).tolist(), intended.max(0).tolist()] if len(intended) else 'empty', [lo.tolist(), hi.tolist()]))
    # The accepted 720 x 240 source alone has 345,600 faces. Scale the
    # resource allowance to that real workload before both geology packages;
    # numerical error and noncollapse gates retain their existing values.
    raw['meshing'].setdefault('budget', {})['max_predicate_pairs'] = max(1_000_000, len(faces) * 64)
    for key, value in (geo.get('meshing') or {}).items():
        if key in ('budget', 'tolerance'):
            raw['meshing'].setdefault(key, {}).update(deepcopy(value))
        else:
            raw['meshing'][key] = deepcopy(value)
    terrain = next(item for item in raw['geometry'] if item['id'] == 'section')
    terrain.update(vertices=vertices.tolist(), faces=faces.tolist(), closed=False)
    raw['model_id'] = 'workbench-%s-%s' % (geo['exposure'], cfg['seed'])
    raw['metadata']['workbench'] = dict(layer_count=layers, exposure=geo['exposure'],
        surface_processes='none; original complete exposure geometry',
        environment_is_geological_truth=False)
    source = dict(vertices=vertices, faces=faces, exposure=expo, surface=surf)
    from .shape_config import effective_shape
    exterior = dict(requested_shape=deepcopy(geo['shape']), effective_shape=effective_shape(expo),
        source_bounds_enu_m=[vertices.min(0).tolist(), vertices.max(0).tolist()],
        dimensions_are='source construction parameters; ordinary surface processes may change the final extent')
    if accepted is not None:
        exterior['effective_shape'].update(length_m=accepted['fit']['actual_length_m'],
            face_dip_deg=accepted['fit']['nominal_face_dip_deg'],
            cliff_height_m=accepted['fit']['sampled_height_m'])
    raw['metadata']['workbench']['exterior'] = exterior
    source['exterior'] = exterior
    if accepted is not None:
        source.update(accepted)
        raw['metadata']['workbench'].update(exposure_parameters=expo,
            processor=accepted['processor'], exposure_fit=accepted['fit'])
    return raw, source


def _normals_from_source(vertices, faces, source_ids, reference_vertices, reference_faces, reference_normals=None):
    """Interpolate original source shading normals across exact geological cuts."""
    original_triangles = reference_vertices[reference_faces]
    areas = np.cross(original_triangles[:, 1] - original_triangles[:, 0],
                     original_triangles[:, 2] - original_triangles[:, 0])
    original_normals = np.zeros_like(reference_vertices) if reference_normals is None else np.asarray(reference_normals, float).copy()
    if reference_normals is None:
        for corner in range(3):
            np.add.at(original_normals, reference_faces[:, corner], areas)
    lengths = np.linalg.norm(original_normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError('source surface has undefined shading normals')
    original_normals /= lengths[:, None]
    indices = np.asarray([int(token.rsplit(':', 1)[1]) for token in source_ids])
    originals = original_triangles[indices]
    # Use each original triangle's strongest projected area, so vertical and
    # overhanging bluff faces work as well as XY height fields.
    drop = np.argmax(np.abs(areas[indices]), axis=1)
    axes = np.array([[1, 2], [0, 2], [0, 1]])[drop]
    projected = np.take_along_axis(originals, axes[:, None, :], axis=2)
    clipped = np.take_along_axis(vertices[faces], axes[:, None, :], axis=2)
    matrix = np.stack([projected[:, 1] - projected[:, 0], projected[:, 2] - projected[:, 0]], axis=2)
    uv = np.linalg.solve(matrix[:, None], (clipped - projected[:, None, 0])[..., None])[..., 0]
    barycentric = np.concatenate([1. - uv.sum(2, keepdims=True), uv], axis=2)
    corners = np.einsum('fij,fjk->fik', barycentric, original_normals[reference_faces[indices]])
    result = np.zeros_like(vertices)
    result[faces.ravel()] = corners.reshape(-1, 3)
    length = np.linalg.norm(result, axis=1)
    if np.any(length == 0) or not np.isfinite(result).all():
        raise ValueError('cut surface has undefined shading normals')
    return result / length[:, None]


def _linear(color):
    rgb = np.asarray(color, float) / 255.
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4).tolist()


def _preview(mesh, codes, book, target, gray=False):
    """Reuse the current GLB exporter and apply per-region material colours."""
    write_display_glb(dict(mesh, colors=np.ones_like(mesh['vertices'])), codes, target)
    data = target.read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    document = json.loads(data[20:20 + length])
    colors = {record['code']: _linear(record['color_srgb']) for record in book}
    for material in document['materials']:
        code = int(material['name'][1:])
        material['pbrMetallicRoughness']['baseColorFactor'] = ([.25] * 3 if gray else colors[code]) + [1.]
    text = json.dumps(document, separators=(',', ':')).encode('utf8')
    text += b' ' * (-len(text) % 4)
    tail = data[20 + length:]
    target.write_bytes(struct.pack('<4sII', b'glTF', 2, 20 + len(text) + len(tail))
                       + struct.pack('<II', len(text), 0x4E4F534A) + text + tail)


def _json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + '\n', encoding='utf8')
    return json.loads(text)


def _structure_preview(source, target, segments, interface_ids):
    """Overlay only the kernel's actual finite interface/surface intersections."""
    if not len(segments):
        shutil.copyfile(source, target)
        return
    data = source.read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    document = json.loads(data[20:20 + length])
    blob = bytearray(data[28 + length:])
    positions = (np.asarray(segments)[:, :, [0, 2, 1]] * [1, 1, -1]).reshape(-1, 3).astype(np.float32)
    blob.extend(b'\0' * (-len(blob) % 4))
    document['bufferViews'].append(dict(buffer=0, byteOffset=len(blob), byteLength=positions.nbytes, target=34962))
    blob.extend(positions.tobytes())
    document['accessors'].append(dict(bufferView=len(document['bufferViews']) - 1,
        componentType=5126, count=len(positions), type='VEC3', min=positions.min(0).tolist(), max=positions.max(0).tolist()))
    accessor, material = len(document['accessors']) - 1, len(document['materials'])
    document['materials'].append(dict(name='TrueGeologicalTraces',
        pbrMetallicRoughness=dict(baseColorFactor=_linear([255, 92, 32]) + [1.], metallicFactor=0., roughnessFactor=1.),
        extensions={'KHR_materials_unlit': {}}))
    extensions = document.setdefault('extensionsUsed', [])
    if 'KHR_materials_unlit' not in extensions:
        extensions.append('KHR_materials_unlit')
    mesh, node = len(document['meshes']), len(document['nodes'])
    document['meshes'].append(dict(name='TrueGeologicalTraces', primitives=[
        dict(mode=1, attributes=dict(POSITION=accessor), material=material)]))
    document['nodes'].append(dict(name='TrueGeologicalTraces', mesh=mesh,
        extras=dict(label='true geological interface traces', source='package/traces.npz',
                    interface_ids=list(interface_ids), geological_truth=True)))
    document['scenes'][document.get('scene', 0)]['nodes'].append(node)
    blob.extend(b'\0' * (-len(blob) % 4))
    document['buffers'][0]['byteLength'] = len(blob)
    text = json.dumps(document, separators=(',', ':')).encode('utf8')
    text += b' ' * (-len(text) % 4)
    target.write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(text) + len(blob))
        + struct.pack('<II', len(text), 0x4E4F534A) + text
        + struct.pack('<II', len(blob), 0x004E4942) + blob)


def generate_scene(config, output):
    cfg, output = resolve_config(config), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('geometry output must be a new or empty directory')
    if cfg['count'] == 1 and cfg['geometry'].get('model_spec') is not None:
        from .declared import generate_declared_scene
        return generate_declared_scene(cfg, output)
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    if cfg['count'] > 1:
        scenes = []
        for index, actual in enumerate(materialize_scenes(cfg)):
            folder = 'scene-%04d' % index
            result = generate_scene(actual, output / folder)
            record = dict(seed=actual['seed'], directory=folder,
                geometry_json=folder + '/geometry.json', faces=result['faces'],
                scene_faces=result['scene']['faces'])
            if result.get('native_model'):
                record.update(native_root_seed=result['native_model']['root_seed'],
                              native_sample_id=result['native_model']['sample_id'])
            scenes.append(record)
        return _json(output / 'scenes.json', dict(count=len(scenes), scenes=scenes,
            validation=dict(success=True)))
    raw, source = build_declaration(cfg)
    from .boundary import validate_boundary, verify_source_coverage
    from .environment import build_scene
    from outcrop.dsgen.processes import erode
    from outcrop.dsgen.truth import TruthModel

    reference_vertices, reference_faces = source['vertices'], source['faces']
    np.savez_compressed(output / 'reference_surface.npz', vertices=reference_vertices, faces=reference_faces)
    exposure_surface = source['surface']
    truth = None
    accepted = source.get('processor') == 'accepted_cliff_surface'
    if accepted:
        processed = {key: source[key] for key in ('vertices', 'faces', 'normals', 'cover', 'colors')}
        processed['stats'] = dict(processor='accepted_cliff_surface', ordinary_erode_used=False,
            tile_mask_used=False, fold_guard_final_flipped=0,
            applied_recession_m=dict(median=0., p95=0., max=0.))
    else:
        # A small diagnostic section packages the complete geological volume
        # for ordinary erode; it is not the displayed mesh.
        preliminary = deepcopy(raw)
        lo, hi = np.asarray(raw['domain']['min_m']), np.asarray(raw['domain']['max_m'])
        y = float((lo[1] + hi[1]) / 2)
        section = next(item for item in preliminary['geometry'] if item['id'] == 'section')
        section.update(vertices=[[lo[0] + .5, y, lo[2] + .5], [hi[0] - .5, y, lo[2] + .5],
            [hi[0] - .5, y, hi[2] - .5], [lo[0] + .5, y, hi[2] - .5]], faces=[[0, 1, 2], [0, 2, 3]])
        preliminary['model_id'] += '-source-geology'
        source_spec = decode_model(preliminary)
        source_resolved = resolve_model(source_spec, SamplingContext(cfg['seed'], 'workbench-source'))
        source_request = model_request(source_resolved.numeric)
        source_products = build_products(source_resolved, preliminary['exposures'][0], source_request)
        write_package(source_spec, source_resolved, source_products, output / 'source_package')
        truth = TruthModel(output / 'source_package')
        items, exposure_surface.items = exposure_surface.items, []
        try:
            processed = erode(exposure_surface, preliminary, truth, np.random.default_rng(cfg['seed'] + 7919),
                               clay=cfg['geometry']['exposure'] == 'bluff', tint_jitter='luma')
        finally:
            exposure_surface.items = items
    # Ordinary erode adds only arithmetic midpoint subdivisions in its final
    # approximate contact-label pass. Use the same processed PL surface on
    # its original triangles, then replace that pass by exact contacts once.
    # Preserve the full leaf output as evidence; no shape positions move.
    np.savez_compressed(output / 'processed_surface.npz', vertices=processed['vertices'],
        faces=processed['faces'], normals=processed['normals'])
    original_count = len(reference_vertices)
    pv, pf, pn = processed['vertices'][:original_count], reference_faces.copy(), processed['normals'][:original_count]
    orientation = np.cross(pv[pf[:, 1]] - pv[pf[:, 0]], pv[pf[:, 2]] - pv[pf[:, 0]])
    if (orientation * pn[pf].mean(1)).sum() < 0:
        pf = pf[:, ::-1]
    cover = processed['cover'][:original_count]
    cover_faces = cover[pf].mean(1) > .5
    candidates = np.ones(len(pf), bool)
    if truth is not None:
        status = truth.query(pv[pf].mean(1))[0]
        candidates = status == 'occupied'
    # The accepted coast does not infer labels from centres. All its physical
    # bare faces must survive final exact clipping in full, or generation fails.
    semantic = np.where(cover_faces, 'soil', 'bedrock')
    if accepted:
        sea = source['fit']['mean_sea_level_m']
        semantic = np.where(cover_faces & (pv[pf].mean(1)[:, 2] < sea + .1), 'sediment', semantic)
    processed.update(vertices=pv, faces=pf, normals=pn, cover=cover, cover_faces=cover_faces,
        colors=processed['colors'][:original_count], semantic=semantic)
    source['exterior']['processed_bounds_enu_m'] = [pv.min(0).tolist(), pv.max(0).tolist()]
    contract = 'closed_rock' if cfg['geometry']['exposure'] == 'bluff' else 'open_terrain'
    reference = (reference_vertices, reference_faces) if contract == 'open_terrain' else None
    source_boundary = validate_boundary(pv, pf, contract=contract, reference=reference)
    if processed['stats']['fold_guard_final_flipped']:
        raise ValueError('Natural surface processes left inverted terrain triangles')
    source.update(vertices=pv, faces=pf, normals=pn)
    np.savez_compressed(output / 'source_surface.npz', vertices=pv, faces=pf, normals=pn,
        cover=processed['cover'], cover_faces=processed['cover_faces'], semantic=np.asarray(processed['semantic'], str))
    # Full terrain stays present. Geological cutting consumes only actual
    # bare occupied faces; covered/outside faces remain its exact complement.
    rock = np.ones(len(pf), bool) if contract == 'closed_rock' else (
        ~processed['cover_faces'] & (processed['semantic'] == 'bedrock')
        & candidates)
    if not rock.any():
        raise ValueError('Natural exposure has no visible geological rock surface')
    selected_faces = pf[rock]
    source['environment'] = dict(vertices=pv, faces=pf[~rock], normals=pn,
        colors=processed['colors'], semantic=np.asarray(processed['semantic'][~rock], str))
    section = next(item for item in raw['geometry'] if item['id'] == 'section')
    section.update(vertices=pv.tolist(), faces=selected_faces.tolist())
    raw['metadata']['workbench']['surface_processes'] = source.get('description',
        'accepted ordinary erode geometry; no scan, block-spalling or image-derived displacement')
    _json(output / 'declaration.json', raw)
    spec = decode_model(raw)
    resolved = resolve_model(spec, SamplingContext(cfg['seed'], 'workbench'))
    request = model_request(resolved.numeric)
    products = build_products(resolved, raw['exposures'][0], request)
    surface = products.extraction.surface
    if not len(surface.faces):
        raise ValueError('geology does not occupy the generated exposure surface')
    if any(value is None for values in (surface.body_ids, surface.unit_ids, surface.lithology_ids) for value in values):
        raise ValueError('exposure has ambiguous geological identities')
    # Existing package writer enforces error budgets and float32 noncollapse;
    # no faces or scientific checks are dropped for the display export.
    write_package(spec, resolved, products, output / 'package')
    vertices, faces = surface.vertices, surface.faces
    source_coverage = verify_source_coverage(pv, selected_faces, vertices, faces, surface.source_ids)
    normals = _normals_from_source(vertices, faces, surface.source_ids,
                                   pv, selected_faces, pn)
    environment = source['environment']
    assembled_vertices = np.vstack([vertices, pv])
    assembled_faces = np.vstack([faces, environment['faces'] + len(vertices)])
    boundary = validate_boundary(assembled_vertices, assembled_faces, contract=contract, reference=reference)
    validation = dict(success=True, source=source_boundary, assembled=boundary,
        source_coverage=source_coverage, fold_guard_final_flipped=0,
        geological_labels='exact final-volume clipping; environment carries no geological labels')
    _json(output / 'boundary_validation.json', validation)
    keys = sorted(set(zip(surface.body_ids, surface.unit_ids, surface.lithology_ids)),
                  key=lambda k: (k[0], k[1], order_key(k[2])))   # region codes in the catalogue's order field
    book = []
    overrides = cfg['appearance']['unit_overrides']
    for code, (body, unit, name) in enumerate(keys, 1):
        base = [int(round(255 * value)) for value in lithology(name)['base'][0]]
        color = overrides.get(unit, {}).get('color_srgb', base)
        book.append(dict(code=code, body=body, unit=unit, lithology=name, color_srgb=color))
    lookup = {(record['body'], record['unit'], record['lithology']): record['code'] for record in book}
    codes = np.asarray([lookup[key] for key in zip(surface.body_ids, surface.unit_ids, surface.lithology_ids)], np.int32)
    np.savez_compressed(output / 'surface.npz', vertices=vertices, faces=faces, normals=normals,
        codes=codes, body_ids=np.asarray(surface.body_ids, dtype=str),
        unit_ids=np.asarray(surface.unit_ids, dtype=str), lithology_ids=np.asarray(surface.lithology_ids, dtype=str),
        source_ids=np.asarray(surface.source_ids, dtype=str))
    _json(output / 'codebook.json', book)
    mesh = dict(vertices=vertices, faces=faces, normals=normals)
    _preview(mesh, codes, book, output / 'preview.glb')
    _preview(mesh, codes, book, output / 'gray.glb', gray=True)
    bounds = [vertices.min(0).tolist(), vertices.max(0).tolist()]
    scene = build_scene(cfg, output, exposure_surface, environment, mesh, source['exposure'])
    traces = products.extraction.traces
    np.savez_compressed(output / 'traces.npz', segments=traces.segments,
                        interface_ids=np.asarray(traces.interface_ids, dtype=str))
    _structure_preview(output / 'scene_preview.glb', output / 'structure_preview.glb',
                       traces.segments, traces.interface_ids)
    _structure_preview(output / 'preview.glb', output / 'structure_rock.glb', traces.segments, traces.interface_ids)
    interface_counts = {oid: traces.interface_ids.count(oid) for oid in sorted(set(traces.interface_ids))}
    features = {item['id']: item['kind'] for item in raw['features'] + raw['events']}
    trace_interfaces = [dict(id=item.id, feature_id=item.feature_id,
        kind=features.get(item.feature_id, 'interface'), segment_count=interface_counts[item.id])
        for item in products.extraction.interfaces if item.id in interface_counts]
    metadata = dict(vertices=len(vertices), faces=len(faces),
        pipeline_version=2, geometry_kind=contract, boundary=boundary,
        source_boundary=source_boundary, validation=validation, boundary_validation='boundary_validation.json',
        source_coverage=source_coverage, surface_processes=processed['stats'],
        resource_budget=asdict(request.budget),
        contact_tessellation=('accepted complete cliff PL geometry; final exact geological cut' if accepted else
            'ordinary process PL geometry; approximate midpoint label tessellation replaced by final exact geological cut'),
        exposure_fit=source.get('fit'),
        exterior=dict(source['exterior'], final_rock_bounds_enu_m=[vertices.min(0).tolist(), vertices.max(0).tolist()]),
        bounds=bounds,
        preview_glb='preview.glb', gray_glb='gray.glb', scene_glb='scene.glb',
        scene_preview_glb='scene_preview.glb', scene_gray_glb='scene_gray.glb',
        structure_preview_glb='structure_preview.glb', trace_count=len(traces.segments),
        structure_rock_glb='structure_rock.glb',
        trace_interfaces=trace_interfaces,
        trace_legend=[dict(label='true geological interface traces', color_srgb=[255, 92, 32],
            source='package/traces.npz', interface_ids=sorted(interface_counts))],
        surface='surface.npz', codebook='codebook.json', declaration='declaration.json', package='package',
        exposure=cfg['geometry']['exposure'], source_vertices=len(source['vertices']), source_faces=len(source['faces']),
        layer_count=cfg['geometry']['layer_count'], exposed_units=sorted(set(surface.unit_ids)),
        error_report=asdict(products.extraction.error_report), elapsed_s=time.monotonic() - start,
        scene=scene, geometry_frozen=True)
    return _json(output / 'geometry.json', metadata)


# Existing Python consumers retain the same function, never a second pipeline.
generate_geometry = generate_scene
