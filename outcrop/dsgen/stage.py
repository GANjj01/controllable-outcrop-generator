"""Stage a geology package for a multi-view capture."""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

ENGINE = Path('UnrealEditor.exe')
# stays untouched): plugin list and renderer settings for every capture / viewer project.
TEMPLATE = Path(__file__).absolute().parents[3] / 'configs' / 'vog_ue' / 'template.uproject'
SETTINGS = TEMPLATE.parent / 'DefaultEngine.ini'
SHORT_ROOT = Path(__file__).resolve().parents[2] / 'ue'
VIEWER_PROJECT = SHORT_ROOT / 'VOGViewer' / 'VOGViewer.uproject'
# Region codes travel through an unlit emissive ID pass that saturates near 100 (semantics.py).
MAX_CODES = 49

DEFAULT_SAMPLING = {
    'resolution': [1280, 720],
    'shots': 12,
    'yaw_deg': [-40., 40.],
    'pitch_deg': [-10., 25.],
    'fill': [.55, 1.05],
    'focal_length_mm': [24., 50.],
    'sensor_width_mm': 36.,
    'exposure_ev': [.6, 1.6],
    # Sun azimuth relative to the exposure's outward normal: front- to side-lit, rarely grazing.
    'sun_azimuth_rel_deg': [-100., 100.],
    'display': True,
    'max_edge_m': .08,
    'sun_elevation_deg': [15., 70.],
    'sun_lux': [18000., 45000.],
    'sky_intensity': [1.0, 2.2],
    'ground_burial_m': [.05, .6],
    'engine_warmup': 64,
    'render_warmup': 24,
    # Rays (azimuth, latitude) of the bluff surface: ~1M vertices.
    'bluff_rays': [1800, 520],
    # Sampling of exposure sheets (escarpment, ...) over the tile, metres.
    'sheet_edge_m': .07,
    # Captures staged before have no such key and used 'rgb'; reproduce them with tint_jitter='rgb'.
    'tint_jitter': 'luma',
}

from .legacy import former as _former, name as _label  # noqa: E402  (the frozen VOG texlib matches the former names)
from .lithology import order_key as _order_key  # noqa: E402
LITHOLOGY_COLORS = {
    'group_a_01': ([0.62, 0.5, 0.36], [0.48, 0.38, 0.27]),
    'group_a_02': ([0.3, 0.3, 0.29], [0.2, 0.2, 0.2]),
    'group_a_03': ([0.5303921568627451, 0.5362745098039216, 0.7343137254901961], [0.37127450980392157, 0.3753921568627451, 0.5140196078431373]),
    'group_a_04': ([0.72, 0.7, 0.63], [0.58, 0.56, 0.5]),
    'group_a_05': ([0.42647058823529416, 0.6284313725490196, 0.4715686274509804], [0.2985294117647059, 0.4399019607843137, 0.33009803921568626]),
    'group_a_06': ([0.31862745098039214, 0.40098039215686276, 0.3323529411764706], [0.22303921568627447, 0.28068627450980393, 0.23264705882352937]),
    'group_a_07': ([0.5696078431372549, 0.557843137254902, 0.5911764705882353], [0.3987254901960784, 0.3904901960784314, 0.4138235294117647]),
    'group_b_01': ([0.18, 0.18, 0.19], [0.1, 0.1, 0.11]),
    'group_b_02': ([0.7284313725490197, 0.45, 0.3284313725490196], [0.5099019607843137, 0.315, 0.2299019607843137]),
    'group_b_03': ([0.37549019607843137, 0.6990196078431372, 0.2872549019607843], [0.26284313725490194, 0.489313725490196, 0.201078431372549]),
    'group_b_04': ([0.653921568627451, 0.3284313725490196, 0.7382352941176471], [0.45774509803921565, 0.2299019607843137, 0.5167647058823529]),
    'group_b_05': ([0.4852941176470588, 0.6852941176470588, 0.49901960784313726], [0.33970588235294114, 0.47970588235294115, 0.34931372549019607]),
    'group_b_06': ([0.66, 0.62, 0.58], [0.45, 0.42, 0.4]),
    'group_b_07': ([0.3970588235294118, 0.35784313725490197, 0.75], [0.27794117647058825, 0.25049019607843137, 0.5249999999999999]),
    'group_b_08': ([0.653921568627451, 0.40098039215686276, 0.553921568627451], [0.45774509803921565, 0.28068627450980393, 0.3877450980392157]),
    'group_b_09': ([0.34215686274509804, 0.4872549019607843, 0.3343137254901961], [0.23950980392156862, 0.341078431372549, 0.23401960784313727]),
    'group_b_10': ([0.5343137254901961, 0.5303921568627451, 0.3284313725490196], [0.3740196078431372, 0.37127450980392157, 0.2299019607843137]),
    'group_c_01': ([0.6068627450980393, 0.34215686274509804, 0.36568627450980395], [0.42480392156862745, 0.23950980392156862, 0.25598039215686275]),
    'group_c_02': ([0.4323529411764706, 0.6049019607843138, 0.5088235294117647], [0.3026470588235294, 0.42343137254901964, 0.35617647058823526]),
    'group_c_03': ([0.42, 0.44, 0.4], [0.28, 0.3, 0.28]),
    'group_c_04': ([0.6676470588235295, 0.3931372549019608, 0.5735294117647058], [0.4673529411764706, 0.2751960784313725, 0.4014705882352941]),
    'group_c_05': ([0.4480392156862745, 0.6990196078431372, 0.7480392156862745], [0.31362745098039213, 0.489313725490196, 0.5236274509803921]),
    'group_c_06': ([0.5813725490196078, 0.6833333333333333, 0.5480392156862746], [0.4069607843137254, 0.47833333333333333, 0.3836274509803922]),
    'group_c_07': ([0.5323529411764706, 0.3323529411764706, 0.2950980392156863], [0.3726470588235294, 0.23264705882352937, 0.2065686274509804]),
    'group_c_08': ([0.6401960784313725, 0.41666666666666663, 0.6147058823529412], [0.4481372549019607, 0.29166666666666663, 0.4302941176470588]),
    'sand': ([0.62, 0.5, 0.36], [0.48, 0.38, 0.27]),
}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def load_package(package):
    package = Path(package)
    surface = np.load(package / 'surface.npz', allow_pickle=False)
    labels = _read(package / 'labels.json')
    vertices = np.asarray(surface['vertices'], np.float64)
    faces = np.asarray(surface['faces'], np.int64)
    if not len(faces):
        raise ValueError('nonempty exposure surface required')
    n = len(faces)
    lith = labels['surface_lithology_ids']
    units = labels['surface_unit_ids']
    bodies = labels['surface_body_ids']
    if not (len(lith) == len(units) == len(bodies) == n):
        raise ValueError('per-face labels must close the exposure surface')
    traces = np.load(package / 'traces.npz', allow_pickle=False)['segments']
    trace_ids = _read(package / 'traces.json')['interface_ids']
    if len(trace_ids) != len(traces):
        raise ValueError('trace segment/interface id mismatch')
    model = _read(package / 'model_spec.json')
    cliff = next((g for g in model['geometry'] if g['id'] == 'cliff' and g.get('kind') == 'mesh'), None)
    return dict(vertices=vertices, faces=faces, lithology=lith, unit=units, body=bodies,
                source_ids=labels['source_ids'],
                model_cliff_vertices=np.asarray(cliff['vertices'], float) if cliff else None,
                segments=np.asarray(traces, np.float64), segment_ids=trace_ids,
                model=_read(package / 'model_spec.json'))


def classify_interfaces(model, interface_ids):
    """Map trace interface ids to line classes using the declared model objects."""
    kinds = {}
    for f in model.get('features', []):
        kinds[f['id']] = {'fault': 'fault', 'joint_set': 'joint', 'joint': 'joint'}.get(f['kind'], 'contact')
    for e in model.get('events', []):
        kinds[e['id']] = {'fault': 'fault', 'intrusion': 'contact'}.get(e['kind'], 'contact')
    result = {}
    for iid in sorted(set(interface_ids)):
        owner = iid.split(':', 1)[0]
        result[iid] = kinds.get(owner, 'contact')
    return result


def codebook_for(pkg):
    keys = sorted({(b, u if u is not None else '', l) for b, u, l in zip(pkg['body'], pkg['unit'], pkg['lithology'])},
                  key=lambda k: (k[0], k[1], _order_key(k[2])))   # region codes in the catalogue's order field
    if len(keys) > MAX_CODES:
        raise ValueError('too many region labels for exact half-float transport')
    book = [dict(code=i + 1, body=b, unit=u or None, lithology=l) for i, (b, u, l) in enumerate(keys)]
    index = {(e['body'], e['unit'] or '', e['lithology']): e['code'] for e in book}
    face_codes = np.array([index[(b, u if u is not None else '', l)]
                           for b, u, l in zip(pkg['body'], pkg['unit'], pkg['lithology'])], np.int64)
    return book, face_codes


def write_glb(vertices, faces, face_codes, target):
    """One flat-shaded primitive per region code; glTF frame (E, U, -N), metres."""
    doc = {'asset': {'version': '2.0', 'generator': 'outcrop-dsgen'}, 'scene': 0,
           'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0, 'name': 'Outcrop'}],
           'meshes': [{'name': 'Outcrop', 'primitives': []}], 'materials': [],
           'accessors': [], 'bufferViews': [], 'buffers': []}
    blob = bytearray()

    def add(array, kind, component, target_kind):
        blob.extend(b'\0' * (-len(blob) % 4))
        view = {'buffer': 0, 'byteOffset': len(blob), 'byteLength': array.nbytes, 'target': target_kind}
        blob.extend(array.tobytes())
        doc['bufferViews'].append(view)
        acc = {'bufferView': len(doc['bufferViews']) - 1, 'componentType': component, 'count': len(array), 'type': kind}
        if kind == 'VEC3':
            acc.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        doc['accessors'].append(acc)
        return len(doc['accessors']) - 1

    gltf = vertices[:, [0, 2, 1]] * np.array([1., 1., -1.])
    for code in sorted(set(face_codes.tolist())):
        tri = gltf[faces[face_codes == code]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.where(length > 0, normal / np.maximum(length, 1e-30), [0., 1., 0.])
        pos = tri.reshape(-1, 3).astype(np.float32)
        nor = np.repeat(normal, 3, axis=0).astype(np.float32)
        # World-planar UV (metres) so later texture materials have a usable channel.
        uv = np.stack([tri.reshape(-1, 3)[:, 0] + tri.reshape(-1, 3)[:, 2], tri.reshape(-1, 3)[:, 1]], 1).astype(np.float32)
        idx = np.arange(len(pos), dtype=np.uint32)
        doc['materials'].append({'name': 'C%04d' % code, 'doubleSided': True,
                                 'pbrMetallicRoughness': {'baseColorFactor': [.5, .5, .5, 1.], 'metallicFactor': 0., 'roughnessFactor': 1.}})
        doc['meshes'][0]['primitives'].append({
            'attributes': {'POSITION': add(pos, 'VEC3', 5126, 34962), 'NORMAL': add(nor, 'VEC3', 5126, 34962),
                           'TEXCOORD_0': add(uv, 'VEC2', 5126, 34962)},
            'indices': add(idx, 'SCALAR', 5125, 34963), 'material': len(doc['materials']) - 1})
    blob.extend(b'\0' * (-len(blob) % 4))
    doc['buffers'].append({'byteLength': len(blob)})
    text = json.dumps(doc, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 4)
    data = struct.pack('<4sII', b'glTF', 2, 28 + len(text) + len(blob)) + struct.pack('<II', len(text), 0x4E4F534A) + text \
        + struct.pack('<II', len(blob), 0x004E4942) + bytes(blob)
    Path(target).write_bytes(data)
    return _sha(data)


def slope_group_normals(pos, nor, local, outward, slope_deg):
    """Smoothing groups of one display primitive split at steep / gentle face boundaries. The detail's lip walls (~75 deg)
    and gentle faces met at shared vertices, whose smooth normals averaged across both and painted comb / fan streaks
    over the thin triangles. pos, nor: float32 glTF-frame positions / normals of the primitive; local: (n, 3) indices;
    outward: the patch outward normal (ENU). A face is steep when its winding normal makes more than slope_deg with
    outward; an edge is hard when its two faces differ in class. Around each vertex on a hard edge, the faces joined
    through its non-hard edges form groups; group 0 overwrites the vertex normal, further groups get copies of the
    vertex. Vertices with one group keep their normal. Returns (source vertex per output vertex, normals, local
    indices, stats); positions and every other attribute of a copy are its source's."""
    P = pos.astype(np.float64)
    I = np.asarray(local, np.int64)
    Pe = np.c_[P[:, 0], -P[:, 2], P[:, 1]]
    fn = np.cross(Pe[I[:, 1]] - Pe[I[:, 0]], Pe[I[:, 2]] - Pe[I[:, 0]])
    steep = (fn @ np.asarray(outward, float)) / np.maximum(np.linalg.norm(fn, axis=1), 1e-30) < np.cos(np.radians(slope_deg))
    ef = {}
    for fi, f in enumerate(I):
        for j in range(3):
            ef.setdefault((min(int(f[j]), int(f[(j + 1) % 3])), max(int(f[j]), int(f[(j + 1) % 3]))), []).append(fi)
    hard = {e for e, ff in ef.items() if len(ff) == 2 and steep[ff[0]] != steep[ff[1]]}
    normals = nor.copy()
    newI = I.copy()
    extra_src, extra_nrm = [], []
    split = 0
    if hard:
        vf = {}
        for fi, f in enumerate(I):
            for v in f:
                vf.setdefault(int(v), []).append(fi)
        Nf = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
        for v in {x for e in hard for x in e}:
            fs = vf[v]
            parent = {f: f for f in fs}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            ebuck = {}
            for f in fs:
                for w in I[f]:
                    w = int(w)
                    if w != v:
                        ebuck.setdefault(w, []).append(f)
            for w, ff in ebuck.items():
                if (min(v, w), max(v, w)) in hard:
                    continue
                for f in ff[1:]:
                    parent[find(f)] = find(ff[0])
            comps = {}
            for f in fs:
                comps.setdefault(find(f), []).append(f)
            if len(comps) < 2:
                continue
            split += 1
            for gi, (_, ff) in enumerate(sorted(comps.items())):
                n = Nf[ff].sum(0)
                n = n / max(np.linalg.norm(n), 1e-30)
                if gi == 0:
                    normals[v] = n.astype(np.float32)
                    continue
                nid = len(P) + len(extra_src)
                extra_src.append(v)
                extra_nrm.append(n)
                for f in ff:
                    newI[f][newI[f] == v] = nid
    src = np.r_[np.arange(len(P)), np.asarray(extra_src, np.int64)]
    if extra_src:
        normals = np.concatenate([normals, np.asarray(extra_nrm, np.float32)])
    return src, normals, newI, {'faces': int(len(I)), 'steep_faces': int(steep.sum()), 'hard_edges': len(hard),
                                'vertices_split': split, 'copies_added': len(extra_src)}


def write_display_glb(mesh, codes, target, uv_scale=1., normal_groups=None, report=None, wall_view=None):
    """Welded, smooth-shaded, vertex-coloured primitive per region code; glTF frame (E, U, -N). normal_groups (opt-in, B
    scan-detail footprint only): (slope_deg, patch outward normal) -- the footprint primitives X#### (faces by
    xfer_faces membership, not by position) get slope_group_normals; stats per primitive go to
    report['normal_groups']. wall_view: the outward direction d of the wall. The bed mapping projects along d (axes:
    horizontal e_u normal to d, e_v = d x e_u) instead of along (1, -1, 0) -- (x + y, z) stretches ~2x on this
    roadcut's face and up to ~8x on the rock of the hillside above it; and the rock primitives' cover alpha is
    remapped by smoothstep(0.2, 0.5) so a rock face at the soil boundary (cover 0.5) shows no rock texture, like the
    soil face next to it (the per-face material switch otherwise leaves a stair-stepped texture edge)."""
    doc = {'asset': {'version': '2.0', 'generator': 'outcrop-dsgen-display'}, 'scene': 0,
           'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0, 'name': 'Outcrop'}],
           'meshes': [{'name': 'Outcrop', 'primitives': []}], 'materials': [],
           'accessors': [], 'bufferViews': [], 'buffers': []}
    blob = bytearray()

    def add(array, kind, component, target_kind):
        blob.extend(b'\0' * (-len(blob) % 4))
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': array.nbytes, 'target': target_kind})
        blob.extend(array.tobytes())
        acc = {'bufferView': len(doc['bufferViews']) - 1, 'componentType': component, 'count': len(array), 'type': kind}
        if kind == 'VEC3':
            acc.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        doc['accessors'].append(acc)
        return len(doc['accessors']) - 1

    to_gltf = lambda a: a[:, [0, 2, 1]] * np.array([1., 1., -1.])
    V, F = mesh['vertices'], mesh['faces']
    # B scan-detail footprint (blockface 'detail'): its faces form their own primitive per code,
    # 'X####', with the donor texture coordinates (TEXCOORD_1) and blend weight (TEXCOORD_2 x); the
    # default mapping (TEXCOORD_0) is that of code ####, so w = 0 reproduces the C#### material.
    xf = mesh.get('xfer_faces')
    parts = [(code, False) for code in sorted(set(codes.tolist()))] if xf is None else \
        [(code, x) for code in sorted(set(codes.tolist())) for x in (False, True) if ((codes == code) & (xf == x)).any()]
    for code, is_x in parts:
        sel = F[(codes == code) & ((xf == is_x) if xf is not None else True)]
        used, local = np.unique(sel, return_inverse=True)
        local = local.reshape(-1, 3)
        pos = to_gltf(V[used]).astype(np.float32)
        nor = to_gltf(mesh['normals'][used]).astype(np.float32)
        src = None
        if is_x and normal_groups is not None:
            src, nor, local, st_ = slope_group_normals(pos, nor, local, normal_groups[1], normal_groups[0])
            pos = pos[src]
            used = used[src]
            if report is not None:
                report.setdefault('normal_groups', {'slope_deg': float(normal_groups[0]), 'primitives': {}})['primitives']['X%04d' % code] = st_
        # Alpha carries soil/vegetation cover; the rock material fades its texture detail with it.
        col = np.c_[mesh['colors'][used], mesh['cover'][used] if 'cover' in mesh else np.zeros(len(used))].astype(np.float32)
        if wall_view is not None:
            t_ = np.clip((col[:, 3] - .2) / .3, 0, 1)
            col[:, 3] = (t_ * t_ * (3 - 2 * t_)).astype(np.float32)
        def bed_uv(P):
            if wall_view is None:
                uv_ = np.stack([P[:, 0] + P[:, 1], P[:, 2]], 1) * uv_scale
            else:
                d_ = np.asarray(wall_view, float) / np.linalg.norm(wall_view)
                eu = np.cross([0., 0., 1.], d_)
                eu /= np.linalg.norm(eu)
                ev = np.cross(d_, eu)
                uv_ = np.stack([P @ eu, P @ ev], 1) * uv_scale
            if 'riser_uv' in mesh:
                # B risers (blockface riser_uv='plane'): metres along the riser's own plane instead of
                # the world (x + y, z) projection, which stretches steep risers up to ~9x
                own = mesh['riser_uv'][used]
                has = np.isfinite(own[:, 0])
                uv_[has] = own[has] * uv_scale
            # Per-bed rotation and offset of the planar mapping, so scanned-texture tiling does not
            # line up into a regular dot pattern along the beds.
            g = np.random.default_rng(int(code) * 7919 + 13)
            a = g.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            return (uv_ @ rot.T + g.uniform(0, 10, 2)).astype(np.float32)
        uv = bed_uv(V[used])
        doc['materials'].append({'name': ('X%04d' if is_x else 'C%04d') % code, 'doubleSided': True,
                                 'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0., 'roughnessFactor': 1.}})
        attributes = {'POSITION': add(pos, 'VEC3', 5126, 34962), 'NORMAL': add(nor, 'VEC3', 5126, 34962),
                      'COLOR_0': add(col, 'VEC4', 5126, 34962), 'TEXCOORD_0': add(uv, 'VEC2', 5126, 34962)}
        if is_x:
            attributes['TEXCOORD_1'] = add(mesh['xfer_uv'][used].astype(np.float32), 'VEC2', 5126, 34962)
            # y: footprint material 'lum-2' fill code (xfermat.fill_codes; opt-in), max over the vertex's faces
            fy = np.zeros(len(used), np.float32)
            if 'xfer_fill' in mesh:
                fsel = np.flatnonzero((codes == code) & (xf == is_x))
                np.maximum.at(fy, local.reshape(-1), np.repeat(mesh['xfer_fill'][fsel].astype(np.float32), 3))
            attributes['TEXCOORD_2'] = add(np.c_[mesh['xfer_w'][used], fy].astype(np.float32), 'VEC2', 5126, 34962)
            if 'xfer_ref' in mesh:
                # position (final minus the scan-detail displacement); used only for the core's bed lightness
                attributes['TEXCOORD_3'] = add(bed_uv(mesh['xfer_ref'][used]), 'VEC2', 5126, 34962)
            if 'xfer_uv3' in mesh:
                # TEXCOORD_1) and the relative weights of classes 1, 2 (class 0 = 1 - both)
                attributes['TEXCOORD_4'] = add(mesh['xfer_uv3'][used, 1].astype(np.float32), 'VEC2', 5126, 34962)
                attributes['TEXCOORD_5'] = add(mesh['xfer_uv3'][used, 2].astype(np.float32), 'VEC2', 5126, 34962)
                attributes['TEXCOORD_6'] = add(mesh['xfer_cw'][used, 1:3].astype(np.float32), 'VEC2', 5126, 34962)
        doc['meshes'][0]['primitives'].append({
            'attributes': attributes,
            'indices': add(local.astype(np.uint32).reshape(-1), 'SCALAR', 5125, 34963), 'material': len(doc['materials']) - 1})
    blob.extend(b'\0' * (-len(blob) % 4))
    doc['buffers'].append({'byteLength': len(blob)})
    text = json.dumps(doc, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 4)
    data = struct.pack('<4sII', b'glTF', 2, 28 + len(text) + len(blob)) + struct.pack('<II', len(text), 0x4E4F534A) + text + struct.pack('<II', len(blob), 0x004E4942) + bytes(blob)
    Path(target).write_bytes(data)
    return _sha(data)


def xfer_material_request(out, detail, display_stats, request, uv_scale, code):
    from . import xfermat
    mat = detail['material']
    dz = display_stats['blockface']['detail']
    fill = Path(mat['fill_texture']).resolve()
    out = Path(out).resolve()
    bed = request['textures'][request['slot_texture'][str(code)]]
    alb = sorted(fill.glob('*_diff_*.png'))[0]
    to_ue = lambda v: [float(v[0]), float(-v[1]), float(v[2])]
    nb = np.asarray(dz['material_fill']['bedding_normal'], float)
    s = np.cross([0., 0., 1.], nb)
    s /= np.linalg.norm(s)
    d = np.cross(nb, s)
    a1, a2 = np.asarray(dz['a1'], float), np.asarray(dz['a2'], float)
    frames = {'donor_t_ue': to_ue(a1), 'donor_b_ue': to_ue(-a2), 'bedding': {'nb_ue': to_ue(nb), 's_ue': to_ue(s), 'd_ue': to_ue(d)}, 'uv_scale': float(uv_scale)}
    if mat.get('recipe') in ('lum-3', 'lum-4', 'lum-5'):
        # the bed texture itself at the reference UV (TEXCOORD_3), donor lightness off (capture script)
        target = xfermat.bed_clamped_mean(bed)
        d8, dst = xfermat.donor_detail(detail['donor'], target)
        paths, stats = xfermat.lum3_maps(out / 'xfer_maps', bed, detail['donor'], d8, alb, sorted(fill.glob('*_nor_dx_*.png'))[0], float(mat['fill_tile_m']),
                                         request['detail_strength'], request['normal_strength'], float(uv_scale), float(dz['W_m']),
                                         sigma_low_mm=float(mat.get('sigma_low_mm', 46.8)), sigma_c_mm=float(mat.get('sigma_c_mm', 4.)),
                                         mid_weight=float(mat.get('mid_weight', 0.)))
        (out / 'xfer_maps' / 'maps.json').write_text(json.dumps({'bed_clamped_mean': target, 'donor_detail': dst, **stats, 'paths': paths,
                                                                 'bed_texture': bed['id'], 'code': code}, indent=1), encoding='utf8')
        req = {'recipe': mat['recipe'], 'detail': paths['detail'], 'normal': paths['normal'], 'normal_maps': True, 'bed_low': paths['bed_low'],
               'bed_low_normal': paths['bed_low_normal'], 'soft_range': stats['soft_range'],
               'fill': {'detail': paths['fill_detail'], 'normal': paths['fill_normal'], 'tile_m': float(mat['fill_tile_m'])}, **frames}
        if 'bed_lowmid' in paths:
            req.update(bed_lowmid=paths['bed_lowmid'], bed_mid=paths['bed_mid'])
        return req
    paths, stats = xfermat.write_maps(out / 'xfer_maps', detail['donor'], bed, request['detail_strength'], float(dz['W_m']), mat.get('match'))
    lum_mean, mu, k = xfermat.fill_lum_mean(alb, stats['bed_clamped_mean'])
    (out / 'xfer_maps' / 'maps.json').write_text(json.dumps({**stats, 'paths': paths, 'bed_texture': bed['id'], 'code': code}, indent=1), encoding='utf8')
    return {'recipe': mat.get('recipe', 'lum-2'), 'detail': paths['detail'], 'normal': paths['normal'], 'normal_maps': True,
            'donor_t_ue': to_ue(a1), 'donor_b_ue': to_ue(-a2),
            'fill': {'albedo': str(alb), 'normal': str(sorted(fill.glob('*_nor_dx_*.png'))[0]), 'tile_m': float(mat['fill_tile_m']),
                     'lum_mean': lum_mean, 'lum_mean_raw': mu, 'k': k, 'clamped_mean_target': stats['bed_clamped_mean']},
            'bedding': {'nb_ue': to_ue(nb), 's_ue': to_ue(s), 'd_ue': to_ue(d)}, 'uv_scale': float(uv_scale)}


def wall_material_request(out, detail, display_stats, request, uv_scale, codes):
    """Capture-request keys of the whole-wall scene: recipe lum-5 per bed texture with the donor atlas."""
    from . import xfermat
    mat = detail['material']
    bf = display_stats['blockface']
    fill = Path(mat['fill_texture']).resolve()
    out = Path(out).resolve()
    atlas = Path(detail['donor_atlas']).resolve()
    ainfo = json.loads((atlas / 'atlas.json').read_text(encoding='utf8'))
    to_ue = lambda v: [float(v[0]), float(-v[1]), float(v[2])]
    nb = np.asarray(bf['bedding_normal'], float)
    nb = nb if nb[2] >= 0 else -nb
    s_ = np.cross([0., 0., 1.], nb)
    s_ /= np.linalg.norm(s_)
    d_ = np.cross(nb, s_)
    sl, sc = float(mat.get('sigma_low_mm', 46.8)), float(mat.get('sigma_c_mm', 4.))
    npaths, nst = xfermat.atlas_normal_split(out / 'xfer_maps' / 'atlas', atlas / 'normal_dx.png', float(ainfo['texel_mm']), sl, sc)
    alb = sorted(fill.glob('*_diff_*.png'))[0]
    nor = sorted(fill.glob('*_nor_dx_*.png'))[0]
    beds, bstats = {}, {}
    for code in codes:
        ti = request['slot_texture'][str(code)]
        if str(ti) in beds:
            continue
        bed = request['textures'][ti]
        paths, st = xfermat.bed_wall_maps(out / 'xfer_maps' / ('bed_%d' % ti), bed, alb, nor, float(mat['fill_tile_m']), request['detail_strength'],
                                          request['normal_strength'], float(uv_scale), nst['coarse_std_slope'], sl, sc,
                                          float(mat.get('mid_weight', .7071067811865476)))
        beds[str(ti)] = {**paths, 'donor_slope_gain': st['donor_slope_gain'], 'bed_texture': bed['id']}
        bstats[str(ti)] = st
    (out / 'xfer_maps' / 'maps.json').write_text(json.dumps({'atlas_normal_split': {**nst, 'paths': npaths}, 'beds': bstats,
                                                              'codes': [int(c) for c in codes]}, indent=1), encoding='utf8')
    a1, a2 = np.asarray(bf['a1'], float), np.asarray(bf['a2'], float)
    return {'recipe': mat['recipe'], 'atlas': True, 'albedo': str(atlas / 'albedo.png'), 'normal': str(atlas / 'normal_dx.png'),
            'arm': str(atlas / 'arm.png'), 'normal_fine': npaths['fine'], 'normal_coarse': npaths['coarse'], 'normal_scale': nst['scale'],
            'normal_maps': True, 'soft_range': list(xfermat.LUM3_SOFT_RANGE), 'beds': beds,
            'fill': {'tile_m': float(mat['fill_tile_m'])}, 'donor_t_ue': to_ue(a1), 'donor_b_ue': to_ue(-a2),
            'bedding': {'nb_ue': to_ue(nb), 's_ue': to_ue(s_), 'd_ue': to_ue(d_)}, 'uv_scale': float(uv_scale)}


TEXTURE_ROOT = Path(__file__).resolve().parents[2] / 'assets/realistic_outcrop/catalog'
# them into detail maps so bed colour still comes from the randomised vertex tint.
TEXTURES = [
    {'id': 'sandstone_public_a', 'tile_m': 2.0, 'mean_linear': [.6442, .3626, .1805]},
    {'id': 'sandstone_public_b', 'tile_m': 1.5, 'mean_linear': [.2480, .1385, .0676]},
    {'id': 'mudstone_public_a', 'tile_m': 1.5, 'mean_linear': [.1087, .0919, .0715]},
    {'id': 'granite_public_a', 'tile_m': 2.0, 'mean_linear': [.1813, .0969, .0590]},
    {'id': 'volcanic_public_a', 'tile_m': 1.5, 'mean_linear': [.2316, .1541, .0956]},
    {'id': 'volcanic_public_b', 'tile_m': 2.0, 'mean_linear': [.0637, .0541, .0396]},
    {'id': 'mafic_public_a', 'tile_m': 1.5, 'mean_linear': [.2790, .2107, .1355]},
    {'id': 'foliated_public_a', 'tile_m': 1.5, 'mean_linear': [.0155, .0117, .0089]},
    {'id': 'foliated_public_b', 'tile_m': 2.0, 'mean_linear': [.0432, .0341, .0253]},
    {'id': 'marble_public_a', 'tile_m': 2.0, 'mean_linear': [.5483, .3920, .2828]},
    {'id': 'marble_public_b', 'tile_m': 1.5, 'mean_linear': [.3362, .2769, .2110]},
]
# Small floor probability (about 10% in total) for every texture on every lithology: texture alone never
# identifies the class.
TEXTURE_FLOOR = .01


def texture_assignment(book, rng):
    """One scanned texture per body, drawn from the catalog's lithology preference.
    Returns (slot -> index into the scene's texture list, the scene's texture records)."""
    from .lithology import get
    ids = [t['id'] for t in TEXTURES]
    by_body = {}
    for e in book:
        if e['body'] not in by_body:
            pref = get(e['lithology'])['textures']
            w = np.array([pref.get(i, 0.) for i in ids]) + TEXTURE_FLOOR
            by_body[e['body']] = int(rng.choice(len(TEXTURES), p=w / w.sum()))
    used = sorted(set(by_body.values()))
    local = {g: k for k, g in enumerate(used)}
    return {str(e['code']): local[by_body[e['body']]] for e in book}, [TEXTURES[g] for g in used]


def _ground_colors(rng):
    base = np.array([[.30, .25, .19], [.36, .33, .27], [.27, .27, .22], [.40, .34, .26]])[rng.integers(4)]
    base = base * rng.uniform(.8, 1.2)
    return np.clip(base, .05, .9), np.clip(base * rng.uniform(.6, .8), .03, .9)


def enu_to_ue_vec(v):
    return [float(v[0]), -float(v[1]), float(v[2])]


def enu_to_ue_pos(p):
    return [100. * float(p[0]), -100. * float(p[1]), 100. * float(p[2])]


def _exposure_frame(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    centroid = (tri.mean(1) * area[:, None]).sum(0) / area.sum()
    mean_normal = cross.sum(0)
    used = vertices[np.unique(faces)]
    _, _, vt = np.linalg.svd(used - used.mean(0), full_matrices=False)
    normal = vt[2]
    if np.dot(normal, mean_normal) < 0:
        normal = -normal
    # Prefer a sub-horizontal viewing direction even for gently dipping exposures.
    horizontal = normal - np.array([0, 0, normal[2]])
    if np.linalg.norm(horizontal) > .2:
        normal = horizontal / np.linalg.norm(horizontal)
    radius = float(np.linalg.norm(used - centroid, axis=1).max())
    return centroid, normal / np.linalg.norm(normal), radius, used


def sample_shots(pkg, sampling, seed, focus=None):
    """Cameras framed on the exposed rock, combined with an astronomical lighting plan."""
    from .lighting import plan
    rng = np.random.default_rng(seed)
    centroid, normal, radius, used = _exposure_frame(pkg['vertices'], pkg['faces'])
    if focus is not None and len(focus) > 10:
        # Frame the exposed (uncovered) rock rather than the whole terrain.
        used = focus[np.random.default_rng(seed + 3).choice(len(focus), min(len(focus), 20000), replace=False)]
        centroid = used.mean(0)
        radius = float(np.percentile(np.linalg.norm(used - centroid, axis=1), 95))
    view = sampling.get('view_direction') or pkg['model'].get('metadata', {}).get('view_direction')
    if view is not None:
        normal = np.asarray(view, float) / np.linalg.norm(view)  # outward: from rock towards the viewer
    w, h = sampling['resolution']
    sensor_w = sampling['sensor_width_mm']
    sensor_h = sensor_w * h / w
    up = np.array([0., 0., 1.])
    u = lambda key: float(rng.uniform(*sampling[key]))
    states, cam_index, n_cams = plan(sampling.get('lighting'), np.random.default_rng(seed + 101), sampling['shots'])
    # samples. Drawn from its own stream so the default framing stream of every seed is unchanged.
    fixed_range = sampling.get('camera_distance_m')
    range_rng = np.random.default_rng(seed + 211)
    if sampling.get('camera_profile', 'legacy') == 'field':
        return _field_shots(sampling, seed, states, cam_index, n_cams, centroid, normal, used, sensor_w, sensor_h, up)
    cameras = []
    for _ in range(n_cams):
        yaw, pitch = math.radians(u('yaw_deg')), math.radians(u('pitch_deg'))
        side = np.cross(up, normal)
        side /= np.linalg.norm(side)
        direction = normal * math.cos(yaw) + side * math.sin(yaw)
        direction = direction * math.cos(pitch) + up * math.sin(pitch)
        direction /= np.linalg.norm(direction)
        focal = u('focal_length_mm')
        half = min(math.atan(sensor_w / 2 / focal), math.atan(sensor_h / 2 / focal))
        fill = u('fill')
        # Jitter the aim point inside the exposure so close-ups are not always centred.
        aim = centroid + (used[rng.integers(len(used))] - centroid) * rng.uniform(0, .5) * (fill < .8)
        distance = radius * fill / math.tan(half) + .5
        if fixed_range is not None:
            distance = float(range_rng.uniform(*fixed_range))
        position = aim + direction * distance
        position[2] = max(position[2], float(sampling.get('min_camera_z_m', 1.7)))
        cameras.append(_camera(position, aim, up, focal, sensor_w, sensor_h))
    return _shots(cameras, states, cam_index, u('sky_intensity'))


# "field" camera profile: how people photograph outcrops in the field. Most shots from in front at
# weight, yaw range (deg from the exposure's front), horizontal distance (m; slant for 'high'),
# camera height above ground (m; elevation angle deg for 'high'), focal length (mm).
FIELD_CAMERAS = {
    'front_low': dict(weight=.55, yaw=(-35., 35.), dist=(15., 40.), height=(1.5, 2.2), focal=(24., 50.)),
    'wide': dict(weight=.20, yaw=(-45., 45.), dist=(45., 80.), height=(1.6, 8.), focal=(24., 35.)),
    'close': dict(weight=.15, yaw=(-30., 30.), dist=(5., 12.), height=(1.5, 2.), focal=(35., 70.)),
    'high': dict(weight=.10, yaw=(-40., 40.), dist=(30., 60.), elevation=(20., 35.), focal=(24., 50.)),
}


def _camera(position, aim, up, focal, sensor_w, sensor_h):
    forward = aim - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return {'position_enu_m': position.tolist(), 'target_enu_m': aim.tolist(),
            'position_cm': enu_to_ue_pos(position), 'forward': enu_to_ue_vec(forward),
            'right': enu_to_ue_vec(right), 'up': enu_to_ue_vec(cam_up),
            'focal_length_mm': focal, 'filmback_mm': [sensor_w, sensor_h], 'near_clip_cm': 10.}


def _shots(cameras, states, cam_index, sky_intensity):
    shots = []
    for i, (st, ci) in enumerate(zip(states, cam_index)):
        shots.append({**cameras[ci], 'id': 's%04d' % i, 'camera_index': ci,
                      'sun_ray_direction': enu_to_ue_vec(-np.asarray(st['to_sun_scene'])), 'sun_lux': st['sun_ue_lux'],
                      'moon_ray_direction': enu_to_ue_vec(-np.asarray(st['to_moon_scene'])), 'moon_lux': st['moon_ue_lux'],
                      'sky_intensity': sky_intensity, 'exposure_bias': st['exposure_bias'],
                      'auto_exposure_ev100': st['auto_exposure_ev100'], 'lighting': st})
    return shots


def _field_shots(sampling, seed, states, cam_index, n_cams, centroid, normal, used, sensor_w, sensor_h, up):
    """Cameras of the "field" profile; own stream (seed + 307), the kind recorded per camera."""
    g = np.random.default_rng(seed + 307)
    kinds = list(FIELD_CAMERAS)
    weights = np.array([FIELD_CAMERAS[k]['weight'] for k in kinds])
    ground = float(sampling.get('min_camera_z_m', 1.7)) - 1.7
    side = np.cross(up, normal)
    side /= np.linalg.norm(side)
    flat = normal - up * normal[2]
    flat /= np.linalg.norm(flat)
    cameras = []
    for _ in range(n_cams):
        kind = kinds[int(g.choice(len(kinds), p=weights / weights.sum()))]
        c = FIELD_CAMERAS[kind]
        yaw = math.radians(float(g.uniform(*c['yaw'])))
        horiz = flat * math.cos(yaw) + side * math.sin(yaw)
        # wide and high views aim at the exposure centre, the others at a point of exposed rock
        aim = centroid.copy() if kind in ('wide', 'high') else used[int(g.integers(len(used)))].copy()
        d = float(g.uniform(*c['dist']))
        if kind == 'high':
            el = math.radians(float(g.uniform(*c['elevation'])))
            position = aim + horiz * d * math.cos(el) + up * d * math.sin(el)
        else:
            position = aim + horiz * d
            position[2] = ground + float(g.uniform(*c['height']))
        position[2] = max(position[2], ground + 1.5)
        cam = _camera(position, aim, up, float(g.uniform(*c['focal'])), sensor_w, sensor_h)
        cam['camera_kind'] = kind
        cameras.append(cam)
    return _shots(cameras, states, cam_index, float(g.uniform(*sampling['sky_intensity'])))


def bootstrap_project(name, project=None, description='outcrop dsgen capture'):
    project = project or SHORT_ROOT / name / 'DS.uproject'
    if project.parent.exists():
        raise ValueError('UE project directory already exists: ' + str(project.parent))
    (project.parent / 'Config').mkdir(parents=True)
    # Lumen uses hardware ray tracing here, so mesh distance fields are not needed; skipping
    # their build roughly halves import time for the dense display meshes.
    ini = SETTINGS.read_text(encoding='utf8').replace('r.GenerateMeshDistanceFields=True', 'r.GenerateMeshDistanceFields=False')
    # Auto-exposure limits are given in EV100 (camera-like metering for the lighting plans).
    ini = ini.replace('[/Script/Engine.RendererSettings]',
                      '[/Script/Engine.RendererSettings]\nr.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange=True', 1)
    (project.parent / 'Config/DefaultEngine.ini').write_text(ini, encoding='utf8')
    document = _read(TEMPLATE)
    document['Description'] = description
    project.write_text(json.dumps(document, indent=2), encoding='utf8')
    return project


def _truth_classifier(pkg, truth):
    """Section-like view of v2 truth for display.build: classify(points) -> index into the growing
    body/unit/lithology lists of the returned package dict (-1 outside rock)."""
    lists = {'body': [], 'unit': [], 'lithology': []}
    index = {}

    def classify(points):
        st, b, u, l = truth.query(points)
        out = np.full(len(points), -1, np.int64)
        for i, (s_, bb, uu, ll) in enumerate(zip(st, b, u, l)):
            if s_ != 'occupied':
                continue
            k = (bb, uu, ll)
            if k not in index:
                index[k] = len(lists['body'])
                lists['body'].append(bb)
                lists['unit'].append(uu)
                lists['lithology'].append(ll)
            out[i] = index[k]
        return out
    view = dict(pkg, segments=np.zeros((0, 2, 3)), segment_ids=[], **lists)
    return view, classify


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_digest():
    """Every label depends on the v2 kernel: one combined digest over src/outcrop/modeling."""
    kernel = hashlib.sha256()
    modeling = Path(__file__).absolute().parent.parent / 'modeling'
    for f in sorted(f for f in modeling.rglob('*') if f.suffix in ('.py', '.cpp')):
        kernel.update(str(f.relative_to(modeling)).replace('\\', '/').encode() + b'\0' + f.read_bytes())
    return kernel.hexdigest()


def provenance(package, exposure, seed, sampling):
    """Everything needed to reproduce a scene: geology spec, exposure, process seeds, code."""
    here = Path(__file__).absolute().parent
    code = {p.name: _sha_file(p) for p in sorted(here.glob('*.py'))}
    script = Path(__file__).absolute().parents[3] / 'scripts' / 'dsgen_ue_capture.py'
    code[script.name] = _sha_file(script)
    code['outcrop.modeling (combined)'] = kernel_digest()
    model = _read(Path(package) / 'model_spec.json')
    return {'geology': {'seed': model.get('metadata', {}).get('seed'),
                        'model_spec_sha256': _sha_file(Path(package) / 'model_spec.json'),
                        'package': str(Path(package).absolute())},
            'exposure': exposure, 'stage_seed': seed,
            'sampling': {k: v for k, v in sampling.items() if k != 'lighting'}, 'lighting': sampling.get('lighting'),
            'code_sha256': code}


def stage(package, out, *, seed, sampling=None, package_id='p0000', exposure=None):
    """Stage one exposure of a v2 geology package. ``exposure``: kind ('bluff', 'escarpment', ...)
    for sandbox-tile packages; the same package can be staged with several exposures."""
    sampling = {**DEFAULT_SAMPLING, **(sampling or {})}
    import time as _time
    _ts = [('start', _time.time())]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pkg = load_package(package)
    book, face_codes = codebook_for(pkg)
    glb = out / 'scene.glb'
    line_classes = classify_interfaces(pkg['model'], pkg['segment_ids'])
    display_stats = None
    meta = pkg['model'].get('metadata', {})
    expo = None
    if 'tile' in meta:
        from .exposure import sample_exposure
        expo = sample_exposure(exposure or sampling.get('exposure') or 'escarpment', pkg['model'], seed=sampling.get('exposure_seed'))
    elif exposure:
        raise ValueError('package predates the geology/exposure split; it has one fixed exposure')
    bluff_env = expo if expo is not None and expo['kind'] == 'bluff' else meta.get('bluff')
    if sampling['display']:
        from .display import build
        _, outward, _, _ = _exposure_frame(pkg['vertices'], pkg['faces'])
        from .truth import TruthModel
        truth = TruthModel(package)
        if sampling.get('cm') is not None:
            truth.workers = int(sampling['cm'].get('truth_workers', 12))
        line_classes.update(classify_interfaces(pkg['model'], [iid for iid, _, _ in truth.interfaces]))
        rng = np.random.default_rng(seed + 7919)
        if expo is not None and not sampling.get('legacy_display'):
            # Sandbox tile: exposure patches -> shared surface processes -> v2 labels on the final surface.
            from .exposure import build_surface
            from .processes import erode
            surf = build_surface(expo, samples=tuple(sampling['bluff_rays']), max_edge=sampling['sheet_edge_m'])
            cm = sampling.get('cm')
            if cm is not None and abs(float(cm['sheet_edge_m']) - float(sampling['sheet_edge_m'])) > 1e-12:
                raise ValueError('cm.sheet_edge_m must equal sampling.sheet_edge_m (smoothing scale)')
            _ts.append(('surface', _time.time()))
            mesh = erode(surf, pkg['model'], truth, rng, clay=bool(sampling.get('clay')), tint_jitter=sampling['tint_jitter'],
                         fracture=sampling.get('fracture'), blockface=sampling.get('blockface'), cm=cm)
            _ts.append(('erode', _time.time()))
        elif bluff_env:
            from .morph import build_bluff
            mesh = build_bluff(pkg, truth, rng, env=bluff_env, samples=tuple(sampling['bluff_rays']), clay=bool(sampling.get('clay')))
        elif expo is not None:
            # The node-grid display layer only serves packages from before the geology/exposure split.
            raise ValueError('legacy_display needs a pre-split package; sandbox escarpments use the profile sheet')
        else:  # legacy escarpment package (v2 cliff section)
            if meta.get('surface_up') is not None:
                outward = np.asarray(meta['surface_up'], float)
            mesh = build(pkg, face_codes, line_classes, outward, rng,
                         max_edge=sampling['max_edge_m'], grid=meta.get('cliff_grid'),
                         ext=meta.get('terrain_ext_nodes'), ext_offset=tuple(meta.get('terrain_ext_offset', (0, 0))),
                         truth=truth)
        if bluff_env:
            sampling['min_camera_z_m'] = bluff_env['ground_z_m'] + 1.7
        elif expo is not None:
            sampling['min_camera_z_m'] = expo['ground_z_m'] + 1.7
        # Region codes from the exact v2 answer at each displayed face (not from the source section).
        keys = sorted({k for k in mesh['face_keys'] if k is not None}, key=lambda k: tuple('' if x is None else x for x in k))
        from .lithology import get as lith_get
        book = [dict(code=i + 1, body=b, unit=u, lithology=l, lithology_code=lith_get(l)['code'], rock_family=lith_get(l)['family'],
                     lineage=truth.lineage.get(b, [])) for i, (b, u, l) in enumerate(keys)]
        index = {k: i + 1 for i, k in enumerate(keys)}
        # Visible semantics apart from geology: bedrock faces carry v2 region codes, every other
        # visible class (soil on gentle rock, ground, road, water, ...) a reserved semantic code.
        from .semantics import ROCK_CODE_MAX, code_of
        if len(keys) > ROCK_CODE_MAX:
            raise ValueError('too many region labels for the reserved semantic code range')
        sub_codes = np.array([index[k] if k is not None else 0 for k in mesh['face_keys']], np.int64)
        sem = mesh.get('semantic')
        if sem is None:
            sem = np.where(mesh['cover_faces'], 'soil', 'bedrock').astype(object)
        for cls in sorted(set(sem.tolist()) - {'bedrock'}):
            sub_codes[sem == cls] = code_of(cls)
        sub_codes[sub_codes == 0] = code_of('soil')  # rock-surface faces outside v2 rock: soil
        tex_rng = np.random.default_rng(seed + 41)
        uv_scale = float(tex_rng.uniform(.6, 1.6))
        if sampling.get('texture_library'):
            uv_scale = 1.
        # B scan-detail footprint (opt-in): steep / gentle smoothing groups of its display normals
        bfo = sampling.get('blockface') or {}
        ng = (bfo.get('detail') or {}).get('normal_groups') if 'xfer_faces' in mesh else None
        ng_report = {}
        ng_normal = None
        if ng:
            # whole-wall scene: the groups split about the wall's outward direction -m (no sample frame)
            ng_normal = -np.asarray(mesh['stats']['blockface']['m'], float) if bfo.get('mode') == 'wall' else np.asarray(bfo['frame']['normal'], float)
        wall_view = -np.asarray(mesh['stats']['blockface']['m'], float) if bfo.get('mode') == 'wall' else None
        glb_sha = write_display_glb(mesh, sub_codes, glb, uv_scale=uv_scale, wall_view=wall_view,
                                    normal_groups=(float(ng['slope_deg']), ng_normal) if ng else None,
                                    report=ng_report)
        _ts.append(('glb', _time.time()))
        if ng_report:
            mesh['stats']['blockface']['detail']['normal_groups'] = ng_report['normal_groups']
        lengths = [len(p) for _, p in mesh['polylines']]
        extra = {}
        if 'state' in mesh:
            # Surface state of the rock vertices (first rock_vertices of the mesh): recession m, cover.
            extra = dict(state=mesh['state'], state_fields=np.array(['recession_m', 'cover']), rock_vertices=mesh['rock_vertices'])
        if 'coplanar_faces' in mesh:
            # B patch, rule 2: faces lying in a v2 interface plane (index + 1 into display_coplanar.json)
            extra['coplanar_faces'] = mesh['coplanar_faces']
            (out / 'display_coplanar.json').write_text(json.dumps(mesh['coplanar_records'], indent=1), encoding='utf8')
        if 'riser_faces' in mesh:
            extra['riser_faces'] = mesh['riser_faces']  # B risers with their own plane texture mapping
        if 'relief_faces' in mesh:
            extra['relief_faces'] = mesh['relief_faces']  # B floor relief faces (relief.py)
        if 'xfer_faces' in mesh:
            # B scan-detail footprint (scandetail.py): donor-textured faces, per-vertex donor UV and weight
            extra.update(xfer_faces=mesh['xfer_faces'], xfer_uv=mesh['xfer_uv'], xfer_w=mesh['xfer_w'])
            if 'xfer_fill' in mesh:
                extra['xfer_fill'] = mesh['xfer_fill']  # footprint material 'lum-2' fill code per face (xfermat.py)
            if 'xfer_ref' in mesh:
                extra['xfer_ref'] = mesh['xfer_ref']  # 'lum-5' reference positions (final minus the scan-detail displacement)
            if 'xfer_uv3' in mesh:
                extra.update(xfer_uv3=mesh['xfer_uv3'], xfer_cw=mesh['xfer_cw'])
            if 'b_region' in mesh:
                extra.update(b_region=mesh['b_region'], b_cell=mesh['b_cell'])  # whole-wall scene: B region / cell per face  # whole-wall scene: atlas UV per colour class, class weights
        if 'b_region' in mesh and 'xfer_faces' not in mesh:
            extra.update(b_region=mesh['b_region'], b_cell=mesh['b_cell'], relief_dvec=mesh['relief_dvec'])
        np.savez_compressed(out / 'display.npz', vertices=mesh['vertices'], faces=mesh['faces'], codes=sub_codes,
                            polyline_points=np.concatenate([p for _, p in mesh['polylines']]) if lengths else np.zeros((0, 3)),
                            polyline_lengths=np.asarray(lengths, np.int64), **extra)
        (out / 'display_polylines.json').write_text(json.dumps([i for i, _ in mesh['polylines']]), encoding='utf8')
        if mesh.get('cm_shape_exceed'):
            np.savez_compressed(out / 'shape_exceed.npz', **{'%s__%s' % (k, f): v for k, d in mesh['cm_shape_exceed'].items() for f, v in d.items()})
        display_stats = mesh['stats']
        # Framing uses bedrock faces only (v2 rock, not covered; clay mode clears the cover colour).
        rock_faces = ~mesh['cover_faces'] & np.array([k is not None for k in mesh['face_keys']])
        exposed = np.unique(mesh['faces'][rock_faces])
        view_dir = (expo or {}).get('view_direction') or meta.get('view_direction')
        if bluff_env or expo is not None:
            # Frame the exposed rock: steep faces turned towards the viewing side (not the plain or
            # plateau, which in clay mode carry no soil cover to exclude them).
            nrm = mesh['normals'][exposed]
            keep = (nrm @ np.asarray(view_dir or [0., -1., 0.], float) > .2) & (np.abs(nrm[:, 2]) < .8)
            if keep.sum() > 100:
                exposed = exposed[keep]
        if view_dir is not None:
            sampling['view_direction'] = view_dir
        focus = mesh['vertices'][exposed]
        bounds_pts = mesh['vertices']
    else:
        glb_sha = write_glb(pkg['vertices'], pkg['faces'], face_codes, glb)
        bounds_pts = pkg['vertices'][pkg['faces']].reshape(-1, 3)
    shots = sample_shots(pkg, sampling, seed, focus=focus if sampling['display'] else None)
    slot_texture, scene_textures = texture_assignment(book, np.random.default_rng(seed + 43)) if sampling['display'] else (None, [])
    library_record = None
    if sampling['display'] and sampling.get('texture_library'):
        from . import texlib
        am = sampling['texture_library']
        lib = texlib.load(am['library'])
        texlib.verify(lib)
        slot_texture, scene_textures, assign_rec = texlib.assign([dict(e, lithology=_former(e['lithology'])) for e in book], lib, int(am.get('seed', 0)))
        library_record = {'library': str(Path(am['library']).absolute()), 'library_json_sha256': _sha_file(Path(am['library']) / 'library.json'),
                     'seed': int(am.get('seed', 0)), 'assignment': assign_rec, 'uv_scale': 1.}
        if am.get('variation'):
            vt = Path(am['library']) / '_shared' / 'variation.png'
            want = json.loads((Path(am['library']) / 'library.json').read_text(encoding='utf8')).get('shared', {}).get('variation.png', {}).get('sha256')
            if want is None or _sha_file(vt) != want:
                raise ValueError('texture library variation map missing or changed: %s' % vt)
            library_record['variation'] = {'texture': str(vt.absolute()), 'period_m': float(am['variation'].get('period_m', 6.4)),
                                      'amplitude': float(am['variation'].get('amplitude', .1))}
    colors = {}
    for entry in book:
        pair = LITHOLOGY_COLORS.get(_label(entry['lithology']))   # a former name (old scene, large_cliff_scene) keeps its colour
        if pair is None:
            digest = hashlib.sha256(entry['lithology'].encode()).digest()
            base = [.25 + .5 * digest[k] / 255 for k in range(3)]
            pair = (base, [c * .7 for c in base])
        colors[str(entry['code'])] = {'a': list(pair[0]), 'b': list(pair[1])}
    if bluff_env:
        ground_z = bluff_env['ground_z_m'] - 1.5
    elif expo is not None:
        ground_z = expo['ground_z_m'] + .07
    else:
        ground_z = float(bounds_pts[:, 2].min()) + (-.03 if meta.get('surface_up') else
                                                    float(np.random.default_rng(seed + 17).uniform(*sampling['ground_burial_m'])))
    request = {
        'schema': 'outcrop-dsgen-capture-1', 'package': str(Path(package).absolute()), 'package_id': package_id,
        'seed': seed, 'directory': str(out.absolute()), 'glb': str(glb.absolute()), 'glb_sha256': glb_sha,
        'content_root': '/Game/DS/' + package_id, 'codebook': book, 'colors': colors,
        'mode': sampling.get('mode', 'final'),
        'resolution': sampling['resolution'], 'engine_warmup': sampling['engine_warmup'],
        'render_warmup': sampling['render_warmup'], 'shots': shots,
        'expected_bounds_cm': [enu_to_ue_pos(bounds_pts.min(0)), enu_to_ue_pos(bounds_pts.max(0))],
        'line_classes': line_classes, 'display': bool(sampling['display']), 'display_stats': display_stats,
        'textures': ([dict(t) for t in scene_textures] if library_record else
                     [{**t, 'albedo': str(TEXTURE_ROOT / t['id'] / 'albedo.png'), 'normal': str(TEXTURE_ROOT / t['id'] / 'normal.png')}
                      for t in scene_textures]) if sampling['display'] else None,
        'slot_texture': slot_texture if sampling['display'] else None,
        # Clay mode: geometry only (uniform grey, no texture detail or texture normals).
        'detail_strength': 0. if sampling.get('clay') else float(np.random.default_rng(seed + 47).uniform(.55, .95)),
        'normal_strength': 0. if sampling.get('clay') else float(np.random.default_rng(seed + 53).uniform(.5, 1.)),
        'ground_z_cm': 100. * ground_z,
        'ground_colors': [list(c) for c in _ground_colors(np.random.default_rng(seed + 23))],
        'semantic_codes': {str(c): cls for c, cls in __import__('outcrop.dsgen.semantics', fromlist=['RESERVED']).RESERVED.items()},
        'exposure': expo,
        'provenance': provenance(package, expo, seed, sampling),
    }
    if sampling.get('label_options'):
        request['label_options'] = dict(sampling['label_options'])
    if sampling.get('cm') is not None:
        _ts.append(('request', _time.time()))
        request['stage_timing_s'] = [(k, t - _ts[0][1]) for k, t in _ts]
    if library_record is not None:
        request['material_recipe'] = 'library-rgb'
        request['texture_library'] = library_record
        if library_record.get('variation'):
            request['texture_variation'] = library_record['variation']
        soil = [e for e in lib['entries'] if 'soil' in e['lithologies']] if sampling.get('texture_library', {}).get('soil') else []
        if soil:
            e_ = soil[0]
            alb_ = 'albedo_cal.png' if 'albedo_cal.png' in e_['files'] else 'albedo.png'
            request['texture_soil'] = {'id': e_['id'], 'albedo': str(Path(e_['dir']) / alb_), 'normal': str(Path(e_['dir']) / 'normal_dx.png'),
                                  'tile_m': float(e_['tile_m'])}
            library_record['soil'] = e_['id']
    detail = (sampling.get('blockface') or {}).get('detail')
    if sampling['display'] and detail and detail.get('textures', True):
        donor = Path(detail['donor'])
        request['xfer'] = {'albedo': str(donor / 'albedo.png'), 'normal': str(donor / 'normal_dx.png'), 'arm': str(donor / 'arm.png')}
        mat = detail.get('material')
        if mat and (sampling.get('blockface') or {}).get('mode') == 'wall':
            request['xfer'] = wall_material_request(out, detail, display_stats, request, uv_scale, sorted(set(sub_codes[mesh['xfer_faces']].tolist())))
        elif mat:
            xcode = int(np.bincount(sub_codes[mesh['xfer_faces']]).argmax())
            if detail['material'].get('recipe') in ('lum-3', 'lum-4', 'lum-5') and len(np.unique(sub_codes[mesh['xfer_faces']])) > 1:
                raise ValueError('lum-3: the footprint carries more than one region code (one bed low part per material)')
            request['xfer'].update(xfer_material_request(out, detail, display_stats, request, uv_scale, xcode))
    (out / 'request.json').write_text(json.dumps(request, indent=2), encoding='utf8')
    return request
