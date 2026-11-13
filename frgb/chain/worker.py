"""Resident Blender Cycles render worker for the fracture RGB study.

Run: blender -b --factory-startup --python worker.py -- QUEUE_DIR --samples N [--device OPTIX|CUDA|CPU] [--aov]
[--filter W]

Denoising, adaptive sampling and sample clamping are off; the integrator settings below are fixed and read back with
every frame. A job moves to QUEUE/running, then its result is written to QUEUE/done/<id>.json, or
QUEUE/failed/<id>.json with the traceback. QUEUE/stop ends the worker after the current job. QUEUE/worker.json is the
heartbeat. - A job names its mesh (native surface + display normals); the mesh is rebuilt only when that key changes.
- Output per job: one multilayer EXR (Combined linear RGBA, Depth, IndexMA; AOV stain_m with --aov) and the scene
readback taken from the actual Blender objects after the render (objects, Sun, sky, materials, settings, geometry). -
Check sessions only: --aov adds AOVs stain_m (M) and stain_c (stained BaseColor); --filter W overrides the pixel
filter width (0.01 = pixel-centre sampling for per-pixel comparison with the CPU raster). The readback then gains
'aerial'. Without them a job is not changed; the vertical-view camera rule is in look(). Coordinates:
Blender world = source ENU metres (x East, y North, z Up)."""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

GRAY = .22
ROUGHNESS = .9
VERTICAL_UP_SWITCH = 0.99
# Ground: material pass_index 900 (rock codes are >= 1, sky has IndexMA 0).
GROUND_PASS_INDEX = 900
GROUND_COLOR_SRGB8 = (155, 139, 110)                   # ground_color of outcrop/dsgen/generation_config.py
GROUND_SPECULAR_IOR_LEVEL = 0.
STAIN_SCALARS = ('StainWeight', 'StainP0X', 'StainP0Y', 'StainP0Z', 'StainNX', 'StainNY', 'StainNZ', 'StainTX', 'StainTY',
                 'StainTZ', 'StainAmp', 'StainLambda', 'StainPhase', 'StainWidth', 'StainSoft', 'StainFR', 'StainFG', 'StainFB')
INTEGRATOR = dict(max_bounces=8, diffuse_bounces=6, glossy_bounces=4, transmission_bounces=0, volume_bounces=0,
                  transparent_max_bounces=0, sample_clamp_direct=0., sample_clamp_indirect=0., caustics_reflective=False,
                  caustics_refractive=False, blur_glossy=0., pixel_filter_type='BLACKMAN_HARRIS', filter_width=1.5,
                  use_denoising=False, use_adaptive_sampling=False, sampling_pattern='TABULATED_SOBOL', use_light_tree=True)
STATE = dict(mesh_key=None, obj=None, materials={}, jobs=0, samples=None, device=None, aov=False, filter=None)


def parse(argv):
    args = dict(queue=Path(argv[0]), samples=None, device='OPTIX', aov=False, filter=None)
    i = 1
    while i < len(argv):
        if argv[i] == '--samples': args['samples'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--device': args['device'] = argv[i + 1]; i += 2
        elif argv[i] == '--aov': args['aov'] = True; i += 1
        elif argv[i] == '--filter': args['filter'] = float(argv[i + 1]); i += 2
        else: raise SystemExit('unknown argument ' + argv[i])
    if not args['samples'] or args['samples'] <= 0:
        raise SystemExit('--samples N required')
    return args


def configure(samples, device, aov, filter_width=None):
    scene = bpy.context.scene
    for o in list(scene.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scene.render.engine = 'CYCLES'
    if device == 'CPU':
        scene.cycles.device = 'CPU'
    else:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = device
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == device
        if not any(d.use for d in prefs.devices):
            raise RuntimeError('no %s device' % device)
        scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    for key, value in INTEGRATOR.items():
        setattr(scene.cycles, key, value)
    if filter_width is not None:
        scene.cycles.filter_width = float(filter_width)
    scene.render.use_persistent_data = True
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.
    scene.view_settings.gamma = 1.
    layer = scene.view_layers[0]
    layer.use_pass_combined = True
    layer.use_pass_z = True
    layer.use_pass_material_index = True
    if aov:
        for name, kind in (('stain_m', 'VALUE'), ('stain_c', 'COLOR')):
            a = layer.aovs.add(); a.name = name; a.type = kind
    s = scene.render.image_settings
    s.file_format = 'OPEN_EXR_MULTILAYER'; s.color_depth = '32'; s.exr_codec = 'ZIP'
    STATE.update(samples=samples, device=device, aov=aov, filter=filter_width)


def look(obj, forward, up_world=None):
    """up_world None (the camera): world z, or world +y (north) when |forward z| > VERTICAL_UP_SWITCH - the same rule as
    frgb/raster.py::camera. The Sun passes its own up_world."""
    fwd = Vector(forward).normalized()
    if up_world is None:
        up_world = (0., 0., 1.) if abs(fwd.z) <= VERTICAL_UP_SWITCH else (0., 1., 0.)
    right = fwd.cross(Vector(up_world)).normalized()
    up = right.cross(fwd)
    loc = obj.location.copy()
    obj.matrix_world = Matrix(((right.x, up.x, -fwd.x, loc.x), (right.y, up.y, -fwd.y, loc.y),
                               (right.z, up.z, -fwd.z, loc.z), (0., 0., 0., 1.)))


# ---------------------------------------------------------------- materials
def _math(nodes, op, a=None, b=None, value=None):
    n = nodes.new('ShaderNodeMath'); n.operation = op; n.use_clamp = False
    for i, x in ((0, a), (1, b)):
        if isinstance(x, (int, float)): n.inputs[i].default_value = float(x)
    if value is not None: n.inputs[1].default_value = value
    return n


def code_material(code, stain_eligible):
    """Principled BSDF, base 0.22, roughness 0.9; the stain graph (weight 0 = off) on stain-eligible codes."""
    mat = bpy.data.materials.new('Code%d' % code)
    mat.use_nodes = True
    mat.pass_index = int(code)
    nt = mat.node_tree; nodes, links = nt.nodes, nt.links
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (GRAY, GRAY, GRAY, 1.)
    bsdf.inputs['Roughness'].default_value = ROUGHNESS
    bsdf.inputs['Metallic'].default_value = 0.
    m_out = None
    if stain_eligible:
        val = {}
        for name in STAIN_SCALARS:
            n = nodes.new('ShaderNodeValue'); n.name = n.label = name; n.outputs[0].default_value = 0.; val[name] = n.outputs[0]
        geo = nodes.new('ShaderNodeNewGeometry')
        scale = nodes.new('ShaderNodeVectorMath'); scale.operation = 'MULTIPLY'; scale.inputs[1].default_value = (100., -100., 100.)
        links.new(geo.outputs['Position'], scale.inputs[0])
        p0 = nodes.new('ShaderNodeCombineXYZ')
        for axis, name in zip('XYZ', ('StainP0X', 'StainP0Y', 'StainP0Z')): links.new(val[name], p0.inputs[axis])
        q = nodes.new('ShaderNodeVectorMath'); q.operation = 'SUBTRACT'
        links.new(scale.outputs[0], q.inputs[0]); links.new(p0.outputs[0], q.inputs[1])
        nvec = nodes.new('ShaderNodeCombineXYZ'); tvec = nodes.new('ShaderNodeCombineXYZ')
        for axis, a, b in zip('XYZ', ('StainNX', 'StainNY', 'StainNZ'), ('StainTX', 'StainTY', 'StainTZ')):
            links.new(val[a], nvec.inputs[axis]); links.new(val[b], tvec.inputs[axis])
        dn = nodes.new('ShaderNodeVectorMath'); dn.operation = 'DOT_PRODUCT'
        links.new(q.outputs[0], dn.inputs[0]); links.new(nvec.outputs[0], dn.inputs[1])
        dt = nodes.new('ShaderNodeVectorMath'); dt.operation = 'DOT_PRODUCT'
        links.new(q.outputs[0], dt.inputs[0]); links.new(tvec.outputs[0], dt.inputs[1])
        # d = dot(q,n) + A sin(6.28318531 dot(q,t)/lambda + phase)
        k = _math(nodes, 'MULTIPLY', None, 6.28318531); links.new(dt.outputs['Value'], k.inputs[0])
        kd = _math(nodes, 'DIVIDE'); links.new(k.outputs[0], kd.inputs[0]); links.new(val['StainLambda'], kd.inputs[1])
        ph = _math(nodes, 'ADD'); links.new(kd.outputs[0], ph.inputs[0]); links.new(val['StainPhase'], ph.inputs[1])
        sn = _math(nodes, 'SINE'); links.new(ph.outputs[0], sn.inputs[0])
        am = _math(nodes, 'MULTIPLY'); links.new(val['StainAmp'], am.inputs[0]); links.new(sn.outputs[0], am.inputs[1])
        d = _math(nodes, 'ADD'); links.new(dn.outputs['Value'], d.inputs[0]); links.new(am.outputs[0], d.inputs[1])
        a = _math(nodes, 'ABSOLUTE'); links.new(d.outputs[0], a.inputs[0])
        h = _math(nodes, 'MULTIPLY', None, .5); links.new(val['StainWidth'], h.inputs[0])
        # soft > 0: m = 1 - smoothstep(h-s, h+s, a), t = saturate((a-(h-s))/(2s)), smoothstep = t*t*(3-2t)
        e0 = _math(nodes, 'SUBTRACT'); links.new(h.outputs[0], e0.inputs[0]); links.new(val['StainSoft'], e0.inputs[1])
        num = _math(nodes, 'SUBTRACT'); links.new(a.outputs[0], num.inputs[0]); links.new(e0.outputs[0], num.inputs[1])
        den = _math(nodes, 'MULTIPLY', None, 2.); links.new(val['StainSoft'], den.inputs[0])
        t = _math(nodes, 'DIVIDE'); links.new(num.outputs[0], t.inputs[0]); links.new(den.outputs[0], t.inputs[1])
        t.use_clamp = True
        tt = _math(nodes, 'MULTIPLY'); links.new(t.outputs[0], tt.inputs[0]); links.new(t.outputs[0], tt.inputs[1])
        two_t = _math(nodes, 'MULTIPLY', None, 2.); links.new(t.outputs[0], two_t.inputs[0])
        three = _math(nodes, 'SUBTRACT', 3.); links.new(two_t.outputs[0], three.inputs[1])
        ss = _math(nodes, 'MULTIPLY'); links.new(tt.outputs[0], ss.inputs[0]); links.new(three.outputs[0], ss.inputs[1])
        m_soft = _math(nodes, 'SUBTRACT', 1.); links.new(ss.outputs[0], m_soft.inputs[1])
        # soft == 0: m = (a <= h); LESS_THAN gives a < h (equality has measure zero on a continuous surface)
        m_hard = _math(nodes, 'LESS_THAN'); links.new(a.outputs[0], m_hard.inputs[0]); links.new(h.outputs[0], m_hard.inputs[1])
        is_soft = _math(nodes, 'GREATER_THAN', None, 0.); links.new(val['StainSoft'], is_soft.inputs[0])
        m = nodes.new('ShaderNodeMix'); m.data_type = 'FLOAT'; m.clamp_factor = True
        links.new(is_soft.outputs[0], m.inputs['Factor']); links.new(m_hard.outputs[0], m.inputs[2]); links.new(m_soft.outputs[0], m.inputs[3])
        m_out = m.outputs[0]
        wm = _math(nodes, 'MULTIPLY'); links.new(val['StainWeight'], wm.inputs[0]); links.new(m_out, wm.inputs[1])
        f = nodes.new('ShaderNodeCombineColor')
        for ch, name in zip(('Red', 'Green', 'Blue'), ('StainFR', 'StainFG', 'StainFB')): links.new(val[name], f.inputs[ch])
        c = nodes.new('ShaderNodeRGB'); c.name = c.label = 'BaseC'; c.outputs[0].default_value = (GRAY, GRAY, GRAY, 1.)
        cf = nodes.new('ShaderNodeMix'); cf.data_type = 'RGBA'; cf.blend_type = 'MULTIPLY'; cf.inputs['Factor'].default_value = 1.
        links.new(c.outputs[0], cf.inputs[6]); links.new(f.outputs[0], cf.inputs[7])
        mix = nodes.new('ShaderNodeMix'); mix.name = 'StainLerp'; mix.data_type = 'RGBA'; mix.blend_type = 'MIX'; mix.clamp_factor = True
        links.new(wm.outputs[0], mix.inputs['Factor']); links.new(c.outputs[0], mix.inputs[6]); links.new(cf.outputs[2], mix.inputs[7])
        links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    if STATE['aov']:
        aov = nodes.new('ShaderNodeOutputAOV'); aov.aov_name = 'stain_m'
        if m_out is not None: links.new(m_out, aov.inputs['Value'])
        else: aov.inputs['Value'].default_value = 0.
        aovc = nodes.new('ShaderNodeOutputAOV'); aovc.aov_name = 'stain_c'
        if base_input_link(bsdf) is not None: links.new(base_input_link(bsdf), aovc.inputs['Color'])
        else: aovc.inputs['Color'].default_value = (GRAY, GRAY, GRAY, 1.)
    return mat


def base_input_link(bsdf):
    links = bsdf.inputs['Base Color'].links
    return links[0].from_socket if links else None


def set_stain(mat, params):
    nodes = mat.node_tree.nodes
    for name in STAIN_SCALARS:
        nodes[name].outputs[0].default_value = float(params[name])


def material_readback(mat):
    nodes = mat.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    base_input = bsdf.inputs['Base Color']
    row = dict(name=mat.name, pass_index=mat.pass_index, roughness=float(bsdf.inputs['Roughness'].default_value),
               metallic=float(bsdf.inputs['Metallic'].default_value),
               normal=None if not bsdf.inputs['Normal'].is_linked else 'linked',
               ambient_occlusion=None, world_position_offset=None,
               displacement=None if not nodes['Material Output'].inputs['Displacement'].is_linked else 'linked')
    if base_input.is_linked:
        row['basecolor_linear'] = list(nodes['BaseC'].outputs[0].default_value)[:3]
        row['stain'] = {name: float(nodes[name].outputs[0].default_value) for name in STAIN_SCALARS}
        row['stain_graph'] = 'P_cm=(100,-100,100)*Position; lerp(C, C*F, w*m(d)) (UE _STAIN port)'
    else:
        row['basecolor_linear'] = list(base_input.default_value)[:3]
    return row


# ---------------------------------------------------------------- mesh
def build_mesh(spec):
    native = np.load(spec['native_surface'])
    v, f, codes = native['vertices'].astype(np.float64), native['faces'].astype(np.int64), native['codes'].astype(np.int64)
    if spec.get('display_surface'):
        display = np.load(spec['display_surface'])
        if not np.array_equal(display['vertices'][display['faces']], v[f]):
            raise ValueError('display corners differ from the native faces')
        corner_normals = display['normals'][display['faces']].reshape(-1, 3)
    else:
        corner_normals = None
    if STATE['obj'] is not None:
        old = STATE['obj']; mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True); bpy.data.meshes.remove(mesh)
    for mat in STATE['materials'].values():
        bpy.data.materials.remove(mat)
    STATE['materials'] = {}
    stain_codes = set(int(c) for c in spec.get('stain_codes', []))
    used = sorted(set(codes.tolist()))
    if corner_normals is not None and spec.get('flat'):
        raise ValueError('flat shading takes no display normals')
    if corner_normals is not None:
        # Unwelded corners (3 vertices per face, same positions): Blender averages custom normals within a smooth fan of
        # welded corners, which would merge the flat cavity corners and hard host edges with their neighbours
        # (measured: 90 deg corner error with welded vertices). Isolated faces keep every corner normal exactly.
        v = v[f].reshape(-1, 3); f = np.arange(len(v), dtype=np.int64).reshape(-1, 3)
    mesh = bpy.data.meshes.new('Outcrop')
    mesh.vertices.add(len(v)); mesh.vertices.foreach_set('co', v.astype(np.float32).ravel())
    mesh.loops.add(f.size); mesh.loops.foreach_set('vertex_index', f.ravel().astype(np.int32))
    mesh.polygons.add(len(f)); mesh.polygons.foreach_set('loop_start', (np.arange(len(f)) * 3).astype(np.int32))
    slot = {c: i for i, c in enumerate(used)}
    mesh.polygons.foreach_set('material_index', np.array([slot[c] for c in codes.tolist()], np.int32))
    mesh.update(calc_edges=True)
    mesh.validate(verbose=False, clean_customdata=False)
    if len(mesh.polygons) != len(f):
        raise ValueError('Blender dropped faces on validate')
    # flat=True (view-adaptive refined meshes): shading uses the true geometric face normals, so shading and visibility agree.
    mesh.polygons.foreach_set('use_smooth', np.full(len(f), not spec.get('flat', False)))
    if corner_normals is not None:
        mesh.normals_split_custom_set(corner_normals.astype(np.float32))
    for c in used:
        mat = code_material(c, c in stain_codes); STATE['materials'][c] = mat; mesh.materials.append(mat)
    obj = bpy.data.objects.new('Outcrop', mesh)
    # Cycles per-object shadow-terminator offsets; None keeps Blender's defaults (geometry 0.1, shading 0); read back.
    terminator = spec.get('shadow_terminator')
    if terminator is not None:
        obj.shadow_terminator_geometry_offset = float(terminator['geometry_offset'])
        obj.shadow_terminator_shading_offset = float(terminator['shading_offset'])
    bpy.context.scene.collection.objects.link(obj)
    STATE.update(obj=obj, mesh_key=spec['key'])
    # geometry readback: vertex positions (float32) and the evaluated corner normals
    co = np.empty(len(mesh.vertices) * 3, np.float32); mesh.vertices.foreach_get('co', co)
    loops = np.empty(len(mesh.loops), np.int32); mesh.loops.foreach_get('vertex_index', loops)
    readback = dict(vertices=len(mesh.vertices), faces=len(mesh.polygons), loops=len(mesh.loops),
                    loop_vertex_index_equal=bool(np.array_equal(loops, f.ravel())),
                    max_vertex_position_error_m=float(np.abs(co.reshape(-1, 3).astype(np.float64) - v).max()))
    if corner_normals is not None:
        cn = np.empty(len(mesh.loops) * 3, np.float32); mesh.corner_normals.foreach_get('vector', cn)
        cn = cn.reshape(-1, 3).astype(np.float64)
        dots = np.einsum('ij,ij->i', cn / np.linalg.norm(cn, axis=1)[:, None], corner_normals / np.linalg.norm(corner_normals, axis=1)[:, None])
        err = np.degrees(np.arccos(np.clip(dots, -1, 1)))
        tri = v[f]; area = np.repeat(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2, 3)
        # Blender's corner-space encoding is ill-conditioned on extreme slivers (in one test 130 of 1.03 M faces, area ~1e-10 m2)
        readback.update(max_corner_normal_error_deg=float(err.max()), corners_error_gt_1deg=int((err > 1).sum()),
                        corners_error_gt_0p1deg=int((err > .1).sum()),
                        max_corner_normal_error_deg_area_ge_1em8_m2=float(err[area >= 1e-8].max()) if (area >= 1e-8).any() else 0.,
                        max_area_of_corners_error_gt_1deg_m2=float(area[err > 1].max()) if (err > 1).any() else None)
    readback['flat_shading'] = bool(spec.get('flat', False))
    STATE['geometry_readback'] = readback
    return readback


# ---------------------------------------------------------------- scene
def ensure_camera_sun_world():
    scene = bpy.context.scene
    if 'Cam' not in bpy.data.objects:
        cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam')); scene.collection.objects.link(cam); scene.camera = cam
    if 'Sun' not in bpy.data.objects:
        sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN')); scene.collection.objects.link(sun)
    if 'StudyWorld' not in bpy.data.worlds:         # looked up by name: an overcast job may have switched scene.world
        world = bpy.data.worlds.new('StudyWorld'); scene.world = world; world.use_nodes = True
        nodes = world.node_tree.nodes
        sky = nodes.new('ShaderNodeTexSky'); sky.name = 'Sky'; sky.sky_type = 'NISHITA'
        world.node_tree.links.new(sky.outputs['Color'], nodes['Background'].inputs['Color'])
    return bpy.data.objects['Cam'], bpy.data.objects['Sun'], bpy.data.worlds['StudyWorld']


def srgb8_to_linear(c):
    c = [float(x) / 255. for x in c]
    return [x / 12.92 if x <= .04045 else ((x + .055) / 1.055) ** 2.4 for x in c]


def set_ground(spec):
    """spec None: remove the ground. Else a flat square z = z_m about center_xy_m (half_size_m), Principled BSDF with the sRGB
    8-bit colour converted to linear BaseColor, roughness ROUGHNESS, Specular IOR Level GROUND_SPECULAR_IOR_LEVEL (Lambertian),
    pass_index GROUND_PASS_INDEX; visible to every ray type."""
    old = bpy.data.objects.get('Ground')
    if old is not None:
        mesh = old.data; bpy.data.objects.remove(old, do_unlink=True); bpy.data.meshes.remove(mesh)
    if bpy.data.materials.get('GroundMat') is not None:          # a fresh material per job: no setting carries over
        bpy.data.materials.remove(bpy.data.materials['GroundMat'])
    if spec is None:
        STATE['ground'] = None; return
    z, (cx, cy), h = float(spec['z_m']), map(float, spec['center_xy_m']), float(spec['half_size_m'])
    mesh = bpy.data.meshes.new('Ground')
    mesh.from_pydata([(cx - h, cy - h, z), (cx + h, cy - h, z), (cx + h, cy + h, z), (cx - h, cy + h, z)], [], [(0, 1, 2), (0, 2, 3)])
    mesh.polygons.foreach_set('use_smooth', [False, False]); mesh.update()
    mat = bpy.data.materials.new('GroundMat')
    mat.use_nodes = True; mat.pass_index = GROUND_PASS_INDEX
    bsdf = mat.node_tree.nodes['Principled BSDF']; colour = spec.get('color_srgb8', GROUND_COLOR_SRGB8); lin = srgb8_to_linear(colour)
    bsdf.inputs['Base Color'].default_value = (lin[0], lin[1], lin[2], 1.)
    bsdf.inputs['Roughness'].default_value = ROUGHNESS; bsdf.inputs['Metallic'].default_value = 0.
    bsdf.inputs['Specular IOR Level'].default_value = float(spec.get('specular_ior_level', GROUND_SPECULAR_IOR_LEVEL))
    mesh.materials.append(mat)
    obj = bpy.data.objects.new('Ground', mesh); bpy.context.scene.collection.objects.link(obj)
    STATE['ground'] = dict(z_m=z, center_xy_m=[cx, cy], half_size_m=h, color_srgb8=[int(x) for x in colour], basecolor_linear=lin,
                           roughness=ROUGHNESS, pass_index=GROUND_PASS_INDEX)


def ensure_overcast_world():
    """CIE standard overcast sky, neutral: L(dir) = L_z (1 + 2 max(dir_z, 0)) / 3 (below the horizon L_z / 3)."""
    world = bpy.data.worlds.get('OvercastWorld')
    if world is None:
        world = bpy.data.worlds.new('OvercastWorld'); world.use_nodes = True
        nt = world.node_tree; nodes, links = nt.nodes, nt.links
        tc = nodes.new('ShaderNodeTexCoord'); sep = nodes.new('ShaderNodeSeparateXYZ'); links.new(tc.outputs['Generated'], sep.inputs[0])
        mx = _math(nodes, 'MAXIMUM', None, 0.); links.new(sep.outputs['Z'], mx.inputs[0])
        m2 = _math(nodes, 'MULTIPLY', None, 2.); links.new(mx.outputs[0], m2.inputs[0])
        a1 = _math(nodes, 'ADD', None, 1.); links.new(m2.outputs[0], a1.inputs[0])
        d3 = _math(nodes, 'DIVIDE', None, 3.); links.new(a1.outputs[0], d3.inputs[0])
        lz = _math(nodes, 'MULTIPLY', None, 1.); lz.name = lz.label = 'ZenithRadiance'; links.new(d3.outputs[0], lz.inputs[0])
        bg = nodes['Background']; bg.inputs['Color'].default_value = (1., 1., 1., 1.); links.new(lz.outputs[0], bg.inputs['Strength'])
    return world


def set_overcast(sun, light):
    """light: mode 'overcast', zenith_radiance (L_z, Blender radiance units). The Sun lamp is hidden: no directional light."""
    world = ensure_overcast_world()
    world.node_tree.nodes['ZenithRadiance'].inputs[1].default_value = float(light['zenith_radiance'])
    bpy.context.scene.world = world
    sun.hide_render = True
    STATE['overcast'] = dict(zenith_radiance=float(light['zenith_radiance']))


def restore_sunny(sun, world):
    """After an overcast job: the Sun lamp renders again and the Nishita StudyWorld is back (no-op in sessions without one)."""
    if sun.hide_render: sun.hide_render = False
    if bpy.context.scene.world is not world: bpy.context.scene.world = world
    STATE['overcast'] = None


def set_camera(cam, shot, resolution):
    scene = bpy.context.scene
    w, h = resolution
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = int(w), int(h), 100
    cam.data.type = 'PERSP'
    cam.data.lens = float(shot['focal_length_mm'])
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = float(shot['filmback_mm'][0])
    cam.data.sensor_height = float(shot['filmback_mm'][1])
    cam.data.shift_x = cam.data.shift_y = 0.
    cam.data.clip_start = float(shot.get('near_clip_cm', 10.)) / 100.
    cam.data.clip_end = 1e4
    pos = Vector(shot['position_enu_m'])
    cam.location = pos
    look(cam, Vector(shot['target_enu_m']) - pos)


def set_light(sun, world, light):
    """light: toward_sun_enu (unit), angle_deg, strength (W/m2 normal irradiance), color (linear rgb), sky dict."""
    to_sun = Vector(light['toward_sun_enu']).normalized()
    sun.data.angle = math.radians(float(light['angle_deg']))
    sun.data.energy = float(light['strength'])
    sun.data.color = tuple(float(x) for x in light['color'])
    sun.data.use_shadow = True
    look(sun, -to_sun, up_world=(0., 1., 0.) if abs(to_sun.z) > .99 else (0., 0., 1.))
    sky = world.node_tree.nodes['Sky']; s = light['sky']
    sky.sky_type = 'NISHITA'; sky.sun_disc = False
    sky.sun_size = math.radians(float(light['angle_deg']))
    sky.sun_elevation = float(math.asin(max(-1., min(1., to_sun.z))))
    sky.sun_rotation = float(s['sun_rotation_rad'])
    sky.altitude = float(s['altitude_m']); sky.air_density = float(s['air_density'])
    sky.dust_density = float(s['dust_density']); sky.ozone_density = float(s['ozone_density'])
    sky.sun_intensity = 1.
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = float(s['strength'])


def scene_readback():
    scene = bpy.context.scene
    sun = bpy.data.objects['Sun']; world = bpy.data.worlds['StudyWorld']   # = scene.world except in an overcast job (see 'aerial')
    sky = world.node_tree.nodes['Sky']
    ray = (sun.matrix_world.to_3x3() @ Vector((0., 0., -1.))).normalized()      # the lamp shines along its local -Z
    objects = [dict(name=o.name, type=o.type, visible=not o.hide_render) for o in scene.objects]
    casters = [o.name for o in scene.objects if o.type == 'MESH' and not o.hide_render and o.visible_shadow]
    c = scene.cycles
    return dict(
        renderer='blender-%s-cycles' % bpy.app.version_string, schema='blender-scene-readback-1',
        objects=objects, shadow_casters=casters, closed=(casters == ['Outcrop'] and sorted(o['type'] for o in objects) == ['CAMERA', 'LIGHT', 'MESH']),
        sun=dict(ray_direction_enu=list(ray), angle_deg=math.degrees(sun.data.angle), strength=sun.data.energy,
                 color=list(sun.data.color), cast_shadows=bool(sun.data.use_shadow), soft_angle_deg=0.),
        sky=dict(type=sky.sky_type, sun_disc=bool(sky.sun_disc), sun_size_deg=math.degrees(sky.sun_size),
                 sun_elevation_deg=math.degrees(sky.sun_elevation), sun_rotation_rad=sky.sun_rotation, altitude_m=sky.altitude,
                 air_density=sky.air_density, dust_density=sky.dust_density, ozone_density=sky.ozone_density,
                 strength=float(world.node_tree.nodes['Background'].inputs['Strength'].default_value)),
        materials={str(code): material_readback(m) for code, m in STATE['materials'].items()},
        camera=dict(location=list(bpy.data.objects['Cam'].location), matrix=[list(r) for r in bpy.data.objects['Cam'].matrix_world],
                    lens_mm=bpy.data.objects['Cam'].data.lens, sensor_width_mm=bpy.data.objects['Cam'].data.sensor_width,
                    clip_start_m=bpy.data.objects['Cam'].data.clip_start),
        render=dict(samples=c.samples, seed=c.seed, device=c.device, compute=STATE['device'],
                    resolution=[scene.render.resolution_x, scene.render.resolution_y],
                    **{k: (getattr(c, k) if not isinstance(getattr(c, k), float) else float(getattr(c, k))) for k in INTEGRATOR},
                    persistent_data=scene.render.use_persistent_data, view_transform=scene.view_settings.view_transform,
                    passes=['Combined', 'Depth', 'IndexMA'] + (['stain_m', 'stain_c'] if STATE['aov'] else []), filter_override=STATE['filter']),
        geometry=STATE.get('geometry_readback'), mesh_key=STATE['mesh_key'],
        shadow_terminator=dict(geometry_offset=float(STATE['obj'].shadow_terminator_geometry_offset),
                               shading_offset=float(STATE['obj'].shadow_terminator_shading_offset)))


def aerial_readback():
    """Read back the ground and the overcast state from the Blender objects (added only to jobs that use them; 'closed' is
    re-evaluated with the Ground as the second allowed mesh / caster)."""
    scene = bpy.context.scene; out = {}
    objects = [o for o in scene.objects if not o.hide_render]
    casters = sorted(o.name for o in objects if o.type == 'MESH' and o.visible_shadow)
    g = bpy.data.objects.get('Ground')
    expected = ['Ground', 'Outcrop'] if g is not None else ['Outcrop']
    out['closed'] = casters == expected and sorted(o.type for o in scene.objects) == sorted(['CAMERA', 'LIGHT'] + ['MESH'] * len(expected))
    if g is not None:
        co = [list(g.matrix_world @ v.co) for v in g.data.vertices]
        bsdf = g.active_material.node_tree.nodes['Principled BSDF']
        out['ground'] = dict(STATE['ground'], vertices_world=co, visible_camera=g.visible_camera, visible_shadow=g.visible_shadow,
                             visible_diffuse=g.visible_diffuse, material_pass_index=g.active_material.pass_index,
                             basecolor_readback=list(bsdf.inputs['Base Color'].default_value)[:3],
                             roughness_readback=float(bsdf.inputs['Roughness'].default_value),
                             specular_ior_level_readback=float(bsdf.inputs['Specular IOR Level'].default_value))
    sun = bpy.data.objects['Sun']
    out['light_mode'] = 'overcast' if STATE.get('overcast') else 'sun'
    out['sun_renders'] = not sun.hide_render
    out['world'] = scene.world.name
    if STATE.get('overcast'):
        nodes = scene.world.node_tree.nodes
        out['overcast'] = dict(STATE['overcast'], zenith_radiance_readback=float(nodes['ZenithRadiance'].inputs[1].default_value),
                               background_color=list(nodes['Background'].inputs['Color'].default_value)[:3],
                               distribution='L_z (1 + 2 max(dir_z, 0)) / 3 (CIE standard overcast), neutral',
                               lamps_rendering=[o.name for o in scene.objects if o.type == 'LIGHT' and not o.hide_render])
    return dict(aerial=out)


def run_job(job, queue):
    t0 = time.perf_counter()
    if job.get('samples') not in (None, STATE['samples']):
        raise ValueError('job sample count %s differs from the session sample count %s' % (job.get('samples'), STATE['samples']))
    timing = {}
    if job['mesh']['key'] != STATE['mesh_key']:
        t = time.perf_counter(); build_mesh(job['mesh']); timing['mesh_s'] = time.perf_counter() - t
    for code, params in (job.get('stain') or {}).items():
        mat = STATE['materials'].get(int(code))
        if mat is None or 'StainWeight' not in mat.node_tree.nodes:
            raise ValueError('stain parameters for a code without a stain graph: %s' % code)
        set_stain(mat, params)
    for code, mat in STATE['materials'].items():
        if 'StainWeight' in mat.node_tree.nodes and str(code) not in (job.get('stain') or {}):
            raise ValueError('stain-eligible code %d needs explicit parameters (weight 0 for off)' % code)
    cam, sun, world = ensure_camera_sun_world()
    set_camera(cam, job['shot'], job['resolution'])
    if job.get('ground') is not None or 'Ground' in bpy.data.objects:
        set_ground(job.get('ground'))
    if job['light'].get('mode') == 'overcast':
        set_overcast(sun, job['light'])
    else:
        restore_sunny(sun, world)
        set_light(sun, world, job['light'])
    scene = bpy.context.scene
    scene.cycles.seed = int(job['seed'])
    out = Path(job['output_exr']); out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out)
    t = time.perf_counter(); bpy.ops.render.render(write_still=True); timing['render_s'] = time.perf_counter() - t
    if not out.is_file():
        raise RuntimeError('render wrote no EXR: ' + str(out))
    readback = scene_readback()
    if STATE.get('ground') or STATE.get('overcast'):
        readback.update(aerial_readback())
    timing['total_s'] = time.perf_counter() - t0
    STATE['jobs'] += 1
    return dict(id=job['id'], success=True, output_exr=str(out), readback=readback, timing=timing, session_jobs=STATE['jobs'],
                pid=os.getpid())


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    args = parse(argv)
    q = args['queue']
    for sub in ('pending', 'running', 'done', 'failed'):
        (q/sub).mkdir(parents=True, exist_ok=True)
    configure(args['samples'], args['device'], args['aov'], args['filter'])
    started = time.time()

    def beat(state):
        (q/'worker.json').write_text(json.dumps(dict(pid=os.getpid(), samples=STATE['samples'], device=STATE['device'], aov=STATE['aov'],
                                                     jobs=STATE['jobs'], state=state, started=started, time=time.time(),
                                                     blender=bpy.app.version_string)) + '\n', encoding='utf8')
    beat('ready')
    print('WORKER ready pid %d samples %d device %s' % (os.getpid(), STATE['samples'], STATE['device']), flush=True)
    while True:
        if (q/'stop').exists():
            beat('stopped'); print('WORKER stop', flush=True); return
        pending = sorted((q/'pending').glob('*.json'))
        if not pending:
            time.sleep(.05); continue
        path = pending[0]
        running = q/'running'/path.name
        path.replace(running)
        job = json.loads(running.read_text(encoding='utf8'))
        beat('job ' + job.get('id', path.stem))
        try:
            result = run_job(job, q)
            tmp = q/'done'/(path.name + '.tmp'); tmp.write_text(json.dumps(result, indent=1) + '\n', encoding='utf8'); tmp.replace(q/'done'/path.name)
        except Exception as error:  # the job fails closed; the worker keeps serving
            tmp = q/'failed'/(path.name + '.tmp')
            tmp.write_text(json.dumps(dict(id=job.get('id'), success=False, error='%s: %s' % (type(error).__name__, error),
                                           traceback=traceback.format_exc()), indent=1) + '\n', encoding='utf8')
            tmp.replace(q/'failed'/path.name)
        running.unlink()
        beat('idle')


if __name__ == '__main__':
    main()
