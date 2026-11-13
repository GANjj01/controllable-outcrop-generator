"""Multi-scene engineering: generate ONE scene with the D1 relief rescaled by the scene's background parameters
(amplitude a, Hurst H; frgb/scene_relief.py).

Phases (each one process, single-threaded: the multi-scene driver runs up to 6 of these in parallel, never nested):
spec      OUT/relief_spec.json = C_r2 octaves rescaled by (a, H) generate  the unchanged
outcrop.dsgen.pipeline.generate_scene on CONFIG (seed override) into OUT/geom, with erode called with
self_affine=OUT/relief_spec.json fields bluff_reproduce.main on OUT/geom -> OUT/erode_fields.pkl (the exact
evaluator's input); its own report says whether the second erode reproduces the generated processed/source surface bit
for bit wall      OUT/wall_relief.json = same rescaled groups 0-3, seed = the scene's field seed (erode scalars)
Usage: python scene_generate.py OUT_DIR --config CONFIG_JSON [--seed N] --amplitude A --hurst H [--phase
spec|generate|fields|wall|all]"""
import functools, hashlib, json, pickle, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; P1 = HERE.parent; M = P1/'frgb'/'chain'
sys.path[:0] = [str(P1), str(M)]
from frgb import scene_relief  # noqa: E402
from frgb import runconfig as rc  # noqa: E402

BASE_SPEC = rc.path('relief.spec_C_r2')
BASE_WALL = rc.path('relief.wall_C_r2')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def phase_spec(out, a, h):
    out.mkdir(parents=True, exist_ok=True)
    scene_relief.scaled_spec(BASE_SPEC, a, h, out/'relief_spec.json')


def _glb_unchecked(products, path):
    """Copy of outcrop.modeling.package._glb with its float32 check removed: the same quantized vertices, primitives and
    bytes. Used only after the original raised its scene.glb constraint error."""
    import numpy as np
    import trimesh
    mesh=products.extraction.surface;vertices=mesh.vertices[:,[0,2,1]].copy();vertices[:,2]*=-1
    quantized=vertices.astype(np.float32).astype(np.float64);error=np.linalg.norm(vertices-quantized,axis=1).max()
    scene=trimesh.Scene();mapping=[]
    ids=products.appearance.face_appearance_ids
    for oid in sorted(set(ids),key=lambda x:'' if x is None else x):
        face_ids=np.array([i for i,a in enumerate(ids) if a==oid],np.int64);v=quantized[mesh.faces[face_ids]].reshape(-1,3);f=np.arange(len(v)).reshape(-1,3)
        record=products.appearance.appearances.get(oid,{})
        color=record.get('color',[.65,.65,.65,1]);color=list(color)+([1] if len(color)==3 else [])
        material=trimesh.visual.material.PBRMaterial(name=oid or 'diagnostic',baseColorFactor=color,roughnessFactor=record.get('roughness',.8),metallicFactor=record.get('metallic',0))
        uv=products.appearance.uv_m[face_ids].reshape(-1,2);visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
        name='primitive:'+str(len(mapping));scene.add_geometry(trimesh.Trimesh(v,f,visual=visual,process=False),geom_name=name,node_name=name);mapping.append({'primitive':name,'appearance':oid,'authoritative_face_indices':face_ids.tolist()})
    path.write_bytes(scene.export(file_type='glb'));return mapping,float(error)


def _glb_record_only(record):
    """The original _glb runs first, so a passing scene is byte-identical; on its scene.glb constraint error the failure
    (quantization error, budget, triangles collapsed in float32) is recorded and the same GLB is written without the
    check. The float64 surface.npz and every other package payload and check are unchanged. Both write_package calls
    of generate_scene (source_package, package) go through it."""
    import numpy as np
    from outcrop.modeling import package
    from outcrop.modeling.errors import ModelError
    original = package._glb

    @functools.wraps(original)
    def glb(products, path):
        try:
            out = original(products, path)
            record.append(dict(call=len(record), passed=True)); return out
        except ModelError as error:
            if error.field != 'scene.glb' or error.code != 'constraint_failed': raise
            mesh = products.extraction.surface; v = mesh.vertices[:, [0, 2, 1]].copy(); v[:, 2] *= -1
            q = v.astype(np.float32).astype(np.float64); t = q[mesh.faces]
            collapsed = int(np.sum(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) == 0))
            record.append(dict(call=len(record), passed=False, error=str(error), details=error.details, collapsed_triangles=collapsed,
                               faces=int(len(mesh.faces))))
            return _glb_unchecked(products, path)
    return package, original, glb


def phase_generate(out, config, seed):
    geom = out/'geom'
    if geom.exists(): raise FileExistsError(geom)
    cfg = json.loads(Path(config).read_text(encoding='utf8'))
    if seed is not None: cfg['seed'] = int(seed)
    (out/'config.json').write_text(json.dumps(cfg, indent=1) + '\n', encoding='utf8')
    import outcrop.dsgen.processes as proc
    from outcrop.dsgen.pipeline import generate_scene
    original, calls, spec = proc.erode, [], str((out/'relief_spec.json').resolve())

    @functools.wraps(original)
    def erode(*x, **k):
        if 'self_affine' in k: raise ValueError('caller already sets self_affine')
        calls.append(1)
        return original(*x, **k, self_affine=spec)
    proc.erode = erode
    glb_record = []; package, glb_original, glb = _glb_record_only(glb_record)
    package._glb = glb
    t0 = time.time()
    try:
        generate_scene(cfg, geom)
    finally:
        proc.erode = original; package._glb = glb_original
        (out/'glb_check.json').write_text(json.dumps(glb_record, indent=1) + '\n', encoding='utf8')
    if len(calls) != 1: raise RuntimeError('expected exactly one erode call, got %d' % len(calls))
    return time.time() - t0


def phase_fields(out):
    import bluff_reproduce
    t0 = time.time()
    bluff_reproduce.main(str(out/'erode_fields.pkl'), str((out/'relief_spec.json').resolve()), config=str(out/'config.json'),
                         original=str(out/'geom'))
    return time.time() - t0


def phase_wall(out):
    d = pickle.load(open(out/'erode_fields.pkl', 'rb'))
    sa = d['scalars']['self_affine']
    if sa['sha256'] != sha(out/'relief_spec.json'): raise ValueError('fields were made with another spec')
    scene_relief.wall_spec(out/'relief_spec.json', sa['seed'], out/'wall_relief.json', BASE_WALL)
    rep = d['report']
    return dict(field_seed=sa['seed'], reproduce_processed_equal=rep['processed_equal'], reproduce_source_equal=rep['source_vertices_equal'])


def main(argv):
    out = Path(argv[0]).resolve(); g = lambda k, d=None: argv[argv.index('--' + k) + 1] if '--' + k in argv else d
    a, h, phase = float(g('amplitude')), float(g('hurst')), g('phase', 'all')
    seed = g('seed'); config = g('config')
    log = {}
    if phase in ('spec', 'all'): phase_spec(out, a, h)
    if phase in ('generate', 'all'):
        log['generate_s'] = phase_generate(out, config, seed)
        log['glb_check'] = json.loads((out/'glb_check.json').read_text(encoding='utf8'))
    if phase in ('fields', 'all'): log['fields_s'] = phase_fields(out)
    if phase in ('wall', 'all'): log.update(phase_wall(out))
    rec = out/'scene_generate.json'
    old = json.loads(rec.read_text(encoding='utf8')) if rec.is_file() else {}
    old.update(log, amplitude=a, hurst=h, config=str(config) if config else old.get('config'), seed=seed if seed else old.get('seed'))
    rec.write_text(json.dumps(old, indent=1) + '\n', encoding='utf8')
    print(json.dumps(log))


if __name__ == '__main__':
    main(sys.argv[1:])
