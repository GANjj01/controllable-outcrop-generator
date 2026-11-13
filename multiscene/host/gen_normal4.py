# the generator's resolve_config rejects unknown keys, so that one key is stripped before validation (wrapper below).
"""Installs one replacement before the unchanged phases run: outcrop.dsgen.pipeline.build_surface (the name
build_declaration calls) builds the exposure exactly as before and, for a bluff, replaces its ray patch by
normal_surface.normal_patch (offset sheet on the envelope, outward = analytic normal). The replacement is active in
the generate phase AND in the fields phase, whose bluff_reproduce calls build_declaration again (its
reference/processed/source equality checks therefore compare the normal-route surfaces). Everything else - D1 spec,
erode with self_affine, the pipeline's checks, the GLB record-only wrapper, the field export, the wall spec - is
scene_generate's. Writes OUT/normal_route.json (grid spacing, base-mesh self-check per build).
--fracture-density-per-m2 D: opt-in fracture count from area. --focal-guard EPS: the overlay erode's sheet fold guard
also halves offsets where 1 - total k < EPS (k = principal curvatures of the base); the reduction is carried in gscale
like any fold-guard reduction."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parent)) if str(_p.Path(__file__).resolve().parent) not in _s.path else None
import nogpu  # noqa: F401,E402  (CPU only)
import functools, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; MS = HERE.parent
sys.path[:0] = [str(HERE), str(MS)]
import normal_surface as ns  # noqa: E402


def install(h, record, crest_cap=False, groove_tip=None, focal_guard=None):
    import outcrop.dsgen.pipeline as pl
    original = pl.build_surface

    @functools.wraps(original)
    def build_surface(expo, **kw):
        surf = original(expo, **kw)
        if expo['kind'] != 'bluff':
            return surf
        t = time.time()
        expo['normal_route'] = dict(edge_m=h, crest_cap=dict(eps=ns.CAP_EPS, knee=ns.CAP_KNEE, curvature_step_m=ns.CURV_H) if crest_cap else None,
                                    groove_tip=groove_tip, focal_guard=focal_guard)
        patch, info = ns.normal_patch(expo, h, crest_cap=crest_cap, focal_curvature=focal_guard is not None)
        info['seconds'] = time.time() - t
        surf.patches = [patch]
        record.append(info)
        return surf
    pl.build_surface = build_surface
    return pl, original


def install_exact_slab(record):
    import outcrop.dsgen.exposure as ex
    original = ex._bluff

    @functools.wraps(original)
    def _bluff(g, tile, shape=None):
        res = original(g, tile, shape=shape)
        if shape and len(res['masses']) == 1:
            m = res['masses'][0]; g0 = tile['ground_z_m']; base = -shape.get('base_depth_m', 2.5)
            W, D, H = res['width_m'], res['depth_m'], res['height_m']; hz = (H - base) / 2
            half = [W / 2, D / 2, hz]; r = min(m['rounding_m'], .45 * min(half))
            before = json.loads(json.dumps(m))
            m.update(center_m=[res['origin_m'][0], res['origin_m'][1], g0 + (H + base) / 2], half_size_m=half, rounding_m=r)
            res['exact_slab'] = True
            record.append(dict(sampled=before, exact=json.loads(json.dumps(m))))
        return res
    ex._bluff = _bluff
    return ex, original


def install_outcrop_type():
    """outcrop.dsgen.pipeline.resolve_config ignores geometry.outcrop_type (a label for the camera driver, not a generator setting)"""
    import copy
    import outcrop.dsgen.pipeline as pl
    original = pl.resolve_config

    @functools.wraps(original)
    def resolve_config(config, *a, **k):
        if isinstance(config, dict) and isinstance(config.get('geometry'), dict) and 'outcrop_type' in config['geometry']:
            config = copy.deepcopy(config); config['geometry'].pop('outcrop_type')
        return original(config, *a, **k)
    pl.resolve_config = resolve_config
    return pl, original


def main(argv):
    otype = None
    if '--outcrop-type' in argv:
        j = argv.index('--outcrop-type'); otype = argv[j + 1]; argv = argv[:j] + argv[j + 2:]
    exact = '--exact-slab' in argv
    if exact and otype is None:
        otype = 'flat'
    argv = [a for a in argv if a != '--exact-slab']
    density = None
    if '--fracture-density-per-m2' in argv:
        j = argv.index('--fracture-density-per-m2'); density = float(argv[j + 1]); argv = argv[:j] + argv[j + 2:]
    crest_cap = '--crest-cap' in argv
    argv = [a for a in argv if a != '--crest-cap']
    focal_guard = None
    if '--focal-guard' in argv:
        j = argv.index('--focal-guard'); focal_guard = dict(eps=float(argv[j + 1])); argv = argv[:j] + argv[j + 2:]
    if '--focal-local' in argv:
        argv = [a for a in argv if a != '--focal-local']
        if focal_guard is None: raise ValueError('--focal-local needs --focal-guard EPS')
        focal_guard['mode'] = 'local'; focal_guard['rings'] = 2
    groove_tip = None
    if '--groove-tip-m' in argv:
        j = argv.index('--groove-tip-m'); groove_tip = dict(taper_m=float(argv[j + 1])); argv = argv[:j] + argv[j + 2:]
    i = argv.index('--edge-m'); h = float(argv[i + 1]); rest = argv[:i] + argv[i + 2:]
    out = Path(rest[0]).resolve(); out.mkdir(parents=True, exist_ok=True)
    phase = rest[rest.index('--phase') + 1] if '--phase' in rest else 'all'
    record, slab = [], []; pl, original = install(h, record, crest_cap, groove_tip, focal_guard)
    if groove_tip is not None or focal_guard is not None:
        import overlay4 as overlay
        prm, pr_original = overlay.install(groove_tip, focal_guard)
    if exact:
        exm, ex_original = install_exact_slab(slab)
    dens = []
    if density is not None:
        import fracture_density
        pgm, pg_original = fracture_density.install(density, dens)
    plo, rc_original = install_outcrop_type()
    import scene_generate
    try:
        scene_generate.main(rest)
        cj = out/'config.json'
        if otype is not None and cj.is_file():
            c = json.loads(cj.read_text(encoding='utf8'))
            if c['geometry'].get('outcrop_type') != otype:
                c['geometry']['outcrop_type'] = otype; cj.write_text(json.dumps(c, indent=1) + chr(10), encoding='utf8')
    finally:
        plo.resolve_config = rc_original
        pl.build_surface = original
        if exact:
            exm._bluff = ex_original
        if density is not None:
            pgm.apply_fracture_network = pg_original
        if groove_tip is not None or focal_guard is not None:
            prm.erode = pr_original
        f = out/'normal_route.json'
        old = json.loads(f.read_text(encoding='utf8')) if f.is_file() else {}
        old[phase] = dict(grid_spacing_m=h, builds=record, construction=ns.__doc__.split('\n')[0], exact_slab=exact, slab=slab, fracture_density=dens, crest_cap=crest_cap, groove_tip=groove_tip, focal_guard=focal_guard, outcrop_type=otype)
        f.write_text(json.dumps(old, indent=1) + '\n', encoding='utf8')


if __name__ == '__main__':
    main(sys.argv[1:])
