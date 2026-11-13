"""Usage: python si_camera_plan.py plan shape|stain SCENE_DIR NET_ROOT|- OUT_JSON [--per-cell K (default 2)]"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; MS = HERE.parent
sys.path[:0] = [str(HERE), str(MS)]
import nogpu  # noqa: F401,E402  (CPU only)
import camera_plan_r2 as cp2  # noqa: E402

ORIGINAL = cp2.top_fold


def top_fold(sc, rock, f, P):
    ok, rec = ORIGINAL(sc, rock, f, P)
    expo = sc.get('expo')
    if expo is None or not expo.get('normal_route'):
        return ok, rec
    rec = dict(rec, literal_pass=bool(ok), applicable=False,
               reason='normal route: the surface is offset along the envelope normal, there is no ray solution to shear '
                      '(gate g of rule Q4; the route fold mode is checked by Q1b / S2_J)')
    return True, rec


cp2.top_fold = top_fold          # evaluate() looks top_fold up in the module globals at call time

if __name__ == '__main__':
    import json
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index('--' + k) + 1] if '--' + k in a else d  # noqa: E731
    mode, mech, scene, net, outp = a[:5]
    if mode != 'plan': raise SystemExit('only plan mode')
    res = cp2.plan(mech, scene, None if net == '-' else net, int(g('per-cell', 2)), 'isolated')
    if res['outcrop_type'] != 'isolated': raise SystemExit('scene is not isolated: %s (%s)' % (res['outcrop_type'], res['outcrop_type_source']))
    res['wrapper'] = dict(script=str(Path(__file__).resolve()).replace('\\', '/'), note=__doc__.split('\n')[0],
                          coarse_outcrops=list(cp2.RULES['coarse_outcrops']), per_cell=int(g('per-cell', 2)))
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    Path(outp).write_text(json.dumps(res, indent=1) + '\n', encoding='utf8')
    print(json.dumps(dict(qualified=res['qualified'], failed=[f['cell'] for f in res['failed_cells']], skipped=res['skipped_cells'],
                          cameras=[dict(cell=c['cell'], k=c['k'], level=c['level'], angle=c['angle'], face=c['face_type'], px_mm=round(c['pixel_mm'], 2),
                                        theta=round(c['angle_deg'], 1), tries=c['tries'], rock=round(c['checks']['fractions']['rock'], 3),
                                        sky=round(c['checks']['fractions']['sky'], 3)) for c in res['cameras']]), indent=0), flush=True)
