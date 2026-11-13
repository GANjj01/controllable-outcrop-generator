"""CPU pixel-centre raster (frgb.network_training_labels.rasterize_native, as make_source_raster.py) at the camera: -
isolated misses: pixels with no hit whose 4 neighbours all hit (a crack seen as a hole); must be 0; - with a
single-level mesh of the same camera: pixels hit by the single-level mesh, missed by the zoned one, farther than 3 px
from the single-level silhouette (the hit / no-hit boundary); must be 0. The same counts for the single-level mesh are
reported as the rasterizer's own baseline. Usage: python zone_raster_check.py ZONED_DIR REQUEST_JSON OUT_JSON
[SINGLE_DIR]"""
import json, sys
from pathlib import Path
import numpy as np

P1 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P1)]
from frgb.raster import camera  # noqa: E402
from frgb.network_training_labels import rasterize_native  # noqa: E402


def hits(d, cam):
    s = np.load(Path(d)/'source_surface.npz')
    return rasterize_native(s['vertices'], s['faces'], cam)['face'].reshape(cam['h'], cam['w']) >= 0


def isolated_misses(h):
    m = ~h[1:-1, 1:-1]
    return int((m & h[:-2, 1:-1] & h[2:, 1:-1] & h[1:-1, :-2] & h[1:-1, 2:]).sum())


def dilate(mask, r):
    out = mask.copy()
    for _ in range(r):
        o = out.copy(); o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]; out = o
    return out


def main(zoned, request, out, single=None):
    req = json.loads(Path(request).read_text(encoding='utf8')); cam = camera(req['shots'][0], req['resolution'])
    hz = hits(zoned, cam); rep = dict(zoned=str(zoned), rock_px_zoned=int(hz.sum()), isolated_misses_zoned=isolated_misses(hz))
    ok = rep['isolated_misses_zoned'] == 0
    if single:
        hs = hits(single, cam)
        sil = (dilate(hs, 1) & ~hs) | (hs & dilate(~hs, 1))           # pixels on either side of the hit / no-hit boundary
        near = dilate(sil, 3)
        leak = hs & ~hz & ~near
        rep.update(single=str(single), rock_px_single=int(hs.sum()), isolated_misses_single=isolated_misses(hs),
                   single_hit_zoned_miss=int((hs & ~hz).sum()), leak_px_beyond_3px=int(leak.sum()),
                   zoned_hit_single_miss=int((hz & ~hs).sum()))
        ok = ok and rep['leak_px_beyond_3px'] == 0
    rep['PASS'] = bool(ok)
    Path(out).write_text(json.dumps(rep, indent=1) + '\n', encoding='utf8'); print(json.dumps(rep))


if __name__ == '__main__':
    main(*sys.argv[1:5])
