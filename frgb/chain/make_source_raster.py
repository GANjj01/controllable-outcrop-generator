"""Usage: python make_source_raster.py SURFACE_NPZ REQUEST_JSON OUT_NPZ"""
import json, sys
from pathlib import Path
import numpy as np

P1 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(P1)]
from frgb.raster import camera  # noqa: E402
from frgb.network_training_labels import rasterize_native  # noqa: E402


def main(surface, request, out):
    req = json.loads(Path(request).read_text(encoding='utf8'))
    with np.load(surface) as a: v, f = a['vertices'], a['faces']
    cam = camera(req['shots'][0], req['resolution'])
    r = rasterize_native(v, f, cam)
    meta = dict(surface=str(Path(surface).resolve()), request=str(Path(request).resolve()), resolution=req['resolution'])
    np.savez_compressed(out, **r, metadata=json.dumps(meta))
    print(json.dumps(dict(meta, visible_faces=int(len(np.unique(r['face'][r['face'] >= 0]))), covered=float((r['face'] >= 0).mean()))))


if __name__ == '__main__':
    main(*sys.argv[1:4])
