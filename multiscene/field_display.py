"""Display normals with FIELD normals on the host.

Usage: python field_display.py CASE_DIR FIELD_NPZ"""
import json, sys
from pathlib import Path
import numpy as np


def main(case, field_npz):
    case = Path(case); out = case/'shading_field'; out.mkdir(exist_ok=False)
    build = json.loads((case/'build.json').read_text()); src = np.load(Path(build['source'])/'source_surface.npz')
    sv, sf = src['vertices'], src['faces'].astype(np.int64)
    sn = np.load(field_npz)['normals']
    if sn.shape != sv.shape: raise ValueError('field normals do not belong to this source')
    nat = np.load(case/'geometry/native_surface.npz', allow_pickle=True); v, f, roles = nat['vertices'], nat['faces'].astype(np.int64), nat['face_roles']
    sid = np.load(case/'difference.npz')['source_face_id'].astype(np.int64)
    host = roles == 'original_host'
    tri = v[f]; cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); fn = cr / np.linalg.norm(cr, axis=1)[:, None]
    corner = np.repeat(fn[:, None, :], 3, axis=1)
    h = np.flatnonzero(host); st = sv[sf[sid[h]]]; p = tri[h]
    e1, e2 = st[:, 1] - st[:, 0], st[:, 2] - st[:, 0]
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1); den = d00 * d11 - d01 * d01
    q = p - st[:, 0, None, :]; d20, d21 = (q * e1[:, None]).sum(2), (q * e2[:, None]).sum(2)
    bv = (d11[:, None] * d20 - d01[:, None] * d21) / den[:, None]; bw = (d00[:, None] * d21 - d01[:, None] * d20) / den[:, None]
    bary = np.stack([1 - bv - bw, bv, bw], 2)
    if bary.min() < -1e-6 or bary.max() > 1 + 1e-6: raise ValueError('host corner outside its declared refined source face')
    bary = np.clip(bary, 0, 1); bary /= bary.sum(2, keepdims=True)
    interp = np.einsum('fij,fjk->fik', bary, sn[sf[sid[h]]]); interp /= np.linalg.norm(interp, axis=2)[:, :, None]
    behind = np.einsum('fij,fj->fi', interp, fn[h]) <= 1e-3
    interp[behind] = np.repeat(fn[h][:, None, :], 3, axis=1)[behind]
    corner[h] = interp
    rep = dict(method=__doc__, field=str(field_npz), host_faces=int(len(h)), flat_fallback_corners=int(behind.sum()), cavity_faces=int((~host).sum()), roles={})
    for role in ('left_wall', 'right_wall', 'floor'):
        idx = np.flatnonzero(roles == role)
        if not len(idx): continue
        vn = np.zeros_like(v)
        for j in range(3):
            np.add.at(vn, f[idx, j], cr[idx])
        vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]
        c = vn[f[idx]]
        bh = np.einsum('fij,fj->fi', c, fn[idx]) <= 1e-3
        c[bh] = np.repeat(fn[idx][:, None, :], 3, axis=1)[bh]
        corner[idx] = c
        rep['roles'][role] = dict(faces=int(len(idx)), flat_fallback_corners=int(bh.sum()))
    display = dict(nat); display['vertices'] = tri.reshape(-1, 3); display['faces'] = np.arange(3 * len(f)).reshape(-1, 3); display['normals'] = corner.reshape(-1, 3)
    np.savez_compressed(out/'globaldisplay_surface.npz', **display)
    dev = np.degrees(np.arccos(np.clip(np.einsum('fij,fj->fi', interp, fn[h]), -1, 1)))
    rep.update(max_normal_change_deg=float(dev.max()), p99_normal_change_deg=float(np.percentile(dev, 99)))
    (out/'shading_report.json').write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps({k: v for k, v in rep.items() if k != 'method'}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
