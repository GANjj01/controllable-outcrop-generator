"""Task-specific native fracture-mouth candidates, never automatic E_syn acceptance.

Only host/cavity shared edges with explicit fracture provenance are considered.
Ordinary triangle edges, shading groups, RGB gradients and shadows create no F.
"""
from fractions import Fraction
import re
import numpy as np
from frgb.raster import camera_triangles, rays


def extract_mouth_edges(vertices, faces, roles, source_ids, trace_ids):
    v = np.asarray(vertices, float); f = np.asarray(faces, np.int64)
    roles, source_ids = np.asarray(roles).astype(str), np.asarray(source_ids).astype(str)
    if len(roles) != len(f) or len(source_ids) != len(f):
        raise ValueError('One role and source identity required per native face')
    oriented = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    unique, inverse, counts = np.unique(np.sort(oriented, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError('Native indexed mesh must be closed before mouth extraction')
    winding = np.bincount(inverse, weights=np.where(oriented[:, 0] < oriented[:, 1], 1, -1))
    if np.any(winding != 0):
        raise ValueError('Native closed mesh has inconsistent edge winding')
    pairs = np.argsort(inverse, kind='stable').reshape(-1, 2) // 3
    host = roles == 'original_host'
    take = np.flatnonzero(host[pairs[:, 0]] != host[pairs[:, 1]])
    cavity = np.where(host[pairs[take, 0]], pairs[take, 1], pairs[take, 0])
    path, segment, trace = [], [], []
    for face in cavity:
        match = re.fullmatch(r'path(\d+):segment(\d+)', source_ids[face])
        if match is None or int(match[1]) not in trace_ids:
            raise ValueError('Mouth cavity face has unresolved fracture identity: ' + source_ids[face])
        path.append(int(match[1])); segment.append(int(match[2])); trace.append(trace_ids[path[-1]])
    role = roles[cavity]
    return dict(vertices=unique[take], faces=pairs[take], native_edge_id=take, cavity_face=cavity,
                cavity_role=role, path_index=np.asarray(path, np.int32), segment_id=np.asarray(segment, np.int64),
                trace_id=np.asarray(trace, dtype='<U64'), candidate=np.isin(role, ['left_wall', 'right_wall']),
                length_m=np.linalg.norm(v[unique[take, 1]] - v[unique[take, 0]], axis=1))


def ray_pair_hits(triangles, origin, directions, forward, near):
    """One two-sided ray per triangle pair; view-axis depth with exact fallback.

    No fixed absolute determinant/area threshold discards triangles."""
    t = np.asarray(triangles, float); d = np.broadcast_to(np.asarray(directions, float), (len(t), 3))
    origin, forward = np.asarray(origin, float), np.asarray(forward, float)
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    h = np.cross(d, e2); det = np.einsum('ij,ij->i', e1, h)
    scale = np.linalg.norm(e1, axis=1)*np.linalg.norm(e2, axis=1)*np.linalg.norm(d, axis=1)
    uncertain = np.abs(det) <= 64*np.finfo(float).eps*scale
    inv = np.divide(1., det, out=np.zeros_like(det), where=~uncertain)
    s = origin - t[:, 0]; q = np.cross(s, e1)
    u = np.einsum('ij,ij->i', s, h)*inv
    w = np.einsum('ij,ij->i', d, q)*inv
    along = np.einsum('ij,ij->i', q, e2)*inv
    depth = along*(d @ forward)
    bary = np.column_stack([1-u-w, u, w])
    border = (bary.min(1) >= -1e-8) & (bary.max(1) <= 1+1e-8) & (np.abs(bary).min(1) <= 1e-8)
    fallback = uncertain | border | (np.abs(depth-near) <= 1e-10)
    answer = np.where((~fallback) & (bary.min(1) >= 0) & (depth >= near), depth, np.inf)
    def vec(a): return tuple(Fraction(float(x)) for x in a)
    def sub(a, b): return tuple(x-y for x, y in zip(a, b))
    def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def dot(a, b): return sum(x*y for x, y in zip(a, b))
    if fallback.any():
        o, fw, clip = vec(origin), vec(forward), Fraction(float(near))
        for i in np.flatnonzero(fallback):
            a, b, c = map(vec, t[i]); dd = vec(d[i]); one, two = sub(b, a), sub(c, a)
            hh = cross(dd, two); denominator = dot(one, hh)
            if denominator == 0:
                continue
            sv = sub(o, a); qq = cross(sv, one)
            uu, vv = dot(sv, hh)/denominator, dot(dd, qq)/denominator
            zz = dot(two, qq)/denominator*dot(dd, fw)
            if uu >= 0 and vv >= 0 and uu+vv <= 1 and zz >= clip:
                answer[i] = float(zz); bary[i] = [float(1-uu-vv), float(uu), float(vv)]
    return answer, bary


def _projected_boxes(vertices, faces, cam):
    tri, ids, crossing = camera_triangles(vertices, faces, cam)
    x = cam['w']/2 + cam['fpx']*tri[..., 0]/tri[..., 2] - .5
    y = cam['h']/2 - cam['fpx']*tri[..., 1]/tri[..., 2] - .5
    lo, hi = np.column_stack([x.min(1), y.min(1)]), np.column_stack([x.max(1), y.max(1)])
    pad = 64*np.finfo(float).eps*(1+np.maximum(np.abs(lo), np.abs(hi)))
    return lo-pad, hi+pad, ids, crossing


def rasterize_native(vertices, faces, cam, max_pairs=250000):
    """Native-grid direct nearest rays with conservative projected-AABB bounds."""
    v, f = np.asarray(vertices), np.asarray(faces, np.int64)
    lo, hi, ids, crossing = _projected_boxes(v, f, cam)
    x0 = np.clip(np.ceil(lo[:, 0]), 0, cam['w']).astype(np.int64)
    x1 = np.clip(np.floor(hi[:, 0]), -1, cam['w']-1).astype(np.int64)
    y0 = np.clip(np.ceil(lo[:, 1]), 0, cam['h']).astype(np.int64)
    y1 = np.clip(np.floor(hi[:, 1]), -1, cam['h']-1).astype(np.int64)
    bw, bh = x1-x0+1, y1-y0+1
    size = bw*bh; selected = np.flatnonzero((bw > 0) & (bh > 0)); selected = selected[np.argsort(size[selected], kind='stable')]
    depth = np.full(cam['w']*cam['h'], np.inf); face = np.full(len(depth), -1, np.int64)
    bmin = np.full(len(depth), np.nan); view_cos = np.full(len(depth), np.nan)
    cross = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    normals = cross/np.linalg.norm(cross, axis=1)[:, None]
    pairs = 0
    def process(sel, start=0, stop=None):
        nonlocal pairs
        if len(sel) == 1 and stop is not None:
            t = np.full(stop-start, sel[0]); k = np.arange(start, stop)
        else:
            counts = size[sel]; t = np.repeat(sel, counts)
            k = np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts, counts)
        xx, yy = x0[t]+k % bw[t], y0[t]+k//bw[t]
        fid = ids[t]; direction = rays(cam, xx, yy)
        z, bary = ray_pair_hits(v[f[fid]], cam['pos'], direction, cam['fwd'], cam['near'])
        pairs += len(z); pixel = yy*cam['w']+xx
        np.minimum.at(depth, pixel, z)
        win = np.isfinite(z) & (z == depth[pixel])
        pixel, fid = pixel[win], fid[win]
        face[pixel] = fid; bmin[pixel] = bary[win].min(1)
        view_cos[pixel] = np.abs(np.einsum('ij,ij->i', normals[fid], direction[win]))/np.linalg.norm(direction[win], axis=1)
    start = 0
    while start < len(selected):
        if size[selected[start]] > max_pairs:
            current = selected[start:start+1]
            for offset in range(0, int(size[current[0]]), max_pairs):
                process(current, offset, min(offset+max_pairs, int(size[current[0]])))
            start += 1; continue
        stop = start+1; total = int(size[selected[start]])
        while stop < len(selected) and total+size[selected[stop]] <= max_pairs:
            total += int(size[selected[stop]]); stop += 1
        process(selected[start:stop]); start = stop
    shape = (cam['h'], cam['w'])
    return dict(depth=depth.reshape(shape), face=face.reshape(shape), barycentric_min=bmin.reshape(shape),
                view_cos=view_cos.reshape(shape), ray_triangle_pairs=pairs, near_crossing_faces=crossing)


class NativeRayIndex:
    """Complete-mesh projected bounds for subpixel mouth-point occlusion rays."""
    def __init__(self, vertices, faces, cam, cell=16):
        self.v, self.f, self.cam, self.cell = np.asarray(vertices), np.asarray(faces), cam, cell
        lo, hi, ids, _ = _projected_boxes(vertices, faces, cam)
        keep = (hi[:, 0] >= 0) & (lo[:, 0] <= cam['w']-1) & (hi[:, 1] >= 0) & (lo[:, 1] <= cam['h']-1)
        self.lo, self.hi, self.ids = lo[keep], hi[keep], ids[keep]
        self.buckets = {}
        for i, (a, b) in enumerate(zip(self.lo, self.hi)):
            lower = np.floor(np.maximum(a, 0)/cell).astype(int)
            upper = np.floor(np.minimum(b, [cam['w']-1, cam['h']-1])/cell).astype(int)
            for y in range(lower[1], upper[1]+1):
                for x in range(lower[0], upper[0]+1):
                    self.buckets.setdefault((x, y), []).append(i)

    def nearest(self, uv):
        uv = np.asarray(uv, float); z = np.full(len(uv), np.inf); face = np.full(len(uv), -1, np.int64)
        for i, point in enumerate(uv):
            possible = np.asarray(self.buckets.get(tuple(np.floor(point/self.cell).astype(int)), []), np.int64)
            if not len(possible): continue
            possible = possible[np.all((point >= self.lo[possible]) & (point <= self.hi[possible]), axis=1)]
            ids = np.unique(self.ids[possible])
            if not len(ids): continue
            direction = rays(self.cam, point[0], point[1])
            hits, _ = ray_pair_hits(self.v[self.f[ids]], self.cam['pos'], direction, self.cam['fwd'], self.cam['near'])
            j = int(hits.argmin())
            if np.isfinite(hits[j]): z[i], face[i] = hits[j], ids[j]
        return z, face


def sample_edges(vertices, edges, cam, spacing_px=.5):
    """Near/frame clip in homogeneous camera space, then uniform screen samples."""
    points, owner, uv, depth = [], [], [], []
    for i, pair in enumerate(edges['vertices']):
        world = np.asarray(vertices)[pair]; rel = world-cam['pos']
        a = np.column_stack([rel @ cam['right'], rel @ cam['up'], rel @ cam['fwd']])
        left, right = 0., 1.
        constraints = [a[:, 2]-cam['near']]
        for axis, span in [(0, cam['w']), (1, cam['h'])]:
            constraints.extend([(span/2-.5)*a[:, 2]+cam['fpx']*a[:, axis],
                                (span/2-.5)*a[:, 2]-cam['fpx']*a[:, axis]])
        for values in constraints:
            slope = values[1]-values[0]
            if slope > 0: left = max(left, -values[0]/slope)
            elif slope < 0: right = min(right, -values[0]/slope)
            elif values[0] < 0: left, right = 1., 0.; break
        if right <= left: continue
        ends = world[0]+np.array([left, right])[:, None]*(world[1]-world[0])
        r = ends-cam['pos']; zz = r @ cam['fwd']
        screen = np.column_stack([cam['w']/2+cam['fpx']*(r @ cam['right'])/zz-.5,
                                  cam['h']/2-cam['fpx']*(r @ cam['up'])/zz-.5])
        length = float(np.linalg.norm(screen[1]-screen[0]))
        count = max(2, int(np.ceil(length/spacing_px))+1)
        s = np.linspace(0, 1, count)
        t = s*zz[0]/(zz[1]*(1-s)+s*zz[0])
        p = ends[0]+t[:, None]*(ends[1]-ends[0])
        points.append(p); owner.append(np.full(count, i, np.int64))
        uv.append(screen[0]+s[:, None]*(screen[1]-screen[0])); depth.append((p-cam['pos']) @ cam['fwd'])
    if not points:
        return dict(points=np.empty((0, 3)), owner=np.empty(0, np.int64), uv=np.empty((0, 2)), depth=np.empty(0))
    return dict(points=np.vstack(points), owner=np.concatenate(owner), uv=np.vstack(uv), depth=np.concatenate(depth))


def vertex_face_csr(faces, vertex_count):
    faces = np.asarray(faces, np.int64); order = np.argsort(faces.ravel(), kind='stable')
    counts = np.bincount(faces.ravel(), minlength=vertex_count)
    return np.r_[0, np.cumsum(counts)].astype(np.int64), (order // 3).astype(np.int64)


def raster_targets(candidate, rock, aligned, roles, *, pending_u=None, material_u=None, protect=None):
    candidate, rock, aligned = map(lambda x: np.asarray(x, bool), (candidate, rock, aligned))
    roles = np.asarray(roles).astype(str)
    if candidate.shape != rock.shape or aligned.shape != rock.shape or roles.shape != rock.shape:
        raise ValueError('Labels must share the native 2D pixel grid')
    ignore = ~aligned
    for mask in (pending_u, material_u):
        if mask is not None:
            if np.asarray(mask).shape != rock.shape: raise ValueError('U must occupy the native grid')
            ignore |= np.asarray(mask, bool)
    valid = rock & ~ignore
    target = candidate & valid
    negative = valid & ~target & (roles == 'original_host')
    if protect is not None: negative &= ~np.asarray(protect, bool)
    return dict(target=target, valid=valid, ignore=rock & ~valid, mechanism_negative=negative,
                qualified=False, status='CPU_GEOMETRIC_CANDIDATE_RGB_AND_COVERAGE_ACCEPTANCE_PENDING')


def qualify_mouth_samples(samples, edges, first_depth, first_face, ue_depth_m, ue_rock, aligned, vertex_faces=None):
    """Separate hidden samples from unresolved/visible; never infer F from RGB.

    vertex_faces: (indptr, indices) CSR of the faces incident to each native vertex. A sample whose nearest hit is a face
    sharing one of its edge's two end vertices is then also owned by that edge: at a shared vertex the hit point is
    the same point, only the face index is a tie. Needed for millimetre mouth edges (view-adaptive meshes), where
    end-point ties dominate the samples."""
    uv = np.asarray(samples['uv']); owner = samples['owner']; expected = samples['depth']
    h, w = np.asarray(ue_rock).shape
    pixel = np.floor(uv+.5).astype(np.int64)
    inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    x, y = np.clip(pixel[:, 0], 0, w-1), np.clip(pixel[:, 1], 0, h-1)
    first_depth, first_face = np.asarray(first_depth), np.asarray(first_face)
    agrees = np.isfinite(first_depth) & (np.abs(first_depth-expected) <= 1e-7)
    own = (edges['faces'][owner] == first_face[:, None]).any(1)
    if vertex_faces is not None:
        indptr, indices = vertex_faces
        for k in (0, 1):
            vert = edges['vertices'][owner, k]; start, stop = indptr[vert], indptr[vert + 1]
            width = int((stop - start).max(initial=0))
            if width:
                cols = start[:, None] + np.arange(width)[None, :]
                inc = np.where(cols < stop[:, None], indices[np.minimum(cols, len(indices) - 1)], -1)
                own |= (inc == first_face[:, None]).any(1)
    hidden = inside & np.isfinite(first_depth) & (first_depth < expected-1e-7)
    native_visible = inside & agrees & own
    registered = np.asarray(ue_rock)[y, x] & np.asarray(aligned)[y, x] & np.isfinite(np.asarray(ue_depth_m)[y, x])
    visible = native_visible & registered
    return dict(pixel_xy=pixel, visible=visible, hidden=hidden,
                uncertain=inside & ~hidden & ~visible, native_visible=native_visible,
                candidate=visible & edges['candidate'][owner], pending=visible & ~edges['candidate'][owner])


def material_u_from_readback(readiness, shape, explicit=None):
    """Empty material U requires actual verified baseline-gray graph evidence."""
    if explicit is not None:
        mask = np.asarray(explicit)
        if mask.shape != tuple(shape) or not np.isin(mask, [0, 1]).all():
            raise ValueError('Explicit material U must be binary on the native image grid')
        return mask.astype(bool), 'explicit_mapped_material_u'
    gray = [row for row in readiness.get('gray_maps', []) if row['mode'] == 'Lit']
    if not readiness.get('success') or len(gray) != 1 or not gray[0]['materials']:
        raise ValueError('No verified constant-gray material: explicit mapped material U is required')
    for record in gray[0]['materials'].values():
        if (not np.allclose(record['basecolor_linear'], [.22]*3, atol=1e-6, rtol=0)
                or abs(record['roughness']-.9) > 1e-6
                or any(record.get(k) is not None for k in ['normal', 'ambient_occlusion', 'world_position_offset'])):
            raise ValueError('Material is not the verified baseline gray; supply its reviewed material U')
    return np.zeros(shape, bool), 'verified_uniform_gray_no_painted_texture'
