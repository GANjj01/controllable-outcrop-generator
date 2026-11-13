"""Visible semantic classes and their ID-pass codes.

Geological truth (v2) and visible semantics are kept apart: bedrock pixels carry v2 region codes 1..ROCK_CODE_MAX;
every other visible class gets a reserved code ``RESERVED_BASE + class id``."""
import numpy as np

CLASSES = ['background', 'bedrock', 'soil', 'debris', 'sediment', 'water', 'vegetation', 'manmade']
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
RESERVED_BASE = 50
ROCK_CODE_MAX = RESERVED_BASE - 1
RESERVED = {RESERVED_BASE + i: c for i, c in enumerate(CLASSES) if i >= 2}
# Bedrock under a non-bedrock pixel (soil plain, road, ...) is probed this far below its surface.
BEDROCK_PROBE_M = .5


def code_of(semantic):
    if semantic in ('bedrock', 'background'):
        raise ValueError('bedrock pixels carry v2 region codes; background is 0')
    return RESERVED_BASE + CLASS_ID[semantic]


def semantic_image(codes):
    """uint8 class image from an ID image (0 background, 1 bedrock, reserved -> class)."""
    out = np.zeros(codes.shape, np.uint8)
    out[(codes > 0) & (codes <= ROCK_CODE_MAX)] = 1
    for code, cls in RESERVED.items():
        out[codes == code] = CLASS_ID[cls]
    return out


def is_rock(codes):
    return (codes > 0) & (codes <= ROCK_CODE_MAX)


class VertexHash:
    """Voxel hash of mesh vertices for nearest_vertex_fast; build once, query many times."""

    def __init__(self, vertices, cell=.1):
        self.vertices, self.cell = vertices, cell
        keys = np.floor(vertices / cell).astype(np.int64)
        order = np.lexsort(keys.T[::-1])
        k = keys[order]
        flat = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)
        srt = np.argsort(flat, kind='stable')
        flat, self.order = flat[srt], order[srt]
        # One lookup per voxel: unique hashes with the [start, end) range of their vertices.
        self.hashes, self.start, counts = np.unique(flat, return_index=True, return_counts=True)
        self.end = self.start + counts

    def ranges(self, h):
        pos = np.minimum(np.searchsorted(self.hashes, h), len(self.hashes) - 1)
        hit = self.hashes[pos] == h
        return np.where(hit, self.start[pos], 0), np.where(hit, self.end[pos], 0)


# GPU version of the loop below, bit-identical to it: same voxel order (dx, dy, dz), same
# candidate order and cap, same hash (wrapping int64), distances with round-to-nearest operations
# in NumPy's order (sqrt((dx*dx + dy*dy) + dz*dz), no fused multiply-add).
_NN_KERNEL = r"""
extern "C" __global__ void nn_hash(const double* p, const long long* q, int n, const long long* hashes, int nh,
                                   const long long* start, const long long* end_, const long long* order,
                                   const double* v, int per_cell, long long* arg, double* best) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double px = p[3 * i], py = p[3 * i + 1], pz = p[3 * i + 2];
    double b = 1.0 / 0.0;
    long long a = -1;
    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++) {
        long long c0 = q[3 * i] + dx, c1 = q[3 * i + 1] + dy, c2 = q[3 * i + 2] + dz;
        long long h = (long long)(((unsigned long long)c0 * 73856093ULL) ^ ((unsigned long long)c1 * 19349663ULL)
                                  ^ ((unsigned long long)c2 * 83492791ULL));
        int lo = 0, hi = nh;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (hashes[mid] < h) lo = mid + 1; else hi = mid; }
        int pos = lo < nh ? lo : nh - 1;
        if (hashes[pos] != h) continue;
        long long s0 = start[pos], s1 = end_[pos];
        for (int j = 0; j < per_cell && s0 + j < s1; j++) {
            long long cand = order[s0 + j];
            double ex = __dsub_rn(v[3 * cand], px), ey = __dsub_rn(v[3 * cand + 1], py), ez = __dsub_rn(v[3 * cand + 2], pz);
            double d = __dsqrt_rn(__dadd_rn(__dadd_rn(__dmul_rn(ex, ex), __dmul_rn(ey, ey)), __dmul_rn(ez, ez)));
            if (d < b) { b = d; a = cand; }
        }
    }
    arg[i] = a;
    best[i] = b;
}
"""
_NN = {}


def _nn_kernel():
    if 'kernel' not in _NN:
        from ..gpu_runtime import configure_gpu_cache
        configure_gpu_cache()
        try:
            import cupy as cp
            _NN['cp'], _NN['kernel'] = cp, cp.RawKernel(_NN_KERNEL, 'nn_hash')
        except Exception:  # no usable CUDA/cupy: NumPy loop
            _NN['kernel'] = None
    return _NN['kernel']


def nearest_vertex_fast(points, vertices, cell=.1, per_cell=12, index=None, gpu=True):
    """Vectorised nearest vertex for points lying on (or within ~cell of) the mesh: voxel hash,
    27 neighbouring voxels, at most ``per_cell`` candidates per voxel. Returns (index, distance).
    ``index``: a VertexHash of the same vertices and cell, reused across calls."""
    index = index or VertexHash(vertices, cell)
    order = index.order
    q = np.floor(points / cell).astype(np.int64)
    kernel = _nn_kernel() if gpu and len(points) and len(index.hashes) else None
    if kernel is not None:
        cp = _NN['cp']
        if getattr(index, '_gpu', None) is None:
            index._gpu = [cp.asarray(np.ascontiguousarray(x)) for x in
                          (index.hashes, index.start.astype(np.int64), index.end.astype(np.int64), order.astype(np.int64),
                           np.ascontiguousarray(vertices, np.float64))]
        gh, gs, ge, go, gv = index._gpu
        n = len(points)
        arg_g, best_g = cp.empty(n, cp.int64), cp.empty(n, cp.float64)
        kernel(((n + 255) // 256,), (256,), (cp.asarray(np.ascontiguousarray(points, np.float64)), cp.asarray(q), np.int32(n),
                                             gh, np.int32(len(index.hashes)), gs, ge, go, gv, np.int32(per_cell), arg_g, best_g))
        return cp.asnumpy(arg_g), cp.asnumpy(best_g)
    best = np.full(len(points), np.inf)
    arg = np.full(len(points), -1, np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = q + [dx, dy, dz]
                h = (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791)
                lo, hi = index.ranges(h)
                for j in range(per_cell):
                    idx = lo + j
                    ok = idx < hi
                    if not ok.any():
                        break
                    cand = order[np.where(ok, idx, 0)]
                    d = np.linalg.norm(vertices[cand] - points, axis=1)
                    better = ok & (d < best)
                    best[better] = d[better]
                    arg[better] = cand[better]
    return arg, best
