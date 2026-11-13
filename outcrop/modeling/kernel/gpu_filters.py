"""Batched conservative plane signs; zero always requires the exact predicate.

CuPy/CUDA are imported only for an explicit nonempty GPU batch or availability
query. The geological caller retains the original rational planes and points.
"""
from functools import lru_cache
import os
from time import perf_counter

import numpy as np

from .predicates import _fl, _FILTER_ABS, _FILTER_REL


# Bound temporary CPU products and GPU output chunks, independently of the
# final caller-owned planes-by-points table.
_MAX_CHUNK_PAIRS = 1024 * 1024

_SOURCE = r'''
extern "C" __global__ void plane_signs(
    const double* normals, const double* offsets, const double* points,
    unsigned long long point_count, unsigned long long start,
    unsigned long long count, signed char* result) {
    unsigned long long local = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (local >= count) return;
    unsigned long long pair = start + local;
    unsigned long long plane = pair / point_count, point = pair % point_count;
    double n0=normals[3*plane], n1=normals[3*plane+1], n2=normals[3*plane+2];
    double p0=points[3*point], p1=points[3*point+1], p2=points[3*point+2];
    double offset=offsets[plane];
    result[local]=0;
    if (!(isfinite(n0) && isfinite(n1) && isfinite(n2) && isfinite(offset)
          && isfinite(p0) && isfinite(p1) && isfinite(p2))) return;
    double a=__dmul_rn(n0,p0), b=__dmul_rn(n1,p1), c=__dmul_rn(n2,p2);
    double ab=__dadd_rn(a,b), abc=__dadd_rn(ab,c), value=__dsub_rn(abc,offset);
    double absolute=__dadd_rn(__dadd_rn(__dadd_rn(fabs(a),fabs(b)),fabs(c)),fabs(offset));
    double scale=__dadd_rn(__dadd_rn(fabs(n0),fabs(n1)),fabs(n2));
    scale=__dadd_rn(scale,fabs(p0));
    scale=__dadd_rn(scale,fabs(p1));
    scale=__dadd_rn(scale,fabs(p2));
    scale=__dadd_rn(scale,1.0);
    double bound=__dadd_rn(__dmul_rn(1e-14,absolute),__dmul_rn(1e-300,scale));
    if (!(isfinite(a) && isfinite(b) && isfinite(c) && isfinite(ab)
          && isfinite(abc) && isfinite(value) && isfinite(bound) && bound<1e300)) return;
    if (value>bound) result[local]=1;
    else if (value<-bound) result[local]=-1;
}
'''


@lru_cache(maxsize=1)
def _cupy():
    from ...gpu_runtime import configure_gpu_cache
    configure_gpu_cache()
    import cupy
    return cupy


@lru_cache(maxsize=1)
def _kernel():
    # FP64 RN intrinsics prevent contraction as well. CuPy's appended -ftz=true
    # controls float32 only; the filter uses no float32 arithmetic.
    kernel = _cupy().RawKernel(_SOURCE, 'plane_signs', options=('--std=c++11', '--fmad=false'))
    # Compile before recording CUDA events: cold JIT belongs to host elapsed
    # time, while gpu_device_ms records execution rather than idle JIT time.
    kernel.compile()
    return kernel


def gpu_available():
    """Report optional CUDA availability without compiling or launching a kernel."""
    try:
        cp = _cupy()
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def gpu_info():
    """Return device/runtime metadata without compiling or launching a kernel."""
    try:
        cp = _cupy()
        device = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props['name']
        if isinstance(name, bytes):
            name = name.decode()
        return {'available': True, 'device_id': device, 'name': name,
                'capability': (props['major'], props['minor']), 'cupy': cp.__version__,
                'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
                'cuda_driver': cp.cuda.runtime.driverGetVersion(),
                'cupy_cache_dir': os.environ.get('CUPY_CACHE_DIR'),
                'cuda_cache_path': os.environ.get('CUDA_CACHE_PATH')}
    except Exception as exc:
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}


def _inputs(normals, offsets, exact_points):
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError('normals must have shape (planes, 3)')
    if offsets.ndim != 1 or offsets.shape[0] != normals.shape[0]:
        raise ValueError('offsets must have one value for each plane')
    rows = tuple(exact_points)
    points = np.empty((len(rows), 3), dtype=np.float64)
    for i, row in enumerate(rows):
        row = tuple(row)
        if len(row) != 3:
            raise ValueError('points must have three coordinates')
        for j, value in enumerate(row):
            try:
                points[i, j] = _fl(value)
            except OverflowError:
                points[i, j] = np.nan
    return normals, offsets, points


def _cpu(normals, offsets, points, result):
    # NumPy products are bounded even if the requested final result is large.
    plane_step = min(len(normals), _MAX_CHUNK_PAIRS)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for first in range(0, len(normals), plane_step):
            n = normals[first:first + plane_step]
            o = offsets[first:first + plane_step]
            point_step = max(1, _MAX_CHUNK_PAIRS // len(n))
            for start in range(0, len(points), point_step):
                p = points[start:start + point_step]
                prod = n[:, None, :] * p[None, :, :]
                a, b, c = prod[..., 0], prod[..., 1], prod[..., 2]
                ab = a + b
                abc = ab + c
                value = abc - o[:, None]
                bound = _FILTER_REL * (np.abs(prod).sum(-1) + np.abs(o)[:, None])
                bound += _FILTER_ABS * (np.abs(n).sum(1)[:, None]
                                       + np.abs(p).sum(1)[None, :] + 1.)
                finite = (np.isfinite(prod).all(-1) & np.isfinite(ab)
                          & np.isfinite(abc) & np.isfinite(value) & np.isfinite(bound)
                          & (bound < 1e300))
                result[first:first + len(n), start:start + len(p)] = np.where(
                    finite & (value > bound), np.int8(1),
                    np.where(finite & (value < -bound), np.int8(-1), np.int8(0)))


def _add(stats, name, amount):
    if stats is not None:
        stats[name] = stats.get(name, 0) + amount


def _gpu(normals, offsets, points, result, stats):
    cp = _cupy()
    kernel = _kernel()
    gn, go, gp = cp.asarray(normals), cp.asarray(offsets), cp.asarray(points)
    flat = result.reshape(-1)
    output = cp.empty(min(flat.size, _MAX_CHUNK_PAIRS), dtype=cp.int8)
    if stats is not None:
        begin, end = cp.cuda.Event(), cp.cuda.Event()
    for start in range(0, flat.size, _MAX_CHUNK_PAIRS):
        count = min(flat.size - start, _MAX_CHUNK_PAIRS)
        if stats is not None:
            begin.record()
        kernel(((count + 255) // 256,), (256,),
               (gn, go, gp, np.uint64(len(points)), np.uint64(start), np.uint64(count), output))
        if stats is not None:
            end.record()
        cp.asnumpy(output[:count], out=flat[start:start + count])
        _add(stats, 'gpu_launches', 1)
        _add(stats, 'gpu_pairs', count)
        if stats is not None:
            _add(stats, 'gpu_device_ms', cp.cuda.get_elapsed_time(begin, end))


def filtered_signs_batch(N, O, exact_points, backend='cpu', stats=None):
    """Return int8 signs of shape (planes, points); zero is always undecided.

    Normal/offset doubles must be the correctly rounded stored coefficients.
    Query points are converted once using the same CPython conversion as the
    scalar predicate. Explicit GPU errors propagate; exact fallback belongs to
    the caller and must use its retained rational source data.
    """
    if backend not in ('cpu', 'gpu'):
        raise ValueError('backend must be cpu or gpu')
    began = perf_counter()
    try:
        normals, offsets, points = _inputs(N, O, exact_points)
        result = np.empty((len(normals), len(points)), dtype=np.int8)
        if result.size:
            if backend == 'gpu':
                _gpu(normals, offsets, points, result, stats)
            else:
                _cpu(normals, offsets, points, result)
        if stats is not None:
            _add(stats, 'unknown_pairs', int(result.size - np.count_nonzero(result)))
        return result
    finally:
        _add(stats, 'filter_seconds', perf_counter() - began)
