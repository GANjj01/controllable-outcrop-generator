"""Optional exact C++ predicates; backend policy is separate from model parameters."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import importlib
import os


_MODES = ('auto', 'python', 'native', 'cpu_batch', 'gpu', 'cpp_clip')
_mode = ContextVar('outcrop_geometry_backend', default=None)
_filter_stats = ContextVar('outcrop_geometry_filter_stats', default=None)


def _validate(mode):
    if mode not in _MODES:
        raise ValueError(f'geometry backend must be one of {_MODES}, got {mode!r}')
    return mode


_default_mode = _validate(os.environ.get('OUTCROP_GEOMETRY_BACKEND', 'auto'))


def backend_mode():
    return _mode.get() or _default_mode


@contextmanager
def geometry_backend(mode):
    """Select a backend for this context without modifying a geological declaration."""
    token = _mode.set(_validate(mode))
    statistics = _filter_stats.set(None)
    try:
        if mode in ('cpu_batch', 'gpu', 'cpp_clip'):
            get_native()
        if mode == 'gpu':
            from .gpu_filters import gpu_available
            if not gpu_available():
                raise RuntimeError('GPU geometry backend is not available; CUDA device and CuPy required')
        yield
    finally:
        _filter_stats.reset(statistics)
        _mode.reset(token)


@lru_cache(maxsize=1)
def _load_native():
    try:
        module = importlib.import_module('_outcrop_exact')
    except ModuleNotFoundError as exc:
        if exc.name == '_outcrop_exact':
            return None
        raise
    if module.API_VERSION != 4:
        raise RuntimeError('unsupported _outcrop_exact API; rebuild for this source tree')
    return module


def native_available():
    return _load_native() is not None


def get_native():
    mode = backend_mode()
    if mode == 'python':
        return None
    module = _load_native()
    if module is None and mode in ('native', 'cpu_batch', 'gpu', 'cpp_clip'):
        raise RuntimeError('native geometry backend is not available; run scripts/build_native_geometry.py')
    return module


def native_info():
    module = get_native()
    stats = dict(filter_statistics())
    selected = 'native' if module else 'python'
    if any(stats[k] for k in ('cpp_clip_calls','cpp_split_calls','cpp_chain_calls','cpp_bisect_calls','cpp_arrange_calls')):
        selected += '+exact-cuts'
    if stats['gpu_launches']:
        selected += '+gpu'
    result = {'requested': backend_mode(), 'selected': selected,
            'api_version': module.API_VERSION if module else None,
            'module': module.__file__ if module else None, 'filters': stats}
    if stats['gpu_launches']:
        from .gpu_filters import gpu_info
        result['gpu'] = gpu_info()
    return result


def filter_statistics():
    """Actual work since this execution context began; no device probing here."""
    stats = _filter_stats.get()
    if stats is None:
        stats = {'gpu_launches': 0, 'gpu_pairs': 0, 'gpu_device_ms': 0.,
                 'filter_seconds': 0., 'unknown_pairs': 0,
                 'cpu_batches': 0, 'facets': 0, 'cpp_candidate_rows': 0,
                 'cpp_clip_calls': 0, 'cpp_split_calls': 0, 'cpp_chain_calls': 0,
                 'cpp_bisect_calls': 0, 'cpp_arrange_calls': 0}
        _filter_stats.set(stats)
    return stats


def facet_batch_enabled(face_count):
    mode = backend_mode()
    return mode in ('cpu_batch', 'gpu', 'cpp_clip') or (
        mode == 'auto' and face_count >= 128 and get_native() is not None)


def exact_cut_native():
    """Old explicit modes remain a reference for the new arithmetic loops."""
    return get_native() if backend_mode() in ('auto', 'cpp_clip') else None
