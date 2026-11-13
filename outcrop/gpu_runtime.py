"""Configure optional CUDA caches before any project consumer imports CuPy."""
import os
import sys
import tempfile
from pathlib import Path


def configure_gpu_cache():
    # CuPy14 fixes its disk backend during import. All lazy GPU entry points
    # share this setup; changing only the later geometry import is too late.
    if 'cupy' in sys.modules and sys.modules['cupy'] is None:
        return
    cache = Path(os.environ.get('FRGB_GPU_CACHE') or Path(tempfile.gettempdir()) / 'geometry-gpu-cache')
    for variable, leaf in (('CUPY_CACHE_DIR', 'cupy'), ('CUDA_CACHE_PATH', 'cuda')):
        selected = Path(os.environ.setdefault(variable, str(cache / leaf)))
        if selected == cache / leaf:
            selected.mkdir(parents=True, exist_ok=True)
