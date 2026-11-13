"""Keep every importing process on the CPU. The generator's optional CuPy paths (dsgen.display.segment_distance,
dsgen.semantics, modeling.kernel.gpu_filters) all fall back to NumPy on ImportError; a None entry in sys.modules makes
`import cupy` raise it. The GPU segment distance is also float32 relative to the batch mean, i.e. batch dependent,
which breaks the exact evaluator's reproduction. Import first."""
import sys

if 'cupy' not in sys.modules or sys.modules['cupy'] is not None:
    if 'cupy' in sys.modules:
        raise RuntimeError('cupy was imported before nogpu')
    sys.modules['cupy'] = None
