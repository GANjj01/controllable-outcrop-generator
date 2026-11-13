"""Shared-vertex broadphase and exact C++ source confirmation for fixed planes."""
import numpy as np

from .gpu_filters import filtered_signs_batch, gpu_available
from .native import backend_mode, filter_statistics, get_native
from .predicates import plane_rows


# Complete A/B cold builds favour CPU batching; C has larger reusable batches.
# Keep smaller batches on CPU. Explicit GPU remains available for measurements.
AUTO_GPU_PAIRS = 512 * 1024
MAX_BATCH_PAIRS = 4 * 1024 * 1024


def _blocks(faces, row_count, pair_limit):
    block=[];vertices=0
    for item in faces:
        size=len(item[1][0][1])
        if block and row_count*(vertices+size)>pair_limit:
            yield block;block=[];vertices=0
        block.append(item);vertices+=size
    if block:yield block


def fixed_facet_sources(faces, candidates, request):
    """Yield original source-ID sets in facet order, retaining coincident rows.

    Only use when source cells share the same fixed planes. Local nonlinear
    proxies continue through the original per-source-cell path.
    """
    native=get_native()
    prepared=native.prepare_planes(tuple((p.normal,p.offset) for p in candidates))
    N,O=plane_rows(candidates);stats=filter_statistics();mode=backend_mode()
    pair_limit=max(1,min(MAX_BATCH_PAIRS,request.budget.max_bytes//16))
    for block in _blocks(faces,len(candidates),pair_limit):
        lookup={};points=[];indices=[]
        for _,entries in block:
            ids=[]
            for point in entries[0][1]:
                if point not in lookup:
                    lookup[point]=len(points);points.append(point)
                ids.append(lookup[point])
            indices.append(ids)
        pairs=len(candidates)*len(points)
        # A single exceptionally wide facet can exceed the temporary budget.
        # Keep its original scalar filter, rather than allocate an unbounded table.
        if pairs>pair_limit:
            from .predicates import possibly_on_planes
            for _,entries in block:
                poly=entries[0][1]
                maybe=np.flatnonzero(possibly_on_planes(N,O,poly)).tolist()
                kept=native.planes_on_polygon(prepared,maybe,poly)
                stats['facets']+=1;stats['cpp_candidate_rows']+=len(maybe)
                yield tuple(sorted({candidates[j].source_id for j in kept}))
            continue
        use_gpu=mode=='gpu' or (mode=='auto' and pairs>=AUTO_GPU_PAIRS and gpu_available())
        backend='gpu' if use_gpu else 'cpu'
        signs=filtered_signs_batch(N,O,points,backend=backend,stats=stats)
        if not use_gpu:stats['cpu_batches']+=1
        for (_,entries),ids in zip(block,indices):
            poly=entries[0][1]
            maybe=np.flatnonzero(~np.any(signs[:,ids],axis=1)).tolist()
            kept=native.planes_on_polygon(prepared,maybe,poly)
            stats['facets']+=1;stats['cpp_candidate_rows']+=len(maybe)
            yield tuple(sorted({candidates[j].source_id for j in kept}))
