"""P1A native-grid public-label agreement, with explicit edge/trace semantics.

Both masks are clipped to valid before the same Zhang-Suen thinning. Length is
the sum of graph_node_weights: half each incident 8-neighbour Euclidean edge,
with redundant corner diagonals suppressed. Isolated pixels have zero length
and are separately reported. Neither thinning nor distance matching merges two
fracture edges into a physical centreline. No model inference is performed here.
"""
from collections import defaultdict
import importlib.util
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

_PRIMITIVES = Path(__file__).resolve().parent/'chain'/'evaluate_lines.py'
_spec = importlib.util.spec_from_file_location('_frgb_existing_line_primitives', _PRIMITIVES)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
graph_node_weights = _module.graph_node_weights
skeletonize = _module.skeletonize

GEO_SEMANTICS = 'geocrack_public_fracture_edges'
AUX_SEMANTICS = 'fraxet_native_traces_auxiliary'


def _binary(a, name):
    a = np.asarray(a)
    if a.ndim != 2 or not np.isin(a, (0, 1)).all():
        raise ValueError(name + ' must be a 2D binary array, not soft labels')
    return a.astype(bool)


def _ratio(a, b):
    return float(a / b) if b else None


def _evaluate(probability, target, valid, *, native_shape, threshold, semantics):
    g = _binary(target, 'target'); v = _binary(valid, 'valid')
    p = np.asarray(probability)
    if p.shape != g.shape or v.shape != g.shape or tuple(native_shape) != g.shape:
        raise ValueError('Prediction, target and valid must occupy the full native grid')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('An explicit fixed probability threshold in [0,1] is required')
    if not np.isfinite(p[v]).all() or np.any((p[v] < 0) | (p[v] > 1)):
        raise ValueError('Valid probabilities must be finite and in [0,1]')
    raw_p = (p >= threshold) & v
    raw_g = g & v
    ps, gs = skeletonize(raw_p), skeletonize(raw_g)
    pw, gw = graph_node_weights(ps), graph_node_weights(gs)
    dg = ndi.distance_transform_edt(~gs) if gs.any() else np.full(g.shape, np.inf)
    dp = ndi.distance_transform_edt(~ps) if ps.any() else np.full(g.shape, np.inf)
    nvalid = int(v.sum()); bg = v & ~raw_g
    nbg = int(bg.sum()); gl = float(gw.sum()); pl = float(pw.sum())
    tp = int((raw_p & raw_g).sum()); fp = int((raw_p & ~raw_g).sum())
    fn = int((~raw_p & raw_g).sum()); tn = int((~raw_p & ~raw_g & v).sum())
    pixel = dict(precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                 f1=_ratio(2 * tp, 2 * tp + fp + fn), true_positive_pixels=tp,
                 false_positive_pixels=fp, false_negative_pixels=fn, true_negative_pixels=tn,
                 false_positive_area_fraction=_ratio(fp, nvalid),
                 false_positive_background_fraction=_ratio(fp, nbg))
    recall_key = 'edge_recall_vs_public_labels' if semantics == GEO_SEMANTICS else 'edge_to_trace_recall_reference'
    density_key = 'fp_edge_density_vs_public_labels' if semantics == GEO_SEMANTICS else 'unmatched_edge_density_vs_trace_reference'
    tolerances = {}
    for tolerance in (1, 2, 3):
        matched_g = gs & (dp <= tolerance)
        unmatched = ps & bg & (dg > tolerance)
        unmatched_length = float(pw[unmatched].sum())
        tolerances[str(tolerance)] = {
            'tolerance_px': tolerance, recall_key: _ratio(float(gw[matched_g].sum()), gl),
            density_key: _ratio(unmatched_length * 1e6, nbg),
            'gt_graph_length_px': gl, 'prediction_graph_length_px': pl,
            'matched_gt_graph_length_px': float(gw[matched_g].sum()),
            'unmatched_prediction_graph_length_px': unmatched_length,
            'valid_background_pixels': nbg,
            'valid_background_outside_tolerance_pixels': int((bg & (dg > tolerance)).sum()),
            'unmatched_isolated_prediction_pixels': int((unmatched & (pw == 0)).sum()),
        }
    return dict(status='evaluated' if nvalid else 'not_evaluable_empty_valid_domain',
                label_semantics=semantics, native_shape=list(g.shape), threshold=float(threshold),
                valid_pixels=nvalid, ignored_pixels=int(v.size - nvalid),
                ignored_fraction=float(1 - nvalid / v.size), pixel=pixel,
                main=tolerances['2'], tolerances=tolerances,
                gt_isolated_skeleton_pixels=int((gs & (gw == 0)).sum()),
                prediction_isolated_skeleton_pixels=int((ps & (pw == 0)).sum()),
                length_definition='half incident Euclidean graph-edge length; redundant corner diagonals suppressed',
                fp_density_denominator='original public binary-mask background within valid; not physical rock area; tolerance band only excludes numerator',
                interpretation=('public fracture-edge label agreement' if semantics == GEO_SEMANTICS else
                                'auxiliary edge-to-native-trace distance reference; not physical trace recognition or merged two-sided edges'))


def evaluate_geocrack(probability, target, valid, *, native_shape, threshold):
    return _evaluate(probability, target, valid, native_shape=native_shape, threshold=threshold, semantics=GEO_SEMANTICS)


def evaluate_fraxet(probability, target, valid, *, native_shape, threshold):
    return _evaluate(probability, target, valid, native_shape=native_shape, threshold=threshold, semantics=AUX_SEMANTICS)


def evaluate_thresholds(probability, target, valid, *, native_shape, thresholds, semantics):
    """Does not invent a grid, choose a checkpoint, or run the model. Reuses the same supplied probability image. Main and
    auxiliary results stay separate."""
    if semantics not in (GEO_SEMANTICS, AUX_SEMANTICS):
        raise ValueError('An explicit supported label semantic is required')
    values = tuple(thresholds)
    if not values or len(values) != len(set(values)):
        raise ValueError('A nonempty unique predeclared threshold sequence is required')
    return [_evaluate(probability, target, valid, native_shape=native_shape, threshold=t,
                      semantics=semantics) for t in values]


def aggregate_sources(rows):
    """One already-normalized result per original image; image then source means.

    N/A values are excluded per metric and explicit image/source counts retained. Duplicate image identities and mixed
    edge/aux semantics are rejected."""
    rows = list(rows)
    if not rows:
        return dict(source_count=0, image_count=0, sources={}, source_equal_mean={})
    semantics = {r['metrics']['label_semantics'] for r in rows}
    if len(semantics) != 1:
        raise ValueError('GeoCrack and auxiliary trace metrics cannot be pooled')
    if len({r['metrics']['threshold'] for r in rows}) != 1:
        raise ValueError('All images must use the same fixed working-point threshold')
    identities = [(r['source_id'], r['image_id']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('One result per source/image is required; do not count patches or seeds as images')
    sem = next(iter(semantics))
    recall = 'edge_recall_vs_public_labels' if sem == GEO_SEMANTICS else 'edge_to_trace_recall_reference'
    density = 'fp_edge_density_vs_public_labels' if sem == GEO_SEMANTICS else 'unmatched_edge_density_vs_trace_reference'
    grouped = defaultdict(list)
    for row in rows:
        r = row['metrics']
        grouped[row['source_id']].append({recall: r['main'][recall], density: r['main'][density],
                                         **{'pixel_' + k: r['pixel'][k] for k in ('precision', 'recall', 'f1', 'false_positive_area_fraction')},
                                         'ignored_fraction': r['ignored_fraction']})
    keys = tuple(next(iter(grouped.values()))[0])
    sources = {}
    for source, images in grouped.items():
        values = {key: [r[key] for r in images if r[key] is not None] for key in keys}
        sources[source] = dict(image_count=len(images),
                               mean={k: float(np.mean(v)) if v else None for k, v in values.items()},
                               evaluable_image_counts={k: len(v) for k, v in values.items()})
    values = {k: [s['mean'][k] for s in sources.values() if s['mean'][k] is not None] for k in keys}
    return dict(label_semantics=sem, source_count=len(sources), image_count=len(rows), sources=sources,
                source_equal_mean={k: float(np.mean(v)) if v else None for k, v in values.items()},
                evaluable_source_counts={k: len(v) for k, v in values.items()})


def select_validation_working_point(candidates):
    """Each candidate contains split='validation', threshold, round, summary from aggregate_sources. At an unmet working
    point choose highest recall then lowest FP and report failure to reach it."""
    valid = []
    for item in candidates:
        if item['split'] != 'validation' or item['summary']['label_semantics'] != GEO_SEMANTICS:
            raise ValueError('Working-point selection is only for GeoCrack development validation')
        mean = item['summary']['source_equal_mean']
        recall, fp = mean['edge_recall_vs_public_labels'], mean['fp_edge_density_vs_public_labels']
        if recall is not None and fp is not None and np.isfinite([recall, fp]).all():
            valid.append((item, recall, fp))
    if not valid:
        return dict(candidate=None, working_point_reached=False, status='not_evaluable')
    reached = [row for row in valid if row[1] >= .80]
    if reached:
        selected = min(reached, key=lambda row: (row[2], -row[1], row[0]['round']))
    else:
        selected = min(valid, key=lambda row: (-row[1], row[2], row[0]['round']))
    return dict(candidate=selected[0], working_point_reached=bool(reached),
                status='working_point_reached' if reached else 'descriptive_unmet_working_point')


TRACE_SEMANTICS = 'trace_vs_trace'
TRACE_INTERPRETATION = ('trace-to-trace agreement: predicted 1 px trace centrelines against 1 px trace centreline '
                        'references (synthetic fracture-surface/host intersection, or native FraXet traces); '
                        'length-weighted recall, unmatched prediction density, length precision and F1 within tolerance')


def _trace_from_core(core):
    """Rename the shared numerical core (computed by _evaluate) to trace keys and add length precision and F1.

    Precision = 1 - unmatched prediction length / prediction skeleton length (unmatched = farther than t from the
    reference skeleton); F1 = 2PR/(P+R). Recall/density formulas are the core's, unchanged.
    """
    tolerances = {}
    for key, row in core['tolerances'].items():
        recall = row['edge_to_trace_recall_reference']
        pl = row['prediction_graph_length_px']
        precision = 1. - row['unmatched_prediction_graph_length_px'] / pl if pl else None
        if recall is None or precision is None:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        out = {k: v for k, v in row.items()
               if k not in ('edge_to_trace_recall_reference', 'unmatched_edge_density_vs_trace_reference')}
        out.update(trace_recall=recall, unmatched_trace_density=row['unmatched_edge_density_vs_trace_reference'],
                   trace_precision=precision, trace_f1=f1)
        tolerances[key] = out
    result = dict(core)
    result.update(label_semantics=TRACE_SEMANTICS, tolerances=tolerances, main=tolerances['2'],
                  interpretation=TRACE_INTERPRETATION,
                  main_tolerance_px=2, sensitivity_tolerances_px=[1, 3],
                  precision_definition='1 - unmatched prediction graph length / prediction graph length (same tolerance)')
    return result


def evaluate_trace(probability, target, valid, *, native_shape, threshold):
    """Trace-vs-trace record on the native grid: 1 px centrelines, skeletonised after the valid clip, t = 2 px main."""
    core = _evaluate(probability, target, valid, native_shape=native_shape, threshold=threshold,
                     semantics=AUX_SEMANTICS)
    return _trace_from_core(core)


def evaluate_trace_thresholds(probability, target, valid, *, native_shape, thresholds):
    values = tuple(thresholds)
    if not values or len(values) != len(set(values)):
        raise ValueError('A nonempty unique predeclared threshold sequence is required')
    return [evaluate_trace(probability, target, valid, native_shape=native_shape, threshold=t) for t in values]


def aggregate_trace_sources(rows):
    """aggregate_sources for trace rows: image then source means at the main tolerance; one semantic and threshold."""
    rows = list(rows)
    if not rows:
        return dict(source_count=0, image_count=0, sources={}, source_equal_mean={})
    if {r['metrics']['label_semantics'] for r in rows} != {TRACE_SEMANTICS}:
        raise ValueError('Only trace_vs_trace rows can be pooled here')
    if len({r['metrics']['threshold'] for r in rows}) != 1:
        raise ValueError('All images must use the same fixed working-point threshold')
    identities = [(r['source_id'], r['image_id']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('One result per source/image is required; do not count patches or seeds as images')
    keys = ('trace_recall', 'unmatched_trace_density', 'trace_precision', 'trace_f1')
    grouped = defaultdict(list)
    for row in rows:
        r = row['metrics']
        grouped[row['source_id']].append({**{k: r['main'][k] for k in keys},
                                          **{'pixel_' + k: r['pixel'][k] for k in ('precision', 'recall', 'f1', 'false_positive_area_fraction')},
                                          'ignored_fraction': r['ignored_fraction']})
    names = tuple(next(iter(grouped.values()))[0])
    sources = {}
    for source, images in grouped.items():
        values = {key: [r[key] for r in images if r[key] is not None] for key in names}
        sources[source] = dict(image_count=len(images),
                               mean={k: float(np.mean(v)) if v else None for k, v in values.items()},
                               evaluable_image_counts={k: len(v) for k, v in values.items()})
    values = {k: [s['mean'][k] for s in sources.values() if s['mean'][k] is not None] for k in names}
    return dict(label_semantics=TRACE_SEMANTICS, source_count=len(sources), image_count=len(rows), sources=sources,
                source_equal_mean={k: float(np.mean(v)) if v else None for k, v in values.items()},
                evaluable_source_counts={k: len(v) for k, v in values.items()})
