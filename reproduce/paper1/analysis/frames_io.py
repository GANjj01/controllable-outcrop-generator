"""Where the analysis scripts find the files of a frame.

FRAMES is a folder with one sub-folder per frame, named by the frame identifier of reproduce/paper1/frames.csv (a0001 ...) or of
reproduce/paper1/sweeps.csv, as written by

    python -m fracgen frame SCENE_DIR --out FRAMES --name FRAME_ID ...

(a frame written with --out FRAMES/FRAME_ID --name FRAME_ID, which gives FRAMES/FRAME_ID/FRAME_ID/, is found as well). Inside it

    frame.json, checks.json        summary of the frame (shot, light, pixel size) and its checks
    plan.json                      camera plan of the scene (the plan camera of the frame gives the angle theta)
    q0/render_spec.json            camera shot, ground plane, light and samples of the rendering
    q0/light.json                  shape parameters and light of the frame
    q0/geometry.json               'case': folder of the cut mesh of this frame inside the scene folder
    q0/render/frame.png, frame.npz rendered image; linear RGB, depth, material index, region mask
    q0/trace_labels/               label layers (*.npy), report.json, windows.json, window_pixels.npz
    q0/trace_labels_state.pkl      trace samples of the label program (3-D points, mid-surface normals, host faces)

The list of the paper's 115 labelled frames with their experiment block, scene and camera is
reproduce/paper1/records/frames_index.json. Standard library only; samples() needs NumPy."""
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path

# A conda environment whose interpreter is started by its path, without activation, does not have its library folder on PATH;
# on Windows NumPy and SciPy then fail when they first load BLAS / LAPACK. Same effect as activating the environment.
_lib = Path(sys.executable).resolve().parent / 'Library' / 'bin'
if _lib.is_dir():
    os.environ['PATH'] = os.pathsep.join([str(_lib.parents[1]), str(_lib), os.environ.get('PATH', '')])
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(str(_lib))

PAPER1 = Path(__file__).resolve().parents[1]
INDEX = PAPER1 / 'records' / 'frames_index.json'
RECORDS = PAPER1 / 'records' / 'round3'

# The 11 viewpoints with one frame each: the 5 viewpoints of the multi-illumination aerial set (its frame with render seed 0 and the
# first clear-sky illumination) and 6 further viewpoints (frame -> outcrop type, image set, scene and camera name in the index).
VIEWPOINTS = {
    'a0077': ('flat', 'multi-illumination aerial', 'v4_26120001', 'r2b_c0_q0'),
    'a0074': ('standalone', 'multi-illumination aerial', 'v4_26130002', 'r2b_c1_q0'),
    'a0062': ('standalone', 'multi-illumination aerial', 'v4_26130002', 'r2b_c0_q0'),
    'a0099': ('standalone', 'multi-illumination aerial', 'v4_26130002', 'r2b_c3_q0'),
    'a0044': ('standalone', 'multi-illumination aerial', 'v4_26130000', 'r2b_c1_q0'),
    'a0001': ('flat', 'further viewpoints', 'b_b055_a26120005_c2', 'sun0'),
    'a0002': ('flat', 'further viewpoints', 'b_b055_a26120005_c3', 'sun0'),
    'a0061': ('standalone', 'further viewpoints', 'b_b055_a26130005_c1', 'sun0'),
    'd0001': ('standalone', 'further viewpoints', 'd_a26130007_c0', 'sun0'),
    'a0018': ('flat', 'further viewpoints', 'b_b055_n26120003_c1', 'sun0'),
    'a0020': ('standalone', 'further viewpoints', 'b_b055_n26130007_c0', 'sun0'),
}
AERIAL_VIEWS = ['26120001_c0', '26130002_c1', '26130002_c0', '26130002_c3', '26130000_c1']      # scene seed and plan camera
AERIAL_OUTCROP = {'26120001_c0': 'flat', '26130002_c1': 'standalone', '26130002_c0': 'standalone', '26130002_c3': 'standalone',
                  '26130000_c1': 'standalone'}


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def _dumps(value, depth, level=0):
    """JSON text indented by one space per level down to the given depth, with everything below written on one line"""
    if level >= depth or not isinstance(value, (dict, list)) or not value:
        return json.dumps(value, ensure_ascii=True, allow_nan=False)
    pad, end = '\n' + ' ' * (level + 1), '\n' + ' ' * level
    if isinstance(value, dict):
        items = ['%s: %s' % (json.dumps(str(k), ensure_ascii=True), _dumps(v, depth, level + 1)) for k, v in value.items()]
        return '{' + pad + (',' + pad).join(items) + end + '}'
    return '[' + pad + (',' + pad).join(_dumps(v, depth, level + 1) for v in value) + end + ']'


def write_json(p, value, depth=None):
    """depth: number of levels written one item per line (None: all levels); deeper levels are written on one line, which keeps a
    record with many rows small"""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=True, indent=1, allow_nan=False) if depth is None else _dumps(value, depth)
    p.write_text(text + '\n', encoding='utf-8', newline='\n')


def index(path=None):
    """frame identifier -> dict(block, sid, pid, pixel_mm, ...) of the paper's formal label set"""
    return read_json(path or INDEX)['frames']


def frame_dir(frames, key):
    """folder of one frame (the folder that holds frame.json and q0/)"""
    root = Path(frames)
    for d in (root / key, root / key / key):
        if (d / 'q0').is_dir():
            return d
    raise FileNotFoundError('frame %s not found under %s (expected %s/q0)' % (key, root, root / key))


def point_dir(frames, key):
    return frame_dir(frames, key) / 'q0'


def label_dir(frames, key):
    return frame_dir(frames, key) / 'q0' / 'trace_labels'


def case_dir(frames, key):
    """folder of the cut mesh of the frame (inside the scene folder)"""
    return Path(read_json(point_dir(frames, key) / 'geometry.json')['case'])


def render_spec(frames, key):
    """camera shot, resolution, ground plane and light of the rendering"""
    return read_json(point_dir(frames, key) / 'render_spec.json')['frames'][0]


def view_of(row):
    """viewpoint name of a frame of the multi-illumination aerial set: scene seed and plan camera, e.g. 26130000_c1"""
    m = re.match(r'r2b_(c\d+)_(q\d+|oc)(_r1)?$', row['pid'])
    if not m:
        raise ValueError('unexpected point name: %s' % row['pid'])
    return row['sid'].split('_')[1] + '_' + m.group(1)


def aerial_frames(idx):
    """the 50 frames of the multi-illumination aerial set (block C of the index), sorted by identifier"""
    keys = sorted(k for k, v in idx.items() if v.get('block') == 'C')
    if len(keys) != 50:
        raise SystemExit('the multi-illumination aerial set has 50 frames, the index lists %d' % len(keys))
    return keys


def check_viewpoints(idx):
    """the 11 one-frame viewpoints are the frames the index says they are"""
    for key, (_, _, sid, pid) in VIEWPOINTS.items():
        e = idx[key]
        if e['sid'] != sid or e['pid'] != pid:
            raise ValueError('%s: the index gives %s/%s, expected %s/%s' % (key, e['sid'], e['pid'], sid, pid))


def view_angle(frames, key):
    """Angle theta (degrees) between the line of sight and the rock-surface normal at the camera target, and the view class
    (frontal / oblique), of the plan camera whose shot equals the shot of the frame. The normal is the area-weighted rock-surface
    normal within 0.5 m of the target that the camera plan stores."""
    shot = render_spec(frames, key)['shot']
    plan = read_json(frame_dir(frames, key) / 'plan.json')
    cams = [c for c in plan['cameras'] if c and c['shot'] == shot]
    if len(cams) != 1:
        raise ValueError('%s: %d plan cameras have the shot of the frame (a plan camera is needed, not a user-placed one)' % (key, len(cams)))
    cam = cams[0]
    d = math.dist(shot['position_enu_m'], shot['target_enu_m'])
    v = [(p - t) / d for p, t in zip(shot['position_enu_m'], shot['target_enu_m'])]
    n = cam['normal']
    nn = math.sqrt(sum(x * x for x in n))
    geo = math.degrees(math.acos(max(-1., min(1., abs(sum(a * b for a, b in zip(v, n))) / nn))))
    return dict(theta_deg=float(cam['angle_deg']), view_class=cam['angle'], theta_from_view_and_normal_deg=geo)


def joint_sets(key):
    """{number of the source trace: dict(id, mean_direction_deg) of its joint set} for the scene of a paper frame. The scene
    configuration (frames.csv, column config) lists the joint sets with their numbers of joints; the traces are numbered set by
    set in that order."""
    import csv
    with open(PAPER1 / 'frames.csv', encoding='utf-8-sig', newline='') as f:
        row = next((r for r in csv.DictReader(f) if r['frame'] == key), None)
    if row is None:
        raise KeyError('frame %s is not listed in frames.csv' % key)
    groups = read_json(PAPER1.parents[1] / 'configs' / 'scenes' / row['config'])['geometry']['fracture_network']['direction_groups']
    out, n = {}, 0
    for g in groups:
        for _ in range(int(g['count'])):
            out[n] = dict(id=g['id'], mean_direction_deg=float(g['mean_direction_deg']))
            n += 1
    return out


def samples(frames, key):
    """Trace samples of the label program in its row order: dict(point (n, 3): position after the position refinement, or before
    it where the refinement did not converge; face (n,): triangle of the uncut rock surface that holds the sample; trace (n,):
    number of the source trace; path (n,) or None: path number, frames with faults only; source: where the rows were read).

    Read from q0/trace_labels_state.pkl. A frame labelled with the --keep-scratch option of the label program also keeps
    trace_labels/refine_in.npz, which holds the same rows before the refinement; it is used when present."""
    import numpy as np
    lab = label_dir(frames, key)
    with np.load(lab / 'refine_in_refined.npz', allow_pickle=False) as a:
        P_ref, face_ref, normal = a['point'], a['face'].astype(np.int64), a['normal']
    ok = np.isfinite(normal).all(1)
    if (lab / 'refine_in.npz').is_file():
        with np.load(lab / 'refine_in.npz', allow_pickle=False) as a:
            P_pre, face_pre, trace = a['point'], a['face'].astype(np.int64), a['trace'].astype(np.int64)
            path = a['path'].astype(np.int64) if 'path' in a.files else None
        if not (len(P_ref) == len(P_pre) == len(trace)):
            raise ValueError('%s: refine_in.npz and refine_in_refined.npz differ in length' % key)
        return dict(point=np.where(ok[:, None], P_ref, P_pre).astype(float), face=np.where(ok, face_ref, face_pre), trace=trace, path=path,
                    rows_using_pre_refinement_point=int((~ok).sum()), source='refine_in.npz')
    st = pickle.loads((point_dir(frames, key) / 'trace_labels_state.pkl').read_bytes())
    S = st['samples']
    order = sorted(S)
    trace_of = {int(r['path']): r['source_trace_id'] for r in read_json(lab / 'report.json')['per_path']}
    for i, p in (st.get('paths_added') or {}).items():           # extension paths of frames with faults
        trace_of[i] = p['row']['source_trace_id']
    missing = sorted({str(S[k]['path']) for k in order if S[k]['path'] not in trace_of})
    if missing:
        raise ValueError('%s: no source trace recorded for path(s) %s' % (key, ', '.join(missing)))
    cat = lambda name: np.concatenate([S[k][name] for k in order])  # noqa: E731
    P, face = cat('point').astype(float), cat('host_face').astype(np.int64)
    trace = np.concatenate([np.full(len(S[k]['u']), int(trace_of[S[k]['path']].split('-')[1]), np.int64) for k in order])
    if len(P) != len(P_ref):
        raise ValueError('%s: trace_labels_state.pkl and refine_in_refined.npz differ in length' % key)
    return dict(point=P, face=face, trace=trace, path=None, rows_using_pre_refinement_point=int((~ok).sum()),
                source='trace_labels_state.pkl')
