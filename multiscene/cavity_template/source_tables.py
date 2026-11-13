"""Per-face cutter source tables path_XX_sources.

A panel table (every face record exactly {source_segment_id: int, role: str, construction: str}) is stored as
compressed npz columns in insertion order; decoding gives the dict json.loads would give for the json of the same
document, so prepare.save of the decoded document reproduces the json bytes. Any other table stays json. Readers call
load_sources: npz if only the npz exists, json if only the json exists, an error if both or neither. A table is
written to PATH.tmp and moved onto PATH with os.replace: repair_host_needles hard-links the tables of <case>_raw into
the repaired case, and a later rewrite of either name then replaces that directory entry only, never the shared bytes."""
import json
import os
from pathlib import Path
import numpy as np

FORMAT = 'face_sources_columns_r1'
PANEL_KEYS = ('source_segment_id', 'role', 'construction')


def paths(folder, index):
    stem = Path(folder)/('path_%02d_sources' % index)
    return stem.with_suffix('.json'), stem.with_suffix('.npz')


def _columns(face_sources):
    """(face_id, source_segment_id, role, construction) lists, or None when a record is not of the panel form"""
    fid, seg, role, cons = [], [], [], []
    for key, row in face_sources.items():
        if type(key) is str:
            if not (key.isascii() and key.isdigit()) or str(int(key)) != key: return None
            key = int(key)
        elif type(key) is not int: return None
        if (type(row) is not dict or tuple(row) != PANEL_KEYS or type(row['source_segment_id']) is not int
                or type(row['role']) is not str or type(row['construction']) is not str): return None
        fid.append(key); seg.append(row['source_segment_id']); role.append(row['role']); cons.append(row['construction'])
    try:
        fid, seg = np.array(fid, dtype=np.int64), np.array(seg, dtype=np.int64)
    except OverflowError:
        return None
    role, cons = _codes(role), _codes(cons)
    return None if role is None or cons is None else (fid, seg, role, cons)


def _codes(values):
    """(names, codes) of a string column, or None if a name does not survive numpy's unicode array unchanged"""
    names = list(dict.fromkeys(values)); arr = np.array(names, dtype=str)
    if arr.tolist() != names: return None
    index = {v: k for k, v in enumerate(names)}
    return arr, np.array([index[v] for v in values], dtype=np.uint8 if len(names) <= 256 else np.int32)


def _replace(path, write):
    """write(tmp) then os.replace(tmp, path): a hard link to the old PATH keeps its bytes; no partial PATH is ever visible"""
    tmp = path.with_name(path.name + '.tmp')
    try:
        write(tmp); os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True); raise
    return path


def save_sources(folder, index, document, fmt, save_json):
    """write path_XX_sources.npz (fmt 'npz', panel table) or .json via save_json (prepare.save); returns the path written"""
    js, nz = paths(folder, index)
    cols = _columns(document['face_sources']) if fmt == 'npz' else None
    if cols is None:
        if fmt == 'npz': print(json.dumps(dict(event='source_table_json_fallback', index=index)), flush=True)
        return _replace(js, lambda tmp: save_json(tmp, document))
    fid, seg, (role_names, role_code), (cons_names, cons_code) = cols
    meta = {k: (None if k == 'face_sources' else v) for k, v in document.items()}

    def write(tmp):
        with open(tmp, 'wb') as f:   # a file object: savez would append .npz to a path not ending in it
            np.savez_compressed(f, format=np.array(FORMAT), meta_json=np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False)),
                                face_id=fid, source_segment_id=seg, role_names=role_names, role_code=role_code,
                                construction_names=cons_names, construction_code=cons_code)
    return _replace(nz, write)


def decode(npz_path):
    with np.load(npz_path, allow_pickle=False) as a:
        if str(a['format']) != FORMAT: raise ValueError('%s: unknown source table format %s' % (npz_path, a['format']))
        document = json.loads(str(a['meta_json']))
        rn, cn = a['role_names'].tolist(), a['construction_names'].tolist()
        rows = zip(a['face_id'].tolist(), a['source_segment_id'].tolist(), a['role_code'].tolist(), a['construction_code'].tolist())
        document['face_sources'] = {str(f): dict(source_segment_id=s, role=rn[r], construction=cn[c]) for f, s, r, c in rows}
    return document


def load_sources(folder, index):
    """the source table document of path INDEX in FOLDER, from whichever of its json / npz exists"""
    js, nz = paths(folder, index)
    if js.is_file() and nz.is_file(): raise ValueError('both %s and %s exist' % (js.name, nz.name))
    if nz.is_file(): return decode(nz)
    return json.loads(js.read_text(encoding='utf8'))
