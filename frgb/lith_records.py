"""Records that carry lithology or template labels. The table is data (run-config name labels.lithology_aliases_r1:
neutral_to_source, stainable_host); the code names no lithology. Not caught here, by design: a legacy name inside an
identifier (a report key such as <name>_px, an asset id such as <name>_public_a)."""
import json
import re
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parents[1]
if str(P1) not in sys.path:
    sys.path.append(str(P1))
from frgb import runconfig as frgb_rc  # noqa: E402

NAME_FIELDS = ('lithology', 'lithologies', 'rock_family', 'family', 'host_family', 'origin_lithology', 'beds', 'young_lithologies')
FAMILY_EXTRA = ('mixed', 'other')     # rock_family values outcrop.modeling.validation allows besides the families (not names)


class LithologyNameError(ValueError):
    pass


def _load(table=None):
    t = table if table is not None else json.loads(frgb_rc.path('labels.lithology_aliases_r1').read_text(encoding='utf-8'))
    labels = dict(t['neutral_to_source'])
    legacy = {v: k for k, v in labels.items()}
    if len(legacy) != len(labels):
        raise LithologyNameError('alias table: two labels share one legacy name')
    both = set(legacy) & set(labels)
    if both:
        raise LithologyNameError('alias table: %s is both a legacy name and a label' % sorted(both))
    rx = re.compile(r'(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])' % '|'.join(map(re.escape, sorted(legacy, key=len, reverse=True))))
    return dict(labels=set(labels), legacy=legacy, rx=rx, host=t['stainable_host'])


_T = None


def table():
    global _T
    if _T is None:
        _T = _load()
    return _T


def translate(x, t=None):
    t = t or table()
    if isinstance(x, str):
        return t['rx'].sub(lambda m: t['legacy'][m.group(0)], x)
    if isinstance(x, list):
        return [translate(v, t) for v in x]
    if isinstance(x, dict):
        return {translate(k, t) if isinstance(k, str) else k: translate(v, t) for k, v in x.items()}
    return x


def check(x, where='', t=None, field=None):
    t = t or table()
    if isinstance(x, str):
        if t['rx'].search(x):
            raise LithologyNameError('%s: legacy name left in %r' % (where, x[:120]))
        if field in NAME_FIELDS and x not in t['labels'] and not (field in ('rock_family', 'family', 'host_family') and x in FAMILY_EXTRA):
            raise LithologyNameError('%s: %r under %r is not a known label' % (where, x, field))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            check(v, '%s[%d]' % (where, i), t, field)
    elif isinstance(x, dict):
        for k, v in x.items():
            if isinstance(k, str) and t['rx'].search(k):
                raise LithologyNameError('%s: legacy name left in key %r' % (where, k))
            if field == 'lithologies' and k == 'id':          # lithology records: [{'id': ..., 'code': ..., 'subclass': ...}]
                check(v, '%s/%s' % (where, k), t, 'lithology')
            else:
                check(v, '%s/%s' % (where, k), t, k if k in NAME_FIELDS else None)
    return x


def read_json(path, t=None):
    x = translate(json.loads(Path(path).read_text(encoding='utf-8')), t)
    return check(x, str(path), t)


def read_codebook(path, t=None):
    x = read_json(path, t)
    return x if isinstance(x, list) else x.get('codebook', x)


def host_codes(book, t=None):
    host = (t or table())['host']
    return sorted(int(e['code']) for e in book if e.get('lithology') == host)


def host_names(t=None):
    t = t or table()
    return [t['host']]+sorted(k for k, v in t['legacy'].items() if v == t['host'])


def is_host(name, t=None):
    return name in host_names(t)
