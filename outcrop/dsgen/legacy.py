"""The generator's former catalogue names, ONLY for reading configurations and records written before the neutral labels.
Data: data/legacy_names.json. New records never use these names. name(n)              a former name -> its label
(anything else unchanged) former(n)            a label (name, subclass or setting) -> its former name (records in the
former vocabulary) translate(x)         every whole former token of every string and key of a JSON value -> its label
(records, not free text) translate_config(c)  a generation configuration: geometry.family, geometry.igneous_setting
(key and value), geometry.lithologies and geometry.stratigraphy.unconformity.young_lithologies only; free text such as
appearance.notes stays as written translate_stratigraphy(s)  the stratigraphy option alone (sample_spec's entry)"""
import json
import re
from pathlib import Path

TABLE = json.loads((Path(__file__).with_name('data') / 'legacy_names.json').read_text(encoding='utf-8'))
TOKENS = {}
for _section in ('names', 'subclasses', 'settings', 'keys', 'display'):
    TOKENS.update(TABLE[_section])
_RX = re.compile(r'(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])' % '|'.join(map(re.escape, sorted(TOKENS, key=len, reverse=True))))
_FORMER = {v: k for _section in ('names', 'subclasses', 'settings') for k, v in TABLE[_section].items()}


def name(n):
    return TABLE['names'].get(n, n) if isinstance(n, str) else n


def former(n):
    return _FORMER.get(n, n)


def translate(x):
    if isinstance(x, str):
        return _RX.sub(lambda m: TOKENS[m.group(0)], x)
    if isinstance(x, list):
        return [translate(v) for v in x]
    if isinstance(x, tuple):
        return tuple(translate(v) for v in x)
    if isinstance(x, dict):
        return {translate(k) if isinstance(k, str) else k: translate(v) for k, v in x.items()}
    return x


def translate_config(config):
    if not isinstance(config, dict) or not isinstance(config.get('geometry'), dict):
        return config
    geo = config['geometry']
    if 'igneous_setting' in geo and 'group_b_setting' in geo:
        raise ValueError('geometry gives both igneous_setting and group_b_setting')
    out = {}
    for k, v in geo.items():
        if k == 'igneous_setting':
            k = 'group_b_setting'
        if k == 'group_b_setting' and isinstance(v, str):
            v = TABLE['settings'].get(v, v)
        elif k == 'family':
            v = name(v)
        elif k == 'lithologies' and isinstance(v, list):
            v = [name(x) for x in v]
        elif k == 'stratigraphy':
            v = translate_stratigraphy(v)
        out[k] = v
    return dict(config, geometry=out)


def translate_stratigraphy(s):
    # the stratigraphy option's unconformity.young_lithologies -> labels; everything else unchanged
    if not isinstance(s, dict):
        return s
    u = s.get('unconformity')
    if isinstance(u, dict) and isinstance(u.get('young_lithologies'), list):
        s = dict(s, unconformity=dict(u, young_lithologies=[name(x) for x in u['young_lithologies']]))
    return s
