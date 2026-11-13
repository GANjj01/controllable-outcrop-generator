"""Charge before executing; failed attempts stay charged."""
import json
from datetime import datetime, timezone
from pathlib import Path

KINDS = ('rgb', 'score', 'steps', 'proposals', 'eval_score', 'control_rgb', 'control_score', 'fixed_rgb', 'fixed_score')
PILOT_KINDS = ('rgb', 'score', 'steps', 'proposals')
PILOT_CAPS = dict(rgb_and_score=128, steps=1000, proposals=64)


def totals(path):
    p = Path(path)
    events = json.loads(p.read_text(encoding='utf8'))['events'] if p.exists() else []
    result = {kind: sum(e['count'] for e in events if e['kind'] == kind) for kind in KINDS}
    result['rgb_and_score'] = result['rgb']+result['score']
    return result


def create_ledger(path, caps, purpose):
    p = Path(path)
    if p.exists():
        raise FileExistsError(p)
    _check_caps(caps)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(caps=caps, purpose=purpose, created_utc=datetime.now(timezone.utc).isoformat(),
                                 events=[]), indent=2, ensure_ascii=False)+'\n', encoding='utf8')


def _check_caps(caps):
    if not caps or any(k not in KINDS+('rgb_and_score',) or type(v) is not int or v < 0 for k, v in caps.items()):
        raise ValueError('caps must map known kinds to non-negative integers')


def revise_caps(path, new_caps, reason):
    p = Path(path)
    data = json.loads(p.read_text(encoding='utf8'))
    if 'caps' not in data:
        raise ValueError('the pilot ledger caps are frozen')
    _check_caps(new_caps)
    if not reason:
        raise ValueError('a reason is required')
    old, current = data['caps'], totals(p)
    changed = {k for k in set(old) | set(new_caps) if old.get(k) != new_caps.get(k)}
    affected = {x for k in changed for x in (('rgb', 'score') if k == 'rgb_and_score' else (k,))}
    if any(current[k] for k in affected):
        raise ValueError('cannot revise caps of kinds already charged: %s' % sorted(k for k in affected if current[k]))
    data.setdefault('caps_revisions', []).append(dict(old=old, new=new_caps, reason=reason, totals_at_revision=current,
                                                      revised_utc=datetime.now(timezone.utc).isoformat()))
    data['caps'] = new_caps
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    return data['caps']


def charge(path, kind, count, purpose):
    if kind not in KINDS or type(count) is not int or count <= 0 or not purpose:
        raise ValueError('positive integer cost and a purpose are required')
    p = Path(path)
    data = json.loads(p.read_text(encoding='utf8')) if p.exists() else {'events': []}
    caps = data.get('caps')
    result = totals(p)
    result[kind] += count
    result['rgb_and_score'] = result['rgb']+result['score']
    if caps is None:
        if kind not in PILOT_KINDS:
            raise ValueError('the pilot ledger has no separate %s kind' % kind)
        if result['rgb']+result['score'] > PILOT_CAPS['rgb_and_score']:
            raise ValueError('pilot RGB + complete scores would exceed 128')
        if result['steps'] > PILOT_CAPS['steps'] or result['proposals'] > PILOT_CAPS['proposals']:
            raise ValueError('pilot step/proposal cap exceeded')
    else:
        over = [k for k, v in caps.items() if result[k] > v]
        if over or (kind not in caps and kind not in ('rgb', 'score')):
            raise ValueError('ledger cap exceeded or kind not budgeted: %s' % (over or kind))
    data['events'].append(dict(kind=kind, count=count, purpose=purpose,
                               charged_utc=datetime.now(timezone.utc).isoformat()))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    return totals(p)
