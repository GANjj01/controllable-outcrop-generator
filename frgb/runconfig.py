"""Run configuration of the fracture chain: logical name -> path, plus run parameters.

Code locations are not in it: modules find each other inside the package (frgb/chain) by relative paths.

A file may name a "base" file (relative to itself) whose sections it overrides key by key. Sections: roots, tools,
inputs (name -> path string or null), params (name -> JSON value). tools and inputs share one name space. A root can
also be set by the environment variable FRGB_ROOT_<NAME> (for example FRGB_ROOT_DATA). A path without placeholder
that is not absolute is relative to the file that sets it. No fallback: an unknown name, a null entry or an unresolved
placeholder raises ConfigError (a silent fallback to another location would hide a missing entry). Standard library
only (both environments, any Python).
CLI: python -m frgb.runconfig path NAME | param NAME | check | shell   (shell: export lines for bash run scripts)"""
import json
import os
import re
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parents[1]
DEFAULT = P1 / 'inputs' / 'runconfig_default.json'
LOCAL = P1 / 'inputs' / 'runconfig_local.json'
SECTIONS = ('roots', 'tools', 'inputs', 'params')
_PH = re.compile(r'\$\{([A-Za-z0-9_.]+)\}')
_cache = {}


class ConfigError(KeyError):
    pass


def config_file():
    env = os.environ.get('FRGB_RUNCONFIG')
    if env:
        return Path(env).resolve()
    return LOCAL if LOCAL.is_file() else DEFAULT


def _read(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ConfigError('base cycle at %s' % path)
    raw = json.loads(path.read_text(encoding='utf-8'))
    out = {s: {} for s in SECTIONS}
    if raw.get('base'):
        out = _read(path.parent / raw['base'], seen + (path,))
    for s in SECTIONS:
        for k, v in (raw.get(s) or {}).items():
            out[s][k] = (v, path.parent) if s != 'params' else v
    out['_files'] = out.get('_files', []) + [str(path)]
    return out


def load(path=None):
    """the merged configuration (cached per file); raises ConfigError on a name in both tools and inputs"""
    path = Path(path).resolve() if path else config_file()
    key = (str(path), tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith('FRGB_ROOT_'))))
    if key not in _cache:
        cfg = _read(path)
        both = set(cfg['tools']) & set(cfg['inputs'])
        if both:
            raise ConfigError('names in both tools and inputs: %s' % sorted(both))
        cfg['_path'] = str(path)
        _cache[key] = cfg
    return _cache[key]


def _expand(value, where, cfg, depth=0):
    if depth > 8:
        raise ConfigError('placeholder nesting too deep in %r' % value)

    def sub(m):
        name = m.group(1)
        if name == 'p1':
            return P1.as_posix()
        if name == 'inputs':
            return DEFAULT.parent.as_posix()
        return root(name, cfg, depth + 1)
    s = _PH.sub(sub, value)
    if not _PH.search(value) and not re.match(r'^([A-Za-z]:[\\/]|/)', s):
        s = (where / s).as_posix()
    return s


def root(name, cfg=None, depth=0):
    cfg = cfg or load()
    env = os.environ.get('FRGB_ROOT_' + name.upper())
    if env:
        return Path(env).as_posix()
    if name not in cfg['roots']:
        raise ConfigError('unknown root ${%s} (%s)' % (name, cfg['_path']))
    v, where = cfg['roots'][name]
    if v is None:
        raise ConfigError('root ${%s} is not configured: set it in runconfig_local.json, FRGB_RUNCONFIG or FRGB_ROOT_%s'
                          % (name, name.upper()))
    return _expand(v, where, cfg, depth)


def path(name, cfg=None):
    """Path of a tool or input; ConfigError when the name is unknown or not configured (null)"""
    cfg = cfg or load()
    for s in ('tools', 'inputs'):
        if name in cfg[s]:
            v, where = cfg[s][name]
            if v is None:
                raise ConfigError('%s %r is not configured in %s (set it in runconfig_local.json or FRGB_RUNCONFIG)'
                                  % (s[:-1], name, cfg['_path']))
            return Path(_expand(v, where, cfg))
    raise ConfigError('unknown name %r (%s)' % (name, cfg['_path']))


def optional_path(name, cfg=None):
    """Path, or None when the entry exists but is null; unknown names raise"""
    cfg = cfg or load()
    for s in ('tools', 'inputs'):
        if name in cfg[s] and cfg[s][name][0] is None:
            return None
    return path(name, cfg)


def param(name, cfg=None):
    cfg = cfg or load()
    if name not in cfg['params']:
        raise ConfigError('unknown param %r (%s)' % (name, cfg['_path']))
    return cfg['params'][name]


def pythonpath(cfg=None):
    cfg = cfg or load()
    return os.pathsep.join([P1.as_posix(), path('outcrop_src', cfg).as_posix(), path('manifold_site', cfg).as_posix()])


def env_dir_path(env_dir, base=None):
    """PATH with a conda env's directory and its Library/bin in front (Windows DLL lookup)"""
    b = Path(env_dir).as_posix()
    return os.pathsep.join([b, b + '/Library/bin', (base if base is not None else os.environ.get('PATH', ''))])


def check(cfg=None):
    """(name, path or None, exists) for every tool and input; null entries give (name, None, False)"""
    cfg = cfg or load()
    rows = []
    for s in ('tools', 'inputs'):
        for name in sorted(cfg[s]):
            p = optional_path(name, cfg)
            rows.append((name, p, bool(p and p.exists())))
    return rows


def shell(cfg=None):
    """export lines for bash run scripts (values single-quoted)"""
    cfg = cfg or load()
    q = lambda v: "'" + str(v).replace("'", "'\\''") + "'"  # noqa: E731
    env = dict(FRGB_P1=P1.as_posix(), FRGB_MS=(P1 / 'multiscene').as_posix(),
               FRGB_PY_PHASE0=path('python.geometry', cfg).as_posix(), FRGB_PY_CNN=path('python.labels', cfg).as_posix(),
               FRGB_PHASE0_DIR=path('env.geometry_dir', cfg).as_posix(), FRGB_PYTHONPATH=pythonpath(cfg),
               FRGB_BASH=path('bash', cfg).as_posix(), FRGB_MIN_FREE_GB=param('min_free_gb_generation', cfg),
               FRGB_NICE=param('nice', cfg),
               FRGB_RUNCONFIG_USED=cfg['_path'])
    return '\n'.join('export %s=%s' % (k, q(v)) for k, v in env.items())


def main(argv):
    if argv[:1] == ['path'] and len(argv) == 2:
        print(path(argv[1]).as_posix())
    elif argv[:1] == ['param'] and len(argv) == 2:
        print(json.dumps(param(argv[1])))
    elif argv == ['shell']:
        print(shell())
    elif argv == ['check']:
        bad = 0
        for name, p, ok in check():
            bad += p is not None and not ok
            print('%-34s %-7s %s' % (name, 'ok' if ok else ('null' if p is None else 'MISSING'), p or ''))
        print('config', load()['_path'], '; missing', bad)
        return 1 if bad else 0
    else:
        print(__doc__.strip().splitlines()[-1])
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
