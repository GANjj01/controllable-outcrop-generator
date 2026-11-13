"""Repository checks that need no rendering: configuration, text hygiene, paper tables and main-text values, label checks, preview.

Run in the geometry environment:  python -m pytest -q tests"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEXT = ('.py', '.md', '.json', '.txt', '.yml', '.cff', '.cpp', '.csv', '.toml', '.cfg', '.sh')
BS = chr(92)
CJK = re.compile('[' + chr(0x3000) + '-' + chr(0x303f) + chr(0x3400) + '-' + chr(0x9fff) + chr(0xff00) + '-' + chr(0xffef) + ']')
MACHINE = re.compile('[A-Za-z]:[/' + BS + BS + ']+(Users|geomodel|Program Files|ProgramData)', re.I)


LOCAL_FILE = 'inputs/runconfig_local.json'                # the machine's own files, see .gitignore
LOCAL_DIRS = ('inputs/large/', 'example_output/')


def text_files():
    for p in ROOT.rglob('*'):
        rel = p.relative_to(ROOT).as_posix()
        if (p.is_file() and p.suffix in TEXT and '.git' not in p.parts and '__pycache__' not in p.parts
                and rel != LOCAL_FILE and not rel.startswith(LOCAL_DIRS)):
            yield p


def test_default_configuration_loads():
    from frgb import runconfig as rc
    cfg = rc.load(ROOT / 'inputs/runconfig_default.json')
    for name in ('relief.spec_C_r2', 'relief.wall_C_r2', 'render.sky_calibration_r1', 'exposure.gate_r14'):
        assert rc.path(name, cfg).is_file(), name


def test_no_machine_paths():
    allowed = ('README.md', 'runconfig_local.example.json')
    bad = [str(p.relative_to(ROOT)) for p in text_files()
           if p.name not in allowed and MACHINE.search(p.read_text(encoding='utf-8', errors='replace'))]
    assert not bad, bad


def test_text_is_english():
    bad = [str(p.relative_to(ROOT)) for p in text_files() if CJK.search(p.read_text(encoding='utf-8', errors='replace'))]
    assert not bad, bad


def test_paper_tables_recompute():
    r = subprocess.run([sys.executable, str(ROOT / 'reproduce/paper1/tables.py')], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert 'all recomputed summaries equal the stored ones' in r.stdout


def test_main_text_values_recompute():
    r = subprocess.run([sys.executable, str(ROOT / 'reproduce/paper1/main_text.py')], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert 'all main-text values recomputed' in r.stdout


def test_main_text_reports_a_changed_value(tmp_path):
    values = json.loads((ROOT / 'reproduce/paper1/main_text_values.json').read_text(encoding='utf-8'))
    item = [i for i in values['items'] if i['id'] == 'view_flat'][0]
    item['printed'][2] = '46.8'
    changed = tmp_path / 'values.json'
    changed.write_text(json.dumps(values), encoding='utf-8')
    r = subprocess.run([sys.executable, str(ROOT / 'reproduce/paper1/main_text.py'), '--values', str(changed)], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout[-2000:] + r.stderr[-2000:]
    assert 'DIFF view_flat' in r.stdout and 'all main-text values recomputed' not in r.stdout


def test_label_checks_items():
    from fracgen.checks import label_items
    rep = dict(exit_code=0, stop={'g1_fail': False}, visible_length_px=600.,
               jump_check=dict(in_frame_length_px=800., run_breaks_before=3, run_breaks_after=3, jumped_px=0,
                               displaced_over_2px_fraction=0., capped_px=4))
    items = label_items(rep)
    assert all(v['ok'] for v in items.values())
    rep['jump_check']['capped_px'] = 40
    assert not label_items(rep)['5_capped']['ok']


def test_preview_runs(tmp_path):
    pytest.importorskip('PIL')
    pytest.importorskip('numpy')
    from fracgen import preview
    assert preview.main([str(ROOT / 'configs/scenes/flat_outcrop_with_fault.json'), '--out', str(tmp_path)]) == 0
    rec = json.loads((tmp_path / 'preview.json').read_text(encoding='utf-8'))
    assert [f['id'] for f in rec['faults']] == ['fault-user-1']
    assert len(rec['joints']) == 16
