"""Camera plan of a generated scene: 1024 px aerial / oblique cameras aimed at exposed fracture paths.

Cameras are drawn per cell (pixel-size level x view-angle class) from a stream fixed by the scene seed and accepted only when the
frame is filled by rock, the target path is visible over at least 256 px and the view-depth check passes. Standalone outcrops use
the fine cells only (4-11 mm/px); flat outcrops also use the coarse cells (11-32 mm/px). Writes OUT/plan.json; the frame step
takes a camera by its index in plan['cameras']."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fracgen.env import MS, load_json, python, run  # noqa: E402


def plan(scene_root, outcrop, per_cell=None):
    """per_cell: cameras drawn per cell (default as in the paper: 2 on standalone outcrops, 1 on flat outcrops)"""
    root = Path(scene_root).resolve()
    tool = MS / ('scene_sets/fs_camera_plan.py' if outcrop == 'flat' else 'scene_sets/si_camera_plan.py')
    cmd = [python('geometry'), tool, 'plan', 'shape', root / 'scene', root / 'net', root / 'plan.json']
    if per_cell is not None:
        cmd += ['--per-cell', per_cell]
    elif outcrop != 'flat':
        cmd += ['--per-cell', 2]
    if outcrop == 'flat':
        cmd += ['--outcrop', 'flat']
    run(cmd, 'geometry', root / 'logs/camera_plan.log')
    return load_json(root / 'plan.json')


def describe(p):
    rows = []
    for i, c in enumerate(p['cameras']):
        if c is None:
            continue
        rows.append('%2d  level %-6s angle %-8s pixel %.2f mm/px' % (i, c.get('level'), c.get('angle'), float(c.get('pixel_mm') or 0)))
    return '\n'.join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Draw the camera plan of a generated scene.')
    ap.add_argument('scene_root', help='output directory of "fracgen scene"')
    ap.add_argument('--outcrop', choices=['standalone', 'flat'])
    ap.add_argument('--per-cell', type=int, help='cameras per cell (default: 2 standalone, 1 flat)')
    a = ap.parse_args(argv)
    outcrop = a.outcrop or load_json(Path(a.scene_root) / 'scene_request.json')['outcrop']
    print(describe(plan(a.scene_root, outcrop, a.per_cell)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
