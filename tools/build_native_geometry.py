"""Build the optional exact-geometry extension _outcrop_exact for the running 64-bit Windows Python.

The geological model uses it, when present, for exact filtered predicates; without it the same predicates run in Python.
Needs a Zig compiler (https://ziglang.org, used as a portable C++ compiler) and the Boost headers (boost/multiprecision).
Compiler caches and intermediate files go to the temporary directory.

    python tools/build_native_geometry.py --compiler PATH/TO/zig.exe --boost-include PATH/TO/boost_1_xx_0
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


MODULE = '_outcrop_exact'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path,
                        help='Path to the portable Zig executable')
    parser.add_argument('--boost-include', required=True, type=Path,
                        help='Existing directory containing boost/multiprecision/cpp_int.hpp')
    parser.add_argument('--output', type=Path,
                        help='Output .pyd path or directory; defaults to this Python platlib')
    args = parser.parse_args()
    if sys.platform != 'win32' or sys.maxsize <= 2**32:
        parser.error('this build helper requires a 64-bit Windows Python')
    compiler = args.compiler.resolve(strict=True)
    boost = args.boost_include.resolve(strict=True)
    if not (boost / 'boost/multiprecision/cpp_int.hpp').is_file():
        parser.error('--boost-include does not contain the required existing Boost headers')

    repo = Path(__file__).resolve().parents[1]
    scratch = Path(tempfile.gettempdir()) / 'native-geometry-build'
    scratch.mkdir(parents=True, exist_ok=True)
    include = Path(sysconfig.get_path('include'))
    import_lib = Path(sys.base_prefix) / 'libs' / f'python{sys.version_info.major}{sys.version_info.minor}.lib'
    if not (include / 'Python.h').is_file() or not import_lib.is_file():
        parser.error('the selected Python must provide Python.h and its matching import library')
    filename = MODULE + sysconfig.get_config_var('EXT_SUFFIX')
    output = args.output or Path(sysconfig.get_path('platlib')) / filename
    if output.is_dir() or not output.suffix:
        output = output / filename
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR'] = str(scratch / 'zig-global-cache')
    env['ZIG_LOCAL_CACHE_DIR'] = str(scratch / 'zig-local-cache')
    env['TEMP'] = env['TMP'] = str(scratch)
    command = [str(compiler), 'c++', '-target', 'x86_64-windows-gnu', '-shared',
               '-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off',
               '-DNDEBUG', '-DBOOST_ALL_NO_LIB', f'-I{include}', f'-I{boost}',
               str(repo / 'outcrop/modeling/kernel/_exact_native.cpp'),
               str(import_lib), '-o', str(output)]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, env=env, cwd=scratch, check=True)
    print(f'Built {output}', flush=True)


if __name__ == '__main__':
    main()
