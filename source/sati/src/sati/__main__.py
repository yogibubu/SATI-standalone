"""Explicit SATI entry points, reusing the qualified scientific owners."""
import argparse
import json
from pathlib import Path
from . import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    esalc = commands.add_parser('run-esalc', help='Run the established ESALC TI/TD workflow')
    esalc.add_argument('request', type=Path)
    esalc.add_argument('output', type=Path)
    native = commands.add_parser('build-native', help='Explicitly compile the C++17 spectrum engine')
    native.add_argument('output', type=Path)
    native.add_argument('--compiler', default='c++')
    native.add_argument('--lapack-flag', action='append', help='Repeat for link flags; default: -lopenblas')
    run = commands.add_parser('run-native', help='Execute a prepared FC/HT or general-operator input')
    run.add_argument('input', type=Path)
    run.add_argument('output', type=Path)
    run.add_argument('--executable', type=Path, required=True)
    run.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'run-esalc':
        from matrix_trinity import run_esalc_vibronic_workflow
        result = run_esalc_vibronic_workflow(json.loads(args.request.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    elif args.command == 'build-native':
        from matrix_trinity.vibronic_native import build_native_harmonic_runner
        result = build_native_harmonic_runner(args.output, compiler=args.compiler,
                                             lapack_flags=args.lapack_flag or ('-lopenblas',))
    else:
        from matrix_trinity.vibronic_native import execute_prepared_native_harmonic
        result = execute_prepared_native_harmonic(args.input, args.output,
                                                 executable=args.executable, blas_threads=args.threads)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
