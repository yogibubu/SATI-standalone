#!/usr/bin/env python3
"""Export pinned SATI sources, then build an offline distribution on a compute host.

Export reads committed objects only. Build never reads the parent checkout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

from build_smith_release import _requirement_name, _stage_project

ROOTS = ('matrix-trinity', 'matrix-dvr', 'matrix-vpt2-vci', 'matrix-zaff',
         'matrix-gaussian', 'matrix-orca', 'matrix-apoc', 'matrix-architect')
TESTS = ('test_vibronic_native.py', 'test_common_quadratic_modes.py',
         'test_common_quadratic_correlation.py', 'test_vibronic_variational.py',
         'test_vibronic_revision_general_tools.py')


def sha256(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def export(output: Path, revision: str):
    root = Path(__file__).resolve().parents[2]
    if output.exists():
        raise FileExistsError(f'Output must be new: {output}')
    revision = subprocess.check_output(['git', 'rev-parse', revision], cwd=root, text=True).strip()
    queue = list(ROOTS)
    projects = {}
    while queue:
        name = queue.pop()
        if name in projects:
            continue
        data = subprocess.check_output(['git', 'show', f'{revision}:packages/{name}/pyproject.toml'], cwd=root)
        metadata = tomllib.loads(data.decode())['project']
        projects[name] = metadata['version']
        queue.extend(_requirement_name(item) for item in metadata.get('dependencies', ())
                     if _requirement_name(item).startswith('matrix-'))
    output.mkdir(parents=True)
    source = output / 'source'
    source.mkdir()
    paths = [f'packages/{name}' for name in sorted(projects)]
    paths += ['LICENSE', 'VERSION', 'pyproject.toml', 'tools/build/build_smith_release.py',
              'examples/trinity/esalc_vibronic', 'examples/trinity/targeted_ti',
              'benchmarks/vibronic_targeted_ti/qualify_extensions.py',
              *[f'tests/{name}' for name in TESTS]]
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['git', 'archive', revision, *paths], cwd=root, stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            # git archive contains only tracked, relative project paths.
            archive.extractall(source, filter='data')
    # Exporter and facade are independently inventoried so they can be developed
    # against an immutable scientific revision without exporting a dirty tree.
    shutil.copytree(root / 'tools/standalone/sati', source / 'sati')
    shutil.copyfile(Path(__file__), source / 'tools/build/build_sati_standalone.py')
    shutil.copyfile(source / 'sati/README.md', output / 'README.md')
    shutil.copyfile(source / 'sati/install.py', output / 'install.py')
    shutil.copyfile(source / 'LICENSE', output / 'LICENSE')
    for path in source.rglob('__pycache__'):
        shutil.rmtree(path)
    manifest = dict(schema='sati.standalone.source.v1', scientific_revision=revision,
                    components=projects, python_requires='>=3.11',
                    files={str(p.relative_to(output)): sha256(p)
                           for p in sorted(output.rglob('*')) if p.is_file()})
    (output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(output / 'source-manifest.json')


def build(output: Path):
    manifest = json.loads((output / 'source-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256(output / name) != digest:
            raise ValueError(f'Source hash mismatch: {name}')
    wheels = output / 'wheels'
    if wheels.exists():
        raise FileExistsError(f'Wheel output must be new: {wheels}')
    wheels.mkdir()
    source = output / 'source'
    with tempfile.TemporaryDirectory(prefix='sati-build-') as directory:
        staged = Path(directory)
        projects = [_stage_project(source / 'packages' / name, staged)
                    for name in manifest['components']]
        projects.append(_stage_project(source / 'sati', staged))
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--wheel-dir', str(wheels),
                        *map(str, projects)], check=True)
    # Include the check runner for the supplied scientific regressions offline.
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                    '--dest', str(wheels), 'pytest>=8,<9'], check=True)
    record = dict(schema='sati.standalone.wheelhouse.v1',
                  scientific_revision=manifest['scientific_revision'],
                  python=sys.version, platform=__import__('platform').platform(),
                  files={p.name: sha256(p) for p in sorted(wheels.glob('*.whl'))})
    (output / 'wheelhouse-manifest.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(output / 'wheelhouse-manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    exp = commands.add_parser('export')
    exp.add_argument('output', type=Path)
    exp.add_argument('--revision', default='HEAD')
    bld = commands.add_parser('build')
    bld.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.output.expanduser().resolve(), args.revision)
    else:
        build(args.output.expanduser().resolve())


if __name__ == '__main__':
    main()
