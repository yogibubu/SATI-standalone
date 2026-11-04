# SATI: Symmetry-Adapted Transition Intensities

Standalone harmonic and anharmonic vibronic spectra in automatically generated
nonredundant internal coordinates, combining time-independent (TI) assignments
and time-dependent (TD) profiles. All 23 internal scientific components and
C++17 sources are included. No private repository or parent-suite installation
is required. Internal `matrix_*` import names preserve the scientific modules.

## Install from GitHub source

Requirements: Python 3.11, a C++17 compiler and BLAS/LAPACK development libraries.
The reference platform is Linux x86_64 with OpenBLAS. On Debian/Ubuntu install
Python 3.11 and its venv/development packages, `g++` and `libopenblas-dev`.
Run from the repository root on a compute host:

```sh
git clone https://github.com/yogibubu/SATI-standalone.git
cd SATI-standalone
python3.11 -m venv env
env/bin/python -m pip install --upgrade pip
env/bin/python source/tools/build/build_sati_standalone.py build .
env/bin/python install.py
env/bin/sati --version
env/bin/sati build-native bin/sati-harmonic
```

The first build verifies exported-source hashes, builds the included scientific
packages and downloads standard Python dependencies into `wheels/`; it needs
Internet access. The subsequent installer checks wheel hashes and installs
without contacting a package index. Python and system libraries are prerequisites.

## Offline Linux release

The [release](https://github.com/yogibubu/SATI-standalone/releases/tag/v0.1.0rc8)
provides the validated Linux x86_64 / CPython 3.11 distribution with its complete
wheelhouse. Extract it and follow the offline instructions below. Rebuild native
components on the target host: archived binaries link to the reference host's
OpenBLAS installation. Other platforms need their own wheels and native build.

## Offline installation

Use Python 3.11 on the operating system and architecture recorded in
`wheelhouse-manifest.json`. Create an isolated environment and run:

```
python3.11 -m venv env
env/bin/python install.py
env/bin/sati --version
```

The wheelhouse contains Python dependencies as well as the bundled components.
A Python interpreter and platform system libraries are prerequisites. Rebuild
wheels on the target host for another Python ABI, operating system or CPU.
Electronic-structure executables (Gaussian, ORCA, etc.) are external programs;
the adapters read their existing results and do not include licensed programs.
The continuum electronic solver is a separate project; supplied continuum
operator coefficients can be used in the general FC/HT operator interface.

## Run and verify

Run numerical work and compilation on a compute host:

```
env/bin/sati run-esalc source/examples/trinity/esalc_vibronic/request.json results/esalc.json
env/bin/sati build-native bin/sati-harmonic
MATRIX_VIBRONIC_EXECUTABLE="$PWD/bin/sati-harmonic" env/bin/python -m pytest --import-mode=importlib source/tests
```

The C++ build requires a C++17 compiler and BLAS/LAPACK development libraries
(default `-lopenblas`; override with repeated `--lapack-flag=...`). Compilation
is explicit; importing SATI never triggers a build. Prepared physical harmonic
and general operator inputs run with `sati run-native INPUT OUTPUT --executable
bin/sati-harmonic`. The Python APIs in `matrix_trinity`, `matrix_dvr` and
`matrix_vpt2_vci` provide the established anharmonic workflows. The supplied
regressions cover native FC/HT, common quadratic modes and finite-space
variational nuclear contractions; they are not new molecular benchmarks.

## Sources and provenance

`source-manifest.json` pins the scientific revision and hashes each exported
source. `wheelhouse-manifest.json` hashes every wheel and records the build
platform. The installer verifies wheel hashes and checks dependency consistency.
To rebuild on a connected compute host with Python 3.11 and a compiler:

```
python source/tools/build/build_sati_standalone.py build RELEASE_DIRECTORY
```

Run this on a fresh source export before the `wheels` directory exists. The
export command is run by the maintainer against a pinned Git revision. It reads
committed source objects only, excluding uncommitted work and numerical caches.
BSD-3-Clause applies to the original code (`LICENSE`); bundled third-party
sources retain the license files included alongside them. Build tools contact
Python package indexes; the completed distribution installs without them.

## Window-qualified state assignments

The installed APIs include positive-bank and per-observable profile stopping,
per-window occupation rankings with one shared integral cache, deduplication,
restart and automatic state-to-state FC/HT assignment reports. The complete
small example uses two windows and finite-temperature signed ECD:

```
env/bin/python source/examples/trinity/targeted_ti/run.py \
  --executable bin/sati-harmonic --output results/targeted-ti
env/bin/python source/benchmarks/vibronic_targeted_ti/qualify_extensions.py \
  bin/sati-harmonic results/targeted-qualification.json
```

The example must return `passed: true`; its README describes the profiles,
complete transition columns, assignment report and resumable checkpoint.
The qualification driver must pass all 17 extension checks. These calculations
use installed modules, not a source-tree `PYTHONPATH` override. No electronic
program, external geometry or network request is used by the example.

## Validation and citation

Scientific revision: `e801a16f1e7b065ccdc7975d669ebcbce18e23b3`.
The archived Linux release passed 188 native/common-mode/variational regressions
and 17 targeted-selection checks on the reference compute host; recorded results
are in `validation/`. These are archived checks of the reference release.

Associated manuscript: Vincenzo Barone, *Effective Harmonic and Anharmonic Models
for Vibronic Spectra*, submitted to arXiv on 3 November 2026. The public arXiv
identifier was pending when this release was prepared.

Please cite [CITATION.cff](CITATION.cff). The original code uses the BSD 3-Clause
license; bundled third-party sources retain their accompanying licenses.
