# SATI standalone

This distribution contains the harmonic and anharmonic vibronic implementation,
its coordinate and input adapters, ZAFF, DVR and VPT2/VCI services, C++17 sources,
licenses, an ESALC TI/TD example and selected scientific regression checks.
It does not require a checkout or installation of the parent software suite.
The `matrix_*` Python namespaces are retained to reuse the scientific owners;
they are supplied by this archive, rather than fetched from a private registry.

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
