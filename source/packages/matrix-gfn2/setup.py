"""Build the native MATRIX-GFN2 numerical kernels."""

from __future__ import annotations

import sys
from ctypes.util import find_library

import numpy
from setuptools import Extension, setup


native_link_args: list[str] = []
native_libraries: list[str] = []
native_macros: list[tuple[str, str]] = []
if sys.platform == "darwin":
    native_link_args.extend(["-framework", "Accelerate"])
    native_macros.append(("MATRIX_GFN2_NATIVE_LAPACK", "1"))
elif sys.platform.startswith("linux"):
    lapack = find_library("lapack")
    blas = find_library("blas")
    if lapack and blas:
        native_libraries.extend(["lapack", "blas"])
    else:
        # Minimal compute nodes often ship OpenBLAS as a versioned runtime
        # library without the liblapack.so/libblas.so development symlinks.
        # OpenBLAS provides both the CBLAS and LAPACK entry points used here.
        openblas = find_library("openblas")
        if not openblas:
            raise RuntimeError(
                "matrix-gfn2 native kernels require LAPACK/BLAS or OpenBLAS; "
                "install the system development libraries or OpenBLAS"
            )
        native_link_args.append(f"-Wl,-l:{openblas}")
    native_macros.append(("MATRIX_GFN2_NATIVE_LAPACK", "1"))


setup(
    ext_modules=[
        Extension(
            "matrix_gfn2._gfn2_native",
            ["src/matrix_gfn2/_gfn2_native.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++",
            optional=False,
            libraries=native_libraries,
            define_macros=native_macros,
            extra_link_args=native_link_args,
            extra_compile_args=(
                ["/std:c++17", "/O2"]
                if sys.platform == "win32"
                else ["-std=c++17", "-O3"]
            ),
        )
    ]
)
