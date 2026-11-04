"""Build the optional portable MATRIX chemistry kernels."""

from __future__ import annotations

import sys

from setuptools import Extension, setup

import numpy


setup(
    ext_modules=[
        Extension(
            "matrix_chem._chem_native",
            ["src/matrix_chem/_chem_native.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++",
            optional=False,
            extra_compile_args=(
                ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
            ),
        )
    ]
)
