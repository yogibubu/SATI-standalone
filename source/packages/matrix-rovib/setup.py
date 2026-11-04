"""Build the portable compiled LOCK Hamiltonian kernels."""

from __future__ import annotations

import sys

import numpy
from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "matrix_rovib._lock_native",
            ["src/matrix_rovib/_lock_native.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++",
            optional=False,
            extra_compile_args=(
                ["/std:c++17", "/O2"]
                if sys.platform == "win32"
                else ["-std=c++17", "-O3"]
            ),
        )
    ]
)
