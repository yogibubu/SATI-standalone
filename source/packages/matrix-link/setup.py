"""Build the native LINK optimizer extension."""

from __future__ import annotations

import sys

import numpy
from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "matrix_link._optimizer_native",
            ["src/matrix_link/_optimizer_native.cpp"],
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
