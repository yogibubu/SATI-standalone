"""Deprecated compatibility facade for EPOC confinement kernels."""

import warnings

warnings.warn(
    "matrix_zaff.confinement is deprecated; import matrix_epoc.confinement",
    DeprecationWarning,
    stacklevel=2,
)

from matrix_epoc.confinement import *  # noqa: F401,F403
from matrix_epoc.confinement import __all__ as __all__
