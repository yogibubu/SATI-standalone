"""Deprecated compatibility facade for EPOC reaction-field kernels."""

import warnings

warnings.warn(
    "matrix_zaff.cpcm is deprecated; import matrix_epoc.cpcm",
    DeprecationWarning,
    stacklevel=2,
)

from matrix_epoc.cpcm import *  # noqa: F401,F403
from matrix_epoc.cpcm import __all__ as __all__
