"""Deprecated compatibility facade for the EPOC variational cavity."""

import warnings

warnings.warn(
    "matrix_zaff.variational_cavity is deprecated; import matrix_epoc.variational_cavity",
    DeprecationWarning,
    stacklevel=2,
)

from matrix_epoc.variational_cavity import *  # noqa: F401,F403
from matrix_epoc.variational_cavity import __all__ as __all__
