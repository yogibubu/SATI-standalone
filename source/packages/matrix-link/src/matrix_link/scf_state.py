"""Compatibility imports for the APOC-owned SCF state contract."""

import warnings

warnings.warn(
    "matrix_link.scf_state is deprecated; import matrix_apoc.state_reuse",
    DeprecationWarning,
    stacklevel=2,
)

from matrix_apoc.state_reuse import *  # noqa: F401,F403
from matrix_apoc.state_reuse import __all__ as __all__
