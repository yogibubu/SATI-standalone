"""Compatibility imports for APOC-owned electronic-state adapters."""

import warnings

warnings.warn(
    "matrix_link.scf_adapters is deprecated; import matrix_apoc.scf_adapters",
    DeprecationWarning,
    stacklevel=2,
)

from matrix_apoc.scf_adapters import *  # noqa: F401,F403
from matrix_apoc.scf_adapters import __all__ as __all__
