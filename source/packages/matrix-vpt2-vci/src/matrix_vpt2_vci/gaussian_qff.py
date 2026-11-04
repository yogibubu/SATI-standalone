from __future__ import annotations

"""Compatibility wrappers for Gaussian QFF adapters.

Gaussian file parsing is owned by `matrix-gaussian`; VPT2/VCI keeps these
imports so older ORACLE call sites continue to work.
"""

from matrix_gaussian import (
    FCHKData,
    anharmonic_input_from_gaussian_fchk,
    hessian_input_from_gaussian_fchk,
    lower_to_symmetric,
    read_gaussian_fchk_qff,
    read_indexed_qff_text,
)

__all__ = [
    "FCHKData",
    "anharmonic_input_from_gaussian_fchk",
    "hessian_input_from_gaussian_fchk",
    "lower_to_symmetric",
    "read_gaussian_fchk_qff",
    "read_indexed_qff_text",
]
