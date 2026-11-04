"""Certified Cartesian-to-primitive harmonic transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from matrix_smith import primitive_b_matrix_from_manifest

from .internal import RedundantPrimitiveHessian, redundant_primitive_hessian_from_cartesian


@dataclass(frozen=True)
class HarmonicFieldTransform:
    b_matrix: np.ndarray
    primitive_labels: tuple[str, ...]
    redundant_hessian: RedundantPrimitiveHessian


def transform_cartesian_hessian(
    cartesian_hessian: np.ndarray,
    coordinates_bohr: np.ndarray,
    masses_amu: np.ndarray,
    primitive_manifest: Mapping[str, Any] | None = None,
    *,
    frozen_b_matrix: np.ndarray | None = None,
    frozen_labels: tuple[str, ...] | None = None,
) -> HarmonicFieldTransform:
    """Return the single certified primitive representation of a Hessian."""
    if frozen_b_matrix is None:
        if primitive_manifest is None or frozen_labels is not None:
            raise ValueError("a primitive manifest or frozen SMITH rows are required")
        b_matrix, labels = primitive_b_matrix_from_manifest(coordinates_bohr, primitive_manifest)
    else:
        if primitive_manifest is not None or frozen_labels is None:
            raise ValueError("frozen SMITH rows require labels and no primitive manifest")
        b_matrix = np.asarray(frozen_b_matrix, dtype=float)
        labels = tuple(frozen_labels)
        if b_matrix.shape != (len(labels), np.asarray(coordinates_bohr).size):
            raise ValueError("frozen SMITH B matrix has the wrong shape")
        if not labels or len(set(labels)) != len(labels) or not np.all(np.isfinite(b_matrix)):
            raise ValueError("frozen SMITH B rows or labels are invalid")
    transformed = redundant_primitive_hessian_from_cartesian(
        cartesian_hessian, b_matrix, coordinates_bohr, masses_amu
    )
    centered = np.asarray(coordinates_bohr, dtype=float) - np.mean(coordinates_bohr, axis=0)
    linear = np.linalg.matrix_rank(centered, tol=1.0e-8) <= 1
    expected_rank = 3 * len(np.asarray(masses_amu).reshape(-1)) - (5 if linear else 6)
    if transformed.primitive_rank != expected_rank:
        raise ValueError(
            f"primitive vibrational rank is {transformed.primitive_rank}, expected {expected_rank}"
        )
    return HarmonicFieldTransform(b_matrix, labels, transformed)


__all__ = ["HarmonicFieldTransform", "transform_cartesian_hessian"]
