"""Auditable ZAFF1 Hessian baseline subtraction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


ZAFF1_RESIDUAL_SCHEMA = "matrix.architect.zaff1_hessian_residual.v1"


def _hash_matrix(matrix: np.ndarray) -> str:
    return sha256(np.asarray(matrix, dtype=np.float64).tobytes()).hexdigest()


@dataclass(frozen=True)
class Zaff1HessianResidual:
    """QM Hessian minus explicitly fixed ZAFF-fast component Hessians."""

    qm_hessian: np.ndarray
    fixed_components: Mapping[str, np.ndarray]
    residual_hessian: np.ndarray
    fixed_terms: tuple[str, ...]
    fit_terms: tuple[str, ...]
    schema: str = ZAFF1_RESIDUAL_SCHEMA

    def __post_init__(self) -> None:
        qm = np.asarray(self.qm_hessian, dtype=float)
        residual = np.asarray(self.residual_hessian, dtype=float)
        if qm.ndim != 2 or qm.shape[0] != qm.shape[1] or residual.shape != qm.shape:
            raise ValueError("ZAFF1 Hessians must be square and shape-compatible")
        if not np.all(np.isfinite(qm)) or not np.all(np.isfinite(residual)):
            raise ValueError("ZAFF1 Hessian residual contains non-finite values")
        if set(self.fixed_terms) != set(self.fixed_components):
            raise ValueError("fixed_terms must match the supplied baseline components")
        if set(self.fixed_terms) & set(self.fit_terms):
            raise ValueError("a ZAFF1 term cannot be both fixed and fitted")
        for name, component in self.fixed_components.items():
            matrix = np.asarray(component, dtype=float)
            if matrix.shape != qm.shape or not np.all(np.isfinite(matrix)):
                raise ValueError(f"invalid ZAFF1 baseline component: {name}")
        expected = qm - sum((np.asarray(self.fixed_components[name], dtype=float) for name in self.fixed_terms), start=np.zeros_like(qm))
        if not np.allclose(residual, expected, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("ZAFF1 residual does not equal QM minus fixed baseline components")
        if not np.allclose(residual, residual.T, atol=1.0e-10, rtol=0.0):
            raise ValueError("ZAFF1 residual Hessian must be symmetric")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "qm_hessian_sha256": _hash_matrix(self.qm_hessian),
            "baseline_component_sha256": {
                name: _hash_matrix(component) for name, component in self.fixed_components.items()
            },
            "residual_hessian_sha256": _hash_matrix(self.residual_hessian),
            "fixed_terms": list(self.fixed_terms),
            "fit_terms": list(self.fit_terms),
            "definition": "H_QM_MINUS_ZAFF_FAST_FIXED_COMPONENTS",
        }


def subtract_zaff_fast_hessian_baseline(
    qm_hessian: Sequence[Sequence[float]] | np.ndarray,
    fixed_components: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    *,
    fit_terms: Sequence[str] = (),
) -> Zaff1HessianResidual:
    """Subtract non-bonded/electrostatic ZAFF-fast Hessian components once."""

    qm = np.asarray(qm_hessian, dtype=float)
    components = {str(name): np.asarray(value, dtype=float) for name, value in fixed_components.items()}
    fixed_terms = tuple(components)
    return Zaff1HessianResidual(
        qm_hessian=qm,
        fixed_components=components,
        residual_hessian=qm - sum(components.values(), start=np.zeros_like(qm)),
        fixed_terms=fixed_terms,
        fit_terms=tuple(str(term) for term in fit_terms),
    )


__all__ = ["ZAFF1_RESIDUAL_SCHEMA", "Zaff1HessianResidual", "subtract_zaff_fast_hessian_baseline"]
