"""Analytic electrostatics for externally perceived local charge response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, TYPE_CHECKING

import numpy as np

from .charge_response import ChargeResponseSite, GaussianChargeResponseOperator
from .nonbonded import (
    electrostatic_energy_gradient,
    electrostatic_hessian_vector_product,
)

if TYPE_CHECKING:
    from matrix_epoc.cpcm import CPCMReactionField


ZAFF_LOCAL_CHARGE_ELECTROSTATICS_SCHEMA = (
    "matrix.zaff.local_charge_electrostatics.v1"
)


class LocalChargeResponse(Protocol):
    """Structural interface accepted from ORACLE or another perception layer."""

    charges_e: np.ndarray
    charge_jacobian_e_per_bohr: np.ndarray
    charge_hessian_entries_e_per_bohr2: Sequence[tuple[int, int, int, float]]

    def charge_directional_derivative(self, direction_bohr: np.ndarray) -> np.ndarray:
        """Return the charge derivative along one Cartesian direction."""


@dataclass(frozen=True)
class LocalChargeElectrostaticEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    charges_e: np.ndarray
    backend: str
    schema: str = ZAFF_LOCAL_CHARGE_ELECTROSTATICS_SCHEMA


def evaluate_local_charge_electrostatics(
    coordinates_bohr: np.ndarray,
    gaussian_widths_bohr: Sequence[float],
    response: LocalChargeResponse,
    *,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_sites: int = 256,
    reaction_field: CPCMReactionField | None = None,
) -> LocalChargeElectrostaticEvaluation:
    """Return E/G with the complete geometry-dependent-charge chain rule."""

    xyz, widths, charges = _validated_inputs(
        coordinates_bohr, gaussian_widths_bohr, response
    )
    operator = _operator(
        xyz,
        widths,
        charges,
        backend=backend,
        fmm_precision=fmm_precision,
        fmm_minimum_sites=fmm_minimum_sites,
        reaction_field=reaction_field,
    )
    selected = "fmm" if operator.dense_matrix is None else "direct"
    common = {
        "electrostatic_gaussian_widths_bohr": widths,
        "backend": selected,
        "fmm_precision": float(fmm_precision),
        "fmm_minimum_atoms": 0,
    }
    fixed = electrostatic_energy_gradient(xyz, charges, **common)
    energy = float(fixed.energy_hartree)
    gradient = fixed.gradient_hartree_per_bohr.copy()
    potential = operator.matrix_vector(charges) - operator.diagonal * charges
    if reaction_field is not None:
        reaction_energy, reaction_gradient = reaction_field.reaction_energy_gradient(
            xyz, charges
        )
        energy += float(reaction_energy)
        gradient += np.asarray(reaction_gradient).reshape(-1)
    gradient += np.asarray(
        response.charge_jacobian_e_per_bohr.T @ potential
    ).reshape(-1)
    return LocalChargeElectrostaticEvaluation(
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        charges_e=charges.copy(),
        backend=f"{operator.backend}_LOCAL_HBOND_CHAIN_RULE",
    )


def local_charge_electrostatic_hessian_vector_product(
    coordinates_bohr: np.ndarray,
    gaussian_widths_bohr: Sequence[float],
    response: LocalChargeResponse,
    vector_bohr: np.ndarray,
    *,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_sites: int = 256,
    reaction_field: CPCMReactionField | None = None,
) -> np.ndarray:
    """Apply the exact local-response electrostatic Hessian analytically."""

    xyz, widths, charges = _validated_inputs(
        coordinates_bohr, gaussian_widths_bohr, response
    )
    direction = np.asarray(vector_bohr, dtype=float)
    if direction.size == xyz.size:
        direction = direction.reshape(xyz.shape)
    if direction.shape != xyz.shape or np.any(~np.isfinite(direction)):
        raise ValueError("local-charge Hessian direction has the wrong shape")
    operator = _operator(
        xyz,
        widths,
        charges,
        backend=backend,
        fmm_precision=fmm_precision,
        fmm_minimum_sites=fmm_minimum_sites,
        reaction_field=reaction_field,
    )
    selected = "fmm" if operator.dense_matrix is None else "direct"
    common = {
        "electrostatic_gaussian_widths_bohr": widths,
        "backend": selected,
        "fmm_precision": float(fmm_precision),
        "fmm_minimum_atoms": 0,
    }
    flat_direction = direction.reshape(-1)
    charge_dot = response.charge_directional_derivative(flat_direction)
    potential = operator.matrix_vector(charges) - operator.diagonal * charges

    product = electrostatic_hessian_vector_product(
        xyz, charges, direction, **common
    )
    plus_gradient = electrostatic_energy_gradient(
        xyz, charges + charge_dot, **common
    ).gradient_hartree_per_bohr
    minus_gradient = electrostatic_energy_gradient(
        xyz, charges - charge_dot, **common
    ).gradient_hartree_per_bohr
    product += 0.5 * (plus_gradient - minus_gradient)

    potential_dot = operator.directional_matrix_vector(charges, direction)
    potential_dot += (
        operator.matrix_vector(charge_dot) - operator.diagonal * charge_dot
    )
    if reaction_field is not None:
        product += reaction_field.hessian_vector_product(
            xyz, charges, direction
        ).reshape(-1)
        _, plus_reaction_gradient = reaction_field.reaction_energy_gradient(
            xyz, charges + charge_dot
        )
        _, minus_reaction_gradient = reaction_field.reaction_energy_gradient(
            xyz, charges - charge_dot
        )
        product += 0.5 * (
            np.asarray(plus_reaction_gradient).reshape(-1)
            - np.asarray(minus_reaction_gradient).reshape(-1)
        )

    product += np.asarray(
        response.charge_jacobian_e_per_bohr.T @ potential_dot
    ).reshape(-1)
    product += _weighted_charge_hessian_vector(
        response, potential, flat_direction
    )
    return np.asarray(product, dtype=float).reshape(-1)


def local_charge_electrostatic_hessian(
    coordinates_bohr: np.ndarray,
    gaussian_widths_bohr: Sequence[float],
    response: LocalChargeResponse,
    **kwargs: object,
) -> np.ndarray:
    """Materialize the analytic Hessian for stationary-point/frequency work."""

    size = np.asarray(coordinates_bohr).size
    columns = [
        local_charge_electrostatic_hessian_vector_product(
            coordinates_bohr,
            gaussian_widths_bohr,
            response,
            np.eye(size)[column],
            **kwargs,
        )
        for column in range(size)
    ]
    hessian = np.column_stack(columns)
    return 0.5 * (hessian + hessian.T)


def _weighted_charge_hessian_vector(
    response: LocalChargeResponse,
    weights: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(vector)
    for atom, left, right, value in response.charge_hessian_entries_e_per_bohr2:
        coefficient = float(weights[atom]) * value
        result[left] += coefficient * vector[right]
        if left == right:
            continue
        result[right] += coefficient * vector[left]
    return result


def _operator(
    xyz: np.ndarray,
    widths: np.ndarray,
    charges: np.ndarray,
    *,
    backend: Literal["auto", "direct", "fmm"],
    fmm_precision: float,
    fmm_minimum_sites: int,
    reaction_field: CPCMReactionField | None,
) -> GaussianChargeResponseOperator:
    sites = tuple(
        ChargeResponseSite(float(charges[index]), float(widths[index]), ((index, 1.0),))
        for index in range(len(charges))
    )
    return GaussianChargeResponseOperator.compile(
        xyz,
        sites,
        backend=backend,
        fmm_precision=fmm_precision,
        fmm_minimum_sites=fmm_minimum_sites,
        reaction_field=reaction_field,
    )


def _validated_inputs(
    coordinates_bohr: np.ndarray,
    gaussian_widths_bohr: Sequence[float],
    response: LocalChargeResponse,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(coordinates_bohr, dtype=float)
    widths = np.asarray(gaussian_widths_bohr, dtype=float).reshape(-1)
    charges = np.asarray(response.charges_e, dtype=float).reshape(-1)
    if (
        xyz.shape != (len(charges), 3)
        or widths.shape != charges.shape
        or np.any(~np.isfinite(xyz))
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise ValueError("local-charge electrostatic inputs are inconsistent")
    return xyz, widths, charges


__all__ = [
    "ZAFF_LOCAL_CHARGE_ELECTROSTATICS_SCHEMA",
    "LocalChargeResponse",
    "LocalChargeElectrostaticEvaluation",
    "evaluate_local_charge_electrostatics",
    "local_charge_electrostatic_hessian",
    "local_charge_electrostatic_hessian_vector_product",
]
