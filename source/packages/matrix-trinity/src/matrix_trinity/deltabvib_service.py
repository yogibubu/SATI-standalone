"""Cache-independent TRINITY service for isotope-specific vibration--rotation corrections.

MORPHEUS owns the decision to reuse corrections stored in XYZin.  This module owns only
their calculation from a persistent vibrational field, so it can also be called by other
consumers without importing the semiexperimental fitting layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .isotopic_internal_qff import NonredundantInternalCubicField


@dataclass(frozen=True)
class TrinityDeltaBVibCorrection:
    """One calculated correction and the provenance needed to audit it."""

    label: str
    substitutions: Mapping[int, int]
    delta_MHz: tuple[float, float, float]
    source: str
    acquisition: str
    normal_mode_basis: str


@dataclass(frozen=True)
class CurvilinearDeltaBVibJob:
    """Mass-independent SONIC F2/F3 field used for any requested isotopologue.

    ``acquisition`` records whether the cubic information came from modal gradients or
    Hessians.  ``normal_mode_basis`` records whether the parent field was constructed in
    Cartesian or SONIC normal modes.  Both choices lead to the same isotope-specific
    calculation contract and remain explicit in the cached provenance.
    """

    field: NonredundantInternalCubicField
    b_matrix: np.ndarray
    b_prime: np.ndarray
    atoms: tuple[str, ...]
    geometry_angstrom: np.ndarray
    acquisition: str = "hessian"
    normal_mode_basis: str = "sonic"

    def __post_init__(self) -> None:
        b_matrix = np.asarray(self.b_matrix, dtype=float)
        b_prime = np.asarray(self.b_prime, dtype=float)
        geometry = np.asarray(self.geometry_angstrom, dtype=float)
        ncoord = self.field.coordinate_count
        ncart = 3 * len(self.atoms)
        if b_matrix.shape != (ncoord, ncart):
            raise ValueError("B matrix must have shape (ncoord, 3N)")
        if b_prime.shape != (ncoord, ncart, ncart):
            raise ValueError("B-prime must have shape (ncoord, 3N, 3N)")
        if geometry.shape != (len(self.atoms), 3):
            raise ValueError("geometry must have shape (N, 3)")
        if self.acquisition not in {"gradient", "hessian"}:
            raise ValueError("DeltaBvib acquisition must be gradient or hessian")
        if self.normal_mode_basis not in {"cartesian", "sonic"}:
            raise ValueError("normal-mode basis must be cartesian or sonic")
        if not all(np.all(np.isfinite(value)) for value in (b_matrix, b_prime, geometry)):
            raise ValueError("DeltaBvib job arrays must be finite")
        object.__setattr__(self, "b_matrix", b_matrix)
        object.__setattr__(self, "b_prime", b_prime)
        object.__setattr__(self, "geometry_angstrom", geometry)

    def calculate(
        self, label: str, substitutions: Mapping[int, int]
    ) -> TrinityDeltaBVibCorrection:
        # The stationary-point chain rule returns the mass-independent Cartesian field.
        # Isotope masses, Eckart projection, normal modes, Coriolis constants and inertia
        # derivatives are then recomputed for the requested isotopologue.
        force2 = np.asarray(self.field.harmonic_internal, dtype=float)
        force3 = np.asarray(self.field.cubic_internal, dtype=float)
        b_matrix = self.b_matrix
        hessian = b_matrix.T @ force2 @ b_matrix
        cubic = np.einsum(
            "ijk,ia,jb,kc->abc", force3, b_matrix, b_matrix, b_matrix, optimize=True
        )
        term = np.einsum(
            "ij,iab,jc->abc", force2, self.b_prime, b_matrix, optimize=True
        )
        cubic += term + term.transpose(0, 2, 1) + term.transpose(2, 1, 0)
        cubic = _symmetrize_rank3(cubic)

        # Imported lazily because Gaussian-format adapters are not required by the core
        # TRINITY representation.  The numerical rovibrational kernel is provider-neutral.
        try:
            from matrix_gaussian import (
                compute_deltabvib_from_semidiagonal_cubic_data,
                semidiagonal_data_from_cartesian_cubic,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CurvilinearDeltaBVibJob requires the matrix-trinity[spectroscopy] "
                "optional dependencies"
            ) from exc

        data = semidiagonal_data_from_cartesian_cubic(
            self.atoms,
            self.geometry_angstrom,
            hessian,
            cubic,
            substitutions={int(key): int(value) for key, value in substitutions.items()},
        )
        result = compute_deltabvib_from_semidiagonal_cubic_data(data)
        if not np.all(np.isfinite(result.total_MHz)):
            raise ValueError("TRINITY produced a non-finite DeltaBvib correction")
        source = (
            "TRINITY nonredundant-internal F2/F3; "
            f"acquisition={self.acquisition}; normal_modes={self.normal_mode_basis}"
        )
        return TrinityDeltaBVibCorrection(
            label=str(label),
            substitutions={int(key): int(value) for key, value in substitutions.items()},
            delta_MHz=tuple(float(value) for value in result.total_MHz),
            source=source,
            acquisition=self.acquisition,
            normal_mode_basis=self.normal_mode_basis,
        )


@dataclass(frozen=True)
class TrinityDeltaBVibService:
    """Select and execute a TRINITY correction job.

    A single mass-independent job normally serves the parent and every isotopologue.  A
    sequence is accepted so future providers can dispatch distinct molecular parents.
    """

    jobs: tuple[CurvilinearDeltaBVibJob, ...]

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("TRINITY DeltaBvib service requires at least one job")

    def calculate(
        self, label: str, substitutions: Mapping[int, int]
    ) -> TrinityDeltaBVibCorrection:
        if len(self.jobs) != 1:
            raise ValueError(
                "multiple TRINITY DeltaBvib jobs require an explicit molecular dispatcher"
            )
        return self.jobs[0].calculate(label, substitutions)


def _symmetrize_rank3(tensor: np.ndarray) -> np.ndarray:
    return (
        tensor
        + tensor.transpose(0, 2, 1)
        + tensor.transpose(1, 0, 2)
        + tensor.transpose(1, 2, 0)
        + tensor.transpose(2, 0, 1)
        + tensor.transpose(2, 1, 0)
    ) / 6.0


__all__ = [
    "CurvilinearDeltaBVibJob",
    "TrinityDeltaBVibCorrection",
    "TrinityDeltaBVibService",
]
