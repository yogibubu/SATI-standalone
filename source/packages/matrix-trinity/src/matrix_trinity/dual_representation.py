"""Paired Cartesian and curvilinear representations for spectroscopy.

The two channels are deliberately retained together.  Curvilinear SONIC normal
modes are the preferred vibrational coordinates for GF/VPT2, whereas Cartesian
property derivatives remain the authoritative source for intensities and form
an independent representation check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Literal, Mapping

import numpy as np


VIBRATIONAL_DUAL_REPRESENTATION_SCHEMA = (
    "matrix.trinity.vibrational-dual-representation.v1"
)
RepresentationSelection = Literal["cartesian", "curvilinear", "both"]


@dataclass(frozen=True)
class CartesianSpectroscopicChannel:
    """Mass-independent Cartesian derivatives and property surfaces."""

    harmonic_hessian: np.ndarray
    cubic_force_field: Any | None = None
    dipole_derivatives: np.ndarray | None = None
    polarizability_derivatives: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hessian = np.asarray(self.harmonic_hessian, dtype=float)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("Cartesian harmonic Hessian must be square")
        object.__setattr__(self, "harmonic_hessian", hessian)
        for name in ("dipole_derivatives", "polarizability_derivatives"):
            value = getattr(self, name)
            if value is None:
                continue
            derivative = np.asarray(value, dtype=float)
            if derivative.ndim != 2 or derivative.shape[1] != hessian.shape[0]:
                raise ValueError(
                    f"{name} must have one column per Cartesian coordinate"
                )
            object.__setattr__(self, name, derivative)

    @property
    def carries_intensity_information(self) -> bool:
        return (
            self.dipole_derivatives is not None
            or self.polarizability_derivatives is not None
        )


@dataclass(frozen=True)
class CurvilinearSonicChannel:
    """Potential and kinetic derivatives in nonredundant SONIC coordinates."""

    harmonic_force_field: np.ndarray
    cubic_force_field: np.ndarray
    gf_modes: np.ndarray
    kinetic_metric: np.ndarray
    kinetic_metric_first: np.ndarray | None = None
    kinetic_metric_second_diagonal: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        harmonic = np.asarray(self.harmonic_force_field, dtype=float)
        cubic = np.asarray(self.cubic_force_field, dtype=float)
        modes = np.asarray(self.gf_modes, dtype=float)
        metric = np.asarray(self.kinetic_metric, dtype=float)
        if harmonic.ndim != 2 or harmonic.shape[0] != harmonic.shape[1]:
            raise ValueError("SONIC harmonic force field must be square")
        ncoord = harmonic.shape[0]
        if cubic.shape != (ncoord, ncoord, ncoord):
            raise ValueError("SONIC cubic force field dimensions are inconsistent")
        if modes.shape != (ncoord, ncoord):
            raise ValueError("GF mode matrix must span the nonredundant SONIC space")
        if metric.shape != (ncoord, ncoord):
            raise ValueError("SONIC kinetic metric must be square")
        if np.linalg.matrix_rank(modes) != ncoord:
            raise ValueError("GF mode matrix is rank deficient")
        object.__setattr__(self, "harmonic_force_field", harmonic)
        object.__setattr__(self, "cubic_force_field", cubic)
        object.__setattr__(self, "gf_modes", modes)
        object.__setattr__(self, "kinetic_metric", metric)
        if self.kinetic_metric_first is not None:
            first = np.asarray(self.kinetic_metric_first, dtype=float)
            if first.shape != (ncoord, ncoord, ncoord):
                raise ValueError("first kinetic-metric derivative has invalid shape")
            object.__setattr__(self, "kinetic_metric_first", first)
        if self.kinetic_metric_second_diagonal is not None:
            second = np.asarray(self.kinetic_metric_second_diagonal, dtype=float)
            if second.shape != (ncoord, ncoord, ncoord):
                raise ValueError(
                    "diagonal second kinetic-metric derivative has invalid shape"
                )
            object.__setattr__(self, "kinetic_metric_second_diagonal", second)


@dataclass(frozen=True)
class DualRepresentationVibrationalField:
    """One immutable result retaining both spectroscopic representations."""

    cartesian: CartesianSpectroscopicChannel
    curvilinear: CurvilinearSonicChannel
    schema: str = VIBRATIONAL_DUAL_REPRESENTATION_SCHEMA

    def select(self, purpose: str) -> RepresentationSelection:
        """Return the required channel without discarding the other one."""

        normalized = purpose.strip().lower().replace("-", "_")
        if normalized in {
            "ir",
            "raman",
            "vcd",
            "roa",
            "intensity",
            "intensities",
            "property_surface",
        }:
            return "cartesian"
        if normalized in {
            "gf",
            "vpt2",
            "delta_b_vib",
            "deltabvib",
            "rovibrational",
            "isotopic_cubic",
        }:
            return "curvilinear"
        if normalized in {"archive", "validation", "serialize", "replay"}:
            return "both"
        raise ValueError(f"unknown vibrational purpose: {purpose}")


def prepare_variational_handoff(
    field: DualRepresentationVibrationalField,
    *,
    solver: str,
    basis_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the immutable TRINITY handoff to a variational solver."""
    solver_name = str(solver).strip()
    if not solver_name or not basis_contract:
        raise ValueError("variational handoff requires a solver and basis contract")
    channel = field.select("rovibrational")
    fingerprint_payload = {
        "schema": field.schema,
        "solver": solver_name,
        "basis_contract": dict(basis_contract),
        "harmonic_shape": list(field.curvilinear.harmonic_force_field.shape),
        "cubic_shape": list(field.curvilinear.cubic_force_field.shape),
    }
    fingerprint = sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "matrix.trinity.variational_handoff.v1",
        "status": "READY",
        "representation": channel,
        "solver": solver_name,
        "basis_contract": dict(basis_contract),
        "field_schema": field.schema,
        "sha256": fingerprint,
    }


__all__ = [
    "CartesianSpectroscopicChannel",
    "CurvilinearSonicChannel",
    "DualRepresentationVibrationalField",
    "RepresentationSelection",
    "VIBRATIONAL_DUAL_REPRESENTATION_SCHEMA",
    "prepare_variational_handoff",
]
