"""Provider-neutral DeltaML residual contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaMLEvaluation:
    energy: float
    gradient: np.ndarray
    residual_energy: float
    provenance: dict[str, str]


@dataclass(frozen=True)
class DeltaMLRecord:
    """Canonical, serializable training/evaluation record for a DeltaML point."""

    geometry_id: str
    dataset_id: str
    reference_model: str
    qm_method: str
    energy_reference_hartree: float
    energy_qm_hartree: float
    residual_energy_hartree: float
    gradient_qm_hartree_per_bohr: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "matrix.smith.delta_ml.record.v1",
            "geometry_id": self.geometry_id,
            "dataset_id": self.dataset_id,
            "reference_model": self.reference_model,
            "qm_method": self.qm_method,
            "energy_reference_hartree": float(self.energy_reference_hartree),
            "energy_qm_hartree": float(self.energy_qm_hartree),
            "residual_energy_hartree": float(self.residual_energy_hartree),
            "gradient_qm_hartree_per_bohr": list(self.gradient_qm_hartree_per_bohr),
        }

    def __post_init__(self) -> None:
        if any(not str(value).strip() for value in (self.geometry_id, self.dataset_id, self.reference_model, self.qm_method)):
            raise ValueError("DeltaML record identifiers and methods must be non-empty")
        if not np.isfinite(self.energy_qm_hartree) or not np.isfinite(self.energy_reference_hartree):
            raise ValueError("DeltaML record energies must be finite")


def evaluate_delta_ml_residual(
    reference_energy: float,
    reference_gradient: np.ndarray,
    residual_energy: float,
    residual_gradient: np.ndarray,
    *,
    reference_model: str,
    qm_reference: str,
    dataset_id: str,
) -> DeltaMLEvaluation:
    """Combine a reference surface and a learned residual without hiding provenance."""

    ref_gradient = np.asarray(reference_gradient, dtype=float).reshape(-1)
    delta_gradient = np.asarray(residual_gradient, dtype=float).reshape(-1)
    if ref_gradient.shape != delta_gradient.shape:
        raise ValueError("DeltaML reference and residual gradients must have equal shape")
    if not np.all(np.isfinite(ref_gradient)) or not np.all(np.isfinite(delta_gradient)):
        raise ValueError("DeltaML gradients must be finite")
    provenance = {
        "reference_model": str(reference_model),
        "qm_reference": str(qm_reference),
        "dataset_id": str(dataset_id),
    }
    if any(not value for value in provenance.values()):
        raise ValueError("DeltaML provenance fields must be non-empty")
    return DeltaMLEvaluation(
        float(reference_energy) + float(residual_energy),
        ref_gradient + delta_gradient,
        float(residual_energy),
        provenance,
    )
