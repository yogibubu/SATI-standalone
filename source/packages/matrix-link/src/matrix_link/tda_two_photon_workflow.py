"""Immutable KEYMAKER/LINK workflow for maximizing resident TDA 2PA strength."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .optimizer import (
    OptimizerCoordinateModel,
    OptimizerResult,
    OptimizerSettings,
    optimize_geometry,
)
from .scan import QMScanBackend


@dataclass(frozen=True)
class TDATwoPhotonOptimizationProtocol:
    """Frozen closed-shell target and response controls for a 2PA search."""

    target_state_id: int
    spectral_range_ev: float = 20.0

    def __post_init__(self) -> None:
        if int(self.target_state_id) < 1:
            raise ValueError("TDA two-photon target states are one-based")
        if float(self.spectral_range_ev) <= 0.0:
            raise ValueError("TDA two-photon spectral range must be positive")

    def backend(self, *, charge: int, processors: int = 1) -> QMScanBackend:
        return QMScanBackend(
            name="matrix_gfn2",
            method="FAST_TDA-2PA",
            charge=int(charge),
            multiplicity=1,
            processors=int(processors),
            electronic_state=int(self.target_state_id),
            state_spin="two-photon",
            two_photon_spectral_range_ev=float(self.spectral_range_ev),
            gradient_mode="analytic",
            properties=("energy", "gradient"),
        )


def optimize_tda_two_photon_strength(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    coordinate_model: OptimizerCoordinateModel,
    target_state_id: int,
    charge: int = 0,
    processors: int = 1,
    protocol: TDATwoPhotonOptimizationProtocol | None = None,
    settings: OptimizerSettings | None = None,
) -> OptimizerResult:
    """Maximize one degenerate TDA two-photon strength with LINK."""

    frozen = protocol or TDATwoPhotonOptimizationProtocol(target_state_id)
    if int(frozen.target_state_id) != int(target_state_id):
        raise ValueError("TDA two-photon protocol and requested state disagree")
    from matrix_core import resolve_keymaker_protocol

    keymaker = resolve_keymaker_protocol("link-fast-tda-two-photon.v1")
    if (
        keymaker.owner_api != "matrix_link.tda_two_photon_workflow"
        or keymaker.action != "optimize_tda_two_photon_strength"
    ):
        raise RuntimeError("KEYMAKER TDA two-photon LINK binding is inconsistent")
    return optimize_geometry(
        xyzin_path,
        run_dir=run_dir,
        coordinate_model=coordinate_model,
        backend=frozen.backend(charge=int(charge), processors=int(processors)),
        settings=settings or OptimizerSettings(stationary_point="minimum"),
        allow_native_sycart=False,
    )


__all__ = ["TDATwoPhotonOptimizationProtocol", "optimize_tda_two_photon_strength"]
