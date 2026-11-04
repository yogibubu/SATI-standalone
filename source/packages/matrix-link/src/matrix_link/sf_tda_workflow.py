"""Immutable KEYMAKER/LINK workflow for resident spin-flip TDA states."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from .optimizer import (
    GeometryEvaluationService,
    OptimizerCoordinateModel,
    OptimizerResult,
    OptimizerSettings,
    optimize_geometry,
)
from .mex import MECIResult, meci_manifold_from_link_service, optimize_meci
from .scan import QMScanBackend


KEYMAKER_SF_TDA_PROTOCOL_SCHEMA = "matrix.keymaker.sf_tda_optimization.v1"
SF_TDA_MECI_RESTART_SCHEMA = "matrix.link.sf_tda_meci_restart.v1"


@dataclass(frozen=True)
class SFTDAOptimizationProtocol:
    """Frozen scientific choices for a root-followed SF-TDA optimization."""

    target_state_id: int
    initial_roots: int = 6
    root_increment: int = 2
    maximum_roots: int = 12
    minimum_overlap: float = 0.70
    small_gap_minimum_overlap: float = 0.50
    ambiguity_margin: float = 0.05

    def __post_init__(self) -> None:
        target = int(self.target_state_id)
        initial = int(self.initial_roots)
        increment = int(self.root_increment)
        maximum = int(self.maximum_roots)
        if target < 1:
            raise ValueError("SF-TDA target states are one-based")
        if initial < target or increment < 1 or maximum < initial:
            raise ValueError("SF-TDA root-expansion limits are inconsistent")
        if not 0.0 < float(self.small_gap_minimum_overlap) <= float(
            self.minimum_overlap
        ) <= 1.0:
            raise ValueError("SF-TDA overlap thresholds are inconsistent")
        if not 0.0 <= float(self.ambiguity_margin) < 1.0:
            raise ValueError("SF-TDA ambiguity margin must lie in [0, 1)")

    def backend(self, *, charge: int, multiplicity: int, processors: int = 1) -> QMScanBackend:
        if int(multiplicity) < 2:
            raise ValueError("spin-flip TDA requires an open-shell reference")
        return QMScanBackend(
            name="matrix_gfn2",
            method="MATRIX-GFN2",
            charge=int(charge),
            multiplicity=int(multiplicity),
            processors=int(processors),
            electronic_state=int(self.target_state_id),
            excited_states=int(self.initial_roots),
            state_spin="spin-flip",
            state_tracking="apoc",
            state_tracking_minimum_overlap=float(self.minimum_overlap),
            state_tracking_small_gap_minimum_overlap=float(
                self.small_gap_minimum_overlap
            ),
            state_tracking_ambiguity_margin=float(self.ambiguity_margin),
            state_tracking_initial_roots=int(self.initial_roots),
            state_tracking_root_increment=int(self.root_increment),
            state_tracking_max_roots=int(self.maximum_roots),
            gradient_mode="analytic",
            properties=("energy", "gradient"),
        )


@dataclass(frozen=True)
class SFTDAMECIProtocol:
    """Frozen FAST_TDA state-pair contract for a LINK MECI search."""

    state_ids: tuple[int, int]
    roots: int = 6
    gap_tolerance_hartree: float = 1.0e-6
    seam_gradient_tolerance: float = 1.0e-5
    maximum_iterations: int = 100
    minimum_principal_overlap: float = 0.70
    ambiguity_margin: float = 0.02
    maximum_step: float = 0.10

    def __post_init__(self) -> None:
        first, second = (int(value) for value in self.state_ids)
        if min(first, second) < 1 or first == second:
            raise ValueError("SF-TDA MECI states must be distinct and one-based")
        if int(self.roots) < max(first, second):
            raise ValueError("SF-TDA MECI root count must contain both states")
        if min(self.gap_tolerance_hartree, self.seam_gradient_tolerance) <= 0.0:
            raise ValueError("SF-TDA MECI tolerances must be positive")
        if int(self.maximum_iterations) < 1:
            raise ValueError("SF-TDA MECI iteration limit must be positive")
        if not 0.0 < float(self.minimum_principal_overlap) <= 1.0:
            raise ValueError("SF-TDA MECI principal-overlap threshold is invalid")
        if not 0.0 <= float(self.ambiguity_margin) < 1.0:
            raise ValueError("SF-TDA MECI ambiguity margin is invalid")
        if float(self.maximum_step) <= 0.0:
            raise ValueError("SF-TDA MECI maximum step must be positive")

    def backend(self, *, charge: int, multiplicity: int, processors: int = 1) -> QMScanBackend:
        return SFTDAOptimizationProtocol(
            target_state_id=min(self.state_ids),
            initial_roots=int(self.roots),
            maximum_roots=max(int(self.roots), 12),
        ).backend(
            charge=int(charge),
            multiplicity=int(multiplicity),
            processors=int(processors),
        )


def optimize_sf_tda(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    coordinate_model: OptimizerCoordinateModel,
    target_state_id: int,
    charge: int,
    multiplicity: int,
    processors: int = 1,
    protocol: SFTDAOptimizationProtocol | None = None,
    settings: OptimizerSettings | None = None,
) -> OptimizerResult:
    """Optimize one SF-TDA state with strict APOC character following."""

    frozen = protocol or SFTDAOptimizationProtocol(target_state_id)
    if int(frozen.target_state_id) != int(target_state_id):
        raise ValueError("SF-TDA protocol and requested target state disagree")
    optimizer_settings = settings or OptimizerSettings(stationary_point="minimum")
    if not optimizer_settings.excited_state_gap_control:
        optimizer_settings = replace(
            optimizer_settings,
            excited_state_gap_control=True,
        )
    return optimize_geometry(
        xyzin_path,
        run_dir=run_dir,
        coordinate_model=coordinate_model,
        backend=frozen.backend(
            charge=int(charge),
            multiplicity=int(multiplicity),
            processors=int(processors),
        ),
        settings=optimizer_settings,
        allow_native_sycart=False,
    )


def optimize_sf_tda_meci(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    coordinate_model: OptimizerCoordinateModel,
    state_ids: tuple[int, int],
    charge: int,
    multiplicity: int,
    processors: int = 1,
    protocol: SFTDAMECIProtocol | None = None,
    settings: OptimizerSettings | None = None,
    resume_state_tracking: bool = False,
) -> MECIResult:
    """Locate a FAST_TDA minimum-energy conical intersection through LINK."""

    frozen = protocol or SFTDAMECIProtocol(state_ids)
    if tuple(int(value) for value in state_ids) != tuple(frozen.state_ids):
        raise ValueError("SF-TDA MECI protocol and requested states disagree")
    from matrix_core import resolve_keymaker_protocol

    keymaker = resolve_keymaker_protocol("link-sf-tda-meci.v1")
    if (
        keymaker.owner_api != "matrix_link.sf_tda_workflow"
        or keymaker.action != "optimize_sf_tda_meci"
    ):
        raise RuntimeError("KEYMAKER SF-TDA MECI owner binding is inconsistent")
    protocol_record_path = Path(run_dir) / "keymaker_protocol.json"
    protocol_record_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_record = {
        "protocol_id": keymaker.protocol_id,
        "manifest_version": keymaker.manifest_version,
        "sha256": keymaker.sha256,
        "owner_api": keymaker.owner_api,
        "action": keymaker.action,
    }
    protocol_temporary = protocol_record_path.with_name(
        f".{protocol_record_path.name}.tmp"
    )
    protocol_temporary.write_text(
        json.dumps(protocol_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol_temporary.replace(protocol_record_path)
    service = GeometryEvaluationService(
        xyzin_path=xyzin_path,
        run_dir=run_dir,
        coordinate_model=coordinate_model,
        backend=frozen.backend(
            charge=int(charge),
            multiplicity=int(multiplicity),
            processors=int(processors),
        ),
        settings=settings or OptimizerSettings(stationary_point="minimum"),
    )
    checkpoint_path = Path(run_dir) / "sf_tda_tracking_checkpoint.json"
    optimizer_checkpoint_path = Path(run_dir) / "sf_tda_meci_restart.json"
    session = service._matrix_gfn2_session
    if session is None:
        raise RuntimeError("FAST_TDA MECI workflow did not create a resident session")
    restart_q = service.actual_q(service.reference_coordinates)
    restart_hessian = None
    iteration_offset = 0
    if resume_state_tracking:
        if not checkpoint_path.is_file() or not optimizer_checkpoint_path.is_file():
            raise FileNotFoundError(
                "complete SF-TDA MECI restart checkpoints do not exist in "
                f"{Path(run_dir)}"
            )
        session.load_sf_tda_tracking_checkpoint(checkpoint_path)
        payload = json.loads(optimizer_checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("schema") != SF_TDA_MECI_RESTART_SCHEMA:
            raise ValueError("unsupported SF-TDA MECI optimizer checkpoint")
        if (
            payload.get("keymaker_protocol_id") != keymaker.protocol_id
            or payload.get("keymaker_protocol_sha256") != keymaker.sha256
        ):
            raise ValueError("SF-TDA MECI restart uses another KEYMAKER protocol")
        identity = (
            tuple(payload.get("state_ids", ())),
            str(payload.get("coordinate_kind", "")),
            tuple(payload.get("coordinate_labels", ())),
        )
        expected = (
            tuple(int(value) for value in frozen.state_ids),
            str(coordinate_model.kind),
            tuple(coordinate_model.labels),
        )
        if identity != expected:
            raise ValueError("SF-TDA MECI optimizer checkpoint has another contract")
        restart_q = np.asarray(payload.get("coordinates"), dtype=float)
        restart_hessian = np.asarray(payload.get("approximate_hessian"), dtype=float)
        iteration_offset = int(payload.get("accepted_iteration", -1))

    def checkpoint_accepted_geometry(
        _iteration: int,
        _coordinates: object,
        _evaluation: object,
        _hessian: object,
    ) -> None:
        session.write_sf_tda_tracking_checkpoint(checkpoint_path)
        payload = {
            "schema": SF_TDA_MECI_RESTART_SCHEMA,
            "keymaker_protocol_id": keymaker.protocol_id,
            "keymaker_protocol_sha256": keymaker.sha256,
            "state_ids": [int(value) for value in frozen.state_ids],
            "coordinate_kind": str(coordinate_model.kind),
            "coordinate_labels": list(coordinate_model.labels),
            "accepted_iteration": int(_iteration),
            "coordinates": np.asarray(_coordinates, dtype=float).tolist(),
            "approximate_hessian": np.asarray(_hessian, dtype=float).tolist(),
        }
        optimizer_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = optimizer_checkpoint_path.with_name(
            f".{optimizer_checkpoint_path.name}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(optimizer_checkpoint_path)

    manifold = meci_manifold_from_link_service(
        service,
        state_ids=frozen.state_ids,
        roots=int(frozen.roots),
        gap_tolerance_hartree=float(frozen.gap_tolerance_hartree),
        minimum_principal_overlap=float(frozen.minimum_principal_overlap),
        ambiguity_margin=float(frozen.ambiguity_margin),
    )
    return optimize_meci(
        manifold,
        restart_q,
        max_iterations=int(frozen.maximum_iterations),
        seam_gradient_tolerance=float(frozen.seam_gradient_tolerance),
        gap_tolerance=float(frozen.gap_tolerance_hartree),
        maximum_step=float(frozen.maximum_step),
        accepted_geometry_callback=checkpoint_accepted_geometry,
        initial_hessian=restart_hessian,
        iteration_offset=iteration_offset,
    )


__all__ = [
    "KEYMAKER_SF_TDA_PROTOCOL_SCHEMA",
    "SF_TDA_MECI_RESTART_SCHEMA",
    "SFTDAOptimizationProtocol",
    "SFTDAMECIProtocol",
    "optimize_sf_tda",
    "optimize_sf_tda_meci",
]
