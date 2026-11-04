"""Fail-closed symmetry continuity gate for SONIC paths."""

from __future__ import annotations

from dataclasses import dataclass

from .salc_snapshot import SALCPathDiagnostics


SYMMETRY_PATH_GATE_SCHEMA = "matrix.smith.symmetry_path_gate.v1"


@dataclass(frozen=True)
class SymmetryPathGate:
    schema: str
    status: str
    minimum_overlap: float
    warning_threshold: float
    failure_threshold: float
    warning_steps: tuple[int, ...]
    failure_steps: tuple[int, ...]
    message: str

    def scientific_validation_record(self):
        from matrix_core import ScientificValidationRecord

        return ScientificValidationRecord(
            owner="SMITH",
            subject="SYMMETRY_PATH_CONTINUITY",
            passed=self.status != "FAIL",
            metrics={
                "minimum_subspace_overlap": self.minimum_overlap,
                "warning_step_count": float(len(self.warning_steps)),
                "failure_step_count": float(len(self.failure_steps)),
            },
            thresholds={
                "warning_overlap": self.warning_threshold,
                "failure_overlap": self.failure_threshold,
            },
            context={"status": self.status, "message": self.message},
        )


def evaluate_symmetry_path_gate(
    diagnostics: SALCPathDiagnostics,
    *,
    warning_threshold: float = 0.98,
    failure_threshold: float = 0.90,
) -> SymmetryPathGate:
    """Convert SALC path diagnostics into a workflow-level safety decision."""

    warning = float(warning_threshold)
    failure = float(failure_threshold)
    if not 0.0 < failure <= warning <= 1.0:
        raise ValueError("symmetry path thresholds must satisfy 0 < failure <= warning <= 1")
    overlaps = tuple(float(step.min_subspace_overlap) for step in diagnostics.steps)
    minimum = min(overlaps, default=1.0)
    warnings = tuple(
        int(step.step) for step in diagnostics.steps if step.min_subspace_overlap < warning
    )
    failures = tuple(
        int(step.step) for step in diagnostics.steps if step.min_subspace_overlap < failure
    )
    if failures:
        status = "FAIL"
        message = "SALC subspace continuity is below the fail-closed threshold"
    elif warnings or not diagnostics.ok:
        status = "WARN"
        message = "SALC path needs gauge/overlap review before production use"
    else:
        status = "PASS"
        message = "SALC subspaces remain continuous along the path"
    return SymmetryPathGate(
        schema=SYMMETRY_PATH_GATE_SCHEMA,
        status=status,
        minimum_overlap=minimum,
        warning_threshold=warning,
        failure_threshold=failure,
        warning_steps=warnings,
        failure_steps=failures,
        message=message,
    )
