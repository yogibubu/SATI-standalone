"""Common end-to-end capability contract for MATRIX QM backends."""

from __future__ import annotations

from typing import Any, Mapping


QM_BACKENDS = ("GDV", "ORCA", "XTB", "Molpro", "MRCC", "PySCF", "PyFock", "eT")
QM_BACKEND_PHASES = ("optimization", "hessian", "controlled_failure")


def validate_qm_backend_matrix(matrix: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the backend coverage used by the E2E test harness."""

    missing_backend = [name for name in QM_BACKENDS if name not in matrix]
    if missing_backend:
        raise ValueError(f"QM backend matrix missing: {', '.join(missing_backend)}")
    errors: list[str] = []
    for name in QM_BACKENDS:
        entry = matrix[name]
        for phase in QM_BACKEND_PHASES:
            if not bool(entry.get(phase)):
                errors.append(f"{name}:{phase}")
    if errors:
        raise ValueError("QM backend matrix missing phases: " + ", ".join(errors))
    return {"backends": list(QM_BACKENDS), "phases": list(QM_BACKEND_PHASES), "valid": True}


def validate_qm_backend_fixture(backend: str, fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the data contract of one backend's offline E2E fixture."""
    if backend not in QM_BACKENDS:
        raise ValueError(f"unknown QM backend: {backend}")
    optimization = dict(fixture.get("optimization", {}))
    hessian = dict(fixture.get("hessian", {}))
    failure = dict(fixture.get("controlled_failure", {}))
    missing = []
    if not optimization.get("geometry") or "energy" not in optimization:
        missing.append("optimization.geometry/energy")
    if int(hessian.get("dimension", 0)) <= 0 or not hessian.get("frequencies_cm1"):
        missing.append("hessian.dimension/frequencies_cm1")
    if failure.get("status") != "failed" or not str(failure.get("reason", "")).strip():
        missing.append("controlled_failure.status/reason")
    if missing:
        raise ValueError(f"{backend} fixture missing: {', '.join(missing)}")
    return {"backend": backend, "valid": True}
