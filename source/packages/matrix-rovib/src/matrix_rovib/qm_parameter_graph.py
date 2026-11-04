"""Traceable mapping from quantum-chemical quantities to fitted LOCK parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


LOCK_QM_PARAMETER_GRAPH_SCHEMA = "matrix.lock.qm_parameter_graph.v1"


def build_qm_parameter_graph(
    qm_parameters: Sequence[Mapping[str, Any]],
    fitted_parameters: Sequence[Mapping[str, Any]],
    *,
    qm_covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
    experimental_jacobian: Sequence[Sequence[float]] | np.ndarray | None = None,
    augmented_jacobian: Sequence[Sequence[float]] | np.ndarray | None = None,
    rank_rcond: float = 1.0e-10,
    large_shift_sigma: float = 3.0,
) -> dict[str, Any]:
    """Build the auditable QM → operator → fit graph used in reports and Studio."""

    qm = [dict(item) for item in qm_parameters]
    fitted = [dict(item) for item in fitted_parameters]
    if not qm or not fitted or rank_rcond <= 0.0 or large_shift_sigma <= 0.0:
        raise ValueError("QM parameter graph requires nonempty parameters and valid thresholds")
    required_qm = {"path", "value", "sigma", "source_quantity", "operator_family"}
    required_fit = {"path", "value"}
    if any(not required_qm <= set(item) for item in qm):
        raise ValueError(f"QM parameters require fields {sorted(required_qm)}")
    if any(not required_fit <= set(item) for item in fitted):
        raise ValueError(f"fitted parameters require fields {sorted(required_fit)}")
    qm_paths = [str(item["path"]) for item in qm]
    fit_by_path = {str(item["path"]): item for item in fitted}
    if len(set(qm_paths)) != len(qm_paths) or len(fit_by_path) != len(fitted):
        raise ValueError("parameter graph paths must be unique")

    covariance = _covariance(qm, qm_covariance)
    rows = []
    missing = []
    for index, item in enumerate(qm):
        path = str(item["path"])
        fit = fit_by_path.get(path)
        if fit is None:
            missing.append(path)
            continue
        initial = float(item["value"])
        final = float(fit["value"])
        sigma = float(item["sigma"])
        if not np.isfinite((initial, final, sigma)).all() or sigma <= 0.0:
            raise ValueError(f"nonfinite value or invalid uncertainty for {path}")
        shift = final - initial
        rows.append({
            "path": path,
            "source_quantity": str(item["source_quantity"]),
            "source_artifact": item.get("source_artifact"),
            "transformation": item.get("transformation", "identity"),
            "operator_family": str(item["operator_family"]),
            "unit": item.get("unit", fit.get("unit", "")),
            "qm_value": initial,
            "qm_sigma": sigma,
            "fitted_value": final,
            "fitted_standard_error": fit.get("standard_error"),
            "shift": shift,
            "shift_over_qm_sigma": shift / sigma,
            "large_qm_shift": abs(shift / sigma) >= large_shift_sigma,
            "fit_status": fit.get("status", "refined"),
            "sensitivity": fit.get("sensitivity"),
            "interpretation": item.get("interpretation", ""),
        })

    displacement = np.asarray([
        float(fit_by_path[path]["value"]) - float(item["value"])
        for path, item in zip(qm_paths, qm, strict=True)
        if path in fit_by_path
    ])
    matched = [index for index, path in enumerate(qm_paths) if path in fit_by_path]
    correlated_distance = None
    if matched:
        precision = np.linalg.pinv(
            covariance[np.ix_(matched, matched)], rcond=rank_rcond
        )
        correlated_distance = float(np.sqrt(max(displacement @ precision @ displacement, 0.0)))

    experimental = _rank_report(experimental_jacobian, len(fitted), rank_rcond)
    augmented = _rank_report(augmented_jacobian, len(fitted), rank_rcond)
    data_rank = None if experimental is None else experimental["rank"]
    augmented_rank = None if augmented is None else augmented["rank"]
    for column, fit in enumerate(fitted):
        path = str(fit["path"])
        row = next((entry for entry in rows if entry["path"] == path), None)
        if row is None:
            continue
        if experimental is not None and column in experimental["null_dominated_columns"]:
            row["information_source"] = "QM-prior-dominated"
        elif data_rank is not None:
            row["information_source"] = "experiment-informed"
        else:
            row["information_source"] = "not-assessed"

    return {
        "schema": LOCK_QM_PARAMETER_GRAPH_SCHEMA,
        "parameters": rows,
        "missing_fitted_paths": missing,
        "unmapped_fitted_paths": sorted(set(fit_by_path) - set(qm_paths)),
        "qm_covariance": covariance.tolist(),
        "correlated_qm_displacement": correlated_distance,
        "large_shift_sigma": large_shift_sigma,
        "experimental_identifiability": experimental,
        "augmented_identifiability": augmented,
        "rank_gain_from_qm": (
            None if data_rank is None or augmented_rank is None else augmented_rank - data_rank
        ),
        "complete": not missing,
    }


def write_qm_parameter_graph(path: Path | str, graph: Mapping[str, Any]) -> Path:
    if graph.get("schema") != LOCK_QM_PARAMETER_GRAPH_SCHEMA:
        raise ValueError("unsupported QM parameter graph schema")
    from matrix_core.exchange_contracts import validate_exchange_tree
    graph = validate_exchange_tree(graph, expected_schema=LOCK_QM_PARAMETER_GRAPH_SCHEMA)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(dict(graph), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def _covariance(
    parameters: Sequence[Mapping[str, Any]],
    declared: Sequence[Sequence[float]] | np.ndarray | None,
) -> np.ndarray:
    if declared is None:
        result = np.diag([float(item["sigma"]) ** 2 for item in parameters])
    else:
        result = np.asarray(declared, dtype=float)
    if result.shape != (len(parameters), len(parameters)) or not np.all(np.isfinite(result)):
        raise ValueError("QM covariance has an incompatible shape or nonfinite entries")
    if not np.allclose(result, result.T, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("QM covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(result)) < -1.0e-12 * max(1.0, np.max(np.abs(result))):
        raise ValueError("QM covariance must be positive semidefinite")
    if np.any(np.diag(result) <= 0.0):
        raise ValueError("QM covariance diagonal must be positive")
    return result


def _rank_report(
    jacobian: Sequence[Sequence[float]] | np.ndarray | None,
    parameter_count: int,
    rcond: float,
) -> dict[str, Any] | None:
    if jacobian is None:
        return None
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != parameter_count or not np.all(np.isfinite(matrix)):
        raise ValueError("parameter graph Jacobian is incompatible")
    _left, singular, right_h = np.linalg.svd(matrix, full_matrices=True)
    threshold = rcond * singular[0] if singular.size else 0.0
    rank = int(np.count_nonzero(singular > threshold))
    null = right_h[rank:]
    null_weight = np.sum(np.abs(null) ** 2, axis=0) if null.size else np.zeros(parameter_count)
    return {
        "row_count": matrix.shape[0],
        "column_count": matrix.shape[1],
        "rank": rank,
        "singular_values": singular.tolist(),
        "condition_number": (
            None if rank == 0 else float(singular[0] / singular[rank - 1])
        ),
        "null_dominated_columns": [
            int(index) for index, value in enumerate(null_weight) if value >= 0.5
        ],
    }


__all__ = [
    "LOCK_QM_PARAMETER_GRAPH_SCHEMA",
    "build_qm_parameter_graph",
    "write_qm_parameter_graph",
]
