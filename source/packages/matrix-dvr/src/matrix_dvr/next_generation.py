"""Next-generation numerical and qualification utilities for SONIC/DVR."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable, Sequence

import numpy as np


def validate_experimental_reference(predicted: Sequence[float], observed: Sequence[float], *, sigma: Sequence[float] | None = None) -> dict[str, object]:
    pred, obs = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    if pred.shape != obs.shape or pred.ndim != 1:
        raise ValueError("predicted and observed values must have equal one-dimensional shapes")
    residual = pred - obs
    weights = np.ones_like(residual) if sigma is None else 1.0 / np.maximum(np.asarray(sigma, dtype=float), 1.0e-15) ** 2
    chi2 = float(np.sum(weights * residual * residual))
    return {"count": int(pred.size), "rmse": float(np.sqrt(np.mean(residual * residual))), "max_abs": float(np.max(np.abs(residual), initial=0.0)), "chi2": chi2, "pass": bool(np.sqrt(np.mean(residual * residual)) <= 1.0 if sigma is None else chi2 <= pred.size)}


def propagate_derivatives(values: Sequence[float], jacobian: np.ndarray, covariance: np.ndarray) -> dict[str, object]:
    vector = np.asarray(values, dtype=float)
    jac = np.asarray(jacobian, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if jac.shape[1] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        raise ValueError("Jacobian and covariance dimensions are incompatible")
    output_cov = jac @ cov @ jac.T
    return {"values": vector.tolist(), "covariance": output_cov.tolist(), "sigma": np.sqrt(np.maximum(np.diag(output_cov), 0.0)).tolist()}


def sparse_laplacian_1d(size: int, spacing: float, *, periodic: bool = True):
    try:
        from scipy.sparse import diags
        matrix = diags(([-np.ones(size - 1), 2 * np.ones(size), -np.ones(size - 1)]), (-1, 0, 1), shape=(size, size), format="lil") / spacing**2
        if periodic:
            matrix[0, -1] = matrix[-1, 0] = -1.0 / spacing**2
        return matrix.tocsr()
    except ImportError:
        matrix = np.diag(np.full(size, 2.0 / spacing**2)) + np.diag(np.full(size - 1, -1.0 / spacing**2), 1) + np.diag(np.full(size - 1, -1.0 / spacing**2), -1)
        if periodic:
            matrix[0, -1] = matrix[-1, 0] = -1.0 / spacing**2
        return matrix


def solve_sparse_levels(operator: object, levels: int = 12) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.sparse.linalg import eigsh
        values, vectors = eigsh(operator, k=min(int(levels), operator.shape[0] - 1), which="SA")
    except (ImportError, ValueError, TypeError):
        values, vectors = np.linalg.eigh(np.asarray(operator.toarray() if hasattr(operator, "toarray") else operator))
        values, vectors = values[:levels], vectors[:, :levels]
    order = np.argsort(values)
    return np.asarray(values)[order], np.asarray(vectors)[:, order]


def persistent_parallel_map(function: Callable[[object], object], items: Sequence[object], workers: int = 2) -> tuple[object, ...]:
    values = tuple(items)
    if len(values) <= 1 or workers <= 1:
        return tuple(function(item) for item in values)
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        return tuple(pool.map(function, values))


@dataclass(frozen=True)
class PolynomialSurface:
    coefficients: np.ndarray
    degree: int
    domain: tuple[float, float]

    def __call__(self, coordinate: object) -> np.ndarray:
        scaled = 2 * (np.asarray(coordinate, dtype=float) - self.domain[0]) / (self.domain[1] - self.domain[0]) - 1
        return np.polynomial.chebyshev.chebval(scaled, self.coefficients)


def fit_chebyshev(coordinates: Sequence[float], values: Sequence[float], degree: int = 8) -> PolynomialSurface:
    q = np.asarray(coordinates, dtype=float)
    if q.ndim != 1 or q.size < degree + 1:
        raise ValueError("insufficient points for requested Chebyshev degree")
    domain = (float(np.min(q)), float(np.max(q)))
    scaled = 2 * (q - domain[0]) / (domain[1] - domain[0]) - 1
    return PolynomialSurface(np.polynomial.chebyshev.chebfit(scaled, values, degree), degree, domain)


def detect_degenerate_coordinates(coordinates: np.ndarray, *, tolerance: float = 1.0e-8) -> dict[str, object]:
    values = np.asarray(coordinates, dtype=float)
    spans = np.ptp(values, axis=0) if values.ndim == 2 else np.ptp(values)
    degenerate = np.flatnonzero(np.asarray(spans) <= tolerance).tolist()
    return {"degenerate_indices": degenerate, "spans": np.asarray(spans).tolist(), "valid": not degenerate, "tolerance": tolerance}


def track_vibrational_states(vectors: np.ndarray) -> dict[str, object]:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 3:
        raise ValueError("vectors must have shape (scan_points, basis, states)")
    permutations: list[list[int]] = [list(range(values.shape[2]))]
    overlaps: list[list[float]] = []
    for index in range(1, values.shape[0]):
        overlap = np.abs(values[index - 1].T @ values[index])
        assignment = []
        available = set(range(values.shape[2]))
        for row in np.argsort(-np.max(overlap, axis=1)):
            choices = sorted(available, key=lambda col: overlap[row, col], reverse=True)
            choice = choices[0]
            available.remove(choice)
            assignment.append(choice)
        permutations.append(assignment)
        overlaps.append([float(overlap[state, assignment[state]]) for state in range(values.shape[2])])
    return {"permutations": permutations, "overlaps": overlaps, "minimum_overlap": float(min((min(item) for item in overlaps), default=1.0))}


def fit_wms_rot(parameters: np.ndarray, observed: Sequence[float], design: np.ndarray, sigma: Sequence[float] | None = None) -> dict[str, object]:
    matrix, target = np.asarray(design, dtype=float), np.asarray(observed, dtype=float)
    weights = np.eye(target.size) if sigma is None else np.diag(1.0 / np.maximum(np.asarray(sigma, dtype=float), 1.0e-15) ** 2)
    covariance = np.linalg.pinv(matrix.T @ weights @ matrix)
    fitted = covariance @ matrix.T @ weights @ target
    residual = matrix @ fitted - target
    return {"parameters": fitted.tolist(), "covariance": covariance.tolist(), "sigma": np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist(), "rmse": float(np.sqrt(np.mean(residual * residual))), "residuals": residual.tolist()}


def reproducibility_manifest(outdir: Path, *, inputs: Sequence[Path], command: Sequence[str] = ()) -> Path:
    def digest(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    payload = {"schema": "matrix.tunneling.reproducibility.v1", "python": sys.version, "platform": platform.platform(), "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(), "inputs": {str(path): digest(path) for path in inputs}, "command": list(command), "timestamp_utc": time.time()}
    path = outdir / "reproducibility_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["PolynomialSurface", "detect_degenerate_coordinates", "fit_chebyshev", "fit_wms_rot", "persistent_parallel_map", "propagate_derivatives", "reproducibility_manifest", "solve_sparse_levels", "sparse_laplacian_1d", "track_vibrational_states", "validate_experimental_reference"]
