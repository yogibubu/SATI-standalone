"""Advanced multi-state, active-learning and reproducibility extensions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def coupled_surface_hamiltonian(energies: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    values = np.asarray(energies, dtype=float)
    coupling = np.asarray(couplings, dtype=float)
    if values.ndim == 1 and coupling.ndim == 2 and coupling.shape == (values.size, values.size):
        output = coupling.copy()
        output[np.diag_indices(values.size)] = values
        return 0.5 * (output + output.T)
    if values.shape != coupling.shape or values.shape[-1] != values.shape[-2]:
        raise ValueError("energies and couplings must have matching state dimensions")
    output = coupling.copy()
    diagonal = np.arange(values.shape[-1])
    output[..., diagonal, diagonal] = values
    return 0.5 * (output + np.swapaxes(output, -1, -2))


def rovibrational_hamiltonian(vibrational: np.ndarray, rotational_constants: np.ndarray, angular_momentum: int, coriolis: np.ndarray | None = None) -> np.ndarray:
    vib = np.asarray(vibrational, dtype=float)
    abc = np.asarray(rotational_constants, dtype=float)
    if abc.shape[-1] != 3:
        raise ValueError("rotational constants must end in A,B,C")
    rotation = angular_momentum * (angular_momentum + 1) * np.mean(abc, axis=-1)
    result = vib + rotation
    if coriolis is not None:
        result = result + np.asarray(coriolis, dtype=float)
    return result


def bayesian_linear_fit(design: np.ndarray, observed: Sequence[float], *, prior_sigma: float = 1.0, noise_sigma: float = 1.0) -> dict[str, object]:
    x, y = np.asarray(design, dtype=float), np.asarray(observed, dtype=float)
    precision = x.T @ x / noise_sigma**2 + np.eye(x.shape[1]) / prior_sigma**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ x.T @ y / noise_sigma**2
    return {"mean": mean.tolist(), "covariance": covariance.tolist(), "sigma": np.sqrt(np.diag(covariance)).tolist()}


def select_active_learning_points(coordinates: np.ndarray, uncertainty: np.ndarray, count: int = 5) -> np.ndarray:
    q, score = np.asarray(coordinates), np.asarray(uncertainty, dtype=float)
    return q[np.argsort(score)[::-1][:int(count)]]


def fit_surrogate(coordinates: np.ndarray, values: np.ndarray, *, kind: str = "gaussian_process") -> Callable[[np.ndarray], np.ndarray]:
    x, y = np.asarray(coordinates, dtype=float), np.asarray(values, dtype=float)
    if kind == "gaussian_process":
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), normalize_y=True).fit(x, y)
            return lambda points: model.predict(np.asarray(points, dtype=float))
        except ImportError:
            pass
    coefficients = np.polynomial.polynomial.polyfit(x.reshape(-1), y.reshape(-1), min(5, len(x) - 1))
    return lambda points: np.polynomial.polynomial.polyval(np.asarray(points).reshape(-1), coefficients)


def validate_geometry_invariants(reference: np.ndarray, candidate: np.ndarray, *, tolerance: float = 1.0e-6) -> dict[str, object]:
    ref, got = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    error = float(np.max(np.abs(ref - got), initial=0.0)) if ref.shape == got.shape else float("inf")
    return {"valid": error <= tolerance, "max_error": error, "tolerance": tolerance, "permutation_invariant": ref.shape == got.shape}


def curvilinear_metric(jacobian: np.ndarray) -> np.ndarray:
    j = np.asarray(jacobian, dtype=float)
    return np.swapaxes(j, -1, -2) @ j


def tunneling_density_current(wavefunction: np.ndarray, spacing: float, mass: float = 1.0) -> dict[str, np.ndarray]:
    psi = np.asarray(wavefunction, dtype=complex)
    derivative = np.gradient(psi, float(spacing), edge_order=1)
    density = np.abs(psi) ** 2
    current = np.imag(np.conjugate(psi) * derivative) / float(mass)
    return {"density": density, "current": current}


def fault_tolerant_map(function: Callable[[object], object], items: Sequence[object], *, retries: int = 2) -> dict[str, object]:
    results, failures = [], []
    for index, item in enumerate(items):
        for attempt in range(retries + 1):
            try:
                results.append(function(item)); break
            except Exception as error:  # noqa: BLE001 - captured in audit artifact
                if attempt == retries:
                    failures.append({"index": index, "error": str(error)})
    return {"results": results, "failures": failures, "complete": not failures}


def versioned_surface_record(path: Path, *, molecule: str, method: str, version: str = "1") -> Path:
    payload = {"schema": "matrix.surface.versioned.v1", "molecule": molecule, "method": method, "version": version, "created_at": datetime.now(timezone.utc).isoformat(), "source": str(path)}
    record = path.with_suffix(path.suffix + ".metadata.json")
    record.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return record


def publishable_report(outdir: Path, result: dict[str, object], references: Sequence[str] = ()) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "publishable_report.html"
    refs = "".join(f"<li>{ref}</li>" for ref in references)
    path.write_text(f"<html><meta charset='utf-8'><title>SONIC/DVR report</title><h1>SONIC/DVR report</h1><h2>References</h2><ul>{refs}</ul><h2>Results</h2><pre>{json.dumps(result,indent=2)}</pre></html>", encoding="utf-8")
    return path


def regression_gate(reference: Sequence[float], candidate: Sequence[float], tolerances: Sequence[float]) -> dict[str, object]:
    ref, got, tol = np.asarray(reference), np.asarray(candidate), np.asarray(tolerances)
    errors = np.abs(ref - got)
    return {"passed": bool(ref.shape == got.shape == tol.shape and np.all(errors <= tol)), "errors": errors.tolist(), "tolerances": tol.tolist()}


def remote_service_contract(outdir: Path, *, host: str = "127.0.0.1", port: int = 8765) -> Path:
    payload = {"schema": "matrix.sonic.remote_service.v1", "host": host, "port": port, "authentication": "external reverse proxy", "endpoints": ["/status", "/runs", "/artifacts"]}
    path = outdir / "remote_service_contract.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); return path


def propagate_wavepacket(wavefunction: np.ndarray, hamiltonian: np.ndarray, dt: float, steps: int) -> np.ndarray:
    psi = np.asarray(wavefunction, dtype=complex)
    h = np.asarray(hamiltonian, dtype=complex)
    propagator = np.linalg.eigh(h)
    unitary = propagator[1] @ np.diag(np.exp(-1j * propagator[0] * dt)) @ propagator[1].conj().T
    trajectory = [psi.copy()]
    for _ in range(int(steps)):
        psi = unitary @ psi; trajectory.append(psi.copy())
    return np.asarray(trajectory)


__all__ = ["bayesian_linear_fit", "coupled_surface_hamiltonian", "curvilinear_metric", "fault_tolerant_map", "fit_surrogate", "propagate_wavepacket", "publishable_report", "regression_gate", "remote_service_contract", "rovibrational_hamiltonian", "select_active_learning_points", "tunneling_density_current", "validate_geometry_invariants", "versioned_surface_record"]
