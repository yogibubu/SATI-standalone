"""Symmetry-equivalent two-stretch harmonic fit with an explicit cross term."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from matrix_qm import hessian_input_from_engine
from matrix_gf.harmonic_field import transform_cartesian_hessian

from .diagonal_harmonic import (
    _electrostatic_hessian,
    _exppe_hessian,
    _normal_modes,
)
@dataclass(frozen=True)
class StretchCoupledHarmonicFitRequest:
    hessian_path: Path
    primitive_manifest_path: Path
    cm5_path: Path
    zaff_seed_path: Path
    output_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str
    engine: str = "gdv"


def fit_stretch_coupled_harmonic_zaff(request: StretchCoupledHarmonicFitRequest) -> dict:
    """Fit both equal local stretches, their cross curvature, and the bend."""

    protocol = dict(request.protocol_manifest)
    if protocol.get("protocol_id") != "zaff-stretch-coupled-harmonic-fit.v1" or protocol.get("status") != "approved":
        raise ValueError("KEYMAKER requires the approved stretch-coupled harmonic protocol")
    if len(request.protocol_sha256) != 64:
        raise ValueError("KEYMAKER requires the installed protocol artifact hash")
    reference = hessian_input_from_engine(request.engine, request.hessian_path)
    numbers = tuple(int(value) for value in reference.atomic_numbers)
    if len(numbers) != 3 or numbers[0] != numbers[2] or numbers[0] == numbers[1]:
        raise ValueError("stretch-coupled v1 requires a symmetric X–Y–X triatomic")
    manifest = json.loads(Path(request.primitive_manifest_path).read_text())
    coordinates = manifest.get("coordinates", {})
    if coordinates != {"R": [[1, 2], [2, 3]], "A": [[1, 2, 3]], "SOOP": []}:
        raise ValueError("stretch-coupled v1 requires R(1,2), R(2,3), A(1,2,3)")
    xyz_bohr = np.asarray(reference.cartesian_coordinates_bohr, dtype=float)
    if abs(np.linalg.norm(xyz_bohr[0] - xyz_bohr[1]) - np.linalg.norm(xyz_bohr[2] - xyz_bohr[1])) > 2.0e-4:
        raise ValueError("the two reference X–Y bond lengths are not symmetry equivalent")
    charges = np.asarray(json.loads(Path(request.cm5_path).read_text())["raw_cm5_charges_e"], dtype=float)
    if charges.shape != (3,) or not np.all(np.isfinite(charges)):
        raise ValueError("the CM5 source must contain three finite charges")
    charges -= (np.sum(charges) - float(round(np.sum(charges)))) / 3.0
    if abs(charges[0] - charges[2]) > 1.0e-4:
        raise ValueError("the two terminal CM5 charges are not symmetry equivalent")
    seed_payload = json.loads(Path(request.zaff_seed_path).read_text())
    seed = seed_payload.get("metadata", {}).get("diagonal_seed_model", seed_payload)
    nonbonded = _electrostatic_hessian(xyz_bohr, charges, reference.atomic_numbers)
    nonbonded += _exppe_hessian(xyz_bohr, seed, coordinates["R"])
    transformed = transform_cartesian_hessian(
        reference.cartesian_hessian - nonbonded, xyz_bohr,
        reference.masses_amu, manifest,
    )
    b = transformed.b_matrix
    primitive_hessian = transformed.redundant_hessian.force_constants
    basis = transformed.redundant_hessian.vibrational_basis
    target, reference_modes = _normal_modes(
        reference.cartesian_hessian, reference.masses_amu, basis,
    )
    initial = np.asarray([
        0.5 * (primitive_hessian[0, 0] + primitive_hessian[1, 1]),
        primitive_hessian[2, 2], primitive_hessian[0, 1],
    ])

    def model(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stretch, bend, coupling = parameters
        hessian = b.T @ np.diag([stretch, stretch, bend]) @ b + nonbonded
        hessian += coupling * (np.outer(b[0], b[1]) + np.outer(b[1], b[0]))
        return _normal_modes(hessian, reference.masses_amu, basis)

    solution = least_squares(
        lambda parameters: model(parameters)[0] - target,
        initial, max_nfev=5000, xtol=1.0e-13, ftol=1.0e-13, gtol=1.0e-13,
    )
    frequencies, modes = model(solution.x)
    errors = frequencies - target
    overlap_matrix = np.abs(reference_modes.T @ modes)
    rows, columns = linear_sum_assignment(-overlap_matrix)
    assigned = overlap_matrix[rows, columns]
    rms = float(np.sqrt(np.mean(errors**2)))
    overlap = float(np.mean(assigned))
    accepted = bool(solution.success and np.all(frequencies > 0.0) and rms <= 10.0 and overlap >= 0.80)
    payload = {
        "schema": "matrix.architect.zaff_stretch_coupled_harmonic_fit.v1",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "protocol": {"id": protocol["protocol_id"], "sha256": request.protocol_sha256},
        "inputs": {
            "hessian": str(request.hessian_path),
            "primitive_manifest": str(request.primitive_manifest_path),
            "cm5": str(request.cm5_path),
            "zaff_seed": str(request.zaff_seed_path),
        },
        "primitive_labels": list(transformed.primitive_labels),
        "primitive_rank": transformed.redundant_hessian.primitive_rank,
        "excluded_modes": [],
        "reference_frequencies_cm-1": target.tolist(),
        "model_frequencies_cm-1": frequencies.tolist(),
        "frequency_errors_cm-1": errors.tolist(),
        "frequency_rms_error_cm-1": rms,
        "assigned_mode_overlaps": assigned.tolist(),
        "mean_assigned_mode_overlap": overlap,
        "parameters_hartree_per_internal_unit_squared": {
            "equivalent_stretch_diagonal": float(solution.x[0]),
            "bend_diagonal": float(solution.x[1]),
            "stretch_stretch_off_diagonal": float(solution.x[2]),
        },
        "solver_success": bool(solution.success),
        "accepted": accepted,
    }
    Path(request.output_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
