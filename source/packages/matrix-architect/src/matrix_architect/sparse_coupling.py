"""Forward sparse-coupling refinement for a diagonal ZAFF field."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from matrix_qm import HESSIAN_EIGENVALUE_TO_CM, hessian_input_from_engine
from matrix_gf.harmonic_field import transform_cartesian_hessian

from .diagonal_harmonic import (
    _atom_equivalence_classes, _electrostatic_hessian, _exppe_hessian,
    _normal_modes, _primitive_equivalence_groups,
    primitive_b_matrix_from_manifest,
)


@dataclass(frozen=True)
class SparseCouplingFitRequest:
    hessian_path: Path
    primitive_manifest_path: Path
    cm5_path: Path
    zaff_seed_path: Path
    diagonal_fit_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str
    output_path: Path
    engine: str = "gdv"
    maximum_couplings: int = 20
    frequency_rms_target_cm: float = 5.0
    mean_overlap_target: float = 0.85


def _coupling_groups(equivalence_groups: tuple[tuple[int, ...], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    groups: list[tuple[tuple[int, int], ...]] = []
    for left, left_members in enumerate(equivalence_groups):
        for right in range(left, len(equivalence_groups)):
            right_members = equivalence_groups[right]
            pairs = tuple(
                (min(i, j), max(i, j))
                for i in left_members for j in right_members if i != j
            )
            pairs = tuple(sorted(set(pairs)))
            if pairs:
                groups.append(pairs)
    return tuple(groups)


def fit_sparse_zaff_couplings(request: SparseCouplingFitRequest) -> dict:
    protocol = dict(request.protocol_manifest)
    if protocol.get("protocol_id") != "zaff-sparse-coupling-fit.v1" or protocol.get("status") != "approved":
        raise ValueError("Keymaker requires the approved sparse-coupling protocol")
    if len(request.protocol_sha256) != 64:
        raise ValueError("Keymaker requires the installed protocol artifact hash")
    reference = hessian_input_from_engine(request.engine, request.hessian_path)
    manifest = json.loads(Path(request.primitive_manifest_path).read_text())
    diagonal_record = json.loads(Path(request.diagonal_fit_path).read_text())
    b_matrix, labels = primitive_b_matrix_from_manifest(reference.cartesian_coordinates_bohr, manifest)
    atom_classes = _atom_equivalence_classes(reference.atomic_numbers, reference.cartesian_coordinates_bohr)
    expansion, equivalence_groups = _primitive_equivalence_groups(manifest, atom_classes)
    diagonal_full = np.asarray(diagonal_record["fitted_diagonal_force_constants"], dtype=float)
    diagonal_shared = np.asarray([np.mean(diagonal_full[list(group)]) for group in equivalence_groups])

    charges_payload = json.loads(Path(request.cm5_path).read_text())
    charges = np.asarray(charges_payload["raw_cm5_charges_e"], dtype=float)
    charges -= (np.sum(charges) - float(round(np.sum(charges)))) / len(charges)
    seed_payload = json.loads(Path(request.zaff_seed_path).read_text())
    seed = seed_payload.get("metadata", {}).get("diagonal_seed_model", seed_payload)
    nonbonded = _electrostatic_hessian(reference.cartesian_coordinates_bohr, charges, reference.atomic_numbers)
    nonbonded += _exppe_hessian(reference.cartesian_coordinates_bohr, seed, manifest["coordinates"]["R"])
    transformed = transform_cartesian_hessian(
        reference.cartesian_hessian - nonbonded,
        reference.cartesian_coordinates_bohr,
        reference.masses_amu,
        manifest,
    ).redundant_hessian
    reference_frequencies, reference_modes = _normal_modes(
        reference.cartesian_hessian, reference.masses_amu, transformed.vibrational_basis
    )
    ordinary = np.arange(1, len(reference_frequencies))
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(reference.masses_amu), 3)

    def normal_block(cartesian_hessian: np.ndarray) -> np.ndarray:
        weighted = inverse_sqrt_mass[:, None] * cartesian_hessian * inverse_sqrt_mass[None, :]
        return reference_modes[:, ordinary].T @ weighted @ reference_modes[:, ordinary]

    base_block = normal_block(nonbonded)
    diagonal_blocks = tuple(normal_block(b_matrix.T @ np.diag(expansion[:, index]) @ b_matrix) for index in range(expansion.shape[1]))
    candidate_groups = _coupling_groups(equivalence_groups)
    candidate_blocks = []
    for group in candidate_groups:
        hessian = np.zeros_like(reference.cartesian_hessian)
        for left, right in group:
            hessian += np.outer(b_matrix[left], b_matrix[right]) + np.outer(b_matrix[right], b_matrix[left])
        candidate_blocks.append(normal_block(hessian))

    target_curvatures = np.sign(reference_frequencies[ordinary]) * (np.abs(reference_frequencies[ordinary]) / HESSIAN_EIGENVALUE_TO_CM) ** 2
    gaps = np.abs(target_curvatures[:, None] - target_curvatures[None, :])
    gap_floor = (25.0 / HESSIAN_EIGENVALUE_TO_CM) ** 2
    gaps = np.maximum(gaps, gap_floor)
    off = np.triu_indices(len(ordinary), 1)
    selected: list[int] = []
    coupling_values = np.empty(0)

    def cartesian_model(diagonal: np.ndarray, couplings: np.ndarray) -> np.ndarray:
        hessian = b_matrix.T @ np.diag(expansion @ diagonal) @ b_matrix + nonbonded
        for value, candidate in zip(couplings, selected):
            for left, right in candidate_groups[candidate]:
                hessian += value * (np.outer(b_matrix[left], b_matrix[right]) + np.outer(b_matrix[right], b_matrix[left]))
        return hessian

    def metrics(diagonal: np.ndarray, couplings: np.ndarray) -> dict:
        frequencies, modes = _normal_modes(cartesian_model(diagonal, couplings), reference.masses_amu, transformed.vibrational_basis)
        errors = frequencies[ordinary] - reference_frequencies[ordinary]
        overlaps = np.abs(reference_modes[:, ordinary].T @ modes[:, ordinary])
        rows, columns = linear_sum_assignment(-overlaps)
        assigned = overlaps[rows, columns]
        return {
            "frequency_rms_error_cm-1": float(np.sqrt(np.mean(errors**2))),
            "frequency_mean_absolute_error_cm-1": float(np.mean(np.abs(errors))),
            "frequency_maximum_absolute_error_cm-1": float(np.max(np.abs(errors))),
            "mean_assigned_mode_overlap": float(np.mean(assigned)),
            "minimum_assigned_mode_overlap": float(np.min(assigned)),
        }

    history = [{"iteration": 0, "coupling_count": 0, **metrics(diagonal_shared, coupling_values)}]
    for iteration in range(1, int(request.maximum_couplings) + 1):
        current_block = base_block.copy()
        for value, block in zip(diagonal_shared, diagonal_blocks):
            current_block += value * block
        for value, candidate in zip(coupling_values, selected):
            current_block += value * candidate_blocks[candidate]
        residual = -current_block[off] / gaps[off]
        scores = []
        for index, block in enumerate(candidate_blocks):
            if index in selected:
                scores.append(-np.inf)
                continue
            vector = block[off] / gaps[off]
            scores.append(abs(float(np.dot(vector, residual))) / max(float(np.linalg.norm(vector)), 1.0e-30))
        selected.append(int(np.argmax(scores)))
        start = np.concatenate((diagonal_shared, coupling_values, np.zeros(1)))

        def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return values[: len(diagonal_shared)], values[len(diagonal_shared):]

        def mixing_objective(values: np.ndarray) -> float:
            diagonal, couplings = unpack(values)
            block = base_block.copy()
            for value, basis in zip(diagonal, diagonal_blocks):
                block += value * basis
            for value, candidate in zip(couplings, selected):
                block += value * candidate_blocks[candidate]
            mixing = block[off] / gaps[off]
            diagonal_drift = (diagonal - diagonal_shared) / np.maximum(np.abs(diagonal_shared), 1.0e-5)
            return float(np.mean(mixing**2) + 1.0e-5 * np.mean(diagonal_drift**2))

        def frequency_constraint(values: np.ndarray) -> float:
            diagonal, couplings = unpack(values)
            return float(request.frequency_rms_target_cm - metrics(diagonal, couplings)["frequency_rms_error_cm-1"])

        solution = minimize(
            mixing_objective, start, method="SLSQP",
            constraints=({"type": "ineq", "fun": frequency_constraint},),
            options={"maxiter": 600, "ftol": 1.0e-12},
        )
        diagonal_shared, coupling_values = unpack(solution.x)
        point = {
            "iteration": iteration,
            "coupling_count": len(selected),
            "added_coupling": [[labels[left], labels[right]] for left, right in candidate_groups[selected[-1]]],
            "solver_success": bool(solution.success),
            "solver_message": str(solution.message),
            **metrics(diagonal_shared, coupling_values),
        }
        history.append(point)
        if (
            point["frequency_rms_error_cm-1"] <= request.frequency_rms_target_cm
            and point["mean_assigned_mode_overlap"] >= request.mean_overlap_target
        ):
            break

    final = history[-1]
    best_sparse = max(
        (point for point in history if point["frequency_rms_error_cm-1"] <= request.frequency_rms_target_cm),
        key=lambda point: point["mean_assigned_mode_overlap"],
    )
    payload = {
        "schema": "matrix.architect.zaff_sparse_coupling_fit.v1",
        "status": "TARGET_REACHED" if final["frequency_rms_error_cm-1"] <= request.frequency_rms_target_cm and final["mean_assigned_mode_overlap"] >= request.mean_overlap_target else "LIMIT_REACHED",
        "protocol": {"id": protocol["protocol_id"], "sha256": request.protocol_sha256},
        "targets": {"frequency_rms_error_cm-1": request.frequency_rms_target_cm, "mean_assigned_mode_overlap": request.mean_overlap_target, "maximum_couplings": request.maximum_couplings},
        "primitive_labels": list(labels),
        "fitted_diagonal_force_constants": (expansion @ diagonal_shared).tolist(),
        "selected_couplings": [
            {"primitive_pairs": [[labels[left], labels[right]] for left, right in candidate_groups[candidate]], "force_constant": float(value)}
            for candidate, value in zip(selected, coupling_values)
        ],
        "history": history,
        "final_metrics": final,
        "best_sparse_metrics": best_sparse,
        "promotion_recommended": bool(best_sparse["mean_assigned_mode_overlap"] >= request.mean_overlap_target),
    }
    Path(request.output_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
