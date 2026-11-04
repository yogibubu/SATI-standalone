"""Keymaker-owned diagonal harmonic ZAFF fitting.

This module implements the approved ``zaff-diagonal-harmonic-fit.v1``
protocol. Coordinate transformation remains owned by GF and SMITH; ARCHITECT
only builds the declared model and optimizes its diagonal curvatures.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment, minimize

from matrix_chem import read_xyzin_geometry
from matrix_chem.topology.elements import atomic_number
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM, hessian_input_from_engine
from matrix_gf.harmonic_field import transform_cartesian_hessian
from matrix_smith import primitive_b_matrix_from_manifest
from matrix_zaff.nonbonded import electrostatic_energy_gradient, electrostatic_hessian_vector_product
from matrix_zaff.radial import exppe_from_minimum, mie_minimum_derivatives
from matrix_zaff.seed_runtime import BOHR_TO_ANGSTROM, _seed_nonbonded_pairs


@dataclass(frozen=True)
class DiagonalHarmonicFitRequest:
    hessian_path: Path
    primitive_manifest_path: Path
    cm5_path: Path
    zaff_seed_path: Path
    output_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str
    stationarity_geometry_path: Path | None = None
    pareto_weights_path: Path | None = None
    engine: str = "gdv"
    maximum_frequency_error_cm: float = 10.0
    excluded_mode_indices: tuple[int, ...] = ()


def _angle_value(xyz: np.ndarray, i: int, j: int, k: int) -> float:
    left, right = xyz[i] - xyz[j], xyz[k] - xyz[j]
    return float(np.arccos(np.clip(np.dot(left, right) / np.linalg.norm(left) / np.linalg.norm(right), -1.0, 1.0)))


def _electrostatic_hessian(xyz_bohr: np.ndarray, charges: np.ndarray, atomic_numbers: np.ndarray) -> np.ndarray:
    size = xyz_bohr.size
    result = np.zeros((size, size))
    for column in range(size):
        direction = np.zeros(size); direction[column] = 1.0
        result[:, column] = electrostatic_hessian_vector_product(
            xyz_bohr, charges, direction, (), atomic_numbers=tuple(int(v) for v in atomic_numbers),
            electrostatic_model="gaussian_erf_all_pairs", backend="direct",
        )
    return 0.5 * (result + result.T)


def _exppe_hessian(xyz_bohr: np.ndarray, model: dict, bonds_one_based: list) -> np.ndarray:
    xyz_angstrom = xyz_bohr * BOHR_TO_ANGSTROM
    parameters = model["nonbonded_atoms"]
    pairs = _seed_nonbonded_pairs(xyz_angstrom, xyz_bohr, "direct_pair_list", float(model["nonbonded"]["short_range_cutoff_angstrom"]), None, native=False)
    bonded = {tuple(sorted((int(i)-1, int(j)-1))) for i, j in bonds_one_based}
    matrix_angstrom = np.zeros((xyz_bohr.size, xyz_bohr.size))
    for left, right in pairs:
        if tuple(sorted((int(left), int(right)))) in bonded:
            continue
        pleft, pright = parameters[left], parameters[right]
        epsilon = np.sqrt(float(pleft["epsilon_hartree"]) * float(pright["epsilon_hartree"]))
        rmin = float(pleft["rmin_half_angstrom"]) + float(pright["rmin_half_angstrom"])
        distance = float(np.linalg.norm(xyz_angstrom[left] - xyz_angstrom[right]))
        radial = exppe_from_minimum(mie_minimum_derivatives(epsilon, rmin, 12.0, 6.0)).derivatives(distance)
        first = radial.first
        second = radial.second
        unit = (xyz_angstrom[left] - xyz_angstrom[right]) / distance
        block = (second - first / distance) * np.outer(unit, unit) + first / distance * np.eye(3)
        sl, sr = slice(3*left, 3*left+3), slice(3*right, 3*right+3)
        matrix_angstrom[sl, sl] += block; matrix_angstrom[sr, sr] += block
        matrix_angstrom[sl, sr] -= block; matrix_angstrom[sr, sl] -= block
    return matrix_angstrom * BOHR_TO_ANGSTROM**2


def _exppe_gradient(xyz_bohr: np.ndarray, model: dict, bonds_one_based: list) -> np.ndarray:
    xyz_angstrom = xyz_bohr * BOHR_TO_ANGSTROM
    parameters = model["nonbonded_atoms"]
    pairs = _seed_nonbonded_pairs(xyz_angstrom, xyz_bohr, "direct_pair_list", float(model["nonbonded"]["short_range_cutoff_angstrom"]), None, native=False)
    bonded = {tuple(sorted((int(i)-1, int(j)-1))) for i, j in bonds_one_based}
    gradient_angstrom = np.zeros_like(xyz_angstrom)
    for left, right in pairs:
        if tuple(sorted((int(left), int(right)))) in bonded:
            continue
        pleft, pright = parameters[left], parameters[right]
        epsilon = np.sqrt(float(pleft["epsilon_hartree"]) * float(pright["epsilon_hartree"]))
        rmin = float(pleft["rmin_half_angstrom"]) + float(pright["rmin_half_angstrom"])
        delta = xyz_angstrom[left] - xyz_angstrom[right]
        distance = float(np.linalg.norm(delta))
        radial = exppe_from_minimum(mie_minimum_derivatives(epsilon, rmin, 12.0, 6.0)).derivatives(distance)
        first = radial.first
        vector = first * delta / distance
        gradient_angstrom[left] += vector
        gradient_angstrom[right] -= vector
    return gradient_angstrom.reshape(-1) * BOHR_TO_ANGSTROM


def _primitive_values(xyz_bohr: np.ndarray, manifest: dict) -> np.ndarray:
    values: list[float] = []
    for raw_i, raw_j in manifest["coordinates"]["R"]:
        values.append(float(np.linalg.norm(xyz_bohr[int(raw_i)-1] - xyz_bohr[int(raw_j)-1])))
    for raw_i, raw_j, raw_k in manifest["coordinates"]["A"]:
        values.append(_angle_value(xyz_bohr, int(raw_i)-1, int(raw_j)-1, int(raw_k)-1))
    # Frozen SOOP coordinates are defined at their own reference geometry.
    values.extend(0.0 for _ in manifest["coordinates"]["SOOP"])
    return np.asarray(values)


def _frequencies(hessian: np.ndarray, masses: np.ndarray, basis: np.ndarray) -> np.ndarray:
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    weighted = inverse_sqrt_mass[:, None] * hessian * inverse_sqrt_mass[None, :]
    eigenvalues = np.linalg.eigvalsh(basis.T @ (0.5 * (weighted + weighted.T)) @ basis)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * HESSIAN_EIGENVALUE_TO_CM


def _normal_modes(hessian: np.ndarray, masses: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    weighted = inverse_sqrt_mass[:, None] * hessian * inverse_sqrt_mass[None, :]
    reduced = basis.T @ (0.5 * (weighted + weighted.T)) @ basis
    eigenvalues, reduced_modes = np.linalg.eigh(reduced)
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * HESSIAN_EIGENVALUE_TO_CM
    return frequencies, basis @ reduced_modes


def _atom_equivalence_classes(atomic_numbers: np.ndarray, xyz_bohr: np.ndarray, tolerance_bohr: float = 2.0e-4) -> tuple[int, ...]:
    """Infer exact-geometry atom orbits from element-resolved distance fingerprints."""
    numbers = np.asarray(atomic_numbers, dtype=int)
    xyz = np.asarray(xyz_bohr, dtype=float)
    fingerprints = []
    for atom in range(len(numbers)):
        entries = sorted((int(numbers[other]), float(np.linalg.norm(xyz[atom] - xyz[other]))) for other in range(len(numbers)) if other != atom)
        fingerprints.append(entries)
    classes: list[int] = []
    representatives: list[int] = []
    for atom, fingerprint in enumerate(fingerprints):
        assigned = None
        for class_id, representative in enumerate(representatives):
            other = fingerprints[representative]
            if numbers[atom] == numbers[representative] and all(a[0] == b[0] and abs(a[1] - b[1]) <= tolerance_bohr for a, b in zip(fingerprint, other)):
                assigned = class_id
                break
        if assigned is None:
            assigned = len(representatives)
            representatives.append(atom)
        classes.append(assigned)
    return tuple(classes)


def _primitive_equivalence_groups(manifest: dict, atom_classes: tuple[int, ...]) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    keys: list[tuple] = []
    for i, j in manifest["coordinates"]["R"]:
        keys.append(("R", *sorted((atom_classes[int(i)-1], atom_classes[int(j)-1]))))
    for i, j, k in manifest["coordinates"]["A"]:
        keys.append(("A", atom_classes[int(j)-1], *sorted((atom_classes[int(i)-1], atom_classes[int(k)-1]))))
    for record in manifest["coordinates"]["SOOP"]:
        if "unique_key" in record:
            keys.append(("SOOP", str(record["unique_key"])))
            continue
        center, moving = int(record["center"])-1, int(record["moving"])-1
        others = [int(value)-1 for value in record["plane"] if int(value)-1 != center]
        keys.append(("SOOP", atom_classes[center], atom_classes[moving], *sorted(atom_classes[value] for value in others)))
    for atoms in manifest["coordinates"].get("D", ()):
        keys.append(("D", tuple(int(atom) for atom in atoms)))
    unique: dict[tuple, int] = {}
    members: list[list[int]] = []
    group_index = np.empty(len(keys), dtype=int)
    for index, key in enumerate(keys):
        group = unique.get(key)
        if group is None:
            group = len(members); unique[key] = group; members.append([])
        group_index[index] = group; members[group].append(index)
    expansion = np.zeros((len(keys), len(members)))
    expansion[np.arange(len(keys)), group_index] = 1.0
    return expansion, tuple(tuple(group) for group in members)


def fit_diagonal_harmonic_zaff(request: DiagonalHarmonicFitRequest) -> dict:
    protocol = dict(request.protocol_manifest)
    if protocol.get("protocol_id") != "zaff-diagonal-harmonic-fit.v1" or protocol.get("status") != "approved":
        raise ValueError("Keymaker requires the approved diagonal harmonic fit protocol")
    if len(request.protocol_sha256) != 64:
        raise ValueError("Keymaker requires the installed protocol artifact hash")
    reference = hessian_input_from_engine(request.engine, request.hessian_path)
    manifest = json.loads(Path(request.primitive_manifest_path).read_text())
    b_matrix, labels = primitive_b_matrix_from_manifest(
        reference.cartesian_coordinates_bohr, manifest
    )
    charges_payload = json.loads(Path(request.cm5_path).read_text())
    charges = np.asarray(charges_payload["raw_cm5_charges_e"], dtype=float)
    charges -= (np.sum(charges) - float(round(np.sum(charges)))) / len(charges)
    seed_payload = json.loads(Path(request.zaff_seed_path).read_text())
    seed = seed_payload.get("metadata", {}).get("diagonal_seed_model", seed_payload)
    electrostatic = _electrostatic_hessian(reference.cartesian_coordinates_bohr, charges, reference.atomic_numbers)
    vdw = _exppe_hessian(reference.cartesian_coordinates_bohr, seed, manifest["coordinates"]["R"])
    nonbonded = electrostatic + vdw
    transformed = transform_cartesian_hessian(
        reference.cartesian_hessian - nonbonded,
        reference.cartesian_coordinates_bohr,
        reference.masses_amu,
        manifest,
    ).redundant_hessian
    transformed_diagonal = np.diag(transformed.force_constants).copy()
    atom_classes = _atom_equivalence_classes(reference.atomic_numbers, reference.cartesian_coordinates_bohr)
    expansion, equivalence_groups = _primitive_equivalence_groups(manifest, atom_classes)
    initial_shared = np.asarray([np.mean(transformed_diagonal[list(group)]) for group in equivalence_groups])
    initial = expansion @ initial_shared
    target, reference_modes = _normal_modes(reference.cartesian_hessian, reference.masses_amu, transformed.vibrational_basis)
    included = np.ones(len(target), dtype=bool)
    included[list(request.excluded_mode_indices)] = False
    scale = np.maximum(np.abs(initial_shared), 1.0e-5)

    def model_frequencies(shared_constants: np.ndarray) -> np.ndarray:
        constants = expansion @ shared_constants
        model_hessian = b_matrix.T @ np.diag(constants) @ b_matrix + nonbonded
        return _frequencies(model_hessian, reference.masses_amu, transformed.vibrational_basis)

    def spectral_residual(shared_constants: np.ndarray) -> np.ndarray:
        return model_frequencies(shared_constants)[included] - target[included]

    optimization = least_squares(spectral_residual, initial_shared, method="trf", max_nfev=2500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    best_shared = optimization.x
    best_sse = float(np.dot(spectral_residual(best_shared), spectral_residual(best_shared)))
    ordinary_indices = np.flatnonzero(included)

    def descriptors(shared_constants: np.ndarray) -> tuple[float, float, float]:
        constants = expansion @ shared_constants
        hessian = b_matrix.T @ np.diag(constants) @ b_matrix + nonbonded
        frequencies, modes = _normal_modes(hessian, reference.masses_amu, transformed.vibrational_basis)
        frequency_rms = float(np.sqrt(np.mean((frequencies[included] - target[included]) ** 2)))
        overlaps = np.abs(reference_modes[:, ordinary_indices].T @ modes[:, ordinary_indices])
        rows, columns = linear_sum_assignment(-overlaps)
        mean_overlap = float(np.mean(overlaps[rows, columns]))
        displacement = float(np.sqrt(np.mean(((shared_constants - initial_shared) / scale) ** 2)))
        return frequency_rms, mean_overlap, displacement

    default_profiles = (
        {"name": "frequency", "frequency": 1.0, "overlap": 0.0, "diagonal": 0.0},
        {"name": "balanced", "frequency": 1.0, "overlap": 1.0, "diagonal": 0.01},
        {"name": "transition_3", "frequency": 1.0, "overlap": 3.0, "diagonal": 0.01},
        {"name": "transition_4", "frequency": 1.0, "overlap": 4.0, "diagonal": 0.01},
        {"name": "transition_5", "frequency": 1.0, "overlap": 5.0, "diagonal": 0.01},
        {"name": "mode", "frequency": 1.0, "overlap": 10.0, "diagonal": 0.01},
        {"name": "mode_strong", "frequency": 1.0, "overlap": 30.0, "diagonal": 0.01},
        {"name": "mode_limit", "frequency": 1.0, "overlap": 100.0, "diagonal": 0.01},
    )
    profiles = default_profiles
    if request.pareto_weights_path is not None:
        loaded = json.loads(Path(request.pareto_weights_path).read_text())
        profiles = tuple(loaded["profiles"] if isinstance(loaded, dict) else loaded)
    pareto_candidates: list[dict] = []
    starts = (best_shared, initial_shared)
    for profile in profiles:
        wf, wo, wd = (float(profile[key]) for key in ("frequency", "overlap", "diagonal"))
        if min(wf, wo, wd) < 0.0 or wf + wo + wd <= 0.0:
            raise ValueError("Pareto descriptor weights must be non-negative and not all zero")

        def pareto_objective(values: np.ndarray) -> float:
            frequency_rms, mean_overlap, displacement = descriptors(values)
            return (
                wf * (frequency_rms / 10.0) ** 2
                + wo * ((1.0 - mean_overlap) / 0.20) ** 2
                + wd * displacement**2
            )

        solutions = [
            minimize(pareto_objective, start, method="L-BFGS-B", options={"maxiter": 250, "ftol": 1.0e-10})
            for start in starts
        ]
        solution = min(solutions, key=lambda item: float(item.fun))
        frequency_rms, mean_overlap, displacement = descriptors(solution.x)
        pareto_candidates.append({
            "name": str(profile.get("name", f"profile_{len(pareto_candidates)+1}")),
            "weights": {"frequency": wf, "overlap": wo, "diagonal": wd},
            "frequency_rms_error_cm-1": frequency_rms,
            "mean_assigned_mode_overlap": mean_overlap,
            "normalized_diagonal_displacement": displacement,
            "accepted": bool(frequency_rms <= 10.0 and mean_overlap >= 0.80),
            "solver_success": bool(solution.success),
            "solver_message": str(solution.message),
            "shared_constants": solution.x.tolist(),
        })
    for candidate in pareto_candidates:
        candidate["dominated"] = any(
            other is not candidate
            and other["frequency_rms_error_cm-1"] <= candidate["frequency_rms_error_cm-1"]
            and other["mean_assigned_mode_overlap"] >= candidate["mean_assigned_mode_overlap"]
            and other["normalized_diagonal_displacement"] <= candidate["normalized_diagonal_displacement"]
            and (
                other["frequency_rms_error_cm-1"] < candidate["frequency_rms_error_cm-1"]
                or other["mean_assigned_mode_overlap"] > candidate["mean_assigned_mode_overlap"]
                or other["normalized_diagonal_displacement"] < candidate["normalized_diagonal_displacement"]
            )
            for other in pareto_candidates
        )
    frontier = [candidate for candidate in pareto_candidates if not candidate["dominated"]]
    acceptable = [candidate for candidate in frontier if candidate["accepted"]]
    frequency_admissible = [candidate for candidate in frontier if candidate["frequency_rms_error_cm-1"] <= 10.0]
    if acceptable:
        selected = min(acceptable, key=lambda item: (item["normalized_diagonal_displacement"], item["frequency_rms_error_cm-1"]))
    elif frequency_admissible:
        # Parsimonious fallback: exploit the admitted frequency budget to
        # retain the largest mode overlap without adding couplings.
        selected = max(frequency_admissible, key=lambda item: (item["mean_assigned_mode_overlap"], item["frequency_rms_error_cm-1"]))
    else:
        selected = min(frontier, key=lambda item: (item["frequency_rms_error_cm-1"] / 10.0) ** 2 + (max(0.0, 0.80 - item["mean_assigned_mode_overlap"]) / 0.20) ** 2 + 0.01 * item["normalized_diagonal_displacement"] ** 2)
    selected_shared = np.asarray(selected["shared_constants"], dtype=float)
    fitted = expansion @ selected_shared
    model_hessian = b_matrix.T @ np.diag(fitted) @ b_matrix + nonbonded
    model, model_modes = _normal_modes(model_hessian, reference.masses_amu, transformed.vibrational_basis)
    errors = model - target
    ordinary_errors = errors[included]
    maximum = float(np.max(np.abs(ordinary_errors)))
    spectral_optimum_reached = bool(np.sqrt(np.mean(ordinary_errors**2)) <= 10.0)
    overlap_matrix = np.abs(reference_modes[:, ordinary_indices].T @ model_modes[:, ordinary_indices])
    assigned_reference, assigned_model = linear_sum_assignment(-overlap_matrix)
    assigned_overlaps = overlap_matrix[assigned_reference, assigned_model]
    payload = {
        "schema": "matrix.architect.zaff_diagonal_harmonic_fit.v1",
        "status": "COMPLETED",
        "protocol": {
            "id": protocol["protocol_id"],
            "sha256": request.protocol_sha256,
        },
        "inputs": {"hessian": str(request.hessian_path), "primitive_manifest": str(request.primitive_manifest_path), "cm5": str(request.cm5_path), "zaff_seed": str(request.zaff_seed_path)},
        "primitive_count": len(labels), "primitive_rank": transformed.primitive_rank,
        "primitive_labels": list(labels),
        "transformed_hessian_diagonal_force_constants": transformed_diagonal.tolist(),
        "symmetry_projected_initial_diagonal_force_constants": initial.tolist(),
        "fitted_diagonal_force_constants": fitted.tolist(),
        "reference_frequencies_cm-1": target.tolist(),
        "model_frequencies_cm-1": model.tolist(),
        "frequency_errors_cm-1": errors.tolist(),
        "excluded_large_amplitude_modes": list(request.excluded_mode_indices),
        "ordinary_mode_metrics": {"rms_error_cm-1": float(np.sqrt(np.mean(ordinary_errors**2))), "mean_absolute_error_cm-1": float(np.mean(np.abs(ordinary_errors))), "maximum_absolute_error_cm-1": maximum, "acceptance_rms_error_cm-1": 10.0, "accepted": bool(np.sqrt(np.mean(ordinary_errors**2)) <= 10.0)},
        "mode_overlap": {"assignment_reference_indices": ordinary_indices[assigned_reference].tolist(), "assignment_model_indices": ordinary_indices[assigned_model].tolist(), "matrix": overlap_matrix.tolist(), "assigned_overlaps": assigned_overlaps.tolist(), "mean_assigned_overlap": float(np.mean(assigned_overlaps)), "minimum_assigned_overlap": float(np.min(assigned_overlaps)), "acceptance_mean_overlap": 0.80, "accepted": bool(np.mean(assigned_overlaps) >= 0.80)},
        "symmetry_equivalence": {"atom_class_by_atom": list(atom_classes), "independent_parameter_count": len(equivalence_groups), "primitive_groups": [[labels[index] for index in group] for group in equivalence_groups]},
        "transformed_diagonal_displacement": {"normalized_rms": float(np.sqrt(np.mean(((selected_shared-initial_shared)/scale)**2))), "relative_l2_full_diagonal": float(np.linalg.norm(fitted-transformed_diagonal)/max(np.linalg.norm(transformed_diagonal), 1.0e-30))},
        "pareto": {"descriptor_scales": {"frequency_rms_cm-1": 10.0, "mode_overlap_deficit": 0.20, "diagonal_displacement": 1.0}, "profiles": [{key: value for key, value in candidate.items() if key != "shared_constants"} for candidate in pareto_candidates], "frontier_profiles": [candidate["name"] for candidate in frontier], "selected_profile": selected["name"], "acceptance_reached": bool(selected["accepted"])},
        "optimization": {"spectral_solver_success": bool(optimization.success), "spectral_optimum_reached": spectral_optimum_reached, "spectral_message": str(optimization.message), "function_evaluations": int(optimization.nfev), "best_spectral_sse_cm-2": best_sse, "selected_spectral_sse_cm-2": float(np.dot(ordinary_errors, ordinary_errors))},
        "nonbonded": {"electrostatics": "CM5 Gaussian-erf all-pairs", "van_der_waals": "ZAFF Exp-PE seed"},
        "off_diagonal_force_constants": "identically zero",
    }
    if request.stationarity_geometry_path is not None:
        stationarity = read_xyzin_geometry(Path(request.stationarity_geometry_path))
        xyz_target = np.asarray(stationarity.coordinates_angstrom, dtype=float) / BOHR_TO_ANGSTROM
        if xyz_target.shape != reference.cartesian_coordinates_bohr.shape:
            raise ValueError("stationarity geometry has a different atom count")
        target_b, target_labels = primitive_b_matrix_from_manifest(xyz_target, manifest)
        if target_labels != labels:
            raise ValueError("stationarity primitive contract differs from the curvature contract")
        atomic_numbers = np.asarray([atomic_number(atom) for atom in stationarity.atoms], dtype=int)
        electrostatic_gradient = electrostatic_energy_gradient(
            xyz_target, charges, (), atomic_numbers=atomic_numbers,
            electrostatic_model="gaussian_erf_all_pairs", backend="direct",
        ).gradient_hartree_per_bohr
        nonbonded_gradient = electrostatic_gradient + _exppe_gradient(xyz_target, seed, manifest["coordinates"]["R"])
        operator = target_b.T @ np.diag(fitted)
        displacement = np.linalg.lstsq(operator, -nonbonded_gradient, rcond=1.0e-10)[0]
        equilibrium_values = _primitive_values(xyz_target, manifest) - displacement
        residual_gradient = operator @ displacement + nonbonded_gradient
        payload["stationarity"] = {
            "geometry": str(request.stationarity_geometry_path),
            "equilibrium_primitive_values": equilibrium_values.tolist(),
            "total_gradient_rms_hartree_per_bohr": float(np.sqrt(np.mean(residual_gradient**2))),
            "total_gradient_maximum_hartree_per_bohr": float(np.max(np.abs(residual_gradient))),
            "force_constants_refitted": False,
        }
    Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(request.output_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
