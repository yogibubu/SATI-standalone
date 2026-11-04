"""Harmonic ZAFF fit with mandatory dissociating geminal X–H couplings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from matrix_qm import hessian_input_from_engine
from matrix_chem.topology.elements import atomic_symbol
from matrix_gf.harmonic_field import transform_cartesian_hessian
from matrix_zaff.geminal_coupling import ZaffGeminalXHCoupling
from matrix_zaff.seed_runtime import BOHR_TO_ANGSTROM

from .diagonal_harmonic import (
    _atom_equivalence_classes,
    _electrostatic_hessian,
    _exppe_hessian,
    _normal_modes,
    _primitive_equivalence_groups,
)
from .model import ZaffForceField, save_force_field


@dataclass(frozen=True)
class GeminalCoupledHarmonicFitRequest:
    hessian_path: Path
    primitive_manifest_path: Path
    cm5_path: Path
    zaff_seed_path: Path
    output_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str
    engine: str = "gdv"
    frozen_primitive_b_matrix: np.ndarray | None = None
    frozen_primitive_labels: tuple[str, ...] | None = None


def geminal_xh_pairs(
    coordinates: Mapping[str, Any], atomic_numbers: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    """Return (first primitive, second primitive, common heavy atom) pairs."""

    by_center: dict[int, list[int]] = {}
    for index, raw in enumerate(coordinates["R"]):
        left, right = (int(value) - 1 for value in raw)
        if left == right or min(left, right) < 0 or max(left, right) >= len(atomic_numbers):
            raise ValueError("invalid frozen bond primitive")
        if atomic_numbers[left] == 1 and atomic_numbers[right] != 1:
            by_center.setdefault(right, []).append(index)
        elif atomic_numbers[right] == 1 and atomic_numbers[left] != 1:
            by_center.setdefault(left, []).append(index)
    return tuple(
        (first, second, center)
        for center, indices in sorted(by_center.items())
        for first, second in combinations(indices, 2)
    )


def dissociating_stretch_coordinate(
    distance_bohr: float, reference_bohr: float, alpha_per_bohr: float,
) -> float:
    """Unit-slope stretch with two-sided dissociation damping."""

    distance, reference, alpha = map(float, (distance_bohr, reference_bohr, alpha_per_bohr))
    if not np.isfinite(distance) or distance <= 0 or reference <= 0 or alpha <= 0:
        raise ValueError("geminal stretch distances and alpha must be positive and finite")
    displacement = distance - reference
    scaled = alpha * displacement
    return float(displacement * np.exp(-(scaled**4)))


def geminal_coupling_energy(
    distance1_bohr: float, distance2_bohr: float, *,
    reference1_bohr: float, reference2_bohr: float,
    alpha1_per_bohr: float, alpha2_per_bohr: float,
    force_constant_hartree_per_bohr2: float,
) -> float:
    return float(force_constant_hartree_per_bohr2) * dissociating_stretch_coordinate(
        distance1_bohr, reference1_bohr, alpha1_per_bohr,
    ) * dissociating_stretch_coordinate(
        distance2_bohr, reference2_bohr, alpha2_per_bohr,
    )


def _seed_bond_alpha(seed: Mapping[str, Any], left: int, right: int) -> float:
    target = {left, right}
    matches = [
        float(item["alpha_per_angstrom"])
        for item in seed["bonds"] if set(map(int, item["atoms"])) == target
    ]
    if len(matches) != 1 or not np.isfinite(matches[0]) or matches[0] <= 0:
        raise ValueError(f"ZAFF seed lacks one positive Morse alpha for bond {left + 1}-{right + 1}")
    return matches[0] * BOHR_TO_ANGSTROM


def fit_geminal_coupled_harmonic_zaff(
    request: GeminalCoupledHarmonicFitRequest,
) -> dict[str, Any]:
    protocol = dict(request.protocol_manifest)
    if protocol.get("protocol_id") != "zaff-geminal-coupled-harmonic-fit.v1" or protocol.get("status") != "approved":
        raise ValueError("KEYMAKER requires the approved geminal-coupled protocol")
    if len(request.protocol_sha256) != 64:
        raise ValueError("KEYMAKER requires the installed protocol hash")
    reference = hessian_input_from_engine(request.engine, request.hessian_path)
    numbers = tuple(int(value) for value in reference.atomic_numbers)
    xyz = np.asarray(reference.cartesian_coordinates_bohr, dtype=float)
    masses = np.asarray(reference.masses_amu, dtype=float)
    manifest = json.loads(Path(request.primitive_manifest_path).read_text())
    coordinates = manifest["coordinates"]
    pairs = geminal_xh_pairs(coordinates, numbers)
    charges = np.asarray(json.loads(Path(request.cm5_path).read_text())["raw_cm5_charges_e"], dtype=float)
    if charges.shape != (len(numbers),) or not np.all(np.isfinite(charges)):
        raise ValueError("CM5 charges differ from the frozen Hessian atom order")
    charges -= (np.sum(charges) - float(round(np.sum(charges)))) / len(charges)
    seed_payload = json.loads(Path(request.zaff_seed_path).read_text())
    seed = seed_payload.get("metadata", {}).get("diagonal_seed_model", seed_payload)
    symbols = tuple(atomic_symbol(number) for number in numbers)
    if tuple(seed["atoms"]) != symbols:
        raise ValueError("ZAFF seed atom order differs from the Hessian source")
    nonbonded = _electrostatic_hessian(xyz, charges, reference.atomic_numbers)
    nonbonded += _exppe_hessian(xyz, seed, coordinates["R"])
    transform_options = (
        {"primitive_manifest": manifest}
        if request.frozen_primitive_b_matrix is None else
        {
            "frozen_b_matrix": np.asarray(request.frozen_primitive_b_matrix, dtype=float),
            "frozen_labels": request.frozen_primitive_labels,
        }
    )
    transformed = transform_cartesian_hessian(
        reference.cartesian_hessian - nonbonded, xyz, masses, **transform_options,
    )
    b = transformed.b_matrix
    redundant = transformed.redundant_hessian
    basis = redundant.vibrational_basis
    atom_classes = _atom_equivalence_classes(reference.atomic_numbers, xyz)
    expansion, diagonal_groups = _primitive_equivalence_groups(manifest, atom_classes)
    diagonal_blocks = [b.T @ np.diag(expansion[:, index]) @ b for index in range(expansion.shape[1])]
    pair_groups: dict[tuple[int, tuple[int, int]], list[tuple[int, int, int]]] = {}
    for first, second, center in pairs:
        endpoints = []
        for index in (first, second):
            left, right = (int(value) - 1 for value in coordinates["R"][index])
            endpoints.append(right if left == center else left)
        key = (atom_classes[center], tuple(sorted(atom_classes[atom] for atom in endpoints)))
        pair_groups.setdefault(key, []).append((first, second, center))
    coupling_blocks = []
    for group in pair_groups.values():
        block = np.zeros_like(reference.cartesian_hessian)
        for first, second, _ in group:
            block += np.outer(b[first], b[second]) + np.outer(b[second], b[first])
        coupling_blocks.append(block)
    blocks = diagonal_blocks + coupling_blocks
    inverse_mass = np.repeat(1.0 / np.sqrt(masses), 3)

    def projected(matrix: np.ndarray) -> np.ndarray:
        return basis.T @ (inverse_mass[:, None] * matrix * inverse_mass[None, :]) @ basis

    design = np.column_stack([projected(block).reshape(-1) for block in blocks])
    target_matrix = projected(reference.cartesian_hessian - nonbonded)
    values, _, rank, _ = np.linalg.lstsq(design, target_matrix.reshape(-1), rcond=1.0e-10)
    target_frequencies, target_modes = _normal_modes(reference.cartesian_hessian, masses, basis)

    def model_for(parameters: np.ndarray) -> np.ndarray:
        return nonbonded + sum(
            (value * block for value, block in zip(parameters, blocks)),
            np.zeros_like(reference.cartesian_hessian),
        )

    if len(values) <= len(target_frequencies):
        refined = least_squares(
            lambda parameters: _normal_modes(model_for(parameters), masses, basis)[0]
            - target_frequencies,
            values, max_nfev=2500, xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12,
        )
        if refined.success:
            values = refined.x
    model_hessian = model_for(values)
    frequencies, modes = _normal_modes(model_hessian, masses, basis)
    overlaps = np.abs(target_modes.T @ modes)
    rows, columns = linear_sum_assignment(-overlaps)
    assigned = overlaps[rows, columns]
    rms = float(np.sqrt(np.mean((frequencies - target_frequencies) ** 2)))
    coupling_values = values[len(diagonal_blocks):]
    coupling_records = []
    for group, force_constant in zip(pair_groups.values(), coupling_values):
        for first, second, center in group:
            bond1 = tuple(int(value) - 1 for value in coordinates["R"][first])
            bond2 = tuple(int(value) - 1 for value in coordinates["R"][second])
            first_h = bond1[1] if bond1[0] == center else bond1[0]
            second_h = bond2[1] if bond2[0] == center else bond2[0]
            coupling_records.append({
                "center_atom_one_based": center + 1,
                "hydrogens_one_based": [first_h + 1, second_h + 1],
                "primitive_indices_zero_based": [first, second],
                "force_constant_hartree_per_bohr2": float(force_constant),
                "reference_distances_bohr": [
                    float(np.linalg.norm(xyz[center] - xyz[first_h])),
                    float(np.linalg.norm(xyz[center] - xyz[second_h])),
                ],
                "alpha_per_bohr": [
                    _seed_bond_alpha(seed, center, first_h),
                    _seed_bond_alpha(seed, center, second_h),
                ],
                "functional_form": "DISPLACEMENT_QUARTIC_DAMPED_PRODUCT",
            })
    accepted = bool(
        rank == len(blocks) and np.all(frequencies > 0.0)
        and rms <= 10.0 and np.mean(assigned) >= 0.80
    )
    field_path = Path(request.output_path).with_name(
        Path(request.output_path).stem + ".zaff-field.json"
    )
    payload = {
        "schema": "matrix.architect.zaff_geminal_coupled_harmonic_fit.v1",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "protocol": {"id": protocol["protocol_id"], "sha256": request.protocol_sha256},
        "inputs": {
            "hessian": str(request.hessian_path),
            "primitive_manifest": str(request.primitive_manifest_path),
            "cm5": str(request.cm5_path), "zaff_seed": str(request.zaff_seed_path),
        },
        "primitive_labels": list(transformed.primitive_labels),
        "primitive_rank": redundant.primitive_rank,
        "vibrational_rank": len(target_frequencies),
        "design_rank": int(rank),
        "parameter_count": len(blocks),
        "diagonal_equivalence_groups": [list(group) for group in diagonal_groups],
        "fitted_diagonal_force_constants": (expansion @ values[:len(diagonal_blocks)]).tolist(),
        "geminal_xh_couplings": coupling_records,
        "model_cartesian_hessian_hartree_per_bohr2": model_hessian.tolist(),
        "dissociation_policy": "each coupled stretch factor tends to zero when its X-H distance tends to infinity",
        "reference_frequencies_cm-1": target_frequencies.tolist(),
        "model_frequencies_cm-1": frequencies.tolist(),
        "frequency_rms_error_cm-1": rms,
        "mean_assigned_mode_overlap": float(np.mean(assigned)),
        "all_modes_included": True,
        "accepted": accepted,
        "fitted_field": str(field_path) if accepted else None,
    }
    if accepted:
        field = ZaffForceField(
            atoms=symbols,
            reference_coordinates_angstrom=xyz * BOHR_TO_ANGSTROM,
            hessian_hartree_per_bohr2=model_hessian,
            metadata={
                "diagonal_seed_model": seed,
                "geminal_fit_protocol": payload["protocol"],
            },
            geminal_xh_couplings=tuple(
                ZaffGeminalXHCoupling.from_dict(item) for item in coupling_records
            ),
        )
        save_force_field(field_path, field)
    Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(request.output_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
