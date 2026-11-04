"""KeyMaker fit of a local ZAFF correction to a resident GFN2 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from matrix_chem import read_xyzin_geometry
from matrix_chem.topology.elements import atomic_symbol
from matrix_core import load_keymaker_protocol_library
from matrix_gfn2 import GFN2Calculator, harmonic_analysis
from matrix_qm import hessian_input_from_engine
from matrix_smith import (
    build_primitive_b_matrix, evaluate_primitive_values,
    gic_definition_to_dict, read_gic_definition_from_xyzin,
)

from .model import BOHR_TO_ANGSTROM, ZaffForceField, evaluate_force_field, save_force_field
from .primitive_residual import PRIMITIVE_RESIDUAL_SCHEMA
from .seed_compiler import compile_zaff_seed_model


GFN2_MORSE_STENCIL_STEP_ANGSTROM = 0.04


@dataclass(frozen=True)
class Gfn2ResidualFitRequest:
    geometry_path: Path
    qm_hessian_path: Path
    output_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str
    qm_engine: str = "gaussian"
    charge: int = 0
    multiplicity: int = 1
    maximum_gradient_error_hartree_per_bohr: float = 1.0e-6
    maximum_hessian_relative_error: float = 0.05
    soft_mode_target_frequency_cm1: float | None = None


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _aligned_hessian(
    source_coordinates_bohr: np.ndarray,
    target_coordinates_bohr: np.ndarray,
    hessian: np.ndarray,
) -> tuple[np.ndarray, float]:
    source = np.asarray(source_coordinates_bohr, dtype=float)
    target = np.asarray(target_coordinates_bohr, dtype=float)
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(source_centered.T @ target_centered)
    handedness = np.linalg.det(left @ right)
    rotation = left @ np.diag((1.0, 1.0, np.sign(handedness))) @ right
    discrepancy = source_centered @ rotation - target_centered
    rmsd_bohr = float(np.sqrt(np.mean(np.sum(discrepancy**2, axis=1))))
    transform = np.kron(np.eye(len(source)), rotation.T)
    aligned = transform @ hessian @ transform.T
    return 0.5 * (aligned + aligned.T), rmsd_bohr


def _smith_b(
    definition: Any, coordinates_angstrom: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    result = build_primitive_b_matrix(definition, coordinates_angstrom=coordinates_angstrom)
    rows = np.asarray(result.rows, dtype=float) * BOHR_TO_ANGSTROM
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        raise ValueError("frozen SMITH primitive B matrix is invalid")
    return rows, tuple(result.coordinate_labels)


def _smith_second_derivatives(
    definition: Any, coordinates_angstrom: np.ndarray, *, step_bohr: float = 1.0e-4,
) -> np.ndarray:
    """Differentiate the authoritative SMITH B rows in Cartesian bohr."""
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    reference, _ = _smith_b(definition, coordinates)
    size = coordinates.size
    seconds = np.empty((len(reference), size, size), dtype=float)
    for column in range(size):
        displaced = np.zeros(size)
        displaced[column] = step_bohr * BOHR_TO_ANGSTROM
        plus, _ = _smith_b(definition, coordinates + displaced.reshape(coordinates.shape))
        minus, _ = _smith_b(definition, coordinates - displaced.reshape(coordinates.shape))
        seconds[:, :, column] = (plus - minus) / (2.0 * step_bohr)
    return 0.5 * (seconds + seconds.transpose(0, 2, 1))


def _project_primitive_field(
    b: np.ndarray, second: np.ndarray, gradient: np.ndarray, hessian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    primitive_gradient = np.linalg.pinv(b.T, rcond=1.0e-10) @ gradient
    correction = np.einsum("p,pij->ij", primitive_gradient, second)
    inverse = np.linalg.pinv(b, rcond=1.0e-10)
    primitive_hessian = inverse.T @ (hessian - correction) @ inverse
    return primitive_gradient, 0.5 * (primitive_hessian + primitive_hessian.T)


def _morse_alpha_from_derivatives(gradient: float, curvature: float, cubic: float) -> float:
    """Solve F3 = -3 alpha F2 - 2 alpha² F1 for the positive Morse exponent."""
    c, k, t = float(gradient), float(curvature), float(cubic)
    if not all(np.isfinite(value) for value in (c, k, t)) or k <= 0.0:
        raise ValueError("GFN2 bond Morse derivatives are invalid")
    if abs(c) <= 1.0e-10 * k:
        alpha = -t / (3.0 * k)
    else:
        roots = np.roots((2.0 * c, 3.0 * k, t))
        positive = [float(root.real) for root in roots if abs(root.imag) < 1.0e-9 and root.real > 0.0]
        if not positive:
            raise ValueError("GFN2 bond derivatives admit no positive Morse alpha")
        alpha = min(positive, key=lambda value: abs(value + t / (3.0 * k)))
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("GFN2 bond derivatives admit no positive Morse alpha")
    return float(alpha)


def _gfn2_stretch_cubics(
    calculator: GFN2Calculator, definition: Any, coordinates_angstrom: np.ndarray,
    b: np.ndarray, labels: tuple[str, ...], primitive_gradient: np.ndarray,
    primitive_hessian: np.ndarray, reference_gradient: np.ndarray,
    *, symbols: tuple[str, ...], reference_energy: float,
    xh_energy_only: bool = False,
    step_angstrom: float = GFN2_MORSE_STENCIL_STEP_ANGSTROM,
) -> tuple[list[float], list[dict[str, float]]]:
    """Recover local GFN2 primitive F3 from stable directional force stencils.

    Redundant primitives use the same Moore-Penrose gauge as the harmonic fit.
    The Cartesian directional third derivative is corrected for the second
    and third derivatives of the frozen SMITH primitive map.  The remaining
    diagonal F3 is the local bond term in the fitted primitive model.
    """
    inverse = np.linalg.pinv(b, rcond=1.0e-10)
    projector = b @ inverse
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    cubics = [0.0] * len(labels)
    evidence: list[dict[str, float]] = []
    values = evaluate_primitive_values(
        definition, primitive_ids=labels, coordinates_angstrom=coordinates_angstrom,
    )
    reference_values = np.asarray([values[label] for label in labels], dtype=float)

    def sampled(displaced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = calculator.evaluate(displaced, compute_bond_orders=False)
        if not result.converged:
            raise ValueError("GFN2 Morse stencil did not converge")
        local_values = evaluate_primitive_values(
            definition, primitive_ids=labels, coordinates_angstrom=displaced,
        )
        return (
            np.asarray(result.gradient_hartree_per_bohr, dtype=float).reshape(-1),
            np.asarray([local_values[label] for label in labels], dtype=float),
        )

    for index, label in enumerate(labels):
        primitive = primitive_by_id[label]
        if primitive.family != "STRETCH":
            continue
        atom_pair = tuple(int(atom) - 1 for atom in primitive.atoms)
        tangent = inverse[:, index]
        projected_tangent = projector[:, index]
        if xh_energy_only and len(atom_pair) == 2 and sum(symbols[atom] == "H" for atom in atom_pair) == 1:
            if projector[index, index] <= 0.0:
                raise ValueError("GFN2 X-H primitive has no independent tangent")
            energy_points = []
            for sign in (1.0, -1.0):
                displaced = np.asarray(coordinates_angstrom, dtype=float).copy()
                displaced += (
                    sign * step_angstrom * BOHR_TO_ANGSTROM * tangent
                ).reshape(coordinates_angstrom.shape)
                energy = calculator.evaluate_energy(displaced)
                local_values = evaluate_primitive_values(
                    definition, primitive_ids=labels, coordinates_angstrom=displaced,
                )
                delta = np.asarray([local_values[item] for item in labels]) - reference_values
                harmonic_energy = (
                    reference_energy + primitive_gradient @ delta
                    + 0.5 * delta @ primitive_hessian @ delta
                )
                energy_points.append((float(energy), float(energy - harmonic_energy), delta))
            denominator = energy_points[0][2][index]**3 - energy_points[1][2][index]**3
            if abs(denominator) <= 1.0e-12:
                raise ValueError("GFN2 X-H energy stencil has no cubic displacement")
            cubic = 6.0 * (energy_points[0][1] - energy_points[1][1]) / denominator
            if not np.isfinite(cubic):
                raise ValueError("GFN2 X-H energy stencil is non-finite")
            evidence.append({
                "primitive_index": index,
                "cubic_hartree_per_angstrom3": float(cubic),
                "energy_plus_hartree": energy_points[0][0],
                "energy_minus_hartree": energy_points[1][0],
                "reference_energy_hartree": float(reference_energy),
                "off_target_primitive_norm": float(max(
                    np.linalg.norm(np.delete(item[2], index)) for item in energy_points
                )),
                "tangent_projection": float(projector[index, index]),
                "step_angstrom": float(step_angstrom),
                "method": "TWO_XH_PRIMITIVE_TANGENT_ENERGIES",
            })
            cubics[index] = float(cubic)
            continue

        def stencil(step: float) -> float:
            def point(multiple: float) -> tuple[np.ndarray, np.ndarray]:
                displacement = (
                    multiple * step * BOHR_TO_ANGSTROM * tangent
                ).reshape(coordinates_angstrom.shape)
                return sampled(coordinates_angstrom + displacement)

            plus_gradient, plus_values = point(1.0)
            minus_gradient, minus_values = point(-1.0)
            _, plus_two_values = point(2.0)
            _, minus_two_values = point(-2.0)
            directional_cubic = (
                (plus_gradient - 2.0 * reference_gradient + minus_gradient) @ tangent
            ) / step**2
            primitive_second = (
                plus_values - 2.0 * reference_values + minus_values
            ) / step**2
            primitive_third = (
                plus_two_values - 2.0 * plus_values
                + 2.0 * minus_values - minus_two_values
            ) / (2.0 * step**3)
            geometry_contribution = (
                3.0 * primitive_second @ primitive_hessian @ projected_tangent
                + primitive_gradient @ primitive_third
            )
            if projector[index, index] <= 0.0:
                raise ValueError("GFN2 stretch primitive has no independent tangent")
            return float(
                (directional_cubic - geometry_contribution)
                / projector[index, index] ** 3
            )

        coarse = stencil(step_angstrom)
        fine = stencil(step_angstrom / 2.0)
        cubic = (4.0 * fine - coarse) / 3.0
        evidence.append({
            "primitive_index": index,
            "cubic_hartree_per_angstrom3": cubic,
            "coarse_cubic_hartree_per_angstrom3": coarse,
            "fine_cubic_hartree_per_angstrom3": fine,
            "relative_stencil_difference": abs(fine - coarse) / max(abs(cubic), 1.0e-12),
            "tangent_projection": float(projector[index, index]),
            "step_angstrom": float(step_angstrom),
        })
        cubics[index] = cubic
    return cubics, evidence


def fit_gfn2_residual_zaff(request: Gfn2ResidualFitRequest) -> dict[str, Any]:
    protocol = dict(request.protocol_manifest)
    protocol_id = protocol.get("protocol_id")
    if protocol_id not in {"zaff-gfn2-residual-fit.v1", "zaff-gfn2-residual-fit.v2", "zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"} or protocol.get("status") != "approved":
        raise ValueError("KeyMaker requires the approved GFN2 residual protocol")
    installed = load_keymaker_protocol_library().resolve(str(protocol_id))
    if request.protocol_sha256 != installed.sha256 or protocol != installed.to_dict():
        raise ValueError("KeyMaker requires the unchanged installed GFN2 residual protocol")
    geometry_path = Path(request.geometry_path)
    hessian_path = Path(request.qm_hessian_path)
    geometry = read_xyzin_geometry(geometry_path)
    definition = read_gic_definition_from_xyzin(geometry_path)
    symbols = tuple(geometry.atoms)
    xyz_angstrom = np.asarray(geometry.coordinates_angstrom, dtype=float)
    xyz_bohr = xyz_angstrom / BOHR_TO_ANGSTROM
    reference = hessian_input_from_engine(request.qm_engine, hessian_path)
    if symbols != tuple(atomic_symbol(int(number)) for number in reference.atomic_numbers):
        raise ValueError("QM Hessian atom order differs from frozen G")
    if request.maximum_gradient_error_hartree_per_bohr <= 0 or request.maximum_hessian_relative_error <= 0:
        raise ValueError("residual fit tolerances must be positive")

    qm_hessian, geometry_rmsd_bohr = _aligned_hessian(
        reference.cartesian_coordinates_bohr, xyz_bohr, reference.cartesian_hessian,
    )
    soft_mode_override = None
    if request.soft_mode_target_frequency_cm1 is not None:
        if protocol_id not in {"zaff-gfn2-residual-fit.v2", "zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"}:
            raise ValueError("soft-mode curvature adjustment requires the primitive protocol")
        target_frequency = float(request.soft_mode_target_frequency_cm1)
        if not np.isfinite(target_frequency) or target_frequency <= 0.0:
            raise ValueError("soft-mode target frequency must be positive")
        modes = harmonic_analysis(qm_hessian, xyz_angstrom, reference.masses_amu)
        imaginary = np.flatnonzero(modes.frequencies_cm1 < 0.0)
        if len(imaginary) != 1:
            raise ValueError("soft-mode adjustment requires exactly one imaginary QM mode")
        index = int(imaginary[0])
        old_frequency = float(modes.frequencies_cm1[index])
        factor = 5140.487143715828
        eigenvalue_change = (target_frequency / factor)**2 + (old_frequency / factor)**2
        direction = np.repeat(np.sqrt(reference.masses_amu), 3) * modes.mass_weighted_modes[index]
        qm_hessian = qm_hessian + eigenvalue_change * np.outer(direction, direction)
        qm_hessian = 0.5 * (qm_hessian + qm_hessian.T)
        soft_mode_override = {
            "mode_one_based": index + 1,
            "original_frequency_cm-1": old_frequency,
            "target_frequency_cm-1": target_frequency,
            "operation": "MASS_WEIGHTED_RANK_ONE_CURVATURE_REPLACEMENT",
        }
    calculator = GFN2Calculator(
        symbols, charge=request.charge, multiplicity=request.multiplicity,
        temperature_kelvin=0.0,
    )
    baseline = calculator.evaluate(
        xyz_angstrom, retain_orbitals=True, compute_bond_orders=protocol_id != "zaff-gfn2-residual-fit.v1",
    )
    if not baseline.converged or baseline.orbital_state is None:
        raise ValueError("resident GFN2 did not produce a converged orbital state")
    hessian_result = calculator.ground_state_hessian(baseline.orbital_state)
    gfn2_gradient = np.asarray(baseline.gradient_hartree_per_bohr, dtype=float).reshape(-1)
    gfn2_hessian = np.asarray(hessian_result.hessian_hartree_per_bohr2, dtype=float)
    target_gradient = -gfn2_gradient
    target_hessian = qm_hessian - gfn2_hessian

    b, primitive_labels = _smith_b(definition, xyz_angstrom)
    if b.shape[1] != xyz_bohr.size:
        raise ValueError("SMITH B matrix does not match G")
    second = _smith_second_derivatives(definition, xyz_angstrom)
    if protocol_id in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"}:
        qm_primitive_gradient, qm_primitive_hessian = _project_primitive_field(
            b, second, np.zeros_like(gfn2_gradient), qm_hessian,
        )
        gfn2_primitive_gradient, gfn2_primitive_hessian = _project_primitive_field(
            b, second, gfn2_gradient, gfn2_hessian,
        )
        primitive_gradient = qm_primitive_gradient - gfn2_primitive_gradient
        primitive_hessian = qm_primitive_hessian - gfn2_primitive_hessian
    else:
        primitive_gradient, primitive_hessian = _project_primitive_field(
            b, second, target_gradient, target_hessian,
        )
    curvature_from_gradient = np.einsum("p,pij->ij", primitive_gradient, second)
    values = evaluate_primitive_values(
        definition, primitive_ids=primitive_labels, coordinates_angstrom=xyz_angstrom,
    )
    primitive_reference = np.asarray([values[label] for label in primitive_labels])
    primitive_equilibrium = primitive_reference - np.linalg.pinv(
        primitive_hessian, rcond=1.0e-10,
    ) @ primitive_gradient
    equilibrium_parameter_error = float(np.linalg.norm(
        primitive_hessian @ (primitive_reference - primitive_equilibrium)
        - primitive_gradient
    ))
    fitted_gradient = b.T @ primitive_gradient
    fitted_hessian = b.T @ primitive_hessian @ b + curvature_from_gradient
    fitted_hessian = 0.5 * (fitted_hessian + fitted_hessian.T)
    gradient_error = float(np.max(np.abs(fitted_gradient - target_gradient)))
    hessian_error = float(np.linalg.norm(fitted_hessian - target_hessian) / max(np.linalg.norm(target_hessian), 1.0e-12))
    accepted = bool(
        gradient_error <= request.maximum_gradient_error_hartree_per_bohr
        and hessian_error <= request.maximum_hessian_relative_error
    )

    output = Path(request.output_path)
    field_path = output.with_name(output.stem + ".zaff-field.json")
    baseline_path = output.with_name(output.stem + ".gfn2-baseline.json")
    baseline_payload = {
        "schema": "matrix.gfn2.residual_baseline.v1",
        "geometry_angstrom": xyz_angstrom.tolist(),
        "atoms": list(symbols),
        "charge": request.charge,
        "multiplicity": request.multiplicity,
        "energy_hartree": float(baseline.energy_hartree),
        "converged": bool(baseline.converged),
        "native_scc_charges_electron": np.asarray(baseline.charges_electron, dtype=float).tolist(),
        "native_mayer_bond_orders": (
            None if protocol_id == "zaff-gfn2-residual-fit.v1"
            else np.asarray(baseline.bond_orders, dtype=float).tolist()
        ),
        "gradient_hartree_per_bohr": gfn2_gradient.tolist(),
        "hessian_hartree_per_bohr2": gfn2_hessian.tolist(),
        "reciprocity_error_hartree_per_bohr2": hessian_result.reciprocity_error_hartree_per_bohr2,
        "translational_residual_hartree_per_bohr2": hessian_result.translational_residual_hartree_per_bohr2,
        "maximum_cpscf_residual": hessian_result.maximum_cpscf_residual,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline_payload, indent=2) + "\n")
    field = None
    if accepted:
        metadata = {
            "baseline": "GFN2-XTB",
            "gfn2_charge": int(request.charge),
            "gfn2_multiplicity": int(request.multiplicity),
            "scope": "FROZEN_SMITH_PRIMITIVE_RESIDUAL" if protocol_id != "zaff-gfn2-residual-fit.v1" else "LOCAL_HARMONIC_RESIDUAL",
            "nonbonded_terms": "NONE",
            "zaff_atomic_charges_e": [0.0] * len(symbols),
            "zaff_vdw_terms": [],
            "protocol_id": protocol_id,
            "protocol_sha256": request.protocol_sha256,
            "smith_primitive_labels": list(primitive_labels),
            "smith_primitive_gradient": primitive_gradient.tolist(),
            "smith_primitive_hessian": primitive_hessian.tolist(),
            "smith_primitive_reference_values": primitive_reference.tolist(),
            "smith_primitive_equilibrium_values": primitive_equilibrium.tolist(),
        }
        if protocol_id != "zaff-gfn2-residual-fit.v1":
            bond_orders = np.asarray(baseline.bond_orders, dtype=float)
            native_orders = {
                (left, right): float(bond_orders[left, right])
                for left in range(len(symbols)) for right in range(left + 1, len(symbols))
                if bond_orders[left, right] > 0.1
            }
            seed = compile_zaff_seed_model(
                symbols, xyz_angstrom, cm5_charges=np.zeros(len(symbols)),
                mayer_bond_orders=native_orders,
                charge_source="ZAFF zero local charges",
                bond_order_source="resident GFN2 Mayer",
            )
            alpha_by_bond = {
                frozenset(int(atom) for atom in item["atoms"]): float(item["alpha_per_angstrom"])
                for item in seed["bonds"]
            }
            primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
            morse_alpha = []
            for label in primitive_labels:
                primitive = primitive_by_id[label]
                if primitive.family != "STRETCH":
                    morse_alpha.append(0.0)
                    continue
                key = frozenset(int(atom) - 1 for atom in primitive.atoms)
                if key not in alpha_by_bond:
                    raise ValueError(f"ZAFF Morse alpha is missing for SMITH primitive {label}")
                morse_alpha.append(alpha_by_bond[key])
            gfn2_morse_alpha = None
            gfn2_cubic_evidence = None
            if protocol_id in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"}:
                gfn2_cubic, gfn2_cubic_evidence = _gfn2_stretch_cubics(
                    calculator, definition, xyz_angstrom, b, primitive_labels,
                    gfn2_primitive_gradient, gfn2_primitive_hessian, gfn2_gradient,
                    symbols=symbols, reference_energy=float(baseline.energy_hartree),
                    xh_energy_only=protocol_id.endswith(".v4"),
                )
                gfn2_morse_alpha = [
                    _morse_alpha_from_derivatives(
                        gfn2_primitive_gradient[index],
                        gfn2_primitive_hessian[index, index], cubic,
                    ) if morse_alpha[index] > 0.0 else 0.0
                    for index, cubic in enumerate(gfn2_cubic)
                ]
            metadata["primitive_residual_model"] = {
                "schema": ("matrix.zaff.primitive_residual.v2" if protocol_id in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"} else PRIMITIVE_RESIDUAL_SCHEMA),
                "smith_definition": gic_definition_to_dict(definition),
                "primitive_labels": list(primitive_labels),
                "reference_values": primitive_reference.tolist(),
                "primitive_gradient": primitive_gradient.tolist(),
                "primitive_hessian": primitive_hessian.tolist(),
                "morse_alpha_per_angstrom": morse_alpha,
                "morse_alpha_source": "ZAFF_MM3_SYNTHON_INTERPOLATION_WITH_NATIVE_GFN2_MAYER_AND_ZERO_ZAFF_CHARGES",
                "stretch_form": "CENTERED_MORSE_LOCAL_HARMONIC_MATCH",
                "cross_form": "BOND_LOCAL_QUARTIC_DAMPED_STRETCH_PRODUCT",
                "zaff_hierarchy_role": "ZAFF1_GFN2_DELTA_MORSE_CORRECTION",
            }
            if protocol_id in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"}:
                metadata["primitive_residual_model"].update({
                    "qm_primitive_gradient": qm_primitive_gradient.tolist(),
                    "qm_primitive_hessian": qm_primitive_hessian.tolist(),
                    "gfn2_primitive_gradient": gfn2_primitive_gradient.tolist(),
                    "gfn2_primitive_hessian": gfn2_primitive_hessian.tolist(),
                    "gfn2_morse_alpha_per_angstrom": gfn2_morse_alpha,
                    "gfn2_morse_alpha_source": (
                        "TWO_XH_PRIMITIVE_TANGENT_ENERGIES_AND_OTHER_STRETCH_GRADIENT_STENCILS"
                        if protocol_id.endswith(".v4") else "PROJECTED_GFN2_PRIMITIVE_THIRD_DERIVATIVE"
                    ),
                    "gfn2_cubic_stencil": gfn2_cubic_evidence,
                    "stretch_form": "QM_ZAFF_MORSE_MINUS_GFN2_DERIVED_MORSE",
                })
        field = ZaffForceField(
            atoms=symbols,
            reference_coordinates_angstrom=xyz_angstrom,
            hessian_hartree_per_bohr2=fitted_hessian,
            energy_reference_hartree=0.0,
            gradient_reference_hartree_per_bohr=fitted_gradient,
            metadata=metadata,
        )
        save_force_field(field_path, field)
        evaluated = evaluate_force_field(field, xyz_angstrom)
        combined_gradient = gfn2_gradient + evaluated.gradient_hartree_per_bohr.reshape(-1)
        combined_hessian = gfn2_hessian + evaluated.hessian_hartree_per_bohr2
        if evaluated.energy_hartree != 0.0 or not np.allclose(combined_gradient, 0.0, atol=request.maximum_gradient_error_hartree_per_bohr) or not np.allclose(
            combined_hessian, qm_hessian,
            atol=request.maximum_hessian_relative_error * max(float(np.max(np.abs(qm_hessian))), 1.0e-8),
        ):
            field_path.unlink(missing_ok=True)
            accepted = False
            field = None

    payload = {
        "schema": "matrix.architect.zaff_gfn2_residual_fit.v1",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "accepted": accepted,
        "protocol": {"id": protocol["protocol_id"], "sha256": request.protocol_sha256},
        "inputs": {
            "geometry": {"path": str(geometry_path), "sha256": _file_sha256(geometry_path)},
            "qm_hessian": {"path": str(hessian_path), "sha256": _file_sha256(hessian_path)},
        },
        "assumption": "G and F are the best available independent sources; target QM gradient at G is zero",
        "soft_mode_override": soft_mode_override,
        "subtraction": "QM_MINUS_COMPLETE_RESIDENT_GFN2_ONLY",
        "bond_morse_construction": (
            "QM_ZAFF_MORSE_MINUS_GFN2_DERIVED_MORSE"
            if protocol_id in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"} else "RESIDUAL_DERIVATIVES_TO_MORSE"
        ),
        "zaff_reference_energy_hartree": 0.0,
        "zaff_atomic_charges_e": [0.0] * len(symbols),
        "zaff_vdw_terms": [],
        "qm_hessian_geometry_rmsd_angstrom": geometry_rmsd_bohr * BOHR_TO_ANGSTROM,
        "primitive_labels": list(primitive_labels),
        "primitive_rank": int(np.linalg.matrix_rank(b)),
        "primitive_gradient": primitive_gradient.tolist(),
        "primitive_hessian": primitive_hessian.tolist(),
        "primitive_reference_values": primitive_reference.tolist(),
        "primitive_equilibrium_values": primitive_equilibrium.tolist(),
        "equilibrium_parameterization_l2_error": equilibrium_parameter_error,
        "target_gradient_hartree_per_bohr": target_gradient.tolist(),
        "target_hessian_hartree_per_bohr2": target_hessian.tolist(),
        "fitted_gradient_hartree_per_bohr": fitted_gradient.tolist(),
        "fitted_hessian_hartree_per_bohr2": fitted_hessian.tolist(),
        "gradient_max_error_hartree_per_bohr": gradient_error,
        "hessian_relative_frobenius_error": hessian_error,
        "maximum_gradient_error_hartree_per_bohr": request.maximum_gradient_error_hartree_per_bohr,
        "maximum_hessian_relative_error": request.maximum_hessian_relative_error,
        "gfn2_baseline": str(baseline_path),
        "fitted_field": str(field_path) if accepted else None,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = ["Gfn2ResidualFitRequest", "fit_gfn2_residual_zaff"]
