"""Refit QM Morse exponents against a matched normal-mode cubic field."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import lsq_linear

from matrix_gaussian import (
    read_gaussian_anharmonic_normal_cubic,
    read_gaussian_log_geometry,
    read_gaussian_log_normal_modes,
)
from matrix_qm import hessian_input_from_engine
from matrix_rovib import isotopic_masses_amu
from matrix_smith import build_primitive_b_matrix, gic_definition_from_dict
from matrix_trinity.computed_cubic_job import read_computed_cubic_job
from matrix_trinity.curvilinear_rovib import (
    CurvilinearIsotopologueDefinition,
    curvilinear_deltabvib_from_definitions,
    curvilinear_isotopologue_state_from_internal_field,
)
from matrix_trinity.isotopic_internal_qff import normal_cubic_from_internal_field

from .model import BOHR_TO_ANGSTROM, evaluate_force_field, load_force_field, save_force_field


PROTOCOL_ID = "zaff-gfn2-morse-cubic-refit.v1"


@dataclass(frozen=True)
class MorseCubicRefitRequest:
    base_field_path: Path
    computed_cubic_job_path: Path
    reference_anharmonic_log_path: Path
    output_field_path: Path
    output_report_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str


def _semidiagonal(tensor: np.ndarray) -> np.ndarray:
    return np.einsum("ijj->ij", tensor)


def _matched_reference_cubic(
    reference: np.ndarray, overlap: np.ndarray, current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve the arbitrary Gaussian normal-coordinate phases deterministically."""
    left, _, right = np.linalg.svd(overlap)
    orthogonal = left @ right

    def transformed(signs: np.ndarray) -> np.ndarray:
        mapping = orthogonal * signs[:, None]
        return np.einsum(
            "abc,ai,bj,ck->ijk", reference, mapping, mapping, mapping,
            optimize=True,
        )

    generator = np.random.default_rng(3)
    best: tuple[float, np.ndarray] | None = None
    for trial in range(40):
        signs = (
            np.ones(overlap.shape[0]) if trial == 0
            else generator.choice((-1.0, 1.0), overlap.shape[0])
        )
        for _ in range(10):
            improved = False
            for mode in range(len(signs)):
                before = np.linalg.norm(_semidiagonal(transformed(signs)) - _semidiagonal(current))
                signs[mode] *= -1.0
                after = np.linalg.norm(_semidiagonal(transformed(signs)) - _semidiagonal(current))
                if after >= before:
                    signs[mode] *= -1.0
                else:
                    improved = True
            if not improved:
                break
        score = float(np.linalg.norm(_semidiagonal(transformed(signs)) - _semidiagonal(current)))
        if best is None or score < best[0]:
            best = score, signs.copy()
    assert best is not None
    return transformed(best[1]), best[1], best[0]


def refit_gfn2_zaff_morse_cubics(request: MorseCubicRefitRequest) -> dict[str, Any]:
    """Fit only the QM Morse alphas; retain GFN2 alphas and E/G/H at G."""
    manifest = dict(request.protocol_manifest)
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "approved":
        raise ValueError("Morse cubic refit requires its approved KEYMAKER protocol")
    if len(request.protocol_sha256) != 64:
        raise ValueError("Morse cubic refit requires the installed protocol hash")
    base_path = Path(request.base_field_path)
    job_path = Path(request.computed_cubic_job_path)
    log_path = Path(request.reference_anharmonic_log_path)
    base = load_force_field(base_path)
    metadata = dict(base.metadata)
    if metadata.get("protocol_id") != "zaff-gfn2-residual-fit.v3":
        raise ValueError("Morse cubic refit requires a v3 two-Morse field")
    primitive = dict(metadata["primitive_residual_model"])
    if primitive.get("schema") != "matrix.zaff.primitive_residual.v2":
        raise ValueError("Morse cubic refit requires the two-Morse primitive runtime")
    job, job_payload = read_computed_cubic_job(job_path)
    field_hash = sha256(base_path.read_bytes()).hexdigest()
    if job_payload["method_artifact_sha256"] != field_hash:
        raise ValueError("computed F2/F3 job does not belong to the input Morse field")
    coordinates = np.asarray(base.reference_coordinates_angstrom, dtype=float)
    if job.symbols != base.atoms or not np.allclose(job.coordinates_angstrom, coordinates, atol=1e-10):
        raise ValueError("computed F2/F3 atom order or geometry differs from the Morse field")

    gaussian_geometry = read_gaussian_log_geometry(log_path)
    gaussian_modes = read_gaussian_log_normal_modes(log_path)
    gaussian_cubic = read_gaussian_anharmonic_normal_cubic(log_path)
    gaussian_hessian = hessian_input_from_engine("gaussian", log_path)
    if gaussian_geometry.atoms != base.atoms:
        raise ValueError("B3LYP cubic reference atom order differs from the Morse field")
    nmode = job.field.coordinate_count
    if len(gaussian_modes.frequencies_cm) != nmode or len(gaussian_cubic.harmonic_frequencies_cm1) != nmode:
        raise ValueError("B3LYP cubic reference mode count differs from TRINITY")
    if not np.allclose(
        gaussian_cubic.harmonic_frequencies_cm1[::-1], gaussian_modes.frequencies_cm,
        atol=0.05, rtol=0.0,
    ):
        raise ValueError("B3LYP cubic and harmonic mode orders disagree")
    source = np.asarray(gaussian_geometry.coordinates_angstrom, dtype=float)
    left, _, right = np.linalg.svd((source - source.mean(0)).T @ (coordinates - coordinates.mean(0)))
    rotation = left @ np.diag((1.0, 1.0, np.linalg.det(left @ right))) @ right
    geometry_rmsd = float(np.sqrt(np.mean(np.sum(
        ((source - source.mean(0)) @ rotation - (coordinates - coordinates.mean(0))) ** 2,
        axis=1,
    ))))
    if geometry_rmsd > 0.05:
        raise ValueError("B3LYP cubic reference geometry differs by more than 0.05 angstrom RMSD")

    masses = isotopic_masses_amu(base.atoms)
    state = curvilinear_isotopologue_state_from_internal_field(
        job.field, job.b_matrix, masses,
        CurvilinearIsotopologueDefinition("parent", {}, "Ir"),
    )
    composite_modes = state.modes_mw.reshape((nmode, -1))
    cartesian_per_q = composite_modes.T / np.sqrt(np.repeat(masses, 3))[:, None]
    _, current_cubic = normal_cubic_from_internal_field(
        job.field, job.b_matrix, job.b_prime, cartesian_per_q,
    )
    gaussian_mw = (
        gaussian_modes.modes.reshape((nmode, len(base.atoms), 3)) @ rotation
    ).reshape((nmode, -1)) * np.sqrt(np.repeat(gaussian_hessian.masses_amu, 3))[None, :]
    gaussian_mw /= np.linalg.norm(gaussian_mw, axis=1)[:, None]
    overlap = gaussian_mw @ composite_modes.T
    from scipy.optimize import linear_sum_assignment
    rows, columns = linear_sum_assignment(-np.abs(overlap))
    matched_overlap = np.abs(overlap[rows, columns])
    if np.min(matched_overlap) < 0.75:
        raise ValueError("B3LYP/composite normal-mode overlap is too small for a Morse cubic refit")
    reference_cubic, phase_signs, _ = _matched_reference_cubic(
        gaussian_cubic.cubic_qmw_hartree_amu32_bohr3[::-1, ::-1, ::-1],
        overlap, current_cubic,
    )

    definition = gic_definition_from_dict(primitive["smith_definition"])
    primitive_b = np.asarray(
        build_primitive_b_matrix(definition, coordinates_angstrom=coordinates).rows,
        dtype=float,
    ) * BOHR_TO_ANGSTROM
    if len(primitive_b) != len(primitive["primitive_labels"]):
        raise ValueError("SMITH primitive order changed since the two-Morse fit")
    primitive_per_q = primitive_b @ cartesian_per_q
    stretch_indices = [i for i, alpha in enumerate(primitive["morse_alpha_per_angstrom"]) if alpha > 0.0]
    library_alphas = np.asarray(primitive["morse_alpha_per_angstrom"], dtype=float)
    response = np.stack([
        -3.0 * float(primitive["qm_primitive_hessian"][i][i])
        * np.einsum("a,b,c->abc", primitive_per_q[i], primitive_per_q[i], primitive_per_q[i])
        for i in stretch_indices
    ], axis=-1)
    design = np.einsum("ijjp->ijp", response).reshape((-1, len(stretch_indices)))
    target = (_semidiagonal(reference_cubic) - _semidiagonal(current_cubic)).reshape(-1)
    prior = library_alphas[stretch_indices]
    result = lsq_linear(design, target, bounds=(-0.5 * prior, prior))
    if not result.success:
        raise ValueError("B3LYP Morse cubic fit did not converge")
    fitted_alphas = library_alphas.copy()
    fitted_alphas[stretch_indices] += result.x
    fitted_cubic = current_cubic + np.tensordot(response, result.x, axes=(-1, 0))
    initial_rmse = float(np.sqrt(np.mean((_semidiagonal(current_cubic) - _semidiagonal(reference_cubic)) ** 2)))
    fitted_rmse = float(np.sqrt(np.mean((_semidiagonal(fitted_cubic) - _semidiagonal(reference_cubic)) ** 2)))
    if fitted_rmse >= initial_rmse:
        raise ValueError("B3LYP Morse cubic refit did not improve the semidiagonal cubics")

    primitive["morse_alpha_library_prior_per_angstrom"] = library_alphas.tolist()
    primitive["morse_alpha_per_angstrom"] = fitted_alphas.tolist()
    primitive["morse_alpha_source"] = "B3LYP_NORMAL_CUBIC_CONSTRAINED_REFIT"
    primitive["cubic_reference_sha256"] = sha256(log_path.read_bytes()).hexdigest()
    metadata["primitive_residual_model"] = primitive
    metadata["protocol_id"] = PROTOCOL_ID
    metadata["protocol_sha256"] = request.protocol_sha256
    metadata["cubic_refit_base_field_sha256"] = field_hash
    refined = replace(base, metadata=metadata)
    old_eval = evaluate_force_field(base, coordinates)
    new_eval = evaluate_force_field(refined, coordinates)
    if new_eval.energy_hartree != 0.0 or not np.allclose(
        new_eval.gradient_hartree_per_bohr, old_eval.gradient_hartree_per_bohr, atol=1e-12,
    ) or not np.allclose(
        new_eval.hessian_hartree_per_bohr2, old_eval.hessian_hartree_per_bohr2, atol=1e-12,
    ):
        raise ValueError("Morse cubic refit changed reference energy, gradient or Hessian")

    internal_b = primitive_b @ np.linalg.pinv(job.b_matrix, rcond=1e-10)
    if np.linalg.norm(internal_b @ job.b_matrix - primitive_b) / np.linalg.norm(primitive_b) > 1e-8:
        raise ValueError("SMITH primitives cannot be represented by the TRINITY internal basis")
    new_internal_cubic = job.field.cubic_internal.copy()
    for index, change in zip(stretch_indices, result.x, strict=True):
        coefficient = -3.0 * float(primitive["qm_primitive_hessian"][index][index]) * change
        new_internal_cubic += coefficient * np.einsum(
            "i,j,k->ijk", internal_b[index], internal_b[index], internal_b[index],
        )
    new_field = replace(job.field, cubic_internal=new_internal_cubic)
    deltavib = curvilinear_deltabvib_from_definitions(
        new_field, job.b_matrix, job.b_prime, base.atoms, coordinates,
        (CurvilinearIsotopologueDefinition("parent", {}, "Ir"),),
    )[0]
    report = {
        "schema": "matrix.architect.morse_cubic_refit.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": request.protocol_sha256,
        "base_field_sha256": field_hash,
        "computed_cubic_job_sha256": sha256(job_path.read_bytes()).hexdigest(),
        "reference_anharmonic_log_sha256": primitive["cubic_reference_sha256"],
        "reference_geometry_rmsd_angstrom": geometry_rmsd,
        "minimum_matched_mode_overlap": float(np.min(matched_overlap)),
        "mode_phase_signs": phase_signs.astype(int).tolist(),
        "stretch_labels": [primitive["primitive_labels"][i] for i in stretch_indices],
        "library_alpha_per_angstrom": prior.tolist(),
        "fitted_alpha_per_angstrom": fitted_alphas[stretch_indices].tolist(),
        "semidiagonal_cubic_rmse_before": initial_rmse,
        "semidiagonal_cubic_rmse_after": fitted_rmse,
        "full_cubic_relative_error_before": float(np.linalg.norm(current_cubic - reference_cubic) / np.linalg.norm(reference_cubic)),
        "full_cubic_relative_error_after": float(np.linalg.norm(fitted_cubic - reference_cubic) / np.linalg.norm(reference_cubic)),
        "parent_deltavib_MHz": [-float(value) for value in deltavib.delta_MHz],
        "reference_energy_hartree": new_eval.energy_hartree,
        "reference_gradient_max_change_hartree_per_bohr": float(np.max(np.abs(new_eval.gradient_hartree_per_bohr - old_eval.gradient_hartree_per_bohr))),
        "reference_hessian_max_change_hartree_per_bohr2": float(np.max(np.abs(new_eval.hessian_hartree_per_bohr2 - old_eval.hessian_hartree_per_bohr2))),
    }
    save_force_field(request.output_field_path, refined)
    Path(request.output_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(request.output_report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


__all__ = ["MorseCubicRefitRequest", "refit_gfn2_zaff_morse_cubics"]
