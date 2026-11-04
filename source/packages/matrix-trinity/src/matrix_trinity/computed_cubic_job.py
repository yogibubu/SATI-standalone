"""Persistent, auditable SONIC F2/F3 jobs for isotope-specific MORPHEUS fits."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Callable

import numpy as np

from .curvilinear_rovib import CurvilinearDeltaBVibJob
from .isotopic_internal_qff import NonredundantInternalCubicField
from .isotopic_internal_qff import internal_cubic_from_parent_normal_field


COMPUTED_CUBIC_JOB_SCHEMA = "matrix.trinity.computed_cubic_job.v1"


def acquire_computed_cubic_job_from_hessians(
    *,
    symbols: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    cartesian_hessian: np.ndarray,
    b_matrix_per_bohr: np.ndarray,
    b_prime_per_bohr2: np.ndarray,
    evaluator: Callable[[np.ndarray], object],
    step_sqrt_amu_bohr: float = 0.02,
) -> tuple[CurvilinearDeltaBVibJob, dict[str, object]]:
    """Acquire full F3 from centered analytic-Hessian stencils in parent modes."""

    from matrix_chem import BOHR_TO_ANGSTROM
    from matrix_qm import cartesian_normal_modes_from_hessian

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    hessian = np.asarray(cartesian_hessian, dtype=float)
    hessian = 0.5 * (hessian + hessian.T)
    step = float(step_sqrt_amu_bohr)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("TRINITY Hessian stencil step must be positive and finite")
    modes = cartesian_normal_modes_from_hessian(
        hessian, masses, xyz / BOHR_TO_ANGSTROM,
        source="TRINITY same-method analytic-Hessian acquisition",
    )
    if np.any(modes.frequencies_cm <= 0.0):
        raise ValueError("TRINITY Hessian stencil requires a vibrational minimum")
    n = len(modes.frequencies_cm)
    c = modes.mass_weighted_modes.reshape((n, -1)).T / np.sqrt(
        np.repeat(masses, 3)
    )[:, None]

    def evaluate_hessian(at_xyz: np.ndarray) -> np.ndarray:
        result = evaluator(at_xyz)
        value = np.asarray(result.hessian_hartree_per_bohr2, dtype=float)
        if value.shape != hessian.shape or not np.all(np.isfinite(value)):
            raise ValueError("same-method evaluator did not return a finite analytic Hessian")
        return 0.5 * (value + value.T)

    reference = evaluate_hessian(xyz)
    reference_error = float(
        np.linalg.norm(reference - hessian) / max(np.linalg.norm(hessian), 1e-15)
    )
    if reference_error > 1e-8:
        raise ValueError("same-method stencil Hessian differs from the frozen reference")

    def acquire_at(h: float) -> np.ndarray:
        cubic = np.empty((n, n, n), dtype=float)
        for mode in range(n):
            displacement = (c[:, mode] * h * BOHR_TO_ANGSTROM).reshape(xyz.shape)
            plus = evaluate_hessian(xyz + displacement)
            minus = evaluate_hessian(xyz - displacement)
            cubic[:, :, mode] = c.T @ ((plus - minus) / (2.0 * h)) @ c
        return cubic

    coarse = acquire_at(step)
    fine = acquire_at(step / 2.0)
    richardson = (4.0 * fine - coarse) / 3.0
    permutations_ = (
        richardson,
        richardson.transpose(0, 2, 1),
        richardson.transpose(1, 0, 2),
        richardson.transpose(1, 2, 0),
        richardson.transpose(2, 0, 1),
        richardson.transpose(2, 1, 0),
    )
    cubic = sum(permutations_) / 6.0
    scale = max(float(np.linalg.norm(cubic)), 1e-15)
    evidence = {
        "source": "TRINITY_ANALYTIC_HESSIAN_STENCIL",
        "mode_count": n,
        "step_sqrt_amu_bohr": step,
        "reference_hessian_relative_error": reference_error,
        "step_relative_difference": float(np.linalg.norm(fine - coarse) / scale),
        "raw_symmetry_residual": float(np.linalg.norm(richardson - cubic) / scale),
        "hessian_evaluations": 4 * n + 1,
    }
    field = internal_cubic_from_parent_normal_field(
        np.diag(modes.eigenvalues), cubic,
        b_matrix_per_bohr, b_prime_per_bohr2, c,
    )
    return CurvilinearDeltaBVibJob(
        field, np.asarray(b_matrix_per_bohr, dtype=float),
        np.asarray(b_prime_per_bohr2, dtype=float), tuple(symbols), xyz,
        source="TRINITY same-method analytic-Hessian stencil in nonredundant SONIC F2/F3",
    ), evidence


def acquire_computed_cubic_job(
    *,
    symbols: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    cartesian_hessian: np.ndarray,
    b_matrix_per_bohr: np.ndarray,
    b_prime_per_bohr2: np.ndarray,
    evaluator: Callable[[np.ndarray], tuple[float, np.ndarray]],
    config=None,
) -> tuple[CurvilinearDeltaBVibJob, object]:
    """Sample one E/G method and transform its full parent cubic field to SONIC.

    The same method must supply the Hessian and every E/G stencil point.  The
    caller binds the resulting artifact to that method's frozen source hash.
    """

    from matrix_chem import BOHR_TO_ANGSTROM
    from matrix_qm import cartesian_normal_modes_from_hessian
    from matrix_vpt2_vci.zaff_qff import CM_PER_HARTREE, NormalModeQFFFitConfig

    from .anharmonic_fields import build_normal_mode_anharmonic_field

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    modes = cartesian_normal_modes_from_hessian(
        np.asarray(cartesian_hessian, dtype=float), masses,
        xyz / BOHR_TO_ANGSTROM,
        source="TRINITY same-method F2/F3 acquisition",
    )
    frequencies = np.asarray(modes.frequencies_cm, dtype=float)
    if np.any(frequencies <= 0.0):
        raise ValueError("computed TRINITY F2/F3 requires a vibrational minimum")
    qff = build_normal_mode_anharmonic_field(
        reference_coordinates_angstrom=xyz,
        frequencies_cm=frequencies,
        eigenvalues=np.asarray(modes.eigenvalues, dtype=float),
        mass_weighted_modes=np.asarray(modes.mass_weighted_modes, dtype=float),
        masses_amu=masses,
        evaluator=evaluator,
        config=NormalModeQFFFitConfig(workers=1) if config is None else config,
    )
    n = frequencies.size
    cubic = np.zeros((n, n, n), dtype=float)
    scale = np.sqrt((frequencies / CM_PER_HARTREE) / modes.eigenvalues)
    for indices, value in qff.force_field.cubic_cm.items():
        derivative = float(value) / (CM_PER_HARTREE * np.prod(scale[list(indices)]))
        for permuted in set(permutations(indices)):
            cubic[permuted] = derivative
    cartesian_per_q = (
        modes.mass_weighted_modes.reshape((n, -1)).T
        / np.sqrt(np.repeat(masses, 3))[:, None]
    )
    field = internal_cubic_from_parent_normal_field(
        np.diag(modes.eigenvalues), cubic,
        b_matrix_per_bohr, b_prime_per_bohr2, cartesian_per_q,
    )
    return CurvilinearDeltaBVibJob(
        field,
        np.asarray(b_matrix_per_bohr, dtype=float),
        np.asarray(b_prime_per_bohr2, dtype=float),
        tuple(symbols), xyz,
        source="TRINITY same-method E/G QFF transformed to nonredundant SONIC F2/F3",
    ), qff


def computed_cubic_job_to_dict(
    job: CurvilinearDeltaBVibJob,
    *,
    method: str,
    method_artifact_sha256: str,
    hessian_sha256: str,
    qff_result=None,
    acquisition_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Capture a complete field and the exact derivative source used to acquire it."""

    if not method.strip() or not all(
        len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        for value in (method_artifact_sha256, hessian_sha256)
    ):
        raise ValueError("computed F2/F3 requires method and SHA-256 input identities")
    field = job.field
    n = field.coordinate_count
    b = np.asarray(job.b_matrix, dtype=float)
    bp = np.asarray(job.b_prime, dtype=float)
    f2 = np.asarray(field.harmonic_internal, dtype=float)
    f3 = np.asarray(field.cubic_internal, dtype=float)
    xyz = np.asarray(job.coordinates_angstrom, dtype=float)
    if (
        b.shape != (n, 3 * len(job.symbols))
        or bp.shape != (n, b.shape[1], b.shape[1])
        or f2.shape != (n, n)
        or f3.shape != (n, n, n)
        or xyz.shape != (len(job.symbols), 3)
        or not all(np.all(np.isfinite(x)) for x in (b, bp, f2, f3, xyz))
        or not np.allclose(f2, f2.T, atol=1e-9)
        or not np.allclose(f3, f3.transpose(1, 0, 2), atol=1e-9)
        or not np.allclose(f3, f3.transpose(0, 2, 1), atol=1e-9)
        or not np.allclose(bp, bp.transpose(0, 2, 1), atol=1e-6)
        or np.linalg.matrix_rank(b) != n
        or not np.isfinite(field.parent_d_condition)
        or field.parent_d_condition < 1.0
        or not job.source.strip()
    ):
        raise ValueError("computed F2/F3 has invalid dimensions, symmetry or SONIC rank")
    if (qff_result is None) == (acquisition_evidence is None):
        raise ValueError("provide exactly one TRINITY F3 acquisition evidence source")
    evidence = (
        {
            "source": qff_result.source,
            "diagnostics": qff_result.diagnostics.to_dict(),
            "sampling": qff_result.config.to_dict(),
        }
        if qff_result is not None else dict(acquisition_evidence)
    )
    return {
        "schema": COMPUTED_CUBIC_JOB_SCHEMA,
        "method": method,
        "method_artifact_sha256": method_artifact_sha256,
        "hessian_sha256": hessian_sha256,
        "source": job.source,
        "symbols": list(job.symbols),
        "coordinates_angstrom": xyz.tolist(),
        "b_matrix_per_bohr": b.tolist(),
        "b_prime_per_bohr2": bp.tolist(),
        "harmonic_internal": f2.tolist(),
        "cubic_internal": f3.tolist(),
        "parent_d_condition": float(field.parent_d_condition),
        "coordinate_count": n,
        "acquisition_evidence": evidence,
    }


def computed_cubic_job_from_dict(payload: dict[str, object]) -> CurvilinearDeltaBVibJob:
    """Restore only a complete, nonredundant TRINITY field."""

    if payload.get("schema") != COMPUTED_CUBIC_JOB_SCHEMA:
        raise ValueError("unsupported computed TRINITY F2/F3 schema")
    evidence = payload.get("acquisition_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("computed TRINITY F2/F3 lacks same-method acquisition evidence")
    if evidence.get("source") == "TRINITY_DIRECT_ENERGY_GRADIENT":
        diagnostics = evidence.get("diagnostics")
        if (
            not isinstance(diagnostics, dict)
            or diagnostics.get("fit_rank") != diagnostics.get("parameter_count")
            or not isinstance(diagnostics.get("fit_rank"), int)
            or diagnostics["fit_rank"] < 1
        ):
            raise ValueError("computed TRINITY F2/F3 lacks a full-rank QFF fit")
        for name in (
            "maximum_holdout_energy_error_hartree",
            "maximum_holdout_gradient_error_hartree",
        ):
            value = diagnostics.get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("computed TRINITY F2/F3 lacks finite holdout evidence")
    elif evidence.get("source") == "TRINITY_ANALYTIC_HESSIAN_STENCIL":
        count = evidence.get("mode_count")
        if (
            not isinstance(count, int)
            or count != int(payload.get("coordinate_count", -1))
            or evidence.get("hessian_evaluations") != 4 * count + 1
        ):
            raise ValueError("computed TRINITY Hessian stencil is incomplete")
        for name in (
            "step_sqrt_amu_bohr",
            "reference_hessian_relative_error",
            "step_relative_difference",
            "raw_symmetry_residual",
        ):
            value = evidence.get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError("computed TRINITY Hessian stencil evidence is invalid")
        if evidence["step_sqrt_amu_bohr"] == 0 or evidence["reference_hessian_relative_error"] > 1e-8:
            raise ValueError("computed TRINITY Hessian stencil does not share its reference")
    else:
        raise ValueError("computed TRINITY F2/F3 lacks same-method acquisition evidence")
    field = NonredundantInternalCubicField(
        np.asarray(payload["harmonic_internal"], dtype=float),
        np.asarray(payload["cubic_internal"], dtype=float),
        float(payload["parent_d_condition"]),
        int(payload["coordinate_count"]),
    )
    job = CurvilinearDeltaBVibJob(
        field,
        np.asarray(payload["b_matrix_per_bohr"], dtype=float),
        np.asarray(payload["b_prime_per_bohr2"], dtype=float),
        tuple(str(item) for item in payload["symbols"]),
        np.asarray(payload["coordinates_angstrom"], dtype=float),
        source=str(payload["source"]),
    )
    computed_cubic_job_to_dict(
        job,
        method=str(payload["method"]),
        method_artifact_sha256=str(payload["method_artifact_sha256"]),
        hessian_sha256=str(payload["hessian_sha256"]),
        acquisition_evidence=evidence,
    )
    return job


def read_computed_cubic_job(path: Path | str) -> tuple[CurvilinearDeltaBVibJob, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("computed TRINITY job must be a JSON object")
    return computed_cubic_job_from_dict(payload), payload


def write_computed_cubic_job(
    path: Path | str,
    job: CurvilinearDeltaBVibJob,
    *,
    method: str,
    method_artifact_sha256: str,
    hessian_sha256: str,
    qff_result=None,
    acquisition_evidence: dict[str, object] | None = None,
) -> Path:
    """Persist the complete computed job consumed by the KeyMaker route."""

    payload = computed_cubic_job_to_dict(
        job,
        method=method,
        method_artifact_sha256=method_artifact_sha256,
        hessian_sha256=hessian_sha256,
        qff_result=qff_result,
        acquisition_evidence=acquisition_evidence,
    )
    computed_cubic_job_from_dict(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = [
    "COMPUTED_CUBIC_JOB_SCHEMA",
    "acquire_computed_cubic_job",
    "acquire_computed_cubic_job_from_hessians",
    "computed_cubic_job_from_dict",
    "computed_cubic_job_to_dict",
    "read_computed_cubic_job",
    "write_computed_cubic_job",
]
