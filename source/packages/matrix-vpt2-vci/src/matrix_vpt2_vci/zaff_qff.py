"""TRINITY projection of sampled ZAFF E/G data onto a normal-mode QFF."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations_with_replacement
import json
from math import ceil, factorial
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .vci import QuarticForceField


CM_PER_HARTREE = 219474.6313705
ZAFF_NORMAL_QFF_SCHEMA = "matrix.trinity.zaff_normal_qff.v1"


@dataclass(frozen=True)
class NormalModeQFFFitConfig:
    """Sampling and least-squares controls in dimensionless normal coordinates."""

    amplitude: float = 0.03
    training_pairs: int = 0
    holdout_pairs: int = 8
    oversampling: float = 1.25
    seed: int = 20260716
    energy_weight: float = 1.0
    gradient_weight: float = 1.0
    coefficient_threshold_cm: float = 1.0e-8
    workers: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in self.__dict__.items()
        }

    def validate(self) -> None:
        if not np.isfinite(self.amplitude) or self.amplitude <= 0.0:
            raise ValueError("normal-mode sampling amplitude must be positive")
        if self.training_pairs < 0 or self.holdout_pairs < 1:
            raise ValueError("QFF training pairs must be non-negative and holdout pairs positive")
        if not np.isfinite(self.oversampling) or self.oversampling < 1.0:
            raise ValueError("QFF oversampling must be at least one")
        if self.energy_weight <= 0.0 or self.gradient_weight <= 0.0:
            raise ValueError("QFF energy and gradient weights must be positive")
        if self.coefficient_threshold_cm < 0.0 or self.workers < 0:
            raise ValueError("QFF coefficient threshold and worker count cannot be negative")


@dataclass(frozen=True)
class NormalModeQFFDiagnostics:
    parameter_count: int
    cubic_count: int
    semidiagonal_quartic_count: int
    training_point_count: int
    holdout_point_count: int
    fit_rank: int
    condition_number: float
    training_energy_rms_hartree: float
    training_gradient_rms_hartree: float
    holdout_energy_rms_hartree: float
    holdout_gradient_rms_hartree: float
    maximum_holdout_energy_error_hartree: float
    maximum_holdout_gradient_error_hartree: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class NormalModeQFFFitResult:
    force_field: QuarticForceField
    diagnostics: NormalModeQFFDiagnostics
    displacement_bohr_per_dimensionless: np.ndarray
    config: NormalModeQFFFitConfig = NormalModeQFFFitConfig()
    source: str = "ARCHITECT/ZAFF_LOCAL_SONIC_TAYLOR"
    schema: str = ZAFF_NORMAL_QFF_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "owner": "TRINITY",
            "source": self.source,
            "normal_coordinate_convention": "DIMENSIONLESS_HARMONIC_OSCILLATOR",
            "sampling_and_fit": {
                **self.config.to_dict(),
                "point_distribution": "COMPONENTWISE_UNIFORM_TIMES_UNIFORM_RADIAL_SCALE",
                "component_interval": [
                    -float(self.config.amplitude),
                    float(self.config.amplitude),
                ],
                "radial_scale_interval": [0.55, 1.0],
                "parity_pairing": "EXACT_PLUS_MINUS_PAIRS",
                "holdout_seed": int(self.config.seed) + 104729,
                "least_squares": "COLUMN_NORMALIZED_SVD_RCOND_1E-12",
            },
            "harmonic_frequencies_cm-1": np.asarray(
                self.force_field.harmonic_frequencies_cm, dtype=float
            ).tolist(),
            "cubic_terms": [
                {"indices_zero_based": list(indices), "value_cm-1": value}
                for indices, value in sorted(self.force_field.cubic_cm.items())
            ],
            "semidiagonal_quartic_terms": [
                {"indices_zero_based": list(indices), "value_cm-1": value}
                for indices, value in sorted(self.force_field.quartic_cm.items())
            ],
            "cartesian_displacement_bohr_per_dimensionless": np.asarray(
                self.displacement_bohr_per_dimensionless, dtype=float
            ).tolist(),
            "diagnostics": self.diagnostics.to_dict(),
        }


EnergyGradientEvaluator = Callable[[np.ndarray], tuple[float, np.ndarray]]


def full_cubic_indices(mode_count: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(combinations_with_replacement(range(mode_count), 3))


def semidiagonal_quartic_indices(mode_count: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return quartics containing one or two distinct mode indices."""

    indices: list[tuple[int, int, int, int]] = []
    for i in range(mode_count):
        indices.append((i, i, i, i))
        for j in range(i + 1, mode_count):
            indices.extend(((i, i, i, j), (i, i, j, j), (i, j, j, j)))
    return tuple(indices)


def dimensionless_normal_displacements(
    frequencies_cm: np.ndarray,
    eigenvalues: np.ndarray,
    mass_weighted_modes: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Return Cartesian bohr displacements for unit dimensionless normal coordinates."""

    frequencies = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    eigenvalues = np.asarray(eigenvalues, dtype=float).reshape(-1)
    modes = np.asarray(mass_weighted_modes, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    if modes.shape != (frequencies.size, masses.size, 3):
        raise ValueError("mass-weighted normal-mode dimensions are inconsistent")
    if eigenvalues.shape != frequencies.shape or np.any(eigenvalues <= 0.0):
        raise ValueError("ZAFF QFF projection requires a positive-definite vibrational minimum")
    if np.any(frequencies <= 0.0) or np.any(masses <= 0.0):
        raise ValueError("normal frequencies and masses must be positive")
    scale = np.sqrt((frequencies / CM_PER_HARTREE) / eigenvalues)
    cartesian = modes / np.sqrt(masses)[None, :, None]
    return cartesian.reshape((frequencies.size, -1)).T * scale[None, :]


def fit_normal_mode_qff(
    *,
    reference_coordinates_angstrom: np.ndarray,
    frequencies_cm: np.ndarray,
    eigenvalues: np.ndarray,
    mass_weighted_modes: np.ndarray,
    masses_amu: np.ndarray,
    evaluator: EnergyGradientEvaluator,
    config: NormalModeQFFFitConfig = NormalModeQFFFitConfig(),
    source: str = "ENERGY_GRADIENT_EVALUATOR",
) -> NormalModeQFFFitResult:
    """Fit full cubic and semi-diagonal quartic normal-mode derivatives.

    The harmonic frequencies are held fixed.  Energies and projected gradients
    are fitted simultaneously, so neither displaced Hessians nor derivatives of
    the internal-coordinate Wilson matrix are required.
    """

    config.validate()
    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    frequencies = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    mode_count = frequencies.size
    if reference.ndim != 2 or reference.shape[1] != 3 or mode_count < 1:
        raise ValueError("QFF projection needs a molecular geometry and at least one mode")
    displacement_matrix = dimensionless_normal_displacements(
        frequencies, eigenvalues, mass_weighted_modes, masses_amu
    )
    cubic_indices = full_cubic_indices(mode_count)
    quartic_indices = semidiagonal_quartic_indices(mode_count)
    term_indices: tuple[tuple[int, ...], ...] = (*cubic_indices, *quartic_indices)
    parameter_count = len(term_indices)
    # For paired +/-Q sampling, cubic-gradient rows are even and quartic-gradient
    # rows are odd.  The two parity blocks must each be independently overdetermined;
    # counting only the total number of energy-plus-gradient rows underestimates the
    # directions required by the much larger full-cubic block.
    minimum_pairs = ceil(
        config.oversampling * max(len(cubic_indices), len(quartic_indices)) / mode_count
    )
    training_pairs = config.training_pairs or max(2, minimum_pairs)
    training_points = _paired_points(mode_count, training_pairs, config.amplitude, config.seed)
    holdout_points = _paired_points(
        mode_count,
        config.holdout_pairs,
        config.amplitude,
        config.seed + 104729,
    )

    reference_energy, reference_gradient = _normalize_evaluation(evaluator(reference.copy()))
    reference_gradient_q = displacement_matrix.T @ reference_gradient
    harmonic = frequencies / CM_PER_HARTREE
    workers = _resolved_workers(config.workers, len(training_points) + len(holdout_points))
    all_points = (*training_points, *holdout_points)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        evaluations = tuple(
            pool.map(
                lambda point: _evaluate_point(
                    point,
                    reference,
                    displacement_matrix,
                    evaluator,
                ),
                all_points,
            )
        )
    training_evaluations = evaluations[: len(training_points)]
    holdout_evaluations = evaluations[len(training_points) :]

    design, target, row_kinds = _fit_system(
        training_evaluations,
        term_indices,
        harmonic,
        reference_energy,
        reference_gradient_q,
        config,
    )
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("normal-mode QFF design contains an unobservable coefficient")
    scaled = design / norms[None, :]
    scaled_coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        scaled, target, rcond=1.0e-12
    )
    if int(rank) != parameter_count:
        raise ValueError(
            f"normal-mode QFF fit is rank deficient ({int(rank)}/{parameter_count}); "
            "increase --zaff-fit-pairs or change the seed"
        )
    coefficients = scaled_coefficients / norms
    training_prediction = design @ coefficients
    training_errors = training_prediction - target
    holdout_errors, holdout_kinds = _validation_errors(
        holdout_evaluations,
        term_indices,
        coefficients,
        harmonic,
        reference_energy,
        reference_gradient_q,
    )
    cubic_values = coefficients[: len(cubic_indices)] * CM_PER_HARTREE
    quartic_values = coefficients[len(cubic_indices) :] * CM_PER_HARTREE
    threshold = config.coefficient_threshold_cm
    qff = QuarticForceField(
        harmonic_frequencies_cm=frequencies.copy(),
        cubic_cm={
            indices: float(value)
            for indices, value in zip(cubic_indices, cubic_values, strict=True)
            if abs(float(value)) >= threshold
        },
        quartic_cm={
            indices: float(value)
            for indices, value in zip(quartic_indices, quartic_values, strict=True)
            if abs(float(value)) >= threshold
        },
    )
    diagnostics = NormalModeQFFDiagnostics(
        parameter_count=parameter_count,
        cubic_count=len(cubic_indices),
        semidiagonal_quartic_count=len(quartic_indices),
        training_point_count=len(training_points),
        holdout_point_count=len(holdout_points),
        fit_rank=int(rank),
        condition_number=float(singular_values[0] / singular_values[-1]),
        training_energy_rms_hartree=_kind_rms(training_errors, row_kinds, "energy"),
        training_gradient_rms_hartree=_kind_rms(training_errors, row_kinds, "gradient"),
        holdout_energy_rms_hartree=_kind_rms(holdout_errors, holdout_kinds, "energy"),
        holdout_gradient_rms_hartree=_kind_rms(holdout_errors, holdout_kinds, "gradient"),
        maximum_holdout_energy_error_hartree=_kind_max(holdout_errors, holdout_kinds, "energy"),
        maximum_holdout_gradient_error_hartree=_kind_max(holdout_errors, holdout_kinds, "gradient"),
    )
    return NormalModeQFFFitResult(
        force_field=qff,
        diagnostics=diagnostics,
        displacement_bohr_per_dimensionless=displacement_matrix,
        config=config,
        source=source,
    )


def fit_normal_mode_qff_from_zaff(
    xyzin: Path | str,
    force_field_path: Path | str,
    *,
    config: NormalModeQFFFitConfig = NormalModeQFFFitConfig(),
) -> NormalModeQFFFitResult:
    """Build the GF normal modes and project a compiled ZAFF SONIC field."""

    from matrix_chem import BOHR_TO_ANGSTROM, Structure
    from matrix_gf import cartesian_normal_modes_from_hessian
    from matrix_zaff import load_zaff_artifact
    from matrix_zaff.sonic_runtime import ZaffSonicTaylorRuntime

    force_field = load_zaff_artifact(force_field_path)
    runtime = ZaffSonicTaylorRuntime(force_field, xyzin)
    structure = Structure(
        list(force_field.atoms),
        [tuple(row) for row in force_field.reference_coordinates_angstrom],
    )
    masses = np.asarray(structure.mass_isotope, dtype=float)
    modes = cartesian_normal_modes_from_hessian(
        runtime.harmonic_cartesian_hessian_hartree_per_bohr2,
        masses,
        force_field.reference_coordinates_angstrom / BOHR_TO_ANGSTROM,
        source="ARCHITECT/ZAFF local-SONIC GF",
    )

    def evaluate(coordinates: np.ndarray) -> tuple[float, np.ndarray]:
        result = runtime.evaluate(coordinates)
        return float(result.energy_hartree), np.asarray(
            result.gradient_hartree_per_bohr, dtype=float
        )

    return fit_normal_mode_qff(
        reference_coordinates_angstrom=force_field.reference_coordinates_angstrom,
        frequencies_cm=modes.frequencies_cm,
        eigenvalues=modes.eigenvalues,
        mass_weighted_modes=modes.mass_weighted_modes,
        masses_amu=masses,
        evaluator=evaluate,
        config=config,
        source="ARCHITECT/ZAFF_LOCAL_SONIC_TAYLOR",
    )


def write_normal_mode_qff_result(result: NormalModeQFFFitResult, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def write_indexed_qff(result: NormalModeQFFFitResult, path: Path | str) -> Path:
    """Write the existing one-based MATRIX indexed-QFF interchange format."""

    rows = ["# TRINITY QFF projected from ARCHITECT/ZAFF energies and gradients"]
    rows.extend(
        f"FREQ {index} {value:.16g}"
        for index, value in enumerate(result.force_field.harmonic_frequencies_cm, start=1)
    )
    rows.extend(
        "CUBIC " + " ".join(str(index + 1) for index in indices) + f" {value:.16g}"
        for indices, value in sorted(result.force_field.cubic_cm.items())
    )
    rows.extend(
        "QUARTIC " + " ".join(str(index + 1) for index in indices) + f" {value:.16g}"
        for indices, value in sorted(result.force_field.quartic_cm.items())
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target


def compare_reduced_normal_qff(
    candidate: QuarticForceField,
    reference_cubic_cm: Mapping[tuple[int, int, int], float],
    reference_quartic_cm: Mapping[tuple[int, int, int, int], float],
    *,
    reference_frequencies_cm: np.ndarray,
    reference_to_candidate_signs: np.ndarray | None = None,
) -> dict[str, object]:
    """Compare reduced QFF tensors after an explicit normal-mode gauge match.

    Eigenvectors are defined only up to sign.  ``reference_to_candidate_signs``
    records the sign of the reference/candidate mode overlap; a rank-n force
    constant is multiplied by the product of the n participating signs before
    comparison.  Gaussian terms omitted by its print threshold are treated as
    zero, which is below the precision relevant to this benchmark.
    """

    frequencies = np.asarray(candidate.harmonic_frequencies_cm, dtype=float)
    reference_frequencies = np.asarray(reference_frequencies_cm, dtype=float)
    if frequencies.shape != reference_frequencies.shape:
        raise ValueError("candidate and reference QFF dimensions differ")
    signs = (
        np.ones(frequencies.size, dtype=float)
        if reference_to_candidate_signs is None
        else np.asarray(reference_to_candidate_signs, dtype=float).reshape(-1)
    )
    if signs.shape != frequencies.shape or np.any(np.abs(signs) != 1.0):
        raise ValueError("normal-mode gauge signs must contain one +/-1 per mode")

    cubic_indices = full_cubic_indices(frequencies.size)
    quartic_indices = semidiagonal_quartic_indices(frequencies.size)
    cubic = _qff_error_block(
        cubic_indices, candidate.cubic_cm, reference_cubic_cm, signs
    )
    quartic = _qff_error_block(
        quartic_indices, candidate.quartic_cm, reference_quartic_cm, signs
    )
    diagonal_cubic = _qff_error_block(
        tuple((i, i, i) for i in range(frequencies.size)),
        candidate.cubic_cm,
        reference_cubic_cm,
        signs,
    )
    diagonal_quartic = _qff_error_block(
        tuple((i, i, i, i) for i in range(frequencies.size)),
        candidate.quartic_cm,
        reference_quartic_cm,
        signs,
    )
    frequency_error = frequencies - reference_frequencies
    return {
        "normal_mode_gauge": "REFERENCE_COEFFICIENT_TIMES_PRODUCT_OF_MODE_OVERLAP_SIGNS",
        "frequency_rms_cm-1": float(np.sqrt(np.mean(frequency_error**2))),
        "frequency_max_abs_cm-1": float(np.max(np.abs(frequency_error))),
        "full_cubic": cubic,
        "diagonal_cubic": diagonal_cubic,
        "semidiagonal_quartic": quartic,
        "diagonal_quartic": diagonal_quartic,
    }


def _qff_error_block(indices, candidate, reference, signs) -> dict[str, float | int]:
    candidate_values = np.asarray([candidate.get(key, 0.0) for key in indices], dtype=float)
    reference_values = np.asarray(
        [reference.get(key, 0.0) * np.prod(signs[list(key)]) for key in indices],
        dtype=float,
    )
    errors = candidate_values - reference_values
    norm = float(np.linalg.norm(reference_values))
    centered_candidate = candidate_values - np.mean(candidate_values)
    centered_reference = reference_values - np.mean(reference_values)
    denominator = float(np.linalg.norm(centered_candidate) * np.linalg.norm(centered_reference))
    return {
        "term_count": len(indices),
        "rms_error_cm-1": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_cm-1": float(np.max(np.abs(errors))),
        "relative_l2_error": float(np.linalg.norm(errors) / norm) if norm else 0.0,
        "correlation": float(np.dot(centered_candidate, centered_reference) / denominator)
        if denominator
        else 0.0,
    }


def _paired_points(
    mode_count: int, pair_count: int, amplitude: float, seed: int
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    for _ in range(pair_count):
        point = rng.uniform(-amplitude, amplitude, size=mode_count)
        radial_scale = rng.uniform(0.55, 1.0)
        point *= radial_scale
        points.extend((point, -point))
    return tuple(points)


def _evaluate_point(
    point: np.ndarray,
    reference_angstrom: np.ndarray,
    displacement_bohr: np.ndarray,
    evaluator: EnergyGradientEvaluator,
) -> tuple[np.ndarray, float, np.ndarray]:
    from matrix_chem import BOHR_TO_ANGSTROM

    cartesian_bohr = displacement_bohr @ point
    geometry = (
        reference_angstrom + cartesian_bohr.reshape(reference_angstrom.shape) * BOHR_TO_ANGSTROM
    )
    energy, gradient = _normalize_evaluation(evaluator(geometry))
    return point, energy, displacement_bohr.T @ gradient


def _normalize_evaluation(value: tuple[float, np.ndarray]) -> tuple[float, np.ndarray]:
    energy, gradient = value
    energy_value = float(energy)
    gradient_value = np.asarray(gradient, dtype=float).reshape(-1)
    if not np.isfinite(energy_value) or not np.all(np.isfinite(gradient_value)):
        raise ValueError("QFF evaluator returned non-finite energy or gradient")
    return energy_value, gradient_value


def _fit_system(
    evaluations: tuple[tuple[np.ndarray, float, np.ndarray], ...],
    terms: tuple[tuple[int, ...], ...],
    harmonic: np.ndarray,
    reference_energy: float,
    reference_gradient: np.ndarray,
    config: NormalModeQFFFitConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    kinds: list[str] = []
    for point, energy, gradient in evaluations:
        energy_design, gradient_design = _design_rows(point, terms)
        energy_target = (
            energy
            - reference_energy
            - float(reference_gradient @ point)
            - 0.5 * float(np.dot(harmonic, point**2))
        )
        gradient_target = gradient - reference_gradient - harmonic * point
        rows.append(config.energy_weight * energy_design)
        targets.append(config.energy_weight * energy_target)
        kinds.append("energy")
        rows.extend(config.gradient_weight * gradient_design)
        targets.extend(config.gradient_weight * gradient_target)
        kinds.extend(("gradient",) * point.size)
    return np.vstack(rows), np.asarray(targets, dtype=float), tuple(kinds)


def _validation_errors(
    evaluations: tuple[tuple[np.ndarray, float, np.ndarray], ...],
    terms: tuple[tuple[int, ...], ...],
    coefficients: np.ndarray,
    harmonic: np.ndarray,
    reference_energy: float,
    reference_gradient: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    errors: list[float] = []
    kinds: list[str] = []
    for point, energy, gradient in evaluations:
        energy_design, gradient_design = _design_rows(point, terms)
        predicted_energy = (
            reference_energy
            + float(reference_gradient @ point)
            + 0.5 * float(np.dot(harmonic, point**2))
            + float(energy_design @ coefficients)
        )
        predicted_gradient = reference_gradient + harmonic * point + gradient_design @ coefficients
        errors.append(predicted_energy - energy)
        kinds.append("energy")
        errors.extend(predicted_gradient - gradient)
        kinds.extend(("gradient",) * point.size)
    return np.asarray(errors, dtype=float), tuple(kinds)


def _design_rows(
    point: np.ndarray, terms: tuple[tuple[int, ...], ...]
) -> tuple[np.ndarray, np.ndarray]:
    mode_count = point.size
    energy = np.empty(len(terms), dtype=float)
    gradient = np.empty((mode_count, len(terms)), dtype=float)
    for column, indices in enumerate(terms):
        counts = np.bincount(indices, minlength=mode_count)
        energy[column] = _monomial_with_factorials(point, counts)
        for mode in range(mode_count):
            if counts[mode] == 0:
                gradient[mode, column] = 0.0
            else:
                derivative_counts = counts.copy()
                derivative_counts[mode] -= 1
                gradient[mode, column] = _monomial_with_factorials(point, derivative_counts)
    return energy, gradient


def _monomial_with_factorials(point: np.ndarray, counts: np.ndarray) -> float:
    value = 1.0
    for coordinate, power in zip(point, counts, strict=True):
        if power:
            value *= float(coordinate) ** int(power) / factorial(int(power))
    return value


def _kind_rms(errors: np.ndarray, kinds: tuple[str, ...], target: str) -> float:
    selected = errors[np.asarray([kind == target for kind in kinds], dtype=bool)]
    return float(np.sqrt(np.mean(selected**2))) if selected.size else 0.0


def _kind_max(errors: np.ndarray, kinds: tuple[str, ...], target: str) -> float:
    selected = errors[np.asarray([kind == target for kind in kinds], dtype=bool)]
    return float(np.max(np.abs(selected))) if selected.size else 0.0


def _resolved_workers(requested: int, task_count: int) -> int:
    import os

    available = max(1, int(os.cpu_count() or 1))
    return min(task_count, requested if requested > 0 else min(available, 32))
