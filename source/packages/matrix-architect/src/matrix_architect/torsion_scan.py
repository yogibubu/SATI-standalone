"""Fourier fitting and validation for economical ZAFF torsional prescans."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np


HARTREE_TO_KCAL_MOL = 627.5094740631


@dataclass(frozen=True)
class ZaffTorsionScanPlan:
    coordinate_index: int
    coordinate_identifier: str
    periodicity: int
    symmetry_number: int
    reference_value_radian: float
    phase_offsets_radian: tuple[float, ...]
    harmonics: tuple[int, ...]
    barrier_seed_kcal_mol: float
    barrier_seed_status: str
    escalation_policy: str = "PRESCAN_THEN_FULL_LEVEL_ON_FAILED_CURVATURE_OR_RESIDUAL"

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate_index": self.coordinate_index,
            "coordinate_identifier": self.coordinate_identifier,
            "periodicity": self.periodicity,
            "symmetry_number": self.symmetry_number,
            "reference_value_radian": self.reference_value_radian,
            "reference_value_degree": float(np.degrees(self.reference_value_radian)),
            "phase_offsets_radian": list(self.phase_offsets_radian),
            "phase_offsets_degree": [
                float(np.degrees(value)) for value in self.phase_offsets_radian
            ],
            "harmonics": list(self.harmonics),
            "barrier_seed_kcal_mol": self.barrier_seed_kcal_mol,
            "barrier_seed_status": self.barrier_seed_status,
            "escalation_policy": self.escalation_policy,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ZaffTorsionScanPlan":
        return cls(
            coordinate_index=int(payload["coordinate_index"]),
            coordinate_identifier=str(payload["coordinate_identifier"]),
            periodicity=int(payload["periodicity"]),
            symmetry_number=int(payload["symmetry_number"]),
            reference_value_radian=float(payload["reference_value_radian"]),
            phase_offsets_radian=tuple(float(value) for value in payload["phase_offsets_radian"]),
            harmonics=tuple(int(value) for value in payload["harmonics"]),
            barrier_seed_kcal_mol=float(payload["barrier_seed_kcal_mol"]),
            barrier_seed_status=str(payload["barrier_seed_status"]),
            escalation_policy=str(
                payload.get(
                    "escalation_policy",
                    "PRESCAN_THEN_FULL_LEVEL_ON_FAILED_CURVATURE_OR_RESIDUAL",
                )
            ),
        )


def build_zaff_torsion_scan_plans(
    periodic_coordinates: Sequence[object],
    anharmonic_coordinates: Sequence[object],
) -> tuple[ZaffTorsionScanPlan, ...]:
    """Convert SMITH torsion seeds into the production six-point prescan contract."""

    coordinate_by_label = {
        label: coordinate
        for coordinate in anharmonic_coordinates
        for label in (str(coordinate.identifier), str(coordinate.name))
    }
    plans = []
    for record in periodic_coordinates:
        if str(record.target) not in {"TORSION", "IMPROPER_DIHEDRAL"}:
            continue
        coordinate = coordinate_by_label.get(str(record.identifier)) or coordinate_by_label.get(
            str(record.name)
        )
        if coordinate is None:
            raise ValueError("periodic torsion has no matching frozen SONIC")
        symmetry = max(1, int(record.symmetry_number))
        span = 2.0 * pi / symmetry
        phases = tuple(float(value) for value in np.linspace(0.0, span, 6, endpoint=False))
        periodicity = max(1, int(record.periodicity))
        harmonics = tuple(sorted({1, 2, 3, periodicity}))
        plans.append(
            ZaffTorsionScanPlan(
                coordinate_index=int(coordinate.index),
                coordinate_identifier=str(coordinate.identifier),
                periodicity=periodicity,
                symmetry_number=symmetry,
                reference_value_radian=float(record.reference_value_radian or 0.0),
                phase_offsets_radian=phases,
                harmonics=harmonics,
                barrier_seed_kcal_mol=float(record.barrier_kcal_mol),
                barrier_seed_status=str(record.status),
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class ZaffTorsionFourierFit:
    harmonics: tuple[int, ...]
    constant: float
    cosine_coefficients: tuple[float, ...]
    sine_coefficients: tuple[float, ...]
    coordinate_scale_dq_dphi: float
    rmse_hartree: float
    maximum_residual_hartree: float
    local_gradient: float
    local_quadratic: float
    local_cubic: float
    local_quartic: float

    @property
    def coordinate_period(self) -> float:
        return 2.0 * pi * abs(self.coordinate_scale_dq_dphi)

    def energy(self, phase_offset_radian: float) -> float:
        phase = float(phase_offset_radian)
        return float(
            self.constant
            + sum(
                cosine * np.cos(harmonic * phase) + sine * np.sin(harmonic * phase)
                for harmonic, cosine, sine in zip(
                    self.harmonics,
                    self.cosine_coefficients,
                    self.sine_coefficients,
                    strict=True,
                )
            )
        )

    def phase_gradient(self, phase_offset_radian: float) -> float:
        """Derivative of the fitted energy with respect to the torsion angle."""

        phase = float(phase_offset_radian)
        return float(
            sum(
                harmonic * (
                    -cosine * np.sin(harmonic * phase)
                    + sine * np.cos(harmonic * phase)
                )
                for harmonic, cosine, sine in zip(
                    self.harmonics,
                    self.cosine_coefficients,
                    self.sine_coefficients,
                    strict=True,
                )
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "harmonics": list(self.harmonics),
            "constant_hartree": self.constant,
            "cosine_coefficients_hartree": list(self.cosine_coefficients),
            "sine_coefficients_hartree": list(self.sine_coefficients),
            "coordinate_scale_dq_dphi": self.coordinate_scale_dq_dphi,
            "coordinate_period": self.coordinate_period,
            "rmse_hartree": self.rmse_hartree,
            "maximum_residual_hartree": self.maximum_residual_hartree,
            "local_gradient": self.local_gradient,
            "local_quadratic": self.local_quadratic,
            "local_cubic": self.local_cubic,
            "local_quartic": self.local_quartic,
        }


@dataclass(frozen=True)
class ZaffTorsionPrescanControl:
    passed: bool
    reasons: tuple[str, ...]
    curvature_relative_error: float
    gradient_absolute_error: float
    fit_rmse_kcal_mol: float
    constrained_fit_rmse_kcal_mol: float | None
    barrier_kcal_mol: float

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "curvature_relative_error": self.curvature_relative_error,
            "gradient_absolute_error": self.gradient_absolute_error,
            "fit_rmse_kcal_mol": self.fit_rmse_kcal_mol,
            "constrained_fit_rmse_kcal_mol": self.constrained_fit_rmse_kcal_mol,
            "barrier_kcal_mol": self.barrier_kcal_mol,
        }


def fit_zaff_torsion_fourier(
    phase_offsets_radian: Sequence[float],
    energies_hartree: Sequence[float],
    *,
    coordinate_scale_dq_dphi: float,
    harmonics: Sequence[int] = (1, 2, 3),
    local_derivatives: tuple[float, float, float] | None = None,
) -> ZaffTorsionFourierFit:
    """Fit a phase-flexible Fourier series, optionally preserving local QM derivatives."""

    phase = np.asarray(phase_offsets_radian, dtype=float).reshape(-1)
    energy = np.asarray(energies_hartree, dtype=float).reshape(-1)
    modes = tuple(int(value) for value in harmonics)
    scale = float(coordinate_scale_dq_dphi)
    coefficient_count = 1 + 2 * len(modes)
    constraint_count = 0 if local_derivatives is None else 5
    minimum_points = max(2, coefficient_count - constraint_count)
    if phase.size != energy.size or phase.size < minimum_points:
        raise ValueError("torsional Fourier fit has insufficient paired scan points")
    if not modes or any(value <= 0 for value in modes) or len(set(modes)) != len(modes):
        raise ValueError("torsional Fourier harmonics must be unique positive integers")
    if not np.isfinite(scale) or abs(scale) < 1.0e-8:
        raise ValueError("torsional SONIC/phase scale must be finite and nonzero")
    if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(energy)):
        raise ValueError("torsional scan values must be finite")

    design = _design_matrix(phase, modes)
    relative_energy = energy - energy[0]
    if local_derivatives is None:
        coefficients, *_ = np.linalg.lstsq(design, relative_energy, rcond=1.0e-12)
    else:
        constraints, targets = _local_constraint_matrix(modes, scale, local_derivatives)
        normal = design.T @ design
        rhs = design.T @ relative_energy
        kkt = np.block(
            [
                [normal, constraints.T],
                [constraints, np.zeros((len(targets), len(targets)), dtype=float)],
            ]
        )
        solution, *_ = np.linalg.lstsq(
            kkt,
            np.concatenate((rhs, targets)),
            rcond=1.0e-12,
        )
        coefficients = solution[: design.shape[1]]

    predicted = design @ coefficients
    residual = predicted - relative_energy
    constant = float(coefficients[0])
    count = len(modes)
    cosine = np.asarray(coefficients[1 : count + 1], dtype=float)
    sine = np.asarray(coefficients[count + 1 :], dtype=float)
    gradient, quadratic, cubic, quartic = _local_derivatives(modes, cosine, sine, scale)
    return ZaffTorsionFourierFit(
        harmonics=modes,
        constant=constant,
        cosine_coefficients=tuple(float(value) for value in cosine),
        sine_coefficients=tuple(float(value) for value in sine),
        coordinate_scale_dq_dphi=scale,
        rmse_hartree=float(np.sqrt(np.mean(residual**2))),
        maximum_residual_hartree=float(np.max(np.abs(residual))),
        local_gradient=gradient,
        local_quadratic=quadratic,
        local_cubic=cubic,
        local_quartic=quartic,
    )


def fit_zaff_reflection_symmetric_torsion(
    angles_radian: Sequence[float],
    energies_hartree: Sequence[float],
    phase_gradients_hartree_per_radian: Sequence[float],
    *,
    minimum_angle_radian: float,
    minimum_energy_hartree: float,
    reference_curvature_hartree_per_radian2: float | None = None,
    curvature_angle_radian: float | None = None,
    coordinate_scale_dq_dphi: float = 1.0,
    maximum_harmonic: int | None = None,
) -> ZaffTorsionFourierFit:
    """Fit an even torsion from 0, 60, 120, 180 degrees and its minimum.

    Reflection symmetry must be established for the molecular torsion before
    this reduced acquisition is used. The interior phase gradients supply the
    information missing from the four energies; no odd Fourier terms are fitted.
    """

    phase = np.asarray(angles_radian, dtype=float).reshape(-1)
    energy = np.asarray(energies_hartree, dtype=float).reshape(-1)
    gradient = np.asarray(phase_gradients_hartree_per_radian, dtype=float).reshape(-1)
    expected = np.deg2rad((0.0, 60.0, 120.0, 180.0))
    if phase.shape != (4,) or not np.allclose(phase, expected, atol=1.0e-9, rtol=0.0):
        raise ValueError("reflection-symmetric torsion requires 0, 60, 120, 180 degrees")
    if energy.shape != (4,) or gradient.shape != (4,):
        raise ValueError("each torsion angle needs an energy and phase gradient")
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(gradient)):
        raise ValueError("torsion energies and phase gradients must be finite")
    minimum_angle = float(minimum_angle_radian)
    minimum_energy = float(minimum_energy_hartree)
    scale = float(coordinate_scale_dq_dphi)
    if not 0.0 < minimum_angle_radian < pi or not np.isfinite(minimum_energy):
        raise ValueError("the additional minimum must lie inside the half-period")
    if not np.isfinite(scale) or abs(scale) < 1.0e-8:
        raise ValueError("torsional SONIC/phase scale must be finite and nonzero")
    if maximum_harmonic is None:
        maximum_harmonic = 6 if reference_curvature_hartree_per_radian2 is not None else 5
    maximum_supported = 6 if reference_curvature_hartree_per_radian2 is not None else 5
    if maximum_harmonic < 1 or maximum_harmonic > maximum_supported:
        raise ValueError("the requested Fourier order exceeds the independent observations")
    modes = np.arange(1, maximum_harmonic + 1, dtype=float)
    all_phase = np.concatenate((phase, (minimum_angle,)))
    all_energy = np.concatenate((energy, (minimum_energy,))) - energy[0]
    energy_design = np.column_stack((np.ones(5), np.cos(np.outer(all_phase, modes))))
    gradient_design = np.column_stack(
        (np.zeros(4), -np.sin(np.outer(phase, modes)) * modes)
    )
    # One 15-degree interval converts the phase derivative into an energy
    # difference of comparable scale. Endpoint gradients are symmetry checks.
    weight = pi / 12.0
    design = np.vstack((energy_design, weight * gradient_design))
    observations = np.concatenate((all_energy, weight * gradient))
    if reference_curvature_hartree_per_radian2 is None:
        coefficients, _, rank, _ = np.linalg.lstsq(design, observations, rcond=1.0e-12)
        if rank != maximum_harmonic + 1:
            raise ValueError("torsional Fourier coefficients are not identifiable")
    else:
        curvature = float(reference_curvature_hartree_per_radian2)
        curvature_angle = (
            minimum_angle if curvature_angle_radian is None else float(curvature_angle_radian)
        )
        if not np.isfinite(curvature) or not np.isfinite(curvature_angle):
            raise ValueError("reference torsional curvature and angle must be finite")
        constraint = np.concatenate(((0.0,), -(modes**2) * np.cos(modes * curvature_angle)))
        null_space = np.linalg.svd(constraint[None, :], full_matrices=True)[2][1:].T
        particular = constraint * curvature / float(constraint @ constraint)
        reduced = design @ null_space
        correction, _, rank, _ = np.linalg.lstsq(
            reduced, observations - design @ particular, rcond=1.0e-12
        )
        if rank != maximum_harmonic:
            raise ValueError("torsional Fourier coefficients are not identifiable")
        coefficients = particular + null_space @ correction
    coefficients[0] -= np.sum(coefficients)
    residual = energy_design @ coefficients - all_energy
    cosine = np.asarray(coefficients[1:], dtype=float)
    sine = np.zeros(maximum_harmonic, dtype=float)
    local_gradient, quadratic, cubic, quartic = _local_derivatives(
        tuple(range(1, maximum_harmonic + 1)), cosine, sine, scale
    )
    return ZaffTorsionFourierFit(
        harmonics=tuple(range(1, maximum_harmonic + 1)),
        constant=float(coefficients[0]),
        cosine_coefficients=tuple(float(value) for value in cosine),
        sine_coefficients=tuple(float(value) for value in sine),
        coordinate_scale_dq_dphi=scale,
        rmse_hartree=float(np.sqrt(np.mean(residual**2))),
        maximum_residual_hartree=float(np.max(np.abs(residual))),
        local_gradient=local_gradient,
        local_quadratic=quadratic,
        local_cubic=cubic,
        local_quartic=quartic,
    )


def assess_zaff_torsion_prescan(
    fit: ZaffTorsionFourierFit,
    energies_hartree: Sequence[float],
    *,
    reference_quadratic: float,
    constrained_fit: ZaffTorsionFourierFit | None = None,
    maximum_curvature_relative_error: float = 0.35,
    maximum_gradient_absolute_error: float = 0.02,
    maximum_fit_rmse_kcal_mol: float = 0.25,
    minimum_barrier_kcal_mol: float = 0.01,
) -> ZaffTorsionPrescanControl:
    """Decide whether a cheap scan can supply the global Fourier shape."""

    energies = np.asarray(energies_hartree, dtype=float)
    denominator = max(abs(float(reference_quadratic)), 1.0e-12)
    curvature_error = abs(fit.local_quadratic - reference_quadratic) / denominator
    gradient_error = abs(fit.local_gradient)
    rmse = fit.rmse_hartree * HARTREE_TO_KCAL_MOL
    constrained_rmse = (
        None if constrained_fit is None else constrained_fit.rmse_hartree * HARTREE_TO_KCAL_MOL
    )
    barrier = float(np.ptp(energies) * HARTREE_TO_KCAL_MOL)
    reasons = []
    if fit.local_quadratic * reference_quadratic <= 0.0:
        reasons.append("CURVATURE_SIGN_MISMATCH")
    if curvature_error > maximum_curvature_relative_error:
        reasons.append("CURVATURE_MISMATCH")
    if gradient_error > maximum_gradient_absolute_error:
        reasons.append("REFERENCE_GRADIENT_MISMATCH")
    if rmse > maximum_fit_rmse_kcal_mol:
        reasons.append("FOURIER_RESIDUAL_TOO_LARGE")
    if constrained_rmse is not None and constrained_rmse > maximum_fit_rmse_kcal_mol:
        reasons.append("LOCAL_DERIVATIVE_CONSTRAINT_RESIDUAL")
    if barrier < minimum_barrier_kcal_mol:
        reasons.append("SCAN_BARRIER_UNRESOLVED")
    return ZaffTorsionPrescanControl(
        passed=not reasons,
        reasons=tuple(reasons),
        curvature_relative_error=float(curvature_error),
        gradient_absolute_error=float(gradient_error),
        fit_rmse_kcal_mol=float(rmse),
        constrained_fit_rmse_kcal_mol=(
            None if constrained_rmse is None else float(constrained_rmse)
        ),
        barrier_kcal_mol=barrier,
    )


def _design_matrix(phase: np.ndarray, harmonics: tuple[int, ...]) -> np.ndarray:
    modes = np.asarray(harmonics, dtype=float)
    return np.column_stack(
        (
            np.ones(len(phase), dtype=float),
            np.cos(np.outer(phase, modes)),
            np.sin(np.outer(phase, modes)),
        )
    )


def _local_constraint_matrix(
    harmonics: tuple[int, ...],
    scale: float,
    derivatives: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    modes = np.asarray(harmonics, dtype=float)
    count = len(modes)
    rows = np.zeros((5, 1 + 2 * count), dtype=float)
    rows[0, 0] = 1.0
    rows[0, 1 : count + 1] = 1.0
    rows[1, count + 1 :] = modes / scale
    rows[2, 1 : count + 1] = -(modes**2) / scale**2
    rows[3, count + 1 :] = -(modes**3) / scale**3
    rows[4, 1 : count + 1] = modes**4 / scale**4
    quadratic, cubic, quartic = (float(value) for value in derivatives)
    return rows, np.asarray((0.0, 0.0, quadratic, cubic, quartic), dtype=float)


def _local_derivatives(
    harmonics: tuple[int, ...],
    cosine: np.ndarray,
    sine: np.ndarray,
    scale: float,
) -> tuple[float, float, float, float]:
    modes = np.asarray(harmonics, dtype=float)
    gradient = float(np.sum(modes * sine) / scale)
    quadratic = float(-np.sum(modes**2 * cosine) / scale**2)
    cubic = float(-np.sum(modes**3 * sine) / scale**3)
    quartic = float(np.sum(modes**4 * cosine) / scale**4)
    return gradient, quadratic, cubic, quartic
