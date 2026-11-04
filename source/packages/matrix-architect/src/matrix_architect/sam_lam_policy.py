"""Acceptance gates for a local ZAFF fit and its large-amplitude scans."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Mapping, Sequence

import numpy as np

from .native_sparse_fit import (
    NativeSparseFit, fit_native_sparse_refinement,
    native_zaff_candidate_powers, native_zaff_value_gradient,
)


@dataclass(frozen=True)
class SamFitAcceptance:
    support_size: int
    normalized_holdout_errors: Mapping[str, float]
    condition_number: float


@dataclass(frozen=True)
class AutomaticSamFit:
    """Selected native terms and the independent acceptance evidence."""

    powers: tuple[tuple[int, ...], ...]
    coefficients: np.ndarray
    acceptance: SamFitAcceptance


def fit_automatic_sam_terms(
    primitive_atoms: Sequence[Sequence[int]],
    kinds: Sequence[str],
    reference: np.ndarray,
    primitive_from_modal: np.ndarray,
    training_displacements: np.ndarray,
    training_energies: np.ndarray,
    training_gradients: np.ndarray,
    holdout_displacements: np.ndarray,
    holdout_energies: np.ndarray,
    holdout_gradients: np.ndarray,
    *,
    excluded_lam_indices: Sequence[int] = (),
    energy_tolerance: float,
    gradient_tolerance: float,
    maximum_active_primitives: int = 2,
    maximum_terms: int | None = None,
    condition_limit: float = 1.0e10,
) -> AutomaticSamFit:
    """Select cubic/quartic SAM terms from QM-minus-complete-baseline E/G.

    Train and held-out geometries are supplied separately.  Their residuals
    must already have the same electrostatic, vdW and ZAFF2 contributions
    removed.  The connected candidate catalog and smallest passing support
    are determined here, with no molecule-specific list of couplings.
    Anharmonic reference frequencies do not enter this fit or its selection.
    Compute the spectrum only after freezing the returned field; any available
    reference spectrum is a subsequent benchmark.
    """

    reference = np.asarray(reference, dtype=float)
    transform = np.asarray(primitive_from_modal, dtype=float)
    count = len(primitive_atoms)
    if (len(kinds) != count or reference.shape != (count,)
            or transform.ndim != 2 or transform.shape[0] != count):
        raise ValueError("automatic SAM primitive dimensions differ")
    modal_count = transform.shape[1]
    data = []
    for displacement, energy, gradient in (
        (training_displacements, training_energies, training_gradients),
        (holdout_displacements, holdout_energies, holdout_gradients),
    ):
        displacement = np.asarray(displacement, dtype=float)
        energy = np.asarray(energy, dtype=float)
        gradient = np.asarray(gradient, dtype=float)
        if (displacement.ndim != 2 or displacement.shape[1] != count
                or displacement.shape[0] == 0
                or energy.shape != (displacement.shape[0],)
                or gradient.shape != (displacement.shape[0], modal_count)
                or not all(np.all(np.isfinite(item)) for item in (displacement, energy, gradient))):
            raise ValueError("automatic SAM point data are incomplete or nonfinite")
        data.append((displacement, energy, gradient))
    if (not np.isfinite(energy_tolerance) or energy_tolerance <= 0
            or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0):
        raise ValueError("automatic SAM tolerances must be positive")
    powers = native_zaff_candidate_powers(
        primitive_atoms, excluded_indices=excluded_lam_indices,
        maximum_active_primitives=maximum_active_primitives,
    )
    if not powers:
        raise ValueError("no connected SAM refinement candidate remains")

    def rows(points: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        displacement, energies, gradients = points
        energy_design = np.empty((len(displacement), len(powers)))
        gradient_design = np.empty((len(displacement) * modal_count, len(powers)))
        for row, point in enumerate(displacement):
            for column, power in enumerate(powers):
                value, derivative = native_zaff_value_gradient(
                    point, reference, transform, power, tuple(kinds),
                )
                energy_design[row, column] = value
                gradient_design[row * modal_count:(row + 1) * modal_count, column] = derivative
        return energy_design, energies, gradient_design, gradients.reshape(-1)

    a_energy, y_energy, a_gradient, y_gradient = rows(data[0])
    h_energy, z_energy, h_gradient, z_gradient = rows(data[1])
    combined = np.vstack((a_energy / energy_tolerance, a_gradient / gradient_tolerance))
    target = np.r_[y_energy / energy_tolerance, y_gradient / gradient_tolerance]
    observed = np.linalg.norm(combined, axis=0) > 1.0e-14
    if not np.any(observed):
        raise ValueError("training displacements observe no native SAM terms")
    candidate_columns = np.flatnonzero(observed)
    limit = min(len(candidate_columns), combined.shape[0])
    if maximum_terms is not None:
        if maximum_terms < 1:
            raise ValueError("maximum SAM support must be positive")
        limit = min(limit, maximum_terms)
    holdout_energy = h_energy[:, candidate_columns]
    holdout_gradient = h_gradient[:, candidate_columns]

    def passes_holdout(coefficients: np.ndarray) -> bool:
        return (
            np.max(np.abs(holdout_energy @ coefficients - z_energy)) <= energy_tolerance
            and np.max(np.abs(holdout_gradient @ coefficients - z_gradient)) <= gradient_tolerance
        )

    path = fit_native_sparse_refinement(
        combined[:, candidate_columns], target,
        max_terms=limit, condition_limit=condition_limit,
        stop_when=passes_holdout,
    )
    acceptance = select_sam_fit(
        path,
        {"energy": holdout_energy, "gradient": holdout_gradient},
        {"energy": z_energy, "gradient": z_gradient},
        {"energy": energy_tolerance, "gradient": gradient_tolerance},
        condition_limit=condition_limit,
    )
    coefficients = np.zeros(len(powers))
    coefficients[candidate_columns] = path.coefficients(
        acceptance.support_size, len(candidate_columns),
    )
    return AutomaticSamFit(powers, coefficients, acceptance)


def select_sam_fit(
    path: NativeSparseFit,
    holdout_design: Mapping[str, np.ndarray],
    holdout_target: Mapping[str, np.ndarray],
    tolerances: Mapping[str, float],
    *,
    condition_limit: float = 1.0e10,
) -> SamFitAcceptance:
    """Take the smallest rank-safe support that passes every independent observable.

    The caller supplies QM-minus-electrostatic-minus-vdW residuals after the
    immutable quadratic field and physical diagonal ZAFF terms are removed.
    Every matrix contains only previously acquired, held-out geometries.  An
    unavailable derivative is omitted rather than estimated from a quartic fit.
    """

    keys = set(holdout_design)
    if not keys or keys != set(holdout_target) or keys != set(tolerances):
        raise ValueError("SAM holdout observables and tolerances must match")
    if any(not np.isfinite(value) or value <= 0 for value in tolerances.values()):
        raise ValueError("SAM holdout tolerances must be finite and positive")
    if not np.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("SAM condition limit must be finite and at least one")
    column_count: int | None = None
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in sorted(keys):
        design = np.asarray(holdout_design[name], dtype=float)
        target = np.asarray(holdout_target[name], dtype=float).reshape(-1)
        if design.ndim != 2 or design.shape[0] == 0 or target.shape != (design.shape[0],):
            raise ValueError(f"invalid SAM {name} holdout shape")
        if column_count is None:
            column_count = design.shape[1]
        if design.shape[1] != column_count or not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
            raise ValueError(f"invalid SAM {name} holdout values")
        prepared[name] = design, target
    assert column_count is not None
    if path.support_order and max(path.support_order) >= column_count:
        raise ValueError("SAM holdout columns do not match the fitted support")
    for size in sorted(path.coefficients_by_size):
        condition = path.condition_by_size[size]
        if not np.isfinite(condition) or condition > condition_limit:
            continue
        coefficients = path.coefficients(size, column_count)
        errors = {
            name: float(np.max(np.abs(design @ coefficients - target)) / tolerances[name])
            for name, (design, target) in prepared.items()
        }
        if all(error <= 1.0 for error in errors.values()):
            return SamFitAcceptance(size, errors, float(condition))
    raise ValueError("no SAM support satisfies all independent observable gates")


def require_lam_scan_coverage(
    coordinates: Sequence[object],
    scan_identifiers: Sequence[str],
) -> tuple[str, ...]:
    """Require an acquired scan for every large-amplitude coordinate.

    Periodic torsions and ring phases need a full symmetry-unique cycle;
    nonperiodic transfer/inversion coordinates need both sides and the relevant
    turning region.  The provider certifies scan completeness before passing
    its identifier here; this gate prevents a SAM-only fit from being accepted.
    """

    required = tuple(
        str(coordinate.identifier)
        for coordinate in coordinates
        if str(getattr(coordinate, "large_amplitude_role", "LOCAL_DIAGONAL"))
        not in {"LOCAL_DIAGONAL", "SMALL_AMPLITUDE_NONPERIODIC", "PERIODIC_CANDIDATE"}
    )
    observed = set(map(str, scan_identifiers))
    missing = tuple(identifier for identifier in required if identifier not in observed)
    if missing:
        raise ValueError("missing mandatory LAM scans: " + ", ".join(missing))
    return required


def certify_sam_lam_fit(
    path: NativeSparseFit,
    holdout_design: Mapping[str, np.ndarray],
    holdout_target: Mapping[str, np.ndarray],
    tolerances: Mapping[str, float],
    coordinates: Sequence[object],
    certified_scan_identifiers: Sequence[str],
) -> SamFitAcceptance:
    """Check local SAM observables and coverage of the declared LAM scans.

    This is one part of certification, not a complete ZAFF3 certificate.
    The caller must also check the serialized physical field. Compute the
    requested anharmonic spectrum after freezing the fit. If a reference
    spectrum is available, compare matched resonant states with the same LAM
    treatment; it is not required as input for fitting or selecting terms.
    Passing energy/gradient rows alone does not establish spectral accuracy.
    """

    accepted = select_sam_fit(path, holdout_design, holdout_target, tolerances)
    require_lam_scan_coverage(coordinates, certified_scan_identifiers)
    return accepted


@dataclass(frozen=True)
class PeriodicLamScanAssessment:
    accepted_harmonic_order: int | None
    best_harmonic_order: int
    best_holdout_maximum: float
    sampled_points: int


@dataclass(frozen=True)
class PeriodicLamDifferenceFit:
    """Fourier correction to an evaluated ZAFF field on a periodic scan."""

    harmonic_order: int
    coefficients: np.ndarray
    training_maximum: float
    training_cross_validation_maximum: float
    independent_maximum: float


def fit_periodic_lam_difference(
    training_angles: Sequence[float],
    training_qm_energies: Sequence[float],
    training_previous_zaff_energies: Sequence[float],
    independent_angles: Sequence[float],
    independent_qm_energies: Sequence[float],
    independent_previous_zaff_energies: Sequence[float],
    *,
    training_replaced_zaff_terms: Sequence[float] | None = None,
    independent_replaced_zaff_terms: Sequence[float] | None = None,
    maximum_harmonic_order: int = 11,
    asymptotically_correct_previous_field: bool,
) -> PeriodicLamDifferenceFit:
    """Fit QM minus the *complete* previous field, including nonbonded terms.

    A replaceable contribution of the old field can be removed before fitting.
    For example, a nonperiodic local torsion coupling is replaced by adding
    its energy to the residual target and subtracting it from the final field.
    The same replacement must be evaluated at training and independent points.
    Energies are referenced to the first training point; the caller must also
    certify the reference gradient and Hessian before accepting a new field.
    An uncertified previous field cannot be rescued by a compensating ZAFF3
    correction: it must first be rebuilt in a physical native form.
    """

    if not asymptotically_correct_previous_field:
        raise ValueError("previous ZAFF level must pass its own asymptotic gate")

    def vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float).reshape(-1)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"invalid {name} for periodic ZAFF difference fit")
        return result

    angles = np.mod(np.asarray(training_angles, dtype=float).reshape(-1), 2 * pi)
    independent = np.mod(np.asarray(independent_angles, dtype=float).reshape(-1), 2 * pi)
    if (len(angles) < 5 or len(independent) == 0
            or not np.all(np.isfinite(angles))
            or not np.all(np.isfinite(independent))
            or maximum_harmonic_order < 1):
        raise ValueError("invalid periodic ZAFF difference scan")
    qm = vector(training_qm_energies, len(angles), "training QM energies")
    previous = vector(training_previous_zaff_energies, len(angles), "training ZAFF energies")
    independent_qm = vector(independent_qm_energies, len(independent), "independent QM energies")
    independent_previous = vector(
        independent_previous_zaff_energies, len(independent), "independent ZAFF energies"
    )
    if (training_replaced_zaff_terms is None) != (independent_replaced_zaff_terms is None):
        raise ValueError("replaced ZAFF terms must be supplied for both point sets")
    removed = (np.zeros(len(angles)) if training_replaced_zaff_terms is None else
               vector(training_replaced_zaff_terms, len(angles), "training replaced terms"))
    independent_removed = (
        np.zeros(len(independent)) if independent_replaced_zaff_terms is None else
        vector(independent_replaced_zaff_terms, len(independent), "independent replaced terms")
    )
    target = qm - previous + removed
    independent_target = independent_qm - independent_previous + independent_removed
    origin = target[0]
    target -= origin
    independent_target -= origin
    best: PeriodicLamDifferenceFit | None = None
    best_cv = float("inf")
    for order in range(1, maximum_harmonic_order + 1):
        if 1 + 2 * order > len(angles):
            break
        def design(x: np.ndarray) -> np.ndarray:
            return np.column_stack([
                np.ones(len(x)),
                *(function(harmonic * x)
                  for harmonic in range(1, order + 1)
                  for function in (np.cos, np.sin)),
            ])
        fit = design(angles)
        coefficients, *_ = np.linalg.lstsq(fit, target, rcond=1.0e-12)
        training_error = float(np.max(np.abs(fit @ coefficients - target)))
        # Select complexity using training data alone; the independent points
        # are inspected only once after the model has been fixed.
        if fit.shape[1] >= len(angles):
            continue
        leverage = np.sum(fit * (fit @ np.linalg.pinv(fit.T @ fit)), axis=1)
        if np.any(1.0 - leverage < 1.0e-8):
            continue
        cross_validation_error = float(np.max(np.abs(
            (fit @ coefficients - target) / (1.0 - leverage)
        )))
        independent_error = float(np.max(np.abs(
            design(independent) @ coefficients - independent_target
        )))
        result = PeriodicLamDifferenceFit(
            order, coefficients, training_error, cross_validation_error,
            independent_error,
        )
        if cross_validation_error < best_cv:
            best_cv = cross_validation_error
            best = result
    if best is None:
        raise ValueError("periodic ZAFF difference scan has insufficient training points")
    return best


def assess_periodic_lam_scan(
    angles_radian: Sequence[float],
    residual_energies: Sequence[float],
    *,
    maximum_harmonic_order: int,
    maximum_holdout_error: float,
) -> PeriodicLamScanAssessment:
    """Escalate a Fourier LAM scan only while held-out points need it.

    For final ZAFF fitting, energies have their electrostatic and van der Waals
    parts removed. Raw QM energies may be used to diagnose scan smoothness.
    Even and odd points form interleaved train/holdout sets. Failure means
    that this scan cannot certify one smooth periodic branch.
    """

    angle = np.mod(np.asarray(angles_radian, dtype=float).reshape(-1), 2 * pi)
    energy = np.asarray(residual_energies, dtype=float).reshape(-1)
    if (angle.size != energy.size or angle.size < 8
            or not np.all(np.isfinite(angle)) or not np.all(np.isfinite(energy))):
        raise ValueError("periodic LAM scan needs at least eight finite paired points")
    if (maximum_harmonic_order < 1 or not np.isfinite(maximum_holdout_error)
            or maximum_holdout_error <= 0):
        raise ValueError("invalid periodic LAM fit limits")
    order = np.argsort(angle)
    angle, energy = angle[order], energy[order]
    gaps = np.diff(np.r_[angle, angle[0] + 2 * pi])
    if np.min(gaps) <= 1.0e-10 or np.max(gaps) > pi / 3:
        raise ValueError("periodic LAM scan does not cover a distinct full cycle")
    training = np.arange(angle.size) % 2 == 0
    best_order = 0
    best_error = float("inf")
    accepted = None
    for harmonic_order in range(1, maximum_harmonic_order + 1):
        if 1 + 2 * harmonic_order > np.count_nonzero(training):
            break
        columns = [np.ones_like(angle)]
        for harmonic in range(1, harmonic_order + 1):
            columns.extend((np.cos(harmonic * angle), np.sin(harmonic * angle)))
        design = np.column_stack(columns)
        coefficients, *_ = np.linalg.lstsq(
            design[training], energy[training], rcond=1.0e-12
        )
        error = float(np.max(np.abs(design[~training] @ coefficients - energy[~training])))
        if error < best_error:
            best_order, best_error = harmonic_order, error
        if error <= maximum_holdout_error:
            accepted = harmonic_order
            break
    if best_order == 0:
        raise ValueError("periodic LAM scan cannot identify a Fourier order")
    return PeriodicLamScanAssessment(accepted, best_order, best_error, angle.size)
