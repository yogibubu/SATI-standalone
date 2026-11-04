"""Pure partitioned-RFO mathematics for first-order saddles.

This module owns no chemistry, coordinate construction, chart lifecycle, or
backend policy.  It receives one Hessian eigensystem in an orthonormal metric
chart and returns an auditable P-RFO step.  This includes both LINK's legacy
conditioned variants and the literal raw-spectrum GDV ``DXRFO``/``GDIRFO``
equations.
Keeping these operations together prevents root selection and trust
restriction from drifting into independent fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RestrictedPartitionedRFOResult:
    """One auditable index-one RS-P-RFO solution."""

    step: np.ndarray
    effective_eigenvalues: np.ndarray
    alpha: float
    restricted: bool
    spectral_floor: float
    condition_number: float
    raw_index: int


@dataclass(frozen=True)
class DualShiftPartitionedRFOResult:
    """One index-one P-RFO model with independent partition shifts."""

    step: np.ndarray
    effective_eigenvalues: np.ndarray
    ascending_shift: float
    descending_shift: float | None
    spectral_floor: float
    condition_number: float
    raw_index: int


@dataclass(frozen=True)
class GDVGDIRFOResult:
    """Literal GDV ``GDIRFO`` transition-state step.

    ``lambda0`` is the ascending root for the first ordered Hessian mode;
    ``lambda_stable`` is the independent descending root for all remaining
    modes.  The physical spectrum is returned unchanged.
    """

    step: np.ndarray
    eigenvalues: np.ndarray
    lambda0: float
    lambda_stable: float | None
    raw_index: int
    ok: bool


@dataclass(frozen=True)
class GDVDXRFOResult:
    """Literal first-order-saddle result from GDV ``utilam.F:DXRFO``."""

    step: np.ndarray
    eigenvalues: np.ndarray
    lambda0: float
    lambda_stable: float | None
    raw_index: int
    ok: bool


@dataclass(frozen=True)
class ReactionModeSelection:
    """Auditable choice inside a multi-negative seed eigenspace."""

    index: int
    default_overlap: float
    selected_overlap: float
    isotropic_overlap: float
    policy: str


def index_one_spectrum(
    eigenvalues: np.ndarray,
    transition_mode: int,
    *,
    absolute_floor: float,
    maximum_condition: float,
) -> tuple[np.ndarray, float, float, int]:
    """Return the magnitude-preserving index-one step spectrum.

    The stored Hessian is not changed.  Its eigenvalue magnitudes define the
    local stiffnesses; the tracked transition mode receives the sole negative
    sign and every orthogonal mode a positive sign.  A common spectral floor
    enforces the configured condition bound without introducing a
    coordinate- or molecule-dependent branch.
    """

    raw = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if not raw.size:
        raise ValueError("a saddle-point Hessian spectrum cannot be empty")
    if np.any(~np.isfinite(raw)):
        raise ValueError("a saddle-point Hessian spectrum must be finite")
    mode = int(transition_mode)
    if mode < 0 or mode >= raw.size:
        raise ValueError("transition mode is outside the Hessian spectrum")
    floor0 = float(absolute_floor)
    condition_limit = float(maximum_condition)
    if not math.isfinite(floor0) or floor0 <= 0.0:
        raise ValueError("saddle-point spectral floor must be positive and finite")
    if not math.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("saddle-point condition bound must exceed one")

    maximum = max(float(np.max(np.abs(raw))), floor0)
    floor = max(floor0, maximum / condition_limit)
    effective = np.maximum(np.abs(raw), floor)
    effective[mode] *= -1.0
    magnitudes = np.abs(effective)
    condition = float(np.max(magnitudes) / np.min(magnitudes))
    raw_index = int(np.count_nonzero(raw < -floor0))
    return effective, float(floor), condition, raw_index


def generalized_rfo_subspace_step(
    curvatures: np.ndarray,
    gradient: np.ndarray,
    *,
    maximize: bool,
    alpha: float = 1.0,
) -> np.ndarray:
    """Solve one finite generalized-RFO root of a diagonal partition."""

    diagonal = np.asarray(curvatures, dtype=float).reshape(-1)
    vector = np.asarray(gradient, dtype=float).reshape(-1)
    if diagonal.shape != vector.shape:
        raise ValueError("RFO subspace curvature and gradient shapes differ")
    if not diagonal.size:
        return np.zeros(0, dtype=float)
    if np.any(~np.isfinite(diagonal)) or np.any(~np.isfinite(vector)):
        raise ValueError("RFO subspace inputs must be finite")
    scale = float(alpha)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("RFO generalized-eigenproblem scale must be positive")

    # Bofill convention: the homogeneous coordinate is first and is not
    # scaled; all physical coordinates share the same alpha.
    augmented = np.zeros((diagonal.size + 1, diagonal.size + 1), dtype=float)
    augmented[0, 1:] = vector
    augmented[1:, 0] = vector
    augmented[1:, 1:] = np.diag(diagonal)
    inverse_sqrt_scale = np.ones(diagonal.size + 1, dtype=float)
    inverse_sqrt_scale[1:] = 1.0 / math.sqrt(scale)
    transformed = (
        inverse_sqrt_scale[:, None]
        * augmented
        * inverse_sqrt_scale[None, :]
    )
    values, transformed_vectors = np.linalg.eigh(transformed)
    vectors = inverse_sqrt_scale[:, None] * transformed_vectors

    # After index-one conditioning, the negative one-dimensional partition
    # has a unique finite maximum root and the positive orthogonal partition
    # a unique finite minimum root.  This remains true at exactly zero
    # gradient, where that root is the homogeneous zero-step eigenvector.
    selected = int(np.argmax(values) if maximize else np.argmin(values))
    root = vectors[:, selected]
    denominator = float(root[0])
    if denominator == 0.0:
        raise np.linalg.LinAlgError("generalized RFO physical root is singular")
    step = np.asarray(root[1:] / denominator, dtype=float)
    if np.any(~np.isfinite(step)):
        raise np.linalg.LinAlgError("generalized RFO produced a non-finite step")
    return step


def partitioned_rfo_step(
    effective_eigenvalues: np.ndarray,
    projected_gradient: np.ndarray,
    transition_mode: int,
    *,
    alpha: float,
) -> np.ndarray:
    """Maximize the tracked mode and minimize its orthogonal complement."""

    curvatures = np.asarray(effective_eigenvalues, dtype=float).reshape(-1)
    gradient = np.asarray(projected_gradient, dtype=float).reshape(-1)
    if curvatures.shape != gradient.shape:
        raise ValueError("P-RFO curvature and gradient shapes differ")
    mode = int(transition_mode)
    if mode < 0 or mode >= curvatures.size:
        raise ValueError("transition mode is outside the P-RFO partition")
    if curvatures[mode] >= 0.0:
        raise ValueError("the tracked P-RFO mode must have negative curvature")
    stable = np.asarray([index for index in range(curvatures.size) if index != mode])
    if stable.size and np.any(curvatures[stable] <= 0.0):
        raise ValueError("the orthogonal P-RFO partition must have positive curvature")

    step = np.zeros_like(gradient)
    step[mode] = generalized_rfo_subspace_step(
        curvatures[mode : mode + 1],
        gradient[mode : mode + 1],
        maximize=True,
        alpha=alpha,
    )[0]
    if stable.size:
        step[stable] = generalized_rfo_subspace_step(
            curvatures[stable],
            gradient[stable],
            maximize=False,
            alpha=alpha,
        )
    return step


def dual_shift_partitioned_rfo_step(
    eigenvalues: np.ndarray,
    projected_gradient: np.ndarray,
    transition_mode: int,
    *,
    absolute_floor: float,
    maximum_condition: float,
) -> DualShiftPartitionedRFOResult:
    """Solve ascending and descending P-RFO roots with separate shifts.

    The physical Hessian is first conditioned to an index-one step model.
    Raw negative modes other than the tracked reaction mode therefore remain
    in the descending partition as positive-magnitude curvatures.
    """
    effective, floor, condition, raw_index = index_one_spectrum(
        eigenvalues,
        transition_mode,
        absolute_floor=absolute_floor,
        maximum_condition=maximum_condition,
    )
    gradient = np.asarray(projected_gradient, dtype=float).reshape(-1)
    if gradient.shape != effective.shape:
        raise ValueError("saddle spectrum and projected gradient shapes differ")
    if np.any(~np.isfinite(gradient)):
        raise ValueError("projected saddle gradient must be finite")
    mode = int(transition_mode)
    ascending, ascending_shift = _rfo_partition_root(
        effective[mode : mode + 1], gradient[mode : mode + 1], maximize=True
    )
    stable = np.asarray([i for i in range(effective.size) if i != mode], dtype=int)
    step = np.zeros_like(gradient)
    step[mode] = ascending[0]
    descending_shift: float | None = None
    if stable.size:
        step[stable], descending_shift = _rfo_partition_root(
            effective[stable], gradient[stable], maximize=False
        )
    return DualShiftPartitionedRFOResult(
        step=step,
        effective_eigenvalues=effective,
        ascending_shift=ascending_shift,
        descending_shift=descending_shift,
        spectral_floor=floor,
        condition_number=condition,
        raw_index=raw_index,
    )


def gdv_gdirfo_step(
    eigenvalues: np.ndarray,
    projected_gradient: np.ndarray,
) -> GDVGDIRFOResult:
    """Replicate GDV ``l103.F:GDIRFO`` for a first-order saddle search.

    GDV always assigns the first ordered Hessian mode to the maximizing
    partition (``ModMax=1``) and minimizes over every remaining raw mode
    (``ModMin=2``).  It neither conditions the spectrum to index one nor
    rejects an index-zero Hessian.  ``projected_gradient`` uses LINK's
    gradient convention; GDV's ``FTempU`` is the corresponding force, so the
    signs below are the direct gradient-form equivalent of the Fortran.
    """

    spectrum = np.asarray(eigenvalues, dtype=float).reshape(-1)
    gradient = np.asarray(projected_gradient, dtype=float).reshape(-1)
    if not spectrum.size:
        raise ValueError("GDV GDIRFO requires at least one Hessian mode")
    if spectrum.shape != gradient.shape:
        raise ValueError("GDV GDIRFO spectrum and gradient shapes differ")
    if np.any(~np.isfinite(spectrum)) or np.any(~np.isfinite(gradient)):
        raise ValueError("GDV GDIRFO inputs must be finite")
    if np.any(np.diff(spectrum) < 0.0):
        raise ValueError("GDV GDIRFO requires an ordered Hessian spectrum")

    first_eigenvalue = float(spectrum[0])
    first_gradient = float(gradient[0])
    lambda0 = 0.5 * (
        first_eigenvalue
        + math.sqrt(first_eigenvalue * first_eigenvalue + 4.0 * first_gradient**2)
    )
    # Literal counterpart of the GDV guard that separates Lambda0 from the
    # first eigenvalue when the projected force vanishes to machine precision.
    if abs(lambda0 - first_eigenvalue) < 1.0e-8:
        lambda0 += 1.0e-8

    step = np.zeros_like(gradient)
    lambda_stable: float | None = None
    if spectrum.size > 1:
        lambda_stable = min(0.0, float(spectrum[1]) - 0.05)
        upper = float(spectrum[1])
        lower = -1.0e6
        converged = False
        for _iteration in range(999):
            total = float(
                np.sum(gradient[1:] ** 2 / (lambda_stable - spectrum[1:]))
            )
            if abs(lambda_stable - total) <= 1.0e-8:
                converged = True
                break
            if spectrum[1] > 0.0:
                lambda_stable = total
            else:
                if total < lambda_stable:
                    upper = lambda_stable
                else:
                    lower = lambda_stable
                if lower == -1.0e6:
                    lambda_stable -= 0.05
                else:
                    lambda_stable = 0.5 * (upper + lower)
                    if abs(upper - lower) <= 1.0e-8:
                        converged = True
                        break
        if not converged:
            return GDVGDIRFOResult(
                step=step,
                eigenvalues=spectrum.copy(),
                lambda0=lambda0,
                lambda_stable=lambda_stable,
                raw_index=int(np.count_nonzero(spectrum < 0.0)),
                ok=False,
            )
        if lambda_stable > float(spectrum[1]) - 1.0e-4:
            return GDVGDIRFOResult(
                step=step,
                eigenvalues=spectrum.copy(),
                lambda0=lambda0,
                lambda_stable=lambda_stable,
                raw_index=int(np.count_nonzero(spectrum < 0.0)),
                ok=False,
            )
        if lambda_stable > 0.0 and float(spectrum[1]) > 0.0:
            return GDVGDIRFOResult(
                step=step,
                eigenvalues=spectrum.copy(),
                lambda0=lambda0,
                lambda_stable=lambda_stable,
                raw_index=int(np.count_nonzero(spectrum < 0.0)),
                ok=False,
            )

    step[0] = first_gradient / (lambda0 - first_eigenvalue)
    if lambda_stable is not None:
        step[1:] = gradient[1:] / (lambda_stable - spectrum[1:])

    return GDVGDIRFOResult(
        step=step,
        eigenvalues=spectrum.copy(),
        lambda0=lambda0,
        lambda_stable=lambda_stable,
        raw_index=int(np.count_nonzero(spectrum < 0.0)),
        ok=True,
    )


def gdv_dxrfo_step(
    eigenvalues: np.ndarray,
    projected_gradient: np.ndarray,
    *,
    maximum_internal_step: float = 0.3,
    climb: bool = False,
) -> GDVDXRFOResult:
    """Replicate GDV's ordinary geometry step for a first-order saddle.

    This is a direct gradient-convention translation of
    ``utilam.F:DXRFO`` with ``Neg=1``, ``NGoDwn=0`` and ``IEStpM=0``.
    ``l103.F:GrdOpt`` supplies the ordered eigenbasis explicitly for an
    ordinary ReadAllGIC calculation (``AlUnit=False``); this function returns
    the corresponding eigenbasis components.  Raw negative modes after the
    first mode therefore use DXRFO's bounded downhill rule; they are not
    passed through the stable RFO root as GDIRFO does.
    """

    spectrum = np.asarray(eigenvalues, dtype=float).reshape(-1)
    gradient = np.asarray(projected_gradient, dtype=float).reshape(-1)
    if not spectrum.size:
        raise ValueError("GDV DXRFO requires at least one Hessian mode")
    if spectrum.shape != gradient.shape:
        raise ValueError("GDV DXRFO spectrum and gradient shapes differ")
    if np.any(~np.isfinite(spectrum)) or np.any(~np.isfinite(gradient)):
        raise ValueError("GDV DXRFO inputs must be finite")
    if np.any(np.diff(spectrum) < 0.0):
        raise ValueError("GDV DXRFO requires an ordered Hessian spectrum")
    dx_max = float(maximum_internal_step)
    if not math.isfinite(dx_max) or dx_max <= 0.0:
        raise ValueError("GDV DXRFO maximum internal step must be positive")

    # l103 carries forces, whereas LINK carries gradients.
    force = -gradient
    conv = 1.0e-8
    eigen_negative = -1.0e-4
    lambda0 = 0.5 * (
        float(spectrum[0])
        + math.sqrt(float(spectrum[0]) ** 2 + 4.0 * float(force[0]) ** 2)
    )
    if abs(lambda0 - float(spectrum[0])) < conv:
        lambda0 += conv

    lambda_stable: float | None
    if spectrum.size > 1:
        first_stable = float(spectrum[1])
        lambda_stable = 0.0
        if first_stable <= 0.0:
            lambda_stable = min(first_stable - 0.05, 2.0 * first_stable)
        upper = first_stable
        lower = -1.0e6
        converged = False
        for _iteration in range(9999):
            total = float(
                np.sum(force[1:] ** 2 / (lambda_stable - spectrum[1:]))
            )
            if abs(lambda_stable - total) <= conv:
                converged = True
                break
            if first_stable > 0.0:
                lambda_stable = total
            else:
                if total < lambda_stable:
                    upper = lambda_stable
                else:
                    lower = lambda_stable
                if lower == -1.0e6:
                    lambda_stable -= 0.05
                else:
                    lambda_stable = 0.5 * (upper + lower)
                    if abs(upper - lower) <= conv:
                        converged = True
                        break
        if not converged or lambda_stable > first_stable or (
            lambda_stable > 0.0 and first_stable > 0.0
        ):
            return GDVDXRFOResult(
                step=np.zeros_like(gradient),
                eigenvalues=spectrum.copy(),
                lambda0=lambda0,
                lambda_stable=lambda_stable,
                raw_index=int(np.count_nonzero(spectrum < 0.0)),
                ok=False,
            )
    else:
        lambda_stable = lambda0

    # DXRFO first accumulates the complete stable-partition step norm.  Extra
    # negative modes use the bounded downhill rule; all other stable modes use
    # the minimizing RFO root.  The resulting norm caps a non-negative first
    # mode when Climb is false (the standard non-QST path).
    stable_step_norm2 = 0.0
    stable_cap = 1.0
    for index in range(1, spectrum.size):
        if spectrum[index] < eigen_negative:
            raw = abs(float(force[index]) / (lambda0 - float(spectrum[index])))
            component = min(max(raw, 2.5 * dx_max), stable_cap)
            if force[index] < 0.0:
                component = -component
            stable_cap *= 0.5
        else:
            assert lambda_stable is not None
            component = -float(force[index]) / (
                lambda_stable - float(spectrum[index])
            )
        stable_step_norm2 += component * component

    step = np.zeros_like(gradient)
    stable_cap = 1.0
    for index in range(spectrum.size):
        eigenvalue = float(spectrum[index])
        component_force = float(force[index])
        if index == 0:
            if eigenvalue >= eigen_negative and not climb:
                cap = min(math.sqrt(stable_step_norm2), 0.1)
                if abs(component_force) > conv:
                    component = -component_force / (lambda0 - eigenvalue)
                    if abs(component) >= cap:
                        component = -math.copysign(cap, component_force)
                else:
                    component = -cap
            else:
                component = -component_force / (lambda0 - eigenvalue)
        elif eigenvalue < eigen_negative:
            raw = abs(component_force / (lambda0 - eigenvalue))
            component = min(max(raw, 2.5 * dx_max), stable_cap)
            if component_force < 0.0:
                component = -component
            stable_cap *= 0.5
        else:
            assert lambda_stable is not None
            component = -component_force / (lambda_stable - eigenvalue)
        step[index] = component

    return GDVDXRFOResult(
        step=step,
        eigenvalues=spectrum.copy(),
        lambda0=lambda0,
        lambda_stable=lambda_stable,
        raw_index=int(np.count_nonzero(spectrum < 0.0)),
        ok=True,
    )


def _rfo_partition_root(
    curvatures: np.ndarray, gradient: np.ndarray, *, maximize: bool
) -> tuple[np.ndarray, float]:
    diagonal = np.asarray(curvatures, dtype=float).reshape(-1)
    vector = np.asarray(gradient, dtype=float).reshape(-1)
    if diagonal.shape != vector.shape or not diagonal.size:
        raise ValueError("RFO partition curvature and gradient shapes differ")
    augmented = np.zeros((diagonal.size + 1, diagonal.size + 1), dtype=float)
    augmented[0, 1:] = vector
    augmented[1:, 0] = vector
    augmented[1:, 1:] = np.diag(diagonal)
    values, vectors = np.linalg.eigh(augmented)
    selected = int(np.argmax(values) if maximize else np.argmin(values))
    root = vectors[:, selected]
    if abs(float(root[0])) <= np.finfo(float).eps:
        raise np.linalg.LinAlgError("partitioned-RFO physical root is singular")
    return np.asarray(root[1:] / root[0], dtype=float), float(values[selected])


def restricted_partitioned_rfo_step(
    eigenvalues: np.ndarray,
    projected_gradient: np.ndarray,
    transition_mode: int,
    *,
    maximum_step_norm: float,
    absolute_floor: float,
    maximum_condition: float,
    relative_tolerance: float = 1.0e-10,
    maximum_iterations: int = 80,
) -> RestrictedPartitionedRFOResult:
    """Solve the common-alpha RS-P-RFO trust problem."""

    target = float(maximum_step_norm)
    tolerance = float(relative_tolerance)
    iterations = int(maximum_iterations)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("P-RFO trust target must be positive and finite")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("P-RFO relative tolerance must lie in (0, 1)")
    if iterations < 1:
        raise ValueError("P-RFO trust solver requires at least one iteration")

    effective, floor, condition, raw_index = index_one_spectrum(
        eigenvalues,
        transition_mode,
        absolute_floor=absolute_floor,
        maximum_condition=maximum_condition,
    )
    gradient = np.asarray(projected_gradient, dtype=float).reshape(-1)
    if gradient.shape != effective.shape:
        raise ValueError("P-RFO spectrum and gradient shapes differ")

    def solve(alpha: float) -> np.ndarray:
        return partitioned_rfo_step(
            effective,
            gradient,
            transition_mode,
            alpha=alpha,
        )

    alpha = 1.0
    step = solve(alpha)
    norm = float(np.linalg.norm(step))
    if not math.isfinite(norm):
        raise np.linalg.LinAlgError("unrestricted P-RFO step norm is non-finite")
    if norm <= target * (1.0 + tolerance):
        return RestrictedPartitionedRFOResult(
            step=step,
            effective_eigenvalues=effective,
            alpha=alpha,
            restricted=False,
            spectral_floor=floor,
            condition_number=condition,
            raw_index=raw_index,
        )

    lower_alpha = alpha
    upper_alpha = 2.0
    upper_step = solve(upper_alpha)
    for _ in range(iterations):
        upper_norm = float(np.linalg.norm(upper_step))
        if math.isfinite(upper_norm) and upper_norm <= target:
            break
        lower_alpha = upper_alpha
        upper_alpha *= 2.0
        upper_step = solve(upper_alpha)
    else:
        raise RuntimeError("unable to bracket the RS-P-RFO trust boundary")

    # Keep the upper endpoint feasible throughout the safeguarded solve.
    feasible_step = upper_step
    for _ in range(iterations):
        alpha = 0.5 * (lower_alpha + upper_alpha)
        trial_step = solve(alpha)
        trial_norm = float(np.linalg.norm(trial_step))
        if not math.isfinite(trial_norm) or trial_norm > target:
            lower_alpha = alpha
        else:
            upper_alpha = alpha
            feasible_step = trial_step
            if target - trial_norm <= tolerance * target:
                break
        if upper_alpha - lower_alpha <= tolerance * max(1.0, upper_alpha):
            break

    return RestrictedPartitionedRFOResult(
        step=np.asarray(feasible_step, dtype=float),
        effective_eigenvalues=effective,
        alpha=float(upper_alpha),
        restricted=True,
        spectral_floor=floor,
        condition_number=condition,
        raw_index=raw_index,
    )


def condition_aware_reaction_mode(
    eigenvalues: np.ndarray,
    cartesian_candidates: np.ndarray,
    reaction_directions: np.ndarray | None,
    default_mode: int,
    *,
    absolute_floor: float,
    maximum_condition: float,
) -> ReactionModeSelection:
    """Choose a reactive mode only when a multi-negative seed resolves it.

    A unique negative direction is invariant and is never replaced.  For a
    higher-index seed, the default eigenmode is retained unless its squared
    projection on the ORACLE/SMITH reaction subspace lies below the isotropic
    expectation ``rank / dimension`` and another negative eigenmode lies
    above it.  SVD removes unresolved reaction directions using the same
    condition bound as the step model.  Thus the rule has no molecule- or
    benchmark-specific overlap threshold.
    """

    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    candidates = np.asarray(cartesian_candidates, dtype=float)
    mode = int(default_mode)
    if candidates.ndim != 2 or candidates.shape[1] != values.size:
        raise ValueError("reaction-mode candidates and eigenvalues are inconsistent")
    if mode < 0 or mode >= values.size:
        raise ValueError("default reaction mode is outside the Hessian spectrum")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(candidates)):
        raise ValueError("reaction-mode selection inputs must be finite")

    negative = np.flatnonzero(values < -float(absolute_floor))
    if reaction_directions is None:
        return ReactionModeSelection(mode, 0.0, 0.0, 0.0, "ordinal_invariant")
    directions = np.asarray(reaction_directions, dtype=float)
    if directions.ndim != 2 or directions.shape[0] != candidates.shape[0]:
        raise ValueError("reaction directions must be Cartesian tangent columns")
    if directions.shape[1] == 0:
        return ReactionModeSelection(mode, 0.0, 0.0, 0.0, "ordinal_no_reaction_subspace")
    if np.any(~np.isfinite(directions)):
        raise ValueError("reaction directions must be finite")

    left, singular, _right = np.linalg.svd(directions, full_matrices=False)
    if not singular.size or singular[0] <= 0.0:
        return ReactionModeSelection(mode, 0.0, 0.0, 0.0, "ordinal_singular_reaction_subspace")
    condition_limit = float(maximum_condition)
    if not math.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("reaction-subspace condition bound must exceed one")
    retained = singular >= singular[0] / condition_limit
    subspace = left[:, retained]
    if subspace.shape[1] == 0:
        return ReactionModeSelection(mode, 0.0, 0.0, 0.0, "ordinal_singular_reaction_subspace")

    overlaps = np.linalg.norm(subspace.T @ candidates, axis=0)
    pool = negative if negative.size else np.arange(candidates.shape[1], dtype=int)
    best = int(pool[np.argmax(overlaps[pool])])
    default_overlap = float(overlaps[mode])
    best_overlap = float(overlaps[best])
    isotropic = math.sqrt(subspace.shape[1] / candidates.shape[1])
    if negative.size == 0:
        policy = "reaction_subspace_index_zero"
    elif mode in pool and default_overlap < isotropic and best_overlap >= isotropic:
        policy = "conditioned_reaction_subspace"
    else:
        return ReactionModeSelection(
            mode,
            default_overlap,
            default_overlap,
            isotropic,
            "ordinal_reaction_subspace_not_decisive",
        )
    return ReactionModeSelection(
        best,
        default_overlap,
        best_overlap,
        isotropic,
        policy,
    )


def symmetric_multisecant_hessian_refresh(
    hessian: np.ndarray,
    directions: np.ndarray,
    directional_gradients: np.ndarray,
) -> np.ndarray:
    """Return the least-change symmetric Hessian satisfying block secants."""

    matrix = np.asarray(hessian, dtype=float)
    steps = np.asarray(directions, dtype=float)
    images = np.asarray(directional_gradients, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hessian refresh requires a square Hessian")
    if steps.ndim != 2 or steps.shape[0] != matrix.shape[0] or not steps.shape[1]:
        raise ValueError("Hessian refresh directions have the wrong shape")
    if images.shape != steps.shape:
        raise ValueError("directional gradients must match refresh directions")
    if any(np.any(~np.isfinite(item)) for item in (matrix, steps, images)):
        raise ValueError("Hessian refresh inputs must be finite")
    gram = steps.T @ steps
    if np.linalg.matrix_rank(gram) != gram.shape[0]:
        raise ValueError("Hessian refresh directions must be independent")
    dual = steps @ np.linalg.inv(gram)
    sampled_block = steps.T @ images
    compatible_images = images + dual @ (0.5 * (sampled_block.T - sampled_block))
    residual = compatible_images - 0.5 * (matrix + matrix.T) @ steps
    compatibility = steps.T @ residual
    update = (
        residual @ dual.T
        + dual @ residual.T
        - dual @ compatibility @ dual.T
    )
    refreshed = 0.5 * (matrix + matrix.T) + update
    return 0.5 * (refreshed + refreshed.T)


__all__ = [
    "DualShiftPartitionedRFOResult",
    "GDVDXRFOResult",
    "GDVGDIRFOResult",
    "ReactionModeSelection",
    "condition_aware_reaction_mode",
    "RestrictedPartitionedRFOResult",
    "generalized_rfo_subspace_step",
    "index_one_spectrum",
    "partitioned_rfo_step",
    "dual_shift_partitioned_rfo_step",
    "gdv_dxrfo_step",
    "gdv_gdirfo_step",
    "restricted_partitioned_rfo_step",
    "symmetric_multisecant_hessian_refresh",
]
