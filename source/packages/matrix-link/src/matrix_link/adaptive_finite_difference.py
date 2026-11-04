"""Curvature- and model-adaptive finite-difference steps for LINK.

This module provides curvature- and model-adaptive numerical derivatives for
energy-only and energy/gradient/Hessian surfaces.  It does not alter LINK's
optimization coordinates or optimizer.  Hessians may be supplied in Cartesian,
SYCART, SONIC, or a common mixed basis; all finite-difference geometries are
returned in Cartesian space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np


Representation = Literal["cartesian", "sycart", "sonic", "mixed"]
Stencil = Literal["forward", "central"]


@dataclass(frozen=True)
class LocalModeSet:
    """Physical local modes used only for finite-difference displacements."""

    eigenvalues: np.ndarray
    cartesian_modes: np.ndarray
    symmetry_labels: tuple[str, ...]
    representation: str

    def __post_init__(self) -> None:
        values = np.asarray(self.eigenvalues, dtype=float).reshape(-1)
        modes = np.asarray(self.cartesian_modes, dtype=float)
        if modes.ndim != 2 or modes.shape[1] != values.size:
            raise ValueError("cartesian_modes must have shape (ncart, nmodes)")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(modes)):
            raise ValueError("local modes must be finite")
        norms = np.linalg.norm(modes, axis=0)
        if np.any(norms <= 1.0e-14):
            raise ValueError("local modes cannot contain zero vectors")
        modes = modes / norms
        labels = tuple(str(x) for x in self.symmetry_labels)
        if labels and len(labels) != values.size:
            raise ValueError("one symmetry label is required per local mode")
        object.__setattr__(self, "eigenvalues", values)
        object.__setattr__(self, "cartesian_modes", modes)
        object.__setattr__(self, "symmetry_labels", labels or ("" for _ in values))
        object.__setattr__(self, "representation", str(self.representation))


@dataclass(frozen=True)
class AdaptiveStepConfig:
    energy_error: float
    minimum_step: float = 1.0e-5
    maximum_step: float = 2.0e-2
    trust_radius: float | None = None
    curvature_floor: float = 1.0e-10

    def __post_init__(self) -> None:
        if not math.isfinite(self.energy_error) or self.energy_error <= 0:
            raise ValueError("energy_error must be positive")
        if not 0 < self.minimum_step <= self.maximum_step:
            raise ValueError("invalid finite-difference step bounds")
        if self.trust_radius is not None and self.trust_radius <= 0:
            raise ValueError("trust_radius must be positive")


@dataclass(frozen=True)
class FiniteDifferenceStep:
    step: float
    derivative_order: int
    stencil: str
    model: str
    branch: str = "symmetric"


@dataclass(frozen=True)
class NumericalGradientResult:
    """Energy-only gradient reconstructed in the supplied modal subspace."""

    gradient: np.ndarray
    directional_derivatives: np.ndarray
    steps: tuple[FiniteDifferenceStep, ...]
    evaluated_displacements: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class NumericalSecondDerivativeResult:
    """Directional second derivatives reconstructed from two-sided energies."""

    second_derivatives: np.ndarray
    steps: tuple[FiniteDifferenceStep, ...]
    evaluated_displacements: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class OneSidedCurvatureEstimate:
    """One-sided modal curvature corrected with energy-gradient consistency."""

    curvature: float
    third_derivative: float
    raw_gradient_difference: float
    energy_gradient_curvature: float
    correction_applied: bool = True


@dataclass(frozen=True)
class OneSidedHigherDerivativeResult:
    """Higher derivatives obtained from one-sided energy/gradient/Hessian data.

    ``third_semidiagonal[i, j, k]`` is the derivative
    :math:`d^3E/(dq_i dq_j dq_k)` associated with a displacement along mode
    ``k``.  ``fourth_semidiagonal[i, j, k]`` is the corresponding
    :math:`d^4E/(dq_i dq_j dq_k^2)`.  The latter uses the optional ``+2h``
    point; the diagonal fourth derivatives are also available from a single
    ``+h`` energy, gradient, and Hessian evaluation.
    """

    third_semidiagonal: np.ndarray
    fourth_semidiagonal: np.ndarray
    fourth_diagonal: np.ndarray
    fifth_diagonal: np.ndarray
    sixth_diagonal: np.ndarray
    steps: tuple[FiniteDifferenceStep, ...]
    used_second_one_sided_point: bool


@dataclass(frozen=True)
class TwoSidedDiagonalDerivativeResult:
    """Diagonal derivatives through sixth order from energy and gradients."""

    second_diagonal: np.ndarray
    third_diagonal: np.ndarray
    fourth_diagonal: np.ndarray
    fifth_diagonal: np.ndarray
    sixth_diagonal: np.ndarray
    steps: tuple[FiniteDifferenceStep, ...]


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Hessian must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("Hessian must be finite")
    return 0.5 * (array + array.T)


def _metric_inverse_sqrt(metric: np.ndarray, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrized(metric))
    if np.any(values <= floor):
        raise ValueError("coordinate metric must be positive definite")
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def local_modes(
    hessian: Sequence[Sequence[float]] | np.ndarray,
    *,
    representation: Representation,
    basis_to_cartesian: np.ndarray | None = None,
    metric: np.ndarray | None = None,
    symmetry_labels: Sequence[str] = (),
    curvature_floor: float = 1.0e-10,
) -> LocalModeSet:
    """Return normalized Cartesian modes from a declared Hessian representation.

    ``basis_to_cartesian`` maps a displacement in the supplied representation to
    Cartesian coordinates.  For SYCART it is normally an orthonormal basis.  For
    SONIC, ``metric`` is required when the basis is not orthonormal.  A mixed
    Hessian must already be assembled in its declared common tangent basis; its
    transformation is supplied by ``basis_to_cartesian``.
    """

    rep = str(representation).strip().casefold()
    if rep not in {"cartesian", "sycart", "sonic", "mixed"}:
        raise ValueError("representation must be cartesian, sycart, sonic, or mixed")
    h = _symmetrized(np.asarray(hessian, dtype=float))
    if rep == "sonic" and metric is not None:
        m_inv_sqrt = _metric_inverse_sqrt(np.asarray(metric, dtype=float), curvature_floor)
        values, whitened = np.linalg.eigh(m_inv_sqrt @ h @ m_inv_sqrt)
        vectors = m_inv_sqrt @ whitened
    else:
        values, vectors = np.linalg.eigh(h)
    if basis_to_cartesian is None:
        if rep != "cartesian":
            raise ValueError("non-Cartesian Hessians require basis_to_cartesian")
        cartesian_modes = vectors
    else:
        basis = np.asarray(basis_to_cartesian, dtype=float)
        if basis.ndim != 2 or basis.shape[1] != vectors.shape[0]:
            raise ValueError("basis_to_cartesian has incompatible dimensions")
        cartesian_modes = basis @ vectors
    return LocalModeSet(values, cartesian_modes, tuple(symmetry_labels), rep)


def classify_morse_branch(e0: float, e1: float, *, tolerance: float = 0.0) -> str:
    """Classify the sampled direction without moving the derivative center."""

    delta = float(e1) - float(e0)
    if not math.isfinite(delta):
        raise ValueError("energies must be finite")
    if delta < -abs(tolerance):
        return "downhill-toward-minimum"
    if delta > abs(tolerance):
        return "uphill-away-from-minimum"
    return "locally-flat"


def choose_step(
    curvature: float,
    config: AdaptiveStepConfig,
    *,
    derivative_order: int = 1,
    stencil: Stencil = "forward",
    third_derivative: float | None = None,
    fourth_derivative: float | None = None,
    model: str = "curvature",
    branch: str = "symmetric",
) -> FiniteDifferenceStep:
    """Choose a noise-balanced step for a specified derivative and stencil."""

    if derivative_order not in {1, 2}:
        raise ValueError("derivative_order must be 1 or 2")
    if stencil not in {"forward", "central"}:
        raise ValueError("stencil must be forward or central")
    kappa = max(abs(float(curvature)), config.curvature_floor)
    noise = config.energy_error
    if derivative_order == 1 and stencil == "forward":
        # Balance the O(h) forward-difference truncation against energy noise.
        raw = math.sqrt(2.0 * noise / kappa)
    elif derivative_order == 1 and stencil == "central":
        if third_derivative is None or abs(third_derivative) < config.curvature_floor:
            raw = (2.0 * noise / kappa) ** 0.5
        else:
            raw = (6.0 * noise / abs(float(third_derivative))) ** (1.0 / 3.0)
    else:
        quartic = abs(float(fourth_derivative)) if fourth_derivative is not None else kappa
        quartic = max(quartic, config.curvature_floor)
        raw = (12.0 * noise / quartic) ** 0.25
    upper = config.maximum_step
    if config.trust_radius is not None:
        upper = min(upper, config.trust_radius)
    step = min(max(raw, config.minimum_step), upper)
    return FiniteDifferenceStep(step, derivative_order, stencil, model, branch)


def energy_model_parameters(
    e0: float,
    e1: float,
    e2: float,
    *,
    step: float,
    curvature: float,
    model: Literal["morse", "even-exponential"],
) -> dict[str, float | str]:
    """Estimate local model parameters from three positive-direction energies.

    The fit is intentionally conservative: the Hessian supplies the curvature,
    while the energy ratio supplies the local anharmonic scale.  Ill-conditioned
    ratios raise ``ValueError`` so the caller can retain the curvature fallback.
    """

    if step <= 0 or not all(math.isfinite(float(x)) for x in (e0, e1, e2, curvature)):
        raise ValueError("invalid local model data")
    d1 = float(e1) - float(e0)
    d2 = float(e2) - float(e0)
    if abs(d1) <= 1.0e-14:
        raise ValueError("energy ratio is ill-conditioned")
    ratio = d2 / d1
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError("energy ratio does not define a stable local fit")
    if model == "morse":
        root = math.sqrt(ratio)
        if root <= 1.0:
            raise ValueError("invalid Morse energy ratio")
        a = -math.log(root - 1.0) / step
        return {
            "model": "morse",
            "a": a,
            "curvature": float(curvature),
            "cubic": -3.0 * float(curvature) * a,
            "quartic": 7.0 * float(curvature) * a * a,
        }
    if model == "even-exponential":
        # A robust initial alpha from the observed quartic deviation.
        alpha = max(abs(2.0 * (d2 - 4.0 * d1) / (float(curvature) * step**4)), 1.0e-14)
        return {
            "model": "even-exponential",
            "alpha": alpha,
            "curvature": float(curvature),
            "quartic": -6.0 * float(curvature) * alpha,
        }
    raise ValueError("model must be morse or even-exponential")


def numerical_gradient_from_energies(
    coordinates: np.ndarray,
    energy: float,
    modes: LocalModeSet,
    evaluate_energy,
    config: AdaptiveStepConfig,
    *,
    stencil: Stencil = "forward",
    third_derivatives: Sequence[float | None] | None = None,
    fourth_derivatives: Sequence[float | None] | None = None,
    models: Sequence[str] | None = None,
    branches: Sequence[str] | None = None,
    step_selection_order: int | None = None,
) -> NumericalGradientResult:
    """Evaluate an energy-only numerical gradient along local Cartesian modes.

    ``evaluate_energy`` receives an ``(natoms, 3)`` Cartesian geometry and must
    return one scalar energy in the same units as ``energy``.  The returned
    gradient is the minimum-norm Cartesian gradient in the span of the supplied
    modes.  This function is intentionally independent of LINK's optimizer and
    can therefore be used by external or resident calculators alike.
    """

    x0 = np.asarray(coordinates, dtype=float)
    if x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError("coordinates must have shape natoms x 3")
    if not math.isfinite(float(energy)):
        raise ValueError("central energy must be finite")
    count = modes.eigenvalues.size
    third = tuple(third_derivatives or (None,) * count)
    fourth = tuple(fourth_derivatives or (None,) * count)
    model_values = tuple(models or ("curvature",) * count)
    branch_values = tuple(branches or modes.symmetry_labels)
    if not all(len(values) == count for values in (third, fourth, model_values, branch_values)):
        raise ValueError("per-mode metadata must match the number of modes")
    directions = np.asarray(modes.cartesian_modes, dtype=float)
    selected_order = (
        int(step_selection_order)
        if step_selection_order is not None
        else 1
    )
    if selected_order not in {1, 2}:
        raise ValueError("step_selection_order must be 1 or 2")
    directional = np.empty(count, dtype=float)
    selected: list[FiniteDifferenceStep] = []
    displacements: list[np.ndarray] = []
    for index, (direction, curvature) in enumerate(zip(directions.T, modes.eigenvalues)):
        selected_step = choose_step(
            curvature,
            config,
            derivative_order=selected_order,
            stencil=stencil,
            third_derivative=third[index],
            fourth_derivative=fourth[index],
            model=model_values[index],
            branch=branch_values[index],
        )
        h = selected_step.step
        plus = x0.reshape(-1) + h * direction
        if stencil == "forward":
            derivative = (float(evaluate_energy(plus.reshape(x0.shape))) - float(energy)) / h
            sampled = (plus - x0.reshape(-1)).copy()
        else:
            minus = x0.reshape(-1) - h * direction
            plus_energy = float(evaluate_energy(plus.reshape(x0.shape)))
            minus_energy = float(evaluate_energy(minus.reshape(x0.shape)))
            derivative = (plus_energy - minus_energy) / (2.0 * h)
            sampled = np.concatenate((minus - x0.reshape(-1), plus - x0.reshape(-1)))
        directional[index] = derivative
        selected.append(selected_step)
        displacements.append(sampled)
    gram = directions.T @ directions
    coefficients = np.linalg.lstsq(gram, directional, rcond=None)[0]
    return NumericalGradientResult(
        gradient=directions @ coefficients,
        directional_derivatives=directional,
        steps=tuple(selected),
        evaluated_displacements=tuple(displacements),
    )


def numerical_second_derivatives_from_energies(
    coordinates: np.ndarray,
    energy: float,
    modes: LocalModeSet,
    evaluate_energy,
    config: AdaptiveStepConfig,
    *,
    fourth_derivatives: Sequence[float | None] | None = None,
    models: Sequence[str] | None = None,
) -> NumericalSecondDerivativeResult:
    """Evaluate diagonal modal curvatures from mandatory two-sided energies."""

    x0 = np.asarray(coordinates, dtype=float)
    if x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError("coordinates must have shape natoms x 3")
    if not math.isfinite(float(energy)):
        raise ValueError("central energy must be finite")
    count = modes.eigenvalues.size
    fourth = tuple(fourth_derivatives or (None,) * count)
    model_values = tuple(models or ("curvature",) * count)
    if len(fourth) != count or len(model_values) != count:
        raise ValueError("per-mode metadata must match the number of modes")
    values = np.empty(count, dtype=float)
    selected: list[FiniteDifferenceStep] = []
    displacements: list[np.ndarray] = []
    flat = x0.reshape(-1)
    for index, (direction, curvature) in enumerate(
        zip(modes.cartesian_modes.T, modes.eigenvalues)
    ):
        selected_step = choose_step(
            curvature,
            config,
            derivative_order=2,
            stencil="central",
            fourth_derivative=fourth[index],
            model=model_values[index],
        )
        h = selected_step.step
        plus = flat + h * direction
        minus = flat - h * direction
        plus_energy = float(evaluate_energy(plus.reshape(x0.shape)))
        minus_energy = float(evaluate_energy(minus.reshape(x0.shape)))
        values[index] = (plus_energy - 2.0 * float(energy) + minus_energy) / h**2
        selected.append(selected_step)
        displacements.append(np.concatenate((minus - flat, plus - flat)))
    return NumericalSecondDerivativeResult(
        second_derivatives=values,
        steps=tuple(selected),
        evaluated_displacements=tuple(displacements),
    )


def one_sided_curvature_from_energy_gradient(
    energy0: float,
    energy1: float,
    gradient0: float,
    gradient1: float,
    step: float,
) -> OneSidedCurvatureEstimate:
    """Estimate a modal curvature from one displaced energy-gradient point.

    The central energy and gradient are those at the geometry where the
    derivative is required.  The returned correction cancels the leading
    third-derivative term in the forward gradient difference:

        A = (g1-g0)/h,
        B = 2*(E1-E0-g0*h)/h**2,
        T = 6*(A-B)/h,
        H = 3*B - 2*A.

    ``step`` and the energies must use one consistent coordinate unit.
    """

    values = (energy0, energy1, gradient0, gradient1, step)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("one-sided curvature inputs must be finite")
    h = float(step)
    if h <= 0.0:
        raise ValueError("one-sided curvature step must be positive")
    raw = (float(gradient1) - float(gradient0)) / h
    from_energy = 2.0 * (
        float(energy1) - float(energy0) - float(gradient0) * h
    ) / h**2
    third = 6.0 * (raw - from_energy) / h
    corrected = 3.0 * from_energy - 2.0 * raw
    return OneSidedCurvatureEstimate(
        curvature=corrected,
        third_derivative=third,
        raw_gradient_difference=raw,
        energy_gradient_curvature=from_energy,
    )


def one_sided_higher_derivatives(
    coordinates: np.ndarray,
    energy0: float,
    gradient0: np.ndarray,
    hessian0: np.ndarray,
    modes: LocalModeSet,
    evaluate_energy_gradient_hessian,
    config: AdaptiveStepConfig,
    *,
    use_second_one_sided_point: bool = True,
) -> OneSidedHigherDerivativeResult:
    """Build third and fourth derivatives with a one-sided stencil.

    ``evaluate_energy_gradient_hessian(geometry)`` returns an energy, Cartesian
    gradient, and Cartesian Hessian.  With only ``+h`` this gives every third
    derivative slice and the diagonal fourth derivative through Taylor
    identities.  With ``+2h`` it also gives the complete fourth
    semi-diagonal slice while remaining one-sided.
    """

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape natoms x 3")
    gradient0 = np.asarray(gradient0, dtype=float).reshape(-1)
    hessian0 = _symmetrized(np.asarray(hessian0, dtype=float))
    flat = coordinates.reshape(-1)
    directions = np.asarray(modes.cartesian_modes, dtype=float)
    count = directions.shape[1]
    if gradient0.size != flat.size or hessian0.shape != (flat.size, flat.size):
        raise ValueError("central data do not match coordinates")
    if not math.isfinite(float(energy0)):
        raise ValueError("central energy must be finite")
    h0m = directions.T @ hessian0 @ directions
    g0m = directions.T @ gradient0
    third = np.empty((count, count, count), dtype=float)
    fourth = np.empty((count, count, count), dtype=float)
    diagonal = np.empty(count, dtype=float)
    steps: list[FiniteDifferenceStep] = []
    for k, direction in enumerate(directions.T):
        selected = choose_step(
            modes.eigenvalues[k], config, derivative_order=2,
            stencil="forward", model="one-sided-hessian"
        )
        h = selected.step
        x1 = flat + h * direction
        e1, g1, hessian1 = evaluate_energy_gradient_hessian(x1.reshape(coordinates.shape))
        h1m = directions.T @ _symmetrized(np.asarray(hessian1, dtype=float)) @ directions
        g1m = directions.T @ np.asarray(g1, dtype=float).reshape(-1)
        e1 = float(e1)
        # One-sided third derivative from the Hessian; this is the leading
        # estimate for the complete T[:,:,k] slice.
        t_h = (h1m - h0m) / h
        third[:, :, k] = t_h
        # The scalar Taylor identities combine E, g, and H to cancel the
        # leading one-sided error for the diagonal fourth derivative.
        gk0, gk1 = g0m[k], g1m[k]
        hkk0 = h0m[k, k]
        t_g = 2.0 * (gk1 - gk0 - hkk0 * h) / h**2
        t_e = 6.0 * (e1 - float(energy0) - gk0 * h - 0.5 * hkk0 * h**2) / h**3
        q_hg = 6.0 * (t_h[k, k] - t_g) / h
        q_eg = 12.0 * (t_g - t_e) / h
        diagonal[k] = 0.5 * (q_hg + q_eg)
        fourth[:, :, k] = np.nan
        if use_second_one_sided_point:
            x2 = flat + 2.0 * h * direction
            _, _, hessian2 = evaluate_energy_gradient_hessian(x2.reshape(coordinates.shape))
            h2m = directions.T @ _symmetrized(np.asarray(hessian2, dtype=float)) @ directions
            d1 = (h1m - h0m) / h
            d2 = (h2m - h0m) / (2.0 * h)
            third[:, :, k] = 2.0 * d1 - d2
            fourth[:, :, k] = 2.0 * (d2 - d1) / h
            diagonal[k] = fourth[k, k, k]
        else:
            fourth[k, k, k] = diagonal[k]
        steps.append(selected)
    return OneSidedHigherDerivativeResult(
        third_semidiagonal=third,
        fourth_semidiagonal=fourth,
        fourth_diagonal=diagonal,
        fifth_diagonal=np.full(count, np.nan),
        sixth_diagonal=np.full(count, np.nan),
        steps=tuple(steps),
        used_second_one_sided_point=use_second_one_sided_point,
    )


def two_sided_higher_derivatives(
    coordinates: np.ndarray,
    energy0: float,
    gradient0: np.ndarray,
    hessian0: np.ndarray,
    modes: LocalModeSet,
    evaluate_energy_gradient_hessian,
    config: AdaptiveStepConfig,
) -> OneSidedHigherDerivativeResult:
    """Build third and fourth semi-diagonal derivatives from ``+h`` and ``-h``.

    For every mode ``k`` the Hessians at the two displaced geometries provide

    ``T[:,:,k] = (H(+h)-H(-h))/(2h)`` and
    ``Q[:,:,k,k] = (H(+h)+H(-h)-2H(0))/h**2``.

    For fully diagonal terms, energy, gradient, and Hessian are combined.
    Their independent Taylor estimates also provide diagonal fifth and sixth
    derivatives without additional displaced points.

    The step is selected with the second-derivative criterion.  Energy and
    gradients are still evaluated and returned only through the common
    calculator contract; the Hessian stencil itself is the quantity used for
    the higher derivatives.  Compared with the one-sided ``+h/+2h`` option,
    this costs the same number of displaced points but has better cancellation
    of odd truncation terms.
    """

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape natoms x 3")
    gradient0 = np.asarray(gradient0, dtype=float).reshape(-1)
    hessian0 = _symmetrized(np.asarray(hessian0, dtype=float))
    flat = coordinates.reshape(-1)
    directions = np.asarray(modes.cartesian_modes, dtype=float)
    count = directions.shape[1]
    if gradient0.size != flat.size or hessian0.shape != (flat.size, flat.size):
        raise ValueError("central data do not match coordinates")
    if not math.isfinite(float(energy0)):
        raise ValueError("central energy must be finite")
    h0m = directions.T @ hessian0 @ directions
    g0m = directions.T @ gradient0
    third = np.empty((count, count, count), dtype=float)
    fourth = np.empty((count, count, count), dtype=float)
    third_diagonal = np.empty(count, dtype=float)
    fourth_diagonal = np.empty(count, dtype=float)
    fifth_diagonal = np.empty(count, dtype=float)
    sixth_diagonal = np.empty(count, dtype=float)
    steps: list[FiniteDifferenceStep] = []
    for k, direction in enumerate(directions.T):
        selected = choose_step(
            modes.eigenvalues[k], config, derivative_order=2,
            stencil="central", model="two-sided-hessian"
        )
        h = selected.step
        plus = flat + h * direction
        minus = flat - h * direction
        ep, gp, hp = evaluate_energy_gradient_hessian(plus.reshape(coordinates.shape))
        em, gm, hm = evaluate_energy_gradient_hessian(minus.reshape(coordinates.shape))
        ep, em = float(ep), float(em)
        gp = directions.T @ np.asarray(gp, dtype=float).reshape(-1)
        gm = directions.T @ np.asarray(gm, dtype=float).reshape(-1)
        hp = directions.T @ _symmetrized(np.asarray(hp, dtype=float)) @ directions
        hm = directions.T @ _symmetrized(np.asarray(hm, dtype=float)) @ directions
        t_h = (hp - hm) / (2.0 * h)
        q_h = (hp + hm - 2.0 * h0m) / h**2
        third[:, :, k] = t_h
        fourth[:, :, k] = q_h
        g0k = float(g0m[k])
        h0k = float(h0m[k, k])
        t_e = 3.0 * (ep - em - 2.0 * g0k * h) / h**3
        t_h_diag = float(t_h[k, k])
        fifth_diagonal[k] = (60.0 / 7.0) * (t_h_diag - t_e) / h**2
        third_diagonal[k] = (-3.0 * t_h_diag + 10.0 * t_e) / 7.0
        q_e = 12.0 * (ep + em - 2.0 * float(energy0) - h0k * h**2) / h**4
        q_g = 3.0 * (float(gp[k] - gm[k]) - 2.0 * h0k * h) / h**3
        sixth_diagonal[k] = 60.0 * (q_g - q_e) / h**2
        fourth_diagonal[k] = 3.0 * q_e - 2.0 * q_g
        steps.append(selected)
    return OneSidedHigherDerivativeResult(
        third_semidiagonal=third,
        fourth_semidiagonal=fourth,
        fourth_diagonal=fourth_diagonal,
        fifth_diagonal=fifth_diagonal,
        sixth_diagonal=sixth_diagonal,
        steps=tuple(steps),
        used_second_one_sided_point=False,
    )


def two_sided_diagonal_derivatives_from_energy_gradients(
    coordinates: np.ndarray,
    energy0: float,
    gradient0: np.ndarray,
    hessian0: np.ndarray,
    modes: LocalModeSet,
    evaluate_energy_gradient,
    config: AdaptiveStepConfig,
) -> TwoSidedDiagonalDerivativeResult:
    """Obtain diagonal derivatives through order six using only E and g.

    Energies and gradients are evaluated at ``+h`` and ``-h``.  The even and
    odd combinations provide independent estimates which are combined to
    cancel the leading higher-order contamination:

    ``T = (5 A - 3 B)/2``, ``V5 = 30 (B-A)/h**2`` from energy-odd ``A`` and
    gradient-even ``B``; and ``Q = 3 C - 2 D``,
    ``V6 = 60 (D-C)/h**2`` from energy-even ``C`` and gradient-odd ``D``.

    This is the efficient gradient-based route for diagonal derivatives.  It
    does not provide mixed/semi-diagonal tensor elements; those require the
    displaced Hessians handled by :func:`two_sided_higher_derivatives`.
    """

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape natoms x 3")
    gradient0 = np.asarray(gradient0, dtype=float).reshape(-1)
    hessian0 = _symmetrized(np.asarray(hessian0, dtype=float))
    flat = coordinates.reshape(-1)
    directions = np.asarray(modes.cartesian_modes, dtype=float)
    count = directions.shape[1]
    if gradient0.size != flat.size or hessian0.shape != (flat.size, flat.size):
        raise ValueError("central data do not match coordinates")
    g0m = directions.T @ gradient0
    h0m = directions.T @ hessian0 @ directions
    second = np.empty(count, dtype=float)
    third = np.empty(count, dtype=float)
    fourth = np.empty(count, dtype=float)
    fifth = np.empty(count, dtype=float)
    sixth = np.empty(count, dtype=float)
    steps: list[FiniteDifferenceStep] = []
    for k, direction in enumerate(directions.T):
        selected = choose_step(
            modes.eigenvalues[k], config, derivative_order=2,
            stencil="central", model="two-sided-gradient"
        )
        h = selected.step
        plus = flat + h * direction
        minus = flat - h * direction
        ep, gp = evaluate_energy_gradient(plus.reshape(coordinates.shape))
        em, gm = evaluate_energy_gradient(minus.reshape(coordinates.shape))
        ep, em = float(ep), float(em)
        gp = float(directions[:, k] @ np.asarray(gp, dtype=float).reshape(-1))
        gm = float(directions[:, k] @ np.asarray(gm, dtype=float).reshape(-1))
        h0 = float(h0m[k, k])
        g0 = float(g0m[k])
        a = 3.0 * (ep - em - 2.0 * g0 * h) / h**3
        b = (gp + gm - 2.0 * g0) / h**2
        c = 12.0 * (ep + em - 2.0 * float(energy0) - h0 * h**2) / h**4
        d = 3.0 * (gp - gm - 2.0 * h0 * h) / h**3
        third[k] = 2.5 * a - 1.5 * b
        fifth[k] = 30.0 * (b - a) / h**2
        fourth[k] = 3.0 * c - 2.0 * d
        sixth[k] = 60.0 * (d - c) / h**2
        second[k] = h0
        steps.append(selected)
    return TwoSidedDiagonalDerivativeResult(second, third, fourth, fifth, sixth, tuple(steps))


__all__ = [
    "AdaptiveStepConfig",
    "FiniteDifferenceStep",
    "LocalModeSet",
    "NumericalGradientResult",
    "NumericalSecondDerivativeResult",
    "OneSidedCurvatureEstimate",
    "OneSidedHigherDerivativeResult",
    "TwoSidedDiagonalDerivativeResult",
    "choose_step",
    "classify_morse_branch",
    "energy_model_parameters",
    "local_modes",
    "numerical_gradient_from_energies",
    "numerical_second_derivatives_from_energies",
    "one_sided_curvature_from_energy_gradient",
    "one_sided_higher_derivatives",
    "two_sided_higher_derivatives",
    "two_sided_diagonal_derivatives_from_energy_gradients",
]
