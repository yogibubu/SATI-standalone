"""IR/Raman transition moments and intensity borrowing for SONIC GVPT2 polyads."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import permutations, product
from math import factorial, sqrt
import os
from threading import Lock
from typing import Literal

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_gf.harmonic import CM_PER_HARTREE
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM

from .curvilinear_vpt2 import (
    SonicGVPT2Result, SonicVPT2Input, _SelectedOperatorEvaluator, _apply,
    _canonical_cubic_terms, _cubic_couplings, _factorized_operator_element,
    _modalize_sonic_field,
)
from .chiroptical import ROABackscattering, roa_scp_backscattering, vcd_rotatory_strength
from .property_spectroscopy import GAUSSIAN_IR_CONVERSION_KM_MOL


IR_TRANSITION_FACTOR = (
    2 * GAUSSIAN_IR_CONVERSION_KM_MOL * CM_PER_HARTREE
    / HESSIAN_EIGENVALUE_TO_CM**2
)


@dataclass(frozen=True)
class SonicPropertyDerivatives:
    """Property Taylor coefficients in dimensionless SONIC GF modes."""

    first: np.ndarray
    second: np.ndarray | None = None
    third: np.ndarray | None = None
    fourth: np.ndarray | None = None


@dataclass(frozen=True)
class SonicDiagonalPropertyExpectation:
    """Successive diagonal-property averages in one vibrational state.

    ``standard`` is the established leading vibrational correction.  ``complete_qff``
    adds the next nonzero term for the supplied quartic Hamiltonian.  It is ``None``
    when the fourth property derivative is absent.  A physical expansion complete
    at that order additionally needs the optional quintic Hamiltonian coupling.
    """

    equilibrium: np.ndarray
    standard: np.ndarray
    complete_qff: np.ndarray | None
    next_nonzero_correction: np.ndarray | None
    rovibrational_correction: np.ndarray | None = None
    rovibrational_qff_correction: np.ndarray | None = None
    rovibrational_supplied_h5_correction: np.ndarray | None = None
    rovibrational_morse_h5_correction: np.ndarray | None = None
    complete_with_supplied_h5: np.ndarray | None = None
    complete_qff_morse_h5: np.ndarray | None = None
    morse_quartic_relative_rms: float | None = None


@dataclass(frozen=True)
class SonicMomentumPropertyDerivatives:
    """Coefficients of ``{b_a,i(q),p_i}/2`` in dimensionless modes.

    ``first[a,i]=b_i``, ``second[a,i,j]=b_i,j``, and
    ``third[a,i,j,k]=b_i,jk`` with the last two indices symmetric.
    The supplied axial tensors must include electronic and nuclear
    contributions in one common origin and body-frame gauge.
    """

    first: np.ndarray
    second: np.ndarray | None = None
    third: np.ndarray | None = None


@dataclass(frozen=True)
class SonicROAPropertySurfaces:
    """Frequency-matched dynamic response surfaces for off-resonant ROA.

    ``alpha`` and ``g_prime`` have nine row-major Cartesian components;
    ``a`` has 27 in ``A[sigma,rho,eta]`` order.  All three must share
    the same gauge, origin, coordinate frame, and incident frequency.
    """

    alpha: SonicPropertyDerivatives
    g_prime: SonicPropertyDerivatives
    a: SonicPropertyDerivatives
    incident_frequency_au: float


def modal_property_derivatives_from_packed_mass_weighted(
    packed: np.ndarray,
    frequencies_cm: np.ndarray,
    *,
    property_rotation: np.ndarray | None = None,
    mode_signs: np.ndarray | None = None,
) -> SonicPropertyDerivatives:
    """Read full-precision D1/D2/D3 stored in a packed mass-weighted mode basis.

    The packed order is equilibrium value, D1[i], D2[i,j] for i >= j,
    then D3[i,j,k] for i >= j >= k; every entry has the same number of
    Cartesian property components. ``property_rotation`` maps those components
    into the fixed reference frame and ``mode_signs`` maps the source normal-mode
    phases to the acquired SONIC modes. Derivatives are returned with respect
    to dimensionless oscillator coordinates.
    """
    frequencies = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    data = np.asarray(packed, dtype=float)
    n = len(frequencies)
    count = 1 + n + n * (n + 1) // 2 + n * (n + 1) * (n + 2) // 6
    if (n == 0 or np.any(frequencies <= 0) or not np.all(np.isfinite(frequencies))
            or data.ndim != 2 or data.shape[0] != count or data.shape[1] < 1
            or not np.all(np.isfinite(data))):
        raise ValueError("packed modal property derivatives have invalid shape or values")
    components = data.shape[1]
    rotation = (np.eye(components) if property_rotation is None
                else np.asarray(property_rotation, dtype=float))
    signs = (np.ones(n) if mode_signs is None else np.asarray(mode_signs, dtype=float))
    if (rotation.shape != (components, components) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(components), atol=1e-8)
            or signs.shape != (n,) or not np.all(np.isin(signs, (-1.0, 1.0)))):
        raise ValueError("invalid property rotation or modal phases")
    gamma = np.sqrt(frequencies * CM_PER_HARTREE) / HESSIAN_EIGENVALUE_TO_CM
    first = np.zeros((components, n))
    second = np.zeros((components, n, n))
    third = np.zeros((components, n, n, n))
    cursor = 1
    for i in range(n):
        first[:, i] = rotation @ data[cursor] * signs[i] / gamma[i]
        cursor += 1
    for i in range(n):
        for j in range(i + 1):
            value = rotation @ data[cursor] * signs[i] * signs[j] / (gamma[i] * gamma[j])
            second[:, i, j] = second[:, j, i] = value
            cursor += 1
    for i in range(n):
        for j in range(i + 1):
            for k in range(j + 1):
                value = (rotation @ data[cursor] * signs[i] * signs[j] * signs[k]
                         / (gamma[i] * gamma[j] * gamma[k]))
                for indices in set(permutations((i, j, k))):
                    third[(slice(None), *indices)] = value
                cursor += 1
    return SonicPropertyDerivatives(first, second, third)


def project_sonic_property_derivatives(
    derivatives: SonicPropertyDerivatives,
    mode_action: np.ndarray,
    property_action: np.ndarray,
) -> SonicPropertyDerivatives:
    """Average D1/D2/D3 over an involutive point-group operation.

    ``mode_action`` maps dimensionless modes under the operation; the
    property action maps the Cartesian components of a vector property.
    A non-diagonal mode action is allowed for near-degenerate modes.
    This projects only the property surface; it does not alter the PES.
    """
    first = np.asarray(derivatives.first, dtype=float)
    if first.ndim != 2 or first.shape[0] != 3:
        raise ValueError("symmetry projection requires three-component D1")
    n = first.shape[1]
    mode = np.asarray(mode_action, dtype=float)
    property_matrix = np.asarray(property_action, dtype=float)
    for name, matrix, dimension in (("mode", mode, n), ("property", property_matrix, 3)):
        if (matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix))
                or not np.allclose(matrix @ matrix, np.eye(dimension), atol=1e-5, rtol=0)):
            raise ValueError(f"{name} symmetry action must be a finite involution")

    def project(value: np.ndarray | None, rank: int) -> np.ndarray | None:
        if value is None:
            return None
        tensor = np.asarray(value, dtype=float)
        if tensor.shape != (3,) + (n,) * rank or not np.all(np.isfinite(tensor)):
            raise ValueError(f"D{rank} has invalid shape or nonfinite entries")
        transformed = np.einsum("ab,b...->a...", property_matrix, tensor)
        for axis in range(rank):
            transformed = np.moveaxis(
                np.tensordot(transformed, mode, axes=([axis + 1], [0])),
                -1, axis + 1,
            )
        return 0.5 * (tensor + transformed)

    return SonicPropertyDerivatives(project(first, 1), project(derivatives.second, 2),
                                    project(derivatives.third, 3),
                                    project(derivatives.fourth, 4))


@dataclass(frozen=True)
class SonicGVPT2Spectrum:
    states: tuple[tuple[int, ...], ...]
    transition_energies_cm: np.ndarray
    dipole_moments_au: np.ndarray | None
    ir_intensities_km_mol: np.ndarray | None
    polarizability_tensors_au: np.ndarray | None
    raman_strength_ang6: np.ndarray | None
    polyad_eigenvectors: np.ndarray
    mechanical_correction: str
    dipole_moment_orders_au: np.ndarray | None = None
    ir_intensity_orders_km_mol: np.ndarray | None = None
    initial_state: tuple[int, ...] | None = None
    magnetic_moments_au: np.ndarray | None = None
    vcd_rotatory_strength_au: np.ndarray | None = None
    magnetic_moment_orders_au: np.ndarray | None = None
    vcd_rotatory_strength_orders_au: np.ndarray | None = None
    roa_scp_backscatter: ROABackscattering | None = None
    evaluated_states: tuple[tuple[int, ...], ...] | None = None


def mix_sonic_polyad_property_moments(
    result: SonicGVPT2Result, basis_moments: np.ndarray,
) -> np.ndarray:
    """Apply sparse PT2D polyad blocks to any real or complex property tensor.

    The first axis enumerates the selected perturbative states. Remaining
    axes may contain Cartesian tensor components and incident frequencies.
    This shared operation is the mixing step for IR/Raman and for later
    magnetic-dipole or optical-activity response providers.
    """
    moments = np.asarray(basis_moments)
    if moments.ndim < 1 or moments.shape[0] != len(result.states):
        raise ValueError("property moments must have one leading entry per GVPT2 state")
    mixed = moments.copy()
    for block in result.blocks:
        members = np.asarray(block["indices"], dtype=int)
        vectors = np.asarray(block["eigenvectors"], dtype=float)
        if vectors.shape != (len(members), len(members)):
            raise ValueError("GVPT2 polyad has invalid eigenvectors")
        mixed[members] = (vectors.T @ moments[members].reshape(len(members), -1)).reshape(
            moments[members].shape,
        )
    return mixed


def property_derivatives_from_cartesian(
    first_cartesian: np.ndarray,
    cartesian_displacements_per_q: np.ndarray,
) -> SonicPropertyDerivatives:
    """Project Cartesian property gradients onto dimensionless SONIC GF modes.

    ``first_cartesian`` is components × 3N, and the displacement matrix is
    3N × modes in Cartesian bohr per dimensionless oscillator coordinate.
    A six-component Raman tensor uses the order xx, xy, yy, xz, yz, zz.
    """
    first = np.asarray(first_cartesian, dtype=float)
    transform = np.asarray(cartesian_displacements_per_q, dtype=float)
    if first.ndim != 2 or transform.ndim != 2 or first.shape[1] != transform.shape[0]:
        raise ValueError("Cartesian property derivatives and SONIC displacements disagree")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(transform)):
        raise ValueError("Cartesian property derivatives and SONIC modes must be finite")
    return SonicPropertyDerivatives(first @ transform)


def acquire_modal_property_derivatives_from_gradients(
    gradient_at_s,
    displacement_per_q: np.ndarray,
    *,
    component_count: int,
    step_dimensionless: float | np.ndarray = 0.02,
    single_mode_only: bool = False,
) -> SonicPropertyDerivatives:
    """Acquire second/third IR or Raman property derivatives.

    The callback returns the Cartesian or SONIC property Jacobian already
    expressed with respect to the input SONIC coordinates. With
    ``single_mode_only``, the full second derivative and semidiagonal third
    derivative come from plus/minus one-mode shifts of that Jacobian;
    all-distinct third derivatives are zero. Otherwise mixed displacements
    supply the complete third derivative.
    """
    transform = np.asarray(displacement_per_q, dtype=float)
    if transform.ndim != 2 or transform.shape[0] != transform.shape[1]:
        raise ValueError("property stencil requires a square SONIC GF transform")
    n = transform.shape[0]
    steps = np.broadcast_to(np.asarray(step_dimensionless, dtype=float), (n,))
    if component_count < 1 or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("invalid property component count or stencil step")

    def gradient(displacement: np.ndarray) -> np.ndarray:
        value = np.asarray(gradient_at_s(displacement), dtype=float)
        if value.shape != (component_count, n) or not np.all(np.isfinite(value)):
            raise ValueError("property Jacobian callback has invalid shape or nonfinite values")
        return value @ transform

    zero = np.zeros(n)
    first = gradient(zero)
    second = np.zeros((component_count, n, n))
    third = np.zeros((component_count, n, n, n))
    for mode in range(n):
        step = steps[mode]
        displacement = step * transform[:, mode]
        # Acquire the stored negative point first so an exact symmetry
        # operation can supply the positive point on a cache-only restart.
        minus, plus = gradient(-displacement), gradient(displacement)
        second[:, :, mode] = (plus - minus) / (2 * step)
        third[:, :, mode, mode] = (plus - 2 * first + minus) / step**2
    second = (second + second.swapaxes(1, 2)) / 2
    if not single_mode_only:
        for i in range(n):
            for j in range(i + 1, n):
                di, dj = steps[i] * transform[:, i], steps[j] * transform[:, j]
                mixed = (
                    gradient(di + dj) - gradient(di - dj)
                    - gradient(-di + dj) + gradient(-di - dj)
                ) / (4 * steps[i] * steps[j])
                third[:, :, i, j] = third[:, :, j, i] = mixed
    from itertools import permutations

    if single_mode_only:
        single_mode_third = third.copy()
        third.fill(0.0)
        for repeated in range(n):
            for other in range(n):
                derivative = single_mode_third[:, other, repeated, repeated]
                for indices in set(permutations((other, repeated, repeated))):
                    third[(slice(None), *indices)] = derivative
    else:
        third = sum(np.transpose(third, (0, *(axis + 1 for axis in order)))
                    for order in permutations(range(3))) / 6
    return SonicPropertyDerivatives(first, second, third)


def acquire_modal_momentum_derivatives_from_values(
    coefficients_at_s,
    displacement_per_q: np.ndarray,
    *,
    component_count: int = 3,
    step_dimensionless: float | np.ndarray = 0.02,
    single_mode_only: bool = False,
) -> SonicMomentumPropertyDerivatives:
    """Differentiate the coefficients of {b_i(q),p_i}/2 along fixed modes.

    The momentum index is distinct from the coordinate derivative indices;
    only the last two indices of the cubic tensor are symmetrized.  One-mode
    stencils give b_i,jj for all i,j, while pair stencils give b_i,jk.
    """
    from .coefficient_derivatives import acquire_coefficient_derivative_jet
    transform=np.asarray(displacement_per_q,dtype=float)
    if transform.ndim!=2 or transform.shape[0]!=transform.shape[1]:
        raise ValueError("momentum stencil requires a square SONIC GF transform")
    n=len(transform)
    if component_count<1:raise ValueError("positive momentum component count required")
    def coefficients(s):
        result=np.asarray(coefficients_at_s(s),dtype=float)
        if result.shape!=(component_count,n):
            raise ValueError("momentum coefficient provider has invalid shape")
        return result
    jet=acquire_coefficient_derivative_jet(coefficients,transform,
        step_dimensionless=step_dimensionless,single_mode_only=single_mode_only)
    # Preserve the established semidiagonal momentum-property API; callers
    # requesting only single-mode stencils already declare that coverage.
    third=np.where(jet.second_observed,jet.second,0.)
    return SonicMomentumPropertyDerivatives(jet.value,jet.first,third)


def _basis_moments(
    states: tuple[tuple[int, ...], ...], derivatives: SonicPropertyDerivatives,
    components: int,
) -> np.ndarray:
    n = len(states[0])
    arrays = (
        np.asarray(derivatives.first, dtype=float),
        None if derivatives.second is None else np.asarray(derivatives.second, dtype=float),
        None if derivatives.third is None else np.asarray(derivatives.third, dtype=float),
    )
    expected = ((components, n), (components, n, n), (components, n, n, n))
    if any(value is not None and (value.shape != shape or not np.all(np.isfinite(value)))
           for value, shape in zip(arrays, expected)):
        raise ValueError("SONIC property derivatives have invalid shape or nonfinite values")
    from .reduced_transition_polynomials import _property_coefficients, evaluate_reduced_transition
    filled = tuple(np.zeros(shape) if value is None else value
                   for value, shape in zip(arrays, expected, strict=True))
    coefficients = _property_coefficients(*filled, momentum=False)
    ground = (0,) * n
    return np.asarray([sum(evaluate_reduced_transition(p, ground, state, components)
                           for p in coefficients).real for state in states])


def _property_couplings(
    state: tuple[int, ...], derivatives: SonicPropertyDerivatives | SonicMomentumPropertyDerivatives,
    components: int, *, ranks: tuple[int, ...] = (1, 2, 3), momentum: bool = False,
) -> dict[tuple[int, ...], np.ndarray]:
    n = len(state)
    arrays = (derivatives.first, derivatives.second, derivatives.third)
    output: dict[tuple[int, ...], np.ndarray] = {}
    for rank, values in enumerate(arrays, 1):
        if rank not in ranks or values is None:
            continue
        tensor = np.asarray(values, dtype=float)
        if tensor.shape != (components, *([n] * rank)):
            raise ValueError("SONIC property derivative shape disagrees with the mode basis")
        for indices in product(range(n), repeat=rank):
            coefficient = tensor[(slice(None), *indices)] / factorial(rank - 1 if momentum else rank)
            if not np.any(coefficient):
                continue
            if momentum:
                p = (("p", indices[0]),)
                q = tuple(("q", mode) for mode in indices[1:])
                transitions = _apply(state, p + q)
                for target, value in _apply(state, q + p).items():
                    transitions[target] = transitions.get(target, 0j) + value
                transitions = {target: value / 2 for target, value in transitions.items()}
            else:
                transitions = _apply(state, tuple(("q", mode) for mode in indices))
            for target, amplitude in transitions.items():
                output[target] = output.get(target, np.zeros(components, complex)) + coefficient * amplitude
    return output


def _selected_property_element(
    source: tuple[int, ...], target: tuple[int, ...],
    tensor: np.ndarray, *, rank: int, momentum: bool = False,
) -> np.ndarray:
    """One D1/D2 matrix element without expanding unrelated virtual states."""
    components = tensor.shape[0]
    if momentum:
        if sum(abs(a - b) for a, b in zip(source, target, strict=True)) > rank:
            return np.zeros(components, dtype=complex)
        value = np.zeros(components, dtype=complex)
        for indices in product(range(len(source)), repeat=rank):
            coefficient = tensor[(slice(None), *indices)] / factorial(rank - 1)
            if not np.any(coefficient):
                continue
            p = (("p", indices[0]),)
            q = tuple(("q", mode) for mode in indices[1:])
            amplitude = (_factorized_operator_element(source, target, p + q)
                         + _factorized_operator_element(source, target, q + p)) / 2
            value += coefficient * amplitude
        return value
    changed: list[tuple[int, int]] = []
    distance = 0
    for mode, (occupation, final) in enumerate(zip(source, target, strict=True)):
        delta = final - occupation
        if delta:
            distance += abs(delta)
            if distance > rank:
                return np.zeros(components, dtype=complex)
            changed.append((mode, delta))
    if rank == 1:
        if distance != 1:
            return np.zeros(components, dtype=complex)
        mode, delta = changed[0]
        return tensor[:, mode] * sqrt(max(source[mode], target[mode]) / 2)
    if rank == 2:
        if distance == 0:
            return np.einsum(
                "ci,i->c", tensor[:, np.arange(len(source)), np.arange(len(source))],
                np.asarray(source, dtype=float) + 0.5,
            ) / 2
        if distance != 2:
            return np.zeros(components, dtype=complex)
        if len(changed) == 1:
            mode, delta = changed[0]
            high = max(source[mode], target[mode])
            return tensor[:, mode, mode] * sqrt(high * (high - 1)) / 4
        (i, _), (j, _) = changed
        factor = sqrt(max(source[i], target[i]) * max(source[j], target[j])) / 4
        return (tensor[:, i, j] + tensor[:, j, i]) * factor
    changes = tuple(abs(a - b) for a, b in zip(source, target))
    if distance % 2 != rank % 2:
        return np.zeros(components, dtype=complex)
    required = tuple(i for i, count in enumerate(changes) for _ in range(count))
    if distance == rank:
        multisets = (tuple(sorted(required)),)
    else:
        multisets = tuple((mode, mode) for mode in range(len(source)))
    result = np.zeros(components, dtype=complex)
    for indices in multisets:
        for ordered in set(permutations(indices)):
            coefficient = tensor[(slice(None), *ordered)] / factorial(rank)
            if np.any(coefficient):
                result += coefficient * _factorized_operator_element(
                    source, target, tuple(("q", mode) for mode in ordered),
                )
    return result


def _direct_polyad_exclusions(
    result: SonicGVPT2Result,
) -> dict[tuple[int, ...], frozenset[tuple[int, ...]]]:
    """Exclude direct variational couplings, not their transitive closure."""
    states = result.states
    excluded: dict[tuple[int, ...], frozenset[tuple[int, ...]]] = {
        state: frozenset((state,)) for state in states
    }
    for block in result.blocks:
        indices = block["indices"]
        hamiltonian = np.asarray(block["hamiltonian_cm"])
        for left, right in block.get("direct_resonance_states", ()):
            left, right = tuple(left), tuple(right)
            if left not in excluded or right not in excluded:
                raise ValueError("explicit resonance mask refers to an absent state")
            excluded[left] = excluded[left] | frozenset((right,))
            excluded[right] = excluded[right] | frozenset((left,))
        for row, index in enumerate(indices):
            source = states[int(index)]
            excluded[source] = excluded[source] | frozenset(
                states[int(indices[column])]
                for column in range(len(indices))
                if row != column and abs(hamiltonian[row, column]) > 1e-12
            )
    return excluded


def _rectilinear_dvpt2_fundamental_mechanical(
    frequencies: np.ndarray,
    f3: np.ndarray,
    f4: np.ndarray,
    first: np.ndarray,
    mode: int,
    exclusions: dict[tuple[int, ...], frozenset[tuple[int, ...]]],
) -> np.ndarray:
    """Closed L717 DVPT2 potential correction multiplying ``D1``.

    The ordinary nonresonant contact transformation and the specialized
    rectilinear formula are algebraically identical.  In a deperturbed
    polyad, the latter removes each 1:2 and 1:1 pole term separately before
    collecting the normalization contribution.  Keeping this collected
    expression is essential: applying a generic projected wave operator to
    the already collected polynomial changes the finite remainder beside a
    removed pole.

    This is the potential part of the fundamental transition moment.  The
    Coriolis term remains represented by ``G''`` and the electric and mixed
    terms by the common reduced contractions.
    """
    omega = np.asarray(frequencies, dtype=float)
    cubic = np.asarray(f3, dtype=float)
    quartic = np.asarray(f4, dtype=float)
    d1 = np.asarray(first)
    n = len(omega)
    fundamental = tuple(int(index == mode) for index in range(n))
    direct = exclusions[fundamental]
    fermi = {
        tuple(index for index, occupation in enumerate(state)
              for _ in range(occupation))
        for state in direct if sum(state) == 2
    }
    one_one = {
        state.index(1) for state in direct
        if sum(state) == 1 and state != fundamental
    }

    def fermi_resonant(j: int, k: int) -> bool:
        return tuple(sorted((j, k))) in fermi

    wi = omega[mode]
    prefactor = 1.0 / (16.0 * sqrt(2.0))
    same_mode_cubic = 0.0
    same_mode_normalization = 0.0
    correction = np.zeros(d1.shape[0], dtype=np.result_type(d1, float))
    for j, wj in enumerate(omega):
        dij = j == mode
        resonant_11 = j in one_one or (
            not dij and abs(wi - wj) <= 1e-10
        )
        sum_ij = wi + wj
        difference_ij = wi - wj
        fiij = cubic[mode, mode, j]
        t2a1 = 1.0 / (wi * wj)
        t2a2 = 2.0 / (9.0 * wi * wj)
        t2c1 = 2.0 / (wi * sum_ij)
        if resonant_11 or dij:
            t2c2 = 1.0 / (3.0 * wi * (2.0 * wi + wj))
            t2c3 = 3.0 / (wi * wj)
        else:
            t2c1 -= 2.0 / (wi * difference_ij)
            t2c2 = t2c3 = 0.0
        t1a = t1c = t2b = t2c = 0.0
        for k, wk in enumerate(omega):
            dik = k == mode
            resonant_12 = fermi_resonant(j, k)
            sum_ijk = wi + wj + wk
            difference_ijk = wi - wj - wk
            fiik = cubic[mode, mode, k]
            fijk = cubic[mode, j, k]
            fjkk = cubic[j, k, k]
            fikk = cubic[mode, k, k]
            same_mode_normalization += (
                prefactor * fiij * fjkk
                * (t2a1 + (t2a2 if dij and dik else 0.0))
            )
            if dij:
                continue
            factor = (2.0 if dik else 1.0) * prefactor * fijk**2
            value = factor / (2.0 * wi * sum_ijk) + factor / (2.0 * sum_ijk**2)
            if not resonant_12:
                value -= factor / (2.0 * wi * difference_ijk)
                value -= factor / (2.0 * difference_ijk**2)
            same_mode_cubic += value
            value = (5.0 * t2c1 / 3.0 + t2c2 + t2c3
                     if dik else t2c1 + t2c3)
            t2c += prefactor * fiij * fikk * value
            if dik:
                continue
            value = 4.0 / (wk * sum_ij) + 2.0 / (sum_ij * (2.0 * wi + wk))
            if resonant_11:
                value += 2.0 / (sum_ijk * (2.0 * wi + wk))
                if not resonant_12:
                    value -= 1.0 / (wk * difference_ijk)
            else:
                value -= 4.0 / (wk * difference_ij)
                value -= 2.0 / (difference_ij * (2.0 * wi + wk))
            t1c += prefactor * fiik * fijk * value
            t2b1 = 2.0 / (wk * sum_ij)
            if resonant_11:
                if not resonant_12:
                    t2b1 -= 1.0 / (wk * difference_ijk)
            else:
                t2b1 -= 2.0 / (wk * difference_ij)
            for l, wl in enumerate(omega):
                if l == mode:
                    continue
                resonant_ikl = fermi_resonant(k, l)
                sum_ikl = wi + wk + wl
                difference_ikl = wi - wk - wl
                value = 1.0 / (sum_ij * sum_ikl)
                if not resonant_ikl:
                    value -= 1.0 / (sum_ij * difference_ikl)
                if resonant_11:
                    value += 1.0 / (sum_ikl * (wj + wk + wl))
                else:
                    value -= 1.0 / (difference_ij * sum_ikl)
                    if not resonant_ikl:
                        value += 1.0 / (difference_ij * difference_ikl)
                t1a += prefactor * cubic[mode, k, l] * cubic[j, k, l] * value
                t2b += prefactor * fijk * cubic[k, l, l] * t2b1
        quartic_term = 0.0
        for k in range(n):
            value = quartic[mode, j, k, k] / (8.0 * sqrt(2.0))
            quartic_term -= value / sum_ij
            if not (dij or resonant_11):
                quartic_term += value / difference_ij
        correction += d1[:, j] * (t1a + t1c + t2b + t2c + quartic_term)
    correction += d1[:, mode] * (same_mode_cubic + same_mode_normalization)
    return correction


def _explicit_coordinate_first_order_moment(
    frequencies: np.ndarray, f3: np.ndarray, g1: np.ndarray,
    first: np.ndarray, second: np.ndarray,
    initial: tuple[int, ...], final: tuple[int, ...],
    excluded: frozenset[tuple[int, ...]], denominator_tolerance_cm: float,
) -> np.ndarray | None:
    """Collected curvilinear two-quantum amplitudes (full PT is the fallback).

    The rho/sigma contractions include the first metric derivative.  The
    resonant rho denominator is omitted only for a directly coupled polyad
    edge; its contribution is restored by the variational eigenvectors.
    """
    n = len(frequencies)
    change = np.asarray(final) - np.asarray(initial)
    if np.any(change < 0) or sum(change) != 2:
        return None
    changed = np.flatnonzero(change)
    if len(changed) == 1 and change[changed[0]] == 2:
        i = j = int(changed[0])
    elif len(changed) == 2:
        i, j = (int(mode) for mode in changed)
    else:
        return None

    def sigma(a: int, b: int, c: int) -> float:
        return float(f3[a, b, c] - g1[a, b, c] - g1[a, c, b] - g1[b, c, a])

    def rho(a: int, b: int, c: int) -> float:
        return float(f3[a, b, c] - g1[a, b, c] + g1[a, c, b] + g1[b, c, a])

    if i != j:
        ladder = sqrt((initial[i] + 1) * (initial[j] + 1))
        moment = ladder * (second[:, i, j] + second[:, j, i]) / 4
        scale = ladder / 4
        base = frequencies[i] + frequencies[j]
    else:
        ladder = sqrt((initial[i] + 1) * (initial[i] + 2))
        moment = ladder * second[:, i, i] / 4
        scale = ladder / 8
        base = 2 * frequencies[i]

    for k in range(n):
        resonant = list(initial)
        resonant[k] += 1
        resonant_state = tuple(resonant)
        plus = 0.0
        if resonant_state not in excluded:
            gap = base - frequencies[k]
            if abs(gap) <= denominator_tolerance_cm:
                raise ValueError(
                    f"unresolved curvilinear transition denominator for {final}, {resonant_state}"
                )
            plus = rho(i, j, k) / gap
        minus = sigma(i, j, k) / (base + frequencies[k])
        moment += scale * first[:, k] * (plus - minus)
    return np.asarray(moment, dtype=complex)


def _explicit_momentum_first_order_moment(
    frequencies: np.ndarray, f3: np.ndarray, g1: np.ndarray,
    first: np.ndarray, second: np.ndarray,
    initial: tuple[int, ...], final: tuple[int, ...],
    excluded: frozenset[tuple[int, ...]], denominator_tolerance_cm: float,
) -> np.ndarray | None:
    """Collected first-order amplitudes for the ordered magnetic operator."""
    n = len(frequencies)
    change = np.asarray(final) - np.asarray(initial)
    if np.any(change < 0) or sum(change) != 2:
        return None
    changed = np.flatnonzero(change)
    if len(changed) == 1 and change[changed[0]] == 2:
        i = j = int(changed[0])
    elif len(changed) == 2:
        i, j = (int(mode) for mode in changed)
    else:
        return None

    def sigma(a: int, b: int, c: int) -> float:
        return float(f3[a, b, c] - g1[a, b, c] - g1[a, c, b] - g1[b, c, a])

    def rho(a: int, b: int, c: int) -> float:
        return float(f3[a, b, c] - g1[a, b, c] + g1[a, c, b] + g1[b, c, a])

    if i != j:
        ladder = sqrt((initial[i] + 1) * (initial[j] + 1))
        moment = 1j * ladder * (second[:, i, j] + second[:, j, i]) / 2
        scale = 1j * ladder / 4
        base = frequencies[i] + frequencies[j]
    else:
        ladder = sqrt((initial[i] + 1) * (initial[i] + 2))
        moment = 1j * ladder * second[:, i, i] / 2
        scale = 1j * ladder / 8
        base = 2 * frequencies[i]

    for k in range(n):
        resonant = list(initial)
        resonant[k] += 1
        plus = 0.0
        if tuple(resonant) not in excluded:
            gap = base - frequencies[k]
            if abs(gap) <= denominator_tolerance_cm:
                raise ValueError(
                    f"unresolved curvilinear transition denominator for {final}, {tuple(resonant)}"
                )
            plus = rho(i, j, k) / gap
        moment += scale * first[:, k] * (plus + sigma(i, j, k) / (base + frequencies[k]))
    return np.asarray(moment, dtype=complex)


def _explicit_coordinate_expectation_first_order(
    frequencies: np.ndarray, f3: np.ndarray, g1: np.ndarray,
    equilibrium: np.ndarray, first: np.ndarray, second: np.ndarray,
    occupation: tuple[int, ...],
) -> np.ndarray:
    """Collected first-order expectation for a coordinate-valued property.

    The term linear in the property surface mixes with the cubic potential
    and the first curvilinear metric derivative. No virtual-state sum is used.
    """
    occupied = np.asarray(occupation, dtype=float) + 0.5
    direct = 0.5 * np.einsum("ci,i->c", second[:, np.arange(len(occupied)),
                                                  np.arange(len(occupied))], occupied)
    cubic_metric = np.einsum("ijj->ij", f3) + np.einsum("jji->ij", g1)
    mechanical = 0.5 * np.einsum(
        "ci,ij,j->c", first / frequencies[None, :], cubic_metric, occupied,
    )
    return np.asarray(equilibrium + direct - mechanical, dtype=float)


def _reduced_diagonal_property_expectation_1d(
    frequencies: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    g1: np.ndarray, g2: np.ndarray, first: np.ndarray, second: np.ndarray,
    third: np.ndarray, fourth: np.ndarray, occupation: tuple[int, ...],
    *, fifth: np.ndarray | None = None, g3: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed one-mode diagonal-property reduction through the next even order.

    This is the production kernel for a separated torsion.  It contains only
    numerical coefficients multiplying the supplied force, metric, and
    property derivatives; it constructs no oscillator matrices,
    intermediate tensors, or virtual-state lists.
    """
    omega = np.asarray(frequencies, dtype=float)
    if omega.shape != (1,) or len(occupation) != 1 or occupation[0] < 0:
        raise ValueError("the fully reduced next-order kernel requires one separated mode")
    n = float(occupation[0])
    w = float(omega[0])
    f_3 = float(np.asarray(f3)[0, 0, 0])
    f_4 = float(np.asarray(f4)[0, 0, 0, 0])
    g_1 = float(np.asarray(g1)[0, 0, 0])
    g_2 = float(np.asarray(g2)[0, 0, 0, 0])
    f_5 = 0.0 if fifth is None else float(np.asarray(fifth)[0, 0, 0, 0, 0])
    g_3 = 0.0 if g3 is None else float(np.asarray(g3)[0, 0, 0, 0, 0])
    p_1 = np.asarray(first, dtype=float)[:, 0]
    p_2 = np.asarray(second, dtype=float)[:, 0, 0]
    p_3 = np.asarray(third, dtype=float)[:, 0, 0, 0]
    p_4 = np.asarray(fourth, dtype=float)[:, 0, 0, 0, 0]
    a = 2 * n * n + 2 * n + 1
    b = 6 * n * n + 6 * n + 1
    c = 30 * n * n + 30 * n + 11
    leading = (2 * n + 1) * (p_2 - p_1 * (f_3 + g_1) / w) / 4
    following = (
        p_4 * a / 32
        - p_3 * g_1 * b / (48 * w)
        - p_3 * f_3 * c / (144 * w)
        - p_2 * f_4 * a / (32 * w)
        + p_2 * f_3 * g_1 * b / (48 * w**2)
        + p_2 * f_3**2 * c / (144 * w**2)
        - p_1 * g_3 * a / (16 * w)
        + p_1 * g_1 * g_2 * (3 * a - 1) / (16 * w**2)
        - p_1 * g_1**3 * (3 * a - 1) / (32 * w**3)
        - p_1 * f_5 * a / (32 * w)
        + p_1 * f_4 * g_1 * (6 * n * n + 6 * n - 1) / (96 * w**2)
        + p_1 * f_3 * g_2 * b / (48 * w**2)
        + p_1 * f_3 * f_4 * (78 * n * n + 78 * n + 31) / (288 * w**2)
        - p_1 * f_3**2 * g_1 * (6 * n * n + 6 * n - 5) / (288 * w**3)
        - p_1 * f_3**3 * c / (144 * w**3)
    )
    return np.asarray(leading, dtype=float), np.asarray(following, dtype=float)


def _next_nonzero_coordinate_expectation_qff(
    frequencies: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    g1: np.ndarray, g2: np.ndarray, first: np.ndarray, second: np.ndarray,
    third: np.ndarray, fourth: np.ndarray, occupation: tuple[int, ...],
    *, fifth: np.ndarray | None = None, g3: np.ndarray | None = None,
    denominator_tolerance_cm: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the final algebraic reduction used by production calculations."""
    del denominator_tolerance_cm
    return _reduced_diagonal_property_expectation_1d(
        frequencies, f3, f4, g1, g2, first, second, third, fourth,
        occupation, fifth=fifth, g3=g3,
    )


def _diagonal_morse_quintic(
    frequencies: np.ndarray, f3: np.ndarray, f4: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate diagonal fifth derivatives by the local Morse closure.

    The diagnostic is the relative RMS discrepancy between the supplied
    diagonal quartics and 7*a_i**2*omega_i.
    """
    omega = np.asarray(frequencies, dtype=float)
    n = len(omega)
    if np.any(omega <= 0):
        raise ValueError("Morse quintic estimate requires positive modal frequencies")
    cubic = np.asarray([f3[(i, i, i)] for i in range(n)], dtype=float)
    quartic = np.asarray([f4[(i, i, i, i)] for i in range(n)], dtype=float)
    alpha = -cubic / (3 * omega)
    estimated_quartic = 7 * alpha**2 * omega
    denominator = float(np.linalg.norm(quartic))
    mismatch = float(np.linalg.norm(estimated_quartic - quartic)
                     / (denominator if denominator else 1.0))
    output = np.zeros((n,) * 5)
    for mode in range(n):
        fifth = -15 * alpha[mode] ** 3 * omega[mode]
        output[(mode,) * 5] = fifth
    return output, mismatch


def sonic_diagonal_property_expectation(
    field: SonicVPT2Input, derivatives: SonicPropertyDerivatives,
    equilibrium: np.ndarray, occupation: tuple[int, ...], *,
    rotational_second_moment: np.ndarray | None = None,
    rotational_surface: tuple[np.ndarray, SonicPropertyDerivatives] | None = None,
    fifth_potential: np.ndarray | None = None,
    third_metric: np.ndarray | None = None,
    estimate_diagonal_morse_quintic: bool = False,
    denominator_tolerance_cm: float = 1e-6,
) -> SonicDiagonalPropertyExpectation:
    """Vibrational and optional rotation--vibration average of a property.

    ``rotational_surface`` is ``(R_eq, R_derivatives)`` for the coefficient
    multiplying the symmetrized body-fixed product ``{J_a,J_b}/2``.  Its
    component axis is flattened as ``property_component, a, b`` and is
    contracted with ``rotational_second_moment`` after vibrational averaging.
    """
    frequencies, _modes, f3, f4, g1, g2 = _modalize_sonic_field(field)
    eq = np.asarray(equilibrium, dtype=float).reshape(-1)
    first = np.asarray(derivatives.first, dtype=float)
    second = np.asarray(derivatives.second, dtype=float)
    if first.shape != (len(eq), len(frequencies)) or second.shape != (len(eq), len(frequencies), len(frequencies)):
        raise ValueError("property equilibrium value and D1/D2 disagree with the force field")
    standard = _explicit_coordinate_expectation_first_order(
        frequencies, f3, g1, eq, first, second, occupation,
    )
    correction = complete = complete_supplied = complete_morse = None
    morse_mismatch = None
    if derivatives.fourth is not None:
        if derivatives.third is None:
            raise ValueError("the next nonzero expectation requires property D3 and D4")
        leading, correction = _next_nonzero_coordinate_expectation_qff(
            frequencies, f3, f4, g1, g2, first, second,
            np.asarray(derivatives.third, dtype=float),
            np.asarray(derivatives.fourth, dtype=float), occupation,
            denominator_tolerance_cm=denominator_tolerance_cm,
        )
        if not np.allclose(eq + leading, standard, atol=1e-8, rtol=1e-10):
            raise ValueError("reduced and recursive leading expectations disagree")
        complete = standard + correction
        if fifth_potential is not None or third_metric is not None:
            _leading, supplied_correction = _next_nonzero_coordinate_expectation_qff(
                frequencies, f3, f4, g1, g2, first, second,
                np.asarray(derivatives.third, dtype=float),
                np.asarray(derivatives.fourth, dtype=float), occupation,
                fifth=fifth_potential, g3=third_metric,
                denominator_tolerance_cm=denominator_tolerance_cm,
            )
            complete_supplied = standard + supplied_correction
        if estimate_diagonal_morse_quintic:
            if fifth_potential is not None or third_metric is not None:
                raise ValueError("choose supplied H5 or the diagonal Morse estimate, not both")
            morse_h5, morse_mismatch = _diagonal_morse_quintic(
                frequencies, f3, f4,
            )
            _leading, morse_correction = _next_nonzero_coordinate_expectation_qff(
                frequencies, f3, f4, g1, g2, first, second,
                np.asarray(derivatives.third, dtype=float),
                np.asarray(derivatives.fourth, dtype=float), occupation,
                fifth=morse_h5,
                denominator_tolerance_cm=denominator_tolerance_cm,
            )
            complete_morse = standard + morse_correction
    rovib = rovib_qff = rovib_supplied = rovib_morse = None
    if rotational_surface is not None or rotational_second_moment is not None:
        if rotational_surface is None or rotational_second_moment is None:
            raise ValueError("rotational surface and angular-momentum moment are both required")
        rotational_eq, rotational_derivatives = rotational_surface
        angular = np.asarray(rotational_second_moment, dtype=float)
        rotational_eq = np.asarray(rotational_eq, dtype=float)
        if angular.ndim != 2 or angular.shape[0] != angular.shape[1] or rotational_eq.shape[1:] != angular.shape:
            raise ValueError("rotational property surface and angular-momentum moment disagree")
        flat_eq = rotational_eq.reshape(-1)
        nmode = len(frequencies)

        def flattened(value: np.ndarray | None, rank: int) -> np.ndarray | None:
            if value is None:
                return None
            return np.asarray(value, dtype=float).reshape((len(flat_eq),) + (nmode,) * rank)

        flat = SonicPropertyDerivatives(
            flattened(rotational_derivatives.first, 1),
            flattened(rotational_derivatives.second, 2),
            flattened(rotational_derivatives.third, 3),
            flattened(rotational_derivatives.fourth, 4),
        )
        averaged_standard = _explicit_coordinate_expectation_first_order(
            frequencies, f3, g1, flat_eq, flat.first, flat.second, occupation,
        )
        rovib = np.einsum(
            "cab,ab->c", averaged_standard.reshape(rotational_eq.shape), angular,
        )
        if flat.fourth is not None:
            if flat.third is None:
                raise ValueError("the next vibro-rotational order requires rotational D3 and D4")
            rotational_leading, rotational_following = _next_nonzero_coordinate_expectation_qff(
                frequencies, f3, f4, g1, g2, flat.first, flat.second,
                flat.third, flat.fourth, occupation,
                denominator_tolerance_cm=denominator_tolerance_cm,
            )
            if not np.allclose(
                flat_eq + rotational_leading, averaged_standard, atol=1e-8, rtol=1e-10,
            ):
                raise ValueError("reduced vibro-rotational leading averages disagree")
            averaged_qff = averaged_standard + rotational_following
            rovib_qff = np.einsum(
                "cab,ab->c", averaged_qff.reshape(rotational_eq.shape), angular,
            )
            if fifth_potential is not None or third_metric is not None:
                _same, rotational_supplied = _next_nonzero_coordinate_expectation_qff(
                    frequencies, f3, f4, g1, g2, flat.first, flat.second,
                    flat.third, flat.fourth, occupation,
                    fifth=fifth_potential, g3=third_metric,
                    denominator_tolerance_cm=denominator_tolerance_cm,
                )
                rovib_supplied = np.einsum(
                    "cab,ab->c",
                    (averaged_standard + rotational_supplied).reshape(rotational_eq.shape),
                    angular,
                )
            if estimate_diagonal_morse_quintic:
                morse_h5, _unused = _diagonal_morse_quintic(frequencies, f3, f4)
                _same, rotational_morse = _next_nonzero_coordinate_expectation_qff(
                    frequencies, f3, f4, g1, g2, flat.first, flat.second,
                    flat.third, flat.fourth, occupation, fifth=morse_h5,
                    denominator_tolerance_cm=denominator_tolerance_cm,
                )
                rovib_morse = np.einsum(
                    "cab,ab->c",
                    (averaged_standard + rotational_morse).reshape(rotational_eq.shape),
                    angular,
                )
        standard = standard + rovib
        if complete is not None:
            complete = complete + (rovib if rovib_qff is None else rovib_qff)
        if complete_supplied is not None:
            complete_supplied = complete_supplied + (
                rovib_supplied if rovib_supplied is not None
                else (rovib_qff if rovib_qff is not None else rovib)
            )
        if complete_morse is not None:
            complete_morse = complete_morse + (
                rovib_morse if rovib_morse is not None
                else (rovib_qff if rovib_qff is not None else rovib)
            )
    return SonicDiagonalPropertyExpectation(
        equilibrium=eq,
        standard=standard,
        complete_qff=complete,
        next_nonzero_correction=correction,
        rovibrational_correction=rovib,
        rovibrational_qff_correction=rovib_qff,
        rovibrational_supplied_h5_correction=rovib_supplied,
        rovibrational_morse_h5_correction=rovib_morse,
        complete_with_supplied_h5=complete_supplied,
        complete_qff_morse_h5=complete_morse,
        morse_quartic_relative_rms=morse_mismatch,
    )


def _explicit_one_mode_coordinate_second_order_moment(
    frequencies: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    g1: np.ndarray, g2: np.ndarray,
    first: np.ndarray, second: np.ndarray, third: np.ndarray,
    initial: tuple[int, ...], final: tuple[int, ...],
) -> np.ndarray | None:
    """Collected rational second-order limits for two one-mode bands."""
    if len(frequencies) != 1:
        return None
    omega = float(frequencies[0])
    cubic = float(f3[0, 0, 0])
    quartic = float(f4[0, 0, 0, 0])
    metric1 = float(g1[0, 0, 0])
    metric2 = float(g2[0, 0, 0, 0])
    p1, p2, p3 = first[:, 0], second[:, 0, 0], third[:, 0, 0, 0]
    if initial == (0,) and final == (1,):
        numerator = (
            p1 * (11 * cubic**2 - 24 * cubic * metric1 - 27 * metric1**2
                  - 9 * quartic * omega)
            - p2 * (60 * cubic + 36 * metric1) * omega
            + 36 * p3 * omega**2
        )
        return np.asarray(sqrt(2) * numerator / (288 * omega**2), dtype=complex)
    if initial == (0,) and final == (3,):
        return np.asarray(sqrt(3) * (
            p3 / 12 + p2 * (cubic + 3 * metric1) / (12 * omega)
            + p1 * (cubic**2 + 10 * cubic * metric1 + 9 * metric1**2
                    + omega * (quartic + 6 * metric2)) / (96 * omega**2)
        ), dtype=complex)
    if initial == (1,) and final == (2,):
        return np.asarray((
            p1 * (11 * cubic**2 - 24 * cubic * metric1 - 27 * metric1**2
                  - 9 * quartic * omega)
            - p2 * (60 * cubic + 36 * metric1) * omega
            + 36 * p3 * omega**2
        ) / (72 * omega**2), dtype=complex)
    return None


def _explicit_one_mode_momentum_second_order_moment(
    frequencies: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    g1: np.ndarray, g2: np.ndarray,
    first: np.ndarray, second: np.ndarray, third: np.ndarray,
    initial: tuple[int, ...], final: tuple[int, ...],
) -> np.ndarray | None:
    """Collected one-mode VCD coefficients with the magnetic p ordering."""
    if len(frequencies) != 1:
        return None
    omega = float(frequencies[0])
    cubic = float(f3[0, 0, 0])
    quartic = float(f4[0, 0, 0, 0])
    metric1 = float(g1[0, 0, 0])
    metric2 = float(g2[0, 0, 0, 0])
    b1, b2, b3 = first[:, 0], second[:, 0, 0], third[:, 0, 0, 0]
    if initial == (0,) and final == (1,):
        numerator = (
            b1 * (-19 * cubic**2 + 9 * quartic * omega + 27 * metric1**2)
            - b2 * (60 * cubic + 108 * metric1) * omega
            + 36 * b3 * omega**2
        )
        return np.asarray(1j * sqrt(2) * numerator / (288 * omega**2), dtype=complex)
    if initial == (0,) and final == (3,):
        numerator = (
            b1 * ((cubic + metric1)**2 + omega * (quartic - 2 * metric2))
            + 8 * b2 * (cubic + metric1) * omega
            + 8 * b3 * omega**2
        )
        return np.asarray(1j * sqrt(3) * numerator / (32 * omega**2), dtype=complex)
    if initial == (1,) and final == (2,):
        numerator = (
            b1 * (-19 * cubic**2 + 9 * quartic * omega + 27 * metric1**2)
            - b2 * (60 * cubic + 108 * metric1) * omega
            + 36 * b3 * omega**2
        )
        return np.asarray(1j * numerator / (72 * omega**2), dtype=complex)
    return None


def _curvilinear_basis_moments_by_order(
    result: SonicGVPT2Result, derivatives: SonicPropertyDerivatives | SonicMomentumPropertyDerivatives,
    components: int, field: SonicVPT2Input, *, denominator_tolerance_cm: float = 1e-6,
    initial_state: tuple[int, ...] | None = None,
    momentum: bool = False,
    explicit_formulas: bool = True,
    _reduced_cache: dict | None = None,
    final_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed Fock-index contractions from one initial state through VPT2 order.

    The force field uses the equilibrium-only geometric pseudopotential
    supplied by ``SonicVPT2Input``. Directly coupled polyad pairs are
    excluded from external resolvents; their dressed basis is
    orthonormalized before mixing by the GVPT2 eigenvectors.
    """
    # Modalization includes dense rank-three/rank-four basis transforms and is
    # independent of both endpoints of a transition.  Hot-band runs call this
    # routine for many initial states, so retain the transformed field in the
    # same calculation-local cache that owns the reduced contractions.
    cache = {} if _reduced_cache is None else _reduced_cache
    modalized_entry = cache.get("modalized_field")
    if modalized_entry is None or modalized_entry[0] is not field:
        modalized_entry = (field, _modalize_sonic_field(field))
        cache["modalized_field"] = modalized_entry
    frequencies, _modes, f3, f4, g1, g2 = modalized_entry[1]
    states = result.states
    selected_indices = (tuple(range(len(states))) if final_indices is None
                        else tuple(dict.fromkeys(int(index) for index in final_indices)))
    if any(index < 0 or index >= len(states) for index in selected_indices):
        raise ValueError("final transition-state index is outside the GVPT2 basis")
    if not states or len(frequencies) != len(states[0]) or len(result.frequencies_cm) != len(frequencies):
        raise ValueError("property and curvilinear force field mode counts disagree")
    if not np.allclose(result.frequencies_cm, frequencies, rtol=1e-9, atol=1e-6):
        raise ValueError("GVPT2 result and curvilinear force field disagree")
    n = len(frequencies)
    arrays = (derivatives.first, derivatives.second, derivatives.third)
    validated = cache.get("validated_property")
    if validated is None or validated[0] is not derivatives or validated[1] != components:
        for rank, values in enumerate(arrays, 1):
            expected = (components, *([n] * rank))
            if values is None or np.asarray(values).shape != expected or not np.all(np.isfinite(values)):
                raise ValueError("second-order transition moments require complete finite property D1/D2/D3")
        cache["validated_property"] = derivatives, components
    if not np.isfinite(denominator_tolerance_cm) or denominator_tolerance_cm <= 0:
        raise ValueError("denominator tolerance must be positive and finite")

    # Component membership determines only the diagonalization block.
    # Nonadjacent states remain in the external PT resolvents.
    polyad_entry = cache.get("direct_polyad_exclusions")
    if polyad_entry is None or polyad_entry[0] is not result:
        polyad_entry = result, _direct_polyad_exclusions(result)
        cache["direct_polyad_exclusions"] = polyad_entry
        cache["all_polyads_isolated"] = all(len(members) == 1 for members in polyad_entry[1].values())
    polyad = polyad_entry[1]
    initial = tuple(0 for _ in frequencies) if initial_state is None else tuple(initial_state)
    if initial not in polyad:
        raise ValueError("initial GVPT2 state is absent from the state basis")

    # Production uses coefficient reductions only. The state contraction
    # below is accessible explicitly as a verification reference.
    if explicit_formulas:
        from .reduced_transition_polynomials import (
            ResonantTransitionPolynomialError, evaluate_reduced_transition,
            prepare_reduced_transition_field, reduced_property_coefficients,
        )
        from .reduced_resonant_transitions import PairReducedField, PairReducedProperty

        if cache.get("validated_modal_symmetry") is not modalized_entry[1]:
            symmetric = all(np.allclose(tensor, tensor.swapaxes(a, b), atol=1e-10, rtol=1e-12)
                            for tensor, a, b in (
                                (f3, 0, 1), (f3, 1, 2),
                                (f4, 0, 1), (f4, 1, 2), (f4, 2, 3),
                                (g1, 0, 1), (g2, 0, 1), (g2, 2, 3)))
            if not symmetric:
                raise ValueError("reduced property equations require symmetric force and metric tensors")
            cache["validated_modal_symmetry"] = modalized_entry[1]
        orders = None
        if cache["all_polyads_isolated"]:
            if "field" not in cache:
                try:
                    cache["field"] = prepare_reduced_transition_field(
                        frequencies, f3, f4, g1, g2,
                        denominator_tolerance_cm=denominator_tolerance_cm,
                    )
                except ResonantTransitionPolynomialError:
                    cache["field"] = None
            if cache["field"] is not None:
                coefficients = reduced_property_coefficients(
                    cache["field"], *(np.asarray(values) for values in arrays),
                    momentum=momentum,
                )
                orders = tuple(np.asarray([
                    evaluate_reduced_transition(values, initial, final, components)
                    for final in states
                ]) for values in coefficients)
        if orders is None:
            if "pair_field" not in cache:
                cache["pair_field"] = PairReducedField(
                    frequencies, f3, f4, g1, g2, polyad,
                    tolerance=denominator_tolerance_cm,
                )
            property_key = (momentum, *(id(values) for values in arrays))
            properties = cache.setdefault("pair_properties", {})
            if property_key not in properties:
                properties[property_key] = PairReducedProperty(
                    cache["pair_field"],
                    *(np.asarray(values) for values in arrays),
                    momentum=momentum,
                )
            reduced = properties[property_key]
            selected = np.asarray([reduced.moment(initial, states[index])
                                   for index in selected_indices]).transpose(1, 0, 2)
            orders = tuple(np.zeros((len(states), components), dtype=complex) for _ in range(3))
            for rank in range(3):
                orders[rank][list(selected_indices)] = selected[rank]
        # In the rectilinear DVPT2 limit, collect the potential part of the
        # D1 mechanical correction only after removing individual Fermi and
        # 1:1 poles.  The generic projected contact transformation is
        # identical away from a resonance but leaves a different finite
        # remainder when applied after the polynomial has been collected.
        # L717 and the closed fundamental equations use the pole-first form.
        if (not momentum and initial == tuple(0 for _ in frequencies)
                and np.max(abs(g1)) <= 1e-14):
            resonant_fundamentals = tuple(
                index for index in selected_indices
                if sum(states[index]) == 1 and len(polyad[states[index]]) > 1
            )
            if resonant_fundamentals:
                key = ("rectilinear_d1_potential", id(arrays[0]))
                entry = cache.get(key)
                if entry is None:
                    potential_field = PairReducedField(
                        frequencies, f3, f4, g1, np.zeros_like(g2), polyad,
                        tolerance=denominator_tolerance_cm,
                    )
                    entry = PairReducedProperty(
                        potential_field, np.asarray(arrays[0]),
                        np.zeros_like(arrays[1]), np.zeros_like(arrays[2]),
                        momentum=False,
                    )
                    cache[key] = entry
                corrected_orders = list(orders)
                corrected_orders[2] = corrected_orders[2].copy()
                for index in resonant_fundamentals:
                    final = states[index]
                    mode = final.index(1)
                    generic = entry.moment(initial, final)[2]
                    collected = _rectilinear_dvpt2_fundamental_mechanical(
                        frequencies, f3, f4, np.asarray(arrays[0]), mode, polyad,
                    )
                    corrected_orders[2][index] += collected - generic
                orders = tuple(corrected_orders)
        if momentum:
            if any(np.max(abs(values.real)) > 1e-8 for values in orders):
                raise ValueError("time-odd momentum moments have an unexpected real component")
            return orders
        if any(np.max(abs(values.imag)) > 1e-8 for values in orders):
            raise ValueError("curvilinear transition moments have an unexpected imaginary component")
        return tuple(values.real for values in orders)

    cubic: dict[tuple[int, ...], dict[tuple[int, ...], complex]] = {}
    cubic_terms = _canonical_cubic_terms(f3, g1, 0.0)
    selected_hamiltonian = _SelectedOperatorEvaluator(f3, f4, g1, g2)
    property_terms: dict[tuple[tuple[int, ...], int], dict[tuple[int, ...], np.ndarray]] = {}
    zero = np.zeros(components, dtype=complex)

    def energy(state: tuple[int, ...]) -> float:
        return float(np.dot(frequencies, state))

    def w(state: tuple[int, ...]) -> dict[tuple[int, ...], complex]:
        if state not in cubic:
            cubic[state] = _cubic_couplings(state, f3, g1, 0.0, terms=cubic_terms)
        return cubic[state]

    def d(state: tuple[int, ...], rank: int) -> dict[tuple[int, ...], np.ndarray]:
        key = (state, rank)
        if key not in property_terms:
            property_terms[key] = _property_couplings(
                state, derivatives, components, ranks=(rank,), momentum=momentum,
            )
        return property_terms[key]

    def divide(state: tuple[int, ...], source: tuple[int, ...], value: complex) -> complex:
        gap = energy(source) - energy(state)
        if abs(gap) <= denominator_tolerance_cm:
            raise ValueError(f"unresolved curvilinear transition denominator for {source}, {state}")
        return value / gap

    first: dict[tuple[int, ...], dict[tuple[int, ...], complex]] = {}
    second: dict[tuple[int, ...], dict[tuple[int, ...], complex]] = {}
    d1_initial = d(initial, 1)
    initial_second_targets: set[tuple[int, ...]] = set()
    for index in selected_indices:
        final = states[index]
        for mode in range(n):
            for change in (-1, 1):
                if final[mode] + change < 0:
                    continue
                target = list(final)
                target[mode] += change
                initial_second_targets.add(tuple(target))
    for source in states:
        excluded = polyad[source]
        first[source] = {
            state: divide(state, source, value)
            for state, value in w(source).items()
            if state not in excluded and abs(value) > 1e-12
        }
    def second_for(source: tuple[int, ...]) -> dict[tuple[int, ...], complex]:
        if source in second:
            return second[source]
        excluded = polyad[source]
        # D1 connects states differing by one quantum. The second external
        # wave operator is needed only at those targets for each final state,
        # and at D1-neighbours of requested final states for the initial state.
        # All virtual intermediate states remain in the
        # W Q W contraction; no occupation or energy cutoff is introduced.
        targets = ((initial_second_targets if source == initial else set(d1_initial))
                   - set(excluded))
        target_list = tuple(targets)
        if not target_list:
            second[source] = {}
            return second[source]
        target_array = np.asarray(target_list, dtype=np.int16)
        numerator = {
            target: selected_hamiltonian.element(source, target, order=4)
            for target in targets
        }
        for intermediate, amplitude in first[source].items():
            distances = np.sum(
                np.abs(target_array - np.asarray(intermediate, dtype=np.int16)),
                axis=1,
            )
            for position in np.flatnonzero((distances == 1) | (distances == 3)):
                target = target_list[int(position)]
                value = selected_hamiltonian.element(intermediate, target, order=3)
                if value:
                    numerator[target] += value * amplitude
        # The second-order external wave operator contains -A W_PP.
        # Internal Fermi couplings are diagonalized in the selected polyad.
        for member in excluded:
            internal = w(source).get(member, 0j)
            if abs(internal) > 1e-12:
                for target in targets:
                    numerator[target] -= first[member].get(target, 0j) * internal
        second[source] = {
            state: divide(state, source, value)
            for state, value in numerator.items()
            if state not in excluded and abs(value) > 1e-12
        }
        return second[source]

    def overlap(left: tuple[int, ...], right: tuple[int, ...]) -> complex:
        return sum((np.conjugate(value) * first[right].get(state, 0j)
                    for state, value in first[left].items()), 0j)

    result_orders = tuple(np.zeros((len(states), components), dtype=complex) for _ in range(3))
    d2_initial, d3_initial = (d(initial, rank) for rank in (2, 3))
    first_tensor = np.asarray(derivatives.first)
    second_tensor = np.asarray(derivatives.second)
    third_tensor = np.asarray(derivatives.third)
    symmetric_modal_field = (
        np.allclose(f3, np.transpose(f3, (1, 0, 2)), atol=1e-12, rtol=1e-12)
        and np.allclose(f3, np.transpose(f3, (0, 2, 1)), atol=1e-12, rtol=1e-12)
        and np.allclose(g1, np.transpose(g1, (1, 0, 2)), atol=1e-12, rtol=1e-12)
    ) if explicit_formulas else False
    for index in selected_indices:
        final = states[index]
        m0 = d1_initial.get(final, zero).copy()
        even_parity = (sum(final) - sum(initial)) % 2 == 0
        explicit_first = (_explicit_momentum_first_order_moment if momentum
                          else _explicit_coordinate_first_order_moment)
        m1 = (explicit_first(
            frequencies, f3, g1, first_tensor, second_tensor,
            initial, final, polyad[final], denominator_tolerance_cm,
        ) if symmetric_modal_field and even_parity else None)
        if not even_parity:
            m1 = zero.copy()
        elif m1 is None:
            m1 = d2_initial.get(final, zero).copy()
            m1 += sum((np.conjugate(value) * d1_initial.get(state, zero)
                       for state, value in first[final].items()), zero.copy())
            m1 += sum((value * _selected_property_element(state, final, first_tensor, rank=1,
                                                          momentum=momentum)
                       for state, value in first[initial].items()), zero.copy())

        if even_parity:
            # A rank-one property and its second-order correction are odd
            # under simultaneous inversion of all normal coordinates.
            # The rank-two property and its first-order correction are even.
            result_orders[0][index] = m0
            result_orders[1][index] = m1
            continue

        explicit_second = (_explicit_one_mode_momentum_second_order_moment if momentum
                           else _explicit_one_mode_coordinate_second_order_moment)
        one_mode_m2 = (explicit_second(
            frequencies, f3, f4, g1, g2, first_tensor, second_tensor,
            third_tensor, initial, final,
        ) if (symmetric_modal_field
              and polyad[final] == frozenset((final,))
              and polyad[initial] == frozenset((initial,))) else None)
        if one_mode_m2 is not None:
            result_orders[0][index] = m0
            result_orders[1][index] = m1
            result_orders[2][index] = one_mode_m2
            continue

        m2 = d3_initial.get(final, zero).copy()
        m2 += sum((np.conjugate(value) * d2_initial.get(state, zero)
                   for state, value in first[final].items()), zero.copy())
        m2 += sum((value * _selected_property_element(state, final, second_tensor, rank=2,
                                                      momentum=momentum)
                   for state, value in first[initial].items()), zero.copy())
        m2 += sum((np.conjugate(value) * d1_initial.get(state, zero)
                   for state, value in second_for(final).items()), zero.copy())
        m2 += sum((value * _selected_property_element(state, final, first_tensor, rank=1,
                                                      momentum=momentum)
                   for state, value in second_for(initial).items()), zero.copy())
        for excited_state, excited_amplitude in first[final].items():
            cross = zero.copy()
            for mode in range(n):
                for change in (-1, 1):
                    if excited_state[mode] + change < 0:
                        continue
                    neighbour = list(excited_state)
                    neighbour[mode] += change
                    initial_neighbour = tuple(neighbour)
                    initial_amplitude = first[initial].get(initial_neighbour, 0j)
                    if initial_amplitude:
                        cross += initial_amplitude * _selected_property_element(
                            initial_neighbour, excited_state, first_tensor, rank=1,
                            momentum=momentum,
                        )
            m2 += np.conjugate(excited_amplitude) * cross
        m2 -= 0.5 * sum((overlap(final, member) * d1_initial.get(member, zero)
                         for member in polyad[final]), zero.copy())
        m2 -= 0.5 * sum((_selected_property_element(member, final, first_tensor, rank=1,
                                                    momentum=momentum)
                         * overlap(member, initial)
                         for member in polyad[initial]), zero.copy())
        result_orders[0][index] = m0
        result_orders[1][index] = m1
        result_orders[2][index] = m2
    if momentum:
        for values in result_orders:
            if np.max(abs(values.real)) > 1e-8:
                raise ValueError("time-odd momentum moments have an unexpected real component")
        return result_orders
    for values in result_orders:
        if np.max(abs(values.imag)) > 1e-8:
            raise ValueError("curvilinear transition moments have an unexpected imaginary component")
    return tuple(values.real for values in result_orders)


def _mechanical_basis_moments(
    states: tuple[tuple[int, ...], ...], derivatives: SonicPropertyDerivatives,
    components: int, field: SonicVPT2Input,
    resonances: tuple[dict[str, object], ...],
) -> np.ndarray:
    # Preserve the explicit legacy first-order diagnostic, including its
    # supplied property surface, but evaluate it by reduced coefficients too.
    from .reduced_resonant_transitions import PairReducedField, PairReducedProperty
    from .reduced_transition_polynomials import evaluate_reduced_transition
    frequencies, _modes, f3, f4, g1, g2 = _modalize_sonic_field(field)
    n = len(frequencies)
    if n != len(states[0]):
        raise ValueError("property and curvilinear force field mode counts disagree")
    exclusions = {state: {state} for state in states}
    for item in resonances:
        if item["decision"] == "accepted" and item["kind"] == "1-2":
            a, b = states[int(item["i"])], states[int(item["j"])]
            exclusions[a].add(b)
            exclusions[b].add(a)
    arrays = tuple(np.zeros((components,) + (n,) * rank) if value is None
                   else np.asarray(value)
                   for rank, value in enumerate((derivatives.first, derivatives.second,
                                                  derivatives.third), 1))
    reduced = PairReducedProperty(PairReducedField(frequencies, f3, f4, g1, g2, exclusions),
                                  *arrays)
    ground = (0,) * n
    output = np.zeros((len(states), components), complex)
    for i, state in enumerate(states):
        for rank in range(3):
            name = f"p{rank}"
            polynomial = reduced.terms(name, ground, state).copy()
            for left, right in (("ad", name), (name, "a")):
                for key, value in reduced.multiply(left, right, ground, state).items():
                    polynomial[key] = polynomial.get(key, 0) + value
            output[i] += evaluate_reduced_transition(polynomial, ground, state, components)
    if np.max(abs(output.imag)) > 1e-8:
        raise ValueError("SONIC transition property has an unexpected imaginary component")
    return output.real


def _raman_strength(tensors: np.ndarray) -> np.ndarray:
    xx, xy, yy, xz, yz, zz = tensors.T
    isotropic = (xx + yy + zz) / 3
    anisotropic = 0.5 * (
        (xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2
        + 6 * (xy**2 + xz**2 + yz**2)
    )
    return (45 * isotropic**2 + 7 * anisotropic) * BOHR_TO_ANGSTROM**6


def sonic_gvpt2_spectrum(
    result: SonicGVPT2Result, *,
    dipole: SonicPropertyDerivatives | None = None,
    polarizability: SonicPropertyDerivatives | None = None,
    magnetic_dipole: SonicMomentumPropertyDerivatives | None = None,
    roa_response: SonicROAPropertySurfaces | None = None,
    field: SonicVPT2Input | None = None,
    transition_order: Literal["first", "second"] = "second",
    initial_state: tuple[int, ...] | None = None,
    transition_states: tuple[tuple[int, ...], ...] | None = None,
    _reduced_cache: dict | None = None,
) -> SonicGVPT2Spectrum:
    """Mix property transition moments by the same GVPT2 polyad eigenvectors.

    The default ``second`` path contracts the complete D1/D2/D3 property surface with
    F3/F4/G'/G'' through second order, with external polyad dressing.
    The input field retains its equilibrium-only geometric potential.
    ``first`` is retained only for explicit legacy diagnostics; the
    production KeyMaker protocol always requests second order.
    ``initial_state`` selects a fundamental as the initial PT2D eigenstate;
    its polyad coefficients mix the initial side of every transition.
    """
    if dipole is None and polarizability is None and roa_response is None:
        raise ValueError("GVPT2 spectrum requires IR, Raman, or ROA property derivatives")
    if transition_order not in {"first", "second"}:
        raise ValueError("transition order must be 'first' or 'second'")
    if transition_order == "second" and field is None:
        raise ValueError("second-order curvilinear transition moments require the force field")
    if magnetic_dipole is not None and (dipole is None or transition_order != "second"):
        raise ValueError("VCD requires electric dipole and second-order transition transport")
    if roa_response is not None and transition_order != "second":
        raise ValueError("ROA requires second-order transition transport")
    states = result.states
    count = len(states)
    ground_state = tuple(0 for _ in states[0])
    initial = ground_state if initial_state is None else tuple(initial_state)
    if initial not in states or (initial != ground_state and sum(initial) != 1):
        raise ValueError("initial state must be the ground state or an included fundamental")
    if initial != ground_state and transition_order != "second":
        raise ValueError("hot bands require second-order transition moments")
    if transition_states is None:
        requested_indices = tuple(range(count))
    else:
        state_index = {state: index for index, state in enumerate(states)}
        requested = tuple(tuple(state) for state in transition_states)
        if any(state not in state_index for state in requested):
            raise ValueError("requested transition state is absent from the GVPT2 basis")
        requested_indices = tuple(dict.fromkeys(state_index[state] for state in requested))
    evaluation = set(requested_indices)
    for block in result.blocks:
        members = {int(index) for index in block["indices"]}
        if evaluation & members:
            evaluation.update(members)
    evaluation_indices = tuple(sorted(evaluation))
    energies = np.asarray(result.deperturbed_cm, dtype=float).copy()
    polyad_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for block in result.blocks:
        members = np.asarray(block["indices"], dtype=int)
        vectors = np.asarray(block["eigenvectors"], dtype=float)
        roots = np.asarray(block["eigenvalues_cm"], dtype=float)
        if vectors.shape != (len(members), len(members)) or roots.shape != (len(members),):
            raise ValueError("GVPT2 polyad has invalid eigenvectors or eigenvalues")
        polyad_blocks.append((members, vectors))
        energies[members] = roots
    def mix(values: np.ndarray) -> np.ndarray:
        return mix_sonic_polyad_property_moments(result, values)
    assigned_indices = list(range(count))
    for block, (members, vectors) in zip(result.blocks, polyad_blocks, strict=True):
        basis_to_root = block.get("basis_to_root")
        if basis_to_root is None:
            from .local_quadratic import _maximum_overlap_assignment
            basis_to_root = _maximum_overlap_assignment(np.abs(vectors) ** 2)
        for row, column in enumerate(basis_to_root):
            assigned_indices[int(members[column])] = int(members[row])
    assigned = tuple(states[index] for index in assigned_indices)
    initial_basis = states.index(initial)
    initial_root = initial_basis
    initial_components = ((initial_basis, 1.0),)
    for block, (members, vectors) in zip(result.blocks, polyad_blocks, strict=True):
        if initial_basis in members:
            row = int(np.flatnonzero(members == initial_basis)[0])
            basis_to_root = block.get("basis_to_root")
            if basis_to_root is None:
                from .local_quadratic import _maximum_overlap_assignment
                basis_to_root = _maximum_overlap_assignment(np.abs(vectors) ** 2)
            column = int(basis_to_root[row])
            initial_root = int(members[column])
            initial_components = tuple(
                (int(member), float(vectors[position, column]))
                for position, member in enumerate(members)
            )
            break
    energies -= energies[initial_root]
    reduced_cache: dict = {} if _reduced_cache is None else _reduced_cache

    def dressed_orders(
        properties: SonicPropertyDerivatives | SonicMomentumPropertyDerivatives,
        components: int, *, momentum: bool = False,
    ) -> np.ndarray:
        orders = np.zeros((3, count, components), dtype=complex if momentum else float)
        for basis_index, coefficient in initial_components:
            basis_orders = _curvilinear_basis_moments_by_order(
                result, properties, components, field, initial_state=states[basis_index],
                momentum=momentum, _reduced_cache=reduced_cache,
                final_indices=evaluation_indices,
            )
            orders += coefficient * np.stack(basis_orders)
        return np.stack([mix(values) for values in orders])

    dipole_orders = ir_orders = None
    if dipole is None:
        dipole_moments = ir = None
    else:
        if transition_order == "second":
            dipole_orders = dressed_orders(dipole, 3)
            dipole_moments = np.sum(dipole_orders, axis=0)
            m0, m1, m2 = dipole_orders
            prefactor = IR_TRANSITION_FACTOR * np.maximum(energies, 0)
            ir_orders = np.stack((
                prefactor * np.sum(m0 * m0, axis=1),
                prefactor * 2 * np.sum(m0 * m1, axis=1),
                prefactor * np.sum(m1 * m1 + 2 * m0 * m2, axis=1),
            ))
            ir_orders[:, initial_root] = 0.0
        else:
            basis = (_basis_moments(states, dipole, 3) if field is None else
                     _mechanical_basis_moments(states, dipole, 3, field, result.resonances))
            dipole_moments = mix(basis)
        ir = IR_TRANSITION_FACTOR * np.maximum(energies, 0) * np.sum(dipole_moments**2, axis=1)
        ir[initial_root] = 0.0
    if polarizability is None:
        tensors = raman = None
    else:
        if transition_order == "second":
            tensors = np.sum(dressed_orders(polarizability, 6), axis=0)
        else:
            basis = (_basis_moments(states, polarizability, 6) if field is None else
                     _mechanical_basis_moments(states, polarizability, 6, field, result.resonances))
            tensors = mix(basis)
        raman = _raman_strength(tensors)
        raman[initial_root] = 0.0
    magnetic_moments = vcd = magnetic_orders = vcd_orders = None
    if magnetic_dipole is not None:
        magnetic_orders = dressed_orders(magnetic_dipole, 3, momentum=True)
        magnetic_moments = np.sum(magnetic_orders, axis=0)
        vcd = vcd_rotatory_strength(dipole_moments, magnetic_moments)
        vcd[initial_root] = 0.0
        if dipole_orders is not None:
            e0, e1, e2 = dipole_orders
            m0, m1, m2 = magnetic_orders
            vcd_orders = np.stack((
                vcd_rotatory_strength(e0, m0),
                vcd_rotatory_strength(e0, m1) + vcd_rotatory_strength(e1, m0),
                vcd_rotatory_strength(e0, m2) + vcd_rotatory_strength(e1, m1)
                + vcd_rotatory_strength(e2, m0),
            ))
            vcd_orders[:, initial_root] = 0.0
    roa = None
    if roa_response is not None:
        alpha_roa = np.sum(dressed_orders(roa_response.alpha, 9), axis=0).reshape(count, 3, 3)
        g_roa = np.sum(dressed_orders(roa_response.g_prime, 9), axis=0).reshape(count, 3, 3)
        a_roa = np.sum(dressed_orders(roa_response.a, 27), axis=0).reshape(count, 3, 3, 3)
        roa = roa_scp_backscattering(
            alpha_roa, g_roa, a_roa,
            incident_frequency_au=roa_response.incident_frequency_au,
        )
    # Preserve the established public array result, while all contractions
    # above operate only on the populated local polyad blocks.
    mixing = np.eye(count)
    for members, vectors in polyad_blocks:
        mixing[np.ix_(members, members)] = vectors
    return SonicGVPT2Spectrum(
        states=assigned, transition_energies_cm=energies,
        dipole_moments_au=dipole_moments, ir_intensities_km_mol=ir,
        polarizability_tensors_au=tensors, raman_strength_ang6=raman,
        polyad_eigenvectors=mixing,
        mechanical_correction=("curvilinear-second-order" if transition_order == "second"
                               else "cubic-first-order" if field is not None else "none"),
        dipole_moment_orders_au=dipole_orders,
        ir_intensity_orders_km_mol=ir_orders,
        initial_state=initial,
        magnetic_moments_au=magnetic_moments,
        vcd_rotatory_strength_au=vcd,
        magnetic_moment_orders_au=magnetic_orders,
        vcd_rotatory_strength_orders_au=vcd_orders,
        roa_scp_backscatter=roa,
        evaluated_states=tuple(states[index] for index in evaluation_indices),
    )


_FORKED_SPECTRA_CONTEXT: tuple | None = None
_FORKED_SPECTRA_LOCK = Lock()
SONIC_PROPERTY_WORKER_POLICY = "trinity.property-workers.v2"


def _sonic_gvpt2_spectra_chunk(payload: tuple) -> tuple[tuple[int, SonicGVPT2Spectrum], ...]:
    """Evaluate one process-local group with one shared reduction cache."""
    indexed_initials, result, options = payload
    cache: dict = {}
    return tuple(
        (index, sonic_gvpt2_spectrum(
            result, initial_state=initial, _reduced_cache=cache, **options,
        ))
        for index, initial in indexed_initials
    )


def _sonic_gvpt2_spectra_forked_chunk(
    indexed_initials: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[tuple[int, SonicGVPT2Spectrum], ...]:
    """Use the read-mostly reduction state inherited by a forked worker."""
    if _FORKED_SPECTRA_CONTEXT is None:
        raise RuntimeError("forked GVPT2 spectrum worker has no inherited context")
    result, options, cache = _FORKED_SPECTRA_CONTEXT
    return tuple(
        (index, sonic_gvpt2_spectrum(
            result, initial_state=initial, _reduced_cache=cache, **options,
        ))
        for index, initial in indexed_initials
    )


def _sonic_gvpt2_final_chunk(payload: tuple) -> SonicGVPT2Spectrum:
    states, result, options = payload
    return sonic_gvpt2_spectrum(result, transition_states=states, _reduced_cache={}, **options)


def _sonic_gvpt2_forked_final_chunk(
    states: tuple[tuple[int, ...], ...],
) -> SonicGVPT2Spectrum:
    if _FORKED_SPECTRA_CONTEXT is None:
        raise RuntimeError("forked GVPT2 spectrum worker has no inherited context")
    result, options, cache = _FORKED_SPECTRA_CONTEXT
    return sonic_gvpt2_spectrum(
        result, transition_states=states, _reduced_cache=cache, **options,
    )


def _available_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size / 1024**3)
    except (AttributeError, OSError, ValueError):
        try:
            # macOS does not expose SC_AVPHYS_PAGES.  Half the physical
            # memory is a conservative process budget that leaves space for
            # the parent and unrelated applications.
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(0.5 * pages * page_size / 1024**3)
        except (AttributeError, OSError, ValueError):
            return None


def resolve_sonic_property_workers(
    mode_count: int, task_count: int, *, requested: int = 0,
    available_memory_gb: float | None = None,
) -> int:
    """Choose a bounded property-contraction process count.

    ``requested=0`` applies the version-2 automatic policy: small spectra stay
    serial, CPU use is capped at six processes, and at least 0.75 GB of
    available memory is reserved per process.  An explicit positive request is
    still reduced to the number of independent tasks and the memory bound.
    """
    if mode_count < 1 or task_count < 1 or requested < 0:
        raise ValueError("invalid property-worker dimensions or request")
    if requested == 0 and (task_count < 12 or mode_count < 8):
        return 1
    cpu_bound = max(1, min(6, (os.cpu_count() or 1) // 2, task_count))
    desired = cpu_bound if requested == 0 else min(requested, task_count)
    memory = _available_memory_gb() if available_memory_gb is None else available_memory_gb
    if memory is not None:
        if not np.isfinite(memory) or memory <= 0:
            raise ValueError("available property-worker memory must be positive")
        desired = min(desired, max(1, int(0.70 * memory / 0.75)))
    return max(1, desired)


def _spectrum_transition_groups(
    result: SonicGVPT2Result, requested: tuple[tuple[int, ...], ...], workers: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Keep every variational block in one balanced final-state group."""
    state_index = {state: index for index, state in enumerate(result.states)}
    remaining = {state_index[state] for state in requested}
    units: list[tuple[int, ...]] = []
    for block in result.blocks:
        members = tuple(int(index) for index in block["indices"])
        if remaining.intersection(members):
            units.append(members)
            remaining.difference_update(members)
    units.extend((index,) for index in sorted(remaining))
    groups: list[list[int]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for unit in sorted(units, key=len, reverse=True):
        target = min(range(workers), key=loads.__getitem__)
        groups[target].extend(unit)
        loads[target] += len(unit)
    return tuple(
        tuple(result.states[index] for index in sorted(set(group)))
        for group in groups if group
    )


def _merge_partial_spectra(
    result: SonicGVPT2Result, parts: tuple[SonicGVPT2Spectrum, ...],
) -> SonicGVPT2Spectrum:
    """Assemble disjoint final-state contractions without changing values."""
    base = parts[0]
    state_index = {state: index for index, state in enumerate(result.states)}
    evaluated = set()
    for part in parts:
        evaluated.update(part.evaluated_states or ())

    def merge_array(name: str, *, state_axis: int = 0):
        template = getattr(base, name)
        if template is None:
            return None
        output = np.zeros_like(template)
        for part in parts:
            values = getattr(part, name)
            indices = [state_index[state] for state in (part.evaluated_states or ())]
            selector = [slice(None)] * output.ndim
            selector[state_axis] = indices
            output[tuple(selector)] = values[tuple(selector)]
        return output

    roa = None
    # ROA is stored as a nested dataclass rather than flattened attributes.
    if base.roa_scp_backscatter is not None:
        roa_values = {}
        for item in fields(ROABackscattering):
            output = np.zeros_like(getattr(base.roa_scp_backscatter, item.name))
            for part in parts:
                indices = [state_index[state] for state in (part.evaluated_states or ())]
                output[indices] = getattr(part.roa_scp_backscatter, item.name)[indices]
            roa_values[item.name] = output
        roa = ROABackscattering(**roa_values)
    return replace(
        base,
        dipole_moments_au=merge_array("dipole_moments_au"),
        ir_intensities_km_mol=merge_array("ir_intensities_km_mol"),
        polarizability_tensors_au=merge_array("polarizability_tensors_au"),
        raman_strength_ang6=merge_array("raman_strength_ang6"),
        dipole_moment_orders_au=merge_array("dipole_moment_orders_au", state_axis=1),
        ir_intensity_orders_km_mol=merge_array("ir_intensity_orders_km_mol", state_axis=1),
        magnetic_moments_au=merge_array("magnetic_moments_au"),
        vcd_rotatory_strength_au=merge_array("vcd_rotatory_strength_au"),
        magnetic_moment_orders_au=merge_array("magnetic_moment_orders_au", state_axis=1),
        vcd_rotatory_strength_orders_au=merge_array(
            "vcd_rotatory_strength_orders_au", state_axis=1,
        ),
        roa_scp_backscatter=roa,
        evaluated_states=tuple(state for state in result.states if state in evaluated),
    )


def sonic_gvpt2_spectra(
    result: SonicGVPT2Result, *,
    initial_states: tuple[tuple[int, ...], ...],
    dipole: SonicPropertyDerivatives | None = None,
    polarizability: SonicPropertyDerivatives | None = None,
    magnetic_dipole: SonicMomentumPropertyDerivatives | None = None,
    roa_response: SonicROAPropertySurfaces | None = None,
    field: SonicVPT2Input | None = None,
    transition_order: Literal["first", "second"] = "second",
    transition_states: tuple[tuple[int, ...], ...] | None = None,
    workers: int = 0,
    _reduced_cache: dict | None = None,
) -> tuple[SonicGVPT2Spectrum, ...]:
    """Evaluate spectra from several initial states in reusable blocks.

    Every worker prepares the modal field and reduced property equations once,
    then reuses them for all initial states assigned to that worker.  Processes
    are used because the occupation-dependent algebra is dominated by Python
    integer and sparse-polynomial operations.  Result ordering always follows
    ``initial_states``.
    """
    initials = tuple(tuple(state) for state in initial_states)
    if not initials:
        return ()
    if not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    options = {
        "dipole": dipole,
        "polarizability": polarizability,
        "magnetic_dipole": magnetic_dipole,
        "roa_response": roa_response,
        "field": field,
        "transition_order": transition_order,
        "transition_states": transition_states,
    }
    indexed = tuple(enumerate(initials))
    worker_count = resolve_sonic_property_workers(
        len(result.frequencies_cm), len(indexed), requested=workers,
    )
    groups = tuple(tuple(indexed[offset::worker_count]) for offset in range(worker_count))
    payloads = tuple((group, result, options) for group in groups if group)
    if worker_count == 1:
        cache = {} if _reduced_cache is None else _reduced_cache
        completed = tuple(
            (index, sonic_gvpt2_spectrum(
                result, initial_state=initial, _reduced_cache=cache, **options,
            ))
            for index, initial in indexed
        )
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_all_start_methods, get_context
        methods = get_all_start_methods()
        use_fork = "fork" in methods
        context = get_context("fork" if use_fork else methods[0])
        if use_fork:
            global _FORKED_SPECTRA_CONTEXT
            with _FORKED_SPECTRA_LOCK:
                _FORKED_SPECTRA_CONTEXT = (
                    result, options, {} if _reduced_cache is None else _reduced_cache,
                )
                try:
                    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                        completed = tuple(
                            item for group in executor.map(
                                _sonic_gvpt2_spectra_forked_chunk, groups,
                            ) for item in group
                        )
                finally:
                    _FORKED_SPECTRA_CONTEXT = None
        else:
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                completed = tuple(
                    item for group in executor.map(_sonic_gvpt2_spectra_chunk, payloads)
                    for item in group
                )
    ordered: list[SonicGVPT2Spectrum | None] = [None] * len(initials)
    for index, spectrum in completed:
        ordered[index] = spectrum
    if any(spectrum is None for spectrum in ordered):
        raise RuntimeError("parallel GVPT2 spectrum evaluation returned an incomplete result")
    return tuple(ordered)  # type: ignore[return-value]


def sonic_gvpt2_spectrum_parallel(
    result: SonicGVPT2Result, *,
    dipole: SonicPropertyDerivatives | None = None,
    polarizability: SonicPropertyDerivatives | None = None,
    magnetic_dipole: SonicMomentumPropertyDerivatives | None = None,
    roa_response: SonicROAPropertySurfaces | None = None,
    field: SonicVPT2Input | None = None,
    transition_order: Literal["first", "second"] = "second",
    initial_state: tuple[int, ...] | None = None,
    transition_states: tuple[tuple[int, ...], ...] | None = None,
    workers: int = 0,
    _reduced_cache: dict | None = None,
) -> SonicGVPT2Spectrum:
    """Evaluate one spectrum with balanced, polyad-preserving final blocks."""
    requested = (tuple(result.states) if transition_states is None else
                 tuple(dict.fromkeys(tuple(state) for state in transition_states)))
    state_set = set(result.states)
    if not requested or any(state not in state_set for state in requested):
        raise ValueError("parallel spectrum requires nonempty included transition states")
    if transition_order == "second" and field is None:
        raise ValueError("second-order curvilinear transition moments require the force field")
    worker_count = resolve_sonic_property_workers(
        len(result.frequencies_cm), len(requested), requested=workers,
    )
    if worker_count == 1:
        return sonic_gvpt2_spectrum(
            result, dipole=dipole, polarizability=polarizability,
            magnetic_dipole=magnetic_dipole, roa_response=roa_response,
            field=field, transition_order=transition_order,
            initial_state=initial_state, transition_states=requested,
            _reduced_cache=_reduced_cache,
        )
    options = {
        "dipole": dipole, "polarizability": polarizability,
        "magnetic_dipole": magnetic_dipole, "roa_response": roa_response,
        "field": field, "transition_order": transition_order,
        "initial_state": initial_state,
    }
    groups = _spectrum_transition_groups(result, requested, worker_count)
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_all_start_methods, get_context
    methods = get_all_start_methods()
    use_fork = "fork" in methods
    context = get_context("fork" if use_fork else methods[0])
    cache = {} if _reduced_cache is None else _reduced_cache
    if use_fork:
        global _FORKED_SPECTRA_CONTEXT
        with _FORKED_SPECTRA_LOCK:
            # Build the read-mostly pair field and property surfaces once.
            warm_state = (tuple(0 for _ in result.states[0])
                          if initial_state is None else tuple(initial_state))
            warm_index = result.states.index(warm_state)
            surfaces = [
                (dipole, 3, False), (polarizability, 6, False),
                (magnetic_dipole, 3, True),
            ]
            if roa_response is not None:
                surfaces.extend((
                    (roa_response.alpha, 9, False),
                    (roa_response.g_prime, 9, False),
                    (roa_response.a, 27, False),
                ))
            for properties, components, momentum in surfaces:
                if properties is not None and transition_order == "second":
                    _curvilinear_basis_moments_by_order(
                        result, properties, components, field,
                        initial_state=warm_state, momentum=momentum,
                        _reduced_cache=cache, final_indices=(warm_index,),
                    )
            _FORKED_SPECTRA_CONTEXT = (result, options, cache)
            try:
                with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as executor:
                    parts = tuple(executor.map(_sonic_gvpt2_forked_final_chunk, groups))
            finally:
                _FORKED_SPECTRA_CONTEXT = None
    else:
        payloads = tuple((group, result, options) for group in groups)
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as executor:
            parts = tuple(executor.map(_sonic_gvpt2_final_chunk, payloads))
    return _merge_partial_spectra(result, parts)


__all__ = [
    "SonicPropertyDerivatives", "SonicDiagonalPropertyExpectation",
    "SonicMomentumPropertyDerivatives",
    "SonicROAPropertySurfaces", "SonicGVPT2Spectrum",
    "modal_property_derivatives_from_packed_mass_weighted",
    "project_sonic_property_derivatives",
    "property_derivatives_from_cartesian",
    "acquire_modal_property_derivatives_from_gradients",
    "acquire_modal_momentum_derivatives_from_values",
    "sonic_gvpt2_spectrum", "sonic_gvpt2_spectra", "sonic_gvpt2_spectrum_parallel",
    "resolve_sonic_property_workers", "SONIC_PROPERTY_WORKER_POLICY",
    "sonic_diagonal_property_expectation",
]
