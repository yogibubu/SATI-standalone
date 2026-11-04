"""Directional CALPGM matrix elements for operators that change ``l``.

CALPGM correlates the signs of delta-l and delta-K before symmetry
adaptation.  Building a Hermitian vibrational projector and a Hermitian
rotational operator separately loses that correlation.  This module keeps
the three directional branches explicit in the reduced Wang basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .calpgm_adapter import calpgm_getmask
from .calpgm_native import CALPGMOperatorCode
from .spin_hyperfine import (
    _angular_momentum_rank1_component,
    _rank2_rotational_space_component,
    clebsch_gordan_doubled,
    wigner_3j_doubled,
    wigner_6j_doubled,
)


_WANG_PHASE = (0, 1, 2, 3)
_MODD = 0x40
_MIDEN = 0x10
_MLZ = 0x100
_MFCM = 0x200


@dataclass(frozen=True)
class CALPGMDirectionalRoute:
    direction_rank: int
    delta_k: int
    tensor_rank: int
    delta_l: int
    factor: float
    block_symmetries: tuple[int, ...]
    identity_itot: bool
    commutator_n2: bool = False
    odd_order: bool = False
    sine_fourier: bool = False
    lz_operator: bool = False


def directional_route(
    code: CALPGMOperatorCode, l_values: Sequence[int], nitot: int
) -> CALPGMDirectionalRoute | None:
    """Derive the positive-K DIRCOS route from a CALPGM operator code."""

    if code.vibrational_upper >= len(l_values) or code.vibrational_lower >= len(l_values):
        # CALPGM's terminal vibrational index is the identity in vibrational
        # space.  Pure rotational and centrifugal-distortion operators must
        # use the same positive-K Wang convention as Fourier terms; otherwise
        # the two operator sets acquire incompatible phases in signed-l
        # manifolds.
        if (
            code.operator_family == "rotation"
            and code.vibrational_upper == code.vibrational_lower
            and code.projection_type < 80
        ):
            rank = code.projection_rank
            delta_k = code.delta_k
            if delta_k is None:
                return None
            family = code.projection_type // 20
            symmetry = (0, 3, 2, 1)[family] if family < 4 else 0
            factor = (
                -2.0
                if delta_k == 2 and symmetry == 0 and rank == 2
                else 1.0
            )
            return CALPGMDirectionalRoute(
                rank,
                delta_k,
                0,
                0,
                factor,
                (symmetry,),
                identity_itot=True,
                odd_order=bool(max(delta_k, rank) & 1),
            )
        return None
    upper = int(l_values[code.vibrational_upper])
    lower = int(l_values[code.vibrational_lower])
    delta_l = upper - lower
    spins = int(bool(code.spin_1)) + int(bool(code.spin_2))
    if (
        spins == 0
        and code.vibrational_upper == code.vibrational_lower
        and code.fourier_flag
        and code.projection_type < 80
    ):
        # PASORT changes the Wang symmetry of a sine Fourier carrier before
        # DIRCOS is evaluated.  Applying the Fourier factor later in a full
        # signed-K product basis loses this discrete routing, most visibly for
        # the c-type, delta-K=1 family.  Reconstruct the route directly from
        # IDPAR, as SPINV's idpari/pasort pair does.
        from .calpgm_native import _calpgm_fourier_mode

        rank = code.projection_rank
        delta_k = code.delta_k
        if delta_k is None:
            return None
        family = code.projection_type // 20
        base_symmetry = (0, 3, 2, 1)[family] if family < 4 else 0
        _harmonic, sine = _calpgm_fourier_mode(code.fourier_flag)
        symmetry = 3 - base_symmetry if sine else base_symmetry
        imaginary_order = max(delta_k, rank) + int(sine)
        # NEWLZ stores the second member of a signed-l pair with the
        # complementary Wang symmetry and one additional imaginary phase.
        # This is a property of the encoded state, not of a molecule.
        if upper < 0:
            symmetry = 3 - symmetry
            imaginary_order += 1
        # A nonzero twofold Cartesian component is stored with CALPGM's
        # historical half normalization.  The correction follows from the
        # encoded tensor symmetry, so it also applies to future molecules
        # and vibrational manifolds using the same operator algebra.
        factor = -2.0 if delta_k == 2 and symmetry == 0 and rank == 2 else 1.0
        values = (rank, delta_k, 0, factor)
        symmetries = (symmetry,)
        return CALPGMDirectionalRoute(
            values[0],
            values[1],
            values[2],
            0,
            values[3],
            symmetries,
            identity_itot=True,
            odd_order=bool(imaginary_order & 1),
            sine_fourier=sine,
            lz_operator=upper < 0,
        )
    if code.projection_type == 63 and spins == 0:
        values = (4, 3, 0, 1.0)
        delta_l = 0
        base_symmetry = 1
    elif code.vibrational_upper == code.vibrational_lower:
        return None
    elif not upper or upper != -lower:
        return None
    elif code.projection_type == 4:
        if spins == 0:
            values = (0, 2, 0, -2.0)
        elif spins == 1:
            values = (2, 2, 1, -math.sqrt(6.0))
        elif spins == 2:
            values = (2, 2, 2, -3.0 if delta_l > 0 else -1.5)
        else:
            return None
        base_symmetry = 0
    elif code.projection_type == 61:
        if spins == 0:
            values = (2, 1, 0, 1.0)
        elif spins == 1:
            values = (1, 1, 1, math.sqrt(1.5))
        else:
            return None
        base_symmetry = 1
    elif spins == 2 and code.projection_type == 0:
        values = (0, 0, 0, 0.75)
        base_symmetry = 0
    elif spins == 2 and code.projection_type == 1:
        values = (2, 0, 2, 0.5)
        base_symmetry = 0
    else:
        return None
    symmetries = (base_symmetry,)
    if spins == 2 and nitot >= 3 and base_symmetry == 0:
        symmetries = (0, 3)
    return CALPGMDirectionalRoute(
        values[0],
        values[1],
        values[2],
        delta_l,
        values[3],
        symmetries,
        identity_itot=spins < 2,
        commutator_n2=(spins == 1 and code.projection_type == 61),
    )


def directional_subblock_elements(
    bra: Mapping[str, Any],
    ket: Mapping[str, Any],
    route: CALPGMDirectionalRoute,
    *,
    block_symmetry: int,
    alpha: int,
    nitot: int,
    ifup: int,
) -> tuple[tuple[int, int, float], ...]:
    mask = calpgm_getmask(
        bra,
        ket,
        delta_k=route.delta_k,
        delta_l=route.delta_l,
        alpha=alpha,
        nuclear_symmetry_order=nitot,
        identity_itot=route.identity_itot,
    )
    if not mask:
        return ()
    flags = _MIDEN if route.identity_itot else 0
    if route.odd_order:
        flags |= _MODD
    if route.sine_fourier:
        flags |= _MFCM
    if route.lz_operator:
        flags |= _MLZ
    return _dircos(
        _directional_label(bra, nitot),
        _directional_label(ket, nitot),
        route.direction_rank,
        route.delta_k,
        ifup,
        flags,
        mask,
    )


def rotational_reduced_factor(rank: int, tensor_rank: int, ni: int, nj: int) -> float:
    if rank == tensor_rank:
        return 1.0
    nsum = ni + nj + 1
    dn = float(nsum)
    dnsq = dn * dn
    difference = ni - nj
    difference_sq = float(difference * difference) if difference else dn
    result = 1.0
    if rank > tensor_rank:
        lower, upper = tensor_rank, rank
        if lower == 0:
            lower = 1
            result = ni * (ni + 1.0) / dn
    else:
        lower, upper = rank, tensor_rank
        if lower == 0:
            lower = 1
            result = ni * (ni + 1.0) * dn
    if upper >= nsum:
        return 0.0
    while upper > lower:
        square = float(upper * upper)
        result *= 0.25 * (dnsq - square)
        if difference:
            result -= result * difference_sq / square
        upper -= 1
    return math.sqrt(max(0.0, result))


def electron_spin_tensor_factor(
    ni: int, nj: int, two_ji: int, two_jj: int, two_ii: int, two_ij: int, two_s: int
) -> float:
    """Reduced rank-one electron-spin factor in the CALPGM phase convention."""

    if two_ji != two_jj or two_ii != two_ij:
        return 0.0
    value = wigner_6j_doubled(2 * ni, 2, 2 * nj, two_s, two_ji, two_s)
    if (two_ji + two_s + 2 * nj) & 2:
        value = -value
    return value


def equivalent_spin_tensor_factor(
    ni: int,
    nj: int,
    two_ji: int,
    two_jj: int,
    two_ii: int,
    two_ij: int,
    two_f: int,
    symmetry_i: int,
    symmetry_j: int,
    tensor_rank: int,
    alpha: int,
    equivalent_spins: tuple[int, ...],
    multiplicity_i: int = 0,
    multiplicity_j: int = 0,
) -> float | None:
    """Reduced tensor for a cyclic set of equivalent nuclei.

    The construction is performed in the product-spin space and projected on
    eigenfunctions of the cyclic permutation.  It therefore covers both the
    A and E nuclear-spin species, including the A/E matrix elements that a
    total-I approximation necessarily misses.
    """

    if not equivalent_spins or len(set(equivalent_spins)) != 1 or tensor_rank not in (0, 2):
        return None
    order = len(equivalent_spins)
    qi, qj = symmetry_i % order, symmetry_j % order
    pseudo = alpha >= order
    alpha %= order
    pairs = [((order - qi) % order, qj)] if pseudo else [(qi, qj)]
    candidates: list[complex] = []
    for qbra, qket in pairs:
        for fourier_sign in (-1, 1):
            if tensor_rank == 0:
                value = _equivalent_scalar_contraction(
                    ni, two_ji, two_ii, qbra, two_jj, two_ij, qket, two_f,
                    alpha, fourier_sign, equivalent_spins,
                    multiplicity_i, multiplicity_j,
                )
            else:
                value = _equivalent_rank2_contraction(
                    ni, nj, two_ji, two_ii, qbra, two_jj, two_ij, qket, two_f,
                    alpha, fourier_sign, equivalent_spins,
                    multiplicity_i, multiplicity_j,
                )
            candidates.append(value)
    value = max(candidates, key=abs, default=0.0j)
    value = complex(np.real_if_close(value, tol=1000))
    if abs(value.imag) > 1.0e-10:
        return None
    if tensor_rank == 2:
        return math.sqrt(2.0) * value.real
    return (2.0 / math.sqrt(order)) * value.real


@lru_cache(maxsize=128)
def _legacy_three_half_state(two_i: int, symmetry: int, two_m: int) -> np.ndarray:
    states = tuple(
        (a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)
    )
    result = np.zeros(8, dtype=complex)
    if two_i == 3 and symmetry == 0:
        selected = [index for index, state in enumerate(states) if sum(state) == two_m]
        if selected:
            result[selected] = 1.0 / math.sqrt(len(selected))
        result.setflags(write=False)
        return result
    if two_i != 1 or symmetry not in (1, 2) or abs(two_m) != 1:
        result.setflags(write=False)
        return result
    omega = np.exp(2.0j * math.pi / 3.0)
    minority = -1 if two_m == 1 else 1
    phase = 1.0 if two_m == 1 else -1.0
    lookup = {state: index for index, state in enumerate(states)}
    for site in range(3):
        state = [-minority] * 3
        state[site] = minority
        result[lookup[tuple(state)]] = phase * omega ** (symmetry * site) / math.sqrt(3.0)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=128)
def _spin_cartesian(two_s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = two_s + 1
    m_values = np.arange(-two_s, two_s + 1, 2, dtype=float) / 2.0
    plus = np.zeros((dimension, dimension), dtype=complex)
    spin = two_s / 2.0
    for column, m_value in enumerate(m_values[:-1]):
        plus[column + 1, column] = math.sqrt(
            spin * (spin + 1.0) - m_value * (m_value + 1.0)
        )
    minus = plus.T
    return (plus + minus) / 2.0, (plus - minus) / (2.0j), np.diag(m_values)


@lru_cache(maxsize=128)
def _equivalent_spin_space(
    equivalent_spins: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    states = tuple(itertools.product(*(range(-spin, spin + 1, 2) for spin in equivalent_spins)))
    dimensions = tuple(spin + 1 for spin in equivalent_spins)
    total = [np.zeros((len(states), len(states)), dtype=complex) for _ in range(3)]
    for site, two_s in enumerate(equivalent_spins):
        local = _spin_cartesian(two_s)
        for axis in range(3):
            embedded = np.ones((1, 1), dtype=complex)
            for position, dimension in enumerate(dimensions):
                embedded = np.kron(
                    embedded,
                    local[axis] if position == site else np.eye(dimension),
                )
            total[axis] += embedded
    i_squared = sum(component @ component for component in total)
    lower = total[0] - 1.0j * total[1]
    i_squared.setflags(write=False)
    lower.setflags(write=False)
    return states, i_squared, lower


@lru_cache(maxsize=512)
def _equivalent_nuclear_state(
    equivalent_spins: tuple[int, ...], two_i: int, symmetry: int, two_m: int,
    multiplicity: int = 0,
) -> np.ndarray:
    if isinstance(multiplicity, bool) or int(multiplicity) != multiplicity or multiplicity < 0:
        raise ValueError("equivalent-spin multiplicity must be a nonnegative integer")
    multiplicity = int(multiplicity)
    if equivalent_spins == (1, 1, 1):
        if multiplicity:
            raise ValueError("the three-spin CALPGM sector has multiplicity one")
        return _legacy_three_half_state(two_i, symmetry, two_m)
    states, i_squared, lower = _equivalent_spin_space(equivalent_spins)
    result = np.zeros(len(states), dtype=complex)
    if (
        abs(two_m) > two_i
        or ((two_i - two_m) & 1)
        or two_i > sum(equivalent_spins)
    ):
        result.setflags(write=False)
        return result
    order = len(equivalent_spins)
    if order > 1 and len(set(equivalent_spins)) != 1:
        result.setflags(write=False)
        return result
    stretched = [index for index, state in enumerate(states) if sum(state) == two_i]
    if not stretched:
        result.setflags(write=False)
        return result
    eigenvalues, eigenvectors = np.linalg.eigh(i_squared[np.ix_(stretched, stretched)])
    target = (two_i / 2.0) * (two_i / 2.0 + 1.0)
    selected = eigenvectors[:, np.isclose(eigenvalues, target, atol=1.0e-9, rtol=1.0e-9)]
    if selected.size == 0:
        result.setflags(write=False)
        return result
    projector_i = selected @ selected.conj().T
    omega = np.exp(2.0j * math.pi / order)
    permutation = np.zeros((len(stretched), len(stretched)), dtype=complex)
    lookup = {states[index]: row for row, index in enumerate(stretched)}
    for column, index in enumerate(stretched):
        rotated = states[index][-1:] + states[index][:-1]
        permutation[lookup[rotated], column] = 1.0
    projector_q = sum(
        omega ** (-symmetry * power) * np.linalg.matrix_power(permutation, power)
        for power in range(order)
    ) / order
    projector = projector_i @ projector_q
    basis = _canonical_projector_basis(projector)
    if not basis:
        result.setflags(write=False)
        return result
    if multiplicity >= len(basis):
        raise ValueError(
            f"equivalent-spin multiplicity {multiplicity} exceeds sector rank {len(basis)}"
        )
    vector = basis[multiplicity]
    result[stretched] = vector
    current_m = two_i
    while current_m > two_m:
        denominator = math.sqrt(
            (two_i / 2.0) * (two_i / 2.0 + 1.0)
            - (current_m / 2.0) * (current_m / 2.0 - 1.0)
        )
        result = lower @ result / denominator
        current_m -= 2
    pivot = int(np.argmax(np.abs(result)))
    if abs(result[pivot]) > 0.0:
        result *= np.exp(-1.0j * np.angle(result[pivot]))
    result[np.abs(result) < 1.0e-14] = 0.0
    result.setflags(write=False)
    return result


def _canonical_projector_basis(projector: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return a deterministic orthonormal basis for a projector range."""

    vectors: list[np.ndarray] = []
    for column in range(projector.shape[1]):
        candidate = np.array(projector[:, column], dtype=complex, copy=True)
        for previous in vectors:
            candidate -= previous * np.vdot(previous, candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= 1.0e-10:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        candidate *= np.exp(-1.0j * np.angle(candidate[pivot]))
        candidate[np.abs(candidate) < 1.0e-14] = 0.0
        candidate.setflags(write=False)
        vectors.append(candidate)
    return tuple(vectors)


@lru_cache(maxsize=256)
def _equivalent_fourier_component(
    equivalent_spins: tuple[int, ...], alpha: int, component: int, sign: int
) -> np.ndarray:
    order = len(equivalent_spins)
    omega = np.exp(2.0j * math.pi / order)
    dimension = math.prod(spin + 1 for spin in equivalent_spins)
    result = np.zeros((dimension, dimension), dtype=complex)
    for site, two_s in enumerate(equivalent_spins):
        spin = _angular_momentum_rank1_component(two_s, component)
        embedded = np.ones((1, 1), dtype=complex)
        for position, other_two_s in enumerate(equivalent_spins):
            embedded = np.kron(
                embedded, spin if position == site else np.eye(other_two_s + 1)
            )
        result += omega ** (sign * alpha * site) * embedded
    result = result / math.sqrt(order)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=4096)
def _equivalent_coupled_state(
    N: int, two_j: int, two_i: int, symmetry: int, two_f: int,
    equivalent_spins: tuple[int, ...], multiplicity: int = 0,
) -> np.ndarray:
    nuclear_dimension = math.prod(spin + 1 for spin in equivalent_spins)
    result = np.zeros((2 * N + 1, 2, nuclear_dimension), dtype=complex)
    for n_index, m_n in enumerate(range(-N, N + 1)):
        for spin_index, two_m_s in enumerate((-1, 1)):
            two_m_j = 2 * m_n + two_m_s
            first = clebsch_gordan_doubled(2 * N, 2 * m_n, 1, two_m_s, two_j, two_m_j)
            if first == 0.0:
                continue
            # A scalar Hamiltonian is independent of M_F.  Use the stretched
            # projection, which is valid for integer and half-integer F and
            # avoids the F=0 specialization that previously corrupted every
            # hyperfine block above F=0.
            two_m_f = two_f
            two_m_i = two_m_f - two_m_j
            if abs(two_m_i) > two_i:
                continue
            second = clebsch_gordan_doubled(
                two_j, two_m_j, two_i, two_m_i, two_f, two_m_f
            )
            result[n_index, spin_index] += (
                first * second * _equivalent_nuclear_state(
                    equivalent_spins, two_i, symmetry, two_m_i, multiplicity
                )
            )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=32768)
def _equivalent_scalar_contraction(
    N: int,
    two_ji: int,
    two_ii: int,
    symmetry_i: int,
    two_jj: int,
    two_ij: int,
    symmetry_j: int,
    two_f: int,
    alpha: int,
    sign: int,
    equivalent_spins: tuple[int, ...],
    multiplicity_i: int = 0,
    multiplicity_j: int = 0,
) -> complex:
    left = _equivalent_coupled_state(
        N, two_ji, two_ii, symmetry_i, two_f, equivalent_spins, multiplicity_i
    )
    right = _equivalent_coupled_state(
        N, two_jj, two_ij, symmetry_j, two_f, equivalent_spins, multiplicity_j
    )
    total = 0.0j
    for component in (-1, 0, 1):
        internal = np.kron(
            _angular_momentum_rank1_component(1, component),
            _equivalent_fourier_component(equivalent_spins, alpha, -component, sign),
        )
        transformed = right.reshape((2 * N + 1, -1)) @ internal.T
        total += (-1.0 if component & 1 else 1.0) * np.vdot(
            left.reshape((2 * N + 1, -1)), transformed
        )
    return total


@lru_cache(maxsize=32768)
def _equivalent_rank2_contraction(
    ni: int,
    nj: int,
    two_ji: int,
    two_ii: int,
    symmetry_i: int,
    two_jj: int,
    two_ij: int,
    symmetry_j: int,
    two_f: int,
    alpha: int,
    sign: int,
    equivalent_spins: tuple[int, ...],
    multiplicity_i: int = 0,
    multiplicity_j: int = 0,
) -> complex:
    left = _equivalent_coupled_state(
        ni, two_ji, two_ii, symmetry_i, two_f, equivalent_spins, multiplicity_i
    )
    right = _equivalent_coupled_state(
        nj, two_jj, two_ij, symmetry_j, two_f, equivalent_spins, multiplicity_j
    )
    total = 0.0j
    for projection in range(-2, 3):
        rotational = _rank2_rotational_space_component(ni, nj, projection)
        internal_dimension = 2 * math.prod(spin + 1 for spin in equivalent_spins)
        internal = np.zeros((internal_dimension, internal_dimension), dtype=complex)
        for electron_component in (-1, 0, 1):
            nuclear_component = -projection - electron_component
            if nuclear_component not in (-1, 0, 1):
                continue
            coefficient = clebsch_gordan_doubled(
                2, 2 * electron_component, 2, 2 * nuclear_component,
                4, -2 * projection,
            )
            if coefficient:
                internal += coefficient * np.kron(
                    _angular_momentum_rank1_component(1, electron_component),
                    _equivalent_fourier_component(
                        equivalent_spins, alpha, nuclear_component, sign
                    ),
                )
        transformed = rotational @ right.reshape((2 * nj + 1, -1)) @ internal.T
        total += (-1.0 if projection & 1 else 1.0) * np.vdot(
            left.reshape((2 * ni + 1, -1)), transformed
        )
    return total


def _directional_label(label: Mapping[str, Any], nitot: int) -> dict[str, int]:
    packed = 2 * int(label.get("nuclear_permutation_symmetry", 0))
    if nitot >= 3:
        symmetry = (packed >> 2) % nitot
        iqn = packed - (symmetry << 2)
        partner = nitot - symmetry
        esym = 0 if symmetry == 0 or symmetry == partner else (1 if symmetry < partner else -1)
        if esym:
            iqn += 1
    else:
        symmetry = iqn = 0
    if int(label.get("l_value", 0)) or int(label.get("k_start", 0)):
        iqn |= 1
    return {
        "dimension": int(label["dimension"]),
        "symmetry": int(label["wang_symmetry_code"]),
        "N": int(label["N"]),
        "k_start": int(label["k_start"]),
        "l": int(label.get("l_value", 0)),
        "vibration": int(label["vibration"]),
        "iqn": iqn,
    }


def _ladder(N: int, k: int) -> float:
    return math.sqrt(max(0.0, N * (N + 1.0) - k * (k + 1.0)))


def _dircos(
    bra: Mapping[str, int],
    ket: Mapping[str, int],
    rank: int,
    delta_k: int,
    ifup: int,
    flags: int,
    mask: int,
) -> tuple[tuple[int, int, float], ...]:
    """Three-branch Wang-basis direction cosine construction."""

    kbra0, kket0 = bra["k_start"], ket["k_start"]
    kbra0x, kket0x = bra["iqn"] & 1, ket["iqn"] & 1
    nsbra, nsket = bra["dimension"], ket["dimension"]
    isbra, isket = bra["symmetry"], ket["symmetry"]
    symmetry = (isbra ^ isket) & 3
    if flags & _MODD:
        symmetry = 3 - symmetry
    reverse = symmetry >> 1
    phase_sum = _WANG_PHASE[isket] - _WANG_PHASE[isbra] + (symmetry & 1)
    if (phase_sum & 1) and bra["vibration"] < ket["vibration"]:
        phase_sum += 2
    sign = phase_sum >> 1
    if (flags & _MLZ) and ket["l"] < 0:
        sign += 1
    Nbra, Nket = bra["N"], ket["N"]
    kdel = delta_k
    ikdel = kdel - rank
    kkdel = kdel if ikdel <= 0 else rank
    kdel2, doubled_rank = 2 * kkdel, 2 * rank
    two_nbra, two_nket = 2 * Nbra, 2 * Nket
    if rank > 0:
        element = float(two_nbra + 1)
        if two_nket != two_nbra:
            element = math.sqrt(element * (two_nket + 1))
        if kkdel:
            element *= _fac(rank)
        sign += Nbra + kket0 + kkdel
    else:
        element = 0.5 if kdel else 1.0
    matrix_symmetric = True
    side_mask = mask & 5
    if kdel == 0:
        if side_mask:
            mask = (mask & 2) + 1
        matrix_symmetric = False
    elif side_mask in (1, 4):
        matrix_symmetric = False
    if matrix_symmetric:
        matrix_symmetric = all(
            bra[key] == ket[key]
            for key in ("dimension", "symmetry", "N", "k_start", "l", "vibration")
        )
    if (sign + reverse) & 1:
        element = -element
    output: list[tuple[int, int, float]] = []
    first_branch: list[tuple[int, int, float]] = []
    for branch in (1, 2, 4):
        if mask < branch:
            break
        if branch != 2:
            if not (mask & branch):
                continue
            if branch == 4:
                if matrix_symmetric:
                    break
                kdel, kkdel = -kdel, -kkdel
            total = kdel + kket0 - kbra0
            if total >= 0:
                il, ir = total >> 1, 0
                kbra, kket = kdel + kket0, kket0
            else:
                il, ir = 0, (-total) >> 1
                kbra, kket = kbra0, kbra0 - kdel
            count = min(nsket - ir, nsbra - il)
            if count <= 0:
                continue
            begin = len(output)
            kket2 = 2 * kket
            for offset in range(count):
                value = element
                if rank:
                    value *= wigner_3j_doubled(
                        two_nbra, doubled_rank, two_nket,
                        -kdel2 - kket2, kdel2, kket2,
                    )
                    kket2 += 4
                output.append((il + offset, ir + offset, value))
            left_zero = kbra0x if kbra == 0 else 1
            right_zero = kket0x if kket == 0 else 1
            if (left_zero + right_zero) & 1:
                row, column, value = output[begin]
                output[begin] = row, column, value * _fac(0)
            if ikdel > 0:
                kbgn = kbra - ikdel if kdel > 0 else kbra
                for index in range(begin, len(output)):
                    row, column, value = output[index]
                    for kval in range(kbgn, kbgn + ikdel):
                        value *= _ladder(Nbra, kval)
                    output[index] = row, column, value
                    kbgn += 2
            first_branch = output[begin:]
        else:
            kdel2 = -kdel2
            if reverse & 1:
                element = -element
            if not (mask & 2):
                continue
            total = kdel - kbra0 - kket0
            if total < 0:
                continue
            total >>= 1
            ir = max(0, total - nsbra + 1)
            if total < nsbra and kket0x == 0:
                ir += 1
            count = total + 1
            if count > nsket:
                count = nsket
            elif kbra0x == 0:
                count -= 1
            count -= ir
            if count <= 0:
                continue
            il = total - ir
            if ifup == 0:
                count = (il - ir + 2) >> 1
            kket = kket0 + 2 * ir
            kbra = kdel - kket
            branch_element = -element if ((isbra + two_nbra) & 2) else element
            sine_fourier = bool(flags & _MFCM)
            for _ in range(count):
                value = branch_element
                if sine_fourier and kbra == kket:
                    il, ir = il - 1, ir + 1
                    kbra, kket = kbra - 2, kket + 2
                    continue
                if sine_fourier and kbra > kket:
                    value = -value
                if rank:
                    kket2 = 2 * kket
                    value *= wigner_3j_doubled(
                        two_nbra, doubled_rank, two_nket,
                        -kdel2 - kket2, kdel2, kket2,
                    )
                for offset in range(max(0, ikdel)):
                    kval = kbra - offset
                    value *= _ladder(Nbra, -kval) if kval <= 0 else _ladder(Nbra, kval - 1)
                output.append((il, ir, value))
                il, ir = il - 1, ir + 1
                kbra, kket = kbra - 2, kket + 2
    if matrix_symmetric and first_branch and ifup:
        output.extend((column, row, value) for row, column, value in first_branch)
    return tuple(output)


def _fac(rank: int) -> float:
    values = [0.5]
    for order in range(1, rank + 2):
        values.append(values[-1] * math.sqrt(order / (order - 0.5)))
    values[0] = 1.0 / values[1]
    return values[rank]


__all__ = [
    "CALPGMDirectionalRoute",
    "directional_route",
    "directional_subblock_elements",
    "electron_spin_tensor_factor",
    "equivalent_spin_tensor_factor",
    "rotational_reduced_factor",
]
