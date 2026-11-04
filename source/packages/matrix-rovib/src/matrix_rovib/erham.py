"""Effective rotational Hamiltonians for one or two periodic LAMs.

This module implements the Fourier expansion introduced by Groner for ERHAM.
The internal-motion dependence is projected from each rho-axis onto the
principal-axis symmetric-top basis.  It is therefore an effective-Hamiltonian
backend complementary to LOCK's explicit DVR backends; no molecule name or
fixed molecular-symmetry group enters the operator construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .native_rotor import (
    NativeRotorParameters,
    angular_momentum_operators,
    centrifugal_distortion_operators,
    diagonalize_rotor,
)


ERHAM_MODEL_SCHEMA = "matrix.lock.erham-effective-model.v1"
ERHAM_RESULT_SCHEMA = "matrix.lock.erham-effective-result.v1"


@dataclass(frozen=True)
class ERHAMPeriodicMotion:
    """Orientation and periodicity of one rho axis in the principal-axis frame."""

    rho: float
    beta_rad: float
    alpha_rad: float = 0.0
    fold: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.rho) or self.rho < 0.0:
            raise ValueError("rho must be finite and nonnegative")
        if not all(math.isfinite(value) for value in (self.beta_rad, self.alpha_rad)):
            raise ValueError("rho-axis angles must be finite")
        if int(self.fold) != self.fold or self.fold < 2:
            raise ValueError("the periodic-motion fold must be an integer >= 2")


@dataclass(frozen=True)
class ERHAMFourierTerm:
    """One real Fourier coefficient and its symmetry-related harmonics.

    Each item in ``harmonics`` is a vector of integer Fourier orders, one per
    periodic motion.  ERHAM's factor of two for a nonzero real Fourier pair is
    supplied by ``multiplicity``.  ``operator`` names a rotational primitive;
    ``identity`` is the pure tunnelling-energy term.
    """

    name: str
    coefficient_MHz: float
    harmonics: tuple[tuple[int, ...], ...]
    operator: str = "identity"
    multiplicity: float = 2.0
    operator_frame: str = "principal"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.harmonics:
            raise ValueError("an ERHAM term needs a name and at least one harmonic")
        widths = {len(item) for item in self.harmonics}
        if len(widths) != 1 or any(not any(item) for item in self.harmonics):
            raise ValueError("Fourier harmonics must be nonzero vectors of equal length")
        if not math.isfinite(self.coefficient_MHz) or not math.isfinite(self.multiplicity):
            raise ValueError("ERHAM coefficients and multiplicities must be finite")
        if self.operator_frame not in {"rho", "principal", "effective-phase"}:
            raise ValueError(
                "the ERHAM operator frame must be rho, principal or effective-phase"
            )


@dataclass(frozen=True)
class ERHAMKPRTerm:
    """One coefficient of ERHAM's general :math:`B_{kprq}` expansion.

    ``kappa`` is the change in signed K, ``j_power`` and ``k_power`` are the
    powers p and r in Groner's notation, and ``harmonic`` is q.  The same
    primitive represents ordinary high-order centrifugal-distortion terms
    when q is zero and torsion-rotation terms when q is nonzero.
    """

    name: str
    coefficient_MHz: float
    harmonic: tuple[int, ...]
    kappa: int
    j_power: int
    k_power: int
    trig_parity: int = 1
    imaginary: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not math.isfinite(self.coefficient_MHz):
            raise ValueError("an ERHAM B_kprq term needs a name and finite coefficient")
        if self.kappa < 0 or self.j_power < 0 or self.k_power < 0:
            raise ValueError("ERHAM kappa, p and r must be nonnegative")
        if self.j_power % 2:
            raise ValueError("ERHAM p must be even because it is a power of J(J+1)")
        if self.trig_parity not in {-1, 1}:
            raise ValueError("ERHAM trigonometric parity must be +1 or -1")


@dataclass(frozen=True)
class ERHAMModel:
    """General Watson plus ERHAM Fourier effective Hamiltonian."""

    motions: tuple[ERHAMPeriodicMotion, ...]
    species: Mapping[str, tuple[int, ...]]
    rotational_MHz: tuple[float, float, float]
    quartic_MHz: tuple[float, ...] = (0.0,) * 5
    sextic_MHz: tuple[float, ...] = (0.0,) * 7
    octic_MHz: tuple[float, ...] = (0.0,) * 9
    fourier_terms: tuple[ERHAMFourierTerm, ...] = ()
    kpr_terms: tuple[ERHAMKPRTerm, ...] = ()
    equivalent_motions: bool = False
    projection_sign: int = -1
    symmetry_code: int = 0
    reduction: str = "A"
    representation: str = "Ir"

    def __post_init__(self) -> None:
        count = len(self.motions)
        if count not in {1, 2}:
            raise ValueError("ERHAM supports one or two periodic large-amplitude motions")
        if len(self.rotational_MHz) != 3 or any(value <= 0 for value in self.rotational_MHz):
            raise ValueError("three positive rotational constants are required")
        if (len(self.quartic_MHz), len(self.sextic_MHz), len(self.octic_MHz)) != (5, 7, 9):
            raise ValueError("Watson constants require 5 quartic, 7 sextic and 9 octic values")
        if self.reduction.upper() not in {"A", "S"}:
            raise ValueError("Watson reduction must be A or S")
        if not self.species:
            raise ValueError("at least one torsional species is required")
        for label, sigmas in self.species.items():
            if not label or len(sigmas) != count:
                raise ValueError("each species must provide one sigma label per motion")
            for sigma, motion in zip(sigmas, self.motions, strict=True):
                if int(sigma) != sigma or abs(int(sigma)) >= motion.fold:
                    raise ValueError("torsional sigma labels must be integers within the fold")
        if any(len(harmonic) != count for term in self.fourier_terms for harmonic in term.harmonics):
            raise ValueError("Fourier-term dimensionality must match the periodic motions")
        if any(len(term.harmonic) != count for term in self.kpr_terms):
            raise ValueError("B_kprq dimensionality must match the periodic motions")
        if self.equivalent_motions and count != 2:
            raise ValueError("equivalent-motion symmetrization requires two motions")
        if self.projection_sign not in {-1, 1}:
            raise ValueError("the relative projection sign must be +1 or -1")
        if self.symmetry_code not in range(-5, 5):
            raise ValueError("the ERHAM symmetry code must lie between -5 and 4")
        names = [term.name for term in (*self.fourier_terms, *self.kpr_terms)]
        if len(names) != len(set(names)) or any(name in _WATSON_NAMES for name in names):
            raise ValueError("ERHAM coefficient names must be unique and not shadow Watson terms")


@dataclass(frozen=True)
class ERHAMBlock:
    J: int
    species: str
    energies_MHz: np.ndarray
    vectors: np.ndarray
    assignments: tuple[tuple[int, int, str], ...]
    derivatives: Mapping[str, np.ndarray]


def _diagonalize_hermitian_components(
    hamiltonian: np.ndarray, *, relative_threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Diagonalize exact/discrete symmetry components independently."""
    size = hamiltonian.shape[0]
    scale = max(1.0, float(np.max(np.abs(hamiltonian))))
    if relative_threshold is None:
        relative_threshold = 64.0 * np.finfo(float).eps
    adjacency = np.abs(hamiltonian) > relative_threshold * scale
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, True)
    count, labels = connected_components(csr_matrix(adjacency), directed=False)
    components = tuple(tuple(np.flatnonzero(labels == block)) for block in range(count))
    if count == 1:
        energies, vectors = np.linalg.eigh(hamiltonian)
        return energies.real, vectors, components
    energies = np.empty(size, dtype=float)
    vectors = np.zeros((size, size), dtype=complex)
    offset = 0
    for component in components:
        indices = np.asarray(component, dtype=int)
        values, local = np.linalg.eigh(hamiltonian[np.ix_(indices, indices)])
        width = len(indices)
        energies[offset:offset + width] = values.real
        vectors[np.ix_(indices, np.arange(offset, offset + width))] = local
        offset += width
    order = np.argsort(energies, kind="stable")
    return energies[order], vectors[:, order], components


def _erham_symmetry_basis(
    J: int, jsym: int
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return the discrete Wang/GETMASK basis associated with ERHAM JSYM."""
    size = 2 * J + 1
    columns: list[np.ndarray] = []
    labels: list[tuple[int, ...]] = []
    if jsym in {1, 2, 4}:
        reversal_signs = (1, -1)
        k_parities = (0, 1) if jsym == 4 else (-1,)
        for reversal_sign in reversal_signs:
            for k_parity in k_parities:
                group: list[np.ndarray] = []
                if reversal_sign == 1 and k_parity in {-1, 0}:
                    vector = np.zeros(size, dtype=complex)
                    vector[J] = 1.0
                    group.append(vector)
                for magnitude in range(1, J + 1):
                    if k_parity >= 0 and magnitude % 2 != k_parity:
                        continue
                    phase = (-1 if magnitude % 2 else 1) if jsym == 2 else 1
                    vector = np.zeros(size, dtype=complex)
                    vector[J - magnitude] = 1.0 / math.sqrt(2.0)
                    vector[J + magnitude] = reversal_sign * phase / math.sqrt(2.0)
                    group.append(vector)
                if group:
                    start = len(columns)
                    columns.extend(group)
                    labels.append(tuple(range(start, len(columns))))
    elif jsym == 3:
        for parity in (0, 1):
            group = []
            for K in range(-J, J + 1):
                if K % 2 == parity:
                    vector = np.zeros(size, dtype=complex)
                    vector[K + J] = 1.0
                    group.append(vector)
            start = len(columns)
            columns.extend(group)
            labels.append(tuple(range(start, len(columns))))
    else:
        return np.eye(size, dtype=complex), (tuple(range(size)),)
    return np.column_stack(columns), tuple(labels)


def _diagonalize_erham_symmetry(
    hamiltonian: np.ndarray, J: int, jsym: int
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Diagonalize directly in ERHAM's discrete GETMASK symmetry sectors."""
    basis, sectors = _erham_symmetry_basis(J, jsym)
    transformed = _hermitian(basis.conj().T @ hamiltonian @ basis)
    scale = max(1.0, float(np.max(np.abs(transformed))))
    mask = np.zeros_like(transformed, dtype=bool)
    for sector in sectors:
        indices = np.asarray(sector, dtype=int)
        mask[np.ix_(indices, indices)] = True
    if float(np.max(np.abs(transformed[~mask]), initial=0.0)) > 1.0e-11 * scale:
        return _diagonalize_hermitian_components(hamiltonian)
    energies = np.empty(2 * J + 1, dtype=float)
    vectors = np.zeros_like(transformed)
    offset = 0
    physical_sectors: list[tuple[int, ...]] = []
    for sector in sectors:
        indices = np.asarray(sector, dtype=int)
        values, local = np.linalg.eigh(transformed[np.ix_(indices, indices)])
        width = len(indices)
        energies[offset:offset + width] = values.real
        vectors[np.ix_(indices, np.arange(offset, offset + width))] = local
        physical_sectors.append(tuple(range(offset, offset + width)))
        offset += width
    vectors = basis @ vectors
    order = np.argsort(energies, kind="stable")
    return energies[order], vectors[:, order], tuple(physical_sectors)


@lru_cache(maxsize=512)
def erham_rotational_operators(
    J: int, reduction: str = "A", representation: str = "Ir"
) -> Mapping[str, np.ndarray]:
    """Return the conventional ERHAM/Watson operator registry for one J block."""
    Ja, Jb, Jc = angular_momentum_operators(J, representation)
    J2 = Ja @ Ja + Jb @ Jb + Jc @ Jc
    distortion = centrifugal_distortion_operators(J, reduction, representation)
    names = (
        "DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK",
        "PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK",
        "LJ", "LJJK", "LJK", "LJKK", "LK", "lJ", "lJJK", "lJKK", "lK",
    )
    # Conventional Watson signs alternate with centrifugal order.  The native
    # primitives carry the tensor signs, including the off-diagonal factors.
    signed = tuple(-item for item in distortion[:5]) + distortion[5:12] + tuple(
        -item for item in distortion[12:]
    )
    registry = {
        "A": Ja @ Ja,
        "B": Jb @ Jb,
        "C": Jc @ Jc,
        "AminusMeanBC": Ja @ Ja,
        "MeanBC": J2,
        "QuarterBmC": 2.0 * (Jb @ Jb - Jc @ Jc),
        "identity": np.eye(2 * J + 1, dtype=complex),
    }
    registry.update(zip(names, signed, strict=True))
    result = {name: _hermitian(value) for name, value in registry.items()}
    for value in result.values():
        value.setflags(write=False)
    return MappingProxyType(result)


@lru_cache(maxsize=4096)
def _rho_axis_spectral_basis(
    J: int, beta_rad: float, alpha_rad: float, representation: str
) -> tuple[np.ndarray, np.ndarray]:
    Ja, Jb, Jc = angular_momentum_operators(J, representation)
    axis = (
        math.cos(beta_rad) * Ja
        + math.sin(beta_rad) * math.cos(alpha_rad) * Jb
        + math.sin(beta_rad) * math.sin(alpha_rad) * Jc
    )
    values, vectors = np.linalg.eigh(_hermitian(axis))
    values.setflags(write=False)
    vectors.setflags(write=False)
    return values, vectors


def erham_fourier_modulation(
    J: int,
    motions: Sequence[ERHAMPeriodicMotion],
    sigmas: Sequence[int],
    harmonic: Sequence[int],
    representation: str = "Ir",
) -> np.ndarray:
    """Project one cosine Fourier factor into the principal-axis K basis.

    For two non-collinear rho axes the two possible projection orders are
    averaged, which is the matrix form of Groner's symmetrized projection
    operator Y(K1,K2).
    """
    if not (len(motions) == len(sigmas) == len(harmonic)):
        raise ValueError("motion, sigma and harmonic vectors must have equal length")
    exponentials: list[np.ndarray] = []
    for motion, sigma, q in zip(motions, sigmas, harmonic, strict=True):
        values, vectors = _rho_axis_spectral_basis(
            J, motion.beta_rad, motion.alpha_rad, representation
        )
        phase = 2.0 * math.pi * int(q) * (int(sigma) - motion.rho * values) / motion.fold
        exponentials.append((vectors * np.exp(1.0j * phase)[None, :]) @ vectors.conj().T)
    forward = np.eye(2 * J + 1, dtype=complex)
    backward = forward.copy()
    for factor in exponentials:
        forward = forward @ factor
    for factor in reversed(exponentials):
        backward = backward @ factor
    return _hermitian(0.5 * (forward + backward)).real.astype(complex)


def _erham_fourier_modulation_derivative(
    J: int,
    motions: Sequence[ERHAMPeriodicMotion],
    sigmas: Sequence[int],
    harmonic: Sequence[int],
    representation: str,
    motion_index: int,
    field: str,
) -> np.ndarray:
    """Analytic derivative of the symmetrized multi-axis Fourier factor."""
    Ja, Jb, Jc = angular_momentum_operators(J, representation)
    size = 2 * J + 1
    factors: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    for index, (motion, sigma, order) in enumerate(
        zip(motions, sigmas, harmonic, strict=True)
    ):
        beta, alpha = motion.beta_rad, motion.alpha_rad
        axis = (
            math.cos(beta) * Ja
            + math.sin(beta) * math.cos(alpha) * Jb
            + math.sin(beta) * math.sin(alpha) * Jc
        )
        daxis = np.zeros_like(axis)
        if index == motion_index and field == "beta_rad":
            daxis = (
                -math.sin(beta) * Ja
                + math.cos(beta) * math.cos(alpha) * Jb
                + math.cos(beta) * math.sin(alpha) * Jc
            )
        elif index == motion_index and field == "alpha_rad":
            daxis = math.sin(beta) * (
                -math.sin(alpha) * Jb + math.cos(alpha) * Jc
            )
        scale = 2.0 * math.pi * int(order) / motion.fold
        generator = -1.0j * scale * motion.rho * axis
        dgenerator = np.zeros_like(generator)
        if index == motion_index:
            if field == "rho":
                dgenerator = -1.0j * scale * axis
            elif field in {"beta_rad", "alpha_rad"}:
                dgenerator = -1.0j * scale * motion.rho * daxis
        scalar = np.exp(1.0j * scale * int(sigma))
        factor, derivative = expm_frechet(generator, dgenerator, compute_expm=True)
        factors.append(scalar * factor)
        derivatives.append(scalar * derivative)

    def product_derivative(order: Sequence[int]) -> np.ndarray:
        result = np.zeros((size, size), dtype=complex)
        for active in range(len(order)):
            product = np.eye(size, dtype=complex)
            for position, index in enumerate(order):
                product = product @ (derivatives[index] if position == active else factors[index])
            result += product
        return result

    forward = tuple(range(len(factors)))
    backward = tuple(reversed(forward))
    derivative = 0.5 * (product_derivative(forward) + product_derivative(backward))
    return _hermitian(derivative).real.astype(complex)


def erham_operator_matrices(
    model: ERHAMModel, J: int, species: str
) -> Mapping[str, np.ndarray]:
    """Build coefficient derivative matrices for a species/J block."""
    try:
        sigmas = model.species[species]
    except KeyError as error:
        raise ValueError(f"unknown ERHAM species {species!r}") from error
    rotational = erham_rotational_operators(J, model.reduction, model.representation)
    matrices: dict[str, np.ndarray] = {
        name: rotational[name] for name in (
            "A", "B", "C", "DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK",
            "PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK",
            "LJ", "LJJK", "LJK", "LJKK", "LK", "lJ", "lJJK", "lJKK", "lK",
        )
    }
    for term in model.fourier_terms:
        if term.operator == "identity" and len(model.motions) == 2:
            matrices[term.name] = erham_energy_fourier_matrix(
                model, J, sigmas, term
            )
            continue
        matrix = np.zeros((2 * J + 1, 2 * J + 1), dtype=complex)
        for harmonic in term.harmonics:
            operator = rotational[term.operator]
            if term.operator_frame == "effective-phase" and term.operator != "identity":
                K = np.arange(-J, J + 1, dtype=float)
                Ksum = K[:, None] + K[None, :]
                phase = np.zeros_like(Ksum)
                for motion, sigma, order in zip(
                    model.motions, sigmas, harmonic, strict=True
                ):
                    axis_sign = 1.0 if math.cos(motion.beta_rad) >= 0.0 else -1.0
                    phase += (
                        math.pi * int(order)
                        * (2.0 * int(sigma) - axis_sign * motion.rho * Ksum)
                        / motion.fold
                    )
                matrix += term.multiplicity * np.cos(phase) * operator
                continue
            modulation = erham_fourier_modulation(
                J, model.motions, sigmas, harmonic, model.representation
            )
            active = [index for index, order in enumerate(harmonic) if order]
            if term.operator_frame == "rho" and len(active) == 1 and term.operator != "identity":
                operator = _rho_axis_rotational_operators(
                    J, model.motions[active[0]], model.representation
                )[term.operator]
            matrix += 0.5 * term.multiplicity * (
                modulation @ operator + operator @ modulation
            )
        matrices[term.name] = _hermitian(matrix)
    for term in model.kpr_terms:
        matrices[term.name] = erham_kpr_matrix(model, J, sigmas, term)
    return matrices


@lru_cache(maxsize=4096)
def _erham_wigner_d(J: int, beta_rad: float, representation: str) -> np.ndarray:
    """Stable full Wigner d matrix in the signed-K ordering used by ERHAM."""
    _Ja, _Jb, Jc = angular_momentum_operators(J, representation)
    matrix = np.real_if_close(expm(-1.0j * beta_rad * Jc), tol=1000).astype(complex)
    matrix.setflags(write=False)
    return matrix


def _erham_motion_parity(model: ERHAMModel, J: int) -> np.ndarray:
    """Return BMAT's molecular-symmetry phase F(K)."""
    first = (2, 2, 3, 1, 1, 3, 3, 3, 1, 2)
    second = (0, 0, 4, 1, 1, 4, 4, 1, 3, 0)
    table = first if model.projection_sign == -1 else second
    mcode = table[model.symmetry_code + 5]
    K = np.arange(-J, J + 1)
    return np.where(K % 2 == 0, 1.0, -1.0) if mcode % 2 else np.ones(2 * J + 1)


def erham_species_class(model: ERHAMModel, species: str) -> int:
    """Return ERHAM's ISZ class (1..4) from physical torsional labels."""
    sigmas = model.species[species]
    if len(sigmas) == 1:
        return 1 if sigmas[0] == 0 else 2
    first, second = sigmas
    if first == second:
        return 1 if first == 0 else 2
    if (first + second) % model.motions[0].fold == 0:
        return 3
    return 4


def erham_jsym(model: ERHAMModel, species: str) -> int:
    """Reproduce ERHAM's MSYM/NC symmetry selector for one torsional species."""
    table = (
        (3, 2, 4, 4, 2, 3, 0, 3, 1, 0),
        (3, 2, 1, 3, 0, 0, 0, 3, 0, 0),
        (0, 0, 2, 1, 0, 0, 0, 0, 1, 0),
    )
    species_class = erham_species_class(model, species)
    if species_class == 4:
        return 0
    jsym = table[species_class - 1][model.symmetry_code + 5]
    if (
        model.projection_sign == -1
        and jsym not in {0, 4}
        and 2 <= abs(model.symmetry_code) <= 3
        and model.symmetry_code != -3
    ):
        jsym = 4 - jsym
    return jsym


def _erham_exchange_phase(
    model: ERHAMModel, kappa: int, trig_parity: int, imaginary: bool
) -> int:
    phase = trig_parity * (1 if kappa % 2 == 0 else -1)
    if abs(model.symmetry_code) == 4:
        phase = trig_parity
    if model.symmetry_code == 2 and imaginary and model.projection_sign == -1:
        phase = -phase
    if model.symmetry_code == 3 and imaginary and model.projection_sign == 1:
        phase = -phase
    return phase


def erham_energy_fourier_matrix(
    model: ERHAMModel,
    J: int,
    sigmas: Sequence[int],
    term: ERHAMFourierTerm,
) -> np.ndarray:
    """Project a pure tunnelling coefficient exactly as INDMAT/BMAT/HAMILT.

    This is the two-axis Wigner projection used by ERHAM.  It differs from a
    product of one-axis exponentials when the rho axes are non-collinear and
    retains the discrete molecular-symmetry phase required by EA-like blocks.
    """
    if len(model.motions) != 2 or len(sigmas) != 2:
        raise ValueError("the ERHAM two-axis energy projection needs two motions")
    size = 2 * J + 1
    K = np.arange(-J, J + 1, dtype=int)
    first, second = model.motions
    d1 = _erham_wigner_d(J, first.beta_rad, model.representation)
    d2 = _erham_wigner_d(J, second.beta_rad, model.representation)
    parity = _erham_motion_parity(model, J)
    bmat = 0.5 * (d1 @ np.diag(parity) @ d2.T)
    phi1 = (
        2 * int(sigmas[0]) - first.rho * 2 * K.astype(float)
    ) * math.pi / first.fold
    phi2 = (
        2 * int(sigmas[1]) - second.rho * 2 * K.astype(float)
    ) * math.pi / second.fold
    energy = np.zeros((size, size), dtype=float)
    for harmonic in term.harmonics:
        q1, q2 = (int(value) for value in harmonic)
        direct = q1 * phi1[:, None] + q2 * phi2[None, :]
        trig = np.cos(direct)
        if model.equivalent_motions:
            paired = min(1, (q1 + q2) * (q1 - q2)) == 1
            if paired:
                swapped = q2 * phi1[:, None] + q1 * phi2[None, :]
                trig = trig + np.cos(swapped)
        energy += 2.0 * trig
    if model.projection_sign == -1:
        energy = energy[:, ::-1]
    core = bmat * energy
    left_phase = np.exp(1.0j * first.alpha_rad * K)
    right_phase = parity * np.exp(-1.0j * second.alpha_rad * K)
    projected = (
        left_phase[:, None]
        * (d1.T @ core @ d2)
        * right_phase[None, :]
    )
    return _hermitian(projected + projected.conj().T)


def _energy_fourier_motion_derivative(
    model: ERHAMModel,
    J: int,
    sigmas: Sequence[int],
    term: ERHAMFourierTerm,
    motion_index: int,
    field: str,
) -> np.ndarray:
    """Analytic derivative of the INDMAT/BMAT pure-tunnelling projection."""
    size = 2 * J + 1
    K = np.arange(-J, J + 1, dtype=int)
    first, second = model.motions
    d1 = _erham_wigner_d(J, first.beta_rad, model.representation)
    d2 = _erham_wigner_d(J, second.beta_rad, model.representation)
    dd1 = np.zeros_like(d1)
    dd2 = np.zeros_like(d2)
    if field == "beta_rad":
        _Ja, _Jb, Jc = angular_momentum_operators(J, model.representation)
        if motion_index == 0:
            dd1 = (-1.0j * Jc) @ d1
        else:
            dd2 = (-1.0j * Jc) @ d2
    parity = _erham_motion_parity(model, J)
    bmat = 0.5 * (d1 @ np.diag(parity) @ d2.T)
    dbmat = 0.5 * (
        dd1 @ np.diag(parity) @ d2.T + d1 @ np.diag(parity) @ dd2.T
    )
    phi1 = (
        2 * int(sigmas[0]) - first.rho * 2 * K.astype(float)
    ) * math.pi / first.fold
    phi2 = (
        2 * int(sigmas[1]) - second.rho * 2 * K.astype(float)
    ) * math.pi / second.fold
    dphi1 = (
        -2.0 * K.astype(float) * math.pi / first.fold
        if field == "rho" and motion_index == 0
        else np.zeros(size)
    )
    dphi2 = (
        -2.0 * K.astype(float) * math.pi / second.fold
        if field == "rho" and motion_index == 1
        else np.zeros(size)
    )
    energy = np.zeros((size, size), dtype=float)
    denergy = np.zeros_like(energy)
    for harmonic in term.harmonics:
        q1, q2 = (int(value) for value in harmonic)
        paired = min(1, (q1 + q2) * (q1 - q2)) == 1
        direct = q1 * phi1[:, None] + q2 * phi2[None, :]
        value = np.cos(direct)
        derivative = -np.sin(direct) * (
            q1 * dphi1[:, None] + q2 * dphi2[None, :]
        )
        if model.equivalent_motions and paired:
            swapped = q2 * phi1[:, None] + q1 * phi2[None, :]
            value = value + np.cos(swapped)
            derivative = derivative - np.sin(swapped) * (
                q2 * dphi1[:, None] + q1 * dphi2[None, :]
            )
        energy += 2.0 * value
        denergy += 2.0 * derivative
    if model.projection_sign == -1:
        energy = energy[:, ::-1]
        denergy = denergy[:, ::-1]
    core = bmat * energy
    dcore = dbmat * energy + bmat * denergy
    left_phase = np.exp(1.0j * first.alpha_rad * K)
    right_phase = parity * np.exp(-1.0j * second.alpha_rad * K)
    dleft = np.zeros(size, dtype=complex)
    dright = np.zeros(size, dtype=complex)
    if field == "alpha_rad" and motion_index == 0:
        dleft = 1.0j * K * left_phase
    if field == "alpha_rad" and motion_index == 1:
        dright = -1.0j * K * right_phase
    middle = d1.T @ core @ d2
    dmiddle = dd1.T @ core @ d2 + d1.T @ dcore @ d2 + d1.T @ core @ dd2
    derivative = (
        dleft[:, None] * middle * right_phase[None, :]
        + left_phase[:, None] * dmiddle * right_phase[None, :]
        + left_phase[:, None] * middle * dright[None, :]
    )
    return _hermitian(derivative + derivative.conj().T)


def _kpr_rho_derivative(
    model: ERHAMModel,
    J: int,
    sigmas: Sequence[int],
    term: ERHAMKPRTerm,
    motion_index: int,
) -> np.ndarray:
    size = 2 * J + 1
    matrix = np.zeros((size, size), dtype=complex)
    jfactor = float(J * (J + 1)) ** (term.j_power // 2)
    projections = (1, model.projection_sign) if len(model.motions) == 2 else (1,)
    for K in range(-J, J - term.kappa + 1):
        Kq, ksum = K + term.kappa, 2 * K + term.kappa
        phases = [
            (2 * int(sigma) - motion.rho * projection * ksum) * math.pi / motion.fold
            for motion, sigma, projection in zip(model.motions, sigmas, projections, strict=True)
        ]
        dphase = -projections[motion_index] * ksum * math.pi / model.motions[motion_index].fold
        q = term.harmonic
        direct = sum(int(order) * phase for order, phase in zip(q, phases, strict=True))
        ddirect = int(q[motion_index]) * dphase
        if term.trig_parity == 1:
            trig_derivative = -2.0 * math.sin(direct) * ddirect
        else:
            trig_derivative = 2.0 * math.cos(direct) * ddirect
        if model.equivalent_motions:
            q1, q2 = (int(value) for value in q)
            paired = min(1, (q1 + q2) * (q1 - q2)) == 1
            if paired:
                swapped = q2 * phases[0] + q1 * phases[1]
                dswapped = (q2 if motion_index == 0 else q1) * dphase
                exchange = _erham_exchange_phase(
                    model, term.kappa, term.trig_parity, term.imaginary
                )
                trig_derivative += 2.0 * exchange * (
                    -math.sin(swapped) * dswapped
                    if term.trig_parity == 1 else math.cos(swapped) * dswapped
                )
        polynomial = jfactor * (K ** term.k_power + Kq ** term.k_power)
        value = 0.5 * polynomial * _raising_product(J, K, term.kappa) * trig_derivative
        row, column = Kq + J, K + J
        element = (1.0j if term.imaginary else 1.0) * value
        matrix[row, column] += element
        if row != column:
            matrix[column, row] += element.conjugate()
    return _hermitian(matrix)


def erham_hamiltonian_motion_derivative(
    model: ERHAMModel,
    J: int,
    species: str,
    motion_index: int,
    field: str,
) -> np.ndarray:
    """Analytic Hamiltonian derivative for rho-axis magnitude and angles."""
    if field not in {"rho", "beta_rad", "alpha_rad"}:
        raise ValueError("ERHAM motion derivatives support rho, beta_rad and alpha_rad")
    if not 0 <= motion_index < len(model.motions):
        raise IndexError("ERHAM motion index is outside the model")
    sigmas = model.species[species]
    derivative = np.zeros((2 * J + 1, 2 * J + 1), dtype=complex)
    rotational = erham_rotational_operators(J, model.reduction, model.representation)
    for term in model.fourier_terms:
        if term.operator == "identity" and len(model.motions) == 2:
            derivative += term.coefficient_MHz * _energy_fourier_motion_derivative(
                model, J, sigmas, term, motion_index, field
            )
            continue
        term_derivative = np.zeros_like(derivative)
        for harmonic in term.harmonics:
            operator = rotational[term.operator]
            if term.operator_frame == "effective-phase" and term.operator != "identity":
                if field != "rho":
                    continue
                K = np.arange(-J, J + 1, dtype=float)
                Ksum = K[:, None] + K[None, :]
                phase = np.zeros_like(Ksum)
                dphase = np.zeros_like(Ksum)
                for index, (motion, sigma, order) in enumerate(
                    zip(model.motions, sigmas, harmonic, strict=True)
                ):
                    axis_sign = 1.0 if math.cos(motion.beta_rad) >= 0.0 else -1.0
                    phase += (
                        math.pi * int(order)
                        * (2.0 * int(sigma) - axis_sign * motion.rho * Ksum)
                        / motion.fold
                    )
                    if index == motion_index:
                        dphase = (
                            -math.pi * int(order) * axis_sign * Ksum / motion.fold
                        )
                term_derivative += (
                    -term.multiplicity * np.sin(phase) * dphase * operator
                )
                continue
            modulation = erham_fourier_modulation(
                J, model.motions, sigmas, harmonic, model.representation
            )
            dmodulation = _erham_fourier_modulation_derivative(
                J, model.motions, sigmas, harmonic, model.representation,
                motion_index, field,
            )
            doperator = np.zeros_like(operator)
            active = [index for index, order in enumerate(harmonic) if order]
            if term.operator_frame == "rho" and len(active) == 1 and term.operator != "identity":
                operator = _rho_axis_rotational_operators(
                    J, model.motions[active[0]], model.representation
                )[term.operator]
                if active[0] == motion_index and field in {"beta_rad", "alpha_rad"}:
                    doperator = _rho_axis_rotational_operator_derivative(
                        J, model.motions[active[0]], model.representation, field
                    )[term.operator]
            term_derivative += 0.5 * term.multiplicity * (
                dmodulation @ operator + operator @ dmodulation
                + modulation @ doperator + doperator @ modulation
            )
        derivative += term.coefficient_MHz * _hermitian(term_derivative)
    if field == "rho":
        for term in model.kpr_terms:
            derivative += term.coefficient_MHz * _kpr_rho_derivative(
                model, J, sigmas, term, motion_index
            )
    return _hermitian(derivative)


def _raising_product(J: int, K: int, kappa: int) -> float:
    result = 1.0
    jj = float(J * (J + 1))
    for step in range(kappa):
        value = jj - (K + step) * (K + step + 1)
        if value <= 0.0:
            return 0.0
        result *= math.sqrt(value)
    return result


def _erham_trig(
    model: ERHAMModel,
    sigmas: Sequence[int],
    harmonic: Sequence[int],
    ksum: int,
    kappa: int,
    trig_parity: int,
    imaginary: bool = False,
) -> float:
    """Translate ERHAM V16g-R3 TRIG for a Hamiltonian matrix element."""
    if not any(harmonic):
        return 1.0
    phases = [
        (2 * int(sigma) - motion.rho * projection * ksum)
        * math.pi / motion.fold
        for motion, sigma, projection in zip(
            model.motions,
            sigmas,
            (1, model.projection_sign) if len(model.motions) == 2 else (1,),
            strict=True,
        )
    ]
    direct = sum(int(q) * phase for q, phase in zip(harmonic, phases, strict=True))
    if not model.equivalent_motions:
        return 2.0 * (math.cos(direct) if trig_parity == 1 else math.sin(direct))
    swapped = int(harmonic[1]) * phases[0] + int(harmonic[0]) * phases[1]
    exchange_phase = _erham_exchange_phase(
        model, kappa, trig_parity, imaginary
    )
    q1, q2 = (int(value) for value in harmonic)
    paired = min(1, (q1 + q2) * (q1 - q2)) == 1
    if trig_parity == 1:
        value = math.cos(direct)
        return 2.0 * (value + exchange_phase * math.cos(swapped) if paired else value)
    value = math.sin(direct)
    return 2.0 * (value + exchange_phase * math.sin(swapped) if paired else value)


def erham_kpr_matrix(
    model: ERHAMModel,
    J: int,
    sigmas: Sequence[int],
    term: ERHAMKPRTerm,
) -> np.ndarray:
    """Construct the exact reduced-basis matrix of one B_kprq coefficient."""
    size = 2 * J + 1
    matrix = np.zeros((size, size), dtype=complex)
    jfactor = float(J * (J + 1)) ** (term.j_power // 2)
    K = np.arange(-J, J - term.kappa + 1, dtype=float)
    Kq = K + term.kappa
    polynomial = jfactor * (K ** term.k_power + Kq ** term.k_power)
    raising = np.ones_like(K)
    jj = float(J * (J + 1))
    for step in range(term.kappa):
        raising *= np.sqrt(np.maximum(0.0, jj - (K + step) * (K + step + 1.0)))
    ksum = K + Kq
    projections = (1, model.projection_sign) if len(model.motions) == 2 else (1,)
    phases = [
        (2 * int(sigma) - motion.rho * projection * ksum)
        * math.pi / motion.fold
        for motion, sigma, projection in zip(
            model.motions, sigmas, projections, strict=True
        )
    ]
    direct = sum(
        int(order) * phase
        for order, phase in zip(term.harmonic, phases, strict=True)
    )
    trig = np.cos(direct) if term.trig_parity == 1 else np.sin(direct)
    if model.equivalent_motions:
        q1, q2 = (int(value) for value in term.harmonic)
        paired = min(1, (q1 + q2) * (q1 - q2)) == 1
        if paired:
            swapped = q2 * phases[0] + q1 * phases[1]
            exchange = _erham_exchange_phase(
                model, term.kappa, term.trig_parity, term.imaginary
            )
            trig = trig + exchange * (
                np.cos(swapped) if term.trig_parity == 1 else np.sin(swapped)
            )
    values = polynomial * raising * trig
    rows = (Kq + J).astype(int)
    columns = (K + J).astype(int)
    elements = (1.0j if term.imaginary else 1.0) * values
    matrix[rows, columns] = elements
    off_diagonal = rows != columns
    matrix[columns[off_diagonal], rows[off_diagonal]] = elements[off_diagonal].conj()
    return _hermitian(matrix)


@lru_cache(maxsize=4096)
def _rho_axis_rotational_operators(
    J: int, motion: ERHAMPeriodicMotion, representation: str
) -> Mapping[str, np.ndarray]:
    """Watson primitives in a rho-axis frame with local y normal to its plane."""
    Ja, Jb, Jc = angular_momentum_operators(J, representation)
    cb, sb = math.cos(motion.beta_rad), math.sin(motion.beta_rad)
    ca, sa = math.cos(motion.alpha_rad), math.sin(motion.alpha_rad)
    radial = ca * Jb + sa * Jc
    transverse = -sa * Jb + ca * Jc
    Jz = cb * Ja + sb * radial
    Jx = -sb * Ja + cb * radial
    Jy = transverse
    J2 = Ja @ Ja + Jb @ Jb + Jc @ Jc
    K2 = Jz @ Jz
    asymmetry = Jy @ Jy - Jx @ Jx
    sym = lambda left, right: 0.5 * (left @ right + right @ left)
    raw = (
        J2 @ J2, J2 @ K2, K2 @ K2, -2.0 * J2 @ asymmetry,
        -2.0 * sym(K2, asymmetry),
        np.linalg.matrix_power(J2, 3), J2 @ J2 @ K2, J2 @ K2 @ K2,
        np.linalg.matrix_power(K2, 3), -2.0 * J2 @ J2 @ asymmetry,
        -2.0 * J2 @ sym(K2, asymmetry), -2.0 * sym(K2 @ K2, asymmetry),
        np.linalg.matrix_power(J2, 4), np.linalg.matrix_power(J2, 3) @ K2,
        J2 @ J2 @ K2 @ K2, J2 @ np.linalg.matrix_power(K2, 3),
        np.linalg.matrix_power(K2, 4), -2.0 * J2 @ J2 @ J2 @ asymmetry,
        -2.0 * J2 @ J2 @ sym(K2, asymmetry),
        -2.0 * J2 @ sym(K2 @ K2, asymmetry),
        -2.0 * sym(np.linalg.matrix_power(K2, 3), asymmetry),
    )
    names = (
        "DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK",
        "PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK",
        "LJ", "LJJK", "LJK", "LJKK", "LK", "lJ", "lJJK", "lJKK", "lK",
    )
    signed = tuple(-item for item in raw[:5]) + raw[5:12] + tuple(-item for item in raw[12:])
    result = {
        "identity": np.eye(2 * J + 1, dtype=complex),
        "AminusMeanBC": K2,
        "MeanBC": J2,
        "QuarterBmC": 2.0 * (Jx @ Jx - Jy @ Jy),
    }
    result.update(zip(names, signed, strict=True))
    frozen = {name: _hermitian(value) for name, value in result.items()}
    for value in frozen.values():
        value.setflags(write=False)
    return MappingProxyType(frozen)


def _rho_axis_rotational_operator_derivative(
    J: int, motion: ERHAMPeriodicMotion, representation: str, field: str
) -> Mapping[str, np.ndarray]:
    """Differentiate rho-frame Watson primitives with respect to axis angles."""
    Ja, Jb, Jc = angular_momentum_operators(J, representation)
    cb, sb = math.cos(motion.beta_rad), math.sin(motion.beta_rad)
    ca, sa = math.cos(motion.alpha_rad), math.sin(motion.alpha_rad)
    radial = ca * Jb + sa * Jc
    transverse = -sa * Jb + ca * Jc
    Jz = cb * Ja + sb * radial
    Jx = -sb * Ja + cb * radial
    Jy = transverse
    dJz = np.zeros_like(Jz)
    dJx = np.zeros_like(Jx)
    dJy = np.zeros_like(Jy)
    if field == "beta_rad":
        dJz = -sb * Ja + cb * radial
        dJx = -cb * Ja - sb * radial
    elif field == "alpha_rad":
        dradial = transverse
        dtransverse = -radial
        dJz = sb * dradial
        dJx = cb * dradial
        dJy = dtransverse

    def power_derivative(matrix: np.ndarray, derivative: np.ndarray, power: int) -> np.ndarray:
        return sum(
            (
                np.linalg.matrix_power(matrix, left)
                @ derivative
                @ np.linalg.matrix_power(matrix, power - left - 1)
                for left in range(power)
            ),
            np.zeros_like(matrix),
        )

    J2 = Ja @ Ja + Jb @ Jb + Jc @ Jc
    K2 = Jz @ Jz
    dK2 = dJz @ Jz + Jz @ dJz
    asymmetry = Jy @ Jy - Jx @ Jx
    dasymmetry = dJy @ Jy + Jy @ dJy - dJx @ Jx - Jx @ dJx
    dsym = lambda left, dleft, right, dright: 0.5 * (
        dleft @ right + left @ dright + dright @ left + right @ dleft
    )
    K4 = K2 @ K2
    dK4 = dK2 @ K2 + K2 @ dK2
    K6 = K4 @ K2
    dK6 = dK4 @ K2 + K4 @ dK2
    raw = (
        np.zeros_like(J2), J2 @ dK2, dK4, -2.0 * J2 @ dasymmetry,
        -2.0 * dsym(K2, dK2, asymmetry, dasymmetry),
        np.zeros_like(J2), J2 @ J2 @ dK2,
        J2 @ (dK2 @ K2 + K2 @ dK2), dK6,
        -2.0 * J2 @ J2 @ dasymmetry,
        -2.0 * J2 @ dsym(K2, dK2, asymmetry, dasymmetry),
        -2.0 * dsym(K4, dK4, asymmetry, dasymmetry),
        np.zeros_like(J2), J2 @ J2 @ J2 @ dK2,
        J2 @ J2 @ dK4, J2 @ dK6,
        power_derivative(K2, dK2, 4),
        -2.0 * J2 @ J2 @ J2 @ dasymmetry,
        -2.0 * J2 @ J2 @ dsym(K2, dK2, asymmetry, dasymmetry),
        -2.0 * J2 @ dsym(K4, dK4, asymmetry, dasymmetry),
        -2.0 * dsym(K6, dK6, asymmetry, dasymmetry),
    )
    names = (
        "DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK",
        "PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK",
        "LJ", "LJJK", "LJK", "LJKK", "LK", "lJ", "lJJK", "lJKK", "lK",
    )
    signed = tuple(-item for item in raw[:5]) + raw[5:12] + tuple(
        -item for item in raw[12:]
    )
    result = {
        "identity": np.zeros_like(J2),
        "AminusMeanBC": dK2,
        "MeanBC": np.zeros_like(J2),
        "QuarterBmC": 2.0 * (dJx @ Jx + Jx @ dJx - dJy @ Jy - Jy @ dJy),
    }
    result.update(zip(names, signed, strict=True))
    return {name: _hermitian(value) for name, value in result.items()}


def solve_erham_block(model: ERHAMModel, J: int, species: str, *, parameter_names: Sequence[str] | None = None) -> ERHAMBlock:
    """Diagonalize one effective block and return analytic linear derivatives."""
    matrices = erham_operator_matrices(model, J, species)
    names = tuple(matrices)
    requested = names if parameter_names is None else tuple(parameter_names)
    if len(set(requested)) != len(requested) or set(requested)-set(names):
        raise ValueError("ERHAM derivative names must be unique registered coefficients")
    coefficients = _model_coefficients(model)
    hamiltonian = sum(
        (coefficients[name] * matrices[name] for name in names),
        np.zeros((2 * J + 1, 2 * J + 1), dtype=complex),
    )
    energies, vectors, _components = _diagonalize_erham_symmetry(
        _hermitian(hamiltonian), J, erham_jsym(model, species)
    )
    reference_block = diagonalize_rotor(
        NativeRotorParameters(
            *model.rotational_MHz,
            quartic_MHz=tuple(-value for value in model.quartic_MHz),
            sextic_MHz=model.sextic_MHz,
            octic_MHz=tuple(-value for value in model.octic_MHz),
            reduction=model.reduction,
            representation=model.representation,
        ),
        J,
    )
    reference_vectors = reference_block.vectors
    assignments = reference_block.assignments
    # The rigid/asymmetric reference eigenvalues are already in IDIAG order.
    # Match the small tunnelling perturbation to that order by maximum overlap.
    old, new = linear_sum_assignment(-(np.abs(reference_vectors.conj().T @ vectors) ** 2))
    if not np.array_equal(old, np.arange(2 * J + 1)):
        raise RuntimeError("ERHAM state matching lost a reference state")
    energies = energies[new]
    vectors = vectors[:, new]
    # ERHAM's IDIAG labelling follows the energy order inside each K_a
    # asymmetry doublet after the tunnelling contribution has been added.
    # An adiabatic rigid-rotor label alone fails when an EA doublet inverts.
    # Keep the overlap-based K_a identity, then reconstruct K_c discretely.
    relabelled = list(assignments)
    for Ka in range(1, J + 1):
        pair = [index for index, label in enumerate(assignments) if label[0] == Ka]
        if len(pair) != 2:
            raise RuntimeError(f"ERHAM K_a={Ka} does not contain one asymmetry doublet")
        lower, upper = sorted(pair, key=lambda index: float(energies[index]))
        for index, Kc in ((lower, J - Ka + 1), (upper, J - Ka)):
            relabelled[index] = (Ka, Kc, "e" if Ka + Kc == J else "o")
    assignments = tuple(relabelled)
    groups = _erham_degenerate_groups(energies)
    derivatives = {
        name: _degenerate_hellmann_feynman(energies, vectors, matrices[name], _groups=groups)
        for name in requested
    }
    return ERHAMBlock(J, species, energies.real, vectors, assignments, derivatives)


def _erham_degenerate_groups(energies):
    """Connected energy components, computed once per eigensystem.

    Sorting and adjacent gaps give exactly the transitive one-dimensional
    connectivity used by the previous pairwise search, including chained gaps.
    """
    values=np.asarray(energies,float)
    scale=max(1.,float(np.max(np.abs(values),initial=0.)))
    tolerance=max(1.e-9,256.*float(np.spacing(scale)))
    order=np.argsort(values,kind='stable')
    cuts=np.r_[0,np.flatnonzero(np.diff(values[order])>tolerance)+1,len(values)]
    return tuple(np.sort(order[a:b]) for a,b in zip(cuts[:-1],cuts[1:],strict=True) if b-a>1)


def _degenerate_hellmann_feynman(
    energies: np.ndarray, vectors: np.ndarray, derivative: np.ndarray, *, _groups=None
) -> np.ndarray:
    """Return branch-resolved derivatives, including exact degeneracies.

    Ordinary diagonal expectation values vanish or depend on the arbitrary
    eigensolver gauge inside a degenerate manifold.  Degenerate perturbation
    theory diagonalizes the projected derivative and then assigns its branches
    back to the current state gauge by maximum overlap.
    """

    values = np.asarray(energies, dtype=float)
    states = np.asarray(vectors, dtype=complex)
    result = np.real(np.einsum("ij,ij->j", states.conj(), derivative @ states))
    for indices in (_erham_degenerate_groups(values) if _groups is None else _groups):
        subspace = states[:, indices]
        slopes, rotations = np.linalg.eigh(
            _hermitian(subspace.conj().T @ derivative @ subspace)
        )
        resolved = subspace @ rotations
        rows, columns = linear_sum_assignment(
            -(np.abs(subspace.conj().T @ resolved) ** 2)
        )
        for row, column in zip(rows, columns, strict=True):
            result[indices[int(row)]] = float(slopes[int(column)])
    return result


def _model_coefficients(model: ERHAMModel) -> dict[str, float]:
    names = (
        "A", "B", "C", "DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK",
        "PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK",
        "LJ", "LJJK", "LJK", "LJKK", "LK", "lJ", "lJJK", "lJKK", "lK",
    )
    values = (
        model.rotational_MHz + model.quartic_MHz + model.sextic_MHz + model.octic_MHz
    )
    result = dict(zip(names, values, strict=True))
    result.update((term.name, term.coefficient_MHz) for term in model.fourier_terms)
    result.update((term.name, term.coefficient_MHz) for term in model.kpr_terms)
    return result


_WATSON_NAMES = frozenset((
    "A", "B", "C", "DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK",
    "PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK",
    "LJ", "LJJK", "LJK", "LJKK", "LK", "lJ", "lJJK", "lJKK", "lK",
))


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


__all__ = [
    "ERHAM_MODEL_SCHEMA", "ERHAM_RESULT_SCHEMA", "ERHAMBlock", "ERHAMFourierTerm",
    "ERHAMKPRTerm",
    "ERHAMModel", "ERHAMPeriodicMotion", "erham_energy_fourier_matrix",
    "erham_fourier_modulation",
    "_diagonalize_erham_symmetry", "_diagonalize_hermitian_components",
    "erham_kpr_matrix", "erham_operator_matrices", "erham_rotational_operators",
    "erham_hamiltonian_motion_derivative",
    "erham_jsym", "erham_species_class",
    "solve_erham_block",
]
