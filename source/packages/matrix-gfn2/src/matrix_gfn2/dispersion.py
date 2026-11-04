"""Self-consistent charge-dependent D4 term used by MATRIX-GFN2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from math import exp, pi
from typing import Sequence

import numpy as np

from ._array_cache import triangular_indices

from .localized_solver import SparseSymmetricMatrix

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover
    _gfn2_native = None


@dataclass(frozen=True)
class D4ReferenceState:
    coordination: float
    charge: float
    polarizabilities: tuple[float, ...]
    gaussian_count: int = 1


@dataclass(frozen=True)
class D4AtomModel:
    effective_nuclear_charge: float
    hardness: float
    r2r4: float
    references: tuple[D4ReferenceState, ...]


@dataclass(frozen=True)
class D4Parameters:
    frequency_weights: tuple[float, ...]
    s6: float = 1.0
    s8: float = 2.7
    s9: float = 5.0
    a1: float = 0.52
    a2: float = 5.0
    s10: float = 0.0
    neutral_atm: bool = False


@dataclass(frozen=True)
class D4Result:
    energy_hartree: float
    charge_derivative_hartree: np.ndarray
    coordination_derivative_hartree: np.ndarray
    c6: np.ndarray
    c8: np.ndarray
    c6_coordination_derivative: np.ndarray


@dataclass(frozen=True)
class D4PreparedGeometry:
    frequency_weights: np.ndarray
    weights: np.ndarray
    weight_derivatives: np.ndarray
    reference_charges: np.ndarray
    references: np.ndarray
    mask: np.ndarray
    effective: np.ndarray
    hardness: np.ndarray
    ratio: np.ndarray
    pair_factor: np.ndarray


@dataclass(frozen=True)
class SparseD4PreparedGeometry:
    """Cell-list D4 geometry with no atom-by-atom dense matrices."""

    frequency_weights: np.ndarray
    weights: np.ndarray
    weight_derivatives: np.ndarray
    reference_charges: np.ndarray
    references: np.ndarray
    mask: np.ndarray
    effective: np.ndarray
    hardness: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    distances: np.ndarray
    ratio: np.ndarray
    pair_factor: np.ndarray
    cutoff_bohr: float
    omitted_pairs: int


@dataclass(frozen=True)
class SparseD4Result:
    energy_hartree: float
    charge_derivative_hartree: np.ndarray
    coordination_derivative_hartree: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    c6: np.ndarray
    c8: np.ndarray
    c6_coordination_left: np.ndarray
    c6_coordination_right: np.ndarray
    omitted_energy_bound_hartree: float


@dataclass(frozen=True)
class SparseATMResult:
    energy_hartree: float
    triples: np.ndarray
    omitted_triples: int
    omitted_energy_bound_hartree: float
    pair_positions: np.ndarray | None = None
    explicit_gradient_hartree_per_bohr: np.ndarray | None = None
    coordination_derivative_hartree: np.ndarray | None = None
    included_triples: int = 0



@dataclass(frozen=True)
class D4ChargeDirection:
    rows: np.ndarray
    columns: np.ndarray
    c6_derivative: np.ndarray
    c8_derivative: np.ndarray
    coordination_potential_derivative: np.ndarray


@dataclass(frozen=True)
class D4CoefficientDirection:
    """Directional derivatives of pair-packed D4 response coefficients."""

    rows: np.ndarray
    columns: np.ndarray
    c6_derivative: np.ndarray
    c8_derivative: np.ndarray
    c6_coordination_left_derivative: np.ndarray
    c6_coordination_right_derivative: np.ndarray
    charge_potential_derivative: np.ndarray
    coordination_potential_derivative: np.ndarray


def _d4_weight_second_derivatives(
    atom_models: Sequence[D4AtomModel], coordination: np.ndarray
) -> np.ndarray:
    """Return normalized D4 reference-weight second CN derivatives."""

    models = tuple(atom_models)
    result = _prepared_d4_weight_second_derivatives(
        models, tuple(float(value) for value in np.asarray(coordination, dtype=float))
    )
    return np.asarray(result)


@lru_cache(maxsize=32)
def _prepared_d4_weight_second_derivatives(
    models: tuple[D4AtomModel, ...], coordination: tuple[float, ...]
) -> np.ndarray:
    """Cache geometry-fixed second CN derivatives across Hessian columns."""

    reference_coordination, gaussian_count, _, _, mask = (
        _packed_d4_reference_data(models)
    )
    cn = np.asarray(coordination, dtype=float)
    if cn.shape != (len(models),):
        raise ValueError("D4 coordination has an inconsistent atom count")
    maximum_gaussians = int(np.max(gaussian_count, initial=0))
    orders = np.arange(1, maximum_gaussians + 1, dtype=float)
    delta = cn[:, None] - reference_coordination
    exponential = np.exp(-6.0 * delta[:, :, None] ** 2 * orders)
    active = orders <= gaussian_count[:, :, None]
    raw = np.sum(np.where(active, exponential, 0.0), axis=2)
    first = np.sum(
        np.where(active, -12.0 * orders * delta[:, :, None] * exponential, 0.0),
        axis=2,
    )
    second = np.sum(
        np.where(
            active,
            (-12.0 * orders + 144.0 * orders**2 * delta[:, :, None] ** 2)
            * exponential,
            0.0,
        ),
        axis=2,
    )
    raw = np.where(mask, raw, 0.0)
    first = np.where(mask, first, 0.0)
    second = np.where(mask, second, 0.0)
    total = np.sum(raw, axis=1)
    remote = total <= np.finfo(float).tiny
    if np.any(remote):
        raw[remote] = 0.0
        nearest = np.argmin(np.where(mask, np.abs(delta), np.inf), axis=1)
        raw[np.flatnonzero(remote), nearest[remote]] = 1.0
        first[remote] = 0.0
        second[remote] = 0.0
        total[remote] = 1.0
    total_first = np.sum(first, axis=1)
    total_second = np.sum(second, axis=1)
    result = (
        second / total[:, None]
        - raw * total_second[:, None] / total[:, None] ** 2
        - 2.0 * first * total_first[:, None] / total[:, None] ** 2
        + 2.0 * raw * total_first[:, None] ** 2 / total[:, None] ** 3
    )
    result.setflags(write=False)
    return result


def d4_coefficient_direction(
    charges: Sequence[float],
    charge_direction: Sequence[float],
    coordination: Sequence[float],
    coordination_direction: Sequence[float],
    atom_models: Sequence[D4AtomModel],
    prepared: D4PreparedGeometry | SparseD4PreparedGeometry,
) -> D4CoefficientDirection:
    """Differentiate D4 pair and CN-response coefficients in one direction."""

    charge = np.asarray(charges, dtype=float)
    qdot = np.asarray(charge_direction, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    cndot = np.asarray(coordination_direction, dtype=float)
    natoms = len(prepared.effective)
    if any(item.shape != (natoms,) for item in (charge, qdot, cn, cndot)):
        raise ValueError("D4 coefficient direction has an inconsistent atom count")
    models = tuple(atom_models)
    if len(models) != natoms:
        raise ValueError("D4 atom models have an inconsistent atom count")
    denominator = prepared.effective + charge
    if np.any(denominator <= 0.0):
        raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
    numerator = prepared.effective[:, None] + prepared.reference_charges
    inner = 2.0 * prepared.hardness[:, None] * (
        1.0 - numerator / denominator[:, None]
    )
    inner_first = (
        2.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 2
    )
    inner_second = (
        -4.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 3
    )
    exponential = np.exp(inner)
    scale = np.exp(3.0 * (1.0 - exponential)) * prepared.mask
    logarithmic_first = -3.0 * exponential * inner_first
    scale_first = scale * logarithmic_first
    scale_second = scale * (
        logarithmic_first**2
        - 3.0 * exponential * (inner_first**2 + inner_second)
    )
    weights = prepared.weights
    weights_first = prepared.weight_derivatives
    weights_second = _d4_weight_second_derivatives(models, cn)
    references = prepared.references
    alpha = np.sum((weights * scale)[:, :, None] * references, axis=1)
    alpha_q = np.sum((weights * scale_first)[:, :, None] * references, axis=1)
    alpha_cn = np.sum((weights_first * scale)[:, :, None] * references, axis=1)
    alpha_dot = alpha_q * qdot[:, None] + alpha_cn * cndot[:, None]
    alpha_q_dot = np.sum(
        (
            weights_first * cndot[:, None] * scale_first
            + weights * scale_second * qdot[:, None]
        )[:, :, None]
        * references,
        axis=1,
    )
    alpha_cn_dot = np.sum(
        (
            weights_second * cndot[:, None] * scale
            + weights_first * scale_first * qdot[:, None]
        )[:, :, None]
        * references,
        axis=1,
    )
    if isinstance(prepared, SparseD4PreparedGeometry):
        rows, columns = prepared.rows, prepared.columns
        ratio = prepared.ratio
        pair_factor = prepared.pair_factor
    else:
        rows, columns = triangular_indices(natoms, diagonal=False)
        ratio = prepared.ratio[rows, columns]
        pair_factor = prepared.pair_factor[rows, columns]
    frequency = prepared.frequency_weights[None, :]
    factor = 3.0 / pi
    dc6 = factor * np.sum(
        alpha_dot[rows] * frequency * alpha[columns]
        + alpha[rows] * frequency * alpha_dot[columns],
        axis=1,
    )
    dcharge_left = factor * np.sum(
        alpha_q_dot[rows] * frequency * alpha[columns]
        + alpha_q[rows] * frequency * alpha_dot[columns],
        axis=1,
    )
    dcharge_right = factor * np.sum(
        alpha_q_dot[columns] * frequency * alpha[rows]
        + alpha_q[columns] * frequency * alpha_dot[rows],
        axis=1,
    )
    dcn_left = factor * np.sum(
        alpha_cn_dot[rows] * frequency * alpha[columns]
        + alpha_cn[rows] * frequency * alpha_dot[columns],
        axis=1,
    )
    dcn_right = factor * np.sum(
        alpha_cn_dot[columns] * frequency * alpha[rows]
        + alpha_cn[columns] * frequency * alpha_dot[rows],
        axis=1,
    )
    coordination_potential = np.zeros(natoms, dtype=float)
    charge_potential = np.zeros(natoms, dtype=float)
    np.add.at(charge_potential, rows, -pair_factor * dcharge_left)
    np.add.at(charge_potential, columns, -pair_factor * dcharge_right)
    np.add.at(coordination_potential, rows, -pair_factor * dcn_left)
    np.add.at(coordination_potential, columns, -pair_factor * dcn_right)
    return D4CoefficientDirection(
        rows,
        columns,
        dc6,
        dc6 * ratio,
        dcn_left,
        dcn_right,
        charge_potential,
        coordination_potential,
    )

def d4_charge_hessian(
    charges: Sequence[float],
    prepared: D4PreparedGeometry | SparseD4PreparedGeometry,
) -> np.ndarray:
    """Return the exact charge--charge Hessian of the two-body D4 energy."""

    charge = np.asarray(charges, dtype=float)
    natoms = len(prepared.effective)
    if charge.shape != (natoms,):
        raise ValueError("D4 charges have an inconsistent atom count")
    denominator = prepared.effective + charge
    if np.any(denominator <= 0.0):
        raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
    numerator = prepared.effective[:, None] + prepared.reference_charges
    first_inner = 2.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 2
    second_inner = -4.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 3
    inner = 2.0 * prepared.hardness[:, None] * (
        1.0 - numerator / denominator[:, None]
    )
    exponential = np.exp(inner)
    scale = np.exp(3.0 * (1.0 - exponential)) * prepared.mask
    logarithmic_first = -3.0 * exponential * first_inner
    logarithmic_second = -3.0 * exponential * (
        first_inner * first_inner + second_inner
    )
    scale_first = scale * logarithmic_first
    scale_second = scale * (
        logarithmic_first * logarithmic_first + logarithmic_second
    )
    alpha = np.sum(
        (prepared.weights * scale)[:, :, None] * prepared.references, axis=1
    )
    alpha_first = np.sum(
        (prepared.weights * scale_first)[:, :, None] * prepared.references, axis=1
    )
    alpha_second = np.sum(
        (prepared.weights * scale_second)[:, :, None] * prepared.references, axis=1
    )
    if isinstance(prepared, SparseD4PreparedGeometry):
        rows, columns = prepared.rows, prepared.columns
        pair_factor = prepared.pair_factor
    else:
        rows, columns = triangular_indices(natoms, diagonal=False)
        pair_factor = prepared.pair_factor[rows, columns]
    frequency = prepared.frequency_weights[None, :]
    factor = 3.0 / pi
    mixed = -pair_factor * factor * np.sum(
        alpha_first[rows] * frequency * alpha_first[columns], axis=1
    )
    diagonal_left = -pair_factor * factor * np.sum(
        alpha_second[rows] * frequency * alpha[columns], axis=1
    )
    diagonal_right = -pair_factor * factor * np.sum(
        alpha_second[columns] * frequency * alpha[rows], axis=1
    )
    hessian = np.zeros((natoms, natoms), dtype=float)
    hessian[rows, columns] = mixed
    hessian[columns, rows] = mixed
    np.add.at(hessian, (rows, rows), diagonal_left)
    np.add.at(hessian, (columns, columns), diagonal_right)
    return hessian


def d4_charge_hessian_matvec(
    charges: Sequence[float],
    direction: Sequence[float],
    prepared: D4PreparedGeometry | SparseD4PreparedGeometry,
) -> np.ndarray:
    """Apply the D4 charge Hessian without constructing an atom-square matrix."""

    charge = np.asarray(charges, dtype=float)
    vector = np.asarray(direction, dtype=float)
    natoms = len(prepared.effective)
    if charge.shape != (natoms,) or vector.shape != (natoms,):
        raise ValueError("D4 charge-Hessian direction has an inconsistent atom count")
    denominator = prepared.effective + charge
    if np.any(denominator <= 0.0):
        raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
    numerator = prepared.effective[:, None] + prepared.reference_charges
    first_inner = 2.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 2
    second_inner = -4.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 3
    inner = 2.0 * prepared.hardness[:, None] * (1.0 - numerator / denominator[:, None])
    exponential = np.exp(inner)
    scale = np.exp(3.0 * (1.0 - exponential)) * prepared.mask
    logarithmic_first = -3.0 * exponential * first_inner
    logarithmic_second = -3.0 * exponential * (
        first_inner * first_inner + second_inner
    )
    alpha = np.sum(
        (prepared.weights * scale)[:, :, None] * prepared.references, axis=1
    )
    alpha_first = np.sum(
        (prepared.weights * scale * logarithmic_first)[:, :, None]
        * prepared.references,
        axis=1,
    )
    alpha_second = np.sum(
        (
            prepared.weights
            * scale
            * (logarithmic_first * logarithmic_first + logarithmic_second)
        )[:, :, None]
        * prepared.references,
        axis=1,
    )
    if isinstance(prepared, SparseD4PreparedGeometry):
        rows, columns = prepared.rows, prepared.columns
        pair_factor = prepared.pair_factor
    else:
        rows, columns = triangular_indices(natoms, diagonal=False)
        pair_factor = prepared.pair_factor[rows, columns]
    frequency = prepared.frequency_weights[None, :]
    factor = 3.0 / pi
    mixed = -pair_factor * factor * np.sum(
        alpha_first[rows] * frequency * alpha_first[columns], axis=1
    )
    diagonal_left = -pair_factor * factor * np.sum(
        alpha_second[rows] * frequency * alpha[columns], axis=1
    )
    diagonal_right = -pair_factor * factor * np.sum(
        alpha_second[columns] * frequency * alpha[rows], axis=1
    )
    result = np.zeros(natoms, dtype=float)
    np.add.at(result, rows, diagonal_left * vector[rows] + mixed * vector[columns])
    np.add.at(
        result,
        columns,
        diagonal_right * vector[columns] + mixed * vector[rows],
    )
    return result



def d4_charge_direction(
    charges: Sequence[float],
    charge_direction: Sequence[float],
    prepared: D4PreparedGeometry | SparseD4PreparedGeometry,
) -> D4ChargeDirection:
    """Differentiate D4 pair coefficients and CN potential along charges."""

    charge = np.asarray(charges, dtype=float)
    direction = np.asarray(charge_direction, dtype=float)
    natoms = len(prepared.effective)
    if charge.shape != (natoms,) or direction.shape != (natoms,):
        raise ValueError("D4 charge direction has an inconsistent atom count")
    denominator = prepared.effective + charge
    numerator = prepared.effective[:, None] + prepared.reference_charges
    inner = 2.0 * prepared.hardness[:, None] * (1.0 - numerator / denominator[:, None])
    exponential = np.exp(inner)
    scale = np.exp(3.0 * (1.0 - exponential)) * prepared.mask
    inner_first = (
        2.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 2
    )
    scale_first = scale * (-3.0 * exponential) * inner_first
    weighted = prepared.weights
    cn_weighted = prepared.weight_derivatives
    references = prepared.references
    alpha = np.sum((weighted * scale)[:, :, None] * references, axis=1)
    alpha_q = np.sum((weighted * scale_first)[:, :, None] * references, axis=1)
    alpha_cn = np.sum((cn_weighted * scale)[:, :, None] * references, axis=1)
    alpha_qcn = np.sum((cn_weighted * scale_first)[:, :, None] * references, axis=1)
    if isinstance(prepared, SparseD4PreparedGeometry):
        rows, columns = prepared.rows, prepared.columns
        ratio = prepared.ratio
        pair_factor = prepared.pair_factor
    else:
        rows, columns = triangular_indices(natoms, diagonal=False)
        ratio = prepared.ratio[rows, columns]
        pair_factor = prepared.pair_factor[rows, columns]
    frequency = prepared.frequency_weights[None, :]
    factor = 3.0 / pi
    dalpha = alpha_q * direction[:, None]
    dc6 = factor * np.sum(
        dalpha[rows] * frequency * alpha[columns]
        + alpha[rows] * frequency * dalpha[columns],
        axis=1,
    )
    dcn_left = factor * np.sum(
        (alpha_qcn[rows] * direction[rows, None]) * frequency * alpha[columns]
        + alpha_cn[rows] * frequency * dalpha[columns],
        axis=1,
    )
    dcn_right = factor * np.sum(
        (alpha_qcn[columns] * direction[columns, None]) * frequency * alpha[rows]
        + alpha_cn[columns] * frequency * dalpha[rows],
        axis=1,
    )
    coordination = np.zeros(natoms, dtype=float)
    np.add.at(coordination, rows, -pair_factor * dcn_left)
    np.add.at(coordination, columns, -pair_factor * dcn_right)
    return D4ChargeDirection(rows, columns, dc6, dc6 * ratio, coordination)


def d4_charge_potential_radial_direction(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    charges: Sequence[float],
    prepared: D4PreparedGeometry | SparseD4PreparedGeometry,
    parameters: D4Parameters,
) -> np.ndarray:
    """Pair-radial part of the fixed-charge D4 potential geometry derivative."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(direction_bohr, dtype=float)
    charge = np.asarray(charges, dtype=float)
    natoms = len(prepared.effective)
    if (coordinates.shape != (natoms, 3) or direction.shape != coordinates.shape
            or charge.shape != (natoms,)):
        raise ValueError("D4 radial charge-potential direction has invalid dimensions")
    if not (np.all(np.isfinite(coordinates)) and np.all(np.isfinite(direction))
            and np.all(np.isfinite(charge))):
        raise ValueError("D4 radial charge-potential direction must be finite")
    denominator = prepared.effective + charge
    if np.any(denominator <= 0.0):
        raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
    numerator = prepared.effective[:, None] + prepared.reference_charges
    inner = 2.0 * prepared.hardness[:, None] * (
        1.0 - numerator / denominator[:, None]
    )
    exponential = np.exp(inner)
    scale = np.exp(3.0 * (1.0 - exponential)) * prepared.mask
    scale_first = scale * (-3.0 * exponential) * (
        2.0 * prepared.hardness[:, None]
        * numerator / denominator[:, None] ** 2
    )
    alpha = np.sum(
        (prepared.weights * scale)[:, :, None] * prepared.references, axis=1
    )
    alpha_q = np.sum(
        (prepared.weights * scale_first)[:, :, None] * prepared.references,
        axis=1,
    )
    if isinstance(prepared, SparseD4PreparedGeometry):
        rows, columns = prepared.rows, prepared.columns
        ratio = prepared.ratio
        distance = prepared.distances
    else:
        rows, columns = triangular_indices(natoms, diagonal=False)
        ratio = prepared.ratio[rows, columns]
        distance = np.linalg.norm(coordinates[rows] - coordinates[columns], axis=1)
    if np.any(distance <= 0.0):
        raise ValueError("D4 pair distance must be positive")
    weighted = prepared.frequency_weights[None, :]
    factor = 3.0 / pi
    charge_left = factor * np.sum(
        alpha_q[rows] * weighted * alpha[columns], axis=1
    )
    charge_right = factor * np.sum(
        alpha_q[columns] * weighted * alpha[rows], axis=1
    )
    displacement = coordinates[rows] - coordinates[columns]
    distance_direction = np.einsum(
        "pk,pk->p", displacement, direction[rows] - direction[columns]
    ) / distance
    cutoff = parameters.a1 * np.sqrt(ratio) + parameters.a2
    radial_factor = distance_direction * (
        6.0 * parameters.s6 * distance**5
        / (distance**6 + cutoff**6) ** 2
        + 8.0 * parameters.s8 * ratio * distance**7
        / (distance**8 + cutoff**8) ** 2
        + 10.0 * parameters.s10 * (49.0 / 40.0) * ratio**2 * distance**9
        / (distance**10 + cutoff**10) ** 2
    )
    potential_direction = np.zeros(natoms, dtype=float)
    np.add.at(potential_direction, rows, charge_left * radial_factor)
    np.add.at(potential_direction, columns, charge_right * radial_factor)
    return potential_direction

def prepare_d4_geometry(
    coordinates_bohr: np.ndarray,
    covalent_coordination: Sequence[float],
    atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters,
) -> D4PreparedGeometry:
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    coordination = np.asarray(covalent_coordination, dtype=float)
    models = tuple(atom_models)
    weights, weight_derivatives, reference_charges, references, mask = (
        _prepared_d4_references(models, tuple(float(value) for value in coordination))
    )
    frequency_weights = np.asarray(parameters.frequency_weights, dtype=float)
    if references.shape[2] != frequency_weights.size:
        raise ValueError("D4 frequency weights do not match reference polarizabilities")
    effective = np.asarray([model.effective_nuclear_charge for model in models])
    hardness = np.asarray([model.hardness for model in models])
    r2r4 = np.asarray([model.r2r4 for model in models], dtype=float)
    ratio = 3.0 * r2r4[:, None] * r2r4[None, :]
    displacement = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    off_diagonal = ~np.eye(len(models), dtype=bool)
    safe_distance = np.where(off_diagonal, distances, 1.0)
    critical = np.sqrt(ratio)
    cutoff = parameters.a1 * critical + parameters.a2
    pair_factor = (
        parameters.s6 / (safe_distance**6 + cutoff**6)
        + parameters.s8 * ratio / (safe_distance**8 + cutoff**8)
        + parameters.s10 * (49.0 / 40.0) * ratio**2 / (safe_distance**10 + cutoff**10)
    )
    pair_factor = np.where(off_diagonal, pair_factor, 0.0)
    return D4PreparedGeometry(
        frequency_weights, weights, weight_derivatives, reference_charges,
        references, mask, effective, hardness, ratio, pair_factor,
    )


def prepare_d4_geometry_sparse(
    coordinates_bohr: np.ndarray,
    covalent_coordination: Sequence[float],
    atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters,
    *,
    cutoff_bohr: float,
) -> SparseD4PreparedGeometry:
    """Prepare only D4 pairs inside a native cell-list cutoff.

    The resulting object is linear in system size for bounded spatial density.
    The evaluator reports a conservative bound for the omitted tail; callers
    must enlarge the cutoff or use a far-field evaluator when that bound is
    above their requested accuracy.
    """

    from .kernels import sparse_pair_geometry

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    coordination = np.asarray(covalent_coordination, dtype=float)
    models = tuple(atom_models)
    natoms = len(models)
    if coordinates.shape != (natoms, 3) or coordination.shape != (natoms,):
        raise ValueError("D4 coordinates and coordination have inconsistent counts")
    if cutoff_bohr <= 0.0:
        raise ValueError("D4 cutoff must be positive")
    pairs = sparse_pair_geometry(coordinates, cutoff_bohr=float(cutoff_bohr))
    rows = pairs.left
    columns = pairs.right
    distances = pairs.distance
    weights, weight_derivatives, reference_charges, references, mask = (
        _prepared_d4_references(models, tuple(float(value) for value in coordination))
    )
    frequency_weights = np.asarray(parameters.frequency_weights, dtype=float)
    if references.shape[2] != frequency_weights.size:
        raise ValueError("D4 frequency weights do not match reference polarizabilities")
    effective = np.asarray([model.effective_nuclear_charge for model in models])
    hardness = np.asarray([model.hardness for model in models])
    r2r4 = np.asarray([model.r2r4 for model in models], dtype=float)
    ratio = 3.0 * r2r4[rows] * r2r4[columns]
    critical = np.sqrt(ratio)
    damping_cutoff = parameters.a1 * critical + parameters.a2
    pair_factor = (
        parameters.s6 / (distances**6 + damping_cutoff**6)
        + parameters.s8 * ratio / (distances**8 + damping_cutoff**8)
        + parameters.s10 * (49.0 / 40.0) * ratio**2 / (distances**10 + damping_cutoff**10)
    )
    return SparseD4PreparedGeometry(
        frequency_weights, weights, weight_derivatives, reference_charges,
        references, mask, effective, hardness, rows, columns, distances,
        ratio, pair_factor, float(cutoff_bohr), pairs.omitted_pairs,
    )


def d4_two_body_sparse(
    charges: Sequence[float],
    prepared: SparseD4PreparedGeometry,
    parameters: D4Parameters,
) -> SparseD4Result:
    """Evaluate pair-packed charge-dependent D4 and its SCC derivatives."""

    charges = np.asarray(charges, dtype=float)
    natoms = len(prepared.effective)
    if charges.shape != (natoms,):
        raise ValueError("D4 charges have an inconsistent atom count")
    denominator = prepared.effective + charges
    if np.any(denominator <= 0.0):
        raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
    if _gfn2_native is not None and hasattr(_gfn2_native, "d4_two_body_pairs"):
        (
            energy,
            charge_derivative,
            coordination_derivative,
            c6,
            c8,
            coordination_left,
            coordination_right,
            maximum_c6,
        ) = _gfn2_native.d4_two_body_pairs(
            charges,
            prepared.frequency_weights,
            prepared.weights,
            prepared.weight_derivatives,
            prepared.reference_charges,
            prepared.references,
            prepared.mask,
            prepared.effective,
            prepared.hardness,
            prepared.rows,
            prepared.columns,
            prepared.ratio,
            prepared.pair_factor,
        )
        maximum_ratio = float(np.max(prepared.ratio, initial=0.0))
        radius = prepared.cutoff_bohr
        tail_factor = (
            abs(parameters.s6) / radius**6
            + abs(parameters.s8) * maximum_ratio / radius**8
            + abs(parameters.s10) * (49.0 / 40.0) * maximum_ratio**2 / radius**10
        )
        tail_bound = prepared.omitted_pairs * float(maximum_c6) * tail_factor
        return SparseD4Result(
            float(energy),
            np.asarray(charge_derivative),
            np.asarray(coordination_derivative),
            prepared.rows,
            prepared.columns,
            np.asarray(c6),
            np.asarray(c8),
            np.asarray(coordination_left),
            np.asarray(coordination_right),
            float(tail_bound),
        )
    numerator = prepared.effective[:, None] + prepared.reference_charges
    inner = 2.0 * prepared.hardness[:, None] * (
        1.0 - numerator / denominator[:, None]
    )
    exponential = np.exp(inner)
    scale = np.exp(3.0 * (1.0 - exponential)) * prepared.mask
    scale_derivative = (
        scale * (-3.0 * exponential)
        * (2.0 * prepared.hardness[:, None] * numerator / denominator[:, None] ** 2)
    )
    alpha = np.sum(
        (prepared.weights * scale)[:, :, None] * prepared.references, axis=1
    )
    charge_alpha = np.sum(
        (prepared.weights * scale_derivative)[:, :, None] * prepared.references,
        axis=1,
    )
    coordination_alpha = np.sum(
        (prepared.weight_derivatives * scale)[:, :, None] * prepared.references,
        axis=1,
    )
    rows, columns = prepared.rows, prepared.columns
    weighted = prepared.frequency_weights[None, :]
    factor = 3.0 / pi
    c6 = factor * np.sum(alpha[rows] * weighted * alpha[columns], axis=1)
    c8 = c6 * prepared.ratio
    charge_left = factor * np.sum(
        charge_alpha[rows] * weighted * alpha[columns], axis=1
    )
    charge_right = factor * np.sum(
        charge_alpha[columns] * weighted * alpha[rows], axis=1
    )
    coordination_left = factor * np.sum(
        coordination_alpha[rows] * weighted * alpha[columns], axis=1
    )
    coordination_right = factor * np.sum(
        coordination_alpha[columns] * weighted * alpha[rows], axis=1
    )
    energy = -float(np.sum(c6 * prepared.pair_factor))
    charge_derivative = np.zeros(natoms, dtype=float)
    coordination_derivative = np.zeros(natoms, dtype=float)
    np.add.at(charge_derivative, rows, -charge_left * prepared.pair_factor)
    np.add.at(charge_derivative, columns, -charge_right * prepared.pair_factor)
    np.add.at(
        coordination_derivative, rows,
        -coordination_left * prepared.pair_factor,
    )
    np.add.at(
        coordination_derivative, columns,
        -coordination_right * prepared.pair_factor,
    )
    # Positive D4 polarizabilities permit a conservative N-pair tail bound.
    # This is deliberately fail-closed: it never labels an uncontrolled
    # cutoff as exact and can later be replaced by the far-field tree bound.
    self_c6 = factor * np.sum(alpha * weighted * alpha, axis=1)
    maximum_c6 = float(np.max(self_c6, initial=0.0))
    maximum_ratio = float(np.max(prepared.ratio, initial=0.0))
    radius = prepared.cutoff_bohr
    tail_factor = (
        abs(parameters.s6) / radius**6
        + abs(parameters.s8) * maximum_ratio / radius**8
        + abs(parameters.s10) * (49.0 / 40.0) * maximum_ratio**2 / radius**10
    )
    tail_bound = prepared.omitted_pairs * maximum_c6 * tail_factor
    return SparseD4Result(
        energy, charge_derivative, coordination_derivative,
        rows, columns, c6, c8, coordination_left, coordination_right,
        float(tail_bound),
    )


def _prepared_d4_references(
    atom_models: tuple[D4AtomModel, ...],
    coordination: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Evaluate geometry-dependent D4 reference weights.

    Coordination numbers vary at every trajectory step, so caching this
    layer forces Python to hash the full model and floating-point CN tuples
    without producing resident cache hits.  The topology-only packed model
    data used below remains cached independently by
    :func:`_packed_d4_reference_data`.
    """
    (
        reference_coordination,
        gaussian_count,
        reference_charges,
        polarizabilities,
        mask,
    ) = _packed_d4_reference_data(atom_models)
    natoms = len(atom_models)
    cn = np.asarray(coordination, dtype=float)
    if cn.shape != (natoms,):
        raise ValueError("D4 coordination has an inconsistent atom count")
    if _gfn2_native is not None and hasattr(_gfn2_native, "d4_reference_weights"):
        weights, weight_derivatives = _gfn2_native.d4_reference_weights(
            cn, reference_coordination, gaussian_count, mask
        )
        return (
            np.asarray(weights),
            np.asarray(weight_derivatives),
            reference_charges,
            polarizabilities,
            mask,
        )
    maximum_gaussians = int(np.max(gaussian_count, initial=0))
    orders = np.arange(1, maximum_gaussians + 1, dtype=float)
    delta = cn[:, None] - reference_coordination
    exponent = np.exp(-6.0 * delta[:, :, None] ** 2 * orders)
    active = orders <= gaussian_count[:, :, None]
    raw = np.sum(np.where(active, exponent, 0.0), axis=2)
    derivative = np.sum(
        np.where(active, -12.0 * orders * delta[:, :, None] * exponent, 0.0),
        axis=2,
    )
    raw = np.where(mask, raw, 0.0)
    derivative = np.where(mask, derivative, 0.0)
    total = np.sum(raw, axis=1)
    remote = total <= np.finfo(float).tiny
    if np.any(remote):
        nearest_distance = np.where(mask, np.abs(delta), np.inf)
        nearest = np.argmin(nearest_distance, axis=1)
        raw[remote] = 0.0
        raw[np.flatnonzero(remote), nearest[remote]] = 1.0
        derivative[remote] = 0.0
        total[remote] = 1.0
    total_derivative = np.sum(derivative, axis=1)
    weights = raw / total[:, None]
    weight_derivatives = (
        derivative * total[:, None] - raw * total_derivative[:, None]
    ) / total[:, None] ** 2
    return weights, weight_derivatives, reference_charges, polarizabilities, mask


@lru_cache(maxsize=32)
def _packed_d4_reference_data(
    atom_models: tuple[D4AtomModel, ...],
) -> tuple[np.ndarray, ...]:
    """Pack immutable universal D4 reference tables once per molecule."""

    natoms = len(atom_models)
    maximum_references = max(len(model.references) for model in atom_models)
    frequencies = len(atom_models[0].references[0].polarizabilities)
    reference_coordination = np.zeros((natoms, maximum_references))
    gaussian_count = np.zeros((natoms, maximum_references), dtype=np.int16)
    reference_charges = np.zeros((natoms, maximum_references))
    polarizabilities = np.zeros((natoms, maximum_references, frequencies))
    mask = np.zeros((natoms, maximum_references), dtype=bool)
    for atom, model in enumerate(atom_models):
        count = len(model.references)
        mask[atom, :count] = True
        for reference, item in enumerate(model.references):
            if len(item.polarizabilities) != frequencies:
                raise ValueError("D4 reference frequency grids must have equal length")
            reference_coordination[atom, reference] = item.coordination
            gaussian_count[atom, reference] = item.gaussian_count
            reference_charges[atom, reference] = item.charge
            polarizabilities[atom, reference] = item.polarizabilities
    for array in (
        reference_coordination,
        gaussian_count,
        reference_charges,
        polarizabilities,
        mask,
    ):
        array.setflags(write=False)
    return (
        reference_coordination,
        gaussian_count,
        reference_charges,
        polarizabilities,
        mask,
    )


def _coordination_weights(coordination: float, references: Sequence[D4ReferenceState]) -> np.ndarray:
    raw = np.asarray(
        [
            sum(
                exp(-6.0 * order * (coordination - reference.coordination) ** 2)
                for order in range(1, reference.gaussian_count + 1)
            )
            for reference in references
        ],
        dtype=float,
    )
    total = float(np.sum(raw))
    if total <= np.finfo(float).tiny:
        # In a remote extrapolation regime the closest reference is the only
        # defensible local model; avoid a zero/zero normalization.
        closest = int(
            np.argmin([abs(coordination - reference.coordination) for reference in references])
        )
        raw[closest] = 1.0
        total = 1.0
    return raw / total


def _coordination_weights_and_derivative(
    coordination: float, references: Sequence[D4ReferenceState]
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(
        [
            sum(
                exp(-6.0 * order * (coordination - reference.coordination) ** 2)
                for order in range(1, reference.gaussian_count + 1)
            )
            for reference in references
        ]
    )
    derivative = np.asarray(
        [
            sum(
                -12.0
                * order
                * (coordination - reference.coordination)
                * exp(-6.0 * order * (coordination - reference.coordination) ** 2)
                for order in range(1, reference.gaussian_count + 1)
            )
            for reference in references
        ]
    )
    total = float(np.sum(raw))
    if total <= np.finfo(float).tiny:
        return _coordination_weights(coordination, references), np.zeros(len(references))
    total_derivative = float(np.sum(derivative))
    return raw / total, (derivative * total - raw * total_derivative) / total**2


def _charge_scale_and_derivative(
    charge: float,
    reference_charge: float,
    model: D4AtomModel,
) -> tuple[float, float]:
    denominator = model.effective_nuclear_charge + charge
    if denominator <= 0.0:
        raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
    numerator = model.effective_nuclear_charge + reference_charge
    inner = 2.0 * model.hardness * (1.0 - numerator / denominator)
    exponential = exp(inner)
    scale = exp(3.0 * (1.0 - exponential))
    derivative = scale * (-3.0 * exponential) * (
        2.0 * model.hardness * numerator / denominator**2
    )
    return scale, derivative


def atomic_polarizability(
    charge: float,
    coordination: float,
    model: D4AtomModel,
) -> tuple[np.ndarray, np.ndarray]:
    weights = _coordination_weights(coordination, model.references)
    size = len(model.references[0].polarizabilities)
    value = np.zeros(size, dtype=float)
    derivative = np.zeros(size, dtype=float)
    for weight, reference in zip(weights, model.references, strict=True):
        if len(reference.polarizabilities) != size:
            raise ValueError("D4 reference frequency grids must have equal length")
        scale, scale_derivative = _charge_scale_and_derivative(
            charge, reference.charge, model
        )
        polarizability = np.asarray(reference.polarizabilities, dtype=float)
        value += weight * scale * polarizability
        derivative += weight * scale_derivative * polarizability
    return value, derivative


def atomic_polarizability_derivatives(
    charge: float, coordination: float, model: D4AtomModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights, weight_derivative = _coordination_weights_and_derivative(
        coordination, model.references
    )
    value = np.zeros(len(model.references[0].polarizabilities))
    charge_derivative = np.zeros_like(value)
    coordination_derivative = np.zeros_like(value)
    for weight, dweight, reference in zip(
        weights, weight_derivative, model.references, strict=True
    ):
        scale, dscale = _charge_scale_and_derivative(charge, reference.charge, model)
        polarizability = np.asarray(reference.polarizabilities, dtype=float)
        value += weight * scale * polarizability
        charge_derivative += weight * dscale * polarizability
        coordination_derivative += dweight * scale * polarizability
    return value, charge_derivative, coordination_derivative


def d4_two_body(
    coordinates_bohr: np.ndarray,
    charges: Sequence[float],
    covalent_coordination: Sequence[float],
    atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters,
    *,
    prepared: D4PreparedGeometry | None = None,
) -> D4Result:
    """Evaluate charge-self-consistent D4 two-body energy and dE/dq."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    charges = np.asarray(charges, dtype=float)
    coordination = np.asarray(covalent_coordination, dtype=float)
    natoms = len(atom_models)
    if coordinates.shape != (natoms, 3) or charges.shape != (natoms,):
        raise ValueError("D4 coordinates and charges have inconsistent atom counts")
    if coordination.shape != (natoms,):
        raise ValueError("D4 coordination must contain one value per atom")
    models = tuple(atom_models)
    state = prepared or prepare_d4_geometry(
        coordinates, coordination, models, parameters
    )
    frequency_weights = state.frequency_weights
    weights, weight_derivatives = state.weights, state.weight_derivatives
    reference_charges, references, mask = (
        state.reference_charges, state.references, state.mask
    )
    if references.shape[2] != frequency_weights.size:
        raise ValueError("D4 frequency weights do not match reference polarizabilities")
    effective, hardness = state.effective, state.hardness
    denominator = effective + charges
    if np.any(denominator <= 0.0):
        raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
    if _gfn2_native is not None and hasattr(_gfn2_native, "d4_two_body_pairs"):
        rows, columns = triangular_indices(natoms, diagonal=False)
        (
            energy,
            charge_derivative,
            coordination_derivative,
            pair_c6,
            pair_c8,
            coordination_left,
            coordination_right,
            _maximum_c6,
        ) = _gfn2_native.d4_two_body_pairs(
            charges,
            state.frequency_weights,
            state.weights,
            state.weight_derivatives,
            state.reference_charges,
            state.references,
            state.mask,
            state.effective,
            state.hardness,
            rows,
            columns,
            state.ratio[rows, columns],
            state.pair_factor[rows, columns],
        )
        c6 = np.zeros((natoms, natoms), dtype=float)
        c8 = np.zeros_like(c6)
        c6[rows, columns] = pair_c6
        c6[columns, rows] = pair_c6
        c8[rows, columns] = pair_c8
        c8[columns, rows] = pair_c8
        c6_coordination_derivative = np.zeros((natoms, natoms, natoms))
        c6_coordination_derivative[rows, columns, rows] = coordination_left
        c6_coordination_derivative[rows, columns, columns] = coordination_right
        c6_coordination_derivative[columns, rows, rows] = coordination_left
        c6_coordination_derivative[columns, rows, columns] = coordination_right
        return D4Result(
            float(energy),
            np.asarray(charge_derivative),
            np.asarray(coordination_derivative),
            c6,
            c8,
            c6_coordination_derivative,
        )
    numerator = effective[:, None] + reference_charges
    inner = 2.0 * hardness[:, None] * (
        1.0 - numerator / denominator[:, None]
    )
    exponential = np.exp(inner)
    scale = np.exp(3.0 * (1.0 - exponential)) * mask
    scale_derivative = (
        scale
        * (-3.0 * exponential)
        * (2.0 * hardness[:, None] * numerator / denominator[:, None] ** 2)
    )
    polarizability_array = np.sum(
        (weights * scale)[:, :, None] * references, axis=1
    )
    derivative_array = np.sum(
        (weights * scale_derivative)[:, :, None] * references, axis=1
    )
    coordination_derivative_array = np.sum(
        (weight_derivatives * scale)[:, :, None] * references, axis=1
    )
    factor = 3.0 / pi
    c6 = factor * (
        (polarizability_array * frequency_weights[None, :])
        @ polarizability_array.T
    )
    np.fill_diagonal(c6, 0.0)
    ratio = state.ratio
    c8 = c6 * ratio
    pair_factor = state.pair_factor
    energy = -0.5 * float(np.sum(c6 * pair_factor))
    charge_c6 = factor * (
        (derivative_array * frequency_weights[None, :])
        @ polarizability_array.T
    )
    coordination_c6 = factor * (
        (coordination_derivative_array * frequency_weights[None, :])
        @ polarizability_array.T
    )
    charge_derivative = -np.sum(charge_c6 * pair_factor, axis=1)
    coordination_derivative = -np.sum(coordination_c6 * pair_factor, axis=1)
    identity = np.eye(natoms)
    c6_coordination_derivative = (
        coordination_c6[:, :, None] * identity[:, None, :]
        + coordination_c6.T[:, :, None] * identity[None, :, :]
    )
    return D4Result(
        energy,
        charge_derivative,
        coordination_derivative,
        c6,
        c8,
        c6_coordination_derivative,
    )


def d4_fock(
    overlap: np.ndarray,
    orbital_atom: Sequence[int],
    charge_derivative_hartree: np.ndarray,
) -> np.ndarray:
    """Map physical-charge derivatives to the AO Fock matrix."""

    overlap = np.asarray(overlap, dtype=float)
    assignment = np.asarray(tuple(orbital_atom), dtype=np.intp)
    derivative = np.asarray(charge_derivative_hartree, dtype=float)
    orbital_derivative = derivative[assignment]
    return -0.5 * overlap * (
        orbital_derivative[:, None] + orbital_derivative[None, :]
    )


def d4_fock_sparse(
    overlap: SparseSymmetricMatrix,
    orbital_atom: Sequence[int],
    charge_derivative_hartree: np.ndarray,
) -> SparseSymmetricMatrix:
    assignment = np.asarray(tuple(orbital_atom), dtype=np.intp)
    derivative = np.asarray(charge_derivative_hartree, dtype=float)[assignment]
    values = -0.5 * overlap.values * (
        derivative[overlap.rows] + derivative[overlap.columns]
    )
    return SparseSymmetricMatrix(
        overlap.size, overlap.rows.copy(), overlap.columns.copy(), values
    )


def atm_three_body_energy(
    coordinates_bohr: np.ndarray,
    c6_reference: np.ndarray,
    c8_reference: np.ndarray,
    *,
    s9: float = 5.0,
    a1: float = 0.52,
    a2: float = 5.0,
) -> float:
    """Evaluate the charge-independent Axilrod-Teller-Muto contribution."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    c6 = np.asarray(c6_reference, dtype=float)
    c8 = np.asarray(c8_reference, dtype=float)
    natoms = len(coordinates)
    if _gfn2_native is not None and hasattr(_gfn2_native, "atm_three_body_energy"):
        return float(
            _gfn2_native.atm_three_body_energy(coordinates, c6, c8, s9, a1, a2)
        )
    triples = np.asarray(tuple(combinations(range(natoms), 3)), dtype=np.intp)
    if not triples.size:
        return 0.0
    right, middle, left = triples.T
    rab = coordinates[left] - coordinates[middle]
    rac = coordinates[left] - coordinates[right]
    rbc = coordinates[middle] - coordinates[right]
    dab = np.linalg.norm(rab, axis=1)
    dac = np.linalg.norm(rac, axis=1)
    dbc = np.linalg.norm(rbc, axis=1)
    cos_left = np.sum(rab * rac, axis=1) / (dab * dac)
    cos_middle = np.sum((-rab) * rbc, axis=1) / (dab * dbc)
    cos_right = np.sum(rac * rbc, axis=1) / (dac * dbc)
    angular = 3.0 * cos_left * cos_middle * cos_right + 1.0
    c9 = np.sqrt(c6[left, middle] * c6[left, right] * c6[middle, right])
    critical = (
        (a1 * np.sqrt(c8[left, middle] / c6[left, middle]) + a2)
        * (a1 * np.sqrt(c8[left, right] / c6[left, right]) + a2)
        * (a1 * np.sqrt(c8[middle, right] / c6[middle, right]) + a2)
    ) ** (1.0 / 3.0)
    product_distance = dab * dac * dbc
    mean_distance = product_distance ** (1.0 / 3.0)
    damping = 1.0 / (1.0 + 6.0 * (critical / mean_distance) ** 16)
    return float(np.sum(s9 * angular * c9 * damping / product_distance**3))


def atm_three_body_energy_charge_derivative_dense(
    coordinates_bohr: np.ndarray,
    charges: np.ndarray,
    coordination: np.ndarray,
    atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters,
    d4: D4Result,
) -> tuple[float, np.ndarray]:
    """Charge-dependent molecular ATM energy and its SCC potential.

    This dense development route uses the same charge-dependent C6 values as
    the pair term. The BJ C8/C6 ratio is charge independent, so differentiating
    the geometric-mean C9 is sufficient for the charge potential.
    """
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    charges = np.asarray(charges, dtype=float)
    coordination = np.asarray(coordination, dtype=float)
    natoms = len(atom_models)
    if coordinates.shape != (natoms, 3) or charges.shape != (natoms,) or coordination.shape != (natoms,):
        raise ValueError("ATM coordinates, charges and coordination have inconsistent sizes")
    derivative = np.zeros(natoms, dtype=float)
    if natoms < 3:
        return 0.0, derivative
    if parameters.neutral_atm:
        neutral = d4_two_body(coordinates, np.zeros(natoms), coordination,
                              atom_models, parameters)
        return atm_three_body_energy(coordinates, neutral.c6, neutral.c8,
            s9=parameters.s9, a1=parameters.a1, a2=parameters.a2), derivative
    frequency_weights = np.asarray(parameters.frequency_weights, dtype=float)
    alpha = np.stack([
        atomic_polarizability_derivatives(float(charges[index]), float(coordination[index]), model)
        for index, model in enumerate(atom_models)
    ])
    c6_charge_first = (3.0 / pi) * (alpha[:, 1, :] * frequency_weights[None, :]) @ alpha[:, 0, :].T
    c6 = np.asarray(d4.c6, dtype=float)
    c8 = np.asarray(d4.c8, dtype=float)
    energy = atm_three_body_energy(
        coordinates, c6, c8,
        s9=parameters.s9, a1=parameters.a1, a2=parameters.a2,
    )
    triples = np.asarray(tuple(combinations(range(natoms), 3)), dtype=np.intp)
    i, j, k = triples.T
    ri, rj, rk = coordinates[i], coordinates[j], coordinates[k]
    rij = np.linalg.norm(ri - rj, axis=1)
    rik = np.linalg.norm(ri - rk, axis=1)
    rjk = np.linalg.norm(rj - rk, axis=1)
    cos_i = np.einsum("ij,ij->i", rj - ri, rk - ri) / (rij * rik)
    cos_j = np.einsum("ij,ij->i", ri - rj, rk - rj) / (rij * rjk)
    cos_k = np.einsum("ij,ij->i", ri - rk, rj - rk) / (rik * rjk)
    angular = 1.0 + 3.0 * cos_i * cos_j * cos_k
    critical = ((parameters.a1 * np.sqrt(c8[i, j] / c6[i, j]) + parameters.a2)
                * (parameters.a1 * np.sqrt(c8[i, k] / c6[i, k]) + parameters.a2)
                * (parameters.a1 * np.sqrt(c8[j, k] / c6[j, k]) + parameters.a2))
    rproduct = rij * rik * rjk
    damping = 1.0 / (1.0 + 6.0 * (critical / rproduct) ** (16.0 / 3.0))
    contribution = (parameters.s9 * angular * np.sqrt(c6[i, j] * c6[i, k] * c6[j, k])
                    * damping / rproduct**3)
    np.add.at(derivative, i, 0.5 * contribution * (
        c6_charge_first[i, j] / c6[i, j] + c6_charge_first[i, k] / c6[i, k]
    ))
    np.add.at(derivative, j, 0.5 * contribution * (
        c6_charge_first[j, i] / c6[i, j] + c6_charge_first[j, k] / c6[j, k]
    ))
    np.add.at(derivative, k, 0.5 * contribution * (
        c6_charge_first[k, i] / c6[i, k] + c6_charge_first[k, j] / c6[j, k]
    ))
    return energy, derivative


def d4_charge_potential_geometry_direction_dense(
    coordinates_bohr: np.ndarray,
    direction_bohr: np.ndarray,
    charges: np.ndarray,
    coordination: np.ndarray,
    coordination_direction: np.ndarray,
    atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters,
) -> np.ndarray:
    """Exact fixed-charge geometry derivative of pair + ATM SCC potential.

    The charge and CN-dependent polarizabilities are differentiated once;
    the same pair damping and ATM energy expression as the resident D4 route
    are then differentiated directionally. Charge relaxation is excluded.
    """
    xyz = np.asarray(coordinates_bohr, dtype=float)
    dxyz = np.asarray(direction_bohr, dtype=float)
    charge = np.asarray(charges, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    dcn = np.asarray(coordination_direction, dtype=float)
    natoms = len(atom_models)
    if (xyz.shape != (natoms, 3) or dxyz.shape != xyz.shape
            or any(value.shape != (natoms,) for value in (charge, cn, dcn))):
        raise ValueError("D4 potential geometry direction dimensions are inconsistent")
    if not all(np.all(np.isfinite(value)) for value in (xyz, dxyz, charge, cn, dcn)):
        raise ValueError("D4 potential geometry direction must be finite")
    if natoms < 2:
        return np.zeros(natoms)
    alpha = []
    alpha_q = []
    alpha_r = []
    alpha_qr = []
    for index, model in enumerate(atom_models):
        value, charge_first, cn_first = atomic_polarizability_derivatives(
            float(charge[index]), float(cn[index]), model,
        )
        _, weight_first = _coordination_weights_and_derivative(
            float(cn[index]), model.references,
        )
        mixed = np.zeros_like(value)
        for weight_first_ref, reference in zip(
            weight_first, model.references, strict=True,
        ):
            _, charge_scale_first = _charge_scale_and_derivative(
                float(charge[index]), reference.charge, model,
            )
            mixed += (weight_first_ref * charge_scale_first
                      * np.asarray(reference.polarizabilities, dtype=float))
        alpha.append(value)
        alpha_q.append(charge_first)
        alpha_r.append(cn_first * dcn[index])
        alpha_qr.append(mixed * dcn[index])
    a = np.stack(alpha)
    aq = np.stack(alpha_q)
    ar = np.stack(alpha_r)
    aqr = np.stack(alpha_qr)
    freq = np.asarray(parameters.frequency_weights, dtype=float)
    factor = 3.0 / pi
    c6 = factor * (a * freq[None, :]) @ a.T
    c6_r = factor * ((ar * freq[None, :]) @ a.T + (a * freq[None, :]) @ ar.T)
    c6_q = factor * (aq * freq[None, :]) @ a.T
    c6_qr = factor * ((aqr * freq[None, :]) @ a.T + (aq * freq[None, :]) @ ar.T)
    r2r4 = np.asarray([model.r2r4 for model in atom_models], dtype=float)
    ratio = 3.0 * r2r4[:, None] * r2r4[None, :]
    cutoff = parameters.a1 * np.sqrt(ratio) + parameters.a2
    potential_r = np.zeros(natoms)
    for i in range(natoms):
        for j in range(i):
            rij = xyz[i] - xyz[j]
            radius = float(np.linalg.norm(rij))
            radius_r = float(np.dot(rij, dxyz[i] - dxyz[j]) / radius)
            c = cutoff[i, j]
            g6 = radius**6 + c**6
            g8 = radius**8 + c**8
            g10 = radius**10 + c**10
            c10_ratio = (49.0 / 40.0) * ratio[i, j]**2
            pair_factor = (parameters.s6 / g6 + parameters.s8 * ratio[i, j] / g8
                           + parameters.s10 * c10_ratio / g10)
            pair_factor_r = -radius_r * (
                6.0 * parameters.s6 * radius**5 / g6**2
                + 8.0 * parameters.s8 * ratio[i, j] * radius**7 / g8**2
                + 10.0 * parameters.s10 * c10_ratio * radius**9 / g10**2
            )
            potential_r[i] -= pair_factor_r * c6_q[i, j] + pair_factor * c6_qr[i, j]
            potential_r[j] -= pair_factor_r * c6_q[j, i] + pair_factor * c6_qr[j, i]
    if parameters.neutral_atm:
        return potential_r
    for i, j, k in combinations(range(natoms), 3):
        edges = ((i, j), (i, k), (j, k))
        distances = []
        distance_rates = []
        for left, right in edges:
            vector = xyz[left] - xyz[right]
            radius = float(np.linalg.norm(vector))
            distances.append(radius)
            distance_rates.append(float(np.dot(
                vector, dxyz[left] - dxyz[right],
            ) / radius))
        rij, rik, rjk = distances
        radial_log_r = sum(rate / distance for rate, distance in zip(
            distance_rates, distances, strict=True,
        ))

        def cosine(vertex, other, third):
            u = xyz[other] - xyz[vertex]
            v = xyz[third] - xyz[vertex]
            du = dxyz[other] - dxyz[vertex]
            dv = dxyz[third] - dxyz[vertex]
            uu = float(np.dot(u, u))
            vv = float(np.dot(v, v))
            c = float(np.dot(u, v) / np.sqrt(uu * vv))
            dc = (float(np.dot(du, v) + np.dot(u, dv)) / np.sqrt(uu * vv)
                  - c * (float(np.dot(u, du)) / uu + float(np.dot(v, dv)) / vv))
            return c, dc

        ci, dci = cosine(i, j, k)
        cj, dcj = cosine(j, i, k)
        ck, dck = cosine(k, i, j)
        angular = 1.0 + 3.0 * ci * cj * ck
        angular_r = 3.0 * (dci * cj * ck + ci * dcj * ck + ci * cj * dck)
        critical = cutoff[i, j] * cutoff[i, k] * cutoff[j, k]
        product = rij * rik * rjk
        damping = 1.0 / (1.0 + 6.0 * (critical / product) ** (16.0 / 3.0))
        c9 = np.sqrt(c6[i, j] * c6[i, k] * c6[j, k])
        c9_log_r = 0.5 * sum(c6_r[left, right] / c6[left, right]
                             for left, right in edges)
        scalar = parameters.s9 * c9 * damping / product**3
        scalar_r = scalar * (
            c9_log_r + ((16.0 / 3.0) * (1.0 - damping) - 3.0) * radial_log_r
        )
        contribution = scalar * angular
        contribution_r = scalar_r * angular + scalar * angular_r
        for atom, neighbors in ((i, (j, k)), (j, (i, k)), (k, (i, j))):
            charge_ratio = sum(c6_q[atom, other] / c6[atom, other]
                               for other in neighbors)
            charge_ratio_r = sum(
                c6_qr[atom, other] / c6[atom, other]
                - c6_q[atom, other] * c6_r[atom, other] / c6[atom, other]**2
                for other in neighbors
            )
            potential_r[atom] += 0.5 * (
                contribution_r * charge_ratio
                + contribution * charge_ratio_r
            )
    return potential_r


def atm_three_body_charge_hessian_dense(
    coordinates_bohr: np.ndarray,
    charges: np.ndarray,
    coordination: np.ndarray,
    atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters,
    d4: D4Result,
) -> np.ndarray:
    """Analytic charge--charge Hessian of molecular charge-dependent ATM."""
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    q = np.asarray(charges, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    natoms = len(atom_models)
    if coordinates.shape != (natoms, 3) or q.shape != (natoms,) or cn.shape != (natoms,):
        raise ValueError("ATM charge Hessian state has inconsistent dimensions")
    hessian = np.zeros((natoms, natoms), dtype=float)
    if natoms < 3 or parameters.neutral_atm:
        return hessian
    alpha = np.empty((natoms, 3, len(parameters.frequency_weights)))
    for a, model in enumerate(atom_models):
        weights = _coordination_weights(float(cn[a]), model.references)
        alpha[a] = 0.0
        for weight, reference in zip(weights, model.references, strict=True):
            denominator = model.effective_nuclear_charge + q[a]
            numerator = model.effective_nuclear_charge + reference.charge
            if denominator <= 0.0:
                raise ValueError("D4 effective nuclear charge plus partial charge must be positive")
            inner = 2.0 * model.hardness * (1.0 - numerator / denominator)
            exponential = np.exp(inner)
            scale = np.exp(3.0 * (1.0 - exponential))
            inner_first = 2.0 * model.hardness * numerator / denominator**2
            inner_second = -4.0 * model.hardness * numerator / denominator**3
            log_first = -3.0 * exponential * inner_first
            log_second = -3.0 * exponential * (inner_first**2 + inner_second)
            reference_alpha = np.asarray(reference.polarizabilities, dtype=float)
            alpha[a, 0] += weight * scale * reference_alpha
            alpha[a, 1] += weight * scale * log_first * reference_alpha
            alpha[a, 2] += weight * scale * (log_first**2 + log_second) * reference_alpha
    frequency = np.asarray(parameters.frequency_weights, dtype=float)
    c6 = np.asarray(d4.c6, dtype=float)
    c8 = np.asarray(d4.c8, dtype=float)
    factor = 3.0 / pi
    for i, j, k in combinations(range(natoms), 3):
        atoms = (i, j, k)
        ri, rj, rk = coordinates[list(atoms)]
        rij = np.linalg.norm(ri - rj)
        rik = np.linalg.norm(ri - rk)
        rjk = np.linalg.norm(rj - rk)
        cos_i = np.dot(rj - ri, rk - ri) / (rij * rik)
        cos_j = np.dot(ri - rj, rk - rj) / (rij * rjk)
        cos_k = np.dot(ri - rk, rj - rk) / (rik * rjk)
        angular = 1.0 + 3.0 * cos_i * cos_j * cos_k
        critical = np.prod([
            parameters.a1 * np.sqrt(c8[a, b] / c6[a, b]) + parameters.a2
            for a, b in ((i, j), (i, k), (j, k))
        ])
        rproduct = rij * rik * rjk
        damping = 1.0 / (1.0 + 6.0 * (critical / rproduct)**(16.0 / 3.0))
        energy = (
            parameters.s9 * angular * np.sqrt(c6[i, j] * c6[i, k] * c6[j, k])
            * damping / rproduct**3
        )
        log_first = np.zeros(3)
        log_second = np.zeros((3, 3))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            a, b = atoms[left], atoms[right]
            c = c6[a, b]
            ua = factor * np.sum(alpha[a, 1] * frequency * alpha[b, 0]) / c
            ub = factor * np.sum(alpha[a, 0] * frequency * alpha[b, 1]) / c
            vaa = factor * np.sum(alpha[a, 2] * frequency * alpha[b, 0]) / c - ua**2
            vbb = factor * np.sum(alpha[a, 0] * frequency * alpha[b, 2]) / c - ub**2
            vab = factor * np.sum(alpha[a, 1] * frequency * alpha[b, 1]) / c - ua * ub
            log_first[left] += 0.5 * ua
            log_first[right] += 0.5 * ub
            log_second[left, left] += 0.5 * vaa
            log_second[right, right] += 0.5 * vbb
            log_second[left, right] += 0.5 * vab
            log_second[right, left] += 0.5 * vab
        block = energy * (np.outer(log_first, log_first) + log_second)
        hessian[np.ix_(atoms, atoms)] += block
    return hessian


def _neutral_pair_reference(positions, prepared, parameters):
    """Reuse the prepared native D4 kernel at the fixed neutral ATM charges.

    All element/CN data are already in ``prepared``; the unused atom-model
    placeholders only supply its atom count to the existing pair kernel.
    """
    n = len(positions)
    return d4_two_body(positions, np.zeros(n), np.zeros(n), (None,) * n,
                       parameters, prepared=prepared)


def d4_global_parameter_partials_dense(
    coordinates_bohr: np.ndarray,
    d4: D4Result,
    parameters: D4Parameters,
    prepared: D4PreparedGeometry,
) -> dict[str, float]:
    """Analytic frozen-charge derivatives of pair D4 plus molecular ATM."""
    positions = np.asarray(coordinates_bohr, dtype=float)
    natoms = len(positions)
    if positions.shape != (natoms, 3) or d4.c6.shape != (natoms, natoms):
        raise ValueError("D4 global partials require a dense molecular state")
    partials = {name: 0.0 for name in ("a1", "a2", "s6", "s8", "s9", "s10")}
    for i in range(natoms):
        for j in range(i + 1, natoms):
            r = float(np.linalg.norm(positions[i] - positions[j]))
            ratio = float(prepared.ratio[i, j])
            root = np.sqrt(ratio)
            damping_radius = parameters.a1 * root + parameters.a2
            c6 = float(d4.c6[i, j])
            derivative_radius = 0.0
            for order, scale, factor in ((6, parameters.s6, 1.), (8, parameters.s8, ratio),
                                          (10, parameters.s10, 49. / 40. * ratio**2)):
                denominator = r**order + damping_radius**order
                partials[f"s{order}"] -= c6 * factor / denominator
                derivative_radius += c6 * scale * factor * order * damping_radius**(order-1) / denominator**2
            partials["a1"] += derivative_radius * root
            partials["a2"] += derivative_radius
    if parameters.neutral_atm:
        d4 = _neutral_pair_reference(positions, prepared, parameters)
    for i, j, k in combinations(range(natoms), 3):
        ri, rj, rk = positions[[i, j, k]]
        rij = np.linalg.norm(ri - rj)
        rik = np.linalg.norm(ri - rk)
        rjk = np.linalg.norm(rj - rk)
        cos_i = np.dot(rj - ri, rk - ri) / (rij * rik)
        cos_j = np.dot(ri - rj, rk - rj) / (rij * rjk)
        cos_k = np.dot(ri - rk, rj - rk) / (rik * rjk)
        angular = 1.0 + 3.0 * cos_i * cos_j * cos_k
        edges = ((i, j), (i, k), (j, k))
        roots = np.asarray([np.sqrt(prepared.ratio[a, b]) for a, b in edges])
        radii = parameters.a1 * roots + parameters.a2
        rproduct = rij * rik * rjk
        critical = float(np.prod(radii))
        damping = 1.0 / (1.0 + 6.0 * (critical / rproduct)**(16.0 / 3.0))
        c9 = np.sqrt(np.prod([d4.c6[a, b] for a, b in edges]))
        unit = angular * c9 * damping / rproduct**3
        energy = parameters.s9 * unit
        partials["s9"] += unit
        damping_log_derivative = -(16.0 / 3.0) * (1.0 - damping)
        partials["a1"] += energy * damping_log_derivative * np.sum(roots / radii)
        partials["a2"] += energy * damping_log_derivative * np.sum(1.0 / radii)
    return partials


def d4_global_parameter_charge_potential_direction_dense(
    coordinates_bohr: np.ndarray, charges: np.ndarray,
    coordination: np.ndarray, atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters, d4: D4Result,
    parameter_direction: dict[str, float],
) -> np.ndarray:
    """Analytic ``d²E_D4/(dq dp) dp`` including pair and ATM terms.

    The BJ ratio ``C8/C6`` is independent of charge, so only the C6 charge
    derivative enters. This vector is the explicit D4 CPSCF source before
    contraction with the fixed AO charge operators.
    """
    positions = np.asarray(coordinates_bohr, dtype=float)
    q = np.asarray(charges, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    natoms = len(atom_models)
    if positions.shape != (natoms, 3) or q.shape != (natoms,) or cn.shape != (natoms,):
        raise ValueError("D4 parameter charge source dimensions are inconsistent")
    if set(parameter_direction) - {"a1", "a2", "s6", "s8", "s9", "s10"}:
        raise ValueError("unsupported D4 parameter direction")
    da1, da2, ds6, ds8, ds9, ds10 = (
        float(parameter_direction.get(name, 0.0))
        for name in ("a1", "a2", "s6", "s8", "s9", "s10")
    )
    if not np.all(np.isfinite((da1, da2, ds6, ds8, ds9, ds10))):
        raise ValueError("D4 parameter direction must be finite")
    frequency = np.asarray(parameters.frequency_weights, dtype=float)
    alpha = np.stack([
        atomic_polarizability_derivatives(float(q[a]), float(cn[a]), model)
        for a, model in enumerate(atom_models)
    ])
    dc6 = (3.0 / pi) * (
        (alpha[:, 1, :] * frequency[None, :]) @ alpha[:, 0, :].T
    )
    c6 = np.asarray(d4.c6, dtype=float)
    c8 = np.asarray(d4.c8, dtype=float)
    potential = np.zeros(natoms)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            radius = float(np.linalg.norm(positions[i] - positions[j]))
            ratio = c8[i, j] / c6[i, j]
            root = np.sqrt(ratio)
            damping_radius = parameters.a1 * root + parameters.a2
            ddamping_radius = da1 * root + da2
            coefficient_direction = 0.0
            for order, scale, dscale, factor in (
                (6, parameters.s6, ds6, 1.), (8, parameters.s8, ds8, ratio),
                (10, parameters.s10, ds10, 49. / 40. * ratio**2),
            ):
                den = radius**order + damping_radius**order
                coefficient_direction += factor * (-dscale / den + scale * order
                    * damping_radius**(order-1) / den**2 * ddamping_radius)
            potential[i] += dc6[i, j] * coefficient_direction
            potential[j] += dc6[j, i] * coefficient_direction
    if parameters.neutral_atm:
        return potential
    for i, j, k in combinations(range(natoms), 3):
        ri, rj, rk = positions[[i, j, k]]
        rij = float(np.linalg.norm(ri - rj))
        rik = float(np.linalg.norm(ri - rk))
        rjk = float(np.linalg.norm(rj - rk))
        cos_i = np.dot(rj - ri, rk - ri) / (rij * rik)
        cos_j = np.dot(ri - rj, rk - rj) / (rij * rjk)
        cos_k = np.dot(ri - rk, rj - rk) / (rik * rjk)
        angular = 1.0 + 3.0 * cos_i * cos_j * cos_k
        edges = ((i, j), (i, k), (j, k))
        radii = np.asarray([
            parameters.a1 * np.sqrt(c8[a, b] / c6[a, b]) + parameters.a2
            for a, b in edges
        ])
        dradii = np.asarray([
            da1 * np.sqrt(c8[a, b] / c6[a, b]) + da2
            for a, b in edges
        ])
        rproduct = rij * rik * rjk
        damping = 1.0 / (1.0 + 6.0 * (np.prod(radii) / rproduct)**(16.0 / 3.0))
        damping_direction = (
            damping * (-(16.0 / 3.0) * (1.0 - damping))
            * np.sum(dradii / radii)
        )
        coefficient_direction = ds9 * damping + parameters.s9 * damping_direction
        contribution = (
            angular * np.sqrt(c6[i, j] * c6[i, k] * c6[j, k])
            * coefficient_direction / rproduct**3
        )
        potential[i] += 0.5 * contribution * (
            dc6[i, j] / c6[i, j] + dc6[i, k] / c6[i, k]
        )
        potential[j] += 0.5 * contribution * (
            dc6[j, i] / c6[i, j] + dc6[j, k] / c6[j, k]
        )
        potential[k] += 0.5 * contribution * (
            dc6[k, i] / c6[i, k] + dc6[k, j] / c6[j, k]
        )
    return potential


def d4_global_parameter_second_direction_dense(
    coordinates_bohr: np.ndarray, d4: D4Result,
    parameters: D4Parameters, prepared: D4PreparedGeometry,
    parameter_direction: dict[str, float],
) -> dict[str, float]:
    """Explicit fixed-charge D4 global-head Hessian action ``E_pp dp``."""
    positions = np.asarray(coordinates_bohr, dtype=float)
    natoms = len(positions)
    if positions.shape != (natoms, 3) or d4.c6.shape != (natoms, natoms):
        raise ValueError("D4 second parameter direction requires dense molecular data")
    if set(parameter_direction) - {"a1", "a2", "s6", "s8", "s9", "s10"}:
        raise ValueError("unsupported D4 second parameter direction")
    da1, da2, ds6, ds8, ds9, ds10 = (
        float(parameter_direction.get(name, 0.0))
        for name in ("a1", "a2", "s6", "s8", "s9", "s10")
    )
    if not np.all(np.isfinite((da1, da2, ds6, ds8, ds9, ds10))):
        raise ValueError("D4 second parameter direction must be finite")
    result = {name: 0.0 for name in ("a1", "a2", "s6", "s8", "s9", "s10")}
    for i in range(natoms):
        for j in range(i + 1, natoms):
            radius = float(np.linalg.norm(positions[i] - positions[j]))
            ratio = float(prepared.ratio[i, j])
            root = np.sqrt(ratio)
            damping_radius = parameters.a1 * root + parameters.a2
            ddamping_radius = da1 * root + da2
            c6 = float(d4.c6[i, j])
            radial_direction = 0.0
            for order, scale, dscale, factor in (
                (6, parameters.s6, ds6, 1.), (8, parameters.s8, ds8, ratio),
                (10, parameters.s10, ds10, 49. / 40. * ratio**2),
            ):
                den = radius**order + damping_radius**order
                first = c6 * factor * order * damping_radius**(order-1) / den**2
                second = c6 * factor * order * ((order-1) * damping_radius**(order-2) / den**2
                    - 2*order * damping_radius**(2*order-2) / den**3)
                radial_direction += scale * second * ddamping_radius + dscale * first
                result[f"s{order}"] += first * ddamping_radius
            result["a1"] += radial_direction * root
            result["a2"] += radial_direction
    if parameters.neutral_atm:
        d4 = _neutral_pair_reference(positions, prepared, parameters)
    exponent = 16.0 / 3.0
    for i, j, k in combinations(range(natoms), 3):
        ri, rj, rk = positions[[i, j, k]]
        rij = float(np.linalg.norm(ri - rj))
        rik = float(np.linalg.norm(ri - rk))
        rjk = float(np.linalg.norm(rj - rk))
        cos_i = np.dot(rj - ri, rk - ri) / (rij * rik)
        cos_j = np.dot(ri - rj, rk - rj) / (rij * rjk)
        cos_k = np.dot(ri - rk, rj - rk) / (rik * rjk)
        angular = 1.0 + 3.0 * cos_i * cos_j * cos_k
        edges = ((i, j), (i, k), (j, k))
        roots = np.asarray([np.sqrt(prepared.ratio[a, b]) for a, b in edges])
        radii = parameters.a1 * roots + parameters.a2
        dradii = da1 * roots + da2
        rproduct = rij * rik * rjk
        damping = 1.0 / (1.0 + 6.0 * (np.prod(radii) / rproduct)**exponent)
        unit = angular * np.sqrt(np.prod([d4.c6[a, b] for a, b in edges])) / rproduct**3
        energy = parameters.s9 * unit * damping
        log_factor = -exponent * (1.0 - damping)
        damping_dot = damping * log_factor * np.sum(dradii / radii)
        log_factor_dot = exponent * damping_dot
        energy_dot = ds9 * unit * damping + parameters.s9 * unit * damping_dot
        root_sum = np.sum(roots / radii)
        unit_sum = np.sum(1.0 / radii)
        result["s9"] += unit * damping_dot
        result["a1"] += (
            energy_dot * log_factor * root_sum
            + energy * log_factor_dot * root_sum
            - energy * log_factor * np.sum(roots * dradii / radii**2)
        )
        result["a2"] += (
            energy_dot * log_factor * unit_sum
            + energy * log_factor_dot * unit_sum
            - energy * log_factor * np.sum(dradii / radii**2)
        )
    return result


def d4_global_head_adjoint_direction_dense(
    coordinates_bohr: np.ndarray, charges: np.ndarray,
    charge_direction: np.ndarray, coordination: np.ndarray,
    atom_models: Sequence[D4AtomModel], parameters: D4Parameters,
    d4: D4Result, prepared: D4PreparedGeometry,
    parameter_direction: dict[str, float],
    *, coordinate_direction: np.ndarray | None = None,
    coordination_direction: np.ndarray | None = None,
) -> dict[str, float]:
    """D4 head response to parameters/charges and optional geometry/CN."""
    qdot = np.asarray(charge_direction, dtype=float)
    if qdot.shape != np.asarray(charges).shape or not np.all(np.isfinite(qdot)):
        raise ValueError("D4 charge direction has incompatible dimensions")
    result = d4_global_parameter_second_direction_dense(
        coordinates_bohr, d4, parameters, prepared, parameter_direction,
    )
    for name in result:
        source = d4_global_parameter_charge_potential_direction_dense(
            coordinates_bohr, charges, coordination, atom_models,
            parameters, d4, {name: 1.0},
        )
        result[name] += float(np.dot(source, qdot))
    if coordinate_direction is not None or coordination_direction is not None:
        geometry = d4_global_parameter_geometry_direction_dense(
            coordinates_bohr, charges, coordination, atom_models, parameters,
            d4, prepared, coordinate_direction=coordinate_direction,
            coordination_direction=coordination_direction,
        )
        for name in result:
            result[name] += geometry[name]
    return result


def d4_global_parameter_geometry_direction_dense(
    coordinates_bohr: np.ndarray, charges: np.ndarray,
    coordination: np.ndarray, atom_models: Sequence[D4AtomModel],
    parameters: D4Parameters, d4: D4Result, prepared: D4PreparedGeometry,
    *, coordinate_direction: np.ndarray | None = None,
    coordination_direction: np.ndarray | None = None,
) -> dict[str, float]:
    """Exact geometry/CN derivative of the pair and ATM global adjoints."""
    xyz = np.asarray(coordinates_bohr, dtype=float)
    n = len(xyz)
    dx = np.zeros_like(xyz) if coordinate_direction is None else np.asarray(coordinate_direction, dtype=float)
    dcn = np.zeros(n) if coordination_direction is None else np.asarray(coordination_direction, dtype=float)
    if (xyz.shape != (n, 3) or dx.shape != xyz.shape or dcn.shape != (n,)
            or not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dcn))):
        raise ValueError("invalid D4 global geometry direction")
    coefficients = d4_coefficient_direction(
        charges, np.zeros(n), coordination, dcn, atom_models, prepared,
    )
    dc6 = np.zeros((n, n))
    dc6[coefficients.rows, coefficients.columns] = coefficients.c6_derivative
    dc6[coefficients.columns, coefficients.rows] = coefficients.c6_derivative
    result = {name: 0.0 for name in ("a1", "a2", "s6", "s8", "s9", "s10")}
    for i in range(n):
        for j in range(i + 1, n):
            vector = xyz[i] - xyz[j]
            r = float(np.linalg.norm(vector))
            dr = float(vector @ (dx[i] - dx[j])) / r
            ratio = float(prepared.ratio[i, j])
            root = np.sqrt(ratio)
            c = parameters.a1 * root + parameters.a2
            c6, c6dot = d4.c6[i, j], dc6[i, j]
            radial = radial_dot = 0.0
            for order, scale, factor in ((6, parameters.s6, 1.), (8, parameters.s8, ratio),
                                          (10, parameters.s10, 49. / 40. * ratio**2)):
                den = r**order + c**order
                radial += scale * factor * order * c**(order-1) / den**2
                radial_dot -= dr * scale * factor * 2*order**2 * c**(order-1) * r**(order-1) / den**3
                result[f"s{order}"] += -c6dot * factor / den + c6 * factor * order * r**(order-1) * dr / den**2
            value = c6dot * radial + c6 * radial_dot
            result["a1"] += value * root
            result["a2"] += value
    if parameters.neutral_atm:
        d4 = _neutral_pair_reference(xyz, prepared, parameters)
        coefficients = d4_coefficient_direction(
            np.zeros(n), np.zeros(n), coordination, dcn, atom_models, prepared,
        )
        dc6 = np.zeros((n, n))
        dc6[coefficients.rows, coefficients.columns] = coefficients.c6_derivative
        dc6[coefficients.columns, coefficients.rows] = coefficients.c6_derivative
    exponent = 16.0 / 3.0
    for i, j, k in combinations(range(n), 3):
        edges = ((i, j), (i, k), (j, k))
        lengths, rates = [], []
        for a, b in edges:
            vector = xyz[a] - xyz[b]
            r = float(np.linalg.norm(vector))
            lengths.append(r)
            rates.append(float(vector @ (dx[a] - dx[b])) / r**2)
        cosines, cosine_rates = [], []
        for a, b, c in ((i, j, k), (j, i, k), (k, i, j)):
            u, v = xyz[b] - xyz[a], xyz[c] - xyz[a]
            du, dv = dx[b] - dx[a], dx[c] - dx[a]
            norm = np.linalg.norm(u) * np.linalg.norm(v)
            cosine = float(u @ v) / norm
            cosine_rate = float(du @ v + u @ dv) / norm - cosine * (
                float(u @ du) / float(u @ u) + float(v @ dv) / float(v @ v)
            )
            cosines.append(cosine)
            cosine_rates.append(cosine_rate)
        c0, c1, c2 = cosines
        t0, t1, t2 = cosine_rates
        angular = 1.0 + 3.0 * c0 * c1 * c2
        angular_dot = 3.0 * (t0 * c1 * c2 + c0 * t1 * c2 + c0 * c1 * t2)
        product = float(np.prod(lengths))
        product_log_dot = sum(rates)
        roots = np.asarray([np.sqrt(prepared.ratio[a, b]) for a, b in edges])
        radii = parameters.a1 * roots + parameters.a2
        damping = 1.0 / (1.0 + 6.0 * (np.prod(radii) / product)**exponent)
        damping_dot = exponent * damping * (1.0 - damping) * product_log_dot
        c9 = np.sqrt(np.prod([d4.c6[a, b] for a, b in edges]))
        c9dot = 0.5 * c9 * sum(dc6[a, b] / d4.c6[a, b] for a, b in edges)
        unit = angular * c9 * damping / product**3
        unit_dot = (
            angular_dot * c9 * damping + angular * c9dot * damping
            + angular * c9 * damping_dot
        ) / product**3 - 3.0 * unit * product_log_dot
        factor = -exponent * (1.0 - damping)
        factor_dot = exponent * damping_dot
        value = parameters.s9 * (unit_dot * factor + unit * factor_dot)
        result["a1"] += value * np.sum(roots / radii)
        result["a2"] += value * np.sum(1.0 / radii)
        result["s9"] += unit_dot
    return result


def atm_three_body_energy_sparse(
    coordinates_bohr: np.ndarray,
    d4: SparseD4Result,
    *,
    s9: float = 5.0,
    a1: float = 0.52,
    a2: float = 5.0,
) -> SparseATMResult:
    """Evaluate ATM on triangles of the D4 neighbor graph in linear storage."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    natoms = len(coordinates)
    if _gfn2_native is None or not hasattr(_gfn2_native, "sparse_triangles"):
        raise RuntimeError("sparse ATM requires the native triangle kernel")
    triples = np.asarray(
        _gfn2_native.sparse_triangles(
            int(natoms),
            np.asarray(d4.rows, dtype=np.int64),
            np.asarray(d4.columns, dtype=np.int64),
        ),
        dtype=np.intp,
    )
    total_triples = natoms * (natoms - 1) * (natoms - 2) // 6
    if not len(triples):
        return SparseATMResult(
            0.0, triples, total_triples,
            float("inf") if total_triples else 0.0,
        )
    low, middle, high = triples.T
    keys = d4.rows.astype(np.int64) * max(natoms, 1) + d4.columns

    pair_keys = np.column_stack(
        (
            middle.astype(np.int64) * max(natoms, 1) + low,
            high.astype(np.int64) * max(natoms, 1) + low,
            high.astype(np.int64) * max(natoms, 1) + middle,
        )
    )
    pair_positions = np.searchsorted(keys, pair_keys)
    if np.any(pair_positions >= len(keys)) or np.any(
        keys[pair_positions] != pair_keys
    ):
        raise ValueError("ATM triangle references a missing D4 pair")
    lm, lh, mh = pair_positions.T
    c6_lm, c6_lh, c6_mh = d4.c6[lm], d4.c6[lh], d4.c6[mh]
    c8_lm, c8_lh, c8_mh = d4.c8[lm], d4.c8[lh], d4.c8[mh]
    r_lm = coordinates[low] - coordinates[middle]
    r_lh = coordinates[low] - coordinates[high]
    r_mh = coordinates[middle] - coordinates[high]
    d_lm = np.linalg.norm(r_lm, axis=1)
    d_lh = np.linalg.norm(r_lh, axis=1)
    d_mh = np.linalg.norm(r_mh, axis=1)
    cos_low = np.sum(r_lm * r_lh, axis=1) / (d_lm * d_lh)
    cos_middle = np.sum((-r_lm) * r_mh, axis=1) / (d_lm * d_mh)
    cos_high = np.sum(r_lh * r_mh, axis=1) / (d_lh * d_mh)
    angular = 1.0 + 3.0 * cos_low * cos_middle * cos_high
    c9 = np.sqrt(c6_lm * c6_lh * c6_mh)
    critical = (
        (a1 * np.sqrt(c8_lm / c6_lm) + a2)
        * (a1 * np.sqrt(c8_lh / c6_lh) + a2)
        * (a1 * np.sqrt(c8_mh / c6_mh) + a2)
    ) ** (1.0 / 3.0)
    product_distance = d_lm * d_lh * d_mh
    mean_distance = product_distance ** (1.0 / 3.0)
    damping = 1.0 / (1.0 + 6.0 * (critical / mean_distance) ** 16)
    terms = s9 * angular * c9 * damping / product_distance**3
    omitted = total_triples - len(triples)
    # |1+3 cos(A)cos(B)cos(C)| <= 4.  Use the weakest retained-edge
    # distance and largest C9 as a deliberately conservative fail-closed tail.
    minimum_edge = float(np.min(np.concatenate((d_lm, d_lh, d_mh))))
    maximum_c9 = float(np.max(c9, initial=0.0))
    tail_bound = omitted * abs(s9) * 4.0 * maximum_c9 / minimum_edge**9
    return SparseATMResult(
        float(np.sum(terms)), triples, omitted, float(tail_bound), pair_positions
    )


def atm_three_body_energy_gradient_sparse(
    coordinates_bohr: np.ndarray,
    d4: SparseD4Result,
    *,
    cutoff_bohr: float | None = None,
    s9: float = 5.0,
    a1: float = 0.52,
    a2: float = 5.0,
    alpha: float = 16.0,
) -> SparseATMResult:
    """Fused sparse ATM energy/gradient without allocating the triangle list."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if cutoff_bohr is not None:
        if not np.isfinite(cutoff_bohr) or cutoff_bohr <= 0.0:
            raise ValueError("ATM cutoff must be positive and finite")
        displacement = coordinates[d4.rows] - coordinates[d4.columns]
        retained = np.einsum("ij,ij->i", displacement, displacement) <= cutoff_bohr**2
        d4 = replace(
            d4,
            rows=d4.rows[retained],
            columns=d4.columns[retained],
            c6=d4.c6[retained],
            c8=d4.c8[retained],
            c6_coordination_left=d4.c6_coordination_left[retained],
            c6_coordination_right=d4.c6_coordination_right[retained],
        )
    if _gfn2_native is None or not hasattr(
        _gfn2_native, "sparse_atm_energy_gradient"
    ):
        # Compatibility fallback for an older extension build.
        atm = atm_three_body_energy_sparse(
            coordinates, d4, s9=s9, a1=a1, a2=a2
        )
        return SparseATMResult(
            atm.energy_hartree,
            atm.triples,
            atm.omitted_triples,
            atm.omitted_energy_bound_hartree,
            atm.pair_positions,
            included_triples=len(atm.triples),
        )
    energy, gradient, coordination_derivative, omitted, tail_bound, included = (
        _gfn2_native.sparse_atm_energy_gradient(
            coordinates,
            np.asarray(d4.rows, dtype=np.int64),
            np.asarray(d4.columns, dtype=np.int64),
            np.asarray(d4.c6, dtype=float),
            np.asarray(d4.c8, dtype=float),
            np.asarray(d4.c6_coordination_left, dtype=float),
            np.asarray(d4.c6_coordination_right, dtype=float),
            float(s9),
            float(a1),
            float(a2),
            float(alpha),
        )
    )
    return SparseATMResult(
        float(energy),
        np.empty((0, 3), dtype=np.intp),
        int(omitted),
        float(tail_bound),
        None,
        np.asarray(gradient, dtype=float),
        np.asarray(coordination_derivative, dtype=float),
        int(included),
    )


def atm_three_body_energy_gradient_dense(
    coordinates_bohr: np.ndarray,
    d4: D4Result,
    coordination_jacobian: np.ndarray,
    *,
    s9: float = 5.0,
    a1: float = 0.52,
    a2: float = 5.0,
    alpha: float = 16.0,
) -> SparseATMResult:
    """Evaluate the complete dense ATM graph with the fused native kernel."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    natoms = len(coordinates)
    if _gfn2_native is None or not hasattr(
        _gfn2_native, "sparse_atm_energy_gradient"
    ):
        return SparseATMResult(
            atm_three_body_energy(coordinates, d4.c6, d4.c8),
            np.empty((0, 3), dtype=np.intp),
            0,
            0.0,
        )
    rows, columns = np.tril_indices(natoms, k=-1)
    derivatives = np.asarray(d4.c6_coordination_derivative, dtype=float)
    packed = SparseD4Result(
        0.0,
        np.asarray(d4.charge_derivative_hartree, dtype=float),
        np.asarray(d4.coordination_derivative_hartree, dtype=float),
        rows,
        columns,
        np.asarray(d4.c6, dtype=float)[rows, columns],
        np.asarray(d4.c8, dtype=float)[rows, columns],
        derivatives[rows, columns, rows],
        derivatives[rows, columns, columns],
        0.0,
    )
    result = atm_three_body_energy_gradient_sparse(
        coordinates, packed, s9=s9, a1=a1, a2=a2, alpha=alpha
    )
    jacobian = np.asarray(coordination_jacobian, dtype=float)
    if jacobian.shape != (natoms, natoms, 3):
        raise ValueError("D4 coordination Jacobian has an inconsistent shape")
    gradient = np.asarray(result.explicit_gradient_hartree_per_bohr).copy()
    gradient += np.einsum(
        "a,axk->xk",
        np.asarray(result.coordination_derivative_hartree),
        jacobian,
        optimize=False,
    )
    return SparseATMResult(
        result.energy_hartree,
        result.triples,
        result.omitted_triples,
        result.omitted_energy_bound_hartree,
        result.pair_positions,
        gradient,
        np.zeros(natoms, dtype=float),
        result.included_triples,
    )


__all__ = [
    "D4AtomModel",
    "D4ChargeDirection",
    "D4CoefficientDirection",
    "D4Parameters",
    "D4PreparedGeometry",
    "SparseD4PreparedGeometry",
    "D4ReferenceState",
    "D4Result",
    "SparseD4Result",
    "SparseATMResult",
    "atm_three_body_energy",
    "atm_three_body_energy_sparse",
    "atm_three_body_energy_gradient_sparse",
    "atm_three_body_energy_gradient_dense",
    "atomic_polarizability",
    "atomic_polarizability_derivatives",
    "d4_charge_hessian",
    "d4_charge_hessian_matvec",
    "d4_charge_direction",
    "d4_coefficient_direction",
    "d4_fock",
    "d4_fock_sparse",
    "d4_two_body",
    "d4_two_body_sparse",
    "prepare_d4_geometry_sparse",
    "prepare_d4_geometry",
]
