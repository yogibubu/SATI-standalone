"""Lindh Cartesian model Hessian for geometry-optimization seeds.

The chemical model follows Lindh, Bernhardsson, Karlstroem, and Malmqvist,
Chem. Phys. Lett. 241, 423-428 (1995), DOI 10.1016/0009-2614(95)00646-L.
Its constants, damped D2 contribution, and approximate-normal-coordinate
conditioning are independently translated from the LGPL-3.0-or-later xTB
6.7.1 implementation (``ddvopt``/``detrotra8``), used here as the parity
reference.  This module is the single MATRIX kernel: ARCHITECT may expose it
and LINK may consume it, but neither package reimplements the equations.

Input coordinates are angstrom; returned Hessians are hartree/bohr**2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .primitive_coordinates import Primitive, grad_primitive
from .structural_corrections import BOHR_TO_ANGSTROM
from .topology.elements import atomic_number
from .vibrational import is_linear_geometry


LINDH_HESSIAN_SCHEMA = "matrix.chem.lindh_hessian.v1"
LINDH_REFERENCE = "Lindh_et_al_CPL_1995_DOI_10.1016/0009-2614(95)00646-L"
LINDH_XTB_PARITY_REFERENCE = "xtb-6.7.1-ddvopt-detrotra8"

_R_AV = np.asarray(
    (
        (1.3500, 2.1000, 2.5300),
        (2.1000, 2.8700, 3.4000),
        (2.5300, 3.4000, 3.4000),
    ),
    dtype=float,
)
_A_AV = np.asarray(
    (
        (1.0000, 0.3949, 0.3949),
        (0.3949, 0.2800, 0.2800),
        (0.3949, 0.2800, 0.2800),
    ),
    dtype=float,
)
_STRETCH_FORCE = 0.4000
_BEND_FORCE = 0.1300
_TORSION_FORCE = 0.007500
_PAIR_CUTOFF2_BOHR = 70.0
_TORSION_COSINE_LIMIT = math.cos(math.radians(35.0))

# Grimme D2 parameters used by xTB's Lindh ANC model.  Entries after Xe use
# the same documented fallback values as the parity implementation.
_VDW_RADII_ANGSTROM = np.asarray(
    (
        0.91, 0.92,
        0.75, 1.28, 1.35, 1.32, 1.27, 1.22, 1.17, 1.13,
        1.04, 1.24, 1.49, 1.56, 1.55, 1.53, 1.49, 1.45,
        1.35, 1.34,
        1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42,
        1.50, 1.57, 1.60, 1.61, 1.59, 1.57,
        1.48, 1.46,
        1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49,
        1.52, 1.64, 1.71, 1.72, 1.72, 1.71,
    ),
    dtype=float,
)


def _d2_c6_table() -> np.ndarray:
    values = np.full(100, 50.0, dtype=float)
    values[:18] = (
        0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75,
        0.63, 5.71, 5.71, 10.79, 9.23, 7.84, 5.57, 5.07, 4.61,
    )
    values[18:30] = 10.8
    values[30:36] = (16.99, 17.10, 16.37, 12.64, 12.47, 12.01)
    values[36:48] = 24.67
    values[48:54] = (37.32, 38.71, 38.44, 31.74, 31.50, 29.99)
    return values


_D2_C6 = _d2_c6_table()


@dataclass(frozen=True)
class LindhHessianResult:
    """Raw and ANC-conditioned Cartesian Lindh Hessians."""

    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    raw_hessian_hartree_per_bohr2: np.ndarray
    effective_hessian_hartree_per_bohr2: np.ndarray
    raw_eigenvalues: np.ndarray
    effective_eigenvalues: np.ndarray
    nulled_mode_indices: tuple[int, ...]
    damping_shift: float
    linear: bool
    s6: float
    hlow: float
    excluded_valence_pairs: tuple[tuple[int, int], ...] = ()
    schema: str = LINDH_HESSIAN_SCHEMA


def lindh_1995_cartesian_hessian(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    s6: float = 20.0,
    hlow: float = 0.01,
    condition_anc: bool = True,
    excluded_valence_pairs: Sequence[tuple[int, int]] = (),
) -> LindhHessianResult:
    """Build the general Lindh-1995 model Hessian and xTB ANC conditioning.

    ``excluded_valence_pairs`` uses zero-based atom indices.  It suppresses
    Lindh stretch, bend and torsion terms that would treat a declared
    center-based interaction as ordinary atom--atom valence structure.  The
    D2 pair contribution is retained, so this option replaces only the
    valence model and does not remove the general non-bonded curvature.
    """

    atom_tuple = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(atom_tuple), 3):
        raise ValueError("coordinates must have shape natoms x 3")
    if not atom_tuple:
        raise ValueError("the Lindh Hessian requires at least one atom")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates contain non-finite values")
    if not np.isfinite(s6) or float(s6) < 0.0:
        raise ValueError("s6 must be finite and non-negative")
    if not np.isfinite(hlow) or float(hlow) <= 0.0:
        raise ValueError("hlow must be finite and positive")
    atomic_numbers = tuple(int(atomic_number(atom) or 0) for atom in atom_tuple)
    if any(number <= 0 or number > 100 for number in atomic_numbers):
        raise ValueError("the Lindh Hessian requires recognized elements through Z=100")
    excluded_pairs = _validated_valence_pairs(excluded_valence_pairs, len(atom_tuple))

    coordinates_bohr = coordinates / BOHR_TO_ANGSTROM
    raw = _raw_lindh_hessian(
        atomic_numbers,
        coordinates_bohr,
        s6=float(s6),
        excluded_valence_pairs=excluded_pairs,
    )
    raw = 0.5 * (raw + raw.T)
    raw_values, raw_vectors = np.linalg.eigh(raw)
    linear = is_linear_geometry(coordinates)
    if condition_anc and len(atom_tuple) > 1:
        effective_values, nulled, damping = _condition_anc_eigenvalues(
            raw_values,
            raw_vectors,
            coordinates_bohr,
            linear=linear,
            hlow=float(hlow),
        )
        effective = raw_vectors @ np.diag(effective_values) @ raw_vectors.T
        effective = 0.5 * (effective + effective.T)
    else:
        effective_values = raw_values.copy()
        nulled = ()
        damping = 0.0
        effective = raw.copy()
    return LindhHessianResult(
        atoms=atom_tuple,
        coordinates_angstrom=coordinates.copy(),
        raw_hessian_hartree_per_bohr2=raw,
        effective_hessian_hartree_per_bohr2=effective,
        raw_eigenvalues=raw_values,
        effective_eigenvalues=effective_values,
        nulled_mode_indices=tuple(int(index) for index in nulled),
        damping_shift=float(damping),
        linear=bool(linear),
        s6=float(s6),
        hlow=float(hlow),
        excluded_valence_pairs=excluded_pairs,
    )


def _validated_valence_pairs(
    pairs: Sequence[tuple[int, int]],
    natoms: int,
) -> tuple[tuple[int, int], ...]:
    canonical: set[tuple[int, int]] = set()
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("excluded Lindh valence pairs must contain two atom indices")
        left, right = (int(item) for item in pair)
        if left == right or min(left, right) < 0 or max(left, right) >= natoms:
            raise ValueError("excluded Lindh valence pair lies outside the geometry")
        canonical.add(tuple(sorted((left, right))))
    return tuple(sorted(canonical))


def _raw_lindh_hessian(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    *,
    s6: float,
    excluded_valence_pairs: tuple[tuple[int, int], ...] = (),
) -> np.ndarray:
    natoms = len(atomic_numbers)
    size = 3 * natoms
    hessian = np.zeros((size, size), dtype=float)
    rows = tuple(_period_row(number) for number in atomic_numbers)
    excluded = set(excluded_valence_pairs)
    distance2 = np.sum(
        (coordinates_bohr[:, None, :] - coordinates_bohr[None, :, :]) ** 2,
        axis=2,
    )

    for high in range(natoms):
        for low in range(high):
            displacement = coordinates_bohr[high] - coordinates_bohr[low]
            radius2 = float(distance2[high, low])
            if radius2 <= np.finfo(float).tiny:
                raise ValueError("coincident atoms are invalid for the Lindh Hessian")
            row_high, row_low = rows[high], rows[low]
            r0 = float(_R_AV[row_high, row_low])
            alpha = float(_A_AV[row_high, row_low])
            unit = displacement / math.sqrt(radius2)
            block = -_d2_pair_hessian(
                displacement,
                atomic_numbers[high],
                atomic_numbers[low],
                s6=s6,
            )
            if (low, high) not in excluded:
                stiffness = _STRETCH_FORCE * math.exp(alpha * (r0 * r0 - radius2))
                block += stiffness * np.outer(unit, unit)
            _add_pair_block(hessian, high, low, block)

    for center in range(natoms):
        for first in range(natoms):
            if first == center or distance2[first, center] > _PAIR_CUTOFF2_BOHR:
                continue
            for second in range(first):
                if second == center:
                    continue
                if (
                    tuple(sorted((center, first))) in excluded
                    or tuple(sorted((center, second))) in excluded
                ):
                    continue
                if (
                    distance2[second, first] > _PAIR_CUTOFF2_BOHR
                    or distance2[second, center] > _PAIR_CUTOFF2_BOHR
                ):
                    continue
                left = coordinates_bohr[first] - coordinates_bohr[center]
                right = coordinates_bohr[second] - coordinates_bohr[center]
                left_norm = float(np.linalg.norm(left))
                right_norm = float(np.linalg.norm(right))
                if min(left_norm, right_norm) <= np.finfo(float).tiny:
                    continue
                cosine = float(np.dot(left, right) / (left_norm * right_norm))
                if cosine >= 1.0 - 1.0e-14:
                    continue
                row_center = rows[center]
                alpha_left = float(_A_AV[row_center, rows[first]])
                alpha_right = float(_A_AV[row_center, rows[second]])
                reference_left = float(_R_AV[row_center, rows[first]])
                reference_right = float(_R_AV[row_center, rows[second]])
                stiffness = _BEND_FORCE * math.exp(
                    alpha_left * (reference_left**2 - left_norm**2)
                    + alpha_right * (reference_right**2 - right_norm**2)
                )
                for gradient in _angle_gradient_components(
                    first,
                    center,
                    second,
                    coordinates_bohr,
                    cosine=cosine,
                ):
                    flat = gradient.reshape(-1)
                    hessian += stiffness * np.outer(flat, flat)

    for second in range(natoms):
        for third in range(natoms):
            if third == second or distance2[third, second] > _PAIR_CUTOFF2_BOHR:
                continue
            for first in range(natoms):
                if first in {second, third}:
                    continue
                if (
                    distance2[first, third] > _PAIR_CUTOFF2_BOHR
                    or distance2[first, second] > _PAIR_CUTOFF2_BOHR
                ):
                    continue
                first_second_index = natoms * second + first
                for fourth in range(natoms):
                    second_fourth_index = natoms * third + fourth
                    if first_second_index <= second_fourth_index:
                        continue
                    if fourth in {first, second, third}:
                        continue
                    if (
                        distance2[fourth, first] > _PAIR_CUTOFF2_BOHR
                        or distance2[fourth, third] > _PAIR_CUTOFF2_BOHR
                        or distance2[fourth, second] > _PAIR_CUTOFF2_BOHR
                    ):
                        continue
                    r12 = coordinates_bohr[first] - coordinates_bohr[second]
                    r23 = coordinates_bohr[second] - coordinates_bohr[third]
                    r34 = coordinates_bohr[third] - coordinates_bohr[fourth]
                    norms2 = tuple(float(np.dot(vector, vector)) for vector in (r12, r23, r34))
                    if min(norms2) <= np.finfo(float).tiny:
                        continue
                    cosine_1 = float(np.dot(r12, r23) / math.sqrt(norms2[0] * norms2[1]))
                    cosine_2 = float(np.dot(r34, r23) / math.sqrt(norms2[2] * norms2[1]))
                    if (
                        abs(cosine_1) > _TORSION_COSINE_LIMIT
                        or abs(cosine_2) > _TORSION_COSINE_LIMIT
                    ):
                        continue
                    pairs = (
                        (first, second, norms2[0]),
                        (second, third, norms2[1]),
                        (third, fourth, norms2[2]),
                    )
                    if any(tuple(sorted((atom_a, atom_b))) in excluded for atom_a, atom_b, _radius2 in pairs):
                        continue
                    exponent = 0.0
                    for atom_a, atom_b, radius2 in pairs:
                        alpha = float(_A_AV[rows[atom_a], rows[atom_b]])
                        reference = float(_R_AV[rows[atom_a], rows[atom_b]])
                        exponent += alpha * (reference**2 - radius2)
                    stiffness = _TORSION_FORCE * math.exp(exponent)
                    local_gradient = _analytic_dihedral_gradient(
                        coordinates_bohr[first],
                        coordinates_bohr[second],
                        coordinates_bohr[third],
                        coordinates_bohr[fourth],
                    )
                    if np.all(np.isfinite(local_gradient)):
                        _add_local_outer_product(
                            hessian,
                            (first, second, third, fourth),
                            local_gradient,
                            stiffness,
                        )
    return hessian


def _analytic_dihedral_gradient(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> np.ndarray:
    """Closed-form local Wilson row for a signed four-atom dihedral."""

    left = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    axis = np.asarray(third, dtype=float) - np.asarray(second, dtype=float)
    right = np.asarray(fourth, dtype=float) - np.asarray(third, dtype=float)
    axis2 = float(np.dot(axis, axis))
    axis_norm = math.sqrt(axis2)
    first_normal = np.cross(left, axis)
    second_normal = np.cross(axis, right)
    if min(
        axis2,
        float(np.dot(first_normal, first_normal)),
        float(np.dot(second_normal, second_normal)),
    ) <= np.finfo(float).tiny:
        return np.full((4, 3), np.nan, dtype=float)
    identity = np.eye(3, dtype=float)
    zero = np.zeros((3, 3), dtype=float)
    left_jacobian = np.hstack((identity, -identity, zero, zero))
    axis_jacobian = np.hstack((zero, -identity, identity, zero))
    right_jacobian = np.hstack((zero, zero, -identity, identity))

    def cross_matrix(vector: np.ndarray) -> np.ndarray:
        x, y, z = (float(value) for value in vector)
        return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))

    first_normal_jacobian = (
        -cross_matrix(axis) @ left_jacobian
        + cross_matrix(left) @ axis_jacobian
    )
    second_normal_jacobian = (
        -cross_matrix(right) @ axis_jacobian
        + cross_matrix(axis) @ right_jacobian
    )
    unit_axis = axis / axis_norm
    unit_axis_jacobian = (
        (identity - np.outer(unit_axis, unit_axis)) / axis_norm
    ) @ axis_jacobian
    cross_normals = np.cross(first_normal, second_normal)
    cross_normals_jacobian = (
        -cross_matrix(second_normal) @ first_normal_jacobian
        + cross_matrix(first_normal) @ second_normal_jacobian
    )
    x_value = float(np.dot(first_normal, second_normal))
    y_value = float(np.dot(cross_normals, unit_axis))
    x_derivative = (
        second_normal @ first_normal_jacobian
        + first_normal @ second_normal_jacobian
    )
    y_derivative = (
        unit_axis @ cross_normals_jacobian
        + cross_normals @ unit_axis_jacobian
    )
    denominator = x_value * x_value + y_value * y_value
    if denominator <= np.finfo(float).tiny:
        return np.full((4, 3), np.nan, dtype=float)
    derivative = (x_value * y_derivative - y_value * x_derivative) / denominator
    return derivative.reshape(4, 3)


def _add_local_outer_product(
    hessian: np.ndarray,
    atoms: tuple[int, int, int, int],
    local_gradient: np.ndarray,
    stiffness: float,
) -> None:
    """Accumulate one analytic 12x12 torsional block without a global row."""

    block = float(stiffness) * np.outer(local_gradient.reshape(-1), local_gradient.reshape(-1))
    indices = np.asarray(
        [3 * atom + component for atom in atoms for component in range(3)],
        dtype=int,
    )
    hessian[np.ix_(indices, indices)] += block


def _period_row(atomic_number_value: int) -> int:
    if atomic_number_value <= 2:
        return 0
    if atomic_number_value <= 10:
        return 1
    return 2


def _angle_gradient_components(
    first: int,
    center: int,
    second: int,
    coordinates_bohr: np.ndarray,
    *,
    cosine: float,
) -> tuple[np.ndarray, ...]:
    sine2 = max(0.0, 1.0 - cosine * cosine)
    if sine2 > 1.0e-20:
        gradient = grad_primitive(
            Primitive("angle", (first, center, second)), coordinates_bohr
        )
        return (gradient,)
    # At a linear angle the ordinary angle derivative is singular.  MATRIX's
    # two analytic linear-bend rows span the same rotationally invariant
    # tangent plane as the two vectors used by ddvopt.
    return tuple(
        grad_primitive(
            Primitive("linear_bend", (first, center, second), mode=mode),
            coordinates_bohr,
        )
        for mode in (-1, -2)
    )


def _add_pair_block(
    hessian: np.ndarray,
    first: int,
    second: int,
    block: np.ndarray,
) -> None:
    first_slice = slice(3 * first, 3 * first + 3)
    second_slice = slice(3 * second, 3 * second + 3)
    hessian[first_slice, first_slice] += block
    hessian[second_slice, second_slice] += block
    hessian[first_slice, second_slice] -= block
    hessian[second_slice, first_slice] -= block


def _d2_pair_hessian(
    displacement_bohr: np.ndarray,
    atomic_number_first: int,
    atomic_number_second: int,
    *,
    s6: float,
) -> np.ndarray:
    if s6 == 0.0:
        return np.zeros((3, 3), dtype=float)
    c66 = math.sqrt(
        float(_D2_C6[atomic_number_first - 1])
        * float(_D2_C6[atomic_number_second - 1])
    )
    radius_first = (
        float(_VDW_RADII_ANGSTROM[atomic_number_first - 1])
        if atomic_number_first <= len(_VDW_RADII_ANGSTROM)
        else 2.0
    )
    radius_second = (
        float(_VDW_RADII_ANGSTROM[atomic_number_second - 1])
        if atomic_number_second <= len(_VDW_RADII_ANGSTROM)
        else 2.0
    )
    r0 = (radius_first + radius_second) / 0.52917721
    x, y, z = (float(value) for value in displacement_bohr)
    return np.asarray(
        (
            (
                _d2_second_same(x, y, z, c66, s6, r0),
                _d2_second_mixed(x, y, z, c66, s6, r0),
                _d2_second_mixed(x, z, y, c66, s6, r0),
            ),
            (
                _d2_second_mixed(x, y, z, c66, s6, r0),
                _d2_second_same(y, x, z, c66, s6, r0),
                _d2_second_mixed(y, z, x, c66, s6, r0),
            ),
            (
                _d2_second_mixed(x, z, y, c66, s6, r0),
                _d2_second_mixed(y, z, x, c66, s6, r0),
                _d2_second_same(z, x, y, c66, s6, r0),
            ),
        ),
        dtype=float,
    )


def _d2_second_mixed(
    rx: float, ry: float, rz: float, c66: float, s6: float, r0: float
) -> float:
    avdw = 20.0
    t1 = s6 * c66
    radius2 = rx * rx + ry * ry + rz * rz
    radius = math.sqrt(radius2)
    radius4 = radius2**2
    radius8 = radius4**2
    damping = math.exp(-avdw * (radius / r0 - 1.0))
    denominator = 1.0 + damping
    inverse_denominator2 = 1.0 / denominator**2
    inverse_radius8 = 1.0 / radius8
    avdw2_over_r02 = avdw**2 / r0**2
    return float(
        -48.0 * t1 / radius8 / radius2 / denominator * rx * ry
        + 13.0
        * t1
        / radius
        / radius8
        * inverse_denominator2
        * rx
        * avdw
        / r0
        * ry
        * damping
        - 2.0
        * t1
        * inverse_radius8
        / denominator**3
        * avdw2_over_r02
        * rx
        * damping**2
        * ry
        + t1
        * inverse_radius8
        * inverse_denominator2
        * avdw2_over_r02
        * rx
        * ry
        * damping
    )


def _d2_second_same(
    rx: float, ry: float, rz: float, c66: float, s6: float, r0: float
) -> float:
    avdw = 20.0
    t1 = s6 * c66
    x2 = rx * rx
    radius2 = x2 + ry * ry + rz * rz
    radius = math.sqrt(radius2)
    radius4 = radius2**2
    radius8 = radius4**2
    damping = math.exp(-avdw * (radius / r0 - 1.0))
    denominator = 1.0 + damping
    inverse_denominator = 1.0 / denominator
    inverse_denominator2 = 1.0 / denominator**2
    inverse_radius8 = 1.0 / radius8
    avdw2_over_r02 = avdw**2 / r0**2
    return float(
        -48.0 * t1 / radius8 / radius2 * inverse_denominator * x2
        + 13.0
        * t1
        / radius
        / radius8
        * inverse_denominator2
        * x2
        * avdw
        / r0
        * damping
        + 6.0 * t1 * inverse_radius8 * inverse_denominator
        - 2.0
        * t1
        * inverse_radius8
        / denominator**3
        * avdw2_over_r02
        * x2
        * damping**2
        - t1
        / radius
        / radius4
        / radius2
        * inverse_denominator2
        * avdw
        / r0
        * damping
        + t1
        * inverse_radius8
        * inverse_denominator2
        * avdw2_over_r02
        * x2
        * damping
    )


def _condition_anc_eigenvalues(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    coordinates_bohr: np.ndarray,
    *,
    linear: bool,
    hlow: float,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    values = np.asarray(eigenvalues, dtype=float).copy()
    candidates: list[tuple[float, int]] = []
    natoms = coordinates_bohr.shape[0]
    for index, value in enumerate(values):
        if value > 0.05:
            continue
        distorted = coordinates_bohr + eigenvectors[:, index].reshape(natoms, 3)
        distance_change2 = 0.0
        for high in range(1, natoms):
            for low in range(high):
                original = float(
                    np.linalg.norm(coordinates_bohr[high] - coordinates_bohr[low])
                )
                displaced = float(np.linalg.norm(distorted[high] - distorted[low]))
                distance_change2 += (original - displaced) ** 2
        score = math.sqrt(distance_change2 / natoms) * abs(float(value))
        candidates.append((score, index))
    null_count = 5 if linear else 6
    if len(candidates) < null_count:
        raise ValueError("Lindh Hessian lacks the expected rigid-body null-space candidates")
    nulled = tuple(index for _score, index in sorted(candidates)[:null_count])
    values[list(nulled)] = 0.0
    retained = values[np.abs(values) > 1.0e-10]
    if retained.size == 0:
        damping = 0.0
    else:
        damping = max(float(hlow) - float(np.min(retained)), 0.0)
    shifted = np.abs(values) > 1.0e-11
    values[shifted] += damping
    return values, nulled, float(damping)


__all__ = [
    "LINDH_HESSIAN_SCHEMA",
    "LINDH_REFERENCE",
    "LINDH_XTB_PARITY_REFERENCE",
    "LindhHessianResult",
    "lindh_1995_cartesian_hessian",
]
