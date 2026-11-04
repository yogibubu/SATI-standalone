"""Targeted local symmetric-stretch/bend quartics from four QM gradients."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_link import hybrid_internal_coordinate_step, nonlinear_internal_coordinate_step
from matrix_smith import (
    build_gic_b_matrix,
    evaluate_gic_values,
)

from .acquisition import ZaffAnharmonicPointRequest


ZAFF_TWO_MODE_QUARTIC_SCHEMA = "matrix.zaff.two_mode_quartic.v1"


@dataclass(frozen=True)
class LocalSymmetricStretchBendPair:
    """One XH2 local block represented in the frozen SONIC chart."""

    identifier: str
    center_atom_one_based: int
    hydrogen_atoms_one_based: tuple[int, int]
    stretch_coordinate_indices: tuple[int, int]
    bend_coordinate_index: int
    stretch_direction: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "center_atom_one_based": self.center_atom_one_based,
            "hydrogen_atoms_one_based": list(self.hydrogen_atoms_one_based),
            "stretch_coordinate_indices_zero_based": list(self.stretch_coordinate_indices),
            "bend_coordinate_index_zero_based": self.bend_coordinate_index,
            "stretch_direction": np.asarray(self.stretch_direction, dtype=float).tolist(),
            "stretch_definition": "(r_XH1+r_XH2)/sqrt(2)",
            "selection": "LOCAL_XH2_TOTAL_SYMMETRIC_STRETCH_WITH_MATCHING_H_X_H_BEND",
        }


@dataclass(frozen=True)
class TwoModeQuarticFit:
    """Least-squares estimate of one residual K_ssbb term."""

    pair_identifier: str
    value_hartree_per_angstrom2_radian2: float
    residual_rms_internal_gradient: float
    residual_max_internal_gradient: float
    observation_count: int
    design_rank: int
    point_estimates: tuple[float, ...]
    nuisance_cubic_derivatives: Mapping[str, float]
    unpromoted_cross_quartic_diagnostics: Mapping[str, object]
    minimum_k_gradient_signal: float
    signal_to_residual_ratio: float
    promotion_recommended: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_identifier": self.pair_identifier,
            "K_ssbb_hartree_per_angstrom2_radian2": (
                self.value_hartree_per_angstrom2_radian2
            ),
            "residual_rms_internal_gradient": self.residual_rms_internal_gradient,
            "residual_max_internal_gradient": self.residual_max_internal_gradient,
            "observation_count": self.observation_count,
            "design_rank": self.design_rank,
            "point_estimates_hartree_per_angstrom2_radian2": list(self.point_estimates),
            "nuisance_cubic_derivatives": dict(self.nuisance_cubic_derivatives),
            "unpromoted_cross_quartic_diagnostics": dict(
                self.unpromoted_cross_quartic_diagnostics
            ),
            "promotion": {
                "recommended": self.promotion_recommended,
                "minimum_K_ssbb_gradient_signal": self.minimum_k_gradient_signal,
                "signal_to_residual_ratio": self.signal_to_residual_ratio,
                "minimum_signal_to_residual_ratio": 5.0,
                "selection_uses_external_reference": False,
            },
            "fit_model": (
                "DELTA_V_CUBIC(phi_sss,phi_ssb,phi_sbb,phi_bbb)"
                "+diagnostic(K_sssb,K_sbbb)+K_ssbb*s^2*b^2/4"
            ),
        }


@dataclass(frozen=True)
class TwoModeQuarticCorrection:
    """Energy and internal derivatives of the promoted K_ssbb terms."""

    energy_hartree: float
    gradient_internal: np.ndarray
    hessian_internal: np.ndarray


def identify_local_symmetric_stretch_bend_pairs(
    definition: object,
    atoms: Sequence[str],
    *,
    center_elements: tuple[str, ...] = ("C", "N"),
) -> tuple[LocalSymmetricStretchBendPair, ...]:
    """Find XH2 symmetric stretches and their native H--X--H bend.

    Only native one-primitive stretch and angle GICs are admitted.  This keeps
    the acquisition contract independent of global molecular symmetry and
    prevents an approximate chemical match from silently selecting a mode.
    """

    symbols = tuple(str(atom).upper() for atom in atoms)
    gics = tuple(getattr(definition, "gics"))
    primitive_by_id = {
        primitive.identifier: primitive for primitive in getattr(definition, "primitives")
    }
    stretches_by_center: dict[int, list[tuple[int, int]]] = {}
    for index, gic in enumerate(gics):
        primitive = _single_native_primitive(gic, primitive_by_id)
        if primitive is None or primitive.function != "R" or len(primitive.atoms) != 2:
            continue
        first, second = (int(value) for value in primitive.atoms)
        if symbols[first - 1] == "H" and symbols[second - 1] != "H":
            center, hydrogen = second, first
        elif symbols[second - 1] == "H" and symbols[first - 1] != "H":
            center, hydrogen = first, second
        else:
            continue
        if symbols[center - 1] in center_elements:
            stretches_by_center.setdefault(center, []).append((index, hydrogen))

    pairs: list[LocalSymmetricStretchBendPair] = []
    for center, stretch_rows in sorted(stretches_by_center.items()):
        if len(stretch_rows) != 2:
            continue
        ordered = tuple(sorted(stretch_rows, key=lambda item: item[1]))
        stretch_indices = tuple(sorted((ordered[0][0], ordered[1][0])))
        hydrogens = (ordered[0][1], ordered[1][1])
        bend_index = _matching_hxh_bend(
            gics,
            primitive_by_id,
            center=center,
            hydrogens=hydrogens,
        )
        if bend_index is None:
            continue
        direction = np.zeros(len(gics), dtype=float)
        direction[list(stretch_indices)] = 1.0 / sqrt(2.0)
        pairs.append(
            LocalSymmetricStretchBendPair(
                identifier=f"{symbols[center - 1]}H2_CENTER_{center}",
                center_atom_one_based=center,
                hydrogen_atoms_one_based=hydrogens,
                stretch_coordinate_indices=stretch_indices,
                bend_coordinate_index=bend_index,
                stretch_direction=direction,
            )
        )
    return tuple(pairs)


def build_two_mode_quartic_requests(
    definition: object,
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    pairs: Sequence[LocalSymmetricStretchBendPair],
    *,
    bond_displacement_angstrom: float = 0.01,
    bend_displacement_radian: float = 0.02,
    backtransform_tolerance: float = 2.0e-8,
) -> tuple[ZaffAnharmonicPointRequest, ...]:
    """Build the four simultaneous +/-s,+/-b gradient points per pair."""

    reference_coordinates = np.asarray(reference_coordinates_angstrom, dtype=float)
    reference_values = np.asarray(
        evaluate_gic_values(definition, coordinates_angstrom=reference_coordinates), dtype=float
    )
    stretch_step = sqrt(2.0) * float(bond_displacement_angstrom)
    bend_step = float(bend_displacement_radian)
    if stretch_step <= 0.0 or bend_step <= 0.0:
        raise ValueError("two-mode displacement steps must be positive")

    def evaluate(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(
            evaluate_gic_values(definition, coordinates_angstrom=coordinates), dtype=float
        )
        b_matrix = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows, dtype=float
        )
        return values, b_matrix

    requests: list[ZaffAnharmonicPointRequest] = []
    for pair in pairs:
        for stretch_sign in (-1, 1):
            for bend_sign in (-1, 1):
                target = reference_values.copy()
                target += stretch_sign * stretch_step * pair.stretch_direction
                target[pair.bend_coordinate_index] += bend_sign * bend_step
                result = hybrid_internal_coordinate_step(
                    definition,
                    reference_coordinates,
                    target,
                    evaluate,
                    tolerance=backtransform_tolerance,
                )
                if not result.converged:
                    result = nonlinear_internal_coordinate_step(
                        reference_coordinates,
                        target,
                        evaluate,
                        tolerance=backtransform_tolerance,
                    )
                if not result.converged:
                    raise ValueError(
                        f"two-mode back-transform failed for {pair.identifier} "
                        f"({stretch_sign:+d}s,{bend_sign:+d}b); "
                        f"residual={float(np.linalg.norm(result.residual)):.3e}"
                    )
                point_id = (
                    f"{pair.identifier.lower()}_"
                    f"s{'p' if stretch_sign > 0 else 'm'}_"
                    f"b{'p' if bend_sign > 0 else 'm'}"
                )
                requests.append(
                    ZaffAnharmonicPointRequest(
                        point_id=point_id,
                        purpose="TARGETED_LOCAL_SYMMETRIC_STRETCH_BEND_K_SSBB",
                        atoms=tuple(str(atom) for atom in atoms),
                        coordinates_angstrom=result.coordinates_angstrom,
                        coordinate_identifier=pair.identifier,
                        offset_multiplier=stretch_sign,
                        operation="SINGLE_POINT_GRADIENT",
                        constraints=(
                            {
                                "pair_identifier": pair.identifier,
                                "stretch_sign": stretch_sign,
                                "bend_sign": bend_sign,
                                "s_displacement_angstrom": stretch_sign * stretch_step,
                                "per_bond_displacement_angstrom": (
                                    stretch_sign * bond_displacement_angstrom
                                ),
                                "b_displacement_radian": bend_sign * bend_step,
                                "bend_coordinate_index_zero_based": pair.bend_coordinate_index,
                                "stretch_direction": pair.stretch_direction.tolist(),
                                "all_other_sonic_coordinates": "FIXED_AT_REFERENCE",
                            },
                        ),
                    )
                )
    return tuple(requests)


def cartesian_gradient_to_internal(
    definition: object,
    coordinates_angstrom: np.ndarray,
    cartesian_gradient_hartree_per_bohr: np.ndarray,
) -> np.ndarray:
    """Transform a Cartesian gradient to the frozen nonredundant SONIC chart."""

    b_matrix = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=coordinates_angstrom).rows,
        dtype=float,
    )
    gradient_bohr = np.asarray(cartesian_gradient_hartree_per_bohr, dtype=float).reshape(-1)
    if gradient_bohr.shape != (b_matrix.shape[1],):
        raise ValueError("Cartesian gradient size differs from the SONIC B matrix")
    cartesian_from_q = np.linalg.pinv(b_matrix, rcond=1.0e-8)
    gradient_per_angstrom = gradient_bohr / BOHR_TO_ANGSTROM
    return cartesian_from_q.T @ gradient_per_angstrom


def fit_residual_k_ssbb(
    pair: LocalSymmetricStretchBendPair,
    samples: Sequence[Mapping[str, object]],
) -> TwoModeQuarticFit:
    """Fit one residual K_ssbb after subtracting the current ZAFF gradient.

    Each sample supplies ``s``, ``b`` and the residual internal-gradient
    projections ``gradient_s`` and ``gradient_b``.  The harmonic and diagonal
    one-mode field is held fixed.  Four residual cubic derivatives are fitted
    as nuisance terms because a symmetric combination of two native stretches
    can expose three-coordinate cubics absent from a semidiagonal baseline.
    The four corners therefore provide eight observations for seven unknowns:
    ``phi_sss``, ``phi_ssb``, ``phi_sbb``, ``phi_bbb`` and the physical
    ``K_sssb``, ``K_ssbb`` and ``K_sbbb``.  Only ``K_ssbb`` is promoted;
    the two odd cross quartics are retained as diagnostics because they do not
    contribute directly to fundamentals at first order.
    """

    design: list[list[float]] = []
    target: list[float] = []
    point_estimates: list[float] = []
    for sample in samples:
        s = float(sample["s_displacement_angstrom"])
        b = float(sample["b_displacement_radian"])
        gradient_s = float(sample["residual_gradient_s_hartree_per_angstrom"])
        gradient_b = float(sample["residual_gradient_b_hartree_per_radian"])
        row_s = 0.5 * s * b**2
        row_b = 0.5 * s**2 * b
        for row, value in ((row_s, gradient_s), (row_b, gradient_b)):
            if abs(row) <= np.finfo(float).eps:
                raise ValueError("two-mode K_ssbb design contains a zero displacement")
            target.append(value)
            point_estimates.append(value / row)
        design.extend(
            (
                [
                    0.5 * s**2,
                    s * b,
                    0.5 * b**2,
                    0.0,
                    0.5 * s**2 * b,
                    row_s,
                    b**3 / 6.0,
                ],
                [
                    0.0,
                    0.5 * s**2,
                    s * b,
                    0.5 * b**2,
                    s**3 / 6.0,
                    row_b,
                    0.5 * s * b**2,
                ],
            )
        )
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(target, dtype=float)
    column_norms = np.linalg.norm(matrix, axis=0)
    if np.any(column_norms <= np.finfo(float).eps):
        raise ValueError("two-mode quartic design contains an unobservable coefficient")
    scaled_coefficients, _residuals, rank, singular = np.linalg.lstsq(
        matrix / column_norms[None, :], values, rcond=1.0e-12
    )
    coefficients = scaled_coefficients / column_norms
    prediction = matrix @ coefficients
    residual = prediction - values
    residual_max = float(np.max(np.abs(residual)))
    minimum_signal = float(np.min(np.abs(matrix[:, 5] * coefficients[5])))
    signal_to_residual = (
        float("inf") if residual_max <= np.finfo(float).eps else minimum_signal / residual_max
    )
    return TwoModeQuarticFit(
        pair_identifier=pair.identifier,
        value_hartree_per_angstrom2_radian2=float(coefficients[5]),
        residual_rms_internal_gradient=float(np.sqrt(np.mean(residual**2))),
        residual_max_internal_gradient=residual_max,
        observation_count=len(values),
        design_rank=int(rank),
        point_estimates=tuple(float(value) for value in point_estimates),
        nuisance_cubic_derivatives={
            "phi_sss_hartree_per_angstrom3": float(coefficients[0]),
            "phi_ssb_hartree_per_angstrom2_radian": float(coefficients[1]),
            "phi_sbb_hartree_per_angstrom_radian2": float(coefficients[2]),
            "phi_bbb_hartree_per_radian3": float(coefficients[3]),
        },
        unpromoted_cross_quartic_diagnostics={
            "K_sssb_hartree_per_angstrom3_radian": float(coefficients[4]),
            "K_sbbb_hartree_per_angstrom_radian3": float(coefficients[6]),
            "promotion_policy": "DIAGNOSTIC_ONLY_NOT_A_FIRST_ORDER_FUNDAMENTAL_TERM",
            "column_normalized_condition_number": float(singular[0] / singular[-1]),
        },
        minimum_k_gradient_signal=minimum_signal,
        signal_to_residual_ratio=signal_to_residual,
        promotion_recommended=signal_to_residual >= 5.0,
    )


def evaluate_promoted_k_ssbb(
    displacement: np.ndarray,
    pairs: Sequence[LocalSymmetricStretchBendPair],
    coefficients: Mapping[str, float],
) -> TwoModeQuarticCorrection:
    """Evaluate only the physically promoted ``K_ssbb*s^2*b^2/4`` terms."""

    q = np.asarray(displacement, dtype=float).reshape(-1)
    gradient = np.zeros_like(q)
    hessian = np.zeros((q.size, q.size), dtype=float)
    energy = 0.0
    for pair in pairs:
        if pair.identifier not in coefficients:
            continue
        direction = np.asarray(pair.stretch_direction, dtype=float)
        if direction.shape != q.shape:
            raise ValueError("two-mode stretch direction differs from the SONIC dimension")
        bend_direction = np.zeros_like(q)
        bend_direction[pair.bend_coordinate_index] = 1.0
        s = float(direction @ q)
        b = float(q[pair.bend_coordinate_index])
        coefficient = float(coefficients[pair.identifier])
        energy += 0.25 * coefficient * s**2 * b**2
        gradient += 0.5 * coefficient * s * b**2 * direction
        gradient += 0.5 * coefficient * s**2 * b * bend_direction
        hessian += 0.5 * coefficient * b**2 * np.outer(direction, direction)
        hessian += coefficient * s * b * (
            np.outer(direction, bend_direction) + np.outer(bend_direction, direction)
        )
        hessian += 0.5 * coefficient * s**2 * np.outer(
            bend_direction, bend_direction
        )
    return TwoModeQuarticCorrection(float(energy), gradient, hessian)


def _single_native_primitive(gic: object, primitive_by_id: Mapping[str, object]) -> object | None:
    components = tuple(getattr(gic, "coefficients")) or ((gic.primitive_id, 1.0),)
    if len(components) != 1 or not np.isclose(float(components[0][1]), 1.0):
        return None
    return primitive_by_id.get(components[0][0])


def _matching_hxh_bend(
    gics: Sequence[object],
    primitive_by_id: Mapping[str, object],
    *,
    center: int,
    hydrogens: tuple[int, int],
) -> int | None:
    expected = {int(hydrogens[0]), int(hydrogens[1])}
    matches: list[int] = []
    for index, gic in enumerate(gics):
        primitive = _single_native_primitive(gic, primitive_by_id)
        if primitive is None or primitive.function != "A" or len(primitive.atoms) != 3:
            continue
        first, middle, third = (int(value) for value in primitive.atoms)
        if middle == center and {first, third} == expected:
            matches.append(index)
    if len(matches) > 1:
        raise ValueError(f"multiple native H-X-H bends found for center {center}")
    return None if not matches else matches[0]
