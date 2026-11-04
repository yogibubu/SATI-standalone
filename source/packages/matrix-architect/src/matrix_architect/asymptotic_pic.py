"""Asymptotic evaluation of a fitted primitive ZAFF bonded field.

The local harmonic two-jet is retained. Angular/torsional terms and all cross
terms vanish when a defining covalent bond dissociates; torsions are periodic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from matrix_chem.primitive_coordinates import Primitive, eval_primitives, primitive_b_matrix
from matrix_chem.primitive_force_field import (
    PrimitiveMMForceField, evaluate_primitive_functional_parameter,
    evaluate_torsion_fourier_parameter,
)


@dataclass(frozen=True)
class AsymptoticPicFactor:
    primitive_index: int
    form: str
    power: int = 1
    harmonic: int = 1
    window_start: float = 0.0
    window_end: float = 0.0


def periodic_outer_window(
    delta: float, start: float, end: float,
) -> tuple[float, float, float]:
    """C2 periodic activation outside a protected reference neighbourhood.

    Returns value and its first two derivatives with respect to the signed,
    wrapped dihedral displacement.  The factor is exactly zero through
    ``start`` and exactly one from ``end`` onwards.
    """
    if not (np.isfinite(start) and np.isfinite(end) and 0.0 < start < end < math.pi):
        raise ValueError("invalid periodic outer-window radii")
    wrapped = math.atan2(math.sin(delta), math.cos(delta))
    distance = abs(wrapped)
    boundary_epsilon = 8.0 * np.finfo(float).eps
    if distance <= start + boundary_epsilon:
        return 0.0, 0.0, 0.0
    if distance >= end - boundary_epsilon:
        return 1.0, 0.0, 0.0
    width = end - start
    t = (distance - start) / width
    value = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
    first_distance = 30.0 * t**2 * (1.0 - t)**2 / width
    second_distance = 60.0 * t * (1.0 - t) * (1.0 - 2.0 * t) / width**2
    return value, math.copysign(first_distance, wrapped), second_distance


@dataclass(frozen=True)
class AsymptoticPicCorrection:
    coefficient: float
    factors: tuple[AsymptoticPicFactor, ...]


def asymptotic_pic_bonded_energy_gradient(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    *,
    corrections: Sequence[AsymptoticPicCorrection] = (),
) -> tuple[float, np.ndarray]:
    """Bonded energy (Eh) and Cartesian gradient (Eh/angstrom)."""
    return _evaluate_asymptotic_pic(field, primitives, coordinates_angstrom,
                                    corrections=corrections, design_only=False)


def asymptotic_pic_correction_design(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    factors: Sequence[Sequence[AsymptoticPicFactor]],
) -> tuple[np.ndarray, np.ndarray]:
    """Linear E/G columns of the exact production correction functions.

    Includes every defining-bond switch and the Cartesian chain rule, using
    the same evaluator as the final field. Returns (terms,) and (3N, terms).
    No QM derivative above first order is needed to construct these columns.
    """
    corrections = tuple(AsymptoticPicCorrection(1., tuple(items)) for items in factors)
    return _evaluate_asymptotic_pic(field, primitives, coordinates_angstrom,
                                    corrections=corrections, design_only=True)


def _evaluate_asymptotic_pic(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    *,
    corrections: Sequence[AsymptoticPicCorrection] = (),
    design_only: bool = False,
) -> tuple[float, np.ndarray]:
    """Evaluate a ZAFF2 PIC field using reference-jet-preserving switches.

    Each support bond must carry its ZAFF-library radial scale.  For the
    permanent inverse-power form this is ``p/R0``; legacy Morse fields retain
    ``alpha``. Missing scales are rejected, so no arbitrary decay length is
    introduced.
    Gaussian electrostatics and Exp-PE are added by the caller unchanged.
    """

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("invalid asymptotic ZAFF geometry")
    q = eval_primitives(primitives, xyz)
    if len(primitives) != len(field.primitive_type_ids):
        raise ValueError("asymptotic ZAFF primitive catalog differs from fitted field")
    centers = field.coupling_reference_values
    if centers is None or not np.allclose(
        centers, field.equilibrium_values, atol=1.0e-10, rtol=0.0,
    ):
        raise ValueError("asymptotic ZAFF couplings need reference-centered coordinates")
    functions = {item.primitive_type_id: item for item in field.functional_parameters}
    torsions = {item.primitive_type_id: item for item in field.torsion_fourier_parameters}
    bond_index = {
        frozenset(primitive.atoms): index
        for index, primitive in enumerate(primitives) if primitive.kind == "bond"
    }
    bond_alpha: dict[int, float] = {}
    for index in bond_index.values():
        parameter = functions[field.primitive_type_ids[index]]
        values = dict(parameter.parameters)
        if parameter.form == "inverse_power_stretch":
            alpha = float(values.get("P", 0.0)) / float(values.get("R0", 0.0))
        else:
            alpha = float(values.get("ALPHA", 0.0))
        if (
            parameter.form
            not in {"inverse_power_stretch", "morse", "inverse_distance_quartic_switched"}
            or not np.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("asymptotic ZAFF support bond needs a library radial scale")
        bond_alpha[index] = alpha

    def support(index: int) -> tuple[int, ...]:
        primitive = primitives[index]
        atoms = primitive.atoms
        if primitive.kind == "bond":
            edges = (atoms,)
        elif primitive.kind in {"angle", "linear_bend"}:
            edges = ((atoms[0], atoms[1]), (atoms[1], atoms[2]))
        elif primitive.kind == "dihedral":
            edges = tuple(zip(atoms[:-1], atoms[1:]))
        elif primitive.kind == "out_of_plane":
            edges = tuple((atoms[0], atom) for atom in atoms[1:])
        else:
            raise ValueError(f"unsupported asymptotic ZAFF primitive: {primitive.kind}")
        try:
            return tuple(sorted({bond_index[frozenset(edge)] for edge in edges}))
        except KeyError as exc:
            raise ValueError("asymptotic ZAFF term lacks a defining covalent bond") from exc

    def switch(indices: tuple[int, ...]) -> tuple[float, dict[int, float]]:
        exponent = 0.0
        log_derivative: dict[int, float] = {}
        for index in indices:
            delta = float(q[index] - field.equilibrium_values[index])
            alpha = bond_alpha[index]
            scaled = alpha * delta
            exponent -= scaled**4
            log_derivative[index] = -4.0 * alpha * scaled**3
        factor = math.exp(exponent) if exponent > -745.0 else 0.0
        return factor, {index: factor * value for index, value in log_derivative.items()}

    def coupling_coordinate(index: int) -> tuple[float, float]:
        primitive = primitives[index]
        reference = float(centers[index])
        delta = float(q[index] - reference)
        if primitive.kind == "bond":
            parameter = functions[field.primitive_type_ids[index]]
            if parameter.form == "inverse_power_stretch":
                values = dict(parameter.parameters)
                r0 = float(values["R0"])
                power = float(values["P"])
                ratio = (r0 / float(q[index])) ** power
                ratio_reference = (r0 / reference) ** power
                derivative_reference = power * ratio_reference / reference
                return (
                    (ratio_reference - ratio) / derivative_reference,
                    power * ratio / (float(q[index]) * derivative_reference),
                )
            alpha = bond_alpha[index]
            exponential = math.exp(-alpha * delta)
            return -math.expm1(-alpha * delta) / alpha, exponential
        if primitive.kind == "angle":
            sine = math.sin(reference)
            if abs(sine) > 1.0e-6:
                return ((math.cos(q[index]) - math.cos(reference)) / -sine,
                        math.sin(q[index]) / sine)
        if primitive.kind in {"dihedral", "out_of_plane"}:
            return math.sin(delta), math.cos(delta)
        return delta, 1.0

    energy = 0.0
    gradient_q = np.zeros(len(primitives))
    if not design_only:
        for index, (primitive, type_id) in enumerate(
            zip(primitives, field.primitive_type_ids, strict=True)
        ):
            if primitive.kind == "dihedral":
                value, first, _second = evaluate_torsion_fourier_parameter(torsions[type_id], q[index])
            else:
                value, first, _second = evaluate_primitive_functional_parameter(functions[type_id], q[index])
            if primitive.kind == "bond":
                energy += value
                gradient_q[index] += first
            else:
                factor, derivatives = switch(support(index))
                energy += value * factor
                gradient_q[index] += first * factor
                for bond, derivative in derivatives.items():
                    gradient_q[bond] += value * derivative
        for coupling in field.retained_couplings:
            left, right = coupling.primitive_indices
            left_value, left_first = coupling_coordinate(left)
            right_value, right_first = coupling_coordinate(right)
            factor, derivatives = switch(tuple(sorted(set(support(left)) | set(support(right)))))
            k = coupling.force_constant
            energy += k * left_value * right_value * factor
            gradient_q[left] += k * left_first * right_value * factor
            gradient_q[right] += k * left_value * right_first * factor
            for bond, derivative in derivatives.items():
                gradient_q[bond] += k * left_value * right_value * derivative
    # Every fitted product carries the switch of every defining bond, including
    # bonds supporting angular factors. Its asymptote is independent of all
    # other fitted terms and does not rely on cancellation between ZAFF levels.
    design_energy = []
    design_gradient = []
    window_cache: dict[tuple[int, float, float], tuple[float, float, float]] = {}
    for term in corrections:
        if not term.factors or not np.isfinite(term.coefficient):
            raise ValueError("invalid asymptotic PIC correction")
        values, derivatives, indices = [], [], []
        # Dynamic continuation terms are exactly dormant in the protected
        # spectroscopic neighbourhood. Cache the few rotor windows and reject
        # dormant products before evaluating their other factors.
        outer = next((item for item in term.factors if item.form == "outer_window"), None)
        inactive = False
        if outer is not None:
            if (primitives[outer.primitive_index].kind != "dihedral"
                    or outer.power != 1 or outer.harmonic != 1):
                raise ValueError("outer window requires a unit-power dihedral factor")
            window_key = (outer.primitive_index, outer.window_start, outer.window_end)
            window_value = window_cache.get(window_key)
            if window_value is None:
                window_value = periodic_outer_window(
                    float(q[outer.primitive_index] - centers[outer.primitive_index]),
                    outer.window_start,
                    outer.window_end,
                )
                window_cache[window_key] = window_value
            inactive = inactive or (window_value[0] == 0.0 and window_value[1] == 0.0)
        if inactive:
            if design_only:
                design_energy.append(0.0)
                design_gradient.append(np.zeros(len(primitives)))
            continue
        for item in term.factors:
            index, power, harmonic = item.primitive_index, item.power, item.harmonic
            if (not 0 <= index < len(primitives) or power < 1 or harmonic < 1
                    or int(power) != power or int(harmonic) != harmonic):
                raise ValueError("invalid asymptotic PIC factor")
            delta = float(q[index] - centers[index])
            if item.form == "native":
                value, first = coupling_coordinate(index)
            elif item.form == "inverse_distance":
                if primitives[index].kind != "bond" or q[index] <= 0:
                    raise ValueError("inverse-distance correction requires a bond")
                reference = float(centers[index])
                value = reference**2 * (1.0 / reference - 1.0 / q[index])
                first = reference**2 / q[index]**2
            elif item.form == "sin":
                value, first = math.sin(harmonic * delta), harmonic * math.cos(harmonic * delta)
            elif item.form == "one_minus_cos":
                value, first = 1.0 - math.cos(harmonic * delta), harmonic * math.sin(harmonic * delta)
            elif item.form == "jet_cos":
                value = math.cos(harmonic * delta) - 1.0 + harmonic**2 * (1.0 - math.cos(delta))
                first = -harmonic * math.sin(harmonic * delta) + harmonic**2 * math.sin(delta)
            elif item.form == "jet_sin":
                value = math.sin(harmonic * delta) - harmonic * math.sin(delta)
                first = harmonic * (math.cos(harmonic * delta) - math.cos(delta))
            elif item.form == "outer_window":
                value, first, _second = window_cache[
                    (index, item.window_start, item.window_end)
                ]
            else:
                raise ValueError(f"unsupported asymptotic PIC factor: {item.form}")
            values.append(value**power)
            derivatives.append(power * value**(power - 1) * first)
            indices.append(index)
        factor, switch_derivatives = switch(tuple(sorted(set().union(
            *(support(index) for index in indices)
        ))))
        product = term.coefficient * math.prod(values)
        term_gradient = np.zeros(len(primitives))
        for position, index in enumerate(indices):
            term_gradient[index] += (factor * term.coefficient * derivatives[position]
                                  * math.prod(values[:position] + values[position + 1:]))
        for bond, derivative in switch_derivatives.items():
            term_gradient[bond] += product * derivative
        if design_only:
            design_energy.append(factor * product)
            design_gradient.append(term_gradient)
        else:
            energy += factor * product
            gradient_q += term_gradient
    if design_only:
        rows = np.asarray(design_gradient).reshape(len(corrections), len(primitives))
        return np.asarray(design_energy), primitive_b_matrix(primitives, xyz).T @ rows.T
    gradient = primitive_b_matrix(primitives, xyz).T @ gradient_q
    return float(energy), gradient
