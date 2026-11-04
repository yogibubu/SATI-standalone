"""Morse ZAFF residual evaluated on frozen SMITH primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from matrix_smith import (
    build_primitive_b_matrix,
    evaluate_primitive_values,
    gic_definition_from_dict,
)
from matrix_zaff.bonded import CenteredMorseStretchCoordinate
from matrix_zaff.geminal_coupling import ZaffGeminalXHCoupling
from matrix_zaff.primitive_second_derivative import analytic_primitive_second_derivative


BOHR_TO_ANGSTROM = 0.529177210903
PRIMITIVE_RESIDUAL_SCHEMA = "matrix.zaff.primitive_residual.v1"
TWO_MORSE_RESIDUAL_SCHEMA = "matrix.zaff.primitive_residual.v2"


@dataclass(frozen=True)
class PrimitiveResidualEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray | None


class PrimitiveResidualRuntime:
    """Evaluate fitted primitive E/G/H with Morse stretches and damped cross terms.

    ``q`` retains SMITH's native units: stretches in Angstrom and angular
    coordinates in radians. Each stretch diagonal is a shifted Morse form.
    Every cross term involving that stretch uses a bond-local radial factor
    that tends to zero upon dissociation.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.two_morse = payload.get("schema") == TWO_MORSE_RESIDUAL_SCHEMA
        if payload.get("schema") not in {PRIMITIVE_RESIDUAL_SCHEMA, TWO_MORSE_RESIDUAL_SCHEMA}:
            raise ValueError("unsupported primitive ZAFF residual schema")
        self.definition = gic_definition_from_dict(payload["smith_definition"])
        self.labels = tuple(str(label) for label in payload["primitive_labels"])
        available = {primitive.identifier: primitive for primitive in self.definition.primitives}
        if len(self.labels) != len(set(self.labels)) or set(self.labels) - set(available):
            raise ValueError("primitive ZAFF labels differ from frozen SMITH definition")
        count = len(self.labels)
        self.reference = np.asarray(payload["reference_values"], dtype=float)
        self.gradient = np.asarray(payload["primitive_gradient"], dtype=float)
        self.hessian = np.asarray(payload["primitive_hessian"], dtype=float)
        self.stretch_indices = np.asarray(
            [index for index, label in enumerate(self.labels)
             if available[label].family == "STRETCH"], dtype=int,
        )
        self.alpha_per_angstrom = np.asarray(payload["morse_alpha_per_angstrom"], dtype=float)
        if self.two_morse:
            self.qm_gradient = np.asarray(payload["qm_primitive_gradient"], dtype=float)
            self.gfn2_gradient = np.asarray(payload["gfn2_primitive_gradient"], dtype=float)
            self.qm_hessian = np.asarray(payload["qm_primitive_hessian"], dtype=float)
            self.gfn2_hessian = np.asarray(payload["gfn2_primitive_hessian"], dtype=float)
            self.gfn2_alpha_per_angstrom = np.asarray(
                payload["gfn2_morse_alpha_per_angstrom"], dtype=float,
            )
        self._stretch_mask = np.zeros(count, dtype=bool)
        self._stretch_mask[self.stretch_indices] = True
        if (
            self.reference.shape != (count,)
            or self.gradient.shape != (count,)
            or self.hessian.shape != (count, count)
            or not np.all(np.isfinite(self.reference))
            or not np.all(np.isfinite(self.gradient))
            or not np.all(np.isfinite(self.hessian))
            or not np.allclose(self.hessian, self.hessian.T, atol=1.0e-12, rtol=0.0)
            or self.alpha_per_angstrom.shape != (count,)
            or not np.all(np.isfinite(self.alpha_per_angstrom))
            or np.any(self.alpha_per_angstrom[self.stretch_indices] <= 0.0)
            or np.any(self.alpha_per_angstrom[~self._stretch_mask] != 0.0)
        ):
            raise ValueError("invalid primitive ZAFF residual parameters")
        if self.two_morse and (
            self.qm_gradient.shape != (count,)
            or self.gfn2_gradient.shape != (count,)
            or self.qm_hessian.shape != (count, count)
            or self.gfn2_hessian.shape != (count, count)
            or self.gfn2_alpha_per_angstrom.shape != (count,)
            or not np.all(np.isfinite(self.gfn2_alpha_per_angstrom))
            or np.any(self.gfn2_alpha_per_angstrom[self.stretch_indices] <= 0.0)
            or np.any(self.gfn2_alpha_per_angstrom[~self._stretch_mask] != 0.0)
            or not np.allclose(self.qm_gradient - self.gfn2_gradient, self.gradient, atol=1.0e-10, rtol=0.0)
            or not np.allclose(self.qm_hessian - self.gfn2_hessian, self.hessian, atol=1.0e-10, rtol=0.0)
        ):
            raise ValueError("two-Morse primitive fields do not reconstruct their residual")
        if self.two_morse:
            for gradient, hessian, alpha in (
                (self.qm_gradient, self.qm_hessian, self.alpha_per_angstrom),
                (self.gfn2_gradient, self.gfn2_hessian, self.gfn2_alpha_per_angstrom),
            ):
                for index in self.stretch_indices:
                    c = float(gradient[index])
                    k = float(hessian[index, index])
                    a = float(alpha[index])
                    if k + a * c <= 0.0 or k + 2.0 * a * c <= 0.0:
                        raise ValueError("bond derivatives do not define a bound Morse well")
        self._primitives = tuple(available[label] for label in self.labels)
        self._reference_coordinates = np.asarray(
            self.definition.reference_coordinates_angstrom, dtype=float,
        )
        self._morse = {
            int(index): CenteredMorseStretchCoordinate(
                float(self.reference[index]), float(self.alpha_per_angstrom[index]),
            )
            for index in self.stretch_indices
        }
        self._gfn2_morse = (
            {int(index): CenteredMorseStretchCoordinate(
                float(self.reference[index]), float(self.gfn2_alpha_per_angstrom[index]),
            ) for index in self.stretch_indices}
            if self.two_morse else {}
        )
        self._cross_terms = tuple(
            (left, right, float(self.hessian[left, right]))
            for left in range(count) for right in range(left + 1, count)
            if self.hessian[left, right] != 0.0
        )

    def _b(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        result = build_primitive_b_matrix(
            self.definition, coordinates_angstrom=coordinates_angstrom,
        )
        if tuple(result.coordinate_labels) != self.labels:
            raise ValueError("SMITH primitive derivative order changed")
        return np.asarray(result.rows, dtype=float) * BOHR_TO_ANGSTROM

    def _weighted_second(self, coordinates_angstrom: np.ndarray, weights: np.ndarray) -> np.ndarray:
        size = coordinates_angstrom.size
        result = np.zeros((size, size))
        for primitive, weight in zip(self._primitives, weights, strict=True):
            if abs(float(weight)) <= 1.0e-15:
                continue
            second = analytic_primitive_second_derivative(
                primitive, coordinates_angstrom,
                reference_coordinates=self._reference_coordinates, zero_tolerance=0.0,
            )
            for row, column, value in second.entries:
                result[row, column] += float(weight) * value * BOHR_TO_ANGSTROM**2
        return 0.5 * (result + result.T)

    def evaluate(self, coordinates_angstrom: np.ndarray, *, include_hessian: bool = True) -> PrimitiveResidualEvaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != np.asarray(self.definition.reference_coordinates_angstrom).shape or not np.all(np.isfinite(coordinates)):
            raise ValueError("primitive ZAFF evaluation geometry is invalid")
        values = evaluate_primitive_values(
            self.definition, primitive_ids=self.labels, coordinates_angstrom=coordinates,
        )
        delta = np.asarray([values[label] for label in self.labels]) - self.reference
        b = self._b(coordinates)
        count = len(self.labels)
        energy = 0.0
        q_gradient = np.zeros(count)
        q_hessian = np.zeros((count, count)) if include_hessian else None
        cross_value = delta.copy()
        cross_first = np.ones(count)
        cross_second = np.zeros(count)
        for index in range(count):
            displacement = float(delta[index])
            c = float(self.gradient[index])
            k = float(self.hessian[index, index])
            if self._stretch_mask[index]:
                alpha = float(self.alpha_per_angstrom[index])
                fields = (
                    (
                        (1.0, self.qm_gradient[index], self.qm_hessian[index, index],
                         alpha, self._morse[index]),
                        (-1.0, self.gfn2_gradient[index], self.gfn2_hessian[index, index],
                         self.gfn2_alpha_per_angstrom[index], self._gfn2_morse[index]),
                    ) if self.two_morse else ((1.0, c, k, alpha, self._morse[index]),)
                )
                for sign, local_c, local_k, local_alpha, coordinate in fields:
                    derivatives = coordinate.derivatives(float(self.reference[index] + displacement))
                    morse = derivatives.value
                    curvature = local_k + local_alpha * local_c
                    energy += sign * (local_c * morse + 0.5 * curvature * morse**2)
                    q_gradient[index] += sign * (local_c + curvature * morse) * derivatives.first
                    if q_hessian is not None:
                        q_hessian[index, index] += sign * (
                            curvature * derivatives.first**2
                            + (local_c + curvature * morse) * derivatives.second
                        )
                radial, radial_first, radial_second = ZaffGeminalXHCoupling._radial(
                    float(self.reference[index] / BOHR_TO_ANGSTROM + displacement / BOHR_TO_ANGSTROM),
                    float(self.reference[index] / BOHR_TO_ANGSTROM),
                    alpha * BOHR_TO_ANGSTROM,
                )
                cross_value[index] = radial * BOHR_TO_ANGSTROM
                cross_first[index] = radial_first
                cross_second[index] = radial_second / BOHR_TO_ANGSTROM
            else:
                energy += c * displacement + 0.5 * k * displacement**2
                q_gradient[index] += c + k * displacement
                if q_hessian is not None:
                    q_hessian[index, index] += k
        for left, right, k in self._cross_terms:
            energy += k * cross_value[left] * cross_value[right]
            q_gradient[left] += k * cross_first[left] * cross_value[right]
            q_gradient[right] += k * cross_value[left] * cross_first[right]
            if q_hessian is not None:
                q_hessian[left, left] += k * cross_second[left] * cross_value[right]
                q_hessian[right, right] += k * cross_value[left] * cross_second[right]
                q_hessian[left, right] += k * cross_first[left] * cross_first[right]
                q_hessian[right, left] = q_hessian[left, right]
        cartesian_gradient = b.T @ q_gradient
        cartesian_hessian = None
        if q_hessian is not None:
            cartesian_hessian = b.T @ q_hessian @ b + self._weighted_second(
                coordinates, q_gradient,
            )
            cartesian_hessian = 0.5 * (cartesian_hessian + cartesian_hessian.T)
        return PrimitiveResidualEvaluation(energy, cartesian_gradient, cartesian_hessian)


__all__ = ["PRIMITIVE_RESIDUAL_SCHEMA", "TWO_MORSE_RESIDUAL_SCHEMA", "PrimitiveResidualRuntime"]
