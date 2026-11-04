"""Opt-in local half-angle coordinates; existing SONIC defaults are unchanged.

Angles and Gaussian trigonometric expressions use radians. A frozen reference
and an explicitly monitored branch are required: this is not a global angular
coordinate and is not an automatic Hessian-recentering policy.
"""

from dataclasses import dataclass
from numbers import Integral
import re

import numpy as np


@dataclass(frozen=True)
class HalfAngleCoordinate:
    reference_radian: float
    harmonic: int = 1
    minimum_slope: float = 0.25

    def __post_init__(self):
        if not np.isfinite(self.reference_radian):
            raise ValueError("reference angle must be finite")
        if isinstance(self.harmonic, bool) or not isinstance(self.harmonic, Integral):
            raise ValueError("harmonic must be a positive integer")
        if self.harmonic < 1:
            raise ValueError("harmonic must be a positive integer")
        if not 0.0 < self.minimum_slope < 1.0:
            raise ValueError("minimum_slope must lie strictly between zero and one")

    def evaluate(self, angle_radian):
        """Return value, first derivative and second derivative in angle space."""
        angle = np.asarray(angle_radian, dtype=float)
        phase = self.harmonic * (angle - self.reference_radian) / 2.0
        slope = np.cos(phase)
        if (
            not np.all(np.isfinite(phase))
            or np.any(np.abs(phase) >= np.pi / 2)
            or np.any(slope < self.minimum_slope)
        ):
            raise ValueError("half-angle branch exhausted; rebuild the local reference/model")
        return (
            2.0 / self.harmonic * np.sin(phase),
            slope,
            -self.harmonic / 2.0 * np.sin(phase),
        )

    def inverse(self, value):
        value = np.asarray(value, dtype=float)
        argument = self.harmonic * value / 2.0
        if not np.all(np.isfinite(argument)) or np.any(np.abs(argument) >= 1):
            raise ValueError("value outside invertible half-angle branch")
        angle = self.reference_radian + 2.0 / self.harmonic * np.arcsin(argument)
        self.evaluate(angle)
        return angle

    def evaluate_periodic(self, angle_radian):
        """Center a principal dihedral on the reference before taking its half.

        This removes the original D=+/-pi cut from the local neighborhood.
        The remaining branch boundary is opposite the frozen reference.
        """
        angle = np.asarray(angle_radian, dtype=float)
        delta = angle - self.reference_radian
        relative = np.arctan2(np.sin(delta), np.cos(delta))
        return self.evaluate(self.reference_radian + relative)

    def transform_derivatives(self, angle_radian, b_row, b_prime=None):
        """Apply the exact Cartesian chain rule to one scalar angular row."""
        value, slope, curvature = self.evaluate(float(angle_radian))
        row = np.asarray(b_row, dtype=float)
        if row.ndim != 1 or not np.all(np.isfinite(row)):
            raise ValueError("B row must be a finite vector")
        transformed_prime = None
        if b_prime is not None:
            second = np.asarray(b_prime, dtype=float)
            if second.shape != (row.size, row.size) or not np.all(np.isfinite(second)):
                raise ValueError("B-prime must be a finite square matrix matching B")
            transformed_prime = slope * second + curvature * np.outer(row, row)
        return float(value), slope * row, transformed_prime

    def gaussian_expression(self, primitive_expression):
        # Restrict the opt-in adapter to the scalar D/F flap contract; do not
        # silently transform already mixed, scaled or height-based ring rows.
        match = re.fullmatch(
            r"[DF]\(\s*([1-9]\d*)\s*,\s*([1-9]\d*)\s*,\s*"
            r"([1-9]\d*)\s*,\s*([1-9]\d*)\s*\)",
            primitive_expression.strip(),
            flags=re.IGNORECASE,
        )
        if match is None or len(set(match.groups())) != 4:
            raise ValueError("half-angle export requires a four-distinct-atom scalar flap")
        primitive = "D(" + ",".join(match.groups()) + ")"
        delta = f"{primitive}-({self.reference_radian:.17g})"
        centered = f"ATAN2(SIN({delta}),COS({delta}))"
        return (
            f"({2.0 / self.harmonic:.17g})*SIN("
            f"({self.harmonic / 2.0:.17g})*"
            f"({centered}))"
        )


def half_angle_ring_gic_lines(
    lines,
    reference_coordinates_angstrom,
    *,
    harmonic=1,
    minimum_slope=0.25,
    ring_labels=None,
):
    """Replace scalar flap expressions in an existing SMITH Gaussian export.

    By default select SMITH's RPck labels. Explicit labels support other export
    naming schemes. Preserve labels, order, activity/frozen flags and every
    non-ring line. Return lines plus immutable coordinate specifications.

    No QM calculation is launched. Gaussian evaluates the expressions, but
    does not enforce minimum_slope; callers must monitor this branch guard.
    """
    from .numerics import _dihedral_value

    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise ValueError("reference coordinates must be finite with shape (natoms, 3)")
    requested = None if ring_labels is None else set(ring_labels)
    result, specifications = [], {}
    for line in lines:
        left, separator, right = line.partition("=")
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9]*)(?:\(([^)]*)\))?\s*", left)
        label = match.group(1) if match else ""
        selected = label in requested if requested is not None else label.startswith("RPck")
        if not selected:
            result.append(line)
            continue
        if not separator or label in specifications:
            raise ValueError("missing or duplicate selected ring definition")
        primitive_match = re.fullmatch(
            r"\s*[DF]\((\d+),(\d+),(\d+),(\d+)\)\s*", right, re.IGNORECASE
        )
        if primitive_match is None:
            raise ValueError(f"{label}: expected an unmixed scalar D/F flap")
        atoms = tuple(int(atom) for atom in primitive_match.groups())
        if min(atoms) < 1 or max(atoms) > len(coords) or len(set(atoms)) != 4:
            raise ValueError(f"{label}: invalid flap atom indices")
        p, j, k, c = coords[np.asarray(atoms) - 1]
        hinge = k - j
        if (
            np.linalg.norm(hinge) < 1e-12
            or np.linalg.norm(np.cross(p - j, hinge)) < 1e-12
            or np.linalg.norm(np.cross(c - k, hinge)) < 1e-12
        ):
            raise ValueError(f"{label}: degenerate flap geometry")
        transform = HalfAngleCoordinate(
            _dihedral_value(coords, atoms), harmonic, minimum_slope
        )
        specifications[label] = transform
        result.append(f"{left.strip()} = {transform.gaussian_expression(right.strip())}")
    if not specifications or (requested is not None and requested != set(specifications)):
        raise ValueError("not all requested scalar ring coordinates were found")
    return result, specifications
