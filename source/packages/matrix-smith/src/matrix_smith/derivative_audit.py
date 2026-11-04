"""Finite-difference audits of the production analytic SONIC Wilson matrix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .evaluation import build_gic_b_matrix, evaluate_gic_values
from .models import GICDefinition
from .second_derivatives import analytic_primitive_hessian


@dataclass(frozen=True)
class GICDerivativeAuditRow:
    identifier: str
    name: str
    family: str
    max_abs_error: float
    rms_error: float
    passed: bool


@dataclass(frozen=True)
class GICDerivativeAudit:
    step_angstrom: float
    tolerance: float
    rows: tuple[GICDerivativeAuditRow, ...]
    max_abs_error: float
    passed: bool

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({row.family for row in self.rows}))


@dataclass(frozen=True)
class GICDerivativeEnsembleMember:
    index: int
    displacement_rms_angstrom: float
    max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class GICDerivativeEnsembleAudit:
    seed: int
    perturbation_scale_angstrom: float
    members: tuple[GICDerivativeEnsembleMember, ...]
    max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class SOOPDerivativeAuditRow:
    """Derivative errors for one frozen SOOP coordinate at one geometry."""

    geometry: str
    identifier: str
    max_gradient_error: float
    max_hessian_error: float
    max_hessian_asymmetry: float
    passed: bool


@dataclass(frozen=True)
class SOOPDerivativeAudit:
    """Finite-difference validation of the linear frozen-SOOP contract."""

    step_angstrom: float
    gradient_tolerance: float
    hessian_tolerance: float
    rows: tuple[SOOPDerivativeAuditRow, ...]
    max_gradient_error: float
    max_hessian_error: float
    passed: bool

    def scientific_validation_record(self):
        """Return the common cross-tool telemetry envelope for this audit."""

        from matrix_core import ScientificValidationRecord

        return ScientificValidationRecord(
            owner="SMITH",
            subject="SOOP_ANALYTIC_DERIVATIVES",
            passed=self.passed,
            metrics={
                "max_gradient_error": self.max_gradient_error,
                "max_hessian_error": self.max_hessian_error,
            },
            thresholds={
                "gradient_tolerance": self.gradient_tolerance,
                "hessian_tolerance": self.hessian_tolerance,
            },
            context={
                "step_angstrom": self.step_angstrom,
                "geometry_count": len({row.geometry for row in self.rows}),
                "coordinate_count": len(self.rows),
            },
        )


def audit_gic_b_matrix_derivatives(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | None = None,
    *,
    step_angstrom: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> GICDerivativeAudit:
    """Compare every analytic SONIC B row with a Cartesian central difference.

    The audit differentiates the complete frozen SONIC definition rather than
    isolated primitives, so it also covers SALCs and ring-coordinate linear
    combinations.  Coordinate differences are reduced to the principal
    angular branch; Cartesian perturbations are sufficiently small that this
    is a no-op for distance-like coordinates.
    """
    step = float(step_angstrom)
    threshold = float(tolerance)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("SONIC derivative-audit step must be positive and finite")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("SONIC derivative-audit tolerance must be positive and finite")
    coords = np.asarray(
        definition.reference_coordinates_angstrom
        if coordinates_angstrom is None
        else coordinates_angstrom,
        dtype=float,
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("SONIC derivative-audit coordinates must have shape (natoms, 3)")

    analytic = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=coords).rows,
        dtype=float,
    )
    numeric = np.zeros_like(analytic)
    for column in range(coords.size):
        plus = coords.reshape(-1).copy()
        minus = coords.reshape(-1).copy()
        plus[column] += step
        minus[column] -= step
        plus_values = evaluate_gic_values(
            definition, coordinates_angstrom=plus.reshape(coords.shape)
        )
        minus_values = evaluate_gic_values(
            definition, coordinates_angstrom=minus.reshape(coords.shape)
        )
        delta = np.asarray(plus_values - minus_values, dtype=float)
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        numeric[:, column] = delta / (2.0 * step)

    row_errors = analytic - numeric
    rows = tuple(
        GICDerivativeAuditRow(
            identifier=gic.identifier,
            name=gic.name,
            family=gic.family,
            max_abs_error=float(np.max(np.abs(error), initial=0.0)),
            rms_error=float(np.sqrt(np.mean(error * error))) if error.size else 0.0,
            passed=bool(np.max(np.abs(error), initial=0.0) <= threshold),
        )
        for gic, error in zip(definition.gics, row_errors, strict=True)
    )
    maximum = max((row.max_abs_error for row in rows), default=0.0)
    return GICDerivativeAudit(
        step_angstrom=step,
        tolerance=threshold,
        rows=rows,
        max_abs_error=maximum,
        passed=all(row.passed for row in rows),
    )


def audit_gic_b_matrix_ensemble(
    definition: GICDefinition,
    *,
    sample_count: int = 8,
    perturbation_scale_angstrom: float = 2.0e-4,
    seed: int = 1729,
    step_angstrom: float = 1.0e-6,
    tolerance: float = 2.0e-5,
) -> GICDerivativeEnsembleAudit:
    """Audit B rows at deterministic, randomly perturbed geometries.

    One fixed seed makes failures reproducible across machines. Rigid
    translation is removed from every displacement so the samples exercise
    internal-coordinate derivatives rather than a changing origin.
    """
    count = int(sample_count)
    scale = float(perturbation_scale_angstrom)
    if count < 1 or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("derivative ensemble requires positive count and scale")
    reference = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rng = np.random.default_rng(int(seed))
    members: list[GICDerivativeEnsembleMember] = []
    for index in range(count):
        displacement = rng.normal(0.0, scale, size=reference.shape)
        displacement -= np.mean(displacement, axis=0, keepdims=True)
        audit = audit_gic_b_matrix_derivatives(
            definition,
            reference + displacement,
            step_angstrom=step_angstrom,
            tolerance=tolerance,
        )
        members.append(GICDerivativeEnsembleMember(
            index=index,
            displacement_rms_angstrom=float(np.sqrt(np.mean(displacement**2))),
            max_abs_error=audit.max_abs_error,
            passed=audit.passed,
        ))
    maximum = max(member.max_abs_error for member in members)
    return GICDerivativeEnsembleAudit(
        seed=int(seed),
        perturbation_scale_angstrom=scale,
        members=tuple(members),
        max_abs_error=maximum,
        passed=all(member.passed for member in members),
    )


def audit_soop_derivatives(
    definition: GICDefinition,
    geometries_angstrom: Mapping[str, np.ndarray] | None = None,
    *,
    step_angstrom: float = 1.0e-6,
    gradient_tolerance: float = 1.0e-8,
    hessian_tolerance: float = 1.0e-8,
) -> SOOPDerivativeAudit:
    """Audit frozen SOOP gradients and B-prime at one or more geometries.

    The selected GICs must be composed exclusively of SOOP primitives. Their
    analytic Hessian is assembled from the public primitive second-derivative
    kernel and compared with central differences of the production Wilson
    matrix. Multiple named geometries make reference-planar and displaced-chart
    checks part of one reproducible audit.
    """

    step = float(step_angstrom)
    gradient_threshold = float(gradient_tolerance)
    hessian_threshold = float(hessian_tolerance)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("SOOP derivative-audit step must be positive and finite")
    if not np.isfinite(gradient_threshold) or gradient_threshold <= 0.0:
        raise ValueError("SOOP gradient tolerance must be positive and finite")
    if not np.isfinite(hessian_threshold) or hessian_threshold <= 0.0:
        raise ValueError("SOOP Hessian tolerance must be positive and finite")

    primitive_by_id = {
        primitive.identifier: primitive for primitive in definition.primitives
    }
    soop_rows = []
    for index, gic in enumerate(definition.gics):
        components = gic.coefficients or ((gic.primitive_id, 1.0),)
        if components and all(
            primitive_by_id[primitive_id].function == "SOOP"
            for primitive_id, _coefficient in components
        ):
            soop_rows.append((index, gic, components))
    if not soop_rows:
        raise ValueError("SOOP derivative audit requires at least one SOOP-only GIC")

    if geometries_angstrom is None:
        geometries_angstrom = {
            "reference": np.asarray(
                definition.reference_coordinates_angstrom, dtype=float
            )
        }
    rows: list[SOOPDerivativeAuditRow] = []
    expected_shape = np.asarray(definition.reference_coordinates_angstrom).shape
    for geometry_name, coordinates in geometries_angstrom.items():
        coords = np.asarray(coordinates, dtype=float)
        if (
            coords.shape != expected_shape
            or coords.ndim != 2
            or coords.shape[1] != 3
            or not np.all(np.isfinite(coords))
        ):
            raise ValueError(
                f"SOOP audit geometry {geometry_name!r} must be finite with shape "
                f"{expected_shape}"
            )
        analytic_b = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=coords).rows,
            dtype=float,
        )
        numeric_b = np.zeros_like(analytic_b)
        numeric_hessians = np.zeros(
            (len(definition.gics), coords.size, coords.size), dtype=float
        )
        for column in range(coords.size):
            plus = coords.reshape(-1).copy()
            minus = coords.reshape(-1).copy()
            plus[column] += step
            minus[column] -= step
            plus_coords = plus.reshape(coords.shape)
            minus_coords = minus.reshape(coords.shape)
            plus_values = np.asarray(
                evaluate_gic_values(definition, coordinates_angstrom=plus_coords),
                dtype=float,
            )
            minus_values = np.asarray(
                evaluate_gic_values(definition, coordinates_angstrom=minus_coords),
                dtype=float,
            )
            numeric_b[:, column] = (plus_values - minus_values) / (2.0 * step)
            plus_b = np.asarray(
                build_gic_b_matrix(definition, coordinates_angstrom=plus_coords).rows,
                dtype=float,
            )
            minus_b = np.asarray(
                build_gic_b_matrix(definition, coordinates_angstrom=minus_coords).rows,
                dtype=float,
            )
            numeric_hessians[:, :, column] = (plus_b - minus_b) / (2.0 * step)

        for index, gic, components in soop_rows:
            analytic_hessian = np.zeros((coords.size, coords.size), dtype=float)
            for primitive_id, coefficient in components:
                analytic_hessian += float(coefficient) * analytic_primitive_hessian(
                    primitive_by_id[primitive_id], coords
                )
            gradient_error = float(
                np.max(np.abs(analytic_b[index] - numeric_b[index]), initial=0.0)
            )
            hessian_error = float(
                np.max(
                    np.abs(analytic_hessian - numeric_hessians[index]), initial=0.0
                )
            )
            asymmetry = float(
                np.max(
                    np.abs(numeric_hessians[index] - numeric_hessians[index].T),
                    initial=0.0,
                )
            )
            rows.append(
                SOOPDerivativeAuditRow(
                    geometry=str(geometry_name),
                    identifier=gic.identifier,
                    max_gradient_error=gradient_error,
                    max_hessian_error=hessian_error,
                    max_hessian_asymmetry=asymmetry,
                    passed=(
                        gradient_error <= gradient_threshold
                        and hessian_error <= hessian_threshold
                        and asymmetry <= hessian_threshold
                    ),
                )
            )

    max_gradient_error = max(
        (row.max_gradient_error for row in rows), default=0.0
    )
    max_hessian_error = max((row.max_hessian_error for row in rows), default=0.0)
    return SOOPDerivativeAudit(
        step_angstrom=step,
        gradient_tolerance=gradient_threshold,
        hessian_tolerance=hessian_threshold,
        rows=tuple(rows),
        max_gradient_error=max_gradient_error,
        max_hessian_error=max_hessian_error,
        passed=all(row.passed for row in rows),
    )
