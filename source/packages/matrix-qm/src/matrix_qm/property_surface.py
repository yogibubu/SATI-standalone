"""Backend-neutral multi-component property surfaces with constrained SVD fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .surface_fit import (
    ArchitectLinearConstraint,
    ArchitectLinearFitProblem,
    ArchitectLinearFitResult,
    fit_architect_linear_surface,
)


ARCHITECT_PROPERTY_SURFACE_PROBLEM_SCHEMA = (
    "matrix.architect.property_surface.problem.v1"
)
ARCHITECT_PROPERTY_SURFACE_RESULT_SCHEMA = (
    "matrix.architect.property_surface.result.v1"
)
_QUALIFIER = "::"


@dataclass(frozen=True)
class ArchitectPropertyComponent:
    """Observed values and fit controls for one property component."""

    label: str
    observations: np.ndarray
    units: str = ""
    base_weights: np.ndarray | None = None
    weight_components: Mapping[str, np.ndarray] = field(default_factory=dict)
    fixed_coefficients: Mapping[str, float] = field(default_factory=dict)
    constraints: tuple[ArchitectLinearConstraint, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label = str(self.label)
        observations = np.asarray(self.observations, dtype=float).reshape(-1)
        if not label or _QUALIFIER in label:
            raise ValueError("property-component labels must be nonempty and cannot contain '::'")
        if not len(observations) or not np.all(np.isfinite(observations)):
            raise ValueError("property observations must be nonempty and finite")
        base_weights = (
            np.ones(len(observations), dtype=float)
            if self.base_weights is None
            else np.asarray(self.base_weights, dtype=float).reshape(-1)
        )
        if base_weights.shape != observations.shape:
            raise ValueError("component base weights must cover every observation")
        weight_components = {
            str(name): np.asarray(values, dtype=float).reshape(-1)
            for name, values in self.weight_components.items()
        }
        if any(values.shape != observations.shape for values in weight_components.values()):
            raise ValueError("component weight factors must cover every observation")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "base_weights", base_weights)
        object.__setattr__(self, "weight_components", weight_components)
        object.__setattr__(
            self,
            "fixed_coefficients",
            {str(name): float(value) for name, value in self.fixed_coefficients.items()},
        )
        object.__setattr__(self, "constraints", tuple(self.constraints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "observations": self.observations.tolist(),
            "units": self.units,
            "base_weights": np.asarray(self.base_weights).tolist(),
            "weight_components": {
                name: values.tolist() for name, values in self.weight_components.items()
            },
            "fixed_coefficients": dict(self.fixed_coefficients),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectPropertyComponent":
        return cls(
            label=str(payload["label"]),
            observations=np.asarray(payload["observations"], dtype=float),
            units=str(payload.get("units", "")),
            base_weights=np.asarray(payload["base_weights"], dtype=float),
            weight_components={
                str(name): np.asarray(values, dtype=float)
                for name, values in payload.get("weight_components", {}).items()
            },
            fixed_coefficients={
                str(name): float(value)
                for name, value in payload.get("fixed_coefficients", {}).items()
            },
            constraints=tuple(
                ArchitectLinearConstraint.from_dict(value)
                for value in payload.get("constraints", ())
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ArchitectPropertySurfaceProblem:
    """A scalar, vector or tensor property surface on a common analytic basis.

    Component-local constraints use unqualified basis labels. Coupled constraints
    use ``component::basis`` labels and can therefore impose exact relations
    between Cartesian or tensor components.
    """

    property_name: str
    representation: str
    basis_labels: tuple[str, ...]
    design_matrix: np.ndarray
    components: tuple[ArchitectPropertyComponent, ...]
    point_labels: tuple[str, ...] = ()
    coupled_constraints: tuple[ArchitectLinearConstraint, ...] = ()
    rcond: float = 1.0e-12
    uncertainty_scale: str = "absolute"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ARCHITECT_PROPERTY_SURFACE_PROBLEM_SCHEMA

    def __post_init__(self) -> None:
        basis_labels = tuple(str(value) for value in self.basis_labels)
        design = np.asarray(self.design_matrix, dtype=float)
        components = tuple(self.components)
        if not self.property_name:
            raise ValueError("a property surface needs a property name")
        if not basis_labels or len(set(basis_labels)) != len(basis_labels):
            raise ValueError("property basis labels must be unique and nonempty")
        if any(_QUALIFIER in label for label in basis_labels):
            raise ValueError("property basis labels cannot contain '::'")
        if design.ndim != 2 or design.shape[1] != len(basis_labels):
            raise ValueError("property design matrix differs from the declared basis")
        if not np.all(np.isfinite(design)):
            raise ValueError("property design matrix must be finite")
        if not components or len({component.label for component in components}) != len(
            components
        ):
            raise ValueError("property component labels must be unique and nonempty")
        if any(len(component.observations) != design.shape[0] for component in components):
            raise ValueError("every property component must cover every point")
        points = self.point_labels or tuple(
            f"point_{index:06d}" for index in range(design.shape[0])
        )
        if len(points) != design.shape[0] or len(set(points)) != len(points):
            raise ValueError("point labels must be unique and cover the design matrix")
        known_basis = set(basis_labels)
        for component in components:
            unknown = set(component.fixed_coefficients) - known_basis
            unknown.update(
                name
                for constraint in component.constraints
                for name in constraint.coefficients
                if name not in known_basis
            )
            if unknown:
                raise ValueError(
                    f"unknown basis labels for component {component.label}: {sorted(unknown)}"
                )
        qualified = {
            _qualified(component.label, basis)
            for component in components
            for basis in basis_labels
        }
        unknown_coupled = {
            name
            for constraint in self.coupled_constraints
            for name in constraint.coefficients
            if name not in qualified
        }
        if unknown_coupled:
            raise ValueError(
                f"unknown qualified property coefficients: {sorted(unknown_coupled)}"
            )
        object.__setattr__(self, "basis_labels", basis_labels)
        object.__setattr__(self, "design_matrix", design)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "point_labels", tuple(str(value) for value in points))
        object.__setattr__(self, "coupled_constraints", tuple(self.coupled_constraints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "property_name": self.property_name,
            "representation": self.representation,
            "basis_labels": list(self.basis_labels),
            "design_matrix": self.design_matrix.tolist(),
            "components": [component.to_dict() for component in self.components],
            "point_labels": list(self.point_labels),
            "coupled_constraints": [
                constraint.to_dict() for constraint in self.coupled_constraints
            ],
            "rcond": self.rcond,
            "uncertainty_scale": self.uncertainty_scale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectPropertySurfaceProblem":
        if payload.get("schema") != ARCHITECT_PROPERTY_SURFACE_PROBLEM_SCHEMA:
            raise ValueError("unsupported ARCHITECT property-surface problem schema")
        return cls(
            property_name=str(payload["property_name"]),
            representation=str(payload.get("representation", "general")),
            basis_labels=tuple(payload["basis_labels"]),
            design_matrix=np.asarray(payload["design_matrix"], dtype=float),
            components=tuple(
                ArchitectPropertyComponent.from_dict(value)
                for value in payload["components"]
            ),
            point_labels=tuple(payload.get("point_labels", ())),
            coupled_constraints=tuple(
                ArchitectLinearConstraint.from_dict(value)
                for value in payload.get("coupled_constraints", ())
            ),
            rcond=float(payload.get("rcond", 1.0e-12)),
            uncertainty_scale=str(payload.get("uncertainty_scale", "absolute")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ArchitectPropertySurfaceResult:
    """Fitted multi-component property surface and its numerical audit."""

    property_name: str
    representation: str
    basis_labels: tuple[str, ...]
    component_labels: tuple[str, ...]
    component_units: tuple[str, ...]
    coefficients: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray
    point_labels: tuple[str, ...]
    linear_fit: ArchitectLinearFitResult
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ARCHITECT_PROPERTY_SURFACE_RESULT_SCHEMA

    @property
    def coefficient_maps(self) -> dict[str, dict[str, float]]:
        return {
            component: dict(
                zip(self.basis_labels, map(float, self.coefficients[index]), strict=True)
            )
            for index, component in enumerate(self.component_labels)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "property_name": self.property_name,
            "representation": self.representation,
            "basis_labels": list(self.basis_labels),
            "component_labels": list(self.component_labels),
            "component_units": list(self.component_units),
            "coefficients": self.coefficients.tolist(),
            "coefficient_maps": self.coefficient_maps,
            "predicted": self.predicted.tolist(),
            "residuals": self.residuals.tolist(),
            "point_labels": list(self.point_labels),
            "linear_fit": self.linear_fit.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectPropertySurfaceResult":
        if payload.get("schema") != ARCHITECT_PROPERTY_SURFACE_RESULT_SCHEMA:
            raise ValueError("unsupported ARCHITECT property-surface result schema")
        return cls(
            property_name=str(payload["property_name"]),
            representation=str(payload["representation"]),
            basis_labels=tuple(payload["basis_labels"]),
            component_labels=tuple(payload["component_labels"]),
            component_units=tuple(payload["component_units"]),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            predicted=np.asarray(payload["predicted"], dtype=float),
            residuals=np.asarray(payload["residuals"], dtype=float),
            point_labels=tuple(payload["point_labels"]),
            linear_fit=ArchitectLinearFitResult.from_dict(payload["linear_fit"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ArchitectPropertyEvaluation:
    """Property values and optional coordinate derivatives."""

    values: np.ndarray
    jacobian: np.ndarray | None = None
    hessian: np.ndarray | None = None


def fit_architect_property_surface(
    problem: ArchitectPropertySurfaceProblem,
) -> ArchitectPropertySurfaceResult:
    """Fit all components together, including optional cross-component constraints."""

    component_count = len(problem.components)
    point_count, basis_count = problem.design_matrix.shape
    coefficient_labels = tuple(
        _qualified(component.label, basis)
        for component in problem.components
        for basis in problem.basis_labels
    )
    block_design = np.zeros(
        (component_count * point_count, component_count * basis_count), dtype=float
    )
    for index in range(component_count):
        rows = slice(index * point_count, (index + 1) * point_count)
        columns = slice(index * basis_count, (index + 1) * basis_count)
        block_design[rows, columns] = problem.design_matrix
    observations = np.concatenate(
        [component.observations for component in problem.components]
    )
    observation_labels = tuple(
        _qualified(component.label, point)
        for component in problem.components
        for point in problem.point_labels
    )
    base_weights = np.concatenate(
        [np.asarray(component.base_weights) for component in problem.components]
    )
    weight_names = sorted(
        {name for component in problem.components for name in component.weight_components}
    )
    weight_components = {
        name: np.concatenate(
            [
                component.weight_components.get(
                    name, np.ones(point_count, dtype=float)
                )
                for component in problem.components
            ]
        )
        for name in weight_names
    }
    fixed = {
        _qualified(component.label, basis): value
        for component in problem.components
        for basis, value in component.fixed_coefficients.items()
    }
    constraints = [
        ArchitectLinearConstraint(
            name=_qualified(component.label, constraint.name),
            coefficients={
                _qualified(component.label, basis): value
                for basis, value in constraint.coefficients.items()
            },
            target=constraint.target,
        )
        for component in problem.components
        for constraint in component.constraints
    ]
    constraints.extend(problem.coupled_constraints)
    linear_problem = ArchitectLinearFitProblem(
        coefficient_labels=coefficient_labels,
        design_matrix=block_design,
        observations=observations,
        observation_labels=observation_labels,
        base_weights=base_weights,
        weight_components=weight_components,
        fixed_coefficients=fixed,
        constraints=tuple(constraints),
        rcond=problem.rcond,
        uncertainty_scale=problem.uncertainty_scale,
        metadata={
            **dict(problem.metadata),
            "surface_kind": "property",
            "property_name": problem.property_name,
            "representation": problem.representation,
        },
    )
    linear_result = fit_architect_linear_surface(linear_problem)
    coefficients = linear_result.coefficients.reshape(component_count, basis_count)
    predicted = linear_result.predicted.reshape(component_count, point_count)
    residuals = linear_result.residuals.reshape(component_count, point_count)
    return ArchitectPropertySurfaceResult(
        property_name=problem.property_name,
        representation=problem.representation,
        basis_labels=problem.basis_labels,
        component_labels=tuple(component.label for component in problem.components),
        component_units=tuple(component.units for component in problem.components),
        coefficients=coefficients,
        predicted=predicted,
        residuals=residuals,
        point_labels=problem.point_labels,
        linear_fit=linear_result,
        metadata=dict(problem.metadata),
    )


def evaluate_architect_property_surface(
    surface: ArchitectPropertySurfaceResult,
    basis_values: Sequence[float] | np.ndarray,
    *,
    basis_jacobian: np.ndarray | None = None,
    basis_hessian: np.ndarray | None = None,
) -> ArchitectPropertyEvaluation:
    """Evaluate a property surface and optional analytic basis derivatives."""

    values = np.asarray(basis_values, dtype=float)
    if values.shape != (len(surface.basis_labels),):
        raise ValueError("basis values do not match the fitted property basis")
    property_values = surface.coefficients @ values
    jacobian = None
    if basis_jacobian is not None:
        basis_jacobian = np.asarray(basis_jacobian, dtype=float)
        if basis_jacobian.ndim != 2 or basis_jacobian.shape[0] != len(
            surface.basis_labels
        ):
            raise ValueError("basis Jacobian must have shape (basis, coordinates)")
        jacobian = surface.coefficients @ basis_jacobian
    hessian = None
    if basis_hessian is not None:
        basis_hessian = np.asarray(basis_hessian, dtype=float)
        if basis_hessian.ndim != 3 or basis_hessian.shape[0] != len(
            surface.basis_labels
        ):
            raise ValueError("basis Hessian must have shape (basis, coordinates, coordinates)")
        hessian = np.einsum("cb,bij->cij", surface.coefficients, basis_hessian)
    return ArchitectPropertyEvaluation(
        values=property_values,
        jacobian=jacobian,
        hessian=hessian,
    )


def _qualified(component: str, basis: str) -> str:
    return f"{component}{_QUALIFIER}{basis}"


# Canonical backend-neutral API. Architect-prefixed names are retained as
# compatibility names for the established serialized surface schemas.
PropertyComponent = ArchitectPropertyComponent
PropertySurfaceProblem = ArchitectPropertySurfaceProblem
PropertySurfaceResult = ArchitectPropertySurfaceResult
PropertyEvaluation = ArchitectPropertyEvaluation
fit_property_surface = fit_architect_property_surface
evaluate_property_surface = evaluate_architect_property_surface
