"""Backend-neutral weighted, exactly constrained linear surface fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


ARCHITECT_LINEAR_FIT_PROBLEM_SCHEMA = "matrix.architect.linear_fit.problem.v1"
ARCHITECT_LINEAR_FIT_RESULT_SCHEMA = "matrix.architect.linear_fit.result.v1"


@dataclass(frozen=True)
class ArchitectLinearConstraint:
    """One exact sparse equality between named surface coefficients."""

    name: str
    coefficients: Mapping[str, float]
    target: float

    def __post_init__(self) -> None:
        values = {str(key): float(value) for key, value in self.coefficients.items()}
        if not self.name or not values:
            raise ValueError("a linear constraint needs a name and coefficients")
        if not all(isfinite(value) for value in (*values.values(), float(self.target))):
            raise ValueError("linear constraints must be finite")
        if not any(value != 0.0 for value in values.values()):
            raise ValueError("a linear constraint cannot have an all-zero row")
        object.__setattr__(self, "coefficients", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coefficients": dict(self.coefficients),
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectLinearConstraint":
        return cls(
            name=str(payload["name"]),
            coefficients={
                str(key): float(value)
                for key, value in payload["coefficients"].items()
            },
            target=float(payload["target"]),
        )


@dataclass(frozen=True)
class ArchitectLinearFitProblem:
    """A weighted surface fit with fixed coefficients and exact equalities."""

    coefficient_labels: tuple[str, ...]
    design_matrix: np.ndarray
    observations: np.ndarray
    observation_labels: tuple[str, ...] = ()
    base_weights: np.ndarray | None = None
    weight_components: Mapping[str, np.ndarray] = field(default_factory=dict)
    fixed_coefficients: Mapping[str, float] = field(default_factory=dict)
    constraints: tuple[ArchitectLinearConstraint, ...] = ()
    rcond: float = 1.0e-12
    uncertainty_scale: str = "absolute"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ARCHITECT_LINEAR_FIT_PROBLEM_SCHEMA

    def __post_init__(self) -> None:
        labels = tuple(str(value) for value in self.coefficient_labels)
        matrix = np.asarray(self.design_matrix, dtype=float)
        observed = np.asarray(self.observations, dtype=float).reshape(-1)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("coefficient labels must be unique and nonempty")
        if matrix.ndim != 2 or matrix.shape != (len(observed), len(labels)):
            raise ValueError("design matrix shape differs from labels/observations")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(observed)):
            raise ValueError("fit design and observations must be finite")
        observation_labels = self.observation_labels or tuple(
            f"observation_{index:06d}" for index in range(len(observed))
        )
        if len(observation_labels) != len(observed) or len(set(observation_labels)) != len(
            observation_labels
        ):
            raise ValueError("observation labels must be unique and cover every row")
        base = (
            np.ones(len(observed), dtype=float)
            if self.base_weights is None
            else np.asarray(self.base_weights, dtype=float).reshape(-1)
        )
        if base.shape != observed.shape:
            raise ValueError("base weights must cover every observation")
        components = {
            str(name): np.asarray(values, dtype=float).reshape(-1)
            for name, values in self.weight_components.items()
        }
        if any(values.shape != observed.shape for values in components.values()):
            raise ValueError("every weight component must cover every observation")
        all_weights = (base, *components.values())
        if any(
            not np.all(np.isfinite(values)) or np.any(values < 0.0)
            for values in all_weights
        ):
            raise ValueError("weights must be finite and nonnegative")
        total = base.copy()
        for values in components.values():
            total *= values
        if not np.any(total > 0.0):
            raise ValueError("at least one observation must have positive total weight")
        fixed = {str(key): float(value) for key, value in self.fixed_coefficients.items()}
        unknown_fixed = set(fixed) - set(labels)
        if unknown_fixed:
            raise ValueError(f"unknown fixed coefficients: {sorted(unknown_fixed)}")
        if not all(isfinite(value) for value in fixed.values()):
            raise ValueError("fixed coefficients must be finite")
        constraints = tuple(self.constraints)
        unknown_constraint = {
            key
            for constraint in constraints
            for key in constraint.coefficients
            if key not in labels
        }
        if unknown_constraint:
            raise ValueError(f"unknown constrained coefficients: {sorted(unknown_constraint)}")
        if not isfinite(float(self.rcond)) or self.rcond <= 0.0:
            raise ValueError("rcond must be positive")
        if self.uncertainty_scale not in {"absolute", "reduced_chi_square"}:
            raise ValueError("uncertainty_scale must be absolute or reduced_chi_square")
        object.__setattr__(self, "coefficient_labels", labels)
        object.__setattr__(self, "design_matrix", matrix)
        object.__setattr__(self, "observations", observed)
        object.__setattr__(self, "observation_labels", tuple(observation_labels))
        object.__setattr__(self, "base_weights", base)
        object.__setattr__(self, "weight_components", components)
        object.__setattr__(self, "fixed_coefficients", fixed)
        object.__setattr__(self, "constraints", constraints)

    @property
    def total_weights(self) -> np.ndarray:
        values = np.asarray(self.base_weights, dtype=float).copy()
        for component in self.weight_components.values():
            values *= component
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "coefficient_labels": list(self.coefficient_labels),
            "design_matrix": self.design_matrix.tolist(),
            "observations": self.observations.tolist(),
            "observation_labels": list(self.observation_labels),
            "base_weights": np.asarray(self.base_weights).tolist(),
            "weight_components": {
                name: values.tolist() for name, values in self.weight_components.items()
            },
            "fixed_coefficients": dict(self.fixed_coefficients),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "rcond": self.rcond,
            "uncertainty_scale": self.uncertainty_scale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectLinearFitProblem":
        if payload.get("schema") != ARCHITECT_LINEAR_FIT_PROBLEM_SCHEMA:
            raise ValueError("unsupported ARCHITECT linear-fit problem schema")
        return cls(
            coefficient_labels=tuple(payload["coefficient_labels"]),
            design_matrix=np.asarray(payload["design_matrix"], dtype=float),
            observations=np.asarray(payload["observations"], dtype=float),
            observation_labels=tuple(payload.get("observation_labels", ())),
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
            rcond=float(payload.get("rcond", 1.0e-12)),
            uncertainty_scale=str(payload.get("uncertainty_scale", "absolute")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ArchitectLinearFitDiagnostics:
    observation_count: int
    positive_weight_count: int
    coefficient_count: int
    fixed_coefficient_count: int
    constraint_count: int
    constraint_rank: int
    free_dimension_after_constraints: int
    numerical_rank: int
    degrees_of_freedom: int
    rank_deficient: bool
    singular_values: tuple[float, ...]
    svd_cutoff: float
    effective_condition_number: float | None
    full_condition_number: float | None
    chi_square: float
    reduced_chi_square: float | None
    weighted_rms: float
    residual_rms: float
    maximum_absolute_residual: float
    maximum_constraint_error: float
    maximum_leverage: float
    column_scales: tuple[float, ...]
    parameter_identifiability: tuple[float, ...]
    weak_parameter_combinations: tuple[Mapping[str, float], ...]
    zero_weight_observations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "positive_weight_count": self.positive_weight_count,
            "coefficient_count": self.coefficient_count,
            "fixed_coefficient_count": self.fixed_coefficient_count,
            "constraint_count": self.constraint_count,
            "constraint_rank": self.constraint_rank,
            "free_dimension_after_constraints": self.free_dimension_after_constraints,
            "numerical_rank": self.numerical_rank,
            "degrees_of_freedom": self.degrees_of_freedom,
            "rank_deficient": self.rank_deficient,
            "singular_values": list(self.singular_values),
            "svd_cutoff": self.svd_cutoff,
            "effective_condition_number": self.effective_condition_number,
            "full_condition_number": self.full_condition_number,
            "chi_square": self.chi_square,
            "reduced_chi_square": self.reduced_chi_square,
            "weighted_rms": self.weighted_rms,
            "residual_rms": self.residual_rms,
            "maximum_absolute_residual": self.maximum_absolute_residual,
            "maximum_constraint_error": self.maximum_constraint_error,
            "maximum_leverage": self.maximum_leverage,
            "column_scales": list(self.column_scales),
            "parameter_identifiability": list(self.parameter_identifiability),
            "weak_parameter_combinations": [
                dict(value) for value in self.weak_parameter_combinations
            ],
            "zero_weight_observations": list(self.zero_weight_observations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectLinearFitDiagnostics":
        return cls(
            observation_count=int(payload["observation_count"]),
            positive_weight_count=int(payload["positive_weight_count"]),
            coefficient_count=int(payload["coefficient_count"]),
            fixed_coefficient_count=int(payload["fixed_coefficient_count"]),
            constraint_count=int(payload["constraint_count"]),
            constraint_rank=int(payload["constraint_rank"]),
            free_dimension_after_constraints=int(payload["free_dimension_after_constraints"]),
            numerical_rank=int(payload["numerical_rank"]),
            degrees_of_freedom=int(payload["degrees_of_freedom"]),
            rank_deficient=bool(payload["rank_deficient"]),
            singular_values=tuple(float(value) for value in payload["singular_values"]),
            svd_cutoff=float(payload["svd_cutoff"]),
            effective_condition_number=(
                None if payload.get("effective_condition_number") is None
                else float(payload["effective_condition_number"])
            ),
            full_condition_number=(
                None if payload.get("full_condition_number") is None
                else float(payload["full_condition_number"])
            ),
            chi_square=float(payload["chi_square"]),
            reduced_chi_square=(
                None if payload.get("reduced_chi_square") is None
                else float(payload["reduced_chi_square"])
            ),
            weighted_rms=float(payload["weighted_rms"]),
            residual_rms=float(payload["residual_rms"]),
            maximum_absolute_residual=float(payload["maximum_absolute_residual"]),
            maximum_constraint_error=float(payload["maximum_constraint_error"]),
            maximum_leverage=float(payload["maximum_leverage"]),
            column_scales=tuple(float(value) for value in payload["column_scales"]),
            parameter_identifiability=tuple(
                float(value) for value in payload["parameter_identifiability"]
            ),
            weak_parameter_combinations=tuple(
                {str(key): float(value) for key, value in row.items()}
                for row in payload["weak_parameter_combinations"]
            ),
            zero_weight_observations=tuple(payload["zero_weight_observations"]),
        )


@dataclass(frozen=True)
class ArchitectLinearFitResult:
    """Coefficients, uncertainty and influence diagnostics for one fit."""

    coefficient_labels: tuple[str, ...]
    coefficients: np.ndarray
    observation_labels: tuple[str, ...]
    predicted: np.ndarray
    residuals: np.ndarray
    total_weights: np.ndarray
    leverage: np.ndarray
    standardized_weighted_residuals: np.ndarray
    covariance_absolute: np.ndarray
    covariance_reduced_chi_square: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    fixed_coefficients: Mapping[str, float]
    constraints: tuple[ArchitectLinearConstraint, ...]
    diagnostics: ArchitectLinearFitDiagnostics
    uncertainty_scale: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ARCHITECT_LINEAR_FIT_RESULT_SCHEMA

    @property
    def coefficient_map(self) -> dict[str, float]:
        return dict(zip(self.coefficient_labels, map(float, self.coefficients), strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "coefficient_labels": list(self.coefficient_labels),
            "coefficients": self.coefficients.tolist(),
            "coefficient_map": self.coefficient_map,
            "observation_labels": list(self.observation_labels),
            "predicted": self.predicted.tolist(),
            "residuals": self.residuals.tolist(),
            "total_weights": self.total_weights.tolist(),
            "leverage": self.leverage.tolist(),
            "standardized_weighted_residuals": (
                self.standardized_weighted_residuals.tolist()
            ),
            "covariance_absolute": self.covariance_absolute.tolist(),
            "covariance_reduced_chi_square": (
                self.covariance_reduced_chi_square.tolist()
            ),
            "covariance": self.covariance.tolist(),
            "correlation": self.correlation.tolist(),
            "fixed_coefficients": dict(self.fixed_coefficients),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "diagnostics": self.diagnostics.to_dict(),
            "uncertainty_scale": self.uncertainty_scale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectLinearFitResult":
        if payload.get("schema") != ARCHITECT_LINEAR_FIT_RESULT_SCHEMA:
            raise ValueError("unsupported ARCHITECT linear-fit result schema")
        return cls(
            coefficient_labels=tuple(payload["coefficient_labels"]),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            observation_labels=tuple(payload["observation_labels"]),
            predicted=np.asarray(payload["predicted"], dtype=float),
            residuals=np.asarray(payload["residuals"], dtype=float),
            total_weights=np.asarray(payload["total_weights"], dtype=float),
            leverage=np.asarray(payload["leverage"], dtype=float),
            standardized_weighted_residuals=np.asarray(
                payload["standardized_weighted_residuals"], dtype=float
            ),
            covariance_absolute=np.asarray(payload["covariance_absolute"], dtype=float),
            covariance_reduced_chi_square=np.asarray(
                payload["covariance_reduced_chi_square"], dtype=float
            ),
            covariance=np.asarray(payload["covariance"], dtype=float),
            correlation=np.asarray(payload["correlation"], dtype=float),
            fixed_coefficients={
                str(key): float(value)
                for key, value in payload.get("fixed_coefficients", {}).items()
            },
            constraints=tuple(
                ArchitectLinearConstraint.from_dict(value)
                for value in payload.get("constraints", ())
            ),
            diagnostics=ArchitectLinearFitDiagnostics.from_dict(payload["diagnostics"]),
            uncertainty_scale=str(payload["uncertainty_scale"]),
            metadata=dict(payload.get("metadata", {})),
        )


def fit_architect_linear_surface(
    problem: ArchitectLinearFitProblem,
) -> ArchitectLinearFitResult:
    """Solve a weighted linear problem while satisfying exact constraints."""

    labels = problem.coefficient_labels
    label_to_index = {label: index for index, label in enumerate(labels)}
    design = problem.design_matrix
    observed = problem.observations
    weights = problem.total_weights
    coefficient_count = len(labels)
    fixed_indices = tuple(sorted(label_to_index[label] for label in problem.fixed_coefficients))
    fixed_set = set(fixed_indices)
    free_indices = tuple(index for index in range(coefficient_count) if index not in fixed_set)
    baseline = np.zeros(coefficient_count, dtype=float)
    for label, value in problem.fixed_coefficients.items():
        baseline[label_to_index[label]] = value

    constraint_matrix, constraint_targets = _constraint_matrix(
        problem.constraints, label_to_index, coefficient_count
    )
    adjusted_targets = constraint_targets - constraint_matrix @ baseline
    free_constraint = constraint_matrix[:, free_indices]
    particular, null_basis, constraint_rank = _constraint_parameterization(
        free_constraint,
        adjusted_targets,
        problem.rcond,
    )
    baseline[list(free_indices)] = particular
    reduced_design = design[:, free_indices] @ null_basis
    reduced_target = observed - design @ baseline
    positive = weights > 0.0
    sqrt_weights = np.sqrt(weights[positive])
    weighted_design = reduced_design[positive] * sqrt_weights[:, None]
    weighted_target = reduced_target[positive] * sqrt_weights
    reduced_dimension = null_basis.shape[1]

    solution, covariance_reduced, singular_values, rank, cutoff, column_scales, u_rank = (
        _scaled_svd_solve(weighted_design, weighted_target, problem.rcond)
    )
    coefficients = baseline.copy()
    if reduced_dimension:
        coefficients[list(free_indices)] += null_basis @ solution
    predicted = design @ coefficients
    residuals = observed - predicted
    weighted_residuals = np.sqrt(weights) * residuals

    covariance_absolute = np.zeros((coefficient_count, coefficient_count), dtype=float)
    if free_indices and reduced_dimension:
        free_covariance = null_basis @ covariance_reduced @ null_basis.T
        covariance_absolute[np.ix_(free_indices, free_indices)] = free_covariance
    chi_square = float(weighted_residuals @ weighted_residuals)
    dof = int(np.count_nonzero(positive) - rank)
    reduced_chi_square = None if dof <= 0 else chi_square / dof
    covariance_relative = covariance_absolute * (
        1.0 if reduced_chi_square is None else reduced_chi_square
    )
    covariance = (
        covariance_absolute
        if problem.uncertainty_scale == "absolute"
        else covariance_relative
    )
    correlation = _correlation(covariance)

    leverage = np.zeros(len(observed), dtype=float)
    if rank:
        leverage[positive] = np.sum(u_rank**2, axis=1)
    standardized = np.zeros(len(observed), dtype=float)
    denominator = np.sqrt(np.maximum(1.0 - leverage, np.finfo(float).eps))
    standardized[positive] = weighted_residuals[positive] / denominator[positive]

    null_directions = _data_null_directions(
        null_basis,
        free_indices,
        coefficient_count,
        weighted_design,
        column_scales,
        rank,
    )
    identifiability = _parameter_identifiability(
        null_directions, coefficient_count, fixed_indices
    )
    weak_combinations = tuple(
        _labelled_direction(direction, labels) for direction in null_directions.T
    )
    constraint_error = (
        constraint_matrix @ coefficients - constraint_targets
        if len(problem.constraints) else np.zeros(0, dtype=float)
    )
    effective_condition = (
        None if rank == 0 else float(singular_values[0] / singular_values[rank - 1])
    )
    full_condition = effective_condition if rank == reduced_dimension else None
    diagnostics = ArchitectLinearFitDiagnostics(
        observation_count=len(observed),
        positive_weight_count=int(np.count_nonzero(positive)),
        coefficient_count=coefficient_count,
        fixed_coefficient_count=len(fixed_indices),
        constraint_count=len(problem.constraints),
        constraint_rank=constraint_rank,
        free_dimension_after_constraints=reduced_dimension,
        numerical_rank=rank,
        degrees_of_freedom=dof,
        rank_deficient=rank < reduced_dimension,
        singular_values=tuple(map(float, singular_values)),
        svd_cutoff=cutoff,
        effective_condition_number=effective_condition,
        full_condition_number=full_condition,
        chi_square=chi_square,
        reduced_chi_square=reduced_chi_square,
        weighted_rms=float(np.sqrt(chi_square / np.count_nonzero(positive))),
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
        maximum_absolute_residual=float(np.max(np.abs(residuals), initial=0.0)),
        maximum_constraint_error=float(
            np.max(np.abs(constraint_error), initial=0.0)
        ),
        maximum_leverage=float(np.max(leverage, initial=0.0)),
        column_scales=tuple(map(float, column_scales)),
        parameter_identifiability=tuple(map(float, identifiability)),
        weak_parameter_combinations=weak_combinations,
        zero_weight_observations=tuple(
            label
            for label, weight in zip(problem.observation_labels, weights, strict=True)
            if weight == 0.0
        ),
    )
    return ArchitectLinearFitResult(
        coefficient_labels=labels,
        coefficients=coefficients,
        observation_labels=problem.observation_labels,
        predicted=predicted,
        residuals=residuals,
        total_weights=weights,
        leverage=leverage,
        standardized_weighted_residuals=standardized,
        covariance_absolute=covariance_absolute,
        covariance_reduced_chi_square=covariance_relative,
        covariance=covariance,
        correlation=correlation,
        fixed_coefficients=problem.fixed_coefficients,
        constraints=problem.constraints,
        diagnostics=diagnostics,
        uncertainty_scale=problem.uncertainty_scale,
        metadata={
            **dict(problem.metadata),
            "weight_components": {
                name: {
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "zero_count": int(np.count_nonzero(values == 0.0)),
                }
                for name, values in problem.weight_components.items()
            },
            "fixed_coefficients_preserved_exactly": True,
            "linear_constraints_enforced_in_null_space": True,
        },
    )


def _constraint_matrix(
    constraints: Sequence[ArchitectLinearConstraint],
    label_to_index: Mapping[str, int],
    coefficient_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(constraints), coefficient_count), dtype=float)
    targets = np.zeros(len(constraints), dtype=float)
    for row, constraint in enumerate(constraints):
        for label, value in constraint.coefficients.items():
            matrix[row, label_to_index[label]] = value
        targets[row] = constraint.target
    return matrix, targets


def _constraint_parameterization(
    matrix: np.ndarray,
    targets: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    column_count = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.zeros(column_count), np.eye(column_count), 0
    if column_count == 0:
        if np.max(np.abs(targets), initial=0.0) > rcond:
            raise ValueError("fixed coefficients violate an exact constraint")
        return np.zeros(0), np.zeros((0, 0)), 0
    row_norm = np.linalg.norm(matrix, axis=1)
    active = row_norm > rcond
    if np.any(np.abs(targets[~active]) > rcond):
        raise ValueError("fixed coefficients violate an exact constraint")
    reduced_matrix = matrix[active]
    reduced_targets = targets[active]
    if not len(reduced_targets):
        return np.zeros(column_count), np.eye(column_count), 0
    u_matrix, singular, vt_matrix = np.linalg.svd(reduced_matrix, full_matrices=True)
    cutoff = rcond * (singular[0] if len(singular) else 1.0)
    rank = int(np.count_nonzero(singular > cutoff))
    particular = np.zeros(column_count, dtype=float)
    if rank:
        particular = vt_matrix[:rank].T @ (
            (u_matrix[:, :rank].T @ reduced_targets) / singular[:rank]
        )
    residual = reduced_matrix @ particular - reduced_targets
    if np.max(np.abs(residual), initial=0.0) > max(rcond, cutoff) * 10.0:
        raise ValueError("linear constraints are mutually inconsistent")
    return particular, vt_matrix[rank:].T, rank


def _scaled_svd_solve(
    design: np.ndarray,
    target: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray]:
    parameter_count = design.shape[1]
    if parameter_count == 0:
        return (
            np.zeros(0),
            np.zeros((0, 0)),
            np.zeros(0),
            0,
            0.0,
            np.zeros(0),
            np.zeros((len(target), 0)),
        )
    scales = np.linalg.norm(design, axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    scaled = design / scales
    full_matrices = scaled.shape[0] < parameter_count
    u_matrix, singular, vt_matrix = np.linalg.svd(
        scaled, full_matrices=full_matrices
    )
    cutoff = rcond * (singular[0] if len(singular) else 1.0)
    rank = int(np.count_nonzero(singular > cutoff))
    scaled_solution = np.zeros(parameter_count, dtype=float)
    if rank:
        scaled_solution = vt_matrix[:rank].T @ (
            (u_matrix[:, :rank].T @ target) / singular[:rank]
        )
    inverse_scale = 1.0 / scales
    solution = scaled_solution * inverse_scale
    covariance_scaled = np.zeros((parameter_count, parameter_count), dtype=float)
    if rank:
        covariance_scaled = (vt_matrix[:rank].T / singular[:rank] ** 2) @ vt_matrix[:rank]
    covariance = inverse_scale[:, None] * covariance_scaled * inverse_scale[None, :]
    return solution, covariance, singular, rank, cutoff, scales, u_matrix[:, :rank]


def _data_null_directions(
    constraint_null_basis: np.ndarray,
    free_indices: Sequence[int],
    coefficient_count: int,
    weighted_design: np.ndarray,
    column_scales: np.ndarray,
    rank: int,
) -> np.ndarray:
    reduced_dimension = constraint_null_basis.shape[1]
    if reduced_dimension == 0 or rank == reduced_dimension:
        return np.zeros((coefficient_count, 0), dtype=float)
    scaled = weighted_design / column_scales
    full_matrices = scaled.shape[0] < scaled.shape[1]
    _u_matrix, _singular, vt_matrix = np.linalg.svd(
        scaled, full_matrices=full_matrices
    )
    reduced_null = (vt_matrix[rank:].T / column_scales[:, None])
    free_null = constraint_null_basis @ reduced_null
    full = np.zeros((coefficient_count, free_null.shape[1]), dtype=float)
    full[list(free_indices), :] = free_null
    q_matrix, _ = np.linalg.qr(full)
    return q_matrix[:, : free_null.shape[1]]


def _parameter_identifiability(
    null_directions: np.ndarray,
    coefficient_count: int,
    fixed_indices: Sequence[int],
) -> np.ndarray:
    values = np.ones(coefficient_count, dtype=float)
    if null_directions.shape[1]:
        values -= np.sum(null_directions**2, axis=1)
    values[list(fixed_indices)] = 1.0
    return np.clip(values, 0.0, 1.0)


def _labelled_direction(
    direction: np.ndarray,
    labels: Sequence[str],
) -> Mapping[str, float]:
    scale = float(np.max(np.abs(direction), initial=0.0))
    if scale == 0.0:
        return {}
    return {
        label: float(value / scale)
        for label, value in zip(labels, direction, strict=True)
        if abs(value / scale) >= 1.0e-8
    }


def _correlation(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = diagonal[:, None] * diagonal[None, :]
    result = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    np.fill_diagonal(result, np.where(diagonal > 0.0, 1.0, 0.0))
    return result


# Canonical backend-neutral API. The Architect-prefixed names and serialized
# schemas remain available so existing fitting inputs and results stay readable.
LinearConstraint = ArchitectLinearConstraint
LinearFitProblem = ArchitectLinearFitProblem
LinearFitDiagnostics = ArchitectLinearFitDiagnostics
LinearFitResult = ArchitectLinearFitResult
fit_linear_surface = fit_architect_linear_surface
