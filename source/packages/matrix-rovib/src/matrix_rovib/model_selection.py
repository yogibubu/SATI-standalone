"""Numerically defensible selection of LOCK Hamiltonian hierarchies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


LOCK_MODEL_SELECTION_SCHEMA = "matrix.lock.model_selection.v1"


@dataclass(frozen=True)
class ModelCandidateEvaluation:
    name: str
    parameter_names: tuple[str, ...]
    training_residuals: tuple[float, ...]
    jacobian: np.ndarray
    validation_residuals: tuple[float, ...] = ()
    prospective_residuals: tuple[float, ...] = ()
    physical_violations: tuple[str, ...] = ()
    parameter_scales: tuple[float, ...] = ()
    predicate_jacobian: np.ndarray | None = None
    numerical_error_ratio: float | None = None
    fit_converged: bool | None = None

    def __post_init__(self) -> None:
        jacobian = np.asarray(self.jacobian, dtype=float)
        if not self.name.strip() or not self.parameter_names or not self.training_residuals:
            raise ValueError("model candidates require a name, parameters, and training residuals")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("model candidate parameter names must be unique")
        if jacobian.shape != (len(self.training_residuals), len(self.parameter_names)):
            raise ValueError("candidate Jacobian shape does not match residuals and parameters")
        if not np.all(np.isfinite(jacobian)) or not np.all(
            np.isfinite((
                *self.training_residuals,
                *self.validation_residuals,
                *self.prospective_residuals,
            ))
        ):
            raise ValueError("model candidate diagnostics must be finite")
        if any(not str(item).strip() for item in self.physical_violations):
            raise ValueError("physical violations must have nonempty descriptions")
        scales = np.asarray(self.parameter_scales or (1.,) * len(self.parameter_names), float)
        if scales.shape != (len(self.parameter_names),) or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
            raise ValueError("candidate physical parameter scales must be positive and finite")
        predicate = (np.empty((0, len(self.parameter_names))) if self.predicate_jacobian is None
                     else np.asarray(self.predicate_jacobian, float))
        if predicate.ndim != 2 or predicate.shape[1] != len(self.parameter_names) or not np.all(np.isfinite(predicate)):
            raise ValueError("candidate predicate Jacobian must be finite and have one column per parameter")
        if self.numerical_error_ratio is not None and (
            not np.isfinite(self.numerical_error_ratio) or self.numerical_error_ratio < 0):
            raise ValueError("candidate numerical error ratio must be finite and nonnegative")
        if self.fit_converged is not None and not isinstance(self.fit_converged, bool):
            raise ValueError("fit_converged must be an explicit boolean or unknown")
        object.__setattr__(self, "parameter_scales", tuple(scales))
        object.__setattr__(self, "predicate_jacobian", predicate.copy())
        object.__setattr__(self, "jacobian", jacobian.copy())


@dataclass(frozen=True)
class OperatorFamily:
    """One symmetry-allowed level in a molecule-independent Hamiltonian hierarchy."""

    name: str
    order: int
    parameter_patterns: tuple[str, ...]
    prerequisites: tuple[str, ...] = ()
    model_classes: tuple[str, ...] = ("watson", "ram", "lam", "spin", "hyperfine")

    def __post_init__(self) -> None:
        if (
            not self.name.strip()
            or self.order < 0
            or not self.parameter_patterns
            or any(not item.strip() for item in self.parameter_patterns)
        ):
            raise ValueError("operator families require a name, order and parameter patterns")


LOCK_STANDARD_OPERATOR_REGISTRY = (
    OperatorFamily("rigid", 2, ("A", "B", "C")),
    OperatorFamily(
        "quartic", 4, ("D*", "d*"), prerequisites=("rigid",),
        model_classes=("watson", "ram", "lam", "spin", "hyperfine"),
    ),
    OperatorFamily(
        "sextic", 6, ("H*", "h*"), prerequisites=("quartic",),
        model_classes=("watson", "ram", "lam", "spin", "hyperfine"),
    ),
    OperatorFamily(
        "octic", 8, ("L*", "l*"), prerequisites=("sextic",),
        model_classes=("watson", "ram", "lam", "spin", "hyperfine"),
    ),
    OperatorFamily(
        "large-amplitude", 2, ("F*", "rho*", "V*", "RAM*", "LAM*"),
        prerequisites=("rigid",), model_classes=("ram", "lam", "spin", "hyperfine"),
    ),
    OperatorFamily(
        "lam-quartic", 4, ("p4*", "k4*", "lam_p4.*"), prerequisites=("large-amplitude",),
        model_classes=("ram", "lam", "spin", "hyperfine"),
    ),
    OperatorFamily(
        "torsion-rotation-high", 4,
        ("p2J2*", "p2R.*", "pJ3*", "pJ3.*", "pi2rot*", "pi2.*"),
        prerequisites=("large-amplitude",),
        model_classes=("ram", "lam", "spin", "hyperfine"),
    ),
    OperatorFamily(
        "multi-lam-coupling", 4, ("p1p2*", "V12*", "R12*"),
        prerequisites=("large-amplitude",),
        model_classes=("ram", "lam", "spin", "hyperfine"),
    ),
    OperatorFamily(
        "spin", 2, ("epsilon*", "D_zfs*", "E_zfs*"),
        prerequisites=("rigid",), model_classes=("spin", "hyperfine"),
    ),
    OperatorFamily(
        "hyperfine", 2, ("chi*", "a_F*", "T*", "C_I*"),
        prerequisites=("spin",), model_classes=("hyperfine",),
    ),
)


def select_model_hierarchy(
    candidates: Sequence[ModelCandidateEvaluation],
    *,
    rank_rcond: float = 1.0e-10,
    correlation_limit: float = 0.98,
    require_qualification: bool = False,
    allow_predicate_supported: bool = False,
    validation_tolerance: float = 0.0,
) -> dict[str, object]:
    """Rank fitted candidate models using validation, identifiability and BIC.

    Validation RMS is preferred when every candidate supplies a held-out set.
    Otherwise BIC selects the smallest adequate model. Rank deficiency and
    excessive parameter correlation make a candidate scientifically ineligible.
    Residuals and Jacobians must be covariance-whitened on identical rows.
    Predicates enter identifiability only, never the observation RSS or BIC.
    Prospective observations are reported but never used to select a model.
    Set require_qualification for production selection: both numerical error
    budget and nonlinear stationarity must then be explicitly qualified.
    """

    ordered = tuple(candidates)
    if not ordered or len({item.name for item in ordered}) != len(ordered):
        raise ValueError("model hierarchy requires nonempty uniquely named candidates")
    if (not np.isfinite(rank_rcond) or not 0 < rank_rcond < 1
        or not np.isfinite(correlation_limit) or not 0.0 < correlation_limit < 1.0
        or not np.isfinite(validation_tolerance) or validation_tolerance < 0):
        raise ValueError("invalid model-selection numerical thresholds")
    if len({len(item.training_residuals) for item in ordered}) != 1 or len(
        {len(item.validation_residuals) for item in ordered}) != 1:
        raise ValueError("compare candidates on the same training and validation observations")
    has_validation = all(item.validation_residuals for item in ordered)
    rows = []
    for item in ordered:
        residuals = np.asarray(item.training_residuals, dtype=float)
        jacobian = np.asarray(item.jacobian, dtype=float)
        from .parameter_selection import select_spectral_parameters
        scaled = jacobian * np.asarray(item.parameter_scales)
        augmented = np.vstack((jacobian, item.predicate_jacobian))
        information = select_spectral_parameters(jacobian, item.parameter_names,
            item.parameter_scales, augmented_jacobian=augmented, rcond=rank_rcond)
        rank = information['experimental']['rank']
        augmented_rank = information['augmented']['rank']
        selected_jacobian = augmented * item.parameter_scales if allow_predicate_supported else scaled
        # Work directly with the scaled Jacobian. Normal equations square its
        # condition number and can erase weak but identifiable directions.
        _, singular, vh = np.linalg.svd(selected_jacobian, full_matrices=False)
        keep = singular > (rank_rcond * singular[0] if singular.size else 0.)
        inverse = vh[keep].T / singular[keep]
        covariance = inverse @ inverse.T
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(errors, errors)
        correlation = np.zeros_like(covariance)
        np.divide(covariance, denominator, out=correlation, where=denominator > 0.0)
        off_diagonal = correlation - np.diag(np.diag(correlation))
        maximum_correlation = float(np.max(np.abs(off_diagonal), initial=0.0))
        n, k = len(residuals), len(item.parameter_names)
        rss = max(float(residuals @ residuals), np.finfo(float).tiny)
        aic = n * math.log(rss / n) + 2 * k
        aicc = math.inf if n <= k + 1 else aic + 2 * k * (k + 1) / (n - k - 1)
        bic = n * math.log(rss / n) + k * math.log(n)
        validation_rms = (
            float(np.sqrt(np.mean(np.square(item.validation_residuals))))
            if item.validation_residuals
            else None
        )
        prospective_rms = (
            float(np.sqrt(np.mean(np.square(item.prospective_residuals))))
            if item.prospective_residuals
            else None
        )
        physically_valid = not item.physical_violations
        identifiable = (augmented_rank if allow_predicate_supported else rank) == k and maximum_correlation <= correlation_limit
        qualified = item.fit_converged is True and item.numerical_error_ratio is not None and item.numerical_error_ratio <= 1
        rejection_reasons = []
        if not physically_valid:
            rejection_reasons.extend(item.physical_violations)
        if not identifiable:
            rejection_reasons.append('rank deficient or excessive parameter correlation')
        if require_qualification and not qualified:
            rejection_reasons.append('fit stationarity or numerical convergence not qualified')
        rows.append(
            {
                "name": item.name,
                "parameter_names": list(item.parameter_names),
                "parameter_count": k,
                "observation_count": n,
                "rank": rank,
                "augmented_rank": augmented_rank,
                "parameter_scales": list(item.parameter_scales),
                "parameter_information": information['information_sources'],
                "experimentally_identifiable": rank == k,
                "qualification_passed": qualified,
                "numerical_error_ratio": item.numerical_error_ratio,
                "fit_converged": item.fit_converged,
                "rejection_reasons": rejection_reasons,
                "identifiable": identifiable,
                "maximum_absolute_correlation": maximum_correlation,
                "training_rms": float(np.sqrt(rss / n)),
                "validation_rms": validation_rms,
                "prospective_rms": prospective_rms,
                "physically_valid": physically_valid,
                "physical_violations": list(item.physical_violations),
                "aicc": None if not math.isfinite(aicc) else aicc,
                "bic": bic,
            }
        )
    eligible = [
        row for row in rows if not row['rejection_reasons']
    ]
    if not eligible:
        selected = None
        reason = "no candidate meets the requested physical, identifiability and qualification gates"
    elif has_validation:
        best = min(row['validation_rms'] for row in eligible)
        adequate = [row for row in eligible if row['validation_rms'] <= best + validation_tolerance]
        selected = min(adequate, key=lambda row: (row['parameter_count'], row['validation_rms'], row['name']))
        reason = "smallest qualified model within the declared tolerance of minimum validation RMS"
    else:
        selected = min(eligible, key=lambda row: (row["bic"], row["parameter_count"]))
        reason = "minimum BIC among identifiable candidates"
    return {
        "schema": LOCK_MODEL_SELECTION_SCHEMA,
        "selected": None if selected is None else selected["name"],
        "reason": reason,
        "rank_rcond": rank_rcond,
        "correlation_limit": correlation_limit,
        "require_qualification": require_qualification,
        "allow_predicate_supported": allow_predicate_supported,
        "validation_tolerance": validation_tolerance,
        "prospective_used_for_selection": False,
        "candidate_comparison_contract": "same whitened training and validation rows, in the same order; prospective rows are audit-only",
        "candidates": rows,
    }


def select_registered_operator_hierarchy(
    candidates: Sequence[ModelCandidateEvaluation],
    registry: Sequence[OperatorFamily | Mapping[str, object]],
    *,
    model_class: str,
    rank_rcond: float = 1.0e-10,
    correlation_limit: float = 0.98,
    require_qualification: bool = False,
    allow_predicate_supported: bool = False,
    validation_tolerance: float = 0.0,
) -> dict[str, object]:
    """Validate and select a nested Watson/RAM/LAM/spin operator hierarchy.

    This layer prevents a molecule-specific fit script from introducing an
    arbitrary term: every candidate level must correspond to a registered
    family, obey declared prerequisites, and remain nested in registry order.
    """

    families = tuple(
        item if isinstance(item, OperatorFamily) else OperatorFamily(
            name=str(item["name"]),
            order=int(item["order"]),
            parameter_patterns=tuple(str(value) for value in item["parameter_patterns"]),
            prerequisites=tuple(str(value) for value in item.get("prerequisites", ())),
            model_classes=tuple(str(value) for value in item.get(
                "model_classes", ("watson", "ram", "lam", "spin", "hyperfine")
            )),
        )
        for item in registry
    )
    active = tuple(item for item in families if model_class in item.model_classes)
    if not active:
        raise ValueError(f"operator registry has no families for {model_class!r}")
    names = [item.name for item in active]
    if len(set(names)) != len(names) or any(
        prerequisite not in names
        for item in active for prerequisite in item.prerequisites
    ):
        raise ValueError("operator registry names or prerequisites are invalid")
    position = {name: index for index, name in enumerate(names)}
    for item in active:
        if any(position[required] >= position[item.name] for required in item.prerequisites):
            raise ValueError("operator-family prerequisite must precede its dependent family")
    previous: tuple[str, ...] = ()
    for index, candidate in enumerate(candidates):
        declared = tuple(item for item in candidate.name.split("+") if item)
        if not declared or len(set(declared)) != len(declared) or any(
            item not in names for item in declared
        ):
            raise ValueError(f"candidate {index} contains an unregistered operator family")
        if index and (
            declared[: len(previous)] != previous or len(declared) <= len(previous)
        ):
            raise ValueError("operator candidates must form a strictly nested hierarchy")
        declared_set = set(declared)
        for family_name in declared:
            family = active[position[family_name]]
            if any(required not in declared_set for required in family.prerequisites):
                raise ValueError(
                    f"candidate {candidate.name!r} omits prerequisite for {family_name!r}"
                )
        previous = declared
    result = select_model_hierarchy(
        candidates, rank_rcond=rank_rcond, correlation_limit=correlation_limit,
        require_qualification=require_qualification, allow_predicate_supported=allow_predicate_supported,
        validation_tolerance=validation_tolerance,
    )
    result.update({
        "registry": [
            {
                "name": item.name,
                "order": item.order,
                "parameter_patterns": list(item.parameter_patterns),
                "prerequisites": list(item.prerequisites),
            }
            for item in active
        ],
        "model_class": model_class,
        "selection_scope": "registered molecule-independent operator hierarchy",
    })
    return result


def linearized_nested_model_selection(
    parameter_names: Sequence[str],
    parameter_groups: Sequence[str],
    initial_values: Sequence[float],
    optimum_values: Sequence[float],
    weighted_residuals: Sequence[float],
    weighted_jacobian: np.ndarray,
    *,
    validation_fraction: float = 0.2,
    rank_rcond: float = 1.0e-10,
    correlation_limit: float = 0.98,
) -> dict[str, object]:
    """Assess a nested physical hierarchy around a converged nonlinear fit.

    Parameters sharing a physical group enter together.  Every candidate fixes
    omitted variables at their declared initial values, refits the retained
    directions on a deterministic training subset, and evaluates the remaining
    observations.  The calculation is a local post-fit diagnostic: it never
    changes the returned nonlinear optimum.
    """

    names = tuple(str(value) for value in parameter_names)
    groups = tuple(str(value) for value in parameter_groups)
    initial = np.asarray(initial_values, dtype=float)
    optimum = np.asarray(optimum_values, dtype=float)
    residual = np.asarray(weighted_residuals, dtype=float)
    jacobian = np.asarray(weighted_jacobian, dtype=float)
    count = len(names)
    if (
        not names
        or len(groups) != count
        or initial.shape != (count,)
        or optimum.shape != (count,)
        or jacobian.shape != (residual.size, count)
        or len(set(names)) != count
        or any(not value for value in groups)
    ):
        raise ValueError("invalid arrays for linearized nested model selection")
    if not np.all(np.isfinite(jacobian)) or not np.all(
        np.isfinite(np.concatenate((initial, optimum, residual)))
    ):
        raise ValueError("model-selection inputs must be finite")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in [0, 0.5)")

    group_order = tuple(dict.fromkeys(groups))
    use_validation = residual.size >= 10 and validation_fraction > 0.0
    validation = np.zeros(residual.size, dtype=bool)
    if use_validation:
        stride = max(int(round(1.0 / validation_fraction)), 2)
        validation[::stride] = True
        if np.all(validation) or not np.any(validation):
            validation[:] = False
            use_validation = False
    training = ~validation
    candidates = []
    candidate_metadata = []
    fixed_displacement = initial - optimum
    for group_count in range(1, len(group_order) + 1):
        active_groups = set(group_order[:group_count])
        active = np.asarray([group in active_groups for group in groups])
        omitted = ~active
        baseline = residual.copy()
        if np.any(omitted):
            baseline += jacobian[:, omitted] @ fixed_displacement[omitted]
        active_jacobian = jacobian[:, active]
        if active_jacobian.size:
            correction = np.linalg.lstsq(
                active_jacobian[training], -baseline[training], rcond=rank_rcond
            )[0]
            candidate_residual = baseline + active_jacobian @ correction
        else:  # pragma: no cover - the first physical group is always active
            candidate_residual = baseline
        label = "+".join(group_order[:group_count])
        candidates.append(
            ModelCandidateEvaluation(
                name=label,
                parameter_names=tuple(
                    name for name, enabled in zip(names, active, strict=True) if enabled
                ),
                training_residuals=tuple(candidate_residual[training]),
                validation_residuals=(
                    tuple(candidate_residual[validation]) if use_validation else ()
                ),
                jacobian=active_jacobian[training],
            )
        )
        candidate_metadata.append(
            {
                "name": label,
                "active_groups": list(group_order[:group_count]),
                "omitted_groups": list(group_order[group_count:]),
            }
        )
    result = select_model_hierarchy(
        candidates, rank_rcond=rank_rcond, correlation_limit=correlation_limit
    )
    result.update(
        {
            "method": "deterministic holdout linearized about the converged nonlinear optimum",
            "advisory": True,
            "changes_returned_fit": False,
            "validation_fraction": validation_fraction if use_validation else 0.0,
            "training_observation_count": int(np.count_nonzero(training)),
            "validation_observation_count": int(np.count_nonzero(validation)),
            "physical_group_order": list(group_order),
            "candidate_structure": candidate_metadata,
        }
    )
    return result


__all__ = [
    "LOCK_STANDARD_OPERATOR_REGISTRY",
    "LOCK_MODEL_SELECTION_SCHEMA",
    "ModelCandidateEvaluation",
    "OperatorFamily",
    "linearized_nested_model_selection",
    "select_registered_operator_hierarchy",
    "select_model_hierarchy",
]
