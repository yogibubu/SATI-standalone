"""Assigned-line refinement of a contracted local-channel Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .lam_rotation import _hermitian
from .native_rotor import (
    NativeRotorParameters,
    angular_momentum_operators,
    centrifugal_distortion_operators,
    diagonalize_rotor,
)


@dataclass(frozen=True)
class LocalChannelLevel:
    J: int
    Ka: int
    Kc: int
    parent: int


@dataclass(frozen=True)
class LocalChannelTransition:
    upper: LocalChannelLevel
    lower: LocalChannelLevel
    weight: float = 1.0


@dataclass(frozen=True)
class LocalChannelObservation:
    frequency_MHz: float
    uncertainty_MHz: float
    components: tuple[LocalChannelTransition, ...]
    label: str = ""
    role: str = "training"
    dataset: str = "default"


@dataclass(frozen=True)
class LocalChannelRASInteraction:
    left: int
    right: int
    axes: str = "ab"
    order: int = 8


def _parameter_family(name: str) -> str:
    """Return a convention-independent physical family for a fit coordinate."""
    stem = name.split("_", 1)[0]
    if stem in {"gap", "A", "B", "C"}:
        return "origins_and_rotation"
    if stem in {"DJ", "DJK", "DK", "d1", "d2"}:
        return "watson_quartic"
    if stem in {"HJ", "HJK", "HKJ", "HK", "h1", "h2", "h3"}:
        return "watson_sextic"
    if stem in {"LJ", "LJJK", "LJK", "LKKJ", "LK", "l1", "l2", "l3", "l4"}:
        return "watson_octic"
    if stem == "F":
        return "ras_order_2"
    if stem in {"FK", "FJ", "F2"}:
        return "ras_order_4"
    if stem in {"FKK", "FJK", "F2J"}:
        return "ras_order_6"
    if stem in {"FJKK", "FJJK"}:
        return "ras_order_8"
    return "other"


def _local_channel_identifiability_report(
    names: Sequence[str], scales: np.ndarray, data_vh: np.ndarray, data_rank: int,
    augmented_rank: int, covariance: np.ndarray,
    predicates: Mapping[str, tuple[float, float]],
) -> dict[str, object]:
    """Resolve experimental, correlated, and predicate-supported directions.

    Scores are diagonal elements of the projector onto the experimentally
    observable right-singular subspace of the uncertainty-whitened Jacobian.
    Right-singular vectors use the declared physical parameter scales, so the
    reported null combinations are dimensionless and comparable.
    """
    parameter_count = len(names)
    identifiable = data_vh[:data_rank]
    scores = np.sum(np.abs(identifiable) ** 2, axis=0)
    null_vh = data_vh[data_rank:]
    null_fractions = np.maximum(0.0, 1.0 - scores)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.maximum(np.diag(covariance), 0.0)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    correlations = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0.0,
    )
    correlations = np.clip(correlations, -1.0, 1.0)
    np.fill_diagonal(correlations, 0.0)
    maximum_correlations = np.max(np.abs(correlations), axis=1)

    rows = []
    for index, name in enumerate(names):
        score = float(np.clip(scores[index], 0.0, 1.0))
        correlated = bool(maximum_correlations[index] >= 0.95)
        if score >= 0.95 and not correlated:
            classification = "data_determined"
        elif score >= 0.95:
            classification = "correlated"
        elif name in predicates:
            classification = "predicate_regularized"
        else:
            classification = "undetermined"
        rows.append({
            "name": name,
            "family": _parameter_family(name),
            "classification": classification,
            "experimental_identifiability": score,
            "experimental_null_fraction": float(null_fractions[index]),
            "maximum_absolute_augmented_correlation": float(maximum_correlations[index]),
            "predicate_present": name in predicates,
            "physical_scale": float(scales[index]),
        })

    families = []
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        categories = {
            category: sum(row["classification"] == category for row in members)
            for category in (
                "data_determined", "correlated", "predicate_regularized", "undetermined",
            )
        }
        families.append({
            "family": family,
            "parameter_count": len(members),
            "minimum_experimental_identifiability": float(min(
                row["experimental_identifiability"] for row in members
            )),
            "median_experimental_identifiability": float(np.median([
                row["experimental_identifiability"] for row in members
            ])),
            "class_counts": categories,
        })

    null_directions = []
    for direction_index, vector in enumerate(null_vh):
        order = np.argsort(np.abs(vector))[::-1]
        dominant = [
            {"name": names[index], "scaled_loading": float(vector[index])}
            for index in order[:8] if abs(vector[index]) >= 0.05
        ]
        null_directions.append({
            "index": direction_index,
            "dominant_scaled_parameters": dominant,
        })
    return {
        "schema": "matrix.lock.parameter_identifiability.v1",
        "method": "right-singular projector of the scaled experimental Jacobian",
        "classification_thresholds": {
            "data_identifiability": 0.95,
            "absolute_augmented_correlation": 0.95,
            "dominant_null_loading": 0.05,
        },
        "parameter_count": parameter_count,
        "experimental_direction_count": data_rank,
        "predicate_regularized_direction_count": max(augmented_rank - data_rank, 0),
        "unresolved_direction_count": max(parameter_count - augmented_rank, 0),
        "parameters": rows,
        "families": families,
        "experimental_null_directions": null_directions,
    }


def select_local_channel_fit_hierarchy(
    candidates: Mapping[str, Mapping[str, object]], *,
    minimum_prospective_improvement: float = 0.02,
    maximum_holdout_degradation: float = 0.25,
) -> dict[str, object]:
    """Select nested operator families using rank and two validation layers.

    The in-range deterministic holdout guards interpolation.  The contiguous
    ``J_extrapolation`` block tests whether the added operator family carries
    transferable high-J information.  Training RMS is reported but is never a
    selection criterion.
    """
    if not 0.0 <= minimum_prospective_improvement < 1.0:
        raise ValueError("minimum prospective improvement must lie in [0, 1)")
    if maximum_holdout_degradation < 0.0:
        raise ValueError("maximum holdout degradation must be nonnegative")
    rows = []
    for label, payload in candidates.items():
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"candidate {label!r} has no metrics")
        datasets = metrics.get("datasets")
        if not isinstance(datasets, Mapping):
            raise ValueError(f"candidate {label!r} has no dataset metrics")
        holdout = datasets.get("fit_range_holdout")
        prospective = datasets.get("J_extrapolation")
        if not isinstance(holdout, Mapping) or not isinstance(prospective, Mapping):
            raise ValueError(f"candidate {label!r} lacks the two validation layers")
        parameter_count = int(metrics["parameter_count"])
        augmented_rank = int(metrics["augmented_rank"])
        rows.append({
            "label": str(label),
            "parameter_count": parameter_count,
            "data_rank": int(metrics["data_rank"]),
            "augmented_rank": augmented_rank,
            "data_rank_deficiency": parameter_count - int(metrics["data_rank"]),
            "predicate_regularized_direction_count": (
                augmented_rank - int(metrics["data_rank"])
            ),
            "unresolved_direction_count": parameter_count - augmented_rank,
            "eligible": bool(payload.get("success", False) and augmented_rank == parameter_count),
            "training_rms_MHz": float(metrics["training_rms_MHz"]),
            "holdout_rms_MHz": float(holdout["rms_MHz"]),
            "prospective_rms_MHz": float(prospective["rms_MHz"]),
        })
    rows.sort(key=lambda row: (row["parameter_count"], row["label"]))
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no full-augmented-rank converged hierarchy candidate")
    selected = eligible[0]
    decisions = []
    for candidate in eligible[1:]:
        prospective_gain = 1.0 - (
            candidate["prospective_rms_MHz"] / selected["prospective_rms_MHz"]
        )
        holdout_change = (
            candidate["holdout_rms_MHz"] / selected["holdout_rms_MHz"] - 1.0
        )
        accepted = bool(
            prospective_gain >= minimum_prospective_improvement
            and holdout_change <= maximum_holdout_degradation
        )
        decisions.append({
            "from": selected["label"], "to": candidate["label"],
            "prospective_relative_improvement": prospective_gain,
            "holdout_relative_change": holdout_change,
            "accepted": accepted,
            "classification": (
                "predictively_necessary_promotion" if accepted
                else "rejected_by_validation"
            ),
        })
        if accepted:
            selected = candidate
    for row in rows:
        if not row["eligible"]:
            row["validation_status"] = "ineligible"
        elif row["label"] == selected["label"]:
            row["validation_status"] = "selected"
        elif any(
            decision["to"] == row["label"] and not decision["accepted"]
            for decision in decisions
        ):
            row["validation_status"] = "rejected_by_validation"
        else:
            row["validation_status"] = "accepted_then_superseded"
    return {
        "schema": "matrix.lock.local_channel_hierarchy.v1",
        "selection_uses_training_rms": False,
        "minimum_prospective_improvement": minimum_prospective_improvement,
        "maximum_holdout_degradation": maximum_holdout_degradation,
        "candidates": rows,
        "decisions": decisions,
        "selected": selected["label"],
        "identifiability_summary": {
            "selected_model": selected["label"],
            "parameter_count": selected["parameter_count"],
            "experimental_direction_count": selected["data_rank"],
            "predicate_regularized_direction_count": (
                selected["augmented_rank"] - selected["data_rank"]
            ),
            "unresolved_direction_count": (
                selected["parameter_count"] - selected["augmented_rank"]
            ),
            "interpretation": (
                "model selected by holdout and prospective validation; individual "
                "parameters require the fit-level singular-projector report"
            ),
        },
    }


def build_local_channel_fit_setup(
    energies_MHz: np.ndarray,
    quadratic_MHz: np.ndarray,
    *,
    calpgm_model: Mapping[str, object] | None = None,
    distortion_order: int | None = None,
    ras_interaction_order: int | None = None,
) -> dict[str, object]:
    """Build deterministic predicates and topology without molecule code."""
    energies = np.asarray(energies_MHz, dtype=float)
    quadratic = np.asarray(quadratic_MHz)
    states = len(energies)
    if quadratic.shape != (3, 3, states, states):
        raise ValueError("quadratic_MHz must have shape (3, 3, states, states)")
    discovered = None
    if calpgm_model is not None:
        from .calpgm_project import infer_calpgm_local_channel
        discovered = infer_calpgm_local_channel(calpgm_model)
        if int(discovered["state_count"]) != states:
            raise ValueError("CALPGM and contracted QM state counts differ")
    distortion_order = int(
        (discovered["distortion_order"] if discovered else 4)
        if distortion_order is None else distortion_order
    )
    if distortion_order not in {4, 6, 8}:
        raise ValueError("distortion_order must be 4, 6, or 8")

    specifications: list[LocalChannelRASInteraction] = []
    if discovered is not None:
        for row in discovered["ras_interactions"]:
            order = int(row["order"] if ras_interaction_order is None else ras_interaction_order)
            if order:
                specifications.append(LocalChannelRASInteraction(
                    int(row["left"]), int(row["right"]), str(row["axes"]), order,
                ))
    elif ras_interaction_order:
        axis_pairs = ((0, 1, "ab"), (0, 2, "ac"), (1, 2, "bc"))
        scale = max(float(np.max(np.abs(quadratic))), 1.0)
        tolerance = 100.0 * np.finfo(float).eps * scale
        for left in range(states):
            for right in range(left + 1, states):
                candidates = [
                    (abs(0.5 * (quadratic[a, b, left, right] + quadratic[b, a, left, right])), axes)
                    for a, b, axes in axis_pairs
                ]
                amplitude, axes = max(candidates, key=lambda item: float(item[0]))
                if float(amplitude) > tolerance:
                    specifications.append(LocalChannelRASInteraction(
                        left, right, axes, int(ras_interaction_order),
                    ))

    predicates: dict[str, tuple[float, float]] = {}
    for state in range(states):
        if state:
            gap = float(energies[state] - energies[0])
            predicates[f"gap_{state}_MHz"] = (gap, max(abs(gap), 20.0 * 29979.2458))
        for axis, name in enumerate("ABC"):
            value = float(quadratic[axis, axis, state, state].real)
            predicates[f"{name}_{state}_MHz"] = (value, max(0.05 * abs(value), 100.0))
        scales = (
            (4, ("DJ", "DJK", "DK", "d1", "d2"), 5.0e-2),
            (6, ("HJ", "HJK", "HKJ", "HK", "h1", "h2", "h3"), 1.0e-6),
            (8, ("LJ", "LJJK", "LJK", "LKKJ", "LK", "l1", "l2", "l3", "l4"), 2.0e-12),
        )
        for order, names, width in scales:
            if order <= distortion_order:
                predicates.update({f"{name}_{state}_MHz": (0.0, width) for name in names})
    interaction_scales = (
        (4, ("FK", "FJ", "F2"), 1.0e-3),
        (6, ("FKK", "FJK", "F2J"), 1.0e-8),
        (8, ("FJKK", "FJJK"), 1.0e-12),
    )
    for specification in specifications:
        a, b = ("abc".index(axis) for axis in specification.axes)
        amplitude = float(abs(0.5 * (
            quadratic[a, b, specification.left, specification.right]
            + quadratic[b, a, specification.left, specification.right]
        )))
        suffix = f"{specification.axes}_{specification.left}_{specification.right}_MHz"
        predicates[f"F_{suffix}"] = (amplitude, max(0.5 * amplitude, 10.0))
        for order, names, width in interaction_scales:
            if order <= specification.order:
                predicates.update({f"{name}_{suffix}": (0.0, width) for name in names})
    initial = (
        None if calpgm_model is None else local_channel_initial_from_calpgm(
            calpgm_model, distortion_order=distortion_order,
            ras_interaction_order=(max((item.order for item in specifications), default=0)),
        )
    )
    return {
        "predicates": predicates,
        "distortion_order": distortion_order,
        "ras_interactions": tuple(specifications),
        "initial_parameters": initial,
    }


def local_channel_initial_from_calpgm(
    model: Mapping[str, object], *, distortion_order: int | None = None,
    ras_interaction_order: int | None = None, left: int | None = None,
    right: int | None = None, axes: str | None = None,
) -> dict[str, float]:
    """Convert an automatically recognized local-channel CALPGM model.

    CALPGM represents the two diagonal tunneling components as an average
    Watson Hamiltonian plus ``E`` operators. LOCK stores the corresponding
    constants explicitly for each contracted state. Off-diagonal reduced-axis
    terms retain their published ``F`` labels. Ratio-constrained duplicate
    rows are ignored because their positive partner carries the fitted value.
    """
    from .calpgm_project import infer_calpgm_local_channel

    discovered = infer_calpgm_local_channel(model)
    interactions = discovered["ras_interactions"]
    if interactions:
        if len(interactions) != 1:
            raise ValueError("this local-channel backend requires one RAS state pair")
        inferred = interactions[0]
    else:
        inferred = {"left": 0, "right": 1, "axes": "ab", "order": 0}
    distortion_order = int(discovered["distortion_order"] if distortion_order is None else distortion_order)
    ras_interaction_order = int(inferred["order"] if ras_interaction_order is None else ras_interaction_order)
    left = int(inferred["left"] if left is None else left)
    right = int(inferred["right"] if right is None else right)
    axes = str(inferred["axes"] if axes is None else axes)
    if distortion_order not in {4, 6, 8} or ras_interaction_order not in {0, 2, 4, 6, 8}:
        raise ValueError("invalid CALPGM local-channel truncation order")
    if axes not in {"ab", "ac", "bc"} or left < 0 or right <= left:
        raise ValueError("invalid CALPGM local-channel state or axis selection")
    parameters = model.get("parameters")
    if not isinstance(parameters, (list, tuple)):
        raise ValueError("CALPGM model has no parameter array")
    values = {
        str(item.get("label", "")).strip().upper(): float(item["value"])
        for item in parameters if isinstance(item, Mapping)
        and not bool(item.get("ratio_constrained", False))
    }
    required = {"A", "B", "C", "E"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"CALPGM model lacks required two-state terms: {', '.join(missing)}")

    result = {f"gap_{right}_MHz": 2.0 * values["E"]}
    ej, ek, e2 = (values.get(name, 0.0) for name in ("EJ", "EK", "E2"))
    rotational_delta = {
        "A": ej + ek,
        "B": ej + 2.0 * e2,
        "C": ej - 2.0 * e2,
    }
    for name in "ABC":
        result[f"{name}_{left}_MHz"] = values[name] - rotational_delta[name]
        result[f"{name}_{right}_MHz"] = values[name] + rotational_delta[name]

    families = [
        (
            4,
            (("DJ", "-DJ", "EJJ"), ("DJK", "-DJK", "EJK"),
             ("DK", "-DK", "EKK"), ("d1", "D1", "E2J"),
             ("d2", "D2", "E4")),
        ),
        (
            6,
            (("HJ", "HJ", "EJJJ"), ("HJK", "HJK", "EJJK"),
             ("HKJ", "HKJ", "EJKK"), ("HK", "HK", "EKKK"),
             ("h1", "H1", "E2JJ"), ("h2", "H2", "E4J"),
             ("h3", "H3", "E6")),
        ),
        (
            8,
            (("LJ", "LJ", "EJJJJ"), ("LJJK", "LJJK", "EJJJK"),
             ("LJK", "LJK", "EJJKK"), ("LKKJ", "LKKJ", "EJKKK"),
             ("LK", "LK", "EKKKK"), ("l1", "L1", "E2JJJ"),
             ("l2", "L2", "E4JJ"), ("l3", "L3", "E6J"),
             ("l4", "L4", "E8")),
        ),
    ]
    for order, family in families:
        if order > distortion_order:
            continue
        for lock_name, average_name, difference_name in family:
            average = values.get(average_name, 0.0)
            difference = values.get(difference_name, 0.0)
            result[f"{lock_name}_{left}_MHz"] = average - difference
            result[f"{lock_name}_{right}_MHz"] = average + difference

    ras_families = (
        (2, (("F", "FAB"),)),
        (4, (("FK", "FAB_K"), ("FJ", "FAB_J"), ("F2", "F2AB"))),
        (6, (("FKK", "FAB_KK"), ("FJK", "FAB_JK"), ("F2J", "F2ABJ"))),
        (8, (("FJKK", "FABJKK"), ("FJJK", "FABJJK"))),
    )
    for order, family in ras_families:
        if order > ras_interaction_order:
            continue
        for lock_name, calpgm_name in family:
            result[f"{lock_name}_{axes}_{left}_{right}_MHz"] = values.get(calpgm_name, 0.0)
    return result


def fit_local_channel_effective_spectrum(
    energies_MHz: np.ndarray,
    linear_MHz: np.ndarray,
    quadratic_MHz: np.ndarray,
    observations: Sequence[LocalChannelObservation],
    *,
    predicates: Mapping[str, tuple[float, float]] | None = None,
    representation: str = "Ir",
    robust_loss: str = "linear",
    distortion_order: int = 4,
    ras_interactions: Sequence[LocalChannelRASInteraction] = (),
    initial_parameters: Mapping[str, float] | None = None,
    max_nfev: int = 200,
    assignment_subspace_tolerance_MHz: float = 1.0e-3,
) -> dict:
    """Fit gaps, state rotational constants, and Watson distortion terms.

    Off-diagonal LAM--rotation matrix elements are retained from the supplied
    theoretical contraction. Predicates are explicit Gaussian priors keyed by
    the reported parameter names; they never replace experimental rows.
    """
    energies = np.asarray(energies_MHz, dtype=float)
    linear = np.asarray(linear_MHz, dtype=complex)
    quadratic = np.asarray(quadratic_MHz, dtype=complex)
    states = len(energies)
    if (states < 1 or linear.shape != (3, states, states)
            or quadratic.shape != (3, 3, states, states)
            or not all(np.all(np.isfinite(value)) for value in (energies, linear, quadratic))):
        raise ValueError("contracted local-channel coefficients are invalid")
    if np.any(np.diff(energies) <= 0.0):
        raise ValueError("contracted state energies must be strictly increasing")
    if any(not np.allclose(value, value.conj().T, rtol=1e-10, atol=1e-10)
           for value in linear):
        raise ValueError("linear rotation--LAM coefficient matrices must be Hermitian")
    if any(not np.allclose(quadratic[a, b], quadratic[b, a].conj().T,
                           rtol=1e-10, atol=1e-10)
           for a in range(3) for b in range(3)):
        raise ValueError("quadratic rotation coefficient matrices violate Hermiticity")
    rows = tuple(observations)
    if not rows or any(row.role not in {"training", "validation"} for row in rows):
        raise ValueError("observations require training or validation roles")
    if not any(row.role == "training" for row in rows):
        raise ValueError("at least one training observation is required")
    if distortion_order not in {4, 6, 8}:
        raise ValueError("distortion_order must be 4, 6, or 8")
    if (not math.isfinite(assignment_subspace_tolerance_MHz)
            or assignment_subspace_tolerance_MHz <= 0.0):
        raise ValueError("assignment subspace tolerance must be positive and finite")
    interactions = tuple(ras_interactions)
    seen_interactions = set()
    for interaction in interactions:
        key = (interaction.left, interaction.right, interaction.axes)
        if (not 0 <= interaction.left < interaction.right < states
                or interaction.axes not in {"ab", "ac", "bc"}
                or interaction.order not in {2, 4, 6, 8}):
            raise ValueError("invalid local-channel RAS interaction specification")
        if key in seen_interactions:
            raise ValueError("duplicate local-channel RAS interaction")
        seen_interactions.add(key)
    for row in rows:
        if (not math.isfinite(row.frequency_MHz) or not math.isfinite(row.uncertainty_MHz)
                or row.uncertainty_MHz <= 0 or not row.components):
            raise ValueError("observed frequencies and uncertainties are invalid")
        total = sum(component.weight for component in row.components)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("blend weights must have a positive finite sum")
        for component in row.components:
            if (not math.isfinite(component.weight) or component.weight <= 0
                    or min(component.upper.parent, component.lower.parent) < 0
                    or max(component.upper.parent, component.lower.parent) >= states):
                raise ValueError("transition weights or local-channel parents are invalid")
            for level in (component.upper, component.lower):
                if any(int(value) != value for value in (level.J, level.Ka, level.Kc, level.parent)):
                    raise ValueError("level quantum numbers must be integers")
                if (level.J < 0 or not 0 <= level.Ka <= level.J
                        or not 0 <= level.Kc <= level.J
                        or level.Ka + level.Kc not in {level.J, level.J + 1}):
                    raise ValueError("invalid asymmetric-top J, Ka, Kc assignment")

    distortion_names = ["DJ", "DJK", "DK", "d1", "d2"]
    if distortion_order >= 6:
        distortion_names += ["HJ", "HJK", "HKJ", "HK", "h1", "h2", "h3"]
    if distortion_order >= 8:
        distortion_names += [
            "LJ", "LJJK", "LJK", "LKKJ", "LK", "l1", "l2", "l3", "l4",
        ]
    distortion_count = len(distortion_names)
    ras_names_by_order = (
        (2, ("F",)),
        (4, ("FK", "FJ", "F2")),
        (6, ("FKK", "FJK", "F2J")),
        (8, ("FJKK", "FJJK")),
    )
    interaction_names = [
        f"{name}_{interaction.axes}_{interaction.left}_{interaction.right}_MHz"
        for interaction in interactions
        for order, family in ras_names_by_order if order <= interaction.order
        for name in family
    ]
    names = ([f"gap_{state}_MHz" for state in range(1, states)]
             + [f"{axis}_{state}_MHz" for state in range(states) for axis in "ABC"]
             + [f"{name}_{state}_MHz" for state in range(states)
                for name in distortion_names]
             + interaction_names)
    gaps = energies[1:] - energies[0]
    rotations = np.asarray([
        [quadratic[axis, axis, state, state].real for axis in range(3)]
        for state in range(states)
    ])
    interaction_initial = []
    interaction_base_amplitudes = []
    interaction_phases = []
    axis_index = {axis: index for index, axis in enumerate("abc")}
    for interaction in interactions:
        a, b = (axis_index[axis] for axis in interaction.axes)
        raw = 0.5 * (
            quadratic[a, b, interaction.left, interaction.right]
            + quadratic[b, a, interaction.left, interaction.right]
        )
        if abs(raw) <= 1e-14:
            raise ValueError("RAS interaction requires a nonzero QM phase reference")
        interaction_phases.append(raw / abs(raw))
        interaction_base_amplitudes.append(float(abs(raw)))
        count = sum(len(family) for order, family in ras_names_by_order
                    if order <= interaction.order)
        interaction_initial.extend((float(abs(raw)),) + (0.0,) * (count - 1))
    x0 = np.concatenate((gaps, rotations.reshape(-1),
                         np.zeros(distortion_count * states), interaction_initial))
    name_to_index = {name: index for index, name in enumerate(names)}
    supplied_initial = dict(initial_parameters or {})
    unknown_initial = sorted(set(supplied_initial) - set(names))
    if unknown_initial:
        raise ValueError(f"unknown initial local-channel parameters: {', '.join(unknown_initial)}")
    for name, value in supplied_initial.items():
        if not math.isfinite(value):
            raise ValueError("initial local-channel parameters must be finite")
        x0[name_to_index[name]] = float(value)
    distortion_scales = np.r_[np.full(5, 0.01)]
    if distortion_order >= 6:
        distortion_scales = np.r_[distortion_scales, np.full(7, 1.0e-7)]
    if distortion_order >= 8:
        distortion_scales = np.r_[distortion_scales, np.full(9, 1.0e-12)]
    interaction_scales = []
    for interaction, initial in zip(interactions, interaction_base_amplitudes, strict=True):
        family_scales = (max(abs(initial), 1.0), 1e-3, 1e-3, 1e-3,
                         1e-8, 1e-8, 1e-8, 1e-12, 1e-12)
        count = sum(len(family) for order, family in ras_names_by_order
                    if order <= interaction.order)
        interaction_scales.extend(family_scales[:count])
    scales = np.concatenate((np.maximum(np.abs(gaps), 1.0),
                             np.maximum(np.abs(rotations).reshape(-1), 1.0),
                             np.tile(distortion_scales, states), interaction_scales))
    lower = np.full(len(x0), -np.inf)
    upper = np.full(len(x0), np.inf)
    lower[:states - 1 + 3 * states] = np.nextafter(0.0, 1.0)
    interaction_start = len(x0) - len(interaction_initial)
    interaction_offset = 0
    for interaction in interactions:
        lower[interaction_start + interaction_offset] = np.nextafter(0.0, 1.0)
        interaction_offset += sum(
            len(family) for order, family in ras_names_by_order
            if order <= interaction.order
        )
    priors = dict(predicates or {})
    unknown = sorted(set(priors) - set(names))
    if unknown:
        raise ValueError(f"unknown local-channel predicates: {', '.join(unknown)}")
    prior_indices, prior_values, prior_sigmas = [], [], []
    for name, (value, sigma) in priors.items():
        if not math.isfinite(value) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("predicates require finite centers and positive sigmas")
        prior_indices.append(name_to_index[name])
        prior_values.append(float(value))
        prior_sigmas.append(float(sigma))
    prior_indices = np.asarray(prior_indices, dtype=int)
    prior_values = np.asarray(prior_values)
    prior_sigmas = np.asarray(prior_sigmas)
    training = np.asarray([row.role == "training" for row in rows])
    observed = np.asarray([row.frequency_MHz for row in rows])
    sigma = np.asarray([row.uncertainty_MHz for row in rows])
    j_values = sorted({level.J for row in rows for component in row.components
                       for level in (component.upper, component.lower)})

    def unpack(x):
        stop = states - 1
        term = np.r_[0.0, x[:stop]]
        rotation = x[stop:stop + 3 * states].reshape(states, 3)
        distortion_stop = stop + 3 * states + states * distortion_count
        distortion = x[stop + 3 * states:distortion_stop].reshape(states, distortion_count)
        interaction = x[distortion_stop:]
        return term, rotation, distortion, interaction

    def reference_product_basis(x):
        """Build a fixed IDIAG product basis at the supplied warm start.

        Reconstructing the rotor continuation inside every residual and
        Jacobian call is both expensive and makes the objective depend on an
        iteration-local relabeling.  A fixed diabatic reference gives a smooth
        fit while the reported overlaps expose any loss of assignment purity.
        """
        _term, rotation, distortion, _interaction = unpack(x)
        result = {}
        for J in j_values:
            product_vectors = []
            labels = []
            for state in range(states):
                quartic = tuple(float(value) for value in distortion[state, :5])
                sextic = tuple(float(value) for value in distortion[state, 5:12])
                octic = tuple(float(value) for value in distortion[state, 12:21])
                rotor = diagonalize_rotor(NativeRotorParameters(
                    *rotation[state], quartic_MHz=quartic,
                    sextic_MHz=sextic or (0.0,) * 7,
                    octic_MHz=octic or (0.0,) * 9,
                    reduction="S", representation=representation,
                ), J)
                for rotor_index, (Ka, Kc, _parity) in enumerate(rotor.assignments):
                    vector = np.zeros((states, 2 * J + 1), dtype=complex)
                    vector[state] = rotor.vectors[:, rotor_index]
                    product_vectors.append(vector.reshape(-1))
                    labels.append((J, Ka, Kc, state))
            result[J] = (np.column_stack(product_vectors), tuple(labels))
        return result

    def ras_operators(angular, interaction):
        a, b = (axis_index[axis] for axis in interaction.axes)
        c = next(index for index in range(3) if index not in {a, b})
        Ja, Jb, Jc = angular[a], angular[b], angular[c]
        J2 = sum(operator @ operator for operator in angular)
        K2 = Ja @ Ja
        asymmetry = Jb @ Jb - Jc @ Jc
        base = Ja @ Jb + Jb @ Ja
        symmetrize = lambda left, right: 0.5 * (left @ right + right @ left)
        all_operators = (
            base,
            symmetrize(K2, base), J2 @ base, symmetrize(asymmetry, base),
            symmetrize(K2 @ K2, base), J2 @ symmetrize(K2, base),
            J2 @ symmetrize(asymmetry, base),
            J2 @ symmetrize(K2 @ K2, base), J2 @ J2 @ symmetrize(K2, base),
        )
        count = sum(len(family) for order, family in ras_names_by_order
                    if order <= interaction.order)
        return tuple(_hermitian(operator) for operator in all_operators[:count])

    def block_matrix(x, J):
        term, rotation, distortion, interaction_values = unpack(x)
        angular = angular_momentum_operators(J, representation)
        nrot = 2 * J + 1
        hamiltonian = np.kron(np.diag(term), np.eye(nrot)).astype(complex)
        coefficients = quadratic.copy()
        for state in range(states):
            for axis in range(3):
                coefficients[axis, axis, state, state] = rotation[state, axis]
        for a in range(3):
            hamiltonian += np.kron(linear[a], angular[a])
            for b in range(3):
                product = 0.5 * (angular[a] @ angular[b] + angular[b] @ angular[a])
                hamiltonian += np.kron(coefficients[a, b], product)
        operators = centrifugal_distortion_operators(J, "S", representation)[:distortion_count]
        for state in range(states):
            projector = np.zeros((states, states))
            projector[state, state] = 1.0
            for coefficient, operator in zip(distortion[state], operators, strict=True):
                hamiltonian += np.kron(coefficient * projector, operator)
        offset = 0
        for specification, phase in zip(interactions, interaction_phases, strict=True):
            state_operator = np.zeros((states, states), dtype=complex)
            state_operator[specification.left, specification.right] = phase
            state_operator[specification.right, specification.left] = phase.conjugate()
            operators = ras_operators(angular, specification)
            coefficients = interaction_values[offset:offset + len(operators)]
            # Replace the QM F term while retaining all other contracted tensor elements.
            a, b = (axis_index[axis] for axis in specification.axes)
            qm_amplitude = float(abs(0.5 * (
                quadratic[a, b, specification.left, specification.right]
                + quadratic[b, a, specification.left, specification.right]
            )))
            hamiltonian -= np.kron(qm_amplitude * state_operator, operators[0])
            for coefficient, operator in zip(coefficients, operators, strict=True):
                hamiltonian += np.kron(coefficient * state_operator, operator)
            offset += len(operators)
        return _hermitian(hamiltonian), angular, rotation

    reference_products = reference_product_basis(x0)
    component_cache = {}

    def label_symmetry_components():
        adjacency = np.eye(4 * states, dtype=bool)

        def node(state, ka_parity, kc_parity):
            return 4 * state + 2 * ka_parity + kc_parity

        def connect(left, right, flip):
            for ka_parity in (0, 1):
                for kc_parity in (0, 1):
                    first = node(left, ka_parity, kc_parity)
                    second = node(
                        right, ka_parity ^ flip[0], kc_parity ^ flip[1],
                    )
                    adjacency[first, second] = adjacency[second, first] = True

        linear_flip = ((0, 1), (1, 1), (1, 0))
        quadratic_flip = {
            (0, 0): (0, 0), (1, 1): (0, 0), (2, 2): (0, 0),
            (0, 1): (1, 0), (1, 0): (1, 0),
            (0, 2): (1, 1), (2, 0): (1, 1),
            (1, 2): (0, 1), (2, 1): (0, 1),
        }
        structural_scale = max(
            float(np.max(np.abs(linear))), float(np.max(np.abs(quadratic))), 1.0,
        )
        structural_tolerance = 100.0 * np.finfo(float).eps * structural_scale
        for left in range(states):
            for right in range(left + 1, states):
                for axis in range(3):
                    if abs(linear[axis, left, right]) > structural_tolerance:
                        connect(left, right, linear_flip[axis])
                for a in range(3):
                    for b in range(3):
                        if abs(quadratic[a, b, left, right]) > structural_tolerance:
                            connect(left, right, quadratic_flip[(a, b)])
        ras_flip = {"ab": (1, 0), "ac": (1, 1), "bc": (0, 1)}
        for specification in interactions:
            connect(
                specification.left, specification.right,
                ras_flip[specification.axes],
            )
        _count, labels = connected_components(
            csr_matrix(adjacency), directed=False, return_labels=True,
        )
        return labels

    symmetry_labels = label_symmetry_components()

    def symmetry_components(J, hamiltonian, derivative_operators):
        cached = component_cache.get(J)
        if cached is not None:
            return cached
        adjacency = np.zeros(hamiltonian.shape, dtype=bool)
        for operator in (hamiltonian, *derivative_operators):
            scale = float(np.max(np.abs(operator)))
            if scale:
                adjacency |= np.abs(operator) > 1.0e-13 * scale
        adjacency |= adjacency.T
        count, labels = connected_components(
            csr_matrix(adjacency), directed=False, return_labels=True,
        )
        components = tuple(np.flatnonzero(labels == index) for index in range(count))
        component_cache[J] = components
        return components

    def diagonalize_components(hamiltonian, components):
        dimension = len(hamiltonian)
        block_values = []
        block_vectors = []
        block_columns = []
        offset = 0
        for indices in components:
            values, vectors = np.linalg.eigh(hamiltonian[np.ix_(indices, indices)])
            block_values.append(values)
            embedded = np.zeros((dimension, len(indices)), dtype=complex)
            embedded[indices] = vectors
            block_vectors.append(embedded)
            block_columns.append(np.arange(offset, offset + len(indices)))
            offset += len(indices)
        values = np.concatenate(block_values)
        vectors = np.column_stack(block_vectors)
        order = np.argsort(values)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        columns = tuple(inverse[item] for item in block_columns)
        return values[order], vectors[:, order], columns

    def coupled_reference_basis():
        result = {}
        for J in j_values:
            hamiltonian, _angular, _rotation = block_matrix(x0, J)
            product_vectors, labels = reference_products[J]
            ordered = np.empty_like(product_vectors)
            character = np.empty(len(labels))
            product_masks = np.asarray([
                symmetry_labels[4 * state + 2 * (Ka % 2) + Kc % 2]
                for _J, Ka, Kc, state in labels
            ])
            for mask in np.unique(product_masks):
                slots = np.flatnonzero(product_masks == mask)
                basis = product_vectors[:, slots]
                block = _hermitian(basis.conj().T @ hamiltonian @ basis)
                _values, coefficients = np.linalg.eigh(block)
                overlap = np.abs(coefficients.conj().T) ** 2
                energy_rows, label_slots = linear_sum_assignment(-overlap)
                for energy, local_slot in zip(energy_rows, label_slots, strict=True):
                    slot = slots[local_slot]
                    ordered[:, slot] = basis @ coefficients[:, energy]
                    character[slot] = overlap[energy, local_slot]
            result[J] = (ordered, labels, character)
        return result

    coupled_references = coupled_reference_basis()
    reference_character = {
        label: float(coupled_references[J][2][index])
        for J in j_values
        for index, label in enumerate(coupled_references[J][1])
    }

    def levels(x):
        _term, rotation, distortion, _interaction_values = unpack(x)
        result = {}
        for J in j_values:
            hamiltonian, angular, _rotation = block_matrix(x, J)
            derivative_operators = []
            identity = np.eye(2 * J + 1)
            for state in range(1, states):
                projector = np.zeros((states, states)); projector[state, state] = 1.0
                derivative_operators.append(np.kron(projector, identity))
            for state in range(states):
                projector = np.zeros((states, states)); projector[state, state] = 1.0
                derivative_operators.extend(
                    np.kron(projector, angular[axis] @ angular[axis]) for axis in range(3)
                )
            distortion_operators = centrifugal_distortion_operators(
                J, "S", representation,
            )[:distortion_count]
            for state in range(states):
                projector = np.zeros((states, states)); projector[state, state] = 1.0
                derivative_operators.extend(
                    np.kron(projector, operator) for operator in distortion_operators
                )
            for specification, phase in zip(interactions, interaction_phases, strict=True):
                state_operator = np.zeros((states, states), dtype=complex)
                state_operator[specification.left, specification.right] = phase
                state_operator[specification.right, specification.left] = phase.conjugate()
                derivative_operators.extend(
                    np.kron(state_operator, operator)
                    for operator in ras_operators(angular, specification)
                )
            components = symmetry_components(J, hamiltonian, derivative_operators)
            values, vectors, component_columns = diagonalize_components(
                hamiltonian, components,
            )
            column_component = np.empty(len(values), dtype=int)
            for component_index, columns in enumerate(component_columns):
                column_component[columns] = component_index
            reference_vectors, labels, _character = coupled_references[J]
            overlap = np.abs(vectors.conj().T @ reference_vectors) ** 2
            assignment_score = np.empty_like(overlap)
            assignment_subspaces = []
            for energy in range(len(values)):
                subspace = np.flatnonzero(
                    (column_component == column_component[energy])
                    & (np.abs(values - values[energy]) <= assignment_subspace_tolerance_MHz)
                )
                assignment_subspaces.append(subspace)
                # Projector character is invariant to arbitrary rotations in
                # a degenerate eigenspace.  The individual overlap only breaks
                # ties and cannot move a label to a different subspace.
                assignment_score[energy] = (
                    np.sum(overlap[subspace], axis=0) + 1.0e-12 * overlap[energy]
                )
            energy_rows, label_slots = linear_sum_assignment(-assignment_score)
            derivatives = np.empty((len(values), len(derivative_operators)))
            for operator_index, operator in enumerate(derivative_operators):
                for indices, columns in zip(components, component_columns, strict=True):
                    block_vectors = vectors[np.ix_(indices, columns)]
                    block_operator = operator[np.ix_(indices, indices)]
                    derivatives[columns, operator_index] = np.real(np.einsum(
                        "ai,ab,bi->i", block_vectors.conj(), block_operator,
                        block_vectors, optimize=True,
                    ))
            for energy, slot in zip(energy_rows, label_slots, strict=True):
                subspace = assignment_subspaces[energy]
                result[labels[int(slot)]] = (
                    float(values[energy]), float(overlap[energy, slot]), derivatives[energy],
                    float(np.sum(overlap[subspace, slot])), int(len(subspace)),
                )
        return result

    def predict(x, assigned=None):
        if assigned is None:
            assigned = levels(x)
        values = []
        jacobian = []
        for row in rows:
            weights = np.asarray([component.weight for component in row.components])
            components = np.asarray([
                assigned[(item.upper.J, item.upper.Ka, item.upper.Kc, item.upper.parent)][0]
                - assigned[(item.lower.J, item.lower.Ka, item.lower.Kc, item.lower.parent)][0]
                for item in row.components
            ])
            component_jacobian = np.asarray([
                assigned[(item.upper.J, item.upper.Ka, item.upper.Kc, item.upper.parent)][2]
                - assigned[(item.lower.J, item.lower.Ka, item.lower.Kc, item.lower.parent)][2]
                for item in row.components
            ])
            values.append(float(weights @ components / weights.sum()))
            jacobian.append(weights @ component_jacobian / weights.sum())
        return np.asarray(values), np.asarray(jacobian)

    evaluation_cache = {"x": None, "assigned": None, "prediction": None, "jacobian": None}

    def evaluate(x):
        previous = evaluation_cache["x"]
        if previous is None or not np.array_equal(previous, x):
            assigned = levels(x)
            prediction, jacobian = predict(x, assigned)
            evaluation_cache.update(
                x=np.array(x, copy=True), assigned=assigned,
                prediction=prediction, jacobian=jacobian,
            )
        return evaluation_cache["prediction"], evaluation_cache["jacobian"]

    def residual(x):
        data = ((evaluate(x)[0] - observed) / sigma)[training]
        if len(prior_indices):
            data = np.r_[data, (x[prior_indices] - prior_values) / prior_sigmas]
        return data

    def residual_jacobian(x):
        data = (evaluate(x)[1] / sigma[:, None])[training]
        if len(prior_indices):
            prior_rows = np.zeros((len(prior_indices), len(x0)))
            prior_rows[np.arange(len(prior_indices)), prior_indices] = 1.0 / prior_sigmas
            data = np.vstack((data, prior_rows))
        return data

    optimized = least_squares(
        residual, x0, bounds=(lower, upper), x_scale=scales, method="trf",
        jac=residual_jacobian, loss=robust_loss, max_nfev=max_nfev,
        ftol=1e-9, xtol=1e-9, gtol=1e-9,
    )
    calculated, _final_jacobian = evaluate(optimized.x)
    final_assignment = evaluation_cache["assigned"]
    raw = calculated - observed
    jacobian = np.asarray(optimized.jac)
    data_rows = int(np.count_nonzero(training))
    data_jacobian = jacobian[:data_rows]
    scaled_singular = np.linalg.svd(jacobian * scales, compute_uv=False)
    _data_u, data_singular, data_vh = np.linalg.svd(
        data_jacobian * scales, full_matrices=False,
    )
    # The experimental Jacobian uses a deliberately conservative scientific
    # identifiability threshold.  Once explicit Gaussian predicates have been
    # appended, numerical rank must instead use the matrix precision limit:
    # otherwise a valid prior-only direction is incorrectly reported as a
    # singular fit merely because another spectroscopic direction is stiff.
    data_rank_threshold = 1.0e-10
    augmented_rank_threshold = (
        np.finfo(float).eps * max(jacobian.shape)
    )
    rank = int(np.count_nonzero(
        scaled_singular > scaled_singular[0] * augmented_rank_threshold,
    ))
    data_rank = int(np.count_nonzero(
        data_singular > data_singular[0] * data_rank_threshold,
    ))
    covariance = np.linalg.pinv(
        jacobian.T @ jacobian, rcond=augmented_rank_threshold,
    )
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    identifiability = _local_channel_identifiability_report(
        names, scales, data_vh, data_rank, rank, covariance, priors,
    )
    fitted_terms, fitted_rotation, fitted_distortion, fitted_interactions = unpack(optimized.x)
    used_labels = {
        (level.J, level.Ka, level.Kc, level.parent)
        for row in rows for component in row.components
        for level in (component.upper, component.lower)
    }
    assignment_purities = np.asarray(
        [final_assignment[label][1] for label in sorted(used_labels)], dtype=float
    )
    assignment_subspace_purities = np.asarray(
        [final_assignment[label][3] for label in sorted(used_labels)], dtype=float
    )
    assignment_subspace_dimensions = np.asarray(
        [final_assignment[label][4] for label in sorted(used_labels)], dtype=int
    )
    product_characters = np.asarray(
        [reference_character[label] for label in sorted(used_labels)], dtype=float
    )
    least_pure_label = sorted(used_labels)[int(np.argmin(assignment_purities))]
    dataset_metrics = {}
    for dataset in sorted({row.dataset for row in rows}):
        selected = np.asarray([row.dataset == dataset for row in rows])
        dataset_metrics[dataset] = {
            "count": int(np.count_nonzero(selected)),
            "rms_MHz": float(np.sqrt(np.mean(raw[selected] ** 2))),
            "weighted_rms": float(np.sqrt(np.mean((raw[selected] / sigma[selected]) ** 2))),
            "maximum_absolute_residual_MHz": float(np.max(np.abs(raw[selected]))),
        }
    converged = bool(optimized.success and rank == len(x0))
    if not optimized.success:
        fit_status = "optimizer_failed"
    elif rank < len(x0):
        fit_status = "rank_deficient"
    elif data_rank < len(x0):
        fit_status = "predicate_regularized"
    else:
        fit_status = "converged"
    return {
        "schema": "matrix.lock.local_channel_spectral_fit.v1",
        "success": converged,
        "status": fit_status,
        "optimizer": {"message": str(optimized.message), "nfev": int(optimized.nfev),
                      "cost": float(optimized.cost), "optimality": float(optimized.optimality)},
        "parameters": [
            {"name": name, "initial": float(x0[index]), "value": float(optimized.x[index]),
             "standard_error": float(errors[index]) if rank == len(x0) else None,
             "predicate": None if name not in priors else
             {"value": priors[name][0], "sigma": priors[name][1],
              "pull": float((optimized.x[index] - priors[name][0]) / priors[name][1])}}
            for index, name in enumerate(names)
        ],
        "fitted_term_origins_MHz": fitted_terms.tolist(),
        "fitted_rotational_constants_MHz": fitted_rotation.tolist(),
        "fitted_quartic_distortion_MHz": fitted_distortion[:, :5].tolist(),
        "fitted_sextic_distortion_MHz": fitted_distortion[:, 5:12].tolist(),
        "fitted_octic_distortion_MHz": fitted_distortion[:, 12:21].tolist(),
        "fitted_ras_interaction_MHz": {
            name: float(value)
            for name, value in zip(interaction_names, fitted_interactions, strict=True)
        },
        "distortion_order": distortion_order,
        "metrics": {
            "observation_count": len(rows), "component_count": sum(len(row.components) for row in rows),
            "training_count": int(np.count_nonzero(training)),
            "validation_count": int(np.count_nonzero(~training)),
            "parameter_count": len(x0), "data_rank": data_rank, "augmented_rank": rank,
            "data_rank_relative_tolerance": data_rank_threshold,
            "augmented_rank_relative_tolerance": augmented_rank_threshold,
            "training_rms_MHz": float(np.sqrt(np.mean(raw[training] ** 2))),
            "training_weighted_rms": float(np.sqrt(np.mean((raw[training] / sigma[training]) ** 2))),
            "validation_rms_MHz": float(np.sqrt(np.mean(raw[~training] ** 2))) if np.any(~training) else None,
            "validation_weighted_rms": float(np.sqrt(np.mean((raw[~training] / sigma[~training]) ** 2))) if np.any(~training) else None,
            "maximum_absolute_residual_MHz": float(np.max(np.abs(raw))),
            "predicate_count": len(priors),
            "assignment": {
                "method": "symmetry-subblock IDIAG with quasi-degenerate projector matching",
                "level_count": len(assignment_purities),
                "minimum_overlap": float(np.min(assignment_purities)),
                "fifth_percentile_overlap": float(np.percentile(assignment_purities, 5.0)),
                "median_overlap": float(np.median(assignment_purities)),
                "levels_below_half_overlap": int(np.count_nonzero(assignment_purities < 0.5)),
                "subspace_tolerance_MHz": assignment_subspace_tolerance_MHz,
                "subspace_overlap": {
                    "minimum": float(np.min(assignment_subspace_purities)),
                    "fifth_percentile": float(np.percentile(assignment_subspace_purities, 5.0)),
                    "median": float(np.median(assignment_subspace_purities)),
                    "levels_below_half": int(np.count_nonzero(
                        assignment_subspace_purities < 0.5
                    )),
                    "maximum_dimension": int(np.max(assignment_subspace_dimensions)),
                    "multidimensional_level_count": int(np.count_nonzero(
                        assignment_subspace_dimensions > 1
                    )),
                },
                "reference_product_character": {
                    "minimum_overlap": float(np.min(product_characters)),
                    "fifth_percentile_overlap": float(np.percentile(product_characters, 5.0)),
                    "median_overlap": float(np.median(product_characters)),
                    "levels_below_half_overlap": int(np.count_nonzero(product_characters < 0.5)),
                },
                "least_pure_level": {
                    "J": least_pure_label[0], "Ka": least_pure_label[1],
                    "Kc": least_pure_label[2], "parent": least_pure_label[3],
                },
            },
            "datasets": dataset_metrics,
        },
        "comparisons": [
            {"label": row.label, "role": row.role, "observed_MHz": row.frequency_MHz,
             "calculated_MHz": float(calculated[index]),
             "calculated_minus_observed_MHz": float(raw[index]),
             "standardized_residual": float(raw[index] / sigma[index]),
             "component_count": len(row.components), "dataset": row.dataset}
            for index, row in enumerate(rows)
        ],
        "scaled_singular_values": scaled_singular.tolist(),
        "scaled_experimental_singular_values": data_singular.tolist(),
        "identifiability": identifiability,
    }
