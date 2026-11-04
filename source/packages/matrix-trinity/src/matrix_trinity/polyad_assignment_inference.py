"""Frequency/property inference for alternative roots of one physical polyad."""

from dataclasses import dataclass
import numpy as np
from .spectroscopic_inference import (
    assigned_columns,
    electric_dipole_rates,
    score_joint_assignments,
)
from .transition_inference_statistics import conditional_branching


@dataclass(frozen=True)
class AssignmentObservations:
    """Declared observable coordinates and their covariance, with no implicit data.

    Each selector is (kind,index), where kind is excitation_energy, rate,
    log_rate, branching, strength or log_strength. Pair indices refer to the
    supplied explicit inventory. Energy indices refer to result.states.
    """

    values: np.ndarray
    covariance: np.ndarray
    selectors: tuple

    def select(self, energies, properties, branching, pairs=None):
        selected = []
        kinds = []
        for selector in self.selectors:
            name, index, *extra = selector
            if name == "excitation_energy":
                selected.append(float(energies[index]))
                kinds.append("frequency")
                continue
            if name == "energy_combination":
                selected.append(float(sum(energies[i] * w for i, w in index)))
                kinds.append("frequency")
                continue
            if name == "gap":
                selected.append(float(properties["gaps_cm"][index]))
                kinds.append("frequency")
                continue
            if name == "absorption_area_ratio":
                from .transition_inference_statistics import thermal_net_absorption_ratio

                if pairs is None or len(extra) != 2:
                    raise ValueError("hot/cold pair indices and temperature required")
                selected.append(
                    thermal_net_absorption_ratio(
                        energies, pairs, properties["rates_s_inverse"], index, extra[0], extra[1]
                    )
                )
                kinds.append("probability")
                continue
            if name in ("rate_sum", "strength_sum"):
                array = properties["rates_s_inverse" if name == "rate_sum" else "strengths_au"]
                selected.append(float(np.sum(array[list(index)])))
                kinds.append("probability")
                continue
            if name in ("rate_ratio", "strength_ratio", "log_rate_ratio", "log_strength_ratio"):
                array = properties["rates_s_inverse" if "rate" in name else "strengths_au"]
                if len(extra) != 1 or array[extra[0]] <= 0:
                    raise ValueError("ratio requires a positive reference transition")
                value = array[index] / array[extra[0]]
                if name.startswith("log_"):
                    if value <= 0:
                        raise ValueError("log ratio requires positive transitions")
                    value = np.log(value)
                selected.append(float(value))
                kinds.append("probability")
                continue
            value = {
                "rate": properties["rates_s_inverse"],
                "strength": properties["strengths_au"],
                "branching": branching["branching_probabilities"],
                "log_rate": properties["rates_s_inverse"],
                "log_strength": properties["strengths_au"],
            }[name][index]
            if name.startswith("log_"):
                if value <= 0:
                    raise ValueError(
                        "a logarithmic observation requires a positive allowed transition"
                    )
                value = np.log(value)
            selected.append(float(value))
            kinds.append("probability")
        return np.array(selected), kinds


def compare_polyad_roots(
    workspace,
    state_index,
    pairs,
    *,
    observations=None,
    energy_combinations=None,
    frequency_values=None,
    frequency_covariance=None,
):
    """Compare complete roots while retaining the same field and lower states.

    Optional frequency combinations evaluate band origins or hot-minus-cold
    spacings. Probability observations carry their own explicit covariance.
    Without them the ranking remains frequency-only and predicted brightness
    is reported as a discriminator, never used as a fabricated measurement.
    """
    result = workspace.result
    count = len(result.states)
    pairs = np.asarray(pairs, int)
    if not 0 <= state_index < count:
        raise ValueError("candidate state index outside reference")
    block = next((b for b in result.blocks if state_index in b["indices"]), None)
    if block is None:
        raise ValueError("the requested state has no mixed polyad")
    columns = assigned_columns(result)
    ix = np.asarray(block["indices"], int)
    current = block["basis_to_root"][list(ix).index(state_index)]
    rows = []
    scores = []
    for root, energy in enumerate(block["eigenvalues_cm"]):
        vectors = columns.copy()
        # Keep a one-to-one assignment of all physical roots. A candidate
        # exchanges labels with its current owner rather than duplicating it.
        owner = int(ix[list(block["basis_to_root"]).index(root)])
        vectors[:, [state_index, owner]] = columns[:, [owner, state_index]]
        energies = np.array(result.energies_cm, copy=True)
        energies[[state_index, owner]] = result.energies_cm[[owner, state_index]]
        orders = workspace.moments(pairs, columns=vectors)
        properties = electric_dipole_rates(energies, pairs, orders)
        branching = conditional_branching(energies, pairs, properties["rates_s_inverse"])
        predicted = []
        values = []
        kinds = []
        covariances = []
        if energy_combinations is not None:
            a = np.asarray(energy_combinations, float)
            predicted.extend(a @ energies)
            values.extend(frequency_values)
            kinds.extend(["frequency"] * len(frequency_values))
            covariances.append(np.asarray(frequency_covariance, float))
        if observations is not None:
            prediction, observed_kinds = observations.select(energies, properties, branching, pairs)
            predicted.extend(prediction)
            values.extend(observations.values)
            kinds.extend(observed_kinds)
            covariances.append(np.asarray(observations.covariance, float))
        if not predicted:
            raise ValueError("at least one measured frequency or probability is required")
        covariance = np.zeros((len(values), len(values)))
        start = 0
        for part in covariances:
            covariance[start : start + len(part), start : start + len(part)] = part
            start += len(part)
        score = score_joint_assignments(
            np.array([predicted]), np.asarray(values), covariance=covariance, kinds=kinds
        )
        scores.append(float(score["scores"][0]))
        rows.append(
            dict(
                root=int(root),
                selected_reference_branch=bool(root == current),
                energy_cm=float(energy),
                parent_weight=float(abs(vectors[state_index, state_index]) ** 2),
                frequency_or_joint_score=scores[-1],
                inference=score["inference"],
                observed_probabilities=score["observed_probabilities"],
                gaps_cm=properties["gaps_cm"],
                rates_s_inverse=properties["rates_s_inverse"],
                strengths_au=properties["strengths_au"],
                conditional_branching=branching["branching_probabilities"],
            )
        )
    return dict(
        state_index=state_index,
        selected_reference_root=int(current),
        candidates=rows,
        ranking=np.argsort(scores),
        parameter_scope="fixed fitted field; alternative discrete assignments",
        uncertainty_scope="discrete predictions kept separate from conditional parameter covariance",
    )
