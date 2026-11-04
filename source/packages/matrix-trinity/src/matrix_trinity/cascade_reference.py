"""Acquire exact cascade intermediates by probability flux, without rate fitting."""
from dataclasses import replace

import numpy as np

from .gvpt2_rates import GVPT2RateReference
from .gvpt2_hybrid_rates import HybridQuantumRateModel, _key
from .gvpt2_conditional_rates import ConditionalQuantumRateModel


def _inventory(reference):
    keys = [_key(n, s, b) for n, s, b in zip(reference.occupations, reference.sectors, reference.branches)]
    a, b = reference.pairs.T
    high = np.where(reference.energies_cm[a] >= reference.energies_cm[b], a, b)
    low = np.where(high == a, b, a)
    edges = [frozenset((keys[i], keys[j])) for i, j in reference.pairs]
    return keys, high, low, edges


def exact_cascade_frontier(model, reference, initial_population):
    """Probability of first reaching a source with unacquired outgoing rates.

    Propagation uses *only exact acquired rates*. Unknown sources are absorbing
    frontiers; their reference rate values are not inspected. The probability
    bounds the chance that a trajectory ever needs the kernel on this inventory.
    It does not by itself bound an unbounded waiting time after that event.
    """
    if not isinstance(model, HybridQuantumRateModel) or not isinstance(reference, GVPT2RateReference):
        raise TypeError('hybrid model and resolved reference inventory required')
    p = np.asarray(initial_population, float)
    if (reference.coordinate_id != model.coordinate_id or p.shape != reference.energies_cm.shape
            or not np.all(np.isfinite(p)) or np.any(p < 0)
            or not np.isclose(p.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('compatible inventory and normalized initial population required')
    keys, high, low, edges = _inventory(reference)
    for i, key in enumerate(keys):
        if key in model.tail.base._exact_energies and model.tail.base._exact_energies[key] != reference.energies_cm[i]:
            raise ValueError('inventory changes an already acquired exact energy')
    for j, edge in enumerate(edges):
        if edge in model._exact_rates and model._exact_rates[edge] != (float(reference.rates_s_inverse[j]), bool(reference.allowed[j])):
            raise ValueError('inventory changes an already acquired exact rate/mask')
    outgoing = [[] for _ in p]
    for edge in np.flatnonzero(reference.allowed):
        if reference.energies_cm[high[edge]] == reference.energies_cm[low[edge]]:
            # The known photon frequency already fixes spontaneous A=0.
            continue
        outgoing[high[edge]].append(edge)
    visits, frontier = p.copy(), np.zeros(len(p))
    exact_source = np.ones(len(p), bool)
    for i, indices in enumerate(outgoing):
        exact_source[i] = (keys[i] in model.tail.base._exact_energies
            and all(edges[j] in model._exact_rates for j in indices))
    for source in np.argsort(reference.energies_cm, kind='stable')[::-1]:
        if not visits[source]:
            continue
        if not exact_source[source]:
            frontier[source] = visits[source]
            continue
        indices = outgoing[source]
        rates = np.array([model._exact_rates[edges[j]][0] for j in indices])
        total = rates.sum()
        if total > 0:
            np.add.at(visits, low[indices], visits[source]*rates/total)
    if frontier.sum() > 1.+1e-10:
        raise ArithmeticError('exact-prefix first-hit probability exceeds one')
    return frontier


def _attach_source(model, reference, source, maximum_exact_states, maximum_exact_pairs):
    keys, high, low, edges = _inventory(reference)
    previous = model.tail.base.window_reference
    retained_states = np.zeros(len(keys), bool)
    retained_pairs = high == source
    if previous is not None:
        previous_keys, _, _, previous_edges = _inventory(previous)
        lookup = {key: i for i, key in enumerate(keys)}
        if any(key not in lookup for key in previous_keys) or not set(previous_edges) <= set(edges):
            raise ValueError('new inventory must retain the acquired exact-window identities')
        retained_states[[lookup[key] for key in previous_keys]] = True
        old_edges = set(previous_edges)
        retained_pairs |= np.array([edge in old_edges for edge in edges])
    retained_states[source] = True
    retained_states[reference.pairs[retained_pairs].ravel()] = True
    if retained_states.sum() > maximum_exact_states or retained_pairs.sum() > maximum_exact_pairs:
        raise MemoryError('cascade reference acquisition exceeds exact-state/pair budget')
    mapping = np.full(len(keys), -1, int)
    mapping[retained_states] = np.arange(sum(retained_states))
    acquired = GVPT2RateReference(reference.occupations[retained_states], reference.energies_cm[retained_states],
        mapping[reference.pairs[retained_pairs]], reference.rates_s_inverse[retained_pairs],
        reference.allowed[retained_pairs], reference.coordinate_id,
        tuple(np.asarray(reference.sectors)[retained_states]), tuple(np.asarray(reference.branches)[retained_states]),
        dict(reference.provenance))
    window = dict(model.tail.base.energy_window or {})
    window['cascade_endpoint_inclusive_states'] = int(sum(retained_states))
    window['cascade_listed_pairs'] = int(sum(retained_pairs))
    energy = replace(model.tail.base, window_reference=acquired, energy_window=window)
    return HybridQuantumRateModel(ConditionalQuantumRateModel(energy, model.tail.heads,
                                                              dict(model.tail.diagnostics)))


def refine_exact_cascade_reference(model, reference, initial_population, *,
        maximum_unresolved_probability=.001, maximum_added_sources=128,
        maximum_exact_states=512, maximum_exact_pairs=10000):
    """Add the largest first-hit frontier until its remaining mass is small.

    This adapter acquires values from an existing GVPT2 inventory; no new QC or
    diagonalization is launched. Selection never consults unknown rate errors or
    unknown outgoing rate magnitudes. It is preparation-dependent and applies
    to the explicit listed-channel graph. A budget failure raises, rather than
    reporting a probability certificate for a partially acquired model.
    """
    if (not np.isfinite(maximum_unresolved_probability) or not 0 <= maximum_unresolved_probability < 1
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
                (maximum_added_sources, maximum_exact_states, maximum_exact_pairs))):
        raise ValueError('probability tolerance in [0,1) and positive acquisition budgets required')
    history = []
    for step in range(maximum_added_sources+1):
        frontier = exact_cascade_frontier(model, reference, initial_population)
        unresolved = float(frontier.sum())
        if unresolved <= maximum_unresolved_probability:
            report = dict(maximum_unresolved_probability=maximum_unresolved_probability,
                unresolved_first_hit_probability=unresolved, added_source_states=len(history),
                history=history, exact_listed_pairs=len(model._exact_rates),
                selection_uses_unknown_rates=False, waiting_time_error_bound=False,
                preparation_specific=True, inventory_completeness_asserted=False)
            window = dict(model.tail.base.energy_window or {}, cascade_refinement=report)
            energy = replace(model.tail.base, energy_window=window)
            return HybridQuantumRateModel(ConditionalQuantumRateModel(energy, model.tail.heads,
                dict(model.tail.diagnostics, cascade_refinement=report))), report
        if step == maximum_added_sources:
            raise MemoryError('cascade frontier did not reach target within source budget')
        source = int(np.argmax(frontier))
        before = len(model._exact_rates)
        model = _attach_source(model, reference, source, maximum_exact_states, maximum_exact_pairs)
        history.append(dict(occupations=reference.occupations[source].tolist(),
            sector=reference.sectors[source], branch=reference.branches[source],
            first_hit_probability=float(frontier[source]), remaining_before=unresolved,
            added_pairs=len(model._exact_rates)-before))
    raise AssertionError('unreachable refinement state')
