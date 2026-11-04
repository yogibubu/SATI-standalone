"""Persistent state identities and sparse term values for LAM spectra."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


@dataclass(frozen=True)
class TransitionTermValueResult:
    """Weighted term values reconstructed from an assigned transition network."""

    energies_MHz: Mapping[str, float]
    component_by_state: Mapping[str, int]
    anchored_components: tuple[int, ...]
    floating_components: tuple[int, ...]
    transition_residuals_MHz: tuple[float, ...]
    weighted_rms: float
    iterations: int


def lam_idiag_energy_order_level(
    result: Any,
    *,
    J: int,
    Ka: int,
    Kc: int,
    lam_symmetry: str,
    torsional_state: int = 0,
    anchor_symmetry: str = "A/A",
) -> Any:
    """Resolve an IDIAG-style label by its energy rank in a LAM block.

    Some coupled internal-rotation programs retain ``Ka,Kc`` fields after the
    labels have lost their rigid-rotor meaning.  In that convention the label
    identifies the energy rank established by an unperturbed anchor species.
    This routine transfers that rank to another torsional species and returns
    the unique native eigenstate used by subsequent overlap tracking.
    """
    if J not in result.blocks:
        raise ValueError(f"LAM result does not contain J={J}")
    block = result.blocks[J]
    anchor = sorted(
        (
            level for level in block.levels
            if level.lam_symmetry == anchor_symmetry
            and level.torsional_state == torsional_state
        ),
        key=lambda level: (level.energy_MHz, level.index),
    )
    target = sorted(
        (
            level for level in block.levels
            if level.lam_symmetry == lam_symmetry
            and level.torsional_state == torsional_state
        ),
        key=lambda level: (level.energy_MHz, level.index),
    )
    ranks = [
        index for index, level in enumerate(anchor)
        if level.Ka == Ka and level.Kc == Kc
    ]
    if len(ranks) != 1:
        raise ValueError(
            f"anchor {anchor_symmetry!r} has {len(ranks)} states labelled "
            f"J={J}, Ka={Ka}, Kc={Kc}, vt={torsional_state}"
        )
    rank = ranks[0]
    if rank >= len(target):
        raise ValueError(
            f"target {lam_symmetry!r} has only {len(target)} states for "
            f"IDIAG rank {rank} in J={J}, vt={torsional_state}"
        )
    return target[rank]


def solve_transition_term_values(
    transitions: Iterable[Mapping[str, Any]],
    anchors: Iterable[Mapping[str, Any]] = (),
) -> TransitionTermValueResult:
    """Solve ``E_upper - E_lower = frequency`` as a sparse weighted network.

    Each transition supplies ``lower_id``, ``upper_id``, ``frequency_MHz`` and
    optionally ``uncertainty_MHz`` and ``weight``.  Absolute term-value anchors
    supply ``state_id``, ``energy_MHz`` and optionally ``uncertainty_MHz`` and
    ``weight``.  A disconnected component without an anchor is assigned an
    explicit zero of energy and reported in ``floating_components``.
    """
    edges = tuple(_network_edge(item, index) for index, item in enumerate(transitions))
    fixed = tuple(_network_anchor(item, index) for index, item in enumerate(anchors))
    if not edges:
        raise ValueError("a term-value network requires at least one transition")
    state_ids = sorted(
        {item[0] for item in edges} | {item[1] for item in edges} | {item[0] for item in fixed}
    )
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    parent = list(range(len(state_ids)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for lower, upper, _frequency, _sigma in edges:
        union(state_index[lower], state_index[upper])
    roots = sorted({root(index) for index in range(len(state_ids))})
    component_for_root = {value: index for index, value in enumerate(roots)}
    components = {
        state_id: component_for_root[root(state_index[state_id])] for state_id in state_ids
    }
    anchored = {components[item[0]] for item in fixed}
    floating = sorted(set(components.values()) - anchored)

    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    rhs: list[float] = []

    def append_row(coefficients: Mapping[int, float], value: float, sigma: float) -> None:
        row = len(rhs)
        inverse_sigma = 1.0 / sigma
        for column, coefficient in coefficients.items():
            row_indices.append(row)
            column_indices.append(column)
            data.append(coefficient * inverse_sigma)
        rhs.append(value * inverse_sigma)

    for lower, upper, frequency, sigma in edges:
        append_row({state_index[lower]: -1.0, state_index[upper]: 1.0}, frequency, sigma)
    for state_id, energy, sigma in fixed:
        append_row({state_index[state_id]: 1.0}, energy, sigma)
    for component in floating:
        state_id = next(key for key in state_ids if components[key] == component)
        append_row({state_index[state_id]: 1.0}, 0.0, 1.0)

    matrix = coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(rhs), len(state_ids)),
        dtype=float,
    ).tocsr()
    solved = lsmr(
        matrix, np.asarray(rhs), atol=1.0e-14, btol=1.0e-14,
        conlim=1.0e14, maxiter=max(1000, 4 * len(state_ids)),
    )
    values = np.asarray(solved[0])
    residuals = tuple(
        float(values[state_index[upper]] - values[state_index[lower]] - frequency)
        for lower, upper, frequency, _sigma in edges
    )
    normalized = np.asarray(
        [residual / edge[3] for residual, edge in zip(residuals, edges, strict=True)]
    )
    return TransitionTermValueResult(
        energies_MHz={key: float(values[index]) for key, index in state_index.items()},
        component_by_state=components,
        anchored_components=tuple(sorted(anchored)),
        floating_components=tuple(floating),
        transition_residuals_MHz=residuals,
        weighted_rms=float(np.sqrt(np.mean(normalized * normalized))),
        iterations=int(solved[2]),
    )


def _network_edge(payload: Mapping[str, Any], index: int) -> tuple[str, str, float, float]:
    if not isinstance(payload, Mapping):
        raise ValueError("each network transition must be an object")
    lower, upper = str(payload["lower_id"]), str(payload["upper_id"])
    if not lower or not upper or lower == upper:
        raise ValueError(f"invalid state identities in transition {index + 1}")
    frequency = float(payload["frequency_MHz"])
    sigma = float(payload.get("uncertainty_MHz", 1.0)) / math.sqrt(
        float(payload.get("weight", 1.0))
    )
    if not math.isfinite(frequency) or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("network frequencies must be finite and uncertainties positive")
    return lower, upper, frequency, sigma


def _network_anchor(payload: Mapping[str, Any], index: int) -> tuple[str, float, float]:
    if not isinstance(payload, Mapping):
        raise ValueError("each term-value anchor must be an object")
    state_id = str(payload["state_id"])
    energy = float(payload["energy_MHz"])
    sigma = float(payload.get("uncertainty_MHz", 1.0)) / math.sqrt(
        float(payload.get("weight", 1.0))
    )
    if not state_id or not math.isfinite(energy) or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"invalid term-value anchor {index + 1}")
    return state_id, energy, sigma


__all__ = [
    "TransitionTermValueResult",
    "lam_idiag_energy_order_level",
    "solve_transition_term_values",
]
