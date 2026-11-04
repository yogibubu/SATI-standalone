"""Diagnostics between the two availability levels of one ORACLE model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .pipeline import build_topology_objects


@dataclass(frozen=True)
class OracleObservableComparison:
    atom_count: int
    bond_count: int
    charge_pearson: float
    charge_spearman: float
    bond_order_pearson: float
    bond_order_spearman: float
    electronegativity_charges: tuple[float, ...]
    cm5_charges: tuple[float, ...]
    bond_pairs: tuple[tuple[int, int], ...]
    pauling_bond_orders: tuple[float, ...]
    mayer_bond_orders: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "matrix.oracle.observable-comparison.v1",
            "model": "ORACLE_UNIFIED",
            "qm_level": {"charge": "CM5", "bond_order": "Mayer"},
            "geometry_level": {
                "charge": "electronegativity estimate",
                "bond_order": "Pauling estimate",
            },
            "atom_count": self.atom_count,
            "bond_count": self.bond_count,
            "charge": {
                "pearson": self.charge_pearson,
                "spearman": self.charge_spearman,
                "electronegativity": list(self.electronegativity_charges),
                "cm5": list(self.cm5_charges),
            },
            "bond_order": {
                "pearson": self.bond_order_pearson,
                "spearman": self.bond_order_spearman,
                "pairs_one_based": [list(pair) for pair in self.bond_pairs],
                "pauling": list(self.pauling_bond_orders),
                "mayer": list(self.mayer_bond_orders),
            },
        }


def compare_qm_and_geometric_observables(
    coordinates_angstrom,
    atomic_numbers,
    *,
    cm5_charges: dict[int, float],
    mayer_bond_orders: dict[tuple[int, int], float],
) -> OracleObservableComparison:
    """Compare complete zero-based QM observables with geometry-only estimates.

    This is a diagnostic of trend preservation, not a second production model.
    Production selects the paired CM5/Mayer level only when both complete
    vectors are present, and the paired electronegativity/Pauling level
    otherwise.
    """

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(numbers), 3):
        raise ValueError("coordinates must have shape natoms x 3")
    if set(cm5_charges) != set(range(len(numbers))):
        raise ValueError("CM5 comparison requires one charge for every atom")
    _continuous, discrete, _rings, estimates, _aromaticity = build_topology_objects(
        coordinates, numbers
    )
    pairs = tuple(tuple(sorted(pair)) for pair in discrete.bonds)
    missing = tuple(pair for pair in pairs if pair not in mayer_bond_orders)
    if missing:
        raise ValueError(f"Mayer comparison lacks {len(missing)} perceived bonds")
    q_est = np.asarray([estimates.charge(index) for index in range(len(numbers))])
    q_qm = np.asarray([cm5_charges[index] for index in range(len(numbers))])
    bo_est = np.asarray([estimates.bond_order(*pair) for pair in pairs])
    bo_qm = np.asarray([mayer_bond_orders[pair] for pair in pairs])
    return OracleObservableComparison(
        atom_count=len(numbers),
        bond_count=len(pairs),
        charge_pearson=_pearson(q_est, q_qm),
        charge_spearman=_pearson(_average_ranks(q_est), _average_ranks(q_qm)),
        bond_order_pearson=_pearson(bo_est, bo_qm),
        bond_order_spearman=_pearson(_average_ranks(bo_est), _average_ranks(bo_qm)),
        electronegativity_charges=tuple(float(value) for value in q_est),
        cm5_charges=tuple(float(value) for value in q_qm),
        bond_pairs=tuple((left + 1, right + 1) for left, right in pairs),
        pauling_bond_orders=tuple(float(value) for value in bo_est),
        mayer_bond_orders=tuple(float(value) for value in bo_qm),
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation requires equally sized vectors with at least two values")
    centered_left = left - np.mean(left)
    centered_right = right - np.mean(right)
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if denominator <= np.finfo(float).eps or not math.isfinite(denominator):
        raise ValueError("correlation is undefined for a constant observable vector")
    return float(np.dot(centered_left, centered_right) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks
