"""Universal ZAFF parameter priors for the complete periodic table."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from matrix_chem.topology.periodic_properties import periodic_atomic_properties


ZAFF_PERIODIC_ELEMENT_PRIOR_SCHEMA = "matrix.zaff.periodic_element_prior.v1"


@dataclass(frozen=True)
class ZaffPeriodicElementPrior:
    atomic_number: int
    vdw_radius_angstrom: float
    vdw_well_depth_kcal_per_mol: float
    covalent_radius_angstrom: float
    polarizability_angstrom3: float
    electronegativity: float
    source: str
    uncertainty_class: str
    schema: str = ZAFF_PERIODIC_ELEMENT_PRIOR_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=118)
def zaff_periodic_element_prior(atomic_number: int) -> ZaffPeriodicElementPrior:
    """Return a finite ZAFF-fast atomic prior for every Z=1--118."""

    item = periodic_atomic_properties(atomic_number)
    transferred = any("TRANSFER" in source or "PRIOR" in source for source in item.sources)
    return ZaffPeriodicElementPrior(
        atomic_number=item.atomic_number,
        vdw_radius_angstrom=item.vdw_radius_angstrom,
        vdw_well_depth_kcal_per_mol=item.vdw_well_depth_kcal_per_mol,
        covalent_radius_angstrom=item.covalent_radius_angstrom,
        polarizability_angstrom3=item.polarizability_angstrom3,
        electronegativity=item.electronegativity,
        source=";".join(item.sources),
        uncertainty_class="PERIODIC_TRANSFER" if transferred else "ELEMENT_TABULATION",
    )


def extend_periodic_atomic_series(
    values: Sequence[float],
    *,
    maximum_atomic_number: int = 118,
    positive: bool = True,
) -> np.ndarray:
    """Extend a positive atomic series by group/period-nearest scaling.

    This is used only for GFN-FF-analogous factors outside the source table;
    the resulting entries retain periodic-transfer provenance in the compiled
    model and are never described as native GFN-FF parameters.
    """

    source = np.asarray(values, dtype=float).reshape(-1)
    if not len(source) or np.any(~np.isfinite(source)) or (
        positive and np.any(source <= 0.0)
    ):
        raise ValueError("periodic atomic series has invalid source values")
    end = int(maximum_atomic_number)
    if end < len(source) or end > 118:
        raise ValueError("periodic series extension target must be in [len(values), 118]")
    result = np.empty(end, dtype=float)
    result[: len(source)] = source
    for z in range(len(source) + 1, end + 1):
        target = zaff_periodic_element_prior(z)
        candidates = [
            item for item in range(1, len(source) + 1)
            if positive or abs(float(source[item - 1])) > 1.0e-15
        ]
        contributor = min(
            candidates,
            key=lambda other: _periodic_distance(z, other),
        )
        reference = zaff_periodic_element_prior(contributor)
        volume_scale = (
            target.covalent_radius_angstrom / reference.covalent_radius_angstrom
        ) ** 0.5
        transferred = float(source[contributor - 1]) * volume_scale
        result[z - 1] = max(transferred, 1.0e-12) if positive else transferred
    return result


def _periodic_distance(left: int, right: int) -> tuple[float, int]:
    a = periodic_atomic_properties(left)
    b = periodic_atomic_properties(right)
    block_penalty = 0.0 if a.block == b.block else 8.0
    group_penalty = abs(a.group - b.group)
    period_penalty = abs(a.period - b.period)
    return block_penalty + 2.0 * group_penalty + period_penalty, abs(left - right)


__all__ = [
    "ZAFF_PERIODIC_ELEMENT_PRIOR_SCHEMA",
    "ZaffPeriodicElementPrior",
    "extend_periodic_atomic_series",
    "zaff_periodic_element_prior",
]
