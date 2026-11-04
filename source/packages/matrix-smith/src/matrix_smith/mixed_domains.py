"""SMITH plans for spatially mixed SONIC/SYCART optimization domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bmatrix import SparseBMatrix
from .evaluation import evaluate_gic_values_and_sparse_b_matrix
from .models import GICDefinition


SMITH_MIXED_DOMAIN_SCHEMA = "matrix.smith.mixed_domain_plan.v1"


@dataclass(frozen=True)
class MixedDomainPlan:
    """Non-overlapping block-coordinate plan owned by SMITH."""

    atom_count: int
    uncertain_atoms: tuple[int, ...]
    stable_atoms: tuple[int, ...]
    boundary_atoms: tuple[int, ...]
    stable_sonic_indices: tuple[int, ...]
    stable_sonic_identifiers: tuple[str, ...]
    excluded_sonic_identifiers: tuple[str, ...]
    stable_sonic_nnz: int
    local_sycart_rank: int
    task_regime: str
    schema: str = SMITH_MIXED_DOMAIN_SCHEMA

    def __post_init__(self) -> None:
        natoms = int(self.atom_count)
        uncertain = tuple(sorted({int(atom) for atom in self.uncertain_atoms}))
        stable = tuple(sorted({int(atom) for atom in self.stable_atoms}))
        boundary = tuple(sorted({int(atom) for atom in self.boundary_atoms}))
        if natoms <= 0 or set(uncertain).intersection(stable):
            raise ValueError("invalid mixed-domain atom partition")
        if set(uncertain).union(stable) != set(range(natoms)):
            raise ValueError("mixed-domain atom partition must cover the molecule")
        if not set(boundary).issubset(stable):
            raise ValueError("mixed-domain boundary must be stable")
        if int(self.local_sycart_rank) != 3 * len(uncertain):
            raise ValueError("local SYCART rank must contain three axes per uncertain atom")
        regime = str(self.task_regime).strip().upper().replace("-", "_")
        if regime not in {"MINIMUM", "TRANSITION_STATE"}:
            raise ValueError("mixed-domain plan has an unsupported task regime")
        object.__setattr__(self, "atom_count", natoms)
        object.__setattr__(self, "uncertain_atoms", uncertain)
        object.__setattr__(self, "stable_atoms", stable)
        object.__setattr__(self, "boundary_atoms", boundary)
        object.__setattr__(self, "task_regime", regime)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "task_regime": self.task_regime,
            "atom_count": self.atom_count,
            "uncertain_atoms_zero_based": list(self.uncertain_atoms),
            "stable_atoms_zero_based": list(self.stable_atoms),
            "boundary_atoms_zero_based": list(self.boundary_atoms),
            "stable_sonic_indices": list(self.stable_sonic_indices),
            "stable_sonic_identifiers": list(self.stable_sonic_identifiers),
            "excluded_sonic_identifiers": list(self.excluded_sonic_identifiers),
            "stable_sonic_nnz": self.stable_sonic_nnz,
            "local_sycart_rank": self.local_sycart_rank,
            "schedule": "LOCAL_SYCART_THEN_STABLE_SONIC_THEN_REASSESS",
        }


def build_mixed_domain_plan(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray,
    *,
    uncertain_atom_indices: Sequence[int],
    boundary_atom_indices: Sequence[int] = (),
    task_regime: str = "MINIMUM",
) -> MixedDomainPlan:
    """Keep only SONIC rows independent of the uncertain Cartesian domain."""

    if not isinstance(definition, GICDefinition):
        raise TypeError("mixed-domain planning requires a frozen GICDefinition")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("mixed-domain coordinates must be finite natoms by 3")
    natoms = len(coordinates)
    uncertain = tuple(sorted({int(atom) for atom in uncertain_atom_indices}))
    if any(atom < 0 or atom >= natoms for atom in uncertain):
        raise ValueError("mixed-domain uncertain atom lies outside the molecule")
    boundary = tuple(sorted({int(atom) for atom in boundary_atom_indices}))
    if any(atom < 0 or atom >= natoms or atom in uncertain for atom in boundary):
        raise ValueError("mixed-domain boundary atom is invalid")

    _values, b_matrix = evaluate_gic_values_and_sparse_b_matrix(
        definition,
        coordinates_angstrom=coordinates,
    )
    touched = set(b_matrix.rows_touching_atoms(uncertain))
    retained = tuple(index for index in range(b_matrix.row_count) if index not in touched)
    retained_matrix = SparseBMatrix(
        rows=tuple(b_matrix.rows[index] for index in retained),
        column_count=b_matrix.column_count,
        row_labels=tuple(b_matrix.row_labels[index] for index in retained),
        backend="matrix-smith-stable-sonic-domain.v1",
    )
    if retained_matrix.row_count and retained_matrix.structural_rank() != retained_matrix.row_count:
        raise ValueError("stable SONIC domain lost structural row independence")
    # The display name is not an identity: local generators may legitimately
    # reuse it in distinct fragments or symmetry domains.  The frozen GIC
    # identifier is unique and is the only safe key across SMITH and LINK.
    identifiers = tuple(str(gic.identifier) for gic in definition.gics)
    return MixedDomainPlan(
        atom_count=natoms,
        uncertain_atoms=uncertain,
        stable_atoms=tuple(atom for atom in range(natoms) if atom not in uncertain),
        boundary_atoms=boundary,
        stable_sonic_indices=retained,
        stable_sonic_identifiers=tuple(identifiers[index] for index in retained),
        excluded_sonic_identifiers=tuple(identifiers[index] for index in sorted(touched)),
        stable_sonic_nnz=retained_matrix.nnz,
        local_sycart_rank=3 * len(uncertain),
        task_regime=task_regime,
    )


def build_global_sycart_domain_plan(
    atom_count: int,
    *,
    task_regime: str = "MINIMUM",
) -> MixedDomainPlan:
    """Build the chart-free plan used when ORACLE certifies no stable topology.

    This constructor deliberately needs neither primitive coordinates nor a
    GIC definition.  It therefore keeps topology-free aggregates and wholly
    reactive geometries on the matrix-free Cartesian route without asking
    SMITH to invent a chemical chart.
    """

    natoms = int(atom_count)
    if natoms <= 0:
        raise ValueError("global SYCART planning requires at least one atom")
    return MixedDomainPlan(
        atom_count=natoms,
        uncertain_atoms=tuple(range(natoms)),
        stable_atoms=(),
        boundary_atoms=(),
        stable_sonic_indices=(),
        stable_sonic_identifiers=(),
        excluded_sonic_identifiers=(),
        stable_sonic_nnz=0,
        local_sycart_rank=3 * natoms,
        task_regime=task_regime,
    )


def local_sycart_axis_labels(atom_indices_zero_based: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        f"LOCAL_SYCART_A{int(atom) + 1:06d}_{axis}"
        for atom in atom_indices_zero_based
        for axis in "XYZ"
    )


__all__ = [
    "SMITH_MIXED_DOMAIN_SCHEMA",
    "MixedDomainPlan",
    "build_global_sycart_domain_plan",
    "build_mixed_domain_plan",
    "local_sycart_axis_labels",
]
