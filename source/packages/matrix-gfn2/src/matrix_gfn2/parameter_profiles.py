"""Frozen, atom-resolved orbital-level profiles for the native GFN2 model.

The published element table is never edited. A profile supplies constant
level offsets to the existing H0 value and derivative kernels. Atom membership
is explicit and stays fixed throughout a geometry or response calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

from .hamiltonian import OrbitalParameters
from .parameters import EV_PER_HARTREE


@dataclass(frozen=True)
class GFN2OnsiteProfile:
    """One constant eV offset per explicitly selected atom, on all its shells."""

    atom_shifts_ev: tuple[tuple[int, float], ...]
    name: str = "atom-onsite-v1"

    def __post_init__(self) -> None:
        entries = []
        for atom, value in self.atom_shifts_ev:
            if isinstance(atom, bool) or not isinstance(atom, Integral) or atom < 0:
                raise ValueError("profile atom indices must be nonnegative integers")
            if isinstance(value, bool) or not isfinite(float(value)):
                raise ValueError("profile orbital shifts must be finite eV values")
            entries.append((int(atom), float(value)))
        if len({atom for atom, _ in entries}) != len(entries):
            raise ValueError("an atom may occur only once in an onsite profile")
        if not self.name.strip():
            raise ValueError("an onsite profile requires a nonempty name")
        object.__setattr__(self, "atom_shifts_ev", tuple(sorted(entries)))

    @property
    def is_identity(self) -> bool:
        return all(value == 0.0 for _, value in self.atom_shifts_ev)

    @property
    def content_sha256(self) -> str:
        payload = {"schema": "matrix.gfn2.onsite-profile.v1",
                   "atom_shifts_ev": self.atom_shifts_ev}
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def apply(
        self,
        atomic_numbers: Sequence[int],
        baseline: Mapping[tuple[int, str], OrbitalParameters],
    ):
        """Return immutable per-atom shell records for the existing H0 kernels."""
        if any(atom >= len(atomic_numbers) for atom, _ in self.atom_shifts_ev):
            raise ValueError("onsite profile refers to an atom outside the molecule")
        offsets = dict(self.atom_shifts_ev)
        records = []
        for atom, number in enumerate(atomic_numbers):
            delta = offsets.get(atom, 0.0) / EV_PER_HARTREE
            records.append(MappingProxyType({
                shell: replace(value, level_hartree=value.level_hartree + delta)
                for (element, shell), value in baseline.items() if element == number
            }))
        return tuple(records)


def water_onsite_profile(
    atoms: Sequence[str],
    water_fragments: Sequence[Sequence[int]],
    *,
    oxygen_shift_ev: float,
) -> GFN2OnsiteProfile:
    """Assign a profile to explicit O/H/H fragments, never all molecular oxygen."""
    used = set()
    oxygens = []
    for fragment in water_fragments:
        indices = tuple(fragment)
        if len(indices) != 3 or any(
            isinstance(i, bool) or not isinstance(i, Integral)
            or i < 0 or i >= len(atoms) for i in indices
        ):
            raise ValueError("a water fragment requires three valid atom indices")
        if len(set(indices)) != 3 or used.intersection(indices):
            raise ValueError("water fragments must contain distinct, disjoint atoms")
        if sorted(str(atoms[i]) for i in indices) != ["H", "H", "O"]:
            raise ValueError("the declared water fragment must contain O/H/H")
        used.update(indices)
        oxygens.append(next(i for i in indices if atoms[i] == "O"))
    return GFN2OnsiteProfile(
        tuple((i, oxygen_shift_ev) for i in oxygens), "water-oxygen-onsite-v1",
    )
