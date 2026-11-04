"""Import complete GDV EPT diagnostic vectors; rounded weights are insufficient."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dyson import DysonOrbital


def read_gdv_ept_dysons(path, state, *, frozen_core, method, energy_tolerance=1e-7):
    """Read SPARKS_EPT_DYSON_V1 blocks from an instrumented GDV L909 log.

    The matching checkpoint must come from the same run: MO phases cannot be
    recovered from energies. ``frozen_core`` is explicit, not guessed from the
    channel number. Only restricted electron-removal poles are supported here.
    All active orbitals and their signed coefficients must be present. Orbital
    energies qualify the active-to-full index map against the imported state.
    """
    if len(state.mo_coefficients) != 1 or not state.mo_energies_hartree:
        raise ValueError("GDV EPT import requires restricted MOs and their energies")
    if int(frozen_core) != frozen_core or frozen_core < 0:
        raise ValueError("Frozen core count must be a nonnegative integer")
    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0:
        raise ValueError("Energy tolerance must be positive")
    lines = iter(Path(path).read_text().splitlines())
    result = []
    nmo = state.mo_coefficients[0].shape[1]
    for line in lines:
        if not line.startswith("SPARKS_EPT_DYSON_V1 "):
            continue
        try:
            symmetry, root, count, occupied, virtual = map(int, line.split()[1:])
            if count != occupied + virtual or count < 1 or count + frozen_core != nmo:
                raise ValueError("Active-space dimensions do not match the checkpoint")
            pole_line = next(lines).split()
            if len(pole_line) != 3 or pole_line[0] != "SPARKS_EPT_POLE":
                raise ValueError("Missing full pole data")
            pole_ev, strength = map(float, pole_line[1:])
            coefficients = np.zeros(nmo)
            seen = set()
            for _ in range(count):
                parts = next(lines).split()
                if len(parts) != 5 or parts[0] != "SPARKS_EPT_COMPONENT":
                    raise ValueError("Missing full signed Dyson components")
                index = int(parts[1]) - 1
                energy, coefficient = map(float, parts[3:])
                if index in seen or index not in range(count):
                    raise ValueError("Invalid or duplicated active MO index")
                seen.add(index)
                full_index = index + int(frozen_core)
                if not np.isfinite(energy) or abs(energy - state.mo_energies_hartree[0][full_index]) > energy_tolerance:
                    raise ValueError("Dyson orbital energies do not match checkpoint ordering")
                coefficients[full_index] = coefficient
            if next(lines).strip() != "SPARKS_EPT_END":
                raise ValueError("Missing Dyson block terminator")
            result.append(DysonOrbital.from_mo(
                state, coefficients, pole_strength=strength,
                ionization_energy_ev=-pole_ev, channel=f"{symmetry}:{root}",
                method=method, source=str(Path(path).resolve()),
            ))
        except (StopIteration, TypeError) as error:
            raise ValueError("Truncated GDV Dyson export") from error
    if not result:
        raise ValueError("No complete GDV Dyson export; printed pole weights are not a Dyson vector")
    return tuple(result)
