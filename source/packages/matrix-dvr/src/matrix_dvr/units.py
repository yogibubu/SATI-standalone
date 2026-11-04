"""Explicit unit conversions used by tunneling and rovibrational exports."""

from __future__ import annotations

import numpy as np

HARTREE_TO_CM1 = 219474.6313705
CM1_TO_HARTREE = 1.0 / HARTREE_TO_CM1
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


def convert_energy(values: object, source: str, target: str) -> np.ndarray:
    source, target = source.casefold(), target.casefold()
    factors = {"hartree": HARTREE_TO_CM1, "cm-1": 1.0, "cm1": 1.0}
    if source not in factors or target not in factors:
        raise ValueError("energy units must be hartree or cm-1")
    return np.asarray(values, dtype=float) * factors[source] / factors[target]


def convert_length(values: object, source: str, target: str) -> np.ndarray:
    source, target = source.casefold(), target.casefold()
    factors = {"angstrom": ANGSTROM_TO_BOHR, "bohr": 1.0}
    if source not in factors or target not in factors:
        raise ValueError("length units must be angstrom or bohr")
    return np.asarray(values, dtype=float) * factors[source] / factors[target]


__all__ = ["ANGSTROM_TO_BOHR", "BOHR_TO_ANGSTROM", "CM1_TO_HARTREE", "HARTREE_TO_CM1", "convert_energy", "convert_length"]
