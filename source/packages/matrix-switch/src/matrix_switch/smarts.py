"""Documented executable subset of SMARTS for MATRIX substructure queries."""

from __future__ import annotations

import re

from matrix_chem.topology.elements import atomic_symbol

from .model import SwitchMolecularGraph
from .parser import SmilesParseError, parse_smiles


SMARTS_SUBSET_SCHEMA = "matrix.switch.smarts_subset.v1"
_ATOMIC_NUMBER = re.compile(
    r"\[#(?P<number>\d+)(?P<hydrogen>H\d*)?(?P<charge>[+-]\d*)?\]"
)


class SmartsSubsetError(ValueError):
    pass


def parse_smarts(pattern: str) -> SwitchMolecularGraph:
    """Parse the element/bond/ring/stereo subset used by MATRIX.

    Supported query features are element symbols, ``*``, ``[#Z]``, isotopes,
    charges, explicit hydrogens, aromatic atoms, branches, ring closures,
    directional bonds and ``~`` (any bond). Recursive SMARTS, logical
    conjunction/disjunction, property ranges and reaction operators fail
    explicitly.
    """

    source = str(pattern).strip()
    if any(token in source for token in ("$(", "!", ";", ",", ">>")):
        raise SmartsSubsetError(
            "SMARTS logical/recursive/reaction operators are outside "
            f"{SMARTS_SUBSET_SCHEMA}"
        )

    def replace(match: re.Match[str]) -> str:
        number = int(match.group("number"))
        symbol = atomic_symbol(number)
        if symbol == "??":
            raise SmartsSubsetError(f"SMARTS atomic number is invalid: {number}")
        hydrogen = match.group("hydrogen") or ""
        charge = match.group("charge") or ""
        if not hydrogen and not charge and symbol in {
            "B",
            "C",
            "N",
            "O",
            "P",
            "S",
            "F",
            "Cl",
            "Br",
            "I",
        }:
            return symbol
        return f"[{symbol}{hydrogen}{charge}]"

    normalized = _ATOMIC_NUMBER.sub(replace, source).replace("[*]", "*")
    try:
        return parse_smiles(normalized)
    except SmilesParseError as exc:
        raise SmartsSubsetError(str(exc)) from exc


__all__ = ["SMARTS_SUBSET_SCHEMA", "SmartsSubsetError", "parse_smarts"]
