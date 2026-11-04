"""Resident monatomic ions for finite-domain ZAFF/SERAPH environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidentIon:
    name: str
    symbol: str
    charge: int
    aliases: tuple[str, ...]
    provenance: str = "ZAFF resident monatomic ion; formal charge"


_IONS = (
    ResidentIon("lithium", "Li", 1, ("li+", "litio")),
    ResidentIon("sodium", "Na", 1, ("na+", "sodio")),
    ResidentIon("potassium", "K", 1, ("k+", "potassio")),
    ResidentIon("fluoride", "F", -1, ("f-", "fluoruro")),
    ResidentIon("chloride", "Cl", -1, ("cl-", "cloruro")),
    ResidentIon("bromide", "Br", -1, ("br-", "bromuro")),
    ResidentIon("iodide", "I", -1, ("i-", "ioduro")),
)


def available_resident_ions() -> tuple[str, ...]:
    return tuple(item.name for item in _IONS)


def resident_ion(name: str) -> ResidentIon:
    requested = str(name).strip().casefold()
    for item in _IONS:
        if requested == item.name or requested in {
            alias.casefold() for alias in item.aliases
        }:
            return item
    raise KeyError(f"unknown resident ZAFF ion {name!r}")


__all__ = ["ResidentIon", "available_resident_ions", "resident_ion"]
