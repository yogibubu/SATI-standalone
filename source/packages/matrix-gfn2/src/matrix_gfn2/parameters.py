"""Validated access to the published GFN2-xTB parameter tables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import tomllib
from typing import Mapping


# GFN2 was fitted and is distributed with this historical conversion factor.
# Replacing it by a newer CODATA value changes the fitted Hamiltonian at the
# microhartree level and therefore defines a different model.
EV_PER_HARTREE = 27.21138505
PARAMETER_SHA256 = "a835b0d47a3bdb78e5d2f1cc45737fd69e2301d0053b1cc7b5f799d6447df522"
ELEMENT_SYMBOLS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
)


@dataclass(frozen=True)
class GFN2ShellParameters:
    label: str
    level_hartree: float
    slater_exponent: float
    gaussian_count: int
    reference_occupation: float
    polynomial: float
    coordination_shift_hartree: float
    hubbard_scale: float


@dataclass(frozen=True)
class GFN2ElementParameters:
    atomic_number: int
    symbol: str
    hubbard_hartree: float
    hubbard_derivative_hartree: float
    repulsion_exponent: float
    effective_nuclear_charge: float
    electronegativity: float
    dipole_xc: float
    quadrupole_xc: float
    multipole_radius_bohr: float
    multipole_valence_coordination: float
    shells: tuple[GFN2ShellParameters, ...]


@dataclass(frozen=True)
class GFN2ParameterSet:
    name: str
    reference: str
    elements: Mapping[int, GFN2ElementParameters]
    global_parameters: Mapping[str, object]
    content_sha256: str


def _parallel(entry: Mapping[str, object], name: str, size: int) -> list[object]:
    values = list(entry[name])  # type: ignore[arg-type]
    if len(values) != size:
        raise ValueError(f"parameter {name} has {len(values)} values for {size} shells")
    return values


@lru_cache(maxsize=1)
def load_gfn2_parameters() -> GFN2ParameterSet:
    resource = files("matrix_gfn2").joinpath("data/gfn2-xtb.toml")
    raw = resource.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != PARAMETER_SHA256:
        raise RuntimeError("GFN2 parameter table checksum mismatch")
    payload = tomllib.loads(raw.decode("utf-8"))
    entries = payload["element"]
    if tuple(entries) != ELEMENT_SYMBOLS:
        raise RuntimeError("GFN2 parameter table must contain ordered H-Rn coverage")
    elements: dict[int, GFN2ElementParameters] = {}
    for atomic_number, symbol in enumerate(ELEMENT_SYMBOLS, start=1):
        entry = entries[symbol]
        labels = list(entry["shells"])
        size = len(labels)
        levels = _parallel(entry, "levels", size)
        slater = _parallel(entry, "slater", size)
        ngauss = _parallel(entry, "ngauss", size)
        refocc = _parallel(entry, "refocc", size)
        shpoly = _parallel(entry, "shpoly", size)
        kcn = _parallel(entry, "kcn", size)
        lgam = _parallel(entry, "lgam", size)
        shells = tuple(
            GFN2ShellParameters(
                label=str(labels[index]),
                level_hartree=float(levels[index]) / EV_PER_HARTREE,
                slater_exponent=float(slater[index]),
                gaussian_count=int(ngauss[index]),
                reference_occupation=float(refocc[index]),
                polynomial=float(shpoly[index]),
                coordination_shift_hartree=float(kcn[index]) / EV_PER_HARTREE,
                hubbard_scale=float(lgam[index]),
            )
            for index in range(size)
        )
        elements[atomic_number] = GFN2ElementParameters(
            atomic_number=atomic_number,
            symbol=symbol,
            hubbard_hartree=float(entry["gam"]),
            hubbard_derivative_hartree=float(entry["gam3"]),
            repulsion_exponent=float(entry["arep"]),
            effective_nuclear_charge=float(entry["zeff"]),
            electronegativity=float(entry["en"]),
            dipole_xc=float(entry["dkernel"]),
            quadrupole_xc=float(entry["qkernel"]),
            multipole_radius_bohr=float(entry["mprad"]),
            multipole_valence_coordination=float(entry["mpvcn"]),
            shells=shells,
        )
    global_parameters = {
        key: payload[key]
        for key in (
            "hamiltonian",
            "dispersion",
            "repulsion",
            "charge",
            "thirdorder",
            "multipole",
        )
    }
    return GFN2ParameterSet(
        name=str(payload["meta"]["name"]),
        reference=str(payload["meta"]["reference"]),
        elements=elements,
        global_parameters=global_parameters,
        content_sha256=digest,
    )


__all__ = [
    "ELEMENT_SYMBOLS",
    "EV_PER_HARTREE",
    "GFN2ElementParameters",
    "GFN2ParameterSet",
    "GFN2ShellParameters",
    "PARAMETER_SHA256",
    "load_gfn2_parameters",
]
