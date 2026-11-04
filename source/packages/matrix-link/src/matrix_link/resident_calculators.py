"""Canonical resident calculator boundary used by LINK and KeyMaker.

Production semiempirical work is owned by autonomous MATRIX implementations.
External xTB/libxTB adapters are available only through explicit validation
paths and can never be selected by a production alias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


RESIDENT_CALCULATOR_SCHEMA = "matrix.link.resident-calculator.v1"
PRODUCTION_RESIDENT_CALCULATORS = ("matrix_gfn2", "gfnff")
CANDIDATE_RESIDENT_CALCULATORS = ()
REFERENCE_RESIDENT_CALCULATORS = ("gfnff_reference",)


@dataclass(frozen=True)
class ResidentCalculatorIdentity:
    name: str
    engine: str
    model: str
    owner: str
    reference_only: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RESIDENT_CALCULATOR_SCHEMA,
            "name": self.name,
            "engine": self.engine,
            "model": self.model,
            "owner": self.owner,
            "reference_only": self.reference_only,
            "process_launches": 0,
        }


def normalize_resident_calculator_name(name: str) -> str:
    key = str(name).strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "gfn2xtb": "matrix_gfn2",
        "matrixgfn2": "matrix_gfn2",
        "matrixgfn2xtb": "matrix_gfn2",
        "gfnff": "gfnff",
        "gfn2ff": "gfnff",
        "gfnffreference": "gfnff_reference",
        "libxtbreference": "gfnff_reference",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"unknown resident calculator: {name}") from exc


def resident_calculator_identity(name: str) -> ResidentCalculatorIdentity:
    normalized = normalize_resident_calculator_name(name)
    if normalized == "matrix_gfn2":
        return ResidentCalculatorIdentity(
            name="matrix_gfn2",
            engine="matrix_gfn2",
            model="MATRIX-GFN2",
            owner="MATRIX/LINK",
        )
    if normalized == "gfnff":
        return ResidentCalculatorIdentity(
            name="gfnff",
            engine="matrix_gfnff",
            model="GFN-FF",
            owner="ARCHITECT/LINK",
        )
    return ResidentCalculatorIdentity(
        name="gfnff_reference",
        engine="libxtb",
        model="GFN-FF",
        owner="matrix-xtb parity audit",
        reference_only=True,
    )


def create_resident_calculator(
    name: str,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    processors: int = 1,
    method: str = "",
    force_field: Path | str | None = None,
    solver_mode: str = "dense",
    linear_scaling_threshold: int = 128,
    alpb_solvent: str | None = None,
    topology_directory: Path | str | None = None,
    allow_reference: bool = False,
    symmetry_contract: object | None = None,
    population_analysis: str = "scc",
    scf_energy_tolerance_hartree: float | None = None,
    scf_wavefunction_tolerance_electron: float = 1.0e-6,
    scf_mixer_tolerance: float = 2.0e-5,
):
    """Create the one canonical persistent calculator for a LINK run."""

    identity = resident_calculator_identity(name)
    atom_tuple = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(atom_tuple), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("resident calculator coordinates must have shape natoms x 3")
    if identity.reference_only and not allow_reference:
        raise ValueError("libxtb/GFN-FF is restricted to explicit parity audits")
    if identity.name == "matrix_gfn2":
        if force_field is not None:
            raise ValueError(
                "MATRIX-GFN2 does not consume an ARCHITECT force-field artifact"
            )
        from .matrix_gfn2_backend import MatrixGFN2Session

        return MatrixGFN2Session(
            atom_tuple,
            charge=int(charge),
            multiplicity=int(multiplicity),
            method=method or "MATRIX-GFN2",
            processors=int(processors),
            solver_mode=solver_mode,
            linear_scaling_threshold=int(linear_scaling_threshold),
            alpb_solvent=alpb_solvent,
            symmetry_contract=symmetry_contract,
            population_analysis=population_analysis,
            scf_energy_tolerance_hartree=scf_energy_tolerance_hartree,
            scf_wavefunction_tolerance_electron=scf_wavefunction_tolerance_electron,
            scf_mixer_tolerance=scf_mixer_tolerance,
        )
    if identity.name == "gfnff":
        if force_field is None:
            raise ValueError(
                "native GFN-FF requires force_field=<complete ARCHITECT model>"
            )
        from matrix_zaff.gfnff_session import NativeGFNFFSession

        return NativeGFNFFSession(
            atom_tuple,
            force_field=force_field,
            charge=int(charge),
            multiplicity=int(multiplicity),
            processors=int(processors),
        )
    from matrix_xtb import GfnffSession

    return GfnffSession(
        atom_tuple,
        coordinates,
        charge=int(charge),
        multiplicity=int(multiplicity),
        processors=int(processors),
        topology_directory=topology_directory,
    )


__all__ = [
    "CANDIDATE_RESIDENT_CALCULATORS",
    "PRODUCTION_RESIDENT_CALCULATORS",
    "REFERENCE_RESIDENT_CALCULATORS",
    "RESIDENT_CALCULATOR_SCHEMA",
    "ResidentCalculatorIdentity",
    "create_resident_calculator",
    "normalize_resident_calculator_name",
    "resident_calculator_identity",
]
