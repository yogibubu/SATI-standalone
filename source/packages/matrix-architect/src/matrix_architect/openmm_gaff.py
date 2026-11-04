"""Translate ORACLE GAFF assignments into a resident OpenMM System."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from matrix_chem.topology.elements import atomic_number

from .gaff import GaffParameterSet, default_gaff_parameter_path


def write_openmm_gaff_system(
    atoms: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    atom_types: Sequence[str],
    atom_charges: Sequence[float],
    output: Path | str,
) -> Path:
    """Write an OpenMM System from ORACLE types and the official GAFF2 table."""

    try:
        import openmm
        from openmm import app, unit
    except ImportError as exc:
        raise RuntimeError("GAFF2 translation requires the openmm Python package") from exc

    symbols = tuple(str(value) for value in atoms)
    types = tuple(str(value) for value in atom_types)
    charges = tuple(float(value) for value in atom_charges)
    canonical_bonds = tuple(
        sorted({tuple(sorted((int(left), int(right)))) for left, right in bonds})
    )
    if len(types) != len(symbols) or len(charges) != len(symbols):
        raise ValueError("GAFF2 types and charges must match the atom count")
    if any(left < 0 or right >= len(symbols) or left == right for left, right in canonical_bonds):
        raise ValueError("GAFF2 topology contains an invalid bond")
    parameters = GaffParameterSet.from_file(default_gaff_parameter_path())
    system = openmm.System()
    for symbol in symbols:
        number = int(atomic_number(symbol) or 0)
        if number < 1:
            raise ValueError(f"unknown element in GAFF2 topology: {symbol!r}")
        system.addParticle(app.element.Element.getByAtomicNumber(number).mass)

    adjacency = [set() for _ in symbols]
    bond_force = openmm.HarmonicBondForce()
    for left, right in canonical_bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
        item = parameters.bond(types[left], types[right])
        bond_force.addBond(
            left,
            right,
            item.equilibrium_angstrom * unit.angstrom,
            2.0
            * item.force_constant_kcal_per_mol_angstrom2
            * unit.kilocalorie_per_mole
            / unit.angstrom**2,
        )
    system.addForce(bond_force)

    angle_force = openmm.HarmonicAngleForce()
    for center, neighbors in enumerate(adjacency):
        ordered = sorted(neighbors)
        for offset, left in enumerate(ordered):
            for right in ordered[offset + 1 :]:
                item = parameters.angle(types[left], types[center], types[right])
                angle_force.addAngle(
                    left,
                    center,
                    right,
                    np.deg2rad(item.equilibrium_degrees) * unit.radian,
                    2.0
                    * item.force_constant_kcal_per_mol_radian2
                    * unit.kilocalorie_per_mole
                    / unit.radian**2,
                )
    system.addForce(angle_force)

    torsion_force = openmm.PeriodicTorsionForce()
    seen = set()
    for center_left, center_right in canonical_bonds:
        for left in adjacency[center_left] - {center_right}:
            for right in adjacency[center_right] - {center_left}:
                key = min(
                    (left, center_left, center_right, right),
                    (right, center_right, center_left, left),
                )
                if key in seen:
                    continue
                seen.add(key)
                for item in parameters.torsion(tuple(types[index] for index in key)):
                    torsion_force.addTorsion(
                        *key,
                        item.periodicity,
                        np.deg2rad(item.phase_degrees) * unit.radian,
                        item.barrier_kcal_per_mol
                        / item.divisor
                        * unit.kilocalorie_per_mole,
                    )
    system.addForce(torsion_force)

    nonbonded = openmm.NonbondedForce()
    for atom_type, charge in zip(types, charges, strict=True):
        item = parameters.vdw_for(atom_type)
        sigma = 2.0 * item.rmin_half_angstrom / (2.0 ** (1.0 / 6.0))
        nonbonded.addParticle(
            charge * unit.elementary_charge,
            sigma * unit.angstrom,
            item.epsilon_kcal_per_mol * unit.kilocalorie_per_mole,
        )
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nonbonded.createExceptionsFromBonds(canonical_bonds, 1.0 / 1.2, 0.5)
    system.addForce(nonbonded)

    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(openmm.XmlSerializer.serialize(system), encoding="utf-8")
    return target


__all__ = ["write_openmm_gaff_system"]
