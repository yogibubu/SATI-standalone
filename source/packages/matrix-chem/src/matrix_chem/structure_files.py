from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import MolecularGeometry
from .geometry_io import GeometryParseError, normalize_atom_symbol


def read_molfile(path: Path) -> MolecularGeometry:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    record = _first_sdf_record(lines)
    if len(record) < 4:
        raise GeometryParseError("MOL/SDF record is too short")
    counts = record[3]
    try:
        natoms = int(counts[0:3])
        nbonds = int(counts[3:6])
    except ValueError as exc:
        raise GeometryParseError("invalid MOL/SDF counts line") from exc
    atom_lines = record[4 : 4 + natoms]
    bond_lines = record[4 + natoms : 4 + natoms + nbonds]
    if len(atom_lines) != natoms or len(bond_lines) != nbonds:
        raise GeometryParseError("MOL/SDF atom or bond block is truncated")
    atoms: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for raw in atom_lines:
        parts = raw.split()
        if len(parts) < 4:
            raise GeometryParseError(f"invalid MOL atom line: {raw}")
        atoms.append(normalize_atom_symbol(parts[3]))
        coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    bond_orders: dict[tuple[int, int], float] = {}
    for raw in bond_lines:
        parts = raw.split()
        if len(parts) < 3:
            raise GeometryParseError(f"invalid MOL bond line: {raw}")
        left = int(parts[0]) - 1
        right = int(parts[1]) - 1
        order = _mol_bond_order(parts[2])
        if left < 0 or right < 0 or left >= natoms or right >= natoms or left == right:
            raise GeometryParseError(f"invalid MOL bond indexes: {raw}")
        bond_orders[tuple(sorted((left, right)))] = order
    return MolecularGeometry(
        atoms=tuple(atoms),
        coordinates_angstrom=np.asarray(coords, dtype=float),
        comment=record[0].strip(),
        source_format="sdf" if target.suffix.lower() == ".sdf" else "mol",
        source_path=target,
        metadata={
            "explicit_bond_orders": bond_orders,
            "bond_order_source": "SDF/MOL explicit bond block",
        },
    )


def read_mol2(path: Path) -> MolecularGeometry:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    molecule_name = target.stem
    atoms: dict[int, tuple[str, tuple[float, float, float]]] = {}
    raw_bonds: list[tuple[int, int, float]] = []
    section = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@<TRIPOS>"):
            section = line.upper()
            continue
        if section == "@<TRIPOS>MOLECULE":
            molecule_name = line
            section = "@<TRIPOS>MOLECULE_SEEN"
            continue
        if section == "@<TRIPOS>ATOM":
            parts = line.split()
            if len(parts) < 6:
                raise GeometryParseError(f"invalid MOL2 atom line: {raw}")
            idx = int(parts[0])
            symbol = _mol2_symbol(parts[1], parts[5])
            atoms[idx] = (symbol, (float(parts[2]), float(parts[3]), float(parts[4])))
            continue
        if section == "@<TRIPOS>BOND":
            parts = line.split()
            if len(parts) < 4:
                raise GeometryParseError(f"invalid MOL2 bond line: {raw}")
            left = int(parts[1])
            right = int(parts[2])
            if left == right:
                raise GeometryParseError(f"invalid MOL2 bond indexes: {raw}")
            raw_bonds.append((left, right, _mol2_bond_order(parts[3])))
    if not atoms:
        raise GeometryParseError("MOL2 file contains no atoms")
    ordered_ids = sorted(atoms)
    id_to_position = {atom_id: position for position, atom_id in enumerate(ordered_ids)}
    bond_orders: dict[tuple[int, int], float] = {}
    for left, right, order in raw_bonds:
        if left not in id_to_position or right not in id_to_position:
            raise GeometryParseError(f"invalid MOL2 bond indexes: {left} {right}")
        bond_orders[tuple(sorted((id_to_position[left], id_to_position[right])))] = order
    ordered = [atoms[idx] for idx in ordered_ids]
    return MolecularGeometry(
        atoms=tuple(item[0] for item in ordered),
        coordinates_angstrom=np.asarray([item[1] for item in ordered], dtype=float),
        comment=molecule_name,
        source_format="mol2",
        source_path=target,
        metadata={
            "explicit_bond_orders": bond_orders,
            "bond_order_source": "MOL2 explicit bond block",
        },
    )


def _first_sdf_record(lines: list[str]) -> list[str]:
    record: list[str] = []
    for line in lines:
        if line.strip() == "$$$$":
            break
        record.append(line)
    return record


def _mol_bond_order(token: str) -> float:
    try:
        value = int(token)
    except ValueError as exc:
        raise GeometryParseError(f"invalid MOL bond order: {token}") from exc
    if value == 4:
        return 1.5
    if value <= 0:
        return 1.0
    return float(value)


def _mol2_bond_order(token: str) -> float:
    text = token.strip().lower()
    if text in {"ar", "am"}:
        return 1.5
    if text in {"du", "un", "nc"}:
        return 1.0
    try:
        return float(text)
    except ValueError as exc:
        raise GeometryParseError(f"invalid MOL2 bond order: {token}") from exc


def _mol2_symbol(name: str, atom_type: str) -> str:
    head = atom_type.split(".", 1)[0]
    try:
        return normalize_atom_symbol(head)
    except GeometryParseError:
        return normalize_atom_symbol(name)
