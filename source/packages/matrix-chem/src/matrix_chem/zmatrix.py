from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re

import numpy as np

from .geometry import MolecularGeometry
from .geometry_io import GeometryParseError, normalize_atom_symbol


DUMMY_ATOMS = {"X", "XX", "-1", "DUMMY", "DU"}


@dataclass(frozen=True)
class ZMatrixValue:
    raw: str | float

    def resolve(self, variables: dict[str, float]) -> float:
        if isinstance(self.raw, float):
            return self.raw
        key = str(self.raw).strip()
        if key in variables:
            return float(variables[key])
        folded = key.upper()
        folded_variables = {name.upper(): value for name, value in variables.items()}
        if folded in folded_variables:
            return float(folded_variables[folded])
        try:
            return _float_token(key)
        except ValueError as exc:
            raise GeometryParseError(f"unresolved Z-matrix variable: {key}") from exc


@dataclass(frozen=True)
class ZMatrixAtom:
    symbol: str
    bond_ref: int | None = None
    bond_length: ZMatrixValue | None = None
    angle_ref: int | None = None
    bond_angle: ZMatrixValue | None = None
    dihedral_ref: int | None = None
    dihedral_angle: ZMatrixValue | None = None

    @property
    def is_dummy(self) -> bool:
        return self.symbol.upper() in DUMMY_ATOMS


@dataclass(frozen=True)
class ZMatrix:
    atoms: tuple[ZMatrixAtom, ...]
    variables: dict[str, float] = field(default_factory=dict)
    title: str = ""
    charge: int | None = None
    multiplicity: int | None = None


def read_zmatrix(path: Path) -> MolecularGeometry:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    zmat = parse_zmatrix_text(text, title=target.stem)
    return zmatrix_to_geometry(zmat, source_path=target, source_format="zmatrix")


def parse_zmatrix_text(
    text: str,
    *,
    title: str = "",
    charge: int | None = None,
    multiplicity: int | None = None,
) -> ZMatrix:
    lines = _strip_comments(text.splitlines())
    lines = _drop_gaussian_preamble(lines)
    parsed_charge, parsed_mult, lines = _consume_charge_multiplicity(lines)
    charge = parsed_charge if parsed_charge is not None else charge
    multiplicity = parsed_mult if parsed_mult is not None else multiplicity
    geom_lines, variable_lines = _split_geometry_and_variables(lines)
    if not geom_lines:
        raise GeometryParseError("Z-matrix contains no geometry lines")
    variables = _parse_variables(variable_lines)
    atoms = tuple(_parse_zmatrix_atom(idx, line) for idx, line in enumerate(geom_lines))
    _validate_zmatrix_atoms(atoms)
    return ZMatrix(
        atoms=atoms, variables=variables, title=title, charge=charge, multiplicity=multiplicity
    )


def zmatrix_to_geometry(
    zmat: ZMatrix,
    *,
    source_path: Path | None = None,
    source_format: str = "zmatrix",
) -> MolecularGeometry:
    all_coords = _zmatrix_coordinates(zmat)
    atoms: list[str] = []
    coords: list[np.ndarray] = []
    for atom, xyz in zip(zmat.atoms, all_coords):
        if atom.is_dummy:
            continue
        atoms.append(normalize_atom_symbol(atom.symbol))
        coords.append(np.asarray(xyz, dtype=float))
    if not atoms:
        raise GeometryParseError("Z-matrix contains no real atoms")
    return MolecularGeometry(
        atoms=tuple(atoms),
        coordinates_angstrom=np.vstack(coords),
        comment=zmat.title or "Z-matrix",
        source_format=source_format,
        source_path=source_path,
        charge=zmat.charge,
        multiplicity=zmat.multiplicity,
        metadata={
            "zmatrix_atoms": len(zmat.atoms),
            "dummy_atoms": sum(a.is_dummy for a in zmat.atoms),
        },
    )


def _strip_comments(lines: list[str]) -> list[str]:
    out: list[str] = []
    for raw in lines:
        line = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _drop_gaussian_preamble(lines: list[str]) -> list[str]:
    idx = 0
    while idx < len(lines) and (lines[idx].startswith("%") or lines[idx].lower().startswith("#")):
        idx += 1
    # Gaussian title line, if followed by charge/multiplicity.
    if idx + 1 < len(lines) and _is_charge_multiplicity(lines[idx + 1]):
        return lines[idx + 1 :]
    return lines[idx:]


def _consume_charge_multiplicity(lines: list[str]) -> tuple[int | None, int | None, list[str]]:
    if lines and _is_charge_multiplicity(lines[0]):
        charge, multiplicity = (int(value) for value in lines[0].split()[:2])
        return charge, multiplicity, lines[1:]
    return None, None, lines


def _is_charge_multiplicity(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


def _split_geometry_and_variables(lines: list[str]) -> tuple[list[str], list[str]]:
    geom: list[str] = []
    variables: list[str] = []
    in_variables = False
    for line in lines:
        low = line.lower().rstrip(":")
        if low in {"variables", "variable", "constants", "constant", "parameters", "parameter"}:
            in_variables = True
            continue
        if "=" in line:
            variables.append(line)
            in_variables = True
            continue
        if in_variables:
            variables.append(line)
        else:
            geom.append(line)
    return geom, variables


def _parse_variables(lines: list[str]) -> dict[str, float]:
    variables: dict[str, float] = {}
    for line in lines:
        if not line:
            continue
        if "=" in line:
            name, value = line.split("=", 1)
        else:
            parts = line.split()
            if len(parts) != 2:
                raise GeometryParseError(f"invalid Z-matrix variable line: {line}")
            name, value = parts
        name = name.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise GeometryParseError(f"invalid Z-matrix variable name: {name}")
        try:
            variables[name] = _float_token(value)
        except ValueError as exc:
            raise GeometryParseError(f"invalid Z-matrix variable value: {line}") from exc
    return variables


def _parse_value(token: str) -> ZMatrixValue:
    try:
        return ZMatrixValue(_float_token(token))
    except ValueError:
        return ZMatrixValue(token.strip())


def _float_token(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _parse_zmatrix_atom(index: int, line: str) -> ZMatrixAtom:
    parts = line.replace(",", " ").split()
    if not parts:
        raise GeometryParseError("empty Z-matrix atom line")
    symbol = parts[0].strip()
    if symbol.upper() not in DUMMY_ATOMS:
        symbol = normalize_atom_symbol(symbol)
    expected = {0: 1, 1: 3, 2: 5}.get(index, 7)
    if len(parts) < expected:
        raise GeometryParseError(f"Z-matrix line {index + 1} has too few fields: {line}")
    if index == 0:
        return ZMatrixAtom(symbol)
    if index == 1:
        return ZMatrixAtom(symbol, _ref(parts[1]), _parse_value(parts[2]))
    if index == 2:
        return ZMatrixAtom(
            symbol, _ref(parts[1]), _parse_value(parts[2]), _ref(parts[3]), _parse_value(parts[4])
        )
    return ZMatrixAtom(
        symbol,
        _ref(parts[1]),
        _parse_value(parts[2]),
        _ref(parts[3]),
        _parse_value(parts[4]),
        _ref(parts[5]),
        _parse_value(parts[6]),
    )


def _ref(token: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise GeometryParseError(f"Z-matrix reference must be an integer: {token}") from exc
    if value < 1:
        raise GeometryParseError(f"Z-matrix references are one-based: {token}")
    return value - 1


def _validate_zmatrix_atoms(atoms: tuple[ZMatrixAtom, ...]) -> None:
    for idx, atom in enumerate(atoms):
        refs = [atom.bond_ref, atom.angle_ref, atom.dihedral_ref]
        needed = 0 if idx == 0 else 1 if idx == 1 else 2 if idx == 2 else 3
        for ref in refs[:needed]:
            if ref is None or ref >= idx:
                raise GeometryParseError(
                    f"Z-matrix atom {idx + 1} references atom {None if ref is None else ref + 1}, "
                    "but references must point to earlier atoms"
                )
        if len({ref for ref in refs[:needed] if ref is not None}) != needed:
            raise GeometryParseError(f"Z-matrix atom {idx + 1} uses duplicate references")


def _zmatrix_coordinates(zmat: ZMatrix) -> list[np.ndarray]:
    coords: list[np.ndarray] = []
    for idx, atom in enumerate(zmat.atoms):
        if idx == 0:
            coords.append(np.array([0.0, 0.0, 0.0], dtype=float))
            continue
        r = atom.bond_length.resolve(zmat.variables) if atom.bond_length is not None else 0.0
        if r < 0.0:
            raise GeometryParseError(f"negative bond length for Z-matrix atom {idx + 1}")
        if idx == 1:
            p1 = coords[atom.bond_ref]
            coords.append(p1 + np.array([0.0, 0.0, r], dtype=float))
            continue
        theta = math.radians(atom.bond_angle.resolve(zmat.variables))
        if idx == 2:
            p1 = coords[atom.bond_ref]
            p2 = coords[atom.angle_ref]
            ez = _normalize(p1 - p2)
            refv = np.array([0.0, 0.0, 1.0]) if abs(ez[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ex = _normalize(np.cross(refv, ez))
            ey = np.cross(ez, ex)
            coords.append(p1 - r * math.cos(theta) * ez + r * math.sin(theta) * ey)
            continue
        phi = math.radians(atom.dihedral_angle.resolve(zmat.variables))
        p1 = coords[atom.bond_ref]
        p2 = coords[atom.angle_ref]
        p3 = coords[atom.dihedral_ref]
        ez = _normalize(p1 - p2)
        vref = p3 - p2
        if np.linalg.norm(np.cross(ez, vref)) > 1.0e-10:
            ex = _normalize(np.cross(vref, ez))
        else:
            refv = np.array([0.0, 0.0, 1.0]) if abs(ez[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ex = _normalize(np.cross(refv, ez))
        ey = np.cross(ez, ex)
        coords.append(
            p1
            - r * math.cos(theta) * ez
            + r * math.sin(theta) * math.cos(phi) * ex
            + r * math.sin(theta) * math.sin(phi) * ey
        )
    return coords


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise GeometryParseError("degenerate Z-matrix reference geometry")
    return np.asarray(vector, dtype=float) / norm
