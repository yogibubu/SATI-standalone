from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

import numpy as np

from .geometry import MolecularGeometry
from .topology.elements import atomic_number, atomic_symbol


class GeometryParseError(ValueError):
    """Raised when a geometry source cannot be parsed into MATRIX geometry."""


def normalize_atom_symbol(token: str) -> str:
    cleaned = str(token).split("(", 1)[0].split("-", 1)[0].strip()
    if not cleaned:
        raise GeometryParseError("empty atom token")
    if cleaned.isdigit():
        number = int(cleaned)
        symbol = atomic_symbol(number)
        if number <= 0 or symbol == "??":
            raise GeometryParseError(f"invalid atomic number: {token}")
        return symbol
    match = re.match(r"([A-Za-z]{1,3})", cleaned)
    if not match:
        raise GeometryParseError(f"invalid atom token: {token}")
    text = match.group(1)
    symbol = text[0].upper() + text[1:].lower()
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise GeometryParseError(f"invalid atom token: {token}")
    return symbol


def parse_xyz_lines(
    lines: list[str],
    *,
    source_path: Path | None = None,
    source_format: str = "xyz",
) -> MolecularGeometry:
    if len(lines) < 2:
        raise GeometryParseError("XYZ input has fewer than two lines")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise GeometryParseError("first XYZ line must be an atom count") from exc
    if natoms < 0:
        raise GeometryParseError("XYZ atom count cannot be negative")

    comment = lines[1].rstrip()
    charge, multiplicity = _xyz_comment_electronic_state(comment)
    atoms: list[str] = []
    coords: list[list[float]] = []
    for raw in lines[2:]:
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 4:
            continue
        atom = normalize_atom_symbol(parts[0])
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError as exc:
            raise GeometryParseError(f"invalid XYZ coordinate line: {raw}") from exc
        atoms.append(atom)
        coords.append(xyz)
        if len(atoms) == natoms:
            break

    if len(atoms) != natoms:
        raise GeometryParseError(f"expected {natoms} atoms, found {len(atoms)}")

    return MolecularGeometry(
        atoms=tuple(atoms),
        coordinates_angstrom=np.asarray(coords, dtype=float),
        comment=comment,
        source_format=source_format,
        source_path=source_path,
        charge=charge,
        multiplicity=multiplicity,
    )


def _xyz_comment_electronic_state(comment: str) -> tuple[int | None, int | None]:
    """Read explicit ``charge=`` and ``multiplicity=`` XYZ comment metadata."""

    charge_match = re.search(
        r"(?:^|[;,\s])charge\s*=\s*([+-]?\d+)(?=$|[;,\s])",
        comment,
        re.I,
    )
    multiplicity_match = re.search(
        r"(?:^|[;,\s])multiplicity\s*=\s*(\d+)(?=$|[;,\s])",
        comment,
        re.I,
    )
    charge = None if charge_match is None else int(charge_match.group(1))
    multiplicity = None if multiplicity_match is None else int(multiplicity_match.group(1))
    if multiplicity is not None and multiplicity < 1:
        raise GeometryParseError("XYZ multiplicity metadata must be positive")
    return charge, multiplicity


def read_xyz(path: Path) -> MolecularGeometry:
    target = Path(path)
    return parse_xyz_lines(
        target.read_text(encoding="utf-8", errors="replace").splitlines(),
        source_path=target,
        source_format="xyz",
    )


def read_xyz_atoms_coords(path: Path) -> tuple[tuple[str, ...], np.ndarray, str]:
    geometry = read_xyz(path)
    return geometry.atoms, geometry.coordinates_angstrom, geometry.comment


def write_xyz(
    path: Path,
    atoms,
    coordinates_angstrom,
    *,
    comment: str = "",
    extra_lines=None,
    precision: int = 8,
) -> Path:
    target = Path(path)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    atoms_tuple = tuple(str(atom) for atom in atoms)
    if coords.shape != (len(atoms_tuple), 3):
        raise GeometryParseError("XYZ coordinates must have shape natoms x 3")
    digits = int(precision)
    if digits < 1 or digits > 17:
        raise GeometryParseError("XYZ coordinate precision must be between 1 and 17")
    lines = [str(len(atoms_tuple)), str(comment)]
    for atom, (x, y, z) in zip(atoms_tuple, coords):
        lines.append(f"{atom:2s} {x:24.{digits}f} {y:24.{digits}f} {z:24.{digits}f}")
    if extra_lines:
        lines.extend(str(line) for line in extra_lines)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


def read_enriched_xyz(path: Path) -> MolecularGeometry:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    geometry = parse_xyz_lines(lines, source_path=target, source_format="enriched_xyz")
    section_names = tuple(
        line.strip()[1:].strip().upper()
        for line in lines[2 + geometry.natoms :]
        if line.strip().startswith("#")
    )
    return MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=geometry.coordinates_angstrom,
        comment=geometry.comment,
        source_format="enriched_xyz",
        source_path=target,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        metadata={"sections": section_names},
    )


GeometrySourceKind = Literal[
    "auto",
    "smiles",
    "xyz",
    "enriched_xyz",
    "gaussian",
    "fchk",
    "mol",
    "sdf",
    "mol2",
    "molpro",
    "mrcc",
    "orca",
    "xtb",
    "pyscf",
    "et",
]


def read_geometry_with_kind(
    path: Path,
    source_kind: GeometrySourceKind = "auto",
    *,
    smiles_fallback_policy: str | None = None,
) -> MolecularGeometry:
    target = Path(path)
    kind = source_kind
    if kind == "auto":
        suffix = target.suffix.lower()
        if target.name.lower() == "xyzin" or suffix == ".xyzin":
            return read_enriched_xyz(target)
        if suffix == ".xyz" or suffix == "":
            return read_xyz(target)
        if suffix in {".smi", ".smiles"}:
            return read_smiles_file(
                target,
                fallback_policy=smiles_fallback_policy,
            )
        if suffix in {".mol", ".sdf"}:
            from .structure_files import read_molfile

            return read_molfile(target)
        if suffix == ".mol2":
            from .structure_files import read_mol2

            return read_mol2(target)
        if suffix in {".fchk", ".fch"}:
            from matrix_gaussian import read_gaussian_fchk_geometry

            return read_gaussian_fchk_geometry(target)
        if suffix in {".zmat", ".zmt"}:
            from .zmatrix import read_zmatrix

            return read_zmatrix(target)
        if suffix in {".gjf", ".gau", ".com", ".inp"}:
            from matrix_link import is_legacy_smiles_input, read_legacy_smiles_input
            from matrix_gaussian import read_gaussian_input

            if is_legacy_smiles_input(target):
                return read_legacy_smiles_input(
                    target,
                    fallback_policy=smiles_fallback_policy,
                )
            return read_gaussian_input(target)
        if suffix in {".log", ".out"}:
            detected = detect_qm_output_format(target)
            if detected is not None:
                return read_geometry_with_kind(target, detected)
            return _try_qm_output_readers(target)
        raise GeometryParseError(f"unsupported geometry format: {target}")
    if kind == "xyz":
        return read_xyz(target)
    if kind == "smiles":
        return read_smiles_file(
            target,
            fallback_policy=smiles_fallback_policy,
        )
    if kind == "enriched_xyz":
        return read_enriched_xyz(target)
    if kind in {"mol", "sdf"}:
        from .structure_files import read_molfile

        return read_molfile(target)
    if kind == "mol2":
        from .structure_files import read_mol2

        return read_mol2(target)
    if kind == "fchk":
        from matrix_gaussian import read_gaussian_fchk_geometry

        return read_gaussian_fchk_geometry(target)
    if kind == "gaussian":
        suffix = target.suffix.lower()
        if suffix in {".gjf", ".gau", ".com", ".inp"}:
            from matrix_gaussian import read_gaussian_input

            return read_gaussian_input(target)
        from matrix_gaussian import read_gaussian_log_geometry

        return read_gaussian_log_geometry(target)
    if kind == "molpro":
        from matrix_molpro import read_molpro_output_geometry

        return read_molpro_output_geometry(target)
    if kind == "mrcc":
        from matrix_mrcc import read_mrcc_output_geometry

        return read_mrcc_output_geometry(target)
    if kind == "orca":
        from matrix_orca import read_orca_output_geometry

        return read_orca_output_geometry(target)
    if kind == "xtb":
        from matrix_xtb import read_xtb_output_geometry

        return read_xtb_output_geometry(target)
    if kind == "pyscf":
        from matrix_pyscf import read_pyscf_output_geometry

        return read_pyscf_output_geometry(target)
    if kind == "et":
        from matrix_et import read_et_output_geometry

        return read_et_output_geometry(target)
    raise GeometryParseError(f"unsupported geometry source kind: {source_kind}")


def read_smiles_file(
    path: Path,
    *,
    fallback_policy: str | None = None,
) -> MolecularGeometry:
    """Read the first molecule in a conventional ``.smi`` text file."""
    target = Path(path)
    line = next(
        (raw.strip() for raw in target.read_text(encoding="utf-8").splitlines() if raw.strip()),
        "",
    )
    if not line:
        raise GeometryParseError(f"empty SMILES source: {target}")
    fields = line.split(maxsplit=1)
    smiles = fields[0]
    title = fields[1].strip() if len(fields) == 2 else target.stem
    from matrix_link import smiles_to_geometry

    return smiles_to_geometry(
        smiles,
        title=title,
        source_path=target,
        fallback_policy=fallback_policy,
    )


def read_geometry(
    path: Path,
    *,
    smiles_fallback_policy: str | None = None,
) -> MolecularGeometry:
    return read_geometry_with_kind(
        Path(path),
        "auto",
        smiles_fallback_policy=smiles_fallback_policy,
    )


def detect_qm_output_format(
    path: Path,
) -> Literal["gaussian", "molpro", "mrcc", "orca", "xtb", "pyscf", "et"] | None:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    upper = text.upper()
    if "ET TERMINATED SUCCESSFULLY!" in upper or re.search(
        r"^\s*ET(?:\s+\S+)?\s+-\s+AN ELECTRONIC STRUCTURE PROGRAM\s*$",
        upper,
        re.MULTILINE,
    ):
        return "et"
    if (
        "PROGRAM ORCA" in upper
        or "ORCA TERMINATED" in upper
        or "FINAL SINGLE POINT ENERGY" in upper
    ):
        return "orca"
    if "MATRIX_PYSCF_RESULT " in text or "MATRIX PYSCF TERMINATED" in upper:
        return "pyscf"
    if "NORMAL TERMINATION OF XTB" in upper or "GFN2-XTB" in upper:
        return "xtb"
    if (
        "GAUSSIAN" in upper
        or "STANDARD ORIENTATION:" in upper
        or "INPUT ORIENTATION:" in upper
        or "SCF DONE:" in upper
    ):
        return "gaussian"
    if "MRCC" in upper or "CHARGE OF THE SYSTEM" in upper or "SPIN MULTIPLICITY" in upper:
        return "mrcc"
    if "MOLPRO" in upper or "ATOMIC COORDINATES" in upper or "SPIN QUANTUM NUMBER" in upper:
        return "molpro"
    return None


def _try_qm_output_readers(path: Path) -> MolecularGeometry:
    errors: list[str] = []
    for kind in ("gaussian", "molpro", "mrcc", "orca", "xtb", "pyscf"):
        try:
            return read_geometry_with_kind(path, kind)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(f"{kind}: {exc}")
    raise GeometryParseError(
        f"unsupported QM output format for {path}; tried Gaussian, Molpro, MRCC, ORCA, xTB "
        "and PySCF "
        f"({'; '.join(errors)})"
    )
