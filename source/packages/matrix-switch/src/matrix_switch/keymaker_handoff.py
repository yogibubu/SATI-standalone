"""File-based SWITCH input handoff for the KEYMAKER owner chain."""

from __future__ import annotations

from pathlib import Path

from matrix_chem import read_xyz, write_xyz

from .workflow import smiles_to_cartesian


def normalize_source(
    source: Path | str, output: Path | str, *, source_kind: str = "auto"
) -> Path:
    """Write one mapped Cartesian seed without doing ORACLE perception."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    kind = source_kind.strip().casefold()
    if kind == "auto":
        kind = {".xyz": "xyz", ".smi": "smiles", ".smiles": "smiles"}.get(
            path.suffix.casefold(), ""
        )
    if kind == "xyz":
        geometry = read_xyz(path)
    elif kind == "smiles":
        line = next((row.strip() for row in path.read_text().splitlines() if row.strip()), "")
        if not line:
            raise ValueError("empty SMILES source")
        geometry = smiles_to_cartesian(line.split()[0])
    else:
        raise ValueError(f"unsupported SWITCH source format: {source_kind}")
    return write_xyz(
        Path(output).expanduser().resolve(), geometry.atoms,
        geometry.coordinates_angstrom, comment=geometry.comment,
    )
