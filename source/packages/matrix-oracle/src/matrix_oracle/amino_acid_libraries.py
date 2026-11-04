"""Versioned amino-acid and amino-acidic-residue fragment libraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matrix_chem import read_xyzin_geometry


AMINO_ACID_FRAGMENT_LIBRARY_SCHEMA = "matrix.oracle.amino_acid_fragment_libraries.v1"
SCIENTIFIC_POPULATION_LEVEL = "PBE0/def2-TZVP"
GDV_POPULATION_KEYWORD = "PBE1PBE/def2TZVP"
AMINO_ACID_CONFORMERS = ("I", "II")
AMINO_ACIDIC_RESIDUE_CONFORMERS = ("C5", "C7")


def load_amino_acid_fragment_libraries(root: Path | str) -> dict[str, Any]:
    """Load and validate the two dedicated LCB26 fragment-library manifests."""

    base = Path(root).expanduser().resolve()
    result = {
        "schema": AMINO_ACID_FRAGMENT_LIBRARY_SCHEMA,
        "scientific_population_level": SCIENTIFIC_POPULATION_LEVEL,
        "gdv_population_keyword": GDV_POPULATION_KEYWORD,
        "libraries": {},
    }
    for name in ("amino_acids", "aminoacidic_residues"):
        path = base / name / "manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid amino-acid fragment library: {path}") from exc
        if payload.get("schema") != AMINO_ACID_FRAGMENT_LIBRARY_SCHEMA:
            raise ValueError(f"unsupported amino-acid fragment schema: {path}")
        _validate_library(name, payload)
        result["libraries"][name] = payload
    return result


def _validate_library(name: str, payload: dict[str, Any]) -> None:
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError(f"{name} library must contain exactly 20 records")
    identifiers = {str(row.get("three_letter", "")) for row in rows}
    if len(identifiers) != 20 or "" in identifiers:
        raise ValueError(f"{name} library identifiers are not unique")
    expected = AMINO_ACID_CONFORMERS if name == "amino_acids" else AMINO_ACIDIC_RESIDUE_CONFORMERS
    for row in rows:
        if row.get("status") == "COMPLETE":
            conformers = row.get("conformers", {})
            for conformer in expected:
                record = conformers.get(conformer)
                if not isinstance(record, dict):
                    raise ValueError(f"complete {name} record lacks conformer {conformer}")
                required = {"geometry_l1", "geometry_pl1", "cm5_charges_e", "mayer_bond_orders"}
                if not required <= set(record):
                    population = record.get("population_artifact")
                    population_path = Path(__file__).resolve().parents[4] / "data" / "lcb26" / name / str(population or "")
                    try:
                        population_payload = json.loads(population_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError(f"complete {name} record lacks QM artifacts: {row['three_letter']}/{conformer}") from exc
                    if not required <= set(population_payload):
                        raise ValueError(f"complete {name} record lacks QM artifacts: {row['three_letter']}/{conformer}")
        _validate_conformer_atom_contract(name, row)


def _validate_conformer_atom_contract(name: str, row: dict[str, Any]) -> None:
    """Require one atom ordering for every conformer in one catalogue entry."""

    conformers = row.get("conformers", {})
    expected = AMINO_ACID_CONFORMERS if name == "amino_acids" else AMINO_ACIDIC_RESIDUE_CONFORMERS
    sequences: dict[str, tuple[str, ...]] = {}
    for variant in expected:
        record = conformers.get(variant)
        if not isinstance(record, dict):
            continue
        for key in ("geometry_l1", "geometry_pl1"):
            raw = record.get(key)
            if not raw:
                continue
            path = Path(__file__).resolve().parents[4] / "data" / "lcb26" / name / str(raw)
            if not path.is_file():
                continue
            atoms = tuple(str(atom) for atom in read_xyzin_geometry(path).atoms)
            sequences[f"{variant}:{key}"] = atoms
    if sequences and len(set(sequences.values())) != 1:
        details = ", ".join(f"{key}={value}" for key, value in sorted(sequences.items()))
        raise ValueError(f"{name}/{row.get('three_letter')} conformers have different atom order: {details}")


__all__ = [
    "AMINO_ACID_FRAGMENT_LIBRARY_SCHEMA",
    "AMINO_ACID_CONFORMERS",
    "AMINO_ACIDIC_RESIDUE_CONFORMERS",
    "GDV_POPULATION_KEYWORD",
    "SCIENTIFIC_POPULATION_LEVEL",
    "load_amino_acid_fragment_libraries",
]
