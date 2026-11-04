"""ORACLE-side access to the unified LCB26 reference index.

ORACLE owns perception (including ring counts); this module only selects a
reference geometry/electronic record for a requested construction.  ZAFF
parameter generation remains an ARCHITECT operation.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable

from matrix_switch import canonical_smiles, parse_smiles


class LCB26ReferenceError(ValueError):
    """Raised when an LCB26 reference cannot be selected."""


LCB26_L1_GEOMETRY_DATASET = "L1_GEOMETRY"
LCB26_HARMONIC_RECORD_SCHEMA = "matrix.lcb26.harmonic_record.v1"


@lru_cache(maxsize=256)
def _parse_json_content(content: bytes) -> Any:
    """Reuse parsing only when the resource bytes are identical."""

    return json.loads(content)


def _read_revisioned_json(path: Path) -> Any:
    try:
        return _parse_json_content(path.read_bytes())
    except OSError as exc:
        raise LCB26ReferenceError(f"missing LCB26 resource: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LCB26ReferenceError(f"invalid LCB26 JSON resource: {path}") from exc


def _normal(value: str) -> str:
    return " ".join(
        str(value).casefold().replace("_", " ").replace("-", " ").replace(":", " ").split()
    )


def query_lcb26(
    library_root: Path,
    *,
    identifier: str | None = None,
    name: str | None = None,
    alias: str | None = None,
    smiles: str | None = None,
    formula: str | None = None,
    dataset: str | None = None,
    electronic_level: str | None = None,
    elements: Iterable[str] | None = None,
    element_counts: dict[str, int] | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    open_shell: bool | None = None,
    atom_count: int | None = None,
    heavy_atom_count: int | None = None,
    ring_count: int | None = None,
    contains: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select LCB26 records for ORACLE geometry construction."""

    if limit is not None and int(limit) <= 0:
        return ()
    index_path = Path(library_root).expanduser().resolve() / "enriched" / "index.json"
    payload = _read_revisioned_json(index_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records", ()), list):
        raise LCB26ReferenceError(f"invalid LCB26 query index: {index_path}")
    aliases = [_normal(value) for value in (identifier, name, alias) if value is not None]
    canonical = canonical_smiles(parse_smiles(smiles)) if smiles is not None else None
    required = {str(element) for element in elements or ()}
    required_counts = {
        str(element): int(count) for element, count in (element_counts or {}).items()
    }
    normalized_contains = _normal(contains) if contains is not None else None
    selected = []
    for row in payload.get("records", ()):
        if aliases and not all(value in row.get("normalized_aliases", ()) for value in aliases):
            continue
        if canonical is not None and row.get("canonical_smiles") != canonical:
            continue
        if formula is not None and str(row.get("formula", "")).casefold() != formula.casefold():
            continue
        if dataset is not None and str(row.get("dataset", "")).casefold() != dataset.casefold():
            continue
        if required and not required.issubset(set(row.get("elements", ()))):
            continue
        if any(
            row.get("element_counts", {}).get(element, 0) != count
            for element, count in required_counts.items()
        ):
            continue
        scalar_filters = {
            "electronic_level": electronic_level,
            "charge": charge,
            "multiplicity": multiplicity,
            "open_shell": open_shell,
            "atom_count": atom_count,
            "heavy_atom_count": heavy_atom_count,
            "ring_count": ring_count,
        }
        if any(
            expected is not None and str(row.get(key)).casefold() != str(expected).casefold()
            for key, expected in scalar_filters.items()
        ):
            continue
        if normalized_contains is not None and not any(
            normalized_contains in value for value in row.get("normalized_aliases", ())
        ):
            continue
        selected.append(deepcopy(row))
        if limit is not None and len(selected) >= int(limit):
            break
    return tuple(selected)


def load_lcb26_reference(library_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = Path(library_root).expanduser().resolve() / str(row["record_path"])
    payload = _read_revisioned_json(path)
    if not isinstance(payload, dict):
        raise LCB26ReferenceError(f"invalid LCB26 reference: {path}")
    # Public callers receive an isolated mutable mapping, while immutable JSON
    # parsing is shared across repeated molecule preparations.
    return deepcopy(payload)


def load_lcb26_harmonic_record(
    library_root: Path,
    parent: dict[str, Any],
    *,
    identifier: str | None = None,
    method: str | None = None,
    basis: str | None = None,
) -> dict[str, Any]:
    """Load and certify one Cartesian harmonic record linked by an LCB26 entry.

    Harmonic records are deliberately separate from the standard LCB26
    electronic record: adding MP2 Hessians must not replace its PBE0 CM5
    charges or PBE0 Mayer bond orders.  The loader rejects ambiguous matches,
    atom permutations, improper frame rotations, nonsymmetric Hessians, and
    records whose dimensions differ from the parent atom list.
    """

    links = parent.get("vibro_rotational_records", ())
    if not isinstance(links, list):
        raise LCB26ReferenceError("invalid LCB26 vibro-rotational record links")
    matches: list[dict[str, Any]] = []
    for link in links:
        if not isinstance(link, dict) or not isinstance(
            link.get("record_path", link.get("path")), str
        ):
            raise LCB26ReferenceError("invalid LCB26 harmonic-record link")
        if identifier is not None and str(link.get("identifier")) != str(identifier):
            continue
        if method is not None and str(link.get("method", "")).casefold() != str(method).casefold():
            continue
        if basis is not None and str(link.get("basis", "")).casefold() != str(basis).casefold():
            continue
        matches.append(link)
    if len(matches) != 1:
        raise LCB26ReferenceError(
            f"expected one matching LCB26 harmonic record, found {len(matches)}"
        )

    root = Path(library_root).expanduser().resolve()
    relative_path = matches[0].get("record_path", matches[0].get("path"))
    path = (root / str(relative_path)).resolve()
    if root not in path.parents:
        raise LCB26ReferenceError("harmonic record escapes the LCB26 library")
    payload = _read_revisioned_json(path)
    if not isinstance(payload, dict):
        raise LCB26ReferenceError(f"invalid LCB26 harmonic record: {path}")
    _validate_lcb26_harmonic_record(payload, parent, path, matches[0])
    return deepcopy(payload)


def _validate_lcb26_harmonic_record(
    payload: dict[str, Any],
    parent: dict[str, Any],
    path: Path,
    link: dict[str, Any],
) -> None:
    if payload.get("schema") != LCB26_HARMONIC_RECORD_SCHEMA:
        raise LCB26ReferenceError(f"unsupported LCB26 harmonic schema: {path}")
    if payload.get("parent_identifier") != parent.get("identifier"):
        raise LCB26ReferenceError(f"harmonic record has the wrong parent: {path}")
    atoms = list(parent.get("atoms", ()))
    if not atoms or list(payload.get("atoms", ())) != atoms:
        raise LCB26ReferenceError(f"harmonic record changes LCB26 atom order: {path}")
    atom_count = len(atoms)
    dimension = 3 * atom_count

    atom_map = payload.get("atom_map", {}).get("lcb26_to_mp2_one_based")
    if atom_map != list(range(1, atom_count + 1)):
        raise LCB26ReferenceError(f"harmonic record lacks a certified identity atom map: {path}")
    coordinates = payload.get("mp2_optimized_coordinates_angstrom_in_lcb26_frame")
    if (
        not isinstance(coordinates, list)
        or len(coordinates) != atom_count
        or any(not isinstance(row, list) or len(row) != 3 for row in coordinates)
    ):
        raise LCB26ReferenceError(f"invalid aligned coordinates in harmonic record: {path}")

    rotation = payload.get("frame", {}).get("row_vector_rotation_source_to_stored")
    if not isinstance(rotation, list) or len(rotation) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in rotation
    ):
        raise LCB26ReferenceError(f"invalid frame rotation in harmonic record: {path}")
    det = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isfinite(float(det)) or abs(float(det) - 1.0) > 1.0e-8:
        raise LCB26ReferenceError(f"harmonic record uses an improper frame rotation: {path}")
    orthogonality_error = max(
        abs(
            sum(float(rotation[k][i]) * float(rotation[k][j]) for k in range(3))
            - (1.0 if i == j else 0.0)
        )
        for i in range(3)
        for j in range(3)
    )
    if not math.isfinite(orthogonality_error) or orthogonality_error > 1.0e-8:
        raise LCB26ReferenceError(f"nonorthogonal frame rotation in harmonic record: {path}")

    hessian = payload.get("cartesian_hessian", {})
    values = hessian.get("values") if isinstance(hessian, dict) else None
    if hessian.get("units") != "hartree/bohr^2" or hessian.get("dimension") != dimension:
        raise LCB26ReferenceError(f"invalid Hessian metadata in harmonic record: {path}")
    if not isinstance(values, list) or len(values) != dimension or any(
        not isinstance(row, list) or len(row) != dimension for row in values
    ):
        raise LCB26ReferenceError(f"invalid Hessian dimensions in harmonic record: {path}")
    if any(not math.isfinite(float(value)) for row in values for value in row):
        raise LCB26ReferenceError(f"nonfinite Cartesian Hessian in harmonic record: {path}")
    max_asymmetry = max(
        abs(float(values[i][j]) - float(values[j][i]))
        for i in range(dimension)
        for j in range(i)
    )
    if not math.isfinite(max_asymmetry) or max_asymmetry > 1.0e-9:
        raise LCB26ReferenceError(f"nonsymmetric Cartesian Hessian in harmonic record: {path}")

    electronic = payload.get("electronic_structure", {})
    if not isinstance(electronic, dict):
        raise LCB26ReferenceError(f"missing electronic-structure metadata: {path}")
    if "harmonic" not in str(payload.get("target_quantity", "")).casefold():
        raise LCB26ReferenceError(f"harmonic record does not declare its target quantity: {path}")
    if linked_levels := {
        key: link.get(key)
        for key in ("method", "basis")
        if link.get(key) is not None
    }:
        for key, expected in linked_levels.items():
            if str(electronic.get(key, "")).casefold() != str(expected).casefold():
                raise LCB26ReferenceError(
                    f"harmonic record {key} disagrees with its parent link: {path}"
                )


def query_lcb26_l1_geometry(
    library_root: Path,
    *,
    identifier: str | None = None,
    elements: Iterable[str] | None = None,
    element_counts: dict[str, int] | None = None,
    atom_count: int | None = None,
    dispersion: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Query the geometry-only L1 archive.

    L1 records deliberately live outside :func:`query_lcb26`: they contain
    optimized geometries for paired L1/L2 calibration, but no CM5/Mayer
    populations.  The returned rows are lightweight and can be passed to
    :func:`load_lcb26_l1_geometry`.
    """
    if limit is not None and int(limit) <= 0:
        return ()
    root = Path(library_root).expanduser().resolve()
    manifest_path = root / "l1_geometries" / "manifest.json"
    payload = _read_revisioned_json(manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise LCB26ReferenceError(f"invalid LCB26 L1 geometry archive: {manifest_path}")
    requested = _normal(identifier) if identifier is not None else None
    required = {str(element).capitalize() for element in elements or ()}
    required_counts = {str(element).capitalize(): int(count) for element, count in (element_counts or {}).items()}
    selected: list[dict[str, Any]] = []
    excluded_identifiers = {
        _normal(value) for value in payload.get("excluded_identifiers", ())
    }
    for source in payload["records"]:
        row = dict(source)
        if _normal(row.get("identifier", "")) in excluded_identifiers:
            continue
        row["dataset"] = LCB26_L1_GEOMETRY_DATASET
        row["geometry_path"] = row["file"]
        row["record_path"] = row["file"]
        row["reference_level"] = payload.get("reference_level", "L2")
        if requested is not None and requested not in _normal(row.get("identifier", "")):
            continue
        if dispersion is not None and str(row.get("dispersion", "")).casefold() != str(dispersion).casefold():
            continue
        if atom_count is not None and int(row.get("atom_count", -1)) != int(atom_count):
            continue
        symbols = _xyz_symbols(root / "l1_geometries" / str(row["file"]))
        counts = {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}
        if required and not required.issubset(counts):
            continue
        if any(counts.get(element, 0) != count for element, count in required_counts.items()):
            continue
        row["elements"] = tuple(counts)
        row["element_counts"] = counts
        row["normalized_aliases"] = (_normal(row.get("identifier", "")),)
        selected.append(row)
        if limit is not None and len(selected) >= int(limit):
            break
    return tuple(deepcopy(row) for row in selected)


def load_lcb26_l1_geometry(library_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Load an L1 XYZ geometry row without inventing electronic properties."""
    root = Path(library_root).expanduser().resolve()
    relative = str(row.get("geometry_path", row.get("file", "")))
    path = (root / "l1_geometries" / relative).resolve()
    archive_root = (root / "l1_geometries").resolve()
    if archive_root not in path.parents:
        raise LCB26ReferenceError(f"L1 geometry escapes archive: {relative}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LCB26ReferenceError(f"missing LCB26 L1 geometry: {path}") from exc
    result = deepcopy(row)
    result["geometry_xyz"] = text
    result["cm5_mayer_imported"] = False
    return result


def _xyz_symbols(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LCB26ReferenceError(f"missing LCB26 L1 geometry: {path}") from exc
    if not lines:
        raise LCB26ReferenceError(f"empty LCB26 L1 geometry: {path}")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise LCB26ReferenceError(f"invalid XYZ atom count: {path}") from exc
    symbols = [line.split()[0].capitalize() for line in lines[2:] if line.split()]
    if len(symbols) != count:
        raise LCB26ReferenceError(f"invalid XYZ atom rows in {path}")
    return symbols


__all__ = [
    "LCB26_HARMONIC_RECORD_SCHEMA",
    "LCB26ReferenceError",
    "LCB26_L1_GEOMETRY_DATASET",
    "load_lcb26_l1_geometry",
    "load_lcb26_harmonic_record",
    "load_lcb26_reference",
    "query_lcb26",
    "query_lcb26_l1_geometry",
]
