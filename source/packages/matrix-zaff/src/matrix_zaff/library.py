"""Read-only resolver for compiled ZAFF artifacts.

Library creation and mutation belong to ARCHITECT.  The runtime only validates
and selects existing immutable artifacts; it never starts a fitting or
compilation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact import load_zaff_artifact
from .compatibility import normalize_legacy_zaff_payload


ZAFF_LIBRARY_SCHEMA = "matrix.zaff.force_field_library.v1"
ZAFF_MONOMER_REFERENCE_SCHEMA = "matrix.zaff.monomer_reference.v1"
ZAFF_RESOLUTION_SCHEMA = "matrix.zaff.force_field_resolution.v2"
ZAFF_LIBRARY_ENV = "MATRIX_ZAFF_LIBRARY"


def default_zaff_library_path() -> Path:
    configured = os.environ.get(ZAFF_LIBRARY_ENV, "").strip()
    if configured:
        return _manifest_path(Path(configured).expanduser())
    return Path(__file__).with_name("data") / "zaff_library.json"


@dataclass(frozen=True)
class ZaffLibraryResolution:
    requested_backend: str
    realized_backend: str
    selection: str
    library_path: Path
    force_field_path: Path | None = None
    force_field_sha256: str | None = None
    entry_id: str | None = None
    molecule_fingerprint: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ZAFF_RESOLUTION_SCHEMA,
            "requested_backend": self.requested_backend,
            "realized_backend": self.realized_backend,
            "selection": self.selection,
            "library_path": str(self.library_path),
            "force_field_path": (
                None if self.force_field_path is None else str(self.force_field_path)
            ),
            "force_field_sha256": self.force_field_sha256,
            "entry_id": self.entry_id,
            "molecule_fingerprint": self.molecule_fingerprint,
            "fallback_model": (
                "GFN-FF" if self.selection == "GFN_FF_FALLBACK" else None
            ),
            "message": self.message,
        }


def resolve_zaff_force_field(
    atoms: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]] | np.ndarray,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    library: Path | str | None = None,
) -> ZaffLibraryResolution:
    """Resolve an existing field or return the explicit GFN-FF fallback."""

    manifest = (
        _manifest_path(Path(library))
        if library is not None
        else default_zaff_library_path()
    )
    fingerprint = zaff_molecule_fingerprint(
        atoms,
        coordinates_angstrom,
        charge=charge,
        multiplicity=multiplicity,
    )
    payload = _load_manifest(manifest)
    candidates: list[tuple[int, str, Path, str]] = []
    rejected = 0
    for raw in payload["entries"]:
        if (
            tuple(str(atom) for atom in raw.get("atoms", ()))
            != tuple(str(atom) for atom in atoms)
            or int(raw.get("charge", 0)) != int(charge)
            or int(raw.get("multiplicity", 1)) != int(multiplicity)
            or str(raw.get("molecule_fingerprint", "")) != fingerprint
        ):
            continue
        try:
            field_path = (manifest.parent / str(raw["path"])).resolve()
            expected_sha = str(raw.get("sha256", ""))
            if not field_path.is_file() or _sha256_file(field_path) != expected_sha:
                rejected += 1
                continue
            artifact = load_zaff_artifact(field_path)
            if artifact.atoms != tuple(str(atom) for atom in atoms):
                rejected += 1
                continue
        except (OSError, TypeError, ValueError):
            rejected += 1
            continue
        candidates.append(
            (int(raw.get("priority", 0)), str(raw["id"]), field_path, expected_sha)
        )
    if candidates:
        priority, entry_id, field_path, field_sha = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return ZaffLibraryResolution(
            requested_backend="zaff",
            realized_backend="zaff",
            selection="ZAFF_LIBRARY_MATCH",
            library_path=manifest.resolve(),
            force_field_path=field_path,
            force_field_sha256=field_sha,
            entry_id=entry_id,
            molecule_fingerprint=fingerprint,
            message=f"selected priority {priority} exact library match",
        )
    detail = (
        f"; ignored {rejected} invalid matching entr{'y' if rejected == 1 else 'ies'}"
        if rejected
        else ""
    )
    return ZaffLibraryResolution(
        requested_backend="zaff",
        realized_backend="gfn-ff",
        selection="GFN_FF_FALLBACK",
        library_path=manifest.resolve(),
        molecule_fingerprint=fingerprint,
        message=f"no existing ZAFF library match{detail}",
    )


def explicit_zaff_resolution(force_field: Path | str) -> ZaffLibraryResolution:
    path = Path(force_field).expanduser().resolve()
    artifact = load_zaff_artifact(path)
    metadata = dict(artifact.metadata)
    fingerprint = zaff_molecule_fingerprint(
        artifact.atoms,
        artifact.reference_coordinates_angstrom,
        charge=int(metadata.get("charge", 0)),
        multiplicity=int(metadata.get("multiplicity", 1)),
    )
    return ZaffLibraryResolution(
        requested_backend="zaff",
        realized_backend="zaff",
        selection="EXPLICIT_ZAFF_OVERRIDE",
        library_path=default_zaff_library_path().resolve(),
        force_field_path=path,
        force_field_sha256=_sha256_file(path),
        molecule_fingerprint=fingerprint,
        message="explicit compiled artifact overrides library resolution",
    )


def list_zaff_monomers(
    path: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return immutable monomer geometry/CM5/Mayer/synthon references."""

    manifest = _manifest_path(Path(path)) if path is not None else default_zaff_library_path()
    return tuple(dict(item) for item in _load_manifest(manifest)["monomers"])


def resolve_zaff_monomer(
    identifier: str,
    *,
    library: Path | str | None = None,
    require_definitive: bool = False,
) -> dict[str, Any]:
    requested = str(identifier).strip().casefold()
    if not requested:
        raise ValueError("ZAFF monomer identifier cannot be empty")
    matches = [
        item
        for item in list_zaff_monomers(library)
        if requested
        in {
            str(item["id"]).casefold(),
            *(str(alias).casefold() for alias in item.get("aliases", ())),
        }
    ]
    if len(matches) != 1:
        raise KeyError(
            f"ZAFF monomer reference {identifier!r} has {len(matches)} matches"
        )
    match = dict(matches[0])
    if require_definitive and match["reference_status"] != "DEFINITIVE":
        raise ValueError(
            f"ZAFF monomer reference {identifier!r} is "
            f"{match['reference_status']}, not DEFINITIVE"
        )
    return match


def zaff_molecule_fingerprint(
    atoms: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]] | np.ndarray,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> str:
    """Return the established atom-order and covalent-graph fingerprint."""

    from matrix_chem import build_topology_objects
    from matrix_chem.topology.elements import atomic_number

    symbols = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if not symbols or coordinates.shape != (len(symbols), 3):
        raise ValueError("ZAFF library query geometry must have shape (natoms, 3)")
    numbers = tuple(int(atomic_number(atom) or 0) for atom in symbols)
    if any(number <= 0 for number in numbers):
        raise ValueError("ZAFF library query contains an unknown element")
    if len(symbols) == 1:
        bonds: list[list[int]] = []
    else:
        _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(
            coordinates,
            numbers,
        )
        bonds = [list(map(int, pair)) for pair in sorted(map(tuple, graph.bonds))]
    document = {
        "atom_order": list(numbers),
        "bonds": bonds,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
    }
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_path(path: Path) -> Path:
    return path / "zaff_library.json" if path.suffix.lower() != ".json" else path


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ZAFF library manifest does not exist: {path}")
    payload = normalize_legacy_zaff_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(payload, Mapping) or payload.get("schema") != ZAFF_LIBRARY_SCHEMA:
        raise ValueError(f"unsupported ZAFF library schema in {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or any(
        not isinstance(item, Mapping) for item in entries
    ):
        raise ValueError("ZAFF library entries must be a JSON array of objects")
    identifiers = tuple(str(item.get("id", "")) for item in entries)
    if any(not item for item in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("ZAFF library entry identifiers must be nonempty and unique")
    monomers = payload.get("monomers", [])
    if not isinstance(monomers, list) or any(
        not isinstance(item, Mapping) for item in monomers
    ):
        raise ValueError("ZAFF library monomers must be a JSON array of objects")
    normalized_monomers = [validate_zaff_monomer_reference(item) for item in monomers]
    monomer_ids = tuple(item["id"].casefold() for item in normalized_monomers)
    if len(monomer_ids) != len(set(monomer_ids)):
        raise ValueError("ZAFF monomer identifiers must be unique")
    return {
        "schema": ZAFF_LIBRARY_SCHEMA,
        "entries": [dict(item) for item in entries],
        "monomers": normalized_monomers,
    }


def validate_zaff_monomer_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical, JSON-safe representation of one monomer record."""

    normalized = normalize_legacy_zaff_payload(
        json.loads(json.dumps(dict(record), allow_nan=False))
    )
    if normalized.get("schema") != ZAFF_MONOMER_REFERENCE_SCHEMA:
        raise ValueError("unsupported ZAFF monomer-reference schema")
    identifier = str(normalized.get("id", "")).strip()
    atoms = tuple(str(atom).strip() for atom in normalized.get("atoms", ()))
    coordinates = np.asarray(normalized.get("coordinates_angstrom", ()), dtype=float)
    cm5 = np.asarray(normalized.get("intrinsic_cm5_charges_e", ()), dtype=float)
    synthons = normalized.get("atomic_synthons", ())
    mayer = normalized.get("mayer_bond_orders", ())
    if not identifier or not atoms or coordinates.shape != (len(atoms), 3):
        raise ValueError("ZAFF monomer needs an id, atoms and matching coordinates")
    if cm5.shape != (len(atoms),) or np.any(~np.isfinite(cm5)):
        raise ValueError("ZAFF monomer needs one finite intrinsic CM5 charge per atom")
    if abs(float(np.sum(cm5)) - int(normalized.get("charge", 0))) > 5.0e-5:
        raise ValueError("ZAFF monomer CM5 charges do not sum to molecular charge")
    if not isinstance(synthons, list) or len(synthons) != len(atoms):
        raise ValueError("ZAFF monomer needs one atomic synthon per atom")
    if not isinstance(mayer, list) or any(
        not isinstance(row, list) or len(row) != 3 for row in mayer
    ):
        raise ValueError("ZAFF monomer Mayer orders must be one-based [i,j,value] rows")
    if len(mayer) != len(atoms) * (len(atoms) - 1) // 2:
        raise ValueError("ZAFF monomer Mayer matrix must contain every atom pair")
    normalized["id"] = identifier
    normalized["atoms"] = list(atoms)
    normalized["coordinates_angstrom"] = coordinates.tolist()
    normalized["intrinsic_cm5_charges_e"] = cm5.tolist()
    normalized["aliases"] = sorted(
        {str(alias).strip() for alias in normalized.get("aliases", ()) if str(alias).strip()}
    )
    status = str(normalized.get("reference_status", "")).strip().upper()
    if not status:
        level = str(normalized.get("electronic_observables", {}).get("level", ""))
        status = (
            "DEFINITIVE"
            if level.casefold().startswith("pbe0/def2-tzvp")
            else "TRANSITIONAL_NOT_FOR_PRODUCTION_GA"
        )
    if status not in {"DEFINITIVE", "TRANSITIONAL_NOT_FOR_PRODUCTION_GA"}:
        raise ValueError("unsupported ZAFF monomer reference status")
    normalized["reference_status"] = status
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ZAFF_LIBRARY_ENV",
    "ZAFF_LIBRARY_SCHEMA",
    "ZAFF_MONOMER_REFERENCE_SCHEMA",
    "ZAFF_RESOLUTION_SCHEMA",
    "ZaffLibraryResolution",
    "default_zaff_library_path",
    "explicit_zaff_resolution",
    "list_zaff_monomers",
    "resolve_zaff_monomer",
    "resolve_zaff_force_field",
    "validate_zaff_monomer_reference",
    "zaff_molecule_fingerprint",
]
