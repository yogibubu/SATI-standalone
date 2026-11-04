"""Portable registry and exact resolver for compiled ZAFF force fields."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matrix_core import atomic_json_write
from matrix_zaff.compatibility import normalize_legacy_zaff_payload
from matrix_zaff.library import (
    ZAFF_LIBRARY_ENV,
    ZAFF_LIBRARY_SCHEMA,
    ZAFF_MONOMER_REFERENCE_SCHEMA,
    ZAFF_RESOLUTION_SCHEMA,
    ZaffLibraryResolution,
    default_zaff_library_path,
    explicit_zaff_resolution,
    list_zaff_monomers,
    resolve_zaff_force_field,
    resolve_zaff_monomer,
    validate_zaff_monomer_reference,
    zaff_molecule_fingerprint,
)

from .model import ZAFF_FORCE_FIELD_SCHEMA, load_force_field, save_force_field

ZAFF_CACHE_ENV = "MATRIX_ZAFF_CACHE"


def create_zaff_library(path: Path | str) -> Path:
    """Create an empty, versioned ZAFF library manifest."""

    manifest = _manifest_path(Path(path))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        raise FileExistsError(f"ZAFF library already exists: {manifest}")
    _write_manifest(
        manifest,
        {"schema": ZAFF_LIBRARY_SCHEMA, "entries": [], "monomers": []},
    )
    return manifest


def add_force_field_to_library(
    library: Path | str,
    force_field: Path | str,
    *,
    identifier: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    priority: int = 0,
) -> dict[str, Any]:
    """Validate and copy one compiled field into a portable library."""

    manifest = _manifest_path(Path(library))
    payload = _load_manifest(manifest)
    source = Path(force_field).resolve()
    field = load_force_field(source)
    field_sha = _sha256_file(source)
    metadata = dict(field.metadata)
    resolved_charge = int(metadata.get("charge", 0) if charge is None else charge)
    resolved_multiplicity = int(
        metadata.get("multiplicity", 1) if multiplicity is None else multiplicity
    )
    if resolved_multiplicity <= 0:
        raise ValueError("ZAFF library multiplicity must be positive")
    entry_id = str(identifier or metadata.get("identifier") or field_sha[:16]).strip()
    if not entry_id:
        raise ValueError("ZAFF library entry identifier must be nonempty")
    if any(str(item.get("id")) == entry_id for item in payload["entries"]):
        raise ValueError(f"duplicate ZAFF library entry identifier: {entry_id}")
    fields_dir = manifest.parent / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    target = fields_dir / f"{field_sha}.zaff.json"
    if not target.exists():
        shutil.copy2(source, target)
    fingerprint = zaff_molecule_fingerprint(
        field.atoms,
        field.reference_coordinates_angstrom,
        charge=resolved_charge,
        multiplicity=resolved_multiplicity,
    )
    entry = {
        "id": entry_id,
        "path": str(target.relative_to(manifest.parent)),
        "sha256": field_sha,
        "force_field_schema": ZAFF_FORCE_FIELD_SCHEMA,
        "atoms": list(field.atoms),
        "charge": resolved_charge,
        "multiplicity": resolved_multiplicity,
        "molecule_fingerprint": fingerprint,
        "priority": int(priority),
    }
    payload["entries"].append(entry)
    payload["entries"].sort(key=lambda item: str(item["id"]))
    _write_manifest(manifest, payload)
    return dict(entry)


def list_zaff_library(path: Path | str | None = None) -> tuple[dict[str, Any], ...]:
    manifest = _manifest_path(Path(path)) if path is not None else default_zaff_library_path()
    return tuple(dict(item) for item in _load_manifest(manifest)["entries"])


def add_monomer_reference_to_library(
    library: Path | str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Add one immutable geometry/CM5/Mayer/synthon reference record.

    Monomer references and compiled force fields share one manifest but remain
    separate artifact classes.  This lets every MATRIX tool consume the same
    intrinsic electronic and covalent state without implying that a fitted
    force field already exists for the molecule.
    """

    manifest = _manifest_path(Path(library))
    payload = _load_manifest(manifest)
    normalized = validate_zaff_monomer_reference(record)
    identifiers = {str(item["id"]).casefold() for item in payload["monomers"]}
    if normalized["id"].casefold() in identifiers:
        raise ValueError(f"duplicate ZAFF monomer identifier: {normalized['id']}")
    payload["monomers"].append(normalized)
    payload["monomers"].sort(key=lambda item: str(item["id"]))
    _write_manifest(manifest, payload)
    return dict(normalized)


def promote_monomer_reference_in_library(
    library: Path | str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace one transitional monomer record by its definitive L0 record."""

    manifest = _manifest_path(Path(library))
    payload = _load_manifest(manifest)
    normalized = validate_zaff_monomer_reference(record)
    if normalized["reference_status"] != "DEFINITIVE":
        raise ValueError("only a DEFINITIVE monomer record can be promoted")
    matching = [
        index
        for index, item in enumerate(payload["monomers"])
        if str(item["id"]).casefold() == normalized["id"].casefold()
    ]
    if len(matching) != 1:
        raise KeyError(f"promotion requires one existing monomer {normalized['id']!r}")
    index = matching[0]
    previous = validate_zaff_monomer_reference(payload["monomers"][index])
    if previous["reference_status"] == "DEFINITIVE":
        raise ValueError("a definitive monomer record cannot be overwritten")
    if (
        previous["atoms"] != normalized["atoms"]
        or int(previous.get("charge", 0)) != int(normalized.get("charge", 0))
        or int(previous.get("multiplicity", 1)) != int(normalized.get("multiplicity", 1))
    ):
        raise ValueError("promoted monomer must preserve atom order and electronic state")
    normalized.setdefault("provenance", {})["supersedes_reference_status"] = previous[
        "reference_status"
    ]
    payload["monomers"][index] = normalized
    payload["monomers"].sort(key=lambda item: str(item["id"]))
    _write_manifest(manifest, payload)
    return dict(normalized)


def resolve_or_compile_zaff_force_field(
    atoms: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]] | np.ndarray,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    library: Path | str | None = None,
    cm5_charges: Sequence[float] | None = None,
    mayer_bond_orders: Mapping[tuple[int, int], float] | None = None,
    gaff_path: Path | str | None = None,
    charge_source: str = "APOC CM5",
    bond_order_source: str = "APOC Mayer",
) -> ZaffLibraryResolution:
    """Resolve an exact field, then compile a transferable seed when possible."""

    exact = resolve_zaff_force_field(
        atoms,
        coordinates_angstrom,
        charge=charge,
        multiplicity=multiplicity,
        library=library,
    )
    if exact.force_field_path is not None:
        return exact
    from .seed_compiler import ZaffSeedCoverageError, build_zaff_seed_force_field

    try:
        field = build_zaff_seed_force_field(
            atoms,
            np.asarray(coordinates_angstrom, dtype=float),
            cm5_charges,
            mayer_bond_orders=mayer_bond_orders,
            gaff_path=gaff_path,
            charge=charge,
            multiplicity=multiplicity,
            charge_source=charge_source,
            bond_order_source=bond_order_source,
        )
    except (OSError, ValueError, ZaffSeedCoverageError) as exc:
        return ZaffLibraryResolution(
            requested_backend=exact.requested_backend,
            realized_backend=exact.realized_backend,
            selection=exact.selection,
            library_path=exact.library_path,
            molecule_fingerprint=exact.molecule_fingerprint,
            message=f"{exact.message}; transferable compilation unavailable: {exc}",
        )
    model = dict(field.metadata["diagonal_seed_model"])
    provenance = dict(model["parameter_provenance"])
    cache_document = {
        "fingerprint": exact.molecule_fingerprint,
        "electronic_observable_level": model["electronic_observable_level"],
        "electronic_observables": model["electronic_observables"],
        "charge_source": model["charge_source"],
        "bond_order_source": model["bond_order_source"],
        "gaff_sha256": provenance["gaff_sha256"],
        "seed_schema": model["schema"],
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_root = Path(
        os.environ.get(
            ZAFF_CACHE_ENV,
            str(Path.home() / ".cache" / "matrix" / "zaff"),
        )
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{cache_key}.zaff.json"
    if not target.is_file():
        save_force_field(target, field)
    else:
        cached = load_force_field(target)
        if cached.atoms != tuple(str(atom) for atom in atoms):
            raise ValueError("cached transferable ZAFF field has an incompatible atom order")
    digest = _sha256_file(target)
    return ZaffLibraryResolution(
        requested_backend="zaff",
        realized_backend="zaff",
        selection="ZAFF_TRANSFERABLE_COMPILED",
        library_path=exact.library_path,
        force_field_path=target.resolve(),
        force_field_sha256=digest,
        entry_id=f"transferable-{cache_key[:16]}",
        molecule_fingerprint=exact.molecule_fingerprint,
        message=(
            "compiled complete MM3/GAFF transferable field with a coherent "
            f"{model['electronic_observable_level']} observable level and cached it "
            "for LINK, SENTINEL and ORACLE"
        ),
    )


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
    if not isinstance(entries, list) or any(not isinstance(item, Mapping) for item in entries):
        raise ValueError("ZAFF library entries must be a JSON array of objects")
    ids = tuple(str(item.get("id", "")) for item in entries)
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError("ZAFF library entry identifiers must be nonempty and unique")
    monomers = payload.get("monomers", [])
    if not isinstance(monomers, list) or any(not isinstance(item, Mapping) for item in monomers):
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


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_json_write(path, payload, allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ZAFF_CACHE_ENV",
    "ZAFF_LIBRARY_ENV",
    "ZAFF_LIBRARY_SCHEMA",
    "ZAFF_MONOMER_REFERENCE_SCHEMA",
    "ZAFF_RESOLUTION_SCHEMA",
    "ZaffLibraryResolution",
    "add_force_field_to_library",
    "add_monomer_reference_to_library",
    "create_zaff_library",
    "default_zaff_library_path",
    "explicit_zaff_resolution",
    "list_zaff_library",
    "list_zaff_monomers",
    "promote_monomer_reference_in_library",
    "resolve_or_compile_zaff_force_field",
    "resolve_zaff_force_field",
    "resolve_zaff_monomer",
    "validate_zaff_monomer_reference",
    "zaff_molecule_fingerprint",
]
