"""Build and query the unified LCB26 molecular/electronic index."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Iterable, Mapping

from matrix_core import atomic_output_path, atomic_write_text
from matrix_switch import canonical_smiles, graph_from_topology, parse_smiles
from matrix_switch import render_molecule_png


LCB26_QUERY_INDEX_SCHEMA = "matrix.lcb26.query_index.v1"


class LCB26IndexError(ValueError):
    """Raised when an LCB26 index cannot be built or queried safely."""


def normalize_lcb26_selector(value: str) -> str:
    """Normalize identifiers and names without losing chemical punctuation."""

    ascii_value = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def canonical_constitutional_smiles(smiles: str) -> str:
    """Return SWITCH's canonical SMILES with stereochemical annotations removed."""

    graph = parse_smiles(str(smiles).strip())
    graph = replace(
        graph,
        atoms=tuple(
            replace(atom, chirality=None, atom_class=None, source_span=(0, 0))
            for atom in graph.atoms
        ),
        bonds=tuple(replace(bond, direction=None) for bond in graph.bonds),
    )
    return canonical_smiles(graph)


def molecular_formula(atoms: Iterable[str]) -> str:
    counts = Counter(str(atom) for atom in atoms)
    order: list[str] = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
    order.extend(sorted(element for element in counts if element not in order))
    return "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in order
    )


def smiles_from_electronic_record(record: Mapping[str, Any]) -> str:
    """Infer a canonical constitutional SMILES from Mayer connectivity."""

    atoms = tuple(str(atom) for atom in record.get("atoms", ()))
    if not atoms:
        raise LCB26IndexError("electronic record has no atoms")
    heavy = [index for index, symbol in enumerate(atoms) if symbol != "H"]
    if not heavy:
        raise LCB26IndexError("electronic record has no heavy atoms")
    remap = {old: new for new, old in enumerate(heavy)}
    descriptors = record.get("synthon_descriptors", ())
    aromatic_atoms = {
        remap[index]
        for index in heavy
        if index < len(descriptors)
        and len(descriptors[index].get("canonical_signature", ())) > 1
        and bool(descriptors[index]["canonical_signature"][1])
    }
    bonds: list[tuple[int, int]] = []
    orders: dict[tuple[int, int], float] = {}
    for component in record.get("mayer_bond_components", ()):
        raw_pair = component.get("atoms", ())
        if len(raw_pair) != 2:
            continue
        left, right = int(raw_pair[0]) - 1, int(raw_pair[1]) - 1
        if left not in remap or right not in remap:
            continue
        pair = tuple(sorted((remap[left], remap[right])))
        total = float(component.get("total", 1.0))
        if pair[0] in aromatic_atoms and pair[1] in aromatic_atoms:
            order = 1.5
        elif total >= 2.35:
            order = 3.0
        elif total >= 1.55:
            order = 2.0
        else:
            order = 1.0
        bonds.append(pair)
        orders[pair] = order
    graph = graph_from_topology(
        [atoms[index] for index in heavy],
        bonds,
        bond_orders=orders,
        aromatic_atoms=tuple(aromatic_atoms),
    )
    graph = replace(
        graph,
        atoms=tuple(replace(atom, bracketed=False) for atom in graph.atoms),
    )
    return canonical_smiles(graph)


def _manifest_metadata(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "manifest.json"
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LCB26IndexError(f"invalid LCB26 manifest: {path}") from exc
    return {
        str(entry.get("id")): entry
        for entry in manifest.get("entries", ())
        if entry.get("id")
    }


def _record_summary(
    root: Path,
    path: Path,
    record: Mapping[str, Any],
    manifest_entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    identifier = str(record.get("identifier") or path.name.removesuffix(".cm5_mayer.json"))
    dataset = str(record.get("source_dataset") or path.parent.name)
    stem = path.name.removesuffix(".cm5_mayer.json")
    entry = manifest_entries.get(stem, {})
    comment_name = str(record.get("geometry_comment", "")).split("|", 1)[0].strip()
    name = str(record.get("name") or entry.get("name") or comment_name or stem.replace("_", " "))
    aliases = {
        identifier,
        stem,
        stem.replace("_", " "),
        stem.replace("-", " "),
        name,
        *(str(alias) for alias in entry.get("aliases", ())),
    }
    atoms = tuple(str(atom) for atom in record.get("atoms", ()))
    element_counts = Counter(atoms)
    heavy_count = sum(count for element, count in element_counts.items() if element != "H")
    heavy_bonds = 0
    for component in record.get("mayer_bond_components", ()):
        pair = component.get("atoms", ())
        if len(pair) == 2 and all(atoms[int(index) - 1] != "H" for index in pair):
            heavy_bonds += 1
    # Ring perception belongs to ORACLE.  Enriched records and the audited
    # LCB26 manifest are the authoritative sources; the Mayer graph is not a
    # substitute for ORACLE's minimum-cycle-basis contract.
    oracle_audit = record.get("oracle_audit", {})
    ring_count_value = oracle_audit.get("ring_count", record.get("ring_count"))
    if ring_count_value is None:
        ring_count_value = len(oracle_audit.get("rings", ()))
    if ring_count_value is None and entry.get("rings") is not None:
        ring_count_value = len(entry.get("rings", ()))
    ring_count = None if ring_count_value is None else int(ring_count_value)
    smiles = str(record.get("canonical_smiles") or smiles_from_electronic_record(record))
    electron_count = record.get("electron_count")
    multiplicity = int(record.get("multiplicity", 1))
    open_shell = bool(multiplicity != 1 or (electron_count is not None and int(electron_count) % 2))
    raw_geometry = str(record.get("source_geometry", entry.get("geometry", "")))
    geometry_path = raw_geometry
    marker = "geometries/"
    if marker in raw_geometry:
        geometry_path = marker + raw_geometry.split(marker, 1)[1]
    elif not raw_geometry or raw_geometry.startswith("/"):
        geometry_path = f"geometries/{dataset}/{stem}.xyz"
    raw_level = record.get("electronic_level") or dataset
    if isinstance(raw_level, Mapping):
        raw_level = raw_level.get("accuracy_level") or raw_level.get("level") or dataset
    return {
        "identifier": identifier,
        "name": name,
        "aliases": sorted(aliases),
        "normalized_aliases": sorted({normalize_lcb26_selector(alias) for alias in aliases}),
        "canonical_smiles": smiles,
        "formula": str(record.get("formula") or entry.get("formula") or molecular_formula(atoms)),
        "elements": sorted(element_counts),
        "element_counts": dict(sorted(element_counts.items())),
        "dataset": dataset,
        "electronic_level": str(raw_level),
        "charge": int(record.get("molecular_charge", record.get("charge", entry.get("charge", 0)))),
        "multiplicity": multiplicity,
        "open_shell": open_shell,
        "atom_count": len(atoms),
        "heavy_atom_count": heavy_count,
        "ring_count": ring_count,
        "record_path": relative,
        "geometry_path": geometry_path,
    }


def build_lcb26_index(lcb26_root: Path) -> dict[str, Any]:
    """Build portable JSON and SQLite indexes for every LCB26 electronic record."""

    root = Path(lcb26_root).expanduser().resolve()
    enriched = root / "enriched"
    paths = sorted(enriched.rglob("*.cm5_mayer.json"))
    if not paths:
        raise LCB26IndexError(f"no LCB26 electronic records found under {enriched}")
    manifest_entries = _manifest_metadata(root)
    summaries = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LCB26IndexError(f"invalid electronic record: {path}") from exc
        summaries.append(_record_summary(root, path, record, manifest_entries))
    identifiers = [summary["identifier"] for summary in summaries]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(key for key, count in Counter(identifiers).items() if count > 1)
        raise LCB26IndexError(f"duplicate LCB26 identifiers: {duplicates}")
    payload = {
        "schema": LCB26_QUERY_INDEX_SCHEMA,
        "library": "LCB26",
        "record_count": len(summaries),
        "selectors": [
            "identifier", "name", "alias", "smiles", "formula", "elements",
            "dataset", "electronic_level", "charge", "multiplicity", "open_shell",
            "atom_count", "heavy_atom_count", "ring_count",
        ],
        "records": summaries,
    }
    _synchronize_unified_manifest(root, summaries)
    enriched.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        enriched / "index.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_sqlite_atomically(enriched / "lcb26.sqlite", payload)
    return payload


def _synchronize_unified_manifest(
    root: Path,
    summaries: Iterable[Mapping[str, Any]],
) -> None:
    """Keep generated record counts in the single LCB26 manifest coherent."""

    path = root / "manifest.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LCB26IndexError(f"invalid LCB26 manifest: {path}") from exc
    if payload.get("schema") != "matrix.lcb26.unified_library_manifest.v2":
        return
    rows = tuple(summaries)
    datasets = dict(sorted(Counter(str(row["dataset"]) for row in rows).items()))
    if payload.get("datasets") == datasets and payload.get("record_count") == len(rows):
        return
    payload["datasets"] = datasets
    payload["record_count"] = len(rows)
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)


def _write_sqlite_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    with atomic_output_path(path) as temporary:
        connection = sqlite3.connect(temporary)
        connection.executescript(
            """
            DROP TABLE IF EXISTS aliases;
            DROP TABLE IF EXISTS records;
            DROP TABLE IF EXISTS metadata;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE records (
                identifier TEXT PRIMARY KEY, name TEXT NOT NULL,
                canonical_smiles TEXT NOT NULL, formula TEXT NOT NULL,
                elements TEXT NOT NULL, dataset TEXT NOT NULL,
                electronic_level TEXT NOT NULL, charge INTEGER NOT NULL,
                multiplicity INTEGER NOT NULL, open_shell INTEGER NOT NULL,
                atom_count INTEGER NOT NULL, heavy_atom_count INTEGER NOT NULL,
                ring_count INTEGER, record_path TEXT NOT NULL,
                geometry_path TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE aliases (
                normalized_alias TEXT NOT NULL, identifier TEXT NOT NULL,
                alias TEXT NOT NULL,
                FOREIGN KEY(identifier) REFERENCES records(identifier)
            );
            CREATE INDEX aliases_lookup ON aliases(normalized_alias);
            CREATE INDEX records_smiles ON records(canonical_smiles);
            CREATE INDEX records_formula ON records(formula);
            CREATE INDEX records_dataset ON records(dataset);
            """
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (("schema", str(payload["schema"])), ("record_count", str(payload["record_count"]))),
        )
        for summary in payload["records"]:
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    summary["identifier"], summary["name"], summary["canonical_smiles"],
                    summary["formula"], json.dumps(summary["elements"]), summary["dataset"],
                    summary["electronic_level"], summary["charge"], summary["multiplicity"],
                    int(summary["open_shell"]), summary["atom_count"], summary["heavy_atom_count"],
                    summary["ring_count"], summary["record_path"], summary["geometry_path"],
                    json.dumps(summary, sort_keys=True),
                ),
            )
            connection.executemany(
                "INSERT INTO aliases VALUES (?, ?, ?)",
                (
                    (normalize_lcb26_selector(alias), summary["identifier"], alias)
                    for alias in summary["aliases"]
                ),
            )
        connection.commit()
        connection.close()


def query_lcb26(
    lcb26_root: Path,
    *,
    identifier: str | None = None,
    name: str | None = None,
    alias: str | None = None,
    smiles: str | None = None,
    formula: str | None = None,
    elements: Iterable[str] | None = None,
    element_counts: Mapping[str, int] | None = None,
    dataset: str | None = None,
    electronic_level: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    open_shell: bool | None = None,
    atom_count: int | None = None,
    heavy_atom_count: int | None = None,
    ring_count: int | None = None,
    contains: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Query LCB26 by identity, structure, composition or calculation metadata."""

    if limit is not None and int(limit) <= 0:
        return ()
    root = Path(lcb26_root).expanduser().resolve()
    index_path = root / "enriched" / "index.json"
    if not index_path.is_file():
        build_lcb26_index(root)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LCB26IndexError(f"invalid LCB26 query index: {index_path}") from exc
    exact_aliases = [value for value in (identifier, name, alias) if value is not None]
    normalized = [normalize_lcb26_selector(value) for value in exact_aliases]
    canonical_query = canonical_constitutional_smiles(smiles) if smiles is not None else None
    required_elements = {str(element) for element in elements or ()}
    required_counts = {str(element): int(count) for element, count in (element_counts or {}).items()}
    results = []
    for summary in payload.get("records", ()):
        if normalized and not all(value in summary["normalized_aliases"] for value in normalized):
            continue
        if canonical_query is not None and summary["canonical_smiles"] != canonical_query:
            continue
        if formula is not None and summary["formula"].casefold() != formula.strip().casefold():
            continue
        if required_elements and not required_elements.issubset(summary["elements"]):
            continue
        if any(summary["element_counts"].get(element, 0) != count for element, count in required_counts.items()):
            continue
        scalar_filters = {
            "dataset": dataset,
            "electronic_level": electronic_level,
            "charge": charge,
            "multiplicity": multiplicity,
            "open_shell": open_shell,
            "atom_count": atom_count,
            "heavy_atom_count": heavy_atom_count,
            "ring_count": ring_count,
        }
        if any(
            expected is not None
            and (str(summary[key]).casefold() != str(expected).casefold())
            for key, expected in scalar_filters.items()
        ):
            continue
        if contains is not None:
            needle = normalize_lcb26_selector(contains)
            if not any(needle in value for value in summary["normalized_aliases"]):
                continue
        results.append(dict(summary))
        if limit is not None and len(results) >= int(limit):
            break
    return tuple(results)


def load_lcb26_record(lcb26_root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(lcb26_root).expanduser().resolve() / str(summary["record_path"])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LCB26IndexError(f"invalid indexed LCB26 record: {path}") from exc


def render_lcb26_query_png(
    rows: Iterable[Mapping[str, Any]],
    output: Path,
    *,
    columns: int = 3,
) -> Path:
    """Render a deterministic 2D contact sheet for a query result set."""

    from PIL import Image, ImageDraw, ImageFont

    selected = list(rows)
    if columns < 1:
        raise ValueError("columns must be positive")
    cell_width, cell_height = 440, 370
    sheet = Image.new(
        "RGB",
        (cell_width * min(columns, max(1, len(selected))), cell_height * max(1, (len(selected) + columns - 1) // columns)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        graph = parse_smiles(str(row["canonical_smiles"]))
        molecule = Image.open(BytesIO(render_molecule_png(graph, explicit_hydrogens=False))).convert("RGB")
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(molecule, (x, y))
        label = f"{row.get('name', row.get('identifier', ''))} | {row.get('identifier', '')}"
        draw.text((x + 8, y + 340), label[:70], fill="black", font=font)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", optimize=True)
    return destination


__all__ = [
    "LCB26IndexError",
    "LCB26_QUERY_INDEX_SCHEMA",
    "build_lcb26_index",
    "canonical_constitutional_smiles",
    "load_lcb26_record",
    "molecular_formula",
    "normalize_lcb26_selector",
    "query_lcb26",
    "render_lcb26_query_png",
    "smiles_from_electronic_record",
]
