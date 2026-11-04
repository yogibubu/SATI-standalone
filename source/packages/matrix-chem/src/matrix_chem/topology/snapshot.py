from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from matrix_core import read_sectioned_lines, section_content

from ..geometry_io import read_enriched_xyz
from .contracts import (
    SUPPORTED_TOPOLOGY_SCHEMAS,
    schema_from_line,
    schema_line_supported,
)


TOPOLOGY_SNAPSHOT_SCHEMA = "matrix.chem.topology_snapshot.v3"
DEFAULT_SELECTED_RING_RELATIONS = 24
# SMILES-to-3D conformer coordinates can differ by several 1e-5 A between
# supported architectures.  Discrete topology is protected by an exact hash;
# this tolerance applies only to the derived continuous Pauling descriptors.
DEFAULT_NUMERIC_TOLERANCE = 1.0e-3


@dataclass(frozen=True)
class TopologySnapshotComparison:
    ok: bool
    messages: tuple[str, ...] = ()


def topology_snapshot_from_xyzin(
    path: Path,
    *,
    case_id: str = "",
    source: str = "",
    rounding_decimals: int = 8,
    selected_ring_relations: int = DEFAULT_SELECTED_RING_RELATIONS,
) -> dict[str, Any]:
    lines = read_sectioned_lines(Path(path))
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        raise ValueError("missing supported #TOPOLOGY section")
    bonds = _parse_bonds(topology)
    bond_orders = _parse_bond_orders(topology, rounding_decimals=rounding_decimals)
    bond_order_components = _parse_bond_order_components(
        topology, rounding_decimals=rounding_decimals
    )
    rings = _parse_rings(topology)
    aromatic = _parse_aromaticity(topology)
    transitional_contacts = _parse_transitional_contacts(topology)
    fragments = _connected_components(bonds, natoms=read_enriched_xyz(Path(path)).natoms)
    relations = _ring_relations(rings)
    ring_basis = _parse_ring_basis(topology)
    discrete = {
        "bonds": bonds,
        "rings": rings,
        "ring_relations": relations,
        "ring_basis": ring_basis,
        "aromatic_atoms": aromatic["atoms"],
        "aromatic_bonds": aromatic["bonds"],
        "fragments": fragments,
    }
    return {
        "id": case_id,
        "source": source,
        "topology_schema": schema_from_line(topology[0]),
        "bond_count": len(bonds),
        "ring_count": len(rings),
        "ring_relation_count": len(relations),
        "fragment_count": len(fragments),
        "bonds": bonds,
        "bond_orders": bond_orders,
        "bond_order_components": bond_order_components,
        "rings": rings,
        "ring_basis": ring_basis,
        "selected_ring_relations": relations[:selected_ring_relations],
        "aromatic_atoms": aromatic["atoms"],
        "aromatic_bonds": aromatic["bonds"],
        "fragments": fragments,
        "transitional_contacts": transitional_contacts,
        "topology_sha256": _stable_sha256(discrete),
    }


def topology_snapshot_document(
    entries: tuple[dict[str, Any], ...],
    *,
    rounding_decimals: int = 8,
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
    selected_ring_relations: int = DEFAULT_SELECTED_RING_RELATIONS,
) -> dict[str, Any]:
    return {
        "schema": TOPOLOGY_SNAPSHOT_SCHEMA,
        "description": (
            "Golden topology snapshots for MATRIX LINK output. The hash covers discrete "
            "bonds, rings, ring relations, aromaticity and connected fragments; "
            "continuous bond-order fields use an explicit cross-platform tolerance."
        ),
        "rounding_decimals": int(rounding_decimals),
        "numeric_tolerance": float(numeric_tolerance),
        "selected_ring_relations": int(selected_ring_relations),
        "entries": entries,
    }


def write_topology_snapshot(path: Path, xyzin: Path) -> Path:
    target = Path(path)
    payload = topology_snapshot_document((topology_snapshot_from_xyzin(Path(xyzin)),))
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def topology_report_lines(path: Path) -> list[str]:
    snapshot = topology_snapshot_from_xyzin(Path(path))
    lines = [
        "MATRIX Topology Report",
        "======================",
        "",
        f"Source: {snapshot['source'] or Path(path)}",
        f"Schema: {snapshot['topology_schema']}",
        f"Bonds: {snapshot['bond_count']}",
        f"Bond orders: {len(snapshot['bond_orders'])}",
        f"Bond-order component rows: {len(snapshot['bond_order_components'])}",
        f"Rings: {snapshot['ring_count']}",
        f"Ring basis policy: {snapshot['ring_basis'].get('policy', 'UNKNOWN')}",
        f"Ring basis algorithm: {snapshot['ring_basis'].get('algorithm', 'UNKNOWN')}",
        f"Ring basis complete: {snapshot['ring_basis'].get('complete', 'UNKNOWN')}",
        f"Ring candidates: {snapshot['ring_basis'].get('candidate_count', 'UNKNOWN')}",
        "Ring basis rank/count: "
        f"{snapshot['ring_basis'].get('rank', 'UNKNOWN')}/"
        f"{snapshot['ring_basis'].get('count', 'UNKNOWN')}",
        f"Ring basis excluded atoms: {_csv(snapshot['ring_basis'].get('excluded_atoms', ()))}",
        f"Ring relations: {snapshot['ring_relation_count']}",
        f"Fragments: {snapshot['fragment_count']}",
        f"Aromatic atoms: {_csv(snapshot['aromatic_atoms'])}",
        f"Aromatic bonds: {_csv(snapshot['aromatic_bonds'])}",
        f"Topology hash: {snapshot['topology_sha256']}",
        "",
        "Rings",
        "-----",
    ]
    if snapshot["rings"]:
        lines.extend(
            f"{ring['index']}: size={ring['size']} atoms={_csv(ring['atoms'])}"
            for ring in snapshot["rings"]
        )
    else:
        lines.append("NONE")
    lines.extend(["", "Ring Relations", "--------------"])
    if snapshot["selected_ring_relations"]:
        lines.extend(
            f"{item['left']}-{item['right']}: {item['kind']} shared_atoms={_csv(item['shared_atoms'])}"
            for item in snapshot["selected_ring_relations"]
        )
        if snapshot["ring_relation_count"] > len(snapshot["selected_ring_relations"]):
            lines.append(
                f"... {snapshot['ring_relation_count'] - len(snapshot['selected_ring_relations'])} more"
            )
    else:
        lines.append("NONE")
    return lines


def write_topology_report(path: Path, output: Path) -> Path:
    target = Path(output)
    target.write_text("\n".join(topology_report_lines(Path(path))) + "\n", encoding="utf-8")
    return target


def compare_topology_snapshot_entry(
    expected: dict[str, Any],
    xyzin: Path,
    *,
    rounding_decimals: int,
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> TopologySnapshotComparison:
    current = topology_snapshot_from_xyzin(
        Path(xyzin),
        case_id=str(expected.get("id", "")),
        source=str(expected.get("source", "")),
        rounding_decimals=rounding_decimals,
    )
    messages: list[str] = []
    for key in (
        "bond_count",
        "ring_count",
        "ring_relation_count",
        "fragment_count",
        "topology_sha256",
    ):
        if current.get(key) != expected.get(key):
            messages.append(
                f"{expected.get('id', '<unknown>')}: {key} changed "
                f"expected={expected.get(key)!r} current={current.get(key)!r}"
            )
    if current.get("topology_sha256") != expected.get("topology_sha256"):
        for key in (
            "bonds",
            "bond_orders",
            "rings",
            "ring_basis",
            "selected_ring_relations",
            "fragments",
        ):
            if current.get(key) != expected.get(key):
                messages.append(f"{expected.get('id', '<unknown>')}: first differing block {key}")
                break
    for key, numeric_fields in (
        ("bond_orders", ("value",)),
        ("bond_order_components", ("bond_order", "sigma", "pi", "pi_pi")),
    ):
        difference = _numeric_row_difference(
            expected.get(key, ()),
            current.get(key, ()),
            numeric_fields=numeric_fields,
            tolerance=numeric_tolerance,
        )
        if difference:
            messages.append(f"{expected.get('id', '<unknown>')}: {key} {difference}")
    return TopologySnapshotComparison(ok=not messages, messages=tuple(messages))


def _numeric_row_difference(
    expected: Any,
    current: Any,
    *,
    numeric_fields: tuple[str, ...],
    tolerance: float,
) -> str:
    expected_rows = tuple(expected)
    current_rows = tuple(current)
    if len(expected_rows) != len(current_rows):
        return f"row count changed expected={len(expected_rows)} current={len(current_rows)}"
    for index, (expected_row, current_row) in enumerate(
        zip(expected_rows, current_rows, strict=True)
    ):
        if tuple(expected_row.get("atoms", ())) != tuple(current_row.get("atoms", ())):
            return f"row {index} atoms changed"
        for field in numeric_fields:
            expected_value = float(expected_row[field])
            current_value = float(current_row[field])
            if abs(expected_value - current_value) > tolerance:
                return (
                    f"row {index} {field} changed expected={expected_value!r} "
                    f"current={current_value!r} tolerance={tolerance!r}"
                )
    return ""


def _parse_bonds(topology: list[str]) -> tuple[tuple[int, int], ...]:
    bonds = []
    for line in _subsection(topology, "BONDS"):
        if line.strip().upper() == "NONE":
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        i, j = int(parts[0]), int(parts[1])
        bonds.append(tuple(sorted((i, j))))
    return tuple(sorted(dict.fromkeys(bonds)))


def _parse_transitional_contacts(topology: list[str]) -> tuple[dict[str, Any], ...]:
    """Return raw ORACLE TS candidates without assigning a task or category."""

    records: dict[tuple[int, int], str] = {}
    for line in _subsection(topology, "TRANSITIONAL_CONTACTS"):
        fields = line.split()
        if not fields or fields[0].upper() == "NONE":
            continue
        if len(fields) != 3 or not fields[2].upper().startswith("KIND="):
            raise ValueError(f"invalid transitional-contact line: {line}")
        try:
            left, right = sorted((int(fields[0]), int(fields[1])))
        except ValueError as exc:
            raise ValueError(f"invalid transitional-contact atom indexes: {line}") from exc
        if left < 1 or left == right:
            raise ValueError(f"invalid transitional-contact atom indexes: {line}")
        kind = fields[2].split("=", 1)[1].strip().upper()
        if kind not in {"WEAK_ACYCLIC", "NEAR_COVALENT"}:
            raise ValueError(f"unsupported transitional-contact kind: {kind}")
        previous = records.get((left, right))
        if previous is not None and previous != kind:
            raise ValueError(f"conflicting transitional-contact kinds: {line}")
        records[(left, right)] = kind
    return tuple(
        {"atoms": pair, "kind": records[pair]} for pair in sorted(records)
    )


def _parse_bond_orders(
    topology: list[str],
    *,
    rounding_decimals: int,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in _subsection(topology, "BOND_ORDERS"):
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("(", " ").replace(")", " ").split()
        if len(parts) < 3:
            continue
        i, j = int(parts[0]), int(parts[1])
        rows.append(
            {
                "atoms": tuple(sorted((i, j))),
                "value": round(float(parts[2]), rounding_decimals),
            }
        )
    return tuple(sorted(rows, key=lambda item: item["atoms"]))


def _parse_bond_order_components(
    topology: list[str],
    *,
    rounding_decimals: int,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in _subsection(topology, "BOND_ORDER_COMPONENTS"):
        upper = line.strip().upper()
        if upper == "NONE" or upper.startswith("COLUMNS "):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        i, j = int(parts[0]), int(parts[1])
        numeric = tuple(float(value) for value in parts[2:7])
        if len(numeric) == 5:
            _legacy_connectivity, order, sigma, pi, pi_pi = numeric
        else:
            order = numeric[0]
            sigma, pi, pi_pi = numeric[1:]
        rows.append(
            {
                "atoms": tuple(sorted((i, j))),
                "bond_order": round(order, rounding_decimals),
                "sigma": round(sigma, rounding_decimals),
                "pi": round(pi, rounding_decimals),
                "pi_pi": round(pi_pi, rounding_decimals),
            }
        )
    return tuple(sorted(rows, key=lambda item: item["atoms"]))


def _parse_rings(topology: list[str]) -> tuple[dict[str, Any], ...]:
    rings = []
    for line in _subsection(topology, "RINGS"):
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
        if not parts:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        atoms: list[int] = []
        reading_atoms = False
        for part in parts[1:]:
            token = part.strip()
            if token.upper().startswith("ATOMS="):
                reading_atoms = True
                token = token.split("=", 1)[1]
            elif "=" in token and reading_atoms:
                break
            if reading_atoms and token:
                atoms.append(int(token))
        if atoms:
            rings.append({"index": index, "size": len(atoms), "atoms": tuple(atoms)})
    return tuple(rings)


def _parse_aromaticity(topology: list[str]) -> dict[str, tuple[Any, ...]]:
    atoms: tuple[int, ...] = ()
    bonds: tuple[str, ...] = ()
    for line in _subsection(topology, "AROMATICITY"):
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        values = parts[1:]
        if key == "ATOMS" and values and values[0].upper() != "NONE":
            atoms = tuple(int(value) for value in values)
        if key == "BONDS" and values and values[0].upper() != "NONE":
            bonds = tuple(values)
    return {"atoms": atoms, "bonds": bonds}


def _parse_ring_basis(topology: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "policy": _header_value(topology, "RING_BASIS_POLICY") or "UNSPECIFIED",
        "algorithm": _header_value(topology, "RING_BASIS_ALGORITHM") or "UNSPECIFIED",
    }
    complete = _header_value(topology, "RING_BASIS_COMPLETE")
    data["complete"] = complete is None or complete.upper() == "YES"
    numeric_keys = {
        "RING_CANDIDATE_COUNT": "candidate_count",
        "RING_BASIS_RANK": "rank",
        "RING_BASIS_COUNT": "count",
        "RING_BASIS_ALLOWED_ATOMS": "allowed_atom_count",
        "RING_BASIS_ALLOWED_EDGES": "allowed_edge_count",
        "RING_BASIS_MAXIMUM_SIZE": "maximum_size",
    }
    for source_key, target_key in numeric_keys.items():
        value = _header_value(topology, source_key)
        if value is not None:
            data[target_key] = int(value)
    excluded = _header_value(topology, "RING_BASIS_EXCLUDED_ATOMS")
    if excluded and excluded.upper() != "NONE":
        data["excluded_atoms"] = tuple(int(item) for item in excluded.replace(",", " ").split())
    else:
        data["excluded_atoms"] = ()
    return data


def _connected_components(
    bonds: tuple[tuple[int, int], ...],
    *,
    natoms: int,
) -> tuple[tuple[int, ...], ...]:
    atoms = list(range(1, int(natoms) + 1))
    parent = {atom: atom for atom in atoms}

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in bonds:
        union(left, right)
    components: dict[int, list[int]] = {}
    for atom in atoms:
        components.setdefault(find(atom), []).append(atom)
    return tuple(tuple(sorted(component)) for component in sorted(components.values()))


def _ring_relations(rings: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    relations = []
    for left_index, left in enumerate(rings):
        left_atoms = set(left["atoms"])
        left_bonds = _ring_bonds(tuple(left["atoms"]))
        for right in rings[left_index + 1 :]:
            shared_atoms = tuple(sorted(left_atoms & set(right["atoms"])))
            if not shared_atoms:
                continue
            shared_bonds = tuple(sorted(left_bonds & _ring_bonds(tuple(right["atoms"]))))
            if shared_bonds:
                kind = "FUSED"
            elif len(shared_atoms) == 1:
                kind = "SPIRO"
            else:
                kind = "BRIDGED"
            relations.append(
                {
                    "left": left["index"],
                    "right": right["index"],
                    "kind": kind,
                    "shared_atoms": shared_atoms,
                    "shared_bonds": shared_bonds,
                }
            )
    return tuple(relations)


def _ring_bonds(atoms: tuple[int, ...]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((atoms[index], atoms[(index + 1) % len(atoms)])))
        for index in range(len(atoms))
    }


def _subsection(section_lines: list[str], name: str) -> list[str]:
    marker = f"[{name.upper()}]"
    start = None
    for index, line in enumerate(section_lines):
        if line.strip().upper() == marker:
            start = index + 1
            break
    if start is None:
        return []
    end = len(section_lines)
    for index in range(start, len(section_lines)):
        text = section_lines[index].strip()
        if text.startswith("[") and text.endswith("]"):
            end = index
            break
    return section_lines[start:end]


def _header_value(section_lines: list[str], key: str) -> str | None:
    prefix = key.upper()
    for line in section_lines:
        text = line.strip()
        if not text or text.startswith("["):
            continue
        parts = text.split(maxsplit=1)
        if parts and parts[0].upper() == prefix:
            return parts[1].strip() if len(parts) > 1 else ""
    return None


def _stable_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _csv(values: Any) -> str:
    if not values:
        return "NONE"
    return ",".join(str(value) for value in values)
