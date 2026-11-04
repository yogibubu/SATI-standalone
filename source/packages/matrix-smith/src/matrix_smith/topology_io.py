"""Read topology data consumed by the SMITH coordinate builder."""

from __future__ import annotations

import numpy as np

from matrix_chem.topology.contracts import SUPPORTED_TOPOLOGY_SCHEMAS, schema_line_supported
from matrix_core import section_content

from .contracts import GICForgeContractError


def _topology_bonds(lines: list[str], *, natoms: int) -> tuple[tuple[int, int], ...]:
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        raise GICForgeContractError("missing valid #TOPOLOGY section")
    bond_lines = _subsection(topology, "BONDS")
    if not bond_lines or any(line.strip().upper() == "NONE" for line in bond_lines):
        raise GICForgeContractError("#TOPOLOGY contains no bonds")
    bonds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for line in bond_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond line: {line}") from exc
        if i == j or i < 1 or j < 1 or i > natoms or j > natoms:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond indexes: {line}")
        bond = tuple(sorted((i, j)))
        if bond not in seen:
            seen.add(bond)
            bonds.append(bond)
    return tuple(sorted(bonds))


def topology_bond_orders_from_lines(
    lines: list[str],
    *,
    natoms: int,
) -> dict[tuple[int, int], float]:
    """Read optional one-based #TOPOLOGY [BOND_ORDERS] rows."""
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        return {}
    bond_order_lines = _subsection(topology, "BOND_ORDERS")
    orders: dict[tuple[int, int], float] = {}
    for line in bond_order_lines:
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("(", " ").replace(")", " ").split()
        if len(parts) < 3:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
            value = float(parts[2])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond-order line: {line}") from exc
        if i == j or i < 1 or j < 1 or i > natoms or j > natoms:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond-order indexes: {line}")
        orders[tuple(sorted((i, j)))] = value
    return orders


def topology_bond_order_components_from_lines(
    lines: list[str],
    *,
    natoms: int,
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Read one-based canonical ``(sigma, pi, pi_pi)`` bond indices."""

    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        return {}
    rows = _subsection(topology, "BOND_ORDER_COMPONENTS")
    result: dict[tuple[int, int], tuple[float, float, float]] = {}
    for line in rows:
        upper = line.strip().upper()
        if upper == "NONE" or upper.startswith("COLUMNS "):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
            numeric = tuple(float(value) for value in parts[2:7])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid bond-order-component line: {line}") from exc
        if len(numeric) == 5:
            _topology, multiplicity, sigma, pi, pi_pi = numeric
        else:
            multiplicity, sigma, pi, pi_pi = numeric
        if i == j or i < 1 or j < 1 or i > natoms or j > natoms:
            raise GICForgeContractError(f"invalid bond-order-component indexes: {line}")
        if min(multiplicity, sigma, pi, pi_pi) < 0.0 or not np.isclose(
            multiplicity, sigma + pi + pi_pi, rtol=1.0e-8, atol=1.0e-10
        ):
            raise GICForgeContractError(f"inconsistent bond-order components: {line}")
        result[tuple(sorted((i, j)))] = (sigma, pi, pi_pi)
    return result


def _topology_rings(lines: list[str], *, natoms: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        raise GICForgeContractError("missing valid #TOPOLOGY section")
    ring_lines = _subsection(topology, "RINGS")
    rings: list[tuple[int, tuple[int, ...]]] = []
    for line in ring_lines:
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
        if not parts:
            continue
        try:
            ring_index = int(parts[0])
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
            if not reading_atoms or not token:
                continue
            try:
                atoms.append(int(token))
            except ValueError as exc:
                raise GICForgeContractError(f"invalid #TOPOLOGY ring line: {line}") from exc
        if len(atoms) < 3:
            continue
        if any(atom < 1 or atom > natoms for atom in atoms):
            raise GICForgeContractError(f"invalid #TOPOLOGY ring atom indexes: {line}")
        rings.append((ring_index, tuple(dict.fromkeys(atoms))))
    return tuple(rings)


def topology_aromatic_atoms_from_lines(
    lines: list[str],
    *,
    natoms: int,
    required: bool = True,
) -> frozenset[int]:
    """Read ORACLE's one-based aromatic atom assignment from ``#TOPOLOGY``."""

    topology = section_content(lines, "TOPOLOGY")
    if not topology:
        if not required:
            return frozenset()
        raise GICForgeContractError("missing valid #TOPOLOGY section")
    if not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        raise GICForgeContractError("missing valid #TOPOLOGY section")
    aromaticity = _subsection(topology, "AROMATICITY")
    atoms: set[int] = set()
    for line in aromaticity:
        parts = line.replace(",", " ").split()
        if not parts or parts[0].upper() != "ATOMS":
            continue
        if any(token.upper() == "NONE" for token in parts[1:]):
            return frozenset()
        try:
            atoms.update(int(token) for token in parts[1:])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid #TOPOLOGY aromatic atom line: {line}") from exc
        break
    if any(atom < 1 or atom > natoms for atom in atoms):
        raise GICForgeContractError("invalid #TOPOLOGY aromatic atom indexes")
    return frozenset(atoms)


def _subsection(section_lines: list[str], name: str) -> list[str]:
    wanted = name.strip().upper()
    content: list[str] = []
    active = False
    for line in section_lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            active = stripped[1:-1].strip().upper() == wanted
            continue
        if active and stripped:
            content.append(stripped)
    return content
