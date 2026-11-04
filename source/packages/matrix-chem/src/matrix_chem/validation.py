from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from matrix_core import read_sectioned_lines, replace_section, section_content

from .geometry_io import GeometryParseError, read_enriched_xyz
from .topology.elements import atomic_number
from .topology.periodic_properties import periodic_atomic_properties
from .topology.contracts import (
    MATRIX_XYZ_FRAGMENTS_SCHEMA,
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    MATRIX_XYZ_VALIDATION_SCHEMA,
    SUPPORTED_FRAGMENTS_SCHEMAS,
    SUPPORTED_SYNTHONS_SCHEMAS,
    SUPPORTED_TOPOLOGY_SCHEMAS,
    schema_from_line,
    schema_line_supported,
    supported_schema_text,
)


REQUIRED_SYMMETRY_SCHEMA = "oracle.xyz.symmetry.v1"
REQUIRED_TOPOLOGY_SCHEMA = MATRIX_XYZ_TOPOLOGY_SCHEMA
REQUIRED_SYNTHONS_SCHEMA = MATRIX_XYZ_SYNTHONS_SCHEMA
OPTIONAL_FRAGMENTS_SCHEMA = MATRIX_XYZ_FRAGMENTS_SCHEMA


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    code: str
    text: str


@dataclass(frozen=True)
class ValidationResult:
    status: str
    messages: tuple[ValidationMessage, ...]


def validate_enriched_molecule(path: Path, *, require_fragments: bool = False) -> ValidationResult:
    """Validate an enriched XYZ after topology/synthon preprocessing."""
    target = Path(path)
    lines = read_sectioned_lines(target)
    messages: list[ValidationMessage] = []

    atom_symbols = _validate_geometry(target, messages)
    try:
        atom_coordinates = read_enriched_xyz(target).coordinates_angstrom
    except GeometryParseError:
        atom_coordinates = ()
    atom_count = len(atom_symbols)
    _require_schema(lines, "SYMMETRY", (REQUIRED_SYMMETRY_SCHEMA,), messages)
    _require_schema(lines, "TOPOLOGY", SUPPORTED_TOPOLOGY_SCHEMAS, messages)
    _require_schema(lines, "SYNTHONS", SUPPORTED_SYNTHONS_SCHEMAS, messages)
    if require_fragments:
        _require_schema(lines, "FRAGMENTS", SUPPORTED_FRAGMENTS_SCHEMAS, messages)
    elif section_content(lines, "FRAGMENTS"):
        _require_schema(lines, "FRAGMENTS", SUPPORTED_FRAGMENTS_SCHEMAS, messages)
    if atom_count > 0:
        _validate_topology_contract(lines, atom_symbols, atom_coordinates, messages)
        _validate_fragments_contract(lines, atom_count, messages)

    status = _status_from_messages(messages)
    if not messages:
        messages.append(
            ValidationMessage("INFO", "VALIDATION_PASS", "Molecule is ready for SMITH coordinate construction")
        )
    return ValidationResult(status=status, messages=tuple(messages))


def validation_section_lines(result: ValidationResult) -> list[str]:
    lines = [
        f"SCHEMA {MATRIX_XYZ_VALIDATION_SCHEMA}",
        f"STATUS {result.status}",
        "DEPENDENCIES SYMMETRY=oracle.xyz.symmetry.v1 "
        f"TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA}",
        f"OPTIONAL FRAGMENTS={MATRIX_XYZ_FRAGMENTS_SCHEMA}",
        "[MESSAGES]",
    ]
    lines.extend(f"{msg.level} {msg.code} {msg.text}" for msg in result.messages)
    return lines


def write_validation_section(path: Path, *, require_fragments: bool = False) -> ValidationResult:
    result = validate_enriched_molecule(path, require_fragments=require_fragments)
    replace_section(Path(path), "VALIDATION", validation_section_lines(result))
    return result


def _validate_geometry(path: Path, messages: list[ValidationMessage]) -> tuple[str, ...]:
    try:
        geometry = read_enriched_xyz(path)
    except GeometryParseError as exc:
        messages.append(ValidationMessage("ERROR", "INVALID_XYZ", str(exc)))
        return ()
    if geometry.natoms <= 0:
        messages.append(ValidationMessage("ERROR", "NO_ATOMS", "XYZ block contains no atoms"))
        return ()
    for idx, row in enumerate(geometry.coordinates_angstrom, start=1):
        if not all(math.isfinite(float(value)) for value in row):
            messages.append(
                ValidationMessage(
                    "ERROR",
                    "NONFINITE_COORDINATE",
                    f"Atom {idx} has invalid coordinates",
                )
            )
            return ()
    return tuple(str(atom) for atom in geometry.atoms)


def _require_schema(
    lines: list[str],
    section_name: str,
    schemas: tuple[str, ...],
    messages: list[ValidationMessage],
) -> None:
    content = section_content(lines, section_name)
    if not content:
        messages.append(
            ValidationMessage("ERROR", f"MISSING_{section_name}", f"Missing #{section_name}")
        )
        return
    if not schema_line_supported(content[0], schemas):
        messages.append(
            ValidationMessage(
                "ERROR",
                f"BAD_{section_name}_SCHEMA",
                f"#{section_name} must start with {supported_schema_text(schemas)}; "
                f"found SCHEMA {schema_from_line(content[0]) or '<missing>'}",
            )
        )


def _status_from_messages(messages: list[ValidationMessage]) -> str:
    levels = {message.level for message in messages}
    if "ERROR" in levels:
        return "FAIL"
    if "WARN" in levels:
        return "WARN"
    return "PASS"


def _validate_topology_contract(
    lines: list[str],
    atom_symbols: tuple[str, ...],
    atom_coordinates,
    messages: list[ValidationMessage],
) -> None:
    atom_count = len(atom_symbols)
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        return
    bonds = _parse_bonds(_subsection(topology, "BONDS"), atom_count, messages)
    bond_orders = _parse_bond_orders(_subsection(topology, "BOND_ORDERS"), atom_count, messages)
    missing_orders = sorted(set(bonds) - set(bond_orders))
    if missing_orders:
        text = ", ".join(f"{left}-{right}" for left, right in missing_orders[:8])
        messages.append(
            ValidationMessage(
                "ERROR",
                "MISSING_BOND_ORDERS",
                f"#TOPOLOGY lacks bond orders for {text}",
            )
        )
    if schema_from_line(topology[0]) == MATRIX_XYZ_TOPOLOGY_SCHEMA:
        components = _parse_bond_order_components(
            _subsection(topology, "BOND_ORDER_COMPONENTS"), atom_count, messages
        )
        missing_components = sorted(set(bonds) - set(components))
        if missing_components:
            text = ", ".join(f"{left}-{right}" for left, right in missing_components[:8])
            messages.append(
                ValidationMessage(
                    "ERROR",
                    "MISSING_BOND_ORDER_COMPONENTS",
                    f"#TOPOLOGY lacks sigma/pi/pi-pi indices for {text}",
                )
            )
        for bond, values in components.items():
            topology_order, multiplicity, sigma, pi, pi_pi = values
            if abs(multiplicity - (sigma + pi + pi_pi)) > 1.0e-8 * max(
                1.0, multiplicity
            ):
                messages.append(
                    ValidationMessage(
                        "ERROR",
                        "INCONSISTENT_BOND_ORDER_COMPONENTS",
                        f"Bond {bond[0]}-{bond[1]} components do not sum to the bond order",
                    )
                )
            if bond in bond_orders and abs(topology_order - bond_orders[bond]) > 1.0e-8 * max(
                1.0, topology_order
            ):
                messages.append(
                    ValidationMessage(
                        "ERROR",
                        "BOND_ORDER_COMPONENT_TOTAL_MISMATCH",
                        f"Bond {bond[0]}-{bond[1]} component total differs from [BOND_ORDERS]",
                    )
                )
    bonded_atoms = {atom for bond in bonds for atom in bond}
    isolated = [atom for atom in range(1, atom_count + 1) if atom not in bonded_atoms]
    fragment_memberships = _fragment_memberships(lines)
    complete_fragments = (
        _section_value(section_content(lines, "FRAGMENTS"), "STATUS") == "BUILT"
        and sorted(atom for fragment in fragment_memberships for atom in fragment)
        == list(range(1, atom_count + 1))
    )
    singleton_atoms = {
        fragment[0] for fragment in fragment_memberships if len(fragment) == 1
    }
    # A geometrically disconnected centre is still a valid fragment member
    # when ORACLE has supplied the deterministic nearest-centre pseudo-bond
    # used by the primitive chart.  This is required for distorted
    # intermolecular inputs (e.g. QIQGOM), where a heteroatom may temporarily
    # fall outside the covalent bond threshold.  Do not accept an unlinked
    # isolated atom: the nearest-fragment support must be explicit.
    try:
        from .structural_corrections import pseudo_bond_pairs

        linked_singletons = {
            atom
            for pair, _source in pseudo_bond_pairs(
                tuple(atomic_number(symbol) for symbol in atom_symbols),
                atom_coordinates,
                tuple((left - 1, right - 1) for left, right in bonds),
            )
            for atom in pair
            if atom + 1 in isolated
        }
        linked_singletons = {atom + 1 for atom in linked_singletons}
    except (ImportError, OSError, ValueError, TypeError):
        linked_singletons = set()
    invalid_isolated = [
        atom
        for atom in isolated
        if not complete_fragments
        or atom not in singleton_atoms
        or atom not in linked_singletons
    ]
    if invalid_isolated:
        messages.append(
            ValidationMessage(
                "ERROR",
                "ISOLATED_TOPOLOGY_ATOMS",
                "Unsupported isolated atoms or atoms without a complete singleton fragment: "
                + ",".join(str(atom) for atom in invalid_isolated),
            )
        )
    bond_set = set(bonds)
    for ring_index, atoms in _parse_rings(_subsection(topology, "RINGS"), atom_count, messages):
        ring_bonds = {
            tuple(sorted((atoms[index], atoms[(index + 1) % len(atoms)])))
            for index in range(len(atoms))
        }
        missing = sorted(ring_bonds - bond_set)
        if missing:
            messages.append(
                ValidationMessage(
                    "ERROR",
                    "INVALID_RING_BONDS",
                    f"Ring {ring_index} contains nonbonded edges "
                    + ",".join(f"{left}-{right}" for left, right in missing),
                )
            )


def _singleton_element_is_supported(symbol: str) -> bool:
    """Restrict isolated atomic fragments to the current explicit policy."""

    number = atomic_number(symbol)
    if number is None or not 1 <= int(number) <= 118:
        return False
    return periodic_atomic_properties(int(number)).group in {1, 2, 17, 18}


def _validate_fragments_contract(
    lines: list[str],
    atom_count: int,
    messages: list[ValidationMessage],
) -> None:
    fragments = section_content(lines, "FRAGMENTS")
    if not fragments or not schema_line_supported(fragments[0], SUPPORTED_FRAGMENTS_SCHEMAS):
        return
    fragment_atoms: list[int] = []
    for line in fragments:
        text = line.strip()
        if not text or text.upper().startswith(("SCHEMA ", "ALIAS_SCHEMA ", "INDEXING ", "[")):
            continue
        if "ATOMS=" not in text.upper():
            continue
        reading_atoms = False
        for token in text.replace(",", " ").split():
            upper = token.upper()
            if upper.startswith("ATOMS="):
                reading_atoms = True
                token = token.split("=", 1)[1]
            elif "=" in token and reading_atoms:
                break
            elif not reading_atoms:
                continue
            if not token:
                continue
            for atom_text in token.replace(";", " ").split():
                if not atom_text:
                    continue
                try:
                    fragment_atoms.append(int(atom_text))
                except ValueError:
                    continue
    if not fragment_atoms:
        return
    invalid = [atom for atom in fragment_atoms if atom < 1 or atom > atom_count]
    if invalid:
        messages.append(
            ValidationMessage(
                "ERROR",
                "INVALID_FRAGMENT_ATOMS",
                "Fragment atom indexes out of range: " + ",".join(str(atom) for atom in invalid),
            )
        )
    expected = list(range(1, atom_count + 1))
    if sorted(fragment_atoms) != expected:
        messages.append(
            ValidationMessage(
                "ERROR",
                "INCOMPLETE_FRAGMENT_COVERAGE",
                "Built fragments must contain every atom exactly once",
            )
        )


def _fragment_memberships(lines: list[str]) -> tuple[tuple[int, ...], ...]:
    fragments = section_content(lines, "FRAGMENTS")
    memberships: list[tuple[int, ...]] = []
    in_records = False
    for line in fragments:
        text = line.strip()
        if text.startswith("[") and text.endswith("]"):
            in_records = text.upper() == "[FRAGMENTS]"
            continue
        if not in_records or "ATOMS=" not in text.upper():
            continue
        atoms_text = text.upper().split("ATOMS=", 1)[1].split(None, 1)[0]
        try:
            memberships.append(
                tuple(int(atom) for atom in atoms_text.replace(";", ",").split(",") if atom)
            )
        except ValueError:
            continue
    return tuple(memberships)


def _section_value(lines: list[str], key: str) -> str | None:
    prefix = key.upper() + " "
    for line in lines:
        text = line.strip()
        if text.upper().startswith(prefix):
            return text[len(prefix) :].strip().upper()
    return None


def _parse_bonds(
    rows: list[str],
    atom_count: int,
    messages: list[ValidationMessage],
) -> tuple[tuple[int, int], ...]:
    bonds: list[tuple[int, int]] = []
    for line in rows:
        if line.strip().upper() == "NONE":
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            left, right = int(parts[0]), int(parts[1])
        except ValueError:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_ROW", line))
            continue
        if left == right or left < 1 or right < 1 or left > atom_count or right > atom_count:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_INDEX", line))
            continue
        bonds.append(tuple(sorted((left, right))))
    return tuple(sorted(dict.fromkeys(bonds)))


def _parse_bond_orders(
    rows: list[str],
    atom_count: int,
    messages: list[ValidationMessage],
) -> dict[tuple[int, int], float]:
    orders: dict[tuple[int, int], float] = {}
    for line in rows:
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("(", " ").replace(")", " ").split()
        if len(parts) < 3:
            continue
        try:
            left, right = int(parts[0]), int(parts[1])
            value = float(parts[2])
        except ValueError:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_ORDER_ROW", line))
            continue
        if left == right or left < 1 or right < 1 or left > atom_count or right > atom_count:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_ORDER_INDEX", line))
            continue
        if not math.isfinite(value) or value < 0.0:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_ORDER_VALUE", line))
            continue
        orders[tuple(sorted((left, right)))] = value
    return orders


def _parse_bond_order_components(
    rows: list[str],
    atom_count: int,
    messages: list[ValidationMessage],
) -> dict[tuple[int, int], tuple[float, float, float, float, float]]:
    components: dict[tuple[int, int], tuple[float, float, float, float, float]] = {}
    for line in rows:
        upper = line.strip().upper()
        if upper == "NONE" or upper.startswith("COLUMNS "):
            continue
        parts = line.split()
        if len(parts) < 6:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_COMPONENT_ROW", line))
            continue
        try:
            left, right = int(parts[0]), int(parts[1])
            numeric = tuple(float(value) for value in parts[2:7])
        except ValueError:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_COMPONENT_ROW", line))
            continue
        if left == right or left < 1 or right < 1 or left > atom_count or right > atom_count:
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_COMPONENT_INDEX", line))
            continue
        # Accept the short-lived six-column draft as a self-consistent legacy
        # form while always emitting the v2 seven-column contract.
        values = numeric if len(numeric) == 5 else (numeric[0], *numeric)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            messages.append(ValidationMessage("ERROR", "INVALID_BOND_COMPONENT_VALUE", line))
            continue
        components[tuple(sorted((left, right)))] = values  # type: ignore[assignment]
    return components


def _parse_rings(
    rows: list[str],
    atom_count: int,
    messages: list[ValidationMessage],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    rings: list[tuple[int, tuple[int, ...]]] = []
    for line in rows:
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
            if reading_atoms and token:
                try:
                    atoms.append(int(token))
                except ValueError:
                    messages.append(ValidationMessage("ERROR", "INVALID_RING_ROW", line))
                    atoms = []
                    break
        if not atoms:
            continue
        if len(atoms) < 3 or len(set(atoms)) != len(atoms):
            messages.append(ValidationMessage("ERROR", "INVALID_RING_ATOMS", line))
            continue
        if any(atom < 1 or atom > atom_count for atom in atoms):
            messages.append(ValidationMessage("ERROR", "INVALID_RING_INDEX", line))
            continue
        rings.append((ring_index, tuple(atoms)))
    return tuple(rings)


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
