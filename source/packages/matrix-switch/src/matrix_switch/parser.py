"""Dependency-free SMILES parser for the SWITCH molecular-graph contract."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from functools import lru_cache
import re

from matrix_chem.topology.elements import atomic_symbol

from .model import SwitchAtom, SwitchBond, SwitchMolecularGraph


class SmilesParseError(ValueError):
    """SMILES syntax error carrying an exact source offset."""

    def __init__(self, message: str, source: str, offset: int):
        pointer = " " * max(0, offset) + "^"
        super().__init__(f"{message} at offset {offset}\n{source}\n{pointer}")
        self.message = message
        self.source = source
        self.offset = int(offset)


class SwitchUnsupportedFeatureError(SmilesParseError):
    """Valid SMILES feature that the current SWITCH contract does not implement."""


@dataclass(frozen=True)
class _BondSpec:
    order: float
    aromatic: bool = False
    direction: str | None = None
    dative: str | None = None
    explicit: bool = True


_BONDS = {
    "-": _BondSpec(1.0),
    "=": _BondSpec(2.0),
    "#": _BondSpec(3.0),
    "$": _BondSpec(4.0),
    ":": _BondSpec(1.5, aromatic=True),
    "/": _BondSpec(1.0, direction="/"),
    "\\": _BondSpec(1.0, direction="\\"),
    "~": _BondSpec(0.0),
    "->": _BondSpec(1.0, dative="->"),
    "<-": _BondSpec(1.0, dative="<-"),
}
_ORGANIC = ("Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I")
_AROMATIC = ("se", "as", "b", "c", "n", "o", "p", "s")
_ELEMENT_SYMBOLS = frozenset(atomic_symbol(number) for number in range(1, 119))
_CHIRALITY = re.compile(r"@(?:@|TH[12]|AL[12]|SP[123]|TB(?:[1-9]|1[0-9]|20)|OH(?:[1-9]|[12][0-9]|30))?")


def parse_smiles(smiles: str) -> SwitchMolecularGraph:
    source = str(smiles).strip()
    if not source:
        raise SmilesParseError("empty SMILES", source, 0)
    return _parse_smiles_cached(source)


def clear_smiles_parse_cache() -> None:
    """Clear the bounded parser cache for cold-path benchmarks and diagnostics."""

    _parse_smiles_cached.cache_clear()


@lru_cache(maxsize=512)
def _parse_smiles_cached(source: str) -> SwitchMolecularGraph:
    atoms: list[SwitchAtom] = []
    bonds: list[SwitchBond] = []
    components: list[list[int]] = []
    component: list[int] = []
    branch_stack: list[tuple[int, int, int]] = []
    rings: dict[str, tuple[int, _BondSpec | None, int]] = {}
    neighbor_events: dict[int, list[int | tuple[str, str]]] = {}
    incoming_atoms: set[int] = set()
    current: int | None = None
    pending: _BondSpec | None = None
    position = 0

    while position < len(source):
        char = source[position]
        if char == "(":
            if current is None:
                raise SmilesParseError("branch has no parent atom", source, position)
            branch_stack.append((current, position, len(atoms)))
            position += 1
            continue
        if char == ")":
            if not branch_stack:
                raise SmilesParseError("unmatched branch close", source, position)
            parent, opening, atom_count = branch_stack.pop()
            if pending is not None:
                raise SmilesParseError("branch ends with a bond", source, position)
            if len(atoms) == atom_count:
                raise SmilesParseError("empty branch", source, opening)
            current = parent
            position += 1
            continue
        if char == ".":
            if pending is not None:
                raise SmilesParseError("bond cannot precede a component separator", source, position)
            if branch_stack:
                raise SmilesParseError(
                    "component separator cannot occur inside a branch", source, position
                )
            if not component:
                raise SmilesParseError("empty disconnected component", source, position)
            components.append(component)
            component = []
            current = None
            position += 1
            continue
        bond_token = _bond_token(source, position)
        if bond_token is not None:
            if current is None or pending is not None:
                raise SmilesParseError("bond is not between two atoms", source, position)
            pending = _BONDS[bond_token]
            position += len(bond_token)
            continue
        ring = _ring_token(source, position)
        if ring is not None:
            label, consumed = ring
            if current is None:
                raise SmilesParseError("ring closure has no atom", source, position)
            if label not in rings:
                rings[label] = (current, pending, position)
                neighbor_events[current].append(("ring", label))
            else:
                other, opening, _ = rings.pop(label)
                if other == current:
                    raise SmilesParseError("ring closure cannot join an atom to itself", source, position)
                spec = _merge_ring_bond(opening, pending, source, position)
                bonds.append(_make_bond(other, current, spec, atoms, ring_label=label))
                marker = ("ring", label)
                neighbor_events[other][neighbor_events[other].index(marker)] = current
                neighbor_events[current].append(other)
            pending = None
            position += consumed
            continue

        atom, consumed = _parse_atom(source, position, len(atoms))
        atoms.append(atom)
        neighbor_events[atom.index] = []
        component.append(atom.index)
        if current is not None:
            bonds.append(_make_bond(current, atom.index, pending, atoms))
            neighbor_events[current].append(atom.index)
            neighbor_events[atom.index].append(current)
            incoming_atoms.add(atom.index)
        elif pending is not None:
            raise SmilesParseError("bond has no left atom", source, position)
        current = atom.index
        pending = None
        position += consumed

    if branch_stack:
        _, opening, _ = branch_stack[-1]
        raise SmilesParseError("unclosed branch", source, opening)
    if rings:
        label, (_, _, opening) = next(iter(rings.items()))
        raise SmilesParseError(f"unclosed ring {label!r}", source, opening)
    if pending is not None:
        raise SmilesParseError("trailing bond", source, len(source) - 1)
    if not component:
        raise SmilesParseError("trailing component separator", source, len(source) - 1)
    components.append(component)
    if not atoms:
        raise SmilesParseError("SMILES contains no atoms", source, 0)
    _validate_bonds(bonds, source)
    finalized_atoms = []
    for atom in atoms:
        explicit = tuple(
            value
            for value in neighbor_events[atom.index]
            if isinstance(value, int)
        )
        implicit = (None,) * (atom.hydrogen_count or 0)
        stereo_neighbors = (
            explicit + implicit
            if atom.index in incoming_atoms
            else implicit + explicit
        )
        finalized_atoms.append(
            replace(atom, stereo_neighbors=stereo_neighbors)
        )
    return SwitchMolecularGraph(
        atoms=tuple(finalized_atoms),
        bonds=tuple(bonds),
        components=tuple(tuple(indices) for indices in components),
        source_smiles=source,
        total_formal_charge=sum(atom.formal_charge for atom in atoms),
    )


def _parse_atom(source: str, position: int, index: int) -> tuple[SwitchAtom, int]:
    if source[position] == "[":
        close = source.find("]", position + 1)
        if close < 0:
            raise SmilesParseError("unclosed bracket atom", source, position)
        content = source[position + 1 : close]
        return _parse_bracket_atom(content, source, position, close + 1, index), close + 1 - position
    for token in (*_ORGANIC, *_AROMATIC):
        if source.startswith(token, position):
            aromatic = token in _AROMATIC
            symbol = token.capitalize() if aromatic else token
            return (
                SwitchAtom(
                    index=index,
                    symbol=symbol,
                    aromatic=aromatic,
                    bracketed=False,
                    source_span=(position, position + len(token)),
                ),
                len(token),
            )
    raise SmilesParseError("expected an atom", source, position)


def _parse_bracket_atom(
    content: str,
    source: str,
    start: int,
    stop: int,
    index: int,
) -> SwitchAtom:
    cursor = 0
    isotope_match = re.match(r"\d+", content)
    isotope = None
    if isotope_match:
        isotope = int(isotope_match.group())
        cursor = isotope_match.end()
    token = None
    if cursor < len(content) and content[cursor] == "*":
        token = "*"
        cursor += 1
    elif cursor < len(content) and content[cursor].islower():
        candidate = content[cursor : cursor + 2]
        if candidate in {"se", "as"}:
            token = candidate
            cursor += 2
        elif content[cursor] in {"b", "c", "n", "o", "p", "s"}:
            token = content[cursor]
            cursor += 1
    elif cursor < len(content) and content[cursor].isupper():
        candidate = content[cursor]
        if cursor + 1 < len(content) and content[cursor + 1].islower():
            candidate += content[cursor + 1]
        if candidate in _ELEMENT_SYMBOLS:
            token = candidate
            cursor += len(candidate)
    if token is None:
        raise SmilesParseError("invalid bracket atom element", source, start + 1 + cursor)
    aromatic = token in _AROMATIC
    symbol = token.capitalize() if aromatic else token

    chirality = None
    match = _CHIRALITY.match(content, cursor)
    if match:
        chirality = match.group()
        cursor = match.end()
    hydrogen_count = 0
    if cursor < len(content) and content[cursor] == "H":
        cursor += 1
        count_match = re.match(r"\d+", content[cursor:])
        hydrogen_count = 1
        if count_match:
            hydrogen_count = int(count_match.group())
            cursor += len(count_match.group())
    formal_charge = 0
    if cursor < len(content) and content[cursor] in "+-":
        sign_char = content[cursor]
        sign = 1 if sign_char == "+" else -1
        cursor += 1
        number_match = re.match(r"\d+", content[cursor:])
        if number_match:
            formal_charge = sign * int(number_match.group())
            cursor += len(number_match.group())
        else:
            repetitions = 1
            while cursor < len(content) and content[cursor] == sign_char:
                repetitions += 1
                cursor += 1
            formal_charge = sign * repetitions
    atom_class = None
    if cursor < len(content) and content[cursor] == ":":
        cursor += 1
        class_match = re.match(r"\d+", content[cursor:])
        if not class_match:
            raise SmilesParseError("atom class needs an integer", source, start + 1 + cursor)
        atom_class = int(class_match.group())
        cursor += len(class_match.group())
    if cursor != len(content):
        raise SmilesParseError(
            "invalid bracket atom qualifier", source, start + 1 + cursor
        )
    return SwitchAtom(
        index=index,
        symbol=symbol,
        isotope=isotope,
        formal_charge=formal_charge,
        hydrogen_count=hydrogen_count,
        aromatic=aromatic,
        chirality=chirality,
        atom_class=atom_class,
        bracketed=True,
        source_span=(start, stop),
    )


def _make_bond(
    left: int,
    right: int,
    spec: _BondSpec | None,
    atoms: list[SwitchAtom],
    *,
    ring_label: str | None = None,
) -> SwitchBond:
    if spec is None:
        aromatic = atoms[left].aromatic and atoms[right].aromatic
        spec = _BondSpec(1.5 if aromatic else 1.0, aromatic=aromatic, explicit=False)
    return SwitchBond(
        left=left,
        right=right,
        order=spec.order,
        aromatic=spec.aromatic,
        direction=spec.direction,
        dative=spec.dative,
        ring_label=ring_label,
    )


def _merge_ring_bond(
    opening: _BondSpec | None,
    closing: _BondSpec | None,
    source: str,
    position: int,
) -> _BondSpec | None:
    if opening is not None and closing is not None and opening != closing:
        raise SmilesParseError("inconsistent bond symbols on ring closure", source, position)
    return closing if closing is not None else opening


def _bond_token(source: str, position: int) -> str | None:
    for token in ("->", "<-", "-", "=", "#", "$", ":", "/", "\\", "~"):
        if source.startswith(token, position):
            return token
    return None


def _ring_token(source: str, position: int) -> tuple[str, int] | None:
    if source[position].isdigit():
        return source[position], 1
    if source[position] != "%":
        return None
    if position + 1 < len(source) and source[position + 1] == "(":
        close = source.find(")", position + 2)
        if close < 0 or not source[position + 2 : close].isdigit():
            raise SmilesParseError("invalid extended ring label", source, position)
        return source[position + 2 : close], close + 1 - position
    digits = source[position + 1 : position + 3]
    if len(digits) != 2 or not digits.isdigit():
        raise SmilesParseError("percent ring labels need two digits", source, position)
    return digits, 3


def _validate_bonds(bonds: list[SwitchBond], source: str) -> None:
    seen: set[tuple[int, int]] = set()
    for bond in bonds:
        if bond.key in seen:
            raise SmilesParseError("duplicate bond", source, 0)
        seen.add(bond.key)


__all__ = ["SmilesParseError", "clear_smiles_parse_cache", "parse_smiles"]
