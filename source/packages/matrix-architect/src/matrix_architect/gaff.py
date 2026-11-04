"""Strict GAFF/GAFF2 parameter import and transferable atom typing.

ARCHITECT ships a hash-pinned official ``gaff2.dat`` inside MATRIX, so normal
operation is local and network-free.  A user path remains an explicit
reproducibility override, never a routine requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Sequence

from matrix_oracle import assign_gaff_atom_types


GAFF_PARAMETER_ENV = "MATRIX_GAFF_PARAMETER_FILE"
GAFF_PARAMETER_SET_SCHEMA = "matrix.zaff.gaff_parameter_set.v1"
GAFF_SUPPORTED_DOMAINS = ("hydrocarbons", "amino_acids", "sugars")
GAFF_BUNDLED_COMMIT = "0b35bfeb96026ffa4e5876391a0828f39b3cfc8d"
GAFF_BUNDLED_SHA256 = "2244b58627a85693776d12a0155a292f72b65b4398a03b87958136ad7460c67f"
GAFF_BUNDLED_SOURCE_URL = (
    "https://raw.githubusercontent.com/Amber-MD/AmberClassic/"
    f"{GAFF_BUNDLED_COMMIT}/dat/leap/parm/gaff2.dat"
)
GAFF_BUNDLED_LICENSE = "AmberClassic GPL-2.0-or-later; see bundled LICENSE"

_TYPE = r"(?:X|[A-Za-z][A-Za-z0-9+*]?)"
_BOND_RE = re.compile(rf"^\s*({_TYPE})\s*-\s*({_TYPE})\s+")
_ANGLE_RE = re.compile(rf"^\s*({_TYPE})\s*-\s*({_TYPE})\s*-\s*({_TYPE})\s+")
_TORSION_RE = re.compile(
    rf"^\s*({_TYPE})\s*-\s*({_TYPE})\s*-\s*({_TYPE})\s*-\s*({_TYPE})\s+"
)


@dataclass(frozen=True)
class GaffBond:
    types: tuple[str, str]
    force_constant_kcal_per_mol_angstrom2: float
    equilibrium_angstrom: float


@dataclass(frozen=True)
class GaffAngle:
    types: tuple[str, str, str]
    force_constant_kcal_per_mol_radian2: float
    equilibrium_degrees: float


@dataclass(frozen=True)
class GaffTorsionTerm:
    types: tuple[str, str, str, str]
    divisor: float
    barrier_kcal_per_mol: float
    phase_degrees: float
    periodicity: int


@dataclass(frozen=True)
class GaffVdw:
    atom_type: str
    rmin_half_angstrom: float
    epsilon_kcal_per_mol: float


@dataclass(frozen=True)
class GaffParameterSet:
    banner: str
    source_path: str
    source_sha256: str
    source_origin: str
    source_license: str
    bonds: tuple[GaffBond, ...]
    angles: tuple[GaffAngle, ...]
    torsions: tuple[GaffTorsionTerm, ...]
    impropers: tuple[GaffTorsionTerm, ...]
    vdw: tuple[GaffVdw, ...]
    schema: str = GAFF_PARAMETER_SET_SCHEMA

    @cached_property
    def _bond_lookup(self) -> dict[tuple[str, str], tuple[GaffBond, ...]]:
        return _symmetric_parameter_index(self.bonds)

    @cached_property
    def _angle_lookup(
        self,
    ) -> dict[tuple[str, str, str], tuple[GaffAngle, ...]]:
        return _symmetric_parameter_index(self.angles)

    @cached_property
    def _vdw_lookup(self) -> dict[str, tuple[GaffVdw, ...]]:
        result: dict[str, list[GaffVdw]] = {}
        for item in self.vdw:
            result.setdefault(item.atom_type, []).append(item)
        return {key: tuple(values) for key, values in result.items()}

    @classmethod
    def from_file(cls, path: Path | str) -> "GaffParameterSet":
        target = Path(path).expanduser().resolve()
        raw = target.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        lines = text.splitlines()
        if not lines or "AMBER General Force Field" not in lines[0]:
            raise ValueError("GAFF file lacks the canonical AMBER version banner")
        blocks = _parameter_blocks(lines)
        if len(blocks) < 5:
            raise ValueError("GAFF file does not contain bond/angle/torsion/improper blocks")
        bond_lines, angle_lines, torsion_lines, improper_lines = blocks[:4]
        vdw_lines = _vdw_block(lines)
        result = cls(
            banner=lines[0].strip(),
            source_path=str(target),
            source_sha256=digest,
            source_origin=(
                GAFF_BUNDLED_SOURCE_URL
                if digest == GAFF_BUNDLED_SHA256
                else str(target)
            ),
            source_license=(
                GAFF_BUNDLED_LICENSE
                if digest == GAFF_BUNDLED_SHA256
                else "externally supplied parameter data; consult its distribution"
            ),
            bonds=tuple(_parse_bond(line) for line in bond_lines),
            angles=tuple(_parse_angle(line) for line in angle_lines),
            torsions=tuple(_parse_torsion(line) for line in torsion_lines),
            impropers=tuple(_parse_improper(line) for line in improper_lines),
            vdw=tuple(_parse_vdw(line) for line in vdw_lines),
        )
        if not result.bonds or not result.angles or not result.torsions or not result.vdw:
            raise ValueError("GAFF parameter blocks must be nonempty")
        return result

    def bond(self, left: str, right: str) -> GaffBond:
        key = (str(left), str(right))
        return _indexed_unique_match(
            self._bond_lookup,
            min(key, tuple(reversed(key))),
            "bond",
            key,
        )

    def angle(self, left: str, center: str, right: str) -> GaffAngle:
        key = (str(left), str(center), str(right))
        return _indexed_unique_match(
            self._angle_lookup,
            min(key, tuple(reversed(key))),
            "angle",
            key,
        )

    def torsion(self, types: Sequence[str]) -> tuple[GaffTorsionTerm, ...]:
        key = tuple(str(value) for value in types)
        if len(key) != 4:
            raise ValueError("GAFF torsion lookup requires four atom types")
        candidates: list[tuple[int, int, GaffTorsionTerm]] = []
        for order, item in enumerate(self.torsions):
            score = _torsion_match_score(item.types, key)
            if score >= 0:
                candidates.append((score, order, item))
        if not candidates:
            raise KeyError(f"GAFF torsion parameters are unavailable for {key}")
        best = max(score for score, _order, _item in candidates)
        # GAFF represents one Fourier series by consecutive records sharing the
        # same most-specific pattern.  Preserve file order and every term.
        return tuple(item for score, _order, item in candidates if score == best)

    def vdw_for(self, atom_type: str) -> GaffVdw:
        key = str(atom_type)
        return _indexed_unique_match(
            self._vdw_lookup,
            key,
            "van der Waals",
            (key,),
        )


def default_gaff_parameter_path() -> Path | None:
    """Return the explicit override or MATRIX's resident, hash-pinned GAFF2."""

    configured = os.environ.get(GAFF_PARAMETER_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{GAFF_PARAMETER_ENV} points to a missing file: {path}"
            )
        return path
    bundled = Path(__file__).with_name("data") / "amberclassic" / "gaff2.dat"
    if not bundled.is_file():
        raise FileNotFoundError("MATRIX installation lacks its resident gaff2.dat")
    digest = hashlib.sha256(bundled.read_bytes()).hexdigest()
    if digest != GAFF_BUNDLED_SHA256:
        raise OSError("MATRIX resident gaff2.dat failed its pinned SHA-256 check")
    return bundled


def assign_initial_gaff_types(
    atomic_numbers: Sequence[int],
    graph,
    synthons,
    aromaticity,
    *,
    allow_periodic_fallback: bool = False,
) -> tuple[str, ...]:
    """Assign the conservative GAFF subset used by the first ZAFF library.

    The rules cover hydrocarbons, ordinary amino-acid functional groups and
    neutral sugars.  Unsupported elements or ambiguous hypervalence fail
    closed, which makes the shared resolver select GFN-FF.
    """

    translation = assign_gaff_atom_types(
        atomic_numbers,
        graph,
        synthons,
        aromaticity,
        strict=not allow_periodic_fallback,
    )
    return tuple(
        str(label) if label is not None else f"Z{int(atomic_numbers[index])}:PERIODIC"
        for index, label in enumerate(translation.labels)
    )


def _parameter_blocks(lines: Sequence[str]) -> list[list[str]]:
    try:
        first_blank = next(index for index, line in enumerate(lines) if not line.strip())
    except StopIteration as exc:
        raise ValueError("GAFF file has no parameter block separators") from exc
    index = first_blank + 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    # Hydrophilic atom-type list precedes the bonded parameter blocks.
    index += 1
    blocks: list[list[str]] = []
    for _ in range(5):
        block: list[str] = []
        while index < len(lines) and lines[index].strip():
            block.append(lines[index])
            index += 1
        blocks.append(block)
        while index < len(lines) and not lines[index].strip():
            index += 1
    return blocks


def _vdw_block(lines: Sequence[str]) -> list[str]:
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith("MOD4")) + 1
        end = next(index for index in range(start, len(lines)) if lines[index].strip() == "END")
    except StopIteration as exc:
        raise ValueError("GAFF file lacks its MOD4/END van der Waals block") from exc
    return [line for line in lines[start:end] if line.strip()]


def _tail(line: str, match: re.Match[str]) -> list[str]:
    return line[match.end() :].split()


def _parse_bond(line: str) -> GaffBond:
    match = _BOND_RE.match(line)
    if match is None:
        raise ValueError(f"invalid GAFF bond record: {line!r}")
    fields = _tail(line, match)
    return GaffBond((match[1], match[2]), float(fields[0]), float(fields[1]))


def _parse_angle(line: str) -> GaffAngle:
    match = _ANGLE_RE.match(line)
    if match is None:
        raise ValueError(f"invalid GAFF angle record: {line!r}")
    fields = _tail(line, match)
    return GaffAngle(
        (match[1], match[2], match[3]), float(fields[0]), float(fields[1])
    )


def _parse_torsion(line: str) -> GaffTorsionTerm:
    match = _TORSION_RE.match(line)
    if match is None:
        raise ValueError(f"invalid GAFF torsion record: {line!r}")
    fields = _tail(line, match)
    periodicity = int(round(abs(float(fields[3]))))
    if periodicity < 1:
        raise ValueError("GAFF torsion periodicity must be nonzero")
    return GaffTorsionTerm(
        (match[1], match[2], match[3], match[4]),
        float(fields[0]),
        float(fields[1]),
        float(fields[2]),
        periodicity,
    )


def _parse_improper(line: str) -> GaffTorsionTerm:
    match = _TORSION_RE.match(line)
    if match is None:
        raise ValueError(f"invalid GAFF improper record: {line!r}")
    fields = _tail(line, match)
    return GaffTorsionTerm(
        (match[1], match[2], match[3], match[4]),
        1.0,
        float(fields[0]),
        float(fields[1]),
        int(round(abs(float(fields[2])))),
    )


def _parse_vdw(line: str) -> GaffVdw:
    fields = line.split()
    if len(fields) < 3:
        raise ValueError(f"invalid GAFF van der Waals record: {line!r}")
    radius, epsilon = float(fields[1]), float(fields[2])
    if not (math.isfinite(radius) and radius > 0.0 and math.isfinite(epsilon) and epsilon > 0.0):
        raise ValueError(f"nonphysical GAFF van der Waals record: {line!r}")
    return GaffVdw(fields[0], radius, epsilon)




def _symmetric_parameter_index(items):
    result = {}
    for item in items:
        key = min(item.types, tuple(reversed(item.types)))
        result.setdefault(key, []).append(item)
    return {key: tuple(values) for key, values in result.items()}


def _indexed_unique_match(index, lookup_key, label: str, display_key):
    matches = index.get(lookup_key, ())
    if not matches:
        raise KeyError(f"GAFF {label} parameters are unavailable for {display_key}")
    if len(matches) != 1:
        raise ValueError(f"GAFF {label} lookup is ambiguous for {display_key}")
    return matches[0]


def _torsion_match_score(pattern: tuple[str, ...], key: tuple[str, ...]) -> int:
    best = -1
    for oriented in (key, tuple(reversed(key))):
        if all(expected == "X" or expected == actual for expected, actual in zip(pattern, oriented)):
            best = max(best, sum(expected != "X" for expected in pattern))
    return best


__all__ = [
    "GAFF_BUNDLED_COMMIT",
    "GAFF_BUNDLED_LICENSE",
    "GAFF_BUNDLED_SHA256",
    "GAFF_BUNDLED_SOURCE_URL",
    "GAFF_PARAMETER_ENV",
    "GAFF_PARAMETER_SET_SCHEMA",
    "GAFF_SUPPORTED_DOMAINS",
    "GaffAngle",
    "GaffBond",
    "GaffParameterSet",
    "GaffTorsionTerm",
    "GaffVdw",
    "assign_initial_gaff_types",
    "default_gaff_parameter_path",
]
