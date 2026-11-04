"""Topology-level periodicity and barrier seeds for frozen SONIC coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable

from matrix_chem.topology.elements import atomic_number
from matrix_core import section_content


CM_PER_KCAL_MOL = 349.755088918
SYNTHON_ZEFF_EQUIVALENCE_TOLERANCE = 0.035
TORSION_FAMILIES = frozenset({"TORSION", "PSEUDO_CYCLE_TORSION"})
IMPROPER_FAMILIES = frozenset({"IMPROPER_DIHEDRAL"})
PERIODIC_FAMILIES = TORSION_FAMILIES | IMPROPER_FAMILIES
SMITH_PERIODIC_CACHE_VERSION = "smith-periodic-estimates-v2"


@dataclass(frozen=True)
class PeriodicCoordinateEstimate:
    """A low-cost seed that must be replaced by a scan or force-field fit."""

    coordinate_identifier: str
    coordinate_name: str
    family: str
    periodicity: int
    barrier_kcal_mol: float
    barrier_cm1: float
    target: str
    coordinate_definition: str = "FROZEN_GIC"
    reference_value_radian: float | None = None
    reference_value_status: str = "DEFINED"
    coordinate_domain: str = "PERIODIC_2PI"
    symmetry_number: int = 1
    priority_atom: int | None = None
    central_bonds: tuple[tuple[int, int], ...] = ()
    ring_atoms: tuple[int, ...] = ()
    source_coordinates: tuple[str, ...] = ()
    periodicity_source: str = ""
    barrier_source: str = "SMITH_TOPOLOGY_HEURISTIC_V1"
    status: str = "SEED_REQUIRES_HESSIAN_OR_SCAN_VALIDATION"


@dataclass(frozen=True)
class _SynthonAtom:
    atomic_number: int
    zeff: float | None
    signature: str


@dataclass(frozen=True)
class RingPhaseCoordinate:
    """Canonical hyperspherical phase derived from native Fourier RPck rows."""

    identifier: str
    source_coordinates: tuple[str, ...]
    ring_atoms: tuple[int, ...]
    reference_value_radian: float | None
    reference_value_status: str
    coordinate_domain: str
    periodicity: int
    coordinate_definition: str
    priority_atom: int


def build_periodic_coordinate_estimates(
    definition: object,
    sectioned_lines: list[str],
    atom_symbols: tuple[str, ...],
    ring_phase_coordinates: tuple[RingPhaseCoordinate, ...] = (),
    coordinate_values_radian: dict[str, float] | None = None,
) -> tuple[PeriodicCoordinateEstimate, ...]:
    """Return one record per dihedral and per derived RPck phase coordinate."""

    topology = section_content(sectioned_lines, "TOPOLOGY")
    bonds = _topology_bonds(topology)
    bond_orders = _topology_bond_orders(topology)
    bond_components = _topology_bond_components(topology)
    aromatic_atoms = _topology_aromatic_atoms(topology)
    synthons = _synthon_atoms(section_content(sectioned_lines, "SYNTHONS"), atom_symbols)
    graph = _graph(bonds)
    primitive_by_id = {
        str(primitive.identifier): primitive for primitive in getattr(definition, "primitives", ())
    }
    records: list[PeriodicCoordinateEstimate] = []
    coordinate_values = coordinate_values_radian or {}
    for gic in getattr(definition, "gics", ()):
        family = str(gic.family).upper()
        if family not in PERIODIC_FAMILIES:
            continue
        components = tuple(getattr(gic, "coefficients", ())) or ((str(gic.primitive_id), 1.0),)
        primitives = tuple(
            (primitive_by_id.get(str(identifier)), abs(float(coefficient)))
            for identifier, coefficient in components
            if primitive_by_id.get(str(identifier)) is not None
        )
        if family in TORSION_FAMILIES:
            # Periodicity is a physical/topological property of a torsional
            # coordinate, independent of whether its local numerical chart is
            # represented on a principal branch or a continued branch.
            central_bonds = tuple(
                sorted(
                    {
                        tuple(sorted((int(primitive.atoms[1]), int(primitive.atoms[2]))))
                        for primitive, _weight in primitives
                        if len(tuple(primitive.atoms)) == 4
                    }
                )
            )
            periodicities = tuple(
                _torsion_periodicity(bond, graph=graph, synthons=synthons) for bond in central_bonds
            )
            periodicity = _lcm(periodicities)
            barriers = tuple(
                _torsion_barrier(bond, bond_orders=bond_orders, components=bond_components)
                for bond in central_bonds
            )
            barrier = _mean(barriers, default=3.0)
            records.append(
                _record(
                    str(gic.identifier),
                    str(gic.name),
                    family,
                    periodicity=periodicity,
                    barrier=barrier,
                    target="TORSION",
                    reference_value_radian=coordinate_values.get(str(gic.identifier)),
                    reference_value_status=(
                        "DEFINED"
                        if str(gic.identifier) in coordinate_values
                        else "UNDEFINED_SINGULAR_REFERENCE_COORDINATE"
                    ),
                    central_bonds=central_bonds,
                    periodicity_source="SUBSTITUENT_SYNTHON_EQUIVALENCE",
                )
            )
            continue
        if family in IMPROPER_FAMILIES:
            records.append(
                _record(
                    str(gic.identifier),
                    str(gic.name),
                    family,
                    periodicity=2,
                    barrier=15.0,
                    target="IMPROPER_DIHEDRAL",
                    reference_value_radian=coordinate_values.get(str(gic.identifier)),
                    reference_value_status=(
                        "DEFINED"
                        if str(gic.identifier) in coordinate_values
                        else "UNDEFINED_SINGULAR_REFERENCE_COORDINATE"
                    ),
                    periodicity_source="PLANARITY_TWOFOLD_SEED",
                )
            )
            continue

    for phase in ring_phase_coordinates:
        ring = phase.ring_atoms
        barriers = (
            _ring_barrier(
                ring,
                bond_orders=bond_orders,
                components=bond_components,
                aromatic_atoms=aromatic_atoms,
            ),
        )
        rotational_equivalence = _ring_periodicity(ring, synthons, graph=graph)
        symmetry_number = (
            max(1, rotational_equivalence // math.gcd(rotational_equivalence, 2))
            if phase.coordinate_domain == "PERIODIC_2PI"
            else 1
        )
        records.append(
            _record(
                phase.identifier,
                phase.identifier,
                "RING_PUCKERING_PHASE",
                periodicity=phase.periodicity,
                barrier=_mean(barriers, default=9.0),
                target="RING_PUCKERING_PHASE",
                coordinate_definition=phase.coordinate_definition,
                reference_value_radian=phase.reference_value_radian,
                reference_value_status=phase.reference_value_status,
                coordinate_domain=phase.coordinate_domain,
                symmetry_number=symmetry_number,
                priority_atom=phase.priority_atom,
                ring_atoms=ring,
                source_coordinates=phase.source_coordinates,
                periodicity_source="NATIVE_RPCK_FOURIER_MODE",
            )
        )
    return tuple(records)


def periodic_coordinate_estimate_line(record: PeriodicCoordinateEstimate) -> str:
    bonds = ",".join(f"{left}-{right}" for left, right in record.central_bonds) or "NONE"
    ring = ",".join(str(atom) for atom in record.ring_atoms) or "NONE"
    sources = ",".join(record.source_coordinates) or "NONE"
    reference = (
        "UNDEFINED"
        if record.reference_value_radian is None
        else f"{record.reference_value_radian:.17g}"
    )
    reference_degree = (
        "UNDEFINED"
        if record.reference_value_radian is None
        else f"{math.degrees(record.reference_value_radian):.17g}"
    )
    return (
        f"{record.coordinate_identifier} NAME={record.coordinate_name} FAMILY={record.family} "
        f"TARGET={record.target} DEFINITION={record.coordinate_definition} "
        f"PERIODICITY={record.periodicity} "
        f"BARRIER_KCAL_MOL={record.barrier_kcal_mol:.17g} "
        f"BARRIER_CM-1={record.barrier_cm1:.17g} CENTRAL_BONDS={bonds} RING_ATOMS={ring} "
        f"SOURCES={sources} "
        f"REFERENCE_VALUE_RADIAN={reference} VALUE_STATUS={record.reference_value_status} "
        f"REFERENCE_VALUE_DEGREE={reference_degree} "
        f"COORDINATE_DOMAIN={record.coordinate_domain} SYMMETRY_NUMBER={record.symmetry_number} "
        f"PRIORITY_ATOM={record.priority_atom if record.priority_atom is not None else 'NONE'} "
        f"PERIODICITY_SOURCE={record.periodicity_source} "
        f"BARRIER_SOURCE={record.barrier_source} STATUS={record.status}"
    )


@lru_cache(maxsize=4096)
def parse_periodic_coordinate_estimate(line: str) -> PeriodicCoordinateEstimate:
    parts = line.split()
    if not parts:
        raise ValueError("empty periodic-coordinate estimate")
    fields = dict(token.split("=", 1) for token in parts[1:] if "=" in token)
    try:
        bonds = tuple(
            tuple(int(atom) for atom in token.split("-"))
            for token in fields.get("CENTRAL_BONDS", "").split(",")
            if token and token.upper() != "NONE"
        )
        ring_atoms = tuple(
            int(atom)
            for atom in fields.get("RING_ATOMS", "").split(",")
            if atom and atom.upper() != "NONE"
        )
        return PeriodicCoordinateEstimate(
            coordinate_identifier=parts[0],
            coordinate_name=fields["NAME"],
            family=fields["FAMILY"],
            periodicity=int(fields["PERIODICITY"]),
            barrier_kcal_mol=float(fields["BARRIER_KCAL_MOL"]),
            barrier_cm1=float(fields["BARRIER_CM-1"]),
            target=fields["TARGET"],
            coordinate_definition=fields.get("DEFINITION", "FROZEN_GIC"),
            reference_value_radian=(
                None
                if fields.get("REFERENCE_VALUE_RADIAN", "UNDEFINED").upper() == "UNDEFINED"
                else float(fields["REFERENCE_VALUE_RADIAN"])
            ),
            reference_value_status=fields.get("VALUE_STATUS", "DEFINED"),
            coordinate_domain=fields.get("COORDINATE_DOMAIN", "PERIODIC_2PI"),
            symmetry_number=int(fields.get("SYMMETRY_NUMBER", "1")),
            priority_atom=(
                None
                if fields.get("PRIORITY_ATOM", "NONE").upper() == "NONE"
                else int(fields["PRIORITY_ATOM"])
            ),
            central_bonds=bonds,
            ring_atoms=ring_atoms,
            source_coordinates=tuple(
                token
                for token in fields.get("SOURCES", "").split(",")
                if token and token.upper() != "NONE"
            ),
            periodicity_source=fields["PERIODICITY_SOURCE"],
            barrier_source=fields["BARRIER_SOURCE"],
            status=fields["STATUS"],
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid periodic-coordinate estimate: {line}") from exc


def invalidate_periodic_coordinate_cache() -> None:
    """Invalidate parsed SONIC periodic estimates after model changes."""

    parse_periodic_coordinate_estimate.cache_clear()


def _record(
    coordinate_identifier: str,
    coordinate_name: str,
    family: str,
    *,
    periodicity: int,
    barrier: float,
    target: str,
    periodicity_source: str,
    coordinate_definition: str = "FROZEN_GIC",
    reference_value_radian: float | None = None,
    reference_value_status: str = "DEFINED",
    coordinate_domain: str = "PERIODIC_2PI",
    symmetry_number: int = 1,
    priority_atom: int | None = None,
    central_bonds: tuple[tuple[int, int], ...] = (),
    ring_atoms: tuple[int, ...] = (),
    source_coordinates: tuple[str, ...] = (),
) -> PeriodicCoordinateEstimate:
    value = max(float(barrier), 0.0)
    return PeriodicCoordinateEstimate(
        coordinate_identifier=coordinate_identifier,
        coordinate_name=coordinate_name,
        family=family,
        periodicity=max(int(periodicity), 1),
        barrier_kcal_mol=value,
        barrier_cm1=value * CM_PER_KCAL_MOL,
        target=target,
        coordinate_definition=coordinate_definition,
        reference_value_radian=reference_value_radian,
        reference_value_status=reference_value_status,
        coordinate_domain=coordinate_domain,
        symmetry_number=max(1, int(symmetry_number)),
        priority_atom=priority_atom,
        central_bonds=central_bonds,
        ring_atoms=ring_atoms,
        source_coordinates=source_coordinates,
        periodicity_source=periodicity_source,
    )


def _torsion_periodicity(
    central_bond: tuple[int, int],
    *,
    graph: dict[int, set[int]],
    synthons: dict[int, _SynthonAtom],
) -> int:
    left, right = central_bond
    # An equivalent terminal XY3 group defines a physical C3 identity
    # operation by itself: a 120-degree rotation only permutes the three Y
    # atoms,
    # irrespective of accidental or local pseudo-equivalence on the other end
    # of the bond. Taking an LCM with that other end can incorrectly promote
    # the rotor to periodicity six and split equivalent phase minima.
    if _is_equivalent_xy3_end(
        left, right, graph=graph, synthons=synthons
    ) or _is_equivalent_xy3_end(
        right, left, graph=graph, synthons=synthons
    ):
        return 3
    if _is_equivalent_xy2_end(
        left, right, graph=graph, synthons=synthons
    ) or _is_equivalent_xy2_end(
        right, left, graph=graph, synthons=synthons
    ):
        return 2
    periods = []
    for center, other in ((left, right), (right, left)):
        substituents = sorted(atom for atom in graph.get(center, set()) if atom != other)
        clusters: list[list[int]] = []
        for atom in substituents:
            for cluster in clusters:
                if _equivalent(atom, cluster[0], synthons):
                    cluster.append(atom)
                    break
            else:
                clusters.append([atom])
        periods.append(max((len(cluster) for cluster in clusters), default=1))
    return max(1, math.lcm(*periods))


def _is_equivalent_xy3_end(
    center: int,
    other: int,
    *,
    graph: dict[int, set[int]],
    synthons: dict[int, _SynthonAtom],
) -> bool:
    substituents = [atom for atom in graph.get(center, set()) if atom != other]
    if len(substituents) != 3 or any(synthons.get(atom) is None for atom in substituents):
        return False
    if (
        len({synthons[atom].atomic_number for atom in substituents}) == 1
        and all(len(graph.get(atom, set())) == 1 for atom in substituents)
    ):
        # Terminal identical atoms are exactly permutable even when continuous
        # perception assigns slightly different environment-sensitive Zeff
        # values in a distorted reference geometry.
        return True
    threshold = 5.0e-4
    keys = {
        (
            synthons[atom].atomic_number,
            (
                f"ZEFF:{round(float(synthons[atom].zeff) / threshold)}"
                if synthons[atom].zeff is not None
                else f"SIG:{synthons[atom].signature}"
            ),
        )
        for atom in substituents
    }
    return len(keys) == 1


def _is_equivalent_xy2_end(
    center: int,
    other: int,
    *,
    graph: dict[int, set[int]],
    synthons: dict[int, _SynthonAtom],
) -> bool:
    substituents = [atom for atom in graph.get(center, set()) if atom != other]
    if len(substituents) != 2 or any(synthons.get(atom) is None for atom in substituents):
        return False
    if (
        synthons[substituents[0]].atomic_number
        == synthons[substituents[1]].atomic_number
        and len(graph.get(substituents[0], set())) == 1
        and len(graph.get(substituents[1], set())) == 1
    ):
        return True
    return _equivalent(substituents[0], substituents[1], synthons)


def _ring_periodicity(
    ring: tuple[int, ...],
    synthons: dict[int, _SynthonAtom],
    *,
    graph: dict[int, set[int]],
) -> int:
    size = len(ring)
    if size < 2:
        return 1
    colors = _topological_atom_colors(graph, synthons, ring_atoms=frozenset(ring))
    return max(
        1,
        sum(
            1
            for shift in range(size)
            if all(
                colors.get(ring[index]) == colors.get(ring[(index + shift) % size])
                and _equivalent(ring[index], ring[(index + shift) % size], synthons)
                for index in range(size)
            )
        ),
    )


def _topological_atom_colors(
    graph: dict[int, set[int]],
    synthons: dict[int, _SynthonAtom],
    *,
    ring_atoms: frozenset[int],
) -> dict[int, int]:
    """Refine atom colors so exocyclic substitution decorates the ring."""

    atoms = sorted(set(graph) | set(synthons))
    initial = {
        atom: (
            synthons[atom].atomic_number if atom in synthons else 0,
            atom in ring_atoms,
        )
        for atom in atoms
    }
    palette = {label: index for index, label in enumerate(sorted(set(initial.values())))}
    colors = {atom: palette[label] for atom, label in initial.items()}
    for _ in atoms:
        labels = {
            atom: (colors[atom], tuple(sorted(colors[other] for other in graph.get(atom, ()))))
            for atom in atoms
        }
        palette = {label: index for index, label in enumerate(sorted(set(labels.values())))}
        refined = {atom: palette[label] for atom, label in labels.items()}
        if all(
            (colors[left] == colors[right]) == (refined[left] == refined[right])
            for left in atoms
            for right in atoms
        ):
            return refined
        colors = refined
    return colors


def _equivalent(left: int, right: int, synthons: dict[int, _SynthonAtom]) -> bool:
    a = synthons.get(left)
    b = synthons.get(right)
    if a is None or b is None or a.atomic_number != b.atomic_number:
        return False
    if a.zeff is not None and b.zeff is not None:
        return abs(a.zeff - b.zeff) <= SYNTHON_ZEFF_EQUIVALENCE_TOLERANCE
    return a.signature == b.signature


def _torsion_barrier(
    bond: tuple[int, int],
    *,
    bond_orders: dict[tuple[int, int], float],
    components: dict[tuple[int, int], tuple[float, float, float]],
) -> float:
    sigma_pi = components.get(bond)
    pi_index = None if sigma_pi is None else float(sigma_pi[1] + sigma_pi[2])
    if pi_index is None:
        order = bond_orders.get(bond)
        pi_index = 0.0 if order is None else max(0.0, float(order) - 1.0)
    return min(65.0, 3.0 + 45.0 * max(0.0, pi_index))


def _ring_barrier(
    ring: tuple[int, ...],
    *,
    bond_orders: dict[tuple[int, int], float],
    components: dict[tuple[int, int], tuple[float, float, float]],
    aromatic_atoms: set[int],
) -> float:
    ring_bonds = tuple(
        tuple(sorted((left, right))) for left, right in zip(ring, ring[1:] + ring[:1])
    )
    pi_values = []
    for bond in ring_bonds:
        values = components.get(bond)
        if values is not None:
            pi_values.append(max(0.0, float(values[1] + values[2])))
        else:
            pi_values.append(max(0.0, float(bond_orders.get(bond, 1.0)) - 1.0))
    if ring and set(ring).issubset(aromatic_atoms) and not any(pi_values):
        pi_values = [0.5 for _bond in ring_bonds]
    size = len(ring)
    strain_seed = float(max(0, 6 - size) ** 2)
    return min(65.0, 1.5 * float(size) + strain_seed + 30.0 * _mean(pi_values, default=0.0))




def _synthon_atoms(lines: list[str], symbols: tuple[str, ...]) -> dict[int, _SynthonAtom]:
    columns: tuple[str, ...] = ()
    result: dict[int, _SynthonAtom] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].upper() == "COLUMNS":
            columns = tuple(part.upper() for part in parts[1:])
            continue
        if not columns or not parts[0].isdigit() or len(parts) < len(columns):
            continue
        row = dict(zip(columns, parts, strict=False))
        index = int(row["ATOM"])
        symbol = symbols[index - 1] if 1 <= index <= len(symbols) else row.get("Z", "")
        try:
            zeff = float(row["ZEFF"])
        except (KeyError, ValueError):
            zeff = None
        result[index] = _SynthonAtom(
            atomic_number=int(atomic_number(symbol)),
            zeff=zeff,
            signature=row.get("SIGNATURE", symbol),
        )
    for index, symbol in enumerate(symbols, start=1):
        result.setdefault(index, _SynthonAtom(int(atomic_number(symbol)), None, symbol))
    return result


def _topology_bonds(lines: list[str]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                tuple(sorted((int(parts[0]), int(parts[1]))))
                for line in _subsection(lines, "BONDS")
                if len(parts := line.split()) >= 2 and parts[0].isdigit() and parts[1].isdigit()
            }
        )
    )


def _topology_bond_orders(lines: list[str]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for line in _subsection(lines, "BOND_ORDERS"):
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            result[tuple(sorted((int(parts[0]), int(parts[1]))))] = float(parts[2])
    return result


def _topology_bond_components(
    lines: list[str],
) -> dict[tuple[int, int], tuple[float, float, float]]:
    result: dict[tuple[int, int], tuple[float, float, float]] = {}
    for line in _subsection(lines, "BOND_ORDER_COMPONENTS"):
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            result[tuple(sorted((int(parts[0]), int(parts[1]))))] = tuple(
                float(value) for value in parts[-3:]
            )
    return result


def _topology_rings(lines: list[str]) -> tuple[tuple[int, ...], ...]:
    result = []
    for line in _subsection(lines, "RINGS"):
        parts = line.replace(",", " ").split()
        atoms = []
        active = False
        for token in parts:
            if token.upper().startswith("ATOMS="):
                active = True
                token = token.split("=", 1)[1]
            elif active and "=" in token:
                break
            if active and token:
                atoms.append(int(token))
        if len(atoms) >= 3:
            result.append(tuple(atoms))
    return tuple(result)


def _topology_aromatic_atoms(lines: list[str]) -> set[int]:
    for line in _subsection(lines, "AROMATICITY"):
        parts = line.replace(",", " ").split()
        if parts and parts[0].upper() == "ATOMS":
            return {int(token) for token in parts[1:] if token.isdigit()}
    return set()


def _subsection(lines: list[str], name: str) -> list[str]:
    marker = f"[{name.upper()}]"
    active = False
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            active = stripped.upper() == marker
            continue
        if active:
            result.append(stripped)
    return result


def _graph(bonds: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for left, right in bonds:
        result.setdefault(left, set()).add(right)
        result.setdefault(right, set()).add(left)
    return result


def _lcm(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = math.lcm(result, max(1, int(value)))
    return result


def _mean(values: Iterable[float], *, default: float) -> float:
    items = tuple(float(value) for value in values)
    return default if not items else sum(items) / float(len(items))
