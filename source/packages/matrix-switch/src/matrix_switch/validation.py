"""General SWITCH seed validation before a QM job is allowed to start."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from matrix_chem.geometry import MolecularGeometry
from matrix_chem.topology.elements import atomic_number

from .model import SwitchMolecularGraph


_MAX_VALENCE = {
    "H": 1.0,
    "B": 4.0,
    "C": 4.0,
    "N": 4.0,
    "O": 2.0,
    "F": 1.0,
    "P": 6.0,
    "S": 6.0,
    "Cl": 1.0,
    "Br": 1.0,
    "I": 1.0,
}

_HYDROGEN_COVALENT_MAX = {
    "B": 1.35,
    "C": 1.35,
    "N": 1.25,
    "O": 1.20,
    "F": 1.15,
    "P": 1.75,
    "S": 1.75,
}


@dataclass(frozen=True)
class HbondCandidate:
    donor: int
    hydrogen: int
    acceptor: int
    distance_angstrom: float
    angle_deg: float


@dataclass(frozen=True)
class SwitchValidation:
    formula: str
    hydrogen_bonds: tuple[HbondCandidate, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def bifurcated_hydrogen_bonds(self) -> tuple[tuple[HbondCandidate, ...], ...]:
        """Return donor/acceptor pairs with more than one H-bonding H.

        This preserves both contacts of motifs such as NH2--O: the two
        hydrogens are not collapsed into one donor flag.
        """

        grouped: dict[tuple[int, int], list[HbondCandidate]] = {}
        for candidate in self.hydrogen_bonds:
            grouped.setdefault((candidate.donor, candidate.acceptor), []).append(candidate)
        return tuple(
            tuple(items)
            for items in grouped.values()
            if len(items) > 1
        )

    @property
    def has_bifurcated_hydrogen_bond(self) -> bool:
        return bool(self.bifurcated_hydrogen_bonds)


@dataclass(frozen=True)
class SwitchSeedComparison:
    valid: bool
    errors: tuple[str, ...]
    reference_formula: str
    candidate_formula: str


@dataclass(frozen=True)
class GeometryFileValidation:
    formula: str
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_geometry_file(geometry: MolecularGeometry) -> GeometryFileValidation:
    """Audit an existing XYZ/XY ZIN before it enters a SWITCH/QM workflow."""
    errors: list[str] = []
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(geometry.atoms), 3):
        errors.append("coordinate shape does not match atom count")
    elif not np.all(np.isfinite(coordinates)):
        errors.append("coordinates contain NaN or infinity")
    symbols = tuple(str(atom) for atom in geometry.atoms)
    for symbol in symbols:
        if atomic_number(symbol) is None:
            errors.append(f"unknown element symbol: {symbol}")
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    formula = "".join(
        symbol + (str(count) if count != 1 else "")
        for symbol, count in sorted(counts.items(), key=lambda item: (item[0] != "C", item[0] != "H", item[0]))
    )
    if coordinates.shape == (len(symbols), 3):
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if float(np.min(distances, initial=np.inf)) < 0.35:
            errors.append("two atoms are unrealistically coincident")
        heavy = np.array([symbol != "H" for symbol in symbols], dtype=bool)
        for index, symbol in enumerate(symbols):
            if symbol != "H" or not np.any(heavy):
                continue
            heavy_indices = np.flatnonzero(heavy)
            nearest_index = int(heavy_indices[np.argmin(distances[index, heavy])])
            nearest = float(distances[index, nearest_index])
            maximum = _HYDROGEN_COVALENT_MAX.get(symbols[nearest_index], 1.35)
            if not 0.55 <= nearest <= maximum:
                errors.append(f"hydrogen {index + 1} has no covalent heavy-atom neighbor")
    return GeometryFileValidation(formula, tuple(errors))


def compare_switch_graphs(
    reference: SwitchMolecularGraph,
    candidate: SwitchMolecularGraph,
) -> SwitchSeedComparison:
    """Compare SWITCH and seed constitution without RDKit or atom reordering."""
    errors: list[str] = []
    reference_formula = _graph_formula(reference)
    candidate_formula = _graph_formula(candidate)
    if reference_formula != candidate_formula:
        errors.append(f"formula mismatch: {candidate_formula} != {reference_formula}")
    if tuple(atom.symbol for atom in reference.atoms) != tuple(atom.symbol for atom in candidate.atoms):
        errors.append("atom order or element sequence differs")
    reference_bonds = {(bond.key, float(bond.order)) for bond in reference.bonds}
    candidate_bonds = {(bond.key, float(bond.order)) for bond in candidate.bonds}
    if reference_bonds != candidate_bonds:
        errors.append("bond topology or bond order differs")
    if reference.total_formal_charge != candidate.total_formal_charge:
        errors.append("formal charge differs")
    return SwitchSeedComparison(not errors, tuple(errors), reference_formula, candidate_formula)


def _graph_formula(graph: SwitchMolecularGraph) -> str:
    counts: dict[str, int] = {}
    for atom in graph.atoms:
        counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
        if atom.hydrogen_count:
            counts["H"] = counts.get("H", 0) + int(atom.hydrogen_count)
    return "".join(
        symbol + (str(count) if count != 1 else "")
        for symbol, count in sorted(counts.items(), key=lambda item: (item[0] != "C", item[0] != "H", item[0]))
    )


def validate_switch_geometry(
    graph: SwitchMolecularGraph,
    geometry: MolecularGeometry,
    *,
    expected_formula: str | None = None,
    hbond_distance_angstrom: float = 2.8,
) -> SwitchValidation:
    """Validate constitution, coordinates, valence and obvious H-bonds.

    The graph atom order is immutable: completed hydrogens may only be appended
    after the graph atoms.  This catches stale acetyl/methyl templates and
    hydrogen reordering before SWITCH output is passed to Oracle.
    """
    errors: list[str] = []
    warnings: list[str] = []
    atoms = tuple(str(atom) for atom in geometry.atoms)
    graph_atoms = tuple(atom.symbol for atom in graph.atoms)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(atoms), 3):
        errors.append(f"coordinates shape {coordinates.shape} does not match {len(atoms)} atoms")
    elif not np.all(np.isfinite(coordinates)):
        errors.append("coordinates contain NaN or infinity")
    if atoms[: len(graph_atoms)] != graph_atoms:
        errors.append("geometry does not preserve SWITCH graph atom order")
    if any(symbol != "H" for symbol in atoms[len(graph_atoms) :]):
        errors.append("atoms appended after the SWITCH graph must all be hydrogens")
    if int(geometry.charge or 0) != int(graph.total_formal_charge):
        errors.append("formal charge differs from the SWITCH graph")

    counts: dict[str, int] = {}
    for symbol in atoms:
        counts[symbol] = counts.get(symbol, 0) + 1
    formula = "".join(
        symbol + (str(count) if count != 1 else "")
        for symbol, count in sorted(counts.items(), key=lambda item: (item[0] != "C", item[0] != "H", item[0]))
    )
    if expected_formula is not None and formula != expected_formula:
        errors.append(f"formula {formula} does not match expected {expected_formula}")

    for index, atom in enumerate(graph.atoms):
        if atom.hydrogen_count is None:
            continue
        valence = sum(bond.order for bond in graph.bonds if index in bond.key) + atom.hydrogen_count
        maximum = _MAX_VALENCE.get(atom.symbol)
        if maximum is not None and valence > maximum + 1.0e-8:
            errors.append(f"atom {index + 1} ({atom.symbol}) exceeds valence {maximum:g}")

    hbonds: list[HbondCandidate] = []
    if coordinates.shape == (len(atoms), 3):
        bonded = {tuple(sorted(bond.key)) for bond in graph.bonds}
        for hydrogen in range(len(graph_atoms), len(atoms)):
            near_donors = [
                donor for donor, symbol in enumerate(atoms[: len(graph_atoms)])
                if symbol in {"N", "O", "S"}
                and float(np.linalg.norm(coordinates[hydrogen] - coordinates[donor])) < 1.25
            ]
            for donor in near_donors:
                for acceptor, symbol in enumerate(atoms[: len(graph_atoms)]):
                    if acceptor == donor or symbol not in {"N", "O", "S"}:
                        continue
                    distance = float(np.linalg.norm(coordinates[hydrogen] - coordinates[acceptor]))
                    if distance > hbond_distance_angstrom or tuple(sorted((donor, acceptor))) in bonded:
                        continue
                    donor_vector = coordinates[donor] - coordinates[hydrogen]
                    acceptor_vector = coordinates[acceptor] - coordinates[hydrogen]
                    cosine = float(np.dot(donor_vector, acceptor_vector)) / (
                        np.linalg.norm(donor_vector) * np.linalg.norm(acceptor_vector)
                    )
                    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                    if angle >= 100.0:
                        hbonds.append(HbondCandidate(donor, hydrogen, acceptor, distance, angle))
    if not hbonds:
        warnings.append("no intramolecular hydrogen-bond candidate detected")
    return SwitchValidation(formula, tuple(hbonds), tuple(errors), tuple(warnings))


__all__ = [
    "GeometryFileValidation",
    "HbondCandidate",
    "SwitchSeedComparison",
    "SwitchValidation",
    "compare_switch_graphs",
    "validate_switch_geometry",
    "validate_geometry_file",
]
