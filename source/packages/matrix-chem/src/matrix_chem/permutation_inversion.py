"""ORACLE's finite permutation--inversion contract for cyclic internal rotors.

This is a kinematic subgroup, not a barrier or tunnelling-feasibility model.
Only geometrically certified rotations about single bridge bonds augment the
retained point group. Graph automorphisms alone are not admitted as motions.
Atom indices are zero based. A permutation maps old labels to new labels;
``(p, s) X = (-1)**s X[argsort(p)]``. Products apply the right operand first.
No body-fixed Cartesian matrix is attached to a pure internal permutation:
the compensating rotation depends on the embedding of the LAM coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from operator import index
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content

from .geometry import MolecularGeometry
from .symmetry import MolecularSymmetry


ORACLE_PI_SCHEMA = "matrix.oracle.permutation_inversion.v1"
ORACLE_PI_SECTION = "PERMUTATION_INVERSION"
_SCOPE = "KINEMATIC_RIGID_CYCLIC_BRIDGE_ROTORS"
_PATH_SCOPE = "KINEMATIC_SCAN_CERTIFIED_CONFINED_PATH"
_POINT_SCOPE = "KINEMATIC_POINT_OPERATIONS"
_GRID_SCOPE = "KINEMATIC_GEOMETRY_GRID_COMPLETE"


class PIExpansionLimit(ValueError):
    """The generators remain exact, but explicit closure exceeds the budget."""


@dataclass(frozen=True)
class PermutationInversionOperation:
    permutation: tuple[int, ...]
    inverted: bool = False

    def __post_init__(self) -> None:
        p = tuple(self.permutation)
        if (
            not p or any(type(i) is not int for i in p)
            or sorted(p) != list(range(len(p)))
            or type(self.inverted) is not bool
        ):
            raise ValueError("PI requires a zero-based atom permutation and a boolean inversion")
        object.__setattr__(self, "permutation", p)

    def compose(self, right: PermutationInversionOperation) -> PermutationInversionOperation:
        """Return self after right, including XOR of the inversion bits."""
        if len(self.permutation) != len(right.permutation):
            raise ValueError("cannot compose PI operations on different atom sets")
        return PermutationInversionOperation(
            tuple(self.permutation[i] for i in right.permutation),
            self.inverted != right.inverted,
        )

    def apply(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        """Apply a PI operation in the space-fixed frame (no implicit alignment)."""
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape[-2:] != (len(self.permutation), 3):
            raise ValueError("PI coordinates must end in (natoms, 3)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("PI coordinates must be finite")
        return (-1 if self.inverted else 1) * coordinates[..., np.argsort(self.permutation), :]


@dataclass(frozen=True)
class InternalRotorPermutation:
    """A positive rotation about fixed -> moving atom, with a frozen top."""

    axis: tuple[int, int]
    top_atoms: tuple[int, ...]
    order: int
    operation: PermutationInversionOperation
    maximum_deviation_angstrom: float
    certification: str = "rigid_geometry"
    scan_sha256: str | None = None

    @property
    def angle_radians(self) -> float:
        return float(2.0 * np.pi / self.order)


@dataclass(frozen=True)
class OraclePermutationInversion:
    """Finite group generated from point operations and certified rotor cycles.

    ``words[k]`` is a sequence of generator indices. Starting at identity,
    successively compose these generators on the right to recover operation k.
    This stores O(|G|) words instead of an O(|G|**2) multiplication table.
    If enumeration exceeds the budget, operations and words are both empty;
    the generating set still defines the exact group. No partial group is used.
    ``rotor_diagnostics`` records bridge directions with no certified cycle;
    absence of a cycle does not establish absence of internal rotation.
    """

    atoms: tuple[str, ...]
    nuclear_labels: tuple[str, ...]
    geometry_sha256: str
    point_group: str
    generators: tuple[PermutationInversionOperation, ...]
    generator_sources: tuple[str, ...]
    operations: tuple[PermutationInversionOperation, ...]
    words: tuple[tuple[int, ...], ...]
    rotors: tuple[InternalRotorPermutation, ...]
    rotor_diagnostics: tuple[str, ...]
    tolerance_angstrom: float
    labelled_bonds: tuple[tuple[int, int, str], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.atoms)
        if not n or len(self.nuclear_labels) != n or any(not x for x in self.nuclear_labels):
            raise ValueError("PI nuclear labels must identify every atom")
        if len(self.generator_sources) != len(self.generators):
            raise ValueError("PI generators must have provenance")
        if not np.isfinite(self.tolerance_angstrom) or self.tolerance_angstrom <= 0:
            raise ValueError("PI tolerance must be finite and positive")
        for op in (*self.generators, *self.operations):
            if len(op.permutation) != n or any(
                self.atoms[i] != self.atoms[j] or self.nuclear_labels[i] != self.nuclear_labels[j]
                for i, j in enumerate(op.permutation)
            ):
                raise ValueError("PI operation exchanges distinguishable nuclei")
        if self.operations:
            operations, words = _closure(n, self.generators, len(self.operations))
            if self.operations != operations or self.words != words:
                raise ValueError("PI operations/words do not equal the complete generated closure")
        elif self.words:
            raise ValueError("unexpanded PI contracts cannot contain partial closure words")
        if any(len(edge) != 3 or type(edge[0]) is not int or type(edge[1]) is not int
               or not 0 <= edge[0] < edge[1] < n or edge[2] not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
               for edge in self.labelled_bonds):
            raise ValueError("invalid frozen PI labelled bond graph")
        if len({edge[:2] for edge in self.labelled_bonds}) != len(self.labelled_bonds):
            raise ValueError("duplicate PI graph edges")
        for rotor in self.rotors:
            _validate_rotor(rotor, n)
            if rotor.operation not in (*self.operations, *self.generators):
                raise ValueError("rotor operation is absent from the PI group")
            if rotor.maximum_deviation_angstrom > self.tolerance_angstrom:
                raise ValueError("rotor residual exceeds the certification tolerance")

    @property
    def order(self) -> int | None:
        """Explicit order, or None when only the exact generators are stored."""
        return len(self.operations) if self.operations else None

    def expand(self, *, max_operations: int = 4096):
        """Return operations and words; never return a truncated group."""
        if type(max_operations) is not int or max_operations < 1:
            raise ValueError("max_operations must be a positive integer")
        return _closure(len(self.atoms), self.generators, max_operations)

    def multiplication_table(self) -> tuple[tuple[int, ...], ...]:
        """Materialize the exact discrete table only when a consumer needs it."""
        if not self.operations:
            raise PIExpansionLimit("PI closure is unexpanded; request expand with an explicit budget")
        index = {op: i for i, op in enumerate(self.operations)}
        return tuple(
            tuple(index[left.compose(right)] for right in self.operations)
            for left in self.operations
        )

    def to_dict(self) -> dict:
        """Portable shared contract; independent of a path or a consumer tool."""
        scope = (_GRID_SCOPE if any(source.startswith("GRID:") for source in self.generator_sources)
                 else _PATH_SCOPE if any(source.startswith("PATH:") for source in self.generator_sources)
                 else _POINT_SCOPE if any(source.startswith("POINT_ONLY:") for source in self.generator_sources)
                 else "KINEMATIC_SCAN_CERTIFIED_BRIDGE_ROTORS" if any(r.certification == "scan_translation" for r in self.rotors)
                 else _SCOPE)
        return {"schema": ORACLE_PI_SCHEMA, "scope": scope, "data": asdict(self)}

    def nuclear_masses_amu(self) -> np.ndarray:
        """Resolve canonical element:mass-number labels using ORACLE isotope data.

        Opaque user labels are allowed in the group contract, but require the
        consumer to supply masses explicitly to the geometry-grid API.
        """
        from .isotopes_table import get_isotope
        from .topology.elements import atomic_number

        masses = []
        for atom, label in zip(self.atoms, self.nuclear_labels, strict=True):
            element, separator, mass_number = label.partition(":")
            isotope = get_isotope(atomic_number(atom) or 0, int(mass_number)) if separator and mass_number.isdigit() and element == atom else None
            if isotope is None:
                raise ValueError("automatic isotope masses require ORACLE element:mass-number nuclear labels")
            masses.append(isotope.mass)
        return np.asarray(masses, dtype=float)

    @classmethod
    def from_dict(cls, payload: Mapping) -> OraclePermutationInversion:
        if payload.get("schema") != ORACLE_PI_SCHEMA or payload.get("scope") not in {_SCOPE, "KINEMATIC_SCAN_CERTIFIED_BRIDGE_ROTORS", _PATH_SCOPE, _POINT_SCOPE, _GRID_SCOPE}:
            raise ValueError("unsupported shared ORACLE PI contract")
        return _contract_from_data(payload["data"])


def _validate_rotor(rotor: InternalRotorPermutation, natoms: int) -> None:
    if (
        len(rotor.axis) != 2 or len(set(rotor.axis)) != 2
        or any(type(i) is not int or i < 0 or i >= natoms for i in rotor.axis)
        or not rotor.top_atoms or len(set(rotor.top_atoms)) != len(rotor.top_atoms)
        or any(type(i) is not int or i < 0 or i >= natoms for i in rotor.top_atoms)
        or rotor.axis[0] in rotor.top_atoms or rotor.axis[1] not in rotor.top_atoms
        or type(rotor.order) is not int or rotor.order < 2 or rotor.order > len(rotor.top_atoms)
        or len(rotor.operation.permutation) != natoms
        or rotor.operation.inverted
        or not np.isfinite(rotor.maximum_deviation_angstrom)
        or rotor.maximum_deviation_angstrom < 0
        or rotor.certification not in {"rigid_geometry", "scan_translation"}
        or (rotor.certification == "scan_translation" and (
            not isinstance(rotor.scan_sha256, str) or len(rotor.scan_sha256) != 64
            or any(c not in "0123456789abcdef" for c in rotor.scan_sha256)))
    ):
        raise ValueError("invalid certified internal rotor")
    identity = PermutationInversionOperation(tuple(range(natoms)))
    op = identity
    for power in range(1, rotor.order + 1):
        op = op.compose(rotor.operation)
        if (op == identity) != (power == rotor.order):
            raise ValueError("rotor permutation has an incorrect order")
    if any(
        rotor.operation.permutation[i] != i
        for i in range(natoms) if i not in rotor.top_atoms or i in rotor.axis
    ):
        raise ValueError("rotor cycle must fix the frame and the bond axis")


def _closure(natoms, generators, max_operations):
    identity = PermutationInversionOperation(tuple(range(natoms)))
    operations, words, seen = [identity], [()], {identity}
    for i, left in enumerate(operations):
        for j, right in enumerate(generators):
            product = left.compose(right)
            if product in seen:
                continue
            if len(operations) >= max_operations:
                raise PIExpansionLimit(f"PI group exceeds the explicit {max_operations}-operation limit")
            seen.add(product)
            operations.append(product)
            words.append((*words[i], j))
    return tuple(operations), tuple(words)


def _geometry_sha256(geometry):
    payload = [list(geometry.atoms), np.asarray(geometry.coordinates_angstrom).tolist()]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def permutation_inversion_from_point_symmetry(
    geometry: MolecularGeometry,
    symmetry: MolecularSymmetry,
    *,
    nuclear_labels: Sequence[str],
    tolerance_angstrom: float = 1e-5,
    max_operations: int = 4096,
) -> OraclePermutationInversion:
    """Freeze ORACLE-detected point operations as a shared PI contract.

    This has no bond-graph or cyclic-rotor assumption. A later geometry-grid
    certification determines whether each operation fixes or moves a LAM
    point and derives its body-frame compensation.
    """
    labels = tuple(nuclear_labels)
    n = geometry.natoms
    if len(labels) != n or any(not isinstance(value, str) or not value for value in labels):
        raise ValueError("one nuclear label is required for every point-symmetry atom")
    if not np.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0:
        raise ValueError("point PI tolerance must be finite and positive")
    if type(max_operations) is not int or max_operations < 1:
        raise ValueError("max_operations must be a positive integer")
    identity = PermutationInversionOperation(tuple(range(n)))
    generators, sources = [], []
    for point in symmetry.operations:
        pullback = np.asarray(point.permutation, dtype=int) - 1
        rotation = np.asarray(point.rotation, dtype=float)
        if (pullback.shape != (n,) or sorted(pullback.tolist()) != list(range(n))
                or rotation.shape != (3, 3)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)):
            raise ValueError("invalid ORACLE point operation")
        p = tuple(int(i) for i in np.argsort(pullback))
        if any(geometry.atoms[i] != geometry.atoms[j] or labels[i] != labels[j]
               for i, j in enumerate(p)):
            continue
        operation = PermutationInversionOperation(p, bool(np.linalg.det(rotation) < 0))
        if operation != identity and operation not in generators:
            generators.append(operation)
            sources.append(f"POINT_ONLY:{point.label}")
    operations, words = _closure(n, tuple(generators), max_operations)
    return OraclePermutationInversion(
        atoms=tuple(geometry.atoms), nuclear_labels=labels,
        geometry_sha256=_geometry_sha256(geometry), point_group=symmetry.point_group,
        generators=tuple(generators), generator_sources=tuple(sources),
        operations=operations, words=words, rotors=(), rotor_diagnostics=(),
        tolerance_angstrom=tolerance_angstrom,
    )


def build_permutation_inversion_symmetry(
    geometry: MolecularGeometry,
    symmetry: MolecularSymmetry,
    *,
    bonds: Sequence[tuple[int, int]],
    edge_labels: Mapping[tuple[int, int], str],
    nuclear_labels: Sequence[str] | None = None,
    tolerance_angstrom: float = 1.0e-5,
    max_operations: int = 4096,
) -> OraclePermutationInversion:
    """Extend ORACLE point operations with certified cyclic bridge rotations.

    Use the frozen ORACLE topology and SINGLE/DOUBLE/TRIPLE/AROMATIC edge
    labels. Both sides of every SINGLE graph bridge are examined. A geometric
    rotation must be bijective and preserve atom identities and every labelled
    graph edge. Distorted tops are reported as unconfirmed, never idealized.
    Labels default to ORACLE's standard isotope for each element. Supply a
    distinct nuclear label for each substituted isotope of the active species.
    The barrier height and the electronic-state symmetry are outside this API.
    """
    from .isotopes_table import get_default_isotope
    from .topology.elements import atomic_number

    if not np.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0:
        raise ValueError("PI tolerance must be finite and positive")
    if type(max_operations) is not int or max_operations < 1:
        raise ValueError("max_operations must be a positive integer")
    n = geometry.natoms
    if nuclear_labels is None:
        isotopes = [get_default_isotope(atomic_number(atom) or 0) for atom in geometry.atoms]
        labels = tuple(
            f"{atom}:{isotope.A if isotope is not None else 'unspecified'}"
            for atom, isotope in zip(geometry.atoms, isotopes, strict=True)
        )
    else:
        labels = tuple(nuclear_labels)
    if len(labels) != n or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("one nonempty nuclear label is required per atom")
    edges = {tuple(sorted(index(i) for i in pair)): value for pair, value in edge_labels.items()}
    bond_set = {tuple(sorted(index(i) for i in pair)) for pair in bonds}
    if set(edges) != bond_set or any(
        len(pair) != 2 or pair[0] < 0 or pair[1] >= n or pair[0] == pair[1]
        for pair in bond_set
    ):
        raise ValueError("PI requires the complete labelled ORACLE bond graph")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(bond_set)
    generators, sources, diagnostics, rotors = [], [], [], []
    identity = PermutationInversionOperation(tuple(range(n)))

    def preserves_graph(p):
        return all(
            geometry.atoms[i] == geometry.atoms[j] and labels[i] == labels[j]
            for i, j in enumerate(p)
        ) and all(edges.get(tuple(sorted((p[a], p[b])))) == kind for (a, b), kind in edges.items())

    def add_generator(op, source):
        if op != identity and op not in generators:
            generators.append(op)
            sources.append(source)

    for point_op in symmetry.operations:
        rotation = np.asarray(point_op.rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("nonorthogonal ORACLE point operation")
        # ORACLE stores x_i = R x_p(i). Inverting its atom map converts that
        # pullback to the old-label -> new-label convention of this contract.
        p = tuple(int(i) for i in np.argsort(np.asarray(point_op.permutation) - 1))
        if sorted(point_op.permutation) != list(range(1, n + 1)):
            raise ValueError("invalid ORACLE point-operation atom permutation")
        if not preserves_graph(p):
            diagnostics.append(f"POINT:{point_op.label}:EXCLUDED_NUCLEAR_OR_GRAPH_LABELS")
            continue
        add_generator(PermutationInversionOperation(p, bool(np.linalg.det(rotation) < 0)),
                      f"POINT:{point_op.label}")

    if n and nx.is_connected(graph):
        for a, b in sorted(tuple(sorted(pair)) for pair in nx.bridges(graph)):
            if edges[(a, b)] != "SINGLE":
                continue
            cut = graph.copy()
            cut.remove_edge(a, b)
            for fixed, moving in ((a, b), (b, a)):
                top = tuple(sorted(nx.node_connected_component(cut, moving)))
                rotor, reason = _certify_rotor(
                    geometry.coordinates_angstrom, labels, fixed, moving, top,
                    preserves_graph, tolerance_angstrom,
                )
                if rotor is None:
                    # Terminal X-H bonds and collinear fragments have no
                    # torsional coordinate; retain diagnostics for actual tops.
                    if reason != "NO_TORSIONAL_DEGREE_OF_FREEDOM":
                        diagnostics.append(f"AXIS:{fixed},{moving}:{reason}")
                    continue
                rotors.append(rotor)
                add_generator(rotor.operation, f"ROTOR:{fixed},{moving}:C{rotor.order}")
    else:
        diagnostics.append("DISCONNECTED_GRAPH:INTERNAL_ROTOR_SEARCH_NOT_APPLIED")
    try:
        operations, words = _closure(n, generators, max_operations)
    except PIExpansionLimit:
        # Many independent tops make full enumeration exponential. Preserve
        # the exact generating set and never serialize a truncated group.
        operations, words = (), ()
    return OraclePermutationInversion(
        tuple(geometry.atoms), labels, _geometry_sha256(geometry), symmetry.point_group,
        tuple(generators), tuple(sources), operations, words, tuple(rotors),
        tuple(diagnostics), float(tolerance_angstrom),
        tuple((a, b, kind) for (a, b), kind in sorted(edges.items())),
    )


def _certify_rotor(xyz, labels, fixed, moving, top, preserves_graph, tolerance):
    from scipy.spatial import cKDTree

    axis = xyz[moving] - xyz[fixed]
    length = np.linalg.norm(axis)
    if length <= tolerance:
        return None, "DEGENERATE_AXIS"
    axis = axis / length
    vectors = xyz - xyz[moving]
    heights = vectors @ axis
    radial = vectors - heights[:, None] * axis
    radii = np.linalg.norm(radial, axis=1)
    frame = tuple(i for i in range(len(xyz)) if i not in top)
    if max(radii[list(top)]) <= tolerance or max(radii[list(frame)]) <= tolerance:
        return None, "NO_TORSIONAL_DEGREE_OF_FREEDOM"
    anchor = max(top, key=lambda i: radii[i])
    angles = []
    for target in top:
        if target == anchor or labels[target] != labels[anchor]:
            continue
        if (
            abs(heights[target] - heights[anchor]) > tolerance
            or abs(radii[target] - radii[anchor]) > tolerance
        ):
            continue
        angle = np.arctan2(
            axis @ np.cross(radial[anchor], radial[target]), radial[anchor] @ radial[target]
        ) % (2 * np.pi)
        if angle > 1e-12:
            angles.append(float(angle))
    if not angles:
        return None, "NO_CERTIFIED_RIGID_CYCLE"
    # Spatial lookup is local to each nuclear species. This avoids a quadratic
    # atom assignment for every proposed rotation and detects ambiguous matches.
    by_label = {}
    for atom in top:
        by_label.setdefault(labels[atom], []).append(atom)
    searches = tuple((indices, cKDTree(xyz[indices])) for indices in by_label.values())
    tried_orders = set()
    for angle in sorted(angles):
        order = int(round(2 * np.pi / angle))
        if order < 2 or order > len(top) or order in tried_orders:
            continue
        tried_orders.add(order)
        # Certify the exact rational rotation, not a fitted angular correction.
        phi = 2 * np.pi / order
        rotated = (
            vectors * np.cos(phi) + np.cross(axis, vectors) * np.sin(phi)
            + heights[:, None] * axis * (1 - np.cos(phi)) + xyz[moving]
        )
        permutation = list(range(len(xyz)))
        residual = 0.0
        for indices, tree in searches:
            matches = tree.query_ball_point(rotated[indices], r=tolerance)
            if any(len(match) != 1 for match in matches):
                break
            targets = [indices[match[0]] for match in matches]
            for source, target in zip(indices, targets, strict=True):
                permutation[source] = target
            residual = max(residual, float(np.max(
                np.linalg.norm(xyz[targets] - rotated[indices], axis=1)
            )))
        else:
            p = tuple(permutation)
            if len(set(p)) != len(xyz) or not preserves_graph(p):
                continue
            rotor = InternalRotorPermutation(
                (fixed, moving), top, order, PermutationInversionOperation(p), residual
            )
            try:
                _validate_rotor(rotor, len(xyz))
            except ValueError:
                continue
            return rotor, "CERTIFIED"
    return None, "NO_CERTIFIED_RIGID_CYCLE"


def write_permutation_inversion_symmetry(path: Path | str, contract: OraclePermutationInversion) -> None:
    """Persist an ORACLE-owned PI contract beside (not inside) #SYMMETRY."""
    from .geometry_io import read_enriched_xyz

    if contract.geometry_sha256 != _geometry_sha256(read_enriched_xyz(Path(path))):
        raise ValueError("cannot write PI contract for a different Cartesian geometry")
    sections = read_sectioned_lines(Path(path))
    payload = json.dumps(asdict(contract), sort_keys=True, separators=(",", ":"))
    replace_section(Path(path), ORACLE_PI_SECTION, [
        f"SCHEMA {ORACLE_PI_SCHEMA}", "OWNER ORACLE", f"SCOPE {contract.to_dict()['scope']}",
        "INDEXING ZERO_BASED", "COMPOSITION LEFT_AFTER_RIGHT",
        f"ORDER {contract.order if contract.order is not None else 'UNEXPANDED'}",
        f"SYMMETRY_SHA256 {_section_sha256(sections, 'SYMMETRY')}",
        f"TOPOLOGY_SHA256 {_section_sha256(sections, 'TOPOLOGY')}",
        f"PAYLOAD_SHA256 {hashlib.sha256(payload.encode()).hexdigest()}", f"JSON {payload}",
    ])


def read_permutation_inversion_symmetry(path: Path | str) -> OraclePermutationInversion:
    """Read, validate closure, and reject a contract from a different geometry."""
    from .geometry_io import read_enriched_xyz

    sections = read_sectioned_lines(Path(path))
    content = section_content(sections, ORACLE_PI_SECTION)
    metadata = dict(line.split(" ", 1) for line in content if " " in line)
    if any(metadata.get(key) != value for key, value in {
        "SCHEMA": ORACLE_PI_SCHEMA, "OWNER": "ORACLE",
        "INDEXING": "ZERO_BASED", "COMPOSITION": "LEFT_AFTER_RIGHT",
    }.items()):
        raise ValueError("missing or unsupported ORACLE PI contract")
    payload = metadata.get("JSON", "")
    if hashlib.sha256(payload.encode()).hexdigest() != metadata.get("PAYLOAD_SHA256"):
        raise ValueError("ORACLE PI payload hash mismatch")
    for name in ("SYMMETRY", "TOPOLOGY"):
        if _section_sha256(sections, name) != metadata.get(f"{name}_SHA256"):
            raise ValueError(f"ORACLE PI contract is stale: #{name} has changed")
    contract = _contract_from_data(json.loads(payload))
    if metadata.get("SCOPE") != contract.to_dict()["scope"]:
        raise ValueError("ORACLE PI certification scope mismatch")
    if str(contract.order if contract.order is not None else 'UNEXPANDED') != metadata.get("ORDER"):
        raise ValueError("ORACLE PI order mismatch")
    if contract.geometry_sha256 != _geometry_sha256(read_enriched_xyz(Path(path))):
        raise ValueError("ORACLE PI contract is stale: Cartesian geometry has changed")
    return contract


def _contract_from_data(raw):

    def operation(item):
        return PermutationInversionOperation(tuple(item["permutation"]), item["inverted"])

    return OraclePermutationInversion(
        tuple(raw["atoms"]), tuple(raw["nuclear_labels"]), raw["geometry_sha256"],
        raw["point_group"], tuple(operation(item) for item in raw["generators"]),
        tuple(raw["generator_sources"]), tuple(operation(item) for item in raw["operations"]),
        tuple(tuple(word) for word in raw["words"]),
        tuple(InternalRotorPermutation(
            tuple(item["axis"]), tuple(item["top_atoms"]), item["order"],
            operation(item["operation"]), item["maximum_deviation_angstrom"],
            item.get("certification", "rigid_geometry"), item.get("scan_sha256"),
        ) for item in raw["rotors"]),
        tuple(raw["rotor_diagnostics"]), raw["tolerance_angstrom"],
        tuple(tuple(edge) for edge in raw.get("labelled_bonds", ())),
    )


def _section_sha256(sections, name):
    content = section_content(sections, name)
    if not content:
        raise ValueError(f"ORACLE PI contract requires a frozen #{name} section")
    return hashlib.sha256("\n".join(content).encode()).hexdigest()
