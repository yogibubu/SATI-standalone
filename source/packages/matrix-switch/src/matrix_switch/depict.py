"""Deterministic, toolkit-independent 2D molecular depiction."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from io import BytesIO
from itertools import combinations
from math import pi

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from matrix_chem import complete_valence_hydrogens
from matrix_chem.geometry import MolecularGeometry
from matrix_numerics import eigh_arrays

from .geometry import build_cartesian_seed
from .model import SwitchAtom, SwitchBond, SwitchMolecularGraph


SWITCH_DEPICTION_SCHEMA = "matrix.switch.depiction.v1"
_COLORS = {
    "H": "#555555",
    "B": "#C9895B",
    "C": "#222222",
    "N": "#2457D6",
    "O": "#D12B2B",
    "F": "#198C3B",
    "P": "#D67B16",
    "S": "#B28A00",
    "Cl": "#198C3B",
    "Br": "#8A3A28",
    "I": "#6D3AA8",
}


@dataclass(frozen=True)
class MoleculeDepictionLayout:
    """The graph and coordinates used by every SWITCH 2-D renderer."""

    graph: SwitchMolecularGraph
    coordinates: np.ndarray
    size: tuple[int, int]


def build_molecule_depiction_layout(
    graph: SwitchMolecularGraph,
    *,
    size: tuple[int, int] = (440, 330),
    explicit_hydrogens: bool = True,
    coordinates_angstrom: np.ndarray | None = None,
) -> MoleculeDepictionLayout:
    """Build the canonical toolkit-independent SWITCH depiction scene."""

    expanded = graph
    if explicit_hydrogens and not any(atom.symbol == "H" for atom in graph.atoms):
        expanded = _expand_hydrogens(graph)
    scene_graph = _hide_carbon_hydrogens(expanded)
    coordinates = _canvas_coordinates(scene_graph, size, coordinates_angstrom)
    return MoleculeDepictionLayout(
        graph=scene_graph,
        coordinates=coordinates,
        size=(int(size[0]), int(size[1])),
    )


def render_molecule_svg(
    graph: SwitchMolecularGraph,
    *,
    size: tuple[int, int] = (440, 330),
    explicit_hydrogens: bool = True,
    coordinates_angstrom: np.ndarray | None = None,
) -> str:
    layout = build_molecule_depiction_layout(
        graph,
        size=size,
        explicit_hydrogens=explicit_hydrogens,
        coordinates_angstrom=coordinates_angstrom,
    )
    scene_graph = layout.graph
    coordinates = layout.coordinates
    width, height = size
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Molecular structure">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        f'<!-- {SWITCH_DEPICTION_SCHEMA} -->',
        '<g stroke="#2A2A2A" stroke-width="2.2" stroke-linecap="round">',
    ]
    for bond in scene_graph.bonds:
        elements.extend(_svg_bond(bond, coordinates, scene_graph))
    for cycle in _aromatic_cycles(scene_graph):
        centre = np.mean(coordinates[list(cycle)], axis=0)
        radius = float(np.mean(np.linalg.norm(coordinates[list(cycle)] - centre, axis=1))) * 0.62
        elements.append(
            f'<circle data-aromatic-ring="true" cx="{centre[0]:.2f}" '
            f'cy="{centre[1]:.2f}" r="{radius:.2f}" fill="none" '
            f'stroke="#555555" stroke-width="1.4"/>'
        )
    elements.append("</g>")
    elements.append('<g font-family="Arial,Helvetica,sans-serif" text-anchor="middle">')
    for atom in scene_graph.atoms:
        label = _atom_label(atom, scene_graph)
        if not label:
            continue
        x, y = coordinates[atom.index]
        color = _COLORS.get(atom.symbol, "#6B3FA0")
        radius = max(9.0, 4.8 * len(label))
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="white"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{y + 5.0:.2f}" font-size="15" '
            f'font-weight="600" fill="{color}">{escape(label)}</text>'
        )
    elements.extend(("</g>", "</svg>"))
    return "\n".join(elements) + "\n"


def render_molecule_png(
    graph: SwitchMolecularGraph,
    *,
    size: tuple[int, int] = (440, 330),
    explicit_hydrogens: bool = True,
    coordinates_angstrom: np.ndarray | None = None,
) -> bytes:
    layout = build_molecule_depiction_layout(
        graph,
        size=size,
        explicit_hydrogens=explicit_hydrogens,
        coordinates_angstrom=coordinates_angstrom,
    )
    scene_graph = layout.graph
    coordinates = layout.coordinates
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = _font(15)
    for bond in scene_graph.bonds:
        _draw_bond(draw, bond, coordinates, scene_graph)
    for cycle in _aromatic_cycles(scene_graph):
        centre = np.mean(coordinates[list(cycle)], axis=0)
        radius = float(np.mean(np.linalg.norm(coordinates[list(cycle)] - centre, axis=1))) * 0.62
        draw.ellipse(
            (centre[0] - radius, centre[1] - radius,
             centre[0] + radius, centre[1] + radius),
            outline="#555555",
            width=1,
        )
    for atom in scene_graph.atoms:
        label = _atom_label(atom, scene_graph)
        if not label:
            continue
        x, y = coordinates[atom.index]
        box = draw.textbbox((x, y), label, font=font, anchor="mm")
        draw.rounded_rectangle(
            (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2),
            radius=3,
            fill="white",
        )
        draw.text(
            (x, y),
            label,
            font=font,
            fill=_COLORS.get(atom.symbol, "#6B3FA0"),
            anchor="mm",
        )
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _expand_hydrogens(graph: SwitchMolecularGraph) -> SwitchMolecularGraph:
    if all(atom.hydrogen_count == 0 for atom in graph.atoms):
        return graph
    seed = build_cartesian_seed(graph, complete_hydrogens=False)
    dummy = MolecularGeometry(
        atoms=tuple(atom.symbol for atom in graph.atoms),
        coordinates_angstrom=np.asarray(seed.coordinates_angstrom, dtype=float),
        charge=graph.total_formal_charge,
    )
    completion = complete_valence_hydrogens(
        dummy,
        tuple(bond.key for bond in graph.bonds),
        bond_orders={bond.key: bond.order for bond in graph.bonds},
        requested_counts=tuple(atom.hydrogen_count for atom in graph.atoms),
    )
    if not completion.additions:
        return graph
    atoms = list(graph.atoms)
    bonds = list(graph.bonds)
    components = [list(component) for component in graph.components]
    component_by_atom = {
        atom: component_index
        for component_index, component in enumerate(components)
        for atom in component
    }
    for addition in completion.additions:
        atoms.append(
            SwitchAtom(
                index=addition.atom,
                symbol="H",
                hydrogen_count=0,
                bracketed=True,
            )
        )
        bonds.append(SwitchBond(left=addition.parent, right=addition.atom))
        components[component_by_atom[addition.parent]].append(addition.atom)
    return SwitchMolecularGraph(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        components=tuple(tuple(component) for component in components),
        source_smiles=graph.source_smiles,
        total_formal_charge=graph.total_formal_charge,
    )


def _hide_carbon_hydrogens(graph: SwitchMolecularGraph) -> SwitchMolecularGraph:
    """Remove carbon-bound H atoms from the layout while retaining polar H.

    Hidden carbon hydrogens must not participate in the 2-D force/layout
    calculation: doing so stretches peptide backbones and makes aromatic
    substituents appear detached.  Hydrogens on N/O/S remain visible because
    they carry useful chemical information (OH, NH and NH2).
    """
    hidden = {
        atom.index
        for atom in graph.atoms
        if atom.symbol == "H"
        and not any(
            graph.atoms[parent].symbol in {"N", "O", "S"}
            for parent in graph.neighbors(atom.index)
        )
    }
    if not hidden:
        return graph
    kept = [atom for atom in graph.atoms if atom.index not in hidden]
    mapping = {atom.index: index for index, atom in enumerate(kept)}
    atoms = tuple(
        SwitchAtom(
            index=index,
            symbol=atom.symbol,
            isotope=atom.isotope,
            formal_charge=atom.formal_charge,
            hydrogen_count=atom.hydrogen_count,
            aromatic=atom.aromatic,
            chirality=atom.chirality,
            atom_class=atom.atom_class,
            bracketed=atom.bracketed,
            source_span=atom.source_span,
            stereo_neighbors=atom.stereo_neighbors,
        )
        for index, atom in enumerate(kept)
    )
    bonds = tuple(
        SwitchBond(
            left=mapping[bond.left],
            right=mapping[bond.right],
            order=bond.order,
            aromatic=bond.aromatic,
            direction=bond.direction,
            dative=bond.dative,
            ring_label=bond.ring_label,
        )
        for bond in graph.bonds
        if bond.left not in hidden and bond.right not in hidden
    )
    components = tuple(
        tuple(mapping[index] for index in component if index in mapping)
        for component in graph.components
    )
    return SwitchMolecularGraph(
        atoms=atoms,
        bonds=bonds,
        components=components,
        source_smiles=graph.source_smiles,
        total_formal_charge=graph.total_formal_charge,
    )


def _canvas_coordinates(
    graph: SwitchMolecularGraph,
    size: tuple[int, int],
    coordinates_angstrom: np.ndarray | None = None,
) -> np.ndarray:
    if coordinates_angstrom is not None:
        cartesian = np.asarray(coordinates_angstrom, dtype=float)
        if cartesian.shape == (len(graph.atoms), 3) and np.all(np.isfinite(cartesian)):
            return _project_cartesian_coordinates(cartesian, size)
    coordinates = np.zeros((len(graph.atoms), 2), dtype=float)
    offset = 0.0
    for component in graph.components:
        local = _component_layout(graph, component)
        local[:, 0] -= np.min(local[:, 0])
        coordinates[np.asarray(component)] = local + np.asarray((offset, 0.0))
        offset += float(np.ptp(local[:, 0])) + 2.0
    coordinates -= np.mean(coordinates, axis=0)
    coordinates = _regularize_valence_angles(graph, coordinates)
    coordinates = _regularize_aromatic_rings(graph, coordinates)
    coordinates = _place_aromatic_hydrogens(graph, coordinates)
    width, height = size
    span = np.ptp(coordinates, axis=0)
    scale = min(
        (width - 88.0) / max(float(span[0]), 1.0),
        (height - 88.0) / max(float(span[1]), 1.0),
        42.0,
    )
    coordinates *= scale
    coordinates[:, 0] += width / 2.0
    coordinates[:, 1] = height / 2.0 - coordinates[:, 1]
    return coordinates


def _project_cartesian_coordinates(
    coordinates: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Project real Cartesian coordinates with the SONIC viewer convention.

    The SONIC motion canvas draws the actual 3-D structure after a fixed
    camera rotation, rather than rebuilding a 2-D graph layout.  Reusing the
    same deterministic camera here preserves real valence angles and avoids
    MDS straightening for XYZ/ORACLE/catalogue inputs.
    """

    centred = np.asarray(coordinates, dtype=float) - np.mean(coordinates, axis=0)
    rotation_y = 0.55
    rotation_x = -0.35
    ry = np.asarray(
        [[np.cos(rotation_y), 0.0, np.sin(rotation_y)],
         [0.0, 1.0, 0.0],
         [-np.sin(rotation_y), 0.0, np.cos(rotation_y)]]
    )
    rx = np.asarray(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(rotation_x), -np.sin(rotation_x)],
         [0.0, np.sin(rotation_x), np.cos(rotation_x)]]
    )
    projected = (centred @ ry.T @ rx.T)[:, :2]
    width, height = size
    span = np.ptp(projected, axis=0)
    scale = min(
        (width - 88.0) / max(float(span[0]), 1.0),
        (height - 88.0) / max(float(span[1]), 1.0),
        42.0,
    )
    projected *= scale
    projected[:, 0] += width / 2.0
    projected[:, 1] = height / 2.0 - projected[:, 1]
    return projected


def _aromatic_cycles(graph: SwitchMolecularGraph) -> tuple[tuple[int, ...], ...]:
    """Return the smallest five- and six-membered aromatic cycles.

    Fused systems must retain both rings.  A degree-two-only walk loses the
    shared atoms of indole and naphthalene, which is precisely what caused the
    old renderer to bend the second ring into a spurious polygon.
    """

    aromatic = {atom.index for atom in graph.atoms if atom.aromatic}
    adjacency = {atom: tuple(neighbour for neighbour in graph.neighbors(atom) if neighbour in aromatic) for atom in aromatic}
    cycles: set[tuple[int, ...]] = set()
    for start in sorted(aromatic):
        stack = [(start, (start,))]
        while stack:
            current, path = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour == start and len(path) in {5, 6}:
                    variants = []
                    for orientation in (path, tuple(reversed(path))):
                        variants.extend(
                            orientation[index:] + orientation[:index]
                            for index in range(len(orientation))
                        )
                    cycles.add(min(variants))
                elif neighbour > start and neighbour not in path and len(path) < 6:
                    stack.append((neighbour, path + (neighbour,)))
    return tuple(sorted(cycles, key=lambda cycle: (-len(cycle), cycle)))


def _regularize_aromatic_rings(
    graph: SwitchMolecularGraph,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Make isolated aromatic rings regular while preserving substituent pose."""

    result = coordinates.copy()
    placed: set[int] = set()
    for cycle in _aromatic_cycles(graph):
        shared = [atom for atom in cycle if atom in placed]
        if len(shared) >= 2:
            adjacent = next(
                (index for index in range(len(cycle))
                 if {cycle[index], cycle[(index + 1) % len(cycle)]}.issubset(shared)),
                None,
            )
            if adjacent is None:
                continue
            first, second = cycle[adjacent], cycle[(adjacent + 1) % len(cycle)]
            sequence = list(cycle[adjacent:]) + list(cycle[:adjacent])
            if sequence[1] != second:
                sequence = [sequence[0]] + list(reversed(sequence[1:]))
            candidates = _regular_polygon_from_edge(
                result[first], result[second], len(cycle),
            )
            # Fused rings must be placed on the opposite side of the shared
            # edge from the ring already drawn.  Comparing distances to the
            # shared atoms cannot distinguish the two mirror images because
            # both candidates contain that edge exactly.
            reference_atoms = [
                atom for atom in placed
                if atom not in shared
                and any(atom in graph.neighbors(shared_atom) for shared_atom in shared)
            ]
            side = candidates[0]
            if reference_atoms:
                edge = result[second] - result[first]
                reference = np.mean(result[reference_atoms], axis=0)
                reference_side = float(np.cross(edge, reference - result[first]))
                candidate_sides = [
                    float(np.cross(edge, np.mean(candidate[2:], axis=0) - result[first]))
                    for candidate in candidates
                ]
                opposite = [
                    candidate for candidate, candidate_side in zip(candidates, candidate_sides, strict=True)
                    if candidate_side * reference_side < 0.0
                ]
                if opposite:
                    side = opposite[0]
            for atom, point in zip(sequence, side, strict=True):
                result[atom] = point
            placed.update(cycle)
            continue
        points = result[list(cycle)]
        centre = np.mean(points, axis=0)
        radius = max(float(np.mean(np.linalg.norm(points - centre, axis=1))), 0.5)
        radius = 1.0 / (2.0 * np.sin(np.pi / len(cycle)))
        radial = points[0] - centre
        start_angle = float(np.arctan2(radial[1], radial[0]))
        candidates = tuple(
            centre + np.column_stack((
                np.cos(start_angle + direction * np.arange(len(cycle)) * 2.0 * np.pi / len(cycle)),
                np.sin(start_angle + direction * np.arange(len(cycle)) * 2.0 * np.pi / len(cycle)),
            )) * radius
            for direction in (1.0, -1.0)
        )
        selected = min(
            candidates,
            key=lambda candidate: float(np.sum((candidate - points) ** 2)),
        )
        result[list(cycle)] = selected
        placed.update(cycle)
    return result


def _regular_polygon_from_edge(
    first: np.ndarray,
    second: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the two regular polygons sharing the directed first edge."""

    vector = second - first
    length = max(float(np.linalg.norm(vector)), 1.0e-8)
    unit = vector / length
    angle = 2.0 * np.pi / size
    candidates = []
    for direction in (1.0, -1.0):
        points = [first.copy(), second.copy()]
        edge = unit * length
        for _ in range(size - 2):
            rotation = np.asarray(
                [[np.cos(direction * angle), -np.sin(direction * angle)],
                 [np.sin(direction * angle), np.cos(direction * angle)]]
            )
            edge = rotation @ edge
            points.append(points[-1] + edge)
        candidates.append(np.asarray(points))
    return tuple(candidates)


def _place_aromatic_hydrogens(
    graph: SwitchMolecularGraph,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Place implicit aromatic H atoms on the outward ring bisector.

    Spectral layouts are deterministic but can put substituent hydrogens on
    the inward bisector of a ring.  The chemical placement is independent of
    atom order: for every aromatic carbon with two ring neighbours, move its
    H along the outward bisector defined by those neighbours.
    """

    result = coordinates.copy()
    for atom in graph.atoms:
        if atom.symbol != "H":
            continue
        parents = graph.neighbors(atom.index)
        if len(parents) != 1:
            continue
        parent = graph.atoms[parents[0]]
        if not parent.aromatic or parent.symbol not in {"C", "B", "N"}:
            continue
        ring_neighbours = tuple(
            neighbour
            for neighbour in graph.neighbors(parent.index)
            if graph.atoms[neighbour].aromatic
        )
        if len(ring_neighbours) != 2:
            continue
        first = result[ring_neighbours[0]] - result[parent.index]
        second = result[ring_neighbours[1]] - result[parent.index]
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
            continue
        direction = -(first / first_norm + second / second_norm)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            continue
        direction /= direction_norm
        # Hydrogen coordinates from the generic graph layout are not a
        # chemical bond-length source (especially in fused rings).  Use the
        # normalized 2-D bond scale instead of inheriting a potentially huge
        # provisional H--parent distance.
        ring_lengths = [
            float(np.linalg.norm(result[neighbour] - result[parent.index]))
            for neighbour in ring_neighbours
        ]
        distance = max(0.82 * float(np.mean(ring_lengths)), 0.42)
        result[atom.index] = result[parent.index] + distance * direction
    return result


def _component_layout(
    graph: SwitchMolecularGraph,
    component: tuple[int, ...],
) -> np.ndarray:
    count = len(component)
    if count == 1:
        return np.zeros((1, 2))
    lookup = {atom: local for local, atom in enumerate(component)}
    rows: list[int] = []
    columns: list[int] = []
    for bond in graph.bonds:
        if bond.left in lookup and bond.right in lookup:
            left, right = lookup[bond.left], lookup[bond.right]
            rows.extend((left, right))
            columns.extend((right, left))
    adjacency = csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(count, count),
    )
    distances = np.asarray(shortest_path(adjacency, directed=False))
    centering = np.eye(count) - np.full((count, count), 1.0 / count)
    gram = -0.5 * centering @ (distances * distances) @ centering
    values, vectors = eigh_arrays(gram)
    order = np.argsort(values)[::-1]
    coordinates = np.zeros((count, 2))
    for axis, index in enumerate(order[:2]):
        if values[index] > 1.0e-10:
            coordinates[:, axis] = vectors[:, index] * np.sqrt(values[index])
            pivot = int(np.argmax(np.abs(coordinates[:, axis])))
            if coordinates[pivot, axis] < 0:
                coordinates[:, axis] *= -1
    if np.ptp(coordinates[:, 1]) < 1.0e-6 and count > 2:
        coordinates[:, 1] = 0.28 * np.sin(np.arange(count) * pi * 0.72)
    return _relax_layout(coordinates, tuple((lookup[b.left], lookup[b.right]) for b in graph.bonds if b.left in lookup and b.right in lookup))


def _regularize_valence_angles(
    graph: SwitchMolecularGraph,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Prevent chemically meaningful two-bond angles from becoming linear.

    Graph-distance MDS is excellent at separating rings and components, but
    it has no notion of valence geometry and commonly projects an acyclic
    chain into 175--180 degree zigzags.  For every non-aromatic degree-two
    centre we therefore impose the standard 2-D drawing target of 120 degrees
    while preserving both bond lengths.  Applying this before ring
    regularization keeps fused aromatic systems under the dedicated ring
    layout and makes the rule independent of molecule family.
    """

    result = coordinates.copy()
    for _ in range(8):
        changed = False
        for atom in graph.atoms:
            neighbours = graph.neighbors(atom.index)
            if len(neighbours) != 2 or atom.aromatic:
                continue
            left, right = neighbours
            centre = result[atom.index]
            first = result[left] - centre
            second = result[right] - centre
            first_length = float(np.linalg.norm(first))
            second_length = float(np.linalg.norm(second))
            if min(first_length, second_length) <= 1.0e-8:
                continue
            first_angle = float(np.arctan2(first[1], first[0]))
            second_angle = float(np.arctan2(second[1], second[0]))
            separation = (second_angle - first_angle + np.pi) % (2.0 * np.pi) - np.pi
            magnitude = abs(separation)
            if magnitude > np.pi:
                magnitude = 2.0 * np.pi - magnitude
            target = 2.0 * np.pi / 3.0
            if abs(magnitude - target) < 0.08:
                continue
            sign = 1.0 if separation >= 0.0 else -1.0
            correction = 0.55 * (target - magnitude) / 2.0
            rotation_left = sign * correction
            rotation_right = -sign * correction
            left_rotation = np.asarray(
                [[np.cos(rotation_left), -np.sin(rotation_left)],
                 [np.sin(rotation_left), np.cos(rotation_left)]]
            )
            right_rotation = np.asarray(
                [[np.cos(rotation_right), -np.sin(rotation_right)],
                 [np.sin(rotation_right), np.cos(rotation_right)]]
            )
            result[left] = centre + left_rotation @ first
            result[right] = centre + right_rotation @ second
            changed = True
        if not changed:
            break
    return result


def _relax_layout(
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> np.ndarray:
    result = coordinates.copy()
    count = len(result)
    if count > 180:
        return result
    bonded = {tuple(sorted(bond)) for bond in bonds}
    for iteration in range(80):
        force = np.zeros_like(result)
        for left, right in combinations(range(count), 2):
            delta = result[left] - result[right]
            distance = max(float(np.linalg.norm(delta)), 0.08)
            direction = delta / distance
            force[left] += 0.025 * direction / (distance * distance)
            force[right] -= 0.025 * direction / (distance * distance)
        for left, right in bonded:
            delta = result[right] - result[left]
            distance = max(float(np.linalg.norm(delta)), 0.08)
            direction = delta / distance
            attraction = 0.06 * (distance - 1.0) * direction
            force[left] += attraction
            force[right] -= attraction
        temperature = 0.08 * (1.0 - iteration / 80.0)
        norm = np.linalg.norm(force, axis=1)
        scale = np.minimum(1.0, temperature / np.maximum(norm, 1.0e-12))
        result += force * scale[:, None]
        result -= np.mean(result, axis=0)
    return result


def _svg_bond(
    bond: SwitchBond,
    coordinates: np.ndarray,
    graph: SwitchMolecularGraph,
) -> list[str]:
    left = coordinates[bond.left]
    right = coordinates[bond.right]
    lines = []
    if graph.atoms[bond.left].chirality in {"@", "@@", "@TH1", "@TH2"}:
        left, right = _clip_bond_endpoints(bond, left, right, graph)
        vector = right - left
        norm = max(float(np.linalg.norm(vector)), 1.0)
        normal = np.asarray((-vector[1], vector[0])) / norm * 4.5
        if graph.atoms[bond.left].chirality in {"@", "@TH1"}:
            points = (left, right + normal, right - normal)
            payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            return [
                f'<polygon data-stereo="solid" points="{payload}" '
                f'fill="#2A2A2A" stroke="none"/>'
            ]
        hashed = []
        for fraction in np.linspace(0.18, 0.92, 6):
            center = left + fraction * vector
            half = fraction * normal
            hashed.append(
                f'<line data-stereo="hashed" x1="{center[0] - half[0]:.2f}" '
                f'y1="{center[1] - half[1]:.2f}" x2="{center[0] + half[0]:.2f}" '
                f'y2="{center[1] + half[1]:.2f}"/>'
            )
        return hashed
    offsets = _bond_offsets(
        bond,
        left,
        right,
            _label_radius(graph.atoms[bond.left], graph),
            _label_radius(graph.atoms[bond.right], graph),
    )
    dash = ""
    for start, stop in offsets:
        lines.append(
            f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" '
            f'x2="{stop[0]:.2f}" y2="{stop[1]:.2f}"{dash}/>'
        )
    return lines


def _draw_bond(draw, bond, coordinates, graph) -> None:
    left, right = _clip_bond_endpoints(
        bond, coordinates[bond.left], coordinates[bond.right], graph,
    )
    if graph.atoms[bond.left].chirality in {"@", "@@", "@TH1", "@TH2"}:
        vector = right - left
        norm = max(float(np.linalg.norm(vector)), 1.0)
        normal = np.asarray((-vector[1], vector[0])) / norm * 4.5
        if graph.atoms[bond.left].chirality in {"@", "@TH1"}:
            draw.polygon(
                [tuple(left), tuple(right + normal), tuple(right - normal)],
                fill="#2A2A2A",
            )
        else:
            for fraction in np.linspace(0.18, 0.92, 6):
                center = left + fraction * vector
                half = fraction * normal
                draw.line(
                    (tuple(center - half), tuple(center + half)),
                    fill="#2A2A2A",
                    width=1,
                )
        return
    for start, stop in _bond_offsets(
        bond,
        left,
        right,
        _label_radius(graph.atoms[bond.left], graph),
        _label_radius(graph.atoms[bond.right], graph),
    ):
        draw.line((tuple(start), tuple(stop)), fill="#2A2A2A", width=2)


def _bond_offsets(
    bond: SwitchBond,
    left: np.ndarray,
    right: np.ndarray,
    left_radius: float = 0.0,
    right_radius: float = 0.0,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    vector = right - left
    norm = max(float(np.linalg.norm(vector)), 1.0)
    normal = np.asarray((-vector[1], vector[0])) / norm
    if bond.order >= 2.5:
        values = (-4.0, 0.0, 4.0)
    elif bond.order >= 1.75:
        values = (-2.8, 2.8)
    else:
        values = (0.0,)
    return tuple((left + value * normal, right + value * normal) for value in values)


def _label_radius(atom: SwitchAtom, graph: SwitchMolecularGraph) -> float:
    """Return the white mask radius used for an atom label."""

    label = _atom_label(atom, graph)
    return 0.0 if not label else max(9.0, 4.8 * len(label))


def _clip_bond_endpoints(
    bond: SwitchBond,
    left: np.ndarray,
    right: np.ndarray,
    graph: SwitchMolecularGraph,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep bond strokes outside atom labels.

    The old renderer relied on a white label background to mask bonds.  That
    works for single bonds but leaves double/triple strokes visibly entering
    N/O/S labels.  Clipping before drawing gives the same clean convention as
    standard chemical drawing programs and is independent of molecule class.
    """

    vector = right - left
    distance = float(np.linalg.norm(vector))
    if distance <= 1.0e-12:
        return left, right
    direction = vector / distance
    start = left + direction * _label_radius(graph.atoms[bond.left], graph)
    stop = right - direction * _label_radius(graph.atoms[bond.right], graph)
    if float(np.dot(stop - start, direction)) <= 0.0:
        midpoint = (left + right) / 2.0
        return midpoint, midpoint
    return start, stop


def _atom_label(atom: SwitchAtom, graph: SwitchMolecularGraph) -> str:
    degree = len(graph.neighbors(atom.index))
    if atom.symbol == "C" and degree > 0 and atom.formal_charge == 0 and atom.isotope is None:
        return ""
    isotope = "" if atom.isotope is None else str(atom.isotope)
    if atom.formal_charge == 0:
        charge = ""
    elif abs(atom.formal_charge) == 1:
        charge = "+" if atom.formal_charge > 0 else "−"
    else:
        charge = f"{abs(atom.formal_charge)}{'+' if atom.formal_charge > 0 else '−'}"
    return f"{isotope}{atom.symbol}{charge}"


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


__all__ = [
    "SWITCH_DEPICTION_SCHEMA",
    "render_molecule_png",
    "render_molecule_svg",
]
