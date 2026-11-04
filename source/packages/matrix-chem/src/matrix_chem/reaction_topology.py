"""Endpoint graph partition for atlas-driven transition-state coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import MolecularGeometry
from .average_atomic_masses import atomic_mass
from .oracle_sonic_contract import graph_cycle_rank
from .primitive_coordinates import (
    Primitive,
    build_primitives,
    eval_primitive,
    primitive_b_matrix,
)
from .reaction_atlas import reaction_atlas_entry
from .topology.elements import atomic_number
from .topology.pipeline import build_topology_objects


REACTION_TOPOLOGY_SCHEMA = "matrix.reaction_topology.v1"


@dataclass(frozen=True)
class ReactionTopologyPartition:
    schema: str
    reaction_class_id: str
    atom_mapping: tuple[int, ...]
    common_edges: tuple[tuple[int, int], ...]
    broken_edges: tuple[tuple[int, int], ...]
    formed_edges: tuple[tuple[int, int], ...]
    mobile_atoms: tuple[int, ...]
    fixed_atoms: tuple[int, ...]
    start_anchor_atoms: tuple[int, ...]
    end_anchor_atoms: tuple[int, ...]
    start_domain_atoms: tuple[int, ...]
    end_domain_atoms: tuple[int, ...]
    start_cycle_rank: int
    end_cycle_rank: int


@dataclass(frozen=True)
class EndpointPrimitivePartition:
    common: tuple[Primitive, ...]
    start_reactive: tuple[Primitive, ...]
    end_reactive: tuple[Primitive, ...]
    start_unmatched_nonreactive: tuple[Primitive, ...]
    end_unmatched_nonreactive: tuple[Primitive, ...]


@dataclass(frozen=True)
class AtomTransferReactiveBlock:
    """Three common coordinates built from two endpoint attachment charts."""

    start_chart: tuple[Primitive, ...]
    end_chart: tuple[Primitive, ...]
    source_labels: tuple[str, ...]
    transformation: tuple[tuple[float, ...], ...]
    reactive_rank: int
    start_projected_rank: int
    end_projected_rank: int
    endpoint_displacement: tuple[float, ...]
    minimum_singular_value_start: float
    minimum_singular_value_end: float


def partition_reaction_topology(
    start: MolecularGeometry,
    end: MolecularGeometry,
    *,
    atom_mapping: Sequence[int] | None = None,
    mobile_atoms: Sequence[int] | None = None,
    attachment_shells: int = 2,
) -> ReactionTopologyPartition:
    """Classify endpoint graph edits and identify mobile and attachment domains."""

    mapping = _atom_mapping(start, end, atom_mapping)
    end_xyz = end.coordinates_angstrom[np.asarray(mapping, dtype=int)]
    numbers = tuple(_atomic_number(atom) for atom in start.atoms)
    start_edges = _perceived_edges(start.coordinates_angstrom, numbers)
    end_edges = _perceived_edges(end_xyz, numbers)
    common = start_edges & end_edges
    broken = start_edges - end_edges
    formed = end_edges - start_edges
    start_rank = graph_cycle_rank(
        start.natoms, ((left + 1, right + 1) for left, right in start_edges)
    )
    end_rank = graph_cycle_rank(start.natoms, ((left + 1, right + 1) for left, right in end_edges))
    components = _components(start.natoms, common)
    selected_mobile = (
        tuple(sorted({int(atom) for atom in mobile_atoms}))
        if mobile_atoms is not None
        else _default_mobile_atoms(
            start.natoms,
            components,
            common,
            broken,
            formed,
            cycle_rank_changed=end_rank != start_rank,
        )
    )
    if any(atom < 0 or atom >= start.natoms for atom in selected_mobile):
        raise ValueError("mobile fragment contains an invalid atom index")
    mobile_set = set(selected_mobile)
    start_anchors = _opposite_atoms(broken, mobile_set)
    end_anchors = _opposite_atoms(formed, mobile_set)
    mobile_component_count = sum(
        bool(set(component).intersection(selected_mobile)) for component in components
    )
    reaction_class = _classify(
        selected_mobile,
        broken,
        formed,
        end_rank - start_rank,
        component_count=len(components),
        mobile_component_count=mobile_component_count,
    )
    _validate_atlas_class(
        reaction_class,
        mobile_count=len(selected_mobile),
        broken_count=len(broken),
        formed_count=len(formed),
        cycle_delta=end_rank - start_rank,
    )
    fixed = tuple(atom for atom in range(start.natoms) if atom not in mobile_set)
    fixed_edges = {
        edge for edge in common if edge[0] not in mobile_set and edge[1] not in mobile_set
    }
    return ReactionTopologyPartition(
        schema=REACTION_TOPOLOGY_SCHEMA,
        reaction_class_id=reaction_class,
        atom_mapping=mapping,
        common_edges=tuple(sorted(common)),
        broken_edges=tuple(sorted(broken)),
        formed_edges=tuple(sorted(formed)),
        mobile_atoms=selected_mobile,
        fixed_atoms=fixed,
        start_anchor_atoms=start_anchors,
        end_anchor_atoms=end_anchors,
        start_domain_atoms=_graph_shell(start_anchors, fixed_edges, attachment_shells),
        end_domain_atoms=_graph_shell(end_anchors, fixed_edges, attachment_shells),
        start_cycle_rank=start_rank,
        end_cycle_rank=end_rank,
    )


def build_endpoint_primitive_partition(
    start: MolecularGeometry,
    end: MolecularGeometry,
    topology: ReactionTopologyPartition,
    *,
    include_pseudo_bonds: bool = False,
) -> EndpointPrimitivePartition:
    """Separate identical definitions from endpoint-specific reactive primitives."""

    mapping = topology.atom_mapping
    end_xyz = end.coordinates_angstrom[np.asarray(mapping, dtype=int)]
    numbers = tuple(_atomic_number(atom) for atom in start.atoms)
    _continuous, start_graph, _rings, _synthons, _aromaticity = build_topology_objects(
        start.coordinates_angstrom, numbers
    )
    _continuous, end_graph, _rings, _synthons, _aromaticity = build_topology_objects(
        end_xyz, numbers
    )
    start_pool = tuple(
        build_primitives(
            start_graph,
            start.coordinates_angstrom,
            include_pseudo_bonds=include_pseudo_bonds,
        )
    )
    end_pool = tuple(
        build_primitives(end_graph, end_xyz, include_pseudo_bonds=include_pseudo_bonds)
    )
    start_by_key = {_primitive_key(primitive): primitive for primitive in start_pool}
    end_by_key = {_primitive_key(primitive): primitive for primitive in end_pool}
    common_keys = start_by_key.keys() & end_by_key.keys()
    mobile = set(topology.mobile_atoms)
    reactive_domain = mobile | set(topology.start_domain_atoms) | set(
        topology.end_domain_atoms
    )
    reactive_domain.update(atom for edge in topology.broken_edges for atom in edge)
    reactive_domain.update(atom for edge in topology.formed_edges for atom in edge)
    start_only = start_by_key.keys() - common_keys
    end_only = end_by_key.keys() - common_keys
    return EndpointPrimitivePartition(
        common=tuple(start_by_key[key] for key in start_by_key if key in common_keys),
        start_reactive=tuple(
            start_by_key[key]
            for key in start_by_key
            if key in start_only and reactive_domain.intersection(start_by_key[key].atoms)
        ),
        end_reactive=tuple(
            end_by_key[key]
            for key in end_by_key
            if key in end_only and reactive_domain.intersection(end_by_key[key].atoms)
        ),
        start_unmatched_nonreactive=tuple(
            start_by_key[key]
            for key in start_by_key
            if key in start_only
            and not reactive_domain.intersection(start_by_key[key].atoms)
        ),
        end_unmatched_nonreactive=tuple(
            end_by_key[key]
            for key in end_by_key
            if key in end_only
            and not reactive_domain.intersection(end_by_key[key].atoms)
        ),
    )


def build_atom_transfer_reactive_block(
    start: MolecularGeometry,
    end: MolecularGeometry,
    topology: ReactionTopologyPartition,
    coordinates: EndpointPrimitivePartition,
    *,
    rank_tolerance: float = 1.0e-9,
) -> AtomTransferReactiveBlock:
    """Combine the two attachment triplets into one rank-three reactive block."""

    if topology.reaction_class_id != "ATOM_TRANSFER":
        raise ValueError("dual attachment chart is implemented only for ATOM_TRANSFER")
    if (
        len(topology.mobile_atoms) != 1
        or len(topology.start_anchor_atoms) != 1
        or len(topology.end_anchor_atoms) != 1
    ):
        raise ValueError("ATOM_TRANSFER requires one mobile atom and one anchor per endpoint")
    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank tolerance must be finite and positive")
    mobile = topology.mobile_atoms[0]
    start_chart = _attachment_chart(mobile, topology.start_anchor_atoms[0], topology.common_edges)
    end_chart = _attachment_chart(mobile, topology.end_anchor_atoms[0], topology.common_edges)
    sources = start_chart + end_chart
    end_xyz = end.coordinates_angstrom[np.asarray(topology.atom_mapping, dtype=int)]
    masses = np.asarray([atomic_mass(_atomic_number(atom)) for atom in start.atoms], dtype=float)
    if np.any(masses <= 0.0):
        raise ValueError("ATOM_TRANSFER requires positive atomic masses")
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        common_start = primitive_b_matrix(coordinates.common, start.coordinates_angstrom)
        common_end = primitive_b_matrix(coordinates.common, end_xyz)
        source_start_values, source_start = _attachment_source_data(
            sources, start.coordinates_angstrom
        )
        source_end_values, source_end = _attachment_source_data(sources, end_xyz)
    common_start = common_start * inverse_sqrt_mass[None, :]
    common_end = common_end * inverse_sqrt_mass[None, :]
    source_start = source_start * inverse_sqrt_mass[None, :]
    source_end = source_end * inverse_sqrt_mass[None, :]
    projected_start = source_start @ _row_complement_projector(common_start, tolerance)
    projected_end = source_end @ _row_complement_projector(common_end, tolerance)
    norms = np.sqrt(
        0.5
        * (
            np.sum(projected_start * projected_start, axis=1)
            + np.sum(projected_end * projected_end, axis=1)
        )
    )
    if np.any(~np.isfinite(norms)) or np.any(norms <= tolerance):
        raise ValueError("an attachment-chart coordinate is singular after common-block projection")
    normalized_start = projected_start / norms[:, None]
    normalized_end = projected_end / norms[:, None]
    start_rank = int(np.linalg.matrix_rank(normalized_start, tol=tolerance))
    end_rank = int(np.linalg.matrix_rank(normalized_end, tol=tolerance))
    reactive_rank = min(start_rank, end_rank)
    if reactive_rank != 3:
        raise ValueError(
            "ATOM_TRANSFER attachment charts must have projected rank three at both endpoints"
        )
    # Keep the in-plane distance/angle sector separate from the out-of-plane
    # periodic-torsion sector.  A global PCA followed by reaction alignment
    # would otherwise mix opposite symmetry sectors for planar endpoints.
    even_indices = np.asarray((0, 1, 3, 4), dtype=int)
    odd_indices = np.asarray((2, 5), dtype=int)
    even_transformation = _sector_transformation(
        normalized_start,
        normalized_end,
        norms,
        even_indices,
        expected_rank=2,
        tolerance=tolerance,
    )
    odd_transformation = _sector_transformation(
        normalized_start,
        normalized_end,
        norms,
        odd_indices,
        expected_rank=1,
        tolerance=tolerance,
    )
    source_delta = source_end_values - source_start_values
    even_transformation = _reaction_aligned_rows(
        even_transformation,
        even_transformation @ source_delta,
        tolerance,
    )
    odd_pivot = int(np.argmax(np.abs(odd_transformation[0])))
    if odd_transformation[0, odd_pivot] < 0.0:
        odd_transformation[0] *= -1.0
    transformation = np.vstack((even_transformation, odd_transformation))
    transformed_delta = transformation @ source_delta
    final_start = transformation @ projected_start
    final_end = transformation @ projected_end
    final_start_singular = np.linalg.svd(final_start, compute_uv=False)
    final_end_singular = np.linalg.svd(final_end, compute_uv=False)
    if (
        int(np.linalg.matrix_rank(final_start, tol=tolerance)) != reactive_rank
        or int(np.linalg.matrix_rank(final_end, tol=tolerance)) != reactive_rank
    ):
        raise ValueError("combined ATOM_TRANSFER chart loses rank at an endpoint")
    return AtomTransferReactiveBlock(
        start_chart=start_chart,
        end_chart=end_chart,
        source_labels=tuple(
            f"SIN[{primitive.label}]" if primitive.kind == "dihedral" else primitive.label
            for primitive in sources
        ),
        transformation=tuple(tuple(float(value) for value in row) for row in transformation),
        reactive_rank=reactive_rank,
        start_projected_rank=start_rank,
        end_projected_rank=end_rank,
        endpoint_displacement=tuple(float(value) for value in transformed_delta),
        minimum_singular_value_start=float(final_start_singular[-1]),
        minimum_singular_value_end=float(final_end_singular[-1]),
    )


def _atom_mapping(
    start: MolecularGeometry,
    end: MolecularGeometry,
    mapping: Sequence[int] | None,
) -> tuple[int, ...]:
    if start.natoms != end.natoms:
        raise ValueError("reaction endpoints must contain the same number of atoms")
    values = tuple(range(start.natoms)) if mapping is None else tuple(int(item) for item in mapping)
    if len(values) != start.natoms or sorted(values) != list(range(end.natoms)):
        raise ValueError("reaction atom mapping must be a complete bijection")
    if tuple(end.atoms[index] for index in values) != start.atoms:
        raise ValueError("reaction atom mapping changes element identity")
    return values


def _atomic_number(atom: str) -> int:
    number = atomic_number(atom)
    if number is None or number <= 0:
        raise ValueError(f"unsupported atom label in reaction endpoint: {atom}")
    return int(number)


def _perceived_edges(coordinates: np.ndarray, numbers: Sequence[int]) -> set[tuple[int, int]]:
    _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(
        coordinates, numbers
    )
    return {tuple(sorted((int(left), int(right)))) for left, right in graph.bonds}


def _infer_mobile_fragment(
    natoms: int,
    common_edges: set[tuple[int, int]],
    broken: set[tuple[int, int]],
    formed: set[tuple[int, int]],
) -> tuple[int, ...]:
    components = _components(natoms, common_edges)
    edited_atoms = {atom for edge in broken | formed for atom in edge}
    participating = tuple(
        component for component in components if edited_atoms.intersection(component)
    )
    if len(participating) < 2:
        raise ValueError("mobile fragment is ambiguous; provide it explicitly")
    largest_size = max(len(component) for component in participating)
    references = tuple(
        component for component in participating if len(component) == largest_size
    )
    if len(references) != 1:
        raise ValueError(
            "fixed reaction domain is not unique; provide the mobile fragment explicitly"
        )
    reference = set(references[0])
    mobile = tuple(
        atom
        for component in participating
        if set(component) != reference
        for atom in component
    )
    if not mobile:
        raise ValueError("mobile fragment is ambiguous; provide it explicitly")
    return tuple(sorted(mobile))


def _default_mobile_atoms(
    natoms: int,
    components: tuple[tuple[int, ...], ...],
    common_edges: set[tuple[int, int]],
    broken: set[tuple[int, int]],
    formed: set[tuple[int, int]],
    *,
    cycle_rank_changed: bool,
) -> tuple[int, ...]:
    """Choose a deterministic moving domain without hiding graph-edit ambiguity."""

    if not broken and not formed:
        if len(components) == 1:
            return ()
        # The largest covalent component fixes the external Cartesian gauge;
        # ties are resolved by atom order, which is already part of the
        # endpoint mapping contract.  Every remaining fragment is mobile.
        reference = min(components, key=lambda item: (-len(item), item))
        reference_atoms = set(reference)
        return tuple(atom for atom in range(natoms) if atom not in reference_atoms)
    if cycle_rank_changed:
        return ()
    return _infer_mobile_fragment(natoms, common_edges, broken, formed)


def _classify(
    mobile: tuple[int, ...],
    broken: set[tuple[int, int]],
    formed: set[tuple[int, int]],
    cycle_delta: int,
    *,
    component_count: int,
    mobile_component_count: int,
) -> str:
    if not broken and not formed:
        return (
            "NONCOVALENT_REORIENTATION"
            if component_count > 1
            else "CONFORMATIONAL_TRANSITION"
        )
    if cycle_delta < 0:
        return "RING_OPENING"
    if cycle_delta > 0:
        return "RING_CLOSURE"
    mobile_set = set(mobile)
    broken_boundary = {
        edge for edge in broken if (edge[0] in mobile_set) ^ (edge[1] in mobile_set)
    }
    formed_boundary = {
        edge for edge in formed if (edge[0] in mobile_set) ^ (edge[1] in mobile_set)
    }
    if len(mobile) == 1 and len(broken) == len(formed) == 1 and cycle_delta == 0:
        return "ATOM_TRANSFER"
    if len(mobile) == 2 and broken_boundary and not formed_boundary:
        return "DIATOMIC_ELIMINATION"
    if len(mobile) == 2 and formed_boundary and not broken_boundary:
        return "DIATOMIC_ADDITION"
    if (
        len(mobile) >= 2
        and broken_boundary
        and formed_boundary
        and mobile_component_count == 1
    ):
        return "FRAGMENT_TRANSFER"
    if broken_boundary and formed_boundary and mobile_component_count >= 2:
        return "SUBSTITUTION"
    if len(mobile) == 1 and broken_boundary and not formed_boundary:
        return "ATOM_ELIMINATION"
    if len(mobile) == 1 and formed_boundary and not broken_boundary:
        return "ATOM_ADDITION"
    return "REARRANGEMENT"


def _validate_atlas_class(
    class_id: str,
    *,
    mobile_count: int,
    broken_count: int,
    formed_count: int,
    cycle_delta: int,
) -> None:
    entry = reaction_atlas_entry(class_id)
    checks = (
        (mobile_count, entry.mobile_min_atoms, entry.mobile_max_atoms, "mobile atoms"),
        (broken_count, entry.broken_bonds_min, entry.broken_bonds_max, "broken bonds"),
        (formed_count, entry.formed_bonds_min, entry.formed_bonds_max, "formed bonds"),
    )
    for value, minimum, maximum, label in checks:
        if value < minimum or (maximum >= 0 and value > maximum):
            raise ValueError(f"{class_id} has an atlas-incompatible number of {label}")
    if cycle_delta not in entry.cycle_rank_deltas:
        raise ValueError(f"{class_id} has an atlas-incompatible cycle-rank change")


def _opposite_atoms(edges: set[tuple[int, int]], mobile: set[int]) -> tuple[int, ...]:
    atoms = {
        right if left in mobile else left
        for left, right in edges
        if (left in mobile) ^ (right in mobile)
    }
    return tuple(sorted(atoms))


def _components(natoms: int, edges: set[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(natoms)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(natoms))
    result: list[tuple[int, ...]] = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[int] = set()
        while stack:
            atom = stack.pop()
            if atom in component:
                continue
            component.add(atom)
            stack.extend(adjacency[atom] - component)
        unseen -= component
        result.append(tuple(sorted(component)))
    return tuple(result)


def _graph_shell(
    anchors: tuple[int, ...],
    edges: set[tuple[int, int]],
    shells: int,
) -> tuple[int, ...]:
    if shells < 0:
        raise ValueError("attachment shells cannot be negative")
    selected = set(anchors)
    frontier = set(anchors)
    for _ in range(shells):
        neighbors = {
            right if left in frontier else left
            for left, right in edges
            if (left in frontier) ^ (right in frontier)
        }
        frontier = neighbors - selected
        selected |= frontier
    return tuple(sorted(selected))


def _primitive_key(primitive: Primitive) -> tuple[object, ...]:
    atoms = tuple(int(atom) for atom in primitive.atoms)
    if primitive.kind in {"bond", "hbond_dist", "pseudo_bond"}:
        atoms = tuple(sorted(atoms))
    elif primitive.kind == "angle" and len(atoms) == 3:
        atoms = min(atoms, tuple(reversed(atoms)))
    elif primitive.kind == "dihedral" and len(atoms) == 4:
        atoms = min(atoms, tuple(reversed(atoms)))
    return primitive.kind, atoms, int(primitive.mode), tuple(primitive.ref)


def _attachment_chart(
    mobile: int,
    anchor: int,
    common_edges: Sequence[tuple[int, int]],
) -> tuple[Primitive, ...]:
    adjacency: dict[int, set[int]] = {}
    for left, right in common_edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    first_candidates = sorted(adjacency.get(anchor, set()) - {mobile})
    if not first_candidates:
        raise ValueError("attachment anchor has no fixed reference atom")
    first = first_candidates[0]
    second_candidates = sorted(adjacency.get(first, set()) - {anchor, mobile})
    if not second_candidates:
        raise ValueError("attachment chart has no second fixed reference atom")
    second = second_candidates[0]
    return (
        Primitive("bond", (mobile, anchor)),
        Primitive("angle", (mobile, anchor, first)),
        Primitive("dihedral", (mobile, anchor, first, second)),
    )


def _row_complement_projector(rows: np.ndarray, tolerance: float) -> np.ndarray:
    columns = rows.shape[1]
    if rows.size == 0:
        return np.eye(columns, dtype=float)
    _u_matrix, singular, vt_matrix = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 0:
        return np.eye(columns, dtype=float)
    basis = vt_matrix[:rank]
    return np.eye(columns, dtype=float) - basis.T @ basis


def _attachment_source_data(
    sources: Sequence[Primitive],
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate attachment torsions through their periodic sine embedding."""

    values = np.asarray(
        [eval_primitive(primitive, coordinates) for primitive in sources], dtype=float
    )
    rows = primitive_b_matrix(sources, coordinates)
    for index, primitive in enumerate(sources):
        if primitive.kind == "dihedral":
            angle = float(values[index])
            rows[index] *= np.cos(angle)
            values[index] = np.sin(angle)
    return values, rows


def _sector_transformation(
    normalized_start: np.ndarray,
    normalized_end: np.ndarray,
    norms: np.ndarray,
    indices: np.ndarray,
    *,
    expected_rank: int,
    tolerance: float,
) -> np.ndarray:
    """Build a shared linear chart within one symmetry-compatible sector."""

    start_sector = normalized_start[indices]
    end_sector = normalized_end[indices]
    if (
        int(np.linalg.matrix_rank(start_sector, tol=tolerance)) != expected_rank
        or int(np.linalg.matrix_rank(end_sector, tol=tolerance)) != expected_rank
    ):
        raise ValueError("ATOM_TRANSFER symmetry sector has an unexpected projected rank")
    covariance = start_sector @ start_sector.T + end_sector @ end_sector.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:expected_rank]
    local = eigenvectors[:, order].T / norms[indices][None, :]
    result = np.zeros((expected_rank, normalized_start.shape[0]), dtype=float)
    result[:, indices] = local
    return result


def _reaction_aligned_rows(
    transformation: np.ndarray,
    displacement: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    norm = float(np.linalg.norm(displacement))
    if norm <= tolerance:
        raise ValueError("endpoint attachment charts have no finite reactive displacement")
    first = displacement / norm
    _u_matrix, _singular, vt_matrix = np.linalg.svd(first.reshape(1, -1))
    rotation = np.vstack((first, vt_matrix[1:]))
    aligned = rotation @ transformation
    for row in range(aligned.shape[0]):
        pivot = int(np.argmax(np.abs(aligned[row])))
        if aligned[row, pivot] < 0.0:
            aligned[row] *= -1.0
    if float(aligned[0] @ transformation.T @ displacement) < 0.0:
        aligned[0] *= -1.0
    return aligned


__all__ = [
    "REACTION_TOPOLOGY_SCHEMA",
    "EndpointPrimitivePartition",
    "AtomTransferReactiveBlock",
    "ReactionTopologyPartition",
    "build_atom_transfer_reactive_block",
    "build_endpoint_primitive_partition",
    "partition_reaction_topology",
]
