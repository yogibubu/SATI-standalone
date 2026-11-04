"""Permutation-safe comparison of an assembled geometry with an L2 record.

The L2 electronic record and an overlap assembly are independent producers of
atom ordering.  This module is deliberately small and strict: no Cartesian
metric is returned until the heavy-atom graphs are fully isomorphic and every
hydrogen has been assigned to the corresponding heavy atom.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import permutations
from typing import Mapping, Sequence

import numpy as np

from matrix_switch import graph_from_topology, maximum_common_connected_subgraphs, parse_smiles


class L2MappingError(ValueError):
    """The assembly and withheld L2 record cannot be compared safely."""


@dataclass(frozen=True)
class L2AtomMapping:
    """Complete zero-based mapping from assembly atoms to L2-record atoms."""

    assembly_to_l2: tuple[int, ...]
    heavy_assembly_to_l2: tuple[tuple[int, int], ...]
    hydrogen_assembly_to_l2: tuple[tuple[int, int], ...]


def compare_assembly_to_l2(
    canonical_smiles: str,
    assembly_atoms: Sequence[str],
    assembly_coordinates: Sequence[Sequence[float]],
    l2_atoms: Sequence[str],
    l2_coordinates: Sequence[Sequence[float]],
    l2_mayer_bond_components: Sequence[Mapping[str, object]],
    *,
    canonical_synthon_labels: Mapping[int, str] | None = None,
    l2_synthon_labels: Mapping[int, str] | None = None,
    bond_threshold: float = 0.30,
) -> dict[str, object]:
    """Return auditable L2 comparison metrics, or reject the comparison.

    ``canonical_smiles`` is the atom-order authority for the target.  Mayer
    edges are used only to reconstruct the withheld L2 topology; Cartesian
    distances are never used to invent a bond graph.
    """

    assembly_xyz = _coordinates(assembly_coordinates, "assembly")
    l2_xyz = _coordinates(l2_coordinates, "L2")
    assembly_symbols = tuple(str(value) for value in assembly_atoms)
    l2_symbols = tuple(str(value) for value in l2_atoms)
    target = parse_smiles(canonical_smiles)
    target_heavy = tuple(atom.index for atom in target.atoms if atom.symbol != "H")
    l2_heavy = tuple(index for index, symbol in enumerate(l2_symbols) if symbol != "H")
    assembly_heavy = tuple(index for index, symbol in enumerate(assembly_symbols) if symbol != "H")
    if len(target_heavy) != len(l2_heavy) or len(target_heavy) != len(assembly_heavy):
        raise L2MappingError("heavy-atom counts differ between canonical target, assembly and L2")

    target_graph, target_lookup = _canonical_heavy_graph(target, target_heavy)
    l2_graph = _l2_heavy_graph(l2_symbols, l2_heavy, l2_mayer_bond_components, bond_threshold)
    _require_full_isomorphism(target_graph, l2_graph, "canonical target↔L2")
    l2_matches = _full_matches(target_graph, l2_graph)
    if not l2_matches:
        raise L2MappingError("canonical target and L2 Mayer graphs are not isomorphic")
    assembly_graph, assembly_lookup = _assembly_heavy_graph(
        assembly_symbols, assembly_heavy, target_graph
    )
    _require_full_isomorphism(target_graph, assembly_graph, "canonical target↔assembly")
    assembly_matches = _full_matches(target_graph, assembly_graph)
    if not assembly_matches:
        raise L2MappingError("assembly heavy-atom graph is not isomorphic to canonical target")
    assembly_match = assembly_matches[0]
    # Symmetric constitutional graphs can have several valid isomorphisms.
    # Resolve that residual permutation against the assembled heavy geometry,
    # never against H coordinates or a raw atom index.
    l2_match = min(
        l2_matches,
        key=lambda match: _mapped_heavy_fit_rmsd(
            assembly_xyz,
            l2_xyz,
            target_heavy,
            assembly_heavy,
            l2_heavy,
            assembly_match,
            match,
        ),
    )

    # target index -> record/assembly index; invert to get assembly -> L2.
    target_to_l2 = {target_heavy[s]: l2_heavy[t] for s, t in zip(l2_match.source_atoms, l2_match.target_atoms)}
    target_to_assembly = {target_heavy[s]: assembly_heavy[t] for s, t in zip(assembly_match.source_atoms, assembly_match.target_atoms)}
    heavy_map = {target_to_assembly[t]: target_to_l2[t] for t in target_heavy}
    _check_synthon_labels(target_to_l2, canonical_synthon_labels, l2_synthon_labels)
    hydrogen_map = _map_hydrogens(
        assembly_symbols, assembly_xyz, l2_symbols, l2_xyz, heavy_map,
        target_graph, target_lookup, l2_mayer_bond_components, bond_threshold,
    )
    mapping = dict(heavy_map)
    mapping.update(hydrogen_map)
    if len(mapping) != len(assembly_symbols) or len(set(mapping.values())) != len(l2_symbols):
        raise L2MappingError("atom mapping is not complete and one-to-one")

    ordered_assembly = np.asarray([assembly_xyz[index] for index in range(len(assembly_symbols))])
    ordered_l2 = np.asarray([l2_xyz[mapping[index]] for index in range(len(assembly_symbols))])
    rotation, translation = _kabsch(ordered_assembly, ordered_l2)
    fitted = ordered_assembly @ rotation + translation
    displacement = np.linalg.norm(fitted - ordered_l2, axis=1)
    bonds = _bond_pairs(target_graph)
    bond_errors = np.asarray([
        np.linalg.norm(assembly_xyz[left] - assembly_xyz[right])
        - np.linalg.norm(l2_xyz[mapping[left]] - l2_xyz[mapping[right]])
        for left, right in _mapped_target_bonds(bonds, target_to_assembly)
    ], dtype=float)
    angles = _angles(target_graph)
    angle_errors = np.asarray([
        _angle(assembly_xyz[a], assembly_xyz[b], assembly_xyz[c])
        - _angle(l2_xyz[mapping[a]], l2_xyz[mapping[b]], l2_xyz[mapping[c]])
        for a, b, c in _mapped_target_angles(angles, target_to_assembly)
    ], dtype=float)
    hydroxyl_torsions = _hydroxyl_torsion_comparison(
        assembly_xyz,
        l2_xyz,
        assembly_symbols,
        l2_symbols,
        mapping,
        l2_mayer_bond_components,
        bond_threshold,
    )
    return {
        "mapping": L2AtomMapping(
            assembly_to_l2=tuple(mapping[index] for index in range(len(assembly_symbols))),
            heavy_assembly_to_l2=tuple(sorted(heavy_map.items())),
            hydrogen_assembly_to_l2=tuple(sorted(hydrogen_map.items())),
        ),
        "mapping_complete": True,
        "mapping_is_one_to_one": True,
        "topology_isomorphic": True,
        "all_atom_rmsd_angstrom": float(np.sqrt(np.mean(displacement ** 2))),
        "heavy_atom_rmsd_angstrom": _heavy_fit_rmsd(
            ordered_assembly, ordered_l2, assembly_heavy
        ),
        "bond_rmse_angstrom": float(np.sqrt(np.mean(bond_errors ** 2))) if bond_errors.size else 0.0,
        "bond_mae_angstrom": float(np.mean(np.abs(bond_errors))) if bond_errors.size else 0.0,
        "bond_max_abs_angstrom": float(np.max(np.abs(bond_errors))) if bond_errors.size else 0.0,
        "bond_errors_angstrom": bond_errors.tolist(),
        "angle_rmse_degrees": float(np.sqrt(np.mean(angle_errors ** 2))) if angle_errors.size else 0.0,
        "angle_mae_degrees": float(np.mean(np.abs(angle_errors))) if angle_errors.size else 0.0,
        "angle_max_abs_degrees": float(np.max(np.abs(angle_errors))) if angle_errors.size else 0.0,
        "cartesian_internal_closure_rmsd_angstrom": float(np.sqrt(np.mean(bond_errors ** 2))) if bond_errors.size else 0.0,
        "hydroxyl_torsions": hydroxyl_torsions,
        "conformer_assignment_pass": bool(
            hydroxyl_torsions
            and all(item["assignment_matches_reference"] for item in hydroxyl_torsions)
        ),
    }


def _hydroxyl_torsion_comparison(
    assembly_xyz,
    l2_xyz,
    assembly_symbols,
    l2_symbols,
    assembly_to_l2,
    components,
    threshold,
):
    """Compare every unambiguous H-O-C-C torsion under the frozen atom map."""

    adjacency = {index: set() for index in range(len(l2_symbols))}
    for component in components:
        pair = component.get("atoms", ())
        if len(pair) != 2 or float(component.get("total", 0.0)) < threshold:
            continue
        left, right = int(pair[0]) - 1, int(pair[1]) - 1
        adjacency[left].add(right)
        adjacency[right].add(left)
    l2_to_assembly = {l2: assembly for assembly, l2 in assembly_to_l2.items()}
    records = []
    for oxygen, symbol in enumerate(l2_symbols):
        if symbol != "O":
            continue
        hydrogens = [atom for atom in adjacency[oxygen] if l2_symbols[atom] == "H"]
        carbons = [atom for atom in adjacency[oxygen] if l2_symbols[atom] == "C"]
        if len(hydrogens) != 1 or len(carbons) != 1:
            continue
        carbon = carbons[0]
        adjacent = [
            atom
            for atom in adjacency[carbon]
            if atom != oxygen and l2_symbols[atom] == "C"
        ]
        if len(adjacent) != 1:
            continue
        l2_atoms = (hydrogens[0], oxygen, carbon, adjacent[0])
        if any(atom not in l2_to_assembly for atom in l2_atoms):
            continue
        assembly_atoms = tuple(l2_to_assembly[atom] for atom in l2_atoms)
        reference = _dihedral_degrees(l2_xyz, l2_atoms)
        predicted = _dihedral_degrees(assembly_xyz, assembly_atoms)
        reference_cis = _circular_error_degrees(reference, 0.0) <= 30.0
        predicted_cis = _circular_error_degrees(predicted, 0.0) <= 30.0
        records.append(
            {
                "reference_atoms_one_based": [atom + 1 for atom in l2_atoms],
                "assembly_atoms_one_based": [atom + 1 for atom in assembly_atoms],
                "reference_degrees": reference,
                "predicted_degrees": predicted,
                "absolute_circular_error_degrees": _circular_error_degrees(
                    predicted, reference
                ),
                "cis_assignment": predicted_cis,
                "assignment_matches_reference": predicted_cis == reference_cis,
            }
        )
    return records


def _dihedral_degrees(coordinates, atoms):
    first, second, third, fourth = (np.asarray(coordinates[index], dtype=float) for index in atoms)
    b0 = first - second
    b1 = third - second
    b2 = fourth - third
    b1 /= max(float(np.linalg.norm(b1)), 1.0e-15)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _circular_error_degrees(value, reference):
    return float(abs((float(value) - float(reference) + 180.0) % 360.0 - 180.0))


def _coordinates(value, label):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise L2MappingError(f"{label} coordinates are not finite Nx3 values")
    return array


def _canonical_heavy_graph(graph, heavy_indices):
    lookup = {old: new for new, old in enumerate(heavy_indices)}
    atoms = [graph.atoms[index].symbol for index in heavy_indices]
    bonds = []
    orders = {}
    for bond in graph.bonds:
        if bond.left in lookup and bond.right in lookup:
            pair = (lookup[bond.left], lookup[bond.right])
            bonds.append(pair)
            orders[tuple(sorted(pair))] = 1.5 if bond.aromatic else float(bond.order)
    return graph_from_topology(atoms, bonds, bond_orders=orders), lookup


def _l2_heavy_graph(symbols, heavy_indices, components, threshold):
    lookup = {old: new for new, old in enumerate(heavy_indices)}
    bonds = []
    orders = {}
    for component in components:
        pair = component.get("atoms", ())
        if len(pair) != 2 or float(component.get("total", 0.0)) < threshold:
            continue
        left, right = (int(pair[0]) - 1, int(pair[1]) - 1)
        if left not in lookup or right not in lookup:
            continue
        key = tuple(sorted((lookup[left], lookup[right])))
        bonds.append(key)
        pi = float(component.get("pi", 0.0))
        total = float(component.get("total", 0.0))
        # Small Mayer pi terms on polar single bonds are not aromaticity.  A
        # conservative threshold keeps carbonyl-adjacent C--O bonds in the
        # single class while retaining genuinely delocalized ring edges.
        orders[key] = 1.5 if pi > 0.35 and total < 1.80 else (2.0 if total >= 1.75 else 1.0)
    if len(set(bonds)) != len(bonds):
        raise L2MappingError("L2 Mayer record contains duplicate heavy-atom bonds")
    return graph_from_topology([symbols[index] for index in heavy_indices], bonds, bond_orders=orders)


def _assembly_heavy_graph(symbols, heavy_indices, target_graph):
    if tuple(symbols[index] for index in heavy_indices) != tuple(atom.symbol for atom in target_graph.atoms):
        raise L2MappingError("assembly heavy-atom elements do not match canonical target")
    # Assembly coordinates are produced from the canonical target graph; use its
    # immutable connectivity and only validate the atom labels here.
    return graph_from_topology(
        [atom.symbol for atom in target_graph.atoms],
        [bond.key for bond in target_graph.bonds],
        bond_orders={bond.key: bond.order for bond in target_graph.bonds},
    ), {index: index for index in heavy_indices}


def _require_full_isomorphism(source, target, label):
    if len(source.atoms) != len(target.atoms):
        raise L2MappingError(f"{label} atom counts differ")
    if _best_full_match(source, target) is None:
        raise L2MappingError(f"{label} topology is not isomorphic")


def _best_full_match(source, target):
    matches = _full_matches(source, target)
    return matches[0] if matches else None


def _full_matches(source, target):
    matches = maximum_common_connected_subgraphs(
        source, target, minimum_atoms=len(source.atoms), timeout_seconds=2.0, max_matches=256
    )
    return tuple(match for match in matches if match.atom_count == len(source.atoms))


def _mapped_heavy_fit_rmsd(assembly_xyz, l2_xyz, target_heavy, assembly_heavy, l2_heavy, assembly_match, l2_match):
    target_to_assembly = {
        target_heavy[source]: assembly_heavy[target]
        for source, target in zip(assembly_match.source_atoms, assembly_match.target_atoms)
    }
    target_to_l2 = {
        target_heavy[source]: l2_heavy[target]
        for source, target in zip(l2_match.source_atoms, l2_match.target_atoms)
    }
    moving = np.asarray([assembly_xyz[target_to_assembly[index]] for index in target_heavy])
    fixed = np.asarray([l2_xyz[target_to_l2[index]] for index in target_heavy])
    rotation, translation = _kabsch(moving, fixed)
    return float(np.sqrt(np.mean(np.sum((moving @ rotation + translation - fixed) ** 2, axis=1))))


def _check_synthon_labels(target_to_l2, canonical, l2):
    if canonical is None or l2 is None:
        return
    for target, label in canonical.items():
        if l2.get(target_to_l2.get(int(target))) != label:
            raise L2MappingError("synthon labels are incompatible under the heavy-atom mapping")


def _map_hydrogens(assembly_symbols, assembly_xyz, l2_symbols, l2_xyz, heavy_map, target_graph, target_lookup, components, threshold):
    l2_neighbors = {index: [] for index, symbol in enumerate(l2_symbols) if symbol != "H"}
    for component in components:
        pair = component.get("atoms", ())
        if len(pair) != 2 or float(component.get("total", 0.0)) < threshold:
            continue
        left, right = (int(pair[0]) - 1, int(pair[1]) - 1)
        if l2_symbols[left] == "H": l2_neighbors.setdefault(right, []).append(left)
        if l2_symbols[right] == "H": l2_neighbors.setdefault(left, []).append(right)
    assembly_heavy = tuple(sorted(heavy_map))
    expected = _expected_hydrogen_counts(target_graph)
    capacities = {assembly_heavy[target_index]: expected[target_index] for target_index in range(len(target_graph.atoms))}
    hydrogen_indices = [index for index, symbol in enumerate(assembly_symbols) if symbol == "H"]
    distances = {
        hydrogen: sorted(
            (float(np.linalg.norm(assembly_xyz[hydrogen] - assembly_xyz[heavy])), heavy)
            for heavy in assembly_heavy
        )
        for hydrogen in hydrogen_indices
    }
    order = sorted(hydrogen_indices, key=lambda hydrogen: distances[hydrogen][1][0] - distances[hydrogen][0][0])
    assignment = {}

    def assign(position: int) -> bool:
        if position == len(order):
            return True
        hydrogen = order[position]
        for distance, heavy in distances[hydrogen]:
            if distance >= 1.35 or capacities[heavy] <= 0:
                continue
            capacities[heavy] -= 1
            assignment[hydrogen] = heavy
            if assign(position + 1):
                return True
            capacities[heavy] += 1
            del assignment[hydrogen]
        return False

    if not assign(0) or any(value != 0 for value in capacities.values()):
        raise L2MappingError("assembly hydrogens cannot satisfy constitutional heavy-atom valences")
    assembly_neighbors = {index: [] for index in assembly_heavy}
    for hydrogen, heavy in assignment.items():
        assembly_neighbors[heavy].append(hydrogen)
    result = {}
    for assembly_heavy_index, l2_heavy_index in heavy_map.items():
        left = sorted(
            assembly_neighbors[assembly_heavy_index],
            key=lambda index: float(np.linalg.norm(assembly_xyz[index] - assembly_xyz[assembly_heavy_index])),
        )
        right = sorted(
            l2_neighbors.get(l2_heavy_index, ()),
            key=lambda index: float(np.linalg.norm(l2_xyz[index] - l2_xyz[l2_heavy_index])),
        )
        if len(left) != len(right):
            raise L2MappingError(f"hydrogen count mismatch at heavy atom {assembly_heavy_index + 1}")
        rotation, translation = _kabsch(
            np.asarray([assembly_xyz[index] for index in sorted(assembly_heavy)]),
            np.asarray([l2_xyz[heavy_map[index]] for index in sorted(assembly_heavy)]),
        )
        best = min(
            permutations(right),
            key=lambda candidate: sum(
                np.linalg.norm(
                    assembly_xyz[assembly_h] @ rotation + translation - l2_xyz[l2_h]
                ) ** 2
                + 25.0
                * (
                    np.linalg.norm(assembly_xyz[assembly_h] - assembly_xyz[assembly_heavy_index])
                    - np.linalg.norm(l2_xyz[l2_h] - l2_xyz[l2_heavy_index])
                ) ** 2
                for assembly_h, l2_h in zip(left, candidate)
            ),
        )
        for assembly_h, l2_h in zip(left, best):
            if assembly_symbols[assembly_h] != l2_symbols[l2_h]:
                raise L2MappingError("hydrogen element mismatch")
            result[assembly_h] = l2_h
    return result


def _expected_hydrogen_counts(graph):
    valence = {"C": 4.0, "N": 3.0, "O": 2.0, "S": 2.0, "P": 3.0}
    result = []
    for atom in graph.atoms:
        used = sum(float(bond.order) for bond in graph.bonds if bond.left == atom.index or bond.right == atom.index)
        result.append(max(0, int(round(valence.get(atom.symbol, 0.0) - used))))
    return tuple(result)


def _bond_pairs(graph):
    return tuple(tuple(bond.key) for bond in graph.bonds)


def _mapped_target_bonds(bonds, target_to_assembly):
    return tuple((target_to_assembly[a], target_to_assembly[b]) for a, b in bonds)


def _angles(graph):
    adjacency = {index: set() for index in range(len(graph.atoms))}
    for bond in graph.bonds:
        adjacency[bond.left].add(bond.right); adjacency[bond.right].add(bond.left)
    return tuple((left, center, right) for center, neighbors in adjacency.items() for left in neighbors for right in neighbors if left < right)


def _mapped_target_angles(angles, target_to_assembly):
    return tuple((target_to_assembly[a], target_to_assembly[b], target_to_assembly[c]) for a, b, c in angles)


def _angle(a, b, c):
    left, right = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cosine = np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-15)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _kabsch(moving, fixed):
    mc, fc = moving.mean(axis=0), fixed.mean(axis=0)
    left, _, right_t = np.linalg.svd((moving - mc).T @ (fixed - fc))
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation, fc - mc @ rotation


def _heavy_fit_rmsd(assembly, l2, assembly_heavy):
    indices = list(assembly_heavy)
    moving, fixed = assembly[indices], l2[indices]
    rotation, translation = _kabsch(moving, fixed)
    return float(np.sqrt(np.mean(np.sum((moving @ rotation + translation - fixed) ** 2, axis=1))))


__all__ = ["L2AtomMapping", "L2MappingError", "compare_assembly_to_l2"]
