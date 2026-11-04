"""Local SYCART completion when ORACLE changes a reactive topology."""

from __future__ import annotations

from dataclasses import replace
import re

import numpy as np

from .evaluation import build_gic_b_matrix
from .models import FrozenGIC, GICPrimitive


_MINIMUM_RETAINED_SONIC_FRACTION = 0.25


def _external_rows(xyz: np.ndarray) -> np.ndarray:
    centered = xyz - xyz.mean(axis=0)
    rows = [np.tile(axis, (len(xyz), 1)).ravel() for axis in np.eye(3)]
    rows.extend(np.cross(axis, centered).ravel() for axis in np.eye(3))
    _u, singular, right = np.linalg.svd(np.asarray(rows), full_matrices=False)
    rank = int(np.count_nonzero(singular > 1.0e-10 * singular[0]))
    return right[:rank]


def _rank(rows: np.ndarray) -> int:
    if not rows.size:
        return 0
    singular = np.linalg.svd(rows, compute_uv=False)
    return int(np.count_nonzero(singular > max(1.0e-9, 1.0e-9 * singular[0])))


def _depends_on_changed_bond(gic, primitives, changed_bonds) -> bool:
    changed_atoms = {int(atom) for bond in changed_bonds for atom in bond}
    for identifier, _coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
        primitive = primitives[identifier]
        function = primitive.function
        atoms = tuple(int(atom) for atom in primitive.atoms)
        if function == "SYCART" or (
            function == "SOOP" and changed_atoms.intersection(atoms)
        ):
            return True
        for bond in changed_bonds:
            if function == "R" and frozenset(atoms) == bond:
                return True
            if function in {"A", "L"} and len(atoms) >= 3 and (
                frozenset(atoms[:2]) == bond or frozenset(atoms[1:3]) == bond
            ):
                return True
            if function in {"D", "U", "IMPD"} and len(atoms) >= 4 and (
                frozenset(atoms[1:3]) == bond
            ):
                return True
    return False


def build_local_sycart_transition_definition(
    previous_definition,
    previous_contract,
    current_contract,
    coordinates_angstrom,
    *,
    sycart_definition=None,
    symmetry_operations=(),
    whole_molecule=False,
):
    """Retain nonreactive GICs and complete the reactive space with SYCART.

    The changed ORACLE bond endpoints seed the local Cartesian domain. It is
    enlarged by graph neighbours only until the complete vibrational rank is
    available. No molecule identity, energy, or convergence result enters the
    choice.
    """
    old_bonds = {frozenset(pair) for pair in previous_contract.primary_topology.bonds}
    new_bonds = {frozenset(pair) for pair in current_contract.primary_topology.bonds}
    changed_bonds = old_bonds.symmetric_difference(new_bonds)
    if not changed_bonds and not whole_molecule:
        raise ValueError("ORACLE requested local SYCART without a changed bond topology")
    zone = (
        set(range(len(coordinates_angstrom)))
        if whole_molecule else {atom - 1 for bond in changed_bonds for atom in bond}
    )
    return _build_local_sycart_definition(
        previous_definition,
        coordinates_angstrom,
        changed_bonds=changed_bonds,
        topology_bonds=old_bonds | new_bonds,
        zone=zone,
        sycart_definition=sycart_definition,
        symmetry_operations=symmetry_operations,
        whole_molecule=whole_molecule,
    )


def build_local_sycart_reactive_definition(
    previous_definition,
    oracle_contract,
    coordinates_angstrom,
    *,
    reactive_atoms_one_based,
    reactive_bonds_one_based,
):
    """Use ORACLE's initial reactive zone without selecting atoms from outcomes."""

    zone = {int(atom) - 1 for atom in reactive_atoms_one_based}
    reactive_bonds = {frozenset(map(int, pair)) for pair in reactive_bonds_one_based}
    if not zone or not reactive_bonds:
        raise ValueError("initial local SYCART needs an ORACLE reactive zone and bonds")
    topology_bonds = {
        frozenset(pair) for pair in oracle_contract.primary_topology.bonds
    }
    return _build_local_sycart_definition(
        previous_definition,
        coordinates_angstrom,
        changed_bonds=reactive_bonds,
        topology_bonds=topology_bonds | reactive_bonds,
        zone=zone,
    )


def build_local_sycart_uncertain_definition(
    previous_definition,
    coordinates_angstrom,
    *,
    uncertain_atoms_zero_based,
    uncertain_contacts_one_based,
    topology_bonds_one_based,
):
    """Complete an ORACLE-declared ambiguous contact region with SYCART."""
    zone = {int(atom) for atom in uncertain_atoms_zero_based}
    contacts = {frozenset(map(int, pair)) for pair in uncertain_contacts_one_based}
    bonds = {frozenset(map(int, pair)) for pair in topology_bonds_one_based}
    if not zone or not contacts:
        raise ValueError("local SYCART needs an uncertain atom region and contacts")
    return _build_local_sycart_definition(
        previous_definition,
        coordinates_angstrom,
        changed_bonds=contacts,
        topology_bonds=bonds | contacts,
        zone=zone,
        reason_label="LOCAL_SYCART_UNCERTAIN_CONTACTS",
    )


def build_initial_local_sycart_from_oracle(
    definition,
    contract,
    coordinates_angstrom,
    atomic_numbers,
):
    """Replace only ORACLE's ambiguous metal--ring contacts at a minimum."""
    from matrix_chem import build_topology_objects
    from matrix_oracle.topology_domains import assess_topology_domains

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    numbers = tuple(int(number) for number in atomic_numbers)
    domains = assess_topology_domains(numbers, xyz, task_regime="MINIMUM")
    if not domains.uncertain_atoms:
        raise ValueError("ORACLE has no uncertain local domain for SYCART completion")
    _continuous, _discrete, rings, _synthons, _aromaticity = build_topology_objects(
        xyz, numbers
    )
    contacts = set()
    for reason in domains.reasons:
        match = re.fullmatch(
            r"PARTIAL_METAL_RING_HAPTICITY:M(\d+):R(\d+):\d+/\d+", reason
        )
        if match is None:
            continue
        metal, ring = (int(value) for value in match.groups())
        contacts.update((metal, int(atom) + 1) for atom in rings.rings[ring - 1].atoms)
    if not contacts:
        raise ValueError("ORACLE uncertain domain has no supported local contact")
    return build_local_sycart_uncertain_definition(
        definition,
        xyz,
        uncertain_atoms_zero_based=domains.uncertain_atoms,
        uncertain_contacts_one_based=contacts,
        topology_bonds_one_based=contract.primary_topology.bonds,
    )


def _build_local_sycart_definition(
    previous_definition,
    coordinates_angstrom,
    *,
    changed_bonds,
    topology_bonds,
    zone,
    reason_label="LOCAL_SYCART_TOPOLOGY_CHANGE",
    sycart_definition=None,
    symmetry_operations=(),
    whole_molecule=False,
):
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    symmetric = previous_definition.point_group.strip().upper() != "C1"
    if symmetric:
        if (
            sycart_definition is None
            or sycart_definition.point_group != previous_definition.point_group
            or sycart_definition.target_rank != previous_definition.target_rank
        ):
            raise ValueError("local SYCART needs the matching ORACLE symmetry basis")
        symmetry_vectors = np.asarray(sycart_definition.vectors, dtype=float)
        symmetry_labels = tuple(sycart_definition.irreps)
        if symmetry_vectors.shape != (previous_definition.target_rank, xyz.size):
            raise ValueError("local SYCART symmetry basis has the wrong dimensions")
        permutations = tuple(tuple(int(atom) - 1 for atom in operation.permutation)
                             for operation in symmetry_operations)
        if not permutations:
            raise ValueError("local SYCART needs frozen molecular symmetry operations")

        def symmetry_closure(atoms):
            closure = set(atoms)
            for permutation in permutations:
                closure.update(permutation[atom] for atom in atoms)
            return closure

        zone = symmetry_closure(zone)
    # Once a region has entered the Cartesian chart it remains there even if
    # ORACLE later restores an earlier bond topology.  A later topology change
    # may only enlarge that region.
    for diagnostic in previous_definition.semantic_diagnostics:
        if diagnostic.startswith("LOCAL_SYCART_ATOMS="):
            zone.update(
                int(atom) - 1
                for atom in diagnostic.partition("=")[2].split(",")
                if atom
            )
    if symmetric:
        zone = symmetry_closure(zone)
    neighbours = {atom: set() for atom in range(len(xyz))}
    for bond in topology_bonds:
        left, right = (int(atom) - 1 for atom in bond)
        neighbours[left].add(right)
        neighbours[right].add(left)
    previous_b = (
        np.zeros((previous_definition.target_rank, xyz.size), dtype=float)
        if whole_molecule else np.asarray(
            build_gic_b_matrix(previous_definition, coordinates_angstrom=xyz).rows,
            dtype=float,
        )
    )
    external = _external_rows(xyz)
    primitives = {item.identifier: item for item in previous_definition.primitives}
    eligible = () if whole_molecule else tuple(
        index for index, gic in enumerate(previous_definition.gics)
        if not _depends_on_changed_bond(gic, primitives, changed_bonds)
    )
    target = int(previous_definition.target_rank)

    def local_axes(atoms):
        if symmetric:
            mask = np.zeros(xyz.size)
            for atom in atoms:
                mask[3 * atom:3 * atom + 3] = 1.0
            axes = symmetry_vectors * mask
            return axes - axes @ external.T @ external
        columns = [3 * atom + axis for atom in sorted(atoms) for axis in range(3)]
        axes = np.eye(xyz.size)[columns]
        return axes - axes @ external.T @ external

    while True:
        axes = local_axes(zone)
        if _rank(np.vstack((external, axes, previous_b[list(eligible)]))) >= target + len(external):
            break
        frontier = sorted(set().union(*(neighbours[atom] for atom in zone)) - zone)
        if not frontier:
            raise ValueError("local SYCART domain cannot complete vibrational rank")
        gains = [
            (_rank(np.vstack((external, local_axes(
                symmetry_closure(zone | {atom}) if symmetric else zone | {atom}
            ), previous_b[list(eligible)]))), atom)
            for atom in frontier
        ]
        best = sorted(gains, key=lambda item: (-item[0], item[1]))[0][1]
        zone = symmetry_closure(zone | {best}) if symmetric else zone | {best}

    # Retain as many chemically stable GICs as possible subject to the local
    # Cartesian block spanning the full vibrational space.
    chosen = []
    occupied = np.vstack((external, axes))
    for index in eligible:
        trial = np.vstack((occupied, previous_b[index]))
        if _rank(trial) > _rank(occupied):
            occupied = trial
            chosen.append(index)
    # A tiny residual SONIC block does not justify a mixed representation.
    # This rank fraction is independent of molecule identity or outcome.
    full_sycart = whole_molecule or len(chosen) / target <= _MINIMUM_RETAINED_SONIC_FRACTION
    if full_sycart:
        zone = set(range(len(xyz)))
        axes = local_axes(zone)
        chosen = []
    needed = target - len(chosen)
    fixed = np.vstack((external, previous_b[chosen]))
    _u, singular, right = np.linalg.svd(fixed, full_matrices=False)
    fixed_rank = int(np.count_nonzero(singular > max(1.0e-9, 1.0e-9 * singular[0])))
    fixed_basis = right[:fixed_rank]
    residual = axes - axes @ fixed_basis.T @ fixed_basis
    if needed <= 0 or _rank(residual) < needed:
        raise ValueError("local SYCART completion has insufficient independent directions")
    if symmetric:
        selected_rows = []
        selected_labels = []
        for row, label in zip(residual, symmetry_labels, strict=True):
            trial = row.copy()
            for selected in selected_rows:
                trial -= float(trial @ selected) * selected
            norm = float(np.linalg.norm(trial))
            if norm > 1.0e-9:
                selected_rows.append(trial / norm)
                selected_labels.append(label)
            if len(selected_rows) == needed:
                break
        if len(selected_rows) != needed:
            raise ValueError("local SYCART symmetry blocks cannot close the residual rank")
        sycart_rows = np.asarray(selected_rows)
    else:
        _left, _local_singular, local_right = np.linalg.svd(residual, full_matrices=False)
        sycart_rows = local_right[:needed]
        selected_labels = ["A"] * needed
    retained_gics = tuple(previous_definition.gics[index] for index in chosen)
    retained_ids = {
        identifier for gic in retained_gics
        for identifier, _coefficient in gic.coefficients or ((gic.primitive_id, 1.0),)
    }
    retained_primitives = tuple(
        primitive for primitive in previous_definition.primitives
        if primitive.identifier in retained_ids
    )
    additions = []
    gics = list(retained_gics)
    for index, (row, irrep) in enumerate(zip(sycart_rows, selected_labels, strict=True), 1):
        identifier = f"LOCAL_SYCART_{index:04d}"
        support = np.flatnonzero(np.abs(row) > 1.0e-14)
        additions.append(GICPrimitive(
            identifier=identifier,
            name=identifier,
            family="SYCART",
            function="SYCART",
            atoms=tuple(sorted({int(column) // 3 + 1 for column in support})),
            refs=(f"OFFSET:{float(row @ xyz.ravel()):.17g}",) + tuple(
                f"B:{int(column)}:{float(row[column]):.17g}" for column in support
            ),
            chart="FROZEN_REFERENCE_CARTESIAN",
            semantic_type="LOCAL_REACTIVE_SYCART_PROJECTED",
        ))
        gics.append(FrozenGIC(
            identifier=identifier,
            name=identifier,
            family="SYCART",
            irrep=irrep,
            primitive_id=identifier,
            gaussian_expression="NONE",
            coefficients=((identifier, 1.0),),
        ))
    result = replace(
        previous_definition,
        backend="matrix-smith-local-sycart-transition.v1",
        reference_coordinates_angstrom=tuple(tuple(float(v) for v in row) for row in xyz),
        primitives=retained_primitives + tuple(additions),
        gics=tuple(gics),
        rank=target,
        candidate_count=len(retained_primitives) + len(additions),
        semantic_diagnostics=(
            *previous_definition.semantic_diagnostics,
            *(
                marker
                for marker in (
                    "ONIC_CORE COMMON_TYPED_NONREDUNDANT_ALGEBRA",
                    "CHART_ORIENTATION SONIC",
                    "CHART_ROLE EXPLOITATION",
                )
                if marker not in previous_definition.semantic_diagnostics
            ),
            reason_label + "=" + ",".join(
                "-".join(str(atom) for atom in sorted(bond))
                for bond in sorted(changed_bonds, key=lambda bond: tuple(sorted(bond)))
            ),
            "LOCAL_SYCART_ATOMS=" + ",".join(str(atom + 1) for atom in sorted(zone)),
            "LOCAL_SYCART_FULL_REPLACEMENT=" + str(full_sycart).upper(),
        ),
    )
    result_b = np.asarray(build_gic_b_matrix(result, coordinates_angstrom=xyz).rows)
    if result_b.shape != previous_b.shape or _rank(result_b) != target:
        raise ValueError("local SYCART replacement changed vibrational rank")
    return result, tuple(sorted(zone))
