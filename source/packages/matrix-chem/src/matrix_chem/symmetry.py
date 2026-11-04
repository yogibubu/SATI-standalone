"""Molecular point-group detection and serialized symmetry operations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations, product
import re
from typing import Sequence

import numpy as np

from .average_atomic_masses import atomic_mass
from .geometry import MolecularGeometry
from .topology.elements import atomic_number


CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM = 5.0e-6
QUASI_SYMMETRY_MAX_DISTANCE_TOLERANCE_ANGSTROM = 3.0e-2


@dataclass(frozen=True)
class SymmetryOperation:
    label: str
    rotation: tuple[tuple[float, float, float], ...]
    permutation: tuple[int, ...]
    max_deviation: float


@dataclass(frozen=True)
class MolecularSymmetry:
    point_group: str
    operations: tuple[SymmetryOperation, ...]
    atom_classes: tuple[tuple[int, ...], ...]
    max_deviation: float
    mean_deviation: float
    orientation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    supported_subgroups: tuple[str, ...] = ()


def restrict_molecular_symmetry(
    symmetry: MolecularSymmetry,
    symmetry_group: str | None,
) -> MolecularSymmetry:
    """Return the frozen operation subgroup requested by SMITH/LINK.

    ORACLE normally stores the largest group recognized at the input geometry.
    A downstream calculation may deliberately request a subgroup (for example
    ``C3`` for a C3v Fe4S4 model).  This routine keeps the complete closed
    subgroup, including every power of the principal rotation, and updates
    atom classes accordingly.  It never invents operations that were not
    certified by ORACLE.
    """

    target = str(symmetry_group or "").strip().upper()
    source = str(symmetry.point_group or "").strip().upper()
    if not target or target == source:
        return symmetry
    operations = tuple(symmetry.operations)
    identity = next((op for op in operations if op.label.upper() == "E"), None)
    if identity is None:
        natoms = max(
            (len(operation.permutation) for operation in operations),
            default=sum(len(group) for group in symmetry.atom_classes),
        )
        identity = SymmetryOperation(
            label="E",
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            permutation=tuple(range(1, natoms + 1)),
            max_deviation=0.0,
        )
    selected: list[SymmetryOperation] = [identity]
    if target == "C1":
        return _replace_symmetry_subgroup(symmetry, target, selected)
    if target == "CS":
        selected.extend(
            op for op in operations
            if op.label.lower().startswith("sigma") and _is_reflection(op)
        )
        if len(selected) < 2:
            raise ValueError(f"cannot reduce {source} to {target}: no mirror operation")
        return _replace_symmetry_subgroup(symmetry, target, selected[:2])
    if target == "CI":
        inversion = next((op for op in operations if op.label.lower() == "i"), None)
        if inversion is None:
            raise ValueError(f"cannot reduce {source} to {target}: no inversion operation")
        return _replace_symmetry_subgroup(symmetry, target, selected + [inversion])
    if target == "C2":
        c2 = next((op for op in operations if re.fullmatch(r"C2[xyzXYZ]\^\d+", op.label)), None)
        if c2 is None:
            raise ValueError(f"cannot reduce {source} to {target}: no C2 principal rotation")
        return _replace_symmetry_subgroup(symmetry, target, selected + [c2])
    cyclic = re.fullmatch(r"C(\d+)([VHD]?)", target)
    if cyclic is None:
        raise ValueError(
            f"unsupported requested symmetry subgroup {target!r}; use C1, Cs, Ci, C2, Cn, Cnv or Cnh"
        )
    order = int(cyclic.group(1))
    suffix = cyclic.group(2)
    if order < 2:
        return _replace_symmetry_subgroup(symmetry, "C1", selected)
    by_axis: dict[str, dict[int, SymmetryOperation]] = {}
    for operation in operations:
        match = re.fullmatch(rf"C{order}([xyzXYZ])\^(\d+)", operation.label)
        if match is not None:
            by_axis.setdefault(match.group(1).lower(), {})[int(match.group(2))] = operation
    if not by_axis:
        raise ValueError(f"cannot reduce {source} to {target}: no C{order} principal rotation")
    axis, powers = max(
        by_axis.items(),
        key=lambda item: (len(item[1]), -"xyz".index(item[0])),
    )
    del axis  # the selected operation dictionary already fixes the axis
    required = set(range(1, order))
    if not required.issubset(powers):
        raise ValueError(f"cannot reduce {source} to {target}: incomplete C{order} operation set")
    selected.extend(powers[index] for index in sorted(required))
    if suffix == "V":
        mirrors = [
            op for op in operations
            if re.fullmatch(rf"sigma_v_{order}_.+", op.label, re.I)
        ]
        if len(mirrors) < order:
            raise ValueError(f"cannot reduce {source} to {target}: incomplete sigma-v operation set")
        selected.extend(mirrors[:order])
    elif suffix == "H":
        horizontal = next(
            (op for op in operations if op.label.lower() in {"sigma_xy", "sigma_h"}),
            None,
        )
        if horizontal is None:
            raise ValueError(f"cannot reduce {source} to {target}: no sigma-h operation")
        selected.append(horizontal)
    elif suffix == "D":
        raise ValueError(f"unsupported requested symmetry subgroup {target!r}")
    return _replace_symmetry_subgroup(symmetry, target, selected)


def _is_reflection(operation: SymmetryOperation) -> bool:
    return float(np.linalg.det(np.asarray(operation.rotation, dtype=float))) < 0.0


def _replace_symmetry_subgroup(
    symmetry: MolecularSymmetry,
    point_group: str,
    operations: Sequence[SymmetryOperation],
) -> MolecularSymmetry:
    classes: dict[int, list[int]] = {}
    natoms = max(
        (len(operation.permutation) for operation in operations),
        default=sum(len(group) for group in symmetry.atom_classes),
    )
    parent = list(range(1, natoms + 1))

    def find(item: int) -> int:
        while parent[item - 1] != item:
            parent[item - 1] = parent[parent[item - 1] - 1]
            item = parent[item - 1]
        return item

    for operation in operations:
        for atom, image in enumerate(operation.permutation, start=1):
            left, right = find(atom), find(int(image))
            if left != right:
                parent[right - 1] = left
    for atom in range(1, len(parent) + 1):
        classes.setdefault(find(atom), []).append(atom)
    return replace(
        symmetry,
        point_group=point_group,
        operations=tuple(operations),
        atom_classes=tuple(tuple(values) for values in classes.values()),
        supported_subgroups=(
            symmetry.supported_subgroups
            or tuple(sorted({"C1", point_group}))
        ),
    )


def available_symmetry_subgroups(
    symmetry: MolecularSymmetry,
) -> tuple[str, ...]:
    """List standard subgroup contracts recoverable from frozen operations."""

    labels = tuple(operation.label for operation in symmetry.operations)
    groups = {"C1", str(symmetry.point_group).strip().upper()}
    if any(label.lower() == "i" for label in labels):
        groups.add("Ci")
    if any(label.lower().startswith("sigma") for label in labels):
        groups.add("Cs")
    orders: set[int] = set()
    for label in labels:
        match = re.fullmatch(r"C(\d+)[xyzXYZ]\^(\d+)", label)
        if match:
            orders.add(int(match.group(1)))
    for order in sorted(orders):
        groups.add(f"C{order}")
        if sum(
            bool(re.fullmatch(rf"sigma_v_{order}_.+", label, re.I))
            for label in labels
        ) >= order:
            groups.add(f"C{order}v")
        if any(
            label.lower() in {"sigma_xy", "sigma_h"} for label in labels
        ):
            groups.add(f"C{order}h")
    return tuple(sorted(groups, key=lambda value: (value != "C1", value)))


def cartesian_operation_matrix(
    rotation: np.ndarray | Sequence[Sequence[float]],
    source_to_target_zero_based: Sequence[int],
    *,
    natoms: int | None = None,
) -> np.ndarray:
    """Return the shared Cartesian polar-vector operation matrix.

    The mapping follows ORACLE's detector convention: source atom ``i`` maps
    to target atom ``j`` such that ``x_i`` matches ``R x_j``.  All SMITH and
    MORPHEUS symmetry projectors use this one block-matrix construction.
    """

    mapping = tuple(int(atom) for atom in source_to_target_zero_based)
    atom_count = len(mapping) if natoms is None else int(natoms)
    if atom_count < 0 or len(mapping) != atom_count:
        raise ValueError("Cartesian symmetry mapping size does not match natoms")
    if sorted(mapping) != list(range(atom_count)):
        raise ValueError("Cartesian symmetry mapping must be a zero-based permutation")
    matrix3 = np.asarray(rotation, dtype=float)
    if matrix3.shape != (3, 3) or not np.all(np.isfinite(matrix3)):
        raise ValueError("Cartesian symmetry rotation must be a finite 3-by-3 matrix")
    matrix = np.zeros((3 * atom_count, 3 * atom_count), dtype=float)
    for source_index, target_index in enumerate(mapping):
        matrix[
            3 * source_index : 3 * source_index + 3,
            3 * target_index : 3 * target_index + 3,
        ] = matrix3
    return matrix


@dataclass(frozen=True)
class SymmetryProjectionDiagnostics:
    status: str
    max_displacement_angstrom: float
    rms_displacement_angstrom: float
    noop_tolerance_angstrom: float = CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM


@dataclass(frozen=True)
class QuasiSymmetryAnalysis:
    """Strict and topology-qualified quasi-symmetry assignments.

    A relaxed assignment is admissible only when every Cartesian operation
    preserves the frozen primary graph and extends the operation permutations
    already recognized at the strict threshold.  This prevents a larger
    geometric matching tolerance from inventing connectivity or replacing an
    established subgroup by an unrelated one.
    """

    strict_symmetry: MolecularSymmetry
    proposed_symmetry: MolecularSymmetry
    recognition_tolerance_angstrom: float
    search_limit_angstrom: float
    topology_validated: bool

    @property
    def promoted(self) -> bool:
        return (
            self.proposed_symmetry.point_group != self.strict_symmetry.point_group
            or len(self.proposed_symmetry.operations) != len(self.strict_symmetry.operations)
        )


def symmetry_projection_diagnostics(
    original: MolecularGeometry,
    projected: MolecularGeometry,
    symmetry: MolecularSymmetry,
) -> SymmetryProjectionDiagnostics:
    displacement = np.linalg.norm(
        np.asarray(projected.coordinates_angstrom, dtype=float)
        - np.asarray(original.coordinates_angstrom, dtype=float),
        axis=1,
    )
    max_displacement = float(np.max(displacement)) if len(displacement) else 0.0
    rms_displacement = (
        float(np.sqrt(np.mean(displacement * displacement))) if len(displacement) else 0.0
    )
    point_group = symmetry.point_group.strip().upper()
    expected_order = _finite_point_group_order(point_group)
    if point_group in {"", "C1", "UNKNOWN"}:
        status = "NOT_APPLICABLE_C1"
    elif expected_order is not None and len(symmetry.operations) != expected_order:
        status = "SKIPPED_INCOMPLETE_GROUP"
    elif symmetry.max_deviation <= CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM:
        status = "NOT_REQUIRED_NUMERICAL_NOISE"
    else:
        status = "APPLIED"
    return SymmetryProjectionDiagnostics(
        status=status,
        max_displacement_angstrom=max_displacement,
        rms_displacement_angstrom=rms_displacement,
    )


def symmetrize_molecular_geometry(
    geometry: MolecularGeometry,
    symmetry: MolecularSymmetry,
    *,
    minimum_deviation_angstrom: float = CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM,
    force_projection: bool = False,
) -> MolecularGeometry:
    """Return the exact group-average geometry for an assigned point group."""
    point_group = symmetry.point_group.strip().upper()
    if point_group in {"", "C1", "UNKNOWN"}:
        return geometry
    expected_order = _finite_point_group_order(point_group)
    if expected_order is not None and len(symmetry.operations) != expected_order:
        # A group average is a projector only when it contains the complete
        # group.  Averaging an incomplete operation set makes the Cartesian
        # result depend on the platform-specific choice inside a nearly
        # degenerate inertia eigenspace (the historical spiro/D2 regression).
        return geometry
    if not force_projection and symmetry.max_deviation <= float(minimum_deviation_angstrom):
        # Do not turn few-microangstrom input/printing noise into a new
        # Cartesian reference.  Above this floor, accepted quasi-symmetry is
        # projected exactly by the complete finite-group average below.
        return geometry
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    weights = _molecular_masses(geometry.atoms)
    center = np.average(coords, axis=0, weights=weights)
    centered = coords - center
    projected = np.zeros_like(centered).reshape(-1)
    for operation in symmetry.operations:
        mapping = np.asarray([int(atom) - 1 for atom in operation.permutation], dtype=int)
        action = cartesian_operation_matrix(operation.rotation, mapping, natoms=len(coords))
        projected += action @ centered.reshape(-1)
    projected /= float(len(symmetry.operations))
    exact = projected.reshape((-1, 3)) + center
    exact[np.abs(exact) < 5.0e-13] = 0.0
    return MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=exact,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        fixed_parameters=geometry.fixed_parameters,
        metadata={**geometry.metadata, "cartesian_symmetrized_point_group": symmetry.point_group},
    )


def symmetrize_cartesian_gradient(
    gradient_hartree_per_bohr: np.ndarray,
    symmetry: MolecularSymmetry,
) -> np.ndarray:
    """Project a Cartesian gradient onto the totally symmetric subspace.

    The Cartesian gradient transforms as a polar vector under the molecular
    symmetry operations. Group-averaging it at every QM point removes
    numerical symmetry breaking before LINK transforms the gradient into
    SONIC coordinates or updates its Hessian model.
    """
    original = np.asarray(gradient_hartree_per_bohr, dtype=float)
    flattened = original.ndim == 1
    if flattened:
        if original.size % 3:
            raise ValueError("flattened Cartesian gradients must have length 3N")
        gradient = original.reshape((-1, 3))
    else:
        gradient = original
    if gradient.ndim != 2 or gradient.shape[1] != 3:
        raise ValueError("Cartesian gradients must have shape natoms x 3")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("Cartesian gradient contains non-finite values")
    point_group = symmetry.point_group.strip().upper()
    if point_group in {"", "C1", "UNKNOWN"}:
        return original.copy()
    expected_order = _finite_point_group_order(point_group)
    if expected_order is not None and len(symmetry.operations) != expected_order:
        return original.copy()
    projected = np.zeros_like(gradient)
    for operation in symmetry.operations:
        rotation = np.asarray(operation.rotation, dtype=float)
        mapping = np.asarray([int(atom) - 1 for atom in operation.permutation], dtype=int)
        if mapping.shape != (gradient.shape[0],):
            raise ValueError("symmetry operation has incompatible atom mapping")
        inverse_mapping = np.argsort(mapping)
        projected += gradient[inverse_mapping] @ rotation
    projected /= float(len(symmetry.operations))
    return projected.reshape(-1) if flattened else projected


def canonicalize_molecular_geometry_by_symmetry(
    geometry: MolecularGeometry,
    *,
    distance_tolerance: float,
    inertia_tolerance: float,
    max_rotation_order: int,
    topology_bonds: Sequence[tuple[int, int]] | None = None,
    topology_edge_labels: Mapping[tuple[int, int], Hashable] | None = None,
    coordinate_decimals: int = 10,
    max_cycles: int = 4,
    convergence_tolerance_angstrom: float = 1.0e-12,
) -> tuple[MolecularGeometry, MolecularSymmetry, MolecularSymmetry]:
    """Return the general ORACLE Cartesian symmetry projection.

    The first symmetry object records the tolerance-based assignment made from
    the centered, principal-axis geometry.  The second records the operations
    of the final canonical geometry.  Reorienting and reprojecting removes the
    small principal-axis drift introduced by a single group average, including
    for symmetric tops whose perpendicular inertia axes are degenerate.
    """

    decimals = int(coordinate_decimals)
    if decimals < 0:
        raise ValueError("coordinate_decimals must be non-negative")
    cycles = max(1, int(max_cycles))
    if topology_edge_labels is not None and topology_bonds is None:
        raise ValueError("topology_edge_labels require topology_bonds")
    bonds = (
        None
        if topology_bonds is None
        else _canonical_topology_bonds(topology_bonds, natoms=len(geometry.atoms))
    )
    edge_labels = (
        None
        if topology_edge_labels is None or bonds is None
        else _canonical_topology_edge_labels(
            topology_edge_labels,
            topology_bonds=bonds,
        )
    )

    def admissible_supergroup(
        subgroup: MolecularSymmetry,
        candidate: MolecularSymmetry,
    ) -> bool:
        return _symmetry_operations_extend(subgroup, candidate) and (
            bonds is None
            or _symmetry_operations_preserve_topology(
                candidate,
                topology_bonds=bonds,
                topology_edge_labels=edge_labels,
                natoms=len(geometry.atoms),
            )
        )

    current = orient_molecular_geometry_to_principal_axes(
        geometry,
        inertia_tolerance=float(inertia_tolerance),
    )
    assigned = analyze_molecular_symmetry(
        current,
        distance_tolerance=float(distance_tolerance),
        inertia_tolerance=float(inertia_tolerance),
        max_rotation_order=int(max_rotation_order),
    )
    point_group = assigned.point_group
    final_symmetry = assigned
    current_coordinates = np.asarray(current.coordinates_angstrom, dtype=float)
    requires_axis_refinement = _has_degenerate_inertia(
        current.atoms,
        current_coordinates,
        tolerance=float(inertia_tolerance),
    ) and not _fully_degenerate_inertia(
        current.atoms,
        current_coordinates,
        tolerance=float(inertia_tolerance),
    )
    refinement_cycles = cycles if requires_axis_refinement else 1
    for _cycle in range(refinement_cycles):
        projected = symmetrize_molecular_geometry(
            current,
            final_symmetry,
            minimum_deviation_angstrom=CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM,
        )
        if not requires_axis_refinement:
            current = projected
            break
        canonical = (
            orient_molecular_geometry_to_principal_axes(
                projected,
                inertia_tolerance=float(inertia_tolerance),
            )
        )
        refined = analyze_molecular_symmetry(
            canonical,
            distance_tolerance=float(distance_tolerance),
            inertia_tolerance=float(inertia_tolerance),
            max_rotation_order=int(max_rotation_order),
        )
        if refined.point_group != point_group:
            if not admissible_supergroup(final_symmetry, refined):
                # Refinement may retain the assigned group or reveal a
                # topology-preserving supergroup.  It must never replace the
                # group by an unrelated one or lower the assigned symmetry.
                current = projected
                break
            point_group = refined.point_group
        refined_geometry = symmetrize_molecular_geometry(
            canonical,
            refined,
            minimum_deviation_angstrom=0.0,
        )
        displacement = np.linalg.norm(
            refined_geometry.coordinates_angstrom - canonical.coordinates_angstrom,
            axis=1,
        )
        current = refined_geometry
        final_symmetry = refined
        if (
            float(np.max(displacement, initial=0.0))
            <= float(convergence_tolerance_angstrom)
            and refined.max_deviation <= float(convergence_tolerance_angstrom)
        ):
            break

    rounded_coordinates = np.round(current.coordinates_angstrom, decimals=decimals)
    rounded_coordinates[np.abs(rounded_coordinates) < 0.5 * 10.0 ** (-decimals)] = 0.0
    rounded = MolecularGeometry(
        atoms=current.atoms,
        coordinates_angstrom=rounded_coordinates,
        comment=current.comment,
        source_format=current.source_format,
        source_path=current.source_path,
        charge=current.charge,
        multiplicity=current.multiplicity,
        fixed_parameters=current.fixed_parameters,
        metadata={
            **current.metadata,
            "cartesian_symmetry_canonicalized": True,
            "cartesian_coordinate_decimals": decimals,
        },
    )
    serialized_symmetry = analyze_molecular_symmetry(
        rounded,
        distance_tolerance=float(distance_tolerance),
        inertia_tolerance=float(inertia_tolerance),
        max_rotation_order=int(max_rotation_order),
    )
    if serialized_symmetry.point_group != point_group:
        if not admissible_supergroup(final_symmetry, serialized_symmetry):
            raise ValueError(
                "Cartesian coordinate serialization changed the assigned point group: "
                f"{point_group} -> {serialized_symmetry.point_group}"
            )
        promoted = symmetrize_molecular_geometry(
            rounded,
            serialized_symmetry,
            minimum_deviation_angstrom=0.0,
            force_projection=True,
        )
        promoted_coordinates = np.round(
            promoted.coordinates_angstrom,
            decimals=decimals,
        )
        promoted_coordinates[
            np.abs(promoted_coordinates) < 0.5 * 10.0 ** (-decimals)
        ] = 0.0
        rounded = MolecularGeometry(
            atoms=promoted.atoms,
            coordinates_angstrom=promoted_coordinates,
            comment=promoted.comment,
            source_format=promoted.source_format,
            source_path=promoted.source_path,
            charge=promoted.charge,
            multiplicity=promoted.multiplicity,
            fixed_parameters=promoted.fixed_parameters,
            metadata=promoted.metadata,
        )
        point_group = serialized_symmetry.point_group
        serialized_symmetry = analyze_molecular_symmetry(
            rounded,
            distance_tolerance=float(distance_tolerance),
            inertia_tolerance=float(inertia_tolerance),
            max_rotation_order=int(max_rotation_order),
        )
        if serialized_symmetry.point_group != point_group:
            raise ValueError(
                "Cartesian coordinate serialization did not retain the promoted "
                f"point group: {point_group} -> {serialized_symmetry.point_group}"
            )
    final_symmetry = serialized_symmetry
    return rounded, assigned, final_symmetry


def _symmetry_operations_extend(
    subgroup: MolecularSymmetry,
    candidate: MolecularSymmetry,
    *,
    rotation_tolerance: float = 1.0e-4,
) -> bool:
    """Return whether ``candidate`` contains every operation of ``subgroup``."""

    subgroup_order = _finite_point_group_order(subgroup.point_group)
    candidate_order = _finite_point_group_order(candidate.point_group)
    if (
        subgroup_order is None
        or candidate_order is None
        or len(subgroup.operations) != subgroup_order
        or len(candidate.operations) != candidate_order
        or candidate_order <= subgroup_order
    ):
        return False
    for operation in subgroup.operations:
        rotation = np.asarray(operation.rotation, dtype=float)
        if not any(
            other.permutation == operation.permutation
            and np.allclose(
                np.asarray(other.rotation, dtype=float),
                rotation,
                rtol=0.0,
                atol=float(rotation_tolerance),
            )
            for other in candidate.operations
        ):
            return False
    return True


def orient_molecular_geometry_to_principal_axes(
    geometry: MolecularGeometry,
    *,
    inertia_tolerance: float = 1.0e-3,
) -> MolecularGeometry:
    """Center a molecular geometry at its mass centre and orient its principal axes.

    This is the canonical Cartesian frame used by ORACLE before point-group
    assignment.  The atom order and all non-Cartesian metadata are preserved.
    """

    coordinates, frame = orient_coords_with_frame(
        geometry.coordinates_angstrom,
        weights=_molecular_masses(geometry.atoms),
        degeneracy_tolerance=float(inertia_tolerance),
    )
    coordinates[np.abs(coordinates) < 5.0e-13] = 0.0
    return MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=coordinates,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        fixed_parameters=geometry.fixed_parameters,
        metadata={
            **geometry.metadata,
            "cartesian_center": "CENTER_OF_MASS",
            "cartesian_orientation": "PRINCIPAL_INERTIA_AXES",
            "cartesian_orientation_frame": tuple(
                tuple(float(value) for value in row) for row in frame
            ),
        },
    )


def _molecular_masses(symbols: tuple[str, ...] | list[str]) -> np.ndarray:
    return np.asarray(
        [atomic_mass(atomic_number(symbol) or 1) for symbol in symbols],
        dtype=float,
    )


def _finite_point_group_order(point_group: str) -> int | None:
    label = point_group.strip().upper()
    if label == "C1":
        return 1
    if label == "CI" or label == "CS":
        return 2
    match = re.fullmatch(r"C(\d+)", label)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"C(\d+)[VH]", label)
    if match:
        return 2 * int(match.group(1))
    match = re.fullmatch(r"D(\d+)", label)
    if match:
        return 2 * int(match.group(1))
    match = re.fullmatch(r"D(\d+)[DH]", label)
    if match:
        return 4 * int(match.group(1))
    match = re.fullmatch(r"S(\d+)", label)
    if match and int(match.group(1)) % 2 == 0:
        return int(match.group(1))
    if label == "T":
        return 12
    if label in {"TD", "TH"}:
        return 24
    if label == "O":
        return 24
    if label == "OH":
        return 48
    if label == "I":
        return 60
    if label == "IH":
        return 120
    return None


def dihedral_operation_label_from_matrix(
    matrix: np.ndarray,
    principal_order: int,
) -> str | None:
    """Return the canonical label of a proper ``D_n`` operation.

    The matrix must be expressed in the principal-axis frame, with the
    highest-order axis along ``z``.  This matrix-based classifier is shared by
    ORACLE's complete-operation recovery and SMITH's character construction,
    so operation labels cannot drift between molecular perception and SALC
    generation.
    """

    n = int(principal_order)
    arr = np.asarray(matrix, dtype=float)
    if n < 2 or arr.shape != (3, 3):
        return None
    if not np.allclose(arr.T @ arr, np.eye(3), atol=1.0e-7):
        return None
    if float(np.linalg.det(arr)) < 0.0:
        return None
    if np.allclose(arr, np.eye(3), atol=1.0e-7):
        return "E"
    if not (
        np.allclose(arr[2, :2], 0.0, atol=1.0e-7)
        and np.allclose(arr[:2, 2], 0.0, atol=1.0e-7)
    ):
        return None
    if arr[2, 2] > 0.0:
        angle = np.arctan2(float(arr[1, 0]), float(arr[0, 0]))
        power = int(round((angle % (2.0 * np.pi)) * n / (2.0 * np.pi))) % n
        if power == 0:
            return "E"
        if n % 2 == 0 and power == n // 2:
            return "C2z"
        return f"C{n}z^{power}"
    if abs(float(np.trace(arr)) + 1.0) > 1.0e-7:
        return None
    values, vectors = np.linalg.eig(arr)
    axis_index = int(np.argmin(np.abs(values - 1.0)))
    axis = np.real(vectors[:, axis_index])
    if abs(float(axis[2])) > 1.0e-6:
        return None
    angle = np.arctan2(float(axis[1]), float(axis[0])) % np.pi
    index = int(round(angle * n / np.pi)) % n
    return f"C2_xy_{n}_{index}"


def _finite_operation_set_is_closed(
    elements: Sequence[tuple[str, np.ndarray, float]],
    permutations: Sequence[tuple[int, ...]],
) -> bool:
    """Return whether stored matrices and atom maps form a finite group."""

    if len(elements) != len(permutations) or not elements:
        return False
    records = tuple(
        (tuple(int(atom) for atom in permutation), np.asarray(element[1], dtype=float))
        for element, permutation in zip(elements, permutations)
    )
    keys = {permutation for permutation, _matrix in records}
    for left_permutation, _left_matrix in records:
        for right_permutation, _right_matrix in records:
            product_permutation = tuple(
                right_permutation[atom] for atom in left_permutation
            )
            if product_permutation not in keys:
                return False
    return True


def analyze_molecular_symmetry(
    geometry: MolecularGeometry,
    *,
    distance_tolerance: float,
    inertia_tolerance: float,
    max_rotation_order: int,
) -> MolecularSymmetry:
    symbols = list(geometry.atoms)
    # Keep ORACLE's established symmetry-classification weights.  Cartesian
    # centering/orientation is mass based, while point-set classification
    # remains tied to the frozen ORACLE contract.
    weights = np.array([atomic_number(symbol) or 1 for symbol in symbols], dtype=float)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    center = np.sum(coordinates * weights[:, None], axis=0) / max(
        float(np.sum(weights)), 1.0e-12
    )
    centered = coordinates - center
    if _fully_degenerate_inertia(symbols, centered, tolerance=inertia_tolerance):
        # A spherical top has no physical principal-axis frame.  Retaining the
        # input Cartesian frame avoids a platform-dependent eigenbasis; the
        # point-set operation enumeration below is covariant under a rigid
        # rotation of this frame.
        oriented = centered
        frame = np.eye(3, dtype=float)
    elif _has_degenerate_inertia(symbols, centered, tolerance=inertia_tolerance):
        # A symmetric top has a physically unique axis but no unique basis in
        # the perpendicular plane.  The legacy eigensolver frame can therefore
        # turn a complete Dnh group into an equally self-consistent Cnh
        # subgroup after a rigid rotation.  Anchor the degenerate plane to the
        # ordered molecular point set before enumerating operations.
        oriented, frame = orient_coords_with_frame(
            geometry.coordinates_angstrom,
            weights=weights,
            degeneracy_tolerance=inertia_tolerance,
        )
    else:
        oriented, frame = _legacy_orient_coords_with_frame(
            coordinates,
            weights=weights,
        )
    elements, atom_classes, permutations = symmetry_elements_from_geometry(
        symbols,
        oriented,
        tol=distance_tolerance,
        max_n=max_rotation_order,
        tol_H=distance_tolerance,
        ignore_isotopes=True,
        auto_max_n=True,
        inertia_tol=inertia_tolerance,
    )
    preliminary_group = group_label(
        elements,
        linear=is_linear(oriented, tol=distance_tolerance),
    )
    expected_order = _finite_point_group_order(preliminary_group)
    if expected_order is not None and len(elements) != expected_order:
        # The historical principal-axis convention preserves established
        # operation labels and SALC snapshots whenever it yields a complete
        # group.  Only a genuinely incomplete result activates the covariant
        # construction for a degenerate inertia subspace.
        oriented, frame = orient_coords_with_frame(
            geometry.coordinates_angstrom,
            weights=weights,
            degeneracy_tolerance=inertia_tolerance,
        )
        elements, atom_classes, permutations = symmetry_elements_from_geometry(
            symbols,
            oriented,
            tol=distance_tolerance,
            max_n=max_rotation_order,
            tol_H=distance_tolerance,
            ignore_isotopes=True,
            auto_max_n=True,
            inertia_tol=inertia_tolerance,
        )
    if not elements:
        elements = [("E", np.eye(3), 0.0)]
        permutations = [tuple(range(len(symbols)))]
        atom_classes = tuple((idx,) for idx in range(len(symbols)))
    point_group = group_label(
        elements,
        linear=is_linear(oriented, tol=distance_tolerance),
    )
    final_expected_order = _finite_point_group_order(point_group)
    if final_expected_order is not None and len(elements) != final_expected_order:
        # Conformer builders can differ by a few 1e-5 A across architectures.
        # Retry only an otherwise incomplete finite group with a narrow guard
        # band, rather than making every symmetry decision globally looser.
        guarded_tolerance = float(distance_tolerance) * 1.05
        elements, atom_classes, permutations = symmetry_elements_from_geometry(
            symbols,
            oriented,
            tol=guarded_tolerance,
            max_n=max_rotation_order,
            tol_H=guarded_tolerance,
            ignore_isotopes=True,
            auto_max_n=True,
            inertia_tol=inertia_tolerance,
        )
        point_group = group_label(
            elements,
            linear=is_linear(oriented, tol=guarded_tolerance),
        )
        final_expected_order = _finite_point_group_order(point_group)
    if final_expected_order is not None and len(elements) != final_expected_order:
        raise ValueError(
            "incomplete finite point-group operation set: "
            f"{point_group} requires {final_expected_order} operations, "
            f"but ORACLE recovered {len(elements)}"
        )
    if final_expected_order is not None and not _finite_operation_set_is_closed(
        elements,
        permutations,
    ):
        raise ValueError(
            "non-closed finite point-group operation set: "
            f"ORACLE operations for {point_group} fail matrix/permutation closure"
        )
    operations = tuple(
        SymmetryOperation(
            label=str(label),
            # Persist operations in the original Cartesian frame.  Consumers
            # must not need the arbitrary principal-axis frame to apply the
            # stored matrix to the enriched-XYZ coordinates.
            rotation=tuple(
                tuple(float(value) for value in row)
                for row in frame @ np.asarray(rotation, dtype=float) @ frame.T
            ),
            permutation=tuple(int(item) + 1 for item in permutation),
            max_deviation=float(max_deviation),
        )
        for (label, rotation, max_deviation), permutation in zip(elements, permutations)
    )
    result = MolecularSymmetry(
        point_group=point_group,
        operations=operations,
        atom_classes=tuple(tuple(int(atom) + 1 for atom in cls) for cls in atom_classes),
        max_deviation=float(max((op.max_deviation for op in operations), default=0.0)),
        mean_deviation=(
            float(np.mean([op.max_deviation for op in operations])) if operations else 0.0
        ),
        orientation=tuple(tuple(float(value) for value in row) for row in frame),
    )
    return replace(result, supported_subgroups=available_symmetry_subgroups(result))


def analyze_molecular_quasisymmetry(
    geometry: MolecularGeometry,
    *,
    distance_tolerance: float,
    inertia_tolerance: float,
    max_rotation_order: int,
    topology_bonds: Sequence[tuple[int, int]] | None,
    topology_edge_labels: Mapping[tuple[int, int], Hashable] | None = None,
    maximum_distance_tolerance: float = (
        QUASI_SYMMETRY_MAX_DISTANCE_TOLERANCE_ANGSTROM
    ),
) -> QuasiSymmetryAnalysis:
    """Return the strict assignment and any topology-qualified larger group.

    The relaxed search is deliberately bounded in Cartesian space.  It is not
    a replacement for structural refinement: it only recognizes a nearby
    exact-symmetry manifold when the complete point group is geometrically
    recoverable and every atom permutation is an automorphism of ORACLE's
    primary graph.  Without a primary graph, no relaxed promotion is made.
    """

    strict_tolerance = float(distance_tolerance)
    if strict_tolerance <= 0.0:
        raise ValueError("distance_tolerance must be positive")
    # The configured maximum bounds only the additional relaxed search.  A
    # caller may deliberately request a looser strict assignment (for example
    # for a raw SMILES embedding); in that case there is no wider
    # quasi-symmetry interval to inspect.
    search_limit = max(strict_tolerance, float(maximum_distance_tolerance))
    inertia_limit = float(inertia_tolerance)
    rotation_order = int(max_rotation_order)
    coordinate_array = np.asarray(geometry.coordinates_angstrom, dtype=float)
    geometry_key = (
        tuple(str(symbol) for symbol in geometry.atoms),
        coordinate_array.shape,
        coordinate_array.dtype.str,
        coordinate_array.tobytes(),
    )
    analysis_cache: dict[tuple[object, ...], MolecularSymmetry] = {}

    def analyze_at(tolerance: float) -> MolecularSymmetry:
        key = (geometry_key, float(tolerance), inertia_limit, rotation_order)
        cached = analysis_cache.get(key)
        if cached is None:
            cached = analyze_molecular_symmetry(
                geometry,
                distance_tolerance=float(tolerance),
                inertia_tolerance=inertia_limit,
                max_rotation_order=rotation_order,
            )
            analysis_cache[key] = cached
        return cached

    strict = analyze_at(strict_tolerance)
    if topology_edge_labels is not None and topology_bonds is None:
        raise ValueError("topology_edge_labels require topology_bonds")
    natoms = len(geometry.atoms)
    bonds = (
        _canonical_topology_bonds(topology_bonds, natoms=natoms)
        if topology_bonds is not None
        else None
    )
    edge_labels = (
        _canonical_topology_edge_labels(topology_edge_labels, topology_bonds=bonds)
        if topology_edge_labels is not None and bonds is not None
        else None
    )
    strict_topology_validated = bool(
        bonds is not None
        and _symmetry_operations_preserve_topology(
            strict,
            topology_bonds=bonds,
            topology_edge_labels=edge_labels,
            natoms=natoms,
        )
    )
    unchanged = QuasiSymmetryAnalysis(
        strict_symmetry=strict,
        proposed_symmetry=strict,
        recognition_tolerance_angstrom=strict_tolerance,
        search_limit_angstrom=search_limit,
        topology_validated=strict_topology_validated,
    )
    if bonds is None or not strict_topology_validated or search_limit <= strict_tolerance:
        return unchanged

    relaxed = analyze_at(search_limit)
    if relaxed.point_group.strip().upper() in {"", "C1", "UNKNOWN"}:
        return unchanged
    if (
        relaxed.point_group == strict.point_group
        and len(relaxed.operations) == len(strict.operations)
    ):
        return unchanged

    if not _symmetry_operations_preserve_topology(
        relaxed,
        topology_bonds=bonds,
        topology_edge_labels=edge_labels,
        natoms=natoms,
    ):
        return unchanged
    strict_permutations = {
        tuple(int(atom) for atom in operation.permutation)
        for operation in strict.operations
    }
    relaxed_permutations = {
        tuple(int(atom) for atom in operation.permutation)
        for operation in relaxed.operations
    }
    if not strict_permutations.issubset(relaxed_permutations):
        return unchanged

    # Recover the smallest reproducible tolerance that contains the complete
    # candidate group.  This value, rather than the search ceiling, is frozen
    # in diagnostics and used for the exact Cartesian group projection.
    recognition_tolerance = max(
        strict_tolerance,
        float(relaxed.max_deviation)
        + max(1.0e-10, 1.0e-8 * float(relaxed.max_deviation)),
    )
    recognition_tolerance = min(recognition_tolerance, search_limit)
    proposed = analyze_at(recognition_tolerance)
    if (
        proposed.point_group != relaxed.point_group
        or len(proposed.operations) != len(relaxed.operations)
    ):
        proposed = relaxed
        recognition_tolerance = search_limit
    if not _symmetry_operations_preserve_topology(
        proposed,
        topology_bonds=bonds,
        topology_edge_labels=edge_labels,
        natoms=natoms,
    ):
        return unchanged
    return QuasiSymmetryAnalysis(
        strict_symmetry=strict,
        proposed_symmetry=proposed,
        recognition_tolerance_angstrom=recognition_tolerance,
        search_limit_angstrom=search_limit,
        topology_validated=True,
    )


def _canonical_topology_bonds(
    topology_bonds: Sequence[tuple[int, int]],
    *,
    natoms: int,
) -> frozenset[tuple[int, int]]:
    output: set[tuple[int, int]] = set()
    for raw_left, raw_right in topology_bonds:
        left, right = int(raw_left), int(raw_right)
        if left == right or min(left, right) < 0 or max(left, right) >= int(natoms):
            raise ValueError("topology bond is outside the molecular atom range")
        output.add(tuple(sorted((left, right))))
    return frozenset(output)


def _canonical_topology_edge_labels(
    topology_edge_labels: Mapping[tuple[int, int], Hashable],
    *,
    topology_bonds: frozenset[tuple[int, int]],
) -> dict[tuple[int, int], Hashable]:
    output: dict[tuple[int, int], Hashable] = {}
    for raw_pair, label in topology_edge_labels.items():
        if len(raw_pair) != 2:
            raise ValueError("topology edge-label key must contain two atom indices")
        pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
        if pair not in topology_bonds:
            raise ValueError("topology edge label is assigned to a non-bonded atom pair")
        output[pair] = label
    if set(output) != set(topology_bonds):
        raise ValueError("topology edge labels must cover every primary bond")
    return output


def _symmetry_operations_preserve_topology(
    symmetry: MolecularSymmetry,
    *,
    topology_bonds: frozenset[tuple[int, int]],
    topology_edge_labels: Mapping[tuple[int, int], Hashable] | None,
    natoms: int,
) -> bool:
    for operation in symmetry.operations:
        permutation = tuple(int(atom) - 1 for atom in operation.permutation)
        if len(permutation) != natoms or tuple(sorted(permutation)) != tuple(range(natoms)):
            return False
        mapped = frozenset(
            tuple(sorted((permutation[left], permutation[right])))
            for left, right in topology_bonds
        )
        if mapped != topology_bonds:
            return False
        if topology_edge_labels is not None:
            for pair, label in topology_edge_labels.items():
                image = tuple(sorted((permutation[pair[0]], permutation[pair[1]])))
                if topology_edge_labels.get(image) != label:
                    return False
    return True


def symmetry_section_lines(
    symmetry: MolecularSymmetry,
    *,
    thresholds,
    projection: SymmetryProjectionDiagnostics | None = None,
    cartesian_symmetry_status: str | None = None,
    symmetrization_required_threshold_angstrom: float | None = None,
    cartesian_symmetrization_decision: str | None = None,
    input_symmetry_max_deviation_angstrom: float | None = None,
    proposed_point_group: str | None = None,
) -> list[str]:
    lines = [
        "SCHEMA oracle.xyz.symmetry.v1",
        f"POINT_GROUP {symmetry.point_group}",
        f"OPERATION_COUNT {len(symmetry.operations)}",
        f"MAX_OPERATION_DEVIATION_ANGSTROM {symmetry.max_deviation:.12g}",
        f"MEAN_OPERATION_DEVIATION_ANGSTROM {symmetry.mean_deviation:.12g}",
        "SUPPORTED_SUBGROUPS "
        + ",".join(symmetry.supported_subgroups or available_symmetry_subgroups(symmetry)),
        f"THRESHOLD_DISTANCE_ANGSTROM {thresholds.distance_angstrom:.12g}",
        f"THRESHOLD_INERTIA_RELATIVE {thresholds.inertia_relative:.12g}",
        f"MAX_ROTATION_ORDER {thresholds.max_rotation_order}",
    ]
    if projection is not None:
        lines.extend(
            (
                f"CARTESIAN_PROJECTION_STATUS {projection.status}",
                "CARTESIAN_PROJECTION_MAX_DISPLACEMENT_ANGSTROM "
                f"{projection.max_displacement_angstrom:.12g}",
                "CARTESIAN_PROJECTION_RMS_DISPLACEMENT_ANGSTROM "
                f"{projection.rms_displacement_angstrom:.12g}",
                "CARTESIAN_PROJECTION_NOOP_TOLERANCE_ANGSTROM "
                f"{projection.noop_tolerance_angstrom:.12g}",
            )
        )
    if cartesian_symmetry_status is not None:
        lines.append(f"CARTESIAN_SYMMETRY_STATUS {cartesian_symmetry_status}")
    if symmetrization_required_threshold_angstrom is not None:
        lines.append(
            "CARTESIAN_SYMMETRIZATION_REQUIRED_THRESHOLD_ANGSTROM "
            f"{float(symmetrization_required_threshold_angstrom):.12g}"
        )
    if cartesian_symmetrization_decision is not None:
        lines.append(
            f"CARTESIAN_SYMMETRIZATION_DECISION {cartesian_symmetrization_decision}"
        )
    if input_symmetry_max_deviation_angstrom is not None:
        lines.append(
            "INPUT_MAX_OPERATION_DEVIATION_ANGSTROM "
            f"{float(input_symmetry_max_deviation_angstrom):.12g}"
        )
    if proposed_point_group is not None:
        lines.append(f"PROPOSED_POINT_GROUP {proposed_point_group}")
    lines.append("[OPERATIONS]")
    for idx, operation in enumerate(symmetry.operations, start=1):
        matrix = ",".join(f"{value:.12g}" for row in operation.rotation for value in row)
        permutation = ",".join(str(atom) for atom in operation.permutation)
        lines.append(
            f"{idx} LABEL={operation.label} "
            f"MAX_DEVIATION={operation.max_deviation:.12g} "
            f"PERMUTATION={permutation} MATRIX={matrix}"
        )
    lines.append("[ATOM_CLASSES]")
    if symmetry.atom_classes:
        for idx, atoms in enumerate(symmetry.atom_classes, start=1):
            lines.append(f"{idx} ATOMS=" + ",".join(str(atom) for atom in atoms))
    else:
        lines.append("NONE")
    return lines


def orient_coords(coords, weights=None):
    oriented, _frame = orient_coords_with_frame(coords, weights=weights)
    return oriented


def _legacy_orient_coords_with_frame(coords, weights=None):
    """Return the established principal-axis frame used by frozen snapshots."""
    x = np.asarray(coords, dtype=float)
    w = np.ones(len(x), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    center = np.sum(x * w[:, None], axis=0) / max(float(np.sum(w)), 1.0e-12)
    x = x - center
    squared_radii = np.einsum("ij,ij->i", x, x)
    inertia = float(w @ squared_radii) * np.eye(3) - np.einsum("i,ij,ik->jk", w, x, x)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    frame = eigenvectors[:, np.argsort(eigenvalues)]
    if np.linalg.det(frame) < 0.0:
        frame[:, -1] *= -1.0
    return x @ frame, frame


def orient_coords_with_frame(coords, weights=None, degeneracy_tolerance=1.0e-8):
    x = np.array(coords, dtype=float)
    w = np.ones(len(x), dtype=float) if weights is None else np.array(weights, dtype=float)
    center = np.sum(x * w[:, None], axis=0) / max(float(np.sum(w)), 1.0e-12)
    x = x - center
    squared_radii = np.einsum("ij,ij->i", x, x)
    inertia = float(w @ squared_radii) * np.eye(3) - np.einsum("i,ij,ik->jk", w, x, x)
    evals, evecs = np.linalg.eigh(inertia)
    order = np.argsort(evals)
    evals = evals[order]
    frame = evecs[:, order]
    scale = max(float(np.max(np.abs(evals))), 1.0e-12)
    gap01 = abs(float(evals[0] - evals[1])) / scale
    gap12 = abs(float(evals[1] - evals[2])) / scale
    tolerance = float(degeneracy_tolerance)
    if gap01 <= tolerance and gap12 <= tolerance:
        # A spherical top has no inertia-defined axes.  LAPACK is therefore
        # free to return any orthonormal eigenbasis, and that arbitrary choice
        # is observably different across architectures.  Build a covariant
        # right-handed frame from the first ordered pair of centered atom
        # vectors that fixes two independent directions instead.  Atom order
        # is part of the molecular input contract, so this choice is stable
        # without depending on floating-point tie breaking.
        frame = _ordered_spherical_top_frame(x)
    elif (gap01 <= tolerance) != (gap12 <= tolerance):
        # A symmetric top has one unique inertia axis and an arbitrary basis
        # in the perpendicular degenerate plane.  Anchor that plane to an atom
        # vector, making the frame covariant under rigid rotations.
        unique_index = 2 if gap01 <= tolerance else 0
        axis = np.asarray(frame[:, unique_index], dtype=float)
        axial_projections = x @ axis
        axial_nonzero = np.flatnonzero(np.abs(axial_projections) > 1.0e-12)
        if len(axial_nonzero) and axial_projections[int(axial_nonzero[0])] < 0.0:
            axis *= -1.0
        projections = x - np.outer(x @ axis, axis)
        norms = np.linalg.norm(projections, axis=1)
        nonzero = np.flatnonzero(norms > 1.0e-12)
        anchor_index = int(nonzero[0]) if len(nonzero) else 0
        if len(nonzero):
            # Atom order is already part of the molecular contract.  Using
            # its first off-axis vector is deterministic in a degenerate
            # plane and, unlike a largest-radius choice, remains stable when
            # several symmetry-equivalent atoms tie numerically.  Treat the
            # anchor as y so conventional axis-aligned planar inputs retain
            # their historical x/y irrep naming.
            ey = projections[anchor_index] / norms[anchor_index]
            ez = axis
            ex = np.cross(ey, ez)
            ex /= max(float(np.linalg.norm(ex)), 1.0e-12)
            ey = np.cross(ez, ex)
            ey /= max(float(np.linalg.norm(ey)), 1.0e-12)
            frame = np.column_stack((ex, ey, ez))
    elif gap01 > tolerance and gap12 > tolerance:
        # Fix eigenvector signs from the first nonzero atomic projection.  This
        # removes platform-dependent sign flips for asymmetric tops.
        frame = frame.copy()
        for column in range(3):
            projections = x @ frame[:, column]
            nonzero = np.flatnonzero(np.abs(projections) > 1.0e-12)
            if len(nonzero) and projections[int(nonzero[0])] < 0.0:
                frame[:, column] *= -1.0
    if np.linalg.det(frame) < 0.0:
        frame[:, -1] *= -1.0
    return x @ frame, frame


def _ordered_spherical_top_frame(points: np.ndarray) -> np.ndarray:
    """Return a rigid-covariant frame for a fully degenerate point set."""
    x = np.asarray(points, dtype=float)
    radii = np.linalg.norm(x, axis=1)
    scale = max(float(np.max(radii, initial=0.0)), 1.0)
    threshold = 1.0e-12 * scale
    for first in range(len(x)):
        for second in range(first + 1, len(x)):
            difference = x[first] - x[second]
            difference_norm = float(np.linalg.norm(difference))
            if difference_norm <= threshold:
                continue
            ex = difference / difference_norm
            bisector = -(x[first] + x[second])
            bisector -= float(np.dot(bisector, ex)) * ex
            bisector_norm = float(np.linalg.norm(bisector))
            if bisector_norm <= threshold:
                continue
            ez = bisector / bisector_norm
            ey = np.cross(ez, ex)
            ey_norm = float(np.linalg.norm(ey))
            if ey_norm <= threshold:
                continue
            ey /= ey_norm
            ez = np.cross(ex, ey)
            ez /= max(float(np.linalg.norm(ez)), threshold)
            return np.column_stack((ex, ey, ez))
    raise ValueError("fully degenerate molecular geometry does not define an ordered 3D frame")


def is_linear(coords, tol=1.0e-3):
    x = np.array(coords, dtype=float)
    inertia = np.zeros((3, 3), dtype=float)
    for vec in x:
        inertia += np.dot(vec, vec) * np.eye(3) - np.outer(vec, vec)
    return bool(np.linalg.eigvalsh(inertia)[0] < tol)


def symmetry_elements_from_geometry(
    symbols,
    coords_oriented,
    tol=1.0e-3,
    max_n=6,
    tol_H=None,
    ignore_isotopes=False,
    auto_max_n=False,
    inertia_tol=1.0e-3,
):
    coords = np.asarray(coords_oriented, dtype=float)
    fully_degenerate = _fully_degenerate_inertia(symbols, coords, tolerance=inertia_tol)
    partly_degenerate = (
        not fully_degenerate
        and _has_degenerate_inertia(symbols, coords, tolerance=inertia_tol)
    )
    elements, atom_classes, permutations = _symmetry_elements_in_frame(
        symbols,
        coords,
        tol=tol,
        max_n=max_n,
        tol_H=tol_H,
        ignore_isotopes=ignore_isotopes,
        auto_max_n=auto_max_n,
        inertia_tol=inertia_tol,
    )
    nmax, axis = _highest_cn_axis([element[0] for element in elements])
    if nmax >= 3 and axis in {"x", "y"}:
        alignment = _axis_to_z_alignment(axis)
        aligned_elements, aligned_classes, aligned_permutations = _symmetry_elements_in_frame(
            symbols,
            coords @ alignment,
            tol=tol,
            max_n=max_n,
            tol_H=tol_H,
            ignore_isotopes=ignore_isotopes,
            auto_max_n=auto_max_n,
            inertia_tol=inertia_tol,
        )
        aligned_nmax, _aligned_axis = _highest_cn_axis([element[0] for element in aligned_elements])
        if aligned_nmax >= nmax and len(aligned_elements) >= len(elements):
            elements = tuple(
                (
                    label,
                    alignment @ np.asarray(rotation, dtype=float) @ alignment.T,
                    max_deviation,
                )
                for label, rotation, max_deviation in aligned_elements
            )
            atom_classes = aligned_classes
            permutations = aligned_permutations
    if partly_degenerate:
        direct = _symmetry_elements_from_point_set(
            symbols,
            coords,
            tol=tol,
            tol_H=tol_H,
            ignore_isotopes=ignore_isotopes,
        )
        # The unique axis of a symmetric top is inertia-defined, but the
        # perpendicular plane is not.  Enumerate the point-set operations once
        # and use an actual perpendicular C2 axis or vertical mirror normal to
        # fix that plane.  Re-running the established operation catalogue in
        # this symmetry-aligned frame preserves all historical labels while
        # recovering the complete Cnv/Dn/Dnh/Dnd operation set for an
        # arbitrary rigid orientation.
        if len(direct[0]) > len(elements):
            alignment = _axial_degenerate_alignment_from_operations(direct[0])
            if alignment is not None:
                aligned_elements, aligned_classes, aligned_permutations = (
                    _symmetry_elements_in_frame(
                        symbols,
                        coords @ alignment,
                        tol=tol,
                        max_n=max_n,
                        tol_H=tol_H,
                        ignore_isotopes=ignore_isotopes,
                        auto_max_n=auto_max_n,
                        inertia_tol=inertia_tol,
                    )
                )
                if len(aligned_elements) >= len(direct[0]):
                    elements = tuple(
                        (
                            label,
                            alignment
                            @ np.asarray(rotation, dtype=float)
                            @ alignment.T,
                            max_deviation,
                        )
                        for label, rotation, max_deviation in aligned_elements
                    )
                    atom_classes = aligned_classes
                    permutations = aligned_permutations
    if fully_degenerate:
        direct = _symmetry_elements_from_point_set(
            symbols,
            coords,
            tol=tol,
            tol_H=tol_H,
            ignore_isotopes=ignore_isotopes,
        )
        # A spherical top has no canonical inertia frame.  Prefer point-set
        # enumeration whenever it recovers at least the same group so a
        # platform-dependent eigenbasis cannot select a different, but equally
        # complete, operation contract.
        if len(direct[0]) >= len(elements):
            elements, atom_classes, permutations = direct
    # Linear molecules represent Cinfv/Dinfh with a finite diagnostic sample
    # of axial operations.  That sample is intentionally not a finite group:
    # closing products of different sampled rotations would generate an
    # unbounded catalogue and is neither required nor mathematically valid.
    if is_linear(coords, tol=tol):
        return elements, _atom_classes(len(symbols), permutations), permutations
    elements, permutations = _complete_verified_operation_set(
        symbols,
        coords,
        elements,
        permutations,
        tol=tol,
        tol_H=tol_H,
        max_n=max_n,
        ignore_isotopes=ignore_isotopes,
    )
    return elements, _atom_classes(len(symbols), permutations), permutations


def _complete_verified_operation_set(
    symbols,
    coordinates: np.ndarray,
    elements: Sequence[tuple[str, np.ndarray, float]],
    permutations: Sequence[tuple[int, ...]],
    *,
    tol: float,
    tol_H: float | None,
    max_n: int,
    ignore_isotopes: bool,
) -> tuple[tuple[tuple[str, np.ndarray, float], ...], tuple[tuple[int, ...], ...]]:
    """Close accepted generators, retaining only products verified on atoms."""

    if len(elements) != len(permutations) or not elements:
        return tuple(elements), tuple(permutations)
    points = np.asarray(coordinates, dtype=float)
    radius = max(float(np.max(np.linalg.norm(points, axis=1), initial=0.0)), 1.0)
    special_gate = max(1.0e-8, 2.0 * float(max(tol, tol_H or tol)) / radius)

    def canonical_special(matrix: np.ndarray) -> np.ndarray:
        """Snap the two frame-independent special operations exactly."""

        value = np.asarray(matrix, dtype=float)
        if float(np.max(np.abs(value - np.eye(3)))) <= special_gate:
            return np.eye(3)
        if float(np.max(np.abs(value + np.eye(3)))) <= special_gate:
            return -np.eye(3)
        return value

    records = [
        (
            str(element[0]),
            canonical_special(np.asarray(element[1], dtype=float)),
            float(element[2]),
            tuple(int(atom) for atom in permutation),
        )
        for element, permutation in zip(elements, permutations)
    ]
    symbol_keys = tuple(
        _symmetry_symbol_key(symbol, ignore_isotopes=ignore_isotopes) for symbol in symbols
    )
    catalogue = candidate_ops(max_n=max_n)

    def known(_matrix: np.ndarray, permutation: tuple[int, ...]) -> bool:
        # The atomic permutation is a faithful identifier for every operation
        # of a finite, non-linear molecular point group.  Quasi-symmetric
        # Cartesian inputs give independently fitted operation matrices small
        # numerical inconsistencies; comparing both matrix and permutation
        # would therefore insert the same group element repeatedly during
        # closure.  Linear groups bypass this routine above because rotations
        # about their axis need not act faithfully on the nuclei.
        return any(
            permutation == existing_permutation
            for _label, _existing_matrix, _deviation, existing_permutation in records
        )

    while True:
        added = False
        snapshot = tuple(records)
        for _left_label, left_matrix, left_deviation, left_permutation in snapshot:
            for _right_label, right_matrix, right_deviation, right_permutation in snapshot:
                product_matrix = canonical_special(left_matrix @ right_matrix)
                product_permutation = tuple(
                    right_permutation[atom] for atom in left_permutation
                )
                if known(product_matrix, product_permutation):
                    continue
                transformed = points[np.asarray(product_permutation, dtype=int)] @ product_matrix.T
                deviations = np.linalg.norm(transformed - points, axis=1)
                # For a quasi-symmetric input, the residual of a product is
                # bounded by the sum of the two accepted operation residuals.
                # Permit that mathematically required accumulation, but never
                # widen the original per-atom tolerance by more than a factor
                # of two.  Exact geometries therefore retain the exact gate.
                composition_bound = float(left_deviation + right_deviation) + 1.0e-10
                accepted = all(
                    float(deviation)
                    <= min(
                        composition_bound,
                        2.0
                        * (
                            tol_H
                            if tol_H is not None and symbol_keys[index] in {"H", "Z1"}
                            else tol
                        ),
                    )
                    for index, deviation in enumerate(deviations)
                )
                if not accepted:
                    continue
                label = next(
                    (
                        candidate_label
                        for candidate_label, candidate_matrix in catalogue
                        if np.allclose(product_matrix, candidate_matrix, atol=1.0e-8)
                    ),
                    None,
                )
                if label is None:
                    # Products remain valid symmetry operations when the
                    # molecular principal frame is not one of the finite
                    # axis-aligned catalogue frames.  Preserve the exact
                    # verified matrix and give it an orientation-independent
                    # class label; downstream character construction also
                    # receives the matrix, so no physical information is
                    # inferred from this fallback name.
                    base_label = _automatic_operation_label(product_matrix)
                    existing_labels = {record[0] for record in records}
                    label = base_label
                    suffix = 1
                    while label in existing_labels:
                        suffix += 1
                        label = f"{base_label}_{suffix}"
                records.append(
                    (
                        str(label),
                        product_matrix,
                        float(np.max(deviations, initial=0.0)),
                        product_permutation,
                    )
                )
                added = True
                if len(records) > 120:
                    raise ValueError("finite point-group closure exceeded order 120")
        if not added:
            break
    return (
        tuple((label, matrix, deviation) for label, matrix, deviation, _mapping in records),
        tuple(mapping for _label, _matrix, _deviation, mapping in records),
    )


def _axial_degenerate_alignment_from_operations(
    elements: Sequence[tuple[str, np.ndarray, float]],
) -> np.ndarray | None:
    """Align a symmetric top's free plane to one of its exact operations."""

    z_axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
    in_plane: np.ndarray | None = None
    for _label, matrix, _deviation in elements:
        arr = np.asarray(matrix, dtype=float)
        if float(np.linalg.det(arr)) <= 0.0 or abs(float(np.trace(arr)) + 1.0) > 1.0e-7:
            continue
        values, vectors = np.linalg.eig(arr)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        norm = float(np.linalg.norm(axis))
        if norm > 1.0e-12 and abs(float(axis @ z_axis)) <= 1.0e-6:
            in_plane = axis / norm
            break
    if in_plane is None:
        for _label, matrix, _deviation in elements:
            arr = np.asarray(matrix, dtype=float)
            if float(np.linalg.det(arr)) >= 0.0 or abs(float(np.trace(arr)) - 1.0) > 1.0e-7:
                continue
            values, vectors = np.linalg.eig(arr)
            normal = np.real(vectors[:, int(np.argmin(np.abs(values + 1.0)))])
            norm = float(np.linalg.norm(normal))
            if norm > 1.0e-12 and abs(float(normal @ z_axis)) <= 1.0e-6:
                in_plane = normal / norm
                break
    if in_plane is None:
        return None
    in_plane = in_plane.copy()
    in_plane[2] = 0.0
    norm = float(np.linalg.norm(in_plane))
    if norm <= 1.0e-12:
        return None
    in_plane /= norm
    nonzero = np.flatnonzero(np.abs(in_plane) > 1.0e-12)
    if len(nonzero) and in_plane[int(nonzero[0])] < 0.0:
        in_plane *= -1.0
    second = np.cross(z_axis, in_plane)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    return np.column_stack((in_plane, second, z_axis))


def _fully_degenerate_inertia(symbols, coords: np.ndarray, *, tolerance: float) -> bool:
    if len(coords) < 4:
        return False
    weights = np.asarray([atomic_number(symbol) or 1 for symbol in symbols], dtype=float)
    points = np.asarray(coords, dtype=float)
    squared_radii = np.einsum("ij,ij->i", points, points)
    inertia = float(weights @ squared_radii) * np.eye(3) - np.einsum(
        "i,ij,ik->jk", weights, points, points
    )
    eigenvalues = np.linalg.eigvalsh(inertia)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-12)
    return bool(float(np.ptp(eigenvalues)) / scale <= float(tolerance))


def _has_degenerate_inertia(symbols, coords: np.ndarray, *, tolerance: float) -> bool:
    """Return whether any principal-moment pair is degenerate."""
    if len(coords) < 3:
        return False
    weights = np.asarray([atomic_number(symbol) or 1 for symbol in symbols], dtype=float)
    points = np.asarray(coords, dtype=float)
    squared_radii = np.einsum("ij,ij->i", points, points)
    inertia = float(weights @ squared_radii) * np.eye(3) - np.einsum(
        "i,ij,ik->jk", weights, points, points
    )
    eigenvalues = np.linalg.eigvalsh(inertia)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-12)
    gaps = np.abs(np.diff(eigenvalues)) / scale
    return bool(np.any(gaps <= float(tolerance)))


def _symmetry_elements_from_point_set(
    symbols,
    coords: np.ndarray,
    *,
    tol: float,
    tol_H: float | None,
    ignore_isotopes: bool,
):
    """Recover all point-set isometries without choosing principal axes.

    A non-coplanar source triple fixes an orthogonal transformation uniquely.
    Candidate image triples are pruned by element, radius and Gram matrix, so
    the expensive full-geometry match is performed only for viable isometries.
    This path is used for spherical tops, whose inertia eigenvectors carry no
    orientation information.
    """
    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        return (), (), ()
    source_indices = _independent_point_triple(points)
    if source_indices is None:
        return (), (), ()
    source = points[list(source_indices)]
    source_symbols = [
        _symmetry_symbol_key(symbols[index], ignore_isotopes=ignore_isotopes)
        for index in source_indices
    ]
    match_symbols = [
        _symmetry_symbol_key(symbol, ignore_isotopes=ignore_isotopes) for symbol in symbols
    ]
    radii = np.linalg.norm(points, axis=1)
    max_radius = max(float(np.max(radii)), 1.0)
    candidates: list[tuple[int, ...]] = []
    for source_index, source_symbol in zip(source_indices, source_symbols):
        radius_tolerance = tol_H if tol_H is not None and source_symbol in {"H", "Z1"} else tol
        candidates.append(
            tuple(
                index
                for index, symbol in enumerate(match_symbols)
                if symbol == source_symbol
                and abs(float(radii[index] - radii[source_index])) <= float(radius_tolerance)
            )
        )
    gram = source @ source.T
    gram_tolerance = max(1.0e-10, 4.0 * max_radius * float(max(tol, tol_H or tol)))
    operations: list[tuple[str, np.ndarray, float]] = []
    mappings: list[tuple[int, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for target_indices in _gram_compatible_target_triples(
        points,
        candidates,
        gram=gram,
        tolerance=gram_tolerance,
    ):
        target = points[list(target_indices)]
        row_action = np.linalg.solve(source, target)
        rotation = row_action.T
        # XYZ serialization can perturb an otherwise exact spherical top by
        # a few 1e-8 in the inferred operation matrix.  Rejecting before the
        # physical Cartesian tolerance is applied turns a rotated Td/O_h
        # geometry into an incomplete subgroup.  Admit only a small
        # tolerance-scaled neighbourhood of O(3), project to its nearest
        # orthogonal matrix, and retain the existing full-geometry match as
        # the authoritative acceptance gate.
        orthogonality_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        )
        orthogonality_gate = max(1.0e-8, 8.0 * float(max(tol, tol_H or tol)) / max_radius)
        if orthogonality_error > orthogonality_gate:
            continue
        left, _singular_values, right = np.linalg.svd(rotation)
        rotation = left @ right
        mapping, max_deviation = _match_with_map(
            match_symbols,
            points,
            points @ rotation.T,
            tol,
            tol_H=tol_H,
        )
        if mapping is None:
            continue
        key = tuple(round(float(value), 10) for value in rotation.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        operations.append(("", rotation, float(max_deviation)))
        mappings.append(tuple(int(item) for item in mapping))
    ordered = sorted(
        zip(operations, mappings),
        key=lambda item: _automatic_operation_sort_key(item[0][1]),
    )
    counters: Counter[str] = Counter()
    labeled_operations: list[tuple[str, np.ndarray, float]] = []
    ordered_mappings: list[tuple[int, ...]] = []
    for (_label, rotation, max_deviation), mapping in ordered:
        base = _automatic_operation_label(rotation)
        if base not in {"E", "i"}:
            counters[base] += 1
            label = f"{base}_{counters[base]}"
        else:
            label = base
        labeled_operations.append((label, rotation, max_deviation))
        ordered_mappings.append(mapping)
    return (
        tuple(labeled_operations),
        _atom_classes(len(symbols), ordered_mappings),
        tuple(ordered_mappings),
    )


def _gram_compatible_target_triples(
    points: np.ndarray,
    candidates: Sequence[Sequence[int]],
    *,
    gram: np.ndarray,
    tolerance: float,
    batch_size: int = 262_144,
):
    """Yield Gram-compatible triples in ``itertools.product`` order.

    Spherical tops can have tens of thousands to millions of element/radius
    compatible image triples.  The former scalar loop entered ``np.allclose``
    once per triple.  Here each bounded batch is screened by compiled NumPy
    kernels while preserving exactly the original lexicographic candidate
    order.  Keeping the batches bounded avoids a cubic temporary-memory peak
    for larger point sets.
    """

    if len(candidates) != 3:
        return
    first, second, third = (
        np.asarray(candidate, dtype=np.intp) for candidate in candidates
    )
    second_third = int(len(second) * len(third))
    total = int(len(first) * second_third)
    if total == 0:
        return
    points_array = np.asarray(points, dtype=float)
    gram_array = np.asarray(gram, dtype=float)
    chunk = max(1, int(batch_size))
    for start in range(0, total, chunk):
        flat = np.arange(start, min(start + chunk, total), dtype=np.intp)
        first_position = flat // second_third
        remainder = flat - first_position * second_third
        second_position = remainder // len(third)
        third_position = remainder - second_position * len(third)
        triples = np.column_stack(
            (
                first[first_position],
                second[second_position],
                third[third_position],
            )
        )
        distinct = (
            (triples[:, 0] != triples[:, 1])
            & (triples[:, 0] != triples[:, 2])
            & (triples[:, 1] != triples[:, 2])
        )
        if not np.any(distinct):
            continue
        triples = triples[distinct]
        target = points_array[triples]
        target_gram = target @ np.swapaxes(target, 1, 2)
        compatible = np.all(
            np.abs(target_gram - gram_array) <= float(tolerance),
            axis=(1, 2),
        )
        for triple in triples[compatible]:
            yield tuple(int(index) for index in triple)


def _independent_point_triple(points: np.ndarray) -> tuple[int, int, int] | None:
    x = np.asarray(points, dtype=float)
    scale = max(float(np.max(np.linalg.norm(x, axis=1), initial=0.0)), 1.0)
    vector_threshold = 1.0e-12 * scale
    area_threshold = 1.0e-12 * scale * scale
    volume_threshold = 1.0e-12 * scale * scale * scale
    for first in range(len(x)):
        if float(np.linalg.norm(x[first])) <= vector_threshold:
            continue
        for second in range(first + 1, len(x)):
            normal = np.cross(x[first], x[second])
            if float(np.linalg.norm(normal)) <= area_threshold:
                continue
            for third in range(second + 1, len(x)):
                if abs(float(np.dot(x[third], normal))) > volume_threshold:
                    return first, second, third
    return None


def _automatic_operation_sort_key(matrix: np.ndarray) -> tuple[int, int, tuple[float, ...]]:
    arr = np.asarray(matrix, dtype=float)
    identity_rank = 0 if np.allclose(arr, np.eye(3), atol=1.0e-8) else 1
    determinant_rank = 0 if float(np.linalg.det(arr)) > 0.0 else 1
    return identity_rank, determinant_rank, tuple(np.round(arr.reshape(-1), 10))


def _automatic_operation_label(matrix: np.ndarray) -> str:
    arr = np.asarray(matrix, dtype=float)
    if np.allclose(arr, np.eye(3), atol=1.0e-8):
        return "E"
    if np.allclose(arr, -np.eye(3), atol=1.0e-8):
        return "i"
    determinant = float(np.linalg.det(arr))
    proper = arr if determinant > 0.0 else -arr
    cosine = float(np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    order = 2 if angle <= 1.0e-10 else max(2, int(round(2.0 * np.pi / angle)))
    if determinant > 0.0:
        return f"C{order}_auto"
    if abs(float(np.trace(arr)) - 1.0) <= 1.0e-8:
        return "sigma_auto"
    return f"iC{order}_auto"


def _symmetry_symbol_key(symbol, *, ignore_isotopes: bool) -> str:
    text = str(symbol).strip()
    if not ignore_isotopes:
        return text
    number = atomic_number(text)
    return f"Z{number}" if number is not None else re.sub(r"^\d+|\d+$", "", text).capitalize()


def _symmetry_elements_in_frame(
    symbols,
    coords: np.ndarray,
    *,
    tol=1.0e-3,
    max_n=6,
    tol_H=None,
    ignore_isotopes=False,
    auto_max_n=False,
    inertia_tol=1.0e-3,
):
    radii = np.linalg.norm(coords, axis=1)
    max_radius = float(np.max(radii)) if len(radii) else 1.0
    if max_radius <= 0.0:
        max_radius = 1.0
    scaled = coords / max_radius
    # The public ORACLE threshold is an absolute Cartesian distance in
    # angstrom.  Candidate matching is performed on radius-normalized
    # coordinates, so both the acceptance tolerance and the reported residual
    # must be converted explicitly rather than silently becoming relative to
    # molecular size.
    scaled_tol = float(tol) / max_radius
    scaled_tol_H = None if tol_H is None else float(tol_H) / max_radius
    sym_use = [_symmetry_symbol_key(symbol, ignore_isotopes=ignore_isotopes) for symbol in symbols]
    if auto_max_n:
        inertia = np.zeros((3, 3), dtype=float)
        for vec in coords:
            inertia += np.dot(vec, vec) * np.eye(3) - np.outer(vec, vec)
        evals = np.linalg.eigvalsh(inertia)
        max_inertia = float(np.max(evals)) if len(evals) else 0.0
        if max_inertia > 0.0:
            d01 = abs(evals[0] - evals[1]) / max_inertia
            d12 = abs(evals[1] - evals[2]) / max_inertia
            if d01 > inertia_tol and d12 > inertia_tol:
                max_n = min(max_n, 2)
    elements = []
    permutations = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    for label, rotation in candidate_ops(max_n=max_n):
        mapped, max_dev = _match_with_map(
            sym_use,
            scaled,
            scaled @ rotation.T,
            scaled_tol,
            tol_H=scaled_tol_H,
        )
        if mapped is not None:
            unique_key = (
                tuple(int(item) for item in mapped),
                tuple(round(float(value), 10) for value in rotation.reshape(-1)),
            )
            if unique_key in seen:
                continue
            seen.add(unique_key)
            elements.append((label, rotation, float(max_dev) * max_radius))
            permutations.append(tuple(mapped))
    return elements, _atom_classes(len(symbols), permutations), permutations


def group_label(elements, linear=False):
    labels = [item[0] for item in elements]
    polyhedral = _polyhedral_group_label(elements)
    if polyhedral:
        return polyhedral
    improper_rotation = _pure_improper_rotation_group_label(elements)
    if improper_rotation:
        return improper_rotation
    order_two = _order_two_point_group_label(elements)
    if order_two:
        return order_two
    # Automatically enumerated axes need not have x/y/z labels. Recognize
    # complete order-four groups from their matrices, not those labels.
    if len(elements) == 4:
        matrices = tuple(np.asarray(item[1], dtype=float) for item in elements)
        identity = sum(np.allclose(m, np.eye(3), atol=1.e-8) for m in matrices)
        inversion = any(np.allclose(m, -np.eye(3), atol=1.e-8) for m in matrices)
        proper = tuple(m for m in matrices if np.linalg.det(m) > 0.)
        c2_count = sum(abs(float(np.trace(m)) + 1.) < 1.e-8 for m in proper)
        mirrors = sum(np.linalg.det(m) < 0. and abs(float(np.trace(m)) - 1.) < 1.e-8
                      for m in matrices)
        if identity == 1 and len(proper) == 2 and c2_count == 1:
            if inversion and mirrors == 1:
                return "C2h"
            if mirrors == 2:
                return "C2v"
        if identity == 1 and len(proper) == 4 and c2_count == 3:
            return "D2"
    nmax, axis = _highest_cn_axis(labels)
    has_i = "i" in labels
    has_sigma = any(label.startswith("sigma") for label in labels)
    if linear:
        return "Dinfh" if has_i else "Cinfv"
    if nmax >= 2:
        c2_axes = {
            match.group(1)
            for label in labels
            if (match := re.match(r"C2([xyz])\^", str(label))) is not None
        }
        has_c2_perp = any(c2_axis != axis for c2_axis in c2_axes)
        if axis == "z" and any(str(label).startswith("C2_xy") for label in labels):
            has_c2_perp = True
        sigma_h = {"x": "sigma_yz", "y": "sigma_xz", "z": "sigma_xy"}.get(axis or "z")
        has_sigma_h = sigma_h in labels
        has_sigma_v = has_sigma and not has_sigma_h
        if has_sigma_h and has_c2_perp:
            return f"D{nmax}h"
        if has_sigma_h:
            return f"C{nmax}h"
        # Check Dnd only after an explicit horizontal plane.  High-order Dnh
        # groups also contain diagonal C2 operations which can match the Dnd
        # label pattern when a degenerate inertia plane is anchored to a
        # different, but equivalent, molecular direction.
        dnd_n = _dnd_group_order(labels, principal_order=nmax)
        if dnd_n:
            return f"D{dnd_n}d"
        if has_sigma_v:
            return f"C{nmax}v"
        if has_c2_perp:
            return f"D{nmax}"
        return f"C{nmax}"
    if has_i:
        return "Ci"
    if has_sigma:
        return "Cs"
    return "C1"


def _order_two_point_group_label(elements) -> str | None:
    """Classify a complete order-two point group from its matrices.

    An arbitrary Cartesian frame need not give the nonidentity operation one
    of the axis-aligned catalogue labels consumed by the historical parser.
    Matrix invariants distinguish all three order-two molecular point groups
    without introducing a frame convention: a proper pi rotation is ``C2``,
    inversion is ``Ci``, and a plane reflection is ``Cs``.
    """

    if len(elements) != 2:
        return None
    matrices = tuple(np.asarray(item[1], dtype=float) for item in elements)
    identity_indices = tuple(
        index
        for index, matrix in enumerate(matrices)
        if np.allclose(matrix, np.eye(3), atol=1.0e-8)
    )
    if len(identity_indices) != 1:
        return None
    operation = matrices[1 - identity_indices[0]]
    if not np.allclose(operation.T @ operation, np.eye(3), atol=1.0e-8):
        return None
    if not np.allclose(operation @ operation, np.eye(3), atol=1.0e-8):
        return None
    determinant = float(np.linalg.det(operation))
    trace = float(np.trace(operation))
    if determinant > 0.0 and abs(trace + 1.0) <= 1.0e-8:
        return "C2"
    if determinant < 0.0:
        if np.allclose(operation, -np.eye(3), atol=1.0e-8):
            return "Ci"
        if abs(trace - 1.0) <= 1.0e-8:
            return "Cs"
    return None


def _pure_improper_rotation_group_label(elements) -> str | None:
    """Identify a finite ``S_2n`` group from its complete matrix set."""

    matrices = tuple(np.asarray(item[1], dtype=float) for item in elements)
    group_order = len(matrices)
    if group_order < 4 or group_order % 2:
        return None
    improper = tuple(matrix for matrix in matrices if float(np.linalg.det(matrix)) < 0.0)
    if len(improper) != group_order // 2:
        return None
    # Cnh and Dnd contain genuine mirror planes; a pure improper-rotation
    # group does not.  This separates isomorphic cyclic groups such as C3h
    # and S6 by their actual Cartesian operation sets.
    if any(abs(float(np.trace(matrix)) - 1.0) <= 1.0e-8 for matrix in improper):
        return None
    keys = {
        tuple(round(float(value), 8) for value in matrix.reshape(-1))
        for matrix in matrices
    }
    identity = np.eye(3)
    for generator in improper:
        powers: list[np.ndarray] = []
        current = identity.copy()
        for _power in range(group_order):
            powers.append(current.copy())
            current = current @ generator
        if not np.allclose(current, identity, atol=1.0e-8):
            continue
        generated = {
            tuple(round(float(value), 8) for value in matrix.reshape(-1))
            for matrix in powers
        }
        if len(generated) == group_order and generated == keys:
            return f"S{group_order}"
    return None


def _match_with_map(symbols, coords1, coords2, tol, tol_H=None):
    used = np.zeros(len(coords2), dtype=bool)
    mapping = [-1] * len(coords1)
    by_symbol: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        by_symbol.setdefault(symbol, []).append(idx)
    points1 = np.asarray(coords1, dtype=float)
    points2 = np.asarray(coords2, dtype=float)
    radii1 = np.linalg.norm(points1, axis=1)
    radii2 = np.linalg.norm(points2, axis=1)
    max_dev = 0.0
    for idx in sorted(range(len(coords1)), key=lambda item: (len(by_symbol[symbols[item]]), item)):
        eff_tol = tol_H if tol_H is not None and symbols[idx] in {"H", "Z1"} else tol
        radius = float(radii1[idx])
        symbol_candidates = np.asarray(by_symbol.get(symbols[idx], ()), dtype=int)
        if len(symbol_candidates):
            available = symbol_candidates[
                (~used[symbol_candidates]) & (np.abs(radii2[symbol_candidates] - radius) <= eff_tol)
            ]
        else:
            available = symbol_candidates
        if len(available):
            order = np.argsort(np.abs(radii2[available] - radius), kind="stable")
            available = available[order]
            deviations = np.linalg.norm(points2[available] - points1[idx], axis=1)
            accepted = np.flatnonzero(deviations < eff_tol)
            if len(accepted):
                selected = int(accepted[0])
                cand = int(available[selected])
                deviation = float(deviations[selected])
                mapping[idx] = cand
                used[cand] = True
                max_dev = max(max_dev, deviation)
        if mapping[idx] < 0:
            return None, None
    return tuple(mapping), max_dev


def _atom_classes(natoms: int, permutations) -> tuple[tuple[int, ...], ...]:
    parent = list(range(natoms))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for permutation in permutations:
        for left, right in enumerate(permutation):
            union(left, int(right))
    classes: dict[int, list[int]] = {}
    for atom in range(natoms):
        classes.setdefault(find(atom), []).append(atom)
    return tuple(tuple(values) for values in classes.values())


def _highest_cn_axis(labels):
    best = 1
    axis = None
    for label in labels:
        match = re.match(r"C(\d+)([xyz])", str(label))
        if match and int(match.group(1)) > best:
            best = int(match.group(1))
            axis = match.group(2)
    return best, axis


def _polyhedral_group_label(elements) -> str | None:
    matrices = [np.asarray(item[1], dtype=float) for item in elements]
    proper_matrices = tuple(matrix for matrix in matrices if float(np.linalg.det(matrix)) > 0.0)
    has_inversion = any(np.allclose(matrix, -np.eye(3), atol=1.0e-8) for matrix in matrices)
    tetrahedral_proper_classes = Counter(
        operation_class
        for operation_class in (_td_candidate_class(matrix) for matrix in proper_matrices)
        if operation_class is not None
    )
    tetrahedral_proper = Counter({"E": 1, "C3": 8, "C2": 3})
    if tetrahedral_proper_classes == tetrahedral_proper:
        if len(matrices) == 12:
            return "T"
        if len(matrices) == 24 and has_inversion:
            return "Th"
    td_classes = Counter(
        operation_class
        for operation_class in (_td_candidate_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if td_classes == Counter(
        {
            "E": 1,
            "C3": 8,
            "C2": 3,
            "S4": 6,
            "sigma_d": 6,
        }
    ):
        return "Td"
    invariant_oh_classes = Counter(
        operation_class
        for operation_class in (_orientation_invariant_oh_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if invariant_oh_classes == Counter(
        {
            "E": 1,
            "C3": 8,
            "C4": 6,
            "C2": 9,
            "i*E": 1,
            "i*C3": 8,
            "i*C4": 6,
            "i*C2": 9,
        }
    ):
        return "Oh"
    if invariant_oh_classes == Counter(
        {
            "E": 1,
            "C3": 8,
            "C4": 6,
            "C2": 9,
        }
    ):
        return "O"
    oh_classes = Counter(
        operation_class
        for operation_class in (_polyhedral_matrix_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if oh_classes == Counter(
        {
            "E": 1,
            "C3": 8,
            "C2_axis": 3,
            "C4": 6,
            "C2_edge": 6,
            "i*E": 1,
            "i*C3": 8,
            "i*C2_axis": 3,
            "i*C4": 6,
            "i*C2_edge": 6,
        }
    ):
        return "Oh"
    ih_classes = Counter(
        operation_class
        for operation_class in (_icosahedral_candidate_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if len(matrices) in {60, 120} and sum(ih_classes.values()) != len(matrices):
        # A topology-qualified quasi-Ih geometry supplies all 60/120 faithful
        # atom permutations while its independently fitted Cartesian matrices
        # retain deviations commensurate with the recognition tolerance.  Use
        # a narrow class-angle gate only for these complete polyhedral orders;
        # exact and lower-order assignments retain the strict 1e-8 criterion.
        ih_classes = Counter(
            operation_class
            for operation_class in (
                _icosahedral_candidate_class(matrix, tolerance=5.0e-3)
                for matrix in matrices
            )
            if operation_class is not None
        )
    if ih_classes == Counter(
        {
            "E": 1,
            "C2": 15,
            "C3": 20,
            "C5": 12,
            "C5_2": 12,
            "i*E": 1,
            "i*C2": 15,
            "i*C3": 20,
            "i*C5": 12,
            "i*C5_2": 12,
        }
    ):
        return "Ih"
    if ih_classes == Counter({"E": 1, "C2": 15, "C3": 20, "C5": 12, "C5_2": 12}):
        return "I"
    return None


def _orientation_invariant_oh_class(matrix: np.ndarray) -> str | None:
    arr = np.asarray(matrix, dtype=float)
    determinant = float(np.linalg.det(arr))
    proper = arr if determinant > 0.0 else -arr
    trace = float(np.trace(proper))
    if abs(trace - 3.0) <= 1.0e-8:
        label = "E"
    elif abs(trace) <= 1.0e-8:
        label = "C3"
    elif abs(trace - 1.0) <= 1.0e-8:
        label = "C4"
    elif abs(trace + 1.0) <= 1.0e-8:
        label = "C2"
    else:
        return None
    return f"i*{label}" if determinant < 0.0 else label


def _polyhedral_matrix_class(matrix: np.ndarray) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=1.0e-8):
        return "E"
    det = float(np.linalg.det(matrix))
    trace = float(np.trace(matrix))
    if det > 0.0:
        if abs(trace) <= 1.0e-8:
            return "C3"
        if abs(trace - 1.0) <= 1.0e-8:
            return "C4"
        if abs(trace + 1.0) <= 1.0e-8:
            return "C2_axis" if _is_coordinate_axis_c2(matrix) else "C2_edge"
    if det < 0.0:
        if np.allclose(matrix, -np.eye(3), atol=1.0e-8):
            return "i*E"
        proper = -matrix
        proper_class = _polyhedral_matrix_class(proper)
        if proper_class is not None:
            return f"i*{proper_class}"
        if abs(trace - 1.0) <= 1.0e-8:
            return "sigma_d"
        if abs(trace + 1.0) <= 1.0e-8:
            return "S4"
    return None


def _is_coordinate_axis_c2(matrix: np.ndarray) -> bool:
    rounded = np.rint(matrix)
    if not np.allclose(matrix, rounded, atol=1.0e-8):
        return False
    return bool(np.count_nonzero(np.abs(np.diag(rounded)) > 0.5) == 3)


def _dnd_group_order(labels: list[str], *, principal_order: int) -> int | None:
    orders = []
    for label in labels:
        for pattern in (r"Dnd_C(\d+)z\^", r"Dnd_C2_xy_(\d+)_"):
            match = re.search(pattern, str(label))
            if match:
                order = int(match.group(1)) // 2
                if order == principal_order:
                    orders.append(order)
        match = re.fullmatch(r"C2_xy_(\d+)_(\d+)", str(label))
        if match:
            operation_order, index = (int(item) for item in match.groups())
            order = operation_order // 2
            if operation_order == 2 * principal_order and index % 2 == 1:
                orders.append(order)
        match = re.fullmatch(r"sigma_h\*C(\d+)z\^(\d+)", str(label))
        if match:
            operation_order, power = (int(item) for item in match.groups())
            order = operation_order // 2
            if operation_order == 2 * principal_order and power % 2 == 1:
                orders.append(order)
    return max(orders) if orders else None


def _axis_to_z_alignment(axis: str) -> np.ndarray:
    if axis == "x":
        return np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    if axis == "y":
        return np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
    return np.eye(3)


@lru_cache(maxsize=16)
def candidate_ops(max_n=6):
    ops = [("E", np.eye(3)), ("i", -np.eye(3))]
    for axis, name in [(0, "sigma_yz"), (1, "sigma_xz"), (2, "sigma_xy")]:
        matrix = np.eye(3)
        matrix[axis, axis] = -1.0
        ops.append((name, matrix))
    for n in range(2, max_n + 1):
        for power in range(1, n):
            theta = 2.0 * np.pi * power / n
            ops.append((f"C{n}z^{power}", _rotation_matrix((0, 0, 1), theta)))
            ops.append((f"C{n}x^{power}", _rotation_matrix((1, 0, 0), theta)))
            ops.append((f"C{n}y^{power}", _rotation_matrix((0, 1, 0), theta)))
            sigma_h = np.diag((1.0, 1.0, -1.0))
            ops.append(
                (
                    f"sigma_h*C{n}z^{power}",
                    sigma_h @ _rotation_matrix((0, 0, 1), theta),
                )
            )
    for n in range(2, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            ops.append(
                (
                    f"C2_xy_{n}_{k}",
                    _rotation_matrix((np.cos(theta), np.sin(theta), 0), np.pi),
                )
            )
    for n in range(3, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            normal = np.array((np.cos(theta), np.sin(theta), 0.0), dtype=float)
            normal /= np.linalg.norm(normal)
            ops.append((f"sigma_v_{n}_{k}", np.eye(3) - 2.0 * np.outer(normal, normal)))
            if n % 2 == 1 and 2 * n > max_n:
                shifted = theta + 0.5 * np.pi
                shifted_normal = np.array((np.cos(shifted), np.sin(shifted), 0.0), dtype=float)
                shifted_normal /= np.linalg.norm(shifted_normal)
                ops.append(
                    (
                        f"sigma_v_{n}_{k + n}",
                        np.eye(3) - 2.0 * np.outer(shifted_normal, shifted_normal),
                    )
                )
    for n in range(3, max_n + 1):
        order = 2 * n
        sigma_h = np.diag((1.0, 1.0, -1.0))
        for power in range(1, order, 2):
            theta = 2.0 * np.pi * power / order
            ops.append(
                (
                    f"sigma_h*C{order}z^{power}",
                    sigma_h @ _rotation_matrix((0, 0, 1), theta),
                )
            )
            ops.append(
                (
                    f"C2_xy_{order}_{power}",
                    _rotation_matrix((np.cos(theta / 2.0), np.sin(theta / 2.0), 0), np.pi),
                )
            )
            normal = np.array((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0), dtype=float)
            normal /= np.linalg.norm(normal)
            ops.append((f"sigma_v_{order}_{power}", np.eye(3) - 2.0 * np.outer(normal, normal)))
    ops.extend(_dnd_candidate_ops(max_n))
    ops.extend(_cubic_candidate_ops())
    ops.extend(_icosahedral_candidate_ops())
    return ops


def _dnd_candidate_ops(max_n: int) -> list[tuple[str, np.ndarray]]:
    ops: list[tuple[str, np.ndarray]] = []
    sd = _diagonal_reflection_matrix()
    for n in range(2, max_n + 1):
        order = 2 * n
        for power in range(n):
            theta = 2.0 * np.pi * power / n
            ops.append(
                (
                    f"Dnd_C2_xy_{order}_{(2 * power + 1) % order}",
                    sd @ _rotation_matrix((0, 0, 1), theta),
                )
            )
        for k in range(n):
            theta = np.pi * k / n
            c2 = _rotation_matrix((np.cos(theta), np.sin(theta), 0), np.pi)
            ops.append((f"Dnd_C{order}z^{(2 * k + 1) % order}", sd @ c2))
    return ops


def _diagonal_reflection_matrix() -> np.ndarray:
    return np.array(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _cubic_candidate_ops() -> list[tuple[str, np.ndarray]]:
    ops: list[tuple[str, np.ndarray]] = []
    counters: Counter[str] = Counter()
    seen: set[tuple[float, ...]] = set()

    for matrix in _signed_permutation_matrices():
        if _sign_product(matrix) != 1:
            continue
        label = _unique_cubic_label(_td_candidate_class(matrix), "td", counters)
        if label is not None:
            key = tuple(float(value) for value in matrix.reshape(-1))
            seen.add(key)
            ops.append((label, matrix))

    for matrix in _signed_permutation_matrices():
        key = tuple(float(value) for value in matrix.reshape(-1))
        if key in seen:
            continue
        label = _unique_cubic_label(_oh_candidate_class(matrix), "oh", counters)
        if label is not None:
            ops.append((label, matrix))
    return ops


def _signed_permutation_matrices() -> tuple[np.ndarray, ...]:
    matrices: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            matrices.append(matrix)
    return tuple(matrices)


def _sign_product(matrix: np.ndarray) -> int:
    product_value = 1.0
    for row in range(3):
        nonzero = np.flatnonzero(np.abs(matrix[row]) > 0.5)
        if len(nonzero) != 1:
            return 0
        product_value *= float(matrix[row, nonzero[0]])
    return 1 if product_value > 0.0 else -1


def _td_candidate_class(matrix: np.ndarray) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=1.0e-8):
        return "E"
    det = float(np.linalg.det(matrix))
    trace = float(np.trace(matrix))
    if det > 0.0:
        if abs(trace) <= 1.0e-8:
            return "C3"
        if abs(trace + 1.0) <= 1.0e-8:
            return "C2"
    if det < 0.0:
        if abs(trace + 1.0) <= 1.0e-8:
            return "S4"
        if abs(trace - 1.0) <= 1.0e-8:
            return "sigma_d"
    return None


def _oh_candidate_class(matrix: np.ndarray) -> str | None:
    operation_class = _polyhedral_matrix_class(matrix)
    if operation_class is None:
        return None
    return operation_class.replace("*", "_")


def _unique_cubic_label(
    operation_class: str | None,
    prefix: str,
    counters: Counter[str],
) -> str | None:
    if operation_class is None:
        return None
    if operation_class == "E":
        return f"{prefix}_E"
    if operation_class == "i_E":
        return "i"
    counters[f"{prefix}_{operation_class}"] += 1
    return f"{prefix}_{operation_class}_{counters[f'{prefix}_{operation_class}']}"


def _icosahedral_candidate_ops() -> list[tuple[str, np.ndarray]]:
    ops: list[tuple[str, np.ndarray]] = []
    counters: Counter[str] = Counter()
    for matrix in _icosahedral_operation_matrices():
        operation_class = _icosahedral_candidate_class(matrix)
        if operation_class is None:
            continue
        label = _unique_icosahedral_label(operation_class, counters)
        ops.append((label, matrix))
    return ops


@lru_cache(maxsize=1)
def _icosahedral_operation_matrices() -> tuple[np.ndarray, ...]:
    vertices = _icosahedral_vertices()
    source_indices = _independent_vertex_triple(vertices)
    source = vertices[list(source_indices)]
    source_inverse = np.linalg.inv(source)
    matrices: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for target_indices in permutations(range(len(vertices)), 3):
        target = vertices[list(target_indices)]
        if abs(float(np.linalg.det(target))) <= 1.0e-8:
            continue
        matrix = target.T @ source_inverse.T
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-8):
            continue
        if not _maps_vertices_to_self(matrix, vertices):
            continue
        key = tuple(round(float(value), 10) for value in matrix.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        matrices.append(matrix)
    return tuple(matrices)


def _icosahedral_vertices() -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = []
    for y in (-1.0, 1.0):
        for z in (-phi, phi):
            vertices.append((0.0, y, z))
    for x in (-1.0, 1.0):
        for y in (-phi, phi):
            vertices.append((x, y, 0.0))
    for x in (-phi, phi):
        for z in (-1.0, 1.0):
            vertices.append((x, 0.0, z))
    return np.array(vertices, dtype=float)


def _independent_vertex_triple(vertices: np.ndarray) -> tuple[int, int, int]:
    for candidate in permutations(range(len(vertices)), 3):
        if abs(float(np.linalg.det(vertices[list(candidate)]))) > 1.0e-8:
            return tuple(int(item) for item in candidate)
    raise ValueError("no independent icosahedral vertex triple")


def _maps_vertices_to_self(matrix: np.ndarray, vertices: np.ndarray) -> bool:
    transformed = vertices @ matrix.T
    for vertex in transformed:
        if not np.any(np.all(np.isclose(vertices, vertex, atol=1.0e-8), axis=1)):
            return False
    return True


def _icosahedral_candidate_class(
    matrix: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=float(tolerance)):
        return "E"
    det = float(np.linalg.det(matrix))
    proper = matrix if det > 0.0 else -matrix
    proper_class = _icosahedral_proper_class(proper, tolerance=float(tolerance))
    if proper_class is None:
        return None
    return f"i*{proper_class}" if det < 0.0 else proper_class


def _icosahedral_proper_class(
    matrix: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=float(tolerance)):
        return "E"
    trace = float(np.trace(matrix))
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phi_bar = (1.0 - np.sqrt(5.0)) / 2.0
    if abs(trace) <= float(tolerance):
        return "C3"
    if abs(trace + 1.0) <= float(tolerance):
        return "C2"
    if abs(trace - phi) <= float(tolerance):
        return "C5"
    if abs(trace - phi_bar) <= float(tolerance):
        return "C5_2"
    return None


def _unique_icosahedral_label(operation_class: str, counters: Counter[str]) -> str:
    if operation_class == "E":
        return "ih_E"
    if operation_class == "i*E":
        return "i"
    key = operation_class.replace("*", "_")
    counters[key] += 1
    return f"ih_{key}_{counters[key]}"


def _rotation_matrix(axis, theta):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )
