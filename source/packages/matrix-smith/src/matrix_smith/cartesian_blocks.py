"""SMITH-owned symmetry-adapted Cartesian coordinate blocks.

The kernel in this module is shared by typed ONIC substrate blocks and by
MORPHEUS.  ORACLE supplies the frozen symmetry operations; SMITH removes the
subset rigid motions, constructs complete isotypic subspaces and fixes their
component gauge from an explicitly declared structural site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from matrix_chem import MolecularSymmetry, cartesian_operation_matrix
from matrix_chem.topology.elements import atomic_number
from matrix_numerics import select_rank_revealing_rows

from .bmatrix import SparseBMatrix
from .block_payload import embed_local_sparse_rows
from .onic_blocks import (
    OnicBlockDiagnostics,
    OnicCoordinateBlock,
    OnicDegeneracyGroup,
    OnicMatrixRecord,
    OnicSiteFrame,
    OnicSymmetryOperation,
    onic_reference_fingerprint,
)
from .symmetry_labels import irrep_characters_for_operations, irrep_name_prefix


CARTESIAN_BLOCK_RANK_METHOD = "SHARED_MAX_RESIDUAL_TWICE_MGS"
CARTESIAN_BLOCK_GAUGE = "SITE_FRAME_PROJECTED_SEED_RANK_REVEALING"
CARTESIAN_BLOCK_ABSOLUTE_RANK_TOLERANCE = 1.0e-10
CARTESIAN_BLOCK_RELATIVE_RANK_TOLERANCE = 1.0e-8
CARTESIAN_BLOCK_COVARIANCE_TOLERANCE = 5.0e-8


@dataclass(frozen=True)
class CartesianIrrepSubspace:
    irrep: str
    dimension: int
    multiplicity: int
    column_indices: tuple[int, ...]
    projector: np.ndarray
    representation_matrices: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SymmetryAdaptedCartesianBasis:
    """A complete TR-free Cartesian basis for one symmetry-closed subset."""

    atom_indices_one_based: tuple[int, ...]
    reference_coordinates_angstrom: np.ndarray
    cartesian_from_q: np.ndarray
    irreps: tuple[str, ...]
    point_group: str
    physical_point_group: str
    external_mode_count: int
    linearity: str
    site_frame: OnicSiteFrame
    operation_labels: tuple[str, ...]
    operation_permutations_zero_based: tuple[tuple[int, ...], ...]
    irrep_subspaces: tuple[CartesianIrrepSubspace, ...]
    orthonormality_residual: float
    projector_residual: float
    projector_idempotency_residual: float
    projector_symmetry_residual: float
    covariance_residual: float

    @property
    def target_rank(self) -> int:
        return int(self.cartesian_from_q.shape[1])


@dataclass(frozen=True)
class CartesianBlockEvaluation:
    """Current values and constant analytic Wilson rows of a frozen block."""

    coordinate_values_angstrom: np.ndarray
    b_matrix: SparseBMatrix


def onic_degeneracy_groups_from_cartesian_basis(
    basis: SymmetryAdaptedCartesianBasis,
    coordinate_identifiers: Sequence[str],
    *,
    source_count: int,
    block_identifier: str,
    component_gauge: str,
) -> tuple[OnicDegeneracyGroup, ...]:
    """Serialize complete isotypic subspaces without materializing projectors."""

    coordinate_ids = tuple(str(identifier) for identifier in coordinate_identifiers)
    if len(coordinate_ids) != basis.target_rank:
        raise ValueError("coordinate identifiers must match the Cartesian basis rank")
    groups: list[OnicDegeneracyGroup] = []
    for group_index, subspace in enumerate(basis.irrep_subspaces, start=1):
        group_coordinate_ids = tuple(coordinate_ids[index] for index in subspace.column_indices)
        representation_matrices = tuple(
            OnicSymmetryOperation(
                label=label,
                matrix=OnicMatrixRecord(
                    rows=len(group_coordinate_ids),
                    columns=len(group_coordinate_ids),
                    storage="DENSE",
                    dense_rows=tuple(tuple(float(value) for value in row) for row in matrix),
                ),
            )
            for label, matrix in zip(
                basis.operation_labels,
                subspace.representation_matrices,
                strict=True,
            )
        )
        group_prefix = (
            f"{block_identifier}.{irrep_name_prefix(subspace.irrep)}Iso{group_index:03d}"
        )
        groups.append(
            OnicDegeneracyGroup(
                identifier=group_prefix,
                irrep=subspace.irrep,
                coordinate_identifiers=group_coordinate_ids,
                component_gauge=component_gauge,
                projector=OnicMatrixRecord(
                    rows=int(source_count),
                    columns=int(source_count),
                    storage="IMPLICIT_FROM_COEFFICIENTS",
                    reference=f"{group_prefix}.coefficients",
                ),
                representation_matrices=representation_matrices,
            )
        )
    return tuple(groups)


def symmetry_adapted_cartesian_basis(
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    symmetry: MolecularSymmetry,
    atom_indices_one_based: Sequence[int] | None = None,
    site_anchor_atom_indices_one_based: Sequence[int] = (),
    frame_axes_global: np.ndarray | Sequence[Sequence[float]] | None = None,
    require_site_anchor: bool = False,
    rank_absolute_tolerance: float = CARTESIAN_BLOCK_ABSOLUTE_RANK_TOLERANCE,
    rank_relative_tolerance: float = CARTESIAN_BLOCK_RELATIVE_RANK_TOLERANCE,
) -> SymmetryAdaptedCartesianBasis:
    """Construct a deterministic symmetry-adapted Cartesian subset basis.

    The retained atom subset must be closed under every frozen ORACLE
    operation.  Rank selection is performed on each complete isotypic
    projector, never on isolated members of a multidimensional irrep.
    """

    symbols = tuple(str(atom) for atom in atoms)
    full_reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    if full_reference.shape != (len(symbols), 3) or not np.all(np.isfinite(full_reference)):
        raise ValueError("Cartesian block reference must be a finite natoms-by-3 array")
    absolute_tolerance = _positive_tolerance(rank_absolute_tolerance, "absolute rank tolerance")
    relative_tolerance = _positive_tolerance(rank_relative_tolerance, "relative rank tolerance")
    subset = _normalized_subset(atom_indices_one_based, natoms=len(symbols))
    subset_zero = tuple(atom - 1 for atom in subset)
    subset_reference = full_reference[np.asarray(subset_zero, dtype=int)]
    subset_symbols = tuple(symbols[index] for index in subset_zero)
    site_frame = _site_frame(
        subset_reference,
        subset,
        tuple(int(atom) for atom in site_anchor_atom_indices_one_based),
        frame_axes_global=frame_axes_global,
        require_site_anchor=require_site_anchor,
    )
    physical_operation_data = _closed_subset_operations(
        symmetry,
        subset,
        natoms=len(symbols),
    )
    operation_data, computational_point_group = _linear_computational_subgroup(
        physical_operation_data,
        physical_point_group=symmetry.point_group,
    )
    operation_labels = tuple(item[0] for item in operation_data)
    rotations = tuple(item[1] for item in operation_data)
    permutations = tuple(item[2] for item in operation_data)
    cartesian_operations = tuple(
        cartesian_operation_matrix(rotation, permutation, natoms=len(subset))
        for rotation, permutation in zip(rotations, permutations, strict=True)
    )
    physical_cartesian_operations = tuple(
        cartesian_operation_matrix(rotation, permutation, natoms=len(subset))
        for _label, rotation, permutation in physical_operation_data
    )
    reference_scale = max(
        1.0,
        float(np.max(np.linalg.norm(subset_reference - np.mean(subset_reference, axis=0), axis=1))),
    )
    operation_geometry_residual = max(
        (
            min(
                float(
                    np.max(
                        np.linalg.norm(
                            subset_reference @ rotation.T
                            - subset_reference[np.asarray(permutation, dtype=int)],
                            axis=1,
                        )
                    )
                ),
                float(
                    np.max(
                        np.linalg.norm(
                            subset_reference @ rotation
                            - subset_reference[np.asarray(permutation, dtype=int)],
                            axis=1,
                        )
                    )
                ),
            )
            for _label, rotation, permutation in physical_operation_data
        ),
        default=0.0,
    )
    covariance_tolerance = max(
        CARTESIAN_BLOCK_COVARIANCE_TOLERANCE,
        5.0 * operation_geometry_residual / reference_scale,
    )

    external_rows, external_mode_count, linearity = _external_motion_basis(
        subset_reference,
        site_frame,
        tolerance=absolute_tolerance,
    )
    identity = np.eye(3 * len(subset), dtype=float)
    vibrational_projector = identity - external_rows.T @ external_rows
    vibrational_projector = _symmetric(vibrational_projector)
    target_rank = 3 * len(subset) - external_mode_count
    covariance_residual = max(
        (
            float(
                np.linalg.norm(
                    operation @ vibrational_projector @ operation.T
                    - vibrational_projector,
                    ord=2,
                )
            )
            for operation in physical_cartesian_operations
        ),
        default=0.0,
    )
    if covariance_residual > covariance_tolerance:
        raise ValueError(
            "Cartesian subset rigid-motion projector is not covariant under the retained group "
            f"(residual={covariance_residual:.3e})"
        )

    characters = irrep_characters_for_operations(
        operation_labels,
        point_group=computational_point_group,
        operation_matrices=rotations,
    )
    if not characters:
        raise ValueError(f"no character table is available for retained group {symmetry.point_group}")
    identity_index = _identity_operation_index(rotations, permutations)
    seed_rows = _site_frame_seed_rows(
        subset_symbols,
        subset_reference,
        site_frame,
    )
    coefficient_rows: list[np.ndarray] = []
    ordered_irreps: list[str] = []
    subspaces: list[CartesianIrrepSubspace] = []
    projector_sum = np.zeros_like(vibrational_projector)

    for irrep, character_values in characters:
        chars = np.asarray(character_values, dtype=float)
        if chars.shape != (len(cartesian_operations),) or not np.all(np.isfinite(chars)):
            raise ValueError(f"invalid character row for irrep {irrep}")
        irrep_dimension = int(round(float(chars[identity_index])))
        if irrep_dimension < 1 or abs(float(chars[identity_index]) - irrep_dimension) > 1.0e-8:
            raise ValueError(f"invalid identity character for irrep {irrep}")
        symmetry_projector = sum(
            (
                float(character) * operation
                for character, operation in zip(chars, cartesian_operations, strict=True)
            ),
            start=np.zeros_like(vibrational_projector),
        )
        # Normalize by the real-character norm rather than |G|.  For an
        # absolutely irreducible real representation the two expressions are
        # identical.  Cyclic groups such as C3 are different: their real E
        # representation combines a complex-conjugate pair, so
        # sum_g chi(g)^2 = 2 |G|.  The complex normalization d/|G| would then
        # produce an operator with eigenvalue two instead of an idempotent
        # isotypic projector.  d/sum chi^2 is the corresponding real-character
        # normalization and covers both cases.
        character_norm = float(chars @ chars)
        if not np.isfinite(character_norm) or character_norm <= absolute_tolerance:
            raise ValueError(f"character row for irrep {irrep} has zero norm")
        symmetry_projector *= irrep_dimension / character_norm
        projector = _symmetric(
            vibrational_projector @ symmetry_projector @ vibrational_projector
        )
        eigenvalues = np.linalg.eigvalsh(projector)
        cutoff = max(
            absolute_tolerance,
            relative_tolerance * max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0),
        )
        rank = int(np.count_nonzero(eigenvalues > cutoff))
        if rank == 0:
            continue
        if rank % irrep_dimension:
            raise ValueError(
                f"retained {irrep} Cartesian subspace rank {rank} is not a multiple of "
                f"its dimension {irrep_dimension}"
            )
        projected_seeds = seed_rows @ projector
        selection = select_rank_revealing_rows(
            projected_seeds,
            target_rank=rank,
            tolerance=absolute_tolerance,
        )
        if selection.rank != rank:
            raise ValueError(
                f"site-frame gauge spans only {selection.rank}/{rank} directions for irrep {irrep}"
            )
        rows = np.asarray(selection.orthonormal_basis, dtype=float)
        row_projector = rows.T @ rows
        subspace_residual = float(np.linalg.norm(row_projector - projector, ord=2))
        if subspace_residual > covariance_tolerance:
            raise ValueError(
                f"rank-revealing {irrep} gauge does not reproduce its projector "
                f"(residual={subspace_residual:.3e})"
            )
        subspace_covariance_residual = max(
            (
                float(
                    np.linalg.norm(
                        operation @ row_projector @ operation.T - row_projector,
                        ord=2,
                    )
                )
                for operation in physical_cartesian_operations
            ),
            default=0.0,
        )
        if subspace_covariance_residual > covariance_tolerance:
            raise ValueError(
                f"retained {irrep} Cartesian subspace is not covariant "
                f"(residual={subspace_covariance_residual:.3e})"
            )
        covariance_residual = max(covariance_residual, subspace_covariance_residual)
        start = len(coefficient_rows)
        coefficient_rows.extend(row.copy() for row in rows)
        ordered_irreps.extend(irrep for _ in range(rank))
        representation_matrices = tuple(
            rows @ operation @ rows.T for operation in cartesian_operations
        )
        subspaces.append(
            CartesianIrrepSubspace(
                irrep=irrep,
                dimension=irrep_dimension,
                multiplicity=rank // irrep_dimension,
                column_indices=tuple(range(start, start + rank)),
                projector=row_projector,
                representation_matrices=representation_matrices,
            )
        )
        projector_sum += row_projector

    if len(coefficient_rows) != target_rank:
        raise ValueError(
            "retained character projectors do not span the complete Cartesian internal space "
            f"({len(coefficient_rows)}/{target_rank})"
        )
    coefficients = (
        np.vstack(coefficient_rows)
        if coefficient_rows
        else np.zeros((0, 3 * len(subset)), dtype=float)
    )
    cartesian_from_q = coefficients.T
    orthonormality_residual = (
        float(np.linalg.norm(coefficients @ coefficients.T - np.eye(target_rank), ord=2))
        if target_rank
        else 0.0
    )
    projector_residual = float(np.linalg.norm(projector_sum - vibrational_projector, ord=2))
    projector_idempotency_residual = float(
        np.linalg.norm(projector_sum @ projector_sum - projector_sum, ord=2)
    )
    projector_symmetry_residual = float(np.linalg.norm(projector_sum - projector_sum.T, ord=2))
    maximum_residual = max(
        orthonormality_residual,
        projector_residual,
        projector_idempotency_residual,
        projector_symmetry_residual,
    )
    if maximum_residual > covariance_tolerance:
        raise ValueError(
            "symmetry-adapted Cartesian basis failed its global projector audit "
            f"(residual={maximum_residual:.3e})"
        )
    return SymmetryAdaptedCartesianBasis(
        atom_indices_one_based=subset,
        reference_coordinates_angstrom=subset_reference.copy(),
        cartesian_from_q=cartesian_from_q,
        irreps=tuple(ordered_irreps),
        point_group=computational_point_group,
        physical_point_group=symmetry.point_group,
        external_mode_count=external_mode_count,
        linearity=linearity,
        site_frame=site_frame,
        operation_labels=operation_labels,
        operation_permutations_zero_based=permutations,
        irrep_subspaces=tuple(subspaces),
        orthonormality_residual=orthonormality_residual,
        projector_residual=projector_residual,
        projector_idempotency_residual=projector_idempotency_residual,
        projector_symmetry_residual=projector_symmetry_residual,
        covariance_residual=covariance_residual,
    )


def build_symmetry_adapted_cartesian_block(
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    symmetry: MolecularSymmetry,
    atom_indices_one_based: Sequence[int],
    site_anchor_atom_indices_one_based: Sequence[int],
    block_identifier: str = "SUB1",
    local_symmetry_provenance: str = "ORACLE_DECLARED_SITE_SUBGROUP",
    protected: bool = True,
    active: bool = True,
    observable: bool = False,
    rank_absolute_tolerance: float = CARTESIAN_BLOCK_ABSOLUTE_RANK_TOLERANCE,
    rank_relative_tolerance: float = CARTESIAN_BLOCK_RELATIVE_RANK_TOLERANCE,
) -> OnicCoordinateBlock:
    """Build a frozen first-order Cartesian substrate block for typed ONIC."""

    basis = symmetry_adapted_cartesian_basis(
        atoms,
        reference_coordinates_angstrom,
        symmetry=symmetry,
        atom_indices_one_based=atom_indices_one_based,
        site_anchor_atom_indices_one_based=site_anchor_atom_indices_one_based,
        require_site_anchor=True,
        rank_absolute_tolerance=rank_absolute_tolerance,
        rank_relative_tolerance=rank_relative_tolerance,
    )
    coordinate_ids: list[str] = []
    counters: dict[str, int] = {}
    for irrep in basis.irreps:
        counters[irrep] = counters.get(irrep, 0) + 1
        coordinate_ids.append(
            f"{block_identifier}.{irrep_name_prefix(irrep)}Cart{counters[irrep]:04d}"
        )
    source_order = tuple(
        f"ATOM{atom}.{axis}"
        for atom in basis.atom_indices_one_based
        for axis in ("X", "Y", "Z")
    )
    degeneracy_groups = onic_degeneracy_groups_from_cartesian_basis(
        basis,
        coordinate_ids,
        source_count=len(source_order),
        block_identifier=block_identifier,
        component_gauge=CARTESIAN_BLOCK_GAUGE,
    )
    reference = tuple(
        tuple(float(value) for value in row)
        for row in basis.reference_coordinates_angstrom
    )
    coefficient_rows = tuple(
        tuple(float(value) for value in row) for row in basis.cartesian_from_q.T
    )
    return OnicCoordinateBlock(
        identifier=block_identifier,
        kind="SUBSTRATE",
        representation="SYMMETRY_ADAPTED_CARTESIAN",
        atom_indices_one_based=basis.atom_indices_one_based,
        atom_indices_zero_based=tuple(atom - 1 for atom in basis.atom_indices_one_based),
        reference_coordinates_angstrom=reference,
        reference_fingerprint_sha256=onic_reference_fingerprint(
            basis.atom_indices_one_based,
            reference,
        ),
        source_family_identifiers=("CARTESIAN_DISPLACEMENT",),
        source_order=source_order,
        coordinate_identifiers=tuple(coordinate_ids),
        target_rank=basis.target_rank,
        source_count=len(source_order),
        nullity=basis.external_mode_count,
        linearity=basis.linearity,
        rank_method=CARTESIAN_BLOCK_RANK_METHOD,
        rank_absolute_tolerance=rank_absolute_tolerance,
        rank_relative_tolerance=rank_relative_tolerance,
        coefficient_operator=OnicMatrixRecord(
            rows=basis.target_rank,
            columns=len(source_order),
            storage="DENSE",
            dense_rows=coefficient_rows,
        ),
        local_symmetry_provenance=local_symmetry_provenance,
        exact_retained_group=basis.point_group,
        irrep_labels=basis.irreps,
        degeneracy_groups=degeneracy_groups,
        component_gauge=CARTESIAN_BLOCK_GAUGE,
        unit="ANGSTROM",
        scaling_policy="EXPLICIT_UNIT_SCALE",
        scale_factors=(1.0,) * basis.target_rank,
        protected=protected,
        active=active,
        observable=observable,
        analytic_derivative_status="ANALYTIC_FIRST_ORDER",
        second_derivative_status="GENERAL_SPARSE_B_PRIME",
        diagnostics=OnicBlockDiagnostics(
            spectrum=(1.0,) * basis.target_rank,
            condition_number=1.0,
            projector_symmetry_residual=basis.projector_symmetry_residual,
            projector_idempotency_residual=basis.projector_idempotency_residual,
            row_space_residual=basis.projector_residual,
            covariance_residual=basis.covariance_residual,
            chirality_policy="SITE_FRAME_RIGHT_HANDED",
            messages=(
                f"EXTERNAL_MODES_REMOVED={basis.external_mode_count}",
                f"PHYSICAL_POINT_GROUP={basis.physical_point_group}",
                f"COMPUTATIONAL_POINT_GROUP={basis.point_group}",
            ),
        ),
        site_frame=basis.site_frame,
        provenance=(
            "SMITH_OWNED_CARTESIAN_KERNEL",
            f"POINT_GROUP={basis.point_group}",
            f"PHYSICAL_POINT_GROUP={basis.physical_point_group}",
            "OPERATIONS=" + ",".join(basis.operation_labels),
        ),
    )


def evaluate_symmetry_adapted_cartesian_block(
    block: OnicCoordinateBlock,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
) -> CartesianBlockEvaluation:
    """Evaluate one frozen linear substrate chart in the complete atom frame."""

    coordinates, current, reference, coefficients = _cartesian_block_evaluation_arrays(
        block,
        coordinates_angstrom,
    )
    local_b = SparseBMatrix.from_dense(
        coefficients,
        row_labels=block.coordinate_identifiers,
        backend="smith-symmetry-adapted-cartesian-constant.v1",
    )
    rows = embed_local_sparse_rows(
        local_b.rows,
        block.atom_indices_one_based,
        full_natoms=len(coordinates),
        payload_name="symmetry-adapted Cartesian",
    )
    return CartesianBlockEvaluation(
        coordinate_values_angstrom=coefficients @ (current - reference).reshape(-1),
        b_matrix=SparseBMatrix(
            rows=rows,
            column_count=coordinates.size,
            row_labels=block.coordinate_identifiers,
            backend=local_b.backend,
        ),
    )


def evaluate_symmetry_adapted_cartesian_block_values(
    block: OnicCoordinateBlock,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Evaluate only the frozen Cartesian block values."""

    _coordinates, current, reference, coefficients = _cartesian_block_evaluation_arrays(
        block,
        coordinates_angstrom,
    )
    return np.asarray(coefficients @ (current - reference).reshape(-1), dtype=float)


def _cartesian_block_evaluation_arrays(
    block: OnicCoordinateBlock,
    coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated arrays shared by value-only and full Cartesian evaluation."""

    if block.kind != "SUBSTRATE" or block.representation != "SYMMETRY_ADAPTED_CARTESIAN":
        raise ValueError("Cartesian block evaluator received another block type")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("Cartesian block evaluation needs a finite natoms-by-3 geometry")
    if max(block.atom_indices_one_based) > len(coordinates):
        raise ValueError("Cartesian block references atoms outside the current geometry")
    subset = np.asarray([atom - 1 for atom in block.atom_indices_one_based], dtype=int)
    current = coordinates[subset]
    reference = np.asarray(block.reference_coordinates_angstrom, dtype=float)
    coefficients = np.asarray(block.coefficient_operator.to_dense(), dtype=float)
    expected_shape = (block.target_rank, 3 * len(subset))
    if coefficients.shape != expected_shape:
        raise ValueError("Cartesian block coefficient operator has an inconsistent shape")
    return coordinates, current, reference, coefficients


def _positive_tolerance(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _normalized_subset(indices: Sequence[int] | None, *, natoms: int) -> tuple[int, ...]:
    subset = tuple(range(1, natoms + 1)) if indices is None else tuple(int(item) for item in indices)
    if not subset or any(item < 1 or item > natoms for item in subset):
        raise ValueError("Cartesian block atom subset contains an invalid one-based index")
    if len(set(subset)) != len(subset):
        raise ValueError("Cartesian block atom subset contains duplicates")
    return subset


def _site_frame(
    subset_coordinates: np.ndarray,
    subset_atoms_one_based: tuple[int, ...],
    site_anchors_one_based: tuple[int, ...],
    *,
    frame_axes_global: np.ndarray | Sequence[Sequence[float]] | None,
    require_site_anchor: bool,
) -> OnicSiteFrame:
    subset_lookup = {atom: index for index, atom in enumerate(subset_atoms_one_based)}
    if site_anchors_one_based:
        if len(site_anchors_one_based) < 3 or len(set(site_anchors_one_based)) != len(
            site_anchors_one_based
        ):
            raise ValueError("site-anchored Cartesian gauge requires at least three unique atoms")
        if any(atom not in subset_lookup for atom in site_anchors_one_based):
            raise ValueError("site-frame anchors must belong to the Cartesian subset")
        anchors = subset_coordinates[
            np.asarray([subset_lookup[atom] for atom in site_anchors_one_based], dtype=int)
        ]
        origin = np.mean(anchors, axis=0)
        subset_center = np.mean(subset_coordinates, axis=0)
        z_axis = origin - subset_center
        if np.linalg.norm(z_axis) <= 1.0e-10:
            z_axis = np.cross(anchors[1] - anchors[0], anchors[2] - anchors[0])
        z_axis = _normalized_vector(z_axis, "site normal")
        x_axis = None
        for anchor in anchors:
            candidate = anchor - origin
            candidate -= float(candidate @ z_axis) * z_axis
            if np.linalg.norm(candidate) > 1.0e-10:
                x_axis = _normalized_vector(candidate, "site in-plane axis")
                break
        if x_axis is None:
            raise ValueError("site anchors do not define an in-plane gauge")
        y_axis = _normalized_vector(np.cross(z_axis, x_axis), "site secondary axis")
        x_axis = _normalized_vector(np.cross(y_axis, z_axis), "site primary axis")
        axes = np.vstack((x_axis, y_axis, z_axis))
        return OnicSiteFrame(
            anchor_atom_indices_one_based=site_anchors_one_based,
            origin_angstrom=tuple(float(value) for value in origin),
            axes_global=tuple(tuple(float(value) for value in row) for row in axes),
            policy="DECLARED_STRUCTURAL_SITE",
            orientation_sign_policy="OUTWARD_NORMAL_THEN_ORDERED_FIRST_ANCHOR",
            provenance=("NO_INERTIA_TENSOR_GAUGE",),
        )
    if require_site_anchor:
        raise ValueError("typed Cartesian substrate blocks require declared site anchors")
    axes = (
        np.eye(3, dtype=float)
        if frame_axes_global is None
        else np.asarray(frame_axes_global, dtype=float)
    )
    if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
        raise ValueError("fallback Cartesian gauge must provide a finite 3-by-3 frame")
    # The supplied vectors are rows in the OnicSiteFrame convention.
    u_matrix, _singular, vh_matrix = np.linalg.svd(axes)
    axes = u_matrix @ vh_matrix
    if np.linalg.det(axes) < 0.0:
        axes[-1] *= -1.0
    origin = np.mean(subset_coordinates, axis=0)
    return OnicSiteFrame(
        anchor_atom_indices_one_based=(),
        origin_angstrom=tuple(float(value) for value in origin),
        axes_global=tuple(tuple(float(value) for value in row) for row in axes),
        policy="DECLARED_FALLBACK_FRAME",
        orientation_sign_policy="RIGHT_HANDED_POLAR_FACTOR",
        provenance=("NO_SITE_ANCHOR_REQUESTED",),
    )


def _normalized_vector(vector: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{name} is singular")
    return result / norm


def _closed_subset_operations(
    symmetry: MolecularSymmetry,
    subset_atoms_one_based: tuple[int, ...],
    *,
    natoms: int,
) -> tuple[tuple[str, np.ndarray, tuple[int, ...]], ...]:
    operations = tuple(symmetry.operations)
    if not operations:
        if str(symmetry.point_group).strip().upper() not in {"", "C1"}:
            raise ValueError("nontrivial retained group has no frozen ORACLE operations")
        return (("E", np.eye(3, dtype=float), tuple(range(len(subset_atoms_one_based)))),)
    subset_position = {atom: index for index, atom in enumerate(subset_atoms_one_based)}
    output: list[tuple[str, np.ndarray, tuple[int, ...]]] = []
    for operation in operations:
        if len(operation.permutation) != natoms:
            raise ValueError("ORACLE symmetry operation size does not match the complete geometry")
        full_mapping = tuple(int(atom) for atom in operation.permutation)
        if sorted(full_mapping) != list(range(1, natoms + 1)):
            raise ValueError(f"ORACLE operation {operation.label} is not a permutation")
        mapped_atoms = tuple(full_mapping[atom - 1] for atom in subset_atoms_one_based)
        outside = tuple(atom for atom in mapped_atoms if atom not in subset_position)
        if outside:
            raise ValueError(
                f"Cartesian subset is not closed under operation {operation.label}; "
                f"mapped atoms outside subset: {outside}"
            )
        local_mapping = tuple(subset_position[atom] for atom in mapped_atoms)
        rotation = np.asarray(operation.rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"ORACLE operation {operation.label} has an invalid rotation")
        if np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=2) > 1.0e-8:
            raise ValueError(f"ORACLE operation {operation.label} is not orthogonal")
        output.append((str(operation.label), rotation, local_mapping))
    return tuple(output)


def _linear_computational_subgroup(
    operations: tuple[tuple[str, np.ndarray, tuple[int, ...]], ...],
    *,
    physical_point_group: str,
) -> tuple[tuple[tuple[str, np.ndarray, tuple[int, ...]], ...], str]:
    """Select a closed finite subgroup from ORACLE's infinite-group sample.

    ORACLE deliberately serializes a non-closed diagnostic sample for linear
    groups.  Character projectors require a finite group, so SMITH extracts
    the highest available C_n_v or D_n_h subgroup without changing ORACLE's
    physical point-group assignment.
    """

    group = str(physical_point_group).strip().upper()
    if group not in {"CINFV", "DINFH"}:
        return operations, str(physical_point_group)
    identity = np.eye(3, dtype=float)
    inversion = -identity
    matrices = tuple(np.asarray(item[1], dtype=float) for item in operations)

    def matrix_order(matrix: np.ndarray, *, maximum: int = 12) -> int | None:
        current = identity.copy()
        for order in range(1, maximum + 1):
            current = current @ matrix
            if np.allclose(current, identity, atol=1.0e-8, rtol=0.0):
                return order
        return None

    def matched_records(
        generated: tuple[np.ndarray, ...],
        canonical_labels: tuple[str, ...],
    ) -> tuple[tuple[str, np.ndarray, tuple[int, ...]], ...] | None:
        if len(generated) != len(canonical_labels):
            raise ValueError("linear subgroup matrices and labels must have equal length")
        selected: list[tuple[str, np.ndarray, tuple[int, ...]]] = []
        selected_indices: set[int] = set()
        for target, canonical_label in zip(generated, canonical_labels, strict=True):
            matches = [
                index
                for index, matrix in enumerate(matrices)
                if index not in selected_indices
                and np.allclose(matrix, target, atol=1.0e-8, rtol=0.0)
            ]
            if not matches:
                return None
            index = matches[0]
            selected_indices.add(index)
            _source_label, source_matrix, source_permutation = operations[index]
            selected.append((canonical_label, source_matrix, source_permutation))
        return tuple(selected)

    for order in (6, 5, 4, 3, 2):
        generators = [
            matrix
            for matrix in matrices
            if float(np.linalg.det(matrix)) > 0.0 and matrix_order(matrix) == order
        ]
        for rotation in generators:
            rotation_inverse = rotation.T
            involutions = [
                matrix
                for matrix in matrices
                if (float(np.linalg.det(matrix)) > 0.0) == (group == "DINFH")
                and np.allclose(matrix @ matrix, identity, atol=1.0e-8, rtol=0.0)
                and np.allclose(
                    matrix @ rotation @ matrix,
                    rotation_inverse,
                    atol=1.0e-8,
                    rtol=0.0,
                )
                and not np.allclose(matrix, identity, atol=1.0e-8, rtol=0.0)
            ]
            for involution in involutions:
                rotations = tuple(np.linalg.matrix_power(rotation, power) for power in range(order))
                base = (*rotations, *(involution @ item for item in rotations))
                rotation_labels = tuple(
                    "E" if power == 0 else f"C{order}z^{power}"
                    for power in range(order)
                )
                if group == "DINFH":
                    if not any(
                        np.allclose(matrix, inversion, atol=1.0e-8, rtol=0.0)
                        for matrix in matrices
                    ):
                        continue
                    generated = (*base, *(inversion @ item for item in base))
                    c2_labels = tuple(f"C2_xy_{order}_{power}" for power in range(order))
                    reflected_rotation_labels = tuple(
                        f"sigma_h*C{order}z^{(power - order // 2) % order}"
                        for power in range(order)
                    )
                    reflected_c2_labels = tuple(
                        f"sigma_v_{order}_{power}" for power in range(order)
                    )
                    canonical_labels = (
                        *rotation_labels,
                        *c2_labels,
                        *reflected_rotation_labels,
                        *reflected_c2_labels,
                    )
                    effective_group = f"D{order}h"
                else:
                    generated = base
                    canonical_labels = (
                        *rotation_labels,
                        *(f"sigma_v_{order}_{power}" for power in range(order)),
                    )
                    effective_group = f"C{order}v"
                selected = matched_records(tuple(generated), tuple(canonical_labels))
                if selected is not None and len(selected) == len(generated):
                    return selected, effective_group
    raise ValueError(
        f"SMITH cannot extract a closed finite subgroup from the {physical_point_group} "
        "ORACLE operation sample"
    )


def _external_motion_basis(
    coordinates: np.ndarray,
    site_frame: OnicSiteFrame,
    *,
    tolerance: float,
) -> tuple[np.ndarray, int, str]:
    natoms = len(coordinates)
    centered = coordinates - np.mean(coordinates, axis=0)
    frame_axes = np.asarray(site_frame.axes_global, dtype=float)
    candidates: list[np.ndarray] = []
    for axis in frame_axes:
        candidates.append(np.tile(axis, natoms))
    for axis in frame_axes:
        candidates.append(np.asarray([np.cross(axis, row) for row in centered]).reshape(-1))
    normalized = []
    for candidate in candidates:
        norm = float(np.linalg.norm(candidate))
        normalized.append(candidate / norm if norm > tolerance else np.zeros_like(candidate))
    selection = select_rank_revealing_rows(
        np.asarray(normalized, dtype=float),
        target_rank=6,
        tolerance=tolerance,
    )
    external_mode_count = selection.rank
    if natoms == 1 and external_mode_count == 3:
        linearity = "MONATOMIC"
    elif external_mode_count == 5:
        linearity = "LINEAR"
    elif external_mode_count == 6:
        linearity = "NONLINEAR"
    else:
        raise ValueError(
            f"Cartesian subset has unsupported rigid-motion rank {external_mode_count}"
        )
    return np.asarray(selection.orthonormal_basis, dtype=float), external_mode_count, linearity


def _site_frame_seed_rows(
    symbols: tuple[str, ...],
    coordinates: np.ndarray,
    site_frame: OnicSiteFrame,
) -> np.ndarray:
    axes = np.asarray(site_frame.axes_global, dtype=float)
    origin = np.asarray(site_frame.origin_angstrom, dtype=float)
    site_coordinates = (coordinates - origin) @ axes.T
    canonical_atoms = sorted(
        range(len(symbols)),
        key=lambda index: (
            atomic_number(symbols[index]),
            tuple(float(value) for value in np.round(site_coordinates[index], 12)),
        ),
    )
    rows: list[np.ndarray] = []
    for atom_index in canonical_atoms:
        for axis in axes:
            row = np.zeros(3 * len(symbols), dtype=float)
            row[3 * atom_index : 3 * atom_index + 3] = axis
            rows.append(row)
    return np.asarray(rows, dtype=float)


def _identity_operation_index(
    rotations: tuple[np.ndarray, ...],
    permutations: tuple[tuple[int, ...], ...],
) -> int:
    for index, (rotation, permutation) in enumerate(zip(rotations, permutations, strict=True)):
        if np.allclose(rotation, np.eye(3), atol=1.0e-10, rtol=0.0) and permutation == tuple(
            range(len(permutation))
        ):
            return index
    raise ValueError("retained ORACLE operation set has no identity")


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    return 0.5 * (array + array.T)


__all__ = [
    "CARTESIAN_BLOCK_ABSOLUTE_RANK_TOLERANCE",
    "CARTESIAN_BLOCK_COVARIANCE_TOLERANCE",
    "CARTESIAN_BLOCK_GAUGE",
    "CARTESIAN_BLOCK_RANK_METHOD",
    "CARTESIAN_BLOCK_RELATIVE_RANK_TOLERANCE",
    "CartesianIrrepSubspace",
    "SymmetryAdaptedCartesianBasis",
    "build_symmetry_adapted_cartesian_block",
    "evaluate_symmetry_adapted_cartesian_block",
    "evaluate_symmetry_adapted_cartesian_block_values",
    "onic_degeneracy_groups_from_cartesian_basis",
    "symmetry_adapted_cartesian_basis",
]
