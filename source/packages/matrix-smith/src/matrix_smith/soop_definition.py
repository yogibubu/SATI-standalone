"""Materialize a full mixed SOOP/curvilinear chart for ordinary LINK consumers.

SOOP rows have a frozen Cartesian frame. Retained valence coordinates remain
nonlinear. This explicit factory does not silently change existing contracts,
recenter a trajectory, or claim invariance to finite rotation of its reference.
"""
from dataclasses import replace
import numpy as np

from .models import GICPrimitive, FrozenGIC
from .coupled_ring_reduction import coupled_ring_reduction_plan_from_oracle_contract
from .selective_out_of_plane import (
    build_selective_out_of_plane_plan,
    build_ring_soop_displacements,
)


def refresh_frozen_soop_definition(
    reference_definition,
    frozen_contract,
    coordinates_angstrom,
    *,
    symmetry_operations=(),
):
    """Reorient fixed-support SOOP planes without reperceiving topology.

    The centers, moving atoms, plane atoms, ring owners, and coordinate count
    remain frozen. Only the plane normals and Cartesian reference positions
    follow the accepted molecular geometry.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (frozen_contract.primary_topology.natoms, 3):
        raise ValueError("SOOP refresh geometry has the wrong molecular frame")
    updated = []
    for primitive in frozen_contract.selective_out_of_plane_primitives:
        center, first, second = (atom - 1 for atom in primitive.plane_atoms)
        first_vector = xyz[first] - xyz[center]
        second_vector = xyz[second] - xyz[center]
        normal = np.cross(first_vector, second_vector)
        magnitude = float(np.linalg.norm(normal))
        scale = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
        regular = scale > 1.0e-20 and magnitude > 1.0e-10 * scale
        updated.append(replace(
            primitive,
            reference_position_angstrom=tuple(float(value) for value in xyz[primitive.moving_atom - 1]),
            normal=tuple(float(value) for value in normal / magnitude) if regular else (),
            status="REGULAR" if regular else "SINGULAR_PLANE",
        ))
    contract = replace(
        frozen_contract,
        selective_out_of_plane_primitives=tuple(updated),
    )
    definition = replace(
        reference_definition,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in xyz
        ),
    )
    return materialize_soop_definition(
        definition,
        contract,
        symmetry_operations=symmetry_operations,
    )


def materialize_reference_soop_definition(
    reference_definition,
    current_definition,
    oracle_pool,
    *,
    symmetry_operations=(),
):
    """Keep a reference coordinate selection while refreshing its ORACLE binding.

    The reference fixes the number and chemical families of GICs. Primitive
    identifiers are rebound by their physical Wilson B rows at the accepted
    geometry; no molecule-specific atom or coordinate choice enters here.
    """
    from .numerics import _analytic_b_row
    from .evaluation import build_gic_b_matrix

    xyz = np.asarray(current_definition.reference_coordinates_angstrom, dtype=float)
    if len(reference_definition.gics) != current_definition.target_rank:
        raise SOOPMaterializationError("reference and current vibrational ranks differ")
    if (reference_definition.point_group != current_definition.point_group or
            reference_definition.symmetrize != current_definition.symmetrize):
        raise SOOPMaterializationError("reference and current symmetry contracts differ")
    candidate_rows = {
        candidate.identifier: np.asarray(_analytic_b_row(candidate, xyz), dtype=float)
        for candidate in oracle_pool.candidates
    }
    binding = {}
    bound_primitives = {}
    for primitive in reference_definition.primitives:
        row = np.asarray(_analytic_b_row(primitive, xyz), dtype=float)
        norm = float(np.linalg.norm(row))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise SOOPMaterializationError("reference primitive has a singular Wilson B row")
        matches = []
        for candidate in oracle_pool.candidates:
            if candidate.function != primitive.function or candidate.family != primitive.family:
                continue
            trial = candidate_rows[candidate.identifier]
            denominator = float(trial @ trial)
            if denominator <= 1.0e-24:
                continue
            factor = float(row @ trial) / denominator
            error = float(np.linalg.norm(row - factor * trial) / norm)
            matches.append((error, candidate.identifier, factor, candidate))
        if not matches:
            raise SOOPMaterializationError(
                f"no current ORACLE candidate for {primitive.identifier}"
            )
        error, identifier, factor, candidate = min(matches, key=lambda item: (item[0], item[1]))
        if error > 1.0e-8:
            raise SOOPMaterializationError(
                f"reference primitive {primitive.identifier} changed physical direction ({error:.3g})"
            )
        binding[primitive.identifier] = (identifier, factor)
        bound_primitives[identifier] = candidate
    bound_gics = []
    for gic in reference_definition.gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        mapped = {}
        for identifier, coefficient in coefficients:
            current_id, factor = binding[identifier]
            mapped[current_id] = mapped.get(current_id, 0.0) + float(coefficient) * factor
        terms = tuple((identifier, value) for identifier, value in mapped.items() if abs(value) > 1.0e-12)
        if not terms:
            raise SOOPMaterializationError(f"reference GIC {gic.identifier} vanished on rebinding")
        bound_gics.append(replace(gic, primitive_id=terms[0][0], coefficients=terms))
    rebound = replace(
        reference_definition,
        backend=current_definition.backend,
        reference_coordinates_angstrom=current_definition.reference_coordinates_angstrom,
        primitives=tuple(bound_primitives.values()),
        gics=tuple(bound_gics),
        candidate_count=len(oracle_pool.candidates),
        primitive_source=current_definition.primitive_source,
        primitive_source_schema=current_definition.primitive_source_schema,
        primitive_b_matrix_sha256=current_definition.primitive_b_matrix_sha256,
        semantic_diagnostics=current_definition.semantic_diagnostics,
    )
    matrix = np.asarray(build_gic_b_matrix(rebound, coordinates_angstrom=xyz).rows)
    if np.linalg.matrix_rank(matrix) != len(rebound.gics):
        raise SOOPMaterializationError("rebound reference coordinates lost vibrational rank")
    return materialize_soop_definition(
        rebound,
        oracle_pool.contract,
        symmetry_operations=symmetry_operations,
    )


_MAX_COUPLED_RING_SOURCE_CONDITION = 1.0e8


class SOOPMaterializationError(ValueError):
    """SOOP chart failure carrying the ORACLE fragments to route through SYCART."""

    def __init__(self, message, *, fragment_ids=(), atom_indices_zero_based=()):
        super().__init__(message)
        self.fragment_ids = tuple(str(value) for value in fragment_ids)
        self.atom_indices_zero_based = tuple(
            sorted({int(value) for value in atom_indices_zero_based})
        )


def materialize_soop_definition(
    definition,
    contract,
    *,
    validate_basis: bool = True,
    symmetry_operations=(),
):
    """Materialize SOOP or report the complete fragments requiring SYCART."""

    try:
        return _materialize_soop_definition(
            definition,
            contract,
            validate_basis=validate_basis,
            symmetry_operations=symmetry_operations,
        )
    except SOOPMaterializationError:
        raise
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        fragment_ids, atoms = _soop_failure_domains(contract)
        raise SOOPMaterializationError(
            str(exc),
            fragment_ids=fragment_ids,
            atom_indices_zero_based=atoms,
        ) from exc


def _soop_failure_domains(contract):
    """Return whole ORACLE fragments touched by a failed SOOP construction."""

    records = tuple(contract.selective_out_of_plane_primitives)
    singular_atoms = {
        atom
        for record in records
        if record.status != "REGULAR"
        for atom in (
            record.center_atom,
            record.moving_atom,
            *record.plane_atoms,
        )
    }
    support_atoms = singular_atoms or {
        atom
        for record in records
        for atom in (
            record.center_atom,
            record.moving_atom,
            *record.plane_atoms,
        )
    }
    fragments = tuple(
        fragment
        for fragment in contract.primary_topology.fragments
        if support_atoms.intersection(fragment.atoms)
    )
    if not fragments:
        fragments = tuple(contract.primary_topology.fragments)
    return (
        tuple(fragment.fragment_id for fragment in fragments),
        tuple(atom - 1 for fragment in fragments for atom in fragment.atoms),
    )


def _materialize_soop_definition(
    definition,
    contract,
    *,
    validate_basis: bool = True,
    symmetry_operations=(),
):
    """Replace ring and terminal OOP blocks; retain all other coordinate functions.

    Requires exact-rank closure. Missing ring primitives, fused-ring
    dependencies and unresolved symmetry mixing raise errors, not U fallbacks.
    Frame coefficients and offsets are serialized in the primitive refs.
    """
    from .evaluation import build_gic_b_matrix, build_primitive_b_matrix
    xyz = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    if any(p.function == "SOOP" for p in definition.primitives):
        raise ValueError("definition already contains SOOP coordinates")
    source_primitive_by_id = {
        primitive.identifier: primitive for primitive in definition.primitives
    }
    # A saturated tetracoordinate center has no umbrella coordinate: its five
    # angular degrees of freedom are spanned by redundant valence angles.  Some
    # older symmetry projectors used U rows internally to orient that angular
    # space.  Re-express those rows exactly in the A-primitive pool before the
    # SOOP selection, so no legacy out-of-plane primitive is delivered and no
    # nonexistent tetrahedral SOOP pool is requested.
    adjacency = {
        atom: set() for atom in range(1, contract.primary_topology.natoms + 1)
    }
    for left, right in contract.primary_topology.bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    legacy_centers = {
        primitive.identifier: int(primitive.atoms[0])
        for primitive in definition.primitives
        if primitive.function in {"U", "IMPD"} and primitive.atoms
    }
    tetrahedral_rows = []
    for index, coordinate in enumerate(definition.gics):
        components = coordinate.coefficients or ((coordinate.primitive_id, 1.0),)
        centers = {
            legacy_centers[identifier]
            for identifier, _coefficient in components
            if identifier in legacy_centers
        }
        if (
            coordinate.family == "COUPLED_ANGULAR"
            and centers
            and all(len(adjacency[center]) == 4 for center in centers)
            and all(
                primitive.center_atom not in centers
                for primitive in contract.selective_out_of_plane_primitives
            )
        ):
            tetrahedral_rows.append((index, centers))
    if tetrahedral_rows:
        gic_rows = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=xyz).rows, dtype=float
        )
        angle_definition = replace(
            definition,
            primitives=tuple(
                primitive for primitive in definition.primitives
                if primitive.function == "A"
            ),
            gics=(),
        )
        primitive_matrix = build_primitive_b_matrix(
            angle_definition, coordinates_angstrom=xyz
        )
        primitive_rows = np.asarray(primitive_matrix.rows, dtype=float)
        primitive_index = {
            label: index for index, label in enumerate(primitive_matrix.coordinate_labels)
        }
        rewritten = list(definition.gics)
        for index, centers in tetrahedral_rows:
            angle_ids = tuple(
                primitive.identifier
                for primitive in definition.primitives
                if primitive.function == "A"
                and len(primitive.atoms) == 3
                and int(primitive.atoms[1]) in centers
            )
            candidate = primitive_rows[[primitive_index[item] for item in angle_ids]]
            coefficients, *_ = np.linalg.lstsq(candidate.T, gic_rows[index], rcond=1.0e-10)
            residual = candidate.T @ coefficients - gic_rows[index]
            if not angle_ids or np.linalg.norm(residual) > 1.0e-8 * max(
                1.0, np.linalg.norm(gic_rows[index])
            ):
                raise SOOPMaterializationError(
                    "tetracoordinate angular row is not spanned by valence angles",
                    atom_indices_zero_based=tuple(center - 1 for center in centers),
                )
            terms = tuple(
                (identifier, float(coefficient))
                for identifier, coefficient in zip(angle_ids, coefficients, strict=True)
                if abs(float(coefficient)) > 1.0e-12
            )
            rewritten[index] = replace(
                rewritten[index], primitive_id=terms[0][0], coefficients=terms
            )
        definition = replace(definition, gics=tuple(rewritten))

    # ORACLE deliberately supplies no SOOP umbrella at saturated and
    # hypercoordinate centers.  Rank reduction can nevertheless select U
    # covectors there, especially when the selected A subset is sparse.  Use
    # the complete frozen ORACLE angle pool to express those local angular
    # directions; reserving SOOP substitution for centers that actually have
    # a physical selective displacement avoids requesting nonexistent SOOPs.
    soop_centers = {
        primitive.center_atom
        for primitive in contract.selective_out_of_plane_primitives
    }
    angle_candidates = {}
    for candidate in contract.primitive_candidates:
        if candidate.function == "A" and len(candidate.atoms) == 3:
            angle_candidates.setdefault(int(candidate.atoms[1]), []).append(candidate)
    if angle_candidates and "SCIENTIFIC_PATH MINIMUM" in definition.semantic_diagnostics:
        from .numerics import _analytic_b_row
        from .oracle_contract_consumer import _primitive_from_candidate

        rewritten = list(definition.gics)
        added_primitives = {}
        angle_rows = {}
        for center, candidates in angle_candidates.items():
            if center not in soop_centers:
                angle_rows[center] = np.asarray(
                    [_analytic_b_row(_primitive_from_candidate(item), xyz)
                     for item in candidates], dtype=float
                )
        for index, coordinate in enumerate(definition.gics):
            components = coordinate.coefficients or ((coordinate.primitive_id, 1.0),)
            source = tuple(source_primitive_by_id[identifier] for identifier, _ in components)
            if not source or not all(
                primitive.function in {"U", "IMPD"}
                and primitive.atoms
                and int(primitive.atoms[0]) not in soop_centers
                for primitive in source
            ):
                continue
            centers = sorted({int(primitive.atoms[0]) for primitive in source})
            candidates = tuple(
                candidate
                for center in centers
                for candidate in angle_candidates.get(center, ())
            )
            if not candidates:
                continue
            matrix = np.vstack([angle_rows[center] for center in centers])
            target = sum(
                float(weight) * _analytic_b_row(primitive, xyz)
                for primitive, (_identifier, weight) in zip(source, components, strict=True)
            )
            coefficients, *_ = np.linalg.lstsq(matrix.T, target, rcond=1.0e-10)
            residual = matrix.T @ coefficients - target
            if np.linalg.norm(residual) > 1.0e-8 * max(1.0, np.linalg.norm(target)):
                continue
            terms = tuple(
                (candidate.candidate_id, float(value))
                for candidate, value in zip(candidates, coefficients, strict=True)
                if abs(float(value)) > 1.0e-12
            )
            if not terms:
                raise SOOPMaterializationError("non-SOOP angular row vanished")
            for candidate in candidates:
                if any(identifier == candidate.candidate_id for identifier, _ in terms):
                    added_primitives[candidate.candidate_id] = _primitive_from_candidate(candidate)
            rewritten[index] = replace(
                coordinate,
                family="BEND" if coordinate.family == "OUT_OF_PLANE" else coordinate.family,
                primitive_id=terms[0][0],
                gaussian_expression="NONE",
                coefficients=terms,
            )
        if added_primitives:
            existing = {primitive.identifier for primitive in definition.primitives}
            definition = replace(
                definition,
                primitives=(*definition.primitives, *(
                    primitive for identifier, primitive in added_primitives.items()
                    if identifier not in existing
                )),
                gics=tuple(rewritten),
                semantic_diagnostics=(
                    *definition.semantic_diagnostics,
                    "SOOP_NONSELECTIVE_ANGULAR_ROWS=FROZEN_ORACLE_VALENCE_ANGLES",
                ),
            )

    def is_contact_pose_support(coordinate):
        components = coordinate.coefficients or ((coordinate.primitive_id, 1.0),)
        primitives = tuple(
            source_primitive_by_id[primitive_id]
            for primitive_id, _coefficient in components
            if primitive_id in source_primitive_by_id
        )
        # Projected TS contact-pose rows may mix the explicitly tagged contact
        # support with ordinary U covectors from the same local mathematical
        # completion.  They remain interfragment pose coordinates as a whole;
        # they are not molecular valence out-of-plane coordinates.
        return bool(primitives) and any(
            primitive.refs and primitive.refs[0] == "PSEUDOBOND_CONTACT_SUPPORT"
            for primitive in primitives
        )
    # All ring puckering, including the provisional CHARM/flap block at
    # tetracoordinate centers, is re-expressed in ORACLE's endocyclic SOOP
    # directions.  Retaining that block would constrain some selective
    # displacements and falsely make the external SOOP pool look deficient.
    ring_indices = [
        i
        for i, coordinate in enumerate(definition.gics)
        if coordinate.family in {"RING_PUCKER_COMPONENT", "BUTTERFLY"}
    ]
    external_indices = [
        index
        for index, coordinate in enumerate(definition.gics)
        if coordinate.family == "OUT_OF_PLANE"
        and not is_contact_pose_support(coordinate)
    ]
    legacy_ids = {
        primitive.identifier
        for primitive in definition.primitives
        if primitive.function in {"U", "IMPD"}
    }
    # A derived SALC may mix U rows into a family whose public label is not
    # OUT_OF_PLANE (for example the coupled angular block of allene).  The
    # SOOP-default protocol applies to the primitive support, not merely to the
    # final family label, so replace every delivered coordinate that depends
    # on a legacy out-of-plane primitive.
    derived_legacy_indices = [
        index
        for index, coordinate in enumerate(definition.gics)
        if not is_contact_pose_support(coordinate)
        and any(
            identifier in legacy_ids
            for identifier, _coefficient in (
                coordinate.coefficients
                or ((coordinate.primitive_id, 1.0),)
            )
        )
    ]
    selected = list(dict.fromkeys((*ring_indices, *external_indices, *derived_legacy_indices)))
    if not selected:
        # The ORACLE primitive pool may contain provisional U/IMPD candidates
        # even when rank reduction selected none of them.  They are not part
        # of the delivered chart and must not leak into a frozen KEYMAKER
        # artifact: retain exactly the primitive support of the final GICs.
        referenced = {
            identifier
            for coordinate in definition.gics
            for identifier, _coefficient in (
                coordinate.coefficients
                or ((coordinate.primitive_id, 1.0),)
            )
        }
        primitives = tuple(
            primitive
            for primitive in definition.primitives
            if primitive.identifier in referenced
        )
        if any(primitive.function in {"U", "IMPD"} for primitive in primitives):
            raise ValueError("delivered coordinates retain legacy out-of-plane primitives")
        return replace(
            definition,
            backend="matrix-smith-soop-mixed.v1",
            primitives=primitives,
            candidate_count=len(primitives),
            semantic_diagnostics=(
                *definition.semantic_diagnostics,
                "KEYMAKER_COORDINATE_PROTOCOL=sonic-soop-default.v1",
            ),
        )
    b = np.asarray(build_gic_b_matrix(definition, coordinates_angstrom=xyz).rows)
    if np.linalg.matrix_rank(b) != len(definition.gics):
        raise ValueError("input coordinate definition is not full rank")
    retained = [i for i in range(len(definition.gics)) if i not in selected]
    retained_rows = b[retained]
    # A selective displacement is defined only up to an infinitesimal rigid
    # translation or rotation.  Fix that gauge before dualizing the SOOP
    # directions.  Projecting only against the retained internal rows leaves
    # rigid-body components in the frozen Cartesian columns; their Euclidean
    # duals then cease to be genuine internal-coordinate covectors and pollute
    # the low-frequency GF spectrum.  The joint null space below preserves the
    # intended relative (ring versus substituent) motion while enforcing both
    # the retained-coordinate and Eckart tangent constraints exactly.
    centered = xyz - xyz.mean(axis=0)
    external_columns = []
    for axis in np.eye(3):
        external_columns.append(np.tile(axis, (len(xyz), 1)).reshape(-1))
    for axis in np.eye(3):
        external_columns.append(np.cross(axis, centered).reshape(-1))
    external = np.column_stack(external_columns)
    external_left, external_values, _ = np.linalg.svd(external, full_matrices=False)
    external_rank = int(np.count_nonzero(external_values > 1.0e-12 * external_values[0]))
    external = external_left[:, :external_rank]
    tangent_constraints = np.vstack((retained_rows, external.T))
    internal_projector = np.eye(xyz.size) - (
        np.linalg.pinv(tangent_constraints, rcond=1e-10) @ tangent_constraints
    )
    plan = build_selective_out_of_plane_plan(contract, ring_representation="soop")
    coupled_ring_plan = coupled_ring_reduction_plan_from_oracle_contract(contract)
    old_directions = np.linalg.pinv(b, rcond=1e-10)[:, selected]
    rings = build_ring_soop_displacements(
        contract,
        xyz,
        target_rank=len(ring_indices),
        complete_pool=True,
    )
    ring_directions = rings.displacements
    external_primitives = tuple(
        primitive for primitive in contract.selective_out_of_plane_primitives
        if primitive.status == "REGULAR" and not primitive.endocyclic
    )
    external_directions = np.column_stack((
        plan.displacements,
        *(
            primitive.displacement(contract.primary_topology.natoms).ravel()
            for primitive in external_primitives
        ),
    )) if external_primitives else plan.displacements
    # Frozen records must belong to this exact molecular reference frame.
    for p in contract.selective_out_of_plane_primitives:
        if not np.allclose(xyz[p.moving_atom-1],p.reference_position_angstrom,atol=1e-10,rtol=0):
            raise ValueError("stale SOOP reference geometry: regenerate with ORACLE")
        if p.status == "REGULAR":
            center, first, second = (atom - 1 for atom in p.plane_atoms)
            normal = np.cross(xyz[first] - xyz[center], xyz[second] - xyz[center])
            magnitude = np.linalg.norm(normal)
            if magnitude <= 1e-14 or not np.allclose(
                normal / magnitude, p.normal, atol=1e-10, rtol=0
            ):
                raise ValueError("stale SOOP face plane: regenerate with ORACLE")
    # Hold every retained internal coordinate fixed to first order.  Within
    # each ORACLE ring-component symmetry orbit, endocyclic and attached
    # exocyclic SOOP directions deliberately enter one joint pool.  This is
    # the coupled local angular space: separating its ring and substituent
    # parts before rank reduction creates the dependence that this contract is
    # designed to remove.  Disconnected, non-symmetry-related components and
    # genuinely acyclic SOOPs remain separate blocks.
    aligned = np.zeros((xyz.size, len(selected)), dtype=float)
    labels = [definition.gics[index].irrep for index in selected]
    unassigned_labels = {None, "", "UNASSIGNED", "UNKNOWN", "NONE"}
    occupied = np.empty((xyz.size, 0), dtype=float)

    def component_failure(orbit, message):
        domains = tuple(
            domain
            for domain in coupled_ring_plan.domains
            if domain.symmetry_orbit_id == orbit
        )
        atoms_one_based = {
            atom
            for domain in domains
            for atom in (*domain.core_atoms, *domain.attachment_atoms)
        }
        fragments = tuple(
            fragment.fragment_id
            for fragment in contract.primary_topology.fragments
            if atoms_one_based.intersection(fragment.atoms)
        )
        raise SOOPMaterializationError(
            message,
            fragment_ids=fragments,
            atom_indices_zero_based=tuple(atom - 1 for atom in atoms_one_based),
        )

    def close_semantic_block(
        positions,
        candidate_directions,
        block_name,
        *,
        failure_orbit=None,
    ):
        nonlocal occupied
        positions = [int(position) for position in positions]
        if not positions:
            return
        candidates = internal_projector @ np.asarray(candidate_directions, dtype=float)
        # Candidate directions come exclusively from the ORACLE SOOP pool.
        # The provisional chart supplies irrep orientation below, not missing
        # physical directions.
        if occupied.shape[1]:
            candidates -= occupied @ (occupied.T @ candidates)
        left, singular, _right = np.linalg.svd(candidates, full_matrices=False)
        rank = int(np.count_nonzero(singular > 1.0e-10))
        if rank < len(positions):
            message = (
                f"{block_name} SOOP motions span "
                f"{rank}/{len(positions)} required directions"
            )
            if failure_orbit is not None:
                component_failure(failure_orbit, message)
            raise ValueError(message)
        source_condition = float(singular[0] / singular[len(positions) - 1])
        if (
            not np.isfinite(source_condition)
            or source_condition > _MAX_COUPLED_RING_SOURCE_CONDITION
        ):
            message = (
                f"{block_name} SOOP source condition {source_condition:.12g} exceeds "
                f"{_MAX_COUPLED_RING_SOURCE_CONDITION:.12g}"
            )
            if failure_orbit is not None:
                component_failure(failure_orbit, message)
            raise ValueError(message)
        basis = left[:, :rank]
        block_occupied = np.empty((xyz.size, 0), dtype=float)

        def align_irrep(group_positions, label):
            nonlocal block_occupied
            target = old_directions[:, group_positions]
            projected = basis @ (basis.T @ target)
            if block_occupied.shape[1]:
                projected -= block_occupied @ (block_occupied.T @ projected)
            group_left, group_singular, _group_right = np.linalg.svd(
                projected, full_matrices=False
            )
            if group_singular[len(group_positions) - 1] <= 1.0e-10:
                if not str(label).endswith("_UNASSIGNED_COMPLEMENT"):
                    message = f"SOOP motions do not close symmetry block {label}"
                    if failure_orbit is not None:
                        component_failure(failure_orbit, message)
                    raise ValueError(message)
                # No symmetry label constrains this residual block.  Use the
                # exact orthogonal complement of the already assigned irreps
                # and orient it as closely as possible to the provisional
                # chart.  Requiring full overlap with that provisional target
                # incorrectly rejects valid degenerate complements (ferrocene
                # is the smallest reproducible example).
                projected = basis.copy()
                if block_occupied.shape[1]:
                    projected -= block_occupied @ (block_occupied.T @ projected)
                group_left, group_singular, _group_right = np.linalg.svd(
                    projected, full_matrices=False
                )
                if group_singular[len(group_positions) - 1] <= 1.0e-10:
                    message = f"SOOP motions do not close symmetry block {label}"
                    if failure_orbit is not None:
                        component_failure(failure_orbit, message)
                    raise ValueError(message)
            group_basis = group_left[:, :len(group_positions)]
            rotation_left, _values, rotation_right = np.linalg.svd(
                group_basis.T @ target, full_matrices=False
            )
            vectors = group_basis @ rotation_left @ rotation_right
            aligned[:, group_positions] = vectors
            block_occupied = np.column_stack((block_occupied, vectors))

        irreps = []
        for position in positions:
            label = labels[position]
            if label not in unassigned_labels and label not in irreps:
                irreps.append(label)
        for irrep in irreps:
            align_irrep(
                [position for position in positions if labels[position] == irrep],
                irrep,
            )
        unassigned = [
            position for position in positions if labels[position] in unassigned_labels
        ]
        if unassigned:
            align_irrep(unassigned, f"{block_name}_UNASSIGNED_COMPLEMENT")
        occupied = np.column_stack((occupied, block_occupied))

    primitive_by_id = {
        primitive.identifier: primitive for primitive in definition.primitives
    }
    orbit_by_endocyclic_id = {
        identifier: domain.symmetry_orbit_id
        for domain in coupled_ring_plan.domains
        for identifier in domain.endocyclic_soop_ids
    }
    orbit_by_attached_id = {
        identifier: domain.symmetry_orbit_id
        for domain in coupled_ring_plan.domains
        for identifier in domain.attached_exocyclic_soop_ids
    }
    core_by_orbit: dict[str, set[int]] = {}
    for domain in coupled_ring_plan.domains:
        core_by_orbit.setdefault(domain.symmetry_orbit_id, set()).update(
            domain.core_atoms
        )

    def coordinate_orbit(local_position):
        coordinate = definition.gics[selected[int(local_position)]]
        component_ids = coordinate.coefficients or ((coordinate.primitive_id, 1.0),)
        primitives = [
            primitive_by_id[identifier]
            for identifier, _coefficient in component_ids
            if identifier in primitive_by_id
        ]
        if coordinate.family == "OUT_OF_PLANE":
            support = {
                int(primitive.atoms[0])
                for primitive in primitives
                if primitive.atoms
            }
        else:
            support = {
                int(atom)
                for primitive in primitives
                for atom in (
                    *primitive.atoms,
                    *primitive.ref_atoms,
                    *primitive.frame_atoms,
                    *primitive.ref_frame_atoms,
                )
            }
        matches = {
            orbit
            for orbit, atoms in core_by_orbit.items()
            if support.intersection(atoms)
        }
        if len(matches) > 1:
            raise ValueError(
                f"coordinate {coordinate.identifier} crosses unrelated ring-component orbits"
            )
        if coordinate.family in {"RING_PUCKER_COMPONENT", "BUTTERFLY"} and not matches:
            raise ValueError(
                f"cyclic coordinate {coordinate.identifier} has no ORACLE ring component"
            )
        return next(iter(matches), None)

    position_orbits = tuple(coordinate_orbit(position) for position in range(len(selected)))

    def unique_orbit(primitive_ids, *, attached=False):
        owner = orbit_by_attached_id if attached else orbit_by_endocyclic_id
        matches = {owner[identifier] for identifier in primitive_ids if identifier in owner}
        if len(matches) > 1:
            raise ValueError("SOOP candidate crosses unrelated ring-component orbits")
        return next(iter(matches), None)

    ring_column_orbits = [
        unique_orbit(
            tuple(
                identifier
                for identifier, coefficient in zip(
                    rings.primitive_ids,
                    rings.coefficients[:, column],
                    strict=True,
                )
                if abs(float(coefficient)) > 1.0e-14
            )
        )
        for column in range(rings.coefficients.shape[1])
    ]

    external_column_orbits = [
        unique_orbit(combination.primitive_ids, attached=True)
        for combination in plan.combinations
    ]
    external_column_orbits.extend(
        unique_orbit((primitive.identifier,), attached=True)
        for primitive in external_primitives
    )
    if len(ring_column_orbits) != ring_directions.shape[1]:
        raise ValueError("ring SOOP column ownership is inconsistent")
    if len(external_column_orbits) != external_directions.shape[1]:
        raise ValueError("external SOOP column ownership is inconsistent")

    candidate_columns: dict[str | None, list[np.ndarray]] = {}
    for column, orbit in zip(ring_directions.T, ring_column_orbits, strict=True):
        candidate_columns.setdefault(orbit, []).append(column)
    for column, orbit in zip(external_directions.T, external_column_orbits, strict=True):
        candidate_columns.setdefault(orbit, []).append(column)

    reduction_blocks = []
    ordered_orbits = [
        orbit for orbit in sorted(core_by_orbit) if orbit in set(position_orbits)
    ]
    if None in position_orbits:
        ordered_orbits.append(None)
    for orbit in ordered_orbits:
        positions = [
            position
            for position, position_orbit in enumerate(position_orbits)
            if position_orbit == orbit
        ]
        columns = candidate_columns.get(orbit, [])
        if not columns:
            name = "acyclic" if orbit is None else orbit
            raise ValueError(f"{name} has no SOOP candidate directions")
        block_name = "acyclic-out-of-plane" if orbit is None else str(orbit)
        close_semantic_block(
            positions,
            np.column_stack(columns),
            block_name,
            failure_orbit=orbit,
        )
        reduction_blocks.append(tuple(positions))
    if definition.symmetrize and 1 < len(symmetry_operations) <= 8:
        aligned = _close_soop_symmetry_subspaces(
            aligned,
            blocks=tuple(reduction_blocks),
            symmetry_operations=symmetry_operations,
            internal_projector=internal_projector,
        )
    if not np.allclose(
        aligned.T @ aligned,
        np.eye(len(selected)),
        atol=1.0e-9,
        rtol=1.0e-9,
    ):
        raise ValueError("SOOP symmetry blocks are not mutually orthogonal")
    replacement = aligned.T
    dual = b.copy()
    dual[selected] = replacement
    if np.linalg.matrix_rank(dual) != len(definition.gics):
        raise ValueError("SOOP substitution does not preserve the full coordinate rank")
    primitives = list(definition.primitives)
    gics = list(definition.gics)
    used_ids = {p.identifier for p in primitives}
    for j,i in enumerate(selected):
        identifier=f"SOOP_DUAL_{i+1:06d}"
        if identifier in used_ids:
            raise ValueError("SOOP primitive identifier collision")
        row = dual[i]
        support = np.flatnonzero(abs(row)>1e-14)
        offset = float(row @ xyz.ravel())
        refs = (f"OFFSET:{offset:.17g}",) + tuple(f"B:{k}:{row[k]:.17g}" for k in support)
        atoms = tuple(sorted({int(k)//3+1 for k in support}))
        old = gics[i]
        primitives.append(GICPrimitive(identifier,old.name,old.family,"SOOP",atoms,
            refs=refs,chart="FROZEN_REFERENCE_CARTESIAN",semantic_type="SELECTIVE_OOP_DUAL"))
        gics[i] = FrozenGIC(old.identifier,old.name,old.family,old.irrep,
            identifier,"NONE",((identifier,1.),))
    referenced = {key for c in gics for key,_ in (c.coefficients or ((c.primitive_id,1.),))}
    primitives = tuple(p for p in primitives if p.identifier in referenced)
    remaining_legacy = tuple(
        primitive.identifier
        for primitive in primitives
        if primitive.function in {"U", "IMPD"}
    )
    if remaining_legacy:
        raise ValueError(
            "delivered coordinates retain legacy out-of-plane primitives: "
            + ", ".join(remaining_legacy)
        )
    result = replace(definition, backend="matrix-smith-soop-mixed.v1",
        primitives=primitives,gics=tuple(replace(c,gaussian_expression="NONE") for c in gics),candidate_count=len(primitives),
        ring_puckering_diagnostics=tuple(
            record.replace("PRIMITIVE=GAUSSIAN_U", "PRIMITIVE=SOOP")
            .replace("PRIMITIVE=GAUSSIAN_H", "PRIMITIVE=SOOP")
            for record in definition.ring_puckering_diagnostics
        ),
        semantic_diagnostics=(*definition.semantic_diagnostics,
            "KEYMAKER_COORDINATE_PROTOCOL=sonic-soop-default.v1",
            "SOOP_FRAME=FROZEN_REFERENCE_CARTESIAN",
            "SOOP_SOURCE_POOL=ORACLE_PHYSICAL_SOOP_ONLY",
            "SOOP_PROVISIONAL_U_COMPLETION=DISABLED",
            f"SOOP_COUPLED_RING_DOMAINS={len(coupled_ring_plan.domains)}",
            "SOOP_COUPLED_RING_COMPONENT_RULE=RING_ATOM_OVERLAP",
            "SOOP_COUPLED_RING_REDUCTION=JOINT_ENDO_EXO_PER_COMPONENT_ORBIT_THEN_IRREP",
            "SOOP_COUPLED_RING_LOCAL_ANGULAR_RANK=2CN_MINUS_3",
            f"SOOP_COUPLED_RING_MAX_SOURCE_CONDITION={_MAX_COUPLED_RING_SOURCE_CONDITION:.12g}",
            *(
                f"SOOP_COUPLED_RING_DOMAIN={domain.domain_id} "
                f"ORBIT={domain.symmetry_orbit_id} "
                f"RINGS={','.join(str(value) for value in domain.ring_ids)} "
                f"ENDO={len(domain.endocyclic_soop_ids)} "
                f"ATTACHED_EXO={len(domain.attached_exocyclic_soop_ids)} "
                f"CN3={sum(center.coordination == 3 for center in domain.centers)} "
                f"CN4={sum(center.coordination == 4 for center in domain.centers)}"
                for domain in coupled_ring_plan.domains
            ),
            "SOOP_ANALYTIC_B_PRIME=ZERO_ONLY_FOR_SOOP RETAINED_CURVATURE=ANALYTIC"))
    actual = np.asarray(build_gic_b_matrix(result,coordinates_angstrom=xyz).rows)
    if not np.allclose(actual,dual,atol=1e-10,rtol=1e-10):
        raise ValueError("materialized SOOP Wilson matrix differs from its audited dual")
    # Certification must apply to the delivered SOOP chart, not merely to the
    # provisional curvilinear basis from which its candidate subspaces were
    # identified. In particular, a full-rank replacement can still be too
    # ill-conditioned for stable back-transformation.
    if not validate_basis:
        return result
    from .basis_validation import validate_frozen_sonic_basis
    from .policy import RANK_TOLERANCE

    certificate = validate_frozen_sonic_basis(result, rank_tolerance=RANK_TOLERANCE)
    return replace(
        result,
        semantic_diagnostics=(
            *result.semantic_diagnostics,
            f"SOOP_FINAL_RANK={certificate['rank']}",
            f"SOOP_FINAL_NORMALIZED_CONDITION={certificate['condition_number']:.6g}",
        ),
    )


def _close_soop_symmetry_subspaces(
    aligned,
    *,
    blocks,
    symmetry_operations,
    internal_projector,
):
    """Project each semantic SOOP block onto an exactly invariant group subspace."""

    from matrix_chem import cartesian_operation_matrix

    dimension = np.asarray(aligned, dtype=float).shape[0]
    natoms = dimension // 3
    operations = []
    for operation in symmetry_operations:
        operations.append(
            cartesian_operation_matrix(
                operation.rotation,
                tuple(int(atom) - 1 for atom in operation.permutation),
                natoms=natoms,
            )
        )
    result = np.asarray(aligned, dtype=float).copy()
    occupied = np.empty((dimension, 0), dtype=float)
    for positions in blocks:
        positions = tuple(int(position) for position in positions)
        if not positions:
            continue
        source = result[:, positions]
        projector = source @ source.T
        averaged = sum(
            operation.T @ projector @ operation for operation in operations
        ) / len(operations)
        admissible = np.asarray(internal_projector, dtype=float)
        if occupied.shape[1]:
            admissible = admissible - occupied @ occupied.T
        averaged = admissible @ averaged @ admissible
        averaged = 0.5 * (averaged + averaged.T)
        values, vectors = np.linalg.eigh(averaged)
        count = len(positions)
        basis = vectors[:, np.argsort(values)[-count:]]
        left, _singular, right = np.linalg.svd(basis.T @ source, full_matrices=False)
        block = basis @ left @ right
        result[:, positions] = block
        occupied = np.column_stack((occupied, block))
    projector = result @ result.T
    residual = max(
        float(np.linalg.norm(operation.T @ projector @ operation - projector, ord=2))
        for operation in operations
    )
    # Match LINK's frozen-projector closure gate.  Smaller residuals are
    # numerical noise from the final eigenspace/Procrustes orientation, not a
    # distinct physical subspace.
    if residual > 1.0e-7:
        # Degenerate multidimensional irreps and deliberately retained
        # quasi-symmetry can make this optional averaged-projector gauge less
        # pure than the already certified SOOP basis.  Preserve that basis;
        # its rank/condition certificate remains authoritative, while LINK's
        # own projector gate still decides whether a symmetry-restricted task
        # may use it.
        return np.asarray(aligned, dtype=float)
    return result


def build_soop_definition_from_xyzin(path):
    """Read ORACLE's frozen primitives and construct an ordinary full definition."""
    from .oracle_contract_consumer import build_gic_definition_from_oracle_contract
    return build_gic_definition_from_oracle_contract(path, out_of_plane_model="soop")


def write_soop_definition_from_xyzin(path):
    """Write the full definition consumed by LINK; never regenerate perception."""
    from matrix_core import replace_section
    from .definition import gic_definition_section_lines
    definition = build_soop_definition_from_xyzin(path)
    replace_section(path, "GIC", gic_definition_section_lines(definition))
    return definition
