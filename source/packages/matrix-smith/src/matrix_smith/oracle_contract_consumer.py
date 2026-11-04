"""Fail-closed SMITH consumer for the frozen ORACLE SONIC contract.

This module contains no chemical perception. It validates ORACLE's frozen
geometry, topology, local semantics and finite Wilson rows. Production chart
construction is then delegated to SMITH's single primitive/SALC builder;
missing chemistry is an error, never an invitation to reconstruct it here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.linalg import qr as scipy_qr

from matrix_chem import (
    ATLAS_TASK_MINIMUM,
    ATLAS_TASK_TRANSITION_STATE,
    GEOMETRY_IDENTICAL,
    GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION,
    GEOMETRY_RIGID_FRAME_ONLY,
    GEOMETRY_TRUE_CHANGE,
    GeometryIdentityError,
    OracleSonicContract,
    PrimitiveCandidate,
    expected_vibrational_mode_count,
    read_enriched_xyz,
    read_geometry_identity_certificate,
    read_molecular_symmetry,
    read_oracle_coordinate_atlas_contract,
    topology_snapshot_from_xyzin,
    validate_geometry_identity_certificate,
    validate_oracle_coordinate_atlas_contract,
)
from matrix_chem.topology.elements import atomic_number
from matrix_chem.oracle_sonic_contract import primitive_candidate_wilson_row
from matrix_numerics import select_rank_revealing_rows

from .contracts import (
    GICForgeContractError,
    load_frozen_oracle_sonic_contract,
    validate_complete_frozen_oracle_semantics,
)
from .models import FrozenGIC, GICDefinition, GICPrimitive
from .bmatrix import SparseBRow
from .numerics import _analytic_b_row, _sparse_analytic_b_row
from .onic_taxonomy import onic_branch_for_role
from .policy import PRIMITIVE_FAMILY_ORDER, RANK_TOLERANCE, SPECIAL_PRIMITIVE_FAMILIES
from .symmetrization import _partition_quota_rank_revealing_selection


SMITH_ORACLE_CONSUMER_SCHEMA = "matrix.smith.oracle_contract_consumer.v1"
_ROW_VALIDATION_ATOL = 2.0e-7


@dataclass(frozen=True)
class FrozenOracleCandidateSelection:
    """Numerical selection made exclusively from a frozen ORACLE contract."""

    contract: OracleSonicContract
    candidates: tuple[GICPrimitive, ...]
    selected: tuple[GICPrimitive, ...]
    selected_rows: tuple[tuple[float, ...], ...]
    rank: int
    target_rank: int
    family_quotas: tuple[tuple[str, int], ...]
    provenance: str = SMITH_ORACLE_CONSUMER_SCHEMA


@dataclass(frozen=True)
class FrozenOracleCandidatePool:
    """Validated finite primitive functions and their ORACLE-owned rows."""

    contract: OracleSonicContract
    candidates: tuple[GICPrimitive, ...]
    rows: tuple[tuple[float, ...] | SparseBRow, ...]
    atom_symbols: tuple[str, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]


def load_validated_oracle_candidate_pool(path: Path) -> FrozenOracleCandidatePool:
    """Load all frozen candidates without selecting or interpreting chemistry."""

    contract = load_frozen_oracle_sonic_contract(Path(path))
    validate_complete_frozen_oracle_semantics(contract)
    geometry = read_enriched_xyz(Path(path))
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    _validate_frozen_geometry(Path(path), contract, tuple(geometry.atoms), coords)
    _validate_candidate_ownership(contract)
    records = tuple(
        (candidate, _primitive_from_candidate(candidate))
        for candidate in contract.primitive_candidates
    )
    validated_rows = tuple(
        _validated_supplied_row(candidate, primitive, coords)
        for candidate, primitive in records
    )
    rows = tuple(
        row
        if isinstance(row, SparseBRow)
        else tuple(float(value) for value in row)
        for row in validated_rows
    )
    return FrozenOracleCandidatePool(
        contract=contract,
        candidates=tuple(primitive for _candidate, primitive in records),
        rows=rows,
        atom_symbols=tuple(geometry.atoms),
        coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coords
        ),
    )


def select_frozen_oracle_candidates(
    path: Path,
    *,
    target_rank: int | None = None,
    family_quotas: Mapping[str, int] | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
    allow_closing_contacts: bool = False,
    excluded_functions: frozenset[str] = frozenset(),
    use_family_priorities: bool = True,
) -> FrozenOracleCandidateSelection:
    """Validate and select only ORACLE-supplied primitive candidates.

    Closing contact orbits require the dedicated contact--pose selector and
    are rejected here by default.  This prevents the generic selector from
    accidentally treating closing pseudobonds as ordinary graph edges.
    """

    pool = load_validated_oracle_candidate_pool(Path(path))
    contract = pool.contract
    coords = np.asarray(pool.coordinates_angstrom, dtype=float)

    closing_candidate_ids = {
        candidate_id
        for contact in contract.auxiliary_contacts
        if contact.open_or_closing == "CLOSING"
        and len(set(contact.fragment_ids)) > 1
        for candidate_id in contact.primitive_candidate_ids
    }
    if closing_candidate_ids and not allow_closing_contacts:
        raise GICForgeContractError(
            "closing ORACLE contact candidates require the contact-pose selector"
        )

    records_all = tuple(zip(contract.primitive_candidates, pool.candidates, strict=True))
    rows_all = tuple(
        row.to_dense() if isinstance(row, SparseBRow) else np.asarray(row, dtype=float)
        for row in pool.rows
    )
    # ORACLE may carry historical linear-bend candidates for rank recovery.
    # They are invalid for any polyhedral center (CN >= 3), even if a later
    # rank-revealing selector would otherwise choose them.
    adjacency: dict[int, set[int]] = {}
    for first, second in contract.primary_topology.bonds:
        first -= 1
        second -= 1
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    excluded = frozenset(str(value).strip().upper() for value in excluded_functions)
    records_rows = tuple(
        (record, row)
        for record, row in zip(records_all, rows_all, strict=True)
        if not (
            record[1].family == "LINEAR_BEND"
            and len(adjacency.get(record[1].atoms[1], ())) >= 3
        )
        and record[1].function.upper() not in excluded
    )
    records = tuple(record for record, _row in records_rows)
    rows = tuple(row for _record, row in records_rows)
    requested = (
        expected_vibrational_mode_count(coords)
        if target_rank is None
        else int(target_rank)
    )
    if requested < 0:
        raise GICForgeContractError("requested Wilson rank must be non-negative")

    quotas = _normalized_family_quotas(family_quotas)
    selection_contract = replace(
        contract,
        primitive_candidates=tuple(record[0] for record, _row in records_rows),
    )
    selected_indices = _select_indices(
        selection_contract,
        rows,
        target_rank=requested,
        family_quotas=quotas,
        rank_tolerance=float(rank_tolerance),
        use_family_priorities=bool(use_family_priorities),
    )
    if len(selected_indices) != requested:
        raise GICForgeContractError(
            "frozen ORACLE primitive pool has insufficient Wilson rank: "
            f"need {requested}, selected {len(selected_indices)}"
        )
    selected = tuple(records[index][1] for index in selected_indices)
    selected_rows = tuple(tuple(float(value) for value in rows[index]) for index in selected_indices)
    return FrozenOracleCandidateSelection(
        contract=contract,
        candidates=tuple(primitive for _candidate, primitive in records),
        selected=selected,
        selected_rows=selected_rows,
        rank=len(selected_indices),
        target_rank=requested,
        family_quotas=tuple(sorted(quotas.items())),
    )


def build_gic_definition_from_oracle_contract(
    path: Path,
    *,
    symmetrize: bool = False,
    orientation: str = "SONIC",
    scientific_path: str = "MINIMUM",
    family_quotas: Mapping[str, int] | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
    out_of_plane_model: str = "soop",
    validate_sonic_basis: bool = True,
    materialize_soop: bool = True,
) -> GICDefinition:
    """Validate ORACLE, then realize one ONIC chart with the single SMITH builder."""

    oop_model = str(out_of_plane_model).lower()
    if oop_model != "soop":
        raise ValueError("out_of_plane_model must be soop; legacy out-of-plane is unsupported")
    branch = onic_branch_for_role(orientation)
    path_role = _normalized_scientific_path(scientific_path)
    if family_quotas:
        raise GICForgeContractError(
            "the canonical ORACLE-to-SMITH chart does not accept family quotas"
        )
    pool = load_validated_oracle_candidate_pool(Path(path))
    _validate_frozen_fragment_handoff(Path(path), pool.contract)
    try:
        atlas = read_oracle_coordinate_atlas_contract(Path(path))
        validate_oracle_coordinate_atlas_contract(atlas)
    except ValueError as exc:
        raise GICForgeContractError(
            "canonical ORACLE-to-SMITH construction requires a valid frozen coordinate atlas"
        ) from exc
    expected_task = (
        ATLAS_TASK_MINIMUM
        if path_role == "MINIMUM"
        else ATLAS_TASK_TRANSITION_STATE
    )
    if atlas.implicit_unclassified_fragments and set(atlas.implicit_unclassified_fragments) != {
        fragment.fragment_id for fragment in pool.contract.primary_topology.fragments
    }:
        raise GICForgeContractError("implicit atlas fragment domain contradicts frozen ORACLE topology")
    # Implicit pairs are observables, not extra primitive or pseudobond
    # requests. Keep them implicit throughout coordinate construction.
    if atlas.task_regime != expected_task:
        raise GICForgeContractError(
            "ORACLE coordinate atlas task regime contradicts scientific_path"
        )
    if atlas.topology_hash != pool.contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE coordinate atlas topology fingerprint is stale")
    has_closing_atlas_interaction = any(
        item.graph_role == "CLOSING" and len(set(item.fragment_ids)) > 1
        for item in atlas.interactions
    )
    if has_closing_atlas_interaction and path_role != "MINIMUM":
        raise GICForgeContractError(
            "closed noncovalent contacts belong to the MINIMUM scientific path; "
            "a transition-state chart requires ORACLE's explicit TS contract"
        )
    has_closing_auxiliary_contact = any(
        contact.open_or_closing == "CLOSING"
        and len(set(contact.fragment_ids)) > 1
        for contact in pool.contract.auxiliary_contacts
    )

    # Primitive generation, exact-rank reduction and SALCs have one SMITH
    # implementation. This consumer validates ORACLE's frozen state and only
    # routes the explicitly distinct minimum/transition-state context.
    from .definition import build_gic_definition_from_xyzin

    try:
        definition = build_gic_definition_from_xyzin(
            Path(path),
            symmetrize=symmetrize,
            fragment_context=(
                "minimum" if path_role == "MINIMUM" else "transition_state"
            ),
            rank_tolerance=rank_tolerance,
            coordinate_atlas_contract=atlas,
            _oracle_pool=pool,
            validate_sonic_basis=bool(validate_sonic_basis),
        )
    except GICForgeContractError as exc:
        if not validate_sonic_basis or not _automatic_chart_failure(exc):
            raise
        definition = _automatic_delocalized_chart(
            Path(path), pool, atlas=atlas, rank_tolerance=rank_tolerance,
            scientific_path=path_role, trigger=str(exc),
        )
    if definition.primitive_source != "ORACLE_CONTRACT":
        raise GICForgeContractError(
            "canonical ORACLE-to-SMITH construction requires the frozen #PRIMITIVES contract"
        )


    controlled_prefixes = (
        "ONIC_CORE ",
        "CHART_ORIENTATION ",
        "CHART_ROLE ",
        "LOCAL_PERCEPTION ",
        "CLOSED_CONTACT_POLICY ",
        "SCIENTIFIC_ATLAS ",
    )
    diagnostics = tuple(
        record
        for record in definition.semantic_diagnostics
        if not record.startswith(controlled_prefixes)
    )
    contract_diagnostics = (
        "ONIC_CORE COMMON_TYPED_NONREDUNDANT_ALGEBRA",
        f"CHART_ORIENTATION {branch.acronym}",
        f"CHART_ROLE {_chart_role(branch.acronym)}",
        *_local_perception_diagnostics(pool.contract),
        f"SCIENTIFIC_ATLAS SCHEMA={atlas.schema} POLICY={atlas.policy_id}@"
        f"{atlas.policy_version} TASK={atlas.task_regime}",
    )
    if has_closing_auxiliary_contact:
        contract_diagnostics = (
            *contract_diagnostics,
            "CLOSED_CONTACT_POLICY "
            + (
                "PSEUDOBOND_CONTACT_NATURAL_COORDINATES"
                if definition.fragment_mode == "PSEUDO_BONDS"
                else "SPECIAL_COORDINATES"
            ),
        )
    result = replace(
        definition,
        semantic_diagnostics=(*diagnostics, *contract_diagnostics),
    )
    if not materialize_soop:
        return result
    from .soop_definition import SOOPMaterializationError, materialize_soop_definition
    symmetry_operations = read_molecular_symmetry(Path(path)).operations
    try:
        result = materialize_soop_definition(
            result,
            pool.contract,
            validate_basis=bool(validate_sonic_basis),
            symmetry_operations=symmetry_operations,
        )
    except SOOPMaterializationError:
        if path_role != "MINIMUM":
            raise
        from .local_sycart_transition import build_initial_local_sycart_from_oracle

        geometry = read_enriched_xyz(Path(path))
        try:
            result, _zone = build_initial_local_sycart_from_oracle(
                result,
                pool.contract,
                np.asarray(geometry.coordinates_angstrom, dtype=float),
                tuple(atomic_number(atom) for atom in geometry.atoms),
            )
        except ValueError:
            raise
        result = materialize_soop_definition(
            result,
            pool.contract,
            validate_basis=bool(validate_sonic_basis),
            symmetry_operations=symmetry_operations,
        )
    return result


def _automatic_chart_failure(exc: BaseException) -> bool:
    """Return whether a chart failure is recoverable by changing representation."""
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "condition gate",
            "rank deficient",
            "ill-conditioned",
            "ill conditioned",
            "soop",
            "conditioning",
        )
    )


def _automatic_delocalized_chart(
    path: Path,
    pool: FrozenOracleCandidatePool,
    *,
    atlas: object,
    rank_tolerance: float,
    scientific_path: str,
    trigger: str,
) -> GICDefinition:
    """Build the automatic last-resort chart from the frozen primitive row space.

    The primitive contract is never changed.  A thin SVD only rotates its
    already validated row space into an orthonormal, non-redundant chart;
    this is used only after the local/SALC chart fails certification.  The
    resulting coordinates remain exact linear combinations of ORACLE
    primitives and are therefore consumable by LINK without an external
    chart-selection request.
    """
    from .definition import build_gic_definition_from_xyzin

    base = build_gic_definition_from_xyzin(
        path,
        # Keep the first automatic fallback local: retain the requested
        # symmetry blocks while replacing only the ill-conditioned scalar
        # chart by small primitive combinations below.
        symmetrize=True,
        fragment_context=("minimum" if scientific_path == "MINIMUM" else "transition_state"),
        rank_tolerance=rank_tolerance,
        coordinate_atlas_contract=atlas,
        _oracle_pool=pool,
        validate_sonic_basis=False,
    )
    rows = []
    for row in pool.rows:
        if isinstance(row, SparseBRow):
            dense = np.zeros(len(pool.coordinates_angstrom) * 3, dtype=float)
            for column, value in row.entries:
                dense[int(column)] = float(value)
            rows.append(dense)
        else:
            rows.append(np.asarray(row, dtype=float))
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise GICForgeContractError("automatic chart fallback received an empty primitive row space")
    # Delocalize independently inside the immutable intra-/inter-fragment
    # domains.  The two domains are never mixed.  Each domain is processed in
    # bounded primitive blocks, so the fallback is only partially
    # delocalized rather than one global normal-mode rotation.
    from matrix_oracle import assign_primitive_classes

    assignments = assign_primitive_classes(pool.candidates)
    domain_indices: dict[str, list[int]] = {}
    for index, assignment in enumerate(assignments):
        domain_indices.setdefault(str(assignment.domain_kind), []).append(index)
    maximum_support = min(1536, matrix.shape[0])
    candidate_transforms: list[np.ndarray] = []
    for indices in domain_indices.values():
        for start in range(0, len(indices), maximum_support):
            block = indices[start : start + maximum_support]
            u, singular, _vt = np.linalg.svd(
                matrix[np.asarray(block, dtype=int)], full_matrices=False
            )
            cutoff = float(rank_tolerance) * max(float(singular[0]), 1.0)
            keep = singular > cutoff
            if not np.any(keep):
                continue
            transform = np.zeros((int(np.count_nonzero(keep)), matrix.shape[0]))
            transform[:, np.asarray(block, dtype=int)] = u[:, keep].T
            candidate_transforms.append(transform)
    if not candidate_transforms:
        raise GICForgeContractError("automatic chart fallback received no domain row space")
    candidate_transform = np.vstack(candidate_transforms)
    candidate_rows = candidate_transform @ matrix
    _q, _r, pivot = scipy_qr(candidate_rows.T, pivoting=True, mode="economic")
    selected_rows = np.asarray(pivot[: int(base.target_rank)], dtype=int)
    if selected_rows.size != int(base.target_rank):
        raise GICForgeContractError("automatic delocalized chart cannot realize the vibrational rank")
    coefficients = candidate_transform[selected_rows]
    rank = int(base.target_rank)
    if rank < int(base.target_rank):
        raise GICForgeContractError(
            "automatic delocalized chart cannot realize the vibrational rank"
        )
    primitive_ids = tuple(item.identifier for item in pool.candidates)
    gics = []
    for index, coeff_row in enumerate(coefficients, start=1):
        terms = tuple(
            (primitive_ids[column], float(value))
            for column, value in enumerate(coeff_row)
            if abs(float(value)) > 1.0e-12
        )
        if not terms:
            raise GICForgeContractError("automatic delocalized chart produced an empty row")
        gics.append(
            FrozenGIC(
                identifier=f"DL{index:04d}",
                name=f"DL{index:04d}",
                family="DELOCALIZED",
                irrep="A",
                primitive_id=terms[0][0],
                gaussian_expression="NONE",
                coefficients=terms,
            )
        )
    diagnostics = base.reduction_diagnostics
    if diagnostics is not None:
        diagnostics = replace(
            diagnostics,
            rank_method="AUTOMATIC_PRIMITIVE_ROWSPACE_ORTHONORMALIZATION",
            reduction_policy="AUTOMATIC_CHART_FALLBACK",
            conditioning_decisions=(
                *diagnostics.conditioning_decisions,
                "AUTOMATIC_CHART_CHANGE PARTIAL_DELOCALIZATION",
                f"AUTOMATIC_CHART_MAX_PRIMITIVE_SUPPORT={maximum_support}",
                f"TRIGGER={trigger[:240]}",
            ),
        )
    return replace(
        base,
        point_group="C1",
        symmetrize=False,
        rank=rank,
        primitives=pool.candidates,
        gics=tuple(gics),
        reduction_diagnostics=diagnostics,
        semantic_diagnostics=(
            *base.semantic_diagnostics,
            "AUTOMATIC_CHART_CHANGE DELOCALIZED_PRIMITIVE_ROWSPACE",
            "AUTOMATIC_CHART_SOURCE ORACLE_PRIMITIVES",
        ),
    )


def write_gic_definition_from_oracle_contract(
    path: Path,
    *,
    symmetrize: bool = False,
    sycart: bool = False,
    orientation: str = "SONIC",
    scientific_path: str = "MINIMUM",
    family_quotas: Mapping[str, int] | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
    out_of_plane_model: str = "soop",
) -> GICDefinition:
    """Serialize a strict ONIC chart without invoking chemical perception."""

    from matrix_core import read_sectioned_lines, replace_section, section_content

    target = Path(path)
    if not section_content(read_sectioned_lines(target), "FRAGMENTS"):
        from matrix_fragments import write_fragment_build_section

        write_fragment_build_section(target)

    from .definition import gic_definition_section_lines

    definition = build_gic_definition_from_oracle_contract(
        target,
        symmetrize=symmetrize,
        orientation=orientation,
        scientific_path=scientific_path,
        family_quotas=family_quotas,
        rank_tolerance=rank_tolerance,
        out_of_plane_model=out_of_plane_model,
        validate_sonic_basis=not sycart,
    )
    replace_section(target, "GIC", gic_definition_section_lines(definition))
    if sycart:
        from .definition import (
            build_sycart_definition_from_xyzin,
            sycart_definition_section_lines,
        )

        sycart_definition = build_sycart_definition_from_xyzin(target)
        replace_section(
            target,
            "SYCART",
            sycart_definition_section_lines(sycart_definition),
        )
    return definition


def _validate_frozen_fragment_handoff(
    path: Path,
    contract: OracleSonicContract,
) -> None:
    """Require the built fragment transport to match ORACLE's frozen partition."""

    from matrix_fragments import read_fragment_records

    observed = tuple(
        (record.identifier, tuple(int(atom) for atom in record.atoms))
        for record in read_fragment_records(Path(path))
    )
    expected = tuple(
        (record.fragment_id, tuple(int(atom) for atom in record.atoms))
        for record in contract.primary_topology.fragments
    )
    if observed != expected:
        raise GICForgeContractError(
            "#FRAGMENTS must be built and must match the frozen ORACLE partition"
        )


def _normalized_scientific_path(value: str) -> str:
    normalized = str(value).strip().upper().replace("-", "_")
    if normalized not in {"MINIMUM", "TRANSITION_STATE"}:
        raise ValueError("scientific_path must be 'MINIMUM' or 'TRANSITION_STATE'")
    return normalized


def _chart_role(orientation: str) -> str:
    return {
        "TONIC": "GENERAL",
        "CONIC": "EXPLORATION",
        "SONIC": "EXPLOITATION",
    }[str(orientation).strip().upper()]


def _local_perception_diagnostics(contract: OracleSonicContract) -> tuple[str, ...]:
    """Expose frozen ORACLE decisions without re-perceiving local chemistry."""

    domains_by_id = {domain.domain_id: domain for domain in contract.local_perception_domains}
    records = []
    for domain_id in sorted(domains_by_id):
        domain = domains_by_id[domain_id]
        classes = ",".join(
            "+".join(str(atom) for atom in item.members)
            for item in domain.equivalence_classes
        )
        template = domain.template_decision
        records.append(
            "LOCAL_PERCEPTION "
            f"ID={domain.domain_id} KIND={domain.kind} GROUP={domain.proposed_group} "
            f"CONFIDENCE={domain.confidence} CLASSES={classes or 'NONE'} "
            f"TEMPLATE={(template.selected_template if template else None) or 'NONE'} "
            f"STATUS={(template.status if template else 'NOT_APPLICABLE')} "
            f"PROVIDER={domain.provider}@{domain.provider_version}"
        )
    return tuple(records)


def _validate_frozen_geometry(
    path: Path,
    contract: OracleSonicContract,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
) -> None:
    topology = contract.primary_topology
    if coords.shape != (topology.natoms, 3):
        raise GICForgeContractError("ORACLE contract atom count does not match the geometry")
    try:
        numbers = tuple(atomic_number(symbol) for symbol in atom_symbols)
    except (KeyError, ValueError) as exc:
        raise GICForgeContractError("geometry contains an unsupported atomic symbol") from exc
    if numbers != topology.atomic_numbers:
        raise GICForgeContractError("ORACLE contract atomic numbers do not match the geometry")
    try:
        identity = read_geometry_identity_certificate(Path(path))
        validate_geometry_identity_certificate(
            identity,
            canonical_atoms=atom_symbols,
            canonical_coordinates_angstrom=coords,
        )
    except GeometryIdentityError as exc:
        raise GICForgeContractError(
            f"invalid frozen ORACLE Cartesian provenance: {exc}"
        ) from exc
    if identity.canonical_geometry_sha256 != contract.reference_geometry_sha256:
        raise GICForgeContractError(
            "ORACLE SONIC reference geometry contradicts Cartesian provenance"
        )
    permitted_relation = identity.relation in {
        GEOMETRY_IDENTICAL,
        GEOMETRY_RIGID_FRAME_ONLY,
    } or (
        identity.relation == GEOMETRY_TRUE_CHANGE
        and identity.geometry_change_authorization
        == GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION
    )
    if not permitted_relation:
        raise GICForgeContractError(
            "production SONIC requires a proper rigid frame or an explicitly "
            "ORACLE-authorized symmetry projection; "
            f"ORACLE classified {identity.relation} with authorization "
            f"{identity.geometry_change_authorization}"
        )
    try:
        snapshot = topology_snapshot_from_xyzin(Path(path))
        symmetry = read_molecular_symmetry(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(f"invalid frozen ORACLE topology/symmetry: {exc}") from exc
    snapshot_bonds = tuple(tuple(int(atom) for atom in pair) for pair in snapshot["bonds"])
    snapshot_rings = tuple(tuple(int(atom) for atom in row["atoms"]) for row in snapshot["rings"])
    if snapshot_bonds != topology.bonds or snapshot_rings != topology.rings:
        raise GICForgeContractError(
            "ORACLE SONIC PRIMARY_TOPOLOGY contradicts the frozen #TOPOLOGY section"
        )
    operation_permutations = tuple(operation.permutation for operation in symmetry.operations)
    expected_permutations = topology.symmetry_permutations
    if operation_permutations and set(operation_permutations) != set(expected_permutations):
        raise GICForgeContractError(
            "ORACLE SONIC symmetry permutations contradict the frozen #SYMMETRY section"
        )


def _validate_candidate_ownership(contract: OracleSonicContract) -> None:
    domains = {record.domain_id: record for record in contract.multicenter_domains}
    contacts = {record.contact_id: record for record in contract.auxiliary_contacts}
    sites = {record.site_id for record in contract.structural_sites}
    referenced: dict[str, str] = {}
    for domain in domains.values():
        for candidate_id in domain.primitive_candidate_ids:
            previous = referenced.setdefault(candidate_id, domain.domain_id)
            if previous != domain.domain_id:
                raise GICForgeContractError(
                    f"primitive candidate {candidate_id} is claimed by multiple domains"
                )
    for contact in contacts.values():
        for candidate_id in contact.primitive_candidate_ids:
            previous = referenced.setdefault(candidate_id, contact.contact_id)
            if previous != contact.contact_id:
                raise GICForgeContractError(
                    f"primitive candidate {candidate_id} is claimed by multiple domains"
                )
    for candidate in contract.primitive_candidates:
        owner = candidate.owner_id
        if owner == "PRIMARY_TOPOLOGY":
            if candidate.domain_id != "PRIMARY_TOPOLOGY" and not (
                candidate.domain_id.startswith("PRIMARY_TOPOLOGY::")
            ):
                raise GICForgeContractError(
                    f"primitive candidate {candidate.candidate_id} has contradictory topology ownership"
                )
            continue
        if owner in domains:
            if referenced.get(candidate.candidate_id) != owner:
                raise GICForgeContractError(
                    f"multicenter primitive {candidate.candidate_id} is not claimed by its domain"
                )
            continue
        if owner in contacts:
            if referenced.get(candidate.candidate_id) != owner:
                raise GICForgeContractError(
                    f"contact primitive {candidate.candidate_id} is not claimed by its contact"
                )
            continue
        if owner in sites and candidate.domain_id == "STRUCTURAL_SITES":
            continue
        if candidate.domain_id == "FRAGMENT_POSE" and owner.startswith("FRAGMENT_PAIR:"):
            continue
        raise GICForgeContractError(
            f"primitive candidate {candidate.candidate_id} has an unknown ORACLE owner"
        )


def _primitive_from_candidate(candidate: PrimitiveCandidate) -> GICPrimitive:
    frame_atoms, ref_frame_atoms, refs = _split_frame_refs(candidate.refs)
    frame_atoms = candidate.frame_atoms or frame_atoms
    ref_frame_atoms = candidate.ref_frame_atoms or ref_frame_atoms
    return GICPrimitive(
        identifier=candidate.candidate_id,
        name=candidate.candidate_id,
        family=candidate.family,
        function=candidate.function,
        atoms=candidate.atoms,
        mode=candidate.mode,
        ref_atoms=candidate.ref_atoms,
        refs=refs,
        frame_atoms=frame_atoms,
        ref_frame_atoms=ref_frame_atoms,
        provenance=candidate.provenance,
        semantic_id=candidate.owner_id,
        semantic_type=candidate.domain_id,
    )


def _split_frame_refs(
    refs: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    frame: tuple[int, ...] = ()
    ref_frame: tuple[int, ...] = ()
    retained: list[str] = []
    for item in refs:
        if item.startswith("FRAME_ATOMS="):
            frame = _parse_atom_csv(item.split("=", 1)[1])
        elif item.startswith("REF_FRAME_ATOMS="):
            ref_frame = _parse_atom_csv(item.split("=", 1)[1])
        else:
            retained.append(item)
    return frame, ref_frame, tuple(retained)


def _parse_atom_csv(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in text.split(",") if value)
    except ValueError as exc:
        raise GICForgeContractError("invalid ORACLE primitive frame metadata") from exc
    if len(set(values)) != len(values):
        raise GICForgeContractError("ORACLE primitive frame repeats atom indices")
    return values


def _validated_supplied_row(
    candidate: PrimitiveCandidate,
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray | SparseBRow:
    if candidate.wilson_column_count and candidate.wilson_column_count != coords.size:
        raise GICForgeContractError(
            f"ORACLE primitive {candidate.candidate_id} Wilson column count contradicts geometry"
        )
    if candidate.analytic_wilson_indices:
        supplied_sparse = SparseBRow(
            size=int(candidate.wilson_column_count or coords.size),
            entries=tuple(
                zip(
                    candidate.analytic_wilson_indices,
                    candidate.analytic_wilson_values,
                    strict=True,
                )
            ),
        )
        if supplied_sparse.norm() <= 1.0e-12:
            raise GICForgeContractError(
                f"ORACLE primitive {candidate.candidate_id} has a null Wilson row"
            )
        try:
            evaluated_sparse = _sparse_analytic_b_row(
                primitive,
                coords,
                reference_coords=coords if primitive.function == "FROT" else None,
            )
        except (ArithmeticError, FloatingPointError, ValueError) as exc:
            raise GICForgeContractError(
                f"ORACLE primitive {candidate.candidate_id} has no finite SMITH realization"
            ) from exc
        supplied_entries = dict(supplied_sparse.entries)
        evaluated_entries = dict(evaluated_sparse.entries)
        residual = float(
            np.sqrt(
                sum(
                    (supplied_entries.get(index, 0.0) - evaluated_entries.get(index, 0.0)) ** 2
                    for index in supplied_entries.keys() | evaluated_entries.keys()
                )
            )
        )
        scale = max(1.0, supplied_sparse.norm(), evaluated_sparse.norm())
        if residual > _ROW_VALIDATION_ATOL * scale:
            raise GICForgeContractError(
                f"ORACLE primitive {candidate.candidate_id} Wilson row contradicts its "
                f"finite function (residual {residual:.6g})"
            )
        return supplied_sparse
    supplied = primitive_candidate_wilson_row(candidate, len(coords))
    # Redundant finite measurements may be weak without being identically
    # null. Rank/conditioning thresholds belong to selection, not validation
    # of whether ORACLE's stored derivative represents its finite function.
    if float(np.linalg.norm(supplied)) <= 1.0e-12:
        raise GICForgeContractError(
            f"ORACLE primitive {candidate.candidate_id} has a null Wilson row"
        )
    try:
        evaluated = _analytic_b_row(
            primitive,
            coords,
            reference_coords=coords if primitive.function == "FROT" else None,
        )
    except (ArithmeticError, FloatingPointError, ValueError) as exc:
        raise GICForgeContractError(
            f"ORACLE primitive {candidate.candidate_id} has no finite SMITH realization"
        ) from exc
    scale = max(1.0, float(np.linalg.norm(supplied)), float(np.linalg.norm(evaluated)))
    residual = float(np.linalg.norm(supplied - evaluated))
    if residual > _ROW_VALIDATION_ATOL * scale:
        raise GICForgeContractError(
            f"ORACLE primitive {candidate.candidate_id} Wilson row contradicts its finite function "
            f"(residual {residual:.6g})"
        )
    return supplied


def _normalized_family_quotas(quotas: Mapping[str, int] | None) -> dict[str, int]:
    if quotas is None:
        return {}
    result = {str(family): int(count) for family, count in quotas.items()}
    if any(not family or count < 0 for family, count in result.items()):
        raise GICForgeContractError("family quotas must be named non-negative integers")
    return result


def _select_indices(
    contract: OracleSonicContract,
    rows: tuple[np.ndarray, ...],
    *,
    target_rank: int,
    family_quotas: dict[str, int],
    rank_tolerance: float,
    use_family_priorities: bool = True,
) -> tuple[int, ...]:
    if target_rank == 0:
        return ()
    norms = np.asarray([np.linalg.norm(row) for row in rows], dtype=float)
    normalized = np.vstack(
        [row / norm if norm > rank_tolerance else row for row, norm in zip(rows, norms, strict=True)]
    )
    if family_quotas:
        if sum(family_quotas.values()) != target_rank:
            raise GICForgeContractError("exact family quotas must sum to the target Wilson rank")
        records = [
            (
                candidate.family,
                "",
                (),
                np.zeros(0),
                normalized[index],
                index,
                index,
            )
            for index, candidate in enumerate(contract.primitive_candidates)
        ]
        selected = _partition_quota_rank_revealing_selection(
            records,
            quotas={family: count for family, count in family_quotas.items() if count},
        )
        if selected is None:
            raise GICForgeContractError("ORACLE primitive pool cannot satisfy exact family quotas")
        return tuple(int(record[6]) for record in selected)

    family_order = {family: index for index, family in enumerate(PRIMITIVE_FAMILY_ORDER)}
    priorities = (
        tuple(
            0
            if candidate.family in SPECIAL_PRIMITIVE_FAMILIES
            else 1 + family_order.get(candidate.family, len(family_order))
            for candidate in contract.primitive_candidates
        )
        if use_family_priorities
        else None
    )
    selection = select_rank_revealing_rows(
        normalized,
        target_rank=target_rank,
        tolerance=rank_tolerance,
        priorities=priorities,
        tie_tolerance=1.0e-12,
    )
    return _complete_linear_bend_selection_units(
        tuple(int(index) for index in selection.indices),
        contract=contract,
        normalized_rows=normalized,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
    )


def _complete_linear_bend_selection_units(
    selected: tuple[int, ...],
    *,
    contract: OracleSonicContract,
    normalized_rows: np.ndarray,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[int, ...]:
    """Repair QR selections so every linear bend contains both components.

    A row-wise rank-revealing selection can choose only L(-1) or L(-2).  The
    two directions define one indivisible coordinate and must be exchanged as
    a pair.  Complete each partial pair and remove the best scalar row while
    preserving the exact numerical rank.  This operates only on the small
    recovery/audit path; production construction remains sparse.
    """

    candidates = contract.primitive_candidates

    def key(index: int) -> tuple[object, ...]:
        item = candidates[index]
        return (
            item.family,
            item.atoms,
            item.ref_atoms,
            item.refs,
            item.owner_id,
            item.domain_id,
        )

    groups: dict[tuple[object, ...], dict[int, list[int]]] = {}
    linear_indices: set[int] = set()
    for index, item in enumerate(candidates):
        if item.function.strip().upper() != "L":
            continue
        linear_indices.add(index)
        groups.setdefault(key(index), {}).setdefault(int(item.mode), []).append(index)

    units: list[tuple[int, ...]] = [
        (index,) for index in range(len(candidates)) if index not in linear_indices
    ]
    for modes in groups.values():
        minus_one = tuple(modes.get(-1, ()))
        minus_two = tuple(modes.get(-2, ()))
        if len(minus_one) != len(minus_two):
            raise GICForgeContractError(
                "ORACLE linear-bend candidate pool contains an incomplete component pair"
            )
        units.extend(
            (left, right)
            for left, right in zip(sorted(minus_one), sorted(minus_two), strict=True)
        )

    # Pivot whole units in Wilson-row space.  The residual matrix is updated
    # by BLAS rank-one operations, so this remains a small audit/recovery cost
    # rather than a combinatorial search over possible pairs.
    units.sort()
    work = np.asarray(normalized_rows, dtype=float).copy()
    current: list[int] = []
    remaining_units = list(units)
    while len(current) < target_rank:
        capacity = target_rank - len(current)
        best: tuple[float, tuple[int, ...], tuple[np.ndarray, ...]] | None = None
        for unit in remaining_units:
            if len(unit) > capacity:
                continue
            residuals: list[np.ndarray] = []
            score = float("inf")
            for index in unit:
                residual = work[index].copy()
                for vector in residuals:
                    residual -= float(np.dot(residual, vector)) * vector
                norm = float(np.linalg.norm(residual))
                if not np.isfinite(norm) or norm <= rank_tolerance:
                    residuals = []
                    break
                score = min(score, norm)
                residuals.append(residual / norm)
            if not residuals:
                continue
            record = (score, tuple(-index for index in unit), tuple(residuals))
            if best is None or record[:2] > best[:2]:
                best = record
                best_unit = unit
        if best is None:
            break
        residuals = best[2]
        current.extend(best_unit)
        remaining_units.remove(best_unit)
        for vector in residuals:
            for _pass in range(2):
                overlaps = work @ vector
                work -= overlaps[:, None] * vector[None, :]
    if len(current) == target_rank:
        return tuple(current)
    raise GICForgeContractError(
        "ORACLE component-aware primitive pool has insufficient exact Wilson rank"
    )


__all__ = [
    "FrozenOracleCandidateSelection",
    "FrozenOracleCandidatePool",
    "SMITH_ORACLE_CONSUMER_SCHEMA",
    "build_gic_definition_from_oracle_contract",
    "load_validated_oracle_candidate_pool",
    "select_frozen_oracle_candidates",
    "write_gic_definition_from_oracle_contract",
]
