from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from matrix_chem.coordinate_atlas_contract import (
    ATLAS_TASK_EXPLORATION,
    ATLAS_TASK_MINIMUM,
    ATLAS_TASK_TRANSITION_STATE,
    BODY_NONLINEAR,
    COORDINATE_PHYSICAL_CONTACT_DISTANCE,
    COORDINATE_REACTION_DISTANCE,
    COORDINATE_REACTION_DISTANCE_ONLY,
    COORDINATE_TS_SUPPORT,
    GRAPH_ROLE_REACTIVE_SUPPORT,
    PSEUDOBOND_REQUIRED,
    OracleCoordinateAtlasContract,
    read_oracle_coordinate_atlas_contract,
    validate_oracle_coordinate_atlas_contract,
)
from matrix_chem.topology.elements import atomic_number
from matrix_chem import (
    MATRIX_XYZ_PRIMITIVES_SCHEMA,
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    MATRIX_XYZ_VALIDATION_SCHEMA,
    preprocess_to_enriched_xyz,
    read_enriched_xyz,
    read_molecular_symmetry,
    read_oracle_sonic_contract,
    read_oracle_transition_state_geometry_contract,
    read_primitive_contract,
    TS_CHART_REACTIVE_DISTANCE,
    TS_CHART_REACTIVE_PSEUDOBOND,
    transition_state_descriptor,
    validate_primitive_contract,
    write_validation_section,
)
from matrix_chem.primitive_coordinates import (
    LEGACY_MATRIX_XYZ_PRIMITIVES_SCHEMA,
    LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA,
)
from matrix_core import read_sectioned_lines, replace_section, section_content

from .contracts import (
    ORACLE_XYZ_GIC_SCHEMA as ORACLE_XYZ_GIC_SCHEMA,
    ORACLE_XYZ_SYCART_SCHEMA,
    GICForgeContractError,
    GICForgeRankDeficiencyError,
    validate_gicforge_prerequisites,
)
from .basis_validation import validate_frozen_sonic_basis
from .component_invariants import validate_indivisible_gic_components
from .definition_io import (
    _apply_symmetry_group_limit,
    _key_values as _key_values,
    _parse_atom_list as _parse_atom_list,
    _point_group,
    _subsection as _subsection,
    _symmetry_operations,
)
from .periodic_estimates import (
    parse_periodic_coordinate_estimate as parse_periodic_coordinate_estimate,
    periodic_coordinate_estimate_line,
)
from .semantic import (
    semantic_contract_from_sectioned_lines,
)
from .policy import (
    normalize_fragment_context as _fragment_context,
    AROMATIC_LOCAL_MODEL_DIAGNOSTIC,
    FRAGMENT_MODE_NONE,
    FRAGMENT_MODE_PSEUDO_BONDS,
    FRAGMENT_MODE_SPECIAL_COORDINATES,
    GIC_BACKEND,
    PHASE_ZERO_TOLERANCE,
    FRAGMENT_CONTEXT_TRANSITION_STATE,
    FRAGMENT_MODES as FRAGMENT_MODES,
    LOCAL_SYMMETRIZATION_METHOD as LOCAL_SYMMETRIZATION_METHOD,
    POINT_GROUP_PROJECTOR_METHOD as POINT_GROUP_PROJECTOR_METHOD,
    RANK_METHOD,
    RANK_TOLERANCE,
    REDUCTION_POLICY,
    SALC_PATH_OVERLAP_WARNING_THRESHOLD as SALC_PATH_OVERLAP_WARNING_THRESHOLD,
    SYCART_BACKEND,
    SYMMETRIZATION_POLICY as SYMMETRIZATION_POLICY,
    SYMMETRY_OPERATION_TOLERANCE_ANGSTROM as SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
    XH_STRETCH_CLASSES,
    XH_STRETCH_POLICIES,
    XH_STRETCH_POLICY_LOCAL_ALL,
    XH_STRETCH_POLICY_LOCAL_SELECTED,
    XH_STRETCH_POLICY_SYMMETRIZE,
)
from .symmetry_labels import (
    is_total_symmetric_irrep,
    total_symmetric_irrep,
)
from .models import (
    FallbackEvent,
    FrozenGIC,
    GICBMatrix,
    GICDefinition,
    GICPointGroupOperation,
    GICPrimitive,
    GICReductionDiagnostics,
    GICSymmetrizationDiagnostics as GICSymmetrizationDiagnostics,
    GICSymmetrizedGroup as GICSymmetrizedGroup,
    SYCartDefinition,
)
from .minimum_domain_conditioning import condition_minimum_pseudobond_chart
from .ring_salc_materialization import materialize_ring_out_of_plane_salcs
from .topology_io import (
    _topology_bonds,
    _topology_rings,
    topology_aromatic_atoms_from_lines,
    topology_bond_order_components_from_lines,
    topology_bond_orders_from_lines,
)
from .symmetrization import (
    _apply_local_symmetrization,
    _empty_symmetry_diagnostics,
    _protected_gic_count,
    _reduction_diagnostics_lines,
    _renumber_frozen_gic,
    _skipped_dependent_count,
    _skipped_singular_count,
    _symmetry_closed_projector_primitives,
    _symmetry_diagnostics_lines,
    validate_molecular_vibrational_irrep_counts,
    _select_ranked_primitives,
)
from .semantic import USER_PROVENANCE
from .cartesian_blocks import CARTESIAN_BLOCK_GAUGE, symmetry_adapted_cartesian_basis
from .atlas_realization import (
    apply_reactive_zone_exclusions,
    validate_atlas_chart_realization,
)
from .fallback_ledger import (
    build_fallback_ledger,
    fallback_ledger_section_lines,
    fallback_provenance_from_lines as fallback_provenance_from_lines,
    make_fallback_event,
    merge_fallback_events,
)
from .coordinate_registry import MODE_BEARING_PRIMITIVE_FUNCTIONS as MODE_BEARING_PRIMITIVE_FUNCTIONS
from .semantic import AUTO_PROVENANCE as AUTO_PROVENANCE
from .semantic import SEMANTIC_GRAMMAR_VERSION as SEMANTIC_GRAMMAR_VERSION


_GAUSSIAN_DEPENDENCY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_PHASE_ZERO_TOLERANCE = PHASE_ZERO_TOLERANCE
_FRAGMENT_CONTEXT_MINIMUM = "minimum"
_FRAGMENT_CONTEXT_OPTIMIZATION = "optimization"
_FRAGMENT_CONTEXT_EXPLORATION = "exploration"
_FRAGMENT_CONTEXT_TRANSITION_STATE = FRAGMENT_CONTEXT_TRANSITION_STATE


@dataclass(frozen=True)
class _ChartExecutionContext:
    coordinate_atlas_contract: OracleCoordinateAtlasContract
    topology_bonds: tuple[tuple[int, int], ...]
    rings: tuple[tuple[int, tuple[int, ...]], ...]
    coords: np.ndarray
    atom_symbols: tuple[str, ...]
    xh_stretch_policy: str
    local_xh_bonds: tuple[tuple[int, int], ...]
    local_xh_classes: tuple[str, ...]
    body_prescriptions: tuple[object, ...]
    fragment_contacts: tuple[tuple[int, int, str], ...]
    semantic_contract: object
    symmetry_operations: tuple[GICPointGroupOperation, ...]
    prescribed_ts_reaction_distances: tuple[tuple[int, int], ...]
    reactive_support_edges: tuple[tuple[int, int], ...]
    bond_orders: dict[tuple[int, int], float]
    aromatic_atoms: frozenset[int]
    ring_puckering_model: str
    target_rank: int
    rank_tolerance: float
    task_regime: str
    condition_ordinary: bool
    reactive_ts: bool
    pseudobond_out_of_plane_support: bool
    condition_transition_state_exact_chart: bool


def build_gic_definition_from_xyzin(
    path: Path,
    *,
    symmetrize: bool = False,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
    rank_tolerance: float = RANK_TOLERANCE,
    coordinate_atlas_contract: OracleCoordinateAtlasContract | None = None,
    _oracle_pool: object | None = None,
    validate_sonic_basis: bool = True,
) -> GICDefinition:
    """Build a frozen SONIC coordinate definition from saved xyzin state."""
    from .oracle_contract_consumer import (
        load_validated_oracle_candidate_pool,
        select_frozen_oracle_candidates,
    )

    # SMITH/SONIC is a topology-aware route.  ORACLE owns the certification;
    # fail closed before any candidate matching or chart construction.
    oracle_contract = read_oracle_sonic_contract(Path(path))
    if not oracle_contract.topology_certified:
        raise GICForgeContractError(
            "SMITH/SONIC requires a certified ORACLE CHEMICAL_PRIMITIVES topology"
        )
    from .primitive_handoff import bind_oracle_measurements

    oracle_pool = _oracle_pool or load_validated_oracle_candidate_pool(Path(path))
    context = _fragment_context(fragment_context)
    separate_exocyclic_torsions = _resolved_separate_exocyclic_torsions(
        Path(path),
        context=context,
        requested=separate_exocyclic_torsions,
    )
    coordinate_atlas_contract = _required_coordinate_atlas_contract(
        Path(path),
        context=context,
        supplied=coordinate_atlas_contract,
    )
    prescribed_fragment_mode = _atlas_fragment_mode(coordinate_atlas_contract)
    if fragment_mode is not None and _fragment_mode(fragment_mode) != prescribed_fragment_mode:
        raise GICForgeContractError(
            "fragment_mode cannot override the frozen ORACLE coordinate atlas"
        )
    fragment_mode = prescribed_fragment_mode
    construct_kwargs = {
        "symmetry_group": symmetry_group,
        "improper_dihedrals": improper_dihedrals,
        "fragment_mode": fragment_mode,
        "fragment_context": fragment_context,
        "xh_stretch_policy": xh_stretch_policy,
        "local_xh_bonds": local_xh_bonds,
        "local_xh_classes": local_xh_classes,
        "local_salc": local_salc,
        "local_salc_settings": local_salc_settings,
        "xy3_torsions": xy3_torsions,
        "xy2_torsions": xy2_torsions,
        "separate_exocyclic_torsions": separate_exocyclic_torsions,
        "ring_puckering_model": ring_puckering_model,
        "rank_tolerance": rank_tolerance,
        "retain_candidate_primitives": (
            symmetrize or bool(coordinate_atlas_contract.local_domains)
        ),
        "coordinate_atlas_contract": coordinate_atlas_contract,
        # Preserve the established local/SALC chart whenever its final frozen
        # Jacobian passes. Exact-rank primitive pivoting is a fail-closed
        # recovery for a final C1 TS chart that actually exceeds the gate.
        "condition_transition_state_exact_chart": False,
    }
    definition, atom_symbols, operations = construct_gic_definition_from_xyzin(
        path, **construct_kwargs
    )
    if not symmetrize:
        definition = condition_minimum_pseudobond_chart(
            definition,
            coordinate_atlas_contract,
            sonic_contract=read_oracle_sonic_contract(Path(path)),
            rank_tolerance=rank_tolerance,
        )
        from .cyclic_conditioning import condition_cyclic_minimum_chart

        definition = condition_cyclic_minimum_chart(
            definition,
            rank_tolerance=rank_tolerance,
        )
        frozen = _with_periodic_coordinate_estimates(
            definition,
            Path(path),
            native_definition=definition,
        )
        validate_indivisible_gic_components(frozen)
        try:
            validate_frozen_sonic_basis(frozen, rank_tolerance=rank_tolerance)
        except GICForgeContractError as exc:
            if not any(
                marker in str(exc)
                for marker in ("rank deficient", "condition gate")
            ):
                raise
            recovery_trigger = (
                "SEMANTIC_CHART_ILL_CONDITIONED"
                if "condition gate" in str(exc)
                else "SEMANTIC_CHART_RANK_DEFICIENT"
            )
            # The semantic reducer can occasionally choose one dependent row
            # although ORACLE's complete primitive pool has exact vibrational
            # rank (notably for high-coordinate metal complexes).  Recover by
            # running the canonical rank selector over that frozen pool and
            # materializing its primitive basis as a C1 SONIC chart.
            try:
                selection = select_frozen_oracle_candidates(
                    Path(path),
                    target_rank=frozen.target_rank,
                    rank_tolerance=rank_tolerance,
                    allow_closing_contacts=True,
                    excluded_functions=frozenset({"FROT", "FTRANS"}),
                    use_family_priorities=False,
                )
            except GICForgeContractError:
                selection = select_frozen_oracle_candidates(
                    Path(path),
                    target_rank=frozen.target_rank,
                    rank_tolerance=rank_tolerance,
                    allow_closing_contacts=True,
                    use_family_priorities=False,
                )
            irrep = total_symmetric_irrep("C1")
            primitives = tuple(selection.selected)
            gics = tuple(
                FrozenGIC(
                    identifier=f"Q{index:04d}",
                    name=f"{irrep}{primitive.name}",
                    family=primitive.family,
                    irrep=irrep,
                    primitive_id=primitive.identifier,
                    gaussian_expression=primitive.gaussian_expression(),
                    coefficients=((primitive.identifier, 1.0),),
                )
                for index, primitive in enumerate(primitives, 1)
            )
            frozen = replace(
                frozen,
                point_group="C1",
                symmetrize=False,
                rank=selection.rank,
                primitives=primitives,
                gics=gics,
                reduction_diagnostics=GICReductionDiagnostics(
                    rank_method="ORACLE_FROZEN_PRIMITIVE_EXACT_RANK",
                    reduction_policy="C1_PRIMITIVE_POOL_RANK_RECOVERY",
                    selected=tuple(item.identifier for item in primitives),
                    conditioning_decisions=(
                        recovery_trigger,
                        "RECOVERED_FROM_ORACLE_FROZEN_PRIMITIVE_POOL",
                    ),
                ),
            )
            validate_frozen_sonic_basis(frozen, rank_tolerance=rank_tolerance)
        validate_atlas_chart_realization(frozen, coordinate_atlas_contract)
        return _attach_chart_atlas(bind_oracle_measurements(frozen, oracle_pool))
    sonic_contract = read_oracle_sonic_contract(Path(path))

    def finalize_symmetrized_chart(
        native_definition: GICDefinition,
        native_atom_symbols: tuple[str, ...],
        native_operations: tuple[GICPointGroupOperation, ...],
    ) -> GICDefinition:
        # Candidate retention is required by the local-domain conditioner, but
        # the semantic first pass may then retain an exact *structural* basis
        # whose evaluated Wilson rows are numerically dependent.  Repair the
        # operative rows before any SALC projection while retaining the full
        # ORACLE pool for subsequent local substitutions.
        try:
            if native_definition.wilson_tangent_rank < native_definition.target_rank:
                raise GICForgeContractError("pre-symmetrization Wilson tangent is rank deficient")
        except GICForgeContractError as exc:
            if "rank deficient" not in str(exc):
                raise
            try:
                selection = select_frozen_oracle_candidates(
                    Path(path),
                    target_rank=native_definition.target_rank,
                    rank_tolerance=rank_tolerance,
                    allow_closing_contacts=False,
                    excluded_functions=frozenset({"CONTACT_CENTER_ANGLE"}),
                    use_family_priorities=False,
                )
            except GICForgeContractError:
                selection = select_frozen_oracle_candidates(
                    Path(path),
                    target_rank=native_definition.target_rank,
                    rank_tolerance=rank_tolerance,
                    allow_closing_contacts=True,
                    excluded_functions=frozenset({"CONTACT_CENTER_ANGLE"}),
                    use_family_priorities=False,
                )
            selected = tuple(selection.selected)
            native_definition = replace(
                native_definition,
                rank=selection.rank,
                # The recovered ORACLE rows are already the selected robust
                # chart.  Do not merge them with the rejected local pool:
                # catalog identifiers are stable only within one provider
                # and a collision can otherwise bind a GIC to another finite
                # function.  A later refresh obtains a new complete pool from
                # ORACLE rather than mutating this frozen chart.
                primitives=selected,
                gics=tuple(
                    FrozenGIC(
                        identifier=f"GIC{index:03d}",
                        name=primitive.name,
                        family=primitive.family,
                        irrep=(
                            total_symmetric_irrep("C1")
                            if native_definition.point_group.upper() == "C1"
                            else "UNASSIGNED"
                        ),
                        primitive_id=primitive.identifier,
                        gaussian_expression=primitive.gaussian_expression(),
                        coefficients=((primitive.identifier, 1.0),),
                    )
                    for index, primitive in enumerate(selected, 1)
                ),
                reduction_diagnostics=GICReductionDiagnostics(
                    rank_method="ORACLE_FROZEN_PRIMITIVE_EXACT_RANK",
                    reduction_policy="PRE_SYMMETRIZATION_NUMERICAL_RANK_RECOVERY",
                    selected=tuple(item.identifier for item in selected),
                    conditioning_decisions=(
                        "SEMANTIC_CHART_NUMERICALLY_RANK_DEFICIENT",
                        "RECOVERED_BEFORE_SYMMETRIZATION_FROM_ORACLE_POOL",
                    ),
                ),
            )
            validate_frozen_sonic_basis(
                native_definition,
                rank_tolerance=rank_tolerance,
            )
        candidate = _condition_transition_state_salc_alternatives(
            native_definition,
            context=context,
            atom_symbols=native_atom_symbols,
            symmetry_operations=native_operations,
            coordinate_atlas_contract=coordinate_atlas_contract,
            separate_exocyclic_torsions=separate_exocyclic_torsions,
        )
        candidate = condition_minimum_pseudobond_chart(
            candidate,
            coordinate_atlas_contract,
            sonic_contract=sonic_contract,
            rank_tolerance=rank_tolerance,
        )
        from .cyclic_conditioning import condition_cyclic_minimum_chart

        candidate = condition_cyclic_minimum_chart(
            candidate,
            rank_tolerance=rank_tolerance,
        )
        candidate = _with_periodic_coordinate_estimates(
            candidate,
            Path(path),
            native_definition=native_definition,
        )
        if candidate.symmetry_diagnostics is not None:
            candidate = replace(
                candidate,
                symmetry_diagnostics=replace(
                    candidate.symmetry_diagnostics,
                    total_symmetric_gics=tuple(
                        gic.name
                        for gic in candidate.gics
                        if is_total_symmetric_irrep(candidate.point_group, gic.irrep)
                    ),
                ),
            )
        return candidate

    frozen = finalize_symmetrized_chart(definition, atom_symbols, operations)
    validate_indivisible_gic_components(frozen)
    try:
        if validate_sonic_basis:
            validate_frozen_sonic_basis(frozen, rank_tolerance=rank_tolerance)
    except GICForgeContractError as local_error:
        if context != _FRAGMENT_CONTEXT_TRANSITION_STATE:
            raise
        conditioned_definition, conditioned_symbols, conditioned_operations = (
            construct_gic_definition_from_xyzin(
                path,
                **{
                    **construct_kwargs,
                    "condition_transition_state_exact_chart": True,
                },
            )
        )
        conditioned = finalize_symmetrized_chart(
            conditioned_definition,
            conditioned_symbols,
            conditioned_operations,
        )
        try:
            validate_indivisible_gic_components(conditioned)
            if validate_sonic_basis:
                validate_frozen_sonic_basis(conditioned, rank_tolerance=rank_tolerance)
        except GICForgeContractError:
            raise local_error
        frozen = replace(
            conditioned,
            semantic_diagnostics=(
                *conditioned.semantic_diagnostics,
                "TS_EXACT_CHART_CONDITIONING FINAL_SALC_GATE_REBUILD=CONDITIONED",
            ),
        )
    validate_atlas_chart_realization(frozen, coordinate_atlas_contract)
    return _attach_chart_atlas(
        bind_oracle_measurements(
            frozen,
            oracle_pool,
            allow_missing=not validate_sonic_basis,
        )
    )


def _required_transition_state_contract(path: Path):
    """Read ORACLE's complete TS prescription and verify its frozen topology."""

    try:
        contract = read_oracle_transition_state_geometry_contract(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(
            "an explicit transition_state build requires a valid ORACLE single-geometry TS contract"
        ) from exc
    if contract.natoms != len(read_enriched_xyz(Path(path)).atoms):
        raise GICForgeContractError("ORACLE TS contract atom count is stale")
    try:
        sonic_contract = read_oracle_sonic_contract(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(
            "the ORACLE TS contract requires its frozen SONIC topology"
        ) from exc
    if contract.topology_hash != sonic_contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE TS contract topology fingerprint is stale")
    return contract


def _resolved_separate_exocyclic_torsions(
    path: Path,
    *,
    context: str,
    requested: bool,
) -> bool:
    """Consume ORACLE's torsion prescription for TS; retain the non-TS option."""

    if context != _FRAGMENT_CONTEXT_TRANSITION_STATE:
        return bool(requested)
    contract = _required_transition_state_contract(path)
    prescribed = transition_state_descriptor(contract, "SEPARATE_EXOCYCLIC_TORSIONS") == "TRUE"
    if requested and not prescribed:
        raise GICForgeContractError(
            "separate_exocyclic_torsions cannot override the ORACLE transition-state contract"
        )
    return prescribed


def _required_coordinate_atlas_contract(
    path: Path,
    *,
    context: str,
    supplied: OracleCoordinateAtlasContract | None = None,
) -> OracleCoordinateAtlasContract:
    """Load the sole ORACLE scientific prescription and validate its scope."""

    try:
        contract = supplied or read_oracle_coordinate_atlas_contract(Path(path))
        validate_oracle_coordinate_atlas_contract(contract)
    except ValueError as exc:
        raise GICForgeContractError(
            "SMITH requires a valid frozen ORACLE coordinate atlas; "
            "scientific chart choices cannot be reconstructed downstream"
        ) from exc
    expected_task = (
        ATLAS_TASK_TRANSITION_STATE
        if context == _FRAGMENT_CONTEXT_TRANSITION_STATE
        else ATLAS_TASK_EXPLORATION
        if context == _FRAGMENT_CONTEXT_EXPLORATION
        else ATLAS_TASK_MINIMUM
    )
    if contract.task_regime != expected_task:
        raise GICForgeContractError(
            "ORACLE coordinate atlas task regime contradicts the requested scientific path"
        )
    try:
        sonic_contract = read_oracle_sonic_contract(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(
            "the coordinate atlas requires its frozen ORACLE SONIC contract"
        ) from exc
    if contract.topology_hash != sonic_contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE coordinate atlas topology fingerprint is stale")
    return contract


def _atlas_fragment_mode(contract: OracleCoordinateAtlasContract) -> str:
    """Translate ORACLE's frozen graph effects to an execution mode only."""

    validate_oracle_coordinate_atlas_contract(contract)
    if any(item.pseudobond_policy == PSEUDOBOND_REQUIRED for item in contract.interactions):
        return FRAGMENT_MODE_PSEUDO_BONDS
    return FRAGMENT_MODE_SPECIAL_COORDINATES if len(contract.bodies) > 1 else FRAGMENT_MODE_NONE


def _atlas_needs_fragment_geometry(
    contract: OracleCoordinateAtlasContract,
) -> bool:
    """Return whether mathematical chart construction needs fragment atoms."""

    return bool(contract.bodies) or (
        any(item.pseudobond_policy == PSEUDOBOND_REQUIRED for item in contract.interactions)
    )


def _atlas_declares_pseudobond_out_of_plane_support(
    contract: OracleCoordinateAtlasContract,
) -> bool:
    """Return whether the frozen atlas admits U in local contact completion."""

    return any(
        item.block_id == "TS_REACTIVE_COMPLETION" and "OUT_OF_PLANE" in item.families
        for item in contract.family_compatibility
    )


def _atlas_atom_contacts(
    contract: OracleCoordinateAtlasContract,
    *,
    pseudobond_policy: str | None = None,
    coordinate_role: str | None = None,
) -> tuple[tuple[int, int, str], ...]:
    """Return exact atom contacts; no geometry or chemistry is interpreted."""

    records = []
    for item in contract.interactions:
        if pseudobond_policy is not None and item.pseudobond_policy != pseudobond_policy:
            continue
        if coordinate_role is not None and item.coordinate_role != coordinate_role:
            continue
        if item.endpoint_a[0] != "ATOM" or item.endpoint_b[0] != "ATOM":
            continue
        left, right = sorted((int(item.endpoint_a[1]), int(item.endpoint_b[1])))
        records.append(
            (
                left,
                right,
                item.semantic_kind,
            )
        )
    return tuple(records)


def _atlas_graph_edges(
    contract: OracleCoordinateAtlasContract,
    *,
    graph_role: str,
) -> tuple[tuple[int, int], ...]:
    """Return atom edges carrying one explicit ORACLE graph role."""

    return tuple(
        dict.fromkeys(
            tuple(sorted((int(item.endpoint_a[1]), int(item.endpoint_b[1]))))
            for item in contract.interactions
            if item.graph_role == graph_role and item.endpoint_a[0] == item.endpoint_b[0] == "ATOM"
        )
    )


def _task_context_diagnostics(
    context: str,
    *,
    transition_state_contract: object | None,
) -> tuple[str, ...]:
    records = [f"SCIENTIFIC_PATH {context.upper()}"]
    if context != _FRAGMENT_CONTEXT_TRANSITION_STATE:
        return tuple(records)
    if transition_state_contract is None:
        raise GICForgeContractError("missing ORACLE transition-state contract")
    records.extend(
        (
            f"TS_SOURCE {getattr(transition_state_contract, 'source')}",
            f"TS_GEOMETRY_CLASS {getattr(transition_state_contract, 'category_id')}",
            f"TS_CATALOG {getattr(transition_state_contract, 'catalog_id')}@"
            f"{getattr(transition_state_contract, 'catalog_version')}",
            f"TS_CHART_POLICY {getattr(transition_state_contract, 'chart_policy')}",
            f"TS_FROM_ENDPOINTS {getattr(transition_state_contract, 'endpoints_route_status')}",
        )
    )
    return tuple(records)


def _retain_individual_exocyclic_torsions(
    symmetrized: GICDefinition,
    unsymmetrized: GICDefinition,
) -> GICDefinition:
    """Restore one-dihedral exocyclic rows after retained-group adaptation."""

    source = tuple(gic for gic in unsymmetrized.gics if gic.family == "TORSION")
    if not source:
        return symmetrized
    output: list[FrozenGIC] = []
    inserted = False
    for gic in symmetrized.gics:
        if gic.family == "TORSION":
            if not inserted:
                output.extend(source)
                inserted = True
            continue
        output.append(gic)
    if not inserted:
        output.extend(source)
    renumbered = tuple(_renumber_frozen_gic(gic, index) for index, gic in enumerate(output, 1))
    diagnostics = symmetrized.symmetry_diagnostics
    if diagnostics is not None:
        diagnostics = replace(
            diagnostics,
            status=f"{diagnostics.status}_EXOCYCLIC_TORSIONS_SEPARATE",
            total_symmetric_gics=tuple(
                gic.name
                for gic in renumbered
                if gic.family != "TORSION"
                and is_total_symmetric_irrep(symmetrized.point_group, gic.irrep)
            ),
            groups=tuple(group for group in diagnostics.groups if group.family != "TORSION"),
        )
    return replace(
        symmetrized,
        gics=renumbered,
        symmetry_diagnostics=diagnostics,
    )


def construct_gic_definition_from_xyzin(
    path: Path,
    *,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
    rank_tolerance: float = RANK_TOLERANCE,
    retain_candidate_primitives: bool = False,
    coordinate_atlas_contract: OracleCoordinateAtlasContract | None = None,
    condition_transition_state_exact_chart: bool = True,
) -> tuple[GICDefinition, tuple[str, ...], tuple[GICPointGroupOperation, ...]]:
    """Construct and reduce GICs without applying symmetry adaptation."""
    target = Path(path)
    validate_gicforge_prerequisites(target)
    lines = read_sectioned_lines(target)
    semantic_contract = semantic_contract_from_sectioned_lines(lines)
    geometry = read_enriched_xyz(target)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    point_group = _point_group(lines)
    symmetry_operations = _symmetry_operations(lines)
    point_group, symmetry_operations = _apply_symmetry_group_limit(
        point_group,
        symmetry_operations,
        symmetry_group,
    )
    context = _fragment_context(fragment_context)
    coordinate_atlas_contract = _required_coordinate_atlas_contract(
        target,
        context=context,
        supplied=coordinate_atlas_contract,
    )
    transition_state_contract = (
        _required_transition_state_contract(target)
        if context == _FRAGMENT_CONTEXT_TRANSITION_STATE
        else None
    )
    separate_exocyclic_torsions = _resolved_separate_exocyclic_torsions(
        target,
        context=context,
        requested=separate_exocyclic_torsions,
    )
    mode = _atlas_fragment_mode(coordinate_atlas_contract)
    if fragment_mode is not None and _fragment_mode(fragment_mode) != mode:
        raise GICForgeContractError(
            "fragment_mode cannot override the frozen ORACLE coordinate atlas"
        )
    fragment_records = _fragment_records(target)
    # A complete one-component fragment contract is required before
    # validation, but it must be coordinate-neutral for a connected molecule.
    if len(fragment_records) <= 1:
        fragment_records = ()
    planned_xh_policy = _planned_xh_stretch_policy(lines)
    explicit_local_xh_selection = local_xh_bonds is not None or local_xh_classes is not None
    resolved_xh_policy = _xh_stretch_policy(
        XH_STRETCH_POLICY_LOCAL_SELECTED
        if xh_stretch_policy is None and explicit_local_xh_selection
        else planned_xh_policy
        if xh_stretch_policy is None
        else xh_stretch_policy
    )
    resolved_local_xh_bonds = _normalize_pairs(
        _planned_local_xh_bonds(lines) if local_xh_bonds is None else local_xh_bonds
    )
    resolved_local_xh_classes = _normalize_xh_classes(
        _planned_local_xh_classes(lines) if local_xh_classes is None else local_xh_classes
    )
    interaction_centers = _interaction_center_definition(target)
    bonds = _topology_bonds(lines, natoms=geometry.natoms)
    prescribed_pseudobonds = _atlas_atom_contacts(
        coordinate_atlas_contract,
        pseudobond_policy=PSEUDOBOND_REQUIRED,
    )
    prescribed_ts_reaction_distances = tuple(
        dict.fromkeys(
            tuple(sorted((left, right)))
            for role in (
                COORDINATE_REACTION_DISTANCE,
                COORDINATE_REACTION_DISTANCE_ONLY,
                COORDINATE_TS_SUPPORT,
            )
            for left, right, _kind in _atlas_atom_contacts(
                coordinate_atlas_contract,
                coordinate_role=role,
            )
        )
    )
    reactive_support_edges = _atlas_graph_edges(
        coordinate_atlas_contract,
        graph_role=GRAPH_ROLE_REACTIVE_SUPPORT,
    )
    task_context_diagnostics = _task_context_diagnostics(
        context,
        transition_state_contract=transition_state_contract,
    )
    rings = _topology_rings(lines, natoms=geometry.natoms)
    aromatic_atoms = topology_aromatic_atoms_from_lines(lines, natoms=geometry.natoms)
    bond_orders = topology_bond_orders_from_lines(lines, natoms=geometry.natoms)
    bond_order_components = topology_bond_order_components_from_lines(lines, natoms=geometry.natoms)
    primitive_source, primitive_source_schema, primitive_b_matrix_sha256 = (
        _consume_oracle_primitive_contract(target, coords, bonds=bonds)
    )
    fragment_index_by_atom = _fragment_index_by_atom(fragment_records)
    physical_fragment_contacts: dict[tuple[int, int], str] = {}
    atlas_distance_contacts = tuple(
        contact
        for contact in _atlas_atom_contacts(
            coordinate_atlas_contract,
            coordinate_role=COORDINATE_PHYSICAL_CONTACT_DISTANCE,
        )
        if tuple(sorted((int(contact[0]), int(contact[1]))))
        not in {
            tuple(sorted((int(left), int(right)))) for left, right, _kind in prescribed_pseudobonds
        }
    )
    for left, right, kind in atlas_distance_contacts:
        pair = tuple(sorted((int(left), int(right))))
        if pair in bonds or pair[0] == pair[1]:
            continue
        if (
            pair[0] not in fragment_index_by_atom
            or pair[1] not in fragment_index_by_atom
            or fragment_index_by_atom[pair[0]] == fragment_index_by_atom[pair[1]]
        ):
            continue
        physical_fragment_contacts.setdefault(pair, str(kind))
    fragment_contact_candidates = tuple(
        (left, right, physical_fragment_contacts[(left, right)])
        for left, right in sorted(physical_fragment_contacts)
    )

    if (
        context != _FRAGMENT_CONTEXT_TRANSITION_STATE
        and not semantic_contract.protect_coordinates
        and not fragment_records
        and not _requires_interaction_center_coordinates(interaction_centers)
        and mode != FRAGMENT_MODE_PSEUDO_BONDS
    ):
        definition = _connected_minimum_definition(
            atom_symbols=tuple(geometry.atoms),
            coords=coords,
            bonds=bonds,
            point_group=point_group,
            xh_stretch_policy=resolved_xh_policy,
            local_xh_bonds=resolved_local_xh_bonds,
            local_xh_classes=resolved_local_xh_classes,
            local_salc=local_salc,
            local_salc_settings=local_salc_settings,
            xy3_torsions=xy3_torsions,
            xy2_torsions=xy2_torsions,
            separate_exocyclic_torsions=separate_exocyclic_torsions,
            ring_puckering_model=ring_puckering_model,
            rings=rings,
            aromatic_atoms=aromatic_atoms,
            bond_orders=bond_orders,
            bond_order_components=bond_order_components,
            primitive_source=primitive_source,
            primitive_source_schema=primitive_source_schema,
            primitive_b_matrix_sha256=primitive_b_matrix_sha256,
            task_context_diagnostics=task_context_diagnostics,
        )
        return definition, tuple(geometry.atoms), symmetry_operations

    pseudo_bonds: tuple[tuple[int, int], ...] = ()
    pseudo_bond_kinds: tuple[str, ...] = ()
    target_rank = _vibrational_rank(coords)
    topology_bonds = bonds
    _constrained_disconnected_chart = bool(
        fragment_records
        and any(body.dimension != BODY_NONLINEAR for body in coordinate_atlas_contract.bodies)
    )
    execution_context = _ChartExecutionContext(
        coordinate_atlas_contract=coordinate_atlas_contract,
        topology_bonds=topology_bonds,
        rings=rings,
        coords=coords,
        atom_symbols=tuple(geometry.atoms),
        xh_stretch_policy=resolved_xh_policy,
        local_xh_bonds=resolved_local_xh_bonds,
        local_xh_classes=resolved_local_xh_classes,
        body_prescriptions=coordinate_atlas_contract.bodies,
        fragment_contacts=fragment_contact_candidates,
        semantic_contract=semantic_contract,
        symmetry_operations=symmetry_operations,
        prescribed_ts_reaction_distances=prescribed_ts_reaction_distances,
        reactive_support_edges=reactive_support_edges,
        bond_orders=bond_orders,
        aromatic_atoms=aromatic_atoms,
        ring_puckering_model=ring_puckering_model,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
        task_regime=coordinate_atlas_contract.task_regime,
        condition_ordinary=(mode == FRAGMENT_MODE_PSEUDO_BONDS and bool(fragment_records)),
        reactive_ts=(
            transition_state_contract is not None
            and transition_state_contract.chart_policy
            in {TS_CHART_REACTIVE_DISTANCE, TS_CHART_REACTIVE_PSEUDOBOND}
        ),
        pseudobond_out_of_plane_support=(
            _atlas_declares_pseudobond_out_of_plane_support(coordinate_atlas_contract)
        ),
        condition_transition_state_exact_chart=bool(condition_transition_state_exact_chart),
    )

    normalized_contacts = tuple(
        _normalize_pseudo_contact(contact) for contact in prescribed_pseudobonds
    )
    (
        candidate_bonds,
        pseudo_bonds,
        pseudo_bond_kinds,
        candidates,
        semantic_contract,
        semantic_diagnostics,
        fallback_events,
        selected,
        rank,
        reduction_diagnostics,
    ) = _evaluate_atlas_candidate_chart(
        execution_context,
        fragment_records if _atlas_needs_fragment_geometry(coordinate_atlas_contract) else (),
        interaction_centers,
        normalized_contacts,
    )
    bonds = candidate_bonds
    if rank != target_rank:
        raise GICForgeRankDeficiencyError(
            target_rank=target_rank,
            selected_rank=rank,
            candidate_count=len(candidates),
        )

    definition = _selected_chart_definition(
        candidates=tuple(candidates),
        selected=tuple(selected),
        retain_candidate_primitives=retain_candidate_primitives,
        candidate_bonds=candidate_bonds,
        coords=coords,
        atom_symbols=tuple(geometry.atoms),
        point_group=point_group,
        target_rank=target_rank,
        rank=rank,
        reduction_diagnostics=reduction_diagnostics,
        fragment_mode=mode if pseudo_bonds or fragment_records else FRAGMENT_MODE_NONE,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        xh_stretch_policy=resolved_xh_policy,
        local_xh_bonds=resolved_local_xh_bonds,
        local_xh_classes=resolved_local_xh_classes,
        rings=rings,
        bond_orders=bond_orders,
        bond_order_components=bond_order_components,
        aromatic_atoms=aromatic_atoms,
        ring_puckering_model=ring_puckering_model,
        semantic_contract=semantic_contract,
        semantic_diagnostics=semantic_diagnostics,
        fallback_events=fallback_events,
        task_context_diagnostics=task_context_diagnostics,
        primitive_source=primitive_source,
        primitive_source_schema=primitive_source_schema,
        primitive_b_matrix_sha256=primitive_b_matrix_sha256,
    )
    return definition, tuple(geometry.atoms), symmetry_operations


def _connected_minimum_definition(
    *,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    point_group: str,
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    local_salc: bool,
    local_salc_settings: object | None,
    xy3_torsions: bool,
    xy2_torsions: bool,
    separate_exocyclic_torsions: bool,
    ring_puckering_model: str,
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    aromatic_atoms: frozenset[int],
    bond_orders: dict[tuple[int, int], float],
    bond_order_components: dict,
    primitive_source: str,
    primitive_source_schema: str,
    primitive_b_matrix_sha256: str,
    task_context_diagnostics: tuple[str, ...],
) -> GICDefinition:
    definition = _construct_smith_definition(
        atom_symbols,
        coords,
        topology_bonds=bonds,
        frozen_rings=tuple(tuple(a - 1 for a in ring) for _, ring in rings),
        frozen_aromatic_atoms=frozenset(a - 1 for a in aromatic_atoms),
        point_group=point_group,
        improper_dihedrals=False,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        ring_puckering_model=ring_puckering_model,
        ring_puckering_diagnostics=_ring_puckering_diagnostics(
            rings,
            bond_orders=bond_orders,
            bond_order_components=bond_order_components,
            ring_puckering_model=ring_puckering_model,
        ),
    )
    definition = replace(
        definition,
        primitive_source=primitive_source,
        primitive_source_schema=primitive_source_schema,
        primitive_b_matrix_sha256=primitive_b_matrix_sha256,
        semantic_diagnostics=(
            *definition.semantic_diagnostics,
            *task_context_diagnostics,
        ),
        **_wilson_tangent_diagnostics(definition, coords),
    )
    return _migrate_legacy_u_definition(definition)


def _selected_chart_definition(
    *,
    candidates: tuple[GICPrimitive, ...],
    selected: tuple[GICPrimitive, ...],
    retain_candidate_primitives: bool,
    candidate_bonds: tuple[tuple[int, int], ...],
    coords: np.ndarray,
    atom_symbols: tuple[str, ...],
    point_group: str,
    target_rank: int,
    rank: int,
    reduction_diagnostics: GICReductionDiagnostics,
    fragment_mode: str,
    pseudo_bonds: tuple[tuple[int, int], ...],
    pseudo_bond_kinds: tuple[str, ...],
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    bond_orders: dict[tuple[int, int], float],
    bond_order_components: dict,
    aromatic_atoms: frozenset[int],
    ring_puckering_model: str,
    semantic_contract: object,
    semantic_diagnostics: tuple[str, ...],
    fallback_events: tuple[FallbackEvent, ...],
    task_context_diagnostics: tuple[str, ...],
    primitive_source: str,
    primitive_source_schema: str,
    primitive_b_matrix_sha256: str,
) -> GICDefinition:
    primitives = candidates if retain_candidate_primitives else selected
    gics = tuple(
        FrozenGIC(
            identifier=f"GIC{idx:03d}",
            name=primitive.name,
            family=primitive.family,
            irrep="A" if point_group.upper() == "C1" else "UNASSIGNED",
            primitive_id=primitive.identifier,
            gaussian_expression=primitive.gaussian_expression(),
            coefficients=((primitive.identifier, 1.0),),
        )
        for idx, primitive in enumerate(selected, start=1)
    )
    primitives, gics = materialize_ring_out_of_plane_salcs(primitives, gics)
    primitives, gics = _freeze_balanced_out_of_plane_coordinates(
        primitives,
        gics,
        selected=selected,
        bonds=candidate_bonds,
        coords=coords,
        atom_symbols=atom_symbols,
        bond_orders=bond_orders,
    )
    definition = GICDefinition(
        backend=GIC_BACKEND,
        point_group=point_group,
        symmetrize=False,
        target_rank=target_rank,
        rank=rank,
        candidate_count=len(candidates),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coords
        ),
        primitives=primitives,
        gics=gics,
        aromatic_atoms=tuple(sorted(int(atom) for atom in aromatic_atoms)),
        reduction_diagnostics=reduction_diagnostics,
        symmetry_diagnostics=_empty_symmetry_diagnostics(point_group, requested=False),
        fragment_mode=fragment_mode,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        ring_puckering_diagnostics=_ring_puckering_diagnostics(
            rings,
            bond_orders=bond_orders,
            bond_order_components=bond_order_components,
            ring_puckering_model=ring_puckering_model,
            aromatic_atoms=aromatic_atoms,
        ),
        semantic_grammar_version=(
            semantic_contract.grammar_version if semantic_contract.coordinates else ""
        ),
        semantic_diagnostics=(*semantic_diagnostics, *task_context_diagnostics),
        fallback_events=fallback_events,
        primitive_source=primitive_source,
        primitive_source_schema=primitive_source_schema,
        primitive_b_matrix_sha256=primitive_b_matrix_sha256,
    )
    definition = replace(definition, **_wilson_tangent_diagnostics(definition, coords))
    return _migrate_legacy_u_definition(definition)


def _candidate_chart_for_bonds(
    context: _ChartExecutionContext,
    fragment_records: tuple[object, ...],
    interaction_centers: object | None,
    *,
    bonds: tuple[tuple[int, int], ...],
    pseudo_bonds: tuple[tuple[int, int], ...],
    pseudo_bond_kinds: tuple[str, ...],
    include_fragment_pose_candidates: bool = True,
):
    """Generate and rank one mechanical realization of frozen atlas edges."""

    candidates = _primitive_candidates(
        bonds,
        rings=context.rings,
        coords=context.coords,
        natoms=len(context.atom_symbols),
        atom_symbols=context.atom_symbols,
        xh_stretch_policy=context.xh_stretch_policy,
        local_xh_bonds=context.local_xh_bonds,
        local_xh_classes=context.local_xh_classes,
        improper_dihedrals=False,
        fragment_records=fragment_records,
        body_prescriptions=context.body_prescriptions,
        fragment_contacts=context.fragment_contacts if fragment_records else (),
        interaction_centers=interaction_centers,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        pseudobond_out_of_plane_support=context.pseudobond_out_of_plane_support,
        bond_orders=context.bond_orders,
        aromatic_atoms=context.aromatic_atoms,
        ring_puckering_model=context.ring_puckering_model,
        reactive_support_edges=context.reactive_support_edges,
        include_fragment_pose_candidates=include_fragment_pose_candidates,
    )
    # Ordinary/ring valence torsions require a real central topology bond.
    # Metal-site and noncovalent atlas edges are represented by their
    # dedicated center/pose families and must never acquire an artificial
    # covalent torsional force constant.
    covalent_bonds = {tuple(sorted((int(left), int(right)))) for left, right in bonds}
    central_bond_torsion_families = {
        "TORSION",
        "CYCLIC_TORSION",
        "CONDENSED_RING_TORSION",
        "BUTTERFLY",
    }
    candidates = tuple(
        replace(primitive, provenance="ORACLE_REACTIVE_SUPPORT")
        if primitive.family in central_bond_torsion_families
        and len(primitive.atoms) == 4
        and tuple(sorted(primitive.atoms[1:3])) not in covalent_bonds
        and tuple(sorted(primitive.atoms[1:3])) in context.reactive_support_edges
        else primitive
        for primitive in candidates
        if not (
            primitive.family in central_bond_torsion_families
            and len(primitive.atoms) == 4
            and tuple(sorted((primitive.atoms[1], primitive.atoms[2]))) not in covalent_bonds
            and tuple(sorted(primitive.atoms[1:3])) not in context.reactive_support_edges
        )
    )
    candidates = _with_prescribed_distance_only_primitives(
        candidates,
        context.prescribed_ts_reaction_distances,
    )
    candidates = _with_transition_state_reaction_distance_family(
        candidates,
        context.prescribed_ts_reaction_distances,
    )
    candidates, reactive_zone_diagnostics = apply_reactive_zone_exclusions(
        candidates,
        context.coordinate_atlas_contract,
    )
    candidates, semantic_contract, semantic_diagnostics, fallback_events = (
        _apply_semantic_contract_to_candidates(
            candidates,
            context.semantic_contract,
            symmetry_operations=context.symmetry_operations,
        )
    )
    partitioned_candidates = _fragment_partitioned_candidate_order(
        candidates,
        fragment_records=fragment_records,
        coords=context.coords,
        rank_tolerance=context.rank_tolerance,
    )
    if partitioned_candidates is not None:
        candidates_for_selection = partitioned_candidates
    else:
        candidates_for_selection = candidates
    selected, rank, reduction_diagnostics = _select_rank_aware_chart(
        candidates_for_selection,
        context.coords,
        target_rank=context.target_rank,
        rank_tolerance=context.rank_tolerance,
        condition_ordinary=context.condition_ordinary,
        transition_state=(context.task_regime == ATLAS_TASK_TRANSITION_STATE),
        coordinate_atlas_contract=context.coordinate_atlas_contract,
        condition_pseudobond_support=(
            context.task_regime == ATLAS_TASK_MINIMUM and len(context.symmetry_operations) <= 1
        ),
        condition_transition_state_exact_chart=(
            context.condition_transition_state_exact_chart and len(context.symmetry_operations) <= 1
        ),
    )
    if partitioned_candidates is not None:
        original_by_id = {primitive.identifier: primitive for primitive in candidates}
        selected = tuple(original_by_id[primitive.identifier] for primitive in selected)
    return (
        candidates,
        semantic_contract,
        (*reactive_zone_diagnostics, *semantic_diagnostics),
        fallback_events,
        selected,
        rank,
        reduction_diagnostics,
    )


def _fragment_partitioned_candidate_order(
    candidates: tuple[GICPrimitive, ...],
    *,
    fragment_records: tuple[object, ...],
    coords: np.ndarray,
    rank_tolerance: float,
) -> tuple[GICPrimitive, ...] | None:
    """Protect each fragment's complete internal tangent space before closure.

    Contact and rigid-pose rows may have support on many atoms and can span
    intrafragment motions numerically.  A global pivot can therefore replace
    chemically meaningful internal SONICs with redundant interfragment rows.
    ORACLE already supplies the fragment partition, so SMITH first freezes the
    exact vibrational rank of every fragment and only then completes the six
    relative rigid-body degrees of freedom.
    """

    if len(fragment_records) <= 1:
        return None
    protected: list[GICPrimitive] = []
    protected_ids: set[str] = set()
    for record in sorted(fragment_records, key=lambda item: str(getattr(item, "identifier"))):
        atoms = frozenset(int(atom) for atom in getattr(record, "atoms"))
        local_pool = tuple(
            primitive
            for primitive in candidates
            if primitive.identifier not in protected_ids
            and primitive.family
            not in {
                "FRAG_DISTANCE",
                "FRAG_TRANSLATION",
                "FRAG_ORIENTATION",
                "PSEUDO_BOND_DISTANCE",
                "PSEUDO_BOND_BEND",
                "PSEUDO_BOND_TORSION",
            }
            and primitive.refs[:1] != ("PSEUDOBOND_CONTACT_SUPPORT",)
            and frozenset(
                (
                    *primitive.atoms,
                    *primitive.ref_atoms,
                    *primitive.frame_atoms,
                    *primitive.ref_frame_atoms,
                )
            )
            and frozenset(
                (
                    *primitive.atoms,
                    *primitive.ref_atoms,
                    *primitive.frame_atoms,
                    *primitive.ref_frame_atoms,
                )
            ).issubset(atoms)
        )
        local_target = _vibrational_rank(
            coords[np.asarray(tuple(sorted(atom - 1 for atom in atoms)), dtype=int)]
        )
        local_selected, local_rank = _select_ranked_primitives(
            local_pool,
            coords,
            target_rank=local_target,
            rank_tolerance=rank_tolerance,
        )
        if local_rank != local_target:
            return None
        protected.extend(local_selected)
        protected_ids.update(item.identifier for item in local_selected)
    # The ORACLE-classified physical contact is part of the chart semantics,
    # not merely one numerical completion candidate.  Freeze one distance for
    # every required pseudobond before selecting the remaining relative pose.
    required_contacts = tuple(
        primitive
        for primitive in candidates
        if primitive.family == "PSEUDO_BOND_DISTANCE" and primitive.identifier not in protected_ids
    )
    protected.extend(required_contacts)
    protected_ids.update(item.identifier for item in required_contacts)
    protected_candidates = tuple(
        replace(primitive, provenance=USER_PROVENANCE) for primitive in protected
    )
    return protected_candidates + tuple(
        primitive for primitive in candidates if primitive.identifier not in protected_ids
    )


def _evaluate_atlas_candidate_chart(
    context: _ChartExecutionContext,
    fragment_records: tuple[object, ...],
    interaction_centers: object | None,
    normalized_contacts: tuple[tuple[int, int, str], ...],
):
    pseudo_bonds = tuple((left, right) for left, right, _kind in normalized_contacts)
    pseudo_bond_kinds = tuple(kind for _left, _right, kind in normalized_contacts)
    # A noncovalent pseudobond closes the interfragment chart but is not a
    # covalent topology edge.  Feeding it to the ordinary valence generator
    # creates cross-fragment angles and torsions that ORACLE never prescribed
    # and destroys the intra/inter partition.
    primary_bonds = tuple(
        sorted(
            set(context.topology_bonds + pseudo_bonds)
            if context.task_regime == ATLAS_TASK_TRANSITION_STATE
            else set(context.topology_bonds)
        )
    )
    local_completion_available = bool(
        context.pseudobond_out_of_plane_support and pseudo_bonds and fragment_records
    )
    attempt = _candidate_chart_for_bonds(
        context,
        () if local_completion_available else fragment_records,
        interaction_centers,
        bonds=primary_bonds,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        # Select the chemical/contact chart without letting an additional
        # rigid-pose pool consume intrafragment rank.  Pose primitives are
        # retained below as valid alternatives for conditioning and refresh.
        include_fragment_pose_candidates=not bool(pseudo_bonds),
    )
    selected_with_pose = False
    if (
        attempt[-2] != context.target_rank
        and pseudo_bonds
        and fragment_records
        and context.body_prescriptions
    ):
        # A contact graph need not provide all 6(F-1) relative rigid-body
        # directions (large solvent clusters are the canonical case).  Retry
        # with ORACLE's body-pose measurements, while the fragment partition
        # above still protects every intrafragment vibrational subspace.
        attempt = _candidate_chart_for_bonds(
            context,
            fragment_records,
            interaction_centers,
            bonds=primary_bonds,
            pseudo_bonds=pseudo_bonds,
            pseudo_bond_kinds=pseudo_bond_kinds,
            include_fragment_pose_candidates=True,
        )
        selected_with_pose = True
    if attempt[-2] == context.target_rank:
        if (
            pseudo_bonds
            and fragment_records
            and context.body_prescriptions
            and not selected_with_pose
        ):
            enriched = _candidate_chart_for_bonds(
                context,
                fragment_records,
                interaction_centers,
                bonds=primary_bonds,
                pseudo_bonds=pseudo_bonds,
                pseudo_bond_kinds=pseudo_bond_kinds,
                include_fragment_pose_candidates=True,
            )
            base_candidates = tuple(attempt[0])
            pose_candidates = tuple(
                primitive
                for primitive in enriched[0]
                if primitive.function
                in {"FC_DIST", "FCA_DIST", "FTRANS", "FLIN_TRANS", "FROT", "FAXIS"}
            )
            # Adding pose primitives changes the generator's global P-number
            # sequence.  Never replace the authoritative base pool with that
            # independently numbered pool: doing so silently rebinds existing
            # GIC coefficients to unrelated primitives.  Append only the new
            # measurements under fresh immutable identifiers.
            numeric_ids = tuple(
                int(match.group(1))
                for primitive in base_candidates
                if (match := re.fullmatch(r"P(\d+)", primitive.identifier)) is not None
            )
            next_id = max(numeric_ids, default=0) + 1
            existing_keys = {
                (
                    primitive.function,
                    primitive.atoms,
                    primitive.mode,
                    primitive.ref_atoms,
                    primitive.frame_atoms,
                    primitive.ref_frame_atoms,
                )
                for primitive in base_candidates
            }
            appended = []
            for primitive in pose_candidates:
                key = (
                    primitive.function,
                    primitive.atoms,
                    primitive.mode,
                    primitive.ref_atoms,
                    primitive.frame_atoms,
                    primitive.ref_frame_atoms,
                )
                if key in existing_keys:
                    continue
                appended.append(replace(primitive, identifier=f"P{next_id:03d}"))
                existing_keys.add(key)
                next_id += 1
            attempt = (
                (*base_candidates, *appended),
                attempt[1],
                attempt[2],
                attempt[3],
                attempt[4],
                attempt[5],
                attempt[6],
            )
        return (primary_bonds, pseudo_bonds, pseudo_bond_kinds, *attempt)

    support_bonds = tuple(sorted(set(primary_bonds + context.reactive_support_edges)))
    best_bonds, best = primary_bonds, attempt
    if support_bonds != primary_bonds:
        graph_completed = _candidate_chart_for_bonds(
            context,
            () if local_completion_available else fragment_records,
            interaction_centers,
            bonds=support_bonds,
            pseudo_bonds=pseudo_bonds,
            pseudo_bond_kinds=pseudo_bond_kinds,
            include_fragment_pose_candidates=bool(context.body_prescriptions),
        )
        if graph_completed[-2] > best[-2]:
            edge_text = ",".join(
                f"{left}-{right}" for left, right in context.reactive_support_edges
            )
            completion_event = make_fallback_event(
                stage="SMITH_RANK_COMPLETION",
                algorithm_id="ATLAS_REACTIVE_SUPPORT_GRAPH",
                trigger="PRIMARY_TS_CHART_BELOW_EXACT_RANK_AFTER_REACTIVE_ZONE_EXCLUSIONS",
                domain="TS_REACTIVE_ZONE",
                macrofamily="REACTIVE_LOCAL_VALENCE",
                rank_before=best[-2],
                rank_after=graph_completed[-2],
            )
            best_bonds = support_bonds
            best = (
                graph_completed[0],
                graph_completed[1],
                (
                    f"ATLAS_REACTIVE_SUPPORT_COMPLETION EDGES={edge_text} "
                    f"RANK={attempt[-2]}/{context.target_rank}->"
                    f"{graph_completed[-2]}/{context.target_rank}",
                    *graph_completed[2],
                ),
                merge_fallback_events(graph_completed[3], (completion_event,)),
                *graph_completed[4:],
            )
    if best[-2] == context.target_rank:
        return (best_bonds, pseudo_bonds, pseudo_bond_kinds, *best)
    if not local_completion_available and context.reactive_support_edges:
        return _evaluate_reactive_connectivity_pool(
            context,
            fragment_records,
            interaction_centers,
            pseudo_bonds,
            pseudo_bond_kinds,
        )
    if not local_completion_available:
        return (best_bonds, pseudo_bonds, pseudo_bond_kinds, *best)

    local_completed = _candidate_chart_for_bonds(
        context,
        fragment_records,
        interaction_centers,
        bonds=best_bonds,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        include_fragment_pose_candidates=bool(context.body_prescriptions),
    )
    if local_completed[-2] <= best[-2]:
        if context.reactive_support_edges:
            return _evaluate_reactive_connectivity_pool(
                context,
                fragment_records,
                interaction_centers,
                pseudo_bonds,
                pseudo_bond_kinds,
            )
        return (best_bonds, pseudo_bonds, pseudo_bond_kinds, *best)
    completion_event = make_fallback_event(
        stage="SMITH_RANK_COMPLETION",
        algorithm_id="ATLAS_TS_REACTIVE_LOCAL_COMPLETION",
        trigger="SUPPORT_GRAPH_BELOW_EXACT_RANK_AFTER_REACTIVE_ZONE_EXCLUSIONS",
        domain="TS_REACTIVE_ZONE",
        macrofamily="TS_REACTIVE_COMPLETION",
        rank_before=best[-2],
        rank_after=local_completed[-2],
    )
    local_completed = (
        local_completed[0],
        local_completed[1],
        (
            "ATLAS_TS_REACTIVE_LOCAL_COMPLETION "
            f"RANK={best[-2]}/{context.target_rank}->"
            f"{local_completed[-2]}/{context.target_rank}",
            *best[2],
            *local_completed[2],
        ),
        merge_fallback_events(
            best[3],
            local_completed[3],
            (completion_event,),
        ),
        *local_completed[4:],
    )
    if local_completed[-2] < context.target_rank and context.reactive_support_edges:
        return _evaluate_reactive_connectivity_pool(
            context,
            fragment_records,
            interaction_centers,
            pseudo_bonds,
            pseudo_bond_kinds,
        )
    return (best_bonds, pseudo_bonds, pseudo_bond_kinds, *local_completed)


def _evaluate_reactive_connectivity_pool(
    context: _ChartExecutionContext,
    fragment_records: tuple[object, ...],
    interaction_centers: object | None,
    pseudo_bonds: tuple[tuple[int, int], ...],
    pseudo_bond_kinds: tuple[str, ...],
):
    """Pool primitives from every local connectivity allowed by ORACLE.

    The frozen TS kernel defines the only edges whose connectivity may vary.
    All other topology edges form the conserved graph. Primitive candidates
    common to every realization are placed first so the rank selector retains the
    conserved SONIC block before completing it from the reactive union.
    """

    support = tuple(sorted(set(context.reactive_support_edges)))
    support_set = set(support)
    conserved_bonds = set(context.topology_bonds).difference(support_set)
    optional = support
    bond_sets = tuple(
        tuple(sorted(conserved_bonds.union(subset)))
        for count in range(len(optional) + 1)
        for subset in combinations(optional, count)
    )
    bond_sets = tuple(dict.fromkeys(bond_sets))
    attempts = tuple(
        _candidate_chart_for_bonds(
            context,
            fragment_records,
            interaction_centers,
            bonds=bonds,
            pseudo_bonds=pseudo_bonds,
            pseudo_bond_kinds=pseudo_bond_kinds,
            include_fragment_pose_candidates=bool(context.body_prescriptions),
        )
        for bonds in bond_sets
    )
    candidate_sets = tuple(attempt[0] for attempt in attempts)
    structural_keys = tuple(
        tuple(_reactive_pool_primitive_key(primitive) for primitive in candidates)
        for candidates in candidate_sets
    )
    common_keys = set(structural_keys[0])
    for keys in structural_keys[1:]:
        common_keys.intersection_update(keys)
    ordered_keys: list[tuple[object, ...]] = []
    primitive_by_key: dict[tuple[object, ...], GICPrimitive] = {}
    for primitive in candidate_sets[0]:
        key = _reactive_pool_primitive_key(primitive)
        if key in common_keys and key not in primitive_by_key:
            primitive_by_key[key] = primitive
            ordered_keys.append(key)
    for candidates in candidate_sets:
        for primitive in candidates:
            key = _reactive_pool_primitive_key(primitive)
            if key not in primitive_by_key:
                primitive_by_key[key] = primitive
                ordered_keys.append(key)
    used_names: set[str] = set()
    merged: list[GICPrimitive] = []
    for index, key in enumerate(ordered_keys, 1):
        primitive = primitive_by_key[key]
        name = primitive.name
        if name in used_names:
            serial = 2
            while f"{name}_C{serial}" in used_names:
                serial += 1
            name = f"{name}_C{serial}"
        used_names.add(name)
        merged.append(
            replace(
                primitive,
                identifier=f"P{index:03d}",
                name=name,
            )
        )
    merged_candidates = tuple(merged)
    selected, rank, reduction_diagnostics = _select_rank_aware_chart(
        merged_candidates,
        context.coords,
        target_rank=context.target_rank,
        rank_tolerance=context.rank_tolerance,
        condition_ordinary=context.condition_ordinary,
        transition_state=True,
        coordinate_atlas_contract=context.coordinate_atlas_contract,
        condition_transition_state_exact_chart=(
            context.condition_transition_state_exact_chart and len(context.symmetry_operations) <= 1
        ),
    )
    if rank == context.target_rank:
        selected, reduction_diagnostics = _select_conserved_reactive_exact_chart(
            merged_candidates,
            len(common_keys),
            context.coords,
            target_rank=context.target_rank,
            rank_tolerance=context.rank_tolerance,
            diagnostics=reduction_diagnostics,
            coordinate_atlas_contract=context.coordinate_atlas_contract,
        )
    best_index = min(
        range(len(attempts)),
        key=lambda index: (
            -int(attempts[index][-2]),
            len(bond_sets[index]),
            bond_sets[index],
        ),
    )
    best = attempts[best_index]
    semantic_diagnostics = (
        "TS_REACTIVE_CONNECTIVITY_POOL "
        f"VARIANTS={len(bond_sets)} COMMON={len(common_keys)} "
        f"UNION={len(merged_candidates)} SELECTED={len(selected)}",
        *best[2],
    )
    fallback_events = merge_fallback_events(*(attempt[3] for attempt in attempts))
    candidate_bonds = tuple(sorted(set().union(*map(set, bond_sets))))
    return (
        candidate_bonds,
        pseudo_bonds,
        pseudo_bond_kinds,
        merged_candidates,
        best[1],
        semantic_diagnostics,
        fallback_events,
        selected,
        rank,
        reduction_diagnostics,
    )


def _reactive_pool_primitive_key(primitive: GICPrimitive) -> tuple[object, ...]:
    """Geometry-independent identity used across connectivity realizations."""

    return (
        primitive.family,
        primitive.function,
        primitive.atoms,
        primitive.mode,
        primitive.ref_atoms,
        primitive.refs,
        primitive.frame_atoms,
        primitive.ref_frame_atoms,
        primitive.provenance,
        primitive.semantic_id,
        primitive.semantic_type,
        primitive.chart,
        primitive.chart_reference_radian,
    )


def _requires_interaction_center_coordinates(interaction_centers: object | None) -> bool:
    """Return whether declared atom--center interactions require special SONICs.

    Materialized bond and ring centers are reference data. Their mere
    presence must not replace the ordinary, full-rank molecular SONIC path;
    only an actual atom--center interaction consumes those centers.
    """

    if interaction_centers is None:
        return False
    interactions = tuple(getattr(interaction_centers, "interactions", ()) or ())
    return bool(interactions)


def _migrate_legacy_u_definition(definition: GICDefinition) -> GICDefinition:
    """Preserve frozen v1 U values while exposing only Gaussian-order primitives."""
    if definition.primitive_source_schema not in {
        LEGACY_MATRIX_XYZ_PRIMITIVES_SCHEMA,
        LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA,
    }:
        return definition
    replacements: dict[str, tuple[str, str]] = {}
    primitives: list[GICPrimitive] = []
    for primitive in definition.primitives:
        if primitive.function != "U" or len(primitive.atoms) != 4:
            primitives.append(primitive)
            continue
        i, j, k, ell = primitive.atoms
        migrated = replace(primitive, atoms=(j, ell, k, i))
        replacements[primitive.identifier] = (
            primitive.gaussian_expression(),
            migrated.gaussian_expression(),
        )
        primitives.append(migrated)
    if not replacements:
        return definition
    gics: list[FrozenGIC] = []
    for gic in definition.gics:
        expression = gic.gaussian_expression
        for old, new in replacements.values():
            expression = expression.replace(old, new)
        gics.append(replace(gic, gaussian_expression=expression))
    return replace(definition, primitives=tuple(primitives), gics=tuple(gics))


def _consume_oracle_primitive_contract(
    path: Path,
    coordinates_angstrom: np.ndarray,
    *,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[str, str, str]:
    """Validate the required ORACLE primitive/B contract without reperception."""
    if not section_content(read_sectioned_lines(Path(path)), "PRIMITIVES"):
        raise GICForgeContractError("SMITH requires the frozen ORACLE #PRIMITIVES contract")
    try:
        contract = read_primitive_contract(Path(path))
        validate_primitive_contract(contract, coordinates_angstrom)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid ORACLE #PRIMITIVES contract: {exc}") from exc
    contract_bonds = {
        tuple(sorted(atom + 1 for atom in primitive.atoms))
        for primitive in contract.primitives
        if primitive.kind == "bond"
    }
    topology_bonds = {tuple(sorted(pair)) for pair in bonds}
    if contract_bonds != topology_bonds:
        raise GICForgeContractError(
            "ORACLE #PRIMITIVES bond rows do not match the frozen #TOPOLOGY graph"
        )
    return (
        "ORACLE_CONTRACT",
        contract.schema or MATRIX_XYZ_PRIMITIVES_SCHEMA,
        contract.b_matrix_sha256,
    )


def _construct_smith_definition(
    atom_symbols: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    *,
    topology_bonds: tuple[tuple[int, int], ...],
    frozen_rings: tuple[tuple[int, ...], ...],
    frozen_aromatic_atoms: frozenset[int],
    point_group: str,
    improper_dihedrals: bool,
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
    ring_puckering_diagnostics: tuple[str, ...] = (),
) -> GICDefinition:
    from matrix_smith.runtime.gicforge_python import build_gicforge_python_model

    model = build_gicforge_python_model(
        atom_symbols,
        coordinates_angstrom,
        topology_bonds=tuple((left - 1, right - 1) for left, right in topology_bonds),
        frozen_rings=frozen_rings,
        frozen_aromatic_atoms=frozen_aromatic_atoms,
        impdih=improper_dihedrals,
        onedih=True,
        svd_local=False,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        ring_puckering_model=ring_puckering_model,
    )
    primitive_ids: dict[tuple[object, str], str] = {}
    primitive_indices: dict[str, int] = {}
    primitives: list[GICPrimitive] = []
    gics: list[FrozenGIC] = []
    xh_class_by_bond = _smith_xh_class_by_bond(model.primitive_candidates, atom_symbols)

    def primitive_id(
        primitive: object,
        family: str,
        *,
        refs: tuple[str, ...] = (),
    ) -> str:
        key = (primitive, family)
        existing = primitive_ids.get(key)
        if existing is not None:
            if refs:
                primitive_index = primitive_indices[existing]
                current = primitives[primitive_index]
                merged_refs = current.refs + tuple(ref for ref in refs if ref not in current.refs)
                if merged_refs != current.refs:
                    primitives[primitive_index] = replace(current, refs=merged_refs)
            return existing
        identifier = f"P{len(primitives) + 1:03d}"
        primitive_ids[key] = identifier
        primitive_indices[identifier] = len(primitives)
        primitives.append(_smith_runtime_primitive(identifier, primitive, family, refs=refs))
        return identifier

    for index, coordinate in enumerate(model.coordinates, start=1):
        family = _smith_coordinate_family(coordinate.name, coordinate.block)
        if "FULLERENE_CAGE_FACE_DIAGONAL=YES" in str(coordinate.diagnostic):
            family = "FULLERENE_FACE_DIAGONAL"
        family = _smith_local_xh_family(
            coordinate,
            family,
            atom_symbols,
            xh_stretch_policy=xh_stretch_policy,
            local_xh_bonds=local_xh_bonds,
            local_xh_classes=local_xh_classes,
            xh_class_by_bond=xh_class_by_bond,
        )
        refs = _smith_coordinate_primitive_refs(coordinate, family)
        coefficients = tuple(
            (primitive_id(primitive, family, refs=refs), float(coefficient))
            for coefficient, primitive in coordinate.terms
            if abs(float(coefficient)) > 1.0e-14
        )
        coefficients = _normalized_coefficients(coefficients)
        if not coefficients:
            raise GICForgeContractError(f"empty SMITH coordinate {coordinate.name!r}")
        gics.append(
            FrozenGIC(
                identifier=f"GIC{index:03d}",
                name=coordinate.name,
                family=family,
                irrep="A" if point_group.upper() == "C1" else "UNASSIGNED",
                primitive_id=coefficients[0][0],
                gaussian_expression="SMITH_ACTIVE",
                coefficients=coefficients,
            )
        )

    for candidate in model.primitive_candidates:
        family = _smith_coordinate_family(candidate.name, candidate.block)
        if "FULLERENE_CAGE_FACE_DIAGONAL=YES" in str(candidate.diagnostic):
            family = "FULLERENE_FACE_DIAGONAL"
        family = _smith_local_xh_family(
            candidate,
            family,
            atom_symbols,
            xh_stretch_policy=xh_stretch_policy,
            local_xh_bonds=local_xh_bonds,
            local_xh_classes=local_xh_classes,
            xh_class_by_bond=xh_class_by_bond,
        )
        refs = _smith_coordinate_primitive_refs(candidate, family)
        for _coefficient, primitive in candidate.terms:
            primitive_id(primitive, family, refs=refs)

    diagnostics = model.diagnostics
    ring_puckering_diagnostics = _effective_ring_puckering_diagnostics(
        ring_puckering_diagnostics,
        model.primitive_candidates,
    )
    return GICDefinition(
        backend="matrix-smith-gicforge.v1",
        point_group=point_group,
        symmetrize=False,
        target_rank=model.target_rank,
        rank=len(model.coordinates),
        candidate_count=len(model.primitive_candidates),
        reference_coordinates_angstrom=model.coordinates_angstrom,
        primitives=tuple(primitives),
        gics=tuple(gics),
        reduction_diagnostics=GICReductionDiagnostics(
            rank_method="smith_type_local_pruning",
            reduction_policy="SMITH_HOMOGENEOUS_TYPE_BLOCKS",
            selected=tuple(gic.identifier for gic in gics),
            skipped_singular=(),
            skipped_dependent=tuple(
                f"{block}:{count}"
                for block, count in sorted(
                    dict(diagnostics.get("removed_counts_by_block", {})).items()
                )
                if int(count) > 0
            ),
            skipped_dependent_details=tuple(
                str(item) for item in diagnostics.get("local_equivalence", ())
            ),
        ),
        symmetry_diagnostics=_empty_symmetry_diagnostics(point_group, requested=False),
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        ring_puckering_diagnostics=ring_puckering_diagnostics,
        fallback_events=model.fallback_events,
    )


def _effective_ring_puckering_diagnostics(
    report_lines: tuple[str, ...],
    candidates: tuple[object, ...],
) -> tuple[str, ...]:
    """Make the public report describe the primitive model actually built."""

    aromatic_rings: set[int] = set()
    for coordinate in candidates:
        if str(getattr(coordinate, "block", "")) != "RPck":
            continue
        diagnostic = str(getattr(coordinate, "diagnostic", ""))
        if AROMATIC_LOCAL_MODEL_DIAGNOSTIC not in diagnostic:
            continue
        match = re.search(r"\bDOMAIN=RING:(\d+)\b", diagnostic)
        if match is not None:
            aromatic_rings.add(int(match.group(1)))

    if not aromatic_rings:
        return report_lines

    output: list[str] = []
    for line in report_lines:
        match = re.match(r"RING\s+(\d+)\b", line)
        if match is None or int(match.group(1)) not in aromatic_rings:
            output.append(line)
            continue
        prefix = line.split(" MODEL=", 1)[0]
        output.append(
            f"{prefix} {AROMATIC_LOCAL_MODEL_DIAGNOSTIC} PRIMITIVE=GAUSSIAN_U WEIGHTING=LOCAL_SALC"
        )
    return tuple(output)


def _smith_runtime_primitive(
    identifier: str,
    primitive: object,
    family: str,
    *,
    refs: tuple[str, ...] = (),
) -> GICPrimitive:
    kind = str(getattr(primitive, "kind"))
    atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms"))
    ref_atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "ref", ()))
    mode = int(getattr(primitive, "mode", 0))
    if kind == "bond":
        return GICPrimitive(identifier, identifier, family, "R", atoms, refs=refs)
    if kind == "inverse_bond":
        return GICPrimitive(identifier, identifier, family, "IR", atoms, refs=refs)
    if kind == "inverse_cubic_bond":
        return GICPrimitive(identifier, identifier, family, "IR3", atoms, refs=refs)
    if kind == "exponential_bond":
        return GICPrimitive(identifier, identifier, family, "ER", atoms, refs=refs)
    if kind == "angle":
        return GICPrimitive(identifier, identifier, family, "A", atoms, refs=refs)
    if kind == "linear_bend":
        return GICPrimitive(
            identifier,
            identifier,
            family,
            "L",
            atoms,
            mode=mode,
            ref_atoms=ref_atoms,
            refs=refs,
        )
    if kind == "dihedral":
        return GICPrimitive(identifier, identifier, family, "D", atoms, refs=refs)
    if kind == "out_of_plane":
        return GICPrimitive(identifier, identifier, family, "U", atoms, refs=refs)
    if kind == "out_of_plane_height":
        return GICPrimitive(identifier, identifier, family, "H", atoms, refs=refs)
    raise GICForgeContractError(f"unsupported SMITH primitive kind: {kind}")


def _normalized_coefficients(
    coefficients: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    norm = float(np.sqrt(sum(float(coefficient) ** 2 for _identifier, coefficient in coefficients)))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        return coefficients
    return tuple(
        (identifier, float(coefficient) / norm) for identifier, coefficient in coefficients
    )


def _smith_coordinate_primitive_refs(
    coordinate: object,
    family: str,
) -> tuple[str, ...]:
    if family not in {"RING_PUCKER_COMPONENT", "FULLERENE_FACE_DIAGONAL"}:
        return ()
    ring = _smith_coordinate_ring_atoms(coordinate)
    if not ring:
        return ()
    return (_ring_ref_text(ring),)


def _smith_local_xh_family(
    coordinate: object,
    family: str,
    atom_symbols: tuple[str, ...],
    *,
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    xh_class_by_bond: dict[tuple[int, int], str],
) -> str:
    if family != "STRETCH":
        return family
    terms = tuple(getattr(coordinate, "terms", ()))
    if not terms:
        return family
    for _coefficient, primitive in terms:
        if str(getattr(primitive, "kind", "")) != "bond":
            return family
        atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms", ()))
        if len(atoms) != 2 or not _use_local_xh_stretch(
            atoms[0],
            atoms[1],
            atom_symbols,
            xh_stretch_policy,
            local_xh_bonds,
            local_xh_classes,
            xh_class_by_bond,
        ):
            return family
    return "LOCAL_XH_STRETCH"


def _use_local_xh_stretch(
    left: int,
    right: int,
    atom_symbols: tuple[str, ...],
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    xh_class_by_bond: dict[tuple[int, int], str],
) -> bool:
    if not _is_xh_bond(left, right, atom_symbols):
        return False
    policy = _xh_stretch_policy(xh_stretch_policy)
    if policy == XH_STRETCH_POLICY_SYMMETRIZE:
        return False
    if policy == XH_STRETCH_POLICY_LOCAL_ALL:
        return True
    pair = _pair_key(left, right)
    if pair in set(local_xh_bonds):
        return True
    return xh_class_by_bond.get(pair, "") in set(local_xh_classes)


def _is_xh_bond(left: int, right: int, atom_symbols: tuple[str, ...]) -> bool:
    if left < 1 or right < 1 or left > len(atom_symbols) or right > len(atom_symbols):
        return False
    try:
        left_z = atomic_number(str(atom_symbols[left - 1]))
        right_z = atomic_number(str(atom_symbols[right - 1]))
    except KeyError:
        return False
    return (left_z == 1) ^ (right_z == 1)


def _smith_xh_class_by_bond(
    candidates: tuple[object, ...],
    atom_symbols: tuple[str, ...],
) -> dict[tuple[int, int], str]:
    bonds: set[tuple[int, int]] = set()
    for candidate in candidates:
        for _coefficient, primitive in getattr(candidate, "terms", ()):
            if str(getattr(primitive, "kind", "")) != "bond":
                continue
            atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms", ()))
            if len(atoms) == 2:
                bonds.add(_pair_key(atoms[0], atoms[1]))
    return _xh_class_by_bond(tuple(sorted(bonds)), atom_symbols)


def _xh_class_by_bond(
    bonds: tuple[tuple[int, int], ...],
    atom_symbols: tuple[str, ...],
) -> dict[tuple[int, int], str]:
    hydrogens_by_heavy: dict[int, set[int]] = {}
    for left, right in bonds:
        if not _is_xh_bond(left, right, atom_symbols):
            continue
        heavy, hydrogen = _xh_heavy_and_hydrogen(left, right, atom_symbols)
        if heavy is None or hydrogen is None:
            continue
        hydrogens_by_heavy.setdefault(heavy, set()).add(hydrogen)
    result: dict[tuple[int, int], str] = {}
    for heavy, hydrogens in hydrogens_by_heavy.items():
        count = len(hydrogens)
        if count <= 1:
            xh_class = "XH"
        elif count == 2:
            xh_class = "XH2"
        else:
            xh_class = "XH3"
        for hydrogen in hydrogens:
            result[_pair_key(heavy, hydrogen)] = xh_class
    return result


def _xh_heavy_and_hydrogen(
    left: int,
    right: int,
    atom_symbols: tuple[str, ...],
) -> tuple[int | None, int | None]:
    if left < 1 or right < 1 or left > len(atom_symbols) or right > len(atom_symbols):
        return None, None
    try:
        left_z = atomic_number(str(atom_symbols[left - 1]))
        right_z = atomic_number(str(atom_symbols[right - 1]))
    except KeyError:
        return None, None
    if left_z == 1 and right_z != 1:
        return right, left
    if right_z == 1 and left_z != 1:
        return left, right
    return None, None


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (int(left), int(right)) if int(left) <= int(right) else (int(right), int(left))


def _xh_stretch_policy(value: str | None) -> str:
    if value is None:
        return XH_STRETCH_POLICY_SYMMETRIZE
    text = str(value).strip().replace("-", "_").upper()
    if not text:
        return XH_STRETCH_POLICY_SYMMETRIZE
    aliases = {
        "YES": XH_STRETCH_POLICY_SYMMETRIZE,
        "TRUE": XH_STRETCH_POLICY_SYMMETRIZE,
        "ALL": XH_STRETCH_POLICY_LOCAL_ALL,
        "LOCAL": XH_STRETCH_POLICY_LOCAL_ALL,
        "LOCAL_ALL": XH_STRETCH_POLICY_LOCAL_ALL,
        "SELECTED": XH_STRETCH_POLICY_LOCAL_SELECTED,
        "LOCAL_SELECTED": XH_STRETCH_POLICY_LOCAL_SELECTED,
        "NO": XH_STRETCH_POLICY_LOCAL_ALL,
        "FALSE": XH_STRETCH_POLICY_LOCAL_ALL,
    }
    normalized = aliases.get(text, text)
    if normalized not in XH_STRETCH_POLICIES:
        raise GICForgeContractError(
            "invalid X-H stretch policy: "
            f"{value!r}; expected SYMMETRIZE, LOCAL_ALL or LOCAL_SELECTED"
        )
    return normalized


def _normalize_pairs(raw_pairs: object | None) -> tuple[tuple[int, int], ...]:
    if raw_pairs is None:
        return ()
    pairs: list[tuple[int, int]] = []
    if isinstance(raw_pairs, str):
        items: object = re.split(r"[,;]\s*", raw_pairs.strip()) if raw_pairs.strip() else ()
    else:
        items = raw_pairs
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text or text.upper() in {"NA", "NONE"}:
                continue
            parts = re.split(r"[-:,/]", text)
            if len(parts) != 2:
                raise GICForgeContractError(f"invalid X-H bond selector: {item!r}")
            left, right = int(parts[0]), int(parts[1])
        else:
            values = tuple(item)
            if len(values) != 2:
                raise GICForgeContractError(f"invalid X-H bond selector: {item!r}")
            left, right = int(values[0]), int(values[1])
        pairs.append(_pair_key(left, right))
    return tuple(dict.fromkeys(pairs))


def _normalize_xh_classes(raw_classes: object | None) -> tuple[str, ...]:
    if raw_classes is None:
        return ()
    if isinstance(raw_classes, str):
        items: object = re.split(r"[,;]\s*", raw_classes.strip()) if raw_classes.strip() else ()
    else:
        items = raw_classes
    classes: list[str] = []
    aliases = {"XH1": "XH", "H1": "XH", "H2": "XH2", "H3": "XH3"}
    for item in items:
        text = str(item).strip().upper().replace("-", "")
        if not text or text in {"NA", "NONE"}:
            continue
        normalized = aliases.get(text, text)
        if normalized not in XH_STRETCH_CLASSES:
            raise GICForgeContractError(
                f"invalid X-H stretch class: {item!r}; expected XH, XH2 or XH3"
            )
        classes.append(normalized)
    return tuple(dict.fromkeys(classes))


def _smith_coordinate_ring_atoms(coordinate: object) -> tuple[int, ...]:
    atoms: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for _coefficient, primitive in getattr(coordinate, "terms", ()):
        if str(getattr(primitive, "kind", "")) != "dihedral":
            return ()
        term_atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms"))
        if len(term_atoms) != 4:
            return ()
        atoms.update(term_atoms)
        for left, right in zip(term_atoms, term_atoms[1:]):
            edges.add(tuple(sorted((left, right))))
    if len(atoms) < 4:
        return ()
    return _ordered_cycle_from_edges(tuple(sorted(atoms)), edges)


def _ordered_cycle_from_edges(
    atoms: tuple[int, ...],
    edges: set[tuple[int, int]],
) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = {atom: [] for atom in atoms}
    for left, right in edges:
        if left not in adjacency or right not in adjacency:
            continue
        adjacency[left].append(right)
        adjacency[right].append(left)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return tuple(sorted(atoms))
    start = min(atoms)
    candidates: list[tuple[int, ...]] = []
    for second in sorted(adjacency[start]):
        path = [start, second]
        previous = start
        current = second
        while len(path) < len(atoms):
            next_atoms = [atom for atom in adjacency[current] if atom != previous]
            if not next_atoms:
                break
            next_atom = next_atoms[0]
            if next_atom in path:
                break
            path.append(next_atom)
            previous, current = current, next_atom
        if len(path) == len(atoms) and start in adjacency[path[-1]]:
            candidates.append(tuple(path))
    if not candidates:
        return tuple(sorted(atoms))
    return min(candidates)


def _smith_coordinate_family(name: str, block: str) -> str:
    prefix = str(block or name[:4])
    if prefix == "Stre":
        return "STRETCH"
    if prefix in {
        "Rock",
        "Bend",
        "SymD",
        "Scis",
        "SciL",
        "Wagg",
        "Twst",
        "AsyD",
        "EEee",
        "T2xx",
        "T2yy",
        "T2zz",
        "B1GE",
        "EUU",
        "HCAn",
        "XAng",
    }:
        return "BEND"
    if prefix == "Spir":
        return "SPIRO_BEND"
    if prefix == "RDef":
        return "CYCLIC_BEND"
    if prefix == "LAng":
        return "LINEAR_BEND"
    if prefix == "BtFl":
        return "BUTTERFLY"
    if prefix == "RPck":
        return "RING_PUCKER_COMPONENT"
    if prefix in {"Tors", "Dihe"}:
        return "TORSION"
    if prefix == "OuPl":
        return "OUT_OF_PLANE"
    if prefix == "ImpD":
        return "IMPROPER_DIHEDRAL"
    return "TORSION"


def symmetrize_gic_definition(
    definition: GICDefinition,
    *,
    atom_symbols: tuple[str, ...],
    symmetry_operations: tuple[GICPointGroupOperation, ...] = (),
) -> GICDefinition:
    """Apply the frozen GIC symmetrization utility to a reduced definition."""
    # A reactive TS pseudobond is a chart edge prescribed by ORACLE, not an
    # intermolecular body decomposition.  Its selected local analytic SALCs
    # must therefore remain local.  The residual B-metric torsion eigenbasis
    # is retained only as the established safeguard for genuine weak-complex
    # minimum charts, where it was introduced.
    is_transition_state_chart = any(gic.family == "TS_REACTION_DISTANCE" for gic in definition.gics)
    is_intermolecular = (
        definition.fragment_mode != FRAGMENT_MODE_NONE and not is_transition_state_chart
    )
    has_cartesian_intercomponent_coordinates = any(
        primitive.family in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}
        for primitive in definition.primitives
    )
    closed_primitives = _symmetry_closed_projector_primitives(
        definition.primitives,
        symmetry_operations=tuple(symmetry_operations),
        include_cartesian_interfragment_orbits=(
            (is_intermolecular or has_cartesian_intercomponent_coordinates)
            and definition.point_group.strip().upper() not in {"C1", "UNKNOWN"}
        ),
    )
    gics, symmetry_diagnostics = _apply_local_symmetrization(
        definition.gics,
        closed_primitives,
        atom_symbols=tuple(atom_symbols),
        point_group=definition.point_group,
        requested=True,
        symmetry_operations=tuple(symmetry_operations),
        reference_coordinates_angstrom=definition.reference_coordinates_angstrom,
        intermolecular=(is_intermolecular or has_cartesian_intercomponent_coordinates),
        pseudobond_mode=(
            definition.fragment_mode == FRAGMENT_MODE_PSEUDO_BONDS and not is_transition_state_chart
        ),
    )
    result = GICDefinition(
        backend=definition.backend,
        point_group=definition.point_group,
        symmetrize=True,
        target_rank=definition.target_rank,
        rank=definition.rank,
        candidate_count=definition.candidate_count,
        reference_coordinates_angstrom=definition.reference_coordinates_angstrom,
        primitives=closed_primitives,
        gics=gics,
        reduction_diagnostics=definition.reduction_diagnostics,
        symmetry_diagnostics=symmetry_diagnostics,
        fragment_mode=definition.fragment_mode,
        pseudo_bonds=definition.pseudo_bonds,
        pseudo_bond_kinds=definition.pseudo_bond_kinds,
        xh_stretch_policy=definition.xh_stretch_policy,
        local_xh_bonds=definition.local_xh_bonds,
        local_xh_classes=definition.local_xh_classes,
        ring_puckering_diagnostics=definition.ring_puckering_diagnostics,
        contract_schema_version=definition.contract_schema_version,
        semantic_grammar_version=definition.semantic_grammar_version,
        semantic_diagnostics=definition.semantic_diagnostics,
        fallback_events=definition.fallback_events,
        primitive_source=definition.primitive_source,
        primitive_source_schema=definition.primitive_source_schema,
        primitive_b_matrix_sha256=definition.primitive_b_matrix_sha256,
    )
    validate_molecular_vibrational_irrep_counts(
        result,
        symmetry_operations=tuple(symmetry_operations),
    )
    return result


def build_sycart_definition_from_xyzin(path: Path) -> SYCartDefinition:
    """Build the shared SMITH-owned symmetry-adapted Cartesian basis.

    SYCART is the topology-independent recovery chart.  Its construction uses
    only the Cartesian geometry and molecular symmetry and must remain
    available when ORACLE deliberately marks the chemical-topology validation
    as failed.  SONIC construction retains the stricter prerequisite gate.
    """
    target = Path(path)
    geometry = read_enriched_xyz(target)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    symmetry = read_molecular_symmetry(target)
    basis = symmetry_adapted_cartesian_basis(
        geometry.atoms,
        coords,
        symmetry=symmetry,
        # MolecularSymmetry.orientation stores principal axes by columns;
        # OnicSiteFrame stores local axes by rows.
        frame_axes_global=np.asarray(symmetry.orientation, dtype=float).T,
    )
    return SYCartDefinition(
        backend=SYCART_BACKEND,
        point_group=basis.point_group,
        target_rank=basis.target_rank,
        vectors=tuple(tuple(float(value) for value in row) for row in basis.cartesian_from_q.T),
        irreps=basis.irreps,
        external_mode_count=basis.external_mode_count,
        linearity=basis.linearity,
        gauge_policy=CARTESIAN_BLOCK_GAUGE,
    )


def gic_definition_section_lines(definition: GICDefinition) -> list[str]:
    if not definition.primitive_source or definition.primitive_source == "LEGACY_RECONSTRUCTED":
        raise GICForgeContractError("GIC serialization requires an explicit primitive source")
    if definition.primitive_source == "ORACLE_CONTRACT" and (
        not definition.primitive_source_schema or not definition.primitive_b_matrix_sha256
    ):
        raise GICForgeContractError("ORACLE primitive source metadata is incomplete")
    primitive_dependency = (
        definition.primitive_source_schema
        if definition.primitive_source == "ORACLE_CONTRACT"
        else definition.primitive_source
    )
    fallback_events = build_fallback_ledger(definition)
    lines = [
        f"SCHEMA {definition.contract_schema_version}",
        f"CONTRACT_SCHEMA_VERSION {definition.contract_schema_version}",
        "STATUS BUILT",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} "
        f"TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA} "
        f"SYMMETRY=oracle.xyz.symmetry.v1 PRIMITIVES={primitive_dependency}",
        "OWNERSHIP ORACLE_PERCEPTION=ORACLE SMITH_CONSTRUCTION=SMITH "
        "GAUSSIAN_SERIALIZATION=GAUSSIAN_READALLGIC",
        "CONSTRUCTION_GEOMETRY_REFERENCE FROZEN_ORACLE_INPUT",
        f"PRIMITIVE_SOURCE {definition.primitive_source}",
        f"PRIMITIVE_SOURCE_SCHEMA {definition.primitive_source_schema or 'NONE'}",
        f"PRIMITIVE_B_MATRIX_SHA256 {definition.primitive_b_matrix_sha256 or 'NONE'}",
        f"FROZEN_SONIC_IDENTITY_SHA256 {sonic_definition_identity_sha256(definition)}",
        f"SEMANTIC_GRAMMAR {definition.semantic_grammar_version or 'NONE'}",
        f"FALLBACK_COUNT {len(fallback_events)}",
        "INDEXING ATOMS=ONE_BASED",
        f"BACKEND {definition.backend}",
        f"POINT_GROUP {definition.point_group}",
        f"SYMMETRY_GROUP {definition.point_group}",
        f"TOTAL_SYMMETRIC_IRREP {total_symmetric_irrep(definition.point_group)}",
        f"TOTAL_SYMMETRIC_GIC_COUNT {len(total_symmetric_gics(definition))}",
        f"TOTAL_SYMMETRIC_GICS {_csv_or_none(total_symmetric_gic_names(definition))}",
        f"SYMMETRIZE {_bool_text(definition.symmetrize)}",
        f"SYMMETRY_MODE {_symmetry_mode(definition)}",
        f"OUT_OF_PLANE_MODE {_out_of_plane_mode(definition.primitives)}",
        f"FRAGMENT_MODE {definition.fragment_mode}",
        f"XH_STRETCH_POLICY {definition.xh_stretch_policy}",
        f"LOCAL_XH_BONDS {_pairs_text(definition.local_xh_bonds)}",
        f"LOCAL_XH_CLASSES {_csv_or_none_from_strings(definition.local_xh_classes)}",
        f"PSEUDO_BOND_COUNT {len(definition.pseudo_bonds)}",
        f"TARGET_RANK {definition.target_rank}",
        f"RANK {definition.rank}",
        f"CANDIDATE_COUNT {definition.candidate_count}",
        f"PRIMITIVE_COUNT {len(definition.primitives)}",
        f"GIC_COUNT {len(definition.gics)}",
        f"WILSON_TANGENT_RANK {definition.wilson_tangent_rank}",
        f"WILSON_TANGENT_SINGULAR_MIN {definition.wilson_tangent_singular_min:.12g}",
        f"WILSON_TANGENT_SINGULAR_MAX {definition.wilson_tangent_singular_max:.12g}",
        f"PERIODIC_COORDINATE_ESTIMATE_COUNT {len(definition.periodic_coordinate_estimates)}",
        f"PROTECTED_GIC_COUNT {_protected_gic_count(definition.primitives)}",
        f"SKIPPED_SINGULAR_COUNT {_skipped_singular_count(definition)}",
        f"SKIPPED_DEPENDENT_COUNT {_skipped_dependent_count(definition)}",
        f"RANK_METHOD {RANK_METHOD}",
        f"REDUCTION_POLICY {REDUCTION_POLICY}",
        "B_MATRIX_DERIVATIVE_MODE ANALYTIC",
        f"RANK_TOLERANCE {RANK_TOLERANCE:.12g}",
        "[PRIMITIVES]",
    ]
    if definition.primitives:
        lines.extend(_primitive_line(primitive) for primitive in definition.primitives)
    else:
        lines.append("NONE")
    lines.append("[FROZEN_GICS]")
    if definition.gics:
        lines.extend(_frozen_gic_line(gic) for gic in definition.gics)
    else:
        lines.append("NONE")
    lines.append("[REDUCTION_DIAGNOSTICS]")
    lines.extend(_reduction_diagnostics_lines(definition))
    lines.append("[SYMMETRY_DIAGNOSTICS]")
    lines.extend(_symmetry_diagnostics_lines(definition))
    lines.append("[RING_PUCKERING_DIAGNOSTICS]")
    lines.extend(definition.ring_puckering_diagnostics or ("NONE",))
    lines.append("[SEMANTIC_DIAGNOSTICS]")
    lines.extend(definition.semantic_diagnostics or ("NONE",))
    lines.extend(fallback_ledger_section_lines(fallback_events))
    lines.append("[PSEUDO_BONDS]")
    if definition.pseudo_bonds:
        kinds = definition.pseudo_bond_kinds or (
            ("INTERFRAGMENT_CLOSEST",) * len(definition.pseudo_bonds)
        )
        if len(kinds) != len(definition.pseudo_bonds):
            kinds = ("INTERFRAGMENT_CLOSEST",) * len(definition.pseudo_bonds)
        for index, ((left, right), kind) in enumerate(
            zip(definition.pseudo_bonds, kinds),
            start=1,
        ):
            lines.append(f"{index} {left} {right} KIND={kind}")
    else:
        lines.append("NONE")
    lines.append("[GAUSSIAN_GIC]")
    gaussian_gics = _gaussian_gic_block_lines(definition)
    if gaussian_gics:
        lines.extend(gaussian_gics)
    else:
        lines.append("NONE")
    lines.append("[PERIODIC_COORDINATE_ESTIMATES]")
    if definition.periodic_coordinate_estimates:
        lines.extend(
            periodic_coordinate_estimate_line(record)
            for record in definition.periodic_coordinate_estimates
        )
    else:
        lines.append("NONE")
    return lines


def sycart_definition_section_lines(definition: SYCartDefinition) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_SYCART_SCHEMA}",
        "STATUS BUILT",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} GIC=oracle.xyz.gic.v1 "
        "SYMMETRY=oracle.xyz.symmetry.v1",
        "INDEXING ATOMS=ONE_BASED",
        f"BACKEND {definition.backend}",
        f"POINT_GROUP {definition.point_group}",
        "SYMMETRY_MODE COMPLETE_ISOTYPIC_PROJECTORS",
        f"EXTERNAL_MODE_COUNT {definition.external_mode_count}",
        f"LINEARITY {definition.linearity or 'UNDECLARED'}",
        f"COMPONENT_GAUGE {definition.gauge_policy or 'UNDECLARED'}",
        f"TARGET_RANK {definition.target_rank}",
        f"COORD_COUNT {len(definition.vectors)}",
        "[SYCART]",
    ]
    if definition.vectors:
        irreps = definition.irreps or ("A",) * len(definition.vectors)
        lines.extend(
            f"SYC{idx:03d} IRREP={irrep} " + _sycart_components(vector)
            for idx, (vector, irrep) in enumerate(zip(definition.vectors, irreps), start=1)
        )
    else:
        lines.append("NONE")
    return lines


def write_gicforge_build_sections(
    path: Path,
    *,
    symmetrize: bool = False,
    sycart: bool = False,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
    out_of_plane_model: str = "soop",
) -> GICDefinition:
    target = Path(path)
    # Disconnected molecular components are an intermolecular system even when
    # the caller did not explicitly materialize #FRAGMENTS.  The SONIC API
    # must not silently fall back to one local coordinate set per component:
    # build the shared fragment contract automatically, while leaving
    # covalently connected molecules unchanged.
    if fragment_mode is None or _fragment_mode(fragment_mode) != FRAGMENT_MODE_NONE:
        build_lines = read_sectioned_lines(target)
        if not section_content(build_lines, "FRAGMENTS"):
            from matrix_fragments import (
                build_fragment_definition_from_xyzin,
                write_fragment_build_section,
            )

            fragment_definition = build_fragment_definition_from_xyzin(target)
            if len(fragment_definition.fragments) > 1:
                write_fragment_build_section(target)
    if not section_content(
        read_sectioned_lines(target),
        "ORACLE_COORDINATE_ATLAS",
    ):
        context = _fragment_context(fragment_context)
        if context == _FRAGMENT_CONTEXT_TRANSITION_STATE:
            raise GICForgeContractError(
                "a transition-state build requires a valid ORACLE TS coordinate atlas"
            )
        from matrix_oracle import write_oracle_sonic_contract_from_xyzin

        minimum_fragment_representation = "automatic"
        if context == _FRAGMENT_CONTEXT_MINIMUM and fragment_mode is not None:
            requested_fragment_mode = _fragment_mode(fragment_mode)
            if requested_fragment_mode == FRAGMENT_MODE_PSEUDO_BONDS:
                minimum_fragment_representation = "pseudo-bonds"
            elif requested_fragment_mode == FRAGMENT_MODE_SPECIAL_COORDINATES:
                minimum_fragment_representation = "quaternion"
        try:
            write_oracle_sonic_contract_from_xyzin(
                target,
                minimum_fragment_representation=minimum_fragment_representation,
            )
        except ValueError as exc:
            if context == _FRAGMENT_CONTEXT_MINIMUM and fragment_mode is not None:
                raise GICForgeContractError(
                    f"fragment_mode cannot override the ORACLE minimum coordinate atlas: {exc}"
                ) from exc
            raise
        from matrix_oracle import (
            write_exploration_coordinate_atlas_contract,
        )

        if context == _FRAGMENT_CONTEXT_EXPLORATION:
            write_exploration_coordinate_atlas_contract(target)
    definition = build_gic_definition_from_xyzin(
        target,
        symmetrize=symmetrize,
        symmetry_group=symmetry_group,
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        fragment_context=fragment_context,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        ring_puckering_model=ring_puckering_model,
        validate_sonic_basis=not sycart,
    )
    sonic_contract = read_oracle_sonic_contract(target)
    oop_model = str(out_of_plane_model).strip().lower()
    if oop_model != "soop":
        raise ValueError("out_of_plane_model must be soop; legacy out-of-plane is unsupported")
    from .soop_definition import materialize_soop_definition

    definition = materialize_soop_definition(
        definition,
        sonic_contract,
        symmetry_operations=read_molecular_symmetry(target).operations,
    )
    if any(
        contact.open_or_closing == "CLOSING" and len(set(contact.fragment_ids)) > 1
        for contact in sonic_contract.auxiliary_contacts
    ):
        closed_contact_policy = (
            "PSEUDOBOND_CONTACT_NATURAL_COORDINATES"
            if definition.fragment_mode == FRAGMENT_MODE_PSEUDO_BONDS
            else "SPECIAL_COORDINATES"
        )
        diagnostics = tuple(
            record
            for record in definition.semantic_diagnostics
            if not record.startswith("CLOSED_CONTACT_POLICY ")
        )
        definition = replace(
            definition,
            semantic_diagnostics=(
                *diagnostics,
                f"CLOSED_CONTACT_POLICY {closed_contact_policy}",
            ),
        )
    replace_section(target, "GIC", gic_definition_section_lines(definition))
    if sycart:
        sycart_definition = build_sycart_definition_from_xyzin(target)
        replace_section(target, "SYCART", sycart_definition_section_lines(sycart_definition))
    return definition


def write_sycart_build_section(path: Path) -> SYCartDefinition:
    """Build and freeze SYCART without requiring a valid SONIC chart.

    Cartesian optimization is the recovery path when the local internal
    topology cannot be certified.  It must therefore depend only on the
    ORACLE geometry/symmetry contract, never on successful GIC construction.
    """

    target = Path(path)
    definition = build_sycart_definition_from_xyzin(target)
    replace_section(target, "SYCART", sycart_definition_section_lines(definition))
    return definition


def freeze_sonic_definition_from_xyzin(
    path: Path,
    **kwargs: object,
) -> GICDefinition:
    """Read the existing frozen SONIC identity or build it exactly once."""

    target = Path(path)
    if section_content(read_sectioned_lines(target), "GIC"):
        try:
            return read_gic_definition_from_xyzin(target)
        except GICForgeContractError as exc:
            if "legacy out-of-plane primitives" not in str(exc):
                raise
            # Historical U/IMPD charts are input material only.  Rebuild the
            # persisted chart through the current ORACLE -> SMITH contract so
            # every active consumer receives SOOP.
            return write_gicforge_build_sections(target, **kwargs)
    return write_gicforge_build_sections(target, **kwargs)


def sonic_definition_identity_sha256(definition: GICDefinition) -> str:
    """Return the geometry-independent identity of a frozen SONIC chart."""

    payload = {
        "primitives": [
            {
                "identifier": item.identifier,
                "family": item.family,
                "function": item.function,
                "atoms": list(item.atoms),
                "refs": list(item.refs),
            }
            for item in definition.primitives
        ],
        "gics": [
            {
                "identifier": item.identifier,
                "family": item.family,
                "primitive_id": item.primitive_id,
                "coefficients": [
                    [identifier, f"{float(coefficient):.12g}"]
                    for identifier, coefficient in item.coefficients
                ],
            }
            for item in definition.gics
        ],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_pes_exploration_gic_definition_from_xyzin(
    path: Path,
    *,
    retained_group: str = "C1",
    fragment_mode: str | None = None,
) -> GICDefinition:
    """Build the transient SONIC contract shared by scans and PES exploration.

    The retained group is the symmetry common to the complete path, not the
    point group perceived at the reference geometry.  C1 is therefore the
    conservative default.  Local SALCs remain available for stretches, bends
    and rings, while every exocyclic torsion is forced to the one-dihedral form
    so that a scan variable is never hidden inside a torsional combination.
    """

    target = Path(path)
    retained = str(retained_group).strip().upper() or "C1"
    base = read_gic_definition_from_xyzin(target)
    # Exploration is a distinct ORACLE task regime.  A structure normally
    # reaches this entry point with the minimum atlas that was used to build
    # its frozen local chart; reusing that prescription would make the result
    # depend on which operation happened to run first.  Ask ORACLE to author
    # the exploration atlas explicitly before SMITH constructs the transient
    # path chart.  SMITH still consumes the resulting contract and makes no
    # topology or coordinate-policy decision of its own.
    from matrix_oracle import write_exploration_coordinate_atlas_contract

    write_exploration_coordinate_atlas_contract(target)
    diagnostics = base.reduction_diagnostics
    diagnostic_rows = ()
    if diagnostics is not None:
        diagnostic_rows = (
            diagnostics.selected_by_family
            + diagnostics.skipped_singular_details
            + diagnostics.skipped_dependent_details
        )
    uses_local_salc = any("LOCAL_SALC" in row.upper() for row in diagnostic_rows)
    return build_gic_definition_from_xyzin(
        target,
        symmetrize=retained not in {"C1", "UNKNOWN"},
        symmetry_group=retained,
        improper_dihedrals=False,
        fragment_mode=fragment_mode,
        fragment_context=_FRAGMENT_CONTEXT_EXPLORATION,
        xh_stretch_policy=base.xh_stretch_policy,
        local_xh_bonds=base.local_xh_bonds,
        local_xh_classes=base.local_xh_classes,
        local_salc=uses_local_salc,
        separate_exocyclic_torsions=True,
    )


def build_sonic_definition_from_xyzin(
    path: Path,
    **kwargs: object,
) -> GICDefinition:
    """Build the public SONIC definition, using SOOP out-of-plane blocks."""
    out_of_plane_model = str(kwargs.pop("out_of_plane_model", "soop")).casefold()
    definition = build_gic_definition_from_xyzin(path, **kwargs)
    if out_of_plane_model != "soop":
        raise ValueError("out_of_plane_model must be 'soop'")
    from matrix_chem import read_oracle_sonic_contract
    from .soop_definition import materialize_soop_definition

    return materialize_soop_definition(
        definition,
        read_oracle_sonic_contract(path),
        symmetry_operations=read_molecular_symmetry(path).operations,
    )


def write_sonic_build_sections(
    path: Path,
    **kwargs: object,
) -> GICDefinition:
    """Public SONIC alias for writing the frozen GIC and optional SYCART sections."""
    kwargs.setdefault("out_of_plane_model", "soop")
    return write_gicforge_build_sections(path, **kwargs)


def write_sonic_build_sections_from_cartesian(
    source: Path,
    target: Path | None = None,
    *,
    source_kind: str = "auto",
    symmetrize: bool = False,
    sycart: bool = False,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
) -> GICDefinition:
    """Build a standalone SONIC contract from Cartesian input.

    If the input already contains MATRIX topology, synthon, symmetry and
    validation sections, they are consumed as the frozen molecular state.  A
    plain Cartesian file is first enriched with the bundled MATRIX perception
    tools and then passed to the same coordinate builder.
    """
    source_path = Path(source)
    build_path = Path(target) if target is not None else source_path
    lines = read_sectioned_lines(source_path)
    required = ("VALIDATION", "TOPOLOGY", "SYNTHONS", "SYMMETRY")
    if not all(section_content(lines, name) for name in required):
        if target is None:
            build_path = source_path.with_suffix(source_path.suffix + ".xyzin")
        preprocess_to_enriched_xyz(source_path, build_path, source_kind=source_kind)  # type: ignore[arg-type]
    elif build_path != source_path:
        build_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    build_lines = read_sectioned_lines(build_path)
    if not section_content(build_lines, "FRAGMENTS"):
        from matrix_fragments import write_fragment_build_section

        write_fragment_build_section(build_path)
    write_validation_section(build_path)
    return write_gicforge_build_sections(
        build_path,
        symmetrize=symmetrize,
        sycart=sycart,
        symmetry_group=symmetry_group,
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        fragment_context=fragment_context,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        out_of_plane_model="soop",
    )


build_gsnic_definition_from_xyzin = build_sonic_definition_from_xyzin
write_gsnic_build_sections = write_sonic_build_sections
write_gsnic_build_sections_from_cartesian = write_sonic_build_sections_from_cartesian


from .evaluation import (
    build_gic_b_matrix,
)


def retained_sonic_definition(
    definition: GICDefinition,
    *,
    retained_group: str,
    closed_contact_policy: str | None = None,
) -> GICDefinition:
    """Change path-level symmetry metadata without changing a SONIC chart.

    Primitive identities, GIC coefficients, reference geometry and periodic
    references remain untouched.  This is the common retained-group view used
    by an exploration path whose endpoint geometries may have higher symmetry.
    """

    retained = str(retained_group).strip().upper() or "C1"
    retained_irrep = total_symmetric_irrep(retained)
    result = replace(
        definition,
        point_group=retained,
        symmetrize=retained not in {"C1", "UNKNOWN"},
        gics=tuple(replace(gic, irrep=retained_irrep) for gic in definition.gics),
        symmetry_diagnostics=None,
    )
    if closed_contact_policy is not None:
        policy = str(closed_contact_policy).strip().upper()
        expected_mode = {
            "SPECIAL_COORDINATES": FRAGMENT_MODE_SPECIAL_COORDINATES,
            "PSEUDOBOND_CONTACT_NATURAL_COORDINATES": FRAGMENT_MODE_PSEUDO_BONDS,
        }.get(policy)
        if expected_mode is None:
            raise GICForgeContractError(f"unsupported closed-contact policy: {policy}")
        if result.fragment_mode != expected_mode:
            raise GICForgeContractError(
                "closed-contact policy contradicts the frozen fragment chart"
            )
        result = replace(
            result,
            semantic_diagnostics=tuple(
                record
                for record in result.semantic_diagnostics
                if not record.startswith("CLOSED_CONTACT_POLICY ")
            )
            + (f"CLOSED_CONTACT_POLICY {policy}",),
        )
    if sonic_definition_identity_sha256(result) != sonic_definition_identity_sha256(definition):
        raise GICForgeContractError("retained-group view changed SONIC coordinate identity")
    return result


def recenter_sonic_definition_from_xyzin(
    path: Path,
    definition: GICDefinition,
    *,
    retained_group: str | None = None,
    closed_contact_policy: str | None = None,
    preserve_primitive_source: bool = False,
) -> GICDefinition:
    """Recenter one frozen SONIC identity on a new ORACLE reference geometry.

    The primitive/GIC identities, families, coefficients and endocyclic/
    exocyclic classification are preserved verbatim.  Only the reference
    Cartesian geometry, geometry-dependent primitive-B provenance and periodic
    reference values are refreshed after ORACLE has rewritten its sections.
    A supplied retained group changes only symmetry metadata and GIC irrep
    labels; it is used by exploration paths whose common group is lower than
    either endpoint group.  A supplied closed-contact policy records how that
    same frozen chart crosses a contact-closing region without changing its
    adjacency or coordinate identity.
    """

    target = Path(path)
    geometry = read_enriched_xyz(target)
    contract = read_primitive_contract(target)
    if len(definition.reference_coordinates_angstrom) != geometry.natoms:
        raise GICForgeContractError(
            "cannot recenter a frozen SONIC definition on a different atom count"
        )
    source_updates = (
        {}
        if preserve_primitive_source
        else {
            "primitive_source": "ORACLE_CONTRACT",
            "primitive_source_schema": contract.schema,
            "primitive_b_matrix_sha256": contract.b_matrix_sha256,
        }
    )
    recentered = replace(
        definition,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in geometry.coordinates_angstrom
        ),
        **source_updates,
    )
    if retained_group is not None:
        recentered = retained_sonic_definition(
            recentered,
            retained_group=retained_group,
            closed_contact_policy=closed_contact_policy,
        )
    elif closed_contact_policy is not None:
        raise GICForgeContractError("a closed-contact policy requires an explicit retained group")
    periodic_values = _selected_gic_values(
        recentered,
        families={"TORSION", "PSEUDO_CYCLE_TORSION", "IMPROPER_DIHEDRAL"},
    )
    recentered = replace(
        recentered,
        periodic_coordinate_estimates=tuple(
            replace(
                estimate,
                reference_value_radian=periodic_values.get(
                    estimate.coordinate_identifier,
                    estimate.reference_value_radian,
                ),
            )
            for estimate in recentered.periodic_coordinate_estimates
        ),
    )
    if sonic_definition_identity_sha256(recentered) != sonic_definition_identity_sha256(definition):
        raise GICForgeContractError("recentring changed the frozen SONIC coordinate identity")
    replace_section(target, "GIC", gic_definition_section_lines(recentered))
    return recentered


def build_gic_b_matrix_from_xyzin(path: Path) -> GICBMatrix:
    """Evaluate the B matrix from the frozen #GIC section of an enriched XYZ."""
    target = Path(path)
    definition = read_gic_definition_from_xyzin(target)
    geometry = read_enriched_xyz(target)
    return build_gic_b_matrix(
        definition,
        coordinates_angstrom=geometry.coordinates_angstrom,
    )


def total_symmetric_gics(definition: GICDefinition) -> tuple[FrozenGIC, ...]:
    """Return frozen GICs active in symmetry-preserving optimization/fitting."""
    return tuple(
        gic
        for gic in definition.gics
        if is_total_symmetric_irrep(definition.point_group, gic.irrep)
    )


def total_symmetric_gic_names(definition: GICDefinition) -> tuple[str, ...]:
    return tuple(gic.name for gic in total_symmetric_gics(definition))


from .gaussian_export import (
    _gaussian_gic_block_lines,
    _ring_ref_text,
)

from .primitive_generation import (
    _apply_semantic_contract_to_candidates,
    _freeze_balanced_out_of_plane_coordinates,
    _fragment_records,
    _fragment_index_by_atom,
    _interaction_center_definition,
    _normalize_pseudo_contact,
    _primitive_candidates,
    _ring_puckering_diagnostics,
    _vibrational_rank,
)


from .definition_serialization import (
    read_gic_definition_from_xyzin,
    _csv_or_none,
    _planned_xh_stretch_policy,
    _planned_local_xh_bonds,
    _planned_local_xh_classes,
    _fragment_mode,
    _out_of_plane_mode,
    _pairs_text,
    _csv_or_none_from_strings,
    _primitive_line,
    _frozen_gic_line,
    _sycart_components,
    _symmetry_mode,
    _bool_text,
)
from .definition_serialization import (
    _cartesian_column_labels as _cartesian_column_labels,
    _parse_bool as _parse_bool,
    _parse_coefficients as _parse_coefficients,
    _parse_float_value as _parse_float_value,
    _parse_frozen_gic_line as _parse_frozen_gic_line,
    _parse_int as _parse_int,
    _parse_primitive_line as _parse_primitive_line,
    _parse_pseudo_bond_kinds as _parse_pseudo_bond_kinds,
    _parse_pseudo_bonds as _parse_pseudo_bonds,
    _parse_reduction_diagnostics as _parse_reduction_diagnostics,
    _parse_symmetry_diagnostics as _parse_symmetry_diagnostics,
    _parse_text_list as _parse_text_list,
    _section_value as _section_value,
    gic_b_matrix_lines as gic_b_matrix_lines,
    write_gic_b_matrix as write_gic_b_matrix,
)
from .audits.transition_state import (
    _wilson_tangent_diagnostics,
    _condition_transition_state_salc_alternatives,
    _select_rank_aware_chart,
    _select_conserved_reactive_exact_chart,
    _with_prescribed_distance_only_primitives,
    _with_transition_state_reaction_distance_family,
)

from .chart_freezing import (
    _with_periodic_coordinate_estimates,
    _selected_gic_values,
    _attach_chart_atlas,
)
