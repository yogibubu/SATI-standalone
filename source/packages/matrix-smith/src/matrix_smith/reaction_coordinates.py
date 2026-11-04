"""Combine two endpoint SONIC charts into one atlas-driven TS chart."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.linalg import qr

from matrix_chem import atomic_mass
from matrix_chem.topology.elements import atomic_number

from .coordinate_registry import (
    FRAGMENT_BODY_PAIR_FUNCTIONS,
    FRAGMENT_BODY_SINGLE_FUNCTIONS,
)
from .contracts import GICForgeContractError
from .evaluation import build_gic_b_matrix, evaluate_gic_values
from .models import FrozenGIC, GICDefinition, GICPrimitive
from .policy import (
    TS_REACTIVE_COORDINATE_CLASS,
    TS_REACTIVE_FAMILY_MIXING_POLICY,
    TS_REACTIVE_FAMILY_PRESERVING_POLICY,
)


ENDPOINT_SONIC_COMBINATION_SCHEMA = "matrix.smith.endpoint_sonic_combination.v1"
ENDPOINT_CONNECTIVITY_SONIC_SCHEMA = "matrix.smith.endpoint_connectivity_sonic_combination.v1"
HESSIAN_ADAPTED_REACTIVE_SONIC_SCHEMA = "matrix.smith.hessian_adapted_reactive_sonic.v1"
MULTITOPOLOGY_REACTIVE_SONIC_POOL_SCHEMA = (
    "matrix.smith.multitopology_reactive_sonic_pool.v1"
)
ENDPOINT_CONNECTIVITY_TS_SELECTION_SCHEMA = "matrix.smith.endpoint_connectivity_ts_selection.v1"
MAXIMUM_REACTION_PROGRESS_CONDITION_PENALTY = 0.05
MINIMUM_STABILITY_ADJUSTED_REACTION_MODE_GAIN = 0.20
MINIMUM_REACTION_ROW_FALLBACK_CURVATURE_REDUCTION = 0.50
MINIMUM_REACTION_ROW_FALLBACK_PARTICIPATION_GAIN = 0.50
MAXIMUM_REACTION_ROW_FALLBACK_CONDITION = 10.0
REACTION_MODE_CURVATURE_PROBE_ANGSTROM = 0.05


@dataclass(frozen=True)
class EndpointSonicCombination:
    """Auditable composition of common and endpoint-reactive SONICs."""

    schema: str
    definition: GICDefinition
    common_count: int
    start_reactive_gics: tuple[str, ...]
    end_reactive_gics: tuple[str, ...]
    source_labels: tuple[str, ...]
    source_irreps: tuple[str, ...]
    transformation: tuple[tuple[float, ...], ...]
    endpoint_displacement: tuple[float, ...]
    rank_start: int
    rank_end: int
    minimum_singular_value_start: float
    minimum_singular_value_end: float


@dataclass(frozen=True)
class EndpointConnectivitySonicCombination:
    """Exact-rank TS chart built from two endpoint connectivity pools."""

    schema: str
    definition: GICDefinition
    reactive_pool_definition: GICDefinition
    common_count: int
    reactive_source_labels: tuple[str, ...]
    transformation: tuple[tuple[float, ...], ...]
    minimum_singular_value: float
    connectivity_change: bool
    reactive_family_mixing_allowed: bool


@dataclass(frozen=True)
class MultitopologyReactiveSonicPool:
    """Redundant reactive pool with provenance across arbitrary topologies."""

    schema: str
    definition: GICDefinition
    common_count: int
    topology_ids: tuple[str, ...]
    reactive_source_labels: tuple[str, ...]
    reactive_source_topology_memberships: tuple[tuple[str, ...], ...]
    reactive_rank: int
    projected_source_rank: int
    projected_source_condition: float


@dataclass(frozen=True)
class EndpointConnectivityTSSelection:
    """Hessian-aware choice between standard and endpoint-informed TS charts."""

    schema: str
    definition: GICDefinition
    selected_chart: str
    reason: str
    endpoint_reaction_mode_alignment: float | None
    standard_best_mode_alignment: float
    endpoint_reaction_mode_curvature: float | None
    standard_best_mode_curvature: float
    endpoint_stability_adjusted_alignment: float | None
    standard_stability_adjusted_alignment: float
    endpoint_mode_weighted_curvature: float | None
    standard_mode_weighted_curvature: float
    endpoint_mode_participation: float | None
    standard_mode_participation: float
    endpoint_wilson_condition: float | None
    standard_wilson_condition: float


def select_endpoint_connectivity_ts_definition(
    standard: GICDefinition,
    endpoint: GICDefinition,
    atom_symbols: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    cartesian_hessian: np.ndarray,
    *,
    transition_mode: int = 0,
    minimum_stability_adjusted_gain: float = (MINIMUM_STABILITY_ADJUSTED_REACTION_MODE_GAIN),
    curvature_probe_angstrom: float = REACTION_MODE_CURVATURE_PROBE_ANGSTROM,
) -> EndpointConnectivityTSSelection:
    """Choose an endpoint-informed exact-rank chart only when it is locally safer.

    An explicit connectivity-reaction row is compared with the best individual
    row of the ordinary TS SONIC chart using the selected mass-weighted Hessian
    mode.  It is retained only when its alignment discounted by local
    coordinate curvature improves materially.  If rank selection omits that
    row, the endpoint chart is retained only when it both distributes the mode
    over materially more coordinates and halves its mode-weighted curvature,
    with both alternatives inside a conservative Wilson-condition gate.
    Otherwise the ordinary chart is returned unchanged.  The Hessian must be
    expressed in the Cartesian frame of
    ``reference_coordinates_angstrom``.
    """

    coordinates = np.asarray(reference_coordinates_angstrom, dtype=float)
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    if coordinates.shape != (len(symbols), 3):
        raise GICForgeContractError("TS reference geometry and atom ordering are incompatible")
    size = coordinates.size
    hessian = np.asarray(cartesian_hessian, dtype=float)
    if hessian.shape != (size, size) or not np.all(np.isfinite(hessian)):
        raise GICForgeContractError("Cartesian Hessian must be finite with shape (3N, 3N)")
    gain = float(minimum_stability_adjusted_gain)
    probe = float(curvature_probe_angstrom)
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("minimum stability-adjusted gain must be finite and nonnegative")
    if not np.isfinite(probe) or probe <= 0.0:
        raise ValueError("curvature probe must be finite and positive")

    masses = np.asarray([atomic_mass(_required_atomic_number(item)) for item in symbols])
    cartesian_masses = np.repeat(masses, 3)
    inverse_sqrt_mass = 1.0 / np.sqrt(cartesian_masses)
    weighted_hessian = hessian * inverse_sqrt_mass[:, None] * inverse_sqrt_mass[None, :]
    _eigenvalues, eigenvectors = np.linalg.eigh(weighted_hessian)
    mode_index = int(transition_mode)
    if mode_index < 0 or mode_index >= eigenvectors.shape[1]:
        raise ValueError("transition mode index is outside the Cartesian Hessian")
    mass_weighted_mode = eigenvectors[:, mode_index]
    cartesian_mode = (mass_weighted_mode * inverse_sqrt_mass).reshape(coordinates.shape)
    maximum_atom_norm = float(np.max(np.linalg.norm(cartesian_mode, axis=1)))
    if not np.isfinite(maximum_atom_norm) or maximum_atom_norm <= 0.0:
        raise GICForgeContractError("selected Hessian mode has zero Cartesian norm")
    displacement = cartesian_mode * (probe / maximum_atom_norm)

    standard_rows = _mass_weighted_definition_rows(standard, coordinates, inverse_sqrt_mass)
    standard_alignments = _row_mode_alignments(standard_rows, mass_weighted_mode)
    standard_condition = float(np.linalg.cond(standard_rows))
    standard_index = int(np.argmax(standard_alignments))
    standard_alignment = float(standard_alignments[standard_index])
    standard_curvature = _definition_row_curvature(
        standard,
        coordinates,
        displacement,
        standard_index,
        probe,
    )
    standard_score = standard_alignment / (1.0 + standard_curvature)
    standard_curvatures = _definition_mode_curvatures(
        standard,
        coordinates,
        displacement,
        probe,
    )
    standard_weighted_curvature = _mode_weighted_curvature(
        standard_alignments,
        standard_curvatures,
    )
    standard_participation = _mode_projection_participation(standard_alignments)
    if standard.target_rank != endpoint.target_rank:
        return EndpointConnectivityTSSelection(
            schema=ENDPOINT_CONNECTIVITY_TS_SELECTION_SCHEMA,
            definition=standard,
            selected_chart="STANDARD_TS_SONIC",
            reason="TARGET_RANK_MISMATCH",
            endpoint_reaction_mode_alignment=None,
            standard_best_mode_alignment=standard_alignment,
            endpoint_reaction_mode_curvature=None,
            standard_best_mode_curvature=standard_curvature,
            endpoint_stability_adjusted_alignment=None,
            standard_stability_adjusted_alignment=standard_score,
            endpoint_mode_weighted_curvature=None,
            standard_mode_weighted_curvature=standard_weighted_curvature,
            endpoint_mode_participation=None,
            standard_mode_participation=standard_participation,
            endpoint_wilson_condition=None,
            standard_wilson_condition=standard_condition,
        )
    endpoint_rows = _mass_weighted_definition_rows(endpoint, coordinates, inverse_sqrt_mass)
    endpoint_condition = float(np.linalg.cond(endpoint_rows))
    endpoint_reaction_indices = tuple(
        index for index, gic in enumerate(endpoint.gics) if gic.family == "TS_REACTION_DISTANCE"
    )
    endpoint_alignment: float | None = None
    endpoint_curvature: float | None = None
    endpoint_weighted_curvature: float | None = None
    endpoint_participation: float | None = None
    selected = "STANDARD_TS_SONIC"
    reason = "ENDPOINT_REACTION_PROGRESS_ABSENT"
    if len(endpoint_reaction_indices) == 1:
        endpoint_index = endpoint_reaction_indices[0]
        endpoint_alignment = float(
            _row_mode_alignments(
                endpoint_rows[np.asarray((endpoint_index,), dtype=int)],
                mass_weighted_mode,
            )[0]
        )
        endpoint_curvature = _definition_row_curvature(
            endpoint,
            coordinates,
            displacement,
            endpoint_index,
            probe,
        )
        endpoint_score = endpoint_alignment / (1.0 + endpoint_curvature)
        if endpoint_score <= standard_score * (1.0 + gain):
            reason = "NO_MATERIAL_STABILITY_ADJUSTED_ALIGNMENT_GAIN"
        else:
            selected = "ENDPOINT_CONNECTIVITY_SONIC"
            reason = "STABILITY_ADJUSTED_ALIGNMENT_GAIN"
    elif len(endpoint_reaction_indices) > 1:
        reason = "ENDPOINT_REACTION_PROGRESS_NOT_UNIQUE"
    else:
        endpoint_alignments = _row_mode_alignments(endpoint_rows, mass_weighted_mode)
        endpoint_index = int(np.argmax(endpoint_alignments))
        endpoint_alignment = float(endpoint_alignments[endpoint_index])
        endpoint_curvature = _definition_row_curvature(
            endpoint,
            coordinates,
            displacement,
            endpoint_index,
            probe,
        )
        endpoint_score = endpoint_alignment / (1.0 + endpoint_curvature)
        endpoint_curvatures = _definition_mode_curvatures(
            endpoint,
            coordinates,
            displacement,
            probe,
        )
        endpoint_weighted_curvature = _mode_weighted_curvature(
            endpoint_alignments,
            endpoint_curvatures,
        )
        endpoint_participation = _mode_projection_participation(endpoint_alignments)
        well_conditioned = max(standard_condition, endpoint_condition) <= (
            MAXIMUM_REACTION_ROW_FALLBACK_CONDITION
        )
        curvature_gain = endpoint_weighted_curvature <= standard_weighted_curvature * (
            1.0 - MINIMUM_REACTION_ROW_FALLBACK_CURVATURE_REDUCTION
        )
        participation_gain = endpoint_participation >= standard_participation * (
            1.0 + MINIMUM_REACTION_ROW_FALLBACK_PARTICIPATION_GAIN
        )
        if well_conditioned and curvature_gain and participation_gain:
            selected = "ENDPOINT_CONNECTIVITY_SONIC"
            reason = "DISTRIBUTED_LOW_CURVATURE_REACTION_MODE_GAIN"
        else:
            reason = "NO_MATERIAL_DISTRIBUTED_MODE_GAIN"

    definition = endpoint if selected == "ENDPOINT_CONNECTIVITY_SONIC" else standard
    return EndpointConnectivityTSSelection(
        schema=ENDPOINT_CONNECTIVITY_TS_SELECTION_SCHEMA,
        definition=definition,
        selected_chart=selected,
        reason=reason,
        endpoint_reaction_mode_alignment=endpoint_alignment,
        standard_best_mode_alignment=standard_alignment,
        endpoint_reaction_mode_curvature=endpoint_curvature,
        standard_best_mode_curvature=standard_curvature,
        endpoint_stability_adjusted_alignment=(
            None
            if endpoint_alignment is None or endpoint_curvature is None
            else endpoint_alignment / (1.0 + endpoint_curvature)
        ),
        standard_stability_adjusted_alignment=standard_score,
        endpoint_mode_weighted_curvature=endpoint_weighted_curvature,
        standard_mode_weighted_curvature=standard_weighted_curvature,
        endpoint_mode_participation=endpoint_participation,
        standard_mode_participation=standard_participation,
        endpoint_wilson_condition=endpoint_condition,
        standard_wilson_condition=standard_condition,
    )


def combine_endpoint_connectivity_sonic_definitions(
    start: GICDefinition,
    end: GICDefinition,
    atom_symbols: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    *,
    reaction_class_id: str,
    rank_tolerance: float = 1.0e-9,
) -> EndpointConnectivitySonicCombination:
    """Select a TS SONIC basis from the union of two endpoint primitive pools.

    Endpoint geometries are intentionally not used.  The endpoints supply only
    connectivity-dependent primitive definitions, so approximate structures or
    graph-derived endpoint charts are sufficient.  Coordinates common to both
    endpoint SONIC definitions are retained verbatim only when their primitive
    support does not touch an atom of a forming or breaking bond.  Common rows
    supported on that exact topological reaction kernel join the reactive pool,
    together with the coordinates unique to either endpoint.  Cross-family
    mixing is confined to this reactive region.  When connectivity is unchanged,
    the reaction kernel is empty and ordinary family-preserving minimum SONIC
    rules remain in force.  The reactive pool is reduced in the orthogonal
    complement of the conserved block using the Wilson matrix at the TS
    reference geometry alone.
    """

    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank tolerance must be finite and positive")
    reaction_class = str(reaction_class_id).strip().upper()
    if not reaction_class:
        raise ValueError("reaction class identifier cannot be empty")
    if start.symmetrize or end.symmetrize:
        raise GICForgeContractError(
            "endpoint connectivity composition requires unsymmetrized endpoint charts"
        )
    coordinates = np.asarray(reference_coordinates_angstrom, dtype=float)
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    if coordinates.shape != (len(symbols), 3):
        raise GICForgeContractError(
            "TS reference geometry and endpoint atom ordering are incompatible"
        )
    if start.target_rank != end.target_rank:
        raise GICForgeContractError(
            "endpoint connectivity charts and TS reference require the same vibrational rank"
        )
    target_rank = int(start.target_rank)

    start_connectivity_keys = _covalent_connectivity_primitive_keys(start)
    end_connectivity_keys = _covalent_connectivity_primitive_keys(end)
    changing_distance_keys = start_connectivity_keys ^ end_connectivity_keys
    reactive_atoms = _connectivity_change_atoms(changing_distance_keys)
    start_signatures = _exploration_gics_by_signature(start)
    end_signatures = _exploration_gics_by_signature(end)
    common_signatures = start_signatures.keys() & end_signatures.keys()
    common_start = tuple(
        gic
        for gic in start.gics
        if _exploration_gic_signature(start, gic) in common_signatures
        and not (_gic_support_atoms(start, gic) & reactive_atoms)
    )
    reactive_common = tuple(
        (start, gic)
        for gic in start.gics
        if _exploration_gic_signature(start, gic) in common_signatures
        and bool(_gic_support_atoms(start, gic) & reactive_atoms)
    )
    changing = tuple(
        (definition, gic)
        for definition in (start, end)
        for gic in definition.gics
        if _exploration_gic_signature(definition, gic) not in common_signatures
    )
    selected_sources = (
        tuple((start, gic) for gic in common_start) + reactive_common + changing
    )
    primitives, primitive_id_by_key = _unified_exploration_primitives(selected_sources)
    changing_distance_ids = tuple(
        primitive.identifier
        for primitive in primitives
        if primitive.function == "R"
        and len(primitive.atoms) == 2
        and _exploration_primitive_key(primitive) in changing_distance_keys
    )
    reaction_sources: tuple[FrozenGIC, ...] = ()
    if changing_distance_ids:
        start_only_keys = start_connectivity_keys - end_connectivity_keys
        reaction_coefficients = tuple(
            (
                primitive.identifier,
                1.0 if _exploration_primitive_key(primitive) in start_only_keys else -1.0,
            )
            for primitive in primitives
            if primitive.identifier in changing_distance_ids
        )
        reaction_norm = float(
            np.linalg.norm([coefficient for _identifier, coefficient in reaction_coefficients])
        )
        reaction_sources = (
            FrozenGIC(
                identifier="RXN000",
                name="ConnectivityReaction",
                family="TS_REACTION_DISTANCE",
                irrep="A",
                primitive_id=reaction_coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=tuple(
                    (identifier, coefficient / reaction_norm)
                    for identifier, coefficient in reaction_coefficients
                ),
            ),
        )
    common = tuple(
        replace(
            _remap_exploration_gic(start, gic, primitive_id_by_key),
            irrep="A",
        )
        for gic in common_start
    )
    sources = reaction_sources + tuple(
        replace(
            _remap_exploration_gic(definition, gic, primitive_id_by_key),
            irrep="A",
        )
        for definition, gic in (*reactive_common, *changing)
    )
    if not sources:
        raise GICForgeContractError(
            "endpoint connectivity definitions contain no changing coordinate pool"
        )
    source_definition = replace(
        start,
        backend=(
            "oracle-native-primitive.v1"
            if any(item.function in {"FROT", "FTRANS"} for item in primitives)
            else start.backend
        ),
        point_group="C1",
        symmetrize=False,
        symmetry_diagnostics=None,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coordinates
        ),
        primitives=primitives,
        gics=common + sources,
        candidate_count=len(primitives),
        periodic_coordinate_estimates=(),
        primitive_b_matrix_sha256="",
    )
    rows = np.asarray(
        build_gic_b_matrix(
            source_definition,
            coordinates_angstrom=coordinates,
        ).rows,
        dtype=float,
    )
    masses = np.asarray([atomic_mass(_required_atomic_number(item)) for item in symbols])
    weighted = rows * np.repeat(1.0 / np.sqrt(masses), 3)[None, :]

    retained_common = _joint_independent_row_indices((weighted[: len(common)],), tolerance)
    retained_set = set(retained_common)
    demoted_common = tuple(index for index in range(len(common)) if index not in retained_set)
    ordered_sources = tuple(common[index] for index in demoted_common) + sources
    source_indices = (*demoted_common, *range(len(common), len(common) + len(sources)))
    common = tuple(common[index] for index in retained_common)
    common_rows = weighted[np.asarray(retained_common, dtype=int)]
    source_rows = weighted[np.asarray(source_indices, dtype=int)]
    reactive_rank = target_rank - len(common)
    if reactive_rank <= 0 or len(ordered_sources) < reactive_rank:
        raise GICForgeContractError(
            "endpoint connectivity pool cannot supply the changing TS subspace"
        )
    projected = source_rows @ _row_complement_projector(common_rows, tolerance)
    norms = np.linalg.norm(projected, axis=1)
    valid = np.isfinite(norms) & (norms > tolerance)
    if int(np.count_nonzero(valid)) < reactive_rank:
        raise GICForgeContractError(
            "endpoint connectivity pool is rank deficient at the TS reference"
        )
    valid_indices = np.flatnonzero(valid)
    normalized = projected[valid] / norms[valid, None]
    required_source_indices = [len(demoted_common)] if reaction_sources else []
    valid_position_by_source = {
        int(source_index): position for position, source_index in enumerate(valid_indices)
    }
    try:
        required_positions = tuple(
            valid_position_by_source[index] for index in required_source_indices
        )
    except KeyError as exc:
        raise GICForgeContractError(
            "endpoint-changing bond distance is dependent on conserved SONICs"
        ) from exc
    required_rows = normalized[np.asarray(required_positions, dtype=int)]
    if required_rows.size:
        _left, required_singular, required_right = np.linalg.svd(required_rows, full_matrices=False)
        required_rank = int(np.count_nonzero(required_singular > tolerance))
        if required_rank != len(required_positions):
            raise GICForgeContractError(
                "endpoint-changing bond distances are not mutually independent"
            )
        required_basis = required_right[:required_rank].T
    else:
        required_basis = np.zeros((normalized.shape[1], 0), dtype=float)
    remaining_positions = tuple(
        index for index in range(len(valid_indices)) if index not in required_positions
    )
    residual = normalized[np.asarray(remaining_positions, dtype=int)]
    if required_basis.shape[1]:
        residual = residual - (residual @ required_basis) @ required_basis.T
    _q_matrix, triangular, pivots = qr(residual.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    remaining_rank = reactive_rank - len(required_positions)
    if diagonal.size < remaining_rank or np.any(diagonal[:remaining_rank] <= tolerance):
        raise GICForgeContractError("endpoint connectivity pool cannot restore the exact TS rank")
    selected_valid_positions = np.asarray(
        required_positions
        + tuple(remaining_positions[int(pivot)] for pivot in pivots[:remaining_rank]),
        dtype=int,
    )
    selected_source_indices = valid_indices[selected_valid_positions]
    selected_source_indices = np.asarray(
        _refine_reactive_source_selection(
            common_rows,
            source_rows,
            tuple(int(index) for index in valid_indices),
            tuple(int(index) for index in selected_source_indices),
            tuple(required_source_indices),
            tolerance=tolerance,
        ),
        dtype=int,
    )
    if reaction_sources:
        _free_q, free_triangular, free_pivots = qr(normalized.T, mode="economic", pivoting=True)
        free_diagonal = np.abs(np.diag(free_triangular))
        if free_diagonal.size < reactive_rank or np.any(free_diagonal[:reactive_rank] <= tolerance):
            raise GICForgeContractError(
                "unconstrained endpoint connectivity pool is rank deficient"
            )
        unconstrained_source_indices = tuple(
            int(index)
            for index in valid_indices[np.asarray(free_pivots[:reactive_rank], dtype=int)]
        )
        unconstrained_source_indices = _refine_reactive_source_selection(
            common_rows,
            source_rows,
            tuple(int(index) for index in valid_indices),
            unconstrained_source_indices,
            (),
            tolerance=tolerance,
        )
        required_condition = _source_selection_condition(
            common_rows,
            source_rows,
            tuple(int(index) for index in selected_source_indices),
            tolerance=tolerance,
        )
        unconstrained_condition = _source_selection_condition(
            common_rows,
            source_rows,
            unconstrained_source_indices,
            tolerance=tolerance,
        )
        if required_condition > unconstrained_condition * (
            1.0 + MAXIMUM_REACTION_PROGRESS_CONDITION_PENALTY
        ):
            selected_source_indices = np.asarray(unconstrained_source_indices, dtype=int)
    reaction_progress_selected = bool(
        reaction_sources and len(demoted_common) in selected_source_indices
    )
    reactive_pool_gics = tuple(
        _renumber_gic(gic, index)
        for index, gic in enumerate((*common, *ordered_sources), 1)
    )
    reactive_pool_definition = replace(
        source_definition,
        gics=reactive_pool_gics,
        target_rank=target_rank,
        rank=target_rank,
        primitive_source="ENDPOINT_CONNECTIVITY_REACTIVE_POOL",
        primitive_source_schema=ENDPOINT_CONNECTIVITY_SONIC_SCHEMA,
        primitive_b_matrix_sha256="",
        wilson_tangent_rank=target_rank,
        wilson_tangent_singular_min=0.0,
        wilson_tangent_singular_max=0.0,
        semantic_diagnostics=(
            *source_definition.semantic_diagnostics,
            "TS_REACTIVE_REGION "
            f"CONNECTIVITY_CHANGE={'YES' if reaction_sources else 'NO'} "
            f"CLASS={TS_REACTIVE_COORDINATE_CLASS} "
            "MIXING="
            f"{TS_REACTIVE_FAMILY_MIXING_POLICY if reaction_sources else TS_REACTIVE_FAMILY_PRESERVING_POLICY} "
            "CONSERVED=FROZEN_FAMILY_PRESERVING",
        ),
    )
    transformation = np.zeros((reactive_rank, len(ordered_sources)), dtype=float)
    transformation[np.arange(reactive_rank), selected_source_indices] = 1.0
    reactive_gics = tuple(
        replace(
            ordered_sources[int(source_index)],
            identifier=f"RXN{index:03d}",
            name=f"Reactive{index:03d}",
            irrep="A",
        )
        for index, source_index in enumerate(selected_source_indices, 1)
    )
    final_gics = tuple(
        _renumber_gic(gic, index) for index, gic in enumerate(common + reactive_gics, 1)
    )
    definition = replace(
        start,
        backend=source_definition.backend,
        point_group="C1",
        symmetrize=False,
        symmetry_diagnostics=None,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coordinates
        ),
        primitives=primitives,
        gics=final_gics,
        target_rank=target_rank,
        rank=target_rank,
        candidate_count=len(primitives),
        reduction_diagnostics=None,
        periodic_coordinate_estimates=(),
        semantic_diagnostics=(
            *start.semantic_diagnostics,
            "ENDPOINT_CONNECTIVITY_SONIC_COMBINATION "
            f"CLASS={reaction_class} COMMON={len(common)} "
            f"REACTIVE_POOL={len(ordered_sources)}->{reactive_rank} "
            f"CHANGING_DISTANCES={len(changing_distance_ids)} "
            f"REACTIVE_ATOMS={','.join(str(atom) for atom in sorted(reactive_atoms)) or 'NONE'} "
            f"REACTIVE_COMMON={len(reactive_common)} "
            f"REACTION_PROGRESS_SELECTED={int(reaction_progress_selected)} "
            f"MAX_PROGRESS_CONDITION_PENALTY={MAXIMUM_REACTION_PROGRESS_CONDITION_PENALTY:g} "
            "SELECTION=CONDITION_REFINED_PIVOTING "
            "NUMERICAL_REFERENCE=TRANSITION_STATE_ONLY",
        ),
        primitive_source="ENDPOINT_CONNECTIVITY_SONIC_COMBINATION",
        primitive_source_schema=ENDPOINT_CONNECTIVITY_SONIC_SCHEMA,
        primitive_b_matrix_sha256="",
        wilson_tangent_rank=0,
        wilson_tangent_singular_min=0.0,
        wilson_tangent_singular_max=0.0,
    )
    final_rows = (
        np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
            dtype=float,
        )
        * np.repeat(1.0 / np.sqrt(masses), 3)[None, :]
    )
    final_singular = np.linalg.svd(final_rows, compute_uv=False)
    if int(np.linalg.matrix_rank(final_rows, tol=tolerance)) != target_rank:
        raise GICForgeContractError(
            "endpoint connectivity SONIC chart is not exact rank at the TS reference"
        )
    definition = replace(
        definition,
        wilson_tangent_rank=target_rank,
        wilson_tangent_singular_min=float(final_singular[-1]),
        wilson_tangent_singular_max=float(final_singular[0]),
    )
    return EndpointConnectivitySonicCombination(
        schema=ENDPOINT_CONNECTIVITY_SONIC_SCHEMA,
        definition=definition,
        reactive_pool_definition=reactive_pool_definition,
        common_count=len(common),
        reactive_source_labels=tuple(gic.name for gic in ordered_sources),
        transformation=tuple(tuple(float(value) for value in row) for row in transformation),
        minimum_singular_value=float(final_singular[-1]),
        connectivity_change=bool(reaction_sources),
        reactive_family_mixing_allowed=bool(reaction_sources),
    )


def combine_multitopology_reactive_sonic_pool_definitions(
    definitions: Sequence[GICDefinition],
    atom_symbols: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    *,
    topology_ids: Sequence[str] | None = None,
    rank_tolerance: float = 1.0e-9,
) -> MultitopologyReactiveSonicPool:
    """Build one auditable reactive pool from two or more topology charts.

    Coordinates shared by every source topology form the conserved prefix only
    when their primitive support is disjoint from every atom in a bond whose
    membership varies across the topology set.  Shared rows on that exact
    reaction kernel and every topology-specific row form the redundant reactive
    union, with exact per-row topology provenance.  No Hessian, fitted cutoff,
    or benchmark outcome participates in this topological partition.
    """

    sources = tuple(definitions)
    if len(sources) < 2:
        raise ValueError("multi-topology reactive composition requires at least two charts")
    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank tolerance must be finite and positive")
    ids = (
        tuple(f"topology-{index:03d}" for index in range(1, len(sources) + 1))
        if topology_ids is None
        else tuple(str(value).strip() for value in topology_ids)
    )
    if len(ids) != len(sources) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("topology IDs must be non-empty, unique, and match the charts")
    if any(definition.symmetrize for definition in sources):
        raise GICForgeContractError(
            "multi-topology reactive composition requires unsymmetrized charts"
        )
    target_rank = int(sources[0].target_rank)
    if any(
        definition.rank != definition.target_rank
        or int(definition.target_rank) != target_rank
        for definition in sources
    ):
        raise GICForgeContractError(
            "multi-topology charts must have the same complete vibrational rank"
        )
    connectivity_sets = tuple(
        frozenset(_covalent_connectivity_primitive_keys(definition))
        for definition in sources
    )
    if all(connectivity == connectivity_sets[0] for connectivity in connectivity_sets[1:]):
        raise GICForgeContractError(
            "unchanged connectivity requires family-preserving minimum SONIC rules"
        )
    coordinates = np.asarray(reference_coordinates_angstrom, dtype=float)
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    if coordinates.shape != (len(symbols), 3):
        raise GICForgeContractError(
            "multi-topology reference geometry and atom ordering are incompatible"
        )

    signature_maps = tuple(_exploration_gics_by_signature(definition) for definition in sources)
    common_signatures = set(signature_maps[0])
    for mapping in signature_maps[1:]:
        common_signatures.intersection_update(mapping)
    all_common_pairs = tuple(
        (sources[0], gic)
        for gic in sources[0].gics
        if _exploration_gic_signature(sources[0], gic) in common_signatures
    )
    varying_connectivity = set().union(*connectivity_sets) - set.intersection(
        *(set(connectivity) for connectivity in connectivity_sets)
    )
    reactive_atoms = _connectivity_change_atoms(varying_connectivity)
    common_pairs = tuple(
        pair
        for pair in all_common_pairs
        if not (_gic_support_atoms(pair[0], pair[1]) & reactive_atoms)
    )
    reactive_common_pairs = tuple(
        pair
        for pair in all_common_pairs
        if bool(_gic_support_atoms(pair[0], pair[1]) & reactive_atoms)
    )
    changing_pairs: list[tuple[GICDefinition, FrozenGIC]] = []
    changing_memberships: list[set[str]] = []
    changing_position: dict[tuple[object, ...], int] = {}
    for topology_id, definition in zip(ids, sources, strict=True):
        for gic in definition.gics:
            signature = _exploration_gic_signature(definition, gic)
            if signature in common_signatures:
                continue
            position = changing_position.get(signature)
            if position is None:
                position = len(changing_pairs)
                changing_position[signature] = position
                changing_pairs.append((definition, gic))
                changing_memberships.append(set())
            changing_memberships[position].add(topology_id)
    if not changing_pairs:
        raise GICForgeContractError(
            "multi-topology definitions contain no changing coordinate pool"
        )

    primitives, primitive_id_by_key = _unified_exploration_primitives(
        (*common_pairs, *reactive_common_pairs, *changing_pairs)
    )
    common = tuple(
        replace(
            _remap_exploration_gic(definition, gic, primitive_id_by_key),
            irrep="A",
        )
        for definition, gic in common_pairs
    )
    changing = tuple(
        replace(
            _remap_exploration_gic(definition, gic, primitive_id_by_key),
            irrep="A",
        )
        for definition, gic in (*reactive_common_pairs, *changing_pairs)
    )
    source_definition = replace(
        sources[0],
        backend=(
            "oracle-native-primitive.v1"
            if any(primitive.function in {"FROT", "FTRANS"} for primitive in primitives)
            else sources[0].backend
        ),
        point_group="C1",
        symmetrize=False,
        symmetry_diagnostics=None,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coordinates
        ),
        primitives=primitives,
        gics=common + changing,
        target_rank=target_rank,
        rank=target_rank,
        candidate_count=len(primitives),
        reduction_diagnostics=None,
        periodic_coordinate_estimates=(),
        primitive_source="MULTITOPOLOGY_REACTIVE_POOL",
        primitive_source_schema=MULTITOPOLOGY_REACTIVE_SONIC_POOL_SCHEMA,
        primitive_b_matrix_sha256="",
    )
    rows = np.asarray(
        build_gic_b_matrix(source_definition, coordinates_angstrom=coordinates).rows,
        dtype=float,
    )
    masses = np.asarray([atomic_mass(_required_atomic_number(symbol)) for symbol in symbols])
    weighted = rows * np.repeat(1.0 / np.sqrt(masses), 3)[None, :]
    retained_common = _joint_independent_row_indices((weighted[: len(common)],), tolerance)
    retained_set = set(retained_common)
    demoted_common = tuple(index for index in range(len(common)) if index not in retained_set)
    conserved = tuple(common[index] for index in retained_common)
    reactive = tuple(common[index] for index in demoted_common) + changing
    memberships = (
        tuple(ids for _index in demoted_common)
        + tuple(ids for _pair in reactive_common_pairs)
        + tuple(
        tuple(topology_id for topology_id in ids if topology_id in membership)
        for membership in changing_memberships
        )
    )
    common_rows = weighted[np.asarray(retained_common, dtype=int)]
    source_indices = (*demoted_common, *range(len(common), len(common) + len(changing)))
    source_rows = weighted[np.asarray(source_indices, dtype=int)]
    projected = source_rows @ _row_complement_projector(common_rows, tolerance)
    singular = np.linalg.svd(projected, compute_uv=False)
    singular_floor = max(float(singular[0]) * tolerance, tolerance) if singular.size else tolerance
    retained_singular = singular[singular > singular_floor]
    projected_rank = int(retained_singular.size)
    reactive_rank = target_rank - len(conserved)
    if projected_rank != reactive_rank:
        raise GICForgeContractError(
            "multi-topology reactive union does not span the exact conserved complement"
        )
    condition = float(retained_singular[0] / retained_singular[-1])
    pool_gics = tuple(
        _renumber_gic(gic, index)
        for index, gic in enumerate((*conserved, *reactive), 1)
    )
    definition = replace(
        source_definition,
        gics=pool_gics,
        semantic_diagnostics=(
            *source_definition.semantic_diagnostics,
            "MULTITOPOLOGY_TS_REACTIVE_REGION "
            "CONNECTIVITY_CHANGE=YES "
            f"TOPOLOGIES={len(ids)} COMMON={len(conserved)} "
            f"REACTIVE_POOL={len(reactive)}->{reactive_rank} "
            f"REACTIVE_ATOMS={','.join(str(atom) for atom in sorted(reactive_atoms))} "
            f"REACTIVE_COMMON={len(reactive_common_pairs)} "
            f"CLASS={TS_REACTIVE_COORDINATE_CLASS} "
            f"MIXING={TS_REACTIVE_FAMILY_MIXING_POLICY}",
        ),
        wilson_tangent_rank=target_rank,
        wilson_tangent_singular_min=float(retained_singular[-1]),
        wilson_tangent_singular_max=float(retained_singular[0]),
    )
    return MultitopologyReactiveSonicPool(
        schema=MULTITOPOLOGY_REACTIVE_SONIC_POOL_SCHEMA,
        definition=definition,
        common_count=len(conserved),
        topology_ids=ids,
        reactive_source_labels=tuple(gic.name or gic.identifier for gic in reactive),
        reactive_source_topology_memberships=memberships,
        reactive_rank=reactive_rank,
        projected_source_rank=projected_rank,
        projected_source_condition=condition,
    )


def materialize_hessian_adapted_reactive_sonic_definition(
    reactive_pool_definition: GICDefinition,
    common_count: int,
    pool_to_reactive_transformation: np.ndarray,
    atom_symbols: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    *,
    rank_tolerance: float = 1.0e-9,
    allow_reactive_family_mixing: bool = False,
) -> GICDefinition:
    """Freeze a full-rank reactive-mode basis over a redundant SONIC pool.

    The first ``common_count`` rows are copied unchanged and retain their
    original family. Each row of
    ``pool_to_reactive_transformation`` combines the *complete* coordinate
    pool, including conserved rows.  This permits an exact subtraction of the
    conserved projection before Hessian adaptation; restricting the map to
    reactive-source rows would preserve the total span while contaminating
    the nominal reactive block with conserved components. Cross-family
    primitive mixing is legal only when ``allow_reactive_family_mixing`` is
    explicitly enabled for a connectivity-changing region; those rows enter
    the resulting ``TS_REACTIVE_MODE`` class. Contributions from the conserved
    prefix are limited to the metric projection that makes the reactive block
    orthogonal to the frozen block.
    """

    pool = reactive_pool_definition
    count = int(common_count)
    tolerance = float(rank_tolerance)
    coordinates = np.asarray(reference_coordinates_angstrom, dtype=float)
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    if count < 0 or count >= pool.target_rank or count > len(pool.gics):
        raise ValueError("common_count is incompatible with the reactive SONIC pool")
    if coordinates.shape != (len(symbols), 3):
        raise GICForgeContractError("reactive SONIC geometry and atom ordering are incompatible")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank tolerance must be finite and positive")
    reactive_rank = int(pool.target_rank) - count
    transformation = np.asarray(pool_to_reactive_transformation, dtype=float)
    if transformation.shape != (reactive_rank, len(pool.gics)):
        raise ValueError(
            "reactive transformation must map the complete coordinate pool to exact rank"
        )
    if (
        not np.all(np.isfinite(transformation))
        or np.linalg.matrix_rank(transformation, tol=tolerance) != reactive_rank
    ):
        raise ValueError("reactive transformation must be finite and full row rank")

    pool_coefficients = _gic_coefficient_matrix(pool.gics, pool.primitives)
    mode_coefficients = transformation @ pool_coefficients
    primitive_family = {
        primitive.identifier: primitive.family for primitive in pool.primitives
    }
    reactive_gics: list[FrozenGIC] = []
    for index, coefficients in enumerate(mode_coefficients, 1):
        scale = float(np.max(np.abs(coefficients), initial=0.0))
        if not np.isfinite(scale) or scale <= tolerance:
            raise GICForgeContractError("Hessian-adapted reactive mode has zero support")
        filtered = np.where(np.abs(coefficients) > scale * 1.0e-12, coefficients, 0.0)
        terms = tuple(
            (primitive.identifier, float(coefficient))
            for primitive, coefficient in zip(pool.primitives, filtered, strict=True)
            if coefficient != 0.0
        )
        families = {
            primitive_family[primitive_id] for primitive_id, _coefficient in terms
        }
        if len(families) > 1 and not allow_reactive_family_mixing:
            raise GICForgeContractError(
                "cross-family reactive coordinates require a connectivity change"
            )
        reactive_gics.append(
            FrozenGIC(
                identifier=f"RXN{index:03d}",
                name=f"ReactiveMode{index:03d}",
                family=(
                    TS_REACTIVE_COORDINATE_CLASS
                    if allow_reactive_family_mixing
                    else next(iter(families))
                ),
                irrep="A",
                primitive_id=terms[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=terms,
            )
        )
    final_gics = tuple(
        _renumber_gic(gic, index)
        for index, gic in enumerate((*pool.gics[:count], *reactive_gics), 1)
    )
    definition = replace(
        pool,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coordinates
        ),
        gics=final_gics,
        rank=pool.target_rank,
        reduction_diagnostics=None,
        primitive_source="HESSIAN_ADAPTED_REACTIVE_SONIC",
        primitive_source_schema=HESSIAN_ADAPTED_REACTIVE_SONIC_SCHEMA,
        primitive_b_matrix_sha256="",
        wilson_tangent_rank=0,
        wilson_tangent_singular_min=0.0,
        wilson_tangent_singular_max=0.0,
        semantic_diagnostics=(
            *pool.semantic_diagnostics,
            "HESSIAN_ADAPTED_REACTIVE_SONIC "
            f"COMMON={count} REACTIVE_POOL={len(pool.gics) - count}->{reactive_rank} "
            "CONSERVE=IDENTITY_FAMILY_PRESERVING "
            "REACTIVE_CLASS="
            f"{TS_REACTIVE_COORDINATE_CLASS if allow_reactive_family_mixing else 'ORIGINAL_FAMILIES'} "
            "REACTIVE_MIXING="
            f"{TS_REACTIVE_FAMILY_MIXING_POLICY if allow_reactive_family_mixing else TS_REACTIVE_FAMILY_PRESERVING_POLICY}",
        ),
    )
    masses = np.asarray(
        [atomic_mass(_required_atomic_number(symbol)) for symbol in symbols],
        dtype=float,
    )
    rows = (
        np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
            dtype=float,
        )
        * np.repeat(1.0 / np.sqrt(masses), 3)[None, :]
    )
    singular = np.linalg.svd(rows, compute_uv=False)
    if int(np.linalg.matrix_rank(rows, tol=tolerance)) != pool.target_rank:
        raise GICForgeContractError("Hessian-adapted reactive SONIC chart is not exact rank")
    common_rows = rows[:count]
    reactive_rows = rows[count:]
    if count:
        cross = common_rows @ reactive_rows.T
        denominator = np.outer(
            np.linalg.norm(common_rows, axis=1),
            np.linalg.norm(reactive_rows, axis=1),
        )
        normalized_cross = np.divide(
            np.abs(cross),
            denominator,
            out=np.zeros_like(cross),
            where=denominator > 0.0,
        )
        if float(np.max(normalized_cross, initial=0.0)) > max(
            10.0 * tolerance,
            np.sqrt(np.finfo(float).eps),
        ):
            raise GICForgeContractError(
                "reactive modes may mix coordinate families only inside the "
                "metric-orthogonal reactive complement"
            )
    if allow_reactive_family_mixing and any(
        gic.family != TS_REACTIVE_COORDINATE_CLASS for gic in final_gics[count:]
    ):
        raise GICForgeContractError("reactive modes require the TS reactive coordinate class")
    return replace(
        definition,
        wilson_tangent_rank=pool.target_rank,
        wilson_tangent_singular_min=float(singular[-1]),
        wilson_tangent_singular_max=float(singular[0]),
    )


def combine_endpoint_sonic_definitions(
    start: GICDefinition,
    end: GICDefinition,
    atom_symbols: Sequence[str],
    *,
    reaction_class_id: str,
    rank_tolerance: float = 1.0e-9,
) -> EndpointSonicCombination:
    """Build one unsymmetrized exact-rank chart from two endpoint SONIC sets.

    Coordinates whose normalized primitive definitions are identical are kept
    unchanged. The remaining endpoint coordinates are pooled, projected out
    of the common block in mass-weighted Cartesian space, and reduced to the
    missing vibrational rank by one deterministic joint endpoint SVD. The
    resulting linear combinations are accepted only when the complete chart
    has full rank at both minima and at their aligned Cartesian midpoint.

    This is the general exploration construction. It deliberately carries no
    parent-point-group constraint; symmetry is perceived again from the
    selected Cartesian seed when the standard TS exploitation phase starts.
    """

    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank tolerance must be finite and positive")
    reaction_class = str(reaction_class_id).strip().upper()
    if not reaction_class:
        raise ValueError("reaction class identifier cannot be empty")
    if start.symmetrize or end.symmetrize:
        raise GICForgeContractError(
            "general endpoint SONIC composition requires unsymmetrized exploration charts"
        )
    start_xyz = np.asarray(start.reference_coordinates_angstrom, dtype=float)
    end_xyz = np.asarray(end.reference_coordinates_angstrom, dtype=float)
    if start_xyz.shape != end_xyz.shape or start_xyz.ndim != 2 or start_xyz.shape[1] != 3:
        raise GICForgeContractError("endpoint SONIC geometries must have identical shape")
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    if len(symbols) != start_xyz.shape[0]:
        raise GICForgeContractError("atom symbols do not match endpoint SONIC geometries")
    if (
        start.rank != start.target_rank
        or end.rank != end.target_rank
        or start.target_rank != end.target_rank
    ):
        raise GICForgeContractError(
            "endpoint SONIC charts must have the same complete vibrational rank"
        )
    target_rank = int(start.target_rank)

    start_signatures = _exploration_gics_by_signature(start)
    end_signatures = _exploration_gics_by_signature(end)
    common_signatures = start_signatures.keys() & end_signatures.keys()
    common_start = tuple(
        gic for gic in start.gics if _exploration_gic_signature(start, gic) in common_signatures
    )
    start_reactive = tuple(
        gic for gic in start.gics if _exploration_gic_signature(start, gic) not in common_signatures
    )
    end_reactive = tuple(
        gic for gic in end.gics if _exploration_gic_signature(end, gic) not in common_signatures
    )
    common_count = len(common_start)
    reactive_rank = target_rank - common_count
    if reactive_rank <= 0:
        raise GICForgeContractError("endpoint topology change has no endpoint-specific SONIC rank")

    selected = tuple((start, gic) for gic in common_start + start_reactive) + tuple(
        (end, gic) for gic in end_reactive
    )
    primitives, primitive_id_by_key = _unified_exploration_primitives(selected)
    combination_backend = (
        "oracle-native-primitive.v1"
        if any(primitive.function in {"FROT", "FTRANS"} for primitive in primitives)
        else start.backend
    )
    common = tuple(
        replace(
            _remap_exploration_gic(start, gic, primitive_id_by_key),
            irrep="A",
        )
        for gic in common_start
    )
    source_gics = tuple(
        replace(
            _remap_exploration_gic(start, gic, primitive_id_by_key),
            irrep="A",
        )
        for gic in start_reactive
    ) + tuple(
        replace(
            _remap_exploration_gic(end, gic, primitive_id_by_key),
            irrep="A",
        )
        for gic in end_reactive
    )
    oriented_primitives, oriented_gics = _oriented_reaction_domain_coordinates(
        start,
        end,
        start_xyz,
        end_xyz,
        existing_primitives=primitives,
        tolerance=tolerance,
    )
    if oriented_primitives:
        primitives = (*primitives, *oriented_primitives)
        source_gics = (*source_gics, *oriented_gics)
    if len(source_gics) < reactive_rank:
        raise GICForgeContractError(
            "endpoint-specific SONIC pool is smaller than the missing vibrational rank"
        )
    source_definition = replace(
        start,
        backend=combination_backend,
        point_group="C1",
        symmetrize=False,
        symmetry_diagnostics=None,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in start_xyz
        ),
        primitives=primitives,
        gics=common + source_gics,
        candidate_count=len(primitives),
        periodic_coordinate_estimates=(),
        primitive_b_matrix_sha256="",
    )
    rows_start = np.asarray(
        build_gic_b_matrix(source_definition, coordinates_angstrom=start_xyz).rows,
        dtype=float,
    )
    rows_end = np.asarray(
        build_gic_b_matrix(source_definition, coordinates_angstrom=end_xyz).rows,
        dtype=float,
    )
    midpoint_xyz = 0.5 * (start_xyz + end_xyz)
    rows_midpoint = np.asarray(
        build_gic_b_matrix(source_definition, coordinates_angstrom=midpoint_xyz).rows,
        dtype=float,
    )
    values_start = evaluate_gic_values(source_definition, coordinates_angstrom=start_xyz)
    values_end = evaluate_gic_values(source_definition, coordinates_angstrom=end_xyz)
    masses = np.asarray([atomic_mass(_required_atomic_number(item)) for item in symbols])
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    weighted_start = rows_start * inverse_sqrt_mass[None, :]
    weighted_midpoint = rows_midpoint * inverse_sqrt_mass[None, :]
    weighted_end = rows_end * inverse_sqrt_mass[None, :]
    oriented_absolute_indices = tuple(
        index
        for index, gic in enumerate(source_definition.gics)
        if gic.family == "OUT_OF_PLANE" and gic.name.startswith("REACTION_ORIENTED_U(")
    )
    retained_common = _joint_independent_row_indices(
        (
            weighted_start[:common_count],
            weighted_midpoint[:common_count],
            weighted_end[:common_count],
        ),
        tolerance,
        fixed_matrices=(
            weighted_start[np.asarray(oriented_absolute_indices, dtype=int)],
            weighted_midpoint[np.asarray(oriented_absolute_indices, dtype=int)],
            weighted_end[np.asarray(oriented_absolute_indices, dtype=int)],
        ),
    )
    retained_set = set(retained_common)
    demoted_common = tuple(index for index in range(common_count) if index not in retained_set)
    source_indices = (*demoted_common, *range(common_count, len(source_definition.gics)))
    order = (*retained_common, *source_indices)
    all_source_order_gics = common + source_gics
    common = tuple(common[index] for index in retained_common)
    source_gics = tuple(all_source_order_gics[index] for index in source_indices)
    common_count = len(common)
    reactive_rank = target_rank - common_count
    ordered_start = weighted_start[np.asarray(order, dtype=int)]
    ordered_midpoint = weighted_midpoint[np.asarray(order, dtype=int)]
    ordered_end = weighted_end[np.asarray(order, dtype=int)]
    ordered_values_start = values_start[np.asarray(order, dtype=int)]
    ordered_values_end = values_end[np.asarray(order, dtype=int)]
    common_start_rows = ordered_start[:common_count]
    common_midpoint_rows = ordered_midpoint[:common_count]
    common_end_rows = ordered_end[:common_count]
    source_start_rows = ordered_start[common_count:]
    source_midpoint_rows = ordered_midpoint[common_count:]
    source_end_rows = ordered_end[common_count:]
    source_delta = ordered_values_end[common_count:] - ordered_values_start[common_count:]
    oriented_indices = tuple(
        index
        for index, gic in enumerate(source_gics)
        if gic.family == "OUT_OF_PLANE" and gic.name.startswith("REACTION_ORIENTED_U(")
    )
    if len(oriented_indices) > reactive_rank:
        raise GICForgeContractError(
            "oriented reaction coordinates exceed the missing vibrational rank"
        )
    oriented_set = set(oriented_indices)
    reduced_indices = tuple(index for index in range(len(source_gics)) if index not in oriented_set)
    reduced_rank = reactive_rank - len(oriented_indices)

    fixed_start_rows = source_start_rows[np.asarray(oriented_indices, dtype=int)]
    fixed_midpoint_rows = source_midpoint_rows[np.asarray(oriented_indices, dtype=int)]
    fixed_end_rows = source_end_rows[np.asarray(oriented_indices, dtype=int)]
    extended_common_start = np.vstack((common_start_rows, fixed_start_rows))
    extended_common_midpoint = np.vstack((common_midpoint_rows, fixed_midpoint_rows))
    extended_common_end = np.vstack((common_end_rows, fixed_end_rows))
    for label, rows in (
        ("start", extended_common_start),
        ("midpoint", extended_common_midpoint),
        ("end", extended_common_end),
    ):
        if int(np.linalg.matrix_rank(rows, tol=tolerance)) != len(rows):
            raise GICForgeContractError(
                f"oriented reaction coordinate is dependent at the {label} geometry"
            )

    reduced_start_rows = source_start_rows[np.asarray(reduced_indices, dtype=int)]
    reduced_midpoint_rows = source_midpoint_rows[np.asarray(reduced_indices, dtype=int)]
    reduced_end_rows = source_end_rows[np.asarray(reduced_indices, dtype=int)]
    projected_start = reduced_start_rows @ _row_complement_projector(
        extended_common_start, tolerance
    )
    projected_end = reduced_end_rows @ _row_complement_projector(extended_common_end, tolerance)
    projected_midpoint = reduced_midpoint_rows @ _row_complement_projector(
        extended_common_midpoint, tolerance
    )
    if (
        int(np.linalg.matrix_rank(projected_start, tol=tolerance)) < reduced_rank
        or int(np.linalg.matrix_rank(projected_end, tol=tolerance)) < reduced_rank
        or int(np.linalg.matrix_rank(projected_midpoint, tol=tolerance)) < reduced_rank
    ):
        raise GICForgeContractError(
            "endpoint-specific SONIC pool cannot restore full rank at endpoints and midpoint"
        )
    fixed_transformation = np.zeros((len(oriented_indices), len(source_gics)), dtype=float)
    for row, source_index in enumerate(oriented_indices):
        fixed_transformation[row, source_index] = 1.0
    reduced_transformation = np.zeros((reduced_rank, len(source_gics)), dtype=float)
    if reduced_rank:
        norms = np.sqrt(
            (1.0 / 3.0)
            * (
                np.sum(projected_start * projected_start, axis=1)
                + np.sum(projected_end * projected_end, axis=1)
                + np.sum(projected_midpoint * projected_midpoint, axis=1)
            )
        )
        valid = np.isfinite(norms) & (norms > tolerance)
        if int(np.count_nonzero(valid)) < reduced_rank:
            raise GICForgeContractError(
                "endpoint-specific SONIC is singular after common-block projection"
            )
        valid_indices = tuple(
            source_index
            for source_index, keep in zip(reduced_indices, valid, strict=True)
            if bool(keep)
        )
        norms = norms[valid]
        projected_start = projected_start[valid]
        projected_end = projected_end[valid]
        projected_midpoint = projected_midpoint[valid]
        normalized_start = projected_start / norms[:, None]
        normalized_end = projected_end / norms[:, None]
        normalized_midpoint = projected_midpoint / norms[:, None]
        pooled = np.hstack((normalized_start, normalized_midpoint, normalized_end))
        left, _singular, _right = np.linalg.svd(pooled, full_matrices=False)
        reduced_local = left[:, :reduced_rank].T / norms[None, :]
        reduced_transformation[:, np.asarray(valid_indices, dtype=int)] = reduced_local
        reduced_transformation = _reaction_aligned_rows(
            reduced_transformation,
            reduced_transformation @ source_delta,
            tolerance,
        )
    transformation = np.vstack((fixed_transformation, reduced_transformation))

    source_coefficients = _gic_coefficient_matrix(source_gics, primitives)
    combined_coefficients = transformation @ source_coefficients
    for row in range(combined_coefficients.shape[0]):
        norm = float(np.linalg.norm(combined_coefficients[row]))
        if not np.isfinite(norm) or norm <= tolerance:
            raise GICForgeContractError("combined reactive SONIC has zero coefficients")
        combined_coefficients[row] /= norm
        transformation[row] /= norm
    reactive_gics = tuple(
        _combined_gic(index, "A", coefficients, primitives)
        for index, coefficients in enumerate(combined_coefficients, 1)
    )
    final_gics = tuple(
        _renumber_gic(gic, index) for index, gic in enumerate(common + reactive_gics, 1)
    )
    combined_definition = replace(
        start,
        backend=combination_backend,
        point_group="C1",
        symmetrize=False,
        symmetry_diagnostics=None,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in start_xyz
        ),
        primitives=primitives,
        gics=final_gics,
        target_rank=target_rank,
        rank=target_rank,
        candidate_count=len(primitives),
        reduction_diagnostics=None,
        periodic_coordinate_estimates=(),
        semantic_diagnostics=(
            *start.semantic_diagnostics,
            "ENDPOINT_SONIC_COMBINATION "
            f"CLASS={reaction_class} COMMON={common_count} "
            f"REACTIVE={len(source_gics)}->{reactive_rank} "
            f"ORIENTED_REACTION_COORDINATES={len(oriented_gics)} "
            "SYMMETRY=EXPLORATION_DYNAMIC",
        ),
        primitive_source="ENDPOINT_SONIC_COMBINATION",
        primitive_source_schema=ENDPOINT_SONIC_COMBINATION_SCHEMA,
        primitive_b_matrix_sha256="",
        wilson_tangent_rank=0,
        wilson_tangent_singular_min=0.0,
        wilson_tangent_singular_max=0.0,
    )
    endpoint_singular: list[np.ndarray] = []
    for label, coordinates in (
        ("start", start_xyz),
        ("midpoint", midpoint_xyz),
        ("end", end_xyz),
    ):
        rows = (
            np.asarray(
                build_gic_b_matrix(combined_definition, coordinates_angstrom=coordinates).rows,
                dtype=float,
            )
            * inverse_sqrt_mass[None, :]
        )
        singular = np.linalg.svd(rows, compute_uv=False)
        if int(np.linalg.matrix_rank(rows, tol=tolerance)) != target_rank:
            raise GICForgeContractError(
                f"combined endpoint SONIC chart loses rank at the {label} geometry"
            )
        endpoint_singular.append(singular)
    midpoint_singular = endpoint_singular[1]
    combined_definition = replace(
        combined_definition,
        wilson_tangent_rank=target_rank,
        wilson_tangent_singular_min=float(midpoint_singular[-1]),
        wilson_tangent_singular_max=float(midpoint_singular[0]),
    )
    displacement = transformation @ source_delta
    return EndpointSonicCombination(
        schema=ENDPOINT_SONIC_COMBINATION_SCHEMA,
        definition=combined_definition,
        common_count=common_count,
        start_reactive_gics=tuple(gic.name for gic in start_reactive),
        end_reactive_gics=tuple(gic.name for gic in end_reactive),
        source_labels=tuple(gic.name for gic in source_gics),
        source_irreps=tuple("A" for _gic in source_gics),
        transformation=tuple(tuple(float(value) for value in row) for row in transformation),
        endpoint_displacement=tuple(float(value) for value in displacement),
        rank_start=target_rank,
        rank_end=target_rank,
        minimum_singular_value_start=float(endpoint_singular[0][-1]),
        minimum_singular_value_end=float(endpoint_singular[2][-1]),
    )


def _oriented_reaction_domain_coordinates(
    start: GICDefinition,
    end: GICDefinition,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    existing_primitives: Sequence[GICPrimitive],
    tolerance: float,
) -> tuple[tuple[GICPrimitive, ...], tuple[FrozenGIC, ...]]:
    """Add signed local-volume coordinates when an attachment changes.

    Distances and valence angles do not distinguish the two Cartesian sheets
    of a pyramidal inversion.  At a center that keeps at least three bonded
    neighbours while exchanging another attachment, one oriented
    out-of-plane coordinate supplies the missing sheet label.  It enters the
    endpoint-only source pool and is subsequently reduced with all other
    reactive sources; the final chart therefore remains exactly 3N-6.
    A candidate is mandatory only when its endpoint values have opposite
    signs; a merely different pyramidalization belongs to the ordinary
    endpoint source reduction and must not be mistaken for an inversion.
    """

    start_adjacency = _bond_adjacency_from_definition(start)
    end_adjacency = _bond_adjacency_from_definition(end)
    used_keys = {_exploration_primitive_key(item) for item in existing_primitives}
    primitives: list[GICPrimitive] = []
    gics: list[FrozenGIC] = []
    for center in sorted(start_adjacency.keys() | end_adjacency.keys()):
        start_neighbours = start_adjacency.get(center, set())
        end_neighbours = end_adjacency.get(center, set())
        if start_neighbours == end_neighbours:
            continue
        common_neighbours = tuple(sorted(start_neighbours & end_neighbours))
        if len(common_neighbours) < 3:
            continue
        for plane1, plane2, out in combinations(common_neighbours, 3):
            serial = len(primitives) + 1
            primitive = GICPrimitive(
                identifier=f"RXU{serial:03d}",
                name=f"RXU{serial:03d}",
                family="OUT_OF_PLANE",
                function="U",
                atoms=(center, plane1, plane2, out),
                provenance="ENDPOINT_ATTACHMENT_CHANGE_ORIENTED_VOLUME",
                semantic_id=f"REACTION_ORIENTED_U:{center}:{plane1}:{plane2}:{out}",
                semantic_type="REACTION_ORIENTED_OUT_OF_PLANE",
            )
            if _exploration_primitive_key(primitive) in used_keys:
                continue
            probe_gic = FrozenGIC(
                identifier=f"RXQ{serial:03d}",
                name=f"REACTION_ORIENTED_U({center},{plane1},{plane2},{out})",
                family="OUT_OF_PLANE",
                irrep="A",
                primitive_id=primitive.identifier,
                gaussian_expression=primitive.gaussian_expression(),
            )
            probe = replace(
                start,
                backend="oracle-native-primitive.v1",
                point_group="C1",
                symmetrize=False,
                primitives=(primitive,),
                gics=(probe_gic,),
                target_rank=1,
                rank=1,
                reference_coordinates_angstrom=tuple(
                    tuple(float(value) for value in row) for row in start_xyz
                ),
                periodic_coordinate_estimates=(),
                primitive_b_matrix_sha256="",
            )
            try:
                start_value = float(evaluate_gic_values(probe, coordinates_angstrom=start_xyz)[0])
                end_value = float(evaluate_gic_values(probe, coordinates_angstrom=end_xyz)[0])
                rows = tuple(
                    np.asarray(
                        build_gic_b_matrix(probe, coordinates_angstrom=coordinates).rows,
                        dtype=float,
                    )[0]
                    for coordinates in (start_xyz, 0.5 * (start_xyz + end_xyz), end_xyz)
                )
            except (ArithmeticError, FloatingPointError, ValueError):
                continue
            orientation_threshold = max(1.0e-3, 100.0 * tolerance)
            if (
                min(abs(start_value), abs(end_value)) <= orientation_threshold
                or start_value * end_value >= 0.0
                or any(
                    np.any(~np.isfinite(row)) or float(np.linalg.norm(row)) <= tolerance
                    for row in rows
                )
            ):
                continue
            used_keys.add(_exploration_primitive_key(primitive))
            primitives.append(primitive)
            gics.append(probe_gic)
            break
    return tuple(primitives), tuple(gics)


def _bond_adjacency_from_definition(
    definition: GICDefinition,
) -> dict[int, set[int]]:
    """Recover only the explicit covalent stretches frozen in one SONIC chart."""

    adjacency: dict[int, set[int]] = {}
    for primitive in definition.primitives:
        if primitive.function != "R" or len(primitive.atoms) != 2:
            continue
        left, right = (int(atom) for atom in primitive.atoms)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def _exploration_primitive_key(primitive: GICPrimitive) -> tuple[object, ...]:
    atoms = tuple(int(atom) for atom in primitive.atoms)
    if primitive.function == "R":
        atoms = tuple(sorted(atoms))
    elif primitive.function in {"A", "D"}:
        atoms = min(atoms, tuple(reversed(atoms)))
    refs = tuple(primitive.refs)
    if primitive.function in FRAGMENT_BODY_PAIR_FUNCTIONS:
        # Fragment labels are local to one topology chart.  The atom
        # memberships already present in this key carry the physical
        # identity, whereas retaining F001/F002 here would make equivalent
        # bodies look different (or reuse one label for different bodies)
        # when several topology charts are pooled.
        refs = refs[2:]
    elif primitive.function in FRAGMENT_BODY_SINGLE_FUNCTIONS:
        refs = refs[1:]
    return (
        primitive.function,
        atoms,
        int(primitive.mode),
        tuple(int(atom) for atom in primitive.ref_atoms),
        refs,
        tuple(int(atom) for atom in primitive.frame_atoms),
        tuple(int(atom) for atom in primitive.ref_frame_atoms),
    )


def _covalent_connectivity_primitive_keys(
    definition: GICDefinition,
) -> set[tuple[object, ...]]:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    referenced = {
        primitive_id
        for gic in definition.gics
        for primitive_id, _coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        )
    }
    return {
        _exploration_primitive_key(primitive)
        for primitive_id, primitive in primitive_by_id.items()
        if primitive_id in referenced
        and primitive.function == "R"
        and primitive.family in {"STRETCH", "LOCAL_XH_STRETCH"}
        and len(primitive.atoms) == 2
    }


def _connectivity_change_atoms(
    changing_connectivity_keys: Sequence[tuple[object, ...]],
) -> frozenset[int]:
    """Return the exact atom kernel of the bonds whose membership changes."""

    return frozenset(
        int(atom)
        for key in changing_connectivity_keys
        for atom in tuple(key[1])
    )


def _gic_support_atoms(
    definition: GICDefinition,
    gic: FrozenGIC,
) -> frozenset[int]:
    """Return every atom on which one frozen coordinate explicitly depends."""

    primitive_by_id = {
        primitive.identifier: primitive for primitive in definition.primitives
    }
    support: set[int] = set()
    for primitive_id, _coefficient in (
        gic.coefficients or ((gic.primitive_id, 1.0),)
    ):
        primitive = primitive_by_id[primitive_id]
        support.update(int(atom) for atom in primitive.atoms)
        support.update(int(atom) for atom in primitive.ref_atoms)
        support.update(int(atom) for atom in primitive.frame_atoms)
        support.update(int(atom) for atom in primitive.ref_frame_atoms)
    return frozenset(support)


def _normalized_exploration_gic_terms(
    definition: GICDefinition,
    gic: FrozenGIC,
) -> tuple[tuple[tuple[object, ...], float], ...]:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    terms = sorted(
        (
            _exploration_primitive_key(primitive_by_id[primitive_id]),
            float(coefficient),
        )
        for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
    )
    norm = float(np.linalg.norm([coefficient for _key, coefficient in terms]))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise GICForgeContractError(f"SONIC {gic.identifier} has zero coefficients")
    normalized = [(key, coefficient / norm) for key, coefficient in terms]
    if normalized and normalized[0][1] < 0.0:
        normalized = [(key, -coefficient) for key, coefficient in normalized]
    return tuple((key, round(coefficient, 11)) for key, coefficient in normalized)


def _exploration_gic_signature(
    definition: GICDefinition,
    gic: FrozenGIC,
) -> tuple[object, ...]:
    return gic.family, _normalized_exploration_gic_terms(definition, gic)


def _exploration_gics_by_signature(
    definition: GICDefinition,
) -> dict[tuple[object, ...], FrozenGIC]:
    result: dict[tuple[object, ...], FrozenGIC] = {}
    for gic in definition.gics:
        signature = _exploration_gic_signature(definition, gic)
        if signature in result:
            raise GICForgeContractError("endpoint SONIC definition contains duplicate coordinates")
        result[signature] = gic
    return result


def _unified_exploration_primitives(
    selected_gics: Sequence[tuple[GICDefinition, FrozenGIC]],
) -> tuple[tuple[GICPrimitive, ...], dict[tuple[object, ...], str]]:
    primitive_by_key: dict[tuple[object, ...], GICPrimitive] = {}
    for definition, gic in selected_gics:
        by_id = {primitive.identifier: primitive for primitive in definition.primitives}
        for primitive_id, _coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            primitive = by_id[primitive_id]
            primitive_by_key.setdefault(_exploration_primitive_key(primitive), primitive)
    ordered = tuple(primitive_by_key)
    identifier_by_key = {key: f"P{index:03d}" for index, key in enumerate(ordered, 1)}
    fragment_memberships = sorted(
        {
            tuple(sorted(int(atom) for atom in members))
            for primitive in primitive_by_key.values()
            for members in _fragment_body_memberships(primitive)
        }
    )
    fragment_id_by_membership = {
        members: f"F{index:03d}"
        for index, members in enumerate(fragment_memberships, 1)
    }
    primitives = tuple(
        replace(
            primitive_by_key[key],
            identifier=identifier_by_key[key],
            refs=_canonical_fragment_body_refs(
                primitive_by_key[key],
                fragment_id_by_membership,
            ),
            # A SOOP is a serialized Cartesian covector.  Resetting its chart
            # to PRINCIPAL discards the contract that gives the B row meaning
            # when endpoint pools are unified.  Ordinary angular/periodic
            # primitives are canonicalized as before.
            chart=(
                primitive_by_key[key].chart
                if primitive_by_key[key].function == "SOOP"
                else "PRINCIPAL"
            ),
            chart_reference_radian=None,
        )
        for key in ordered
    )
    return primitives, identifier_by_key


def _fragment_body_memberships(
    primitive: GICPrimitive,
) -> tuple[tuple[int, ...], ...]:
    if primitive.function in FRAGMENT_BODY_PAIR_FUNCTIONS:
        return (tuple(primitive.atoms), tuple(primitive.ref_atoms))
    if primitive.function in FRAGMENT_BODY_SINGLE_FUNCTIONS:
        return (tuple(primitive.atoms),)
    return ()


def _canonical_fragment_body_refs(
    primitive: GICPrimitive,
    fragment_id_by_membership: dict[tuple[int, ...], str],
) -> tuple[str, ...]:
    """Give pooled fragment bodies globally unique membership-based IDs."""

    memberships = _fragment_body_memberships(primitive)
    if not memberships:
        return tuple(primitive.refs)
    prefix = tuple(
        fragment_id_by_membership[tuple(sorted(int(atom) for atom in members))]
        for members in memberships
    )
    return (*prefix, *tuple(primitive.refs[len(memberships) :]))


def _remap_exploration_gic(
    definition: GICDefinition,
    gic: FrozenGIC,
    identifier_by_key: dict[tuple[object, ...], str],
) -> FrozenGIC:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    coefficients = tuple(
        (
            identifier_by_key[_exploration_primitive_key(primitive_by_id[primitive_id])],
            float(coefficient),
        )
        for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
    )
    return replace(
        gic,
        primitive_id=coefficients[0][0],
        gaussian_expression="LINEAR_COMBINATION",
        coefficients=coefficients,
    )


def _required_atomic_number(symbol: str) -> int:
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise GICForgeContractError(f"unsupported atom symbol in SONIC composition: {symbol}")
    return int(number)


def _gic_coefficient_matrix(
    gics: Sequence[FrozenGIC],
    primitives: Sequence[GICPrimitive],
) -> np.ndarray:
    index = {primitive.identifier: column for column, primitive in enumerate(primitives)}
    result = np.zeros((len(gics), len(primitives)), dtype=float)
    for row, gic in enumerate(gics):
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            result[row, index[primitive_id]] += float(coefficient)
    return result


def _combined_gic(
    index: int,
    irrep: str,
    coefficients: np.ndarray,
    primitives: Sequence[GICPrimitive],
) -> FrozenGIC:
    terms = tuple(
        (primitive.identifier, float(coefficient))
        for primitive, coefficient in zip(primitives, coefficients, strict=True)
        if abs(float(coefficient)) > 1.0e-12
    )
    return FrozenGIC(
        identifier=f"RXN{index:03d}",
        name=f"Reaction{index:03d}",
        family="REACTION_COORDINATE",
        irrep=irrep,
        primitive_id=terms[0][0],
        gaussian_expression="LINEAR_COMBINATION",
        coefficients=terms,
    )


def _renumber_gic(gic: FrozenGIC, index: int) -> FrozenGIC:
    return replace(gic, identifier=f"GIC{index:03d}")


def _row_complement_projector(rows: np.ndarray, tolerance: float) -> np.ndarray:
    columns = rows.shape[1]
    if rows.size == 0:
        return np.eye(columns, dtype=float)
    _u_matrix, singular, vt_matrix = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.count_nonzero(singular > tolerance))
    basis = vt_matrix[:rank]
    return np.eye(columns, dtype=float) - basis.T @ basis


def _refine_reactive_source_selection(
    common_rows: np.ndarray,
    source_rows: np.ndarray,
    valid_source_indices: tuple[int, ...],
    selected_source_indices: tuple[int, ...],
    required_source_indices: tuple[int, ...],
    *,
    tolerance: float,
) -> tuple[int, ...]:
    """Improve a pivoted complement without replacing its required reaction row."""

    selected = list(selected_source_indices)
    required = set(required_source_indices)
    valid = set(valid_source_indices)
    current = _source_selection_condition(common_rows, source_rows, selected, tolerance=tolerance)
    while True:
        unselected = tuple(sorted(valid - set(selected)))
        best_condition = current
        best_trial: list[int] | None = None
        for position, source_index in enumerate(selected):
            if source_index in required:
                continue
            for replacement in unselected:
                trial = selected.copy()
                trial[position] = replacement
                trial_condition = _source_selection_condition(
                    common_rows, source_rows, trial, tolerance=tolerance
                )
                if trial_condition < best_condition * (1.0 - 1.0e-12):
                    best_condition = trial_condition
                    best_trial = trial
        if best_trial is None:
            return tuple(selected)
        selected = best_trial
        current = best_condition


def _source_selection_condition(
    common_rows: np.ndarray,
    source_rows: np.ndarray,
    selected_source_indices: Sequence[int],
    *,
    tolerance: float,
) -> float:
    common = np.asarray(common_rows, dtype=float)
    sources = np.asarray(source_rows, dtype=float)
    if common.size:
        common = common / np.linalg.norm(common, axis=1)[:, None]
    selected = sources[np.asarray(selected_source_indices, dtype=int)]
    selected = selected / np.linalg.norm(selected, axis=1)[:, None]
    rows = np.vstack((common, selected))
    singular = np.linalg.svd(rows, compute_uv=False)
    if singular.size != rows.shape[0] or singular[-1] <= tolerance:
        return float("inf")
    return float(singular[0] / singular[-1])


def _joint_independent_row_indices(
    matrices: Sequence[np.ndarray],
    tolerance: float,
    *,
    fixed_matrices: Sequence[np.ndarray] | None = None,
) -> tuple[int, ...]:
    """Select the largest deterministic row block independent in every matrix.

    ``fixed_matrices`` contains mandatory rows that seed each row space.  This
    is used for signed reaction-domain coordinates: a formally common SONIC
    row is demoted when retaining it would make an oriented coordinate
    dependent at any protected geometry.
    """

    values = tuple(np.asarray(matrix, dtype=float) for matrix in matrices)
    if not values:
        return ()
    row_count = values[0].shape[0]
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in values):
        raise ValueError("joint row-selection matrices have incompatible shapes")
    if any(not np.all(np.isfinite(matrix)) for matrix in values):
        raise ValueError("joint row-selection matrices contain non-finite values")
    if row_count == 0:
        return ()
    scales = np.sqrt(sum(np.sum(matrix * matrix, axis=1) for matrix in values) / len(values))
    available = {index for index, scale in enumerate(scales) if float(scale) > tolerance}
    normalized = tuple(matrix / scales[:, None] for matrix in values)
    bases: list[list[np.ndarray]] = [[] for _matrix in normalized]
    if fixed_matrices is not None:
        fixed = tuple(np.asarray(matrix, dtype=float) for matrix in fixed_matrices)
        if len(fixed) != len(values) or any(
            matrix.ndim != 2 or matrix.shape[1] != values[index].shape[1]
            for index, matrix in enumerate(fixed)
        ):
            raise ValueError("fixed joint row-selection matrices have incompatible shapes")
        if any(not np.all(np.isfinite(matrix)) for matrix in fixed):
            raise ValueError("fixed joint row-selection matrices contain non-finite values")
        fixed_count = fixed[0].shape[0]
        if any(matrix.shape[0] != fixed_count for matrix in fixed):
            raise ValueError("fixed joint row-selection matrices have different row counts")
        for row_index in range(fixed_count):
            joint_scale = float(
                np.sqrt(
                    sum(float(matrix[row_index] @ matrix[row_index]) for matrix in fixed)
                    / len(fixed)
                )
            )
            if not np.isfinite(joint_scale) or joint_scale <= tolerance:
                raise GICForgeContractError("mandatory reaction coordinate has a zero Wilson row")
            residuals = tuple(
                _twice_orthogonalized_vector(
                    matrix[row_index] / joint_scale,
                    basis,
                )
                for matrix, basis in zip(fixed, bases, strict=True)
            )
            if min(float(np.linalg.norm(residual)) for residual in residuals) <= tolerance:
                raise GICForgeContractError("mandatory reaction coordinates are jointly dependent")
            for basis, residual in zip(bases, residuals, strict=True):
                basis.append(residual / np.linalg.norm(residual))
    selected: list[int] = []
    while available:
        scored: list[tuple[float, int, tuple[np.ndarray, ...]]] = []
        for index in sorted(available):
            residuals = tuple(
                _twice_orthogonalized_vector(matrix[index], basis)
                for matrix, basis in zip(normalized, bases, strict=True)
            )
            score = min(float(np.linalg.norm(residual)) for residual in residuals)
            scored.append((score, index, residuals))
        maximum = max(score for score, _index, _residuals in scored)
        if maximum <= tolerance:
            break
        score, pivot, residuals = next(item for item in scored if item[0] >= maximum - 1.0e-12)
        if score <= tolerance:
            break
        for basis, residual in zip(bases, residuals, strict=True):
            basis.append(residual / np.linalg.norm(residual))
        selected.append(pivot)
        available.remove(pivot)
    return tuple(selected)


def _twice_orthogonalized_vector(
    vector: np.ndarray,
    basis: Sequence[np.ndarray],
) -> np.ndarray:
    residual = np.array(vector, dtype=float, copy=True)
    if not basis:
        return residual
    orthonormal = np.vstack(basis)
    for _pass in range(2):
        residual -= (orthonormal @ residual) @ orthonormal
    return residual


def _mass_weighted_definition_rows(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray,
    inverse_sqrt_mass: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates_angstrom,
        ).rows,
        dtype=float,
    )
    if rows.shape[1] != inverse_sqrt_mass.size:
        raise GICForgeContractError("TS chart and Cartesian masses are incompatible")
    return rows * inverse_sqrt_mass[None, :]


def _row_mode_alignments(rows: np.ndarray, mode: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rows, dtype=float)
    vector = np.asarray(mode, dtype=float).reshape(-1)
    norms = np.linalg.norm(matrix, axis=1)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError("coordinate rows and Hessian mode are incompatible")
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise GICForgeContractError("TS chart contains a zero or non-finite Wilson row")
    return np.abs(matrix @ vector) / norms


def _definition_row_curvature(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray,
    displacement_angstrom: np.ndarray,
    row_index: int,
    probe_angstrom: float,
) -> float:
    center = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates_angstrom,
        ).rows,
        dtype=float,
    )[row_index]
    plus = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates_angstrom + displacement_angstrom,
        ).rows,
        dtype=float,
    )[row_index]
    minus = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates_angstrom - displacement_angstrom,
        ).rows,
        dtype=float,
    )[row_index]
    center_norm = float(np.linalg.norm(center))
    if not np.isfinite(center_norm) or center_norm <= 0.0:
        raise GICForgeContractError("TS chart contains a zero reaction-mode Wilson row")
    curvature = float(np.linalg.norm((plus - minus) / (2.0 * probe_angstrom)) / center_norm)
    if not np.isfinite(curvature):
        raise GICForgeContractError("TS reaction-mode curvature is non-finite")
    return curvature


def _definition_mode_curvatures(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray,
    displacement_angstrom: np.ndarray,
    probe_angstrom: float,
) -> np.ndarray:
    center = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates_angstrom,
        ).rows,
        dtype=float,
    )
    plus = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates_angstrom + displacement_angstrom,
        ).rows,
        dtype=float,
    )
    minus = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates_angstrom - displacement_angstrom,
        ).rows,
        dtype=float,
    )
    norms = np.linalg.norm(center, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise GICForgeContractError("TS chart contains a zero reaction-mode Wilson row")
    curvatures = (
        np.linalg.norm(
            (plus - minus) / (2.0 * probe_angstrom),
            axis=1,
        )
        / norms
    )
    if np.any(~np.isfinite(curvatures)):
        raise GICForgeContractError("TS reaction-mode curvature is non-finite")
    return curvatures


def _mode_weighted_curvature(
    alignments: np.ndarray,
    curvatures: np.ndarray,
) -> float:
    weights = np.square(np.asarray(alignments, dtype=float))
    values = np.asarray(curvatures, dtype=float)
    total = float(np.sum(weights))
    if values.shape != weights.shape or not np.isfinite(total) or total <= 0.0:
        raise GICForgeContractError("TS mode weights are invalid")
    return float(np.sqrt(np.sum(weights * np.square(values)) / total))


def _mode_projection_participation(alignments: np.ndarray) -> float:
    weights = np.square(np.asarray(alignments, dtype=float))
    total = float(np.sum(weights))
    squared_total = float(np.sum(np.square(weights)))
    if not np.isfinite(total) or not np.isfinite(squared_total) or squared_total <= 0.0:
        raise GICForgeContractError("TS mode participation is invalid")
    return float(total * total / squared_total)


def _reaction_aligned_rows(
    transformation: np.ndarray,
    displacement: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    norm = float(np.linalg.norm(displacement))
    if norm <= tolerance:
        return _deterministic_row_signs(transformation)
    first = displacement / norm
    _u_matrix, _singular, vt_matrix = np.linalg.svd(first.reshape(1, -1))
    aligned = np.vstack((first, vt_matrix[1:])) @ transformation
    aligned = _deterministic_row_signs(aligned)
    if float(aligned[0] @ transformation.T @ displacement) < 0.0:
        aligned[0] *= -1.0
    return aligned


def _deterministic_row_signs(rows: np.ndarray) -> np.ndarray:
    result = np.array(rows, dtype=float, copy=True)
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


__all__ = [
    "ENDPOINT_CONNECTIVITY_SONIC_SCHEMA",
    "ENDPOINT_CONNECTIVITY_TS_SELECTION_SCHEMA",
    "ENDPOINT_SONIC_COMBINATION_SCHEMA",
    "EndpointConnectivitySonicCombination",
    "EndpointConnectivityTSSelection",
    "EndpointSonicCombination",
    "combine_endpoint_connectivity_sonic_definitions",
    "combine_endpoint_sonic_definitions",
    "select_endpoint_connectivity_ts_definition",
]
