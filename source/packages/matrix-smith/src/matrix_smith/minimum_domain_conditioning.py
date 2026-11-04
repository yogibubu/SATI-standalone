"""Separated intra/inter conditioning for stable pseudobond MINIMUM charts.

ORACLE owns the chemistry, domains, compatibility blocks, and thresholds.
SMITH only realizes the frozen prescription by exact-rank linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np

from matrix_chem import (
    CoordinateComponent,
    coordinate_selection_units,
    validate_coordinate_component_transform,
)
from matrix_chem.coordinate_atlas_contract import (
    ATLAS_TASK_MINIMUM,
    CANDIDATE_OPTIONAL,
    PSEUDOBOND_REQUIRED,
    OracleCoordinateAtlasContract,
    validate_oracle_coordinate_atlas_contract,
)
from matrix_chem.oracle_sonic_contract import (
    OracleSonicContract,
    validate_oracle_sonic_contract,
)

from .component_invariants import validate_indivisible_gic_components
from .contracts import GICForgeContractError
from .audits.dense import normalized_dense_condition_and_rank
from .bmatrix import SparseBMatrix, SparseBRow
from .evaluation import (
    build_sparse_gic_and_primitive_b_matrices,
    build_sparse_gic_b_matrix,
    recombine_sparse_gic_rows,
)
from .minimum_sparse_certificate import IncrementalSupportMatcher, certify_minimum_sparse_rows
from .minimum_domain_sampling import local_domain_samples
from .models import GICDefinition, GICPrimitive, GICReductionDiagnostics
from .policy import (
    FRAGMENT_MODE_PSEUDO_BONDS,
    MAX_NORMALIZED_SONIC_CONDITION,
    RANK_TOLERANCE,
)
from .parallel import resolve_local_workers


MINIMUM_DOMAIN_SELECTION_SCHEMA = "matrix.smith.minimum_domain_partition.v2"
_CONTACT_SUPPORT_REF = "PSEUDOBOND_CONTACT_SUPPORT"
_INTER_FAMILIES = frozenset(
    {
        "PSEUDO_BOND_DISTANCE",
        "FRAG_DISTANCE",
        "PSEUDO_BOND_BEND",
        "PSEUDO_BOND_TORSION",
        "FRAG_TRANSLATION",
        "FRAG_ORIENTATION",
    }
)
_MAX_INTERNAL_EXCHANGE_ORDER = 2
# Dense linear algebra is retained solely as a reference oracle on a closed
# local support.  A MINIMUM chart must never materialize its complete Wilson
# matrix just to rank or condition it.
_MAX_LOCAL_DENSE_AUDIT_ROWS = 96
_MAX_LOCAL_DENSE_AUDIT_COLUMNS = 288
_MIN_HIERARCHICAL_SUPPORT_BRANCHES = 512
_MAX_HIERARCHICAL_SUPPORT_BRANCHES = 512


@dataclass(frozen=True)
class _Audit:
    maximum_condition: float
    mean_condition: float
    worst_sample: str
    full_rank: bool


@dataclass(frozen=True)
class _InternalExchange:
    slot: int
    source_id: str
    alternative_id: str
    block_id: str


@dataclass(frozen=True)
class _PrimitiveAtlasEntry:
    prescription: object
    domain_id: str


@dataclass(frozen=True)
class _SupportCandidate:
    primitive_ids: tuple[str, ...]
    inter_audit: _Audit
    locality: int
    sparse_score: int


@dataclass(frozen=True)
class _SampleRows:
    label: str
    gic_rows: SparseBMatrix
    primitive_rows: SparseBMatrix


def condition_minimum_pseudobond_chart(
    definition: GICDefinition,
    contract: OracleCoordinateAtlasContract,
    *,
    sonic_contract: OracleSonicContract,
    rank_tolerance: float = RANK_TOLERANCE,
) -> GICDefinition:
    """Realize one exact fixed MINIMUM chart over ORACLE's local domain."""

    if not _requires_conditioning(contract, sonic_contract=sonic_contract):
        return definition
    _validate_input_chart(definition)
    # This certificate is the production sparse gate.  The following local
    # candidate comparisons retain dense reference algebra for now and are
    # checked against it during the equivalence migration.
    sparse_certificate = certify_minimum_sparse_rows(
        build_sparse_gic_b_matrix(definition), expected_rank=definition.target_rank
    )
    if sparse_certificate.status != "PASS":
        raise GICForgeContractError("MINIMUM sparse Wilson certificate failed")
    samples = local_domain_samples(
        definition.reference_coordinates_angstrom,
        contract.local_domains,
    )
    internal, internal_exchange, intra_before, intra_after = _condition_internal_chart(
        definition,
        contract=contract,
        sonic_contract=sonic_contract,
        rank_tolerance=rank_tolerance,
    )
    conditioned, support_ids, inter_before, inter_after, global_after = (
        _condition_interfragment_chart(
            internal,
            contract=contract,
            sonic_contract=sonic_contract,
            samples=samples,
            rank_tolerance=rank_tolerance,
        )
    )
    validate_indivisible_gic_components(conditioned)
    if not global_after.full_rank:
        raise GICForgeContractError(
            "separated MINIMUM conditioning did not retain exact global rank"
        )
    return _with_partition_diagnostics(
        conditioned,
        sample_count=len(samples),
        internal_exchange=internal_exchange,
        support_ids=support_ids,
        intra_before=intra_before,
        intra_after=intra_after,
        inter_before=inter_before,
        inter_after=inter_after,
        global_after=global_after,
    )


def _requires_conditioning(
    contract: OracleCoordinateAtlasContract,
    *,
    sonic_contract: OracleSonicContract,
) -> bool:
    validate_oracle_coordinate_atlas_contract(contract)
    validate_oracle_sonic_contract(sonic_contract)
    if contract.topology_hash != sonic_contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE SONIC and coordinate-atlas topology hashes differ")
    return contract.task_regime == ATLAS_TASK_MINIMUM and any(
        item.pseudobond_policy == PSEUDOBOND_REQUIRED for item in contract.interactions
    )


def _validate_input_chart(definition: GICDefinition) -> None:
    if definition.fragment_mode != FRAGMENT_MODE_PSEUDO_BONDS:
        raise GICForgeContractError(
            "a required MINIMUM pseudobond domain needs a PSEUDO_BONDS chart"
        )
    if len(definition.gics) != definition.target_rank:
        raise GICForgeContractError(
            "MINIMUM domain conditioning requires an exact nonredundant chart"
        )


def _condition_internal_chart(
    definition: GICDefinition,
    *,
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
    rank_tolerance: float,
) -> tuple[GICDefinition, tuple[str, ...], _Audit, _Audit]:
    internal_indices = _internal_gic_indices(definition)
    reference = (("REFERENCE", np.asarray(definition.reference_coordinates_angstrom)),)
    cache = _row_cache(definition, reference)
    baseline = _audit_cached(
        cache,
        row_indices=internal_indices,
        projected_against=(),
        rank_tolerance=rank_tolerance,
    )
    blocks = _substitution_blocks(contract, prefix="INTERNAL::")
    trigger = min((item.condition_trigger for item in blocks.values()), default=None)
    if trigger is None or baseline.maximum_condition <= trigger:
        return definition, (), baseline, baseline

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    prescriptions = _primitive_prescriptions(
        definition,
        contract=contract,
        sonic_contract=sonic_contract,
    )
    exchanges = _eligible_internal_exchanges(
        definition,
        primitive_by_id=primitive_by_id,
        prescription_by_id=prescriptions,
        substitution_blocks=frozenset(blocks),
    )
    candidates: list[tuple[_Audit, tuple[_InternalExchange, ...]]] = []
    for order in range(1, _MAX_INTERNAL_EXCHANGE_ORDER + 1):
        for exchange_set in combinations(exchanges, order):
            if not _compatible_exchange_set(exchange_set):
                continue
            audit = _audit_cached(
                cache,
                row_indices=internal_indices,
                projected_against=(),
                rank_tolerance=rank_tolerance,
                replacement_by_slot={item.slot: item.alternative_id for item in exchange_set},
                primitive_index={
                    item.identifier: index for index, item in enumerate(definition.primitives)
                },
            )
            if audit.full_rank:
                candidates.append((audit, exchange_set))
    if not candidates:
        return definition, (), baseline, baseline
    audit, selected = min(
        candidates,
        key=lambda item: (
            item[0].maximum_condition,
            item[0].mean_condition,
            tuple(exchange.alternative_id for exchange in item[1]),
        ),
    )
    minimum_gain = max(
        float(blocks[item.block_id].minimum_relative_gain or 0.0) for item in selected
    )
    if not _material_gain(
        baseline.maximum_condition,
        audit.maximum_condition,
        minimum_gain=minimum_gain,
    ):
        return definition, (), baseline, baseline
    candidate = _apply_internal_exchanges(
        definition,
        selected,
        primitive_by_id=primitive_by_id,
    )
    labels = tuple(f"{item.source_id}->{item.alternative_id}" for item in selected)
    return candidate, labels, baseline, audit


def _condition_interfragment_chart(
    definition: GICDefinition,
    *,
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
    samples: tuple[tuple[str, np.ndarray], ...],
    rank_tolerance: float,
) -> tuple[GICDefinition, tuple[str, ...], _Audit, _Audit, _Audit]:
    internal_indices = _internal_gic_indices(definition)
    fragment_by_atom = {
        atom: fragment.fragment_id
        for fragment in sonic_contract.primary_topology.fragments
        for atom in fragment.atoms
    }
    pairs = tuple(
        sorted(
            {
                _support_pair(primitive)
                for primitive in definition.primitives
                if primitive.refs and primitive.refs[0] == _CONTACT_SUPPORT_REF
            }
        )
    )
    current = definition
    # Certify the frozen chart before evaluating the much larger retained
    # primitive pool.  In the overwhelmingly common case the chart selected
    # by ORACLE is already valid, so conditioning must pay only for the rows
    # that LINK will actually consume.
    chart_cache = _gic_row_cache(definition, samples)
    global_baseline = _audit_cached(
        chart_cache,
        row_indices=tuple(range(len(definition.gics))),
        projected_against=(),
        rank_tolerance=rank_tolerance,
    )
    if (
        global_baseline.full_rank
        and global_baseline.maximum_condition <= MAX_NORMALIZED_SONIC_CONDITION
    ):
        # The canonical chart is already certified over every ORACLE local
        # sample.  Searching alternative support combinations cannot improve
        # its validity and only adds a large combinatorial prefactor.
        return (
            definition,
            (),
            global_baseline,
            global_baseline,
            global_baseline,
        )
    if not pairs:
        # A component-aware recovery may freeze a complete ORACLE pose/contact
        # basis directly, with no replaceable local-support alternatives left
        # in the chart.  Certify that fixed basis over every prescribed local
        # sample and retain it unchanged.
        audit = global_baseline
        if not audit.full_rank:
            raise GICForgeContractError(
                "fixed ORACLE interfragment chart loses rank in its local domain"
            )
        return definition, (), audit, audit, audit
    primitive_cache = _row_cache(definition, samples)
    support_ids: list[str] = []
    before_audits: list[_Audit] = []
    after_audits: list[_Audit] = []
    for pair in pairs:
        pair_cache = _contact_pair_row_cache(
            primitive_cache,
            pair=pair,
            contract=contract,
        )
        current, selected_ids, before, after = _condition_contact_pair(
            current,
            pair=pair,
            fragment_by_atom=fragment_by_atom,
            contract=contract,
            sonic_contract=sonic_contract,
            internal_indices=internal_indices,
            rank_tolerance=rank_tolerance,
            defer_global_selection=len(pairs) > 1,
            cache=_recombine_row_cache(current, pair_cache),
        )
        support_ids.extend(selected_ids)
        before_audits.append(before)
        after_audits.append(after)
    cache = _recombine_row_cache(current, primitive_cache)
    global_audit = _audit_cached(
        cache,
        row_indices=tuple(range(len(current.gics))),
        projected_against=(),
        rank_tolerance=rank_tolerance,
    )
    return (
        current,
        tuple(support_ids),
        _combine_audits(before_audits),
        _combine_audits(after_audits),
        global_audit,
    )


def _contact_pair_row_cache(
    cache: tuple[_SampleRows, ...],
    *,
    pair: tuple[str, str],
    contract: OracleCoordinateAtlasContract,
) -> tuple[_SampleRows, ...]:
    """Return only the reference and perturbations owned by one contact.

    ORACLE defines an independent local domain for every stable contact.  A
    perturbation of another contact cannot constrain this pair's support
    choice and made the previous implementation repeat every local audit for
    every pair (quadratic work and memory).  Domain labels are immutable
    contract identifiers, so this filtering is exact rather than geometric.
    """

    pair_ids = frozenset(pair)
    domain_ids = tuple(
        domain.domain_id
        for domain in contract.local_domains
        if frozenset(
            (domain.reference_fragment_id, domain.moving_fragment_id)
        )
        == pair_ids
    )
    prefixes = tuple(f"{domain_id}:" for domain_id in domain_ids)
    selected = tuple(
        sample
        for sample in cache
        if sample.label == "REFERENCE" or sample.label.startswith(prefixes)
    )
    if len(selected) <= 1:
        raise GICForgeContractError(
            "a required pseudobond contact has no ORACLE-owned local-domain samples"
        )
    return selected


def _condition_contact_pair(
    definition: GICDefinition,
    *,
    pair: tuple[str, str],
    fragment_by_atom: dict[int, str],
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
    internal_indices: tuple[int, ...],
    rank_tolerance: float,
    defer_global_selection: bool,
    cache: tuple[_SampleRows, ...],
) -> tuple[GICDefinition, tuple[str, ...], _Audit, _Audit]:
    support_slots = _selected_support_slots(definition, pair=pair)
    required_slots = _required_contact_slots(
        definition,
        pair=pair,
        fragment_by_atom=fragment_by_atom,
    )
    inter_indices = tuple(sorted((*required_slots, *support_slots)))
    support_count = len(support_slots)
    if support_count <= 0:
        # A complete ORACLE/SMITH fragment-pose chart may span this pair with
        # rigid translations/orientations (or other required interfragment
        # primitives) and therefore contain no replaceable contact-support
        # slot.  This is a valid alternative representation, not a malformed
        # pseudobond chart.  Audit it, but do not invent a local substitution.
        if inter_indices:
            audit = _audit_cached(
                cache,
                row_indices=inter_indices,
                projected_against=internal_indices,
                rank_tolerance=rank_tolerance,
            )
        else:
            audit = _Audit(
                maximum_condition=0.0,
                mean_condition=0.0,
                worst_sample="REFERENCE",
                full_rank=True,
            )
        return definition, (), audit, audit
    primitive_index = {item.identifier: index for index, item in enumerate(definition.primitives)}
    baseline = _audit_cached(
        cache,
        row_indices=inter_indices,
        projected_against=internal_indices,
        rank_tolerance=rank_tolerance,
    )
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    support_pool = tuple(
        item
        for item in definition.primitives
        if _is_support_candidate_for_pair(item, pair)
    )
    units = _preselect_equivalent_support_units(
        _support_selection_units(support_pool),
        cache=cache,
        primitive_index=primitive_index,
        primitive_by_id=primitive_by_id,
    )
    selections = _hierarchical_support_id_sets(
        units,
        target_count=support_count,
        cache=cache,
        required_slots=required_slots,
        primitive_index=primitive_index,
        # Other contacts will still be replaced in a multi-pair sweep; only
        # immutable intrafragment/required rows may constrain their prefixes.
        fixed_slots=(tuple(sorted(set(internal_indices) | set(required_slots)))
                     if defer_global_selection else
                     tuple(i for i in range(len(definition.gics)) if i not in support_slots)),
        rank_tolerance=rank_tolerance,
    )
    if not selections:
        raise GICForgeContractError(
            "ORACLE contact-support pool cannot span the required exact dimension"
        )

    trigger, minimum_gain = _contact_condition_policy(contract)
    baseline_complete = _selected_support_is_component_complete(
        definition,
        support_slots=support_slots,
        support_pool=support_pool,
    )
    if baseline_complete and baseline.full_rank and baseline.maximum_condition <= trigger:
        return definition, _support_ids(definition, support_slots), baseline, baseline

    candidates = _contact_support_candidates(
        cache=cache,
        selections=selections,
        support_slots=support_slots,
        inter_indices=inter_indices,
        internal_indices=internal_indices,
        primitive_index=primitive_index,
        primitive_by_id=primitive_by_id,
        sonic_contract=sonic_contract,
        contract=contract,
        pair=pair,
        rank_tolerance=rank_tolerance,
    )
    if not candidates:
        raise GICForgeContractError(
            "LOCAL_NONREDUNDANT_DOMAIN_UNSAFE: no complete exact MINIMUM interfragment "
            "chart is full rank over the ORACLE local domain"
        )
    selected = _choose_contact_support(
        candidates,
        cache=cache,
        support_slots=support_slots,
        primitive_index=primitive_index,
        gic_count=len(definition.gics),
        rank_tolerance=rank_tolerance,
        minimum_gain=minimum_gain,
        defer_global_selection=defer_global_selection,
    )
    # Dense numerical algebra is an audit oracle only: the sparse selector has
    # already made the decision.  Audit the winner and one nearest rejected
    # alternative so parity drift is visible without scaling with pool size.
    selected_svd = _audit_cached(
        cache,
        row_indices=inter_indices,
        projected_against=internal_indices,
        rank_tolerance=rank_tolerance,
        replacement_by_slot=dict(zip(support_slots, selected.primitive_ids, strict=True)),
        primitive_index=primitive_index,
    )
    if not selected_svd.full_rank or selected_svd.maximum_condition > MAX_NORMALIZED_SONIC_CONDITION:
        ordered = sorted(
            (item for item in candidates if item.primitive_ids != selected.primitive_ids),
            key=lambda candidate: (
                candidate.sparse_score, candidate.locality, candidate.primitive_ids
            ),
        )
        for alternative in ordered:
            global_audit = _audit_cached(
                cache,
                row_indices=tuple(range(len(definition.gics))),
                projected_against=(),
                rank_tolerance=rank_tolerance,
                replacement_by_slot=dict(zip(support_slots, alternative.primitive_ids, strict=True)),
                primitive_index=primitive_index,
            )
            if not global_audit.full_rank:
                continue
            audit = _audit_cached(
                cache,
                row_indices=inter_indices,
                projected_against=internal_indices,
                rank_tolerance=rank_tolerance,
                replacement_by_slot=dict(zip(support_slots, alternative.primitive_ids, strict=True)),
                primitive_index=primitive_index,
            )
            if audit.full_rank and audit.maximum_condition <= MAX_NORMALIZED_SONIC_CONDITION:
                selected, selected_svd = alternative, audit
                break
        else:
            raise GICForgeContractError(
                "LOCAL_NONREDUNDANT_DOMAIN_UNSAFE: sparse candidates fail the local SVD oracle"
            )
    selected = replace(selected, inter_audit=selected_svd)
    excluded = next(
        (
            item for item in sorted(
                candidates,
                key=lambda candidate: (
                    candidate.sparse_score,
                    candidate.locality,
                    candidate.primitive_ids,
                ),
            )
            if item.primitive_ids != selected.primitive_ids
        ),
        None,
    )
    excluded_svd = None
    if excluded is not None:
        excluded_svd = _audit_cached(
            cache,
            row_indices=inter_indices,
            projected_against=internal_indices,
            rank_tolerance=rank_tolerance,
            replacement_by_slot=dict(zip(support_slots, excluded.primitive_ids, strict=True)),
            primitive_index=primitive_index,
        )
    selected_definition = _replace_support_slots(
        definition,
        support_slots=support_slots,
        primitive_ids=selected.primitive_ids,
        primitive_by_id=primitive_by_id,
    )
    audit_records = (
        f"MINIMUM_LOCAL_SVD_WINNER pair={pair[0]}:{pair[1]} "
        f"condition={selected_svd.maximum_condition:.16g} full_rank={selected_svd.full_rank}",
    )
    if excluded is not None and excluded_svd is not None:
        audit_records += (
            f"MINIMUM_LOCAL_SVD_NEAREST_EXCLUDED pair={pair[0]}:{pair[1]} "
            f"condition={excluded_svd.maximum_condition:.16g} full_rank={excluded_svd.full_rank}",
        )
    selected_definition = replace(
        selected_definition,
        semantic_diagnostics=(*selected_definition.semantic_diagnostics, *audit_records),
    )
    return (
        selected_definition,
        selected.primitive_ids,
        baseline,
        selected.inter_audit,
    )


def _contact_support_candidates(
    *,
    cache: tuple[_SampleRows, ...],
    selections: tuple[tuple[str, ...], ...],
    support_slots: tuple[int, ...],
    inter_indices: tuple[int, ...],
    internal_indices: tuple[int, ...],
    primitive_index: dict[str, int],
    primitive_by_id: dict[str, GICPrimitive],
    sonic_contract: OracleSonicContract,
    contract: OracleCoordinateAtlasContract,
    pair: tuple[str, str],
    rank_tolerance: float,
) -> tuple[_SupportCandidate, ...]:
    candidates: list[_SupportCandidate] = []
    for primitive_ids in selections:
        audit, sparse_score = _sparse_local_support_audit(
            cache,
            support_slots=support_slots,
            primitive_ids=primitive_ids,
            inter_indices=inter_indices,
            internal_indices=internal_indices,
            primitive_index=primitive_index,
        )
        if audit.full_rank:
            candidates.append(
                _SupportCandidate(
                    primitive_ids=primitive_ids,
                    inter_audit=audit,
                    locality=_support_locality(
                        primitive_ids,
                        primitive_by_id=primitive_by_id,
                        sonic_contract=sonic_contract,
                        contract=contract,
                        pair=pair,
                    ),
                    sparse_score=sparse_score,
                )
            )
    return tuple(candidates)


def _choose_contact_support(
    candidates: tuple[_SupportCandidate, ...],
    *,
    cache: tuple[_SampleRows, ...],
    support_slots: tuple[int, ...],
    primitive_index: dict[str, int],
    gic_count: int,
    rank_tolerance: float,
    minimum_gain: float,
    defer_global_selection: bool,
) -> _SupportCandidate:
    best_condition = min(item.inter_audit.maximum_condition for item in candidates)
    locality_band = best_condition / max(1.0e-12, 1.0 - minimum_gain)
    eligible = tuple(
        item for item in candidates if item.inter_audit.maximum_condition <= locality_band
    )
    if defer_global_selection:
        return min(
            eligible,
            key=lambda item: (
                item.sparse_score,
                item.locality,
                item.primitive_ids,
            ),
        )
    selected, _global_audit = _select_global_safe_support(
        eligible,
        cache=cache,
        support_slots=support_slots,
        primitive_index=primitive_index,
        gic_count=gic_count,
        rank_tolerance=rank_tolerance,
    )
    return selected


def _support_selection_units(
    primitives: tuple[GICPrimitive, ...],
) -> tuple[tuple[str, ...], ...]:
    components = tuple(
        CoordinateComponent(
            operator=item.function,
            atoms=item.atoms,
            mode=item.mode,
            ref_atoms=item.ref_atoms,
            context=(item.family, *item.refs),
        )
        for item in primitives
    )
    try:
        units = coordinate_selection_units(components)
    except ValueError as exc:
        raise GICForgeContractError(
            f"ORACLE contact-support component pool is invalid: {exc}"
        ) from exc
    return tuple(tuple(primitives[index].identifier for index in unit) for unit in units)


def _support_id_sets(
    units: tuple[tuple[str, ...], ...],
    *,
    target_count: int,
) -> tuple[tuple[str, ...], ...]:
    selected: list[tuple[str, ...]] = []

    def visit(index: int, current: tuple[str, ...]) -> None:
        if len(current) == target_count:
            selected.append(current)
            return
        if index == len(units) or len(current) > target_count:
            return
        visit(index + 1, current)
        visit(index + 1, current + units[index])

    visit(0, ())
    return tuple(sorted(set(selected)))


def _preselect_equivalent_support_units(
    units: tuple[tuple[str, ...], ...],
    *,
    cache: tuple[_SampleRows, ...],
    primitive_index: dict[str, int],
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[tuple[str, ...], ...]:
    """Keep one deterministic representative of each local support class."""

    representatives: dict[tuple[object, ...], tuple[int, tuple[str, ...]]] = {}
    for unit in units:
        families = tuple(sorted(primitive_by_id[item].family for item in unit))
        supports = tuple(
            tuple(
                sorted(
                    tuple(
                        column
                        for column, _value in sample.primitive_rows.rows[
                            primitive_index[item]
                        ].entries
                    )
                    for item in unit
                )
            )
            for sample in cache
        )
        normalized_rows = []
        for sample in cache:
            sample_rows = []
            for item in unit:
                row = sample.primitive_rows.rows[primitive_index[item]]
                norm = row.norm()
                entries = tuple((column, value / norm) for column, value in row.entries)
                if entries and entries[0][1] < 0:
                    entries = tuple((column, -value) for column, value in entries)
                sample_rows.append(tuple((column, round(value, 12)) for column, value in entries))
            normalized_rows.append(tuple(sorted(sample_rows)))
        key = (families, supports, tuple(normalized_rows))
        cost = sum(
            sample.primitive_rows.rows[primitive_index[item]].nnz
            for sample in cache
            for item in unit
        )
        candidate = (cost, tuple(unit))
        previous = representatives.get(key)
        if previous is None or candidate < previous:
            representatives[key] = candidate
    return tuple(value[1] for value in sorted(representatives.values()))


def _numeric_support_rows(cache, units, primitive_index, fixed_slots):
    """Project the bounded contact pool against the unchanged chart once.

    Structural matching cannot detect linearly dependent physical Wilson rows.
    Reuse local numerical residuals to prune those prefixes before the branch
    budget is spent. Nonlocal pools keep the sparse path and its final gate.
    """
    identifiers = tuple(dict.fromkeys(item for unit in units for item in unit))
    position = {item: i for i, item in enumerate(identifiers)}
    projected = []
    for sample in cache:
        candidates = tuple(sample.primitive_rows.rows[primitive_index[item]] for item in identifiers)
        offset = len(sample.gic_rows.rows)
        local = _closed_local_audit_rows(
            (*sample.gic_rows.rows, *candidates),
            row_indices=tuple(range(offset, offset + len(candidates))),
            projected_against=fixed_slots,
        )
        if local is not None:
            norms = np.asarray([row.norm() for row in candidates])
            local = np.divide(local, norms[:, None], out=np.zeros_like(local), where=norms[:, None] > 0)
        projected.append(local)
    return position, tuple(projected)


def _extend_numeric_prefix(basis, rows, tolerance):
    if rows is None:
        return basis
    result = basis
    for row in rows:
        residual = row.copy()
        # Twice-reorthogonalized modified Gram--Schmidt in the closed contact
        # support; candidate units remain indivisible.
        for _ in range(2):
            residual -= result.T @ (result @ residual)
        norm = float(np.linalg.norm(residual))
        if norm <= tolerance:
            return None
        result = np.vstack((result, residual / norm))
    return result


def _hierarchical_support_id_sets(
    units: tuple[tuple[str, ...], ...],
    *,
    target_count: int,
    cache: tuple[_SampleRows, ...],
    required_slots: tuple[int, ...],
    primitive_index: dict[str, int],
    branch_budget: int | None = None,
    fixed_slots: tuple[int, ...] | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
) -> tuple[tuple[str, ...], ...]:
    """Reuse structural and local numerical prefixes in a bounded search."""

    if branch_budget is None:
        scale = max(1, len(cache)) * max(1, target_count) * max(1, len(units).bit_length())
        branch_budget = min(
            _MAX_HIERARCHICAL_SUPPORT_BRANCHES,
            max(_MIN_HIERARCHICAL_SUPPORT_BRANCHES, 64 * scale),
        )
    if branch_budget <= 0:
        raise ValueError("MINIMUM hierarchical branch budget must be positive")
    base: list[IncrementalSupportMatcher] = []
    for sample in cache:
        matcher = IncrementalSupportMatcher.empty(sample.gic_rows.column_count).extend(
            tuple(sample.gic_rows.rows[index] for index in required_slots)
        )
        if matcher is None:
            return ()
        base.append(matcher)
    position, projected = ({}, (None,) * len(cache)) if fixed_slots is None else _numeric_support_rows(
        cache, units, primitive_index, fixed_slots)
    numerical = tuple(np.zeros((0, 0 if rows is None else rows.shape[1])) for rows in projected)
    frontier = [(0, (), tuple(base), numerical)]
    while frontier:
        next_frontier = []
        completed: list[tuple[str, ...]] = []
        for start, selected, matchers, numerical in frontier:
            for unit_index in range(start, len(units)):
                unit = units[unit_index]
                combined = selected + unit
                if len(combined) > target_count:
                    continue
                extended = []
                numerical_extended = []
                for sample, matcher, basis, rows in zip(cache, matchers, numerical, projected, strict=True):
                    candidate = matcher.extend(
                        tuple(
                            sample.primitive_rows.rows[primitive_index[item]]
                            for item in unit
                        )
                    )
                    if candidate is None:
                        break
                    numeric = _extend_numeric_prefix(
                        basis, None if rows is None else rows[[position[item] for item in unit]],
                        rank_tolerance)
                    if numeric is None:
                        break
                    extended.append(candidate)
                    numerical_extended.append(numeric)
                else:
                    if len(combined) == target_count:
                        completed.append(combined)
                    else:
                        next_frontier.append((unit_index + 1, combined, tuple(extended), tuple(numerical_extended)))
        if completed:
            # A complete layer can itself contain a combinatorial number of
            # charts.  The budget applies to terminal states as well as to the
            # live frontier; otherwise the subsequent numerical audit quietly
            # becomes exhaustive even though expansion was advertised as
            # bounded.
            return tuple(sorted(set(completed)))[:branch_budget]
        if len(next_frontier) > branch_budget:
            next_frontier.sort(key=lambda state: (len(state[1]), state[1]))
            next_frontier = next_frontier[:branch_budget]
        frontier = next_frontier
    return ()


def _replace_support_slots(
    definition: GICDefinition,
    *,
    support_slots: tuple[int, ...],
    primitive_ids: tuple[str, ...],
    primitive_by_id: dict[str, GICPrimitive],
) -> GICDefinition:
    if len(support_slots) != len(primitive_ids):
        raise GICForgeContractError("contact-support replacement changed chart size")
    gics = list(definition.gics)
    for slot, primitive_id in zip(support_slots, primitive_ids, strict=True):
        source = gics[slot]
        primitive = primitive_by_id[primitive_id]
        gics[slot] = replace(
            source,
            name=f"{source.irrep}{primitive.name}",
            family=primitive.family,
            primitive_id=primitive_id,
            gaussian_expression=primitive.gaussian_expression(),
            coefficients=((primitive_id, 1.0),),
        )
    candidate = replace(definition, gics=tuple(gics))
    diagnostics = _synchronize_reduction_selection(
        candidate,
        primitive_by_id=primitive_by_id,
    )
    return replace(candidate, reduction_diagnostics=diagnostics)


def _row_cache(
    definition: GICDefinition,
    samples: tuple[tuple[str, np.ndarray], ...],
) -> tuple[_SampleRows, ...]:
    def build(sample) -> _SampleRows:
        label, coordinates = sample
        gic_matrix, primitive_matrix = build_sparse_gic_and_primitive_b_matrices(
            definition,
            coordinates_angstrom=coordinates,
        )
        return _SampleRows(
            label=label,
            # Candidate matching revisits these rows many times.  Expanding a
            # native CSR view once avoids reconstructing Python row objects in
            # every branch of the bounded search.
            gic_rows=_materialize_sparse_rows(gic_matrix),
            primitive_rows=_materialize_sparse_rows(primitive_matrix),
        )
    workers = resolve_local_workers(min(8, max(1, len(samples))), len(samples))
    if workers == 1:
        return tuple(build(sample) for sample in samples)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="smith-minimum") as executor:
        return tuple(executor.map(build, samples))


def _gic_row_cache(
    definition: GICDefinition,
    samples: tuple[tuple[str, np.ndarray], ...],
) -> tuple[_SampleRows, ...]:
    """Evaluate only operative GIC rows for the common certification path."""

    def build(sample) -> _SampleRows:
        label, coordinates = sample
        gic_matrix = build_sparse_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates,
        )
        return _SampleRows(
            label=label,
            gic_rows=_materialize_sparse_rows(gic_matrix),
            primitive_rows=SparseBMatrix(
                rows=(),
                column_count=gic_matrix.column_count,
                backend=gic_matrix.backend,
            ),
        )

    workers = resolve_local_workers(min(8, max(1, len(samples))), len(samples))
    if workers == 1:
        return tuple(build(sample) for sample in samples)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="smith-minimum") as executor:
        return tuple(executor.map(build, samples))


def _materialize_sparse_rows(matrix: SparseBMatrix) -> SparseBMatrix:
    """Freeze a CSR compatibility view when rows will be revisited repeatedly."""

    if isinstance(matrix.rows, tuple):
        return matrix
    return SparseBMatrix(
        rows=tuple(matrix.rows),
        column_count=matrix.column_count,
        row_labels=matrix.row_labels,
        backend=matrix.backend,
    )


def _recombine_row_cache(
    definition: GICDefinition,
    cache: tuple[_SampleRows, ...],
) -> tuple[_SampleRows, ...]:
    """Rebuild only GIC combinations over an unchanged primitive tangent cache."""

    recombined: list[_SampleRows] = []
    for sample in cache:
        recombined.append(
            _SampleRows(
                label=sample.label,
                gic_rows=recombine_sparse_gic_rows(definition, sample.primitive_rows),
                primitive_rows=sample.primitive_rows,
            )
        )
    return tuple(recombined)


def _audit_cached(
    cache: tuple[_SampleRows, ...],
    *,
    row_indices: tuple[int, ...],
    projected_against: tuple[int, ...],
    rank_tolerance: float,
    replacement_by_slot: dict[int, str] | None = None,
    primitive_index: dict[str, int] | None = None,
) -> _Audit:
    """Audit a local MINIMUM closure without materializing full Wilson rows.

    The sparse structural-rank certificate is the production fallback when a
    requested audit would escape its local closure.  The dense SVD path is
    deliberately limited to that closure and remains the numerical oracle for
    the local support-choice policy.
    """
    replacements = replacement_by_slot or {}
    if replacements and primitive_index is None:
        raise ValueError("primitive indices are required for row replacements")
    conditions: list[float] = []
    worst_sample = cache[0].label
    full_rank = True
    for sample in cache:
        try:
            rows = _replaced_sparse_rows(
                sample,
                replacements=replacements,
                primitive_index=primitive_index,
            )
            local = _closed_local_audit_rows(
                rows,
                row_indices=row_indices,
                projected_against=projected_against,
            )
            if local is None:
                # The complete candidate remains CSR.  Do not silently create
                # a global dense B merely to obtain a condition number.
                selected = tuple(rows[index] for index in row_indices)
                certificate = certify_minimum_sparse_rows(
                    SparseBMatrix(
                        rows=selected,
                        column_count=sample.gic_rows.column_count,
                        backend="smith-minimum-local-rank.v1",
                    ),
                    expected_rank=len(row_indices),
                )
                condition = 0.0 if certificate.status == "PASS" else float("inf")
                rank = len(row_indices) if certificate.status == "PASS" else 0
            else:
                condition, rank = _normalized_condition(local, rank_tolerance=rank_tolerance)
        except (FloatingPointError, ValueError, GICForgeContractError):
            condition, rank = float("inf"), 0
        conditions.append(condition)
        if condition >= max(conditions):
            worst_sample = sample.label
        full_rank = full_rank and rank == len(row_indices)
    return _Audit(
        maximum_condition=max(conditions),
        mean_condition=float(np.mean(conditions)),
        worst_sample=worst_sample,
        full_rank=full_rank,
    )


def _replaced_sparse_rows(
    sample: _SampleRows,
    *,
    replacements: dict[int, str],
    primitive_index: dict[str, int] | None,
) -> tuple[SparseBRow, ...]:
    """Return GIC rows with optional support substitutions, still in CSR form."""

    rows = list(sample.gic_rows.rows)
    for slot, primitive_id in replacements.items():
        if primitive_index is None:
            raise ValueError("primitive indices are required for row replacements")
        rows[slot] = sample.primitive_rows.rows[primitive_index[primitive_id]]
    return tuple(rows)


def _closed_local_audit_rows(
    rows: tuple[SparseBRow, ...],
    *,
    row_indices: tuple[int, ...],
    projected_against: tuple[int, ...],
) -> np.ndarray | None:
    """Return one support-closed dense microblock, or ``None`` if nonlocal.

    A fixed row that touches the active support must be included completely;
    repeatedly growing this closure makes projection equivalent to the former
    full-row calculation while avoiding unrelated Cartesian directions.
    """

    selected = tuple(rows[index] for index in row_indices)
    fixed_candidates = tuple(rows[index] for index in projected_against)
    active = {column for row in selected for column, _value in row.entries}
    relevant: tuple[SparseBRow, ...] = ()
    while active:
        relevant = tuple(
            row
            for row in fixed_candidates
            if any(column in active for column, _value in row.entries)
        )
        expanded = active | {
            column for row in relevant for column, _value in row.entries
        }
        if expanded == active:
            break
        active = expanded
    columns = tuple(sorted(active))
    if (
        len(selected) > _MAX_LOCAL_DENSE_AUDIT_ROWS
        or len(columns) > _MAX_LOCAL_DENSE_AUDIT_COLUMNS
    ):
        return None
    column_position = {column: position for position, column in enumerate(columns)}

    def dense(local_rows: tuple[SparseBRow, ...]) -> np.ndarray:
        matrix = np.zeros((len(local_rows), len(columns)), dtype=float)
        for row_index, row in enumerate(local_rows):
            for column, value in row.entries:
                position = column_position.get(column)
                if position is not None:
                    matrix[row_index, position] = value
        return matrix

    matrix = dense(selected)
    if relevant:
        fixed = dense(relevant)
        matrix = matrix - matrix @ np.linalg.pinv(fixed) @ fixed
    return matrix


def _select_global_safe_support(
    candidates: tuple[_SupportCandidate, ...],
    *,
    cache: tuple[_SampleRows, ...],
    support_slots: tuple[int, ...],
    primitive_index: dict[str, int],
    gic_count: int,
    rank_tolerance: float,
) -> tuple[_SupportCandidate, _Audit]:
    best: tuple[_SupportCandidate, _Audit] | None = None
    for item in sorted(
        candidates,
        key=lambda candidate: (
            candidate.inter_audit.maximum_condition,
            candidate.inter_audit.mean_condition,
            candidate.sparse_score,
            candidate.locality,
            candidate.primitive_ids,
        ),
    ):
        audit = _audit_cached(
            cache,
            row_indices=tuple(range(gic_count)),
            projected_against=(),
            rank_tolerance=rank_tolerance,
            replacement_by_slot=dict(zip(support_slots, item.primitive_ids, strict=True)),
            primitive_index=primitive_index,
        )
        if audit.full_rank and (
            best is None
            or audit.maximum_condition < best[1].maximum_condition
            or (
                audit.maximum_condition == best[1].maximum_condition
                and item.primitive_ids < best[0].primitive_ids
            )
        ):
            best = (item, audit)
    if best is not None:
        if best[1].maximum_condition <= MAX_NORMALIZED_SONIC_CONDITION:
            return best
        raise GICForgeContractError(
            "LOCAL_NONREDUNDANT_DOMAIN_UNSAFE: best complete MINIMUM chart has "
            f"condition {best[1].maximum_condition:.12g}, above "
            f"{MAX_NORMALIZED_SONIC_CONDITION:.12g}"
        )
    raise GICForgeContractError(
        "LOCAL_NONREDUNDANT_DOMAIN_UNSAFE: no MINIMUM interfragment chart "
        "passes the global numerical/sparse rank gate"
    )


def _sparse_local_support_audit(
    cache: tuple[_SampleRows, ...],
    *,
    support_slots: tuple[int, ...],
    primitive_ids: tuple[str, ...],
    inter_indices: tuple[int, ...],
    internal_indices: tuple[int, ...],
    primitive_index: dict[str, int],
) -> tuple[_Audit, int]:
    """Certify a support substitution by sparse rank extension alone.

    The candidate is admissible when its interfragment rows add their full
    dimensionality to the frozen internal span.  This is the production
    selector: it has no numerical decomposition and therefore cannot turn a
    large MINIMUM chart into a dense conditioning problem.  The returned
    score is a deterministic locality proxy (the number of nonzero Cartesian
    derivative entries); smaller supports are preferred.  The SVD equivalent
    remains below as an explicit audit oracle, not a selection dependency.
    """

    replacements = dict(zip(support_slots, primitive_ids, strict=True))
    full_rank = True
    worst_sample = cache[0].label
    score = 0
    for sample in cache:
        rows = _replaced_sparse_rows(
            sample,
            replacements=replacements,
            primitive_index=primitive_index,
        )
        inter = tuple(rows[index] for index in inter_indices)
        inter_matrix = SparseBMatrix(
            rows=inter,
            column_count=sample.gic_rows.column_count,
            backend="smith-minimum-local-inter-rank.v1",
        )
        if inter_matrix.structural_rank() < len(inter):
            full_rank = False
            worst_sample = sample.label
        score += sum(row.nnz for row in inter)
    return (
        _Audit(
            maximum_condition=0.0 if full_rank else float("inf"),
            mean_condition=0.0 if full_rank else float("inf"),
            worst_sample=worst_sample,
            full_rank=full_rank,
        ),
        score,
    )


def _audit_support_candidates(
    cache: tuple[_SampleRows, ...],
    *,
    selections: tuple[tuple[str, ...], ...],
    support_slots: tuple[int, ...],
    inter_indices: tuple[int, ...],
    internal_indices: tuple[int, ...],
    primitive_index: dict[str, int],
    rank_tolerance: float,
) -> tuple[_Audit, ...]:
    return tuple(
        _audit_cached(
            cache,
            row_indices=inter_indices,
            projected_against=internal_indices,
            rank_tolerance=rank_tolerance,
            replacement_by_slot=dict(zip(support_slots, selection, strict=True)),
            primitive_index=primitive_index,
        )
        for selection in selections
    )


def _sparse_global_rank_audit(
    cache: tuple[_SampleRows, ...],
    *,
    expected_rank: int,
    replacement_by_slot: dict[int, str] | None = None,
    primitive_index: dict[str, int] | None = None,
) -> _Audit:
    """Certify the complete chart in CSR form, without a global dense SVD."""

    replacements = replacement_by_slot or {}
    full_rank = True
    worst_sample = cache[0].label
    for sample in cache:
        rows = _replaced_sparse_rows(
            sample,
            replacements=replacements,
            primitive_index=primitive_index,
        )
        certificate = certify_minimum_sparse_rows(
            SparseBMatrix(
                rows=rows,
                column_count=sample.gic_rows.column_count,
                backend="smith-minimum-global-rank.v1",
            ),
            expected_rank=expected_rank,
        )
        if certificate.status != "PASS":
            full_rank = False
            worst_sample = sample.label
    return _Audit(
        maximum_condition=0.0 if full_rank else float("inf"),
        mean_condition=0.0 if full_rank else float("inf"),
        worst_sample=worst_sample,
        full_rank=full_rank,
    )


def _normalized_condition(
    rows: np.ndarray,
    *,
    rank_tolerance: float,
) -> tuple[float, int]:
    return normalized_dense_condition_and_rank(
        rows, rank_tolerance=rank_tolerance
    )


def _internal_gic_indices(definition: GICDefinition) -> tuple[int, ...]:
    return tuple(
        index for index, gic in enumerate(definition.gics) if gic.family not in _INTER_FAMILIES
    )




def _selected_support_slots(
    definition: GICDefinition,
    *,
    pair: tuple[str, str] | None = None,
) -> tuple[int, ...]:
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    slots = []
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        primitives = tuple(primitive_by_id[item[0]] for item in coefficients)
        if primitives and all(
            item.refs
            and item.refs[0] == _CONTACT_SUPPORT_REF
            and (pair is None or _support_pair(item) == pair)
            for item in primitives
        ):
            slots.append(index)
    return tuple(slots)


def _selected_support_is_component_complete(
    definition: GICDefinition,
    *,
    support_slots: tuple[int, ...],
    support_pool: tuple[GICPrimitive, ...],
) -> bool:
    try:
        primitive_index = {
            primitive.identifier: index for index, primitive in enumerate(support_pool)
        }
        transform = np.zeros((len(support_pool), len(support_slots)))
        for column, slot in enumerate(support_slots):
            gic = definition.gics[slot]
            for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
                transform[primitive_index[primitive_id], column] += float(coefficient)
        components = tuple(
            CoordinateComponent(
                operator=primitive.function,
                atoms=primitive.atoms,
                mode=primitive.mode,
                ref_atoms=primitive.ref_atoms,
                context=(primitive.family, *primitive.refs),
            )
            for primitive in support_pool
        )
        validate_coordinate_component_transform(components, transform)
    except (KeyError, ValueError):
        return False
    return bool(support_slots)


def _support_pair(primitive: GICPrimitive) -> tuple[str, str]:
    refs = primitive.refs[1:] if primitive.refs[:1] == (_CONTACT_SUPPORT_REF,) else primitive.refs
    fragments = tuple(sorted(ref for ref in refs if ref.startswith("F")))
    if len(fragments) != 2:
        raise GICForgeContractError(
            f"contact-support primitive {primitive.identifier} lacks one fragment pair"
        )
    return fragments


def _is_support_candidate_for_pair(
    primitive: GICPrimitive,
    pair: tuple[str, str],
) -> bool:
    if not primitive.refs:
        return False
    is_contact = primitive.refs[0] == _CONTACT_SUPPORT_REF
    is_pose = primitive.family in {
        "FRAG_DISTANCE",
        "FRAG_TRANSLATION",
        "FRAG_ORIENTATION",
    } and primitive.function in {
        "FC_DIST",
        "FCA_DIST",
        "FTRANS",
        "FLIN_TRANS",
        "FROT",
        "FAXIS",
    }
    return (is_contact or is_pose) and _support_pair(primitive) == pair


def _required_contact_slots(
    definition: GICDefinition,
    *,
    pair: tuple[str, str],
    fragment_by_atom: dict[int, str],
) -> tuple[int, ...]:
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    slots: list[int] = []
    for index, gic in enumerate(definition.gics):
        if gic.family != "PSEUDO_BOND_DISTANCE":
            continue
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        owners = {
            tuple(sorted(fragment_by_atom[atom] for atom in primitive_by_id[primitive_id].atoms))
            for primitive_id, _coefficient in coefficients
        }
        if owners == {pair}:
            slots.append(index)
    return tuple(slots)


def _combine_audits(audits: list[_Audit]) -> _Audit:
    if not audits:
        raise GICForgeContractError("a pseudobond chart has no fragment-pair audit")
    worst = max(audits, key=lambda item: item.maximum_condition)
    return _Audit(
        maximum_condition=worst.maximum_condition,
        mean_condition=float(np.mean([item.mean_condition for item in audits])),
        worst_sample=worst.worst_sample,
        full_rank=all(item.full_rank for item in audits),
    )


def _support_ids(
    definition: GICDefinition,
    support_slots: tuple[int, ...],
) -> tuple[str, ...]:
    return tuple(
        (definition.gics[index].coefficients or ((definition.gics[index].primitive_id, 1.0),))[0][0]
        for index in support_slots
    )


def _substitution_blocks(
    contract: OracleCoordinateAtlasContract,
    *,
    prefix: str,
) -> dict[str, object]:
    return {
        item.block_id: item
        for item in contract.family_compatibility
        if item.substitutions
        and item.block_id.startswith(prefix)
        and item.condition_trigger is not None
        and item.minimum_relative_gain is not None
    }


def _contact_condition_policy(
    contract: OracleCoordinateAtlasContract,
) -> tuple[float, float]:
    matches = tuple(
        item
        for item in contract.family_compatibility
        if item.block_id == "MINIMUM_CONTACT::PSEUDOBOND" and item.substitutions
    )
    if len(matches) != 1:
        raise GICForgeContractError(
            "ORACLE atlas must define one MINIMUM pseudobond compatibility block"
        )
    item = matches[0]
    if item.condition_trigger is None or item.minimum_relative_gain is None:
        raise GICForgeContractError("ORACLE MINIMUM pseudobond block lacks conditioning thresholds")
    return float(item.condition_trigger), float(item.minimum_relative_gain)


def _primitive_prescriptions(
    definition: GICDefinition,
    *,
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
) -> dict[str, _PrimitiveAtlasEntry]:
    atlas_by_candidate_id = {item.candidate_id: item for item in contract.candidates}
    oracle_by_signature = {
        (item.function, item.atoms, item.mode, item.ref_atoms): item
        for item in sonic_contract.primitive_candidates
    }
    prescriptions: dict[str, _PrimitiveAtlasEntry] = {}
    for primitive in definition.primitives:
        candidate = oracle_by_signature.get(
            (primitive.function, primitive.atoms, primitive.mode, primitive.ref_atoms)
        )
        if candidate is None:
            continue
        prescription = atlas_by_candidate_id.get(candidate.candidate_id)
        if prescription is not None:
            prescriptions[primitive.identifier] = _PrimitiveAtlasEntry(
                prescription=prescription,
                domain_id=candidate.domain_id,
            )
    return prescriptions


def _eligible_internal_exchanges(
    definition: GICDefinition,
    *,
    primitive_by_id: dict[str, GICPrimitive],
    prescription_by_id: dict[str, _PrimitiveAtlasEntry],
    substitution_blocks: frozenset[str],
) -> tuple[_InternalExchange, ...]:
    selected_ids = {
        primitive_id
        for gic in definition.gics
        for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
    }
    alternatives: dict[tuple[str, str, str], list[str]] = {}
    for primitive in definition.primitives:
        entry = prescription_by_id.get(primitive.identifier)
        if (
            entry is None
            or entry.prescription.requirement != CANDIDATE_OPTIONAL
            or entry.prescription.mixing_block not in substitution_blocks
        ):
            continue
        alternatives.setdefault(
            (
                entry.prescription.mixing_block,
                primitive.family,
                entry.domain_id,
            ),
            [],
        ).append(primitive.identifier)
    exchanges: list[_InternalExchange] = []
    for slot, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1 or abs(float(coefficients[0][1]) - 1.0) > 1.0e-12:
            continue
        source_id = coefficients[0][0]
        source = primitive_by_id[source_id]
        if source.family in _INTER_FAMILIES:
            continue
        entry = prescription_by_id.get(source_id)
        if entry is None or entry.prescription.mixing_block not in substitution_blocks:
            continue
        for alternative_id in alternatives.get(
            (
                entry.prescription.mixing_block,
                source.family,
                entry.domain_id,
            ),
            (),
        ):
            if alternative_id == source_id or alternative_id in selected_ids:
                continue
            exchanges.append(
                _InternalExchange(
                    slot=slot,
                    source_id=source_id,
                    alternative_id=alternative_id,
                    block_id=entry.prescription.mixing_block,
                )
            )
    return tuple(
        sorted(
            exchanges,
            key=lambda item: (item.slot, item.alternative_id, item.block_id),
        )
    )


def _compatible_exchange_set(exchanges: tuple[_InternalExchange, ...]) -> bool:
    return len({item.slot for item in exchanges}) == len(exchanges) and len(
        {item.alternative_id for item in exchanges}
    ) == len(exchanges)


def _apply_internal_exchanges(
    definition: GICDefinition,
    exchanges: tuple[_InternalExchange, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
) -> GICDefinition:
    gics = list(definition.gics)
    replacements: dict[str, str] = {}
    for exchange in exchanges:
        source = gics[exchange.slot]
        primitive = primitive_by_id[exchange.alternative_id]
        replacements[exchange.source_id] = exchange.alternative_id
        gics[exchange.slot] = replace(
            source,
            name=f"{source.irrep}{primitive.name}",
            family=primitive.family,
            primitive_id=primitive.identifier,
            gaussian_expression=primitive.gaussian_expression(),
            coefficients=((primitive.identifier, 1.0),),
        )
    diagnostics = _replace_reduction_selection(
        definition.reduction_diagnostics,
        replacements=replacements,
        primitive_by_id=primitive_by_id,
    )
    return replace(definition, gics=tuple(gics), reduction_diagnostics=diagnostics)


def _replace_reduction_selection(
    diagnostics: GICReductionDiagnostics | None,
    *,
    replacements: dict[str, str],
    primitive_by_id: dict[str, GICPrimitive],
) -> GICReductionDiagnostics | None:
    if diagnostics is None or not replacements:
        return diagnostics
    selected = tuple(replacements.get(item, item) for item in diagnostics.selected)
    skipped = set(diagnostics.skipped_dependent)
    skipped.difference_update(replacements.values())
    skipped.update(replacements)
    details = tuple(
        f"{identifier}:{primitive_by_id[identifier].family}:{primitive_by_id[identifier].name}"
        for identifier in sorted(skipped)
    )
    return replace(
        diagnostics,
        selected=selected,
        skipped_dependent=tuple(sorted(skipped)),
        skipped_dependent_details=details,
    )


def _synchronize_reduction_selection(
    definition: GICDefinition,
    *,
    primitive_by_id: dict[str, GICPrimitive],
) -> GICReductionDiagnostics | None:
    diagnostics = definition.reduction_diagnostics
    if diagnostics is None:
        return None
    support_pool = {
        identifier
        for identifier, primitive in primitive_by_id.items()
        if primitive.refs and primitive.refs[0] == _CONTACT_SUPPORT_REF
    }
    active_support = tuple(
        dict.fromkeys(
            primitive_id
            for gic in definition.gics
            for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            if primitive_id in support_pool
        )
    )
    previous = list(diagnostics.selected)
    insertion = next(
        (index for index, identifier in enumerate(previous) if identifier in support_pool),
        len(previous),
    )
    retained = [identifier for identifier in previous if identifier not in support_pool]
    retained[insertion:insertion] = active_support
    selected = tuple(retained)
    skipped_set = set(diagnostics.skipped_dependent)
    skipped_set.update(support_pool.difference(active_support))
    skipped_set.difference_update(active_support)
    skipped = tuple(sorted(skipped_set))
    details = tuple(
        f"{identifier}:{primitive_by_id[identifier].family}:{primitive_by_id[identifier].name}"
        for identifier in skipped
    )
    return replace(
        diagnostics,
        selected=selected,
        skipped_dependent=skipped,
        skipped_dependent_details=details,
    )


def _material_gain(
    baseline: float,
    candidate: float,
    *,
    minimum_gain: float,
) -> bool:
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(baseline):
        return True
    return (baseline - candidate) / max(baseline, 1.0e-12) >= minimum_gain


def _support_locality(
    primitive_ids: tuple[str, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    sonic_contract: OracleSonicContract,
    contract: OracleCoordinateAtlasContract,
    pair: tuple[str, str],
) -> int:
    contact_atoms = {
        int(endpoint[1])
        for item in contract.interactions
        if item.pseudobond_policy == PSEUDOBOND_REQUIRED
        and tuple(sorted(item.fragment_ids)) == pair
        for endpoint in (item.endpoint_a, item.endpoint_b)
        if endpoint[0] == "ATOM"
    }
    adjacency = _adjacency(
        sonic_contract.primary_topology.natoms,
        sonic_contract.primary_topology.bonds,
    )
    distances = _graph_distances(contact_atoms, adjacency)
    score = 0
    for primitive_id in primitive_ids:
        primitive = primitive_by_id[primitive_id]
        score += sum(distances.get(atom, len(adjacency)) for atom in primitive.atoms)
        score += sum(distances.get(atom, len(adjacency)) for atom in primitive.ref_atoms)
    return score


def _adjacency(
    natoms: int,
    bonds: tuple[tuple[int, int], ...],
) -> dict[int, set[int]]:
    adjacency = {atom: set() for atom in range(1, natoms + 1)}
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _graph_distances(
    roots: set[int],
    adjacency: dict[int, set[int]],
) -> dict[int, int]:
    distances = {root: 0 for root in roots}
    frontier = list(sorted(roots))
    for atom in frontier:
        for neighbor in sorted(adjacency.get(atom, ())):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[atom] + 1
            frontier.append(neighbor)
    return distances


def _with_partition_diagnostics(
    definition: GICDefinition,
    *,
    sample_count: int,
    internal_exchange: tuple[str, ...],
    support_ids: tuple[str, ...],
    intra_before: _Audit,
    intra_after: _Audit,
    inter_before: _Audit,
    inter_after: _Audit,
    global_after: _Audit,
) -> GICDefinition:
    status = "SEPARATED_SELECTION" if internal_exchange else "SEPARATED_CANONICAL_INTRA"
    diagnostics = tuple(
        item
        for item in definition.semantic_diagnostics
        if not item.startswith("MINIMUM_LOCAL_DOMAIN ")
        and not item.startswith("MINIMUM_DOMAIN_PARTITION ")
    )
    diagnostics += (
        f"MINIMUM_DOMAIN_PARTITION SCHEMA={MINIMUM_DOMAIN_SELECTION_SCHEMA} "
        f"STATUS={status} L_COMPONENTS=COMPLETE",
        f"MINIMUM_DOMAIN_PARTITION INTRA_SAMPLES=1 "
        f"BASE_MAX_K={intra_before.maximum_condition:.12g} "
        f"SELECTED_MAX_K={intra_after.maximum_condition:.12g} "
        f"EXCHANGES={','.join(internal_exchange) if internal_exchange else 'NONE'}",
        f"MINIMUM_DOMAIN_PARTITION INTER_SAMPLES={sample_count} "
        f"BASE_MAX_K={inter_before.maximum_condition:.12g} "
        f"SELECTED_MAX_K={inter_after.maximum_condition:.12g} "
        f"WORST_SAMPLE={inter_after.worst_sample} "
        f"SUPPORT={','.join(support_ids)}",
        f"MINIMUM_DOMAIN_PARTITION GLOBAL_MAX_K={global_after.maximum_condition:.12g} "
        f"GLOBAL_WORST_SAMPLE={global_after.worst_sample}",
    )
    return replace(definition, semantic_diagnostics=diagnostics)


__all__ = ["MINIMUM_DOMAIN_SELECTION_SCHEMA", "condition_minimum_pseudobond_chart"]
