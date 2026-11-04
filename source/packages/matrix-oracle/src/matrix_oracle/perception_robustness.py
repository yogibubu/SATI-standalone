"""Deterministic, auditable robustness analysis for ORACLE perception.

This module perturbs Cartesian geometries but compares canonical chemical
state identities.  It never averages incompatible topologies or silently
promotes quasi-symmetry.  SMITH is intentionally absent from the dependency
graph: ORACLE alone produces and certifies these semantic decisions.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from time import perf_counter
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from matrix_numerics import numerical_matrix_rank

from matrix_chem import (
    DecisionRobustness,
    DecisionThreshold,
    FragmentMembership,
    LocalPerceptionSettings,
    MolecularGeometry,
    PerceptionNoiseAudit,
    PerceptionNoiseSettings,
    PrimaryTopology,
    StructuralSite,
    SymmetryThresholds,
    analyze_molecular_quasisymmetry,
    build_primitives,
    build_topology_objects,
    graph_cycle_rank,
    primary_topology_hash,
)
from matrix_chem.topology.automorphisms import topology_automorphism_edge_labels
from matrix_chem.topology.elements import atomic_symbol
from matrix_chem.average_atomic_masses import atomic_mass

from .atom_classes import SynthonAtomClassThresholds, classify_synthon_atoms
from .auxiliary_contacts import (
    AuxiliaryContactProviderSettings,
    StructuralSiteContactRequest,
    perceive_auxiliary_contact_evidence,
)
from .contact_graph import complete_and_classify_contact_orbits
from .local_perception import perceive_local_perception_domains
from .multicenter_domains import perceive_multicenter_domains


ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER = "ORACLE_PERCEPTION_ROBUSTNESS_AUDIT"
ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER_VERSION = "1"


@dataclass(frozen=True)
class PerceptionAuditPolicy:
    """Frozen policy for one deterministic single-geometry audit."""

    minimum_stability_fraction: float = 0.95
    symmetry_thresholds: SymmetryThresholds = SymmetryThresholds()
    quasi_symmetry_search_limit_angstrom: float = 3.0e-2
    atom_class_thresholds: SynthonAtomClassThresholds = SynthonAtomClassThresholds()
    local_perception_settings: LocalPerceptionSettings = LocalPerceptionSettings()
    contact_settings: AuxiliaryContactProviderSettings = AuxiliaryContactProviderSettings()

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_stability_fraction <= 1.0:
            raise ValueError("minimum stability fraction must lie in (0, 1]")
        if (
            not math.isfinite(self.quasi_symmetry_search_limit_angstrom)
            or self.quasi_symmetry_search_limit_angstrom
            < self.symmetry_thresholds.distance_angstrom
        ):
            raise ValueError("quasi-symmetry search limit is inconsistent")


@dataclass(frozen=True)
class OraclePerceptionDecision:
    decision_id: str
    family: str
    accepted_class: str
    competing_class: str | None
    raw_score: float | None
    normalized_score: float | None
    signed_margin: float | None
    thresholds: tuple[DecisionThreshold, ...]
    provider: str
    provider_version: str


@dataclass(frozen=True)
class OraclePerceptionState:
    """Canonical identity and evidence for one ORACLE perception pass."""

    topology_hash: str
    state_hash: str
    strict_group: str
    proposed_group: str
    strict_operation_signatures: tuple[str, ...]
    proposed_operation_signatures: tuple[str, ...]
    fragment_signatures: tuple[str, ...]
    ring_signatures: tuple[str, ...]
    atom_class_signatures: tuple[str, ...]
    local_signatures: tuple[str, ...]
    structural_site_signatures: tuple[str, ...]
    multicenter_signatures: tuple[str, ...]
    contact_signatures: tuple[str, ...]
    primitive_signatures: tuple[str, ...]
    transition_signatures: tuple[str, ...]
    primary_cycle_rank: int
    auxiliary_cycle_rank: int
    decisions: tuple[OraclePerceptionDecision, ...]


@dataclass(frozen=True)
class OraclePerceptionFailure:
    """Deterministic certificate for a failed perturbed perception pass."""

    topology_hash: str
    state_hash: str
    decisions: tuple[OraclePerceptionDecision, ...]
    error_type: str
    error_message: str


def audit_perception_robustness(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    noise_settings: PerceptionNoiseSettings,
    *,
    policy: PerceptionAuditPolicy | None = None,
    symmetry_decision: str | None = None,
    structural_sites: Sequence[StructuralSite] = (),
    configured_site_contacts: Iterable[StructuralSiteContactRequest] = (),
    parallel_workers: int = 1,
) -> PerceptionNoiseAudit:
    """Audit whether ORACLE's proposed state survives declared uncertainty.

    ``symmetry_decision`` is explicit: ``PROJECT`` and ``RETAIN`` are accepted;
    omission yields ``REQUIRES_DECISION`` whenever quasi-symmetry proposes a
    larger group.  The audit never projects coordinates itself.
    """

    numbers, xyz = _validated_geometry(atomic_numbers, coordinates_angstrom)
    if noise_settings.model.natoms != len(numbers):
        raise ValueError("noise model atom count does not match the geometry")
    options = policy or PerceptionAuditPolicy()
    site_records = tuple(structural_sites)
    contact_requests = tuple(configured_site_contacts)
    workers = int(parallel_workers)
    if workers < 1:
        raise ValueError("parallel_workers must be positive")

    started = perf_counter()
    reference = perceive_oracle_state(
        numbers,
        xyz,
        policy=options,
        structural_sites=site_records,
        configured_site_contacts=contact_requests,
    )
    ordinary_runtime = perf_counter() - started
    perturbations = deterministic_cartesian_perturbations(xyz, numbers, noise_settings)
    ensemble_started = perf_counter()
    tasks = tuple(
        (numbers, xyz + displacement, options, site_records, contact_requests)
        for displacement in perturbations
    )
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            sampled = tuple(executor.map(_perception_audit_worker, tasks))
    else:
        sampled = tuple(_perception_audit_worker(task) for task in tasks)
    ensemble_runtime = perf_counter() - ensemble_started
    fragment_loss = any(
        isinstance(state, OraclePerceptionState)
        and len(state.fragment_signatures) != len(reference.fragment_signatures)
        for state in sampled
    )
    if fragment_loss:
        sampled = tuple(
            _topology_failure(reference, state)
            if isinstance(state, OraclePerceptionState)
            else state
            for state in sampled
        )
    failures = tuple(state for state in sampled if isinstance(state, OraclePerceptionFailure))

    decision_records, worst_case = _compare_decisions(
        reference,
        sampled,
        minimum_stability=options.minimum_stability_fraction,
    )
    topology_transition = any(
        isinstance(state, OraclePerceptionState)
        and (
            state.topology_hash != reference.topology_hash
            or state.transition_signatures != reference.transition_signatures
        )
        for state in sampled
    )
    unstable = tuple(
        item
        for item in decision_records
        if item.stability_fraction < options.minimum_stability_fraction
    )
    explicit_symmetry = _symmetry_decision(reference, symmetry_decision)
    if failures:
        status = "FAILED"
        handoff = "EXPLORATION"
    elif topology_transition:
        status = "TOPOLOGY_TRANSITION"
        handoff = "EXPLORATION"
    elif explicit_symmetry == "REQUIRES_DECISION":
        status = "REQUIRES_DECISION"
        handoff = "REQUIRES_DECISION"
    elif unstable:
        status = "AMBIGUOUS"
        handoff = "EXPLORATION"
    else:
        status = "ROBUST"
        handoff = "PROPOSED"
    diagnostics = (
        "STATE_COMPARISON CANONICAL_IDENTITIES_NOT_FLOATING_OBJECT_ORDER",
        "PERSISTENCE_FIELD SINGLE_GEOMETRY_SCORE_UNCHANGED",
        "QUASI_SYMMETRY_PROMOTION NEVER_AUTOMATIC",
        f"UNSTABLE_DECISIONS {len(unstable)}",
        f"TOPOLOGY_TRANSITION {str(topology_transition).upper()}",
        f"FAILED_PERTURBATIONS {len(failures)}",
        f"PARALLEL_WORKERS {min(workers, len(tasks))}",
        *tuple(
            f"FAILURE {item.error_type}:{item.error_message}" for item in failures
        ),
    )
    return PerceptionNoiseAudit(
        status=status,
        noise_settings=noise_settings,
        reference_state_hash=reference.state_hash,
        sampled_state_hashes=tuple(state.state_hash for state in sampled),
        decisions=decision_records,
        strict_group=reference.strict_group,
        proposed_group=reference.proposed_group,
        symmetry_decision=explicit_symmetry,
        handoff_status=handoff,
        worst_case_perturbation=worst_case,
        ordinary_runtime_seconds=float(ordinary_runtime),
        ensemble_runtime_seconds=float(ensemble_runtime),
        diagnostics=diagnostics,
    )


def _safe_perceive_oracle_state(*args, **kwargs) -> OraclePerceptionState | OraclePerceptionFailure:
    try:
        return perceive_oracle_state(*args, **kwargs)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        certificate = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        digest = _stable_sha256(certificate)
        return OraclePerceptionFailure(
            topology_hash=f"FAILED:{digest}",
            state_hash=f"FAILED:{digest}",
            decisions=(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _topology_failure(
    reference: OraclePerceptionState,
    state: OraclePerceptionState,
) -> OraclePerceptionFailure:
    certificate = {
        "error_type": "TopologyPersistenceError",
        "reference_topology": reference.topology_hash,
        "sampled_topology": state.topology_hash,
        "reference_fragments": len(reference.fragment_signatures),
        "sampled_fragments": len(state.fragment_signatures),
    }
    digest = _stable_sha256(certificate)
    return OraclePerceptionFailure(
        topology_hash=f"FAILED:{digest}",
        state_hash=f"FAILED:{digest}",
        decisions=state.decisions,
        error_type="TopologyPersistenceError",
        error_message="perturbation changed the number of perceived fragments",
    )


def _perception_audit_worker(task) -> OraclePerceptionState | OraclePerceptionFailure:
    numbers, xyz, policy, structural_sites, configured_site_contacts = task
    return _safe_perceive_oracle_state(
        numbers,
        xyz,
        policy=policy,
        structural_sites=structural_sites,
        configured_site_contacts=configured_site_contacts,
    )


def perceive_oracle_state(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    *,
    policy: PerceptionAuditPolicy | None = None,
    structural_sites: Sequence[StructuralSite] = (),
    configured_site_contacts: Iterable[StructuralSiteContactRequest] = (),
    candidate_pairs: Sequence[tuple[int, int]] | None = None,
) -> OraclePerceptionState:
    """Evaluate one complete in-memory ORACLE semantic state."""

    numbers, xyz = _validated_geometry(atomic_numbers, coordinates_angstrom)
    options = policy or PerceptionAuditPolicy()
    continuous, discrete, ringset, synthons, aromaticity = build_topology_objects(
        xyz, numbers, candidate_pairs=candidate_pairs
    )
    bonds = tuple(sorted(tuple(sorted(map(int, pair))) for pair in discrete.bonds))
    rings = tuple(
        sorted(tuple(int(atom) for atom in ring.atoms) for ring in ringset.rings)
    )
    fragments_zero = _connected_components(len(numbers), bonds)
    topology_hash = _isomorphism_invariant_topology_hash(numbers, bonds)
    cycle_rank = graph_cycle_rank(
        len(numbers), tuple((left + 1, right + 1) for left, right in bonds)
    )
    symbols, _masses = _symbols_and_masses(numbers)
    geometry = MolecularGeometry(symbols, xyz, source_format="ORACLE_IN_MEMORY_AUDIT")
    edge_labels = topology_automorphism_edge_labels(
        discrete, synthons, aromaticity=aromaticity
    )
    quasi = analyze_molecular_quasisymmetry(
        geometry,
        distance_tolerance=options.symmetry_thresholds.distance_angstrom,
        inertia_tolerance=options.symmetry_thresholds.inertia_relative,
        max_rotation_order=options.symmetry_thresholds.max_rotation_order,
        topology_bonds=bonds,
        topology_edge_labels=edge_labels,
        maximum_distance_tolerance=options.quasi_symmetry_search_limit_angstrom,
    )
    strict_permutations = tuple(
        tuple(int(atom) for atom in operation.permutation)
        for operation in quasi.strict_symmetry.operations
    )
    fragments = tuple(
        FragmentMembership(f"F{index:03d}", tuple(atom + 1 for atom in component))
        for index, component in enumerate(fragments_zero, start=1)
    )
    primary = PrimaryTopology(
        natoms=len(numbers),
        atomic_numbers=numbers,
        bonds=tuple((left + 1, right + 1) for left, right in bonds),
        rings=tuple(tuple(atom + 1 for atom in ring) for ring in rings),
        fragments=fragments,
        topology_hash=primary_topology_hash(
            numbers,
            tuple((left + 1, right + 1) for left, right in bonds),
            tuple(tuple(atom + 1 for atom in ring) for ring in rings),
            fragments,
        ),
        cycle_rank=cycle_rank,
        symmetry_permutations=strict_permutations,
    )
    effective = tuple(float(synthons.Zeff(atom)) for atom in range(len(numbers)))
    local_domains = perceive_local_perception_domains(
        numbers,
        xyz,
        bonds,
        rings,
        effective_atomic_numbers=effective,
        settings=options.local_perception_settings,
    )
    atom_classes = classify_synthon_atoms(
        _synthon_records(symbols, synthons), options.atom_class_thresholds
    )
    multicenter, multicenter_candidates = perceive_multicenter_domains(
        numbers, xyz, bonds
    )
    sites = tuple(structural_sites)
    evidence = perceive_auxiliary_contact_evidence(
        numbers,
        xyz,
        bonds,
        settings=options.contact_settings,
        configured_site_contacts=tuple(configured_site_contacts),
    )
    contacts = complete_and_classify_contact_orbits(
        evidence,
        primary,
        structural_sites=sites,
    )

    signatures = _perception_signature_bundle(
        numbers=numbers,
        xyz=xyz,
        bonds=bonds,
        rings=rings,
        fragments_zero=fragments_zero,
        atom_classes=atom_classes,
        local_domains=local_domains,
        sites=sites,
        multicenter=multicenter,
        multicenter_candidates=multicenter_candidates,
        contacts=contacts.contacts,
        discrete=discrete,
        quasi=quasi,
    )
    return _finalize_perception_state(
        numbers=numbers,
        topology_hash=topology_hash,
        cycle_rank=cycle_rank,
        auxiliary_cycle_rank=contacts.auxiliary_cycle_rank,
        options=options,
        local_domains=local_domains,
        sites=sites,
        contacts=contacts.contacts,
        quasi=quasi,
        signatures=signatures,
    )


def _perception_signature_bundle(
    *,
    numbers,
    xyz,
    bonds,
    rings,
    fragments_zero,
    atom_classes,
    local_domains,
    sites,
    multicenter,
    multicenter_candidates,
    contacts,
    discrete,
    quasi,
) -> dict[str, tuple[str, ...]]:
    """Build canonical semantic signatures without changing perception state."""

    contact_signatures = tuple(
        sorted(_contact_signature(numbers, item) for item in contacts)
    )
    primitive_signatures = tuple(
        sorted(
            [
                _primitive_signature(numbers, primitive.function, primitive.atoms, primitive.kind)
                for primitive in build_primitives(discrete, xyz, include_pseudo_bonds=False)
            ]
            + [
                f"{candidate.family}:{candidate.function}:"
                f"{_member_element_signature(numbers, candidate.atoms)}"
                for candidate in multicenter_candidates
            ]
            + [f"AUXILIARY_CONTACT:{value}" for value in contact_signatures]
        )
    )
    return {
        "fragments": tuple(
            sorted(_component_signature(numbers, bonds, component) for component in fragments_zero)
        ),
        "rings": tuple(sorted(_ring_signature(numbers, ring) for ring in rings)),
        "atom_classes": tuple(
            sorted(f"{item.element}:{len(item.atoms)}" for item in atom_classes.classes)
        ),
        "local": tuple(sorted(_local_signature(numbers, item) for item in local_domains)),
        "sites": tuple(
            sorted(
                f"{site.kind}:{_member_element_signature(numbers, site.members)}:"
                f"{','.join(site.fragment_ids)}"
                for site in sites
            )
        ),
        "multicenter": tuple(
            sorted(
                f"{domain.kind}:{_member_element_signature(numbers, domain.atoms)}:"
                f"{domain.provider}@{domain.provider_version}"
                for domain in multicenter
            )
        ),
        "contacts": contact_signatures,
        "primitives": primitive_signatures,
        "transitions": tuple(
            sorted(
                [
                    f"BREAKING:{left + 1}-{right + 1}"
                    for left, right in discrete.transitional_contacts
                ]
                + [
                    f"FORMING:{left + 1}-{right + 1}"
                    for left, right in discrete.near_covalent_contacts
                ]
            )
        ),
        "strict_operations": _operation_signatures(quasi.strict_symmetry.operations),
        "proposed_operations": _operation_signatures(quasi.proposed_symmetry.operations),
    }


def _finalize_perception_state(
    *,
    numbers,
    topology_hash,
    cycle_rank,
    auxiliary_cycle_rank,
    options,
    local_domains,
    sites,
    contacts,
    quasi,
    signatures,
) -> OraclePerceptionState:
    """Hash and freeze one already perceived ORACLE state."""

    decisions = _state_decisions(
        atomic_numbers=numbers,
        topology_hash=_stable_sha256(
            {"primary": topology_hash, "transitions": signatures["transitions"]}
        ),
        fragment_signatures=signatures["fragments"],
        ring_signatures=signatures["rings"],
        atom_class_signatures=signatures["atom_classes"],
        local_domains=local_domains,
        local_signatures=signatures["local"],
        sites=sites,
        structural_site_signatures=signatures["sites"],
        multicenter_signatures=signatures["multicenter"],
        contacts=contacts,
        contact_signatures=signatures["contacts"],
        primitive_signatures=signatures["primitives"],
        quasi=quasi,
        cycle_rank=cycle_rank,
        auxiliary_cycle_rank=auxiliary_cycle_rank,
        policy=options,
    )
    identity = {
        "topology": topology_hash,
        "fragments": signatures["fragments"],
        "rings": signatures["rings"],
        "atom_classes": signatures["atom_classes"],
        "local": signatures["local"],
        "strict_symmetry": (
            quasi.strict_symmetry.point_group,
            signatures["strict_operations"],
        ),
        "proposed_symmetry": (
            quasi.proposed_symmetry.point_group,
            signatures["proposed_operations"],
        ),
        "sites": signatures["sites"],
        "multicenter": signatures["multicenter"],
        "contacts": signatures["contacts"],
        "primitive_candidates": signatures["primitives"],
        "cycle_ranks": (cycle_rank, auxiliary_cycle_rank),
    }
    state_hash = _stable_sha256(identity)
    return OraclePerceptionState(
        topology_hash=topology_hash,
        state_hash=state_hash,
        strict_group=quasi.strict_symmetry.point_group,
        proposed_group=quasi.proposed_symmetry.point_group,
        strict_operation_signatures=signatures["strict_operations"],
        proposed_operation_signatures=signatures["proposed_operations"],
        fragment_signatures=signatures["fragments"],
        ring_signatures=signatures["rings"],
        atom_class_signatures=signatures["atom_classes"],
        local_signatures=signatures["local"],
        structural_site_signatures=signatures["sites"],
        multicenter_signatures=signatures["multicenter"],
        contact_signatures=signatures["contacts"],
        primitive_signatures=signatures["primitives"],
        transition_signatures=signatures["transitions"],
        primary_cycle_rank=cycle_rank,
        auxiliary_cycle_rank=auxiliary_cycle_rank,
        decisions=decisions,
    )


def deterministic_cartesian_perturbations(
    coordinates_angstrom: np.ndarray,
    atomic_numbers: Sequence[int],
    settings: PerceptionNoiseSettings,
) -> tuple[np.ndarray, ...]:
    """Generate symmetric constrained perturbations with external modes removed."""

    numbers, xyz = _validated_geometry(atomic_numbers, coordinates_angstrom)
    count = int(settings.perturbation_count)
    if settings.scheme.strip().upper() == "SYMMETRIC_SIGMA_POINTS" and count % 2:
        raise ValueError("symmetric sigma-point count must be even")
    projector = _allowed_internal_projector(xyz, numbers, settings)
    square_root = _uncertainty_square_root(projector, settings)
    rank = numerical_matrix_rank(square_root, absolute_tolerance=1.0e-12)
    if rank < 1:
        raise ValueError("noise model has no allowed internal perturbation direction")
    scheme = settings.scheme.strip().upper()
    if scheme == "SYMMETRIC_SIGMA_POINTS":
        covariance = square_root @ square_root.T
        values, vectors = np.linalg.eigh(covariance)
        active = vectors[:, values > max(1.0e-16, 1.0e-12 * float(values.max()))]
        scales = np.sqrt(values[values > max(1.0e-16, 1.0e-12 * float(values.max()))])
        active = active[:, ::-1]
        scales = scales[::-1]
        half = count // 2
        positive = []
        for index in range(half):
            if index < active.shape[1]:
                vector = active[:, index] * scales[index] * math.sqrt(rank)
            else:
                coefficients = np.cos(
                    (np.arange(active.shape[1], dtype=float) + 1.0)
                    * (index + 1.0)
                    * math.pi
                    / (active.shape[1] + 1.0)
                )
                vector = active @ (scales * coefficients)
                norm = float(np.linalg.norm(coefficients))
                if norm > 0.0:
                    vector *= math.sqrt(rank) / norm
            positive.append(_bound_if_requested(vector, settings))
        flattened = tuple(positive) + tuple(-value for value in positive)
    elif scheme == "FIXED_LOW_DISCREPANCY":
        flattened = tuple(
            _bound_if_requested(
                square_root @ _deterministic_unit_vector(square_root.shape[1], index),
                settings,
            )
            for index in range(count)
        )
    else:
        generator = np.random.default_rng(int(settings.seed))
        flattened = tuple(
            _bound_if_requested(square_root @ generator.normal(size=3 * len(numbers)), settings)
            for _index in range(count)
        )
    return tuple(np.asarray(value, dtype=float).reshape((len(numbers), 3)) for value in flattened)


def _allowed_internal_projector(
    xyz: np.ndarray,
    numbers: tuple[int, ...],
    settings: PerceptionNoiseSettings,
) -> np.ndarray:
    size = 3 * len(numbers)
    if settings.model.constraint_projector:
        allowed = np.asarray(settings.model.constraint_projector, dtype=float)
    else:
        allowed = np.eye(size)
    for atom in settings.model.frozen_atoms:
        start = 3 * (int(atom) - 1)
        allowed[start : start + 3, :] = 0.0
        allowed[:, start : start + 3] = 0.0
    external = []
    if settings.remove_translations:
        for axis in range(3):
            vector = np.zeros((len(numbers), 3), dtype=float)
            vector[:, axis] = 1.0
            external.append(vector.reshape(-1))
    if settings.remove_rotations and len(numbers) > 1:
        _symbols, cached_masses = _symbols_and_masses(numbers)
        masses = np.asarray(cached_masses, dtype=float)
        center = np.average(xyz, axis=0, weights=masses)
        centered = xyz - center
        axes = np.eye(3)
        for axis in axes:
            external.append(np.cross(np.broadcast_to(axis, centered.shape), centered).reshape(-1))
    if external:
        matrix = allowed @ np.column_stack(external)
        u, singular, _vh = np.linalg.svd(matrix, full_matrices=False)
        keep = singular > max(1.0e-12, 1.0e-10 * float(singular.max(initial=0.0)))
        if np.any(keep):
            basis = u[:, keep]
            allowed = allowed - basis @ basis.T
    allowed = 0.5 * (allowed + allowed.T)
    values, vectors = np.linalg.eigh(allowed)
    values = np.where(values > 0.5, 1.0, 0.0)
    return (vectors * values) @ vectors.T


def _uncertainty_square_root(
    projector: np.ndarray,
    settings: PerceptionNoiseSettings,
) -> np.ndarray:
    model = settings.model
    if model.representation.strip().upper() == "COVARIANCE":
        covariance = np.asarray(model.covariance_angstrom2, dtype=float)
        covariance = projector @ covariance @ projector
        values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        values = np.maximum(values, 0.0)
        return (vectors * np.sqrt(values)) @ vectors.T
    return float(model.amplitude_angstrom) * projector


def _bound_if_requested(vector: np.ndarray, settings: PerceptionNoiseSettings) -> np.ndarray:
    output = np.asarray(vector, dtype=float)
    if settings.model.representation.strip().upper() != "BOUND":
        return output
    reshaped = output.reshape((-1, 3))
    maximum = float(np.max(np.linalg.norm(reshaped, axis=1), initial=0.0))
    bound = float(settings.model.amplitude_angstrom)
    return output if maximum <= bound else output * (bound / maximum)


def _deterministic_unit_vector(size: int, index: int) -> np.ndarray:
    values = np.cos(
        (np.arange(size, dtype=float) + 0.5)
        * (index + 1.0)
        * math.pi
        / float(size)
    )
    norm = float(np.linalg.norm(values))
    return values / norm


def _state_decisions(**data) -> tuple[OraclePerceptionDecision, ...]:
    policy: PerceptionAuditPolicy = data["policy"]
    numbers: tuple[int, ...] = data["atomic_numbers"]
    records = [
        _simple_decision("PRIMARY_TOPOLOGY", "TOPOLOGY", data["topology_hash"]),
        _simple_decision("FRAGMENTS", "FRAGMENTS", data["fragment_signatures"]),
        _simple_decision("RINGS", "RINGS", data["ring_signatures"]),
        _simple_decision(
            "CYCLE_RANK",
            "TOPOLOGY",
            (data["cycle_rank"], data["auxiliary_cycle_rank"]),
        ),
        _simple_decision("ATOM_CLASSES", "ATOM_EQUIVALENCE", data["atom_class_signatures"]),
        _simple_decision("STRUCTURAL_SITES", "STRUCTURAL_SITES", data["structural_site_signatures"]),
        _simple_decision("MULTICENTER_DOMAINS", "MULTICENTER", data["multicenter_signatures"]),
        _simple_decision("PRIMITIVE_CANDIDATES", "PRIMITIVES", data["primitive_signatures"]),
    ]
    quasi = data["quasi"]
    strict = quasi.strict_symmetry
    proposed = quasi.proposed_symmetry
    records.extend(
        (
            OraclePerceptionDecision(
                "STRICT_GROUP",
                "MOLECULAR_SYMMETRY",
                strict.point_group,
                None,
                float(strict.max_deviation),
                float(strict.max_deviation / policy.symmetry_thresholds.distance_angstrom),
                float(policy.symmetry_thresholds.distance_angstrom - strict.max_deviation),
                (
                    DecisionThreshold(
                        "STRICT_CARTESIAN_TOLERANCE",
                        policy.symmetry_thresholds.distance_angstrom,
                        "ANGSTROM",
                        provider="MATRIX_SYMMETRY",
                        provider_version="1",
                    ),
                ),
                "MATRIX_TOPOLOGY_QUALIFIED_SYMMETRY",
                "1",
            ),
            OraclePerceptionDecision(
                "PROPOSED_GROUP",
                "QUASI_SYMMETRY",
                proposed.point_group,
                strict.point_group,
                float(proposed.max_deviation),
                float(proposed.max_deviation / policy.quasi_symmetry_search_limit_angstrom),
                float(policy.quasi_symmetry_search_limit_angstrom - proposed.max_deviation),
                (
                    DecisionThreshold(
                        "QUASI_CARTESIAN_SEARCH_LIMIT",
                        policy.quasi_symmetry_search_limit_angstrom,
                        "ANGSTROM",
                        provider="MATRIX_SYMMETRY",
                        provider_version="1",
                    ),
                ),
                "MATRIX_TOPOLOGY_QUALIFIED_QUASI_SYMMETRY",
                "1",
            ),
        )
    )
    local_pairs = sorted(
        ((_local_signature(numbers, domain), domain) for domain in data["local_domains"]),
        key=lambda item: _local_decision_sort_key(item[0], item[1]),
    )
    for index, (signature, domain) in enumerate(local_pairs, start=1):
        template = domain.template_decision
        records.append(
            OraclePerceptionDecision(
                f"LOCAL:{index:04d}",
                "LOCAL_EQUIVALENCE_TEMPLATE",
                signature,
                (template.competing_template if template is not None else None),
                (template.score if template is not None else None),
                (template.score if template is not None else None),
                (template.margin if template is not None else None),
                tuple(
                    DecisionThreshold(name, value, unit, provider=domain.provider,
                                      provider_version=domain.provider_version)
                    for name, value, unit in domain.thresholds
                ),
                domain.provider,
                domain.provider_version,
            )
        )
    contact_pairs = sorted(
        ((_contact_signature(numbers, contact), contact) for contact in data["contacts"]),
        key=lambda item: _contact_decision_sort_key(item[0], item[1]),
    )
    for index, (signature, contact) in enumerate(contact_pairs, start=1):
        maximum_rho = policy.contact_settings.maximum_rho_for(contact.kind)
        minimum_confidence = policy.contact_settings.minimum_confidence_for(contact.kind)
        radial_margin = maximum_rho - contact.rho_vdw
        confidence_margin = contact.confidence - minimum_confidence
        records.append(
            OraclePerceptionDecision(
                f"CONTACT:{index:04d}",
                contact.kind,
                signature,
                None,
                float(contact.confidence),
                float(contact.persistence),
                float(min(radial_margin, confidence_margin)),
                (
                    DecisionThreshold(
                        "MAXIMUM_RHO_VDW",
                        maximum_rho,
                        "DIMENSIONLESS",
                        provider=contact.provider,
                        provider_version=contact.provider_version,
                    ),
                    DecisionThreshold(
                        "MINIMUM_CONFIDENCE",
                        minimum_confidence,
                        "DIMENSIONLESS",
                        provider=contact.provider,
                        provider_version=contact.provider_version,
                    ),
                ),
                contact.provider,
                contact.provider_version,
            )
        )
    return tuple(sorted(records, key=lambda item: item.decision_id))


def _simple_decision(identifier: str, family: str, accepted) -> OraclePerceptionDecision:
    return OraclePerceptionDecision(
        identifier,
        family,
        _canonical_text(accepted),
        None,
        None,
        None,
        None,
        (),
        ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER,
        ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER_VERSION,
    )


def _compare_decisions(
    reference: OraclePerceptionState,
    sampled: tuple[OraclePerceptionState | OraclePerceptionFailure, ...],
    *,
    minimum_stability: float,
) -> tuple[tuple[DecisionRobustness, ...], int]:
    reference_map = {item.decision_id: item for item in reference.decisions}
    sample_maps = [
        {item.decision_id: item for item in state.decisions} for state in sampled
    ]
    mismatch_by_sample = [0 for _state in sampled]
    output = []
    for identifier in sorted(reference_map):
        expected = reference_map[identifier]
        observed = []
        for sample_index, mapping in enumerate(sample_maps):
            candidate = mapping.get(identifier)
            value = candidate.accepted_class if candidate is not None else "ABSENT"
            observed.append(value)
            if value != expected.accepted_class:
                mismatch_by_sample[sample_index] += 1
        matches = sum(value == expected.accepted_class for value in observed)
        fraction = matches / len(sampled)
        alternatives = Counter(
            value for value in observed if value != expected.accepted_class
        )
        competitor = alternatives.most_common(1)[0][0] if alternatives else expected.competing_class
        worst = next(
            (index for index, value in enumerate(observed) if value != expected.accepted_class),
            0,
        )
        fallback = None if fraction >= minimum_stability else _fallback_for_family(expected.family)
        output.append(
            DecisionRobustness(
                decision_id=identifier,
                family=expected.family,
                accepted_class=expected.accepted_class,
                competing_class=competitor,
                raw_score=expected.raw_score,
                normalized_score=expected.normalized_score,
                signed_margin=expected.signed_margin,
                stability_fraction=float(fraction),
                worst_case_perturbation=worst,
                fallback=fallback,
                decision_reason=("ENSEMBLE_STABLE" if fallback is None else "ENSEMBLE_UNSTABLE"),
                thresholds=expected.thresholds,
                provider=expected.provider,
                provider_version=expected.provider_version,
            )
        )
    worst_case = max(
        range(len(sampled)), key=lambda index: (mismatch_by_sample[index], -index)
    )
    return tuple(output), int(worst_case)


def _fallback_for_family(family: str) -> str:
    label = family.upper()
    if label in {"TOPOLOGY", "FRAGMENTS", "RINGS", "MULTICENTER"}:
        return "TOPOLOGY_TRANSITION_OR_EXPLICIT_DECISION"
    if "SYMMETRY" in label:
        return "RETAIN_STRICT_GROUP"
    if "CONTACT" in label or "BOND" in label:
        return "OMIT_UNTIL_TEMPORALLY_PERSISTENT"
    if "EQUIVALENCE" in label or "TEMPLATE" in label:
        return "SPLIT_CLASSES_OR_C1"
    return "CONSERVATIVE_OMISSION"


def _symmetry_decision(
    reference: OraclePerceptionState, requested: str | None
) -> str:
    if requested is None:
        return (
            "REQUIRES_DECISION"
            if reference.proposed_group != reference.strict_group
            or reference.proposed_operation_signatures
            != reference.strict_operation_signatures
            else "RETAIN"
        )
    normalized = str(requested).strip().upper()
    if normalized not in {"PROJECT", "RETAIN"}:
        raise ValueError("symmetry_decision must be PROJECT or RETAIN")
    return normalized


def _validated_geometry(
    atomic_numbers: Sequence[int], coordinates_angstrom: np.ndarray
) -> tuple[tuple[int, ...], np.ndarray]:
    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if not numbers or any(value < 1 or value > 118 for value in numbers):
        raise ValueError("atomic numbers must lie in [1, 118]")
    if xyz.shape != (len(numbers), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("coordinates must be finite with shape (natoms, 3)")
    return numbers, xyz


def _connected_components(
    natoms: int, bonds: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, ...], ...]:
    graph = nx.Graph()
    graph.add_nodes_from(range(natoms))
    graph.add_edges_from(bonds)
    return tuple(sorted(tuple(sorted(component)) for component in nx.connected_components(graph)))


def _isomorphism_invariant_topology_hash(
    numbers: tuple[int, ...], bonds: tuple[tuple[int, int], ...]
) -> str:
    graph = nx.Graph()
    graph.add_nodes_from(
        (index, {"element": str(number)}) for index, number in enumerate(numbers)
    )
    graph.add_edges_from((left, right, {"kind": "PRIMARY"}) for left, right in bonds)
    iterations = min(8, max(3, int(math.ceil(math.log2(len(numbers) + 1)))))
    wl = nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="element", edge_attr="kind", iterations=iterations
    )
    formula = ",".join(
        f"{number}:{count}" for number, count in sorted(Counter(numbers).items())
    )
    return _stable_sha256((formula, len(bonds), wl))


@lru_cache(maxsize=256)
def _symbols_and_masses(numbers: tuple[int, ...]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    symbols = tuple(atomic_symbol(number) for number in numbers)
    masses = tuple(
        mass if (mass := atomic_mass(number)) > 0.0 else float(number)
        for number in numbers
    )
    return symbols, masses


def _component_signature(
    numbers: tuple[int, ...],
    bonds: tuple[tuple[int, int], ...],
    component: tuple[int, ...],
) -> str:
    selected = set(component)
    mapping = {atom: index for index, atom in enumerate(component)}
    local_bonds = tuple(
        (mapping[left], mapping[right])
        for left, right in bonds
        if left in selected and right in selected
    )
    local_numbers = tuple(numbers[atom] for atom in component)
    return _isomorphism_invariant_topology_hash(local_numbers, local_bonds)


def _ring_signature(numbers: tuple[int, ...], ring: tuple[int, ...]) -> str:
    colors = tuple(numbers[atom] for atom in ring)
    rotations = [colors[index:] + colors[:index] for index in range(len(colors))]
    reversed_colors = tuple(reversed(colors))
    rotations.extend(
        reversed_colors[index:] + reversed_colors[:index] for index in range(len(colors))
    )
    return f"RING:{len(ring)}:" + "-".join(str(value) for value in min(rotations))


def _synthon_records(symbols, synthons) -> tuple[dict[str, float | int | str], ...]:
    return tuple(
        {
            "atom": index + 1,
            "element": symbols[index],
            "z_eff": float(synthons.Zeff(index)),
            "charge": float(synthons.charge(index)),
            "covalency": float(synthons.covalency(index)),
            "delocalization": float(synthons.delocalization(index)),
            "radial_dispersion": float(synthons.radial_dispersion(index)),
            "strain": float(synthons.strain(index)),
            "pi_index": float(synthons.pi_index(index)),
            "pi_pi_index": float(synthons.pi_pi_index(index)),
        }
        for index in range(len(symbols))
    )


def _member_element_signature(numbers: tuple[int, ...], one_based_atoms) -> str:
    return "-".join(str(value) for value in sorted(numbers[int(atom) - 1] for atom in one_based_atoms))


def _local_signature(numbers: tuple[int, ...], domain) -> str:
    center = "RING" if domain.center_atom is None else str(numbers[domain.center_atom - 1])
    classes = ",".join(str(len(item.members)) for item in domain.equivalence_classes)
    template = domain.template_decision
    return (
        f"{domain.kind}:{center}:{_member_element_signature(numbers, domain.members)}:"
        f"CLASSES={classes}:GROUP={domain.proposed_group}:"
        f"TEMPLATE={(template.selected_template if template else None) or 'NONE'}:"
        f"STATUS={(template.status if template else 'NOT_APPLICABLE')}"
    )


def _local_decision_sort_key(signature: str, domain) -> tuple:
    """Canonicalize repeated equivalent local domains without atom labels."""

    def optional_score(value: float | None) -> float:
        return float("inf") if value is None else round(float(value), 12)

    template = domain.template_decision
    equivalence = tuple(
        sorted(
            (
                len(item.members),
                round(float(item.centroid_effective_atomic_number), 12),
                round(float(item.centroid_distance_angstrom), 12),
                round(float(item.maximum_zeff_spread), 12),
                round(float(item.maximum_distance_spread_angstrom), 12),
            )
            for item in domain.equivalence_classes
        )
    )
    template_key = (
        "NONE",
        "",
        float("inf"),
        float("inf"),
        "",
    ) if template is None else (
        template.selected_template or "NONE",
        template.competing_template or "NONE",
        optional_score(template.score),
        optional_score(template.margin),
        template.status,
    )
    return (
        signature,
        equivalence,
        template_key,
        domain.confidence,
        domain.provider,
        domain.provider_version,
    )


def _contact_signature(numbers: tuple[int, ...], contact) -> str:
    endpoints = []
    for endpoint in (contact.endpoint_a, contact.endpoint_b):
        if endpoint.kind == "ATOM":
            endpoints.append(f"ATOM_Z{numbers[int(endpoint.identifier) - 1]}")
        else:
            endpoints.append(f"SITE_{endpoint.identifier}")
    return (
        f"{contact.kind}:{'-'.join(sorted(endpoints))}:"
        f"{contact.open_or_closing}:DBETA={contact.delta_beta1_if_added}:"
        f"{contact.provider}@{contact.provider_version}"
    )


def _contact_decision_sort_key(signature: str, contact) -> tuple:
    """Canonicalize equivalent-fragment contacts using chemical evidence."""

    return (
        signature,
        round(float(contact.rho_vdw), 12),
        round(float(contact.distance_angstrom), 12),
        tuple(
            sorted(
                (name, round(float(value), 12))
                for name, value in contact.directional_descriptors
            )
        ),
        round(float(contact.confidence), 12),
        round(float(contact.persistence), 12),
    )


def _primitive_signature(numbers, function, atoms, family) -> str:
    one_based = tuple(int(atom) + 1 for atom in atoms)
    return f"{family}:{function}:{_member_element_signature(numbers, one_based)}"


def _operation_signatures(operations) -> tuple[str, ...]:
    output = []
    for operation in operations:
        permutation = tuple(int(atom) - 1 for atom in operation.permutation)
        visited: set[int] = set()
        cycles = []
        for atom in range(len(permutation)):
            if atom in visited:
                continue
            current = atom
            size = 0
            while current not in visited:
                visited.add(current)
                size += 1
                current = permutation[current]
            cycles.append(size)
        output.append(f"{operation.label}:{','.join(map(str, sorted(cycles)))}")
    return tuple(sorted(output))


def _canonical_text(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER",
    "ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER_VERSION",
    "OraclePerceptionDecision",
    "OraclePerceptionFailure",
    "OraclePerceptionState",
    "PerceptionAuditPolicy",
    "audit_perception_robustness",
    "deterministic_cartesian_perturbations",
    "perceive_oracle_state",
]
