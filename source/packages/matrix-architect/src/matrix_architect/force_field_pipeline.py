"""Single construction boundary for internal GFN-FF/SYCART and ZAFF/SONIC.

ARCHITECT consumes ORACLE evidence and SMITH coordinates; runtimes never
import this construction module. Rebuilding a topology creates a fresh
objective/chart epoch, not a silent change of potential during an iteration.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from matrix_oracle.force_field_topology import OracleForceFieldTopology


def force_field_route(topology: OracleForceFieldTopology, family: str):
    if family not in {"GFN-FF", "ZAFF"}:
        raise ValueError("standard internal force fields are GFN-FF and ZAFF; OpenMM is not a route")
    if not topology.reliable:
        return "MATRIX-GFN2/SYCART"
    return "GFN-FF/SYCART" if family == "GFN-FF" else "ZAFF/SONIC"


def _contract(topology, family):
    route = force_field_route(topology, family)
    if route == "MATRIX-GFN2/SYCART":
        raise ValueError(
            "uncertain ORACLE topology requires MATRIX-GFN2/SYCART before force-field construction"
        )
    return {"schema": "matrix.architect.force_field_construction.v1", "owner": "ARCHITECT",
            "family": family, "route": route, "topology": topology.to_dict(),
            "topology_sha256": topology.fingerprint, "epoch": topology.epoch,
            "energy_unit": "hartree", "cartesian_unit": "bohr",
            "automatic_model_switch": False, "openmm_required": False}


def prepare_gfnff_model(topology, model):
    """Seal a complete compiled GFN-FF model against the conserved topology.

    GFN-FF EEQ remains its own variational charge model; ORACLE population
    evidence is retained for typing, not substituted for the EEQ solution.
    """
    from .gfnff_compiler import assert_complete_gfnff_model, GFNFF_REQUIRED_TERM_FAMILIES

    contract = _contract(topology, "GFN-FF")
    assert_complete_gfnff_model(model)
    ledger = model["gfnff_migration"]
    if (ledger.get("missing_term_families")
            or set(ledger.get("implemented_term_families", ())) != set(GFNFF_REQUIRED_TERM_FAMILIES)):
        raise ValueError("GFN-FF coverage ledger is inconsistent")
    if tuple(model.get("atoms", ())) != topology.atoms:
        raise ValueError("GFN-FF atom ordering differs from ORACLE")
    bonds = {tuple(sorted(map(int, pair))) for pair in model.get("topology_bonds", ())}
    if bonds != {(a, b) for a, b, _ in topology.bond_orders}:
        raise ValueError("GFN-FF connectivity differs from the ORACLE snapshot")
    charge = sum(model.get("gfnff_eeq", {}).get("fragment_charges", ()))
    if not np.isclose(charge, sum(topology.charges), atol=1e-8, rtol=0):
        raise ValueError("GFN-FF total charge differs from ORACLE")
    result = deepcopy(model)
    result["architect_construction"] = contract
    return result


def build_gfnff_runtime(topology, model, **runtime_policy):
    from matrix_zaff.gfnff_runtime import GFNFFDynamicsRuntime

    return GFNFFDynamicsRuntime(prepare_gfnff_model(topology, model), **runtime_policy)


@dataclass(frozen=True)
class SonicTermSpecification:
    """A transferable family and its ordered local SONIC supports.

    Parameters are owned by the selected family factory, which receives the
    full ORACLE snapshot, selected SMITH primitives and explicit parameters.
    Multi-family/ reactive combinations are not forced into scalar stretches.
    """
    family: str
    local_labels: tuple[str, ...]
    parameters: object
    source: str


def prepare_equilibrium_refitted_terms(specifications, local_labels, *, coordinates,
                                       target_diagonal, fixed_gradient, fixed_diagonal,
                                       provenance):
    """Refit selected valence specifications once, before runtime construction.

    Inputs are already resolved donor parameters (current ZAFF, future LCB25
    synthon interpolation, or a declared GFN-FF-derived reference). This is
    not an implicit donor selector. Other retained families must be included
    in the supplied fixed derivatives and passed separately to the builder.
    The existing family factories and hot-loop kernels are unchanged.
    """
    from .equilibrium_refit import (
        AngleEquilibriumTerm, RadialEquilibriumTerm, refit_valence_equilibria,
    )
    from matrix_zaff.bonded import BondOrderDampedAnglePotential
    from matrix_zaff.radial import ExpPEPotential, MorsePotential

    labels, specifications = tuple(local_labels), tuple(specifications)
    if len(set(labels)) != len(labels):
        raise ValueError("equilibrium-refit coordinate labels must be unique")
    lookup = {label: i for i, label in enumerate(labels)}
    radial, angular, order = [], [], []
    for spec in specifications:
        if not spec.source or any(label not in lookup for label in spec.local_labels):
            raise ValueError("equilibrium refit requires covered labels and source provenance")
        indices = tuple(lookup[label] for label in spec.local_labels)
        if isinstance(spec.parameters, (MorsePotential, ExpPEPotential)) and len(indices) == 1:
            order.append(("radial", len(radial)))
            radial.append(RadialEquilibriumTerm(indices[0], spec.parameters))
        elif isinstance(spec.parameters, BondOrderDampedAnglePotential) and len(indices) == 3:
            order.append(("angular", len(angular)))
            angular.append(AngleEquilibriumTerm(indices, spec.parameters))
        else:
            raise ValueError("unsupported equilibrium-refit family/support; no implicit conversion")
    radial, angular, report = refit_valence_equilibria(
        coordinates, target_diagonal, fixed_gradient, fixed_diagonal, radial, angular,
        provenance=provenance)
    fitted = {"radial": radial, "angular": angular}
    result = tuple(replace(spec, parameters=fitted[kind][index].potential,
                           source=spec.source+"; ARCHITECT equilibrium/electrostatic compensation")
                   for spec, (kind, index) in zip(specifications, order, strict=True))
    report["local_labels"] = list(labels)
    return result, report


class SonicFamilyRegistry:
    """Incrementable ARCHITECT family factories; missing coverage fails closed."""

    def __init__(self):
        self._factories = {}

    def register(self, family, factory):
        if not family or family in self._factories or not callable(factory):
            raise ValueError("invalid or already registered SONIC force-field family")
        self._factories[family] = factory

    def build(self, specification, topology, primitives):
        if specification.family not in self._factories:
            raise ValueError(f"uncovered ZAFF SONIC family: {specification.family}")
        if not specification.source:
            raise ValueError("ZAFF parameters require a provenance source")
        kernel = self._factories[specification.family](topology, primitives, specification.parameters)
        if not callable(getattr(kernel, "evaluate", None)):
            raise ValueError("SONIC family factory did not produce an E/G/Hv kernel")
        return kernel


def build_zaff_sonic_objective(topology, chart, variables, embedding, *,
                              specifications, registry, nonbonded_factory,
                              energy_reference_hartree=0., geometry_policy=None,
                              nonbonded_only_labels=()):
    """Construct ZAFF on the case-specific SONIC and conserved ORACLE evidence.

    Family factories perform chemical typing from the snapshot. No generic
    harmonic potential is substituted for missing physical parameterization.
    Nonbonded construction receives the same snapshot; no external engine is
    selected implicitly. Existing energy shapes and switches belong to the
    registered kernels, not to the optimizer coordinate representation.
    """
    from matrix_zaff.sonic_bonded import DirectSonicBondedRuntime, LocalSonicTerm
    from matrix_zaff.sonic_geometry import SonicGeometryMap
    from matrix_link.zaff_sonic import ZaffSonicObjective

    contract = _contract(topology, "ZAFF")
    if chart.atoms != topology.atoms or chart.topology_sha256 != topology.fingerprint:
        raise ValueError("SMITH chart belongs to a different ORACLE topology/epoch")
    definition = chart.definition
    primitive_by_id = {p.identifier: p for p in definition.primitives}
    indices = {label: i for i, label in enumerate(embedding.local_labels)}
    terms, records = [], []
    for spec in specifications:
        labels = tuple(spec.local_labels)
        if (not labels or len(set(labels)) != len(labels)
                or any(label not in indices or label not in primitive_by_id for label in labels)):
            raise ValueError("ZAFF term labels do not match the SMITH local basis")
        primitives = tuple(primitive_by_id[label] for label in labels)
        kernel = registry.build(spec, topology, primitives)
        terms.append(LocalSonicTerm(tuple(indices[label] for label in labels), kernel))
        records.append({"family": spec.family, "local_labels": list(labels), "source": spec.source})
    if not terms:
        raise ValueError("ZAFF SONIC construction requires explicit bonded coverage")
    nonbonded_only = tuple(nonbonded_only_labels)
    covered = {label for record in records for label in record["local_labels"]}
    if (len(set(nonbonded_only)) != len(nonbonded_only)
            or not set(nonbonded_only) <= set(indices)
            or covered & set(nonbonded_only)
            or covered | set(nonbonded_only) != set(indices)):
        raise ValueError("ZAFF local coordinate coverage is incomplete or contradictory")
    contract["terms"] = records
    contract["nonbonded_only_labels"] = list(nonbonded_only)
    contract["active_sonic_labels"] = list(embedding.active_labels)
    bonded = DirectSonicBondedRuntime(variables, terms, energy_reference_hartree=energy_reference_hartree)
    geometry = SonicGeometryMap(definition, variables, embedding, **(geometry_policy or {}))
    nonbonded = nonbonded_factory(topology)
    if not callable(getattr(nonbonded, "evaluate", None)):
        raise ValueError("nonbonded factory must provide the internal E/G/Hv runtime")
    return ZaffSonicObjective(geometry, bonded, nonbonded, construction_contract=contract)
