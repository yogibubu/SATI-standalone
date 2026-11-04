"""Versioned immutable chemical perception policy owned by ORACLE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math

from matrix_chem.local_perception import (
    LOCAL_DISTANCE_TOLERANCE_ANGSTROM,
    LOCAL_TEMPLATE_MIN_MARGIN,
    LOCAL_TEMPLATE_RMS_THRESHOLD,
    LOCAL_ZEFF_TOLERANCE,
)
from matrix_chem.topology.discrete_graph import BOND_THRESHOLD

from .multicenter_domains import MAXIMUM_NORMALIZED_STRUCTURAL_BRIDGE_DISTANCE


ORACLE_CHEMICAL_PERCEPTION_POLICY_SCHEMA = "matrix.oracle.chemical_perception_policy.v1"


@dataclass(frozen=True)
class ChemicalPerceptionPolicy:
    family: str
    metric: str
    unit: str
    entry_threshold: float
    exit_threshold: float
    active_direction: str
    fallback: str
    applicability: str
    provider: str
    provider_version: str

    def __post_init__(self) -> None:
        direction = self.active_direction.strip().upper()
        if direction not in {"HIGHER", "LOWER"}:
            raise ValueError("chemical policy direction must be HIGHER or LOWER")
        if not math.isfinite(self.entry_threshold) or not math.isfinite(self.exit_threshold):
            raise ValueError("chemical policy thresholds must be finite")
        if direction == "HIGHER" and self.entry_threshold <= self.exit_threshold:
            raise ValueError("HIGHER policy entry must exceed exit")
        if direction == "LOWER" and self.entry_threshold >= self.exit_threshold:
            raise ValueError("LOWER policy entry must be below exit")
        if not all(
            (
                self.family,
                self.metric,
                self.unit,
                self.fallback,
                self.applicability,
                self.provider,
                self.provider_version,
            )
        ):
            raise ValueError("chemical perception policy is incomplete")


def _contact_policy(family: str) -> ChemicalPerceptionPolicy:
    return ChemicalPerceptionPolicy(
        family=family,
        metric="RHO_VDW_AND_DIRECTIONAL_CONFIDENCE",
        unit="DIMENSIONLESS",
        entry_threshold=0.95,
        exit_threshold=1.05,
        active_direction="LOWER",
        fallback="OMIT_UNTIL_TEMPORALLY_PERSISTENT",
        applicability=f"ORACLE qualified {family.lower().replace('_', ' ')} provider",
        provider="ORACLE_AUXILIARY_CONTACT_PROVIDERS",
        provider_version="1",
    )


ORACLE_CHEMICAL_PERCEPTION_POLICIES = (
    ChemicalPerceptionPolicy(
        "COVALENT_BOND",
        "CONTINUOUS_COVALENT_CONNECTIVITY",
        "DIMENSIONLESS",
        float(BOND_THRESHOLD),
        0.15,
        "HIGHER",
        "RETAIN_PREVIOUS_GRAPH_OR_REQUIRE_TRANSITION",
        "geometry-first covalent topology with hydrogen-bridge validation",
        "MATRIX_CONTINUOUS_GRAPH",
        "1",
    ),
    *tuple(
        _contact_policy(family)
        for family in (
            "HYDROGEN_BOND",
            "DATIVE_CONTACT",
            "STRUCTURAL_LIGAND_CONTACT",
            "TETREL_BOND",
            "PNICTOGEN_BOND",
            "CHALCOGEN_BOND",
            "HALOGEN_BOND",
        )
    ),
    ChemicalPerceptionPolicy(
        "LOCAL_ZEFF_EQUIVALENCE",
        "ABSOLUTE_ZEFF_DIFFERENCE",
        "DIMENSIONLESS",
        LOCAL_ZEFF_TOLERANCE,
        1.5 * LOCAL_ZEFF_TOLERANCE,
        "LOWER",
        "SPLIT_LOCAL_EQUIVALENCE_CLASS",
        "same element, compatible graph environment and local radial shell",
        "ORACLE_LOCAL_EQUIVALENCE_AND_PSEUDOSYMMETRY",
        "1",
    ),
    ChemicalPerceptionPolicy(
        "LOCAL_RADIAL_EQUIVALENCE",
        "ABSOLUTE_CENTER_DISTANCE_DIFFERENCE",
        "ANGSTROM",
        LOCAL_DISTANCE_TOLERANCE_ANGSTROM,
        1.5 * LOCAL_DISTANCE_TOLERANCE_ANGSTROM,
        "LOWER",
        "SPLIT_LOCAL_EQUIVALENCE_CLASS",
        "same element, compatible graph environment and Zeff",
        "ORACLE_LOCAL_EQUIVALENCE_AND_PSEUDOSYMMETRY",
        "1",
    ),
    ChemicalPerceptionPolicy(
        "LOCAL_TEMPLATE_RMS",
        "PAIR_COSINE_RMS",
        "DIMENSIONLESS",
        LOCAL_TEMPLATE_RMS_THRESHOLD,
        1.25 * LOCAL_TEMPLATE_RMS_THRESHOLD,
        "LOWER",
        "GENERIC_LOCAL_COORDINATION",
        "coordination-number-qualified template library",
        "ORACLE_LOCAL_EQUIVALENCE_AND_PSEUDOSYMMETRY",
        "1",
    ),
    ChemicalPerceptionPolicy(
        "LOCAL_TEMPLATE_MARGIN",
        "BEST_MINUS_COMPETITOR_SEPARATION",
        "DIMENSIONLESS",
        LOCAL_TEMPLATE_MIN_MARGIN,
        0.5 * LOCAL_TEMPLATE_MIN_MARGIN,
        "HIGHER",
        "AMBIGUOUS_NO_TEMPLATE_FREEZE",
        "only after template RMS applicability succeeds",
        "ORACLE_LOCAL_EQUIVALENCE_AND_PSEUDOSYMMETRY",
        "1",
    ),
    ChemicalPerceptionPolicy(
        "QUASI_SYMMETRY",
        "TOPOLOGY_QUALIFIED_CARTESIAN_MAX_DEVIATION",
        "ANGSTROM",
        0.02,
        0.03,
        "LOWER",
        "RETAIN_STRICT_GROUP_AND_REQUIRE_DECISION",
        "complete group, topology automorphism and strict-subgroup preservation",
        "MATRIX_TOPOLOGY_QUALIFIED_QUASI_SYMMETRY",
        "1",
    ),
    ChemicalPerceptionPolicy(
        "PRIMARY_MULTICENTER_HYPEREDGE",
        "COVALENT_RADIUS_NORMALIZED_MULTICENTER_GEOMETRY",
        "DIMENSIONLESS",
        MAXIMUM_NORMALIZED_STRUCTURAL_BRIDGE_DISTANCE,
        MAXIMUM_NORMALIZED_STRUCTURAL_BRIDGE_DISTANCE + 0.10,
        "LOWER",
        "RETAIN_PREVIOUS_HYPEREDGE_OR_REQUIRE_TRANSITION",
        "shared protons and periodic-role H/halide structural-center bridges; never vdW pseudobonds",
        "ORACLE_MULTICENTER_DOMAINS",
        "1",
    ),
)


def validate_chemical_perception_policies(
    policies: tuple[ChemicalPerceptionPolicy, ...] = ORACLE_CHEMICAL_PERCEPTION_POLICIES,
) -> None:
    keys = [(item.family, item.metric) for item in policies]
    if len(keys) != len(set(keys)):
        raise ValueError("chemical perception policies must have unique family/metric keys")
    required_contacts = {
        "HYDROGEN_BOND",
        "DATIVE_CONTACT",
        "STRUCTURAL_LIGAND_CONTACT",
        "TETREL_BOND",
        "PNICTOGEN_BOND",
        "CHALCOGEN_BOND",
        "HALOGEN_BOND",
    }
    if not required_contacts.issubset({item.family for item in policies}):
        raise ValueError("chemical perception policy omits a contact family")


@lru_cache(maxsize=1)
def _chemical_perception_policy_manifest_json() -> str:
    validate_chemical_perception_policies()
    records = [asdict(item) for item in ORACLE_CHEMICAL_PERCEPTION_POLICIES]
    payload = {
        "schema": ORACLE_CHEMICAL_PERCEPTION_POLICY_SCHEMA,
        "owner": "ORACLE",
        "rank_tolerances_owner": "SMITH",
        "policies": records,
        "invariants": [
            "NO_SILENT_TOPOLOGY_OR_SYMMETRY_PROMOTION",
            "PRIMARY_MULTICENTER_HYPEREDGES_NEVER_VDW_PSEUDOBONDS",
            "ENTRY_AND_EXIT_THRESHOLDS_ALWAYS_DISTINCT",
            "GRAPH_AND_ENVIRONMENT_COMPATIBILITY_PRECEDES_GEOMETRIC_EQUIVALENCE",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return json.dumps(
        {**payload, "sha256": hashlib.sha256(encoded).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    )


def chemical_perception_policy_manifest() -> dict:
    """Return a fresh view of the cached immutable perception policy."""

    return json.loads(_chemical_perception_policy_manifest_json())


__all__ = [
    "ChemicalPerceptionPolicy",
    "ORACLE_CHEMICAL_PERCEPTION_POLICIES",
    "ORACLE_CHEMICAL_PERCEPTION_POLICY_SCHEMA",
    "chemical_perception_policy_manifest",
    "validate_chemical_perception_policies",
]
