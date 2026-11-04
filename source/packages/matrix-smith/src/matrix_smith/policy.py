"""Shared GICForge policy constants.

This module is the Python source of truth for the frozen GSNIC GIC contract.
Fortran kernels mirror these names and policies explicitly; downstream tools
should import these constants instead of hard-coding family names locally.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from matrix_chem import DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER


COORDINATE_METHOD_NAME = "ESALC"
COORDINATE_METHOD_LONG_NAME = "Effective Symmetry-Adapted Local Coordinates"
SERIALIZED_COORDINATE_METHOD_NAME = "SONIC"
LEGACY_COORDINATE_METHOD_NAME = "SMITH"
GIC_BACKEND = "oracle-native-primitive.v1"
SYCART_BACKEND = "oracle-native-cartesian-nullspace.v1"
B_MATRIX_BACKEND = "oracle-native-analytic-bmatrix.v1"
RING_PUCKERING_ROLE_OPERATIVE = "OPERATIVE"
RING_PUCKERING_ROLE_GLOBAL = "GLOBAL"
RING_PUCKERING_ROLE_DESCRIPTOR = "DESCRIPTOR"
RING_PUCKERING_ROLE_LEGACY = "LEGACY"
RING_PUCKERING_MODEL_TRIANGULAR_FLAP = "triangular_flap"
RING_PUCKERING_MODEL_CHARM = "charm"
RING_PUCKERING_MODEL_CREMER_POPLE = "cremer_pople"
RING_PUCKERING_MODEL_LEGACY = "legacy"
# Public names for the discrete minimum-optimization policy.  The internal
# implementation retains the historical ``local_out_of_plane`` spelling.
RING_LOCAL_FOURIER = "RING_LOCAL_FOURIER"
CHARM = "CHARM"
RING_PUCKERING_ROLE_MODELS = {
    RING_PUCKERING_ROLE_OPERATIVE: RING_PUCKERING_MODEL_TRIANGULAR_FLAP,
    RING_PUCKERING_ROLE_GLOBAL: RING_PUCKERING_MODEL_CHARM,
    RING_PUCKERING_ROLE_DESCRIPTOR: RING_PUCKERING_MODEL_CREMER_POPLE,
    RING_PUCKERING_ROLE_LEGACY: RING_PUCKERING_MODEL_LEGACY,
}
RING_PUCKERING_ROLES = frozenset(RING_PUCKERING_ROLE_MODELS)
RANK_METHOD = "analytic_b_matrix_mgs_greedy"
RANK_TOLERANCE = 1.0e-7
PHASE_ZERO_TOLERANCE = 1.0e-10
FRAGMENT_CONTEXT_TRANSITION_STATE = "transition_state"
DIAGNOSTIC_FINITE_DIFFERENCE_STEP = 1.0e-5
# Gaussian/GDV DefRed and FCEstG classify an angle as linear when it lies
# within 15 degrees of pi.  All SONIC generators use this single threshold.
LINEAR_ANGLE_DEGREES = 165.0
PSEUDO_BOND_EFFECTIVE_ORDER = DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER
PSEUDO_CYCLE_CLOSURE_VDW_SCALE = 1.05
ORDINARY_REDUCTION_CLASS = "ORDINARY"
SPECIAL_REDUCTION_CLASS = "SPECIAL_PROTECTED"
TS_REACTIVE_COORDINATE_CLASS = "TS_REACTIVE_MODE"
TS_REACTIVE_FAMILY_MIXING_POLICY = (
    "ALL_PRIMITIVE_FAMILIES_WITHIN_REACTIVE_COMPLEMENT_ONLY"
)
TS_REACTIVE_FAMILY_PRESERVING_POLICY = "WITHIN_FAMILY_REACTIVE_PARTITION"
REDUCTION_POLICY = "SPECIAL_PROTECTED_FIRST_THEN_ORDINARY_ANALYTIC_RANK"
LOCAL_SYMMETRIZATION_METHOD = "LOCAL_BLOCK_SALC"
POINT_GROUP_PROJECTOR_METHOD = "POINT_GROUP_PROJECTOR"
SYMMETRIZATION_POLICY = "ORACLE_TYPE_BLOCK_SUM_AND_DIFFERENCE"
PSEUDOBOND_TORSION_METRIC_SYMMETRIZATION_POLICY = (
    "ORACLE_TYPE_BLOCK_PLUS_RESIDUAL_B_METRIC_TORSION_EIGENBASIS"
)
MAX_NORMALIZED_SONIC_CONDITION = 100.0
# A single certified fragment does not need the stricter local-domain gate,
# but a basis sitting directly on the numerical-rank cutoff is not reproducible
# across BLAS implementations.  Crossing this limit triggers SMITH's frozen
# ORACLE-pool recovery before LINK ever sees the chart.
MAX_UNPARTITIONED_SONIC_CONDITION = 1.0e6
CANONICAL_SALC_DIAGNOSTIC = "CANONICAL_CATALOG=YES"
AROMATIC_LOCAL_SALC_DIAGNOSTIC = "AROMATIC_LOCAL_SALC=YES"
AROMATIC_LOCAL_MODEL_DIAGNOSTIC = "MODEL=AROMATIC_LOCAL_OUT_OF_PLANE"
ANALYTIC_SALC_BLOCKS = frozenset({"RDef", "RPck", "BtFl", "Spir"})
CAGE_SPECIAL_BLOCKS = frozenset({"BtFl", "Spir"})
RING_SPECIAL_BLOCKS = frozenset({"RDef", "RPck"})
PROJECTOR_SYMMETRIZATION_POLICY = "HOMOGENEOUS_TYPE_BLOCK_POINT_GROUP_PROJECTOR"
SYMMETRY_OPERATION_TOLERANCE_ANGSTROM = 1.0e-3
SYMMETRY_OPERATION_NEAR_THRESHOLD_FRACTION = 0.25
SALC_PATH_OVERLAP_WARNING_THRESHOLD = 0.98
SALC_PATH_PIVOT_GAP_WARNING = 1.0e-6
# Cross-platform LAPACK implementations may rotate a numerically equivalent
# SALC basis at the last few decimal places.  Golden gates compare projectors,
# not basis gauges, with this shared absolute tolerance.
SALC_SUBSPACE_PROJECTOR_TOLERANCE = 1.0e-7
# Reference phase construction passes through eigensolvers and trigonometric
# reduction.  This bound is shared by every supported CPU architecture.
REFERENCE_PHASE_TOLERANCE_DEGREES = 2.0e-5
FRAGMENT_MODE_SPECIAL_COORDINATES = "SPECIAL_COORDINATES"
FRAGMENT_MODE_PSEUDO_BONDS = "PSEUDO_BONDS"
FRAGMENT_MODE_NONE = "NONE"
XH_STRETCH_POLICY_SYMMETRIZE = "SYMMETRIZE"
XH_STRETCH_POLICY_LOCAL_ALL = "LOCAL_ALL"
XH_STRETCH_POLICY_LOCAL_SELECTED = "LOCAL_SELECTED"
XH_STRETCH_CLASS_XH = "XH"
XH_STRETCH_CLASS_XH2 = "XH2"
XH_STRETCH_CLASS_XH3 = "XH3"
XH_STRETCH_CLASSES = frozenset(
    {
        XH_STRETCH_CLASS_XH,
        XH_STRETCH_CLASS_XH2,
        XH_STRETCH_CLASS_XH3,
    }
)
XH_STRETCH_POLICIES = frozenset(
    {
        XH_STRETCH_POLICY_SYMMETRIZE,
        XH_STRETCH_POLICY_LOCAL_ALL,
        XH_STRETCH_POLICY_LOCAL_SELECTED,
    }
)


@dataclass(frozen=True)
class SonicConstructionPolicy:
    """Single policy contract shared by SONIC construction and reporting."""

    analytic_salc_blocks: frozenset[str] = ANALYTIC_SALC_BLOCKS
    cage_special_blocks: frozenset[str] = CAGE_SPECIAL_BLOCKS
    ring_special_blocks: frozenset[str] = RING_SPECIAL_BLOCKS
    canonical_salc_diagnostic: str = CANONICAL_SALC_DIAGNOSTIC
    aromatic_salc_diagnostic: str = AROMATIC_LOCAL_SALC_DIAGNOSTIC
    aromatic_model_diagnostic: str = AROMATIC_LOCAL_MODEL_DIAGNOSTIC
    aromatic_default_model: str = "local_out_of_plane"
    normalized_condition_advisory: float = MAX_NORMALIZED_SONIC_CONDITION

    def is_analytic_salc(self, *, block: str, diagnostic: str) -> bool:
        return (
            self.canonical_salc_diagnostic in diagnostic
            or self.aromatic_salc_diagnostic in diagnostic
            or block in self.analytic_salc_blocks
        )

    def effective_ring_model(self, requested: str, *, aromatic: bool) -> str:
        model = normalize_ring_puckering_model(requested)
        if aromatic and model == RING_PUCKERING_MODEL_TRIANGULAR_FLAP:
            return self.aromatic_default_model
        return model

    def discrete_minimum_ring_model(self, *, aromatic: bool) -> str:
        """Return the frozen minimum-optimization ring model from ORACLE data."""
        return "local_out_of_plane" if bool(aromatic) else "charm"


SONIC_CONSTRUCTION_POLICY = SonicConstructionPolicy()


def ring_puckering_model_for_role(role: str) -> str:
    """Return the canonical ring model for an explicit scientific role."""

    normalized = str(role).strip().upper().replace("-", "_")
    try:
        return RING_PUCKERING_ROLE_MODELS[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(RING_PUCKERING_ROLES))
        raise ValueError(f"unsupported ring puckering role {role!r}; use {allowed}") from exc


def normalize_ring_puckering_model(value: str) -> str:
    """Normalize the shared ring-model vocabulary and emit legacy warnings."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "triangular_flap": "triangular_flap",
        "flap": "triangular_flap",
        "hill_reilly": "triangular_flap",
        "local_out_of_plane": "local_out_of_plane",
        "out_of_plane": "local_out_of_plane",
        "u_fourier": "local_out_of_plane",
        "charm": "charm",
        "cyclic_height_averaged_ring_modes": "charm",
        "frisch_height": "charm",
        "frisch": "charm",
        "h_fourier": "charm",
        "symmetrized_height": "charm",
        "endocyclic_dihedral": "endocyclic_dihedral",
        "dihedral": "endocyclic_dihedral",
        "legacy": "endocyclic_dihedral",
    }
    try:
        result = aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "ring_puckering_model must be triangular_flap, charm, "
            "local_out_of_plane, or endocyclic_dihedral"
        ) from exc
    if normalized in {"frisch_height", "frisch", "h_fourier", "symmetrized_height"}:
        warnings.warn(
            f"ring_puckering_model={value!r} is deprecated; use 'charm'",
            DeprecationWarning,
            stacklevel=3,
        )
    if result == "endocyclic_dihedral":
        warnings.warn(
            "endocyclic ring dihedrals are legacy/deprecated; use triangular_flap, "
            "charm, or local_out_of_plane",
            DeprecationWarning,
            stacklevel=3,
        )
    return result


FRAGMENT_MODES = frozenset(
    {
        FRAGMENT_MODE_NONE,
        FRAGMENT_MODE_PSEUDO_BONDS,
        FRAGMENT_MODE_SPECIAL_COORDINATES,
    }
)


@dataclass(frozen=True)
class PrimitiveFamilyPolicy:
    family: str
    function: str
    prefix: str
    reduction_class: str
    symmetry_block: str


PRIMITIVE_FAMILY_POLICIES = (
    PrimitiveFamilyPolicy("FACE_FLAP", "FACE_FLAP", "FcFl", ORDINARY_REDUCTION_CLASS, "FACE_FLAP"),
    PrimitiveFamilyPolicy("CENTER_FACE_DIST", "CENTER_FACE_DIST", "CeFa", SPECIAL_REDUCTION_CLASS, "CENTER_FACE_DIST"),
    PrimitiveFamilyPolicy("FACE_DIRECTION_ANGLE", "FACE_DIRECTION_ANGLE", "FaDi", SPECIAL_REDUCTION_CLASS, "FACE_DIRECTION_ANGLE"),
    PrimitiveFamilyPolicy("FACE_HINGE", "FACE_HINGE", "FaHi", ORDINARY_REDUCTION_CLASS, "FACE_HINGE"),
    PrimitiveFamilyPolicy("CONTACT_CENTER_ANGLE", "CONTACT_CENTER_ANGLE", "CoAn", SPECIAL_REDUCTION_CLASS, "CONTACT_CENTER_ANGLE"),
    PrimitiveFamilyPolicy("STRETCH", "R", "Str", ORDINARY_REDUCTION_CLASS, "STRETCH"),
    PrimitiveFamilyPolicy(
        "TS_REACTION_DISTANCE",
        "R",
        "TSRe",
        SPECIAL_REDUCTION_CLASS,
        "TS_REACTION_DISTANCE",
    ),
    PrimitiveFamilyPolicy(
        TS_REACTIVE_COORDINATE_CLASS,
        "DIMENSIONLESS",
        "RxMd",
        SPECIAL_REDUCTION_CLASS,
        TS_REACTIVE_COORDINATE_CLASS,
    ),
    PrimitiveFamilyPolicy(
        "LOCAL_XH_STRETCH",
        "R",
        "XHSt",
        ORDINARY_REDUCTION_CLASS,
        "LOCAL_XH_STRETCH",
    ),
    PrimitiveFamilyPolicy("BEND", "A", "Bend", ORDINARY_REDUCTION_CLASS, "BEND"),
    PrimitiveFamilyPolicy(
        "CYCLIC_BEND",
        "A",
        "CyBe",
        ORDINARY_REDUCTION_CLASS,
        "CYCLIC_BEND",
    ),
    PrimitiveFamilyPolicy(
        "SPIRO_BEND",
        "A",
        "Spir",
        ORDINARY_REDUCTION_CLASS,
        "SPIRO_BEND",
    ),
    PrimitiveFamilyPolicy(
        "LINEAR_BEND",
        "L",
        "LinB",
        ORDINARY_REDUCTION_CLASS,
        "LINEAR_BEND",
    ),
    PrimitiveFamilyPolicy(
        "COUPLED_ANGULAR",
        "A",
        "AngC",
        ORDINARY_REDUCTION_CLASS,
        "COUPLED_ANGULAR",
    ),
    PrimitiveFamilyPolicy("TORSION", "D", "Tors", ORDINARY_REDUCTION_CLASS, "TORSION"),
    PrimitiveFamilyPolicy(
        "PSEUDO_CYCLE_BEND",
        "RPCB",
        "PsAn",
        ORDINARY_REDUCTION_CLASS,
        "PSEUDO_CYCLE_BEND",
    ),
    PrimitiveFamilyPolicy(
        "PSEUDO_CYCLE_TORSION",
        "RPCK",
        "PsTo",
        ORDINARY_REDUCTION_CLASS,
        "PSEUDO_CYCLE_TORSION",
    ),
    PrimitiveFamilyPolicy(
        "CYCLIC_TORSION",
        "D",
        "CyTo",
        ORDINARY_REDUCTION_CLASS,
        "CYCLIC_TORSION",
    ),
    PrimitiveFamilyPolicy(
        "RING_PUCKER_COMPONENT",
        "RPCK",
        "RPck",
        ORDINARY_REDUCTION_CLASS,
        "RING_PUCKER_COMPONENT",
    ),
    PrimitiveFamilyPolicy(
        "FULLERENE_FACE_DIAGONAL",
        "R",
        "FCDg",
        ORDINARY_REDUCTION_CLASS,
        "FULLERENE_FACE_DIAGONAL",
    ),
    PrimitiveFamilyPolicy(
        "CONDENSED_RING_TORSION",
        "D",
        "CoTo",
        ORDINARY_REDUCTION_CLASS,
        "CONDENSED_RING_TORSION",
    ),
    PrimitiveFamilyPolicy(
        "BUTTERFLY",
        "D",
        "BtFl",
        ORDINARY_REDUCTION_CLASS,
        "BUTTERFLY",
    ),
    PrimitiveFamilyPolicy(
        "OUT_OF_PLANE",
        "U",
        "OuPl",
        ORDINARY_REDUCTION_CLASS,
        "OUT_OF_PLANE",
    ),
    PrimitiveFamilyPolicy(
        "IMPROPER_DIHEDRAL",
        "IMPD",
        "ImpD",
        ORDINARY_REDUCTION_CLASS,
        "OUT_OF_PLANE",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_DISTANCE",
        "FC_DIST",
        "FCDi",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_DISTANCE",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_CENTER_ATOM_DISTANCE",
        "FCA_DIST",
        "FCAt",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_CENTER_ATOM",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_TRANSLATION",
        "FTRANS",
        "FTrn",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_TRANSLATION",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_ORIENTATION",
        "FROT",
        "FRot",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_ORIENTATION",
    ),
    PrimitiveFamilyPolicy(
        "CENTER_ATOM_DISTANCE",
        "CENTER_ATOM_DIST",
        "CnAt",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_CENTER_ATOM",
    ),
    PrimitiveFamilyPolicy(
        "HBOND_DISTANCE",
        "R",
        "HBDi",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_HBOND_DISTANCE",
    ),
    PrimitiveFamilyPolicy(
        "PSEUDO_BOND_DISTANCE",
        "R",
        "PBDi",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_PSEUDO_BOND_DISTANCE",
    ),
    PrimitiveFamilyPolicy(
        "PSEUDO_BOND_BEND",
        "A",
        "PBAn",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_PSEUDO_BOND_CONTACT",
    ),
    PrimitiveFamilyPolicy(
        "PSEUDO_BOND_TORSION",
        "D",
        "PBTo",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_PSEUDO_BOND_CONTACT",
    ),
)

PRIMITIVE_POLICY_BY_FAMILY = {policy.family: policy for policy in PRIMITIVE_FAMILY_POLICIES}
PRIMITIVE_POLICY_BY_FUNCTION = {policy.function: policy for policy in PRIMITIVE_FAMILY_POLICIES}
PRIMITIVE_FAMILY_ORDER = tuple(policy.family for policy in PRIMITIVE_FAMILY_POLICIES)
SPECIAL_PRIMITIVE_FAMILIES = frozenset(
    policy.family
    for policy in PRIMITIVE_FAMILY_POLICIES
    if policy.reduction_class == SPECIAL_REDUCTION_CLASS
)
SYMMETRY_BLOCK_BY_FAMILY = {
    policy.family: policy.symmetry_block for policy in PRIMITIVE_FAMILY_POLICIES
}


def primitive_reduction_class(family: str) -> str:
    policy = PRIMITIVE_POLICY_BY_FAMILY.get(family)
    if policy is None:
        return ORDINARY_REDUCTION_CLASS
    return policy.reduction_class


def primitive_prefix(family: str) -> str:
    return PRIMITIVE_POLICY_BY_FAMILY[family].prefix


def primitive_symmetry_block(family: str) -> str:
    return SYMMETRY_BLOCK_BY_FAMILY.get(family, family)


def normalize_fragment_context(value: str) -> str:
    """Normalize the shared SMITH task policy without changing its meaning."""
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "optimization":
        return "minimum"
    if normalized not in {"minimum", "exploration", FRAGMENT_CONTEXT_TRANSITION_STATE}:
        raise ValueError("fragment_context must be 'minimum', 'exploration' or 'transition_state'")
    return normalized
