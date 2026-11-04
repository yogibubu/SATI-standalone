"""Immutable provider boundaries for construction and use of a ZAFF surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ZAFF_WORKFLOW_SCHEMA = "matrix.architect.zaff_workflow.v2"


def zaff_workflow_contract() -> dict[str, Any]:
    """Return the normative SMITH--LINK--ARCHITECT--TRINITY handoff."""

    from matrix_core import (
        ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID,
        ARCHITECT_ZAFF_REFERENCE_SHA256,
        load_architect_zaff_reference_protocol,
    )

    reference = load_architect_zaff_reference_protocol()
    return {
        "schema": ZAFF_WORKFLOW_SCHEMA,
        "reference_protocol_id": ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID,
        "reference_protocol_sha256": ARCHITECT_ZAFF_REFERENCE_SHA256,
        "reference_protocol_change_policy": reference["change_policy"],
        "providers": {
            "ORACLE": "continuous perception, topology, synthons and primitive semantics",
            "SMITH": (
                "redundant ZAFF field catalog, Wilson rows and rank-certified "
                "nonredundant analysis chart"
            ),
            "LINK": (
                "reference optimization, constrained internal-coordinate displacements "
                "and parallel electronic-structure point requests"
            ),
            "ARCHITECT": (
                "nonbonded subtraction, complete quadratic fit, anharmonic "
                "refinement, certification and serialization"
            ),
            "ZAFF": "analytic energy, gradient and Hessian evaluation",
            "TRINITY": (
                "Wilson GF modes, normal-coordinate derivative acquisition and "
                "vibrational analysis"
            ),
        },
        "invariants": {
            "atom_order": "IMMUTABLE",
            "reference_geometry": "IMMUTABLE_WITH_CONTENT_HASH",
            "complete_quadratic_field": (
                "IDENTICAL_FROM_ZAFF2_THROUGH_ZAFF4"
            ),
            "field_coordinates": "MAY_BE_REDUNDANT",
            "analysis_coordinates": "RANK_CERTIFIED_NONREDUNDANT_SMITH_VIEW",
            "nonbonded_subtraction": (
                "ANALYTIC_GAUSSIAN_ELECTROSTATICS_AND_VDW_BEFORE_LOCAL_FIT"
            ),
            "asymptotic_correctness": (
                "REQUIRED_INDEPENDENTLY_AT_EVERY_ZAFF_LEVEL_WITH_NO_BARE_QUADRATIC_TAIL"
            ),
        },
        "production_zoom": [
            "ZAFF0",
            "ZAFF1",
            "ZAFF2",
            "ZAFF3",
            "ZAFF4",
        ],
        "diagnostic_harmonic_levels": ["H0", "H1", "H2", "H3"],
        "physical_level_gate": (
            "ONLY_NATIVE_ASYMPTOTIC_ZAFF0_ZAFF1_ZAFF2_ZAFF3_ZAFF4_MAY_BE_PROMOTED"
        ),
        "level1_exception": (
            "DIAGONAL_PHYSICAL_FIELD_PLUS_DISSOCIATING_GEMINAL_XH_COUPLINGS_ONLY"
        ),
        "level1_stretches": (
            "MORSE_WITH_STATIONARITY_REFITTED_R0_RESIDUAL_QM_K2_AND_ZAFF_LIBRARY_DE"
        ),
        "level0_geometry": "RIGID_BONDS_AND_ANGLES_WITH_TORSIONAL_AND_RIGID_BODY_EXPLORATION",
    }


def build_trinity_field_from_zaff(
    xyzin: Path | str,
    force_field_path: Path | str,
    *,
    config: Any = None,
) -> Any:
    """Delegate ZAFF normal-mode sampling to TRINITY without duplicating it."""

    from matrix_trinity import build_normal_mode_anharmonic_field_from_architect

    return build_normal_mode_anharmonic_field_from_architect(
        xyzin,
        force_field_path,
        config=config,
    )


__all__ = [
    "ZAFF_WORKFLOW_SCHEMA",
    "build_trinity_field_from_zaff",
    "zaff_workflow_contract",
]
