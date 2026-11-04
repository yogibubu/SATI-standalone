"""Canonical artifact schemas owned by EPOC."""

from __future__ import annotations


EPOC_BOUNDARY_SCHEMA = "matrix.epoc.boundary.v1"
EPOC_REACTION_FIELD_SCHEMA = "matrix.epoc.reaction_field.v1"
EPOC_CONFINEMENT_SCHEMA = "matrix.epoc.confinement.v1"
EPOC_EVALUATION_SCHEMA = "matrix.epoc.evaluation.v1"

LEGACY_ZAFF_CPCM_SCHEMA = "matrix.zaff.cpcm_reaction_field.v1"
LEGACY_ZAFF_ASC_CPCM_SCHEMA = "matrix.zaff.ellipsoidal_asc_cpcm.v1"
LEGACY_ZAFF_VDW_CONFINEMENT_SCHEMA = "matrix.zaff.vdw_confinement.v1"
LEGACY_ZAFF_VARIATIONAL_CAVITY_SCHEMA = "matrix.zaff.variational_cavity.v1"
LEGACY_KEYMAKER_BOUNDARY_SCHEMA = "matrix.keymaker.confinement.v1"

EPOC_SCHEMAS = (
    EPOC_BOUNDARY_SCHEMA,
    EPOC_REACTION_FIELD_SCHEMA,
    EPOC_CONFINEMENT_SCHEMA,
    EPOC_EVALUATION_SCHEMA,
)


__all__ = [
    "EPOC_BOUNDARY_SCHEMA",
    "EPOC_CONFINEMENT_SCHEMA",
    "EPOC_EVALUATION_SCHEMA",
    "EPOC_REACTION_FIELD_SCHEMA",
    "EPOC_SCHEMAS",
    "LEGACY_ZAFF_ASC_CPCM_SCHEMA",
    "LEGACY_KEYMAKER_BOUNDARY_SCHEMA",
    "LEGACY_ZAFF_CPCM_SCHEMA",
    "LEGACY_ZAFF_VARIATIONAL_CAVITY_SCHEMA",
    "LEGACY_ZAFF_VDW_CONFINEMENT_SCHEMA",
]
