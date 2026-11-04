from __future__ import annotations

MATRIX_XYZ_TOPOLOGY_SCHEMA = "matrix.xyz.topology.v3"
MATRIX_XYZ_SYNTHONS_SCHEMA = "matrix.xyz.synthons.v4"
MATRIX_XYZ_FRAGMENTS_SCHEMA = "matrix.xyz.fragments.v1"
MATRIX_XYZ_VALIDATION_SCHEMA = "matrix.xyz.validation.v1"
MATRIX_XYZ_AROMATICITY_SCHEMA = "matrix.xyz.aromaticity.v1"

# Read-only migration identifiers. New ORACLE states emit only the MATRIX
# schema above and never advertise an alternative scientific model.
LEGACY_ORACLE_XYZ_TOPOLOGY_SCHEMA_V2 = "oracle.xyz.topology.v2"
LEGACY_ORACLE_XYZ_SYNTHONS_SCHEMA_V2 = "oracle.xyz.synthons.v2"
LEGACY_ORACLE_XYZ_FRAGMENTS_SCHEMA = "oracle.xyz.fragments.v1"
LEGACY_ORACLE_XYZ_VALIDATION_SCHEMA = "oracle.xyz.validation.v1"

LEGACY_MATRIX_XYZ_TOPOLOGY_SCHEMA = "matrix.xyz.topology.v1"
LEGACY_MATRIX_XYZ_SYNTHONS_SCHEMA = "matrix.xyz.synthons.v1"
LEGACY_MATRIX_XYZ_TOPOLOGY_SCHEMA_V2 = "matrix.xyz.topology.v2"
LEGACY_MATRIX_XYZ_SYNTHONS_SCHEMA_V2 = "matrix.xyz.synthons.v2"
LEGACY_MATRIX_XYZ_SYNTHONS_SCHEMA_V3 = "matrix.xyz.synthons.v3"
LEGACY_ORACLE_XYZ_TOPOLOGY_SCHEMA = "oracle.xyz.topology.v1"
LEGACY_ORACLE_XYZ_SYNTHONS_SCHEMA = "oracle.xyz.synthons.v1"

SUPPORTED_TOPOLOGY_SCHEMAS = (
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    LEGACY_ORACLE_XYZ_TOPOLOGY_SCHEMA_V2,
    LEGACY_MATRIX_XYZ_TOPOLOGY_SCHEMA_V2,
    LEGACY_MATRIX_XYZ_TOPOLOGY_SCHEMA,
    LEGACY_ORACLE_XYZ_TOPOLOGY_SCHEMA,
)
SUPPORTED_SYNTHONS_SCHEMAS = (
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    LEGACY_MATRIX_XYZ_SYNTHONS_SCHEMA_V3,
    LEGACY_ORACLE_XYZ_SYNTHONS_SCHEMA_V2,
    LEGACY_MATRIX_XYZ_SYNTHONS_SCHEMA_V2,
    LEGACY_MATRIX_XYZ_SYNTHONS_SCHEMA,
    LEGACY_ORACLE_XYZ_SYNTHONS_SCHEMA,
)
SUPPORTED_FRAGMENTS_SCHEMAS = (
    MATRIX_XYZ_FRAGMENTS_SCHEMA,
    LEGACY_ORACLE_XYZ_FRAGMENTS_SCHEMA,
)
SUPPORTED_VALIDATION_SCHEMAS = (
    MATRIX_XYZ_VALIDATION_SCHEMA,
    LEGACY_ORACLE_XYZ_VALIDATION_SCHEMA,
)


def schema_line(schema: str) -> str:
    return f"SCHEMA {schema}"


def schema_from_line(line: str) -> str:
    text = str(line).strip()
    if not text.upper().startswith("SCHEMA "):
        return ""
    return text.split(None, 1)[1].strip()


def schema_line_supported(line: str, supported: tuple[str, ...]) -> bool:
    return schema_from_line(line) in supported


def supported_schema_text(supported: tuple[str, ...]) -> str:
    return " or ".join(f"SCHEMA {schema}" for schema in supported)
