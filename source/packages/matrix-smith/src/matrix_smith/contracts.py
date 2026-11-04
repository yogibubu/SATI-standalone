from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import TYPE_CHECKING

from matrix_core import read_sectioned_lines, replace_section, section_content
from matrix_chem.topology.contracts import (
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    MATRIX_XYZ_VALIDATION_SCHEMA,
    SUPPORTED_VALIDATION_SCHEMAS,
    schema_line_supported,
)
from matrix_chem import (
    ORACLE_SONIC_CONTRACT_SCHEMA,
    OracleSonicContract,
    OracleSonicContractError,
    GeometryIdentityError,
    geometry_identity_payload_sha256,
    read_geometry_identity_certificate,
    read_oracle_sonic_contract,
)
from matrix_gaussian import DEFAULT_GAUSSIAN_GIC_MAX_ADDENDS

if TYPE_CHECKING:
    from matrix_gaussian import GaussianRouteOverride


ORACLE_XYZ_GIC_SCHEMA = "oracle.xyz.gic.v1"
ORACLE_XYZ_SYCART_SCHEMA = "oracle.xyz.sycart.v1"
REQUIRED_VALIDATION_SCHEMA = MATRIX_XYZ_VALIDATION_SCHEMA


class GICForgeContractError(ValueError):
    """Raised when GICForge cannot consume the enriched XYZ state."""


class GICForgeRankDeficiencyError(GICForgeContractError):
    """Raised when an atlas-compliant candidate pool cannot reach exact rank."""

    def __init__(self, *, target_rank: int, selected_rank: int, candidate_count: int) -> None:
        self.target_rank = int(target_rank)
        self.selected_rank = int(selected_rank)
        self.candidate_count = int(candidate_count)
        super().__init__(
            "insufficient independent primitive coordinates: "
            f"need {self.target_rank}, selected rank {self.selected_rank} "
            f"from {self.candidate_count} candidates"
        )


def load_frozen_oracle_sonic_contract(path: Path) -> OracleSonicContract:
    """Load the mandatory ORACLE-owned chemistry contract without reperception.

    This entry point deliberately wraps the shared transport validator in a
    SMITH-specific error.  It never repairs, supplements, or infers missing
    chemical records.
    """

    source = Path(path)
    try:
        stamp = source.stat().st_mtime_ns
    except OSError as exc:
        raise GICForgeContractError(f"cannot stat ORACLE SONIC contract: {source}") from exc
    contract = _load_frozen_oracle_sonic_contract(str(source), int(stamp))
    if contract.geometry_identity_payload_sha256:
        try:
            identity = read_geometry_identity_certificate(source)
        except GeometryIdentityError as exc:
            raise GICForgeContractError(
                f"invalid or missing ORACLE Cartesian provenance: {exc}"
            ) from exc
        if (
            geometry_identity_payload_sha256(identity)
            != contract.geometry_identity_payload_sha256
        ):
            raise GICForgeContractError(
                "ORACLE SONIC contract contradicts its Cartesian provenance"
            )
    return contract


@lru_cache(maxsize=16)
def _load_frozen_oracle_sonic_contract(path: str, _stamp: int) -> OracleSonicContract:
    try:
        contract = read_oracle_sonic_contract(Path(path))
    except OracleSonicContractError as exc:
        raise GICForgeContractError(
            f"invalid or missing ORACLE SONIC contract ({ORACLE_SONIC_CONTRACT_SCHEMA}): {exc}"
        ) from exc
    return contract


def validate_complete_frozen_oracle_semantics(contract: OracleSonicContract) -> None:
    """Require the complete v2 chemistry state before production SONIC use.

    This is a coverage check over ORACLE-owned records already present in the
    contract.  It deliberately performs no geometric or chemical perception.
    """

    from matrix_chem.soop_contract import frozen_cyclic_partition_manifest

    frozen_cyclic_partition_manifest()
    if contract.schema != ORACLE_SONIC_CONTRACT_SCHEMA:
        raise GICForgeContractError(
            "production SONIC requires the current ORACLE contract schema"
        )
    if "MIGRATED_V1_TO_V2_NO_LOCAL_RECONSTRUCTION" in contract.provenance:
        raise GICForgeContractError(
            "migrated v1 contract has no local perception; ORACLE must rebuild it"
        )
    if len(contract.chemical_policy_sha256) != 64:
        raise GICForgeContractError(
            "ORACLE contract is missing the frozen chemical-policy fingerprint"
        )
    if len(contract.reference_geometry_sha256) != 64 or len(
        contract.geometry_identity_payload_sha256
    ) != 64:
        raise GICForgeContractError(
            "ORACLE contract is missing frozen Cartesian-provenance fingerprints"
        )

    topology = contract.primary_topology
    degree = [0] * topology.natoms
    for left, right in topology.bonds:
        degree[left - 1] += 1
        degree[right - 1] += 1
    if contract.primitive_mode == "CHEMICAL_PRIMITIVES":
        from matrix_chem.soop_contract import expected_soop_support_keys, soop_support_key

        expected_soop = expected_soop_support_keys(topology)
        actual_soop = {soop_support_key(primitive)
                       for primitive in contract.selective_out_of_plane_primitives}
        if not expected_soop.issubset(actual_soop):
            raise GICForgeContractError(
                "ORACLE SOOP primitives are incomplete; regenerate the primitive contract before SMITH"
            )
    expected_centers = {index + 1 for index, value in enumerate(degree) if value >= 2}
    supplied_centers = {
        domain.center_atom
        for domain in contract.local_perception_domains
        if domain.kind == "ATOM_CENTER"
    }
    if supplied_centers != expected_centers:
        missing = sorted(expected_centers - supplied_centers)
        extra = sorted(supplied_centers - expected_centers)
        raise GICForgeContractError(
            "ORACLE local atom-center semantics are incomplete "
            f"(missing={missing}, extra={extra})"
        )

    expected_rings = {tuple(sorted(ring)) for ring in topology.rings}
    supplied_rings = {
        tuple(sorted(domain.members))
        for domain in contract.local_perception_domains
        if domain.kind == "RING"
    }
    if supplied_rings != expected_rings:
        raise GICForgeContractError(
            "ORACLE local ring semantics are incomplete or inconsistent"
        )
    required_thresholds = {
        "ZEFF_EQUIVALENCE",
        "RADIAL_EQUIVALENCE",
        "TEMPLATE_RMS",
        "TEMPLATE_MARGIN",
        "ANGLE_CLASS",
    }
    for domain in contract.local_perception_domains:
        available = {name for name, _value, _unit in domain.thresholds}
        if not required_thresholds.issubset(available):
            raise GICForgeContractError(
                f"ORACLE local domain {domain.domain_id} lacks frozen threshold provenance"
            )


def validate_gicforge_prerequisites(path: Path) -> None:
    lines = read_sectioned_lines(Path(path))
    validation = section_content(lines, "VALIDATION")
    if not validation:
        raise GICForgeContractError("missing #VALIDATION section")
    expected = f"SCHEMA {REQUIRED_VALIDATION_SCHEMA}"
    if not schema_line_supported(validation[0], SUPPORTED_VALIDATION_SCHEMAS):
        raise GICForgeContractError(
            f"#VALIDATION must start with {expected!r}; found {validation[0]!r}"
        )
    status = _validation_status(validation)
    if status != "PASS":
        raise GICForgeContractError(f"#VALIDATION status must be PASS; found {status or 'UNKNOWN'}")


def gic_plan_section_lines(
    *,
    symmetrize: bool = False,
    improper_dihedrals: bool = False,
    fragment_mode: str = "SPECIAL_COORDINATES",
    xh_stretch_policy: str = "SYMMETRIZE",
    local_xh_bonds: tuple[tuple[int, int], ...] = (),
    local_xh_classes: tuple[str, ...] = (),
) -> list[str]:
    mode = _normalized_fragment_mode(fragment_mode)
    policy = str(xh_stretch_policy or "SYMMETRIZE").strip().replace("-", "_").upper()
    bonds = (
        ",".join(f"{int(left)}-{int(right)}" for left, right in local_xh_bonds)
        if local_xh_bonds
        else "NONE"
    )
    classes = (
        ",".join(str(item).strip().upper() for item in local_xh_classes)
        if local_xh_classes
        else "NONE"
    )
    return [
        f"SCHEMA {ORACLE_XYZ_GIC_SCHEMA}",
        "STATUS PLANNED",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} "
        f"TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA} "
        "SYMMETRY=oracle.xyz.symmetry.v1",
        "INDEXING ATOMS=ONE_BASED",
        f"SYMMETRIZE {_bool_text(symmetrize)}",
        "OUT_OF_PLANE_MODE SOOP",
        f"FRAGMENT_MODE {mode}",
        f"XH_STRETCH_POLICY {policy}",
        f"LOCAL_XH_BONDS {bonds}",
        f"LOCAL_XH_CLASSES {classes}",
        "BACKEND UNASSIGNED",
        "[FROZEN_GICS]",
        "PENDING GICFORGE_IMPLEMENTATION",
    ]


def sycart_plan_section_lines() -> list[str]:
    return [
        f"SCHEMA {ORACLE_XYZ_SYCART_SCHEMA}",
        "STATUS PLANNED",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} GIC=oracle.xyz.gic.v1",
        "INDEXING ATOMS=ONE_BASED",
        "[SYCART]",
        "PENDING GICFORGE_IMPLEMENTATION",
    ]


def write_gicforge_plan_sections(
    path: Path,
    *,
    symmetrize: bool = False,
    sycart: bool = False,
    improper_dihedrals: bool = False,
    fragment_mode: str = "SPECIAL_COORDINATES",
    xh_stretch_policy: str = "SYMMETRIZE",
    local_xh_bonds: tuple[tuple[int, int], ...] = (),
    local_xh_classes: tuple[str, ...] = (),
) -> None:
    target = Path(path)
    validate_gicforge_prerequisites(target)
    replace_section(
        target,
        "GIC",
        gic_plan_section_lines(
            symmetrize=symmetrize,
            improper_dihedrals=improper_dihedrals,
            fragment_mode=fragment_mode,
            xh_stretch_policy=xh_stretch_policy,
            local_xh_bonds=local_xh_bonds,
            local_xh_classes=local_xh_classes,
        ),
    )
    if sycart:
        replace_section(target, "SYCART", sycart_plan_section_lines())


def write_gicforge_gaussian_input(
    path: Path,
    output: Path,
    *,
    route: str = "#p hf/sto-3g opt=readallgic",
    title: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    link0: tuple[str, ...] = (),
    basis_set_file: Path | str | None = None,
    total_symmetric_only: bool = False,
    freeze_non_total: bool | None = None,
    g16_compatibility: bool = False,
    max_gic_expression_addends: int | None = DEFAULT_GAUSSIAN_GIC_MAX_ADDENDS,
    route_override: GaussianRouteOverride | None = None,
) -> Path:
    """Create parser-safe Gaussian input with a complete independent chart.

    The default is the native GDV/SONIC representation.  Commercial Gaussian
    compatibility transformations require an explicit opt-in.  Native export
    consumes SMITH's complete chart and activation state literally: the writer
    cannot select a symmetry subset or infer Frozen coordinates.  TS physical
    charts are therefore completely active; minimum/exploration retain only
    the constraints already encoded by their SMITH contracts.  Diagnostic
    ``total_symmetric_only`` transformations belong exclusively to the
    explicitly requested Gaussian-16 compatibility path.
    """
    validate_gicforge_prerequisites(Path(path))
    from matrix_gaussian import (
        validate_gaussian_readallgic_input,
        write_gicforge_gaussian_input as write_gaussian,
    )

    written = write_gaussian(
        Path(path),
        Path(output),
        route=route,
        title=title,
        charge=charge,
        multiplicity=multiplicity,
        link0=link0,
        basis_set_file=basis_set_file,
        total_symmetric_only=total_symmetric_only,
        freeze_non_total=freeze_non_total,
        g16_compatibility=g16_compatibility,
        max_gic_expression_addends=max_gic_expression_addends,
        route_override=route_override,
    )
    validate_gaussian_readallgic_input(
        written,
        reference_xyzin=Path(path),
        g16_compatibility=g16_compatibility,
    )
    return written


def _validation_status(validation_lines: list[str]) -> str | None:
    for line in validation_lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "STATUS":
            return parts[1].upper()
    return None


def _bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def _normalized_fragment_mode(value: str | None) -> str:
    text = (value or "SPECIAL_COORDINATES").strip().upper().replace("-", "_")
    aliases = {
        "PSEUDO": "PSEUDO_BONDS",
        "PSEUDOBONDS": "PSEUDO_BONDS",
        "HBOND": "PSEUDO_BONDS",
        "HBONDS": "PSEUDO_BONDS",
        "H_BONDS": "PSEUDO_BONDS",
        "SPECIAL": "SPECIAL_COORDINATES",
        "FRAGMENT_COORDINATES": "SPECIAL_COORDINATES",
        "FRAGMENT": "SPECIAL_COORDINATES",
        "NONE": "NONE",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"SPECIAL_COORDINATES", "PSEUDO_BONDS", "NONE"}:
        raise GICForgeContractError(f"unsupported fragment mode: {value}")
    return normalized
