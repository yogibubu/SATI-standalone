"""Serialization and parsing for frozen SMITH GIC definitions.

Persisted contracts can be read, audited and replayed independently of chart
construction and coordinate selection.
"""

from __future__ import annotations

from pathlib import Path

from matrix_chem import read_enriched_xyz
from matrix_core import read_sectioned_lines, section_content

from .component_invariants import validate_indivisible_gic_components
from .contracts import GICForgeContractError, ORACLE_XYZ_GIC_SCHEMA
from .coordinate_registry import MODE_BEARING_PRIMITIVE_FUNCTIONS
from .definition_io import (
    _apply_symmetry_group_limit,
    _key_values,
    _parse_atom_list,
    _point_group,
    _subsection,
    _symmetry_operations,
)
from .fallback_ledger import fallback_provenance_from_lines
from .models import (
    FrozenGIC,
    GICBMatrix,
    GICDefinition,
    GICPrimitive,
    GICReductionDiagnostics,
    GICSymmetrizationDiagnostics,
    GICSymmetrizedGroup,
)
from .periodic_estimates import parse_periodic_coordinate_estimate
from .policy import (
    FRAGMENT_MODE_NONE,
    FRAGMENT_MODE_PSEUDO_BONDS,
    FRAGMENT_MODE_SPECIAL_COORDINATES,
    FRAGMENT_MODES,
    GIC_BACKEND,
    LOCAL_SYMMETRIZATION_METHOD,
    POINT_GROUP_PROJECTOR_METHOD,
    RANK_METHOD,
    REDUCTION_POLICY,
    SALC_PATH_OVERLAP_WARNING_THRESHOLD,
    SYMMETRIZATION_POLICY,
    SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
)
from .semantic import AUTO_PROVENANCE, SEMANTIC_GRAMMAR_VERSION
from .symmetrization import validate_molecular_vibrational_irrep_counts
from .symmetry_labels import total_symmetric_irrep
from .topology_io import topology_aromatic_atoms_from_lines


def read_gic_definition_from_xyzin(
    path: Path,
    *,
    validate_symmetry_irreps: bool = True,
    allow_legacy_out_of_plane: bool = False,
) -> GICDefinition:
    """Read a frozen ORACLE GIC definition without regenerating coordinates.

    Active consumers always retain the default molecular-irrep contract gate
    and the Keymaker SOOP-default protocol. ``allow_legacy_out_of_plane`` is
    reserved for archival conversion code: scientific consumers must never
    receive Gaussian ``U`` or ``IMPD`` primitives as their default chart.
    """
    from .definition import (
        _normalize_pairs,
        _normalize_xh_classes,
        _xh_stretch_policy,
        sonic_definition_identity_sha256,
    )

    target = Path(path)
    lines = read_sectioned_lines(target)
    geometry = read_enriched_xyz(target)
    section = section_content(lines, "GIC")
    if not section:
        raise GICForgeContractError("missing #GIC section")
    if section[0].strip() != f"SCHEMA {ORACLE_XYZ_GIC_SCHEMA}":
        raise GICForgeContractError("invalid #GIC schema")
    status = _section_value(section, "STATUS")
    if (status or "").upper() != "BUILT":
        raise GICForgeContractError(f"#GIC status must be BUILT; found {status or 'UNKNOWN'}")
    primitives = tuple(
        _parse_primitive_line(line)
        for line in _subsection(section, "PRIMITIVES")
        if line.strip() and line.strip().upper() != "NONE"
    )
    # Preserve ORACLE's aromaticity assignment for native-U ring blocks; LINK
    # must never reperceive it from a later geometry.  A standalone frozen GIC
    # that contains no aromatic ring coordinate is self-sufficient and may be
    # replayed without a topology section.
    requires_aromatic_topology = any(
        primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "U"
        for primitive in primitives
    )
    aromatic_atoms = topology_aromatic_atoms_from_lines(
        lines,
        natoms=geometry.natoms,
        required=requires_aromatic_topology,
    )
    gics = tuple(
        _parse_frozen_gic_line(line)
        for line in _subsection(section, "FROZEN_GICS")
        if line.strip() and line.strip().upper() != "NONE"
    )
    diagnostics = _parse_reduction_diagnostics(
        section,
        selected=tuple(p.identifier for p in primitives),
    )
    symmetry_diagnostics = _parse_symmetry_diagnostics(section)
    primitive_source = (_section_value(section, "PRIMITIVE_SOURCE") or "").strip()
    if not primitive_source or primitive_source == "LEGACY_RECONSTRUCTED":
        raise GICForgeContractError("#GIC requires an explicit primitive source")
    primitive_source_schema = (_section_value(section, "PRIMITIVE_SOURCE_SCHEMA") or "").strip()
    primitive_b_matrix_sha256 = (_section_value(section, "PRIMITIVE_B_MATRIX_SHA256") or "").strip()
    if primitive_source == "ORACLE_CONTRACT" and (
        not primitive_source_schema
        or primitive_source_schema == "NONE"
        or not primitive_b_matrix_sha256
        or primitive_b_matrix_sha256 == "NONE"
    ):
        raise GICForgeContractError("#GIC ORACLE primitive metadata is incomplete")
    try:
        fallback_diagnostics, fallback_events = fallback_provenance_from_lines(
            _subsection(section, "FALLBACK_DIAGNOSTICS"),
            _subsection(section, "FALLBACK_LEDGER"),
        )
    except ValueError as exc:
        raise GICForgeContractError("malformed #GIC fallback ledger") from exc
    definition = GICDefinition(
        backend=_section_value(section, "BACKEND") or GIC_BACKEND,
        point_group=_section_value(section, "POINT_GROUP") or _point_group(lines),
        symmetrize=_parse_bool(_section_value(section, "SYMMETRIZE")),
        target_rank=_parse_int(_section_value(section, "TARGET_RANK")),
        rank=_parse_int(_section_value(section, "RANK")),
        candidate_count=_parse_int(_section_value(section, "CANDIDATE_COUNT")),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in geometry.coordinates_angstrom
        ),
        primitives=primitives,
        gics=gics,
        reduction_diagnostics=diagnostics,
        symmetry_diagnostics=symmetry_diagnostics,
        fragment_mode=_fragment_mode(_section_value(section, "FRAGMENT_MODE")),
        pseudo_bonds=_parse_pseudo_bonds(section),
        pseudo_bond_kinds=_parse_pseudo_bond_kinds(section),
        aromatic_atoms=tuple(sorted(int(atom) for atom in aromatic_atoms)),
        xh_stretch_policy=_xh_stretch_policy(_section_value(section, "XH_STRETCH_POLICY")),
        local_xh_bonds=_normalize_pairs(_section_value(section, "LOCAL_XH_BONDS")),
        local_xh_classes=_normalize_xh_classes(_section_value(section, "LOCAL_XH_CLASSES")),
        ring_puckering_diagnostics=tuple(
            line
            for line in _subsection(section, "RING_PUCKERING_DIAGNOSTICS")
            if line.strip() and line.strip().upper() != "NONE"
        ),
        periodic_coordinate_estimates=tuple(
            parse_periodic_coordinate_estimate(line)
            for line in _subsection(section, "PERIODIC_COORDINATE_ESTIMATES")
            if line.strip() and line.strip().upper() != "NONE"
        ),
        contract_schema_version=_section_value(section, "CONTRACT_SCHEMA_VERSION")
        or ORACLE_XYZ_GIC_SCHEMA,
        semantic_grammar_version=(
            ""
            if (_section_value(section, "SEMANTIC_GRAMMAR") or "NONE").upper() == "NONE"
            else (_section_value(section, "SEMANTIC_GRAMMAR") or SEMANTIC_GRAMMAR_VERSION)
        ),
        semantic_diagnostics=tuple(
            line
            for line in _subsection(section, "SEMANTIC_DIAGNOSTICS")
            if line.strip() and line.strip().upper() != "NONE"
        ),
        fallback_diagnostics=fallback_diagnostics,
        fallback_events=fallback_events,
        primitive_source=primitive_source,
        primitive_source_schema=(
            "" if primitive_source_schema == "NONE" else primitive_source_schema
        ),
        primitive_b_matrix_sha256=(
            "" if primitive_b_matrix_sha256 == "NONE" else primitive_b_matrix_sha256
        ),
        wilson_tangent_rank=_parse_int(_section_value(section, "WILSON_TANGENT_RANK")),
        wilson_tangent_singular_min=float(
            _section_value(section, "WILSON_TANGENT_SINGULAR_MIN") or 0.0
        ),
        wilson_tangent_singular_max=float(
            _section_value(section, "WILSON_TANGENT_SINGULAR_MAX") or 0.0
        ),
    )
    declared_fallback_count = _parse_int(_section_value(section, "FALLBACK_COUNT"))
    if declared_fallback_count != len(definition.fallback_events):
        raise GICForgeContractError("#GIC fallback count does not match its structured ledger")
    declared_identity = _section_value(section, "FROZEN_SONIC_IDENTITY_SHA256")
    actual_identity = sonic_definition_identity_sha256(definition)
    if declared_identity and declared_identity != actual_identity:
        raise GICForgeContractError(
            "#GIC frozen SONIC identity checksum does not match its coordinates"
        )
    validate_indivisible_gic_components(definition)
    _retained_point_group, retained_operations = _apply_symmetry_group_limit(
        _point_group(lines),
        _symmetry_operations(lines),
        definition.point_group,
    )
    if validate_symmetry_irreps:
        validate_molecular_vibrational_irrep_counts(
            definition,
            symmetry_operations=retained_operations,
        )
    legacy_out_of_plane = tuple(
        primitive.identifier
        for primitive in definition.primitives
        if primitive.function in {"U", "IMPD"}
    )
    if legacy_out_of_plane and not allow_legacy_out_of_plane:
        raise GICForgeContractError(
            "legacy out-of-plane primitives are not an active MATRIX chart; "
            "rebuild through Keymaker/SMITH with protocol sonic-soop-default.v1 "
            f"(legacy identifiers: {', '.join(legacy_out_of_plane)})"
        )
    return definition


def gic_b_matrix_lines(matrix: GICBMatrix) -> list[str]:
    """Serialize a GIC B matrix in a compact machine-readable text format."""
    lines = [
        "SCHEMA oracle.gic.bmatrix.v1",
        f"BACKEND {matrix.backend}",
        "UNITS MIXED_GIC_PER_ANGSTROM",
        "DERIVATIVE_MODE ANALYTIC",
        f"ROW_COUNT {len(matrix.rows)}",
        f"COLUMN_COUNT {len(matrix.cartesian_columns)}",
        "[COLUMNS]",
        " ".join(matrix.cartesian_columns) if matrix.cartesian_columns else "NONE",
        "[ROWS]",
    ]
    if not matrix.rows:
        lines.append("NONE")
        return lines
    for label, name, irrep, row in zip(
        matrix.coordinate_labels,
        matrix.coordinate_names,
        matrix.irreps,
        matrix.rows,
    ):
        values = ",".join(f"{value:.12g}" for value in row)
        lines.append(f"{label} NAME={name} IRREP={irrep} VALUES={values}")
    return lines


def write_gic_b_matrix(path: Path, output: Path) -> GICBMatrix:
    from .definition import build_gic_b_matrix_from_xyzin

    matrix = build_gic_b_matrix_from_xyzin(path)
    target = Path(output)
    target.write_text("\n".join(gic_b_matrix_lines(matrix)) + "\n", encoding="utf-8")
    return matrix


def _section_value(section_lines: list[str], key: str) -> str | None:
    key_upper = key.upper()
    for line in section_lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].upper() == key_upper:
            return parts[1].strip()
    return None


def _parse_primitive_line(line: str) -> GICPrimitive:
    parts = line.split()
    if not parts:
        raise GICForgeContractError("empty primitive line")
    fields = _key_values(parts[1:])
    try:
        return GICPrimitive(
            identifier=parts[0],
            name=fields["NAME"],
            family=fields["FAMILY"],
            function=fields["FUNCTION"],
            atoms=_parse_atom_list(fields["ATOMS"]),
            mode=int(fields.get("MODE", "0")),
            ref_atoms=_parse_atom_list(fields.get("REF_ATOMS", "")),
            refs=_parse_text_list(fields.get("REFS", "")),
            frame_atoms=_parse_atom_list(fields.get("FRAME_ATOMS", "")),
            ref_frame_atoms=_parse_atom_list(fields.get("REF_FRAME_ATOMS", "")),
            provenance=fields.get("PROVENANCE", AUTO_PROVENANCE).upper(),
            semantic_id=fields.get("SEMANTIC_ID", ""),
            semantic_type=fields.get("SEMANTIC_TYPE", "").upper(),
            chart=fields.get("CHART", "PRINCIPAL").upper(),
            chart_reference_radian=(
                float(fields["CHART_REFERENCE_RADIAN"])
                if "CHART_REFERENCE_RADIAN" in fields
                else None
            ),
        )
    except KeyError as exc:
        raise GICForgeContractError(f"invalid primitive line: {line}") from exc
    except ValueError as exc:
        raise GICForgeContractError(f"invalid primitive numeric field: {line}") from exc


def _parse_frozen_gic_line(line: str) -> FrozenGIC:
    parts = line.split()
    if not parts:
        raise GICForgeContractError("empty frozen GIC line")
    fields = _key_values(parts[1:])
    coefficients = _parse_coefficients(fields.get("COEFFS", ""))
    if not coefficients:
        primitive_id = fields.get("PRIMITIVE")
        if not primitive_id:
            raise GICForgeContractError(f"invalid frozen GIC coefficients: {line}")
        coefficients = ((primitive_id, 1.0),)
    try:
        return FrozenGIC(
            identifier=parts[0],
            name=fields["NAME"],
            family=fields["FAMILY"],
            irrep=fields["IRREP"],
            primitive_id=coefficients[0][0],
            gaussian_expression=fields.get("GAUSSIAN", "NONE"),
            coefficients=coefficients,
        )
    except KeyError as exc:
        raise GICForgeContractError(f"invalid frozen GIC line: {line}") from exc


def _parse_reduction_diagnostics(
    section_lines: list[str],
    *,
    selected: tuple[str, ...],
) -> GICReductionDiagnostics:
    lines = _subsection(section_lines, "REDUCTION_DIAGNOSTICS")
    if not lines:
        return GICReductionDiagnostics(
            rank_method=_section_value(section_lines, "RANK_METHOD") or RANK_METHOD,
            reduction_policy=_section_value(section_lines, "REDUCTION_POLICY") or REDUCTION_POLICY,
            selected=selected,
        )
    return GICReductionDiagnostics(
        rank_method=_section_value(lines, "RANK_METHOD") or RANK_METHOD,
        reduction_policy=_section_value(lines, "REDUCTION_POLICY") or REDUCTION_POLICY,
        selected=_parse_text_list(_section_value(lines, "SELECTED") or "") or selected,
        selected_by_family=_parse_text_list(_section_value(lines, "SELECTED_BY_FAMILY") or ""),
        skipped_singular=_parse_text_list(_section_value(lines, "SKIPPED_SINGULAR") or ""),
        skipped_dependent=_parse_text_list(_section_value(lines, "SKIPPED_DEPENDENT") or ""),
        skipped_singular_details=_parse_text_list(
            _section_value(lines, "SKIPPED_SINGULAR_DETAILS") or ""
        ),
        skipped_dependent_details=_parse_text_list(
            _section_value(lines, "SKIPPED_DEPENDENT_DETAILS") or ""
        ),
        conditioning_decisions=_parse_text_list(
            _section_value(lines, "CONDITIONING_DECISIONS") or ""
        ),
    )


def _parse_symmetry_diagnostics(
    section_lines: list[str],
) -> GICSymmetrizationDiagnostics | None:
    lines = _subsection(section_lines, "SYMMETRY_DIAGNOSTICS")
    if not lines:
        return None
    groups: list[GICSymmetrizedGroup] = []
    for line in lines:
        parts = line.split()
        if not parts or parts[0].upper() != "GROUP":
            continue
        fields = _key_values(parts[2:] if len(parts) > 1 else parts[1:])
        try:
            groups.append(
                GICSymmetrizedGroup(
                    block=fields["BLOCK"],
                    family=fields["FAMILY"],
                    signature=fields["SIGNATURE"],
                    source_gics=_parse_text_list(fields.get("SOURCES", "")),
                    output_gics=_parse_text_list(fields.get("OUTPUTS", "")),
                )
            )
        except KeyError as exc:
            raise GICForgeContractError(f"invalid symmetry diagnostic group line: {line}") from exc
    return GICSymmetrizationDiagnostics(
        method=_section_value(lines, "METHOD") or "UNKNOWN",
        policy=_section_value(lines, "POLICY") or SYMMETRIZATION_POLICY,
        status=_section_value(lines, "STATUS") or "UNKNOWN",
        point_group=_section_value(lines, "POINT_GROUP")
        or _section_value(section_lines, "POINT_GROUP")
        or "UNKNOWN",
        symmetry_group=_section_value(lines, "SYMMETRY_GROUP")
        or _section_value(section_lines, "SYMMETRY_GROUP")
        or _section_value(lines, "POINT_GROUP")
        or _section_value(section_lines, "POINT_GROUP")
        or "UNKNOWN",
        total_symmetric_irrep=_section_value(lines, "TOTAL_SYMMETRIC_IRREP")
        or _section_value(section_lines, "TOTAL_SYMMETRIC_IRREP")
        or total_symmetric_irrep(_section_value(section_lines, "POINT_GROUP")),
        total_symmetric_gics=_parse_text_list(
            _section_value(lines, "TOTAL_SYMMETRIC_GICS")
            or _section_value(section_lines, "TOTAL_SYMMETRIC_GICS")
            or ""
        ),
        groups=tuple(groups),
        sign_gauge_policy=_section_value(lines, "SIGN_GAUGE_POLICY")
        or "largest_abs_coefficient_pivot",
        path_gauge_policy=_section_value(lines, "PATH_GAUGE_POLICY")
        or "subspace_overlap_procrustes",
        path_overlap_warning_threshold=_parse_float_value(
            _section_value(lines, "PATH_OVERLAP_WARNING_THRESHOLD"),
            default=SALC_PATH_OVERLAP_WARNING_THRESHOLD,
        ),
        operation_tolerance_angstrom=_parse_float_value(
            _section_value(lines, "OPERATION_TOLERANCE_ANGSTROM"),
            default=SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
        ),
        max_operation_residual_angstrom=_parse_float_value(
            _section_value(lines, "MAX_OPERATION_RESIDUAL_ANGSTROM"),
            default=0.0,
        ),
        min_operation_margin_angstrom=_parse_float_value(
            _section_value(lines, "MIN_OPERATION_MARGIN_ANGSTROM"),
            default=0.0,
        ),
        near_threshold_operations=_parse_text_list(
            _section_value(lines, "NEAR_THRESHOLD_OPERATIONS") or ""
        ),
    )


def _parse_float_value(text: str | None, *, default: float) -> float:
    if text is None or text == "":
        return float(default)
    try:
        return float(text)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid float value: {text}") from exc


def _parse_text_list(text: str) -> tuple[str, ...]:
    if not text or text.upper() == "NONE":
        return ()
    return tuple(item for item in text.split(",") if item)


def _csv_or_none(values: tuple[str, ...]) -> str:
    return ",".join(values) if values else "NONE"


def _parse_coefficients(text: str) -> tuple[tuple[str, float], ...]:
    if not text:
        return ()
    coefficients: list[tuple[str, float]] = []
    for item in text.replace(";", ",").split(","):
        if not item:
            continue
        if ":" not in item:
            raise GICForgeContractError(f"invalid GIC coefficient: {item}")
        primitive_id, value = item.split(":", 1)
        try:
            coefficients.append((primitive_id, float(value)))
        except ValueError as exc:
            raise GICForgeContractError(f"invalid GIC coefficient value: {item}") from exc
    return tuple(coefficients)


def _parse_bool(text: str | None) -> bool:
    return bool((text or "").strip().upper() == "TRUE")


def _planned_xh_stretch_policy(lines: list[str]) -> str:
    from .definition import _xh_stretch_policy

    return _xh_stretch_policy(_section_value(section_content(lines, "GIC"), "XH_STRETCH_POLICY"))


def _planned_local_xh_bonds(lines: list[str]) -> tuple[tuple[int, int], ...]:
    from .definition import _normalize_pairs

    return _normalize_pairs(_section_value(section_content(lines, "GIC"), "LOCAL_XH_BONDS"))


def _planned_local_xh_classes(lines: list[str]) -> tuple[str, ...]:
    from .definition import _normalize_xh_classes

    return _normalize_xh_classes(_section_value(section_content(lines, "GIC"), "LOCAL_XH_CLASSES"))


def _fragment_mode(value: str | None) -> str:
    text = (value or FRAGMENT_MODE_NONE).strip().upper().replace("-", "_")
    aliases = {
        "PSEUDO": FRAGMENT_MODE_PSEUDO_BONDS,
        "PSEUDOBONDS": FRAGMENT_MODE_PSEUDO_BONDS,
        "HBOND": FRAGMENT_MODE_PSEUDO_BONDS,
        "HBONDS": FRAGMENT_MODE_PSEUDO_BONDS,
        "H_BONDS": FRAGMENT_MODE_PSEUDO_BONDS,
        "SPECIAL": FRAGMENT_MODE_SPECIAL_COORDINATES,
        "FRAGMENT": FRAGMENT_MODE_SPECIAL_COORDINATES,
        "FRAGMENT_COORDINATES": FRAGMENT_MODE_SPECIAL_COORDINATES,
    }
    normalized = aliases.get(text, text)
    if normalized not in FRAGMENT_MODES:
        raise GICForgeContractError(f"unsupported fragment mode: {value}")
    return normalized


def _out_of_plane_mode(primitives: tuple[GICPrimitive, ...]) -> str:
    if any(primitive.function == "SOOP" for primitive in primitives):
        return "SOOP"
    if any(
        primitive.function == "IMPD" or primitive.family == "IMPROPER_DIHEDRAL"
        for primitive in primitives
    ):
        return "IMPROPER_DIHEDRAL"
    return "OUT_OF_PLANE"


def _pairs_text(pairs: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{left}-{right}" for left, right in pairs) if pairs else "NONE"


def _csv_or_none_from_strings(values: tuple[str, ...]) -> str:
    return ",".join(values) if values else "NONE"


def _parse_int(text: str | None) -> int:
    if text is None:
        return 0
    try:
        return int(text)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid integer field: {text}") from exc


def _parse_pseudo_bonds(section: list[str]) -> tuple[tuple[int, int], ...]:
    bonds: list[tuple[int, int]] = []
    for line in _subsection(section, "PSEUDO_BONDS"):
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        if len(parts) < 3:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}")
        try:
            left = int(parts[1])
            right = int(parts[2])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}") from exc
        if left == right or left < 1 or right < 1:
            raise GICForgeContractError(f"invalid pseudo-bond indexes: {line}")
        bonds.append(tuple(sorted((left, right))))
    return tuple(sorted(set(bonds)))


def _parse_pseudo_bond_kinds(section: list[str]) -> tuple[str, ...]:
    kinds: list[tuple[tuple[int, int], str]] = []
    for line in _subsection(section, "PSEUDO_BONDS"):
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        if len(parts) < 3:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}")
        fields = _key_values(parts[3:])
        try:
            pair = tuple(sorted((int(parts[1]), int(parts[2]))))
        except ValueError as exc:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}") from exc
        kinds.append((pair, fields.get("KIND", "INTERFRAGMENT_CLOSEST").upper()))
    return tuple(kind for _pair, kind in sorted(kinds))


def _cartesian_column_labels(natoms: int) -> tuple[str, ...]:
    axes = ("X", "Y", "Z")
    return tuple(f"{atom}:{axis}" for atom in range(1, natoms + 1) for axis in axes)


def _primitive_line(primitive: GICPrimitive) -> str:
    atoms = ",".join(str(atom) for atom in primitive.atoms)
    mode = (
        f" MODE={primitive.mode}" if primitive.function in MODE_BEARING_PRIMITIVE_FUNCTIONS else ""
    )
    ref_atoms = (
        " REF_ATOMS=" + ",".join(str(atom) for atom in primitive.ref_atoms)
        if primitive.ref_atoms
        else ""
    )
    refs = " REFS=" + ",".join(primitive.refs) if primitive.refs else ""
    frame_atoms = (
        " FRAME_ATOMS=" + ",".join(str(atom) for atom in primitive.frame_atoms)
        if primitive.frame_atoms
        else ""
    )
    ref_frame_atoms = (
        " REF_FRAME_ATOMS=" + ",".join(str(atom) for atom in primitive.ref_frame_atoms)
        if primitive.ref_frame_atoms
        else ""
    )
    provenance = (
        f" PROVENANCE={primitive.provenance}" if primitive.provenance != AUTO_PROVENANCE else ""
    )
    semantic_id = f" SEMANTIC_ID={primitive.semantic_id}" if primitive.semantic_id else ""
    semantic_type = f" SEMANTIC_TYPE={primitive.semantic_type}" if primitive.semantic_type else ""
    chart = f" CHART={primitive.chart}" if primitive.chart != "PRINCIPAL" else ""
    chart_reference = (
        f" CHART_REFERENCE_RADIAN={primitive.chart_reference_radian:.17g}"
        if primitive.chart_reference_radian is not None
        else ""
    )
    return (
        f"{primitive.identifier} NAME={primitive.name} FAMILY={primitive.family} "
        f"CLASS={primitive.reduction_class} FUNCTION={primitive.function} "
        f"ATOMS={atoms}{ref_atoms}{refs}"
        f"{frame_atoms}{ref_frame_atoms}{mode}"
        f"{provenance}{semantic_id}{semantic_type}{chart}{chart_reference} "
        f"GAUSSIAN={primitive.gaussian_expression()}"
    )


def _frozen_gic_line(gic: FrozenGIC) -> str:
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    coeffs = ",".join(
        # Seventeen significant digits guarantee an exact binary-float
        # round trip.  Twelve fixed decimals changed SONIC identities for
        # highly symmetric ring combinations such as cubane and cyclohexane.
        f"{primitive_id}:{coefficient:.17g}"
        for primitive_id, coefficient in coefficients
    )
    return f"{gic.identifier} NAME={gic.name} FAMILY={gic.family} IRREP={gic.irrep} COEFFS={coeffs}"


def _sycart_components(vector: tuple[float, ...]) -> str:
    parts = []
    axes = ("X", "Y", "Z")
    for idx, value in enumerate(vector):
        if abs(value) <= 1.0e-12:
            continue
        atom = idx // 3 + 1
        axis = axes[idx % 3]
        parts.append(f"{atom}:{axis}={value:.12g}")
    return "COMPONENTS=" + (";".join(parts) if parts else "NONE")


def _symmetry_mode(definition: GICDefinition) -> str:
    diagnostics = definition.symmetry_diagnostics
    if (
        definition.symmetrize
        and diagnostics is not None
        and diagnostics.method == POINT_GROUP_PROJECTOR_METHOD
        and diagnostics.groups
    ):
        return "POINT_GROUP_PROJECTOR"
    if (
        definition.symmetrize
        and diagnostics is not None
        and diagnostics.method == LOCAL_SYMMETRIZATION_METHOD
        and diagnostics.groups
    ):
        return "LOCAL_BLOCK_C1" if definition.point_group.upper() == "C1" else "LOCAL_BLOCK"
    if definition.point_group.upper() == "C1":
        return "IDENTITY_C1" if definition.symmetrize else "UNSYMMETRIZED_C1"
    return "SYMMETRIZED" if definition.symmetrize else "UNSYMMETRIZED"


def _bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"
