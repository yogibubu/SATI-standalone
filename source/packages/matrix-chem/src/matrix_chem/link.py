from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping


from matrix_core import (
    BasicSection,
    read_sectioned_lines,
    replace_section,
    replace_xyz_block,
    section_content,
    write_sectioned_lines,
    write_basic_section,
)

from .geometry import MolecularGeometry
from .geometry_identity import (
    GEOMETRY_CHANGE_NOT_AUTHORIZED,
    GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION,
    GEOMETRY_TRUE_CHANGE,
    GeometryIdentityCertificate,
    build_geometry_identity_certificate,
    write_geometry_identity_certificate,
)
from .geometry_io import GeometrySourceKind, read_geometry_with_kind
from .symmetry import (
    CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM,
    analyze_molecular_quasisymmetry,
    canonicalize_molecular_geometry_by_symmetry,
    MolecularSymmetry,
    SymmetryOperation,
    SymmetryProjectionDiagnostics,
    analyze_molecular_symmetry,
    orient_molecular_geometry_to_principal_axes,
    symmetry_section_lines,
    symmetry_projection_diagnostics,
    symmetrize_molecular_geometry,
)
from .topology.contracts import (
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
)
from .topology.pipeline import build_topology_objects
from .topology.aromaticity import aromaticity_section_lines
from .topology.automorphisms import (
    build_topology_automorphism_orbits,
    topology_automorphism_edge_labels,
    topology_automorphism_lines,
)
from .primitive_coordinates import build_primitive_contract, write_primitive_contract


@dataclass(frozen=True)
class SymmetryThresholds:
    distance_angstrom: float = 1.0e-3
    inertia_relative: float = 1.0e-3
    max_rotation_order: int = 6


@dataclass(frozen=True)
class LinkPreprocessResult:
    path: Path
    geometry: MolecularGeometry
    point_group: str
    topology_bond_count: int
    ring_count: int
    cartesian_symmetry_status: str
    symmetrization_required_threshold_angstrom: float | None
    cartesian_symmetrization_decision: str
    input_geometry: MolecularGeometry
    input_symmetry_max_deviation_angstrom: float
    proposed_point_group: str
    geometry_identity: GeometryIdentityCertificate


def preprocess_to_enriched_xyz(
    source: Path,
    target: Path,
    *,
    source_kind: GeometrySourceKind = "auto",
    symmetry_thresholds: SymmetryThresholds = SymmetryThresholds(),
    cartesian_symmetrization: str = "apply",
    external_charges: Mapping[int, float] | None = None,
    bond_order_overrides: Mapping[tuple[int, int], float] | None = None,
    charge_source: str = "ORACLE electronegativity estimate",
    bond_order_source: str = "ORACLE Pauling estimate",
    electronic_connectivity: bool = False,
) -> LinkPreprocessResult:
    """Import a geometry source and materialize initial MATRIX sections.

    Program-specific and SMILES adapters call into LINK once they have produced
    a `MolecularGeometry`.
    """
    imported_geometry = read_geometry_with_kind(Path(source), source_kind)
    mode = str(cartesian_symmetrization).strip().casefold()
    if mode not in {"inspect", "apply", "retain"}:
        raise ValueError(
            "cartesian_symmetrization must be 'inspect', 'apply', or 'retain'"
        )
    oriented_geometry = orient_molecular_geometry_to_principal_axes(
        imported_geometry,
        inertia_tolerance=symmetry_thresholds.inertia_relative,
    )
    initial_topology_bonds, initial_topology_edge_labels = _primary_topology_signature(
        oriented_geometry,
        external_charges=external_charges,
        bond_order_overrides=bond_order_overrides,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
        electronic_connectivity=electronic_connectivity,
    )
    quasi_analysis = analyze_molecular_quasisymmetry(
        oriented_geometry,
        distance_tolerance=symmetry_thresholds.distance_angstrom,
        inertia_tolerance=symmetry_thresholds.inertia_relative,
        max_rotation_order=symmetry_thresholds.max_rotation_order,
        topology_bonds=initial_topology_bonds,
        topology_edge_labels=initial_topology_edge_labels,
    )
    assigned_symmetry = quasi_analysis.proposed_symmetry
    recognition_tolerance = quasi_analysis.recognition_tolerance_angstrom
    if quasi_analysis.promoted:
        projected_preview = symmetrize_molecular_geometry(
            oriented_geometry,
            assigned_symmetry,
            minimum_deviation_angstrom=0.0,
            force_projection=True,
        )
        if _primary_topology_signature(
            projected_preview,
            external_charges=external_charges,
            bond_order_overrides=bond_order_overrides,
            charge_source=charge_source,
            bond_order_source=bond_order_source,
            electronic_connectivity=electronic_connectivity,
        ) != (
            initial_topology_bonds,
            initial_topology_edge_labels,
        ):
            # A quasi-symmetry projection is never allowed to alter ORACLE's
            # primary connectivity.  Retain the strict assignment instead.
            assigned_symmetry = quasi_analysis.strict_symmetry
            recognition_tolerance = symmetry_thresholds.distance_angstrom
    input_symmetry_max_deviation = float(assigned_symmetry.max_deviation)
    proposed_point_group = str(assigned_symmetry.point_group)
    nontrivial_group = assigned_symmetry.point_group.strip().upper() not in {
        "", "C1", "UNKNOWN"
    }
    quasi_symmetry = bool(
        nontrivial_group
        and assigned_symmetry.max_deviation
        > CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM
    )
    required_threshold = (
        float(assigned_symmetry.max_deviation) if quasi_symmetry else None
    )
    projection_applied = False
    if quasi_symmetry and mode == "apply":
        geometry, assigned_symmetry, symmetry = canonicalize_molecular_geometry_by_symmetry(
            imported_geometry,
            distance_tolerance=recognition_tolerance,
            inertia_tolerance=symmetry_thresholds.inertia_relative,
            max_rotation_order=symmetry_thresholds.max_rotation_order,
            topology_bonds=initial_topology_bonds,
            topology_edge_labels=initial_topology_edge_labels,
            coordinate_decimals=10,
        )
        projection = symmetry_projection_diagnostics(
            oriented_geometry,
            geometry,
            assigned_symmetry,
        )
        cartesian_symmetry_status = "QUASI_SYMMETRY_PROJECTED"
        cartesian_symmetrization_decision = "SYMMETRIZE"
        projection_applied = True
    else:
        symmetry = assigned_symmetry
        if quasi_symmetry:
            geometry = oriented_geometry
            # Inspection/retention never accepts a higher quasi-symmetry
            # group implicitly.  Keep the exact retained geometry's group as
            # the operative contract and serialize the topology-qualified
            # larger group only in diagnostics until PROJECT is explicit.
            symmetry = analyze_molecular_symmetry(
                oriented_geometry,
                distance_tolerance=CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM,
                inertia_tolerance=symmetry_thresholds.inertia_relative,
                max_rotation_order=symmetry_thresholds.max_rotation_order,
            )
            projection_status = (
                "AWAITING_USER_CONFIRMATION" if mode == "inspect" else "DECLINED_BY_USER"
            )
            if mode == "retain":
                cartesian_symmetry_status = "QUASI_SYMMETRY_RETAINED"
            else:
                cartesian_symmetry_status = "QUASI_SYMMETRY"
            cartesian_symmetrization_decision = (
                "PENDING" if mode == "inspect" else "RETAIN"
            )
        elif nontrivial_group:
            if mode == "apply":
                geometry = symmetrize_molecular_geometry(
                    oriented_geometry,
                    assigned_symmetry,
                    minimum_deviation_angstrom=0.0,
                )
                projection = symmetry_projection_diagnostics(
                    oriented_geometry,
                    geometry,
                    assigned_symmetry,
                )
                projection_status = "APPLIED_WITHIN_NUMERICAL_SYMMETRY_BAND"
                cartesian_symmetrization_decision = "SYMMETRIZE"
                projection_applied = True
            else:
                geometry = oriented_geometry
                projection = SymmetryProjectionDiagnostics(
                    status="NOT_REQUIRED_EXACT_SYMMETRY",
                    max_displacement_angstrom=0.0,
                    rms_displacement_angstrom=0.0,
                )
                projection_status = "NOT_REQUIRED_EXACT_SYMMETRY"
                cartesian_symmetrization_decision = "NOT_REQUIRED"
            cartesian_symmetry_status = "EXACT_SYMMETRY"
        else:
            geometry = oriented_geometry
            projection_status = "NOT_APPLICABLE_C1"
            cartesian_symmetry_status = "C1"
            cartesian_symmetrization_decision = "NOT_APPLICABLE"
        if not (nontrivial_group and mode == "apply"):
            projection = SymmetryProjectionDiagnostics(
                status=projection_status,
                max_displacement_angstrom=0.0,
                rms_displacement_angstrom=0.0,
            )
    # A newly imported Cartesian geometry starts a new ORACLE state.  Do not
    # retain stale downstream sections (for example SMITH #GIC) from an older
    # geometry that happened to use the same output path.
    write_sectioned_lines(Path(target), geometry.xyz_lines())
    write_source_section(
        target, source=Path(source), source_kind=source_kind, geometry=imported_geometry
    )
    serialized_geometry = read_geometry_with_kind(Path(target), "enriched_xyz")
    geometry_identity = build_geometry_identity_certificate(
        imported_geometry.atoms,
        imported_geometry.coordinates_angstrom,
        serialized_geometry.atoms,
        serialized_geometry.coordinates_angstrom,
        geometry_change_authorization=GEOMETRY_CHANGE_NOT_AUTHORIZED,
    )
    if projection_applied and geometry_identity.relation == GEOMETRY_TRUE_CHANGE:
        geometry_identity = build_geometry_identity_certificate(
            imported_geometry.atoms,
            imported_geometry.coordinates_angstrom,
            serialized_geometry.atoms,
            serialized_geometry.coordinates_angstrom,
            geometry_change_authorization=GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION,
        )
    write_geometry_identity_certificate(target, geometry_identity)
    write_gaussian_topology_section(target, source=Path(source))
    write_smiles_section(target, imported_geometry)
    # From this point onward every ORACLE and downstream tool consumes only
    # the centered, principal-axis-oriented and symmetry-projected geometry.
    bond_count, ring_count = write_topology_and_synthons_sections(
        target,
        geometry,
        external_charges=external_charges,
        bond_order_overrides=bond_order_overrides,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
        electronic_connectivity=electronic_connectivity,
    )
    point_group = symmetry.point_group
    write_basic_section_from_geometry(target, geometry=geometry, point_group=point_group)
    write_symmetry_section(
        target,
        symmetry=symmetry,
        thresholds=symmetry_thresholds,
        projection=projection,
        cartesian_symmetry_status=cartesian_symmetry_status,
        symmetrization_required_threshold_angstrom=required_threshold,
        cartesian_symmetrization_decision=cartesian_symmetrization_decision,
        input_symmetry_max_deviation_angstrom=input_symmetry_max_deviation,
        proposed_point_group=proposed_point_group,
    )
    write_primitive_coordinate_section(
        target,
        geometry,
        external_charges=external_charges,
        bond_order_overrides=bond_order_overrides,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
        electronic_connectivity=electronic_connectivity,
    )
    return LinkPreprocessResult(
        path=Path(target),
        geometry=geometry,
        point_group=point_group,
        topology_bond_count=bond_count,
        ring_count=ring_count,
        cartesian_symmetry_status=cartesian_symmetry_status,
        symmetrization_required_threshold_angstrom=required_threshold,
        cartesian_symmetrization_decision=cartesian_symmetrization_decision,
        input_geometry=imported_geometry,
        input_symmetry_max_deviation_angstrom=input_symmetry_max_deviation,
        proposed_point_group=proposed_point_group,
        geometry_identity=geometry_identity,
    )


def _primary_topology_signature(
    geometry: MolecularGeometry,
    *,
    external_charges=None,
    bond_order_overrides=None,
    charge_source="ORACLE electronegativity estimate",
    bond_order_source="ORACLE Pauling estimate",
    electronic_connectivity=False,
) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], str]]:
    atomic_numbers = tuple(_atomic_number(atom) for atom in geometry.atoms)
    _continuous, discrete, _rings, synthons, aromaticity = build_topology_objects(
        geometry.coordinates_angstrom,
        atomic_numbers,
        external_charges=external_charges,
        bond_order_overrides=bond_order_overrides,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
        electronic_connectivity=electronic_connectivity,
    )
    bonds = tuple(
        sorted(tuple(sorted((int(left), int(right)))) for left, right in discrete.bonds)
    )
    edge_labels = topology_automorphism_edge_labels(
        discrete,
        synthons,
        aromaticity=aromaticity,
    )
    return bonds, edge_labels


def write_primitive_coordinate_section(
    path: Path,
    geometry: MolecularGeometry,
    *,
    external_charges=None,
    bond_order_overrides=None,
    charge_source="ORACLE electronegativity estimate",
    bond_order_source="ORACLE Pauling estimate",
    electronic_connectivity=False,
) -> None:
    """Freeze ORACLE redundant primitives and their reference Wilson matrix."""
    # Build from the serialized Cartesian block, not the higher-precision import
    # object, so its fingerprints remain stable after the XYZ text round trip.
    serialized_geometry = read_geometry_with_kind(Path(path), "enriched_xyz")
    atomic_numbers = [_atomic_number(atom) for atom in serialized_geometry.atoms]
    electronic = electronic_population_overrides_from_xyzin(Path(path))
    selected_charges = external_charges if external_charges is not None else electronic["charges"]
    selected_orders = (
        bond_order_overrides if bond_order_overrides is not None else electronic["bond_orders"]
    )
    _continuous, discrete, _rings, _synthons, _aromaticity = build_topology_objects(
        serialized_geometry.coordinates_angstrom,
        atomic_numbers,
        bond_order_overrides=selected_orders,
        external_charges=selected_charges,
        charge_source=charge_source if external_charges is not None else electronic["charge_source"],
        bond_order_source=(
            bond_order_source if bond_order_overrides is not None else electronic["bond_order_source"]
        ),
        electronic_connectivity=electronic_connectivity,
    )
    contract = build_primitive_contract(discrete, serialized_geometry.coordinates_angstrom)
    write_primitive_contract(Path(path), contract)
    # The PI extension has the same frozen geometry/topology lifecycle as the
    # redundant primitives. Reuse this perception pass rather than enumerating
    # the automorphisms of the full molecular graph a second time.
    frozen_sections = read_sectioned_lines(Path(path))
    if all(section_content(frozen_sections, name) for name in ("SYMMETRY", "TOPOLOGY")):
        from .permutation_inversion import (
            build_permutation_inversion_symmetry,
            write_permutation_inversion_symmetry,
        )

        pi_symmetry = build_permutation_inversion_symmetry(
            serialized_geometry,
            read_molecular_symmetry(Path(path)),
            bonds=discrete.bonds,
            edge_labels=topology_automorphism_edge_labels(
                discrete, _synthons, aromaticity=_aromaticity,
            ),
        )
        write_permutation_inversion_symmetry(Path(path), pi_symmetry)


def write_smiles_section(path: Path, geometry: MolecularGeometry) -> None:
    """Preserve a canonical SMILES source for GUI depiction and provenance."""
    smiles = str(geometry.metadata.get("smiles", "")).strip()
    if not smiles:
        return
    replace_section(
        Path(path),
        "SMILES",
        ["SCHEMA oracle.xyz.smiles.v1", f"VALUE {smiles}"],
    )


BabelPreprocessResult = LinkPreprocessResult


def write_enriched_geometry(path: Path, geometry: MolecularGeometry) -> None:
    replace_xyz_block(Path(path), geometry.xyz_lines())


def write_source_section(
    path: Path,
    *,
    source: Path,
    source_kind: str,
    geometry: MolecularGeometry,
) -> None:
    replace_section(
        Path(path),
        "SOURCE",
        [
            "SCHEMA oracle.xyz.source.v1",
            f"KIND {source_kind}",
            f"FORMAT {geometry.source_format}",
            f"PATH {source}",
        ],
    )


def write_basic_section_from_geometry(
    path: Path,
    *,
    geometry: MolecularGeometry,
    point_group: str,
) -> None:
    write_basic_section(
        Path(path),
        BasicSection(
            charge=0 if geometry.charge is None else int(geometry.charge),
            multiplicity=1 if geometry.multiplicity is None else int(geometry.multiplicity),
            point_group=point_group,
        ),
    )


def write_gaussian_topology_section(path: Path, *, source: Path) -> int:
    """Write Gaussian CM5/Mayer annotations when the import source provides them."""
    suffix = Path(source).suffix.lower()
    if suffix not in {".log", ".out"}:
        return 0
    from matrix_gaussian import gaussian_topology_section_lines

    lines = gaussian_topology_section_lines(Path(source))
    if not lines:
        return 0
    replace_section(Path(path), "GAUSSIAN_TOPOLOGY", lines)
    return len(lines)


def determine_initial_symmetry(
    geometry: MolecularGeometry,
    thresholds: SymmetryThresholds,
) -> object:
    return analyze_molecular_symmetry(
        geometry,
        distance_tolerance=thresholds.distance_angstrom,
        inertia_tolerance=thresholds.inertia_relative,
        max_rotation_order=thresholds.max_rotation_order,
    )


def determine_initial_point_group(
    geometry: MolecularGeometry,
    thresholds: SymmetryThresholds,
) -> str:
    return str(determine_initial_symmetry(geometry, thresholds).point_group)


def read_symmetry_thresholds(path: Path) -> SymmetryThresholds:
    """Read the frozen ORACLE symmetry thresholds from an enriched XYZ file."""
    content = section_content(read_sectioned_lines(Path(path)), "SYMMETRY")
    values: dict[str, str] = {}
    for line in content:
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            values[fields[0]] = fields[1]
    try:
        return SymmetryThresholds(
            distance_angstrom=float(values["THRESHOLD_DISTANCE_ANGSTROM"]),
            inertia_relative=float(values["THRESHOLD_INERTIA_RELATIVE"]),
            max_rotation_order=int(values["MAX_ROTATION_ORDER"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid or incomplete SYMMETRY thresholds in {Path(path)}") from exc


def read_molecular_symmetry(path: Path) -> MolecularSymmetry:
    """Read the frozen ORACLE point-group operations from enriched XYZ."""
    content = section_content(read_sectioned_lines(Path(path)), "SYMMETRY")
    if not content:
        raise ValueError(f"missing SYMMETRY section in {Path(path)}")
    values: dict[str, str] = {}
    operations: list[SymmetryOperation] = []
    atom_classes: list[tuple[int, ...]] = []
    subsection = ""
    for line in content:
        text = line.strip()
        if text.startswith("[") and text.endswith("]"):
            subsection = text[1:-1].strip().upper()
            continue
        if not text or text.upper() == "NONE":
            continue
        if subsection == "OPERATIONS":
            fields = _inline_key_values(text.split()[1:])
            try:
                matrix_values = tuple(float(value) for value in fields["MATRIX"].split(","))
                permutation = tuple(int(value) for value in fields["PERMUTATION"].split(","))
                if len(matrix_values) != 9:
                    raise ValueError("operation matrix must contain 9 values")
                operations.append(
                    SymmetryOperation(
                        label=fields["LABEL"],
                        rotation=tuple(
                            tuple(matrix_values[start : start + 3]) for start in (0, 3, 6)
                        ),
                        permutation=permutation,
                        max_deviation=float(fields.get("MAX_DEVIATION", "0")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid SYMMETRY operation in {Path(path)}: {text}") from exc
            continue
        if subsection == "ATOM_CLASSES":
            fields = _inline_key_values(text.split()[1:])
            try:
                atom_classes.append(tuple(int(value) for value in fields["ATOMS"].split(",")))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid SYMMETRY atom class in {Path(path)}: {text}") from exc
            continue
        fields = text.split(maxsplit=1)
        if len(fields) == 2:
            values[fields[0]] = fields[1]
    try:
        return MolecularSymmetry(
            point_group=values["POINT_GROUP"],
            operations=tuple(operations),
            atom_classes=tuple(atom_classes),
            max_deviation=float(values.get("MAX_OPERATION_DEVIATION_ANGSTROM", "0")),
            mean_deviation=float(values.get("MEAN_OPERATION_DEVIATION_ANGSTROM", "0")),
            supported_subgroups=tuple(
                item.strip()
                for item in values.get("SUPPORTED_SUBGROUPS", "").split(",")
                if item.strip()
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid or incomplete SYMMETRY section in {Path(path)}") from exc


def _inline_key_values(tokens: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            output[key] = value
    return output


def write_symmetry_section(
    path: Path,
    *,
    symmetry,
    thresholds: SymmetryThresholds,
    projection=None,
    cartesian_symmetry_status: str | None = None,
    symmetrization_required_threshold_angstrom: float | None = None,
    cartesian_symmetrization_decision: str | None = None,
    input_symmetry_max_deviation_angstrom: float | None = None,
    proposed_point_group: str | None = None,
) -> None:
    replace_section(
        Path(path),
        "SYMMETRY",
        symmetry_section_lines(
            symmetry,
            thresholds=thresholds,
            projection=projection,
            cartesian_symmetry_status=cartesian_symmetry_status,
            symmetrization_required_threshold_angstrom=(
                symmetrization_required_threshold_angstrom
            ),
            cartesian_symmetrization_decision=cartesian_symmetrization_decision,
            input_symmetry_max_deviation_angstrom=(
                input_symmetry_max_deviation_angstrom
            ),
            proposed_point_group=proposed_point_group,
        ),
    )


def write_topology_and_synthons_sections(
    path: Path,
    geometry: MolecularGeometry,
    *,
    external_charges=None,
    bond_order_overrides=None,
    charge_source="ORACLE electronegativity estimate",
    bond_order_source="ORACLE Pauling estimate",
    electronic_connectivity=False,
) -> tuple[int, int]:
    atomic_numbers = [_atomic_number(atom) for atom in geometry.atoms]
    electronic = electronic_population_overrides_from_xyzin(Path(path))
    selected_charges = external_charges if external_charges is not None else electronic["charges"]
    selected_orders = (
        bond_order_overrides if bond_order_overrides is not None else electronic["bond_orders"]
    )
    continuous, discrete, ringset, synthons, aromaticity = build_topology_objects(
        geometry.coordinates_angstrom,
        atomic_numbers,
        bond_order_overrides=selected_orders,
        external_charges=selected_charges,
        charge_source=charge_source if external_charges is not None else electronic["charge_source"],
        bond_order_source=(
            bond_order_source if bond_order_overrides is not None else electronic["bond_order_source"]
        ),
        electronic_connectivity=electronic_connectivity,
    )
    topology_lines = [
        f"SCHEMA {MATRIX_XYZ_TOPOLOGY_SCHEMA}",
        "INDEXING ATOMS=ONE_BASED",
        f"BOND_ORDER_SOURCE {synthons._bond_order_source}",
        "RING_BASIS_POLICY CHORDLESS_NONMETAL_MINIMUM_CYCLE_BASIS",
        *_ring_basis_diagnostic_lines(ringset),
        "[BONDS]",
    ]
    if discrete.bonds:
        topology_lines.extend(f"{i + 1} {j + 1}" for i, j in discrete.bonds)
    else:
        topology_lines.append("NONE")
    topology_lines.append("[TRANSITIONAL_CONTACTS]")
    transitional_contacts = tuple(getattr(discrete, "transitional_contacts", ()))
    if transitional_contacts:
        topology_lines.extend(
            f"{i + 1} {j + 1} KIND=WEAK_ACYCLIC"
            for i, j in transitional_contacts
        )
    else:
        topology_lines.append("NONE")
    topology_lines.append("[BOND_ORDERS]")
    bond_order_rows = []
    for i, j in discrete.bonds:
        try:
            value = float(synthons.bond_order(i, j))
        except Exception:
            continue
        bond_order_rows.append(f"{i + 1} {j + 1} {value:.10g}")
    if bond_order_rows:
        topology_lines.extend(bond_order_rows)
    else:
        topology_lines.append("NONE")
    topology_lines.extend(
        [
            "[BOND_ORDER_COMPONENTS]",
            "COLUMNS ATOM_I ATOM_J BO BO_SIGMA BO_PI BO_PI_PI",
        ]
    )
    component_rows = []
    for i, j in discrete.bonds:
        components = synthons.bond_order_components(i, j)
        component_rows.append(
            f"{i + 1} {j + 1} {components.total:.10g} "
            f"{components.sigma:.10g} {components.pi:.10g} {components.pi_pi:.10g}"
        )
    topology_lines.extend(component_rows or ["NONE"])
    topology_lines.append("[RINGS]")
    if ringset.rings:
        for idx, ring in enumerate(ringset.rings, start=1):
            atoms = " ".join(str(atom + 1) for atom in ring.atoms)
            topology_lines.append(f"{idx} SIZE={len(ring)} ATOMS={atoms}")
    else:
        topology_lines.append("NONE")
    automorphism_orbits = build_topology_automorphism_orbits(
        discrete,
        ringset,
        synthons,
        atomic_numbers,
        aromaticity=aromaticity,
    )
    topology_lines.extend(topology_automorphism_lines(automorphism_orbits))
    topology_lines.append("[AROMATICITY]")
    aromatic_atoms = sorted(getattr(aromaticity, "aromatic_atoms", set()))
    topology_lines.append(
        "ATOMS "
        + (" ".join(str(atom + 1) for atom in aromatic_atoms) if aromatic_atoms else "NONE")
    )
    replace_section(Path(path), "TOPOLOGY", topology_lines)
    replace_section(Path(path), "AROMATICITY", aromaticity_section_lines(aromaticity))

    synthon_lines = [
        f"SCHEMA {MATRIX_XYZ_SYNTHONS_SCHEMA}",
        "INDEXING ATOMS=ONE_BASED",
        f"CHARGE_SOURCE {synthons._charge_source}",
        f"BOND_ORDER_SOURCE {synthons._bond_order_source}",
        "COLUMNS ATOM Z ZEFF CHARGE COVALENCY DELOCALIZATION RADIAL_DISPERSION STRAIN "
        "SIGMA_INDEX PI_INDEX PI_PI_INDEX SIGNATURE",
    ]
    for idx, atom in enumerate(geometry.atoms):
        signature = synthons.canonical_signature(idx)
        signature_text = ",".join(str(item) for item in signature)
        synthon_lines.append(
            f"{idx + 1} {atom} "
            f"{float(synthons.Zeff(idx)):.8g} "
            f"{float(synthons.charge(idx)):.8g} "
            f"{float(synthons.covalency(idx)):.8g} "
            f"{float(synthons.delocalization(idx)):.8g} "
            f"{float(synthons.radial_dispersion(idx)):.8g} "
            f"{float(synthons.strain(idx)):.8g} "
            f"{float(synthons.sigma_index(idx)):.8g} "
            f"{float(synthons.pi_index(idx)):.8g} "
            f"{float(synthons.pi_pi_index(idx)):.8g} "
            f"{signature_text}"
        )
    replace_section(Path(path), "SYNTHONS", synthon_lines)
    return len(discrete.bonds), len(ringset.rings)


def _ring_basis_diagnostic_lines(ringset) -> list[str]:
    diagnostics = getattr(ringset, "cycle_basis_diagnostics", None)
    if diagnostics is None:
        return []
    excluded = (
        ",".join(str(atom + 1) for atom in diagnostics.excluded_atoms)
        if diagnostics.excluded_atoms
        else "NONE"
    )
    return [
        f"RING_BASIS_ALGORITHM {diagnostics.algorithm}",
        f"RING_BASIS_COMPLETE {'YES' if diagnostics.complete else 'NO'}",
        f"RING_CANDIDATE_COUNT {diagnostics.candidate_cycle_count}",
        f"RING_BASIS_RANK {diagnostics.cycle_rank}",
        f"RING_BASIS_COUNT {diagnostics.selected_cycle_count}",
        f"RING_BASIS_MAXIMUM_SIZE {diagnostics.maximum_selected_size}",
        f"RING_BASIS_ALLOWED_ATOMS {diagnostics.allowed_atom_count}",
        f"RING_BASIS_ALLOWED_EDGES {diagnostics.allowed_edge_count}",
        f"RING_BASIS_EXCLUDED_ATOMS {excluded}",
    ]


def gaussian_topology_overrides_from_xyzin(path: Path) -> dict[str, object]:
    """Read #GAUSSIAN_TOPOLOGY as ORACLE topology overrides."""
    content = section_content(read_sectioned_lines(Path(path)), "GAUSSIAN_TOPOLOGY")
    charges: dict[int, float] = {}
    bond_orders: dict[tuple[int, int], float] = {}
    bo_source: str | None = None
    for raw in content:
        text = raw.strip()
        if not text or text.upper().startswith(("SCHEMA ", "INDEXING ", "CM5_COUNT", "BO_COUNT")):
            continue
        parts = text.replace("=", " = ").split()
        key = parts[0].upper() if parts else ""
        if key == "CM5" and len(parts) >= 3:
            idx = int(parts[1]) - 1
            if idx >= 0:
                charges[idx] = float(parts[2])
            continue
        if key == "BO_SOURCE" and len(parts) >= 2:
            bo_source = parts[2] if len(parts) >= 3 and parts[1] == "=" else parts[1]
            continue
        if key == "BO" and len(parts) >= 4:
            i = int(parts[1]) - 1
            j = int(parts[2]) - 1
            if i >= 0 and j >= 0 and i != j:
                pair = (i, j) if i < j else (j, i)
                bond_orders[pair] = float(parts[3])
    return {
        "charges": charges,
        "bond_orders": bond_orders,
        "charge_source": "QM CM5" if charges else "ORACLE electronegativity estimate",
        "bond_order_source": (
            f"QM {bo_source}"
            if bond_orders and bo_source
            else "ORACLE Pauling estimate"
        ),
    }


def electronic_population_overrides_from_xyzin(path: Path) -> dict[str, object]:
    """Return APOC CM5/Mayer data, accepting Gaussian v1 only for migration."""

    try:
        from matrix_qm import read_qm_population_section

        population, source = read_qm_population_section(path)
    except (KeyError, ValueError):
        return gaussian_topology_overrides_from_xyzin(path)
    return {
        "charges": {
            index: float(value)
            for index, value in enumerate(population.cm5_charges)
        },
        "bond_orders": {
            (left, right): float(population.mayer_bond_orders[left, right])
            for left in range(population.natoms)
            for right in range(left + 1, population.natoms)
        },
        "charge_source": f"APOC CM5 {source}",
        "bond_order_source": f"APOC Mayer {source}",
    }


def _atomic_number(symbol: str) -> int:
    from .topology.elements import atomic_number

    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise ValueError(f"unknown element symbol: {symbol}")
    return int(number)
