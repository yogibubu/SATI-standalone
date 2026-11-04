from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_chem import (
    DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER,
    hbond_angular_factor,
    perceive_aromatic_ring_atoms,
    prepare_hydrogen_bond_recognition,
    read_enriched_xyz,
)
from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.contracts import (
    MATRIX_XYZ_FRAGMENTS_SCHEMA,
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    SUPPORTED_FRAGMENTS_SCHEMAS,
    SUPPORTED_SYNTHONS_SCHEMAS,
    SUPPORTED_TOPOLOGY_SCHEMAS,
    schema_line_supported,
    supported_schema_text,
)
from matrix_core import read_sectioned_lines, replace_section, section_content


ORACLE_XYZ_FRAGMENT_LIBRARY_SCHEMA = "oracle.xyz.fragment_library.v1"
ORACLE_XYZ_ASSEMBLY_SCHEMA = "oracle.xyz.assembly.v1"
ORACLE_XYZ_INTERACTION_CENTERS_SCHEMA = "oracle.xyz.interaction_centers.v2"
SUPPORTED_INTERACTION_CENTER_SCHEMAS = (
    "oracle.xyz.interaction_centers.v1",
    ORACLE_XYZ_INTERACTION_CENTERS_SCHEMA,
)

GEOMETRIC_PARAMETER_BINDING_SCHEMA = "matrix.geometric_parameter_source_binding.v1"
GEOMETRIC_PARAMETER_SOURCE_BINDINGS = {
    "L1_geometry_only": {
        "level_family": "L1",
        "charges": "electronegativity_estimated",
        "bond_orders": "Pauling_estimated",
        "cm5_mayer_allowed": False,
        "required_electronic_source": "ORACLE_geometry_topology",
    },
    "L2_PL2_geometry_only": {
        "level_family": "L2_PL2",
        "charges": "CM5",
        "bond_orders": "Mayer",
        "cm5_mayer_allowed": True,
        "required_electronic_source": "LCB26_enriched_record",
    },
}


def geometric_parameter_source_binding(parameter_source: str) -> dict[str, object]:
    """Return the immutable electronic-observable binding for a geometry fit."""
    try:
        binding = GEOMETRIC_PARAMETER_SOURCE_BINDINGS[str(parameter_source)]
    except KeyError as exc:
        raise ValueError(f"unsupported geometric parameter source: {parameter_source}") from exc
    return {
        "schema": GEOMETRIC_PARAMETER_BINDING_SCHEMA,
        "parameter_source": str(parameter_source),
        **binding,
    }

REQUIRED_TOPOLOGY_SCHEMA = MATRIX_XYZ_TOPOLOGY_SCHEMA
REQUIRED_SYNTHONS_SCHEMA = MATRIX_XYZ_SYNTHONS_SCHEMA
RANK_TOLERANCE = 1.0e-8
METAL_SYMBOLS = frozenset(
    {
        "Li",
        "Na",
        "K",
        "Rb",
        "Cs",
        "Fr",
        "Be",
        "Mg",
        "Ca",
        "Sr",
        "Ba",
        "Ra",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Al",
        "Ga",
        "In",
        "Tl",
        "Sn",
        "Pb",
        "Bi",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    }
)


class FragmentContractError(ValueError):
    """Raised when a file cannot enter the MATRIX fragment workflow."""


@dataclass(frozen=True)
class FragmentRecord:
    identifier: str
    label: str
    atoms: tuple[int, ...]
    center: tuple[float, float, float]
    frame: tuple[tuple[float, float, float], ...]
    charge: int | None = None
    multiplicity: int | None = None


@dataclass(frozen=True)
class AttachmentSiteRecord:
    """Reusable covalent attachment metadata owned by one rigid fragment."""

    identifier: str
    fragment_id: str
    connecting_atom: int
    direction: tuple[float, float, float]
    label: str
    leaving_atoms: tuple[int, ...] = ()
    allowed_elements: tuple[str, ...] = ()
    allowed_bond_orders: tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class FragmentDefinition:
    strategy: str
    reference_fragment: str
    fragments: tuple[FragmentRecord, ...]
    attachment_sites: tuple[AttachmentSiteRecord, ...] = ()


@dataclass(frozen=True)
class InteractionCenterRecord:
    identifier: str
    kind: str
    label: str
    atoms: tuple[int, ...]
    center: tuple[float, float, float]
    source: str
    hapticity: int = 0


@dataclass(frozen=True)
class AtomCenterInteractionRecord:
    identifier: str
    kind: str
    atom: int
    center_id: str
    score: float
    source: str
    effective_order: float = DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER


@dataclass(frozen=True)
class HapticInteractionRequest:
    """ORACLE-declared metal--donor-set center with explicit hapticity."""

    metal_atom: int
    donor_atoms: tuple[int, ...]
    effective_order: float = DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER
    source: str = "ORACLE_COORDINATION_INPUT"

    def __post_init__(self) -> None:
        donors = tuple(int(atom) for atom in self.donor_atoms)
        if len(donors) < 2 or len(set(donors)) != len(donors):
            raise ValueError("haptic donor atoms must be unique and hapticity must be at least two")
        order = float(self.effective_order)
        if not np.isfinite(order) or order <= 0.0:
            raise ValueError("haptic effective order must be finite and positive")
        object.__setattr__(self, "metal_atom", int(self.metal_atom))
        object.__setattr__(self, "donor_atoms", donors)
        object.__setattr__(self, "effective_order", order)


@dataclass(frozen=True)
class InteractionCenterDefinition:
    strategy: str
    centers: tuple[InteractionCenterRecord, ...]
    interactions: tuple[AtomCenterInteractionRecord, ...]


def validate_fragment_prerequisites(path: Path) -> None:
    """Require saved topology and synthons before any fragment workflow starts."""
    lines = read_sectioned_lines(Path(path))
    _require_schema(lines, "TOPOLOGY", SUPPORTED_TOPOLOGY_SCHEMAS)
    _require_schema(lines, "SYNTHONS", SUPPORTED_SYNTHONS_SCHEMAS)


def fragment_plan_section_lines(
    *,
    status: str = "PLANNED",
    strategy: str = "TOPOLOGY_SYNTHON",
) -> list[str]:
    """Return the initial #FRAGMENTS section without computing fragments yet."""
    return [
        f"SCHEMA {MATRIX_XYZ_FRAGMENTS_SCHEMA}",
        f"STATUS {status.strip().upper()}",
        f"DEPENDENCIES TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} "
        f"SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA}",
        "INDEXING ATOMS=ONE_BASED",
        f"STRATEGY {strategy.strip().upper()}",
        "[FRAGMENTS]",
        "PENDING ROBUST_TOPOLOGY_CONTRACT",
    ]


def write_fragment_plan_section(path: Path) -> None:
    """Mark an enriched XYZ as ready for future topology-backed fragmentation."""
    target = Path(path)
    validate_fragment_prerequisites(target)
    replace_section(target, "FRAGMENTS", fragment_plan_section_lines())


def build_fragment_definition_from_xyzin(path: Path) -> FragmentDefinition:
    """Build concrete fragments from the saved topology connected components."""
    target = Path(path)
    validate_fragment_prerequisites(target)
    lines = read_sectioned_lines(target)
    geometry = read_enriched_xyz(target)
    bonds = _topology_bonds(lines, natoms=geometry.natoms)
    components = _connected_components(bonds, natoms=geometry.natoms)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    fragments = tuple(
        FragmentRecord(
            identifier=f"F{idx:03d}",
            label=f"component_{idx}",
            atoms=tuple(component),
            center=_center(coords, component),
            frame=_frame(coords, component),
        )
        for idx, component in enumerate(components, start=1)
    )
    reference = max(fragments, key=lambda item: (len(item.atoms), -int(item.identifier[1:])))
    return FragmentDefinition(
        strategy="CONNECTED_COMPONENTS",
        reference_fragment=reference.identifier,
        fragments=fragments,
    )


def fragment_build_section_lines(definition: FragmentDefinition) -> list[str]:
    lines = [
        f"SCHEMA {MATRIX_XYZ_FRAGMENTS_SCHEMA}",
        "STATUS BUILT",
        f"DEPENDENCIES TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} "
        f"SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA}",
        "INDEXING ATOMS=ONE_BASED",
        f"STRATEGY {definition.strategy}",
        f"FRAGMENT_COUNT {len(definition.fragments)}",
        f"REFERENCE_FRAGMENT {definition.reference_fragment}",
        "[FRAGMENTS]",
    ]
    for fragment in definition.fragments:
        atoms = ",".join(str(atom) for atom in fragment.atoms)
        lines.append(
            f"{fragment.identifier} LABEL={fragment.label} SIZE={len(fragment.atoms)} ATOMS={atoms} "
            f"CHARGE={_optional_integer_text(fragment.charge)} "
            f"MULTIPLICITY={_optional_integer_text(fragment.multiplicity)}"
        )
    lines.append("[CENTERS]")
    for fragment in definition.fragments:
        x, y, z = fragment.center
        lines.append(f"{fragment.identifier} X={x:.12g} Y={y:.12g} Z={z:.12g}")
    lines.append("[FRAMES]")
    for fragment in definition.fragments:
        axes = []
        for label, axis in zip(("X", "Y", "Z"), fragment.frame):
            axes.append(f"{label}={axis[0]:.12g},{axis[1]:.12g},{axis[2]:.12g}")
        lines.append(f"{fragment.identifier} {' '.join(axes)}")
    lines.append("[ATTACHMENT_SITES]")
    if not definition.attachment_sites:
        lines.append("NONE")
    for site in definition.attachment_sites:
        leaving = ",".join(str(atom) for atom in site.leaving_atoms) or "NONE"
        elements = ",".join(site.allowed_elements) or "ANY"
        orders = ",".join(f"{value:.12g}" for value in site.allowed_bond_orders)
        direction = ",".join(f"{value:.12g}" for value in site.direction)
        lines.append(
            f"{site.identifier} FRAGMENT={site.fragment_id} LABEL={site.label} "
            f"CONNECTING_ATOM={site.connecting_atom} LEAVING_ATOMS={leaving} "
            f"DIRECTION={direction} ALLOWED_ELEMENTS={elements} "
            f"ALLOWED_BOND_ORDERS={orders}"
        )
    return lines


def write_fragment_build_section(path: Path) -> FragmentDefinition:
    """Materialize the #FRAGMENTS section from saved topology."""
    target = Path(path)
    definition = build_fragment_definition_from_xyzin(target)
    replace_section(target, "FRAGMENTS", fragment_build_section_lines(definition))
    return definition


def write_fragment_electronic_states(
    path: Path,
    states: dict[str, tuple[int, int]],
) -> FragmentDefinition:
    """Set explicit charge/multiplicity values in the shared fragment contract."""

    target = Path(path)
    lines = read_sectioned_lines(target)
    section = section_content(lines, "FRAGMENTS")
    current = read_fragment_records(target)
    if not current:
        raise FragmentContractError("fragment electronic states require a built #FRAGMENTS section")
    unknown = set(states) - {fragment.identifier for fragment in current}
    if unknown:
        raise FragmentContractError(f"unknown fragment identifiers: {', '.join(sorted(unknown))}")
    fragments: list[FragmentRecord] = []
    for fragment in current:
        state = states.get(fragment.identifier)
        charge = fragment.charge if state is None else int(state[0])
        multiplicity = fragment.multiplicity if state is None else int(state[1])
        if multiplicity is not None and multiplicity < 1:
            raise FragmentContractError("fragment multiplicity must be positive")
        fragments.append(
            FragmentRecord(
                identifier=fragment.identifier,
                label=fragment.label,
                atoms=fragment.atoms,
                center=fragment.center,
                frame=fragment.frame,
                charge=charge,
                multiplicity=multiplicity,
            )
        )
    reference_fragment = _section_value(section, "REFERENCE_FRAGMENT")
    if reference_fragment not in {fragment.identifier for fragment in current}:
        raise FragmentContractError("#FRAGMENTS has an invalid REFERENCE_FRAGMENT")
    definition = FragmentDefinition(
        strategy=_section_value(section, "STRATEGY") or "CONNECTED_COMPONENTS",
        reference_fragment=reference_fragment,
        fragments=tuple(fragments),
        attachment_sites=read_fragment_attachment_sites(target),
    )
    replace_section(target, "FRAGMENTS", fragment_build_section_lines(definition))
    return definition


def build_interaction_center_definition_from_xyzin(
    path: Path,
    *,
    haptic_interactions: tuple[HapticInteractionRequest, ...] = (),
) -> InteractionCenterDefinition:
    """Build topology-backed bond/ring centers and atom-center candidates."""
    target = Path(path)
    validate_fragment_prerequisites(target)
    lines = read_sectioned_lines(target)
    geometry = read_enriched_xyz(target)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    bonds = _topology_bonds(lines, natoms=geometry.natoms)
    rings = _topology_rings(lines, natoms=geometry.natoms)
    atomic_numbers = tuple(int(atomic_number(atom) or 0) for atom in geometry.atoms)
    aromatic_rings = _aromatic_ring_atom_sets(
        lines,
        rings=rings,
        coordinates_angstrom=coords,
        atomic_numbers=atomic_numbers,
    )
    centers: list[InteractionCenterRecord] = []

    for left, right in bonds:
        centers.append(
            InteractionCenterRecord(
                identifier=f"C{len(centers) + 1:03d}",
                kind="BOND_CENTER",
                label=f"bond_{left}_{right}",
                atoms=(left, right),
                center=_center(coords, (left, right)),
                source="TOPOLOGY_BOND",
                hapticity=2,
            )
        )
    for ring_index, atoms in _interaction_ring_centers(rings, geometry.atoms):
        centers.append(
            InteractionCenterRecord(
                identifier=f"C{len(centers) + 1:03d}",
                kind="RING_CENTER",
                label=f"ring_{ring_index}",
                atoms=atoms,
                center=_center(coords, atoms),
                source=(
                    "TOPOLOGY_AROMATIC_RING"
                    if frozenset(atoms) in aromatic_rings
                    else "TOPOLOGY_RING"
                ),
                hapticity=len(atoms),
            )
        )

    declared_interactions: list[AtomCenterInteractionRecord] = []
    for request in haptic_interactions:
        if request.metal_atom < 1 or request.metal_atom > geometry.natoms:
            raise ValueError("haptic metal atom lies outside the geometry")
        if any(atom < 1 or atom > geometry.natoms for atom in request.donor_atoms):
            raise ValueError("haptic donor atom lies outside the geometry")
        if not _is_metal_symbol(_atom_symbol(geometry.atoms, request.metal_atom)):
            raise ValueError("haptic interaction center atom must be a metal")
        donor_key = frozenset(request.donor_atoms)
        center = next(
            (item for item in centers if frozenset(item.atoms) == donor_key),
            None,
        )
        if center is None:
            center = InteractionCenterRecord(
                identifier=f"C{len(centers) + 1:03d}",
                kind="HAPTIC_CENTER",
                label=f"eta{len(request.donor_atoms)}_" + "_".join(
                    str(atom) for atom in request.donor_atoms
                ),
                atoms=request.donor_atoms,
                center=_center(coords, request.donor_atoms),
                source=request.source,
                hapticity=len(request.donor_atoms),
            )
            centers.append(center)
        declared_interactions.append(
            AtomCenterInteractionRecord(
                identifier="",
                kind=f"METAL_ETA{len(request.donor_atoms)}_CENTER",
                atom=request.metal_atom,
                center_id=center.identifier,
                score=1.0,
                source=request.source,
                effective_order=request.effective_order,
            )
        )

    inferred_interactions = _atom_center_interactions(
        tuple(centers),
        bonds=bonds,
        coords=coords,
        natoms=geometry.natoms,
        atom_symbols=geometry.atoms,
    )
    aromatic_hbonds = _aromatic_hydrogen_center_interactions(
        tuple(centers),
        bonds=bonds,
        coordinates_angstrom=coords,
        atomic_numbers=atomic_numbers,
    )
    interactions = _merge_atom_center_interactions(
        (*declared_interactions, *inferred_interactions, *aromatic_hbonds)
    )
    return InteractionCenterDefinition(
        strategy="TOPOLOGY_HAPTIC_AND_HBOND_CENTER_CANDIDATES",
        centers=tuple(centers),
        interactions=interactions,
    )


def interaction_center_section_lines(definition: InteractionCenterDefinition) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_INTERACTION_CENTERS_SCHEMA}",
        "STATUS BUILT",
        f"DEPENDENCIES TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} "
        f"SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA}",
        "INDEXING ATOMS=ONE_BASED",
        f"STRATEGY {definition.strategy}",
        f"CENTER_COUNT {len(definition.centers)}",
        f"INTERACTION_COUNT {len(definition.interactions)}",
        "[CENTERS]",
    ]
    if definition.centers:
        for center in definition.centers:
            atoms = ",".join(str(atom) for atom in center.atoms)
            x, y, z = center.center
            lines.append(
                f"{center.identifier} KIND={center.kind} LABEL={center.label} "
                f"ATOMS={atoms} X={x:.12g} Y={y:.12g} Z={z:.12g} "
                f"HAPTICITY={center.hapticity} SOURCE={center.source}"
            )
    else:
        lines.append("NONE")
    lines.append("[INTERACTIONS]")
    if definition.interactions:
        for interaction in definition.interactions:
            lines.append(
                f"{interaction.identifier} KIND={interaction.kind} ATOM={interaction.atom} "
                f"CENTER={interaction.center_id} SCORE={interaction.score:.8g} "
                f"ORDER={interaction.effective_order:.8g} SOURCE={interaction.source}"
            )
    else:
        lines.append("NONE")
    return lines


def write_interaction_center_section(
    path: Path,
    *,
    haptic_interactions: tuple[HapticInteractionRequest, ...] = (),
) -> InteractionCenterDefinition:
    """Materialize virtual bond/ring centers and atom-center interaction candidates."""
    target = Path(path)
    definition = build_interaction_center_definition_from_xyzin(
        target,
        haptic_interactions=haptic_interactions,
    )
    replace_section(target, "INTERACTION_CENTERS", interaction_center_section_lines(definition))
    return definition


def read_fragment_records(path: Path) -> tuple[FragmentRecord, ...]:
    """Read built fragment records from an enriched XYZ file."""
    lines = read_sectioned_lines(Path(path))
    section = section_content(lines, "FRAGMENTS")
    if not section:
        return ()
    if not schema_line_supported(section[0], SUPPORTED_FRAGMENTS_SCHEMAS):
        raise FragmentContractError("invalid #FRAGMENTS schema")
    status = _section_value(section, "STATUS")
    if status != "BUILT":
        return ()
    fragment_rows = _subsection(section, "FRAGMENTS")
    centers = _center_rows(_subsection(section, "CENTERS"))
    frames = _frame_rows(_subsection(section, "FRAMES"))
    records: list[FragmentRecord] = []
    identifiers: set[str] = set()
    assigned_atoms: set[int] = set()
    for row in fragment_rows:
        parts = row.split()
        if not parts:
            continue
        identifier = parts[0]
        if identifier in identifiers:
            raise FragmentContractError(f"duplicate fragment identifier: {identifier}")
        identifiers.add(identifier)
        fields = _key_values(parts[1:])
        atoms_text = fields.get("ATOMS", "")
        if not atoms_text:
            raise FragmentContractError(f"fragment {identifier} has no ATOMS field")
        try:
            atoms = tuple(int(item) for item in atoms_text.split(",") if item)
        except ValueError as exc:
            raise FragmentContractError(f"invalid ATOMS field for {identifier}") from exc
        if not atoms or any(atom < 1 for atom in atoms):
            raise FragmentContractError(f"fragment {identifier} has invalid atom indices")
        if len(set(atoms)) != len(atoms):
            raise FragmentContractError(f"fragment {identifier} repeats atom indices")
        overlap = assigned_atoms.intersection(atoms)
        if overlap:
            repeated = ",".join(str(atom) for atom in sorted(overlap))
            raise FragmentContractError(f"atoms assigned to multiple fragments: {repeated}")
        assigned_atoms.update(atoms)
        charge = _optional_integer(fields.get("CHARGE"))
        multiplicity = _optional_integer(fields.get("MULTIPLICITY"))
        if multiplicity is not None and multiplicity < 1:
            raise FragmentContractError(f"fragment {identifier} multiplicity must be positive")
        records.append(
            FragmentRecord(
                identifier=identifier,
                label=fields.get("LABEL", identifier),
                atoms=atoms,
                center=centers.get(identifier, (0.0, 0.0, 0.0)),
                frame=frames.get(identifier, _identity_frame()),
                charge=charge,
                multiplicity=multiplicity,
            )
        )
    return tuple(records)


def read_fragment_attachment_sites(path: Path) -> tuple[AttachmentSiteRecord, ...]:
    """Read optional reusable attachment sites from a built fragment section."""

    lines = read_sectioned_lines(Path(path))
    section = section_content(lines, "FRAGMENTS")
    if not section or _section_value(section, "STATUS") != "BUILT":
        return ()
    fragments = {record.identifier: record for record in read_fragment_records(path)}
    records: list[AttachmentSiteRecord] = []
    identifiers: set[str] = set()
    for row in _subsection(section, "ATTACHMENT_SITES"):
        if not row.strip() or row.strip().upper() == "NONE":
            continue
        parts = row.split()
        identifier = parts[0]
        if identifier in identifiers:
            raise FragmentContractError(f"duplicate attachment-site identifier: {identifier}")
        identifiers.add(identifier)
        fields = _key_values(parts[1:])
        fragment_id = fields.get("FRAGMENT", "")
        if fragment_id not in fragments:
            raise FragmentContractError(
                f"attachment site {identifier} references unknown fragment {fragment_id}"
            )
        try:
            connecting_atom = int(fields["CONNECTING_ATOM"])
            direction = tuple(float(value) for value in fields["DIRECTION"].split(","))
            leaving = _parse_int_list(
                "" if fields.get("LEAVING_ATOMS", "NONE").upper() == "NONE"
                else fields["LEAVING_ATOMS"]
            )
            orders = tuple(
                float(value)
                for value in fields.get("ALLOWED_BOND_ORDERS", "1").split(",")
            )
        except (KeyError, ValueError) as exc:
            raise FragmentContractError(
                f"invalid attachment-site fields for {identifier}"
            ) from exc
        allowed = set(fragments[fragment_id].atoms)
        if connecting_atom not in allowed or any(atom not in allowed for atom in leaving):
            raise FragmentContractError(
                f"attachment site {identifier} contains atoms outside {fragment_id}"
            )
        vector = np.asarray(direction, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise FragmentContractError(
                f"attachment site {identifier} needs a finite 3-vector direction"
            )
        norm = float(np.linalg.norm(vector))
        if norm <= RANK_TOLERANCE:
            raise FragmentContractError(
                f"attachment site {identifier} direction must be non-zero"
            )
        if not orders or any(not np.isfinite(value) or value <= 0.0 for value in orders):
            raise FragmentContractError(
                f"attachment site {identifier} has invalid allowed bond orders"
            )
        elements_text = fields.get("ALLOWED_ELEMENTS", "ANY")
        elements = (
            ()
            if elements_text.upper() == "ANY"
            else tuple(value for value in elements_text.split(",") if value)
        )
        records.append(
            AttachmentSiteRecord(
                identifier=identifier,
                fragment_id=fragment_id,
                connecting_atom=connecting_atom,
                direction=tuple(float(value) for value in vector / norm),
                label=fields.get("LABEL", identifier),
                leaving_atoms=tuple(leaving),
                allowed_elements=elements,
                allowed_bond_orders=orders,
            )
        )
    return tuple(records)


def write_fragment_attachment_sites(
    path: Path,
    attachment_sites: tuple[AttachmentSiteRecord, ...],
) -> FragmentDefinition:
    """Persist validated attachment sites without changing fragment geometry."""

    target = Path(path)
    section = section_content(read_sectioned_lines(target), "FRAGMENTS")
    fragments = read_fragment_records(target)
    if not fragments:
        raise FragmentContractError("attachment sites require a built #FRAGMENTS section")
    definition = FragmentDefinition(
        strategy=_section_value(section, "STRATEGY") or "CONNECTED_COMPONENTS",
        reference_fragment=_section_value(section, "REFERENCE_FRAGMENT"),
        fragments=fragments,
        attachment_sites=tuple(attachment_sites),
    )
    # Round-trip validation is centralized in the reader.
    replace_section(target, "FRAGMENTS", fragment_build_section_lines(definition))
    read_fragment_attachment_sites(target)
    return definition


def _optional_integer_text(value: int | None) -> str:
    return "UNSPECIFIED" if value is None else str(int(value))


def _optional_integer(value: str | None) -> int | None:
    if value is None or value.strip().upper() in {"", "NONE", "UNSPECIFIED", "UNKNOWN"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise FragmentContractError(f"invalid fragment electronic-state integer: {value}") from exc


def read_interaction_center_definition(path: Path) -> InteractionCenterDefinition:
    """Read a built #INTERACTION_CENTERS section, if present."""
    lines = read_sectioned_lines(Path(path))
    section = section_content(lines, "INTERACTION_CENTERS")
    if not section:
        return InteractionCenterDefinition(strategy="NONE", centers=(), interactions=())
    if not schema_line_supported(section[0], SUPPORTED_INTERACTION_CENTER_SCHEMAS):
        raise FragmentContractError("invalid #INTERACTION_CENTERS schema")
    status = _section_value(section, "STATUS")
    if status != "BUILT":
        return InteractionCenterDefinition(strategy="NONE", centers=(), interactions=())
    centers: list[InteractionCenterRecord] = []
    for row in _subsection(section, "CENTERS"):
        if not row.strip() or row.strip().upper() == "NONE":
            continue
        parts = row.split()
        fields = _key_values(parts[1:])
        atoms = _parse_int_list(fields.get("ATOMS", ""))
        try:
            centers.append(
                InteractionCenterRecord(
                    identifier=parts[0],
                    kind=fields.get("KIND", "UNKNOWN"),
                    label=fields.get("LABEL", parts[0]),
                    atoms=atoms,
                    center=(
                        float(fields.get("X", "0.0")),
                        float(fields.get("Y", "0.0")),
                        float(fields.get("Z", "0.0")),
                    ),
                    source=fields.get("SOURCE", "UNKNOWN"),
                    hapticity=int(fields.get("HAPTICITY", len(atoms))),
                )
            )
        except ValueError as exc:
            raise FragmentContractError(f"invalid interaction center row: {row}") from exc
    center_ids = {center.identifier for center in centers}
    interactions: list[AtomCenterInteractionRecord] = []
    for row in _subsection(section, "INTERACTIONS"):
        if not row.strip() or row.strip().upper() == "NONE":
            continue
        parts = row.split()
        fields = _key_values(parts[1:])
        center_id = fields.get("CENTER", "")
        if center_id not in center_ids:
            raise FragmentContractError(f"interaction references unknown center: {row}")
        try:
            interactions.append(
                AtomCenterInteractionRecord(
                    identifier=parts[0],
                    kind=fields.get("KIND", "ATOM_CENTER"),
                    atom=int(fields["ATOM"]),
                    center_id=center_id,
                    score=float(fields.get("SCORE", "1.0")),
                    source=fields.get("SOURCE", "UNKNOWN"),
                    effective_order=float(
                        fields.get("ORDER", DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER)
                    ),
                )
            )
        except (KeyError, ValueError) as exc:
            raise FragmentContractError(f"invalid interaction row: {row}") from exc
    return InteractionCenterDefinition(
        strategy=_section_value(section, "STRATEGY") or "UNKNOWN",
        centers=tuple(centers),
        interactions=tuple(interactions),
    )


def _require_schema(lines: list[str], section_name: str, schemas: tuple[str, ...]) -> None:
    content = section_content(lines, section_name)
    if not content:
        raise FragmentContractError(f"missing #{section_name} section")
    if not schema_line_supported(content[0], schemas):
        raise FragmentContractError(
            f"#{section_name} must start with {supported_schema_text(schemas)!r}; "
            f"found {content[0]!r}"
        )


def _topology_bonds(lines: list[str], *, natoms: int) -> tuple[tuple[int, int], ...]:
    topology = section_content(lines, "TOPOLOGY")
    bond_lines = _subsection(topology, "BONDS")
    bonds: list[tuple[int, int]] = []
    for line in bond_lines:
        if line.strip().upper() == "NONE":
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise FragmentContractError(f"invalid #TOPOLOGY bond line: {line}") from exc
        if i == j or i < 1 or j < 1 or i > natoms or j > natoms:
            raise FragmentContractError(f"invalid #TOPOLOGY bond indexes: {line}")
        bonds.append(tuple(sorted((i, j))))
    return tuple(sorted(set(bonds)))


def _topology_rings(lines: list[str], *, natoms: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    topology = section_content(lines, "TOPOLOGY")
    ring_lines = _subsection(topology, "RINGS")
    rings: list[tuple[int, tuple[int, ...]]] = []
    for line in ring_lines:
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
        if not parts:
            continue
        try:
            ring_index = int(parts[0])
        except ValueError:
            continue
        atoms: list[int] = []
        reading_atoms = False
        for part in parts[1:]:
            token = part.strip()
            if token.upper().startswith("ATOMS="):
                reading_atoms = True
                token = token.split("=", 1)[1]
            elif "=" in token and reading_atoms:
                break
            if not reading_atoms or not token:
                continue
            try:
                atoms.append(int(token))
            except ValueError as exc:
                raise FragmentContractError(f"invalid #TOPOLOGY ring line: {line}") from exc
        if len(atoms) < 3:
            continue
        if any(atom < 1 or atom > natoms for atom in atoms):
            raise FragmentContractError(f"invalid #TOPOLOGY ring atom indexes: {line}")
        rings.append((ring_index, tuple(dict.fromkeys(atoms))))
    return tuple(rings)


def _interaction_ring_centers(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    atom_symbols: tuple[str, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    accepted: list[tuple[int, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for ring_index, atoms in rings:
        if not _valid_interaction_ring_atoms(atoms, atom_symbols):
            continue
        key = tuple(sorted(atoms))
        if key in seen:
            continue
        seen.add(key)
        accepted.append((ring_index, atoms))
    return tuple(accepted)


def _aromatic_ring_atom_sets(
    lines: list[str],
    *,
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    coordinates_angstrom: np.ndarray,
    atomic_numbers: tuple[int, ...],
) -> set[frozenset[int]]:
    """Read ORACLE's persisted aromatic assignment, with geometry fallback."""

    assigned: set[frozenset[int]] = set()
    aromatic_section = section_content(lines, "AROMATICITY")
    for row in aromatic_section:
        parts = row.split()
        if not parts or parts[0].upper() != "RING":
            continue
        fields = _key_values(parts[2:])
        atoms = _parse_int_list(fields.get("ATOMS", ""))
        if atoms:
            assigned.add(frozenset(atoms))
    if assigned:
        return assigned
    topology = section_content(lines, "TOPOLOGY")
    aromatic_atoms: set[int] = set()
    for row in _subsection(topology, "AROMATICITY"):
        parts = row.split()
        if parts and parts[0].upper() == "ATOMS" and parts[1:]:
            aromatic_atoms.update(
                int(value) for value in parts[1:] if value.upper() != "NONE"
            )
    if aromatic_atoms:
        return {
            frozenset(atoms)
            for _index, atoms in rings
            if set(atoms).issubset(aromatic_atoms)
        }
    try:
        perceived = perceive_aromatic_ring_atoms(coordinates_angstrom, atomic_numbers)
    except ValueError:
        perceived = ()
    return {
        frozenset(int(atom) + 1 for atom in ring)
        for ring in perceived
    }


def _valid_interaction_ring_atoms(
    atoms: tuple[int, ...],
    atom_symbols: tuple[str, ...],
) -> bool:
    if len(atoms) < 3:
        return False
    symbols = tuple(_atom_symbol(atom_symbols, atom) for atom in atoms)
    if any(symbol == "H" for symbol in symbols):
        return False
    return not any(_is_metal_symbol(symbol) for symbol in symbols)


def _atom_center_interactions(
    centers: tuple[InteractionCenterRecord, ...],
    *,
    bonds: tuple[tuple[int, int], ...],
    coords: np.ndarray,
    natoms: int,
    atom_symbols: tuple[str, ...],
) -> tuple[AtomCenterInteractionRecord, ...]:
    bonded = {tuple(sorted(bond)) for bond in bonds}
    interactions: list[AtomCenterInteractionRecord] = []
    for center in centers:
        center_atoms = set(center.atoms)
        for atom in range(1, natoms + 1):
            if atom in center_atoms:
                continue
            if not _allowed_atom_center_candidate(atom, center, atom_symbols):
                continue
            bonded_to_center = any(
                tuple(sorted((atom, member))) in bonded for member in center_atoms
            )
            if bonded_to_center and not _allow_bonded_atom_center_interaction(
                atom,
                center,
                atom_symbols,
            ):
                continue
            score = _atom_center_score(atom, center, coords)
            if score <= 0.0:
                continue
            interactions.append(
                AtomCenterInteractionRecord(
                    identifier=f"I{len(interactions) + 1:03d}",
                    kind=f"ATOM_{center.kind}",
                    atom=atom,
                    center_id=center.identifier,
                    score=score,
                    source="AUTO_EQUIDISTANT_GEOMETRY",
                )
            )
    return _suppress_ring_redundant_bond_center_interactions(
        tuple(interactions),
        centers=centers,
        atom_symbols=atom_symbols,
    )


def _aromatic_hydrogen_center_interactions(
    centers: tuple[InteractionCenterRecord, ...],
    *,
    bonds: tuple[tuple[int, int], ...],
    coordinates_angstrom: np.ndarray,
    atomic_numbers: tuple[int, ...],
) -> tuple[AtomCenterInteractionRecord, ...]:
    """Return intermolecular donor--H...aromatic-centroid contacts.

    Donor chemistry comes from the shared MATRIX H-bond recognition plan.
    Intramolecular contacts remain excluded from SONIC because they introduce
    additional cycles; the separately recognized symmetric proton-transfer
    motif retains its existing explicit exception.
    """

    aromatic = tuple(
        center
        for center in centers
        if center.kind == "RING_CENTER" and center.source == "TOPOLOGY_AROMATIC_RING"
    )
    if not aromatic:
        return ()
    zero_based_bonds = tuple((left - 1, right - 1) for left, right in bonds)
    plan = prepare_hydrogen_bond_recognition(atomic_numbers, zero_based_bonds)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    results: list[AtomCenterInteractionRecord] = []
    for donor, hydrogen in plan.donor_hydrogens:
        donor_vector = xyz[donor] - xyz[hydrogen]
        donor_norm = float(np.linalg.norm(donor_vector))
        if donor_norm <= 1.0e-12:
            continue
        for center in aromatic:
            ring_indices = tuple(atom - 1 for atom in center.atoms)
            if any(plan.component_by_atom[hydrogen] == plan.component_by_atom[atom] for atom in ring_indices):
                continue
            center_vector = np.asarray(center.center, dtype=float) - xyz[hydrogen]
            distance = float(np.linalg.norm(center_vector))
            if distance <= 1.0e-12 or distance > plan.cutoff_angstrom:
                continue
            angle = float(
                np.arccos(
                    np.clip(
                        np.dot(donor_vector, center_vector) / (donor_norm * distance),
                        -1.0,
                        1.0,
                    )
                )
            )
            if angle < plan.minimum_angle_radians:
                continue
            angular = float(hbond_angular_factor(angle))
            if angular < plan.selector_threshold:
                continue
            radial = max(0.0, 1.0 - distance / plan.cutoff_angstrom)
            score = radial * angular
            if score <= 0.0:
                continue
            results.append(
                AtomCenterInteractionRecord(
                    identifier="",
                    kind="DONOR_H_AROMATIC_RING_CENTER",
                    atom=hydrogen + 1,
                    center_id=center.identifier,
                    score=score,
                    source="SHARED_HBOND_DONOR_TO_AROMATIC_CENTER",
                    effective_order=DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER,
                )
            )
    return tuple(results)


def _merge_atom_center_interactions(
    interactions: tuple[AtomCenterInteractionRecord, ...],
) -> tuple[AtomCenterInteractionRecord, ...]:
    """Deduplicate declared/inferred contacts, preferring declared provenance."""

    selected: dict[tuple[int, str], AtomCenterInteractionRecord] = {}
    for interaction in interactions:
        key = (int(interaction.atom), str(interaction.center_id))
        if key not in selected:
            selected[key] = interaction
    return tuple(
        AtomCenterInteractionRecord(
            identifier=f"I{index:03d}",
            kind=interaction.kind,
            atom=interaction.atom,
            center_id=interaction.center_id,
            score=interaction.score,
            source=interaction.source,
            effective_order=interaction.effective_order,
        )
        for index, interaction in enumerate(selected.values(), start=1)
    )


def _allowed_atom_center_candidate(
    atom: int,
    center: InteractionCenterRecord,
    atom_symbols: tuple[str, ...],
) -> bool:
    center_symbols = tuple(_atom_symbol(atom_symbols, member) for member in center.atoms)
    if any(_is_metal_symbol(symbol) for symbol in center_symbols):
        return False
    atom_symbol = _atom_symbol(atom_symbols, atom)
    if _is_metal_symbol(atom_symbol):
        return center.kind in {"BOND_CENTER", "RING_CENTER", "HAPTIC_CENTER"}
    return False


def _allow_bonded_atom_center_interaction(
    atom: int,
    center: InteractionCenterRecord,
    atom_symbols: tuple[str, ...],
) -> bool:
    if center.kind not in {"BOND_CENTER", "RING_CENTER", "HAPTIC_CENTER"}:
        return False
    if not _is_metal_symbol(_atom_symbol(atom_symbols, atom)):
        return False
    center_symbols = tuple(_atom_symbol(atom_symbols, member) for member in center.atoms)
    return not any(_is_metal_symbol(symbol) for symbol in center_symbols)


def _suppress_ring_redundant_bond_center_interactions(
    interactions: tuple[AtomCenterInteractionRecord, ...],
    *,
    centers: tuple[InteractionCenterRecord, ...],
    atom_symbols: tuple[str, ...],
) -> tuple[AtomCenterInteractionRecord, ...]:
    center_by_id = {center.identifier: center for center in centers}
    ring_atoms_by_metal: dict[int, list[frozenset[int]]] = {}
    for interaction in interactions:
        center = center_by_id.get(interaction.center_id)
        if center is None or center.kind not in {"RING_CENTER", "HAPTIC_CENTER"}:
            continue
        if not _is_metal_symbol(_atom_symbol(atom_symbols, interaction.atom)):
            continue
        ring_atoms_by_metal.setdefault(interaction.atom, []).append(frozenset(center.atoms))
    if not ring_atoms_by_metal:
        return interactions

    filtered: list[AtomCenterInteractionRecord] = []
    for interaction in interactions:
        center = center_by_id.get(interaction.center_id)
        if center is None:
            filtered.append(interaction)
            continue
        if center.kind == "BOND_CENTER" and _is_metal_symbol(
            _atom_symbol(atom_symbols, interaction.atom)
        ):
            bond_atoms = frozenset(center.atoms)
            if any(
                bond_atoms.issubset(ring_atoms)
                for ring_atoms in ring_atoms_by_metal.get(interaction.atom, ())
            ):
                continue
        filtered.append(interaction)
    return tuple(
        AtomCenterInteractionRecord(
            identifier=f"I{index:03d}",
            kind=interaction.kind,
            atom=interaction.atom,
            center_id=interaction.center_id,
            score=interaction.score,
            source=interaction.source,
            effective_order=interaction.effective_order,
        )
        for index, interaction in enumerate(filtered, start=1)
    )


def _atom_center_score(atom: int, center: InteractionCenterRecord, coords: np.ndarray) -> float:
    atom_coord = coords[atom - 1]
    member_coords = coords[[member - 1 for member in center.atoms]]
    distances = np.linalg.norm(member_coords - atom_coord, axis=1)
    mean_distance = float(np.mean(distances))
    if mean_distance <= RANK_TOLERANCE:
        return 0.0
    spread = float((np.max(distances) - np.min(distances)) / mean_distance)
    center_distance = float(np.linalg.norm(atom_coord - np.asarray(center.center)))
    if center.kind in {"RING_CENTER", "HAPTIC_CENTER"}:
        if spread > 0.12 or center_distance > 4.0:
            return 0.0
        return max(0.0, 1.0 - spread / 0.12) * max(0.0, 1.0 - center_distance / 4.0)
    if center.kind == "BOND_CENTER":
        if spread > 0.08 or center_distance > 3.5:
            return 0.0
        return max(0.0, 1.0 - spread / 0.08) * max(0.0, 1.0 - center_distance / 3.5)
    return 0.0


def _atom_symbol(atom_symbols: tuple[str, ...], atom: int) -> str:
    if atom < 1 or atom > len(atom_symbols):
        raise FragmentContractError(f"invalid atom index {atom}")
    return str(atom_symbols[atom - 1])


def _is_metal_symbol(symbol: str) -> bool:
    return symbol in METAL_SYMBOLS


def _connected_components(
    bonds: tuple[tuple[int, int], ...],
    *,
    natoms: int,
) -> tuple[tuple[int, ...], ...]:
    adjacency = {idx: set() for idx in range(1, natoms + 1)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for atom in range(1, natoms + 1):
        if atom in seen:
            continue
        stack = [atom]
        seen.add(atom)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: item[0]))


def _center(coords: np.ndarray, atoms: tuple[int, ...]) -> tuple[float, float, float]:
    center = np.mean(coords[[atom - 1 for atom in atoms]], axis=0)
    return tuple(float(value) for value in center)


def _frame(
    coords: np.ndarray,
    atoms: tuple[int, ...],
) -> tuple[tuple[float, float, float], ...]:
    if _frame_rank(coords, atoms) < 2:
        return _identity_frame()
    p_atom, q_atom = _frame_anchor_atoms(coords, atoms)
    center = np.asarray(_center(coords, atoms), dtype=float)
    p_axis = _unit(coords[p_atom - 1] - center)
    q_axis = _unit(np.cross(p_axis, coords[q_atom - 1] - center))
    s_axis = _unit(np.cross(p_axis, q_axis))
    frame = np.column_stack([p_axis, q_axis, s_axis])
    return tuple(tuple(float(value) for value in frame[:, axis]) for axis in range(3))


def fragment_local_frame(
    coordinates_angstrom: np.ndarray,
    atoms: tuple[int, ...],
) -> tuple[tuple[float, float, float], ...]:
    """Return the canonical topology-site frame for one one-based atom set.

    This is the public, shared entry point for ORACLE contract builders.  It
    deliberately delegates to the same frame kernel used by interaction
    centers so site serialization and pose-candidate construction cannot
    silently choose different anchors.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or np.any(~np.isfinite(coords)):
        raise ValueError("fragment coordinates must be finite (natoms, 3)")
    members = tuple(int(atom) for atom in atoms)
    if not members or len(set(members)) != len(members):
        raise ValueError("fragment frame requires unique one-based atoms")
    if min(members) < 1 or max(members) > len(coords):
        raise ValueError("fragment frame atom lies outside the geometry")
    return _frame(coords, members)


def fragment_frame_anchor_atoms(
    coordinates_angstrom: np.ndarray,
    atoms: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the canonical one- or two-atom anchors used by local frames."""

    coords = np.asarray(coordinates_angstrom, dtype=float)
    members = tuple(int(atom) for atom in atoms)
    if not members or len(set(members)) != len(members):
        raise ValueError("fragment anchors require unique one-based atoms")
    if min(members) < 1 or max(members) > len(coords):
        raise ValueError("fragment anchor lies outside the geometry")
    if len(members) == 1:
        return members
    return _frame_anchor_atoms(coords, members)


def _frame_rank(coords: np.ndarray, atoms: tuple[int, ...]) -> int:
    if len(atoms) < 2:
        return 0
    centered = coords[[atom - 1 for atom in atoms]] - np.asarray(_center(coords, atoms))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return int(np.sum(singular_values > RANK_TOLERANCE))


def _frame_anchor_atoms(
    coords: np.ndarray,
    atoms: tuple[int, ...],
) -> tuple[int, int]:
    center = np.asarray(_center(coords, atoms), dtype=float)
    ranked = sorted(
        atoms,
        key=lambda atom: (-float(np.linalg.norm(coords[atom - 1] - center)), atom),
    )
    p_atom = ranked[0]
    p_axis = _unit(coords[p_atom - 1] - center)
    q_candidates = []
    for atom in atoms:
        if atom == p_atom:
            continue
        vector = coords[atom - 1] - center
        norm = float(np.linalg.norm(vector))
        if norm <= RANK_TOLERANCE:
            continue
        dot = abs(float(np.dot(p_axis, vector / norm)))
        q_candidates.append((dot, -norm, atom))
    if not q_candidates:
        return ranked[0], ranked[1]
    _dot, _norm, q_atom = min(q_candidates)
    return p_atom, q_atom


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= RANK_TOLERANCE:
        raise FragmentContractError("cannot normalize zero-length fragment frame vector")
    return vector / norm


def _identity_frame() -> tuple[tuple[float, float, float], ...]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _subsection(section_lines: list[str], name: str) -> list[str]:
    header = f"[{name.upper()}]"
    start = None
    for idx, line in enumerate(section_lines):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(section_lines)
    for idx in range(start, len(section_lines)):
        text = section_lines[idx].strip()
        if text.startswith("[") and text.endswith("]"):
            end = idx
            break
    return list(section_lines[start:end])


def _section_value(section_lines: list[str], key: str) -> str | None:
    key_upper = key.upper()
    for line in section_lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == key_upper:
            return parts[1].upper()
    return None


def _key_values(parts: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.upper()] = value
    return fields


def _parse_int_list(text: str) -> tuple[int, ...]:
    if not text:
        return ()
    try:
        return tuple(int(item) for item in text.replace(";", ",").split(",") if item)
    except ValueError as exc:
        raise FragmentContractError(f"invalid integer list: {text}") from exc


def _center_rows(rows: list[str]) -> dict[str, tuple[float, float, float]]:
    centers: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        parts = row.split()
        if not parts:
            continue
        fields = _key_values(parts[1:])
        try:
            centers[parts[0]] = (
                float(fields.get("X", "0.0")),
                float(fields.get("Y", "0.0")),
                float(fields.get("Z", "0.0")),
            )
        except ValueError as exc:
            raise FragmentContractError(f"invalid center row: {row}") from exc
    return centers


def _frame_rows(rows: list[str]) -> dict[str, tuple[tuple[float, float, float], ...]]:
    frames: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for row in rows:
        parts = row.split()
        if not parts:
            continue
        fields = _key_values(parts[1:])
        axes = []
        for label in ("X", "Y", "Z"):
            text = fields.get(label, "")
            values = text.split(",")
            if len(values) != 3:
                axes = []
                break
            axes.append(tuple(float(value) for value in values))
        if axes:
            frames[parts[0]] = tuple(axes)  # type: ignore[assignment]
    return frames
