"""Coordinate-generator registry for the gradual SMITH refactor.

The registry documents the single logical owner of each coordinate-generation
responsibility. Production generation flows through ``definition.py`` and the
validated Fortran/Python backends.
"""

from __future__ import annotations

from dataclasses import dataclass


STAGE_TOPOLOGY = "TOPOLOGY"
STAGE_GEOMETRY = "GEOMETRY"
STAGE_COORDINATE_GENERATION = "COORDINATE_GENERATION"
STAGE_COORDINATE_CLASSIFICATION = "COORDINATE_CLASSIFICATION"
STAGE_REDUNDANCY_ELIMINATION = "REDUNDANCY_ELIMINATION"
STAGE_SYMMETRIZATION = "SYMMETRIZATION"

STATUS_LEGACY_PORTED = "LEGACY_PORTED"
STATUS_REFACTOR_BOUNDARY = "REFACTOR_BOUNDARY"
STATUS_PLANNED = "PLANNED"


@dataclass(frozen=True)
class CoordinateGeneratorSpec:
    """Declarative ownership record for one coordinate-generation component."""

    name: str
    stage: str
    coordinate_family: str
    produces: tuple[str, ...]
    consumes: tuple[str, ...]
    implemented_by: str
    status: str
    notes: str = ""


@dataclass(frozen=True)
class GeneratedCoordinate:
    """Generator-neutral coordinate definition produced before SMITH reduction."""

    family: str
    function: str
    atoms: tuple[int, ...]
    mode: int = 0
    refs: tuple[str, ...] = ()


DEFAULT_COORDINATE_GENERATOR_REGISTRY: tuple[CoordinateGeneratorSpec, ...] = (
    CoordinateGeneratorSpec(
        name="StretchGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="STRETCH",
        produces=("R(i,j)",),
        consumes=("TOPOLOGY.BONDS",),
        implemented_by="matrix_smith.generators.generate_stretch_coordinates",
        status=STATUS_LEGACY_PORTED,
        notes="Includes the default bonded-stretch path.",
    ),
    CoordinateGeneratorSpec(
        name="LocalXHStretchGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="LOCAL_XH_STRETCH",
        produces=("R(X,H)",),
        consumes=("TOPOLOGY.BONDS", "GIC.XH_STRETCH_POLICY"),
        implemented_by="matrix_smith.generators.generate_stretch_coordinates",
        status=STATUS_LEGACY_PORTED,
        notes="Opt-in unsymmetrized X-H stretches for local-mode workflows.",
    ),
    CoordinateGeneratorSpec(
        name="AngleGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="BEND",
        produces=("A(i,j,k)",),
        consumes=("TOPOLOGY.BONDS", "GEOMETRY.CARTESIAN"),
        implemented_by="matrix_smith.definition._primitive_candidates",
        status=STATUS_LEGACY_PORTED,
    ),
    CoordinateGeneratorSpec(
        name="LocalSymmetryAngleSALCGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="LOCAL_ANGLE_SALC",
        produces=("XAng", "SymD", "Rock", "HCAn"),
        consumes=("TOPOLOGY.BONDS", "GEOMETRY.CARTESIAN", "TOPOLOGY.LOCAL_EQUIVALENCE"),
        implemented_by="matrix_smith.runtime.gicforge_python local-angle paths",
        status=STATUS_REFACTOR_BOUNDARY,
        notes=(
            "Local stretch and angle SALC boundary; "
            "bond lengths use endpoint-equivalence classes, while high coordination "
            "angles 5-9 use polyhedron templates when recognized, otherwise "
            "ligand-equivalence classes before local SVD."
        ),
    ),
    CoordinateGeneratorSpec(
        name="LinearBendGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="LINEAR_BEND",
        produces=("L(i,j,k,mode)",),
        consumes=("TOPOLOGY.BONDS", "GEOMETRY.CARTESIAN"),
        implemented_by="matrix_smith.definition._primitive_candidates",
        status=STATUS_LEGACY_PORTED,
    ),
    CoordinateGeneratorSpec(
        name="DihedralGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="TORSION",
        produces=("D(i,j,k,l)",),
        consumes=("TOPOLOGY.BONDS",),
        implemented_by="matrix_smith.definition._primitive_candidates",
        status=STATUS_LEGACY_PORTED,
        notes="Keeps one representative dihedral unless explicitly changed.",
    ),
    CoordinateGeneratorSpec(
        name="RingGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="RING_PUCKER_COMPONENT",
        produces=("RPck", "RING_BEND"),
        consumes=("TOPOLOGY.RINGS", "GEOMETRY.CARTESIAN"),
        implemented_by="matrix_smith.definition._primitive_candidates",
        status=STATUS_LEGACY_PORTED,
    ),
    CoordinateGeneratorSpec(
        name="ButterflyGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="BUTTERFLY",
        produces=("BtFl",),
        consumes=("TOPOLOGY.RINGS", "TOPOLOGY.FUSED_RING_BONDS"),
        implemented_by="matrix_smith.definition._primitive_candidates",
        status=STATUS_LEGACY_PORTED,
    ),
    CoordinateGeneratorSpec(
        name="OutOfPlaneGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="OUT_OF_PLANE",
        produces=("U(i,j,k,l)", "improper D(i,j,k,l)"),
        consumes=("TOPOLOGY.BONDS", "TOPOLOGY.RINGS"),
        implemented_by="matrix_smith.definition._primitive_candidates",
        status=STATUS_LEGACY_PORTED,
        notes="Improper-dihedral mode is a Gaussian compatibility path.",
    ),
    CoordinateGeneratorSpec(
        name="SpecialCoordinateGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="SPECIAL_COORDINATE",
        produces=("fragment centers", "orientations", "ring/bond/contact centers"),
        consumes=("TOPOLOGY.FRAGMENTS", "GEOMETRY.AUXILIARY_NODES"),
        implemented_by=(
            "matrix_smith.definition._fragment_primitive_candidates and "
            "matrix_smith.definition._interaction_center_primitive_candidates"
        ),
        status=STATUS_LEGACY_PORTED,
    ),
    CoordinateGeneratorSpec(
        name="MetalCoordinateGenerator",
        stage=STAGE_COORDINATE_GENERATION,
        coordinate_family="METAL_COORDINATION",
        produces=("eta centers", "metal-ligand auxiliary coordinates"),
        consumes=("TOPOLOGY.AUXILIARY_NODES", "GEOMETRY.AUXILIARY_NODES"),
        implemented_by=(
            "matrix_smith.definition._interaction_center_primitive_candidates and "
            "matrix_smith.runtime.gicforge_python high-coordination local SALC paths"
        ),
        status=STATUS_LEGACY_PORTED,
        notes=(
            "Active transition-metal coordinates include metal--ring/interaction-center "
            "coordinates and symmetry-adapted metal--ligand local coordinates. "
            "Template recognition accelerates common coordination polyhedra, while the "
            "generic local-SVD fallback has no coordination ceiling."
        ),
    ),
)


def generate_stretch_coordinates(
    bonds: tuple[tuple[int, int], ...],
    *,
    atom_symbols: tuple[str, ...] = (),
    xh_stretch_policy: str = "SYMMETRIZE",
    local_xh_bonds: tuple[tuple[int, int], ...] = (),
    local_xh_classes: tuple[str, ...] = (),
) -> tuple[GeneratedCoordinate, ...]:
    """Generate bonded stretch definitions, including opt-in local X-H rows."""

    xh_class_by_bond = _xh_class_by_bond(bonds, atom_symbols)
    return tuple(
        GeneratedCoordinate(
            family=(
                "LOCAL_XH_STRETCH"
                if _use_local_xh_stretch(
                    left,
                    right,
                    atom_symbols,
                    xh_stretch_policy,
                    local_xh_bonds,
                    local_xh_classes,
                    xh_class_by_bond,
                )
                else "STRETCH"
            ),
            function="R",
            atoms=(int(left), int(right)),
        )
        for left, right in bonds
    )


def default_coordinate_generator_registry() -> tuple[CoordinateGeneratorSpec, ...]:
    """Return the immutable default registry in deterministic order."""

    return DEFAULT_COORDINATE_GENERATOR_REGISTRY


def coordinate_generator_by_family() -> dict[str, CoordinateGeneratorSpec]:
    """Return one registry entry per coordinate family."""

    return {entry.coordinate_family: entry for entry in DEFAULT_COORDINATE_GENERATOR_REGISTRY}


def validate_coordinate_generator_registry(
    registry: tuple[CoordinateGeneratorSpec, ...] = DEFAULT_COORDINATE_GENERATOR_REGISTRY,
) -> tuple[str, ...]:
    """Validate deterministic registry ownership and return diagnostics."""

    diagnostics: list[str] = []
    seen_names: set[str] = set()
    seen_families: set[str] = set()
    for entry in registry:
        if entry.name in seen_names:
            diagnostics.append(f"duplicate generator name: {entry.name}")
        seen_names.add(entry.name)
        if entry.coordinate_family in seen_families:
            diagnostics.append(f"duplicate coordinate family: {entry.coordinate_family}")
        seen_families.add(entry.coordinate_family)
        if not entry.implemented_by:
            diagnostics.append(f"missing implementation owner: {entry.name}")
        if entry.stage != STAGE_COORDINATE_GENERATION:
            diagnostics.append(f"unexpected generator stage for {entry.name}: {entry.stage}")
    return tuple(diagnostics)


def _use_local_xh_stretch(
    left: int,
    right: int,
    atom_symbols: tuple[str, ...],
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    xh_class_by_bond: dict[tuple[int, int], str],
) -> bool:
    if not _is_xh_bond(left, right, atom_symbols):
        return False
    policy = _normalize_xh_stretch_policy(xh_stretch_policy)
    if policy == "SYMMETRIZE":
        return False
    if policy == "LOCAL_ALL":
        return True
    pair = _pair_key(left, right)
    if pair in set(local_xh_bonds):
        return True
    return xh_class_by_bond.get(pair, "") in set(local_xh_classes)


def _is_xh_bond(left: int, right: int, atom_symbols: tuple[str, ...]) -> bool:
    heavy, hydrogen = _xh_heavy_and_hydrogen(left, right, atom_symbols)
    return heavy is not None and hydrogen is not None


def _xh_class_by_bond(
    bonds: tuple[tuple[int, int], ...],
    atom_symbols: tuple[str, ...],
) -> dict[tuple[int, int], str]:
    hydrogens_by_heavy: dict[int, set[int]] = {}
    for left, right in bonds:
        heavy, hydrogen = _xh_heavy_and_hydrogen(left, right, atom_symbols)
        if heavy is None or hydrogen is None:
            continue
        hydrogens_by_heavy.setdefault(heavy, set()).add(hydrogen)
    result: dict[tuple[int, int], str] = {}
    for heavy, hydrogens in hydrogens_by_heavy.items():
        count = len(hydrogens)
        if count <= 1:
            xh_class = "XH"
        elif count == 2:
            xh_class = "XH2"
        else:
            xh_class = "XH3"
        for hydrogen in hydrogens:
            result[_pair_key(heavy, hydrogen)] = xh_class
    return result


def _xh_heavy_and_hydrogen(
    left: int,
    right: int,
    atom_symbols: tuple[str, ...],
) -> tuple[int | None, int | None]:
    if left < 1 or right < 1 or left > len(atom_symbols) or right > len(atom_symbols):
        return None, None
    left_symbol = str(atom_symbols[left - 1]).strip().upper()
    right_symbol = str(atom_symbols[right - 1]).strip().upper()
    if left_symbol == "H" and right_symbol != "H":
        return right, left
    if right_symbol == "H" and left_symbol != "H":
        return left, right
    return None, None


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (int(left), int(right)) if int(left) <= int(right) else (int(right), int(left))


def _normalize_xh_stretch_policy(value: str | None) -> str:
    text = str(value or "SYMMETRIZE").strip().replace("-", "_").upper()
    aliases = {
        "YES": "SYMMETRIZE",
        "TRUE": "SYMMETRIZE",
        "ALL": "LOCAL_ALL",
        "LOCAL": "LOCAL_ALL",
        "SELECTED": "LOCAL_SELECTED",
        "NO": "LOCAL_ALL",
        "FALSE": "LOCAL_ALL",
    }
    return aliases.get(text, text)
