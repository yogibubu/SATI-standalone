from __future__ import annotations

import math
from pathlib import Path
import re
from matrix_chem import read_enriched_xyz
from matrix_core import build_run_manifest
from matrix_core import read_sectioned_lines, section_content

from .route_policy import (
    GaussianRouteOverride,
    validate_gaussian_route_policy,
    validate_gaussian_route_transformation,
)


ORACLE_GAUSSIAN_GIC_INPUT_SCHEMA = "oracle.gaussian.gic_input.v1"
REQUIRED_GIC_SCHEMA = "oracle.xyz.gic.v1"
TYPED_ONIC_ARTIFACT_SCHEMA = "matrix.smith.typed_onic_artifact.v1"
DEFAULT_GIC_ROUTE = "#p hf/sto-3g opt=readallgic"
DEFAULT_SEMIDIAGONAL_ROUTE = "#p B3LYP/6-31G(d)"
DEFAULT_POINT_ROUTE = "#p HF/STO-3G Force"
# GDV uses a smaller expression parse table than ordinary Gaussian GIC input.
# Keep the canonical default below that shared parser limit; callers may still
# request a different explicit limit for a separately qualified backend.
DEFAULT_GAUSSIAN_GIC_MAX_ADDENDS = 8
# ReadAllGIC labels are transport identifiers, not scientific coordinate
# names.  Keep them within the conservative capacity shared by GDV backends.
# Real GDV parsing accepts 11-character labels (for example AppTorsD001)
# and rejects the observed 12/13-character CONTACT labels.  Preserve every
# accepted scientific name and compact only identifiers beyond that boundary.
DEFAULT_GAUSSIAN_GIC_MAX_LABEL_LENGTH = 11
# Canonical native-GDV L1 promotion from an L0 checkpoint.  Keep D3 (not
# D3BJ), retain the packaged Gen/GenECP data with Pseudo=Read, and do not add
# NoSymm: geometry, wavefunction, and the final estimated L0 Hessian are reused.
DEFAULT_GDV_L1_D3_ROUTE = (
    "#p revDSDPBEP86D3 Gen Pseudo=Read "
    "Opt=ReadFC Guess=Read Geom=Checkpoint"
)
_GEOM_PAREN_RE = re.compile(r"\bgeom\s*=\s*\((?P<body>[^)]*)\)", flags=re.IGNORECASE)
_GEOM_VALUE_RE = re.compile(
    r"\bgeom\s*=\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)",
    flags=re.IGNORECASE,
)
_OUTPUT_RE = re.compile(
    r"\boutput\s*=\s*(?:\([^)]*\)|[A-Za-z][A-Za-z0-9_-]*)",
    flags=re.IGNORECASE,
)
_POP_PAREN_RE = re.compile(
    r"\bpop(?:ulation)?\s*=\s*\((?P<body>[^)]*)\)", flags=re.IGNORECASE
)
_POP_VALUE_RE = re.compile(
    r"\bpop(?:ulation)?\s*=\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)",
    flags=re.IGNORECASE,
)
_MAYER_IOP_RE = re.compile(r"\biop\s*\(\s*6\s*/\s*80\s*=\s*1\s*\)", flags=re.IGNORECASE)
_OPT_PAREN_RE = re.compile(r"\bopt\s*=\s*\((?P<body>[^)]*)\)", flags=re.IGNORECASE)
_OPT_VALUE_RE = re.compile(r"\bopt\s*=\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)", flags=re.IGNORECASE)
_OPT_BARE_RE = re.compile(r"\bopt\b(?!\s*=)", flags=re.IGNORECASE)
_READALLGIC_OPTION_RE = re.compile(r"\breadallgic\b", flags=re.IGNORECASE)
_GICALLSYM_RE = re.compile(r"\bgic(?:all)?symm?\b", flags=re.IGNORECASE)
_GIC_VALUE_RE = re.compile(r"\bvalue\s*=", flags=re.IGNORECASE)
_GIC_VALUE_OPTION_RE = re.compile(
    r"\bvalue\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)",
    flags=re.IGNORECASE,
)
_GIC_LABEL_OPTION_RE = re.compile(r"\((?:Frozen|Inactive|Fl[A-Za-z]+|Value\s*=[^)]*)\)", flags=re.IGNORECASE)
_GAUSSIAN_GIC_SAFE_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_GIC_FUNCTION_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)\s*\(([^()]*)\)")
_GIC_DIHEDRAL_RE = re.compile(r"\bD\s*\([^()]*\)", flags=re.IGNORECASE)
_GIC_OUT_OF_PLANE_RE = re.compile(r"\bU\s*\(([^()]*)\)", flags=re.IGNORECASE)
_GIC_NONACTIVE_RE = re.compile(r"\b(?:frozen|freeze|inactive|remove|printonly)\b", flags=re.IGNORECASE)
_GIC_INACTIVE_RE = re.compile(r"\b(?:inactive|remove|printonly)\b", flags=re.IGNORECASE)
ANGSTROM_TO_BOHR = 1.8897261246257702


class GaussianWriteError(ValueError):
    """Raised when ORACLE state cannot be exported to Gaussian input."""


def validate_gaussian_geometry_identity(
    input_path: Path,
    reference_atoms,
    reference_coordinates_angstrom,
    *,
    tolerance_angstrom: float = 5.1e-9,
) -> float:
    """Require the molecule block to reproduce SMITH's Cartesian reference.

    This is strictly a transport check: it performs no alignment, atom
    reordering, symmetry operation, or other scientific transformation.
    """

    import numpy as np

    from .parsers import read_gaussian_cartesian_input

    exported = read_gaussian_cartesian_input(Path(input_path))
    expected_atoms = tuple(str(atom) for atom in reference_atoms)
    if tuple(exported.atoms) != expected_atoms:
        raise GaussianWriteError("Gaussian molecule block changed the SMITH atom order")
    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    candidate = np.asarray(exported.coordinates_angstrom, dtype=float)
    if candidate.shape != reference.shape:
        raise GaussianWriteError("Gaussian molecule block changed the SMITH atom count")
    error = float(np.max(np.abs(candidate - reference), initial=0.0))
    if error > tolerance_angstrom:
        raise GaussianWriteError(
            "Gaussian molecule block changed the SMITH Cartesian reference: "
            f"maximum error {error:.6g} angstrom"
        )
    return error


def _gaussian_basis_lines(basis_set_file: Path | str) -> list[str]:
    """Return a Gen/GenECP block without transport metadata or outer blanks.

    Basis Set Exchange Gaussian94 artifacts may contain a leading ``!`` header
    followed by blank lines.  In a complete Gaussian input a blank line ends
    the Gen section, so copying that leading header verbatim can silently
    produce a calculation in which no atom has basis functions.  Internal
    blank lines must instead be preserved: GenECP uses one to separate the
    orbital-basis and pseudopotential sections.
    """

    basis_path = Path(basis_set_file)
    raw_lines = basis_path.read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    started = False
    for raw_line in raw_lines:
        line = raw_line.rstrip()
        if not started and (not line.strip() or line.lstrip().startswith(("!", "#"))):
            continue
        if line.lstrip().startswith(("!", "#")):
            continue
        started = True
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise GaussianWriteError(f"Gaussian basis-set file is empty: {basis_path}")
    return lines


def write_gaussian_point_input(
    output: Path,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    route: str = DEFAULT_POINT_ROUTE,
    title: str = "MATRIX LINK point",
    charge: int = 0,
    multiplicity: int = 1,
    link0: tuple[str, ...] = (),
    ensure_force: bool = True,
    connectivity_bonds: tuple[tuple[int, int, float], ...] | list[tuple[int, int, float]] | None = None,
    basis_set_file: Path | str | None = None,
    modredundant_lines: tuple[str, ...] | list[str] | None = None,
    route_override: GaussianRouteOverride | None = None,
) -> Path:
    """Write a Cartesian Gaussian single-point/force input."""

    import numpy as np

    target = Path(output)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GaussianWriteError("point coordinates must have shape natoms x 3")
    if len(atoms) != coords.shape[0]:
        raise GaussianWriteError("atom count does not match coordinate array")
    target.parent.mkdir(parents=True, exist_ok=True)
    route_text = _point_route_with_connectivity(
        _normalize_point_route(
            route,
            ensure_force=ensure_force,
            route_override=route_override,
        ),
        connectivity_bonds is not None,
    )
    validate_gaussian_route_transformation(
        route,
        route_text,
        override=route_override,
    )
    lines = [
        *[item.strip() for item in link0 if item.strip()],
        route_text,
        "",
        title,
        "",
        f"{int(charge)} {int(multiplicity)}",
    ]
    for atom, xyz in zip(atoms, coords, strict=True):
        lines.append(
            f"{atom:<3s} {float(xyz[0]): .16f} {float(xyz[1]): .16f} {float(xyz[2]): .16f}"
        )
    if connectivity_bonds is not None:
        lines.append("")
        lines.extend(_gaussian_connectivity_lines(len(atoms), connectivity_bonds))
    if modredundant_lines:
        lines.append("")
        lines.extend(str(line).strip() for line in modredundant_lines if str(line).strip())
    lines.append("")
    if basis_set_file is not None:
        lines.extend(_gaussian_basis_lines(basis_set_file))
        lines.append("")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_gaussian_checkpoint_optimization_input(
    output: Path,
    *,
    checkpoint: str,
    route: str = DEFAULT_GDV_L1_D3_ROUTE,
    title: str = "MATRIX checkpoint optimization",
    charge: int = 0,
    multiplicity: int = 1,
    link0: tuple[str, ...] = (),
    basis_set_file: Path | str | None = None,
    route_override: GaussianRouteOverride | None = None,
) -> Path:
    """Write an optimization that reuses geometry, guess and Hessian from a checkpoint."""

    checkpoint_name = str(checkpoint).strip()
    if not checkpoint_name:
        raise GaussianWriteError("checkpoint optimization needs a checkpoint")
    route_text = _normalize_point_route(
        route,
        ensure_force=False,
        route_override=route_override,
    )
    required = ("geom=checkpoint", "guess=read", "opt=readfc")
    lowered = route_text.lower().replace(" ", "")
    if any(keyword not in lowered for keyword in required):
        raise GaussianWriteError(
            "checkpoint optimization route must request Geom=Checkpoint, "
            "Guess=Read and Opt=ReadFC"
        )
    target = Path(output)
    lines = [
        f"%chk={checkpoint_name}",
        *[
            item.strip()
            for item in link0
            if item.strip() and not item.strip().lower().startswith("%chk=")
        ],
        route_text,
        "",
        title,
        "",
        f"{int(charge)} {int(multiplicity)}",
        "",
    ]
    if basis_set_file is not None:
        lines.extend(_gaussian_basis_lines(basis_set_file))
        lines.append("")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _point_route_with_connectivity(route: str, enabled: bool) -> str:
    if not enabled or re.search(r"\bgeom\s*=", route, flags=re.IGNORECASE):
        return route
    return f"{route} Geom=Connectivity"


def _gaussian_connectivity_lines(
    natoms: int,
    bonds: tuple[tuple[int, int, float], ...] | list[tuple[int, int, float]],
) -> list[str]:
    neighbors: dict[int, list[tuple[int, float]]] = {index: [] for index in range(1, natoms + 1)}
    for left_zero, right_zero, order in bonds:
        left, right = int(left_zero) + 1, int(right_zero) + 1
        if left not in neighbors or right not in neighbors or left == right:
            raise GaussianWriteError("connectivity bond indices are invalid")
        neighbors[min(left, right)].append((max(left, right), float(order)))
    lines = []
    for atom in range(1, natoms + 1):
        suffix = " ".join(
            f"{other} {order:.6g}" for other, order in sorted(neighbors[atom])
        )
        lines.append(f"{atom} {suffix}".rstrip())
    return lines


def write_gaussian_oniom_point_input(
    output: Path,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    high_atoms: tuple[int, ...] | list[int],
    atom_types: tuple[str, ...] | list[str] = (),
    route: str,
    title: str = "MATRIX LINK ONIOM point",
    charge: int = 0,
    multiplicity: int = 1,
    link0: tuple[str, ...] = (),
    ensure_force: bool = True,
    basis_set_file: Path | str | None = None,
    route_override: GaussianRouteOverride | None = None,
) -> Path:
    """Write a two-layer Cartesian ONIOM force point.

    high_atoms uses zero-based MATRIX indices. Atom types default to a
    conservative UFF mapping for common organic and microsolvation elements.
    """

    import numpy as np

    target = Path(output)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(atoms), 3):
        raise GaussianWriteError("ONIOM point coordinates must have shape natoms x 3")
    high = {int(index) for index in high_atoms}
    if not high or min(high) < 0 or max(high) >= len(atoms):
        raise GaussianWriteError("ONIOM high-layer atom indices are invalid")
    types = tuple(atom_types) if atom_types else tuple(_default_uff_type(atom) for atom in atoms)
    if len(types) != len(atoms):
        raise GaussianWriteError("ONIOM atom type count must match atoms")
    route_text = _normalize_point_route(
        route,
        ensure_force=ensure_force,
        route_override=route_override,
    )
    lines = [
        *[item.strip() for item in link0 if item.strip()],
        route_text,
        "",
        title,
        "",
        f"{int(charge)} {int(multiplicity)}",
    ]
    for index, (atom, atom_type, xyz) in enumerate(zip(atoms, types, coords, strict=True)):
        layer = "H" if index in high else "L"
        lines.append(
            f"{atom}-{atom_type}-0.0 {float(xyz[0]): .16f} {float(xyz[1]): .16f} "
            f"{float(xyz[2]): .16f} {layer}"
        )
    lines.append("")
    if basis_set_file is not None:
        lines.extend(_gaussian_basis_lines(basis_set_file))
        lines.append("")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _default_uff_type(atom: str) -> str:
    symbol = str(atom).strip().capitalize()
    mapping = {
        "H": "H_",
        "C": "C_3",
        "N": "N_3",
        "O": "O_3",
        "F": "F_",
        "P": "P_3+3",
        "S": "S_3+2",
        "Cl": "Cl",
        "Br": "Br",
        "I": "I_",
    }
    try:
        return mapping[symbol]
    except KeyError as exc:
        raise GaussianWriteError(
            f"no default UFF atom type for {atom}; provide oniom_atom_types"
        ) from exc


def ensure_gaussian_output_pickett(
    route: str,
    *,
    route_override: GaussianRouteOverride | None = None,
) -> str:
    """Return a Gaussian route that explicitly asks for Pickett output."""
    validate_gaussian_route_policy(route, override=route_override)
    text = _ensure_gaussian_output_pickett_unchecked(route)
    validate_gaussian_route_transformation(route, text, override=route_override)
    return text


def _ensure_gaussian_output_pickett_unchecked(route: str) -> str:
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    if not text.startswith("#"):
        text = f"# {text}"
    if _OUTPUT_RE.search(text):
        text = _OUTPUT_RE.sub("output=pickett", text, count=1)
    else:
        text = f"{text} output=pickett"
    return _collapse_route(text)


def _ensure_oracle_population_contract(route: str) -> str:
    """Request CM5 charges and Mayer bond orders in ORACLE Gaussian jobs."""

    text = route.strip()
    if match := _POP_PAREN_RE.search(text):
        options = [item.strip() for item in match.group("body").split(",") if item.strip()]
        if not any(option.lower() == "hirshfeld" for option in options):
            options.append("Hirshfeld")
            text = _POP_PAREN_RE.sub(f"Pop=({','.join(options)})", text, count=1)
    elif match := _POP_VALUE_RE.search(text):
        value = match.group("value")
        if value.lower() != "hirshfeld":
            text = _POP_VALUE_RE.sub(f"Pop=({value},Hirshfeld)", text, count=1)
    else:
        text = f"{text} Pop=Hirshfeld"
    if not _MAYER_IOP_RE.search(text):
        text = f"{text} IOp(6/80=1)"
    return _collapse_route(text)


def write_gicforge_gaussian_input(
    enriched_xyz: Path,
    output: Path,
    *,
    route: str = DEFAULT_GIC_ROUTE,
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
    """Write a Gaussian input from an enriched XYZ carrying frozen GIC or typed ONIC.

    Long additive definitions are factored automatically through deterministic
    ``Inactive`` helper variables.  This preserves every physical GIC while
    keeping each expression below Gaussian's finite parser capacity.  Pass
    ``None`` only for diagnostics that require the unfactored text.  Typed-ONIC
    artifacts are exported through SMITH's canonical Cartesian, inverse-distance
    and delegated-GIC translators before the same parser-safe factorization.
    """
    if not g16_compatibility:
        if total_symmetric_only:
            raise GaussianWriteError(
                "native SONIC export cannot select a scientific coordinate subset; "
                "the writer must consume the complete SMITH chart"
            )
        if freeze_non_total is True:
            raise GaussianWriteError(
                "native SONIC export cannot infer Frozen coordinates; "
                "activation state must come from the SMITH contract"
            )
    source = Path(enriched_xyz)
    _require_gic_section(source)
    geometry = read_enriched_xyz(source)
    job_charge = (
        charge if charge is not None else geometry.charge if geometry.charge is not None else 0
    )
    job_multiplicity = (
        multiplicity
        if multiplicity is not None
        else geometry.multiplicity
        if geometry.multiplicity is not None
        else 1
    )
    route_text = _ensure_oracle_population_contract(
        _normalize_route(route, route_override=route_override)
    )
    # The Cartesian geometry is the authoritative certified SMITH structure.
    # Leave symmetry detection and its stepwise refresh entirely to GDV.
    # Injecting either PG= or Follow can retain SABF tables that no longer
    # match a later ReadAllGIC geometry.
    route_text = _route_with_gdv_detected_symmetry(route_text)
    validate_gaussian_route_transformation(
        route,
        route_text,
        override=route_override,
    )
    lines = [
        *[item.strip() for item in link0 if item.strip()],
        route_text,
        "",
        title or geometry.comment or source.stem,
        "",
        f"{job_charge} {job_multiplicity}",
    ]
    for atom, (x, y, z) in zip(geometry.atoms, geometry.coordinates_angstrom):
        # Preserve the certified geometry used to evaluate ReadAllGIC Value fields.
        lines.append(f"{atom:2s} {x: .16f} {y: .16f} {z: .16f}")
    lines.append("")
    gic_lines = _gaussian_gic_lines(
        source,
        total_symmetric_only=total_symmetric_only,
        freeze_non_total=freeze_non_total,
    )
    removed_non_total_lines: list[str] = []
    effective_freeze_non_total = freeze_non_total
    if effective_freeze_non_total is None and g16_compatibility and total_symmetric_only:
        # Let SMITH's scientific-path policy remain the single source of
        # truth.  A default-frozen chart exposes at least one Frozen row in
        # its complete export; TS exploitation exposes none.
        effective_freeze_non_total = any(
            "(Frozen)" in line
            for line in _gaussian_gic_lines(
                source,
                total_symmetric_only=False,
                freeze_non_total=None,
            )
        )
    if g16_compatibility and total_symmetric_only and effective_freeze_non_total:
        complete_lines = _gaussian_gic_lines(
            source,
            total_symmetric_only=False,
            freeze_non_total=True,
        )
        retained_labels = {
            _gaussian_base_label(parsed[0])
            for line in gic_lines
            if (parsed := _gaussian_definition(line)) is not None
        }
        removed_non_total_lines = [
            line
            for line in complete_lines
            if (parsed := _gaussian_definition(line)) is not None
            and _gaussian_base_label(parsed[0]) not in retained_labels
        ]
    if gic_lines or removed_non_total_lines:
        if g16_compatibility:
            gic_lines = _gaussian_g16_linear_ring_puckering_lines(gic_lines)
            gic_lines = _gaussian_g16_compatible_active_dihedral_lines(gic_lines)
            gic_lines.extend(
                _gaussian_g16_frozen_primitives_for_removed_gics(
                    removed_non_total_lines,
                    retained_lines=gic_lines,
                )
            )
        else:
            gic_lines = _gaussian_native_inactive_improper_helpers(gic_lines)
        gic_lines = _gaussian_compact_transport_labels(gic_lines)
        gic_lines = _gaussian_gic_lines_with_values(
            gic_lines,
            geometry.coordinates_angstrom,
        )
        gic_lines = _gaussian_factor_long_gic_lines(
            gic_lines,
            max_addends=max_gic_expression_addends,
        )
        lines.extend(gic_lines)
        lines.append("")
    # ReadAllGIC belongs to the molecular-specification input and Gaussian
    # consumes it before any Gen/GenECP basis and pseudopotential sections.
    if basis_set_file is not None:
        lines.extend(_gaussian_basis_lines(basis_set_file))
        lines.append("")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate_gaussian_geometry_identity(
        target,
        geometry.atoms,
        geometry.coordinates_angstrom,
    )
    return target


def write_semidiagonal_cubic_vibrot_gaussian_input(
    output: Path,
    *,
    oldchk: str,
    route: str = DEFAULT_SEMIDIAGONAL_ROUTE,
    harmonic_chk: str | None = None,
    cubic_chk: str | None = None,
    symmetry: str = "sym=com",
    xyzin: Path | str | None = None,
    manifest_path: Path | str | None = None,
    write_manifest: bool = True,
    route_override: GaussianRouteOverride | None = None,
) -> Path:
    """Write the two-link Gaussian vibrot/numerical-gradient job for F3ijj.

    The input assumes `oldchk` already contains the optimized geometry and basis.
    Gaussian then performs a harmonic `freq=vibrot` step followed by
    `freq=(numer,readharm,vibrot)`, which differentiates analytical gradients
    and prints the semidiagonal cubic constants used by MATRIX.
    """
    if not oldchk.strip():
        raise GaussianWriteError("oldchk cannot be empty")
    validate_gaussian_route_policy(route, override=route_override)
    route_body = _route_body(route)
    harmonic = harmonic_chk or _replace_checkpoint_suffix(oldchk, ".harm.chk")
    cubic = cubic_chk or _replace_checkpoint_suffix(oldchk, ".cubic.chk")
    sym = symmetry.strip()
    options = " ".join(item for item in (route_body, sym, "geom=allcheck", "freq=vibrot") if item)
    cubic_options = " ".join(
        item for item in (route_body, sym, "geom=allcheck", "freq=(numer,readharm,vibrot)") if item
    )
    harmonic_route = f"#p {options}"
    cubic_route = f"#p {cubic_options}"
    validate_gaussian_route_transformation(
        route,
        harmonic_route,
        override=route_override,
    )
    validate_gaussian_route_transformation(
        route,
        cubic_route,
        override=route_override,
    )
    lines = [
        f"%oldchk={oldchk}",
        f"%chk={harmonic}",
        harmonic_route,
        "",
        "",
        "--Link1--",
        f"%oldchk={harmonic}",
        f"%chk={cubic}",
        cubic_route,
        "",
        "",
    ]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    if write_manifest:
        inputs = {"oldchk": Path(oldchk)}
        if xyzin is not None:
            inputs["xyzin"] = Path(xyzin)
        manifest = build_run_manifest(
            workflow="gaussian-semidiagonal-cubic-vibrot-input",
            status="prepared",
            run_dir=target.parent,
            inputs=inputs,
            outputs={"gaussian_input": target},
            parameters={
                "route": route,
                "route_body": route_body,
                "oldchk": oldchk,
                "harmonic_chk": harmonic,
                "cubic_chk": cubic,
                "symmetry": sym,
                "keywords": ("freq=vibrot", "freq=(numer,readharm,vibrot)"),
            },
            backend={"name": "Gaussian"},
        )
        manifest.write(
            Path(manifest_path)
            if manifest_path is not None
            else target.with_suffix(target.suffix + ".manifest.json")
        )
    return target


def _require_gic_section(path: Path) -> None:
    lines = read_sectioned_lines(Path(path))
    gic = section_content(lines, "GIC")
    expected = f"SCHEMA {REQUIRED_GIC_SCHEMA}"
    if gic:
        if gic[0].strip() != expected:
            raise GaussianWriteError(f"#GIC must start with {expected!r}; found {gic[0]!r}")
        if not any(line.strip().upper() == "STATUS BUILT" for line in gic):
            raise GaussianWriteError(
                "#GIC must have STATUS BUILT before Gaussian ReadAllGIC export"
            )
        return
    typed = section_content(lines, "TYPED_ONIC")
    typed_expected = f"SCHEMA {TYPED_ONIC_ARTIFACT_SCHEMA}"
    if typed and typed[0].strip() == typed_expected:
        return
    if typed:
        raise GaussianWriteError(
            f"#TYPED_ONIC must start with {typed_expected!r}; found {typed[0]!r}"
        )
    raise GaussianWriteError("missing #GIC or #TYPED_ONIC section")


def _route_with_gdv_detected_symmetry(route: str) -> str:
    """Leave symmetry detection and refresh entirely to Gaussian/GDV.

    Forcing ``PG=...`` from the serialized SMITH chart makes Gaussian combine
    an externally assigned point group with its own standard orientation and
    symmetry-adapted basis.  On later ``ReadAllGIC`` optimization steps that
    can leave the SABF tables inconsistent with the updated geometry.  The
    same corruption is observed with ``Follow`` when GDV's largest Abelian
    subgroup changes after a step.  The Cartesian structure is already the
    certified SMITH structure, so the unmodified default GDV policy is the
    only robust owner of symmetry.  An explicit caller option remains
    authoritative.  ``NoSymm`` is never introduced.
    """
    return route


def _gaussian_gic_lines(
    path: Path,
    *,
    total_symmetric_only: bool = False,
    freeze_non_total: bool | None = None,
) -> list[str]:
    try:
        from matrix_smith import gaussian_gic_lines_from_xyzin
    except ImportError:
        return []
    return gaussian_gic_lines_from_xyzin(
        Path(path),
        total_symmetric_only=total_symmetric_only,
        freeze_non_total=freeze_non_total,
    )


def _gaussian_gic_lines_with_values(
    gic_lines: list[str],
    coordinates_angstrom: object,
) -> list[str]:
    # GAUSSIAN GIC VALUE CONVENTIONS -- KEEP THIS BLOCK WITH THE WRITER.
    #
    # An explicit Value= is not merely documentation: ReadAllGIC uses it as
    # the initial internal-coordinate value.  A value expressed in the wrong
    # units, or evaluated with a different atom order, makes Gaussian rebuild
    # a different Cartesian geometry before the first electronic calculation.
    #
    #  * Cartesian input and direct R(...) Value= fields use Angstrom.
    #  * Direct A(...), D(...), and U(...) Value= fields use degrees.
    #  * Inside a generic expression (for example c1*A(...)+c2*A(...)),
    #    distances/Cartesians use bohr and angular primitives use radians.
    #    The resulting scalar Value= must therefore remain in those native
    #    expression units; never convert the final combination to degrees.
    #  * Gaussian U(center, plane1, plane2, out) is ordered exactly as shown:
    #    plane1 and plane2 orient the normal and the fourth atom is displaced.
    #    Gaussian also uses the opposite sign from the corresponding positive
    #    scalar triple product; see _out_of_plane below.
    #  * Gaussian L(a1, center, a2, reference, component) is a periodic angular
    #    linear-bend component.  The center is the second atom and component is
    #    -1 or -2.  For a locally linear moiety in a globally nonlinear
    #    molecule with more than three atoms, Gaussian must receive the same
    #    well-conditioned real reference atom in both components.  That atom
    #    is part of the primitive constructed by ORACLE/SMITH, so its Wilson
    #    row and rank are frozen before serialization.  The writer must never
    #    invent or replace it.  A plane reference such as 0 is retained only
    #    when the primitive contract has no real fourth atom.  Omit Value= for
    #    every expression containing L and let Gaussian evaluate its exact
    #    periodic angular convention.
    #
    # Regression tests cover direct/composite U and both L components.  Do not
    # "simplify" these asymmetric rules into one global angular conversion.
    coords = [
        (float(row[0]), float(row[1]), float(row[2]))
        for row in coordinates_angstrom
    ]
    values: dict[str, float] = {}
    deferred_labels: set[str] = set()
    out: list[str] = []
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            out.append(line)
            continue
        label, expression = parsed
        base_label = _gaussian_base_label(label)
        supplied_value = _gaussian_supplied_gic_value(label)
        nonactive = _GIC_NONACTIVE_RE.search(label) is not None
        fragment_declaration = (
            re.fullmatch(
                r"\s*Fragment\s*\([^()]*\)\s*",
                expression,
                flags=re.IGNORECASE,
            )
            is not None
        )
        linear_bend = re.search(r"\bL\s*\(", expression, flags=re.IGNORECASE) is not None
        identifiers = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expression))
        depends_on_deferred = bool(identifiers & deferred_labels)
        if nonactive:
            if supplied_value is not None:
                raise GaussianWriteError(
                    f"non-active Gaussian GIC {base_label} must not carry Value="
                )
            if fragment_declaration or linear_bend or depends_on_deferred:
                deferred_labels.add(base_label)
            else:
                try:
                    values[base_label] = _evaluate_gaussian_gic_expression(
                        expression,
                        coords,
                        values,
                    )
                except (ArithmeticError, ValueError, KeyError, SyntaxError, NameError) as exc:
                    raise GaussianWriteError(
                        f"cannot evaluate inactive Gaussian helper {base_label}: {expression}"
                    ) from exc
            out.append(line)
            continue
        # Fragment declarations define Gaussian GIC objects rather than scalar
        # coordinates.  They must remain literal (for example
        # ``F001=Fragment(1-5)``) and must not receive a Value option.
        if fragment_declaration:
            if supplied_value is not None:
                raise GaussianWriteError(
                    f"Gaussian Fragment declaration {base_label} must not carry Value="
                )
            deferred_labels.add(base_label)
            out.append(line)
            continue
        # Gaussian's five-argument L(i,j,k,m,n) is an oriented linear-bend
        # component, not SMITH's adimensional local-frame component.  In the
        # canonical three-atom form i and k are the endpoints, j is the center,
        # m=0 selects Gaussian's automatic projection plane, and n=-1/-2
        # selects the component.  Let Gaussian evaluate its exact angular
        # value instead of manufacturing a mismatched initial geometry.
        if linear_bend:
            if supplied_value is not None:
                raise GaussianWriteError(
                    f"Gaussian linear-bend GIC {base_label} must omit Value="
                )
            deferred_labels.add(base_label)
            out.append(line)
            continue
        if depends_on_deferred:
            # Gaussian evaluates fragment/center helper variables declared as
            # Inactive.  Their numerical convention belongs to Gaussian, so
            # dependent active rows must likewise be left without Value.
            if supplied_value is not None:
                raise GaussianWriteError(
                    f"Gaussian GIC {base_label} depends on deferred helpers and must omit Value="
                )
            deferred_labels.add(base_label)
            out.append(line)
            continue
        try:
            value = _evaluate_gaussian_gic_expression(expression, coords, values)
        except (ArithmeticError, ValueError, KeyError, SyntaxError, NameError) as exc:
            raise GaussianWriteError(
                f"cannot evaluate initial value for Gaussian GIC {base_label}: {expression}"
            ) from exc
        values[base_label] = value
        if supplied_value is not None:
            if not math.isclose(supplied_value, value, rel_tol=1.0e-10, abs_tol=1.0e-9):
                raise GaussianWriteError(
                    f"Gaussian GIC {base_label} has inconsistent Value=: "
                    f"supplied {supplied_value:.12g}, expected {value:.12g} from the "
                    "exported Cartesian geometry and native ReadAllGIC units"
                )
            out.append(line)
        else:
            out.append(_gaussian_line_with_inline_value(label, expression, value))
    return out


def _gaussian_supplied_gic_value(label: str) -> float | None:
    match = _GIC_VALUE_OPTION_RE.search(label)
    if match is None:
        if _GIC_VALUE_RE.search(label):
            raise GaussianWriteError(f"Gaussian GIC has a malformed Value= option: {label}")
        return None
    try:
        value = float(match.group("value").replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise GaussianWriteError(f"Gaussian GIC has a malformed Value= option: {label}") from exc
    if not math.isfinite(value):
        raise GaussianWriteError(f"Gaussian GIC has a non-finite Value= option: {label}")
    return value


def _gaussian_native_inactive_improper_helpers(gic_lines: list[str]) -> list[str]:
    """Append one inactive improper helper for each direct frozen ``U`` row.

    Native GDV/Gaussian input keeps ``U(center,plane1,plane2,out)`` as the
    operative out-of-plane coordinate.  A direct frozen U row also exposes
    its exactly equivalent improper dihedral as an inactive helper.  This
    preserves the primitive periodic direction without replacing, activating,
    or changing the atom order of the native U coordinate.
    """

    output = list(gic_lines)
    used_labels = {
        _gaussian_base_label(parsed[0])
        for line in gic_lines
        if (parsed := _gaussian_definition(line)) is not None
    }
    existing_dihedrals = {
        _canonical_gaussian_expression(dihedral)
        for line in gic_lines
        if (parsed := _gaussian_definition(line)) is not None
        for dihedral in _GIC_DIHEDRAL_RE.findall(parsed[1])
    }
    helper_index = 1
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            continue
        label, expression = parsed
        if "(FROZEN)" not in label.upper():
            continue
        if re.fullmatch(r"\s*U\s*\([^()]*\)\s*", expression, flags=re.IGNORECASE) is None:
            continue
        improper = _canonical_gaussian_expression(
            _gaussian_g16_out_of_plane_to_improper(expression)
        )
        if improper in existing_dihedrals:
            continue
        while f"V{helper_index:04d}" in used_labels:
            helper_index += 1
        helper = f"V{helper_index:04d}"
        helper_index += 1
        used_labels.add(helper)
        existing_dihedrals.add(improper)
        output.append(f"{helper}(Inactive) = {improper}")
    return output


def _gaussian_g16_compatible_active_dihedral_lines(gic_lines: list[str]) -> list[str]:
    """Translate native out-of-plane rows and adapt composite torsions for G16.

    ORACLE and SMITH always store U(center,n1,n2,n3).  Gaussian 16 receives the
    equivalent improper D(n1,center,n3,n2) only in the generated input file.
    Gaussian 16 also does not apply periodic wrapping to the components of an
    active generic GIC containing multiple D(...) terms.  An active composite
    torsion is therefore represented by exactly one of its physical periodic
    dihedrals (the Gaussian-side ONEDIH proxy), rather than by all components,
    which would overcomplete the optimization space.  Frozen/inactive composite
    rows are still expanded componentwise.  Post-processing reads the optimized
    Cartesians and projects them back onto the unchanged native SONIC contract.
    """

    out: list[str] = []
    active_dihedrals: set[str] = set()
    dihedral_aliases: dict[str, str] = {}
    component_index = 1

    def alias_for(canonical: str) -> str:
        nonlocal component_index
        existing = dihedral_aliases.get(canonical)
        if existing is not None:
            return existing
        helper = f"V{component_index:04d}"
        component_index += 1
        dihedral_aliases[canonical] = helper
        out.append(f"{helper}(Inactive) = {canonical}")
        return helper

    def replace_dihedrals(text: str) -> str:
        return _GIC_DIHEDRAL_RE.sub(
            lambda match: alias_for(_canonical_gaussian_expression(match.group(0))),
            text,
        )

    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            out.append(line)
            continue
        label, expression = parsed
        expression = _gaussian_g16_out_of_plane_to_improper(expression)
        line = f"{label} = {expression}"
        direct_dihedral = re.fullmatch(r"\s*D\s*\([^()]*\)\s*", expression, flags=re.IGNORECASE)
        if direct_dihedral:
            canonical = _canonical_gaussian_expression(direct_dihedral.group(0))
            alias = dihedral_aliases.get(canonical)
            if alias is not None:
                out.append(f"{label} = {alias}")
            else:
                dihedral_aliases[canonical] = _gaussian_base_label(label)
                if not _GIC_NONACTIVE_RE.search(label):
                    active_dihedrals.add(canonical)
                out.append(line)
            continue
        dihedrals = _GIC_DIHEDRAL_RE.findall(expression)
        if len(dihedrals) > 1:
            nonactive = _GIC_NONACTIVE_RE.search(label) is not None
            if not nonactive:
                representatives = [
                    _canonical_gaussian_expression(dihedral) for dihedral in dihedrals
                ]
                representative = next(
                    (
                        candidate
                        for candidate in representatives
                        if candidate not in active_dihedrals
                    ),
                    None,
                )
                if representative is None:
                    raise GaussianWriteError(
                        f"active Gaussian torsion {label!r} has no unique dihedral proxy"
                    )
                active_dihedrals.add(representative)
                alias = dihedral_aliases.get(representative)
                if alias is None:
                    dihedral_aliases[representative] = _gaussian_base_label(label)
                    out.append(f"{label} = {representative}")
                else:
                    out.append(f"{label} = {alias}")
                continue
            out.append(f"{label} = {replace_dihedrals(expression)}")
        else:
            out.append(f"{label} = {replace_dihedrals(expression)}")
    return out


def _gaussian_g16_frozen_primitives_for_removed_gics(
    removed_lines: list[str],
    *,
    retained_lines: list[str],
) -> list[str]:
    """Keep primitive torsional directions after total-symmetry filtering.

    Non-totally-symmetric collective ``RPck``/``OuPl`` rows are intentionally
    absent from a symmetry-constrained Gaussian optimization.  Gaussian still
    needs their primitive dihedral directions as frozen chart rows.  Expand
    those removed definitions once, after native U-to-improper conversion, and
    deduplicate them against every retained Gaussian-16 dihedral.
    """

    used_labels = {
        _gaussian_base_label(parsed[0])
        for line in (*retained_lines, *removed_lines)
        if (parsed := _gaussian_definition(line)) is not None
    }
    helper_index = max(
        (
            int(match.group(1))
            for label in used_labels
            if (match := re.fullmatch(r"V([0-9]+)", label, flags=re.IGNORECASE))
        ),
        default=0,
    ) + 1
    seen = {
        _canonical_gaussian_expression(dihedral)
        for line in retained_lines
        if (parsed := _gaussian_definition(line)) is not None
        for dihedral in _GIC_DIHEDRAL_RE.findall(parsed[1])
    }
    output: list[str] = []
    for line in removed_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            continue
        expression = _gaussian_g16_out_of_plane_to_improper(parsed[1])
        for dihedral in _GIC_DIHEDRAL_RE.findall(expression):
            canonical = _canonical_gaussian_expression(dihedral)
            if canonical in seen:
                continue
            while f"V{helper_index:04d}" in used_labels:
                helper_index += 1
            label = f"V{helper_index:04d}"
            helper_index += 1
            used_labels.add(label)
            seen.add(canonical)
            output.append(f"{label}(Frozen) = {canonical}")
    return output


def _gaussian_g16_linear_ring_puckering_lines(gic_lines: list[str]) -> list[str]:
    """Replace Q/Phi functionals by their linear RPck components.

    The unprotected SMITH/GDV contract exposes polar puckering functions.
    Commercial Gaussian 16 instead receives the paired linear components as
    active coordinates.
    """

    polar_labels: set[str] = set()
    referenced_components: set[str] = set()
    frozen_components: set[str] = set()
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            continue
        label, expression = parsed
        base_label = _gaussian_base_label(label)
        if not base_label.startswith(("QPck", "PhiP")):
            continue
        polar_labels.add(base_label)
        dependencies = set(re.findall(r"\b[A-Za-z0-9]*RPck[0-9]+\b", expression))
        referenced_components.update(dependencies)
        if re.search(r"\bFrozen\b", label, flags=re.IGNORECASE):
            frozen_components.update(dependencies)

    if not polar_labels:
        return list(gic_lines)
    out: list[str] = []
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            out.append(line)
            continue
        label, expression = parsed
        base_label = _gaussian_base_label(label)
        if base_label in polar_labels:
            continue
        if base_label in referenced_components:
            label = re.sub(r"\(\s*Inactive\s*\)", "", label, flags=re.IGNORECASE)
            if base_label in frozen_components and not re.search(
                r"\bFrozen\b", label, flags=re.IGNORECASE
            ):
                label = f"{label.strip()}(Frozen)"
            line = f"{label} = {expression}"
        out.append(line)
    return out


def _gaussian_g16_out_of_plane_to_improper(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        atoms = [item.strip() for item in match.group(1).split(",")]
        if len(atoms) != 4 or any(not re.fullmatch(r"[1-9][0-9]*", atom) for atom in atoms):
            raise GaussianWriteError(
                "G16 out-of-plane translation requires U(center,n1,n2,n3) atom indices"
            )
        center, n1, n2, n3 = atoms
        return f"D({n1},{center},{n3},{n2})"

    return _GIC_OUT_OF_PLANE_RE.sub(replace, expression)


def _gaussian_top_level_addends(expression: str) -> list[str]:
    """Return signed top-level addends without splitting function arguments."""

    text = expression.strip()
    if not text:
        return []
    addends: list[str] = []
    stack: list[str] = []
    start = 0
    closing = {"(": ")", "[": "]"}
    for index, character in enumerate(text):
        if character in closing:
            stack.append(character)
            continue
        if character in ")]":
            if not stack or closing[stack.pop()] != character:
                raise GaussianWriteError(
                    f"unbalanced delimiter in Gaussian GIC expression: {expression}"
                )
            continue
        if (
            not stack
            and character in "+-"
            and index > start
            and text[index - 1] not in "eEdD*/^+-("
        ):
            addends.append(text[start:index].strip())
            start = index
    if stack:
        raise GaussianWriteError(
            f"unbalanced delimiter in Gaussian GIC expression: {expression}"
        )
    addends.append(text[start:].strip())
    return [addend for addend in addends if addend]


def _gaussian_additive_expression(addends: list[str]) -> str:
    expression = "".join(addends)
    return expression[1:] if expression.startswith("+") else expression


def _gaussian_factor_long_gic_lines(
    gic_lines: list[str],
    *,
    max_addends: int | None = DEFAULT_GAUSSIAN_GIC_MAX_ADDENDS,
) -> list[str]:
    """Factor long additive GICs into parser-safe ``Inactive`` helpers.

    Helpers are emitted before the coordinate that consumes them.  Factoring
    continues hierarchically, so both helper definitions and the final physical
    definition contain at most ``max_addends`` top-level terms.  Coordinate
    labels, options, order, values, and Frozen/active status are unchanged.
    """

    if max_addends is None:
        return list(gic_lines)
    if isinstance(max_addends, bool) or not isinstance(max_addends, int):
        raise GaussianWriteError("max_gic_expression_addends must be an integer or None")
    if max_addends < 2:
        raise GaussianWriteError("max_gic_expression_addends must be at least 2")

    occupied: set[str] = set()
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is not None:
            occupied.add(_gaussian_base_label(parsed[0]))
    next_helper = 1

    def allocate_helper() -> str:
        nonlocal next_helper
        while True:
            candidate = f"V{next_helper:04d}"
            next_helper += 1
            if candidate not in occupied:
                occupied.add(candidate)
                return candidate

    output: list[str] = []
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            output.append(line)
            continue
        label, expression = parsed
        addends = _gaussian_top_level_addends(expression)
        if len(addends) <= max_addends:
            output.append(line)
            continue

        current = addends
        while len(current) > max_addends:
            references: list[str] = []
            for offset in range(0, len(current), max_addends):
                chunk = current[offset : offset + max_addends]
                # A one-term helper is only an alias.  Native GDV rejects
                # aliases with ``CrInp1: No new entry added`` because the
                # definition introduces no primitive that is not already in
                # the GIC table.  Carry an odd remainder into the next level
                # verbatim, including its sign, instead of serializing it.
                if len(chunk) == 1:
                    references.append(chunk[0])
                    continue
                helper = allocate_helper()
                output.append(
                    f"{helper}(Inactive) = {_gaussian_additive_expression(chunk)}"
                )
                references.append(helper)
            current = []
            for index, reference in enumerate(references):
                if index == 0 or reference.startswith(("+", "-")):
                    current.append(reference)
                else:
                    current.append(f"+{reference}")
        output.append(f"{label} = {_gaussian_additive_expression(current)}")
    return output


def _canonical_gaussian_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression.strip())




def _gaussian_definition(line: str) -> tuple[str, str] | None:
    if "=" not in line:
        return None
    # Active definitions may carry a ``Value=...`` label option.  Generated
    # scalar expressions contain no assignment operator, so the final equals
    # sign is the unambiguous definition boundary.
    label, expression = line.rsplit("=", 1)
    label = label.strip()
    expression = expression.strip()
    if not label or not expression:
        return None
    return label, expression


def _gaussian_base_label(label: str) -> str:
    return _GIC_LABEL_OPTION_RE.sub("", label.strip()).strip()


def _gaussian_compact_transport_labels(
    gic_lines: list[str],
    *,
    max_length: int = DEFAULT_GAUSSIAN_GIC_MAX_LABEL_LENGTH,
) -> list[str]:
    """Map parser-unsafe scientific names to deterministic transport labels.

    The mapping is applied only at the Gaussian boundary and rewrites every
    dependency in the serialized expressions.  SMITH names, coordinate order,
    coefficients, activation state, and Cartesian geometry remain unchanged.
    A transport label is replaced when it exceeds the qualified GDV capacity
    or contains punctuation (for example the primes in A' and A'' irreps)
    that Gaussian's expression parser treats as syntax.
    """

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 5:
        raise GaussianWriteError("Gaussian GIC label capacity must be an integer >= 5")
    definitions = tuple(_gaussian_definition(line) for line in gic_lines)
    bases = tuple(
        _gaussian_base_label(parsed[0])
        for parsed in definitions
        if parsed is not None
    )
    if len(bases) != len(set(bases)):
        raise GaussianWriteError("Gaussian GIC definitions contain duplicate labels")
    def is_transport_safe(label: str) -> bool:
        return bool(
            len(label) <= max_length
            and _GAUSSIAN_GIC_SAFE_LABEL_RE.fullmatch(label) is not None
        )

    occupied = {base for base in bases if is_transport_safe(base)}
    mapping: dict[str, str] = {}
    next_index = 1
    for base in bases:
        if is_transport_safe(base):
            continue
        while True:
            candidate = f"G{next_index:04d}"
            next_index += 1
            if candidate not in occupied:
                occupied.add(candidate)
                mapping[base] = candidate
                break
    if not mapping:
        return list(gic_lines)

    token_pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(
            re.escape(label) for label in sorted(mapping, key=len, reverse=True)
        ) + r")(?![A-Za-z0-9_])"
    )
    output: list[str] = []
    for line, parsed in zip(gic_lines, definitions, strict=True):
        if parsed is None:
            output.append(line)
            continue
        label, expression = parsed
        base = _gaussian_base_label(label)
        if base in mapping:
            label = label.replace(base, mapping[base], 1)
        expression = token_pattern.sub(lambda match: mapping[match.group(1)], expression)
        output.append(f"{label} = {expression}")
    return output


def _gaussian_line_with_inline_value(label: str, expression: str, value: float) -> str:
    label_text = label.strip()
    # Gaussian 16 does not accept Value together with Frozen/Inactive in the
    # option list.  Non-active coordinates are evaluated from the Cartesian
    # geometry by Gaussian, so their explicit initial value is unnecessary.
    if _GIC_NONACTIVE_RE.search(label_text):
        return f"{label_text} = {expression.strip()}"
    if "(" in label_text and label_text.endswith(")"):
        label_text = f"{label_text[:-1]},Value={_gaussian_float(value)})"
    else:
        label_text = f"{label_text}(Value={_gaussian_float(value)})"
    return f"{label_text} = {expression.strip()}"


def _gaussian_float(value: float) -> str:
    text = f"{value:.12g}"
    if "." not in text and "e" not in text.lower():
        text = f"{text}.0"
    return text


def _evaluate_gaussian_gic_expression(
    expression: str,
    coords: list[tuple[float, float, float]],
    values: dict[str, float],
) -> float:
    direct = re.fullmatch(r"\s*([RADHUXYZ])\s*\(([^()]*)\)\s*", expression, flags=re.IGNORECASE)
    if direct:
        args = [item.strip() for item in direct.group(2).split(",") if item.strip()]
        return _evaluate_direct_geometry_function(direct.group(1).upper(), args, coords)
    # Gaussian accepts square brackets as arithmetic grouping.  Python would
    # interpret them as list literals, so normalize only for local evaluation.
    text = (
        expression.replace("^", "**")
        .replace("[", "(")
        .replace("]", ")")
    )
    text = re.sub(
        r"(?P<mantissa>(?:\d+(?:\.\d*)?|\.\d+))[Dd](?P<exponent>[+-]?\d+)",
        r"\g<mantissa>E\g<exponent>",
        text,
    )
    previous = None
    while previous != text:
        previous = text
        text = _GIC_FUNCTION_RE.sub(
            lambda match: _evaluate_gaussian_gic_function(match, coords, values),
            text,
        )
    env = {
        **values,
        "sqrt": math.sqrt,
        "Sqrt": math.sqrt,
        "SQRT": math.sqrt,
        "exp": math.exp,
        "Exp": math.exp,
        "EXP": math.exp,
        "sin": math.sin,
        "Sin": math.sin,
        "SIN": math.sin,
        "cos": math.cos,
        "Cos": math.cos,
        "COS": math.cos,
        "tan": math.tan,
        "Tan": math.tan,
        "TAN": math.tan,
        "acos": math.acos,
        "Acos": math.acos,
        "ARCCOS": math.acos,
        "arccos": math.acos,
        "atan2": math.atan2,
        "Atan2": math.atan2,
        "ATAN2": math.atan2,
    }
    return float(eval(text, {"__builtins__": {}}, env))


def _evaluate_direct_geometry_function(
    name: str,
    args: list[str],
    coords: list[tuple[float, float, float]],
) -> float:
    value = _evaluate_geometry_function(name, args, coords)
    if name in {"A", "D", "U"}:
        return math.degrees(value)
    if name in {"R", "X", "Y", "Z"}:
        return value / ANGSTROM_TO_BOHR
    return value


def _evaluate_gaussian_gic_function(
    match: re.Match[str],
    coords: list[tuple[float, float, float]],
    values: dict[str, float],
) -> str:
    name = match.group(1)
    args = [item.strip() for item in match.group(2).split(",") if item.strip()]
    upper = name.upper()
    if upper in {"R", "A", "D", "H", "U", "X", "Y", "Z"}:
        value = _evaluate_geometry_function(upper, args, coords)
        return f"({value:.17g})"
    if upper in {"SQRT", "EXP", "SIN", "COS", "TAN", "ACOS", "ARCCOS", "ATAN2"}:
        return match.group(0)
    if name in values:
        return f"({values[name]:.17g})"
    return match.group(0)


def _evaluate_geometry_function(
    name: str,
    args: list[str],
    coords: list[tuple[float, float, float]],
) -> float:
    if name in {"X", "Y", "Z"}:
        if len(args) != 1:
            raise ValueError(f"{name} needs one atom index")
        atom = _atom_index(args[0], coords)
        axis = {"X": 0, "Y": 1, "Z": 2}[name]
        return coords[atom][axis] * ANGSTROM_TO_BOHR
    if name == "R":
        if len(args) != 2:
            raise ValueError("R needs two atom indices")
        first, second = (_atom_index(arg, coords) for arg in args)
        return _distance(coords[first], coords[second]) * ANGSTROM_TO_BOHR
    if name == "A":
        if len(args) != 3:
            raise ValueError("A needs three atom indices")
        first, center, third = (_atom_index(arg, coords) for arg in args)
        return _angle(coords[first], coords[center], coords[third])
    if name == "D":
        if len(args) != 4:
            raise ValueError("D needs four atom indices")
        i, j, k, l = (_atom_index(arg, coords) for arg in args)
        return _dihedral(coords[i], coords[j], coords[k], coords[l])
    if name == "H":
        if len(args) != 4:
            raise ValueError("H needs four atom indices")
        center, first, second, third = (_atom_index(arg, coords) for arg in args)
        plane1 = _unit(tuple(a - b for a, b in zip(coords[first], coords[center])))
        plane2 = _unit(tuple(a - b for a, b in zip(coords[second], coords[center])))
        normal = _cross(plane1, plane2)
        normal_norm = _norm(normal)
        if normal_norm <= 1.0e-14:
            raise FloatingPointError("height coordinate is singular")
        displacement = tuple(a - b for a, b in zip(coords[third], coords[center]))
        return _dot(displacement, normal) / normal_norm * ANGSTROM_TO_BOHR
    if name == "U":
        if len(args) != 4:
            raise ValueError("U needs four atom indices")
        center, first, second, third = (_atom_index(arg, coords) for arg in args)
        return _out_of_plane(coords[center], coords[first], coords[second], coords[third])
    raise ValueError(f"unsupported geometry function {name}")


def _atom_index(text: str, coords: list[tuple[float, float, float]]) -> int:
    index = int(text) - 1
    if index < 0 or index >= len(coords):
        raise ValueError(f"atom index out of range: {text}")
    return index


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _angle(
    first: tuple[float, float, float],
    center: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    u = tuple(a - b for a, b in zip(first, center))
    v = tuple(a - b for a, b in zip(third, center))
    norm = _norm(u) * _norm(v)
    if norm <= 1.0e-14:
        raise FloatingPointError("angle is singular")
    cosine = max(-1.0, min(1.0, _dot(u, v) / norm))
    return math.acos(cosine)


def _dihedral(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    b0 = tuple(a - b for a, b in zip(p0, p1))
    b1 = tuple(a - b for a, b in zip(p2, p1))
    b2 = tuple(a - b for a, b in zip(p3, p2))
    b1_unit = _unit(b1)
    v = tuple(a - _dot(b0, b1_unit) * b for a, b in zip(b0, b1_unit))
    w = tuple(a - _dot(b2, b1_unit) * b for a, b in zip(b2, b1_unit))
    x_value = _dot(v, w)
    y_value = _dot(_cross(b1_unit, v), w)
    return math.atan2(y_value, x_value)


def _out_of_plane(
    center: tuple[float, float, float],
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    # Gaussian U(center, plane1, plane2, out) uses plane1 and plane2
    # to orient the plane through the center; the fourth atom is the displaced
    # atom.  Keep this order explicit: normalizing a different cross product
    # changes the coordinate even though the unnormalized scalar triple
    # product is cyclically invariant.
    plane1 = _unit(tuple(a - b for a, b in zip(first, center)))
    plane2 = _unit(tuple(a - b for a, b in zip(second, center)))
    out = _unit(tuple(a - b for a, b in zip(third, center)))
    normal = _cross(plane1, plane2)
    normal_norm = _norm(normal)
    if normal_norm <= 1.0e-14:
        raise FloatingPointError("out-of-plane coordinate is singular")
    sine = max(-1.0, min(1.0, _dot(out, normal) / normal_norm))
    # The leading minus sign is Gaussian's orientation convention for U.
    return -math.asin(sine)


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _norm(vector)
    if norm <= 1.0e-14:
        raise FloatingPointError("zero-length vector")
    return tuple(value / norm for value in vector)


def _normalize_route(
    route: str,
    *,
    route_override: GaussianRouteOverride | None = None,
) -> str:
    validate_gaussian_route_policy(route, override=route_override)
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    _reject_calcfc(text)
    if not text.startswith("#"):
        text = f"# {text}"
    if _GICALLSYM_RE.search(text):
        raise GaussianWriteError(
            "MATRIX exports already symmetrized GICs; do not request Gaussian GICAllSym"
        )
    if _route_has_opt(text):
        text = _strip_readallgic_geom(text)
        normalized = _ensure_gaussian_output_pickett_unchecked(_ensure_readallgic_opt(text))
    else:
        normalized = _ensure_gaussian_output_pickett_unchecked(_ensure_readallgic_geom(text))
    validate_gaussian_route_transformation(route, normalized, override=route_override)
    return normalized


def _normalize_point_route(
    route: str,
    *,
    ensure_force: bool = True,
    route_override: GaussianRouteOverride | None = None,
) -> str:
    validate_gaussian_route_policy(route, override=route_override)
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    _reject_calcfc(text)
    if not text.startswith("#"):
        text = f"# {text}"
    if ensure_force and not re.search(r"\bforce\b", text, flags=re.IGNORECASE):
        text = f"{text} Force"
    normalized = _collapse_route(text)
    validate_gaussian_route_transformation(route, normalized, override=route_override)
    return normalized


def _reject_calcfc(route: str) -> None:
    """Enforce the MATRIX policy against full initial Hessian calculations."""

    if re.search(r"\bcalcfc\b", route, flags=re.IGNORECASE):
        from matrix_core import calculation_execution_directive

        directive = calculation_execution_directive(
            "gaussian.initial_hessian.owner_selected.v1"
        )
        raise GaussianWriteError(
            f"{directive.forbidden_keywords[0]} is forbidden by MATRIX policy; "
            f"only the owner-declared {directive.keyword} directive may be used"
        )


def _route_body(route: str) -> str:
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    if not text.startswith("#"):
        return text
    return re.sub(r"^#\s*[A-Za-z]?\s*", "", text).strip()


def _replace_checkpoint_suffix(path: str, suffix: str) -> str:
    text = path.strip()
    lower = text.lower()
    for ext in (".chk", ".fchk", ".fch"):
        if lower.endswith(ext):
            return text[: -len(ext)] + suffix
    return text + suffix


def _route_has_opt(route: str) -> bool:
    return bool(
        _OPT_PAREN_RE.search(route) or _OPT_VALUE_RE.search(route) or _OPT_BARE_RE.search(route)
    )


def _collapse_route(route: str) -> str:
    return " ".join(route.split())


def _strip_readallgic_geom(route: str) -> str:
    text = _GEOM_PAREN_RE.sub(_remove_readallgic_geom_parenthesized, route)
    return _GEOM_VALUE_RE.sub(_remove_readallgic_geom_value, text)


def _remove_readallgic_geom_parenthesized(match: re.Match[str]) -> str:
    options = [item.strip() for item in match.group("body").split(",") if item.strip()]
    kept = [option for option in options if not _READALLGIC_OPTION_RE.fullmatch(option)]
    if len(kept) == len(options):
        return match.group(0)
    if not kept:
        return ""
    return f"geom=({','.join(kept)})"


def _remove_readallgic_geom_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if value.lower() in {"gic", "readgic", "readallgic"}:
        return ""
    return match.group(0)


def _ensure_readallgic_opt(route: str) -> str:
    if _OPT_PAREN_RE.search(route):
        return _OPT_PAREN_RE.sub(_readallgic_opt_parenthesized, route, count=1)
    if match := _OPT_VALUE_RE.search(route):
        value = match.group("value")
        replacement = (
            "opt=readallgic"
            if value.lower() in {"gic", "readgic", "readallgic"}
            else f"opt=(readallgic,{value})"
        )
        return _OPT_VALUE_RE.sub(replacement, route, count=1)
    if _OPT_BARE_RE.search(route):
        return _OPT_BARE_RE.sub("opt=readallgic", route, count=1)
    return f"{route} opt=readallgic"


def _readallgic_opt_parenthesized(match: re.Match[str]) -> str:
    options = [item.strip() for item in match.group("body").split(",") if item.strip()]
    if not any(_READALLGIC_OPTION_RE.fullmatch(option) for option in options):
        options.insert(0, "readallgic")
    return f"opt=({','.join(options)})"


def _ensure_readallgic_geom(route: str) -> str:
    if _GEOM_PAREN_RE.search(route):
        return _GEOM_PAREN_RE.sub(_readallgic_geom_parenthesized, route, count=1)
    if match := _GEOM_VALUE_RE.search(route):
        value = match.group("value")
        replacement = (
            "geom=readallgic"
            if value.lower() in {"gic", "readgic", "readallgic"}
            else f"geom=(readallgic,{value})"
        )
        return _GEOM_VALUE_RE.sub(replacement, route, count=1)
    return f"{route} geom=readallgic"


def _readallgic_geom_parenthesized(match: re.Match[str]) -> str:
    options = [item.strip() for item in match.group("body").split(",") if item.strip()]
    if not any(_READALLGIC_OPTION_RE.fullmatch(option) for option in options):
        options.insert(0, "readallgic")
    return f"geom=({','.join(options)})"
