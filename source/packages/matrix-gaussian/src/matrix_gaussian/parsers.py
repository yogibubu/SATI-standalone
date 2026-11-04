from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


from matrix_chem import MolecularGeometry, atomic_mass
from matrix_chem.geometry_io import GeometryParseError, normalize_atom_symbol
from matrix_chem.topology.elements import atomic_number
from matrix_chem.zmatrix import parse_zmatrix_text, zmatrix_to_geometry

from .archive import archive_cartesian_hessian


ANGSTROM_TO_BOHR = 1.0 / 0.52917721092
SCF_RE = re.compile(r"SCF Done:\s+E\([^)]+\)\s+=\s+([-+]?\d+\.\d+)")
MP2_ENERGY_RE = re.compile(
    r"\bEUMP2\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    flags=re.IGNORECASE,
)
CCSD_T_ENERGY_RE = re.compile(
    r"^\s*CCSD\(T\)\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    flags=re.IGNORECASE | re.MULTILINE,
)
DOUBLE_HYBRID_ENERGY_RE = re.compile(
    r"^\s*E2\([^)]+\)\s*=.*?\bE\([^)]+\)\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    flags=re.IGNORECASE | re.MULTILINE,
)
EXCITED_TOTAL_ENERGY_RE = re.compile(
    r"Total Energy,\s*E\((?:EOM-CCSD|TD-HF/TD-DFT|CIS/TDA|CIS\(D\))\)\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    flags=re.IGNORECASE,
)
ONIOM_ENERGY_RE = re.compile(
    r"ONIOM:\s+extrapolated energy\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    flags=re.IGNORECASE,
)
MM_ENERGY_RE = re.compile(
    r"\bEnergy=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)"
)
NORMAL_TERMINATION = "Normal termination of Gaussian"
STANDARD_ORIENTATION = "Standard orientation:"
INPUT_ORIENTATION = "Input orientation:"
POINT_GROUP_RE = re.compile(r"Full point group\s+(\S+)", flags=re.IGNORECASE)
RANK_RE = re.compile(r"NTRed=\s*(\d+).*?NRank=\s*(\d+)", flags=re.IGNORECASE)
NIMAG_RE = re.compile(r"N\s*Imag\s*=\s*(\d+)", flags=re.IGNORECASE)
READALLGIC_RE = re.compile(r"\breadallgic\b", flags=re.IGNORECASE)
OPTIMIZATION_STEP_RE = re.compile(
    r"^\s*Step\s+number\s+\d+\s+out\s+of\s+a\s+maximum\s+of\s+\d+",
    flags=re.IGNORECASE | re.MULTILINE,
)
PARAMETER_LINE_RE = re.compile(r"^\s*!\s*(?P<label>\S+)\s+(?P<body>.*?)\s*!\s*$")
THERMOCHEMISTRY_MASS_RE = re.compile(
    r"Atom\s+(?P<index>\d+)\s+has atomic number\s+(?P<number>\d+)\s+and mass\s+"
    r"(?P<mass>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)"
)


@dataclass(frozen=True)
class GaussianLogSummary:
    path: Path
    normal_termination: bool
    scf_energies_hartree: tuple[float, ...]
    standard_orientation_count: int
    input_orientation_count: int
    scan_marker_count: int
    puckering_marker_count: int
    optimization_steps: int = 0
    frequencies_cm: tuple[float, ...] = ()
    last_orientation: MolecularGeometry | None = None


@dataclass(frozen=True)
class GaussianPointResult:
    """Single-point energy and optional Cartesian gradient from a Gaussian log."""

    path: Path
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None
    normal_termination: bool
    model: str = ""


@dataclass(frozen=True)
class GaussianReadAllGICLogCheck:
    path: Path
    normal_termination_count: int
    route_has_readallgic: bool
    point_groups: tuple[str, ...]
    ranks: tuple[tuple[int, int], ...]
    labels: tuple[str, ...]
    active_labels: tuple[str, ...]
    frozen_labels: tuple[str, ...]
    frequency_count: int
    imaginary_frequency_count: int
    n_imag: int | None
    optimization_completed: bool
    stationary_point: bool
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class GaussianLogHessianPromotion:
    xyzin: Path
    log_path: Path
    wrote_cartesian_hessian: bool
    wrote_normal_modes: bool = False


@dataclass(frozen=True)
class GaussianLogNormalModes:
    path: Path
    frequencies_cm: np.ndarray
    modes: np.ndarray
    source_frame: str
    irreps: tuple[str, ...] = ()
    rotated_to_archive_axes: bool = False


@dataclass(frozen=True)
class _GaussianInputBlock:
    route_lines: tuple[str, ...]
    title: str
    charge: int
    multiplicity: int
    geometry_lines: tuple[str, ...]
    tail_lines: tuple[str, ...]


def read_gaussian_input(path: Path) -> MolecularGeometry:
    """Read a Gaussian input file with Cartesian or Z-matrix geometry."""
    target = Path(path)
    block = _read_gaussian_input_block(target)
    if _geometry_looks_cartesian(block.geometry_lines):
        return _geometry_from_cartesian_block(target, block)
    return _geometry_from_zmatrix_block(target, block)


def read_gaussian_cartesian_input(path: Path) -> MolecularGeometry:
    target = Path(path)
    block = _read_gaussian_input_block(target)
    if not _geometry_looks_cartesian(block.geometry_lines) and _route_requests_zmatrix(
        block.route_lines
    ):
        return read_gaussian_zmatrix_input(target)
    return _geometry_from_cartesian_block(target, block)


def read_gaussian_zmatrix_input(path: Path) -> MolecularGeometry:
    target = Path(path)
    block = _read_gaussian_input_block(target)
    if _geometry_looks_cartesian(block.geometry_lines):
        raise GeometryParseError("Gaussian input contains Cartesian coordinates, not a Z-matrix")
    return _geometry_from_zmatrix_block(target, block)


def _read_gaussian_input_block(path: Path) -> _GaussianInputBlock:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    route_end = _route_end(lines)
    if route_end is None:
        raise GeometryParseError("Gaussian input needs a route section starting with #")
    route_lines = _route_lines(lines)
    idx = _next_nonblank(lines, route_end)
    title_start = idx
    while idx < len(lines) and lines[idx].strip():
        idx += 1
    title = " ".join(line.strip() for line in lines[title_start:idx] if line.strip())
    idx = _next_nonblank(lines, idx)
    if idx >= len(lines) or not _is_charge_multiplicity(lines[idx]):
        raise GeometryParseError("Gaussian input needs charge and multiplicity before coordinates")
    charge, multiplicity = (int(value) for value in lines[idx].split()[:2])
    idx = _next_nonblank(lines, idx + 1)
    geometry_start = idx
    while idx < len(lines) and lines[idx].strip():
        idx += 1
    geometry_lines = tuple(lines[geometry_start:idx])
    if not geometry_lines:
        raise GeometryParseError("Gaussian input contains no geometry block")
    tail_lines = tuple(lines[idx + 1 :]) if idx < len(lines) else ()
    return _GaussianInputBlock(
        route_lines=route_lines,
        title=title or target.stem,
        charge=charge,
        multiplicity=multiplicity,
        geometry_lines=geometry_lines,
        tail_lines=tail_lines,
    )


def _geometry_from_cartesian_block(path: Path, block: _GaussianInputBlock) -> MolecularGeometry:
    atoms: list[str] = []
    coords: list[list[float]] = []
    for line in block.geometry_lines:
        atom, xyz = _parse_cartesian_line(line)
        atoms.append(atom)
        coords.append(xyz)
    if not atoms:
        raise GeometryParseError("Gaussian input contains no Cartesian coordinate block")
    return MolecularGeometry(
        atoms=tuple(atoms),
        coordinates_angstrom=np.asarray(coords, dtype=float),
        comment=block.title or path.stem,
        source_format="gaussian_cartesian_input",
        source_path=path,
        charge=block.charge,
        multiplicity=block.multiplicity,
        fixed_parameters=_modredundant_fixed_patterns(block.tail_lines),
        metadata={"route": block.route_lines},
    )


def _geometry_from_zmatrix_block(path: Path, block: _GaussianInputBlock) -> MolecularGeometry:
    body_lines = [
        f"{block.charge} {block.multiplicity}",
        *block.geometry_lines,
    ]
    variable_lines = _leading_zmatrix_variable_lines(block.tail_lines)
    if variable_lines:
        body_lines.append("")
        body_lines.extend(variable_lines)
    zmat = parse_zmatrix_text("\n".join(body_lines), title=block.title)
    return zmatrix_to_geometry(zmat, source_path=path, source_format="gaussian_zmatrix_input")


def summarize_gaussian_log(path: Path) -> GaussianLogSummary:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    scf = tuple(float(match.group(1)) for match in SCF_RE.finditer(text))
    orientation = _parse_last_orientation(lines, source_path=target)
    return GaussianLogSummary(
        path=target,
        normal_termination=NORMAL_TERMINATION in text,
        scf_energies_hartree=scf,
        standard_orientation_count=text.count(STANDARD_ORIENTATION),
        input_orientation_count=text.count(INPUT_ORIENTATION),
        scan_marker_count=text.lower().count("scan"),
        puckering_marker_count=text.count("QPck") + text.count("PhiP") + text.count("RPck"),
        optimization_steps=len(OPTIMIZATION_STEP_RE.findall(text)),
        frequencies_cm=tuple(_parse_frequencies(text)),
        last_orientation=orientation,
    )


def read_gaussian_point_result(
    path: Path,
    *,
    require_gradient: bool = True,
) -> GaussianPointResult:
    """Read the final energy and optional Cartesian gradient from a Gaussian log.

    Gaussian reports Cartesian forces; the returned array is the energy
    gradient, i.e. the negative of the reported force vector.
    """

    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    energy = _parse_last_point_energy(text)
    forces = _parse_last_force_block(lines)
    if energy is None:
        raise GeometryParseError(f"Gaussian log contains no parseable point energy: {target}")
    if forces is None and require_gradient:
        raise GeometryParseError(f"Gaussian log contains no Cartesian force block: {target}")
    return GaussianPointResult(
        path=target,
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=(None if forces is None else -forces.reshape(-1)),
        normal_termination=NORMAL_TERMINATION in text,
        model=_parse_archive_model(text),
    )


def check_gaussian_readallgic_log(
    path: Path,
    *,
    expected_point_group: str | None = None,
    expected_rank: int | None = None,
    expected_frozen_labels: tuple[str, ...] = (),
    expected_active_labels: tuple[str, ...] = (),
    require_frequency: bool = False,
    require_no_imaginary: bool = False,
) -> GaussianReadAllGICLogCheck:
    """Validate the Gaussian-side result of a MATRIX ReadAllGIC input."""
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    summary = summarize_gaussian_log(target)
    normal_termination_count = text.count(NORMAL_TERMINATION)
    labels, frozen_labels = _parse_parameter_labels(text)
    frozen = set(frozen_labels)
    active_labels = tuple(label for label in labels if label not in frozen)
    point_groups = tuple(match.group(1).upper() for match in POINT_GROUP_RE.finditer(text))
    ranks = tuple((int(match.group(1)), int(match.group(2))) for match in RANK_RE.finditer(text))
    n_imag = _parse_last_nimag(text)
    imaginary_frequency_count = sum(1 for frequency in summary.frequencies_cm if frequency < 0.0)

    errors: list[str] = []
    if normal_termination_count == 0:
        errors.append("Gaussian log has no normal termination")
    if not READALLGIC_RE.search(text):
        errors.append("Gaussian log does not contain readallgic")
    if expected_point_group is not None:
        expected = expected_point_group.upper()
        if expected not in point_groups:
            errors.append(f"expected point group {expected}, found {point_groups or 'none'}")
    if expected_rank is not None and (expected_rank, expected_rank) not in ranks:
        errors.append(f"expected NTRed/NRank {expected_rank}, found {ranks or 'none'}")
    missing_frozen = tuple(label for label in expected_frozen_labels if label not in frozen)
    if missing_frozen:
        errors.append(f"missing frozen labels: {', '.join(missing_frozen)}")
    missing_active = tuple(label for label in expected_active_labels if label not in labels)
    if missing_active:
        errors.append(f"missing active labels: {', '.join(missing_active)}")
    frozen_active = tuple(label for label in expected_active_labels if label in frozen)
    if frozen_active:
        errors.append(f"expected active labels are frozen: {', '.join(frozen_active)}")
    if require_frequency and not summary.frequencies_cm:
        errors.append("Gaussian log contains no vibrational frequencies")
    if require_no_imaginary:
        if n_imag is not None and n_imag != 0:
            errors.append(f"expected NImag=0, found NImag={n_imag}")
        if n_imag is None and imaginary_frequency_count:
            errors.append(f"expected no imaginary frequencies, found {imaginary_frequency_count}")

    return GaussianReadAllGICLogCheck(
        path=target,
        normal_termination_count=normal_termination_count,
        route_has_readallgic=READALLGIC_RE.search(text) is not None,
        point_groups=point_groups,
        ranks=ranks,
        labels=labels,
        active_labels=active_labels,
        frozen_labels=frozen_labels,
        frequency_count=len(summary.frequencies_cm),
        imaginary_frequency_count=imaginary_frequency_count,
        n_imag=n_imag,
        optimization_completed="Optimization completed" in text,
        stationary_point="Stationary point found" in text,
        errors=tuple(errors),
    )


def _parse_last_point_energy(text: str) -> float | None:
    oniom = [_float_token(match.group(1)) for match in ONIOM_ENERGY_RE.finditer(text)]
    if oniom:
        return oniom[-1]
    excited = [
        _float_token(match.group(1))
        for match in EXCITED_TOTAL_ENERGY_RE.finditer(text)
    ]
    if excited:
        return excited[-1]
    double_hybrid = [
        _float_token(match.group(1))
        for match in DOUBLE_HYBRID_ENERGY_RE.finditer(text)
    ]
    if double_hybrid:
        return double_hybrid[-1]
    ccsd_t = [_float_token(match.group(1)) for match in CCSD_T_ENERGY_RE.finditer(text)]
    if ccsd_t:
        return ccsd_t[-1]
    mp2 = [_float_token(match.group(1)) for match in MP2_ENERGY_RE.finditer(text)]
    if mp2:
        return mp2[-1]
    scf = [float(match.group(1)) for match in SCF_RE.finditer(text)]
    if scf:
        return scf[-1]
    mm = [_float_token(match.group(1)) for match in MM_ENERGY_RE.finditer(text)]
    return mm[-1] if mm else None


def _parse_last_force_block(lines: Sequence[str]) -> np.ndarray | None:
    starts = [
        index
        for index, line in enumerate(lines)
        if "Forces (Hartrees/Bohr)" in line
    ]
    for start in reversed(starts):
        values: list[list[float]] = []
        index = start + 1
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if stripped.startswith("---") or not stripped:
                index += 1
                continue
            parts = stripped.split()
            if len(parts) >= 5 and parts[0].isdigit() and parts[1].isdigit():
                try:
                    values.append([float(parts[2]), float(parts[3]), float(parts[4])])
                except ValueError:
                    break
                index += 1
                continue
            if values:
                break
            index += 1
        if values:
            return np.asarray(values, dtype=float)
    return None


def _parse_archive_model(text: str) -> str:
    for line in reversed(text.splitlines()):
        if "\\Force\\" not in line:
            continue
        parts = [part for part in line.split("\\") if part]
        try:
            idx = parts.index("Force")
        except ValueError:
            continue
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def _float_token(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def read_gaussian_log_geometry(path: Path) -> MolecularGeometry:
    """Read the final Gaussian log geometry as the shared MATRIX geometry."""
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    geometry = (
        _parse_last_archive_geometry(text, source_path=target)
        or summarize_gaussian_log(target).last_orientation
    )
    if geometry is None:
        raise GeometryParseError("Gaussian log contains no readable orientation block")
    return geometry


def read_gaussian_log_cartesian_hessian(path: Path) -> np.ndarray:
    """Read the last Gaussian Cartesian force-constant matrix."""
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    summary = summarize_gaussian_log(target)
    geometry = _parse_last_archive_geometry(text, source_path=target) or summary.last_orientation
    if geometry is None:
        raise GeometryParseError("Gaussian log contains no readable orientation block")
    return _parse_last_cartesian_hessian(
        text.splitlines(), size=3 * geometry.natoms, archive_text=text
    )


def read_gaussian_log_dipole_derivatives(path: Path) -> np.ndarray:
    """Read the dipole Jacobian (3, 3N), au/bohr, in original archive axes."""
    from .archive import archive_cartesian_dipole_derivatives

    target = Path(path)
    geometry = read_gaussian_log_geometry(target)
    return archive_cartesian_dipole_derivatives(
        target.read_text(encoding="utf-8", errors="replace"), size=3 * geometry.natoms
    )


def read_gaussian_log_normal_modes(path: Path) -> GaussianLogNormalModes:
    """Read printed Gaussian harmonic normal coordinates, when present."""
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    summary = summarize_gaussian_log(target)
    geometry = _parse_last_archive_geometry(text, source_path=target) or summary.last_orientation
    if geometry is None:
        raise GeometryParseError("Gaussian log contains no readable orientation block")
    frequencies, modes, irreps = _parse_last_normal_coordinate_table(
        lines, natoms=geometry.natoms
    )
    source_frame = "gaussian-log-normal-coordinate-frame"
    rotated = False
    archive = _parse_last_archive_geometry(text, source_path=target)
    if archive is not None and summary.last_orientation is not None:
        rotation = _orthogonal_map_between_geometries(summary.last_orientation, archive)
        if rotation is not None:
            modes = (modes.reshape((modes.shape[0], geometry.natoms, 3)) @ rotation).reshape(
                modes.shape
            )
            source_frame = "gaussian-log-archive-original-axes"
            rotated = True
    return GaussianLogNormalModes(
        path=target,
        frequencies_cm=frequencies,
        modes=modes,
        source_frame=source_frame,
        irreps=irreps,
        rotated_to_archive_axes=rotated,
    )


def hessian_input_from_gaussian_log(path: Path):
    """Gaussian log adapter: return the canonical MATRIX Hessian input for GF."""
    from matrix_qm import HessianInput

    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    summary = summarize_gaussian_log(target)
    geometry = _parse_last_archive_geometry(text, source_path=target) or summary.last_orientation
    if geometry is None:
        raise GeometryParseError("Gaussian log contains no readable orientation block")
    numbers = np.asarray(_atomic_numbers_from_geometry(geometry), dtype=int)
    masses = np.asarray(_masses_from_gaussian_log(text, numbers), dtype=float)
    hessian = _parse_last_cartesian_hessian(
        text.splitlines(), size=3 * geometry.natoms, archive_text=text
    )
    data = HessianInput(
        atomic_numbers=numbers,
        cartesian_coordinates_bohr=np.asarray(geometry.coordinates_angstrom, dtype=float)
        * ANGSTROM_TO_BOHR,
        masses_amu=masses,
        cartesian_hessian=hessian,
        harmonic_frequencies_cm=np.asarray(summary.frequencies_cm, dtype=float),
        source="gaussian-log",
    )
    data.validate()
    return data


def promote_gaussian_log_hessian_to_xyzin(
    log_path: Path | str,
    xyzin: Path | str,
    *,
    write_normal_modes: bool = True,
) -> GaussianLogHessianPromotion:
    """Promote a Gaussian log Cartesian Hessian into shared MATRIX xyzin sections."""
    from .topology import read_gaussian_topology

    from matrix_qm import cartesian_hessian_section_from_hessian_input
    from matrix_qm import normal_modes_section_from_arrays
    from matrix_qm import write_cartesian_hessian_section
    from matrix_qm import write_normal_modes_section

    source = Path(log_path)
    target = Path(xyzin)
    data = hessian_input_from_gaussian_log(source)
    topology = read_gaussian_topology(source)
    cm5 = tuple(
        float(topology.cm5_charges[index])
        for index in range(1, len(data.atomic_numbers) + 1)
        if index in topology.cm5_charges
    )
    if len(cm5) != len(data.atomic_numbers):
        cm5 = ()
    write_cartesian_hessian_section(
        target,
        cartesian_hessian_section_from_hessian_input(
            data,
            source=f"gaussian-log {source}",
            atomic_charges=cm5,
            charge_model="CM5" if cm5 else "",
        ),
    )
    wrote_normal_modes = False
    if write_normal_modes:
        from matrix_qm import cartesian_normal_modes_from_hessian

        # Printed Gaussian normal coordinates are rounded for display and are
        # not a sufficiently accurate source for the strict mass-weighted
        # protocol.  A log promotion already requires the Cartesian Hessian,
        # so derive the canonical modes from that unrounded tensor instead.
        canonical_modes = cartesian_normal_modes_from_hessian(
            data.cartesian_hessian,
            data.masses_amu,
            data.cartesian_coordinates_bohr,
            source=f"gaussian-log Hessian {source}",
        )
        write_normal_modes_section(
            target,
            normal_modes_section_from_arrays(
                canonical_modes.frequencies_cm,
                canonical_modes.mass_weighted_modes,
                source=f"gaussian-log Hessian {source}; matrix.link.mass_weighted_normal_modes.v1",
                coordinate_count=data.cartesian_coordinates_bohr.size,
            ),
        )
        wrote_normal_modes = True
    return GaussianLogHessianPromotion(
        xyzin=target,
        log_path=source,
        wrote_cartesian_hessian=True,
        wrote_normal_modes=wrote_normal_modes,
    )


def _route_end(lines: list[str]) -> int | None:
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().startswith("#"):
            while idx < len(lines) and lines[idx].strip():
                idx += 1
            return idx
        idx += 1
    return None


def _route_lines(lines: list[str]) -> tuple[str, ...]:
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().startswith("#"):
            route: list[str] = []
            while idx < len(lines) and lines[idx].strip():
                route.append(lines[idx].strip())
                idx += 1
            return tuple(route)
        idx += 1
    return ()


def _route_requests_zmatrix(route_lines: tuple[str, ...]) -> bool:
    route = " ".join(route_lines).lower()
    return "zmat" in route or "z-matrix" in route


def _geometry_looks_cartesian(lines: tuple[str, ...]) -> bool:
    if not lines:
        return False
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            return False
        try:
            normalize_atom_symbol(parts[0])
            _float_token(parts[1])
            _float_token(parts[2])
            _float_token(parts[3])
        except (GeometryParseError, ValueError):
            return False
    return True


def _leading_zmatrix_variable_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    variables: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "%")):
            break
        if _is_zmatrix_variable_header(stripped) or _is_zmatrix_variable_line(stripped):
            variables.append(raw)
            continue
        break
    return tuple(variables)


def _is_zmatrix_variable_header(line: str) -> bool:
    return line.lower().rstrip(":") in {
        "variables",
        "variable",
        "constants",
        "constant",
        "parameters",
        "parameter",
    }


def _is_zmatrix_variable_line(line: str) -> bool:
    if "=" in line:
        name, value = line.split("=", 1)
    else:
        parts = line.split()
        if len(parts) != 2:
            return False
        name, value = parts
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name.strip()):
        return False
    try:
        _float_token(value)
    except ValueError:
        return False
    return True


def _next_nonblank(lines: list[str], idx: int) -> int:
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def _is_charge_multiplicity(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


def _parse_cartesian_line(line: str) -> tuple[str, list[float]]:
    parts = line.split()
    if len(parts) < 4:
        raise GeometryParseError(f"invalid Gaussian Cartesian line: {line}")
    try:
        coords = [_float_token(parts[1]), _float_token(parts[2]), _float_token(parts[3])]
    except ValueError as exc:
        raise GeometryParseError(f"invalid Gaussian Cartesian coordinates: {line}") from exc
    return normalize_atom_symbol(parts[0]), coords


def _float_token(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _modredundant_fixed_patterns(lines: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    patterns: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        kind = parts[0].upper()
        expected = {"B": 2, "A": 3, "D": 4, "O": 4, "L": 3}.get(kind)
        if expected is None:
            continue
        atoms: list[int] = []
        actions: list[str] = []
        for token in parts[1:]:
            try:
                atoms.append(int(token))
            except ValueError:
                actions.append(token.upper())
        if len(atoms) < expected or "F" not in actions:
            continue
        pattern = f"{kind}({','.join(str(atom) for atom in atoms[:expected])})"
        if pattern not in seen:
            patterns.append(pattern)
            seen.add(pattern)
    return tuple(patterns)


def _parse_frequencies(text: str) -> list[float]:
    values: list[float] = []
    for line in text.splitlines():
        if "Frequencies --" not in line:
            continue
        for token in line.split("--", 1)[1].split():
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


def _parse_parameter_labels(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    labels: list[str] = []
    frozen_labels: list[str] = []
    seen: set[str] = set()
    seen_frozen: set[str] = set()
    for line in text.splitlines():
        match = PARAMETER_LINE_RE.match(line)
        if not match:
            continue
        label = match.group("label")
        if not _looks_like_parameter_label(label):
            continue
        body = match.group("body")
        if label not in seen:
            labels.append(label)
            seen.add(label)
        if "frozen" in body.lower() and label not in seen_frozen:
            frozen_labels.append(label)
            seen_frozen.add(label)
    return tuple(labels), tuple(frozen_labels)


def _looks_like_parameter_label(label: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) and re.search(r"\d", label))


def _parse_last_nimag(text: str) -> int | None:
    matches = tuple(NIMAG_RE.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group(1))


def _parse_last_cartesian_hessian(
    lines: list[str], *, size: int, archive_text: str | None = None
) -> np.ndarray:
    starts = [
        idx for idx, line in enumerate(lines) if "Force constants in Cartesian coordinates" in line
    ]
    if starts:
        return _parse_cartesian_hessian_block(lines[starts[-1] + 1 :], size=size)
    if archive_text is not None:
        try:
            return archive_cartesian_hessian(archive_text, size=size)
        except ValueError as exc:
            raise GeometryParseError(
                "Gaussian log contains no readable Cartesian force-constant matrix"
            ) from exc
    raise GeometryParseError("Gaussian log contains no Cartesian force-constant matrix")


def _parse_cartesian_hessian_block(lines: list[str], *, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    filled = np.zeros((size, size), dtype=bool)
    required = np.tril(np.ones((size, size), dtype=bool))
    columns: list[int] = []
    for raw in lines:
        if "Force constants in internal coordinates" in raw or raw.lstrip().startswith("FormGI"):
            break
        parts = raw.split()
        if not parts:
            continue
        if all(part.isdigit() for part in parts):
            columns = [int(part) - 1 for part in parts]
            continue
        if not columns:
            continue
        try:
            row = int(parts[0]) - 1
        except ValueError:
            continue
        if row < 0 or row >= size:
            continue
        values = [_float_token(token) for token in parts[1:]]
        for column, value in zip(columns, values):
            if column < 0 or column >= size:
                continue
            matrix[row, column] = value
            matrix[column, row] = value
            filled[row, column] = True
            filled[column, row] = True
        if np.all(filled[required]):
            break
    if not np.all(filled[required]):
        missing = np.argwhere(required & ~filled)
        row, column = missing[0]
        raise GeometryParseError(
            "Gaussian Cartesian force-constant matrix is incomplete at "
            f"row {row + 1}, column {column + 1}"
        )
    return 0.5 * (matrix + matrix.T)


def _parse_last_normal_coordinate_table(
    lines: list[str],
    *,
    natoms: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    starts = [idx for idx, line in enumerate(lines) if "and normal coordinates:" in line]
    if not starts:
        raise GeometryParseError("Gaussian log contains no printed normal-coordinate table")
    start = starts[-1] + 1
    frequencies: list[float] = []
    modes: list[np.ndarray] = []
    irreps: list[str] = []
    idx = start
    while idx < len(lines):
        line = lines[idx]
        if line.strip().startswith("- Thermochemistry -"):
            break
        if "Frequencies --" not in line:
            idx += 1
            continue
        block_frequencies = [_float_token(token) for token in line.split("--", 1)[1].split()]
        block_width = len(block_frequencies)
        if block_width == 0:
            idx += 1
            continue
        symmetry_tokens = lines[idx - 1].split() if idx > start else []
        if len(symmetry_tokens) == block_width and not all(
            token.isdigit() for token in symmetry_tokens
        ):
            irreps.extend(token.replace('"', "''") for token in symmetry_tokens)
        else:
            irreps.extend("" for _ in range(block_width))
        header = idx + 1
        while header < len(lines) and not lines[header].lstrip().startswith("Atom  AN"):
            if "Frequencies --" in lines[header] or lines[header].strip().startswith(
                "- Thermochemistry -"
            ):
                raise GeometryParseError("Gaussian normal-coordinate block has no Atom header")
            header += 1
        if header >= len(lines):
            break
        block = np.zeros((block_width, natoms, 3), dtype=float)
        for atom_index in range(natoms):
            row_index = header + 1 + atom_index
            if row_index >= len(lines):
                raise GeometryParseError("Gaussian normal-coordinate block is incomplete")
            parts = lines[row_index].split()
            if len(parts) < 2 + 3 * block_width:
                raise GeometryParseError(
                    "Gaussian normal-coordinate row has too few Cartesian components"
                )
            try:
                int(parts[0])
                int(parts[1])
            except ValueError as exc:
                raise GeometryParseError("Gaussian normal-coordinate row is malformed") from exc
            values = [_float_token(token) for token in parts[2 : 2 + 3 * block_width]]
            for mode_index in range(block_width):
                block[mode_index, atom_index, :] = values[3 * mode_index : 3 * mode_index + 3]
        frequencies.extend(block_frequencies)
        modes.extend(block[mode_index].reshape(-1) for mode_index in range(block_width))
        idx = header + 1 + natoms
    if not modes:
        raise GeometryParseError("Gaussian log contains no readable normal coordinates")
    return np.asarray(frequencies, dtype=float), np.vstack(modes), tuple(irreps)


def _orthogonal_map_between_geometries(
    source: MolecularGeometry,
    target: MolecularGeometry,
    *,
    max_rms_angstrom: float = 1.0e-3,
) -> np.ndarray | None:
    if source.natoms != target.natoms or source.atoms != target.atoms:
        return None
    source_coords = np.asarray(source.coordinates_angstrom, dtype=float)
    target_coords = np.asarray(target.coordinates_angstrom, dtype=float)
    if source_coords.shape != target_coords.shape:
        return None
    source_centered = source_coords - np.mean(source_coords, axis=0)
    target_centered = target_coords - np.mean(target_coords, axis=0)
    try:
        u, _singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vt
    residual = source_centered @ rotation - target_centered
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if rms > max_rms_angstrom:
        return None
    return rotation


def _atomic_numbers_from_geometry(geometry: MolecularGeometry) -> tuple[int, ...]:
    numbers: list[int] = []
    for atom in geometry.atoms:
        number = atomic_number(atom)
        if number is None:
            raise GeometryParseError(f"unsupported Gaussian atom label: {atom}")
        numbers.append(number)
    return tuple(numbers)


def _masses_from_gaussian_log(text: str, atomic_numbers: np.ndarray) -> tuple[float, ...]:
    masses_by_index: dict[int, float] = {}
    numbers_by_index: dict[int, int] = {}
    for match in THERMOCHEMISTRY_MASS_RE.finditer(text):
        index = int(match.group("index"))
        masses_by_index[index] = _float_token(match.group("mass"))
        numbers_by_index[index] = int(match.group("number"))
    if len(masses_by_index) >= len(atomic_numbers):
        masses = []
        for index, number in enumerate(atomic_numbers, start=1):
            if numbers_by_index.get(index) != int(number):
                break
            masses.append(masses_by_index[index])
        if len(masses) == len(atomic_numbers):
            return tuple(masses)
    return tuple(float(atomic_mass(int(number))) for number in atomic_numbers)


def _parse_last_archive_geometry(text: str, *, source_path: Path) -> MolecularGeometry | None:
    starts = [match.start() for match in re.finditer(r"1\\1\\GINC", text)]
    for start in reversed(starts):
        end = text.find(r"\@", start)
        if end < 0:
            continue
        compact = re.sub(r"\s+", "", text[start : end + 2])
        match = re.search(
            r"\\\\(?P<charge>-?\d+),(?P<multiplicity>\d+)\\(?P<body>.*?)(?=\\\\Version=)",
            compact,
        )
        if not match:
            continue
        atoms: list[str] = []
        coords: list[list[float]] = []
        for token in match.group("body").split("\\"):
            parts = token.split(",")
            if len(parts) < 4:
                continue
            try:
                atom = normalize_atom_symbol(parts[0])
                # Gaussian may insert an integer atom-type field after the
                # element (for example ``C,0,x,y,z``).  Coordinates are always
                # the final three fields of an archive geometry record.
                xyz = [_float_token(value) for value in parts[-3:]]
            except (GeometryParseError, ValueError):
                continue
            atoms.append(atom)
            coords.append(xyz)
        if atoms:
            return MolecularGeometry(
                atoms=tuple(atoms),
                coordinates_angstrom=np.asarray(coords, dtype=float),
                comment="Gaussian archive geometry",
                source_format="gaussian_log_archive",
                source_path=source_path,
                charge=int(match.group("charge")),
                multiplicity=int(match.group("multiplicity")),
                metadata={"orientation": "archive_original_axes"},
            )
    return None


def _parse_last_orientation(lines: list[str], *, source_path: Path) -> MolecularGeometry | None:
    start = -1
    orientation_kind = ""
    for idx, line in enumerate(lines):
        if STANDARD_ORIENTATION in line or INPUT_ORIENTATION in line:
            start = idx
            orientation_kind = line.strip().rstrip(":")
    if start < 0:
        return None
    atoms: list[str] = []
    coords: list[list[float]] = []
    dash_count = 0
    for line in lines[start + 1 :]:
        if set(line.strip()) == {"-"}:
            dash_count += 1
            if dash_count >= 3:
                break
            continue
        if dash_count < 2:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            atom = normalize_atom_symbol(parts[1])
            xyz = [float(parts[3]), float(parts[4]), float(parts[5])]
        except ValueError:
            continue
        atoms.append(atom)
        coords.append(xyz)
    if not atoms:
        return None
    return MolecularGeometry(
        atoms=tuple(atoms),
        coordinates_angstrom=np.asarray(coords, dtype=float),
        comment=f"Gaussian {orientation_kind}",
        source_format="gaussian_log_orientation",
        source_path=source_path,
        metadata={"orientation": orientation_kind},
    )
