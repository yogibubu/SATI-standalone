from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_gaussian import ensure_gaussian_output_pickett

from .modify_geom import read_xyz

DEFAULT_GAUSSIAN_ROUTE = (
    "#p b3lyp/def2tzvp empiricaldispersion=gd3bj "
    "opt=(addgic,calchffc,noeigentest,maxcycles=200) "
    "int=ultrafine scf=xqc output=pickett"
)


@dataclass(frozen=True)
class PuckeringState:
    qc: float
    qs: float
    q: float
    phi_deg: float
    modes: tuple["PuckeringMode", ...] = ()


@dataclass(frozen=True)
class PuckeringMode:
    harmonic: int
    qc: float
    qs: float
    q: float
    phi_deg: float


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.atan2(y, x)


def parse_ring_indices(text: str, natoms: int) -> list[int]:
    indices = [int(item.strip()) - 1 for item in text.split(",") if item.strip()]
    if len(indices) < 4:
        raise ValueError("--ring must contain at least 4 comma-separated one-based atom indices")
    if min(indices) < 0 or max(indices) >= natoms:
        raise ValueError("--ring contains an atom index outside the molecular geometry")
    if len(set(indices)) != len(indices):
        raise ValueError("--ring contains duplicated atom indices")
    return canonical_ring_indices(indices)


def _ring_atom_priority(
    atom_index: int,
    ring_set: set[int],
    atomic_numbers: list[int] | None,
    adjacency: list[set[int]] | None,
) -> tuple[float, ...]:
    if atomic_numbers is None:
        z_atom = 0
        degree = 0
        exocyclic: list[int] = []
    else:
        z_atom = int(atomic_numbers[atom_index])
        neighbours = adjacency[atom_index] if adjacency is not None else set()
        degree = len(neighbours)
        exocyclic = sorted(
            (int(atomic_numbers[nbr]) for nbr in neighbours if nbr not in ring_set),
            reverse=True,
        )
    # CIP/Prelog priority starts from atomic number.  The input atom index is
    # only the final deterministic tie-break for topologically equivalent atoms.
    return (
        float(z_atom),
        float(degree),
        *[float(value) for value in exocyclic],
        float(-atom_index),
    )


def prelog_canonical_ring_indices(
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> list[int]:
    """Return a deterministic Prelog-first cyclic numbering for a ring.

    All rotations and both cyclic directions are compared lexicographically by
    local CIP/Prelog-like atom priorities.  Atomic number is primary; local
    graph degree and exocyclic substituent atomic numbers refine the choice when
    topology is available; the input atom index is only a final tie-break.
    """
    indices = list(ring_indices)
    if len(indices) < 4:
        raise ValueError("A canonical ring needs at least 4 atoms")
    if len(set(indices)) != len(indices):
        raise ValueError("A canonical ring cannot contain duplicated atoms")
    n = len(indices)
    ring_set = set(indices)

    def candidate_key(candidate: list[int]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            _ring_atom_priority(atom, ring_set, atomic_numbers, adjacency) for atom in candidate
        )

    candidates: list[list[int]] = []
    for start in range(n):
        candidates.append([indices[(start + i) % n] for i in range(n)])
        candidates.append([indices[(start - i) % n] for i in range(n)])
    return max(candidates, key=candidate_key)


def canonical_ring_indices(
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> list[int]:
    return prelog_canonical_ring_indices(ring_indices, atomic_numbers, adjacency)


def ring_endocyclic_torsions(
    coords: np.ndarray,
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> np.ndarray:
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    ring = coords[ring_indices]
    size = len(ring_indices)
    return np.array(
        [
            dihedral(ring[i], ring[(i + 1) % size], ring[(i + 2) % size], ring[(i + 3) % size])
            for i in range(size)
        ],
        dtype=float,
    )


def five_ring_endocyclic_torsions(coords: np.ndarray, ring_indices: list[int]) -> np.ndarray:
    if len(ring_indices) != 5:
        raise ValueError("five_ring_endocyclic_torsions requires a five-membered ring")
    return ring_endocyclic_torsions(coords, ring_indices)


def four_ring_pucker_torsion(
    coords: np.ndarray,
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> float:
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    ring = coords[ring_indices]
    return dihedral(ring[1], ring[2], ring[3], ring[0])


def ring_puckering_harmonics(ring_size: int) -> list[int]:
    if ring_size < 4:
        raise ValueError("Only rings with at least 4 atoms are supported")
    if ring_size == 4:
        return []
    if ring_size % 2:
        return list(range(2, (ring_size - 1) // 2 + 1))
    return list(range(2, ring_size // 2))


def ring_component_coefficients(ring_size: int, harmonic: int) -> tuple[np.ndarray, np.ndarray]:
    scale = math.sqrt(2.0 / float(ring_size))
    idx = np.arange(ring_size, dtype=float)
    phase = 2.0 * math.pi * float(harmonic) * idx / float(ring_size)
    qc = scale * np.cos(phase)
    qs = scale * np.sin(phase)
    return qc, qs


def five_ring_component_coefficients() -> tuple[np.ndarray, np.ndarray]:
    return ring_component_coefficients(5, 2)


def ring_state(
    coords: np.ndarray,
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> PuckeringState:
    if len(ring_indices) == 4:
        return four_ring_state(coords, ring_indices, atomic_numbers, adjacency)

    torsions = ring_endocyclic_torsions(coords, ring_indices, atomic_numbers, adjacency)
    modes: list[PuckeringMode] = []
    for harmonic in ring_puckering_harmonics(len(ring_indices)):
        qc_coeff, qs_coeff = ring_component_coefficients(len(ring_indices), harmonic)
        qc = float(qc_coeff @ torsions)
        qs = float(qs_coeff @ torsions)
        q = math.hypot(qc, qs)
        phi_deg = math.degrees(math.atan2(qs, qc)) % 360.0
        modes.append(PuckeringMode(harmonic=harmonic, qc=qc, qs=qs, q=q, phi_deg=phi_deg))
    if not modes:
        raise ValueError("No paired puckering modes available for this ring")
    first = modes[0]
    return PuckeringState(
        qc=first.qc, qs=first.qs, q=first.q, phi_deg=first.phi_deg, modes=tuple(modes)
    )


def five_ring_state(coords: np.ndarray, ring_indices: list[int]) -> PuckeringState:
    if len(ring_indices) != 5:
        raise ValueError("five_ring_state requires a five-membered ring")
    torsions = ring_endocyclic_torsions(coords, ring_indices)
    qc_coeff, qs_coeff = five_ring_component_coefficients()
    qc = float(qc_coeff @ torsions)
    qs = float(qs_coeff @ torsions)
    q = math.hypot(qc, qs)
    phi_deg = math.degrees(math.atan2(qs, qc)) % 360.0
    mode = PuckeringMode(harmonic=2, qc=qc, qs=qs, q=q, phi_deg=phi_deg)
    return PuckeringState(qc=qc, qs=qs, q=q, phi_deg=phi_deg, modes=(mode,))


def four_ring_state(
    coords: np.ndarray,
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> PuckeringState:
    puck = four_ring_pucker_torsion(coords, ring_indices, atomic_numbers, adjacency)
    mode = PuckeringMode(harmonic=0, qc=puck, qs=0.0, q=abs(puck), phi_deg=math.degrees(puck))
    return PuckeringState(qc=puck, qs=0.0, q=abs(puck), phi_deg=math.degrees(puck), modes=(mode,))


def puckering_state(
    coords: np.ndarray,
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> PuckeringState:
    if len(ring_indices) == 4:
        return four_ring_state(coords, ring_indices, atomic_numbers, adjacency)
    return ring_state(coords, ring_indices, atomic_numbers, adjacency)


def phase_constraint_coefficients(
    phi_deg: float, ring_size: int = 5, harmonic: int = 2
) -> np.ndarray:
    phi = math.radians(phi_deg)
    qc_coeff, qs_coeff = ring_component_coefficients(ring_size, harmonic)
    return -math.sin(phi) * qc_coeff + math.cos(phi) * qs_coeff


def gic_linear_expression(coefficients: np.ndarray, names: list[str]) -> str:
    terms: list[str] = []
    for coeff, name in zip(coefficients, names):
        if abs(coeff) < 5.0e-13:
            continue
        sign = "+" if coeff >= 0.0 else "-"
        if not terms:
            sign = "" if coeff >= 0.0 else "-"
        terms.append(f"{sign}{abs(coeff):.10f}*{name}")
    return "".join(terms) if terms else "0.0"


def ring_gic_torsion_lines(
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> tuple[list[str], list[str]]:
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    atom = [index + 1 for index in ring_indices]
    size = len(atom)
    names = [f"T{i + 1:03d}" for i in range(size)]
    lines = [
        f"{names[i]}(Inactive)=D({atom[i]},{atom[(i + 1) % size]},{atom[(i + 2) % size]},{atom[(i + 3) % size]})"
        for i in range(size)
    ]
    return lines, names


def five_ring_gic_torsion_lines(ring_indices: list[int]) -> tuple[list[str], list[str]]:
    if len(ring_indices) != 5:
        raise ValueError("five_ring_gic_torsion_lines requires a five-membered ring")
    return ring_gic_torsion_lines(ring_indices)


def ring_puckering_component_lines(
    ring_indices: list[int],
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> tuple[list[str], list[tuple[str, str, str, int]]]:
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    ring_size = len(ring_indices)
    torsion_lines, torsion_names = ring_gic_torsion_lines(ring_indices, atomic_numbers, adjacency)
    lines = list(torsion_lines)
    modes: list[tuple[str, str, str, int]] = []
    rpck_index = 1
    for pair_index, harmonic in enumerate(ring_puckering_harmonics(ring_size), start=1):
        qc_coeff, qs_coeff = ring_component_coefficients(ring_size, harmonic)
        rpck_c = f"RPck{rpck_index:03d}"
        rpck_s = f"RPck{rpck_index + 1:03d}"
        q_name = f"QPck{pair_index:03d}"
        phi_name = f"PhiP{pair_index:03d}"
        lines.append(f"{rpck_c}(Inactive)={gic_linear_expression(qc_coeff, torsion_names)}")
        lines.append(f"{rpck_s}(Inactive)={gic_linear_expression(qs_coeff, torsion_names)}")
        modes.append((q_name, phi_name, rpck_c, rpck_s, harmonic))
        rpck_index += 2
    return lines, modes


def ring_puckering_gic_lines(ring_indices: list[int]) -> list[str]:
    ring_indices = canonical_ring_indices(ring_indices)
    if len(ring_indices) == 4:
        atom = [index + 1 for index in ring_indices]
        return [
            f"RPck001(Inactive)=D({atom[1]},{atom[2]},{atom[3]},{atom[0]})",
            "QPck001=SQRT(RPck001*RPck001)",
            "PhiP001=RPck001",
        ]

    component_lines, modes = ring_puckering_component_lines(ring_indices)
    lines = list(component_lines)
    for q_name, phi_name, rpck_c, rpck_s, _harmonic in modes:
        lines.append(f"{q_name}=SQRT({rpck_c}*{rpck_c}+{rpck_s}*{rpck_s})")
        lines.append(f"{phi_name}=ATAN2({rpck_s},{rpck_c})")
    return lines


def angular_step_to_target(target_deg: float, current_deg: float) -> float:
    delta = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    if abs(delta + 180.0) < 1.0e-10 and target_deg > current_deg:
        delta = 180.0
    return delta


def ring_functional_target_gic(
    ring_indices: list[int],
    target_phi_deg: float,
    current_phi_deg: float,
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> tuple[list[str], float]:
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    if len(ring_indices) == 4:
        return four_ring_target_gic(
            ring_indices, target_phi_deg, current_phi_deg, atomic_numbers, adjacency
        )
    component_lines, modes = ring_puckering_component_lines(ring_indices, atomic_numbers, adjacency)
    step_deg = angular_step_to_target(target_phi_deg, current_phi_deg)
    step_rad = math.radians(step_deg)
    lines = list(component_lines)
    for i, (q_name, phi_name, rpck_c, rpck_s, _harmonic) in enumerate(modes):
        lines.append(f"{q_name}=SQRT({rpck_c}*{rpck_c}+{rpck_s}*{rpck_s})")
        if i == 0:
            lines.append(f"{phi_name}(NSteps=1,StepSize={step_rad:.10f})=ATAN2({rpck_s},{rpck_c})")
        else:
            lines.append(f"{phi_name}=ATAN2({rpck_s},{rpck_c})")
    return lines, step_deg


def five_ring_functional_target_gic(
    ring_indices: list[int],
    target_phi_deg: float,
    current_phi_deg: float,
) -> tuple[list[str], float]:
    if len(ring_indices) != 5:
        raise ValueError("five_ring_functional_target_gic requires a five-membered ring")
    return ring_functional_target_gic(ring_indices, target_phi_deg, current_phi_deg)


def ring_scan_to_zero_gic(
    ring_indices: list[int],
    target_phi_deg: float,
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> list[str]:
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    if len(ring_indices) == 4:
        raise ValueError("scan-to-zero requires a paired puckering mode")
    component_lines, modes = ring_puckering_component_lines(ring_indices, atomic_numbers, adjacency)
    _q_name, _phi_name, rpck_c, rpck_s, harmonic = modes[0]
    _torsion_lines, names = ring_gic_torsion_lines(ring_indices, atomic_numbers, adjacency)
    phase_coeff = phase_constraint_coefficients(target_phi_deg, len(ring_indices), harmonic)
    phi_label = f"{int(round(target_phi_deg * 1000.0)):06d}"
    lines = list(component_lines)
    for q_name, phi_name, c_name, s_name, _harmonic in modes:
        lines.append(f"{q_name}=SQRT({c_name}*{c_name}+{s_name}*{s_name})")
        lines.append(f"{phi_name}=ATAN2({s_name},{c_name})")
    lines.append(f"Cphi{phi_label}(Freeze)={gic_linear_expression(phase_coeff, names)}")
    return lines


def five_ring_scan_to_zero_gic(ring_indices: list[int], target_phi_deg: float) -> list[str]:
    if len(ring_indices) != 5:
        raise ValueError("five_ring_scan_to_zero_gic requires a five-membered ring")
    return ring_scan_to_zero_gic(ring_indices, target_phi_deg)


def four_ring_target_gic(
    ring_indices: list[int],
    target_deg: float,
    current_deg: float,
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> tuple[list[str], float]:
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    atom = [index + 1 for index in ring_indices]
    step_deg = target_deg - current_deg
    step_rad = math.radians(step_deg)
    lines = [
        f"RPck001(Inactive)=D({atom[1]},{atom[2]},{atom[3]},{atom[0]})",
        "QPck001=SQRT(RPck001*RPck001)",
        f"PhiP001(NSteps=1,StepSize={step_rad:.10f})=RPck001",
    ]
    return lines, step_deg


def phi_values(phi_start: float, phi_end: float, phi_step: float) -> list[float]:
    if phi_step <= 0.0:
        raise ValueError("--phi-step must be positive")
    nsteps = int(round((phi_end - phi_start) / phi_step))
    if nsteps < 0:
        raise ValueError("--phi-end must be >= --phi-start")
    values = [phi_start + i * phi_step for i in range(nsteps + 1)]
    if not values:
        raise ValueError("Empty phi grid")
    return values


def format_geometry(atoms: list[str], coords_angstrom: np.ndarray) -> str:
    lines = []
    for symbol, (x, y, z) in zip(atoms, coords_angstrom):
        lines.append(f"{symbol:2s} {x:16.8f} {y:16.8f} {z:16.8f}")
    return "\n".join(lines)


def build_gjf_links(
    atoms: list[str],
    coords_angstrom: np.ndarray,
    ring_indices: list[int],
    phi_start: float,
    phi_end: float,
    phi_step: float,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    mem: str = "16GB",
    nproc: str = "8",
    chk_prefix: str = "mw_path_phi",
    title: str = "Puckering constrained optimization",
    route: str = DEFAULT_GAUSSIAN_ROUTE,
    constraint_mode: str = "functional-targets",
    atomic_numbers: list[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> tuple[list[str], list[dict[str, float]]]:
    if atomic_numbers is None:
        try:
            atomic_number = _load_atomic_number_lookup()
            atomic_numbers = [atomic_number(atom) for atom in atoms]
        except Exception:
            atomic_numbers = None
    ring_indices = canonical_ring_indices(ring_indices, atomic_numbers, adjacency)
    phis = phi_values(phi_start, phi_end, phi_step)
    state = puckering_state(coords_angstrom, ring_indices, atomic_numbers, adjacency)

    lines: list[str] = []
    manifest: list[dict[str, float]] = []
    gaussian_route = ensure_gaussian_output_pickett(route)
    for i, phi in enumerate(phis):
        if constraint_mode == "functional-targets":
            gic_lines, step_deg = ring_functional_target_gic(
                ring_indices, float(phi), state.phi_deg, atomic_numbers, adjacency
            )
        elif constraint_mode == "scan-to-zero":
            gic_lines = ring_scan_to_zero_gic(ring_indices, float(phi), atomic_numbers, adjacency)
            step_deg = angular_step_to_target(float(phi), state.phi_deg)
        else:
            raise ValueError("Supported constraint modes are functional-targets or scan-to-zero")

        if i:
            lines.append("--Link1--")
        lines.extend(
            [
                f"%chk={chk_prefix}_{int(round(phi)):03d}.chk",
                f"%mem={mem}",
                f"%nprocshared={nproc}",
                gaussian_route,
                "",
                f"{title}: absolute phi={phi:.3f} deg",
                "",
                f"{charge} {multiplicity}",
                format_geometry(atoms, coords_angstrom),
                "",
                *gic_lines,
                "",
            ]
        )
        manifest.append(
            {
                "target_phi_deg": float(phi),
                "initial_q_rad": float(state.q),
                "initial_phi_deg": float(state.phi_deg),
                "gic_step_deg": float(step_deg),
            }
        )
    return lines, manifest


def _load_atomic_number_lookup():
    from .pipeline import _load_topology_elements

    return _load_topology_elements()


def _topology_for_atoms(atoms: list[str], coords_angstrom: np.ndarray):
    from .pipeline import build_topology

    atomic_number = _load_atomic_number_lookup()
    z_numbers = [atomic_number(atom) for atom in atoms]
    _cg, dg, ringset = build_topology(coords_angstrom / 0.52917721092, z_numbers)
    adjacency = [set(dg.adjacency[i]) for i in range(dg.natoms)]
    return z_numbers, adjacency, ringset


def auto_ring_indices(atoms: list[str], coords_angstrom: np.ndarray) -> list[int]:
    z_numbers, adjacency, ringset = _topology_for_atoms(atoms, coords_angstrom)
    if ringset is None:
        raise ValueError("No rings detected in molecular geometry")
    candidates = [
        canonical_ring_indices(list(ring.atoms), z_numbers, adjacency)
        for ring in ringset
        if len(ring.atoms) >= 4
    ]
    if not candidates:
        raise ValueError("No ring with at least 4 atoms detected in molecular geometry")
    candidates.sort(key=lambda ring: (len(ring), ring))
    return candidates[0]


def write_gaussian_scan_from_xyz(
    xyz_path: Path,
    ring_text: str | None,
    gjf_out: Path,
    *,
    phi_start: float,
    phi_end: float,
    phi_step: float,
    charge: int = 0,
    multiplicity: int = 1,
    mem: str = "16GB",
    nproc: str = "8",
    chk_prefix: str = "mw_path_phi",
    title: str = "Puckering constrained optimization",
    route: str = DEFAULT_GAUSSIAN_ROUTE,
    constraint_mode: str = "functional-targets",
) -> list[dict[str, float]]:
    atoms, coords_angstrom, _ = read_xyz(Path(xyz_path))
    z_numbers, adjacency, _ringset = _topology_for_atoms(atoms, coords_angstrom)
    if ring_text:
        raw_ring_indices = [int(item.strip()) - 1 for item in ring_text.split(",") if item.strip()]
        if len(raw_ring_indices) < 4:
            raise ValueError(
                "--ring must contain at least 4 comma-separated one-based atom indices"
            )
        if min(raw_ring_indices) < 0 or max(raw_ring_indices) >= len(atoms):
            raise ValueError("--ring contains an atom index outside the molecular geometry")
        if len(set(raw_ring_indices)) != len(raw_ring_indices):
            raise ValueError("--ring contains duplicated atom indices")
        ring_indices = canonical_ring_indices(raw_ring_indices, z_numbers, adjacency)
    else:
        ring_indices = auto_ring_indices(atoms, coords_angstrom)
    lines, manifest = build_gjf_links(
        atoms,
        coords_angstrom,
        ring_indices,
        phi_start,
        phi_end,
        phi_step,
        charge=charge,
        multiplicity=multiplicity,
        mem=mem,
        nproc=nproc,
        chk_prefix=chk_prefix,
        title=title,
        route=route,
        constraint_mode=constraint_mode,
        atomic_numbers=z_numbers,
        adjacency=adjacency,
    )
    Path(gjf_out).write_text("\n".join(lines) + "\n")
    return manifest
