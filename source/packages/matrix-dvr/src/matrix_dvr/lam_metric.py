"""Pointwise molecular metric for coupled LAM--rotation DVR calculations."""

from __future__ import annotations

from collections.abc import Sequence
import csv
from pathlib import Path

import numpy as np

from matrix_rovib.lam_rotation import LAMRotationSurface


ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2 = 505379.0090099937
HARTREE_TO_MHZ = 6.5796839204999e9


def lam_rotation_surface_from_scan_files(
    surface_csv: Path | str,
    geometry_xyz: Path | str,
    coordinates: Sequence[str],
    *,
    periodic: Sequence[bool],
    include_geometric_potential: bool = True,
    torsional_symmetry_order: int = 1,
    lam_symmetry_orders: Sequence[int] = (),
    nuclear_spin_weights: Sequence[float] = (),
) -> LAMRotationSurface:
    """Build the LOCK surface contract from an ESALC 1D or 2D scan."""
    surface_path = Path(surface_csv)
    geometry_path = Path(geometry_xyz)
    labels = tuple(str(value) for value in coordinates)
    if len(labels) not in {1, 2}:
        raise ValueError("LOCK scan import supports one or two coordinates")
    with surface_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    atoms, frames = _read_multiframe_xyz(geometry_path)
    if not rows or len(rows) != len(frames):
        raise ValueError("scan CSV and geometry trajectory must contain the same points")
    required = set(labels) | {"energy_hartree"}
    if not required.issubset(rows[0]):
        raise ValueError(f"scan CSV lacks columns: {', '.join(sorted(required - set(rows[0])))}")
    grids = tuple(
        np.asarray(sorted({float(row[label]) for row in rows}), dtype=float) for label in labels
    )
    shape = tuple(len(grid) for grid in grids)
    if int(np.prod(shape)) != len(rows):
        raise ValueError("scan CSV is not a complete rectangular grid")
    energies = np.empty(shape, dtype=float)
    geometries = np.empty(shape + (len(atoms), 3), dtype=float)
    lookup = tuple({float(value): index for index, value in enumerate(grid)} for grid in grids)
    occupied: set[tuple[int, ...]] = set()
    for row, frame in zip(rows, frames, strict=True):
        index = tuple(lookup[axis][float(row[label])] for axis, label in enumerate(labels))
        if index in occupied:
            raise ValueError(f"duplicate scan point at {tuple(float(row[label]) for label in labels)}")
        occupied.add(index)
        energies[index] = float(row["energy_hartree"])
        geometries[index] = frame
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number

    masses = np.asarray([atomic_mass(atomic_number(atom) or 0) for atom in atoms], dtype=float)
    reference_energy = float(np.min(energies))
    return lam_rotation_surface_from_cartesian_grid(
        grids,
        (energies - reference_energy) * HARTREE_TO_MHZ,
        masses,
        geometries,
        periodic=periodic,
        labels=labels,
        include_geometric_potential=include_geometric_potential,
        torsional_symmetry_order=torsional_symmetry_order,
        lam_symmetry_orders=lam_symmetry_orders,
        nuclear_spin_weights=nuclear_spin_weights,
        metadata={
            "producer": "TRINITY/ESALC",
            "surface_csv": str(surface_path),
            "geometry_xyz": str(geometry_path),
            "reference_energy_hartree": reference_energy,
            "energy_conversion_MHz_per_hartree": HARTREE_TO_MHZ,
        },
    )


def lam_rotation_surface_from_cartesian_grid(
    coordinates: Sequence[np.ndarray],
    potential_MHz: np.ndarray,
    masses_amu: np.ndarray,
    geometries_angstrom: np.ndarray,
    *,
    periodic: Sequence[bool],
    labels: Sequence[str] = (),
    dipole_au: np.ndarray | None = None,
    align: bool = True,
    reference_index: Sequence[int] | None = None,
    include_geometric_potential: bool = True,
    metadata: dict[str, object] | None = None,
    permutation_inversion_contract=None,
    complete_pi_group_from_grid: bool = False,
    atoms: Sequence[str] | None = None,
    nuclear_labels: Sequence[str] | None = None,
    torsional_symmetry_order: int = 1,
    lam_symmetry_orders: Sequence[int] = (),
    nuclear_spin_weights: Sequence[float] = (),
) -> LAMRotationSurface:
    """Derive ``I``, ``C``, ``G`` and their Schur reduction on a LAM grid.

    Derivatives and coefficient tensors use the centered input embedding.
    With ``align=True`` a proper Kabsch field transports the *discretized*
    operator and retained basis to the aligned frame. It never differentiates
    the rotated samples again. Equivalent-atom labels must be continuous.
    """
    grids = tuple(np.asarray(grid, dtype=float).reshape(-1) for grid in coordinates)
    dimension = len(grids)
    if dimension not in {1, 2}:
        raise ValueError("Cartesian LAM metric construction supports one or two coordinates")
    shape = tuple(len(grid) for grid in grids)
    boundary = tuple(bool(value) for value in periodic)
    if len(boundary) != dimension:
        raise ValueError("one boundary flag is required per coordinate")
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    geometry = np.asarray(geometries_angstrom, dtype=float)
    if geometry.shape != shape + (len(masses), 3):
        raise ValueError("geometry grid shape is inconsistent with coordinates and masses")
    if np.any(masses <= 0.0) or not np.all(np.isfinite(masses)) or not np.all(np.isfinite(geometry)):
        raise ValueError("finite geometries and positive finite masses are required")
    potential = np.asarray(potential_MHz, dtype=float)
    if potential.shape != shape or not np.all(np.isfinite(potential)):
        raise ValueError("potential shape is inconsistent with the LAM grid")
    ref = tuple(reference_index or np.unravel_index(int(np.argmin(potential)), shape))
    if len(ref) != dimension or any(not 0 <= value < shape[i] for i, value in enumerate(ref)):
        raise ValueError("reference_index is outside the LAM grid")
    centered = _center_geometries(geometry, masses)
    rotations = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
    if align:
        _aligned, rotations = _align_grid(centered, masses, ref)
    pi_metadata = {}
    if permutation_inversion_contract == "auto":
        if dimension != 1 or boundary[0]:
            raise ValueError("automatic confined-path PI reflection requires one nonperiodic coordinate")
        if atoms is None or nuclear_labels is None:
            raise ValueError("automatic path PI reflection requires atom and isotope labels")
        from matrix_chem import infer_confined_path_reflection

        permutation_inversion_contract, pi_action = infer_confined_path_reflection(
            atoms, nuclear_labels, centered.reshape((-1, len(masses), 3)), masses,
            coordinates=grids[0],
        )
        pi_metadata["oracle_pi_grid_action"] = pi_action.to_dict()
    elif permutation_inversion_contract is not None:
        if complete_pi_group_from_grid:
            from matrix_chem import complete_permutation_inversion_from_geometry_grid

            permutation_inversion_contract, pi_action = (
                complete_permutation_inversion_from_geometry_grid(
                    permutation_inversion_contract,
                    centered.reshape((-1, len(masses), 3)),
                    masses,
                )
            )
        else:
            from matrix_chem import derive_permutation_inversion_grid_action

            pi_action = derive_permutation_inversion_grid_action(
                permutation_inversion_contract,
                centered.reshape((-1, len(masses), 3)),
                masses,
            )
        pi_metadata["oracle_pi_grid_action"] = pi_action.to_dict()
    elif complete_pi_group_from_grid:
        raise ValueError("complete PI grid inference requires an ORACLE seed contract")
    transformed_dipole = None
    if dipole_au is not None:
        dipole = np.asarray(dipole_au, dtype=float)
        if dipole.shape != shape + (3,) or not np.all(np.isfinite(dipole)):
            raise ValueError("dipole_au must have grid shape plus one Cartesian axis")
        transformed_dipole = dipole  # Coefficients remain in the reference embedding.
    tangents = np.stack(
        [_differentiate(centered, grids[axis], axis, boundary[axis]) for axis in range(dimension)],
        axis=-2,
    )
    tangents -= np.sum(masses.reshape((1,) * dimension + (-1, 1, 1)) * tangents, axis=-3)[
        ..., None, :, :
    ] / np.sum(masses)
    inertia, coupling, internal = _mass_metric(masses, centered, tangents)
    surface = lam_rotation_surface_from_metric_tensors(
        grids,
        potential,
        inertia,
        coupling,
        internal,
        periodic=boundary,
        labels=tuple(labels),
        dipole_au=transformed_dipole,
        include_geometric_potential=include_geometric_potential,
        metadata={
            **(metadata or {}),
            **pi_metadata,
            "kinetic_construction": "transported_cartesian_path" if align else "cartesian_path",
            "reference_index": list(ref),
        },
        torsional_symmetry_order=torsional_symmetry_order,
        lam_symmetry_orders=lam_symmetry_orders,
        nuclear_spin_weights=nuclear_spin_weights,
    )
    if align:
        from matrix_rovib.body_frame import transport_lam_rotation_surface
        surface = transport_lam_rotation_surface(surface, rotations.reshape((-1, 3, 3)))
    return surface



def lam_rotation_surface_from_metric_tensors(
    coordinates: Sequence[np.ndarray],
    potential_MHz: np.ndarray,
    inertia_amu_angstrom2: np.ndarray,
    coupling_metric: np.ndarray,
    internal_metric: np.ndarray,
    *,
    periodic: Sequence[bool],
    labels: Sequence[str] = (),
    dipole_au: np.ndarray | None = None,
    include_geometric_potential: bool = True,
    metadata: dict[str, object] | None = None,
    torsional_symmetry_order: int = 1,
    lam_symmetry_orders: Sequence[int] = (),
    nuclear_spin_weights: Sequence[float] = (),
) -> LAMRotationSurface:
    """Reduce supplied pointwise molecular metric tensors by Schur inversion."""
    grids = tuple(np.asarray(grid, dtype=float).reshape(-1) for grid in coordinates)
    dimension = len(grids)
    shape = tuple(len(grid) for grid in grids)
    inertia = np.asarray(inertia_amu_angstrom2, dtype=float)
    coupling = np.asarray(coupling_metric, dtype=float)
    internal = np.asarray(internal_metric, dtype=float)
    if inertia.shape != shape + (3, 3):
        raise ValueError("inertia tensor shape is inconsistent with the LAM grid")
    if coupling.shape != shape + (3, dimension):
        raise ValueError("rotation-LAM coupling shape is inconsistent with the grid")
    if internal.shape != shape + (dimension, dimension):
        raise ValueError("internal metric shape is inconsistent with the grid")
    inverse_inertia = np.linalg.inv(inertia)
    rho = np.einsum("...ab,...bm->...am", inverse_inertia, coupling)
    schur = internal - np.einsum("...am,...an->...mn", coupling, rho)
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError("the rotational inertia must be positive definite at every point")
    if np.min(np.linalg.eigvalsh(schur)) <= 0.0:
        raise ValueError("the reduced LAM metric must be positive definite at every point")
    factor = ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2
    F = factor * np.linalg.inv(schur)
    R = factor * inverse_inertia
    geometric = None
    if include_geometric_potential:
        geometric = _geometric_potential(grids, tuple(periodic), inertia, schur, F)
    return LAMRotationSurface(
        coordinates=grids,
        potential_MHz=np.asarray(potential_MHz, dtype=float),
        F_MHz=F,
        rho=rho,
        R_MHz=R,
        periodic=tuple(bool(value) for value in periodic),
        labels=tuple(labels),
        dipole_au=dipole_au,
        geometric_potential_MHz=geometric,
        metadata=metadata,
        torsional_symmetry_order=int(torsional_symmetry_order),
        lam_symmetry_orders=tuple(int(value) for value in lam_symmetry_orders),
        nuclear_spin_weights=tuple(float(value) for value in nuclear_spin_weights),
    )


def _center_geometries(geometry: np.ndarray, masses: np.ndarray) -> np.ndarray:
    dimension = geometry.ndim - 2
    weights = masses.reshape((1,) * dimension + (-1, 1))
    center = np.sum(weights * geometry, axis=-2, keepdims=True) / np.sum(masses)
    return geometry - center


def _align_grid(
    geometry: np.ndarray, masses: np.ndarray, reference_index: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    shape = geometry.shape[:-2]
    reference = geometry[reference_index]
    aligned = np.empty_like(geometry)
    rotations = np.empty(shape + (3, 3), dtype=float)
    for index in np.ndindex(shape):
        current = geometry[index]
        covariance = (masses[:, None] * current).T @ reference
        left, _singular, right = np.linalg.svd(covariance)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        aligned[index] = current @ rotation
        rotations[index] = rotation
    return aligned, rotations


def _mass_metric(
    masses: np.ndarray, geometry: np.ndarray, tangents: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Batched contractions avoid a Python loop over every grid point/atom.
    second = np.einsum("i,...ia,...ib->...ab", masses, geometry, geometry, optimize=True)
    inertia = np.trace(second, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) - second
    coupling = np.einsum("i,...ima->...am", masses, np.cross(geometry[..., :, None, :], tangents), optimize=True)
    internal = np.einsum("i,...ima,...ina->...mn", masses, tangents, tangents, optimize=True)
    return inertia, coupling, internal


def _geometric_potential(
    grids: tuple[np.ndarray, ...],
    periodic: tuple[bool, ...],
    inertia: np.ndarray,
    schur: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    weight = np.sqrt(np.linalg.det(inertia) * np.linalg.det(schur))
    ell = np.log(weight / float(np.ravel(weight)[0]))
    gradient = [
        _differentiate(ell, grids[axis], axis, periodic[axis]) for axis in range(len(grids))
    ]
    divergence = np.zeros_like(ell)
    quadratic = np.zeros_like(ell)
    for mu in range(len(grids)):
        flux = np.zeros_like(ell)
        for nu in range(len(grids)):
            flux += F[..., mu, nu] * gradient[nu]
            quadratic += 0.25 * gradient[mu] * F[..., mu, nu] * gradient[nu]
        divergence += 0.5 * _differentiate(flux, grids[mu], mu, periodic[mu])
    return np.real_if_close(divergence + quadratic).real


def _differentiate(values: np.ndarray, grid: np.ndarray, axis: int, periodic: bool) -> np.ndarray:
    array = np.asarray(values)
    if not periodic:
        return np.gradient(array, grid, axis=axis, edge_order=2)
    spacing = float(grid[1] - grid[0])
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(len(grid), d=spacing)
    reshape = [1] * array.ndim
    reshape[axis] = len(grid)
    transformed = np.fft.fft(array, axis=axis)
    derivative = np.fft.ifft(1j * wave_numbers.reshape(reshape) * transformed, axis=axis)
    return np.real_if_close(derivative).real


def _read_multiframe_xyz(path: Path) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0
    atoms: tuple[str, ...] | None = None
    frames: list[np.ndarray] = []
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"invalid XYZ atom count at line {cursor + 1}") from exc
        if cursor + count + 1 >= len(lines):
            raise ValueError("truncated multi-frame XYZ trajectory")
        symbols: list[str] = []
        coordinates: list[list[float]] = []
        for line in lines[cursor + 2 : cursor + 2 + count]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError("invalid atom row in multi-frame XYZ trajectory")
            symbols.append(fields[0])
            coordinates.append([float(value) for value in fields[1:4]])
        current_atoms = tuple(symbols)
        if atoms is None:
            atoms = current_atoms
        elif current_atoms != atoms:
            raise ValueError("atom order changes between XYZ frames")
        frames.append(np.asarray(coordinates, dtype=float))
        cursor += count + 2
    if atoms is None or not frames:
        raise ValueError("empty multi-frame XYZ trajectory")
    return atoms, tuple(frames)


__all__ = [
    "HARTREE_TO_MHZ",
    "ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2",
    "lam_rotation_surface_from_cartesian_grid",
    "lam_rotation_surface_from_metric_tensors",
    "lam_rotation_surface_from_scan_files",
]
