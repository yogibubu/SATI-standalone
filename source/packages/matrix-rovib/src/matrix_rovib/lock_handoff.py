"""TRINITY ESALC hand-off to the LOCK rovibrational engine."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from matrix_rovib import (
    ContractedVibronicModel,
    FiniteMolecularSymmetryGroup,
    LAMRotationSurface,
    contract_diabatic_vibronic_hamiltonian,
    write_lam_rotation_surface,
    write_variational_rovibronic_model,
)


def build_lock_surface_from_esalc_scan(
    surface_csv: Path | str,
    geometry_xyz: Path | str,
    coordinates: Sequence[str] | None = None,
    *,
    periodic: Sequence[bool] | None = None,
    oracle_geometry_xyz: Path | str | None = None,
    out: Path | str | None = None,
    include_geometric_potential: bool = True,
    torsional_symmetry_order: int = 1,
    gf_ped_xyz: Path | str | None = None,
    nuclear_spin_weights: Sequence[float] = (),
) -> LAMRotationSurface | Path:
    """Construct a LOCK surface, optionally selecting its LAMs from SMITH.

    When ``coordinates`` is omitted, ``oracle_geometry_xyz`` supplies ORACLE's
    frozen molecular data and SMITH's GIC builder identifies the torsional
    coordinates and their periods. Energies remain paired with the ordered
    ESALC geometry trajectory.
    """
    if coordinates is None:
        if oracle_geometry_xyz is None:
            raise ValueError(
                "automatic LAM selection requires oracle_geometry_xyz; coordinates and "
                "periodicity are supplied by the frozen ORACLE/SMITH artifact"
            )
        surface = _build_lock_surface_from_smith_scan(
            surface_csv,
            geometry_xyz,
            oracle_geometry_xyz,
            include_geometric_potential=include_geometric_potential,
        )
        if out is None:
            return surface
        return write_lam_rotation_surface(out, surface)
    if periodic is None:
        raise ValueError("manual coordinates require explicit periodic flags")
    from importlib import import_module
    lam_rotation_surface_from_scan_files = import_module(
        "matrix_dvr"
    ).lam_rotation_surface_from_scan_files

    symmetry_orders: tuple[int, ...] = ()
    if gf_ped_xyz is not None:
        from .oracle_lock_symmetry import gf_torsion_periodicities

        symmetry_orders = gf_torsion_periodicities(gf_ped_xyz, coordinates)
        if symmetry_orders:
            torsional_symmetry_order = symmetry_orders[0]

    surface = lam_rotation_surface_from_scan_files(
        surface_csv,
        geometry_xyz,
        coordinates,
        periodic=periodic,
        include_geometric_potential=include_geometric_potential,
        torsional_symmetry_order=torsional_symmetry_order,
        lam_symmetry_orders=symmetry_orders,
        nuclear_spin_weights=nuclear_spin_weights,
    )
    if out is None:
        return surface
    return write_lam_rotation_surface(out, surface)


def _build_lock_surface_from_smith_scan(
    surface_csv: Path | str,
    geometry_xyz: Path | str,
    oracle_geometry_xyz: Path | str,
    *,
    include_geometric_potential: bool,
) -> LAMRotationSurface:
    """Use SMITH's frozen GICs to map the ESALC geometry grid to LAM values."""
    import csv

    from matrix_chem import read_enriched_xyz, read_molecular_symmetry, read_permutation_inversion_symmetry
    from importlib import import_module
    from matrix_core import read_sectioned_lines, section_content
    dvr = import_module("matrix_dvr")
    smith = import_module("matrix_smith")
    smith_serialization = import_module("matrix_smith.definition_serialization")
    HARTREE_TO_MHZ = dvr.HARTREE_TO_MHZ
    lam_rotation_surface_from_cartesian_grid = dvr.lam_rotation_surface_from_cartesian_grid
    build_gic_definition_from_xyzin = smith.build_gic_definition_from_xyzin
    evaluate_gic_values = smith.evaluate_gic_values
    read_gic_definition_from_xyzin = smith_serialization.read_gic_definition_from_xyzin
    from .oracle_lock_symmetry import _principal_body_frame, _read_xyz_trajectory

    oracle_path = Path(oracle_geometry_xyz)
    if section_content(read_sectioned_lines(oracle_path), "GIC"):
        definition = read_gic_definition_from_xyzin(oracle_path)
    else:
        # Build coordinates from ORACLE geometry/topology when no frozen SMITH
        # GIC contract is present. The default group is always ORACLE's group.
        definition = build_gic_definition_from_xyzin(oracle_path)
    symmetry = read_molecular_symmetry(oracle_path)
    pi_contract = read_permutation_inversion_symmetry(oracle_path)
    atoms, frames = _read_xyz_trajectory(Path(geometry_xyz))
    oracle_geometry = read_enriched_xyz(oracle_path)
    if atoms != tuple(oracle_geometry.atoms):
        raise ValueError("ESALC trajectory atom order differs from the ORACLE/SMITH geometry")
    if len(atoms) != len(definition.reference_coordinates_angstrom):
        raise ValueError("SMITH geometry atom count differs from the ESALC scan")
    if not definition.periodic_coordinate_estimates:
        raise ValueError("SMITH found no periodic coordinates for the LAM scan")

    estimate_by_identifier = {
        item.coordinate_identifier: item
        for item in definition.periodic_coordinate_estimates
        if item.target == "TORSION" and item.periodicity >= 1
    }
    gic_indices = {
        gic.identifier: index for index, gic in enumerate(definition.gics)
    }
    candidates = [
        (gic_indices[key], estimate)
        for key, estimate in estimate_by_identifier.items()
        if key in gic_indices
    ]
    values = np.vstack(
        [evaluate_gic_values(definition, coordinates_angstrom=frame) for frame in frames]
    )
    selected = [
        (index, estimate)
        for index, estimate in candidates
        if float(np.ptp(values[:, index])) > 1.0e-7
    ]
    if not selected:
        raise ValueError("SMITH found no torsional GICs eligible for LAM treatment")
    if len(selected) > 2:
        raise ValueError(
            "automatic ESALC surface import supports at most two selected SMITH torsions"
        )

    csv_path = Path(surface_csv)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != len(frames) or "energy_hartree" not in rows[0]:
        raise ValueError("ESALC CSV must provide one energy_hartree per DVR geometry")
    energies = np.asarray([float(row["energy_hartree"]) for row in rows], dtype=float)
    selected_values = values[:, [index for index, _estimate in selected]]
    grids: list[np.ndarray] = []
    grid_indices: list[np.ndarray] = []
    for axis in range(selected_values.shape[1]):
        coordinate_values = selected_values[:, axis]
        grid = np.unique(np.round(coordinate_values, decimals=10))
        if len(grid) < 3:
            raise ValueError("each automatically selected SMITH torsion needs at least 3 grid points")
        spacing = np.diff(grid)
        if not np.allclose(spacing, spacing[0], rtol=1.0e-6, atol=1.0e-8):
            raise ValueError("SMITH-derived torsion values do not form a uniform DVR grid")
        # GIC evaluation and decimal XYZ serialization perturb an otherwise
        # uniform sampling grid. Store its uniform lattice after the above
        # tolerance check; geometries and energies are neither fitted nor moved.
        grid = np.linspace(grid[0], grid[-1], len(grid))
        spacing = np.diff(grid)
        nearest = np.rint((coordinate_values - grid[0]) / spacing[0]).astype(int)
        if np.any(nearest < 0) or np.any(nearest >= len(grid)) or not np.allclose(
            grid[nearest], coordinate_values, rtol=0.0, atol=2.0e-7
        ):
            raise ValueError("ESALC geometries cannot be assigned to the SMITH torsion grid")
        grids.append(grid)
        grid_indices.append(nearest)
    shape = tuple(len(grid) for grid in grids)
    if int(np.prod(shape)) != len(frames):
        raise ValueError("ESALC geometries do not form a complete rectangular SMITH torsion grid")
    occupied = np.zeros(shape, dtype=bool)
    potential = np.empty(shape, dtype=float)
    geometry_grid = np.empty(shape + (len(atoms), 3), dtype=float)
    for frame_index, frame in enumerate(frames):
        slot = tuple(int(axis[frame_index]) for axis in grid_indices)
        if occupied[slot]:
            raise ValueError(f"duplicate SMITH torsion grid point {slot}")
        occupied[slot] = True
        potential[slot] = energies[frame_index]
        geometry_grid[slot] = frame
    if not np.all(occupied):
        raise ValueError("ESALC geometries leave holes in the SMITH torsion grid")
    masses = pi_contract.nuclear_masses_amu()
    body_axes, _oracle_masses = _principal_body_frame(oracle_path, masses_amu=masses)
    geometry_grid = np.einsum("...ia,ab->...ib", geometry_grid, body_axes)
    extend_permutation_inversion_from_scan = import_module(
        "matrix_oracle"
    ).extend_permutation_inversion_from_scan
    pi_contract = extend_permutation_inversion_from_scan(
        pi_contract, geometry_grid.reshape((-1, len(atoms), 3)), masses, grid_shape=shape,
    )
    reference_energy = float(np.min(potential))
    return lam_rotation_surface_from_cartesian_grid(
        tuple(grids),
        (potential - reference_energy) * HARTREE_TO_MHZ,
        masses,
        geometry_grid,
        # Keep the scan's continuous Cartesian embedding. Independent Kabsch
        # gauges can introduce torsion-dependent/discontinuous frame rotations
        # that a finite spectral derivative need not represent covariantly.
        align=False,
        periodic=(True,) * len(grids),
        labels=tuple(estimate.coordinate_name for _index, estimate in selected),
        include_geometric_potential=include_geometric_potential,
        # PI adaptation handles the whole molecular group. Topological period
        # seeds must not also impose independent cyclic sector projectors.
        lam_symmetry_orders=(1,) * len(grids),
        torsional_symmetry_order=1,
        permutation_inversion_contract=pi_contract,
        complete_pi_group_from_grid=True,
        metadata={
            "producer": "TRINITY/ESALC",
            "symmetry_source": "ORACLE",
            "coordinate_source": "SMITH GIC",
            "periodicity_source": "ORACLE PI action on the SMITH geometry grid",
            "surface_csv": str(csv_path),
            "geometry_xyz": str(Path(geometry_xyz)),
            "oracle_geometry_xyz": str(oracle_path),
            "point_group": symmetry.point_group,
            "body_axes_from_input_cartesian": body_axes.tolist(),
            "reference_energy_hartree": reference_energy,
            "energy_conversion_MHz_per_hartree": HARTREE_TO_MHZ,
        },
    )


def build_lock_variational_rovibronic_model(
    *,
    lam_vectors: np.ndarray,
    diabatic_surface_MHz: np.ndarray,
    electronic_angular_momentum: Sequence[np.ndarray],
    lam_angular_momentum: Sequence[np.ndarray],
    two_S: int,
    rotational_metric_MHz: np.ndarray,
    primitive_momentum_operators: Sequence[np.ndarray] = (),
    derivative_couplings_MHz: Sequence[np.ndarray] = (),
    spin_orbit_MHz: np.ndarray | None = None,
    spin_lam_MHz: np.ndarray | None = None,
    contracted_internal_hyperfine_MHz: np.ndarray | None = None,
    dipole_surfaces_D: Sequence[np.ndarray] = (),
    symmetry_group: FiniteMolecularSymmetryGroup | None = None,
    vibronic_symmetry_representation: Sequence[np.ndarray] = (),
    dipole_irreps: Sequence[str] = (),
    required_total_irrep: str | None = None,
    vibronic_labels: Sequence[str] = (),
    electronic_labels: Sequence[str] = (),
) -> ContractedVibronicModel:
    """Contract an ESALC-grid diabatic field into the LOCK common basis."""
    vibronic = contract_diabatic_vibronic_hamiltonian(
        lam_vectors,
        diabatic_surface_MHz,
        primitive_momentum_operators=primitive_momentum_operators,
        derivative_couplings_MHz=derivative_couplings_MHz,
    )
    dipoles = None
    if dipole_surfaces_D:
        if len(dipole_surfaces_D) != 3:
            raise ValueError("dipole_surfaces_D requires x, y, and z components")
        dipoles = tuple(
            contract_diabatic_vibronic_hamiltonian(lam_vectors, component)
            for component in dipole_surfaces_D
        )
    return ContractedVibronicModel(
        vibronic_hamiltonian_MHz=vibronic,
        electronic_angular_momentum=tuple(electronic_angular_momentum),
        lam_angular_momentum=tuple(lam_angular_momentum),
        two_S=two_S,
        rotational_metric_MHz=rotational_metric_MHz,
        spin_orbit_MHz=spin_orbit_MHz,
        spin_lam_MHz=spin_lam_MHz,
        contracted_internal_hyperfine_MHz=contracted_internal_hyperfine_MHz,
        transition_dipoles_D=dipoles,
        symmetry_group=symmetry_group,
        vibronic_symmetry_representation=tuple(vibronic_symmetry_representation),
        dipole_irreps=tuple(dipole_irreps),
        required_total_irrep=required_total_irrep,
        vibronic_labels=tuple(vibronic_labels),
        electronic_labels=tuple(electronic_labels),
    )


def write_lock_variational_rovibronic_handoff(
    path: Path | str, **model_data
) -> Path:
    """Write the versioned TRINITY artifact consumed by ``matrix lock``."""
    return write_variational_rovibronic_model(
        path, build_lock_variational_rovibronic_model(**model_data)
    )


__all__ = [
    "build_lock_surface_from_esalc_scan",
    "build_lock_variational_rovibronic_model",
    "write_lock_variational_rovibronic_handoff",
]
