"""TRINITY ESALC hand-off to the LOCK rovibrational engine."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matrix_rovib import LAMRotationSurface, write_lam_rotation_surface


def build_lock_surface_from_esalc_scan(
    surface_csv: Path | str,
    geometry_xyz: Path | str,
    coordinates: Sequence[str],
    *,
    periodic: Sequence[bool],
    out: Path | str | None = None,
    include_geometric_potential: bool = True,
    torsional_symmetry_order: int = 1,
    nuclear_spin_weights: Sequence[float] = (),
) -> LAMRotationSurface | Path:
    """Construct the versioned TRINITY-to-LOCK surface from an ESALC grid."""
    from matrix_dvr import lam_rotation_surface_from_scan_files

    surface = lam_rotation_surface_from_scan_files(
        surface_csv,
        geometry_xyz,
        coordinates,
        periodic=periodic,
        include_geometric_potential=include_geometric_potential,
        torsional_symmetry_order=torsional_symmetry_order,
        nuclear_spin_weights=nuclear_spin_weights,
    )
    if out is None:
        return surface
    return write_lam_rotation_surface(out, surface)


def build_lock_surface_from_shared_surface(
    source,
    *,
    electronic_state: str,
    reference_energy_hartree: float,
    out: Path | str | None = None,
    include_geometric_potential: bool = True,
    torsional_symmetry_order: int = 1,
    lam_symmetry_orders: Sequence[int] = (),
    nuclear_spin_weights: Sequence[float] = (),
    permutation_inversion_contract=None,
) -> LAMRotationSurface | Path:
    """Pass the common E/G/H/property artifact to LOCK's kinetic-model owner.

    ``source`` is a MolecularSurface or its digest-verified JSON file. The same
    source can be opened by ``matrix_trinity.shared_surface.open_trinity_surface``.
    """
    from matrix_rovib.shared_surface import lock_surface_from_molecular_surface

    surface = lock_surface_from_molecular_surface(
        source,
        electronic_state=electronic_state,
        reference_energy_hartree=reference_energy_hartree,
        include_geometric_potential=include_geometric_potential,
        torsional_symmetry_order=torsional_symmetry_order,
        lam_symmetry_orders=lam_symmetry_orders,
        nuclear_spin_weights=nuclear_spin_weights,
        permutation_inversion_contract=permutation_inversion_contract,
    )
    if out is None:
        return surface
    return write_lam_rotation_surface(out, surface)


__all__ = ["build_lock_surface_from_esalc_scan", "build_lock_surface_from_shared_surface"]
