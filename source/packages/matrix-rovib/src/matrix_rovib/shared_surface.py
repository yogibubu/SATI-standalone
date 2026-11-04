"""LOCK kinetic-model construction from the shared molecular surface artifact."""
from __future__ import annotations

from math import isfinite

import numpy as np

from matrix_core.molecular_surface import MolecularSurface, read_molecular_surface
from matrix_chem import BOHR_TO_ANGSTROM


def lock_surface_from_molecular_surface(
    source, *, electronic_state, reference_energy_hartree,
    include_geometric_potential=True, torsional_symmetry_order=1,
    lam_symmetry_orders=(), nuclear_spin_weights=(), permutation_inversion_contract=None,
):
    """Build the existing LOCK LAM operator using shared geometries and energies.

    All Cartesian tensors use the artifact's declared continuous frame. No
    independent Kabsch alignment is imposed. Dipoles are converted explicitly
    and, for charged species, translated to the mass-centred geometry origin.
    Gradients/Hessians and unused properties remain in the shared source; they
    are never converted into an unrequested fitted force field.
    """
    from matrix_dvr.lam_metric import HARTREE_TO_MHZ, lam_rotation_surface_from_cartesian_grid
    from .unified_lock import AU_DIPOLE_TO_DEBYE

    surface = source if isinstance(source, MolecularSurface) else read_molecular_surface(source)
    if not 1 <= len(surface.coordinates) <= 2:
        raise ValueError('the Cartesian LOCK metric adapter supports one or two coordinates')
    offset = reference_energy_hartree
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not isfinite(offset):
        raise ValueError('an explicit finite energy origin in hartree is required')
    if not isinstance(include_geometric_potential, bool):
        raise ValueError('include_geometric_potential must be a boolean')
    state = surface.state(electronic_state)
    grids = tuple(np.asarray(c.values, dtype=float) for c in surface.coordinates)
    for axis, grid in zip(surface.coordinates, grids, strict=True):
        spacing = np.diff(grid)
        if len(grid) < 3 or not np.allclose(spacing, spacing[0], rtol=1e-9, atol=1e-12):
            raise ValueError('LOCK requires uniform axes with at least three grid points')
        if axis.period is not None and not np.isclose(axis.period, len(grid)*spacing[0], rtol=1e-9, atol=1e-12):
            raise ValueError('periodic LOCK axes must sample exactly one endpoint-exclusive period')
    geometry = np.asarray(surface.geometries.values).reshape(surface.geometries.shape)
    energies = np.asarray(state.energies.values).reshape(surface.grid_shape)
    properties = dict(state.properties)
    dipole = None
    if 'dipole' in properties:
        tensor = properties['dipole']
        dipole = np.asarray(tensor.values).reshape(tensor.shape).copy()
        if tensor.unit == 'debye':
            dipole /= AU_DIPOLE_TO_DEBYE
        if surface.charge:
            masses = np.asarray(surface.masses_amu)
            center = np.sum(geometry*masses[:, None], axis=-2)/np.sum(masses)
            dipole -= surface.charge*center/BOHR_TO_ANGSTROM
    metadata = dict(shared_surface_sha256=surface.sha256, shared_surface_identifier=surface.identifier,
                    electronic_state=electronic_state, energy_reference=surface.energy_reference,
                    reference_energy_hartree=float(offset), cartesian_frame=surface.cartesian_frame,
                    source_representation=surface.representation,
                    source_provenance=surface.provenance, source_charge=surface.charge,
                    gradient_samples_available=state.gradients is not None,
                    hessian_samples_available=state.hessians is not None,
                    source_properties=list(properties), dipole_origin='mass_center',
                    physical_accuracy_certified=False)
    return lam_rotation_surface_from_cartesian_grid(
        grids, (energies-offset)*HARTREE_TO_MHZ, np.asarray(surface.masses_amu), geometry,
        periodic=tuple(c.period is not None for c in surface.coordinates),
        labels=tuple(c.identifier for c in surface.coordinates), dipole_au=dipole,
        align=False, include_geometric_potential=include_geometric_potential,
        torsional_symmetry_order=torsional_symmetry_order,
        lam_symmetry_orders=lam_symmetry_orders, nuclear_spin_weights=nuclear_spin_weights,
        permutation_inversion_contract=permutation_inversion_contract, metadata=metadata,
    )
