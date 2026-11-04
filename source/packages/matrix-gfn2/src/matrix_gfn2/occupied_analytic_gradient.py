"""Experimental complete analytic GFN2 gradient for a local occupied factor.

This adapter evaluates the same fixed-density native SCC functional used by
the occupied-only relaxation.  It reuses the production pair-packed gradient
assembler without changing the normal dense SCF route.  The occupied factor
must be stationary enough for its Pulay matrix to be physically meaningful.
"""

from __future__ import annotations

import numpy as np

from . import calculator as model
from .external_dense_scf import prepare_native_dense_scf_function
from .occupied_local_subspace import MetricOccupiedSubspace, OccupiedSubspace


def prepare_occupied_analytic_gradient(calculator, coordinates_angstrom):
    """Return overlap, Fock/energy, and full analytic-gradient callbacks.

    The experimental adapter covers 21--512 atom closed-shell systems. Above
    63 atoms it uses the same sparse dispersion preparation as the ordinary
    dense electronic route; its occupied factor and Fock remain experimental.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if (calculator.multiplicity != 1 or not 21 <= len(xyz) <= 512
            or len(calculator.point_charges_electron)):
        raise ValueError("occupied analytic adapter requires 21--512 closed-shell atoms")
    coordinates = xyz * model.ANGSTROM_TO_BOHR
    prepared_model = calculator._model(coordinates, relocate_basis=False)
    (
        basis, _overlap, _dipole, _second, _core, partition,
        atom_parameters, orbital_parameters, _, _,
    ) = prepared_model
    _, overlap, _, _, fock_and_energy = prepare_native_dense_scf_function(
        calculator, xyz, model_result=prepared_model,
    )
    native = calculator._active_native_context
    coordination = np.asarray(calculator._active_coordination)
    interaction = model.shell_interaction_operator(coordinates, partition)
    d4_parameters = calculator._d4_parameters
    sparse_dispersion = len(xyz) >= 64
    if sparse_dispersion:
        d4_cn, _ = model.d4_coordination_sparse(
            coordinates,
            model._covalent_radii()[calculator._atomic_number_array - 1],
            calculator._electronegativities, cutoff_bohr=40.0,
        )
        d4_prepared = model.prepare_d4_geometry_sparse(
            coordinates, d4_cn, calculator._d4_models, d4_parameters,
            cutoff_bohr=50.0,
        )
    else:
        d4_cn = model._d4_coordination(
            calculator.atomic_numbers, coordinates, model._covalent_radii(),
            calculator._electronegativities,
        )
        d4_prepared = model.prepare_d4_geometry(
            coordinates, d4_cn, calculator._d4_models, d4_parameters,
        )
    effective_radii = calculator._base_multipole_radii + (
        5.0 - calculator._base_multipole_radii
    ) / (1.0 + np.exp(-4.0 * (
        coordination - calculator._valence_coordination - 1.2
    )))
    anisotropic_parameters = model.AnisotropicParameters(
        effective_radii,
        np.asarray([item.dipole_xc for item in calculator._element_values]),
        np.asarray([item.quadrupole_xc for item in calculator._element_values]),
    )
    if sparse_dispersion:
        anisotropic_geometry = model.prepare_anisotropic_geometry_sparse(
            coordinates, anisotropic_parameters, cutoff_bohr=64.0,
        )
        neutral_pairs = model.d4_two_body_sparse(
            np.zeros(len(xyz)), d4_prepared, d4_parameters,
        )
        atm = model.atm_three_body_energy_gradient_sparse(
            coordinates, neutral_pairs, cutoff_bohr=25.0,
            s9=d4_parameters.s9, a1=d4_parameters.a1, a2=d4_parameters.a2,
        )
        neutral_d4 = (neutral_pairs, atm)
    else:
        anisotropic_geometry = model.AnisotropicGeometry(*native.anisotropic_geometry(
            coordinates, anisotropic_parameters.damping_radius_bohr,
            anisotropic_parameters.a3, anisotropic_parameters.a5,
        ))
        neutral_d4 = model.d4_two_body(
            coordinates, np.zeros(len(xyz)), d4_cn, calculator._d4_models,
            d4_parameters, prepared=d4_prepared,
        )
        atm = model.atm_three_body_energy_gradient_dense(
            coordinates, neutral_d4,
            model.d4_coordination_number_jacobian(
                calculator.atomic_numbers, coordinates, model._covalent_radii(),
                calculator._electronegativities,
            ),
            s9=d4_parameters.s9, a1=d4_parameters.a1, a2=d4_parameters.a2,
        )
    geometry_energy = (
        (model.repulsion_energy_sparse(
            calculator.atomic_numbers, coordinates, atom_parameters,
            cutoff_bohr=24.0,
        ) if sparse_dispersion else model.repulsion_energy(
            calculator.atomic_numbers, coordinates, atom_parameters,
        )) + atm.energy_hartree
    )

    def analytic_gradient(subspace: OccupiedSubspace | MetricOccupiedSubspace):
        if subspace.shape[0] != overlap.shape[0]:
            raise ValueError("occupied factor does not match the AO basis")
        if isinstance(subspace, MetricOccupiedSubspace):
            if subspace.metric_error(overlap) > 1.0e-8:
                raise ValueError("occupied factor metric did not certify")
        elif subspace.orthogonality_error(overlap) > 1.0e-8:
            raise ValueError("occupied factor is not S-orthonormal")
        density = subspace.density()
        fock, electronic_energy = fock_and_energy(density)
        weighted = subspace.weighted_density(fock)
        observables, anisotropic, _, _ = native.last_scf_state()
        shell_q = np.asarray(observables[0])
        multipoles = model.AtomicMultipoles(*(
            np.asarray(value) for value in observables[1:4]
        ))
        d4 = (
            model.d4_two_body_sparse(
                multipoles.charges, d4_prepared, d4_parameters,
            ) if sparse_dispersion else model.d4_two_body(
                coordinates, multipoles.charges, d4_cn, calculator._d4_models,
                d4_parameters, prepared=d4_prepared,
            )
        )
        gradient = model._analytic_gradient_pair_packed(
            density=density,
            weighted_density=weighted,
            basis=basis,
            coordinates=coordinates,
            atomic_numbers=calculator.atomic_numbers,
            atom_parameters=atom_parameters,
            orbital_parameters=orbital_parameters,
            partition=partition,
            shell_interaction=interaction,
            shell_q=shell_q,
            multipoles=multipoles,
            anisotropic_parameters=anisotropic_parameters,
            anisotropic_geometry=anisotropic_geometry,
            d4=d4,
            d4_parameters=d4_parameters,
            neutral_d4=neutral_d4,
            valence=calculator._valence,
            integrals=calculator._post_scf_pair_integrals,
            spherical_domains=calculator._post_scf_pair_domains,
            element_values=calculator._element_values,
            base_multipole_radii=calculator._base_multipole_radii,
            valence_coordination=calculator._valence_coordination,
            electronegativities=calculator._electronegativities,
            orbital_metadata=calculator._orbital_metadata,
            coordination_override=coordination,
            native_context=native,
            atm_gradient_override=(
                None if sparse_dispersion else atm.explicit_gradient_hartree_per_bohr
            ),
            anisotropic_potentials_override=tuple(
                np.asarray(value) for value in anisotropic[1:]
            ),
        )
        return float(electronic_energy + geometry_energy), np.asarray(gradient)

    return overlap, fock_and_energy, analytic_gradient
