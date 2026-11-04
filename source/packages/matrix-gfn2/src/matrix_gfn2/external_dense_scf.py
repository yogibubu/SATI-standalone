"""Prepare the existing native dense GFN2 functional without solving SCF.

Experimental adapter for fragment-orbital solvers. The 21--63-atom route
uses dense D4 preparation; larger dense electronic systems use the same
packed dispersion preparation as the production SCF.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import calculator as model_api


def prepare_native_dense_scf_function(
    calculator, coordinates_angstrom: np.ndarray, *, model_result=None,
):
    """Return model and Fock/energy callback; perform no global eigensolve."""
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 21 <= len(xyz) <= 512:
        raise ValueError("external dense SCF preparation supports 21--512 atoms")
    coordinates = xyz * model_api.ANGSTROM_TO_BOHR
    if model_result is None:
        model_result = calculator._model(coordinates, relocate_basis=False)
    basis, overlap, dipole, second, core, partition, *_ = model_result
    native = calculator._active_native_context
    if native is None:
        raise RuntimeError("native integral context unavailable")
    if not getattr(calculator, "_active_native_model_prepared", False):
        operators = (
            model_api.prepare_cumulative_multipole_operators(
                overlap, dipole, second, coordinates, basis.spherical_atom,
            )
            if dipole is not None and second is not None
            else model_api.prepare_pair_cumulative_multipole_operators(
                calculator._post_scf_pair_integrals,
                coordinates, basis.spherical_atom,
            )
        )
        native.update_dense_model(
            core, overlap, operators, partition.orbital_shell,
            partition.reference_population, calculator._valence,
        )
    coordination = calculator._active_coordination
    if coordination is None:
        raise RuntimeError("coordination state unavailable")
    interaction = model_api.shell_interaction_operator(coordinates, partition)
    if not isinstance(interaction, np.ndarray):
        raise RuntimeError("external SCF requires the dense interaction matrix")
    effective_radii = calculator._base_multipole_radii + (
        5.0 - calculator._base_multipole_radii
    ) / (1.0 + np.exp(-4.0 * (
        coordination - calculator._valence_coordination - 1.2
    )))
    anisotropic_parameters = model_api.AnisotropicParameters(
        effective_radii,
        np.asarray([item.dipole_xc for item in calculator._element_values]),
        np.asarray([item.quadrupole_xc for item in calculator._element_values]),
    )
    sparse_dispersion = len(xyz) >= 64
    anisotropic_geometry = (
        model_api.prepare_anisotropic_geometry_sparse(
            coordinates, anisotropic_parameters, cutoff_bohr=64.0,
        )
        if sparse_dispersion else model_api.AnisotropicGeometry(
            *native.anisotropic_geometry(
                coordinates, anisotropic_parameters.damping_radius_bohr,
                anisotropic_parameters.a3, anisotropic_parameters.a5,
            )
        )
    )
    d4_parameters = calculator._d4_parameters
    if sparse_dispersion:
        d4_cn, _ = model_api.d4_coordination_sparse(
            coordinates,
            model_api._covalent_radii()[calculator._atomic_number_array - 1],
            calculator._electronegativities, cutoff_bohr=40.0,
        )
        d4_prepared = model_api.prepare_d4_geometry_sparse(
            coordinates, d4_cn, calculator._d4_models, d4_parameters,
            cutoff_bohr=50.0,
        )
        d4_rows = np.asarray(d4_prepared.rows, dtype=np.int64)
        d4_columns = np.asarray(d4_prepared.columns, dtype=np.int64)
        d4_ratio = np.asarray(d4_prepared.ratio, dtype=float)
        d4_pair_factor = np.asarray(d4_prepared.pair_factor, dtype=float)
    else:
        d4_cn = model_api._d4_coordination(
            calculator.atomic_numbers, coordinates, model_api._covalent_radii(),
            calculator._electronegativities,
        )
        d4_prepared = model_api.prepare_d4_geometry(
            coordinates, d4_cn, calculator._d4_models, d4_parameters,
        )
        d4_rows, d4_columns = model_api.triangular_indices(
            len(coordinates), diagonal=False,
        )
        d4_rows = np.asarray(d4_rows, dtype=np.int64)
        d4_columns = np.asarray(d4_columns, dtype=np.int64)
        d4_ratio = np.asarray(d4_prepared.ratio[d4_rows, d4_columns], dtype=float)
        d4_pair_factor = np.asarray(
            d4_prepared.pair_factor[d4_rows, d4_columns], dtype=float,
        )
    native.configure_scf({
        "interaction": np.asarray(interaction, dtype=float),
        "third_order": np.asarray(partition.third_order_hartree, dtype=float),
        "anis_dipole_xc": np.asarray(anisotropic_parameters.dipole_xc, dtype=float),
        "anis_quadrupole_xc": np.asarray(anisotropic_parameters.quadrupole_xc, dtype=float),
        "anis_left": np.asarray(anisotropic_geometry.left, dtype=np.int64),
        "anis_right": np.asarray(anisotropic_geometry.right, dtype=np.int64),
        "anis_displacement": np.asarray(anisotropic_geometry.displacement, dtype=float),
        "anis_f3": np.asarray(anisotropic_geometry.f3, dtype=float),
        "anis_f5": np.asarray(anisotropic_geometry.f5, dtype=float),
        "anis_distance2": np.asarray(anisotropic_geometry.distance2, dtype=float),
        "d4_frequency_weights": np.asarray(d4_prepared.frequency_weights, dtype=float),
        "d4_weights": np.asarray(d4_prepared.weights, dtype=float),
        "d4_weight_derivatives": np.asarray(d4_prepared.weight_derivatives, dtype=float),
        "d4_reference_charges": np.asarray(d4_prepared.reference_charges, dtype=float),
        "d4_references": np.asarray(d4_prepared.references, dtype=float),
        "d4_mask": np.asarray(d4_prepared.mask, dtype=float),
        "d4_effective": np.asarray(d4_prepared.effective, dtype=float),
        "d4_hardness": np.asarray(d4_prepared.hardness, dtype=float),
        "d4_rows": d4_rows,
        "d4_columns": d4_columns,
        "d4_ratio": d4_ratio,
        "d4_pair_factor": d4_pair_factor,
    })
    return basis, overlap, core, partition, native.scf_state_fock


def prepare_matched_dense_response_kernel(calculator, orbital_state):
    """Use the density-response kernel with this dense SCF route's cutoffs."""
    kernel = calculator.density_response_kernel(orbital_state)
    coordinates = np.asarray(orbital_state.coordinates_bohr, dtype=float)
    if len(coordinates) < 64:
        return kernel
    coordination, _ = model_api.d4_coordination_sparse(
        coordinates,
        model_api._covalent_radii()[calculator._atomic_number_array - 1],
        calculator._electronegativities, cutoff_bohr=40.0,
    )
    prepared = model_api.prepare_d4_geometry_sparse(
        coordinates, coordination, calculator._d4_models,
        calculator._d4_parameters, cutoff_bohr=50.0,
    )
    anisotropic_geometry = model_api.prepare_anisotropic_geometry_sparse(
        coordinates, kernel.anisotropic_parameters, cutoff_bohr=64.0,
    )
    return replace(
        kernel, d4_prepared=prepared, d4_coordination=coordination,
        anisotropic_geometry=anisotropic_geometry,
    )
