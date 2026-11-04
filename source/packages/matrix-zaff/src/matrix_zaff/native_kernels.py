"""Optional compiled ZAFF kernels with an invariant Python fallback policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from matrix_numerics import NativeBackend, resolve_native_backend


try:
    from . import _zaff_native
except ImportError:
    _zaff_native = None


NATIVE_EXTENSION_NAME = "matrix_zaff._zaff_native"


def native_zaff_backend(workload_size: int, requested: str | None = None) -> NativeBackend:
    """Resolve the portable native ZAFF extension for one workload."""

    return resolve_native_backend(
        extension_available=_zaff_native is not None,
        extension_name=NATIVE_EXTENSION_NAME,
        workload_size=int(workload_size),
        requested=requested,
    )


def native_zaff_build_info() -> dict[str, Any]:
    """Return architecture metadata without making the extension mandatory."""

    if _zaff_native is None:
        return {
            "implementation": "python",
            "extension": NATIVE_EXTENSION_NAME,
            "available": False,
        }
    return {
        **dict(_zaff_native.build_info()),
        "extension": NATIVE_EXTENSION_NAME,
        "available": True,
    }


def direct_gaussian_energy(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
) -> tuple[float, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, count = _zaff_native.direct_gaussian_energy(coordinates, charges, widths)
    return float(energy), int(count)


def direct_gaussian_energy_gradient(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.direct_gaussian_energy_gradient(
        coordinates, charges, widths
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def direct_gaussian_hessian_vector(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    return np.asarray(
        _zaff_native.direct_gaussian_hessian_vector(coordinates, charges, widths, direction),
        dtype=float,
    )


def direct_gaussian_potential(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """Apply the full Gaussian Coulomb matrix without materializing it."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    return np.asarray(
        _zaff_native.direct_gaussian_potential(coordinates, charges, widths),
        dtype=float,
    )


def direct_gaussian_directional_potential(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Differentiate the Gaussian Coulomb matrix-vector product."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    return np.asarray(
        _zaff_native.direct_gaussian_directional_potential(coordinates, charges, widths, direction),
        dtype=float,
    )


def planar_image_potential_gradient(
    coordinates: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    potential, gradient = _zaff_native.planar_image_potential_gradient(
        coordinates,
        charges,
        origin,
        normal,
    )
    return (
        np.asarray(potential, dtype=float),
        np.asarray(gradient, dtype=float),
    )


def planar_image_hessian_vector(
    coordinates: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    direction: np.ndarray,
    image_factor: float,
) -> np.ndarray:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    return np.asarray(
        _zaff_native.planar_image_hessian_vector(
            coordinates,
            charges,
            origin,
            normal,
            direction,
            float(image_factor),
        ),
        dtype=float,
    )


def planar_image_hessian_vectors(
    coordinates: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    directions: np.ndarray,
    image_factor: float,
) -> np.ndarray:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    return np.asarray(
        _zaff_native.planar_image_hessian_vector(
            coordinates,
            charges,
            origin,
            normal,
            directions,
            float(image_factor),
        ),
        dtype=float,
    )


def gaussian_correction_energy(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    pairs: np.ndarray,
) -> tuple[float, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, count = _zaff_native.gaussian_correction_energy(coordinates, charges, widths, pairs)
    return float(energy), int(count)


def gaussian_correction_energy_gradient(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    pairs: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.gaussian_correction_energy_gradient(
        coordinates, charges, widths, pairs
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def gaussian_correction_hessian_vector(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    pairs: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    product, count = _zaff_native.gaussian_correction_hessian_vector(
        coordinates, charges, widths, pairs, direction
    )
    return np.asarray(product, dtype=float), int(count)


def gaussian_correction_potential(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    potential, count = _zaff_native.gaussian_correction_potential(
        coordinates, charges, widths, pairs
    )
    return np.asarray(potential, dtype=float), int(count)


def sonic_local_pairs(coordinates_bohr, charges, widths_bohr, epsilon_hartree,
                      rmin_half_angstrom, pairs, cutoff_angstrom, direction_bohr=None):
    """Compiled E/G/Hv with one radial geometry evaluation per local pair."""
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    return _zaff_native.sonic_local_pairs(
        coordinates_bohr, charges, widths_bohr, epsilon_hartree,
        rmin_half_angstrom, pairs, float(cutoff_angstrom), direction_bohr)


def damped_exppe_energy(
    coordinates_angstrom: np.ndarray,
    epsilon_hartree: np.ndarray,
    rmin_half_angstrom: np.ndarray,
    pairs: np.ndarray,
    cutoff_angstrom: float,
) -> tuple[float, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, count = _zaff_native.damped_exppe_energy(
        coordinates_angstrom,
        epsilon_hartree,
        rmin_half_angstrom,
        pairs,
        float(cutoff_angstrom),
    )
    return float(energy), int(count)


def damped_exppe_energy_gradient(
    coordinates_angstrom: np.ndarray,
    epsilon_hartree: np.ndarray,
    rmin_half_angstrom: np.ndarray,
    pairs: np.ndarray,
    cutoff_angstrom: float,
) -> tuple[float, np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.damped_exppe_energy_gradient(
        coordinates_angstrom,
        epsilon_hartree,
        rmin_half_angstrom,
        pairs,
        float(cutoff_angstrom),
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def damped_exppe_hessian_vector(
    coordinates_angstrom: np.ndarray,
    epsilon_hartree: np.ndarray,
    rmin_half_angstrom: np.ndarray,
    pairs: np.ndarray,
    cutoff_angstrom: float,
    direction_bohr: np.ndarray,
) -> tuple[np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    product, count = _zaff_native.damped_exppe_hessian_vector(
        coordinates_angstrom,
        epsilon_hartree,
        rmin_half_angstrom,
        pairs,
        float(cutoff_angstrom),
        direction_bohr,
    )
    return np.asarray(product, dtype=float), int(count)


# Keep the former names callable for existing clients while exposing the
# physical, undamped kernel under its actual model name.
exppe_energy = damped_exppe_energy
exppe_energy_gradient = damped_exppe_energy_gradient
exppe_hessian_vector = damped_exppe_hessian_vector


def switched_lj_energy_gradient(
    coordinates_bohr: np.ndarray,
    pairs: np.ndarray,
    sigma_bohr: float,
    epsilon_hartree: float,
    switch_bohr: float,
    cutoff_bohr: float,
) -> tuple[float, np.ndarray, int]:
    """Evaluate a C2-switched 12-6 Lennard-Jones pair list."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.switched_lj_energy_gradient(
        coordinates_bohr,
        pairs,
        float(sigma_bohr),
        float(epsilon_hartree),
        float(switch_bohr),
        float(cutoff_bohr),
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def morse_bond_energy(
    coordinates_angstrom: np.ndarray,
    bonds: np.ndarray,
    depths_hartree: np.ndarray,
    alphas_per_angstrom: np.ndarray,
    references_angstrom: np.ndarray,
) -> tuple[float, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, count = _zaff_native.morse_bond_energy(
        coordinates_angstrom,
        bonds,
        depths_hartree,
        alphas_per_angstrom,
        references_angstrom,
    )
    return float(energy), int(count)


def morse_bond_energy_gradient(
    coordinates_angstrom: np.ndarray,
    bonds: np.ndarray,
    depths_hartree: np.ndarray,
    alphas_per_angstrom: np.ndarray,
    references_angstrom: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.morse_bond_energy_gradient(
        coordinates_angstrom,
        bonds,
        depths_hartree,
        alphas_per_angstrom,
        references_angstrom,
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def morse_bond_hessian_vector(
    coordinates_angstrom: np.ndarray,
    bonds: np.ndarray,
    depths_hartree: np.ndarray,
    alphas_per_angstrom: np.ndarray,
    references_angstrom: np.ndarray,
    direction_bohr: np.ndarray,
) -> tuple[np.ndarray, int]:
    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    product, count = _zaff_native.morse_bond_hessian_vector(
        coordinates_angstrom,
        bonds,
        depths_hartree,
        alphas_per_angstrom,
        references_angstrom,
        direction_bohr,
    )
    return np.asarray(product, dtype=float), int(count)


def gaussian_bond_energy(
    coordinates_angstrom: np.ndarray,
    bonds: np.ndarray,
    amplitudes_hartree: np.ndarray,
    exponents_per_angstrom2: np.ndarray,
    references_angstrom: np.ndarray,
) -> tuple[float, int]:
    """Evaluate native GFN-FF Gaussian covalent wells."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, count = _zaff_native.gaussian_bond_energy(
        coordinates_angstrom,
        bonds,
        amplitudes_hartree,
        exponents_per_angstrom2,
        references_angstrom,
    )
    return float(energy), int(count)


def gaussian_bond_energy_gradient(
    coordinates_angstrom: np.ndarray,
    bonds: np.ndarray,
    amplitudes_hartree: np.ndarray,
    exponents_per_angstrom2: np.ndarray,
    references_angstrom: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """Evaluate native GFN-FF Gaussian wells and analytic gradients."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.gaussian_bond_energy_gradient(
        coordinates_angstrom,
        bonds,
        amplitudes_hartree,
        exponents_per_angstrom2,
        references_angstrom,
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def gaussian_bond_hessian_vector(
    coordinates_angstrom: np.ndarray,
    bonds: np.ndarray,
    amplitudes_hartree: np.ndarray,
    exponents_per_angstrom2: np.ndarray,
    references_angstrom: np.ndarray,
    direction_bohr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply the analytic GFN-FF Gaussian-bond Hessian."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    product, count = _zaff_native.gaussian_bond_hessian_vector(
        coordinates_angstrom,
        bonds,
        amplitudes_hartree,
        exponents_per_angstrom2,
        references_angstrom,
        direction_bohr,
    )
    return np.asarray(product, dtype=float), int(count)


def dynamic_gaussian_bond_energy_gradient(
    coordinates_angstrom: np.ndarray,
    coordination_pairs: np.ndarray,
    coordination_radii_angstrom: np.ndarray,
    bonds: np.ndarray,
    amplitudes_hartree: np.ndarray,
    exponents_per_angstrom2: np.ndarray,
    reference_bases_angstrom: np.ndarray,
    reference_cn_coefficients_angstrom: np.ndarray,
    cn_max: float = 4.4,
    steepness: float = 7.5,
) -> tuple[float, np.ndarray, int]:
    """Evaluate sparse dynamic-CN GFN-FF bonds in the native kernel."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.dynamic_gaussian_bond_energy_gradient(
        coordinates_angstrom,
        coordination_pairs,
        coordination_radii_angstrom,
        bonds,
        amplitudes_hartree,
        exponents_per_angstrom2,
        reference_bases_angstrom,
        reference_cn_coefficients_angstrom,
        float(cn_max),
        float(steepness),
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def dynamic_gaussian_bond_shared_coordination(
    coordinates_angstrom: np.ndarray,
    coordination_pairs: np.ndarray,
    coordination_radii_angstrom: np.ndarray,
    bonds: np.ndarray,
    amplitudes_hartree: np.ndarray,
    exponents_per_angstrom2: np.ndarray,
    reference_bases_angstrom: np.ndarray,
    reference_cn_coefficients_angstrom: np.ndarray,
    cn_max: float = 4.4,
    steepness: float = 7.5,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Bond E/G (atomic units) plus CN atom/pair buffers (Angstrom derivatives)."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, atoms, pairs = _zaff_native.dynamic_gaussian_bond_shared_coordination(
        coordinates_angstrom, coordination_pairs, coordination_radii_angstrom,
        bonds, amplitudes_hartree, exponents_per_angstrom2,
        reference_bases_angstrom, reference_cn_coefficients_angstrom,
        float(cn_max), float(steepness),
    )
    return float(energy), np.asarray(gradient), np.asarray(atoms), np.asarray(pairs)


def dynamic_gaussian_bond_hessian_vector(
    coordinates_angstrom: np.ndarray,
    coordination_pairs: np.ndarray,
    coordination_radii_angstrom: np.ndarray,
    bonds: np.ndarray,
    amplitudes_hartree: np.ndarray,
    exponents_per_angstrom2: np.ndarray,
    reference_bases_angstrom: np.ndarray,
    reference_cn_coefficients_angstrom: np.ndarray,
    direction_bohr: np.ndarray,
    cn_max: float = 4.4,
    steepness: float = 7.5,
) -> tuple[np.ndarray, int]:
    """Apply the sparse dynamic-CN GFN-FF bond Hessian."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    parameters = (
        coordinates_angstrom,
        coordination_pairs,
        coordination_radii_angstrom,
        bonds,
        amplitudes_hartree,
        exponents_per_angstrom2,
        reference_bases_angstrom,
        reference_cn_coefficients_angstrom,
        float(cn_max),
        float(steepness),
    )
    product, count = _zaff_native.dynamic_gaussian_bond_hessian_vector(parameters, direction_bohr)
    return np.asarray(product, dtype=float), int(count)


def gfnff_repulsion_energy_gradient(
    coordinates_bohr: np.ndarray,
    pairs: np.ndarray,
    amplitudes_hartree_bohr: np.ndarray,
    exponents_per_bohr3over2: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """Evaluate exact GFN-FF ``A exp(-alpha r**1.5) / r`` pairs."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.gfnff_repulsion_energy_gradient(
        coordinates_bohr,
        pairs,
        amplitudes_hartree_bohr,
        exponents_per_bohr3over2,
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def gfnff_repulsion_hessian_vector(
    coordinates_bohr: np.ndarray,
    pairs: np.ndarray,
    amplitudes_hartree_bohr: np.ndarray,
    exponents_per_bohr3over2: np.ndarray,
    direction_bohr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply the exact sparse GFN-FF repulsion Hessian."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    parameters = (
        coordinates_bohr,
        pairs,
        amplitudes_hartree_bohr,
        exponents_per_bohr3over2,
    )
    product, count = _zaff_native.gfnff_repulsion_hessian_vector(parameters, direction_bohr)
    return np.asarray(product, dtype=float), int(count)


def gfnff_d3_reference_derivatives(
    coordination_numbers: np.ndarray,
    pairs: np.ndarray,
    reference_counts: np.ndarray,
    reference_cn_padded: np.ndarray,
    pair_c6_padded: np.ndarray,
    weighting_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate pair C6 and its two CN derivatives in compiled code."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    result = _zaff_native.gfnff_d3_reference_derivatives(
        coordination_numbers,
        pairs,
        reference_counts,
        reference_cn_padded,
        pair_c6_padded,
        float(weighting_factor),
    )
    return tuple(np.asarray(value, dtype=float) for value in result)


def gfnff_valence_energy_gradient(
    coordinates_bohr: np.ndarray,
    angle_atoms: np.ndarray,
    angle_parameters: np.ndarray,
    torsion_atoms: np.ndarray,
    torsion_parameters: np.ndarray,
    special_torsion_atoms: np.ndarray,
    special_torsion_amplitudes: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """Evaluate exact damped GFN-FF angles and torsions."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.gfnff_valence_energy_gradient(
        coordinates_bohr,
        angle_atoms,
        angle_parameters,
        torsion_atoms,
        torsion_parameters,
        special_torsion_atoms,
        special_torsion_amplitudes,
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def gfnff_valence_hessian_vector(
    coordinates_bohr: np.ndarray,
    angle_atoms: np.ndarray,
    angle_parameters: np.ndarray,
    torsion_atoms: np.ndarray,
    torsion_parameters: np.ndarray,
    special_torsion_atoms: np.ndarray,
    special_torsion_amplitudes: np.ndarray,
    direction_bohr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply the exact GFN-FF valence Hessian without assembling it."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    parameters = (
        coordinates_bohr,
        angle_atoms,
        angle_parameters,
        torsion_atoms,
        torsion_parameters,
        special_torsion_atoms,
        special_torsion_amplitudes,
    )
    product, count = _zaff_native.gfnff_valence_hessian_vector(parameters, direction_bohr)
    return np.asarray(product, dtype=float), int(count)


def gfnff_nci_energy_gradient(
    coordinates_bohr: np.ndarray,
    hydrogen_bond_atoms: np.ndarray,
    hydrogen_bond_parameters: np.ndarray,
    halogen_bond_atoms: np.ndarray,
    halogen_bond_parameters: np.ndarray,
    bonded_atm_atoms: np.ndarray,
    bonded_atm_amplitudes: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """Evaluate sparse GFN-FF H/X-bond and bonded-ATM terms."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.gfnff_nci_energy_gradient(
        coordinates_bohr,
        hydrogen_bond_atoms,
        hydrogen_bond_parameters,
        halogen_bond_atoms,
        halogen_bond_parameters,
        bonded_atm_atoms,
        bonded_atm_amplitudes,
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def gfnff_nci_hessian_vector(
    coordinates_bohr: np.ndarray,
    hydrogen_bond_atoms: np.ndarray,
    hydrogen_bond_parameters: np.ndarray,
    halogen_bond_atoms: np.ndarray,
    halogen_bond_parameters: np.ndarray,
    bonded_atm_atoms: np.ndarray,
    bonded_atm_amplitudes: np.ndarray,
    direction_bohr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply the sparse GFN-FF NCI Hessian without assembling it."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    parameters = (
        coordinates_bohr,
        hydrogen_bond_atoms,
        hydrogen_bond_parameters,
        halogen_bond_atoms,
        halogen_bond_parameters,
        bonded_atm_atoms,
        bonded_atm_amplitudes,
    )
    product, count = _zaff_native.gfnff_nci_hessian_vector(parameters, direction_bohr)
    return np.asarray(product, dtype=float), int(count)


def local_valence_energy(
    coordinates_angstrom: np.ndarray,
    angle_atoms: np.ndarray,
    angle_parameters: np.ndarray,
    torsion_atoms: np.ndarray,
    torsion_parameters: np.ndarray,
    term_offsets: np.ndarray,
    terms: np.ndarray,
) -> tuple[float, int]:
    """Evaluate bond-order-damped angles and torsions in native Cartesian form."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, count = _zaff_native.local_valence_energy(
        coordinates_angstrom,
        angle_atoms,
        angle_parameters,
        torsion_atoms,
        torsion_parameters,
        term_offsets,
        terms,
    )
    return float(energy), int(count)


def local_valence_energy_gradient(
    coordinates_angstrom: np.ndarray,
    angle_atoms: np.ndarray,
    angle_parameters: np.ndarray,
    torsion_atoms: np.ndarray,
    torsion_parameters: np.ndarray,
    term_offsets: np.ndarray,
    terms: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """Evaluate local valence energy and analytic Cartesian gradient."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    energy, gradient, count = _zaff_native.local_valence_energy_gradient(
        coordinates_angstrom,
        angle_atoms,
        angle_parameters,
        torsion_atoms,
        torsion_parameters,
        term_offsets,
        terms,
    )
    return float(energy), np.asarray(gradient, dtype=float), int(count)


def local_valence_hessian_vector(
    coordinates_angstrom: np.ndarray,
    angle_atoms: np.ndarray,
    angle_parameters: np.ndarray,
    torsion_atoms: np.ndarray,
    torsion_parameters: np.ndarray,
    term_offsets: np.ndarray,
    terms: np.ndarray,
    direction_bohr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply the exact local-valence Hessian in the supplied Bohr direction."""

    if _zaff_native is None:
        raise RuntimeError(f"{NATIVE_EXTENSION_NAME} is unavailable")
    product, count = _zaff_native.local_valence_hessian_vector(
        coordinates_angstrom,
        angle_atoms,
        angle_parameters,
        torsion_atoms,
        torsion_parameters,
        term_offsets,
        terms,
        direction_bohr,
    )
    return np.asarray(product, dtype=float), int(count)


__all__ = [
    "NATIVE_EXTENSION_NAME",
    "direct_gaussian_energy",
    "direct_gaussian_energy_gradient",
    "direct_gaussian_hessian_vector",
    "direct_gaussian_potential",
    "direct_gaussian_directional_potential",
    "damped_exppe_energy",
    "damped_exppe_energy_gradient",
    "damped_exppe_hessian_vector",
    "exppe_energy",
    "exppe_energy_gradient",
    "exppe_hessian_vector",
    "dynamic_gaussian_bond_energy_gradient",
    "dynamic_gaussian_bond_shared_coordination",
    "dynamic_gaussian_bond_hessian_vector",
    "gfnff_d3_reference_derivatives",
    "gfnff_nci_energy_gradient",
    "gfnff_nci_hessian_vector",
    "gfnff_valence_energy_gradient",
    "gfnff_valence_hessian_vector",
    "gfnff_repulsion_energy_gradient",
    "gfnff_repulsion_hessian_vector",
    "gaussian_correction_energy",
    "gaussian_correction_energy_gradient",
    "gaussian_correction_hessian_vector",
    "gaussian_correction_potential",
    "local_valence_energy",
    "local_valence_energy_gradient",
    "local_valence_hessian_vector",
    "morse_bond_energy",
    "morse_bond_energy_gradient",
    "morse_bond_hessian_vector",
    "native_zaff_backend",
    "native_zaff_build_info",
    "planar_image_hessian_vector",
    "planar_image_hessian_vectors",
    "planar_image_potential_gradient",
    "switched_lj_energy_gradient",
]
