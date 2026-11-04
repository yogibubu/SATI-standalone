"""Native GFN2 ALPB electrostatic primitives.

This module deliberately exposes the electrostatic reaction matrix separately
from the full solvent model.  A full ALPB calculation also requires CDS,
hydrogen-bond and reference-state terms and SCC coupling; callers must not
present this component alone as an ALPB total energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json

import numpy as np


ANGSTROM_TO_BOHR = 1.8897261254578281
ALPB_ALPHA = 0.571412  # tblite ALPB implementation
P16_ZETA = 1.028
KCAL_TO_HARTREE = 1.0 / 627.5094740631


@lru_cache(maxsize=1)
def _solvent_data() -> dict:
    return json.loads(
        files("matrix_gfn2").joinpath("data/alpb_gfn2_solvents.json").read_text()
    )


@lru_cache(maxsize=1)
def _surface_grid() -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(
        files("matrix_gfn2").joinpath("data/alpb_lebedev_230.json").read_text()
    )
    return np.asarray(data["xyz"], dtype=float), np.asarray(data["weights"], dtype=float)


@dataclass(frozen=True)
class ALPBParameters:
    solvent: str
    dielectric: float
    born_scale: float
    born_offset_bohr: float
    descreening: np.ndarray
    radii_bohr: np.ndarray
    probe_bohr: float
    surface_tension_hartree_per_bohr2: np.ndarray
    hydrogen_bond_sqrt_kcal: np.ndarray
    shift_hartree: float
    molar_mass: float
    density: float

    @classmethod
    def available_gfn2_solvents(cls) -> tuple[str, ...]:
        return tuple(sorted(_solvent_data()["solvents"]))

    @classmethod
    def gfn2(cls, solvent: str) -> "ALPBParameters":
        collection = _solvent_data()
        aliases = {
            "h2o": "water",
            "dichloromethane": "ch2cl2",
            "methylenechloride": "ch2cl2",
            "chloroform": "chcl3",
            "carbondisulfide": "cs2",
            "dimethylformamide": "dmf",
            "dimethylsulfoxide": "dmso",
            "diethylether": "ether",
            "ethylacetate": "ethylacetate",
            "nhexane": "hexane",
            "tetrahydrofuran": "thf",
        }
        key = "".join(character for character in solvent.lower() if character.isalnum())
        key = aliases.get(key, key)
        if key not in collection["solvents"]:
            raise ValueError(
                f"GFN2-ALPB solvent {solvent!r} is unavailable; "
                f"choose from {', '.join(sorted(collection['solvents']))}"
            )
        data = collection["solvents"][key]
        return cls(
            solvent=key,
            dielectric=float(data["dielectric"]),
            born_scale=float(data["born_scale"]),
            born_offset_bohr=float(data["born_offset_angstrom_tenth"])
            * 0.1 * ANGSTROM_TO_BOHR,
            descreening=np.asarray(data["descreening"], dtype=float),
            radii_bohr=np.asarray(collection["d3_vdw_radius_angstrom"], dtype=float)
            * ANGSTROM_TO_BOHR,
            probe_bohr=float(data["probe_angstrom"]) * ANGSTROM_TO_BOHR,
            surface_tension_hartree_per_bohr2=np.asarray(
                data["surface_tension_scaled"], dtype=float
            ) * 1.0e-5,
            hydrogen_bond_sqrt_kcal=np.asarray(
                data["hydrogen_bond_sqrt_kcal"], dtype=float
            ),
            shift_hartree=float(data["shift_kcal_mol"]) * KCAL_TO_HARTREE,
            molar_mass=float(data["molar_mass"]),
            density=float(data["density"]),
        )

    @classmethod
    def gfn2_methanol(cls) -> "ALPBParameters":
        return cls.gfn2("methanol")


def _pair_descreening(distance: float, cavity: float, descreen: float) -> float:
    """Directional OBC-II descreening integral for one pair of spheres."""
    if distance <= 0.0:
        raise ValueError("coincident ALPB atom centers")
    if distance >= cavity + descreen:
        ap = distance + descreen
        am = distance - descreen
        return descreen / (ap * am) + 0.5 * np.log(am / ap) / distance
    if distance + descreen <= cavity:
        return 0.0
    ap = distance + descreen
    am = distance - descreen
    inverse_cavity = 1.0 / cavity
    inverse_ap = 1.0 / ap
    ratio = ap * inverse_cavity
    return (
        inverse_cavity
        - inverse_ap
        + 0.5 / distance
        * (0.5 * am * (inverse_ap - inverse_cavity * ratio) - np.log(ratio))
    )


def _pair_descreening_derivative_coefficient(
    distance: float, cavity: float, descreen: float
) -> float:
    """Coefficient multiplying the interatomic displacement in dPsi/dR."""
    if distance >= cavity + descreen:
        ap = distance + descreen
        am = distance - descreen
        ab = ap * am
        rhab = descreen / ab
        logarithm = 0.5 * np.log(am / ap) / distance
        return -2.0 * rhab / ab + (rhab - logarithm) / distance**2
    if distance + descreen <= cavity:
        return 0.0
    ap = distance + descreen
    am = distance - descreen
    inverse_cavity = 1.0 / cavity
    inverse_ap = 1.0 / ap
    ratio = ap * inverse_cavity
    half_inverse_distance = 0.5 / distance
    coefficient = inverse_ap**2 * (
        1.0 - 0.25 * am / distance * (1.0 + ratio**2)
    )
    coefficient += descreen * half_inverse_distance**2 * (
        inverse_ap - inverse_cavity * ratio
    )
    coefficient += half_inverse_distance * (
        np.log(ratio) / distance - inverse_ap
    )
    return coefficient / distance


def born_radii_bohr(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    parameters: ALPBParameters,
) -> np.ndarray:
    """Geometry-dependent OBC-II Born radii with GFN2 solvent parameters."""
    return born_radii_and_jacobian_bohr(
        coordinates_bohr, atomic_numbers, parameters, compute_jacobian=False
    )[0]


def born_radii_and_jacobian_bohr(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    parameters: ALPBParameters,
    *,
    compute_jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return Born radii and their analytic Cartesian Jacobian."""
    xyz = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    if xyz.shape != (len(numbers), 3):
        raise ValueError("coordinates must have shape (natoms, 3)")
    if np.any(numbers < 1) or np.any(numbers > len(parameters.radii_bohr)):
        raise ValueError("ALPB parameters do not cover an atomic number")
    radii = parameters.radii_bohr[numbers - 1]
    rho = radii * parameters.descreening[numbers - 1]
    cavity = radii - parameters.born_offset_bohr
    psi = np.zeros(len(numbers))
    dpsi = np.zeros((len(numbers), len(numbers), 3)) if compute_jacobian else None
    cutoff_bohr = 35.0 * ANGSTROM_TO_BOHR
    for i in range(len(numbers)):
        for j in range(i):
            distance = float(np.linalg.norm(xyz[i] - xyz[j]))
            if distance > cutoff_bohr:
                continue
            psi[i] += _pair_descreening(distance, radii[i], rho[j])
            psi[j] += _pair_descreening(distance, radii[j], rho[i])
            if dpsi is not None:
                vec = xyz[i] - xyz[j]
                coefficient_i = _pair_descreening_derivative_coefficient(
                    distance, radii[i], rho[j]
                )
                coefficient_j = _pair_descreening_derivative_coefficient(
                    distance, radii[j], rho[i]
                )
                dpsi[i, i] += coefficient_i * vec
                dpsi[i, j] -= coefficient_i * vec
                dpsi[j, j] -= coefficient_j * vec
                dpsi[j, i] += coefficient_j * vec
    scaled = 0.5 * cavity * psi
    argument = scaled * (1.0 + scaled * (4.85 * scaled - 0.8))
    inverse_born = 1.0 / cavity - np.tanh(argument) / radii
    if np.any(inverse_born <= 0.0):
        raise ValueError("nonpositive inverse Born radius")
    born = parameters.born_scale / inverse_born
    if dpsi is None:
        return born, None
    argument_derivative = 0.5 * cavity * (
        1.0 - 1.6 * scaled + 14.55 * scaled**2
    )
    radius_derivative = (
        parameters.born_scale
        * (1.0 / radii)
        * (1.0 - np.tanh(argument) ** 2)
        * argument_derivative
        / inverse_born**2
    )
    return born, radius_derivative[:, None, None] * dpsi


def _electrostatic_size_bohr(
    coordinates_bohr: np.ndarray, radii_bohr: np.ndarray
) -> float:
    volumes = radii_bohr**3
    total = float(np.sum(volumes))
    center = np.sum(volumes[:, None] * coordinates_bohr, axis=0) / total
    displacement = coordinates_bohr - center
    second_moment = np.einsum("i,ij,ik->jk", volumes, displacement, displacement)
    trace_term = np.sum(volumes * np.sum(displacement**2, axis=1))
    sphere_term = 0.4 * np.sum(volumes * radii_bohr**2)
    inertia = (trace_term + sphere_term) * np.eye(3) - second_moment
    determinant = float(np.linalg.det(inertia))
    if determinant <= 0.0:
        raise ValueError("nonpositive ALPB shape determinant")
    return float(np.sqrt(np.cbrt(determinant) / (0.4 * total)))


def _electrostatic_size_and_gradient_bohr(
    coordinates_bohr: np.ndarray, radii_bohr: np.ndarray
) -> tuple[float, np.ndarray]:
    volumes = radii_bohr**3
    total = float(np.sum(volumes))
    center = np.sum(volumes[:, None] * coordinates_bohr, axis=0) / total
    displacement = coordinates_bohr - center
    second_moment = np.einsum("i,ij,ik->jk", volumes, displacement, displacement)
    trace_term = np.sum(volumes * np.sum(displacement**2, axis=1))
    sphere_term = 0.4 * np.sum(volumes * radii_bohr**2)
    inertia = (trace_term + sphere_term) * np.eye(3) - second_moment
    determinant = float(np.linalg.det(inertia))
    if determinant <= 0.0:
        raise ValueError("nonpositive ALPB shape determinant")
    size = float(np.sqrt(np.cbrt(determinant) / (0.4 * total)))
    inverse = np.linalg.inv(inertia)
    gradient = size * volumes[:, None] / 3.0 * (
        np.trace(inverse) * displacement - displacement @ inverse
    )
    return size, gradient


def alpb_polar_matrix_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    parameters: ALPBParameters,
) -> np.ndarray:
    """Atomic charge reaction matrix for the GFN2 P16 ALPB polar term."""
    xyz = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    born = born_radii_bohr(xyz, numbers, parameters)
    return _polar_matrix_from_born(xyz, numbers, parameters, born)


def _polar_matrix_from_born(
    xyz: np.ndarray,
    numbers: np.ndarray,
    parameters: ALPBParameters,
    born: np.ndarray,
) -> np.ndarray:
    alpha_beta = ALPB_ALPHA / parameters.dielectric
    dielectric_factor = (1.0 / parameters.dielectric - 1.0) / (1.0 + alpha_beta)
    displacement = xyz[:, None, :] - xyz[None, :, :]
    distance = np.linalg.norm(displacement, axis=-1)
    radius_pair = np.sqrt(born[:, None] * born[None, :])
    p16 = (1.0 + P16_ZETA * distance / (16.0 * radius_pair)) ** -16
    denominator = distance + radius_pair * p16
    matrix = dielectric_factor / denominator
    size = _electrostatic_size_bohr(xyz, parameters.radii_bohr[numbers - 1])
    matrix += dielectric_factor * alpha_beta / size
    return 0.5 * (matrix + matrix.T)


def alpb_polar_energy_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    charges_electron: np.ndarray,
    parameters: ALPBParameters,
) -> float:
    """Return only the ALPB polar term, not the full solvation energy."""
    charges = np.asarray(charges_electron, dtype=float)
    matrix = alpb_polar_matrix_hartree(
        coordinates_bohr, atomic_numbers, parameters
    )
    if charges.shape != (matrix.shape[0],):
        raise ValueError("atomic charge count does not match coordinates")
    return 0.5 * float(charges @ matrix @ charges)


def alpb_scc_interaction_matrix_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    parameters: ALPBParameters,
) -> tuple[np.ndarray, float]:
    """Quadratic SCC charge kernel and charge-independent ALPB contribution."""
    numbers = np.asarray(atomic_numbers, dtype=int)
    matrix = alpb_polar_matrix_hartree(
        coordinates_bohr, numbers, parameters
    )
    surface = alpb_surface_areas_bohr2(
        coordinates_bohr, numbers, parameters
    )
    surface_radii = parameters.radii_bohr[numbers - 1] + parameters.probe_bohr
    hbond = -KCAL_TO_HARTREE * (
        parameters.hydrogen_bond_sqrt_kcal[numbers - 1] ** 2
    ) / (4.0 * np.pi * surface_radii**2)
    matrix[np.diag_indices_from(matrix)] += 2.0 * hbond * surface
    nondensity = (
        float(surface @ parameters.surface_tension_hartree_per_bohr2[numbers - 1])
        + parameters.shift_hartree
    )
    return matrix, nondensity


def alpb_surface_areas_bohr2(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    parameters: ALPBParameters,
) -> np.ndarray:
    """230-point solvent-accessible areas with tblite's smooth switching."""
    return alpb_surface_areas_and_jacobian_bohr2(
        coordinates_bohr, atomic_numbers, parameters, compute_jacobian=False
    )[0]


def alpb_surface_areas_and_jacobian_bohr2(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    parameters: ALPBParameters,
    *,
    compute_jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Surface areas and their analytic derivatives by atomic position."""
    xyz = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    if xyz.shape != (len(numbers), 3):
        raise ValueError("coordinates must have shape (natoms, 3)")
    angular_points, angular_weights = _surface_grid()
    radii = parameters.radii_bohr[numbers - 1] + parameters.probe_bohr
    smoothing = 0.3 * ANGSTROM_TO_BOHR
    cubic = -1.0 / (4.0 * smoothing**3)
    def primitive(value: np.ndarray) -> np.ndarray:
        bracket = (
            0.25 / smoothing
            + 3.0 * cubic
            * (0.2 * value**2 - 0.5 * value * radii + radii**2 / 3.0)
        )
        return bracket * value**3
    radial_weights = primitive(radii + smoothing) - primitive(radii - smoothing)
    center_distance = np.linalg.norm(
        xyz[:, None, :] - xyz[None, :, :], axis=-1,
    )
    areas = np.zeros(len(numbers))
    jacobian = np.zeros((len(numbers), len(numbers), 3)) if compute_jacobian else None
    for i, radius in enumerate(radii):
        points = xyz[i] + radius * angular_points
        other = (
            (np.arange(len(numbers)) != i)
            & (center_distance[i] < radius + radii + smoothing)
        )
        if not np.any(other):
            areas[i] = radial_weights[i] * float(np.sum(angular_weights))
            continue
        displacement = points[None, :, :] - xyz[other, None, :]
        distances = np.linalg.norm(displacement, axis=-1)
        other_radii = radii[other, None]
        reduced = distances - other_radii
        surface_switch = np.where(
            reduced <= -smoothing,
            0.0,
            np.where(
                reduced >= smoothing,
                1.0,
                0.5 + 0.75 * reduced / smoothing
                - 0.25 * (reduced / smoothing) ** 3,
            ),
        )
        exposure = np.prod(surface_switch, axis=0)
        exposure[exposure <= 1.0e-6] = 0.0
        areas[i] = radial_weights[i] * float(angular_weights @ exposure)
        if jacobian is not None:
            other_indices = np.flatnonzero(other)
            active = (
                (reduced > -smoothing) & (reduced < smoothing)
                & (exposure[None, :] > 0.0)
            )
            normalized = reduced / smoothing
            switch_derivative = 0.75 / smoothing * (1.0 - normalized**2)
            exposure_without_neighbor = np.divide(
                exposure[None, :], surface_switch,
                out=np.zeros_like(surface_switch), where=active,
            )
            partial = (
                radial_weights[i] * angular_weights[None, :]
                * exposure_without_neighbor * switch_derivative
                / distances
            )
            derivative = np.einsum("jp,jpk->jk", partial, displacement)
            jacobian[i, i] += np.sum(derivative, axis=0)
            jacobian[i, other_indices] -= derivative
    return areas, jacobian


def alpb_energy_components_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    charges_electron: np.ndarray,
    parameters: ALPBParameters,
) -> dict[str, float]:
    """Full fixed-charge ALPB energy decomposition (not an SCC calculator)."""
    numbers = np.asarray(atomic_numbers, dtype=int)
    charges = np.asarray(charges_electron, dtype=float)
    if charges.shape != (len(numbers),):
        raise ValueError("atomic charge count does not match coordinates")
    surface = alpb_surface_areas_bohr2(coordinates_bohr, numbers, parameters)
    tension = parameters.surface_tension_hartree_per_bohr2[numbers - 1]
    surface_radii = parameters.radii_bohr[numbers - 1] + parameters.probe_bohr
    hbond = -KCAL_TO_HARTREE * (
        parameters.hydrogen_bond_sqrt_kcal[numbers - 1] ** 2
    ) / (4.0 * np.pi * surface_radii**2)
    components = {
        "electrostatic": alpb_polar_energy_hartree(
            coordinates_bohr, numbers, charges, parameters
        ),
        "cavity_dispersion": float(surface @ tension),
        "hydrogen_bond": float(surface @ (hbond * charges**2)),
        "reference_shift": parameters.shift_hartree,
    }
    components["total"] = sum(components.values())
    return components


def alpb_fixed_charge_energy_gradient_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    charges_electron: np.ndarray,
    parameters: ALPBParameters,
) -> tuple[dict[str, float], np.ndarray]:
    """Full ALPB energy and explicit analytic gradient at fixed SCC charges.

    SCC charge response is supplied by the variational electronic problem,
    not by this fixed-charge derivative.
    """
    xyz = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    charges = np.asarray(charges_electron, dtype=float)
    if xyz.shape != (len(numbers), 3) or charges.shape != (len(numbers),):
        raise ValueError("incompatible ALPB coordinates, numbers, or charges")
    born, born_jacobian = born_radii_and_jacobian_bohr(
        xyz, numbers, parameters
    )
    assert born_jacobian is not None
    alpha_beta = ALPB_ALPHA / parameters.dielectric
    dielectric_factor = (1.0 / parameters.dielectric - 1.0) / (1.0 + alpha_beta)
    gradient = np.zeros_like(xyz)
    d_energy_d_born = -0.5 * dielectric_factor * charges**2 / born**2
    for i in range(len(numbers)):
        for j in range(i):
            displacement = xyz[i] - xyz[j]
            distance = float(np.linalg.norm(displacement))
            pair_radius = float(np.sqrt(born[i] * born[j]))
            factor = pair_radius / (
                pair_radius + P16_ZETA * distance / 16.0
            )
            factor16 = factor**16
            denominator = distance + pair_radius * factor16
            pair_charge = charges[i] * charges[j]
            direct = (
                -dielectric_factor * pair_charge
                * (1.0 - P16_ZETA * factor * factor16)
                / (denominator**2 * distance)
                * displacement
            )
            gradient[i] += direct
            gradient[j] -= direct
            born_coefficient = (
                -0.5
                * (distance * P16_ZETA / pair_radius * factor + 1.0)
                / pair_radius
                * factor16 / denominator**2
                * dielectric_factor * pair_charge
            )
            d_energy_d_born[i] += born[j] * born_coefficient
            d_energy_d_born[j] += born[i] * born_coefficient
    gradient += np.einsum("i,ijk->jk", d_energy_d_born, born_jacobian)
    if abs(float(np.sum(charges))) > 1.0e-12:
        size, size_gradient = _electrostatic_size_and_gradient_bohr(
            xyz, parameters.radii_bohr[numbers - 1]
        )
        shape_energy = (
            0.5 * dielectric_factor * alpha_beta
            * float(np.sum(charges)) ** 2 / size
        )
        gradient -= shape_energy / size * size_gradient
    surface, surface_jacobian = alpb_surface_areas_and_jacobian_bohr2(
        xyz, numbers, parameters
    )
    assert surface_jacobian is not None
    surface_radii = parameters.radii_bohr[numbers - 1] + parameters.probe_bohr
    hbond = -KCAL_TO_HARTREE * (
        parameters.hydrogen_bond_sqrt_kcal[numbers - 1] ** 2
    ) / (4.0 * np.pi * surface_radii**2)
    surface_coefficients = (
        parameters.surface_tension_hartree_per_bohr2[numbers - 1]
        + hbond * charges**2
    )
    gradient += np.einsum("i,ijk->jk", surface_coefficients, surface_jacobian)
    polar_matrix = _polar_matrix_from_born(xyz, numbers, parameters, born)
    components = {
        "electrostatic": 0.5 * float(charges @ polar_matrix @ charges),
        "cavity_dispersion": float(
            surface @ parameters.surface_tension_hartree_per_bohr2[numbers - 1]
        ),
        "hydrogen_bond": float(surface @ (hbond * charges**2)),
        "reference_shift": parameters.shift_hartree,
    }
    components["total"] = sum(components.values())
    return components, gradient
