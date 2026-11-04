"""LOCK consumption of production SMITH tunneling-coordinate contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .lam_rotation import PermutationInversionSymmetry
from .multi_lam import GeneralLAMCoordinate, general_lam_surface


HARTREE_TO_MHZ = 6.5796839204999e9
ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2 = 505379.0090099937


@dataclass(frozen=True)
class CurvilinearSplittingConvergence:
    """LOCK grid-convergence certificate for a curvilinear double well."""

    grid_sizes: tuple[int, ...]
    splittings_MHz: tuple[float, ...]
    ground_energies_MHz: tuple[float, ...]
    boundary_probabilities: tuple[float, ...]
    converged: bool
    tolerance_MHz: float

    @property
    def splitting_MHz(self) -> float:
        return self.splittings_MHz[-1]


@dataclass(frozen=True)
class ManifoldSplittingConvergence:
    """Grid convergence of the lowest doublet on a multidimensional manifold."""

    grid_shapes: tuple[tuple[int, ...], ...]
    splittings_MHz: tuple[float, ...]
    boundary_probabilities: tuple[float, ...]
    converged: bool
    tolerance_MHz: float


def _value(contract: object, name: str) -> object:
    if isinstance(contract, Mapping):
        return contract[name]
    return getattr(contract, name)


def rectilinear_tunneling_surface(
    coordinate_contracts: Sequence[object],
    *,
    potential_MHz: np.ndarray,
    kinetic_coefficients_MHz: Sequence[float] | None = None,
    R_MHz: np.ndarray,
    dipole_au: np.ndarray | None = None,
    spin_tensor_surfaces_MHz: Mapping[str, np.ndarray] | None = None,
    permutation_inversion_symmetry: PermutationInversionSymmetry | None = None,
    metadata: Mapping[str, object] | None = None,
):
    """Build a constant-mass LOCK surface from SMITH ``Q_im`` coordinates.

    The adapter deliberately sets rotation--LAM momentum and geometric-scalar
    terms to zero.  Introducing either term requires a different explicit
    kinetic contract and cannot happen through relaxed-path geometry alone.
    """

    contracts = tuple(coordinate_contracts)
    if not contracts:
        raise ValueError("LOCK requires at least one SMITH tunneling coordinate")
    coordinates = []
    contract_metadata = []
    for contract in contracts:
        if (
            str(_value(contract, "coordinate_kind")) != "rectilinear_qim"
            or bool(_value(contract, "periodic"))
            or str(_value(contract, "kinetic_model")) != "constant_mass_rectilinear"
            or str(_value(contract, "vam_treatment")) != "omitted"
        ):
            raise ValueError("LOCK production tunneling requires a rectilinear Qim contract")
        points = tuple(float(value) for value in _value(contract, "points"))
        coordinates.append(
            GeneralLAMCoordinate(
                str(_value(contract, "label")),
                points,
                False,
                "rectilinear_qim",
            )
        )
        to_dict = getattr(contract, "to_dict", None)
        contract_metadata.append(dict(to_dict()) if callable(to_dict) else dict(contract))
    dimension = len(coordinates)
    if kinetic_coefficients_MHz is None:
        kinetic_coefficients_MHz = tuple(
            float(_value(contract, "kinetic_coefficient_MHz")) for contract in contracts
        )
    coefficients = np.asarray(kinetic_coefficients_MHz, dtype=float)
    if (
        coefficients.shape != (dimension,)
        or np.any(coefficients <= 0.0)
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("provide one positive constant kinetic coefficient per Qim axis")
    shape = tuple(len(axis.points) for axis in coordinates)
    F = np.zeros(shape + (dimension, dimension), dtype=float)
    for axis, coefficient in enumerate(coefficients):
        F[..., axis, axis] = coefficient
    rho = np.zeros(shape + (3, dimension), dtype=float)
    return general_lam_surface(
        coordinates,
        potential_MHz=np.asarray(potential_MHz, dtype=float),
        F_MHz=F,
        rho=rho,
        R_MHz=np.asarray(R_MHz, dtype=float),
        dipole_au=dipole_au,
        geometric_potential_MHz=np.zeros(shape, dtype=float),
        spin_tensor_surfaces_MHz=spin_tensor_surfaces_MHz,
        permutation_inversion_symmetry=permutation_inversion_symmetry,
        metadata={
            **dict(metadata or {}),
            "coordinate_source": "SMITH",
            "kinetic_model": "constant_mass_rectilinear",
            "vam_treatment": "omitted",
            "smith_tunneling_coordinates": contract_metadata,
        },
    )


def rectilinear_tunneling_surface_from_qim_result(
    result: object,
    *,
    metadata: Mapping[str, object] | None = None,
    permutation_inversion_symmetry: PermutationInversionSymmetry | None = None,
):
    """Convert one complete LINK Qim result into a ready-to-solve LOCK surface."""

    points = tuple(getattr(result, "points"))
    contract = getattr(result, "contract")
    expected = tuple(float(value) for value in _value(contract, "points"))
    if tuple(float(point.q_im) for point in points) != expected:
        raise ValueError("LINK Qim result does not follow the frozen SMITH grid")
    if not points or not all(bool(point.converged) for point in points):
        raise ValueError("LOCK refuses an incomplete or unconverged Qim path")
    energies = np.asarray([point.energy_hartree for point in points], dtype=float)
    potential = (energies - float(np.min(energies))) * HARTREE_TO_MHZ
    geometry = np.asarray([point.coordinates_angstrom for point in points], dtype=float)
    cartesian_masses = np.asarray(_value(contract, "cartesian_masses"), dtype=float)
    if cartesian_masses.size != geometry.shape[1] * 3:
        raise ValueError("SMITH masses and LINK geometries have inconsistent dimensions")
    triples = cartesian_masses.reshape((-1, 3))
    if not np.allclose(triples, triples[:, :1], rtol=0.0, atol=1.0e-12):
        raise ValueError("each atom must have one common mass for x, y, and z")
    atom_masses = triples[:, 0]
    rotational = _rotational_metric_MHz(geometry, atom_masses)
    dipoles = [point.dipole_au for point in points]
    dipole_surface = None
    if any(value is not None for value in dipoles):
        if not all(value is not None for value in dipoles):
            raise ValueError("the LINK dipole surface is incomplete")
        dipole_surface = np.asarray(dipoles, dtype=float)
    energy_sigma = [point.energy_standard_error_hartree for point in points]
    uncertainty: dict[str, object] = {}
    if any(value is not None for value in energy_sigma):
        if not all(value is not None for value in energy_sigma):
            raise ValueError("the LINK energy uncertainty surface is incomplete")
        uncertainty["energy_standard_error_MHz"] = (
            HARTREE_TO_MHZ * np.asarray(energy_sigma, dtype=float)
        ).tolist()
    dipole_covariance = [point.dipole_covariance_au2 for point in points]
    if any(value is not None for value in dipole_covariance):
        if not all(value is not None for value in dipole_covariance):
            raise ValueError("the LINK dipole covariance surface is incomplete")
        uncertainty["dipole_covariance_au2"] = np.asarray(dipole_covariance, dtype=float).tolist()
    return rectilinear_tunneling_surface(
        (contract,),
        potential_MHz=potential,
        R_MHz=rotational,
        dipole_au=dipole_surface,
        permutation_inversion_symmetry=permutation_inversion_symmetry,
        metadata={
            **dict(metadata or {}),
            "producer": "LINK/SMITH",
            "qim_optimizer": str(getattr(result, "optimizer", "unknown")),
            "qim_all_converged": bool(getattr(result, "all_converged", True)),
            "reference_energy_hartree": float(np.min(energies)),
            "energy_conversion_MHz_per_hartree": HARTREE_TO_MHZ,
            "uncertainty": uncertainty,
        },
    )


def curvilinear_tunneling_surface(
    path: object,
    *,
    grid_size: int,
    outer_wall_hartree: float | None = None,
    adiabatic_correction_hartree: Sequence[float] | None = None,
    metadata: Mapping[str, object] | None = None,
):
    """Interpolate a SMITH Cartesian-metric path onto a uniform LOCK DVR grid."""

    from scipy.interpolate import PchipInterpolator

    if grid_size < 9:
        raise ValueError("curvilinear LOCK grids need at least 9 points")
    if grid_size % 2 == 0:
        raise ValueError("use an odd curvilinear grid so the saddle is represented exactly")
    if str(_value(path, "coordinate_kind")) != "curvilinear_mass_path":
        raise ValueError("LOCK requires a SMITH curvilinear mass-path contract")
    source = np.asarray(_value(path, "coordinates"), dtype=float)
    if not np.isclose(source[0], -source[-1], rtol=1.0e-8, atol=1.0e-10):
        raise ValueError("a symmetric tunneling DVR requires matching path endpoints")
    geometries = np.asarray(_value(path, "geometries_angstrom"), dtype=float)
    electronic_energies = np.asarray(_value(path, "energies_hartree"), dtype=float)
    if adiabatic_correction_hartree is None:
        correction = np.zeros_like(electronic_energies)
    else:
        correction = np.asarray(adiabatic_correction_hartree, dtype=float)
        if correction.shape != electronic_energies.shape or not np.all(np.isfinite(correction)):
            raise ValueError("adiabatic correction must be finite and match the SMITH images")
    energies = electronic_energies + correction
    masses = np.asarray(_value(path, "atom_masses_amu"), dtype=float)
    geometry_spline = PchipInterpolator(source, geometries, axis=0)
    energy_spline = PchipInterpolator(source, energies)
    extent = float(source[-1])
    extension_curvature = None
    extension_slope = None
    if outer_wall_hartree is not None:
        if outer_wall_hartree <= 0.0:
            raise ValueError("the curvilinear DVR outer-wall height must be positive")
        center = int(np.argmin(np.abs(source)))
        minimum = center + int(np.argmin(energies[center:]))
        rising = np.arange(minimum, len(source))
        if len(rising) < 3 or minimum == len(source) - 1:
            raise ValueError("harmonic wall extension requires three images from the minimum outward")
        displacement2 = (source[rising] - source[minimum]) ** 2
        rise = energies[rising] - energies[minimum]
        extension_curvature = float(np.dot(displacement2, rise) / np.dot(displacement2, displacement2))
        extension_slope = float(energy_spline.derivative()(source[-1]))
        if extension_curvature <= 0.0 or extension_slope <= 0.0:
            raise ValueError("the sampled outer branch does not define a rising harmonic wall")
        remaining = float(outer_wall_hartree - (energies[-1] - energies[minimum]))
        if remaining > 0.0:
            delta = (
                -extension_slope
                + np.sqrt(extension_slope**2 + 4.0 * extension_curvature * remaining)
            ) / (2.0 * extension_curvature)
            extent += float(delta)
    grid = np.linspace(-extent, extent, int(grid_size))
    clipped = np.clip(grid, source[0], source[-1])
    interpolated_geometry = np.asarray(geometry_spline(clipped), dtype=float)
    derivative = np.asarray(geometry_spline.derivative()(clipped), dtype=float)
    for side, boundary in ((-1, source[0]), (1, source[-1])):
        mask = grid < source[0] if side < 0 else grid > source[-1]
        if np.any(mask):
            delta = grid[mask] - boundary
            boundary_geometry = np.asarray(geometry_spline(boundary), dtype=float)
            boundary_derivative = np.asarray(geometry_spline.derivative()(boundary), dtype=float)
            interpolated_geometry[mask] = boundary_geometry + delta[:, None, None] * boundary_derivative
            derivative[mask] = boundary_derivative
    metric = np.einsum("i,qia,qia->q", masses, derivative, derivative)
    if np.any(metric <= 0.0) or not np.all(np.isfinite(metric)):
        raise ValueError("interpolated SMITH path has a singular kinetic metric")
    interpolated_energy = np.asarray(energy_spline(clipped), dtype=float)
    if extension_curvature is not None and extension_slope is not None:
        for side, boundary in ((-1, source[0]), (1, source[-1])):
            mask = grid < source[0] if side < 0 else grid > source[-1]
            if np.any(mask):
                delta = np.abs(grid[mask] - boundary)
                interpolated_energy[mask] = (
                    energies[-1] + extension_slope * delta + extension_curvature * delta**2
                )
    potential = (interpolated_energy - float(np.min(interpolated_energy))) * HARTREE_TO_MHZ
    F = (ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2 / metric)[:, None, None]
    rho = np.zeros((grid_size, 3, 1), dtype=float)
    rotational = _rotational_metric_MHz(interpolated_geometry, masses)
    coordinate = GeneralLAMCoordinate(
        str(_value(path, "label")), tuple(float(x) for x in grid), False, "curvilinear_mass_path"
    )
    to_dict = getattr(path, "to_dict", None)
    return general_lam_surface(
        (coordinate,),
        potential_MHz=potential,
        F_MHz=F,
        rho=rho,
        R_MHz=rotational,
        geometric_potential_MHz=np.zeros(grid_size),
        metadata={
            **dict(metadata or {}),
            "producer": "SMITH/LINK curvilinear continuation",
            "kinetic_model": "cartesian_mass_metric",
            "metric_min": float(np.min(metric)),
            "metric_max": float(np.max(metric)),
            "source_image_count": int(len(source)),
            "outer_wall_extension_hartree": outer_wall_hartree,
            "outer_wall_extension_curvature": extension_curvature,
            "adiabatic_correction": adiabatic_correction_hartree is not None,
            "adiabatic_correction_range_hartree": float(np.ptp(correction)),
            "smith_curvilinear_path": to_dict() if callable(to_dict) else None,
        },
    )


def curvilinear_tunneling_manifold_surface(
    manifold: object,
    *,
    grid_sizes: Sequence[int],
    adiabatic_correction_hartree: np.ndarray | None = None,
    metadata: Mapping[str, object] | None = None,
):
    """Interpolate a SMITH multidimensional manifold into a LOCK product DVR."""

    from scipy.interpolate import RegularGridInterpolator

    if str(_value(manifold, "coordinate_kind")) != "curvilinear_mass_manifold":
        raise ValueError("LOCK requires a SMITH curvilinear mass-manifold contract")
    source_axes = tuple(np.asarray(axis, dtype=float) for axis in _value(manifold, "coordinates"))
    sizes = tuple(int(value) for value in grid_sizes)
    dimension = len(source_axes)
    if len(sizes) != dimension or any(value < 3 or value % 2 == 0 for value in sizes):
        raise ValueError("provide one odd product-DVR grid size of at least three per coordinate")
    target_axes = tuple(
        np.linspace(float(axis[0]), float(axis[-1]), size)
        for axis, size in zip(source_axes, sizes, strict=True)
    )
    target_mesh = np.meshgrid(*target_axes, indexing="ij")
    target_points = np.column_stack([item.reshape(-1) for item in target_mesh])
    geometry_source = np.asarray(_value(manifold, "geometries_angstrom"), dtype=float)
    energy = np.asarray(_value(manifold, "energies_hartree"), dtype=float)
    if adiabatic_correction_hartree is not None:
        correction = np.asarray(adiabatic_correction_hartree, dtype=float)
        if correction.shape != energy.shape or not np.all(np.isfinite(correction)):
            raise ValueError("adiabatic correction must match the manifold energy tensor")
        energy = energy + correction
    geometry = RegularGridInterpolator(source_axes, geometry_source)(target_points).reshape(
        sizes + geometry_source.shape[-2:]
    )
    interpolated_energy = RegularGridInterpolator(source_axes, energy)(target_points).reshape(sizes)
    metric_source = np.asarray(_value(manifold, "kinetic_metric"), dtype=float)
    metric = RegularGridInterpolator(source_axes, metric_source)(target_points).reshape(
        sizes + (dimension, dimension)
    )
    metric = 0.5 * (metric + np.swapaxes(metric, -1, -2))
    if np.min(np.linalg.eigvalsh(metric)) <= 0.0:
        raise ValueError("interpolated manifold metric is not positive definite")
    masses = np.asarray(_value(manifold, "atom_masses_amu"), dtype=float)
    flat_rotation = _rotational_metric_MHz(geometry.reshape((-1,) + geometry.shape[-2:]), masses)
    rotation = flat_rotation.reshape(sizes + (3, 3))
    potential = (interpolated_energy - float(np.min(interpolated_energy))) * HARTREE_TO_MHZ
    kinetic = ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2 * np.linalg.inv(metric)
    coordinates = tuple(
        GeneralLAMCoordinate(str(label), tuple(axis), False, "curvilinear_mass_manifold")
        for label, axis in zip(_value(manifold, "labels"), target_axes, strict=True)
    )
    to_dict = getattr(manifold, "to_dict", None)
    return general_lam_surface(
        coordinates,
        potential_MHz=potential,
        F_MHz=kinetic,
        rho=np.zeros(sizes + (3, dimension)),
        R_MHz=rotation,
        geometric_potential_MHz=np.zeros(sizes),
        metadata={
            **dict(metadata or {}),
            "producer": "SMITH/LOCK multidimensional curvilinear manifold",
            "kinetic_model": "full_cartesian_mass_metric",
            "metric_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(metric))),
            "source_grid_shape": list(energy.shape),
            "adiabatic_correction": adiabatic_correction_hartree is not None,
            "smith_curvilinear_manifold": to_dict() if callable(to_dict) else None,
        },
    )


def converge_curvilinear_manifold_splitting(
    manifold: object,
    *,
    grid_shapes: Sequence[Sequence[int]],
    tolerance_MHz: float = 0.1,
    maximum_boundary_probability: float = 1.0e-6,
) -> ManifoldSplittingConvergence:
    """Converge the lowest doublet while refining every active coordinate."""

    from .lam_rotation import solve_lam_basis

    shapes = tuple(tuple(int(value) for value in shape) for shape in grid_shapes)
    if len(shapes) < 2 or tolerance_MHz <= 0.0 or maximum_boundary_probability <= 0.0:
        raise ValueError("provide at least two grids and positive convergence tolerances")
    splittings: list[float] = []
    boundaries: list[float] = []
    used: list[tuple[int, ...]] = []
    converged = False
    for shape in shapes:
        surface = curvilinear_tunneling_manifold_surface(manifold, grid_sizes=shape)
        basis = solve_lam_basis(surface, state_count=4)
        probability = np.abs(basis.vectors[:, :2].reshape(shape + (2,))) ** 2
        boundary_mask = np.zeros(shape, dtype=bool)
        for axis in range(len(shape)):
            selector = [slice(None)] * len(shape)
            selector[axis] = (0, -1)
            boundary_mask[tuple(selector)] = True
        boundary = float(np.sum(probability[boundary_mask]) / np.sum(probability))
        splitting = float(basis.energies_MHz[1] - basis.energies_MHz[0])
        used.append(shape)
        splittings.append(splitting)
        boundaries.append(boundary)
        if (
            len(splittings) > 1
            and abs(splittings[-1] - splittings[-2]) <= tolerance_MHz
            and boundary <= maximum_boundary_probability
        ):
            converged = True
            break
    return ManifoldSplittingConvergence(
        tuple(used), tuple(splittings), tuple(boundaries), converged, float(tolerance_MHz)
    )


def converge_curvilinear_tunneling_splitting(
    path: object,
    *,
    grid_sizes: Sequence[int] = (81, 121, 181, 241, 321),
    tolerance_MHz: float = 0.1,
    maximum_boundary_probability: float = 1.0e-6,
    outer_wall_hartree: float | None = None,
    adiabatic_correction_hartree: Sequence[float] | None = None,
) -> CurvilinearSplittingConvergence:
    """Increase the LOCK grid until the lowest doublet splitting is stable."""

    from .lam_rotation import solve_lam_basis

    sizes = tuple(int(value) for value in grid_sizes)
    if len(sizes) < 2 or any(value < 9 or value % 2 == 0 for value in sizes):
        raise ValueError("provide at least two increasing odd LOCK grid sizes")
    if any(right <= left for left, right in zip(sizes, sizes[1:], strict=False)):
        raise ValueError("LOCK convergence grid sizes must increase")
    if tolerance_MHz <= 0.0 or maximum_boundary_probability <= 0.0:
        raise ValueError("LOCK convergence tolerances must be positive")
    splittings: list[float] = []
    ground: list[float] = []
    boundaries: list[float] = []
    converged = False
    used_sizes: list[int] = []
    for size in sizes:
        surface = curvilinear_tunneling_surface(
            path,
            grid_size=size,
            outer_wall_hartree=outer_wall_hartree,
            adiabatic_correction_hartree=adiabatic_correction_hartree,
        )
        basis = solve_lam_basis(surface, state_count=4)
        splitting = float(basis.energies_MHz[1] - basis.energies_MHz[0])
        boundary = float(
            np.sum(np.abs(basis.vectors[[0, -1], :2]) ** 2) / np.sum(np.abs(basis.vectors[:, :2]) ** 2)
        )
        used_sizes.append(size)
        splittings.append(splitting)
        ground.append(float(basis.energies_MHz[0]))
        boundaries.append(boundary)
        if (
            len(splittings) >= 2
            and abs(splittings[-1] - splittings[-2]) <= tolerance_MHz
            and boundary <= maximum_boundary_probability
        ):
            converged = True
            break
    return CurvilinearSplittingConvergence(
        grid_sizes=tuple(used_sizes),
        splittings_MHz=tuple(splittings),
        ground_energies_MHz=tuple(ground),
        boundary_probabilities=tuple(boundaries),
        converged=converged,
        tolerance_MHz=float(tolerance_MHz),
    )


def isotopologue_tunneling_surfaces_from_qim_results(
    results: Mapping[str, object],
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build mass-specific LOCK surfaces from independently relaxed Qim paths."""

    return {
        str(label): rectilinear_tunneling_surface_from_qim_result(
            result,
            metadata={**dict(metadata or {}), "isotopologue": str(label)},
        )
        for label, result in results.items()
    }


def solve_isotopologue_tunneling_results(
    results: Mapping[str, object],
    *,
    j_values: Sequence[int] | range,
    lam_state_count: int = 12,
    level_count: int | None = None,
    workers: int = 1,
) -> dict[str, dict[str, object]]:
    """Construct and solve each isotope-specific Qim surface with one policy."""

    from .lam_rotation import solve_lam_rotation

    surfaces = isotopologue_tunneling_surfaces_from_qim_results(results)
    return {
        label: {
            "surface": surface,
            "result": solve_lam_rotation(
                surface,
                j_values=j_values,
                lam_state_count=lam_state_count,
                level_count=level_count,
                workers=workers,
            ),
        }
        for label, surface in surfaces.items()
    }


def _rotational_metric_MHz(geometries: np.ndarray, masses: np.ndarray) -> np.ndarray:
    center = np.einsum("i,qia->qa", masses, geometries) / float(np.sum(masses))
    centered = geometries - center[:, None, :]
    second = np.einsum("i,qia,qib->qab", masses, centered, centered, optimize=True)
    inertia = np.trace(second, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) - second
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError("Qim rotational inertia is singular at one or more grid points")
    return ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2 * np.linalg.inv(inertia)


__all__ = [
    "CurvilinearSplittingConvergence",
    "ManifoldSplittingConvergence",
    "converge_curvilinear_tunneling_splitting",
    "converge_curvilinear_manifold_splitting",
    "curvilinear_tunneling_surface",
    "curvilinear_tunneling_manifold_surface",
    "isotopologue_tunneling_surfaces_from_qim_results",
    "rectilinear_tunneling_surface",
    "rectilinear_tunneling_surface_from_qim_result",
    "solve_isotopologue_tunneling_results",
]
