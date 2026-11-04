"""Fully variational affine EPOC reaction-field and confinement runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import autograd.numpy as anp
from autograd import hessian_vector_product, value_and_grad
from autograd.scipy.special import erf
import numpy as np
from scipy.optimize import minimize

from .confinement import EllipsoidalVdwConfinement
from .contracts import EPOC_EVALUATION_SCHEMA, LEGACY_ZAFF_VARIATIONAL_CAVITY_SCHEMA


ZAFF_VARIATIONAL_CAVITY_SCHEMA = LEGACY_ZAFF_VARIATIONAL_CAVITY_SCHEMA


@dataclass(frozen=True)
class VariationalChargeModel:
    """Gaussian reference charges and optional local split-charge channels."""

    reference_charges: np.ndarray
    gaussian_widths_bohr: np.ndarray
    channel_matrix: np.ndarray
    channel_hardness_hartree: np.ndarray
    channel_bias_hartree: np.ndarray

    @classmethod
    def fixed(
        cls, charges: Sequence[float], gaussian_widths_bohr: Sequence[float]
    ) -> "VariationalChargeModel":
        q = np.asarray(charges, dtype=float).reshape(-1)
        widths = np.asarray(gaussian_widths_bohr, dtype=float).reshape(-1)
        return cls(
            q,
            widths,
            np.zeros((len(q), 0)),
            np.zeros((0, 0)),
            np.zeros(0),
        )

    def __post_init__(self) -> None:
        q = np.asarray(self.reference_charges, dtype=float).reshape(-1)
        widths = np.asarray(self.gaussian_widths_bohr, dtype=float).reshape(-1)
        channels = np.asarray(self.channel_matrix, dtype=float)
        hardness = np.asarray(self.channel_hardness_hartree, dtype=float)
        bias = np.asarray(self.channel_bias_hartree, dtype=float).reshape(-1)
        channel_count = channels.shape[1] if channels.ndim == 2 else -1
        if (
            not len(q)
            or widths.shape != q.shape
            or np.any(~np.isfinite(q))
            or np.any(~np.isfinite(widths))
            or np.any(widths <= 0.0)
            or channels.shape != (len(q), channel_count)
            or hardness.shape != (channel_count, channel_count)
            or bias.shape != (channel_count,)
            or np.any(~np.isfinite(channels))
            or np.any(~np.isfinite(hardness))
            or np.any(~np.isfinite(bias))
            or not np.allclose(hardness, hardness.T, atol=1.0e-12)
        ):
            raise ValueError("invalid variational Gaussian charge model")
        if channel_count and np.min(np.linalg.eigvalsh(hardness)) <= 0.0:
            raise ValueError("charge-channel hardness must be positive definite")
        if channel_count and not np.allclose(
            np.sum(channels, axis=0), 0.0, atol=1.0e-12
        ):
            raise ValueError("every split-charge channel must conserve charge")
        object.__setattr__(self, "reference_charges", q)
        object.__setattr__(self, "gaussian_widths_bohr", widths)
        object.__setattr__(self, "channel_matrix", channels)
        object.__setattr__(self, "channel_hardness_hartree", hardness)
        object.__setattr__(self, "channel_bias_hartree", bias)


@dataclass(frozen=True)
class VariationalCavityConfig:
    """Physical and numerical controls for the affine cavity functional."""

    dielectric: float = math.inf
    surface_order: int = 86
    surface_tension_hartree_per_bohr2: float = 0.0
    external_pressure_hartree_per_bohr3: float = 0.0
    wall_strength_hartree: float = 0.02
    wall_exponent: int = 8
    wall_sharpness: float = 30.0
    clearance_fraction: float = 0.03
    optimizer_gradient_tolerance: float = 1.0e-8
    optimizer_maximum_iterations: int = 200

    def __post_init__(self) -> None:
        dielectric = float(self.dielectric)
        if not (math.isinf(dielectric) or dielectric > 1.0):
            raise ValueError("variational CPCM dielectric must exceed one")
        if int(self.surface_order) < 26:
            raise ValueError("variational CPCM needs at least 26 surface nodes")
        if (
            float(self.surface_tension_hartree_per_bohr2) < 0.0
            or float(self.external_pressure_hartree_per_bohr3) < 0.0
            or float(self.wall_strength_hartree) <= 0.0
            or int(self.wall_exponent) < 4
            or int(self.wall_exponent) % 2
            or float(self.wall_sharpness) < 10.0
            or not 0.0 <= float(self.clearance_fraction) < 0.5
            or float(self.optimizer_gradient_tolerance) <= 0.0
            or int(self.optimizer_maximum_iterations) <= 0
        ):
            raise ValueError("invalid variational cavity controls")

    @property
    def dielectric_factor(self) -> float:
        if math.isinf(float(self.dielectric)):
            return 1.0
        return (float(self.dielectric) - 1.0) / float(self.dielectric)


@dataclass(frozen=True)
class VariationalCavityEvaluation:
    energy_hartree: float
    coordinate_gradient_hartree_per_bohr: np.ndarray
    shape_gradient_hartree: np.ndarray
    relaxed_charges: np.ndarray
    apparent_surface_charges: np.ndarray
    center_bohr: np.ndarray
    affine_map_bohr: np.ndarray
    geometry_class: str
    components_hartree: Mapping[str, float]
    schema: str = EPOC_EVALUATION_SCHEMA


@dataclass(frozen=True)
class VariationalCavityOptimization:
    evaluation: VariationalCavityEvaluation
    shape_parameters: np.ndarray
    converged: bool
    iterations: int
    gradient_norm: float
    message: str


class VariationalEllipsoidFunctional:
    """One differentiable functional for QEq/SQE, CPCM, UvdW and cavity shape.

    The conductor surface is represented by variational panel charges on an
    affinely moving spherical quadrature.  This symmetric surface functional
    remains regular in spherical, prolate, oblate and triaxial limits.  All
    first derivatives and Hessian-vector products are obtained by analytic
    automatic differentiation of the same stationary functional.
    """

    def __init__(
        self,
        charge_model: VariationalChargeModel,
        site_types: Sequence[str],
        vdw_template: EllipsoidalVdwConfinement,
        config: VariationalCavityConfig = VariationalCavityConfig(),
        *,
        clearance_fraction_by_site: Sequence[float] | None = None,
        site_clearance_bohr: Sequence[float] | None = None,
    ) -> None:
        self.charge_model = charge_model
        self.site_types = tuple(str(value) for value in site_types)
        self.vdw_template = vdw_template
        self.config = config
        if len(self.site_types) != len(self.charge_model.reference_charges):
            raise ValueError("variational cavity site types and charges differ")
        if any(
            label not in self.vdw_template.morse_parameters
            for label in self.site_types
        ):
            raise ValueError("variational cavity lacks UvdW parameters for a site type")
        clearance = (
            np.full(len(self.site_types), float(config.clearance_fraction))
            if clearance_fraction_by_site is None
            else np.asarray(clearance_fraction_by_site, dtype=float).reshape(-1)
        )
        if (
            clearance.shape != (len(self.site_types),)
            or np.any(~np.isfinite(clearance))
            or np.any(clearance < 0.0)
            or np.any(clearance >= 0.5)
        ):
            raise ValueError("invalid site-specific cavity clearance")
        self.clearance_fraction_by_site = clearance
        physical_clearance = (
            np.asarray(charge_model.gaussian_widths_bohr, dtype=float)
            if site_clearance_bohr is None
            else np.asarray(site_clearance_bohr, dtype=float).reshape(-1)
        )
        if (
            physical_clearance.shape != (len(self.site_types),)
            or np.any(~np.isfinite(physical_clearance))
            or np.any(physical_clearance < 0.0)
        ):
            raise ValueError("invalid physical site clearance radii")
        self.site_clearance_bohr = physical_clearance
        self.unit_nodes = _fibonacci_sphere(config.surface_order)
        self.unit_weight = 4.0 * math.pi / int(config.surface_order)
        reference_radius = float(
            np.prod(vdw_template.semiaxes_bohr) ** (1.0 / 3.0)
        )
        self.layer_fraction = float(vdw_template.layer_depth_bohr) / reference_radius
        self._morse = np.asarray(
            [
                vdw_template.morse_parameters[label]
                for label in self.site_types
            ],
            dtype=float,
        )
        maximum_terms = max(
            len(vdw_template.gaussian_terms[label]) for label in self.site_types
        )
        gaussian = np.zeros((len(self.site_types), maximum_terms, 3))
        gaussian_mask = np.zeros((len(self.site_types), maximum_terms))
        for index, label in enumerate(self.site_types):
            values = np.asarray(vdw_template.gaussian_terms[label], dtype=float)
            gaussian[index, : len(values), :] = values
            gaussian_mask[index, : len(values)] = 1.0
        self._gaussian = gaussian
        self._gaussian_mask = gaussian_mask
        self._combined_value_and_gradient = value_and_grad(self._combined_energy)
        self._combined_hvp = hessian_vector_product(self._combined_energy)
        self._shape_value_and_gradient = value_and_grad(
            self._shape_energy, argnum=0
        )

    @staticmethod
    def pack_shape(center_bohr: np.ndarray, affine_map_bohr: np.ndarray) -> np.ndarray:
        """Pack center and an SPD affine map into unconstrained Cholesky variables."""

        center = np.asarray(center_bohr, dtype=float).reshape(3)
        affine = np.asarray(affine_map_bohr, dtype=float).reshape(3, 3)
        if (
            np.any(~np.isfinite(center))
            or np.any(~np.isfinite(affine))
            or not np.allclose(affine, affine.T, atol=1.0e-12)
            or np.min(np.linalg.eigvalsh(affine)) <= 0.0
        ):
            raise ValueError("shape packing requires a finite SPD affine map")
        lower = np.linalg.cholesky(affine)
        return np.asarray(
            [
                *center,
                math.log(lower[0, 0]),
                math.log(lower[1, 1]),
                math.log(lower[2, 2]),
                lower[1, 0],
                lower[2, 0],
                lower[2, 1],
            ]
        )

    @staticmethod
    def unpack_shape(shape_parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return center and SPD affine map from nine unconstrained variables."""

        values = np.asarray(shape_parameters, dtype=float).reshape(9)
        center = values[:3]
        lower = np.array(
            [
                [math.exp(values[3]), 0.0, 0.0],
                [values[6], math.exp(values[4]), 0.0],
                [values[7], values[8], math.exp(values[5])],
            ]
        )
        return center, lower @ lower.T

    @classmethod
    def shape_from_confinement(
        cls, confinement: EllipsoidalVdwConfinement
    ) -> np.ndarray:
        affine = (
            confinement.rotation
            @ np.diag(confinement.semiaxes_bohr)
            @ confinement.rotation.T
        )
        return cls.pack_shape(confinement.center_bohr, affine)

    @staticmethod
    def classify_affine_shape(
        affine_map_bohr: np.ndarray, *, relative_tolerance: float = 1.0e-8
    ) -> str:
        """Classify an SPD cavity without changing the continuous backend."""

        axes = np.linalg.eigvalsh(
            np.asarray(affine_map_bohr, dtype=float).reshape(3, 3)
        )
        tolerance = float(relative_tolerance) * float(np.max(axes))
        lower_equal = abs(float(axes[1] - axes[0])) <= tolerance
        upper_equal = abs(float(axes[2] - axes[1])) <= tolerance
        if lower_equal and upper_equal:
            return "sphere"
        if lower_equal:
            return "prolate"
        if upper_equal:
            return "oblate"
        return "triaxial"

    def evaluate(
        self,
        coordinates_bohr: np.ndarray,
        shape_parameters: np.ndarray,
    ) -> VariationalCavityEvaluation:
        xyz, shape = self._validated_inputs(coordinates_bohr, shape_parameters)
        combined = np.concatenate((xyz.reshape(-1), shape))
        energy, gradient = self._combined_value_and_gradient(combined)
        components, relaxed_charges, surface_charges = self._stationary_state_numpy(
            xyz, shape
        )
        center, affine = self.unpack_shape(shape)
        ncart = xyz.size
        return VariationalCavityEvaluation(
            energy_hartree=float(energy),
            coordinate_gradient_hartree_per_bohr=np.asarray(
                gradient[:ncart], dtype=float
            ),
            shape_gradient_hartree=np.asarray(gradient[ncart:], dtype=float),
            relaxed_charges=relaxed_charges,
            apparent_surface_charges=surface_charges,
            center_bohr=center,
            affine_map_bohr=affine,
            geometry_class=self.classify_affine_shape(affine),
            components_hartree=components,
        )

    def hessian_vector_product(
        self,
        coordinates_bohr: np.ndarray,
        shape_parameters: np.ndarray,
        coordinate_vector_bohr: np.ndarray,
        shape_vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the exact relaxed Hessian in Cartesian and shape variables."""

        xyz, shape = self._validated_inputs(coordinates_bohr, shape_parameters)
        coordinate_direction = np.asarray(
            coordinate_vector_bohr, dtype=float
        ).reshape(xyz.shape)
        shape_direction = np.asarray(shape_vector, dtype=float).reshape(9)
        combined = np.concatenate((xyz.reshape(-1), shape))
        direction = np.concatenate(
            (coordinate_direction.reshape(-1), shape_direction)
        )
        product = np.asarray(self._combined_hvp(combined, direction), dtype=float)
        return product[: xyz.size], product[xyz.size :]

    def optimize(
        self,
        coordinates_bohr: np.ndarray,
        initial_shape_parameters: np.ndarray,
    ) -> VariationalCavityOptimization:
        """Minimize the cavity variables while all response variables relax."""

        xyz, shape = self._validated_inputs(
            coordinates_bohr, initial_shape_parameters
        )

        def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            energy, gradient = self._shape_value_and_gradient(values, xyz)
            return float(energy), np.asarray(gradient, dtype=float)

        result = minimize(
            objective,
            shape,
            method="L-BFGS-B",
            jac=True,
            bounds=self._shape_bounds(shape),
            options={
                "gtol": float(self.config.optimizer_gradient_tolerance),
                "maxiter": int(self.config.optimizer_maximum_iterations),
                "maxls": 40,
            },
        )
        evaluation = self.evaluate(xyz, np.asarray(result.x, dtype=float))
        rho = _effective_rho_numpy(
            xyz,
            evaluation.center_bohr,
            evaluation.affine_map_bohr,
            self.site_clearance_bohr,
        )
        admissible = bool(
            np.all(
                rho
                < 1.0
                - self.clearance_fraction_by_site
                - 1.0e-8
            )
        )
        converged = bool(result.success and admissible)
        message = str(result.message)
        if not admissible:
            message = f"{message}; final cavity violates site clearance"
        return VariationalCavityOptimization(
            evaluation=evaluation,
            shape_parameters=np.asarray(result.x, dtype=float),
            converged=converged,
            iterations=int(result.nit),
            gradient_norm=float(
                np.linalg.norm(evaluation.shape_gradient_hartree)
            ),
            message=message,
        )

    def _shape_bounds(
        self, initial_shape: np.ndarray
    ) -> tuple[tuple[float, float], ...]:
        _center, affine = self.unpack_shape(initial_shape)
        length = float(np.max(np.linalg.eigvalsh(affine)))
        diagonal_span = math.log(2.5)
        lower_scale = tuple(
            (float(initial_shape[index] - diagonal_span), float(initial_shape[index] + diagonal_span))
            for index in range(3, 6)
        )
        off_diagonal_span = math.sqrt(length) * 1.5
        return (
            *((float(initial_shape[index] - length), float(initial_shape[index] + length)) for index in range(3)),
            *lower_scale,
            *((float(initial_shape[index] - off_diagonal_span), float(initial_shape[index] + off_diagonal_span)) for index in range(6, 9)),
        )

    def _validated_inputs(
        self, coordinates_bohr: np.ndarray, shape_parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        shape = np.asarray(shape_parameters, dtype=float).reshape(-1)
        if (
            xyz.shape != (len(self.site_types), 3)
            or np.any(~np.isfinite(xyz))
            or shape.shape != (9,)
            or np.any(~np.isfinite(shape))
        ):
            raise ValueError("variational cavity coordinates or shape are invalid")
        center, affine = self.unpack_shape(shape)
        rho = _effective_rho_numpy(
            xyz, center, affine, self.site_clearance_bohr
        )
        if np.any(rho >= 1.0 - self.clearance_fraction_by_site):
            raise ValueError("initial variational cavity violates site clearance")
        return xyz, shape

    def _shape_energy(self, shape: anp.ndarray, xyz: anp.ndarray) -> anp.ndarray:
        combined = anp.concatenate((anp.ravel(xyz), shape))
        return self._combined_energy(combined)

    def _combined_energy(self, combined: anp.ndarray) -> anp.ndarray:
        site_count = len(self.site_types)
        xyz = anp.reshape(combined[: 3 * site_count], (site_count, 3))
        shape = combined[3 * site_count :]
        center, affine = _unpack_shape_autograd(shape)
        electrostatic, _charges, _surface = self._electrostatic_state(
            xyz, center, affine
        )
        rho, depth, volume_radius = _depth_autograd(xyz, center, affine)
        uvdw = self._uvdw_energy(depth, volume_radius)
        wall = self._wall_energy(rho, affine)
        surface_area, volume = self._surface_area_volume(affine)
        cavitation = (
            float(self.config.surface_tension_hartree_per_bohr2) * surface_area
            + float(self.config.external_pressure_hartree_per_bohr3) * volume
        )
        return electrostatic + uvdw + wall + cavitation

    def _electrostatic_state(
        self, xyz: anp.ndarray, center: anp.ndarray, affine: anp.ndarray
    ) -> tuple[anp.ndarray, anp.ndarray, anp.ndarray]:
        q0 = anp.asarray(self.charge_model.reference_charges)
        widths = anp.asarray(self.charge_model.gaussian_widths_bohr)
        channels = anp.asarray(self.charge_model.channel_matrix)
        hardness = anp.asarray(self.charge_model.channel_hardness_hartree)
        bias = anp.asarray(self.charge_model.channel_bias_hartree)
        vacuum = _gaussian_coulomb_matrix(xyz, widths)
        surface_xyz, surface_weights = self._surface_geometry(center, affine)
        surface_matrix = _surface_coulomb_matrix(surface_xyz, surface_weights)
        coupling = _surface_gaussian_coupling(surface_xyz, xyz, widths)
        factor = float(self.config.dielectric_factor)
        base = 0.5 * anp.dot(q0, anp.dot(vacuum, q0))
        channel_count = channels.shape[1]
        if channel_count:
            pp = anp.dot(channels.T, anp.dot(vacuum, channels)) + hardness
            ps = factor * anp.dot(channels.T, coupling.T)
            ss = factor * surface_matrix
            hessian = anp.concatenate(
                (
                    anp.concatenate((pp, ps), axis=1),
                    anp.concatenate((ps.T, ss), axis=1),
                ),
                axis=0,
            )
            linear = anp.concatenate(
                (
                    anp.dot(channels.T, anp.dot(vacuum, q0)) - bias,
                    factor * anp.dot(coupling, q0),
                )
            )
            stationary = -anp.linalg.solve(hessian, linear)
            flow = stationary[:channel_count]
            surface_charge = stationary[channel_count:]
            charges = q0 + anp.dot(channels, flow)
        else:
            linear = factor * anp.dot(coupling, q0)
            surface_charge = -anp.linalg.solve(
                factor * surface_matrix, linear
            )
            stationary = surface_charge
            charges = q0
            hessian = factor * surface_matrix
        energy = base - 0.5 * anp.dot(stationary, anp.dot(hessian, stationary))
        return energy, charges, surface_charge

    def _surface_geometry(
        self, center: anp.ndarray, affine: anp.ndarray
    ) -> tuple[anp.ndarray, anp.ndarray]:
        unit = anp.asarray(self.unit_nodes)
        inverse = anp.linalg.inv(affine)
        determinant = anp.linalg.det(affine)
        points = center[None, :] + anp.dot(unit, affine.T)
        jacobian = determinant * anp.sqrt(
            anp.sum(anp.dot(unit, inverse) ** 2, axis=1)
        )
        return points, float(self.unit_weight) * jacobian

    def _surface_area_volume(
        self, affine: anp.ndarray
    ) -> tuple[anp.ndarray, anp.ndarray]:
        _points, weights = self._surface_geometry(anp.zeros(3), affine)
        area = anp.sum(weights)
        volume = (4.0 * math.pi / 3.0) * anp.linalg.det(affine)
        return area, volume

    def _uvdw_energy(
        self, depth: anp.ndarray, volume_radius: anp.ndarray
    ) -> anp.ndarray:
        layer_depth = float(self.layer_fraction) * volume_radius
        morse = anp.asarray(self._morse)
        gaussian = anp.asarray(self._gaussian)
        mask = anp.asarray(self._gaussian_mask)
        site_energies = []
        for index in range(len(self.site_types)):
            site_depth = depth[index]
            safe_depth = anp.where(
                site_depth < 0.0,
                0.0 * site_depth,
                anp.where(site_depth > layer_depth, layer_depth, site_depth),
            )
            x = site_depth / layer_depth
            switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
            exponential = anp.exp(
                -morse[index, 2] * (safe_depth - morse[index, 1])
            )
            radial = morse[index, 0] * (
                exponential**2 - 2.0 * exponential
            )
            displacement = safe_depth - gaussian[index, :, 1]
            corrections = anp.sum(
                mask[index]
                * gaussian[index, :, 0]
                * anp.exp(-gaussian[index, :, 2] * displacement**2)
            )
            # A smooth C2 continuation is zero on the interior side of the
            # layer.  Keeping each site scalar also makes nested analytic AD
            # independent of broadcasting conventions.
            active_switch = anp.where(
                anp.logical_and(site_depth >= 0.0, site_depth < layer_depth),
                switch,
                0.0 * switch,
            )
            site_energies.append(active_switch * (radial + corrections))
        return anp.sum(
            anp.stack(tuple(site_energies))
        )

    def _wall_energy(
        self, rho: anp.ndarray, affine: anp.ndarray
    ) -> anp.ndarray:
        admissible_radius = 1.0 - anp.asarray(
            self.clearance_fraction_by_site
        )
        inverse = anp.linalg.inv(affine)
        inverse_norm2 = (
            inverse[0, 0] ** 2
            + inverse[0, 1] ** 2
            + inverse[0, 2] ** 2
            + inverse[1, 0] ** 2
            + inverse[1, 1] ** 2
            + inverse[1, 2] ** 2
            + inverse[2, 0] ** 2
            + inverse[2, 1] ** 2
            + inverse[2, 2] ** 2
        )
        inverse_scale = anp.sqrt(inverse_norm2 / 3.0)
        effective_rho = anp.stack(
            tuple(
                rho[index]
                + float(self.site_clearance_bohr[index]) * inverse_scale
                for index in range(len(self.site_types))
            )
        )
        zeta = effective_rho / admissible_radius
        sharpness = float(self.config.wall_sharpness)
        soft_wall = anp.logaddexp(0.0, sharpness * (zeta - 1.0))
        radial_weight = zeta ** (int(self.config.wall_exponent) // 2)
        return float(self.config.wall_strength_hartree) * anp.sum(
            (radial_weight * soft_wall) ** 2
        )

    def _stationary_state_numpy(
        self, xyz: np.ndarray, shape: np.ndarray
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        center, affine = _unpack_shape_autograd(anp.asarray(shape))
        electrostatic, charges, surface = self._electrostatic_state(
            anp.asarray(xyz), center, affine
        )
        rho, depth, volume_radius = _depth_autograd(
            anp.asarray(xyz), center, affine
        )
        uvdw = self._uvdw_energy(depth, volume_radius)
        wall = self._wall_energy(rho, affine)
        area, volume = self._surface_area_volume(affine)
        surface_term = (
            float(self.config.surface_tension_hartree_per_bohr2) * area
        )
        pressure_term = (
            float(self.config.external_pressure_hartree_per_bohr3) * volume
        )
        return (
            {
                "electrostatic_and_polarization": float(electrostatic),
                "uvdw": float(uvdw),
                "wall": float(wall),
                "surface": float(surface_term),
                "pressure_volume": float(pressure_term),
            },
            np.asarray(charges, dtype=float),
            np.asarray(surface, dtype=float),
        )


def _unpack_shape_autograd(
    shape: anp.ndarray,
) -> tuple[anp.ndarray, anp.ndarray]:
    center = shape[:3]
    zero = 0.0 * shape[0]
    lower = anp.stack(
        (
            anp.stack((anp.exp(shape[3]), zero, zero)),
            anp.stack((shape[6], anp.exp(shape[4]), zero)),
            anp.stack((shape[7], shape[8], anp.exp(shape[5]))),
        )
    )
    return center, anp.dot(lower, lower.T)


def _depth_autograd(
    xyz: anp.ndarray, center: anp.ndarray, affine: anp.ndarray
) -> tuple[anp.ndarray, anp.ndarray, anp.ndarray]:
    inverse = anp.linalg.inv(affine)
    scaled = anp.dot(xyz - center[None, :], inverse.T)
    rho = anp.sqrt(anp.sum(scaled**2, axis=1))
    volume_radius = anp.linalg.det(affine) ** (1.0 / 3.0)
    return rho, volume_radius * (1.0 - rho), volume_radius


def _effective_rho_numpy(
    xyz: np.ndarray,
    center: np.ndarray,
    affine: np.ndarray,
    site_clearance_bohr: np.ndarray,
) -> np.ndarray:
    inverse = np.linalg.inv(affine)
    rho = np.linalg.norm(
        (np.asarray(xyz) - np.asarray(center)) @ inverse.T, axis=1
    )
    inverse_scale = float(np.sqrt(np.sum(inverse**2) / 3.0))
    return rho + np.asarray(site_clearance_bohr) * inverse_scale


def _gaussian_coulomb_matrix(
    xyz: anp.ndarray, widths: anp.ndarray
) -> anp.ndarray:
    count = xyz.shape[0]
    eye = anp.eye(count)
    delta = xyz[:, None, :] - xyz[None, :, :]
    distance = anp.sqrt(anp.sum(delta**2, axis=2) + eye)
    beta = 1.0 / anp.sqrt(
        2.0 * (widths[:, None] ** 2 + widths[None, :] ** 2)
    )
    pair = erf(beta * distance) / distance
    diagonal = 2.0 * anp.diag(beta) / math.sqrt(math.pi)
    return pair * (1.0 - eye) + anp.diag(diagonal)


def _surface_gaussian_coupling(
    surface_xyz: anp.ndarray, xyz: anp.ndarray, widths: anp.ndarray
) -> anp.ndarray:
    delta = surface_xyz[:, None, :] - xyz[None, :, :]
    distance = anp.sqrt(anp.sum(delta**2, axis=2))
    beta = 1.0 / (math.sqrt(2.0) * widths[None, :])
    return erf(beta * distance) / distance


def _surface_coulomb_matrix(
    surface_xyz: anp.ndarray, surface_weights: anp.ndarray
) -> anp.ndarray:
    count = surface_xyz.shape[0]
    eye = anp.eye(count)
    delta = surface_xyz[:, None, :] - surface_xyz[None, :, :]
    distance = anp.sqrt(anp.sum(delta**2, axis=2) + eye)
    off_diagonal = (1.0 - eye) / distance
    self_potential = 2.0 * anp.sqrt(math.pi / surface_weights)
    # The exact single-layer operator is positive.  Finite point-panel
    # collocation can lose that property for strongly distorted panels, so
    # the analytic disk self term is augmented by the off-diagonal row sum.
    # This symmetric strictly diagonally dominant form preserves the
    # variational minimum for every admissible affine shape.
    diagonal = self_potential + anp.sum(off_diagonal, axis=1)
    return off_diagonal + anp.diag(diagonal)


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(int(count), dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / int(count)
    phi = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


__all__ = [
    "EPOC_EVALUATION_SCHEMA",
    "ZAFF_VARIATIONAL_CAVITY_SCHEMA",
    "VariationalCavityConfig",
    "VariationalCavityEvaluation",
    "VariationalCavityOptimization",
    "VariationalChargeModel",
    "VariationalEllipsoidFunctional",
]
