"""Analytic mean-field van der Waals confinement for EPOC domains."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

from .contracts import EPOC_CONFINEMENT_SCHEMA, LEGACY_ZAFF_VDW_CONFINEMENT_SCHEMA

ZAFF_VDW_CONFINEMENT_SCHEMA = LEGACY_ZAFF_VDW_CONFINEMENT_SCHEMA


@njit(cache=True)
def _single_type_energy_gradient(
    xyz: np.ndarray,
    center: np.ndarray,
    semiaxes: np.ndarray,
    rotation: np.ndarray,
    radius: float,
    cutoff: float,
    morse: np.ndarray,
    gaussians: np.ndarray,
) -> tuple[float, np.ndarray, bool]:
    """Compiled energy/gradient kernel for sites sharing one UvdW type."""

    energy = 0.0
    gradient = np.zeros_like(xyz)
    all_inside = True
    well_depth, minimum_depth, alpha = morse
    for site in range(xyz.shape[0]):
        delta0 = xyz[site, 0] - center[0]
        delta1 = xyz[site, 1] - center[1]
        delta2 = xyz[site, 2] - center[2]
        local0 = (
            delta0 * rotation[0, 0]
            + delta1 * rotation[1, 0]
            + delta2 * rotation[2, 0]
        )
        local1 = (
            delta0 * rotation[0, 1]
            + delta1 * rotation[1, 1]
            + delta2 * rotation[2, 1]
        )
        local2 = (
            delta0 * rotation[0, 2]
            + delta1 * rotation[1, 2]
            + delta2 * rotation[2, 2]
        )
        scaled0 = local0 / semiaxes[0]
        scaled1 = local1 / semiaxes[1]
        scaled2 = local2 / semiaxes[2]
        rho = math.sqrt(
            scaled0 * scaled0 + scaled1 * scaled1 + scaled2 * scaled2
        )
        if rho >= 1.0:
            all_inside = False
            continue
        depth = radius * (1.0 - rho)
        if depth >= cutoff:
            continue

        exponential = math.exp(-alpha * (depth - minimum_depth))
        radial = well_depth * (exponential * exponential - 2.0 * exponential)
        first = (
            2.0
            * well_depth
            * alpha
            * (exponential - exponential * exponential)
        )
        for term in range(gaussians.shape[0]):
            amplitude = gaussians[term, 0]
            displacement = depth - gaussians[term, 1]
            exponent = gaussians[term, 2]
            gaussian = amplitude * math.exp(-exponent * displacement * displacement)
            radial += gaussian
            first += -2.0 * exponent * displacement * gaussian
        reduced = depth / cutoff
        reduced2 = reduced * reduced
        reduced3 = reduced2 * reduced
        reduced4 = reduced3 * reduced
        reduced5 = reduced4 * reduced
        switch = 1.0 - 10.0 * reduced3 + 15.0 * reduced4 - 6.0 * reduced5
        switch_first = (
            -30.0 * reduced2 + 60.0 * reduced3 - 30.0 * reduced4
        ) / cutoff
        energy += switch * radial
        radial_first = switch_first * radial + switch * first

        inverse_rho = 1.0 / rho
        local_gradient0 = -radius * scaled0 / semiaxes[0] * inverse_rho
        local_gradient1 = -radius * scaled1 / semiaxes[1] * inverse_rho
        local_gradient2 = -radius * scaled2 / semiaxes[2] * inverse_rho
        gradient[site, 0] = radial_first * (
            local_gradient0 * rotation[0, 0]
            + local_gradient1 * rotation[0, 1]
            + local_gradient2 * rotation[0, 2]
        )
        gradient[site, 1] = radial_first * (
            local_gradient0 * rotation[1, 0]
            + local_gradient1 * rotation[1, 1]
            + local_gradient2 * rotation[1, 2]
        )
        gradient[site, 2] = radial_first * (
            local_gradient0 * rotation[2, 0]
            + local_gradient1 * rotation[2, 1]
            + local_gradient2 * rotation[2, 2]
        )
    return energy, gradient, all_inside


@dataclass(frozen=True)
class VdwConfinementResult:
    """Energy and Cartesian derivatives of a frozen mean-field potential."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray


@dataclass(frozen=True)
class VdwConfinementSecondOrderResult:
    """Energy, Cartesian gradient, and full analytic Cartesian Hessian."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray


@njit(cache=True)
def _single_type_energy(
    xyz: np.ndarray,
    center: np.ndarray,
    semiaxes: np.ndarray,
    rotation: np.ndarray,
    radius: float,
    cutoff: float,
    morse: np.ndarray,
    gaussians: np.ndarray,
) -> tuple[float, bool]:
    """Compiled energy-only kernel for sites sharing one UvdW type."""

    energy = 0.0
    all_inside = True
    well_depth, minimum_depth, alpha = morse
    for site in range(xyz.shape[0]):
        delta0 = xyz[site, 0] - center[0]
        delta1 = xyz[site, 1] - center[1]
        delta2 = xyz[site, 2] - center[2]
        local0 = (
            delta0 * rotation[0, 0]
            + delta1 * rotation[1, 0]
            + delta2 * rotation[2, 0]
        )
        local1 = (
            delta0 * rotation[0, 1]
            + delta1 * rotation[1, 1]
            + delta2 * rotation[2, 1]
        )
        local2 = (
            delta0 * rotation[0, 2]
            + delta1 * rotation[1, 2]
            + delta2 * rotation[2, 2]
        )
        scaled0 = local0 / semiaxes[0]
        scaled1 = local1 / semiaxes[1]
        scaled2 = local2 / semiaxes[2]
        rho = math.sqrt(
            scaled0 * scaled0 + scaled1 * scaled1 + scaled2 * scaled2
        )
        if rho >= 1.0:
            all_inside = False
            continue
        depth = radius * (1.0 - rho)
        if depth >= cutoff:
            continue
        exponential = math.exp(-alpha * (depth - minimum_depth))
        radial = well_depth * (exponential * exponential - 2.0 * exponential)
        for term in range(gaussians.shape[0]):
            amplitude = gaussians[term, 0]
            displacement = depth - gaussians[term, 1]
            exponent = gaussians[term, 2]
            radial += amplitude * math.exp(-exponent * displacement * displacement)
        reduced = depth / cutoff
        reduced2 = reduced * reduced
        reduced3 = reduced2 * reduced
        reduced4 = reduced3 * reduced
        reduced5 = reduced4 * reduced
        switch = 1.0 - 10.0 * reduced3 + 15.0 * reduced4 - 6.0 * reduced5
        energy += switch * radial
    return energy, all_inside


@dataclass(frozen=True)
class EllipsoidalVdwConfinement:
    """Morse--Gaussian outer-layer potential on homothetic ellipsoidal shells.

    Each type has one Morse baseline ``(well_depth_hartree,
    minimum_depth_bohr, alpha_per_bohr)`` and signed Gaussian corrections
    ``(amplitude_hartree, center_depth_bohr, exponent_per_bohr2)``.  A
    quintic compact switch makes the potential and its first two derivatives
    vanish at ``layer_depth_bohr``.
    """

    center_bohr: np.ndarray
    semiaxes_bohr: np.ndarray
    rotation: np.ndarray
    morse_parameters: Mapping[str, Sequence[float]]
    gaussian_terms: Mapping[str, Sequence[Sequence[float]]]
    layer_depth_bohr: float

    @cached_property
    def _packed_type_parameters(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label in self.morse_parameters:
            morse = np.asarray(self.morse_parameters[label], dtype=float)
            gaussians = np.asarray(self.gaussian_terms[label], dtype=float)
            morse.setflags(write=False)
            gaussians.setflags(write=False)
            packed[label] = morse, gaussians
        return packed

    def __post_init__(self) -> None:
        center = np.asarray(self.center_bohr, dtype=float).reshape(3)
        axes = np.asarray(self.semiaxes_bohr, dtype=float).reshape(3)
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        depth = float(self.layer_depth_bohr)
        if (
            np.any(~np.isfinite(center))
            or np.any(~np.isfinite(axes))
            or np.any(axes <= 0.0)
        ):
            raise ValueError("UvdW center and semiaxes must be finite and positive")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12):
            raise ValueError("UvdW ellipsoid rotation must be orthogonal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("UvdW ellipsoid rotation must be right handed")
        volume_radius = float(np.prod(axes) ** (1.0 / 3.0))
        if not math.isfinite(depth) or not 0.0 < depth < volume_radius:
            raise ValueError(
                "UvdW layer depth must be positive and smaller than the "
                "volume-equivalent radius"
            )
        morse = {
            str(label): tuple(float(value) for value in values)
            for label, values in self.morse_parameters.items()
        }
        terms = {
            str(label): tuple(
                tuple(float(value) for value in raw_term)
                for raw_term in raw_terms
            )
            for label, raw_terms in self.gaussian_terms.items()
        }
        if (
            not morse
            or any(
                len(values) != 3
                or not np.all(np.isfinite(values))
                or values[0] < 0.0
                or not 0.0 <= values[1] <= depth
                or values[2] <= 0.0
                for values in morse.values()
            )
            or set(morse) != set(terms)
            or any(
                not values
                or any(
                    len(term) != 3
                    or not np.all(np.isfinite(term))
                    or term[2] <= 0.0
                    or not 0.0 <= term[1] <= depth
                    for term in values
                )
                for values in terms.values()
            )
        ):
            raise ValueError(
                "every UvdW type needs a valid Morse baseline and finite "
                "Gaussian amplitude, center, and positive exponent"
            )
        object.__setattr__(self, "center_bohr", center)
        object.__setattr__(self, "semiaxes_bohr", axes)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "morse_parameters", morse)
        object.__setattr__(self, "gaussian_terms", terms)
        object.__setattr__(self, "layer_depth_bohr", depth)

    @property
    def volume_equivalent_radius_bohr(self) -> float:
        return float(np.prod(self.semiaxes_bohr) ** (1.0 / 3.0))

    def evaluate(
        self, coordinates_bohr: np.ndarray, site_types: Sequence[str]
    ) -> VdwConfinementResult:
        """Evaluate energy and exact analytic Cartesian gradient."""

        xyz, labels = self._validated_system(coordinates_bohr, site_types)
        energy, gradient, _ = self._derivatives(xyz, labels, second=False)
        return VdwConfinementResult(float(energy), gradient.reshape(-1))

    def evaluate_single_type(
        self,
        coordinates_bohr: np.ndarray,
        site_type: str,
    ) -> VdwConfinementResult:
        """Evaluate same-type sites with the compiled analytic E/G kernel."""

        xyz = np.asarray(coordinates_bohr, dtype=float)
        label = str(site_type)
        if (
            xyz.ndim != 2
            or xyz.shape[1:] != (3,)
            or np.any(~np.isfinite(xyz))
            or label not in self._packed_type_parameters
        ):
            raise ValueError("UvdW coordinates or site type are inconsistent")
        morse, gaussians = self._packed_type_parameters[label]
        energy, gradient, inside = _single_type_energy_gradient(
            xyz,
            self.center_bohr,
            self.semiaxes_bohr,
            self.rotation,
            self.volume_equivalent_radius_bohr,
            self.layer_depth_bohr,
            morse,
            gaussians,
        )
        if not inside:
            raise ValueError("every UvdW site must lie inside confinement")
        return VdwConfinementResult(float(energy), gradient.reshape(-1))

    def energy_single_type(
        self,
        coordinates_bohr: np.ndarray,
        site_type: str,
    ) -> float:
        """Evaluate only the energy for same-type sites."""

        xyz = np.asarray(coordinates_bohr, dtype=float)
        label = str(site_type)
        if (
            xyz.ndim != 2
            or xyz.shape[1:] != (3,)
            or np.any(~np.isfinite(xyz))
            or label not in self._packed_type_parameters
        ):
            raise ValueError("UvdW coordinates or site type are inconsistent")
        morse, gaussians = self._packed_type_parameters[label]
        energy, inside = _single_type_energy(
            xyz,
            self.center_bohr,
            self.semiaxes_bohr,
            self.rotation,
            self.volume_equivalent_radius_bohr,
            self.layer_depth_bohr,
            morse,
            gaussians,
        )
        if not inside:
            raise ValueError("every UvdW site must lie inside confinement")
        return float(energy)

    def evaluate_single_type_with_hessian(
        self,
        coordinates_bohr: np.ndarray,
        site_type: str,
    ) -> VdwConfinementSecondOrderResult:
        """Evaluate E/G/H analytically for sites sharing one UvdW type."""

        xyz = np.asarray(coordinates_bohr, dtype=float)
        label = str(site_type)
        if (
            xyz.ndim != 2
            or xyz.shape[1:] != (3,)
            or np.any(~np.isfinite(xyz))
            or label not in self._packed_type_parameters
        ):
            raise ValueError("UvdW coordinates or site type are inconsistent")
        self._validate_inside(xyz)
        energy, gradient, hessians = self._derivatives(
            xyz,
            (label,) * len(xyz),
            second=True,
        )
        dimension = 3 * len(xyz)
        hessian = np.zeros((dimension, dimension), dtype=float)
        for site in range(len(xyz)):
            start = 3 * site
            hessian[start : start + 3, start : start + 3] = hessians[site]
        return VdwConfinementSecondOrderResult(
            float(energy),
            gradient.reshape(-1),
            hessian,
        )

    def hessian_vector_product(
        self,
        coordinates_bohr: np.ndarray,
        site_types: Sequence[str],
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        """Apply the exact analytic Cartesian Hessian for a frozen ellipsoid."""

        xyz, labels = self._validated_system(coordinates_bohr, site_types)
        direction = np.asarray(vector_bohr, dtype=float)
        if direction.size == xyz.size:
            direction = direction.reshape(xyz.shape)
        if direction.shape != xyz.shape or np.any(~np.isfinite(direction)):
            raise ValueError("UvdW Hessian-vector dimensions are inconsistent")
        _, _, hessians = self._derivatives(xyz, labels, second=True)
        return np.einsum("nij,nj->ni", hessians, direction).reshape(-1)

    def batch_energies(
        self, coordinates_bohr: np.ndarray, site_types: Sequence[str]
    ) -> np.ndarray:
        """Vectorized energy-only evaluation for MC or GA populations."""

        geometries = np.asarray(coordinates_bohr, dtype=float)
        labels = tuple(str(value) for value in site_types)
        if (
            geometries.ndim != 3
            or geometries.shape[1:] != (len(labels), 3)
            or np.any(~np.isfinite(geometries))
        ):
            raise ValueError("UvdW batch must have shape (ngeometry, nsite, 3)")
        flattened = geometries.reshape(-1, 3)
        repeated_labels = labels * len(geometries)
        self._validate_inside(flattened)
        # Recover per-geometry energies without a Python loop over candidates.
        site_energies = self._site_energy(flattened, repeated_labels)
        return site_energies.reshape(len(geometries), len(labels)).sum(axis=1)

    def _site_energy(
        self, xyz: np.ndarray, labels: Sequence[str]
    ) -> np.ndarray:
        depth, _, _ = self._depth_derivatives(xyz, second=False)
        values = np.zeros(len(xyz))
        active = depth < self.layer_depth_bohr
        for label in set(labels):
            selected = active & (np.asarray(labels, dtype=object) == label)
            if np.any(selected):
                values[selected] = _switched_morse_gaussians(
                    depth[selected],
                    self.morse_parameters[label],
                    self.gaussian_terms[label],
                    self.layer_depth_bohr,
                )[0]
        return values

    def _derivatives(
        self, xyz: np.ndarray, labels: Sequence[str], *, second: bool
    ) -> tuple[float, np.ndarray, np.ndarray]:
        depth, depth_gradient, depth_hessian = self._depth_derivatives(
            xyz, second=second
        )
        site_energy = np.zeros(len(xyz))
        first = np.zeros(len(xyz))
        second_derivative = np.zeros(len(xyz))
        active = depth < self.layer_depth_bohr
        label_array = np.asarray(labels, dtype=object)
        for label in set(labels):
            selected = active & (label_array == label)
            if not np.any(selected):
                continue
            value, derivative, curvature = _switched_morse_gaussians(
                depth[selected],
                self.morse_parameters[label],
                self.gaussian_terms[label],
                self.layer_depth_bohr,
            )
            site_energy[selected] = value
            first[selected] = derivative
            second_derivative[selected] = curvature
        gradient = first[:, None] * depth_gradient
        hessian = np.zeros((len(xyz), 3, 3))
        if second:
            hessian = (
                second_derivative[:, None, None]
                * depth_gradient[:, :, None]
                * depth_gradient[:, None, :]
                + first[:, None, None] * depth_hessian
            )
        return float(np.sum(site_energy)), gradient, hessian

    def _depth_derivatives(
        self, xyz: np.ndarray, *, second: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta = xyz - self.center_bohr
        local = delta @ self.rotation
        scaled = local / self.semiaxes_bohr
        rho = np.linalg.norm(scaled, axis=1)
        radius = self.volume_equivalent_radius_bohr
        depth = radius * (1.0 - rho)
        gradient = np.zeros_like(xyz)
        hessian = np.zeros((len(xyz), 3, 3))
        active = depth < self.layer_depth_bohr
        if not np.any(active):
            return depth, gradient, hessian
        inverse_metric = (
            self.rotation
            @ np.diag(1.0 / self.semiaxes_bohr**2)
            @ self.rotation.T
        )
        metric_delta = delta[active] @ inverse_metric
        active_rho = rho[active]
        rho_gradient = metric_delta / active_rho[:, None]
        gradient[active] = -radius * rho_gradient
        if second:
            rho_hessian = (
                inverse_metric[None, :, :] / active_rho[:, None, None]
                - metric_delta[:, :, None]
                * metric_delta[:, None, :]
                / active_rho[:, None, None] ** 3
            )
            hessian[active] = -radius * rho_hessian
        return depth, gradient, hessian

    def _validated_system(
        self, coordinates_bohr: np.ndarray, site_types: Sequence[str]
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        labels = tuple(str(value) for value in site_types)
        if (
            xyz.shape != (len(labels), 3)
            or np.any(~np.isfinite(xyz))
            or any(label not in self.gaussian_terms for label in labels)
        ):
            raise ValueError("UvdW coordinates or site types are inconsistent")
        self._validate_inside(xyz)
        return xyz, labels

    def _validate_inside(self, xyz: np.ndarray) -> None:
        local = (xyz - self.center_bohr) @ self.rotation
        rho = np.linalg.norm(local / self.semiaxes_bohr, axis=1)
        if np.any(rho >= 1.0):
            raise ValueError("every UvdW site must lie inside confinement")


def _switched_morse_gaussians(
    depth: np.ndarray,
    morse_parameters: Sequence[float],
    terms: Sequence[Sequence[float]],
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Morse--Gaussian expansion times a quintic C2 switch."""

    d = np.asarray(depth, dtype=float)
    well_depth, minimum_depth, alpha = (
        float(value) for value in morse_parameters
    )
    morse_exponential = np.exp(-alpha * (d - minimum_depth))
    morse = well_depth * (morse_exponential**2 - 2.0 * morse_exponential)
    morse_first = (
        2.0 * well_depth * alpha
        * (morse_exponential - morse_exponential**2)
    )
    morse_second = (
        2.0 * well_depth * alpha**2
        * (2.0 * morse_exponential**2 - morse_exponential)
    )
    basis = np.asarray(terms, dtype=float)
    amplitude = basis[:, 0, None]
    displacement = d[None, :] - basis[:, 1, None]
    exponent = basis[:, 2, None]
    gaussian_terms = amplitude * np.exp(-exponent * displacement**2)
    radial = morse + np.sum(gaussian_terms, axis=0)
    first = morse_first + np.sum(
        -2.0 * exponent * displacement * gaussian_terms, axis=0
    )
    second = morse_second + np.sum(
        (4.0 * exponent**2 * displacement**2 - 2.0 * exponent)
        * gaussian_terms,
        axis=0,
    )
    x = d / float(cutoff)
    switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    switch_first = (-30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4) / cutoff
    switch_second = (-60.0 * x + 180.0 * x**2 - 120.0 * x**3) / cutoff**2
    return (
        switch * radial,
        switch_first * radial + switch * first,
        switch_second * radial + 2.0 * switch_first * first + switch * second,
    )


def _switched_repulsive_exponential(
    depth: np.ndarray,
    wall_height_hartree: float,
    decay_per_bohr: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a positive monotone exponential wall times a quintic C2 switch."""

    d = np.asarray(depth, dtype=float)
    alpha = float(decay_per_bohr)
    radial = float(wall_height_hartree) * np.exp(-alpha * d)
    x = d / float(cutoff)
    switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    switch_first = (-30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4) / cutoff
    switch_second = (
        -60.0 * x + 180.0 * x**2 - 120.0 * x**3
    ) / cutoff**2
    return (
        switch * radial,
        radial * (switch_first - alpha * switch),
        radial
        * (switch_second - 2.0 * alpha * switch_first + alpha**2 * switch),
    )


class EllipsoidalRepulsiveSiteBoundary(EllipsoidalVdwConfinement):
    """Strictly repulsive all-site boundary with analytic E/G/H."""

    def _radial_derivatives(
        self, depth: np.ndarray, label: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, _unused, decay = self.morse_parameters[label]
        return _switched_repulsive_exponential(
            depth,
            height,
            decay,
            self.layer_depth_bohr,
        )

    def evaluate_single_type(
        self,
        coordinates_bohr: np.ndarray,
        site_type: str,
    ) -> VdwConfinementResult:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        label = str(site_type)
        if (
            xyz.ndim != 2
            or xyz.shape[1:] != (3,)
            or np.any(~np.isfinite(xyz))
            or label not in self.morse_parameters
        ):
            raise ValueError("repulsive-wall coordinates or site type are inconsistent")
        self._validate_inside(xyz)
        energy, gradient, _ = self._derivatives(
            xyz, (label,) * len(xyz), second=False
        )
        return VdwConfinementResult(float(energy), gradient.reshape(-1))

    def energy_single_type(
        self,
        coordinates_bohr: np.ndarray,
        site_type: str,
    ) -> float:
        return self.evaluate_single_type(
            coordinates_bohr, site_type
        ).energy_hartree

    def _site_energy(
        self, xyz: np.ndarray, labels: Sequence[str]
    ) -> np.ndarray:
        depth, _, _ = self._depth_derivatives(xyz, second=False)
        values = np.zeros(len(xyz))
        active = depth < self.layer_depth_bohr
        label_array = np.asarray(labels, dtype=object)
        for label in set(labels):
            selected = active & (label_array == label)
            if np.any(selected):
                values[selected] = self._radial_derivatives(
                    depth[selected], label
                )[0]
        return values

    def _derivatives(
        self, xyz: np.ndarray, labels: Sequence[str], *, second: bool
    ) -> tuple[float, np.ndarray, np.ndarray]:
        depth, depth_gradient, depth_hessian = self._depth_derivatives(
            xyz, second=second
        )
        site_energy = np.zeros(len(xyz))
        first = np.zeros(len(xyz))
        second_derivative = np.zeros(len(xyz))
        active = depth < self.layer_depth_bohr
        label_array = np.asarray(labels, dtype=object)
        for label in set(labels):
            selected = active & (label_array == label)
            if not np.any(selected):
                continue
            value, derivative, curvature = self._radial_derivatives(
                depth[selected], label
            )
            site_energy[selected] = value
            first[selected] = derivative
            second_derivative[selected] = curvature
        gradient = first[:, None] * depth_gradient
        hessian = np.zeros((len(xyz), 3, 3))
        if second:
            hessian = (
                second_derivative[:, None, None]
                * depth_gradient[:, :, None]
                * depth_gradient[:, None, :]
                + first[:, None, None] * depth_hessian
            )
        return float(np.sum(site_energy)), gradient, hessian


def vdw_confinement_to_record(
    confinement: EllipsoidalVdwConfinement,
) -> dict[str, Any]:
    """Serialize a calibrated and frozen mean-field confinement potential."""

    return {
        "schema": EPOC_CONFINEMENT_SCHEMA,
        "boundary_contract": "FROZEN_AFTER_WARMUP",
        "center_bohr": confinement.center_bohr.tolist(),
        "semiaxes_bohr": confinement.semiaxes_bohr.tolist(),
        "rotation": confinement.rotation.tolist(),
        "layer_depth_bohr": float(confinement.layer_depth_bohr),
        "morse_parameters": {
            label: {
                "well_depth_hartree": values[0],
                "minimum_depth_bohr": values[1],
                "alpha_per_bohr": values[2],
            }
            for label, values in confinement.morse_parameters.items()
        },
        "gaussian_terms": {
            label: [
                {
                    "amplitude_hartree": term[0],
                    "center_depth_bohr": term[1],
                    "exponent_per_bohr2": term[2],
                }
                for term in values
            ]
            for label, values in confinement.gaussian_terms.items()
        },
        "shell_coordinate": "HOMOTHETIC_VOLUME_EQUIVALENT_DEPTH",
        "radial_basis": "MORSE_BASELINE_PLUS_SIGNED_GAUSSIANS_IN_INWARD_DEPTH",
        "switch": "QUINTIC_C2_COMPACT",
        "derivative_contract": "ANALYTIC_E_G_HVP",
    }


def vdw_confinement_from_record(
    record: Mapping[str, Any],
) -> EllipsoidalVdwConfinement:
    """Deserialize and validate a frozen mean-field confinement potential."""

    if str(record.get("schema", "")) not in {
        EPOC_CONFINEMENT_SCHEMA,
        LEGACY_ZAFF_VDW_CONFINEMENT_SCHEMA,
    }:
        raise ValueError("unsupported EPOC UvdW confinement record")
    if str(record.get("boundary_contract", "")) != "FROZEN_AFTER_WARMUP":
        raise ValueError("ZAFF UvdW runtime accepts only a frozen confinement")
    return EllipsoidalVdwConfinement(
        center_bohr=np.asarray(record["center_bohr"], dtype=float),
        semiaxes_bohr=np.asarray(record["semiaxes_bohr"], dtype=float),
        rotation=np.asarray(record["rotation"], dtype=float),
        morse_parameters={
            str(label): (
                float(values["well_depth_hartree"]),
                float(values["minimum_depth_bohr"]),
                float(values["alpha_per_bohr"]),
            )
            for label, values in dict(record["morse_parameters"]).items()
        },
        gaussian_terms={
            str(label): [
                (
                    float(term["amplitude_hartree"]),
                    float(term["center_depth_bohr"]),
                    float(term["exponent_per_bohr2"]),
                )
                for term in raw_terms
            ]
            for label, raw_terms in dict(record["gaussian_terms"]).items()
        },
        layer_depth_bohr=float(record["layer_depth_bohr"]),
    )


def attach_vdw_confinement(
    model: Mapping[str, Any], confinement: EllipsoidalVdwConfinement
) -> dict[str, Any]:
    """Return a ZAFF model carrying a calibrated frozen UvdW potential."""

    payload = dict(model)
    payload["vdw_confinement"] = vdw_confinement_to_record(confinement)
    return payload


def ellipsoidal_repulsive_site_boundary(
    center_bohr: np.ndarray,
    semiaxes_bohr: np.ndarray,
    rotation: np.ndarray,
    site_types: Sequence[str],
    *,
    wall_height_hartree: float = 25.0 / 627.5094740631,
    decay_per_bohr: float = 3.0 / 1.8897261254578281,
    layer_depth_bohr: float = 1.5 * 1.8897261254578281,
) -> EllipsoidalRepulsiveSiteBoundary:
    """Build one analytic all-site repulsive wall for arbitrary molecules."""

    labels = tuple(dict.fromkeys(str(value) for value in site_types))
    if not labels:
        raise ValueError("a molecular boundary needs at least one site type")
    parameters = (
        float(wall_height_hartree),
        0.0,
        float(decay_per_bohr),
    )
    gaussian = ((0.0, 0.0, 1.0),)
    return EllipsoidalRepulsiveSiteBoundary(
        center_bohr=np.asarray(center_bohr, dtype=float),
        semiaxes_bohr=np.asarray(semiaxes_bohr, dtype=float),
        rotation=np.asarray(rotation, dtype=float),
        morse_parameters={label: parameters for label in labels},
        gaussian_terms={label: gaussian for label in labels},
        layer_depth_bohr=float(layer_depth_bohr),
    )


def ellipsoidal_molecular_constraint_violation(
    coordinates_bohr: np.ndarray,
    center_bohr: np.ndarray,
    semiaxes_bohr: np.ndarray,
    rotation: np.ndarray,
    *,
    safety_fraction: float = 5.0e-4,
) -> float:
    """Return zero inside the guarded ellipsoid and scaled-radius excess outside."""

    xyz = np.asarray(coordinates_bohr, dtype=float)
    center = np.asarray(center_bohr, dtype=float).reshape(3)
    axes = np.asarray(semiaxes_bohr, dtype=float).reshape(3)
    frame = np.asarray(rotation, dtype=float).reshape(3, 3)
    margin = float(safety_fraction)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or np.any(~np.isfinite(xyz))
        or np.any(~np.isfinite(center))
        or np.any(~np.isfinite(axes))
        or np.any(axes <= 0.0)
        or np.any(~np.isfinite(frame))
        or not 0.0 <= margin < 1.0
    ):
        raise ValueError("ellipsoidal molecular constraint is invalid")
    local = (xyz - center) @ frame
    maximum_rho = float(np.max(np.linalg.norm(local / axes, axis=1)))
    return max(0.0, maximum_rho - (1.0 - margin))


def ellipsoidal_molecular_constraint_violations(
    coordinates_bohr: np.ndarray,
    center_bohr: np.ndarray,
    semiaxes_bohr: np.ndarray,
    rotation: np.ndarray,
    *,
    safety_fraction: float = 5.0e-4,
) -> np.ndarray:
    """Vectorized population form of the guarded ellipsoidal constraint."""

    geometries = np.asarray(coordinates_bohr, dtype=float)
    center = np.asarray(center_bohr, dtype=float).reshape(3)
    axes = np.asarray(semiaxes_bohr, dtype=float).reshape(3)
    frame = np.asarray(rotation, dtype=float).reshape(3, 3)
    margin = float(safety_fraction)
    if (
        geometries.ndim != 3
        or geometries.shape[2] != 3
        or np.any(~np.isfinite(geometries))
        or np.any(~np.isfinite(center))
        or np.any(~np.isfinite(axes))
        or np.any(axes <= 0.0)
        or np.any(~np.isfinite(frame))
        or not 0.0 <= margin < 1.0
    ):
        raise ValueError("ellipsoidal molecular constraint population is invalid")
    local = (geometries - center[None, None, :]) @ frame
    maximum_rho = np.max(np.linalg.norm(local / axes, axis=2), axis=1)
    return np.maximum(0.0, maximum_rho - (1.0 - margin))


__all__ = [
    "EPOC_CONFINEMENT_SCHEMA",
    "ZAFF_VDW_CONFINEMENT_SCHEMA",
    "EllipsoidalRepulsiveSiteBoundary",
    "EllipsoidalVdwConfinement",
    "VdwConfinementResult",
    "VdwConfinementSecondOrderResult",
    "attach_vdw_confinement",
    "ellipsoidal_molecular_constraint_violation",
    "ellipsoidal_molecular_constraint_violations",
    "ellipsoidal_repulsive_site_boundary",
    "vdw_confinement_from_record",
    "vdw_confinement_to_record",
]
