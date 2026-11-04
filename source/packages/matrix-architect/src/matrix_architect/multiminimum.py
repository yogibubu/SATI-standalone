"""Joint multi-minimum torsions, hydrogen bonds and local charge response."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FourierStationaryPoint:
    """Energy, gradient and Hessian constraints at one periodic minimum."""

    phases_radian: tuple[float, ...]
    energy_hartree: float
    hessian_hartree: np.ndarray
    gradient_hartree: np.ndarray | None = None
    label: str = ""

    def __post_init__(self) -> None:
        phases = np.asarray(self.phases_radian, dtype=float).reshape(-1)
        hessian = np.asarray(self.hessian_hartree, dtype=float)
        gradient = (
            np.zeros(len(phases), dtype=float)
            if self.gradient_hartree is None
            else np.asarray(self.gradient_hartree, dtype=float).reshape(-1)
        )
        if not len(phases) or hessian.shape != (len(phases), len(phases)):
            raise ValueError("Fourier stationary-point dimensions are inconsistent")
        if gradient.shape != phases.shape:
            raise ValueError("Fourier stationary-point gradient has an invalid shape")
        if not all(
            np.all(np.isfinite(values)) for values in (phases, gradient, hessian)
        ) or not isfinite(float(self.energy_hartree)):
            raise ValueError("Fourier stationary-point data must be finite")
        if not np.allclose(hessian, hessian.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("Fourier stationary-point Hessian must be symmetric")
        object.__setattr__(self, "phases_radian", tuple(float(value) for value in phases))
        object.__setattr__(self, "gradient_hartree", gradient)
        object.__setattr__(self, "hessian_hartree", hessian)


@dataclass(frozen=True)
class FourierEnergyPoint:
    """One periodic energy observation used to control the global surface."""

    phases_radian: tuple[float, ...]
    energy_hartree: float
    label: str = ""

    def __post_init__(self) -> None:
        phases = np.asarray(self.phases_radian, dtype=float).reshape(-1)
        if (
            not len(phases)
            or not np.all(np.isfinite(phases))
            or not isfinite(float(self.energy_hartree))
        ):
            raise ValueError("Fourier energy-point data must be finite and nonempty")
        object.__setattr__(self, "phases_radian", tuple(float(value) for value in phases))


@dataclass(frozen=True)
class MultiMinimumFourierSurface:
    """Periodic multidimensional Fourier surface with analytic E/G/H."""

    wavevectors: tuple[tuple[int, ...], ...]
    constant_hartree: float
    cosine_coefficients_hartree: tuple[float, ...]
    sine_coefficients_hartree: tuple[float, ...]
    constraint_maximum_residual: float = 0.0
    energy_point_rmse_hartree: float = 0.0
    coefficient_regularization: float = 0.0

    def __post_init__(self) -> None:
        waves = tuple(tuple(int(value) for value in wave) for wave in self.wavevectors)
        if not waves or not waves[0] or any(len(wave) != len(waves[0]) for wave in waves):
            raise ValueError("Fourier wavevectors must share one nonzero dimension")
        if any(not any(wave) for wave in waves) or len(set(waves)) != len(waves):
            raise ValueError("Fourier wavevectors must be unique and nonzero")
        if len(self.cosine_coefficients_hartree) != len(waves) or len(
            self.sine_coefficients_hartree
        ) != len(waves):
            raise ValueError("Fourier coefficient counts differ from the wavevectors")
        values = (
            float(self.constant_hartree),
            float(self.constraint_maximum_residual),
            float(self.energy_point_rmse_hartree),
            float(self.coefficient_regularization),
            *self.cosine_coefficients_hartree,
            *self.sine_coefficients_hartree,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Fourier surface coefficients must be finite")
        object.__setattr__(self, "wavevectors", waves)

    @property
    def dimension(self) -> int:
        return len(self.wavevectors[0])

    def derivatives(self, phases_radian: Sequence[float]) -> tuple[float, np.ndarray, np.ndarray]:
        phases = np.asarray(phases_radian, dtype=float).reshape(-1)
        if phases.shape != (self.dimension,) or not np.all(np.isfinite(phases)):
            raise ValueError("Fourier phase vector has an invalid shape")
        waves = np.asarray(self.wavevectors, dtype=float)
        arguments = waves @ phases
        cosine = np.asarray(self.cosine_coefficients_hartree, dtype=float)
        sine = np.asarray(self.sine_coefficients_hartree, dtype=float)
        cos_arg = np.cos(arguments)
        sin_arg = np.sin(arguments)
        energy = float(self.constant_hartree + cosine @ cos_arg + sine @ sin_arg)
        first_factors = -cosine * sin_arg + sine * cos_arg
        second_factors = -cosine * cos_arg - sine * sin_arg
        gradient = waves.T @ first_factors
        hessian = waves.T @ (second_factors[:, None] * waves)
        return energy, gradient, 0.5 * (hessian + hessian.T)

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_form": "MULTIMINIMUM_FOURIER_ND",
            "wavevectors": [list(wave) for wave in self.wavevectors],
            "constant_hartree": self.constant_hartree,
            "cosine_coefficients_hartree": list(self.cosine_coefficients_hartree),
            "sine_coefficients_hartree": list(self.sine_coefficients_hartree),
            "constraint_maximum_residual": self.constraint_maximum_residual,
            "energy_point_rmse_hartree": self.energy_point_rmse_hartree,
            "coefficient_regularization": self.coefficient_regularization,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MultiMinimumFourierSurface":
        if payload.get("functional_form") != "MULTIMINIMUM_FOURIER_ND":
            raise ValueError("unsupported multi-minimum Fourier functional form")
        return cls(
            wavevectors=tuple(tuple(int(value) for value in row) for row in payload["wavevectors"]),
            constant_hartree=float(payload["constant_hartree"]),
            cosine_coefficients_hartree=tuple(
                float(value) for value in payload["cosine_coefficients_hartree"]
            ),
            sine_coefficients_hartree=tuple(
                float(value) for value in payload["sine_coefficients_hartree"]
            ),
            constraint_maximum_residual=float(payload.get("constraint_maximum_residual", 0.0)),
            energy_point_rmse_hartree=float(payload.get("energy_point_rmse_hartree", 0.0)),
            coefficient_regularization=float(payload.get("coefficient_regularization", 0.0)),
        )


def fit_multiminimum_fourier_surface(
    stationary_points: Sequence[FourierStationaryPoint],
    *,
    wavevectors: Sequence[Sequence[int]],
    energy_points: Sequence[FourierEnergyPoint] = (),
    coefficient_regularization: float = 1.0e-6,
    maximum_constraint_residual: float = 1.0e-9,
) -> MultiMinimumFourierSurface:
    """Fit one minimum-norm Fourier surface to exact E/G/H constraints."""

    points = tuple(stationary_points)
    controls = tuple(energy_points)
    waves = tuple(tuple(int(value) for value in wave) for wave in wavevectors)
    if not points:
        raise ValueError("a multi-minimum Fourier fit needs stationary points")
    dimension = len(points[0].phases_radian)
    if (
        any(len(point.phases_radian) != dimension for point in points)
        or any(len(wave) != dimension for wave in waves)
        or any(len(point.phases_radian) != dimension for point in controls)
    ):
        raise ValueError("Fourier fit dimensions are inconsistent")
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for point in points:
        phases = np.asarray(point.phases_radian, dtype=float)
        for derivative in _constraint_derivatives(dimension):
            rows.append(_fourier_constraint_row(phases, waves, derivative))
            if len(derivative) == 0:
                targets.append(float(point.energy_hartree))
            elif len(derivative) == 1:
                targets.append(float(point.gradient_hartree[derivative[0]]))
            else:
                targets.append(float(point.hessian_hartree[derivative]))
    constraints = np.vstack(rows)
    constrained_values = np.asarray(targets, dtype=float)
    regularization = float(coefficient_regularization)
    if not isfinite(regularization) or regularization < 0.0:
        raise ValueError("Fourier coefficient regularization must be finite and nonnegative")
    if controls:
        design = np.vstack(
            [
                _fourier_constraint_row(np.asarray(point.phases_radian, dtype=float), waves, ())
                for point in controls
            ]
        )
        observed = np.asarray([point.energy_hartree for point in controls], dtype=float)
        penalty = np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        normal = design.T @ design + regularization * penalty
        kkt = np.block(
            [
                [normal, constraints.T],
                [constraints, np.zeros((len(constraints), len(constraints)), dtype=float)],
            ]
        )
        solution, *_ = np.linalg.lstsq(
            kkt,
            np.concatenate(
                (
                    design.T @ observed,
                    constrained_values,
                )
            ),
            rcond=1.0e-12,
        )
        coefficients = solution[: design.shape[1]]
        energy_rmse = float(np.sqrt(np.mean((design @ coefficients - observed) ** 2)))
    else:
        coefficients, *_ = np.linalg.lstsq(constraints, constrained_values, rcond=1.0e-12)
        energy_rmse = 0.0
    residual = constraints @ coefficients - constrained_values
    maximum = float(np.max(np.abs(residual)))
    if maximum > float(maximum_constraint_residual):
        raise ValueError(
            "Fourier basis cannot satisfy the stationary-point constraints: "
            f"maximum residual {maximum:.6g}"
        )
    count = len(waves)
    return MultiMinimumFourierSurface(
        wavevectors=waves,
        constant_hartree=float(coefficients[0]),
        cosine_coefficients_hartree=tuple(float(value) for value in coefficients[1 : 1 + count]),
        sine_coefficients_hartree=tuple(float(value) for value in coefficients[1 + count :]),
        constraint_maximum_residual=maximum,
        energy_point_rmse_hartree=energy_rmse,
        coefficient_regularization=regularization,
    )


@dataclass(frozen=True)
class MorsePolynomialHydrogenBond:
    """Short-ranged, switched H-bond potential in the Morse exponential.

    ``V(r) = sum_n c_n u^n`` with ``u=exp[-beta(r-r0)]``.  The required Morse
    core is ``c1=-2D, c2=D``; optional orders 3 and above refine the well while
    a quintic C2 switch makes energy, gradient, and Hessian vanish exactly at
    the finite outer cutoff.
    """

    donor_hydrogen: int
    acceptor: int
    reference_distance_angstrom: float
    beta_per_angstrom: float
    depth_hartree: float
    polynomial_coefficients_hartree: tuple[float, ...] = ()
    switch_on_angstrom: float | None = None
    switch_off_angstrom: float | None = None

    def __post_init__(self) -> None:
        if self.donor_hydrogen < 0 or self.acceptor < 0 or self.donor_hydrogen == self.acceptor:
            raise ValueError("hydrogen-bond atom indices must be distinct and nonnegative")
        values = (
            self.reference_distance_angstrom,
            self.beta_per_angstrom,
            self.depth_hartree,
            *self.polynomial_coefficients_hartree,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("hydrogen-bond parameters must be finite")
        if self.reference_distance_angstrom <= 0.0 or self.beta_per_angstrom <= 0.0:
            raise ValueError("hydrogen-bond distance and decay must be positive")
        if self.depth_hartree <= 0.0:
            raise ValueError("hydrogen-bond Morse depth must be positive")
        switch_on = (
            self.reference_distance_angstrom + 1.0
            if self.switch_on_angstrom is None
            else float(self.switch_on_angstrom)
        )
        switch_off = (
            self.reference_distance_angstrom + 2.0
            if self.switch_off_angstrom is None
            else float(self.switch_off_angstrom)
        )
        if not self.reference_distance_angstrom < switch_on < switch_off:
            raise ValueError("hydrogen-bond switch must satisfy r0 < r_on < r_off")
        object.__setattr__(self, "switch_on_angstrom", switch_on)
        object.__setattr__(self, "switch_off_angstrom", switch_off)

    @property
    def coefficients_hartree(self) -> tuple[float, ...]:
        return (
            -2.0 * float(self.depth_hartree),
            float(self.depth_hartree),
            *tuple(float(value) for value in self.polynomial_coefficients_hartree),
        )

    def derivatives(self, distance_angstrom: float) -> tuple[float, float, float]:
        distance = float(distance_angstrom)
        if not isfinite(distance) or distance <= 0.0:
            raise ValueError("hydrogen-bond distance must be finite and positive")
        u = exp(-self.beta_per_angstrom * (distance - self.reference_distance_angstrom))
        orders = np.arange(1, len(self.coefficients_hartree) + 1, dtype=float)
        coefficients = np.asarray(self.coefficients_hartree, dtype=float)
        powers = u**orders
        bare = float(coefficients @ powers)
        bare_first = float(-self.beta_per_angstrom * ((orders * coefficients) @ powers))
        bare_second = float(
            self.beta_per_angstrom**2 * (((orders**2) * coefficients) @ powers)
        )
        switch, switch_first, switch_second = self._switch_derivatives(distance)
        return (
            switch * bare,
            switch_first * bare + switch * bare_first,
            switch_second * bare + 2.0 * switch_first * bare_first + switch * bare_second,
        )

    def _switch_derivatives(self, distance_angstrom: float) -> tuple[float, float, float]:
        """C2 quintic switch and its first two radial derivatives."""

        distance = float(distance_angstrom)
        on = float(self.switch_on_angstrom)
        off = float(self.switch_off_angstrom)
        if distance <= on:
            return 1.0, 0.0, 0.0
        if distance >= off:
            return 0.0, 0.0, 0.0
        width = off - on
        x = (distance - on) / width
        switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
        first = (-30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4) / width
        second = (-60.0 * x + 180.0 * x**2 - 120.0 * x**3) / width**2
        return float(switch), float(first), float(second)

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_form": "MORSE_EXPONENTIAL_POLYNOMIAL",
            "donor_hydrogen_zero_based": self.donor_hydrogen,
            "acceptor_zero_based": self.acceptor,
            "reference_distance_angstrom": self.reference_distance_angstrom,
            "beta_per_angstrom": self.beta_per_angstrom,
            "depth_hartree": self.depth_hartree,
            "polynomial_coefficients_hartree_orders_3_plus": list(
                self.polynomial_coefficients_hartree
            ),
            "radial_switch": {
                "family": "C2_QUINTIC",
                "on_angstrom": self.switch_on_angstrom,
                "off_angstrom": self.switch_off_angstrom,
            },
            "dissociation_limit_hartree": 0.0,
        }


@dataclass(frozen=True)
class ExponentialPairChargeTransfer:
    """Charge-conserving H/acceptor response depending only on H---X distance."""

    donor_hydrogen: int
    acceptor: int
    amplitude_e: float
    beta_per_angstrom: float
    reference_distance_angstrom: float

    def delta_charge_e(self, distance_angstrom: float) -> float:
        distance = float(distance_angstrom)
        if not isfinite(distance) or distance <= 0.0:
            raise ValueError("charge-transfer distance must be finite and positive")
        return float(
            self.amplitude_e
            * exp(-self.beta_per_angstrom * (distance - self.reference_distance_angstrom))
        )

    def apply(self, charges_e: Sequence[float], distance_angstrom: float) -> np.ndarray:
        charges = np.asarray(charges_e, dtype=float).copy()
        if max(self.donor_hydrogen, self.acceptor) >= len(charges):
            raise ValueError("charge-transfer atom index exceeds the charge vector")
        delta = self.delta_charge_e(distance_angstrom)
        charges[self.donor_hydrogen] += delta
        charges[self.acceptor] -= delta
        return charges

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_form": "PAIR_CONSERVING_EXPONENTIAL_CHARGE_TRANSFER",
            "donor_hydrogen_zero_based": self.donor_hydrogen,
            "acceptor_zero_based": self.acceptor,
            "amplitude_e": self.amplitude_e,
            "beta_per_angstrom": self.beta_per_angstrom,
            "reference_distance_angstrom": self.reference_distance_angstrom,
            "charge_conservation": "q_H += delta_q; q_X -= delta_q",
        }


@dataclass(frozen=True)
class HydrogenBondNonbondedModel:
    """Morse-polynomial H bonds plus strictly pair-local charge response."""

    baseline_charges_e: tuple[float, ...]
    potentials: tuple[MorsePolynomialHydrogenBond, ...]
    charge_transfers: tuple[ExponentialPairChargeTransfer, ...]

    def __post_init__(self) -> None:
        charges = np.asarray(self.baseline_charges_e, dtype=float).reshape(-1)
        if not len(charges) or not np.all(np.isfinite(charges)):
            raise ValueError("baseline charges must be a finite nonempty vector")
        if abs(float(np.sum(charges))) > 1.0e-8:
            raise ValueError("baseline charges must preserve the molecular charge")
        if any(
            max(term.donor_hydrogen, term.acceptor) >= len(charges)
            for term in (*self.potentials, *self.charge_transfers)
        ):
            raise ValueError("hydrogen-bond model atom index exceeds the charge vector")
        object.__setattr__(self, "baseline_charges_e", tuple(float(value) for value in charges))

    def charges(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        coordinates = self._coordinates(coordinates_angstrom)
        charges = np.asarray(self.baseline_charges_e, dtype=float)
        for transfer in self.charge_transfers:
            distance = np.linalg.norm(
                coordinates[transfer.donor_hydrogen] - coordinates[transfer.acceptor]
            )
            charges = transfer.apply(charges, float(distance))
        return charges

    def energy_gradient_hessian(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        coordinates = self._coordinates(coordinates_angstrom)
        gradient = np.zeros_like(coordinates)
        hessian = np.zeros((3 * len(coordinates), 3 * len(coordinates)), dtype=float)
        energy = 0.0
        identity = np.eye(3)
        for potential in self.potentials:
            hydrogen = potential.donor_hydrogen
            acceptor = potential.acceptor
            vector = coordinates[hydrogen] - coordinates[acceptor]
            distance = float(np.linalg.norm(vector))
            unit = vector / distance
            value, first, second = potential.derivatives(distance)
            energy += value
            gradient[hydrogen] += first * unit
            gradient[acceptor] -= first * unit
            block = second * np.outer(unit, unit) + (first / distance) * (
                identity - np.outer(unit, unit)
            )
            hs = slice(3 * hydrogen, 3 * hydrogen + 3)
            xs = slice(3 * acceptor, 3 * acceptor + 3)
            hessian[hs, hs] += block
            hessian[xs, xs] += block
            hessian[hs, xs] -= block
            hessian[xs, hs] -= block
        return float(energy), gradient.reshape(-1), 0.5 * (hessian + hessian.T)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "matrix.zaff.hydrogen_bond_nonbonded.v2",
            "baseline_charges_e": list(self.baseline_charges_e),
            "potentials": [item.to_dict() for item in self.potentials],
            "charge_transfers": [item.to_dict() for item in self.charge_transfers],
            "variable_charge_atoms_zero_based": sorted(
                {
                    atom
                    for item in self.charge_transfers
                    for atom in (item.donor_hydrogen, item.acceptor)
                }
            ),
        }

    def _coordinates(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.baseline_charges_e), 3) or not np.all(
            np.isfinite(coordinates)
        ):
            raise ValueError("hydrogen-bond model coordinates have an invalid shape")
        return coordinates


@dataclass(frozen=True)
class JointMultiminimumEvaluation:
    """Separated derivatives of the joint torsional/nonbonded model."""

    energy_hartree: float
    torsional_gradient_hartree: np.ndarray
    torsional_hessian_hartree: np.ndarray
    hydrogen_bond_gradient_hartree_per_angstrom: np.ndarray
    hydrogen_bond_hessian_hartree_per_angstrom2: np.ndarray
    atomic_charges_e: np.ndarray


@dataclass(frozen=True)
class JointMultiminimumModel:
    """Evaluable Fourier plus Morse-polynomial H-bond/charge model."""

    torsional_surface: MultiMinimumFourierSurface
    hydrogen_bond_model: HydrogenBondNonbondedModel

    def evaluate(
        self, phases_radian: Sequence[float], coordinates_angstrom: np.ndarray
    ) -> JointMultiminimumEvaluation:
        torsional_energy, torsional_gradient, torsional_hessian = (
            self.torsional_surface.derivatives(phases_radian)
        )
        hbond_energy, hbond_gradient, hbond_hessian = (
            self.hydrogen_bond_model.energy_gradient_hessian(coordinates_angstrom)
        )
        return JointMultiminimumEvaluation(
            energy_hartree=float(torsional_energy + hbond_energy),
            torsional_gradient_hartree=torsional_gradient,
            torsional_hessian_hartree=torsional_hessian,
            hydrogen_bond_gradient_hartree_per_angstrom=hbond_gradient,
            hydrogen_bond_hessian_hartree_per_angstrom2=hbond_hessian,
            atomic_charges_e=self.hydrogen_bond_model.charges(coordinates_angstrom),
        )


def fit_two_state_pair_charge_transfer(
    *,
    active_distance_angstrom: float,
    inactive_distance_angstrom: float,
    active_hydrogen_charge_e: float,
    inactive_hydrogen_charge_e: float,
    active_acceptor_charge_e: float,
    inactive_acceptor_charge_e: float,
    beta_per_angstrom: float,
) -> tuple[float, float]:
    """Return the conservative active-distance amplitude and charge RMSE."""

    decay = exp(
        -float(beta_per_angstrom)
        * (float(inactive_distance_angstrom) - float(active_distance_angstrom))
    )
    response = 1.0 - decay
    if abs(response) < 1.0e-8:
        raise ValueError("active and inactive charge-transfer distances are unresolved")
    hydrogen_delta = float(active_hydrogen_charge_e) - float(inactive_hydrogen_charge_e)
    acceptor_delta = float(active_acceptor_charge_e) - float(inactive_acceptor_charge_e)
    amplitude = (hydrogen_delta - acceptor_delta) / (2.0 * response)
    residuals = np.asarray(
        (amplitude * response - hydrogen_delta, -amplitude * response - acceptor_delta)
    )
    return float(amplitude), float(np.sqrt(np.mean(residuals**2)))


def _constraint_derivatives(dimension: int) -> tuple[tuple[int, ...], ...]:
    return (
        (),
        *((index,) for index in range(dimension)),
        *((left, right) for left in range(dimension) for right in range(left, dimension)),
    )


def _fourier_constraint_row(
    phases: np.ndarray,
    wavevectors: tuple[tuple[int, ...], ...],
    derivative: tuple[int, ...],
) -> np.ndarray:
    waves = np.asarray(wavevectors, dtype=float)
    arguments = waves @ phases
    count = len(waves)
    row = np.zeros(1 + 2 * count, dtype=float)
    if not derivative:
        row[0] = 1.0
        row[1 : 1 + count] = np.cos(arguments)
        row[1 + count :] = np.sin(arguments)
    elif len(derivative) == 1:
        factor = waves[:, derivative[0]]
        row[1 : 1 + count] = -factor * np.sin(arguments)
        row[1 + count :] = factor * np.cos(arguments)
    else:
        factor = waves[:, derivative[0]] * waves[:, derivative[1]]
        row[1 : 1 + count] = -factor * np.cos(arguments)
        row[1 + count :] = -factor * np.sin(arguments)
    return row
