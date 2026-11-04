"""Reactive bonded terms evaluated by the resident ZAFF runtime."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, erf, exp, expm1, isfinite, sin, tanh

import numpy as np

from matrix_chem.topology.descriptor_parameters import (
    ALPHA_LAMBDA,
    BO_LAMBDA_STRONG,
    BO_LAMBDA_WEAK,
)


@dataclass(frozen=True)
class ScalarDerivatives:
    """A scalar function and its first two derivatives."""

    value: float
    first: float
    second: float


@dataclass(frozen=True)
class InternalDerivatives:
    """Energy, gradient and Hessian in the term's internal coordinates."""

    energy: float
    gradient: np.ndarray
    hessian: np.ndarray


@dataclass(frozen=True)
class GaussianBondPotential:
    r"""GFN-FF Gaussian well for one covalent bond.

    The signed amplitude is retained because the native GFN-FF expression is
    referenced to separated atoms,

    ``V(r) = A exp[-a (r-r0)^2]``.

    Consequently a stable bond has ``A < 0`` and positive curvature
    ``-2 a A`` at ``r0``.  Keeping the absolute energy convention here, rather
    than silently shifting the minimum to zero, is required for term-by-term
    comparison with the xTB reference implementation.
    """

    amplitude_hartree: float
    exponent_per_angstrom2: float
    reference_distance_angstrom: float

    def __post_init__(self) -> None:
        if not isfinite(float(self.amplitude_hartree)):
            raise ValueError("Gaussian bond amplitude must be finite")
        if (
            not isfinite(float(self.exponent_per_angstrom2))
            or self.exponent_per_angstrom2 <= 0.0
        ):
            raise ValueError("Gaussian bond exponent must be finite and positive")
        if (
            not isfinite(float(self.reference_distance_angstrom))
            or self.reference_distance_angstrom <= 0.0
        ):
            raise ValueError("Gaussian bond reference distance must be finite and positive")

    def derivatives(self, distance_angstrom: float) -> ScalarDerivatives:
        distance = float(distance_angstrom)
        if not isfinite(distance) or distance <= 0.0:
            raise ValueError("Gaussian bond distance must be finite and positive")
        displacement = distance - self.reference_distance_angstrom
        exponent = self.exponent_per_angstrom2
        value = self.amplitude_hartree * exp(-exponent * displacement**2)
        first = -2.0 * exponent * displacement * value
        second = (4.0 * exponent**2 * displacement**2 - 2.0 * exponent) * value
        return ScalarDerivatives(value, first, second)

    def energy(self, distance_angstrom: float) -> float:
        return float(self.derivatives(distance_angstrom).value)


@dataclass(frozen=True)
class GFNFFDynamicGaussianBondModel:
    """Sparse non-periodic GFN-FF CN response for Gaussian bonds.

    ``coordination_pairs`` is an explicit cutoff/Verlet pair list.  No
    Cartesian CN Jacobian or Hessian is materialised: gradients and HVPs are
    accumulated by two pair sweeps, which keeps memory linear in system size
    for bounded-density molecular systems.
    """

    coordination_pairs: np.ndarray
    coordination_radii_angstrom: np.ndarray
    bonds: np.ndarray
    amplitudes_hartree: np.ndarray
    exponents_per_angstrom2: np.ndarray
    reference_bases_angstrom: np.ndarray
    reference_cn_coefficients_angstrom: np.ndarray
    cn_max: float = 4.4
    steepness: float = 7.5

    def __post_init__(self) -> None:
        pairs = np.asarray(self.coordination_pairs, dtype=np.intp).reshape(-1, 2)
        radii = np.asarray(self.coordination_radii_angstrom, dtype=float).reshape(-1)
        bonds = np.asarray(self.bonds, dtype=np.intp).reshape(-1, 2)
        amplitudes = np.asarray(self.amplitudes_hartree, dtype=float).reshape(-1)
        exponents = np.asarray(self.exponents_per_angstrom2, dtype=float).reshape(-1)
        bases = np.asarray(self.reference_bases_angstrom, dtype=float).reshape(-1)
        coefficients = np.asarray(
            self.reference_cn_coefficients_angstrom, dtype=float
        ).reshape(-1, 2)
        if len(radii) != len(pairs):
            raise ValueError("GFN-FF CN pair radii have inconsistent dimensions")
        if any(len(array) != len(bonds) for array in (amplitudes, exponents, bases, coefficients)):
            raise ValueError("GFN-FF dynamic bond arrays have inconsistent dimensions")
        if (
            np.any(pairs < 0)
            or np.any(pairs[:, 0] == pairs[:, 1])
            or np.any(bonds < 0)
            or np.any(bonds[:, 0] == bonds[:, 1])
        ):
            raise ValueError("GFN-FF dynamic pair indices are invalid")
        if (
            np.any(~np.isfinite(radii))
            or np.any(radii <= 0.0)
            or np.any(~np.isfinite(amplitudes))
            or np.any(~np.isfinite(exponents))
            or np.any(exponents <= 0.0)
            or np.any(~np.isfinite(bases))
            or np.any(bases <= 0.0)
            or np.any(~np.isfinite(coefficients))
        ):
            raise ValueError("GFN-FF dynamic bond parameters are invalid")
        if not isfinite(float(self.cn_max)) or self.cn_max <= 0.0:
            raise ValueError("GFN-FF maximum coordination must be positive")
        if not isfinite(float(self.steepness)) or self.steepness <= 0.0:
            raise ValueError("GFN-FF CN steepness must be positive")
        for name, value in (
            ("coordination_pairs", pairs),
            ("coordination_radii_angstrom", radii),
            ("bonds", bonds),
            ("amplitudes_hartree", amplitudes),
            ("exponents_per_angstrom2", exponents),
            ("reference_bases_angstrom", bases),
            ("reference_cn_coefficients_angstrom", coefficients),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def evaluate(
        self,
        coordinates_angstrom: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray | None, np.ndarray]:
        """Return energy, gradient, optional HVP, and saturated log-CN.

        The gradient and HVP use the same coordinate units as the input.  The
        runtime wrapper applies the Angstrom/bohr factors at its public API.
        """

        xyz = np.asarray(coordinates_angstrom, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or np.any(~np.isfinite(xyz)):
            raise ValueError("GFN-FF coordinates must have shape (N,3)")
        natoms = len(xyz)
        if (
            (len(self.coordination_pairs) and np.max(self.coordination_pairs) >= natoms)
            or (len(self.bonds) and np.max(self.bonds) >= natoms)
        ):
            raise ValueError("GFN-FF pair index exceeds the coordinate table")
        vector = None if direction is None else np.asarray(direction, dtype=float)
        if vector is not None and vector.shape != xyz.shape:
            raise ValueError("GFN-FF HVP direction must have shape (N,3)")

        pair_delta = xyz[self.coordination_pairs[:, 0]] - xyz[self.coordination_pairs[:, 1]]
        pair_distance = np.linalg.norm(pair_delta, axis=1)
        if np.any(pair_distance <= 1.0e-14):
            raise ValueError("coincident atoms in GFN-FF coordination pair")
        pair_unit = pair_delta / pair_distance[:, None]
        reduced = (pair_distance - self.coordination_radii_angstrom) / self.coordination_radii_angstrom
        gaussian = np.exp(-(self.steepness * reduced) ** 2)
        pair_value = 0.5 * (1.0 - np.asarray([erf(self.steepness * value) for value in reduced]))
        pair_first = -self.steepness * gaussian / (np.sqrt(np.pi) * self.coordination_radii_angstrom)
        pair_second = (
            2.0
            * self.steepness**3
            * (pair_distance - self.coordination_radii_angstrom)
            * gaussian
            / (np.sqrt(np.pi) * self.coordination_radii_angstrom**3)
        )
        raw = np.zeros(natoms)
        np.add.at(raw, self.coordination_pairs[:, 0], pair_value)
        np.add.at(raw, self.coordination_pairs[:, 1], pair_value)
        cn = np.logaddexp(0.0, self.cn_max) - np.logaddexp(0.0, self.cn_max - raw)
        hp = 1.0 / (1.0 + np.exp(raw - self.cn_max))
        hpp = -hp * (1.0 - hp)

        bond_delta = xyz[self.bonds[:, 0]] - xyz[self.bonds[:, 1]]
        bond_distance = np.linalg.norm(bond_delta, axis=1)
        if np.any(bond_distance <= 1.0e-14):
            raise ValueError("coincident atoms in GFN-FF Gaussian bond")
        bond_unit = bond_delta / bond_distance[:, None]
        reference = self.reference_bases_angstrom + np.sum(
            self.reference_cn_coefficients_angstrom
            * cn[self.bonds],
            axis=1,
        )
        displacement = bond_distance - reference
        value = self.amplitudes_hartree * np.exp(
            -self.exponents_per_angstrom2 * displacement**2
        )
        first = -2.0 * self.exponents_per_angstrom2 * displacement * value
        second = (
            4.0 * self.exponents_per_angstrom2**2 * displacement**2
            - 2.0 * self.exponents_per_angstrom2
        ) * value

        gradient = np.zeros_like(xyz)
        direct = first[:, None] * bond_unit
        np.add.at(gradient, self.bonds[:, 0], direct)
        np.add.at(gradient, self.bonds[:, 1], -direct)
        cn_adjoint = np.zeros(natoms)
        for side in (0, 1):
            np.add.at(
                cn_adjoint,
                self.bonds[:, side],
                -first * self.reference_cn_coefficients_angstrom[:, side],
            )
        pair_weight = pair_first * (
            cn_adjoint[self.coordination_pairs[:, 0]] * hp[self.coordination_pairs[:, 0]]
            + cn_adjoint[self.coordination_pairs[:, 1]] * hp[self.coordination_pairs[:, 1]]
        )
        cn_gradient = pair_weight[:, None] * pair_unit
        np.add.at(gradient, self.coordination_pairs[:, 0], cn_gradient)
        np.add.at(gradient, self.coordination_pairs[:, 1], -cn_gradient)

        if vector is None:
            return float(np.sum(value)), gradient, None, cn

        pair_vector = vector[self.coordination_pairs[:, 0]] - vector[self.coordination_pairs[:, 1]]
        pair_rdot = np.sum(pair_unit * pair_vector, axis=1)
        pair_udot = (
            pair_vector - pair_unit * pair_rdot[:, None]
        ) / pair_distance[:, None]
        raw_dot = np.zeros(natoms)
        pair_raw_dot = pair_first * pair_rdot
        np.add.at(raw_dot, self.coordination_pairs[:, 0], pair_raw_dot)
        np.add.at(raw_dot, self.coordination_pairs[:, 1], pair_raw_dot)
        cn_dot = hp * raw_dot

        bond_vector = vector[self.bonds[:, 0]] - vector[self.bonds[:, 1]]
        bond_rdot = np.sum(bond_unit * bond_vector, axis=1)
        bond_udot = (
            bond_vector - bond_unit * bond_rdot[:, None]
        ) / bond_distance[:, None]
        s_dot = bond_rdot - np.sum(
            self.reference_cn_coefficients_angstrom * cn_dot[self.bonds], axis=1
        )
        direct_dot = second[:, None] * s_dot[:, None] * bond_unit + first[:, None] * bond_udot
        product = np.zeros_like(xyz)
        np.add.at(product, self.bonds[:, 0], direct_dot)
        np.add.at(product, self.bonds[:, 1], -direct_dot)

        cn_adjoint_dot = np.zeros(natoms)
        for side in (0, 1):
            np.add.at(
                cn_adjoint_dot,
                self.bonds[:, side],
                -second * s_dot * self.reference_cn_coefficients_angstrom[:, side],
            )
        left = self.coordination_pairs[:, 0]
        right = self.coordination_pairs[:, 1]
        saturated_weight = cn_adjoint[left] * hp[left] + cn_adjoint[right] * hp[right]
        saturated_weight_dot = (
            cn_adjoint_dot[left] * hp[left]
            + cn_adjoint[left] * hpp[left] * raw_dot[left]
            + cn_adjoint_dot[right] * hp[right]
            + cn_adjoint[right] * hpp[right] * raw_dot[right]
        )
        pair_weight_dot = pair_second * pair_rdot * saturated_weight + pair_first * saturated_weight_dot
        cn_product = pair_weight_dot[:, None] * pair_unit + pair_weight[:, None] * pair_udot
        np.add.at(product, left, cn_product)
        np.add.at(product, right, -cn_product)
        return float(np.sum(value)), gradient, product, cn


@dataclass(frozen=True)
class BondOrderRadialFactor:
    """Reference-normalized radial form of the continuous ORACLE bond order.

    The coordination-dependent covalent-radius sum is frozen from the ORACLE
    state at the PL1 reference geometry.  The remaining radial function is the
    same smooth strong/weak Pauling blend used by ORACLE and has exact first
    and second derivatives.  Normalization makes the factor one at the
    reference distance and zero upon dissociation.
    """

    reference_distance: float
    covalent_radius_sum: float
    strong_decay: float = BO_LAMBDA_STRONG
    weak_decay: float = BO_LAMBDA_WEAK
    switch_sharpness: float = ALPHA_LAMBDA

    def __post_init__(self) -> None:
        values = (
            self.reference_distance,
            self.covalent_radius_sum,
            self.strong_decay,
            self.weak_decay,
            self.switch_sharpness,
        )
        if not all(isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("bond-order radial parameters must be finite and positive")

    def derivatives(self, distance: float) -> ScalarDerivatives:
        raw = _bond_order_radial_derivatives(
            distance,
            self.covalent_radius_sum,
            self.strong_decay,
            self.weak_decay,
            self.switch_sharpness,
        )
        reference = _bond_order_radial_derivatives(
            self.reference_distance,
            self.covalent_radius_sum,
            self.strong_decay,
            self.weak_decay,
            self.switch_sharpness,
        ).value
        return ScalarDerivatives(
            raw.value / reference,
            raw.first / reference,
            raw.second / reference,
        )

    def value(self, distance: float) -> float:
        """Evaluate only the normalized continuous bond-order factor."""

        raw = _bond_order_radial_derivatives(
            distance,
            self.covalent_radius_sum,
            self.strong_decay,
            self.weak_decay,
            self.switch_sharpness,
        ).value
        reference = _bond_order_radial_derivatives(
            self.reference_distance,
            self.covalent_radius_sum,
            self.strong_decay,
            self.weak_decay,
            self.switch_sharpness,
        ).value
        return float(raw / reference)


@dataclass(frozen=True)
class BondOrderDampedAnglePotential:
    """UFF-like angle damped by the two adjacent continuous bond orders."""

    force_constant: float
    theta0: float
    bond1: BondOrderRadialFactor
    bond2: BondOrderRadialFactor
    linear_sine_threshold: float = 1.0e-5

    def __post_init__(self) -> None:
        if not isfinite(self.force_constant) or self.force_constant < 0.0:
            raise ValueError("angular force constant must be finite and non-negative")
        if not isfinite(self.theta0):
            raise ValueError("reference angle must be finite")
        if not isfinite(self.linear_sine_threshold) or self.linear_sine_threshold <= 0.0:
            raise ValueError("linear-angle threshold must be finite and positive")

    @property
    def linear(self) -> bool:
        return abs(sin(self.theta0)) <= self.linear_sine_threshold

    @property
    def amplitude(self) -> float:
        if self.linear:
            return 0.5 * self.force_constant
        return self.force_constant / (2.0 * sin(self.theta0) ** 2)

    def derivatives(self, r1: float, r2: float, theta: float) -> InternalDerivatives:
        radial = (self.bond1.derivatives(r1), self.bond2.derivatives(r2))
        shape = self._angular_shape(theta)
        return _product_term_derivatives(radial, shape, self.amplitude)

    def energy(self, r1: float, r2: float, theta: float) -> float:
        """Evaluate the angle energy without internal or Cartesian derivatives."""

        if not isfinite(float(theta)):
            raise ValueError("angle must be finite")
        shape = (
            sin(theta) ** 2
            if self.linear
            else (cos(theta) - cos(self.theta0)) ** 2
        )
        return float(
            self.amplitude
            * self.bond1.value(r1)
            * self.bond2.value(r2)
            * shape
        )

    def _angular_shape(self, theta: float) -> ScalarDerivatives:
        if not isfinite(float(theta)):
            raise ValueError("angle must be finite")
        sine = sin(theta)
        cosine = cos(theta)
        if self.linear:
            return ScalarDerivatives(
                sine**2,
                2.0 * sine * cosine,
                2.0 * (cosine**2 - sine**2),
            )
        difference = cosine - cos(self.theta0)
        return ScalarDerivatives(
            difference**2,
            -2.0 * difference * sine,
            2.0 * (sine**2 - difference * cosine),
        )


@dataclass(frozen=True)
class CosineSeriesAngleShape:
    """Three-harmonic, globally smooth valence-angle potential.

    The energy is a sum of ``A_n [cos(n*theta)-cos(n*theta0)]``.  Three
    distinct harmonics are sufficient to impose stationarity and copy the QM
    quadratic and cubic derivatives exactly.  Because the result is a smooth
    function of ``cos(theta)``, its force vanishes at the linear boundary.
    Linear reference angles remain a separate two-component bending problem.
    """

    theta0: float
    harmonics: tuple[int, int, int]
    coefficients: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isfinite(float(self.theta0)) or not 0.0 < float(self.theta0) < np.pi:
            raise ValueError("cosine-series reference angle must lie between 0 and pi")
        harmonics = tuple(int(value) for value in self.harmonics)
        if (
            len(harmonics) != 3
            or len(set(harmonics)) != 3
            or any(value <= 0 for value in harmonics)
        ):
            raise ValueError("cosine-series angle requires three distinct positive harmonics")
        coefficients = tuple(float(value) for value in self.coefficients)
        if len(coefficients) != 3 or not all(isfinite(value) for value in coefficients):
            raise ValueError("cosine-series angle coefficients must be three finite values")
        object.__setattr__(self, "harmonics", harmonics)
        object.__setattr__(self, "coefficients", coefficients)

    @classmethod
    def from_local_derivatives(
        cls,
        theta0: float,
        curvature: float,
        cubic: float,
        *,
        harmonics: tuple[int, int, int] = (1, 2, 3),
        condition_limit: float = 1.0e10,
    ) -> "CosineSeriesAngleShape":
        reference = float(theta0)
        k2 = float(curvature)
        k3 = float(cubic)
        orders = tuple(int(value) for value in harmonics)
        if not isfinite(k2) or k2 <= 0.0 or not isfinite(k3):
            raise ValueError("cosine-series angle construction requires finite k2 > 0 and k3")
        matrix = np.asarray(
            (
                tuple(-order * sin(order * reference) for order in orders),
                tuple(-(order**2) * cos(order * reference) for order in orders),
                tuple(order**3 * sin(order * reference) for order in orders),
            ),
            dtype=float,
        )
        condition = float(np.linalg.cond(matrix))
        if not isfinite(condition) or condition > float(condition_limit):
            raise ValueError(
                "cosine-series angle derivative constraints are ill-conditioned; "
                "use the linear-bend representation or different harmonics"
            )
        coefficients = np.linalg.solve(matrix, np.asarray((0.0, k2, k3), dtype=float))
        return cls(reference, orders, tuple(float(value) for value in coefficients))

    def derivatives(self, theta: float) -> ScalarDerivatives:
        angle = float(theta)
        if not isfinite(angle):
            raise ValueError("angle must be finite")
        value = 0.0
        first = 0.0
        second = 0.0
        for order, coefficient in zip(self.harmonics, self.coefficients):
            value += coefficient * (cos(order * angle) - cos(order * self.theta0))
            first -= coefficient * order * sin(order * angle)
            second -= coefficient * order**2 * cos(order * angle)
        return ScalarDerivatives(value, first, second)


@dataclass(frozen=True)
class BondOrderDampedCosineSeriesAnglePotential:
    """QM-derivative-matched cosine angle switched by two bond orders."""

    bond1: BondOrderRadialFactor
    bond2: BondOrderRadialFactor
    shape: CosineSeriesAngleShape

    def derivatives(self, r1: float, r2: float, theta: float) -> InternalDerivatives:
        radial = (self.bond1.derivatives(r1), self.bond2.derivatives(r2))
        return _product_term_derivatives(radial, self.shape.derivatives(theta), 1.0)


@dataclass(frozen=True)
class CenteredMorseStretchCoordinate:
    """Centered radial coordinate with unit derivative at equilibrium.

    ``S(r) = [1-exp(-alpha*(r-r0))]/alpha`` tends continuously to the
    ordinary displacement ``r-r0`` when ``alpha`` is zero.  Centering makes
    products of two such coordinates unable to create a spurious reference
    energy or gradient, while unit normalization preserves the requested
    local mixed derivatives exactly.
    """

    reference_distance: float
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(float(self.reference_distance)) or self.reference_distance <= 0.0:
            raise ValueError("centered stretch reference distance must be positive")
        if not isfinite(float(self.alpha)) or self.alpha < 0.0:
            raise ValueError("centered stretch alpha must be finite and non-negative")

    def derivatives(self, distance: float) -> ScalarDerivatives:
        r = float(distance)
        if not isfinite(r) or r <= 0.0:
            raise ValueError("centered stretch distance must be finite and positive")
        displacement = r - self.reference_distance
        if self.alpha == 0.0:
            return ScalarDerivatives(displacement, 1.0, 0.0)
        exponential = exp(-self.alpha * displacement)
        return ScalarDerivatives(
            -expm1(-self.alpha * displacement) / self.alpha,
            exponential,
            -self.alpha * exponential,
        )


@dataclass(frozen=True)
class ZaffStretchStretchAnglePotential:
    r"""Physical ``S_i S_j [K+A Delta cos(theta)+B Delta cos^2(theta)]`` term.

    At the reference geometry both centered stretches vanish and have unit
    first derivative.  Consequently ``K`` is the harmonic stretch--stretch
    coupling, while ``A`` and ``B`` copy independently the local
    ``Phi_ijtheta`` and ``K_ijthetatheta`` derivatives without changing the
    reference energy, gradient, or pure angular curvature.
    """

    stretch1: CenteredMorseStretchCoordinate
    stretch2: CenteredMorseStretchCoordinate
    theta0: float
    harmonic_stretch_coupling: float
    cosine_coefficient: float
    cosine_squared_coefficient: float
    linear_sine_threshold: float = 1.0e-5

    def __post_init__(self) -> None:
        values = (
            self.theta0,
            self.harmonic_stretch_coupling,
            self.cosine_coefficient,
            self.cosine_squared_coefficient,
            self.linear_sine_threshold,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("stretch--stretch--angle parameters must be finite")
        if not 0.0 < float(self.theta0) < np.pi:
            raise ValueError("stretch--stretch--angle reference must lie between 0 and pi")
        if self.linear_sine_threshold <= 0.0:
            raise ValueError("linear-angle threshold must be positive")
        if abs(sin(self.theta0)) <= self.linear_sine_threshold:
            raise ValueError("a linear angle requires the two-component bending representation")

    @classmethod
    def from_local_derivatives(
        cls,
        stretch1: CenteredMorseStretchCoordinate,
        stretch2: CenteredMorseStretchCoordinate,
        theta0: float,
        harmonic_stretch_coupling: float,
        cubic_stretch_stretch_angle: float,
        quartic_stretch_stretch_angle_angle: float,
        *,
        linear_sine_threshold: float = 1.0e-5,
    ) -> "ZaffStretchStretchAnglePotential":
        """Determine ``A`` and ``B`` from the two requested mixed derivatives."""

        angle = float(theta0)
        sine = sin(angle)
        cosine = cos(angle)
        if abs(sine) <= float(linear_sine_threshold):
            raise ValueError(
                "stretch--stretch coupling to a linear bend is not a scalar-angle term"
            )
        first_target = float(cubic_stretch_stretch_angle)
        second_target = float(quartic_stretch_stretch_angle_angle)
        common = -first_target / sine
        coefficient_b = (second_target + cosine * common) / (2.0 * sine**2)
        coefficient_a = common - 2.0 * coefficient_b * cosine
        return cls(
            stretch1=stretch1,
            stretch2=stretch2,
            theta0=angle,
            harmonic_stretch_coupling=float(harmonic_stretch_coupling),
            cosine_coefficient=float(coefficient_a),
            cosine_squared_coefficient=float(coefficient_b),
            linear_sine_threshold=float(linear_sine_threshold),
        )

    @property
    def cubic_stretch_stretch_angle(self) -> float:
        return self._angular_derivatives(self.theta0).first

    @property
    def quartic_stretch_stretch_angle_angle(self) -> float:
        return self._angular_derivatives(self.theta0).second

    def derivatives(self, r1: float, r2: float, theta: float) -> InternalDerivatives:
        radial = (self.stretch1.derivatives(r1), self.stretch2.derivatives(r2))
        return _product_term_derivatives(radial, self._angular_derivatives(theta), 1.0)

    def anharmonic_correction_derivatives(
        self,
        r1: float,
        r2: float,
        theta: float,
    ) -> InternalDerivatives:
        """Evaluate only the part not already stored in the base quadratic matrix."""

        radial = (self.stretch1.derivatives(r1), self.stretch2.derivatives(r2))
        angular = self._angular_derivatives(theta)
        correction = ScalarDerivatives(
            angular.value - self.harmonic_stretch_coupling,
            angular.first,
            angular.second,
        )
        return _product_term_derivatives(radial, correction, 1.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_form": "CENTERED_MORSE_SI_SJ_COSINE_ANGLE",
            "stretch1": {
                "reference_distance_angstrom": self.stretch1.reference_distance,
                "alpha_per_angstrom": self.stretch1.alpha,
            },
            "stretch2": {
                "reference_distance_angstrom": self.stretch2.reference_distance,
                "alpha_per_angstrom": self.stretch2.alpha,
            },
            "theta0_radian": self.theta0,
            "harmonic_stretch_coupling": self.harmonic_stretch_coupling,
            "cosine_coefficient": self.cosine_coefficient,
            "cosine_squared_coefficient": self.cosine_squared_coefficient,
            "copied_derivatives": {
                "Phi_ijtheta": self.cubic_stretch_stretch_angle,
                "K_ijthetatheta": self.quartic_stretch_stretch_angle_angle,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ZaffStretchStretchAnglePotential":
        if payload.get("functional_form") != "CENTERED_MORSE_SI_SJ_COSINE_ANGLE":
            raise ValueError("unsupported stretch--stretch--angle functional form")
        first = dict(payload["stretch1"])
        second = dict(payload["stretch2"])
        return cls(
            stretch1=CenteredMorseStretchCoordinate(
                float(first["reference_distance_angstrom"]),
                float(first["alpha_per_angstrom"]),
            ),
            stretch2=CenteredMorseStretchCoordinate(
                float(second["reference_distance_angstrom"]),
                float(second["alpha_per_angstrom"]),
            ),
            theta0=float(payload["theta0_radian"]),
            harmonic_stretch_coupling=float(payload["harmonic_stretch_coupling"]),
            cosine_coefficient=float(payload["cosine_coefficient"]),
            cosine_squared_coefficient=float(payload["cosine_squared_coefficient"]),
        )

    def _angular_derivatives(self, theta: float) -> ScalarDerivatives:
        angle = float(theta)
        if not isfinite(angle):
            raise ValueError("stretch--stretch--angle value must be finite")
        cosine = cos(angle)
        sine = sin(angle)
        cosine0 = cos(self.theta0)
        coefficient_a = self.cosine_coefficient
        coefficient_b = self.cosine_squared_coefficient
        common = coefficient_a + 2.0 * coefficient_b * cosine
        return ScalarDerivatives(
            self.harmonic_stretch_coupling
            + coefficient_a * (cosine - cosine0)
            + coefficient_b * (cosine**2 - cosine0**2),
            -sine * common,
            -cosine * common + 2.0 * coefficient_b * sine**2,
        )


@dataclass(frozen=True)
class ZaffTeamPlusValenceTorsionPotential:
    r"""TEAM+ ``(x-x0) sum_n K_n cos(n phi)`` bond/angle--torsion coupling."""

    reference_valence: float
    reference_torsion: float
    coefficients: tuple[float, float, float]
    valence_kind: str
    harmonics: tuple[int, int, int] = (1, 2, 3)

    def __post_init__(self) -> None:
        if not all(
            isfinite(float(value))
            for value in (self.reference_valence, self.reference_torsion, *self.coefficients)
        ):
            raise ValueError("TEAM+ valence--torsion parameters must be finite")
        harmonics = tuple(int(value) for value in self.harmonics)
        if (
            len(harmonics) != 3
            or len(set(harmonics)) != 3
            or any(value <= 0 for value in harmonics)
        ):
            raise ValueError("TEAM+ valence--torsion coupling requires three harmonics")
        kind = str(self.valence_kind).upper()
        if kind not in {"BOND", "ANGLE"}:
            raise ValueError("TEAM+ valence coordinate must be BOND or ANGLE")
        object.__setattr__(self, "harmonics", harmonics)
        object.__setattr__(self, "valence_kind", kind)
        object.__setattr__(self, "coefficients", tuple(float(value) for value in self.coefficients))

    @classmethod
    def from_local_derivatives(
        cls,
        reference_valence: float,
        reference_torsion: float,
        mixed_quadratic: float,
        mixed_cubic_valence_torsion_torsion: float,
        mixed_quartic_valence_torsion_torsion_torsion: float,
        *,
        valence_kind: str,
        harmonics: tuple[int, int, int] = (1, 2, 3),
        condition_limit: float = 1.0e10,
    ) -> "ZaffTeamPlusValenceTorsionPotential":
        """Determine the three Fourier coefficients from local mixed derivatives."""

        phi0 = float(reference_torsion)
        orders = tuple(int(value) for value in harmonics)
        matrix = np.asarray(
            (
                tuple(-order * sin(order * phi0) for order in orders),
                tuple(-(order**2) * cos(order * phi0) for order in orders),
                tuple(order**3 * sin(order * phi0) for order in orders),
            ),
            dtype=float,
        )
        condition = float(np.linalg.cond(matrix))
        if not isfinite(condition) or condition > float(condition_limit):
            raise ValueError(
                "TEAM+ derivative constraints are ill-conditioned at the reference torsion"
            )
        target = np.asarray(
            (
                mixed_quadratic,
                mixed_cubic_valence_torsion_torsion,
                mixed_quartic_valence_torsion_torsion_torsion,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(target)):
            raise ValueError("TEAM+ local mixed derivatives must be finite")
        coefficients = np.linalg.solve(matrix, target)
        return cls(
            reference_valence=float(reference_valence),
            reference_torsion=phi0,
            coefficients=tuple(float(value) for value in coefficients),
            valence_kind=valence_kind,
            harmonics=orders,
        )

    def derivatives(self, valence: float, torsion: float) -> InternalDerivatives:
        displacement = float(valence) - self.reference_valence
        shape = self._torsion_derivatives(torsion)
        return InternalDerivatives(
            energy=displacement * shape.value,
            gradient=np.asarray((shape.value, displacement * shape.first), dtype=float),
            hessian=np.asarray(
                ((0.0, shape.first), (shape.first, displacement * shape.second)),
                dtype=float,
            ),
        )

    def anharmonic_correction_derivatives(
        self,
        valence: float,
        torsion: float,
    ) -> InternalDerivatives:
        """Return TEAM+ beyond its reference energy, gradient and quadratic coupling."""

        displacement = float(valence) - self.reference_valence
        torsion_displacement = _periodic_displacement(torsion, self.reference_torsion)
        shape = self._torsion_derivatives(torsion)
        reference = self._torsion_derivatives(self.reference_torsion)
        return InternalDerivatives(
            energy=displacement
            * (shape.value - reference.value - reference.first * torsion_displacement),
            gradient=np.asarray(
                (
                    shape.value - reference.value - reference.first * torsion_displacement,
                    displacement * (shape.first - reference.first),
                ),
                dtype=float,
            ),
            hessian=np.asarray(
                (
                    (0.0, shape.first - reference.first),
                    (shape.first - reference.first, displacement * shape.second),
                ),
                dtype=float,
            ),
        )

    @property
    def copied_mixed_derivatives(self) -> tuple[float, float, float]:
        reference = self._torsion_derivatives(self.reference_torsion)
        third = sum(
            coefficient * order**3 * sin(order * self.reference_torsion)
            for order, coefficient in zip(self.harmonics, self.coefficients, strict=True)
        )
        return reference.first, reference.second, float(third)

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_form": "TEAM_PLUS_VALENCE_TORSION_COSINE",
            "valence_kind": self.valence_kind,
            "reference_valence": self.reference_valence,
            "reference_torsion_radian": self.reference_torsion,
            "harmonics": list(self.harmonics),
            "coefficients_hartree": list(self.coefficients),
            "copied_mixed_derivatives": list(self.copied_mixed_derivatives),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ZaffTeamPlusValenceTorsionPotential":
        if payload.get("functional_form") != "TEAM_PLUS_VALENCE_TORSION_COSINE":
            raise ValueError("unsupported TEAM+ valence--torsion functional form")
        return cls(
            reference_valence=float(payload["reference_valence"]),
            reference_torsion=float(payload["reference_torsion_radian"]),
            coefficients=tuple(float(value) for value in payload["coefficients_hartree"]),
            valence_kind=str(payload["valence_kind"]),
            harmonics=tuple(int(value) for value in payload.get("harmonics", (1, 2, 3))),
        )

    def _torsion_derivatives(self, torsion: float) -> ScalarDerivatives:
        angle = float(torsion)
        if not isfinite(angle):
            raise ValueError("TEAM+ torsion value must be finite")
        value = 0.0
        first = 0.0
        second = 0.0
        for order, coefficient in zip(self.harmonics, self.coefficients, strict=True):
            value += coefficient * cos(order * angle)
            first -= coefficient * order * sin(order * angle)
            second -= coefficient * order**2 * cos(order * angle)
        return ScalarDerivatives(value, first, second)


@dataclass(frozen=True)
class EvenOutOfPlanePolynomial:
    """Reflection-symmetric out-of-plane energy in ``s = sin(chi)``.

    The polynomial ``c0 + c2*s^2 + c4*s^4 + c6*s^6`` represents both a
    stable planar center and a symmetric pyramidal double well without using
    an improper dihedral.  Odd powers are deliberately absent: when ORACLE
    identifies the two faces as equivalent, every odd derivative at the
    planar structure is symmetry-forbidden.
    """

    c0: float = 0.0
    c2: float = 0.0
    c4: float = 0.0
    c6: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(float(value)) for value in (self.c0, self.c2, self.c4, self.c6)):
            raise ValueError("out-of-plane polynomial coefficients must be finite")

    @classmethod
    def planar_single_well(
        cls,
        curvature: float,
        quartic_derivative: float,
        *,
        c6: float = 0.0,
    ) -> "EvenOutOfPlanePolynomial":
        """Construct a planar well matching derivatives through fourth order.

        Derivatives are with respect to the signed angle ``chi`` at zero.  The
        ``sin(chi)`` chain rule gives ``c2=k2/2`` and
        ``c4=k4/24+k2/6``.
        """

        k2 = float(curvature)
        k4 = float(quartic_derivative)
        if not isfinite(k2) or k2 <= 0.0 or not isfinite(k4):
            raise ValueError("a planar out-of-plane well requires finite k2 > 0 and finite k4")
        return cls(c0=0.0, c2=0.5 * k2, c4=k4 / 24.0 + k2 / 6.0, c6=float(c6))

    @classmethod
    def symmetric_double_well(
        cls,
        minimum_angle: float,
        minimum_curvature: float,
        planar_barrier: float,
    ) -> "EvenOutOfPlanePolynomial":
        """Construct the minimal sextic double well from physical observables.

        The four coefficients are fixed, not fitted, by the energy at the
        planar structure, zero energy and zero gradient at ``+/-minimum_angle``,
        and the curvature at either minimum.
        """

        chi0 = float(minimum_angle)
        k2 = float(minimum_curvature)
        barrier = float(planar_barrier)
        if not (isfinite(chi0) and 0.0 < abs(chi0) < 0.5 * np.pi):
            raise ValueError("double-well minimum angle must lie strictly between 0 and pi/2")
        if not isfinite(k2) or k2 <= 0.0:
            raise ValueError("double-well minimum curvature must be positive")
        if not isfinite(barrier) or barrier <= 0.0:
            raise ValueError("double-well planar barrier must be positive")
        s0 = sin(abs(chi0))
        cosine0 = cos(abs(chi0))
        target_s_curvature = k2 / cosine0**2
        matrix = np.asarray(
            (
                (s0**2, s0**4, s0**6),
                (2.0 * s0, 4.0 * s0**3, 6.0 * s0**5),
                (2.0, 12.0 * s0**2, 30.0 * s0**4),
            ),
            dtype=float,
        )
        target = np.asarray((-barrier, 0.0, target_s_curvature), dtype=float)
        c2, c4, c6 = np.linalg.solve(matrix, target)
        return cls(c0=barrier, c2=float(c2), c4=float(c4), c6=float(c6))

    def derivatives(self, chi: float) -> ScalarDerivatives:
        angle = float(chi)
        if not isfinite(angle):
            raise ValueError("out-of-plane angle must be finite")
        s = sin(angle)
        c = cos(angle)
        polynomial = self.c0 + self.c2 * s**2 + self.c4 * s**4 + self.c6 * s**6
        derivative_s = 2.0 * self.c2 * s + 4.0 * self.c4 * s**3 + 6.0 * self.c6 * s**5
        second_s = 2.0 * self.c2 + 12.0 * self.c4 * s**2 + 30.0 * self.c6 * s**4
        return ScalarDerivatives(
            polynomial,
            derivative_s * c,
            second_s * c**2 - derivative_s * s,
        )


@dataclass(frozen=True)
class SinOutOfPlanePolynomial:
    r"""General polynomial in ``s = sin(chi)``.

    The coefficient sequence is ordered by increasing power, so that
    ``coefficients[n]`` multiplies ``sin(chi)**n``.  Unlike
    :class:`EvenOutOfPlanePolynomial`, odd powers are retained.  This is the
    native representation needed when a non-planar reference geometry or a
    fitted shifted well is expanded in ``sin(chi)``.
    """

    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("out-of-plane polynomial requires at least one coefficient")
        if not all(isfinite(float(value)) for value in self.coefficients):
            raise ValueError("out-of-plane polynomial coefficients must be finite")

    def derivatives(self, chi: float) -> ScalarDerivatives:
        angle = float(chi)
        if not isfinite(angle):
            raise ValueError("out-of-plane angle must be finite")
        s = sin(angle)
        c = cos(angle)
        value = 0.0
        first_s = 0.0
        second_s = 0.0
        for power, coefficient in enumerate(self.coefficients):
            value += coefficient * s**power
            if power >= 1:
                first_s += power * coefficient * s ** (power - 1)
            if power >= 2:
                second_s += power * (power - 1) * coefficient * s ** (power - 2)
        return ScalarDerivatives(
            value,
            first_s * c,
            second_s * c**2 - first_s * s,
        )


@dataclass(frozen=True)
class BondOrderDampedOutOfPlanePotential:
    """Native out-of-plane potential switched by its three defining bonds."""

    bond1: BondOrderRadialFactor
    bond2: BondOrderRadialFactor
    bond3: BondOrderRadialFactor
    shape: EvenOutOfPlanePolynomial | SinOutOfPlanePolynomial

    def derivatives(
        self,
        r1: float,
        r2: float,
        r3: float,
        chi: float,
    ) -> InternalDerivatives:
        radial = (
            self.bond1.derivatives(r1),
            self.bond2.derivatives(r2),
            self.bond3.derivatives(r3),
        )
        return _product_term_derivatives(radial, self.shape.derivatives(chi), 1.0)

    def energy(self, r1: float, r2: float, r3: float, chi: float) -> float:
        return float(
            self.bond1.value(r1)
            * self.bond2.value(r2)
            * self.bond3.value(r3)
            * self.shape.derivatives(chi).value
        )


@dataclass(frozen=True)
class UFFFourierTerm:
    """One UFF-compatible torsional Fourier contribution."""

    amplitude: float
    periodicity: int
    phase: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.amplitude):
            raise ValueError("Fourier amplitude must be finite")
        if int(self.periodicity) <= 0:
            raise ValueError("Fourier periodicity must be positive")
        if not isfinite(self.phase):
            raise ValueError("Fourier phase must be finite")


@dataclass(frozen=True)
class BondOrderDampedTorsionPotential:
    """UFF/Fourier torsion multiplied by all three ORACLE bond orders.

    Each radial factor is normalized to one at the PL1 geometry.  Consequently
    the local UFF/Fourier prior is unchanged at the reference, while cleavage
    of either terminal bond or the central bond continuously removes the whole
    torsional term.
    """

    bond1: BondOrderRadialFactor
    bond2: BondOrderRadialFactor
    bond3: BondOrderRadialFactor
    terms: tuple[UFFFourierTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("bond-order-damped torsion requires at least one Fourier term")

    def derivatives(
        self,
        r1: float,
        r2: float,
        r3: float,
        phi: float,
    ) -> InternalDerivatives:
        radial = (
            self.bond1.derivatives(r1),
            self.bond2.derivatives(r2),
            self.bond3.derivatives(r3),
        )
        return _product_term_derivatives(radial, self._fourier(phi), 1.0)

    def energy(self, r1: float, r2: float, r3: float, phi: float) -> float:
        """Evaluate the torsional energy without internal derivatives."""

        if not isfinite(float(phi)):
            raise ValueError("torsional angle must be finite")
        fourier = sum(
            term.amplitude
            * (1.0 + cos(int(term.periodicity) * phi - term.phase))
            for term in self.terms
        )
        return float(
            self.bond1.value(r1)
            * self.bond2.value(r2)
            * self.bond3.value(r3)
            * fourier
        )

    def _fourier(self, phi: float) -> ScalarDerivatives:
        if not isfinite(float(phi)):
            raise ValueError("torsional angle must be finite")
        value = 0.0
        first = 0.0
        second = 0.0
        for term in self.terms:
            order = int(term.periodicity)
            argument = order * phi - term.phase
            value += term.amplitude * (1.0 + cos(argument))
            first -= term.amplitude * order * sin(argument)
            second -= term.amplitude * order**2 * cos(argument)
        return ScalarDerivatives(value, first, second)


def _bond_order_radial_derivatives(
    distance: float,
    radius_sum: float,
    strong_decay: float,
    weak_decay: float,
    sharpness: float,
) -> ScalarDerivatives:
    r = float(distance)
    if not isfinite(r) or r <= 0.0:
        raise ValueError("bond-order distance must be finite and positive")
    x = (r - radius_sum) / radius_sum
    hyperbolic = tanh(sharpness * x)
    sech_squared = 1.0 - hyperbolic**2
    weight = 0.5 * (1.0 - hyperbolic)
    weight_first = -0.5 * sharpness * sech_squared / radius_sum
    weight_second = sharpness**2 * sech_squared * hyperbolic / radius_sum**2
    strong = exp((radius_sum - r) / strong_decay)
    weak = exp((radius_sum - r) / weak_decay)
    strong_first = -strong / strong_decay
    weak_first = -weak / weak_decay
    strong_second = strong / strong_decay**2
    weak_second = weak / weak_decay**2
    difference = strong - weak
    difference_first = strong_first - weak_first
    difference_second = strong_second - weak_second
    return ScalarDerivatives(
        weak + weight * difference,
        weak_first + weight_first * difference + weight * difference_first,
        weak_second
        + weight_second * difference
        + 2.0 * weight_first * difference_first
        + weight * difference_second,
    )


def _product_term_derivatives(
    radial: tuple[ScalarDerivatives, ...],
    angular: ScalarDerivatives,
    amplitude: float,
) -> InternalDerivatives:
    values = tuple(item.value for item in radial)
    radial_product = float(np.prod(values))
    energy = float(amplitude) * radial_product * angular.value
    size = len(radial) + 1
    gradient = np.zeros(size, dtype=float)
    hessian = np.zeros((size, size), dtype=float)
    for left, item in enumerate(radial):
        other_product = float(np.prod(values[:left] + values[left + 1 :]))
        gradient[left] = amplitude * item.first * other_product * angular.value
        hessian[left, left] = amplitude * item.second * other_product * angular.value
        hessian[left, -1] = hessian[-1, left] = (
            amplitude * item.first * other_product * angular.first
        )
        for right in range(left + 1, len(radial)):
            remaining = tuple(
                value for index, value in enumerate(values) if index not in {left, right}
            )
            cross = (
                amplitude
                * item.first
                * radial[right].first
                * float(np.prod(remaining))
                * angular.value
            )
            hessian[left, right] = hessian[right, left] = cross
    gradient[-1] = amplitude * radial_product * angular.first
    hessian[-1, -1] = amplitude * radial_product * angular.second
    return InternalDerivatives(energy, gradient, hessian)


def _periodic_displacement(value: float, reference: float) -> float:
    return (float(value) - float(reference) + np.pi) % (2.0 * np.pi) - np.pi
