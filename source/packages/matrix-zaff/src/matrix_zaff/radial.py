"""Analytic radial potential forms used by the resident ZAFF runtime.

The conversion contract preserves the well depth ``epsilon`` and minimum
position ``r_min``.  Exp-PE additionally preserves the source curvature,
whereas the double-exponential form can preserve both the quadratic and cubic
derivatives at the minimum.  Energy and distance units are deliberately
generic, provided they are used consistently for the derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt



@dataclass(frozen=True)
class RadialDerivatives:
    """Energy and first three radial derivatives at one distance."""

    energy: float
    first: float
    second: float
    third: float


@dataclass(frozen=True)
class MinimumDerivatives:
    """Observable description of a radial well at its minimum."""

    epsilon: float
    r_min: float
    second: float
    third: float | None = None

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.second) or self.second <= 0.0:
            raise ValueError("radial curvature at the minimum must be positive")
        if self.third is not None and not isfinite(self.third):
            raise ValueError("radial cubic derivative must be finite")

    @property
    def dimensionless_second(self) -> float:
        return self.second * self.r_min**2 / self.epsilon

    @property
    def dimensionless_third(self) -> float | None:
        if self.third is None:
            return None
        return self.third * self.r_min**3 / self.epsilon


@dataclass(frozen=True)
class ContactCapturePotential:
    """Finite-domain exp/4/6 contact well with an independent COM dipole tail.

    A rigid radial translation gives d(R)^2=(R+s)^2+t. The attractive
    contact terms are an effective finite-distance fit, not separately
    identified multipole constants. No short-range extrapolation is allowed.
    """

    asymptote: float
    c3: float
    contact_parallel_offset: float
    contact_perpendicular_squared: float
    reference_contact: float
    minimum_radius: float
    exponent: float
    repulsion: float
    attraction4: float
    attraction6: float

    def __post_init__(self):
        if (not all(isfinite(v) for v in self.__dict__.values())
                or self.contact_perpendicular_squared < 0
                or min(self.reference_contact, self.minimum_radius, self.exponent) <= 0
                or min(self.repulsion, self.attraction4, self.attraction6) < 0):
            raise ValueError('finite geometry, positive scales and nonnegative amplitudes required')

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        if r < self.minimum_radius:
            raise ValueError('contact capture potential cannot extrapolate below its acquired domain')
        z = r + self.contact_parallel_offset
        t = self.contact_perpendicular_squared
        d = sqrt(z*z+t)
        if d <= 0:
            raise ValueError('positive contact distance required')
        f = [self.repulsion*_exp(-self.exponent*(d-self.reference_contact))
             *(-self.exponent)**k for k in range(4)]
        for n, amplitude in ((4, self.attraction4), (6, self.attraction6)):
            term = -amplitude*(self.reference_contact/d)**n
            for k in range(4):
                if k:
                    term *= -(n+k-1)/d
                f[k] += term
        d1, d2, d3 = z/d, t/d**3, -3*t*z/d**5
        return RadialDerivatives(
            self.asymptote+self.c3/r**3+f[0],
            -3*self.c3/r**4+f[1]*d1,
            12*self.c3/r**5+f[2]*d1*d1+f[1]*d2,
            -60*self.c3/r**6+f[3]*d1**3+3*f[2]*d1*d2+f[1]*d3,
        )

    def to_dict(self):
        return dict(schema='matrix.zaff.contact_capture.v1', **self.__dict__)

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != 'matrix.zaff.contact_capture.v1':
            raise ValueError('unsupported contact capture potential')
        return cls(**{name: float(value[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ExpPEPotential:
    """Undamped Yang--Sun--Deng Exp-PE modified-Morse pair potential."""

    epsilon: float
    r_min: float
    alpha: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.alpha) or self.alpha <= 4.0:
            raise ValueError("Exp-PE alpha must exceed 4 for a stable minimum")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        x = r / self.r_min
        alpha = self.alpha
        exp_full = _exp(alpha * (1.0 - x))
        exp_half = _exp(0.5 * alpha * (1.0 - x))
        polynomial = x**4 - 2.0 * x**2 + 3.0
        polynomial_1 = 4.0 * x**3 - 4.0 * x
        polynomial_2 = 12.0 * x**2 - 4.0
        polynomial_3 = 24.0 * x
        dimensionless_0 = exp_full - polynomial * exp_half
        dimensionless_1 = -alpha * exp_full - exp_half * (polynomial_1 - 0.5 * alpha * polynomial)
        dimensionless_2 = alpha**2 * exp_full + exp_half * (
            -polynomial_2 + alpha * polynomial_1 - 0.25 * alpha**2 * polynomial
        )
        dimensionless_3 = -(alpha**3) * exp_full + exp_half * (
            -polynomial_3
            + 1.5 * alpha * polynomial_2
            - 0.75 * alpha**2 * polynomial_1
            + 0.125 * alpha**3 * polynomial
        )
        return _scale_derivatives(
            self.epsilon,
            self.r_min,
            dimensionless_0,
            dimensionless_1,
            dimensionless_2,
            dimensionless_3,
        )

    def energy(self, distance: float) -> float:
        """Evaluate the radial energy without derivative construction."""

        r = _validate_distance(distance)
        x = r / self.r_min
        exp_full = _exp(self.alpha * (1.0 - x))
        exp_half = _exp(0.5 * self.alpha * (1.0 - x))
        polynomial = x**4 - 2.0 * x**2 + 3.0
        return float(self.epsilon * (exp_full - polynomial * exp_half))

    @property
    def minimum(self) -> MinimumDerivatives:
        values = self.derivatives(self.r_min)
        return MinimumDerivatives(self.epsilon, self.r_min, values.second, values.third)


@dataclass(frozen=True)
class DampedExpPEPotential:
    """Legacy name for the undamped resident Exp-PE radial potential."""

    epsilon: float
    r_min: float
    alpha: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.alpha) or self.alpha <= 4.0:
            raise ValueError("Exp-PE alpha must exceed 4")

    def energy(self, distance: float) -> float:
        r = _validate_distance(distance)
        x = r / self.r_min
        exp_full = _exp(self.alpha * (1.0 - x))
        exp_half = _exp(0.5 * self.alpha * (1.0 - x))
        polynomial = x**4 - 2.0 * x**2 + 3.0
        return float(self.epsilon * (exp_full - polynomial * exp_half))


@dataclass(frozen=True)
class MorsePotential:
    """Shifted standard Morse pair potential with minimum energy ``-epsilon``."""

    epsilon: float
    r_min: float
    beta: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("Morse beta must be positive")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        displacement = r / self.r_min - 1.0
        first_exp = _exp(-self.beta * displacement)
        second_exp = first_exp * first_exp
        b = self.beta
        return _scale_derivatives(
            self.epsilon,
            self.r_min,
            second_exp - 2.0 * first_exp,
            -2.0 * b * second_exp + 2.0 * b * first_exp,
            4.0 * b**2 * second_exp - 2.0 * b**2 * first_exp,
            -8.0 * b**3 * second_exp + 2.0 * b**3 * first_exp,
        )


@dataclass(frozen=True)
class InversePowerStretchPotential:
    """Generalized Kratzer/Thakkar bond potential.

    The energy is ``depth * (1 - (r0 / r)**exponent)**2``.  It is zero at
    ``r0``, approaches ``depth`` upon dissociation and has a repulsive
    inverse-power wall for a positive exponent.  Unlike the simple Kratzer
    special case (``exponent=1``), the exponent can be obtained directly from
    the quadratic and cubic QM derivatives at the reference distance.
    """

    depth: float
    r0: float
    exponent: float

    def __post_init__(self) -> None:
        _validate_well(self.depth, self.r0)
        if not isfinite(self.exponent) or self.exponent <= 0.0:
            raise ValueError("inverse-power stretching exponent must be positive")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        p = float(self.exponent)
        inverse_power = (float(self.r0) / r) ** p
        shape = 1.0 - inverse_power
        shape_1 = p * inverse_power / r
        shape_2 = -p * (p + 1.0) * inverse_power / r**2
        shape_3 = p * (p + 1.0) * (p + 2.0) * inverse_power / r**3
        depth = float(self.depth)
        return RadialDerivatives(
            depth * shape**2,
            2.0 * depth * shape * shape_1,
            2.0 * depth * (shape_1**2 + shape * shape_2),
            2.0 * depth * (3.0 * shape_1 * shape_2 + shape * shape_3),
        )


@dataclass(frozen=True)
class SPFStretchPotential:
    """Simons--Parr--Finlan expansion in ``y = 1 - r0/r``.

    ``coefficients[n-2]`` multiplies ``y**n``.  Starting the expansion at
    second order keeps the supplied reference distance stationary.  A cubic
    truncation is a local model; production use must add a stabilizing even
    term or pass the global-shape audit performed by the ARCHITECT compiler.
    """

    r0: float
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isfinite(float(self.r0)) or float(self.r0) <= 0.0:
            raise ValueError("SPF reference distance must be positive")
        coefficients = tuple(float(value) for value in self.coefficients)
        if not coefficients or not all(isfinite(value) for value in coefficients):
            raise ValueError("SPF coefficients must be a nonempty finite sequence")
        object.__setattr__(self, "coefficients", coefficients)

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        r0 = float(self.r0)
        y = 1.0 - r0 / r
        y_1 = r0 / r**2
        y_2 = -2.0 * r0 / r**3
        y_3 = 6.0 * r0 / r**4
        polynomial = [0.0, 0.0, 0.0, 0.0]
        for power, coefficient in enumerate(self.coefficients, start=2):
            polynomial[0] += coefficient * y**power
            polynomial[1] += coefficient * power * y ** (power - 1)
            if power >= 2:
                polynomial[2] += coefficient * power * (power - 1) * y ** (power - 2)
            if power >= 3:
                polynomial[3] += coefficient * power * (power - 1) * (power - 2) * y ** (power - 3)
        return RadialDerivatives(
            polynomial[0],
            polynomial[1] * y_1,
            polynomial[2] * y_1**2 + polynomial[1] * y_2,
            (polynomial[3] * y_1**3 + 3.0 * polynomial[2] * y_1 * y_2 + polynomial[1] * y_3),
        )


@dataclass(frozen=True)
class DoubleExponentialPotential:
    """Two-exponential well with independently matchable second and third derivatives."""

    epsilon: float
    r_min: float
    repulsive_exponent: float
    attractive_exponent: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not (
            isfinite(self.repulsive_exponent)
            and isfinite(self.attractive_exponent)
            and self.repulsive_exponent > self.attractive_exponent > 0.0
        ):
            raise ValueError("double-exponential exponents must satisfy p > q > 0")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        displacement = r / self.r_min - 1.0
        p = self.repulsive_exponent
        q = self.attractive_exponent
        denominator = p - q
        repulsive = _exp(-p * displacement)
        attractive = _exp(-q * displacement)
        return _scale_derivatives(
            self.epsilon,
            self.r_min,
            (q * repulsive - p * attractive) / denominator,
            p * q * (attractive - repulsive) / denominator,
            p * q * (p * repulsive - q * attractive) / denominator,
            p * q * (q**2 * attractive - p**2 * repulsive) / denominator,
        )


def mie_minimum_derivatives(
    epsilon: float,
    r_min: float,
    repulsive_power: float,
    attractive_power: float,
) -> MinimumDerivatives:
    """Return exact minimum derivatives for a normalized Mie ``m-n`` well.

    The potential is ``epsilon * [n*x^-m - m*x^-n] / (m-n)``, where
    ``x = r/r_min``.  UFF is the ``m=12, n=6`` special case.
    """
    _validate_well(epsilon, r_min)
    m = float(repulsive_power)
    n = float(attractive_power)
    if not (isfinite(m) and isfinite(n) and m > n > 0.0):
        raise ValueError("Mie powers must satisfy m > n > 0")
    dimensionless_second = m * n
    dimensionless_third = -m * n * (m + n + 3.0)
    return MinimumDerivatives(
        float(epsilon),
        float(r_min),
        float(epsilon) * dimensionless_second / float(r_min) ** 2,
        float(epsilon) * dimensionless_third / float(r_min) ** 3,
    )


def exppe_from_minimum(target: MinimumDerivatives) -> ExpPEPotential:
    """Construct Exp-PE preserving ``epsilon``, ``r_min`` and curvature."""
    alpha = sqrt(2.0 * (target.dimensionless_second + 8.0))
    return ExpPEPotential(target.epsilon, target.r_min, alpha)


def damped_exppe_from_minimum(target: MinimumDerivatives) -> DampedExpPEPotential:
    """Legacy entry point preserving the undamped Exp-PE minimum and curvature."""

    potential = exppe_from_minimum(target)
    return DampedExpPEPotential(
        epsilon=potential.epsilon,
        r_min=potential.r_min,
        alpha=potential.alpha,
    )


def morse_from_minimum(target: MinimumDerivatives) -> MorsePotential:
    """Construct standard Morse preserving ``epsilon``, ``r_min`` and curvature."""
    beta = sqrt(0.5 * target.dimensionless_second)
    return MorsePotential(target.epsilon, target.r_min, beta)


def inverse_power_stretch_from_local_derivatives(
    r0: float,
    curvature: float,
    cubic: float,
) -> InversePowerStretchPotential:
    """Construct the positive inverse-power stretch matching ``k2`` and ``k3``.

    For ``V = D [1 - (r0/r)^p]^2``, the local derivatives satisfy
    ``k2 = 2 D p^2/r0^2`` and ``k3 = -6 D p^2(p+1)/r0^3``.  A non-positive
    inferred exponent signals that this two-parameter global form is
    incompatible with the supplied local derivatives; callers should then use
    a Morse or a higher-order SPF expansion rather than clipping the exponent.
    """

    reference = _validate_distance(r0)
    k2 = float(curvature)
    k3 = float(cubic)
    if not isfinite(k2) or k2 <= 0.0 or not isfinite(k3):
        raise ValueError("inverse-power construction requires finite k2 > 0 and finite k3")
    exponent = -1.0 - k3 * reference / (3.0 * k2)
    if exponent <= 0.0:
        raise ValueError(
            "local derivatives imply a non-positive inverse-power exponent; "
            "use Morse or a stabilized SPF expansion"
        )
    depth = k2 * reference**2 / (2.0 * exponent**2)
    return InversePowerStretchPotential(depth, reference, exponent)


def spf_stretch_from_local_derivatives(
    r0: float,
    curvature: float,
    cubic: float,
    *,
    quartic: float | None = None,
    higher_coefficients: tuple[float, ...] = (),
    asymptotic_depth: float | None = None,
) -> SPFStretchPotential:
    """Build an SPF series whose supplied derivatives through fourth are exact.

    ``quartic`` fixes the coefficient of ``y**4`` after including the
    chain-rule contributions from the quadratic and cubic terms.  Extra
    coefficients then start at fifth order.  When ``asymptotic_depth`` is
    supplied, one final coefficient enforces ``sum(c_n) = D_e`` and therefore
    ``V(r -> infinity) = D_e`` without changing derivatives through fourth
    order.
    """

    reference = _validate_distance(r0)
    k2 = float(curvature)
    k3 = float(cubic)
    k4 = None if quartic is None else float(quartic)
    extras = tuple(float(value) for value in higher_coefficients)
    if not isfinite(k2) or k2 <= 0.0 or not isfinite(k3) or (k4 is not None and not isfinite(k4)):
        raise ValueError("SPF construction requires finite derivatives and k2 > 0")
    if not all(isfinite(value) for value in extras):
        raise ValueError("higher SPF coefficients must be finite")
    depth = None if asymptotic_depth is None else float(asymptotic_depth)
    if depth is not None and (not isfinite(depth) or depth <= 0.0):
        raise ValueError("SPF asymptotic depth must be finite and positive")
    c2 = 0.5 * k2 * reference**2
    c3 = k2 * reference**2 + k3 * reference**3 / 6.0
    coefficients = (c2, c3)
    if k4 is not None:
        c4 = 1.5 * k2 * reference**2 + 0.5 * k3 * reference**3 + k4 * reference**4 / 24.0
        coefficients += (c4,)
    coefficients += extras
    if depth is not None:
        coefficients += (depth - sum(coefficients),)
    return SPFStretchPotential(reference, coefficients)


def inverse_power_stretch_from_morse_prior(
    depth: float,
    r0: float,
    alpha: float,
) -> InversePowerStretchPotential:
    """Translate one Morse prior to the permanent monotone ``1/r`` base.

    With ``p = alpha*r0``, ``D_e[1-(r0/r)**p]**2`` preserves the Morse
    minimum, dissociation depth, and harmonic curvature. Higher acquired
    derivatives are represented by additional inverse-distance corrections;
    the base function is never replaced at a later ZAFF zoom.
    """

    well = float(depth)
    reference = float(r0)
    exponent = float(alpha)
    _validate_well(well, reference)
    if not isfinite(exponent) or exponent <= 0.0:
        raise ValueError("Morse-to-inverse-power translation requires alpha > 0")
    return InversePowerStretchPotential(well, reference, exponent * reference)


def morse_from_zero_crossing(
    epsilon: float,
    r_min: float,
    zero_crossing: float,
) -> MorsePotential:
    """Apply the Abraham--Stolevik minimum/zero-crossing Morse mapping."""
    _validate_well(epsilon, r_min)
    r_zero = _validate_distance(zero_crossing)
    if r_zero >= r_min:
        raise ValueError("Morse zero crossing must lie below the minimum distance")
    beta = log(2.0) / (1.0 - r_zero / r_min)
    return MorsePotential(float(epsilon), float(r_min), beta)


def double_exponential_from_minimum(target: MinimumDerivatives) -> DoubleExponentialPotential:
    """Construct a double exponential preserving derivatives through third order."""
    dimensionless_third = target.dimensionless_third
    if dimensionless_third is None:
        raise ValueError("double-exponential conversion requires the cubic derivative")
    product = target.dimensionless_second
    total = -dimensionless_third / product
    discriminant = total**2 - 4.0 * product
    if total <= 0.0 or discriminant < -1.0e-12:
        raise ValueError("target derivatives do not define two positive real exponents")
    root = sqrt(max(0.0, discriminant))
    repulsive = 0.5 * (total + root)
    attractive = 0.5 * (total - root)
    return DoubleExponentialPotential(target.epsilon, target.r_min, repulsive, attractive)


def uff_to_exppe(epsilon: float, r_min: float) -> ExpPEPotential:
    """Convert a UFF 12-6 pair while preserving its minimum Hessian."""
    return exppe_from_minimum(mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0))


def uff_to_double_exponential(epsilon: float, r_min: float) -> DoubleExponentialPotential:
    """Convert a UFF 12-6 pair through the cubic derivative at the minimum."""
    return double_exponential_from_minimum(mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0))


def uff_pair_to_exppe(atomic_number_i: int, atomic_number_j: int) -> ExpPEPotential:
    """Convert the mixed MATRIX/UFF pair for two elements to Exp-PE.

    The returned potential uses hartree and bohr, matching the analytic
    non-bonded derivative engine in :mod:`matrix_chem`.
    """
    from matrix_chem import uff_pair_parameters

    r_min, epsilon = uff_pair_parameters(atomic_number_i, atomic_number_j)
    return uff_to_exppe(epsilon, r_min)


def uff_pair_to_double_exponential(
    atomic_number_i: int,
    atomic_number_j: int,
) -> DoubleExponentialPotential:
    """Convert the mixed MATRIX/UFF pair through cubic order at its minimum."""
    from matrix_chem import uff_pair_parameters

    r_min, epsilon = uff_pair_parameters(atomic_number_i, atomic_number_j)
    return uff_to_double_exponential(epsilon, r_min)


def _scale_derivatives(
    epsilon: float,
    r_min: float,
    energy: float,
    first: float,
    second: float,
    third: float,
) -> RadialDerivatives:
    return RadialDerivatives(
        float(epsilon) * energy,
        float(epsilon) * first / float(r_min),
        float(epsilon) * second / float(r_min) ** 2,
        float(epsilon) * third / float(r_min) ** 3,
    )


def _validate_well(epsilon: float, r_min: float) -> None:
    if not isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("radial well depth must be positive")
    if not isfinite(float(r_min)) or float(r_min) <= 0.0:
        raise ValueError("radial minimum distance must be positive")


def _validate_distance(distance: float) -> float:
    value = float(distance)
    if not isfinite(value) or value <= 0.0:
        raise ValueError("radial distance must be positive")
    return value


def _exp(value: float) -> float:
    # Kept local so every model follows the same scalar evaluation path.
    return exp(value)


@dataclass(frozen=True)
class PiecewiseInverseDistancePotential:
    """Piecewise analytic radial expansion in u=r0/r.

    Coefficients use ascending powers of u-knots[i]. The u=0 interval
    explicitly carries the dissociation asymptote. A caller must qualify the
    finite radial domain and stationary points before using this field.
    """

    r0: float
    knots: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isfinite(self.r0) or self.r0 <= 0:
            raise ValueError('positive finite radial reference required')
        if (len(self.knots) < 2 or len(self.coefficients) != len(self.knots)-1
                or self.knots[0] != 0 or any(not isfinite(x) for x in self.knots)
                or any(b <= a for a,b in zip(self.knots,self.knots[1:]))
                or any(not row or any(not isfinite(c) for c in row) for row in self.coefficients)):
            raise ValueError('ordered inverse-distance knots and finite interval coefficients required')

    def derivatives(self, distance: float) -> RadialDerivatives:
        from bisect import bisect_right
        r = _validate_distance(distance)
        u = self.r0/r
        if u > self.knots[-1]:
            raise ValueError('distance is outside qualified inverse-distance domain')
        i = min(bisect_right(self.knots,u)-1,len(self.coefficients)-1)
        x = u-self.knots[i]
        row = self.coefficients[i]
        values = []
        for order in range(4):
            value = 0.
            for n in range(len(row)-1,order-1,-1):
                factor = 1
                for j in range(order): factor *= n-j
                value = value*x+factor*row[n]
            values.append(value)
        u1,u2,u3 = -self.r0/r**2,2*self.r0/r**3,-6*self.r0/r**4
        return RadialDerivatives(values[0],values[1]*u1,
            values[2]*u1*u1+values[1]*u2,
            values[3]*u1**3+3*values[2]*u1*u2+values[1]*u3)


def inverse_distance_potential_from_jets(r0, radii, radial_jets, *,
                                         dissociation_energy, c6):
    """Interpolate fixed radial jets in 1/R, preserving supplied derivatives.

    Each jet is [V,dV/dR,...], with up to four derivatives. An endpoint at
    u=0 enforces V=D-C6/R^6+O(R^-7). No observations are acquired here.
    This constructs a representation, not a global-shape certificate.
    """
    from math import factorial
    from scipy.interpolate import BPoly,PPoly
    if not isfinite(c6) or c6 <= 0 or not isfinite(dissociation_energy):
        raise ValueError('finite asymptote and positive C6 required')
    if len(radii) != len(radial_jets) or len(radii) == 0:
        raise ValueError('paired nonempty radii and jets required')
    r0 = _validate_distance(r0)
    pairs = sorted(zip(radii,radial_jets),reverse=True)
    knots = [0.]
    jets = [[dissociation_energy,0.,0.,0.,0.,0.,-factorial(6)*c6/r0**6]]
    for r,raw in pairs:
        r = _validate_distance(r)
        v = tuple(float(x) for x in raw)
        if not 1 <= len(v) <= 5 or not all(isfinite(x) for x in v):
            raise ValueError('finite radial jets through at most fourth order required')
        a,b,c,d = -r*r/r0,2*r**3/r0**2,-6*r**4/r0**3,24*r**5/r0**4
        out = [v[0]]
        if len(v)>1:out.append(v[1]*a)
        if len(v)>2:out.append(v[2]*a*a+v[1]*b)
        if len(v)>3:out.append(v[3]*a**3+3*v[2]*a*b+v[1]*c)
        if len(v)>4:out.append(v[4]*a**4+6*v[3]*a*a*b+3*v[2]*b*b+4*v[2]*a*c+v[1]*d)
        knots.append(r0/r);jets.append(out)
    polynomial = PPoly.from_bernstein_basis(BPoly.from_derivatives(knots,jets))
    coefficients = [list(float(c) for c in row[::-1]) for row in polynomial.c.T]
    # Retain exact asymptotic zeros; roundoff must not create lower inverse powers.
    coefficients[0][:7] = [jets[0][n]/factorial(n) for n in range(7)]
    return PiecewiseInverseDistancePotential(float(r0),tuple(knots),
        tuple(tuple(row) for row in coefficients))
