"""Regular ring-puckering charts and analytic resident ZAFF surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import atan2, factorial, hypot, pi, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from .bonded import InternalDerivatives


ZAFF_RING_PUCKERING_SURFACE_SCHEMA = "matrix.zaff.ring_puckering_surface.v1"


@dataclass(frozen=True)
class ZaffRingDimensionContract:
    """Dimension-first RPck contract for a ring of arbitrary size N >= 5."""

    ring_size: int

    def __post_init__(self) -> None:
        if int(self.ring_size) < 5:
            raise ValueError("collective ring puckering requires at least five ring atoms")
        object.__setattr__(self, "ring_size", int(self.ring_size))

    @property
    def puckering_dimension(self) -> int:
        return self.ring_size - 3

    @property
    def phase_count(self) -> int:
        return self.puckering_dimension - 1

    @property
    def angular_basis(self) -> str:
        if self.puckering_dimension == 2:
            return "REAL_FOURIER_HARMONICS_ON_S1"
        if self.puckering_dimension == 3:
            return "REAL_SPHERICAL_HARMONICS_ON_S2"
        return f"REAL_HYPERSPHERICAL_HARMONICS_ON_S{self.puckering_dimension - 1}"

    @property
    def implementation_status(self) -> str:
        return (
            "EXECUTABLE_BY_COMMON_FIVE_SIX_RING_WORKFLOW_REQUIRES_MOLECULAR_VALIDATION"
            if self.ring_size in {5, 6}
            else "GENERIC_CONTRACT_REQUIRES_PROVIDER_CHART_AND_MOLECULAR_VALIDATION"
        )

    def to_dict(self) -> dict[str, Any]:
        dimension = self.puckering_dimension
        return {
            "schema": "matrix.zaff.ring_dimension_contract.v1",
            "ring_size": self.ring_size,
            "puckering_dimension": dimension,
            "native_components": [f"RPck{index + 1}" for index in range(dimension)],
            "canonical_descriptor": "CREMER_POPLE_CARTESIAN_COMPONENTS",
            "optimizer_working_chart": "BALANCED_TRIANGULAR_FLAPS",
            "zaff_surface_chart": "CHARM",
            "amplitude": "Q=SQRT(SUM_i RPck_i^2)",
            "phase_count": self.phase_count,
            "angular_manifold": f"S^{dimension - 1}",
            "angular_basis": self.angular_basis,
            "radial_tail": "(Q0^2+Q^2)^(-(ell+1+k)/2)",
            "confinement": "Q^2*EXP(beta*Q^2); beta>0",
            "zaff_surface_basis": "INVERSE_RADIAL_REAL_HYPERSPHERICAL_HARMONICS",
            "standalone_polynomial_terms": False,
            "reference_embedding": "SUBTRACT_VALUE_GRADIENT_AND_HESSIAN_AT_REFERENCE",
            "implementation_status": self.implementation_status,
        }


def ring_puckering_dimension_contract(ring_size: int) -> ZaffRingDimensionContract:
    """Return the reproducible N -> (N-3)-dimensional puckering contract."""

    return ZaffRingDimensionContract(ring_size=int(ring_size))


@dataclass(frozen=True)
class ZaffRingPuckeringChart:
    ring_atoms: tuple[int, ...]
    source_coordinate_indices: tuple[int, ...]
    source_coordinate_names: tuple[str, ...]
    priority_atom: int
    azimuthal_periodicity: int
    polar_periodicity: int | None
    symmetry_number: int
    barrier_seed_kcal_mol: float

    def __post_init__(self) -> None:
        size = len(self.ring_atoms)
        expected = 2 if size == 5 else 3 if size == 6 else 0
        if expected == 0 or len(self.source_coordinate_indices) != expected:
            raise ValueError("ZAFF ring-puckering charts currently require five or six atoms")
        if len(self.source_coordinate_names) != expected:
            raise ValueError("ring-puckering source names and indices differ")
        if self.priority_atom not in self.ring_atoms:
            raise ValueError("ring-puckering priority atom is outside the ring")

    @property
    def ring_size(self) -> int:
        return len(self.ring_atoms)

    @property
    def chair_sign(self) -> float:
        if self.ring_size != 6:
            return 1.0
        return 1.0 if self.ring_atoms.index(self.priority_atom) % 2 == 0 else -1.0

    def readable_coordinates(
        self,
        native_components: Sequence[float],
        *,
        zero_tolerance: float = 1.0e-10,
    ) -> Mapping[str, float | str]:
        values = np.asarray(native_components, dtype=float).reshape(-1)
        if values.size != len(self.source_coordinate_indices):
            raise ValueError("wrong number of native RPck components")
        radial = hypot(float(values[0]), float(values[1]))
        azimuth = 0.0 if radial <= zero_tolerance else atan2(float(values[1]), float(values[0]))
        result: dict[str, float | str] = {
            "amplitude": float(np.linalg.norm(values)),
            "azimuth_radian": azimuth,
            "azimuth_degree": float(np.degrees(azimuth)),
            "azimuth_status": (
                "GAUGE_FIXED_ZERO_RADIAL_AMPLITUDE" if radial <= zero_tolerance else "DEFINED"
            ),
        }
        if self.ring_size == 6:
            oriented = self.chair_sign * float(values[2])
            total = hypot(radial, oriented)
            polar = 0.0 if total <= zero_tolerance else atan2(radial, oriented)
            if abs(polar) <= zero_tolerance:
                polar = 0.0
            result.update(
                {
                    "polar_radian": polar,
                    "polar_degree": float(np.degrees(polar)),
                    "polar_status": (
                        "GAUGE_FIXED_ZERO_TOTAL_AMPLITUDE" if total <= zero_tolerance else "DEFINED"
                    ),
                }
            )
        return result


@dataclass(frozen=True)
class ZaffRingPuckeringScanPoint:
    identifier: str
    amplitude: float
    azimuth_radian: float
    polar_radian: float | None
    purpose: str


@dataclass(frozen=True)
class ZaffRingPuckeringGeometryPoint:
    scan_point: ZaffRingPuckeringScanPoint
    target_native_components: tuple[float, ...]
    achieved_native_components: tuple[float, ...]
    coordinates_angstrom: np.ndarray
    residual_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.scan_point.identifier,
            "purpose": self.scan_point.purpose,
            "amplitude": self.scan_point.amplitude,
            "azimuth_radian": self.scan_point.azimuth_radian,
            "azimuth_degree": float(np.degrees(self.scan_point.azimuth_radian)),
            "polar_radian": self.scan_point.polar_radian,
            "polar_degree": (
                None
                if self.scan_point.polar_radian is None
                else float(np.degrees(self.scan_point.polar_radian))
            ),
            "target_native_components": list(self.target_native_components),
            "achieved_native_components": list(self.achieved_native_components),
            "coordinates_angstrom": np.asarray(self.coordinates_angstrom, dtype=float).tolist(),
            "residual_norm": self.residual_norm,
        }


def native_components_from_ring_puckering_point(
    chart: ZaffRingPuckeringChart,
    point: ZaffRingPuckeringScanPoint,
) -> tuple[float, ...]:
    radial = float(point.amplitude)
    if chart.ring_size == 6:
        if point.polar_radian is None:
            raise ValueError("six-membered ring scan point has no polar coordinate")
        radial *= float(np.sin(point.polar_radian))
    x = radial * float(np.cos(point.azimuth_radian))
    y = radial * float(np.sin(point.azimuth_radian))
    if chart.ring_size == 5:
        return x, y
    oriented_z = float(point.amplitude) * float(np.cos(point.polar_radian))
    return x, y, oriented_z / chart.chair_sign


def build_ring_puckering_scan_plan(
    chart: ZaffRingPuckeringChart,
    reference_components: Sequence[float],
    *,
    amplitude_scales: Sequence[float] = (1.0,),
    azimuth_count: int | None = None,
    polar_count: int = 7,
) -> tuple[ZaffRingPuckeringScanPoint, ...]:
    """Return a pole-safe full scan; molecular symmetry controls the angular span."""

    readable = chart.readable_coordinates(reference_components)
    amplitude = max(float(readable["amplitude"]), 0.05)
    scales = tuple(float(value) for value in amplitude_scales)
    if not scales or any(value <= 0.0 for value in scales):
        raise ValueError("ring-puckering amplitude scales must be positive")
    azimuth_count = (
        2 * max(3, chart.azimuthal_periodicity) + 1 if azimuth_count is None else int(azimuth_count)
    )
    if azimuth_count < 3 or polar_count < 3:
        raise ValueError("ring-puckering angular grids require at least three points")
    azimuth_span = 2.0 * pi / max(1, chart.symmetry_number)
    azimuths = np.linspace(0.0, azimuth_span, azimuth_count, endpoint=False)
    if chart.ring_size == 5:
        return tuple(
            ZaffRingPuckeringScanPoint(
                identifier=(
                    f"Q{int(round(100.0 * scale)):03d}_PHI_{index:03d}"
                    if len(scales) > 1
                    else f"PHI_{index:03d}"
                ),
                amplitude=amplitude * scale,
                azimuth_radian=float(phi),
                polar_radian=None,
                purpose="FIVE_RING_PSEUDOROTATION",
            )
            for scale in scales
            for index, phi in enumerate(azimuths)
        )
    # Never sample phi exactly at a chair pole: all azimuths represent the same point there.
    polar_values = np.linspace(0.0, pi, int(polar_count))
    points: list[ZaffRingPuckeringScanPoint] = []
    for scale in scales:
        for theta_index, theta in enumerate(polar_values):
            selected_azimuths = (0.0,) if theta in {0.0, pi} else azimuths
            for phi_index, phi in enumerate(selected_azimuths):
                points.append(
                    ZaffRingPuckeringScanPoint(
                        identifier=(
                            f"Q{int(round(100.0 * scale)):03d}_"
                            f"THETA_{theta_index:02d}_PHI_{phi_index:03d}"
                            if len(scales) > 1
                            else f"THETA_{theta_index:02d}_PHI_{phi_index:03d}"
                        ),
                        amplitude=amplitude * scale,
                        azimuth_radian=float(phi),
                        polar_radian=float(theta),
                        purpose="SIX_RING_HYPERSPHERICAL_SURFACE",
                    )
                )
    return tuple(points)


@dataclass(frozen=True)
class ZaffRingPuckeringValue:
    energy_hartree: float
    gradient_native: np.ndarray
    hessian_native: np.ndarray


@dataclass(frozen=True)
class ZaffTeamPlusValenceRingPhasePotential:
    r"""Pole-safe TEAM+ coupling expressed through the two six-ring RPck phases."""

    reference_valence: float
    reference_components: tuple[float, float, float]
    polar_coefficients: tuple[float, float, float]
    azimuth_cosine_coefficients: tuple[float, float, float]
    azimuth_sine_coefficients: tuple[float, float, float]
    valence_kind: str
    chair_sign: float = 1.0

    def __post_init__(self) -> None:
        if any(
            len(values) != 3
            for values in (
                self.reference_components,
                self.polar_coefficients,
                self.azimuth_cosine_coefficients,
                self.azimuth_sine_coefficients,
            )
        ):
            raise ValueError("TEAM+ six-ring coupling requires three components per block")
        values = (
            self.reference_valence,
            *self.reference_components,
            *self.polar_coefficients,
            *self.azimuth_cosine_coefficients,
            *self.azimuth_sine_coefficients,
            self.chair_sign,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("TEAM+ ring-phase parameters must be finite")
        if np.linalg.norm(self.reference_components) <= 1.0e-10:
            raise ValueError("TEAM+ ring phases require nonzero reference puckering amplitude")
        kind = str(self.valence_kind).upper()
        if kind not in {"BOND", "ANGLE"}:
            raise ValueError("TEAM+ ring-phase valence coordinate must be BOND or ANGLE")
        if not np.isclose(abs(float(self.chair_sign)), 1.0):
            raise ValueError("TEAM+ ring-phase chair sign must be +1 or -1")
        object.__setattr__(self, "valence_kind", kind)
        object.__setattr__(self, "chair_sign", 1.0 if self.chair_sign > 0.0 else -1.0)

    def derivatives(
        self,
        valence: float,
        native_components: Sequence[float],
    ) -> InternalDerivatives:
        values = np.asarray(native_components, dtype=float).reshape(-1)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("TEAM+ six-ring coupling requires three finite RPck components")
        variables = (
            _Jet2.variable(float(valence), 0, 4),
            *(
                _Jet2.variable(float(value), index + 1, 4)
                for index, value in enumerate(values)
            ),
        )
        displacement = variables[0] - self.reference_valence
        shape = self._phase_shape(variables[1:])
        result = displacement * shape
        return InternalDerivatives(result.value, result.gradient, result.hessian)

    def anharmonic_correction_derivatives(
        self,
        valence: float,
        native_component_1: float,
        native_component_2: float,
        native_component_3: float,
    ) -> InternalDerivatives:
        """Remove reference stationarity and mixed harmonic terms from the kernel."""

        values = np.asarray(
            (native_component_1, native_component_2, native_component_3), dtype=float
        )
        variables = tuple(_Jet2.variable(float(value), index, 3) for index, value in enumerate(values))
        reference_variables = tuple(
            _Jet2.variable(float(value), index, 3)
            for index, value in enumerate(self.reference_components)
        )
        shape = self._phase_shape(variables)
        reference = self._phase_shape(reference_variables)
        native_displacement = values - np.asarray(self.reference_components)
        valence_displacement = float(valence) - self.reference_valence
        corrected_shape = (
            shape.value - reference.value - float(reference.gradient @ native_displacement)
        )
        gradient = np.concatenate(
            ((corrected_shape,), valence_displacement * (shape.gradient - reference.gradient))
        )
        hessian = np.zeros((4, 4), dtype=float)
        hessian[0, 1:] = shape.gradient - reference.gradient
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = valence_displacement * shape.hessian
        return InternalDerivatives(
            energy=valence_displacement * corrected_shape,
            gradient=gradient,
            hessian=hessian,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_form": "TEAM_PLUS_VALENCE_RING_PHASES_POLE_SAFE",
            "valence_kind": self.valence_kind,
            "reference_valence": self.reference_valence,
            "reference_native_components": list(self.reference_components),
            "chair_sign": self.chair_sign,
            "polar_coefficients_hartree": list(self.polar_coefficients),
            "azimuth_cosine_coefficients_hartree": list(
                self.azimuth_cosine_coefficients
            ),
            "azimuth_sine_coefficients_hartree": list(self.azimuth_sine_coefficients),
            "phase_definitions": {
                "phi1": "ATAN2(RPck2,RPck1)",
                "phi2": "ATAN2(SQRT(RPck1^2+RPck2^2),chair_sign*RPck3)",
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ZaffTeamPlusValenceRingPhasePotential":
        if payload.get("functional_form") != "TEAM_PLUS_VALENCE_RING_PHASES_POLE_SAFE":
            raise ValueError("unsupported TEAM+ ring-phase functional form")
        return cls(
            reference_valence=float(payload["reference_valence"]),
            reference_components=tuple(
                float(value) for value in payload["reference_native_components"]
            ),
            polar_coefficients=tuple(
                float(value) for value in payload["polar_coefficients_hartree"]
            ),
            azimuth_cosine_coefficients=tuple(
                float(value) for value in payload["azimuth_cosine_coefficients_hartree"]
            ),
            azimuth_sine_coefficients=tuple(
                float(value) for value in payload["azimuth_sine_coefficients_hartree"]
            ),
            valence_kind=str(payload["valence_kind"]),
            chair_sign=float(payload.get("chair_sign", 1.0)),
        )

    def _phase_shape(self, components: Sequence["_Jet2"]) -> "_Jet2":
        x, y, native_z = components
        dimension = len(x.gradient)
        z = self.chair_sign * native_z
        inverse_amplitude = (x * x + y * y + z * z) ** -0.5
        unit_x, unit_y, unit_z = x * inverse_amplitude, y * inverse_amplitude, z * inverse_amplitude
        polar_terms = [unit_z]
        if len(self.polar_coefficients) > 1:
            polar_terms.append(2.0 * unit_z * unit_z - 1.0)
        if len(self.polar_coefficients) > 2:
            polar_terms.append(4.0 * unit_z**3 - 3.0 * unit_z)
        real, imaginary = _Jet2.constant(1.0, dimension), _Jet2.constant(0.0, dimension)
        azimuth_real = []
        azimuth_imaginary = []
        for _order in range(3):
            real, imaginary = (
                real * unit_x - imaginary * unit_y,
                real * unit_y + imaginary * unit_x,
            )
            azimuth_real.append(real)
            azimuth_imaginary.append(imaginary)
        return sum(
            (
                coefficient * term
                for coefficient, term in zip(
                    self.polar_coefficients, polar_terms, strict=True
                )
            ),
            _Jet2.constant(0.0, dimension),
        ) + sum(
            (
                cosine * real_term + sine * imaginary_term
                for cosine, sine, real_term, imaginary_term in zip(
                    self.azimuth_cosine_coefficients,
                    self.azimuth_sine_coefficients,
                    azimuth_real,
                    azimuth_imaginary,
                    strict=True,
                )
            ),
            _Jet2.constant(0.0, dimension),
        )


@dataclass(frozen=True)
class ZaffTeamPlusRingPhaseFit:
    potential: ZaffTeamPlusValenceRingPhasePotential
    design_rank: int
    coefficient_count: int
    observation_count: int
    residual_rmse: float
    singular_values: tuple[float, ...]

    @property
    def identifiable(self) -> bool:
        return self.design_rank == self.coefficient_count


def fit_team_plus_ring_phase_from_mixed_hessians(
    *,
    reference_valence: float,
    reference_components: Sequence[float],
    native_components: Sequence[Sequence[float]],
    mixed_valence_native_hessians: Sequence[Sequence[float]],
    valence_kind: str,
    chair_sign: float = 1.0,
) -> ZaffTeamPlusRingPhaseFit:
    """Fit the compact two-phase TEAM+ kernel from valence--RPck Hessian blocks."""

    points = np.asarray(native_components, dtype=float)
    targets = np.asarray(mixed_valence_native_hessians, dtype=float)
    reference = tuple(float(value) for value in reference_components)
    if points.ndim != 2 or points.shape[1] != 3 or targets.shape != points.shape:
        raise ValueError("TEAM+ ring-phase fit requires paired N x 3 points and Hessian blocks")
    if len(points) < 3 or not np.all(np.isfinite(points)) or not np.all(np.isfinite(targets)):
        raise ValueError("TEAM+ ring-phase fit requires at least three finite observations")
    coefficient_vectors = np.eye(9)
    columns = []
    for vector in coefficient_vectors:
        basis = ZaffTeamPlusValenceRingPhasePotential(
            reference_valence=float(reference_valence),
            reference_components=reference,
            polar_coefficients=tuple(float(value) for value in vector[:3]),
            azimuth_cosine_coefficients=tuple(float(value) for value in vector[3:6]),
            azimuth_sine_coefficients=tuple(float(value) for value in vector[6:]),
            valence_kind=valence_kind,
            chair_sign=chair_sign,
        )
        columns.append(
            np.concatenate(
                [
                    basis.derivatives(float(reference_valence), point).hessian[0, 1:]
                    for point in points
                ]
            )
        )
    design = np.column_stack(columns)
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design, targets.reshape(-1), rcond=1.0e-11
    )
    residual = design @ coefficients - targets.reshape(-1)
    potential = ZaffTeamPlusValenceRingPhasePotential(
        reference_valence=float(reference_valence),
        reference_components=reference,
        polar_coefficients=tuple(float(value) for value in coefficients[:3]),
        azimuth_cosine_coefficients=tuple(float(value) for value in coefficients[3:6]),
        azimuth_sine_coefficients=tuple(float(value) for value in coefficients[6:]),
        valence_kind=valence_kind,
        chair_sign=chair_sign,
    )
    return ZaffTeamPlusRingPhaseFit(
        potential=potential,
        design_rank=int(rank),
        coefficient_count=design.shape[1],
        observation_count=design.shape[0],
        residual_rmse=float(np.sqrt(np.mean(residual**2))),
        singular_values=tuple(float(value) for value in singular_values),
    )


def serialize_team_plus_ring_phase_coupled_term(
    potential: ZaffTeamPlusValenceRingPhasePotential,
    *,
    identifier: str,
    valence_index: int,
    ring_component_indices: Sequence[int],
) -> dict[str, Any]:
    """Build the runtime record for one fitted valence--(PhiP1,PhiP2) coupling."""

    ring_indices = tuple(int(index) for index in ring_component_indices)
    ordered = (int(valence_index), *ring_indices)
    if len(ring_indices) != 3 or len(set(ordered)) != 4 or any(index < 0 for index in ordered):
        raise ValueError("TEAM+ ring-phase runtime indices must be four distinct coordinates")
    return {
        "schema": "matrix.zaff.coupled_term.v1",
        "identifier": str(identifier),
        "coordinate_indices_zero_based": {
            "valence": int(valence_index),
            "ring_components": list(ring_indices),
        },
        "potential": potential.to_dict(),
        "runtime_embedding": {
            "phase_coordinates": ["PhiP1", "PhiP2"],
            "analytic_runtime_coordinates": ["RPck1", "RPck2", "RPck3"],
            "reference_gradient_and_mixed_harmonic": (
                "ALREADY_PRESENT_IN_BASE_SONIC_TAYLOR_FIELD"
            ),
            "double_counting": False,
            "pole_safe": True,
        },
    }


@dataclass(frozen=True)
class ZaffRingPuckeringSurface:
    """A pole-regular polynomial/Fourier surface in native RPck components."""

    chart: ZaffRingPuckeringChart
    reference_components: tuple[float, ...]
    coefficients_hartree: tuple[float, ...]
    fit_rmse_hartree: float
    maximum_residual_hartree: float
    fit_gradient_rmse_hartree_per_radian: float | None = None
    gradient_observation_weight_radian: float | None = None
    native_component_unit: str = "angstrom"
    # Keep the record constructor backward-compatible; production fitting
    # selects the inverse-radial hyperspherical basis by default below.
    basis_kind: str = "LEGACY_RING_INVARIANTS"
    polynomial_degree: int = 4
    enforced_symmetry_number: int = 1
    radial_scale: float = 1.0
    confinement_beta: float = 1.0
    inverse_radial_order_count: int = 4
    schema: str = ZAFF_RING_PUCKERING_SURFACE_SCHEMA

    def __post_init__(self) -> None:
        expected = len(
            _surface_basis_values(
                np.asarray(self.reference_components, dtype=float),
                self.chart,
                basis_kind=self.basis_kind,
                polynomial_degree=self.polynomial_degree,
                radial_scale=self.radial_scale,
                confinement_beta=self.confinement_beta,
                inverse_radial_order_count=self.inverse_radial_order_count,
            )
        )
        if len(self.coefficients_hartree) != expected:
            raise ValueError("ring-puckering coefficient count differs from its regular basis")
        if self.schema != ZAFF_RING_PUCKERING_SURFACE_SCHEMA:
            raise ValueError("unsupported ring-puckering surface schema")
        if self.native_component_unit not in {"angstrom", "radian"}:
            raise ValueError("ring-puckering native component unit must be angstrom or radian")

    @property
    def fit_gradient_rmse_hartree_per_native_unit(self) -> float | None:
        return self.fit_gradient_rmse_hartree_per_radian

    @property
    def gradient_observation_weight_native(self) -> float | None:
        return self.gradient_observation_weight_radian

    def derivatives(self, native_components: Sequence[float]) -> ZaffRingPuckeringValue:
        values = np.asarray(native_components, dtype=float).reshape(-1)
        jets = _surface_basis_jets(
            values,
            self.chart,
            basis_kind=self.basis_kind,
            polynomial_degree=self.polynomial_degree,
            radial_scale=self.radial_scale,
            confinement_beta=self.confinement_beta,
            inverse_radial_order_count=self.inverse_radial_order_count,
        )
        result = sum(
            (
                coefficient * jet
                for coefficient, jet in zip(self.coefficients_hartree, jets, strict=True)
            ),
            _Jet2.constant(0.0, len(values)),
        )
        return ZaffRingPuckeringValue(result.value, result.gradient, result.hessian)

    def correction(self, native_components: Sequence[float]) -> ZaffRingPuckeringValue:
        """Return only the beyond-harmonic part, preserving the compiled Hessian."""

        values = np.asarray(native_components, dtype=float).reshape(-1)
        reference = np.asarray(self.reference_components, dtype=float)
        actual = self.derivatives(values)
        origin = self.derivatives(reference)
        delta = values - reference
        taylor_energy = (
            origin.energy_hartree
            + float(origin.gradient_native @ delta)
            + 0.5 * float(delta @ origin.hessian_native @ delta)
        )
        return ZaffRingPuckeringValue(
            energy_hartree=actual.energy_hartree - taylor_energy,
            gradient_native=(
                actual.gradient_native - origin.gradient_native - origin.hessian_native @ delta
            ),
            hessian_native=actual.hessian_native - origin.hessian_native,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "chart": {
                "ring_atoms": list(self.chart.ring_atoms),
                "source_coordinate_indices": list(self.chart.source_coordinate_indices),
                "source_coordinate_names": list(self.chart.source_coordinate_names),
                "priority_atom": self.chart.priority_atom,
                "azimuthal_periodicity": self.chart.azimuthal_periodicity,
                "polar_periodicity": self.chart.polar_periodicity,
                "symmetry_number": self.chart.symmetry_number,
                "barrier_seed_kcal_mol": self.chart.barrier_seed_kcal_mol,
            },
            "reference_components": list(self.reference_components),
            "coefficients_hartree": list(self.coefficients_hartree),
            "fit_rmse_hartree": self.fit_rmse_hartree,
            "maximum_residual_hartree": self.maximum_residual_hartree,
            "native_component_unit": self.native_component_unit,
            "fit_gradient_rmse_hartree_per_native_unit": (
                self.fit_gradient_rmse_hartree_per_native_unit
            ),
            "gradient_observation_weight_native": self.gradient_observation_weight_native,
            "basis_kind": self.basis_kind,
            "polynomial_degree": self.polynomial_degree,
            "enforced_symmetry_number": self.enforced_symmetry_number,
            "radial_scale": self.radial_scale,
            "confinement_beta": self.confinement_beta,
            "inverse_radial_order_count": self.inverse_radial_order_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffRingPuckeringSurface":
        if payload.get("schema") != ZAFF_RING_PUCKERING_SURFACE_SCHEMA:
            raise ValueError("unsupported ring-puckering surface schema")
        chart = payload["chart"]
        gradient_rmse = payload.get(
            "fit_gradient_rmse_hartree_per_native_unit",
            payload.get("fit_gradient_rmse_hartree_per_radian"),
        )
        gradient_weight = payload.get(
            "gradient_observation_weight_native",
            payload.get("gradient_observation_weight_radian"),
        )
        return cls(
            chart=ZaffRingPuckeringChart(
                ring_atoms=tuple(int(value) for value in chart["ring_atoms"]),
                source_coordinate_indices=tuple(
                    int(value) for value in chart["source_coordinate_indices"]
                ),
                source_coordinate_names=tuple(
                    str(value) for value in chart["source_coordinate_names"]
                ),
                priority_atom=int(chart["priority_atom"]),
                azimuthal_periodicity=int(chart["azimuthal_periodicity"]),
                polar_periodicity=(
                    None
                    if chart.get("polar_periodicity") is None
                    else int(chart["polar_periodicity"])
                ),
                symmetry_number=int(chart["symmetry_number"]),
                barrier_seed_kcal_mol=float(chart["barrier_seed_kcal_mol"]),
            ),
            reference_components=tuple(float(value) for value in payload["reference_components"]),
            coefficients_hartree=tuple(float(value) for value in payload["coefficients_hartree"]),
            fit_rmse_hartree=float(payload["fit_rmse_hartree"]),
            maximum_residual_hartree=float(payload["maximum_residual_hartree"]),
            fit_gradient_rmse_hartree_per_radian=(
                None if gradient_rmse is None else float(gradient_rmse)
            ),
            gradient_observation_weight_radian=(
                None if gradient_weight is None else float(gradient_weight)
            ),
            native_component_unit=str(payload.get("native_component_unit", "radian")),
            # v1 records written before basis_kind was mandatory used the
            # compact legacy invariant basis.
            basis_kind=str(payload.get("basis_kind", "LEGACY_RING_INVARIANTS")),
            polynomial_degree=int(payload.get("polynomial_degree", 4)),
            enforced_symmetry_number=int(payload.get("enforced_symmetry_number", 1)),
            radial_scale=float(payload.get("radial_scale", 1.0)),
            confinement_beta=float(payload.get("confinement_beta", 1.0)),
            inverse_radial_order_count=int(payload.get("inverse_radial_order_count", 4)),
        )


def fit_ring_puckering_surface(
    chart: ZaffRingPuckeringChart,
    native_components: Sequence[Sequence[float]],
    energies_hartree: Sequence[float],
    *,
    reference_components: Sequence[float],
    basis_kind: str = "INVERSE_RADIAL_SPHERICAL_HARMONICS",
    polynomial_degree: int = 4,
    gradients_native: Sequence[Sequence[float]] | None = None,
    gradient_observation_weight_radian: float = 0.15,
    enforce_chart_symmetry: bool = True,
    radial_scale: float | None = None,
    confinement_beta: float | None = None,
    inverse_radial_order_count: int = 4,
) -> ZaffRingPuckeringSurface:
    points = np.asarray(native_components, dtype=float)
    energies = np.asarray(energies_hartree, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != len(chart.source_coordinate_indices):
        raise ValueError("ring-puckering observations have the wrong component dimension")
    original_points = points
    original_energies = energies
    gradient_array = None
    if gradients_native is not None:
        gradient_array = np.asarray(gradients_native, dtype=float)
        if gradient_array.shape != points.shape:
            raise ValueError("ring-puckering gradients differ from the observation grid")
    symmetry_number = chart.symmetry_number if enforce_chart_symmetry else 1
    selected_radial_scale = (
        max(float(np.linalg.norm(reference_components)), 0.1)
        if radial_scale is None
        else float(radial_scale)
    )
    if selected_radial_scale <= 0.0:
        raise ValueError("ring-puckering radial scale must be positive")
    selected_confinement_beta = (
        1.0 / selected_radial_scale**2 if confinement_beta is None else float(confinement_beta)
    )
    if selected_confinement_beta <= 0.0:
        raise ValueError("inverse-Gaussian confinement beta must be positive")
    selected_inverse_orders = int(inverse_radial_order_count)
    if selected_inverse_orders < 1 or selected_inverse_orders > 6:
        raise ValueError("inverse-radial order count must be between one and six")
    symmetry_bases = {
        "CONFINING_PERIODIC_NORMALIZED_POLYNOMIAL",
        "INVERSE_RADIAL_SPHERICAL_HARMONICS",
    }
    augmentation_order = 1 if basis_kind in symmetry_bases else symmetry_number
    points, energies, gradient_array = _augment_ring_puckering_symmetry(
        points,
        energies,
        gradient_array,
        symmetry_number=augmentation_order,
    )
    design = np.asarray(
        [
            _surface_basis_values(
                row,
                chart,
                basis_kind=basis_kind,
                polynomial_degree=polynomial_degree,
                radial_scale=selected_radial_scale,
                confinement_beta=selected_confinement_beta,
                inverse_radial_order_count=selected_inverse_orders,
            )
            for row in points
        ],
        dtype=float,
    )
    if len(energies) != len(design):
        raise ValueError("ring-puckering fit has unpaired energy observations")
    derivative_design = None
    if gradient_array is not None:
        derivative_design = np.asarray(
            [
                np.column_stack(
                    [
                        jet.gradient
                        for jet in _surface_basis_jets(
                            row,
                            chart,
                            basis_kind=basis_kind,
                            polynomial_degree=polynomial_degree,
                            radial_scale=selected_radial_scale,
                            confinement_beta=selected_confinement_beta,
                            inverse_radial_order_count=selected_inverse_orders,
                        )
                    ]
                )
                for row in points
            ],
            dtype=float,
        )
        weight = float(gradient_observation_weight_radian)
        if weight <= 0.0:
            raise ValueError("ring-puckering gradient observation weight must be positive")
        fit_design = np.vstack((design, weight * derivative_design.reshape(-1, design.shape[1])))
        fit_target = np.concatenate((energies, weight * gradient_array.reshape(-1)))
    else:
        fit_design = design
        fit_target = energies
    if fit_design.shape[0] < fit_design.shape[1]:
        raise ValueError("ring-puckering fit has insufficient paired observations")
    if basis_kind == "INVERSE_RADIAL_SPHERICAL_HARMONICS":
        from scipy.optimize import lsq_linear

        lower = np.full(fit_design.shape[1], -np.inf)
        upper = np.full(fit_design.shape[1], np.inf)
        lower[0] = 0.0
        coefficients = lsq_linear(
            fit_design,
            fit_target,
            bounds=(lower, upper),
            lsmr_tol="auto",
        ).x
    else:
        coefficients, *_ = np.linalg.lstsq(fit_design, fit_target, rcond=1.0e-12)
    original_design = np.asarray(
        [
            _surface_basis_values(
                row,
                chart,
                basis_kind=basis_kind,
                polynomial_degree=polynomial_degree,
                radial_scale=selected_radial_scale,
                confinement_beta=selected_confinement_beta,
                inverse_radial_order_count=selected_inverse_orders,
            )
            for row in original_points
        ],
        dtype=float,
    )
    residual = original_design @ coefficients - original_energies
    original_derivative_design = (
        None
        if gradients_native is None
        else np.asarray(
            [
                np.column_stack(
                    [
                        jet.gradient
                        for jet in _surface_basis_jets(
                            row,
                            chart,
                            basis_kind=basis_kind,
                            polynomial_degree=polynomial_degree,
                            radial_scale=selected_radial_scale,
                            confinement_beta=selected_confinement_beta,
                            inverse_radial_order_count=selected_inverse_orders,
                        )
                    ]
                )
                for row in original_points
            ],
            dtype=float,
        )
    )
    original_gradient_array = (
        None if gradients_native is None else np.asarray(gradients_native, dtype=float)
    )
    gradient_rmse = (
        None
        if original_derivative_design is None or original_gradient_array is None
        else float(
            sqrt(
                float(
                    np.mean(
                        (original_derivative_design @ coefficients - original_gradient_array) ** 2
                    )
                )
            )
        )
    )
    return ZaffRingPuckeringSurface(
        chart=chart,
        reference_components=tuple(float(value) for value in reference_components),
        coefficients_hartree=tuple(float(value) for value in coefficients),
        fit_rmse_hartree=float(sqrt(float(np.mean(residual**2)))),
        maximum_residual_hartree=float(np.max(np.abs(residual))),
        fit_gradient_rmse_hartree_per_radian=gradient_rmse,
        gradient_observation_weight_radian=(
            float(gradient_observation_weight_radian) if gradients_native is not None else None
        ),
        basis_kind=basis_kind,
        polynomial_degree=polynomial_degree,
        enforced_symmetry_number=symmetry_number,
        radial_scale=selected_radial_scale,
        confinement_beta=selected_confinement_beta,
        inverse_radial_order_count=selected_inverse_orders,
    )


def _augment_ring_puckering_symmetry(
    points: np.ndarray,
    energies: np.ndarray,
    gradients: np.ndarray | None,
    *,
    symmetry_number: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate exact azimuthal symmetry partners in the native Cartesian chart."""

    order = max(1, int(symmetry_number))
    if order == 1:
        return points, energies, gradients
    point_orbits = []
    gradient_orbits = []
    for step in range(order):
        angle = 2.0 * pi * step / order
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        rotation = np.eye(points.shape[1])
        rotation[:2, :2] = ((cosine, -sine), (sine, cosine))
        point_orbits.append(points @ rotation.T)
        if gradients is not None:
            gradient_orbits.append(gradients @ rotation.T)
    return (
        np.vstack(point_orbits),
        np.tile(energies, order),
        None if gradients is None else np.vstack(gradient_orbits),
    )


def attach_ring_puckering_surfaces(
    force_field: Any,
    surfaces: Sequence[ZaffRingPuckeringSurface],
) -> Any:
    """Return a field carrying validated ring surfaces in its anharmonic model."""

    model = force_field.anharmonic_model
    if model is None:
        raise ValueError("ring-puckering surfaces require a ZAFF anharmonic model")
    coordinate_count = len(model.coordinates)
    frozen = tuple(surfaces)
    for surface in frozen:
        indices = surface.chart.source_coordinate_indices
        if any(index < 0 or index >= coordinate_count for index in indices):
            raise ValueError("ring-puckering surface refers to a missing SONIC coordinate")
        labels = tuple(
            {model.coordinates[index].identifier, model.coordinates[index].name}
            for index in indices
        )
        if any(
            name not in candidates
            for name, candidates in zip(surface.chart.source_coordinate_names, labels, strict=True)
        ):
            raise ValueError("ring-puckering surface does not match the frozen SONIC chart")
    return replace(force_field, anharmonic_model=replace(model, ring_puckering_surfaces=frozen))


def ring_puckering_charts_from_anharmonic_model(model: Any) -> tuple[ZaffRingPuckeringChart, ...]:
    coordinate_index = {
        label: coordinate.index
        for coordinate in model.coordinates
        for label in (coordinate.identifier, coordinate.name)
    }
    grouped: dict[tuple[int, ...], list[Any]] = {}
    for record in model.periodic_coordinates:
        if record.target == "RING_PUCKERING_PHASE":
            grouped.setdefault(record.ring_atoms, []).append(record)
    charts = []
    for ring, records in grouped.items():
        azimuth = next(item for item in records if item.coordinate_domain == "PERIODIC_2PI")
        polar = next((item for item in records if item.coordinate_domain == "BOUNDED_0_PI"), None)
        source_names = (
            tuple(reversed(azimuth.source_coordinates))
            if len(ring) == 5
            else tuple(polar.source_coordinates if polar is not None else ())
        )
        if len(source_names) not in {2, 3} or any(
            name not in coordinate_index for name in source_names
        ):
            raise ValueError("ring phase refers to a missing native RPck source")
        charts.append(
            ZaffRingPuckeringChart(
                ring_atoms=ring,
                source_coordinate_indices=tuple(coordinate_index[name] for name in source_names),
                source_coordinate_names=source_names,
                priority_atom=int(azimuth.priority_atom or ring[0]),
                azimuthal_periodicity=azimuth.periodicity,
                polar_periodicity=None if polar is None else polar.periodicity,
                symmetry_number=azimuth.symmetry_number,
                barrier_seed_kcal_mol=float(
                    np.mean([record.barrier_kcal_mol for record in records])
                ),
            )
        )
    return tuple(charts)


def _basis_values(values: np.ndarray, chart: ZaffRingPuckeringChart) -> np.ndarray:
    x, y = (float(values[0]), float(values[1]))
    q2 = x * x + y * y + (0.0 if chart.ring_size == 5 else float(values[2]) ** 2)
    real, imag = _complex_power(x, y, chart.azimuthal_periodicity)
    if chart.ring_size == 5:
        return np.asarray((1.0, q2, q2 * q2, real, imag), dtype=float)
    z = chart.chair_sign * float(values[2])
    return np.asarray(
        (1.0, q2, q2 * q2, z, z * z, z**3, z**4, real, imag, z * real, z * imag),
        dtype=float,
    )


def _surface_basis_values(
    values: np.ndarray,
    chart: ZaffRingPuckeringChart,
    *,
    basis_kind: str,
    polynomial_degree: int,
    radial_scale: float,
    confinement_beta: float,
    inverse_radial_order_count: int,
) -> np.ndarray:
    if basis_kind == "LEGACY_RING_INVARIANTS":
        return _basis_values(values, chart)
    if basis_kind == "CONFINING_PERIODIC_NORMALIZED_POLYNOMIAL":
        return np.asarray(
            [
                jet.value
                for jet in _confining_periodic_basis_jets(
                    values,
                    chart,
                    polynomial_degree=polynomial_degree,
                    radial_scale=radial_scale,
                )
            ],
            dtype=float,
        )
    if basis_kind == "INVERSE_RADIAL_SPHERICAL_HARMONICS":
        return np.asarray(
            [
                jet.value
                for jet in _inverse_radial_spherical_basis_jets(
                    values,
                    chart,
                    maximum_degree=polynomial_degree,
                    radial_scale=radial_scale,
                    confinement_beta=confinement_beta,
                    inverse_radial_order_count=inverse_radial_order_count,
                )
            ],
            dtype=float,
        )
    if basis_kind != "REGULAR_CARTESIAN_POLYNOMIAL":
        raise ValueError("unsupported ring-puckering surface basis")
    if polynomial_degree < 2 or polynomial_degree > 8:
        raise ValueError("ring-puckering polynomial degree must be between two and eight")
    powers = _cartesian_monomial_powers(len(values), polynomial_degree)
    return np.asarray(
        [float(np.prod(values ** np.asarray(exponents, dtype=int))) for exponents in powers],
        dtype=float,
    )


def _basis_jets(values: np.ndarray, chart: ZaffRingPuckeringChart) -> tuple["_Jet2", ...]:
    variables = tuple(
        _Jet2.variable(float(value), index, len(values)) for index, value in enumerate(values)
    )
    x, y = variables[:2]
    q2 = x * x + y * y
    if chart.ring_size == 6:
        q2 = q2 + variables[2] * variables[2]
    real, imag = _complex_power(x, y, chart.azimuthal_periodicity)
    if chart.ring_size == 5:
        return (_Jet2.constant(1.0, len(values)), q2, q2 * q2, real, imag)
    z = chart.chair_sign * variables[2]
    return (
        _Jet2.constant(1.0, len(values)),
        q2,
        q2 * q2,
        z,
        z * z,
        z**3,
        z**4,
        real,
        imag,
        z * real,
        z * imag,
    )


def _surface_basis_jets(
    values: np.ndarray,
    chart: ZaffRingPuckeringChart,
    *,
    basis_kind: str,
    polynomial_degree: int,
    radial_scale: float,
    confinement_beta: float,
    inverse_radial_order_count: int,
) -> tuple["_Jet2", ...]:
    if basis_kind == "LEGACY_RING_INVARIANTS":
        return _basis_jets(values, chart)
    if basis_kind == "CONFINING_PERIODIC_NORMALIZED_POLYNOMIAL":
        return _confining_periodic_basis_jets(
            values,
            chart,
            polynomial_degree=polynomial_degree,
            radial_scale=radial_scale,
        )
    if basis_kind == "INVERSE_RADIAL_SPHERICAL_HARMONICS":
        return _inverse_radial_spherical_basis_jets(
            values,
            chart,
            maximum_degree=polynomial_degree,
            radial_scale=radial_scale,
            confinement_beta=confinement_beta,
            inverse_radial_order_count=inverse_radial_order_count,
        )
    if basis_kind != "REGULAR_CARTESIAN_POLYNOMIAL":
        raise ValueError("unsupported ring-puckering surface basis")
    variables = tuple(
        _Jet2.variable(float(value), index, len(values)) for index, value in enumerate(values)
    )
    terms = []
    for exponents in _cartesian_monomial_powers(len(values), polynomial_degree):
        term = _Jet2.constant(1.0, len(values))
        for variable, exponent in zip(variables, exponents, strict=True):
            term = term * variable**exponent
        terms.append(term)
    return tuple(terms)


def _confining_periodic_basis_jets(
    values: np.ndarray,
    chart: ZaffRingPuckeringChart,
    *,
    polynomial_degree: int,
    radial_scale: float,
) -> tuple["_Jet2", ...]:
    """Return a pole-regular, symmetry-periodic basis with quartic confinement."""

    if polynomial_degree < 2 or polynomial_degree > 8:
        raise ValueError("ring-puckering polynomial degree must be between two and eight")
    size = len(values)
    variables = tuple(
        _Jet2.variable(float(value), index, size) for index, value in enumerate(values)
    )
    radial_squared = sum((variable * variable for variable in variables), _Jet2.constant(0.0, size))
    scaled_radius = _Jet2.constant(float(radial_scale) ** 2, size) + radial_squared
    inverse_scale = scaled_radius**-0.5
    bounded_radius = radial_squared * scaled_radius**-1.0
    normalized = tuple(variable * inverse_scale for variable in variables)
    terms = [_Jet2.constant(1.0, size), radial_squared, radial_squared * radial_squared]
    powers = _cartesian_monomial_powers(size, polynomial_degree)[1:]
    order = max(1, int(chart.symmetry_number))
    for exponents in powers:
        orbit = _Jet2.constant(0.0, size)
        for step in range(order):
            angle = 2.0 * pi * step / order
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            rotated = (
                cosine * normalized[0] - sine * normalized[1],
                sine * normalized[0] + cosine * normalized[1],
                *normalized[2:],
            )
            monomial = _Jet2.constant(1.0, size)
            for variable, exponent in zip(rotated, exponents, strict=True):
                monomial = monomial * variable**exponent
            orbit = orbit + monomial
        angular = (1.0 / order) * orbit
        radial_term = _Jet2.constant(1.0, size)
        for _ in range(4):
            terms.append(radial_term * angular)
            radial_term = radial_term * bounded_radius
    return tuple(terms)


def _inverse_radial_spherical_basis_jets(
    values: np.ndarray,
    chart: ZaffRingPuckeringChart,
    *,
    maximum_degree: int,
    radial_scale: float,
    confinement_beta: float,
    inverse_radial_order_count: int,
) -> tuple["_Jet2", ...]:
    """Regular solid harmonics times inverse-radial tails and inverse-Gaussian confinement."""

    if maximum_degree < 1 or maximum_degree > 8:
        raise ValueError("ring-puckering spherical-harmonic degree must be between one and eight")
    size = len(values)
    variables = tuple(
        _Jet2.variable(float(value), index, size) for index, value in enumerate(values)
    )
    zero = _Jet2.constant(0.0, size)
    radial_squared = sum((variable * variable for variable in variables), zero)
    scaled_radius = _Jet2.constant(float(radial_scale) ** 2, size) + radial_squared
    inverse_radius = scaled_radius**-0.5
    gaussian_argument = float(confinement_beta) * radial_squared
    inverse_gaussian_confinement = radial_squared * gaussian_argument.exp()
    terms = [inverse_gaussian_confinement]
    order = max(1, int(chart.symmetry_number))
    if size == 2:
        x, y = variables
        for angular_degree in range(maximum_degree + 1):
            if angular_degree % order:
                continue
            real, imaginary = _complex_power(x, y, angular_degree)
            angular_terms = (real,) if angular_degree == 0 else (real, imaginary)
            for angular in angular_terms:
                for radial_order in range(inverse_radial_order_count):
                    terms.append(angular * inverse_radius ** (angular_degree + 1 + radial_order))
        return tuple(terms)

    harmonics = _regular_solid_harmonics(variables, maximum_degree)
    for degree in range(maximum_degree + 1):
        for azimuthal_order in range(degree + 1):
            if azimuthal_order % order:
                continue
            real, imaginary = harmonics[(degree, azimuthal_order)]
            normalization = 1.0 / float(factorial(degree + azimuthal_order))
            angular_terms = (
                (normalization * real,)
                if azimuthal_order == 0
                else (normalization * real, normalization * imaginary)
            )
            for angular in angular_terms:
                for radial_order in range(inverse_radial_order_count):
                    terms.append(angular * inverse_radius ** (degree + 1 + radial_order))
    return tuple(terms)


def _regular_solid_harmonics(
    variables: tuple["_Jet2", ...],
    maximum_degree: int,
) -> dict[tuple[int, int], tuple["_Jet2", "_Jet2"]]:
    """Return unnormalized complex regular solid harmonics as real/imaginary jets."""

    x, y, z = variables
    size = len(variables)
    zero = _Jet2.constant(0.0, size)
    one = _Jet2.constant(1.0, size)
    radial_squared = x * x + y * y + z * z
    result: dict[tuple[int, int], tuple[_Jet2, _Jet2]] = {(0, 0): (one, zero)}
    for order in range(1, maximum_degree + 1):
        previous_real, previous_imaginary = result[(order - 1, order - 1)]
        factor = -float(2 * order - 1)
        result[(order, order)] = (
            factor * (x * previous_real - y * previous_imaginary),
            factor * (y * previous_real + x * previous_imaginary),
        )
    for order in range(maximum_degree + 1):
        if order < maximum_degree:
            real, imaginary = result[(order, order)]
            result[(order + 1, order)] = (
                float(2 * order + 1) * z * real,
                float(2 * order + 1) * z * imaginary,
            )
        for degree in range(order + 2, maximum_degree + 1):
            previous = result[(degree - 1, order)]
            earlier = result[(degree - 2, order)]
            denominator = float(degree - order)
            result[(degree, order)] = (
                (
                    float(2 * degree - 1) * z * previous[0]
                    - float(degree + order - 1) * radial_squared * earlier[0]
                )
                * (1.0 / denominator),
                (
                    float(2 * degree - 1) * z * previous[1]
                    - float(degree + order - 1) * radial_squared * earlier[1]
                )
                * (1.0 / denominator),
            )
    return result


def _cartesian_monomial_powers(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        exponents
        for total in range(int(degree) + 1)
        for exponents in product(range(total + 1), repeat=int(dimension))
        if sum(exponents) == total
    )


def _complex_power(x: Any, y: Any, power: int) -> tuple[Any, Any]:
    real, imag = 1.0, 0.0
    for _ in range(int(power)):
        real, imag = real * x - imag * y, real * y + imag * x
    return real, imag


class _Jet2:
    __slots__ = ("value", "gradient", "hessian")

    def __init__(self, value: float, gradient: np.ndarray, hessian: np.ndarray):
        self.value = float(value)
        self.gradient = np.asarray(gradient, dtype=float)
        self.hessian = np.asarray(hessian, dtype=float)

    @classmethod
    def constant(cls, value: float, size: int) -> "_Jet2":
        return cls(value, np.zeros(size), np.zeros((size, size)))

    @classmethod
    def variable(cls, value: float, index: int, size: int) -> "_Jet2":
        gradient = np.zeros(size)
        gradient[index] = 1.0
        return cls(value, gradient, np.zeros((size, size)))

    def _coerce(self, other: Any) -> "_Jet2":
        return (
            other if isinstance(other, _Jet2) else self.constant(float(other), len(self.gradient))
        )

    def __add__(self, other: Any) -> "_Jet2":
        rhs = self._coerce(other)
        return _Jet2(
            self.value + rhs.value, self.gradient + rhs.gradient, self.hessian + rhs.hessian
        )

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_Jet2":
        rhs = self._coerce(other)
        return _Jet2(
            self.value - rhs.value, self.gradient - rhs.gradient, self.hessian - rhs.hessian
        )

    def __rsub__(self, other: Any) -> "_Jet2":
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: Any) -> "_Jet2":
        rhs = self._coerce(other)
        return _Jet2(
            self.value * rhs.value,
            self.value * rhs.gradient + rhs.value * self.gradient,
            self.value * rhs.hessian
            + rhs.value * self.hessian
            + np.outer(self.gradient, rhs.gradient)
            + np.outer(rhs.gradient, self.gradient),
        )

    __rmul__ = __mul__

    def __pow__(self, exponent: float) -> "_Jet2":
        power = float(exponent)
        integer_power = int(power)
        if power == integer_power and integer_power >= 0:
            result = self.constant(1.0, len(self.gradient))
            for _ in range(integer_power):
                result = result * self
            return result
        if self.value <= 0.0:
            raise ValueError("fractional ring-puckering jet powers require a positive base")
        prefactor = power * self.value ** (power - 1.0)
        return _Jet2(
            self.value**power,
            prefactor * self.gradient,
            prefactor * self.hessian
            + power
            * (power - 1.0)
            * self.value ** (power - 2.0)
            * np.outer(self.gradient, self.gradient),
        )

    def exp(self) -> "_Jet2":
        value = float(np.exp(self.value))
        return _Jet2(
            value,
            value * self.gradient,
            value * (self.hessian + np.outer(self.gradient, self.gradient)),
        )
