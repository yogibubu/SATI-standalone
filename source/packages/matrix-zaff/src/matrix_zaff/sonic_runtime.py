"""Local-SONIC physical/Taylor E/G/H runtime for compiled ZAFF fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_smith import (
    build_gic_b_matrix,
    evaluate_gic_values,
    independent_zaff_analysis_definition,
)

from .bmatrix_derivative import build_sparse_gic_b_matrix_derivative
from .bonded import (
    CosineSeriesAngleShape,
    EvenOutOfPlanePolynomial,
    ScalarDerivatives,
    ZaffStretchStretchAnglePotential,
    ZaffTeamPlusValenceTorsionPotential,
)
from .ring_puckering import ZaffTeamPlusValenceRingPhasePotential
from .radial import (
    InversePowerStretchPotential,
    MorsePotential,
    SPFStretchPotential,
)
from .sonic_definition import build_zaff_runtime_sonic_definition
from .sonic_model import ZaffRuntimeSonicModel
from .truncation import ZaffTruncationPolicy, zaff_truncation_contract


@dataclass(frozen=True)
class ZaffSonicPolynomialValue:
    """Energy and gradient of the SONIC polynomial before Cartesian mapping."""

    energy_hartree: float
    gradient_internal: np.ndarray
    hessian_internal: np.ndarray


@dataclass(frozen=True)
class ZaffSonicEvaluation:
    """Analytic resident SONIC evaluation independent of ARCHITECT types."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray | None
    properties: tuple[str, ...]
    source: str
    execution: dict[str, Any]


@dataclass(frozen=True)
class ZaffPhysicalDiagonalTerm:
    """One physical diagonal kernel replacing its local Taylor polynomial."""

    coordinate_index: int
    functional_form: str
    reference_value: float
    quadratic: float
    cubic: float
    quartic: float
    potential: Any
    argument_offset: float
    reference_energy: float
    reference_first: float

    def correction(self, value: float) -> ScalarDerivatives:
        """Return physical minus k2/k3/k4 Taylor derivatives at ``value``."""

        displacement = float(value) - self.reference_value
        physical = _scalar_derivatives(
            self.potential.derivatives(float(value) + self.argument_offset)
        )
        taylor_energy = (
            0.5 * self.quadratic * displacement**2
            + self.cubic * displacement**3 / 6.0
            + self.quartic * displacement**4 / 24.0
        )
        taylor_first = (
            self.quadratic * displacement
            + 0.5 * self.cubic * displacement**2
            + self.quartic * displacement**3 / 6.0
        )
        taylor_second = (
            self.quadratic + self.cubic * displacement + 0.5 * self.quartic * displacement**2
        )
        return ScalarDerivatives(
            physical.value
            - self.reference_energy
            - self.reference_first * displacement
            - taylor_energy,
            physical.first - self.reference_first - taylor_first,
            physical.second - taylor_second,
        )


def evaluate_zaff_sonic_polynomial(
    displacement: np.ndarray,
    quadratic: np.ndarray,
    semidiagonal_cubic_i_j_j: np.ndarray,
    diagonal_quartic: np.ndarray,
) -> ZaffSonicPolynomialValue:
    r"""Evaluate harmonic, two-index cubic and diagonal-quartic SONIC terms.

    ``semidiagonal_cubic_i_j_j[i, j]`` is the symmetric Taylor derivative
    :math:`\Phi_{i j j}`.  For ``i != j`` its energy contribution is
    :math:`\Phi_{i j j} q_i q_j^2/2`; a diagonal element contributes
    :math:`\Phi_{i i i} q_i^3/6`.
    """

    q = np.asarray(displacement, dtype=float).reshape(-1)
    force = np.asarray(quadratic, dtype=float)
    cubic = np.asarray(semidiagonal_cubic_i_j_j, dtype=float)
    quartic = np.asarray(diagonal_quartic, dtype=float).reshape(-1)
    count = q.size
    if force.shape != (count, count) or cubic.shape != (count, count):
        raise ValueError("ZAFF SONIC derivative matrices have incompatible dimensions")
    if quartic.shape != (count,):
        raise ValueError("ZAFF diagonal quartic vector has an incompatible dimension")
    if not all(np.all(np.isfinite(values)) for values in (q, force, cubic, quartic)):
        raise ValueError("ZAFF SONIC polynomial contains non-finite values")

    symmetric_force = 0.5 * (force + force.T)
    gradient = symmetric_force @ q
    hessian = symmetric_force.copy()
    energy = 0.5 * float(q @ symmetric_force @ q)
    diagonal_cubic = np.diag(cubic)
    energy += float(np.dot(diagonal_cubic, q**3)) / 6.0
    gradient += diagonal_cubic * q**2 / 2.0
    hessian[np.diag_indices(count)] += diagonal_cubic * q

    off_diagonal_cubic = cubic.copy()
    off_diagonal_cubic[np.diag_indices(count)] = 0.0
    squared = q**2
    energy += 0.5 * float(q @ (off_diagonal_cubic @ squared))
    gradient += 0.5 * (off_diagonal_cubic @ squared)
    gradient += q * (off_diagonal_cubic.T @ q)
    mixed = off_diagonal_cubic * q[None, :]
    hessian += mixed + mixed.T
    hessian[np.diag_indices(count)] += off_diagonal_cubic.T @ q
    energy += float(np.dot(quartic, q**4)) / 24.0
    gradient += quartic * q**3 / 6.0
    hessian[np.diag_indices(count)] += quartic * q**2 / 2.0
    return ZaffSonicPolynomialValue(float(energy), gradient, hessian)


class ZaffSonicTaylorRuntime:
    """Evaluate physical diagonal kernels with a SONIC Taylor fallback."""

    def __init__(
        self,
        force_field: Any,
        xyzin: Path | str,
        *,
        truncation: ZaffTruncationPolicy | dict[str, Any] | str | None = None,
    ):
        raw_model = force_field.anharmonic_model
        model = (
            ZaffRuntimeSonicModel.from_dict(raw_model)
            if isinstance(raw_model, dict) or hasattr(raw_model, "keys")
            else raw_model
        )
        if model is None or model.status != "COMPILED":
            raise ValueError("ZAFF normal-mode projection requires a compiled anharmonic model")
        contract = dict(model.truncation_contract or zaff_truncation_contract())
        if contract.get("schema") != "matrix.zaff.truncation.v1":
            raise ValueError("compiled ZAFF model has an unsupported truncation contract")
        selected = contract.get("default_level", "FULL") if truncation is None else truncation
        self.truncation = ZaffTruncationPolicy.resolve(selected)
        if model.quadratic_matrix is None or model.semidiagonal_cubic_i_j_j is None:
            raise ValueError("compiled ZAFF model lacks SONIC quadratic or cubic derivatives")
        if len(model.terms) != len(model.coordinates):
            raise ValueError("compiled ZAFF model lacks one diagonal term per SONIC")

        field_definition, audit = build_zaff_runtime_sonic_definition(Path(xyzin))
        definition = independent_zaff_analysis_definition(field_definition)
        if len(definition.gics) != len(model.coordinates):
            raise ValueError("current SMITH SONIC definition differs from the ZAFF model")
        reference_values = np.asarray(
            evaluate_gic_values(
                definition,
                coordinates_angstrom=force_field.reference_coordinates_angstrom,
            ),
            dtype=float,
        )
        serialized_values = np.asarray(
            [coordinate.reference_value for coordinate in model.coordinates], dtype=float
        )
        if not np.allclose(reference_values, serialized_values, atol=2.0e-7, rtol=0.0):
            raise ValueError("current SONIC reference values differ from the ZAFF model")

        terms_by_index = {term.coordinate_index: term for term in model.terms}
        if set(terms_by_index) != set(range(len(model.coordinates))):
            raise ValueError("ZAFF diagonal terms do not cover the complete SONIC basis")
        self.force_field = force_field
        self.definition = definition
        self.definition_audit: dict[str, Any] = dict(audit)
        self.reference_values = reference_values
        self.linear = (
            np.zeros(len(model.coordinates), dtype=float)
            if model.linear_gradient_internal is None
            else np.asarray(model.linear_gradient_internal, dtype=float)
        )
        full_quadratic = 0.5 * (
            np.asarray(model.quadratic_matrix, dtype=float)
            + np.asarray(model.quadratic_matrix, dtype=float).T
        )
        coordinate_ids = tuple(coordinate.identifier for coordinate in model.coordinates)
        self.quadratic = _select_pair_terms(
            full_quadratic,
            coordinate_ids,
            self.truncation,
            include_all_offdiagonal=bool(self.truncation.include_offdiagonal_quadratic),
        )
        raw_cubic = np.asarray(model.semidiagonal_cubic_i_j_j, dtype=float)
        if int(self.truncation.maximum_diagonal_order or 2) < 3:
            self.cubic = np.zeros_like(raw_cubic)
        elif self.truncation.include_semidiagonal_cubic:
            self.cubic = _select_pair_terms(
                raw_cubic,
                coordinate_ids,
                self.truncation,
                include_all_offdiagonal=True,
                symmetric=False,
            )
        else:
            self.cubic = np.diag(np.diag(raw_cubic))
        raw_quartic = np.asarray(
            [terms_by_index[index].quartic for index in range(len(model.coordinates))],
            dtype=float,
        )
        self.diagonal_quartic = (
            raw_quartic
            if int(self.truncation.maximum_diagonal_order or 2) >= 4
            else np.zeros_like(raw_quartic)
        )
        periodic_labels = {
            label
            for record in model.periodic_coordinates
            if record.coordinate_domain == "PERIODIC_2PI"
            for label in (record.identifier, record.name)
        }
        self.periodic = np.asarray(
            [
                (coordinate.identifier in periodic_labels or coordinate.name in periodic_labels)
                if model.periodic_coordinates
                else any(token in coordinate.family.upper() for token in ("TORSION", "DIHEDRAL"))
                for coordinate in model.coordinates
            ],
            dtype=bool,
        )
        self.physical_diagonal_terms = tuple(
            physical
            for coordinate in model.coordinates
            if (
                physical := _build_physical_diagonal_term(
                    terms_by_index[coordinate.index],
                    coordinate.reference_value,
                )
            )
            is not None
        ) if self.truncation.include_physical_diagonal else ()
        coupled_terms = []
        for record in model.coupled_terms:
            if not self.truncation.allows_coupled_term(record):
                continue
            indices = dict(record["coordinate_indices_zero_based"])
            payload = dict(record["potential"])
            functional_form = str(payload.get("functional_form", ""))
            if functional_form == "CENTERED_MORSE_SI_SJ_COSINE_ANGLE":
                ordered = (
                    int(indices["stretch_i"]),
                    int(indices["stretch_j"]),
                    int(indices["angle"]),
                )
                potential = ZaffStretchStretchAnglePotential.from_dict(payload)
                expected = (
                    potential.stretch1.reference_distance,
                    potential.stretch2.reference_distance,
                    potential.theta0,
                )
            elif functional_form == "TEAM_PLUS_VALENCE_TORSION_COSINE":
                ordered = (int(indices["valence"]), int(indices["torsion"]))
                potential = ZaffTeamPlusValenceTorsionPotential.from_dict(payload)
                expected = (potential.reference_valence, potential.reference_torsion)
            elif functional_form == "TEAM_PLUS_VALENCE_RING_PHASES_POLE_SAFE":
                ring_components = tuple(int(index) for index in indices["ring_components"])
                if len(ring_components) != 3:
                    raise ValueError("TEAM+ ring phase requires three native RPck indices")
                ordered = (int(indices["valence"]), *ring_components)
                potential = ZaffTeamPlusValenceRingPhasePotential.from_dict(payload)
                expected = (potential.reference_valence, *potential.reference_components)
            else:
                raise ValueError(f"unsupported serialized ZAFF coupled term {functional_form!r}")
            if len(set(ordered)) != len(ordered) or any(
                index < 0 or index >= len(model.coordinates) for index in ordered
            ):
                raise ValueError("serialized ZAFF coupled term has invalid coordinate indices")
            actual = tuple(float(reference_values[index]) for index in ordered)
            differences = np.asarray(actual) - np.asarray(expected)
            if functional_form == "TEAM_PLUS_VALENCE_TORSION_COSINE":
                differences[-1] = (differences[-1] + np.pi) % (2.0 * np.pi) - np.pi
            if not np.allclose(differences, 0.0, atol=2.0e-7, rtol=0.0):
                raise ValueError("serialized ZAFF coupled term differs from its reference chart")
            coupled_terms.append((ordered, potential))
        self.coupled_terms = tuple(coupled_terms)
        self.ring_puckering_surfaces = (
            tuple(model.ring_puckering_surfaces)
            if self.truncation.include_large_amplitude_surfaces
            else ()
        )
        reference_b = np.asarray(
            build_gic_b_matrix(
                self.definition,
                coordinates_angstrom=force_field.reference_coordinates_angstrom,
            ).rows,
            dtype=float,
        )
        self.reference_b_matrix_per_angstrom = reference_b
        self.harmonic_cartesian_hessian_hartree_per_bohr2 = (
            reference_b.T @ self.quadratic @ reference_b
        ) * BOHR_TO_ANGSTROM**2

    def _evaluate_internal(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> tuple[np.ndarray, ZaffSonicPolynomialValue]:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != self.force_field.reference_coordinates_angstrom.shape:
            raise ValueError("ZAFF SONIC evaluation geometry has an incompatible shape")
        values = np.asarray(
            evaluate_gic_values(self.definition, coordinates_angstrom=coordinates), dtype=float
        )
        displacement = values - self.reference_values
        displacement[self.periodic] = (displacement[self.periodic] + np.pi) % (2.0 * np.pi) - np.pi
        internal = evaluate_zaff_sonic_polynomial(
            displacement,
            self.quadratic,
            self.cubic,
            self.diagonal_quartic,
        )
        internal_energy = internal.energy_hartree + float(np.dot(self.linear, displacement))
        internal_gradient = internal.gradient_internal.copy() + self.linear
        internal_hessian = internal.hessian_internal.copy()
        for physical in self.physical_diagonal_terms:
            correction = physical.correction(float(values[physical.coordinate_index]))
            internal_energy += correction.value
            internal_gradient[physical.coordinate_index] += correction.first
            internal_hessian[physical.coordinate_index, physical.coordinate_index] += (
                correction.second
            )
        for indices, potential in self.coupled_terms:
            contribution = potential.anharmonic_correction_derivatives(
                *(float(values[index]) for index in indices)
            )
            internal_energy += contribution.energy
            internal_gradient[np.asarray(indices)] += contribution.gradient
            internal_hessian[np.ix_(indices, indices)] += contribution.hessian
        for surface in self.ring_puckering_surfaces:
            indices = np.asarray(surface.chart.source_coordinate_indices, dtype=int)
            contribution = surface.correction(values[indices])
            internal_energy += contribution.energy_hartree
            internal_gradient[indices] += contribution.gradient_native
            internal_hessian[np.ix_(indices, indices)] += contribution.hessian_native
        internal = ZaffSonicPolynomialValue(
            float(internal_energy), internal_gradient, internal_hessian
        )
        return coordinates, internal

    def energy(self, coordinates_angstrom: np.ndarray) -> float:
        """Evaluate SONIC energy without building B or B-prime."""

        _coordinates, internal = self._evaluate_internal(coordinates_angstrom)
        return float(
            self.force_field.energy_reference_hartree + internal.energy_hartree
        )

    def evaluate(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        include_hessian: bool = False,
        b_derivative_workers: int = 0,
    ) -> ZaffSonicEvaluation:
        coordinates, internal = self._evaluate_internal(coordinates_angstrom)
        b_matrix = np.asarray(
            build_gic_b_matrix(self.definition, coordinates_angstrom=coordinates).rows,
            dtype=float,
        )
        gradient_bohr = BOHR_TO_ANGSTROM * (b_matrix.T @ internal.gradient_internal)
        hessian_bohr2 = None
        properties = ("energy", "gradient")
        if include_hessian:
            b_derivative = build_sparse_gic_b_matrix_derivative(
                self.definition,
                coordinates_angstrom=coordinates,
                parallel_workers=b_derivative_workers,
            )
            curvature_per_angstrom2 = b_derivative.contract_internal_gradient(
                internal.gradient_internal
            )
            hessian_per_angstrom2 = (
                b_matrix.T @ internal.hessian_internal @ b_matrix + curvature_per_angstrom2
            )
            hessian_bohr2 = hessian_per_angstrom2 * BOHR_TO_ANGSTROM**2
            properties = ("energy", "gradient", "hessian")
        physical_count = len(self.physical_diagonal_terms)
        coupled_count = len(self.coupled_terms)
        ring_surface_count = len(self.ring_puckering_surfaces)
        ring_phase_coupled_count = sum(
            isinstance(potential, ZaffTeamPlusValenceRingPhasePotential)
            for _indices, potential in self.coupled_terms
        )
        physical_runtime = bool(physical_count or coupled_count or ring_surface_count)
        return ZaffSonicEvaluation(
            energy_hartree=self.force_field.energy_reference_hartree + internal.energy_hartree,
            gradient_hartree_per_bohr=gradient_bohr,
            hessian_hartree_per_bohr2=hessian_bohr2,
            properties=properties,
            source=(
                "ARCHITECT/ZAFF_LOCAL_SONIC_PHYSICAL_HYBRID"
                if physical_runtime
                else "ARCHITECT/ZAFF_LOCAL_SONIC_TAYLOR"
            ),
            execution={
                "backend": "numpy",
                "device": "cpu",
                "precision": "float64",
                "accelerated": False,
                "cpu_validation": True,
                "runtime": (
                    "local_sonic_physical_hybrid" if physical_runtime else "local_sonic_taylor"
                ),
                "analytic_hessian": bool(include_hessian),
                "truncation": self.truncation.to_dict(),
                "physical_diagonal_term_count": physical_count,
                "coupled_term_count": coupled_count,
                "team_plus_ring_phase_term_count": ring_phase_coupled_count,
                "ring_puckering_surface_count": ring_surface_count,
                "taylor_diagonal_fallback_count": (len(self.reference_values) - physical_count),
            },
        )


def _build_physical_diagonal_term(
    term: Any,
    reference_value: float,
) -> ZaffPhysicalDiagonalTerm | None:
    if term.status != "COMPILED":
        return None
    parameters = dict(term.parameters)
    form = term.functional_form
    if form == "MORSE_QM_K2_K3":
        potential = MorsePotential(
            float(parameters["epsilon"]),
            float(parameters["r0"]),
            float(parameters["beta_dimensionless"]),
        )
    elif form == "INVERSE_POWER_STRETCH":
        potential = InversePowerStretchPotential(
            float(parameters["depth"]),
            float(parameters["r0"]),
            float(parameters["exponent"]),
        )
    elif form in {
        "SIMONS_PARR_FINLAN_INVERSE_DISTANCE",
        "SIMONS_PARR_FINLAN_FALLBACK",
    }:
        potential = SPFStretchPotential(
            float(parameters["r0"]),
            tuple(float(value) for value in parameters["coefficients"]),
        )
    elif form == "COSINE_SERIES_ANGLE_1_2_3":
        potential = CosineSeriesAngleShape(
            float(parameters["theta0"]),
            tuple(int(value) for value in parameters["harmonics"]),
            tuple(float(value) for value in parameters["coefficients"]),
        )
    elif form in {
        "EVEN_SIN_OUT_OF_PLANE_SINGLE_WELL",
        "EVEN_SIN_OUT_OF_PLANE_DOUBLE_WELL",
    }:
        potential = EvenOutOfPlanePolynomial(
            c0=float(parameters["c0"]),
            c2=float(parameters["c2"]),
            c4=float(parameters["c4"]),
            c6=float(parameters["c6"]),
        )
    else:
        return None
    reference = float(reference_value)
    argument_offset = -reference if form == "EVEN_SIN_OUT_OF_PLANE_SINGLE_WELL" else 0.0
    derivatives = _scalar_derivatives(potential.derivatives(reference + argument_offset))
    if abs(derivatives.second - float(term.quadratic)) > max(
        2.0e-7,
        2.0e-6 * abs(float(term.quadratic)),
    ):
        raise ValueError(
            f"physical ZAFF term {term.coordinate_identifier} does not preserve reference curvature"
        )
    return ZaffPhysicalDiagonalTerm(
        coordinate_index=int(term.coordinate_index),
        functional_form=form,
        reference_value=reference,
        quadratic=float(term.quadratic),
        cubic=float(term.cubic),
        quartic=float(term.quartic),
        potential=potential,
        argument_offset=argument_offset,
        reference_energy=derivatives.value,
        reference_first=derivatives.first,
    )


def _scalar_derivatives(values: Any) -> ScalarDerivatives:
    if hasattr(values, "value"):
        return ScalarDerivatives(
            float(values.value),
            float(values.first),
            float(values.second),
        )
    return ScalarDerivatives(
        float(values.energy),
        float(values.first),
        float(values.second),
    )


# Preferred public name; retain ZaffSonicTaylorRuntime for downstream callers
# written before physical diagonal kernels were activated.
ZaffSonicRuntime = ZaffSonicTaylorRuntime


def _select_pair_terms(
    values: np.ndarray,
    coordinate_identifiers: tuple[str, ...],
    policy: ZaffTruncationPolicy,
    *,
    include_all_offdiagonal: bool,
    symmetric: bool = True,
) -> np.ndarray:
    """Return a copied matrix with unselected inter-coordinate terms removed."""

    selected = np.asarray(values, dtype=float).copy()
    count = selected.shape[0]
    if selected.shape != (count, count) or len(coordinate_identifiers) != count:
        raise ValueError("ZAFF pair-term selection received incompatible dimensions")
    for first in range(count):
        for second in range(count):
            if first == second:
                continue
            keep = include_all_offdiagonal and policy.allows_coordinate_pair(
                coordinate_identifiers[first], coordinate_identifiers[second]
            )
            if not keep:
                selected[first, second] = 0.0
                if symmetric:
                    selected[second, first] = 0.0
    return selected
