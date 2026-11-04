"""Automatic SONIC derivative acquisition and ZAFF anharmonic compilation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
from math import factorial, isfinite
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from matrix_zaff.compatibility import normalize_legacy_zaff_payload

from matrix_link import hybrid_internal_coordinate_step, nonlinear_internal_coordinate_step
from matrix_oracle import write_oracle_sonic_contract_from_xyzin
from matrix_smith import (
    augment_shared_proton_coordinates,
    build_gic_b_matrix,
    build_gic_definition_from_xyzin,
    evaluate_gic_values,
    independent_zaff_analysis_definition,
)
from matrix_zaff.bonded import CosineSeriesAngleShape, EvenOutOfPlanePolynomial
from matrix_zaff.radial import (
    inverse_power_stretch_from_local_derivatives,
    spf_stretch_from_local_derivatives,
)
from matrix_zaff.ring_puckering import ZaffRingPuckeringSurface
from matrix_zaff.truncation import zaff_truncation_contract

from .torsion_scan import ZaffTorsionScanPlan, build_zaff_torsion_scan_plans


ZAFF_ANHARMONIC_SCHEMA = "matrix.zaff.anharmonic.v1"
ZAFF_ANHARMONIC_POINT_REQUEST_SCHEMA = "matrix.zaff.anharmonic.point_request.v1"
ZAFF_ANHARMONIC_POINT_RESULT_SCHEMA = "matrix.zaff.anharmonic.point_result.v1"
BOHR_TO_ANGSTROM = 0.529177210903


def zaff_refinement_policy() -> dict[str, Any]:
    """Normative point selection from the fitted ZAFF1 field through ZAFF3."""

    return {
        "schema": "matrix.zaff.refinement_policy.v1",
        "harmonic_invariant": (
            "COMPLETE_QM_MINUS_ELECTROSTATIC_MINUS_VDW_HESSIAN_FROM_ZAFF2_UPWARD"
        ),
        "level1_stretch_contract": (
            "MORSE_R0_REFITTED_FOR_TOTAL_STATIONARITY_K2_FROM_CORRECTED_QM_"
            "HESSIAN_DE_FROM_ZAFF_LIBRARY"
        ),
        "equilibrium_parameter_contract": (
            "R0_THETA0_AND_TORSION_PHASE_ARE_FITTED_CENTERS; "
            "TOTAL_GRADIENT_ZERO_AT_DECLARED_STATIONARITY_GEOMETRY; "
            "STATIONARITY_GEOMETRY_MAY_DIFFER_FROM_DERIVATIVE_GEOMETRY"
        ),
        "rounds": [
            {
                "name": "ZAFF1",
                "points": "NO_ADDITIONAL_QM_POINTS",
                "purpose": "PHYSICAL_DIAGONAL_PLUS_GEMINAL_XH_COUPLINGS",
            },
            {
                "name": "ZAFF2",
                "points": "NO_ADDITIONAL_QM_POINTS",
                "purpose": "COMPLETE_CORRECTED_QM_HARMONIC_FIELD_WITH_PHYSICAL_DIAGONAL_FORMS",
            },
            {
                "name": "ZAFF3_DIAGONAL",
                "points": "SINGLE_ANALYSIS_COORDINATE",
                "observables": ["ENERGY", "CARTESIAN_GRADIENT"],
                "purpose": "DIAGONAL_K3_K4_AND_GLOBAL_SHAPE",
            },
            {
                "name": "ZAFF3_COUPLINGS",
                "points": "PAIR_POINTS_ONLY_FOR_UNRESOLVED_TERMS",
                "observables": ["ENERGY", "CARTESIAN_GRADIENT"],
                "selection": (
                    "CORRECTED_HARMONIC_COUPLING_THEN_INDEPENDENT_ENERGY_"
                    "GRADIENT_RESIDUAL_OR_DECLARED_CHEMICAL_INTERACTION"
                ),
                "quartic_support": "UP_TO_THREE_DISTINCT_PRIMITIVES_BY_DEFAULT",
            },
            {
                "name": "ZAFF3_FULL_QFF",
                "points": "COMPLETE_REQUESTED_CUBIC_QUARTIC_SUPPORT",
                "optional": True,
            },
        ],
        "sam_selection": {
            "baseline": "FREEZE_COMPLETE_CORRECTED_ZAFF2_HESSIAN_AND_PHYSICAL_DIAGONAL_FORMS",
            "candidates": "NATIVE_LOCAL_FORMS_THEN_CONNECTED_COUPLINGS_BY_RESIDUAL_AND_RANK",
            "training": "SAME_GEOMETRY_ENERGY_GRADIENT_AND_HESSIAN_WHEN_AVAILABLE",
            "acceptance": "SMALLEST_RANK_SAFE_SUPPORT_PASSING_EVERY_INDEPENDENT_OBSERVABLE",
            "on_insufficient_information": "REQUEST_MOST_INFORMATIVE_SYMMETRY_DISTINCT_POINT",
        },
        "lam_scan": {
            "required": True,
            "periodic": "FULL_SYMMETRY_UNIQUE_CYCLE_FITTED_TO_FOURIER",
            "nonperiodic": "BOTH_SIDES_AND_TURNING_REGION_WITH_LOCAL_DERIVATIVES_FIXED",
            "coupling": "ADD_SAM_LAM_TERMS_ONLY_WHEN_SCAN_RESIDUAL_REQUIRES_THEM",
            "acceptance": "INDEPENDENT_SCAN_POINTS_AND_LOCAL_HESSIAN",
        },
        "convergence": (
            "ALL_AVAILABLE_INDEPENDENT_ENERGY_GRADIENT_HESSIAN_HOLDOUTS_"
            "AND_MANDATORY_LAM_SCAN_COVERAGE"
        ),
        "normal_mode_refinement": {
            "selection": "ONLY_IF_INDEPENDENT_DERIVATIVE_RESIDUAL_REQUIRES_INFORMATION",
            "points": "D_OPTIMAL_SYMMETRY_DISTINCT_POINTS_WITHIN_VALIDATED_LOCAL_DOMAIN",
            "observables": ["ENERGY", "CARTESIAN_GRADIENT", "HESSIAN_IF_AVAILABLE"],
            "purpose": "RESOLVE_IDENTIFIABLE_SAM_TERMS_WITHOUT_CHANGING_HARMONIC_FIELD",
        },
        "reflection_symmetric_torsion": {
            "points_degree": [0, 60, 120, 180],
            "additional_point": "REFERENCE_MINIMUM",
            "observables": ["ENERGY", "CARTESIAN_GRADIENT"],
            "fit": "EVEN_FOURIER_TO_FIFTH_HARMONIC_OR_SIXTH_WITH_REFERENCE_CURVATURE",
            "condition": "REFLECTION_SYMMETRY_VERIFIED_FOR_THIS_TORSION",
            "subtraction": "QM_MINUS_ELECTROSTATIC_MINUS_VDW_BEFORE_LOCAL_FIT",
        },
        "geometry_owner": "LINK",
        "coordinate_owner": "SMITH",
        "fit_owner": "ARCHITECT",
        "analysis_owner": "TRINITY",
    }


@dataclass(frozen=True)
class ZaffAnharmonicAcquisitionConfig:
    """Numerical and functional choices for the semidiagonal SONIC QFF."""

    stretch_step_angstrom: float = 0.01
    angular_step_radian: float = 0.02
    fragment_translation_step_angstrom: float = 0.02
    dimensionless_step: float = 0.02
    planar_probe_threshold_radian: float = 0.06
    planar_barrier_tolerance_hartree: float = 1.0e-8
    backtransform_tolerance: float = 2.0e-8
    parallel_workers: int = 0
    stretch_model: str = "spf"
    stretch_dissociation_depths: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        positive = (
            self.stretch_step_angstrom,
            self.angular_step_radian,
            self.fragment_translation_step_angstrom,
            self.dimensionless_step,
            self.planar_probe_threshold_radian,
            self.backtransform_tolerance,
        )
        if any(not isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("ZAFF anharmonic acquisition steps and tolerances must be positive")
        if self.planar_barrier_tolerance_hartree < 0.0:
            raise ValueError("planar barrier tolerance must be non-negative")
        if self.parallel_workers < 0:
            raise ValueError("parallel_workers must be non-negative")
        if self.stretch_model not in {"morse", "inverse_power", "spf"}:
            raise ValueError("stretch_model must be morse, inverse_power or spf")
        depths = tuple((str(key), float(value)) for key, value in self.stretch_dissociation_depths)
        if len({key for key, _value in depths}) != len(depths):
            raise ValueError("ZAFF stretch dissociation-depth identifiers must be unique")
        if any(not isfinite(value) or value <= 0.0 for _key, value in depths):
            raise ValueError("ZAFF stretch dissociation depths must be finite and positive")
        object.__setattr__(self, "stretch_dissociation_depths", depths)


@dataclass(frozen=True)
class ZaffAnharmonicCoordinate:
    index: int
    identifier: str
    name: str
    family: str
    irrep: str
    reference_value: float
    step: float
    unit: str
    primitive_components: tuple[tuple[str, float], ...]
    functional_form: str
    planar_probe: bool = False
    large_amplitude_role: str = "LOCAL_DIAGONAL"
    asymptotic_depth: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "identifier": self.identifier,
            "name": self.name,
            "family": self.family,
            "irrep": self.irrep,
            "reference_value": self.reference_value,
            "step": self.step,
            "unit": self.unit,
            "primitive_components": [
                {"identifier": identifier, "coefficient": coefficient}
                for identifier, coefficient in self.primitive_components
            ],
            "functional_form": self.functional_form,
            "planar_probe": self.planar_probe,
            "large_amplitude_role": self.large_amplitude_role,
            "asymptotic_depth": self.asymptotic_depth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffAnharmonicCoordinate":
        return cls(
            index=int(payload["index"]),
            identifier=str(payload["identifier"]),
            name=str(payload["name"]),
            family=str(payload["family"]),
            irrep=str(payload.get("irrep", "")),
            reference_value=float(payload["reference_value"]),
            step=float(payload["step"]),
            unit=str(payload["unit"]),
            primitive_components=tuple(
                (str(item["identifier"]), float(item["coefficient"]))
                for item in payload.get("primitive_components", ())
            ),
            functional_form=str(payload["functional_form"]),
            planar_probe=bool(payload.get("planar_probe", False)),
            large_amplitude_role=str(payload.get("large_amplitude_role", "LOCAL_DIAGONAL")),
            asymptotic_depth=(
                None
                if payload.get("asymptotic_depth") is None
                else float(payload["asymptotic_depth"])
            ),
        )


@dataclass(frozen=True)
class ZaffPeriodicCoordinate:
    """ARCHITECT copy of one immutable SMITH periodic-coordinate record."""

    identifier: str
    name: str
    family: str
    target: str
    definition: str
    periodicity: int
    coordinate_domain: str
    reference_value_radian: float | None
    reference_value_status: str
    symmetry_number: int
    priority_atom: int | None
    barrier_kcal_mol: float
    barrier_cm1: float
    source_coordinates: tuple[str, ...] = ()
    ring_atoms: tuple[int, ...] = ()
    central_bonds: tuple[tuple[int, int], ...] = ()
    periodicity_source: str = ""
    barrier_source: str = ""
    status: str = "SEED_REQUIRES_HESSIAN_OR_SCAN_VALIDATION"

    def __post_init__(self) -> None:
        if not self.identifier or self.periodicity <= 0:
            raise ValueError("ZAFF periodic coordinate requires an identifier and period")
        if self.coordinate_domain not in {"PERIODIC_2PI", "BOUNDED_0_PI"}:
            raise ValueError("unsupported ZAFF periodic-coordinate domain")
        if self.symmetry_number <= 0:
            raise ValueError("ZAFF periodic-coordinate symmetry number must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "family": self.family,
            "target": self.target,
            "definition": self.definition,
            "periodicity": self.periodicity,
            "coordinate_domain": self.coordinate_domain,
            "reference_value_radian": self.reference_value_radian,
            "reference_value_degree": (
                None
                if self.reference_value_radian is None
                else float(np.degrees(self.reference_value_radian))
            ),
            "reference_value_status": self.reference_value_status,
            "symmetry_number": self.symmetry_number,
            "priority_atom": self.priority_atom,
            "barrier_kcal_mol": self.barrier_kcal_mol,
            "barrier_cm1": self.barrier_cm1,
            "source_coordinates": list(self.source_coordinates),
            "ring_atoms": list(self.ring_atoms),
            "central_bonds": [list(pair) for pair in self.central_bonds],
            "periodicity_source": self.periodicity_source,
            "barrier_source": self.barrier_source,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffPeriodicCoordinate":
        return cls(
            identifier=str(payload["identifier"]),
            name=str(payload.get("name", payload["identifier"])),
            family=str(payload["family"]),
            target=str(payload["target"]),
            definition=str(payload.get("definition", "FROZEN_GIC")),
            periodicity=int(payload["periodicity"]),
            coordinate_domain=str(payload.get("coordinate_domain", "PERIODIC_2PI")),
            reference_value_radian=(
                None
                if payload.get("reference_value_radian") is None
                else float(payload["reference_value_radian"])
            ),
            reference_value_status=str(payload.get("reference_value_status", "DEFINED")),
            symmetry_number=int(payload.get("symmetry_number", 1)),
            priority_atom=(
                None if payload.get("priority_atom") is None else int(payload["priority_atom"])
            ),
            barrier_kcal_mol=float(payload["barrier_kcal_mol"]),
            barrier_cm1=float(payload["barrier_cm1"]),
            source_coordinates=tuple(str(value) for value in payload.get("source_coordinates", ())),
            ring_atoms=tuple(int(value) for value in payload.get("ring_atoms", ())),
            central_bonds=tuple(
                (int(pair[0]), int(pair[1])) for pair in payload.get("central_bonds", ())
            ),
            periodicity_source=str(payload.get("periodicity_source", "")),
            barrier_source=str(payload.get("barrier_source", "")),
            status=str(payload.get("status", "SEED_REQUIRES_HESSIAN_OR_SCAN_VALIDATION")),
        )


@dataclass(frozen=True)
class ZaffAnharmonicPointRequest:
    point_id: str
    purpose: str
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    coordinate_index: int | None = None
    coordinate_identifier: str = ""
    offset_multiplier: int = 0
    requested_coordinate_value: float | None = None
    achieved_coordinate_value: float | None = None
    operation: str = "SINGLE_POINT_GRADIENT"
    constraints: tuple[Mapping[str, Any], ...] = ()
    schema: str = ZAFF_ANHARMONIC_POINT_REQUEST_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "point_id": self.point_id,
            "purpose": self.purpose,
            "atoms": list(self.atoms),
            "coordinates_angstrom": np.asarray(self.coordinates_angstrom, dtype=float).tolist(),
            "coordinate_index": self.coordinate_index,
            "coordinate_identifier": self.coordinate_identifier,
            "offset_multiplier": self.offset_multiplier,
            "requested_coordinate_value": self.requested_coordinate_value,
            "achieved_coordinate_value": self.achieved_coordinate_value,
            "operation": self.operation,
            "requested_properties": ["energy", "gradient"],
            "constraints": [dict(item) for item in self.constraints],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffAnharmonicPointRequest":
        if payload.get("schema") != ZAFF_ANHARMONIC_POINT_REQUEST_SCHEMA:
            raise ValueError("unsupported ZAFF anharmonic point-request schema")
        return cls(
            point_id=str(payload["point_id"]),
            purpose=str(payload["purpose"]),
            atoms=tuple(str(atom) for atom in payload["atoms"]),
            coordinates_angstrom=np.asarray(payload["coordinates_angstrom"], dtype=float),
            coordinate_index=(
                None
                if payload.get("coordinate_index") is None
                else int(payload["coordinate_index"])
            ),
            coordinate_identifier=str(payload.get("coordinate_identifier", "")),
            offset_multiplier=int(payload.get("offset_multiplier", 0)),
            requested_coordinate_value=(
                None
                if payload.get("requested_coordinate_value") is None
                else float(payload["requested_coordinate_value"])
            ),
            achieved_coordinate_value=(
                None
                if payload.get("achieved_coordinate_value") is None
                else float(payload["achieved_coordinate_value"])
            ),
            operation=str(payload.get("operation", "SINGLE_POINT_GRADIENT")),
            constraints=tuple(dict(item) for item in payload.get("constraints", ())),
        )


@dataclass(frozen=True)
class ZaffAnharmonicPointResult:
    point_id: str
    energy_hartree: float
    cartesian_gradient_hartree_per_bohr: np.ndarray
    coordinates_angstrom: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ZAFF_ANHARMONIC_POINT_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "point_id": self.point_id,
            "energy_hartree": self.energy_hartree,
            "cartesian_gradient_hartree_per_bohr": np.asarray(
                self.cartesian_gradient_hartree_per_bohr, dtype=float
            )
            .reshape(-1)
            .tolist(),
            "coordinates_angstrom": (
                None
                if self.coordinates_angstrom is None
                else np.asarray(self.coordinates_angstrom, dtype=float).tolist()
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffAnharmonicPointResult":
        if payload.get("schema") != ZAFF_ANHARMONIC_POINT_RESULT_SCHEMA:
            raise ValueError("unsupported ZAFF anharmonic point-result schema")
        return cls(
            point_id=str(payload["point_id"]),
            energy_hartree=float(payload["energy_hartree"]),
            cartesian_gradient_hartree_per_bohr=np.asarray(
                payload["cartesian_gradient_hartree_per_bohr"], dtype=float
            ),
            coordinates_angstrom=(
                None
                if payload.get("coordinates_angstrom") is None
                else np.asarray(payload["coordinates_angstrom"], dtype=float)
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ZaffAnharmonicTerm:
    coordinate_index: int
    coordinate_identifier: str
    family: str
    functional_form: str
    quadratic: float
    cubic: float
    quartic: float
    parameters: Mapping[str, Any]
    acquisition_point_ids: tuple[str, ...]
    status: str = "COMPILED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_index": self.coordinate_index,
            "coordinate_identifier": self.coordinate_identifier,
            "family": self.family,
            "functional_form": self.functional_form,
            "derivatives": {
                "quadratic": self.quadratic,
                "cubic": self.cubic,
                "quartic": self.quartic,
            },
            "parameters": dict(self.parameters),
            "acquisition_point_ids": list(self.acquisition_point_ids),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffAnharmonicTerm":
        derivatives = payload["derivatives"]
        return cls(
            coordinate_index=int(payload["coordinate_index"]),
            coordinate_identifier=str(payload["coordinate_identifier"]),
            family=str(payload["family"]),
            functional_form=str(payload["functional_form"]),
            quadratic=float(derivatives["quadratic"]),
            cubic=float(derivatives["cubic"]),
            quartic=float(derivatives["quartic"]),
            parameters=dict(payload.get("parameters", {})),
            acquisition_point_ids=tuple(
                str(value) for value in payload.get("acquisition_point_ids", ())
            ),
            status=str(payload.get("status", "COMPILED")),
        )


@dataclass(frozen=True)
class ZaffAnharmonicModel:
    coordinates: tuple[ZaffAnharmonicCoordinate, ...]
    periodic_coordinates: tuple[ZaffPeriodicCoordinate, ...] = ()
    requests: tuple[ZaffAnharmonicPointRequest, ...] = ()
    results: tuple[ZaffAnharmonicPointResult, ...] = ()
    terms: tuple[ZaffAnharmonicTerm, ...] = ()
    linear_gradient_internal: np.ndarray | None = None
    quadratic_matrix: np.ndarray | None = None
    quadratic_matrix_directed: np.ndarray | None = None
    harmonic_reciprocity_audit: Mapping[str, Any] = field(default_factory=dict)
    semidiagonal_cubic_i_j_j: np.ndarray | None = None
    coupled_terms: tuple[Mapping[str, Any], ...] = ()
    ring_puckering_surfaces: tuple[ZaffRingPuckeringSurface, ...] = ()
    torsion_scan_plans: tuple[ZaffTorsionScanPlan, ...] = ()
    truncation_contract: Mapping[str, Any] = field(default_factory=zaff_truncation_contract)
    redundant_field_contract: Mapping[str, Any] = field(default_factory=dict)
    refinement_policy: Mapping[str, Any] = field(default_factory=zaff_refinement_policy)
    status: str = "PLANNED"
    basis: str = "SONIC_LOCAL"
    stencil: str = "CENTERED_GRADIENTS_0_PLUS_MINUS_H_PLUS_MINUS_2H"
    planar_policy: str = "CONSTRAINED_RELAXATION_AT_SIGNED_OUT_OF_PLANE_ZERO"
    schema: str = ZAFF_ANHARMONIC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ZAFF_ANHARMONIC_SCHEMA:
            raise ValueError("unsupported ZAFF anharmonic schema")
        if tuple(item.index for item in self.coordinates) != tuple(range(len(self.coordinates))):
            raise ValueError("ZAFF anharmonic coordinates must use contiguous canonical indices")
        periodic_ids = tuple(item.identifier for item in self.periodic_coordinates)
        if len(set(periodic_ids)) != len(periodic_ids):
            raise ValueError("ZAFF periodic-coordinate identifiers must be unique")
        count = len(self.coordinates)
        if self.linear_gradient_internal is not None:
            linear = np.asarray(self.linear_gradient_internal, dtype=float).reshape(-1)
            if linear.shape != (count,) or not np.all(np.isfinite(linear)):
                raise ValueError("ZAFF anharmonic internal linear gradient has an invalid shape")
            object.__setattr__(self, "linear_gradient_internal", linear)
        for label, matrix in (
            ("quadratic", self.quadratic_matrix),
            ("directed quadratic", self.quadratic_matrix_directed),
            ("semidiagonal cubic", self.semidiagonal_cubic_i_j_j),
        ):
            if matrix is None:
                continue
            values = np.asarray(matrix, dtype=float)
            if values.shape != (count, count) or not np.all(np.isfinite(values)):
                raise ValueError(f"ZAFF anharmonic {label} matrix has an invalid shape")
            object.__setattr__(
                self,
                {
                    "quadratic": "quadratic_matrix",
                    "directed quadratic": "quadratic_matrix_directed",
                    "semidiagonal cubic": "semidiagonal_cubic_i_j_j",
                }[label],
                values,
            )
        request_ids = tuple(item.point_id for item in self.requests)
        result_ids = tuple(item.point_id for item in self.results)
        if len(set(request_ids)) != len(request_ids) or len(set(result_ids)) != len(result_ids):
            raise ValueError("ZAFF anharmonic point identifiers must be unique")
        if not set(result_ids).issubset(request_ids):
            raise ValueError("ZAFF anharmonic results contain an unknown point identifier")
        if any(item.get("schema") != "matrix.zaff.coupled_term.v1" for item in self.coupled_terms):
            raise ValueError("ZAFF anharmonic model contains an unsupported coupled term")
        if self.truncation_contract.get("schema") != "matrix.zaff.truncation.v1":
            raise ValueError("ZAFF anharmonic model contains an unsupported truncation contract")
        if self.refinement_policy.get("schema") != "matrix.zaff.refinement_policy.v1":
            raise ValueError("ZAFF anharmonic model contains an unsupported refinement policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "basis": self.basis,
            "status": self.status,
            "stencil": self.stencil,
            "planar_policy": self.planar_policy,
            "coordinates": [item.to_dict() for item in self.coordinates],
            "periodic_coordinates": [item.to_dict() for item in self.periodic_coordinates],
            "requests": [item.to_dict() for item in self.requests],
            "results": [item.to_dict() for item in self.results],
            "terms": [item.to_dict() for item in self.terms],
            "linear_gradient_internal": (
                None
                if self.linear_gradient_internal is None
                else np.asarray(self.linear_gradient_internal, dtype=float).tolist()
            ),
            "quadratic_matrix": (
                None
                if self.quadratic_matrix is None
                else np.asarray(self.quadratic_matrix, dtype=float).tolist()
            ),
            "quadratic_matrix_directed": (
                None
                if self.quadratic_matrix_directed is None
                else np.asarray(self.quadratic_matrix_directed, dtype=float).tolist()
            ),
            "harmonic_reciprocity_audit": dict(self.harmonic_reciprocity_audit),
            "semidiagonal_cubic_i_j_j": (
                None
                if self.semidiagonal_cubic_i_j_j is None
                else np.asarray(self.semidiagonal_cubic_i_j_j, dtype=float).tolist()
            ),
            "coupled_terms": [dict(item) for item in self.coupled_terms],
            "ring_puckering_surfaces": [item.to_dict() for item in self.ring_puckering_surfaces],
            "torsion_scan_plans": [item.to_dict() for item in self.torsion_scan_plans],
            "truncation_contract": dict(self.truncation_contract),
            "redundant_field_contract": dict(self.redundant_field_contract),
            "refinement_policy": dict(self.refinement_policy),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffAnharmonicModel":
        if payload.get("schema") != ZAFF_ANHARMONIC_SCHEMA:
            raise ValueError("unsupported ZAFF anharmonic schema")
        return cls(
            coordinates=tuple(
                ZaffAnharmonicCoordinate.from_dict(item) for item in payload.get("coordinates", ())
            ),
            periodic_coordinates=tuple(
                ZaffPeriodicCoordinate.from_dict(item)
                for item in payload.get("periodic_coordinates", ())
            ),
            requests=tuple(
                ZaffAnharmonicPointRequest.from_dict(item) for item in payload.get("requests", ())
            ),
            results=tuple(
                ZaffAnharmonicPointResult.from_dict(item) for item in payload.get("results", ())
            ),
            terms=tuple(ZaffAnharmonicTerm.from_dict(item) for item in payload.get("terms", ())),
            linear_gradient_internal=(
                None
                if payload.get("linear_gradient_internal") is None
                else np.asarray(payload["linear_gradient_internal"], dtype=float)
            ),
            quadratic_matrix=(
                None
                if payload.get("quadratic_matrix") is None
                else np.asarray(payload["quadratic_matrix"], dtype=float)
            ),
            quadratic_matrix_directed=(
                None
                if payload.get("quadratic_matrix_directed") is None
                else np.asarray(payload["quadratic_matrix_directed"], dtype=float)
            ),
            harmonic_reciprocity_audit=dict(payload.get("harmonic_reciprocity_audit", {})),
            semidiagonal_cubic_i_j_j=(
                None
                if payload.get("semidiagonal_cubic_i_j_j") is None
                else np.asarray(payload["semidiagonal_cubic_i_j_j"], dtype=float)
            ),
            coupled_terms=tuple(dict(item) for item in payload.get("coupled_terms", ())),
            ring_puckering_surfaces=tuple(
                ZaffRingPuckeringSurface.from_dict(item)
                for item in payload.get("ring_puckering_surfaces", ())
            ),
            torsion_scan_plans=tuple(
                ZaffTorsionScanPlan.from_dict(dict(item))
                for item in payload.get("torsion_scan_plans", ())
            ),
            truncation_contract=dict(
                payload.get("truncation_contract", zaff_truncation_contract())
            ),
            redundant_field_contract=dict(payload.get("redundant_field_contract", {})),
            refinement_policy=dict(payload.get("refinement_policy", zaff_refinement_policy())),
            status=str(payload.get("status", "PLANNED")),
            basis=str(payload.get("basis", "SONIC_LOCAL")),
            stencil=str(payload.get("stencil", "CENTERED_GRADIENTS_0_PLUS_MINUS_H_PLUS_MINUS_2H")),
            planar_policy=str(
                payload.get(
                    "planar_policy",
                    "CONSTRAINED_RELAXATION_AT_SIGNED_OUT_OF_PLANE_ZERO",
                )
            ),
        )


class ZaffAnharmonicEvaluator(Protocol):
    def __call__(self, request: ZaffAnharmonicPointRequest) -> ZaffAnharmonicPointResult: ...


def build_zaff_anharmonic_plan(
    definition: Any,
    coordinates_angstrom: np.ndarray,
    *,
    config: ZaffAnharmonicAcquisitionConfig = ZaffAnharmonicAcquisitionConfig(),
    oracle_soft_torsions: Sequence[Mapping[str, Any]] = (),
    high_barrier_torsion_axes: Sequence[Sequence[int]] = (),
    low_curvature_coordinate_ids: Sequence[str] = (),
    corrected_torsion_force_constants: Mapping[str, float] | None = None,
    torsion_force_constant_cutoff: float | None = None,
) -> ZaffAnharmonicModel:
    """Create the SONIC plan with ORACLE rotor and harmonic-softness evidence.

    ORACLE torsion atom indices are zero-based; SMITH primitive atom indices
    are one-based. A torsion without either indication remains a candidate,
    rather than imposing an unmotivated large-amplitude scan. If corrected
    torsion curvatures are supplied, their cutoff is declared by the caller
    in hartree/radian squared; the planner never invents one.
    """

    if (corrected_torsion_force_constants is None) != (torsion_force_constant_cutoff is None):
        raise ValueError("corrected torsion curvatures and their cutoff must be supplied together")
    if torsion_force_constant_cutoff is not None and (
        not np.isfinite(torsion_force_constant_cutoff) or torsion_force_constant_cutoff <= 0
    ):
        raise ValueError("torsion force-constant cutoff must be finite and positive")

    reference = evaluate_gic_values(definition, coordinates_angstrom=coordinates_angstrom)
    periodic_coordinates = tuple(
        _zaff_periodic_coordinate(record)
        for record in getattr(definition, "periodic_coordinate_estimates", ())
    )
    periodic_labels = {
        label for record in periodic_coordinates for label in (record.identifier, record.name)
    }
    coordinates: list[ZaffAnharmonicCoordinate] = []
    stretch_depths = dict(config.stretch_dissociation_depths)
    primitive_by_id = {
        str(primitive.identifier): primitive for primitive in definition.primitives
    }
    soft_axes = {
        frozenset(int(atom) + 1 for atom in record["central_bond"])
        for record in oracle_soft_torsions
        if bool(record.get("soft", False))
        and len(record.get("central_bond", ())) == 2
    }
    hard_axes = {
        frozenset(int(atom) + 1 for atom in axis)
        for axis in high_barrier_torsion_axes
        if len(axis) == 2
    }
    low_curvature_ids = set(map(str, low_curvature_coordinate_ids))
    if corrected_torsion_force_constants is not None:
        for identifier, curvature in corrected_torsion_force_constants.items():
            if not np.isfinite(curvature):
                raise ValueError("corrected torsion curvatures must be finite")
            if curvature < torsion_force_constant_cutoff:
                low_curvature_ids.add(str(identifier))
    for index, (gic, value) in enumerate(zip(definition.gics, reference, strict=True)):
        family = str(gic.family).upper()
        components = tuple(gic.coefficients or ((gic.primitive_id, 1.0),))
        primitive_functions = tuple(
            str(primitive_by_id[identifier].function).upper()
            for identifier, _coefficient in components
            if identifier in primitive_by_id
        )
        unit, step = _coordinate_unit_and_step(
            family,
            config,
            primitive_functions=primitive_functions,
        )
        coordinates.append(
            ZaffAnharmonicCoordinate(
                index=index,
                identifier=str(gic.identifier),
                name=str(gic.name),
                family=family,
                irrep=str(gic.irrep),
                reference_value=float(value),
                step=step,
                unit=unit,
                primitive_components=tuple(
                    (str(identifier), float(coefficient)) for identifier, coefficient in components
                ),
                functional_form=_planned_functional_form(
                    family,
                    config.stretch_model,
                    primitive_component_count=len(components),
                    reference_value=float(value),
                ),
                planar_probe=(
                    _is_out_of_plane_family(family)
                    and abs(float(value)) >= config.planar_probe_threshold_radian
                ),
                large_amplitude_role=_large_amplitude_role(
                    family,
                    labels=(str(gic.identifier), str(gic.name)),
                    periodic_labels=periodic_labels,
                    oracle_soft_axis=any(
                        len(primitive_by_id[identifier].atoms) == 4
                        and frozenset(primitive_by_id[identifier].atoms[1:3]) in soft_axes
                        for identifier, _coefficient in components
                        if identifier in primitive_by_id
                    ),
                    high_barrier=any(
                        len(primitive_by_id[identifier].atoms) == 4
                        and frozenset(primitive_by_id[identifier].atoms[1:3]) in hard_axes
                        for identifier, _coefficient in components
                        if identifier in primitive_by_id
                    ),
                    low_curvature=str(gic.identifier) in low_curvature_ids,
                ),
                asymptotic_depth=(
                    stretch_depths.get(str(components[0][0]), stretch_depths.get(str(gic.identifier)))
                    if len(components) == 1 and _is_stretch_family(family)
                    else None
                ),
            )
        )
    frozen_coordinates = tuple(coordinates)
    return ZaffAnharmonicModel(
        coordinates=frozen_coordinates,
        periodic_coordinates=periodic_coordinates,
        torsion_scan_plans=build_zaff_torsion_scan_plans(
            periodic_coordinates,
            frozen_coordinates,
        ),
    )


def redundant_zaff_field_contract(
    field_definition: Any,
    analysis_definition: Any,
    construction_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize the redundant force-field chart and its independent view."""

    return {
        "schema": "matrix.zaff.redundant_field.v1",
        "policy": "RETAIN_PHYSICAL_PRIMITIVES_AND_ADD_DERIVED_TRANSFER_COORDINATES",
        "coordinate_count": len(field_definition.gics),
        "tangent_rank": int(field_definition.target_rank),
        "redundancy": len(field_definition.gics) - int(field_definition.target_rank),
        "analysis_coordinate_identifiers": [
            str(gic.identifier) for gic in analysis_definition.gics
        ],
        "coordinates": [
            {
                "identifier": str(gic.identifier),
                "name": str(gic.name),
                "family": str(gic.family),
                "primitive_components": [
                    [str(identifier), float(coefficient)]
                    for identifier, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                ],
            }
            for gic in field_definition.gics
        ],
        "construction_audit": dict(construction_audit),
    }


def _zaff_periodic_coordinate(record: Any) -> ZaffPeriodicCoordinate:
    return ZaffPeriodicCoordinate(
        identifier=str(record.coordinate_identifier),
        name=str(record.coordinate_name),
        family=str(record.family),
        target=str(record.target),
        definition=str(record.coordinate_definition),
        periodicity=int(record.periodicity),
        coordinate_domain=str(record.coordinate_domain),
        reference_value_radian=(
            None if record.reference_value_radian is None else float(record.reference_value_radian)
        ),
        reference_value_status=str(record.reference_value_status),
        symmetry_number=int(record.symmetry_number),
        priority_atom=(None if record.priority_atom is None else int(record.priority_atom)),
        barrier_kcal_mol=float(record.barrier_kcal_mol),
        barrier_cm1=float(record.barrier_cm1),
        source_coordinates=tuple(str(value) for value in record.source_coordinates),
        ring_atoms=tuple(int(value) for value in record.ring_atoms),
        central_bonds=tuple((int(left), int(right)) for left, right in record.central_bonds),
        periodicity_source=str(record.periodicity_source),
        barrier_source=str(record.barrier_source),
        status=str(record.status),
    )


def _large_amplitude_role(
    family: str,
    *,
    labels: tuple[str, str],
    periodic_labels: set[str],
    oracle_soft_axis: bool = False,
    high_barrier: bool = False,
    low_curvature: bool = False,
) -> str:
    name = str(family).upper()
    if name == "RING_PUCKER_COMPONENT":
        return "SOURCE_FOR_DERIVED_RING_PHASE"
    if name == "PROTON_TRANSFER":
        return "NONPERIODIC_SCAN"
    if name == "CYCLIC_BEND":
        return "SMALL_AMPLITUDE_NONPERIODIC"
    if name in {"TORSION", "IMPROPER_DIHEDRAL", "CYCLIC_TORSION"}:
        if high_barrier:
            return "SMALL_AMPLITUDE_NONPERIODIC"
        return "PERIODIC_SCAN" if oracle_soft_axis or low_curvature else "PERIODIC_CANDIDATE"
    if any(label in periodic_labels for label in labels):
        return "PERIODIC_SCAN"
    return "LOCAL_DIAGONAL"


def build_zaff_local_sonic_definition(
    xyzin: Path | str,
    *,
    minimum_relative_singular_value: float = 1.0e-7,
    ring_puckering_model: str = "charm",
) -> tuple[Any, Mapping[str, Any]]:
    """Build ZAFF SONICs and replace a singular local-SALC chart deterministically.

    At an exactly symmetric pyramidal center, a naive totally symmetric angle
    SALC can become first-order parallel to the native inversion coordinate.
    The fallback keeps the native out-of-plane coordinate and obtains the
    complementary local combinations by the established rank-revealing SVD;
    it never invokes global molecular symmetry.

    ZAFF defaults to CHARM for ring self-potentials because its cyclic height
    components give a smooth, symmetry-friendly global chart. SMITH's
    triangular-flap default remains unchanged for optimization and selective
    local-coordinate operations. Frozen ``#GIC`` restart definitions are
    consumed as written, so this policy never rewrites an existing field.
    """

    path = Path(xyzin)
    write_oracle_sonic_contract_from_xyzin(path)
    definition = build_gic_definition_from_xyzin(
        path,
        symmetrize=False,
        symmetry_group="C1",
        local_salc=True,
        ring_puckering_model=ring_puckering_model,
    )
    primary_b = np.asarray(build_gic_b_matrix(definition).rows, dtype=float)
    primary_singular = np.linalg.svd(primary_b, compute_uv=False)
    primary_relative = (
        0.0
        if not primary_singular.size or primary_singular[0] <= 0.0
        else float(primary_singular[-1] / primary_singular[0])
    )
    primary_rank = int(np.linalg.matrix_rank(primary_b, tol=1.0e-7))
    fallback = primary_rank < int(definition.target_rank) or primary_relative < float(
        minimum_relative_singular_value
    )
    if not fallback:
        redundant = augment_shared_proton_coordinates(definition)
        return redundant, {
            "policy": "LOCAL_SALC_WITH_RANK_STABLE_SVD_FALLBACK_V1",
            "local_salc_requested": True,
            "local_salc_realized": True,
            "fallback_used": False,
            "rank": primary_rank,
            "target_rank": int(definition.target_rank),
            "relative_smallest_singular_value": primary_relative,
            "effective_point_group": "C1",
            "global_symmetrization": False,
            "ring_puckering_model": str(ring_puckering_model).upper(),
            "field_coordinate_count": len(redundant.gics),
            "field_redundancy": len(redundant.gics) - primary_rank,
            "normal_mode_chart": "DEFERRED_NONREDUNDANT_REDUCTION",
        }
    stable = build_gic_definition_from_xyzin(
        path,
        symmetrize=False,
        symmetry_group="C1",
        local_salc=False,
        ring_puckering_model=ring_puckering_model,
    )
    stable_b = np.asarray(build_gic_b_matrix(stable).rows, dtype=float)
    stable_singular = np.linalg.svd(stable_b, compute_uv=False)
    stable_relative = (
        0.0
        if not stable_singular.size or stable_singular[0] <= 0.0
        else float(stable_singular[-1] / stable_singular[0])
    )
    stable_rank = int(np.linalg.matrix_rank(stable_b, tol=1.0e-7))
    if stable_rank < int(stable.target_rank) or stable_relative < float(
        minimum_relative_singular_value
    ):
        raise ValueError(
            "local SONIC chart remains rank deficient after the rank-revealing fallback"
        )
    redundant = augment_shared_proton_coordinates(stable)
    return redundant, {
        "policy": "LOCAL_SALC_WITH_RANK_STABLE_SVD_FALLBACK_V1",
        "local_salc_requested": True,
        "local_salc_realized": False,
        "fallback_used": True,
        "fallback_reason": "LOCAL_SALC_B_MATRIX_ILL_CONDITIONED",
        "primary_rank": primary_rank,
        "rank": stable_rank,
        "target_rank": int(stable.target_rank),
        "primary_relative_smallest_singular_value": primary_relative,
        "relative_smallest_singular_value": stable_relative,
        "field_coordinate_count": len(redundant.gics),
        "field_redundancy": len(redundant.gics) - stable_rank,
        "normal_mode_chart": "DEFERRED_NONREDUNDANT_REDUCTION",
        "effective_point_group": "C1",
        "global_symmetrization": False,
        "ring_puckering_model": str(ring_puckering_model).upper(),
    }


def acquire_zaff_anharmonic_model(
    xyzin: Path | str,
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    evaluator: ZaffAnharmonicEvaluator,
    *,
    plan: ZaffAnharmonicModel | None = None,
    config: ZaffAnharmonicAcquisitionConfig = ZaffAnharmonicAcquisitionConfig(),
) -> ZaffAnharmonicModel:
    """Acquire 0, +/-h and +/-2h gradients and compile diagonal ZAFF terms.

    Every displaced geometry is generated in the frozen local-SONIC chart by
    LINK's finite-predictor/constrained-corrector back-transform.  Nonplanar
    signed out-of-plane coordinates additionally request a constrained
    relaxation at zero; its energy supplies the physical planar barrier.
    """

    field_definition, definition_audit = build_zaff_local_sonic_definition(xyzin)
    definition = independent_zaff_analysis_definition(field_definition)
    reference_coordinates = np.asarray(reference_coordinates_angstrom, dtype=float)
    field_contract = redundant_zaff_field_contract(
        field_definition, definition, definition_audit
    )
    current_plan = plan or build_zaff_anharmonic_plan(
        definition,
        reference_coordinates,
        config=config,
    )
    current_plan = replace(current_plan, redundant_field_contract=field_contract)
    _validate_plan_against_definition(current_plan, definition, reference_coordinates)
    requests = _build_acquisition_requests(
        definition,
        tuple(str(atom) for atom in atoms),
        reference_coordinates,
        current_plan,
        config,
    )
    reference_request = requests[0]
    reference_result = _normalize_result(evaluator(reference_request), reference_request)
    remaining = requests[1:]
    workers = _resolved_workers(config.parallel_workers, len(remaining))
    if workers <= 1:
        other_results = tuple(
            _normalize_result(evaluator(request), request) for request in remaining
        )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = tuple(pool.submit(evaluator, request) for request in remaining)
            other_results = tuple(
                _normalize_result(future.result(), request)
                for future, request in zip(futures, remaining, strict=True)
            )
    results = (reference_result, *other_results)
    (
        terms,
        quadratic_matrix,
        semidiagonal_cubic,
        quadratic_matrix_directed,
        harmonic_reciprocity_audit,
    ) = _compile_terms(
        definition,
        current_plan.coordinates,
        requests,
        results,
        config,
    )
    return replace(
        current_plan,
        requests=requests,
        results=results,
        terms=terms,
        quadratic_matrix=quadratic_matrix,
        quadratic_matrix_directed=quadratic_matrix_directed,
        harmonic_reciprocity_audit=harmonic_reciprocity_audit,
        semidiagonal_cubic_i_j_j=semidiagonal_cubic,
        status="COMPILED",
    )


def attach_acquired_anharmonic_model(force_field: Any, model: ZaffAnharmonicModel) -> Any:
    """Return a force field carrying a completed, serialized anharmonic model."""

    if model.status != "COMPILED":
        raise ValueError("only a completed ZAFF anharmonic model can be attached")
    return replace(force_field, anharmonic_model=model)


class ExternalZaffAnharmonicEvaluator:
    """JSON file adapter for Gaussian/ORCA or another external E/G driver.

    The command template must contain ``{request}`` and ``{response}``.  The
    response must obey :data:`ZAFF_ANHARMONIC_POINT_RESULT_SCHEMA`.
    """

    def __init__(self, command: str, run_directory: Path | str) -> None:
        if "{request}" not in command or "{response}" not in command:
            raise ValueError("external command must contain {request} and {response}")
        self.command = str(command)
        self.run_directory = Path(run_directory)

    def __call__(self, request: ZaffAnharmonicPointRequest) -> ZaffAnharmonicPointResult:
        directory = self.run_directory / request.point_id
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        request_path.write_text(
            json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = self.command.format(request=request_path, response=response_path)
        subprocess.run(shlex.split(command), check=True, cwd=directory)
        if not response_path.exists():
            raise ValueError(f"external evaluator did not write {response_path}")
        return ZaffAnharmonicPointResult.from_dict(
            normalize_legacy_zaff_payload(
                json.loads(response_path.read_text(encoding="utf-8"))
            )
        )


def _build_acquisition_requests(
    definition: Any,
    atoms: tuple[str, ...],
    reference_coordinates: np.ndarray,
    plan: ZaffAnharmonicModel,
    config: ZaffAnharmonicAcquisitionConfig,
) -> tuple[ZaffAnharmonicPointRequest, ...]:
    reference_values = evaluate_gic_values(
        definition,
        coordinates_angstrom=reference_coordinates,
    )

    def evaluate(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = evaluate_gic_values(definition, coordinates_angstrom=coordinates)
        rows = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
            dtype=float,
        )
        return values, rows

    requests: list[ZaffAnharmonicPointRequest] = [
        ZaffAnharmonicPointRequest(
            point_id="reference",
            purpose="REFERENCE_GRADIENT",
            atoms=atoms,
            coordinates_angstrom=reference_coordinates.copy(),
        )
    ]
    for coordinate in plan.coordinates:
        for multiplier in (-2, -1, 1, 2):
            target = reference_values.copy()
            target[coordinate.index] += multiplier * coordinate.step
            result = hybrid_internal_coordinate_step(
                definition,
                reference_coordinates,
                target,
                evaluate,
                tolerance=config.backtransform_tolerance,
            )
            if not result.converged:
                # Small hard-coordinate SALCs can be numerically awkward for
                # the soft/hard partition even though the full frozen chart is
                # regular.  Fall back to the complete damped solver; finite
                # predictors remain the primary path for the soft coordinates.
                result = nonlinear_internal_coordinate_step(
                    reference_coordinates,
                    target,
                    evaluate,
                    tolerance=config.backtransform_tolerance,
                )
            if not result.converged:
                raise ValueError(
                    f"SONIC back-transform failed for {coordinate.identifier} at {multiplier:+d}h; "
                    f"residual={float(np.linalg.norm(result.residual)):.3e}"
                )
            achieved = float(result.values[coordinate.index])
            requests.append(
                ZaffAnharmonicPointRequest(
                    point_id=f"q{coordinate.index:04d}_{multiplier:+d}h".replace("+", "p").replace(
                        "-", "m"
                    ),
                    purpose="SEMIDIAGONAL_GRADIENT_STENCIL",
                    atoms=atoms,
                    coordinates_angstrom=result.coordinates_angstrom,
                    coordinate_index=coordinate.index,
                    coordinate_identifier=coordinate.identifier,
                    offset_multiplier=multiplier,
                    requested_coordinate_value=float(target[coordinate.index]),
                    achieved_coordinate_value=achieved,
                )
            )
        if coordinate.planar_probe:
            target = reference_values.copy()
            target[coordinate.index] = 0.0
            result = hybrid_internal_coordinate_step(
                definition,
                reference_coordinates,
                target,
                evaluate,
                tolerance=config.backtransform_tolerance,
            )
            if not result.converged:
                result = nonlinear_internal_coordinate_step(
                    reference_coordinates,
                    target,
                    evaluate,
                    tolerance=config.backtransform_tolerance,
                )
            if not result.converged:
                raise ValueError(
                    f"planar predictor failed for {coordinate.identifier}; "
                    f"residual={float(np.linalg.norm(result.residual)):.3e}"
                )
            requests.append(
                ZaffAnharmonicPointRequest(
                    point_id=f"q{coordinate.index:04d}_planar",
                    purpose="PLANAR_DOUBLE_WELL_BARRIER",
                    atoms=atoms,
                    coordinates_angstrom=result.coordinates_angstrom,
                    coordinate_index=coordinate.index,
                    coordinate_identifier=coordinate.identifier,
                    requested_coordinate_value=0.0,
                    achieved_coordinate_value=float(result.values[coordinate.index]),
                    operation="CONSTRAINED_OPTIMIZATION_THEN_ENERGY_GRADIENT",
                    constraints=(
                        {
                            "basis": "SONIC_LOCAL",
                            "coordinate_index": coordinate.index,
                            "coordinate_identifier": coordinate.identifier,
                            "target_value": 0.0,
                            "unit": coordinate.unit,
                        },
                    ),
                )
            )
    return tuple(requests)


def _compile_terms(
    definition: Any,
    coordinates: tuple[ZaffAnharmonicCoordinate, ...],
    requests: tuple[ZaffAnharmonicPointRequest, ...],
    results: tuple[ZaffAnharmonicPointResult, ...],
    config: ZaffAnharmonicAcquisitionConfig,
) -> tuple[
    tuple[ZaffAnharmonicTerm, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Mapping[str, Any],
]:
    request_by_id = {request.point_id: request for request in requests}
    result_by_id = {result.point_id: result for result in results}
    reference_result = result_by_id["reference"]
    reference_geometry = _result_geometry(reference_result, request_by_id["reference"])
    reference_internal_gradient = _internal_gradient(
        definition,
        reference_geometry,
        reference_result.cartesian_gradient_hartree_per_bohr,
    )
    coordinate_count = len(coordinates)
    quadratic_matrix = np.zeros((coordinate_count, coordinate_count), dtype=float)
    semidiagonal_cubic = np.zeros((coordinate_count, coordinate_count), dtype=float)
    terms: list[ZaffAnharmonicTerm] = []
    for coordinate in coordinates:
        ids = {
            multiplier: f"q{coordinate.index:04d}_{multiplier:+d}h".replace("+", "p").replace(
                "-", "m"
            )
            for multiplier in (-2, -1, 1, 2)
        }
        gradients: dict[int, np.ndarray] = {}
        for multiplier, point_id in ids.items():
            request = request_by_id[point_id]
            result = result_by_id[point_id]
            gradients[multiplier] = _internal_gradient(
                definition,
                _result_geometry(result, request),
                result.cartesian_gradient_hartree_per_bohr,
            )
        component = coordinate.index
        stencil_multipliers = (-2, -1, 0, 1, 2)
        stencil_gradients = np.vstack(
            tuple(
                reference_internal_gradient if multiplier == 0 else gradients[multiplier]
                for multiplier in stencil_multipliers
            )
        )
        reference_value = float(coordinate.reference_value)
        stencil_offsets = np.asarray(
            tuple(
                0.0
                if multiplier == 0
                else float(
                    request_by_id[ids[multiplier]].achieved_coordinate_value
                    - reference_value
                )
                for multiplier in stencil_multipliers
            ),
            dtype=float,
        )
        derivative_weights = _finite_difference_derivative_weights(
            stencil_offsets,
            maximum_derivative=3,
        )
        quadratic_column = derivative_weights[1] @ stencil_gradients
        cubic_column = derivative_weights[2] @ stencil_gradients
        quadratic_matrix[:, component] = quadratic_column
        semidiagonal_cubic[:, component] = cubic_column
        k2 = float(quadratic_column[component])
        k3 = float(cubic_column[component])
        k4 = float(derivative_weights[3] @ stencil_gradients[:, component])
        planar_id = f"q{coordinate.index:04d}_planar"
        barrier = None
        if planar_id in result_by_id:
            planar_request = request_by_id[planar_id]
            planar_result = result_by_id[planar_id]
            planar_values = evaluate_gic_values(
                definition,
                coordinates_angstrom=_result_geometry(planar_result, planar_request),
            )
            if abs(float(planar_values[coordinate.index])) > max(
                10.0 * config.backtransform_tolerance,
                1.0e-6,
            ):
                raise ValueError(
                    f"constrained planar result for {coordinate.identifier} does not satisfy "
                    f"chi=0: {float(planar_values[coordinate.index]):.6g}"
                )
            barrier = planar_result.energy_hartree - reference_result.energy_hartree
        functional_form, parameters, status = _compile_functional_parameters(
            coordinate,
            k2,
            k3,
            k4,
            barrier,
            config,
        )
        point_ids = ("reference", ids[-2], ids[-1], ids[1], ids[2])
        if planar_id in result_by_id:
            point_ids = (*point_ids, planar_id)
        terms.append(
            ZaffAnharmonicTerm(
                coordinate_index=coordinate.index,
                coordinate_identifier=coordinate.identifier,
                family=coordinate.family,
                functional_form=functional_form,
                quadratic=float(k2),
                cubic=float(k3),
                quartic=float(k4),
                parameters=parameters,
                acquisition_point_ids=point_ids,
                status=status,
            )
        )
    directed_quadratic = quadratic_matrix.copy()
    quadratic_matrix = 0.5 * (directed_quadratic + directed_quadratic.T)
    antisymmetric = directed_quadratic - directed_quadratic.T
    upper = antisymmetric[np.triu_indices(coordinate_count, k=1)]
    reciprocity_audit: dict[str, Any] = {
        "policy": "MEAN_OF_RECIPROCAL_GRADIENT_DERIVATIVES_V1",
        "pair_count": int(upper.size),
        "rms_directed_difference": (
            0.0 if upper.size == 0 else float(np.sqrt(np.mean(upper * upper)))
        ),
        "maximum_absolute_directed_difference": (
            0.0 if upper.size == 0 else float(np.max(np.abs(upper)))
        ),
    }
    return (
        tuple(terms),
        quadratic_matrix,
        semidiagonal_cubic,
        directed_quadratic,
        reciprocity_audit,
    )


def _finite_difference_derivative_weights(
    offsets: np.ndarray,
    *,
    maximum_derivative: int,
) -> np.ndarray:
    """Return nonuniform finite-difference weights about zero.

    SONIC back-transformation meets a tight residual threshold but displaced
    coordinates are not bitwise equal to ``+/-h``.  Building weights from the
    achieved coordinates prevents that harmless realization error from being
    amplified by the cubic-gradient stencil used for quartic force constants.
    """

    points = np.asarray(offsets, dtype=float).reshape(-1)
    if len(np.unique(points)) != points.size:
        raise ValueError("anharmonic finite-difference stencil contains duplicate points")
    if maximum_derivative < 0 or maximum_derivative >= points.size:
        raise ValueError("invalid finite-difference derivative order")
    powers = np.vstack(tuple(points**order for order in range(points.size)))
    weights = np.zeros((maximum_derivative + 1, points.size), dtype=float)
    for derivative in range(maximum_derivative + 1):
        target = np.zeros(points.size, dtype=float)
        target[derivative] = float(factorial(derivative))
        weights[derivative] = np.linalg.solve(powers, target)
    return weights


def _compile_functional_parameters(
    coordinate: ZaffAnharmonicCoordinate,
    k2: float,
    k3: float,
    k4: float,
    planar_barrier: float | None,
    config: ZaffAnharmonicAcquisitionConfig,
) -> tuple[str, Mapping[str, Any], str]:
    family = coordinate.family
    if (
        _is_stretch_family(family)
        and len(coordinate.primitive_components) == 1
        and coordinate.reference_value > 0.0
        and k2 > 0.0
    ):
        if config.stretch_model == "inverse_power":
            try:
                shape = inverse_power_stretch_from_local_derivatives(
                    coordinate.reference_value, k2, k3
                )
            except ValueError:
                pass
            else:
                return (
                    "INVERSE_POWER_STRETCH",
                    {
                        "r0": shape.r0,
                        "depth": shape.depth,
                        "exponent": shape.exponent,
                    },
                    "COMPILED",
                )
        if config.stretch_model == "spf":
            shape = spf_stretch_from_local_derivatives(
                coordinate.reference_value,
                k2,
                k3,
                quartic=k4,
                asymptotic_depth=coordinate.asymptotic_depth,
            )
            return (
                "SIMONS_PARR_FINLAN_INVERSE_DISTANCE",
                {
                    "r0": shape.r0,
                    "coefficients": list(shape.coefficients),
                    "asymptotic_depth": coordinate.asymptotic_depth,
                    "asymptotic_policy": (
                        "ZAFF_LIBRARY_CONSTRAINED"
                        if coordinate.asymptotic_depth is not None
                        else "UNCONSTRAINED_DIAGNOSTIC"
                    ),
                },
                "COMPILED",
            )
        beta = -coordinate.reference_value * k3 / (3.0 * k2)
        if beta > 0.0:
            epsilon = k2 * coordinate.reference_value**2 / (2.0 * beta**2)
            return (
                "MORSE_QM_K2_K3",
                {
                    "r0": coordinate.reference_value,
                    "epsilon": epsilon,
                    "beta_dimensionless": beta,
                },
                "COMPILED",
            )
        shape = spf_stretch_from_local_derivatives(
            coordinate.reference_value,
            k2,
            k3,
            quartic=k4,
            asymptotic_depth=coordinate.asymptotic_depth,
        )
        return (
            "SIMONS_PARR_FINLAN_FALLBACK",
            {
                "r0": shape.r0,
                "coefficients": list(shape.coefficients),
                "asymptotic_depth": coordinate.asymptotic_depth,
            },
            "COMPILED",
        )
    if _is_out_of_plane_family(family) and k2 > 0.0:
        if (
            planar_barrier is not None
            and planar_barrier > config.planar_barrier_tolerance_hartree
            and abs(coordinate.reference_value) >= config.planar_probe_threshold_radian
        ):
            shape = EvenOutOfPlanePolynomial.symmetric_double_well(
                abs(coordinate.reference_value),
                k2,
                planar_barrier,
            )
            return (
                "EVEN_SIN_OUT_OF_PLANE_DOUBLE_WELL",
                {
                    "minimum_angle": abs(coordinate.reference_value),
                    "planar_barrier_hartree": planar_barrier,
                    "c0": shape.c0,
                    "c2": shape.c2,
                    "c4": shape.c4,
                    "c6": shape.c6,
                },
                "COMPILED",
            )
        if abs(coordinate.reference_value) < config.planar_probe_threshold_radian:
            shape = EvenOutOfPlanePolynomial.planar_single_well(k2, k4)
            return (
                "EVEN_SIN_OUT_OF_PLANE_SINGLE_WELL",
                {"c0": shape.c0, "c2": shape.c2, "c4": shape.c4, "c6": shape.c6},
                "COMPILED",
            )
    if _is_angle_family(family) and len(coordinate.primitive_components) == 1 and k2 > 0.0:
        try:
            shape = CosineSeriesAngleShape.from_local_derivatives(
                coordinate.reference_value,
                k2,
                k3,
            )
        except ValueError:
            return (
                "LINEAR_OR_ILL_CONDITIONED_BEND",
                {"theta0": coordinate.reference_value, "k2": k2, "k4": k4},
                "REQUIRES_TWO_COMPONENT_LINEAR_BEND",
            )
        return (
            "COSINE_SERIES_ANGLE_1_2_3",
            {
                "theta0": shape.theta0,
                "harmonics": list(shape.harmonics),
                "coefficients": list(shape.coefficients),
            },
            "COMPILED",
        )
    if _is_torsion_family(family):
        return (
            "FOURIER_TORSION",
            {"local_k2": k2, "local_k3": k3, "local_k4": k4},
            "LOCAL_DERIVATIVES_ACQUIRED_TORSIONAL_SCAN_REQUIRED",
        )
    return (
        coordinate.functional_form,
        {"q0": coordinate.reference_value, "k2": k2, "k3": k3, "k4": k4},
        "LOCAL_DERIVATIVES_COMPILED",
    )


def _internal_gradient(
    definition: Any,
    coordinates_angstrom: np.ndarray,
    cartesian_gradient_hartree_per_bohr: np.ndarray,
) -> np.ndarray:
    b_matrix = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=coordinates_angstrom).rows,
        dtype=float,
    )
    gradient_bohr = np.asarray(cartesian_gradient_hartree_per_bohr, dtype=float).reshape(-1)
    if gradient_bohr.size != b_matrix.shape[1]:
        raise ValueError("external gradient size differs from the ZAFF atom count")
    gradient_per_angstrom = gradient_bohr / BOHR_TO_ANGSTROM
    internal, *_ = np.linalg.lstsq(b_matrix.T, gradient_per_angstrom, rcond=1.0e-10)
    return np.asarray(internal, dtype=float)


def _result_geometry(
    result: ZaffAnharmonicPointResult,
    request: ZaffAnharmonicPointRequest,
) -> np.ndarray:
    return np.asarray(
        request.coordinates_angstrom
        if result.coordinates_angstrom is None
        else result.coordinates_angstrom,
        dtype=float,
    )


def _normalize_result(
    result: ZaffAnharmonicPointResult | Mapping[str, Any],
    request: ZaffAnharmonicPointRequest,
) -> ZaffAnharmonicPointResult:
    normalized = (
        ZaffAnharmonicPointResult.from_dict(result) if isinstance(result, Mapping) else result
    )
    if not isinstance(normalized, ZaffAnharmonicPointResult):
        raise TypeError("anharmonic evaluator must return ZaffAnharmonicPointResult or a mapping")
    if normalized.point_id != request.point_id:
        raise ValueError("anharmonic evaluator returned a mismatched point_id")
    gradient = np.asarray(normalized.cartesian_gradient_hartree_per_bohr, dtype=float).reshape(-1)
    if gradient.size != 3 * len(request.atoms) or not np.all(np.isfinite(gradient)):
        raise ValueError("anharmonic evaluator returned an invalid Cartesian gradient")
    if not isfinite(float(normalized.energy_hartree)):
        raise ValueError("anharmonic evaluator returned a non-finite energy")
    geometry = _result_geometry(normalized, request)
    if geometry.shape != (len(request.atoms), 3) or not np.all(np.isfinite(geometry)):
        raise ValueError("anharmonic evaluator returned an invalid geometry")
    return replace(
        normalized,
        cartesian_gradient_hartree_per_bohr=gradient,
        coordinates_angstrom=geometry,
    )


def _validate_plan_against_definition(
    plan: ZaffAnharmonicModel,
    definition: Any,
    coordinates_angstrom: np.ndarray,
) -> None:
    if plan.basis != "SONIC_LOCAL":
        raise ValueError("ZAFF anharmonic acquisition requires the local SONIC basis")
    identifiers = tuple(str(gic.identifier) for gic in definition.gics)
    planned = tuple(item.identifier for item in plan.coordinates)
    if identifiers != planned:
        raise ValueError("serialized anharmonic plan differs from the current SONIC definition")
    values = evaluate_gic_values(definition, coordinates_angstrom=coordinates_angstrom)
    expected = np.asarray([item.reference_value for item in plan.coordinates], dtype=float)
    if not np.allclose(values, expected, atol=2.0e-7, rtol=0.0):
        raise ValueError("serialized anharmonic reference values differ from the current geometry")


def _coordinate_unit_and_step(
    family: str,
    config: ZaffAnharmonicAcquisitionConfig,
    *,
    primitive_functions: Sequence[str] = (),
) -> tuple[str, float]:
    if (
        str(family).upper() == "RING_PUCKER_COMPONENT"
        and primitive_functions
        and all(str(function).upper() == "H" for function in primitive_functions)
    ):
        return "angstrom", float(config.stretch_step_angstrom)
    if _is_stretch_family(family) or "TRANSLATION" in family:
        step = (
            config.fragment_translation_step_angstrom
            if "TRANSLATION" in family
            else config.stretch_step_angstrom
        )
        return "angstrom", float(step)
    if any(
        token in family
        for token in (
            "ANGLE",
            "BEND",
            "TORSION",
            "DIHEDRAL",
            "OUT_OF_PLANE",
            "PUCKER",
            "INVERSION",
            "ROTATION",
        )
    ):
        return "radian", float(config.angular_step_radian)
    return "dimensionless", float(config.dimensionless_step)


def _planned_functional_form(
    family: str,
    stretch_model: str,
    *,
    primitive_component_count: int,
    reference_value: float,
) -> str:
    if _is_stretch_family(family):
        if primitive_component_count != 1 or reference_value <= 0.0:
            return "LOCAL_SONIC_STRETCH_ANHARMONIC_DIAGONAL"
        return {
            "morse": "MORSE_QM_K2_K3",
            "inverse_power": "INVERSE_POWER_STRETCH",
            "spf": "SIMONS_PARR_FINLAN_INVERSE_DISTANCE",
        }[stretch_model]
    if _is_out_of_plane_family(family):
        return "EVEN_SIN_OUT_OF_PLANE_AUTO_SINGLE_OR_DOUBLE_WELL"
    if _is_angle_family(family):
        if primitive_component_count != 1:
            return "LOCAL_SONIC_ANGLE_ANHARMONIC_DIAGONAL"
        return "COSINE_SERIES_ANGLE_1_2_3"
    if _is_torsion_family(family):
        return "FOURIER_TORSION"
    return "LOCAL_ANHARMONIC_DIAGONAL"


def _is_stretch_family(family: str) -> bool:
    return "STRETCH" in family or family in {"BOND", "PSEUDO_BOND"}


def _is_angle_family(family: str) -> bool:
    return any(token in family for token in ("ANGLE", "BEND")) and not _is_out_of_plane_family(
        family
    )


def _is_torsion_family(family: str) -> bool:
    return any(token in family for token in ("TORSION", "DIHEDRAL"))


def _is_out_of_plane_family(family: str) -> bool:
    return any(token in family for token in ("OUT_OF_PLANE", "INVERSION"))


def _resolved_workers(requested: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    available = max(1, int(__import__("os").cpu_count() or 1))
    return min(task_count, requested if requested > 0 else min(available, 32))
