"""Versioned ZAFF force fields and the ARCHITECT analytic runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from matrix_zaff.geminal_coupling import ZaffGeminalXHCoupling
from matrix_zaff.compatibility import normalize_legacy_zaff_payload
from matrix_numerics import (
    gpu_validation_enabled,
    resolve_compute_backend,
)

if TYPE_CHECKING:
    from .acquisition import ZaffAnharmonicModel


ZAFF_FORCE_FIELD_SCHEMA = "matrix.zaff.force_field.v1"
ZAFF_EXPANSION = "Zoomable Asymptotically-Correct Force Field"
# Compatibility name for clients written before the force field was named ZAFF.
ARCHITECT_FORCE_FIELD_SCHEMA = ZAFF_FORCE_FIELD_SCHEMA
ARCHITECT_EVALUATION_SCHEMA = "matrix.architect.evaluation.v1"
ARCHITECT_RUNTIME_CONTRACT_SCHEMA = "matrix.architect.runtime_contract.v1"
ARCHITECT_DERIVATIVE_VALIDATION_SCHEMA = "matrix.architect.derivative_validation.v1"
BOHR_TO_ANGSTROM = 0.529177210903
_PROPERTIES = frozenset({"energy", "gradient", "hessian"})


@dataclass(frozen=True)
class ZaffZoomLevel:
    """One evaluable and quantitatively audited level of a ZAFF field."""

    identifier: str
    description: str
    hessian_hartree_per_bohr2: np.ndarray
    coordinate_basis: str
    parameter_count: int
    atomic_type_count: int
    primitive_type_count: int
    coupling_count: int
    relative_l2_eigenvalue_error: float
    maximum_absolute_eigenvalue_error_hartree_per_bohr2: float
    resolved_sign_mismatch_count: int
    equilibrium_gradient_rms_hartree_per_angstrom: float
    equilibrium_gradient_max_hartree_per_angstrom: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("ZAFF zoom-level identifier must be nonempty")
        if self.coordinate_basis not in {"PIC", "SONIC_LOCAL", "HYBRID_LOCAL"}:
            raise ValueError("ZAFF zoom-level basis must be PIC, SONIC_LOCAL or HYBRID_LOCAL")
        counts = (
            self.parameter_count,
            self.atomic_type_count,
            self.primitive_type_count,
            self.coupling_count,
            self.resolved_sign_mismatch_count,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("ZAFF zoom-level counts must be non-negative")
        diagnostics = (
            self.relative_l2_eigenvalue_error,
            self.maximum_absolute_eigenvalue_error_hartree_per_bohr2,
            self.equilibrium_gradient_rms_hartree_per_angstrom,
            self.equilibrium_gradient_max_hartree_per_angstrom,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in diagnostics):
            raise ValueError("ZAFF zoom-level diagnostics must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "description": self.description,
            "hessian_hartree_per_bohr2": self.hessian_hartree_per_bohr2.tolist(),
            "coordinate_basis": self.coordinate_basis,
            "parameter_count": self.parameter_count,
            "atomic_type_count": self.atomic_type_count,
            "primitive_type_count": self.primitive_type_count,
            "coupling_count": self.coupling_count,
            "relative_l2_eigenvalue_error": self.relative_l2_eigenvalue_error,
            "relative_l2_eigenvalue_error_percent": (100.0 * self.relative_l2_eigenvalue_error),
            "maximum_absolute_eigenvalue_error_hartree_per_bohr2": (
                self.maximum_absolute_eigenvalue_error_hartree_per_bohr2
            ),
            "resolved_sign_mismatch_count": self.resolved_sign_mismatch_count,
            "equilibrium_gradient_rms_hartree_per_angstrom": (
                self.equilibrium_gradient_rms_hartree_per_angstrom
            ),
            "equilibrium_gradient_max_hartree_per_angstrom": (
                self.equilibrium_gradient_max_hartree_per_angstrom
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffZoomLevel":
        return cls(
            identifier=str(payload["identifier"]),
            description=str(payload["description"]),
            hessian_hartree_per_bohr2=np.asarray(payload["hessian_hartree_per_bohr2"], dtype=float),
            coordinate_basis=str(payload["coordinate_basis"]),
            parameter_count=int(payload["parameter_count"]),
            atomic_type_count=int(payload["atomic_type_count"]),
            primitive_type_count=int(payload["primitive_type_count"]),
            coupling_count=int(payload["coupling_count"]),
            relative_l2_eigenvalue_error=float(payload["relative_l2_eigenvalue_error"]),
            maximum_absolute_eigenvalue_error_hartree_per_bohr2=float(
                payload["maximum_absolute_eigenvalue_error_hartree_per_bohr2"]
            ),
            resolved_sign_mismatch_count=int(payload["resolved_sign_mismatch_count"]),
            equilibrium_gradient_rms_hartree_per_angstrom=float(
                payload["equilibrium_gradient_rms_hartree_per_angstrom"]
            ),
            equilibrium_gradient_max_hartree_per_angstrom=float(
                payload["equilibrium_gradient_max_hartree_per_angstrom"]
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ZaffForceField:
    """Compiled harmonic ZAFF model with a stable nonlinear extension slot.

    Coordinates are serialized in Angstrom, whereas derivatives use atomic
    units.  The reference gradient is retained explicitly because a developing
    model may be inspected before the stationarity constraint is satisfied.
    Publication models are expected to have a numerically zero reference
    gradient after bonded/nonbonded equilibrium refitting.
    """

    atoms: tuple[str, ...]
    reference_coordinates_angstrom: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    energy_reference_hartree: float = 0.0
    gradient_reference_hartree_per_bohr: np.ndarray | None = None
    coordinate_contract: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    zoom_levels: tuple[ZaffZoomLevel, ...] = ()
    active_zoom_level: str = ""
    anharmonic_model: ZaffAnharmonicModel | None = None
    geminal_xh_couplings: tuple[ZaffGeminalXHCoupling, ...] = ()
    schema: str = ZAFF_FORCE_FIELD_SCHEMA

    def __post_init__(self) -> None:
        atoms = tuple(str(atom) for atom in self.atoms)
        if not atoms or any(not atom for atom in atoms):
            raise ValueError("ZAFF atom symbols must be nonempty")
        reference = np.asarray(self.reference_coordinates_angstrom, dtype=float)
        if reference.shape != (len(atoms), 3) or not np.all(np.isfinite(reference)):
            raise ValueError("ZAFF reference coordinates must have shape (natoms, 3)")
        ncart = 3 * len(atoms)
        hessian = np.asarray(self.hessian_hartree_per_bohr2, dtype=float)
        if hessian.shape != (ncart, ncart) or not np.all(np.isfinite(hessian)):
            raise ValueError("ZAFF Hessian shape does not match the atom count")
        if not np.allclose(hessian, hessian.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("ZAFF Hessian must be symmetric")
        gradient = (
            np.zeros(ncart, dtype=float)
            if self.gradient_reference_hartree_per_bohr is None
            else np.asarray(self.gradient_reference_hartree_per_bohr, dtype=float).reshape(-1)
        )
        if gradient.shape != (ncart,) or not np.all(np.isfinite(gradient)):
            raise ValueError("ZAFF reference gradient shape does not match the atom count")
        if self.schema != ZAFF_FORCE_FIELD_SCHEMA:
            raise ValueError(f"unsupported ZAFF force-field schema: {self.schema}")
        couplings = tuple(self.geminal_xh_couplings)
        coupling_keys = []
        reference_bohr = reference / BOHR_TO_ANGSTROM
        for coupling in couplings:
            if max(coupling.center, coupling.hydrogen1, coupling.hydrogen2) >= len(atoms):
                raise ValueError("geminal coupling atom index exceeds field atom count")
            if atoms[coupling.hydrogen1] != "H" or atoms[coupling.hydrogen2] != "H" or atoms[coupling.center] == "H":
                raise ValueError("geminal coupling must join two X-H bonds on one heavy atom")
            coupling.reference_hessian(reference_bohr)
            coupling_keys.append((coupling.center, *sorted((coupling.hydrogen1, coupling.hydrogen2))))
        if len(coupling_keys) != len(set(coupling_keys)):
            raise ValueError("duplicate geminal X-H coupling")
        zoom_levels = tuple(self.zoom_levels)
        zoom_ids = tuple(item.identifier for item in zoom_levels)
        if len(set(zoom_ids)) != len(zoom_ids):
            raise ValueError("ZAFF zoom-level identifiers must be unique")
        if zoom_levels:
            expected_shape = (ncart, ncart)
            previous_parameters = -1
            previous_error = float("inf")
            for level in zoom_levels:
                level_hessian = np.asarray(level.hessian_hartree_per_bohr2, dtype=float)
                if level_hessian.shape != expected_shape or not np.all(np.isfinite(level_hessian)):
                    raise ValueError(f"ZAFF {level.identifier} Hessian shape is invalid")
                if level.parameter_count < previous_parameters:
                    raise ValueError("ZAFF zoom levels must not decrease parameter count")
                if level.relative_l2_eigenvalue_error > previous_error + 1.0e-12:
                    raise ValueError("ZAFF zoom levels must not increase spectral error")
                previous_parameters = level.parameter_count
                previous_error = level.relative_l2_eigenvalue_error
            active_zoom_level = self.active_zoom_level or zoom_ids[-1]
            if active_zoom_level not in zoom_ids:
                raise ValueError(f"unknown active ZAFF zoom level: {active_zoom_level}")
            active_hessian = zoom_levels[
                zoom_ids.index(active_zoom_level)
            ].hessian_hartree_per_bohr2
            if not np.allclose(hessian, active_hessian, atol=1.0e-12, rtol=0.0):
                raise ValueError("active ZAFF Hessian differs from the top-level model Hessian")
        else:
            active_zoom_level = self.active_zoom_level
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "reference_coordinates_angstrom", reference)
        object.__setattr__(self, "hessian_hartree_per_bohr2", 0.5 * (hessian + hessian.T))
        object.__setattr__(self, "gradient_reference_hartree_per_bohr", gradient)
        object.__setattr__(self, "energy_reference_hartree", float(self.energy_reference_hartree))
        object.__setattr__(self, "zoom_levels", zoom_levels)
        object.__setattr__(self, "active_zoom_level", active_zoom_level)
        object.__setattr__(self, "geminal_xh_couplings", couplings)

    @property
    def stationary(self) -> bool:
        return bool(np.max(np.abs(self.gradient_reference_hartree_per_bohr)) <= 1.0e-8)

    def with_geminal_fit(self, fit: Mapping[str, Any]) -> "ZaffForceField":
        """Install Hessian-fitted Kij values and their dissociating terms."""

        if self.anharmonic_model is not None:
            raise ValueError("geminal fit must precede anharmonic ZAFF compilation")
        if fit.get("schema") != "matrix.architect.zaff_geminal_coupled_harmonic_fit.v1" or fit.get("accepted") is not True:
            raise ValueError("only an accepted geminal ARCHITECT fit can enter ZAFF")
        hessian = np.asarray(fit["model_cartesian_hessian_hartree_per_bohr2"], dtype=float)
        couplings = tuple(
            ZaffGeminalXHCoupling.from_dict(item)
            for item in fit["geminal_xh_couplings"]
        )
        return replace(
            self,
            hessian_hartree_per_bohr2=hessian,
            zoom_levels=(),
            active_zoom_level="",
            geminal_xh_couplings=couplings,
            metadata={**dict(self.metadata), "geminal_fit_protocol": dict(fit["protocol"])},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "owner": "ARCHITECT",
            "force_field_family": "ZAFF",
            "force_field_expansion": ZAFF_EXPANSION,
            "model": (
                "ZAFF_COMPILED_HARMONIC_PLUS_INTERNAL_ANHARMONIC_V2"
                if self.anharmonic_model is not None
                else "ZAFF_COMPILED_HARMONIC_WITH_DISSOCIATING_GEMINAL_XH_V1"
                if self.geminal_xh_couplings
                else "ZAFF_COMPILED_CARTESIAN_HARMONIC_V1"
            ),
            "units": {
                "coordinates": "angstrom",
                "energy": "hartree",
                "gradient": "hartree/bohr",
                "hessian": "hartree/bohr^2",
            },
            "runtime_contract": architect_runtime_contract(),
            "atoms": list(self.atoms),
            "reference_coordinates_angstrom": self.reference_coordinates_angstrom.tolist(),
            "energy_reference_hartree": self.energy_reference_hartree,
            "gradient_reference_hartree_per_bohr": (
                self.gradient_reference_hartree_per_bohr.tolist()
            ),
            "hessian_hartree_per_bohr2": self.hessian_hartree_per_bohr2.tolist(),
            "coordinate_contract": dict(self.coordinate_contract),
            "metadata": dict(self.metadata),
            "zoom_levels": [item.to_dict() for item in self.zoom_levels],
            "active_zoom_level": self.active_zoom_level,
            "anharmonic_model": (
                None if self.anharmonic_model is None else self.anharmonic_model.to_dict()
            ),
            "geminal_xh_couplings": [item.to_dict() for item in self.geminal_xh_couplings],
            "stationary": self.stationary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffForceField":
        from .acquisition import ZaffAnharmonicModel

        if payload.get("schema") != ZAFF_FORCE_FIELD_SCHEMA:
            raise ValueError(f"unsupported ZAFF force-field schema: {payload.get('schema')!r}")
        if payload.get("owner") not in {None, "ARCHITECT"}:
            raise ValueError("force-field owner must be ARCHITECT")
        if payload.get("force_field_family") not in {None, "ZAFF"}:
            raise ValueError("force-field family must be ZAFF")
        if payload.get("force_field_expansion") not in {None, ZAFF_EXPANSION}:
            raise ValueError("force-field expansion does not match ZAFF")
        runtime_contract = payload.get("runtime_contract")
        if runtime_contract is not None:
            if not isinstance(runtime_contract, Mapping):
                raise ValueError("ARCHITECT runtime contract must be a JSON object")
            for key, expected in architect_runtime_contract().items():
                # Additive capability records are optional in older, otherwise
                # valid force fields. New serializations always include them.
                if key == "charge_response" and key not in runtime_contract:
                    continue
                if runtime_contract.get(key) != expected:
                    raise ValueError(f"unsupported ARCHITECT runtime contract field: {key}")
        return cls(
            atoms=tuple(str(atom) for atom in payload["atoms"]),
            reference_coordinates_angstrom=np.asarray(
                payload["reference_coordinates_angstrom"], dtype=float
            ),
            energy_reference_hartree=float(payload.get("energy_reference_hartree", 0.0)),
            gradient_reference_hartree_per_bohr=(
                None
                if "gradient_reference_hartree_per_bohr" not in payload
                else np.asarray(payload["gradient_reference_hartree_per_bohr"], dtype=float)
            ),
            hessian_hartree_per_bohr2=np.asarray(payload["hessian_hartree_per_bohr2"], dtype=float),
            coordinate_contract=dict(payload.get("coordinate_contract", {})),
            metadata=dict(payload.get("metadata", {})),
            zoom_levels=tuple(
                ZaffZoomLevel.from_dict(item) for item in payload.get("zoom_levels", ())
            ),
            active_zoom_level=str(payload.get("active_zoom_level", "")),
            anharmonic_model=(
                None
                if payload.get("anharmonic_model") is None
                else ZaffAnharmonicModel.from_dict(payload["anharmonic_model"])
            ),
            geminal_xh_couplings=tuple(
                ZaffGeminalXHCoupling.from_dict(item)
                for item in payload.get("geminal_xh_couplings", ())
            ),
        )


@dataclass(frozen=True)
class ArchitectEvaluation:
    energy_hartree: float | None
    gradient_hartree_per_bohr: np.ndarray | None
    hessian_hartree_per_bohr2: np.ndarray | None
    properties: tuple[str, ...]
    source: str = "ARCHITECT"
    execution: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ARCHITECT_EVALUATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "properties": list(self.properties),
            "execution": dict(self.execution),
            "energy_hartree": self.energy_hartree,
            "cartesian_gradient_hartree_per_bohr": (
                None
                if self.gradient_hartree_per_bohr is None
                else self.gradient_hartree_per_bohr.tolist()
            ),
            "cartesian_hessian_hartree_per_bohr2": (
                None
                if self.hessian_hartree_per_bohr2 is None
                else self.hessian_hartree_per_bohr2.tolist()
            ),
        }


@dataclass(frozen=True)
class ArchitectDerivativeValidation:
    """Finite-difference audit of the public analytic E/G/H boundary."""

    zoom_level: str
    step_bohr: float
    probe_displacement_l2_bohr: float
    gradient_max_abs_error_hartree_per_bohr: float
    gradient_rms_error_hartree_per_bohr: float
    hessian_max_abs_error_hartree_per_bohr2: float
    hessian_rms_error_hartree_per_bohr2: float
    hessian_symmetry_max_abs_error_hartree_per_bohr2: float
    gradient_tolerance_hartree_per_bohr: float
    hessian_tolerance_hartree_per_bohr2: float
    passed: bool
    schema: str = ARCHITECT_DERIVATIVE_VALIDATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "zoom_level": self.zoom_level,
            "step_bohr": self.step_bohr,
            "probe_displacement_l2_bohr": self.probe_displacement_l2_bohr,
            "gradient_max_abs_error_hartree_per_bohr": (
                self.gradient_max_abs_error_hartree_per_bohr
            ),
            "gradient_rms_error_hartree_per_bohr": self.gradient_rms_error_hartree_per_bohr,
            "hessian_max_abs_error_hartree_per_bohr2": (
                self.hessian_max_abs_error_hartree_per_bohr2
            ),
            "hessian_rms_error_hartree_per_bohr2": self.hessian_rms_error_hartree_per_bohr2,
            "hessian_symmetry_max_abs_error_hartree_per_bohr2": (
                self.hessian_symmetry_max_abs_error_hartree_per_bohr2
            ),
            "gradient_tolerance_hartree_per_bohr": (self.gradient_tolerance_hartree_per_bohr),
            "hessian_tolerance_hartree_per_bohr2": (self.hessian_tolerance_hartree_per_bohr2),
            "passed": self.passed,
        }


def architect_runtime_contract() -> dict[str, Any]:
    """Return the stable ZAFF boundary consumed by LINK and SENTINEL."""

    return {
        "schema": ARCHITECT_RUNTIME_CONTRACT_SCHEMA,
        "force_field_schema": ZAFF_FORCE_FIELD_SCHEMA,
        "evaluation_schema": ARCHITECT_EVALUATION_SCHEMA,
        "consumers": ["LINK", "SENTINEL"],
        "coordinates": {"layout": "natoms_by_3", "unit": "angstrom"},
        "properties": {
            "energy": "hartree",
            "gradient": "hartree/bohr",
            "hessian": "hartree/bohr^2",
        },
        "batch": True,
        "zoom_level_selectable": True,
        "atom_order_must_match": True,
        "charge_response": {
            "modes": ["VARIATIONAL_QEQ_SQE", "LOCAL_HBOND_CLOSED_FORM"],
            "local_mode_owner": "ORACLE",
            "local_mode_derivatives": "SPARSE_ANALYTIC_FIRST_AND_SECOND",
            "local_mode_parameterization": ("PAIRED_CM5_OR_VARIATIONAL_QEQ_SQE_TEACHER"),
            "finite_box_boundary": "REQUIRED_AND_SHARED_BY_TEACHER_AND_STUDENT",
            "fallback": "VARIATIONAL_QEQ_SQE",
        },
    }


def evaluate_force_field(
    force_field: ZaffForceField,
    coordinates_angstrom: np.ndarray,
    *,
    properties: Sequence[str] = ("energy", "gradient", "hessian"),
    zoom_level: str | None = None,
    xyzin: Path | str | None = None,
) -> ArchitectEvaluation:
    """Evaluate ZAFF E/G/H in one pass for a Cartesian geometry.

    Supplying ``xyzin`` activates the compiled local-SONIC anharmonic runtime.
    Without it, the stable Cartesian harmonic runtime is retained.  Anharmonic
    evaluation is deliberately incompatible with a harmonic zoom-level
    selection because its derivative tensors belong to one frozen SONIC basis.
    """

    requested = _normalize_properties(properties)
    from .native_pic_field import NativePicField
    if isinstance(force_field,NativePicField):
        if xyzin is not None:raise ValueError("native PIC fields carry their frozen primitives; --xyzin is not used")
        if zoom_level is not None:
            from .protocol import select_zaff_level
            selected=select_zaff_level(force_field,zoom_level)
            if selected.value!=force_field.metadata.get("model_level"):raise ValueError("load the corresponding parent artifact for another native level")
        if "hessian" in requested:
            point=force_field(coordinates_angstrom);energy=point.energy_hartree;gradient=point.gradient_hartree_per_bohr;hessian=point.hessian_hartree_per_bohr2
        elif "gradient" in requested:energy,gradient=force_field.energy_gradient(coordinates_angstrom);hessian=None
        else:energy=force_field.energy(coordinates_angstrom);gradient=hessian=None
        return ArchitectEvaluation(energy if "energy" in requested else None,gradient if "gradient" in requested else None,hessian,requested,source="ZAFF_NATIVE_PIC",execution={"runtime":"native_pic","model_level":force_field.metadata.get("model_level"),"physical_field_qualified":False})
    from matrix_zaff.nonbonded import select_mm_runtime_policy

    parameters = dict(force_field.metadata.get("nonbonded_parameters", {}))
    runtime_policy = select_mm_runtime_policy(
        coordinates_angstrom,
        cutoff_angstrom=float(parameters.get("regional_neighbor_cutoff_angstrom", 12.0)),
        neighbor_list_minimum_atoms=int(parameters.get("neighbor_list_minimum_atoms", 64)),
        fmm_minimum_atoms=int(parameters.get("fmm_minimum_atoms", 256)),
        fmm_precision=float(parameters.get("fmm_precision", 1.0e-10)),
    )
    primitive_model = force_field.metadata.get("primitive_residual_model")
    if primitive_model is not None and xyzin is not None:
        raise ValueError("primitive ZAFF residual uses its frozen SMITH primitives; xyzin is unsupported")
    if xyzin is not None:
        if zoom_level is not None:
            raise ValueError("ZAFF anharmonic evaluation does not accept a harmonic zoom level")
        from matrix_zaff.sonic_runtime import ZaffSonicRuntime

        runtime = ZaffSonicRuntime(force_field, xyzin)
        result = _evaluate_sonic_runtime(runtime, coordinates_angstrom, requested)
        return _with_geminal_correction(force_field, coordinates_angstrom, replace(
            result,
            execution={**dict(result.execution), "mm_runtime": runtime_policy.to_dict()},
        ))
    if primitive_model is not None:
        if zoom_level is not None:
            raise ValueError("primitive ZAFF residual does not support harmonic zoom levels")
        from .primitive_residual import PrimitiveResidualRuntime

        primitive_result = PrimitiveResidualRuntime(primitive_model).evaluate(
            coordinates_angstrom, include_hessian="hessian" in requested,
        )
        result = ArchitectEvaluation(
            energy_hartree=primitive_result.energy_hartree if "energy" in requested else None,
            gradient_hartree_per_bohr=(
                primitive_result.gradient_hartree_per_bohr if "gradient" in requested else None
            ),
            hessian_hartree_per_bohr2=(
                primitive_result.hessian_hartree_per_bohr2 if "hessian" in requested else None
            ),
            properties=requested,
            source="ZAFF_FROZEN_SMITH_PRIMITIVES",
            execution={"runtime": "primitive_residual", "mm_runtime": runtime_policy.to_dict()},
        )
        return _with_geminal_correction(force_field, coordinates_angstrom, result)
    seed_model = force_field.metadata.get("diagonal_seed_model")
    if seed_model is not None:
        from matrix_zaff.seed_runtime import ZaffSecondOrderAnchoredSeedRuntime

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != force_field.reference_coordinates_angstrom.shape:
            raise ValueError("evaluation geometry shape differs from the ZAFF model")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("evaluation geometry contains non-finite coordinates")
        displacement_bohr = (
            coordinates - force_field.reference_coordinates_angstrom
        ).reshape(-1) / BOHR_TO_ANGSTROM
        hessian = (
            _zoom_hessian(force_field, zoom_level)
            if force_field.zoom_levels
            else force_field.hessian_hartree_per_bohr2
        )
        reference_gradient = force_field.gradient_reference_hartree_per_bohr
        harmonic_gradient = reference_gradient + hessian @ displacement_bohr
        harmonic_energy = (
            force_field.energy_reference_hartree
            + float(reference_gradient @ displacement_bohr)
            + 0.5 * float(displacement_bohr @ hessian @ displacement_bohr)
        )
        residual_runtime = ZaffSecondOrderAnchoredSeedRuntime(
            seed_model,
            force_field.reference_coordinates_angstrom,
            zoom_level=zoom_level,
        )
        residual = residual_runtime.evaluate(coordinates)
        residual_hessian = (
            residual_runtime.hessian(coordinates) if "hessian" in requested else None
        )
        result = ArchitectEvaluation(
            energy_hartree=(
                harmonic_energy + residual.energy_hartree
                if "energy" in requested
                else None
            ),
            gradient_hartree_per_bohr=(
                harmonic_gradient + residual.gradient_hartree_per_bohr
                if "gradient" in requested
                else None
            ),
            hessian_hartree_per_bohr2=(
                hessian + residual_hessian if "hessian" in requested else None
            ),
            properties=requested,
            execution={
                **dict(residual.execution),
                "runtime": "harmonic_plus_second_order_anchored_seed",
                "mm_runtime": runtime_policy.to_dict(),
            },
        )
        return _with_geminal_correction(force_field, coordinates, result)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != force_field.reference_coordinates_angstrom.shape:
        raise ValueError("evaluation geometry shape differs from the ZAFF model")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("evaluation geometry contains non-finite coordinates")
    displacement_bohr = (coordinates - force_field.reference_coordinates_angstrom).reshape(
        -1
    ) / BOHR_TO_ANGSTROM
    hessian = _zoom_hessian(force_field, zoom_level)
    reference_gradient = force_field.gradient_reference_hartree_per_bohr
    gradient = reference_gradient + hessian @ displacement_bohr
    energy = (
        force_field.energy_reference_hartree
        + float(reference_gradient @ displacement_bohr)
        + 0.5 * float(displacement_bohr @ hessian @ displacement_bohr)
    )
    result = ArchitectEvaluation(
        energy_hartree=energy if "energy" in requested else None,
        gradient_hartree_per_bohr=gradient if "gradient" in requested else None,
        hessian_hartree_per_bohr2=hessian.copy() if "hessian" in requested else None,
        properties=requested,
        execution={
            "backend": "numpy",
            "device": "cpu",
            "precision": "float64",
            "accelerated": False,
            "cpu_validation": True,
            "mm_runtime": runtime_policy.to_dict(),
        },
    )
    return _with_geminal_correction(force_field, coordinates, result)


def _with_geminal_correction(
    force_field: ZaffForceField, coordinates_angstrom: np.ndarray,
    result: ArchitectEvaluation,
) -> ArchitectEvaluation:
    if not force_field.geminal_xh_couplings:
        return result
    reference = force_field.reference_coordinates_angstrom / BOHR_TO_ANGSTROM
    xyz = np.asarray(coordinates_angstrom, dtype=float) / BOHR_TO_ANGSTROM
    energy = 0.0
    gradient = np.zeros(xyz.size)
    hessian = np.zeros((xyz.size, xyz.size))
    for coupling in force_field.geminal_xh_couplings:
        item_energy, item_gradient, item_hessian = coupling.correction(xyz, reference)
        energy += item_energy
        gradient += item_gradient
        hessian += item_hessian
    return replace(
        result,
        energy_hartree=(None if result.energy_hartree is None else result.energy_hartree + energy),
        gradient_hartree_per_bohr=(
            None if result.gradient_hartree_per_bohr is None
            else result.gradient_hartree_per_bohr + gradient
        ),
        hessian_hartree_per_bohr2=(
            None if result.hessian_hartree_per_bohr2 is None
            else result.hessian_hartree_per_bohr2 + hessian
        ),
        execution={**dict(result.execution), "geminal_xh_couplings": len(force_field.geminal_xh_couplings)},
    )


def evaluate_force_field_batch(
    force_field: ZaffForceField,
    geometries_angstrom: Sequence[np.ndarray],
    *,
    properties: Sequence[str] = ("energy",),
    zoom_level: str | None = None,
    device: str | None = None,
    cpu_validation: bool | None = None,
    gpu_min_batch: int | None = None,
    xyzin: Path | str | None = None,
) -> tuple[ArchitectEvaluation, ...]:
    """Evaluate a Cartesian batch using the best validated runtime backend.

    CPU/NumPy ``float64`` is always the reference implementation.  In ``auto``
    mode, CUDA may be used in double precision and Apple MPS may be used in
    mixed precision for sufficiently large energy/gradient batches.  Hessian
    requests remain in double precision and therefore do not use MPS.
    """

    requested = _normalize_properties(properties)
    from .native_pic_field import NativePicField
    if isinstance(force_field,NativePicField):
        if device not in (None,"auto","cpu"):raise ValueError("native PIC batch evaluation currently uses CPU")
        if requested==("energy",) and zoom_level is None and xyzin is None:
            energies=force_field.energies(geometries_angstrom)
            return tuple(ArchitectEvaluation(float(e),None,None,requested,source="ZAFF_NATIVE_PIC",execution={"runtime":"native_pic_batch","device":"cpu"}) for e in energies)
        return tuple(evaluate_force_field(force_field,x,properties=requested,zoom_level=zoom_level,xyzin=xyzin) for x in geometries_angstrom)
    if force_field.geminal_xh_couplings:
        return tuple(
            evaluate_force_field(
                force_field, geometry, properties=requested,
                zoom_level=zoom_level, xyzin=xyzin,
            )
            for geometry in geometries_angstrom
        )
    if xyzin is not None:
        if zoom_level is not None:
            raise ValueError("ZAFF anharmonic batch evaluation does not accept a zoom level")
        # The compiled local-SONIC runtime owns geometry-dependent B and B'
        # transformations.  Keep the public batch contract for LINK and
        # SENTINEL while evaluating each geometry through that exact runtime;
        # harmonic Cartesian fields continue to use the accelerated path.
        return tuple(
            evaluate_force_field(
                force_field,
                geometry,
                properties=requested,
                xyzin=xyzin,
            )
            for geometry in geometries_angstrom
        )
    if force_field.metadata.get("diagonal_seed_model") is not None:
        geometries = tuple(np.asarray(item, dtype=float) for item in geometries_angstrom)
        fast = (
            zoom_level is not None
            and str(zoom_level).strip().upper().replace("-", "_") == "FAST_RIGID_TORSIONAL"
        )
        requested_workers = int(
            os.environ.get(
                "MATRIX_ZAFF_FAST_WORKERS",
                str(max(1, os.cpu_count() or 1)),
            )
        )
        worker_count = min(max(1, requested_workers), max(1, len(geometries)))
        fmm_minimum_atoms = int(
            force_field.metadata.get("nonbonded_parameters", {}).get(
                "fmm_minimum_atoms", 256
            )
        )
        if (
            fast
            and requested == ("energy",)
            and geometries
            and len(force_field.atoms) < fmm_minimum_atoms
        ):
            from matrix_zaff.seed_runtime import (
                ZaffSecondOrderAnchoredSeedRuntime,
                evaluate_zaff_fast_rigid_torsional_batch_energies,
            )

            seed_model = force_field.metadata["diagonal_seed_model"]
            anchor = ZaffSecondOrderAnchoredSeedRuntime(
                seed_model,
                force_field.reference_coordinates_angstrom,
                zoom_level=zoom_level,
            )
            coordinate_batch = np.stack(geometries)
            raw_energies = evaluate_zaff_fast_rigid_torsional_batch_energies(
                seed_model,
                coordinate_batch,
                chunk_size=int(os.environ.get("MATRIX_ZAFF_FAST_BATCH_CHUNK", "256")),
            )
            displacements_bohr = (
                coordinate_batch - force_field.reference_coordinates_angstrom
            ).reshape(len(geometries), -1) / BOHR_TO_ANGSTROM
            seed_taylor_energies = (
                anchor.reference_energy_hartree
                + displacements_bohr
                @ anchor.reference_gradient_hartree_per_bohr
                + 0.5
                * np.einsum(
                    "bi,ij,bj->b",
                    displacements_bohr,
                    anchor.reference_hessian_hartree_per_bohr2,
                    displacements_bohr,
                    optimize=True,
                )
            )
            base_hessian = (
                _zoom_hessian(force_field, zoom_level)
                if force_field.zoom_levels
                else force_field.hessian_hartree_per_bohr2
            )
            base_taylor_energies = (
                force_field.energy_reference_hartree
                + displacements_bohr
                @ force_field.gradient_reference_hartree_per_bohr
                + 0.5
                * np.einsum(
                    "bi,ij,bj->b",
                    displacements_bohr,
                    base_hessian,
                    displacements_bohr,
                    optimize=True,
                )
            )
            energies = base_taylor_energies + raw_energies - seed_taylor_energies
            execution = {
                "backend": "zaff_fast_rigid_torsional",
                "precision": "float64",
                "accelerated": True,
                "polarization_backend": "DISABLED_BY_FAST_RIGID_TORSIONAL_CONTRACT",
                "coordinate_transform_backend": "NONE_DIRECT_CARTESIAN_DIHEDRALS",
                "batch_parallel_backend": "NUMPY_VECTORIZED_CANDIDATE_PAIR_KERNEL",
                "batch_parallel_workers": 1,
                "candidate_vector_width": len(geometries),
                "candidate_count": len(geometries),
                "second_order_anchor": "REFERENCE_CARTESIAN_JET_SUBTRACTED",
            }
            return tuple(
                ArchitectEvaluation(
                    energy_hartree=float(energy),
                    gradient_hartree_per_bohr=None,
                    hessian_hartree_per_bohr2=None,
                    properties=requested,
                    execution=execution,
                )
                for energy in energies
            )

        def evaluate_seed(geometry: np.ndarray) -> ArchitectEvaluation:
            result = evaluate_force_field(
                force_field,
                geometry,
                properties=requested,
                zoom_level=zoom_level,
            )
            return replace(
                result,
                execution={
                    **dict(result.execution),
                    "batch_parallel_backend": (
                        "THREAD_POOL_DETERMINISTIC_MAP" if fast else "SERIAL"
                    ),
                    "batch_parallel_workers": worker_count if fast else 1,
                },
            )

        if not fast or worker_count == 1 or len(geometries) <= 1:
            return tuple(evaluate_seed(geometry) for geometry in geometries)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return tuple(executor.map(evaluate_seed, geometries))
    geometries = _geometry_batch(force_field, geometries_angstrom)
    if geometries.shape[0] == 0:
        return ()
    backend = resolve_compute_backend(
        device,
        workload_size=geometries.shape[0],
        allow_mixed_precision="hessian" not in requested,
        require_float64="hessian" in requested,
        gpu_min_batch=gpu_min_batch,
    )
    hessian = _zoom_hessian(force_field, zoom_level)
    if not backend.accelerated:
        energies, gradients = _evaluate_force_field_batch_numpy(force_field, geometries, hessian)
        execution = {
            **backend.to_dict(),
            "cpu_validation": True,
            "validation_samples": geometries.shape[0],
        }
        return _batch_evaluations(
            energies, gradients, hessian, requested=requested, execution=execution
        )

    try:
        energies, gradients = _evaluate_force_field_batch_torch(
            force_field, geometries, hessian, device=backend.device
        )
    except Exception as exc:
        if _truthy_environment("MATRIX_DEVICE_STRICT", False):
            raise
        cpu_energies, cpu_gradients = _evaluate_force_field_batch_numpy(
            force_field, geometries, hessian
        )
        execution = {
            "schema": "matrix.compute.backend.v1",
            "requested": backend.requested,
            "backend": "numpy",
            "device": "cpu",
            "precision": "float64",
            "available": True,
            "accelerated": False,
            "supports_float64": True,
            "reason": f"GPU runtime fallback: {type(exc).__name__}: {exc}",
            "cpu_validation": True,
            "validation_samples": geometries.shape[0],
        }
        return _batch_evaluations(
            cpu_energies,
            cpu_gradients,
            hessian,
            requested=requested,
            execution=execution,
        )

    validate_gpu = gpu_validation_enabled() if cpu_validation is None else bool(cpu_validation)
    execution = {
        **backend.to_dict(),
        "cpu_validation": validate_gpu,
        "validation_samples": 0,
    }
    if validate_gpu:
        indices = np.unique(
            np.asarray((0, geometries.shape[0] // 2, geometries.shape[0] - 1), dtype=int)
        )
        cpu_energies, cpu_gradients = _evaluate_force_field_batch_numpy(
            force_field, geometries[indices], hessian
        )
        energy_error = float(np.max(np.abs(energies[indices] - cpu_energies)))
        gradient_error = float(np.max(np.abs(gradients[indices] - cpu_gradients)))
        energy_tolerance = _positive_environment_float("MATRIX_GPU_ENERGY_TOLERANCE", 5.0e-6)
        gradient_tolerance = _positive_environment_float("MATRIX_GPU_GRADIENT_TOLERANCE", 5.0e-5)
        execution.update(
            {
                "validation_samples": int(indices.size),
                "validation_energy_max_abs_hartree": energy_error,
                "validation_gradient_max_abs_hartree_per_bohr": gradient_error,
                "validation_energy_tolerance_hartree": energy_tolerance,
                "validation_gradient_tolerance_hartree_per_bohr": gradient_tolerance,
            }
        )
        if energy_error > energy_tolerance or gradient_error > gradient_tolerance:
            energies, gradients = _evaluate_force_field_batch_numpy(
                force_field, geometries, hessian
            )
            execution.update(
                {
                    "backend": "numpy",
                    "device": "cpu",
                    "precision": "float64",
                    "accelerated": False,
                    "supports_float64": True,
                    "reason": "GPU validation exceeded the configured tolerance",
                    "validation_fallback": True,
                }
            )
        else:
            execution["validation_fallback"] = False
    return _batch_evaluations(
        energies, gradients, hessian, requested=requested, execution=execution
    )


def _geometry_batch(
    force_field: ZaffForceField,
    geometries_angstrom: Sequence[np.ndarray],
) -> np.ndarray:
    if isinstance(geometries_angstrom, np.ndarray):
        geometries = np.asarray(geometries_angstrom, dtype=float)
    else:
        values = tuple(np.asarray(item, dtype=float) for item in geometries_angstrom)
        if not values:
            return np.empty((0, *force_field.reference_coordinates_angstrom.shape), dtype=float)
        geometries = np.stack(values)
    if geometries.ndim != 3:
        raise ValueError("ZAFF batch geometries must have shape (npoints, natoms, 3)")
    expected = (geometries.shape[0], *force_field.reference_coordinates_angstrom.shape)
    if geometries.shape != expected:
        raise ValueError("ZAFF batch geometries must have shape (npoints, natoms, 3)")
    if not np.all(np.isfinite(geometries)):
        raise ValueError("ZAFF batch geometries contain non-finite coordinates")
    return geometries


def _evaluate_force_field_batch_numpy(
    force_field: ZaffForceField,
    geometries: np.ndarray,
    hessian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = (geometries - force_field.reference_coordinates_angstrom).reshape(
        geometries.shape[0], -1
    ) / BOHR_TO_ANGSTROM
    reference_gradient = force_field.gradient_reference_hartree_per_bohr
    gradients = displacement @ hessian.T + reference_gradient
    energies = (
        force_field.energy_reference_hartree
        + displacement @ reference_gradient
        + 0.5 * np.sum((displacement @ hessian.T) * displacement, axis=1)
    )
    return np.asarray(energies, dtype=float), np.asarray(gradients, dtype=float)


def _evaluate_force_field_batch_torch(
    force_field: ZaffForceField,
    geometries: np.ndarray,
    hessian: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    dtype = torch.float32 if device == "mps" else torch.float64
    displacement = torch.as_tensor(
        (
            (geometries - force_field.reference_coordinates_angstrom)
            .reshape(geometries.shape[0], -1)
            .copy()
            / BOHR_TO_ANGSTROM
        ),
        dtype=dtype,
        device=device,
    )
    force = torch.as_tensor(hessian.copy(), dtype=dtype, device=device)
    reference_gradient = torch.as_tensor(
        force_field.gradient_reference_hartree_per_bohr.copy(), dtype=dtype, device=device
    )
    harmonic_gradient = displacement @ force.T
    gradients = harmonic_gradient + reference_gradient
    energies = (
        float(force_field.energy_reference_hartree)
        + displacement @ reference_gradient
        + 0.5 * torch.sum(harmonic_gradient * displacement, dim=1)
    )
    return (
        energies.detach().cpu().numpy().astype(float, copy=False),
        gradients.detach().cpu().numpy().astype(float, copy=False),
    )


def _batch_evaluations(
    energies: np.ndarray,
    gradients: np.ndarray,
    hessian: np.ndarray,
    *,
    requested: tuple[str, ...],
    execution: Mapping[str, Any],
) -> tuple[ArchitectEvaluation, ...]:
    return tuple(
        ArchitectEvaluation(
            energy_hartree=float(energies[index]) if "energy" in requested else None,
            gradient_hartree_per_bohr=(
                gradients[index].copy() if "gradient" in requested else None
            ),
            hessian_hartree_per_bohr2=(hessian.copy() if "hessian" in requested else None),
            properties=requested,
            execution=dict(execution),
        )
        for index in range(energies.size)
    )


def _positive_environment_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return float(default)
    return value if np.isfinite(value) and value > 0.0 else float(default)


def _truthy_environment(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def validate_force_field_derivatives(
    force_field: ZaffForceField,
    coordinates_angstrom: np.ndarray | None = None,
    *,
    zoom_level: str | None = None,
    xyzin: Path | str | None = None,
    step_bohr: float | None = None,
    gradient_tolerance: float = 1.0e-8,
    hessian_tolerance: float | None = None,
) -> ArchitectDerivativeValidation:
    """Check analytic gradient and Hessian against centered finite differences.

    When no geometry is supplied, a deterministic nonzero displacement from
    the reference is used so the energy-to-gradient check does not test only a
    stationary point.
    """

    resolved_step_bohr = (
        (5.0e-5 if xyzin is not None else 1.0e-4) if step_bohr is None else float(step_bohr)
    )
    if not np.isfinite(resolved_step_bohr) or resolved_step_bohr <= 0.0:
        raise ValueError("ARCHITECT derivative-validation step must be positive")
    resolved_hessian_tolerance = (
        (2.0e-8 if xyzin is not None else 1.0e-8)
        if hessian_tolerance is None
        else float(hessian_tolerance)
    )
    if gradient_tolerance < 0.0 or resolved_hessian_tolerance < 0.0:
        raise ValueError("ARCHITECT derivative tolerances must be non-negative")
    reference = force_field.reference_coordinates_angstrom
    if reference is None:
        if coordinates_angstrom is None:raise ValueError("native artifact has no reference geometry; provide a validation geometry")
        reference=np.asarray(coordinates_angstrom,float)
    ncart = reference.size
    if coordinates_angstrom is None:
        pattern = np.sin(np.arange(1, ncart + 1, dtype=float))
        probe_bohr = 1.0e-2 * pattern / max(float(np.linalg.norm(pattern)), 1.0)
        coordinates = reference + probe_bohr.reshape(reference.shape) * BOHR_TO_ANGSTROM
    else:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != reference.shape or not np.all(np.isfinite(coordinates)):
            raise ValueError("ARCHITECT validation geometry shape differs from the ZAFF model")
        probe_bohr = (coordinates - reference).reshape(-1) / BOHR_TO_ANGSTROM

    if xyzin is None:

        def evaluate(coordinates: np.ndarray, properties: Sequence[str]) -> ArchitectEvaluation:
            return evaluate_force_field(
                force_field,
                coordinates,
                properties=properties,
                zoom_level=zoom_level,
            )
    else:
        if zoom_level is not None:
            raise ValueError("ZAFF anharmonic validation does not accept a harmonic zoom level")
        from matrix_zaff.sonic_runtime import ZaffSonicRuntime

        runtime = ZaffSonicRuntime(force_field, xyzin)

        def evaluate(coordinates: np.ndarray, properties: Sequence[str]) -> ArchitectEvaluation:
            return _evaluate_sonic_runtime(runtime, coordinates, _normalize_properties(properties))

    analytic = evaluate(coordinates, ("energy", "gradient", "hessian"))
    numeric_gradient = np.empty(ncart, dtype=float)
    numeric_hessian = np.empty((ncart, ncart), dtype=float)
    for index in range(ncart):
        delta = np.zeros(ncart, dtype=float)
        delta[index] = resolved_step_bohr * BOHR_TO_ANGSTROM
        plus = coordinates + delta.reshape(coordinates.shape)
        minus = coordinates - delta.reshape(coordinates.shape)
        plus_result = evaluate(plus, ("energy", "gradient"))
        minus_result = evaluate(minus, ("energy", "gradient"))
        numeric_gradient[index] = (
            float(plus_result.energy_hartree) - float(minus_result.energy_hartree)
        ) / (2.0 * resolved_step_bohr)
        numeric_hessian[:, index] = (
            np.asarray(plus_result.gradient_hartree_per_bohr)
            - np.asarray(minus_result.gradient_hartree_per_bohr)
        ) / (2.0 * resolved_step_bohr)

    gradient_error = numeric_gradient - np.asarray(analytic.gradient_hartree_per_bohr)
    hessian_error = numeric_hessian - np.asarray(analytic.hessian_hartree_per_bohr2)
    gradient_max = float(np.max(np.abs(gradient_error)))
    hessian_max = float(np.max(np.abs(hessian_error)))
    return ArchitectDerivativeValidation(
        zoom_level=(
            "ANHARMONIC_LOCAL_SONIC"
            if xyzin is not None
            else force_field.active_zoom_level
            if zoom_level is None
            else str(zoom_level)
        ),
        step_bohr=resolved_step_bohr,
        probe_displacement_l2_bohr=float(np.linalg.norm(probe_bohr)),
        gradient_max_abs_error_hartree_per_bohr=gradient_max,
        gradient_rms_error_hartree_per_bohr=float(np.sqrt(np.mean(gradient_error**2))),
        hessian_max_abs_error_hartree_per_bohr2=hessian_max,
        hessian_rms_error_hartree_per_bohr2=float(np.sqrt(np.mean(hessian_error**2))),
        hessian_symmetry_max_abs_error_hartree_per_bohr2=float(
            np.max(np.abs(numeric_hessian - numeric_hessian.T))
        ),
        gradient_tolerance_hartree_per_bohr=float(gradient_tolerance),
        hessian_tolerance_hartree_per_bohr2=resolved_hessian_tolerance,
        passed=bool(
            gradient_max <= gradient_tolerance and hessian_max <= resolved_hessian_tolerance
        ),
    )


def _evaluate_sonic_runtime(
    runtime: Any,
    coordinates_angstrom: np.ndarray,
    requested: tuple[str, ...],
) -> ArchitectEvaluation:
    result = runtime.evaluate(
        coordinates_angstrom,
        include_hessian="hessian" in requested,
    )
    return ArchitectEvaluation(
        energy_hartree=result.energy_hartree if "energy" in requested else None,
        gradient_hartree_per_bohr=(
            result.gradient_hartree_per_bohr if "gradient" in requested else None
        ),
        hessian_hartree_per_bohr2=(
            result.hessian_hartree_per_bohr2 if "hessian" in requested else None
        ),
        properties=requested,
        source=result.source,
        execution=result.execution,
    )


def load_force_field(path: Path | str) -> ZaffForceField:
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload,dict) and payload.get("schema")=="matrix.architect.native_pic_field.v1":
        from .native_pic_field import load_native_pic_field
        return load_native_pic_field(path)
    payload = normalize_legacy_zaff_payload(payload)
    if not isinstance(payload, dict):
        raise ValueError("ZAFF force-field file must contain a JSON object")
    return ZaffForceField.from_dict(payload)


def save_force_field(path: Path | str, force_field: ZaffForceField) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    from .native_pic_field import NativePicField,save_native_pic_field
    if isinstance(force_field,NativePicField):return save_native_pic_field(target,force_field)
    target.write_text(
        json.dumps(
            force_field.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _normalize_properties(properties: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(item).strip().lower() for item in properties))
    if not requested or set(requested) - _PROPERTIES:
        raise ValueError("ARCHITECT properties must be energy, gradient and/or hessian")
    return requested


def _zoom_hessian(force_field: ZaffForceField, zoom_level: str | None) -> np.ndarray:
    # FAST_RIGID_TORSIONAL selects the seed runtime policy; it does not
    # define an alternative harmonic Hessian.
    if zoom_level is not None and str(zoom_level).strip().upper().replace("-", "_") == "FAST_RIGID_TORSIONAL":
        zoom_level = None
    requested = force_field.active_zoom_level if zoom_level is None else str(zoom_level)
    if not requested or not force_field.zoom_levels:
        if zoom_level is not None and zoom_level:
            raise ValueError("this ZAFF field does not contain zoom levels")
        return force_field.hessian_hartree_per_bohr2
    for level in force_field.zoom_levels:
        if level.identifier == requested:
            return level.hessian_hartree_per_bohr2
    raise ValueError(f"unknown ZAFF zoom level: {requested}")


# Public compatibility alias.  New code should use ZaffForceField.
ArchitectForceField = ZaffForceField
