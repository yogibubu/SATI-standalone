"""Same-geometry L0/L1/L2 derivative acquisition for ZAFF."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import permutations
from math import factorial, isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from matrix_smith import (
    build_gic_b_matrix,
    evaluate_gic_values,
)
from matrix_zaff.bmatrix_derivative import build_sparse_gic_b_matrix_derivative
from matrix_zaff.bonded import (
    CenteredMorseStretchCoordinate,
    ZaffStretchStretchAnglePotential,
    ZaffTeamPlusValenceTorsionPotential,
)

from .acquisition import (
    BOHR_TO_ANGSTROM,
    ZaffAnharmonicAcquisitionConfig,
    ZaffAnharmonicCoordinate,
    ZaffAnharmonicModel,
    ZaffAnharmonicPointRequest,
    _build_acquisition_requests,
)


ZAFF_MULTILEVEL_REQUEST_SCHEMA = "matrix.zaff.multilevel.request.v1"
ZAFF_MULTILEVEL_RESULT_SCHEMA = "matrix.zaff.multilevel.result.v1"
ZAFF_MULTILEVEL_DERIVATIVE_SCHEMA = "matrix.zaff.multilevel_derivatives.v1"
_SUPPORTED_LEVEL_SETS = {
    ("L0", "L1"),
    ("L1", "L2"),
    ("L0", "L1", "L2"),
}


def _normalise_levels(levels: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(
        level for level in ("L0", "L1", "L2") if level in {str(item).upper() for item in levels}
    )
    if selected not in _SUPPORTED_LEVEL_SETS:
        raise ValueError("levels must be L0+L1, L1+L2, or L0+L1+L2")
    return selected


@dataclass(frozen=True)
class ZaffMultilevelPointRequest:
    """Properties requested at one immutable Cartesian geometry."""

    point_id: str
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    requested_by_level: Mapping[str, tuple[str, ...]]
    coordinate_index: int | None = None
    coordinate_identifier: str = ""
    offset_multiplier: int = 0
    requested_coordinate_value: float | None = None
    achieved_coordinate_value: float | None = None
    schema: str = ZAFF_MULTILEVEL_REQUEST_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "point_id": self.point_id,
            "atoms": list(self.atoms),
            "coordinates_angstrom": np.asarray(self.coordinates_angstrom).tolist(),
            "coordinate_index": self.coordinate_index,
            "coordinate_identifier": self.coordinate_identifier,
            "offset_multiplier": self.offset_multiplier,
            "requested_coordinate_value": self.requested_coordinate_value,
            "achieved_coordinate_value": self.achieved_coordinate_value,
            "same_geometry_for_all_levels": True,
            "requested_by_level": {
                level: list(properties) for level, properties in self.requested_by_level.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffMultilevelPointRequest":
        if payload.get("schema") != ZAFF_MULTILEVEL_REQUEST_SCHEMA:
            raise ValueError("unsupported ZAFF multilevel request schema")
        return cls(
            point_id=str(payload["point_id"]),
            atoms=tuple(str(value) for value in payload["atoms"]),
            coordinates_angstrom=np.asarray(payload["coordinates_angstrom"], dtype=float),
            requested_by_level={
                str(level): tuple(str(value) for value in properties)
                for level, properties in payload["requested_by_level"].items()
            },
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
        )


@dataclass(frozen=True)
class ZaffMultilevelPointResult:
    """Available Cartesian observables for one common geometry."""

    point_id: str
    coordinates_angstrom: np.ndarray
    energy_l1_hartree: float | None = None
    cartesian_gradient_l1_hartree_per_bohr: np.ndarray | None = None
    cartesian_gradient_l0_hartree_per_bohr: np.ndarray | None = None
    cartesian_hessian_l0_hartree_per_bohr2: np.ndarray | None = None
    energy_l2_hartree: float | None = None
    cartesian_gradient_l2_hartree_per_bohr: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ZAFF_MULTILEVEL_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        def array(value: np.ndarray | None, *, matrix: bool = False) -> Any:
            if value is None:
                return None
            data = np.asarray(value, dtype=float)
            return (data if matrix else data.reshape(-1)).tolist()

        return {
            "schema": self.schema,
            "point_id": self.point_id,
            "coordinates_angstrom": np.asarray(self.coordinates_angstrom).tolist(),
            "energy_l1_hartree": self.energy_l1_hartree,
            "cartesian_gradient_l1_hartree_per_bohr": array(
                self.cartesian_gradient_l1_hartree_per_bohr
            ),
            "cartesian_gradient_l0_hartree_per_bohr": array(
                self.cartesian_gradient_l0_hartree_per_bohr
            ),
            "cartesian_hessian_l0_hartree_per_bohr2": array(
                self.cartesian_hessian_l0_hartree_per_bohr2, matrix=True
            ),
            "energy_l2_hartree": self.energy_l2_hartree,
            "cartesian_gradient_l2_hartree_per_bohr": array(
                self.cartesian_gradient_l2_hartree_per_bohr
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffMultilevelPointResult":
        if payload.get("schema") != ZAFF_MULTILEVEL_RESULT_SCHEMA:
            raise ValueError("unsupported ZAFF multilevel result schema")

        def array(name: str) -> np.ndarray | None:
            value = payload.get(name)
            return None if value is None else np.asarray(value, dtype=float)

        return cls(
            point_id=str(payload["point_id"]),
            coordinates_angstrom=np.asarray(payload["coordinates_angstrom"], dtype=float),
            energy_l1_hartree=(
                None
                if payload.get("energy_l1_hartree") is None
                else float(payload["energy_l1_hartree"])
            ),
            cartesian_gradient_l1_hartree_per_bohr=array("cartesian_gradient_l1_hartree_per_bohr"),
            cartesian_gradient_l0_hartree_per_bohr=array("cartesian_gradient_l0_hartree_per_bohr"),
            cartesian_hessian_l0_hartree_per_bohr2=array("cartesian_hessian_l0_hartree_per_bohr2"),
            energy_l2_hartree=(
                None
                if payload.get("energy_l2_hartree") is None
                else float(payload["energy_l2_hartree"])
            ),
            cartesian_gradient_l2_hartree_per_bohr=array("cartesian_gradient_l2_hartree_per_bohr"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ZaffMultilevelDerivativeModel:
    """Hierarchical source tensors and their non-refitted composite."""

    coordinate_identifiers: tuple[str, ...]
    coordinate_steps: np.ndarray
    levels: tuple[str, ...]
    reference_geometry_level: str
    reference_gradients_internal: Mapping[str, np.ndarray]
    harmonic_l0: np.ndarray | None
    harmonic_l1: np.ndarray
    composite_harmonic: np.ndarray
    cubic_l0_full: np.ndarray | None
    cubic_l1_semidiagonal_i_j_j: np.ndarray
    cubic_l2_diagonal: np.ndarray | None
    quartic_l0_semidiagonal_i_k_j_j: np.ndarray | None
    quartic_l1_diagonal: np.ndarray
    quartic_l2_diagonal: np.ndarray | None
    composite_cubic_full: np.ndarray | None
    composite_cubic_semidiagonal_i_j_j: np.ndarray
    composite_quartic_semidiagonal_i_k_j_j: np.ndarray | None
    composite_quartic_diagonal: np.ndarray
    audit: Mapping[str, Any]
    schema: str = ZAFF_MULTILEVEL_DERIVATIVE_SCHEMA

    @property
    def reference_gradient_l0_internal(self) -> np.ndarray | None:
        return self.reference_gradients_internal.get("L0")

    @property
    def reference_gradient_l1_internal(self) -> np.ndarray:
        return self.reference_gradients_internal["L1"]

    @property
    def reference_gradient_l2_internal(self) -> np.ndarray | None:
        return self.reference_gradients_internal.get("L2")

    def to_dict(self) -> dict[str, Any]:
        def array(value: np.ndarray | None) -> Any:
            return None if value is None else np.asarray(value).tolist()

        return {
            "schema": self.schema,
            "coordinate_identifiers": list(self.coordinate_identifiers),
            "coordinate_steps": np.asarray(self.coordinate_steps).tolist(),
            "levels": list(self.levels),
            "reference_geometry_level": self.reference_geometry_level,
            "reference_gradients_internal": {
                level: array(value) for level, value in self.reference_gradients_internal.items()
            },
            "derivatives": {
                "harmonic_L0": array(self.harmonic_l0),
                "harmonic_L1": array(self.harmonic_l1),
                "composite_harmonic": array(self.composite_harmonic),
                "cubic_L0_full": array(self.cubic_l0_full),
                "cubic_L1_semidiagonal_i_j_j": array(self.cubic_l1_semidiagonal_i_j_j),
                "cubic_L2_diagonal": array(self.cubic_l2_diagonal),
                "quartic_L0_semidiagonal_i_k_j_j": array(self.quartic_l0_semidiagonal_i_k_j_j),
                "quartic_L1_diagonal": array(self.quartic_l1_diagonal),
                "quartic_L2_diagonal": array(self.quartic_l2_diagonal),
                "composite_cubic_full": array(self.composite_cubic_full),
                "composite_cubic_semidiagonal_i_j_j": array(
                    self.composite_cubic_semidiagonal_i_j_j
                ),
                "composite_quartic_semidiagonal_i_k_j_j": array(
                    self.composite_quartic_semidiagonal_i_k_j_j
                ),
                "composite_quartic_diagonal": array(self.composite_quartic_diagonal),
            },
            "audit": dict(self.audit),
            "assembly": {
                "available_level_combinations": ["L0+L1", "L1+L2", "L0+L1+L2"],
                "L1_diagonal_quartic": "THREE_ENERGIES_PLUS_THREE_GRADIENTS",
                "L2_diagonal_quartic": (
                    "FIVE_ENERGIES" if self.quartic_l2_diagonal is not None else "NOT_REQUESTED"
                ),
                "fitted_values": False,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffMultilevelDerivativeModel":
        if payload.get("schema") != ZAFF_MULTILEVEL_DERIVATIVE_SCHEMA:
            raise ValueError("unsupported ZAFF multilevel derivative schema")
        derivatives = payload["derivatives"]

        def array(name: str) -> np.ndarray | None:
            value = derivatives.get(name)
            return None if value is None else np.asarray(value, dtype=float)

        return cls(
            coordinate_identifiers=tuple(payload["coordinate_identifiers"]),
            coordinate_steps=np.asarray(payload["coordinate_steps"], dtype=float),
            levels=tuple(payload["levels"]),
            reference_geometry_level=str(payload["reference_geometry_level"]),
            reference_gradients_internal={
                str(level): np.asarray(value, dtype=float)
                for level, value in payload["reference_gradients_internal"].items()
            },
            harmonic_l0=array("harmonic_L0"),
            harmonic_l1=np.asarray(derivatives["harmonic_L1"], dtype=float),
            composite_harmonic=np.asarray(derivatives["composite_harmonic"], dtype=float),
            cubic_l0_full=array("cubic_L0_full"),
            cubic_l1_semidiagonal_i_j_j=np.asarray(
                derivatives["cubic_L1_semidiagonal_i_j_j"], dtype=float
            ),
            cubic_l2_diagonal=array("cubic_L2_diagonal"),
            quartic_l0_semidiagonal_i_k_j_j=array("quartic_L0_semidiagonal_i_k_j_j"),
            quartic_l1_diagonal=np.asarray(derivatives["quartic_L1_diagonal"], dtype=float),
            quartic_l2_diagonal=array("quartic_L2_diagonal"),
            composite_cubic_full=array("composite_cubic_full"),
            composite_cubic_semidiagonal_i_j_j=np.asarray(
                derivatives["composite_cubic_semidiagonal_i_j_j"], dtype=float
            ),
            composite_quartic_semidiagonal_i_k_j_j=array("composite_quartic_semidiagonal_i_k_j_j"),
            composite_quartic_diagonal=np.asarray(
                derivatives["composite_quartic_diagonal"], dtype=float
            ),
            audit=dict(payload.get("audit", {})),
        )


def build_zaff_multilevel_requests(
    definition: Any,
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    plan: ZaffAnharmonicModel,
    *,
    levels: Sequence[str] = ("L0", "L1", "L2"),
    l2_extended_stencil: bool = False,
    config: ZaffAnharmonicAcquisitionConfig = ZaffAnharmonicAcquisitionConfig(),
) -> tuple[ZaffMultilevelPointRequest, ...]:
    """Generate the minimal common geometry set required by selected levels."""

    selected = _normalise_levels(levels)
    if l2_extended_stencil and "L2" not in selected:
        raise ValueError("the extended L2 stencil requires L2")
    base = _build_acquisition_requests(
        definition,
        tuple(str(atom) for atom in atoms),
        np.asarray(reference_coordinates_angstrom, dtype=float),
        plan,
        config,
    )
    output = []
    for request in base:
        multiplier = abs(request.offset_multiplier)
        if request.point_id != "reference" and multiplier > (2 if l2_extended_stencil else 1):
            continue
        requested: dict[str, tuple[str, ...]] = {}
        if request.point_id == "reference" or multiplier <= 1:
            if "L0" in selected:
                requested["L0"] = ("gradient", "hessian")
            requested["L1"] = ("energy", "gradient")
            if "L2" in selected:
                requested["L2"] = (
                    ("energy", "gradient") if request.point_id == "reference" else ("energy",)
                )
        else:
            requested["L2"] = ("energy",)
        output.append(_multilevel_request_from_anharmonic(request, requested))
    return tuple(output)


def cartesian_hessian_to_internal(
    definition: Any,
    coordinates_angstrom: np.ndarray,
    cartesian_gradient_hartree_per_bohr: np.ndarray,
    cartesian_hessian_hartree_per_bohr2: np.ndarray,
    *,
    b_derivative_workers: int = 0,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Remove analytic ``g_q B'`` curvature and invert ``B^T H_q B``."""

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    b_matrix = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
        dtype=float,
    )
    gradient_internal = _cartesian_gradient_to_internal(
        b_matrix, cartesian_gradient_hartree_per_bohr
    )
    derivative = build_sparse_gic_b_matrix_derivative(
        definition,
        coordinates_angstrom=coordinates,
        parallel_workers=b_derivative_workers,
    )
    curvature = derivative.contract_internal_gradient(gradient_internal)
    cartesian = (
        np.asarray(cartesian_hessian_hartree_per_bohr2, dtype=float).reshape(
            (b_matrix.shape[1], b_matrix.shape[1])
        )
        / BOHR_TO_ANGSTROM**2
    )
    residual = 0.5 * (cartesian + cartesian.T) - curvature
    inverse = np.linalg.pinv(b_matrix, rcond=1.0e-10)
    internal = inverse.T @ residual @ inverse
    internal = 0.5 * (internal + internal.T)
    reconstructed = b_matrix.T @ internal @ b_matrix + curvature
    scale = max(float(np.linalg.norm(cartesian)), 1.0e-30)
    return internal, {
        "b_derivative_backend": derivative.backend,
        "b_derivative_nnz": derivative.nnz,
        "b_derivative_density": derivative.density,
        "relative_cartesian_reconstruction_error": float(
            np.linalg.norm(reconstructed - cartesian) / scale
        ),
    }


def compile_zaff_multilevel_derivatives(
    definition: Any,
    coordinates: Sequence[ZaffAnharmonicCoordinate],
    requests: Sequence[ZaffMultilevelPointRequest],
    results: Sequence[ZaffMultilevelPointResult],
    *,
    reference_geometry_level: str,
    stationarity_tolerance: float = 1.0e-7,
    geometry_tolerance_angstrom: float = 2.0e-8,
    b_derivative_workers: int = 0,
) -> ZaffMultilevelDerivativeModel:
    """Compile L0+L1, L1+L2, or L0+L1+L2 without refitting sources."""

    rows = tuple(coordinates)
    count = len(rows)
    if tuple(row.index for row in rows) != tuple(range(count)):
        raise ValueError("multilevel coordinates must have contiguous indices")
    request_by_id = {item.point_id: item for item in requests}
    result_by_id = {item.point_id: item for item in results}
    if set(request_by_id) != set(result_by_id) or "reference" not in request_by_id:
        raise ValueError("multilevel results must cover the request set exactly")
    levels = _normalise_levels(
        tuple({level for request in requests for level in request.requested_by_level})
    )
    level = str(reference_geometry_level).upper()
    if level not in {"L0", "L1", "L2"}:
        raise ValueError("reference_geometry_level must be L0, L1, or L2")
    expected_base = 1 + 2 * count
    extended = any(abs(item.offset_multiplier) == 2 for item in requests)
    expected = expected_base + (2 * count if extended else 0)
    if len(request_by_id) != expected:
        raise ValueError(f"multilevel stencil requires {expected} points")

    reference_geometry = np.asarray(request_by_id["reference"].coordinates_angstrom, dtype=float)
    reference_values = np.asarray(
        evaluate_gic_values(definition, coordinates_angstrom=reference_geometry),
        dtype=float,
    )
    internal_gradients: dict[str, dict[str, np.ndarray]] = {name: {} for name in levels}
    hessians_l0: dict[str, np.ndarray] = {}
    energies: dict[str, dict[str, float]] = {name: {} for name in levels}
    displacements: dict[str, np.ndarray] = {}
    b_audits: dict[str, Mapping[str, Any]] = {}

    for point_id, request in request_by_id.items():
        result = result_by_id[point_id]
        geometry = np.asarray(result.coordinates_angstrom, dtype=float)
        target = np.asarray(request.coordinates_angstrom, dtype=float)
        if (
            geometry.shape != target.shape
            or np.max(np.abs(geometry - target)) > geometry_tolerance_angstrom
        ):
            raise ValueError(f"levels do not share the requested geometry at {point_id}")
        _validate_requested_result(request, result, geometry.size)
        b_matrix = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=geometry).rows,
            dtype=float,
        )
        for source_level, properties in request.requested_by_level.items():
            energy = getattr(result, f"energy_{source_level.lower()}_hartree", None)
            gradient = getattr(
                result,
                f"cartesian_gradient_{source_level.lower()}_hartree_per_bohr",
                None,
            )
            if "energy" in properties:
                energies[source_level][point_id] = float(energy)
            if "gradient" in properties:
                internal_gradients[source_level][point_id] = _cartesian_gradient_to_internal(
                    b_matrix, gradient
                )
        if "L0" in request.requested_by_level:
            hessians_l0[point_id], b_audits[point_id] = cartesian_hessian_to_internal(
                definition,
                geometry,
                result.cartesian_gradient_l0_hartree_per_bohr,
                result.cartesian_hessian_l0_hartree_per_bohr2,
                b_derivative_workers=b_derivative_workers,
            )
        values = np.asarray(
            evaluate_gic_values(definition, coordinates_angstrom=geometry), dtype=float
        )
        displacements[point_id] = values - reference_values

    reference_gradients = {
        source_level: gradients["reference"]
        for source_level, gradients in internal_gradients.items()
    }
    if (
        level in reference_gradients
        and float(np.linalg.norm(reference_gradients[level])) > stationarity_tolerance
    ):
        raise ValueError(f"the declared {level} reference geometry is not stationary")

    harmonic_l1_directed = np.zeros((count, count), dtype=float)
    cubic_l1 = np.zeros((count, count), dtype=float)
    quartic_l1 = np.zeros(count, dtype=float)
    cubic_l0_raw = np.zeros((count, count, count), dtype=float) if "L0" in levels else None
    quartic_l0 = np.zeros((count, count, count), dtype=float) if "L0" in levels else None
    cubic_l2 = np.zeros(count, dtype=float) if "L2" in levels else None
    quartic_l2 = np.zeros(count, dtype=float) if extended else None
    e10 = energies["L1"]["reference"]
    g10 = reference_gradients["L1"]
    e20 = energies.get("L2", {}).get("reference")
    g20 = reference_gradients.get("L2")

    for coordinate in rows:
        j = coordinate.index
        minus = _point_id(j, -1)
        plus = _point_id(j, 1)
        xm = float(displacements[minus][j])
        xp = float(displacements[plus][j])
        nodes3 = np.asarray([xm, 0.0, xp], dtype=float)
        first_weights = _finite_difference_weights(nodes3, 1)
        second_weights = _finite_difference_weights(nodes3, 2)
        gm = internal_gradients["L1"][minus]
        gp = internal_gradients["L1"][plus]
        gradient_samples = np.stack((gm, g10, gp), axis=0)
        harmonic_l1_directed[:, j] = first_weights @ gradient_samples
        cubic_l1[:, j] = second_weights @ gradient_samples
        inactive_minus = displacements[minus].copy()
        inactive_plus = displacements[plus].copy()
        inactive_minus[j] = 0.0
        inactive_plus[j] = 0.0
        # Remove the exact first-order energy contribution from the small
        # inactive-coordinate drift introduced by Cartesian back-transformation.
        # Otherwise that even-in-displacement contamination is amplified as a
        # spurious diagonal quartic term by the one-dimensional Hermite fit.
        l1_derivatives = _hermite_energy_derivatives(
            nodes3,
            np.asarray(
                [
                    energies["L1"][minus] - float(g10 @ inactive_minus),
                    e10,
                    energies["L1"][plus] - float(g10 @ inactive_plus),
                ],
                dtype=float,
            ),
            np.asarray([gm[j], g10[j], gp[j]], dtype=float),
        )
        harmonic_l1_directed[j, j] = l1_derivatives[2]
        cubic_l1[j, j] = l1_derivatives[3]
        quartic_l1[j] = l1_derivatives[4]

        if cubic_l0_raw is not None and quartic_l0 is not None:
            hm = hessians_l0[minus]
            hp = hessians_l0[plus]
            h0 = hessians_l0["reference"]
            hessian_samples = np.stack((hm, h0, hp), axis=0)
            cubic_l0_raw[:, :, j] = np.tensordot(first_weights, hessian_samples, axes=(0, 0))
            quartic_l0[:, :, j] = np.tensordot(second_weights, hessian_samples, axes=(0, 0))
        if cubic_l2 is not None and e20 is not None and g20 is not None and not extended:
            l2_nodes = nodes3
            l2_energies = np.asarray(
                [energies["L2"][minus], e20, energies["L2"][plus]], dtype=float
            )
            l2_derivatives = _energy_with_reference_gradient_derivatives(
                l2_nodes, l2_energies, float(g20[j])
            )
            cubic_l2[j] = l2_derivatives[3]

    harmonic_l1, harmonic_disagreement = _reciprocalize_matrix(harmonic_l1_directed)
    l2_diagonal_fit_audit: Mapping[str, Any] | None = None
    if (
        extended
        and cubic_l2 is not None
        and quartic_l2 is not None
        and e20 is not None
        and g20 is not None
    ):
        cubic_l2, quartic_l2, l2_diagonal_fit_audit = _fit_extended_l2_diagonal_derivatives(
            displacements,
            energies["L2"],
            reference_energy=e20,
            reference_gradient=g20,
            harmonic=harmonic_l1,
        )
    harmonic_l0 = hessians_l0.get("reference")
    composite_harmonic = harmonic_l1.copy() if harmonic_l0 is None else harmonic_l0.copy()
    np.fill_diagonal(composite_harmonic, np.diag(harmonic_l1))
    cubic_l0 = None
    cubic_disagreement = 0.0
    if cubic_l0_raw is not None:
        cubic_l0, cubic_disagreement = _symmetrize_cubic(cubic_l0_raw)
    quartic_disagreement = 0.0
    if quartic_l0 is not None:
        quartic_l0, quartic_disagreement = _reciprocalize_two_pair_quartics(quartic_l0)

    composite_cubic = None if cubic_l0 is None else cubic_l0.copy()
    composite_semidiagonal = cubic_l1.copy()
    if cubic_l2 is not None:
        for j in range(count):
            composite_semidiagonal[j, j] = cubic_l2[j]
    if composite_cubic is not None:
        for i in range(count):
            for j in range(count):
                value = composite_semidiagonal[i, j]
                for indices in set(permutations((i, j, j))):
                    composite_cubic[indices] = value
    composite_quartic = None if quartic_l0 is None else quartic_l0.copy()
    composite_quartic_diagonal = quartic_l2.copy() if quartic_l2 is not None else quartic_l1.copy()
    if composite_quartic is not None:
        for j in range(count):
            composite_quartic[j, j, j] = composite_quartic_diagonal[j]

    return ZaffMultilevelDerivativeModel(
        coordinate_identifiers=tuple(row.identifier for row in rows),
        coordinate_steps=np.asarray([row.step for row in rows], dtype=float),
        levels=levels,
        reference_geometry_level=level,
        reference_gradients_internal=reference_gradients,
        harmonic_l0=harmonic_l0,
        harmonic_l1=harmonic_l1,
        composite_harmonic=composite_harmonic,
        cubic_l0_full=cubic_l0,
        cubic_l1_semidiagonal_i_j_j=cubic_l1,
        cubic_l2_diagonal=cubic_l2,
        quartic_l0_semidiagonal_i_k_j_j=quartic_l0,
        quartic_l1_diagonal=quartic_l1,
        quartic_l2_diagonal=quartic_l2,
        composite_cubic_full=composite_cubic,
        composite_cubic_semidiagonal_i_j_j=composite_semidiagonal,
        composite_quartic_semidiagonal_i_k_j_j=composite_quartic,
        composite_quartic_diagonal=composite_quartic_diagonal,
        audit={
            "same_cartesian_geometries": True,
            "point_count": len(requests),
            "levels": list(levels),
            "reference_stationarity_verified": level in reference_gradients,
            "L2_extended_stencil": extended,
            "L1_diagonal_quartic_source": "THREE_ENERGIES_PLUS_THREE_GRADIENTS",
            "L2_diagonal_quartic_source": "FIVE_ENERGIES" if extended else None,
            "L2_diagonal_fit": l2_diagonal_fit_audit,
            "full_three_index_cubic_available": cubic_l0 is not None,
            "semidiagonal_quartic_available": quartic_l0 is not None,
            "maximum_L1_harmonic_reciprocity_disagreement": harmonic_disagreement,
            "maximum_L0_cubic_permutation_disagreement": cubic_disagreement,
            "maximum_L0_two_pair_quartic_reciprocity_disagreement": quartic_disagreement,
            "maximum_internal_hessian_cartesian_reconstruction_error": max(
                (
                    float(row["relative_cartesian_reconstruction_error"])
                    for row in b_audits.values()
                ),
                default=0.0,
            ),
            "source_values_refitted": False,
        },
    )


def compile_zaff_stretch_stretch_angle_term(
    derivatives: ZaffMultilevelDerivativeModel,
    reference_values: Sequence[float],
    stretch_i: int,
    stretch_j: int,
    angle: int,
    *,
    identifier: str,
    alpha_i_per_angstrom: float = 0.0,
    alpha_j_per_angstrom: float = 0.0,
) -> Mapping[str, Any]:
    """Compile a physical coupled term when L0 supplies full tensors."""

    if derivatives.composite_cubic_full is None or (
        derivatives.composite_quartic_semidiagonal_i_k_j_j is None
    ):
        raise ValueError("stretch--stretch--angle compilation requires L0+L1")
    indices = (int(stretch_i), int(stretch_j), int(angle))
    if len(set(indices)) != 3 or any(index < 0 for index in indices):
        raise ValueError("stretch--stretch--angle coordinates must be distinct")
    values = np.asarray(reference_values, dtype=float).reshape(-1)
    count = len(derivatives.coordinate_identifiers)
    if values.shape != (count,) or any(index >= count for index in indices):
        raise ValueError("stretch--stretch--angle index differs from basis")
    potential = ZaffStretchStretchAnglePotential.from_local_derivatives(
        CenteredMorseStretchCoordinate(values[stretch_i], alpha_i_per_angstrom),
        CenteredMorseStretchCoordinate(values[stretch_j], alpha_j_per_angstrom),
        values[angle],
        float(derivatives.composite_harmonic[stretch_i, stretch_j]),
        float(derivatives.composite_cubic_full[stretch_i, stretch_j, angle]),
        float(derivatives.composite_quartic_semidiagonal_i_k_j_j[stretch_i, stretch_j, angle]),
    )
    return {
        "schema": "matrix.zaff.coupled_term.v1",
        "identifier": str(identifier),
        "coordinate_indices_zero_based": {
            "stretch_i": stretch_i,
            "stretch_j": stretch_j,
            "angle": angle,
        },
        "potential": potential.to_dict(),
        "runtime_embedding": {
            "harmonic_Kij": "ALREADY_PRESENT_IN_BASE_QUADRATIC_MATRIX",
            "added_by_coupled_kernel": ["Phi_ijtheta", "K_ijthetatheta"],
            "double_counting": False,
        },
        "source_levels": {
            "K_ij": "COMPOSITE_HARMONIC",
            "Phi_ijtheta": "L0_HESSIAN_FIRST_DIFFERENCE_FULL_CUBIC",
            "K_ijthetatheta": "L0_HESSIAN_SECOND_DIFFERENCE_SEMIDIAGONAL_QUARTIC",
        },
    }


def compile_zaff_team_plus_valence_torsion_term(
    derivatives: ZaffMultilevelDerivativeModel,
    reference_values: Sequence[float],
    valence: int,
    torsion: int,
    *,
    identifier: str,
    valence_kind: str,
) -> Mapping[str, Any]:
    """Compile one compact TEAM+ bond/angle--torsion coupling from L0 tensors."""

    if derivatives.composite_cubic_full is None or (
        derivatives.composite_quartic_semidiagonal_i_k_j_j is None
    ):
        raise ValueError("TEAM+ valence--torsion compilation requires L0+L1")
    indices = (int(valence), int(torsion))
    if len(set(indices)) != 2 or any(index < 0 for index in indices):
        raise ValueError("TEAM+ valence and torsion coordinates must be distinct")
    values = np.asarray(reference_values, dtype=float).reshape(-1)
    count = len(derivatives.coordinate_identifiers)
    if values.shape != (count,) or any(index >= count for index in indices):
        raise ValueError("TEAM+ valence--torsion index differs from basis")
    potential = ZaffTeamPlusValenceTorsionPotential.from_local_derivatives(
        values[valence],
        values[torsion],
        float(derivatives.composite_harmonic[valence, torsion]),
        float(derivatives.composite_cubic_full[valence, torsion, torsion]),
        float(derivatives.composite_quartic_semidiagonal_i_k_j_j[valence, torsion, torsion]),
        valence_kind=valence_kind,
    )
    return {
        "schema": "matrix.zaff.coupled_term.v1",
        "identifier": str(identifier),
        "coordinate_indices_zero_based": {"valence": valence, "torsion": torsion},
        "potential": potential.to_dict(),
        "runtime_embedding": {
            "harmonic_K_valence_torsion": "ALREADY_PRESENT_IN_BASE_QUADRATIC_MATRIX",
            "added_by_coupled_kernel": [
                "Phi_valence_torsion_torsion",
                "K_valence_torsion_torsion_torsion",
            ],
            "double_counting": False,
        },
        "source_levels": {
            "K_valence_torsion": "COMPOSITE_HARMONIC",
            "Phi_valence_torsion_torsion": "L0_HESSIAN_FIRST_DIFFERENCE_FULL_CUBIC",
            "K_valence_torsion_torsion_torsion": (
                "L0_HESSIAN_SECOND_DIFFERENCE_SEMIDIAGONAL_QUARTIC"
            ),
        },
    }


def attach_zaff_coupled_terms(
    model: ZaffAnharmonicModel,
    *terms: Mapping[str, Any],
) -> ZaffAnharmonicModel:
    return replace(model, coupled_terms=(*model.coupled_terms, *(dict(term) for term in terms)))


def _multilevel_request_from_anharmonic(
    request: ZaffAnharmonicPointRequest,
    requested_by_level: Mapping[str, tuple[str, ...]],
) -> ZaffMultilevelPointRequest:
    return ZaffMultilevelPointRequest(
        point_id=request.point_id,
        atoms=request.atoms,
        coordinates_angstrom=np.asarray(request.coordinates_angstrom, dtype=float),
        requested_by_level=requested_by_level,
        coordinate_index=request.coordinate_index,
        coordinate_identifier=request.coordinate_identifier,
        offset_multiplier=request.offset_multiplier,
        requested_coordinate_value=request.requested_coordinate_value,
        achieved_coordinate_value=request.achieved_coordinate_value,
    )


def _point_id(index: int, multiplier: int) -> str:
    return f"q{index:04d}_{multiplier:+d}h".replace("+", "p").replace("-", "m")


def _cartesian_gradient_to_internal(
    b_matrix: np.ndarray,
    cartesian_gradient_hartree_per_bohr: np.ndarray,
) -> np.ndarray:
    gradient = np.asarray(cartesian_gradient_hartree_per_bohr, dtype=float).reshape(-1)
    if gradient.shape != (b_matrix.shape[1],):
        raise ValueError("Cartesian gradient size differs from frozen SONIC chart")
    internal, *_ = np.linalg.lstsq(b_matrix.T, gradient / BOHR_TO_ANGSTROM, rcond=1.0e-10)
    return np.asarray(internal, dtype=float)


def _validate_requested_result(
    request: ZaffMultilevelPointRequest,
    result: ZaffMultilevelPointResult,
    ncart: int,
) -> None:
    for level, properties in request.requested_by_level.items():
        suffix = level.lower()
        if "energy" in properties:
            value = getattr(result, f"energy_{suffix}_hartree", None)
            if value is None or not isfinite(float(value)):
                raise ValueError(f"missing/non-finite {level} energy at {request.point_id}")
        if "gradient" in properties:
            value = getattr(result, f"cartesian_gradient_{suffix}_hartree_per_bohr", None)
            if value is None or np.asarray(value).size != ncart or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid {level} gradient at {request.point_id}")
        if "hessian" in properties:
            value = getattr(result, f"cartesian_hessian_{suffix}_hartree_per_bohr2", None)
            if (
                value is None
                or np.asarray(value).size != ncart**2
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(f"invalid {level} Hessian at {request.point_id}")


def _reciprocalize_matrix(raw: np.ndarray) -> tuple[np.ndarray, float]:
    result = 0.5 * (raw + raw.T)
    return result, float(np.max(np.abs(raw - raw.T), initial=0.0))


def _finite_difference_weights(nodes: np.ndarray, derivative: int) -> np.ndarray:
    points = np.asarray(nodes, dtype=float)
    degree = len(points) - 1
    matrix = np.asarray([[value**power for value in points] for power in range(degree + 1)])
    target = np.zeros(degree + 1, dtype=float)
    target[derivative] = float(factorial(derivative))
    return np.linalg.solve(matrix, target)


def _hermite_energy_derivatives(
    nodes: np.ndarray,
    energies: np.ndarray,
    gradients: np.ndarray,
) -> dict[int, float]:
    """Derivatives at zero from E and dE/dq at three nearby points."""

    degree = 2 * len(nodes) - 1
    scale = float(np.max(np.abs(nodes)))
    if not isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("Hermite stencil has no finite non-zero displacement")
    scaled_nodes = np.asarray(nodes, dtype=float) / scale
    reference_index = int(np.argmin(np.abs(nodes)))
    reference_energy = float(energies[reference_index])
    reference_gradient = float(gradients[reference_index])
    rows: list[list[float]] = []
    rhs: list[float] = []
    for value, scaled_value, energy, gradient in zip(
        nodes, scaled_nodes, energies, gradients, strict=True
    ):
        rows.append([scaled_value**power for power in range(degree + 1)])
        rhs.append(float(energy) - reference_energy - reference_gradient * value)
        rows.append(
            [0.0]
            + [
                power * scaled_value ** (power - 1)
                for power in range(1, degree + 1)
            ]
        )
        rhs.append(scale * (float(gradient) - reference_gradient))
    coefficients = np.linalg.solve(np.asarray(rows), np.asarray(rhs))
    return {
        order: float(factorial(order) * coefficients[order] / scale**order)
        for order in range(2, 5)
    }


def _energy_with_reference_gradient_derivatives(
    nodes: np.ndarray,
    energies: np.ndarray,
    reference_gradient: float,
) -> dict[int, float]:
    """Interpolate energies plus the exact first derivative at the origin."""

    degree = len(nodes)
    reference_index = int(np.argmin(np.abs(nodes)))
    reference_energy = float(energies[reference_index])
    rows = [[value**power for power in range(degree + 1)] for value in nodes]
    rows.append([0.0, 1.0, *([0.0] * (degree - 1))])
    rhs = [
        float(energy) - reference_energy - reference_gradient * value
        for value, energy in zip(nodes, energies, strict=True)
    ]
    rhs.append(0.0)
    coefficients = np.linalg.solve(np.asarray(rows), np.asarray(rhs))
    return {
        order: float(factorial(order) * coefficients[order])
        for order in range(2, min(4, degree) + 1)
    }


def _fit_extended_l2_diagonal_derivatives(
    displacements: Mapping[str, np.ndarray],
    energies: Mapping[str, float],
    *,
    reference_energy: float,
    reference_gradient: np.ndarray,
    harmonic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Fit diagonal L2 cubic/quartic terms on the realized SONIC points.

    Curvilinear back-transformation reaches the requested active coordinate
    while allowing small, unavoidable drift in the other SONIC coordinates.
    A collection of independent one-dimensional fits aliases that drift into
    the fourth derivative.  The complete extended stencil instead provides an
    overdetermined linear system for all diagonal quadratic, cubic, and
    quartic terms at once, using every realized displacement and the already
    recovered L1 harmonic block.  The fitted quadratic correction is a
    nuisance term: as in the one-dimensional five-point formula, it prevents
    small L1 harmonic errors (or a genuine L1-to-L2 diagonal change) from
    aliasing into the higher derivatives.
    """

    gradient = np.asarray(reference_gradient, dtype=float).reshape(-1)
    hessian = np.asarray(harmonic, dtype=float)
    count = gradient.size
    if hessian.shape != (count, count):
        raise ValueError("L2 diagonal fit harmonic block differs from the SONIC rank")
    point_ids = tuple(sorted(point_id for point_id in energies if point_id != "reference"))
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for point_id in point_ids:
        displacement = np.asarray(displacements[point_id], dtype=float).reshape(-1)
        if displacement.shape != (count,):
            raise ValueError("L2 diagonal fit displacement differs from the SONIC rank")
        rows.append(
            np.concatenate(
                (
                    displacement**2 / 2.0,
                    displacement**3 / 6.0,
                    displacement**4 / 24.0,
                )
            )
        )
        rhs.append(
            float(energies[point_id])
            - float(reference_energy)
            - float(gradient @ displacement)
            - 0.5 * float(displacement @ hessian @ displacement)
        )
    design = np.asarray(rows, dtype=float)
    values = np.asarray(rhs, dtype=float)
    expected_rank = 3 * count
    if design.shape[0] < expected_rank:
        raise ValueError("extended L2 diagonal fit has too few realized stencil points")
    maxima = np.max(np.abs(design), axis=0)
    if np.any(maxima <= np.finfo(float).tiny):
        raise ValueError("extended L2 diagonal fit contains an inactive derivative column")
    column_scaling = 1.0 / maxima
    scaled_design = design * column_scaling[None, :]
    scaled_coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        scaled_design,
        values,
        rcond=1.0e-12,
    )
    if int(rank) != expected_rank:
        raise ValueError(
            f"extended L2 diagonal fit is rank deficient: {int(rank)} < {expected_rank}"
        )
    coefficients = column_scaling * scaled_coefficients
    fitted_residual = scaled_design @ scaled_coefficients - values
    condition = float(singular_values[0] / singular_values[-1])
    audit = {
        "schema": "matrix.zaff.l2_diagonal_fit.v1",
        "method": "GLOBAL_REALIZED_SONIC_DISPLACEMENTS",
        "observation_count": int(design.shape[0]),
        "parameter_count": expected_rank,
        "rank": int(rank),
        "scaled_condition_number": condition,
        "residual_norm_hartree": float(np.linalg.norm(fitted_residual)),
        "least_squares_residual_sum_hartree2": (float(residuals[0]) if len(residuals) else 0.0),
    }
    return coefficients[count : 2 * count], coefficients[2 * count :], audit


def _symmetrize_cubic(raw: np.ndarray) -> tuple[np.ndarray, float]:
    symmetric = np.zeros_like(raw)
    maximum = 0.0
    for i in range(raw.shape[0]):
        for j in range(raw.shape[0]):
            for k in range(raw.shape[0]):
                indices = set(permutations((i, j, k)))
                values = np.asarray([raw[index] for index in indices])
                symmetric[i, j, k] = float(np.mean(values))
                maximum = max(maximum, float(np.ptp(values)))
    return symmetric, maximum


def _reciprocalize_two_pair_quartics(values: np.ndarray) -> tuple[np.ndarray, float]:
    result = np.asarray(values, dtype=float).copy()
    maximum = 0.0
    for i in range(result.shape[0]):
        for j in range(i + 1, result.shape[0]):
            left, right = float(result[i, i, j]), float(result[j, j, i])
            result[i, i, j] = result[j, j, i] = 0.5 * (left + right)
            maximum = max(maximum, abs(left - right))
    return result, maximum
