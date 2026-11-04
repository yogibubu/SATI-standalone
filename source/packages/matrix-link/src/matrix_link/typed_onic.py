"""Native LINK consumption of frozen composite ONIC block contracts.

SMITH owns every coordinate definition and evaluator.  LINK resolves the
checksummed payloads, preserves the frozen block order, applies its existing
finite predictors, and uses the canonical nonlinear corrector for the coupled
Cartesian realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from matrix_smith import (
    CompositeOnicDefinition,
    FragmentRotationAtlas,
    GICDefinition,
    OnicCoordinateBlock,
    SparseBDirectSumCertificate,
    SparseBMatrix,
    certify_sparse_b_direct_sum,
    compact_sparse_b_matrix_sparse,
    evaluate_exponential_map_relative_pose_block,
    evaluate_exponential_map_relative_pose_block_values,
    evaluate_inverse_distance_projector_block,
    evaluate_inverse_distance_projector_block_values,
    evaluate_natural_internal_block,
    evaluate_natural_internal_block_values,
    evaluate_pseudobond_contact_block,
    evaluate_pseudobond_contact_block_values,
    evaluate_symmetry_adapted_cartesian_block,
    evaluate_symmetry_adapted_cartesian_block_values,
    read_onic_block_contract_from_xyzin,
    read_typed_onic_artifact_from_xyzin,
    sparse_full_row_rank_diagnostics,
    sonic_condition_diagnostics,
)

from .fragment_backtransform import direct_fragment_rigid_prediction
from .internal_coordinates import (
    nonlinear_internal_coordinate_step,
    nonlinear_sparse_internal_coordinate_step,
)


LINK_TYPED_ONIC_RUNTIME_SCHEMA = "matrix.link.typed_onic_runtime.v1"
LINK_TYPED_ONIC_DEFAULT_CONTINUATION_INCREMENT = 0.12


@dataclass(frozen=True)
class LinkOnicBlockEvaluation:
    identifier: str
    representation: str
    coordinate_slice: tuple[int, int]
    values: np.ndarray
    b_matrix: SparseBMatrix
    rank: int
    condition_number: float


@dataclass(frozen=True)
class LinkOnicEvaluation:
    values: np.ndarray
    b_matrix: SparseBMatrix
    blocks: tuple[LinkOnicBlockEvaluation, ...]
    direct_sum_certificate: SparseBDirectSumCertificate


@dataclass(frozen=True)
class LinkOnicBlockRealizationDiagnostics:
    identifier: str
    representation: str
    coordinate_slice: tuple[int, int]
    residual_norm: float
    residual_maximum: float
    iterations: int
    rank: int
    condition_number: float
    status: str


@dataclass(frozen=True)
class LinkOnicRealization:
    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    iterations: int
    substeps: int
    converged: bool
    block_diagnostics: tuple[LinkOnicBlockRealizationDiagnostics, ...]
    method: str = "TYPED_ONIC_FINITE_PREDICTOR_NONLINEAR_CORRECTOR"
    schema: str = LINK_TYPED_ONIC_RUNTIME_SCHEMA


class TypedOnicRuntime:
    """Compiled, payload-verified runtime for one frozen composite contract."""

    def __init__(
        self,
        definition: CompositeOnicDefinition,
        *,
        payloads: Mapping[str, GICDefinition] | Sequence[tuple[str, GICDefinition]] = (),
        parallel_workers: int = 1,
    ) -> None:
        if not isinstance(definition, CompositeOnicDefinition):
            raise TypeError("LINK typed ONIC runtime requires a CompositeOnicDefinition")
        if definition.global_audit.status != "PASS":
            raise ValueError(
                "LINK refuses a composite ONIC contract without a passing global audit"
            )
        workers = int(parallel_workers)
        if workers < 1:
            raise ValueError("typed ONIC parallel worker count must be positive")
        items = tuple(payloads.items()) if isinstance(payloads, Mapping) else tuple(payloads)
        payload_by_id: dict[str, GICDefinition] = {}
        for raw_identifier, payload in items:
            identifier = str(raw_identifier)
            if identifier in payload_by_id:
                raise ValueError(f"duplicate typed ONIC payload for block {identifier}")
            if not isinstance(payload, GICDefinition):
                raise TypeError(f"typed ONIC payload {identifier} is not a GICDefinition")
            payload_by_id[identifier] = payload
        block_ids = {block.identifier for block in definition.blocks}
        unknown = sorted(set(payload_by_id) - block_ids)
        if unknown:
            raise ValueError(f"typed ONIC payloads reference unknown blocks: {unknown}")
        required = {
            block.identifier
            for block in definition.blocks
            if block.representation
            in {"NATURAL_INTERNAL", "EXPONENTIAL_MAP", "PSEUDO_BOND_CONTACT"}
        }
        missing = sorted(required - set(payload_by_id))
        if missing:
            raise ValueError(f"typed ONIC runtime is missing frozen payloads for blocks: {missing}")
        representation_by_id = {
            block.identifier: block.representation for block in definition.blocks
        }
        unexpected = sorted(
            identifier
            for identifier in payload_by_id
            if representation_by_id[identifier]
            not in {"NATURAL_INTERNAL", "EXPONENTIAL_MAP", "PSEUDO_BOND_CONTACT"}
        )
        if unexpected:
            raise ValueError(
                "typed ONIC payloads are only valid for delegated natural/pose blocks: "
                f"{unexpected}"
            )
        self.definition = definition
        self.parallel_workers = workers
        self._payload_by_id = payload_by_id
        self._block_by_id = {block.identifier: block for block in definition.blocks}
        self._slices = _block_slices(definition.blocks)
        self._rotation_atlas = {
            identifier: FragmentRotationAtlas(payload_by_id[identifier])
            for identifier in required
            if self._block_by_id[identifier].representation == "EXPONENTIAL_MAP"
        }
        # Evaluating the frozen reference validates payload schema, identity,
        # atom frame, row order and the runtime Jacobian before LINK accepts it.
        reference = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        evaluation = self.evaluate(reference)
        if evaluation.b_matrix.row_count != definition.global_audit.target_rank:
            raise ValueError("typed ONIC runtime row count contradicts the global contract")
        if any(
            item.rank != self._block_by_id[item.identifier].target_rank
            for item in evaluation.blocks
        ):
            raise ValueError("typed ONIC runtime contains a rank-deficient block")

    @classmethod
    def from_xyzin(
        cls,
        path: Path | str,
        *,
        payloads: Mapping[str, GICDefinition] | Sequence[tuple[str, GICDefinition]] = (),
        parallel_workers: int = 1,
    ) -> "TypedOnicRuntime":
        """Compile the canonical serialized typed-block section fail closed."""

        artifact = read_typed_onic_artifact_from_xyzin(Path(path), required=False)
        if artifact is None:
            definition = read_onic_block_contract_from_xyzin(Path(path), required=True)
            assert definition is not None
            embedded_payloads: tuple[tuple[str, GICDefinition], ...] = ()
        else:
            definition = artifact.definition
            embedded_payloads = artifact.payloads
        supplied = tuple(payloads.items()) if isinstance(payloads, Mapping) else tuple(payloads)
        if supplied and embedded_payloads and dict(supplied) != dict(embedded_payloads):
            raise ValueError(
                "explicit typed ONIC payloads contradict the self-contained artifact"
            )
        return cls(
            definition,
            payloads=embedded_payloads or supplied,
            parallel_workers=parallel_workers,
        )

    @property
    def coordinate_count(self) -> int:
        return self.definition.global_audit.target_rank

    @property
    def coordinate_identifiers(self) -> tuple[str, ...]:
        return tuple(
            identifier
            for block in self.definition.blocks
            for identifier in block.coordinate_identifiers
        )

    @property
    def reference_values(self) -> np.ndarray:
        return self.evaluate(self.definition.reference_coordinates_angstrom).values

    def evaluate(
        self,
        coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    ) -> LinkOnicEvaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        expected_shape = (len(self.definition.atom_indices_one_based), 3)
        if coordinates.shape != expected_shape or not np.all(np.isfinite(coordinates)):
            raise ValueError(
                f"typed ONIC coordinates must be a finite array with shape {expected_shape}"
            )
        block_evaluations = tuple(
            self._evaluate_block(block, coordinates) for block in self.definition.blocks
        )
        rows = tuple(row for evaluation in block_evaluations for row in evaluation.b_matrix.rows)
        values = np.concatenate(tuple(item.values for item in block_evaluations))
        b_matrix = SparseBMatrix(
            rows=rows,
            column_count=coordinates.size,
            row_labels=self.coordinate_identifiers,
            backend="matrix-link-typed-onic-ordered-direct-sum.v1",
        )
        separator_rows = tuple(
            row_index
            for block in self.definition.blocks
            if block.kind == "RELATIVE_POSE"
            for row_index in range(*self._slices[block.identifier])
        )
        return LinkOnicEvaluation(
            values=values,
            b_matrix=b_matrix,
            blocks=block_evaluations,
            direct_sum_certificate=certify_sparse_b_direct_sum(
                b_matrix,
                separator_row_indices=separator_rows,
            ),
        )

    def b_prime(
        self,
        coordinates_angstrom: np.ndarray | Sequence[Sequence[float]] | None = None,
        *,
        step_angstrom: float = 1.0e-4,
        zero_tolerance_per_angstrom2: float = 1.0e-10,
        parallel_workers: int = 0,
    ):
        """Return the complete sparse typed-ONIC ``dB/dx`` tensor.

        The established representation-independent ARCHITECT/ZAFF stencil is
        applied to the exact ordered direct-sum evaluator.  This path is kept
        outside ordinary optimization and is intended for full curvilinear
        Hessian transformations and frequency calculations.
        """

        unsupported = tuple(
            block.identifier
            for block in self.definition.blocks
            if block.second_derivative_status != "GENERAL_SPARSE_B_PRIME"
        )
        if unsupported:
            raise ValueError(
                "typed ONIC B-prime is not certified for blocks: "
                + ", ".join(unsupported)
            )
        from matrix_zaff import build_sparse_b_matrix_derivative_numerical

        coordinates = np.asarray(
            self.definition.reference_coordinates_angstrom
            if coordinates_angstrom is None
            else coordinates_angstrom,
            dtype=float,
        )
        return build_sparse_b_matrix_derivative_numerical(
            lambda point: self.evaluate(point).b_matrix,
            coordinates,
            coordinate_labels=self.coordinate_identifiers,
            step_angstrom=step_angstrom,
            zero_tolerance_per_angstrom2=zero_tolerance_per_angstrom2,
            parallel_workers=parallel_workers,
            backend="architect-sparse-general-typed-onic-b-prime.v1",
        )

    def realize(
        self,
        target_values: np.ndarray | Sequence[float],
        *,
        start_coordinates_angstrom: np.ndarray | Sequence[Sequence[float]] | None = None,
        initial_cartesian_from_q: np.ndarray | None = None,
        fixed_atom_indices: tuple[int, ...] = (),
        project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
        tolerance: float = 1.0e-9,
        max_iterations_per_substep: int = 60,
        max_cartesian_step_angstrom: float = 0.25,
        max_continuation_increment: float = LINK_TYPED_ONIC_DEFAULT_CONTINUATION_INCREMENT,
        max_substeps: int = 32,
        matrix_free: bool = False,
    ) -> LinkOnicRealization:
        """Realize one simultaneous typed-block target with existing LINK solvers."""

        target = np.asarray(target_values, dtype=float).reshape(-1)
        if target.shape != (self.coordinate_count,) or not np.all(np.isfinite(target)):
            raise ValueError("typed ONIC target must contain one finite value per coordinate")
        coordinates = np.asarray(
            self.definition.reference_coordinates_angstrom
            if start_coordinates_angstrom is None
            else start_coordinates_angstrom,
            dtype=float,
        ).copy()
        initial_projector = (
            None
            if initial_cartesian_from_q is None
            else np.asarray(initial_cartesian_from_q, dtype=float)
        )
        expected_projector_shape = (coordinates.size, self.coordinate_count)
        if initial_projector is not None and (
            initial_projector.shape != expected_projector_shape
            or not np.all(np.isfinite(initial_projector))
        ):
            raise ValueError(
                "typed ONIC initial Cartesian projector must have shape "
                f"{expected_projector_shape} and contain finite values"
            )
        start = self.evaluate(coordinates)
        continuation_increment = _positive_finite(
            max_continuation_increment,
            "typed ONIC continuation increment",
        )
        substeps = _continuation_substeps(
            self.definition.blocks,
            self._slices,
            target - start.values,
            maximum_increment=continuation_increment,
            maximum_substeps=max_substeps,
        )
        iterations = 0
        converged = False
        final_values = start.values
        final_residual = target - final_values
        for substep in range(1, substeps + 1):
            subtarget = start.values + (substep / float(substeps)) * (target - start.values)
            coordinates = self._apply_cartesian_predictors(
                coordinates,
                subtarget,
                fixed_atom_indices=fixed_atom_indices,
            )
            coordinates = self._apply_pose_predictors(
                coordinates,
                subtarget,
                fixed_atom_indices=fixed_atom_indices,
            )
            if matrix_free:
                corrected = nonlinear_sparse_internal_coordinate_step(
                    coordinates,
                    subtarget,
                    self._sparse_evaluation,
                    evaluate_values=self._values,
                    max_iterations=int(max_iterations_per_substep),
                    tolerance=float(tolerance),
                    max_cartesian_step_angstrom=float(max_cartesian_step_angstrom),
                    fixed_atom_indices=fixed_atom_indices,
                    project_coordinates=project_coordinates,
                )
            else:
                corrected = nonlinear_internal_coordinate_step(
                    coordinates,
                    subtarget,
                    self._dense_evaluation,
                    evaluate_values=self._values,
                    cartesian_from_q=initial_projector,
                    max_iterations=int(max_iterations_per_substep),
                    tolerance=float(tolerance),
                    max_cartesian_step_angstrom=float(max_cartesian_step_angstrom),
                    fixed_atom_indices=fixed_atom_indices,
                    project_coordinates=project_coordinates,
                )
            coordinates = corrected.coordinates_angstrom
            initial_projector = corrected.cartesian_from_q
            iterations += corrected.iterations
            final_values = corrected.values
            final_residual = subtarget - final_values
            if not corrected.converged:
                break
            converged = substep == substeps
        evaluation = self.evaluate(coordinates)
        final_values = evaluation.values
        final_residual = target - final_values
        convergence_tolerance = float(tolerance) * max(
            1.0,
            float(np.linalg.norm(target - start.values)),
        )
        converged = converged and float(np.linalg.norm(final_residual)) <= convergence_tolerance
        return LinkOnicRealization(
            coordinates_angstrom=coordinates,
            values=final_values,
            residual=final_residual,
            iterations=iterations,
            substeps=substeps,
            converged=converged,
            block_diagnostics=tuple(
                _realization_diagnostics(
                    item,
                    final_residual,
                    tolerance=convergence_tolerance,
                    iterations=iterations,
                )
                for item in evaluation.blocks
            ),
        )

    def _evaluate_block(
        self,
        block: OnicCoordinateBlock,
        coordinates: np.ndarray,
    ) -> LinkOnicBlockEvaluation:
        if block.representation == "SYMMETRY_ADAPTED_CARTESIAN":
            result = evaluate_symmetry_adapted_cartesian_block(block, coordinates)
            values = result.coordinate_values_angstrom
            b_matrix = result.b_matrix
        elif block.representation == "INVERSE_DISTANCE_PROJECTOR":
            result = evaluate_inverse_distance_projector_block(block, coordinates)
            values = result.coordinate_values_angstrom
            b_matrix = result.b_matrix
        elif block.representation == "NATURAL_INTERNAL":
            result = evaluate_natural_internal_block(
                block,
                self._payload_by_id[block.identifier],
                coordinates,
                parallel_workers=self.parallel_workers,
            )
            values = result.coordinate_values
            b_matrix = result.b_matrix
        elif block.representation == "EXPONENTIAL_MAP":
            result = evaluate_exponential_map_relative_pose_block(
                block,
                self._payload_by_id[block.identifier],
                coordinates,
                reference_block=self._block_by_id[block.reference_block_id],
                moving_block=self._block_by_id[block.moving_block_id],
                rotation_reference_coordinates=self._rotation_atlas[
                    block.identifier
                ].reference_coordinates,
                parallel_workers=self.parallel_workers,
            )
            values = result.coordinate_values
            dense_b = result.b_matrix.to_dense()
            values, transformed = self._rotation_atlas[block.identifier].transform(values, dense_b)
            assert transformed is not None
            b_matrix = SparseBMatrix.from_dense(
                transformed,
                row_labels=block.coordinate_identifiers,
                backend="matrix-link-relative-pose-rotation-atlas.v1",
            )
        elif block.representation == "PSEUDO_BOND_CONTACT":
            result = evaluate_pseudobond_contact_block(
                block,
                self._payload_by_id[block.identifier],
                coordinates,
                reference_block=self._block_by_id[block.reference_block_id],
                moving_block=self._block_by_id[block.moving_block_id],
                parallel_workers=self.parallel_workers,
            )
            values = result.coordinate_values
            b_matrix = result.b_matrix
        else:
            raise ValueError(
                f"LINK typed ONIC runtime does not support block representation "
                f"{block.representation}"
            )
        if b_matrix.row_count >= 512:
            compact, outside_support = compact_sparse_b_matrix_sparse(
                b_matrix,
                payload_atoms=block.atom_indices_one_based,
            )
            if outside_support > block.rank_absolute_tolerance:
                raise ValueError(
                    f"typed ONIC block {block.identifier} has Cartesian support "
                    "outside its declared atoms"
                )
            nullity = compact.column_count - compact.row_count
            if nullity < 0 or nullity > 6:
                raise ValueError(
                    f"large typed ONIC block {block.identifier} does not expose "
                    "a certifiable rigid-motion gauge"
                )
            owned = np.asarray(block.atom_indices_one_based, dtype=int) - 1
            diagnostics = sparse_full_row_rank_diagnostics(
                compact,
                coordinates[owned],
                tolerance=block.rank_relative_tolerance,
                absolute_tolerance=block.rank_absolute_tolerance,
            )
        else:
            dense = b_matrix.to_dense()
            diagnostics = sonic_condition_diagnostics(
                dense / np.linalg.norm(dense, axis=1)[:, None],
                tolerance=block.rank_relative_tolerance,
                absolute_tolerance=block.rank_absolute_tolerance,
            )
        return LinkOnicBlockEvaluation(
            identifier=block.identifier,
            representation=block.representation,
            coordinate_slice=self._slices[block.identifier],
            values=np.asarray(values, dtype=float),
            b_matrix=b_matrix,
            rank=int(diagnostics["rank"]),
            condition_number=float(diagnostics["condition_number"]),
        )

    def _dense_evaluation(self, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self.evaluate(coordinates)
        return result.values, result.b_matrix.to_dense()

    def _sparse_evaluation(
        self,
        coordinates: np.ndarray,
    ) -> tuple[np.ndarray, SparseBMatrix, SparseBDirectSumCertificate]:
        result = self.evaluate(coordinates)
        return result.values, result.b_matrix, result.direct_sum_certificate

    def _values(self, coordinates: np.ndarray) -> np.ndarray:
        current = np.asarray(coordinates, dtype=float)
        expected_shape = (len(self.definition.atom_indices_one_based), 3)
        if current.shape != expected_shape or not np.all(np.isfinite(current)):
            raise ValueError(
                f"typed ONIC coordinates must be a finite array with shape {expected_shape}"
            )
        return np.concatenate(
            tuple(self._evaluate_block_values(block, current) for block in self.definition.blocks)
        )

    def _evaluate_block_values(
        self,
        block: OnicCoordinateBlock,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """Delegate a true value-only evaluation to the owning SMITH kernel."""

        if block.representation == "SYMMETRY_ADAPTED_CARTESIAN":
            values = evaluate_symmetry_adapted_cartesian_block_values(block, coordinates)
        elif block.representation == "INVERSE_DISTANCE_PROJECTOR":
            values = evaluate_inverse_distance_projector_block_values(block, coordinates)
        elif block.representation == "NATURAL_INTERNAL":
            values = evaluate_natural_internal_block_values(
                block,
                self._payload_by_id[block.identifier],
                coordinates,
            )
        elif block.representation == "EXPONENTIAL_MAP":
            values = evaluate_exponential_map_relative_pose_block_values(
                block,
                self._payload_by_id[block.identifier],
                coordinates,
                reference_block=self._block_by_id[block.reference_block_id],
                moving_block=self._block_by_id[block.moving_block_id],
                rotation_reference_coordinates=self._rotation_atlas[
                    block.identifier
                ].reference_coordinates,
            )
            values, _rows = self._rotation_atlas[block.identifier].transform(values)
        elif block.representation == "PSEUDO_BOND_CONTACT":
            values = evaluate_pseudobond_contact_block_values(
                block,
                self._payload_by_id[block.identifier],
                coordinates,
                reference_block=self._block_by_id[block.reference_block_id],
                moving_block=self._block_by_id[block.moving_block_id],
            )
        else:
            raise ValueError(
                f"LINK typed ONIC runtime does not support block representation "
                f"{block.representation}"
            )
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape != (block.target_rank,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"typed ONIC value-only evaluator returned invalid values for {block.identifier}"
            )
        return array

    def _apply_cartesian_predictors(
        self,
        coordinates: np.ndarray,
        target: np.ndarray,
        *,
        fixed_atom_indices: tuple[int, ...],
    ) -> np.ndarray:
        result = np.asarray(coordinates, dtype=float).copy()
        fixed = {int(atom) for atom in fixed_atom_indices}
        for block in self.definition.blocks:
            if block.representation != "SYMMETRY_ADAPTED_CARTESIAN":
                continue
            block_atoms_zero = set(block.atom_indices_zero_based)
            if fixed.intersection(block_atoms_zero):
                continue
            current = self._evaluate_block(block, result)
            start, stop = self._slices[block.identifier]
            delta = target[start:stop] - current.values
            result += (current.b_matrix.to_dense().T @ delta).reshape(result.shape)
        return result

    def _apply_pose_predictors(
        self,
        coordinates: np.ndarray,
        target: np.ndarray,
        *,
        fixed_atom_indices: tuple[int, ...],
    ) -> np.ndarray:
        result = np.asarray(coordinates, dtype=float).copy()
        fixed = {int(atom) for atom in fixed_atom_indices}
        for block in self.definition.blocks:
            if block.representation != "EXPONENTIAL_MAP":
                continue
            if fixed.intersection(block.atom_indices_zero_based):
                continue
            start, stop = self._slices[block.identifier]

            def evaluate_values(trial: np.ndarray) -> np.ndarray:
                return self._evaluate_block(block, trial).values

            def evaluate_values_subset(
                trial: np.ndarray,
                indices: tuple[int, ...],
            ) -> np.ndarray:
                return evaluate_values(trial)[list(indices)]

            prediction = direct_fragment_rigid_prediction(
                self._payload_by_id[block.identifier],
                result,
                target[start:stop],
                evaluate_values,
                evaluate_values_subset=evaluate_values_subset,
            )
            result = prediction.coordinates_angstrom
        return result


def _block_slices(
    blocks: Sequence[OnicCoordinateBlock],
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    start = 0
    for block in blocks:
        stop = start + block.target_rank
        result[block.identifier] = (start, stop)
        start = stop
    return result


def _continuation_substeps(
    blocks: Sequence[OnicCoordinateBlock],
    slices: Mapping[str, tuple[int, int]],
    delta: np.ndarray,
    *,
    maximum_increment: float,
    maximum_substeps: int,
) -> int:
    limit = int(maximum_substeps)
    if limit < 1:
        raise ValueError("typed ONIC maximum substep count must be positive")
    maximum = 0.0
    for block in blocks:
        if block.representation != "INVERSE_DISTANCE_PROJECTOR":
            continue
        start, stop = slices[block.identifier]
        maximum = max(maximum, float(np.max(np.abs(delta[start:stop]), initial=0.0)))
    return min(limit, max(1, int(np.ceil(maximum / maximum_increment))))


def _realization_diagnostics(
    evaluation: LinkOnicBlockEvaluation,
    residual: np.ndarray,
    *,
    tolerance: float,
    iterations: int,
) -> LinkOnicBlockRealizationDiagnostics:
    start, stop = evaluation.coordinate_slice
    block_residual = np.asarray(residual[start:stop], dtype=float)
    residual_norm = float(np.linalg.norm(block_residual))
    return LinkOnicBlockRealizationDiagnostics(
        identifier=evaluation.identifier,
        representation=evaluation.representation,
        coordinate_slice=evaluation.coordinate_slice,
        residual_norm=residual_norm,
        residual_maximum=float(np.max(np.abs(block_residual), initial=0.0)),
        iterations=int(iterations),
        rank=evaluation.rank,
        condition_number=evaluation.condition_number,
        status="PASS" if residual_norm <= tolerance else "FAIL",
    )


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


__all__ = [
    "LINK_TYPED_ONIC_DEFAULT_CONTINUATION_INCREMENT",
    "LINK_TYPED_ONIC_RUNTIME_SCHEMA",
    "LinkOnicBlockEvaluation",
    "LinkOnicBlockRealizationDiagnostics",
    "LinkOnicEvaluation",
    "LinkOnicRealization",
    "TypedOnicRuntime",
]
