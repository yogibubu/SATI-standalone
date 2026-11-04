"""Bounded-memory quasi-Newton primitives for LINK minimum optimization.

This module deliberately has no coordinate-system or electronic-backend
responsibilities.  It stores secants in one already validated optimizer chart
and returns a direction; the owning optimizer remains responsible for chart
realization, Cartesian trust enforcement, acceptance, and convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np


LBFGS_STATE_SCHEMA = "matrix.link.optimizer.lbfgs_state.v1"
LBFGS_MAXIMUM_SECANT_NORM_RATIO = 1.0e4
LBFGS_MINIMUM_INVERSE_SCALE = 1.0e-6
LBFGS_MAXIMUM_INVERSE_SCALE = 1.0e6


@dataclass(frozen=True)
class StepModelSelection:
    requested: str
    selected: str
    reason: str
    estimated_dense_working_set_bytes: int


def estimated_dense_working_set_bytes(active_coordinates: int) -> int:
    """Conservative step-kernel estimate used only for automatic dispatch.

    The measured dense minimum path retains several square arrays while metric
    and Hessian eigensystems are formed.  Ten float64 matrices is a stable,
    architecture-independent dispatch estimate, not a prediction of process
    RSS or of the separate SONIC construction working set.
    """

    dimension = int(active_coordinates)
    if dimension <= 0:
        raise ValueError("active coordinate count must be positive")
    return 10 * dimension * dimension * np.dtype(np.float64).itemsize


def select_minimum_step_model(
    requested: str,
    *,
    active_coordinates: int,
    minimum_task: bool,
    analytic_gradient_requested: bool,
    joint_coordinate_schedule: bool,
    dynamic_chart_lifecycle: bool,
    minimum_active_coordinates: int,
    dense_memory_limit_bytes: int,
    allow_small_system: bool = False,
) -> StepModelSelection:
    """Select dense RFO or L-BFGS from declared mathematical properties."""

    normalized = (
        str(requested).strip().lower().replace("-", "_").replace("l_bfgs", "lbfgs")
    )
    if normalized not in {"dense_rfo", "quadraticized_rfo", "lbfgs", "auto"}:
        raise ValueError(
            "step_model must be 'dense_rfo', 'quadraticized_rfo', 'lbfgs' or 'auto'"
        )
    dimension = int(active_coordinates)
    minimum_dimension = int(minimum_active_coordinates)
    memory_limit = int(dense_memory_limit_bytes)
    if dimension <= 0 or minimum_dimension <= 0 or memory_limit <= 0:
        raise ValueError("step-model dispatch thresholds must be positive")
    estimated = estimated_dense_working_set_bytes(dimension)
    failed_gates = []
    if not minimum_task:
        failed_gates.append("minimum_task_required")
    if not analytic_gradient_requested:
        failed_gates.append("analytic_gradient_required")
    if not joint_coordinate_schedule:
        failed_gates.append("joint_coordinate_schedule_required")
    # A dynamic chart is safe for L-BFGS: the owner transports the inverse
    # model at each accepted SMITH reframe and clears the secant/GDIIS state.
    # The previous frozen-chart gate predated that transport contract.
    if dimension < minimum_dimension and not allow_small_system:
        failed_gates.append(
            f"active_rank_below_threshold:{dimension}<{minimum_dimension}"
        )
    if estimated < memory_limit and not allow_small_system:
        failed_gates.append(
            f"dense_working_set_below_threshold:{estimated}<{memory_limit}"
        )
    if normalized == "dense_rfo":
        return StepModelSelection(normalized, "dense_rfo", "explicit_dense_rfo", estimated)
    if normalized == "quadraticized_rfo":
        return StepModelSelection(
            normalized,
            "quadraticized_rfo",
            "explicit_quadraticized_rfo",
            estimated,
        )
    if failed_gates:
        reason = ";".join(failed_gates)
        if normalized == "lbfgs":
            raise ValueError("explicit L-BFGS request failed its safety gates: " + reason)
        return StepModelSelection(normalized, "dense_rfo", "automatic_fallback:" + reason, estimated)
    return StepModelSelection(
        normalized,
        "lbfgs",
        "explicit_lbfgs" if normalized == "lbfgs" else "automatic_large_dense_working_set",
        estimated,
    )


@dataclass(frozen=True)
class LBFGSDirection:
    """One limited-memory direction and its auditable safeguard outcome."""

    step: np.ndarray
    status: str
    history_size: int
    directional_derivative: float
    initial_inverse_scale: float


@dataclass
class LimitedMemoryBFGS:
    """L-BFGS inverse-Hessian state confined to one chart/tier epoch."""

    dimension: int
    max_history: int = 36
    curvature_floor: float = 1.0e-10
    chart_epoch: int = 0
    electronic_tier: str = "initial"
    steps: list[np.ndarray] = field(default_factory=list)
    gradient_differences: list[np.ndarray] = field(default_factory=list)
    skipped_updates: int = 0

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        self.max_history = int(self.max_history)
        self.curvature_floor = float(self.curvature_floor)
        self.chart_epoch = int(self.chart_epoch)
        self.electronic_tier = str(self.electronic_tier)
        if self.dimension <= 0:
            raise ValueError("L-BFGS dimension must be positive")
        if self.max_history <= 0:
            raise ValueError("L-BFGS history length must be positive")
        if not np.isfinite(self.curvature_floor) or self.curvature_floor <= 0.0:
            raise ValueError("L-BFGS curvature floor must be finite and positive")
        if len(self.steps) != len(self.gradient_differences):
            raise ValueError("L-BFGS secant arrays must have equal length")
        if len(self.steps) > self.max_history:
            raise ValueError("L-BFGS state exceeds its configured history length")
        self.steps = [self._vector(item, "step") for item in self.steps]
        self.gradient_differences = [
            self._vector(item, "gradient difference") for item in self.gradient_differences
        ]
        for step, gradient_difference in zip(
            self.steps, self.gradient_differences, strict=True
        ):
            if not self._curvature_is_usable(step, gradient_difference):
                raise ValueError("serialized L-BFGS state contains an invalid secant pair")

    def _vector(self, value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1).copy()
        if vector.shape != (self.dimension,):
            raise ValueError(f"L-BFGS {name} has the wrong dimension")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"L-BFGS {name} must be finite")
        return vector

    def _curvature_threshold(self, step: np.ndarray, gradient_difference: np.ndarray) -> float:
        scale = float(np.linalg.norm(step) * np.linalg.norm(gradient_difference))
        return max(self.curvature_floor, self.curvature_floor * scale)

    def _curvature_is_usable(
        self, step: np.ndarray, gradient_difference: np.ndarray
    ) -> bool:
        curvature = float(step @ gradient_difference)
        step_norm = float(np.linalg.norm(step))
        difference_norm = float(np.linalg.norm(gradient_difference))
        secant_norm_ratio = difference_norm / max(
            step_norm,
            np.finfo(float).tiny,
        )
        return bool(
            np.isfinite(curvature)
            and curvature > self._curvature_threshold(step, gradient_difference)
            and np.isfinite(secant_norm_ratio)
            and secant_norm_ratio <= LBFGS_MAXIMUM_SECANT_NORM_RATIO
        )

    def reset_epoch(self, *, chart_epoch: int, electronic_tier: str) -> bool:
        """Clear history when chart identity or electronic accuracy changes."""

        next_epoch = int(chart_epoch)
        next_tier = str(electronic_tier)
        changed = next_epoch != self.chart_epoch or next_tier != self.electronic_tier
        if changed:
            self.steps.clear()
            self.gradient_differences.clear()
            self.chart_epoch = next_epoch
            self.electronic_tier = next_tier
        return changed

    def clear_history(self) -> int:
        """Discard one unreliable quasi-Newton history without changing its epoch."""

        discarded = len(self.steps)
        self.steps.clear()
        self.gradient_differences.clear()
        return discarded

    def transport_epoch(
        self,
        step_transform: np.ndarray | Sequence[Sequence[float]],
        covector_transform: np.ndarray | Sequence[Sequence[float]],
        *,
        chart_epoch: int,
        electronic_tier: str,
    ) -> int:
        """Move valid secants to an equivalent replacement coordinate chart.

        ``step_transform`` maps contravariant displacements from the old chart
        to the new one, whereas ``covector_transform`` maps gradient
        differences.  Their duality is certified before any history is
        retained.  This is the bounded-memory analogue of transporting an
        approximate Hessian during an adaptive chart change.
        """

        step_map = np.asarray(step_transform, dtype=float)
        covector_map = np.asarray(covector_transform, dtype=float)
        expected = (self.dimension, self.dimension)
        if (
            step_map.shape != expected
            or covector_map.shape != expected
            or not np.all(np.isfinite(step_map))
            or not np.all(np.isfinite(covector_map))
        ):
            raise ValueError("L-BFGS chart transport maps must be finite square matrices")
        duality = covector_map.T @ step_map
        if not np.allclose(
            duality,
            np.eye(self.dimension),
            rtol=1.0e-7,
            atol=1.0e-8,
        ):
            raise ValueError("L-BFGS chart transport maps are not dual inverses")

        transported_steps: list[np.ndarray] = []
        transported_differences: list[np.ndarray] = []
        for step, difference in zip(
            self.steps, self.gradient_differences, strict=True
        ):
            next_step = step_map @ step
            next_difference = covector_map @ difference
            if self._curvature_is_usable(next_step, next_difference):
                transported_steps.append(next_step)
                transported_differences.append(next_difference)
            else:
                self.skipped_updates += 1
        self.steps = transported_steps[-self.max_history :]
        self.gradient_differences = transported_differences[-self.max_history :]
        self.chart_epoch = int(chart_epoch)
        self.electronic_tier = str(electronic_tier)
        return len(self.steps)

    def update(
        self,
        step: np.ndarray | Sequence[float],
        gradient_difference: np.ndarray | Sequence[float],
    ) -> str:
        """Append one valid positive-curvature pair, otherwise skip it."""

        try:
            accepted_step = self._vector(step, "step")
            accepted_gradient_difference = self._vector(
                gradient_difference, "gradient difference"
            )
        except ValueError:
            self.skipped_updates += 1
            return "skipped_nonfinite_or_wrong_dimension"
        if not self._curvature_is_usable(accepted_step, accepted_gradient_difference):
            self.skipped_updates += 1
            return "skipped_curvature_gate"
        self.steps.append(accepted_step)
        self.gradient_differences.append(accepted_gradient_difference)
        if len(self.steps) > self.max_history:
            del self.steps[: len(self.steps) - self.max_history]
            del self.gradient_differences[
                : len(self.gradient_differences) - self.max_history
            ]
        return "accepted"

    def hessian_operator(  # noqa: F811
        self,
        *,
        initial_hessian_action: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return the direct BFGS Hessian action without a dense matrix.

        The optimization direction uses the usual inverse two-loop recursion,
        while matrix-free RFO needs the corresponding *direct* Hessian action.
        Replay the same bounded secant history with the rank-two BFGS update;
        only O(m n) vectors are retained for history length ``m``.
        """

        if initial_hessian_action is None:
            scale = 1.0
            if self.steps:
                step = self.steps[-1]
                difference = self.gradient_differences[-1]
                denominator = float(step @ difference)
                if denominator > self.curvature_floor:
                    scale = float(difference @ difference) / denominator

            def base_action(vector: np.ndarray) -> np.ndarray:
                return scale * vector

        else:

            def base_action(vector: np.ndarray) -> np.ndarray:
                result = np.asarray(initial_hessian_action(vector), dtype=float).reshape(-1)
                if result.shape != (self.dimension,) or not np.all(np.isfinite(result)):
                    raise ValueError("L-BFGS initial Hessian action returned an invalid vector")
                return result

        updates: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        for step, difference in zip(
            self.steps, self.gradient_differences, strict=True
        ):
            hessian_step = base_action(step)
            for previous_step, previous_difference, previous_hessian_step, denominator, curvature in updates:
                hessian_step = (
                    hessian_step
                    - previous_hessian_step
                    * float(previous_step @ hessian_step)
                    / denominator
                    + previous_difference
                    * float(previous_difference @ step)
                    / curvature
                )
            denominator = float(step @ hessian_step)
            curvature = float(step @ difference)
            threshold = self._curvature_threshold(step, hessian_step)
            if not np.isfinite(denominator) or denominator <= threshold:
                continue
            updates.append((step, difference, hessian_step, denominator, curvature))

        def action(vector: np.ndarray) -> np.ndarray:
            candidate = self._vector(vector, "Hessian-action vector")
            result = base_action(candidate)
            for step, difference, hessian_step, denominator, curvature in updates:
                result = (
                    result
                    - hessian_step * float(step @ result) / denominator
                    + difference * float(difference @ candidate) / curvature
                )
            return result

        return action

    def direction(
        self,
        gradient: np.ndarray | Sequence[float],
        *,
        active_mask: np.ndarray | Sequence[bool] | None = None,
        fallback_inverse_diagonal: np.ndarray | Sequence[float] | None = None,
        initial_inverse_action: Callable[[np.ndarray], np.ndarray] | None = None,
        enforce_descent: bool = True,
    ) -> LBFGSDirection:
        """Apply the two-loop recursion, normally with a strict descent gate.

        Controlled GDIIS uses ``enforce_descent=False`` so every historical
        residual sees the same linear inverse, without RHS-dependent fallback.
        """

        grad = self._vector(gradient, "gradient")
        if active_mask is None:
            mask = np.ones(self.dimension, dtype=bool)
        else:
            mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if mask.shape != (self.dimension,):
                raise ValueError("L-BFGS active mask has the wrong dimension")
        projected_gradient = np.where(mask, grad, 0.0)
        if not np.any(projected_gradient):
            return LBFGSDirection(
                np.zeros_like(projected_gradient),
                "zero_projected_gradient",
                len(self.steps),
                0.0,
                1.0,
            )

        inverse_scale = 1.0
        if self.steps:
            newest_step = self.steps[-1]
            newest_difference = self.gradient_differences[-1]
            denominator = float(newest_difference @ newest_difference)
            if denominator > self.curvature_floor:
                inverse_scale = float(newest_step @ newest_difference) / denominator
        if not np.isfinite(inverse_scale) or inverse_scale <= 0.0:
            inverse_scale = 1.0
        inverse_scale = float(
            np.clip(
                inverse_scale,
                LBFGS_MINIMUM_INVERSE_SCALE,
                LBFGS_MAXIMUM_INVERSE_SCALE,
            )
        )

        work = projected_gradient.copy()
        alphas: list[float] = []
        inverse_curvatures: list[float] = []
        pairs = list(zip(self.steps, self.gradient_differences, strict=True))
        for step, gradient_difference in reversed(pairs):
            inverse_curvature = 1.0 / float(step @ gradient_difference)
            alpha = inverse_curvature * float(step @ work)
            work -= alpha * gradient_difference
            alphas.append(alpha)
            inverse_curvatures.append(inverse_curvature)
        if initial_inverse_action is None:
            inverse_action = inverse_scale * work
        else:
            inverse_action = np.asarray(initial_inverse_action(work), dtype=float).reshape(-1)
            if inverse_action.shape != (self.dimension,) or not np.all(
                np.isfinite(inverse_action)
            ):
                raise ValueError("L-BFGS initial inverse action returned an invalid vector")
            inverse_scale = 1.0
        for (step, gradient_difference), alpha, inverse_curvature in zip(
            pairs,
            reversed(alphas),
            reversed(inverse_curvatures),
            strict=True,
        ):
            beta = inverse_curvature * float(gradient_difference @ inverse_action)
            inverse_action += step * (alpha - beta)
        candidate = np.where(mask, -inverse_action, 0.0)
        derivative = float(projected_gradient @ candidate)
        descent_limit = -np.finfo(float).eps * max(
            1.0,
            float(np.linalg.norm(projected_gradient) * np.linalg.norm(candidate)),
        )
        if np.all(np.isfinite(candidate)) and (not enforce_descent or derivative < descent_limit):
            return LBFGSDirection(
                candidate,
                (
                    "davidson_preconditioned_lbfgs_two_loop"
                    if initial_inverse_action is not None and pairs
                    else "davidson_preconditioned_steepest_descent"
                    if initial_inverse_action is not None
                    else "lbfgs_two_loop"
                    if pairs
                    else "scaled_steepest_descent"
                ),
                len(pairs),
                derivative,
                inverse_scale,
            )

        if fallback_inverse_diagonal is None:
            fallback = -projected_gradient
        else:
            diagonal = self._vector(fallback_inverse_diagonal, "fallback inverse diagonal")
            if np.any(diagonal <= 0.0):
                raise ValueError("L-BFGS fallback inverse diagonal must be positive")
            fallback = -projected_gradient * diagonal
        fallback = np.where(mask, fallback, 0.0)
        fallback_derivative = float(projected_gradient @ fallback)
        if not np.all(np.isfinite(fallback)) or fallback_derivative >= 0.0:
            raise FloatingPointError("L-BFGS could not construct a finite descent direction")
        return LBFGSDirection(
            fallback,
            "preconditioned_steepest_descent_fallback",
            len(pairs),
            fallback_derivative,
            inverse_scale,
        )

    def hessian_operator(  # noqa: F811
        self,
        *,
        initial_hessian_action: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Compile the direct limited-memory BFGS Hessian action.

        The ordinary two-loop recursion applies the inverse approximation.
        Matrix-free restricted-step RFO instead needs products with the direct
        Hessian.  Replay the same retained secants through

        ``B+ = B - (Bs)(Bs).T/(s.T Bs) + yy.T/(y.T s)``.

        At most ``max_history`` update vectors are retained, so both storage
        and work remain linear in the optimizer dimension for fixed history.
        """

        updates: list[tuple[np.ndarray, np.ndarray, float, float]] = []

        def apply_updates(value: np.ndarray) -> np.ndarray:
            candidate = self._vector(value, "Hessian-action vector")
            result = np.asarray(initial_hessian_action(candidate), dtype=float).reshape(-1)
            if result.shape != (self.dimension,) or not np.all(np.isfinite(result)):
                raise ValueError("initial Hessian action returned an invalid vector")
            for bs, difference, sbs, ys in updates:
                result = (
                    result
                    - bs * (float(bs @ candidate) / sbs)
                    + difference * (float(difference @ candidate) / ys)
                )
            return result

        for step, difference in zip(self.steps, self.gradient_differences, strict=True):
            bs = apply_updates(step)
            sbs = float(step @ bs)
            ys = float(difference @ step)
            threshold = self._curvature_threshold(step, difference)
            if (
                not np.all(np.isfinite(bs))
                or not np.isfinite(sbs)
                or not np.isfinite(ys)
                or sbs <= threshold
                or ys <= threshold
            ):
                continue
            updates.append((bs, difference, sbs, ys))

        def operator(value: np.ndarray) -> np.ndarray:
            result = apply_updates(value)
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("limited-memory BFGS Hessian action is non-finite")
            return result

        return operator

    def hessian_action(
        self,
        vector: np.ndarray | Sequence[float],
        *,
        initial_hessian_action: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Apply one compiled direct L-BFGS Hessian product."""

        return self.hessian_operator(
            initial_hessian_action=initial_hessian_action,
        )(np.asarray(vector, dtype=float))

    def to_json(self) -> dict[str, object]:
        return {
            "schema": LBFGS_STATE_SCHEMA,
            "dimension": self.dimension,
            "max_history": self.max_history,
            "curvature_floor": self.curvature_floor,
            "chart_epoch": self.chart_epoch,
            "electronic_tier": self.electronic_tier,
            "skipped_updates": self.skipped_updates,
            "steps": [item.tolist() for item in self.steps],
            "gradient_differences": [
                item.tolist() for item in self.gradient_differences
            ],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "LimitedMemoryBFGS":
        if payload.get("schema") != LBFGS_STATE_SCHEMA:
            raise ValueError("unsupported L-BFGS state schema")
        steps = payload.get("steps")
        gradient_differences = payload.get("gradient_differences")
        if not isinstance(steps, list) or not isinstance(gradient_differences, list):
            raise ValueError("L-BFGS state secant arrays must be lists")
        return cls(
            dimension=int(payload["dimension"]),
            max_history=int(payload["max_history"]),
            curvature_floor=float(payload["curvature_floor"]),
            chart_epoch=int(payload["chart_epoch"]),
            electronic_tier=str(payload["electronic_tier"]),
            skipped_updates=int(payload.get("skipped_updates", 0)),
            steps=[np.asarray(item, dtype=float) for item in steps],
            gradient_differences=[
                np.asarray(item, dtype=float) for item in gradient_differences
            ],
        )
