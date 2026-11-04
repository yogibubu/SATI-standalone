"""Minimum-energy crossing (MEX) solver for LINK.

The solver is deliberately backend-neutral: a surface needs only an energy
and a gradient. Hessians are optional and are used only to seed the local
Lagrangian model; subsequent updates are quasi-Newton, so xTB E/G surfaces
are sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


MEX_SCHEMA = "matrix.link.mex_result.v1"
MECI_SCHEMA = "matrix.link.meci_result.v1"


@dataclass(frozen=True)
class MEXSurfaceEvaluation:
    energy_hartree: float
    gradient: np.ndarray
    hessian: np.ndarray | None = None

    def __post_init__(self) -> None:
        energy = float(self.energy_hartree)
        gradient = np.asarray(self.gradient, dtype=float).reshape(-1)
        if not np.isfinite(energy) or gradient.size == 0 or not np.all(np.isfinite(gradient)):
            raise ValueError("MEX surface evaluation must contain finite energy and gradient")
        hessian = None if self.hessian is None else np.asarray(self.hessian, dtype=float)
        if hessian is not None and (
            hessian.shape != (gradient.size, gradient.size) or not np.all(np.isfinite(hessian))
        ):
            raise ValueError("MEX surface Hessian has the wrong shape or non-finite values")
        object.__setattr__(self, "energy_hartree", energy)
        object.__setattr__(self, "gradient", gradient.copy())
        object.__setattr__(self, "hessian", None if hessian is None else 0.5 * (hessian + hessian.T))


MEXSurface = Callable[[np.ndarray], MEXSurfaceEvaluation]


@dataclass(frozen=True)
class MECIManifoldEvaluation:
    """Two adiabatic surfaces and the regular interstate branching vector."""

    energies_hartree: np.ndarray
    gradients: np.ndarray
    interstate_numerator: np.ndarray

    def __post_init__(self) -> None:
        energies = np.array(self.energies_hartree, dtype=float, copy=True)
        gradients = np.array(self.gradients, dtype=float, copy=True)
        interstate = np.array(self.interstate_numerator, dtype=float, copy=True)
        if energies.shape != (2,) or gradients.ndim != 2 or gradients.shape[0] != 2:
            raise ValueError("MECI manifold requires two energies and two gradients")
        if interstate.shape != gradients.shape[1:]:
            raise ValueError("MECI interstate vector has the wrong dimension")
        if not all(np.all(np.isfinite(value)) for value in (energies, gradients, interstate)):
            raise ValueError("MECI manifold values must be finite")
        for value in (energies, gradients, interstate):
            value.setflags(write=False)
        object.__setattr__(self, "energies_hartree", energies)
        object.__setattr__(self, "gradients", gradients)
        object.__setattr__(self, "interstate_numerator", interstate)


MECIManifold = Callable[[np.ndarray], MECIManifoldEvaluation]
MECIAcceptedGeometryCallback = Callable[
    [int, np.ndarray, MECIManifoldEvaluation, np.ndarray], None
]


def meci_manifold_from_link_service(
    service: object,
    *,
    state_ids: tuple[int, int],
    roots: int,
    gap_tolerance_hartree: float = 1.0e-7,
    minimum_principal_overlap: float = 0.70,
    ambiguity_margin: float = 0.02,
) -> MECIManifold:
    """Adapt a resident FAST_TDA LINK service to a matrix-free MECI manifold."""

    session = getattr(service, "_matrix_gfn2_session", None)
    if session is None or not callable(getattr(session, "evaluate_sf_tda_coupling", None)):
        raise TypeError("LINK MECI requires a resident FAST_TDA/MATRIX-GFN2 service")

    def evaluate(q: np.ndarray) -> MECIManifoldEvaluation:
        coordinates = service.coordinates_from_q(np.asarray(q, dtype=float))
        result = session.evaluate_sf_tda_coupling(
            coordinates,
            state_ids=state_ids,
            roots=int(roots),
            gap_tolerance_hartree=float(gap_tolerance_hartree),
            minimum_principal_overlap=float(minimum_principal_overlap),
            ambiguity_margin=float(ambiguity_margin),
        )
        gradients = np.stack(
            tuple(
                service.optimizer_covector_from_cartesian(vector, coordinates)
                for vector in result.gradients_hartree_per_bohr
            )
        )
        interstate = service.optimizer_covector_from_cartesian(
            result.interstate_numerator_hartree_per_bohr, coordinates
        )
        return MECIManifoldEvaluation(result.energies_hartree, gradients, interstate)

    return evaluate


def mex_surface_from_link_service(
    service: object,
    *,
    request_hessian: bool = False,
    tag_prefix: str = "mex-surface",
) -> MEXSurface:
    """Adapt any LINK QM/MM ``GeometryEvaluationService`` to a MEX surface."""

    from .optimizer import optimizer_hessian_from_cartesian
    from .scan import ANGSTROM_TO_BOHR

    evaluation_counter = 0

    def evaluate(q: np.ndarray) -> MEXSurfaceEvaluation:
        nonlocal evaluation_counter
        evaluation_counter += 1
        properties = ("energy", "gradient", "hessian") if request_hessian else ("energy", "gradient")
        result = service.evaluate(
            np.asarray(q, dtype=float),
            tag=f"{tag_prefix}-{evaluation_counter:06d}",
            use_cache=True,
            persist_cache=True,
            requested_properties=properties,
        )
        if result.gradient_hartree_per_bohr is None:
            raise RuntimeError("MEX surface backend did not return a gradient")
        directions_bohr = service.coordinate_directions(result.coordinates_angstrom) / (
            1.0 / ANGSTROM_TO_BOHR
        )
        gradient = directions_bohr @ np.asarray(result.gradient_hartree_per_bohr, dtype=float)
        hessian = None
        if request_hessian and result.hessian_hartree_per_bohr2 is not None:
            hessian = optimizer_hessian_from_cartesian(
                result.hessian_hartree_per_bohr2,
                service.coordinate_model,
            )
        return MEXSurfaceEvaluation(result.energy_hartree, gradient, hessian)

    return evaluate


@dataclass(frozen=True)
class MEXIteration:
    iteration: int
    energy_hartree: float
    gap_hartree: float
    lagrange_multiplier: float
    gradient_inf_norm: float
    step_norm: float


@dataclass(frozen=True)
class MEXResult:
    converged: bool
    coordinates: np.ndarray
    energy_a_hartree: float
    energy_b_hartree: float
    gap_hartree: float
    lagrange_multiplier: float
    approximate_hessian: np.ndarray
    iterations: tuple[MEXIteration, ...]
    schema: str = MEX_SCHEMA


@dataclass(frozen=True)
class MECIIteration:
    iteration: int
    average_energy_hartree: float
    gap_hartree: float
    seam_gradient_inf_norm: float
    branching_rank: int
    step_norm: float


@dataclass(frozen=True)
class MECIResult:
    converged: bool
    coordinates: np.ndarray
    energies_hartree: np.ndarray
    gap_hartree: float
    branching_plane: np.ndarray
    approximate_seam_hessian: np.ndarray
    iterations: tuple[MECIIteration, ...]
    schema: str = MECI_SCHEMA


def _branching_plane(gradient_difference: np.ndarray, interstate: np.ndarray) -> np.ndarray:
    trial = np.column_stack((gradient_difference, interstate))
    basis, triangular = np.linalg.qr(trial, mode="reduced")
    diagonal = np.abs(np.diag(triangular))
    scale = max(float(np.max(diagonal, initial=0.0)), 1.0)
    rank = int(np.sum(diagonal > 1.0e-10 * scale))
    if rank < 2:
        raise ValueError("MECI branching vectors are linearly dependent")
    return basis[:, :2]


def optimize_meci(
    manifold: MECIManifold,
    initial_coordinates: Sequence[float] | np.ndarray,
    *,
    max_iterations: int = 100,
    seam_gradient_tolerance: float = 1.0e-5,
    gap_tolerance: float = 1.0e-6,
    step_tolerance: float = 1.0e-6,
    gap_penalty: float = 10.0,
    maximum_step: float = 0.3,
    accepted_geometry_callback: MECIAcceptedGeometryCallback | None = None,
    initial_hessian: np.ndarray | None = None,
    iteration_offset: int = 0,
) -> MECIResult:
    """Locate a minimum-energy conical intersection in the branching plane.

    The regular vectors ``g = grad(E1-E2)`` and ``h = <1|dH/dR|2>`` define
    the two-dimensional branching plane.  The average-state gradient is
    projected onto its orthogonal complement, while the signed energy gap is
    restored along ``g``. A projected quasi-Newton model of the mean energy
    determines tangent steps, accepted by a gap/energy filter. The model
    is updated with projected mean-gradient differences.
    """

    q = np.asarray(initial_coordinates, dtype=float).reshape(-1).copy()
    if q.size < 2 or not np.all(np.isfinite(q)):
        raise ValueError("MECI initial coordinates must contain at least two finite values")
    if (
        int(max_iterations) < 1
        or min(seam_gradient_tolerance, gap_tolerance, step_tolerance, gap_penalty) <= 0.0
        or maximum_step <= 0.0
    ):
        raise ValueError("MECI convergence and step controls must be positive")
    current = manifold(q)
    if current.gradients.shape[1] != q.size:
        raise ValueError("MECI manifold gradients do not match its coordinates")
    if int(iteration_offset) < 0:
        raise ValueError("MECI iteration offset must be nonnegative")
    if initial_hessian is None:
        hessian = np.eye(q.size)
    else:
        hessian = np.asarray(initial_hessian, dtype=float)
        if hessian.shape != (q.size, q.size) or not np.all(np.isfinite(hessian)):
            raise ValueError("MECI restart Hessian is incompatible with its coordinates")
        hessian = 0.5 * (hessian + hessian.T)
    if accepted_geometry_callback is not None:
        accepted_geometry_callback(int(iteration_offset), q.copy(), current, hessian.copy())
    records: list[MECIIteration] = []
    converged = False
    plane = np.empty((q.size, 2))

    def search_quantities(value: MECIManifoldEvaluation):
        gap = float(value.energies_hartree[1] - value.energies_hartree[0])
        difference = value.gradients[1] - value.gradients[0]
        branching = _branching_plane(difference, value.interstate_numerator)
        average = 0.5 * (value.gradients[0] + value.gradients[1])
        seam = average - branching @ (branching.T @ average)
        difference_norm = float(np.linalg.norm(difference))
        if difference_norm <= np.finfo(float).tiny:
            raise ValueError("MECI energy-gap gradient has zero norm")
        search = seam + float(gap_penalty) * gap * difference / difference_norm
        merit = 0.5 * float(seam @ seam)
        merit += 0.5 * (float(gap_penalty) * gap) ** 2
        return gap, branching, seam, search, merit

    def constrained_step(value, position, model):
        """Build a bounded normal-restoration/tangent quasi-Newton step.

        Normal motion restores the energy gap and tangent motion minimizes
        the mean energy. Their acceptance uses residual reduction or an
        exact-penalty energy merit, with the original convergence gates.
        """
        gap, branching, seam, _, residual = search_quantities(value)
        difference = value.gradients[1] - value.gradients[0]
        squared_norm = float(difference @ difference)
        projector = np.eye(q.size) - branching @ branching.T
        average = 0.5 * (value.gradients[0] + value.gradients[1])
        normal = -gap * difference / squared_norm
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm >= maximum_step:
            normal *= float(maximum_step) / normal_norm
            tangent = np.zeros_like(normal)
        else:
            projected_model = projector @ model @ projector + branching @ branching.T
            eigenvalues, eigenvectors = np.linalg.eigh(projected_model)
            floor = max(1.0e-8, 1.0e-4 * float(np.max(np.abs(eigenvalues))))
            tangent = -projector @ (
                eigenvectors @ ((eigenvectors.T @ seam) / np.maximum(eigenvalues, floor))
            )
            available = np.sqrt(max(0.0, float(maximum_step) ** 2 - normal_norm ** 2))
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > available:
                tangent *= available / tangent_norm
        penalty = max(float(gap_penalty), 1.0 + float(np.linalg.norm(average)) / np.sqrt(squared_norm))
        energy_merit = 0.5 * float(np.sum(value.energies_hartree)) + penalty * abs(gap)
        scale = 1.0
        while scale >= 1.0e-4:
            displacement = normal + scale * tangent
            try:
                candidate_q = position + displacement
                candidate = manifold(candidate_q)
                # A tangent step departs from a curved seam at second order.
                # Correct that departure without requiring an exact Hessian.
                for _ in range(3):
                    trial_gap, _, _, _, _ = search_quantities(candidate)
                    if abs(trial_gap) <= gap_tolerance:
                        break
                    trial_difference = candidate.gradients[1] - candidate.gradients[0]
                    correction = -trial_gap * trial_difference / float(trial_difference @ trial_difference)
                    corrected_q = candidate_q + correction
                    if np.linalg.norm(corrected_q - position) > maximum_step * (1.0 + 1.0e-12):
                        break
                    corrected = manifold(corrected_q)
                    corrected_gap = float(corrected.energies_hartree[1] - corrected.energies_hartree[0])
                    if abs(corrected_gap) >= abs(trial_gap):
                        break
                    candidate_q, candidate = corrected_q, corrected
                trial_gap, trial_plane, trial_seam, _, trial_residual = search_quantities(candidate)
                trial_energy_merit = 0.5 * float(np.sum(candidate.energies_hartree)) + penalty * abs(trial_gap)
                predicted = max(0.0, -float(seam @ (scale * tangent)))
                energy_descent = (
                    trial_energy_merit < energy_merit - 1.0e-4 * predicted
                    and abs(trial_gap) <= max(gap_tolerance, 0.9 * abs(gap))
                    and np.linalg.norm(trial_seam) <= max(2.0 * np.linalg.norm(seam), seam_gradient_tolerance)
                )
                if trial_residual < residual or energy_descent:
                    projection = np.eye(q.size) - trial_plane @ trial_plane.T
                    delta = candidate_q - position
                    delta_average = 0.5 * np.sum(candidate.gradients - value.gradients, axis=0)
                    updated = _bfgs_update(model, projection @ delta, projection @ delta_average)
                    return candidate_q, candidate, updated
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
            scale *= 0.5
        return None

    for local_iteration in range(int(max_iterations)):
        iteration = int(iteration_offset) + local_iteration
        gap, plane, seam, search, merit = search_quantities(current)
        seam_norm = float(np.max(np.abs(seam)))
        if abs(gap) <= gap_tolerance and seam_norm <= seam_gradient_tolerance:
            records.append(
                MECIIteration(iteration, 0.5 * float(np.sum(current.energies_hartree)), gap, seam_norm, 2, 0.0)
            )
            converged = True
            break
        # Use the normal/tangent constrained step as the primary model.
        # A single curvature model must describe seam energy, not a mixture
        # of energy and gap residual fields.
        constrained = constrained_step(current, q, hessian)
        if constrained is not None:
            previous_q = q.copy()
            q, current, hessian = constrained
            norm = float(np.linalg.norm(q - previous_q))
            if accepted_geometry_callback is not None:
                accepted_geometry_callback(iteration + 1, q.copy(), current, hessian.copy())
            recorded_gap, _, recorded_seam, _, _ = search_quantities(current)
            records.append(MECIIteration(
                iteration, 0.5 * float(np.sum(current.energies_hartree)),
                recorded_gap, float(np.max(np.abs(recorded_seam))), 2, norm,
            ))
            if norm <= step_tolerance:
                break
            continue
        try:
            step = -np.linalg.solve(hessian, search)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(hessian) @ search
        norm = float(np.linalg.norm(step))
        if norm > maximum_step:
            step *= float(maximum_step) / norm
            norm = float(maximum_step)
        accepted = False
        scale = 1.0
        previous_q = q.copy()
        previous_search = search.copy()
        while scale >= 1.0e-6:
            trial_q = q + scale * step
            try:
                trial = manifold(trial_q)
                _, _, _, trial_search, trial_merit = search_quantities(trial)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                scale *= 0.5
                continue
            if trial_merit < merit:
                q = trial_q
                current = trial
                delta_q = q - previous_q
                hessian = _bfgs_update(hessian, delta_q, trial_search - previous_search)
                norm *= scale
                accepted = True
                if accepted_geometry_callback is not None:
                    accepted_geometry_callback(
                        iteration + 1, q.copy(), current, hessian.copy()
                    )
                break
            scale *= 0.5
        current_gap, _, current_seam, _, _ = search_quantities(current)
        needs_recovery = (not accepted or norm <= step_tolerance) and not (
            abs(current_gap) <= gap_tolerance
            and float(np.max(np.abs(current_seam))) <= seam_gradient_tolerance
        )
        if needs_recovery:
            recovered = constrained_step(current, q, hessian)
            if recovered is not None:
                previous_q = q.copy()
                q, current, hessian = recovered
                norm = float(np.linalg.norm(q - previous_q))
                accepted = True
                if accepted_geometry_callback is not None:
                    accepted_geometry_callback(iteration + 1, q.copy(), current, hessian.copy())
            elif not accepted:
                norm = 0.0
        # Every row describes the accepted endpoint, including its residuals.
        _, _, recorded_seam, _, _ = search_quantities(current)
        records.append(
            MECIIteration(
                iteration,
                0.5 * float(np.sum(current.energies_hartree)),
                float(current.energies_hartree[1] - current.energies_hartree[0]),
                float(np.max(np.abs(recorded_seam))),
                2,
                norm,
            )
        )
        if norm <= step_tolerance:
            break

    final_gap, plane, seam, _, _ = search_quantities(current)
    if abs(final_gap) <= gap_tolerance and float(np.max(np.abs(seam))) <= seam_gradient_tolerance:
        converged = True
    coordinates = np.array(q, copy=True)
    energies = np.array(current.energies_hartree, copy=True)
    plane = np.array(plane, copy=True)
    hessian = np.array(hessian, copy=True)
    for value in (coordinates, energies, plane, hessian):
        value.setflags(write=False)
    return MECIResult(
        converged,
        coordinates,
        energies,
        final_gap,
        plane,
        hessian,
        tuple(records),
    )


def optimize_mex(
    surface_a: MEXSurface,
    surface_b: MEXSurface,
    initial_coordinates: Sequence[float] | np.ndarray,
    *,
    max_iterations: int = 100,
    gradient_tolerance: float = 1.0e-5,
    gap_tolerance: float = 1.0e-6,
    step_tolerance: float = 1.0e-6,
    penalty: float = 10.0,
    active_indices: Sequence[int] | None = None,
    periodic_periods: Mapping[int, float] | None = None,
    inactive_minimizer: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MEXResult:
    """Minimize two surfaces on an active seam subspace.

    ``inactive_minimizer`` may relax all non-active coordinates at each trial.
    Periodic active coordinates are wrapped to ``[-period/2, period/2)``.
    """

    q = np.asarray(initial_coordinates, dtype=float).reshape(-1).copy()
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("MEX initial coordinates must be finite and non-empty")
    if max_iterations <= 0 or penalty <= 0.0:
        raise ValueError("MEX iteration and penalty settings must be positive")
    active = np.arange(q.size, dtype=int) if active_indices is None else np.asarray(active_indices, dtype=int)
    if active.ndim != 1 or active.size == 0 or np.any(active < 0) or np.any(active >= q.size):
        raise ValueError("MEX active_indices must contain valid coordinate indices")
    if np.unique(active).size != active.size:
        raise ValueError("MEX active_indices must be unique")
    periods = {int(index): float(period) for index, period in dict(periodic_periods or {}).items()}
    if any(index not in set(active.tolist()) for index in periods):
        raise ValueError("periodic MEX coordinates must be active")
    if any(period <= 0.0 or not np.isfinite(period) for period in periods.values()):
        raise ValueError("periodic MEX periods must be finite and positive")

    def prepare(vector: np.ndarray) -> np.ndarray:
        candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
        for index, period in periods.items():
            candidate[index] = (candidate[index] + 0.5 * period) % period - 0.5 * period
        if inactive_minimizer is not None:
            candidate = np.asarray(inactive_minimizer(candidate), dtype=float).reshape(-1)
            if candidate.shape != q.shape or not np.all(np.isfinite(candidate)):
                raise ValueError("inactive_minimizer returned an invalid MEX coordinate vector")
        return candidate

    q = prepare(q)
    first = surface_a(q)
    second = surface_b(q)
    if first.gradient.shape != second.gradient.shape or first.gradient.shape != q.shape:
        raise ValueError("MEX surfaces must return gradients with the initial-coordinate shape")
    hessian = _initial_hessian(first, second)
    inactive = np.ones(q.size, dtype=bool)
    inactive[active] = False
    hessian[inactive, :] = 0.0
    hessian[:, inactive] = 0.0
    hessian[inactive, inactive] = np.eye(int(np.sum(inactive)))
    multiplier = 0.0
    records: list[MEXIteration] = []
    converged = False
    for iteration in range(max_iterations):
        average_energy = 0.5 * (first.energy_hartree + second.energy_hartree)
        gap = first.energy_hartree - second.energy_hartree
        lagrangian_gradient = 0.5 * (first.gradient + second.gradient) + (
            multiplier + penalty * gap
        ) * (first.gradient - second.gradient)
        lagrangian_gradient[inactive] = 0.0
        gradient_norm = float(np.max(np.abs(lagrangian_gradient)))
        if abs(gap) <= gap_tolerance and gradient_norm <= gradient_tolerance:
            converged = True
            records.append(MEXIteration(iteration, average_energy, gap, multiplier, gradient_norm, 0.0))
            break
        try:
            step = -np.linalg.solve(hessian, lagrangian_gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(hessian) @ lagrangian_gradient
        step = np.asarray(step, dtype=float)
        step_norm = float(np.linalg.norm(step))
        if step_norm > 0.5:
            step *= 0.5 / step_norm
            step_norm = 0.5
        accepted = False
        merit = average_energy + multiplier * gap + 0.5 * penalty * gap * gap
        scale = 1.0
        previous_q = q.copy()
        previous_gradient = lagrangian_gradient.copy()
        while scale >= 1.0e-5:
            trial_q = prepare(q + scale * step)
            trial_a = surface_a(trial_q)
            trial_b = surface_b(trial_q)
            trial_gap = trial_a.energy_hartree - trial_b.energy_hartree
            trial_merit = 0.5 * (trial_a.energy_hartree + trial_b.energy_hartree)
            trial_merit += multiplier * trial_gap + 0.5 * penalty * trial_gap * trial_gap
            if trial_merit < merit:
                q, first, second = trial_q, trial_a, trial_b
                trial_gradient = 0.5 * (first.gradient + second.gradient) + (
                    multiplier + penalty * trial_gap
                ) * (first.gradient - second.gradient)
                trial_gradient[inactive] = 0.0
                delta_q = q - previous_q
                delta_gradient = trial_gradient - previous_gradient
                hessian = _bfgs_update(hessian, delta_q, delta_gradient)
                accepted = True
                step_norm *= scale
                break
            scale *= 0.5
        if not accepted:
            step_norm = 0.0
        multiplier += penalty * (first.energy_hartree - second.energy_hartree)
        records.append(
            MEXIteration(
                iteration,
                0.5 * (first.energy_hartree + second.energy_hartree),
                first.energy_hartree - second.energy_hartree,
                multiplier,
                gradient_norm,
                step_norm,
            )
        )
        if (
            step_norm <= step_tolerance
            and abs(first.energy_hartree - second.energy_hartree) <= gap_tolerance
            and gradient_norm <= gradient_tolerance
        ):
            converged = True
            break
    return MEXResult(
        converged=converged,
        coordinates=q,
        energy_a_hartree=first.energy_hartree,
        energy_b_hartree=second.energy_hartree,
        gap_hartree=first.energy_hartree - second.energy_hartree,
        lagrange_multiplier=multiplier,
        approximate_hessian=hessian,
        iterations=tuple(records),
    )


def _initial_hessian(first: MEXSurfaceEvaluation, second: MEXSurfaceEvaluation) -> np.ndarray:
    if first.hessian is not None and second.hessian is not None:
        return 0.5 * (first.hessian + second.hessian)
    return np.eye(first.gradient.size, dtype=float)


def _bfgs_update(hessian: np.ndarray, delta_q: np.ndarray, delta_gradient: np.ndarray) -> np.ndarray:
    curvature = float(delta_gradient @ delta_q)
    if curvature <= 1.0e-12:
        return hessian
    h_delta = hessian @ delta_q
    denominator = float(delta_q @ h_delta)
    if denominator <= 1.0e-12:
        return hessian
    updated = hessian - np.outer(h_delta, h_delta) / denominator
    updated += np.outer(delta_gradient, delta_gradient) / curvature
    return 0.5 * (updated + updated.T)


__all__ = [
    "MECI_SCHEMA",
    "MECIIteration",
    "MECIManifold",
    "MECIManifoldEvaluation",
    "MECIResult",
    "MEX_SCHEMA",
    "MEXIteration",
    "MEXResult",
    "MEXSurfaceEvaluation",
    "mex_surface_from_link_service",
    "meci_manifold_from_link_service",
    "optimize_meci",
    "optimize_mex",
]
