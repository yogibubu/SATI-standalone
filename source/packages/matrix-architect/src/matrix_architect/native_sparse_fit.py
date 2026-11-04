"""Sparse selection of native ZAFF refinements from energy/gradient rows.

The caller supplies a physically defined harmonic/diagonal baseline and columns
of native-coordinate functions evaluated on acquired geometries.  This module
only fits the remaining terms; it never uses a quartic reference as a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import solve_triangular


def native_zaff_candidate_powers(
    primitive_atoms: Sequence[Sequence[int]],
    *,
    excluded_indices: Sequence[int] = (),
    degrees: tuple[int, ...] = (3, 4),
    maximum_active_primitives: int = 2,
    require_connected: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate connected native SAM refinements without choosing a molecule by hand.

    Every monomial starts at cubic order, so the fitted reference gradient and
    complete quadratic field remain unchanged.  Mixed supports must be
    connected through shared atoms by default. Set ``require_connected=False``
    when completing the observable cubic rank requires nonlocal products.
    Periodic LAM coordinates are excluded by
    the caller and fitted from their separate scans.
    """

    atoms = tuple(frozenset(int(atom) for atom in row) for row in primitive_atoms)
    if not atoms or any(not row for row in atoms):
        raise ValueError("each ZAFF primitive needs at least one atom")
    if not degrees or any(degree < 3 for degree in degrees):
        raise ValueError("native ZAFF refinements must start at cubic order")
    if maximum_active_primitives < 1:
        raise ValueError("maximum active primitive count must be positive")
    excluded = frozenset(int(index) for index in excluded_indices)
    if any(index < 0 or index >= len(atoms) for index in excluded):
        raise ValueError("excluded primitive index is out of range")
    available = tuple(index for index in range(len(atoms)) if index not in excluded)
    powers: set[tuple[int, ...]] = set()
    for degree in degrees:
        for active_count in range(1, min(degree, maximum_active_primitives) + 1):
            for active in combinations(available, active_count):
                if active_count > 1 and require_connected:
                    reached = {active[0]}
                    while True:
                        neighbors = {
                            index for index in active if index not in reached
                            and any(atoms[index] & atoms[known] for known in reached)
                        }
                        if not neighbors:
                            break
                        reached.update(neighbors)
                    if len(reached) != active_count:
                        continue
                # Compositions of degree into strictly positive exponents.
                for cuts in combinations(range(1, degree), active_count - 1):
                    bounds = (0, *cuts, degree)
                    power = [0] * len(atoms)
                    for part, index in enumerate(active):
                        power[index] = bounds[part + 1] - bounds[part]
                    powers.add(tuple(power))
    return tuple(sorted(powers, key=lambda item: (sum(item), item)))


def native_zaff_value_gradient(
    displacement: np.ndarray,
    reference: np.ndarray,
    primitive_from_modal: np.ndarray,
    powers: tuple[int, ...],
    kinds: tuple[str, ...],
) -> tuple[float, np.ndarray]:
    """Evaluate a native 1/r, cosine, or sine monomial and its modal gradient."""

    x = np.asarray(displacement, dtype=float)
    r = np.asarray(reference, dtype=float)
    transform = np.asarray(primitive_from_modal, dtype=float)
    count = len(x)
    if (r.shape != (count,) or transform.ndim != 2 or transform.shape[0] != count
            or len(powers) != count or len(kinds) != count
            or any(power < 0 for power in powers)):
        raise ValueError("native ZAFF coordinate dimensions are inconsistent")
    values = np.empty(count)
    first = np.empty(count)
    for i, kind in enumerate(kinds):
        if kind == "inverse_distance":
            if r[i] <= 0 or r[i] + x[i] <= 0:
                raise ValueError("native inverse distance must remain positive")
            values[i] = 1 / (r[i] + x[i]) - 1 / r[i]
            first[i] = -1 / (r[i] + x[i])**2
        elif kind == "cosine":
            values[i] = np.cos(r[i] + x[i]) - np.cos(r[i])
            first[i] = -np.sin(r[i] + x[i])
        elif kind == "sine":
            values[i] = np.sin(x[i])
            first[i] = np.cos(x[i])
        else:
            raise ValueError(f"unsupported native ZAFF coordinate kind: {kind}")
    active = [i for i, power in enumerate(powers) if power]
    value = float(np.prod([values[i] ** powers[i] for i in active]))
    gradient = np.zeros(count)
    for i in active:
        power = powers[i]
        gradient[i] = (power * values[i]**(power - 1) * first[i]
                       * np.prod([values[j]**powers[j] for j in active if j != i]))
    return value, transform.T @ gradient


def native_zaff_hessian(
    displacement: np.ndarray,
    reference: np.ndarray,
    primitive_from_modal: np.ndarray,
    powers: tuple[int, ...],
    kinds: tuple[str, ...],
) -> np.ndarray:
    """Analytic modal Hessian of one native ZAFF monomial."""

    x = np.asarray(displacement, dtype=float)
    r = np.asarray(reference, dtype=float)
    transform = np.asarray(primitive_from_modal, dtype=float)
    count = len(x)
    if (r.shape != (count,) or transform.ndim != 2 or transform.shape[0] != count
            or len(powers) != count or len(kinds) != count):
        raise ValueError("native ZAFF coordinate dimensions are inconsistent")
    values, first, second = np.empty((3, count))
    for i, kind in enumerate(kinds):
        if kind == "inverse_distance":
            if r[i] <= 0 or r[i] + x[i] <= 0:
                raise ValueError("native inverse distance must remain positive")
            values[i] = 1 / (r[i] + x[i]) - 1 / r[i]
            first[i] = -1 / (r[i] + x[i])**2
            second[i] = 2 / (r[i] + x[i])**3
        elif kind == "cosine":
            values[i] = np.cos(r[i] + x[i]) - np.cos(r[i])
            first[i] = -np.sin(r[i] + x[i])
            second[i] = -np.cos(r[i] + x[i])
        elif kind == "sine":
            values[i] = np.sin(x[i])
            first[i] = np.cos(x[i])
            second[i] = -np.sin(x[i])
        else:
            raise ValueError(f"unsupported native ZAFF coordinate kind: {kind}")
    hessian = np.zeros((count, count))
    for i, pi in enumerate(powers):
        if not pi:
            continue
        outside = np.prod([values[k]**powers[k] for k in range(count) if k != i])
        diagonal = pi * values[i]**(pi - 1) * second[i]
        if pi > 1:
            diagonal += pi * (pi - 1) * values[i]**(pi - 2) * first[i]**2
        hessian[i, i] = outside * diagonal
        for j in range(i + 1, count):
            pj = powers[j]
            if pj:
                outside = np.prod([values[k]**powers[k] for k in range(count)
                                   if k not in (i, j)])
                value = (outside * pi * pj * values[i]**(pi - 1)
                         * values[j]**(pj - 1) * first[i] * first[j])
                hessian[i, j] = hessian[j, i] = value
    return transform.T @ hessian @ transform


@dataclass(frozen=True)
class NativeSparseFit:
    """An ordered OMP path; coefficients are in the caller's native units."""

    support_order: tuple[int, ...]
    coefficients_by_size: dict[int, np.ndarray]
    residual_norm_by_size: dict[int, float]
    condition_by_size: dict[int, float]

    def coefficients(self, size: int, column_count: int) -> np.ndarray:
        result = np.zeros(column_count)
        result[list(self.support_order[:size])] = self.coefficients_by_size[size]
        return result


def fit_native_sparse_refinement(
    design: np.ndarray,
    target: np.ndarray,
    *,
    max_terms: int,
    condition_limit: float = 1.0e10,
    stop_when: Callable[[np.ndarray], bool] | None = None,
) -> NativeSparseFit:
    """Build a rank-certified orthogonal matching pursuit path.

    Rows can mix energies and gradients after the caller applies its explicit
    physical weighting.  Degenerate or unobserved columns are rejected rather
    than silently given arbitrary force constants.
    """

    a = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if a.ndim != 2 or y.shape != (a.shape[0],):
        raise ValueError("native fit design/target shapes are inconsistent")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(y)):
        raise ValueError("native fit data contain non-finite values")
    if not 1 <= max_terms <= min(a.shape):
        raise ValueError("invalid maximum native support size")
    norms = np.linalg.norm(a, axis=0)
    if np.any(norms <= 1.0e-14):
        raise ValueError("native ZAFF candidate is unobserved")
    normalized = a / norms
    support: list[int] = []
    residual = y.copy()
    coefficients: dict[int, np.ndarray] = {}
    residuals: dict[int, float] = {}
    conditions: dict[int, float] = {}
    rejected: set[int] = set()
    q = np.empty((a.shape[0], max_terms))
    r = np.zeros((max_terms, max_terms))
    projected_target = np.zeros(max_terms)
    for _ in range(max_terms):
        correlations = np.abs(normalized.T @ residual)
        correlations[support + sorted(rejected)] = -1.0
        proposed = None
        for index in np.argsort(-correlations, kind="stable"):
            if correlations[index] < 0:
                break
            trial = support + [int(index)]
            count = len(support)
            column = normalized[:, index].copy()
            projection = q[:, :count].T @ column
            column -= q[:, :count] @ projection
            # Reorthogonalization prevents redundant primitives from losing
            # rank numerically as the native basis grows.
            correction = q[:, :count].T @ column
            projection += correction
            column -= q[:, :count] @ correction
            diagonal = float(np.linalg.norm(column))
            r[:count, count] = projection
            r[count, count] = diagonal
            condition = float(np.linalg.cond(r[:count+1, :count+1]))
            if np.isfinite(condition) and condition <= condition_limit:
                proposed = trial
                break
            # An ill-conditioned addition must not prevent an independent
            # lower-scoring term from entering the model. Adding more columns
            # cannot improve this subset's extreme singular-value ratio.
            rejected.add(int(index))
        if proposed is None:
            break
        q[:, count] = column / diagonal
        projected_target[count] = q[:, count] @ y
        support = proposed
        count = len(support)
        scaled = solve_triangular(r[:count, :count], projected_target[:count])
        residual = y - q[:, :count] @ projected_target[:count]
        coefficients[count] = scaled / norms[support]
        residuals[count] = float(np.linalg.norm(residual))
        conditions[count] = condition
        if stop_when is not None:
            current = np.zeros(a.shape[1], dtype=float)
            current[support] = coefficients[count]
            if stop_when(current):
                break
    if not support:
        raise ValueError("no rank-certified native ZAFF refinement term")
    return NativeSparseFit(tuple(support), coefficients, residuals, conditions)


def select_native_support(
    path: NativeSparseFit,
    validation_errors: dict[int, float],
    *,
    margin: float,
    maximum_size: int,
    admissible: dict[int, bool] | None = None,
) -> int:
    """Choose the smallest support within a held-out error margin."""

    if margin < 0 or maximum_size < 1:
        raise ValueError("invalid native support selection policy")
    allowed = [size for size in validation_errors
               if size in path.coefficients_by_size and size <= maximum_size
               and np.isfinite(validation_errors[size])
               and (admissible is None or admissible.get(size, False))]
    if not allowed:
        raise ValueError("no native support passes validation gates")
    best = min(validation_errors[size] for size in allowed)
    return min(size for size in allowed if validation_errors[size] <= best + margin)
