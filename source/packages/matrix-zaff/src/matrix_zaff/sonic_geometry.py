"""Local inverse of a natural SONIC chart, with sparse implicit derivatives.

Solve S u(x) = S u(reference) + active(z), not u(x) = u0 + L z:
the latter is generally incompatible with redundant primitive geometry.
Cartesian coordinates are in bohr; SMITH primitives use Angstrom/radian.
The constant reference Eckart gauge makes the inverse differentiable without
omitting derivatives of a moving rigid-motion projector. No global dense
Jacobian, pseudoinverse, Hessian or sparse direct factorization is built.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsmr, svds

from matrix_smith.evaluation import (
    build_sparse_primitive_b_matrix, evaluate_primitive_values,
)

from .nonbonded import BOHR_TO_ANGSTROM as B
from .primitive_second_derivative import analytic_primitive_second_derivative


class SonicGeometryError(RuntimeError):
    """The current local chart cannot safely realize the requested point."""


def _vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid inverse SONIC vector")
    return result


class SonicGeometryMap:
    """A full independent chart with optionally frozen SONIC rows.

All declared rows must span 3N minus the reference rigid motions, even when
only totally symmetric rows are active. Undercomplete charts, disconnected
free fragments without relative coordinates, and ill-conditioned charts are
rejected. Atom masks and periodic cells require a separate gauge contract.
The inverse is local: a branch change requires a new map/optimizer epoch.
"""

    def __init__(self, definition, variables, embedding, *, tolerance=2e-11,
                 linear_tolerance=1e-12, maximum_condition=1e7,
                 maximum_iterations=40, maximum_linear_iterations=2000,
                 maximum_step_bohr=0.5, maximum_local_atoms=16):
        if definition.backend == "matrix-smith-gicforge.v1":
            raise ValueError("legacy SMITH backend uses a dense B; a sparse native chart is required")
        if (not all(np.isfinite(v) and v > 0 for v in
                    (tolerance, linear_tolerance, maximum_step_bohr))
                or not np.isfinite(maximum_condition) or maximum_condition <= 1
                or any(int(v) != v or v < 1 for v in
                       (maximum_iterations, maximum_linear_iterations, maximum_local_atoms))):
            raise ValueError("invalid inverse SONIC solver policy")
        self.definition, self.variables, self.embedding = definition, variables, embedding
        self.tolerance, self.linear_tolerance = tolerance, linear_tolerance
        self.maximum_condition = maximum_condition
        self.maximum_iterations = int(maximum_iterations)
        self.maximum_linear_iterations = int(maximum_linear_iterations)
        self.maximum_step_bohr = maximum_step_bohr
        self.maximum_local_atoms = int(maximum_local_atoms)
        reference = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if (reference.ndim != 2 or reference.shape[1] != 3 or not reference.size
                or not np.all(np.isfinite(reference))):
            raise ValueError("inverse SONIC map requires finite reference XYZ")
        self.reference = reference.copy() / B
        self.reference.setflags(write=False)
        self.size = reference.size
        by_id = {p.identifier: (i, p) for i, p in enumerate(definition.primitives)}
        if (len(variables.variables) != len(embedding.local_labels)
                or any(label not in by_id for label in embedding.local_labels)):
            raise ValueError("inverse SONIC local labels/variables do not match SMITH")
        self.indices = tuple(by_id[label][0] for label in embedding.local_labels)
        self.primitives = tuple(by_id[label][1] for label in embedding.local_labels)
        self.active = np.array([embedding.labels.index(label)
                               for label in embedding.active_labels], dtype=int)
        self.gauge = self._reference_gauge()
        self.row_count = embedding.analysis.shape[0]
        if self.row_count + self.gauge.shape[0] != self.size:
            raise ValueError("inverse SONIC requires a complete independent chart (3N minus rigid modes)")
        natural = self._values(self.reference)
        if not np.allclose(natural, embedding.reference, atol=tolerance, rtol=0):
            raise ValueError("SONIC embedding reference does not match the reference geometry")
        self.target_reference = np.asarray(embedding.analysis @ natural)
        # Only O(N * 6) rigid vectors and a 6 by 6 eigensystem are dense.
        state = self._state(self.reference, 0, 0.)
        self.reference_condition = self._condition(state.operator)

    def _reference_gauge(self):
        centered = self.reference - self.reference.mean(axis=0)
        modes = np.column_stack(
            [np.tile(axis, (len(centered), 1)).ravel() for axis in np.eye(3)]
            + [np.cross(axis, centered).ravel() for axis in np.eye(3)])
        # Normalize first so translation/rotation units do not affect rank.
        norms = np.linalg.norm(modes, axis=0)
        modes /= np.where(norms > 0, norms, 1.)
        values, vectors = np.linalg.eigh(modes.T @ modes)
        keep = values > 1e-12 * max(values[-1], 1.)
        return ((modes @ vectors[:, keep]) / np.sqrt(values[keep])).T

    def _values(self, xyz):
        values = evaluate_primitive_values(
            self.definition, primitive_ids=self.embedding.local_labels,
            coordinates_angstrom=xyz * B)
        return self.variables.forward([values[label] for label in self.embedding.local_labels])

    def _state(self, xyz, iterations, residual):
        natural = self._values(xyz)
        _, inverse_first, inverse_second = self.variables.inverse(natural)
        first = 1. / inverse_first
        second = -inverse_second / inverse_first**3
        raw = build_sparse_primitive_b_matrix(
            self.definition, coordinates_angstrom=xyz * B, primitive_indices=self.indices)
        columns, data, indptr = [], [], [0]
        for row in raw.rows:
            for column, value in row.entries:
                columns.append(column)
                data.append(value * B)
            indptr.append(len(data))
        physical_b = sparse.csr_matrix((data, columns, indptr),
                                      shape=(len(self.primitives), self.size))
        natural_b = physical_b.multiply(first[:, None]).tocsr()
        chart_b = (self.embedding.analysis @ natural_b).tocsr()
        gauge = self.gauge

        def push(v):
            v = np.asarray(v).reshape(-1)
            return np.concatenate((chart_b @ v, gauge @ v))

        def pull(v):
            v = np.asarray(v).reshape(-1)
            return chart_b.T @ v[:self.row_count] + gauge.T @ v[self.row_count:]

        operator = LinearOperator((self.size, self.size), matvec=push, rmatvec=pull, dtype=float)
        return SonicGeometryState(self, xyz.copy(), natural, physical_b, natural_b,
                                  first, second, operator, iterations, residual)

    def _condition(self, operator):
        try:
            smallest = float(svds(operator, k=1, which="SM", return_singular_vectors=False,
                                  random_state=0, tol=1e-9,
                                  maxiter=self.maximum_linear_iterations)[0])
            largest = float(svds(operator, k=1, which="LM", return_singular_vectors=False,
                                 random_state=0, tol=1e-9,
                                 maxiter=self.maximum_linear_iterations)[0])
        except Exception as exc:
            raise SonicGeometryError("inverse SONIC condition estimate did not converge") from exc
        if (not np.isfinite(smallest + largest) or smallest <= 0
                or largest / smallest > self.maximum_condition):
            raise SonicGeometryError("singular or ill-conditioned inverse SONIC chart")
        return largest / smallest

    def _solve(self, operator, rhs):
        rhs = _vector(rhs, self.size)
        if not np.any(rhs):
            return np.zeros(self.size)
        result = lsmr(operator, rhs, atol=self.linear_tolerance, btol=self.linear_tolerance,
                      conlim=self.maximum_condition, maxiter=self.maximum_linear_iterations)
        residual = np.linalg.norm(operator @ result[0] - rhs)
        if (result[1] not in (1, 2, 4, 5) or result[6] > self.maximum_condition
                or not np.all(np.isfinite(result[0]))
                or residual > 20 * self.linear_tolerance * np.linalg.norm(rhs)):
            raise SonicGeometryError("inverse SONIC linear solve failed its residual/condition check")
        return result[0]

    def realize(self, coordinates, *, start_coordinates_bohr=None):
        """Reconstruct a compatible geometry; never return an unconverged point."""
        z = _vector(coordinates, len(self.active))
        target = self.target_reference.copy()
        target[self.active] += z
        xyz = self.reference.copy()
        if start_coordinates_bohr is not None:
            start = np.asarray(start_coordinates_bohr, dtype=float)
            if start.shape != xyz.shape or not np.all(np.isfinite(start)):
                raise ValueError("invalid inverse SONIC starting geometry")
            xyz[:] = start

        def residual_at(point):
            return np.concatenate((self.embedding.analysis @ self._values(point) - target,
                                   self.gauge @ (point - self.reference).ravel()))

        for iteration in range(self.maximum_iterations + 1):
            residual = residual_at(xyz)
            error = float(np.max(np.abs(residual)))
            if error <= self.tolerance:
                state = self._state(xyz, iteration, error)
                # A small nonlinear residual alone cannot certify an inverse.
                state.condition_estimate = self._condition(state.operator)
                return state
            if iteration == self.maximum_iterations:
                break
            state = self._state(xyz, iteration, error)
            step = self._solve(state.operator, -residual).reshape(xyz.shape)
            length = np.max(np.linalg.norm(step, axis=1))
            alpha = min(1., self.maximum_step_bohr / max(length, 1e-30))
            for _ in range(25):
                trial = xyz + alpha * step
                try:
                    trial_error = np.linalg.norm(residual_at(trial))
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    trial_error = np.inf
                if trial_error < (1. - 1e-4 * alpha) * np.linalg.norm(residual):
                    xyz = trial
                    break
                alpha *= .5
            else:
                raise SonicGeometryError("inverse SONIC line search failed; rebuild the local chart")
        raise SonicGeometryError("inverse SONIC geometry did not converge")


class SonicGeometryState:
    """Realized chart and Jacobian/adjoint actions, never an explicit inverse."""

    def __init__(self, owner, xyz, natural, physical_b, natural_b, first, second,
                 operator, iterations, residual):
        self.owner, self.coordinates_bohr, self.natural_values = owner, xyz, natural
        self.physical_b, self.natural_b = physical_b, natural_b
        self.first, self.second, self.operator = first, second, operator
        self.iterations, self.residual = iterations, residual
        self.condition_estimate = None
        self._primitive_hessians = {}

    def push(self, direction):
        rhs = np.zeros(self.owner.size)
        rhs[self.owner.active] = _vector(direction, len(self.owner.active))
        return self.owner._solve(self.operator, rhs)

    def multipliers(self, cartesian_gradient):
        return self.owner._solve(self.operator.T, cartesian_gradient)

    def pull(self, cartesian_gradient):
        return self.multipliers(cartesian_gradient)[self.owner.active]

    def _primitive_hessian(self, index):
        if index not in self._primitive_hessians:
            primitive = self.owner.primitives[index]
            # Bound local AD storage before allocating any primitive Hessian.
            atoms = set(primitive.atoms) | set(primitive.ref_atoms)
            atoms.update(primitive.frame_atoms)
            atoms.update(primitive.ref_frame_atoms)
            if primitive.function.upper() in {"RPCB", "RPCK"}:
                from matrix_smith.numerics import (
                    _angle_component_terms_from_refs, _ring_pucker_terms_from_refs,
                )
                parse = (_angle_component_terms_from_refs if primitive.function.upper() == "RPCB"
                         else _ring_pucker_terms_from_refs)
                for _, term_atoms in parse(primitive):
                    atoms.update(term_atoms)
            if len(atoms) > self.owner.maximum_local_atoms:
                raise SonicGeometryError("primitive support exceeds bounded analytic curvature policy")
            try:
                value = analytic_primitive_second_derivative(
                    primitive, self.coordinates_bohr * B,
                    reference_coordinates=self.owner.reference * B, zero_tolerance=0.)
            except (NotImplementedError, FloatingPointError, ZeroDivisionError) as exc:
                raise SonicGeometryError("analytic primitive curvature unavailable") from exc
            self._primitive_hessians[index] = value.entries
        return self._primitive_hessians[index]

    def weighted_curvature(self, weights, cartesian_direction):
        """Apply sum_i weights_i Hessian_x(u_i), including scalar chart curvature."""
        weights = _vector(weights, len(self.natural_values))
        direction = _vector(cartesian_direction, self.owner.size)
        result = np.asarray(self.physical_b.T @ (
            weights * self.second * (self.physical_b @ direction)))
        for index in np.flatnonzero(weights):
            factor = weights[index] * self.first[index] * B**2
            for row, column, value in self._primitive_hessian(index):
                result[row] += factor * value * direction[column]
        return result

    def pull_hessian_vector(self, cartesian_gradient, cartesian_product, direction):
        """J.T Hx Jv plus the nonstationary inverse-map curvature term."""
        tangent = self.push(direction)
        multipliers = self.multipliers(cartesian_gradient)
        weights = self.owner.embedding.analysis.T @ multipliers[:self.owner.row_count]
        corrected = _vector(cartesian_product, self.owner.size) - self.weighted_curvature(weights, tangent)
        return self.pull(corrected)
