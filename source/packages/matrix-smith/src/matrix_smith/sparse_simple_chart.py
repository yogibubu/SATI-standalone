"""Operator-only redundant primitive chart with bounded iterative solves.

No dense B, Gram matrix, pseudoinverse or non-rigid Cartesian basis is built.
Successful solves are residual-qualified; budget exhaustion is an error, not
a claim that the missing directions have disappeared. Scaling additionally
requires bounded local density, primitive degree and symmetry group order.
"""
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsmr
from .models import GICPrimitive


class SparseSimpleChart:
    def __init__(self, path, projector, *, pair_error_tolerance, solve_iterations=128):
        from matrix_oracle import build_initial_hessian
        if pair_error_tolerance is None:
            raise ValueError('sparse simple chart requires an explicit pair error tolerance')
        self.oracle_hessian = build_initial_hessian(path, pair_error_tolerance=pair_error_tolerance)
        self.reference = self.oracle_hessian.coordinates_angstrom.copy()
        self.projector = projector
        self.solve_iterations = int(solve_iterations)
        self.local_primitives = self.oracle_hessian.simple_primitives()
        from matrix_chem.primitive_coordinates import compile_local_primitive_plan
        self._compiled_local_primitive_plan = compile_local_primitive_plan(
            self.local_primitives
        )
        if self._compiled_local_primitive_plan is None:
            raise RuntimeError('compiled primitive CSR plan unavailable')
        self.primitives = tuple(GICPrimitive(f'XTB{i+1:06d}', p.label, p.kind, p.function,
            tuple(a+1 for a in p.atoms), mode=p.mode)
            for i,p in enumerate(self.local_primitives))
        self.periodic = np.array([p.function == 'D' for p in self.primitives])
        self._cache = OrderedDict()
        self._inverse_cache = OrderedDict()
        self.last_inverse_audit = None
        self.reference_values = self.evaluate(self.reference)[0].copy()
        self.audit = dict(self.oracle_hessian.audit, coordinate_runtime='SPARSE_SIMPLE_REDUNDANT',
                          solve_iteration_budget=self.solve_iterations,
                          dense_coordinate_matrices=False, active_rank=projector.active_rank,
                          rank_certificate='PER_APPLIED_COVECTOR_RESIDUAL_NOT_GLOBAL_SVD')

    def evaluate(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != self.reference.shape or not np.isfinite(xyz).all():
            raise ValueError('invalid sparse simple chart geometry')
        key = xyz.tobytes()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        # This chart contains exactly the bounded-support xTB primitive atlas.
        # Use its specialized C++ batch instead of the general SMITH plan
        # evaluator; independent rows are distributed over coordinate workers.
        from matrix_chem.primitive_coordinates import evaluate_compiled_local_primitive_plan
        result = evaluate_compiled_local_primitive_plan(
            self._compiled_local_primitive_plan, xyz
        )
        values, data, indices, pointers = result
        b = csr_matrix((data, indices, pointers), shape=(len(values), xyz.size))
        if not np.isfinite(b.data).all():
            raise FloatingPointError('nonfinite primitive Jacobian')
        self._cache[key] = (np.asarray(values), b)
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return self._cache[key]

    def difference(self, a, b):
        d = np.asarray(a)-np.asarray(b)
        d[self.periodic] = np.arctan2(np.sin(d[self.periodic]), np.cos(d[self.periodic]))
        return d

    def actual_q(self, xyz):
        return self.difference(self.evaluate(xyz)[0], self.reference_values)

    def operator(self, xyz):
        b = self.evaluate(xyz)[1]
        project = self.projector.project
        from matrix_chem import compiled_csr_matvec
        return LinearOperator(b.shape,
            matvec=lambda v: compiled_csr_matvec(
                b.data,b.indices,b.indptr,project(np.asarray(v).ravel()),
                column_count=b.shape[1]),
            rmatvec=lambda v: project(compiled_csr_matvec(
                b.data,b.indices,b.indptr,np.asarray(v).ravel(),
                column_count=b.shape[1],transpose=True)),dtype=float)

    def _solve(self, operator, rhs, *, consistent=False):
        rhs = np.asarray(rhs, dtype=float).ravel()
        result = lsmr(operator, rhs, atol=1.e-11, btol=1.e-11, maxiter=self.solve_iterations)
        residual = operator @ result[0]-rhs
        normal = operator.rmatvec(residual)
        scale = max(float(np.linalg.norm(operator.rmatvec(rhs))), 1.e-14)
        if result[1] not in (0,1,2,4,5) or np.linalg.norm(normal) > 1.e-7*scale:
            raise RuntimeError('sparse primitive inverse exhausted its residual/budget contract')
        if consistent and np.linalg.norm(residual) > 1.e-7*max(np.linalg.norm(rhs),1.e-14):
            raise RuntimeError('primitive pool cannot represent the requested Cartesian tangent')
        return np.asarray(result[0])

    def forward(self, q, xyz):
        return self._refined_rectangular_solve(q, xyz, adjoint=False)

    def adjoint(self, gx, xyz):
        return self._refined_rectangular_solve(gx, xyz, adjoint=True)

    def _refined_rectangular_solve(
        self, rhs, xyz, *, adjoint, relative_tolerance=None
    ):
        from matrix_chem.sparse_krylov import SparseKrylovPreconditioner
        key = np.asarray(xyz,dtype=float).tobytes()
        if key not in self._inverse_cache:
            b = self.evaluate(xyz)[1]
            # Sparse local Cartesian Gram matrix, NOT the dense redundant
            # metric. Its support is bounded by the primitive neighborhoods.
            gram = (b.T@b).tocsr()
            preconditioner = SparseKrylovPreconditioner(gram, diagonal_shift=1.e-10)
            self._inverse_cache[key] = (gram,preconditioner)
            while len(self._inverse_cache)>2:
                self._inverse_cache.popitem(last=False)
        self._inverse_cache.move_to_end(key)
        gram, preconditioner = self._inverse_cache[key]
        p = self.projector.project
        original = self.operator(xyz)
        rectangular = LinearOperator(original.shape,
            matvec=lambda v:original@preconditioner.right(v),
            rmatvec=lambda v:preconditioner.right_transpose(original.rmatvec(v)),dtype=float)
        rhs = p(np.asarray(rhs).ravel()) if adjoint else np.asarray(rhs).ravel()
        solution = np.zeros(original.shape[0] if adjoint else original.shape[1])
        residual = rhs.copy()
        scale = max(np.linalg.norm(rhs if adjoint else original.rmatvec(rhs)),1.e-14)
        tolerance = (
            (1.e-8 if adjoint else 1.e-7)
            if relative_tolerance is None
            else float(relative_tolerance)
        )
        if not 0.0 < tolerance <= 1.e-2:
            raise ValueError('invalid sparse primitive solve tolerance')
        used = 0
        for refinement in range(4):
            result = lsmr(rectangular.T if adjoint else rectangular,
                preconditioner.right_transpose(residual) if adjoint else residual,
                atol=1.e-12,btol=1.e-12,maxiter=self.solve_iterations-used)
            used += result[2]
            solution += result[0] if adjoint else p(preconditioner.right(result[0]))
            residual = rhs-(original.rmatvec(solution) if adjoint else original@solution)
            relative = float(np.linalg.norm(residual if adjoint else original.rmatvec(residual))/scale)
            self.last_inverse_audit = dict(iterations=used,refinements=refinement,
                relative_residual=relative,gram_nnz=gram.nnz,factor_nnz=preconditioner.nnz,
                method='RECTANGULAR_PRECONDITIONED_LSMR_WITH_ORIGINAL_RESIDUAL_REFINEMENT',
                regularization='PRECONDITIONER_ONLY_ORIGINAL_EQUATION_RESIDUAL')
            # Qualify the original equation, not the scaled LSMR stopping
            # estimate. An inner budget stop is harmless only if this passes.
            if np.isfinite(relative) and relative <= tolerance:
                return solution
            if used >= self.solve_iterations:
                break
        raise RuntimeError(f'primitive inverse original residual failed: iterations={used}, residual={relative:.3e}')

    def differential(self, dx, xyz):
        return self.operator(xyz) @ np.asarray(dx).ravel()

    def pullback(self, gq, xyz):
        return self.operator(xyz).rmatvec(np.asarray(gq).ravel())

    def realize(self, q, start=None):
        target = self.reference_values+np.asarray(q)
        start = self.reference if start is None else np.asarray(start)
        p = self.projector.project
        displacement = p((start-self.reference).ravel())
        xyz = self.reference+p(displacement).reshape(self.reference.shape)
        evaluations = 0
        linear_solves = 0
        rejected_trials = 0
        previous_objective = np.inf
        for iteration in range(24):
            values, _ = self.evaluate(xyz)
            evaluations += 1
            residual = self.difference(values, target)
            operator = self.operator(xyz)
            normal = float(np.linalg.norm(operator.rmatvec(residual), ord=np.inf))
            objective = 0.5*float(np.dot(residual, residual))
            if normal <= 1.e-7:
                return xyz, {
                    'method': 'SPARSE_REDUNDANT_PRECONDITIONED_GAUSS_NEWTON',
                    'iterations': iteration,
                    'evaluations': evaluations,
                    'linear_solves': linear_solves,
                    'rejected_trials': rejected_trials,
                    'normal_residual': normal,
                    'residual_norm': float(np.sqrt(2.0*objective)),
                    'converged': True,
                }
            if not np.isfinite(objective) or objective >= previous_objective*(1.0+1.e-10):
                raise RuntimeError('sparse redundant back-transformation failed to descend')
            previous_objective = objective
            # A nonlinear Gauss--Newton microiteration needs a descent
            # direction, not a fully converged inverse of the current
            # linearization.  Requiring the production forward-map tolerance
            # here used to reject useful inexact Newton steps and forced the
            # outer Cartesian-trust solver into repeated halvings.  The final
            # nonlinear residual below remains qualified at the strict
            # 1e-7 level.
            step = self._refined_rectangular_solve(
                -residual,
                xyz,
                adjoint=False,
                relative_tolerance=1.e-2,
            ).reshape(-1)
            linear_solves += 1
            step = p(step)
            directional = float(np.dot(operator @ step, residual))
            if not np.isfinite(directional) or directional >= 0.0:
                raise RuntimeError('sparse redundant back-transformation produced a non-descent step')
            accepted = False
            scale = 1.0
            for _line_search in range(9):
                candidate_displacement = p(displacement+scale*step)
                candidate = self.reference+candidate_displacement.reshape(self.reference.shape)
                try:
                    candidate_residual = self.difference(self.evaluate(candidate)[0], target)
                    evaluations += 1
                    candidate_objective = 0.5*float(
                        np.dot(candidate_residual, candidate_residual)
                    )
                except (FloatingPointError, RuntimeError, ValueError):
                    candidate_objective = np.inf
                if (
                    np.isfinite(candidate_objective)
                    and candidate_objective
                    <= objective+1.e-4*scale*directional
                ):
                    displacement = candidate_displacement
                    xyz = candidate
                    accepted = True
                    break
                rejected_trials += 1
                scale *= 0.5
            if not accepted:
                raise RuntimeError('sparse redundant back-transformation line search failed')
        residual = self.difference(self.evaluate(xyz)[0], target)
        normal = float(np.linalg.norm(self.operator(xyz).rmatvec(residual), ord=np.inf))
        raise RuntimeError(
            'sparse redundant nonlinear back-transformation failed: '
            f'iterations=24 normal_residual={normal:.3e}'
        )
