"""Bounded sparse ILU acceleration; residuals refer to the original operator."""
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu


class SparseKrylovPreconditioner:
    def __init__(self, matrix, *, diagonal_shift=0., fill_factor=4.):
        matrix = csc_matrix(matrix)
        if matrix.shape[0] != matrix.shape[1] or not matrix.shape[0] or not np.isfinite(matrix.data).all():
            raise ValueError('sparse preconditioner requires a finite nonempty square matrix')
        if not np.isfinite(diagonal_shift) or not np.isfinite(fill_factor) or fill_factor < 1:
            raise ValueError('invalid sparse preconditioner controls')
        diagonal = np.asarray(matrix.diagonal())+diagonal_shift
        scale = 1./np.sqrt(np.maximum(np.abs(diagonal), 1.e-12))
        scaled = diags(scale)@(matrix+diags(np.full(matrix.shape[0],diagonal_shift)))@diags(scale)
        self.factor = spilu(csc_matrix(scaled), drop_tol=1.e-5,
                            fill_factor=fill_factor, permc_spec='NATURAL', diag_pivot_thresh=0.)
        self.scale = scale
        self._triangular = None
        self.nnz = self.factor.L.nnz+self.factor.U.nnz
        self.storage_limit = 2*fill_factor*max(matrix.nnz,matrix.shape[0])
        if self.nnz > self.storage_limit:
            raise RuntimeError('sparse preconditioner exceeded its storage budget')

    def apply(self, vector):
        return self.scale*self.factor.solve(self.scale*np.asarray(vector).ravel())

    def _half_factor(self):
        if self._triangular is None:
            upper = self.factor.U
            self._root_diagonal = np.sqrt(np.abs(upper.diagonal()))
            self._triangular = splu(csc_matrix(upper),permc_spec='NATURAL',diag_pivot_thresh=0.)
            self.nnz += self._triangular.L.nnz+self._triangular.U.nnz
            if self.nnz > self.storage_limit:
                raise RuntimeError('rectangular preconditioner exceeded its storage budget')
        return self._triangular

    def right(self, vector):
        factor = self._half_factor()
        return self.scale*factor.solve(self._root_diagonal*np.asarray(vector).ravel())

    def right_transpose(self, vector):
        factor = self._half_factor()
        return self._root_diagonal*factor.solve(self.scale*np.asarray(vector).ravel(),trans='T')


def checked_gmres(action, rhs, precondition, *, maximum_iterations=128, tolerance=1.e-9):
    rhs = np.asarray(rhs,dtype=float).ravel()
    n = len(rhs)
    if not n or not np.isfinite(rhs).all() or maximum_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid bounded sparse solve')
    operator = LinearOperator((n,n), matvec=action, dtype=float)
    inverse = LinearOperator((n,n), matvec=precondition, dtype=float)
    iterations = 0
    def count(_):
        nonlocal iterations
        iterations += 1
    # legacy callback makes maxiter count inner iterations, not restarts.
    solution, info = gmres(operator, rhs, M=inverse, restart=min(32,n),
        maxiter=maximum_iterations, rtol=tolerance, atol=0., callback=count, callback_type='legacy')
    residual = float(np.linalg.norm(action(solution)-rhs))
    relative = residual/max(float(np.linalg.norm(rhs)),1.e-30)
    if not np.isfinite(relative) or info != 0 or relative > tolerance*2:
        raise RuntimeError(f'preconditioned sparse solve failed: iterations={iterations}, relative residual={relative:.3e}')
    return solution, {'iterations':iterations,'relative_residual':relative}
