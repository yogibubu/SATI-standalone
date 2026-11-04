"""Recycled block inverse iteration with one reusable finite shifted factor."""
import time
import warnings
import numpy as np
from scipy.linalg import cholesky, eigh, lu_factor, lu_solve, LinAlgWarning, qr, solve_triangular
from scipy.sparse.linalg import aslinearoperator
from threadpoolctl import threadpool_limits
from .vibronic_energy_windows import QualifiedEigenwindow, qualify_operator_eigenpairs


def _orthonormalize_inverse_block(values):
    """CholeskyQR2 with a full Gram check and Householder fallback."""
    original=values
    try:
        for _ in range(2):
            gram=values.conj().T@values
            r=cholesky(.5*(gram+gram.conj().T),lower=False,check_finite=False)
            values=solve_triangular(r.T,values.T,lower=True,check_finite=False).T
        error=float(abs(values.conj().T@values-np.eye(values.shape[1])).max())
        if not np.isfinite(error) or error>5e-13:
            raise np.linalg.LinAlgError('CholeskyQR2 Gram qualification failed')
        return values,'checked CholeskyQR2'
    except np.linalg.LinAlgError:
        q,_=qr(original,mode='economic',check_finite=False)
        return q,'Householder fallback'


class DenseShiftedInverse:
    """Float64/complex128 LU factor, reused for every block inverse step.

    LU supplies no independent inertia count. Obtain the window count from a
    checked inertia backend or an already qualified complete target spectrum.
    A PARDISO factor's solve method can also be supplied to the iteration.
    """
    def __init__(self, matrix, shift_cm, *, hamiltonian_sha256, workers=1,
            maximum_workspace_bytes=128*1024**3):
        matrix=np.asarray(matrix);n=matrix.shape[0]
        if (matrix.ndim!=2 or matrix.shape!=(n,n) or not n or not hamiltonian_sha256
                or not np.isfinite(shift_cm) or workers<1 or not np.all(np.isfinite(matrix))):
            raise ValueError('identified finite square matrix, shift and workers required')
        dtype=np.result_type(matrix.dtype,np.float64)
        if 6*n*n*np.dtype(dtype).itemsize>maximum_workspace_bytes:
            raise MemoryError('shifted dense factor exceeds workspace budget')
        self.shift_cm=float(shift_cm);self.hamiltonian_sha256=hamiltonian_sha256
        started=time.monotonic();a=np.array(matrix,dtype=dtype,order='F',copy=True)
        a[np.diag_indices(n)]-=self.shift_cm
        with threadpool_limits(limits=workers,user_api='blas'), warnings.catch_warnings():
            warnings.simplefilter('error',LinAlgWarning)
            try:self.factor,self.pivots=lu_factor(a,overwrite_a=True,check_finite=False)
            except LinAlgWarning as error:raise ArithmeticError('singular shifted factor') from error
        self.diagnostics=dict(factor_seconds=time.monotonic()-started,shift_cm=self.shift_cm,
            workers=workers,arithmetic=str(np.dtype(dtype)),independent_count_supplied=False)
        self.shape=(n,n)

    def solve(self, rhs):
        rhs=np.asarray(rhs)
        if rhs.shape[0]!=self.shape[0] or not np.all(np.isfinite(rhs)):
            raise ValueError('finite compatible inverse right-hand side required')
        if np.iscomplexobj(rhs) and not np.iscomplexobj(self.factor):
            return self.solve(rhs.real)+1j*self.solve(rhs.imag)
        return lu_solve((self.factor,self.pivots),rhs,check_finite=False)


def refine_inverse_window(operator, starting_vectors, *, count, shifted_inverse,
        hamiltonian_sha256, qualification_operator=None, workers=1,
        residual_tolerance_cm=1e-7, maximum_iterations=40, maximum_seconds=None,
        maximum_workspace_bytes=128*1024**3, progress=None):
    """Refine every counted interior eigenpair from a recycled block of guesses.

    The shift must be the window midpoint. Guard vectors in the initial block
    accelerate separation of wanted and unwanted levels. No missing level or
    unconverged residual is accepted when the declared budgets expire.
    """
    op=aslinearoperator(operator);initial=np.asarray(starting_vectors)
    n=op.shape[0];k=count.count;center=.5*(count.lower_cm+count.upper_cm)
    if (op.shape!=(n,n) or initial.ndim!=2 or len(initial)!=n
            or not k or not k<=initial.shape[1]<=n or not np.all(np.isfinite(initial))
            or count.hamiltonian_sha256!=hamiltonian_sha256
            or shifted_inverse.hamiltonian_sha256!=hamiltonian_sha256
            or shifted_inverse.shift_cm!=center or workers<1 or maximum_iterations<1
            or not 0<residual_tolerance_cm<=1e-6
            or (maximum_seconds is not None and (not np.isfinite(maximum_seconds) or maximum_seconds<=0))):
        raise ValueError('compatible counted window, midpoint factor and recycled block required')
    if getattr(operator,'approximate_products',False) and qualification_operator is None:
        raise ValueError('approximate products require an independent CPU qualification operator')
    dtype=np.result_type(op.dtype,initial.dtype,np.float64);columns=initial.shape[1]
    if (8*n*columns+6*columns**2)*np.dtype(dtype).itemsize>maximum_workspace_bytes:
        raise MemoryError('inverse block iteration exceeds workspace budget')
    started=time.monotonic();orthogonalization={}
    with threadpool_limits(limits=workers,user_api='blas'):
        q,r=qr(np.asarray(initial,dtype=dtype),mode='economic',check_finite=False)
        if np.min(abs(np.diag(r)))<1e-12:
            raise ValueError('linearly independent inverse-iteration starting vectors required')
        for iteration in range(1,maximum_iterations+1):
            if maximum_seconds is not None and time.monotonic()-started>maximum_seconds:
                raise TimeoutError('inverse iteration exceeded its elapsed-time budget')
            z=shifted_inverse.solve(q)
            # Wanted interior levels are extremal in the inverse operator.
            # Selecting ordinary H-Ritz values here admits spectral pollution
            # from still-unconverged guard vectors near the target midpoint.
            inverse_reduced=q.conj().T@z
            inverse_values,c=eigh(.5*(inverse_reduced+inverse_reduced.conj().T),check_finite=False)
            take=np.argsort(abs(inverse_values))[-k:]
            candidates=q@c[:,take]
            applied=op.matmat(candidates);reduced=candidates.conj().T@applied
            energies,rotation=eigh(.5*(reduced+reduced.conj().T),check_finite=False)
            vectors=candidates@rotation
            error=applied@rotation-vectors*energies
            maximum=float(np.linalg.norm(error,axis=0).max())
            if progress:progress(dict(iteration=iteration,maximum_residual_cm=maximum,block_columns=columns))
            if maximum<=residual_tolerance_cm and np.all((energies>count.lower_cm)&(energies<count.upper_cm)):
                checks=qualify_operator_eigenpairs(
                    qualification_operator if qualification_operator is not None else op,
                    energies,vectors,workers=workers,maximum_workspace_bytes=maximum_workspace_bytes)
                checks.update(solver='recycled block inverse iteration',iterations=iteration,
                    extraction='extremal inverse Ritz then physical Rayleigh--Ritz',
                    orthogonalization=orthogonalization,
                    iteration_seconds=time.monotonic()-started,factor=shifted_inverse.diagnostics,
                    independently_counted_levels=k)
                return QualifiedEigenwindow(energies,vectors,count,checks)
            q,method=_orthonormalize_inverse_block(z)
            orthogonalization[method]=orthogonalization.get(method,0)+1
    raise ArithmeticError('block inverse iteration exhausted its iteration budget')
