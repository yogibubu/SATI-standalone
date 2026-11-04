"""Optional native MKL primitives for finite vibrational spectral acquisition.

The library is supplied explicitly; importing this module loads no native code.
Each process owns its thread setting and PARDISO factorization handles.
"""
import ctypes as ct
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix, triu


class MKLSpectralBackend:
    def __init__(self, library_path, *, workers=1):
        if not isinstance(workers,int) or workers<=0:
            raise ValueError('positive integer worker count required')
        self.library_path = str(Path(library_path).resolve(strict=True))
        self.library = ct.CDLL(self.library_path)
        self.library.MKL_Set_Interface_Layer.argtypes = [ct.c_int]
        self.library.MKL_Set_Interface_Layer.restype = ct.c_int
        if self.library.MKL_Set_Interface_Layer(0)!=0:
            raise RuntimeError('MKL LP64 integer interface unavailable')
        self.library.MKL_Set_Num_Threads.argtypes = [ct.c_int]
        self.library.MKL_Set_Num_Threads.restype = None
        self.library.MKL_Set_Num_Threads(workers)
        self.library.MKL_Set_Dynamic.argtypes = [ct.c_int]
        self.library.MKL_Set_Dynamic.restype = None
        self.library.MKL_Set_Dynamic(0)
        self.workers = workers

    def dense_window(self, matrix, ceiling_cm):
        """All eigenpairs below a ceiling, with the native LAPACKE LP64 API."""
        matrix = csr_matrix(matrix)
        if np.iscomplexobj(matrix.data):
            raise ValueError('native dense window currently requires a real symmetric matrix')
        n = matrix.shape[0]
        if matrix.shape!=(n,n) or not n or n>np.iinfo(np.int32).max or not np.all(np.isfinite(matrix.data)):
            raise ValueError('nonempty LP64 square matrix required')
        difference=matrix-matrix.T
        if np.max(abs(difference.data),initial=0)>1e-12*max(1.,np.max(abs(matrix.data),initial=0)):
            raise ValueError('symmetric matrix required')
        a = np.array(matrix.toarray(),dtype=np.float64,order='F')
        vectors = np.empty((n,n),dtype=np.float64,order='F')
        energies = np.empty(n,np.float64)
        support = np.empty(2*n,np.int32)
        count = ct.c_int()
        pointer = ct.c_void_p
        function = self.library.LAPACKE_dsyevr
        function.restype = ct.c_int
        function.argtypes = [ct.c_int,ct.c_char,ct.c_char,ct.c_char,
            ct.c_int,pointer,ct.c_int,ct.c_double,ct.c_double,ct.c_int,ct.c_int,
            ct.c_double,ct.POINTER(ct.c_int),pointer,pointer,ct.c_int,pointer]
        info = function(102,b'V',b'V',b'L',n,a.ctypes.data,n,
            -np.finfo(float).max,float(ceiling_cm),0,0,0.,ct.byref(count),
            energies.ctypes.data,vectors.ctypes.data,n,support.ctypes.data)
        if info!=0:
            raise ArithmeticError(f'MKL LAPACKE dsyevr failed with info={info}')
        return energies[:count.value].copy(),vectors[:,:count.value].copy(order='F')

    def shifted_factor(self, matrix, shift_cm):
        return SymmetricPardisoFactor(self,matrix,shift_cm)


class SymmetricPardisoFactor:
    """Real symmetric shifted inverse with an unperturbed inertia count.

    Bunch--Kaufman pivots are used. A count is refused if pivots were perturbed
    or zero eigenvalues remain. Numerical roundoff is not interval arithmetic.
    """
    def __init__(self, backend, matrix, shift_cm):
        original = csr_matrix(matrix)
        if (original.shape[0]!=original.shape[1] or not original.shape[0]
                or np.iscomplexobj(original.data) or not np.isfinite(shift_cm)
                or original.shape[0]>np.iinfo(np.int32).max
                or original.nnz>np.iinfo(np.int32).max
                or not np.all(np.isfinite(original.data))):
            raise ValueError('finite real symmetric LP64 matrix and finite shift required')
        original=original.astype(np.float64,copy=False)
        difference = original-original.T
        if np.max(abs(difference.data),initial=0)>1e-12*max(1.,np.max(abs(original.data))):
            raise ValueError('symmetric matrix required')
        self.backend,self.shift_cm = backend,float(shift_cm)
        self.matrix = triu(original,format='csr')
        self.original_diagonal = original.diagonal().copy()
        self.matrix.setdiag(original.diagonal()-shift_cm)
        self.matrix.sort_indices()
        self.data = np.ascontiguousarray(self.matrix.data,dtype=np.float64)
        self.indptr = np.ascontiguousarray(self.matrix.indptr,dtype=np.int32)
        self.indices = np.ascontiguousarray(self.matrix.indices,dtype=np.int32)
        self.pt = (ct.c_void_p*64)()
        self.iparm = np.zeros(64,np.int32)
        self.permutation = np.zeros(original.shape[0],np.int32)
        self.n,self.mtype = ct.c_int(original.shape[0]),ct.c_int(-2)
        self.maxfct,self.mnum,self.message,self.error = ct.c_int(1),ct.c_int(1),ct.c_int(0),ct.c_int(0)
        self.closed = False
        init = backend.library.pardisoinit
        init.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int),ct.c_void_p]
        init.restype = None
        init(self.pt,ct.byref(self.mtype),self.iparm.ctypes.data)
        self.iparm[0]=1
        self.iparm[1]=2
        self.iparm[7]=2
        self.iparm[9]=16
        self.iparm[10]=1
        self.iparm[12]=1
        self.iparm[17]=-1
        self.iparm[20]=1
        self.iparm[34]=1
        integer = ct.POINTER(ct.c_int)
        self.function = backend.library.pardiso
        self.function.argtypes = [ct.c_void_p,integer,integer,integer,integer,integer,
            ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,integer,ct.c_void_p,
            integer,ct.c_void_p,ct.c_void_p,integer]
        self.function.restype = None
        started = time.monotonic()
        try:
            self._call(12,np.zeros(1),np.zeros(1),1)
            positive,negative = int(self.iparm[21]),int(self.iparm[22])
            if self.iparm[13] or positive+negative!=self.n.value:
                raise ArithmeticError('PARDISO inertia refused: perturbed or unresolved pivots')
            self.negative_count = negative
            self.diagnostics = dict(shift_cm=float(shift_cm),positive_count=positive,
                negative_count=negative,perturbed_pivots=int(self.iparm[13]),
                factor_nonzeros=int(self.iparm[17]),
                peak_memory_kib=int(max(self.iparm[14],self.iparm[15]+self.iparm[16])),
                factor_seconds=time.monotonic()-started,workers=backend.workers)
        except BaseException:
            self.close();raise

    def refactor(self, shift_cm):
        """Reuse ordering and symbolic analysis when only the shift changes."""
        if self.closed:raise RuntimeError('factor is closed')
        if not np.isfinite(shift_cm):raise ValueError('finite shift required')
        self.shift_cm=float(shift_cm)
        # Every upper-triangle row has its explicit diagonal first.
        self.data[self.indptr[:-1]]=self.original_diagonal-self.shift_cm
        started=time.monotonic()
        self._call(22,np.zeros(1),np.zeros(1),1)
        positive,negative=int(self.iparm[21]),int(self.iparm[22])
        if self.iparm[13] or positive+negative!=self.n.value:
            raise ArithmeticError('PARDISO inertia refused: perturbed or unresolved pivots')
        self.negative_count=negative
        self.diagnostics=dict(self.diagnostics,shift_cm=self.shift_cm,
            positive_count=positive,negative_count=negative,perturbed_pivots=int(self.iparm[13]),
            factor_seconds=time.monotonic()-started,symbolic_analysis_reused=True)
        return self

    def _call(self, phase, rhs, output, count):
        phase,nrhs = ct.c_int(phase),ct.c_int(count)
        self.error.value=0
        self.function(self.pt,ct.byref(self.maxfct),ct.byref(self.mnum),ct.byref(self.mtype),
            ct.byref(phase),ct.byref(self.n),self.data.ctypes.data,self.indptr.ctypes.data,
            self.indices.ctypes.data,self.permutation.ctypes.data,ct.byref(nrhs),
            self.iparm.ctypes.data,ct.byref(self.message),rhs.ctypes.data,output.ctypes.data,
            ct.byref(self.error))
        if self.error.value:
            raise ArithmeticError(f'PARDISO phase {phase.value} failed with error={self.error.value}')

    def solve(self, rhs):
        if self.closed:raise RuntimeError('factor is closed')
        rhs = np.array(rhs,dtype=np.float64,order='F',copy=True)
        if rhs.ndim not in (1,2) or rhs.shape[0]!=self.n.value or not np.all(np.isfinite(rhs)):
            raise ValueError('finite compatible right-hand sides required')
        output = np.empty_like(rhs,order='F')
        self._call(33,rhs,output,1 if rhs.ndim==1 else rhs.shape[1])
        if not np.all(np.isfinite(output)):
            raise FloatingPointError('nonfinite PARDISO inverse action')
        return output

    def close(self):
        if not self.closed:
            try:self._call(-1,np.zeros(1),np.zeros(1),1)
            finally:self.closed=True

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def lowest_eigenvalue_inertia_bracket(hamiltonian, library_path, *,
                                     tolerance_cm=.02, workers=1):
    """Numerically enclose the lowest finite root with unperturbed inertia.

    An iterative eigenpair only proposes a narrow interval. Zero/positive
    negative-pivot counts certify its two ends independently of convergence
    of that proposal. If it missed a lower root, bisection starts from the
    Hamiltonian's Gershgorin lower enclosure. Floating point inertia is not
    an interval-arithmetic certificate.
    """
    from scipy.sparse.linalg import eigsh
    if not np.isfinite(tolerance_cm) or tolerance_cm <= 0:
        raise ValueError('positive finite eigenvalue bracket width required')
    matrix=hamiltonian.matrix_cm
    if matrix.shape[0] < 2:
        raise ValueError('at least two finite basis states required')
    estimate,vectors=eigsh(matrix,k=1,which='SA',tol=1e-10,
                          v0=np.ones(matrix.shape[0]))
    estimate=float(estimate[0])
    residual=float(np.linalg.norm(matrix@vectors[:,0]-estimate*vectors[:,0]))
    backend=MKLSpectralBackend(library_path,workers=workers)
    lower=estimate-tolerance_cm/2
    upper=estimate+tolerance_cm/2
    records=[]
    with backend.shifted_factor(matrix,lower) as factor:
        records.append(dict(factor.diagnostics))
        if factor.negative_count:
            lower=float(hamiltonian.spectral_interval_cm[0])
            factor.refactor(lower);records.append(dict(factor.diagnostics))
            if factor.negative_count:raise ArithmeticError('Gershgorin/inertia lower-bound disagreement')
        factor.refactor(upper);records.append(dict(factor.diagnostics))
        if factor.negative_count<1:raise ArithmeticError('proposed upper bracket has no eigenvalue below it')
        for _ in range(64):
            if upper-lower <= tolerance_cm*(1+1e-10):break
            middle=(lower+upper)/2
            factor.refactor(middle);records.append(dict(factor.diagnostics))
            if factor.negative_count:upper=middle
            else:lower=middle
        else:raise RuntimeError('lowest-root inertia bracket did not reach its width')
    return dict(lower_bound_cm=lower,upper_bound_cm=upper,
        proposal_eigenvalue_cm=estimate,proposal_residual_cm=residual,
        width_cm=upper-lower,inertia_evaluations=records,
        bounds_include_roundoff_enclosure=False,
        infinite_basis_convergence_certified=False)


__all__ = ['MKLSpectralBackend','SymmetricPardisoFactor',
           'lowest_eigenvalue_inertia_bracket']
