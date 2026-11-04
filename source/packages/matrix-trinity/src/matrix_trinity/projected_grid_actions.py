"""Exact projected grid-operator actions without forming dense operators."""
import numpy as np


def _matmul(left,right):
    # Exact constant phases permit real BLAS; small parts are never dropped.
    from .coupled_bath_galerkin import _constant_phase_real
    pa,a=_constant_phase_real(left);pb,b=_constant_phase_real(right)
    if pa is None or pb is None:return left@right
    phase=pa*pb
    if np.isreal(phase):phase=float(np.real(phase))
    return phase*(a@b)


def projected_grid_action(left,right,coefficients,*,field=None,hermitian=False):
    """Apply L† diag(field) R to state columns, optionally Hermitize it."""
    L=np.asarray(left);R=np.asarray(right);C=np.asarray(coefficients)
    if L.ndim!=2 or R.shape!=L.shape or C.ndim!=2 or C.shape[0]!=L.shape[1]:
        raise ValueError('Matching grid bases and state coefficient columns required')
    f=np.ones(len(L)) if field is None else np.asarray(field)
    if f.shape!=(len(L),) or not all(np.all(np.isfinite(a)) for a in (L,R,C,f)):
        raise ValueError('Matching finite projected operator data required')
    result=_matmul(L.conj().T,f[:,None]*_matmul(R,C))
    if hermitian:result=.5*(result+_matmul(R.conj().T,f.conj()[:,None]*_matmul(L,C)))
    return result
