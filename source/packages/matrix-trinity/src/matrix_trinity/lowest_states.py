"""Lowest Hermitian states with a checked absolute eigenpair residual."""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def lowest_hermitian_states(hamiltonian,state_count=6,*,relative_tolerance=1e-12,
        absolute_tolerance=1e-8,maximum_refinements=3):
    """Tighten ARPACK's relative criterion until the absolute residual passes.

    No Hamiltonian element, eigenvalue or physical constraint is modified.
    Failed convergence raises with the measured residual instead of silently
    accepting eigenpairs based on a relative solver flag alone.
    """
    H=hamiltonian;n=H.shape[0]
    if (H.shape!=(n,n) or not 1<=state_count<n or relative_tolerance<=0
            or absolute_tolerance<=0 or maximum_refinements<1):
        raise ValueError('Hermitian operator, valid state count and positive tolerances required')
    start=np.linspace(1,2,n);history=[]
    for refinement in range(maximum_refinements):
        tolerance=max(np.finfo(float).eps,relative_tolerance*.01**refinement)
        if n<=256:
            dense=H if isinstance(H,np.ndarray) else H@np.eye(n)
            e,U=eigh(dense,subset_by_index=(0,state_count-1))
        else:
            e,U=eigsh(H,k=state_count,which='SA',tol=tolerance,v0=start)
            order=np.argsort(e);e,U=e[order],U[:,order]
        residual=np.linalg.norm(H@U-U*e,axis=0)
        history.append(dict(relative_tolerance=tolerance,maximum_residual=float(residual.max())))
        if residual.max()<absolute_tolerance:
            return dict(energies=e,vectors=U,residuals=residual,refinements=history)
        start=U[:,0]
    raise RuntimeError(f'Absolute eigenpair residual {residual.max():.6g} exceeds {absolute_tolerance:.6g}')
