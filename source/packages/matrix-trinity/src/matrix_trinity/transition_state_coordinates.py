"""Signed Wilson GF modalization for an index-one stationary saddle."""
import numpy as np
from .curvilinear_vpt2 import _validate_field


def modalize_transition_state(field):
    from matrix_gf.harmonic import CM_PER_HARTREE, HESSIAN_EIGENVALUE_TO_CM
    f2, f3, f4, metric, g1, g2 = _validate_field(field)
    values, vectors = np.linalg.eigh(metric)
    root = (vectors*np.sqrt(values))@vectors.T
    eigenvalues, modes = np.linalg.eigh(root@f2@root)
    if np.count_nonzero(eigenvalues < 0) != 1 or np.any(eigenvalues == 0):
        raise ValueError('native saddle GF field requires exactly one negative eigenvalue and no zero modes')
    w = np.sqrt(abs(eigenvalues))*HESSIAN_EIGENVALUE_TO_CM
    modes = root@modes
    displacement = modes/ (np.sqrt(w*CM_PER_HARTREE)/HESSIAN_EIGENVALUE_TO_CM)[None,:]
    momentum = np.linalg.inv(displacement)
    scale = HESSIAN_EIGENVALUE_TO_CM**2/CM_PER_HARTREE
    return (np.sign(eigenvalues)*w,
            CM_PER_HARTREE*np.einsum('abc,ai,bj,ck->ijk',f3,displacement,displacement,displacement,optimize=True),
            CM_PER_HARTREE*np.einsum('abcd,ai,bj,ck,dl->ijkl',f4,displacement,displacement,displacement,displacement,optimize=True),
            scale*np.einsum('ia,jb,abc,ck->ijk',momentum,momentum,g1,displacement,optimize=True),
            scale*np.einsum('ia,jb,abcd,ck,dl->ijkl',momentum,momentum,g2,displacement,displacement,optimize=True))
