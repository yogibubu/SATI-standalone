"""Transform weighted basis values and derivatives in the same volume measure."""
import numpy as np


def orthonormalize_physical_measure(values,momenta,volume_density,*,method='symmetric'):
    """Normalize values and derivatives; Cholesky preserves prefix subspaces."""
    if method not in ('symmetric','cholesky'):
        raise ValueError('Known physical-measure normalization required')
    u=np.asarray(values);p=np.asarray(momenta,complex);density=np.asarray(volume_density,float)
    if (u.ndim!=2 or p.ndim!=3 or p.shape[0]!=u.shape[0] or p.shape[2]!=u.shape[1]
            or density.shape!=(len(u),) or np.any(density<=0)
            or not all(np.all(np.isfinite(a)) for a in (u,p,density))):
        raise ValueError('Finite matching values/derivatives and positive volume density required')
    root=np.sqrt(density);u=u*root[:,None];p=p*root[:,None,None]
    overlap=u.conj().T@u;e,U=np.linalg.eigh(overlap)
    if e.min()<=1e-12*e.max():raise ValueError('Physical volume measure does not resolve the supplied basis')
    if method=='cholesky':
        L=np.linalg.cholesky(overlap)
        R=np.linalg.solve(L.conj().T,np.eye(len(L)))
    else:
        R=(U/np.sqrt(e))@U.conj().T
    u=u@R
    shape=p.shape
    p=(1j*(p.imag.reshape(-1,shape[-1])@R) if not np.iscomplexobj(R) and not np.any(p.real)
       else p.reshape(-1,shape[-1])@R).reshape(shape)
    gram=R.conj().T@overlap@R
    return dict(values=u,momenta=p,coefficient_transform=R,overlap_eigenvalues=e,overlap=gram,
        orthogonality_error=float(np.max(abs(gram-np.eye(u.shape[1])))))
