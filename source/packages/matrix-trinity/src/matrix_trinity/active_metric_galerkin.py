"""Active-coordinate weak Galerkin contraction with the full rotation metric."""
import numpy as np
from scipy.sparse import eye as sparse_eye
from .coupled_bath_galerkin import CoupledBathGalerkin


def contract_active_metric(grid_vectors,grid_momenta,kinetic_cm,potential_cm):
    """Use the supplied weighted measure and unprojected momentum actions.

    The caller transforms values and derivatives together when incorporating
    a physical volume density. No extra volume connection is inserted here.
    The returned rotational resolvent spans the complete supplied active basis.
    """
    u=np.asarray(grid_vectors,complex);p=np.asarray(grid_momenta,complex)
    g=np.asarray(kinetic_cm,float);V=np.asarray(potential_cm,float)
    if u.ndim!=2 or p.ndim!=3:raise ValueError('Weighted values and momentum actions required')
    N,M=u.shape;d=p.shape[1]
    if (p.shape!=(N,d,M) or g.shape!=(N,d+3,d+3) or V.shape!=(N,)
            or not all(np.all(np.isfinite(a)) for a in (u,p,g,V))
            or not np.allclose(u.conj().T@u,np.eye(M),atol=1e-8,rtol=1e-8)
            or not np.allclose(g,g.swapaxes(-1,-2),atol=1e-8)
            or np.linalg.eigvalsh(g).min()<=0):
        raise ValueError('Orthonormal weighted basis and positive complete metric required')
    def project(left,field,right):return left.conj().T@(field[:,None]*right)
    H=project(u,V,u)
    for i in range(d):
        for j in range(d):H+=.5*project(p[:,i],g[:,i,j],p[:,j])
    error=float(np.max(abs(H-H.conj().T)))
    if error>1e-7:raise ValueError('Active weak Hamiltonian is not Hermitian')
    H=.5*(H+H.conj().T);identity=sparse_eye(1,format='csr')
    linear=[];quadratic=[]
    for a in range(3):
        L=np.zeros((M,M),complex)
        for i in range(d):
            first=.5*project(p[:,i],g[:,i,d+a],u);L+=first+first.conj().T
        linear.append(((L,identity),))
        quadratic.append(tuple(((.5*project(u,g[:,d+a,d+b],u),identity),) for b in range(3)))
    return CoupledBathGalerkin(H,tuple(linear),tuple(quadratic),M,1,((0,),),error,'complete positive weak active contraction')
