"""Mass-weighted transverse harmonic field for arbitrary invariant constraints."""
import numpy as np
from .full_metric import KINETIC_CM_AMU_ANGSTROM2
HARTREE_TO_CM=219474.6313632


def constrained_transverse_harmonic(coordinates_angstrom,masses_amu,gradient_hartree_per_A,
                                    hessian_hartree_per_A2,constraint_first,constraint_second):
    x=np.asarray(coordinates_angstrom,float);m=np.asarray(masses_amu,float);n=len(m)
    g=np.asarray(gradient_hartree_per_A,float).reshape(-1);H=np.asarray(hessian_hartree_per_A2,float)
    B=np.asarray(constraint_first,float);curv=np.asarray(constraint_second,float)
    if x.shape!=(n,3) or H.shape!=(3*n,3*n) or B.shape[1]!=3*n or curv.shape!=(len(B),3*n,3*n) or np.any(m<=0):raise ValueError('Compatible geometry, Cartesian field and constraint jets required')
    if not all(np.all(np.isfinite(v)) for v in (x,m,g,H,B,curv)):raise ValueError('Finite source field required')
    root=np.repeat(np.sqrt(m),3);center=x-np.average(x,axis=0,weights=m)
    translation=np.zeros((3,3*n));rotation=np.zeros_like(translation)
    for k in range(3):
        translation[k]=np.tile(np.eye(3)[k],(n,1)).ravel()*root
        rotation[k]=np.cross(np.eye(3)[k],center).ravel()*root
    W=B/root;lagrange=np.linalg.solve(W@W.T,W@(g/root))
    rows=np.vstack([translation,rotation,W]);rows/=np.linalg.norm(rows,axis=1)[:,None]
    _,s,V=np.linalg.svd(rows,full_matrices=True);rank=int(sum(s>1e-10*s[0]));expected=6+len(B)
    if rank!=expected:raise ValueError('Rigid and active constraint rows are not independent')
    Q=V[rank:].T;L=H-np.einsum('k,kij->ij',lagrange,curv);projected=Q.T@(L/root[:,None]/root[None,:])@Q
    projected=.5*(projected+projected.T);eigen=np.linalg.eigvalsh(projected)
    if min(eigen)<=0:raise ValueError('Nonpositive observed constrained transverse Hessian')
    forces=Q.T@(g/root);step=-np.linalg.solve(projected,forces)
    frequencies=np.sqrt(KINETIC_CM_AMU_ANGSTROM2*HARTREE_TO_CM*eigen)
    return dict(frequencies_cm=frequencies,ZPE_cm=float(.5*sum(frequencies)),
        quadratic_stationarity_energy_change_cm=float(.5*forces@step*HARTREE_TO_CM),
        maximum_projected_mass_weighted_force_hartree=float(abs(forces).max()),
        lagrange_multipliers_hartree=lagrange,minimum_transverse_eigenvalue=float(min(eigen)))
