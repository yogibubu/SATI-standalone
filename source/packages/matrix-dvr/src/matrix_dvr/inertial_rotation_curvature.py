"""Inverse-inertia curvature in any smooth body-frame Cartesian section."""
import numpy as np
from .full_metric import KINETIC_CM_AMU_ANGSTROM2


def inverse_inertia_second(coordinates_angstrom,masses_amu,first,second):
    """Return true second derivatives of the inverse-inertia kinetic block.

    These are the inertial contribution, separate from the rotational
    connection's Schur completion. No zero Eckart cross block is required.
    Coordinates and derivatives must share one body-frame section.
    """
    x=np.asarray(coordinates_angstrom,float);m=np.asarray(masses_amu,float)
    D=np.asarray(first,float);DD=np.asarray(second,float)
    if (x.shape!=(len(m),3) or D.ndim!=3 or D.shape[:2]!=x.shape
            or DD.shape!=x.shape+(D.shape[-1],)*2 or np.any(m<=0)
            or not all(np.all(np.isfinite(a)) for a in (x,m,D,DD))
            or not np.allclose(DD,DD.swapaxes(-1,-2))):
        raise ValueError('Matching finite Cartesian jets and positive masses required')
    x=x-np.average(x,axis=0,weights=m);D=D-np.einsum('i,ias->as',m,D)[None]/m.sum()
    DD=DD-np.einsum('i,iast->ast',m,DD)[None]/m.sum();n=D.shape[-1]
    I=np.einsum('i,ia,ia->',m,x,x)*np.eye(3)-np.einsum('i,ia,ib->ab',m,x,x);inv=np.linalg.inv(I)
    I1=2*np.einsum('i,ia,ias->s',m,x,D)[:,None,None]*np.eye(3)
    cross=np.einsum('i,ia,ibs->sab',m,x,D);I1-=cross+cross.swapaxes(1,2)
    result=np.empty((3,3,n,n))
    for i in range(n):
        for j in range(i+1):
            Ei,Ej,F=D[:,:,i],D[:,:,j],DD[:,:,i,j]
            I2=2*(np.einsum('i,ia,ia->',m,Ei,Ej)+np.einsum('i,ia,ia->',m,x,F))*np.eye(3)
            cross=np.einsum('i,ia,ib->ab',m,x,F)+np.einsum('i,ia,ib->ab',m,Ei,Ej);I2-=cross+cross.T
            value=inv@I1[i]@inv@I1[j]@inv+inv@I1[j]@inv@I1[i]@inv-inv@I2@inv
            result[:,:,i,j]=result[:,:,j,i]=KINETIC_CM_AMU_ANGSTROM2*value
    return result
