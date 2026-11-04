"""Analytic reference Eckart curvature of the rotational metric."""
import numpy as np
from .full_metric import KINETIC_CM_AMU_ANGSTROM2

def eckart_rotational_metric_second(coordinates_angstrom,masses_amu,tangents,tangent_derivatives):
    """Analytic rotational metric curvature at a reference Eckart point.

    The zero internal/rotation cross block removes all dependence on third
    Cartesian geometry derivatives. Both inertial curvature and the linear
    Coriolis completion are retained and returned separately.
    """
    x=np.array(coordinates_angstrom,float,copy=True);m=np.asarray(masses_amu,float)
    D=np.array(tangents,float,copy=True);DD=np.array(tangent_derivatives,float,copy=True)
    if (x.ndim!=2 or x.shape[1]!=3 or m.shape!=(len(x),) or np.any(m<=0)
            or D.ndim!=3 or D.shape[:2]!=x.shape or DD.shape!=x.shape+(D.shape[-1],)*2
            or not all(np.all(np.isfinite(a)) for a in (x,m,D,DD))):
        raise ValueError('Compatible finite geometry jets and positive masses required')
    n=D.shape[-1];x-=np.average(x,axis=0,weights=m)
    D-=np.einsum('i,ias->as',m,D)[None]/m.sum()
    DD-=np.einsum('i,iast->ast',m,DD)[None]/m.sum()
    C=np.einsum('i,isa->as',m,np.cross(x[:,None,:],D.swapaxes(1,2)))
    if np.max(abs(C))>1e-8:raise ValueError('Rotational curvature requires a reference Eckart tangent')
    I=np.einsum('i,ia,ia->',m,x,x)*np.eye(3)-np.einsum('i,ia,ib->ab',m,x,x)
    invI=np.linalg.inv(I);invA=np.linalg.inv(np.einsum('i,ias,iat->st',m,D,D))
    I1=[];C1=[]
    for j in range(n):
        E=D[:,:,j]
        I1.append(2*np.einsum('i,ia,ia->',m,x,E)*np.eye(3)-np.einsum('i,ia,ib->ab',m,E,x)-np.einsum('i,ia,ib->ab',m,x,E))
        C1.append(np.einsum('i,isa->as',m,np.cross(E[:,None,:],D.swapaxes(1,2))+np.cross(x[:,None,:],DD[:,:,:,j].swapaxes(1,2))))
    inertial=np.empty((3,3,n,n));connection=np.empty_like(inertial)
    for i in range(n):
        for j in range(i+1):
            Ei,Ej,F=D[:,:,i],D[:,:,j],DD[:,:,i,j]
            I2=2*(np.einsum('i,ia,ia->',m,Ei,Ej)+np.einsum('i,ia,ia->',m,x,F))*np.eye(3)
            I2-=np.einsum('i,ia,ib->ab',m,F,x)+np.einsum('i,ia,ib->ab',m,x,F)
            I2-=np.einsum('i,ia,ib->ab',m,Ei,Ej)+np.einsum('i,ia,ib->ab',m,Ej,Ei)
            a=invI@I1[i]@invI@I1[j]@invI+invI@I1[j]@invI@I1[i]@invI-invI@I2@invI
            b=invI@(C1[i]@invA@C1[j].T+C1[j]@invA@C1[i].T)@invI
            inertial[:,:,i,j]=inertial[:,:,j,i]=KINETIC_CM_AMU_ANGSTROM2*a
            connection[:,:,i,j]=connection[:,:,j,i]=KINETIC_CM_AMU_ANGSTROM2*b
    return dict(rotational_second_cm=inertial+connection,inertial_second_cm=inertial,connection_second_cm=connection)
