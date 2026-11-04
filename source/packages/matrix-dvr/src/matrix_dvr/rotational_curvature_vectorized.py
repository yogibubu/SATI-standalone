"""Vectorized exact reference-Eckart rotational metric second derivative."""
import numpy as np
from .full_metric import KINETIC_CM_AMU_ANGSTROM2


def eckart_rotational_metric_second_vectorized(coordinates_angstrom,masses_amu,tangents,tangent_derivatives):
    x=np.asarray(coordinates_angstrom,float);m=np.asarray(masses_amu,float)
    D=np.asarray(tangents,float);DD=np.asarray(tangent_derivatives,float)
    if (x.shape!=(len(m),3) or D.ndim!=3 or D.shape[:2]!=x.shape or DD.shape!=x.shape+(D.shape[-1],)*2
            or np.any(m<=0) or not all(np.all(np.isfinite(a)) for a in (x,m,D,DD))):raise ValueError('Matching finite Cartesian jets and positive masses required')
    x=x-np.average(x,axis=0,weights=m);D=D-np.einsum('i,ias->as',m,D)[None]/m.sum();DD=DD-np.einsum('i,iast->ast',m,DD)[None]/m.sum()
    V=D.transpose(0,2,1);cross=np.einsum('i,isa->as',m,np.cross(x[:,None,:],V))
    if abs(cross).max()>1e-8:raise ValueError('Reference Eckart tangent required')
    I=np.einsum('i,ia,ia->',m,x,x)*np.eye(3)-np.einsum('i,ia,ib->ab',m,x,x);invI=np.linalg.inv(I);invA=np.linalg.inv(np.einsum('i,ias,iat->st',m,D,D))
    xd=np.einsum('i,ia,ibs->sab',m,x,D);I1=2*np.einsum('i,ia,ias->s',m,x,D)[:,None,None]*np.eye(3)-xd-xd.swapaxes(-1,-2)
    C1=np.einsum('i,ijsa->jas',m,np.cross(V[:,:,None,:],V[:,None,:,:]))
    C1+=np.einsum('i,isja->jas',m,np.cross(x[:,None,None,:],DD.transpose(0,2,3,1)))
    ff=np.einsum('i,ias,ibt->stab',m,D,D);sx=np.einsum('i,iast,ib->stab',m,DD,x)
    scalar=np.einsum('i,ias,iat->st',m,D,D)+np.einsum('i,ia,iast->st',m,x,DD)
    I2=2*scalar[:,:,None,None]*np.eye(3)-ff-ff.swapaxes(-1,-2)-sx-sx.swapaxes(-1,-2)
    value=np.einsum('ab,sbc,cd,tde,ef->staf',invI,I1,invI,I1,invI,optimize=True)
    inertial=value+value.swapaxes(0,1)-np.einsum('ab,stbc,cd->stad',invI,I2,invI,optimize=True)
    value=np.einsum('ab,sbc,cd,ted,ef->staf',invI,C1,invA,C1,invI,optimize=True)
    connection=value+value.swapaxes(0,1);scale=KINETIC_CM_AMU_ANGSTROM2
    inertial=scale*inertial.transpose(2,3,0,1);connection=scale*connection.transpose(2,3,0,1)
    return dict(rotational_second_cm=inertial+connection,inertial_second_cm=inertial,connection_second_cm=connection)
