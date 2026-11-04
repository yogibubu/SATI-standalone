"""Vectorized analytic mass-Eckart geometry jets, without mode selection."""
import numpy as np


def mass_eckart_cartesian_jet_vectorized(coordinates_angstrom,reference_angstrom,
                                       masses_amu,tangents,second_derivatives=None):
    x=np.asarray(coordinates_angstrom,float);ref=np.asarray(reference_angstrom,float)
    m=np.asarray(masses_amu,float);D=np.asarray(tangents,float)
    if (x.shape!=(len(m),3) or ref.shape!=x.shape or D.ndim!=3 or D.shape[:2]!=x.shape
            or np.any(m<=0) or not all(np.all(np.isfinite(a)) for a in (x,ref,m,D))):
        raise ValueError('Matching finite Cartesian geometry jets and positive masses required')
    n=D.shape[-1];x=x-np.average(x,axis=0,weights=m);ref=ref-np.average(ref,axis=0,weights=m)
    D=D-np.einsum('i,ias->as',m,D)[None]/m.sum();U,_,V=np.linalg.svd((x*m[:,None]).T@ref)
    R=U@np.diag([1.,1.,np.linalg.det(U@V)])@V;z=x@R;v=np.einsum('ias,ab->isb',D,R)
    inertia=np.einsum('i,ia,ia->',m,ref,z)*np.eye(3)-np.einsum('i,ia,ib->ab',m,z,ref)
    if np.linalg.cond(inertia)>1e12:raise ValueError('Singular Eckart orientation derivative')
    omega=np.linalg.solve(inertia,np.einsum('i,isa->as',m,np.cross(ref[:,None,:],v))).T
    zw=np.cross(z[:,None,:],omega[None]);dz=v+zw;dd=None
    if second_derivatives is not None:
        DD=np.asarray(second_derivatives,float)
        if DD.shape!=z.shape+(n,n) or not np.all(np.isfinite(DD)) or not np.allclose(DD,DD.swapaxes(-1,-2),atol=1e-9,rtol=1e-9):raise ValueError('Matching symmetric second Cartesian jet required')
        DD=DD-np.einsum('i,iast->ast',m,DD)[None]/m.sum();raw=np.einsum('iast,ab->istb',DD,R)
        value=raw+np.cross(v[:,:,None,:],omega[None,None,:,:])+np.cross(v[:,None,:,:],omega[None,:,None,:])
        value+=.5*(np.cross(zw[:,None,:,:],omega[None,:,None,:])+np.cross(zw[:,:,None,:],omega[None,None,:,:]))
        rhs=np.einsum('i,ista->ast',m,np.cross(ref[:,None,None,:],value))
        omega2=np.linalg.solve(inertia,rhs.reshape(3,-1)).reshape(3,n,n).transpose(1,2,0)
        dd=(value+np.cross(z[:,None,None,:],omega2[None])).transpose(0,3,1,2)
        dd=.5*(dd+dd.swapaxes(-1,-2))
    return dict(coordinates_angstrom=z,tangents=dz.transpose(0,2,1),second_derivatives=dd,rotation=R,
        eckart_residual=float(abs(np.einsum('i,ia->a',m,np.cross(ref,z))).max()))
