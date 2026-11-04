"""Matched rotational counterterms for an explicit adiabatic partition.

The reduced mean-path Hamiltonian and fixed-origin paired Coriolis model are
defined here by their complete force/metric jets. This is not a solution of
the fully coupled transverse vibrational Hamiltonian. Electronic field modes
are never selected or removed by this owner.
"""
import numpy as np
from matrix_dvr.full_metric import cartesian_internal_rotation_metric,KINETIC_CM_AMU_ANGSTROM2
from matrix_dvr.rotational_metric_curvature import eckart_rotational_metric_second
from .saddle_rovibrational import alpha_from_coordinate_metric


def adiabatic_rotation_partition(coordinates,masses,first,second,active_hessian_cm,
                                 active_cubic_cm,bath_hessian_cm,*,active_rotational_second_cm=None):
    x=np.asarray(coordinates,float);m=np.asarray(masses,float);D=np.asarray(first,float);DD=np.asarray(second,float)
    Ha=np.asarray(active_hessian_cm,float);Ca=np.asarray(active_cubic_cm,float);Hb=np.asarray(bath_hessian_cm,float)
    d=len(Ha);n=len(Hb);nv=d+n
    if (x.shape!=(len(m),3) or D.shape!=x.shape+(nv,) or DD.shape!=x.shape+(nv,nv)
            or Ha.shape!=(d,d) or Ca.shape!=(d,d,d) or Hb.shape!=(n,n)
            or not all(np.all(np.isfinite(a)) for a in (x,m,D,DD,Ha,Ca,Hb))):
        raise ValueError('Matching mean-reference Cartesian and potential jets required')
    active=cartesian_internal_rotation_metric(x,m,D[:,:,:d],DD[:,:,:d,:d])
    r2=eckart_rotational_metric_second(x,m,D[:,:,:d],DD[:,:,:d,:d])['rotational_second_cm']
    if active_rotational_second_cm is not None:
        supplied=np.asarray(active_rotational_second_cm,float)
        if supplied.shape!=(3,3,d,d) or not np.all(np.isfinite(supplied)) or not np.allclose(supplied,supplied.swapaxes(0,1)) or not np.allclose(supplied,supplied.swapaxes(2,3)):
            raise ValueError('Matching symmetric rotational second jet required')
        r2=supplied
    reduced=alpha_from_coordinate_metric(Ha,Ca,active['kinetic_cm'],active['kinetic_first_cm'],r2)
    ga=active['kinetic_cm'][:d,:d]
    A=np.einsum('i,ias,iat->st',m,D,D);gb=KINETIC_CM_AMU_ANGSTROM2*np.linalg.inv(A[d:,d:])
    inertia=np.einsum('i,ia,ia->',m,x,x)*np.eye(3)-np.einsum('i,ia,ib->ab',m,x,x);invI=np.linalg.inv(inertia)
    block=np.zeros((nv,nv));block[:d,:d]=ga;block[d:,d:]=gb
    cross=[]
    for j in range(nv):
        cross.append(np.einsum('i,isa->as',m,np.cross(D[:,:,j][:,None,:],D.swapaxes(1,2))+
            np.cross(x[:,None,:],DD[:,:,:,j].swapaxes(1,2))))
    vertex=np.stack([-block@c.T@invI for c in cross],axis=-1)
    a=vertex[:d,:,d:].transpose(0,2,1);dc=vertex[d:,:,:d].transpose(2,0,1)
    G=np.zeros((nv+3,nv+3));G[:nv,:nv]=block;G[nv:,nv:]=active['kinetic_cm'][d:,d:]
    H=np.zeros((nv,nv));H[:d,:d]=Ha;H[d:,d:]=Hb
    G1=np.zeros((nv+3,nv+3,nv));G1[:d,nv:,d:]=a.transpose(0,2,1);G1[nv:,:d,d:]=a.transpose(2,0,1)
    G1[d:nv,nv:,:d]=dc.transpose(1,2,0);G1[nv:,d:nv,:d]=dc.transpose(2,1,0)
    R2=np.zeros((3,3,nv,nv))
    bb=np.einsum('sba,st,tcd->bcad',a,np.linalg.inv(ga),a);bb+=bb.swapaxes(0,1)
    aa=np.einsum('sba,bc,tcd->stad',dc,np.linalg.inv(gb),dc);aa+=aa.swapaxes(0,1)
    R2[:,:,:d,:d]=aa.transpose(2,3,0,1);R2[:,:,d:,d:]=bb.transpose(2,3,0,1)
    paired=alpha_from_coordinate_metric(H,np.zeros((nv,nv,nv)),G,G1,R2)
    return dict(active_kinetic=ga,bath_kinetic=gb,bath_force=Hb,coupling_a=a,coupling_c_derivative=dc,
        equilibrium_tensor_MHz=active['kinetic_cm'][d:,d:]*29979.2458/2,
        reduced=reduced,paired=paired,
        scope='Induced mean-path metric plus fixed-origin paired rotational partition; transverse vibrational cross metric is not variationally solved')


def retained_alpha_tensor(model,omitted_modes):
    """Observable-only contraction, in the model's original Cartesian frame."""
    alpha=np.asarray(model['alpha_tensor_MHz']);omitted=tuple(omitted_modes)
    if len(set(omitted))!=len(omitted) or any(i<0 or i>=len(alpha) for i in omitted):raise ValueError('Unique valid observable rows required')
    mask=np.ones(len(alpha),bool);mask[list(omitted)]=False
    R=np.asarray(model['principal_axes']);tensor=R@(-.5*alpha[mask].sum(axis=0))@R.T
    if abs(tensor.imag).max(initial=0)>1e-6:raise ValueError('Observable row policy leaves a complex rotational tensor')
    return tensor.real
