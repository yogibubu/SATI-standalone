"""Exact conditional partition of a positive quadratic transverse potential.

Selected coordinates remain explicit; only their complement is eliminated.
The full minimum is the origin of the explicit displacements. All input field
components remain present in the Schur complement and embedding. This is a
potential/geometry operation and makes no adiabatic kinetic approximation.
"""
import numpy as np


def conditional_quadratic_bath(hessian,hessian_first,explicit_indices):
    H=np.asarray(hessian,float);H1=np.asarray(hessian_first,float)
    raw=np.asarray(explicit_indices)
    if H.ndim!=3 or H.shape[1]!=H.shape[2]:raise ValueError('Batched square bath Hessians required')
    N,n,_=H.shape
    if (raw.ndim!=1 or np.any(raw!=np.rint(raw)) or not 0<len(raw)<n
            or len(set(raw.tolist()))!=len(raw) or np.any(raw<0) or np.any(raw>=n)
            or H1.ndim!=4 or H1.shape[:3]!=H.shape or not np.allclose(H,H.swapaxes(1,2))
            or not all(np.all(np.isfinite(a)) for a in (H,H1)) or np.linalg.eigvalsh(H).min()<=0):
        raise ValueError('Positive complete Hessians, first jets and a proper explicit subset required')
    selected=raw.astype(int);fast=np.array([i for i in range(n) if i not in selected]);d=H1.shape[-1]
    ff=H[:,fast[:,None],fast[None,:]];fs=H[:,fast[:,None],selected[None,:]]
    ss=H[:,selected[:,None],selected[None,:]]
    ff1=H1[:,fast[:,None],fast[None,:]];fs1=H1[:,fast[:,None],selected[None,:]]
    ss1=H1[:,selected[:,None],selected[None,:]]
    L=-np.linalg.solve(ff,fs)
    rhs=fs1+np.einsum('nfgi,ngs->nfsi',ff1,L)
    L1=-np.linalg.solve(ff,rhs.reshape(N,len(fast),-1)).reshape(N,len(fast),len(selected),d)
    reduced=ss+fs.swapaxes(1,2)@L
    first=ss1+np.einsum('nfsi,nft->nsti',fs1,L)+np.einsum('nfs,nfti->nsti',fs,L1)
    return dict(explicit_indices=selected,fast_indices=fast,fast_hessian=ff,
        explicit_hessian=.5*(reduced+reduced.swapaxes(1,2)),explicit_hessian_first=first,
        conditional_response=L,conditional_response_first=L1)


def conditional_cartesian_frame(frame,frame_first,partition):
    Q=np.asarray(frame,float);Q1=np.asarray(frame_first,float);s=partition['explicit_indices'];f=partition['fast_indices']
    L=partition['conditional_response'];L1=partition['conditional_response_first']
    if (Q.ndim!=4 or Q1.ndim!=5 or Q1.shape[:4]!=Q.shape or len(Q)!=len(L)
            or Q.shape[-1]!=len(s)+len(f) or Q1.shape[-1]!=L1.shape[-1]):
        raise ValueError('Matching complete Cartesian bath frame and first jet required')
    P=Q[:,:,:,s]+np.einsum('niaf,nfs->nias',Q[:,:,:,f],L)
    P1=(Q1[:,:,:,s,:]+np.einsum('niafq,nfs->niasq',Q1[:,:,:,f,:],L)
        +np.einsum('niaf,nfsq->niasq',Q[:,:,:,f],L1))
    return dict(explicit_frame=P,explicit_frame_first=P1,fast_frame=Q[:,:,:,f])
