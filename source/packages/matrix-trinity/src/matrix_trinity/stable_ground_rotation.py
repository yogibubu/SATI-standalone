"""Ground-state rotational tensor for a complete positive local bath.

This contracts every mode of the supplied bath. It does not transform or
truncate an electronic field and does not select observable rows.
"""
import numpy as np


def stable_ground_rotation(hessian_cm, cubic_cm, metric_cm,
                           metric_first_cm, rotational_second_cm):
    H=np.asarray(hessian_cm,float); C=np.asarray(cubic_cm,float)
    G=np.asarray(metric_cm,float); G1=np.asarray(metric_first_cm,float)
    R2=np.asarray(rotational_second_cm,float); n=len(H)
    if (H.shape!=(n,n) or C.shape!=(n,n,n) or G.shape!=(n+3,n+3)
            or G1.shape!=(n+3,n+3,n) or R2.shape!=(3,3,n,n)
            or not all(np.all(np.isfinite(x)) for x in (H,C,G,G1,R2))):
        raise ValueError('Matching finite local force and metric tensors required')
    if (not np.allclose(H,H.T) or not np.allclose(G,G.T)
            or not np.allclose(G1,G1.swapaxes(0,1))
            or not np.allclose(R2,R2.swapaxes(0,1))
            or not np.allclose(R2,R2.swapaxes(2,3))
            or np.max(abs(G[:n,n:]),initial=0)>1e-6):
        raise ValueError('Symmetric tensors in a common local Eckart frame required')
    L=np.linalg.cholesky(G[:n,:n]); eigen,U=np.linalg.eigh(L.T@H@L)
    if eigen.min()<=.0001:raise ValueError('Positive bath frequencies above .01 cm-1 required')
    w=np.sqrt(eigen); A=L@U/np.sqrt(w)[None,:]; P=np.linalg.inv(A)
    covariance=A@A.T/2; momentum_covariance=P.T@P/2
    force=np.einsum('ijk,jk->i',C,covariance)
    force+=np.einsum('jki,jk->i',G1[:n,:n],momentum_covariance)
    mean=-.5*np.linalg.solve(H,force)
    curvature=np.einsum('abij,ij->ab',R2,covariance)/4
    displacement=np.einsum('abi,i->ab',G1[n:,n:],mean)/2
    vertex=np.einsum('is,srt,tj->irj',P,G1[:n,n:,:],A,optimize=True)
    paired=vertex+vertex.transpose(2,1,0)
    virtual=-np.einsum('irj,isj,ij->rs',paired,paired,1/(w[:,None]+w[None,:]),optimize=True)/8
    scale=29979.2458
    return dict(tensor_MHz=(curvature+displacement+virtual)*scale,
        curvature_MHz=curvature*scale,displacement_MHz=displacement*scale,
        virtual_MHz=virtual*scale,mean_coordinates=mean,frequencies_cm=w)
