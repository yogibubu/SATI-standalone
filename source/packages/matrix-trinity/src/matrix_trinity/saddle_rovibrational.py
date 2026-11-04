"""Full modal alpha contraction for stable or saddle curvilinear models.

All modes and mixed interactions remain in the supplied tensors. Signed
frequencies specify analytic continuation; observable row selection is a
separate operation. No imaginary frequency is made physically positive.
"""
import numpy as np


def alpha_from_coordinate_metric(force_hessian_cm,cubic_cm,metric_zero_cm,
                                 metric_first_cm,rotational_metric_second_cm):
    """Modalize a complete coordinate model without removing unstable modes."""
    H=np.asarray(force_hessian_cm,float);C=np.asarray(cubic_cm,float)
    G=np.asarray(metric_zero_cm,float);G1=np.asarray(metric_first_cm,float);R2=np.asarray(rotational_metric_second_cm,float)
    if H.ndim!=2 or H.shape[0]!=H.shape[1]:raise ValueError('Square force Hessian required')
    n=len(H)
    if (C.shape!=(n,n,n) or G.shape!=(n+3,n+3) or G1.shape!=(n+3,n+3,n)
            or R2.shape!=(3,3,n,n) or not all(np.all(np.isfinite(a)) for a in (H,C,G,G1,R2))
            or not np.allclose(H,H.T,atol=1e-9)):
        raise ValueError('Matching full coordinate force and metric jets required')
    L=np.linalg.cholesky(G[:n,:n]);e,U=np.linalg.eigh(L.T@H@L);w=np.sign(e)*np.sqrt(abs(e))
    if np.any(abs(w)<=.01):raise ValueError('Zero-frequency coordinate model is outside the modal alpha contract')
    A=L@U/np.sqrt(abs(w))[None,:];P=np.eye(n+3);P[:n,:n]=np.linalg.inv(A)
    g0=P@G@P.T;g1=np.einsum('ai,ijk,bj,kc->abc',P,G1,P,A,optimize=True)
    g2=np.zeros((n+3,n+3,n));g2[n:,n:]=np.einsum('abij,ik,jk->abk',R2,A,A,optimize=True)
    cubic=np.einsum('ijk,ia,jb,kc->abc',C,A,A,A,optimize=True)
    result=modal_alpha_constants(w,cubic,g1[:n,:n],g0,g1,g2)
    result['coordinate_to_modal']=A
    return result


def modal_alpha_constants(frequencies_cm, cubic_cm, vibrational_metric_first_cm,
                         full_metric_zero_cm, full_metric_first_cm,
                         full_metric_second_diagonal_cm, *,
                         frequency_tolerance_cm=.01, coriolis_resonance_threshold_cm=.01):
    w=np.asarray(frequencies_cm,float);f3=np.asarray(cubic_cm,float)
    vg1=np.asarray(vibrational_metric_first_cm,float);g0=np.asarray(full_metric_zero_cm,float)
    g1=np.asarray(full_metric_first_cm,float);g2=np.asarray(full_metric_second_diagonal_cm,float)
    n=len(w);d=n+3
    if (w.ndim!=1 or np.any(abs(w)<=frequency_tolerance_cm) or f3.shape!=(n,n,n)
            or vg1.shape!=(n,n,n) or g0.shape!=(d,d) or g1.shape!=(d,d,n) or g2.shape!=(d,d,n)
            or not all(np.all(np.isfinite(a)) for a in (w,f3,vg1,g0,g1,g2))
            or frequency_tolerance_cm<=0 or coriolis_resonance_threshold_cm<frequency_tolerance_cm):
        raise ValueError('Complete finite modal field and signed nonzero frequencies required')
    if (not np.allclose(g0,g0.T,atol=1e-7) or not np.allclose(g1,g1.swapaxes(0,1),atol=1e-7)
            or not np.allclose(g2,g2.swapaxes(0,1),atol=1e-7)
            or np.max(abs(g0[:n,n:]))>1e-6
            or not np.allclose(g0[:n,:n],np.diag(abs(w)),rtol=1e-8,atol=1e-3)):
        raise ValueError('Modal tensors must use the same reference Eckart metric and real absolute-frequency normalization')
    values,axes=np.linalg.eigh(g0[n:,n:]);order=np.argsort(values)[::-1];axes=axes[:,order];values=values[order]
    if np.any(values<=0):raise ValueError('Positive rotational metric required')
    # Choose deterministic axes within any degenerate rotational block.
    start=0
    while start<3:
        stop=start+1
        while stop<3 and abs(values[stop]-values[start])<1e-7:stop+=1
        subspace=axes[:,start:stop];chosen=[]
        for anchor in np.eye(3):
            vector=subspace@(subspace.T@anchor)
            for previous in chosen:vector-=previous*np.dot(previous,vector)
            norm=np.linalg.norm(vector)
            if norm>1e-8:chosen.append(vector/norm)
            if len(chosen)==stop-start:break
        axes[:,start:stop]=np.column_stack(chosen);start=stop
    for i in range(3):
        if axes[np.argmax(abs(axes[:,i])),i]<0:axes[:,i]*=-1
    R=np.eye(d);R[n:,n:]=axes
    g0=R.T@g0@R;g1=np.einsum('ia,ijk,jb->abk',R,g1,R);g2=np.einsum('ia,ijk,jb->abk',R,g2,R)
    phase=np.where(w<0,np.exp(-.25j*np.pi),1+0j);omega=np.where(w<0,1j*abs(w),w+0j)
    tensor=np.zeros((n,3,3),complex);components=np.zeros((n,3,3,3),complex)
    for i,wi in enumerate(omega):
        for a in range(3):
            for b in range(3):
                r,s=n+a,n+b;curvature=g2[r,s,i]*phase[i]**2
                tensor[i,a,b]=-curvature/4;components[i,a,b,0]=curvature/8
                for j,wj in enumerate(omega):
                    rho=(f3[j,i,i]*phase[j]*phase[i]**2+vg1[i,i,j]*phase[j]/phase[i]**2)
                    rotational=g1[r,s,j]*phase[j]
                    tensor[i,a,b]+=rho*rotational/(4*wj)
                    components[i,a,b,2]-=rho*rotational/(8*wj)
                    ar=g1[i,r,j]*phase[j]/phase[i];br=g1[j,r,i]*phase[i]/phase[j]
                    az=g1[i,s,j]*phase[j]/phase[i];bz=g1[j,s,i]*phase[i]/phase[j]
                    components[i,a,b,1]-=(ar+br)*(az+bz)/(8*(wi+wj))
                    if abs(wi-wj)>coriolis_resonance_threshold_cm:
                        tensor[i,a,b]-=((ar*az+br*bz)*wj-(ar*bz+br*az)*wi)/(2*(wi*wi-wj*wj))
    for i in range(n):
        group=np.flatnonzero(abs(omega-omega[i])<=frequency_tolerance_cm)
        if len(group)>1:tensor[group]=tensor[group].mean(axis=0)
    alpha=np.diagonal(tensor,axis1=1,axis2=2)
    cm_to_mhz=29979.2458
    return dict(frequencies_cm=w.copy(),equilibrium_B_MHz=np.diag(g0[n:,n:])*cm_to_mhz/2,
        alpha_MHz=alpha*cm_to_mhz,full_delta_MHz=-.5*alpha.sum(axis=0)*cm_to_mhz,
        alpha_tensor_MHz=tensor*cm_to_mhz,full_delta_tensor_MHz=-.5*tensor.sum(axis=0)*cm_to_mhz,
        eq50_delta_MHz=np.diag(components.sum(axis=(0,3)))*cm_to_mhz,principal_axes=axes,
        unstable_mode_indices=np.flatnonzero(w<0),field_modes_removed=False)
