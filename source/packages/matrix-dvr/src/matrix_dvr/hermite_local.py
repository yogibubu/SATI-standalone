"""Estimate missing vector second jets from a finite measured Hermite source.

These are regression estimates, not new derivative observations. Coordinates
must belong to one declared local angular branch; physical periodic images
or parity partners are supplied explicitly by the caller.
"""
import numpy as np


def estimate_local_second_jets(points,values,first,*,lengthscales,relative_ridge=1e-6):
    q=np.asarray(points,float);v=np.asarray(values,float);g=np.asarray(first,float);ell=np.asarray(lengthscales,float)
    if (q.ndim!=2 or v.ndim!=2 or v.shape[0]!=len(q) or g.shape!=v.shape+(q.shape[1],)
            or ell.shape!=(q.shape[1],) or np.any(ell<=0) or not 0<relative_ridge<1
            or not all(np.all(np.isfinite(x)) for x in (q,v,g,ell))):
        raise ValueError('Finite matching vector Hermite observations and positive scales required')
    n,d=q.shape;pairs=[(i,j) for i in range(d) for j in range(i,d)];result=[];errors=[]
    for center in range(n):
        delta=(q-q[center])/ell;weight=np.exp(-.25*np.sum(delta*delta,axis=1))
        design=np.column_stack([delta[:,i]*delta[:,j]*(.5 if i==j else 1.) for i,j in pairs])
        rhs=v-v[center]-np.einsum('bi,ni->nb',g[center]*ell,delta)
        rows=[design*weight[:,None]];targets=[rhs*weight[:,None]]
        for k in range(d):
            A=np.column_stack([delta[:,j] if i==k else delta[:,i] if j==k else np.zeros(n) for i,j in pairs])
            rows.append(A*weight[:,None]);targets.append((g[:,:,k]-g[center,:,k])*ell[k]*weight[:,None])
        A=np.vstack(rows);y=np.vstack(targets);u,s,vt=np.linalg.svd(A,full_matrices=False)
        ridge=relative_ridge*s[0]**2;coeff=vt.T@((s/(s*s+ridge))[:,None]*(u.T@y))
        H=np.zeros((v.shape[1],d,d))
        for (i,j),c in zip(pairs,coeff):H[:,i,j]=H[:,j,i]=c/(ell[i]*ell[j])
        result.append(H);errors.append(float(np.sqrt(np.mean((A@coeff-y)**2))))
    return np.array(result),dict(method='Local weighted quadratic regression of measured values and first derivatives',
        second_jets_are_estimates=True,maximum_weighted_residual=float(max(errors)),relative_ridge=relative_ridge)
