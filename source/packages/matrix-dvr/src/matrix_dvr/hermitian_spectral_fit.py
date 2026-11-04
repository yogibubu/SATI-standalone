"""Fit eigenvalues and lowest-branch derivatives of real symmetric matrix jets.

The caller owns the analytic coordinate basis and independent controls. Matrix
elements are linear in parameters; eigenvector response provides the analytic
Jacobian of the lowest-eigenvalue gradient. This is spectral interpolation,
not an electronic diabatization or a model for derivative couplings.
"""
import numpy as np
from scipy.optimize import least_squares


def spectral_jets_and_jacobian(design,parameters):
    B=np.asarray(design,float);theta=np.asarray(parameters,float)
    if B.ndim!=5 or B.shape[2]!=len(theta) or B.shape[3]!=B.shape[4] or B.shape[1]<1:
        raise ValueError('Sample, jet, parameter, state, state design required')
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(theta)) or not np.allclose(B,np.swapaxes(B,-1,-2),rtol=1e-12,atol=1e-12):
        raise ValueError('Finite real symmetric matrix designs required')
    H=np.einsum('njpab,p->njab',B,theta,optimize=True)
    levels,vectors=np.linalg.eigh(H[:,0])
    if np.any(np.diff(levels,axis=1)<=1e-12):raise ValueError('Distinct training eigenvalues required for spectral derivatives')
    root_jac=np.einsum('nai,npab,nbi->nip',vectors,B[:,0],vectors,optimize=True)
    v=vectors[:,:,0];gradient=np.einsum('na,njab,nb->nj',v,H[:,1:],v,optimize=True)
    direct=np.einsum('na,njpab,nb->njp',v,B[:,1:],v,optimize=True)
    coupling=np.einsum('nak,npab,nb->nkp',vectors,B[:,0],v,optimize=True)
    derivative_coupling=np.einsum('nak,njab,nb->njk',vectors,H[:,1:],v,optimize=True)
    inverse_gap=1/(levels[:,0,None]-levels[:,1:])
    response=2*np.einsum('nkp,njk,nk->njp',coupling[:,1:],derivative_coupling[:,:,1:],inverse_gap,optimize=True)
    return levels,gradient,root_jac,direct+response


def fit_hermitian_spectral_jets(design,energies,lower_gradients,*,starts,
        derivative_weight=.3,anchor_weight=100.,ridge_weights=None,
        relative_ridge=1e-5,max_nfev=1500,root_weights=None,reference_response=None):
    """Finite deterministic multistart least squares, excluding external data."""
    B=np.asarray(design,float);e=np.asarray(energies,float);g=np.asarray(lower_gradients,float)
    if B.ndim!=5 or e.shape!=(len(B),B.shape[-1]) or g.shape!=(len(B),B.shape[1]-1) or not all(np.all(np.isfinite(v)) for v in (B,e,g)):
        raise ValueError('Compatible finite energy and derivative observations required')
    ridge=np.ones(B.shape[2]) if ridge_weights is None else np.asarray(ridge_weights,float)
    if ridge.shape!=(B.shape[2],) or np.any(ridge<0) or derivative_weight<=0 or anchor_weight<1 or relative_ridge<0:
        raise ValueError('Nonnegative regularization and positive observation weights required')
    root_weights=np.ones(e.shape[1]) if root_weights is None else np.asarray(root_weights,float)
    if root_weights.shape!=(e.shape[1],) or np.any(root_weights<=0) or not np.all(np.isfinite(root_weights)):
        raise ValueError('Positive finite root weights required')
    response=None if reference_response is None else np.asarray(reference_response,float)
    if response is not None and (response.shape!=(len(B),1+g.shape[1],1+g.shape[1]) or not np.all(np.isfinite(response)) or not np.allclose(response[0],np.eye(1+g.shape[1]),atol=1e-12)):
        raise ValueError('Reference response must retain the identity reference jet')
    weights=np.ones((len(B),e.shape[1]+g.shape[1]));weights[:,:e.shape[1]]*=root_weights;weights[:,e.shape[1]:]*=derivative_weight;weights[0]*=anchor_weight
    target=np.column_stack([e,g]);cache={}
    def compute(t):
        if 't' not in cache or not np.array_equal(t,cache['t']):
            levels,grad,J,Jg=spectral_jets_and_jacobian(B,t)
            if response is not None:
                delta=np.r_[e[0,0]-levels[0,0],g[0]-grad[0]]
                correction=response@delta
                correction_jac=-np.einsum('nij,jp->nip',response,np.vstack([J[0,0],Jg[0]]))
                levels=levels+correction[:,0,None];grad=grad+correction[:,1:]
                J=J+correction_jac[:,0,None,:];Jg=Jg+correction_jac[:,1:,:]
            prediction=np.column_stack([levels,grad]);jac=np.concatenate([J,Jg],axis=1)
            cache.update(t=t.copy(),residual=np.r_[((prediction-target)*weights).ravel(),relative_ridge*ridge*t],jacobian=np.vstack([(jac*weights[:,:,None]).reshape(-1,len(t)),relative_ridge*np.diag(ridge)]),prediction=prediction,data_jacobian=jac)
        return cache
    records=[];best=None
    for start in starts:
        result=least_squares(lambda t:compute(t)['residual'],np.asarray(start,float),jac=lambda t:compute(t)['jacobian'],max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-8,x_scale='jac')
        cost=float(result.fun@result.fun);records.append(dict(cost=cost,nfev=result.nfev,success=bool(result.success),message=result.message,parameters=result.x.tolist()))
        if best is None or cost<best[0]:best=(cost,result.x.copy())
    if best is None:raise ValueError('At least one initial matrix parameter vector required')
    record=compute(best[1]);J=record['data_jacobian'].reshape(-1,len(best[1]));column=np.maximum(np.linalg.norm(J,axis=0),1e-30);singular=np.linalg.svd(J/column,compute_uv=False);rank=int(sum(singular>singular[0]*1e-9));error=record['prediction']-target
    return best[1],dict(training_maximum_energy_error=float(abs(error[:,:e.shape[1]]).max()),training_maximum_lower_gradient_error=float(abs(error[:,e.shape[1]:]).max()),data_jacobian_rank=rank,parameters=len(best[1]),starts=records,relative_ridge=relative_ridge,weighted_cost=best[0],prediction=record['prediction'].tolist())
