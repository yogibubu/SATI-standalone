"""Positive correlated trial envelopes on an unchanged angular quadrature."""
import numpy as np


def correlate_angular_basis(points,values,momenta,original_widths,envelope_covariance,
                            periodic_center_coupling):
    """Dress a polynomial/Fourier basis and its exact momentum actions.

    The new Gaussian is centered at a*sin(theta) in the nonperiodic angles.
    Physical quadrature weights, boundaries and electronic fields are unchanged.
    A subsequent physical-measure Gram normalization is required.
    """
    q=np.asarray(points,float);u=np.asarray(values);p=np.asarray(momenta)
    widths=np.asarray(original_widths,float);C=np.asarray(envelope_covariance,float)
    a=np.asarray(periodic_center_coupling,float);d=len(widths)
    if (q.ndim!=2 or q.shape[1]!=d+1 or u.ndim!=2 or len(u)!=len(q)
            or p.shape!=(len(q),d+1,u.shape[1]) or C.shape!=(d,d) or a.shape!=(d,)
            or np.any(widths<=0) or not np.allclose(C,C.T)
            or not all(np.all(np.isfinite(x)) for x in (q,u,p,widths,C,a))
            or np.linalg.eigvalsh(C).min()<=0):
        raise ValueError('Matching finite angular basis and positive envelope covariance required')
    residual=q[:,:d]-np.sin(q[:,-1,None])*a
    transformed=np.linalg.solve(C,residual.T).T
    logarithm=-.5*np.einsum('gi,gi->g',residual,transformed)+.5*np.sum((q[:,:d]/widths)**2,axis=1)
    if abs(logarithm).max()>600:raise ValueError('Envelope ratio exceeds safe floating-point range')
    factor=np.exp(logarithm);gradient=np.empty_like(q)
    gradient[:,:d]=-transformed+q[:,:d]/widths**2
    gradient[:,-1]=np.cos(q[:,-1])*np.einsum('i,gi->g',a,transformed)
    return dict(values=u*factor[:,None],momenta=(p-1j*gradient[:,:,None]*u[:,None,:])*factor[:,None,None],
        log_envelope_ratio=logarithm,log_envelope_gradient=gradient,
        physical_domain_unchanged=True)
