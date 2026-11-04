"""Potential and kinetic parent-force jets on periodic nuclear wavefunctions.

The complete field is input unchanged. This owner constructs observable-force
operators and their paired harmonic contraction, not a truncated force field.
"""
import numpy as np
from .kinetic_bath_response import _normal


def periodic_reference_force_coefficients(cubic, kinetic_first, periodic_index, exchange_signs, force_index):
    """Finite Fourier jets for dH/dq_force from all F3 and dG components.

q_theta is continued by sin(2 theta)/2 for even exchange and sin(theta)
for odd exchange. Its squared term has the matching even/odd periodic
continuation. Constant fast-force blocks retain their exchange-odd cos(theta)
part. Every supplied mode and cross term enters this observable construction.
"""
    c=np.asarray(cubic,float);dg=np.asarray(kinetic_first,float);sign=np.asarray(exchange_signs,int)
    n=len(sign);t=periodic_index;b=force_index
    if c.shape!=(n,n,n) or dg.shape!=(n,n,n) or not np.all(np.isin(sign,[-1,1])) or sign[t]!=1:
        raise ValueError('Complete cubic, kinetic jet and explicit coordinate exchange signs required')
    fast=np.delete(np.arange(n),t);sf=np.diag(sign[fast]);s=sign[b]
    mat=c[b][np.ix_(fast,fast)];vec=c[b,fast,t]
    ce=.5*(mat+s*sf@mat@sf);co=mat-ce
    le=.5*(vec+s*sf@vec);lo=vec-le
    d=dg[:,:,b];fulls=np.diag(sign);de=.5*(d+s*fulls@d@fulls);do=d-de
    result={k:dict(scalar=0j,linear=np.zeros(n-1,complex),second=np.zeros((n-1,n-1),complex),
        momentum_second=np.zeros((n-1,n-1),complex),mixed=np.zeros(n-1,complex),rotor=0j) for k in range(-3,4)}
    result[0]['second']=ce;result[0]['momentum_second']=de[np.ix_(fast,fast)]
    result[0]['mixed']=de[fast,t];result[0]['rotor']=de[t,t]/2
    for k in (-1,1):
        result[k]['second']=co/2;result[k]['momentum_second']=do[np.ix_(fast,fast)]/2
        result[k]['mixed']=do[fast,t]/2;result[k]['rotor']=do[t,t]/4
    result[2]['linear']=le/(4j);result[-2]['linear']=-le/(4j)
    result[1]['linear']+=lo/(2j);result[-1]['linear']-=lo/(2j)
    value=c[b,t,t]
    if s==1:
        result[0]['scalar']=value/4;result[2]['scalar']=result[-2]['scalar']=-value/8
    else:
        result[1]['scalar']=result[-1]['scalar']=value/16
        result[3]['scalar']=result[-3]['scalar']=-value/16
    return result


def harmonic_reference_mean_force(kinetic, force, cubic, kinetic_first):
    """Whole-field Gaussian mean force, including all coordinate and momenta pairs."""
    _,q,p=_normal(kinetic,force);cq=q@q.T/2;cp=p@p.T/2
    value=.5*np.einsum('ijk,jk->i',cubic,cq)+.5*np.einsum('jki,jk->i',kinetic_first,cp)
    return dict(mean_force=value,coordinate_covariance=cq,momentum_covariance=cp)


__all__=['periodic_reference_force_coefficients','harmonic_reference_mean_force']
