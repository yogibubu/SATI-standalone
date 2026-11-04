"""Common fixed-chart stencil for scalar, vector and tensor coefficients.

This is the stencil used by the momentum-property acquisition service. The
coefficient axes are opaque: only the appended coordinate derivative axes
are differentiated. No oscillator frequencies or stationary point are needed.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CoefficientDerivativeJet:
    value: np.ndarray
    first: np.ndarray
    second: np.ndarray
    second_observed: np.ndarray
    callback_count: int

    def __post_init__(self):
        for name in ('value','first','second','second_observed'):
            value=np.array(getattr(self,name),copy=True)
            value.flags.writeable=False
            object.__setattr__(self,name,value)


def acquire_coefficient_derivative_jet(coefficients_at_s,displacement_per_q,*,
                                       step_dimensionless=.02,
                                       single_mode_only=False):
    """Reuse the canonical one/pair stencil on any finite coefficient array.

    The provider receives displacements in one frozen chart. The final axes
    of first/second are dimensionless q derivatives. Missing mixed entries
    are NaN with an explicit mask, never an unobserved physical zero.
    """
    transform=np.asarray(displacement_per_q,dtype=float)
    if (transform.ndim!=2 or not transform.shape[0] or not transform.shape[1]
            or not np.all(np.isfinite(transform))):
        raise ValueError('finite frozen coordinate displacement map required')
    n=transform.shape[1]
    steps=np.broadcast_to(np.asarray(step_dimensionless,dtype=float),(n,))
    if not np.all(np.isfinite(steps)) or np.any(steps<=0):
        raise ValueError('positive finite coefficient stencil steps required')
    calls=0
    shape=None
    def value(q):
        nonlocal calls,shape
        result=np.asarray(coefficients_at_s(transform@q),dtype=float)
        calls+=1
        if shape is None:shape=result.shape
        if result.shape!=shape or not np.all(np.isfinite(result)):
            raise ValueError('matching finite coefficient values required')
        return result
    origin=value(np.zeros(n))
    first=np.empty(shape+(n,))
    second=np.full(shape+(n,n),np.nan)
    observed=np.zeros((n,n),dtype=bool)
    for j in range(n):
        shift=steps[j]*np.eye(n)[j]
        minus,plus=value(-shift),value(shift)
        first[...,j]=(plus-minus)/(2*steps[j])
        second[...,j,j]=(plus-2*origin+minus)/steps[j]**2
        observed[j,j]=True
    if not single_mode_only:
        for j in range(n):
            for k in range(j+1,n):
                dj=steps[j]*np.eye(n)[j];dk=steps[k]*np.eye(n)[k]
                mixed=(value(dj+dk)-value(dj-dk)-value(-dj+dk)+value(-dj-dk))/(4*steps[j]*steps[k])
                second[...,j,k]=second[...,k,j]=mixed
                observed[j,k]=observed[k,j]=True
    return CoefficientDerivativeJet(origin,first,second,observed,calls)
