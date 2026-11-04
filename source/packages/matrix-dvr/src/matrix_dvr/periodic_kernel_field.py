"""Finite-source periodic kernel fields with inversion and origin constraints."""
from dataclasses import dataclass
import numpy as np
from .analytic_fit_policy import validate_sample_plan,POLICY_ID


def periodic_kernel(points,centers,lengths,derivative=None):
    x=np.asarray(points,float);c=np.asarray(centers,float);ell=np.asarray(lengths,float)
    order=np.zeros(len(ell),int) if derivative is None else np.asarray(derivative,int)
    if order.shape!=ell.shape or np.any(order<0) or sum(order)>2:raise ValueError('Kernel derivatives through order two required')
    result=np.zeros((len(x),len(c)))
    axes=np.repeat(np.arange(len(ell)),order)
    for sign in (-1,1):
        delta=x[:,None,:]-sign*c[None,:,:]
        k=np.exp(-np.sum((1-np.cos(delta))/ell**2,axis=-1))
        if len(axes)==1:k*=-np.sin(delta[:,:,axes[0]])/ell[axes[0]]**2
        if len(axes)==2:
            i,j=axes;k*=np.sin(delta[:,:,i])*np.sin(delta[:,:,j])/(ell[i]**2*ell[j]**2)-(np.cos(delta[:,:,i])/ell[i]**2 if i==j else 0)
        result+=.5*k
    return result


@dataclass(frozen=True)
class PeriodicKernelField:
    centers: np.ndarray
    coefficients: np.ndarray
    lengths: np.ndarray

    def __post_init__(self):
        for name in ('centers','coefficients','lengths'):
            value=np.array(getattr(self,name),float,copy=True)
            if not np.all(np.isfinite(value)):raise ValueError('Finite periodic field parameters required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        if self.centers.ndim!=2 or self.coefficients.shape!=(len(self.centers),) or self.lengths.shape!=(self.centers.shape[1],) or np.any(self.lengths<=0):raise ValueError('Compatible field centers, weights and positive lengths required')

    def evaluate(self,points,derivative=None):
        q=np.asarray(points,float);zero=np.zeros((1,len(self.lengths)))
        K=periodic_kernel(q,self.centers,self.lengths,derivative)
        K-=periodic_kernel(q,zero,self.lengths,derivative)@periodic_kernel(zero,self.centers,self.lengths)
        return K@self.coefficients

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_kernel_field.v1',protocol_id=POLICY_ID,
            centers=self.centers.tolist(),coefficients=self.coefficients.tolist(),lengths=self.lengths.tolist(),
            origin_value=0.,symmetry='global inversion; componentwise period 2*pi')

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.dvr.periodic_kernel_field.v1':raise ValueError('Unknown periodic kernel field schema')
        return cls(*(np.asarray(data[k],float) for k in ('centers','coefficients','lengths')))


def fit_periodic_kernel_field(points,values,*,lengths,sample_budget,relative_ridge=1e-10):
    q=np.asarray(points,float);y=np.asarray(values,float);ell=np.asarray(lengths,float)
    validate_sample_plan(sample_count=len(q),sample_budget=sample_budget)
    if q.ndim!=2 or y.shape!=(len(q),) or ell.shape!=(q.shape[1],) or np.any(ell<=0) or relative_ridge<=0 or not all(np.all(np.isfinite(v)) for v in (q,y,ell)):raise ValueError('Finite compatible field samples and positive lengths/ridge required')
    zero=np.zeros((1,q.shape[1]));K=periodic_kernel(q,q,ell)-periodic_kernel(q,zero,ell)@periodic_kernel(zero,q,ell)
    scale=max(float(np.trace(K)/len(K)),1e-15);M=K+relative_ridge*scale*np.eye(len(K));inverse=np.linalg.inv(M);coeff=inverse@y
    model=PeriodicKernelField(q,coeff,ell);error=model.evaluate(q)-y;loo=coeff/np.diag(inverse)
    return model,dict(training_RMS=float(np.sqrt(np.mean(error**2))),training_max=float(abs(error).max()),
        leave_one_out_errors=loo.tolist(),leave_one_out_RMS=float(np.sqrt(np.mean(loo**2))),relative_ridge=relative_ridge,
        qualification='Interpolation only; leave-one-out residuals and nuclear-observable sensitivity must be reported separately')
