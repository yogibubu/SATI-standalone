"""Bounded variable projection for affine-in-coefficient least squares.

The supplied linearization returns the residual, full Jacobian and mixed
response d(J_linear)/d(theta), all in the SAME whitened augmented space.
No normal equations are formed to solve for the linear coefficients.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


def _inverse(matrix,rcond):
    norms=np.linalg.norm(matrix,axis=0)
    scales=1./np.where(norms>0,norms,1.)
    u,s,vh=np.linalg.svd(matrix*scales,full_matrices=False)
    keep=s>(s[0]*rcond if len(s) else 0.)
    if np.count_nonzero(keep)!=matrix.shape[1]:
        raise ValueError('rank-deficient linear variable-projection block')
    return (scales[:,None]*(vh.T/s))@u.T,scales


class VariableProjection:
    """Eliminate bounded linear variables; cache one accepted/trial profile.

    Derivatives are exact inside a constant active set and constant-rank
    region, including the nonzero-residual term often omitted in approximate
    variable-projection Jacobians. Bound transitions are piecewise smooth.
    """
    def __init__(self,linearize,initial,linear_indices,lower,upper,*,rcond=1.e-10):
        self.linearize=linearize;self.initial=np.asarray(initial,float)
        self.linear=np.asarray(linear_indices,int)
        self.lower=np.asarray(lower,float);self.upper=np.asarray(upper,float)
        self.nonlinear=np.array([i for i in range(len(self.initial)) if i not in set(self.linear)],int)
        if (self.initial.ndim!=1 or self.linear.ndim!=1 or not len(self.linear)
                or len(set(self.linear))!=len(self.linear) or np.any(self.linear<0) or np.any(self.linear>=len(self.initial))
                or self.lower.shape!=self.initial.shape or self.upper.shape!=self.initial.shape
                or not np.all(np.isfinite(self.initial)) or np.any(np.isnan(self.lower)) or np.any(np.isnan(self.upper))
                or np.any(self.lower>=self.upper) or np.any(self.initial<self.lower) or np.any(self.initial>self.upper)
                or not 0<rcond<1):raise ValueError('invalid variable-projection registry, bounds or rank threshold')
        self.rcond=rcond;self.cache={};self.evaluations=0;self.bounded_solves=0

    def evaluate(self,theta):
        theta=np.asarray(theta,float)
        if theta.shape!=(len(self.nonlinear),) or not np.all(np.isfinite(theta)):
            raise ValueError('invalid nonlinear variable-projection coordinates')
        if np.any(theta<self.lower[self.nonlinear]) or np.any(theta>self.upper[self.nonlinear]):
            raise ValueError('nonlinear variable-projection point outside bounds')
        key=theta.tobytes()
        if key in self.cache:return self.cache[key]
        full=self.initial.copy();full[self.nonlinear]=theta
        residual,jac,mixed=self.linearize(full,self.linear,self.nonlinear)
        residual=np.asarray(residual,float);jac=np.asarray(jac,float);mixed=np.asarray(mixed,float)
        if (residual.ndim!=1 or jac.shape!=(len(residual),len(full))
                or mixed.shape!=(len(residual),len(self.linear),len(self.nonlinear))
                or not all(np.all(np.isfinite(a)) for a in (residual,jac,mixed))):
            raise ValueError('inconsistent or nonfinite augmented variable-projection linearization')
        a=jac[:,self.linear];pinv,scales=_inverse(a,self.rcond)
        offset=residual-a@full[self.linear]
        coefficients=-pinv@offset;lo=self.lower[self.linear];hi=self.upper[self.linear]
        active=np.zeros(len(self.linear),int)
        if np.any(coefficients<lo) or np.any(coefficients>hi):
            solved=lsq_linear(a*scales,-offset,bounds=(lo/scales,hi/scales),method='bvls',tol=1.e-13,max_iter=1000)
            if not solved.success:raise RuntimeError('bounded variable-projection coefficient solve failed')
            coefficients=np.clip(solved.x*scales,lo,hi);active=solved.active_mask;self.bounded_solves+=1
        else:
            active[coefficients==lo]=-1;active[coefficients==hi]=1
        delta=coefficients-full[self.linear];full[self.linear]=coefficients
        r=a@coefficients+offset
        j=jac.copy();j[:,self.nonlinear]+=np.einsum('mln,l->mn',mixed,delta)
        b=j[:,self.nonlinear];free=np.flatnonzero(active==0)
        dc=np.zeros((len(self.linear),len(self.nonlinear)))
        if len(free):
            inverse,_=_inverse(a[:,free],self.rcond)
            correction=np.einsum('mln,m->ln',mixed[:,free,:],r)
            dc[free]=-inverse@b-inverse@(inverse.T@correction)
        reduced=b+a@dc
        self.evaluations+=1
        result=dict(values=full,residual=r,joint_jacobian=j,jacobian=reduced,
            coefficient_jacobian=dc,active_linear_bounds=active,
            linear_rank=len(self.linear),linear_gradient=a.T@r)
        self.cache={key:result};return result
