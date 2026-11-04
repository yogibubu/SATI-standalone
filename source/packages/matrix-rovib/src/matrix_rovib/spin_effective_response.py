"""Analytic parameter response of canonical coupled-spin reduction.

Sylvester differentiation avoids derivatives of arbitrary eigenvectors inside
degenerate PHP/QHQ blocks. Model projectors and the Galerkin frame are fixed
during differentiation; changing the discrete retained space is a new model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class SpinReductionGapError(ValueError):
    """External separation cannot support a differentiable elimination."""


@dataclass
class _Jet:
    v: np.ndarray
    d: np.ndarray
    def __add__(self,other):return _Jet(self.v+other.v,self.d+other.d)
    def __neg__(self):return _Jet(-self.v,-self.d)
    def __sub__(self,other):return self+-other
    def __matmul__(self,other):return _Jet(self.v@other.v,self.d@other.v+self.v@other.d)
    def scale(self,c):return _Jet(c*self.v,c*self.d)
    def dagger(self):return _Jet(self.v.conj().T,self.d.conj().swapaxes(-1,-2))


@dataclass(frozen=True)
class EffectiveSpinResponse:
    hamiltonian_MHz: np.ndarray
    hamiltonian_derivatives: np.ndarray
    embedding: np.ndarray
    embedding_derivatives: np.ndarray
    minimum_external_gap_MHz: float | None
    order: int


def canonical_spin_response(hamiltonian,derivatives,*,frame=None,retained_dimension=None,
                            order=4,memory_budget_bytes=512*1024**2):
    """Value and exact analytic tangent of the order-m coupling expansion.

    Derivatives have shape (nparameter,n,n), in ONE fixed physical basis and
    parameter order, including any physical producer response. frame is a
    complete unitary [P,Q]. Eigenvalues within P or Q may be degenerate.
    Only external separation is inverted; an unseparated P/Q pair raises.
    The isometric embedding resums normalization of the truncated chi.
    """
    h=np.asarray(hamiltonian,complex);dh=np.asarray(derivatives,complex)
    if (h.ndim!=2 or h.shape[0]!=h.shape[1] or not len(h) or dh.ndim!=3
        or dh.shape[1:]!=h.shape or not np.all(np.isfinite(h)) or not np.all(np.isfinite(dh))
        or not np.allclose(h,h.conj().T,rtol=1e-12,atol=1e-12)
        or not np.allclose(dh,dh.conj().swapaxes(-1,-2),rtol=1e-12,atol=1e-12)
        or isinstance(order,bool) or not isinstance(order,(int,np.integer)) or order<0):
        raise ValueError('finite Hermitian Hamiltonian/tangents and nonnegative integer order required')
    n=len(h);np_=len(dh);p=n if retained_dimension is None else retained_dimension
    if isinstance(p,bool) or not isinstance(p,(int,np.integer)) or not 1<=p<=n:
        raise ValueError('valid retained dimension required')
    if 16*n*n*(np_+1)*(20+8*order)>memory_budget_bytes:
        raise MemoryError('canonical spin response exceeds derivative workspace budget')
    frame=np.eye(n,dtype=complex) if frame is None else np.asarray(frame,complex)
    if frame.shape!=(n,n) or not np.all(np.isfinite(frame)) or not np.allclose(frame.conj().T@frame,np.eye(n),atol=1e-12,rtol=1e-12):
        raise ValueError('a complete unitary frozen reference frame is required')
    hh=frame.conj().T@h@frame;dd=frame.conj().T@dh@frame
    if p==n:
        return EffectiveSpinResponse(hh,dd,frame.copy(),np.zeros((np_,n,n),complex),None,int(order))
    origin=float(np.trace(hh).real/n);hh=hh-origin*np.eye(n)
    def zero(rows,cols):return _Jet(np.zeros((rows,cols),complex),np.zeros((np_,rows,cols),complex))
    def constant(value):return _Jet(value,np.zeros((np_,)+value.shape,complex))
    a=_Jet(hh[p:,p:],dd[:,p:,p:]);b=_Jet(hh[:p,:p],dd[:,:p,:p]);c=_Jet(hh[p:,:p],dd[:,p:,:p])
    ea,ua=np.linalg.eigh(a.v);eb,ub=np.linalg.eigh(b.v)
    gap=ea[:,None]-eb[None,:]
    minimum=float(np.min(abs(gap)))
    if minimum<=128*np.finfo(float).eps*max(1.,np.max(abs(hh))):
        raise SpinReductionGapError('P/Q separation exhausted; enlarge the model space or use full solver')
    def solve(rhs):
        value=ua@((ua.conj().T@rhs.v@ub)/gap)@ub.conj().T
        tangent_rhs=rhs.d-a.d@value+value@b.d
        tangent=ua@((ua.conj().T@tangent_rhs@ub)/gap)@ub.conj().T
        return _Jet(value,tangent)
    q=n-p
    chi=[zero(q,p) for _ in range(order+1)]
    if order:chi[1]=solve(-c)
    for k in range(2,order+1):
        rhs=zero(q,p)
        for i in range(1,k-1):rhs=rhs+chi[i]@c.dagger()@chi[k-1-i]
        chi[k]=solve(rhs)
    def add(x,y):return [a+b for a,b in zip(x,y,strict=True)]
    def mul(x,y):
        out=[zero(x[0].v.shape[0],y[0].v.shape[1]) for _ in range(order+1)]
        for k in range(order+1):
            for i in range(k+1):out[k]=out[k]+x[i]@y[k-i]
        return out
    def series(jet):return [jet]+[zero(*jet.v.shape) for _ in range(order)]
    dagger=[x.dagger() for x in chi]
    metric=mul(dagger,chi);identity=series(constant(np.eye(p,dtype=complex)))
    inverse=identity;power=identity;coefficient=1.
    for k in range(1,order//2+1):
        power=mul(power,metric);coefficient*=(-.5-(k-1))/k
        inverse=add(inverse,[x.scale(coefficient) for x in power])
    numerator=add(series(b),mul(mul(dagger,series(a)),chi))
    if order:
        pq=[zero(p,q),c.dagger()]+[zero(p,q) for _ in range(order-1)]
        qp=[zero(q,p),c]+[zero(q,p) for _ in range(order-1)]
        numerator=add(numerator,add(mul(pq,chi),mul(dagger,qp)))
    effective=mul(mul(inverse,numerator),inverse)
    total=zero(p,p)
    for entry in effective:total=total+entry
    total.v+=origin*np.eye(p)
    x=zero(q,p)
    for entry in chi:x=x+entry
    m=constant(np.eye(p,dtype=complex))+x.dagger()@x
    e,u=np.linalg.eigh(m.v);root=np.sqrt(e)
    r=(u/root)@u.conj().T
    divided=-1/(root[:,None]*root[None,:]*(root[:,None]+root[None,:]))
    dr=u@((u.conj().T@m.d@u)*divided)@u.conj().T
    embedding=_Jet(np.concatenate((np.eye(p),x.v)),
        np.concatenate((np.zeros((np_,p,p),complex),x.d),axis=1))@_Jet(r,dr)
    w=frame@embedding.v;dw=frame@embedding.d
    if not all(np.all(np.isfinite(v)) for v in (total.v,total.d,w,dw)):
        raise FloatingPointError('nonfinite canonical response; model is not qualified')
    return EffectiveSpinResponse(.5*(total.v+total.v.conj().T),
        .5*(total.d+total.d.conj().swapaxes(-1,-2)),w,dw,minimum,int(order))


def native_spin_effective_response(model,derivatives,*,order=None,memory_budget_bytes=512*1024**2):
    """Differentiate at fixed retained projector, including denominators and W.

    Requalify a discrete model-space change outside the inner fit. A model
    whose near-degeneracy closure promoted external states must be rebuilt
    with that projector as its block reference before using a finite-order
    response; otherwise different perturbation gradings would be conflated.
    """
    p=model.reduction.active_states;q=model.reduction.external_states
    if q and len(p)!=len(model.requested_basis_indices):
        raise SpinReductionGapError('promoted model space requires a new block reference or full solver')
    frame=model.reference_vectors[:,list(p)+list(q)]
    return canonical_spin_response(model.block.hamiltonian_MHz,derivatives,frame=frame,
        retained_dimension=len(p),order=model.reduction.maximum_order if order is None else order,
        memory_budget_bytes=memory_budget_bytes)


def effective_spin_dipole_response(left,right,operator,derivatives):
    """Analytic d(W_l^dagger mu W_r), including all three product-rule terms."""
    mu=np.asarray(operator,complex);dmu=np.asarray(derivatives,complex)
    if (mu.shape!=(left.embedding.shape[0],right.embedding.shape[0])
        or dmu.shape!=(len(left.embedding_derivatives),)+mu.shape
        or len(right.embedding_derivatives)!=len(dmu) or not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(dmu))):
        raise ValueError('consistent finite physical dipole/tangent spaces required')
    l,r=left.embedding,right.embedding;dl,dr=left.embedding_derivatives,right.embedding_derivatives
    return l.conj().T@mu@r, (dl.conj().swapaxes(-1,-2)@mu@r+l.conj().T@dmu@r+l.conj().T@mu@dr)
