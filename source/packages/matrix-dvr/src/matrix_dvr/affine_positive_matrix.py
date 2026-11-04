"""Positive periodic matrix fields with affine reflection and exact origin jets.

Fits matrix logarithms in diagonal reference units. Origin values are exact;
first jets may be exact or explicitly weighted observations. Reflection transforms matrix
indices with declared coordinate parities. Electronic fields are never queried.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from .analytic_fit_policy import validate_sample_plan,POLICY_ID


def symmetric_log_and_first(matrices,first=None):
    x=np.asarray(matrices,float);ev,U=np.linalg.eigh(x)
    if np.any(ev<=0):raise ValueError('Strictly positive observed matrices required')
    logs=np.einsum('...ik,...k,...jk->...ij',U,np.log(ev),U)
    if first is None:return logs
    # Divided differences of log, with the continuous equal-eigenvalue limit.
    lam=ev[..., :,None];mu=ev[...,None,:];gap=lam-mu
    ratio=np.divide(np.log(lam)-np.log(mu),gap,out=np.zeros_like(gap),where=abs(gap)>1e-10*np.maximum(lam,mu))
    ratio=np.where(abs(gap)>1e-10*np.maximum(lam,mu),ratio,2/(lam+mu))
    local=np.einsum('...ai,...abk,...bj->...ijk',U,np.asarray(first),U)
    derivative=np.einsum('...ai,...ijk,...bj->...abk',U,ratio[...,None]*local,U)
    return logs,derivative


@dataclass(frozen=True)
class AffinePositiveMatrixField:
    powers: np.ndarray
    coefficients: np.ndarray
    scales: np.ndarray
    offset: np.ndarray
    units: np.ndarray
    parities: np.ndarray
    extent: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value=np.array(getattr(self,name),int if name in ('powers','parities') else float,copy=True)
            if not np.all(np.isfinite(value)):raise ValueError('Finite matrix field parameters required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        d=len(self.scales);m=len(self.units)
        if self.powers.ndim!=2 or self.powers.shape[1]!=d or self.coefficients.shape!=(len(self.powers),m*(m+1)//2) or self.parities.shape!=(m,) or self.offset.shape!=(d,) or self.extent.shape!=(d,) or np.any(self.powers<0) or np.any(self.scales<=0) or np.any(self.units<=0) or np.any(abs(self.parities)!=1):raise ValueError('Compatible positive matrix field dimensions and parities required')

    def evaluate(self,points):
        q=np.asarray(points,float);m=len(self.units);tri=np.tril_indices(m)
        signs=self.parities[tri[0]]*self.parities[tri[1]];target=np.empty((len(q),len(tri[0])))
        for sign in (-1,1):
            ids=np.flatnonzero(signs==sign)
            if not len(ids):continue
            # Reuse the stable periodic polynomial feature owner.
            from .periodic_hermite import angular_features
            A=.5*(angular_features(q,self.powers,self.scales)+sign*angular_features(self.offset-q,self.powers,self.scales))
            target[:,ids]=A@self.coefficients[:,ids]
        logs=np.zeros((len(q),m,m));logs[:,tri[0],tri[1]]=target;logs[:,tri[1],tri[0]]=target
        ev,U=np.linalg.eigh(logs)
        if np.max(ev)>600:raise ValueError('Matrix field extrapolation overflows; no clipping permitted')
        result=np.einsum('pik,pk,pjk->pij',U,np.exp(ev),U)*self.units[None,:,None]*self.units[None,None,:]
        if not np.all(np.isfinite(result)):raise ValueError('Nonfinite positive matrix field')
        return result

    def to_dict(self):
        return dict(schema='matrix.dvr.affine_positive_matrix.v1',protocol_id=POLICY_ID,
            **{k:np.asarray(getattr(self,k)).tolist() for k in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.dvr.affine_positive_matrix.v1':raise ValueError('Unknown positive matrix field schema')
        return cls(**{k:np.asarray(data[k],int if k in ('powers','parities') else float) for k in cls.__dataclass_fields__})


def fit_affine_positive_matrix(points,matrices,reference,reference_first,*,powers,scales,offset,parities,sample_budget,ridge=1e-6,rcond=1e-10,reference_first_weight=None):
    from .periodic_hermite import angular_features
    q=np.asarray(points,float);obs=np.asarray(matrices,float);ref=np.asarray(reference,float);d=q.shape[1];m=len(ref)
    validate_sample_plan(sample_count=len(q),sample_budget=sample_budget)
    if obs.shape!=(len(q),m,m) or np.asarray(reference_first).shape!=(m,m,d):raise ValueError('Compatible matrix value and origin first jets required')
    units=np.sqrt(np.diag(ref));den=units[:,None]*units[None,:]
    targets=symmetric_log_and_first(obs/den)
    anchor,first=symmetric_log_and_first(ref/den,np.asarray(reference_first)/den[:,:,None])
    tri=np.tril_indices(m);signs=np.asarray(parities)[tri[0]]*np.asarray(parities)[tri[1]]
    targets=targets[:,tri[0],tri[1]];anchor=np.vstack([anchor[tri],first[tri].T])
    powers=np.asarray(powers,int);coeff=np.zeros((len(powers),len(tri[0])));eye=np.eye(d,dtype=int)
    audits=[]
    def features(x,order,sign):
        return .5*(angular_features(x,powers,scales,order)+sign*(-1)**int(sum(order))*angular_features(np.asarray(offset)-x,powers,scales,order))
    for sign in (-1,1):
        ids=np.flatnonzero(signs==sign)
        if not len(ids):continue
        C=np.array([features(np.zeros((1,d)),order,sign)[0] for order in [np.zeros(d,int),*eye]])
        hard=C if reference_first_weight is None else C[:1]
        hard_target=anchor[:,ids] if reference_first_weight is None else anchor[:1,ids]
        U,s,V=np.linalg.svd(hard,full_matrices=True);rank=int(sum(s>rcond*s[0]))
        particular=V[:rank].T@((U[:,:rank].T@hard_target)/s[:rank,None]);null=V[rank:].T
        A=features(q,np.zeros(d,int),sign);observations=targets[:,ids]
        if reference_first_weight is not None:
            weight=float(reference_first_weight)
            if weight<0 or not np.isfinite(weight):raise ValueError('Finite nonnegative reference-first weight required')
            A=np.vstack([A,weight*C[1:]]);observations=np.vstack([observations,weight*anchor[1:,ids]])
        B=A@null;rhs=observations-A@particular
        u,t,v=np.linalg.svd(B,full_matrices=False);keep=t>rcond*t[0]
        correction=v[keep].T@((t[keep]/(t[keep]**2+ridge*t[0]**2))[:,None]*(u[:,keep].T@rhs))
        coeff[:,ids]=particular+null@correction
        error=float(abs(hard@coeff[:,ids]-hard_target).max())
        if error>1e-6:raise ValueError('Periodic field cannot preserve declared origin jets')
        audits.append(dict(parity=sign,anchor_error=error,anchor_rank=rank,fit_rank=int(sum(keep)),reference_first_max_residual=float(abs(C[1:]@coeff[:,ids]-anchor[1:,ids]).max())))
    reflected=np.angle(np.exp(1j*(offset-q)));extent=np.maximum(np.max(abs(q),axis=0),np.max(abs(reflected),axis=0))
    model=AffinePositiveMatrixField(powers,coeff,np.asarray(scales),np.asarray(offset),units,np.asarray(parities),extent)
    return model,dict(anchors=audits,features=len(powers),ridge=ridge,training_relative_matrix_RMS=float(np.sqrt(np.mean((np.linalg.norm(model.evaluate(q)-obs,axis=(1,2))/np.linalg.norm(obs,axis=(1,2)))**2))))
