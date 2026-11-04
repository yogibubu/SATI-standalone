"""Analytic N-dimensional harmonic bath response with a positive Hessian.

The force is anchored through its quadratic jet. A polynomial Cholesky field
with exponential diagonal keeps the transverse Hessian positive. All response
derivatives follow from the same functions used in the local minimization.
"""
from dataclasses import dataclass
import math
import numpy as np


def fit_weighted_feature_subspace(features,values,weights,*,rcond=1e-12):
    """Fit only the identifiable weighted feature subspace.

    The retained right singular vectors define the actual free coefficients;
    null-space coefficients are fixed to zero. The returned basis makes this
    modeling convention explicit rather than counting unsupported parameters.
    """
    A=np.asarray(features,float);y=np.asarray(values,float);w=np.asarray(weights,float)
    if (A.ndim!=2 or y.ndim!=2 or len(A)!=len(y) or w.shape!=(len(A),)
            or np.any(w<=0) or not 0<rcond<1
            or not all(np.all(np.isfinite(v)) for v in (A,y,w))):
        raise ValueError('Finite compatible features, vector values and positive weights required')
    U,s,Vt=np.linalg.svd(A*w[:,None],full_matrices=False)
    rank=int(np.sum(s>rcond*s[0])) if len(s) else 0
    if not rank:raise ValueError('No identifiable feature direction')
    identified=Vt[:rank].T
    theta=(U[:,:rank].T@(y*w[:,None]))/s[:rank,None]
    return identified@theta,dict(rank=rank,raw_feature_count=A.shape[1],
        identified_feature_basis=identified,identified_coefficients=theta,
        singular_values=s,coefficient_policy='Retained weighted SVD subspace; null-space coefficients fixed to zero')


def polynomial_basis(points, powers, scale, derivative=None):
    q=np.asarray(points,float);scale=np.asarray(scale,float);powers=np.asarray(powers,int)
    order=np.zeros(len(scale),int) if derivative is None else np.asarray(derivative,int)
    if (q.shape[-1:] != scale.shape or powers.ndim!=2 or powers.shape[1]!=len(scale)
            or np.any(scale<=0) or np.any(powers<0) or order.shape!=scale.shape
            or np.any(order<0) or not all(np.all(np.isfinite(v)) for v in (q,scale))):
        raise ValueError('Finite matching polynomial coordinates and nonnegative powers required')
    columns=[]
    for power in powers:
        if np.any(power<order):
            columns.append(np.zeros(q.shape[:-1]));continue
        factor=np.prod([math.factorial(int(p))/math.factorial(int(p-d))/s**int(d) for p,d,s in zip(power,order,scale)])
        columns.append(factor*np.prod((q/scale)**(power-order),axis=-1))
    return np.stack(columns,axis=-1) if columns else np.empty(q.shape[:-1]+(0,))


@dataclass(frozen=True)
class AnalyticHarmonicBath:
    """Local model g(q).b + b.H(q).b/2 and its exact first response jet."""
    scale: np.ndarray
    force_quadratic: np.ndarray
    force_powers: np.ndarray
    force_coefficients: np.ndarray
    reference_cholesky: np.ndarray
    cholesky_powers: np.ndarray
    cholesky_coefficients: np.ndarray
    force_linear: np.ndarray | None = None
    basis_family: str = 'polynomial'

    def __post_init__(self):
        if self.force_linear is None:
            object.__setattr__(self,'force_linear',np.zeros((len(self.force_quadratic),len(self.scale))))
        for name in self.__dataclass_fields__:
            if name=='basis_family':continue
            value=np.array(getattr(self,name),dtype=int if name.endswith('powers') else float,copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError('Finite analytic bath fields required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        d=len(self.scale);n=len(self.force_quadratic)
        if (self.scale.ndim!=1 or not 1<=d<=4 or np.any(self.scale<=0)
                or self.basis_family not in ('polynomial','periodic')
                or self.force_quadratic.shape!=(n,d,d) or n<1
                or self.force_linear.shape!=(n,d)
                or not np.allclose(self.force_quadratic,self.force_quadratic.swapaxes(-1,-2),atol=1e-10)
                or self.reference_cholesky.shape!=(n,n)
                or np.any(np.diag(self.reference_cholesky)<=0)
                or np.any(np.triu(self.reference_cholesky,1)!=0)
                or self.force_powers.ndim!=2 or self.force_powers.shape[1]!=d
                or self.cholesky_powers.ndim!=2 or self.cholesky_powers.shape[1]!=d
                or np.any(self.force_powers<0) or np.any(self.cholesky_powers<0)
                or np.any(self.force_powers.sum(axis=1)<3)
                or np.any(self.cholesky_powers.sum(axis=1)<1)
                or self.force_coefficients.shape!=(n,len(self.force_powers))
                or self.cholesky_coefficients.shape!=(n*(n+1)//2,len(self.cholesky_powers))):
            raise ValueError('Incompatible anchored force and Cholesky fields')

    def evaluate(self, points, *, second=False):
        q=np.asarray(points,float);d=len(self.scale);n=len(self.force_quadratic)
        if self.basis_family=='periodic':
            from .periodic_hermite import angular_features
            basis=angular_features
        else:basis=polynomial_basis
        P=basis(q,self.force_powers,self.scale)
        C=basis(q,self.cholesky_powers,self.scale)
        eye=np.eye(d,dtype=int)
        dP=np.stack([basis(q,self.force_powers,self.scale,e) for e in eye],axis=-1)
        dC=np.stack([basis(q,self.cholesky_powers,self.scale,e) for e in eye],axis=-1)
        if self.basis_family=='periodic':
            bp=list(eye);bc=[self.force_linear[:,i] for i in range(d)]
            for i in range(d):
                bp.append(2*eye[i]);bc.append(.5*self.force_quadratic[:,i,i])
                for j in range(i+1,d):bp.append(eye[i]+eye[j]);bc.append(self.force_quadratic[:,i,j])
            bc=np.array(bc).T
            g=basis(q,bp,np.ones(d))@bc.T
            dg=np.stack([basis(q,bp,np.ones(d),e)@bc.T for e in eye],axis=-1)
        else:
            g=q@self.force_linear.T+.5*np.einsum('...i,bij,...j->...b',q,self.force_quadratic,q)
            dg=self.force_linear+np.einsum('bij,...j->...bi',self.force_quadratic,q)
        g=g+P@self.force_coefficients.T
        dg=dg+np.einsum('...ti,bt->...bi',dP,self.force_coefficients)
        fields=C@self.cholesky_coefficients.T
        df=np.einsum('...ti,bt->...bi',dC,self.cholesky_coefficients)
        L=np.broadcast_to(self.reference_cholesky,q.shape[:-1]+(n,n)).copy()
        dL=np.zeros(q.shape[:-1]+(n,n,d))
        for k,(i,j) in enumerate(zip(*np.tril_indices(n))):
            if i==j:
                L[...,i,j]*=np.exp(fields[...,k]);dL[...,i,j,:]=L[...,i,j,None]*df[...,k,:]
            else:
                s=np.sqrt(self.reference_cholesky[i,i]*self.reference_cholesky[j,j])
                L[...,i,j]+=s*fields[...,k];dL[...,i,j,:]=s*df[...,k,:]
        H=L@L.swapaxes(-1,-2)
        term=np.einsum('...ikt,...jk->...ijt',dL,L)
        dH=term+term.swapaxes(-3,-2)
        if not all(np.all(np.isfinite(v)) for v in (g,dg,H,dH)):
            raise ValueError('Nonfinite analytic bath continuation')
        b=-np.linalg.solve(H,g[...,None])[...,0]
        db=-np.linalg.solve(H,dg+np.einsum('...ijt,...j->...it',dH,b))
        lowering=.5*np.einsum('...i,...i->...',g,b)
        result=dict(force=g,hessian=H,force_first=dg,hessian_first=dH,
            response=b,response_first=db,energy_lowering=lowering,
            energy_lowering_first=np.einsum('...it,...i->...t',dg,b)+.5*np.einsum('...i,...ijt,...j->...t',b,dH,b))
        if second:
            d2P=np.stack([np.stack([basis(q,self.force_powers,self.scale,e+f) for f in eye],axis=-1) for e in eye],axis=-2)
            d2C=np.stack([np.stack([basis(q,self.cholesky_powers,self.scale,e+f) for f in eye],axis=-1) for e in eye],axis=-2)
            if self.basis_family=='periodic':
                d2g=np.stack([np.stack([basis(q,bp,np.ones(d),e+f)@bc.T for f in eye],axis=-1) for e in eye],axis=-2)
            else:d2g=self.force_quadratic
            d2g=d2g+np.einsum('...tij,bt->...bij',d2P,self.force_coefficients)
            d2f=np.einsum('...tij,bt->...bij',d2C,self.cholesky_coefficients)
            d2L=np.zeros(q.shape[:-1]+(n,n,d,d))
            for k,(i,j) in enumerate(zip(*np.tril_indices(n))):
                if i==j:d2L[...,i,j,:,:]=L[...,i,j,None,None]*(d2f[...,k,:,:]+df[...,k,:,None]*df[...,k,None,:])
                else:d2L[...,i,j,:,:]=np.sqrt(self.reference_cholesky[i,i]*self.reference_cholesky[j,j])*d2f[...,k,:,:]
            term=np.einsum('...ikst,...jk->...ijst',d2L,L)
            cross=np.einsum('...iks,...jkt->...ijst',dL,dL)
            d2H=term+term.swapaxes(-4,-3)+cross+cross.swapaxes(-2,-1)
            rhs=d2g+np.einsum('...ijst,...j->...ist',d2H,b)
            mixed=np.einsum('...ijs,...jt->...ist',dH,db)
            rhs+=mixed+mixed.swapaxes(-2,-1)
            d2b=-np.linalg.solve(H,rhs.reshape(q.shape[:-1]+(n,d*d))).reshape(q.shape[:-1]+(n,d,d))
            result.update(force_second=d2g,hessian_second=d2H,response_second=d2b)
        return result

    def to_dict(self):
        version='v2' if self.basis_family!='polynomial' or np.any(self.force_linear) else 'v1'
        return dict(schema='matrix.dvr.analytic_harmonic_bath.'+version,
                    **{name:getattr(self,name) if name=='basis_family' else getattr(self,name).tolist() for name in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls,value):
        if value.get('schema') not in ('matrix.dvr.analytic_harmonic_bath.v1','matrix.dvr.analytic_harmonic_bath.v2'):
            raise ValueError('Unsupported analytic harmonic bath schema')
        return cls(**{name:value.get(name) if name=='force_linear' else value.get(name,'polynomial') if name=='basis_family' else value[name] for name in cls.__dataclass_fields__})
