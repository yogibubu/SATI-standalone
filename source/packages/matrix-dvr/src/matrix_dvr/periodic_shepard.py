"""Positive periodic partition of local, bounded Hermite approximants.

Inverse even powers of periodic distances give positive normalized weights.
At a node every other weight vanishes through the supplied second jet. Local
Taylor corrections decay analytically; constants remain bounded. This avoids
extrapolating a high-degree global polynomial, but does not certify unsampled
electronic behavior. No molecular constants or electronic callback enter.
"""
from dataclasses import dataclass
import numpy as np
from .damped_periodic import damped_angular_features


@dataclass(frozen=True)
class PeriodicShepardHermiteField:
    nodes: np.ndarray
    values: np.ndarray
    first: np.ndarray
    second: np.ndarray
    lengths: np.ndarray
    power: int = 3

    def __post_init__(self):
        for name in ('nodes','values','first','second','lengths'):
            a=np.array(getattr(self,name),float,copy=True)
            if not np.all(np.isfinite(a)):raise ValueError('Finite periodic Hermite data required')
            a.flags.writeable=False;object.__setattr__(self,name,a)
        if self.nodes.ndim!=2 or self.values.ndim!=2:raise ValueError('Node and output matrices required')
        n,d=self.nodes.shape;o=self.values.shape[1]
        if (not n or self.values.shape!=(n,o) or self.first.shape!=(n,o,d) or self.second.shape!=(n,o,d,d)
                or self.lengths.shape!=(d,) or np.any(self.lengths<=0) or self.power not in (2,3,4)
                or not np.allclose(self.second,self.second.swapaxes(-1,-2))):
            raise ValueError('Complete second jets and positive periodic distance scales required')
        distance=(2*np.sin((self.nodes[:,None]-self.nodes[None,:])/2)**2).sum(axis=-1)
        distance[np.diag_indices(n)]=np.inf
        if distance.min(initial=np.inf)<1e-20:raise ValueError('Distinct nodes modulo two pi required')

    @property
    def scales(self):
        """Coordinate scales for the common periodic-field interface."""
        return self.lengths

    @property
    def coefficients(self):
        """Output-size compatibility with the periodic quadratic bath owner."""
        return self.values

    def evaluate(self,points,derivative=None):
        q=np.asarray(points,float);d=self.nodes.shape[1]
        order=np.zeros(d,int) if derivative is None else np.asarray(derivative)
        if (q.ndim!=2 or q.shape[1]!=d or order.shape!=(d,) or np.any(order<0)
                or np.any(order!=np.rint(order)) or order.sum()>2 or not np.all(np.isfinite(q))):
            raise ValueError('Matching finite points and derivative order through two required')
        order=order.astype(int);axes=np.repeat(np.arange(d),order);N=len(q)
        delta=q[:,None,:]-self.nodes[None,:,:]
        distance=np.sum((2*np.sin(delta/2)/self.lengths)**2,axis=-1)
        hit=distance<1e-22;safe=np.where(hit,1.,distance)
        logs=-self.power*np.log(safe);logs-=logs.max(axis=1,keepdims=True)
        w=np.exp(logs);w/=w.sum(axis=1,keepdims=True)
        atnode=hit.any(axis=1);w[atnode]=hit[atnode].astype(float)
        weights={(0,)*d:w};eye=np.eye(d,dtype=int)
        if len(axes):
            dp=2*np.sin(delta)/self.lengths**2
            a=-self.power*dp/safe[:,:,None];abar=np.einsum('pn,pni->pi',w,a)
            centered=a-abar[:,None,:];dw=w[:,:,None]*centered;dw[atnode]=0
            for i in range(d):weights[tuple(eye[i])]=dw[:,:,i]
        if len(axes)==2:
            i,j=axes
            b=self.power*dp[:,:,i]*dp[:,:,j]/safe**2
            if i==j:b-=self.power*2*np.cos(delta[:,:,i])/self.lengths[i]**2/safe
            bbar=np.einsum('pn,pn->p',w,b)
            cov=np.einsum('pn,pn,pn->p',w,centered[:,:,i],centered[:,:,j])
            ddw=w*(centered[:,:,i]*centered[:,:,j]+b-bbar[:,None]-cov[:,None]);ddw[atnode]=0
            weights[tuple(order)]=ddw
        powers=[e for e in eye];coefficients=[self.first[:,:,i] for i in range(d)]
        for i in range(d):
            for j in range(i,d):
                powers.append(eye[i]+eye[j]);coefficients.append(self.second[:,:,i,j]*(.5 if i==j else 1.))
        coefficients=np.stack(coefficients,axis=1);powers=np.array(powers)
        def local(o):
            result=np.empty((N,len(self.nodes),self.values.shape[1]))
            for k in range(len(self.nodes)):
                f=damped_angular_features(delta[:,k],powers,np.ones(d),self.lengths,1,o)
                result[:,k]=f@coefficients[k]
                if not sum(o):result[:,k]+=self.values[k]
            return result
        zero=(0,)*d
        result=np.einsum('pn,pno->po',w,local(tuple(order)))
        if len(axes):
            result+=np.einsum('pn,pno->po',weights[tuple(order)],local(zero))
        if len(axes)==2:
            i,j=axes
            # The preceding line used the second weight derivative. Both
            # first-weight/first-local product-rule terms remain necessary.
            result+=np.einsum('pn,pno->po',weights[tuple(eye[i])],local(tuple(eye[j])))
            result+=np.einsum('pn,pno->po',weights[tuple(eye[j])],local(tuple(eye[i])))
        return result

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_shepard_hermite.v1',power=self.power,
            **{k:getattr(self,k).tolist() for k in ('nodes','values','first','second','lengths')})

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.dvr.periodic_shepard_hermite.v1':raise ValueError('Unsupported periodic Hermite partition')
        return cls(*(np.array(data[k]) for k in ('nodes','values','first','second','lengths')),int(data['power']))
