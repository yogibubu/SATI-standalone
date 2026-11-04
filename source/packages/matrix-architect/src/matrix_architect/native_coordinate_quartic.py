"""Stationary quartic fields in nonlinear, generally coupled native variables.

The supplied coordinate jets must come from one frozen molecular definition.
No primitive redundancy is removed by discarding mixed derivative terms.
"""
from dataclasses import dataclass
from itertools import combinations, permutations
import numpy as np
from .native_variable_quartic import _contract


def _symmetric_input(tensor, degree, n):
    value=np.asarray(tensor,float)
    if value.shape!=(n,)*degree or not np.all(np.isfinite(value)):
        raise ValueError('finite aligned potential tensors required')
    for i in range(degree-1):
        if not np.allclose(value,value.swapaxes(i,i+1),atol=1e-9,rtol=1e-9):
            raise ValueError('symmetric potential tensors required')
    return value


def stationary_composition(p2,p3,p4,first,second,third):
    """Chain rule through order four at a stationary point (potential P1=0)."""
    a,b,c=map(lambda x:np.asarray(x,float),(first,second,third))
    h=_contract(p2,a)
    f3=_contract(p3,a)
    base=np.einsum('ab,aij,bk->ijk',p2,b,a,optimize=True)
    f3+=base+base.transpose(0,2,1)+base.transpose(2,0,1)
    f4=_contract(p4,a)
    base=np.einsum('abc,aij,bk,cl->ijkl',p3,b,a,a,optimize=True)
    for pair in combinations(range(4),2):
        remaining=[i for i in range(4) if i not in pair]
        labels=list(pair)+remaining
        f4+=base.transpose(np.argsort(labels))
    base=np.einsum('ab,aij,bkl->ijkl',p2,b,b,optimize=True)
    f4+=base+base.transpose(0,2,1,3)+base.transpose(0,2,3,1)
    base=np.einsum('ab,aijk,bl->ijkl',p2,c,a,optimize=True)
    for single in range(4):
        labels=[i for i in range(4) if i!=single]+[single]
        f4+=base.transpose(np.argsort(labels))
    return h,f3,f4


@dataclass(frozen=True)
class NativeCoordinateQuartic:
    """One conditional potential; all native radial/angular products retained.

    This representation matches local constraints. Its off-reference accuracy
    and molecular dissociation support require independent qualification.
    """
    quadratic: np.ndarray
    cubic: np.ndarray
    quartic: np.ndarray

    @classmethod
    def from_geometry_realization(cls,hessian,cubic,quartic,kernel,realization,*,steps=(.02,.01)):
        """Transform a stationary field through the shared geometry two-jet.

        The missing coordinate third derivative is a fixed Richardson
        difference of analytic second derivatives, without potential calls.
        Potential tensors must use the caller's realization coordinates.
        """
        n=len(hessian);coarse,fine=map(float,steps)
        if not 0<fine<coarse:raise ValueError('ordered positive coordinate steps required')
        def second(q):
            xyz,a,b=realization(q);_,g,h=kernel.value(xyz)
            return (a.T@g).T,(np.einsum('ia,ijt,jb->abt',a,h,a,optimize=True)
                +np.einsum('it,iab->abt',g,b)).transpose(2,0,1)
        first,curvature=second(np.zeros(n));estimates=[]
        if not np.any(hessian):
            # A correction starting at order three does not use the third
            # coordinate derivative in the quartic chain rule.
            return cls.from_stationary_jet(hessian,cubic,quartic,first,curvature,np.zeros((n,)*4))
        for step in (coarse,fine):
            third=np.zeros((n,)*4)
            for k,d in enumerate(np.eye(n)*step):third[:,:,:,k]=(second(d)[1]-second(-d)[1])/(2*step)
            third=sum(third.transpose((0,)+tuple(i+1 for i in p)) for p in permutations(range(3)))/6
            estimates.append(third)
        third=(coarse**2*estimates[1]-fine**2*estimates[0])/(coarse**2-fine**2)
        return cls.from_stationary_jet(hessian,cubic,quartic,first,curvature,third)

    @classmethod
    def from_stationary_jet(cls,hessian,cubic,quartic,first,second,third):
        a,b,c=map(lambda x:np.asarray(x,float),(first,second,third))
        if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.all(np.isfinite(a)):
            raise ValueError('finite square native coordinate Jacobian required')
        n=len(a)
        if (b.shape!=(n,n,n) or c.shape!=(n,n,n,n)
                or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c))):
            raise ValueError('finite aligned coordinate derivatives required')
        h,f3,f4=[_symmetric_input(t,k,n) for t,k in [(hessian,2),(cubic,3),(quartic,4)]]
        inv=np.linalg.inv(a);p2=_contract(h,inv)
        zero3=np.zeros_like(f3);zero4=np.zeros_like(f4)
        lower3=stationary_composition(p2,zero3,zero4,a,b,c)[1]
        p3=_contract(f3-lower3,inv)
        lower4=stationary_composition(p2,p3,zero4,a,b,c)[2]
        p4=_contract(f4-lower4,inv)
        return cls(p2,p3,p4)

    def potential_jet(self,values,*,correction=None,correction_center=None):
        y=np.asarray(values,float);n=len(self.quadratic)
        if y.shape!=(n,) or not np.all(np.isfinite(y)):
            raise ValueError('finite aligned native coordinate values required')
        p2,p3,p4=self.quadratic,self.cubic,self.quartic
        energy=(y@p2@y/2+np.einsum('ijk,i,j,k',p3,y,y,y)/6
                +np.einsum('ijkl,i,j,k,l',p4,y,y,y,y)/24)
        g=p2@y+np.einsum('ijk,j,k->i',p3,y,y)/2+np.einsum('ijkl,j,k,l->i',p4,y,y,y)/6
        h=p2+np.einsum('ijk,k->ij',p3,y)+np.einsum('ijkl,k,l->ij',p4,y,y)/2
        if correction is not None:
            k=np.asarray(correction,float);center=np.asarray(correction_center,float)
            if k.shape!=(n,n) or center.shape!=(n,) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(center)) or not np.allclose(k,k.T):
                raise ValueError('finite symmetric native curvature and fixed center required')
            delta=y-center;energy+=delta@k@delta/2;g+=k@delta;h+=k
        return float(energy),g,h

    def compose_two_jet(self,values,first,second,**kwargs):
        e,g,h=self.potential_jet(values,**kwargs)
        a,b=np.asarray(first,float),np.asarray(second,float)
        if a.ndim!=2 or a.shape[0]!=len(g) or b.shape!=(len(g),a.shape[1],a.shape[1]):
            raise ValueError('aligned native coordinate two-jet required')
        return e,a.T@g,a.T@h@a+np.einsum('a,aij->ij',g,b)
