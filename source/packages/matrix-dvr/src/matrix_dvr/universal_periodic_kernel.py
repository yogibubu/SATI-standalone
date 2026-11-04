"""Universal periodic Hermite regression with finite coordinate symmetries.

A caller supplies a reference-compatible trend design. The residual kernel is
projected onto a finite signed-permutation group, retaining positive
semidefiniteness. Grouped deletion errors use training observations only.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .periodic_hermite_kernel import periodic_hermite_covariance


def _group_array(group,dimension):
    raw=np.asarray(group);g=np.asarray(raw,int)
    if (g.ndim!=3 or g.shape[1:]!=(dimension,dimension)
            or not np.array_equal(raw,g) or np.any(abs(g)>1)
            or np.any(abs(g).sum(axis=1)!=1) or np.any(abs(g).sum(axis=2)!=1)):
        raise ValueError('A finite signed-permutation group is required')
    keys={tuple(t.ravel()) for t in g}
    if len(keys)!=len(g) or tuple(np.eye(dimension,dtype=int).ravel()) not in keys:
        raise ValueError('Unique group elements including identity required')
    if any(tuple((a@b).ravel()) not in keys for a in g for b in g):
        raise ValueError('Coordinate transformations must form a closed group')
    return g


def _transform(points,orders,matrix):
    permutation=np.argmax(abs(matrix),axis=1);sign=matrix[np.arange(len(matrix)),permutation]
    transformed_orders=np.asarray(orders)[:,permutation]
    return (np.asarray(points)[:,permutation]*sign,transformed_orders,
            np.prod(sign**transformed_orders,axis=1))


def symmetric_periodic_covariance(left,left_orders,right,right_orders,lengthscales,group,identity_fraction=0.):
    """Double group average of the existing periodic derivative covariance."""
    if not np.isfinite(identity_fraction) or not 0<=identity_fraction<=1:
        raise ValueError('Identity covariance fraction must be between zero and one')
    metric=np.diag(1/np.asarray(lengthscales)**2)
    invariant=all(np.allclose(t@metric@t.T,metric,rtol=1e-13,atol=1e-13) for t in group)
    left_group=[np.eye(len(metric),dtype=int)] if invariant else group
    result=np.zeros((len(left),len(right)))
    for a in left_group:
        x,ox,sx=_transform(left,left_orders,a)
        for b in group:
            y,oy,sy=_transform(right,right_orders,b)
            result+=sx[:,None]*sy[None,:]*periodic_hermite_covariance(
                x,ox,y,oy,lengthscales,reflection_even=False)
    result*= (1-identity_fraction)/(len(left_group)*len(group))
    if identity_fraction:
        result+=identity_fraction*periodic_hermite_covariance(left,left_orders,right,right_orders,lengthscales,reflection_even=False)
    return result


@dataclass(frozen=True)
class SymmetricPeriodicKernelField:
    centers: np.ndarray
    orders: np.ndarray
    coefficients: np.ndarray
    lengthscales: np.ndarray
    group: np.ndarray
    identity_fraction: float = 0.

    def __post_init__(self):
        for key in ('centers','orders','coefficients','lengthscales','group'):
            raw=getattr(self,key);v=np.array(raw,dtype=int if key in ('orders','group') else float,copy=True)
            if not np.all(np.isfinite(v)) or (key in ('orders','group') and not np.array_equal(raw,v)):
                raise ValueError('Finite parameters and exact integer transformations required')
            v.flags.writeable=False;object.__setattr__(self,key,v)
        if (self.centers.ndim!=2 or self.orders.shape!=self.centers.shape
                or np.any(self.orders<0) or self.coefficients.shape!=(len(self.centers),)
                or self.lengthscales.shape!=(self.centers.shape[1],) or np.any(self.lengthscales<=0)):
            raise ValueError('Compatible scalar periodic expansion required')
        _group_array(self.group,self.centers.shape[1])
        if not np.isfinite(self.identity_fraction) or not 0<=self.identity_fraction<=1:
            raise ValueError('Identity covariance fraction must be between zero and one')

    def evaluate(self,points,derivative=None):
        q=np.asarray(points,float);d=np.zeros(self.centers.shape[1],int) if derivative is None else np.asarray(derivative)
        if q.ndim!=2 or q.shape[1]!=self.centers.shape[1] or d.shape!=(q.shape[1],) or not np.array_equal(d,np.asarray(d,int)) or np.any(d<0):
            raise ValueError('Coordinate matrix and nonnegative derivative multi-index required')
        result=np.empty(len(q))
        for start in range(0,len(q),256):
            block=q[start:start+256];result[start:start+len(block)]=symmetric_periodic_covariance(block,np.broadcast_to(d,block.shape),self.centers,self.orders,self.lengthscales,self.group,self.identity_fraction)@self.coefficients
        return result

    def to_dict(self):
        return dict(schema='matrix.dvr.symmetric_periodic_kernel.v1',identity_fraction=float(self.identity_fraction),**{k:getattr(self,k).tolist() for k in ('centers','orders','coefficients','lengthscales','group')})

    @classmethod
    def from_dict(cls,r):
        if r.get('schema')!='matrix.dvr.symmetric_periodic_kernel.v1':raise ValueError('Unsupported symmetric periodic field')
        return cls(*(r[k] for k in ('centers','orders','coefficients','lengthscales','group')),identity_fraction=r.get('identity_fraction',0.))


def fit_universal_periodic_kernel(points,orders,values,*,lengthscales,group,
        weights,trend_design,constraint_points,constraint_orders,constraint_values,
        relative_noise=1e-8,deletion_groups=(),identity_fraction=0.):
    """Constrained universal kriging and analytic grouped training deletions.

    Trend columns must vanish at the supplied constraints and be independent
    on the training rows. Deletion groups contain observation-row indices;
    correlated or symmetry-related measurements should be deleted together.
    Deletion residuals are predictions minus observations in original units.
    """
    x=np.asarray(points,float);o=np.asarray(orders);y=np.asarray(values,float);ell=np.asarray(lengthscales,float)
    c=np.asarray(constraint_points,float);co=np.asarray(constraint_orders);cy=np.asarray(constraint_values,float)
    w=np.asarray(weights,float);A=np.asarray(trend_design,float);group=_group_array(group,x.shape[1])
    if (o.shape!=x.shape or co.shape!=c.shape or x.ndim!=2 or c.ndim!=2 or x.shape[1]!=c.shape[1]
            or not np.array_equal(o,np.asarray(o,int)) or not np.array_equal(co,np.asarray(co,int)) or np.any(o<0) or np.any(co<0)
            or y.shape!=(len(x),) or cy.shape!=(len(c),) or w.shape!=y.shape or A.ndim!=2 or len(A)!=len(x)
            or ell.shape!=(x.shape[1],) or np.any(ell<=0) or np.any(w<=0) or not 0<relative_noise<1
            or not all(np.all(np.isfinite(v)) for v in (x,c,y,cy,w,A,ell))):
        raise ValueError('Compatible finite constrained Hermite regression required')
    covariance=lambda a,ao,b,bo:symmetric_periodic_covariance(a,ao,b,bo,ell,group,identity_fraction)
    Kcc=covariance(c,co,c,co);cc=cho_factor((Kcc+Kcc.T)/2,lower=True)
    Kxc=covariance(x,o,c,co);transfer=cho_solve(cc,Kxc.T);reference=cho_solve(cc,cy)
    K=covariance(x,o,x,o)-Kxc@transfer;K=(K+K.T)/2;K*=w[:,None]*w[None,:]
    noise=relative_noise*max(1.,float(np.diag(K).max(initial=0)))
    factor=cho_factor(K+noise*np.eye(len(x)),lower=True)
    inv=cho_solve(factor,np.eye(len(x)));design=A*w[:,None];rhs=(y-Kxc@reference)*w
    inv_design=inv@design
    if A.shape[1]:
        normal=design.T@inv_design;normal=(normal+normal.T)/2
        nf=cho_factor(normal,lower=True);beta=cho_solve(nf,design.T@inv@rhs)
        precision=inv-inv_design@cho_solve(nf,inv_design.T)
    else:beta=np.empty(0);precision=inv
    alpha=inv@(rhs-design@beta)
    coefficients=np.r_[alpha*w,reference-transfer@(alpha*w)]
    field=SymmetricPeriodicKernelField(np.vstack([x,c]),np.vstack([o,co]),coefficients,ell,group,identity_fraction)
    deletion=[]
    for indices in deletion_groups:
        ids=np.asarray(indices,int)
        if ids.ndim!=1 or len(ids)==0 or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(x)):
            raise ValueError('Nonempty unique deletion indices within training rows required')
        block=precision[np.ix_(ids,ids)];block=(block+block.T)/2
        deleted_error=-np.linalg.solve(block,alpha[ids])/w[ids]
        deletion.append(dict(indices=ids.tolist(),prediction_minus_observation=deleted_error.tolist()))
    predicted=A@beta+Kxc@reference+(K@alpha)/w
    audit=dict(relative_noise=relative_noise,absolute_noise=noise,trend_parameters=A.shape[1],training_prediction=predicted.tolist(),deletion_groups=deletion,maximum_constraint_residual=float(np.max(abs(covariance(c,co,field.centers,field.orders)@coefficients-cy))))
    return field,beta,audit
