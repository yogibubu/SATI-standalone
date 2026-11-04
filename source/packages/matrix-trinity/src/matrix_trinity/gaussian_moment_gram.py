"""Gaussian Gram coefficients on the exact lowering closure of a finite basis."""
import numpy as np
from numba import njit


@njit(cache=True)
def _closed_gram(rows,lower,q,r,h):
    size,width=rows.shape;gram=np.empty((size,size))
    gram[0,0]=1.
    for i in range(size):
        for j in range(i+1):
            if i==0:continue
            axis=0
            while rows[i,axis]==0:axis+=1
            previous=lower[i,axis]
            value=h[axis]*gram[previous,j]
            for k in range(width):
                left=rows[previous,k]
                if left:value+=q[axis,k]*np.sqrt(left)*gram[lower[previous,k],j]
                right=rows[j,k]
                if right:value+=r[axis,k]*np.sqrt(right)*gram[previous,lower[j,k]]
            value/=np.sqrt(rows[i,axis])
            gram[i,j]=value;gram[j,i]=value
    return gram


def gaussian_gram_on_lowering_closure(basis,q,r,h,*,maximum_entries=8000000):
    """Exact Mehler Gram matrix without a Cartesian bounding-box tensor.

    The symmetric generating matrix is [[q,r],[r.T,q]], with repeated linear
    vector h. Every needed lower occupation is retained; nothing is pruned by
    coefficient size. The resource guard applies to the actual closure square.
    """
    basis=np.asarray(basis);q=np.asarray(q,float);r=np.asarray(r,float);h=np.asarray(h,float)
    if (basis.ndim!=2 or basis.dtype.kind not in 'iu' or np.any(basis<0)
            or not len(basis) or q.shape!=(basis.shape[1],)*2 or r.shape!=q.shape
            or h.shape!=(basis.shape[1],) or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(h))
            or not np.allclose(q,q.T,atol=1e-12,rtol=1e-12)
            or not np.allclose(r,r.T,atol=1e-12,rtol=1e-12)
            or not isinstance(maximum_entries,int) or maximum_entries<1):
        raise ValueError('finite Gaussian matrices, nonnegative occupations and a positive budget required')
    from math import isqrt
    rows,lower,take=gaussian_lowering_closure(basis,maximum_states=isqrt(maximum_entries))
    gram=_closed_gram(rows,lower,q,r,h)
    return gram[np.ix_(take,take)],dict(initial_moment_grid_entries=len(rows)**2,
        lowering_closure_size=len(rows),moment_algorithm='exact lowering-closure Gaussian Gram',
        coefficient_pruning_used=False)


def gaussian_lowering_closure(basis, *, maximum_states):
    """Shared exact downward closure, its lower neighbors and original row map."""
    basis=np.asarray(basis)
    if (basis.ndim!=2 or basis.dtype.kind not in 'iu' or not len(basis) or np.any(basis<0)
            or isinstance(maximum_states,(bool,np.bool_))
            or not isinstance(maximum_states,(int,np.integer)) or maximum_states<1):
        raise ValueError('nonnegative integer occupations and a positive closure budget required')
    original=[tuple(int(x) for x in row) for row in basis]
    closure=set(original);pending=list(closure)
    if len(closure)>maximum_states:raise MemoryError('Gaussian lowering closure exceeds declared state budget')
    while pending:
        row=pending.pop()
        for axis,count in enumerate(row):
            if count:
                child=row[:axis]+(count-1,)+row[axis+1:]
                if child not in closure:
                    closure.add(child);pending.append(child)
                    if len(closure)>maximum_states:
                        raise MemoryError('Gaussian lowering closure exceeds declared state budget')
    ordered=sorted(closure,key=lambda row:(sum(row),row));lookup={row:i for i,row in enumerate(ordered)}
    rows=np.asarray(ordered,dtype=np.int64);lower=np.full(rows.shape,-1,dtype=np.int64)
    for i,row in enumerate(ordered):
        for axis,count in enumerate(row):
            if count:lower[i,axis]=lookup[row[:axis]+(count-1,)+row[axis+1:]]
    return rows, lower, np.array([lookup[row] for row in original])
