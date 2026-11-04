"""Exact tensor quadrature contractions on symmetric univariate pair tables."""
from math import prod
import numpy as np


def evaluate_pruned_product(basis,functions,coefficients):
    """Evaluate weighted product expansions with the identical retained labels."""
    labels=np.asarray(basis);tables=tuple(np.asarray(f) for f in functions)
    c=np.asarray(coefficients);vector=c.ndim==1
    if vector:c=c[:,None]
    if (labels.ndim!=2 or labels.dtype.kind not in 'iu' or labels.shape[1]!=len(tables)
            or c.ndim!=2 or len(c)!=len(labels) or np.any(labels<0)
            or any(f.ndim!=2 or labels[:,i].max()>=len(f) for i,f in enumerate(tables))):
        raise ValueError('Matching retained product labels, tables and coefficients required')
    tensor=np.zeros(tuple(len(f) for f in tables)+(c.shape[1],),dtype=np.result_type(c,*tables))
    tensor[tuple(labels.T)]=c
    for axis,f in enumerate(tables):
        tensor=np.moveaxis(np.tensordot(f.T,tensor,axes=(1,axis)),0,axis)
    result=tensor.reshape(-1,c.shape[1])
    return result[:,0] if vector else result


def adjoint_pruned_product(basis, functions, grid_values):
    """Apply the exact adjoint of evaluate_pruned_product without a tall matrix.

    Function tables include their original quadrature weights. Additional
    trailing columns are independent right hand sides; no states are removed.
    """
    labels = np.asarray(basis)
    tables = tuple(np.asarray(f) for f in functions)
    values = np.asarray(grid_values)
    if (labels.ndim != 2 or labels.dtype.kind not in 'iu'
            or labels.shape[1] != len(tables) or not len(labels)
            or np.any(labels < 0) or values.ndim not in (1, 2)
            or any(f.ndim != 2 or labels[:, i].max() >= len(f)
                   for i, f in enumerate(tables))):
        raise ValueError('Matching retained product labels, tables and grid values required')
    counts = tuple(f.shape[1] for f in tables)
    if len(values) != prod(counts):
        raise ValueError('Grid values must follow the original tensor quadrature')
    vector = values.ndim == 1
    tensor = values.reshape(counts + (() if vector else (values.shape[1],)))
    for axis, f in enumerate(tables):
        tensor = np.moveaxis(np.tensordot(f.conj(), tensor, axes=(1, axis)), 0, axis)
    return tensor[tuple(labels.T)]


def project_pruned_product_field(core_basis,left_functions,right_functions,field,
        *,maximum_workspace_bytes=8*1024**3):
    """Exact product-field projection, including derivative function tables.

    The last modal coordinate is complete; the other coordinates use the
    supplied pruned integer labels. Row order is core label then last mode.
    Univariate pair products are contracted before gathering retained rows.
    No tensor ranks, matrix entries or numerical couplings are truncated.
    """
    basis=np.asarray(core_basis)
    left=[np.asarray(f) for f in left_functions];right=[np.asarray(f) for f in right_functions]
    if (basis.ndim!=2 or basis.dtype.kind not in 'iu' or np.any(basis<0)
            or len(left)!=basis.shape[1]+1 or len(right)!=len(left) or not len(basis)
            or len({tuple(row) for row in basis})!=len(basis)
            or any(a.ndim!=2 or a.shape!=b.shape for a,b in zip(left,right))
            or any(not np.all(np.isfinite(f)) for f in left+right)):
        raise ValueError('Unique pruned labels and matching finite product tables required')
    if any(np.max(basis[:,i])>=len(f) for i,f in enumerate(left[:-1])):
        raise ValueError('Function tables do not cover the retained core labels')
    values=np.asarray(field)
    counts=tuple(f.shape[1] for f in left)
    if values.size!=prod(counts) or not np.all(np.isfinite(values)):
        raise ValueError('One finite field value per original tensor node required')
    # Exact constant phases keep real/pure-imaginary derivative tables real.
    phase=1.+0j
    for tables,conjugate in ((left,True),(right,False)):
        for i,f in enumerate(tables):
            if np.iscomplexobj(f) and not np.any(f.imag):tables[i]=f.real
            elif np.iscomplexobj(f) and not np.any(f.real):
                tables[i]=f.imag;phase*=(-1j if conjugate else 1j)
    symmetric=(not np.iscomplexobj(values) and all(np.isrealobj(a) and np.array_equal(a,b)
                for a,b in zip(left,right)))
    pairs=[];lookups=[]
    for a,b in zip(left[:-1],right[:-1]):
        n=len(a)
        if np.isrealobj(a) and np.array_equal(a,b):
            i,j=np.triu_indices(n);lookup=np.empty((n,n),int)
            ids=np.arange(len(i));lookup[i,j]=ids;lookup[j,i]=ids
        else:
            i=np.repeat(np.arange(n),n);j=np.tile(np.arange(n),n)
            lookup=np.arange(n*n).reshape(n,n)
        pairs.append(a[i].conj()*b[j]);lookups.append(lookup)
    dtype=np.result_type(values,*[f.dtype for f in left+right])
    order=sorted(range(len(pairs)),key=lambda a:len(pairs[a])/counts[a])
    sizes=list(counts[:-1]);largest=prod(sizes)
    for axis in order:sizes[axis]=len(pairs[axis]);largest=max(largest,prod(sizes))
    last=len(left[-1]);size=len(basis)*last
    if (4*largest+size*size)*np.dtype(dtype).itemsize>maximum_workspace_bytes:
        raise MemoryError('Exact tensor pair projection exceeds declared workspace')
    result=np.empty((size,size),dtype=dtype)
    grid=values.reshape(counts)
    gather=tuple(lookup[basis[:,axis,None],basis[None,:,axis]] for axis,lookup in enumerate(lookups))
    for a in range(last):
        for b in range(a if symmetric else 0,last):
            tensor=np.tensordot(grid,left[-1][a].conj()*right[-1][b],axes=(-1,0))
            for axis in order:
                tensor=np.moveaxis(np.tensordot(pairs[axis],tensor,axes=(1,axis)),0,axis)
            block=tensor[gather];result[a::last,b::last]=block
            if symmetric and a!=b:result[b::last,a::last]=block.T
    if phase.imag==0:return result*phase.real
    return result*phase


def plan_native_tensor_projection(functions, parent_size, *, reserved_bytes,
                                 maximum_workspace_bytes):
    """Select an exact contraction only when its memory and work are favorable."""
    if len(functions)<2:return None
    shape=[f.shape[1] for f in functions]
    pairs=[f.shape[0]*(f.shape[0]+1)//2 for f in functions]
    order=sorted(range(len(shape)),key=lambda axis:pairs[axis]/shape[axis])
    sizes=[prod(shape)];work=0
    for axis in order:
        old=shape[axis];shape[axis]=pairs[axis]
        size=prod(shape);sizes.append(size);work+=size*old
    required=reserved_bytes+4*max(sizes)*8
    rank_work=prod(f.shape[1] for f in functions)*parent_size**2
    if required>maximum_workspace_bytes or work>=rank_work/4:return None
    return dict(axis_order=order,pair_dimensions=pairs,maximum_tensor_entries=max(sizes),
                workspace_guard_bytes=required,estimated_tensor_multiply_adds=work,
                estimated_rank_multiply_adds=rank_work)


def project_native_tensor_correction(matrix, core_basis, blocks, functions,
        residual, plan, *, row_batch_size=128, progress=None):
    """Add every native correction with the original finite tensor rule.

    The last function table belongs to the added coordinate. Each core axis
    contracts products f_n(q) f_m(q) once for every unordered univariate pair.
    No singular values, polynomial coefficients or matrix entries are dropped.
    """
    pair_functions=[];lookups=[]
    for f in functions[:-1]:
        left,right=np.triu_indices(len(f))
        pair_functions.append(f[left]*f[right])
        lookup=np.empty((len(f),len(f)),np.int64)
        ids=np.arange(len(left));lookup[left,right]=ids;lookup[right,left]=ids
        lookups.append(lookup)
    counts=tuple(f.shape[1] for f in functions)
    values=np.asarray(residual).reshape(counts)
    count=len(blocks)*(len(blocks)+1)//2;completed=0
    for i,(si,ei,oi,ti) in enumerate(blocks):
        for sj,ej,oj,tj in blocks[i:]:
            extra=functions[-1][oi]*functions[-1][oj]
            tensor=np.tensordot(values,extra,axes=(-1,0))
            for axis in plan['axis_order']:
                tensor=np.moveaxis(np.tensordot(pair_functions[axis],tensor,axes=(1,axis)),0,axis)
            for start in range(0,len(ti),row_batch_size):
                stop=min(start+row_batch_size,len(ti))
                indices=tuple(lookup[core_basis[ti[start:stop],axis,None],
                                     core_basis[tj,axis][None,:]]
                              for axis,lookup in enumerate(lookups))
                matrix[si+start:si+stop,sj:ej]+=tensor[indices]
            del tensor
            completed+=1
            if progress:progress(dict(stage='tensor projection',completed_blocks=completed,total_blocks=count))
