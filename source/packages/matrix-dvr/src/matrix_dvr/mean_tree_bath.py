"""Exact physical-valence mean paths and conditional transverse frames.

The active physical dihedrals remain exactly fixed when a transverse response
is applied. Internal bond lengths and angles are realized by the complete
tree inverse, rather than a Cartesian Taylor expansion in a finite offset.
This owner is a nuclear integration embedding, not a structural-fit basis.
"""
import numpy as np


def mean_tree_bath_jets(path,points,response,response_first,partition,*,response_second=None,second=True):
    """Realize the complete mean path; optionally omit all second jets.

    ``second=False`` evaluates the same coordinates and tangent columns for
    first-order metric and transverse spectral quadrature without constructing
    unused Cartesian curvature tensors.
    """
    q=np.asarray(points,float);b=np.asarray(response,float);b1=np.asarray(response_first,float)
    a=path.active_indices;all_b=path.bath_indices;d=len(a);n=len(all_b);N=len(q)
    order=np.asarray(partition['bath_order'],int);selected=np.asarray(partition['explicit_indices'],int);fast=np.asarray(partition['fast_indices'],int)
    if (q.shape!=(N,d) or b.shape!=(N,n) or b1.shape!=(N,n,d) or sorted(order.tolist())!=sorted(all_b.tolist())
            or not all(np.all(np.isfinite(x)) for x in (q,b,b1))):raise ValueError('Complete physical tree response and coordinate ordering required')
    nv=d+n;s=len(selected);f=len(fast);eye=np.eye(d,dtype=int)
    c=path.path_field.evaluate(q);c1=np.stack([path.path_field.evaluate(q,e) for e in eye],axis=-1)
    c2=(np.stack([np.stack([path.path_field.evaluate(q,e+e2) for e2 in eye],axis=-1) for e in eye],axis=-2) if second and response_second is not None else None)
    natural=np.tile(path.reference_values,(N,1));natural[:,a]+=q;natural[:,all_b]+=c;natural[:,order]+=b
    T=np.zeros((N,nv,d+s+f));T[:,a,np.arange(d)]=1.;T[:,all_b,:d]=c1;T[:,order,:d]+=b1
    T[:,order[selected],d+np.arange(s)]=1.;T[:,order[fast],d:d+s]=partition['conditional_response']
    T[:,order[fast],d+s+np.arange(f)]=1.
    U=np.zeros((N,nv,d+s+f,d+s+f)) if second else None
    if second:
        U[:,order[fast],d:d+s,:d]=partition['conditional_response_first'];U[:,:,:d,d:d+s]=U[:,:,d:d+s,:d].transpose(0,1,3,2)
    if second and response_second is not None:
        b2=np.asarray(response_second,float)
        if b2.shape!=(N,n,d,d) or not np.all(np.isfinite(b2)):raise ValueError('Matching second response jet required')
        U[:,all_b,:d,:d]=c2;U[:,order,:d,:d]+=b2
    xx=[];dd=[];second_jets=[]
    for k,row in enumerate(natural):
        x=path.tree.geometry(row);jet=path.tree.cartesian_jet(x,second=U is not None);D=jet['tangent']@T[k]
        if U is not None:
            DD=np.einsum('aij,ik,jl->akl',jet['second'],T[k],T[k],optimize=True)+np.einsum('ai,ijk->ajk',jet['tangent'],U[k])
            second_jets.append(DD)
        xx.append(x);dd.append(D)
    return dict(geometry_A=np.array(xx),first=np.array(dd),second=np.array(second_jets) if U is not None else None,natural_values=natural,
        active_dimensions=d+s,fast_dimensions=f,active_physical_angles_unchanged=True)
