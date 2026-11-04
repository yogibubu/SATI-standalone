"""Batched evaluation of an existing retained internal-tree chart.

This changes only the order of matrix operations. The same ORACLE primitives,
ZAFF B', internal path and retained-bath jets are used as by cartesian_jet.
"""
import numpy as np


def retained_tree_cartesian_jets(chart,points):
    q=np.asarray(points,float);path=chart.path;d=len(path.active_indices)
    if q.ndim!=2 or q.shape[1]!=d or not len(q) or not np.all(np.isfinite(q)):
        raise ValueError('Nonempty compatible finite chart points required')
    internal=chart.internal_jets(q);values=np.tile(path.reference_values,(len(q),1))
    values[:,path.active_indices]+=q;values[:,path.bath_indices]+=path.path_field.evaluate(q)
    coordinates=[];first=[];second=[]
    for k,row in enumerate(values):
        x=path.tree.geometry(row);tree=path.tree.cartesian_jet(x);T=internal['first'][k]
        D=tree['tangent']@T
        DD=np.einsum('aij,ik,jl->akl',tree['second'],T,T,optimize=True)+np.einsum('ai,ijk->ajk',tree['tangent'],internal['second'][k])
        coordinates.append(x);first.append(D);second.append(DD)
    return dict(geometry_A=np.array(coordinates),tangent=np.array(first),second=np.array(second))
