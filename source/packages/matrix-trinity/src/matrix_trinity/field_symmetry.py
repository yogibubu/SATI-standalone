"""Finite-group projection and structural zeros for normal-coordinate fields.

The Reynolds projector is valid for abelian and nonabelian real finite groups.
Zero decisions concern its diagonal, never the size of a force constant.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def canonical_actions(actions, *, tolerance=1e-7):
    """Validate a closed orthogonal representation; repair signed permutations.

    The tolerance measures representation/acquisition precision. A nearby
    signed permutation is accepted only when the COMPLETE resulting group
    is closed. General dense representations retain their full matrices.
    """
    matrices=np.asarray(actions,float)
    if (not np.isfinite(tolerance) or tolerance<=0 or matrices.ndim!=3
            or matrices.shape[1]!=matrices.shape[2] or not len(matrices)
            or not np.all(np.isfinite(matrices))):
        raise ValueError('finite square group actions and positive tolerance required')
    n=matrices.shape[1]
    if max(np.max(abs(a.T@a-np.eye(n))) for a in matrices)>tolerance:
        raise ValueError('field symmetry actions are not orthogonal at the declared precision')
    result=[]
    for action in matrices:
        rounded=np.rint(action)
        if (np.max(abs(action-rounded))<=tolerance
                and np.array_equal(rounded.T@rounded,np.eye(n))):
            result.append(rounded)
        else:
            u,_,vt=np.linalg.svd(action)
            result.append(u@vt)
    result=np.asarray(result)
    if min(np.max(abs(a-np.eye(n))) for a in result)>tolerance:
        raise ValueError('field symmetry representation lacks identity')
    for a in result:
        for b in result:
            if min(np.max(abs(a@b-c)) for c in result)>tolerance:
                raise ValueError('field symmetry actions are not a closed finite group')
    # Repeated images of a nonfaithful molecular representation must have
    # equal multiplicity; otherwise averaging would not be a group projector.
    representatives=[];counts=[]
    for a in result:
        matches=[i for i,b in enumerate(representatives) if np.max(abs(a-b))<=tolerance]
        if matches: counts[matches[0]]+=1
        else: representatives.append(a);counts.append(1)
    if len(set(counts))!=1:
        raise ValueError('unequal multiplicities in a nonfaithful group representation')
    return result


def invariant_tensor(tensor, actions):
    """Project a covariant real tensor and enforce algebraic forbidden entries.

    For orthogonal finite groups, P is an orthogonal projector. P[ii]=0
    implies its entire row is zero. For a rank-r product representation,
    P[ii]=mean_g product_k D_g[i_k,i_k], computed without constructing n**2r
    matrices. Only machine-roundoff zeros of this projector are enforced.
    """
    tensor=np.asarray(tensor,float)
    n=actions.shape[1]
    if not tensor.ndim or tensor.shape!=(n,)*tensor.ndim:
        raise ValueError('tensor axes must match the group representation')
    diagonal=np.zeros(tensor.shape)
    for a in actions:
        product=np.diag(a)
        for _ in range(1,tensor.ndim):
            product=np.multiply.outer(product,np.diag(a))
        diagonal+=product/len(actions)
    forbidden=abs(diagonal)<=32*np.finfo(float).eps*len(actions)
    if all(np.array_equal(a,np.diag(np.diag(a))) for a in actions):
        result=tensor.copy()
    else:
        result=np.zeros_like(tensor)
        for a in actions:
            term=tensor
            for axis in range(tensor.ndim):
                term=np.moveaxis(np.tensordot(a,term,axes=(1,axis)),0,axis)
            result+=term/len(actions)
    result[forbidden]=0.
    return result


def invariant_eigh(matrix, actions):
    """Diagonalize invariant structural blocks without cross-block roundoff."""
    projected=invariant_tensor(matrix,actions)
    projected=(projected+projected.T)/2
    count,labels=connected_components(csr_matrix(projected!=0),directed=False)
    values=[];columns=[]
    for label in range(count):
        indices=np.flatnonzero(labels==label)
        local,rotation=np.linalg.eigh(projected[np.ix_(indices,indices)])
        block=np.zeros((len(projected),len(indices)))
        block[indices]=rotation
        values.extend(local);columns.extend(block.T)
    order=np.argsort(values,kind='stable')
    return np.asarray(values)[order],np.asarray(columns).T[:,order]
