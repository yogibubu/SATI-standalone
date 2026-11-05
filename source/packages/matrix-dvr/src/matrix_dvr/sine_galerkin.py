"""Dirichlet sine products and their exact derivatives at Gauss nodes.

Quadrature orthonormalization transforms values and derivatives together.
The basis vanishes at fixed physical walls, independently of resolution.
"""
from itertools import product
from math import prod
import numpy as np
from scipy.fft import dstn


def tensor_sine_dvr_quadrature_wavefunctions(vectors, *, bounds, basis_counts,
        quadrature_counts, maximum_workspace_bytes=2*1024**3):
    """Evaluate saved flat-measure sine-DVR columns without building a Hamiltonian.

    Reuses the same one-dimensional oversampled sine values as the full
    curvilinear operator. Returned waves contain sqrt(quadrature weight),
    and must not receive another physical-volume factor.
    """
    counts=tuple(basis_counts);nq=tuple(quadrature_counts);limits=np.asarray(bounds,float)
    u=np.asarray(vectors);d=len(counts)
    if (not d or limits.shape!=(d,2) or len(nq)!=d or u.ndim!=2 or not u.shape[1]
            or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<1 for n in counts)
            or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<=counts[i] for i,n in enumerate(nq))
            or len(u)!=prod(counts) or not np.all(np.isfinite(u))
            or not np.all(np.isfinite(limits)) or np.any(limits[:,1]<=limits[:,0])):
        raise ValueError('finite saved sine-DVR columns, matching box and resolved quadrature required')
    estimate=prod(nq)*(24*d+8+3*u.shape[1]*np.result_type(u.dtype,float).itemsize)+u.nbytes
    if (isinstance(maximum_workspace_bytes,bool) or not isinstance(maximum_workspace_bytes,int)
            or maximum_workspace_bytes<1 or estimate>maximum_workspace_bytes):
        raise MemoryError('saved-state quadrature exceeds declared numeric workspace')
    axes=[sine_product_quadrature([b],[n],[q]) for b,n,q in zip(limits,counts,nq)]
    coefficients=dstn(u.reshape(counts+(u.shape[1],)),type=1,axes=tuple(range(d)),norm='ortho')
    for axis,rule in enumerate(axes):
        coefficients=np.moveaxis(np.tensordot(rule['values'],coefficients,axes=(1,axis)),0,axis)
    points=np.stack(np.meshgrid(*[a['points'][:,0] for a in axes],indexing='ij'),axis=-1).reshape(-1,d)
    weights=np.prod(np.stack(np.meshgrid(*[a['weights'] for a in axes],indexing='ij'),axis=-1),axis=-1).ravel()
    return dict(points=points,weights=weights,weighted_wavefunctions=coefficients.reshape(-1,u.shape[1]),
        diagnostics=dict(numeric_workspace_estimate_bytes=estimate,normalized=False,
            physical_volume_factor_added=False,hamiltonian_prepared=False))


def sine_product_quadrature(bounds, basis_counts, quadrature_counts, *,
                            quadrature_bounds=None, orthonormalize=True):
    limits=np.asarray(bounds,float);counts=tuple(basis_counts);nq=tuple(quadrature_counts)
    d=len(counts)
    if (limits.shape!=(d,2) or len(nq)!=d or d<1 or not np.all(np.isfinite(limits))
            or np.any(limits[:,1]<=limits[:,0])
            or any(isinstance(n,bool) or int(n)!=n or n<1 for n in counts)
            or any(isinstance(n,bool) or int(n)!=n or n<=counts[i] for i,n in enumerate(nq))):
        raise ValueError('Finite ordered walls and compatible positive sine/quadrature counts required')
    integration=limits if quadrature_bounds is None else np.asarray(quadrature_bounds,float)
    if (integration.shape!=limits.shape or not np.isfinite(integration).all()
            or np.any(integration[:,1]<=integration[:,0])
            or np.any(integration[:,0]<limits[:,0]-1e-13)
            or np.any(integration[:,1]>limits[:,1]+1e-13)
            or not isinstance(orthonormalize,(bool,np.bool_))):
        raise ValueError('Ordered integration walls inside the physical sine box required')
    # A restricted physical overlap must retain the original wavefunction
    # normalization. The default full-box quadrature behavior is unchanged.
    axes=[];weights=[]
    for (lo,hi),n in zip(integration,nq):
        x,w=np.polynomial.legendre.leggauss(n)
        axes.append((lo+hi)/2+(hi-lo)*x/2);weights.append(w*(hi-lo)/2)
    q=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,d)
    w=np.prod(np.stack(np.meshgrid(*weights,indexing='ij'),axis=-1),axis=-1).ravel()
    occupations=np.array(list(product(*(range(1,n+1) for n in counts))))
    width=limits[:,1]-limits[:,0]
    angle=(q[:,None,:]-limits[None,None,:,0])*occupations[None,:,:]*np.pi/width
    normal=np.sqrt(2/width)
    factors=np.sin(angle)*normal
    values=np.prod(factors,axis=-1)*np.sqrt(w)[:,None]
    momenta=[]
    for i in range(d):
        derivative=factors.copy()
        derivative[:,:,i]=np.cos(angle[:,:,i])*normal[i]*occupations[None,:,i]*np.pi/width[i]
        momenta.append(-1j*np.prod(derivative,axis=-1)*np.sqrt(w)[:,None])
    overlap=values.T@values;ev,vec=np.linalg.eigh(overlap)
    momenta=np.stack(momenta,axis=1)
    if orthonormalize:
        if min(ev)<=1e-12*max(ev):raise ValueError('Quadrature cannot resolve the sine basis')
        transform=(vec/np.sqrt(ev))@vec.T
        values=values@transform;momenta=momenta@transform
    return dict(points=q,weights=w,values=values,momenta=momenta,occupations=occupations,
                quadrature_overlap_eigenvalues=ev,
                orthogonality_error=float(np.max(abs(values.T@values-np.eye(len(occupations))))),
                bounds=limits,quadrature_bounds=integration,orthonormalized=bool(orthonormalize),
                basis_counts=counts,quadrature_counts=nq)
