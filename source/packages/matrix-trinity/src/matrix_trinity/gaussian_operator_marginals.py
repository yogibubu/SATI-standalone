"""Small number blocks of a Gaussian source with a linear operator bank.

The existing partial Gaussian trace integrates every spectator analytically.
Cauchy coefficients of the remaining one/two-mode generating function are
extracted on padded deterministic FFT grids. These marginals guide selection;
they never replace the full-model physical transition evaluation.
"""
import numpy as np
from scipy.fft import ifftn, next_fast_len
from numba import njit
from .gaussian_partial_trace import gaussian_partial_number_trace


@njit(cache=True)
def _wick_moment(mean,covariance,ids):
    """Noncentral Gaussian moment of at most six projected linear forms."""
    if len(ids)==0:
        return 1.+0.j
    result=mean[ids[0]]*_wick_moment(mean,covariance,ids[1:])
    for j in range(1,len(ids)):
        rest=np.empty(len(ids)-2,np.int64);index=0
        for k in range(1,len(ids)):
            if k!=j:
                rest[index]=ids[k];index+=1
        result+=covariance[ids[0],ids[j]]*_wick_moment(mean,covariance,rest)
    return result


@njit(cache=True)
def _derivative_number_moments(mean,covariance,h,matrix,channels,modes,norm):
    first=np.zeros(modes);second=np.zeros((modes,modes))
    half=channels+modes
    for i in range(modes):
        for a in range(channels):
            bra=np.array([a,channels+i],np.int64)
            ket=np.array([half+a,half+channels+i],np.int64)
            first[i]+=(abs(h[a,i])**2+h[a,i].conjugate()*_wick_moment(mean,covariance,ket)
                +h[a,i]*_wick_moment(mean,covariance,bra)
                +_wick_moment(mean,covariance,np.concatenate((bra,ket)))).real/norm
        for j in range(i+1):
            value=0.+0.j
            for a in range(channels):
                # d_i d_j [(c+h z)G]/G: four monomials, no numerical derivatives.
                coefficients=np.array([h[a,i],h[a,j],matrix[i,j],1.+0.j])
                indices=np.full((4,3),-1,np.int64)
                indices[0,0]=channels+j;indices[1,0]=channels+i
                indices[2,0]=a
                indices[3,0]=a;indices[3,1]=channels+i;indices[3,2]=channels+j
                counts=np.array([1,1,1,3])
                for left in range(4):
                    for right in range(4):
                        ids=np.empty(counts[left]+counts[right],np.int64)
                        ids[:counts[left]]=indices[left,:counts[left]]
                        ids[counts[left]:]=indices[right,:counts[right]]+half
                        value+=coefficients[left].conjugate()*coefficients[right]*_wick_moment(mean,covariance,ids)
            second[i,j]=value.real/norm;second[j,i]=second[i,j]
    for i in range(modes):
        second[i,i]+=first[i]
    return first,second-first[:,None]*first[None,:]


def gaussian_operator_number_moments(matrix,linear,log_ground,constants,insertions):
    """Exact number means/covariance of a linear-operator Gaussian bank.

    Number factorial moments are norms of analytic Bargmann derivatives.
    Wick contractions use only projected linear forms, avoiding a sixth-order
    tensor or finite differences. The positive bank norm normalizes moments.
    """
    m=np.asarray(matrix);r=np.asarray(linear);c=np.asarray(constants,complex);h=np.asarray(insertions,complex)
    if c.ndim!=1 or not len(c) or h.shape!=(len(c),len(r)) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(h)):
        raise ValueError('finite linear operator bank matching Gaussian source required')
    trace=gaussian_partial_number_trace(m,r,log_ground,np.arange(len(r)),np.ones(len(r)))
    nc=len(c);n=len(r)
    left=np.vstack([h,m]);projection=np.zeros((2*(nc+n),2*n),complex)
    projection[:nc+n,:n]=left.conj();projection[nc+n:,n:]=left
    means=np.r_[c.conj(),r.conj(),c,r]+projection@trace['mean']
    covariance=projection@trace['covariance']@projection.T
    norm=float(sum(means[a]*means[nc+n+a]+covariance[a,nc+n+a] for a in range(nc)).real)
    if not np.isfinite(norm) or norm<=0:
        raise ValueError('positive finite operator-bank norm required')
    mean,cov=_derivative_number_moments(means,covariance,h,m,nc,n,norm)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)) or np.min(np.diag(cov))<-1e-8:
        raise ArithmeticError('operator number moments violate positivity/finite precision')
    return dict(mean=mean,covariance=cov,bank_norm=norm*float(trace['constant'].real))


def gaussian_operator_number_marginal(matrix, linear, log_ground, constants,
        insertions, retained_indices, maximum_quanta, *, padding=8,
        alias_attenuation=1e-12, chunk_size=1024,
        maximum_workspace_bytes=256*1024**2):
    """Positive bank-norm marginal, including every signed interference term.

    Amplitudes are (c_a + L_a z) G(z); channels a are summed incoherently.
    Caps and FFT radial damping belong only to the ranking approximation.
    No spectator occupation cutoff or independent-mode factorization is used
    to acquire a block. Retained indices are returned in the requested order.
    """
    m=np.asarray(matrix);r=np.asarray(linear)
    c=np.asarray(constants,complex);h=np.asarray(insertions,complex)
    ids=np.asarray(retained_indices)
    caps=np.asarray(maximum_quanta)
    if (ids.ndim!=1 or len(ids) not in (1,2) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(r))
            or caps.shape!=ids.shape or caps.dtype.kind not in 'iu' or np.any(caps<0)
            or c.ndim!=1 or not len(c) or h.shape!=(len(c),len(r))
            or not np.all(np.isfinite(c)) or not np.all(np.isfinite(h))
            or isinstance(padding,bool) or not isinstance(padding,int) or padding<4
            or not 0<alias_attenuation<1 or not isinstance(chunk_size,int) or chunk_size<1
            or not isinstance(maximum_workspace_bytes,int) or maximum_workspace_bytes<1):
        raise ValueError('one/two unique source indices, integer caps and a finite linear bank required')
    bath=np.setdiff1d(np.arange(len(r)),ids)
    reduced=gaussian_partial_number_trace(m,r,log_ground,bath,np.ones(len(bath)))
    # The partial-trace owner sorts retained coordinates. Restore caller order.
    order=np.searchsorted(reduced['retained_indices'],ids)
    permutation=np.r_[order,len(ids)+order]
    a=reduced['matrix'][np.ix_(permutation,permutation)]
    b=reduced['linear'][permutation]
    response=reduced['response'][:,permutation]
    covariance=reduced['covariance'];mean=reduced['mean'];nb=len(bath);k=len(ids)
    left=np.zeros((len(c),1+2*k),complex);right=np.zeros_like(left)
    left[:,0]=c.conj()+h[:,bath].conj()@mean[:nb]
    right[:,0]=c+h[:,bath]@mean[nb:]
    left[:,1:1+k]=h[:,ids].conj();right[:,1+k:]=h[:,ids]
    left[:,1:]+=h[:,bath].conj()@covariance[:nb]@response
    right[:,1:]+=h[:,bath]@covariance[nb:]@response
    intrinsic=np.einsum('ai,ij,aj->',h[:,bath].conj(),covariance[:nb,nb:],h[:,bath])
    lengths=[next_fast_len(padding*(int(cap)+1)) if cap else 1 for cap in caps]
    points=int(np.prod(lengths))
    # Include FFT arrays, weight coordinates and a conservatively bounded chunk.
    fixed=16*points*(5+2*k)
    per_point=16*(12*(2*k)**2+8*len(c)*k+10*k)
    capacity=(maximum_workspace_bytes-fixed)//per_point
    if capacity<1:
        raise MemoryError('operator number marginal exceeds declared FFT workspace')
    chunk_size=min(chunk_size,capacity)
    logs=np.array([np.log(alias_attenuation)/length if cap else 0.
        for cap,length in zip(caps,lengths,strict=True)])
    circles=[np.exp(log-2j*np.pi*np.arange(length)/length) if cap else np.zeros(1,complex)
        for cap,length,log in zip(caps,lengths,logs,strict=True)]
    weights=np.stack(np.broadcast_arrays(*np.meshgrid(*circles,indexing='ij')),axis=-1).reshape(-1,k)
    signal=np.empty(points,complex);eye=np.eye(2*k)
    for start in range(0,points,chunk_size):
        z=weights[start:start+chunk_size];count=len(z)
        contraction=np.zeros((count,2*k,2*k),complex)
        for i in range(k):
            contraction[:,i,k+i]=z[:,i];contraction[:,k+i,i]=z[:,i]
        product=contraction@a
        eigen=np.linalg.eigvals(product)
        if np.any(abs(eigen)>1+1e-7) or np.min(abs(1-eigen))<1e-12:
            raise FloatingPointError('ill-conditioned operator marginal contraction')
        rhs=np.concatenate([(contraction@b)[:,:,None],
            contraction@right[:,1:].T],axis=2)
        solved=np.linalg.solve(eye[None]-product,rhs)
        moment=solved[:,:,0]
        polynomial=np.sum((left[:,0]+moment@left[:,1:].T)
            *(right[:,0]+moment@right[:,1:].T),axis=1)+intrinsic
        polynomial+=np.einsum('ai,tia->t',left[:,1:],solved[:,:,1:])
        signal[start:start+count]=reduced['constant']*np.exp(
            -.5*np.log1p(-eigen).sum(axis=1)+.5*moment@b)*polynomial
    coefficients=ifftn(signal.reshape(lengths))
    slices=tuple(slice(0,int(cap)+1) for cap in caps)
    coefficients=coefficients[slices]
    for axis,log in enumerate(logs):
        shape=[1]*k;shape[axis]=int(caps[axis])+1
        coefficients=coefficients*np.exp(-np.arange(int(caps[axis])+1)*log).reshape(shape)
    scale=max(float(np.max(abs(signal))),np.finfo(float).tiny)
    if np.max(abs(coefficients.imag))>1e-8*scale or np.min(coefficients.real)<-1e-8*scale:
        raise ArithmeticError('operator marginal loses real positive coefficient accuracy')
    return dict(strengths=np.maximum(coefficients.real,0.),
        diagnostics=dict(retained_indices=ids.tolist(),maximum_quanta=caps.tolist(),
            fft_lengths=lengths,alias_attenuation=alias_attenuation,
            spectator_trace_analytic=True,ranking_only=True,
            physical_transition_strengths_approximated=False))
