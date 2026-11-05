"""Physical overlap transfers between centered sine DVR and oscillator bases.

Coordinates must use the same units and common reference. A change of basis
is performed here, never a change of nuclear coordinates or a Duschinsky map.
"""
from math import prod
import numpy as np
from scipy.fft import dst
from .sine_galerkin import sine_product_quadrature


def _coefficients(vectors, width):
    u = np.asarray(vectors)
    if (u.ndim != 2 or not min(u.shape) or not np.all(np.isfinite(u))
            or not np.isscalar(width) or not np.isreal(width) or not np.isfinite(width) or width <= 0):
        raise ValueError('finite sine-DVR coefficient columns and positive physical width required')
    return dst(u, type=1, norm='ortho', axis=0)


def sine_dvr_modal_overlap(initial_vectors, final_vectors, *, initial_width, final_width,
        initial_center=0., final_center=0., initial_momentum=False, final_momentum=False,
        coordinate_power=0, coordinate_origin=0., quadrature_count=None):
    """Physical sine overlaps and weak momentum/coordinate insertions.

    Rows are final bra states and columns initial ket states. A requested
    momentum acts on its primitive sine function before modal truncation;
    the final action is conjugated in the weak product. Coordinates and
    physical boxes share the caller's units. Restricted boxes are never
    renormalized. Defaults retain the original analytic centered overlap.
    """
    left = _coefficients(initial_vectors, initial_width); right = _coefficients(final_vectors, final_width)
    if (not all(np.isscalar(x) and np.isreal(x) and np.isfinite(x)
                for x in (initial_center,final_center,coordinate_origin))
            or not isinstance(initial_momentum,(bool,np.bool_))
            or not isinstance(final_momentum,(bool,np.bool_))
            or isinstance(coordinate_power,(bool,np.bool_))
            or not isinstance(coordinate_power,(int,np.integer)) or not 0<=coordinate_power<=2):
        raise ValueError('Finite physical centers and weak insertions through coordinate degree two required')
    if initial_center!=0 or final_center!=0 or initial_momentum or final_momentum or coordinate_power:
        lo=max(initial_center-initial_width/2,final_center-final_width/2)
        hi=min(initial_center+initial_width/2,final_center+final_width/2)
        if hi<=lo:
            return np.zeros((right.shape[1],left.shape[1]),
                dtype=np.result_type(left,right,complex if initial_momentum or final_momentum else float))
        count=max(80,4*max(len(left),len(right))) if quadrature_count is None else quadrature_count
        if (isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer))
                or not max(len(left),len(right))<count<=4096):
            raise ValueError('Resolved bounded physical sine quadrature required')
        initial=sine_product_quadrature(
            [[initial_center-initial_width/2,initial_center+initial_width/2]],
            [len(left)],[count],quadrature_bounds=[[lo,hi]],orthonormalize=False)
        final=sine_product_quadrature(
            [[final_center-final_width/2,final_center+final_width/2]],
            [len(right)],[count],quadrature_bounds=[[lo,hi]],orthonormalize=False)
        ket=(initial['momenta'][:,0,:] if initial_momentum else initial['values'])@left
        bra=(final['momenta'][:,0,:] if final_momentum else final['values'])@right
        profile=(initial['points'][:,0]-coordinate_origin)**coordinate_power
        return bra.conj().T@(profile[:,None]*ket)
    if initial_width == final_width:
        primitive = np.eye(len(right), len(left))
    else:
        n = np.arange(1, len(right)+1)[:, None]; m = np.arange(1, len(left)+1)[None, :]
        width = min(initial_width, final_width)
        def phase(k): return np.where(k % 4 == 0, 1., np.where(k % 4 == 2, -1., 0.))
        primitive = width/np.sqrt(initial_width*final_width)*(
            phase(n-m)*np.sinc(.5*width*(n/final_width-m/initial_width))
            -phase(n+m)*np.sinc(.5*width*(n/final_width+m/initial_width)))
    return right.conj().T@primitive@left


def harmonic_sine_dvr_modal_overlap(initial_vectors, *, width, maximum_degree,
                                    oscillator_length=1., quadrature_count=256):
    """Oscillator rows versus sine-DVR modal columns on the sine box.

    The oscillator is normalized on the full real line; its finite-box overlap
    is not renormalized. Gauss nodes come from the resident sine owner. Vary
    quadrature count independently when transferring very small amplitudes.
    """
    coefficients = _coefficients(initial_vectors, width)
    if (not isinstance(maximum_degree, int) or maximum_degree < 0
            or not isinstance(quadrature_count, int) or not len(coefficients) < quadrature_count <= 4096
            or not np.isreal(oscillator_length) or not np.isfinite(oscillator_length) or oscillator_length <= 0):
        raise ValueError('bounded oscillator degree, resolved quadrature and positive length required')
    rule = sine_product_quadrature([[-width/2, width/2]], [len(coefficients)], [quadrature_count])
    x = rule['points'][:, 0]; weights = rule['weights']; q = x/oscillator_length
    phi = np.empty((maximum_degree+1, len(q)))
    phi[0] = np.pi**(-.25)*np.exp(-.5*q*q)/np.sqrt(oscillator_length)
    if maximum_degree: phi[1] = np.sqrt(2)*q*phi[0]
    for n in range(2, len(phi)):
        phi[n] = np.sqrt(2/n)*q*phi[n-1]-np.sqrt((n-1)/n)*phi[n-2]
    sine = np.sin((x[:, None]+width/2)*np.arange(1, len(coefficients)+1)[None, :]*np.pi/width)*np.sqrt(2/width)
    return phi@(weights[:, None]*(sine@coefficients))


def tensor_sine_dvr_harmonic_projection(vectors, *, basis_counts, widths,
        maximum_degrees, oscillator_lengths, quadrature_counts,
        maximum_workspace_bytes=1024**3):
    """Project flat-measure product sine-DVR columns onto reference HO products.

    The centered physical coordinates, widths and oscillator lengths must use
    the same units. A variable kinetic metric does not prevent this Hilbert
    space projection: its physical-volume factor must already be included in
    the supplied flat-measure wavefunctions. No norm is restored after projection.
    Returned tensor axes label HO occupations, followed by the state column.
    """
    counts=tuple(basis_counts);degrees=tuple(maximum_degrees)
    width=np.asarray(widths,float);length=np.asarray(oscillator_lengths,float)
    nq=tuple(quadrature_counts);u=np.asarray(vectors)
    d=len(counts)
    if (not d or len(degrees)!=d or len(nq)!=d or width.shape!=(d,) or length.shape!=(d,)
            or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<1 for n in counts)
            or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<0 for n in degrees)
            or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer))
                   or not counts[i]<n<=4096 for i,n in enumerate(nq))
            or np.any(width<=0) or np.any(length<=0) or not np.all(np.isfinite(width))
            or not np.all(np.isfinite(length)) or u.ndim!=2 or len(u)!=prod(counts)
            or not u.shape[1] or not np.all(np.isfinite(u))):
        raise ValueError('finite product sine-DVR states and compatible centered HO reference required')
    if (isinstance(maximum_workspace_bytes,bool) or not isinstance(maximum_workspace_bytes,int)
            or maximum_workspace_bytes<1):
        raise ValueError('positive numeric workspace budget required')
    shape=list(counts);itemsize=np.result_type(u.dtype,float).itemsize
    peak=u.nbytes
    for axis,degree in enumerate(degrees):
        before=prod(shape)*u.shape[1]*itemsize
        shape[axis]=degree+1
        after=prod(shape)*u.shape[1]*itemsize
        peak=max(peak,u.nbytes+before+after+8*((degree+1)*(counts[axis]+nq[axis])
            +7*counts[axis]**2+20*nq[axis]*counts[axis]))
    if peak>maximum_workspace_bytes:
        raise MemoryError('tensor HO projection exceeds declared numeric workspace')
    coefficients=u.reshape(counts+(u.shape[1],))
    for axis in range(d):
        overlap=harmonic_sine_dvr_modal_overlap(np.eye(counts[axis]),width=float(width[axis]),
            maximum_degree=degrees[axis],oscillator_length=float(length[axis]),quadrature_count=nq[axis])
        coefficients=np.moveaxis(np.tensordot(overlap,coefficients,axes=(1,axis)),0,axis)
    gram=coefficients.reshape(-1,u.shape[1]).conj().T@coefficients.reshape(-1,u.shape[1])
    return dict(coefficients=coefficients,diagnostics=dict(
        input_norms=np.diag(u.conj().T@u).real.tolist(),retained_norms=np.diag(gram).real.tolist(),
        maximum_gram_defect=float(abs(gram-u.conj().T@u).max()),
        oscillator_lengths=length.tolist(),maximum_degrees=list(degrees),
        quadrature_counts=list(nq),normalized=False,orthogonalized=False,
        physical_volume_factor_added=False,numeric_workspace_estimate_bytes=peak))
