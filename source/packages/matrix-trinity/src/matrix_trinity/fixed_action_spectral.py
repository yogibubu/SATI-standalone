"""Perturbation-subtracted spectral blocks in a saddle normal form.

The unstable coordinate remains a real action parameter. A block calculation
supplies E(I)=a+b I+c I**2 for its complete zero/one/two-quanta manifold.
Subtracting the same block's perturbative expansion preserves the original
normal form's virtual contributions involving modes outside that block.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from .hdc_polynomial import HDCPolynomialEnergy


def occupation_coefficients(labels, energies):
    """Interpolate a complete quadratic occupation manifold, including E0."""
    labels=np.asarray(labels);e=np.asarray(energies,float)
    if labels.ndim!=2 or labels.dtype.kind not in 'iu' or np.any(labels<0):
        raise ValueError('nonnegative integer occupation labels required')
    n=labels.shape[1];lookup={tuple(s):i for i,s in enumerate(labels)}
    required=set()
    for total in range(3):
        for modes in itertools.combinations_with_replacement(range(n),total):
            required.add(tuple(modes.count(j) for j in range(n)))
    if (not n or len(lookup)!=len(labels) or set(lookup)!=required
            or e.shape!=(len(labels),) or not np.all(np.isfinite(e))):
        raise ValueError('complete finite zero/one/two-quanta manifold required')
    zero=float(e[lookup[(0,)*n]]);nu=np.empty(n);x=np.zeros((n,n))
    for i in range(n):
        nu[i]=e[lookup[tuple(int(k==i) for k in range(n))]]-zero
        x[i,i]=(e[lookup[tuple(2*int(k==i) for k in range(n))]]-zero-2*nu[i])/2
    for i in range(n):
        for j in range(i):
            x[i,j]=x[j,i]=e[lookup[tuple(int(k in (i,j)) for k in range(n))]]-zero-nu[i]-nu[j]
    return zero,nu,x


@dataclass(frozen=True)
class FixedActionSpectralCorrection:
    """A local block resummation, with its own perturbative part subtracted.

    Rows of each coefficient array are the constant, I, and I**2 terms;
    columns follow labels. Ground energy is corrected exactly once. The
    caller's Weyl I**2-1/4 convention is already in the full normal form and
    is not applied a second time to these Taylor coefficients.
    """
    labels: np.ndarray
    variational_coefficients_cm: np.ndarray
    perturbative_coefficients_cm: np.ndarray

    def __post_init__(self):
        labels=np.array(self.labels,copy=True)
        for name in ('variational_coefficients_cm','perturbative_coefficients_cm'):
            c=np.array(getattr(self,name),float,copy=True)
            if c.shape!=(3,len(labels)) or not np.all(np.isfinite(c)):
                raise ValueError('three finite action coefficients per occupation required')
            for row in c:occupation_coefficients(labels,row)
            c.setflags(write=False);object.__setattr__(self,name,c)
        labels.setflags(write=False);object.__setattr__(self,'labels',labels)

    def apply(self, full_model, mode_indices, reactive_action):
        """Retain all external couplings and add only the block difference."""
        if not isinstance(full_model,HDCPolynomialEnergy) or full_model.y_tensor_cm is not None:
            raise TypeError('quadratic full occupation energy required')
        indices=np.asarray(mode_indices);action=float(reactive_action)
        if (indices.shape!=(self.labels.shape[1],) or indices.dtype.kind not in 'iu'
                or len(set(indices.tolist()))!=len(indices) or np.any(indices<0)
                or np.any(indices>=len(full_model.fundamentals_cm)) or not np.isfinite(action)):
            raise ValueError('distinct aligned stable modes and finite real action required')
        difference=np.array([1.,action,action*action])@(
            self.variational_coefficients_cm-self.perturbative_coefficients_cm)
        zero,nu,x=occupation_coefficients(self.labels,difference)
        full_nu=full_model.fundamentals_cm.copy();full_x=full_model.x_matrix_cm.copy()
        full_nu[indices]+=nu;full_x[np.ix_(indices,indices)]+=x
        return HDCPolynomialEnergy(full_model.zero_point_cm+zero,full_nu,full_x,
            coordinate_id=full_model.coordinate_id)


def perturbative_action_anchors(frequencies_cm, cubic_cm, quartic_cm, action_slopes_cm, labels,
                               *, denominator_tolerance_cm=1e-7):
    """Exact fixed-basis VPT2 Taylor coefficients of H0+I D+V3+V4.

    D=sum b_i(N_i+1/2). Every virtual cubic product is generated before
    restriction; excited-state resonances are rejected, never shifted.
    Ground denominators are negative positive-frequency sums.
    """
    from .curvilinear_vpt2 import _cubic_couplings, _quartic_expectation
    w=np.asarray(frequencies_cm,float);b=np.asarray(action_slopes_cm,float)
    f3=np.asarray(cubic_cm,float);f4=np.asarray(quartic_cm,float);labels=np.asarray(labels)
    occupation_coefficients(labels,np.zeros(len(labels)))
    n=len(w)
    if (w.shape!=(n,) or b.shape!=w.shape or n!=labels.shape[1] or np.any(w<=0)
            or f3.shape!=(n,)*3 or f4.shape!=(n,)*4
            or not all(np.all(np.isfinite(a)) for a in (w,b,f3,f4))
            or not np.isfinite(denominator_tolerance_cm) or denominator_tolerance_cm<=0):
        raise ValueError('positive stable frequencies and aligned finite action/force tensors required')
    g1=np.zeros((n,)*3);g2=np.zeros((n,)*4);out=np.zeros((3,len(labels)))
    for j,row in enumerate(labels):
        state=tuple(row);out[0,j]=w@(row+.5)+_quartic_expectation(state,f4,g2,0.,0.)
        out[1,j]=b@(row+.5)
        for target,coupling in _cubic_couplings(state,f3,g1,0.).items():
            if not coupling:continue
            dn=row-np.asarray(target);gap=float(w@dn);slope=float(b@dn)
            if abs(gap)<=denominator_tolerance_cm:
                raise ValueError('resonant perturbative block requires a retained spectral polyad')
            square=abs(coupling)**2
            out[0,j]+=square/gap;out[1,j]-=square*slope/gap**2
            out[2,j]+=square*slope**2/gap**3
    return out
