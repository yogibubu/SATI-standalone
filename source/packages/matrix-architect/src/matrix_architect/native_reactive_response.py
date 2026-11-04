"""Native radial/angular/periodic response functions with valence support.

This basis represents the functional dependence of a transverse quadratic
kernel. A complete scalar PES still needs consistent longitudinal terms and
higher transverse derivatives; this owner does not declare it qualified.
"""
import numpy as np
from matrix_chem import BOHR_TO_ANGSTROM
from .asymptotic_pic_hessian import NativeCorrectionHessianDesign


class NativeReactiveResponseBasis:
    """Shared native functions, periodic torsions and optional valence windows.

    support(index, xyz) may supply a smooth occupancy window for a primitive;
    it returns value, gradient_per_bohr and hessian_per_bohr2. The window is
    part of the functional form and is differentiated before fitting. No
    source potential is called. Features vanish at the supplied reference.
    """
    def __init__(self, parent, reference_xyz, *, harmonics=(1,2), angle_quadratic=True,
                 support=None):
        if (not harmonics or len(set(harmonics))!=len(harmonics)
                or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in harmonics)
                or (support is not None and not callable(support))):
            raise ValueError('distinct positive Fourier harmonics and a callable support required')
        self.parent,self.support=parent,support
        self.reference_xyz=np.asarray(reference_xyz,float).copy()
        self.atoms=parent.atoms;self.terms=[];self.indices=[];self.labels=[]
        for i,p in enumerate(parent.primitives):
            prefix=f'{p.kind}:{p.atoms}'
            if p.kind in ('dihedral','out_of_plane'):
                functions=[(form,1,n) for n in harmonics for form in ('sin','one_minus_cos')]
            else:
                functions=[('native',1,1)]
                if angle_quadratic and p.kind=='angle':functions.append(('native',2,1))
            for form,power,harmonic in functions:
                self.terms.append(((i,form,power,harmonic),));self.indices.append(i)
                self.labels.append(f'{prefix}:{form}^{power}:n={harmonic}')
        self.kernel=NativeCorrectionHessianDesign(parent.primitives,parent.field,self.terms)
        for i in self.kernel.alpha:
            if self.kernel.radial_parameters[i].form!='inverse_power_stretch':
                raise ValueError('native reactive response requires inverse-power radial functions')
            self.kernel.alpha[i]=0.
        self.reference_values=self._raw(self.reference_xyz)[0]

    def _raw(self,xyz):
        xyz=np.asarray(xyz,float)
        if xyz.shape!=self.reference_xyz.shape or xyz.ndim!=2 or xyz.shape[1]!=3 or not np.all(np.isfinite(xyz)):
            raise ValueError('finite Cartesian geometry in reference atom order required')
        e,g,h=self.kernel.value(xyz);g*=BOHR_TO_ANGSTROM;h*=BOHR_TO_ANGSTROM**2
        if self.support is not None:
            windows={i:self.support(i,xyz) for i in set(self.indices)}
            for column,i in enumerate(self.indices):
                w=windows[i];v=float(w.value);wg=np.asarray(w.gradient_per_bohr);wh=np.asarray(w.hessian_per_bohr2)
                if (not np.isfinite(v) or not 0<=v<=1 or wg.shape!=(xyz.size,) or wh.shape!=(xyz.size,xyz.size)
                        or not np.all(np.isfinite(wg)) or not np.all(np.isfinite(wh))):
                    raise ValueError('finite primitive occupancy two-jet in [0,1] required')
                value=e[column];gradient=g[:,column].copy();hessian=h[:,:,column].copy()
                e[column]=v*value;g[:,column]=v*gradient+value*wg
                h[:,:,column]=v*hessian+value*wh+np.outer(wg,gradient)+np.outer(gradient,wg)
        return e,g,h

    def design(self,xyz):
        e,g,h=self._raw(xyz)
        return e-self.reference_values,g,h


def harmonic_response_scales(first,second):
    """RMS native functions through quadratic order for <q_i q_j>=delta_ij/2.

    This is a declared regularization scale, not an extra PES observation or
    an estimate of anharmonic accuracy. Nonzero quadratic even functions retain
    a scale when their first derivative vanishes at the reference.
    """
    g,h=np.asarray(first,float),np.asarray(second,float)
    if (g.ndim!=2 or h.shape!=(g.shape[0],g.shape[0],g.shape[1])
            or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h))):
        raise ValueError('aligned dimensionless-normal first and second function jets required')
    trace=np.einsum('iit->t',h)
    return np.sqrt(.5*np.sum(g*g,axis=0)+.125*np.sum(h*h,axis=(0,1))+.0625*trace*trace)
