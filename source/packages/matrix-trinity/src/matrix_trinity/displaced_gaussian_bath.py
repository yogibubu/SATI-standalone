"""Gaussian bath optical kernels between coordinate-dependent translations.

The unchanged Bargmann quadratic form is factorized once per time, shared by
all translated centers and all Cartesian dipole components. The caller owns
the active-coordinate interpolation and its convergence. Oscillator number
states are never enumerated or cut off.
"""
import numpy as np
from .vibronic_harmonic import harmonic_gaussian_generator, harmonic_thermal_factors, PHASE_PER_CM_FS


class DisplacedGaussianBath:
    """Cross correlations of translated bath fibers on an oriented model.

    ``initial_displacements`` and ``final_displacements`` are dimensionless
    normal-coordinate shifts on their respective surfaces, in node order.
    Constants describe the optical operator at the unshifted initial origin;
    gradients refer to its dimensionless initial coordinates. ``base_scalar``
    must be the scalar correlation of the same oriented model/temperature,
    without its electronic or zero-point origin phase. Supplying the existing
    analytic trace preserves its determinant branch without Mehler caustics.
    """
    def __init__(self,model,initial_displacements,final_displacements,constants,
                 gradients,temperature_K,*,direction=-1):
        di=np.asarray(initial_displacements,float);df=np.asarray(final_displacements,float)
        c=np.asarray(constants,complex);h=np.asarray(gradients,complex)
        wi=model.initial_frequencies_cm;wf=model.final_frequencies_cm;n=len(wi)
        if (di.ndim!=2 or di.shape[1]!=n or df.shape!=di.shape
                or c.shape!=(len(di),3) or h.shape!=(3,n)
                or direction not in (-1,1) or not all(np.all(np.isfinite(x)) for x in (di,df,c,h))):
            raise ValueError('Matching translated centers and three-component optical operators required')
        g=harmonic_gaussian_generator(model)
        precision=np.diag(wi)+(model.mode_map.T*wf)@model.mode_map
        l=np.sqrt(2)*np.column_stack([np.diag(np.sqrt(wi)),model.mode_map.T*np.sqrt(wf)])
        response=np.linalg.solve(precision,l)
        transform=np.sqrt(wf)[:,None]*model.mode_map/np.sqrt(wi)[None,:]
        shifts=model.final_shift_dimensionless[None,:]+di@transform.T-df
        k=shifts/np.sqrt(wf)
        b=-(k*wf)@model.mode_map
        mean=np.linalg.solve(precision,b.T).T
        linear=mean@l+np.c_[np.zeros_like(k),np.sqrt(2*wf)*k]
        k0=model.final_shift_dimensionless/np.sqrt(wf)
        b0=-model.mode_map.T@(wf*k0);m0=np.linalg.solve(precision,b0)
        logdelta=-.5*np.sum(k*k*wf,axis=1)+.5*np.sum(b*mean,axis=1)+.5*k0@(wf*k0)-.5*b0@m0
        thermal=harmonic_thermal_factors(wi,temperature_K)
        prefactors=np.r_[thermal,np.ones(n)];active=np.flatnonzero(prefactors!=0)
        self.matrix=g['matrix'][np.ix_(active,active)]
        self.sources=linear[:,active];self.reference=g['linear'][active]
        self.logdelta=logdelta
        self.constants=c+di@h.T+(mean*np.sqrt(wi))@h.T
        self.insertions=(h@(np.sqrt(wi)[:,None]*response))[:,active]
        self.prefactors=prefactors[active]
        self.frequencies=np.r_[-wi,wf][active]
        self.direction=direction

    def evaluate(self,time_fs,base_scalar):
        """Return the node-by-node complete FC+HT cross kernel at one time."""
        z=self.prefactors*np.exp(-1j*self.direction*PHASE_PER_CM_FS*time_fs*self.frequencies)
        m=self.matrix;r=self.sources;h=self.insertions
        a=np.eye(len(z))-(z[:,None]*m.conj()*z[None,:])@m
        solved=np.linalg.solve(a,np.c_[z[:,None]*m.conj()*z[None,:],np.diag(z)])
        P,R=np.split(solved,2,axis=1)
        Q=np.diag(z)+z[:,None]*(m@P);S=z[:,None]*(m@R)
        left=.5*np.einsum('ni,ij,nj->n',r,P,r,optimize=True)
        right=.5*np.einsum('ni,ij,nj->n',r,S,r,optimize=True)
        exponent=left[:,None]+right[None,:]+.5*(r@R@r.T+r@Q.T@r.T)
        r0=self.reference
        baseline=.5*r0@(P+R+Q+S)@r0
        exponent+=self.logdelta[:,None]+self.logdelta[None,:]-baseline
        l0=self.constants+r@P.T@h.T;l1=r@R.T@h.T
        r0=self.constants.conj()+r@S.T@h.conj().T;r1=r@Q.T@h.conj().T
        optical=np.zeros(exponent.shape,complex)
        for j in range(3):
            optical+=(l0[:,j,None]+l1[None,:,j])*(r1[:,j,None]+r0[None,:,j])
        optical+=np.einsum('ai,ij,aj->',h,R,h.conj(),optimize=True)
        return base_scalar*np.exp(exponent)*optical
