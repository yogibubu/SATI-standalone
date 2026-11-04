"""Two-state harmonic FC/HT model in native common ESALC/Wilson-GF coordinates.

All public dipole derivatives refer to initial dimensionless normal coordinates.
The ESALC producer obtains them from derivatives in the frozen common chart.
"""
from __future__ import annotations
from dataclasses import dataclass,field,replace
import hashlib
import numpy as np
from matrix_gf.harmonic import CM_PER_HARTREE

HCK_K_CM=1.4387768775039338
PHASE_PER_CM_FS=2*np.pi*2.99792458e-5


@dataclass(frozen=True)
class ESALCHarmonicState:
    force_constants_hartree: np.ndarray
    metric_inverse_amu: np.ndarray
    reference_esalc: np.ndarray
    coordinate_identity: str
    gradient_hartree: np.ndarray | None = None
    label: str = ''

    @classmethod
    def from_trinity(cls,field,*,reference_esalc,coordinate_identity,gradient_hartree=None,label=''):
        native=getattr(field,'field',field)
        return cls(native.harmonic_sonic,native.metric_sonic,np.asarray(reference_esalc,float),
                   coordinate_identity,gradient_hartree,label)

    def modes(self):
        from matrix_gf.harmonic import solve_wilson_gf,HESSIAN_EIGENVALUE_TO_CM
        f=np.asarray(self.force_constants_hartree,float);g=np.asarray(self.metric_inverse_amu,float)
        s=np.asarray(self.reference_esalc,float);n=len(s)
        gradient=np.zeros(n) if self.gradient_hartree is None else np.asarray(self.gradient_hartree,float)
        if (not n or f.shape!=(n,n) or g.shape!=f.shape or gradient.shape!=(n,)
                or not self.coordinate_identity or not all(np.all(np.isfinite(x)) for x in (f,g,s,gradient))
                or not np.allclose(f,f.T,rtol=1e-12,atol=1e-14)
                or not np.allclose(g,g.T,rtol=1e-12,atol=1e-14)):
            raise ValueError('finite symmetric F/G in an identified common nonredundant ESALC chart required')
        result=solve_wilson_gf(f,g,scale_to_cm=True)
        if np.any(result.frequencies_cm<=0):raise ValueError('harmonic vibronic states require stable positive-frequency modes; select LAM separately')
        eigen,vectors=np.linalg.eigh(g)
        if np.any(eigen<=0):raise ValueError('positive definite Wilson metric required')
        root=(vectors*np.sqrt(eigen))@vectors.T
        transform=root@result.normal_modes*(HESSIAN_EIGENVALUE_TO_CM/np.sqrt(CM_PER_HARTREE))
        shift=np.linalg.solve(f,gradient)
        return result.frequencies_cm,transform,s-shift,float(-.5*gradient@shift)


@dataclass(frozen=True)
class HarmonicVibronicModel:
    initial_frequencies_cm: np.ndarray
    final_frequencies_cm: np.ndarray
    mode_map: np.ndarray
    final_shift_dimensionless: np.ndarray
    electronic_gap_cm: float
    transition_dipole_au: np.ndarray
    transition_gradient_dimensionless_au: np.ndarray
    provenance: dict = field(default_factory=dict)
    transition_momentum_dimensionless_au: np.ndarray | None = None

    def __post_init__(self):
        if self.transition_momentum_dimensionless_au is None:
            object.__setattr__(self,'transition_momentum_dimensionless_au',np.zeros((3,len(self.initial_frequencies_cm)),complex))
        for name,dtype in [('initial_frequencies_cm',float),('final_frequencies_cm',float),('mode_map',float),
            ('final_shift_dimensionless',float),('transition_dipole_au',complex),('transition_gradient_dimensionless_au',complex),('transition_momentum_dimensionless_au',complex)]:
            a=np.array(getattr(self,name),dtype=dtype,copy=True);a.setflags(write=False);object.__setattr__(self,name,a)
        w=self.initial_frequencies_cm;n=len(w)
        if (not n or w.shape!=(n,) or self.final_frequencies_cm.shape!=(n,) or self.mode_map.shape!=(n,n)
                or self.final_shift_dimensionless.shape!=(n,) or self.transition_dipole_au.shape!=(3,)
                or self.transition_gradient_dimensionless_au.shape!=(3,n) or self.transition_momentum_dimensionless_au.shape!=(3,n)
                or not np.isfinite(self.electronic_gap_cm)
                or not all(np.all(np.isfinite(getattr(self,k))) for k in ('initial_frequencies_cm','final_frequencies_cm','mode_map','final_shift_dimensionless','transition_dipole_au','transition_gradient_dimensionless_au','transition_momentum_dimensionless_au'))
                or np.any(w<=0) or np.any(self.final_frequencies_cm<=0)
                or np.linalg.cond(self.mode_map)>1e10):
            raise ValueError('compatible stable two-state harmonic data and an invertible coordinate map required')

    @property
    def origin_cm(self):return float(self.electronic_gap_cm+.5*(self.final_frequencies_cm.sum()-self.initial_frequencies_cm.sum()))

    def emission_model(self):
        jinv=np.linalg.solve(self.mode_map,np.eye(len(self.mode_map)))
        k=self.final_shift_dimensionless/np.sqrt(self.final_frequencies_cm)
        reverse_shift=-np.sqrt(self.initial_frequencies_cm)*(jinv@k)
        qmap=np.sqrt(self.initial_frequencies_cm)[:,None]*jinv/np.sqrt(self.final_frequencies_cm)[None,:]
        mu=(self.transition_dipole_au-self.transition_gradient_dimensionless_au@(np.sqrt(self.initial_frequencies_cm)*(jinv@k))).conj()
        return HarmonicVibronicModel(self.final_frequencies_cm,self.initial_frequencies_cm,jinv,
            reverse_shift,-self.electronic_gap_cm,mu,(self.transition_gradient_dimensionless_au@qmap).conj(),
            {**self.provenance,'orientation':'emission'},
            self.transition_momentum_dimensionless_au.conj()@(np.sqrt(self.final_frequencies_cm)[:,None]*self.mode_map/np.sqrt(self.initial_frequencies_cm)[None,:]).T)


def _prepare_pes_model(initial,final,pes_model):
    """Construct explicit harmonic approximations in the common ESALC chart.

    AS/VG copy the initial F and G. ASF/VGF retain only the diagonal of
    the supplied final F in initial mass-weighted normal coordinates.
    Their metric is frozen to the initial G, so the modes remain parallel.
    """
    names={'local','AH','AS','ASF','VH','VG','VGF','gVH'}
    if pes_model not in names:raise ValueError(f'pes_model must be one of {sorted(names)}')
    if initial.coordinate_identity!=final.coordinate_identity:
        raise ValueError('two states must use the same frozen ESALC coordinate identity; transport the fields first')
    if pes_model in ('local','gVH'):return initial,final
    wi,ti,si,_=initial.modes()
    n=len(wi)
    def stationary(state):
        return state.gradient_hartree is None or np.all(np.abs(np.asarray(state.gradient_hartree,float))<=1e-10)
    if not stationary(initial):raise ValueError(f'{pes_model} requires an initial minimum; use gVH for nonstationary initial fields')
    if pes_model in ('AH','AS','ASF'):
        if not stationary(final):raise ValueError(f'{pes_model} requires final-minimum data, not a vertical gradient')
    else:
        if final.gradient_hartree is None:raise ValueError(f'{pes_model} requires the explicit final gradient at the initial minimum')
        if not np.allclose(final.reference_esalc,si,rtol=0,atol=1e-10):
            raise ValueError(f'{pes_model} requires both expansion points at the initial minimum')
    if pes_model=='AH':return initial,final
    if pes_model=='VH':
        if final.metric_inverse_amu is not None and not np.allclose(final.metric_inverse_amu,initial.metric_inverse_amu,rtol=1e-10,atol=1e-12):
            raise ValueError('VH fields at the same geometry require the same Wilson metric')
        return initial,replace(final,metric_inverse_amu=initial.metric_inverse_amu)
    if pes_model in ('AS','VG'):
        force=initial.force_constants_hartree
    else:
        force=np.asarray(final.force_constants_hartree,float)
        if force.shape!=(n,n) or not np.all(np.isfinite(force)) or not np.allclose(force,force.T,rtol=1e-12,atol=1e-14):
            raise ValueError(f'{pes_model} requires a finite symmetric final Hessian')
        start=0
        while start<n:
            stop=start+1
            while stop<n and abs(wi[stop]-wi[start])<=64*np.finfo(float).eps*max(wi[stop],wi[start]):stop+=1
            if stop-start>1:
                block=ti[:,start:stop]
                _,rotation=np.linalg.eigh(block.T@force@block)
                ti[:,start:stop]=block@rotation
            start=stop
        diagonal=np.diag(ti.T@force@ti)
        if np.any(diagonal<=0):raise ValueError(f'{pes_model} has nonpositive projected curvature; separate the LAM explicitly')
        inverse=np.linalg.solve(ti,np.eye(n))
        force=(inverse.T*diagonal)@inverse
    return initial,replace(final,force_constants_hartree=force,metric_inverse_amu=initial.metric_inverse_amu)


def harmonic_vibronic_from_esalc(initial,final,*,electronic_gap_cm,transition_dipole_au,
        transition_gradient_esalc_au=None,transition_reference_esalc=None,pes_model='local',reference_state='initial'):
    """Use native GF at each state; retain both Wilson metrics, not a polar fit.

    The gap is the final-minus-initial electronic energy at the supplied
    expansion points. Nonzero gradients supply the quadratic relaxation shift.
    Transition properties refer to transition_reference_esalc, defaulting to
    the initial expansion point; their linear expansion is transported to
    the relaxed initial minimum.
    """
    if reference_state not in ('initial','final'):raise ValueError('reference_state must be initial or final')
    if reference_state=='initial':initial,final=_prepare_pes_model(initial,final,pes_model)
    else:final,initial=_prepare_pes_model(final,initial,pes_model)
    wi,ti,si,ei=initial.modes();wf,tf,sf,ef=final.modes();n=len(wi)
    if len(wf)!=n:raise ValueError('the two ESALC spaces must have equal dimension')
    derivatives=np.zeros((3,n)) if transition_gradient_esalc_au is None else np.asarray(transition_gradient_esalc_au,complex)
    if derivatives.shape!=(3,n):raise ValueError('electronic transition-dipole derivatives must have shape (3,ESALC dimension)')
    reference=np.asarray(initial.reference_esalc if transition_reference_esalc is None else transition_reference_esalc,float)
    if reference.shape!=(n,) or not np.all(np.isfinite(reference)):raise ValueError("finite transition-property ESALC reference required")
    mu=np.asarray(transition_dipole_au,complex)+derivatives@(si-reference)
    j=np.linalg.solve(tf,ti);k=np.linalg.solve(tf,si-sf)
    return HarmonicVibronicModel(wi,wf,j,np.sqrt(wf)*k,
        float(electronic_gap_cm+(ef-ei)*CM_PER_HARTREE),mu,
        derivatives@ti/np.sqrt(wi)[None,:],
        dict(initial_normal_basis_sha256=hashlib.sha256(initial.coordinate_identity.encode()+b''.join(np.asarray(a,dtype='<f8').tobytes() for a in (wi,ti,si))).hexdigest(),
             final_normal_basis_sha256=hashlib.sha256(final.coordinate_identity.encode()+b''.join(np.asarray(a,dtype='<f8').tobytes() for a in (wf,tf,sf))).hexdigest(),
             pes_model=pes_model,reference_state=reference_state,coordinate_owner='SMITH/ESALC',harmonic_owner='Wilson GF',coordinate_identity=initial.coordinate_identity,
             initial_label=initial.label,final_label=final.label,metrics='state-specific local Wilson metrics',
             coordinate_map_orthogonality_error=float(np.linalg.norm(j.T@j-np.eye(n))),
             scope='local harmonic potentials and constant local metrics in a common frozen ESALC chart'))


def orient_harmonic_model(model,kind):
    if kind=='absorption':return model,1.
    if kind=='emission':return model.emission_model(),-1.
    raise ValueError('kind must be absorption or emission')


def harmonic_thermal_factors(frequencies_cm,temperature_K):
    if not np.isfinite(temperature_K) or temperature_K<0:raise ValueError('finite nonnegative temperature required')
    q=np.zeros(len(frequencies_cm)) if temperature_K==0 else np.exp(-HCK_K_CM*np.asarray(frequencies_cm)/temperature_K)
    if np.any(q>=1):raise FloatingPointError('thermal partition function is unresolved at this precision')
    return q


def harmonic_strength_sum_rule(model,temperature_K=0.,*,kind='absorption',operator='FCHT'):
    model,_=orient_harmonic_model(model,kind);q=harmonic_thermal_factors(model.initial_frequencies_cm,temperature_K)
    fc=float(np.vdot(model.transition_dipole_au,model.transition_dipole_au).real)
    position=model.transition_gradient_dimensionless_au;momentum=model.transition_momentum_dimensionless_au
    ht=float(np.sum((abs(position-1j*momentum)**2*q[None,:]
        +abs(position+1j*momentum)**2)/(2*(1-q))[None,:]))
    if operator not in ('FC','HT','FCHT'):raise ValueError('operator must be FC, HT or FCHT')
    return dict(FC=fc,HT=ht,interference=0.,total=fc if operator=='FC' else ht if operator=='HT' else fc+ht)


def harmonic_gaussian_generator(model):
    """Normalized Bargmann/Hermite generating function and linear HT insertion."""
    wi=model.initial_frequencies_cm;wf=model.final_frequencies_cm;j=model.mode_map
    k=model.final_shift_dimensionless/np.sqrt(wf);n=len(wi)
    a=np.diag(wi)+(j.T*wf)@j
    l=np.sqrt(2)*np.column_stack([np.diag(np.sqrt(wi)),j.T*np.sqrt(wf)])
    b=-j.T@(wf*k);solved=np.linalg.solve(a,np.column_stack([b,l]));mean=solved[:,0];response=solved[:,1:]
    r=l.T@mean+np.r_[np.zeros(n),np.sqrt(2*wf)*k]
    matrix=l.T@response-np.eye(2*n)
    sign,logdet=np.linalg.slogdet(a);jsign,jlogdet=np.linalg.slogdet(j)
    if sign<=0 or jsign==0:raise ValueError('invalid harmonic Gaussian precision or coordinate Jacobian')
    log_ground=(n*np.log(2)/2+.25*np.log(wi*wf).sum()+.5*jlogdet-.5*logdet
                -.5*k@(wf*k)+.5*b@mean)
    # p=-i d/dx acting on the initial harmonic generating function.
    momentum=model.transition_momentum_dimensionless_au
    insertion=model.transition_gradient_dimensionless_au+1j*momentum
    ht0=insertion@(np.sqrt(wi)*mean)
    htlinear=insertion@(np.sqrt(wi)[:,None]*response)
    htlinear[:,:n]-=1j*np.sqrt(2)*momentum
    return dict(matrix=(matrix+matrix.T)/2,linear=r,log_ground=float(log_ground),ht_constant=ht0,ht_linear=htlinear)


__all__=['ESALCHarmonicState','HarmonicVibronicModel','harmonic_vibronic_from_esalc','harmonic_strength_sum_rule']
