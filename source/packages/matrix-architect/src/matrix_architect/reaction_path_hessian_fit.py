"""Fit native-functional IRC Hessian response within a practical tolerance.

Transported Hessians determine a transverse quadratic kernel. Differentiating
its native functions supplies reaction-index cubics and quartics; purely
transverse derivatives are independent observations.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class NativeHessianResponseFit:
    reference_hessian: np.ndarray
    coefficients: np.ndarray
    report: dict

    def evaluate(self, native_design, derivative=0):
        if derivative not in (0,1,2):raise ValueError('derivative order 0..2 required')
        a=np.asarray(native_design,float)
        if a.shape[-1]!=len(self.coefficients) or not np.all(np.isfinite(a)):
            raise ValueError('finite native design aligned with response coefficients required')
        value=np.einsum('...k,kij->...ij',a,self.coefficients)
        return value+self.reference_hessian if derivative==0 else value


@dataclass(frozen=True)
class PositiveNativeHessianResponseFit:
    reference_half: np.ndarray
    coefficients: np.ndarray
    report: dict

    def evaluate(self,native_design):
        a=np.asarray(native_design,float)
        if a.shape!=(len(self.coefficients),) or not np.all(np.isfinite(a)):
            raise ValueError('one finite native response vector required')
        exponent=np.einsum('k,kij->ij',a,self.coefficients)
        value=self.reference_half@expm(exponent)@self.reference_half
        return (value+value.T)/2


def fit_positive_native_hessian_response(design,hessians,*,reference_index,tolerance,basis_labels,hessian_unit):
    """Fit the logarithm of an SPD kernel in its reference-whitened frame.

    The largest declared ridge satisfying the physical Hessian tolerance is
    retained. All observed kernels must already be positive; no eigenvalue is
    clipped or repaired. The matrix exponential preserves positive curvature
    between observations. Full scalar/spectral qualification remains separate.
    """
    a,h=np.asarray(design,float),np.asarray(hessians,float)
    if (a.ndim!=2 or h.ndim!=3 or len(a)!=len(h) or h.shape[1]!=h.shape[2]
            or not 0<=reference_index<len(h) or len(basis_labels)!=a.shape[1]
            or not all(np.all(np.isfinite(v)) for v in (a,h))
            or not np.allclose(h,h.transpose(0,2,1),atol=1e-12,rtol=1e-10)
            or not np.allclose(a[reference_index],0.,atol=1e-12)
            or not np.isfinite(tolerance) or tolerance<=0 or not hessian_unit):
        raise ValueError('native design zero at reference, symmetric common-chart Hessians and tolerance required')
    w,v=np.linalg.eigh(h[reference_index])
    if w.min()<=0:raise ValueError('positive reference kernel required for native logarithmic response')
    half=(v*np.sqrt(w))@v.T;inverse=(v/np.sqrt(w))@v.T;logs=[]
    for observed in h:
        e,u=np.linalg.eigh(inverse@observed@inverse)
        if e.min()<=0:raise ValueError('positive acquired kernels required; no curvature clipping')
        logs.append((u*np.log(e))@u.T)
    logs=np.array(logs);logs[reference_index]=0.
    scales=np.linalg.norm(a,axis=0)
    if np.any(scales<=0):raise ValueError('unobserved native response column')
    x=a/scales
    u,s,vt=np.linalg.svd(x,full_matrices=False);y=logs.reshape(len(a),-1)
    chosen=None;history=[]
    for ridge in np.r_[0.,np.logspace(-14,4,73)]:
        factors=np.divide(s,s*s+ridge,out=np.zeros_like(s),where=s*s+ridge>0)
        if ridge==0:factors[s<s.max()*1e-12]=0.
        coefficients=(((vt.T*factors)@(u.T@y))/scales[:,None]).reshape((a.shape[1],)+h.shape[1:])
        predicted=np.array([half@expm(np.einsum('k,kij->ij',row,coefficients))@half for row in a])
        residual=float(np.max(abs(predicted-h)));passed=residual<=tolerance
        history.append(dict(ridge=float(ridge),maximum_hessian_residual=residual,passed=passed))
        if passed:chosen=(coefficients.copy(),float(ridge),residual)
    if chosen is None:raise ValueError('positive native response does not meet the physical Hessian tolerance')
    coefficients,ridge,residual=chosen
    report=dict(schema='matrix.architect.positive_native_hessian_response.v1',
        sample_count=len(a),reference_index=int(reference_index),hessian_unit=hessian_unit,
        basis_labels=list(basis_labels),tolerance=float(tolerance),response_singular_values=s.tolist(),
        normalization_scales=scales.tolist(),
        ridge=ridge,maximum_hessian_residual=residual,history=history,
        functional_form='H0**(1/2) exp(sum f_k A_k) H0**(1/2)',
        curvature_clipping=False,complete_global_potential_qualified=False)
    return PositiveNativeHessianResponseFit(half,coefficients,report)


def fit_native_hessian_response(design, hessians, *, reference_index, tolerance,
                                basis_labels, hessian_unit, ridge_parameters=None, normalization_scales=None):
    """Fit one shared native-functional response to transported IRC Hessians.

    The native basis is zero at the TS. Largest declared ridge regularization
    satisfying the observation tolerance is selected. No exact interpolation or
    minimum cubic/quartic field is required. The caller evaluates the same
    native functions and their derivatives for subsequent path derivatives.
    """
    a,h=map(lambda v:np.asarray(v,float),(design,hessians))
    if (a.ndim!=2 or h.ndim!=3 or len(a)!=len(h) or h.shape[1]!=h.shape[2]
            or not 0<=reference_index<len(h) or len(basis_labels)!=a.shape[1]
            or not all(np.all(np.isfinite(v)) for v in (a,h))
            or not np.allclose(h,h.transpose(0,2,1),atol=1e-12,rtol=1e-10)
            or not np.allclose(a[reference_index],0.,atol=1e-12)
            or not np.isfinite(tolerance) or tolerance<=0 or not hessian_unit):
        raise ValueError('native design zero at reference, symmetric common-chart Hessians and tolerance required')
    scales=np.linalg.norm(a,axis=0) if normalization_scales is None else np.asarray(normalization_scales,float)
    if scales.shape!=(a.shape[1],) or not np.all(np.isfinite(scales)):
        raise ValueError('one finite normalization scale per native response function required')
    if np.any(scales<=0):raise ValueError('unobserved native response column')
    x=a/scales;y=(h-h[reference_index]).reshape(len(h),-1)
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    parameters=np.r_[0.,np.logspace(-14,4,73)] if ridge_parameters is None else np.asarray(ridge_parameters,float)
    if parameters.ndim!=1 or np.any(parameters<0) or np.any(np.diff(parameters)<=0) or not np.all(np.isfinite(parameters)):
        raise ValueError('ordered finite nonnegative ridge parameters required')
    history=[];chosen=None
    for penalty in parameters:
        inverse=np.divide(s,s*s+penalty,out=np.zeros_like(s),where=s*s+penalty>0)
        if penalty==0:inverse[s<s.max()*1e-12]=0.
        c=(vt.T*inverse)@(u.T@y)
        error=float(np.max(abs(x@c-y)));passed=error<=tolerance
        history.append(dict(ridge=float(penalty),maximum_hessian_residual=error,passed=passed))
        if passed:chosen=(float(penalty),c.copy(),error)
    if chosen is None:raise ValueError('native functional basis does not meet the declared Hessian tolerance')
    penalty,c,error=chosen
    coefficients=(c/scales[:,None]).reshape((a.shape[1],)+h.shape[1:])
    report=dict(schema='matrix.architect.native_hessian_response.v1',sample_count=len(a),
        reference_index=int(reference_index),hessian_unit=hessian_unit,basis_labels=list(basis_labels),
        tolerance=float(tolerance),maximum_hessian_residual=error,ridge=penalty,
        response_singular_values=s.tolist(),history=history,minimum_cubic_quartic_used=False,
        normalization='observation_column_norm' if normalization_scales is None else 'supplied_physical_function_scale',
        normalization_scales=scales.tolist(),
        scope='native-functional transverse quadratic kernel; reactive-index derivatives from the same functions',
        complete_global_potential_qualified=False)
    return NativeHessianResponseFit(h[reference_index].copy(),coefficients,report)


def transverse_kernel_spectrum(hessian, reference_frequencies):
    """Spectrum in the declared TS-scaled transverse chart, rejecting negatives.

    The kinetic metric is diag(reference_frequencies). A nonpositive squared
    frequency returns no spectrum: zero clipping would hide a failed model.
    This diagnostic is not an anharmonic molecular calculation.
    """
    h,w=np.asarray(hessian,float),np.asarray(reference_frequencies,float)
    if (w.ndim!=1 or h.shape!=(len(w),len(w)) or not np.all(np.isfinite(h))
            or not np.all(np.isfinite(w)) or np.any(w<=0)
            or not np.allclose(h,h.T,rtol=1e-10,atol=1e-12)):
        raise ValueError('symmetric finite transverse kernel and positive reference frequencies required')
    root=np.sqrt(w)
    eigenvalues=np.linalg.eigvalsh(root[:,None]*h*root[None,:])
    minimum=float(np.min(eigenvalues))
    return (None if minimum<=0 else np.sqrt(eigenvalues)),minimum


@dataclass(frozen=True)
class PositiveNativeHessianBlend:
    """C4 convex blend of whitened log kernels in one bounded native variable.

    A shared scalar switch preserves covariance under a transverse orthogonal
    change of frame. No componentwise slope limiter or polynomial extrapolation
    is used. Outside the observed interval the log kernel is held fixed; a
    scalar parent composition must still impose its dissociation limits.
    """
    positions: np.ndarray
    logarithms: np.ndarray
    reference_half: np.ndarray
    report: dict

    def evaluate_position(self,position):
        if not np.isfinite(position):raise ValueError('finite native progress required')
        i=int(np.clip(np.searchsorted(self.positions,position)-1,0,len(self.positions)-2))
        t=float(np.clip((position-self.positions[i])/(self.positions[i+1]-self.positions[i]),0.,1.))
        w=t**5*(126+t*(-420+t*(540+t*(-315+70*t))))
        exponent=(1-w)*self.logarithms[i]+w*self.logarithms[i+1]
        value=self.reference_half@expm(exponent)@self.reference_half
        return (value+value.T)/2


def blend_positive_native_hessians(positions,hessians,*,reference_index):
    """Use bounded scalar switches to avoid a global high-degree overshoot."""
    s,h=np.asarray(positions,float),np.asarray(hessians,float)
    if (s.ndim!=1 or len(s)<2 or h.ndim!=3 or len(h)!=len(s) or h.shape[1]!=h.shape[2]
            or not 0<=reference_index<len(s) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(h))
            or not np.allclose(h,h.transpose(0,2,1),atol=1e-12,rtol=1e-10)
            or len(np.unique(s))!=len(s)):
        raise ValueError('distinct finite native positions and symmetric aligned kernels required')
    eigen,vectors=np.linalg.eigh(h[reference_index])
    if eigen.min()<=0:raise ValueError('positive reference kernel required')
    half=(vectors*np.sqrt(eigen))@vectors.T;inverse=(vectors/np.sqrt(eigen))@vectors.T;logs=[]
    for observed in h:
        w,v=np.linalg.eigh(inverse@observed@inverse)
        if w.min()<=0:raise ValueError('positive acquired kernels required; no curvature clipping')
        logs.append((v*np.log(w))@v.T)
    order=np.argsort(s)
    report=dict(schema='matrix.architect.positive_native_hessian_blend.v1',sample_count=len(s),
        functional_form='reference-whitened log-Euclidean blend with a common C4 scalar switch',
        exterior='constant log kernel; scalar composition retains endpoint asymptotes',
        curvature_clipping=False,reference_index=int(reference_index),complete_global_potential_qualified=False)
    return PositiveNativeHessianBlend(s[order].copy(),np.asarray(logs)[order],half,report)


@dataclass(frozen=True)
class PositiveNativeHessianRational:
    """Pole-free Floater--Hormann response of a reference-whitened log kernel.

    Six acquired kernels (five off-saddle plus the saddle) use fixed d=3.
    The exterior log kernel is constant. Molecular asymptotes remain a
    responsibility of the resident scalar composition.
    """
    positions: np.ndarray
    logarithms: np.ndarray
    reference_half: np.ndarray
    report: dict
    barycentric_weights: np.ndarray

    def evaluate_position(self,position):
        if not np.isfinite(position):raise ValueError('finite native progress required')
        u=float(np.clip(position,self.positions[0],self.positions[-1]))
        exact=np.flatnonzero(self.positions==u)
        if len(exact):exponent=self.logarithms[exact[0]]
        else:
            terms=self.barycentric_weights/(u-self.positions)
            exponent=np.einsum('i,ijk->jk',terms,self.logarithms)/terms.sum()
        exponent=(exponent+exponent.T)/2
        value=self.reference_half@expm(exponent)@self.reference_half
        return (value+value.T)/2


def rational_positive_native_hessians(positions,hessians,*,reference_index):
    """All five off-saddle observations enter one non-polynomial response.

    Floater and Hormann, Numer. Math. 107, 315--331 (2007),
    doi:10.1007/s00211-007-0093-y. No real interpolation poles; positivity
    follows from the matrix exponential, independently of rational weights.
    """
    base=blend_positive_native_hessians(positions,hessians,reference_index=reference_index)
    if len(base.positions)!=6:raise ValueError('five off-saddle kernels plus the saddle required')
    report=dict(base.report,schema='matrix.architect.positive_native_hessian_rational.v1',
        functional_form='reference-whitened log kernel; pole-free Floater-Hormann rational d=3',
        rational_degree=3,reference_DOI='10.1007/s00211-007-0093-y')
    # Published barycentric weights; no newer SciPy interpolator dependency.
    s=base.positions;n=len(s);d=3;weights=np.zeros(n)
    for k in range(n):
        for i in range(max(0,k-d),min(k,n-d-1)+1):
            others=[j for j in range(i,i+d+1) if j!=k]
            weights[k]+=(-1.)**(k-d)/np.prod(abs(s[k]-s[others]))
    weights/=np.max(abs(weights))
    return PositiveNativeHessianRational(base.positions,base.logarithms,base.reference_half,report,weights)


@dataclass(frozen=True)
class PositiveNativeHessianGaussian:
    """Gaussian interpolation of a whitened log kernel, with data-only width."""
    positions: np.ndarray
    coefficients: np.ndarray
    reference_half: np.ndarray
    width: float
    report: dict

    def evaluate_position(self,position):
        if not np.isfinite(position):raise ValueError('finite native progress required')
        u=float(np.clip(position,self.positions[0],self.positions[-1]))
        weights=np.exp(-((u-self.positions)/self.width)**2)
        exponent=np.einsum('i,ijk->jk',weights,self.coefficients)
        exponent=(exponent+exponent.T)/2
        value=self.reference_half@expm(exponent)@self.reference_half
        return (value+value.T)/2


def gaussian_positive_native_hessians(positions,hessians,*,reference_index):
    """Select Gaussian width by leaving out each off-saddle kernel once.

    The saddle remains known in each test. Only measured kernels select width;
    no withheld path field, kinetic rate or extra source observation is used.
    """
    base=blend_positive_native_hessians(positions,hessians,reference_index=reference_index)
    if len(base.positions)!=6:raise ValueError('five off-saddle kernels plus the saddle required')
    s=base.positions;logs=base.logarithms
    ref=float(np.asarray(positions)[reference_index]);held=np.flatnonzero(s!=ref)
    scale=float(np.median(np.diff(s)));best=None;history=[]
    for factor in (.25,.5,1.,2.,4.,8.,16.):
        width=scale*factor;phi=np.exp(-((s[:,None]-s[None,:])/width)**2)
        condition=float(np.linalg.cond(phi))
        if not np.isfinite(condition) or condition>1e12:
            history.append(dict(width=width,condition=condition,admitted=False));continue
        inverse=np.linalg.inv(phi);coeff=np.einsum('ij,jab->iab',inverse,logs)
        errors=[]
        for i in held:
            prediction=logs[i]-coeff[i]/inverse[i,i]
            observed=base.reference_half@expm(logs[i])@base.reference_half
            predicted=base.reference_half@expm(prediction)@base.reference_half
            errors.append(float(np.max(abs(predicted-observed))))
        score=float(np.sqrt(np.mean(np.square(errors))))
        history.append(dict(width=width,condition=condition,admitted=True,
            leave_one_out_RMS_maximum_kernel_element_error=score))
        if np.isfinite(score) and (best is None or score<best[0]):best=(score,width,coeff,condition)
    if best is None:raise ValueError('no stable Gaussian native response')
    score,width,coeff,condition=best
    report=dict(base.report,schema='matrix.architect.positive_native_hessian_gaussian.v1',
        functional_form='reference-whitened log kernel; Gaussian radial basis interpolation',
        width=width,width_selection='five off-saddle leave-one-out kernel errors; TS retained',
        condition=condition,leave_one_out_error=score,selection_history=history,
        reference_rates_used=False)
    return PositiveNativeHessianGaussian(s,coeff,base.reference_half,width,report)


def select_five_hessian_centers(arcs, native_progress):
    """Freeze five off-saddle observations using geometry alone.

    Three reactant targets and two product targets are equally spaced in
    the supplied bounded native progress from the saddle to each channel
    edge. Select the nearest unused existing geometry, breaking ties by arc.
    The TS is returned separately in the six-index ordered tuple. No force,
    anharmonic derivative or reference rate can enter this selection.
    """
    s=np.asarray(arcs,float);u=np.asarray(native_progress,float)
    if (s.ndim!=1 or s.shape!=u.shape or len(s)<6
            or not np.all(np.isfinite(s)) or not np.all(np.isfinite(u))
            or np.any(np.diff(s)<=0) or np.count_nonzero(s==0.)!=1):
        raise ValueError('ordered finite geometry arcs with exactly one zero saddle required')
    ts=int(np.flatnonzero(s==0.)[0]);centered=u-u[ts]
    du=np.diff(u)
    if not (np.all(du>0) or np.all(du<0)):
        raise ValueError('native reaction progress must be strictly monotone over the declared channel')
    selected=[]
    for sign,count in ((-1,3),(1,2)):
        candidates=np.flatnonzero(sign*s>0).tolist()
        if len(candidates)<count:raise ValueError('insufficient existing geometries on a channel side')
        edge=max(candidates,key=lambda i:abs(centered[i]))
        for fraction in np.linspace(1/count,1,count):
            available=[i for i in candidates if i not in selected]
            selected.append(min(available,key=lambda i:(abs(centered[i]-fraction*centered[edge]),s[i])))
    return tuple(sorted([ts,*selected]))
