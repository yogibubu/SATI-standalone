"""Finite radial refit retaining the original ZAFF long-range functions.

The prior is the resident analytic ZAFF evaluator (including electrostatics,
induction/dispersion and switches). Only exponentially decaying corrections
are fitted, so existing inverse-power asymptotes remain unchanged. Angular
features are supplied by the prior's physical orientation representation.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


def fit_contact_capture_energies(radii_angstrom, energies_cm, *, asymptote_cm,
        c3_cm_angstrom3, contact_parallel_offset, contact_perpendicular_squared,
        validation_indices, provenance, sample_budget=5, tolerance_cm=10.):
    """Finite exp/4/6 contact fit; held-out energies never enter optimization.

    The contact geometry is supplied from the frozen fragment translation.
    Only the training energies determine the exponent and positive amplitudes.
    Crisci et al., JCTC 2024, 20, 9435, DOI 10.1021/acs.jctc.4c01076
    motivates the contact inverse powers; the repulsive branch is an extension.
    """
    from scipy.optimize import lsq_linear, minimize_scalar, brentq
    from matrix_dvr.analytic_fit_policy import validate_sample_plan
    from matrix_zaff.radial import ContactCapturePotential
    r = np.asarray(radii_angstrom, float)
    e = np.asarray(energies_cm, float)
    held = np.asarray(validation_indices)
    validate_sample_plan(sample_count=len(r), sample_budget=sample_budget)
    if (r.ndim != 1 or e.shape != r.shape or len(r) < 5 or np.any(r <= 0)
            or np.any(np.diff(r) <= 0) or held.ndim != 1 or not len(held)
            or held.dtype.kind not in 'iu' or np.any(held < 0) or np.any(held >= len(r))
            or len(np.unique(held)) != len(held) or not provenance
            or contact_perpendicular_squared < 0 or tolerance_cm <= 0
            or not all(np.all(np.isfinite(v)) for v in
                       (r,e,[asymptote_cm,c3_cm_angstrom3,contact_parallel_offset,
                             contact_perpendicular_squared,tolerance_cm]))):
        raise ValueError('finite ordered samples, contact geometry and distinct held-out indices required')
    train = np.ones(len(r), bool); train[held] = False
    if train.sum() < 4:
        raise ValueError('four independent training energies required for exp/4/6 fit')
    d = np.sqrt((r+contact_parallel_offset)**2+contact_perpendicular_squared)
    if np.any(d <= 0):
        raise ValueError('positive contact distances required')
    d0 = float(min(d))
    prior = asymptote_cm+c3_cm_angstrom3/r**3

    def solve(a):
        x = np.column_stack((np.exp(-a*(d-d0)), -(d0/d)**4, -(d0/d)**6))
        scale = np.linalg.norm(x[train], axis=0)
        if np.any(scale == 0):
            raise ValueError('unidentified contact feature')
        f = lsq_linear(x[train]/scale, e[train]-prior[train], bounds=(0,np.inf),
                       tol=1e-13, max_iter=1000)
        if not f.success:
            raise ValueError('positive contact amplitude fit did not converge')
        c = f.x/scale
        err = x[train]@c+prior[train]-e[train]
        return float(err@err), c, float(np.linalg.cond(x[train]/scale))

    mesh = np.geomspace(.05,20.,129)
    k = min(range(len(mesh)), key=lambda i: solve(mesh[i])[0])
    f = minimize_scalar(lambda a: solve(a)[0],
        bounds=(mesh[max(0,k-1)],mesh[min(len(mesh)-1,k+1)]), method='bounded',
        options={'xatol':1e-12})
    if not f.success:
        raise ValueError('contact exponent search did not converge')
    cost,c,condition = solve(float(f.x))
    model = ContactCapturePotential(float(asymptote_cm),float(c3_cm_angstrom3),
        float(contact_parallel_offset),float(contact_perpendicular_squared),d0,
        float(min(r)),float(f.x),*map(float,c))
    err = np.array([model.derivatives(float(q)).energy for q in r])-e
    # Dense analytic shape inspection does not evaluate the source potential.
    mesh = np.geomspace(float(min(r)),max(float(max(r)),float(min(r))*1e4),4097)
    slopes = np.array([model.derivatives(float(q)).first for q in mesh])
    roots = [brentq(lambda q:model.derivatives(q).first,a,b)
             for a,b,ga,gb in zip(mesh[:-1],mesh[1:],slopes[:-1],slopes[1:]) if ga*gb < 0]
    minima = [q for q in roots if model.derivatives(q).second > 0]
    training_error = float(max(abs(err[train])))
    validation_error = float(max(abs(err[~train])))
    return model, dict(qualified=validation_error<=tolerance_cm and training_error<=tolerance_cm
        and len(minima)==1, training_error_cm=training_error,held_out_error_cm=validation_error,
        tolerance_cm=float(tolerance_cm),validation_indices=held.tolist(),
        training_cost=cost,scaled_design_condition=condition,stable_minimum_count=len(minima),
        stationary_radii_A=roots,minimum_radii_A=minima,
        model=model.to_dict(),provenance=provenance,new_source_evaluations=0,
        coefficient_selection='training energies only; fixed exponent search interval',
        contact_power_reference='10.1021/acs.jctc.4c01076',
        qualification_scope='fixed acquired radial profile and analytic shape check; no inner extrapolation')


def fixed_inverse_power_radii(minimum, maximum, *, power, count=5):
    """A priori radial nodes, equally spaced in R^-power; no adaptive acquisition."""
    if (not np.isfinite(minimum) or not np.isfinite(maximum) or not 0 < minimum < maximum
            or not np.isfinite(power) or power <= 0 or not isinstance(count,int) or not 3 <= count <= 5):
        raise ValueError("positive ordered bounds/power and three to five radial nodes required")
    return np.linspace(minimum**-power,maximum**-power,count)**(-1/power)


@dataclass(frozen=True)
class AnalyticCaptureRefit:
    prior: object
    exponents_per_angstrom: np.ndarray
    coefficients_cm: np.ndarray
    reference_radius_angstrom: float
    radial_domain_angstrom: tuple[float,float]
    provenance: str
    qualification: dict

    def jets(self,radius_angstrom,angular_features):
        """Energy and first/second radial derivatives, cm-1 and Angstrom units."""
        radius=float(radius_angstrom)
        features=np.asarray(angular_features,float)
        if (not np.isfinite(radius) or radius <= 0 or features.shape != (self.coefficients_cm.shape[0],)
                or not np.all(np.isfinite(features))):
            raise ValueError("positive radius and compatible finite angular features required")
        prior=np.asarray(self.prior(radius,features),float)
        if prior.shape != (3,) or not np.all(np.isfinite(prior)):
            raise ValueError("ZAFF prior must supply its finite radial two-jet")
        exponent=self.exponents_per_angstrom
        basis=np.exp(-exponent*(radius-self.reference_radius_angstrom))
        amplitude=features@self.coefficients_cm
        return prior+np.array([np.sum(amplitude*basis*(-exponent)**order) for order in range(3)])


def fit_capture_refit(prior,radii_angstrom,angular_features,target_jets_cm,*,
                      validation_indices,provenance,exponents_per_angstrom=(.4,.8,1.6),
                      jet_scales=(1.,1.,1.),tolerances=(10.,25.,100.)):
    """Fit already acquired radial jets; all held-out jets remain independent.

    Finite samples, including angular features, must be chosen before this
    function is called. No surface/backend evaluation acquires new information
    during fitting or validation. A failed result remains unqualified.
    """
    radii=np.asarray(radii_angstrom,float);features=np.asarray(angular_features,float)
    target=np.asarray(target_jets_cm,float);exponent=np.asarray(exponents_per_angstrom,float)
    scale=np.asarray(jet_scales,float);tolerance=np.asarray(tolerances,float)
    held=np.asarray(validation_indices)
    if (radii.ndim != 1 or features.ndim != 2 or features.shape[0] != radii.size
            or target.shape != (radii.size,3) or exponent.ndim != 1 or not exponent.size
            or np.any(exponent <= 0) or np.any(radii <= 0) or scale.shape != (3,) or np.any(scale <= 0)
            or tolerance.shape != (3,) or np.any(tolerance <= 0)
            or held.ndim != 1 or not held.size or held.dtype.kind not in 'iu'
            or np.any(held < 0) or np.any(held >= radii.size) or len(np.unique(held)) != held.size
            or not all(np.all(np.isfinite(v)) for v in (radii,features,target,exponent,scale,tolerance))
            or not provenance):
        raise ValueError("finite compatible radial jets/features, positive scales and independent validation indices required")
    center=float(min(radii))
    design=[];differences=[]
    for r,angular,jet in zip(radii,features,target):
        prior_jet=np.asarray(prior(r,angular),float)
        if prior_jet.shape != (3,) or not np.all(np.isfinite(prior_jet)):
            raise ValueError("finite prior radial jets required")
        basis=np.exp(-exponent*(r-center))
        design.append(np.stack([(angular[:,None]*basis[None,:]*(-exponent[None,:])**order).ravel()
                                 for order in range(3)]))
        differences.append(jet-prior_jet)
    design=np.array(design);differences=np.array(differences)
    training=np.ones(radii.size,bool);training[held]=False
    matrix=(design[training]/scale[None,:,None]).reshape(-1,features.shape[1]*len(exponent))
    vector=(differences[training]/scale).ravel()
    column_norm=np.linalg.norm(matrix,axis=0)
    if np.any(column_norm == 0):
        raise ValueError("radial/angular sampling does not identify every correction coefficient")
    normalized=matrix/column_norm
    coefficients,_,rank,_=np.linalg.lstsq(normalized,vector,rcond=None)
    if rank != matrix.shape[1]:
        raise ValueError("radial/angular correction is underdetermined")
    coefficients=coefficients/column_norm
    residual=np.einsum('sij,j->si',design,coefficients)-differences
    error=np.max(abs(residual[held]),axis=0)
    qualified=bool(np.all(error <= tolerance))
    digest=hashlib.sha256()
    for value in (radii,features,target,exponent,held):
        digest.update(value.tobytes())
    report=dict(qualified=qualified,sample_count=len(radii),validation_indices=held.tolist(),
        held_out_errors=error.tolist(),tolerances=tolerance.tolist(),
        derivative_units=['cm-1','cm-1/Angstrom','cm-1/Angstrom^2'],
        scaled_condition_number=float(np.linalg.cond(normalized)),sample_sha256=digest.hexdigest(),
        asymptote='original ZAFF inverse-power terms retained; corrections decay exponentially',
        new_source_evaluations=0,source=provenance)
    return AnalyticCaptureRefit(prior,exponent,coefficients.reshape(features.shape[1],-1),center,
                                (float(min(radii)),float(max(radii))),provenance,report)


def fit_inverse_power_capture_energies(radii_angstrom, energies_cm, *,
        asymptote_cm, c3_cm_angstrom3, validation_indices, provenance,
        sample_budget=5, tolerance_cm=10., powers=(6,8,10,12)):
    """Fit finite scalar observations with a fixed dipolar asymptote.

    The resident ZAFF inverse-distance evaluator owns the resulting function.
    C6 is attractive and the highest inverse power repulsive. No derivatives
    are synthesized, no additional source points are evaluated, and the held
    out observations are never used to determine any coefficient.
    """
    from scipy.optimize import lsq_linear
    from matrix_dvr.analytic_fit_policy import validate_sample_plan
    r=np.asarray(radii_angstrom,float);energy=np.asarray(energies_cm,float)
    p=np.asarray(powers);held=np.asarray(validation_indices)
    validate_sample_plan(sample_count=len(r),sample_budget=sample_budget)
    from matrix_zaff.radial import PiecewiseInverseDistancePotential
    if (r.ndim!=1 or energy.shape!=r.shape or np.any(r<=0) or np.any(np.diff(r)<=0)
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(energy))
            or p.ndim!=1 or len(p)<2 or p.dtype.kind not in 'iu' or p[0]!=6
            or np.any(np.diff(p)<=0) or p[-1]%2 or held.ndim!=1 or not len(held)
            or held.dtype.kind not in 'iu' or np.any(held<0) or np.any(held>=len(r))
            or len(np.unique(held))!=len(held) or not provenance
            or not np.all(np.isfinite([asymptote_cm,c3_cm_angstrom3,tolerance_cm])) or tolerance_cm<=0):
        raise ValueError('ordered positive radii, finite energies/asymptote, powers and held-out indices required')
    train=np.ones(len(r),bool);train[held]=False
    if sum(train)<len(p):raise ValueError('insufficient independent scalar observations')
    scale_radius=float(min(r));u=scale_radius/r
    matrix=u[:,None]**p;prior=asymptote_cm+c3_cm_angstrom3/r**3
    norm=np.linalg.norm(matrix[train],axis=0)
    if np.linalg.matrix_rank(matrix[train]/norm)!=len(p):raise ValueError('radial powers not identified')
    lower=np.full(len(p),-np.inf);upper=np.full(len(p),np.inf)
    upper[0]=0.;lower[-1]=0.
    solution=lsq_linear(matrix[train]/norm,energy[train]-prior[train],bounds=(lower,upper),
                        tol=1e-12,max_iter=1000)
    if not solution.success:raise ValueError('constrained inverse-distance fit did not converge')
    coefficients=solution.x/norm
    polynomial=np.zeros(int(p[-1])+1);polynomial[0]=asymptote_cm
    polynomial[3]=c3_cm_angstrom3/scale_radius**3;polynomial[p]=coefficients
    potential=PiecewiseInverseDistancePotential(scale_radius,(0.,1.),(tuple(polynomial),))
    prediction=np.array([potential.derivatives(float(v)).energy for v in r])
    error=float(max(abs(prediction[held]-energy[held])))
    training=float(max(abs(prediction[train]-energy[train])))
    roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(polynomial))
    critical=sorted(float(z.real) for z in roots if abs(z.imag)<1e-7 and 1e-8<z.real<1.)
    stable=[v for v in critical if np.polynomial.polynomial.polyval(v,
        np.polynomial.polynomial.polyder(polynomial,2))>0]
    qualified=bool(error<=tolerance_cm and training<=tolerance_cm and coefficients[-1]>0 and len(stable)==1)
    report=dict(qualified=qualified,held_out_error_cm=error,training_error_cm=training,
        tolerance_cm=tolerance_cm,validation_indices=held.tolist(),powers=p.tolist(),
        fixed_C3_cm_A3=float(c3_cm_angstrom3),fitted_C6_cm_A6=float(-coefficients[0]*scale_radius**6),
        asymptote_cm=float(asymptote_cm),reference_radius_A=scale_radius,
        inverse_distance_coefficients_cm=polynomial.tolist(),stationary_inverse_radii=critical,
        stable_minimum_count=len(stable),source=provenance,source_evaluations=0,
        extrapolation_below_minimum_sample_radius_allowed=False,
        long_range_form='D + C3/R^3 - C6/R^6 + higher inverse powers')
    return potential,report
