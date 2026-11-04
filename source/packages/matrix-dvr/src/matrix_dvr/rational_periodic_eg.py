"""Reference-anchored periodic rational E/G fits with no angular poles."""
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
import numpy as np
from .periodic_hermite import angular_features
from .analytic_fit_policy import validate_sample_plan,POLICY_ID,POLICY_SHA256


@dataclass(frozen=True)
class PositiveStiffnessPeriodicEG:
    """Periodic softening of a positive reference angular stiffness.

    E=E0+g sin(q)+h(1-cos(q))/[1+a(1-cos(q))+b sin(q)]**p.
    The exponent is an interpolation parameter, not an asymptotic law.
    """
    origin: float
    gradient: float
    curvature: float
    denominator_a: float
    denominator_b: float
    exponent: float

    def __post_init__(self):
        a,b = self.denominator_a,self.denominator_b
        if (not np.all(np.isfinite(list(self.__dict__.values()))) or self.curvature<=0
                or self.exponent<=0 or 1+a-np.hypot(a,b)<=0):
            raise ValueError('finite positive stiffness, exponent and pole-free denominator required')

    def evaluate(self,angles,*,derivative=0):
        q=np.asarray(angles,float)
        if not np.all(np.isfinite(q)) or derivative not in (0,1,2):
            raise ValueError('finite angles and angular derivative 0, 1 or 2 required')
        s,c=np.sin(q),np.cos(q);z=1-c
        a,b,p=self.denominator_a,self.denominator_b,self.exponent
        d=1+a*z+b*s;dp=a*s+b*c;dpp=a*c-b*s
        if derivative==0:
            return self.origin+self.gradient*s+self.curvature*z*d**(-p)
        if derivative==1:
            return self.gradient*c+self.curvature*(s*d**(-p)-p*z*d**(-p-1)*dp)
        return -self.gradient*s+self.curvature*(c*d**(-p)-2*p*s*d**(-p-1)*dp
            +p*(p+1)*z*d**(-p-2)*dp**2-p*z*d**(-p-1)*dpp)

    def to_dict(self):
        return dict(schema='matrix.dvr.positive_stiffness_periodic_eg.v1',**self.__dict__)

    @classmethod
    def from_dict(cls,value):
        if value.get('schema')!='matrix.dvr.positive_stiffness_periodic_eg.v1':
            raise ValueError('unsupported positive-stiffness periodic model')
        return cls(**{name:float(value[name]) for name in cls.__dataclass_fields__})


def fit_positive_stiffness_periodic_eg(angles,energies,*,origin,gradient,curvature,
        coefficient_loss='least_squares'):
    """Select a compact exponent by leave-one-out on TRAINING energies only.

    Three nonzero phases identify two denominator coefficients. Candidate
    exponents are the positive fractions in [1/2,3] with denominator <=3.
    No external checking point is accepted by this API or used for selection.
    """
    from scipy.optimize import least_squares, minimize
    q=np.asarray(angles,float);e=np.asarray(energies,float)
    if (q.ndim!=1 or e.shape!=q.shape or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(e)) or not np.all(np.isfinite([origin,gradient,curvature]))
            or curvature<=0 or coefficient_loss not in ('least_squares','minimax')):
        raise ValueError('finite compatible angular energies and positive curvature required')
    z,s=1-np.cos(q),np.sin(q)
    active=z>1e-10
    if active.sum()!=3:
        raise ValueError('exactly three nonreference training phases required')
    zz,ss,yy=z[active],s[active],e[active]-origin-gradient*s[active]
    if np.any(yy<=0):
        raise ValueError('training energies must admit positive angular stiffness')
    design=np.column_stack((zz,ss))
    if np.linalg.matrix_rank(design)!=2:
        raise ValueError('independent angular phases required')
    powers=sorted({n/d for d in range(1,4) for n in range(1,3*d+1) if .5<=n/d<=3})
    candidates=[]
    for p in powers:
        errors=[]
        for held in range(3):
            mask=np.arange(3)!=held
            if np.linalg.matrix_rank(design[mask])!=2:
                errors=[];break
            target=(curvature*zz[mask]/yy[mask])**(1/p)-1
            a,b=np.linalg.solve(design[mask],target)
            if 1+a-np.hypot(a,b)<=0:
                errors=[];break
            predicted=curvature*zz[held]/(1+a*zz[held]+b*ss[held])**p
            errors.append(float(predicted-yy[held]))
        candidates.append(dict(exponent=p,training_leave_one_out_errors_cm=errors,
            training_leave_one_out_rms_cm=float(np.linalg.norm(errors)/np.sqrt(3)) if errors else None))
    valid=[c for c in candidates if c['training_leave_one_out_rms_cm'] is not None]
    if not valid:
        raise ValueError('no pole-free angular candidate identified by training cross-validation')
    selected=min(valid,key=lambda c:c['training_leave_one_out_rms_cm'])
    p=selected['exponent']
    a,b=np.linalg.lstsq(design,(curvature*zz/yy)**(1/p)-1,rcond=None)[0]
    # This parametrization guarantees positivity around the entire circle.
    u=max(1+2*a-b*b,1e-6)
    def unpack(v):
        return .5*(v[1]**2+np.exp(v[0])-1),v[1]
    def residual(v):
        a,b=unpack(v)
        return curvature*zz/(1+a*zz+b*ss)**p-yy
    fit=least_squares(residual,[np.log(u),b],bounds=([-20.,-100.],[20.,100.]),
        xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=1000)
    if not fit.success or np.linalg.matrix_rank(fit.jac)!=2:
        raise ValueError('positive angular stiffness fit not identified or not converged')
    a,b=unpack(fit.x)
    if coefficient_loss=='minimax':
        # Minimize the same absolute worst-point error used by the energy
        # qualification gate. Only training observations enter the epigraph.
        def errors(ab):
            return curvature*zz/(1+ab[0]*zz+ab[1]*ss)**p-yy
        start=np.array([a,b,float(max(abs(errors([a,b]))))])
        result=minimize(lambda v:v[2],start,method='SLSQP',
            bounds=[(None,None),(None,None),(0,None)],
            constraints=[
                dict(type='ineq',fun=lambda v:np.r_[v[2]-errors(v[:2]),v[2]+errors(v[:2])]),
                dict(type='ineq',fun=lambda v:1+v[0]-np.hypot(*v[:2])-1e-8),
            ],options={'ftol':1e-12,'maxiter':1000})
        if (not result.success or not np.all(np.isfinite(result.x))
                or max(abs(errors(result.x[:2])))>start[2]+1e-7):
            raise ValueError('pole-free minimax angular fit did not converge')
        a,b=result.x[:2]
    model=PositiveStiffnessPeriodicEG(float(origin),float(gradient),float(curvature),float(a),float(b),float(p))
    return model,dict(training_energy_max_error_cm=float(max(abs(model.evaluate(q)-e))),
        minimum_denominator=float(1+a-np.hypot(a,b)),reference_twojet_exact=True,
        identified_rank=2,free_coefficients=2,selected_exponent=p,
        exponent_selection='minimum leave-one-out RMS on training phases only',
        training_leave_one_out_rms_cm=selected['training_leave_one_out_rms_cm'],
        training_leave_one_out_max_error_cm=max(map(abs,selected['training_leave_one_out_errors_cm'])),
        exponent_candidates=candidates,torque_observations_supplied=False,
        coefficient_loss=coefficient_loss)


@dataclass(frozen=True)
class RationalPeriodicEG:
    origin: float
    gradient: float
    curvature: float
    powers: np.ndarray
    coefficients: np.ndarray
    denominator_b: float

    def evaluate(self,angles,*,derivative=0):
        q=np.asarray(angles,float);t=1-np.cos(q);d=1+self.denominator_b*t
        if not np.all(np.isfinite(q)) or np.any(d<=0):raise ValueError('finite angles and positive periodic denominator required')
        N=self.gradient*np.sin(q)+self.curvature*t+angular_features(q[...,None],self.powers,[1])@self.coefficients
        if derivative==0:return self.origin+N/d
        if derivative not in (1,2):raise ValueError('angular derivative 0, 1 or 2 required')
        Np=self.gradient*np.cos(q)+self.curvature*np.sin(q)+angular_features(q[...,None],self.powers,[1],[1])@self.coefficients
        dp=self.denominator_b*np.sin(q)
        if derivative==1:return Np/d-N*dp/d**2
        Npp=-self.gradient*np.sin(q)+self.curvature*np.cos(q)+angular_features(q[...,None],self.powers,[1],[2])@self.coefficients
        return Npp/d-2*Np*dp/d**2-N*self.denominator_b*np.cos(q)/d**2+2*N*dp*dp/d**3

    def to_dict(self):
        return dict(schema='matrix.dvr.rational_periodic_eg.v1',reference_energy_cm=self.origin,
            reference_gradient_cm_per_rad=self.gradient,reference_curvature_cm_per_rad2=self.curvature,
            powers=self.powers.tolist(),coefficients=self.coefficients.tolist(),denominator_b=self.denominator_b)

    @classmethod
    def from_dict(cls,r):
        if r.get('schema')!='matrix.dvr.rational_periodic_eg.v1':raise ValueError('unsupported periodic rational model')
        b=float(r['denominator_b'])
        if not np.isfinite(b) or b<=-.5:raise ValueError('periodic denominator must stay strictly positive')
        return cls(float(r['reference_energy_cm']),float(r['reference_gradient_cm_per_rad']),
            float(r['reference_curvature_cm_per_rad2']),np.asarray(r['powers'],int),np.asarray(r['coefficients'],float),b)


def fit_rational_periodic_eg(angles,energies,gradients,*,origin,gradient,curvature,
        reflection_even=False,derivative_length=.3,fourier_degree=3):
    from scipy.optimize import minimize_scalar
    q=np.asarray(angles,float);e=np.asarray(energies,float);g=None if gradients is None else np.asarray(gradients,float)
    if (q.ndim!=1 or e.shape!=q.shape or (g is not None and (g.shape!=q.shape or not np.all(np.isfinite(g)))) or len(q)<3
            or not all(np.all(np.isfinite(x)) for x in (q,e,[origin,gradient,curvature,derivative_length]))
            or derivative_length<=0):raise ValueError('finite compatible periodic E/G observations required')
    if isinstance(fourier_degree,bool) or not isinstance(fourier_degree,int) or fourier_degree<2:
        raise ValueError('Fourier degree must be an integer >=2')
    powers=np.arange(4 if reflection_even else 3,2*fourier_degree+1,2 if reflection_even else 1)[:,None]
    A=angular_features(q[:,None],powers,[1]);D=angular_features(q[:,None],powers,[1],[1])
    t=1-np.cos(q);tp=np.sin(q);N=gradient*np.sin(q)+curvature*t;Np=gradient*np.cos(q)+curvature*tp
    def solve(b):
        d=1+b*t;dp=b*tp;a=A/d[:,None];da=D/d[:,None]-A*dp[:,None]/d[:,None]**2
        base=origin+N/d;db=Np/d-N*dp/d**2
        system=a;rhs=e-base
        if g is not None:
            system=np.vstack((a,derivative_length*da));rhs=np.r_[rhs,derivative_length*(g-db)]
        scale=np.linalg.norm(system,axis=0)
        if np.any(scale==0):raise ValueError('unidentified periodic rational feature')
        c,_,rank,singular=np.linalg.lstsq(system/scale,rhs,rcond=1.e-12)
        if rank!=len(powers):raise ValueError('periodic rational fit not identified')
        c=c/scale;res=system@c-rhs
        return float(res@res),c,float(singular[0]/singular[-1])
    search=np.r_[np.linspace(-.499,.5,31),np.geomspace(.501,100,25)]
    best=min(range(len(search)),key=lambda i:solve(search[i])[0])
    lo=search[max(0,best-1)];hi=search[min(len(search)-1,best+1)]
    optimum=minimize_scalar(lambda b:solve(b)[0],bounds=(lo,hi),method='bounded',options={'xatol':1.e-10})
    if not optimum.success:raise ValueError('periodic rational variable projection did not converge')
    cost,c,condition=solve(float(optimum.x))
    model=RationalPeriodicEG(float(origin),float(gradient),float(curvature),powers,c,float(optimum.x))
    return model,dict(training_weighted_cost=cost,scaled_design_condition=condition,
        training_energy_max_error_cm=float(max(abs(model.evaluate(q)-e))),
        reflection_even=reflection_even,torque_observations_supplied=g is not None,
        fourier_degree=fourier_degree,denominator_search_bounds=[-.499,100.],
        minimum_denominator=float(min(1.,1+2*model.denominator_b)),reference_twojet_exact=True)


def run_rational_periodic_eg_request(path):
    path=Path(path).resolve();raw=path.read_bytes();r=json.loads(raw)
    if r.get('schema')!='matrix.dvr.analytic_surface_request.v1' or r.get('family')!='rational_periodic_eg':
        raise ValueError('typed periodic rational E/G request required')
    q=np.asarray(r['angles_radian'],float);e=np.asarray(r['energies_cm'],float)
    g=None if r.get('torques_cm_per_rad') is None else np.asarray(r['torques_cm_per_rad'],float)
    validate_sample_plan(sample_count=len(q),sample_budget=r['sample_budget'],adaptive_qm=r.get('adaptive_qm',False))
    ids=r['validation_indices']
    if not ids or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(q) for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('distinct held-out angular indices required')
    mask=np.ones(len(q),bool);mask[ids]=False
    model,audit=fit_rational_periodic_eg(q[mask],e[mask],None if g is None else g[mask],origin=r['reference_energy_cm'],
        gradient=r['reference_torque_cm_per_rad'],curvature=r['reference_curvature_cm_per_rad2'],
        reflection_even=r.get('reflection_even',False),derivative_length=r.get('derivative_length_radian',.3),
        fourier_degree=r.get('fourier_degree',3))
    gate=float(r['max_validation_error_cm'])
    if not np.isfinite(gate) or gate<=0:raise ValueError('positive energy guard required')
    error=float(max(abs(model.evaluate(q[~mask])-e[~mask])))
    passing=error<=gate and audit['training_energy_max_error_cm']<=gate
    out=path.parent/r.get('output_directory','rational_periodic_fit');out.mkdir(parents=True,exist_ok=True)
    artifact=out/'periodic_surface.json';artifact.write_text(json.dumps(model.to_dict(),indent=2)+'\n')
    result=dict(schema='matrix.dvr.analytic_surface_result.v1',family='rational_periodic_eg',
        policy_id=POLICY_ID,policy_sha256=POLICY_SHA256,request_sha256=hashlib.sha256(raw).hexdigest(),
        sample_count=len(q),sample_budget=r['sample_budget'],validation_indices=ids,
        validation_max_error_cm=error,fit_audit=audit,held_out_geometries_excluded_from_fit=True,
        artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        status='fit_validated' if passing else 'fit_not_qualified',grid_source='analytic_function_only',
        QM_evaluations_for_generated_grid=0)
    target=out/'analytic_surface_result.json';target.write_text(json.dumps(result,indent=2)+'\n')
    if not passing:raise ValueError('periodic rational fit fails frozen energy gate; no extra source points')
    return target
