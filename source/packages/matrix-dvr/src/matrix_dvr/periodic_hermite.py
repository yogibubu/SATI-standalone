"""Periodic angular surfaces from a finite energy/gradient/Hessian source.

Physical inputs are angles in radians, each with period 2*pi. A trigonometric
quadratic baseline preserves the exact reference Hessian; higher features have
zero reference value, gradient and Hessian. No electronic callback is present.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from .analytic_fit_policy import validate_sample_plan,POLICY_ID

@lru_cache(maxsize=None)
def _laurent(power):
    coefficients={0:1.+0j}
    for _ in range(power//2):
        result={}
        for k,c in coefficients.items():
            for j,a in ((0,2.),(-1,-1.),(1,-1.)):result[k+j]=result.get(k+j,0)+c*a
        coefficients=result
    if power%2:
        result={}
        for k,c in coefficients.items():
            for j,a in ((-1,.5j),(1,-.5j)):result[k+j]=result.get(k+j,0)+c*a
        coefficients=result
    return tuple(sorted(coefficients.items()))


@lru_cache(maxsize=None)
def _half_angle_terms(power, order):
    """Differentiate (2 sin(theta/2))**power cos(theta/2)**(power%2).

    Every term retains nonnegative powers. Reference zero jets are exact,
    unlike a Laurent sum of nearly cancelling complex exponentials.
    """
    terms = {(int(power), int(power)%2): 1.}
    for _ in range(int(order)):
        derived = {}
        for (a, b), value in terms.items():
            if a:
                key = (a-1, b+1); derived[key] = derived.get(key, 0.)+a*value
            if b:
                key = (a+1, b-1); derived[key] = derived.get(key, 0.)-b*value/4
        terms = {key:value for key, value in derived.items() if value != 0.}
    return tuple((a, b, value) for (a, b), value in terms.items())


def angular_features(points,powers,scales,derivative=None):
    q=np.asarray(points,float);powers=np.asarray(powers,int);scale=np.asarray(scales,float)
    order=np.zeros(len(scale),int) if derivative is None else np.asarray(derivative,int)
    if (q.shape[-1:]!=scale.shape or powers.ndim!=2 or powers.shape[1]!=len(scale)
            or np.any(powers<0) or np.any(scale<=0) or order.shape!=scale.shape or np.any(order<0)):
        raise ValueError('Compatible angular coordinates, integer powers and positive feature scales required')
    result=[]; values={}
    half_sine=2*np.sin(q/2); half_cosine=np.cos(q/2)
    for p in powers:
        column=np.ones(q.shape[:-1])
        for axis,(power,k) in enumerate(zip(p,order)):
            key=(axis,int(power),int(k))
            if key not in values:
                value=np.zeros(q.shape[:-1])
                for a,b,c in _half_angle_terms(int(power),int(k)):
                    value+=c*half_sine[...,axis]**a*half_cosine[...,axis]**b
                values[key]=value/scale[axis]**int(power)
            column*=values[key]
        result.append(column)
    return np.stack(result,axis=-1) if result else np.empty(q.shape[:-1]+(0,))

def derivative_orders(dimension):
    eye=np.eye(dimension,dtype=int)
    return [(0,)*dimension]+[tuple(x) for x in eye]+[tuple(eye[i]+eye[j]) for i in range(dimension) for j in range(i,dimension)]

@dataclass(frozen=True)
class PeriodicHermiteSurface:
    reference_hessian: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray
    feature_scales: np.ndarray
    sampled_extent_radian: np.ndarray | None = None
    global_reflection_even: bool = True

    def __post_init__(self):
        if not isinstance(self.global_reflection_even,(bool,np.bool_)):
            raise ValueError('Explicit Boolean reflection policy required')
        if self.sampled_extent_radian is None:
            object.__setattr__(self,'sampled_extent_radian',np.asarray(self.feature_scales))
        for name in self.__dataclass_fields__:
            if name=='global_reflection_even':continue
            value=np.array(getattr(self,name),dtype=int if name=='powers' else float,copy=True)
            if not np.all(np.isfinite(value)):raise ValueError('Finite periodic-surface parameters required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        d=len(self.feature_scales)
        if (self.feature_scales.ndim!=1 or d<1 or np.any(self.feature_scales<=0)
                or self.sampled_extent_radian.shape!=(d,) or np.any(self.sampled_extent_radian<=0)
                or self.reference_hessian.shape!=(d,d)
                or not np.allclose(self.reference_hessian,self.reference_hessian.T,atol=1e-10)
                or self.powers.ndim!=2 or self.powers.shape[1]!=d
                or self.coefficients.shape!=(len(self.powers),) or np.any(self.powers<0)
                or np.any(self.powers.sum(axis=1)<3)
                or (self.global_reflection_even and np.any(self.powers.sum(axis=1)%2))):
            raise ValueError('Symmetric Hessian and anchored features consistent with declared reflection policy required')

    def baseline(self,points,derivative=None):
        d=len(self.feature_scales);terms=[];values=[]
        for i in range(d):
            power=np.zeros(d,int);power[i]=2;terms.append(power);values.append(.5*self.reference_hessian[i,i])
            for j in range(i+1,d):
                power=np.zeros(d,int);power[i]=power[j]=1;terms.append(power);values.append(self.reference_hessian[i,j])
        return angular_features(points,terms,np.ones(d),derivative)@values

    def evaluate(self,points,derivative=None,*,allow_extrapolation=False):
        wrapped=np.arctan2(np.sin(points),np.cos(points))
        if not allow_extrapolation and np.any(abs(wrapped)>self.sampled_extent_radian+1e-10):
            raise ValueError('Periodic potential queried outside its qualified angular envelope')
        return self.baseline(points,derivative)+angular_features(points,self.powers,self.feature_scales,derivative)@self.coefficients

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_hermite_surface.v1',reference_hessian=self.reference_hessian.tolist(),powers=self.powers.tolist(),coefficients=self.coefficients.tolist(),feature_scales=self.feature_scales.tolist(),sampled_extent_radian=self.sampled_extent_radian.tolist(),domain_geometry='Symmetric componentwise envelope; multidimensional probability coverage requires a separate DVR audit',periods_radian=[2*np.pi]*len(self.feature_scales),protocol_id=POLICY_ID,global_reflection_even=bool(self.global_reflection_even),physical_collision_or_dissociation_asymptotes_claimed=False,feature_evaluator='Stable half-angle polynomial derivatives; exact zero reference jets')

    @classmethod
    def from_dict(cls,value):
        if value.get('schema')!='matrix.dvr.periodic_hermite_surface.v1':raise ValueError('Unsupported periodic surface schema')
        return cls(*[np.asarray(value[key]) for key in ['reference_hessian','powers','coefficients','feature_scales']],sampled_extent_radian=np.asarray(value.get('sampled_extent_radian',value['feature_scales'])),global_reflection_even=value.get('global_reflection_even',True))

def fit_periodic_hermite_surface(coordinates,observations,reference_hessian,*,powers,feature_scales,sample_budget,derivative_length=.15,weights=None,rcond=1e-10,ridge=0.,global_reflection_even=True,observation_mask=None):
    q=np.asarray(coordinates,float);y=np.asarray(observations,float);h=np.asarray(reference_hessian,float);p=np.asarray(powers,int);scale=np.asarray(feature_scales,float);d=q.shape[1];orders=derivative_orders(d)
    observed=np.ones(y.shape,dtype=bool) if observation_mask is None else np.asarray(observation_mask)
    if observed.dtype.kind != "b" or observed.shape != y.shape or not np.any(observed):
        raise ValueError("Nonempty Boolean observation mask matching E/G/H required")
    validate_sample_plan(sample_count=len(q),sample_budget=sample_budget)
    if (y.shape!=(len(orders),len(q)) or h.shape!=(d,d) or p.ndim!=2 or p.shape[1]!=d
            or np.any(p.sum(axis=1)<3) or (global_reflection_even and np.any(p.sum(axis=1)%2)) or derivative_length<=0 or not 0<rcond<1 or ridge<0
            or not all(np.all(np.isfinite(v)) for v in [q,y[observed],h,scale])):raise ValueError('Finite compatible E/G/H angular observations and even anchored terms required')
    extent=np.max(abs(q),axis=0)
    model=PeriodicHermiteSurface(h,p,np.zeros(len(p)),scale,extent,global_reflection_even)
    matrices=np.array([angular_features(q,p,scale,o) for o in orders]);base=np.array([model.baseline(q,o) for o in orders]);f=np.array([derivative_length**sum(o) for o in orders]);w=np.ones(len(q)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(q),) or not np.all(np.isfinite(w)) or np.any(w<=0):raise ValueError('Positive geometry weights required')
    A=(matrices*f[:,None,None]*w[None,:,None])[observed];rhs=((y-base)*f[:,None]*w)[observed]
    U,s,Vt=np.linalg.svd(A,full_matrices=False)
    if not len(s) or s[0]<=0:raise ValueError("No identifiable periodic coefficients in observed data")
    keep=s>rcond*s[0];coef=Vt[keep].T@((s[keep]/(s[keep]**2+ridge*s[0]**2))*(U[:,keep].T@rhs))
    result=PeriodicHermiteSurface(h,p,coef,scale,extent,global_reflection_even);pred=np.array([result.evaluate(q,o) for o in orders]);error=pred-y
    def rms(section):
        values=error[section][observed[section]]
        return float(np.sqrt(np.mean(values**2))) if len(values) else None
    return result,dict(observed_counts=observed.sum(axis=1).tolist(),rank=int(keep.sum()),raw_feature_count=len(p),retained_condition=float(s[0]/s[keep][-1]),singular_values=s.tolist(),rcond=rcond,ridge=ridge,training_energy_RMS=rms(0),training_gradient_RMS=rms(slice(1,1+d)),training_Hessian_RMS=rms(slice(1+d,None)),coefficient_policy='Identified weighted Hermite SVD subspace; unsupported null-space coefficients fixed to zero',exact_reference_E_G_H=True)


def run_periodic_hermite_request(request_path):
    """KeyMaker finite-source owner for E/G/H-constrained physical angles.

    Validation geometries remain excluded from the final coefficients. The
    returned qualification is for the declared electronic energy gate only;
    no nuclear-motion or molecular asymptotic accuracy is inferred.
    """
    from pathlib import Path
    import json,hashlib
    from .analytic_fit_policy import POLICY_SHA256
    path=Path(request_path).resolve();raw=path.read_bytes();r=json.loads(raw)
    if r.get('schema')!='matrix.dvr.analytic_surface_request.v1' or r.get('family')!='periodic_hermite':raise ValueError('Typed periodic Hermite request required')
    if r.get('coordinate_unit')!='radian' or r.get('energy_unit')!='cm^-1':raise ValueError('Declare physical radian angles and cm^-1 energies')
    q=np.asarray(r['coordinates_radian'],float);y=np.asarray(r['observations_E_G_H'],float);h=np.asarray(r['reference_hessian_cm_per_rad2'],float)
    validate_sample_plan(sample_count=len(q),sample_budget=r['sample_budget'],adaptive_qm=r.get('adaptive_qm',False))
    ids=np.asarray(r['validation_indices']);mask=np.zeros(len(q),bool)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(q)):raise ValueError('Distinct held-out geometry indices required')
    mask[ids]=True;orders=derivative_orders(q.shape[1]);scale=np.asarray(r['feature_scales_radian'],float);weights=np.asarray(r.get('geometry_weights',np.ones(len(q))),float)
    gate=float(r['max_validation_error_cm'])
    if not np.isfinite(gate) or gate<=0 or y.shape!=(len(orders),len(q)):raise ValueError('Compatible Hermite observations and positive energy gate required')
    observed=np.asarray(r.get('observation_mask',np.ones(y.shape,dtype=bool)))
    if observed.dtype.kind!='b' or observed.shape!=y.shape or not np.all(observed[0,mask]) or not np.all(np.isfinite(y[observed])):
        raise ValueError('Boolean E/G/H observation mask and finite held-out energies required')
    models=[];candidates=[]
    for option in r['candidates']:
        model,audit=fit_periodic_hermite_surface(q[~mask],y[:,~mask],h,powers=option['powers'],feature_scales=scale,sample_budget=r['sample_budget'],derivative_length=r.get('derivative_length_radian',.15),weights=weights[~mask],rcond=option.get('rcond',1e-10),ridge=option.get('ridge',0.),global_reflection_even=r.get('global_reflection_even',True),observation_mask=observed[:,~mask])
        prediction=np.array([model.evaluate(q,o) for o in orders]);error=prediction-y
        def validation_rms(section):
            values=error[section][:,mask][observed[section][:,mask]]
            return float(np.sqrt(np.mean(values**2))) if len(values) else None
        audit.update(name=option['name'],validation_energy_errors_cm=error[0,mask].tolist(),validation_energy_max_cm=float(np.max(abs(error[0,mask]))),validation_gradient_RMS_cm_per_rad=validation_rms(slice(1,1+q.shape[1])),validation_Hessian_RMS_cm_per_rad2=validation_rms(slice(1+q.shape[1],None)))
        audit['identified_coefficients']=audit['rank']==audit['raw_feature_count']
        require_identified=r.get('workflow_id')=='coupled-local-torsions.v1'
        audit['qualified_energy']=(audit['validation_energy_max_cm']<=gate
            and (not require_identified or audit['identified_coefficients']))
        models.append(model);candidates.append(audit)
    passing=[i for i,v in enumerate(candidates) if v['qualified_energy']]
    out=(path.parent/r.get('output_directory','periodic_surface')).resolve();out.mkdir(parents=True,exist_ok=True)
    if not passing:
        (out/'analytic_surface_result.json').write_text(json.dumps(dict(status='fit_not_qualified',candidates=candidates,new_QM_jobs=0),indent=2)+'\n')
        raise ValueError('No declared periodic model passes the frozen energy gate; no electronic refinement is launched')
    chosen=min(passing,key=lambda i:(candidates[i]['raw_feature_count'],candidates[i]['validation_energy_max_cm']))
    modelpath=out/'periodic_surface.json';modelpath.write_text(json.dumps(models[chosen].to_dict(),indent=2)+'\n')
    report=dict(schema='matrix.dvr.analytic_surface_result.v1',policy_id=POLICY_ID,policy_sha256=POLICY_SHA256,request_sha256=hashlib.sha256(raw).hexdigest(),sample_count=len(q),sample_budget=r['sample_budget'],validation_indices=ids.tolist(),candidates=candidates,selected_index=chosen,selected=candidates[chosen],family='periodic_hermite',status='fit_validated',qualification_scope='Declared electronic energy gate only; derivatives reported separately, no DVR convergence or unobserved collision/dissociation limit claimed',grid_source='frozen_analytic_function_only',QM_evaluations_for_generated_grid=0,artifact=str(modelpath),artifact_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest())
    target=out/'analytic_surface_result.json';target.write_text(json.dumps(report,indent=2)+'\n');return target


def periodic_surface_from_dict(data):
    """Deserialize a declared angular scalar without changing its model."""
    schema=data.get('schema')
    if schema=='matrix.dvr.affine_periodic_hermite_surface.v1':
        from .affine_periodic import AffinePeriodicHermiteSurface
        return AffinePeriodicHermiteSurface.from_dict(data)
    if schema=='matrix.dvr.kernel_corrected_periodic_surface.v1':
        from .periodic_hermite_kernel import KernelCorrectedPeriodicSurface
        return KernelCorrectedPeriodicSurface.from_dict(data)
    if schema=='matrix.dvr.relaxed_quadratic_bath_surface.v1':
        from .periodic_surface_correction import RelaxedQuadraticBathSurface
        return RelaxedQuadraticBathSurface.from_dict(data)
    if schema=='matrix.dvr.hermite_corrected_periodic_surface.v1':
        from .periodic_surface_correction import HermiteCorrectedPeriodicSurface
        return HermiteCorrectedPeriodicSurface.from_dict(data)
    if schema=='matrix.dvr.periodic_valley_surface.v1':
        from .periodic_valley import PeriodicValleySurface
        return PeriodicValleySurface.from_dict(data)
    if schema=='matrix.dvr.positive_periodic_surface.v1':
        from .positive_periodic import PositivePeriodicSurface
        return PositivePeriodicSurface.from_dict(data)
    if schema=='matrix.dvr.damped_periodic_hermite_surface.v1':
        from .damped_periodic import DampedPeriodicHermiteSurface
        return DampedPeriodicHermiteSurface.from_dict(data)
    return PeriodicHermiteSurface.from_dict(data)
