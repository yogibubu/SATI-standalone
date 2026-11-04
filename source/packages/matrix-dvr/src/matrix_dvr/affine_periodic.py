"""Hermite fits invariant under a declared affine angular reflection.

The symmetry is q -> offset-q modulo 2*pi. Group-projected periodic features
and exact linear E/G/H anchors retain the physical origin, even when reflection
maps it onto another equivalent labelled configuration. No QM callback exists.
"""
from dataclasses import dataclass
import hashlib
import json
import itertools
from pathlib import Path

import numpy as np

from .analytic_fit_policy import validate_sample_plan,POLICY_ID,POLICY_SHA256
from .periodic_hermite import angular_features,derivative_orders


@dataclass(frozen=True)
class AffinePeriodicHermiteSurface:
    powers: np.ndarray
    coefficients: np.ndarray
    scales: np.ndarray
    reflection_offset_radian: np.ndarray
    sampled_extent_radian: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value=np.array(getattr(self,name),dtype=int if name=='powers' else float,copy=True)
            if not np.all(np.isfinite(value)):raise ValueError('Finite affine periodic field required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        d=len(self.scales)
        if (self.powers.ndim!=2 or self.powers.shape[1]!=d or np.any(self.powers<0)
            or self.coefficients.shape!=(len(self.powers),) or self.reflection_offset_radian.shape!=(d,)
            or self.sampled_extent_radian.shape!=(d,) or np.any(self.scales<=0) or np.any(self.sampled_extent_radian<=0)):
            raise ValueError('Compatible periodic powers, scales and reflection offset required')

    def features(self,points,derivative=None):
        order=np.zeros(len(self.scales),int) if derivative is None else np.asarray(derivative,int)
        q=np.asarray(points,float)
        return .5*(angular_features(q,self.powers,self.scales,order)
            +(-1)**int(sum(order))*angular_features(self.reflection_offset_radian-q,self.powers,self.scales,order))

    def evaluate(self,points,derivative=None,*,allow_extrapolation=False):
        q=np.asarray(points,float);wrapped=np.angle(np.exp(1j*q))
        if not allow_extrapolation and np.any(abs(wrapped)>self.sampled_extent_radian+1e-10):
            raise ValueError('Outside the declared reflected source envelope')
        return self.features(q,derivative)@self.coefficients

    def to_dict(self):
        return dict(schema='matrix.dvr.affine_periodic_hermite_surface.v1',
            **{name:getattr(self,name).tolist() for name in self.__dataclass_fields__},
            protocol_id=POLICY_ID,symmetry='q -> reflection_offset_radian-q modulo 2*pi',
            qualification='Source envelope only; no DVR/domain/asymptotic qualification implied')

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.dvr.affine_periodic_hermite_surface.v1':raise ValueError('Unsupported affine periodic schema')
        return cls(**{name:np.asarray(data[name]) for name in cls.__dataclass_fields__})


def fit_affine_periodic_hermite(q,y,reference_hessian,*,powers,scales,reflection_offset_radian,
                              sample_budget,weights=None,derivative_length=.15,rcond=1e-10,ridge=0.,reference_cubic=None,
                              observation_mask=None,reference_energy=0.,reference_gradient=None):
    """Fit a periodic Hermite field retaining the actual reference E/G/H.

    Defaults preserve the stationary zero-energy anchor. Supplying a residual
    reference gradient does not alter observations or assume stationarity.
    """
    q=np.asarray(q,float);y=np.asarray(y,float);h=np.asarray(reference_hessian,float)
    validate_sample_plan(sample_count=len(q),sample_budget=sample_budget)
    d=q.shape[1];orders=derivative_orders(d);powers=np.asarray(powers,int);offset=np.asarray(reflection_offset_radian,float)
    if y.shape!=(len(orders),len(q)) or h.shape!=(d,d) or derivative_length<=0 or not 0<rcond<1 or ridge<0:
        raise ValueError('Compatible Hermite observations, Hessian and solve controls required')
    observed=np.ones(y.shape,bool) if observation_mask is None else np.asarray(observation_mask,bool)
    if observed.shape!=y.shape or not np.any(observed) or not np.all(observed[0]):
        raise ValueError('Explicit derivative mask must retain every source energy')
    if not np.allclose(h,h.T,atol=1e-10) or not all(np.all(np.isfinite(x)) for x in (q,h)) or not np.all(np.isfinite(y[observed])):
        raise ValueError('Finite observations and symmetric reference Hessian required')
    reflected=np.angle(np.exp(1j*(offset-q)));extent=np.maximum(np.max(abs(q),axis=0),np.max(abs(reflected),axis=0))
    model=AffinePeriodicHermiteSurface(powers,np.zeros(len(powers)),np.asarray(scales),offset,extent)
    anchor_orders=list(orders)
    gradient=np.zeros(d) if reference_gradient is None else np.asarray(reference_gradient,float)
    if gradient.shape!=(d,) or not np.all(np.isfinite(gradient)) or not np.isfinite(reference_energy):
        raise ValueError('Finite reference energy and compatible reference gradient required')
    target=np.array([reference_energy,*gradient,*[h[i,j] for i in range(d) for j in range(i,d)]])
    if reference_cubic is not None:
        cubic=np.asarray(reference_cubic,float)
        if cubic.shape!=(d,d,d) or not np.all(np.isfinite(cubic)) or not np.allclose(cubic,cubic.transpose(1,0,2)) or not np.allclose(cubic,cubic.transpose(2,1,0)):
            raise ValueError('Finite symmetric reference cubic tensor required')
        triples=list(itertools.combinations_with_replacement(range(d),3))
        anchor_orders.extend(tuple(sum(np.eye(d,dtype=int)[i] for i in triple)) for triple in triples)
        target=np.r_[target,[cubic[triple] for triple in triples]]
    C=np.array([model.features(np.zeros((1,d)),o)[0] for o in anchor_orders])
    U,s,Vt=np.linalg.svd(C,full_matrices=True);rank=int(np.sum(s>rcond*s[0]))
    particular=Vt[:rank].T@((U[:,:rank].T@target)/s[:rank]);null=Vt[rank:].T
    anchor_error=float(abs(C@particular-target).max())
    if anchor_error>max(1e-7,1e-10*abs(target).max()):raise ValueError('Features cannot satisfy exact reference E/G/H')
    factors=np.array([derivative_length**sum(o) for o in orders]);w=np.ones(len(q)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(q),) or np.any(w<=0) or not np.all(np.isfinite(w)):raise ValueError('Positive finite sample weights required')
    A=np.array([model.features(q,o) for o in orders]);A=(A*factors[:,None,None]*w[None,:,None]).reshape(-1,len(powers))[observed.ravel()]
    rhs=(y*factors[:,None]*w).ravel()[observed.ravel()]-A@particular
    if null.shape[1]:
        B=A@null;u,t,v=np.linalg.svd(B,full_matrices=False)
        # A symmetry-projected basis can leave only roundoff in a nominal
        # null-space column. Do not promote that noise into a free fit mode.
        keep=t>rcond*max(t[0],np.linalg.norm(A,2))
        correction=v[keep].T@((t[keep]/(t[keep]**2+ridge*t[0]**2))*(u[:,keep].T@rhs))
        coefficients=particular+null@correction;identified=int(sum(keep))
    else:coefficients=particular;identified=0
    result=AffinePeriodicHermiteSurface(powers,coefficients,np.asarray(scales),offset,extent)
    final_anchor_error=float(abs(C@coefficients-target).max())
    if final_anchor_error>max(1e-7,1e-10*abs(target).max()):
        raise ValueError('Fitted coefficients lost the exact reference E/G/H anchor')
    error=np.array([result.evaluate(q,o) for o in orders])-y
    def rms(start,stop):
        selected=error[start:stop][observed[start:stop]]
        return float(np.sqrt(np.mean(selected**2))) if len(selected) else None
    return result,dict(raw_feature_count=len(powers),anchor_rank=rank,identified_fit_rank=identified,
        reference_anchor_max_error=final_anchor_error,ridge=ridge,rcond=rcond,
        training_energy_RMS_cm=rms(0,1),
        training_gradient_RMS_cm_per_rad=rms(1,1+d),
        training_Hessian_RMS_cm_per_rad2=rms(1+d,len(orders)),
        observed_derivative_rows=int(np.sum(observed[1:])),missing_derivatives_are_observations=False)


def run_affine_periodic_request(path):
    path=Path(path).resolve();raw=path.read_bytes();r=json.loads(raw)
    if r.get('schema')!='matrix.dvr.analytic_surface_request.v1' or r.get('family')!='affine_periodic_hermite':raise ValueError('Typed affine periodic request required')
    if r.get('coordinate_unit')!='radian' or r.get('energy_unit')!='cm^-1':raise ValueError('Explicit radian/cm^-1 units required')
    q=np.asarray(r['coordinates_radian'],float);y=np.asarray(r['observations_E_G_H'],float)
    observed=np.ones(y.shape,bool) if 'observation_mask' not in r else np.asarray(r['observation_mask'],bool)
    if observed.shape!=y.shape or not np.all(observed[0]):raise ValueError('Matching derivative mask retaining all source energies required')
    validate_sample_plan(sample_count=len(q),sample_budget=r['sample_budget'],adaptive_qm=r.get('adaptive_qm',False))
    indices=np.asarray(r['validation_indices']);mask=np.zeros(len(q),bool)
    if indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices) or len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=len(q)):
        raise ValueError('Distinct held-out indices required')
    mask[indices]=True;weights=np.asarray(r.get('geometry_weights',np.ones(len(q))),float)
    if mask.all():raise ValueError('Training observations required')
    gate=float(r['max_validation_error_cm'])
    if not np.isfinite(gate) or gate<=0:raise ValueError('Positive finite validation gate required')
    models=[];audits=[];orders=derivative_orders(q.shape[1])
    for option in r['candidates']:
        model,audit=fit_affine_periodic_hermite(q[~mask],y[:,~mask],r['reference_hessian_cm_per_rad2'],
            powers=option['powers'],scales=r['feature_scales_radian'],reflection_offset_radian=r['reflection_offset_radian'],
            sample_budget=r['sample_budget'],weights=weights[~mask],derivative_length=r.get('derivative_length_radian',.15),
            rcond=option.get('rcond',1e-10),ridge=option.get('ridge',0.),reference_cubic=r.get('reference_cubic_cm_per_rad3'),
            observation_mask=observed[:,~mask],reference_energy=r.get('reference_energy_cm',0.),
            reference_gradient=r.get('reference_gradient_cm_per_rad'))
        error=np.array([model.evaluate(q,o,allow_extrapolation=True) for o in orders])-y
        def validation_rms(start,stop):
            values=error[start:stop,mask][observed[start:stop,mask]]
            return float(np.sqrt(np.mean(values**2))) if len(values) else None
        audit.update(name=option['name'],validation_energy_errors_cm=error[0,mask].tolist(),validation_energy_max_cm=float(abs(error[0,mask]).max()),
            validation_gradient_RMS_cm_per_rad=validation_rms(1,1+q.shape[1]),
            validation_Hessian_RMS_cm_per_rad2=validation_rms(1+q.shape[1],len(orders)))
        audit['qualified_energy']=audit['validation_energy_max_cm']<=gate
        if 'periodic_shape_gate' in r:
            shape=r['periodic_shape_gate']
            if q.shape[1]!=1 or shape.get('minimum_count')!=2:
                raise ValueError('Periodic double-well shape gate requires one angular coordinate')
            nodes=shape.get('nodes',4096);floor=float(shape['minimum_energy_floor_cm'])
            if isinstance(nodes,bool) or not isinstance(nodes,int) or nodes<1024 or not np.isfinite(floor):
                raise ValueError('Finite shape floor and at least 1024 periodic audit nodes required')
            axis=np.linspace(-np.pi,np.pi,nodes,endpoint=False)+float(r['reflection_offset_radian'][0])/2
            potential=model.evaluate(axis[:,None],allow_extrapolation=True)
            minima=np.flatnonzero((potential<np.roll(potential,1))&(potential<np.roll(potential,-1)))
            audit['periodic_shape_audit']=dict(nodes=nodes,minimum_count=len(minima),
                minimum_energy_cm=float(potential.min()),maximum_energy_cm=float(potential.max()),
                minima_coordinates_radian=axis[minima].tolist(),minima_energies_cm=potential[minima].tolist(),
                qualification='sampled full-period shape check; not electronic validation outside the source envelope')
            audit['qualified_shape']=bool(len(minima)==2 and potential.min()>=floor)
        else:audit['qualified_shape']=True
        models.append(model);audits.append(audit)
    passing=[i for i,a in enumerate(audits) if a['qualified_energy'] and a['qualified_shape']]
    out=(path.parent/r.get('output_directory','affine_periodic_surface')).resolve();out.mkdir(parents=True,exist_ok=True)
    report=dict(schema='matrix.dvr.analytic_surface_result.v1',policy_id=POLICY_ID,policy_sha256=POLICY_SHA256,
        request_sha256=hashlib.sha256(raw).hexdigest(),sample_budget=r['sample_budget'],sample_count=len(q),validation_indices=indices.tolist(),
        family=r['family'],candidates=audits,new_QM_jobs=0,status='fit_validated' if passing else 'fit_not_qualified',
        qualification_scope='Declared energy and optional periodic shape gates; no nuclear-motion, transverse-field or electronic extrapolation qualification')
    if passing:
        chosen=min(passing,key=lambda i:(audits[i]['raw_feature_count'],audits[i]['validation_energy_max_cm']))
        artifact=out/'periodic_surface.json';artifact.write_text(json.dumps(models[chosen].to_dict(),indent=2)+'\n')
        report.update(selected_index=chosen,selected=audits[chosen],artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    target=out/'analytic_surface_result.json';target.write_text(json.dumps(report,indent=2)+'\n')
    if not passing:raise ValueError('No affine periodic model passes the declared energy/shape gates; no electronic jobs launched')
    return target
