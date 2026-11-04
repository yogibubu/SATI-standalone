"""Sparse energy/torque fits with an exact reference angular two-jet.

The existing periodic Hermite features are trigonometric polynomials with
zero value, gradient and Hessian at the reference. The default degree-three
Fourier space has four free coefficients after fixing that two-jet; a declared
higher degree requires enough independent E/G observations. Missing angular
Hessians are never filled with synthetic observations.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

from .periodic_hermite import PeriodicHermiteSurface, angular_features
from .analytic_fit_policy import validate_sample_plan, POLICY_ID, POLICY_SHA256


@dataclass(frozen=True)
class AnchoredPeriodicEG:
    surface: PeriodicHermiteSurface
    reference_energy_cm: float
    reference_torque_cm: float

    def evaluate(self, angles, *, derivative=0):
        angles=np.asarray(angles,float)
        if derivative not in (0,1,2):raise ValueError('angular derivative order 0, 1 or 2 required')
        baseline=(self.reference_energy_cm+self.reference_torque_cm*np.sin(angles) if derivative==0
                  else self.reference_torque_cm*np.cos(angles) if derivative==1
                  else -self.reference_torque_cm*np.sin(angles))
        return baseline+self.surface.evaluate(angles[...,None],(derivative,))

    def to_dict(self):
        return dict(schema='matrix.dvr.anchored_periodic_eg.v1',surface=self.surface.to_dict(),
            reference_energy_cm=float(self.reference_energy_cm),reference_torque_cm=float(self.reference_torque_cm))

    @classmethod
    def from_dict(cls, value):
        if value.get('schema')!='matrix.dvr.anchored_periodic_eg.v1':raise ValueError('unsupported anchored periodic model')
        return cls(PeriodicHermiteSurface.from_dict(value['surface']),
            float(value['reference_energy_cm']),float(value['reference_torque_cm']))


def fit_anchored_periodic_eg(angles, energies_cm, torques_cm, *, reference_energy_cm,
        reference_torque_cm, reference_curvature_cm, derivative_length_radian=.3,fourier_degree=3,
        reflection_even=False):
    if isinstance(fourier_degree,(bool,np.bool_)) or not isinstance(fourier_degree,(int,np.integer)) or fourier_degree<2:
        raise ValueError('integer Fourier degree of at least two required')
    q=np.asarray(angles,float);e=np.asarray(energies_cm,float)
    g=None if torques_cm is None else np.asarray(torques_cm,float)
    anchors=np.asarray([reference_energy_cm,reference_torque_cm,reference_curvature_cm],float)
    if (q.ndim!=1 or len(q)<3 or e.shape!=q.shape or (g is not None and g.shape!=q.shape)
            or not all(np.all(np.isfinite(a)) for a in (q,e,anchors))
            or (g is not None and not np.all(np.isfinite(g)))
            or not np.isfinite(derivative_length_radian) or derivative_length_radian<=0):
        raise ValueError('finite matching angular energies/torques and reference two-jet required')
    powers=np.arange(4 if reflection_even else 3,2*fourier_degree+1,2 if reflection_even else 1)[:,None];count=len(powers)
    baseline=AnchoredPeriodicEG(PeriodicHermiteSurface(np.array([[reference_curvature_cm]]),
        powers,np.zeros(count),np.ones(1),np.array([np.pi]),False),reference_energy_cm,reference_torque_cm)
    a=angular_features(q[:,None],powers,np.ones(1))
    y=e-baseline.evaluate(q)
    if g is not None:
        a=np.vstack((a,derivative_length_radian*angular_features(q[:,None],powers,np.ones(1),[1])))
        y=np.r_[y,derivative_length_radian*(g-baseline.evaluate(q,derivative=1))]
    scale=np.linalg.norm(a,axis=0)
    if np.any(scale==0):raise ValueError('unidentified periodic coefficient')
    coefficients,residual,rank,singular=np.linalg.lstsq(a/scale,y,rcond=1.e-12)
    if rank!=count:raise ValueError('insufficient independent E/G observations for declared anchored Fourier degree')
    model=AnchoredPeriodicEG(PeriodicHermiteSurface(np.array([[reference_curvature_cm]]),powers,
        coefficients/scale,np.ones(1),np.array([np.pi]),False),reference_energy_cm,reference_torque_cm)
    return model,dict(fourier_degree=int(fourier_degree),free_coefficients=count,identified_rank=int(rank),
        scaled_design_condition=float(singular[0]/singular[-1]),
        energy_training_max_error_cm=float(max(abs(model.evaluate(q)-e))),
        residual_reflection_even=bool(reflection_even),torque_observations_supplied=g is not None,
        torque_training_max_error_cm_per_rad=(None if g is None else float(max(abs(model.evaluate(q,derivative=1)-g)))))


def run_anchored_periodic_eg_request(request_path):
    path=Path(request_path).resolve();raw=path.read_bytes();r=json.loads(raw)
    if r.get('schema')!='matrix.dvr.analytic_surface_request.v1' or r.get('family')!='anchored_periodic_eg':
        raise ValueError('typed anchored periodic E/G request required')
    angles=np.asarray(r['angles_radian'],float);energies=np.asarray(r['energies_cm'],float)
    torques=None if r.get('torques_cm_per_rad') is None else np.asarray(r['torques_cm_per_rad'],float)
    validate_sample_plan(sample_count=len(angles),sample_budget=r['sample_budget'],adaptive_qm=r.get('adaptive_qm',False))
    ids=r['validation_indices']
    if (not ids or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(angles) for i in ids)
            or len(set(ids))!=len(ids)):raise ValueError('distinct held-out angular indices required')
    mask=np.ones(len(angles),bool);mask[ids]=False
    gate=float(r['max_validation_error_cm'])
    if not np.isfinite(gate) or gate<=0:raise ValueError('positive energy validation gate required')
    model,audit=fit_anchored_periodic_eg(angles[mask],energies[mask],None if torques is None else torques[mask],
        reference_energy_cm=r['reference_energy_cm'],reference_torque_cm=r['reference_torque_cm_per_rad'],
        reference_curvature_cm=r['reference_curvature_cm_per_rad2'],
        derivative_length_radian=r.get('derivative_length_radian',.3),fourier_degree=r.get('fourier_degree',3),
        reflection_even=r.get('reflection_even',False))
    validation=float(max(abs(model.evaluate(angles[~mask])-energies[~mask])))
    qualified=validation<=gate and audit['energy_training_max_error_cm']<=gate
    out=path.parent/r.get('output_directory','anchored_periodic_fit');out.mkdir(parents=True,exist_ok=True)
    artifact=out/'periodic_surface.json'
    artifact.write_text(json.dumps(model.to_dict(),indent=2)+'\n')
    result=dict(schema='matrix.dvr.analytic_surface_result.v1',policy_id=POLICY_ID,policy_sha256=POLICY_SHA256,
        request_sha256=hashlib.sha256(raw).hexdigest(),family='anchored_periodic_eg',sample_count=len(angles),
        sample_budget=r['sample_budget'],validation_indices=ids,validation_max_error_cm=validation,
        reference_twojet_exact=True,held_out_geometries_excluded_from_fit=True,fit_audit=audit,
        status='fit_validated' if qualified else 'fit_not_qualified',artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),grid_source='analytic_function_only',
        QM_evaluations_for_generated_grid=0)
    target=out/'analytic_surface_result.json';target.write_text(json.dumps(result,indent=2)+'\n')
    if not qualified:raise ValueError('anchored periodic E/G fit fails frozen energy gate; no new source points')
    return target
