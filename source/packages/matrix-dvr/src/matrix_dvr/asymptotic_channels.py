"""Frozen-data Hermitian channel surfaces with independently fixed limits.

This extends the resident exponential two-channel analytic family to matrices.
A common coordinate map is fitted to ALL entries, not to sorted adiabatic
energies. Held-out geometries exclude all their matrix observations. The
owner has no PES/acquisition callback; generated nodes only evaluate the fit.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .analytic_surface import validate_sample_plan, POLICY_ID, POLICY_SHA256
from .asymptotic_surface import AsymptoticBarrierSurface


def _hermitian(value, name):
    a=np.array(value,complex,copy=True)
    if (a.ndim<2 or a.shape[-1]!=a.shape[-2] or a.shape[-1]<1
            or not np.all(np.isfinite(a))
            or not np.allclose(a,a.swapaxes(-1,-2).conj(),atol=1e-9,rtol=0)):
        raise ValueError(name+': finite Hermitian channel matrices required')
    return (a+a.swapaxes(-1,-2).conj())/2


@dataclass(frozen=True)
class AsymptoticChannelSurface:
    coordinate_map: AsymptoticBarrierSurface
    coefficients_cm: np.ndarray
    left_matrix_cm: np.ndarray
    right_matrix_cm: np.ndarray
    basis_id: str

    def __post_init__(self):
        left=_hermitian(self.left_matrix_cm,'left limit')
        right=_hermitian(self.right_matrix_cm,'right limit')
        c=_hermitian(self.coefficients_cm,'coefficients')
        if (left.ndim!=2 or right.shape!=left.shape or c.ndim!=3
                or c.shape[1:]!=left.shape or len(c)!=len(self.coordinate_map.coefficients)
                or not isinstance(self.basis_id,str) or not self.basis_id.strip()):
            raise ValueError('matching limits, coefficient matrices and fixed transverse basis identity required')
        for key,value in [('coefficients_cm',c),('left_matrix_cm',left),('right_matrix_cm',right)]:
            value.setflags(write=False);object.__setattr__(self,key,value)

    def evaluate(self,coordinates):
        # Resident scalar family returns the exact smoothstep and envelope.
        m=self.coordinate_map
        switch=AsymptoticBarrierSurface(np.zeros(len(m.coefficients)),m.center,m.inverse_length,
            0.,1.,m.sampled_domain,m.asymmetry)
        parent,basis=switch.design(coordinates)
        value=(self.left_matrix_cm+np.asarray(parent)[...,None,None]
            *(self.right_matrix_cm-self.left_matrix_cm)
            +np.einsum('...k,kij->...ij',basis,self.coefficients_cm))
        return value.reshape(self.left_matrix_cm.shape) if np.ndim(coordinates)==0 else value


def fit_asymptotic_channels(coordinates, matrices_cm, *, left_matrix_cm,
        right_matrix_cm, basis_id, degree, sample_budget, validation_indices,
        max_validation_error_cm=10.,slopes=None,gradient_length_scale=.1):
    x=np.asarray(coordinates,float);y=_hermitian(matrices_cm,'samples')
    left=_hermitian(left_matrix_cm,'left limit');right=_hermitian(right_matrix_cm,'right limit')
    validate_sample_plan(sample_count=len(x),sample_budget=sample_budget)
    ids=np.asarray(validation_indices)
    if (x.ndim!=1 or len(x)<6 or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0)
            or y.ndim!=3 or len(y)!=len(x) or left.shape!=y.shape[1:] or right.shape!=left.shape
            or isinstance(degree,bool) or not isinstance(degree,int) or degree<1
            or ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids))!=len(ids) or np.any(ids<=0) or np.any(ids>=len(x)-1)
            or not np.isfinite(max_validation_error_cm) or max_validation_error_cm<=0):
        raise ValueError('ordered samples, matching matrices, interior held-outs and positive controls required')
    dy=None if slopes is None else _hermitian(slopes,'analytic slopes')
    if dy is not None and dy.shape!=y.shape:raise ValueError('matrix slope shape must match observations')
    if not np.isfinite(gradient_length_scale) or gradient_length_scale<=0:raise ValueError('positive gradient length scale required')
    train=np.ones(len(x),bool);train[ids]=False
    if train.sum()*(1 if dy is None else 2)<=degree+4:raise ValueError('insufficient independent training geometries for matrix fit')
    span=x[-1]-x[0];domain=(float(x[0]),float(x[-1]))
    def solve(mask, initial=None):
        xx=x[mask];yy=y[mask]
        # Starting geometry selection also excludes every held-out value.
        norm=np.linalg.norm(yy-(left+right)/2,axis=(1,2))
        center0=float(xx[np.argmax(norm)])
        def project(par):
            chart=AsymptoticBarrierSurface(np.zeros(degree+1),float(par[0]),float(np.exp(par[1])),0.,1.,domain,float(par[2]))
            switch,basis=chart.design(xx)
            parent=left+switch[:,None,None]*(right-left)
            rhs=(yy-parent).reshape(len(xx),-1)
            if dy is not None:
                dswitch,dbasis=chart.derivative_design(xx)
                drhs=dy[mask]-dswitch[:,None,None]*(right-left)
                basis=np.vstack([basis,gradient_length_scale*dbasis])
                rhs=np.vstack([rhs,gradient_length_scale*drhs.reshape(len(xx),-1)])
            # Exact structural constants have zero residual for every shape.
            # Excluding these zero columns from the linear solve is algebraic,
            # not a cutoff on weak physical couplings. Held-outs retain ALL entries.
            active=np.any(yy.reshape(len(xx),-1)!=left.ravel(),axis=0)|(right.ravel()!=left.ravel())
            if dy is not None:active|=np.any(dy[mask].reshape(len(xx),-1)!=0,axis=0)
            c=np.zeros((degree+1,rhs.shape[1]),complex)
            solved,_,rank,_=np.linalg.lstsq(basis,rhs[:,active],rcond=1e-12)
            c[:,active]=solved
            residual=basis@solved-rhs[:,active]
            vector=np.r_[residual.real.ravel(),residual.imag.ravel()]
            return chart,c.reshape((degree+1,)+left.shape),rank,vector
        trials=[]
        starts=[[center0,np.log(width/span),skew] for width in [1.,2.,4.,8.] for skew in [-.6,0.,.6]]
        if initial is not None:
            m=initial.coordinate_map;starts.insert(0,[m.center,np.log(m.inverse_length),m.asymmetry])
        for start in starts:
            result=least_squares(lambda p:project(p)[3],start,
                bounds=([x[0],np.log(.1/span),-.95],[x[-1],np.log(100/span),.95]),
                max_nfev=1000,ftol=1e-11,xtol=1e-11,gtol=1e-10)
            chart,c,rank,residual=project(result.x)
            if result.success and rank==degree+1:
                trials.append((float(residual@residual),chart,c))
        if not trials:raise ValueError('matrix asymptotic fit did not converge at full rank')
        _,chart,c=min(trials,key=lambda p:p[0])
        return AsymptoticChannelSurface(chart,c,left,right,basis_id)
    candidate=solve(train)
    held=np.linalg.norm(candidate.evaluate(x[ids])-y[ids],ord=2,axis=(1,2))
    if max(held)>max_validation_error_cm:
        raise ValueError(f'matrix analytic fit fails held-out operator error: {max(held):.6g} cm^-1; no acquisition launched')
    model=solve(np.ones(len(x),bool),initial=candidate)
    residual=np.linalg.norm(model.evaluate(x)-y,ord=2,axis=(1,2))
    if max(residual)>max_validation_error_cm:
        raise ValueError('final matrix analytic fit fails its sample operator residual gate')
    return model,dict(sample_count=len(x),sample_budget=sample_budget,degree=degree,
        validation_indices=ids.tolist(),held_out_operator_errors_cm=held.tolist(),
        maximum_validation_operator_error_cm=float(max(held)),maximum_fit_operator_error_cm=float(max(residual)),
        common_shape_uses_all_matrix_entries=True,fit_validated=True,native_gradients_used=dy is not None,
        error_metric='spectral operator norm; bounds every instantaneous channel energy error',
        exact_hermiticity=True,exact_independent_asymptotes=True,
        grid_source='analytic_function_only',QM_evaluations_for_generated_grid=0,
        policy_id=POLICY_ID,policy_sha256=POLICY_SHA256)


def run_asymptotic_channel_request(request_path):
    import json,hashlib
    from pathlib import Path
    path=Path(request_path).resolve();raw=path.read_bytes();r=json.loads(raw)
    if (r.get('schema')!='matrix.dvr.analytic_surface_request.v1'
            or r.get('family')!='asymptotic_channel_matrix'
            or not r.get('state_identity') or not r.get('transverse_basis_id')
            or r.get('kinetic_coordinate')!='constant_mass_weighted'
            or r.get('coordinate_units')!='bohr_sqrt_amu'
            or r.get('basis_representation')!='fixed_transverse_wavefunctions'
            or r.get('transverse_kinetic_included') is not True):
        raise ValueError('typed matrix request with state, fixed basis and constant mass-weighted coordinate required')
    if r.get('adaptive_qm',False):raise ValueError('automatic electronic refinement is forbidden')
    x=np.asarray(r['training_coordinates'],float)
    if x.ndim==2 and x.shape[1]==1:x=x[:,0]
    model,control=fit_asymptotic_channels(x,r['channel_matrices_cm'],left_matrix_cm=r['left_limit_matrix_cm'],
        right_matrix_cm=r['right_limit_matrix_cm'],basis_id=r['transverse_basis_id'],degree=r['degree'],
        sample_budget=r['sample_budget'],validation_indices=r['validation_indices'],
        max_validation_error_cm=r['max_validation_error_cm'])
    if len(r['grid_axes'])!=1:raise ValueError('one reactive grid axis required')
    axis=np.asarray(r['grid_axes'][0],float)
    if axis.ndim!=1 or len(axis)<5 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis)<=0):
        raise ValueError('ordered finite generated grid required')
    if (axis[0]<x[0] or axis[-1]>x[-1]) and r.get('asymptotic_extrapolation') is not True:
        raise ValueError('explicit asymptotic extrapolation declaration required')
    out=(path.parent/r.get('output_directory','analytic_channels')).resolve();out.mkdir(parents=True,exist_ok=True)
    artifact=out/'analytic_dvr_grid.npz';m=model.coordinate_map
    np.savez_compressed(artifact,coordinates=axis,potential_cm=model.evaluate(axis),
        coefficients_cm=model.coefficients_cm,left_matrix_cm=model.left_matrix_cm,right_matrix_cm=model.right_matrix_cm,
        center=m.center,inverse_length=m.inverse_length,asymmetry=m.asymmetry,sampled_domain=m.sampled_domain,
        basis_id=model.basis_id)
    result=dict(schema='matrix.dvr.analytic_surface_result.v1',family=r['family'],**control,
        state_identity=r['state_identity'],transverse_basis_id=model.basis_id,
        kinetic_coordinate=r['kinetic_coordinate'],coordinate_units=r['coordinate_units'],
        basis_representation=r['basis_representation'],transverse_kinetic_included=True,request_sha256=hashlib.sha256(raw).hexdigest(),
        artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),status='fit_validated')
    target=out/'analytic_surface_result.json';target.write_text(json.dumps(result,indent=2)+'\n')
    return target
