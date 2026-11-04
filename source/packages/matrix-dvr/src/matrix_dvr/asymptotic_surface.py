"""Finite-data analytic barriers with exact independently supplied limits.

The fit owner has no PES/QM callback. All DVR/propagation nodes are evaluated
from frozen coefficients. Endpoint limits constrain the function, but do not
certify the unobserved remote tail or multidimensional molecular dynamics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from .analytic_surface import validate_sample_plan,POLICY_ID,POLICY_SHA256


@dataclass(frozen=True)
class AsymptoticBarrierSurface:
    coefficients: np.ndarray
    center: float
    inverse_length: float
    left_limit_cm: float
    right_limit_cm: float
    sampled_domain: tuple
    asymmetry: float=0.

    def __post_init__(self):
        c=np.array(self.coefficients,dtype=float,copy=True)
        if (c.ndim!=1 or not len(c) or not np.all(np.isfinite(c))
                or not np.all(np.isfinite([self.center,self.inverse_length,self.left_limit_cm,self.right_limit_cm,self.asymmetry]))
                or self.inverse_length<=0 or abs(self.asymmetry)>=1
                or len(self.sampled_domain)!=2 or not np.all(np.isfinite(self.sampled_domain))
                or self.sampled_domain[0]>=self.sampled_domain[1]):
            raise ValueError('finite coefficients, positive length scale, |asymmetry|<1 and ordered domain required')
        c.setflags(write=False);object.__setattr__(self,'coefficients',c)

    def design(self,coordinates):
        x=np.asarray(coordinates,float)
        if not np.all(np.isfinite(x)):raise ValueError('finite reaction coordinates required')
        z=self.inverse_length*(x-self.center)
        t=expit(z+self.asymmetry*(np.logaddexp(z,-z)-np.log(2.)));u=2*t-1
        # Cubic switching and envelope have exponential tails on both ends.
        # Unlike a quartic Taylor polynomial they cannot generate divergent
        # remote energies. The two separated-channel limits remain exact.
        switch=t*t*(3-2*t)
        parent=self.left_limit_cm+(self.right_limit_cm-self.left_limit_cm)*switch
        basis=(t*(1-t))[...,None]*np.polynomial.chebyshev.chebvander(u,len(self.coefficients)-1)
        return parent,basis

    def evaluate(self,coordinates):
        parent,basis=self.design(coordinates)
        value=parent+basis@self.coefficients
        return float(np.asarray(value).ravel()[0]) if np.ndim(coordinates)==0 else value

    def derivative_design(self,coordinates):
        x=np.asarray(coordinates,float);z=self.inverse_length*(x-self.center)
        if not np.all(np.isfinite(x)):raise ValueError('finite reaction coordinates required')
        t=expit(z+self.asymmetry*(np.logaddexp(z,-z)-np.log(2.)));u=2*t-1
        dt=t*(1-t)*self.inverse_length*(1+self.asymmetry*np.tanh(z))
        table=np.polynomial.chebyshev.chebvander(u,len(self.coefficients)-1)
        derivative=np.moveaxis(np.polynomial.chebyshev.chebval(u,
            np.polynomial.chebyshev.chebder(np.eye(len(self.coefficients)),axis=0)),0,-1)
        parent=(self.right_limit_cm-self.left_limit_cm)*6*t*(1-t)*dt
        basis=dt[...,None]*((1-2*t)[...,None]*table+2*(t*(1-t))[...,None]*derivative)
        return parent,basis

    def derivative(self,coordinates):
        parent,basis=self.derivative_design(coordinates)
        value=parent+basis@self.coefficients
        return float(np.asarray(value).ravel()[0]) if np.ndim(coordinates)==0 else value

    def metadata(self):
        return dict(schema='matrix.dvr.asymptotic_barrier.v1',family='exponential_two_channel_barrier',
            coefficients_cm=self.coefficients.tolist(),center=self.center,inverse_length=self.inverse_length,
            asymmetry=self.asymmetry,
            left_limit_cm=self.left_limit_cm,right_limit_cm=self.right_limit_cm,
            sampled_domain=list(self.sampled_domain),
            formula='z=a*(s-s0); t=expit(z+b*log(cosh(z))); V=EL+(ER-EL)*t^2*(3-2*t)+t*(1-t)*sum(c_k*T_k(2*t-1))',
            remote_tail_accuracy_qualified=False,grid_source='frozen_analytic_function_only',
            protocol_id=POLICY_ID,policy_sha256=POLICY_SHA256)


def fit_asymptotic_barrier(coordinates,energies_cm,*,left_limit_cm,right_limit_cm,
        degree,sample_budget,validation_indices,max_validation_error_cm=10.,
        slopes=None,gradient_length_scale=0.1):
    """Variable-projection fit with in-budget held-out validation, no new data.

    The endpoint energies are physical input, never optimization variables.
    The analytic family and finite degree are chosen independently of DVR size.
    A failed hold-out gate raises; it never requests or acquires more samples.
    """
    x=np.asarray(coordinates,float);y=np.asarray(energies_cm,float)
    validate_sample_plan(sample_count=len(x),sample_budget=sample_budget)
    if (x.ndim!=1 or y.shape!=x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x)<=0) or not np.isfinite(left_limit_cm) or not np.isfinite(right_limit_cm)
            or isinstance(degree,bool) or not isinstance(degree,int) or degree<1):
        raise ValueError('ordered finite samples, finite physical asymptotes and positive degree required')
    ids=np.asarray(validation_indices)
    if (ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids)
            or np.any(ids<=0) or np.any(ids>=len(x)-1)):
        raise ValueError('distinct interior held-out indices required')
    if not np.isfinite(max_validation_error_cm) or max_validation_error_cm<=0:
        raise ValueError('positive validation tolerance required')
    train=np.ones(len(x),bool);train[ids]=False
    dy=None if slopes is None else np.asarray(slopes,float)
    if dy is not None and (dy.shape!=x.shape or not np.all(np.isfinite(dy))):
        raise ValueError('finite sample-aligned slopes required')
    if not np.isfinite(gradient_length_scale) or gradient_length_scale<=0:
        raise ValueError('positive gradient length scale required')
    if train.sum()*(1 if dy is None else 2)<=degree+4:raise ValueError('insufficient training samples for this finite family')
    domain=(float(x[0]),float(x[-1]));span=x[-1]-x[0]
    def solve(mask):
        xx=x[mask];yy=y[mask]
        def project(par):
            model=AsymptoticBarrierSurface(np.zeros(degree+1),float(par[0]),float(np.exp(par[1])),
                float(left_limit_cm),float(right_limit_cm),domain,float(par[2]))
            parent,basis=model.design(xx)
            rhs=yy-parent
            if dy is not None:
                dparent,dbasis=model.derivative_design(xx)
                basis=np.vstack([basis,gradient_length_scale*dbasis])
                rhs=np.r_[rhs,gradient_length_scale*(dy[mask]-dparent)]
            c,_,rank,_=np.linalg.lstsq(basis,rhs,rcond=1e-12)
            return model,c,rank,basis@c-rhs
        trials=[]
        for width,skew in [(w,b) for w in [1.,2.,4.,8.] for b in [-.6,0.,.6]]:
            start=[x[np.argmax(y)],np.log(width/span),skew]
            result=least_squares(lambda p:project(p)[3],start,
                bounds=([x[0],np.log(.1/span),-.95],[x[-1],np.log(100/span),.95]),
                max_nfev=1000,ftol=1e-11,xtol=1e-11,gtol=1e-10)
            model,c,rank,residual=project(result.x)
            if rank==degree+1 and result.success:trials.append((float(residual@residual),model,c))
        if not trials:raise ValueError('asymptotic fit did not converge at full rank')
        _,model,c=min(trials,key=lambda r:r[0])
        return AsymptoticBarrierSurface(c,model.center,model.inverse_length,
            model.left_limit_cm,model.right_limit_cm,domain,model.asymmetry)
    candidate=solve(train);held=candidate.evaluate(x[ids])-y[ids]
    if max(abs(held))>max_validation_error_cm:
        raise ValueError(f'asymptotic fit held-out error {max(abs(held)):.6g} cm^-1 exceeds {max_validation_error_cm:g}; frozen sample plan is unchanged')
    model=solve(np.ones(len(x),bool));residual=model.evaluate(x)-y
    if max(abs(residual))>max_validation_error_cm:raise ValueError('final asymptotic fit fails sample residual gate')
    return model,dict(validation_indices=ids.tolist(),validation_residuals_cm=held.tolist(),
        max_validation_error_cm=float(max(abs(held))),max_fit_error_cm=float(max(abs(residual))),
        sample_count=len(x),sample_budget=sample_budget,degree=degree,
        native_gradients_used=dy is not None,gradient_length_scale=gradient_length_scale,
        new_PES_evaluations_for_generated_grids=0,fit_validated=True)


def run_asymptotic_surface_request(request_path):
    """Shared KeyMaker entry: same frozen-data policy, explicit physical limits."""
    from pathlib import Path
    import json,hashlib
    path=Path(request_path).resolve();raw=path.read_bytes();r=json.loads(raw)
    if r.get('schema')!='matrix.dvr.analytic_surface_request.v1' or r.get('family')!='asymptotic_barrier':
        raise ValueError('typed asymptotic analytic-surface request required')
    if not r.get('state_identity'):raise ValueError('explicit electronic/channel state identity required')
    if r.get('adaptive_qm',False):raise ValueError('automatic QM refinement is forbidden')
    q=np.asarray(r['training_coordinates'],float)
    if q.ndim==2 and q.shape[1]==1:q=q[:,0]
    model,control=fit_asymptotic_barrier(q,r['energies_cm'],left_limit_cm=r['left_limit_cm'],
        right_limit_cm=r['right_limit_cm'],degree=r['degree'],sample_budget=r['sample_budget'],
        validation_indices=r['validation_indices'],max_validation_error_cm=r['max_validation_error_cm'],
        slopes=r.get('slopes'),gradient_length_scale=r.get('gradient_length_scale',.1))
    axes=r['grid_axes']
    if len(axes)!=1:raise ValueError('one reaction-coordinate grid required')
    grid=np.asarray(axes[0],float)
    if grid.ndim!=1 or len(grid)<2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid)<=0):
        raise ValueError('ordered finite grid required')
    outside=bool(grid[0]<q[0] or grid[-1]>q[-1])
    if outside and r.get('asymptotic_extrapolation') is not True:
        raise ValueError('explicit asymptotic-extrapolation declaration required')
    values=model.evaluate(grid)
    bounds=r.get('potential_bounds_cm')
    if bounds is not None and (len(bounds)!=2 or not np.all(np.isfinite(bounds)) or bounds[0]>=bounds[1]
            or min(values)<bounds[0] or max(values)>bounds[1]):
        raise ValueError('analytic potential violates its declared physical energy bounds')
    out=(path.parent/r.get('output_directory','asymptotic_surface')).resolve();out.mkdir(parents=True,exist_ok=True)
    artifact=out/'analytic_dvr_grid.npz'
    np.savez_compressed(artifact,coordinates=grid[:,None],potential_cm=values,coefficients=model.coefficients,
        center=model.center,inverse_length=model.inverse_length,asymmetry=model.asymmetry,
        left_limit_cm=model.left_limit_cm,right_limit_cm=model.right_limit_cm)
    report=dict(schema='matrix.dvr.analytic_surface_result.v1',policy_id=POLICY_ID,policy_sha256=POLICY_SHA256,
        request_sha256=hashlib.sha256(raw).hexdigest(),**control,model=model.metadata(),
        state_identity=r['state_identity'],asymptotic_extrapolation_used=outside,
        remote_tail_accuracy_qualified=False,QM_evaluations_for_generated_grid=0,
        grid_source='analytic_function_only',artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),status='fit_validated')
    target=out/'analytic_surface_result.json';target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return target
