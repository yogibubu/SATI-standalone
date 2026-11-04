"""Local positive-gap two-root energy interpolation from finite E/G data.

The centroid is quadratic and the squared splitting a positive quadratic.
This interpolates adiabatic energies and forces; it does not define electronic
eigenvectors, derivative couplings, or coupled-surface nuclear dynamics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class LocalAdiabaticPair:
    origin: np.ndarray
    scales: np.ndarray
    centroid: np.ndarray
    gap_center: np.ndarray
    gap_matrix: np.ndarray
    minimum_gap: float

    def __post_init__(self):
        for name in ('origin','scales','centroid','gap_center','gap_matrix'):
            value=np.array(getattr(self,name),float,copy=True)
            if not np.all(np.isfinite(value)):raise ValueError('Finite local pair parameters required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        if (self.origin.shape!=(2,) or self.scales.shape!=(2,) or np.any(self.scales<=0)
                or self.centroid.shape!=(6,) or self.gap_center.shape!=(2,)
                or self.gap_matrix.shape!=(2,2) or not np.allclose(self.gap_matrix,self.gap_matrix.T)
                or np.linalg.eigvalsh(self.gap_matrix).min()<=0
                or not np.isfinite(self.minimum_gap) or self.minimum_gap<=0):
            raise ValueError('Two-dimensional chart, positive scales, gap and metric required')

    def evaluate(self,points,*,root=0):
        if root not in (0,1):raise ValueError('Adiabatic root 0 or 1 required')
        q=np.asarray(points,float)
        if q.shape[-1:]!=(2,) or not np.all(np.isfinite(q)):raise ValueError('Finite coordinate pairs required')
        u=(q-self.origin)/self.scales;v=u-self.gap_center
        h=np.array([[self.centroid[3],self.centroid[4]],[self.centroid[4],self.centroid[5]]])
        mean=self.centroid[0]+u@self.centroid[1:3]+.5*np.einsum('...i,ij,...j->...',u,h,u)
        a=v@self.gap_matrix;gap=np.sqrt(self.minimum_gap**2+np.sum(a*v,axis=-1));sign=-.5 if root==0 else .5
        energy=mean+sign*gap
        gradient=(self.centroid[1:3]+u@h+sign*a/gap[...,None])/self.scales
        curvature=(h+sign*(self.gap_matrix/gap[...,None,None]-a[..., :,None]*a[...,None,:]/gap[...,None,None]**3))/(self.scales[:,None]*self.scales[None,:])
        return energy,gradient,curvature

    def to_dict(self):
        return dict(schema='matrix.dvr.local_adiabatic_pair.v1',origin=self.origin.tolist(),scales=self.scales.tolist(),centroid=self.centroid.tolist(),gap_center=self.gap_center.tolist(),gap_matrix=self.gap_matrix.tolist(),minimum_gap=float(self.minimum_gap))

    def periodic_splitting(self,points):
        """Positive pi-periodic completion preserving the local splitting jet.

        The sine/cosine embedding adds a nonnegative fourth-order completion.
        It prevents the additional zero at half a period of a sine-only chart.
        Only the splitting is continued; the centroid remains a local model.
        """
        q=np.asarray(points,float)
        if q.shape[-1:]!=(2,) or not np.all(np.isfinite(q)):raise ValueError('Finite coordinate pairs required')
        theta=2*(q-self.origin);x=np.sin(theta)/(2*self.scales);y=(1-np.cos(theta))/(2*self.scales)
        v=x-self.gap_center;a=v@self.gap_matrix;b=y@self.gap_matrix
        gap=np.sqrt(self.minimum_gap**2+np.sum(v*a+y*b,axis=-1))
        dx=np.cos(theta)/self.scales;dy=np.sin(theta)/self.scales
        numerator=a*dx+b*dy;gradient=numerator/gap[...,None]
        core=self.gap_matrix*(dx[..., :,None]*dx[...,None,:]+dy[..., :,None]*dy[...,None,:])
        diagonal=-2*a*np.sin(theta)/self.scales+2*b*np.cos(theta)/self.scales
        core=core+np.eye(2)*diagonal[...,None,:]
        hessian=core/gap[...,None,None]-numerator[..., :,None]*numerator[...,None,:]/gap[...,None,None]**3
        return gap,gradient,hessian

    @classmethod
    def from_dict(cls,r):
        if r.get('schema')!='matrix.dvr.local_adiabatic_pair.v1':raise ValueError('Unsupported local pair model')
        return cls(*(r[k] for k in ('origin','scales','centroid','gap_center','gap_matrix','minimum_gap')))


@dataclass(frozen=True)
class PeriodicGapResidualSurface:
    """Positive periodic splitting plus a scalar periodic residual field."""
    gap_model: LocalAdiabaticPair
    reference_gap: float
    residual_reference: object
    residual_correction: object
    angular_factor: float = 2.

    def __post_init__(self):
        if not np.isfinite(self.reference_gap) or self.reference_gap<=0 or self.angular_factor!=2.:
            raise ValueError('Positive reference splitting and matching pi-periodic field required')
        for field in (self.residual_reference,self.residual_correction):
            if field.centers.shape[1]!=2 or field.coefficients.shape[1]!=1:
                raise ValueError('Scalar two-angle residual fields required')

    def evaluate(self,points):
        q=np.asarray(points,float);shape=q.shape[:-1];flat=q.reshape(-1,2)
        gap,g,h=self.gap_model.periodic_splitting(flat);e=-.5*(gap-self.reference_gap);g=-.5*g;h=-.5*h
        for field in (self.residual_reference,self.residual_correction):
            e+=field.evaluate(2*flat)[:,0]
            for j in range(2):
                order=np.eye(2,dtype=int)[j];g[:,j]+=2*field.evaluate(2*flat,order)[:,0]
                for k in range(2):h[:,j,k]+=4*field.evaluate(2*flat,order+np.eye(2,dtype=int)[k])[:,0]
        return e.reshape(shape),g.reshape(shape+(2,)),h.reshape(shape+(2,2))

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_gap_residual.v1',gap_model=self.gap_model.to_dict(),reference_gap=float(self.reference_gap),residual_reference=self.residual_reference.to_dict(),residual_correction=self.residual_correction.to_dict(),angular_factor=2.)

    @classmethod
    def from_dict(cls,r):
        from .periodic_hermite_kernel import PeriodicHermiteKernelField
        if r.get('schema')!='matrix.dvr.periodic_gap_residual.v1':raise ValueError('Unsupported periodic gap-residual model')
        return cls(LocalAdiabaticPair.from_dict(r['gap_model']),r['reference_gap'],PeriodicHermiteKernelField.from_dict(r['residual_reference']),PeriodicHermiteKernelField.from_dict(r['residual_correction']),r.get('angular_factor',2.))


def fit_local_adiabatic_pair(points,energies,lower_gradient,*,origin,scales,centroid_order='diagonal',derivative_length=None,fixed_minimum_gap=None):
    """Variable projection of a local two-root E/G interpolation.

    Only supplied observations enter the objective. The caller owns the
    independent reserved-point and domain qualifications.
    """
    q=np.asarray(points,float);e=np.asarray(energies,float);g=np.asarray(lower_gradient,float);origin=np.asarray(origin,float);scales=np.asarray(scales,float)
    if (q.ndim!=2 or q.shape[1]!=2 or e.shape!=q.shape or g.shape!=q.shape or len(q)<4
            or origin.shape!=(2,) or scales.shape!=(2,) or np.any(scales<=0)
            or not all(np.all(np.isfinite(x)) for x in (q,e,g,origin,scales)) or np.any(e[:,1]<=e[:,0])):
        raise ValueError('At least four finite two-root energies and lower gradients required')
    powers={'affine':[0,1,2],'diagonal':[0,1,2,3,5],'full':list(range(6))}
    if centroid_order not in powers:raise ValueError('Affine, diagonal or full centroid required')
    if fixed_minimum_gap is not None and (not np.isfinite(fixed_minimum_gap) or fixed_minimum_gap<=0):
        raise ValueError('Positive fixed splitting floor required')
    keep=powers[centroid_order];u=(q-origin)/scales;eye=np.eye(2)
    rows=[]
    for x,y in u:
        value=[1,x,y,.5*x*x,x*y,.5*y*y];gx=np.asarray([0,1,0,x,y,0])/scales[0];gy=np.asarray([0,0,1,0,x,y])/scales[1]
        rows.append([value,value,gx,gy])
    design=np.asarray(rows)[:,:,keep];target=np.column_stack([e,g]);weight=np.r_[1.,1.,scales if derivative_length is None else np.broadcast_to(derivative_length,2)]
    system=(design*weight[None,:,None]).reshape(-1,len(keep));scale=np.maximum(np.linalg.norm(system,axis=0),1e-30);system/=scale
    def unpack(t):
        if fixed_minimum_gap is not None:t=np.r_[np.log(fixed_minimum_gap),t]
        L=np.array([[np.exp(t[3]),0.],[t[4],np.exp(t[5])]])
        return (np.exp(t[0]) if fixed_minimum_gap is None else float(fixed_minimum_gap)),t[1:3],L@L.T
    def solve(t,details=False):
        delta,center,A=unpack(t);v=u-center;a=v@A;gap=np.sqrt(delta**2+np.sum(a*v,axis=1));dg=a/gap[:,None]/scales
        offset=np.column_stack([-.5*gap,.5*gap,-.5*dg]);rhs=((target-offset)*weight).ravel();coef=np.linalg.lstsq(system,rhs,rcond=1e-12)[0]/scale
        if details:return coef,design@coef+offset
        return system@(coef*scale)-rhs
    width=float(np.median(e[:,1]-e[:,0]));bound=max(10.,100*width)
    starts=[]
    for delta in (.01,.1,.3):
        t=np.r_[np.log(max(1e-3,delta*width)),0.,0.,np.log(width),0.,np.log(width)]
        lower=[np.log(1e-5),-3,-3,np.log(1e-3),-bound,np.log(1e-3)];upper=[np.log(bound),3,3,np.log(bound),bound,np.log(bound)]
        if fixed_minimum_gap is not None:
            t=t[1:];lower=lower[1:];upper=upper[1:]
            if starts:break
        starts.append(least_squares(solve,t,bounds=(lower,upper),max_nfev=2000,xtol=1e-11,ftol=1e-11,gtol=1e-8))
    fit=min(starts,key=lambda r:r.fun@r.fun);coef,pred=solve(fit.x,True);delta,center,A=unpack(fit.x);full=np.zeros(6);full[keep]=coef
    model=LocalAdiabaticPair(origin,scales,full,center,A,delta);singular=np.linalg.svd(fit.jac,compute_uv=False);rank=int(sum(singular>singular[0]*1e-9))
    return model,dict(training_maximum_energy_error=float(abs(pred[:,:2]-e).max()),training_maximum_gradient_error=float(abs(pred[:,2:]-g).max()),nonlinear_jacobian_rank=rank,nonlinear_parameters=len(fit.x),centroid_parameters=len(keep),centroid_order=centroid_order,fixed_minimum_gap=fixed_minimum_gap,weighted_cost=float(fit.fun@fit.fun),optimizer_success=bool(fit.success),new_electronic_jobs=0)
