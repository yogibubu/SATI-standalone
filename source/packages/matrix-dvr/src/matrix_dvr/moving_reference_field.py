"""Complete stationary force-field pullback into a moving normal chart."""
import numpy as np


def normal_frame_first_jet(geometry, masses, active_first, active_second, normal_frame,
                           *, closing_generator=None, periodic_axis=None, half_period=np.pi):
    """Analytic projector derivative, plus the declared closing-frame gauge.

Normal-frame columns are mass weighted. The horizontal derivative follows
A.T Q=0 and Q.T Q=I for the rigid/active span A. A supplied closing generator
is the same proper rotation used to close the archived periodic frame.
"""
    x=np.asarray(geometry,float);m=np.asarray(masses,float)
    d=np.asarray(active_first,float);dd=np.asarray(active_second,float);q=np.asarray(normal_frame,float)
    n=len(m);na=d.shape[-1];root=np.sqrt(np.repeat(m,3))
    if x.shape!=(n,3) or d.shape!=(n,3,na) or dd.shape!=(n,3,na,na) or q.shape[0]!=3*n:
        raise ValueError('Complete matching reference geometry and active jets required')
    x=x-np.average(x,axis=0,weights=m)
    d=d-np.einsum('i,ias->as',m,d)[None]/sum(m)
    dd=dd-np.einsum('i,iast->ast',m,dd)[None]/sum(m)
    rigid_t=np.tile(np.eye(3),(n,1))
    rigid_r=np.stack([np.cross(v,x).ravel() for v in np.eye(3)],axis=1)
    a=root[:,None]*np.column_stack([rigid_t,rigid_r,d.reshape(3*n,na)])
    gram=a.T@a;result=[]
    for axis in range(na):
        derivative_r=np.stack([np.cross(v,d[:,:,axis]).ravel() for v in np.eye(3)],axis=1)
        derivative_a=root[:,None]*np.column_stack([np.zeros((3*n,3)),derivative_r,dd[:,:,:,axis].reshape(3*n,na)])
        value=-a@np.linalg.solve(gram,derivative_a.T@q)
        if periodic_axis==axis and closing_generator is not None:
            generator=np.asarray(closing_generator,float)
            if generator.shape!=(q.shape[1],q.shape[1]) or not np.allclose(generator,-generator.T,atol=1e-9):
                raise ValueError('Real antisymmetric frame-closing generator required')
            value+=q@generator/half_period
        result.append(value)
    return np.stack(result,axis=-1)


def stationary_cubic_pullback(cartesian_hessian,cartesian_cubic,first,second):
    """Transform every F2/F3 component, including all curvature terms.

The source is the stationary-parent convention: no unprovided third geometry
jet times a finite gradient is fabricated. No active/transverse partition or
observable-specific subtraction is made in this transformation.
"""
    h=np.asarray(cartesian_hessian,float);c=np.asarray(cartesian_cubic,float)
    d=np.asarray(first,float);dd=np.asarray(second,float)
    if (h.ndim!=2 or h.shape[0]!=h.shape[1] or c.shape!=h.shape+(len(h),)
            or d.shape[0]!=len(h) or dd.shape!=(len(h),d.shape[1],d.shape[1])
            or not all(np.all(np.isfinite(v)) for v in (h,c,d,dd))):
        raise ValueError('Complete finite Cartesian field and reference geometry jets required')
    force=d.T@h@d
    third=np.einsum('ijk,ia,jb,kc->abc',c,d,d,d,optimize=True)
    curvature=np.einsum('ij,iab,jc->abc',h,dd,d,optimize=True)
    third+=curvature+curvature.transpose(0,2,1)+curvature.transpose(2,1,0)
    if not np.allclose(third,third.transpose(1,0,2),atol=1e-6,rtol=1e-10):
        raise ValueError('Complete cubic pullback is not symmetric')
    return dict(force_hessian=force,cubic=third,field_modes_removed=False,
                convention='Complete stationary-parent Cartesian F2/F3 with all second-geometry curvature terms')


__all__=['normal_frame_first_jet','stationary_cubic_pullback']
