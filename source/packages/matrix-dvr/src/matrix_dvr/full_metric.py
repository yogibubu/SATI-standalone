"""Full internal/rotation mass metric and analytic Cartesian-jet transport."""
import numpy as np

KINETIC_CM_AMU_ANGSTROM2=2*505379.0090099937/29979.2458


def cartesian_affine_rotation_metric_grid(origin_angstrom,masses_amu,tangents,points):
    """Complete batched metric and analytic volume gradient for X(q)=X0+Dq.

    This reuses the complete value-metric owner and differentiates the same
    covariant blocks. The explicit affine contract supplies zero second
    geometry derivatives; it must not be used for a nonlinear geometry map.
    """
    origin=np.asarray(origin_angstrom,float);m=np.asarray(masses_amu,float)
    D=np.asarray(tangents,float);q=np.asarray(points,float)
    if (origin.shape!=(len(m),3) or D.ndim!=3 or D.shape[:2]!=origin.shape
            or q.ndim!=2 or q.shape[1]!=D.shape[-1]
            or not all(np.all(np.isfinite(a)) for a in (origin,m,D,q)) or np.any(m<=0)):
        raise ValueError('Finite affine geometry, positive masses and compatible coordinate points required')
    xyz=origin[None]+np.einsum('ias,ps->pia',D,q)
    result=cartesian_internal_rotation_metric_grid(xyz,m,np.broadcast_to(D,xyz.shape+(D.shape[-1],)))
    xyz=xyz-np.einsum('i,pia->pa',m,xyz)[:,None,:]/m.sum()
    D=D-np.einsum('i,ias->as',m,D)[None]/m.sum()
    count=D.shape[-1];inverse=result['kinetic_cm']/KINETIC_CM_AMU_ANGSTROM2
    gradient=np.empty((len(q),count))
    for k in range(count):
        E=D[:,:,k]
        cross=np.einsum('i,isa->as',m,np.cross(E[:,None,:],D.swapaxes(1,2)))
        rotation=(2*np.einsum('i,pia,ia->p',m,xyz,E)[:,None,None]*np.eye(3)
            -np.einsum('i,ia,pib->pab',m,E,xyz)-np.einsum('i,pia,ib->pab',m,xyz,E))
        gradient[:,k]=(np.einsum('pai,ai->p',inverse[:,count:,:count],cross)
            +.5*np.einsum('pab,pba->p',inverse[:,count:,count:],rotation))
    result['log_volume_gradient']=gradient
    return result


def cartesian_internal_rotation_metric_grid(coordinates_angstrom,masses_amu,tangents):
    """Batched complete value metric; identical to the scalar first-jet owner."""
    x=np.asarray(coordinates_angstrom,float);m=np.asarray(masses_amu,float)
    D=np.asarray(tangents,float)
    if (x.ndim!=3 or x.shape[1:]!=(len(m),3) or D.ndim!=4
            or D.shape[:3]!=x.shape or np.any(m<=0)
            or not all(np.all(np.isfinite(a)) for a in (x,m,D))):
        raise ValueError('Matching finite batched geometry and tangent fields required')
    x=x-np.einsum('i,pia->pa',m,x)[:,None,:]/m.sum()
    D=D-np.einsum('i,pias->pas',m,D)[:,None,:,:]/m.sum()
    nv=D.shape[-1];cov=np.zeros((len(x),nv+3,nv+3))
    cross=np.einsum('i,pisa->pas',m,np.cross(x[:,:,None,:],D.swapaxes(-1,-2)))
    cov[:,:nv,:nv]=np.einsum('i,pias,piat->pst',m,D,D)
    cov[:,nv:,:nv]=cross;cov[:,:nv,nv:]=cross.swapaxes(-1,-2)
    cov[:,nv:,nv:]=(np.einsum('i,pia,pia->p',m,x,x)[:,None,None]*np.eye(3)
        -np.einsum('i,pia,pib->pab',m,x,x))
    if np.any(np.linalg.eigvalsh(cov)<=0):
        raise ValueError('Nonpositive complete molecular metric in grid')
    inverse=np.linalg.inv(cov)
    # Covariance positivity was checked above. Its inverse is symmetric
    # analytically; remove inversion roundoff before weak-form validation,
    # especially for tiny off-diagonals next to a stiff fourth coordinate.
    inverse=(inverse+inverse.swapaxes(-1,-2))/2
    return dict(covariant_metric=cov,kinetic_cm=KINETIC_CM_AMU_ANGSTROM2*inverse)


def cartesian_internal_rotation_metric(coordinates_angstrom,masses_amu,tangents,
                                      tangent_derivatives=None):
    """Return coefficients in internal coordinates followed by body rotation.

    Tangents have shape (atom,3,internal). Optional second derivatives have
    shape (atom,3,internal,internal). All coordinates share one fixed embedding.
    The volume derivative is d log(sqrt(det covariant_metric))/dq.
    """
    x=np.array(coordinates_angstrom,float,copy=True);m=np.asarray(masses_amu,float)
    D=np.array(tangents,float,copy=True)
    if (x.ndim!=2 or x.shape[1]!=3 or m.shape!=(len(x),) or np.any(m<=0)
            or D.ndim!=3 or D.shape[:2]!=x.shape
            or not all(np.all(np.isfinite(v)) for v in (x,m,D))):
        raise ValueError('Compatible finite molecular coordinates, masses and tangents required')
    nv=D.shape[-1];x-=np.average(x,axis=0,weights=m)
    D-=np.einsum('i,ias->as',m,D)[None]/m.sum()
    inertia=np.einsum('i,ia,ia->',m,x,x)*np.eye(3)-np.einsum('i,ia,ib->ab',m,x,x)
    cross=np.einsum('i,isa->as',m,np.cross(x[:,None,:],D.swapaxes(1,2)))
    cov=np.block([[np.einsum('i,ias,iat->st',m,D,D),cross.T],[cross,inertia]])
    if np.min(np.linalg.eigvalsh(cov))<=0:raise ValueError('Nonpositive full molecular metric')
    inv=np.linalg.inv(cov);inv=(inv+inv.T)/2
    result=dict(covariant_metric=cov,kinetic_cm=KINETIC_CM_AMU_ANGSTROM2*inv)
    if tangent_derivatives is not None:
        DD=np.array(tangent_derivatives,float,copy=True)
        if DD.shape!=x.shape+(nv,nv) or not np.all(np.isfinite(DD)):
            raise ValueError('Matching finite second Cartesian jet required')
        DD-=np.einsum('i,iast->ast',m,DD)[None]/m.sum()
        first=np.empty(cov.shape+(nv,))
        for k in range(nv):
            E=D[:,:,k];dD=DD[:,:,:,k]
            internal=np.einsum('i,ias,iat->st',m,dD,D);internal+=internal.T
            c=np.einsum('i,isa->as',m,np.cross(E[:,None,:],D.swapaxes(1,2))+np.cross(x[:,None,:],dD.swapaxes(1,2)))
            r=2*np.einsum('i,ia,ia->',m,x,E)*np.eye(3)-np.einsum('i,ia,ib->ab',m,E,x)-np.einsum('i,ia,ib->ab',m,x,E)
            first[:,:,k]=np.block([[internal,c.T],[c,r]])
        result['kinetic_first_cm']=-KINETIC_CM_AMU_ANGSTROM2*np.einsum('ab,bck,cd->adk',inv,first,inv)
        result['log_volume_gradient']=.5*np.einsum('ab,bak->k',inv,first)
    return result
