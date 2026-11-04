"""Mass-independent Cartesian normal frames for transverse potential models.

The supplied transverse span is retained. Translation and instantaneous rigid
rotation are removed with the Euclidean metric, then the symmetric Gram root
normalizes the frame. Isotope masses are deliberately absent. Force and
Hessian must be expressed in this same frame; an internal-coordinate Hessian
cannot be reused without its full connection term.
"""
import numpy as np
from scipy.linalg import solve_sylvester


def cartesian_normal_frame(coordinates,transverse_tangents,*,path_tangents=None,
                           transverse_first=None):
    """Return Q, Q' for x(q,b)=x_path(q)+Q(q)b, with Q.T Q = identity.

    Cartesian arrays have shape (atoms,3[,bath[,active]]). First derivatives
    are analytic, including the changing rigid-rotation projector. The bath
    parameter b has angstrom units. No physical continuation is inferred.
    """
    x=np.asarray(coordinates,float);D=np.asarray(transverse_tangents,float)
    if (x.ndim!=2 or x.shape[1]!=3 or D.ndim!=3 or D.shape[:2]!=x.shape
            or not 0<D.shape[2]<=x.size-6 or not all(np.all(np.isfinite(a)) for a in (x,D))):
        raise ValueError('Finite nonlinear geometry and matching transverse span required')
    if (path_tangents is None)!=(transverse_first is None):
        raise ValueError('Both path and transverse first derivatives are required')
    x=x-x.mean(axis=0);D=D-D.mean(axis=0)
    inertia=np.sum(x*x)*np.eye(3)-x.T@x
    if np.linalg.eigvalsh(inertia).min()<=1e-12:raise ValueError('Nonlinear reference required')
    cross=np.cross(x[:,None,:],D.swapaxes(1,2)).sum(axis=0).T
    omega=np.linalg.solve(inertia,cross)
    V=D-np.cross(omega.T[None,:,:],x[:,None,:]).swapaxes(1,2)
    flat=V.reshape(x.size,-1);gram=flat.T@flat;e,U=np.linalg.eigh(gram)
    if e.min()<=1e-12*e.max():raise ValueError('Transverse frame loses physical rank')
    root=(U*np.sqrt(e))@U.T;inverse=(U/np.sqrt(e))@U.T
    Q=flat@inverse
    result=dict(frame=Q.reshape(D.shape),gram=gram,gram_root=root,
        orthogonality_error=float(np.max(abs(Q.T@Q-np.eye(len(e))))),
        normalization_condition=float(e.max()/e.min()))
    if path_tangents is None:return result
    X1=np.asarray(path_tangents,float);D1=np.asarray(transverse_first,float)
    if (X1.ndim!=3 or X1.shape[:2]!=x.shape or D1.shape!=D.shape+(X1.shape[2],)
            or not all(np.all(np.isfinite(a)) for a in (X1,D1))):
        raise ValueError('Matching analytic path/frame derivatives required')
    X1=X1-X1.mean(axis=0);D1=D1-D1.mean(axis=0);first=[]
    for k in range(X1.shape[2]):
        dx=X1[:,:,k];dD=D1[:,:,:,k]
        dI=2*np.sum(x*dx)*np.eye(3)-dx.T@x-x.T@dx
        dC=(np.cross(dx[:,None,:],D.swapaxes(1,2))+
            np.cross(x[:,None,:],dD.swapaxes(1,2))).sum(axis=0).T
        dw=np.linalg.solve(inertia,dC-dI@omega)
        dV=dD-(np.cross(dw.T[None,:,:],x[:,None,:])+
            np.cross(omega.T[None,:,:],dx[:,None,:])).swapaxes(1,2)
        dV=dV.reshape(x.size,-1);dE=dV.T@flat+flat.T@dV
        droot=solve_sylvester(root,root,dE)
        first.append((dV@inverse-Q@droot@inverse).reshape(D.shape))
    result['frame_first']=np.stack(first,axis=-1)
    return result


def linear_cartesian_force_jet(internal_gradient,internal_hessian,
                               coordinate_first,coordinate_second,frame):
    """Contract a complete scalar internal jet into a linear Cartesian frame.

    coordinate_first/second are B/B' = derivatives of the internal coordinates
    with respect to Cartesian positions. The gradient-times-B' term is kept.
    Thus the output is a Hessian in a linear Cartesian displacement, rather
    than a Hessian in the original curved internal displacement.
    """
    g=np.asarray(internal_gradient,float);H=np.asarray(internal_hessian,float)
    B=np.asarray(coordinate_first,float);B2=np.asarray(coordinate_second,float)
    Q=np.asarray(frame,float).reshape(B.shape[1],-1)
    if (g.shape!=(len(B),) or H.shape!=(len(B),len(B))
            or B2.shape!=(len(B),B.shape[1],B.shape[1])
            or not all(np.all(np.isfinite(a)) for a in (g,H,B,B2,Q))):
        raise ValueError('Complete matching scalar and coordinate jets required')
    J=B@Q;force=J.T@g
    curvature=J.T@H@J+np.einsum('i,iab,ac,bd->cd',g,B2,Q,Q,optimize=True)
    return dict(force=force,hessian=.5*(curvature+curvature.T),
                connection_term=np.einsum('i,iab,ac,bd->cd',g,B2,Q,Q,optimize=True))


def cartesian_mean_response_path(path_geometry,path_first,frame,frame_first,
                                 response,response_first):
    """Mean geometry and exact first jets of the *same* linear bath model.

    This realizes x(q,b)=x_path(q)+Q(q)b at b=b*(q). No quadratic geometry
    expansion is applied to a large transverse offset. The bath Hessian must
    already describe this linear Cartesian b, as in linear_cartesian_force_jet.
    """
    x=np.asarray(path_geometry,float);X=np.asarray(path_first,float)
    Q=np.asarray(frame,float);Q1=np.asarray(frame_first,float)
    b=np.asarray(response,float);b1=np.asarray(response_first,float)
    if (x.ndim!=3 or x.shape[-1]!=3 or Q.ndim!=4 or Q.shape[:3]!=x.shape
            or X.ndim!=4 or X.shape[:3]!=x.shape or Q1.shape!=Q.shape+(X.shape[-1],)
            or b.shape!=(len(x),Q.shape[-1]) or b1.shape!=b.shape+(X.shape[-1],)
            or not all(np.all(np.isfinite(a)) for a in (x,X,Q,Q1,b,b1))):
        raise ValueError('Matching batched Cartesian path, frame and response first jets required')
    mean=x+np.einsum('piab,pb->pia',Q,b)
    first=X+np.einsum('piabq,pb->piaq',Q1,b)+np.einsum('piab,pbq->piaq',Q,b1)
    return dict(geometry=mean,active_first=first,bath_first=Q)


def cartesian_mean_reference_jet(path_geometry,path_first,path_second,frame,frame_first,
                                  response_fields):
    """Complete mean-path/bath second jet at an anchored b*(0)=0 reference.

    Only Q and Q' are needed at this reference; Q'' multiplies the vanishing
    mean displacement. The bath embedding is linear, so its second bath jet
    vanishes. A finite nonzero reference response requires a higher frame jet
    and is explicitly rejected instead of silently omitting it.
    """
    x=np.asarray(path_geometry,float);X=np.asarray(path_first,float);X2=np.asarray(path_second,float)
    Q=np.asarray(frame,float);Q1=np.asarray(frame_first,float);v=response_fields
    b=np.asarray(v['response'],float);b1=np.asarray(v['response_first'],float)
    if b.ndim!=1 or abs(b).max(initial=0)>1e-9:raise ValueError('Zero anchored mean response required for this second-jet contract')
    d=X.shape[-1];n=len(b)
    if (x.ndim!=2 or x.shape[1]!=3 or X.shape!=x.shape+(d,) or X2.shape!=x.shape+(d,d)
            or Q.shape!=x.shape+(n,) or Q1.shape!=x.shape+(n,d) or b1.shape!=(n,d)):
        raise ValueError('Complete matching anchored Cartesian jets required')
    H=np.asarray(v['hessian']);H1=np.asarray(v['hessian_first']);H2=np.asarray(v['hessian_second'])
    mixed=np.einsum('ijs,jt->ist',H1,b1)
    rhs=np.asarray(v['force_second'])+np.einsum('ijst,j->ist',H2,b)+mixed+mixed.swapaxes(-1,-2)
    b2=-np.linalg.solve(H,rhs.reshape(n,d*d)).reshape(n,d,d)
    first=np.concatenate([X+np.einsum('iab,bq->iaq',Q,b1),Q],axis=-1)
    second=np.zeros(x.shape+(d+n,d+n))
    mix=np.einsum('iabq,br->iaqr',Q1,b1)
    second[:,:,:d,:d]=X2+mix+mix.swapaxes(-1,-2)+np.einsum('iab,bqr->iaqr',Q,b2)
    second[:,:,:d,d:]=Q1.swapaxes(-1,-2);second[:,:,d:,:d]=Q1
    return dict(geometry=x,first=first,second=second,response_second=b2)
