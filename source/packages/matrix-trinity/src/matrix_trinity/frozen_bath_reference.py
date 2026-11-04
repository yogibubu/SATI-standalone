"""Reference jets of the explicitly defined bath-origin metric model.

The DVR metric depends on active coordinates and is frozen in transverse
displacement. Its counterterm must have the same omitted bath derivatives;
using the complete electronic VPT2 field as this counterterm changes the model.
The complete VPT2 field belongs in the caller's retained full correction.
"""
import numpy as np
from itertools import permutations
from matrix_dvr.full_metric import cartesian_internal_rotation_metric
from matrix_dvr.rotational_metric_curvature import eckart_rotational_metric_second


def potential_reference_jets(evaluate, point, *, third_step=.001):
    """Read a fitted potential's Hessian and differentiate it for the cubic.

    evaluate(points, integer_derivative) reads the existing analytic function,
    not an electronic program. Two steps and Richardson extrapolation audit
    the cubic even when the function's public API stops at second derivatives.
    """
    q=np.asarray(point,float);d=len(q);eye=np.eye(d,dtype=int);h=float(third_step)
    if q.shape!=(d,) or h<=0 or not np.all(np.isfinite(q)):
        raise ValueError('Finite reference point and positive differentiation step required')
    def hessian(points):
        return np.array([[evaluate(points,a+b) for b in eye] for a in eye]).transpose(2,0,1)
    H=hessian(q[None])[0];estimates=[]
    for step in (h,h/2):
        points=np.vstack([q+step*eye,q-step*eye]);values=hessian(points)
        estimates.append(((values[:d]-values[d:])/(2*step)).transpose(1,2,0))
    cubic=(4*estimates[1]-estimates[0])/3
    defect=max(float(abs(cubic-cubic.transpose(p)).max()) for p in permutations(range(3)))
    cubic=sum(cubic.transpose(p) for p in permutations(range(3)))/6
    return H,cubic,dict(third_step=h,Richardson_cubic_error=float(abs(estimates[1]-estimates[0]).max()/3),
                        cubic_permutation_defect=defect)


def frozen_bath_reference_jets(coordinates,masses,first,second,active_hessian,
                               active_cubic,bath_hessian,bath_hessian_first):
    ha=np.asarray(active_hessian,float);ca=np.asarray(active_cubic,float)
    hb=np.asarray(bath_hessian,float);hb1=np.asarray(bath_hessian_first,float)
    d=len(ha);b=len(hb);n=d+b
    if (ha.shape!=(d,d) or ca.shape!=(d,d,d) or hb.shape!=(b,b)
            or hb1.shape!=(b,b,d) or not all(np.all(np.isfinite(x)) for x in (ha,ca,hb,hb1))):
        raise ValueError('Complete matching active and transverse potential jets required')
    metric=cartesian_internal_rotation_metric(coordinates,masses,first,second)
    if metric['kinetic_cm'].shape!=(n+3,n+3):raise ValueError('Geometry and force coordinate counts differ')
    full_r2=eckart_rotational_metric_second(coordinates,masses,first,second)['rotational_second_cm']
    H=np.zeros((n,n));H[:d,:d]=ha;H[d:,d:]=hb
    C=np.zeros((n,n,n));C[:d,:d,:d]=ca
    for s in range(d):
        C[d:,d:,s]=hb1[:,:,s];C[d:,s,d:]=hb1[:,:,s];C[s,d:,d:]=hb1[:,:,s]
    G1=np.zeros((n+3,n+3,n));G1[:,:,:d]=metric['kinetic_first_cm'][:,:,:d]
    R2=np.zeros((3,3,n,n));R2[:,:,:d,:d]=full_r2[:,:,:d,:d]
    return dict(H=H,C=C,G=metric['kinetic_cm'],G1=G1,R2=R2,
                scope='Exact jets of V(a)+b.H(a).b/2 and G(a,b=0); all transverse coordinates retained')
