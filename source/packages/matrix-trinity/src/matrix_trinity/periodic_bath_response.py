"""Periodic linear bath forces and leading adiabatic harmonic response.

V(a,b)=V_active(a)+g(a).b+sum_j w_j*b_j^2/2. Completing the square gives
-sum_j g_j^2/(2*w_j). The optional diagonal Born--Huang term retains the
norm of the derivative of the displaced oscillator fiber. This is an
adiabatic response model, not a full coupled VCI spectrum.
"""
import numpy as np


def periodic_force_design(points):
    q=np.asarray(points,float)
    if q.ndim!=2:raise ValueError('Coordinate matrix required')
    values=[];derivatives=[];d=q.shape[1];s=np.sin(2*q);c=np.cos(2*q)
    for j in range(d):
        for value,derivative in ((s[:,j],2*c[:,j]),(c[:,j]-1,-2*s[:,j])):
            v=np.zeros(q.shape);v[:,j]=derivative;values.append(value);derivatives.append(v)
    for i in range(d):
        for j in range(i):
            for left,right,dl,dr,offset in ((s[:,i],s[:,j],2*c[:,i],2*c[:,j],0),(c[:,i],c[:,j],-2*s[:,i],-2*s[:,j],1),(s[:,i],c[:,j],2*c[:,i],-2*s[:,j],0),(c[:,i],s[:,j],-2*s[:,i],2*c[:,j],0)):
                v=np.zeros(q.shape);v[:,i]=dl*right;v[:,j]=left*dr;values.append(left*right-offset);derivatives.append(v)
    return np.array(values).T,np.stack(derivatives,axis=1)


def fit_periodic_bath_forces(points,forces,reference_force,*,weights=None):
    x,_=periodic_force_design(points);y=np.asarray(forces)-np.asarray(reference_force);weight=np.ones(len(x)) if weights is None else np.asarray(weights)
    if y.ndim!=2 or len(y)!=len(x) or weight.shape!=(len(x),) or np.any(weight<=0):raise ValueError('Compatible force records and positive fitting weights required')
    coeff,_,rank,_=np.linalg.lstsq(x*np.sqrt(weight[:,None]),y*np.sqrt(weight[:,None]),rcond=1e-10)
    if rank!=x.shape[1]:raise ValueError('Periodic bath-force basis is not identified by training data')
    return coeff


def adiabatic_bath_response(points,coefficients,frequencies_cm,*,selected_modes=None,kinetic_cm=None,temperature_K=0.):
    x,dx=periodic_force_design(points);coef=np.asarray(coefficients);w=np.asarray(frequencies_cm,float)
    if coef.shape!=(x.shape[1],len(w)) or np.any(w<=0):raise ValueError('Positive bath frequencies and matching force coefficients required')
    modes=np.arange(len(w)) if selected_modes is None else np.asarray(selected_modes,int);force=x@coef[:,modes];derivative=np.einsum('gkd,kj->gdj',dx,coef[:,modes]);displacement=-force/w[modes];dd=-derivative/w[modes];relax=-.5*np.sum(force*force/w[modes],axis=1);born=np.zeros(len(x))
    if kinetic_cm is not None:
        metric=np.asarray(kinetic_cm);thermal=np.ones(len(modes)) if temperature_K==0 else 1/np.tanh(w[modes]/(2*.6950348004861274*temperature_K));born=.25*np.einsum('gij,gik,gjk,k->g',metric,dd,dd,thermal,optimize=True)
    return dict(relaxation_cm=relax,born_huang_cm=born,total_cm=relax+born,displacements=displacement)
