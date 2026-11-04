"""Positive quadrature adapted to axial volume profiles, with unchanged basis.

Profiles are importance measures only. Physical domains, trial functions and
multidimensional fields are unchanged. Mixed ridges still require convergence
checks; resolution of each separate axial profile is not sufficient.
"""
from itertools import product
import numpy as np
from scipy.interpolate import PchipInterpolator
from .weighted_angular_galerkin import finite_gaussian_axis


def _profile_rule(bounds,count,profile_points,log_profile,envelope,moment_order):
    lo,hi=bounds;x=np.asarray(profile_points,float);v=np.asarray(log_profile,float)
    if (x.ndim!=1 or v.shape!=x.shape or len(x)<3 or np.any(np.diff(x)<=0)
            or x[0]>lo+1e-12 or x[-1]<hi-1e-12 or not np.all(np.isfinite(v))
            or count<2 or moment_order<4*count):
        raise ValueError('Finite profile covering the full interval and resolved moment rule required')
    interpolator=PchipInterpolator(x,v,extrapolate=False)
    t,w=np.polynomial.legendre.leggauss(moment_order);samples=(lo+hi)/2+(hi-lo)/2*t
    log_weight=interpolator(samples)+envelope(samples);shift=float(log_weight.max())
    weights=(hi-lo)/2*w*np.exp(log_weight-shift);mass=weights.sum()
    previous=np.zeros_like(samples);current=np.ones_like(samples)/np.sqrt(mass)
    alphas=[];betas=[0.]
    for k in range(count):
        alpha=float(weights@(samples*current**2));alphas.append(alpha)
        residual=(samples-alpha)*current-betas[k]*previous
        beta=float(np.sqrt(weights@residual**2));betas.append(beta)
        if beta<=1e-14:raise ValueError('Degenerate positive profile quadrature')
        previous,current=current,residual/beta
    jacobi=np.diag(alphas)+np.diag(betas[1:count],1)+np.diag(betas[1:count],-1)
    nodes,U=np.linalg.eigh(jacobi);gaussian_weights=mass*U[0]**2
    physical_weights=gaussian_weights*np.exp(shift-interpolator(nodes)-envelope(nodes))
    return nodes,physical_weights


def profiled_gaussian_fourier_quadrature(hydrogen_half_widths,envelope_widths,total_degree,
        hydrogen_quadratures,fourier_order,periodic_quadrature,*,profiles,moment_order=1024):
    """Return the original trial basis and derivatives on positive adapted nodes.

    Each profile is (axis_points, log_density), covering the complete domain.
    Values and -i derivatives include sqrt(physical quadrature weight). Apply
    physical-volume orthonormalization afterward, as for any weak-measure rule.
    """
    limits=np.asarray(hydrogen_half_widths,float);widths=np.asarray(envelope_widths,float)
    nh=len(limits);degree=int(total_degree);kw=int(fourier_order)
    if (widths.shape!=limits.shape or len(hydrogen_quadratures)!=nh or len(profiles)!=nh+1
            or degree<0 or kw<0 or periodic_quadrature<=2*kw):
        raise ValueError('Compatible unchanged angular trial space and axis profiles required')
    nodes=[];weights=[];values=[];momenta=[]
    for i,(L,sigma,nq) in enumerate(zip(limits,widths,hydrogen_quadratures)):
        if nq<degree+1:raise ValueError('Quadrature cannot underresolve the polynomial trial degree')
        envelope=lambda x:2*np.log1p(-(x/L)**2)-(x/sigma)**2
        q,w=_profile_rule((-L,L),int(nq),*profiles[i],envelope,moment_order)
        original=finite_gaussian_axis(L,sigma,degree+1,max(degree+1,int(nq)),moment_order=moment_order)
        alpha=original['recurrence_alpha'];beta=original['recurrence_beta']
        P=np.zeros((len(q),degree+1));D=np.zeros_like(P);P[:,0]=original['polynomials'][0,0]
        for k in range(degree):
            P[:,k+1]=((q-alpha[k])*P[:,k]-(beta[k]*P[:,k-1] if k else 0))/beta[k+1]
            D[:,k+1]=(P[:,k]+(q-alpha[k])*D[:,k]-(beta[k]*D[:,k-1] if k else 0))/beta[k+1]
        f=(1-(q/L)**2)*np.exp(-.5*(q/sigma)**2);logarithmic=-2*q/(L*L-q*q)-q/sigma**2
        values.append(P*(f*np.sqrt(w))[:,None]);momenta.append(-1j*(D+logarithmic[:,None]*P)*(f*np.sqrt(w))[:,None])
        nodes.append(q);weights.append(w)
    q,w=_profile_rule((-np.pi,np.pi),int(periodic_quadrature),*profiles[-1],lambda x:np.zeros_like(x),moment_order)
    labels_k=np.arange(-kw,kw+1);fourier=np.exp(1j*q[:,None]*labels_k[None,:])*np.sqrt(w[:,None]/(2*np.pi))
    values.append(fourier);momenta.append(fourier*labels_k);nodes.append(q);weights.append(w)
    powers=[p for p in product(range(degree+1),repeat=nh) if sum(p)<=degree]
    labels=np.array([(*p,int(k)) for p in powers for k in labels_k],int)
    ids=np.stack(np.meshgrid(*[np.arange(len(x)) for x in nodes],indexing='ij'),axis=-1).reshape(-1,nh+1)
    columns=labels.copy();columns[:,-1]+=kw
    factors=[values[i][ids[:,i,None],columns[None,:,i]] for i in range(nh+1)]
    u=np.prod(np.stack(factors),axis=0);pu=[]
    for i in range(nh+1):
        f=factors.copy();f[i]=momenta[i][ids[:,i,None],columns[None,:,i]];pu.append(np.prod(np.stack(f),axis=0))
    return dict(points=np.stack([nodes[i][ids[:,i]] for i in range(nh+1)],axis=-1),
        weights=np.prod(np.stack([weights[i][ids[:,i]] for i in range(nh+1)]),axis=0),values=u,momenta=np.stack(pu,axis=1),occupations=labels,
        physical_bounds=np.array([[-L,L] for L in limits]+[[-np.pi,np.pi]]),basis_envelope_widths=widths,
        total_degree=degree,fourier_order=kw,quadrature_kind='positive profile-adapted Gaussian',
        unweighted_overlap_error=float(np.max(abs(u.conj().T@u-np.eye(len(labels))))))
