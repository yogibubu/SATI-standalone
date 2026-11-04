"""Primary/auxiliary/spectator screening and published derivative coverage.

Hierarchy and alpha/beta factors: Barone, Lazzari, Mendolicchio,
JCTC 2026, 22, 2453-2466, doi:10.1021/acs.jctc.5c02123.
Screening does not certify rate accuracy or constancy along a reaction path.
"""
from collections import Counter
from itertools import combinations_with_replacement,permutations
import numpy as np


def screen_pas_modes(frequencies_cm,cubic_cm,mode_vectors,*,primary_modes,
                     alpha_threshold,beta_threshold,metric_first_cm=None,
                     reaction_mode=None,endpoint_overlap=None,
                     endpoint_reaction_threshold=.05,endpoint_mode_threshold=.5):
    """Alpha=|rho_ii,tau/omega_tau| and beta=sum_u |L_uP L_uS|.

    L columns must be normalized in one common orthonormal coordinate frame.
    Internal-coordinate vectors must be metric-whitened by their owner first.
    Endpoint_overlap maps TS columns into aligned endpoint normal coordinates.
    Significant endpoint modes overlapping the unstable TS mode seed additional
    primaries via their total squared overlap with a stable TS mode.
    """
    w=np.asarray(frequencies_cm,float);C=np.asarray(cubic_cm,float);L=np.asarray(mode_vectors,float)
    n=len(w);p=set(primary_modes)
    if (w.ndim!=1 or C.shape!=(n,)*3 or L.ndim!=2 or L.shape[1]!=n
        or not np.all(np.isfinite(w)) or np.any(w==0) or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(L)) or not np.allclose(np.sum(L*L,axis=0),1.,atol=1e-8)):
        raise ValueError('finite consistent fields and normalized common-frame mode columns required')
    if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<n for i in p):
        raise ValueError('valid primary mode indices required')
    thresholds=[alpha_threshold,beta_threshold,endpoint_reaction_threshold,endpoint_mode_threshold]
    if not np.all(np.isfinite(thresholds)) or min(thresholds)<0:raise ValueError('finite nonnegative thresholds required')
    if reaction_mode is not None:
        if isinstance(reaction_mode,bool) or not isinstance(reaction_mode,(int,np.integer)) or not 0<=reaction_mode<n or w[reaction_mode]>=0:
            raise ValueError('reaction mode must index the negative signed frequency')
        p.add(reaction_mode)
    if np.any(w[[i for i in range(n) if i!=reaction_mode]]<=0):raise ValueError('all nonreactive modes must be stable')
    G=np.zeros_like(C) if metric_first_cm is None else np.asarray(metric_first_cm,float)
    if G.shape!=C.shape or not np.all(np.isfinite(G)) or not np.allclose(G,G.swapaxes(0,1)):
        raise ValueError('finite metric first derivatives symmetric in momentum indices required')
    endpoint_weights=np.zeros(n);selected=[]
    if endpoint_overlap is not None:
        J=np.asarray(endpoint_overlap,float)
        if reaction_mode is None or J.ndim!=2 or J.shape[1]!=n or not np.all(np.isfinite(J)):
            raise ValueError('finite endpoint by TS overlap and reaction mode required')
        selected=np.flatnonzero(abs(J[:,reaction_mode])**2>=endpoint_reaction_threshold).tolist()
        endpoint_weights=np.sum(abs(J[selected])**2,axis=0)
        p.update(a for a in range(n) if a!=reaction_mode and endpoint_weights[a]>=endpoint_mode_threshold)
    # rho[i,i,tau] uses g_ii,tau - 2 g_i,tau,i in the paper's convention.
    alpha=np.array([[abs((C[i,i,a]-G[i,i,a]+2*G[i,a,i])/w[a]) if w[a]>0 else 0.
                     for a in range(n)] for i in sorted(p)])
    beta=abs(L[:,sorted(p)]).T@abs(L)
    maxalpha=np.max(alpha,axis=0) if len(p) else np.zeros(n)
    maxbeta=np.max(beta,axis=0) if len(p) else np.zeros(n)
    auxiliary=[a for a in range(n) if a not in p and (maxalpha[a]>alpha_threshold or maxbeta[a]>beta_threshold)]
    return dict(primary_modes=sorted(p),primary_modes_total=sorted(p),
        primary_transverse_modes=sorted(p-({reaction_mode} if reaction_mode is not None else set())),
        reactive_primary_mode=reaction_mode,auxiliary_modes=auxiliary,
        spectator_modes=[a for a in range(n) if a not in p and a not in auxiliary],
        alpha_scores=maxalpha,beta_scores=maxbeta,endpoint_scores=endpoint_weights,
        selected_endpoint_modes=selected,alpha_threshold=alpha_threshold,beta_threshold=beta_threshold,
        hierarchy_doi='10.1021/acs.jctc.5c02123',accuracy_certified=False)


def pas_coverage_projection(cubic,quartic,*,reaction_mode,primary_modes,auxiliary_modes,
                            mixed_auxiliary_quartics=True):
    """Retain all derivatives supplied by the published P/A scan hierarchy.

    Unlike a strict harmonic-spectator truncation, accessible P/S and A/S
    perturbative couplings remain. Q_aabb between auxiliaries is an optional
    mixed-energy extension beyond the paper's axial-gradient approximation.
    Unknown families are explicit model zeros, not measured derivatives.
    """
    C=np.asarray(cubic,float);Q=np.asarray(quartic,float)
    if C.ndim!=3 or C.shape!=(len(C),)*3 or Q.shape!=(len(C),)*4 or not np.all(np.isfinite(C)) or not np.all(np.isfinite(Q)):
        raise ValueError('finite matching potential tensors required')
    n=len(C);p=set(primary_modes);a=set(auxiliary_modes)
    if (p&a or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<n for i in [reaction_mode,*p,*a]) or reaction_mode in a):
        raise ValueError('valid disjoint primary/auxiliary modes required')
    p.add(reaction_mode);out=[];masks=[]
    for order,source in [(3,C),(4,Q)]:
        value=np.zeros_like(source);mask=np.zeros(source.shape,bool)
        for key in combinations_with_replacement(range(n),order):
            counts=Counter(key)
            if order==3:keep=bool(set(key)&p) or any(count>=2 and i in a for i,count in counts.items())
            else:keep=any(count>=2 and i in p or count>=3 and i in a for i,count in counts.items())
            if order==4 and mixed_auxiliary_quartics and sorted(counts.values())==[2,2] and set(key)<=a:keep=True
            if keep:
                for index in set(permutations(key)):value[index]=source[index];mask[index]=True
        out.append(value);masks.append(mask)
    return dict(cubic=out[0],quartic=out[1],cubic_observed=masks[0],quartic_observed=masks[1],
                mixed_auxiliary_quartics=mixed_auxiliary_quartics,
                retained_spectator_couplings=True,unacquired_are_model_zeros=True)
