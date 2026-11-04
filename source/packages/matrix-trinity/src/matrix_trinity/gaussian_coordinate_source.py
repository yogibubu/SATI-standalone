"""Coordinate sources obtained from the existing FC/HT Bargmann generator.

The initial state and omitted final oscillators are vacuum. Omitted final
variables are set to zero, not traced: this is a selected spectator-ground
channel, never a claim of a complete molecular spectrum.
"""
from dataclasses import dataclass
import numpy as np
from .vibronic_harmonic import (harmonic_gaussian_generator, orient_harmonic_model,
    harmonic_strength_sum_rule)
from .gaussian_partial_trace import gaussian_vacuum_operator_trace_batch
from .vibronic_ti import _integral_dag


@dataclass(frozen=True)
class GaussianVacuumCoordinateSource:
    precision: np.ndarray
    linear: np.ndarray
    log_amplitude: float
    constant_fc: np.ndarray
    constant_ht: np.ndarray
    linear_ht: np.ndarray
    diagnostics: dict

    def evaluate(self, points):
        q=np.asarray(points,float)
        if q.ndim!=2 or q.shape[1]!=len(self.linear) or not np.all(np.isfinite(q)):
            raise ValueError('finite points in the identified active dimensionless HO coordinates required')
        envelope=np.exp(self.log_amplitude-.5*np.einsum('pi,ij,pj->p',q,self.precision,q)+q@self.linear)
        return dict(envelope=envelope,FC=envelope[:,None]*self.constant_fc,
            HT=envelope[:,None]*(self.constant_ht+q@self.linear_ht.T))


def _active_generator(model,active_final_modes,kind):
    oriented,_=orient_harmonic_model(model,kind);n=len(oriented.initial_frequencies_cm)
    active=np.asarray(active_final_modes)
    if (active.ndim!=1 or not len(active) or active.dtype.kind not in 'iu'
            or len(np.unique(active))!=len(active) or np.any(active<0) or np.any(active>=n)):
        raise ValueError('unique nonempty final reference modes required')
    g=harmonic_gaussian_generator(oriented);ids=n+active
    return oriented,g,np.asarray(g['matrix'])[np.ix_(ids,ids)],np.asarray(g['linear'])[ids],ids


def gaussian_vacuum_coordinate_source(model,active_final_modes,*,kind='absorption'):
    """Invert only retained Bargmann variables to a flat coordinate source.

    Full Duschinsky/Jacobian and every vacuum spectator overlap are already
    owned by harmonic_gaussian_generator. Only the inverse Bargmann adapter
    is added here. FC and HT phases remain available for coherent contractions.
    """
    oriented,g,A,r,ids=_active_generator(model,active_final_modes,kind)
    d=len(r);B=2*np.linalg.solve(np.eye(d)+A,np.eye(d));P=(B+B.T)/2-np.eye(d)
    if np.min(np.linalg.eigvalsh(P))<=0:
        raise ValueError('coordinate source precision is not positive; no clipping allowed')
    b=np.sqrt(2)*np.linalg.solve(np.eye(d)+A,r)
    sign,logdet=np.linalg.slogdet(B)
    if sign<=0:raise ValueError('positive inverse Bargmann precision required')
    logN=(g['log_ground']+d*np.log(np.pi)/4-d*np.log(2*np.pi)/2
        +.5*logdet-.5*b@np.linalg.solve(B,b))
    L=np.asarray(g['ht_linear'])[:,ids]
    v=L@B/np.sqrt(2);u=np.asarray(g['ht_constant'])-L@b/np.sqrt(2)
    mean=np.linalg.solve(P,b);covariance=.5*np.linalg.inv(P)
    norm=np.exp(2*logN+d*np.log(np.pi)/2-.5*np.linalg.slogdet(P)[1]+b@mean)
    mu=oriented.transition_dipole_au;htmean=u+v@mean
    components=dict(FC=float(norm*np.vdot(mu,mu).real),
        HT=float(norm*(np.vdot(htmean,htmean).real+np.einsum('ai,ij,aj->',v,covariance,v.conj()).real)),
        interference=float(2*norm*np.vdot(mu,htmean).real))
    # Independent existing Mehler owner certifies all operator norms, with no
    # active occupation cutoff and no numerical coordinate quadrature.
    bank=np.r_[mu,np.asarray(g['ht_constant'])]
    insertion=np.vstack([np.zeros_like(L),L])
    traced=gaussian_vacuum_operator_trace_batch(A,r,g['log_ground'],np.ones((1,d)),bank,insertion)
    K=traced['operator_kernel'][0]
    reference=dict(FC=float(np.trace(K[:3,:3]).real),HT=float(np.trace(K[3:,3:]).real),
        interference=float(2*np.trace(K[:3,3:]).real))
    error=max(abs(components[k]-reference[k]) for k in components)
    if error>1e-10*max(sum(abs(x) for x in reference.values()),np.finfo(float).tiny):
        raise ArithmeticError('coordinate FC/HT source violates the existing Mehler norm')
    diagnostics=dict(active_final_modes=np.asarray(active_final_modes).tolist(),
        initial_state='harmonic vacuum',omitted_final_modes='vacuum projection, not occupation trace',
        strength_components=components,mehler_strength_components=reference,
        maximum_mehler_strength_error=error,scalar_source_norm=float(norm),
        full_initial_strength_components=harmonic_strength_sum_rule(oriented,operator='FCHT'),
        source_norm_changed=False,complete_spectrum_claimed=False)
    return GaussianVacuumCoordinateSource(P,b,float(logN),mu.copy(),u,v,diagnostics)


def gaussian_vacuum_operator_coefficients(model,active_final_modes,occupations,*,
        kind='absorption',maximum_integrals=1000000):
    """Phase-preserving FC/HT coefficients through the shared TI recurrence."""
    oriented,g,A,r,ids=_active_generator(model,active_final_modes,kind)
    states=np.asarray(occupations)
    if (states.ndim!=2 or states.shape[1]!=len(ids) or states.dtype.kind not in 'iu'
            or not len(states) or np.any(states<0)):
        raise ValueError('nonnegative active reference occupations required')
    htlinear=np.asarray(g['ht_linear'])[:,ids]
    indices,values=_integral_dag(states,A,r,htlinear,maximum_integrals,np.exp(g['log_ground']))
    fc=values[indices[:,0],None]*oriented.transition_dipole_au[None,:]
    ht=values[indices[:,0],None]*np.asarray(g['ht_constant'])[None,:]
    for axis in range(len(ids)):
        selected=indices[:,axis+1]>=0
        ht[selected]+=((np.sqrt(states[selected,axis])*values[indices[selected,axis+1]])[:,None]
            *htlinear[:,axis][None,:])
    return dict(FC=fc,HT=ht,diagnostics=dict(recurrence_integrals=len(values),
        full_gaussian_generator_retained=True,source_norm_changed=False,
        spectator_state='vacuum',complete_spectrum_claimed=False))


@dataclass(frozen=True)
class GaussianMixedCoordinateSource:
    """Continuous retained coordinates times requested spectator Fock words."""
    coordinate_source: GaussianVacuumCoordinateSource
    fock_matrix: np.ndarray
    fock_linear: np.ndarray
    coordinate_to_fock_linear: np.ndarray
    fock_ht_linear: np.ndarray
    occupations: np.ndarray
    maximum_integrals: int
    diagnostics: dict

    def evaluate(self,points,*,return_overlap=False):
        """Evaluate FC/HT and optionally the scalar mixed overlap.

        The scalar channel allows a fitted coordinate-dependent property to
        be contracted without dividing by a possibly vanishing dipole.
        """
        q=np.asarray(points,float);base=self.coordinate_source.evaluate(q)
        n=len(self.occupations)
        fc=np.empty((len(q),n,3),complex);ht=np.empty_like(fc)
        overlap=np.empty((len(q),n),complex) if return_overlap else None
        mu=self.coordinate_source.constant_fc
        for p,point in enumerate(q):
            r=self.fock_linear+self.coordinate_to_fock_linear@point
            ids,values=_integral_dag(self.occupations,self.fock_matrix,r,
                self.fock_ht_linear,self.maximum_integrals,1.)
            c=values[ids[:,0]]
            if return_overlap:overlap[p]=c*base['envelope'][p]
            constant=self.coordinate_source.constant_ht+self.coordinate_source.linear_ht@point
            fc[p]=c[:,None]*mu
            ht[p]=c[:,None]*constant
            for axis in range(self.occupations.shape[1]):
                live=ids[:,axis+1]>=0
                ht[p,live]+=(np.sqrt(self.occupations[live,axis])*values[ids[live,axis+1]])[:,None]*self.fock_ht_linear[:,axis]
            fc[p]*=base['envelope'][p];ht[p]*=base['envelope'][p]
        result=dict(FC=fc,HT=ht)
        if return_overlap:result['overlap']=overlap
        return result


def gaussian_mixed_coordinate_source(model,coordinate_final_modes,fock_final_modes,
        occupations,*,kind='absorption',maximum_integrals=1000000):
    """Partially invert the existing FC/HT Bargmann generator, retaining Fock legs.

    The resident TI DAG owns every spectator coefficient and every HT lowered
    word. Unlisted final modes remain vacuum projected. Coordinate integrals
    use flat dimensionless HO measure, with no normalization restoration.
    """
    coordinates=np.asarray(coordinate_final_modes);fock=np.asarray(fock_final_modes)
    words=np.asarray(occupations)
    if (coordinates.ndim!=1 or fock.ndim!=1 or not len(fock)
            or coordinates.dtype.kind not in 'iu' or fock.dtype.kind not in 'iu'
            or len(set(coordinates.tolist()+fock.tolist()))!=len(coordinates)+len(fock)
            or words.ndim!=2 or words.shape[1]!=len(fock) or not len(words)
            or words.dtype.kind not in 'iu' or np.any(words<0)
            or isinstance(maximum_integrals,bool) or not isinstance(maximum_integrals,int) or maximum_integrals<1):
        raise ValueError('Disjoint coordinate/Fock mode lists and finite nonnegative Fock words required')
    _,g,A,r,ids=_active_generator(model,np.r_[coordinates,fock],kind)
    d=len(coordinates);aa=A[:d,:d];cross=A[:d,d:]
    inverse=np.linalg.solve(np.eye(d)+aa,np.eye(d))
    matrix=A[d:,d:]-cross.T@inverse@cross
    linear=r[d:]-cross.T@inverse@r[:d]
    coupling=np.sqrt(2)*cross.T@inverse
    insertion=np.asarray(g['ht_linear'])[:,ids]
    ht=insertion[:,d:]-insertion[:,:d]@inverse@cross
    base=gaussian_vacuum_coordinate_source(model,coordinates,kind=kind)
    diagnostics=dict(coordinate_modes=coordinates.tolist(),fock_modes=fock.tolist(),
        fock_words=len(words),full_initial_strength_components=base.diagnostics['full_initial_strength_components'],
        source_norm_changed=False,complete_spectrum_claimed=False,
        coefficient_owner='resident TI Gaussian recurrence',
        partial_inverse='analytic Schur contraction of the resident Bargmann generator')
    return GaussianMixedCoordinateSource(base,(matrix+matrix.T)/2,linear,coupling,ht,
        words.copy(),maximum_integrals,diagnostics)
