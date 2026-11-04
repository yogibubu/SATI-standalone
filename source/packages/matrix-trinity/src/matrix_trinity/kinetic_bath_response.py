"""Bath response to active-mode kinetic forces in a contracted DVR basis."""
from __future__ import annotations
import numpy as np


def _normal(kinetic, force):
    g, h = np.asarray(kinetic, float), np.asarray(force, float)
    if (g.ndim != 2 or g.shape[0] != g.shape[1] or h.shape != g.shape
            or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h))
            or not np.allclose(g, g.T) or not np.allclose(h, h.T)):
        raise ValueError('finite symmetric oscillator matrices required')
    values, vectors = np.linalg.eigh(g)
    if np.min(values) <= 0:
        raise ValueError('positive oscillator kinetic matrix required')
    half = (vectors*np.sqrt(values))@vectors.T
    squared, modes = np.linalg.eigh(half@h@half)
    if np.min(squared) <= 0:
        raise ValueError('positive oscillator force matrix required')
    w = np.sqrt(squared)
    return w, half@(modes/np.sqrt(w)), np.linalg.solve(half, modes*np.sqrt(w))


def dress_active_kinetic_bath_response(
    energies_cm, grid_vectors, grid_momenta, kinetic_bath_gradient_cm,
    rotational_bath_gradient_cm, active_kinetic_zero_cm, active_force_zero_cm,
    bath_kinetic_zero_cm, bath_force_zero_cm,
    kinetic_bath_gradient_zero_cm, rotational_bath_gradient_zero_cm,
):
    """Dress the linear bath force induced by active kinetic energy.

    The zero-J vertex is K_b=1/2 sum_st p_s (dG_st/db_b) p_t.
    The rotational vertex is sum_b (dB_ab/db_b) b_b J_a J_b;
    the B gradient is of the actual quadratic coefficient, G_RR/2.
    Both are contracted in weak form using unprojected pU, then propagated
    through the LAM plus fixed harmonic-bath one-quantum resolvent. Return
    the response minus its own harmonic counterpart for addition to an
    existing full-VPT2-minus-model-VPT2-plus-DVR observable.

    Derivative fields have shapes (grid,bath,LAM,LAM) and (grid,bath,3,3).
    Bath frequencies/fibers are fixed at equilibrium. This leading linear
    bath-force response does not include a variationally coupled bath ground
    state, variable fibers, or higher-order rotational virtual cross terms.
    """
    e = np.asarray(energies_cm, float)
    u, pu = np.asarray(grid_vectors, complex), np.asarray(grid_momenta, complex)
    kg, rg = np.asarray(kinetic_bath_gradient_cm, float), np.asarray(rotational_bath_gradient_cm, float)
    k0, r0 = np.asarray(kinetic_bath_gradient_zero_cm, float), np.asarray(rotational_bath_gradient_zero_cm, float)
    if u.ndim != 2 or pu.ndim != 3:
        raise ValueError('retained DVR vectors and unprojected momenta required')
    ng, nr = u.shape
    dim = pu.shape[1]
    if kg.ndim != 4:
        raise ValueError('kinetic gradient must have four indices')
    nb = kg.shape[1]
    if (e.shape != (nr,) or pu.shape != (ng,dim,nr)
            or kg.shape != (ng,nb,dim,dim) or rg.shape != (ng,nb,3,3)
            or k0.shape != (nb,dim,dim) or r0.shape != (nb,3,3)
            or not all(np.all(np.isfinite(v)) for v in (e,u,pu,kg,rg,k0,r0))
            or not np.allclose(u.conj().T@u,np.eye(nr),atol=1e-8)
            or not all(np.allclose(v,v.swapaxes(-1,-2),atol=1e-8) for v in (kg,rg,k0,r0))
            or np.any(np.diff(e)<0)):
        raise ValueError('compatible finite symmetric kinetic-force fields required')
    wb, qb, _ = _normal(bath_kinetic_zero_cm, bath_force_zero_cm)
    _, _, pa = _normal(active_kinetic_zero_cm, active_force_zero_cm)
    if qb.shape != (nb,nb) or pa.shape != (dim,dim):
        raise ValueError('oscillator dimensions differ from force fields')
    Q = qb/np.sqrt(2)
    force = .5*np.einsum('gsn,gbst,gt,bi->ni',pu.conj(),kg,pu[:,:,0],Q,optimize=True)
    rotation = np.einsum('gn,g,gbac,bi->niac',u.conj(),u[:,0],rg,Q,optimize=True)
    denominator = e[:,None]-e[0]+wb[None,:]
    response = -2*np.real(np.einsum('niac,ni->ac',rotation.conj(),force/denominator))
    harmonic_force = .5*np.einsum('bst,st->b',k0,pa@pa.T/2)
    harmonic_mean = -np.linalg.solve(np.asarray(bath_force_zero_cm),harmonic_force)
    harmonic = np.einsum('b,bac->ac',harmonic_mean,r0)
    return dict(response_tensor_cm=response,harmonic_counterterm_tensor_cm=harmonic,
                replacement_tensor_cm=response-harmonic,
                harmonic_bath_force_cm=harmonic_force,
                minimum_resolvent_gap_cm=float(np.min(denominator)),
                retained_lam_states=nr,bath_dimension=nb,
                definition='active_kinetic_linear_bath_force_rotational_response')


__all__ = ['dress_active_kinetic_bath_response']
