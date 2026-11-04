"""Full-mode rectilinear alpha constants and separate observable contraction.

The metric/alpha contractions use the same conventions as sonic_rovibrational.
At a saddle, complex analytic continuation preserves all mixed couplings.
The unstable diagonal alpha rows remain complex; a physical hybrid DeltaVib
requires an explicit subsequent choice of omitted diagonal rows.
"""
from dataclasses import dataclass

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_gf.harmonic import CM_PER_HARTREE
from matrix_rovib.isotopic_cubic import rovibrational_kinematics_from_modes
from .sonic_rovibrational import _ROTATIONAL_METRIC_CM, _CM_PER_MHZ


@dataclass(frozen=True)
class RectilinearAlphaConstants:
    frequencies_cm1: np.ndarray
    equilibrium_b_mhz: np.ndarray
    alpha_mhz: np.ndarray
    metric_zero_cm: np.ndarray
    metric_first_cm: np.ndarray
    metric_second_diagonal_cm: np.ndarray


def rectilinear_alpha_constants(
    coordinates_angstrom, masses_amu, modes_mw, frequencies_cm1,
    cubic_normal_mw, *, frequency_tolerance_cm=0.01,
    coriolis_resonance_threshold_cm=0.01,
) -> RectilinearAlphaConstants:
    """Calculate every alpha row without removing modes from the field.

    Real mode coordinates have length scale sqrt(K/|omega|),
    K=hbar²/(amu bohr² hc). For omega=i|omega| their analytic scale is
    exp(-i*pi/4) times that real scale. Applying these factors to both
    the potential and the full metric continues the usual alpha formula.
    Coriolis resonance deperturbation, if requested, affects alpha rows only;
    it never modifies the force field. The threshold is explicit and separate
    from the numerical tolerance for exactly degenerate modes.
    This routine handles nonlinear molecules in a rectilinear Eckart chart.
    """
    masses = np.asarray(masses_amu, dtype=float)
    w = np.asarray(frequencies_cm1, dtype=float)
    f3 = np.asarray(cubic_normal_mw, dtype=float)
    n = w.size
    if (w.ndim != 1 or np.any(abs(w) <= frequency_tolerance_cm)
            or f3.shape != (n,)*3 or not np.all(np.isfinite(w))
            or not np.all(np.isfinite(f3)) or frequency_tolerance_cm <= 0
            or not np.isfinite(coriolis_resonance_threshold_cm)
            or coriolis_resonance_threshold_cm < frequency_tolerance_cm):
        raise ValueError("finite complete cubic and nonzero signed frequencies required")
    kin = rovibrational_kinematics_from_modes(coordinates_angstrom, masses, modes_mw)
    if len(kin.modes_principal_mw) != n or n != 3*len(masses)-6:
        raise ValueError("nonlinear full vibrational mode set required")
    xyz = kin.coordinates_principal_A / float(BOHR_TO_ANGSTROM)
    scale = np.sqrt(_ROTATIONAL_METRIC_CM / abs(w))
    displacements = kin.modes_principal_mw / np.sqrt(masses)[None,:,None]
    displacements = displacements * scale[:,None,None]
    velocities = np.concatenate((displacements, np.cross(np.eye(3)[:,None,:],xyz[None,:,:])))
    covariant = np.einsum('aic,i,bic->ab', velocities, masses, velocities)
    inverse = np.linalg.inv(covariant)
    g0 = _ROTATIONAL_METRIC_CM * inverse
    g1 = np.empty((n+3,n+3,n)); g2 = np.empty_like(g1)
    for i in range(n):
        dv = np.zeros_like(velocities)
        dv[n:] = np.cross(np.eye(3)[:,None,:], displacements[i][None,:,:])
        first = np.einsum('aic,i,bic->ab',dv,masses,velocities)
        first = first + first.T
        second = 2*np.einsum('aic,i,bic->ab',dv,masses,dv)
        g1[:,:,i] = -_ROTATIONAL_METRIC_CM * inverse @ first @ inverse
        g2[:,:,i] = _ROTATIONAL_METRIC_CM * (
            2*inverse @ first @ inverse @ first @ inverse - inverse @ second @ inverse)
    if (np.max(abs(g0[:n,n:]), initial=0) > 1e-6 or
            not np.allclose(g0[:n,:n],np.diag(abs(w)),rtol=1e-8,atol=1e-6)):
        raise ValueError("normal modes must use the full mass metric and equilibrium Eckart frame")
    reduced_f3 = f3 * CM_PER_HARTREE * np.einsum('i,j,k->ijk',scale,scale,scale)
    phases = np.where(w < 0, np.exp(-.25j*np.pi), 1+0j)
    omega = np.where(w < 0, 1j*abs(w), w+0j)
    alpha = np.zeros((n,3), dtype=complex)
    for i,wi in enumerate(omega):
        for axis in range(3):
            r = n+axis
            alpha[i,axis] = -g2[r,r,i]*phases[i]**2/4
            for j,wj in enumerate(omega):
                rho = reduced_f3[j,i,i]*phases[j]*phases[i]**2
                alpha[i,axis] += rho*g1[r,r,j]*phases[j]/(4*wj)
                if abs(wi-wj) <= coriolis_resonance_threshold_cm:
                    continue
                a = g1[i,r,j]*phases[j]/phases[i]
                b = g1[j,r,i]*phases[i]/phases[j]
                alpha[i,axis] -= ((a*a+b*b)*wj-2*a*b*wi)/(2*(wi*wi-wj*wj))
    for i in range(n):
        partners = np.where(abs(omega-omega[i]) <= frequency_tolerance_cm)[0]
        if len(partners)>1:
            alpha[partners] = alpha[partners].mean(axis=0)
    return RectilinearAlphaConstants(
        w.copy(), np.diag(g0[n:,n:])/(2*_CM_PER_MHZ), alpha/_CM_PER_MHZ,
        g0,g1,g2,
    )


def contract_hybrid_deltavib(
    alpha_mhz, *, omitted_diagonal_modes: tuple[int,...] = (), additive_dvr_mhz,
    model_vpt2_mhz=None, model_alpha_mhz=None, model_omitted_diagonal_modes=None,
) -> dict:
    """Select alpha rows only at the observable stage; never alter the field.

    Indices are zero based. Mixed interactions already present in retained
    alpha rows stay present. The additive DVR vector is supplied explicitly
    by the caller, so parent reuse cannot happen implicitly.

    For a reduced curvilinear DVR model, supply its own perturbative correction
    as ``model_vpt2_mhz``. The result is full molecular VPT2 minus reduced-model
    VPT2 plus reduced-model DVR. This preserves molecular mixed couplings and
    matches the geometry, metric and coordinate convention of the replacement.
    Whole molecular alpha rows and a model counterterm cannot both be removed.
    Alternatively provide the full matched ``model_alpha_mhz`` matrix. The
    same observable row policy is then applied to both complete alpha fields
    before their difference is formed. Their modes must already be matched.
    A reduced model with its own normal modes must instead supply its own
    explicit ``model_omitted_diagonal_modes``; no molecular row is implicitly
    identified with a reduced-model coordinate.
    """
    alpha = np.asarray(alpha_mhz, dtype=complex)
    dvr = np.asarray(additive_dvr_mhz, dtype=float)
    omitted = tuple(omitted_diagonal_modes)
    if (alpha.ndim != 2 or alpha.shape[1] != 3 or dvr.shape != (3,)
            or not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(dvr))
            or len(set(omitted)) != len(omitted)
            or any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(alpha) for i in omitted)):
        raise ValueError("finite alpha/DVR and unique valid omitted mode indices required")
    retained = np.ones(len(alpha),dtype=bool)
    retained[list(omitted)] = False
    counterterm = None
    if model_omitted_diagonal_modes is not None and model_alpha_mhz is None:
        raise ValueError('A model row policy requires its complete alpha matrix')
    if model_alpha_mhz is not None:
        if model_vpt2_mhz is not None:
            raise ValueError('Supply either matched model alpha rows or a contracted model vector')
        model_alpha = np.asarray(model_alpha_mhz,dtype=complex)
        if (model_alpha.ndim!=2 or model_alpha.shape[1]!=3 or not np.all(np.isfinite(model_alpha))
                or (model_omitted_diagonal_modes is None and model_alpha.shape!=alpha.shape)):
            raise ValueError('Complete model alpha and an explicit separate policy for reduced-model modes required')
        model_omitted=omitted if model_omitted_diagonal_modes is None else tuple(model_omitted_diagonal_modes)
        if (len(set(model_omitted))!=len(model_omitted) or
                any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(model_alpha) for i in model_omitted)):
            raise ValueError('Unique valid model alpha indices required')
        model_retained=np.ones(len(model_alpha),bool);model_retained[list(model_omitted)]=False
        counterterm = -.5*model_alpha[model_retained].sum(axis=0)
        vpt2 = -.5*alpha[retained].sum(axis=0)-counterterm
    elif model_vpt2_mhz is not None:
        if omitted:
            raise ValueError("model counterterm and omitted molecular alpha rows are mutually exclusive")
        counterterm = np.asarray(model_vpt2_mhz, dtype=complex)
        if counterterm.shape != (3,) or not np.all(np.isfinite(counterterm)):
            raise ValueError("model VPT2 counterterm must be a finite three-axis vector")
        vpt2 = -.5 * alpha.sum(axis=0) - counterterm
    else:
        vpt2 = -.5 * alpha[retained].sum(axis=0)
    if np.max(abs(vpt2.imag),initial=0)>1e-8:
        raise ValueError("retained alpha sum is complex; unstable rows require an explicit observable policy")
    result = {'retained_vpt2_mhz':vpt2.real, 'additive_dvr_mhz':dvr.copy(),
              'delta_vib_mhz':vpt2.real+dvr, 'omitted_diagonal_modes':omitted}
    if counterterm is not None:
        result['model_vpt2_mhz'] = counterterm.copy()
        result['replacement_policy'] = ('matched_retained_vpt2_minus_model_retained_vpt2_plus_model_dvr'
            if model_alpha_mhz is not None else 'full_vpt2_minus_model_vpt2_plus_model_dvr')
        if model_alpha_mhz is not None:
            result['full_retained_vpt2_mhz'] = -.5*alpha[retained].sum(axis=0)
            result['model_omitted_diagonal_modes'] = model_omitted
    return result
