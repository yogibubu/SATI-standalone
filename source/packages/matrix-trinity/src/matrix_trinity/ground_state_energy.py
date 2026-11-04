"""Anharmonic ground energy without excited-state resonance denominators.

E0 = sum(w)/2 + <0|H4|0> - sum_{n!=0}|<n|H3|0>|^2/(n.w).
All frequencies are positive, so every denominator is a positive frequency
sum. No HDC regularization, resonance screen, or fundamental/X matrix is used.
The quartic term includes G2 and the supplied metric pseudopotential; H3
includes G1 in the same ordered curvilinear Hamiltonian used by TRINITY.
"""
import numpy as np
from .curvilinear_vpt2 import _modalize_sonic_field, _quartic_expectation, _cubic_couplings


def anharmonic_zero_point_energy(native_field):
    """Ground-only VPT2 in the native ESALC-I metric; exact resonances allowed."""
    w,_,f3,f4,g1,g2=_modalize_sonic_field(native_field)
    ground=(0,)*len(w)
    fourth=_quartic_expectation(ground,f4,g2,native_field.pseudopotential_cm,0.)
    cubic=0.
    for state,coupling in _cubic_couplings(ground,f3,g1,0.).items():
        if state==ground:
            if abs(coupling)>1e-10:
                raise ValueError('odd cubic operator has a nonzero vacuum expectation')
            continue
        gap=float(np.dot(w,state))
        if gap<=0:
            raise ValueError('ground correction requires positive transverse frequencies')
        cubic-=abs(coupling)**2/gap
    energy=.5*w.sum()+fourth+cubic
    if not np.isfinite(energy) or abs(np.imag(energy))>1e-9:
        raise ValueError('finite real perturbative ground energy required')
    return dict(schema='matrix.trinity.anharmonic_zpe.v1',zero_point_cm=float(np.real(energy)),
        harmonic_cm=float(.5*w.sum()),quartic_kinetic_cm=float(np.real(fourth)),
        cubic_second_order_cm=float(cubic),resonance_denominators=False,
        hdc_regularization=False)


def saddle_channel_ground_energy(normal_form, reactive_action):
    """Ground transverse channel at fixed real action of the saddle QNF.

    The scalar barrier and reactive energy are included once. Already eliminated
    virtual terms are in Z4; only retained Z3 contributes to this vacuum sum.
    This is a conditional channel origin, not a bound saddle-point ZPE.
    """
    from .transition_state import _weyl_ladder, _real
    I=_real(reactive_action,'reactive_action')
    if normal_form.normal_cubic is None or normal_form.normal_quartic is None:
        raise ValueError('separate normal Z3/Z4 needed for ground self-energy accounting')
    r=normal_form.field.reaction_mode
    stable=[i for i in range(len(normal_form.field.signed_frequencies_cm)) if i!=r]
    w=normal_form.field.signed_frequencies_cm[stable]
    ground=(0,)*len(w)
    def apply(poly):
        rows={}
        for powers,c in poly.items():
            a,b=powers[2*r:2*r+2]
            if a!=b or a>2:
                raise ValueError('normal form must conserve reactive action through order four')
            value=c*(1. if a==0 else I if a==1 else I*I-.25)
            target=[]
            for mode in stable:
                occupation,factor=_weyl_ladder(powers[2*mode],powers[2*mode+1],0)
                target.append(occupation);value*=factor
            key=tuple(target)
            rows[key]=rows.get(key,0.)+value
        return rows
    value=(normal_form.field.barrier_energy_cm
        +abs(normal_form.field.signed_frequencies_cm[r])*I+.5*w.sum()
        +apply(normal_form.normal_quartic).get(ground,0.))
    for state,coupling in apply(normal_form.normal_cubic).items():
        if state==ground:
            value+=coupling
        elif coupling:
            gap=float(np.dot(w,state))
            if gap<=0:
                raise ValueError('positive transverse frequency sums required')
            value-=abs(coupling)**2/gap
    if not np.isfinite(value) or abs(np.imag(value))>1e-8:
        raise ValueError('nonreal fixed-action channel ground energy')
    return float(np.real(value))
