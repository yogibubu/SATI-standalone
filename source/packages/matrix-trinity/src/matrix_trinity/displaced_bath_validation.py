"""Contracted checks of a linear oscillator response on an existing active PES.

The selected oscillators replace their own adiabatic contribution. Frequencies
and force matrices are inputs; this module never acquires electronic points.
"""
import numpy as np
from scipy.sparse import diags, eye
from scipy.linalg import eigh
from .coupled_bath_galerkin import harmonic_bath_matrices, _TensorSumOperator


def linear_bath_hamiltonian(active_hamiltonian_cm, force_matrices_cm,
                            frequencies_cm, maximum_total_quanta):
    """Return H_active + sum(w*n + g*Q), with the bath ZPE removed.

    Exact oscillator closure is owned by harmonic_bath_matrices. No completed
    square or diagonal Born--Huang term may be included for these same modes
    in the supplied active Hamiltonian.
    """
    h=np.asarray(active_hamiltonian_cm,float)
    f=np.asarray(force_matrices_cm,float);w=np.asarray(frequencies_cm,float)
    if (h.ndim!=2 or h.shape[0]!=h.shape[1] or not len(h)
            or w.ndim!=1 or not len(w) or np.any(w<=0)
            or f.shape!=(len(w),)+h.shape
            or not all(np.all(np.isfinite(x)) for x in (h,f,w))
            or not np.allclose(h,h.T) or not np.allclose(f,f.swapaxes(1,2))):
        raise ValueError('Finite symmetric active/force matrices and positive bath frequencies required')
    bath=harmonic_bath_matrices(np.diag(w),np.diag(w),maximum_total_quanta)
    occupations=np.asarray(bath['occupations'])
    # Normal-mode ordering is sorted by the shared oscillator constructor.
    terms=[(h,eye(len(occupations),format='csr')),
           (np.eye(len(h)),diags(occupations@bath['frequencies_cm'],format='csr'))]
    terms.extend((f[j],bath['coordinate'][j]) for j in range(len(w)))
    return _TensorSumOperator(terms,len(h),len(occupations)),occupations


def adiabatic_oscillator_bands(base_cm,relaxation_cm,born_huang_cm,
                             frequencies_cm,occupations,state_count):
    """Order the adiabatic bands with each oscillator's (2n+1) BH factor.

    A lowest-band energy plus an unchanged oscillator ladder would omit this
    occupation dependence and confuse its error with nonadiabatic mixing.
    All supplied sectors that can enter the requested low spectrum are used.
    """
    base=np.asarray(base_cm);relax=np.asarray(relaxation_cm)
    bh=np.asarray(born_huang_cm);w=np.asarray(frequencies_cm)
    occ=np.asarray(occupations)
    if (base.ndim!=2 or base.shape[0]!=base.shape[1] or relax.shape!=base.shape
            or bh.shape!=(len(w),)+base.shape or occ.ndim!=2 or occ.shape[1]!=len(w)
            or np.any(occ<0) or not np.any(np.all(occ==0,axis=1))
            or not 0<state_count<=len(base)):
        raise ValueError('Compatible projected matrices, occupations and low state count required')
    ground_h=base+relax+bh.sum(axis=0)
    first=eigh(ground_h,eigvals_only=True,subset_by_index=(0,state_count-1))
    energies=[first];ceiling=first[-1]
    for ns in occ:
        if not np.any(ns):continue
        offset=float(ns@w)
        # Positive semidefinite displacement BH matrices can only raise a band.
        if first[0]+offset>ceiling:continue
        block=ground_h+2*np.einsum('j,jab->ab',ns,bh)
        energies.append(eigh(block,eigvals_only=True,subset_by_index=(0,state_count-1))+offset)
    return np.sort(np.concatenate(energies))[:state_count]
