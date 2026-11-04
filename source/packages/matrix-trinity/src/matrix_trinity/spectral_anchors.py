"""Reusable low spectral anchors from independently converged eigensolutions.

Diagonalization occurs once before occupation counting. The returned energy
model is explicit and quadratic, so existing thermochemical and Wang--Landau
consumers do not diagonalize an operator at each occupation. Absolute vacuum
energies and couplings outside the selected block remain caller-owned.
"""
import itertools
import numpy as np


def variational_spectral_anchors(levels_cm, labels, overlap_squared, *,
        source_identifier, maximum_level_change_cm, convergence_tolerance_cm=.1,
        minimum_squared_overlap=.5):
    """Assign all ground/one/two-quanta anchors, retaining complete manifolds.

    Overlaps refer to normalized harmonic products in the declared modal gauge.
    A weak/ambiguous individual assignment is rejected; unresolved resonances
    require a complete low spectral block rather than invented occupations.
    No validation/reference energy is accepted as a fitting argument.
    """
    from scipy.optimize import linear_sum_assignment
    from .hdc_polynomial import HDCPolynomialEnergy
    levels=np.asarray(levels_cm,float);states=np.asarray(labels);overlap=np.asarray(overlap_squared,float)
    if (levels.ndim!=1 or not len(levels) or not np.all(np.isfinite(levels))
            or np.any(np.diff(levels)<0) or states.ndim!=2 or not states.shape[1]
            or states.dtype.kind not in 'iu' or np.any(states<0)
            or overlap.shape!=(len(states),len(levels)) or not np.all(np.isfinite(overlap))
            or np.any(overlap<0) or np.any(overlap>1+1e-9)
            or np.any(overlap.sum(axis=0)>1+1e-8) or np.any(overlap.sum(axis=1)>1+1e-8)
            or not isinstance(source_identifier,str) or not source_identifier.strip()
            or not np.all(np.isfinite([maximum_level_change_cm,convergence_tolerance_cm,minimum_squared_overlap]))
            or maximum_level_change_cm<0 or convergence_tolerance_cm<=0
            or not .5<=minimum_squared_overlap<1):
        raise ValueError('ordered eigenvalues, complete occupation labels, normalized overlaps and identified convergence required')
    if maximum_level_change_cm>convergence_tolerance_cm:
        raise ValueError('variational low anchors have not numerically converged')
    modes=states.shape[1]
    required=set()
    for quanta in range(3):
        for occupied in itertools.combinations_with_replacement(range(modes),quanta):
            state=[0]*modes
            for i in occupied:state[i]+=1
            required.add(tuple(state))
    lookup={tuple(s):i for i,s in enumerate(states)}
    if len(lookup)!=len(states) or set(lookup)!=required or len(levels)<len(states):
        raise ValueError('exactly one complete zero/one/two-quanta occupation manifold required')
    rows,columns=linear_sum_assignment(-overlap)
    if len(rows)!=len(states):raise ValueError('not enough variational roots for the low manifold')
    confidence=overlap[rows,columns]
    if np.any(confidence<=minimum_squared_overlap):
        raise ValueError('low eigenstates cannot be assigned unambiguously to occupation anchors')
    assigned=np.empty(len(states),int);assigned[rows]=columns
    zero=lookup[(0,)*modes]
    if assigned[zero]!=0:raise ValueError('assigned occupation ground is not the variational ground')
    excitations=levels[assigned]-levels[0]
    nu=np.empty(modes);x=np.empty((modes,modes))
    for i in range(modes):
        single=tuple(int(k==i) for k in range(modes))
        overtone=tuple(2*int(k==i) for k in range(modes))
        nu[i]=excitations[lookup[single]]
        x[i,i]=(excitations[lookup[overtone]]-2*nu[i])/2
    for i in range(modes):
        for j in range(i):
            combination=tuple(int(k in (i,j)) for k in range(modes))
            x[i,j]=x[j,i]=excitations[lookup[combination]]-nu[i]-nu[j]
    model=HDCPolynomialEnergy(0.,nu,x,coordinate_id=source_identifier)
    if not np.allclose(model.energies(states),excitations,rtol=0,atol=1e-8):
        raise RuntimeError('quadratic low occupation anchors do not reproduce the supplied manifold')
    return dict(model=model,assigned_root_indices=assigned,assignment_squared_overlaps=confidence,
        low_excitations_cm=excitations,source_identifier=source_identifier,
        maximum_level_change_cm=float(maximum_level_change_cm),
        absolute_ground_energy_replaced=False,diagonalization_per_occupation=False)


def replace_spectral_anchor_block(energy_model, mode_indices, anchor_model, *, perturbative_block=None):
    """Replace a selected block, retaining virtual corrections from its bath.

    For a proper subspace, subtract its independently evaluated perturbative
    block before adding the variational block. Simply overwriting the full
    model's coefficients would lose corrections involving spectator modes.
    Vacuum energy remains owned by the independent ground-channel treatment.
    """
    from .hdc_polynomial import HDCPolynomialEnergy
    if not isinstance(energy_model,HDCPolynomialEnergy) or not isinstance(anchor_model,HDCPolynomialEnergy):
        raise TypeError('quadratic occupation energy models required')
    indices=np.asarray(mode_indices)
    if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices)
            or len(set(indices.tolist()))!=len(indices) or np.any(indices<0)
            or np.any(indices>=len(energy_model.fundamentals_cm))
            or len(indices)!=len(anchor_model.fundamentals_cm)
            or energy_model.y_tensor_cm is not None or anchor_model.y_tensor_cm is not None):
        raise ValueError('distinct valid modes and quadratic anchor block required')
    nu=energy_model.fundamentals_cm.copy();x=energy_model.x_matrix_cm.copy()
    if perturbative_block is None:
        if len(indices)!=len(nu):
            raise ValueError('a proper subspace requires its perturbative block for subtraction')
        nu[indices]=anchor_model.fundamentals_cm;x[np.ix_(indices,indices)]=anchor_model.x_matrix_cm
    else:
        if (not isinstance(perturbative_block,HDCPolynomialEnergy)
                or len(perturbative_block.fundamentals_cm)!=len(indices)
                or perturbative_block.y_tensor_cm is not None):
            raise ValueError('matching quadratic perturbative block required')
        nu[indices]+=anchor_model.fundamentals_cm-perturbative_block.fundamentals_cm
        x[np.ix_(indices,indices)]+=anchor_model.x_matrix_cm-perturbative_block.x_matrix_cm
    return HDCPolynomialEnergy(energy_model.zero_point_cm,nu,x,
        coordinate_id=energy_model.coordinate_id)
