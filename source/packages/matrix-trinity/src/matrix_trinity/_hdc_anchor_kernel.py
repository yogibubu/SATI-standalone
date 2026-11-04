"""Compiled HDC anchor sums with exact interference of equal virtual targets."""
from itertools import product, permutations
from math import copysign, hypot, tanh

import numpy as np
from numba import njit, prange


def diagonal_quartic_terms(field):
    """Only even modal parity can contribute to diagonal anchors: O(m^2)."""
    from .curvilinear_vpt2 import _canonical_operator_key
    terms = {}
    def add(operators, coefficient):
        if coefficient:
            key = _canonical_operator_key(operators)
            terms[key] = terms.get(key,0.)+coefficient
    n = len(field.frequencies_cm)
    for i in range(n):
        add((("q",i),)*4,field.quartic_cm[i,i,i,i]/24)
        add((("p",i),("q",i),("q",i),("p",i)),field.metric_second_cm[i,i,i,i]/4)
        for j in range(i):
            add((("q",i),("q",i),("q",j),("q",j)),field.quartic_cm[i,i,j,j]/4)
            for a,b,c,d in set(permutations((i,i,j,j))):
                add((("p",a),("q",c),("q",d),("p",b)),field.metric_second_cm[a,b,c,d]/4)
    return tuple((key,value) for key,value in terms.items() if value)


def pack_anchor_operators(cubic, quartic, frequencies):
    """Tabulate ordered local ladder factors for occupations 0,1,2.

    Different cubic monomials reaching the same virtual state are summed
    before squaring. No virtual-state pruning or field threshold is used.
    """
    from .curvilinear_vpt2 import _local_operator_transitions
    grouped = {}
    for key, coefficient in cubic:
        choices = []
        for mode, operators in key:
            by_delta = {}
            for occupation in range(3):
                for target, value in _local_operator_transitions(operators,occupation):
                    delta = target-occupation
                    by_delta.setdefault(delta,np.zeros(3,complex))[occupation] = value
            choices.append(tuple(by_delta.items()))
        for selected in product(*choices):
            changed = tuple(sorted((mode,delta) for (mode,_),(delta,_) in zip(key,selected) if delta))
            if not changed:
                continue
            modes = np.full(3,-1,int)
            values = np.ones((3,3),complex)
            for j,((mode,_),(_,table)) in enumerate(zip(key,selected)):
                modes[j] = mode
                values[j] = table
            values[0] *= coefficient
            grouped.setdefault(changed,[]).append((modes,values))
    starts = [0]
    gaps = []
    actions = []
    for changed, entries in sorted(grouped.items()):
        gaps.append(-sum(frequencies[i]*delta for i,delta in changed))
        actions.extend(entries)
        starts.append(len(actions))
    q_modes = []
    q_values = []
    for key, coefficient in quartic:
        if any(len(ops)%2 for _,ops in key):
            continue
        modes = np.full(2,-1,int)
        values = np.ones((2,3),complex)
        for j,(mode,operators) in enumerate(key):
            modes[j] = mode
            values[j] = [dict(_local_operator_transitions(operators,n)).get(n,0.) for n in range(3)]
        values[0] *= coefficient
        q_modes.append(modes)
        q_values.append(values)
    return (np.asarray(starts,np.int64),np.asarray(gaps,float),
        np.asarray([v[0] for v in actions],np.int64).reshape(-1,3),
        np.asarray([v[1] for v in actions],complex).reshape(-1,3,3),
        np.asarray(q_modes,np.int64).reshape(-1,2),
        np.asarray(q_values,complex).reshape(-1,2,3))


@njit(cache=True)
def _hybrid(u,d,alpha,beta):
    if not u or not d:
        return 0.
    half = abs(d)/2
    scale = max(half,u)
    a,b = u/scale,half/scale
    dc = copysign(u*(a/(hypot(a,b)+b)),d)
    w = .5*(1+tanh(alpha*(u*half-beta)))
    return (u*(w*u/d) if w else 0.)+(1-w)*dc


@njit(cache=True,parallel=True)
def evaluate_anchor_energies(states,frequencies,constant,starts,gaps,modes,values,q_modes,q_values,alpha,beta):
    result = np.empty(len(states))
    for row in prange(len(states)):
        state = states[row]
        energy = constant
        for i in range(len(state)):
            energy += (state[i]+.5)*frequencies[i]
        for term in range(len(q_modes)):
            amplitude = 1.+0j
            for j in range(2):
                mode = q_modes[term,j]
                if mode >= 0:
                    amplitude *= q_values[term,j,state[mode]]
                else:
                    # The coefficient lives in the first (always occupied) factor.
                    break
            energy += amplitude.real
        for p in range(len(gaps)):
            coupling = 0.+0j
            for term in range(starts[p],starts[p+1]):
                amplitude = 1.+0j
                for j in range(3):
                    mode = modes[term,j]
                    if mode < 0:
                        break
                    amplitude *= values[term,j,state[mode]]
                coupling += amplitude
            energy += _hybrid(abs(coupling),gaps[p],alpha,beta)
        result[row] = energy
    return result
