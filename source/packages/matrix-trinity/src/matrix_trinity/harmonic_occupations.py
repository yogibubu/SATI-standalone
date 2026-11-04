"""Finite downward-closed oscillator spaces with explicit weighted budgets."""
import numpy as np


def weighted_harmonic_occupations(weights, budget, *, maximum_total_quanta=None,
        maximum_states=None):
    """Enumerate every occupation with sum(weights*n)<=budget, retaining all modes."""
    w=np.asarray(weights)
    if (w.ndim!=1 or not len(w) or np.any(w!=np.rint(w)) or np.any(w<=0)
            or isinstance(budget,bool) or not isinstance(budget,(int,np.integer)) or budget<0):
        raise ValueError('Positive integer mode weights and nonnegative integer budget required')
    for value,label,minimum in ((maximum_total_quanta,'total-quanta ceiling',0),
            (maximum_states,'state budget',1)):
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<minimum):
            raise ValueError('Valid nonnegative '+label+' required')
    result=[]
    def visit(prefix,remaining,axis,total):
        if axis==len(w):
            if maximum_states is not None and len(result)>=maximum_states:
                raise MemoryError('Weighted oscillator space exceeds its declared state budget')
            result.append(tuple(prefix));return
        cap=int(remaining//w[axis])
        if maximum_total_quanta is not None:cap=min(cap,maximum_total_quanta-total)
        for n in range(cap+1):visit(prefix+[n],remaining-n*int(w[axis]),axis+1,total+n)
    visit([],int(budget),0,0)
    return tuple(sorted(result,key=lambda row:(sum(row),row)))


def oscillator_space_with_closure(occupations, mode_count):
    """Validate a downward-closed space and retain every one-quantum exterior row.

    Exterior rows are exact q*q and p*p intermediates, not variational states.
    """
    raw=np.asarray(occupations)
    if (raw.ndim!=2 or raw.shape[1]!=mode_count or not len(raw)
            or np.any(raw!=np.rint(raw)) or np.any(raw<0)):
        raise ValueError('Nonnegative integer occupations for every mode required')
    states=tuple(tuple(int(v) for v in row) for row in raw);selected=set(states)
    if len(selected)!=len(states) or states[0]!=(0,)*mode_count:
        raise ValueError('Unique occupations with the ground state first required')
    extended=list(states);seen=set(states)
    for state in states:
        for mode,n in enumerate(state):
            if n:
                lower=list(state);lower[mode]-=1
                if tuple(lower) not in selected:raise ValueError('Oscillator space must be downward closed')
            upper=list(state);upper[mode]+=1;upper=tuple(upper)
            if upper not in seen:seen.add(upper);extended.append(upper)
    return states,tuple(extended)
