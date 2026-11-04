"""Local SH calibration multipliers; no coordinate or Cartesian transforms."""
import math


def primitive_parameter_key(primitive, atom_symbols, covalent_bonds):
    """Permutation-consistent chemical type, never a molecule identifier."""
    ids=primitive.atoms
    symbols=tuple(atom_symbols[i-1] for i in ids)
    key=primitive.family
    if primitive.function=='R':
        key+=':'+','.join(sorted(symbols))
    elif primitive.function in ('A','L'):
        key+=':'+symbols[1]+':'+','.join(sorted((symbols[0],symbols[2])))
    elif primitive.function=='D':
        key+=':'+','.join(sorted(symbols[1:3]))
    elif primitive.function=='U':
        weak=sum(tuple(sorted((ids[0],i))) not in covalent_bonds for i in ids[1:])
        key+=':'+symbols[0]+':'+','.join(sorted(symbols[1:]))+':weak='+str(weak)
    return key


def checked_parameter_multipliers(parameters):
    result={str(key):float(value) for key,value in parameters.items()}
    if any(not math.isfinite(value) or value<=0 for value in result.values()):
        raise ValueError('SH parameter multipliers must be finite and positive')
    return result
