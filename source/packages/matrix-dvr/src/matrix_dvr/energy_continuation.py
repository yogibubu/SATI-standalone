"""Explicit lower continuation for an unqualified scalar-energy extension.

Above a declared join the original scalar and every derivative are unchanged.
Below it a C6 rational continuation approaches a declared finite lower limit.
The join and asymptote are model assumptions to be checked for sensitivity;
they are not fitted electronic observations or rigorous molecular bounds.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LowerEnergyContinuation:
    join_energy: float
    lower_limit: float
    power: int = 6

    def __post_init__(self):
        if (not np.isfinite(self.join_energy) or not np.isfinite(self.lower_limit)
                or self.lower_limit >= self.join_energy or isinstance(self.power,bool)
                or int(self.power)!=self.power or self.power<3):
            raise ValueError('Finite ordered energy limits and integer smoothness power required')

    def evaluate(self, energies):
        E=np.asarray(energies,float)
        if not np.all(np.isfinite(E)):raise ValueError('Finite scalar energies required')
        width=self.join_energy-self.lower_limit;t=np.maximum(self.join_energy-E,0.)
        r=t/width;den=1+r**self.power
        value=np.where(E>=self.join_energy,E,self.join_energy-t/den**(1/self.power))
        first=np.where(E>=self.join_energy,1.,den**(-1-1/self.power))
        second=np.where(E>=self.join_energy,0.,(self.power+1)*r**(self.power-1)/width/den**(2+1/self.power))
        return value,first,second

    def to_dict(self):
        return dict(schema='matrix.dvr.lower_energy_continuation.v1',join_energy=self.join_energy,
            lower_limit=self.lower_limit,power=self.power,identity_above_join=True,
            scope='Declared continuation of a scalar model, not a proven molecular bound')
