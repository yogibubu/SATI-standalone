"""Compose a quadratic transverse field with an active-coordinate map.

Only active arguments change. The transverse basis and its positive Hessian
are retained; no second quadratic truncation in another bath is performed.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReparameterizedBath:
    base: object
    argument_field: object

    def evaluate(self, points, *, second=False, allow_extrapolation=False):
        q = np.asarray(points, float)
        if q.ndim != 2 or not np.all(np.isfinite(q)): raise ValueError('Finite coordinate matrix required')
        d = q.shape[1]; eye = np.eye(d, dtype=int)
        mapped = self.argument_field.evaluate(q)
        J = np.stack([self.argument_field.evaluate(q, e) for e in eye], axis=-1)
        K = np.stack([np.stack([self.argument_field.evaluate(q, e+f) for f in eye],axis=-1) for e in eye],axis=-2) if second else None
        result = self.base.evaluate(mapped, second=second, allow_extrapolation=allow_extrapolation)
        for name in ('force', 'hessian', 'response', 'energy_lowering'):
            first_key = name+'_first'
            if first_key not in result: continue
            first = result[first_key]
            if second and name+'_second' in result:
                result[name+'_second'] = (np.einsum('p...ij,pia,pjb->p...ab', result[name+'_second'], J, J, optimize=True)
                    +np.einsum('p...i,piab->p...ab', first, K, optimize=True))
            result[first_key] = np.einsum('p...i,pia->p...a', first, J, optimize=True)
        return result
