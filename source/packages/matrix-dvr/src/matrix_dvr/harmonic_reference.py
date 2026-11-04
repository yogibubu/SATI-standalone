"""Quadratic reference replacement with a fixed modal coordinate definition.

The old dimensionless coordinates q = gamma_old Q are kept, including their
declared nonlinear continuation. All higher derivatives in that chart remain
unchanged. Frequencies alone cannot determine a Duschinsky rotation.
"""
from dataclasses import dataclass
import numpy as np


def fixed_modal_hessian_increment(source_cm, target_cm):
    """Return Delta H in cm^-1 per old dimensionless modal coordinate squared."""
    old = np.asarray(source_cm, float)
    new = np.asarray(target_cm, float)
    if (old.ndim != 1 or new.shape != old.shape or np.any(old <= 0)
            or np.any(new <= 0) or not np.all(np.isfinite([old, new]))):
        raise ValueError('Matching finite positive source and target frequencies required')
    return np.diag((new*new-old*old)/old)


@dataclass(frozen=True)
class QuadraticShiftedBath:
    """Add a constant bath Hessian; re-minimize and differentiate consistently.

    Units are those of the underlying analytic bath. No Cholesky refit is
    performed: the original H(q) receives the exact constant increment.
    """
    source: object
    hessian_increment: np.ndarray

    def __post_init__(self):
        delta = np.array(self.hessian_increment, float, copy=True)
        if (delta.ndim != 2 or delta.shape[0] != delta.shape[1]
                or not np.all(np.isfinite(delta))
                or not np.allclose(delta, delta.T, atol=1e-12, rtol=0)):
            raise ValueError('Finite symmetric bath Hessian increment required')
        delta.flags.writeable = False
        object.__setattr__(self, 'hessian_increment', delta)

    def evaluate(self, points, *, second=False):
        result = dict(self.source.evaluate(points, second=second))
        H = result['hessian'] + self.hessian_increment
        # A nonpositive corrected bath is a model failure, never clipped.
        np.linalg.cholesky(H)
        g, dg, dH = result['force'], result['force_first'], result['hessian_first']
        b = -np.linalg.solve(H, g[..., None])[..., 0]
        db = -np.linalg.solve(H, dg + np.einsum('...ijt,...j->...it', dH, b))
        result.update(hessian=H, response=b, response_first=db,
            energy_lowering=.5*np.einsum('...i,...i->...', g, b),
            energy_lowering_first=np.einsum('...it,...i->...t', dg, b)
                + .5*np.einsum('...i,...ijt,...j->...t', b, dH, b))
        if second:
            rhs = result['force_second'] + np.einsum(
                '...ijst,...j->...ist', result['hessian_second'], b)
            mixed = np.einsum('...ijs,...jt->...ist', dH, db)
            rhs = rhs + mixed + mixed.swapaxes(-2, -1)
            shape = rhs.shape
            result['response_second'] = -np.linalg.solve(
                H, rhs.reshape(shape[:-2]+(-1,))).reshape(shape)
        return result
