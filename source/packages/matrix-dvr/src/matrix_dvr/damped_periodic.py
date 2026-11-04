"""Periodic Hermite residuals with smooth tails and exact reference Hessian."""
from dataclasses import dataclass
from math import comb
from itertools import product, combinations_with_replacement
import numpy as np
from .periodic_hermite import PeriodicHermiteSurface, angular_features


def periodic_envelope(points, lengths, power, order):
    q = np.asarray(points, float); lengths = np.asarray(lengths, float)**2
    order = np.asarray(order, int); axes = np.repeat(np.arange(len(order)), order)
    if len(axes) > 2: raise ValueError('Derivatives through second order required')
    t = np.sum((1-np.cos(q))/lengths, axis=1); w = np.exp(-t**power)
    if not len(axes): return w
    i = axes[0]; ti = np.sin(q[:, i])/lengths[i]; li = -power*t**(power-1)*ti
    if len(axes) == 1: return w*li
    j = axes[1]; tj = np.sin(q[:, j])/lengths[j]; lj = -power*t**(power-1)*tj
    lij = -power*t**(power-1)*(np.cos(q[:, i])/lengths[i] if i == j else 0.)
    if power > 1: lij -= power*(power-1)*t**(power-2)*ti*tj
    return w*(li*lj+lij)


def damped_angular_features(points, powers, scales, lengths, power, derivative=None):
    q = np.asarray(points, float); d = len(scales)
    if q.ndim != 2 or q.shape[1] != d or not np.all(np.isfinite(q)): raise ValueError('Coordinate matrix required')
    order = (0,)*d if derivative is None else tuple(derivative)
    if len(order) != d or any(k < 0 or int(k) != k for k in order) or sum(order) > 2:
        raise ValueError('Integer derivative orders through second order required')
    result = np.zeros((len(q), len(powers)))
    for left in product(*[range(k+1) for k in order]):
        right = tuple(k-j for k, j in zip(order, left)); weight = np.prod([comb(k, j) for k, j in zip(order, left)])
        result += weight*periodic_envelope(q, lengths, power, left)[:, None]*angular_features(q, powers, scales, right)
    return result


@dataclass(frozen=True)
class DampedPeriodicHermiteSurface:
    baseline: PeriodicHermiteSurface
    powers: np.ndarray
    coefficients: np.ndarray
    feature_scales: np.ndarray
    tail_lengths_radian: np.ndarray
    envelope_power: int = 1

    def __post_init__(self):
        for name in ('powers', 'coefficients', 'feature_scales', 'tail_lengths_radian'):
            value = np.array(getattr(self, name), dtype=int if name == 'powers' else float, copy=True)
            if not np.all(np.isfinite(value)): raise ValueError('Finite surface required')
            value.flags.writeable = False; object.__setattr__(self, name, value)
        d = len(self.feature_scales)
        if (self.powers.ndim != 2 or self.powers.shape[1] != d or np.any(self.powers < 0)
                or np.any(self.powers.sum(axis=1) < 4) or np.any(self.powers.sum(axis=1) % 2)
                or self.coefficients.shape != (len(self.powers),) or self.tail_lengths_radian.shape != (d,)
                or np.any(self.tail_lengths_radian <= 0) or np.any(self.feature_scales <= 0)
                or self.envelope_power not in (1, 2, 3, 4)):
            raise ValueError('Compatible even anchored features and positive angular tail lengths required')

    def _envelope(self, q, order):
        return periodic_envelope(q, self.tail_lengths_radian, self.envelope_power, order)

    def features(self, points, derivative=None):
        return damped_angular_features(points, self.powers, self.feature_scales, self.tail_lengths_radian, self.envelope_power, derivative)

    def evaluate(self, points, derivative=None, *, allow_extrapolation=False):
        return self.baseline.evaluate(points, derivative, allow_extrapolation=allow_extrapolation)+self.features(points, derivative)@self.coefficients

    def to_dict(self):
        return dict(schema='matrix.dvr.damped_periodic_hermite_surface.v1', baseline=self.baseline.to_dict(),
                    powers=self.powers.tolist(), coefficients=self.coefficients.tolist(), feature_scales=self.feature_scales.tolist(),
                    tail_lengths_radian=self.tail_lengths_radian.tolist(), envelope_power=self.envelope_power,
                    reference_E_G_H_exact=True, periods_radian=[2*np.pi]*len(self.feature_scales),
                    qualification='Tail regularity is not electronic validation outside the source')

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.damped_periodic_hermite_surface.v1': raise ValueError('Unsupported model')
        return cls(PeriodicHermiteSurface.from_dict(data['baseline']), np.array(data['powers']), np.array(data['coefficients']),
                   np.array(data['feature_scales']), np.array(data['tail_lengths_radian']), int(data['envelope_power']))


def fit_damped_periodic_surface(coordinates, observations, reference_hessian, *, powers, feature_scales,
                                tail_lengths_radian, orders, derivative_weights, relative_ridge=1e-10,
                                envelope_power=1, reference_quartic=None):
    q = np.asarray(coordinates, float); y = np.asarray(observations, float); H = np.asarray(reference_hessian, float)
    d = q.shape[1]; eigen, U = np.linalg.eigh(H); Hplus = (U*np.maximum(eigen, 0.))@U.T
    baseline = PeriodicHermiteSurface(H, 4*np.eye(d, dtype=int), np.diag(Hplus-H)/8., np.ones(d), np.max(abs(q), axis=0))
    model = DampedPeriodicHermiteSurface(baseline, np.array(powers), np.zeros(len(powers)), np.array(feature_scales), np.array(tail_lengths_radian), envelope_power)
    orders = tuple(tuple(o) for o in orders); weights = np.asarray(derivative_weights, float)
    if y.shape != (len(orders), len(q)) or weights.shape != (len(orders),) or np.any(weights <= 0) or not 0 < relative_ridge < 1:
        raise ValueError('Compatible finite Hermite observations and positive weights required')
    A = np.concatenate([model.features(q, o)*w for o, w in zip(orders, weights)])
    rhs = np.concatenate([(y[k]-baseline.evaluate(q, o, allow_extrapolation=True))*weights[k] for k, o in enumerate(orders)])
    offset = np.zeros(len(powers)); null = np.eye(len(powers)); constraint_rank = 0
    constraints = None; targets = None
    if reference_quartic is not None:
        quartic = np.asarray(reference_quartic, float)
        if (quartic.shape != (d,)*4 or not np.all(np.isfinite(quartic))
                or any(not np.allclose(quartic, quartic.swapaxes(i, i+1), rtol=1e-10, atol=1e-8)
                       for i in range(3))):
            raise ValueError('Finite permutation-symmetric reference quartic required')
        rows, values = [], []
        for axes in combinations_with_replacement(range(d), 4):
            order = np.bincount(axes, minlength=d)
            # Features start at degree four. Derivatives of their envelope
            # cannot contribute to this reference jet, for any envelope power.
            rows.append(angular_features(np.zeros((1, d)), powers, feature_scales, order)[0])
            values.append(quartic[axes]-baseline.evaluate(
                np.zeros((1, d)), order, allow_extrapolation=True)[0])
        constraints, targets = np.array(rows), np.array(values)
        cu, cs, cv = np.linalg.svd(constraints, full_matrices=True)
        cutoff = max(constraints.shape)*np.finfo(float).eps*(cs[0] if len(cs) else 1.)
        constraint_rank = int(np.sum(cs > cutoff))
        offset = cv[:constraint_rank].T@((cu[:, :constraint_rank].T@targets)/cs[:constraint_rank])
        if np.linalg.norm(constraints@offset-targets) > 1e-9*max(1., np.linalg.norm(targets)):
            raise ValueError('Declared features cannot represent the complete reference quartic')
        null = cv[constraint_rank:].T
    reduced = A@null; shifted = rhs-A@offset
    if null.shape[1]:
        u, s, vt = np.linalg.svd(reduced, full_matrices=False); ridge = relative_ridge*s[0]**2
        coefficients = offset+null@(vt.T@((s/(s*s+ridge))*(u.T@shifted)))
        effective_rank = int(np.sum(s*s > ridge))
    else:
        coefficients = offset; effective_rank = 0
    fitted = DampedPeriodicHermiteSurface(baseline, np.array(powers), coefficients, np.array(feature_scales), np.array(tail_lengths_radian), envelope_power)
    audit = dict(feature_count=len(powers), relative_ridge=relative_ridge, effective_rank=effective_rank,
                 weighted_training_RMS=float(np.sqrt(np.mean((A@coefficients-rhs)**2))), reference_E_G_H_exact=True)
    if constraints is not None:
        audit.update(reference_quartic_exact=True, reference_constraint_rank=constraint_rank,
                     free_feature_count=null.shape[1],
                     maximum_reference_quartic_error=float(abs(constraints@coefficients-targets).max()))
    return fitted, audit
