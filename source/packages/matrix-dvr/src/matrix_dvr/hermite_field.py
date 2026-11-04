"""Vector Gaussian Hermite residuals with exact parity and reference jets.

The caller owns the physical baseline and coordinate domain. This finite-source
owner has no electronic callback. All outputs share one factorization; their
scales do not change the fit because the same linear operator acts on each.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .hermite_kernel import derivative_kernel


def _kernel(x, y, a, b, ell, parity, interaction_order=0):
    if interaction_order:
        result = np.zeros((len(x), len(y)))
        d = np.asarray(x).shape[1]
        for degree in range(1, interaction_order+1):
            for axes in combinations(range(d), degree):
                if any(a[i]+b[i] for i in range(d) if i not in axes):
                    continue
                idx = list(axes)
                result += _kernel(np.asarray(x)[:, idx], np.asarray(y)[:, idx],
                                  tuple(a[i] for i in idx), tuple(b[i] for i in idx),
                                  np.asarray(ell)[idx], parity)
        return result
    direct = derivative_kernel(x, y, a, b, ell)
    if parity == 0:
        return direct
    mirror = derivative_kernel(x, -np.asarray(y), a, b, ell)
    return .5 * (direct + parity * (-1.)**sum(b) * mirror)


@dataclass(frozen=True)
class HermiteResidualField:
    coordinates: np.ndarray
    orders: tuple
    coefficients: np.ndarray
    lengthscales: np.ndarray
    parity: int
    anchor: np.ndarray
    anchor_orders: tuple
    anchor_coefficients: np.ndarray
    interaction_order: int = 0

    def evaluate(self, points, derivative=None):
        q = np.asarray(points, float)
        d = self.coordinates.shape[1]
        a = (0,)*d if derivative is None else tuple(derivative)
        if q.ndim != 2 or q.shape[1] != d or len(a) != d:
            raise ValueError('Matching matrix of query coordinates required')
        k = np.concatenate([_kernel(q, self.coordinates, a, b,
                            self.lengthscales, self.parity, self.interaction_order) for b in self.orders], axis=1)
        result = k @ self.coefficients
        if self.anchor_orders:
            ka = np.concatenate([_kernel(q, self.anchor[None], a, b,
                                  self.lengthscales, self.parity, self.interaction_order)
                                 for b in self.anchor_orders], axis=1)
            result += ka @ self.anchor_coefficients
        return result

    def to_dict(self):
        return dict(schema='matrix.dvr.hermite_residual_field.v1',
                    **{k: v.tolist() if isinstance(v, np.ndarray) else
                       [[int(i) for i in o] for o in v] if k in ('orders', 'anchor_orders') else v
                       for k, v in self.__dict__.items()})

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.hermite_residual_field.v1':
            raise ValueError('Unsupported vector Hermite field schema')
        return cls(*(np.asarray(data[k], float) if k not in
                     ('orders', 'parity', 'anchor_orders', 'interaction_order') else
                     int(data.get(k, 0)) if k in ('parity', 'interaction_order') else
                     tuple(tuple(o) for o in data[k])
                     for k in cls.__dataclass_fields__))


def fit_hermite_residual_field(coordinates, orders, observations, *, lengthscales,
                               relative_ridge, parity=0, zero_jet_orders=(),
                               anchor=None, derivative_weights=None, interaction_order=0):
    """Fit (order, point, output) residual data; preserve exact zero jets.

    ``parity`` is 0 (unrestricted), +1 (even), or -1 (odd) under simultaneous
    inversion. Ridge is relative to the largest eigenvalue after weighting.
    No held-out geometry is accepted here: selection stays with the caller.
    """
    x = np.asarray(coordinates, float)
    y = np.asarray(observations, float)
    ell = np.asarray(lengthscales, float)
    orders = tuple(tuple(o) for o in orders)
    if x.ndim != 2:
        raise ValueError('Coordinate matrix required')
    n, d = x.shape
    weights = np.ones(len(orders)) if derivative_weights is None else np.asarray(derivative_weights, float)
    if (y.ndim != 3 or y.shape[:2] != (len(orders), n) or y.shape[2] == 0
            or ell.shape != (d,) or np.any(ell <= 0)
            or weights.shape != (len(orders),) or np.any(weights <= 0)
            or parity not in (-1, 0, 1) or not 0 < relative_ridge < 1
            or interaction_order not in range(d+1)
            or any(len(o) != d or any(v < 0 or int(v) != v for v in o) for o in orders)
            or not all(np.all(np.isfinite(v)) for v in (x, y, ell, weights))):
        raise ValueError('Finite matching vector Hermite data required')
    zero = tuple(tuple(o) for o in zero_jet_orders)
    origin = np.zeros(d) if anchor is None else np.asarray(anchor, float)
    k = np.block([[_kernel(x, x, a, b, ell, parity, interaction_order) for b in orders] for a in orders])
    w = np.repeat(weights, n)
    unconditioned_trace = float(np.sum(np.diag(k)*w*w))
    cross = np.zeros((len(k), 0)); inverse = np.zeros((0, 0))
    if zero:
        cross = np.block([[_kernel(x, origin[None], a, b, ell, parity, interaction_order)
                           for b in zero] for a in orders])
        aa = np.block([[_kernel(origin[None], origin[None], a, b, ell, parity, interaction_order)
                        for b in zero] for a in zero])
        inverse = np.linalg.pinv(aa, rcond=1e-12)
        k -= cross @ inverse @ cross.T
    kw = k*w[:, None]*w[None, :]
    eig, vec = np.linalg.eigh(.5*(kw+kw.T))
    top = float(max(eig))
    if top <= 0 or min(eig) < -1e-11*max(top, unconditioned_trace):
        raise ValueError('Invalid conditioned Hermite kernel spectrum')
    ridge = relative_ridge*top
    rhs = w[:, None]*y.reshape(len(k), -1)
    alpha = vec @ ((vec.T @ rhs)/(np.maximum(eig, 0)+ridge)[:, None])
    coefficients = w[:, None]*alpha
    model = HermiteResidualField(x, orders, coefficients, ell, parity, origin,
                                zero, -inverse @ cross.T @ coefficients, interaction_order)
    predicted = (k @ coefficients).reshape(y.shape)
    return model, dict(training_RMS=float(np.sqrt(np.mean((predicted-y)**2))),
                       relative_ridge=float(relative_ridge), ridge=ridge,
                       effective_rank=int(np.sum(eig > ridge)),
                       kernel_minimum_eigenvalue=float(min(eig)))
