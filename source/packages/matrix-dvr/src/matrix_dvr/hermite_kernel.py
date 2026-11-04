"""Analytic Gaussian-kernel residual fields fitted to energies and derivatives.

A caller supplies the physical baseline. This module learns only its residual
from existing observations; it has no electronic-evaluation callback. Symmetry
under simultaneous inversion is optional. Leave-group-out predictions remove
all derivative observations for each withheld geometry group together.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import eval_hermitenorm


def derivative_kernel(x, y, orders_x, orders_y, lengthscales, *, inversion_even=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ell = np.asarray(lengthscales, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or ell.shape != (x.shape[1],) or np.any(ell <= 0):
        raise ValueError('invalid kernel coordinates or positive lengthscales')
    a, b = np.asarray(orders_x, int), np.asarray(orders_y, int)
    if a.shape != ell.shape or b.shape != ell.shape or np.any(a < 0) or np.any(b < 0):
        raise ValueError('nonnegative derivative multi-indices are required')
    def block(sign):
        delta = (x[:, None, :] - sign*y[None, :, :])/ell
        k = np.exp(-.5*np.sum(delta*delta, axis=-1))
        for axis, degree in enumerate(a+b):
            k *= eval_hermitenorm(int(degree), delta[..., axis])/ell[axis]**degree
        return k * (-1.)**int(np.sum(a)) * sign**int(np.sum(b))
    k = block(1)
    return .5*(k+block(-1)) if inversion_even else k


@dataclass(frozen=True)
class HermiteKernelResidual:
    coordinates: np.ndarray
    orders: tuple[tuple[int, ...], ...]
    coefficients: np.ndarray
    lengthscales: np.ndarray
    inversion_even: bool = False
    anchor_point: np.ndarray | None = None
    anchor_orders: tuple[tuple[int, ...], ...] = ()
    anchor_coefficients: np.ndarray | None = None

    def evaluate(self, points, derivative=None):
        q = np.asarray(points, float)
        if q.ndim != 2 or q.shape[1] != self.coordinates.shape[1]:
            raise ValueError('query coordinates must have shape (points, dimension)')
        order = tuple((0,)*q.shape[1] if derivative is None else derivative)
        result = np.zeros(len(q))
        n = len(self.coordinates)
        for i, source_order in enumerate(self.orders):
            k = derivative_kernel(q, self.coordinates, order, source_order,
                                  self.lengthscales, inversion_even=self.inversion_even)
            result += k @ self.coefficients[i*n:(i+1)*n]
        if self.anchor_point is not None:
            for i, anchor_order in enumerate(self.anchor_orders):
                k = derivative_kernel(q, self.anchor_point[None, :], order, anchor_order,
                                      self.lengthscales, inversion_even=self.inversion_even)
                result += k[:, 0]*self.anchor_coefficients[i]
        return result

    def to_dict(self):
        return dict(schema='matrix.dvr.hermite_kernel_residual.v1',
                    coordinates=self.coordinates.tolist(), orders=[list(o) for o in self.orders],
                    coefficients=self.coefficients.tolist(), lengthscales=self.lengthscales.tolist(),
                    inversion_even=self.inversion_even,
                    anchor_point=None if self.anchor_point is None else self.anchor_point.tolist(),
                    anchor_orders=[list(o) for o in self.anchor_orders],
                    anchor_coefficients=None if self.anchor_coefficients is None else self.anchor_coefficients.tolist())

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.hermite_kernel_residual.v1':
            raise ValueError('unsupported Hermite residual schema')
        return cls(np.asarray(data['coordinates'], float), tuple(tuple(o) for o in data['orders']),
                   np.asarray(data['coefficients'], float), np.asarray(data['lengthscales'], float),
                   bool(data['inversion_even']),
                   None if data.get('anchor_point') is None else np.asarray(data['anchor_point'], float),
                   tuple(tuple(o) for o in data.get('anchor_orders', ())),
                   None if data.get('anchor_coefficients') is None else np.asarray(data['anchor_coefficients'], float))


def fit_hermite_kernel_residual(coordinates, orders, residual_observations, *,
                                lengthscales, ridge, derivative_weights=None,
                                groups=None, inversion_even=False,
                                zero_jet_orders=(), anchor_point=None):
    """Fit one kernel and return exact leave-group-out errors at fixed settings.

    Observations are (derivative orders, geometries); coefficients use the
    same order-major flattening. Ridge is positive in the weighted kernel.
    Hyperparameters must be selected from independent prediction diagnostics,
    never by comparison with downstream experimental observables.
    ``zero_jet_orders`` conditions the kernel on exactly zero derivatives at
    ``anchor_point`` (the origin by default), preserving a known baseline jet.
    """
    q = np.asarray(coordinates, float)
    orders = tuple(tuple(int(v) for v in o) for o in orders)
    y = np.asarray(residual_observations, float)
    n, d = q.shape
    if y.shape != (len(orders), n) or ridge <= 0 or not np.all(np.isfinite(y)):
        raise ValueError('invalid Hermite observations or positive ridge')
    weights = np.ones(len(orders)) if derivative_weights is None else np.asarray(derivative_weights, float)
    if weights.shape != (len(orders),) or np.any(weights <= 0):
        raise ValueError('positive derivative weights are required')
    k = np.block([[derivative_kernel(q, q, a, b, lengthscales, inversion_even=inversion_even)
                   for b in orders] for a in orders])
    zero_jet_orders = tuple(tuple(int(v) for v in o) for o in zero_jet_orders)
    anchor, anchor_cross, anchor_inverse = None, None, None
    if zero_jet_orders:
        anchor = np.zeros(d) if anchor_point is None else np.asarray(anchor_point, float)
        aq = anchor[None, :]
        anchor_cross = np.block([[derivative_kernel(q, aq, a, b, lengthscales,
            inversion_even=inversion_even) for b in zero_jet_orders] for a in orders])
        anchor_kernel = np.block([[derivative_kernel(aq, aq, a, b, lengthscales,
            inversion_even=inversion_even) for b in zero_jet_orders] for a in zero_jet_orders])
        anchor_inverse = np.linalg.pinv(anchor_kernel, rcond=1e-12)
        k -= anchor_cross@anchor_inverse@anchor_cross.T
    w = np.repeat(weights, n)
    kw = k*w[:, None]*w[None, :]
    eig, vec = np.linalg.eigh(.5*(kw+kw.T))
    tolerance = 1e-10*max(float(np.max(abs(eig))), 1.)
    if np.min(eig) < -tolerance:
        raise ValueError('derivative kernel is not positive semidefinite')
    inv = (vec/(np.maximum(eig, 0.)+ridge))@vec.T
    alpha = inv@(w*y.ravel())
    coefficient = w*alpha
    # ravel() assignments below must address the original array, even when
    # observations arrived as a Fortran-contiguous advanced-indexing result.
    loo = np.full(y.shape, np.nan, dtype=float, order="C")
    for group in groups or []:
        indices = np.concatenate([np.asarray(group, int)+i*n for i in range(len(orders))])
        if len(indices) != len(set(indices.tolist())) or np.any(indices < 0) or np.any(indices >= len(alpha)):
            raise ValueError('invalid leave-out group')
        errors = np.linalg.solve(inv[np.ix_(indices, indices)], alpha[indices])/w[indices]
        loo.ravel()[indices] = errors
    anchor_coefficients = None if anchor is None else -anchor_inverse@anchor_cross.T@coefficient
    model = HermiteKernelResidual(q.copy(), orders, coefficient, np.asarray(lengthscales, float),
                                  inversion_even, anchor, zero_jet_orders, anchor_coefficients)
    predicted = (k@coefficient).reshape(y.shape)
    return model, dict(training_errors=(predicted-y), leave_group_out_errors=loo,
                       kernel_minimum_eigenvalue=float(np.min(eig)), ridge=float(ridge))
