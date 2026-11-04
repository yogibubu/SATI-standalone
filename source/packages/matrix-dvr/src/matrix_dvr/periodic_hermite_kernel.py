"""Smooth periodic Hermite interpolation with exact reference constraints."""
from dataclasses import dataclass
from math import comb
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def periodic_hermite_covariance(left, left_orders, right, right_orders, lengths, *, reflection_even=True):
    """Derivative covariance of a periodic Gaussian, optionally reflection even."""
    if not isinstance(reflection_even,(bool,np.bool_)):
        raise ValueError('Explicit Boolean reflection policy required')
    x, y = np.asarray(left, float), np.asarray(right, float)
    a, b = np.asarray(left_orders, int), np.asarray(right_orders, int)
    lengths = np.asarray(lengths, float)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]
            or a.shape != x.shape or b.shape != y.shape or lengths.shape != (x.shape[1],)
            or np.any(a < 0) or np.any(b < 0) or np.any(lengths <= 0)
            or not all(np.all(np.isfinite(v)) for v in (x, y, lengths))):
        raise ValueError('Compatible finite kernel points, derivative orders and lengths required')
    terms = []
    for sign in ((-1, 1) if reflection_even else (-1,)):
        value = np.ones((len(x), len(y)))
        for axis, length in enumerate(lengths):
            theta = x[:, None, axis]+sign*y[None, :, axis]
            order = a[:, None, axis]+b[None, :, axis]
            maximum = int(order.max(initial=0))
            bell = [np.ones_like(theta)]
            log_derivatives = [None]+[np.cos(theta+k*np.pi/2)/length**2 for k in range(1, maximum+1)]
            for n in range(1, maximum+1):
                bell.append(sum((comb(n-1, k-1)*log_derivatives[k]*bell[n-k]
                                 for k in range(1, n+1)), start=np.zeros_like(theta)))
            derivative = np.take_along_axis(np.array(bell), order[None], axis=0)[0]
            value *= np.exp((np.cos(theta)-1)/length**2)*derivative
        if sign == -1:
            value *= (-1.)**b.sum(axis=1)[None]
        terms.append(value)
    return (terms[0]+terms[1])/2 if reflection_even else terms[0]


@dataclass(frozen=True)
class PeriodicHermiteKernelField:
    centers: np.ndarray
    orders: np.ndarray
    coefficients: np.ndarray
    lengthscales: np.ndarray
    reflection_even: bool = True

    def __post_init__(self):
        if not isinstance(self.reflection_even,(bool,np.bool_)):
            raise ValueError('Explicit Boolean reflection policy required')
        for name in ('centers', 'orders', 'coefficients', 'lengthscales'):
            raw = getattr(self, name)
            value = np.array(raw, dtype=int if name == 'orders' else float, copy=True)
            if not np.all(np.isfinite(value)) or (name == 'orders' and not np.array_equal(value, raw)):
                raise ValueError('Finite parameters and integer kernel derivative orders required')
            value.flags.writeable = False; object.__setattr__(self, name, value)
        if (self.centers.ndim != 2 or self.orders.shape != self.centers.shape
                or self.coefficients.ndim != 2 or len(self.coefficients) != len(self.centers)
                or self.lengthscales.shape != (self.centers.shape[1],)
                or np.any(self.orders < 0) or np.any(self.lengthscales <= 0)):
            raise ValueError('Compatible periodic Hermite field parameters required')

    def evaluate(self, points, derivative=None):
        q = np.asarray(points, float)
        raw = np.zeros(len(self.lengthscales), int) if derivative is None else np.asarray(derivative)
        order = np.asarray(raw, int)
        if (q.ndim != 2 or q.shape[1] != len(self.lengthscales)
                or order.shape != (q.shape[1],) or not np.array_equal(order, raw) or np.any(order < 0)):
            raise ValueError('Coordinate matrix and nonnegative integer derivative required')
        result = np.empty((len(q), self.coefficients.shape[1]))
        for start in range(0, len(q), 512):
            stop = min(start+512, len(q)); left = np.broadcast_to(order, q[start:stop].shape)
            result[start:stop] = periodic_hermite_covariance(q[start:stop], left,
                self.centers, self.orders, self.lengthscales,reflection_even=self.reflection_even)@self.coefficients
        return result

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_hermite_kernel.v1',
            centers=self.centers.tolist(), orders=self.orders.tolist(),
            coefficients=self.coefficients.tolist(), lengthscales=self.lengthscales.tolist(),
            global_reflection_even=bool(self.reflection_even), periods_radian=[2*np.pi]*len(self.lengthscales))

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.periodic_hermite_kernel.v1':
            raise ValueError('Unsupported periodic Hermite kernel field')
        return cls(*(np.asarray(data[k]) for k in ('centers', 'orders', 'coefficients', 'lengthscales')),reflection_even=data.get('global_reflection_even',True))


@dataclass(frozen=True)
class KernelCorrectedPeriodicSurface:
    base: object
    correction: PeriodicHermiteKernelField

    @property
    def baseline(self):
        return self.base.baseline

    def evaluate(self, points, derivative=None, *, allow_extrapolation=False):
        return (self.base.evaluate(points, derivative, allow_extrapolation=allow_extrapolation)
                +self.correction.evaluate(points, derivative)[:, 0])

    def to_dict(self):
        return dict(schema='matrix.dvr.kernel_corrected_periodic_surface.v1',
                    base=self.base.to_dict(), correction=self.correction.to_dict())

    @classmethod
    def from_dict(cls, data):
        from .periodic_hermite import periodic_surface_from_dict
        if data.get('schema') != 'matrix.dvr.kernel_corrected_periodic_surface.v1':
            raise ValueError('Unsupported kernel-corrected periodic surface')
        return cls(periodic_surface_from_dict(data['base']),
                   PeriodicHermiteKernelField.from_dict(data['correction']))


def fit_periodic_hermite_kernel(points, orders, values, *, lengthscales, weights,
                                constraint_points, constraint_orders, relative_noise=1e-8,reflection_even=True):
    """Fit derivative observations while retaining exact homogeneous jet constraints.

    Input values are residuals after a separate physical reference surface.
    Constraints are imposed algebraically before smoothing the noisy data.
    """
    x, a, y = np.asarray(points, float), np.asarray(orders, int), np.asarray(values, float)
    c, ca = np.asarray(constraint_points, float), np.asarray(constraint_orders, int)
    w = np.asarray(weights, float); lengths = np.asarray(lengthscales, float)
    if y.ndim == 1: y = y[:, None]
    if (y.shape[0] != len(x) or w.shape != (len(x),) or np.any(w <= 0)
            or not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)) or not 0 < relative_noise < 1):
        raise ValueError('Finite derivative observations and positive smoothing required')
    # Natural derivative scales keep the reference fourth-jet block conditioned.
    from math import factorial
    cw = np.array([np.prod(lengths**o)/np.sqrt(np.prod([factorial(int(k)) for k in o])) for o in ca])
    Kcc = periodic_hermite_covariance(c, ca, c, ca, lengths,reflection_even=reflection_even)*cw[:, None]*cw[None]
    cc = cho_factor((Kcc+Kcc.T)/2, lower=True)
    Kxc = periodic_hermite_covariance(x, a, c, ca, lengths,reflection_even=reflection_even)*w[:, None]*cw[None]
    transfer = cho_solve(cc, Kxc.T)
    K = periodic_hermite_covariance(x, a, x, a, lengths,reflection_even=reflection_even)*w[:, None]*w[None]-Kxc@transfer
    K = (K+K.T)/2
    noise = relative_noise*max(1., float(np.max(np.diag(K))))
    smooth = cho_factor(K+noise*np.eye(len(K)), lower=True)
    alpha = cho_solve(smooth, y*w[:, None])
    beta = -transfer@alpha
    field = PeriodicHermiteKernelField(np.vstack([x, c]), np.vstack([a, ca]),
        np.vstack([alpha*w[:, None], beta*cw[:, None]]), lengths,reflection_even=reflection_even)
    constraint_residual = periodic_hermite_covariance(c, ca, field.centers, field.orders, lengths,reflection_even=reflection_even)@field.coefficients
    audit = dict(observations=len(x), reference_constraints=len(c),
        relative_noise=relative_noise, absolute_kernel_noise=noise,
        maximum_reference_constraint_residual=float(abs(constraint_residual).max(initial=0)),
        weighted_training_RMS=float(np.sqrt(np.mean((K@alpha-y*w[:, None])**2))))
    audit['global_reflection_even']=bool(reflection_even)
    return field, audit
