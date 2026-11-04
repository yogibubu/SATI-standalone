"""Stable finite Gaussian-polynomial Hermite fields with exact zero jets.

Explicit features avoid subtracting nearly equal Gaussian kernels when the
reference jet is conditioned to zero. This is a finite feature expansion,
not an assertion that a truncated expansion equals the infinite RBF kernel.
"""
from dataclasses import dataclass
from itertools import combinations, product
from math import comb
import numpy as np
from scipy.special import eval_hermitenorm, gammaln


@dataclass(frozen=True)
class GaussianPolynomialField:
    lengthscales: np.ndarray
    blocks: tuple
    coefficients: np.ndarray
    parity: int
    zero_jet_degree: int
    gaussian_envelope: bool = True

    def __post_init__(self):
        ell=np.array(self.lengthscales,float,copy=True)
        coefficients=np.array(self.coefficients,float,copy=True)
        blocks=[];count=0
        for axes,powers in self.blocks:
            axes=tuple(int(i) for i in axes);raw=np.asarray(powers)
            if (not axes or len(set(axes))!=len(axes) or min(axes)<0 or max(axes)>=len(ell)
                    or raw.ndim!=2 or raw.shape[1]!=len(axes) or np.any(raw<0)
                    or not np.all(np.isfinite(raw)) or np.any(raw!=np.rint(raw))):
                raise ValueError('Finite feature powers and distinct coordinate axes required')
            p=np.array(raw,int,copy=True)
            if np.any(p.sum(axis=1)<=self.zero_jet_degree) or (self.parity and np.any((-1.)**p.sum(axis=1)!=self.parity)):
                raise ValueError('Feature powers violate the reference jet or parity')
            p.flags.writeable=False;blocks.append((axes,p));count+=len(p)
        if (ell.ndim!=1 or np.any(ell<=0) or not np.all(np.isfinite(ell))
                or coefficients.ndim!=2 or coefficients.shape[0]!=count
                or not np.all(np.isfinite(coefficients)) or self.parity not in (-1,0,1)):
            raise ValueError('Compatible finite Gaussian-polynomial field required')
        ell.flags.writeable=False;coefficients.flags.writeable=False
        object.__setattr__(self,'lengthscales',ell);object.__setattr__(self,'coefficients',coefficients);object.__setattr__(self,'blocks',tuple(blocks))

    def features(self, points, derivative=None):
        q = np.asarray(points, float); d = len(self.lengthscales)
        order = np.zeros(d, int) if derivative is None else np.asarray(derivative, int)
        if q.ndim != 2 or q.shape[1] != d or order.shape != (d,) or np.any(order < 0):
            raise ValueError('Matching coordinates and nonnegative derivative order required')
        result = []
        for axes, powers in self.blocks:
            axes = tuple(axes); powers = np.asarray(powers, int)
            if any(order[i] for i in range(d) if i not in axes):
                result.append(np.zeros((len(q), len(powers)))); continue
            x = q[:, axes]/self.lengthscales[list(axes)]
            value = np.ones((len(q), len(powers)))
            if self.gaussian_envelope:
                value *= np.exp(-.5*np.sum(x*x, axis=1))[:, None]
            for k, axis in enumerate(axes):
                p = powers[:, k]; a = int(order[axis]); t = x[:, k, None]
                factor = np.zeros_like(value)
                for j in (range(a+1) if self.gaussian_envelope else [a]):
                    valid = p >= j
                    falling = np.exp(gammaln(p[valid]+1)-gammaln(p[valid]-j+1))
                    factor[:, valid] += (comb(a, j)*falling*t**np.maximum(p[valid]-j, 0)
                        * (-1.)**(a-j)*eval_hermitenorm(a-j, t))
                value *= factor/self.lengthscales[axis]**a
            value /= np.exp(.5*np.sum(gammaln(powers+1), axis=1))
            result.append(value)
        return np.concatenate(result, axis=1) if result else np.zeros((len(q), 0))

    def evaluate(self, points, derivative=None):
        return self.features(points, derivative) @ self.coefficients

    def to_dict(self):
        return dict(schema='matrix.dvr.gaussian_polynomial_field.v1',
                    lengthscales=self.lengthscales.tolist(),
                    blocks=[dict(axes=list(a), powers=np.asarray(p).tolist()) for a, p in self.blocks],
                    coefficients=self.coefficients.tolist(), parity=int(self.parity),
                    output_count=int(self.coefficients.shape[1]),
                    zero_jet_degree=int(self.zero_jet_degree),gaussian_envelope=bool(self.gaussian_envelope))

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.gaussian_polynomial_field.v1':
            raise ValueError('Unsupported Gaussian-polynomial field schema')
        return cls(np.array(data['lengthscales']),
                   tuple((tuple(b['axes']), np.array(b['powers'], int)) for b in data['blocks']),
                   np.array(data['coefficients']).reshape(-1,data['output_count']), int(data['parity']), int(data['zero_jet_degree']),bool(data.get('gaussian_envelope',True)))


def fit_gaussian_polynomial_field(coordinates, orders, observations, *, lengthscales,
                                  relative_ridge, parity=0, zero_jet_orders=(),
                                  derivative_weights=None, interaction_order=0,
                                  degree=12,gaussian_envelope=True,maximum_interaction_degree=None):
    q = np.asarray(coordinates, float); y = np.asarray(observations, float)
    if q.ndim != 2:raise ValueError('Coordinate matrix required')
    ell = np.asarray(lengthscales, float); d = q.shape[1]
    orders = tuple(tuple(int(i) for i in o) for o in orders)
    zero = tuple(tuple(int(i) for i in o) for o in zero_jet_orders)
    jets = max((sum(o) for o in zero), default=-1)
    expected = {o for o in product(range(jets+1), repeat=d) if sum(o) <= jets} if jets >= 0 else set()
    if set(zero) != expected:
        raise ValueError('Explicit features require a complete zero reference jet')
    if (q.ndim != 2 or y.ndim != 3 or y.shape[:2] != (len(orders), len(q))
            or ell.shape != (d,) or np.any(ell <= 0) or parity not in (-1, 0, 1)
            or not 0 < relative_ridge < 1 or not 0 <= interaction_order <= d or degree <= jets
            or not all(np.all(np.isfinite(v)) for v in (q, y, ell))):
        raise ValueError('Finite matching Gaussian-polynomial Hermite data required')
    components = (tuple(range(d)),) if not interaction_order else tuple(
        a for n in range(1, interaction_order+1) for a in combinations(range(d), n))
    blocks = []
    for axes in components:
        powers = np.array([p for p in product(range(degree+1), repeat=len(axes))
                           if jets < sum(p) <= degree and (parity == 0 or (-1)**sum(p) == parity)
                           and (maximum_interaction_degree is None or sum(v>0 for v in p)<=1 or sum(p)<=maximum_interaction_degree)], int)
        if len(powers):
            blocks.append((axes, powers))
    feature_count=sum(len(p) for _,p in blocks)
    model = GaussianPolynomialField(ell, tuple(blocks), np.zeros((feature_count, y.shape[-1])), parity, jets,gaussian_envelope)
    A = np.concatenate([model.features(q, o) for o in orders])
    weights = np.ones(len(orders)) if derivative_weights is None else np.asarray(derivative_weights, float)
    if weights.shape != (len(orders),) or np.any(weights <= 0):
        raise ValueError('Positive derivative weights required')
    w = np.repeat(weights, len(q)); A *= w[:, None]
    if A.shape[1]==0:
        return GaussianPolynomialField(ell,tuple(blocks),np.zeros((0,y.shape[-1])),parity,jets,gaussian_envelope),dict(feature_count=0,effective_rank=0,reference_jet_policy='No residual features of declared degree and parity')
    U, singular, Vt = np.linalg.svd(A, full_matrices=False)
    ridge = relative_ridge*singular[0]**2
    coefficients = Vt.T @ ((singular/(singular**2+ridge))[:, None]*(U.T @ (y.reshape(len(A), -1)*w[:, None])))
    model = GaussianPolynomialField(ell, tuple(blocks), coefficients, parity, jets,gaussian_envelope)
    error = A@coefficients-y.reshape(len(A), -1)*w[:, None]
    return model, dict(weighted_training_RMS=float(np.sqrt(np.mean(error**2))),
                       feature_count=A.shape[1], degree=degree,
                       effective_rank=int(np.sum(singular**2 > ridge)),
                       relative_ridge=float(relative_ridge), ridge=float(ridge),
                       reference_jet_policy='Exact by minimum feature degree; no kernel subtraction')
