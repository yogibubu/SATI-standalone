"""Bounded periodic vector fields and positive matrix-log quadratic baths."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.special import gammaln
from .damped_periodic import damped_angular_features


@dataclass(frozen=True)
class PeriodicVectorField:
    powers: np.ndarray
    coefficients: np.ndarray
    scales: np.ndarray
    tail_lengths_radian: np.ndarray
    envelope_power: int = 4
    factorial_normalization: bool = True

    def __post_init__(self):
        for name in ('powers', 'coefficients', 'scales', 'tail_lengths_radian'):
            value = np.array(getattr(self, name), dtype=int if name == 'powers' else float, copy=True)
            if not np.all(np.isfinite(value)): raise ValueError('Finite periodic field required')
            value.flags.writeable = False; object.__setattr__(self, name, value)
        if (self.powers.ndim != 2 or self.powers.shape[1] != len(self.scales) or np.any(self.powers < 0)
                or self.coefficients.ndim != 2 or self.coefficients.shape[0] != len(self.powers)
                or self.tail_lengths_radian.shape != self.scales.shape or np.any(self.scales <= 0)
                or np.any(self.tail_lengths_radian <= 0) or self.envelope_power not in (1, 2, 3, 4)):
            raise ValueError('Compatible periodic field parameters required')

    def features(self, points, derivative=None):
        result = damped_angular_features(points, self.powers, self.scales, self.tail_lengths_radian, self.envelope_power, derivative)
        if self.factorial_normalization: result *= np.exp(-.5*np.sum(gammaln(self.powers+1), axis=1))[None, :]
        return result

    def evaluate(self, points, derivative=None):
        return self.features(points, derivative)@self.coefficients

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_vector_field.v2', powers=self.powers.tolist(), coefficients=self.coefficients.tolist(),
                    scales=self.scales.tolist(), tail_lengths_radian=self.tail_lengths_radian.tolist(), envelope_power=self.envelope_power,
                    factorial_normalization=self.factorial_normalization)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') not in ('matrix.dvr.periodic_vector_field.v1', 'matrix.dvr.periodic_vector_field.v2'): raise ValueError('Unsupported periodic field')
        return cls(*[np.array(data[k]) for k in ('powers', 'coefficients', 'scales', 'tail_lengths_radian')], data['envelope_power'], bool(data.get('factorial_normalization', False)))


def fit_periodic_vector_field(points, observations, *, orders, powers, scales, tail_lengths_radian,
                              anchor_powers, anchor_coefficients, component_parity, relative_ridge,
                              derivative_weights, envelope_power=4, observation_mask=None,
                              anchor_tail_lengths_radian=None, anchor_envelope_power=None):
    q = np.asarray(points, float); y = np.asarray(observations, float); powers = np.asarray(powers, int)
    anchor_powers = np.asarray(anchor_powers, int); anchor_coefficients = np.asarray(anchor_coefficients, float)
    parity = np.asarray(component_parity, int); weights = np.asarray(derivative_weights, float)
    orders = tuple(tuple(o) for o in orders)
    if y.shape != (len(orders), len(q), len(parity)) or np.any(abs(parity) != 1) or weights.shape != (len(orders),) or np.any(weights <= 0) or not 0 < relative_ridge < 1:
        raise ValueError('Compatible vector Hermite observations, parities and weights required')
    mask=np.ones((len(orders),len(q)),bool) if observation_mask is None else np.asarray(observation_mask)
    if mask.shape!=(len(orders),len(q)) or mask.dtype!=np.dtype(bool) or not mask.any() or not np.all(np.isfinite(y[mask])):
        raise ValueError('Boolean observed-jet mask and finite observed values required')
    anchor_lengths=tail_lengths_radian if anchor_tail_lengths_radian is None else anchor_tail_lengths_radian
    anchor_power=envelope_power if anchor_envelope_power is None else anchor_envelope_power
    base = PeriodicVectorField(anchor_powers, anchor_coefficients, np.ones(q.shape[1]), np.array(anchor_lengths), anchor_power, False)
    residual = PeriodicVectorField(powers, np.zeros((len(powers), len(parity))), np.array(scales), np.array(tail_lengths_radian), envelope_power)
    A = np.concatenate([residual.features(q[mask[k]], o)*w for k,(o,w) in enumerate(zip(orders, weights))])
    rhs = np.concatenate([(y[k,mask[k]]-base.evaluate(q[mask[k]], o))*weights[k] for k, o in enumerate(orders)])
    coefficients = np.zeros((len(powers), len(parity))); ranks = []
    for sign in (-1, 1):
        ids = np.flatnonzero((-1.)**powers.sum(axis=1) == sign); outputs = np.flatnonzero(parity == sign)
        if not len(ids) or not len(outputs): continue
        U, s, vt = np.linalg.svd(A[:, ids], full_matrices=False); ridge = relative_ridge*s[0]**2
        c = vt.T@((s/(s*s+ridge))[:, None]*(U.T@rhs[:, outputs]))
        coefficients[np.ix_(ids, outputs)] = c; ranks.append(int(np.sum(s*s > ridge)))
    # Store one common angular scale without changing the physical anchor jets.
    anchor_scaled = anchor_coefficients*(np.prod(np.asarray(scales)[None, :]**anchor_powers, axis=1)*np.exp(.5*np.sum(gammaln(anchor_powers+1), axis=1)))[:, None]
    if anchor_tail_lengths_radian is not None or anchor_envelope_power is not None:
        from .response_bath import AdditivePeriodicResponseField
        fitted_residual=PeriodicVectorField(powers,coefficients,np.array(scales),np.array(tail_lengths_radian),envelope_power)
        model=AdditivePeriodicResponseField((base,fitted_residual))
    else:
        model = PeriodicVectorField(np.vstack([anchor_powers, powers]), np.vstack([anchor_scaled, coefficients]), np.array(scales), np.array(tail_lengths_radian), envelope_power)
    return model, dict(component_ranks=ranks, relative_ridge=relative_ridge,
                       observed_jet_counts=mask.sum(axis=1).tolist(),
                       weighted_training_RMS=float(np.sqrt(np.mean((A@coefficients-rhs)**2))))


@dataclass(frozen=True)
class PeriodicQuadraticBath:
    reference_cholesky: np.ndarray
    force: PeriodicVectorField
    matrix_log: PeriodicVectorField

    def __post_init__(self):
        L = np.array(self.reference_cholesky, float, copy=True)
        if L.ndim != 2 or L.shape[0] != L.shape[1] or np.any(np.diag(L) <= 0) or np.any(np.triu(L, 1) != 0) or not np.all(np.isfinite(L)):
            raise ValueError('Finite positive reference Cholesky matrix required')
        n = len(L)
        if self.force.coefficients.shape[1] != n or self.matrix_log.coefficients.shape[1] != n*(n+1)//2:
            raise ValueError('Compatible force/matrix fields required')
        L.flags.writeable = False; object.__setattr__(self, 'reference_cholesky', L)

    def evaluate(self, points, *, second=False, derivatives=True):
        q = np.asarray(points, float); d = q.shape[1]; n = len(self.reference_cholesky); eye = np.eye(d, dtype=int)
        if second and not derivatives: raise ValueError('Second derivatives require first derivatives')
        g = self.force.evaluate(q); dg = np.stack([self.force.evaluate(q, e) for e in eye], axis=-1) if derivatives else None
        ddg = np.stack([np.stack([self.force.evaluate(q, e+f) for f in eye], axis=-1) for e in eye], axis=-2) if second else None
        tri = np.tril_indices(n)
        def matrices(order=None):
            values = self.matrix_log.evaluate(q, order); A = np.zeros((len(q), n, n))
            A[:, tri[0], tri[1]] = values; A[:, tri[1], tri[0]] = values
            return A
        A = matrices(); A1 = [matrices(e) for e in eye] if derivatives else None; H = np.empty_like(A); H1 = np.empty(A.shape+(d,)) if derivatives else None; H2 = np.empty(A.shape+(d, d)) if second else None
        L = self.reference_cholesky
        for k in range(len(q)):
            spectrum = np.linalg.eigvalsh(A[k])
            log_condition_bound = 2*np.log(np.linalg.cond(L))+spectrum[-1]-spectrum[0]
            if log_condition_bound > np.log(1e12):
                raise ValueError('Matrix-log continuation exceeds the declared numerical condition bound; no Hessian clipping permitted')
            H[k] = L@expm(A[k])@L.T
            for i in range(d) if derivatives else ():
                H1[k, :, :, i] = L@expm_frechet(A[k], A1[i][k], compute_expm=False)@L.T
        if second:
            for i in range(d):
                for j in range(i+1):
                    A2 = matrices(eye[i]+eye[j]); Z = np.zeros((n, n))
                    for k in range(len(q)):
                        block = np.block([[A[k], A1[i][k], Z], [Z, A[k], A1[j][k]], [Z, Z, A[k]]])
                        reverse = np.block([[A[k], A1[j][k], Z], [Z, A[k], A1[i][k]], [Z, Z, A[k]]])
                        mixed = expm(block)[:n, 2*n:]+expm(reverse)[:n, 2*n:]+expm_frechet(A[k], A2[k], compute_expm=False)
                        H2[k, :, :, i, j] = H2[k, :, :, j, i] = L@mixed@L.T
        response = -np.linalg.solve(H, g[..., None])[..., 0]
        result = dict(force=g, force_first=dg, hessian=H, hessian_first=H1, response=response,
                      energy_lowering=.5*np.einsum('ni,ni->n', g, response))
        if derivatives:
            response_first = np.stack([-np.linalg.solve(H, (dg[:, :, i]+np.einsum('pij,pj->pi', H1[:, :, :, i], response))[..., None])[..., 0] for i in range(d)], axis=-1)
            result['response_first'] = response_first
        if second:
            mixed=np.einsum('pijs,pjt->pist',H1,response_first)
            rhs=ddg+np.einsum('pijst,pj->pist',H2,response)+mixed+mixed.swapaxes(-1,-2)
            response_second=-np.linalg.solve(H,rhs.reshape(len(q),n,d*d)).reshape(len(q),n,d,d)
            result.update(force_second=ddg,hessian_second=H2,response_second=response_second)
        return result

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_quadratic_bath.v1', reference_cholesky=self.reference_cholesky.tolist(),
                    force=self.force.to_dict(), matrix_log=self.matrix_log.to_dict(),
                    physical_tail_accuracy_qualified=False)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema')=='matrix.dvr.periodic_response_quadratic_bath.v1':
            from .response_bath import PeriodicResponseQuadraticBath
            return PeriodicResponseQuadraticBath.from_dict(data)
        if data.get('schema') != 'matrix.dvr.periodic_quadratic_bath.v1': raise ValueError('Unsupported periodic bath')
        def field(payload):
            if payload.get('schema')=='matrix.dvr.periodic_shepard_hermite.v1':
                from .periodic_shepard import PeriodicShepardHermiteField
                return PeriodicShepardHermiteField.from_dict(payload)
            return PeriodicVectorField.from_dict(payload)
        return cls(np.array(data['reference_cholesky']),field(data['force']),field(data['matrix_log']))
