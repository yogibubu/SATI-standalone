"""Anchored Hermite force and positive transverse Hessian on a finite domain."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, expm_frechet
from .hermite_field import HermiteResidualField
from .hermite_features import GaussianPolynomialField


def residual_field_from_dict(data):
    if data.get('schema') == 'matrix.dvr.gaussian_polynomial_field.v1':
        return GaussianPolynomialField.from_dict(data)
    return HermiteResidualField.from_dict(data)


@dataclass(frozen=True)
class HermiteQuadraticBath:
    scale: np.ndarray
    force_linear: np.ndarray
    force_quadratic: np.ndarray
    matrix_reference_cholesky: np.ndarray
    matrix_left_factor: np.ndarray
    matrix_linear: np.ndarray
    bath_parity: np.ndarray
    force_residuals: dict
    matrix_residuals: dict
    representation: str = 'cholesky'

    def __post_init__(self):
        for name in ('scale', 'force_linear', 'force_quadratic',
                     'matrix_reference_cholesky', 'matrix_left_factor',
                     'matrix_linear', 'bath_parity'):
            value = np.array(getattr(self, name), copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError('Finite bath fields required')
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        d = len(self.scale); n = len(self.bath_parity)
        if (np.any(self.scale <= 0) or self.force_linear.shape != (n, d)
                or self.force_quadratic.shape != (n, d, d)
                or self.matrix_reference_cholesky.shape != (n, n)
                or self.matrix_left_factor.shape != (n, n)
                or self.matrix_linear.shape != (n*(n+1)//2, d)
                or self.representation not in ('cholesky', 'precision', 'log')
                or np.any(abs(self.bath_parity) != 1)
                or np.any(np.triu(self.matrix_reference_cholesky, 1) != 0)
                or np.any(np.diag(self.matrix_reference_cholesky) <= 0)):
            raise ValueError('Incompatible anchored Hermite bath fields')

    def _field(self, u, linear, quadratic, models, parity, second):
        d = len(self.scale); eye = np.eye(d, dtype=int)
        v = u @ linear.T + .5*np.einsum('pi,bij,pj->pb', u, quadratic, u)
        dv = linear + np.einsum('bij,pj->pbi', quadratic, u)
        ddv = np.broadcast_to(quadratic, (len(u),)+quadratic.shape).copy()
        for sign in (1, -1):
            ids = np.flatnonzero(parity == sign)
            if not len(ids):
                continue
            model = models[str(sign)]
            v[:, ids] += model.evaluate(u)
            for i, e in enumerate(eye):
                dv[:, ids, i] += model.evaluate(u, e)
                if second:
                    for j, f in enumerate(eye):
                        ddv[:, ids, i, j] += model.evaluate(u, e+f)
        return v, dv, ddv

    def evaluate(self, points, *, second=False, allow_extrapolation=False):
        q = np.asarray(points, float); d = len(self.scale); n = len(self.bath_parity)
        if q.ndim != 2 or q.shape[1] != d or not np.all(np.isfinite(q)):
            raise ValueError('Finite matrix of native coordinates required')
        u = q/self.scale
        if not allow_extrapolation and np.any(abs(u) > 1+1e-10):
            raise ValueError('Outside symmetric sampled envelope; extrapolation is unqualified')
        g, dg, ddg = self._field(u, self.force_linear, self.force_quadratic,
                                self.force_residuals, self.bath_parity, second)
        tri = list(zip(*np.tril_indices(n)))
        parity = np.array([self.bath_parity[i]*self.bath_parity[j] for i, j in tri])
        f, df, ddf = self._field(u, self.matrix_linear,
                                np.zeros((len(tri), d, d)),
                                self.matrix_residuals, parity, second)
        left = self.matrix_left_factor
        H = np.empty((len(q), n, n)); dH = np.empty(H.shape+(d,))
        ddH = np.empty(H.shape+(d, d)) if second else None
        if self.representation == 'log':
            A = np.zeros_like(H); A1 = np.zeros(H.shape+(d,))
            A2 = np.zeros(H.shape+(d, d)) if second else None
            for c, (i, j) in enumerate(tri):
                A[:, i, j] = A[:, j, i] = f[:, c]
                A1[:, i, j] = A1[:, j, i] = df[:, c]
                if second:
                    A2[:, i, j] = A2[:, j, i] = ddf[:, c]
            for p in range(len(q)):
                H[p] = left @ expm(A[p]) @ left.T
                for i in range(d):
                    dH[p, :, :, i] = left @ expm_frechet(A[p], A1[p, :, :, i], compute_expm=False) @ left.T
                    if second:
                        for j in range(i+1):
                            E = A1[p, :, :, i]; F = A1[p, :, :, j]
                            Z = np.zeros((n, n))
                            block = np.block([[A[p], E, Z], [Z, A[p], F], [Z, Z, A[p]]])
                            reverse = np.block([[A[p], F, Z], [Z, A[p], E], [Z, Z, A[p]]])
                            mixed = expm(block)[:n, 2*n:] + expm(reverse)[:n, 2*n:]
                            mixed += expm_frechet(A[p], A2[p, :, :, i, j], compute_expm=False)
                            ddH[p, :, :, i, j] = ddH[p, :, :, j, i] = left @ mixed @ left.T
        else:
            L = np.broadcast_to(self.matrix_reference_cholesky, H.shape).copy()
            L1 = np.zeros(H.shape+(d,)); L2 = np.zeros(H.shape+(d, d))
            for c, (i, j) in enumerate(tri):
                if i == j:
                    L[:, i, i] *= np.exp(f[:, c])
                    L1[:, i, i] = L[:, i, i, None]*df[:, c]
                    if second:
                        L2[:, i, i] = L[:, i, i, None, None]*(ddf[:, c]+df[:, c, :, None]*df[:, c, None, :])
                else:
                    s = np.sqrt(self.matrix_reference_cholesky[i, i]*self.matrix_reference_cholesky[j, j])
                    L[:, i, j] += s*f[:, c]; L1[:, i, j] = s*df[:, c]
                    if second:
                        L2[:, i, j] = s*ddf[:, c]
            core = L @ L.swapaxes(1, 2)
            t = np.einsum('pikt,pjk->pijt', L1, L)
            core1 = t+t.swapaxes(1, 2)
            if second:
                t = np.einsum('pikst,pjk->pijst', L2, L)
                cross = np.einsum('piks,pjkt->pijst', L1, L1)
                core2 = t+t.swapaxes(1, 2)+cross+cross.swapaxes(-1, -2)
            if self.representation == 'precision':
                inverse = np.linalg.inv(core); inverse1 = np.empty_like(core1)
                for i in range(d):
                    inverse1[..., i] = -inverse @ core1[..., i] @ inverse
                if second:
                    inverse2 = np.empty_like(core2)
                    for i in range(d):
                        for j in range(d):
                            inverse2[..., i, j] = (inverse @ core1[..., i] @ inverse @ core1[..., j] @ inverse
                                + inverse @ core1[..., j] @ inverse @ core1[..., i] @ inverse
                                - inverse @ core2[..., i, j] @ inverse)
                    core2 = inverse2
                core, core1 = inverse, inverse1
            H[:] = left @ core @ left.T
            dH[:] = np.einsum('ai,pijs,bj->pabs', left, core1, left)
            if second:
                ddH[:] = np.einsum('ai,pijst,bj->pabst', left, core2, left)
        dg /= self.scale; dH /= self.scale
        if second:
            ddg /= self.scale[:, None]*self.scale[None, :]
            ddH /= self.scale[:, None]*self.scale[None, :]
        b = -np.linalg.solve(H, g[..., None])[..., 0]
        db = -np.linalg.solve(H, dg+np.einsum('pijt,pj->pit', dH, b))
        result = dict(force=g, hessian=H, force_first=dg, hessian_first=dH,
                      response=b, response_first=db,
                      energy_lowering=.5*np.einsum('pi,pi->p', g, b),
                      energy_lowering_first=np.einsum('pit,pi->pt', dg, b)
                          + .5*np.einsum('pi,pijt,pj->pt', b, dH, b))
        if second:
            mixed = np.einsum('pijs,pjt->pist', dH, db)
            rhs = ddg+np.einsum('pijst,pj->pist', ddH, b)+mixed+mixed.swapaxes(-1, -2)
            ddb = -np.linalg.solve(H, rhs.reshape(len(q), n, d*d)).reshape(len(q), n, d, d)
            result.update(force_second=ddg, hessian_second=ddH, response_second=ddb)
        if not all(np.all(np.isfinite(v)) for v in result.values()):
            raise ValueError('Nonfinite finite-domain bath continuation')
        return result

    def to_dict(self):
        result = dict(schema='matrix.dvr.hermite_quadratic_bath.v1')
        for name, value in self.__dict__.items():
            result[name] = ({k: v.to_dict() for k, v in value.items()} if name.endswith('residuals')
                            else value.tolist() if isinstance(value, np.ndarray) else value)
        return result

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != 'matrix.dvr.hermite_quadratic_bath.v1':
            raise ValueError('Unsupported Hermite bath schema')
        fields = {name: value[name] for name in cls.__dataclass_fields__}
        for name in ('force_residuals', 'matrix_residuals'):
            fields[name] = {k: residual_field_from_dict(v) for k, v in fields[name].items()}
        return cls(**fields)
