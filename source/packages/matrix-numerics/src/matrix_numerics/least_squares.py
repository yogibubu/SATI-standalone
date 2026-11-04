"""Unit-aware SVD covariance and covariance whitening shared by fit engines."""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular


def svd_covariance(jacobian, *, scales=None, rcond=1.0e-10):
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or not np.all(np.isfinite(j)) or not 0 < rcond < 1:
        raise ValueError("covariance requires a finite Jacobian and a valid rank threshold")
    if scales is None:
        norms = np.linalg.norm(j, axis=0)
        scale = 1.0 / np.where(norms > 0, norms, 1.0)
    else:
        scale = np.asarray(scales, dtype=float)
    if scale.shape != (j.shape[1],) or np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("invalid covariance parameter scales")
    _, singular, vh = np.linalg.svd(j * scale, full_matrices=False)
    keep = singular > (singular[0] * rcond if len(singular) else 0.0)
    inverse = scale[:, None] * (vh[keep].T / singular[keep])
    return inverse @ inverse.T, int(np.sum(keep)), singular


class CovarianceWhitening:
    """Apply a diagonal or SPD covariance factor without forming its inverse."""

    def __init__(self, sigma, covariance=None):
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.ndim != 1 or np.any(self.sigma <= 0) or not np.all(np.isfinite(self.sigma)):
            raise ValueError("whitening sigmas must be positive and finite")
        self.cholesky = None
        if covariance is not None:
            c = np.asarray(covariance, dtype=float)
            if (
                c.shape != (len(self.sigma), len(self.sigma))
                or not np.all(np.isfinite(c))
                or not np.allclose(c, c.T, rtol=1.0e-12, atol=0.0)
                or not np.allclose(np.diag(c), self.sigma**2, rtol=1.0e-10, atol=0.0)
            ):
                raise ValueError(
                    "covariance must be symmetric with diagonal equal to declared variances"
                )
            try:
                self.cholesky = np.linalg.cholesky(c)
            except np.linalg.LinAlgError as exc:
                raise ValueError("covariance must be positive definite") from exc

    def apply(self, values):
        a = np.asarray(values, dtype=float)
        if a.ndim not in (1, 2) or len(a) != len(self.sigma):
            raise ValueError("whitening row count differs")
        if self.cholesky is None:
            return a / self.sigma.reshape((-1,) + (1,) * (a.ndim - 1))
        return solve_triangular(self.cholesky, a, lower=True, check_finite=False)

    @property
    def standardized_transform(self):
        return None if self.cholesky is None else self.cholesky / self.sigma[:, None]
