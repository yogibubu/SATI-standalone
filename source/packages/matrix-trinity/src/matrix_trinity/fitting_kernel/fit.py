from __future__ import annotations

import numpy as np


def huber_weights(r, delta):
    w = np.ones_like(r)
    mask = np.abs(r) > delta
    w[mask] = delta / np.abs(r[mask])
    return w


def solve_weighted_ridge(A, y, w, ridge):
    # apply weights
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    yw = y * sw
    # ridge
    m = A.shape[1]
    lhs = Aw.T @ Aw + ridge * np.eye(m)
    rhs = Aw.T @ yw
    return np.linalg.solve(lhs, rhs)


def robust_fit(A, y, delta=1.0, ridge=1e-6, max_iter=50, tol=1e-10):
    # initial least squares
    w = np.ones_like(y)
    coeff = solve_weighted_ridge(A, y, w, ridge)
    for _ in range(max_iter):
        r = y - A @ coeff
        w_new = huber_weights(r, delta)
        coeff_new = solve_weighted_ridge(A, y, w_new, ridge)
        if np.linalg.norm(coeff_new - coeff) < tol:
            coeff = coeff_new
            break
        coeff = coeff_new
    return coeff
