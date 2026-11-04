"""Restore rigid Cartesian jet identities while preserving a frozen chart.

For nonstationary geometries H times a rotation tangent equals the rotated
gradient, not zero. The supplied six linear gauge rows define the chart whose
internal first and second derivatives are retained exactly by this correction.
"""
import numpy as np


def restore_rigid_cartesian_covariance(coordinates, gradient, hessian, gauge_rows):
    x = np.asarray(coordinates, float); g = np.asarray(gradient, float).reshape(-1); H = np.asarray(hessian, float)
    C = np.asarray(gauge_rows, float); nc = x.size
    if x.ndim != 2 or x.shape[1] != 3 or g.shape != (nc,) or H.shape != (nc, nc) or C.shape != (6, nc) or not all(np.all(np.isfinite(v)) for v in (x, g, H, C)):
        raise ValueError('Compatible finite nonlinear geometry, Cartesian jet and six gauge rows required')
    if not np.allclose(H, H.T, atol=1e-10): raise ValueError('Symmetric Cartesian Hessian required')
    centered = x-x.mean(axis=0); axes = np.eye(3)
    R = np.column_stack([np.tile(e, len(x)) for e in axes]+[np.cross(e, centered).reshape(-1) for e in axes])
    dual = np.linalg.solve(C@R, C).T
    corrected_g = g-dual@(R.T@g)
    target = np.column_stack([np.zeros(nc) for _ in axes]+[np.cross(e, corrected_g.reshape(-1, 3)).reshape(-1) for e in axes])
    residual = target-H@R; inner = R.T@residual
    if np.max(abs(inner-inner.T)) > 1e-8*max(1., float(np.max(abs(inner)))):
        raise ValueError('Rigid jet identities are not integrable')
    corrected_H = H+residual@dual.T+dual@residual.T-dual@((inner+inner.T)/2)@dual.T
    corrected_H = (corrected_H+corrected_H.T)/2
    return dict(gradient=corrected_g, hessian=corrected_H,
        gradient_correction_max=float(np.max(abs(corrected_g-g))), hessian_correction_max=float(np.max(abs(corrected_H-H))),
        rigid_gradient_residual=float(np.max(abs(R.T@corrected_g))),
        rigid_Hessian_residual=float(np.max(abs(corrected_H@R-target))),
        preservation='Internal E/G/H unchanged for first/second Cartesian chart jets annihilated by the supplied linear gauge rows')
