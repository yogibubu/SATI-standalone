"""Complete J=0 metric for any nonlinear valence triatomic, via the metric owner."""

import numpy as np
from .full_metric import cartesian_internal_rotation_metric_grid


def valence_triatomic_metric(points, masses_amu):
    """Coordinates are r01, r02 (angstrom), theta102 (radians); masses 0,1,2.

    The fixed embedding matches InternalTreeChart((0,1,2),((),(0,),(0,1))).
    The full six-dimensional vibration/rotation metric is inverted before its
    vibrational block is selected. Physical volume is proportional to
    r01^2*r02^2*sin(theta), including the rotational volume at J=0.
    """
    q = np.asarray(points, float)
    m = np.asarray(masses_amu, float)
    if (
        q.ndim != 2
        or q.shape[1] != 3
        or m.shape != (3,)
        or np.any(m <= 0)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(m))
        or np.any(q[:, :2] <= 0)
        or np.any(q[:, 2] <= 0)
        or np.any(q[:, 2] >= np.pi)
    ):
        raise ValueError("Positive bonds, nonsingular valence angles and three masses required")
    r, s, t = q.T
    sn = np.sin(t)
    cs = np.cos(t)
    x = np.zeros((len(q), 3, 3))
    D = np.zeros((len(q), 3, 3, 3))
    x[:, 1, 2] = r
    x[:, 2, 0] = s * sn
    x[:, 2, 2] = s * cs
    D[:, 1, 2, 0] = 1
    D[:, 2, 0, 1] = sn
    D[:, 2, 2, 1] = cs
    D[:, 2, 0, 2] = s * cs
    D[:, 2, 2, 2] = -s * sn
    full = cartesian_internal_rotation_metric_grid(x, m, D)
    volume = r * r * s * s * sn
    gradient = np.column_stack([2 / r, 2 / s, cs / sn])
    return dict(
        kinetic_cm=full["kinetic_cm"][:, :3, :3],
        volume_density=volume,
        log_volume_gradient=gradient,
        full_covariant_metric=full["covariant_metric"],
        coordinates_angstrom=x,
    )
