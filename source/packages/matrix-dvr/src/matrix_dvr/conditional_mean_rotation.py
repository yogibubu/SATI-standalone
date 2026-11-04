"""Rotational response to the quadratic conditional minimum of a fast bath.

The supplied force field is never truncated here. Explicit coordinates select
an observable response, not an isotope force-field transformation. A caller
using a frozen cubic away from its reference must record that approximation.
"""
import numpy as np
from .full_metric import KINETIC_CM_AMU_ANGSTROM2


def conditional_mean_second_response(hessian, cubic, explicit_indices):
    """Return d²b_fast/dy² at a conditional stationary quadratic origin.

    Hessians and symmetric cubics may have broadcastable leading batch axes.
    The linear relaxation is included in both explicit cubic legs. Force
    constants and coordinates must use the same unweighted coordinate chart.
    """
    H = np.asarray(hessian, float)
    C = np.asarray(cubic, float)
    raw = np.asarray(explicit_indices)
    if H.ndim < 2 or H.shape[-1] != H.shape[-2]:
        raise ValueError('Square Hessians required')
    n = H.shape[-1]
    if (C.shape[-3:] != (n, n, n) or raw.ndim != 1
            or not 0 < len(raw) < n or np.any(raw != np.rint(raw))
            or len(set(raw.tolist())) != len(raw) or np.any(raw < 0)
            or np.any(raw >= n) or not np.allclose(H, H.swapaxes(-1, -2))
            or not np.allclose(C, C.swapaxes(-1, -2))
            or not np.allclose(C, C.swapaxes(-2, -3))
            or not all(np.all(np.isfinite(a)) for a in (H, C))):
        raise ValueError('Matching symmetric finite force jets and explicit subset required')
    selected = raw.astype(int)
    fast = np.array([i for i in range(n) if i not in selected])
    ff = np.take(np.take(H, fast, axis=-2), fast, axis=-1)
    fs = np.take(np.take(H, fast, axis=-2), selected, axis=-1)
    if np.linalg.eigvalsh(ff).min() <= 0:
        raise ValueError('Stable conditional fast Hessian required')
    linear = -np.linalg.solve(ff, fs)
    E = np.zeros(H.shape[:-2] + (n, len(selected)))
    E[..., fast, :] = linear
    E[..., selected, :] = np.eye(len(selected))
    vertex = np.einsum('...fuv,...ui,...vj->...fij',
                       np.take(C, fast, axis=-3), E, E, optimize=True)
    shape = np.broadcast_shapes(ff.shape[:-2], vertex.shape[:-3])
    rhs = np.broadcast_to(vertex, shape + vertex.shape[-3:])
    second = -np.linalg.solve(ff, rhs.reshape(shape + (len(fast), -1)))
    second = second.reshape(shape + (len(fast), len(selected), len(selected)))
    return dict(fast_indices=fast, explicit_indices=selected,
                conditional_response=linear, conditional_response_second=second)


def conditional_mean_rotational_second(coordinates_angstrom, masses_amu,
                                       fast_first, response_second):
    """Inverse-inertia kinetic curvature induced by a fast mean displacement.

    Fast Cartesian tangents are projected onto the local mass Eckart section.
    The result is a true second derivative of the kinetic rotational block,
    in cm^-1 per squared explicit coordinate. Rotational energy coefficients
    are half that block; an expectation contributes R2:<yy>/4. This term is
    separate from direct inertial curvature and Coriolis Schur completion.
    """
    x = np.asarray(coordinates_angstrom, float)
    m = np.asarray(masses_amu, float)
    D = np.asarray(fast_first, float)
    M = np.asarray(response_second, float)
    if (x.shape[-2:] != (len(m), 3) or D.shape[-3:-1] != x.shape[-2:]
            or M.shape[-3] != D.shape[-1] or M.shape[-2] != M.shape[-1]
            or np.any(m <= 0) or not np.allclose(M, M.swapaxes(-1, -2))
            or not all(np.all(np.isfinite(a)) for a in (x, m, D, M))):
        raise ValueError('Matching Cartesian fast tangents, masses and second response required')
    x = x - np.einsum('i,...ia->...a', m, x)[..., None, :] / m.sum()
    D = D - np.einsum('i,...ias->...as', m, D)[..., None, :, :] / m.sum()
    inertia = (np.einsum('i,...ia,...ia->...', m, x, x)[..., None, None] * np.eye(3)
               - np.einsum('i,...ia,...ib->...ab', m, x, x))
    inv = np.linalg.inv(inertia)
    angular = np.einsum('i,...isa->...as', m,
                        np.cross(x[..., :, None, :], D.swapaxes(-1, -2)))
    omega = np.einsum('...ab,...bs->...as', inv, angular)
    D = D - np.cross(omega.swapaxes(-1, -2)[..., None, :, :],
                     x[..., :, None, :]).swapaxes(-1, -2)
    first = (2 * np.einsum('i,...ia,...ias->...s', m, x, D)[..., :, None, None]
             * np.eye(3))
    cross = np.einsum('i,...ia,...ibs->...sab', m, x, D)
    first -= cross + cross.swapaxes(-1, -2)
    rotation_first = -KINETIC_CM_AMU_ANGSTROM2 * np.einsum(
        '...ab,...sbc,...cd->...sad', inv, first, inv)
    return np.einsum('...fab,...fij->...abij', rotation_first, M, optimize=True)
