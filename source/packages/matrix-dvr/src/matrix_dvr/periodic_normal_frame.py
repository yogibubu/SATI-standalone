"""Moving transverse frames with the actual permutation boundary condition.

A half-period path closes by an atom permutation, not necessarily by the
identity on a harmonic transverse frame. Orientation reversal of the normal
bundle cannot be removed by a continuous proper rotation. An explicit bath
reflection can carry that twist; its coefficient fields must then transform
by that reflection and its oscillator occupations enter the symmetry sector.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import null_space, qr, schur, expm
from scipy.special import expit


def normal_projector_basis(geometry, tangents, masses):
    """Mass-weighted orthonormal complement of rigid and active tangents."""
    x = np.asarray(geometry, float); d = np.asarray(tangents, float); m = np.asarray(masses, float)
    if (x.shape != (len(m), 3) or d.shape[:2] != x.shape or d.ndim != 3
            or np.any(m <= 0) or not all(np.all(np.isfinite(v)) for v in (x, d, m))):
        raise ValueError('Matching finite geometry, active tangents and positive masses required')
    x = x-np.average(x, axis=0, weights=m)
    root = np.sqrt(np.repeat(m, 3))
    translations = np.tile(np.eye(3), (len(m), 1))
    rotations = np.stack([np.cross(axis, x).ravel() for axis in np.eye(3)], axis=1)
    span = root[:, None]*np.column_stack([translations, rotations, d.reshape(x.size, -1)])
    q = null_space(span.T, rcond=1e-11)
    if q.shape[1] != x.size-6-d.shape[-1]:
        raise ValueError('Rigid plus active tangent rank is incomplete')
    return q


def orthogonal_logarithm(matrix):
    """Real antisymmetric logarithm of a proper orthogonal matrix."""
    a = np.asarray(matrix, float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1]
            or not np.allclose(a.T@a, np.eye(len(a)), atol=1e-8)
            or np.linalg.det(a) < 0):
        raise ValueError('A real continuous rotation cannot close an improper orthogonal endpoint')
    t, u = schur(a, output='real'); log = np.zeros_like(t); negative = []; i = 0
    while i < len(t):
        if i+1 < len(t) and abs(t[i+1, i]) > 1e-9:
            angle = np.arctan2(t[i+1, i], t[i, i])
            log[i+1, i] = angle; log[i, i+1] = -angle; i += 2
        else:
            if t[i, i] < 0:
                negative.append(i)
            i += 1
    if len(negative) % 2:
        raise ValueError('Unpaired negative real eigenvalue in a proper endpoint')
    for i, j in zip(negative[::2], negative[1::2]):
        log[j, i] = np.pi; log[i, j] = -np.pi
    result = u@log@u.T
    if np.max(abs(expm(result)-a)) > 1e-7:
        raise ValueError('Orthogonal endpoint logarithm failed reconstruction')
    return result


def exchange_normal_frames(angles, geometries, tangents, masses, permutation, *, preferred_twist_axis=None,
                           initial_mass_weighted_frame=None):
    """Transport a half-cycle and close it with an explicit bath reflection.

Return mass-weighted normal frames Q, with Q(end)=P Q(0) S. S=I for a proper
endpoint; for an improper endpoint its first axis carries one reflection.
preferred_twist_axis is a physical Cartesian vector used to orient that first
normal axis, not a fitted parameter or electronic observation.
"""
    theta = np.asarray(angles, float); x = np.asarray(geometries, float); d = np.asarray(tangents, float)
    m = np.asarray(masses, float); perm = np.asarray(permutation, int)
    if (theta.ndim != 1 or len(theta) < 9 or x.shape != (len(theta), len(m), 3)
            or d.shape[:3] != x.shape or sorted(perm.tolist()) != list(range(len(m)))
            or not np.allclose(m[perm], m, atol=1e-10, rtol=0)
            or abs(x[-1]-x[0, perm]).max() > 1e-8 or not np.all(np.diff(theta)>0)):
        raise ValueError('A resolved permutation-closed half path with equal exchanged masses is required')
    q = normal_projector_basis(x[0], d[0], m); nb = q.shape[1]
    if initial_mass_weighted_frame is not None:
        initial = np.asarray(initial_mass_weighted_frame, float)
        if (initial.shape != q.shape or not np.all(np.isfinite(initial))
                or np.max(abs(initial.T@initial-np.eye(nb)))>1e-9
                or np.max(abs(q@(q.T@initial)-initial))>1e-9):
            raise ValueError('Initial bath normal modes must be a complete orthonormal transverse frame')
        q = initial
    if preferred_twist_axis is not None:
        vector = np.asarray(preferred_twist_axis, float).reshape(-1)*np.sqrt(np.repeat(m, 3))
        projected = q.T@vector
        if np.linalg.norm(projected) < 1e-10:
            raise ValueError('Preferred twist axis has no transverse component')
        rotation, _ = qr(projected[:, None], mode='full'); q = q@rotation
    parallel = [q]; minimum_step_overlap = 1.
    for geometry, derivative in zip(x[1:], d[1:]):
        normal = normal_projector_basis(geometry, derivative, m)
        projected = normal@(normal.T@parallel[-1])
        left, singular, right = np.linalg.svd(projected, full_matrices=False)
        minimum_step_overlap = min(minimum_step_overlap, float(min(singular)))
        if min(singular) < .5:
            raise ValueError('Unresolved normal-frame transport step; refine analytic path sampling only')
        parallel.append(left@right)
    parallel = np.array(parallel)
    permuted = parallel[0].reshape(len(m), 3, nb)[perm].reshape(3*len(m), nb)
    endpoint = parallel[-1].T@permuted
    twist = np.eye(nb)
    determinant = float(np.linalg.det(endpoint))
    if determinant < 0:
        twist[0, 0] = -1
    rotation = endpoint@twist
    log = orthogonal_logarithm(rotation)
    fraction = (theta-theta[0])/(theta[-1]-theta[0])
    # Endpoint values alone do not close a differentiable frame when the
    # proper closing rotation does not commute with the bath reflection.
    # A flat-endpoint gauge leaves every endpoint derivative horizontal and
    # makes the permutation extension smooth, including at the seam.
    schedule=np.zeros_like(fraction);schedule[-1]=1.
    first_schedule=np.zeros_like(fraction);second_schedule=np.zeros_like(fraction)
    inside=(fraction>0)&(fraction<1);t=fraction[inside]
    value=expit(-1/t+1/(1-t));u1=1/t**2+1/(1-t)**2;u2=-2/t**3+2/(1-t)**3
    schedule[inside]=value
    first_schedule[inside]=value*(1-value)*u1/(theta[-1]-theta[0])
    second_schedule[inside]=value*(1-value)*(u2+(1-2*value)*u1**2)/(theta[-1]-theta[0])**2
    corrected = np.array([frame@expm(t*log) for frame, t in zip(parallel, schedule)])
    closure = float(abs(corrected[-1]-permuted@twist).max())
    gram = float(abs(np.einsum('pni,pnj->pij', corrected, corrected)-np.eye(nb)).max())
    if closure > 1e-7 or gram > 1e-8:
        raise ValueError('Permutation-aware moving frame failed closure')
    return dict(mass_weighted_frames=corrected, bath_exchange=twist,
        endpoint_transport_matrix=endpoint, endpoint_determinant=determinant,
        closing_schedule='flat_endpoint', closing_generator=log,
        closing_parameter=schedule, closing_parameter_first=first_schedule, closing_parameter_second=second_schedule,
        exchange_closure_error=closure, orthogonality_error=gram,
        minimum_transport_step_overlap=minimum_step_overlap)


__all__ = ['normal_projector_basis', 'orthogonal_logarithm', 'exchange_normal_frames']
