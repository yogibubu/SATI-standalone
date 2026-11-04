"""Complete an explicit active tangent space without duplicating bath modes.

All Cartesian columns supplied here are mass weighted. The linear rechart is
an exact transformation of an existing coordinate jet, not an acquisition of
derivatives or a claim that the active and complementary sectors decouple.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space


@dataclass(frozen=True)
class ActiveBathPartition:
    active: np.ndarray
    bath: np.ndarray
    rotation: np.ndarray
    tangent_coordinates: np.ndarray


def complete_active_frame(vibrations, tangents, *, tolerance=1e-9):
    """Return an orthonormal active/bath frame spanning exactly ``vibrations``.

    Tangents must already exclude rigid motion and belong to the supplied
    complete vibrational frame. An out-of-span component is an error: silently
    projecting it would change the geometry of the explicit active operator.
    """
    frame = np.asarray(vibrations, float)
    tangent = np.asarray(tangents, float)
    if (frame.ndim != 2 or tangent.ndim != 2
            or tangent.shape[0] != frame.shape[0]
            or not 0 < tangent.shape[1] < frame.shape[1]
            or not np.all(np.isfinite(frame)) or not np.all(np.isfinite(tangent))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('finite compatible vibrational and active frames required')
    if not np.allclose(frame.T @ frame, np.eye(frame.shape[1]), atol=tolerance, rtol=0):
        raise ValueError('vibrational frame must be orthonormal')
    norm = np.linalg.norm(tangent, axis=0)
    if np.any(norm == 0):
        raise ValueError('active tangents must be nonzero and independent')
    coordinates = frame.T @ (tangent / norm)
    residual = tangent / norm - frame @ coordinates
    if np.linalg.norm(residual, axis=0).max() > tolerance:
        raise ValueError('active tangent leaves the supplied vibrational space')
    singular = np.linalg.svd(coordinates, compute_uv=False)
    if singular[-1] <= tolerance * singular[0]:
        raise ValueError('active tangents must be nonzero and independent')
    active, _ = np.linalg.qr(coordinates, mode='reduced')
    # Fix column signs; QR still preserves the declared order of active axes.
    for i in range(active.shape[1]):
        if active[:, i] @ coordinates[:, i] < 0:
            active[:, i] *= -1
    bath = null_space(active.T)
    rotation = np.column_stack([active, bath])
    return ActiveBathPartition(frame @ active, frame @ bath, rotation,
                               active.T @ (frame.T @ tangent))


def linear_rechart_packet(packet, displacement):
    """Transform a complete jet for ``old_coordinates = displacement @ new``.

    The two momentum indices of G, G1 and G2 transform contravariantly;
    derivative indices and every potential index transform covariantly.
    Units, quartic coverage and scalar origin remain those of the input.
    A nonstationary gradient, when supplied, is transformed as well.
    """
    transform = np.asarray(displacement, float)
    if (transform.ndim != 2 or transform.shape[0] != transform.shape[1]
            or not np.all(np.isfinite(transform))):
        raise ValueError('a finite invertible square coordinate transform is required')
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as exc:
        raise ValueError('a finite invertible square coordinate transform is required') from exc
    n = len(transform)
    ranks = dict(hessian=2, cubic=3, quartic=4, G=2, G1=3, G2=4)
    arrays = {key: np.asarray(packet[key], float) for key in ranks}
    if any(a.shape != (n,)*ranks[key] or not np.all(np.isfinite(a))
           for key, a in arrays.items()):
        raise ValueError('complete finite potential and metric jets are required')
    result = dict(packet)
    for key in ('hessian', 'cubic', 'quartic'):
        rank = ranks[key]
        old, new = 'abcd'[:rank], 'ijkl'[:rank]
        indices = ','.join(a+b for a, b in zip(old, new))
        result[key] = np.einsum(old+','+indices+'->'+new, arrays[key],
                                *([transform]*rank), optimize=True)
    result['G'] = inverse @ arrays['G'] @ inverse.T
    result['G1'] = np.einsum('ia,jb,abc,ck->ijk', inverse, inverse,
                             arrays['G1'], transform, optimize=True)
    result['G2'] = np.einsum('ia,jb,abcd,ck,dl->ijkl', inverse, inverse,
                             arrays['G2'], transform, transform, optimize=True)
    if 'gradient' in packet:
        gradient = np.asarray(packet['gradient'], float)
        if gradient.shape != (n,) or not np.all(np.isfinite(gradient)):
            raise ValueError('gradient must be finite and match the coordinate dimension')
        result['gradient'] = transform.T @ gradient
    return result
