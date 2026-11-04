"""Cross-geometry diagnostics for printed single-excitation fingerprints."""
import numpy as np


def configuration_transition_overlaps(left, right, mo_cross_overlap):
    """Normalized transition-density overlaps in a common AO spatial frame.

    Inputs have shape (state, MO, MO), with occupied-to-virtual coefficients.
    The cross overlap is C_left^dagger S_cross C_right. Unlike comparing MO
    indices, this retains orbital phase and mixing. Printed coefficients may
    be thresholded: these are fingerprint overlaps, not many-electron overlaps.
    """
    left, right, cross = map(np.asarray, (left, right, mo_cross_overlap))
    if (left.ndim != 3 or right.ndim != 3 or left.shape[1] != left.shape[2]
            or right.shape[1] != right.shape[2]
            or cross.shape != (left.shape[1], right.shape[1])
            or not all(np.all(np.isfinite(a)) for a in (left, right, cross))):
        raise ValueError('finite state-by-square-MO fingerprints and compatible cross overlap required')
    norms = [np.linalg.norm(a, axis=(1, 2)) for a in (left, right)]
    if any(np.any(a == 0) for a in norms):
        raise ValueError('nonzero transition fingerprints required')
    transported = np.einsum('ik,qkl,jl->qij', cross, right, cross.conj(), optimize=True)
    return np.einsum('pij,qij->pq', left.conj(), transported, optimize=True)/(
        norms[0][:, None]*norms[1][None, :])
