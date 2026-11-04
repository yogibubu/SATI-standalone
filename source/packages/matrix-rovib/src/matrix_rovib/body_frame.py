"""Exact finite-basis transport between molecular body frames.

Coefficients are stored in a reference embedding. A pointwise SO(3) field S
transports both the DVR Hamiltonian and its retained basis with W_J(S).
Consequently no discrete product rule or truncated Fourier expansion of S is
assumed. Tensor surfaces remain in the reference frame, including dipoles.
"""
from dataclasses import replace

import numpy as np


FRAME_KEY = "body_frame_transport"


def frame_rotations(surface):
    """Return the reference-to-output SO(3) field, or None for the reference."""
    payload = dict(surface.metadata or {}).get(FRAME_KEY)
    if payload is None:
        return None
    if payload.get("schema") != "matrix.lock.body_frame_transport.v1" or payload.get("coefficient_frame") != "reference":
        raise ValueError("unsupported LOCK body-frame transport contract")
    return _validate_rotations(payload["rotations"], int(np.prod(surface.grid_shape)))


def _validate_rotations(rotations, size):
    field = np.asarray(rotations, dtype=float)
    if field.shape != (size, 3, 3) or not np.all(np.isfinite(field)):
        raise ValueError("body-frame rotations require finite (grid_size, 3, 3) shape")
    if not np.allclose(field.swapaxes(-1, -2) @ field, np.eye(3), atol=1e-10, rtol=0) or not np.allclose(np.linalg.det(field), 1, atol=1e-10, rtol=0):
        raise ValueError("body-frame transport requires proper SO(3) rotations")
    return field


def transport_lam_rotation_surface(surface, rotations):
    """Reframe a surface without re-discretizing its kinetic operator.

    ``X_output[q] = X_current[q] @ rotations[q]``. All coefficient arrays and
    contracted eigenvector coefficients stay in the reference embedding.
    Use :func:`lam_rotation_grid_wavefunctions` for output-frame amplitudes.
    This also transports embedded ORACLE actions, and composes repeated calls.
    """
    field = _validate_rotations(rotations, int(np.prod(surface.grid_shape)))
    previous = frame_rotations(surface)
    metadata = dict(surface.metadata or {})
    payload = metadata.get("oracle_pi_grid_action")
    if payload is not None:
        from matrix_chem import PermutationInversionGridAction
        metadata["oracle_pi_grid_action"] = PermutationInversionGridAction.from_dict(payload).reframe(field).to_dict()
    metadata[FRAME_KEY] = {
        "schema": "matrix.lock.body_frame_transport.v1",
        "coefficient_frame": "reference",
        "rotations": (field if previous is None else previous @ field).tolist(),
    }
    return replace(surface, metadata=metadata)


def lam_rotation_grid_wavefunctions(surface, result, J, *, state_indices=None):
    """Expand retained states in the output frame, with shape (grid, K, state).

    Transport acts in O(grid*(2J+1)^2*states), without a dense grid-rotation
    matrix. This is the transported contracted basis, not a projection back
    onto the original scalar LAM space.
    """
    from .pi_symmetry import _body_rotation_matrices

    block = result.blocks[J]
    coefficients = block.vectors if state_indices is None else block.vectors[:, state_indices]
    coefficients = np.asarray(coefficients).reshape(len(result.basis.energies_MHz), 2*J+1, -1)
    values = np.einsum("qn,nks->qks", result.basis.vectors, coefficients, optimize=True)
    field = frame_rotations(surface)
    if field is not None:
        for start in range(0, len(field), 64):
            stop = min(start + 64, len(field))
            rotation = _body_rotation_matrices(field[start:stop], J, result.representation)
            values[start:stop] = rotation @ values[start:stop]
    return values
