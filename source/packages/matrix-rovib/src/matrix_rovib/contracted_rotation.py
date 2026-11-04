"""LOCK diagonalization of already contracted rotation--vibration operators.

The supplied coefficients use energy units common to the zero-J eigenvalues.
Every vibrational matrix element is retained, including those of sym(Ja Jb).
"""
import numpy as np
from scipy.linalg import eigh
from .native_rotor import angular_momentum_operators
from .lam_rotation import _assemble_coupled_hamiltonian


def solve_contracted_rotation(energies, linear, quadratic, *, J=1,
                              representation='Ir', backend='auto'):
    """Use the same coupled-Hamiltonian assembly as ``solve_lam_rotation``."""
    e = np.asarray(energies, float)
    linear = np.asarray(linear, complex)
    quadratic = np.asarray(quadratic, complex)
    n = len(e)
    if (e.ndim != 1 or not np.all(np.isfinite(e))
            or linear.shape != (3, n, n)
            or quadratic.shape != (3, 3, n, n)
            or not np.all(np.isfinite(linear))
            or not np.all(np.isfinite(quadratic))):
        raise ValueError('Finite compatible contracted coefficient arrays required')
    error = max(float(abs(linear-linear.conj().swapaxes(-1, -2)).max()),
                float(abs(quadratic-quadratic.conj().swapaxes(-1, -2)).max()),
                float(abs(quadratic-quadratic.swapaxes(0, 1)).max()))
    if error > 1e-9 * max(1., abs(linear).max(), abs(quadratic).max()):
        raise ValueError('Non-Hermitian contracted rotation coefficients')
    rotation = angular_momentum_operators(J, representation)
    h, selected = _assemble_coupled_hamiltonian(
        e, linear, quadratic, rotation, None, backend=backend)
    levels, vectors = eigh(h)
    weights = abs(vectors.reshape(n, 2*J+1, -1))**2
    return dict(energies=levels, vectors=vectors,
                vibrational_parent_weights=weights.sum(axis=1),
                eigen_residual_norm=np.linalg.norm(h@vectors-vectors*levels, axis=0),
                assembly_backend=selected, coefficient_hermiticity_error=error)
