"""Product-state ablation of any finite tensor-product Hamiltonian.

This is a VSCF diagnostic on an already defined Hamiltonian. It does not
replace a rotor by a harmonic oscillator or acquire electronic points.
"""
from dataclasses import dataclass
from math import prod
import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class ProductMeanField:
    energy: float
    modal_energies: tuple[np.ndarray, ...]
    modal_vectors: tuple[np.ndarray, ...]
    iterations: int
    stationary_residual: float
    energy_history: tuple[float, ...]

    def product_spectrum(self):
        """Return sorted independent-modal energies and product eigenvectors.

        Excited energies are a frozen mean-field approximation; they are not
        separately optimized excited VSCF states. The ground energy includes
        the mean-field double-counting correction.
        """
        energies = np.zeros(tuple(len(e) for e in self.modal_energies))
        vectors = np.ones((1, 1), dtype=self.modal_vectors[0].dtype)
        for axis, (levels, modes) in enumerate(zip(self.modal_energies, self.modal_vectors)):
            shape = [1] * len(self.modal_energies)
            shape[axis] = len(levels)
            energies += (levels - levels[0]).reshape(shape)
            vectors = np.kron(vectors, modes)
        indices = np.argsort(energies.ravel(), kind='stable')
        return energies.ravel()[indices] + self.energy, vectors[:, indices]


def solve_product_mean_field(hamiltonian, basis_counts, *, initial_modals=None,
                             energy_tolerance=1e-9, residual_tolerance=1e-8,
                             maximum_iterations=200):
    """Optimize a product ground state by alternating exact modal solves.

    Input ordering is the C-order tensor product. Multiple initial guesses
    should be compared if symmetry breaking admits multiple stationary states.
    Tolerances use the energy units of the supplied Hermitian Hamiltonian.
    Failure to reach both stationarity and energy tolerances raises an error.
    """
    counts = tuple(basis_counts)
    if (not counts or len(counts) > 25 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
            or n < 1 for n in counts) or not isinstance(maximum_iterations, int)
            or isinstance(maximum_iterations, bool) or maximum_iterations < 1
            or not np.isfinite(energy_tolerance) or energy_tolerance <= 0
            or not np.isfinite(residual_tolerance) or residual_tolerance <= 0):
        raise ValueError('Positive tensor dimensions, tolerances and iteration count required')
    h = np.asarray(hamiltonian)
    if (h.shape != (prod(counts),) * 2 or not np.all(np.isfinite(h))
            or not np.allclose(h, h.conj().T, rtol=1e-12, atol=1e-12)):
        raise ValueError('A finite Hermitian tensor-product Hamiltonian is required')
    if initial_modals is None:
        modals = [np.eye(n, dtype=h.dtype)[:, 0] for n in counts]
    else:
        if len(initial_modals) != len(counts):
            raise ValueError('One initial modal per tensor axis required')
        modals = []
        for n, vector in zip(counts, initial_modals):
            v = np.asarray(vector, dtype=np.result_type(h.dtype, float))
            if v.shape != (n,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
                raise ValueError('Finite nonzero matching initial modals required')
            modals.append(v / np.linalg.norm(v))
    tensor = h.reshape(counts + counts)
    dimensions = len(counts)

    def effective(axis):
        args = [tensor, list(range(2 * dimensions))]
        for other in range(dimensions):
            if other != axis:
                args.extend((modals[other].conj(), [other], modals[other], [other + dimensions]))
        args.append([axis, axis + dimensions])
        return np.einsum(*args, optimize=True)

    history = []
    for iteration in range(1, maximum_iterations + 1):
        for axis in range(dimensions):
            _, vectors = eigh(effective(axis), subset_by_index=[0, 0])
            modals[axis] = vectors[:, 0]
        matrices = [effective(axis) for axis in range(dimensions)]
        energy = float(np.vdot(modals[0], matrices[0] @ modals[0]).real)
        residual = max(float(np.linalg.norm(matrix @ v - energy * v))
                       for matrix, v in zip(matrices, modals))
        history.append(energy)
        if len(history) > 1 and abs(history[-1] - history[-2]) <= energy_tolerance and residual <= residual_tolerance:
            levels, modes = zip(*(eigh(matrix) for matrix in matrices))
            return ProductMeanField(energy, tuple(levels), tuple(modes), iteration, residual, tuple(history))
    raise RuntimeError(f'Product mean field did not converge: residual={residual:g}')
