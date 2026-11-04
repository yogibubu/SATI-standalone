"""Moving transverse space and kinetic metric for a constrained torsion.

The torsional coordinate is the physical dihedral.  At each constrained
minimum the bath spans the rigid-free tangent space of its level surface.
Keeping the mixed metric is essential because the relaxed path tangent need
not be orthogonal to that surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import factorial

import numpy as np
from scipy.linalg import null_space, solve_triangular
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import eval_hermite

from matrix_gf.harmonic import CM_PER_HARTREE
from .torsional_dvr import AMU_TO_ELECTRON_MASS, _spectral_derivative_grid


@dataclass(frozen=True)
class TorsionalBathFrame:
    """Mass-weighted, rigid-free transverse frame at one scan point.

    ``constraint_normal`` is the unit vector removed from the vibrational
    space: either the dihedral gradient or the relaxed-path tangent.
    """

    basis: np.ndarray
    constraint_normal: np.ndarray


@dataclass(frozen=True)
class TorsionalBathMetric:
    """Metric at the origin of the moving transverse coordinates."""

    path_inertia_amu_bohr2: float
    normal_inertia_amu_bohr2: float
    mixed_amu_sqrt_bohr: np.ndarray
    covariant: np.ndarray
    contravariant: np.ndarray


@dataclass(frozen=True)
class TorsionOneBathResult:
    """Variational spectrum of a periodic torsion and one transverse mode."""

    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    transition_dipoles_au: np.ndarray | None
    ir_intensities_km_mol: np.ndarray | None
    overlap_condition: float
    spectator_transition_corrections_au: np.ndarray | None = None


def solve_torsion_many_bath(
    potential_cm: np.ndarray,
    metric_covariant: np.ndarray,
    *,
    oscillator_frequencies_cm: tuple[float, ...],
    quadrature_count: int,
    occupation_count: int,
    dipole_au: np.ndarray | None = None,
    spectator_kinetic_response: dict[str, np.ndarray] | None = None,
    level_count: int = 6,
) -> TorsionOneBathResult:
    """Matrix-free Galerkin operator for a moderate moving bath block.

    All metric cross terms and the metric volume element are retained.  The
    tensor grid is intended for a few explicit channels; high-dimensional
    baths need a contracted basis rather than this tensor construction.
    ``spectator_kinetic_response`` contracts local spectator frequencies,
    path--spectator metric coefficients, and dipole gradients into the
    leading transition-frequency response. It uses the full inverse-metric
    torsional row of the explicit block and applies only below spectator
    poles; resonant states need an explicit sector.
    """
    potential = np.asarray(potential_cm, dtype=float)
    metric = np.asarray(metric_covariant, dtype=float)
    if spectator_kinetic_response is not None and dipole_au is None:
        raise ValueError("spectator dipole response requires a dipole surface")
    n_bath = len(oscillator_frequencies_cm)
    expected_shape = (quadrature_count,) * n_bath
    if (n_bath < 1 or potential.ndim != n_bath + 1
            or potential.shape[0] < 8 or potential.shape[0] % 2
            or potential.shape[1:] != expected_shape
            or metric.shape != potential.shape + (n_bath + 1, n_bath + 1)):
        raise ValueError("torsion and bath grids have incompatible shapes")
    if (not np.all(np.isfinite(potential)) or not np.all(np.isfinite(metric))
            or not np.allclose(metric, metric.swapaxes(-1, -2), atol=1e-10)
            or np.any(np.linalg.eigvalsh(metric) <= 0)):
        raise ValueError("finite potential and positive symmetric metric are required")
    if (min(oscillator_frequencies_cm) <= 0
            or not 1 <= occupation_count <= quadrature_count
            or not 1 <= level_count < potential.shape[0] * occupation_count**n_bath - 1):
        raise ValueError("invalid oscillator basis or level count")
    count = potential.shape[0]
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_count)
    values = []
    slopes = []
    for omega in oscillator_frequencies_cm:
        alpha = AMU_TO_ELECTRON_MASS * omega / CM_PER_HARTREE
        value = np.empty((quadrature_count, occupation_count))
        slope = np.empty_like(value)
        for n in range(occupation_count):
            norm = np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi))
            poly = eval_hermite(n, nodes)
            value[:, n] = poly / norm
            lower = np.zeros_like(nodes) if n == 0 else 2 * n * eval_hermite(n - 1, nodes)
            slope[:, n] = np.sqrt(alpha) * (lower - nodes * poly) / norm
        values.append(value)
        slopes.append(slope)
    basis = reduce(np.kron, values)
    bath_derivatives = [
        reduce(np.kron, [slopes[j] if j == a else values[j]
                         for j in range(n_bath)]) for a in range(n_bath)
    ]
    bath_grid, block_size = basis.shape
    _, derivative_phi = _spectral_derivative_grid(count)
    inverse = np.linalg.inv(metric).reshape(count, bath_grid, n_bath + 1, n_bath + 1)
    volume = np.sqrt(np.linalg.det(metric)).reshape(count, bath_grid)
    gauss_weight = reduce(np.kron, [weights] * n_bath)
    weight = volume * gauss_weight[None, :] / count
    overlap_blocks = np.einsum("qm,nq,qr->nmr", basis, weight, basis, optimize=True)
    whitener = np.empty_like(overlap_blocks)
    smallest, largest = float("inf"), 0.0
    for j in range(count):
        eigenvalues = np.linalg.eigvalsh(overlap_blocks[j])
        smallest = min(smallest, float(eigenvalues[0]))
        largest = max(largest, float(eigenvalues[-1]))
        factor = np.linalg.cholesky(overlap_blocks[j])
        whitener[j] = solve_triangular(factor.T, np.eye(block_size), lower=False)
    kinetic_scale = CM_PER_HARTREE / (2.0 * AMU_TO_ELECTRON_MASS)
    flattened_potential = potential.reshape(count, bath_grid)

    def coefficient_matvec(flat: np.ndarray) -> np.ndarray:
        coefficients = flat.reshape(count, block_size)
        angular = derivative_phi @ coefficients
        amplitudes = coefficients @ basis.T
        derivatives = np.empty((n_bath + 1, count, bath_grid), dtype=complex)
        derivatives[0] = angular @ basis.T
        for a in range(n_bath):
            derivatives[a + 1] = coefficients @ bath_derivatives[a].T
        flux = weight[None, ...] * np.einsum(
            "nqab,bnq->anq", inverse, derivatives, optimize=True)
        result = ((weight * flattened_potential * amplitudes) @ basis
                  + kinetic_scale * (derivative_phi.conj().T @ (flux[0] @ basis)))
        for a in range(n_bath):
            result += kinetic_scale * flux[a + 1] @ bath_derivatives[a]
        return result.ravel()

    def whitened_matvec(flat: np.ndarray) -> np.ndarray:
        trial = flat.reshape(count, block_size)
        coefficients = np.einsum("nij,nj->ni", whitener, trial, optimize=True)
        result = coefficient_matvec(coefficients.ravel()).reshape(count, block_size)
        return np.einsum("nji,nj->ni", whitener, result, optimize=True).ravel()

    dimension = count * block_size
    operator = LinearOperator((dimension, dimension), matvec=whitened_matvec,
                              dtype=complex)
    if dimension <= 256:
        dense = operator.matmat(np.eye(dimension, dtype=complex))
        energies, orthonormal_vectors = eigh(
            0.5 * (dense + dense.conj().T), subset_by_index=(0, level_count - 1))
    else:
        solve_count = min(max(2 * level_count, level_count + 4), dimension - 2)
        energies, orthonormal_vectors = eigsh(
            operator, k=solve_count, which="SA", tol=1e-9)
    order = np.argsort(energies)[:level_count]
    energies = energies[order]
    orthonormal_vectors = orthonormal_vectors[:, order]
    coefficients = np.einsum(
        "nij,njk->nik", whitener,
        orthonormal_vectors.reshape(count, block_size, level_count), optimize=True)
    vectors = coefficients.reshape(dimension, level_count)
    transition = None
    intensities = None
    spectator_correction = None
    if dipole_au is not None:
        dipole = np.asarray(dipole_au, dtype=float)
        if dipole.shape != potential.shape + (3,) or not np.all(np.isfinite(dipole)):
            raise ValueError("dipole must match torsion and bath grids")
        amplitudes = np.einsum("qm,nmk->nqk", basis, coefficients, optimize=True)
        transition = np.einsum(
            "nq,nq,nqa,nqk->ka", weight, amplitudes[:, :, 0].conj(),
            dipole.reshape(count, bath_grid, 3), amplitudes, optimize=True)
        if spectator_kinetic_response is not None:
            frequencies = np.asarray(
                spectator_kinetic_response["frequencies_cm"], dtype=float)
            mixed = np.asarray(
                spectator_kinetic_response["mixed_metric"], dtype=float)
            dipole_gradient = np.asarray(
                spectator_kinetic_response["dipole_gradient"], dtype=float)
            if (frequencies.ndim != 2 or frequencies.shape[0] != count
                    or mixed.shape != frequencies.shape
                    or dipole_gradient.shape != frequencies.shape + (3,)
                    or not np.all(np.isfinite(frequencies))
                    or not np.all(np.isfinite(mixed))
                    or not np.all(np.isfinite(dipole_gradient))):
                raise ValueError("invalid spectator response arrays")
            derivative_amplitudes = [np.einsum(
                "qm,nmk->nqk", basis,
                np.einsum("nl,lmk->nmk", derivative_phi, coefficients),
                optimize=True)]
            derivative_amplitudes.extend(
                np.einsum("qm,nmk->nqk", derivative, coefficients,
                          optimize=True)
                for derivative in bath_derivatives)
            spectator_correction = np.zeros_like(transition)
            for state in range(1, level_count):
                gap = float(energies[state] - energies[0])
                if np.any(frequencies <= gap):
                    # A targeted resonant sector is required for this state.
                    # Leave its transition uncorrected; the low first band
                    # remains below every spectator pole.
                    continue
                coefficient = np.einsum(
                    "na,nad->nd", mixed / (frequencies**2 - gap**2),
                    dipole_gradient, optimize=True)
                current = sum(
                    inverse[:, :, 0, axis] * (
                        amplitudes[:, :, 0].conj()
                        * derivative_amplitudes[axis][:, :, state]
                        - derivative_amplitudes[axis][:, :, 0].conj()
                        * amplitudes[:, :, state])
                    for axis in range(n_bath + 1))
                spectator_correction[state] = (
                    kinetic_scale * gap * np.einsum(
                        "nq,nq,nd->d", weight, current, coefficient,
                        optimize=True))
            transition = transition + spectator_correction
        from .gvpt2_spectroscopy import IR_TRANSITION_FACTOR
        intensities = (IR_TRANSITION_FACTOR * np.maximum(energies - energies[0], 0.0)
                       * np.sum(abs(transition)**2, axis=1))
    return TorsionOneBathResult(
        energies_cm=energies, eigenvectors=vectors,
        transition_dipoles_au=transition, ir_intensities_km_mol=intensities,
        overlap_condition=largest / smallest,
        spectator_transition_corrections_au=spectator_correction)


def solve_torsion_three_bath(
    potential_cm: np.ndarray,
    metric_covariant: np.ndarray,
    *,
    oscillator_frequencies_cm: tuple[float, float, float],
    quadrature_count: int,
    occupation_count: int,
    dipole_au: np.ndarray | None = None,
    level_count: int = 6,
) -> TorsionOneBathResult:
    """Matrix-free Galerkin solver for a torsion and three moving bath modes.

    The full four-coordinate metric, its volume element, and all cross terms
    are sampled on a Fourier times tensor Gauss--Hermite grid. Local overlap
    blocks are Cholesky whitened before iterative diagonalization.
    """
    potential = np.asarray(potential_cm, dtype=float)
    metric = np.asarray(metric_covariant, dtype=float)
    if (potential.ndim != 4 or potential.shape[0] < 8 or potential.shape[0] % 2
            or potential.shape[1:] != (quadrature_count,) * 3
            or metric.shape != potential.shape + (4, 4)):
        raise ValueError("torsion and three bath grids have incompatible shapes")
    if (not np.all(np.isfinite(potential)) or not np.all(np.isfinite(metric))
            or not np.allclose(metric, metric.swapaxes(-1, -2), atol=1e-10)
            or np.any(np.linalg.eigvalsh(metric) <= 0)):
        raise ValueError("finite potential and positive symmetric metric are required")
    if (len(oscillator_frequencies_cm) != 3
            or min(oscillator_frequencies_cm) <= 0
            or not 1 <= occupation_count <= quadrature_count
            or not 1 <= level_count < potential.shape[0] * occupation_count**3 - 1):
        raise ValueError("invalid oscillator basis or level count")
    count = potential.shape[0]
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_count)
    values = []
    slopes = []
    for omega in oscillator_frequencies_cm:
        alpha = AMU_TO_ELECTRON_MASS * omega / CM_PER_HARTREE
        value = np.empty((quadrature_count, occupation_count))
        slope = np.empty_like(value)
        for n in range(occupation_count):
            norm = np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi))
            poly = eval_hermite(n, nodes)
            value[:, n] = poly / norm
            lower = np.zeros_like(nodes) if n == 0 else 2 * n * eval_hermite(n - 1, nodes)
            slope[:, n] = np.sqrt(alpha) * (lower - nodes * poly) / norm
        values.append(value)
        slopes.append(slope)
    basis = np.kron(np.kron(values[0], values[1]), values[2])
    bath_derivatives = [
        np.kron(np.kron(slopes[0], values[1]), values[2]),
        np.kron(np.kron(values[0], slopes[1]), values[2]),
        np.kron(np.kron(values[0], values[1]), slopes[2]),
    ]
    bath_grid, block_size = basis.shape
    _, derivative_phi = _spectral_derivative_grid(count)
    inverse = np.linalg.inv(metric).reshape(count, bath_grid, 4, 4)
    volume = np.sqrt(np.linalg.det(metric)).reshape(count, bath_grid)
    gauss_weight = np.kron(np.kron(weights, weights), weights)
    weight = volume * gauss_weight[None, :] / count
    overlap_blocks = np.einsum("qm,nq,qr->nmr", basis, weight, basis, optimize=True)
    whitener = np.empty_like(overlap_blocks)
    smallest, largest = float("inf"), 0.0
    for j in range(count):
        eigenvalues = np.linalg.eigvalsh(overlap_blocks[j])
        smallest = min(smallest, float(eigenvalues[0]))
        largest = max(largest, float(eigenvalues[-1]))
        factor = np.linalg.cholesky(overlap_blocks[j])
        whitener[j] = solve_triangular(
            factor.T, np.eye(block_size), lower=False,
        )
    kinetic_scale = CM_PER_HARTREE / (2.0 * AMU_TO_ELECTRON_MASS)
    flattened_potential = potential.reshape(count, bath_grid)

    def coefficient_matvec(flat: np.ndarray) -> np.ndarray:
        coefficients = flat.reshape(count, block_size)
        angular = derivative_phi @ coefficients
        amplitudes = coefficients @ basis.T
        derivatives = np.empty((4, count, bath_grid), dtype=complex)
        derivatives[0] = angular @ basis.T
        for a in range(3):
            derivatives[a + 1] = coefficients @ bath_derivatives[a].T
        flux = weight[None, ...] * np.einsum(
            "nqab,bnq->anq", inverse, derivatives, optimize=True,
        )
        result = ((weight * flattened_potential * amplitudes) @ basis
                  + kinetic_scale * (derivative_phi.conj().T @ (flux[0] @ basis)))
        for a in range(3):
            result += kinetic_scale * flux[a + 1] @ bath_derivatives[a]
        return result.ravel()

    def whitened_matvec(flat: np.ndarray) -> np.ndarray:
        trial = flat.reshape(count, block_size)
        coefficients = np.einsum("nij,nj->ni", whitener, trial, optimize=True)
        result = coefficient_matvec(coefficients.ravel()).reshape(count, block_size)
        return np.einsum("nji,nj->ni", whitener, result, optimize=True).ravel()

    dimension = count * block_size
    operator = LinearOperator((dimension, dimension), matvec=whitened_matvec,
                              dtype=complex)
    if dimension <= 256:
        dense = operator.matmat(np.eye(dimension, dtype=complex))
        energies, orthonormal_vectors = eigh(
            0.5 * (dense + dense.conj().T),
            subset_by_index=(0, level_count - 1),
        )
    else:
        solve_count = min(max(2 * level_count, level_count + 4), dimension - 2)
        energies, orthonormal_vectors = eigsh(
            operator, k=solve_count, which="SA", tol=1e-9,
        )
    order = np.argsort(energies)[:level_count]
    energies = energies[order]
    orthonormal_vectors = orthonormal_vectors[:, order]
    coefficients = np.einsum(
        "nij,njk->nik", whitener,
        orthonormal_vectors.reshape(count, block_size, level_count),
        optimize=True,
    )
    vectors = coefficients.reshape(dimension, level_count)
    transition = None
    intensities = None
    if dipole_au is not None:
        dipole = np.asarray(dipole_au, dtype=float)
        if dipole.shape != potential.shape + (3,) or not np.all(np.isfinite(dipole)):
            raise ValueError("dipole must match torsion and three bath grids")
        amplitudes = np.einsum("qm,nmk->nqk", basis, coefficients, optimize=True)
        transition = np.einsum(
            "nq,nq,nqa,nqk->ka", weight, amplitudes[:, :, 0].conj(),
            dipole.reshape(count, bath_grid, 3), amplitudes, optimize=True,
        )
        from .gvpt2_spectroscopy import IR_TRANSITION_FACTOR
        intensities = (IR_TRANSITION_FACTOR * np.maximum(energies - energies[0], 0.0)
                       * np.sum(abs(transition)**2, axis=1))
    return TorsionOneBathResult(
        energies_cm=energies,
        eigenvectors=vectors,
        transition_dipoles_au=transition,
        ir_intensities_km_mol=intensities,
        overlap_condition=largest / smallest,
    )


def solve_torsion_two_bath(
    potential_cm: np.ndarray,
    metric_covariant: np.ndarray,
    *,
    oscillator_frequencies_cm: tuple[float, float],
    quadrature_count: int,
    occupation_count: int,
    dipole_au: np.ndarray | None = None,
    level_count: int = 8,
) -> TorsionOneBathResult:
    """Variational torsion and two moving bath coordinates, including their metric.

    A sparse Galerkin form keeps all three inverse-metric cross terms and the
    geometry-dependent volume element. The supplied potential and metric must
    use the product Gauss--Hermite grid in row-major bath order.
    """
    potential = np.asarray(potential_cm, dtype=float)
    metric = np.asarray(metric_covariant, dtype=float)
    if (potential.ndim != 3 or potential.shape[0] < 8 or potential.shape[0] % 2
            or potential.shape[1:] != (quadrature_count, quadrature_count)
            or metric.shape != potential.shape + (3, 3)):
        raise ValueError("torsion and two bath grids have incompatible shapes")
    if (not np.all(np.isfinite(potential)) or not np.all(np.isfinite(metric))
            or not np.allclose(metric, metric.swapaxes(-1, -2), atol=1e-10)
            or np.any(np.linalg.eigvalsh(metric) <= 0)):
        raise ValueError("finite potential and positive symmetric metric are required")
    if (len(oscillator_frequencies_cm) != 2
            or min(oscillator_frequencies_cm) <= 0
            or not 1 <= occupation_count <= quadrature_count
            or not 1 <= level_count < potential.shape[0] * occupation_count**2):
        raise ValueError("invalid oscillator basis or level count")
    count = potential.shape[0]
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_count)
    basis = []
    derivative = []
    for omega in oscillator_frequencies_cm:
        alpha = AMU_TO_ELECTRON_MASS * omega / CM_PER_HARTREE
        values = np.empty((quadrature_count, occupation_count))
        slopes = np.empty_like(values)
        for n in range(occupation_count):
            norm = np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi))
            poly = eval_hermite(n, nodes)
            values[:, n] = poly / norm
            lower = np.zeros_like(nodes) if n == 0 else 2 * n * eval_hermite(n - 1, nodes)
            slopes[:, n] = np.sqrt(alpha) * (lower - nodes * poly) / norm
        basis.append(csr_matrix(values))
        derivative.append(csr_matrix(slopes))
    bath = kron(basis[0], basis[1], format="csr")
    bath_derivatives = (
        kron(derivative[0], basis[1], format="csr"),
        kron(basis[0], derivative[1], format="csr"),
    )
    identity = eye(count, format="csr")
    _, spectral_derivative = _spectral_derivative_grid(count)
    designs = [kron(identity, bath, format="csr")]
    derivatives = [kron(csr_matrix(spectral_derivative), bath, format="csr")]
    derivatives.extend(kron(identity, item, format="csr") for item in bath_derivatives)
    inverse = np.linalg.inv(metric)
    weight = (weights[None, :, None] * weights[None, None, :]
              * np.sqrt(np.linalg.det(metric)) / count).ravel()
    amplitude = designs[0]
    overlap = (amplitude.T @ diags(weight) @ amplitude).tocsr()
    hamiltonian = amplitude.T @ diags(weight * potential.ravel()) @ amplitude
    kinetic_scale = CM_PER_HARTREE / (2.0 * AMU_TO_ELECTRON_MASS)
    for a in range(3):
        for b in range(3):
            hamiltonian += (kinetic_scale * derivatives[a].conj().T
                            @ diags(weight * inverse[..., a, b].ravel())
                            @ derivatives[b])
    hamiltonian = ((hamiltonian + hamiltonian.conj().T) * 0.5).tocsr()
    # ARPACK can return only one member of a nearly degenerate complex
    # conjugate pair when k stops at that cluster. Oversample before slicing.
    if level_count >= hamiltonian.shape[0] - 1:
        energies, vectors = eigh(
            hamiltonian.toarray(), overlap.toarray(),
            subset_by_index=(0, level_count - 1),
        )
    else:
        solve_count = min(max(2 * level_count, level_count + 4),
                          hamiltonian.shape[0] - 2)
        energies, vectors = eigsh(hamiltonian, k=solve_count, M=overlap,
                                  which="SA", tol=1e-10)
    order = np.argsort(energies)
    energies = energies[order[:level_count]]
    vectors = vectors[:, order[:level_count]]
    transition = None
    intensities = None
    if dipole_au is not None:
        dipole = np.asarray(dipole_au, dtype=float)
        if dipole.shape != potential.shape + (3,) or not np.all(np.isfinite(dipole)):
            raise ValueError("dipole must match torsion and two bath grids")
        states = amplitude @ vectors
        transition = np.einsum(
            "i,ia,ij->ja", states[:, 0].conj() * weight,
            dipole.reshape(-1, 3), states, optimize=True,
        )
        from .gvpt2_spectroscopy import IR_TRANSITION_FACTOR
        intensities = (IR_TRANSITION_FACTOR * np.maximum(energies - energies[0], 0.0)
                       * np.sum(abs(transition)**2, axis=1))
    block_size = occupation_count**2
    block_eigenvalues = [
        np.linalg.eigvalsh(overlap[start:start + block_size,
                                   start:start + block_size].toarray())
        for start in range(0, overlap.shape[0], block_size)
    ]
    return TorsionOneBathResult(
        energies_cm=energies,
        eigenvectors=vectors,
        transition_dipoles_au=transition,
        ir_intensities_km_mol=intensities,
        overlap_condition=float(max(values[-1] for values in block_eigenvalues)
                                / min(values[0] for values in block_eigenvalues)),
    )


def constrained_bath_frame(
    xyz_bohr: np.ndarray,
    masses_amu: np.ndarray,
    dihedral_gradient_per_bohr: np.ndarray,
) -> TorsionalBathFrame:
    """Build the 3N-7 bath perpendicular to the dihedral gradient.

    The returned vectors are orthonormal in mass-weighted Cartesian space.
    Rigid translations and rotations are removed before the constraint.
    """
    xyz = np.asarray(xyz_bohr, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    gradient = np.asarray(dihedral_gradient_per_bohr, dtype=float).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or masses.shape != (len(xyz),):
        raise ValueError("coordinates and masses have incompatible shapes")
    if np.any(masses <= 0) or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(masses)):
        raise ValueError("finite coordinates and positive masses are required")
    if gradient.shape != (3 * len(xyz),) or not np.all(np.isfinite(gradient)):
        raise ValueError("dihedral gradient has invalid shape or values")
    centered = xyz - np.average(xyz, axis=0, weights=masses)
    roots = np.sqrt(masses)[:, None]
    rigid = np.column_stack(
        [(roots * np.tile(axis, (len(xyz), 1))).ravel() for axis in np.eye(3)]
        + [(roots * np.cross(axis, centered)).ravel() for axis in np.eye(3)]
    )
    vibration = null_space(rigid.T)
    weighted_gradient = gradient / np.repeat(np.sqrt(masses), 3)
    normal = vibration @ (vibration.T @ weighted_gradient)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ValueError("dihedral gradient vanishes in the vibrational space")
    normal /= length
    bath = vibration @ null_space((vibration.T @ normal)[None, :])
    return TorsionalBathFrame(basis=bath, constraint_normal=normal)


def path_orthogonal_bath_frame(
    xyz_bohr: np.ndarray,
    masses_amu: np.ndarray,
    path_derivative_bohr_per_rad: np.ndarray,
) -> TorsionalBathFrame:
    """Build a bath orthogonal to the relaxed path rather than the constraint.

    In this chart the mixed metric vanishes at Q=0, but the potential has
    generally nonzero linear terms in the bath coordinates. Those terms must
    be retained in any vibrational Hamiltonian built on this frame.
    """
    xyz = np.asarray(xyz_bohr, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    tangent = np.asarray(path_derivative_bohr_per_rad, dtype=float).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or masses.shape != (len(xyz),):
        raise ValueError("coordinates and masses have incompatible shapes")
    if tangent.shape != (3 * len(xyz),) or not np.all(np.isfinite(tangent)):
        raise ValueError("path derivative has invalid shape or values")
    centered = xyz - np.average(xyz, axis=0, weights=masses)
    roots = np.sqrt(masses)[:, None]
    rigid = np.column_stack(
        [(roots * np.tile(axis, (len(xyz), 1))).ravel() for axis in np.eye(3)]
        + [(roots * np.cross(axis, centered)).ravel() for axis in np.eye(3)]
    )
    vibration = null_space(rigid.T)
    weighted = tangent * np.repeat(np.sqrt(masses), 3)
    normal = vibration @ (vibration.T @ weighted)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ValueError("relaxed path has a vanishing vibrational tangent")
    normal /= length
    bath = vibration @ null_space((vibration.T @ normal)[None, :])
    return TorsionalBathFrame(basis=bath, constraint_normal=normal)


def constraint_preserving_second_derivative(
    masses_amu: np.ndarray,
    constraint_gradient: np.ndarray,
    constraint_hessian: np.ndarray,
    transverse_cartesian: np.ndarray,
) -> np.ndarray:
    """Return Cartesian x''(q) that holds an internal coordinate fixed to O(q²).

    For x(q)=x0+u*q+x''*q²/2, the condition is
    b*x''+u*B'*u=0. The minimal mass-weighted correction lies along M^-1 b.
    This also makes the constrained Lagrangian Hessian the quadratic
    potential along the chart when the reference is constraint stationary.
    """
    masses = np.asarray(masses_amu, dtype=float)
    gradient = np.asarray(constraint_gradient, dtype=float).reshape(-1)
    hessian = np.asarray(constraint_hessian, dtype=float)
    displacement = np.asarray(transverse_cartesian, dtype=float).reshape(-1)
    mass = np.repeat(masses, 3)
    if (gradient.shape != mass.shape or displacement.shape != mass.shape
            or hessian.shape != (len(mass), len(mass))
            or np.any(mass <= 0) or not np.all(np.isfinite(mass))
            or not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(hessian))
            or not np.all(np.isfinite(displacement))):
        raise ValueError("constraint chart tensors have invalid shapes or values")
    if abs(gradient @ displacement) > 1e-8 * max(1.0, np.linalg.norm(gradient)):
        raise ValueError("transverse displacement is not tangent to the constraint")
    wilson = float(np.dot(gradient / mass, gradient))
    if wilson <= 0:
        raise ValueError("constraint has vanishing Wilson metric")
    return -(displacement @ hessian @ displacement) / wilson * gradient / mass


def transport_bath_frame(
    previous: TorsionalBathFrame, current: TorsionalBathFrame,
) -> TorsionalBathFrame:
    """Choose the closest orthogonal gauge for consecutive bath subspaces."""
    if previous.basis.shape != current.basis.shape:
        raise ValueError("bath frames must have equal dimensions")
    left, _, right = np.linalg.svd(current.basis.T @ previous.basis)
    return TorsionalBathFrame(
        basis=current.basis @ left @ right,
        constraint_normal=current.constraint_normal,
    )


def torsional_bath_metric(
    frame: TorsionalBathFrame,
    path_derivative_bohr_per_rad: np.ndarray,
    masses_amu: np.ndarray,
    dihedral_gradient_per_bohr: np.ndarray,
) -> TorsionalBathMetric:
    """Return the full metric at Q=0 for (dihedral, moving bath Q).

    The path derivative must satisfy d(phi)/d(phi)=1.  In particular, the
    Schur complement is fixed by the Wilson metric of the dihedral gradient;
    the path inertia alone is not the coupled-system kinetic coefficient.
    """
    masses = np.asarray(masses_amu, dtype=float)
    tangent = np.asarray(path_derivative_bohr_per_rad, dtype=float).reshape(-1)
    gradient = np.asarray(dihedral_gradient_per_bohr, dtype=float).reshape(-1)
    if tangent.shape != gradient.shape or tangent.shape != (3 * len(masses),):
        raise ValueError("path derivative and dihedral gradient have invalid shapes")
    if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(gradient)):
        raise ValueError("path derivative and dihedral gradient must be finite")
    if not np.isclose(gradient @ tangent, 1.0, rtol=0.02, atol=0.02):
        raise ValueError("path derivative is not parametrized by the dihedral")
    weighted = tangent * np.repeat(np.sqrt(masses), 3)
    mixed = frame.basis.T @ weighted
    path_inertia = float(weighted @ weighted)
    normal_inertia = float(path_inertia - mixed @ mixed)
    if normal_inertia <= 0:
        raise ValueError("torsion/bath metric is not positive definite")
    covariant = np.eye(1 + len(mixed))
    covariant[0, 0] = path_inertia
    covariant[0, 1:] = mixed
    covariant[1:, 0] = mixed
    contravariant = np.eye(1 + len(mixed))
    contravariant[0, 0] = 1.0 / normal_inertia
    contravariant[0, 1:] = -mixed / normal_inertia
    contravariant[1:, 0] = -mixed / normal_inertia
    contravariant[1:, 1:] += np.outer(mixed, mixed) / normal_inertia
    return TorsionalBathMetric(
        path_inertia_amu_bohr2=path_inertia,
        normal_inertia_amu_bohr2=normal_inertia,
        mixed_amu_sqrt_bohr=mixed,
        covariant=covariant,
        contravariant=contravariant,
    )


def solve_torsion_one_bath(
    potential_cm: np.ndarray,
    metric_covariant: np.ndarray,
    *,
    oscillator_frequency_cm: float,
    quadrature_count: int = 16,
    occupation_count: int = 5,
    dipole_au: np.ndarray | None = None,
    level_count: int = 8,
) -> TorsionOneBathResult:
    """Solve a two-coordinate Laplace--Beltrami Hamiltonian by Galerkin DVR.

    The first coordinate is periodic and sampled at 2πj/N. The second is a
    mass-weighted Cartesian bath displacement in sqrt(amu)*bohr, expanded in
    fixed harmonic oscillator functions. Potential and covariant metric are
    sampled on the Fourier grid times the Gauss--Hermite grid. The kinetic
    form retains both mixed inverse-metric components and the volume element.
    No assumption about a molecule or a particular electronic backend enters.
    """
    potential = np.asarray(potential_cm, dtype=float)
    metric = np.asarray(metric_covariant, dtype=float)
    if potential.ndim != 2 or potential.shape[0] < 8 or potential.shape[0] % 2:
        raise ValueError("periodic torsion needs an even grid with at least 8 points")
    count, quadrature = potential.shape
    if quadrature_count != quadrature or metric.shape != (count, quadrature, 2, 2):
        raise ValueError("potential and metric must use the same two-dimensional grid")
    if (not np.all(np.isfinite(potential)) or not np.all(np.isfinite(metric))
            or not np.allclose(metric, metric.swapaxes(-1, -2), atol=1e-10)):
        raise ValueError("potential and symmetric metric must be finite")
    determinant = np.linalg.det(metric)
    if np.any(determinant <= 0) or np.any(metric[..., 0, 0] <= 0):
        raise ValueError("kinetic metric must be positive definite")
    if (oscillator_frequency_cm <= 0 or not 1 <= occupation_count <= quadrature
            or not 1 <= level_count <= count * occupation_count):
        raise ValueError("invalid oscillator, occupation, or level count")
    inverse = np.linalg.inv(metric)
    root_det = np.sqrt(determinant)
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature)
    alpha = (AMU_TO_ELECTRON_MASS * oscillator_frequency_cm / CM_PER_HARTREE)
    basis = np.empty((quadrature, occupation_count))
    derivative_q = np.empty_like(basis)
    for n in range(occupation_count):
        norm = np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi))
        poly = eval_hermite(n, nodes)
        basis[:, n] = poly / norm
        lower = np.zeros_like(nodes) if n == 0 else 2 * n * eval_hermite(n - 1, nodes)
        derivative_q[:, n] = np.sqrt(alpha) * (lower - nodes * poly) / norm
    _, derivative_phi = _spectral_derivative_grid(count)
    amplitude = np.kron(np.eye(count), basis)
    dphi = np.kron(derivative_phi, basis)
    dq = np.kron(np.eye(count), derivative_q)
    weight = (np.broadcast_to(weights[None, :], (count, quadrature))
              * root_det / count).ravel()
    overlap = amplitude.conj().T @ (weight[:, None] * amplitude)
    kinetic_scale = CM_PER_HARTREE / (2.0 * AMU_TO_ELECTRON_MASS)
    hamiltonian = (amplitude.conj().T
                   @ ((weight * potential.ravel())[:, None] * amplitude)).astype(complex)
    phi_phi = (weight * inverse[..., 0, 0].ravel())[:, None]
    phi_q = (weight * inverse[..., 0, 1].ravel())[:, None]
    q_q = (weight * inverse[..., 1, 1].ravel())[:, None]
    hamiltonian += kinetic_scale * (
        dphi.conj().T @ (phi_phi * dphi)
        + dphi.conj().T @ (phi_q * dq)
        + dq.conj().T @ (phi_q * dphi)
        + dq.conj().T @ (q_q * dq)
    )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    energies, vectors = eigh(hamiltonian, overlap, subset_by_index=(0, level_count - 1))
    transition = None
    intensities = None
    if dipole_au is not None:
        dipole = np.asarray(dipole_au, dtype=float)
        if dipole.shape != (count, quadrature, 3) or not np.all(np.isfinite(dipole)):
            raise ValueError("dipole must have shape (torsion, bath grid, 3)")
        ground = amplitude @ vectors[:, 0]
        states = amplitude @ vectors
        transition = np.einsum(
            "i,ia,ij->ja", ground.conj() * weight, dipole.reshape(-1, 3), states,
            optimize=True,
        )
        from .gvpt2_spectroscopy import IR_TRANSITION_FACTOR
        intensities = (IR_TRANSITION_FACTOR * np.maximum(energies - energies[0], 0.0)
                       * np.sum(abs(transition)**2, axis=1))
    return TorsionOneBathResult(
        energies_cm=energies.real,
        eigenvectors=vectors,
        transition_dipoles_au=transition,
        ir_intensities_km_mol=intensities,
        overlap_condition=float(np.linalg.cond(overlap)),
    )
