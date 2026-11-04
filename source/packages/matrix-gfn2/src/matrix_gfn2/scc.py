"""Monopole self-consistent-charge equations for MATRIX-GFN2."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from matrix_numerics import eigh_arrays

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover - pure-Python fallback
    _gfn2_native = None

from .localized_solver import SparseSymmetricMatrix


BOLTZMANN_HARTREE_PER_KELVIN = 3.166808578545117e-6


@dataclass(frozen=True)
class ShellPartition:
    orbital_shell: tuple[int, ...]
    shell_atom: tuple[int, ...]
    reference_population: tuple[float, ...]
    hardness_hartree: tuple[float, ...]
    third_order_hartree: tuple[float, ...]
    electrostatic_profile: object | None = None

    def __post_init__(self) -> None:
        nshells = len(self.shell_atom)
        if not self.orbital_shell:
            raise ValueError("shell partition requires orbitals")
        if any(shell < 0 or shell >= nshells for shell in self.orbital_shell):
            raise ValueError("orbital_shell contains an invalid shell index")
        for values in (
            self.reference_population,
            self.hardness_hartree,
            self.third_order_hartree,
        ):
            if len(values) != nshells:
                raise ValueError("all shell parameter arrays must have equal length")
        if any(value <= 0.0 for value in self.hardness_hartree):
            raise ValueError("shell hardnesses must be positive")

    @property
    def nshells(self) -> int:
        return len(self.shell_atom)


@dataclass(frozen=True)
class SCCResult:
    energy_hartree: float
    density: np.ndarray
    shell_charges: np.ndarray
    orbital_energies_hartree: np.ndarray
    occupations: np.ndarray
    iterations: int
    converged: bool


@dataclass(frozen=True)
class InverseSqrtResult:
    inverse_sqrt: np.ndarray
    iterations: int
    residual: float
    converged: bool


def shell_charges(
    density: np.ndarray,
    overlap: np.ndarray,
    partition: ShellPartition,
) -> np.ndarray:
    density = np.asarray(density, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    if density.shape != overlap.shape or density.shape != (
        len(partition.orbital_shell),
        len(partition.orbital_shell),
    ):
        raise ValueError("density and overlap must match the orbital partition")
    populations = np.zeros(partition.nshells, dtype=float)
    mulliken = density * overlap
    for orbital, shell in enumerate(partition.orbital_shell):
        populations[shell] += float(np.sum(mulliken[orbital]))
    return np.asarray(partition.reference_population, dtype=float) - populations


def shell_charges_sparse(
    density: SparseSymmetricMatrix,
    overlap: SparseSymmetricMatrix,
    partition: ShellPartition,
) -> np.ndarray:
    if density.size != overlap.size or density.size != len(partition.orbital_shell):
        raise ValueError("sparse density and overlap must match the orbital partition")
    values = density.values_on(overlap.rows, overlap.columns) * overlap.values
    shells = np.asarray(partition.orbital_shell, dtype=np.intp)
    left_shell = shells[overlap.rows]
    right_shell = shells[overlap.columns]
    populations = np.zeros(partition.nshells, dtype=float)
    np.add.at(populations, left_shell, values)
    off = overlap.rows != overlap.columns
    np.add.at(populations, right_shell[off], values[off])
    return np.asarray(partition.reference_population, dtype=float) - populations


def shell_interaction_matrix(
    coordinates_bohr: np.ndarray,
    partition: ShellPartition,
    *, alphaj: float = 2.0,
) -> np.ndarray:
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates_bohr must have shape natoms x 3")
    if not np.isfinite(alphaj) or alphaj <= 0.0:
        raise ValueError("Klopman-Ohno exponent must be positive and finite")
    hardness = np.asarray(partition.hardness_hartree, dtype=float)
    if alphaj == 2.0 and _gfn2_native is not None and hasattr(_gfn2_native, "shell_interaction"):
        matrix = np.asarray(_gfn2_native.shell_interaction(
            coordinates, np.asarray(partition.shell_atom, dtype=np.int64), hardness))
    else:
        shell_coordinates = coordinates[np.asarray(partition.shell_atom, dtype=np.intp)]
        displacement = shell_coordinates[:, None, :] - shell_coordinates[None, :, :]
        distance_squared = np.einsum("ijk,ijk->ij", displacement, displacement)
        average = 0.5 * (hardness[:, None] + hardness[None, :])
        matrix = (1.0 / np.sqrt(distance_squared + average**-2) if alphaj == 2.0
                  else (distance_squared ** (0.5 * alphaj) + average**-alphaj) ** (-1.0 / alphaj))
    profile = partition.electrostatic_profile
    if profile is not None and not profile.is_identity:
        if alphaj != 2.0:
            raise NotImplementedError("Gaussian SCC currently requires the native exponent two")
        return profile.modify_values(matrix, coordinates, partition)
    return matrix



def shell_interaction_operator(
    coordinates_bohr: np.ndarray,
    partition: ShellPartition,
    *,
    # Dense matrix-vector products remain faster through the medium regime;
    # hand over to FMM only once the quadratic storage/work estimate exceeds
    # this crossover.  The crossover is deliberately below the point where
    # the dense shell matrix becomes a memory risk.
    fmm_minimum_shells: int = 4096,
    alphaj: float = 2.0,
):
    """Select direct or matrix-free SCC while preserving the full kernel."""

    if partition.electrostatic_profile is not None and not partition.electrostatic_profile.is_identity:
        if partition.nshells >= int(fmm_minimum_shells):
            raise NotImplementedError("Gaussian SCC is qualified on the dense molecular route")
        return shell_interaction_matrix(coordinates_bohr, partition, alphaj=alphaj)
    if partition.nshells < int(fmm_minimum_shells) or alphaj != 2.0:
        return shell_interaction_matrix(coordinates_bohr, partition, alphaj=alphaj)
    try:
        import fmm3dpy  # noqa: F401
    except ImportError:
        return shell_interaction_matrix(coordinates_bohr, partition)
    from .hierarchical_scc import FMMShellInteraction

    return FMMShellInteraction(
        coordinates_bohr,
        partition.shell_atom,
        partition.hardness_hartree,
        precision=1.0e-10,
        correction_cutoff_bohr=48.0,
        correction_opening_angle=0.05,
        correction_leaf_size=8,
    )


def monopole_energy(
    charges: np.ndarray,
    interaction: np.ndarray,
    partition: ShellPartition,
    *,
    quadratic_energy_override: float | None = None,
) -> float:
    charges = np.asarray(charges, dtype=float)
    third = np.asarray(partition.third_order_hartree, dtype=float)
    if quadratic_energy_override is not None:
        quadratic = float(quadratic_energy_override)
    elif hasattr(interaction, "evaluate"):
        quadratic = float(interaction.evaluate(charges).energy_hartree)
    else:
        quadratic = 0.5 * float(charges @ interaction @ charges)
    return quadratic + float(np.sum(third * charges**3) / 3.0)


def monopole_fock(
    overlap: np.ndarray,
    charges: np.ndarray,
    interaction: np.ndarray,
    partition: ShellPartition,
) -> np.ndarray:
    overlap = np.asarray(overlap, dtype=float)
    charges = np.asarray(charges, dtype=float)
    potential = interaction @ charges + np.asarray(
        partition.third_order_hartree, dtype=float
    ) * charges**2
    orbital_potential = potential[np.asarray(partition.orbital_shell, dtype=np.intp)]
    return -0.5 * overlap * (
        orbital_potential[:, np.newaxis] + orbital_potential[np.newaxis, :]
    )


def monopole_fock_sparse(
    overlap: SparseSymmetricMatrix,
    charges: np.ndarray,
    interaction: np.ndarray,
    partition: ShellPartition,
    *,
    potential_override: np.ndarray | None = None,
) -> SparseSymmetricMatrix:
    charges = np.asarray(charges, dtype=float)
    quadratic_potential = (
        interaction @ charges
        if potential_override is None
        else np.asarray(potential_override, dtype=float)
    )
    potential = quadratic_potential + np.asarray(
        partition.third_order_hartree, dtype=float
    ) * charges**2
    orbital_potential = potential[np.asarray(partition.orbital_shell, dtype=np.intp)]
    values = -0.5 * overlap.values * (
        orbital_potential[overlap.rows] + orbital_potential[overlap.columns]
    )
    return SparseSymmetricMatrix(
        overlap.size, overlap.rows.copy(), overlap.columns.copy(), values
    )


def fermi_occupations(
    orbital_energies_hartree: np.ndarray,
    electrons: int,
    *,
    multiplicity: int = 1,
    temperature_kelvin: float = 300.0,
) -> np.ndarray:
    alpha, beta = spin_fermi_occupations(
        orbital_energies_hartree,
        electrons,
        multiplicity=multiplicity,
        temperature_kelvin=temperature_kelvin,
    )
    return alpha + beta


def spin_fermi_occupations(
    orbital_energies_hartree: np.ndarray,
    electrons: int,
    *,
    multiplicity: int = 1,
    temperature_kelvin: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate alpha and beta Fermi occupations.

    Keeping the channels separate is essential for the GFN2 entropy and for
    Mayer bond orders of restricted open-shell states, even though both spin
    channels share the same spatial orbitals in the spin-independent model.
    """

    energies = np.asarray(orbital_energies_hartree, dtype=float)
    unpaired = int(multiplicity) - 1
    if electrons < 0 or unpaired < 0 or (electrons - unpaired) % 2:
        raise ValueError("electron count and multiplicity are inconsistent")
    alpha = (int(electrons) + unpaired) // 2
    beta = int(electrons) - alpha
    if alpha > len(energies):
        raise ValueError("not enough orbitals for requested electronic state")
    thermal = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature_kelvin)
    if thermal <= 0.0:
        alpha_occ = np.zeros_like(energies)
        beta_occ = np.zeros_like(energies)
        alpha_occ[:alpha] = 1.0
        beta_occ[:beta] = 1.0
        return alpha_occ, beta_occ

    def spin_occupations(count: int) -> np.ndarray:
        if count == 0:
            return np.zeros_like(energies)
        if count == len(energies):
            return np.ones_like(energies)
        # If the frontier gap makes the total thermal tail smaller than one
        # quarter of a floating-point ulp, Fermi occupations are numerically
        # identical to integer filling. This avoids a bisection on ordinary
        # gapped molecules without changing finite-temperature occupations
        # in the near-degenerate regime.
        frontier_gap = energies[count] - energies[count - 1]
        if (
            frontier_gap > 0.0
            and np.all(np.diff(energies) >= 0.0)
            and frontier_gap >= 2.0 * thermal * np.log(
                4.0 * len(energies) / np.finfo(float).eps
            )
        ):
            occupation = np.zeros_like(energies)
            occupation[:count] = 1.0
            return occupation
        low = float(np.min(energies) - 64.0 * thermal)
        high = float(np.max(energies) + 64.0 * thermal)
        for _ in range(200):
            chemical_potential = 0.5 * (low + high)
            argument = np.clip((energies - chemical_potential) / thermal, -700.0, 700.0)
            occupation = 1.0 / (np.exp(argument) + 1.0)
            population_error = float(np.sum(occupation)) - count
            if abs(population_error) <= 2.0e-14:
                break
            old_low, old_high = low, high
            if population_error > 0.0:
                high = chemical_potential
            else:
                low = chemical_potential
            if low == old_low and high == old_high:
                break
        return occupation

    alpha_occ = spin_occupations(alpha)
    return alpha_occ, (
        alpha_occ.copy() if alpha == beta else spin_occupations(beta)
    )


def electronic_entropy_energy(
    alpha_occupations: np.ndarray,
    beta_occupations: np.ndarray,
    temperature_kelvin: float,
) -> float:
    """Return the finite-temperature ``-T S`` contribution in Hartree."""

    thermal = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature_kelvin)
    if thermal <= 0.0:
        return 0.0
    if _gfn2_native is not None and hasattr(_gfn2_native, "electronic_entropy"):
        return float(
            _gfn2_native.electronic_entropy(
                alpha_occupations, beta_occupations, thermal
            )
        )
    result = 0.0
    for occupations in (alpha_occupations, beta_occupations):
        values = np.asarray(occupations, dtype=float)
        interior = (values > 0.0) & (values < 1.0)
        fractions = values[interior]
        result += thermal * float(
            np.sum(fractions * np.log(fractions) + (1.0 - fractions) * np.log(1.0 - fractions))
        )
    return result


def solve_generalized_symmetric(
    fock: np.ndarray,
    overlap: np.ndarray,
    *,
    linear_dependency_threshold: float = 1.0e-10,
    orthogonalizer: np.ndarray | None = None,
    diagonalizer_backend: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fock = np.asarray(fock, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    if fock.shape != overlap.shape or fock.ndim != 2 or fock.shape[0] != fock.shape[1]:
        raise ValueError("Fock and overlap matrices must be equally sized and square")
    if orthogonalizer is None:
        inverse_sqrt = symmetric_orthogonalizer(
            overlap,
            linear_dependency_threshold=linear_dependency_threshold,
            diagonalizer_backend=diagonalizer_backend,
        )
    else:
        inverse_sqrt = np.asarray(orthogonalizer, dtype=float)
        if inverse_sqrt.shape != overlap.shape:
            raise ValueError("orthogonalizer and overlap dimensions differ")
    orthogonal_fock = inverse_sqrt @ fock @ inverse_sqrt
    energies, orthogonal_vectors = eigh_arrays(
        orthogonal_fock, backend=diagonalizer_backend
    )
    return energies, inverse_sqrt @ orthogonal_vectors


def solve_generalized_symmetric_direct(
    fock: np.ndarray,
    overlap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve FC=SCe directly with LAPACK's Cholesky-reduced driver."""

    from scipy import linalg as scipy_linalg

    values, vectors = scipy_linalg.eigh(
        np.asarray(fock, dtype=float),
        np.asarray(overlap, dtype=float),
        lower=True,
        eigvals_only=False,
        check_finite=False,
        driver="gvd",
    )
    return np.asarray(values), np.asarray(vectors)


def symmetric_orthogonalizer(
    overlap: np.ndarray,
    *,
    linear_dependency_threshold: float = 1.0e-10,
    diagonalizer_backend: str | None = None,
    method: str = "auto",
) -> np.ndarray:
    """Build and validate S^-1/2 once for reuse throughout an SCC cycle."""

    matrix = np.asarray(overlap, dtype=float)
    normalized_method = str(method).strip().lower().replace("_", "-")
    if normalized_method not in {"auto", "newton-schulz", "eigh"}:
        raise ValueError("orthogonalizer method must be auto, newton-schulz or eigh")
    if normalized_method == "auto":
        normalized_backend = str(diagonalizer_backend or "numpy").strip().lower()
        normalized_method = (
            "newton-schulz"
            if normalized_backend
            in {"cupy", "cuda", "torch-cuda", "mps", "torch-mps"}
            else "eigh"
        )
    if normalized_method == "newton-schulz":
        result = newton_schulz_inverse_sqrt(
            matrix, backend=diagonalizer_backend
        )
        conservative_minimum = 1.0 / float(
            np.linalg.norm(result.inverse_sqrt, ord="fro") ** 2
        )
        if result.converged and conservative_minimum > linear_dependency_threshold:
            return result.inverse_sqrt
    eigenvalues, eigenvectors = eigh_arrays(matrix, backend=diagonalizer_backend)
    if float(np.min(eigenvalues)) <= linear_dependency_threshold:
        raise np.linalg.LinAlgError("overlap matrix is linearly dependent")
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def newton_schulz_inverse_sqrt(
    matrix: np.ndarray,
    *,
    tolerance: float = 5.0e-13,
    maximum_iterations: int = 100,
    backend: str | None = None,
) -> InverseSqrtResult:
    """Compute a symmetric inverse square root without diagonalization.

    Frobenius scaling places the positive spectrum in the Newton--Schulz
    convergence basin.  The returned residual is ``||I-XAX||_F``; callers can
    therefore fail closed to an eigensolver when an ill-conditioned overlap
    prevents reliable convergence.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("inverse square root requires a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("inverse square root matrix must be finite")
    values = 0.5 * (values + values.T)
    normalized_backend = str(backend or "numpy").strip().lower()
    if normalized_backend in {"cupy", "cuda", "torch-cuda", "mps", "torch-mps"}:
        try:
            return _accelerated_newton_schulz_inverse_sqrt(
                values,
                tolerance=tolerance,
                maximum_iterations=maximum_iterations,
                backend=normalized_backend,
            )
        except (ImportError, RuntimeError, ValueError, TypeError):
            # The caller will use the certified EIGH fallback below.
            pass
    size = len(values)
    identity = np.eye(size, dtype=np.float64)
    scale = float(np.linalg.norm(values, ord="fro"))
    if scale <= np.finfo(np.float64).tiny:
        raise np.linalg.LinAlgError("inverse square root matrix is singular")
    y = values / scale
    z = identity.copy()
    residual = float("inf")
    for iteration in range(1, int(maximum_iterations) + 1):
        correction = 0.5 * (3.0 * identity - z @ y)
        y = y @ correction
        z = correction @ z
        candidate = 0.5 * (z + z.T) / np.sqrt(scale)
        residual = float(np.linalg.norm(identity - candidate @ values @ candidate))
        if np.isfinite(residual) and residual <= tolerance:
            return InverseSqrtResult(candidate, iteration, residual, True)
        if not np.isfinite(residual):
            break
    candidate = 0.5 * (z + z.T) / np.sqrt(scale)
    return InverseSqrtResult(candidate, iteration, residual, False)


def _accelerated_newton_schulz_inverse_sqrt(
    values: np.ndarray,
    *,
    tolerance: float,
    maximum_iterations: int,
    backend: str,
) -> InverseSqrtResult:
    """Run Newton--Schulz on an accelerator and certify the returned matrix."""

    scale = float(np.linalg.norm(values, ord="fro"))
    if scale <= np.finfo(np.float64).tiny:
        raise np.linalg.LinAlgError("inverse square root matrix is singular")
    if backend == "cupy":
        import cupy as cp

        device_values = cp.asarray(values)
        identity = cp.eye(len(values), dtype=device_values.dtype)

        def residual_norm(array):
            return float(cp.linalg.norm(array).item())

        def to_numpy(array):
            return cp.asnumpy(array)

    else:
        import torch

        device = "mps" if backend in {"mps", "torch-mps"} else "cuda"
        dtype = torch.float32 if device == "mps" else torch.float64
        device_values = torch.as_tensor(values, dtype=dtype, device=device)
        identity = torch.eye(len(values), dtype=dtype, device=device)

        def residual_norm(array):
            return float(torch.linalg.norm(array).item())

        def to_numpy(array):
            return array.detach().cpu().numpy()

    y = device_values / scale
    z = identity.clone() if hasattr(identity, "clone") else identity.copy()
    residual = float("inf")
    device_identity = identity
    for iteration in range(1, int(maximum_iterations) + 1):
        correction = 0.5 * (3.0 * identity - z @ y)
        y = y @ correction
        z = correction @ z
        candidate_device = 0.5 * (z + z.T) / np.sqrt(scale)
        residual = residual_norm(
            device_identity - candidate_device @ device_values @ candidate_device
        )
        if np.isfinite(residual) and residual <= tolerance:
            candidate = to_numpy(candidate_device).astype(np.float64)
            return InverseSqrtResult(candidate, iteration, residual, True)
        if not np.isfinite(residual):
            break
    candidate = to_numpy(candidate_device).astype(np.float64)
    return InverseSqrtResult(candidate, iteration, residual, False)


def solve_monopole_scc(
    core_hamiltonian: np.ndarray,
    overlap: np.ndarray,
    coordinates_bohr: np.ndarray,
    partition: ShellPartition,
    *,
    electrons: int,
    multiplicity: int = 1,
    temperature_kelvin: float = 300.0,
    charge_tolerance: float = 1.0e-9,
    maximum_iterations: int = 250,
    damping: float = 0.4,
) -> SCCResult:
    core = np.asarray(core_hamiltonian, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    interaction = shell_interaction_matrix(coordinates_bohr, partition)
    orthogonalizer = symmetric_orthogonalizer(overlap)
    charges = np.zeros(partition.nshells, dtype=float)
    density = np.zeros_like(core)
    occupations = np.zeros(core.shape[0], dtype=float)
    orbital_energies = np.zeros(core.shape[0], dtype=float)
    converged = False
    for iteration in range(1, int(maximum_iterations) + 1):
        fock = core + monopole_fock(overlap, charges, interaction, partition)
        orbital_energies, coefficients = solve_generalized_symmetric(
            fock, overlap, orthogonalizer=orthogonalizer
        )
        occupations = fermi_occupations(
            orbital_energies,
            electrons,
            multiplicity=multiplicity,
            temperature_kelvin=temperature_kelvin,
        )
        density = (coefficients * occupations) @ coefficients.T
        updated = shell_charges(density, overlap, partition)
        residual = updated - charges
        charges = charges + float(damping) * residual
        if float(np.max(np.abs(residual))) <= charge_tolerance:
            charges = updated
            converged = True
            break
    electronic_energy = float(np.sum(density * core))
    energy = electronic_energy + monopole_energy(charges, interaction, partition)
    return SCCResult(
        energy_hartree=energy,
        density=density,
        shell_charges=charges,
        orbital_energies_hartree=orbital_energies,
        occupations=occupations,
        iterations=iteration,
        converged=converged,
    )


__all__ = [
    "BOLTZMANN_HARTREE_PER_KELVIN",
    "SCCResult",
    "InverseSqrtResult",
    "ShellPartition",
    "fermi_occupations",
    "spin_fermi_occupations",
    "electronic_entropy_energy",
    "monopole_energy",
    "monopole_fock",
    "monopole_fock_sparse",
    "shell_charges",
    "shell_charges_sparse",
    "shell_interaction_matrix",
    "shell_interaction_operator",
    "solve_generalized_symmetric",
    "solve_monopole_scc",
    "symmetric_orthogonalizer",
    "newton_schulz_inverse_sqrt",
]
