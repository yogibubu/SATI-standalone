"""Variational contracted rovibronic Hamiltonians for LOCK.

The basis is ``|LAM> |electronic> |S,M_S> |J,K>``.  Rotation is assembled
from the exact kinetic angular momentum ``R = J-L-l-S`` so Coriolis,
spin-uncoupling, and case mixing are diagonalized together with diabatic
Renner--Teller or Jahn--Teller matrices in the same contracted LAM space.
All matrix elements are in MHz.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from matrix_numerics import diagonalize_hermitian

from .spin_hyperfine import wigner_3j_doubled
from .native_rotor import MHZ_TO_KELVIN
from .molecular_symmetry import FiniteMolecularSymmetryGroup


VARIATIONAL_ROVIBRONIC_SCHEMA = "matrix.lock.variational_rovibronic.v1"
VARIATIONAL_ROVIBRONIC_RESULT_SCHEMA = "matrix.lock.variational_rovibronic_result.v1"


@dataclass(frozen=True)
class ContractedVibronicModel:
    """Coordinate-contracted electronic, spin, and angular-momentum data."""

    vibronic_hamiltonian_MHz: np.ndarray
    electronic_angular_momentum: tuple[np.ndarray, np.ndarray, np.ndarray]
    lam_angular_momentum: tuple[np.ndarray, np.ndarray, np.ndarray]
    two_S: int
    rotational_metric_MHz: np.ndarray
    spin_orbit_MHz: np.ndarray | None = None
    spin_lam_MHz: np.ndarray | None = None
    contracted_internal_hyperfine_MHz: np.ndarray | None = None
    transition_dipoles_D: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    symmetry_group: FiniteMolecularSymmetryGroup | None = None
    vibronic_symmetry_representation: tuple[np.ndarray, ...] = ()
    dipole_irreps: tuple[str, str, str] = ()
    required_total_irrep: str | None = None
    vibronic_labels: tuple[str, ...] = ()
    electronic_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nv = self.lam_angular_momentum[0].shape[0]
        ne = self.electronic_angular_momentum[0].shape[0]
        if nv < 1 or ne < 1 or self.two_S < 0:
            raise ValueError("the contracted rovibronic basis dimensions must be positive")
        _validate_operators(self.lam_angular_momentum, nv, "LAM angular momentum")
        _validate_operators(self.electronic_angular_momentum, ne, "electronic angular momentum")
        vibronic = np.asarray(self.vibronic_hamiltonian_MHz, dtype=complex)
        if vibronic.shape != (nv * ne, nv * ne) or not _is_hermitian(vibronic):
            raise ValueError("vibronic_hamiltonian_MHz must be Hermitian on LAM x electronic space")
        metric = np.asarray(self.rotational_metric_MHz, dtype=float)
        if metric.shape != (3, 3) or not np.allclose(metric, metric.T, atol=1.0e-12):
            raise ValueError("rotational_metric_MHz must be a real symmetric 3 x 3 tensor")
        if np.min(np.linalg.eigvalsh(metric)) <= 0.0:
            raise ValueError("rotational_metric_MHz must be positive definite")
        for name, tensor in (("spin_orbit_MHz", self.spin_orbit_MHz), ("spin_lam_MHz", self.spin_lam_MHz)):
            if tensor is not None:
                values = np.asarray(tensor, dtype=float)
                if values.shape != (3, 3) or not np.all(np.isfinite(values)):
                    raise ValueError(f"{name} must be a finite 3 x 3 tensor")
        internal = nv * ne * (self.two_S + 1)
        if self.contracted_internal_hyperfine_MHz is not None:
            hyperfine = np.asarray(self.contracted_internal_hyperfine_MHz, dtype=complex)
            if hyperfine.shape != (internal, internal) or not _is_hermitian(hyperfine):
                raise ValueError(
                    "contracted_internal_hyperfine_MHz must be Hermitian on "
                    "LAM x electronic x electron-spin space"
                )
        if self.transition_dipoles_D is not None:
            _validate_operators(self.transition_dipoles_D, nv * ne, "transition dipoles")
        if self.symmetry_group is None:
            if self.vibronic_symmetry_representation or self.dipole_irreps or self.required_total_irrep:
                raise ValueError("symmetry metadata requires symmetry_group")
        else:
            if len(self.vibronic_symmetry_representation) != len(self.symmetry_group.operations):
                raise ValueError("provide one vibronic representation matrix per symmetry operation")
            for label in self.symmetry_group.labels:
                self.symmetry_group.projector(self.vibronic_symmetry_representation, label)
            if self.dipole_irreps:
                if len(self.dipole_irreps) != 3 or any(
                    label not in self.symmetry_group.labels for label in self.dipole_irreps
                ):
                    raise ValueError("dipole_irreps must contain valid x, y, z irreps")
                if self.dipole_irreps[0] != self.dipole_irreps[1]:
                    raise ValueError("Cartesian x and y must share an irrep in the spherical basis")
            if self.required_total_irrep not in (None, *self.symmetry_group.labels):
                raise ValueError("required_total_irrep is absent from the symmetry group")
        if self.vibronic_labels and len(self.vibronic_labels) != nv:
            raise ValueError("provide one label per contracted LAM function")
        if self.electronic_labels and len(self.electronic_labels) != ne:
            raise ValueError("provide one label per electronic diabatic function")

    @property
    def lam_dimension(self) -> int:
        return self.lam_angular_momentum[0].shape[0]

    @property
    def electronic_dimension(self) -> int:
        return self.electronic_angular_momentum[0].shape[0]


@dataclass(frozen=True)
class VariationalRovibronicBlock:
    two_J: int
    energies_MHz: np.ndarray
    vectors: np.ndarray
    hermiticity_error_MHz: float
    residual_norms_MHz: np.ndarray
    diagonalizer_backend: str
    irrep: str | None = None
    nuclear_spin_weight: int = 1
    assembly_backend: str = "dense"


@dataclass(frozen=True)
class VariationalRovibronicResult:
    model: ContractedVibronicModel
    blocks: tuple[VariationalRovibronicBlock, ...]


def contract_diabatic_vibronic_hamiltonian(
    lam_vectors: np.ndarray,
    diabatic_surface_MHz: np.ndarray,
    *,
    primitive_momentum_operators: Sequence[np.ndarray] = (),
    derivative_couplings_MHz: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Contract matrix-valued JT/RT surfaces and derivative couplings.

    A derivative surface ``D_mu(q)`` contributes the explicitly Hermitian
    operator ``{D_mu,p_mu}/2`` before contraction.
    """
    vectors = np.asarray(lam_vectors, dtype=complex)
    surface = np.asarray(diabatic_surface_MHz, dtype=complex)
    if vectors.ndim != 2 or surface.ndim != 3 or surface.shape[0] != vectors.shape[0]:
        raise ValueError("diabatic surface and LAM vectors disagree")
    grid, electronic, electronic_right = surface.shape
    if electronic != electronic_right:
        raise ValueError("the diabatic surface must be square in electronic space")
    if not np.allclose(surface, np.swapaxes(surface.conj(), 1, 2), atol=1.0e-10):
        raise ValueError("the diabatic surface must be Hermitian at every grid point")
    if len(primitive_momentum_operators) != len(derivative_couplings_MHz):
        raise ValueError("pair every derivative coupling with a primitive momentum operator")
    nv = vectors.shape[1]
    result = np.zeros((nv * electronic, nv * electronic), dtype=complex)
    for left in range(electronic):
        for right in range(electronic):
            primitive = np.diag(surface[:, left, right])
            for momentum, raw_coupling in zip(
                primitive_momentum_operators, derivative_couplings_MHz, strict=True
            ):
                p = np.asarray(momentum, dtype=complex)
                coupling = np.asarray(raw_coupling, dtype=complex)
                if p.shape != (grid, grid) or coupling.shape != surface.shape:
                    raise ValueError("a derivative coupling has incompatible dimensions")
                diagonal = np.diag(coupling[:, left, right])
                primitive += 0.5 * (diagonal @ p + p @ diagonal)
            block = vectors.conj().T @ primitive @ vectors
            result += np.kron(block, _matrix_unit(electronic, left, right))
    return _hermitian(result)


def variational_rovibronic_hamiltonian(
    model: ContractedVibronicModel, two_J: int
) -> np.ndarray:
    """Assemble one exact contracted ``J`` block without choosing a Hund case."""
    if two_J < 0 or (two_J & 1) != (model.two_S & 1):
        raise ValueError("two_J must be nonnegative and have the electron-spin parity")
    nv, ne, ns, nj = (
        model.lam_dimension,
        model.electronic_dimension,
        model.two_S + 1,
        two_J + 1,
    )
    Iv, Ie, Is, Ij = (np.eye(value, dtype=complex) for value in (nv, ne, ns, nj))
    spin = _angular_momentum_operators_doubled(model.two_S)
    total = _angular_momentum_operators_doubled(two_J, body_fixed=True)
    shape = (nv * ne * ns * nj, nv * ne * ns * nj)
    zero = np.zeros(shape, dtype=complex)
    lam = [_kron4(operator, Ie, Is, Ij) for operator in model.lam_angular_momentum]
    electronic = [_kron4(Iv, operator, Is, Ij) for operator in model.electronic_angular_momentum]
    electron_spin = [_kron4(Iv, Ie, operator, Ij) for operator in spin]
    angular = [_kron4(Iv, Ie, Is, operator) for operator in total]
    kinetic = [angular[axis] - electronic[axis] - lam[axis] - electron_spin[axis] for axis in range(3)]
    hamiltonian = np.kron(np.asarray(model.vibronic_hamiltonian_MHz), np.kron(Is, Ij))
    if model.contracted_internal_hyperfine_MHz is not None:
        hamiltonian += np.kron(
            np.asarray(model.contracted_internal_hyperfine_MHz, dtype=complex), Ij
        )
    metric = np.asarray(model.rotational_metric_MHz)
    for left in range(3):
        for right in range(3):
            hamiltonian += 0.5 * metric[left, right] * (
                kinetic[left] @ kinetic[right] + kinetic[right] @ kinetic[left]
            )
    if model.spin_orbit_MHz is not None:
        tensor = np.asarray(model.spin_orbit_MHz)
        for left in range(3):
            for right in range(3):
                hamiltonian += tensor[left, right] * electronic[left] @ electron_spin[right]
    if model.spin_lam_MHz is not None:
        tensor = np.asarray(model.spin_lam_MHz)
        for left in range(3):
            for right in range(3):
                hamiltonian += tensor[left, right] * lam[left] @ electron_spin[right]
    if hamiltonian.shape != zero.shape:
        raise RuntimeError("internal LOCK rovibronic dimension mismatch")
    return hamiltonian


def variational_rovibronic_factorized_terms(
    model: ContractedVibronicModel, two_J: int
) -> tuple[tuple[complex, tuple[np.ndarray, np.ndarray]], ...]:
    """Return an exact two-factor internal/rotation Kronecker expansion."""
    if two_J < 0 or (two_J & 1) != (model.two_S & 1):
        raise ValueError("two_J must be nonnegative and have the electron-spin parity")
    nv, ne, ns, nj = model.lam_dimension, model.electronic_dimension, model.two_S + 1, two_J + 1
    ive, ispin, ij = np.eye(nv * ne, dtype=complex), np.eye(ns, dtype=complex), np.eye(nj, dtype=complex)
    spin = _angular_momentum_operators_doubled(model.two_S)
    total = _angular_momentum_operators_doubled(two_J, body_fixed=True)
    lam = [np.kron(operator, np.eye(ne)) for operator in model.lam_angular_momentum]
    electronic = [np.kron(np.eye(nv), operator) for operator in model.electronic_angular_momentum]
    internal = np.kron(np.asarray(model.vibronic_hamiltonian_MHz), ispin)
    if model.contracted_internal_hyperfine_MHz is not None:
        internal = internal + np.asarray(model.contracted_internal_hyperfine_MHz)
    if model.spin_orbit_MHz is not None:
        internal = internal + sum(
            model.spin_orbit_MHz[a, b] * np.kron(electronic[a], spin[b])
            for a in range(3) for b in range(3)
        )
    if model.spin_lam_MHz is not None:
        internal = internal + sum(
            model.spin_lam_MHz[a, b] * np.kron(lam[a], spin[b])
            for a in range(3) for b in range(3)
        )
    q = [np.kron(lam[a] + electronic[a], ispin) + np.kron(ive, spin[a]) for a in range(3)]
    identity_internal = np.eye(nv * ne * ns, dtype=complex)
    internal_rotation = np.zeros_like(internal)
    pure_rotation = np.zeros_like(ij)
    for a in range(3):
        for b in range(3):
            coefficient = 0.5 * float(model.rotational_metric_MHz[a, b])
            internal_rotation += coefficient * (q[a] @ q[b] + q[b] @ q[a])
            pure_rotation += coefficient * (total[a] @ total[b] + total[b] @ total[a])
    terms: list[tuple[complex, tuple[np.ndarray, np.ndarray]]] = [
        (1.0, (internal + internal_rotation, ij)),
        (1.0, (identity_internal, pure_rotation)),
    ]
    for a in range(3):
        for b in range(3):
            terms.append((-2.0 * float(model.rotational_metric_MHz[a, b]), (q[a], total[b])))
    return tuple(terms)


def variational_rovibronic_matvec(
    terms: Sequence[tuple[complex, tuple[np.ndarray, np.ndarray]]],
    vector: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Apply factorized rovibronic terms with the compiled kernel when available."""
    values = np.asarray(vector, dtype=complex)
    if backend not in {"auto", "python", "compiled"}:
        raise ValueError("matrix-free backend must be auto, python, or compiled")
    if backend != "python":
        try:
            from . import _lock_native
            return np.asarray(_lock_native.kronecker_sum_matvec(terms, values))
        except (ImportError, AttributeError):
            if backend == "compiled":
                raise RuntimeError("compiled LOCK Kronecker kernel is unavailable")
    result = np.zeros_like(values)
    for scale, factors in terms:
        tensor = values.reshape(tuple(factor.shape[0] for factor in factors))
        for axis, factor in enumerate(factors):
            tensor = np.moveaxis(np.tensordot(factor, tensor, axes=(1, axis)), 0, axis)
        result += scale * tensor.reshape(-1)
    return result


def solve_variational_rovibronic(
    model: ContractedVibronicModel,
    *,
    two_j_values: Sequence[int] | range,
    levels_per_block: int | None = None,
    diagonalizer_backend: str | None = None,
    assembly_backend: str = "auto",
    matrix_free_threshold: int = 256,
) -> VariationalRovibronicResult:
    """Diagonalize contracted rovibronic blocks and report numerical residuals."""
    if assembly_backend not in {"auto", "dense", "matrix-free"}:
        raise ValueError("assembly_backend must be auto, dense, or matrix-free")
    blocks = []
    for two_J in sorted({int(value) for value in two_j_values}):
        dimension = (
            model.lam_dimension * model.electronic_dimension
            * (model.two_S + 1) * (two_J + 1)
        )
        use_matrix_free = assembly_backend == "matrix-free" or (
            assembly_backend == "auto"
            and levels_per_block is not None
            and levels_per_block < dimension - 1
            and dimension >= matrix_free_threshold
        )
        if use_matrix_free:
            blocks.extend(_solve_variational_matrix_free(
                model, two_J, levels_per_block=levels_per_block
            ))
            continue
        matrix = variational_rovibronic_hamiltonian(model, two_J)
        error = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
        matrix = _hermitian(matrix)
        projections: tuple[tuple[str | None, np.ndarray, int], ...]
        if model.symmetry_group is None:
            projections = ((None, np.eye(matrix.shape[0], dtype=complex), 1),)
        else:
            rest = (model.two_S + 1) * (two_J + 1)
            projected = []
            for irrep in model.symmetry_group.labels:
                internal = model.symmetry_group.symmetry_basis(
                    model.vibronic_symmetry_representation, irrep
                )
                if internal.shape[1] == 0:
                    continue
                basis = np.kron(internal, np.eye(rest, dtype=complex))
                commutator = matrix @ (basis @ basis.conj().T) - (basis @ basis.conj().T) @ matrix
                scale = max(float(np.linalg.norm(matrix)), 1.0)
                if np.linalg.norm(commutator) > 1.0e-8 * scale:
                    raise ValueError(f"rovibronic Hamiltonian does not preserve the {irrep} symmetry block")
                required = model.required_total_irrep or model.symmetry_group.identity_irrep
                weight = model.symmetry_group.nuclear_spin_statistical_weight(irrep, required)
                projected.append((irrep, basis, weight))
            projections = tuple(projected)
        for irrep, basis, spin_weight in projections:
            reduced = _hermitian(basis.conj().T @ matrix @ basis)
            retained = reduced.shape[0] if levels_per_block is None else min(levels_per_block, reduced.shape[0])
            if retained < 1:
                raise ValueError("levels_per_block must be positive")
            diagonalized = diagonalize_hermitian(
                reduced,
                backend=diagonalizer_backend,
                subset_by_index=(0, retained - 1) if retained < reduced.shape[0] else None,
            )
            if diagonalized.eigenvectors is None:
                raise RuntimeError("LOCK diagonalizer did not return rovibronic eigenvectors")
            energies = np.asarray(diagonalized.eigenvalues)
            vectors = basis @ np.asarray(diagonalized.eigenvectors)
            residuals = np.linalg.norm(matrix @ vectors - vectors * energies[None, :], axis=0)
            blocks.append(VariationalRovibronicBlock(
                two_J=two_J,
                energies_MHz=energies.real,
                vectors=vectors,
                hermiticity_error_MHz=error,
                residual_norms_MHz=residuals,
                diagonalizer_backend=diagonalized.backend,
                irrep=irrep,
                nuclear_spin_weight=spin_weight,
                assembly_backend="dense",
            ))
    return VariationalRovibronicResult(model=model, blocks=tuple(blocks))


def _solve_variational_matrix_free(
    model: ContractedVibronicModel,
    two_J: int,
    *,
    levels_per_block: int | None,
) -> list[VariationalRovibronicBlock]:
    from scipy.sparse.linalg import LinearOperator, eigsh

    terms = variational_rovibronic_factorized_terms(model, two_J)
    dimension = terms[0][1][0].shape[0] * terms[0][1][1].shape[0]

    def action(vector: np.ndarray) -> np.ndarray:
        return variational_rovibronic_matvec(terms, vector)

    if model.symmetry_group is None:
        projections = ((None, np.eye(dimension, dtype=complex), 1),)
    else:
        rest = (model.two_S + 1) * (two_J + 1)
        projected = []
        for irrep in model.symmetry_group.labels:
            internal = model.symmetry_group.symmetry_basis(
                model.vibronic_symmetry_representation, irrep
            )
            if internal.shape[1] == 0:
                continue
            basis = np.kron(internal, np.eye(rest, dtype=complex))
            required = model.required_total_irrep or model.symmetry_group.identity_irrep
            weight = model.symmetry_group.nuclear_spin_statistical_weight(irrep, required)
            projected.append((irrep, basis, weight))
        projections = tuple(projected)
    blocks = []
    for irrep, basis, spin_weight in projections:
        reduced_dimension = basis.shape[1]
        retained = reduced_dimension if levels_per_block is None else min(
            levels_per_block, reduced_dimension
        )
        if retained < 1:
            raise ValueError("levels_per_block must be positive")

        def reduced_action(vector: np.ndarray) -> np.ndarray:
            return basis.conj().T @ action(basis @ vector)

        if retained >= reduced_dimension - 1:
            reduced = np.column_stack([
                reduced_action(np.eye(reduced_dimension, dtype=complex)[:, index])
                for index in range(reduced_dimension)
            ])
            values, reduced_vectors = np.linalg.eigh(_hermitian(reduced))
            values, reduced_vectors = values[:retained], reduced_vectors[:, :retained]
            backend = "factorized-numpy-eigh"
        else:
            operator = LinearOperator(
                (reduced_dimension, reduced_dimension), matvec=reduced_action,
                dtype=np.complex128,
            )
            values, reduced_vectors = eigsh(
                operator, k=retained, which="SA", tol=1.0e-13,
                maxiter=max(1000, 20 * reduced_dimension),
                v0=np.ones(reduced_dimension, dtype=complex) / math.sqrt(reduced_dimension),
            )
            order = np.argsort(values)
            values, reduced_vectors = values[order], reduced_vectors[:, order]
            backend = "scipy-eigsh"
        vectors = basis @ reduced_vectors
        residuals = np.asarray([
            np.linalg.norm(action(vectors[:, index]) - values[index] * vectors[:, index])
            for index in range(retained)
        ])
        blocks.append(VariationalRovibronicBlock(
            two_J=two_J,
            energies_MHz=np.asarray(values).real,
            vectors=vectors,
            hermiticity_error_MHz=0.0,
            residual_norms_MHz=residuals,
            diagonalizer_backend=backend,
            irrep=irrep,
            nuclear_spin_weight=spin_weight,
            assembly_backend="matrix-free-kronecker",
        ))
    return blocks


def variational_rovibronic_transitions(
    result: VariationalRovibronicResult,
    *,
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
) -> list[dict[str, float | int | str | None]]:
    """Evaluate reduced electric-dipole strengths from variational eigenvectors."""
    model = result.model
    if model.transition_dipoles_D is None:
        raise ValueError("the contracted model contains no transition dipole operators")
    if temperature_K is not None and temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    mu_x, mu_y, mu_z = (np.asarray(item, dtype=complex) for item in model.transition_dipoles_D)
    spherical = (
        (mu_x - 1.0j * mu_y) / math.sqrt(2.0),
        mu_z,
        -(mu_x + 1.0j * mu_y) / math.sqrt(2.0),
    )
    identity_spin = np.eye(model.two_S + 1, dtype=complex)
    ground = min(float(block.energies_MHz.min()) for block in result.blocks)
    partition = 1.0
    if temperature_K is not None:
        partition = sum(
            block.nuclear_spin_weight * (block.two_J + 1.0)
            * float(np.exp(-(block.energies_MHz - ground) * MHZ_TO_KELVIN / temperature_K).sum())
            for block in result.blocks
        )
    lines: list[dict[str, float | int | str | None]] = []
    for lower in result.blocks:
        for upper in result.blocks:
            if abs(upper.two_J - lower.two_J) > 2 or upper.two_J + lower.two_J < 2:
                continue
            strength_matrix = np.zeros((len(upper.energies_MHz), len(lower.energies_MHz)))
            operator_irreps = (
                (model.dipole_irreps[0], model.dipole_irreps[2], model.dipole_irreps[0])
                if model.dipole_irreps else (None, None, None)
            )
            for q, dipole, operator_irrep in zip(
                (-1, 0, 1), spherical, operator_irreps, strict=True
            ):
                if (
                    model.symmetry_group is not None
                    and lower.irrep is not None
                    and upper.irrep is not None
                    and operator_irrep is not None
                    and not model.symmetry_group.contains_scalar(
                        lower.irrep, operator_irrep, upper.irrep
                    )
                ):
                    continue
                rotation = _body_fixed_rank1_matrix(upper.two_J, lower.two_J, q)
                primitive = np.kron(np.kron(dipole, identity_spin), rotation)
                amplitudes = upper.vectors.conj().T @ primitive @ lower.vectors
                strength_matrix += np.abs(amplitudes) ** 2
            for lower_index, lower_energy in enumerate(lower.energies_MHz):
                for upper_index, upper_energy in enumerate(upper.energies_MHz):
                    frequency = float(upper_energy - lower_energy)
                    if frequency <= 1.0e-10:
                        continue
                    strength = float(strength_matrix[upper_index, lower_index])
                    if strength <= 1.0e-24:
                        continue
                    if temperature_K is None:
                        intensity = strength
                    else:
                        population = lower.nuclear_spin_weight * (lower.two_J + 1.0) * math.exp(
                            -(float(lower_energy) - ground) * MHZ_TO_KELVIN / temperature_K
                        ) / partition
                        intensity = strength * population * (
                            1.0 - math.exp(-frequency * MHZ_TO_KELVIN / temperature_K)
                        )
                    if intensity < intensity_cut:
                        continue
                    lines.append({
                        "lower_two_J": lower.two_J,
                        "lower_index": lower_index,
                        "lower_irrep": lower.irrep,
                        "upper_two_J": upper.two_J,
                        "upper_index": upper_index,
                        "upper_irrep": upper.irrep,
                        "frequency_MHz": frequency,
                        "line_strength_D2": strength,
                        "intensity": intensity,
                    })
    return sorted(lines, key=lambda item: float(item["frequency_MHz"]))


def write_variational_rovibronic_model(
    path: Path | str, model: ContractedVibronicModel
) -> Path:
    """Write the compact TRINITY-to-LOCK model as a portable NPZ artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema": np.asarray(VARIATIONAL_ROVIBRONIC_SCHEMA),
        "vibronic_hamiltonian_MHz": np.asarray(model.vibronic_hamiltonian_MHz),
        "two_S": np.asarray(model.two_S, dtype=np.int64),
        "rotational_metric_MHz": np.asarray(model.rotational_metric_MHz),
        "vibronic_labels": np.asarray(model.vibronic_labels, dtype=str),
        "electronic_labels": np.asarray(model.electronic_labels, dtype=str),
    }
    for prefix, operators in (
        ("electronic_angular_momentum", model.electronic_angular_momentum),
        ("lam_angular_momentum", model.lam_angular_momentum),
    ):
        for axis, operator in zip("xyz", operators, strict=True):
            arrays[f"{prefix}_{axis}"] = np.asarray(operator)
    if model.spin_orbit_MHz is not None:
        arrays["spin_orbit_MHz"] = np.asarray(model.spin_orbit_MHz)
    if model.spin_lam_MHz is not None:
        arrays["spin_lam_MHz"] = np.asarray(model.spin_lam_MHz)
    if model.contracted_internal_hyperfine_MHz is not None:
        arrays["contracted_internal_hyperfine_MHz"] = np.asarray(
            model.contracted_internal_hyperfine_MHz
        )
    if model.transition_dipoles_D is not None:
        for axis, operator in zip("xyz", model.transition_dipoles_D, strict=True):
            arrays[f"transition_dipole_{axis}_D"] = np.asarray(operator)
    if model.symmetry_group is not None:
        arrays["symmetry_group_json"] = np.asarray(json.dumps(model.symmetry_group.to_dict()))
        arrays["required_total_irrep"] = np.asarray(model.required_total_irrep or "")
        arrays["dipole_irreps"] = np.asarray(model.dipole_irreps, dtype=str)
        for index, operator in enumerate(model.vibronic_symmetry_representation):
            arrays[f"symmetry_representation_{index}"] = np.asarray(operator)
    with target.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return target


def read_variational_rovibronic_model(path: Path | str) -> ContractedVibronicModel:
    """Read and validate a versioned contracted rovibronic NPZ artifact."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        schema = str(archive["schema"].item())
        if schema != VARIATIONAL_ROVIBRONIC_SCHEMA:
            raise ValueError(f"unsupported variational rovibronic schema: {schema!r}")
        keys = set(archive.files)
        return ContractedVibronicModel(
            vibronic_hamiltonian_MHz=np.asarray(archive["vibronic_hamiltonian_MHz"]),
            electronic_angular_momentum=tuple(
                np.asarray(archive[f"electronic_angular_momentum_{axis}"])
                for axis in "xyz"
            ),
            lam_angular_momentum=tuple(
                np.asarray(archive[f"lam_angular_momentum_{axis}"])
                for axis in "xyz"
            ),
            two_S=int(archive["two_S"].item()),
            rotational_metric_MHz=np.asarray(archive["rotational_metric_MHz"]),
            spin_orbit_MHz=(
                np.asarray(archive["spin_orbit_MHz"])
                if "spin_orbit_MHz" in keys else None
            ),
            spin_lam_MHz=(
                np.asarray(archive["spin_lam_MHz"])
                if "spin_lam_MHz" in keys else None
            ),
            contracted_internal_hyperfine_MHz=(
                np.asarray(archive["contracted_internal_hyperfine_MHz"])
                if "contracted_internal_hyperfine_MHz" in keys else None
            ),
            transition_dipoles_D=(
                tuple(np.asarray(archive[f"transition_dipole_{axis}_D"]) for axis in "xyz")
                if all(f"transition_dipole_{axis}_D" in keys for axis in "xyz") else None
            ),
            symmetry_group=(
                FiniteMolecularSymmetryGroup.from_dict(
                    json.loads(str(archive["symmetry_group_json"].item()))
                ) if "symmetry_group_json" in keys else None
            ),
            vibronic_symmetry_representation=(
                tuple(
                    np.asarray(archive[f"symmetry_representation_{index}"])
                    for index in range(len(json.loads(str(archive["symmetry_group_json"].item()))["operations"]))
                ) if "symmetry_group_json" in keys else ()
            ),
            dipole_irreps=(
                tuple(str(value) for value in archive["dipole_irreps"])
                if "dipole_irreps" in keys else ()
            ),
            required_total_irrep=(
                str(archive["required_total_irrep"].item()) or None
                if "required_total_irrep" in keys else None
            ),
            vibronic_labels=tuple(str(value) for value in archive["vibronic_labels"]),
            electronic_labels=tuple(str(value) for value in archive["electronic_labels"]),
        )


def write_variational_rovibronic_result(
    path: Path | str,
    result: VariationalRovibronicResult,
    *,
    model_path: Path | str | None = None,
) -> Path:
    """Write level energies and numerical diagnostics as versioned JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": VARIATIONAL_ROVIBRONIC_RESULT_SCHEMA,
        "model": None if model_path is None else str(Path(model_path).resolve()),
        "basis": {
            "lam_dimension": result.model.lam_dimension,
            "electronic_dimension": result.model.electronic_dimension,
            "two_S": result.model.two_S,
        },
        "blocks": [
            {
                "two_J": block.two_J,
                "energies_MHz": block.energies_MHz.tolist(),
                "hermiticity_error_MHz": block.hermiticity_error_MHz,
                "residual_norms_MHz": block.residual_norms_MHz.tolist(),
                "diagonalizer_backend": block.diagonalizer_backend,
                "irrep": block.irrep,
                "nuclear_spin_weight": block.nuclear_spin_weight,
                "assembly_backend": block.assembly_backend,
            }
            for block in result.blocks
        ],
    }
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def hund_case_a_to_b_transform(
    *, two_J: int, two_S: int, two_Lambda: int
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Return ``<case-a Sigma | case-b N>`` Clebsch--Gordan coefficients."""
    sigmas = tuple(
        two_sigma
        for two_sigma in range(-two_S, two_S + 1, 2)
        if abs(two_Lambda + two_sigma) <= two_J
    )
    two_ns = tuple(
        two_n
        for two_n in range(abs(two_J - two_S), two_J + two_S + 1, 2)
        if two_n >= abs(two_Lambda)
    )
    transform = np.asarray([
        [
            math.sqrt(two_n + 1.0)
            * wigner_3j_doubled(
                two_J, two_S, two_n,
                two_Lambda + two_sigma, -two_sigma, -two_Lambda,
            )
            for two_n in two_ns
        ]
        for two_sigma in sigmas
    ])
    if transform.shape[0] == transform.shape[1] and not np.allclose(
        transform.T @ transform, np.eye(transform.shape[1]), atol=1.0e-10
    ):
        raise RuntimeError("Hund-case transformation lost unitarity")
    return sigmas, two_ns, transform


def _angular_momentum_operators_doubled(
    two_j: int, *, body_fixed: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if two_j < 0:
        raise ValueError("two_j must be nonnegative")
    j = two_j / 2.0
    m = np.arange(-two_j, two_j + 1, 2, dtype=float) / 2.0
    raising = np.zeros((len(m), len(m)), dtype=complex)
    for index, value in enumerate(m[:-1]):
        raising[index + 1, index] = math.sqrt(j * (j + 1.0) - value * (value + 1.0))
    lower = raising.T
    x = 0.5 * (raising + lower)
    y = (lower - raising) / (2.0j) if body_fixed else (raising - lower) / (2.0j)
    return x, y, np.diag(m).astype(complex)


def _matrix_unit(size: int, row: int, column: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=complex)
    result[row, column] = 1.0
    return result


def _body_fixed_rank1_matrix(two_j_upper: int, two_j_lower: int, q: int) -> np.ndarray:
    result = np.zeros((two_j_upper + 1, two_j_lower + 1), dtype=complex)
    factor = math.sqrt((two_j_upper + 1.0) * (two_j_lower + 1.0))
    for row, two_k_upper in enumerate(range(-two_j_upper, two_j_upper + 1, 2)):
        phase = -1.0 if ((two_j_upper - two_k_upper) // 2) & 1 else 1.0
        for column, two_k_lower in enumerate(range(-two_j_lower, two_j_lower + 1, 2)):
            result[row, column] = phase * factor * wigner_3j_doubled(
                two_j_upper, 2, two_j_lower, -two_k_upper, 2 * q, two_k_lower
            )
    return result


def _kron4(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(np.kron(a, b), c), d)


def _validate_operators(operators: Sequence[np.ndarray], size: int, name: str) -> None:
    if len(operators) != 3:
        raise ValueError(f"{name} needs three Cartesian components")
    if any(np.asarray(item).shape != (size, size) or not _is_hermitian(item) for item in operators):
        raise ValueError(f"{name} components must be Hermitian square matrices")


def _is_hermitian(matrix: np.ndarray, tolerance: float = 1.0e-10) -> bool:
    values = np.asarray(matrix)
    return bool(np.all(np.isfinite(values)) and np.allclose(values, values.conj().T, atol=tolerance))


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=complex)
    return 0.5 * (values + values.conj().T)


__all__ = [
    "ContractedVibronicModel",
    "VARIATIONAL_ROVIBRONIC_SCHEMA",
    "VARIATIONAL_ROVIBRONIC_RESULT_SCHEMA",
    "VariationalRovibronicBlock",
    "VariationalRovibronicResult",
    "contract_diabatic_vibronic_hamiltonian",
    "hund_case_a_to_b_transform",
    "read_variational_rovibronic_model",
    "solve_variational_rovibronic",
    "variational_rovibronic_hamiltonian",
    "variational_rovibronic_factorized_terms",
    "variational_rovibronic_matvec",
    "variational_rovibronic_transitions",
    "write_variational_rovibronic_model",
    "write_variational_rovibronic_result",
]
