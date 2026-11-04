"""Compiled resident closed-shell sTDA-xTB compatible with Grimme std2."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .electronic_state import GFN2OrbitalState
from .sf_tda import HARTREE_TO_EV, grimme_stda_atomic_hardnesses

try:
    from . import _gfn2_native
except ImportError:  # pragma: no cover - source tree without compiled extension
    _gfn2_native = None


@dataclass(frozen=True)
class ClosedShellSTDASelection:
    """Immutable orbital rectangle and rejected configurations for one PES."""
    occupied: tuple[int, ...]
    virtual: tuple[int, ...]
    rejected_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ClosedShellSTDAResult:
    excitation_energies_hartree: np.ndarray
    amplitudes: np.ndarray
    occupied_orbitals: np.ndarray
    virtual_orbitals: np.ndarray
    primary_count: int
    perturbative_count: int
    residual_norms_hartree: np.ndarray
    converged: bool
    solver_backend: str
    operator_batches: int
    reference_model_id: str
    absolute_grimme_reference_compatible: bool
    selection: ClosedShellSTDASelection | None = None


class ClosedShellSTDAMatrixDerivative:
    """Fixed-selection sTDA matrix with pair-packed Cartesian derivatives."""

    def __init__(self, matrix_hartree: np.ndarray, derivative_hartree_per_bohr: np.ndarray, *, selection=None):
        self.selection = selection
        matrix = np.array(matrix_hartree, dtype=float, copy=True, order="C")
        derivative = np.asarray(derivative_hartree_per_bohr, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("closed-shell sTDA matrix must be square")
        if derivative.ndim != 4 or derivative.shape[2:] != matrix.shape:
            raise ValueError("closed-shell sTDA matrix derivative has the wrong shape")
        if not np.allclose(matrix, matrix.T, atol=2.0e-11, rtol=0.0):
            raise ValueError("closed-shell sTDA matrix must be symmetric")
        if not np.allclose(
            derivative, derivative.swapaxes(-1, -2), atol=2.0e-10, rtol=0.0
        ):
            raise ValueError("closed-shell sTDA matrix derivative must be symmetric")
        lower = np.tril_indices(matrix.shape[0])
        packed = np.array(derivative[..., lower[0], lower[1]], copy=True, order="C")
        matrix.setflags(write=False)
        packed.setflags(write=False)
        self.matrix_hartree = matrix
        self.derivative_pair_packed_hartree_per_bohr = packed
        self._lower_rows = lower[0]
        self._lower_columns = lower[1]

    @property
    def derivative_hartree_per_bohr(self) -> np.ndarray:
        """Materialize the legacy dense view only for compatibility callers."""

        dimension = self.matrix_hartree.shape[0]
        derivative = np.zeros(
            (*self.derivative_pair_packed_hartree_per_bohr.shape[:2], dimension, dimension)
        )
        derivative[..., self._lower_rows, self._lower_columns] = (
            self.derivative_pair_packed_hartree_per_bohr
        )
        derivative[..., self._lower_columns, self._lower_rows] = (
            self.derivative_pair_packed_hartree_per_bohr
        )
        diagonal = self._lower_rows == self._lower_columns
        derivative[..., self._lower_rows[diagonal], self._lower_columns[diagonal]] = (
            self.derivative_pair_packed_hartree_per_bohr[..., diagonal]
        )
        derivative.setflags(write=False)
        return derivative

    def apply_derivative(self, atom: int, component: int, vector: np.ndarray) -> np.ndarray:
        """Apply one pair-packed symmetric derivative without densification."""

        values = self.derivative_pair_packed_hartree_per_bohr[atom, component]
        source = np.asarray(vector)
        result = np.zeros_like(source, dtype=np.result_type(values, source))
        diagonal = self._lower_rows == self._lower_columns
        np.add.at(
            result,
            self._lower_rows,
            values * source[self._lower_columns],
        )
        off_diagonal = ~diagonal
        np.add.at(
            result,
            self._lower_columns[off_diagonal],
            values[off_diagonal] * source[self._lower_rows[off_diagonal]],
        )
        return result

    def bilinear_derivative(
        self,
        atom: int,
        component: int,
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        """Contract ``left.T @ A^R @ right`` in pair-packed form."""

        return float(np.asarray(left) @ self.apply_derivative(atom, component, right))


def _lowdin_coefficients(state: GFN2OrbitalState) -> np.ndarray:
    values, vectors = np.linalg.eigh(state.overlap)
    if np.min(values) <= 0.0:
        raise ValueError("sTDA-xTB requires a positive-definite overlap")
    return (vectors * np.sqrt(values)) @ vectors.T @ state.coefficients


def _pair_populations(
    coefficients: np.ndarray, ao_atom: np.ndarray, orbitals: np.ndarray
) -> np.ndarray:
    result = np.empty((len(orbitals), len(orbitals), int(np.max(ao_atom)) + 1))
    for atom in range(result.shape[2]):
        block = coefficients[ao_atom == atom][:, orbitals]
        result[:, :, atom] = block.T @ block
    return result


def _screened_kernel_with_derivative(
    coordinates: np.ndarray,
    screened_hardness: np.ndarray,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    kernel = 1.0 / (
        distance**exponent + screened_hardness ** (-exponent)
    ) ** (1.0 / exponent)
    atoms = len(coordinates)
    derivative = np.zeros((atoms, 3, atoms, atoms), dtype=float)
    for left in range(atoms):
        for right in range(left):
            radius = distance[left, right]
            if radius <= np.finfo(float).tiny:
                continue
            radial = -(radius ** (exponent - 1.0)) * kernel[
                left, right
            ] ** (exponent + 1.0)
            vector = radial * displacement[left, right] / radius
            derivative[left, :, left, right] += vector
            derivative[left, :, right, left] += vector
            derivative[right, :, left, right] -= vector
            derivative[right, :, right, left] -= vector
    return kernel, derivative


def closed_shell_stda_matrix_derivative(
    state: GFN2OrbitalState,
    result: ClosedShellSTDAResult,
    overlap_derivative: np.ndarray,
    coefficient_derivative: np.ndarray,
    orbital_energy_derivative: np.ndarray,
    *,
    spectral_range_ev: float = 7.0,
    maximum_configuration_energy_ev: float | None = None,
) -> ClosedShellSTDAMatrixDerivative:
    """Differentiate the literal fixed-selection ``std2 -xtb`` TDA matrix."""

    from .sf_tda_gradient import _pair_populations as derivative_pair_populations
    from .sf_tda_gradient import _square_root_frechet_derivatives

    ds = np.asarray(overlap_derivative, dtype=float)
    dc = np.asarray(coefficient_derivative, dtype=float)
    de = np.asarray(orbital_energy_derivative, dtype=float)
    expected = (len(state.atomic_numbers), 3, state.nao, state.nao)
    if ds.shape != expected or dc.shape != (*expected[:2], state.nao, state.nmo):
        raise ValueError("closed-shell sTDA orbital derivatives have incompatible shapes")
    if de.shape != (*expected[:2], state.nmo):
        raise ValueError("closed-shell sTDA energy derivatives have incompatible shapes")

    occupation = state.alpha_occupations + state.beta_occupations
    occupied_all = np.flatnonzero(occupation > 1.99)
    virtual_all = np.flatnonzero(occupation < 0.01)
    energies = np.asarray(state.orbital_energies_hartree)
    half_width = 2.0 * (1.0 + 0.8 * 0.5) * float(spectral_range_ev) / HARTREE_TO_EV
    occupied = occupied_all[
        energies[occupied_all] > np.min(energies[virtual_all]) - half_width
    ]
    virtual = virtual_all[
        energies[virtual_all] < np.max(energies[occupied_all]) + half_width
    ]
    if result.selection is not None:
        occupied = np.asarray(result.selection.occupied, dtype=int)
        virtual = np.asarray(result.selection.virtual, dtype=int)
        if not np.all(np.isin(occupied, occupied_all)) or not np.all(np.isin(virtual, virtual_all)):
            raise ValueError("frozen sTDA orbitals changed occupation")
    configuration_occupied = np.repeat(occupied, len(virtual))
    configuration_virtual = np.tile(virtual, len(occupied))
    lookup = {
        (int(i), int(a)): index
        for index, (i, a) in enumerate(
            zip(configuration_occupied, configuration_virtual, strict=True)
        )
    }
    try:
        selected = np.asarray(
            [
                lookup[(int(i), int(a))]
                for i, a in zip(
                    result.occupied_orbitals, result.virtual_orbitals, strict=True
                )
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("closed-shell sTDA result does not match its orbital window") from error

    square_root, derivative_root = _square_root_frechet_derivatives(
        state.overlap, ds
    )
    lowdin = square_root @ state.coefficients
    derivative_lowdin = np.einsum(
        "...pq,qr->...pr", derivative_root, state.coefficients, optimize=True
    )
    derivative_lowdin += np.einsum(
        "pq,...qr->...pr", square_root, dc, optimize=True
    )
    atoms = len(state.atomic_numbers)
    qia = np.empty((len(configuration_occupied), atoms), dtype=float)
    dqia = np.empty((*expected[:2], len(configuration_occupied), atoms), dtype=float)
    for atom in range(atoms):
        mask = state.ao_atom == atom
        left = lowdin[mask][:, configuration_occupied]
        right = lowdin[mask][:, configuration_virtual]
        dleft = derivative_lowdin[:, :, mask][:, :, :, configuration_occupied]
        dright = derivative_lowdin[:, :, mask][:, :, :, configuration_virtual]
        qia[:, atom] = np.sum(left * right, axis=0)
        dqia[:, :, :, atom] = np.einsum(
            "...mi,mi->...i", dleft, right, optimize=True
        ) + np.einsum("mi,...mi->...i", left, dright, optimize=True)
    qij_atom, dqij_atom = derivative_pair_populations(
        lowdin, state.ao_atom, occupied, derivative_lowdin
    )
    qab_atom, dqab_atom = derivative_pair_populations(
        lowdin, state.ao_atom, virtual, derivative_lowdin
    )
    qij = np.moveaxis(qij_atom, 0, -1)
    qab = np.moveaxis(qab_atom, 0, -1)
    dqij = np.moveaxis(dqij_atom, 2, -1)
    dqab = np.moveaxis(dqab_atom, 2, -1)

    hardness = grimme_stda_atomic_hardnesses()[state.atomic_numbers - 1]
    average = 0.5 * (hardness[:, None] + hardness[None, :])
    gamma_j, dgamma_j = _screened_kernel_with_derivative(
        state.coordinates_bohr, 0.5 * average, 4.0
    )
    gamma_k, dgamma_k = _screened_kernel_with_derivative(
        state.coordinates_bohr, average, 2.0
    )
    exchange = 2.0 * np.einsum("xA,AB,yB->xy", qia, gamma_k, qia, optimize=True)
    dexchange = 2.0 * (
        np.einsum("...xA,AB,yB->...xy", dqia, gamma_k, qia, optimize=True)
        + np.einsum("xA,...AB,yB->...xy", qia, dgamma_k, qia, optimize=True)
        + np.einsum("xA,AB,...yB->...xy", qia, gamma_k, dqia, optimize=True)
    )
    coulomb_tensor = np.einsum(
        "ijA,AB,abB->iajb", qij, gamma_j, qab, optimize=True
    )
    coulomb = coulomb_tensor.reshape(exchange.shape)
    dcoulomb = (
        np.einsum("...ijA,AB,abB->...iajb", dqij, gamma_j, qab, optimize=True)
        + np.einsum("ijA,...AB,abB->...iajb", qij, dgamma_j, qab, optimize=True)
        + np.einsum("ijA,AB,...abB->...iajb", qij, gamma_j, dqab, optimize=True)
    ).reshape((*expected[:2], *exchange.shape))
    response = exchange - coulomb
    response_derivative = dexchange - dcoulomb
    gaps = energies[configuration_virtual] - energies[configuration_occupied]
    gap_derivative = (
        de[:, :, configuration_virtual] - de[:, :, configuration_occupied]
    )
    exchange_diagonal = np.einsum(
        "xA,AB,xB->x", qia, gamma_k, qia, optimize=True
    )
    dexchange_diagonal = (
        np.einsum("...xA,AB,xB->...x", dqia, gamma_k, qia, optimize=True)
        + np.einsum("xA,...AB,xB->...x", qia, dgamma_k, qia, optimize=True)
        + np.einsum("xA,AB,...xB->...x", qia, gamma_k, dqia, optimize=True)
    )
    shift_max = 0.500 / HARTREE_TO_EV
    shift_width = 0.10 / HARTREE_TO_EV
    ratio = exchange_diagonal / shift_width
    shift = shift_max / (1.0 + ratio**4)
    shift_derivative = (
        -4.0
        * shift_max
        * ratio**3
        / (shift_width * (1.0 + ratio**4) ** 2)
        * dexchange_diagonal
    )
    diagonal = gaps + np.diag(response) + shift
    diagonal_derivative = gap_derivative + np.diagonal(
        response_derivative, axis1=-2, axis2=-1
    ) + shift_derivative

    primary = selected[: result.primary_count]
    included = selected[result.primary_count :]
    threshold = float(spectral_range_ev) / HARTREE_TO_EV
    maximum = (
        float("inf")
        if maximum_configuration_energy_ev is None
        else float(maximum_configuration_energy_ev) / HARTREE_TO_EV
    )
    candidates = np.flatnonzero((diagonal > threshold) & (diagonal < maximum))
    rejected = candidates[~np.isin(candidates, included)]
    if result.selection is not None:
        rejected = np.asarray([lookup[pair] for pair in result.selection.rejected_pairs], dtype=int)
    selection = ClosedShellSTDASelection(tuple(map(int, occupied)), tuple(map(int, virtual)),
        tuple((int(configuration_occupied[k]), int(configuration_virtual[k])) for k in rejected))
    primary_diagonal = diagonal[primary].copy()
    primary_derivative = diagonal_derivative[:, :, primary].copy()
    if len(rejected):
        coupling = response[np.ix_(rejected, primary)]
        coupling_derivative = response_derivative[:, :, rejected][
            :, :, :, primary
        ]
        denominator = diagonal[rejected, None] - diagonal[primary][None, :] + 1.0e-10
        denominator_derivative = (
            diagonal_derivative[:, :, rejected, None]
            - diagonal_derivative[:, :, None, primary]
        )
        lowering = coupling**2 / denominator
        lowering_derivative = (
            2.0 * coupling[None, None] * coupling_derivative / denominator
            - coupling[None, None] ** 2
            * denominator_derivative
            / denominator**2
        )
        primary_diagonal -= np.sum(lowering, axis=0)
        primary_derivative -= np.sum(lowering_derivative, axis=2)

    matrix = response[np.ix_(selected, selected)].copy()
    derivative = response_derivative[:, :, selected][:, :, :, selected].copy()
    selected_diagonal = np.concatenate((primary_diagonal, diagonal[included]))
    selected_derivative = np.concatenate(
        (primary_derivative, diagonal_derivative[:, :, included]), axis=2
    )
    indices = np.arange(len(selected))
    matrix[indices, indices] = selected_diagonal
    derivative[:, :, indices, indices] = selected_derivative
    return ClosedShellSTDAMatrixDerivative(matrix, derivative, selection=selection)


def grimme_xtb_stda(
    state: GFN2OrbitalState,
    *,
    roots: int = 5,
    spectral_range_ev: float = 7.0,
    perturbative_threshold_hartree: float = 1.0e-4,
    maximum_configuration_energy_ev: float | None = None,
    selection_reference: ClosedShellSTDAResult | None = None,
) -> ClosedShellSTDAResult:
    """Evaluate the literal singlet TDA *kernel* of ``std2 -xtb``.

    The implementation follows the closed-shell ``stda.f`` construction:
    singlet factor two, separate J/K hardness kernels, the xTB K-dependent
    diagonal shift, and the perturbative secondary-space selection.  Absolute
    parity with Grimme additionally requires its ``xtb4stda`` orbital
    reference, loaded by :func:`read_wfn_xtb`; a resident MATRIX-GFN2 orbital
    state intentionally represents a different ground-state parametrization.
    """

    if roots < 1 or not np.isfinite(spectral_range_ev) or spectral_range_ev <= 0.0:
        raise ValueError("sTDA roots and spectral range must be positive and finite")
    if state.reference_kind != "restricted-closed-shell":
        raise ValueError("ordinary sTDA-xTB requires a restricted closed shell")
    if selection_reference is not None:
        if selection_reference.selection is None:
            raise ValueError("sTDA selection reference must be frozen explicitly")
        shape = (len(state.atomic_numbers), 3)
        matrix = closed_shell_stda_matrix_derivative(state, selection_reference,
            np.zeros((*shape, state.nao, state.nao)),
            np.zeros((*shape, state.nao, state.nmo)), np.zeros((*shape, state.nmo)),
            spectral_range_ev=spectral_range_ev,
            maximum_configuration_energy_ev=maximum_configuration_energy_ev).matrix_hartree
        values, vectors = np.linalg.eigh(matrix)
        count = min(int(roots), len(values))
        residual = np.linalg.norm(matrix @ vectors[:, :count] - vectors[:, :count] * values[:count], axis=0)
        return replace(selection_reference, excitation_energies_hartree=values[:count],
            amplitudes=vectors[:, :count].T, residual_norms_hartree=residual,
            converged=bool(np.max(residual) < 1e-10), solver_backend="frozen-selection-lapack")
    if state.reference_kind != "restricted-closed-shell":
        raise ValueError("ordinary sTDA-xTB requires a restricted closed shell")
    if roots < 1 or spectral_range_ev <= 0.0:
        raise ValueError("sTDA roots and spectral range must be positive")
    occupation = state.alpha_occupations + state.beta_occupations
    occupied_all = np.flatnonzero(occupation > 1.99)
    virtual_all = np.flatnonzero(occupation < 0.01)
    energies = np.asarray(state.orbital_energies_hartree)
    exact_exchange = 0.5
    half_width = (
        2.0
        * (1.0 + 0.8 * exact_exchange)
        * float(spectral_range_ev)
        / HARTREE_TO_EV
    )
    occupied = occupied_all[
        energies[occupied_all] > np.min(energies[virtual_all]) - half_width
    ]
    virtual = virtual_all[
        energies[virtual_all] < np.max(energies[occupied_all]) + half_width
    ]
    if not len(occupied) or not len(virtual):
        raise ValueError("no configurations survive the closed-shell sTDA window")
    lowdin = _lowdin_coefficients(state)
    atom_count = len(state.atomic_numbers)
    qia = np.empty((len(occupied) * len(virtual), atom_count), dtype=float)
    configuration_occupied = np.repeat(occupied, len(virtual))
    configuration_virtual = np.tile(virtual, len(occupied))
    for atom in range(atom_count):
        block = lowdin[state.ao_atom == atom]
        qia[:, atom] = np.sum(
            block[:, configuration_occupied] * block[:, configuration_virtual],
            axis=0,
        )
    qij = _pair_populations(lowdin, state.ao_atom, occupied)
    qab = _pair_populations(lowdin, state.ao_atom, virtual)
    hardness = grimme_stda_atomic_hardnesses()[state.atomic_numbers - 1]
    average = 0.5 * (hardness[:, None] + hardness[None, :])
    displacement = state.coordinates_bohr[:, None, :] - state.coordinates_bohr[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    # ``std2 -xtb`` overrides the generic density-functional interpolation.
    beta_j = 4.0
    alpha_k = 2.0
    screened_j = average * exact_exchange
    gamma_j = 1.0 / (
        distance**beta_j + screened_j ** (-beta_j)
    ) ** (1.0 / beta_j)
    gamma_k = 1.0 / (
        distance**alpha_k + average ** (-alpha_k)
    ) ** (1.0 / alpha_k)
    gaps = energies[configuration_virtual] - energies[configuration_occupied]
    threshold = float(spectral_range_ev) / HARTREE_TO_EV
    maximum = (
        float("inf")
        if maximum_configuration_energy_ev is None
        else float(maximum_configuration_energy_ev) / HARTREE_TO_EV
    )
    if _gfn2_native is None or not hasattr(_gfn2_native, "closed_shell_stda"):
        raise RuntimeError("closed-shell sTDA requires the compiled MATRIX-GFN2 library")
    (
        values,
        vectors,
        selected,
        primary_count,
        perturbative_count,
        solver_backend,
        operator_batches,
    ) = (
        _gfn2_native.closed_shell_stda(
            qia,
            qij,
            qab,
            gamma_j,
            gamma_k,
            gaps,
            threshold,
            maximum,
            float(perturbative_threshold_hartree),
            min(int(roots), len(qia)),
        )
    )
    selected = np.asarray(selected, dtype=np.int64)
    return ClosedShellSTDAResult(
        np.asarray(values),
        np.asarray(vectors),
        configuration_occupied[selected],
        configuration_virtual[selected],
        int(primary_count),
        int(perturbative_count),
        np.zeros(len(values)),
        True,
        str(solver_backend),
        int(operator_batches),
        state.model_id,
        state.model_id.startswith("GRIMME-XTB4STDA-STDA2-WFN."),
    )


__all__ = [
    "ClosedShellSTDAMatrixDerivative",
    "ClosedShellSTDAResult",
    "ClosedShellSTDASelection",
    "freeze_closed_shell_stda_selection",
    "closed_shell_stda_matrix_derivative",
    "grimme_xtb_stda",
]


def freeze_closed_shell_stda_selection(state, result, *, spectral_range_ev=7.0,
                                     maximum_configuration_energy_ev=None):
    """Keep the initial parent, primary, secondary and rejected spaces fixed."""
    if result.selection is not None:
        return result
    shape = (len(state.atomic_numbers), 3)
    matrix = closed_shell_stda_matrix_derivative(state, result,
        np.zeros((*shape, state.nao, state.nao)), np.zeros((*shape, state.nao, state.nmo)),
        np.zeros((*shape, state.nmo)), spectral_range_ev=spectral_range_ev,
        maximum_configuration_energy_ev=maximum_configuration_energy_ev)
    return replace(result, selection=matrix.selection)
