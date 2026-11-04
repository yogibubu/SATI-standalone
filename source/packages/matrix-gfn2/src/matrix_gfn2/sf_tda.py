"""Matrix-free foundations for resident spin-flip Tamm--Dancoff theory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Callable

import numpy as np
from matrix_numerics import solve_symmetric_matrix_free

from .electronic_state import GFN2OrbitalState
from .sf_tda_contract import load_sf_tda_contract


VectorKernel = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SFTDAKernelParameters:
    profile_id: str
    exact_exchange_fraction: float
    hardness_scale: float
    distance_exponent: float


GRIMME_XTB_SF_STDA = SFTDAKernelParameters(
    profile_id="grimme-xtb-sf-stda",
    exact_exchange_fraction=0.36,
    hardness_scale=0.36 * 1.4,
    distance_exponent=3.0,
)

HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class FrozenSpinFlipSelection:
    """MO rectangle and fixed configuration roles of an accepted SF PES."""
    occupied: tuple[int, ...]
    virtual: tuple[int, ...]
    primary: tuple[int, ...]
    included: tuple[int, ...]
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class SFTDASelectionPolicy:
    """Literal configuration-selection policy of Grimme's SF-sTDA code."""

    spectral_range_ev: float = 7.0
    perturbative_threshold_hartree: float = 1.0e-4
    maximum_configuration_energy_ev: float | None = None
    frozen_selection: FrozenSpinFlipSelection | None = None

    def __post_init__(self) -> None:
        spectral = float(self.spectral_range_ev)
        perturbative = float(self.perturbative_threshold_hartree)
        maximum = self.maximum_configuration_energy_ev
        if not np.isfinite(spectral) or spectral <= 0.0:
            raise ValueError("SF-TDA spectral range must be positive and finite")
        if not np.isfinite(perturbative) or perturbative < 0.0:
            raise ValueError("SF-TDA perturbative threshold must be finite and nonnegative")
        if maximum is not None and (not np.isfinite(maximum) or maximum <= spectral):
            raise ValueError("maximum CSF energy must exceed the primary spectral range")

    @property
    def spectral_range_hartree(self) -> float:
        return float(self.spectral_range_ev) / HARTREE_TO_EV

    @property
    def maximum_configuration_energy_hartree(self) -> float:
        if self.maximum_configuration_energy_ev is None:
            return float("inf")
        return float(self.maximum_configuration_energy_ev) / HARTREE_TO_EV


@lru_cache(maxsize=1)
def grimme_stda_atomic_hardnesses() -> np.ndarray:
    resource = files("matrix_gfn2").joinpath("data", "grimme-stda-hardness.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    values = np.asarray(document["hardness_hartree"], dtype=float)
    if values.shape != (94,) or np.any(values <= 0.0):
        raise RuntimeError("invalid Grimme sTDA atomic-hardness table")
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class SpinFlipExcitationSpace:
    """The alpha-occupied to beta-virtual spin-flip space."""

    occupied_alpha: np.ndarray
    virtual_beta: np.ndarray
    orbital_gaps_hartree: np.ndarray

    def __post_init__(self) -> None:
        occupied = np.array(self.occupied_alpha, dtype=np.int64, copy=True)
        virtual = np.array(self.virtual_beta, dtype=np.int64, copy=True)
        gaps = np.array(self.orbital_gaps_hartree, dtype=float, copy=True)
        if occupied.ndim != 1 or virtual.ndim != 1 or not len(occupied) or not len(virtual):
            raise ValueError("spin-flip space requires occupied-alpha and virtual-beta orbitals")
        if gaps.shape != (len(occupied), len(virtual)):
            raise ValueError("spin-flip gap matrix has inconsistent dimensions")
        if not np.all(np.isfinite(gaps)):
            raise ValueError("spin-flip orbital gaps must be finite")
        occupied.setflags(write=False)
        virtual.setflags(write=False)
        gaps.setflags(write=False)
        object.__setattr__(self, "occupied_alpha", occupied)
        object.__setattr__(self, "virtual_beta", virtual)
        object.__setattr__(self, "orbital_gaps_hartree", gaps)

    @classmethod
    def from_orbital_state(
        cls,
        state: GFN2OrbitalState,
        *,
        occupation_tolerance: float = 1.0e-7,
        maximum_gap_hartree: float | None = None,
    ) -> "SpinFlipExcitationSpace":
        contract = load_sf_tda_contract()
        if state.reference_kind != contract.reference["kind"]:
            raise ValueError("SF-TDA-MATRIX-GFN2-R requires a restricted open-shell reference")
        tolerance = float(occupation_tolerance)
        if tolerance <= 0.0 or tolerance >= 0.5:
            raise ValueError("occupation tolerance must lie between zero and one half")
        for occupations in (state.alpha_occupations, state.beta_occupations):
            fractional = (occupations > tolerance) & (occupations < 1.0 - tolerance)
            if np.any(fractional):
                raise ValueError(
                    "SF-TDA reference has fractional frontier occupations; rerun "
                    "MATRIX-GFN2 at zero final electronic temperature"
                )
        occupied = np.flatnonzero(state.alpha_occupations >= 1.0 - tolerance)
        virtual = np.flatnonzero(state.beta_occupations <= tolerance)
        gaps = (
            state.orbital_energies_hartree[virtual][None, :]
            - state.orbital_energies_hartree[occupied][:, None]
        )
        if maximum_gap_hartree is not None:
            limit = float(maximum_gap_hartree)
            if not np.isfinite(limit) or limit <= 0.0:
                raise ValueError("maximum spin-flip gap must be positive and finite")
            keep = gaps <= limit
            occupied = occupied[np.any(keep, axis=1)]
            virtual = virtual[np.any(keep, axis=0)]
            gaps = (
                state.orbital_energies_hartree[virtual][None, :]
                - state.orbital_energies_hartree[occupied][:, None]
            )
        return cls(occupied, virtual, gaps)

    @classmethod
    def from_grimme_window(
        cls,
        state: GFN2OrbitalState,
        policy: SFTDASelectionPolicy,
        *,
        occupation_tolerance: float = 1.0e-7,
    ) -> "SpinFlipExcitationSpace":
        """Build the active MO rectangle used by ``stda -xtb -sf``.

        Grimme's code doubles ``(1 + 0.8 ax)`` times the requested spectral
        range, then retains occupied orbitals above ``vmin - deps`` and virtual
        orbitals below ``omax + deps``.  The shared restricted orbitals are
        interpreted as separate alpha and beta spin channels here.
        """

        full = cls.from_orbital_state(state, occupation_tolerance=occupation_tolerance)
        if policy.frozen_selection is not None:
            occupied = np.asarray(policy.frozen_selection.occupied, dtype=int)
            virtual = np.asarray(policy.frozen_selection.virtual, dtype=int)
            if not np.all(np.isin(occupied, full.occupied_alpha)) or not np.all(np.isin(virtual, full.virtual_beta)):
                raise ValueError("frozen SF-TDA orbitals changed occupation")
            gaps = state.orbital_energies_hartree[virtual][None, :] - state.orbital_energies_hartree[occupied][:, None]
            return cls(occupied, virtual, gaps)

        alpha_occupied = np.flatnonzero(state.alpha_occupations >= 1.0 - occupation_tolerance)
        beta_occupied = np.flatnonzero(state.beta_occupations >= 1.0 - occupation_tolerance)
        alpha_virtual = np.flatnonzero(state.alpha_occupations <= occupation_tolerance)
        beta_virtual = np.flatnonzero(state.beta_occupations <= occupation_tolerance)
        occupied_all = np.concatenate((alpha_occupied, beta_occupied))
        virtual_all = np.concatenate((alpha_virtual, beta_virtual))
        if not len(occupied_all) or not len(virtual_all):
            raise ValueError("SF-TDA active-window construction needs occupied and virtual MOs")
        energies = state.orbital_energies_hartree
        occupied_ceiling = float(np.max(energies[occupied_all]))
        virtual_floor = float(np.min(energies[virtual_all]))
        half_width = (
            2.0
            * (1.0 + 0.8 * GRIMME_XTB_SF_STDA.exact_exchange_fraction)
            * policy.spectral_range_hartree
        )
        occupied_floor = virtual_floor - half_width
        virtual_ceiling = occupied_ceiling + half_width
        occupied = full.occupied_alpha[energies[full.occupied_alpha] > occupied_floor]
        virtual = full.virtual_beta[energies[full.virtual_beta] < virtual_ceiling]
        if not len(occupied) or not len(virtual):
            raise ValueError("no spin-flip configurations survive the Grimme active window")
        gaps = energies[virtual][None, :] - energies[occupied][:, None]
        return cls(occupied, virtual, gaps)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.occupied_alpha), len(self.virtual_beta)

    @property
    def size(self) -> int:
        occupied, virtual = self.shape
        return occupied * virtual

    @property
    def diagonal_hartree(self) -> np.ndarray:
        return self.orbital_gaps_hartree.reshape(-1).copy()

    def reshape_vector(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.size,):
            raise ValueError("spin-flip vector has the wrong dimension")
        return value.reshape(self.shape)

    def configuration_orbitals(self) -> tuple[np.ndarray, np.ndarray]:
        occupied = np.repeat(self.occupied_alpha, len(self.virtual_beta))
        virtual = np.tile(self.virtual_beta, len(self.occupied_alpha))
        return occupied, virtual


@dataclass(frozen=True)
class SelectedSpinFlipSpace:
    """Non-rectangular primary plus perturbatively selected SF-CSF space."""

    parent: SpinFlipExcitationSpace
    parent_indices: np.ndarray
    diagonal_hartree: np.ndarray
    primary_count: int
    perturbative_count: int
    selection_reference: FrozenSpinFlipSelection | None = None

    def __post_init__(self) -> None:
        indices = np.array(self.parent_indices, dtype=np.int64, copy=True)
        diagonal = np.array(self.diagonal_hartree, dtype=float, copy=True)
        if indices.ndim != 1 or not len(indices) or diagonal.shape != indices.shape:
            raise ValueError("selected SF-TDA space has inconsistent dimensions")
        if np.any(indices < 0) or np.any(indices >= self.parent.size):
            raise ValueError("selected SF-TDA indices lie outside the parent space")
        if len(np.unique(indices)) != len(indices) or not np.all(np.isfinite(diagonal)):
            raise ValueError("selected SF-TDA configurations must be unique and finite")
        if int(self.primary_count) < 1 or self.primary_count + self.perturbative_count != len(indices):
            raise ValueError("selected SF-TDA primary/PT counts are inconsistent")
        indices.setflags(write=False)
        diagonal.setflags(write=False)
        object.__setattr__(self, "parent_indices", indices)
        object.__setattr__(self, "diagonal_hartree", diagonal)

    @property
    def size(self) -> int:
        return len(self.parent_indices)

    def expand_amplitudes(self, amplitudes: np.ndarray) -> np.ndarray:
        values = np.asarray(amplitudes, dtype=float)
        if values.shape[-1] != self.size:
            raise ValueError("selected SF-TDA amplitudes have the wrong dimension")
        expanded = np.zeros((*values.shape[:-1], self.parent.size), dtype=float)
        expanded[..., self.parent_indices] = values
        return expanded

    def configuration_orbitals(self) -> tuple[np.ndarray, np.ndarray]:
        occupied, virtual = self.parent.configuration_orbitals()
        return occupied[self.parent_indices], virtual[self.parent_indices]


@dataclass(frozen=True)
class SFTDAResult:
    excitation_energies_hartree: np.ndarray
    amplitudes: np.ndarray
    residual_norms: np.ndarray
    iterations: int
    converged: bool
    transition_properties: tuple[SFTransitionProperties, ...] = ()
    solver_backend: str = "python-davidson"
    selection_reference: FrozenSpinFlipSelection | None = None

    def __post_init__(self) -> None:
        energies = np.array(self.excitation_energies_hartree, dtype=float, copy=True)
        amplitudes = np.array(self.amplitudes, dtype=float, copy=True)
        residuals = np.array(self.residual_norms, dtype=float, copy=True)
        if energies.ndim != 1 or amplitudes.shape[0] != len(energies):
            raise ValueError("SF-TDA result dimensions are inconsistent")
        if residuals.shape != energies.shape:
            raise ValueError("one residual norm is required for every SF-TDA root")
        if self.transition_properties and len(self.transition_properties) != len(energies):
            raise ValueError("one electronic-property record is required for every SF-TDA root")
        if not all(np.all(np.isfinite(item)) for item in (energies, amplitudes, residuals)):
            raise ValueError("SF-TDA result arrays must be finite")
        for item in (energies, amplitudes, residuals):
            item.setflags(write=False)
        object.__setattr__(self, "excitation_energies_hartree", energies)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "residual_norms", residuals)


@dataclass(frozen=True)
class SFTransitionProperties:
    """Electronic descriptors for one spin-flip TDA root."""

    transition_density_ao: np.ndarray
    difference_density_ao: np.ndarray
    nto_singular_values: np.ndarray
    occupied_nto_coefficients: np.ndarray
    virtual_nto_coefficients: np.ndarray
    dominant_configurations: tuple[tuple[int, int, float], ...]
    participation_ratio: float
    spin_square: float
    delta_ms: float = -1.0
    electric_dipole_allowed: bool = False
    oscillator_strength: float = 0.0

    def __post_init__(self) -> None:
        transition = np.array(self.transition_density_ao, dtype=float, copy=True)
        difference = np.array(self.difference_density_ao, dtype=float, copy=True)
        singular = np.array(self.nto_singular_values, dtype=float, copy=True)
        occupied = np.array(self.occupied_nto_coefficients, dtype=float, copy=True)
        virtual = np.array(self.virtual_nto_coefficients, dtype=float, copy=True)
        if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
            raise ValueError("SF transition density must be a square AO matrix")
        if difference.shape != transition.shape:
            raise ValueError("SF difference and transition densities use different AO spaces")
        if occupied.shape[0] != transition.shape[0] or virtual.shape[0] != transition.shape[0]:
            raise ValueError("SF NTO coefficients use an incompatible AO space")
        if occupied.shape[1] != len(singular) or virtual.shape[1] != len(singular):
            raise ValueError("SF NTO coefficient and singular-value dimensions differ")
        if not all(np.all(np.isfinite(item)) for item in (transition, difference, singular, occupied, virtual)):
            raise ValueError("SF electronic properties must be finite")
        if self.electric_dipole_allowed or abs(float(self.oscillator_strength)) > 1.0e-15:
            raise ValueError("nonrelativistic reference-to-spin-flip electric transitions are forbidden")
        if not np.isfinite(self.spin_square) or self.spin_square < -1.0e-10:
            raise ValueError("SF-TDA spin-square diagnostic must be finite and nonnegative")
        for item in (transition, difference, singular, occupied, virtual):
            item.setflags(write=False)
        object.__setattr__(self, "transition_density_ao", transition)
        object.__setattr__(self, "difference_density_ao", difference)
        object.__setattr__(self, "nto_singular_values", singular)
        object.__setattr__(self, "occupied_nto_coefficients", occupied)
        object.__setattr__(self, "virtual_nto_coefficients", virtual)


class SpinFlipTDAOperator:
    """Symmetric matrix-free SF-TDA operator with an explicit response kernel."""

    def __init__(
        self,
        space: SpinFlipExcitationSpace,
        response_kernel: VectorKernel,
        *,
        diagonal_correction_hartree: np.ndarray | None = None,
        block_response_kernel: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if not callable(response_kernel):
            raise TypeError("SF-TDA requires an explicit response kernel")
        self.space = space
        self._kernel = response_kernel
        self._block_kernel = block_response_kernel
        diagonal = space.diagonal_hartree
        if diagonal_correction_hartree is not None:
            correction = np.asarray(diagonal_correction_hartree, dtype=float)
            if correction.shape != diagonal.shape or not np.all(np.isfinite(correction)):
                raise ValueError("SF-TDA diagonal correction has the wrong shape")
            diagonal = diagonal + correction
        diagonal.setflags(write=False)
        self.diagonal_hartree = diagonal

    @property
    def size(self) -> int:
        return self.space.size

    def apply(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.size,):
            raise ValueError("SF-TDA trial vector has the wrong dimension")
        response = np.asarray(self._kernel(value), dtype=float)
        if response.shape != value.shape or not np.all(np.isfinite(response)):
            raise ValueError("SF-TDA response kernel returned an invalid vector")
        return self.diagonal_hartree * value + response

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.size:
            raise ValueError("SF-TDA trial block has the wrong dimension")
        if self._block_kernel is None:
            return np.column_stack([self.apply(values[:, index]) for index in range(values.shape[1])])
        response = np.asarray(self._block_kernel(values), dtype=float)
        if response.shape != values.shape or not np.all(np.isfinite(response)):
            raise ValueError("SF-TDA block response kernel returned an invalid block")
        return self.diagonal_hartree[:, None] * values + response


class SelectedSpinFlipTDAOperator:
    """Scatter/gather view of a matrix-free operator on selected CSFs."""

    def __init__(
        self,
        space: SelectedSpinFlipSpace,
        response_kernel: VectorKernel,
        response_diagonal_hartree: np.ndarray,
        block_response_kernel: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if not callable(response_kernel):
            raise TypeError("selected SF-TDA requires an explicit response kernel")
        self.space = space
        self._kernel = response_kernel
        self._block_kernel = block_response_kernel
        response_diagonal = np.asarray(response_diagonal_hartree, dtype=float)
        if response_diagonal.shape != (space.parent.size,) or not np.all(
            np.isfinite(response_diagonal)
        ):
            raise ValueError("selected SF-TDA response diagonal has the wrong dimension")
        self._response_diagonal = response_diagonal.copy()
        self._response_diagonal.setflags(write=False)
        self.diagonal_hartree = space.diagonal_hartree

    @property
    def size(self) -> int:
        return self.space.size

    def apply(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.size,):
            raise ValueError("selected SF-TDA trial vector has the wrong dimension")
        expanded = self.space.expand_amplitudes(value)
        response = np.asarray(self._kernel(expanded), dtype=float)
        if response.shape != (self.space.parent.size,) or not np.all(np.isfinite(response)):
            raise ValueError("SF-TDA response kernel returned an invalid vector")
        indices = self.space.parent_indices
        off_diagonal_response = (
            response[indices] - self._response_diagonal[indices] * value
        )
        return self.diagonal_hartree * value + off_diagonal_response

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.size:
            raise ValueError("selected SF-TDA trial block has the wrong dimension")
        expanded = np.zeros((self.space.parent.size, values.shape[1]), dtype=float)
        expanded[self.space.parent_indices] = values
        if self._block_kernel is None:
            return np.column_stack([self.apply(values[:, index]) for index in range(values.shape[1])])
        response = np.asarray(self._block_kernel(expanded), dtype=float)
        if response.shape != expanded.shape or not np.all(np.isfinite(response)):
            raise ValueError("selected SF-TDA block response kernel returned an invalid block")
        indices = self.space.parent_indices
        off_diagonal = response[indices] - self._response_diagonal[indices, None] * values
        return self.diagonal_hartree[:, None] * values + off_diagonal


class AtomicTransitionChargeResponse:
    """Factorized SF response from atom-resolved Lowdin pair populations.

    The interaction form follows the screened atom-pair structure used by
    simplified TDA models, but its scale and exponent are mandatory inputs.
    They are deliberately not defaulted: fitting these parameters is a
    separate scientific qualification gate for MATRIX-GFN2.
    """

    def __init__(
        self,
        state: GFN2OrbitalState,
        space: SpinFlipExcitationSpace,
        atomic_hardness_hartree: np.ndarray,
        *,
        hardness_scale: float,
        distance_exponent: float,
    ) -> None:
        hardness = np.asarray(atomic_hardness_hartree, dtype=float)
        if hardness.shape != (len(state.atomic_numbers),):
            raise ValueError("one response hardness is required for every atom")
        if np.any(~np.isfinite(hardness)) or np.any(hardness <= 0.0):
            raise ValueError("response hardnesses must be positive and finite")
        scale = float(hardness_scale)
        exponent = float(distance_exponent)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("response hardness scale must be positive and finite")
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("response distance exponent must be positive and finite")

        eigenvalues, eigenvectors = np.linalg.eigh(state.overlap)
        if float(np.min(eigenvalues)) <= 0.0:
            raise ValueError("SF-TDA Lowdin populations require positive-definite overlap")
        square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        lowdin = square_root @ state.coefficients
        occupied = lowdin[:, space.occupied_alpha]
        virtual = lowdin[:, space.virtual_beta]
        atom_index = state.ao_atom
        self._occupied_pairs = tuple(
            occupied[atom_index == atom].T @ occupied[atom_index == atom]
            for atom in range(len(state.atomic_numbers))
        )
        self._virtual_pairs = tuple(
            virtual[atom_index == atom].T @ virtual[atom_index == atom]
            for atom in range(len(state.atomic_numbers))
        )
        displacement = (
            state.coordinates_bohr[:, None, :] - state.coordinates_bohr[None, :, :]
        )
        distance = np.linalg.norm(displacement, axis=2)
        screened_hardness = scale * 0.5 * (hardness[:, None] + hardness[None, :])
        self.gamma_hartree = 1.0 / (
            distance**exponent + screened_hardness ** (-exponent)
        ) ** (1.0 / exponent)
        for matrix in (*self._occupied_pairs, *self._virtual_pairs):
            matrix.setflags(write=False)
        self.gamma_hartree.setflags(write=False)
        self.space = space

    @property
    def diagonal_hartree(self) -> np.ndarray:
        """Return ``-(ii|aa)`` without materializing the response matrix."""

        diagonal = np.zeros(self.space.shape, dtype=float)
        for atom, occupied_pairs in enumerate(self._occupied_pairs):
            occupied_diagonal = np.diag(occupied_pairs)
            for partner, virtual_pairs in enumerate(self._virtual_pairs):
                diagonal -= self.gamma_hartree[atom, partner] * (
                    occupied_diagonal[:, None] * np.diag(virtual_pairs)[None, :]
                )
        return diagonal.reshape(-1)

    def apply(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.space.size,):
            raise ValueError("spin-flip vector has the wrong dimension")
        return self.apply_block(value[:, None])[:, 0]

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the factorized response to a contiguous vector block."""

        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.space.size:
            raise ValueError("spin-flip response block has the wrong dimension")
        amplitudes = values.reshape((*self.space.shape, values.shape[1]))
        result = np.zeros_like(amplitudes)
        for atom, occupied_pairs in enumerate(self._occupied_pairs):
            left = np.einsum("ij,jam->iam", occupied_pairs, amplitudes, optimize=True)
            for partner, virtual_pairs in enumerate(self._virtual_pairs):
                coupling = self.gamma_hartree[atom, partner]
                if coupling != 0.0:
                    result -= coupling * np.einsum(
                        "iam,ab->ibm", left, virtual_pairs, optimize=True
                    )
        return result.reshape((self.space.size, values.shape[1]))

    @classmethod
    def grimme_xtb_spin_flip(
        cls,
        state: GFN2OrbitalState,
        space: SpinFlipExcitationSpace,
    ) -> "AtomicTransitionChargeResponse":
        """Construct the ``stda -xtb -sf`` kernel used by Grimme's code."""

        table = grimme_stda_atomic_hardnesses()
        numbers = np.asarray(state.atomic_numbers, dtype=np.intp)
        if np.any(numbers < 1) or np.any(numbers > len(table)):
            raise ValueError("Grimme sTDA hardness profile supports Z=1 through 94")
        return cls(
            state,
            space,
            table[numbers - 1],
            hardness_scale=GRIMME_XTB_SF_STDA.hardness_scale,
            distance_exponent=GRIMME_XTB_SF_STDA.distance_exponent,
        )

    def materialize_for_validation(self) -> np.ndarray:
        """Build the dense kernel only for small-system parity tests."""

        identity = np.eye(self.space.size)
        return np.column_stack(
            [self.apply(identity[:, index]) for index in range(self.space.size)]
        )


def select_grimme_spin_flip_configurations(
    space: SpinFlipExcitationSpace,
    response: AtomicTransitionChargeResponse,
    policy: SFTDASelectionPolicy,
) -> SelectedSpinFlipSpace:
    """Apply Grimme's energy plus PT2 CSF selection literally.

    Rejected external configurations lower the diagonal of every primary CSF
    by their second-order contribution, exactly as ``sf_ptselect_uks`` does.
    """

    if response.space is not space:
        raise ValueError("configuration selection and response use different SF spaces")
    diagonal = space.diagonal_hartree + response.diagonal_hartree
    primary = np.flatnonzero(diagonal <= policy.spectral_range_hartree)
    if policy.frozen_selection is not None:
        primary = np.asarray(policy.frozen_selection.primary, dtype=int)
    if not len(primary):
        raise ValueError("no primary SF-TDA configurations; increase the spectral range")
    primary = primary[np.argsort(diagonal[primary], kind="stable")]
    candidates = np.flatnonzero(
        (diagonal > policy.spectral_range_hartree)
        & (diagonal < policy.maximum_configuration_energy_hartree)
    )

    if policy.frozen_selection is not None:
        primary = np.asarray(policy.frozen_selection.primary, dtype=int)
        candidates = np.asarray(policy.frozen_selection.candidates, dtype=int)
    primary_diagonal = diagonal[primary].copy()
    included: list[int] = []
    if len(candidates):
        coupling = np.empty((len(candidates), len(primary)), dtype=float)
        for column, configuration in enumerate(primary):
            unit = np.zeros(space.size, dtype=float)
            unit[configuration] = 1.0
            coupling[:, column] = response.apply(unit)[candidates]
        denominators = (
            diagonal[candidates, None] - diagonal[primary][None, :] + 1.0e-10
        )
        contributions = coupling * coupling / denominators
        perturbations = np.sum(contributions, axis=1)
        include_mask = perturbations > policy.perturbative_threshold_hartree
        if policy.frozen_selection is not None:
            include_mask = np.isin(candidates, policy.frozen_selection.included)
        included = candidates[include_mask].tolist()
        primary_diagonal -= np.sum(contributions[~include_mask], axis=0)

    indices = np.concatenate((primary, np.asarray(included, dtype=np.int64)))
    selected_diagonal = np.concatenate((primary_diagonal, diagonal[included]))
    return SelectedSpinFlipSpace(
        parent=space,
        parent_indices=indices,
        diagonal_hartree=selected_diagonal,
        primary_count=len(primary),
        perturbative_count=len(included),
        selection_reference=FrozenSpinFlipSelection(tuple(map(int, space.occupied_alpha)),
            tuple(map(int, space.virtual_beta)), tuple(map(int, primary)),
            tuple(map(int, included)), tuple(map(int, candidates))),
    )


def davidson_sf_tda(
    operator: SpinFlipTDAOperator | SelectedSpinFlipTDAOperator,
    *,
    roots: int = 1,
    tolerance_hartree: float = 1.0e-8,
    maximum_iterations: int = 100,
    maximum_subspace: int | None = None,
    backend: str = "auto",
) -> SFTDAResult:
    """Compute the lowest SF-TDA roots through the shared MATRIX router."""

    solved = solve_symmetric_matrix_free(
        operator.apply,
        operator.diagonal_hartree,
        roots=roots,
        block_matvec=operator.apply_block,
        backend=backend,
        tolerance=tolerance_hartree,
        maximum_iterations=maximum_iterations,
        maximum_subspace=maximum_subspace,
    )
    return SFTDAResult(
        solved.eigenvalues,
        solved.eigenvectors,
        solved.residual_norms,
        solved.operator_batches,
        solved.converged,
        solver_backend=solved.backend,
    )


def spin_flip_transition_properties(
    state: GFN2OrbitalState,
    space: SpinFlipExcitationSpace | SelectedSpinFlipSpace,
    amplitudes: np.ndarray,
    *,
    dominant: int = 5,
) -> tuple[SFTransitionProperties, ...]:
    """Build AO transition/difference densities and NTOs for SF roots.

    The AO transition density is the beta-alpha spin-transition block.  Its
    ordinary electric transition dipole is exactly zero in the nonrelativistic
    spin-pure limit, so no misleading bright-state oscillator strength is
    synthesized here.  APOC can directly use this density for root following.
    """

    values = np.asarray(amplitudes, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    parent = space.parent if isinstance(space, SelectedSpinFlipSpace) else space
    if isinstance(space, SelectedSpinFlipSpace):
        values = space.expand_amplitudes(values)
    if values.shape[1] != parent.size:
        raise ValueError("SF-TDA amplitude dimension does not match its excitation space")
    occupied_indices, virtual_indices = parent.configuration_orbitals()
    occupied_unique = parent.occupied_alpha
    virtual_unique = parent.virtual_beta
    occupied_coefficients = state.coefficients[:, occupied_unique]
    virtual_coefficients = state.coefficients[:, virtual_unique]
    beta_occupied_unique = occupied_unique[
        state.beta_occupations[occupied_unique] >= 1.0 - 1.0e-7
    ]
    beta_occupied_coefficients = state.coefficients[:, beta_occupied_unique]
    occupied_beta_overlap = (
        occupied_coefficients.T @ state.overlap @ beta_occupied_coefficients
    )
    occupied_virtual_overlap = (
        occupied_coefficients.T @ state.overlap @ virtual_coefficients
    )
    ground_overlap_sum = float(np.sum(occupied_beta_overlap**2))
    noa = len(occupied_unique)
    nob = len(beta_occupied_unique)
    count = max(1, int(dominant))
    result: list[SFTransitionProperties] = []
    for vector in values:
        matrix = vector.reshape(parent.shape)
        transition = virtual_coefficients @ matrix.T @ occupied_coefficients.T
        hole_metric = matrix @ matrix.T
        particle_metric = matrix.T @ matrix
        difference = (
            virtual_coefficients @ particle_metric @ virtual_coefficients.T
            - occupied_coefficients @ hole_metric @ occupied_coefficients.T
        )
        left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
        occupied_nto = occupied_coefficients @ left
        virtual_nto = virtual_coefficients @ right_t.T
        order = np.argsort(-(vector * vector), kind="stable")[:count]
        dominant_configurations = tuple(
            (
                int(occupied_indices[index]),
                int(virtual_indices[index]),
                float(vector[index]),
            )
            for index in order
            if abs(float(vector[index])) > 0.0
        )
        fourth_moment = float(np.sum(vector**4))
        participation = 0.0 if fourth_moment == 0.0 else 1.0 / fourth_moment
        xia = float(np.sum(matrix * occupied_virtual_overlap) ** 2)
        virtual_projector = occupied_virtual_overlap.T @ occupied_virtual_overlap
        occupied_projector = occupied_beta_overlap @ occupied_beta_overlap.T
        xiaib = float(np.einsum("ia,ib,ab->", matrix, matrix, virtual_projector))
        xiaka = float(np.einsum("ia,ka,ik->", matrix, matrix, occupied_projector))
        spin_square = (
            nob
            + 1.0
            - ground_overlap_sum
            - xiaib
            + xiaka
            + xia
            + 0.25 * (noa - nob - 2) * (noa - nob)
        )
        result.append(
            SFTransitionProperties(
                transition_density_ao=transition,
                difference_density_ao=difference,
                nto_singular_values=singular,
                occupied_nto_coefficients=occupied_nto,
                virtual_nto_coefficients=virtual_nto,
                dominant_configurations=dominant_configurations,
                participation_ratio=participation,
                spin_square=max(0.0, spin_square),
            )
        )
    return tuple(result)


def grimme_xtb_sf_tda(
    state: GFN2OrbitalState,
    *,
    roots: int = 1,
    tolerance_hartree: float = 1.0e-8,
    maximum_iterations: int = 100,
    maximum_subspace: int | None = None,
    selection_policy: SFTDASelectionPolicy | None = None,
    eigensolver_backend: str = "auto",
) -> SFTDAResult:
    """Run the resident SF-TDA solver with Grimme's ``-xtb -sf`` kernel."""

    policy = selection_policy or SFTDASelectionPolicy()
    space = SpinFlipExcitationSpace.from_grimme_window(state, policy)
    response = AtomicTransitionChargeResponse.grimme_xtb_spin_flip(state, space)
    selected = select_grimme_spin_flip_configurations(space, response, policy)
    operator = SelectedSpinFlipTDAOperator(
        selected,
        response.apply,
        response.diagonal_hartree,
        response.apply_block,
    )
    result = davidson_sf_tda(
        operator,
        roots=roots,
        tolerance_hartree=tolerance_hartree,
        maximum_iterations=maximum_iterations,
        maximum_subspace=maximum_subspace,
        backend=eigensolver_backend,
    )
    properties = spin_flip_transition_properties(
        state,
        selected,
        result.amplitudes,
    )
    return SFTDAResult(
        excitation_energies_hartree=result.excitation_energies_hartree,
        amplitudes=result.amplitudes,
        residual_norms=result.residual_norms,
        iterations=result.iterations,
        converged=result.converged,
        transition_properties=properties,
        solver_backend=result.solver_backend,
        selection_reference=selected.selection_reference,
    )


__all__ = [
    "AtomicTransitionChargeResponse",
    "GRIMME_XTB_SF_STDA",
    "SFTDAKernelParameters",
    "SFTDAResult",
    "SFTDASelectionPolicy",
    "FrozenSpinFlipSelection",
    "SFTransitionProperties",
    "SelectedSpinFlipSpace",
    "SelectedSpinFlipTDAOperator",
    "SpinFlipExcitationSpace",
    "SpinFlipTDAOperator",
    "davidson_sf_tda",
    "grimme_stda_atomic_hardnesses",
    "grimme_xtb_sf_tda",
    "select_grimme_spin_flip_configurations",
    "spin_flip_transition_properties",
]
