"""Backend-independent electronic states and orbital transformations for APOC."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


APOC_ELECTRONIC_STATE_SCHEMA = "matrix.apoc.electronic_state.v1"


@dataclass(frozen=True)
class GaussianBasis:
    """Explicit Gaussian AO basis in TREXIO-compatible indexing."""

    cartesian: bool
    shell_nucleus_index: np.ndarray
    shell_angular_momentum: np.ndarray
    shell_factor: np.ndarray
    shell_r_power: np.ndarray
    primitive_shell_index: np.ndarray
    primitive_exponent: np.ndarray
    primitive_coefficient: np.ndarray
    primitive_factor: np.ndarray
    ao_shell: np.ndarray
    ao_normalization: np.ndarray

    def __post_init__(self) -> None:
        integer_names = (
            "shell_nucleus_index",
            "shell_angular_momentum",
            "shell_r_power",
            "primitive_shell_index",
            "ao_shell",
        )
        float_names = (
            "shell_factor",
            "primitive_exponent",
            "primitive_coefficient",
            "primitive_factor",
            "ao_normalization",
        )
        for name in integer_names:
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=int))
        for name in float_names:
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        shell_count = self.shell_nucleus_index.size
        if shell_count == 0:
            raise ValueError("Gaussian basis must contain at least one shell")
        if any(
            getattr(self, name).shape != (shell_count,)
            for name in ("shell_angular_momentum", "shell_factor", "shell_r_power")
        ):
            raise ValueError("Gaussian shell arrays have inconsistent dimensions")
        primitive_count = self.primitive_shell_index.size
        if any(
            getattr(self, name).shape != (primitive_count,)
            for name in (
                "primitive_exponent",
                "primitive_coefficient",
                "primitive_factor",
            )
        ):
            raise ValueError("Gaussian primitive arrays have inconsistent dimensions")
        if self.ao_shell.shape != self.ao_normalization.shape:
            raise ValueError("Gaussian AO shell and normalization arrays differ")
        if np.any(self.shell_nucleus_index < 0) or np.any(self.primitive_shell_index < 0):
            raise ValueError("Gaussian basis indices cannot be negative")
        if np.any(self.primitive_shell_index >= shell_count) or np.any(self.ao_shell >= shell_count):
            raise ValueError("Gaussian basis contains an out-of-range shell index")

    @property
    def shell_count(self) -> int:
        return int(self.shell_nucleus_index.size)

    @property
    def primitive_count(self) -> int:
        return int(self.primitive_shell_index.size)

    @property
    def ao_count(self) -> int:
        return int(self.ao_shell.size)


@dataclass(frozen=True)
class ExcitedState:
    """Electronic-state descriptor used for root following across geometries."""

    state_id: int
    transition_density_ao: np.ndarray
    energy_hartree: float | None = None
    oscillator_strength: float | None = None
    label: str = ""
    symmetry: str = ""
    difference_density_ao: np.ndarray | None = None
    transition_density_spin_ao: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        transition = _square_matrix(self.transition_density_ao, "transition density")
        object.__setattr__(self, "transition_density_ao", transition)
        spin = tuple(_square_matrix(item, "spin transition density")
                     for item in self.transition_density_spin_ao)
        if spin:
            if len(spin) != 2 or any(item.shape != transition.shape for item in spin):
                raise ValueError("spin transition densities require matching alpha/beta AO matrices")
            if not np.allclose(sum(spin), transition, atol=1e-10):
                raise ValueError("spin transition densities do not sum to the total transition density")
        object.__setattr__(self, "transition_density_spin_ao", spin)
        if self.difference_density_ao is not None:
            difference = _square_matrix(self.difference_density_ao, "difference density")
            if difference.shape != transition.shape:
                raise ValueError("transition and difference densities use different AO spaces")
            object.__setattr__(self, "difference_density_ao", difference)
        if int(self.state_id) < 1:
            raise ValueError("excited-state identifiers start at one")


@dataclass(frozen=True)
class ElectronicState:
    """Complete APOC electronic state, independent of the producing QM code.

    Molecular-orbital arrays are tuples with one restricted channel or two
    alpha/beta channels.  Coefficient matrices use ``(nao, nmo)`` ordering;
    AO density and Fock matrices use the same AO ordering as ``overlap_ao``.
    """

    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    overlap_ao: np.ndarray
    mo_coefficients: tuple[np.ndarray, ...]
    mo_occupations: tuple[np.ndarray, ...]
    mo_energies_hartree: tuple[np.ndarray, ...] = ()
    density_matrices_ao: tuple[np.ndarray, ...] = ()
    fock_matrices_ao: tuple[np.ndarray, ...] = ()
    excited_states: tuple[ExcitedState, ...] = ()
    basis: GaussianBasis | None = None
    charge: int = 0
    multiplicity: int = 1
    total_energy_hartree: float | None = None
    label: str = "S0"
    method: str = ""
    basis_label: str = ""
    source: str = ""
    schema: str = APOC_ELECTRONIC_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != APOC_ELECTRONIC_STATE_SCHEMA:
            raise ValueError(f"unsupported APOC electronic-state schema: {self.schema}")
        numbers = np.asarray(self.atomic_numbers, dtype=int).reshape(-1)
        coordinates = np.asarray(self.coordinates_bohr, dtype=float)
        overlap = _symmetric_matrix(self.overlap_ao, "AO overlap")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "coordinates_bohr", coordinates)
        object.__setattr__(self, "overlap_ao", overlap)
        if numbers.size == 0 or coordinates.shape != (numbers.size, 3):
            raise ValueError("atomic numbers and Cartesian coordinates are inconsistent")
        nao = overlap.shape[0]
        if len(self.mo_coefficients) not in {1, 2}:
            raise ValueError("APOC states require one restricted or two spin MO channels")
        coefficients = tuple(np.asarray(item) for item in self.mo_coefficients)
        occupations = tuple(np.asarray(item, dtype=float).reshape(-1) for item in self.mo_occupations)
        if len(coefficients) != len(occupations):
            raise ValueError("MO coefficients and occupations have different channel counts")
        for coefficient, occupation in zip(coefficients, occupations, strict=True):
            if coefficient.ndim != 2 or coefficient.shape[0] != nao:
                raise ValueError("MO coefficient matrix has an incompatible AO dimension")
            if coefficient.shape[1] != occupation.size:
                raise ValueError("MO coefficient and occupation dimensions differ")
            if np.any(occupation < -1.0e-10):
                raise ValueError("MO occupations cannot be negative")
        object.__setattr__(self, "mo_coefficients", coefficients)
        object.__setattr__(self, "mo_occupations", occupations)
        energies = _channel_vectors(
            self.mo_energies_hartree,
            coefficients,
            "MO energies",
            allow_empty=True,
        )
        object.__setattr__(self, "mo_energies_hartree", energies)
        densities = _channel_matrices(
            self.density_matrices_ao,
            len(coefficients),
            nao,
            "AO density",
            allow_empty=True,
            symmetric=True,
        )
        focks = _channel_matrices(
            self.fock_matrices_ao,
            len(coefficients),
            nao,
            "AO Fock",
            allow_empty=True,
            symmetric=True,
        )
        object.__setattr__(self, "density_matrices_ao", densities)
        object.__setattr__(self, "fock_matrices_ao", focks)
        excited = tuple(self.excited_states)
        if len({item.state_id for item in excited}) != len(excited):
            raise ValueError("excited-state identifiers must be unique")
        if any(item.transition_density_ao.shape != (nao, nao) for item in excited):
            raise ValueError("excited-state density uses an incompatible AO space")
        object.__setattr__(self, "excited_states", excited)
        if self.basis is not None and self.basis.ao_count != nao:
            raise ValueError("Gaussian basis and overlap matrix have different AO counts")
        electron_count = self.electron_count
        expected = float(np.sum(numbers) - int(self.charge))
        if abs(electron_count - expected) > 2.0e-6:
            raise ValueError(
                f"MO occupations contain {electron_count:.8f} electrons; expected {expected:.8f}"
            )
        if int(self.multiplicity) < 1:
            raise ValueError("multiplicity must be positive")

    @property
    def nao(self) -> int:
        return int(self.overlap_ao.shape[0])

    @property
    def natoms(self) -> int:
        return int(self.atomic_numbers.size)

    @property
    def spin_channels(self) -> int:
        return len(self.mo_coefficients)

    @property
    def electron_count(self) -> float:
        return float(sum(np.sum(item) for item in self.mo_occupations))

    def density_ao(self, channel: int = 0) -> np.ndarray:
        if self.density_matrices_ao:
            return self.density_matrices_ao[channel].copy()
        coefficient = self.mo_coefficients[channel]
        occupation = self.mo_occupations[channel]
        return (coefficient * occupation[None, :]) @ coefficient.conj().T

    def fock_ao(self, channel: int = 0) -> np.ndarray:
        if self.fock_matrices_ao:
            return self.fock_matrices_ao[channel].copy()
        if not self.mo_energies_hartree:
            raise ValueError("no AO Fock matrix or canonical MO energies are available")
        coefficient = self.mo_coefficients[channel]
        energies = self.mo_energies_hartree[channel]
        if coefficient.shape[1] != self.nao:
            raise ValueError("a complete canonical MO space is required to reconstruct Fock")
        fock = (
            self.overlap_ao
            @ (coefficient * energies[None, :])
            @ coefficient.conj().T
            @ self.overlap_ao
        )
        return 0.5 * (fock + fock.conj().T)


@dataclass(frozen=True)
class OrbitalSet:
    coefficients: np.ndarray
    occupations: np.ndarray
    energies_hartree: np.ndarray | None = None


@dataclass(frozen=True)
class StateMatch:
    reference_state_id: int
    candidate_state_id: int
    overlap: float
    phase: float
    energy_difference_hartree: float | None
    runner_up_overlap: float = 0.0
    margin: float = 1.0
    continuous: bool = True
    ambiguous: bool = False


@dataclass(frozen=True)
class StateSubspaceMatch:
    """Certified continuation of a possibly degenerate electronic subspace."""

    reference_state_ids: tuple[int, ...]
    candidate_state_ids: tuple[int, ...]
    principal_overlaps: np.ndarray
    alignment_matrix: np.ndarray
    minimum_principal_overlap: float
    mean_squared_overlap: float
    runner_up_score: float
    margin: float
    continuous: bool
    ambiguous: bool

    def __post_init__(self) -> None:
        principal = np.array(self.principal_overlaps, dtype=float, copy=True)
        alignment = np.array(self.alignment_matrix, dtype=float, copy=True)
        count = len(self.reference_state_ids)
        if count < 2 or len(self.candidate_state_ids) != count:
            raise ValueError("state-subspace tracking requires equally sized manifolds")
        if principal.shape != (count,) or alignment.shape != (count, count):
            raise ValueError("state-subspace tracking matrices have incompatible dimensions")
        if not np.all(np.isfinite(principal)) or not np.all(np.isfinite(alignment)):
            raise ValueError("state-subspace tracking values must be finite")
        if np.any(principal < 0.0) or np.any(principal > 1.0 + 1.0e-12):
            raise ValueError("state-subspace principal overlaps must lie in [0, 1]")
        if not np.allclose(alignment.T @ alignment, np.eye(count), atol=2.0e-10):
            raise ValueError("state-subspace alignment must be orthogonal")
        principal.setflags(write=False)
        alignment.setflags(write=False)
        object.__setattr__(self, "principal_overlaps", principal)
        object.__setattr__(self, "alignment_matrix", alignment)


@dataclass(frozen=True)
class StateFingerprint:
    """Backend-neutral electronic-state character for geometry displacements.

    The vector may be an AO transition density flattened in a fixed metric or
    a backend-native response vector whose representation remains unchanged
    across the compared points.  Root indices are metadata only: continuity is
    decided from the normalized character overlap.
    """

    state_id: int
    vector: np.ndarray
    representation: str
    energy_hartree: float | None = None
    label: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=float).reshape(-1)
        if int(self.state_id) < 1:
            raise ValueError("state-fingerprint identifiers start at one")
        if not str(self.representation).strip():
            raise ValueError("state fingerprints require a representation label")
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("state fingerprints require a finite non-empty vector")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(float).tiny:
            raise ValueError("state fingerprints cannot have zero norm")
        object.__setattr__(self, "vector", vector / norm)
        object.__setattr__(self, "representation", str(self.representation).strip())
        if self.energy_hartree is not None:
            object.__setattr__(self, "energy_hartree", float(self.energy_hartree))


class StateContinuityError(RuntimeError):
    """Raised when a requested electronic state cannot be followed safely."""


def fingerprint_overlap(
    reference: StateFingerprint,
    candidate: StateFingerprint,
) -> tuple[float, float]:
    """Return phase-insensitive overlap for two compatible state fingerprints."""

    if reference.representation != candidate.representation:
        raise ValueError(
            "state fingerprints use different representations: "
            f"{reference.representation!r} and {candidate.representation!r}"
        )
    if reference.vector.shape != candidate.vector.shape:
        raise ValueError("state fingerprints have incompatible dimensions")
    value = float(np.dot(reference.vector, candidate.vector))
    return min(1.0, abs(value)), (1.0 if value >= 0.0 else -1.0)


def match_state_fingerprints(
    reference_states: Iterable[StateFingerprint],
    candidate_states: Iterable[StateFingerprint],
    *,
    minimum_overlap: float = 0.70,
    ambiguity_margin: float = 0.05,
    maximum_energy_difference_hartree: float | None = None,
) -> tuple[StateMatch, ...]:
    """Assign a state manifold by character with an optional energy-continuity gate."""

    references = tuple(reference_states)
    candidates = tuple(candidate_states)
    if (
        maximum_energy_difference_hartree is not None
        and maximum_energy_difference_hartree < 0.0
    ):
        raise ValueError("maximum state-energy difference cannot be negative")
    if not references or not candidates:
        return ()
    scores = np.zeros((len(references), len(candidates)), dtype=float)
    phases = np.ones_like(scores)
    energy_allowed = np.ones_like(scores, dtype=bool)
    for row, reference in enumerate(references):
        for column, candidate in enumerate(candidates):
            scores[row, column], phases[row, column] = fingerprint_overlap(
                reference, candidate
            )
            if (
                maximum_energy_difference_hartree is not None
                and reference.energy_hartree is not None
                and candidate.energy_hartree is not None
            ):
                energy_allowed[row, column] = (
                    abs(candidate.energy_hartree - reference.energy_hartree)
                    <= maximum_energy_difference_hartree
                )
    assignment_scores = np.where(energy_allowed, scores, -2.0)
    pairs = _maximum_overlap_assignment(assignment_scores)
    matches: list[StateMatch] = []
    for left, right in pairs:
        reference = references[left]
        candidate = candidates[right]
        alternatives = np.delete(
            np.where(energy_allowed[left], scores[left], -2.0),
            right,
        )
        runner_up = float(np.max(alternatives)) if alternatives.size else 0.0
        if runner_up < 0.0:
            runner_up = 0.0
        margin = float(scores[left, right] - runner_up)
        matches.append(
            StateMatch(
                reference_state_id=reference.state_id,
                candidate_state_id=candidate.state_id,
                overlap=float(scores[left, right]),
                phase=float(phases[left, right]),
                energy_difference_hartree=(
                    None
                    if reference.energy_hartree is None or candidate.energy_hartree is None
                    else candidate.energy_hartree - reference.energy_hartree
                ),
                runner_up_overlap=runner_up,
                margin=margin,
                continuous=bool(
                    energy_allowed[left, right]
                    and scores[left, right] >= minimum_overlap
                ),
                ambiguous=bool(margin < ambiguity_margin),
            )
        )
    return tuple(matches)


def follow_state_fingerprint(
    reference_state: StateFingerprint,
    candidate_states: Iterable[StateFingerprint],
    *,
    minimum_overlap: float = 0.70,
    ambiguity_margin: float = 0.05,
    strict: bool = True,
) -> tuple[StateFingerprint, StateMatch]:
    """Follow one state and reject a discontinuous or ambiguous assignment."""

    candidates = tuple(candidate_states)
    if not candidates:
        raise StateContinuityError("no candidate state fingerprints were supplied")
    matches = match_state_fingerprints(
        (reference_state,),
        candidates,
        minimum_overlap=minimum_overlap,
        ambiguity_margin=ambiguity_margin,
    )
    if not matches:
        raise StateContinuityError("no electronic-state fingerprint assignment is possible")
    match = matches[0]
    if strict and (not match.continuous or match.ambiguous):
        reason = "ambiguous assignment" if match.ambiguous else "insufficient overlap"
        raise StateContinuityError(
            f"electronic state {reference_state.state_id} has {reason}: "
            f"overlap={match.overlap:.6f}, margin={match.margin:.6f}"
        )
    selected = next(item for item in candidates if item.state_id == match.candidate_state_id)
    return selected, match


def _fingerprint_subspace_basis(states: tuple[StateFingerprint, ...]) -> np.ndarray:
    representations = {state.representation for state in states}
    shapes = {state.vector.shape for state in states}
    if len(representations) != 1 or len(shapes) != 1:
        raise ValueError("state-subspace fingerprints must use one representation and dimension")
    vectors = np.column_stack(tuple(state.vector for state in states))
    basis, singular, _ = np.linalg.svd(vectors, full_matrices=False)
    if len(singular) != len(states) or float(np.min(singular)) <= 1.0e-10:
        raise ValueError("state-subspace fingerprints are linearly dependent")
    return basis[:, : len(states)]


def follow_state_subspace(
    reference_states: Iterable[StateFingerprint],
    candidate_states: Iterable[StateFingerprint],
    *,
    minimum_principal_overlap: float = 0.70,
    ambiguity_margin: float = 0.02,
    strict: bool = True,
) -> tuple[tuple[StateFingerprint, ...], StateSubspaceMatch]:
    """Follow a degenerate state manifold by its principal overlaps.

    Individual roots may exchange order, phase, or rotate inside a degenerate
    block.  APOC therefore selects the candidate subset with the largest
    invariant principal-overlap score and returns the orthogonal Procrustes
    alignment of the two subspaces.  Strict mode is fail-closed whenever the
    subspace loses continuity or two different candidate subsets are
    indistinguishable within ``ambiguity_margin``.
    """

    references = tuple(reference_states)
    candidates = tuple(candidate_states)
    count = len(references)
    if count < 2:
        raise ValueError("state-subspace tracking requires at least two reference states")
    if len(candidates) < count:
        raise StateContinuityError("candidate manifold is smaller than the tracked subspace")
    if not 0.0 < float(minimum_principal_overlap) <= 1.0:
        raise ValueError("minimum principal overlap must lie in (0, 1]")
    if not 0.0 <= float(ambiguity_margin) < 1.0:
        raise ValueError("state-subspace ambiguity margin must lie in [0, 1)")
    reference_basis = _fingerprint_subspace_basis(references)
    representation = references[0].representation
    shape = references[0].vector.shape
    compatible = tuple(
        state
        for state in candidates
        if state.representation == representation and state.vector.shape == shape
    )
    if len(compatible) < count:
        raise StateContinuityError("too few compatible candidate state fingerprints")

    ranked: list[
        tuple[tuple[float, float], tuple[StateFingerprint, ...], np.ndarray, np.ndarray]
    ] = []
    for subset in combinations(compatible, count):
        candidate_basis = _fingerprint_subspace_basis(subset)
        overlap = reference_basis.T @ candidate_basis
        left, principal, right_t = np.linalg.svd(overlap, full_matrices=False)
        principal = np.clip(principal, 0.0, 1.0)
        alignment = right_t.T @ left.T
        score = (float(np.min(principal)), float(np.mean(principal**2)))
        ranked.append((score, subset, principal, alignment))
    ranked.sort(
        key=lambda item: (
            item[0][0],
            item[0][1],
            tuple(-state.state_id for state in item[1]),
        ),
        reverse=True,
    )
    score, selected, principal, alignment = ranked[0]
    runner_up = ranked[1][0][1] if len(ranked) > 1 else 0.0
    margin = float(score[1] - runner_up)
    continuous = bool(score[0] >= float(minimum_principal_overlap))
    ambiguous = bool(len(ranked) > 1 and margin < float(ambiguity_margin))
    match = StateSubspaceMatch(
        tuple(state.state_id for state in references),
        tuple(state.state_id for state in selected),
        principal,
        alignment,
        score[0],
        score[1],
        runner_up,
        margin,
        continuous,
        ambiguous,
    )
    if strict and (not continuous or ambiguous):
        reason = "ambiguous candidate subspaces" if ambiguous else "insufficient principal overlap"
        raise StateContinuityError(
            f"electronic subspace {match.reference_state_ids} has {reason}: "
            f"minimum={match.minimum_principal_overlap:.6f}, margin={match.margin:.6f}"
        )
    return selected, match


def natural_orbitals(
    density_ao: np.ndarray,
    overlap_ao: np.ndarray,
    *,
    cutoff: float = 1.0e-10,
) -> OrbitalSet:
    """Diagonalize a one-particle density in a nonorthogonal AO basis."""

    density = _symmetric_matrix(density_ao, "AO density")
    overlap = _symmetric_matrix(overlap_ao, "AO overlap")
    if density.shape != overlap.shape:
        raise ValueError("AO density and overlap dimensions differ")
    root, inverse_root = _metric_roots(overlap, cutoff=cutoff)
    orthogonal_density = root @ density @ root
    occupations, vectors = np.linalg.eigh(
        0.5 * (orthogonal_density + orthogonal_density.conj().T)
    )
    order = np.argsort(occupations)[::-1]
    occupations = np.real_if_close(occupations[order]).astype(float)
    occupations[np.abs(occupations) < cutoff] = 0.0
    coefficients = inverse_root @ vectors[:, order]
    return OrbitalSet(coefficients=coefficients, occupations=occupations)


def recanonicalize_orbitals(
    coefficients: np.ndarray,
    fock_ao: np.ndarray,
    overlap_ao: np.ndarray,
    *,
    occupations: np.ndarray | None = None,
    occupation_tolerance: float = 1.0e-8,
    preserve_density: bool = True,
) -> OrbitalSet:
    """Diagonalize Fock in an orbital space, optionally preserving its density.

    When occupations are supplied, the default only rotates orbitals inside
    equal-occupation subspaces.  This is the unique safe recanonicalization of
    natural orbitals: mixing unequal occupations would change the density that
    defined them.
    """

    coefficient = np.asarray(coefficients)
    fock = _symmetric_matrix(fock_ao, "AO Fock")
    overlap = _symmetric_matrix(overlap_ao, "AO overlap")
    if coefficient.ndim != 2 or coefficient.shape[0] != overlap.shape[0]:
        raise ValueError("orbital coefficients and AO metric are incompatible")
    metric = coefficient.conj().T @ overlap @ coefficient
    if not np.allclose(metric, np.eye(metric.shape[0]), atol=2.0e-7, rtol=0.0):
        raise ValueError("orbitals must be orthonormal in the supplied AO metric")
    projected = coefficient.conj().T @ fock @ coefficient
    projected = 0.5 * (projected + projected.conj().T)
    occupation = (
        np.zeros(coefficient.shape[1], dtype=float)
        if occupations is None
        else np.asarray(occupations, dtype=float).reshape(-1)
    )
    if occupation.shape != (coefficient.shape[1],):
        raise ValueError("orbital occupations have an incompatible dimension")
    groups = (
        _occupation_groups(occupation, occupation_tolerance)
        if preserve_density and occupations is not None
        else (np.arange(coefficient.shape[1], dtype=int),)
    )
    rotated = coefficient.copy()
    energies = np.empty(coefficient.shape[1], dtype=float)
    for indices in groups:
        values, vectors = np.linalg.eigh(projected[np.ix_(indices, indices)])
        rotated[:, indices] = coefficient[:, indices] @ vectors
        energies[indices] = np.real_if_close(values)
    return OrbitalSet(rotated, occupation.copy(), energies)


def transition_density_overlap(
    reference: ExcitedState,
    candidate: ExcitedState,
    overlap_reference_candidate: np.ndarray,
    *,
    overlap_reference: np.ndarray | None = None,
    overlap_candidate: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return normalized transition-density overlap and phase.

    ``overlap_reference_candidate`` is the cross-AO overlap between consecutive
    geometries.  At one geometry it is simply the ordinary AO overlap matrix.
    """

    cross = np.asarray(overlap_reference_candidate)
    left = reference.transition_density_ao
    right = candidate.transition_density_ao
    if cross.shape != (left.shape[0], right.shape[0]):
        raise ValueError("cross-AO overlap has incompatible dimensions")
    left_metric = (
        np.asarray(overlap_reference) if overlap_reference is not None else cross
    )
    right_metric = (
        np.asarray(overlap_candidate) if overlap_candidate is not None else cross.conj().T
    )
    left_channels = reference.transition_density_spin_ao or (left,)
    right_channels = candidate.transition_density_spin_ao or (right,)
    if len(left_channels) != len(right_channels):
        raise ValueError("state following cannot mix spin-resolved and spin-summed transition densities")
    # Retain alpha/beta character, including states whose spin sum vanishes.
    numerator = sum(np.trace(a.conj().T @ cross @ b @ cross.conj().T)
                    for a, b in zip(left_channels, right_channels, strict=True))
    left_norm = sum(np.real(np.trace(a.conj().T @ left_metric @ a @ left_metric))
                    for a in left_channels)
    right_norm = sum(np.real(np.trace(b.conj().T @ right_metric @ b @ right_metric))
                     for b in right_channels)
    denominator = float(np.sqrt(max(left_norm, 0.0) * max(right_norm, 0.0)))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("cannot track a state with a zero transition density")
    normalized = complex(numerator) / denominator
    score = float(min(1.0, abs(normalized)))
    phase = 1.0 if abs(normalized) == 0.0 else float(np.real(normalized / abs(normalized)))
    return score, phase


def match_excited_states(
    reference_states: Iterable[ExcitedState],
    candidate_states: Iterable[ExcitedState],
    overlap_reference_candidate: np.ndarray,
    *,
    overlap_reference: np.ndarray | None = None,
    overlap_candidate: np.ndarray | None = None,
    minimum_overlap: float = 0.70,
    ambiguity_margin: float = 0.05,
) -> tuple[StateMatch, ...]:
    """Find a deterministic one-to-one maximum-overlap state assignment."""

    references = tuple(reference_states)
    candidates = tuple(candidate_states)
    if not references or not candidates:
        return ()
    scores = np.zeros((len(references), len(candidates)), dtype=float)
    phases = np.ones_like(scores)
    for i, reference in enumerate(references):
        for j, candidate in enumerate(candidates):
            scores[i, j], phases[i, j] = transition_density_overlap(
                reference,
                candidate,
                overlap_reference_candidate,
                overlap_reference=overlap_reference,
                overlap_candidate=overlap_candidate,
            )
    pairs = _maximum_overlap_assignment(scores)
    matches = []
    for left, right in pairs:
        ref = references[left]
        cand = candidates[right]
        delta = (
            None
            if ref.energy_hartree is None or cand.energy_hartree is None
            else float(cand.energy_hartree - ref.energy_hartree)
        )
        alternatives = np.delete(scores[left], right)
        runner_up = float(np.max(alternatives)) if alternatives.size else 0.0
        margin = float(scores[left, right] - runner_up)
        matches.append(
            StateMatch(
                reference_state_id=ref.state_id,
                candidate_state_id=cand.state_id,
                overlap=float(scores[left, right]),
                phase=float(phases[left, right]),
                energy_difference_hartree=delta,
                runner_up_overlap=runner_up,
                margin=margin,
                continuous=bool(scores[left, right] >= minimum_overlap),
                ambiguous=bool(margin < ambiguity_margin),
            )
        )
    return tuple(matches)


def follow_excited_state(
    reference_state: ExcitedState,
    candidate_states: Iterable[ExcitedState],
    overlap_reference_candidate: np.ndarray,
    *,
    overlap_reference: np.ndarray | None = None,
    overlap_candidate: np.ndarray | None = None,
    minimum_overlap: float = 0.70,
    ambiguity_margin: float = 0.05,
    strict: bool = True,
) -> tuple[ExcitedState, StateMatch]:
    """Select the continuation of one state for a scan or TD optimization.

    The identity of a root is established from its transition density, not its
    instantaneous energy index.  In strict mode an optimizer must stop before
    accepting a step whose overlap is too small or whose assignment is
    ambiguous; the caller can then reduce the step or request more roots.
    """

    candidates = tuple(candidate_states)
    if not candidates:
        raise StateContinuityError("no candidate excited states were supplied")
    matches = match_excited_states(
        (reference_state,),
        candidates,
        overlap_reference_candidate,
        overlap_reference=overlap_reference,
        overlap_candidate=overlap_candidate,
        minimum_overlap=minimum_overlap,
        ambiguity_margin=ambiguity_margin,
    )
    if not matches:
        raise StateContinuityError("no electronic-state assignment is possible")
    match = matches[0]
    if strict and (not match.continuous or match.ambiguous):
        reason = "ambiguous assignment" if match.ambiguous else "insufficient overlap"
        raise StateContinuityError(
            f"TD state {reference_state.state_id} has {reason}: "
            f"overlap={match.overlap:.6f}, margin={match.margin:.6f}"
        )
    selected = next(
        item for item in candidates if item.state_id == match.candidate_state_id
    )
    return selected, match


def gaussian_basis_from_pyscf(mol) -> GaussianBasis:
    """Capture a PySCF Gaussian basis without retaining a PySCF object."""

    shell_nucleus: list[int] = []
    shell_l: list[int] = []
    shell_factor: list[float] = []
    shell_r_power: list[int] = []
    primitive_shell: list[int] = []
    exponents: list[float] = []
    coefficients: list[float] = []
    primitive_factor: list[float] = []
    ao_shell: list[int] = []
    for source_shell in range(int(mol.nbas)):
        angular = int(mol.bas_angular(source_shell))
        source_exponents = np.asarray(mol.bas_exp(source_shell), dtype=float)
        contractions = np.asarray(mol.bas_ctr_coeff(source_shell), dtype=float)
        function_count = (
            (angular + 1) * (angular + 2) // 2 if bool(mol.cart) else 2 * angular + 1
        )
        for contraction in range(int(mol.bas_nctr(source_shell))):
            target_shell = len(shell_nucleus)
            shell_nucleus.append(int(mol.bas_atom(source_shell)))
            shell_l.append(angular)
            shell_factor.append(1.0)
            shell_r_power.append(0)
            for exponent, coefficient in zip(
                source_exponents,
                contractions[:, contraction],
                strict=True,
            ):
                primitive_shell.append(target_shell)
                exponents.append(float(exponent))
                coefficients.append(float(coefficient))
                primitive_factor.append(1.0)
            ao_shell.extend([target_shell] * function_count)
    return GaussianBasis(
        cartesian=bool(mol.cart),
        shell_nucleus_index=np.asarray(shell_nucleus),
        shell_angular_momentum=np.asarray(shell_l),
        shell_factor=np.asarray(shell_factor),
        shell_r_power=np.asarray(shell_r_power),
        primitive_shell_index=np.asarray(primitive_shell),
        primitive_exponent=np.asarray(exponents),
        primitive_coefficient=np.asarray(coefficients),
        primitive_factor=np.asarray(primitive_factor),
        ao_shell=np.asarray(ao_shell),
        ao_normalization=np.ones(len(ao_shell), dtype=float),
    )


def electronic_state_from_gaussian_fchk(path: Path | str) -> ElectronicState:
    """Import a Gaussian FCHK directly into the native APOC AO contract.

    IOData decodes the formatted checkpoint, but no intermediate wavefunction
    file is written.  Orbital coefficients and density matrices are reordered
    once from Gaussian's AO convention to the convention used by APOC's
    Gaussian-basis numerical engine.
    """

    try:
        from iodata import load_one
        from iodata.convert import convert_conventions, convert_to_segmented
        from iodata.overlap import compute_overlap
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Gaussian FCHK import requires IOData") from exc
    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Gaussian FCHK import requires the APOC PySCF runtime") from exc
    from matrix_chem.topology.elements import atomic_symbol

    source = Path(path)
    if source.suffix.lower() not in {".fchk", ".fch"}:
        raise ValueError("the direct Gaussian adapter requires an FCHK/FCH file")
    if not source.is_file():
        raise FileNotFoundError(source)
    # GDV may write a compound method (e.g. "RB3LYP TDA-FC") in its
    # fixed-width header. IOData tokenizes that line into at most three words.
    # Normalize metadata only in a temporary copy; every numerical field and
    # the original checkpoint remain byte-for-byte unchanged.
    text = source.read_text()
    header = text.splitlines(keepends=True)
    if len(header[1].split()) > 3:
        from tempfile import TemporaryDirectory

        command, method, basis = header[1][:10].strip(), header[1][10:70].strip(), header[1][70:].strip()
        if not command or not method or not basis:
            raise ValueError("unsupported compound Gaussian FCHK header")
        with TemporaryDirectory(prefix="apoc-fchk-header-") as directory:
            normalized = Path(directory)/source.name
            normalized.write_text(header[0]+f"{command} {method.replace(' ', '_')} {basis}\n"+''.join(header[2:]))
            data = load_one(str(normalized))
        data.lot = method
    else:
        data = load_one(str(source))
    if data.obasis is None or data.mo is None:
        raise ValueError("Gaussian FCHK does not contain a complete AO basis and orbitals")
    if data.mo.kind == "generalized":
        raise ValueError("generalized-spinor Gaussian FCHK files are not yet supported")

    obasis = convert_to_segmented(data.obasis)
    cartesian = _gaussian_basis_is_cartesian(obasis)
    conventions = {
        key: (
            _pyscf_cartesian_convention(int(key[0]))
            if key[1] == "c"
            else _pyscf_pure_convention(int(key[0]))
        )
        for key in obasis.conventions
    }
    permutation, signs = convert_conventions(obasis, conventions)
    permutation, signs = _pyscf_shell_order(
        obasis,
        conventions,
        permutation,
        signs,
    )

    numbers = np.asarray(data.atnums, dtype=int)
    coordinates = np.asarray(data.atcoords, dtype=float)
    charge = int(round(float(data.charge)))
    spin = int(round(float(data.spinpol)))
    labels = [
        f"{atomic_symbol(int(number))}{index}"
        for index, number in enumerate(numbers)
    ]
    basis_by_atom: dict[str, list[list[object]]] = {label: [] for label in labels}
    for shell in obasis.shells:
        if shell.ncon != 1:
            raise ValueError("Gaussian FCHK basis segmentation failed")
        angular = int(shell.angmoms[0])
        contracted: list[object] = [angular]
        contracted.extend(
            [
                float(exponent),
                float(coefficient),
            ]
            for exponent, coefficient in zip(
                shell.exponents,
                shell.coeffs[:, 0],
                strict=True,
            )
        )
        basis_by_atom[labels[int(shell.icenter)]].append(contracted)

    molecule = gto.Mole()
    molecule.atom = list(zip(labels, coordinates.tolist(), strict=True))
    molecule.unit = "Bohr"
    molecule.basis = basis_by_atom
    molecule.charge = charge
    molecule.spin = spin
    molecule.cart = cartesian
    molecule.verbose = 0
    molecule.build()
    overlap = np.asarray(molecule.intor_symmetric("int1e_ovlp"), dtype=float)
    source_overlap = np.asarray(compute_overlap(obasis, coordinates), dtype=float)
    reordered_overlap = _reorder_ao_matrix(source_overlap, permutation, signs)
    if overlap.shape != reordered_overlap.shape:
        raise ValueError("Gaussian FCHK and APOC AO dimensions differ")
    ao_factors = np.sqrt(
        np.diag(reordered_overlap) / np.diag(overlap)
    )
    represented_overlap = overlap * ao_factors[:, None] * ao_factors[None, :]
    if not np.allclose(
        represented_overlap,
        reordered_overlap,
        atol=2.0e-7,
        rtol=0.0,
    ):
        raise ValueError(
            "Gaussian FCHK AO normalization cannot be represented by the APOC "
            "Gaussian-basis engine"
        )

    coefficients = tuple(
        _reorder_ao_coefficients(channel, permutation, signs, ao_factors)
        for channel in (
            (data.mo.coeffsa,)
            if data.mo.kind == "restricted"
            else (data.mo.coeffsa, data.mo.coeffsb)
        )
    )
    occupations = tuple(
        np.asarray(channel, dtype=float)
        for channel in (
            (data.mo.occs,)
            if data.mo.kind == "restricted"
            else (data.mo.occsa, data.mo.occsb)
        )
    )
    energies = tuple(
        np.asarray(channel, dtype=float)
        for channel in (
            (data.mo.energies,)
            if data.mo.kind == "restricted"
            else (data.mo.energiesa, data.mo.energiesb)
        )
    )
    densities = _gaussian_fchk_density_channels(
        data,
        permutation,
        signs,
        ao_factors,
        unrestricted=data.mo.kind == "unrestricted",
    )
    return ElectronicState(
        atomic_numbers=numbers,
        coordinates_bohr=coordinates,
        overlap_ao=overlap,
        mo_coefficients=coefficients,
        mo_occupations=occupations,
        mo_energies_hartree=energies,
        density_matrices_ao=densities,
        basis=gaussian_basis_from_pyscf(molecule),
        charge=charge,
        multiplicity=spin + 1,
        total_energy_hartree=(
            None if data.energy is None else float(data.energy)
        ),
        method="" if data.lot is None else str(data.lot),
        basis_label="" if data.obasis_name is None else str(data.obasis_name),
        source=str(source),
    )


def _gaussian_basis_is_cartesian(obasis) -> bool:
    high_kinds = {
        str(kind)
        for shell in obasis.shells
        for angular, kind in zip(shell.angmoms, shell.kinds, strict=True)
        if int(angular) >= 2
    }
    if len(high_kinds) > 1:
        raise ValueError("mixed Cartesian/pure high-angular-momentum FCHK basis")
    return high_kinds == {"c"}


def _pyscf_cartesian_convention(angular: int) -> list[str]:
    if angular == 0:
        return ["1"]
    return [
        "x" * x_power + "y" * y_power + "z" * (angular - x_power - y_power)
        for x_power in range(angular, -1, -1)
        for y_power in range(angular - x_power, -1, -1)
    ]


def _pyscf_pure_convention(angular: int) -> list[str]:
    if angular == 0:
        return ["1"]
    if angular == 1:
        return ["x", "y", "z"]
    return (
        [f"s{order}" for order in range(angular, 0, -1)]
        + ["c0"]
        + [f"c{order}" for order in range(1, angular + 1)]
    )


def _pyscf_shell_order(
    obasis,
    conventions,
    permutation,
    signs,
) -> tuple[np.ndarray, np.ndarray]:
    """Match PySCF's atom-major, angular-momentum-major shell ordering."""

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for shell in obasis.shells:
        angular = int(shell.angmoms[0])
        kind = str(shell.kinds[0])
        count = len(conventions[(angular, kind)])
        offsets.append((cursor, cursor + count))
        cursor += count
    shell_order = sorted(
        range(len(obasis.shells)),
        key=lambda index: (
            int(obasis.shells[index].icenter),
            int(obasis.shells[index].angmoms[0]),
            index,
        ),
    )
    blocks = [
        np.arange(offsets[index][0], offsets[index][1], dtype=int)
        for index in shell_order
    ]
    order = np.concatenate(blocks) if blocks else np.array([], dtype=int)
    return (
        np.asarray(permutation, dtype=int)[order],
        np.asarray(signs, dtype=int)[order],
    )


def _reorder_ao_coefficients(
    values,
    permutation,
    signs,
    factors,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scale = np.asarray(signs, dtype=float) * np.asarray(factors, dtype=float)
    return array[np.asarray(permutation, dtype=int)] * scale[:, None]


def _reorder_ao_matrix(values, permutation, signs) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    indices = np.asarray(permutation, dtype=int)
    phases = np.asarray(signs, dtype=float)
    return array[np.ix_(indices, indices)] * phases[:, None] * phases[None, :]


def _gaussian_fchk_density_channels(
    data,
    permutation,
    signs,
    factors,
    *,
    unrestricted: bool,
) -> tuple[np.ndarray, ...]:
    densities = data.one_rdms or {}
    # SCF densities are reconstructed from the complete orbital block.  This
    # avoids the reduced-precision packed SCF matrix written by some Gaussian
    # jobs.  A correlated one-particle density, when present, is authoritative.
    total = densities.get("post_scf_ao")
    spin = densities.get("post_scf_spin_ao")
    if total is None:
        return ()
    total = _reorder_ao_matrix(total, permutation, signs)
    total *= np.asarray(factors)[:, None] * np.asarray(factors)[None, :]
    if not unrestricted:
        return (total,)
    if spin is None:
        return ()
    spin = _reorder_ao_matrix(spin, permutation, signs)
    spin *= np.asarray(factors)[:, None] * np.asarray(factors)[None, :]
    return (0.5 * (total + spin), 0.5 * (total - spin))


def gaussian_basis_from_psi4(basis_set) -> GaussianBasis:
    """Capture a Psi4 Gaussian basis in the same explicit APOC convention."""

    shell_nucleus: list[int] = []
    shell_l: list[int] = []
    shell_factor: list[float] = []
    shell_r_power: list[int] = []
    primitive_shell: list[int] = []
    exponents: list[float] = []
    coefficients: list[float] = []
    primitive_factor: list[float] = []
    ao_shell: list[int] = []
    for shell_index in range(int(basis_set.nshell())):
        shell = basis_set.shell(shell_index)
        shell_nucleus.append(int(basis_set.shell_to_center(shell_index)))
        shell_l.append(int(shell.am))
        shell_factor.append(1.0)
        shell_r_power.append(0)
        for primitive_index in range(int(shell.nprimitive)):
            primitive_shell.append(shell_index)
            exponents.append(float(shell.exp(primitive_index)))
            original = float(shell.original_coef(primitive_index))
            normalized = float(shell.coef(primitive_index))
            coefficients.append(original)
            primitive_factor.append(normalized / original if original != 0.0 else 1.0)
        ao_shell.extend([shell_index] * int(shell.nfunction))
    return GaussianBasis(
        cartesian=not bool(basis_set.has_puream()),
        shell_nucleus_index=np.asarray(shell_nucleus),
        shell_angular_momentum=np.asarray(shell_l),
        shell_factor=np.asarray(shell_factor),
        shell_r_power=np.asarray(shell_r_power),
        primitive_shell_index=np.asarray(primitive_shell),
        primitive_exponent=np.asarray(exponents),
        primitive_coefficient=np.asarray(coefficients),
        primitive_factor=np.asarray(primitive_factor),
        ao_shell=np.asarray(ao_shell),
        ao_normalization=np.ones(len(ao_shell), dtype=float),
    )


def electronic_state_from_molden(path: Path | str) -> ElectronicState:
    """Import a complete Molden wavefunction into the native APOC contract."""

    from matrix_pyscf import load_molden_wavefunction

    mol, energies, coefficients, occupations, _labels, _spins = load_molden_wavefunction(path)
    coefficient_channels = coefficients if isinstance(coefficients, tuple) else (coefficients,)
    occupation_channels = occupations if isinstance(occupations, tuple) else (occupations,)
    energy_channels = energies if isinstance(energies, tuple) else (energies,)
    multiplicity = int(mol.spin) + 1
    return ElectronicState(
        atomic_numbers=np.asarray(mol.atom_charges(), dtype=int),
        coordinates_bohr=np.asarray(mol.atom_coords(unit="Bohr"), dtype=float),
        overlap_ao=np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float),
        mo_coefficients=tuple(np.asarray(item) for item in coefficient_channels),
        mo_occupations=tuple(np.asarray(item, dtype=float) for item in occupation_channels),
        mo_energies_hartree=tuple(np.asarray(item, dtype=float) for item in energy_channels),
        basis=gaussian_basis_from_pyscf(mol),
        charge=int(mol.charge),
        multiplicity=multiplicity,
        source=str(Path(path)),
    )


def electronic_state_from_pyscf(
    mean_field,
    *,
    excited_states: Iterable[ExcitedState] = (),
    source: str = "PySCF",
) -> ElectronicState:
    """Capture a converged PySCF mean-field object without a Molden round trip."""

    mol = mean_field.mol
    coefficients = _spin_channels(mean_field.mo_coeff, matrix=True)
    occupations = _spin_channels(mean_field.mo_occ, matrix=False)
    energies = _spin_channels(mean_field.mo_energy, matrix=False)
    density_raw = mean_field.make_rdm1()
    densities = (
        _spin_channels(density_raw, matrix=True)
    )
    fock_raw = mean_field.get_fock(dm=density_raw)
    focks = (
        _spin_channels(fock_raw, matrix=True)
    )
    return ElectronicState(
        atomic_numbers=np.asarray(mol.atom_charges(), dtype=int),
        coordinates_bohr=np.asarray(mol.atom_coords(unit="Bohr"), dtype=float),
        overlap_ao=np.asarray(mean_field.get_ovlp(), dtype=float),
        mo_coefficients=tuple(np.asarray(item) for item in coefficients),
        mo_occupations=tuple(np.asarray(item, dtype=float) for item in occupations),
        mo_energies_hartree=tuple(np.asarray(item, dtype=float) for item in energies),
        density_matrices_ao=densities,
        fock_matrices_ao=focks,
        excited_states=tuple(excited_states),
        basis=gaussian_basis_from_pyscf(mol),
        charge=int(mol.charge),
        multiplicity=int(mol.spin) + 1,
        total_energy_hartree=(
            None if getattr(mean_field, "e_tot", None) is None else float(mean_field.e_tot)
        ),
        method=mean_field.__class__.__name__,
        source=source,
    )


def electronic_state_from_psi4(
    wavefunction,
    *,
    excited_states: Iterable[ExcitedState] = (),
    source: str = "Psi4",
) -> ElectronicState:
    """Capture a Psi4 wavefunction directly through PsiAPI arrays."""

    try:
        import psi4
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Psi4 state import requires the Psi4 Python module") from exc
    molecule = wavefunction.molecule()
    overlap = np.asarray(psi4.core.MintsHelper(wavefunction.basisset()).ao_overlap())
    coefficient_alpha = np.asarray(wavefunction.Ca())
    if coefficient_alpha.shape[0] != overlap.shape[0]:
        raise ValueError(
            "Psi4 symmetry-adapted orbitals cannot be exported as an AO state; "
            "run the transferable electronic-state calculation in C1 symmetry"
        )
    same_orbitals = bool(wavefunction.same_a_b_orbs())
    if same_orbitals:
        coefficients = (coefficient_alpha,)
        occupations = (
            np.concatenate(
                (
                    np.full(int(wavefunction.nbeta()), 2.0),
                    np.full(int(wavefunction.nalpha() - wavefunction.nbeta()), 1.0),
                    np.zeros(coefficient_alpha.shape[1] - int(wavefunction.nalpha())),
                )
            ),
        )
        energies = (np.asarray(wavefunction.epsilon_a()),)
        densities = (np.asarray(wavefunction.Da()) + np.asarray(wavefunction.Db()),)
        focks = (np.asarray(wavefunction.Fa()),)
    else:
        coefficients = (coefficient_alpha, np.asarray(wavefunction.Cb()))
        occupations = (
            np.concatenate(
                (np.ones(int(wavefunction.nalpha())), np.zeros(coefficient_alpha.shape[1] - int(wavefunction.nalpha())))
            ),
            np.concatenate(
                (np.ones(int(wavefunction.nbeta())), np.zeros(coefficients[1].shape[1] - int(wavefunction.nbeta())))
            ),
        )
        energies = (np.asarray(wavefunction.epsilon_a()), np.asarray(wavefunction.epsilon_b()))
        densities = (np.asarray(wavefunction.Da()), np.asarray(wavefunction.Db()))
        focks = (np.asarray(wavefunction.Fa()), np.asarray(wavefunction.Fb()))
    numbers = np.asarray(
        [int(round(float(molecule.Z(index)))) for index in range(molecule.natom())],
        dtype=int,
    )
    coordinates = np.asarray(molecule.geometry(), dtype=float)
    return ElectronicState(
        atomic_numbers=numbers,
        coordinates_bohr=coordinates,
        overlap_ao=overlap,
        mo_coefficients=coefficients,
        mo_occupations=occupations,
        mo_energies_hartree=energies,
        density_matrices_ao=densities,
        fock_matrices_ao=focks,
        excited_states=tuple(excited_states),
        basis=gaussian_basis_from_psi4(wavefunction.basisset()),
        charge=int(round(float(molecule.molecular_charge()))),
        multiplicity=int(molecule.multiplicity()),
        total_energy_hartree=float(wavefunction.energy()),
        method=wavefunction.name(),
        source=source,
    )


def cross_ao_overlap_from_molden(
    reference: Path | str,
    candidate: Path | str,
) -> np.ndarray:
    """Evaluate the cross-geometry AO overlap required for state following."""

    from matrix_pyscf import load_molden_wavefunction

    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("cross-AO overlap evaluation requires PySCF") from exc
    reference_mol = load_molden_wavefunction(reference)[0]
    candidate_mol = load_molden_wavefunction(candidate)[0]
    return np.asarray(
        gto.intor_cross("int1e_ovlp", reference_mol, candidate_mol),
        dtype=float,
    )


def _metric_roots(overlap: np.ndarray, *, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(overlap)
    if np.min(values) <= cutoff:
        raise ValueError("AO overlap is singular at the requested cutoff")
    root = (vectors * np.sqrt(values)[None, :]) @ vectors.conj().T
    inverse = (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.conj().T
    return root, inverse


def _spin_channels(value, *, matrix: bool) -> tuple[np.ndarray, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(np.asarray(item) for item in value)
    array = np.asarray(value)
    channel_dimension = 3 if matrix else 2
    if array.ndim == channel_dimension and array.shape[0] == 2:
        return (array[0], array[1])
    return (array,)


def _occupation_groups(occupations: np.ndarray, tolerance: float) -> tuple[np.ndarray, ...]:
    remaining = set(range(occupations.size))
    groups: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        group = sorted(
            index
            for index in remaining
            if abs(float(occupations[index] - occupations[seed])) <= tolerance
        )
        remaining.difference_update(group)
        groups.append(np.asarray(group, dtype=int))
    return tuple(groups)


def _maximum_overlap_assignment(scores: np.ndarray) -> tuple[tuple[int, int], ...]:
    rows, columns = scores.shape
    transposed = False
    matrix = scores
    if rows > columns:
        matrix = scores.T
        rows, columns = matrix.shape
        transposed = True
    if columns <= 18:
        states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
        for row in range(rows):
            updated: dict[int, tuple[float, tuple[int, ...]]] = {}
            for mask, (total, assignment) in states.items():
                for column in range(columns):
                    if mask & (1 << column):
                        continue
                    candidate = (total + float(matrix[row, column]), assignment + (column,))
                    new_mask = mask | (1 << column)
                    previous = updated.get(new_mask)
                    if previous is None or candidate[0] > previous[0] + 1.0e-15 or (
                        abs(candidate[0] - previous[0]) <= 1.0e-15
                        and candidate[1] < previous[1]
                    ):
                        updated[new_mask] = candidate
            states = updated
        _score, assignment = max(states.values(), key=lambda item: (item[0], tuple(-x for x in item[1])))
        pairs = tuple((row, column) for row, column in enumerate(assignment))
    else:
        available = set(range(columns))
        pairs_list = []
        for row in range(rows):
            column = max(available, key=lambda item: (matrix[row, item], -item))
            available.remove(column)
            pairs_list.append((row, column))
        pairs = tuple(pairs_list)
    if transposed:
        return tuple(sorted((column, row) for row, column in pairs))
    return pairs


def _square_matrix(value, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{label} must be a square matrix")
    return array


def _symmetric_matrix(value, label: str) -> np.ndarray:
    array = _square_matrix(value, label)
    if not np.allclose(array, array.conj().T, atol=2.0e-9, rtol=0.0):
        raise ValueError(f"{label} must be Hermitian")
    return 0.5 * (array + array.conj().T)


def _channel_vectors(values, coefficients, label: str, *, allow_empty: bool):
    if not values and allow_empty:
        return ()
    arrays = tuple(np.asarray(item, dtype=float).reshape(-1) for item in values)
    if len(arrays) != len(coefficients):
        raise ValueError(f"{label} channel count differs from MO coefficients")
    if any(array.size != coefficient.shape[1] for array, coefficient in zip(arrays, coefficients)):
        raise ValueError(f"{label} dimension differs from MO coefficients")
    return arrays


def _channel_matrices(
    values,
    channel_count: int,
    dimension: int,
    label: str,
    *,
    allow_empty: bool,
    symmetric: bool,
):
    if not values and allow_empty:
        return ()
    arrays = tuple(
        _symmetric_matrix(item, label) if symmetric else _square_matrix(item, label)
        for item in values
    )
    if len(arrays) != channel_count or any(item.shape != (dimension, dimension) for item in arrays):
        raise ValueError(f"{label} dimensions or channel count are incompatible")
    return arrays
