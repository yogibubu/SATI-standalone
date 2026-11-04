from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product

import numpy as np

from matrix_numerics import eigh_arrays

from .davidson import DavidsonResult, davidson_lowest
from .models import AnharmonicInput


@dataclass(frozen=True)
class QuarticForceField:
    """Dimensionless normal-coordinate force field used by the VCI solver."""

    harmonic_frequencies_cm: np.ndarray
    cubic_cm: dict[tuple[int, int, int], float]
    quartic_cm: dict[tuple[int, int, int, int], float]


@dataclass(frozen=True)
class VCIOptions:
    """Basis, pruning and block controls for VCI."""

    active_modes: tuple[int, ...] | None = None
    frequency_min_cm: float | None = None
    frequency_max_cm: float | None = None
    basis_energy_cutoff_cm: float | None = None
    max_basis_states: int | None = None
    mode_min_quanta: tuple[int, ...] | None = None
    mode_max_quanta: tuple[int, ...] | None = None
    excitation_class_limits: dict[int, tuple[int, int | None]] = field(default_factory=dict)
    force_constant_threshold_cm: float = 0.0
    mode_symmetries: tuple[str, ...] | None = None
    separate_symmetry_blocks: bool = False
    coefficient_threshold: float = 1.0e-3

    def validate(self) -> None:
        if self.frequency_min_cm is not None and self.frequency_min_cm < 0.0:
            raise ValueError("frequency_min_cm must be non-negative")
        if self.frequency_max_cm is not None and self.frequency_max_cm <= 0.0:
            raise ValueError("frequency_max_cm must be positive")
        if (
            self.frequency_min_cm is not None
            and self.frequency_max_cm is not None
            and self.frequency_min_cm > self.frequency_max_cm
        ):
            raise ValueError("frequency_min_cm must be <= frequency_max_cm")
        if self.basis_energy_cutoff_cm is not None and self.basis_energy_cutoff_cm <= 0.0:
            raise ValueError("basis_energy_cutoff_cm must be positive")
        if self.max_basis_states is not None and self.max_basis_states < 1:
            raise ValueError("max_basis_states must be positive")
        if self.mode_min_quanta is not None and any(q < 0 for q in self.mode_min_quanta):
            raise ValueError("mode_min_quanta cannot contain negative values")
        if self.mode_max_quanta is not None and any(q < 0 for q in self.mode_max_quanta):
            raise ValueError("mode_max_quanta cannot contain negative values")
        if self.mode_min_quanta is not None and self.mode_max_quanta is not None:
            if len(self.mode_min_quanta) != len(self.mode_max_quanta):
                raise ValueError("mode_min_quanta and mode_max_quanta must have the same length")
            for qmin, qmax in zip(self.mode_min_quanta, self.mode_max_quanta):
                if qmin > qmax:
                    raise ValueError("mode_min_quanta cannot exceed mode_max_quanta")
        for n_modes, limits in self.excitation_class_limits.items():
            if n_modes < 1 or n_modes > 4:
                raise ValueError("excitation_class_limits keys must be in 1..4")
            qmin, qmax = limits
            if qmin < 0:
                raise ValueError("excitation class minimum cannot be negative")
            if qmax is not None and qmax < qmin:
                raise ValueError("excitation class maximum cannot be below minimum")
        if self.force_constant_threshold_cm < 0.0:
            raise ValueError("force_constant_threshold_cm must be non-negative")
        if self.coefficient_threshold < 0.0:
            raise ValueError("coefficient_threshold must be non-negative")


@dataclass(frozen=True)
class VCIStateContribution:
    """Mode and basis-state contributions to a final VCI eigenstate."""

    mode_quanta: np.ndarray
    dominant_basis_states: tuple[tuple[tuple[int, ...], float], ...]


@dataclass(frozen=True)
class VCIBlockInfo:
    """A symmetry block solved independently."""

    label: str
    basis_indices: tuple[int, ...]
    n_roots: int


@dataclass(frozen=True)
class VCIResult:
    basis: tuple[tuple[int, ...], ...]
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    davidson: DavidsonResult | None = None
    state_contributions: tuple[VCIStateContribution, ...] = ()
    blocks: tuple[VCIBlockInfo, ...] = ()
    options: VCIOptions = field(default_factory=VCIOptions)

    @property
    def excitation_energies_cm(self) -> np.ndarray:
        return self.energies_cm - self.energies_cm[0]


def generate_vibrational_basis(
    n_modes: int,
    max_quanta: int,
    frequencies_cm: np.ndarray | None = None,
    energy_cutoff_cm: float | None = None,
    max_basis_states: int | None = None,
    mode_min_quanta: tuple[int, ...] | None = None,
    mode_max_quanta: tuple[int, ...] | None = None,
    excitation_class_limits: dict[int, tuple[int, int | None]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Generate product harmonic-oscillator states with total quanta cutoff."""
    if n_modes < 1 or max_quanta < 0:
        raise ValueError("n_modes must be positive and max_quanta non-negative")
    if mode_min_quanta is not None and len(mode_min_quanta) != n_modes:
        raise ValueError("mode_min_quanta length must match n_modes")
    if mode_max_quanta is not None and len(mode_max_quanta) != n_modes:
        raise ValueError("mode_max_quanta length must match n_modes")
    freqs = None if frequencies_cm is None else np.asarray(frequencies_cm, dtype=float)
    states = []
    for state in product(range(max_quanta + 1), repeat=n_modes):
        if sum(state) > max_quanta:
            continue
        if mode_min_quanta is not None and any(q < qmin for q, qmin in zip(state, mode_min_quanta)):
            continue
        if mode_max_quanta is not None and any(q > qmax for q, qmax in zip(state, mode_max_quanta)):
            continue
        n_excited = sum(1 for q in state if q > 0)
        if n_excited > 0 and excitation_class_limits and n_excited in excitation_class_limits:
            class_min, class_max = excitation_class_limits[n_excited]
            total_quanta = sum(state)
            if total_quanta < class_min:
                continue
            if class_max is not None and total_quanta > class_max:
                continue
        if (
            freqs is not None
            and energy_cutoff_cm is not None
            and float(np.dot(freqs, state)) > energy_cutoff_cm
        ):
            continue
        states.append(state)
    states = sorted(
        states, key=lambda s: ((0.0 if freqs is None else float(np.dot(freqs, s))), sum(s), s)
    )
    if max_basis_states is not None:
        states = states[:max_basis_states]
    return tuple(states)


def _x_matrix_power(max_n: int, power: int) -> np.ndarray:
    x = np.zeros((max_n + 1, max_n + 1), dtype=float)
    for n in range(max_n + 1):
        if n + 1 <= max_n:
            x[n + 1, n] = np.sqrt(n + 1) / np.sqrt(2.0)
        if n - 1 >= 0:
            x[n - 1, n] = np.sqrt(n) / np.sqrt(2.0)
    out = np.eye(max_n + 1)
    for _ in range(power):
        out = out @ x
    return out


def _mode_powers(indices: tuple[int, ...], n_modes: int) -> tuple[int, ...]:
    powers = [0] * n_modes
    for idx in indices:
        if idx < 0 or idx >= n_modes:
            raise ValueError("Force-field mode index out of range")
        powers[idx] += 1
    return tuple(powers)


def _term_element(
    left: tuple[int, ...], right: tuple[int, ...], xpowers: tuple[np.ndarray, ...]
) -> float:
    value = 1.0
    for mode, power_matrix in enumerate(xpowers):
        value *= power_matrix[left[mode], right[mode]]
        if value == 0.0:
            break
    return value


def _selected_force_field(force_field: QuarticForceField, options: VCIOptions) -> QuarticForceField:
    freqs = np.asarray(force_field.harmonic_frequencies_cm, dtype=float)
    selected = (
        list(range(len(freqs))) if options.active_modes is None else list(options.active_modes)
    )
    if options.frequency_min_cm is not None:
        selected = [idx for idx in selected if freqs[idx] >= options.frequency_min_cm]
    if options.frequency_max_cm is not None:
        selected = [idx for idx in selected if freqs[idx] <= options.frequency_max_cm]
    if not selected:
        raise ValueError("VCI mode selection is empty")
    mode_map = {old: new for new, old in enumerate(selected)}

    def remap_terms(terms: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
        out: dict[tuple[int, ...], float] = {}
        for key, value in terms.items():
            if abs(value) < options.force_constant_threshold_cm:
                continue
            if any(idx not in mode_map for idx in key):
                continue
            out[tuple(sorted(mode_map[idx] for idx in key))] = value
        return out

    return QuarticForceField(
        harmonic_frequencies_cm=freqs[selected],
        cubic_cm=remap_terms(force_field.cubic_cm),
        quartic_cm=remap_terms(force_field.quartic_cm),
    )


def _selected_quanta_limits(
    force_field: QuarticForceField,
    options: VCIOptions,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    freqs = np.asarray(force_field.harmonic_frequencies_cm, dtype=float)
    selected = (
        list(range(len(freqs))) if options.active_modes is None else list(options.active_modes)
    )
    if options.frequency_min_cm is not None:
        selected = [idx for idx in selected if freqs[idx] >= options.frequency_min_cm]
    if options.frequency_max_cm is not None:
        selected = [idx for idx in selected if freqs[idx] <= options.frequency_max_cm]

    def select(values: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if values is None:
            return None
        if len(values) != len(freqs):
            raise ValueError("Per-mode quanta limits must match the original mode count")
        return tuple(values[idx] for idx in selected)

    return select(options.mode_min_quanta), select(options.mode_max_quanta)


def _selected_mode_symmetries(
    force_field: QuarticForceField, options: VCIOptions
) -> tuple[str, ...] | None:
    if options.mode_symmetries is None:
        return None
    freqs = np.asarray(force_field.harmonic_frequencies_cm, dtype=float)
    if len(options.mode_symmetries) != len(freqs):
        raise ValueError("mode_symmetries length must match the original mode count")
    selected = (
        list(range(len(freqs))) if options.active_modes is None else list(options.active_modes)
    )
    if options.frequency_min_cm is not None:
        selected = [idx for idx in selected if freqs[idx] >= options.frequency_min_cm]
    if options.frequency_max_cm is not None:
        selected = [idx for idx in selected if freqs[idx] <= options.frequency_max_cm]
    return tuple(options.mode_symmetries[idx] for idx in selected)


def _state_symmetry(state: tuple[int, ...], labels: tuple[str, ...] | None) -> str:
    if labels is None:
        return "all"
    odd = sorted(
        {label for quanta, label in zip(state, labels) if quanta % 2 == 1 and label != "A"}
    )
    return "A" if not odd else "*".join(odd)


def _symmetry_blocks(
    basis: tuple[tuple[int, ...], ...],
    mode_symmetries: tuple[str, ...] | None,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    blocks: dict[str, list[int]] = {}
    for idx, state in enumerate(basis):
        blocks.setdefault(_state_symmetry(state, mode_symmetries), []).append(idx)
    return tuple((label, tuple(indices)) for label, indices in sorted(blocks.items()))


def build_vci_hamiltonian(
    force_field: QuarticForceField,
    max_quanta: int,
    options: VCIOptions | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Build a small dense VCI Hamiltonian in cm^-1.

    Large spaces will later use the same matrix-element code behind Davidson.
    """
    opts = options or VCIOptions()
    opts.validate()
    mode_min, mode_max = _selected_quanta_limits(force_field, opts)
    force_field = _selected_force_field(force_field, opts)
    freqs = np.asarray(force_field.harmonic_frequencies_cm, dtype=float)
    if np.any(freqs <= 0.0):
        raise ValueError("Harmonic frequencies must be positive")
    n_modes = len(freqs)
    basis = generate_vibrational_basis(
        n_modes,
        max_quanta,
        freqs,
        opts.basis_energy_cutoff_cm,
        opts.max_basis_states,
        mode_min,
        mode_max,
        opts.excitation_class_limits,
    )
    max_n = max(max(state) for state in basis)
    operator_max_n = max_n + 4
    powers = {p: _x_matrix_power(operator_max_n, p) for p in range(5)}
    h = np.zeros((len(basis), len(basis)), dtype=float)

    for i, state in enumerate(basis):
        h[i, i] = float(np.dot(freqs, np.array(state, dtype=float) + 0.5))

    terms: list[tuple[tuple[int, ...], float]] = []
    terms.extend((tuple(sorted(k)), v) for k, v in force_field.cubic_cm.items())
    terms.extend((tuple(sorted(k)), v) for k, v in force_field.quartic_cm.items())
    for indices, coeff in terms:
        mode_powers = _mode_powers(indices, n_modes)
        xpowers = tuple(powers[p] for p in mode_powers)
        for row, left in enumerate(basis):
            for col, right in enumerate(basis[: row + 1]):
                element = coeff * _term_element(left, right, xpowers)
                if element:
                    h[row, col] += element
                    if row != col:
                        h[col, row] += element
    return h, basis


def _analyze_states(
    basis: tuple[tuple[int, ...], ...],
    eigenvectors: np.ndarray,
    threshold: float,
) -> tuple[VCIStateContribution, ...]:
    basis_array = np.array(basis, dtype=float)
    out = []
    for root in range(eigenvectors.shape[1]):
        coeffs = eigenvectors[:, root]
        weights = coeffs * coeffs
        mode_quanta = weights @ basis_array
        dominant = []
        for idx in np.argsort(-weights):
            if weights[idx] < threshold:
                continue
            dominant.append((basis[idx], float(weights[idx])))
        out.append(
            VCIStateContribution(mode_quanta=mode_quanta, dominant_basis_states=tuple(dominant))
        )
    return tuple(out)


def solve_vci(
    force_field: QuarticForceField,
    max_quanta: int,
    n_roots: int | None = None,
    *,
    method: str = "dense",
    options: VCIOptions | None = None,
    max_subspace: int = 80,
    max_iter: int = 200,
    convergence: float = 1.0e-8,
) -> VCIResult:
    opts = options or VCIOptions()
    opts.validate()
    selected_symmetries = _selected_mode_symmetries(force_field, opts)
    hamiltonian, basis = build_vci_hamiltonian(force_field, max_quanta, opts)
    sym_hamiltonian = (hamiltonian + hamiltonian.T) * 0.5
    block_infos: tuple[VCIBlockInfo, ...] = ()
    if method == "dense" and opts.separate_symmetry_blocks:
        block_results = []
        infos = []
        for label, indices in _symmetry_blocks(basis, selected_symmetries):
            sub_h = sym_hamiltonian[np.ix_(indices, indices)]
            sub_e, sub_v = eigh_arrays(sub_h)
            for col, energy in enumerate(sub_e):
                full = np.zeros(len(basis), dtype=float)
                full[list(indices)] = sub_v[:, col]
                block_results.append((float(energy), full, label))
            infos.append(VCIBlockInfo(label=label, basis_indices=indices, n_roots=len(sub_e)))
        block_results.sort(key=lambda item: item[0])
        eig = np.array([item[0] for item in block_results], dtype=float)
        vec = (
            np.column_stack([item[1] for item in block_results])
            if block_results
            else np.zeros((len(basis), 0))
        )
        davidson = None
        block_infos = tuple(infos)
    elif method == "dense":
        eig, vec = eigh_arrays(sym_hamiltonian)
        davidson = None
    elif method == "davidson":
        roots = n_roots or min(10, sym_hamiltonian.shape[0])
        davidson = davidson_lowest(
            lambda vector: sym_hamiltonian @ vector,
            np.diag(sym_hamiltonian),
            n_roots=roots,
            max_subspace=max_subspace,
            max_iter=max_iter,
            convergence=convergence,
            block_matvec=lambda vectors: sym_hamiltonian @ vectors,
        )
        eig = davidson.eigenvalues
        vec = davidson.eigenvectors
    else:
        raise ValueError("VCI method must be 'dense' or 'davidson'")
    if n_roots is not None:
        eig = eig[:n_roots]
        vec = vec[:, :n_roots]
    contributions = _analyze_states(basis, vec, opts.coefficient_threshold)
    return VCIResult(
        basis=basis,
        energies_cm=eig,
        eigenvectors=vec,
        davidson=davidson,
        state_contributions=contributions,
        blocks=block_infos,
        options=opts,
    )


def zero_anharmonic_force_field(frequencies_cm: np.ndarray) -> QuarticForceField:
    return QuarticForceField(np.asarray(frequencies_cm, dtype=float), {}, {})


def force_field_from_anharmonic_input(input_data: AnharmonicInput) -> QuarticForceField:
    """Convert canonical ORACLE anharmonic input to the VCI force-field model."""
    input_data.validate()
    frequencies = (
        input_data.anharmonic_frequencies_cm
        if input_data.anharmonic_frequencies_cm.size
        else input_data.harmonic_frequencies_cm
    )
    return QuarticForceField(
        harmonic_frequencies_cm=np.asarray(frequencies, dtype=float),
        cubic_cm=dict(input_data.cubic_cm),
        quartic_cm=dict(input_data.quartic_cm),
    )


def solve_vci_from_anharmonic_input(
    input_data: AnharmonicInput,
    max_quanta: int,
    n_roots: int | None = None,
    **kwargs: object,
) -> VCIResult:
    """Run VCI from canonical ORACLE anharmonic input."""
    return solve_vci(
        force_field_from_anharmonic_input(input_data),
        max_quanta=max_quanta,
        n_roots=n_roots,
        **kwargs,
    )


def empty_symmetric_terms(n_modes: int, order: int) -> dict[tuple[int, ...], float]:
    """Return all same-index combinations initialized to zero for tests/UI forms."""
    if order not in {3, 4}:
        raise ValueError("Only cubic and quartic terms are supported")
    return {tuple(idx): 0.0 for idx in combinations_with_replacement(range(n_modes), order)}
