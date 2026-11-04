"""Adaptive canonical-virtual convergence for a fixed imported HF/DFT reference.

State overlaps are evaluated in the shared canonical MO configuration basis.
This is a fixed-geometry truncation controller, not a geometry/root optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .electronic_state import _readonly_array
from .imported_tda import (
    ImportedTDAReference, ImportedTDAResult, ImportedTDASelection,
    OhnoTDAParameters, imported_ohno_tda,
)
from .sf_tda import HARTREE_TO_EV
from .tda_state_tracking import (
    TDAStateTrackingPolicy, TDAStateTrackingResult, track_imported_tda_states,
)


@dataclass(frozen=True)
class TDAVirtualConvergencePolicy:
    windows_ev: tuple[float, ...] = (40., 80., 160., 320.)
    energy_tolerance_ev: float = .01
    check_oscillator_strengths: bool = True
    strength_absolute_tolerance: float = 1e-4
    strength_relative_tolerance: float = .01
    minimum_overlap: float = .9
    minimum_margin: float = .1
    stable_enlargements: int = 2
    confirmation_factor: float = 2.
    candidate_buffer: int = 6
    candidate_root_limit: int = 64
    full_space_fallback: bool = True
    maximum_virtuals: int | None = None
    tracking_policy: TDAStateTrackingPolicy = field(default_factory=TDAStateTrackingPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.tracking_policy, TDAStateTrackingPolicy):
            raise TypeError("tracking_policy must be TDAStateTrackingPolicy")
        windows = tuple(float(x) for x in self.windows_ev)
        if not windows or not np.all(np.isfinite(windows)) or windows[0] <= 0 or np.any(np.diff(windows) <= 0):
            raise ValueError("windows_ev must be a nonempty strictly increasing positive sequence")
        object.__setattr__(self, "windows_ev", windows)
        for name in ("energy_tolerance_ev", "strength_absolute_tolerance", "strength_relative_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.minimum_overlap) or not 0 < self.minimum_overlap <= 1:
            raise ValueError("minimum_overlap must lie in (0, 1]")
        if not np.isfinite(self.minimum_margin) or not 0 <= self.minimum_margin <= 1:
            raise ValueError("minimum_margin must lie in [0, 1]")
        for name in ("stable_enlargements", "candidate_buffer", "candidate_root_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.confirmation_factor) or self.confirmation_factor <= 1:
            raise ValueError("confirmation_factor must exceed one")
        if self.maximum_virtuals is not None and (
            isinstance(self.maximum_virtuals, bool)
            or not isinstance(self.maximum_virtuals, (int, np.integer)) or self.maximum_virtuals < 1
        ):
            raise ValueError("maximum_virtuals must be a positive integer or None")


@dataclass(frozen=True)
class TDAVirtualConvergenceStep:
    window_ev: float
    virtual_count: int
    candidate_count: int
    target_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    matched_roots: tuple[int, ...]
    energies_ev: np.ndarray | None
    oscillator_strengths: np.ndarray | None
    overlaps: np.ndarray | None
    margins: np.ndarray | None
    energy_changes_ev: np.ndarray | None
    strength_changes: np.ndarray | None
    reliable: bool
    stable: bool
    stable_enlargements: int
    status: str
    full_space: bool
    tracking: TDAStateTrackingResult | None = None

    def __post_init__(self) -> None:
        for name in ("energies_ev", "oscillator_strengths", "overlaps", "margins",
                     "energy_changes_ev", "strength_changes"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly_array(value))


@dataclass(frozen=True)
class TDAVirtualConvergenceResult:
    converged: bool
    reason: str
    mode: str
    policy: TDAVirtualConvergencePolicy
    candidate_result: ImportedTDAResult | None
    target_roots: tuple[int, ...]
    history: tuple[TDAVirtualConvergenceStep, ...]

    @property
    def excitation_energies_hartree(self) -> np.ndarray | None:
        if self.candidate_result is None or not self.target_roots:
            return None
        return _readonly_array(self.candidate_result.excitation_energies_hartree[
            np.asarray(self.target_roots) - 1])

    @property
    def oscillator_strengths(self) -> np.ndarray | None:
        if self.candidate_result is None or self.candidate_result.oscillator_strengths is None or not self.target_roots:
            return None
        return _readonly_array(self.candidate_result.oscillator_strengths[
            np.asarray(self.target_roots) - 1])


def _match_canonical_states(source: ImportedTDAResult, source_indices: np.ndarray,
                            candidate: ImportedTDAResult):
    """Phase-insensitive overlaps in a common MO basis, with two-sided margins."""
    if source.virtual_space is not None or candidate.virtual_space is not None:
        raise ValueError("automatic energy-window convergence requires canonical virtuals")
    positions = {int(a): j for j, a in enumerate(candidate.retained_virtual_orbitals)}
    try:
        columns = [positions[int(a)] for a in source.retained_virtual_orbitals]
    except KeyError as exc:
        raise ValueError("candidate virtual spaces must contain the source virtual space") from exc
    no = len(source.occupied_orbitals) // len(source.retained_virtual_orbitals)
    old = source.amplitudes[source_indices].reshape(len(source_indices), no, -1)
    new = candidate.amplitudes.reshape(len(candidate.amplitudes), no, -1)[:, :, columns]
    scores = np.abs(np.einsum("ria,sia->rs", old, new))
    rows, cols = linear_sum_assignment(-scores)
    assignment = np.full(len(source_indices), -1, dtype=int)
    overlaps = np.zeros(len(source_indices))
    margins = np.zeros(len(source_indices))
    for i, j in zip(rows, cols):
        assignment[i] = j
        overlaps[i] = scores[i, j]
        other_new = np.max(np.delete(scores[i], j), initial=0)
        other_old = np.max(np.delete(scores[:, j], i), initial=0)
        margins[i] = scores[i, j] - max(other_new, other_old)
    return assignment, overlaps, margins


def converge_imported_ohno_tda(
    reference: ImportedTDAReference, *, parameters: OhnoTDAParameters,
    policy: TDAVirtualConvergencePolicy | None = None, roots: int = 5,
    target_roots: tuple[int, ...] | None = None,
    tolerance_hartree: float = 1e-9, maximum_iterations: int = 500,
    maximum_subspace: int | None = None, backend: str = "auto",
) -> TDAVirtualConvergenceResult:
    """Enlarge canonical virtual spaces and follow states rather than root ranks.

    Default: converge the lowest ``roots`` states, restarting stability when
    another state enters that set. ``target_roots`` instead selects one-based
    roots from the first solved window and follows those states in that order;
    its length must equal ``roots``. Ambiguous fixed-state matches retain the
    last reliable anchor, allowing recovery at a subsequent larger window.

    Window stability is a convergence estimate, not a bound on the omitted tail.
    Two stable enlargements are followed by a larger confirmation window. If
    permitted, reaching all supplied virtuals eliminates truncation error.
    Budgets, unresolved identities and solver failures never imply convergence.
    """
    policy = policy or TDAVirtualConvergencePolicy()
    if isinstance(roots, bool) or not isinstance(roots, (int, np.integer)) or roots < 1:
        raise ValueError("roots must be a positive integer")
    fixed = target_roots is not None
    seed_roots = tuple(range(1, roots + 1)) if target_roots is None else tuple(target_roots)
    if len(seed_roots) != roots or len(set(seed_roots)) != roots or any(
        isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 1 for r in seed_roots
    ):
        raise ValueError("target_roots must contain roots distinct positive integer indices")
    if max(seed_roots) > policy.candidate_root_limit:
        raise ValueError("candidate_root_limit is smaller than a requested seed root")
    if maximum_subspace is not None and (
        isinstance(maximum_subspace, bool) or not isinstance(maximum_subspace, (int, np.integer))
        or maximum_subspace <= max(seed_roots)
    ):
        raise ValueError("maximum_subspace must exceed every requested seed root")
    if policy.check_oscillator_strengths and reference.dipole_ao is None:
        raise ValueError("intensity convergence requires AO dipoles; explicitly disable it for energy-only convergence")
    occupied = np.flatnonzero(reference.occupations > 1)
    virtual = np.flatnonzero(reference.occupations < 1)
    homo = float(np.max(reference.orbital_energies_hartree[occupied]))
    energies = reference.orbital_energies_hartree[virtual]
    pending = list(policy.windows_ev)
    if policy.full_space_fallback:
        full_window = max(pending[-1], float((np.max(energies) - homo) * HARTREE_TO_EV) + 1e-6)
        if full_window > pending[-1]:
            pending.append(full_window)
    maximum_window = pending[-1]
    anchor = last = None
    anchor_indices = np.array([], dtype=int)
    selected: tuple[int, ...] = ()
    history = []
    stable_count = 0
    confirmation = None
    previous_virtual_count = -1

    def finish(ok, reason):
        return TDAVirtualConvergenceResult(ok, reason, "fixed_states" if fixed else "lowest_roots",
                                           policy, last, selected, tuple(history))

    while pending:
        window = pending.pop(0)
        count = int(np.count_nonzero(energies <= homo + window / HARTREE_TO_EV))
        full_space = count == len(virtual)
        status = None
        if count == 0:
            status = "empty_virtual_space"
        elif policy.maximum_virtuals is not None and count > policy.maximum_virtuals:
            status = "virtual_budget_exhausted"
        elif count == previous_virtual_count:
            status = "unchanged_virtual_space"
        elif len(occupied) * count < max(seed_roots):
            status = "insufficient_states"
        if status is not None:
            history.append(TDAVirtualConvergenceStep(window, count, 0, (), (), (), None, None,
                None, None, None, None, False, False, stable_count, status, full_space))
            if status == "virtual_budget_exhausted":
                return finish(False, status)
            if full_space and status == "insufficient_states":
                return finish(False, status)
            continue
        source_roots = tuple(int(i + 1) for i in anchor_indices)
        nconfig = len(occupied) * count
        capacity = min(nconfig, policy.candidate_root_limit)
        if maximum_subspace is not None:
            capacity = min(capacity, maximum_subspace - 1)
        candidates = min(capacity, max(max(seed_roots) + policy.candidate_buffer,
                                       len(last.amplitudes) if last is not None else 0))
        assignment = overlaps = margins = None
        tracking = None
        try:
            while True:
                last = imported_ohno_tda(reference, parameters=parameters,
                    selection=ImportedTDASelection(window), roots=candidates,
                    tolerance_hartree=tolerance_hartree, maximum_iterations=maximum_iterations,
                    maximum_subspace=maximum_subspace, backend=backend)
                if not last.converged:
                    selected = ()
                    history.append(TDAVirtualConvergenceStep(window, count, candidates, (), source_roots, (),
                        None, None, None, None, None, None, False, False, 0,
                        "eigensolver_not_converged", full_space))
                    return finish(False, "eigensolver_not_converged")
                if anchor is None:
                    break
                tracking = track_imported_tda_states(reference, anchor, reference, last,
                    source_state_indices=anchor_indices, policy=replace(policy.tracking_policy,
                        minimum_overlap=policy.minimum_overlap, minimum_margin=policy.minimum_margin))
                assignment = np.array([m.candidate_index for m in tracking.matches])
                overlaps = np.array([m.amplitude_overlap for m in tracking.matches])
                margins = np.array([m.margin for m in tracking.matches])
                missing = any(m.candidate_index < 0 or (
                    not m.individually_resolved and not tracking.subspaces[m.subspace_index].resolved)
                    for m in tracking.matches)
                boundary = np.max(assignment, initial=-1) + 1 + policy.candidate_buffer > candidates
                if not (missing or boundary) or candidates == capacity:
                    break
                candidates = min(capacity, max(2 * candidates, int(np.max(assignment)) + 1 + policy.candidate_buffer))
        except RuntimeError as exc:
            selected = ()
            history.append(TDAVirtualConvergenceStep(window, count, candidates, (), source_roots, (),
                None, None, None, None, None, None, False, False, 0,
                f"eigensolver_failure: {exc}", full_space))
            return finish(False, "eigensolver_failure")
        previous_virtual_count = count
        stable = False
        reliable = anchor is None
        delta_e = delta_f = None
        matched_roots = ()
        if anchor is None:
            chosen = np.asarray(seed_roots, dtype=int) - 1
            status = "initial_states"
        else:
            matched_roots = tuple(int(i + 1) for i in assignment)
            selected_sources = set(map(int, anchor_indices))
            reliable = all(
                m.candidate_index >= 0 and (m.individually_resolved or (
                    group.resolved and set(group.source_indices) <= selected_sources))
                for m in tracking.matches
                for group in (tracking.subspaces[m.subspace_index],))
            closed = fixed or set(assignment) == set(range(roots))
            if np.all(assignment >= 0):
                delta_e = (last.excitation_energies_hartree[assignment]
                           - anchor.excitation_energies_hartree[anchor_indices]) * HARTREE_TO_EV
                if last.oscillator_strengths is not None and anchor.oscillator_strengths is not None:
                    delta_f = last.oscillator_strengths[assignment] - anchor.oscillator_strengths[anchor_indices]
            # Near-degenerate rotations preserve a block, not individual strengths.
            # Compare its sorted eigenvalues and summed oscillator strength.
            covered = set()
            block_stable = True
            for group in tracking.subspaces:
                if len(group.source_indices) < 2 or not group.resolved:
                    continue
                positions = [k for k,i in enumerate(anchor_indices) if i in group.source_indices]
                if len(positions) != len(group.source_indices):
                    continue
                covered.update(positions)
                before = anchor.excitation_energies_hartree[list(group.source_indices)]
                after = last.excitation_energies_hartree[list(group.candidate_indices)]
                block_stable = block_stable and bool(np.all(
                    np.abs(np.sort(after)-np.sort(before))*HARTREE_TO_EV <= policy.energy_tolerance_ev))
                if policy.check_oscillator_strengths:
                    if last.oscillator_strengths is None or anchor.oscillator_strengths is None:
                        block_stable = False
                    else:
                        old_strength = float(np.sum(anchor.oscillator_strengths[list(group.source_indices)]))
                        new_strength = float(np.sum(last.oscillator_strengths[list(group.candidate_indices)]))
                        block_stable = block_stable and abs(new_strength-old_strength) <= (
                            policy.strength_absolute_tolerance
                            + policy.strength_relative_tolerance*max(abs(old_strength),abs(new_strength)))
            individual = [k for k in range(len(anchor_indices)) if k not in covered]
            properties_stable = delta_e is not None and block_stable and np.all(
                np.abs(delta_e[individual]) <= policy.energy_tolerance_ev)
            if policy.check_oscillator_strengths:
                if delta_f is None:
                    properties_stable = False
                else:
                    scale = np.maximum(np.abs(last.oscillator_strengths[assignment]),
                                       np.abs(anchor.oscillator_strengths[anchor_indices]))
                    properties_stable = properties_stable and np.all(np.abs(delta_f[individual]) <=
                        policy.strength_absolute_tolerance + policy.strength_relative_tolerance * scale[individual])
            stable = bool(reliable and closed and properties_stable)
            stable_count = stable_count + 1 if stable else 0
            chosen = assignment if fixed else np.arange(roots)
            status = "stable_enlargement" if stable else (
                "unresolved_tracking" if not reliable else "lowest_set_changed" if not closed else "properties_changed")
        selected = tuple(int(i + 1) for i in chosen) if np.all(chosen >= 0) and reliable else ()
        e_selected = last.excitation_energies_hartree[chosen] * HARTREE_TO_EV if selected else None
        f_selected = last.oscillator_strengths[chosen] if selected and last.oscillator_strengths is not None else None
        is_confirmation = confirmation is not None and window >= confirmation
        if full_space:
            status = "full_space_reached" if reliable else "full_space_tracking_unresolved"
        elif is_confirmation:
            status = "confirmed_window_stability" if stable else "confirmation_failed"
        history.append(TDAVirtualConvergenceStep(window, count, candidates, selected, source_roots,
            matched_roots, e_selected, f_selected, overlaps, margins, delta_e, delta_f,
            reliable, stable, stable_count, status, full_space, tracking))
        if full_space:
            return finish(reliable, status)
        if is_confirmation and stable:
            return finish(True, status)
        if is_confirmation:
            confirmation = None
        if reliable:
            anchor, anchor_indices = last, chosen
        if stable_count >= policy.stable_enlargements and confirmation is None:
            guard = min(maximum_window, window * policy.confirmation_factor)
            if guard > window:
                confirmation = guard
                pending = [guard] + [x for x in pending if x > guard]
    return finish(False, "window_schedule_exhausted")
