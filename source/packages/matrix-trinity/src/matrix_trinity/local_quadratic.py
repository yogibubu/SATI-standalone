"""Quadratic local/bath coupling with variational GVPT2 polyads.

Input diagonal energies are deperturbed VPT2 levels: nonresonant C/Q and
analytic external-K KCC/KQ contractions may already be included.  They must
exclude every selected intra-polyad contribution and the external K^2
self-energy added here.  Supplied anharmonic edges are the effective
Fermi/Darling--Dennison matrix elements.  This partition prevents double
counting; callers must form it before building the model.
Only selected, transitively closed polyads are diagonalized.  Their K_AB
couplings are included to all orders (and their analytic cubic coefficient
is available for audit).  K_AB outside polyads is retained only through
second order, without finite differences or a full-space VCI.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping, Sequence

import numpy as np

EdgeMap = Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class PolyadEdge:
    left: int
    right: int
    gap_cm: float
    coupling_cm: float
    two_level_shift_cm: float
    mixing_weight: float
    variational: bool


@dataclass(frozen=True)
class CouplingSeries:
    """Taylor coefficients (not derivatives) in the intra-polyad K_AB."""

    root: int
    energy_cm: np.ndarray  # E(eta) = sum_k energy_cm[k] eta**k
    wavefunctions: tuple[np.ndarray, ...]  # intermediate normalization


@dataclass(frozen=True)
class TransitionSeries:
    frequency_cm: np.ndarray
    transition_moment: np.ndarray  # (order 0..3, Cartesian component)
    intensity: np.ndarray  # chosen conversion factor times cm^-1 * dipole^2


@dataclass(frozen=True)
class VariationalTransition:
    frequency_cm: float
    transition_moment: np.ndarray
    intensity: float


@dataclass(frozen=True)
class OperativeSpectrum:
    energies_cm: np.ndarray
    root_assignment: tuple[int, ...]
    assignment_overlap: tuple[float, ...]
    eigenvectors: tuple[np.ndarray, ...] = ()  # local basis rows, operative roots columns


def _canonical_edges(edges: EdgeMap, size: int, name: str) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for (i, j), value in edges.items():
        if not (isinstance(i, int) and isinstance(j, int)
                and 0 <= i < size and 0 <= j < size and i != j):
            raise ValueError(f"{name} edges require distinct valid state indices")
        if not np.isfinite(value):
            raise ValueError(f"{name} edges must be finite")
        key = (min(i, j), max(i, j))
        if key in result:
            raise ValueError(f"{name} edge {key} was supplied twice")
        if value:
            result[key] = float(value)
    return result


def _apply_edges(edges: tuple[tuple[int, int, float], ...], vector: np.ndarray) -> np.ndarray:
    result = np.zeros_like(vector)
    for i, j, value in edges:
        result[i] += value * vector[j]
        result[j] += value * vector[i]
    return result


def _dipole_element(left: np.ndarray, right: np.ndarray, edges: EdgeMap) -> float:
    size = len(left)
    value = 0.0
    seen = set()
    for (i, j), coefficient in edges.items():
        if not (0 <= i < size and 0 <= j < size and np.isfinite(coefficient)):
            raise ValueError("invalid dipole matrix element")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError("dipole matrix element was supplied twice")
        seen.add(key)
        value += float(coefficient) * (
            left[i] * right[j] if i == j else left[i] * right[j] + left[j] * right[i]
        )
    return value


def _maximum_overlap_assignment(weights: np.ndarray) -> tuple[int, ...]:
    """Unique row-to-column assignment maximizing total squared overlap."""
    size = len(weights)
    if weights.shape != (size, size) or not np.all(np.isfinite(weights)):
        raise ValueError("overlap weights must form a finite square matrix")
    costs = -weights
    row_potential = np.zeros(size + 1)
    col_potential = np.zeros(size + 1)
    matched_row = np.zeros(size + 1, dtype=int)
    predecessor = np.zeros(size + 1, dtype=int)
    for row in range(1, size + 1):
        matched_row[0] = row
        column = 0
        minimum = np.full(size + 1, np.inf)
        visited = np.zeros(size + 1, dtype=bool)
        while True:
            visited[column] = True
            current_row = matched_row[column]
            step = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if visited[candidate]:
                    continue
                reduced = (costs[current_row - 1, candidate - 1]
                           - row_potential[current_row] - col_potential[candidate])
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < step:
                    step = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if visited[candidate]:
                    row_potential[matched_row[candidate]] += step
                    col_potential[candidate] -= step
                else:
                    minimum[candidate] -= step
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = [0] * size
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return tuple(assignment)


@dataclass(frozen=True)
class LocalQuadraticModel:
    """Small variational polyads and second-order external K_AB."""

    reference_energies_cm: np.ndarray
    polyads: tuple[tuple[int, ...], ...]
    polyad_vectors: tuple[np.ndarray, ...]
    variational_energies_cm: np.ndarray
    variational_vectors: tuple[np.ndarray, ...]
    root_assignment: tuple[int, ...]  # original state index -> variational root
    assignment_overlap: tuple[float, ...]
    polyad_k_edges: tuple[tuple[int, int, float], ...]
    external_k_edges: tuple[tuple[int, int, float], ...]
    selection: tuple[PolyadEdge, ...]
    denominator_tolerance_cm: float
    numerical_zero_cm: float

    def _rotate(self, vector: np.ndarray, *, transpose: bool,
                variational: bool = False) -> np.ndarray:
        result = np.array(vector, dtype=float, copy=True)
        blocks = self.variational_vectors if variational else self.polyad_vectors
        for members, vectors in zip(self.polyads, blocks):
            indices = list(members)
            result[indices] = (vectors.T if transpose else vectors) @ vector[indices]
        return result

    def reference_vector(self, root: int, *, variational: bool = False) -> np.ndarray:
        if not 0 <= root < len(self.reference_energies_cm):
            raise IndexError("reference root is outside the retained state basis")
        unit = np.zeros(len(self.reference_energies_cm))
        unit[root] = 1.0
        return self._rotate(unit, transpose=False, variational=variational)

    def _resolvent(self, root: int, rhs: np.ndarray) -> np.ndarray:
        transformed = self._rotate(rhs, transpose=True)
        gaps = self.reference_energies_cm[root] - self.reference_energies_cm
        near = abs(gaps) <= self.denominator_tolerance_cm
        if np.any(abs(transformed[near & (np.arange(len(gaps)) != root)])
                  > self.numerical_zero_cm):
            raise ValueError("unresolved intra-polyad K denominator: diagonalize the polyad")
        transformed[near] = 0.0
        transformed[~near] /= gaps[~near]
        return self._rotate(transformed, transpose=False)

    def coupling_series(self, root: int) -> CouplingSeries:
        """Analytic K_AB^3 coefficient *inside* one selected polyad.

        If R=Q(E0-Href)^-1 Q, w1=R K|0>, w2=R(K-E1)w1,
        E3=<0|K|w2>.  These coefficients diagnose the variational
        result; they are never added to it a second time.  A singleton
        outside all polyads has zero intra-polyad coefficients.
        """
        vector = self.reference_vector(root)
        if not any(root in members for members in self.polyads):
            zeros = np.zeros_like(vector)
            return CouplingSeries(
                root=root,
                energy_cm=np.array([self.reference_energies_cm[root], 0., 0., 0.]),
                wavefunctions=(vector, zeros.copy(), zeros.copy(), zeros.copy()),
            )
        apply_k = lambda v: _apply_edges(self.polyad_k_edges, v)
        first = float(vector @ apply_k(vector))
        w1 = self._resolvent(root, apply_k(vector) - first * vector)
        second = float(vector @ apply_k(w1))
        w2 = self._resolvent(root, apply_k(w1) - first * w1 - second * vector)
        third = float(vector @ apply_k(w2))
        w3 = self._resolvent(root, apply_k(w2) - first * w2
                             - second * w1 - third * vector)
        return CouplingSeries(
            root=root,
            energy_cm=np.array([self.reference_energies_cm[root], first, second, third]),
            wavefunctions=(vector, w1, w2, w3),
        )

    def external_second_order_block(self, members: tuple[int, ...]) -> np.ndarray:
        """Symmetric K_AB^2 self-energy between variational roots of a block."""
        if not members or (len(members) > 1 and members not in self.polyads):
            raise ValueError("members must be one complete variational polyad or a singleton")
        if len(members) == 1 and any(members[0] in block for block in self.polyads):
            raise ValueError("a polyad cannot be split for the external self-energy")
        energies = self.variational_energies_cm
        vertices = np.array(members, dtype=int)
        outside = np.ones(len(energies), dtype=bool)
        outside[vertices] = False
        actions = []
        for root in members:
            vector = self.reference_vector(root, variational=True)
            action = self._rotate(_apply_edges(self.external_k_edges, vector),
                                  transpose=True, variational=True)
            gaps = energies[root] - energies
            bad = outside & (abs(gaps) <= self.denominator_tolerance_cm)
            if np.any(abs(action[bad]) > self.numerical_zero_cm):
                raise ValueError("unresolved external-K denominator: merge the variational polyads")
            reciprocal = np.zeros(len(energies))
            safe = outside & ~bad
            reciprocal[safe] = 1 / gaps[safe]
            actions.append((action, reciprocal))
        result = np.zeros((len(members), len(members)))
        for a, (action_a, inverse_a) in enumerate(actions):
            for b in range(a, len(members)):
                action_b, inverse_b = actions[b]
                result[a, b] = result[b, a] = float(
                    action_a @ ((inverse_a + inverse_b) * action_b) / 2
                )
        return result

    def operative_spectrum(self) -> OperativeSpectrum:
        """Exact intra-polyad K plus external K^2 with unique root matching."""
        result = self.variational_energies_cm.copy()
        assignment = list(range(len(result)))
        overlaps = [1.0] * len(result)
        operative_vectors = []
        in_polyad = {state for block in self.polyads for state in block}
        for members, variational_vectors in zip(self.polyads, self.variational_vectors):
            block = np.diag(result[list(members)]) + self.external_second_order_block(members)
            energies, correction_vectors = np.linalg.eigh(block)
            result[list(members)] = energies
            total_vectors = variational_vectors @ correction_vectors
            operative_vectors.append(total_vectors)
            local_assignment = _maximum_overlap_assignment(abs(total_vectors) ** 2)
            for row, column in enumerate(local_assignment):
                assignment[members[row]] = members[column]
                overlaps[members[row]] = float(abs(total_vectors[row, column]) ** 2)
        for root in range(len(result)):
            if root not in in_polyad:
                result[root] += self.external_second_order_block((root,))[0, 0]
        return OperativeSpectrum(result, tuple(assignment), tuple(overlaps),
                                 tuple(operative_vectors))

    def operative_energies_cm(self) -> np.ndarray:
        return self.operative_spectrum().energies_cm

    def variational_transition(
        self, ground: int, excited: int, dipole_edges: Sequence[EdgeMap], *,
        intensity_factor: float = 1.0,
    ) -> VariationalTransition:
        """Exact intra-polyad K mixing in the energy and dipole strength.

        External-K corrections to intensities belong to the separate VPT2
        transition operator; this method does not silently add them.
        """
        if not dipole_edges or not np.isfinite(intensity_factor) or intensity_factor < 0:
            raise ValueError("invalid dipole components or intensity factor")
        g = self.reference_vector(ground, variational=True)
        e = self.reference_vector(excited, variational=True)
        moment = np.array([_dipole_element(e, g, component)
                           for component in dipole_edges])
        frequency = float(self.variational_energies_cm[excited]
                          - self.variational_energies_cm[ground])
        return VariationalTransition(
            frequency_cm=frequency, transition_moment=moment,
            intensity=float(intensity_factor * frequency * (moment @ moment)),
        )

    def assigned_variational_energy_cm(self, state: int) -> float:
        """Energy of the uniquely overlap-assigned root for a basis state."""
        return float(self.variational_energies_cm[self.root_assignment[state]])


def build_local_quadratic_model(
    deperturbed_energies_cm: Sequence[float],
    anharmonic_edges: EdgeMap,
    quadratic_edges: EdgeMap,
    *,
    frequency_tolerance_cm: float = 1.0,
    mixing_weight_tolerance: float = 0.01,
    denominator_tolerance_cm: float = 1e-6,
    numerical_zero_cm: float = 1e-10,
    max_polyad_states: int = 100,
) -> LocalQuadraticModel:
    """Select and diagonalize resonance components, retaining all internal edges.

    An edge enters a polyad if its exact two-level shift reaches the requested
    frequency tolerance OR its mixing weight reaches the specified intensity-
    borrowing diagnostic.  Connected components give the *complete* polyads;
    no arbitrary frequency window removes dark partner states.  K edges
    within each component are diagonalized; only inter-component K edges
    enter the second-order self-energy.  When anharmonic and K operators
    connect the same pair, both individual terms and their sum are screened.
    ``deperturbed_energies_cm`` must already include any desired nonresonant
    KCC/KQ correction and exclude the explicit K terms treated here.
    """
    energies = np.asarray(deperturbed_energies_cm, dtype=float)
    if energies.ndim != 1 or not len(energies) or not np.all(np.isfinite(energies)):
        raise ValueError("deperturbed energies must be a finite nonempty vector")
    if (not np.isfinite(frequency_tolerance_cm) or frequency_tolerance_cm <= 0
            or not np.isfinite(mixing_weight_tolerance)
            or not 0 < mixing_weight_tolerance < 0.5
            or not np.isfinite(denominator_tolerance_cm)
            or denominator_tolerance_cm <= 0
            or not np.isfinite(numerical_zero_cm) or numerical_zero_cm < 0
            or max_polyad_states < 2):
        raise ValueError("invalid resonance selection thresholds")
    n = len(energies)
    anharmonic = _canonical_edges(anharmonic_edges, n, "anharmonic")
    quadratic = _canonical_edges(quadratic_edges, n, "quadratic")
    adjacency = [set() for _ in range(n)]
    selection = []
    for i, j in sorted(anharmonic.keys() | quadratic.keys()):
        anharmonic_value = anharmonic.get((i, j), 0.0)
        quadratic_value = quadratic.get((i, j), 0.0)
        coupling = anharmonic_value + quadratic_value
        if max(abs(anharmonic_value), abs(quadratic_value)) <= numerical_zero_cm:
            continue
        gap = abs(float(energies[i] - energies[j]))
        measures = []
        for value in (anharmonic_value, quadratic_value, coupling):
            if abs(value) <= numerical_zero_cm:
                continue
            span = hypot(gap, 2 * value)
            pair_shift = 2 * value * value / (span + gap)
            measures.append((pair_shift, pair_shift / span))
        shift = max(measure[0] for measure in measures)
        mixing = max(measure[1] for measure in measures)
        accepted = shift >= frequency_tolerance_cm or mixing >= mixing_weight_tolerance
        selection.append(PolyadEdge(i, j, gap, coupling, shift, mixing, accepted))
        if accepted:
            adjacency[i].add(j)
            adjacency[j].add(i)
    components = []
    seen = set()
    for start in range(n):
        if start in seen or not adjacency[start]:
            continue
        stack, component = [start], set()
        while stack:
            state = stack.pop()
            if state in component:
                continue
            component.add(state)
            stack.extend(adjacency[state] - component)
        seen.update(component)
        if len(component) > max_polyad_states:
            raise ValueError("variational polyad exceeds max_polyad_states; inspect its resonance graph")
        components.append(tuple(sorted(component)))
    group = {state: group_id for group_id, component in enumerate(components)
             for state in component}
    reference = energies.copy()
    variational = energies.copy()
    vectors = []
    variational_vectors = []
    assignment = list(range(n))
    assignment_overlap = [1.0] * n
    for component in components:
        positions = {state: pos for pos, state in enumerate(component)}
        block = np.diag(energies[list(component)])
        for (i, j), value in anharmonic.items():
            if i in positions and j in positions:
                block[positions[i], positions[j]] += value
                block[positions[j], positions[i]] += value
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        reference[list(component)] = eigenvalues
        vectors.append(eigenvectors)
        for (i, j), value in quadratic.items():
            if i in positions and j in positions:
                block[positions[i], positions[j]] += value
                block[positions[j], positions[i]] += value
        exact_values, exact_vectors = np.linalg.eigh(block)
        variational[list(component)] = exact_values
        variational_vectors.append(exact_vectors)
        local_assignment = _maximum_overlap_assignment(abs(exact_vectors) ** 2)
        for row, column in enumerate(local_assignment):
            assignment[component[row]] = component[column]
            assignment_overlap[component[row]] = float(abs(exact_vectors[row, column]) ** 2)
    internal = tuple((i, j, value) for (i, j), value in sorted(quadratic.items())
                     if group.get(i, -i-1) == group.get(j, -j-1))
    external = tuple((i, j, value) for (i, j), value in sorted(quadratic.items())
                     if group.get(i, -i-1) != group.get(j, -j-1))
    return LocalQuadraticModel(
        reference_energies_cm=reference,
        polyads=tuple(components), polyad_vectors=tuple(vectors),
        variational_energies_cm=variational,
        variational_vectors=tuple(variational_vectors),
        root_assignment=tuple(assignment),
        assignment_overlap=tuple(assignment_overlap),
        polyad_k_edges=internal, external_k_edges=external,
        selection=tuple(selection),
        denominator_tolerance_cm=denominator_tolerance_cm,
        numerical_zero_cm=numerical_zero_cm,
    )


def transition_series(
    ground: CouplingSeries, excited: CouplingSeries,
    dipole_edges: Sequence[EdgeMap], *, intensity_factor: float = 1.0,
) -> TransitionSeries:
    """Analytic IR transition moment and intensity through intra-polyad K^3.

    This is a coefficient audit of variationally treated K; final
    polyad intensities use the exact eigenvectors.  The dipole surface
    is fixed as K varies.  ``dipole_edges`` gives one
    symmetric sparse operator for each Cartesian component; diagonal keys
    (i,i) are allowed.  ``intensity_factor`` converts wavenumber times
    dipole-squared to the desired intensity unit.
    """
    if not np.isfinite(intensity_factor) or intensity_factor < 0:
        raise ValueError("intensity factor must be finite and nonnegative")
    n = len(ground.wavefunctions[0])
    if any(len(w) != n for series in (ground, excited) for w in series.wavefunctions):
        raise ValueError("transition series have incompatible state bases")
    if not dipole_edges:
        raise ValueError("at least one dipole component is required")
    raw = np.zeros((4, len(dipole_edges)))
    for order in range(4):
        for left in range(order + 1):
            right = order - left
            a, b = excited.wavefunctions[left], ground.wavefunctions[right]
            for axis, component in enumerate(dipole_edges):
                raw[order, axis] += _dipole_element(a, b, component)
    n2 = sum(float(series.wavefunctions[1] @ series.wavefunctions[1])
             for series in (ground, excited))
    n3 = sum(2 * float(series.wavefunctions[1] @ series.wavefunctions[2])
             for series in (ground, excited))
    moments = raw.copy()
    moments[2] -= n2 * raw[0] / 2
    moments[3] -= n2 * raw[1] / 2 + n3 * raw[0] / 2
    frequency = excited.energy_cm - ground.energy_cm
    strength = np.array([sum(float(moments[i] @ moments[k-i])
                             for i in range(k + 1)) for k in range(4)])
    intensity = intensity_factor * np.array([
        sum(frequency[i] * strength[k-i] for i in range(k + 1))
        for k in range(4)
    ])
    return TransitionSeries(frequency, moments, intensity)
