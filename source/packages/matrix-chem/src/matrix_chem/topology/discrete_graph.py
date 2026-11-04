"""
discrete_graph.py
=================

Construction of a discrete molecular graph from a ContinuousGraph.

Design principles:
- geometry-first
- no explicit valence rules
- no electronic assumptions
- hydrogens have one bond except for geometrically certified shared bridges
"""

import numpy as np

from .continuous_graph import DISCRETE_DISTANCE_SCALE
from .continuous_graph import pauling_bond_order
from .bonding_roles import is_structural_center
from .covalent_radii import covalent_radius

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

BOND_THRESHOLD = 0.2  # minimum continuous connectivity weight
REFF_SCALE = DISCRETE_DISTANCE_SCALE  # distance sanity factor
TRANSITIONAL_BRIDGE_MAXIMUM_PAULING_ORDER = 0.55


class DiscreteGraph:
    """
    Discrete molecular graph derived from a ContinuousGraph.
    """

    def __init__(self, graph):
        self.graph = graph
        self.Z = graph.Z
        self.coords = graph.coords
        self.natoms = len(self.Z)

        self.adjacency = [set() for _ in range(self.natoms)]
        self.bonds = []
        self.bonds_are_canonical_sorted = True
        self.hydrogen_bridges = ()
        self.transitional_contacts = ()
        self.near_covalent_contacts = ()

        self._build()
        self._validate_hydrogens()
        native_cycles = getattr(graph, "native_cycle_basis", None)
        if native_cycles is not None:
            self.native_cycle_basis = native_cycles
            self.native_cycle_candidate_count = graph.native_cycle_candidate_count
            self.native_cycle_rank = graph.native_cycle_rank
            self.native_cycle_bonds = tuple(self.bonds)

    # --------------------------------------------------------
    # Graph construction
    # --------------------------------------------------------

    def _build(self):
        native_left = getattr(self.graph, "native_accepted_left", None)
        native_right = getattr(self.graph, "native_accepted_right", None)
        preliminary_left = getattr(
            self.graph,
            "discrete_candidate_left",
            None,
        )
        preliminary_right = getattr(
            self.graph,
            "discrete_candidate_right",
            None,
        )
        preliminary_connectivity = getattr(
            self.graph,
            "discrete_candidate_connectivity",
            None,
        )
        if native_left is not None and native_right is not None:
            accepted = zip(native_left, native_right, strict=True)
        elif (
            preliminary_left is not None
            and preliminary_right is not None
            and preliminary_connectivity is not None
        ):
            accepted_mask = preliminary_connectivity >= BOND_THRESHOLD
            accepted_left = preliminary_left[accepted_mask]
            accepted_right = preliminary_right[accepted_mask]
            accepted = zip(accepted_left, accepted_right, strict=True)
        elif (
            (
                preliminary_pairs := getattr(
                    self.graph,
                    "discrete_candidate_pairs",
                    None,
                )
            )
            is not None
            and preliminary_connectivity is not None
        ):
            accepted = (
                pair
                for pair, connectivity in zip(
                    preliminary_pairs,
                    preliminary_connectivity,
                    strict=True,
                )
                if connectivity >= BOND_THRESHOLD
            )
        else:
            candidate_pairs = tuple(
                getattr(
                    self.graph,
                    "candidate_pairs",
                    tuple(
                        (i, j)
                        for i in range(self.natoms)
                        for j in range(i + 1, self.natoms)
                    ),
                )
            )
            accepted = (
                (i, j)
                for i, j in candidate_pairs
                if self._candidate_bond(i, j)
            )
        accepted = tuple((int(i), int(j)) for i, j in accepted)
        from ..structural_corrections import perceive_proton_transfer_bridges

        self.hydrogen_bridges = perceive_proton_transfer_bridges(
            self.Z,
            self.coords,
            accepted,
        )
        bridge_pairs = {
            tuple(sorted((hydrogen, heavy)))
            for hydrogen, left, right in self.hydrogen_bridges
            for heavy in (left, right)
        }
        bridge_hydrogens = {
            int(hydrogen) for hydrogen, _left, _right in self.hydrogen_bridges
        }
        accepted = tuple(
            sorted(
                {
                    tuple(sorted((int(left), int(right))))
                    for left, right in accepted
                }
                | bridge_pairs
            )
        )
        # An ordinary hydrogen has one constitutional partner.  In compact or
        # temporarily distorted geometries the distance/connectivity
        # shortlist can contain two heavy-atom candidates (for example after
        # rigid transport of an aromatic fragment); retain only the closest.
        # The sole exception is a geometrically certified shared-hydrogen
        # bridge, for which both independently detected H--X edges are kept.
        hydrogen_heavy: dict[int, tuple[float, int, int]] = {}
        filtered: list[tuple[int, int]] = []
        for i, j in accepted:
            Zi = int(self.Z[i])
            Zj = int(self.Z[j])
            if (Zi == 1) ^ (Zj == 1):
                hydrogen = i if Zi == 1 else j
                heavy = j if Zi == 1 else i
                if tuple(sorted((hydrogen, heavy))) in bridge_pairs:
                    filtered.append((i, j))
                    continue
                if hydrogen in bridge_hydrogens:
                    # A certified two-centre bridge owns the complete
                    # constitutional neighbourhood of its hydrogen.  Nearby
                    # structural centres remain continuous TS evidence; they
                    # cannot create a third discrete H bond.
                    continue
                distance = float(np.linalg.norm(self.coords[hydrogen] - self.coords[heavy]))
                candidate = (distance, heavy, hydrogen)
                previous = hydrogen_heavy.get(hydrogen)
                if previous is None or candidate < previous:
                    hydrogen_heavy[hydrogen] = candidate
            else:
                filtered.append((i, j))
        selected_hydrogen_heavy = {
            (heavy, hydrogen) for _distance, heavy, hydrogen in hydrogen_heavy.values()
        } | bridge_pairs
        filtered = [
            (i, j)
            for i, j in accepted
            if not ((int(self.Z[i]) == 1) ^ (int(self.Z[j]) == 1))
            or (i, j) in selected_hydrogen_heavy
            or (j, i) in selected_hydrogen_heavy
        ]
        accepted = tuple(filtered)
        accepted, self.transitional_contacts = _separate_transitional_contacts(
            accepted,
            atomic_numbers=self.Z,
            coordinates=self.coords,
            protected_pairs=bridge_pairs,
            bond_order_overrides=getattr(self.graph, "connectivity_overrides", None),
        )
        self.near_covalent_contacts = _near_covalent_contacts(
            self.graph,
            accepted,
            self.transitional_contacts,
        )
        heavy_partner = [False] * self.natoms
        for i, j in accepted:
            if int(self.Z[j]) != 1:
                heavy_partner[i] = True
            if int(self.Z[i]) != 1:
                heavy_partner[j] = True
        for i, j in accepted:
            Zi = int(self.Z[i])
            Zj = int(self.Z[j])
            if Zi == 1 and Zj == 1 and (heavy_partner[i] or heavy_partner[j]):
                continue
            self.adjacency[i].add(j)
            self.adjacency[j].add(i)
            self.bonds.append((i, j))

    def _candidate_bond(self, i, j):
        ri = covalent_radius(int(self.Z[i]))
        rj = covalent_radius(int(self.Z[j]))
        if (
            ri is None
            or rj is None
            or self.graph.CONNECTIVITY[i, j] < BOND_THRESHOLD
        ):
            return False
        distances = getattr(self.graph, "pair_distances", None)
        distance = (
            distances[(i, j)]
            if distances is not None and (i, j) in distances
            else np.linalg.norm(self.coords[i] - self.coords[j])
        )
        return distance <= REFF_SCALE * (ri + rj)

    def _add_bond(self, i, j):
        self.adjacency[i].add(j)
        self.adjacency[j].add(i)
        self.bonds.append((i, j))

    def _has_heavy_partner(self, i):
        ri = covalent_radius(int(self.Z[i]))
        if ri is None:
            return False
        for j in range(self.natoms):
            if i == j or int(self.Z[j]) == 1:
                continue
            rj = covalent_radius(int(self.Z[j]))
            if rj is None:
                continue
            if self.connectivity[i, j] < BOND_THRESHOLD:
                continue
            rij = np.linalg.norm(self.coords[i] - self.coords[j])
            if rij <= REFF_SCALE * (ri + rj):
                return True
        return False

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_hydrogens(self):
        bridge_neighbors: dict[int, set[int]] = {}
        for hydrogen, left, right in self.hydrogen_bridges:
            bridge_neighbors[int(hydrogen)] = {int(left), int(right)}
        for i, Zi in enumerate(self.Z):
            if Zi != 1 or len(self.adjacency[i]) in {0, 1}:
                continue
            if len(self.adjacency[i]) == 2 and self.adjacency[i] == bridge_neighbors.get(i):
                continue
            raise ValueError(f"Hydrogen atom {i + 1} has {len(self.adjacency[i])} bonds")

    # --------------------------------------------------------
    # Public helpers
    # --------------------------------------------------------

    def neighbors(self, i):
        return self.adjacency[i]

    @property
    def connectivity(self):
        return self.graph.CONNECTIVITY


def _separate_transitional_contacts(
    bonds: tuple[tuple[int, int], ...],
    *,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    protected_pairs: set[tuple[int, int]],
    bond_order_overrides=None,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Demote only weak acyclic links from the constitutional graph.

    A low radial order is not sufficient by itself: ring edges remain
    constitutional, as do all accepted bonds to structural centers and the
    two arms of a certified multicenter bridge.  The remaining weak graph
    bridges are transition contacts; removing them materializes the fragments
    needed by the existing exact interfragment SONIC chart.
    """

    canonical = tuple(sorted({tuple(sorted(map(int, pair))) for pair in bonds}))
    graph_bridges = _graph_bridge_edges(len(atomic_numbers), canonical)
    retained = []
    transitional = []
    for left, right in canonical:
        pair = (left, right)
        if (
            pair not in graph_bridges
            or pair in protected_pairs
            or is_structural_center(int(atomic_numbers[left]))
            or is_structural_center(int(atomic_numbers[right]))
        ):
            retained.append(pair)
            continue
        order = float((bond_order_overrides or {}).get(pair, np.nan))
        if not np.isfinite(order):
            order = pauling_bond_order(
                left,
                right,
                atomic_numbers,
                coordinates,
            )
        if order <= TRANSITIONAL_BRIDGE_MAXIMUM_PAULING_ORDER:
            transitional.append(pair)
        else:
            retained.append(pair)
    return tuple(retained), tuple(transitional)


def _near_covalent_contacts(
    graph,
    bonds: tuple[tuple[int, int], ...],
    transitional_contacts: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Retain plausible forming bonds from the existing continuous pool.

    The continuous graph has already applied MATRIX's element-dependent
    covalent-radius screen.  Pairs rejected by the discrete graph are useful
    evidence for a transition state, but they must not be promoted to the
    constitutional graph.  Ordinary 1,3 pairs are excluded because their
    distance is already controlled by a valence angle.  SMITH consumes this
    frozen evidence only when a transition-state chart is requested
    explicitly.
    """

    retained = {tuple(sorted(map(int, pair))) for pair in bonds}
    transitional = {
        tuple(sorted(map(int, pair))) for pair in transitional_contacts
    }
    adjacency = [set() for _ in range(int(graph.natoms))]
    for left, right in retained:
        adjacency[left].add(right)
        adjacency[right].add(left)

    def separated_by_more_than_two_edges(left: int, right: int) -> bool:
        if right in adjacency[left]:
            return False
        return not any(right in adjacency[neighbor] for neighbor in adjacency[left])

    candidates = []
    for pair, connectivity in zip(
        graph.discrete_candidate_pairs,
        graph.discrete_candidate_connectivity,
        strict=True,
    ):
        canonical = tuple(sorted((int(pair[0]), int(pair[1]))))
        if canonical in retained or canonical in transitional:
            continue
        if float(connectivity) <= 0.0:
            continue
        if not separated_by_more_than_two_edges(*canonical):
            continue
        candidates.append(canonical)
    return tuple(sorted(set(candidates)))


def _graph_bridge_edges(
    natoms: int,
    bonds: tuple[tuple[int, int], ...],
) -> set[tuple[int, int]]:
    """Return graph bridges in linear time with deterministic traversal."""

    adjacency = [set() for _ in range(int(natoms))]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    discovery = [-1] * int(natoms)
    low = [-1] * int(natoms)
    bridges: set[tuple[int, int]] = set()
    counter = 0

    ordered_neighbors = [tuple(sorted(neighbors)) for neighbors in adjacency]
    for atom in range(int(natoms)):
        if discovery[atom] >= 0:
            continue
        discovery[atom] = low[atom] = counter
        counter += 1
        stack: list[tuple[int, int, int]] = [(atom, -1, 0)]
        while stack:
            current, parent, neighbor_index = stack[-1]
            neighbors = ordered_neighbors[current]
            if neighbor_index < len(neighbors):
                neighbor = neighbors[neighbor_index]
                stack[-1] = (current, parent, neighbor_index + 1)
                if neighbor == parent:
                    continue
                if discovery[neighbor] < 0:
                    discovery[neighbor] = low[neighbor] = counter
                    counter += 1
                    stack.append((neighbor, current, 0))
                else:
                    low[current] = min(low[current], discovery[neighbor])
                continue
            stack.pop()
            if parent >= 0:
                low[parent] = min(low[parent], low[current])
                if low[current] > discovery[parent]:
                    bridges.add(tuple(sorted((parent, current))))
    return bridges
