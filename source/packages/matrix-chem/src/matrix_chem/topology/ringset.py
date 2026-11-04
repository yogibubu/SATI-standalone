# ============================================================
# RingSet class for MATRIX
#
# Responsibilities:
#   - detect elementary covalent rings in the molecular graph
#   - optionally limit ring size (ring_max)
#   - build Ring objects
#   - manage atom-ring and bond-ring mappings
#   - detect fused / connected rings
#
# Notes:
#   - rings are chordless cycles in the non-metal covalent graph
#   - metal-ring interactions are represented by special centers, not by
#     artificial metal-containing topology rings
# ============================================================

from collections import defaultdict

from .cycle_basis import CycleBasisDiagnostics, elementary_cycle_basis
from .metals import METAL_ATOMIC_NUMBERS, is_metal_atomic_number

try:
    from .rings import Ring
except ImportError:
    from rings import Ring


class RingSet:
    """
    Container and manager for all rings in a molecular graph.
    """

    def __init__(self, graph, coords=None, ring_max=None):
        """
        Parameters
        ----------
        graph : object
            Molecular graph object.
            Must provide:
              - graph.natoms or graph.n_atoms
              - graph.neighbors(atom)
        coords : ndarray, optional
            Cartesian coordinates of the system.
        ring_max : int, optional
            Maximum ring size to consider.
            If None, no size limit is applied.
        """
        self.graph = graph
        self.coords = coords
        self.ring_max = ring_max

        # --- storage ---
        self.rings = []
        self.atom_to_rings = defaultdict(list)
        self.bond_to_rings = defaultdict(list)
        self.cycle_basis_diagnostics = CycleBasisDiagnostics(0, 0, 0, 0, 0, ())

        # --- build ---
        self._detect_rings()
        self._build_connectivity()

    # ============================================================
    # Ring detection
    # ============================================================

    def _detect_rings(self):
        """
        Detect elementary rings.

        MATRIX stores rings as chordless cycles of the covalent, non-metal
        graph.  This removes perimeter cycles in fused systems, diagonal cycles
        in cage systems and artificial metal-containing cycles in metallocenes.
        """
        nat = getattr(self.graph, "natoms", getattr(self.graph, "n_atoms", None))
        if nat is None:
            raise AttributeError("Graph must define natoms or n_atoms")

        atomic_numbers = getattr(self.graph, "Z", ())
        allowed_atoms = (
            {
                atom
                for atom in range(nat)
                if int(atomic_numbers[atom]) not in METAL_ATOMIC_NUMBERS
            }
            if len(atomic_numbers) >= nat
            else set(range(nat))
        )
        selected_cycles, self.cycle_basis_diagnostics = elementary_cycle_basis(
            self.graph,
            allowed_atoms=allowed_atoms,
            ring_max=self.ring_max,
        )
        for ring_index, atoms in enumerate(selected_cycles):
            ring = Ring(
                index=ring_index,
                atoms=list(atoms),
                coords=self.coords,
            )
            self.rings.append(ring)

            for atom in atoms:
                self.atom_to_rings[atom].append(ring_index)

            for bond in ring.bonds:
                self.bond_to_rings[bond].append(ring_index)

    def _is_metal_atom(self, atom):
        Z = int(getattr(self.graph, "Z", [0] * (atom + 1))[atom])
        return is_metal_atomic_number(Z)

    # ============================================================
    # Ring connectivity
    # ============================================================

    def _build_connectivity(self):
        """
        Detect fused / connected rings.
        """
        connected_pairs = set()
        for ring_indices in self.atom_to_rings.values():
            ordered = sorted(set(ring_indices))
            for position, first in enumerate(ordered):
                for second in ordered[position + 1 :]:
                    connected_pairs.add((first, second))
        for first, second in sorted(connected_pairs):
            self.rings[first].add_connected_ring(second)
            self.rings[second].add_connected_ring(first)

    # ============================================================
    # Public API
    # ============================================================

    def __len__(self):
        return len(self.rings)

    def __iter__(self):
        return iter(self.rings)

    def get_ring(self, index):
        return self.rings[index]

    def rings_of_atom(self, atom):
        return self.atom_to_rings.get(atom, [])

    def rings_of_bond(self, bond):
        return self.bond_to_rings.get(tuple(sorted(bond)), [])

    def fused_rings(self, index):
        """
        Return indices of rings fused to the given ring.
        """
        return list(self.rings[index].connected_rings)

    def __repr__(self):
        lim = self.ring_max if self.ring_max is not None else "∞"
        return f"<RingSet: {len(self.rings)} rings (max={lim})>"
