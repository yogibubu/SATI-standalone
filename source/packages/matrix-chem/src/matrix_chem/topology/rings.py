# rings.py
# ============================================================
# Unified Ring class for ORACLE
#
# Represents a single topological ring with optional
# geometric information.
#
# Responsibilities:
#   - store ordered cyclic atoms
#   - store bonds belonging to the ring
#   - know ring–ring connectivity
#   - provide geometric descriptors if coordinates are available
#
# It does NOT:
#   - decide aromaticity
#   - build GNICs
#   - perform pruning
# ============================================================

import numpy as np

try:
    from .functions_ring import (
        best_fit_plane,
        ring_planarity,
        cyclic_triplets,
        cyclic_quartets,
    )
except ImportError:
    # functions_ring is optional at runtime
    best_fit_plane = None
    ring_planarity = None
    cyclic_triplets = None
    cyclic_quartets = None


class Ring:
    """
    Topological ring with optional geometric information.

    Parameters
    ----------
    index : int
        Ring identifier.
    atoms : list[int]
        Ordered list of atom indices defining the ring.
    bonds : list[tuple[int,int]], optional
        List of bonds belonging to the ring.
    coords : array-like (N_atoms, 3), optional
        Cartesian coordinates of the full system.
    """

    def __init__(self, index, atoms, bonds=None, coords=None):
        self.index = index

        # --- Topology ---
        self.atoms = list(atoms)  # ordered cyclic atoms
        self.n_atoms = len(self.atoms)

        self.bonds = bonds if bonds is not None else self._build_bonds()
        self.connected_rings = set()  # indices of fused/adjacent rings

        # --- Geometry (optional) ---
        self.coords = coords
        self._plane = None
        self._normal = None
        self._planarity = None

        # --- Cached cyclic primitives ---
        self._triplets = None
        self._quartets = None

    # ============================================================
    # Topological utilities
    # ============================================================

    def _build_bonds(self):
        """Build ring bonds from cyclic atom ordering."""
        bonds = []
        for i in range(self.n_atoms):
            a = self.atoms[i]
            b = self.atoms[(i + 1) % self.n_atoms]
            bonds.append(tuple(sorted((a, b))))
        return bonds

    def add_connected_ring(self, other_ring_index):
        """Register connectivity to another ring."""
        self.connected_rings.add(other_ring_index)

    def shares_atoms_with(self, other):
        """Return True if two rings share at least one atom."""
        return bool(set(self.atoms) & set(other.atoms))

    def shares_bonds_with(self, other):
        """Return True if two rings share at least one bond."""
        return bool(set(self.bonds) & set(other.bonds))

    # ============================================================
    # Cyclic topology helpers
    # ============================================================

    def cyclic_atom_triplets(self):
        """
        Return cyclic triplets (i,j,k) along the ring.
        """
        if self._triplets is None:
            if cyclic_triplets is None:
                self._triplets = []
            else:
                self._triplets = cyclic_triplets(self.atoms)
        return self._triplets

    def cyclic_atom_quartets(self):
        """
        Return cyclic quartets (i,j,k,l) along the ring.
        """
        if self._quartets is None:
            if cyclic_quartets is None:
                self._quartets = []
            else:
                self._quartets = cyclic_quartets(self.atoms)
        return self._quartets

    # ============================================================
    # Geometry (optional, lazy evaluation)
    # ============================================================

    def has_geometry(self):
        return self.coords is not None

    def _xyz_ring(self):
        """
        Return ring coordinates as a NumPy array (N_ring, 3),
        regardless of input type of self.coords.
        """
        if not self.has_geometry():
            return None

        xyz = np.asarray(self.coords, dtype=float)
        return xyz[self.atoms]

    def plane(self):
        """
        Best-fit plane of the ring atoms.
        Returns (point, normal) or None.
        """
        if not self.has_geometry() or best_fit_plane is None:
            return None

        if self._plane is None or self._normal is None:
            xyz_ring = self._xyz_ring()
            self._plane, self._normal = best_fit_plane(xyz_ring)

        return self._plane, self._normal

    def normal(self):
        """Return normal vector to the ring plane, if available."""
        if self.plane() is None:
            return None
        return self._normal

    def planarity(self):
        """
        Quantify ring planarity (RMS distance from best-fit plane).
        """
        if not self.has_geometry() or ring_planarity is None:
            return None

        if self._planarity is None:
            xyz_ring = self._xyz_ring()
            self._planarity = ring_planarity(xyz_ring)

        return self._planarity

    def is_planar(self, threshold=0.1):
        """
        Return True if ring is planar within a given threshold.
        """
        p = self.planarity()
        if p is None:
            return False
        return p < threshold

    # ============================================================
    # Representation
    # ============================================================

    def __len__(self):
        return self.n_atoms

    def __repr__(self):
        return f"<Ring {self.index}: size={self.n_atoms}, atoms={self.atoms}>"
