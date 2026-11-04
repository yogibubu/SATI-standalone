import numpy as np


def _vector_angle(left, right):
    left_vec = np.asarray(left, dtype=float)
    right_vec = np.asarray(right, dtype=float)
    denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
    if denom <= 1.0e-14:
        return 0.0
    cosine = np.dot(left_vec, right_vec) / denom
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _dihedral_angle(p1, p2, p3, p4):
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    c = np.asarray(p3, dtype=float)
    d = np.asarray(p4, dtype=float)
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    b2_norm = np.linalg.norm(b2)
    if min(n1_norm, n2_norm, b2_norm) <= 1.0e-14:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    m1 = np.cross(n1, b2 / b2_norm)
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


class Ring:
    """
    Topological + geometrical ring object.

    Ring coordinates are intended to be built from
    internal angles (valence and dihedral angles),
    not from Cartesian puckering coordinates.
    """

    def __init__(self, index, atoms, coords):
        self.index = index
        self.atoms = list(atoms)  # ordered cyclic list
        self.coords = np.asarray(coords)  # (N,3)

        # Topology
        self.bonds = self._build_bonds()
        self.adjacent_rings_atoms = set()
        self.adjacent_rings_bonds = set()

    # ----------------------------------------------------------
    # Cyclic topology helpers
    # ----------------------------------------------------------

    def cyclic_triplets(self):
        """
        Triplets (i-1, i, i+1) for valence angles.
        """
        n = len(self.atoms)
        return [(self.atoms[(i - 1) % n], self.atoms[i], self.atoms[(i + 1) % n]) for i in range(n)]

    def cyclic_quartets(self):
        """
        Quartets (i-1, i, i+1, i+2) for dihedral angles.
        """
        n = len(self.atoms)
        return [
            (
                self.atoms[(i - 1) % n],
                self.atoms[i],
                self.atoms[(i + 1) % n],
                self.atoms[(i + 2) % n],
            )
            for i in range(n)
        ]

    # ----------------------------------------------------------
    # Numerical values (optional, but useful)
    # ----------------------------------------------------------

    def valence_angles(self):
        """
        Return list of valence angles (radians) along the ring.
        """
        angles = []
        for i, j, k in self.cyclic_triplets():
            angles.append(self._angle(i, j, k))
        return angles

    def dihedral_angles(self):
        """
        Return list of dihedral angles (radians) along the ring.
        """
        dihedrals = []
        for i, j, k, l in self.cyclic_quartets():
            dihedrals.append(self._dihedral(i, j, k, l))
        return dihedrals

    # ----------------------------------------------------------
    # Geometry primitives
    # ----------------------------------------------------------

    def _angle(self, i, j, k):
        rji = self.coords[self.atoms.index(i)] - self.coords[self.atoms.index(j)]
        rjk = self.coords[self.atoms.index(k)] - self.coords[self.atoms.index(j)]
        return _vector_angle(rji, rjk)

    def _dihedral(self, i, j, k, l):
        p = [self.coords[self.atoms.index(x)] for x in (i, j, k, l)]
        return _dihedral_angle(*p)
