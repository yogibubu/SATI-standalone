"""Geometry-first aromatic-ring perception and planarity preservation."""

from dataclasses import dataclass
import numpy as np

from .metals import is_metal_atomic_number
from .contracts import MATRIX_XYZ_AROMATICITY_SCHEMA


@dataclass(frozen=True)
class AromaticRingAssignment:
    ring_index: int
    atoms: tuple[int, ...]
    electron_count: int | None
    criterion: str
    maximum_plane_deviation_angstrom: float
    rms_plane_deviation_angstrom: float


class Aromaticity:
    """
    Aromaticity detector.
    """

    def __init__(
        self,
        graph,
        discrete_graph,
        ring_set,
        synthons=None,
        force_aromatic=False,
        planarity_tol=0.1,
    ):
        self.graph = graph
        self.dgraph = discrete_graph
        self.ringset = ring_set
        self.synthons = synthons
        self.force_aromatic = force_aromatic
        self.planarity_tol = planarity_tol

        self.Z = graph.Z
        self.coords = graph.coords

        self.aromatic_atoms = set()
        self.aromatic_bonds = set()
        self.aromatic_rings = set()
        self.assignments = []

        self._analyze()

    # --------------------------------------------------------
    # Core logic
    # --------------------------------------------------------

    def _analyze(self):
        for ring in self.ringset:
            plane = self._planarity(ring)
            if plane[0] >= self.planarity_tol:
                continue

            decision = self._ring_aromaticity_decision(ring)
            if decision is None:
                continue
            electron_count, criterion = decision

            self.aromatic_rings.add(ring.index)
            for i in ring.atoms:
                self.aromatic_atoms.add(i)

            for i, j in ring.bonds:
                self.aromatic_bonds.add(tuple(sorted((i, j))))
            self.assignments.append(
                AromaticRingAssignment(
                    ring_index=int(ring.index),
                    atoms=tuple(int(atom) for atom in ring.atoms),
                    electron_count=electron_count,
                    criterion=criterion,
                    maximum_plane_deviation_angstrom=plane[0],
                    rms_plane_deviation_angstrom=plane[1],
                )
            )

    # --------------------------------------------------------
    # Planarity
    # --------------------------------------------------------

    def _planarity(self, ring):
        coords = np.array([self.coords[i] for i in ring.atoms])
        centroid = coords.mean(axis=0)
        coords -= centroid

        _, eigenvectors = np.linalg.eigh(coords.T @ coords)
        normal = eigenvectors[:, 0]

        distances = np.abs(coords @ normal)
        return float(distances.max()), float(np.sqrt(np.mean(distances * distances)))

    # --------------------------------------------------------
    # Aromaticity decision
    # --------------------------------------------------------

    def _ring_aromaticity_decision(self, ring):
        if self.force_aromatic:
            return None, "FORCED"

        # Cyclopropenones and their heavier group-16 analogues have an
        # important charge-separated form: the exocyclic C=X bond is polarized
        # toward X and the three-membered carbon ring becomes a 2-pi-electron
        # cyclopropenyl system.  Count the exocyclicly substituted ring carbon
        # as an empty-p contributor rather than assigning one electron to all
        # three carbons.
        if len(ring) == 3 and all(int(self.Z[atom]) in (6, 14, 32) for atom in ring.atoms):
            ring_atoms = set(ring.atoms)
            polarized_centers = [
                atom
                for atom in ring.atoms
                if any(
                    neighbor not in ring_atoms
                    and int(self.Z[neighbor]) in (8, 16, 34, 52)
                    for neighbor in self.dgraph.neighbors(atom)
                )
            ]
            if len(polarized_centers) == 1:
                return 2, "EXOCYCLIC_GROUP16_POLARIZED_2PI"

        contributions = []
        for atom in ring.atoms:
            contribution = self._pi_electron_contribution(atom, len(ring))
            if contribution is None:
                return None
            contributions.append(contribution)

        electron_count = int(sum(contributions))
        fused = bool(ring.connected_rings)
        coordinated = any(
            is_metal_atomic_number(int(self.Z[neighbor]))
            for atom in ring.atoms
            for neighbor in self.dgraph.neighbors(atom)
        )
        if electron_count >= 2 and electron_count % 4 == 2:
            return electron_count, "HUCKEL_4N_PLUS_2"
        # A minimum-cycle basis can partition a globally aromatic fused
        # hydrocarbon into rings that do not individually carry 4n+2 electrons
        # (azulene is the canonical example).  Apply that fallback only when
        # the complete connected fused component is an all-group-14,
        # 4n+2-electron system.  Merely being fused is insufficient:
        # acenaphthylene retains two aromatic six-membered rings but its
        # five-membered ring has only weak/paratropic local aromaticity.
        if fused:
            component_atoms = self._fused_component_atoms(ring)
            if (
                all(int(self.Z[atom]) in (6, 14, 32) for atom in component_atoms)
                and all(
                    self._pi_electron_contribution(atom, len(component_atoms)) == 1
                    for atom in component_atoms
                )
            ):
                component_electrons = len(component_atoms)
                if component_electrons >= 2 and component_electrons % 4 == 2:
                    return component_electrons, "FUSED_COMPONENT_4N_PLUS_2"
        # Five-membered carbon rings bound to a metal represent an eta5
        # aromatic ligand; metal contacts are excluded from the sigma degree.
        if coordinated and len(ring) == 5 and electron_count == 5:
            return 6, "METAL_COORDINATED_ETA5"
        return None

    def _fused_component_atoms(self, seed_ring):
        rings = {int(ring.index): ring for ring in self.ringset}
        pending = [int(seed_ring.index)]
        visited = set()
        atoms = set()
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            ring = rings[index]
            atoms.update(int(atom) for atom in ring.atoms)
            pending.extend(int(value) for value in ring.connected_rings - visited)
        return tuple(sorted(atoms))

    def _pi_electron_contribution(self, i, ring_size):
        """Return the local p-electron contribution, or ``None`` if ineligible."""
        Z = int(self.Z[i])
        deg = sum(
            not is_metal_atomic_number(int(self.Z[neighbor]))
            for neighbor in self.dgraph.neighbors(i)
        )
        if Z in (5, 13, 31):  # B, Al, Ga: empty p orbital
            return 0 if deg == 3 else None
        if Z in (6, 14, 32):  # C, Si, Ge
            return 1 if deg == 3 else None
        if Z in (7, 15, 33):  # N, P, As
            if deg == 2:
                return 1
            if deg == 3:
                # A three-coordinate pnictogen in a five-membered ring is
                # pyrrole-like.  In six-membered rings it is commonly a
                # protonated pyridine-like center and contributes one electron.
                # A pnictogen adjacent to an electron-deficient group-13
                # center is instead the donor site in borazine-like rings.
                group13_neighbor = any(
                    int(self.Z[neighbor]) in (5, 13, 31)
                    for neighbor in self.dgraph.neighbors(i)
                )
                return 2 if ring_size == 5 or group13_neighbor else 1
            return None
        if Z in (8, 16, 34, 52):  # O, S, Se, Te
            return 2 if deg == 2 else None
        return None


def project_aromatic_ring_planarity(
    coordinates_angstrom,
    aromaticity,
    *,
    maximum_iterations=25,
    tolerance_angstrom=1.0e-10,
):
    """Project every persistently assigned aromatic ring onto a local plane.

    Each ring supplies one local best-fit plane per iteration.  Shared atoms
    receive the average of the proposals, so fused and helically arranged
    systems retain their local ring planes rather than being flattened onto
    one global plane.
    """
    current = np.asarray(coordinates_angstrom, dtype=float).copy()
    rings = [
        ring
        for ring in aromaticity.ringset
        if int(ring.index) in set(aromaticity.aromatic_rings)
    ]
    if not rings:
        return current
    for _iteration in range(max(1, int(maximum_iterations))):
        proposals = [[] for _atom in range(len(current))]
        for ring in rings:
            indices = np.asarray(ring.atoms, dtype=int)
            points = current[indices]
            centroid = points.mean(axis=0)
            centered = points - centroid
            _values, vectors = np.linalg.eigh(centered.T @ centered)
            normal = vectors[:, 0]
            projected = points - np.outer(centered @ normal, normal)
            for atom, point in zip(indices, projected, strict=True):
                proposals[int(atom)].append(point)
        updated = current.copy()
        for atom, rows in enumerate(proposals):
            if rows:
                updated[atom] = np.mean(rows, axis=0)
        displacement = (
            float(np.max(np.linalg.norm(updated - current, axis=1))) if len(current) else 0.0
        )
        current = updated
        if displacement <= float(tolerance_angstrom):
            break
    return current


def aromaticity_section_lines(aromaticity):
    """Serialize the persistent aromatic assignment without changing #TOPOLOGY."""
    atoms = " ".join(str(atom + 1) for atom in sorted(aromaticity.aromatic_atoms))
    bonds = " ".join(
        f"{left + 1}-{right + 1}" for left, right in sorted(aromaticity.aromatic_bonds)
    )
    lines = [
        f"SCHEMA {MATRIX_XYZ_AROMATICITY_SCHEMA}",
        "OWNER ORACLE",
        "INDEXING ATOMS=ONE_BASED",
        "MODEL GEOMETRY_FIRST_HUCKEL_FUSED_V2",
        f"PLANARITY_TOLERANCE_ANGSTROM {float(aromaticity.planarity_tol):.10g}",
        "PLANARITY_PROJECTOR LOCAL_RING_PLANES_SHARED_ATOM_AVERAGE",
        f"ATOMS {atoms or 'NONE'}",
        f"BONDS {bonds or 'NONE'}",
    ]
    for assignment in aromaticity.assignments:
        ring_atoms = ",".join(str(atom + 1) for atom in assignment.atoms)
        electrons = (
            str(assignment.electron_count)
            if assignment.electron_count is not None
            else "UNKNOWN"
        )
        lines.append(
            f"RING {assignment.ring_index + 1} ATOMS={ring_atoms} "
            f"ELECTRONS={electrons} CRITERION={assignment.criterion} "
            f"MAX_PLANE_DEVIATION={assignment.maximum_plane_deviation_angstrom:.10g} "
            f"RMS_PLANE_DEVIATION={assignment.rms_plane_deviation_angstrom:.10g}"
        )
    return lines
