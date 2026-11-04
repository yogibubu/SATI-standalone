"""ORACLE-owned selective out-of-plane displacement records.

These are frozen Cartesian displacement primitives, not Gaussian U/H scalars.
Indices are one based.  Geometry and directions share one fixed reference frame.
"""

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import numpy as np


CYCLIC_PARTITION_MANIFEST_SHA256 = (
    "a28a6ce8986df49b1b11b428b6a285efbe4fff0174c062476ee4c9036c8ecaf7"
)


def frozen_cyclic_partition_manifest():
    """Load the audited cycle rules; fail closed if the frozen file changes."""
    payload = files("matrix_chem").joinpath(
        "data/cyclic_primitive_partition_v1.json"
    ).read_bytes()
    if hashlib.sha256(payload).hexdigest() != CYCLIC_PARTITION_MANIFEST_SHA256:
        raise ValueError("frozen cyclic primitive partition manifest has changed")
    manifest = json.loads(payload)
    if manifest.get("schema") != "matrix.cyclic_primitive_partition.v1":
        raise ValueError("unsupported cyclic primitive partition manifest")
    return manifest


def frozen_soop_covector(primitive, natoms):
    """Decode a frozen Cartesian SOOP or projected local SYCART covector."""
    if primitive.function not in {"SOOP", "SYCART"} or primitive.chart != "FROZEN_REFERENCE_CARTESIAN":
        raise ValueError("invalid frozen Cartesian coordinate contract")
    row = np.zeros(3*natoms)
    offset = None
    seen = set()
    for ref in primitive.refs:
        fields = ref.split(":")
        if len(fields) == 2 and fields[0] == "OFFSET" and offset is None:
            offset = float(fields[1])
        elif len(fields) == 3 and fields[0] == "B":
            column, value = int(fields[1]), float(fields[2])
            if column in seen or not 0 <= column < row.size:
                raise ValueError("invalid SOOP covector support")
            seen.add(column)
            row[column] = value
        else:
            raise ValueError("invalid SOOP covector serialization")
    if offset is None or not np.isfinite(offset) or not np.isfinite(row).all() or not np.any(row):
        raise ValueError("invalid SOOP covector values")
    return row, offset


@dataclass(frozen=True)
class SelectiveOutOfPlanePrimitive:
    identifier: str
    center_atom: int
    moving_atom: int
    plane_atoms: tuple[int, int, int]
    endocyclic_ring_ids: tuple[int, ...]
    reference_position_angstrom: tuple[float, float, float]
    normal: tuple[float, ...]
    moving_atom_ring_ids: tuple[int, ...] = ()
    status: str = "REGULAR"
    frame: str = "FROZEN_REFERENCE_CARTESIAN"

    @property
    def endocyclic(self):
        return bool(self.endocyclic_ring_ids)

    def displacement(self, natoms: int, step_angstrom: float = 1.0):
        """Only the designated atom moves; no implicit centering or rotation."""
        if self.status != "REGULAR":
            raise ValueError(f"{self.identifier}: singular SOOP reference plane")
        if not 1 <= self.moving_atom <= natoms or not np.isfinite(step_angstrom):
            raise ValueError("invalid SOOP displacement request")
        result = np.zeros((natoms, 3))
        result[self.moving_atom - 1] = step_angstrom * np.asarray(self.normal)
        return result

    def value(self, coordinates_angstrom):
        if self.status != "REGULAR":
            raise ValueError(f"{self.identifier}: singular SOOP reference plane")
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        return float(np.dot(xyz[self.moving_atom - 1] - self.reference_position_angstrom,
                            self.normal))

    def cartesian_gradient(self, natoms):
        """Analytic derivative of the fixed-frame scalar, not of moving-plane H."""
        return self.displacement(natoms).ravel()

    def cartesian_hessian(self, natoms):
        self.displacement(natoms)  # Validate support and regularity.
        return np.zeros((3 * natoms, 3 * natoms))


def soop_from_dict(data):
    return SelectiveOutOfPlanePrimitive(
        identifier=str(data["identifier"]), center_atom=int(data["center_atom"]),
        moving_atom=int(data["moving_atom"]),
        plane_atoms=tuple(int(a) for a in data["plane_atoms"]),
        endocyclic_ring_ids=tuple(int(a) for a in data["endocyclic_ring_ids"]),
        reference_position_angstrom=tuple(float(x) for x in data["reference_position_angstrom"]),
        normal=tuple(float(x) for x in data["normal"]),
        moving_atom_ring_ids=tuple(int(x) for x in data.get("moving_atom_ring_ids", ())),
        status=str(data.get("status", "REGULAR")),
        frame=str(data.get("frame", "FROZEN_REFERENCE_CARTESIAN")),
    )


def soop_support_key(primitive):
    """A ring-specific face is distinct from another face of the same bond."""
    return (
        primitive.center_atom, primitive.moving_atom,
        primitive.endocyclic_ring_ids[0] if primitive.endocyclic_ring_ids else 0,
    )


def expected_soop_support_keys(topology):
    adjacency = {atom: set() for atom in range(1, topology.natoms + 1)}
    owners = {}
    for left, right in topology.bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    for ring_id, ring in enumerate(topology.rings, 1):
        for left, right in zip(ring, ring[1:] + ring[:1]):
            owners.setdefault(tuple(sorted((left, right))), set()).add(ring_id)
    atomic_numbers = getattr(topology, "atomic_numbers", None)
    if atomic_numbers is not None and len(atomic_numbers) != topology.natoms:
        raise ValueError("SOOP topology atomic numbers do not match natoms")
    effective = {
        center: neighbors - {
            atom for atom in neighbors
            if atomic_numbers is not None
            and atomic_numbers[center - 1] <= 20
            and atomic_numbers[atom - 1] >= 21
        }
        for center, neighbors in adjacency.items()
    }
    return {
        (center, moving, ring_id)
        for center, neighbors in effective.items()
        if len(neighbors) >= 3
        for moving in neighbors
        for ring_id in owners.get(tuple(sorted((center, moving))), {0})
        if len(neighbors) == 3 or tuple(sorted((center, moving))) in owners
    }


def validate_soop_primitives(records, topology, *, auxiliary_bonds=()):
    # Auxiliary bonds identify a reactive atlas.  Its SOOP frame is frozen at
    # the lifecycle anchor and can therefore contain plane atoms that are not
    # neighbours in the instantaneous primary topology.  Keep the strict
    # equilibrium ownership/support checks for minima, while a reactive chart
    # is checked for valid atom-local geometry and immutable record structure.
    reactive_context = bool(auxiliary_bonds)
    adjacency = {a: set() for a in range(1, topology.natoms + 1)}
    ring_edges = {}
    memberships = {}
    for a, b in topology.bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    primary_adjacency = {atom: set(neighbors) for atom, neighbors in adjacency.items()}
    for a, b in auxiliary_bonds:
        if a == b or a not in adjacency or b not in adjacency:
            raise ValueError("invalid auxiliary SOOP support bond")
        adjacency[a].add(b)
        adjacency[b].add(a)
    for k, ring in enumerate(topology.rings, 1):
        for a in ring:
            memberships.setdefault(a, set()).add(k)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            ring_edges.setdefault(tuple(sorted((a, b))), set()).add(k)
    seen = set()
    seen_support = set()
    atomic_numbers = getattr(topology, "atomic_numbers", None)
    if atomic_numbers is not None and len(atomic_numbers) != topology.natoms:
        raise ValueError("SOOP topology atomic numbers do not match natoms")
    for p in records:
        c, x = p.center_atom, p.moving_atom
        effective_neighbors = adjacency.get(c, set()) - {
            atom for atom in adjacency.get(c, set())
            if atomic_numbers is not None
            and atomic_numbers[c - 1] <= 20
            and atomic_numbers[atom - 1] >= 21
        }
        primary_effective_neighbors = primary_adjacency.get(c, set()) - {
            atom for atom in primary_adjacency.get(c, set())
            if atomic_numbers is not None
            and atomic_numbers[c - 1] <= 20
            and atomic_numbers[atom - 1] >= 21
        }
        if (p.identifier in seen or c not in adjacency
                or len(effective_neighbors) < 3
                or len(p.plane_atoms) != 3 or p.plane_atoms[0] != c
                or len(set((*p.plane_atoms, x))) != 4
                or x not in effective_neighbors):
            raise ValueError("invalid SOOP local support")
        seen.add(p.identifier)
        support_key = soop_support_key(p)
        if support_key in seen_support:
            raise ValueError("duplicate SOOP ring-specific support")
        seen_support.add(support_key)
        owners = ring_edges.get(tuple(sorted((c, x))), set())
        if len(primary_effective_neighbors) >= 4 and not owners and not reactive_context:
            raise ValueError("saturated or hypercoordinate SOOP must be ring-owned")
        if not reactive_context and (len(p.endocyclic_ring_ids) > 1
                or not set(p.endocyclic_ring_ids).issubset(owners)
                or bool(owners) != bool(p.endocyclic_ring_ids)):
            raise ValueError("SOOP ring ownership disagrees with frozen topology")
        if p.endocyclic_ring_ids and not reactive_context:
            ring = topology.rings[p.endocyclic_ring_ids[0] - 1]
            if len(ring) >= 4:
                index = ring.index(c)
                direction = -1 if ring[(index + 1) % len(ring)] == x else 1
                expected_plane = (
                    ring[(index + direction) % len(ring)],
                    ring[(index + 2 * direction) % len(ring)],
                )
                if p.plane_atoms[1:] != expected_plane:
                    raise ValueError("SOOP endocyclic face plane is not local to its ring")
            elif not set(p.plane_atoms[1:]).issubset(effective_neighbors - {x}):
                raise ValueError("SOOP three-membered-ring plane has invalid support")
        elif (
            not reactive_context
            and not set(p.plane_atoms[1:]).issubset(effective_neighbors - {x})
        ):
            raise ValueError("SOOP exocyclic plane has invalid support")
        if not reactive_context and set(p.moving_atom_ring_ids) != memberships.get(x, set()):
            raise ValueError("SOOP moving-atom membership disagrees with frozen topology")
        if p.frame != "FROZEN_REFERENCE_CARTESIAN":
            raise ValueError("unsupported SOOP reference frame")
        if len(p.reference_position_angstrom) != 3 or not np.isfinite(p.reference_position_angstrom).all():
            raise ValueError("invalid SOOP reference position")
        if p.status == "REGULAR":
            if len(p.normal) != 3 or not np.isfinite(p.normal).all() or not np.isclose(np.linalg.norm(p.normal), 1., atol=1e-10):
                raise ValueError("SOOP requires a unit reference normal")
        elif p.status != "SINGULAR_PLANE" or p.normal:
            raise ValueError("invalid SOOP plane status")
