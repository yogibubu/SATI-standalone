"""Produce selective plane displacements from frozen primary topology."""

from itertools import combinations
import numpy as np
from matrix_chem.soop_contract import SelectiveOutOfPlanePrimitive, validate_soop_primitives


def build_selective_out_of_plane_primitives(topology, coordinates_angstrom):
    """Build the complete topology-owned selective out-of-plane pool.

    Endocyclic means the center--moving-atom bond belongs to a declared cycle;
    membership of the moving atom in some unrelated cycle is insufficient.
    Each ring owning a bond gets its own face-local plane. This distinction
    matters when fused faces are noncoplanar (for example, cage molecules).
    Singular planes remain explicit. Three-coordinate centers use their
    complete local pool. Four-coordinate and hypercoordinate centers use
    redundant valence angles for their acyclic local space and contribute only
    ring-owned SOOPs. Those ring directions remain necessary for spiro,
    bridged and multicentre faces, without introducing an unphysical full
    umbrella pool at a saturated center.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (topology.natoms, 3) or not np.isfinite(xyz).all():
        raise ValueError("SOOP geometry must match frozen primary topology")
    adjacency = {a: set() for a in range(1, topology.natoms + 1)}
    for a, b in topology.bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    owners = {}
    memberships = {}
    for k, ring in enumerate(topology.rings, 1):
        for atom in ring:
            memberships.setdefault(atom, []).append(k)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            owners.setdefault(tuple(sorted((a, b))), []).append(k)
    output = []
    atomic_numbers = getattr(topology, "atomic_numbers", None)
    if atomic_numbers is not None and len(atomic_numbers) != topology.natoms:
        raise ValueError("SOOP topology atomic numbers do not match natoms")
    for center, all_neighbors in sorted(adjacency.items()):
        # A main-group atom bound to a transition-metal interaction center is
        # classified by its covalent valence.  In ferrocene, for example, each
        # Cp carbon is locally tricoordinate (two C plus H), not a saturated
        # tetracoordinate carbon merely because the frozen topology also
        # records Fe--C coordination.
        coordination = {
            atom for atom in all_neighbors
            if atomic_numbers is not None
            and atomic_numbers[center - 1] <= 20
            and atomic_numbers[atom - 1] >= 21
        }
        neighbors = all_neighbors - coordination
        if len(neighbors) < 3:
            continue
        for moving in sorted(neighbors):
            ring_owners = tuple(owners.get(tuple(sorted((center, moving))), ()))
            if len(neighbors) >= 4 and not ring_owners:
                continue
            for ring_id in ring_owners or (None,):
                if ring_id is not None and len(topology.rings[ring_id - 1]) >= 4:
                    ring = topology.rings[ring_id - 1]
                    index = ring.index(center)
                    direction = -1 if ring[(index + 1) % len(ring)] == moving else 1
                    plane_pairs = ((ring[(index + direction) % len(ring)],
                                    ring[(index + 2 * direction) % len(ring)]),)
                else:
                    plane_pairs = tuple(combinations(sorted(neighbors - {moving}), 2))
                planes = []
                for first, second in plane_pairs:
                    u = xyz[first - 1] - xyz[center - 1]
                    v = xyz[second - 1] - xyz[center - 1]
                    normal = np.cross(u, v)
                    norm = float(np.linalg.norm(normal))
                    scale = float(np.linalg.norm(u) * np.linalg.norm(v))
                    regular = scale > 1e-20 and norm > 1e-10 * scale
                    unit = normal / norm if regular else np.zeros(3)
                    planes.append((int(regular), norm / max(scale, 1e-20),
                                   -first, -second, first, second, unit))
                _valid, _sine, _first_key, _second_key, first, second, unit = max(
                    planes, key=lambda item: item[:4]
                )
                suffix = f"_R{ring_id:04d}" if ring_id is not None else ""
                output.append(SelectiveOutOfPlanePrimitive(
                    identifier=f"SOOP_C{center:06d}_M{moving:06d}{suffix}",
                    center_atom=center, moving_atom=moving,
                    plane_atoms=(center, first, second),
                    endocyclic_ring_ids=((ring_id,) if ring_id is not None else ()),
                    reference_position_angstrom=tuple(xyz[moving - 1]),
                    normal=tuple(unit) if _valid else (),
                    moving_atom_ring_ids=tuple(memberships.get(moving, ())),
                    status="REGULAR" if _valid else "SINGULAR_PLANE",
                ))
    validate_soop_primitives(output, topology)
    return tuple(output)


def extend_reactive_out_of_plane_primitives(
    records,
    topology,
    coordinates_angstrom,
    atom_orders,
):
    """Add ORACLE-owned SOOP directions for atlas-declared reactive valences."""

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    memberships = {
        atom: tuple(index for index, ring in enumerate(topology.rings, 1) if atom in ring)
        for atom in range(1, topology.natoms + 1)
    }
    best = {}
    for atoms in atom_orders:
        if len(atoms) != 4 or len(set(atoms)) != 4:
            continue
        center, first, second, moving = atoms
        u = xyz[first - 1] - xyz[center - 1]
        v = xyz[second - 1] - xyz[center - 1]
        normal = np.cross(u, v)
        norm = float(np.linalg.norm(normal))
        scale = float(np.linalg.norm(u) * np.linalg.norm(v))
        regular = scale > 1.0e-20 and norm > 1.0e-10 * scale
        score = norm / max(scale, 1.0e-20)
        key = (center, moving)
        candidate = (int(regular), score, -first, -second, first, second, normal, norm)
        if key not in best or candidate[:4] > best[key][:4]:
            best[key] = candidate
    output = list(records)
    occupied = {(record.center_atom, record.moving_atom) for record in output}
    for (center, moving), candidate in sorted(best.items()):
        if (center, moving) in occupied:
            continue
        regular, _score, _first_key, _second_key, first, second, normal, norm = candidate
        output.append(SelectiveOutOfPlanePrimitive(
            identifier=f"SOOP_C{center:06d}_M{moving:06d}_REACTIVE",
            center_atom=center,
            moving_atom=moving,
            plane_atoms=(center, first, second),
            endocyclic_ring_ids=(),
            reference_position_angstrom=tuple(xyz[moving - 1]),
            normal=tuple(normal / norm) if regular else (),
            moving_atom_ring_ids=memberships[moving],
            status="REGULAR" if regular else "SINGULAR_PLANE",
        ))
    auxiliary_bonds = {
        tuple(sorted((atoms[0], atom)))
        for atoms in atom_orders
        for atom in atoms[1:]
    }
    validate_soop_primitives(output, topology, auxiliary_bonds=auxiliary_bonds)
    return tuple(output)
