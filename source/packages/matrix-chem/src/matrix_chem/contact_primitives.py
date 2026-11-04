"""ORACLE-owned contact frame supports; indices are one-based."""
from collections import deque
import numpy as np

def _local_contact_frame_atoms(
    atoms: tuple[int, ...] | frozenset[int],
    *,
    endpoint: int,
    adjacency: dict[int, set[int]],
    coords: np.ndarray,
) -> tuple[int, ...]:
    """Choose contact anchors within three covalent edges of the endpoint.

    Frozen fragment memberships should be supplied as frozensets so repeated
    contacts never copy the whole fragment. With bounded coordination, work
    per contact is bounded independently of molecular size. No distant atom
    is silently used to rescue a locally degenerate frame.
    """

    members = atoms if isinstance(atoms, frozenset) else frozenset(int(atom) for atom in atoms)
    if endpoint not in members:
        raise ValueError("pseudobond endpoint is outside its frozen fragment")
    if len(members) <= 1:
        return (endpoint,)
    distance = {endpoint: 0}
    pending = deque([endpoint])
    while pending:
        current = pending.popleft()
        if distance[current] >= 3:
            continue
        for neighbor in sorted(a for a in adjacency[current] if a in members):
            if neighbor in distance:
                continue
            distance[neighbor] = distance[current] + 1
            pending.append(neighbor)
    candidates = tuple(sorted(set(distance) - {endpoint}))
    if not candidates:
        return (endpoint,)
    second = min(
        candidates,
        key=lambda atom: (
            distance.get(atom, len(members)),
            -float(np.linalg.norm(coords[atom - 1] - coords[endpoint - 1])),
            atom,
        ),
    )
    if len(candidates) == 1:
        return endpoint, second
    axis = coords[second - 1] - coords[endpoint - 1]
    remaining = tuple(atom for atom in candidates if atom != second)
    third = min(
        remaining,
        key=lambda atom: (
            -float(
                np.linalg.norm(
                    np.cross(axis, coords[atom - 1] - coords[endpoint - 1])
                )
            ),
            distance.get(atom, len(members)),
            atom,
        ),
    )
    return endpoint, second, third


def _reactive_contact_frame_atoms(
    endpoint: int,
    opposite: int,
    anchors: tuple[int, ...],
    adjacency: dict[int, set[int]],
    coords: np.ndarray,
) -> tuple[int, ...]:
    """Complete a collinear contact frame within three declared graph edges.

    Exclude the opposite endpoint while searching the declared graph. Include the
    entire bounded local shell, rather than choosing one arbitrary transverse
    atom: symmetry-related anchors must remain available to the SALC builder.
    Frozen fragment ownership and the reactive-zone exclusions are unchanged.
    A wholly linear side has no transverse anchor and remains unsupported.
    """
    axis = coords[opposite - 1] - coords[endpoint - 1]
    length = float(np.linalg.norm(axis))
    if length <= 1.e-10:
        return anchors
    from .primitive_coordinates import LINEAR_ANGLE_DEGREES
    minimum_sine = np.sin(np.deg2rad(180. - LINEAR_ANGLE_DEGREES))
    def regular(atom):
        vector = coords[atom - 1] - coords[endpoint - 1]
        radius = float(np.linalg.norm(vector))
        return radius > 1.e-10 and np.linalg.norm(np.cross(axis, vector)) > minimum_sine * length * radius
    if any(regular(a) for a in anchors):
        return anchors
    distance = {endpoint: 0}
    pending = [endpoint]
    for current in pending:
        if distance[current] >= 3:
            continue
        for neighbor in sorted(adjacency[current]):
            if neighbor == opposite or neighbor in distance:
                continue
            distance[neighbor] = distance[current] + 1
            pending.append(neighbor)
    local = tuple(sorted(distance, key=lambda a: (distance[a], a)))
    if not any(regular(a) for a in local):
        return anchors
    return tuple(dict.fromkeys((*anchors, *local)))
