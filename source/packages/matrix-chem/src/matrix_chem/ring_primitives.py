"""ORACLE-owned elementary ring supports shared with SONIC combiners."""
from .primitive_coordinates import Primitive

def _balanced_ring_anchors(ncyc: int) -> tuple[int, int, int]:
    """Choose a deterministic, maximally balanced reference triangle.

    The canonical ring origin is supplied by the topology layer.  The three
    positive cyclic gaps differ by at most one atom; this recovers alternating
    anchors for a six-membered ring and avoids geometry-dependent chart
    changes during an optimization.
    """

    if ncyc < 4:
        raise ValueError("triangular flap coordinates require at least four atoms")
    quotient, remainder = divmod(ncyc, 3)
    gaps = tuple(quotient + (1 if index < remainder else 0) for index in range(3))
    return 0, gaps[0], gaps[0] + gaps[1]


def _cyclic_arc_indices(start: int, end: int, ncyc: int) -> tuple[int, ...]:
    values: list[int] = []
    index = (start + 1) % ncyc
    while index != end:
        values.append(index)
        index = (index + 1) % ncyc
    return tuple(values)


def _triangular_flap_stencil(ring: tuple[int, ...]) -> tuple[tuple[int, int, int, int], ...]:
    """Return the N-3 oriented hinges of a balanced triangle tree.

    Each quartet is ``(parent_third, edge_start, edge_end, child_third)``.
    Its signed plane-incidence angle is evaluated by the analytic torsional
    kernel, but it is not an endocyclic bond torsion: the central edge is a
    triangulation diagonal and the two adjacent triples are triangular faces.
    """

    ncyc = len(ring)
    if ncyc <= 3:
        return ()
    anchor_indices = _balanced_ring_anchors(ncyc)
    stencils: list[tuple[int, int, int, int]] = []
    for arc_index in range(3):
        start_index = anchor_indices[arc_index]
        end_index = anchor_indices[(arc_index + 1) % 3]
        parent_index = anchor_indices[(arc_index + 2) % 3]
        interior = _cyclic_arc_indices(start_index, end_index, ncyc)
        if not interior:
            continue
        # The outermost face hinges directly on the reference triangle.
        stencils.append(
            (
                ring[parent_index],
                ring[start_index],
                ring[end_index],
                ring[interior[-1]],
            )
        )
        # Remaining faces form a fan rooted at the arc's first anchor.
        for position in range(len(interior) - 1, 0, -1):
            stencils.append(
                (
                    ring[end_index],
                    ring[start_index],
                    ring[interior[position]],
                    ring[interior[position - 1]],
                )
            )
    if len(stencils) != ncyc - 3:
        raise RuntimeError(f"triangular flap stencil has {len(stencils)} rows; expected {ncyc - 3}")
    return tuple(stencils)

def _charm_height_source_primitives(
    ring: tuple[int, ...],
) -> tuple[tuple[Primitive, ...], ...]:
    """Return cyclically balanced signed-height averages for every vertex.

    For an even ring of size at least six, the unique antipodal vertex is
    excluded from the reference pool.  This recovers exactly the four
    leave-one-out planes used by Marenich, Brothers, Hratchian and Frisch for
    each axial carbon of cyclohexane.  Odd rings have no unique antipode and
    therefore use all non-target vertices.  Uniformly spaced triplets keep
    the number of H primitives linear in ring size while treating every
    reference-pool vertex equivalently.
    """

    ncyc = len(ring)
    if ncyc <= 3:
        return ()
    sources: list[tuple[Primitive, ...]] = []
    for target_index, target in enumerate(ring):
        excluded = {target_index}
        if ncyc >= 6 and ncyc % 2 == 0:
            excluded.add((target_index + ncyc // 2) % ncyc)
        pool_indices = tuple(
            (target_index + offset) % ncyc
            for offset in range(1, ncyc)
            if (target_index + offset) % ncyc not in excluded
        )
        if len(pool_indices) < 3:
            pool_indices = tuple(index for index in range(ncyc) if index != target_index)
        size = len(pool_indices)
        first_offset = max(1, size // 3)
        second_offset = max(first_offset + 1, (2 * size) // 3)
        second_offset = min(second_offset, size - 1)
        triples: list[tuple[int, int, int]] = []
        seen: set[frozenset[int]] = set()
        for origin in range(size):
            triple = (
                pool_indices[origin],
                pool_indices[(origin + first_offset) % size],
                pool_indices[(origin + second_offset) % size],
            )
            key = frozenset(triple)
            if len(key) != 3 or key in seen:
                continue
            seen.add(key)
            triples.append(triple)
        if not triples:
            return ()
        # Reverse the cyclic plane orientation to reproduce the sign used in
        # the published cyclohexane H combinations.  H(i,j,k,l) has plane
        # j-i-k and out-of-plane atom l in Gaussian notation.
        sources.append(
            tuple(
                Primitive(
                    "out_of_plane_height",
                    (ring[first], ring[third], ring[second], target),
                )
                for first, second, third in triples
            )
        )
    return tuple(sources)
