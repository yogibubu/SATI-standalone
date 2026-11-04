"""Select elementary face/contact supports from frozen ORACLE evidence."""
from collections import defaultdict
from itertools import combinations, permutations
from matrix_chem.primitive_coordinates import Primitive


def _contact_topology(fragments, bonds):
    """Build the contact lookup once per frozen pool, never globally cache it."""
    memberships = tuple(frozenset(f.atoms) for f in fragments)
    by_atom = {a: members for members in memberships for a in members}
    adjacency = defaultdict(set)
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return by_atom, adjacency


def contact_chart_primitives(contact, fragments, bonds, coords, *, _topology=None,
                             reactive_support_edges=()):
    """Freeze the elementary natural-coordinate supports of an atom contact."""
    import numpy as np
    from matrix_chem.contact_primitives import _local_contact_frame_atoms, _reactive_contact_frame_atoms
    from matrix_chem.primitive_coordinates import angle
    if contact.endpoint_a.kind != "ATOM" or contact.endpoint_b.kind != "ATOM":
        return
    left, right = int(contact.endpoint_a.identifier), int(contact.endpoint_b.identifier)
    by_atom, adjacency = (
        _contact_topology(fragments, bonds) if _topology is None else _topology
    )
    if left not in by_atom or right not in by_atom or by_atom[left] is by_atom[right]:
        return
    anchors = [_local_contact_frame_atoms(by_atom[e], endpoint=e,
                                         adjacency=adjacency, coords=coords)
               for e in (left, right)]
    contact_anchors = anchors
    if reactive_support_edges:
        support = defaultdict(set)
        for a, b in (*bonds, *reactive_support_edges):
            support[a].add(b)
            support[b].add(a)
        anchors = [_reactive_contact_frame_atoms(e, other, group, support, coords)
                   for e, other, group in ((left, right, anchors[0]),
                                           (right, left, anchors[1]))]
    for a in contact_anchors[0]:
        for b in contact_anchors[1]:
            yield Primitive("bond", tuple(sorted((a-1, b-1))))
    for a, b, c in [*((a, left, right) for a in anchors[0] if a != left),
                    *((left, right, a) for a in anchors[1] if a != right)]:
        atoms = (a-1,b-1,c-1)
        degrees = float(np.degrees(angle(*atoms, coords)))
        if degrees >= 165. or degrees <= 15.:
            for ordered in (min(atoms, atoms[::-1]),):
                gauges = {a-1 for group in anchors for a in group} - set(ordered)
                from matrix_chem.primitive_coordinates import linear_bend_reference_atom
                reference = linear_bend_reference_atom(
                    ordered, coords, candidate_indices=sorted(gauges),
                )
                if reference is not None:
                    gauges.add(reference)
                for gauge in sorted(gauges) if gauges else (None,):
                    for mode in (-1, -2):
                        yield Primitive("linear_bend", ordered, mode,
                                        () if gauge is None else (gauge,))
        else:
            yield Primitive("angle", atoms)
    louter = tuple(a for a in anchors[0] if a != left)
    router = tuple(a for a in anchors[1] if a != right)
    torsions = {(a,left,right,b) for a in louter for b in router}
    if len(louter) >= 2:
        torsions.add((louter[1],louter[0],left,right))
    if len(router) >= 2:
        torsions.add((left,right,router[0],router[1]))
    if anchors != contact_anchors:
        for outer in (louter, router):
            torsions.update((left,a,b,right) for a in outer for b in outer if a != b)
    for atoms in sorted(torsions):
        atoms = tuple(a-1 for a in atoms)
        if len(set(atoms)) == 4 and all(15. < np.degrees(angle(*triplet,coords)) < 165.
                                      for triplet in (atoms[:3],atoms[1:])):
            yield Primitive("dihedral", atoms)
    for endpoint, opposite, group in ((left,right,contact_anchors[0]),
                                      (right,left,contact_anchors[1])):
        outer = tuple(a for a in group if a != endpoint)
        for center, out in permutations(outer, 2):
            for triple in permutations((endpoint-1, opposite-1, out-1)):
                yield Primitive("out_of_plane", (center-1, *triple))


def linear_axis_primitives(bonds, coords):
    """Torsion of substituents at the ends of a frozen linear chain."""
    import numpy as np
    from matrix_chem.primitive_coordinates import angle
    adjacency = defaultdict(set)
    for a, b in bonds:
        adjacency[a-1].add(b-1)
        adjacency[b-1].add(a-1)
    linear = {c for c, neighbors in adjacency.items() if len(neighbors) == 2
              and np.degrees(angle(min(neighbors), c, max(neighbors), coords)) >= 165.}
    while linear:
        pending = [min(linear)]
        component = set()
        while pending:
            c = pending.pop()
            if c in component:
                continue
            component.add(c)
            pending.extend(adjacency[c] & linear - component)
        linear -= component
        ends = {a for c in component for a in adjacency[c]} - component
        if len(ends) != 2:
            continue
        left, right = sorted(ends)
        for a in sorted(adjacency[left]-component-{right}):
            for b in sorted(adjacency[right]-component-{left}):
                if len({a,left,right,b}) == 4:
                    yield Primitive("dihedral", (a,left,right,b))


def ring_and_local_plane_primitives(rings, bonds):
    """Elementary plane/flap supports, without any SONIC coefficients."""
    from matrix_chem.ring_primitives import _charm_height_source_primitives, _triangular_flap_stencil
    adjacency = defaultdict(set)
    for left, right in bonds:
        adjacency[left - 1].add(right - 1)
        adjacency[right - 1].add(left - 1)
    seen = set()
    def unique(primitive):
        if primitive not in seen:
            seen.add(primitive)
            return True
        return False
    for center, neighbors in adjacency.items():
        for triple in permutations(sorted(neighbors), 3):
            p = Primitive("out_of_plane", (center, *triple))
            if unique(p):
                yield p
    for face in rings:
        ring = tuple(a - 1 for a in face)
        n = len(ring)
        if n < 4:
            continue
        for i, center in enumerate(ring):
            for atoms in ((center, ring[(i-1)%n], ring[(i+2)%n], ring[(i+1)%n]),
                          (center, ring[(i+1)%n], ring[(i-2)%n], ring[(i-1)%n])):
                p = Primitive("out_of_plane", atoms)
                if unique(p):
                    yield p
        for p in (p for group in _charm_height_source_primitives(ring) for p in group):
            if unique(p):
                yield p
        for atoms in _triangular_flap_stencil(ring):
            p = Primitive("dihedral", atoms)
            if unique(p):
                yield p


def topology_special_primitives(rings, bonds, interaction_centers, contacts):
    edges = defaultdict(list)
    for face in rings:
        face = tuple(a-1 for a in face)
        if len(face) >= 4:
            for atom in face:
                yield Primitive("signed_face_flap", (atom,), ref=tuple(a for a in face if a != atom))
        for a,b in zip(face, face[1:]+face[:1]):
            edges[tuple(sorted((a,b)))].append(face)
    for edge, faces in edges.items():
        for left,right in combinations(faces,2):
            yield Primitive("face_hinge", left, ref=right, frame_atoms=edge)
    centers = {c.identifier:c for c in interaction_centers.centers}
    for contact in interaction_centers.interactions:
        face = centers[contact.center_id]
        if face.kind not in {"RING_CENTER", "HAPTIC_CENTER"}:
            continue
        atoms = (contact.atom-1,)
        ref = tuple(a-1 for a in face.atoms)
        yield Primitive("center_face_distance", atoms, ref=ref)
        yield Primitive("face_direction_angle", atoms, ref=ref)
    adjacency = defaultdict(set)
    for a,b in bonds:
        adjacency[a].add(b); adjacency[b].add(a)
    for contact in contacts:
        if contact.endpoint_a.kind != "ATOM" or contact.endpoint_b.kind != "ATOM":
            continue
        if not any(term in contact.kind for term in ("HBOND", "HYDROGEN", "HALOGEN", "XBOND")):
            continue
        a,b = int(contact.endpoint_a.identifier),int(contact.endpoint_b.identifier)
        for vertex,other in ((a,b),(b,a)):
            for anchor in sorted(adjacency[vertex]-{other}):
                yield Primitive("contact_center_angle",(anchor-1,),ref=(other-1,),frame_atoms=(vertex-1,))
