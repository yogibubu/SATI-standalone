"""
Topology validator for ORACLE.

Guarantees:
- Hydrogens have at most one ordinary constitutional bond
- No isolated non-hydrogen atoms
"""


def validate_topology(dg):
    errors = []

    for i, Z in enumerate(dg.Z):
        deg = len(dg.adjacency[i])

        if Z == 1:
            if deg not in {0, 1}:
                errors.append(f"Hydrogen {i} has degree {deg}")
        else:
            if deg == 0:
                errors.append(f"Atom {i} (Z={Z}) is isolated")

    if errors:
        raise ValueError("Topology validation failed:\n" + "\n".join(errors))
