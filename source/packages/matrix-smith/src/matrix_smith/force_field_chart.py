"""Bind an explicitly selected SMITH chart to its ORACLE perception epoch."""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class SonicForceFieldChart:
    definition: object
    atoms: tuple[str, ...]
    topology_sha256: str


def bind_force_field_chart(definition, *, atoms, oracle_topology):
    """Record the provenance of a chart built from conserved ORACLE evidence.

    Coordinate selection (including reactive combinations) remains SMITH's
    responsibility. This boundary does not perform chemical reperception or
    replace the chart's geometric rank check in the inverse-map runtime.
    """
    atoms = tuple(atoms)
    if (oracle_topology.get("schema") != "matrix.oracle.force_field_topology.v1"
            or oracle_topology.get("owner") != "ORACLE"
            or tuple(oracle_topology.get("atoms", ())) != atoms
            or len(definition.reference_coordinates_angstrom) != len(atoms)):
        raise ValueError("SMITH chart atom ordering does not match the ORACLE topology")
    for primitive in definition.primitives:
        if any(int(i) != i or not 1 <= i <= len(atoms) for i in
               (*primitive.atoms, *primitive.ref_atoms, *primitive.frame_atoms, *primitive.ref_frame_atoms)):
            raise ValueError("SMITH primitive has invalid atom support")
    identity = hashlib.sha256(json.dumps(oracle_topology, sort_keys=True,
                                        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return SonicForceFieldChart(definition, atoms, identity)
