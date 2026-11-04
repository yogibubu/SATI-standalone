"""Immutable, serializable ORACLE evidence consumed by ARCHITECT.

This is a snapshot of an existing perception, not another perception engine.
Indices are zero based; bond orders are retained without integer rounding.
"""

from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class OracleForceFieldTopology:
    atoms: tuple[str, ...]
    charges: tuple[float, ...]
    effective_atomic_numbers: tuple[float, ...]
    bond_orders: tuple[tuple[int, int, float], ...]
    synthon_records_json: str
    reliable: bool
    reactive_atoms: tuple[int, ...] = ()
    epoch: int = 0
    provenance: str = "ORACLE"

    def __post_init__(self):
        from matrix_chem.topology.elements import atomic_number

        atoms = tuple(self.atoms)
        n = len(atoms)
        if not n or any(not atomic_number(symbol) for symbol in atoms):
            raise ValueError("ORACLE force-field snapshot requires recognized atoms")
        charges = tuple(map(float, self.charges))
        effective = tuple(map(float, self.effective_atomic_numbers))
        if (len(charges) != n or len(effective) != n
                or any(not math.isfinite(v) for v in charges)
                or any(not math.isfinite(v) or v <= 0 for v in effective)):
            raise ValueError("ORACLE snapshot requires complete finite charges and effective atomic numbers")
        orders, seen = [], set()
        for a, b, value in self.bond_orders:
            if int(a) != a or int(b) != b or not 0 <= a < n or not 0 <= b < n or a == b:
                raise ValueError("invalid ORACLE bond indices")
            pair = tuple(sorted((int(a), int(b))))
            if pair in seen or not math.isfinite(value) or value < 0:
                raise ValueError("invalid or duplicated ORACLE continuous bond order")
            seen.add(pair)
            orders.append((*pair, float(value)))
        reactive = tuple(self.reactive_atoms)
        if (len(set(reactive)) != len(reactive)
                or any(int(i) != i or not 0 <= i < n for i in reactive)
                or type(self.reliable) is not bool or int(self.epoch) != self.epoch or self.epoch < 0
                or not self.provenance):
            raise ValueError("invalid ORACLE topology reliability/epoch")
        records = json.loads(self.synthon_records_json)
        if not isinstance(records, list) or len(records) != n or any(not isinstance(r, dict) for r in records):
            raise ValueError("ORACLE snapshot requires one synthon record per atom")
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "charges", charges)
        object.__setattr__(self, "effective_atomic_numbers", effective)
        object.__setattr__(self, "bond_orders", tuple(sorted(orders)))
        object.__setattr__(self, "reactive_atoms", tuple(sorted(map(int, reactive))))
        object.__setattr__(self, "epoch", int(self.epoch))
        object.__setattr__(self, "synthon_records_json", canonical)

    def to_dict(self):
        return {
            "schema": "matrix.oracle.force_field_topology.v1", "owner": "ORACLE",
            "atoms": list(self.atoms), "charges": list(self.charges),
            "effective_atomic_numbers": list(self.effective_atomic_numbers),
            "bond_orders": [list(row) for row in self.bond_orders],
            "synthons": json.loads(self.synthon_records_json), "reliable": self.reliable,
            "reactive_atoms": list(self.reactive_atoms), "epoch": self.epoch,
            "provenance": self.provenance, "atom_index_base": 0,
        }

    @classmethod
    def from_dict(cls, record):
        if (record.get("schema") != "matrix.oracle.force_field_topology.v1"
                or record.get("owner") != "ORACLE" or record.get("atom_index_base") != 0):
            raise ValueError("invalid ORACLE force-field snapshot contract")
        return cls(record["atoms"], record["charges"], record["effective_atomic_numbers"],
                   record["bond_orders"], json.dumps(record["synthons"], allow_nan=False),
                   record["reliable"], record["reactive_atoms"], record["epoch"], record["provenance"])

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze_force_field_topology(atoms, bonds, synthons, *, reliable, reactive_atoms=(),
                               epoch=0, provenance="ORACLE"):
    """Keep the already perceived synthons, charges and continuous orders.

    Reliability is an explicit ORACLE decision, never guessed from a graph
    merely being available. No QM call or charge/bond-order fallback occurs.
    """
    from matrix_chem.topology.elements import atomic_number

    atoms = tuple(atoms)
    if tuple(map(int, synthons.Z)) != tuple(atomic_number(atom) for atom in atoms):
        raise ValueError("ORACLE synthon atom ordering differs from the molecular snapshot")
    records = []
    for i in range(len(atoms)):
        records.append({
            "signature": synthons.canonical_signature_str(i),
            **{name: float(getattr(synthons, method)(i)) for name, method in (
                ("coordination", "cna"), ("electron_domains", "electron_domains"),
                ("effective_radius", "covalent_radius_eff"), ("polarizability", "polarizability"),
                ("covalency", "covalency"), ("delocalization", "delocalization"),
                ("strain", "strain"), ("sigma_index", "sigma_index"),
                ("pi_index", "pi_index"), ("pi_pi_index", "pi_pi_index"))}})
    return OracleForceFieldTopology(
        tuple(atoms), tuple(float(synthons.charge(i)) for i in range(len(atoms))),
        tuple(float(synthons.Zeff(i)) for i in range(len(atoms))),
        tuple((a, b, float(synthons.bond_order(a, b))) for a, b in bonds),
        json.dumps(records, allow_nan=False), reliable, tuple(reactive_atoms), epoch, provenance)
