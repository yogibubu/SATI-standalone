"""Explicit ownership boundaries for ORACLE integrations."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class OracleLayer:
    name: str
    owner: str
    responsibilities: tuple[str, ...]

ORACLE_LAYERS = (
    OracleLayer("perception", "matrix_chem", ("geometry", "topology", "symmetry", "primitives")),
    OracleLayer("state", "matrix_oracle", ("xyzin", "reports", "provenance")),
    OracleLayer("integration", "matrix_oracle.commands", ("CLI", "GUI", "remote QM requests")),
)

def layer_contract() -> dict[str, object]:
    return {"schema": "matrix.oracle.layers.v1", "layers": [
        {"name": item.name, "owner": item.owner, "responsibilities": list(item.responsibilities)}
        for item in ORACLE_LAYERS
    ]}
