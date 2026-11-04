"""Canonical dispersion contracts for the PySCF provider.

The scientific parameters remain owned by :mod:`matrix_qm`.  This module only
translates an approved MATRIX contract into the public ``dftd3.pyscf`` API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from matrix_qm import load_canonical_qm_ladder


@dataclass(frozen=True)
class PySCFDispersionContract:
    contract_id: str
    model: str
    version: str
    atm: bool
    parameters: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "model": self.model,
            "version": self.version,
            "atm": self.atm,
            "parameters": deepcopy(self.parameters),
        }


def resolve_pyscf_dispersion_contract(contract_id: str) -> PySCFDispersionContract:
    """Resolve one approved shared contract without duplicating its settings."""

    requested = str(contract_id).strip()
    contracts = load_canonical_qm_ladder().payload.get("dispersion_contracts", {})
    if requested not in contracts:
        raise ValueError(f"unknown canonical dispersion contract: {contract_id}")
    payload = contracts[requested]
    if payload.get("model") != "D3(BJ)-ATM":
        raise ValueError(
            f"PySCF adapter does not implement dispersion model {payload.get('model')!r}"
        )
    required = ("s6", "s8", "a1", "a2", "s9", "alp")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            f"canonical dispersion contract {requested} lacks: " + ", ".join(missing)
        )
    parameters = {name: float(payload[name]) for name in required}
    return PySCFDispersionContract(
        contract_id=requested,
        model="D3(BJ)-ATM",
        version="d3bj",
        atm=True,
        parameters=parameters,
    )


__all__ = ["PySCFDispersionContract", "resolve_pyscf_dispersion_contract"]
