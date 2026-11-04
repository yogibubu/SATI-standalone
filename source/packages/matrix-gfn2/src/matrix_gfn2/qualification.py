"""Fail-closed production qualification for MATRIX-GFN2."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class QualificationStatus:
    schema: str
    production_ready: bool
    gates: Mapping[str, bool]
    reason: str

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.gates.items() if not passed)

    def require_production_ready(self) -> None:
        if not self.production_ready or self.failed_gates:
            failures = ", ".join(self.failed_gates) or "production_ready"
            raise RuntimeError(f"MATRIX-GFN2 production qualification failed: {failures}")


@dataclass(frozen=True)
class ParityMetrics:
    energy_max_abs_hartree: float
    gradient_max_abs_hartree_per_bohr: float
    charge_max_abs_electron: float
    bond_order_max_abs: float
    passed: bool


def qualification_status() -> QualificationStatus:
    resource = files("matrix_gfn2").joinpath("data/qualification.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    gates = {str(name): bool(value) for name, value in payload["gates"].items()}
    ready = bool(payload["production_ready"])
    if ready != all(gates.values()):
        raise RuntimeError("inconsistent MATRIX-GFN2 qualification manifest")
    return QualificationStatus(
        schema=str(payload["schema"]),
        production_ready=ready,
        gates=gates,
        reason=str(payload["reason"]),
    )


def evaluate_parity(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    *,
    energy_tolerance_hartree: float = 2.0e-6,
    gradient_tolerance_hartree_per_bohr: float = 2.0e-5,
    charge_tolerance_electron: float = 2.0e-4,
    bond_order_tolerance: float = 5.0e-4,
) -> ParityMetrics:
    """Compare named corpus records without depending on an oracle runtime."""

    def identifier(item: Mapping[str, object]) -> str:
        value = item.get("case_id", item.get("name"))
        if value is None:
            raise ValueError("parity record requires case_id or name")
        return str(value)

    reference_by_name = {identifier(item): item for item in reference}
    candidate_by_name = {identifier(item): item for item in candidate}
    if candidate_by_name.keys() != reference_by_name.keys():
        raise ValueError("parity candidate and reference case names differ")

    maxima = np.zeros(4, dtype=float)
    for name, expected in reference_by_name.items():
        actual = candidate_by_name[name]
        maxima[0] = max(
            maxima[0],
            abs(float(actual["energy_hartree"]) - float(expected["energy_hartree"])),
        )
        for index, field in enumerate(
            ("gradient_hartree_per_bohr", "charges_electron", "bond_orders"), start=1
        ):
            difference = np.asarray(actual[field], dtype=float) - np.asarray(
                expected[field], dtype=float
            )
            maxima[index] = max(maxima[index], float(np.max(np.abs(difference))))
    thresholds = np.asarray(
        (
            energy_tolerance_hartree,
            gradient_tolerance_hartree_per_bohr,
            charge_tolerance_electron,
            bond_order_tolerance,
        )
    )
    return ParityMetrics(*map(float, maxima), passed=bool(np.all(maxima <= thresholds)))


__all__ = [
    "ParityMetrics",
    "QualificationStatus",
    "evaluate_parity",
    "qualification_status",
]
