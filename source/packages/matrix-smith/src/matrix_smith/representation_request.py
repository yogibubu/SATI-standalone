"""Workflow-level representation requests shared by MATRIX consumers."""

from __future__ import annotations

from dataclasses import dataclass


REPRESENTATION_MODES = frozenset({"SCALAR", "PERIODIC_EMBEDDING", "QUATERNION_POSE", "CARTESIAN"})
REPRESENTATION_PURPOSES = frozenset({"LOCAL_OPTIMIZATION", "GLOBAL_PES", "RIGID_EXPLORATION", "QM_REALIZATION"})


@dataclass(frozen=True)
class RepresentationRequest:
    mode: str
    purpose: str
    contract_schema: str = "matrix.smith.representation_contract.v1"
    continuous: bool = False

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().upper()
        purpose = str(self.purpose).strip().upper()
        if mode not in REPRESENTATION_MODES:
            raise ValueError(f"unsupported representation mode: {self.mode}")
        if purpose not in REPRESENTATION_PURPOSES:
            raise ValueError(f"unsupported representation purpose: {self.purpose}")
        if purpose == "GLOBAL_PES" and mode != "PERIODIC_EMBEDDING":
            raise ValueError("GLOBAL_PES requires PERIODIC_EMBEDDING")
        if purpose == "RIGID_EXPLORATION" and mode != "QUATERNION_POSE":
            raise ValueError("RIGID_EXPLORATION requires QUATERNION_POSE")
        if purpose == "QM_REALIZATION" and mode != "CARTESIAN":
            raise ValueError("QM_REALIZATION requires CARTESIAN")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "purpose", purpose)
