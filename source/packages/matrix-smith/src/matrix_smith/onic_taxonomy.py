"""Public taxonomy for the common ONIC coordinate theory."""

from __future__ import annotations

from dataclasses import dataclass


ONIC_TAXONOMY_SCHEMA = "matrix.onic.taxonomy.v1"


@dataclass(frozen=True)
class OnicBranch:
    acronym: str
    name: str
    orientation: str
    role: str


ONIC = OnicBranch(
    "ONIC",
    "Oriented Non-redundant Internal Coordinates",
    "COMMON_TYPED_NONREDUNDANT_CORE",
    "shared algebra and typed contract",
)
TONIC = OnicBranch(
    "TONIC",
    "Task-Oriented Non-redundant Internal Coordinates",
    "TASK",
    "general framework",
)
CONIC = OnicBranch(
    "CONIC",
    "Continuity-Oriented Non-redundant Internal Coordinates",
    "CONTINUITY",
    "exploration",
)
SONIC = OnicBranch(
    "SONIC",
    "Symmetry-Oriented Non-redundant Internal Coordinates",
    "SYMMETRY",
    "exploitation",
)
ONIC_BRANCHES = (TONIC, CONIC, SONIC)

# ESALC is the public name of the symmetry-adapted branch.  SONIC remains the
# serialized identifier in v1 artifacts, so readers must continue to accept it.
ESALC = OnicBranch(
    "ESALC",
    "Effective Symmetry-Adapted Local Coordinates",
    "SYMMETRY",
    "exploitation",
)


def onic_branch_for_role(role: str) -> OnicBranch:
    normalized = str(role).strip().casefold()
    if normalized in {"general", "task", "task-oriented", "tonic"}:
        return TONIC
    if normalized in {"exploration", "continuity", "continuity-oriented", "conic"}:
        return CONIC
    if normalized in {
        "exploitation", "symmetry", "symmetry-oriented", "esalc", "sonic"
    }:
        return SONIC
    raise ValueError(f"unsupported ONIC chart role: {role}")


__all__ = [
    "CONIC",
    "ESALC",
    "ONIC",
    "ONIC_BRANCHES",
    "ONIC_TAXONOMY_SCHEMA",
    "OnicBranch",
    "SONIC",
    "TONIC",
    "onic_branch_for_role",
]
