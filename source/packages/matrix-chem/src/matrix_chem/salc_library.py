"""Shared catalogue of analytic SALC and exact-completion algorithms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json


SALC_LIBRARY_SCHEMA = "matrix.salc_library.v1"
SALC_NONE = "NONE"
SALC_POINT_GROUP_PROJECTOR = "ANALYTIC_POINT_GROUP_PROJECTOR"
SALC_LOCAL_BLOCK = "ANALYTIC_LOCAL_BLOCK"
SALC_B_ORTHOGONAL = "ANALYTIC_B_ORTHOGONAL"
SALC_PROJECTOR_THEN_B_ORTHOGONAL = "ANALYTIC_PROJECTOR_THEN_B_ORTHOGONAL"
COMPLETION_EXACT_RANK = "EXACT_RANK_REVEALING_COMPLETION"


@dataclass(frozen=True)
class SalcAlgorithm:
    algorithm_id: str
    operation: str
    domain: str
    preserves_span: bool
    preserves_rank: bool
    requires_same_irrep: bool
    analytic: bool
    implementation: str


@lru_cache(maxsize=1)
def salc_algorithms() -> tuple[SalcAlgorithm, ...]:
    payload = json.loads(
        files("matrix_chem")
        .joinpath("data", "salc_library_v1.json")
        .read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or payload.get("schema") != SALC_LIBRARY_SCHEMA:
        raise RuntimeError("invalid declarative SALC library schema")
    records = payload.get("algorithms")
    if not isinstance(records, list):
        raise RuntimeError("SALC library algorithms must be an array")
    expected = set(SalcAlgorithm.__dataclass_fields__)
    algorithms = tuple(
        SalcAlgorithm(**record)
        for record in records
        if isinstance(record, dict) and set(record) == expected
    )
    if len(algorithms) != len(records):
        raise RuntimeError("SALC library entry fields do not match the typed model")
    identifiers = tuple(item.algorithm_id for item in algorithms)
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("SALC algorithm identifiers must be unique")
    return algorithms


def salc_algorithm(algorithm_id: str) -> SalcAlgorithm:
    normalized = str(algorithm_id).strip().upper()
    for item in salc_algorithms():
        if item.algorithm_id == normalized:
            return item
    raise KeyError(f"unregistered SALC algorithm: {normalized}")


def salc_library_manifest() -> dict[str, object]:
    payload = {
        "schema": SALC_LIBRARY_SCHEMA,
        "algorithms": [asdict(item) for item in salc_algorithms()],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "COMPLETION_EXACT_RANK",
    "SALC_B_ORTHOGONAL",
    "SALC_LIBRARY_SCHEMA",
    "SALC_LOCAL_BLOCK",
    "SALC_NONE",
    "SALC_POINT_GROUP_PROJECTOR",
    "SALC_PROJECTOR_THEN_B_ORTHOGONAL",
    "SalcAlgorithm",
    "salc_algorithm",
    "salc_algorithms",
    "salc_library_manifest",
]
