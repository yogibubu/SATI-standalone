"""Deterministic resource planning shared by all LOCK Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Mapping, Sequence


LOCK_RESOURCE_PLAN_SCHEMA = "matrix.lock.resource_plan.v1"


@dataclass(frozen=True)
class LOCKBlockRequirement:
    key: str
    dimension: int
    roots: int
    factorized_term_count: int = 1
    complex_valued: bool = True
    reusable_subspace: bool = True

    def __post_init__(self) -> None:
        if (
            not self.key.strip()
            or self.dimension < 1
            or not 1 <= self.roots <= self.dimension
            or self.factorized_term_count < 1
        ):
            raise ValueError("invalid LOCK block requirement")


def plan_lock_resources(
    blocks: Sequence[LOCKBlockRequirement | Mapping[str, Any]],
    *,
    memory_budget_bytes: int = 2 * 1024**3,
    requested_workers: int | None = None,
    dense_root_fraction: float = 0.35,
    dense_max_dimension: int = 384,
    input_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose solver, storage, workers and reusable checkpoint keys per block."""

    normalized = tuple(
        item if isinstance(item, LOCKBlockRequirement) else LOCKBlockRequirement(
            key=str(item["key"]),
            dimension=int(item["dimension"]),
            roots=int(item["roots"]),
            factorized_term_count=int(item.get("factorized_term_count", 1)),
            complex_valued=bool(item.get("complex_valued", True)),
            reusable_subspace=bool(item.get("reusable_subspace", True)),
        )
        for item in blocks
    )
    if (
        not normalized
        or len({item.key for item in normalized}) != len(normalized)
        or memory_budget_bytes < 1
        or not 0.0 < dense_root_fraction <= 1.0
        or dense_max_dimension < 1
    ):
        raise ValueError("invalid LOCK resource-planning request")
    cpu_count = max(os.cpu_count() or 1, 1)
    requested = cpu_count if requested_workers is None else int(requested_workers)
    if requested < 1:
        raise ValueError("requested_workers must be positive")
    contract_hash = _digest(dict(input_contract or {}))
    rows = []
    for block in normalized:
        scalar_bytes = 16 if block.complex_valued else 8
        dense_bytes = scalar_bytes * block.dimension**2
        vector_bytes = scalar_bytes * block.dimension * max(block.roots + 8, 16)
        factorized_bytes = scalar_bytes * block.dimension * max(
            block.factorized_term_count, 1
        )
        wants_dense = (
            block.dimension <= dense_max_dimension
            or block.roots == block.dimension
            or block.roots / block.dimension >= dense_root_fraction
        )
        dense_allowed = dense_bytes <= memory_budget_bytes
        if block.roots == block.dimension and not dense_allowed:
            backend = "blocked-dense-required"
            reason = "all eigenpairs require more memory than the declared budget"
        elif wants_dense and dense_allowed:
            backend = "dense"
            reason = "small block or large requested root fraction"
        else:
            backend = "matrix-free"
            reason = "partial spectrum with lower Krylov storage cost"
        working_bytes = dense_bytes if backend == "dense" else vector_bytes + factorized_bytes
        cost = (
            block.dimension**3 if backend == "dense"
            else block.factorized_term_count * block.dimension * max(block.roots, 1)
        )
        checkpoint_payload = {
            "contract_sha256": contract_hash,
            "block": block.key,
            "dimension": block.dimension,
            "roots": block.roots,
            "backend": backend,
        }
        rows.append({
            "key": block.key,
            "dimension": block.dimension,
            "roots": block.roots,
            "backend": backend,
            "reason": reason,
            "dense_storage_bytes": dense_bytes,
            "estimated_working_bytes": working_bytes,
            "relative_cost": int(cost),
            "reuse_subspace": block.reusable_subspace and backend == "matrix-free",
            "checkpoint_key": _digest(checkpoint_payload),
        })
    runnable = [row for row in rows if row["backend"] != "blocked-dense-required"]
    largest_worker_bytes = max(
        (int(row["estimated_working_bytes"]) for row in runnable), default=memory_budget_bytes
    )
    memory_workers = max(memory_budget_bytes // max(largest_worker_bytes, 1), 1)
    workers = min(requested, cpu_count, memory_workers, max(len(runnable), 1))
    schedule = [
        row["key"]
        for row in sorted(runnable, key=lambda item: (-item["relative_cost"], item["key"]))
    ]
    blocked = [row["key"] for row in rows if row["backend"] == "blocked-dense-required"]
    return {
        "schema": LOCK_RESOURCE_PLAN_SCHEMA,
        "input_contract_sha256": contract_hash,
        "memory_budget_bytes": memory_budget_bytes,
        "requested_workers": requested,
        "selected_workers": int(workers),
        "deterministic_schedule": schedule,
        "blocks": rows,
        "blocked_blocks": blocked,
        "ready": not blocked,
        "resume_policy": (
            "reuse a subspace only when its checkpoint key and block dimensions match"
        ),
    }


def adaptive_solver_policy_from_plan(plan: Mapping[str, Any]):
    """Translate a resource plan to the existing adaptive eigensolver policy."""

    if plan.get("schema") != LOCK_RESOURCE_PLAN_SCHEMA:
        raise ValueError("unsupported LOCK resource plan")
    from .adaptive_solver import AdaptiveSolverPolicy

    dense_dimensions = [
        int(item["dimension"]) for item in plan["blocks"] if item["backend"] == "dense"
    ]
    return AdaptiveSolverPolicy(
        dense_max_dimension=max(dense_dimensions, default=1),
        memory_budget_bytes=int(plan["memory_budget_bytes"]),
    )


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "LOCK_RESOURCE_PLAN_SCHEMA",
    "LOCKBlockRequirement",
    "adaptive_solver_policy_from_plan",
    "plan_lock_resources",
]


def plan_lam_fit_resources(systems,request):
    """Allocate one CPU/memory budget across isotope, Jacobian and J workers."""
    options=request.get('options',{});budget=int(options.get('memory_budget_bytes',2*1024**3))
    requested=int(options.get('workers',1));outer=int(options.get('dataset_workers',requested))
    if min(budget,requested,outer)<1:raise ValueError('LAM fit resource budgets must be positive')
    cpu=min(requested,max(1,os.cpu_count() or 1));rows=[]
    for dataset in request['datasets']:
        name=str(dataset['system']);surface=systems[name];local=dataset['request'];blocks=[]
        selectors=[c[s] for o in local['observations'] for c in o.get('components',[o]) for s in ('lower','upper')]
        nlam=int(local.get('lam_states',12));k=local.get('options',{}).get('k_contraction_states')
        contracted=nlam if k is None else min(nlam,int(k)*surface.symmetry_sector_count)
        for J in sorted({int(s['J']) for s in selectors}):
            dimension=contracted*(2*J+1);roots=min(dimension,int(local.get('levels_per_j') or dimension))
            blocks.append(LOCKBlockRequirement(str(J),dimension,roots))
        plan=plan_lock_resources(blocks,memory_budget_bytes=budget,requested_workers=1)
        # Includes resident roots, the largest eigensolve and LAM primitive work.
        primitive=int(__import__('math').prod(surface.grid_shape))
        resident=sum(b.dimension*b.roots*16 for b in blocks)
        memory=resident+max(row['estimated_working_bytes'] for row in plan['blocks'])+16*primitive*nlam
        rows.append(dict(system=name,estimated_working_bytes=memory,relative_cost=sum(row['relative_cost'] for row in plan['blocks'])))
    largest=max(row['estimated_working_bytes'] for row in rows)
    if largest>budget:raise MemoryError('LAM system exceeds the declared memory budget; reduce roots or contract the basis')
    count=min(outer,cpu,len(rows),max(1,budget//largest));inner=max(1,cpu//count)
    return dict(schema='matrix.lock.lam_fit_resource_plan.v1',cpu_budget=cpu,memory_budget_bytes=budget,
        dataset_workers=count,workers_per_system=inner,blas_threads=1,systems=rows,
        deterministic_schedule=[r['system'] for r in sorted(rows,key=lambda r:(-r['relative_cost'],r['system']))])
