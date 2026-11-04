from __future__ import annotations

def smith_workflow_plan(operation: str = "gic") -> dict[str, object]:
    plans={"gic":"matrix.smith.gic.v1", "sonic":"matrix.smith.sonic.v1", "sycart":"matrix.smith.sycart.v1"}
    if operation not in plans: raise ValueError(f"unsupported SMITH operation: {operation}")
    return {"schema":"matrix.smith.workflow.v1", "package":"matrix-smith", "public_name":"SMITH", "operation":operation, "produced_artifact":plans[operation], "execution_owner":"keymaker"}
