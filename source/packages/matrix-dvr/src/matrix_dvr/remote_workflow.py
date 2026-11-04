"""Portable packaging and submission plans for remote ESALC/DVR runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


@dataclass(frozen=True)
class RemoteRunRequest:
    host: str
    remote_directory: str = "~/matrix-esalc-runs"
    command: str = "matrix lock tunneling esalc-scan"


def build_remote_bundle(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(shutil.make_archive(str(destination.with_suffix("")), "gztar", root_dir=source))
    return archive


def write_remote_plan(outdir: Path, request: RemoteRunRequest, bundle: Path) -> Path:
    payload = {
        "schema": "matrix.tunneling.remote_plan.v1",
        "host": request.host,
        "remote_directory": request.remote_directory,
        "bundle": str(bundle),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "commands": [
            f"scp {bundle} {request.host}:{request.remote_directory}/",
            f"ssh {request.host} 'cd {request.remote_directory} && tar -xzf {bundle.name} && {request.command}'",
        ],
        "execution": "plan_only",
    }
    path = outdir / "remote_run_plan.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def execute_remote_plan(plan: Path | str, *, dry_run: bool = False) -> dict[str, object]:
    """Submit a previously generated plan; execution is opt-in and auditable."""
    payload = json.loads(Path(plan).read_text(encoding="utf-8"))
    commands: Sequence[str] = payload["commands"]
    if dry_run:
        return {"executed": False, "commands": list(commands), "status": "dry_run"}
    completed = [subprocess.run(command, shell=True, capture_output=True, text=True, check=False) for command in commands]
    return {"executed": True, "status": "ok" if all(item.returncode == 0 for item in completed) else "failed", "returncodes": [item.returncode for item in completed], "stdout": [item.stdout for item in completed], "stderr": [item.stderr for item in completed]}


__all__ = ["RemoteRunRequest", "build_remote_bundle", "execute_remote_plan", "write_remote_plan"]
