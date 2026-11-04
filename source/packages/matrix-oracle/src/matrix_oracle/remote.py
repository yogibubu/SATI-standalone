"""QM host capability manifests; discovery is explicit and side-effect free."""
from __future__ import annotations
import shutil
import subprocess
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable

from matrix_core import atomic_json_write

QM_EXECUTABLES = ("orca", "g16", "gaussian", "molpro", "mrcc", "xtb")

def local_qm_capabilities(executables: Iterable[str] = QM_EXECUTABLES) -> dict[str, object]:
    return {"schema": "matrix.oracle.qm_capabilities.v1", "host": "local",
            "engines": {name: shutil.which(name) for name in executables}}

def remote_qm_manifest(host: str, *, executables: Iterable[str] = QM_EXECUTABLES) -> dict[str, object]:
    return {"schema": "matrix.oracle.qm_capabilities.v1", "host": host,
            "engines": {name: {"probe": f"ssh {host} command -v {name}", "status": "unknown"}
                         for name in executables}}

def probe_remote_qm(host: str, *, executables: Iterable[str] = QM_EXECUTABLES,
                    timeout: float = 5.0) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9_.:@/-]+", host):
        raise ValueError("unsafe remote host")
    result = remote_qm_manifest(host, executables=executables)
    for name in executables:
        try:
            probe = subprocess.run(("ssh", "-o", "BatchMode=yes", host, "command", "-v", name),
                                   capture_output=True, text=True, timeout=timeout, check=False)
            result["engines"][name] = {"status": "available" if probe.returncode == 0 else "unavailable",
                                       "path": probe.stdout.strip() or None}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["engines"][name] = {"status": "unknown", "error": type(exc).__name__}
    return result

def write_capability_manifest(payload: dict[str, object], path: str | Path) -> Path:
    result = dict(payload)
    result["observed_at"] = datetime.now(timezone.utc).isoformat()
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(target, result)
    return target
