"""Best-effort QM executable version manifest."""
from __future__ import annotations
import subprocess
import shutil

def qm_version_manifest(names=("orca", "g16", "molpro", "mrcc", "xtb")):
    result = {}
    for name in names:
        path = shutil.which(name)
        version = None
        if path:
            try:
                version = subprocess.run((path, "--version"), capture_output=True, text=True, timeout=3, check=False).stdout.strip()[:200]
            except (OSError, subprocess.TimeoutExpired):
                version = "unknown"
        result[name] = {"path": path, "version": version}
    return {"schema": "matrix.oracle.qm_versions.v1", "engines": result}
