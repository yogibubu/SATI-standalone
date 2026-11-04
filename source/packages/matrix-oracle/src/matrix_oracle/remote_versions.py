from __future__ import annotations
import re, subprocess

def probe_remote_versions(host, names=("orca", "g16", "molpro", "mrcc", "xtb"), timeout=5.0):
    if not re.fullmatch(r"[A-Za-z0-9_.:@/-]+", host): raise ValueError("unsafe remote host")
    result={}
    for name in names:
        try:
            run=subprocess.run(("ssh", "-o", "BatchMode=yes", host, name, "--version"), capture_output=True, text=True, timeout=timeout, check=False)
            result[name]={"status":"available" if run.returncode == 0 else "unavailable", "version":run.stdout.strip()[:200]}
        except (OSError, subprocess.TimeoutExpired): result[name]={"status":"unknown"}
    return {"schema":"matrix.oracle.qm_versions.v1", "host":host, "engines":result}
