"""Conservative FP64 refinement routing from qualified complete-call timings."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import os
import platform
import subprocess


@dataclass(frozen=True)
class RefinementDecision:
    method: str
    reason: str
    calibration: str | None = None


@lru_cache(maxsize=1)
def runtime_fingerprint():
    """Discover accelerator qualification metadata only on an explicit GPU path."""
    import numpy as np
    import torch
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip() if platform.system() == "Darwin" else platform.processor()
    except (OSError, subprocess.SubprocessError):
        brand = "unknown"
    # Thread limits can change between calls; do not cache them here.
    return {"hardware": brand, "torch": torch.__version__, "numpy": np.__version__}


def runtime_threads():
    from threadpoolctl import threadpool_info
    pools = [int(p["num_threads"]) for p in threadpool_info()
             if p.get("user_api") == "blas"]
    if pools:
        return max(pools)
    return 1 if os.environ.get("VECLIB_MAXIMUM_THREADS") == "1" else 0


@lru_cache(maxsize=1)
def qualified_calibration():
    try:
        return json.loads(files("matrix_gfn2").joinpath("data/mps_refinement_policy.json").read_text())
    except (OSError, ValueError):
        return None


def select_refinement(mode, orbitals, *, eligible, fingerprint=None,
                      host_threads=None, calibration=None, qualified_domain=True):
    """Choose using complete calls, never by relaxing electronic tolerances.

    A bound uses both adjacent qualified sizes; no extrapolation is allowed.
    Invalid/missing calibration fails closed to canonical refinement.
    """
    mode = str(mode).strip().lower()
    if mode not in {"auto", "0", "1", "canonical", "occupied"}:
        raise ValueError("MATRIX_GFN2_MPS_REFINE_OCCUPIED must be auto, 0/canonical, or 1/occupied")
    if not eligible:
        return RefinementDecision("canonical", "occupied-refinement-ineligible")
    if mode in {"0", "canonical"}:
        return RefinementDecision("canonical", "explicit-override")
    if mode in {"1", "occupied"}:
        return RefinementDecision("occupied", "explicit-override")
    if not qualified_domain:
        return RefinementDecision("canonical", "outside-qualified-domain")
    data = qualified_calibration() if calibration is None else calibration
    if not isinstance(data, dict) or data.get("schema") != "matrix.gfn2.refinement-policy.v1":
        return RefinementDecision("canonical", "no-qualified-calibration")
    actual = runtime_fingerprint() if fingerprint is None else fingerprint
    threads = runtime_threads() if host_threads is None else host_threads
    if actual != data.get("fingerprint") or threads != data.get("host_threads"):
        return RefinementDecision("canonical", "unqualified-runtime")
    if not data.get("accuracy_passed") or not data.get("nve_passed"):
        return RefinementDecision("canonical", "unqualified-numerics")
    try:
        cases = sorted(data["cases"], key=lambda c: c["orbitals"])
        lower = [c for c in cases if c["orbitals"] <= orbitals]
        upper = [c for c in cases if c["orbitals"] >= orbitals]
        if not lower or not upper:
            return RefinementDecision("canonical", "outside-qualified-range")
        bounds = (lower[-1], upper[0])
        import math
        times = [(float(c["canonical_seconds"]), float(c["occupied_seconds"])) for c in bounds]
        if any(not math.isfinite(t) or t <= 0 for pair in times for t in pair):
            raise ValueError("invalid timing")
        advantage = min(c/o for c, o in times)
    except (KeyError, TypeError, ValueError, IndexError):
        return RefinementDecision("canonical", "invalid-calibration")
    if advantage >= 1.1:
        return RefinementDecision("occupied", "qualified-complete-call-advantage", data.get("id"))
    return RefinementDecision("canonical", "canonical-cost-preferred", data.get("id"))
