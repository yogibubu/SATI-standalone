"""Electrostatic primitives private to the EPOC boundary runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


FOUR_PI = 4.0 * math.pi


@dataclass(frozen=True)
class LaplaceTargetEvaluation:
    """Potential and Cartesian gradient generated at arbitrary targets."""

    potential: np.ndarray
    gradient: np.ndarray
    backend: str
    requested_precision: float


def _fmm_available() -> bool:
    try:
        import fmm3dpy  # noqa: F401
    except ImportError:
        return False
    return True


def laplace_potential_gradient_targets(
    sources_bohr: np.ndarray,
    charges: np.ndarray,
    targets_bohr: np.ndarray,
    *,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_sources: int = 256,
) -> LaplaceTargetEvaluation:
    """Evaluate the Laplace potential and target gradient."""

    sources = np.asarray(sources_bohr, dtype=float)
    targets = np.asarray(targets_bohr, dtype=float)
    q = np.asarray(charges, dtype=float).reshape(-1)
    if (
        sources.shape != (len(q), 3)
        or targets.ndim != 2
        or targets.shape[1:] != (3,)
        or np.any(~np.isfinite(sources))
        or np.any(~np.isfinite(targets))
        or np.any(~np.isfinite(q))
    ):
        raise ValueError("Laplace sources, charges, and targets are inconsistent")
    if backend not in {"auto", "direct", "fmm"}:
        raise ValueError("Laplace target backend must be auto, direct, or fmm")
    selected = backend
    if selected == "auto":
        selected = (
            "fmm"
            if len(sources) >= int(fmm_minimum_sources) and _fmm_available()
            else "direct"
        )
    if selected == "fmm":
        try:
            from fmm3dpy import lfmm3d
        except ImportError as exc:
            raise RuntimeError(
                "the FMM backend requires the optional matrix-epoc[fmm] dependency"
            ) from exc
        result = lfmm3d(
            eps=float(fmm_precision),
            sources=np.asfortranarray(sources.T),
            charges=np.asfortranarray(q),
            targets=np.asfortranarray(targets.T),
            pgt=2,
        )
        potential = FOUR_PI * np.asarray(result.pottarg, dtype=float).reshape(-1)
        gradient = FOUR_PI * np.asarray(result.gradtarg, dtype=float).reshape(3, -1).T
    else:
        delta = targets[:, None, :] - sources[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        if np.any(distance <= 1.0e-12):
            raise ValueError("coincident Laplace source and target")
        potential = (1.0 / distance) @ q
        gradient = -np.einsum("ts,tsd,s->td", distance**-3, delta, q)
    return LaplaceTargetEvaluation(
        potential=np.asarray(potential, dtype=float),
        gradient=np.asarray(gradient, dtype=float),
        backend=str(selected),
        requested_precision=float(fmm_precision),
    )


__all__ = ["LaplaceTargetEvaluation", "laplace_potential_gradient_targets"]
