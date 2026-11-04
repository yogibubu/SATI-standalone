"""Monte-Carlo uncertainty propagation for rovibrational observables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def propagate_observable_uncertainty(
    levels_cm: Sequence[float],
    *,
    energy_sigma_cm: float = 0.0,
    rotational_constants_MHz: Sequence[float] = (),
    rotational_sigma_MHz: Sequence[float] = (),
    samples: int = 2000,
    seed: int = 20260924,
) -> dict[str, object]:
    """Propagate independent surface uncertainty to levels/splittings/A-B-C."""
    levels = np.asarray(levels_cm, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("levels_cm must be a non-empty one-dimensional sequence")
    rng = np.random.default_rng(seed)
    level_samples = levels[None, :] + rng.normal(0.0, float(energy_sigma_cm), (int(samples), levels.size))
    splitting_samples = np.diff(level_samples, axis=1) if levels.size > 1 else np.empty((samples, 0))
    output: dict[str, object] = {
        "schema": "matrix.tunneling.uncertainty.v1",
        "samples": int(samples),
        "seed": int(seed),
        "levels_cm": _summary(level_samples),
        "splittings_cm": _summary(splitting_samples),
    }
    if rotational_constants_MHz:
        abc = np.asarray(rotational_constants_MHz, dtype=float)
        sigma = np.asarray(rotational_sigma_MHz if rotational_sigma_MHz else np.zeros_like(abc), dtype=float)
        if abc.shape != sigma.shape:
            raise ValueError("rotational sigma shape must match rotational constants")
        output["rotational_constants_MHz"] = _summary(abc[None, :] + rng.normal(0.0, sigma, (int(samples), abc.size)))
    return output


def _summary(values: np.ndarray) -> dict[str, list[float]]:
    if values.shape[1] == 0:
        return {"mean": [], "std": [], "p2_5": [], "p97_5": []}
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "p2_5": np.percentile(values, 2.5, axis=0).tolist(),
        "p97_5": np.percentile(values, 97.5, axis=0).tolist(),
    }


def write_uncertainty_report(path: Path | str, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = ["propagate_observable_uncertainty", "write_uncertainty_report"]
