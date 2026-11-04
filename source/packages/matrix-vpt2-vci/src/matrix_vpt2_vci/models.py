from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnharmonicInput:
    """ORACLE canonical normal-coordinate anharmonic input."""

    harmonic_frequencies_cm: np.ndarray
    anharmonic_frequencies_cm: np.ndarray
    cubic_cm: dict[tuple[int, int, int], float]
    quartic_cm: dict[tuple[int, int, int, int], float]
    source: str = "oracle"

    def validate(self) -> None:
        if self.harmonic_frequencies_cm.ndim != 1:
            raise ValueError("Harmonic frequencies must be a one-dimensional array")
        if self.anharmonic_frequencies_cm.ndim != 1:
            raise ValueError("Anharmonic frequencies must be a one-dimensional array")
