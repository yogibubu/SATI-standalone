from __future__ import annotations

import numpy as np

from matrix_core import ScientificValidationError

from .vci import QuarticForceField


def validate_force_field(qff: QuarticForceField) -> None:
    """Validate basic QFF dimensions before VPT2/VCI."""
    freqs = np.asarray(qff.harmonic_frequencies_cm, dtype=float)
    if freqs.ndim != 1 or freqs.size < 1:
        raise ScientificValidationError("QFF must contain at least one harmonic frequency")
    if np.any(freqs <= 0.0):
        raise ScientificValidationError("QFF harmonic frequencies must be positive")
    n_modes = len(freqs)
    for label, terms, order in (("cubic", qff.cubic_cm, 3), ("quartic", qff.quartic_cm, 4)):
        for key in terms:
            if len(key) != order:
                raise ScientificValidationError(f"{label} term {key} has wrong order")
            if any(idx < 0 or idx >= n_modes for idx in key):
                raise ScientificValidationError(
                    f"{label} term {key} has mode index outside 1..{n_modes}"
                )
