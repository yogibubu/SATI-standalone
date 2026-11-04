"""Backend-independent real Dyson orbitals in APOC's AO convention."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import ElectronicState

APOC_DYSON_SCHEMA = "matrix.apoc.dyson.v1"


@dataclass(frozen=True)
class DysonOrbital:
    """A signed, unnormalised Dyson vector bound to its AO electronic state.

    ``pole_strength`` is the squared norm, not an additional amplitude factor.
    ``ionization_energy_ev`` is positive for electron removal. The producer's
    phase is retained; following a channel across geometries is a separate step.
    """

    state: ElectronicState
    coefficients_ao: np.ndarray
    pole_strength: float
    ionization_energy_ev: float
    channel: str
    method: str
    source: str
    spin_channel: int = 0
    norm_tolerance: float = 1e-7

    def __post_init__(self):
        raw = np.asarray(self.coefficients_ao)
        if np.iscomplexobj(raw) or raw.shape != (self.state.nao,):
            raise ValueError("Dyson requires a real vector in the state's AO space")
        coefficients = np.array(raw, dtype=float, copy=True)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Dyson coefficients must be finite")
        if self.spin_channel not in range(len(self.state.mo_coefficients)):
            raise ValueError("Invalid Dyson spin channel")
        values = (self.pole_strength, self.ionization_energy_ev, self.norm_tolerance)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Pole strength, removal energy and tolerance must be positive")
        norm = float(coefficients @ self.state.overlap_ao @ coefficients)
        if abs(norm - self.pole_strength) > self.norm_tolerance:
            raise ValueError(f"Dyson AO norm {norm:.12g} differs from pole strength {self.pole_strength:.12g}")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients_ao", coefficients)

    @property
    def norm_squared(self) -> float:
        return float(self.coefficients_ao @ self.state.overlap_ao @ self.coefficients_ao)

    @classmethod
    def from_mo(cls, state, coefficients_mo, *, spin_channel=0, **metadata):
        """Transform a complete canonical-MO vector without renormalising it."""
        if spin_channel not in range(len(state.mo_coefficients)):
            raise ValueError("Invalid Dyson spin channel")
        coefficients = np.asarray(coefficients_mo)
        orbitals = state.mo_coefficients[spin_channel]
        if coefficients.shape != (orbitals.shape[1],) or np.iscomplexobj(coefficients):
            raise ValueError("Dyson MO vector must cover the complete real MO space")
        return cls(state, orbitals @ coefficients, spin_channel=spin_channel, **metadata)
