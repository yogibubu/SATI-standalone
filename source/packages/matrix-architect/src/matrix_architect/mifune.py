"""Strict adiabatic ARCHITECT/ZAFF evaluator adapter for MIFUNE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .model import (
    BOHR_TO_ANGSTROM,
    ZaffForceField,
    _zoom_hessian,
    evaluate_force_field,
)


class ZaffMifuneEvaluator:
    """Expose analytic ZAFF E+G without requesting a Hessian.

    The adapter instance is deliberately persistent for a complete trajectory.
    A compiled SONIC runtime is constructed once.  Current generic and seed
    runtimes do not yet expose persistent long-range state, so their MIFUNE
    capabilities fail closed for large-system production requirements.
    """

    def __init__(
        self,
        force_field: ZaffForceField,
        *,
        xyzin: Path | str | None = None,
        zoom_level: str | None = None,
    ) -> None:
        if xyzin is not None and zoom_level is not None:
            raise ValueError("a compiled SONIC runtime does not accept a zoom level")
        self.force_field = force_field
        self.xyzin = None if xyzin is None else Path(xyzin)
        self.zoom_level = zoom_level
        self.evaluation_count = 0
        self.last_execution: dict[str, Any] = {}
        self._sonic_runtime = None
        self._seed_runtime = None
        self._harmonic_hessian = None
        if self.xyzin is not None:
            from matrix_zaff.sonic_runtime import ZaffSonicRuntime

            self._sonic_runtime = ZaffSonicRuntime(force_field, self.xyzin)
        elif force_field.metadata.get("diagonal_seed_model") is not None:
            from matrix_zaff.seed_runtime import ZaffSecondOrderAnchoredSeedRuntime

            self._seed_runtime = ZaffSecondOrderAnchoredSeedRuntime(
                force_field.metadata["diagonal_seed_model"],
                force_field.reference_coordinates_angstrom,
                zoom_level=zoom_level,
            )
            self._harmonic_hessian = (
                _zoom_hessian(force_field, zoom_level)
                if force_field.zoom_levels
                else force_field.hessian_hartree_per_bohr2
            )

    @property
    def mifune_capabilities(self):
        from matrix_mifune import EvaluatorCapabilities

        return EvaluatorCapabilities(
            analytic_gradient=True,
            persistent_long_range=bool(
                self._seed_runtime is not None
                and self._seed_runtime.runtime.electrostatics.persistent_fmm
            ),
            dynamic_pair_list=self._seed_runtime is not None,
            thread_safe=False,
            backend=(
                "architect-zaff-persistent-sonic"
                if self._sonic_runtime is not None
                else "architect-zaff"
            ),
        )

    def energy_gradient(self, coordinates_bohr: np.ndarray):
        """Return a MIFUNE analytic E+G value in atomic units."""

        from matrix_mifune import AnalyticEvaluation

        coordinates = np.asarray(coordinates_bohr, dtype=float)
        if (
            coordinates.shape
            != self.force_field.reference_coordinates_angstrom.shape
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError("MIFUNE geometry is incompatible with the ZAFF field")
        coordinates_angstrom = coordinates * BOHR_TO_ANGSTROM
        if self._seed_runtime is not None:
            residual = self._seed_runtime.evaluate(coordinates_angstrom)
            displacement = (
                coordinates_angstrom
                - self.force_field.reference_coordinates_angstrom
            ).reshape(-1) / BOHR_TO_ANGSTROM
            harmonic_gradient = (
                self.force_field.gradient_reference_hartree_per_bohr
                + self._harmonic_hessian @ displacement
            )
            energy = (
                self.force_field.energy_reference_hartree
                + float(
                    self.force_field.gradient_reference_hartree_per_bohr
                    @ displacement
                )
                + 0.5
                * float(displacement @ self._harmonic_hessian @ displacement)
                + residual.energy_hartree
            )
            gradient = harmonic_gradient + residual.gradient_hartree_per_bohr
            execution = {
                **dict(residual.execution),
                "runtime": "harmonic_plus_second_order_anchored_seed",
            }
        elif self._sonic_runtime is None:
            result = evaluate_force_field(
                self.force_field,
                coordinates_angstrom,
                properties=("energy", "gradient"),
                zoom_level=self.zoom_level,
            )
            energy = result.energy_hartree
            gradient = result.gradient_hartree_per_bohr
            execution = dict(result.execution)
        else:
            result = self._sonic_runtime.evaluate(
                coordinates_angstrom,
                include_hessian=False,
            )
            energy = result.energy_hartree
            gradient = result.gradient_hartree_per_bohr
            execution = dict(result.execution)
        if energy is None or gradient is None:
            raise RuntimeError("ARCHITECT/ZAFF did not return the requested analytic E+G")
        self.evaluation_count += 1
        self.last_execution = execution
        return AnalyticEvaluation(
            float(energy),
            np.asarray(gradient, dtype=float).reshape(coordinates.shape),
        )

    def clone(self) -> ZaffMifuneEvaluator:
        """Build a separate evaluator instance for an independent replica."""

        return type(self)(
            self.force_field,
            xyzin=self.xyzin,
            zoom_level=self.zoom_level,
        )


__all__ = ["ZaffMifuneEvaluator"]
