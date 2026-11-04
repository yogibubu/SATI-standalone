"""Quality gate for whole-molecule APOC CM5 observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


APOC_CM5_QUALITY_SCHEMA = "matrix.apoc.cm5_quality.v1"


@dataclass(frozen=True)
class APOCCM5Quality:
    passed: bool
    charge_conservation_error_e: float
    quadrature_partition_error_e: float
    electron_count_error_e: float
    density_trace_error_e: float | None
    spin_population_error_e: float | None
    maximum_charge_uncertainty_e: float | None
    failure_reasons: tuple[str, ...]
    schema: str = APOC_CM5_QUALITY_SCHEMA

    def scientific_validation_record(self):
        from matrix_core import ScientificValidationRecord

        optional = (
            self.density_trace_error_e,
            self.spin_population_error_e,
            self.maximum_charge_uncertainty_e,
        )
        return ScientificValidationRecord(
            owner="APOC",
            subject="CM5_WHOLE_MOLECULE_QUALITY",
            passed=self.passed,
            metrics={
                "charge_conservation_error_e": self.charge_conservation_error_e,
                "quadrature_partition_error_e": self.quadrature_partition_error_e,
                "electron_count_error_e": self.electron_count_error_e,
                "optional_diagnostic_count": float(sum(value is not None for value in optional)),
            },
            context={"failure_reasons": list(self.failure_reasons)},
        )


def evaluate_cm5_quality(
    observables,
    atomic_numbers,
    *,
    multiplicity: int | None = None,
    density_trace_error_e: float | None = None,
    quadrature_error_e: float | None = None,
    charge_uncertainty_e: np.ndarray | None = None,
    charge_tolerance_e: float = 1.0e-6,
    quadrature_tolerance_e: float = 1.0e-4,
    electron_tolerance_e: float = 2.0e-5,
    density_tolerance_e: float = 1.0e-6,
    spin_tolerance_e: float = 1.0e-3,
    uncertainty_tolerance_e: float = 0.15,
) -> APOCCM5Quality:
    """Return one indivisible acceptance decision for the complete CM5 vector."""

    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    cm5 = np.asarray(observables.cm5_charges, dtype=float).reshape(-1)
    hirshfeld = np.asarray(observables.hirshfeld_charges, dtype=float).reshape(-1)
    if numbers.shape != cm5.shape or hirshfeld.shape != cm5.shape or cm5.size == 0:
        raise ValueError("APOC CM5 quality requires one complete atomic vector")
    molecular_charge = float(observables.charge)
    charge_error = abs(float(np.sum(cm5)) - molecular_charge)
    partition_error = (
        abs(float(np.sum(hirshfeld)) - molecular_charge)
        if quadrature_error_e is None
        else abs(float(quadrature_error_e))
    )
    expected_electrons = float(np.sum(numbers) - molecular_charge)
    electron_error = abs(float(observables.electron_count) - expected_electrons)
    density_error = (
        None if density_trace_error_e is None else abs(float(density_trace_error_e))
    )
    spin_error: float | None = None
    if multiplicity is not None:
        expected_spin = float(int(multiplicity) - 1)
        spin = getattr(observables, "spin_populations", None)
        if spin is None:
            spin_error = 0.0 if expected_spin == 0.0 else float("inf")
        else:
            spin_error = abs(abs(float(np.sum(spin))) - expected_spin)
    uncertainty: float | None = None
    if charge_uncertainty_e is not None:
        values = np.asarray(charge_uncertainty_e, dtype=float).reshape(-1)
        if values.shape != cm5.shape or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("CM5 uncertainty must be finite, non-negative, and per atom")
        uncertainty = float(np.max(values, initial=0.0))

    failures: list[str] = []
    checks = (
        (charge_error <= charge_tolerance_e, "CHARGE_CONSERVATION"),
        (partition_error <= quadrature_tolerance_e, "QUADRATURE_PARTITION"),
        (electron_error <= electron_tolerance_e, "ELECTRON_COUNT"),
        (density_error is None or density_error <= density_tolerance_e, "DENSITY_TRACE"),
        (spin_error is None or spin_error <= spin_tolerance_e, "SPIN_POPULATION"),
        (uncertainty is None or uncertainty <= uncertainty_tolerance_e, "CHARGE_UNCERTAINTY"),
    )
    failures.extend(label for accepted, label in checks if not accepted)
    return APOCCM5Quality(
        passed=not failures,
        charge_conservation_error_e=charge_error,
        quadrature_partition_error_e=partition_error,
        electron_count_error_e=electron_error,
        density_trace_error_e=density_error,
        spin_population_error_e=spin_error,
        maximum_charge_uncertainty_e=uncertainty,
        failure_reasons=tuple(failures),
    )


__all__ = ["APOC_CM5_QUALITY_SCHEMA", "APOCCM5Quality", "evaluate_cm5_quality"]
