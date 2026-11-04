"""LINK adapter for the ORACLE/CV_radial exponential structural field."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np

from matrix_chem import evaluate_cv_exponential_field

from .scan import PointEvaluationResult


def add_cv_exponential_field(
    result: PointEvaluationResult,
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
) -> PointEvaluationResult:
    """Add CV energy and any available analytic derivatives to a LINK result."""
    field = evaluate_cv_exponential_field(
        atomic_numbers, coordinates_angstrom, bonded_pairs
    )
    energy = (
        None
        if result.energy_hartree is None
        else float(result.energy_hartree) + field.energy_hartree
    )
    gradient = result.gradient_hartree_per_bohr
    if gradient is not None:
        gradient = np.asarray(gradient, dtype=float).reshape(-1) + field.gradient_hartree_per_bohr
    execution = dict(result.execution)
    # Frozen composite gradients are assembled later by LINK from the
    # analytic partial stored in the execution record plus the numerical
    # SONIC terms.  Keep the analytic CV contribution in that same channel;
    # otherwise the public result contains the CV gradient but the hybrid
    # assembler silently drops it.
    partial_key = "composite_partial_analytic_gradient_hartree_per_bohr"
    if partial_key in execution:
        partial = execution[partial_key]
        partial_array = (
            np.zeros_like(field.gradient_hartree_per_bohr)
            if partial is None
            else np.asarray(partial, dtype=float).reshape(-1)
        )
        if partial_array.shape != field.gradient_hartree_per_bohr.shape:
            raise ValueError("composite analytic gradient has an invalid CV-field size")
        execution[partial_key] = (
            partial_array + field.gradient_hartree_per_bohr
        ).tolist()
        # The empirical field belongs to the analytic partial.  Keep the
        # energy-only electronic subtotal unchanged so the one-sided F12 or
        # density-based derivative cannot count the same CV force a second
        # time.
    hessian = result.hessian_hartree_per_bohr2
    if hessian is not None:
        hessian = np.asarray(hessian, dtype=float) + field.hessian_hartree_per_bohr2
    source = "+".join(item for item in (result.source, "ORACLE_CV_RADIAL_EXPONENTIAL") if item)
    return replace(
        result,
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        execution=execution,
        source=source,
    )
