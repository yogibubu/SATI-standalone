"""Unified contracted LOCK workflow for LAM, rotation, spin and dc fields."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .external_fields import (
    ExternalFieldResult,
    external_field_model_from_spin_result,
    solve_external_fields,
)
from .lam_rotation import LAMRotationResult, LAMRotationSurface, solve_lam_rotation
from .native_rotor import NativeRotorParameters
from .spin_hyperfine import (
    SpinHyperfineModel,
    SpinHyperfineResult,
    solve_spin_hyperfine_lam_rotation,
    spin_hyperfine_lam_reduced_dipole_matrices,
)


UNIFIED_LOCK_RESULT_SCHEMA = "matrix.lock.unified_result.v1"
AU_DIPOLE_TO_DEBYE = 2.541746473


@dataclass(frozen=True)
class UnifiedLOCKResult:
    lam_rotation: LAMRotationResult
    spin_hyperfine: SpinHyperfineResult
    external_fields: ExternalFieldResult
    contraction: dict[str, Any]


def solve_unified_lock(
    surface: LAMRotationSurface,
    rotor: NativeRotorParameters,
    spin_model: SpinHyperfineModel,
    *,
    n_values: Sequence[int] | range,
    lam_state_count: int = 12,
    levels_per_n: int | None = None,
    levels_per_f: int | None = None,
    dipole_au: Sequence[float] | None = None,
    electric_field_V_cm: float = 0.0,
    magnetic_field_T: float = 0.0,
    electron_g: float = 2.00231930436,
    assembly_backend: str | None = None,
    diagonalizer_backend: str | None = None,
    spin_matrix_free: bool | None = None,
    workers: int = 1,
) -> UnifiedLOCKResult:
    """Solve the retained LAM--rotation--spin--hyperfine--field Hamiltonian.

    The field-free contraction retains every requested eigenvector before the
    exact reduced-moment transformation into M_F blocks.  Therefore the only
    approximation introduced by the staged diagonalization is the explicit
    ``levels_per_n`` and LAM-state truncation recorded in the result.
    """
    values = tuple(sorted({int(value) for value in n_values}))
    if not values or values[0] < 0:
        raise ValueError("unified LOCK requires nonnegative N values")
    lam = solve_lam_rotation(
        surface,
        j_values=values,
        lam_state_count=lam_state_count,
        representation=rotor.representation,
        centrifugal_distortion=rotor,
        level_count=levels_per_n,
        assembly_backend=assembly_backend,
        diagonalizer_backend=diagonalizer_backend,
    )
    spin = solve_spin_hyperfine_lam_rotation(
        surface,
        lam,
        spin_model,
        levels_per_f=levels_per_f,
        assembly_backend=assembly_backend,
        diagonalizer_backend=diagonalizer_backend,
        matrix_free=spin_matrix_free,
        workers=workers,
    )
    reduced_au = spin_hyperfine_lam_reduced_dipole_matrices(
        surface, lam, spin, dipole_au=dipole_au
    )
    reduced_debye = {
        key: np.asarray(value) * AU_DIPOLE_TO_DEBYE for key, value in reduced_au.items()
    }
    field_model = external_field_model_from_spin_result(
        spin,
        electron_g=electron_g,
        reduced_dipole_matrices=reduced_debye,
    )
    fields = solve_external_fields(
        field_model,
        electric_field_V_cm=electric_field_V_cm,
        magnetic_field_T=magnetic_field_T,
    )
    contraction = {
        "n_values": list(values),
        "lam_state_count": len(lam.basis.energies_MHz),
        "levels_per_n": levels_per_n,
        "levels_per_f": levels_per_f,
        "spin_matrix_free": spin_matrix_free,
        "workers": workers,
        "field_free_state_count": len(field_model.states),
        "m_f_block_count": len(fields.blocks),
        "dipole_source": "surface" if surface.dipole_au is not None else "constant",
        "dipole_conversion": "atomic unit to debye",
        "retention_policy": "all requested eigenvectors" if levels_per_n is None else "lowest per N",
    }
    return UnifiedLOCKResult(lam, spin, fields, contraction)


def write_unified_lock_result(path: Path | str, result: UnifiedLOCKResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": UNIFIED_LOCK_RESULT_SCHEMA,
        "contraction": result.contraction,
        "diagnostics": {
            "lam_hermiticity_error_MHz": result.lam_rotation.hermiticity_error_MHz,
            "spin_hermiticity_error_MHz": result.spin_hyperfine.maximum_hermiticity_error_MHz,
            "spin_maximum_scaled_residual": result.spin_hyperfine.maximum_scaled_residual,
            "field_hermiticity_error_MHz": max(
                (block.hermiticity_error_MHz for block in result.external_fields.blocks),
                default=0.0,
            ),
        },
        "fields": {
            "electric_field_V_cm": result.external_fields.electric_field_V_cm,
            "magnetic_field_T": result.external_fields.magnetic_field_T,
        },
        "blocks": [
            {
                "two_M": block.two_M,
                "levels": [
                    {
                        "energy_MHz": float(energy),
                        "parent_state": parent,
                        "parent_label": result.external_fields.model.states[parent].label,
                        "parent_assignment": result.external_fields.model.states[parent].assignment,
                        "parent_purity": purity,
                    }
                    for energy, parent, purity in zip(
                        block.energies_MHz,
                        block.parent_states,
                        block.parent_purities,
                        strict=True,
                    )
                ],
            }
            for block in result.external_fields.blocks
        ],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = [
    "UNIFIED_LOCK_RESULT_SCHEMA",
    "AU_DIPOLE_TO_DEBYE",
    "UnifiedLOCKResult",
    "solve_unified_lock",
    "write_unified_lock_result",
]
