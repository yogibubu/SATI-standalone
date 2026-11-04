"""TRINITY spectroscopy from backend-neutral property surfaces.

The Gaussian adapter is deliberately confined to the ingestion helper.  The
actual intensity contraction consumes only a generic fitted surface, so the same
path can be used with properties acquired from any MATRIX electronic backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRINITY_IR_INTENSITY_SCHEMA = "matrix.trinity.ir_intensity.v1"

# Gaussian's normal-coordinate convention: the Cartesian dipole derivative is
# contracted with a unit Cartesian displacement and divided by sqrt(reduced
# mass).  In those units this factor converts the squared derivative norm to
# km mol-1.  It is retained explicitly in the serialized audit record.
GAUSSIAN_IR_CONVERSION_KM_MOL = 974.8802


@dataclass(frozen=True)
class TrinityIRIntensityResult:
    """Harmonic IR intensities and the property surface used to obtain them."""

    frequencies_cm1: np.ndarray
    normal_derivatives_au: np.ndarray
    intensities_km_mol: np.ndarray
    mode_labels: tuple[str, ...]
    property_surface: Any
    reference_intensities_km_mol: np.ndarray
    metadata: Mapping[str, Any]
    schema: str = TRINITY_IR_INTENSITY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "frequencies_cm1": self.frequencies_cm1.tolist(),
            "normal_derivatives_au_per_sqrt_amu_bohr": (
                self.normal_derivatives_au.tolist()
            ),
            "intensities_km_mol": self.intensities_km_mol.tolist(),
            "mode_labels": list(self.mode_labels),
            "reference_intensities_km_mol": (
                self.reference_intensities_km_mol.tolist()
            ),
            "property_surface": self.property_surface.to_dict(),
            "metadata": dict(self.metadata),
        }


def harmonic_ir_intensities_from_property_surface(
    surface: Any,
    mode_labels: Sequence[str],
    *,
    conversion_factor: float = GAUSSIAN_IR_CONVERSION_KM_MOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract a dipole surface with normal coordinates.

    The returned derivative matrix has shape ``(dipole components, modes)``.
    A surface can therefore use any component frame; the intensity is invariant
    to a rigid rotation because it depends on the squared vector norm.
    """

    labels = tuple(str(value) for value in mode_labels)
    basis_index = {label: index for index, label in enumerate(surface.basis_labels)}
    missing = tuple(label for label in labels if label not in basis_index)
    if missing:
        raise ValueError(f"dipole surface is missing normal coordinates: {missing}")
    derivatives = np.asarray(surface.coefficients, dtype=float)[
        :, [basis_index[label] for label in labels]
    ]
    if derivatives.ndim != 2 or derivatives.shape[0] != 3:
        raise ValueError("IR intensities require a three-component dipole surface")
    intensities = float(conversion_factor) * np.einsum(
        "cm,cm->m", derivatives, derivatives
    )
    return derivatives, intensities


def dipole_surface_and_ir_from_gaussian_fchk(
    path: Path | str,
    *,
    displacement: float = 0.02,
    symmetry_tolerance: float = 1.0e-10,
) -> TrinityIRIntensityResult:
    """Build a symmetry-constrained dipole surface and harmonic IR spectrum.

    Gaussian FCHK Cartesian derivatives are transformed to mass-normalized
    harmonic coordinates.  Symmetry-forbidden equilibrium components and
    derivatives are promoted from small numerical values to exact zero
    constraints before the backend-neutral common-basis SVD fit.
    """

    from matrix_qm import (
        PropertyComponent,
        PropertySurfaceProblem,
        fit_property_surface,
    )
    from matrix_gaussian import read_gaussian_fchk

    source = Path(path)
    data = read_gaussian_fchk(source)
    mode_count = len(data.harmonic_frequencies_cm)
    coordinate_count = 3 * len(data.atomic_numbers)
    if mode_count == 0:
        raise ValueError("FCHK does not contain harmonic modes")
    if data.normal_modes.size != mode_count * coordinate_count:
        raise ValueError("FCHK normal-mode dimensions are inconsistent")
    if data.dipole_moment_au.size != 3:
        raise ValueError("FCHK does not contain a three-component dipole moment")
    if data.dipole_derivatives_au_per_bohr.size != 3 * coordinate_count:
        raise ValueError("FCHK does not contain complete Cartesian dipole derivatives")
    if data.reduced_masses_amu.size != mode_count:
        raise ValueError("FCHK does not contain one reduced mass per normal mode")

    modes = np.asarray(data.normal_modes, dtype=float).reshape(
        mode_count, coordinate_count
    )
    cartesian_derivatives = np.asarray(
        data.dipole_derivatives_au_per_bohr, dtype=float
    ).reshape(coordinate_count, 3).T
    normal_derivatives = (cartesian_derivatives @ modes.T) / np.sqrt(
        np.asarray(data.reduced_masses_amu, dtype=float)[None, :]
    )
    mode_labels = tuple(f"Q{index + 1:04d}" for index in range(mode_count))
    basis_labels = ("1", *mode_labels)

    # One equilibrium point and a central pair for each coordinate provide an
    # auditable, exactly determined linear property surface.
    point_count = 1 + 2 * mode_count
    design = np.zeros((point_count, 1 + mode_count), dtype=float)
    design[:, 0] = 1.0
    point_labels = ["equilibrium"]
    row = 1
    for index, label in enumerate(mode_labels):
        design[row, index + 1] = -float(displacement)
        design[row + 1, index + 1] = float(displacement)
        point_labels.extend((f"{label}_minus", f"{label}_plus"))
        row += 2

    equilibrium = np.asarray(data.dipole_moment_au, dtype=float)
    observations = equilibrium[:, None] + normal_derivatives @ design[:, 1:].T
    component_labels = ("x", "y", "z")
    components = []
    exact_zero_count = 0
    for component_index, component_label in enumerate(component_labels):
        fixed: dict[str, float] = {}
        if abs(equilibrium[component_index]) <= symmetry_tolerance:
            fixed["1"] = 0.0
        for mode_index, mode_label in enumerate(mode_labels):
            if abs(normal_derivatives[component_index, mode_index]) <= symmetry_tolerance:
                fixed[mode_label] = 0.0
        exact_zero_count += len(fixed)
        components.append(
            PropertyComponent(
                label=component_label,
                observations=observations[component_index],
                units="atomic_unit_dipole",
                fixed_coefficients=fixed,
                metadata={"exact_zero_source": "point-group numerical projection"},
            )
        )
    problem = PropertySurfaceProblem(
        property_name="dipole_moment",
        representation="mass_normalized_harmonic_coordinates",
        basis_labels=basis_labels,
        design_matrix=design,
        components=tuple(components),
        point_labels=tuple(point_labels),
        metadata={
            "source_fchk": str(source),
            "normal_coordinate_units": "sqrt(amu)*bohr",
            "displacement": float(displacement),
            "symmetry_tolerance": float(symmetry_tolerance),
            "exact_symmetry_zero_count": exact_zero_count,
        },
    )
    surface = fit_property_surface(problem)
    fitted_derivatives, intensities = harmonic_ir_intensities_from_property_surface(
        surface, mode_labels
    )
    reference = np.asarray(data.ir_intensities_km_mol, dtype=float)
    maximum_reference_error = (
        float(np.max(np.abs(intensities - reference)))
        if reference.size == intensities.size
        else None
    )
    return TrinityIRIntensityResult(
        frequencies_cm1=np.asarray(data.harmonic_frequencies_cm, dtype=float),
        normal_derivatives_au=fitted_derivatives,
        intensities_km_mol=intensities,
        mode_labels=mode_labels,
        property_surface=surface,
        reference_intensities_km_mol=reference,
        metadata={
            "source_fchk": str(source),
            "ir_conversion_km_mol": GAUSSIAN_IR_CONVERSION_KM_MOL,
            "maximum_reference_error_km_mol": maximum_reference_error,
            "exact_symmetry_zero_count": exact_zero_count,
        },
    )
