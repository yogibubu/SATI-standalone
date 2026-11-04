"""GFN-FF-derived torsional priors compiled onto immutable synthon types."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from typing import Sequence

import numpy as np

from .gfnff_nonbonded import GFNFFParameterBundle, load_gfnff_parameter_bundle
from .levels import zaff_synthon_type_thresholds
from .periodic_parameters import (
    extend_periodic_atomic_series,
    zaff_periodic_element_prior,
)
from .synthon_nonbonded import SynthonDescriptor


ZAFF_SYNTHON_TORSION_SCHEMA = "matrix.zaff.synthon_torsion.v1"
BOHR_TO_ANGSTROM = 0.529177210903
GFNFF_TORSION_DAMPING_SCALE = 0.505


def _gfnff_reference_torsion_damping(
    numbers: Sequence[int],
    coordinates_angstrom: Sequence[Sequence[float]] | np.ndarray,
    bundle: GFNFFParameterBundle,
) -> float:
    """Return xTB 6.7.1's three-bond torsion damping at a rigid reference."""

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (4, 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("torsion reference coordinates must have shape (4, 3)")
    xyz_bohr = xyz / BOHR_TO_ANGSTROM
    radii = _extended_covalent_radii()
    damping = 1.0
    for index in range(3):
        distance_squared = float(np.sum((xyz_bohr[index] - xyz_bohr[index + 1]) ** 2))
        radius_sum = float(
            radii[numbers[index] - 1]
            + radii[numbers[index + 1] - 1]
        )
        cutoff_squared = GFNFF_TORSION_DAMPING_SCALE * radius_sum**2
        ratio = (distance_squared / cutoff_squared) ** 2
        damping *= 1.0 / (1.0 + ratio)
    return float(damping)


def synthon_type_key(
    descriptor: SynthonDescriptor,
    thresholds: Sequence[float] | None = None,
) -> str:
    """Return a stable discrete key from the complete intrinsic synthon."""

    scale = np.asarray(
        zaff_synthon_type_thresholds() if thresholds is None else thresholds,
        dtype=float,
    )
    values = np.asarray(descriptor.values, dtype=float)
    if scale.shape != values.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("synthon torsion thresholds do not match the descriptor")
    bins = np.rint(values / scale).astype(np.int64).tolist()
    encoded = json.dumps(
        [int(descriptor.atomic_number), *bins],
        separators=(",", ":"),
    ).encode("ascii")
    return f"S{int(descriptor.atomic_number)}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def compile_gfnff_synthon_torsion(
    descriptors: Sequence[SynthonDescriptor],
    atoms: Sequence[int],
    *,
    central_bond_order: float,
    reference_coordinates_angstrom: Sequence[Sequence[float]] | np.ndarray,
    ring_size: int = 0,
    parameter_bundle: GFNFFParameterBundle | None = None,
) -> dict[str, object] | None:
    """Compile one GFN-FF torsional prior into a four-synthon lookup record.

    GFN-FF's charge, coordination and element factors are evaluated once from
    the complete intrinsic descriptors.  The serialized runtime parameter is
    thereafter selected only by the four fixed synthon keys; no geometry- or
    noncovalent-contact-dependent retyping or charge/CN rescaling is allowed.
    """

    indices = tuple(int(value) for value in atoms)
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("a synthon torsion requires four distinct atoms")
    if min(indices) < 0 or max(indices) >= len(descriptors):
        raise ValueError("synthon torsion atom index is out of range")
    selected = tuple(descriptors[index] for index in indices)
    bundle = load_gfnff_parameter_bundle() if parameter_bundle is None else parameter_bundle
    numbers = tuple(item.atomic_number for item in selected)
    if min(numbers) < 1 or max(numbers) > 118:
        raise ValueError("ZAFF torsional priors require real elements Z=1--118")
    central_parameters, terminal_parameters = _extended_torsion_parameters()

    left, central_left, central_right, right = selected
    z_left, z_central_left, z_central_right, z_right = numbers
    q_central_left = float(central_left.values[0])
    q_central_right = float(central_right.values[0])
    coordination_left = max(float(left.values[1]), 1.0e-8)
    coordination_right = max(float(right.values[1]), 1.0e-8)
    pi_left = float(left.values[9])
    pi_central_left = float(central_left.values[9])
    pi_central_right = float(central_right.values[9])
    pi_right = float(right.values[9])

    central_factor = float(central_parameters[z_central_left - 1]) * float(
        central_parameters[z_central_right - 1]
    )
    terminal_factor = float(terminal_parameters[z_left - 1]) * float(
        terminal_parameters[z_right - 1]
    )
    if z_left == 7 and pi_left < 0.20:
        terminal_factor *= 0.5
    if z_right == 7 and pi_right < 0.20:
        terminal_factor *= 0.5
    terminal_factor *= (coordination_left * coordination_right) ** -0.14
    charge_factor = 1.0 + 12.0 * abs(q_central_left * q_central_right)
    reference_damping = _gfnff_reference_torsion_damping(
        numbers,
        reference_coordinates_angstrom,
        bundle,
    )

    pi_bond = float(np.clip(float(central_bond_order) - 1.0, 0.0, 1.0))
    pi_factor = 0.0
    if pi_bond > 0.0:
        pi_factor = pi_bond * math.exp(-2.5 * (1.24 - pi_bond) ** 14)

    ring = int(ring_size)
    if ring < 0:
        raise ValueError("ring size cannot be negative")
    sp3_left = pi_central_left < 0.20 and float(central_left.values[2]) >= 3.5
    sp3_right = pi_central_right < 0.20 and float(central_right.values[2]) >= 3.5
    sigma_factor = 1.0
    phi0 = math.pi
    periodicity = 3 if sp3_left and sp3_right else 1
    if pi_bond > 0.0:
        periodicity = 2
        sigma_factor = 0.55
    if ring in {3, 4, 5, 6} and pi_bond == 0.0:
        periodicity, phi0, sigma_factor = {
            3: (1, 0.0, 0.3),
            4: (6, math.pi / 6.0, 1.0),
            5: (6, math.pi / 6.0, 1.5),
            6: (3, math.pi / 3.0, 5.7),
        }[ring]

    source_amplitude = (
        sigma_factor + 10.0 * 1.18 * pi_factor
    ) * charge_factor * central_factor * terminal_factor
    amplitude = source_amplitude * reference_damping
    terms: list[dict[str, float | int]] = []
    if abs(amplitude) >= 1.0e-3:
        terms.append(
            {
                "amplitude_hartree": float(amplitude),
                "periodicity": int(periodicity),
                "phase_radian": float(
                    (periodicity * phi0 + math.pi) % (2.0 * math.pi)
                ),
            }
        )

    all_sp3 = sp3_left and sp3_right and pi_left < 0.20 and pi_right < 0.20
    if all_sp3 and ring == 0 and pi_bond < 0.35:
        extra = -0.90
        if 7 in (z_central_left, z_central_right):
            extra = 0.70
        if 8 in (z_central_left, z_central_right):
            extra = -2.00
        extra_source_amplitude = extra * charge_factor * central_factor * terminal_factor
        extra_amplitude = extra_source_amplitude * reference_damping
        if abs(extra_amplitude) >= 1.0e-3:
            terms.append(
                {
                    "amplitude_hartree": float(extra_amplitude),
                    "periodicity": 1,
                    "phase_radian": 0.0,
                }
            )
    periodic_fallback = False
    if not terms:
        periodic_fallback = True
        first_prior = zaff_periodic_element_prior(z_central_left)
        second_prior = zaff_periodic_element_prior(z_central_right)
        periodic_depth = math.sqrt(
            first_prior.vdw_well_depth_kcal_per_mol
            * second_prior.vdw_well_depth_kcal_per_mol
        ) / 627.5094740631
        amplitude = max(0.10 * periodic_depth * reference_damping, 1.0e-8)
        terms.append(
            {
                "amplitude_hartree": float(amplitude),
                "periodicity": int(periodicity),
                "phase_radian": float(
                    (periodicity * phi0 + math.pi) % (2.0 * math.pi)
                ),
            }
        )

    return {
        "schema": ZAFF_SYNTHON_TORSION_SCHEMA,
        "atoms": list(indices),
        "synthon_types": [synthon_type_key(item) for item in selected],
        "terms": terms,
        "source": (
            "ZAFF complete-periodic-table torsional prior for a vanishing native GFN-FF amplitude"
            if periodic_fallback
            else (
                "xTB 6.7.1 GFN-FF torsional construction compiled onto complete ZAFF synthons"
                if max(numbers) <= len(bundle.torsion_central)
                else "ZAFF periodic-table extension of the GFN-FF-analogous torsional construction"
            )
        ),
        "runtime_parameterization": "FOUR_SYNTHON_LOOKUP_NO_SEPARATE_CHARGE_OR_COORDINATION_SCALING",
        "central_bond_order": float(central_bond_order),
        "ring_size": ring,
        "source_components": {
            "charge_factor": float(charge_factor),
            "central_atomic_factor": float(central_factor),
            "terminal_atomic_and_coordination_factor": float(terminal_factor),
            "pi_factor": float(pi_factor),
            "gfnff_reference_radial_damping": float(reference_damping),
            "undamped_primary_amplitude_hartree": float(source_amplitude),
        },
    }


@lru_cache(maxsize=1)
def _extended_covalent_radii() -> np.ndarray:
    values = extend_periodic_atomic_series(
        load_gfnff_parameter_bundle().covalent_radii_bohr
    )
    values.setflags(write=False)
    return values


@lru_cache(maxsize=1)
def _extended_torsion_parameters() -> tuple[np.ndarray, np.ndarray]:
    bundle = load_gfnff_parameter_bundle()
    central = extend_periodic_atomic_series(bundle.torsion_central, positive=False)
    terminal = extend_periodic_atomic_series(bundle.torsion_terminal, positive=False)
    central.setflags(write=False)
    terminal.setflags(write=False)
    return central, terminal


__all__ = [
    "ZAFF_SYNTHON_TORSION_SCHEMA",
    "compile_gfnff_synthon_torsion",
    "synthon_type_key",
]
