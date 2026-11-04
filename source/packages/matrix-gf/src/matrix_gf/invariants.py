"""Fail-fast numerical invariants shared by Cartesian and SONIC GF/PED paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motions import cartesian_normal_modes_from_hessian


@dataclass(frozen=True)
class HessianPEDInvariantReport:
    mode_count: int
    cartesian_external_rank: int
    maximum_frequency_difference_cm: float
    ped_shape: tuple[int, int]
    maximum_ped_sum_error_percent: float


def validate_hessian_ped_invariants(
    cartesian_hessian_au: np.ndarray,
    masses_amu: np.ndarray,
    coordinates_bohr: np.ndarray,
    sonic_result: object,
    *,
    expected_mode_count: int | None = None,
    reported_cartesian_frequencies_cm: np.ndarray | None = None,
    frequency_atol_cm: float = 5.0e-2,
    ped_sum_atol_percent: float = 1.0e-6,
) -> HessianPEDInvariantReport:
    """Validate representation equivalence, dimensions, finiteness and PED completeness."""

    hessian = np.asarray(cartesian_hessian_au, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    expected_hessian_shape = (3 * masses.size, 3 * masses.size)
    if hessian.shape != expected_hessian_shape:
        raise ValueError(
            f"Cartesian Hessian shape must be {expected_hessian_shape}, got {hessian.shape}"
        )
    if not np.all(np.isfinite(hessian)):
        raise ValueError("Cartesian Hessian contains NaN or infinite values")
    if not np.allclose(hessian, hessian.T, rtol=1.0e-8, atol=1.0e-10):
        mismatch = float(np.max(np.abs(hessian - hessian.T)))
        raise ValueError(
            "Cartesian Hessian must be symmetric before GF/PED validation "
            f"(maximum mismatch {mismatch:.3e})"
        )

    cartesian = cartesian_normal_modes_from_hessian(
        hessian,
        masses,
        coordinates,
        source="Cartesian/SONIC invariant",
    )
    cartesian_frequencies = np.asarray(cartesian.frequencies_cm, dtype=float).reshape(-1)
    sonic_frequencies = np.asarray(
        getattr(sonic_result, "frequencies_cm"), dtype=float
    ).reshape(-1)
    count = cartesian_frequencies.size
    derived_expected = hessian.shape[0] - int(cartesian.external_rank)
    expected = derived_expected if expected_mode_count is None else int(expected_mode_count)
    if expected < 1:
        raise ValueError("expected vibrational mode count must be positive")
    if expected != derived_expected:
        raise ValueError(
            "declared vibrational mode count disagrees with Cartesian geometry: "
            f"declared {expected}, derived {derived_expected} from external rank "
            f"{cartesian.external_rank}"
        )
    if count != expected or sonic_frequencies.size != expected:
        raise ValueError(
            "incomplete vibrational modes: "
            f"expected {expected}, Cartesian has {count}, SONIC has {sonic_frequencies.size}"
        )

    modes = np.asarray(getattr(sonic_result, "modes_internal"), dtype=float)
    if modes.ndim != 2 or modes.shape[1] != expected:
        raise ValueError(
            "SONIC normal-mode matrix must contain one complete column per frequency"
        )
    if modes.shape[0] < expected or np.linalg.matrix_rank(modes) < expected:
        raise ValueError("SONIC normal-mode matrix does not span every vibrational mode")
    ped = np.asarray(getattr(getattr(sonic_result, "ped"), "values"), dtype=float)
    labels = tuple(getattr(getattr(sonic_result, "ped"), "labels"))
    if ped.shape != (modes.shape[0], expected):
        raise ValueError(
            f"PED shape must be {(modes.shape[0], expected)}, got {ped.shape}"
        )
    if len(labels) != ped.shape[0]:
        raise ValueError("PED label count must match its coordinate dimension")
    if np.any(ped < -float(ped_sum_atol_percent)):
        raise ValueError("PED contains a negative contribution")

    arrays = (
        np.asarray(cartesian.eigenvalues, dtype=float),
        np.asarray(cartesian.mass_weighted_modes, dtype=float),
        np.asarray(cartesian.cartesian_directions, dtype=float),
        cartesian_frequencies,
        sonic_frequencies,
        modes,
        ped,
    )
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Hessian/PED invariant contains NaN or infinite values")

    ordered_cartesian = np.sort(cartesian_frequencies)
    ordered_sonic = np.sort(sonic_frequencies)
    frequency_differences = np.abs(ordered_cartesian - ordered_sonic)
    maximum_frequency_difference = float(np.max(frequency_differences))
    if maximum_frequency_difference > float(frequency_atol_cm):
        raise ValueError(
            "Cartesian and SONIC frequencies disagree: maximum difference "
            f"{maximum_frequency_difference:.6g} cm-1"
        )

    if reported_cartesian_frequencies_cm is not None:
        reported = np.asarray(reported_cartesian_frequencies_cm, dtype=float).reshape(-1)
        if reported.size not in {0, expected}:
            raise ValueError(
                "reported Cartesian frequencies are incomplete: "
                f"expected {expected}, got {reported.size}"
            )
        if reported.size:
            if not np.all(np.isfinite(reported)):
                raise ValueError("reported Cartesian frequencies contain NaN or infinite values")
            reported_difference = float(
                np.max(np.abs(np.sort(reported) - ordered_cartesian))
            )
            if reported_difference > float(frequency_atol_cm):
                raise ValueError(
                    "reported and Hessian-derived Cartesian frequencies disagree: "
                    f"maximum difference {reported_difference:.6g} cm-1"
                )

    ped_sums = np.sum(ped, axis=0)
    nonzero_modes = np.abs(sonic_frequencies) > float(frequency_atol_cm)
    ped_errors = np.abs(ped_sums[nonzero_modes] - 100.0)
    maximum_ped_error = float(np.max(ped_errors)) if ped_errors.size else 0.0
    if maximum_ped_error > float(ped_sum_atol_percent):
        raise ValueError(
            "PED is incomplete: maximum nonzero-mode normalization error "
            f"{maximum_ped_error:.6g}%"
        )

    return HessianPEDInvariantReport(
        mode_count=expected,
        cartesian_external_rank=int(cartesian.external_rank),
        maximum_frequency_difference_cm=maximum_frequency_difference,
        ped_shape=(int(ped.shape[0]), int(ped.shape[1])),
        maximum_ped_sum_error_percent=maximum_ped_error,
    )


__all__ = ["HessianPEDInvariantReport", "validate_hessian_ped_invariants"]
