from __future__ import annotations

import numpy as np

from matrix_gf import (
    diagonal_high_level_scaling,
    frequency_error_metrics,
    solve_wilson_gf,
)


def main() -> None:
    f_l0 = np.array(
        [
            [4.0, 0.60, 0.10],
            [0.60, 9.0, 0.30],
            [0.10, 0.30, 16.0],
        ]
    )
    g = np.eye(3)

    f_l1_diag = np.array([5.0, 8.0, 20.0])
    f_l1_full_reference = np.array(
        [
            [5.0, 0.70, 0.12],
            [0.70, 8.0, 0.28],
            [0.12, 0.28, 20.0],
        ]
    )

    scaled = diagonal_high_level_scaling(f_l0, f_l1_diag)
    reference = solve_wilson_gf(f_l1_full_reference, g, scale_to_cm=False)
    effective = solve_wilson_gf(scaled.effective_force_constants, g, scale_to_cm=False)
    error = frequency_error_metrics(effective.frequencies_cm, reference.frequencies_cm)

    print("scale factors:", " ".join(f"{x:.6f}" for x in scaled.factors))
    print(
        "effective diagonal:",
        " ".join(f"{x:.6f}" for x in np.diag(scaled.effective_force_constants)),
    )
    print("effective F[1,2]:", f"{scaled.effective_force_constants[0, 1]:.6f}")
    print("frequencies:", " ".join(f"{x:.6f}" for x in effective.frequencies_cm))
    print(
        "reference frequencies:",
        " ".join(f"{x:.6f}" for x in reference.frequencies_cm),
    )
    print("RMS error:", f"{error.rms_delta_cm:.6f}")


if __name__ == "__main__":
    main()
