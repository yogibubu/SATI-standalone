"""Explicit bounded audit of errors AFTER ANC and coordinate congruence.

Only validation uses the exact all-pair reference and dense small matrices.
The raw pair-tail certificate must never be reported as a conditioned SH
certificate. This audit establishes measured errors on the supplied chart.
"""
import numpy as np
from matrix_chem.structural_corrections import BOHR_TO_ANGSTROM


def audit_conditioned_hessian(path, cartesian_from_active, *, pair_tolerance,
                              relative_tolerance=1.e-3, maximum_cartesian_dimension=300):
    from matrix_oracle import build_initial_hessian
    j = np.asarray(cartesian_from_active, dtype=float)
    if j.ndim != 2 or min(j.shape) == 0 or j.shape[0] > maximum_cartesian_dimension or j.shape[1] > 3000:
        raise ValueError('exact conditioned-Hessian audit exceeds its explicit small-system budget')
    exact = build_initial_hessian(path)
    local = build_initial_hessian(path, pair_error_tolerance=pair_tolerance)
    if j.shape[0] != exact.coordinates_angstrom.size or not np.isfinite(j).all():
        raise ValueError('audit tangent has incompatible dimensions or nonfinite values')
    identity = np.eye(j.shape[0])
    hx = np.column_stack([exact.cartesian_action(v) for v in identity])
    hl = np.column_stack([local.cartesian_action(v) for v in identity])
    difference = hl-hx
    hq = j.T @ hx @ j
    dq = j.T @ difference @ j
    norm = lambda x: float(np.linalg.norm(x, 2))
    error = norm(dq)/max(norm(hq),1.e-30)
    diagonal_scale = np.maximum(np.abs(np.diag(hq)), 1.e-12)
    diagonal_error = np.abs(np.diag(dq))/diagonal_scale
    scaled_entries = np.abs(dq)/np.sqrt(diagonal_scale[:,None]*diagonal_scale[None,:])
    return dict(local.audit, audit_kind='BOUNDED_EXACT_POST_ANC_CONGRUENCE',
        measured_cartesian_error_hartree_per_bohr2=norm(difference)*BOHR_TO_ANGSTROM**2,
        measured_active_relative_spectral_error=error,
        measured_active_max_absolute_error=float(np.max(np.abs(dq))),
        measured_active_max_relative_diagonal_error=float(np.max(diagonal_error)),
        measured_active_median_relative_diagonal_error=float(np.median(diagonal_error)),
        measured_active_max_diagonal_scaled_entry_error=float(np.max(scaled_entries)),
        congruence_amplification_bound=norm(j)**2*norm(difference),
        relative_tolerance=relative_tolerance, passed=error <= relative_tolerance,
        exact_reference_audit_only=True, production_linear_scaling_certificate=False)
