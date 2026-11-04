"""Controlled reduction of coordinate-dependent quadratic LAM kinetics.

Averages act on the complete quadratic form in (p, J), in the coefficient
reference frame and the flat primitive measure used by LOCK. Averaging F,
rho and R independently does not average that Hamiltonian.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np


def reduce_lam_kinetic_dependence(surface, *, retained_fraction=0.0, weights=None):
    """Interpolate the full quadratic metric between its mean and input field.

    F, -F rho.T and R + rho F rho.T are averaged together. The Schur
    complement reconstructs R. Convexity preserves a positive quadratic form.
    PES, scalar geometric completion, higher-order terms and spin/dipole
    fields are retained. This is a controlled approximation, not new QC data.
    Nonuniform weights must be supplied in the common coefficient frame;
    users remain responsible for their invariance under molecular symmetry.
    """
    fraction = float(retained_fraction)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('retained_fraction must be finite and in [0, 1]')
    size, d = int(np.prod(surface.grid_shape)), surface.dimension
    if weights is None:
        probability = np.full(size, 1.0 / size)
    else:
        probability = np.asarray(weights, dtype=float)
        if probability.shape != surface.grid_shape:
            raise ValueError('kinetic averaging weights must have the LAM grid shape')
        probability = probability.reshape(-1)
        if (not np.all(np.isfinite(probability)) or np.any(probability < 0)
                or not np.isfinite(probability.sum()) or probability.sum() <= 0):
            raise ValueError('kinetic averaging weights must be finite, nonnegative and nonzero')
        probability = probability / probability.sum()
    f = np.asarray(surface.F_MHz).reshape(size, d, d)
    rho = np.asarray(surface.rho).reshape(size, 3, d)
    r = np.asarray(surface.R_MHz).reshape(size, 3, 3)
    mixed = -f @ rho.swapaxes(-1, -2)
    rotation = r + rho @ f @ rho.swapaxes(-1, -2)
    def blend(field):
        mean = np.einsum('q,qij->ij', probability, field)
        return fraction * field + (1 - fraction) * mean
    mean_f, mean_mixed, mean_rotation = map(blend, (f, mixed, rotation))
    mean_rho = -np.linalg.solve(mean_f, mean_mixed).swapaxes(-1, -2)
    mean_r = mean_rotation - mean_rho @ mean_f @ mean_rho.swapaxes(-1, -2)
    mean_r = .5 * (mean_r + mean_r.swapaxes(-1, -2))
    if np.any(np.linalg.eigvalsh(mean_f) <= 0) or np.any(np.linalg.eigvalsh(mean_r) <= 0):
        raise ValueError('quadratic kinetic reduction is not positive definite at working precision')
    metadata = dict(surface.metadata or {})
    metadata['kinetic_reduction'] = dict(
        schema='matrix.lock.kinetic_reduction.v1', retained_fraction=fraction,
        averaging='joint quadratic coefficients in the reference frame',
        measure='uniform primitive flat measure' if weights is None else 'explicit normalized weights',
        geometric_scalar='retained unchanged', higher_order_fields='retained unchanged')
    return replace(surface, F_MHz=mean_f.reshape(surface.F_MHz.shape),
        rho=mean_rho.reshape(surface.rho.shape), R_MHz=mean_r.reshape(surface.R_MHz.shape),
        metadata=metadata)


def compare_lam_kinetic_models(surface, request, *, frequency_tolerance_MHz,
                               retained_fraction=0.0, weights=None):
    """Compare assigned frequencies at fixed coefficients with state tracking.

    Supply a materialized surface and a request with no free variables. The
    tolerance is an explicit approximation budget, not experimental accuracy.
    This comparison does not certify basis convergence, refit constants, or
    select the omitted vibrational modes. Existing truncation/overlap failures
    propagate rather than producing a favourable comparison.
    """
    from .lam_qualification import _evaluate
    tolerance = float(frequency_tolerance_MHz)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('frequency_tolerance_MHz must be finite and positive')
    if request.get('variables') or request.get('predicates'):
        raise ValueError('materialize the physical coefficients before comparing kinetic models')
    from .lam_checkpoint import copy_workspace_request
    flat = copy_workspace_request(request)
    options = flat.setdefault('options', {})
    for key in ('basis_convergence', 'optimizer_state', '_workspace_resume_state',
                'resume_workspace', 'workspace_checkpoint_path', 'checkpoint_path',
                'resume_checkpoint', 'linearization_path', 'evaluation_log_path',
                'kinetic_comparison'):
        options.pop(key, None)
    full = _evaluate(surface, flat)
    reduced_surface = reduce_lam_kinetic_dependence(surface,
        retained_fraction=retained_fraction, weights=weights)
    reduced = _evaluate(reduced_surface, flat,
        reference=(full['solved'], surface, full['context']))
    difference = reduced['frequency'] - full['frequency']
    maximum = float(np.max(np.abs(difference), initial=0.0))
    return dict(schema='matrix.lock.kinetic_comparison.v1',
        retained_fraction=float(retained_fraction), frequency_tolerance_MHz=tolerance,
        full_frequencies_MHz=full['frequency'].tolist(),
        reduced_frequencies_MHz=reduced['frequency'].tolist(),
        reduced_minus_full_MHz=difference.tolist(), maximum_frequency_change_MHz=maximum,
        rms_frequency_change_MHz=float(np.sqrt(np.mean(difference**2))),
        within_approximation_budget=maximum <= tolerance,
        basis_convergence_certified=False, refitted=False,
        scope='fixed-coefficient kinetic approximation; assigned states tracked by overlap')


__all__ = ['reduce_lam_kinetic_dependence', 'compare_lam_kinetic_models']
