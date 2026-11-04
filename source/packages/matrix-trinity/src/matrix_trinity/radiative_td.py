"""Spontaneous-emission rates from the established harmonic TD correlation.

A Gaussian convolution changes a cubic moment. The Hermite polynomial
nu**3 - 3*sigma**2*nu removes that change on the full energy axis. Restricting
this polynomial to positive photon energies gives a lower approximation to
the physical positive cubic moment; its remaining smoothing error is bounded
by sigma**3/sqrt(2*pi) times the exact positive dipole-strength sum. The bound
is independent of the positions of individual lines. Finite time/grid errors
are separate and must be checked by paired calculations.
"""
from __future__ import annotations
import numpy as np
from .vibronic_rates import EINSTEIN_A_FACTOR
from .vibronic_td import harmonic_vibronic_td


def radiative_rate_from_td_density(wavenumbers_cm, density_au2_per_cm, *,
        gaussian_sigma_cm, total_strength_au2, photon_energy_sign):
    """Contract a uniformly sampled, Gaussian-only TD dipole-strength density.

    ``photon_energy_sign`` explicitly maps transition energies to photon
    energies: +1 for the positive-photon axis returned by SATI emission, -1
    for a density expressed on negative transition energies.
    No Lorentzian broadening is supported because its cubic moment diverges.
    Negative numerical density is retained and reported, never clipped.
    The smoothing interval assumes a converged finite-grid Fourier transform;
    it is not a bound on time truncation, aliasing or electronic-model error.
    """
    if np.iscomplexobj(wavenumbers_cm) or np.iscomplexobj(density_au2_per_cm):
        raise ValueError('Real frequency axis and physical dipole density required')
    nu = np.asarray(wavenumbers_cm, float)
    density = np.asarray(density_au2_per_cm, float)
    sigma = float(gaussian_sigma_cm)
    strength = float(total_strength_au2)
    if (nu.ndim != 1 or len(nu) < 3 or density.shape != nu.shape
            or not np.all(np.isfinite(nu)) or not np.all(np.isfinite(density))
            or not np.isfinite(sigma) or sigma <= 0
            or not np.isfinite(strength) or strength < 0
            or photon_energy_sign not in (-1, 1)):
        raise ValueError('Finite uniform density, positive Gaussian width and nonnegative strength required')
    differences = np.diff(nu)
    step = abs(float(differences[0]))
    if step == 0 or not np.allclose(differences, differences[0], rtol=1e-8, atol=step*1e-8):
        raise ValueError('Uniform, strictly monotonic Fourier frequency grid required')
    photon = photon_energy_sign * nu
    positive = photon > 0
    polynomial = np.where(positive, photon**3 - 3*sigma**2*photon, 0.)
    factor = EINSTEIN_A_FACTOR * step
    corrected = float(factor * np.dot(polynomial, density))
    raw = float(factor * np.dot(np.maximum(photon, 0.)**3, density))
    smoothing = float(EINSTEIN_A_FACTOR * sigma**3 * strength / np.sqrt(2*np.pi))
    negative = float(factor * np.dot(abs(polynomial), -np.minimum(density, 0.)))
    edge = max(1, len(nu)//20)
    edges = np.r_[np.arange(edge), np.arange(len(nu)-edge, len(nu))]
    edge_moment = float(factor * np.sum(abs(polynomial[edges]*density[edges])))
    rate = max(0., corrected)
    return dict(schema='matrix.trinity.td_radiative_rate.v1', rate_s_1=rate,
        radiative_lifetime_s=None if rate == 0 else 1/rate,
        gaussian_smoothing_rate_interval_s_1=[rate, max(0., corrected+smoothing)],
        gaussian_smoothing_bias_bound_s_1=smoothing,
        corrected_cubic_rate_s_1=corrected, uncorrected_broadened_rate_s_1=raw,
        diagnostics=dict(frequency_step_cm=step, gaussian_sigma_cm=sigma,
            exact_strength_sum_au2=strength,
            integrated_density_strength_au2=float(density.sum()*step),
            negative_density_cubic_weight_s_1=negative,
            outer_five_percent_absolute_cubic_weight_s_1=edge_moment,
            physical_zero_floor_applied=corrected < 0,
            negative_density_clipped=False,
            finite_grid_convergence_requires_paired_calculations=True,
            environment='vacuum electric-dipole spontaneous emission',
            smoothing_interval_excludes_time_and_aliasing_error=True))


def harmonic_radiative_emission_rate_td(model, *, temperature_K=0.,
        gaussian_sigma_cm=50., **td_controls):
    """Reuse the analytic harmonic thermal TD service; enumerate no states."""
    if 'lorentz_hwhm_cm' in td_controls or 'kind' in td_controls:
        raise ValueError('This emission-rate service uses Gaussian-only broadening')
    td = harmonic_vibronic_td(model, temperature_K=temperature_K, kind='emission',
        gaussian_sigma_cm=gaussian_sigma_cm, lorentz_hwhm_cm=0., **td_controls)
    result = radiative_rate_from_td_density(td['wavenumbers_cm'],
        td['density_au2_per_cm'], gaussian_sigma_cm=gaussian_sigma_cm,
        total_strength_au2=td['diagnostics']['zero_time_sum_rule']['total'],
        photon_energy_sign=1)
    result['diagnostics']['TD'] = td['diagnostics']
    result['provenance'] = model.provenance
    return result


__all__ = ['radiative_rate_from_td_density', 'harmonic_radiative_emission_rate_td']
