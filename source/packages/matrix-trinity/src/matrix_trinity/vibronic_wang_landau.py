"""Wang--Landau Golden Rule rates for diagonal-anharmonic energy shells.

The sampler supplies the anharmonic energies and density of states. Transition
strengths default to harmonic two-surface FC/HT recurrences, retaining
Duschinsky rotation and displacement; a state-resolved callback can supply
strengths from another wavefunction model. The default is an explicitly hybrid
approximation, not an anharmonic wavefunction calculation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import replace
from scipy.special import voigt_profile
from scipy.stats import t as student_t
from scipy.sparse.csgraph import connected_components

from .vibronic_harmonic import harmonic_gaussian_generator, orient_harmonic_model
from .vibronic_ti import _integral_dag
from .vibronic_rates import GOLDEN_RULE_FACTOR


def diagonal_anharmonicity_from_assigned_levels(states, energies_cm, fundamentals_cm):
    """Fit a quadratic occupation-energy matrix from assigned 0/1/2-quantum levels.

    Energies must share one zero and ``states`` must include the ground state,
    every fundamental, every overtone, and every two-mode combination. The
    convention matches ``matrix_sentinel.quantum_rewl``:

    ``E(n) = fundamentals @ n + diag(X) @ n(n-1) + 0.5*n@(X-diag(X))@n``.

    For resonant or mixed levels, the assigned eigenenergies produce an
    effective diagonal model; they do not retain the corresponding polyad
    eigenvectors or off-diagonal mixing at higher energy.
    """
    occupations = np.asarray(states, dtype=np.int64)
    energies = np.asarray(energies_cm, float)
    fundamentals = np.asarray(fundamentals_cm, float)
    if fundamentals.ndim != 1 or not len(fundamentals):
        raise ValueError("positive one-dimensional fundamentals required")
    nmode = len(fundamentals)
    if (occupations.ndim != 2
            or occupations.shape != (len(energies), nmode)
            or len(energies) == 0 or np.any(occupations < 0)
            or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(fundamentals))
            or np.any(fundamentals <= 0)):
        raise ValueError("finite assigned levels and positive fundamentals with compatible dimensions required")
    lookup = {tuple(row): float(energy) for row, energy in zip(occupations, energies, strict=True)}
    if len(lookup) != len(energies):
        raise ValueError("assigned occupation states must be unique")
    ground = (0,) * nmode
    if ground not in lookup:
        raise ValueError("assigned levels must include the vibrational ground state")
    e0 = lookup[ground]
    x = np.zeros((nmode, nmode), float)
    missing = []
    for i in range(nmode):
        fundamental_state = tuple(int(k == i) for k in range(nmode))
        overtone_state = tuple(2 * int(k == i) for k in range(nmode))
        if fundamental_state not in lookup:
            missing.append(f"fundamental {i}")
        if overtone_state not in lookup:
            missing.append(f"overtone {i}")
        if fundamental_state in lookup and overtone_state in lookup:
            x[i, i] = 0.5 * (lookup[overtone_state] - e0 - 2 * fundamentals[i])
        for j in range(i):
            combination = tuple(int(k == i or k == j) for k in range(nmode))
            if combination not in lookup:
                missing.append(f"combination {i},{j}")
            elif fundamental_state in lookup and tuple(int(k == j) for k in range(nmode)) in lookup:
                x[i, j] = x[j, i] = (
                    lookup[combination] - e0 - fundamentals[i] - fundamentals[j]
                )
    if missing:
        raise ValueError("assigned level set is missing " + ", ".join(missing[:8]))
    return x


def sam_vpt2_transition_strength_callback(
    model,
    initial_field,
    final_field,
    coupling_hartree,
    *,
    gradient_hartree=None,
    momentum_hartree=None,
    channel_weights=None,
    kind="emission",
    operator="FC",
    ti_controls=None,
):
    """Adapt SATI's explicit-state SAM VPT2 strengths to the Wang--Landau API.

    The callback accepts sampled final-surface occupation rows and returns the
    positive squared corrected coupling for each unique state. Resonant or
    high-quanta states outside the nonresonant SAM validity limits raise the
    original SATI error; no harmonic substitution or rate extrapolation occurs.
    """
    nmode = len(model.initial_frequencies_cm)
    constant = np.asarray(coupling_hartree, complex)
    gradient = np.zeros((3, nmode), complex) if gradient_hartree is None else np.asarray(gradient_hartree, complex)
    momentum = np.zeros((3, nmode), complex) if momentum_hartree is None else np.asarray(momentum_hartree, complex)
    weights = np.ones(3) if channel_weights is None else np.asarray(channel_weights, float)
    if (constant.shape != (3,) or gradient.shape != (3, nmode) or momentum.shape != (3, nmode)
            or weights.shape != (3,) or not all(np.all(np.isfinite(value)) for value in (constant, gradient, momentum, weights))
            or np.any(weights < 0)):
        raise ValueError("finite three-channel couplings and nonnegative channel weights required")
    scale = np.sqrt(weights)
    coupled_model = replace(
        model,
        transition_dipole_au=constant * scale,
        transition_gradient_dimensionless_au=gradient * scale[:, None],
        transition_momentum_dimensionless_au=momentum * scale[:, None],
    )
    controls = {} if ti_controls is None else dict(ti_controls)
    reserved = {"final_states", "kind", "operator", "initial_states", "initial_populations"}
    if reserved.intersection(controls):
        raise ValueError("state list, orientation, operator, and initial vacuum are controlled by the callback")

    def callback(states):
        occupations = np.asarray(states, dtype=np.int64)
        if occupations.ndim != 2 or occupations.shape[1] != nmode or np.any(occupations < 0):
            raise ValueError("sampled final states must be nonnegative occupation rows matching the model")
        unique, inverse = np.unique(occupations, axis=0, return_inverse=True)
        if not len(unique):
            return np.zeros(0)
        from .vibronic_vpt2 import vpt2_vibronic_ti

        result = vpt2_vibronic_ti(
            coupled_model,
            initial_field,
            final_field,
            final_states=unique,
            kind=kind,
            operator=operator,
            **controls,
        )
        return np.asarray(result["positive_completed_strengths_au2"], float)[inverse]

    return callback


def wang_landau_nonradiative_rate(
    model,
    fundamentals_cm,
    x_matrix_cm,
    coupling_hartree,
    *,
    kind="emission",
    gaussian_sigma_cm=21.23,
    lorentz_hwhm_cm=0.0,
    channel_weights=None,
    maximum_integrals=500000,
    harmonic_block_tolerance=0.0,
    profile_tail_relative_tolerance=0.0,
    sampling_profile_floor=1.0,
    rate_relative_standard_error_tolerance=0.1,
    profile_tail_rate_bound_tolerance=0.01,
    inverse_temperatures=(1.0,),
    sampler_controls=None,
    vibronic_origin_cm=None,
    transition_strength_callback=None,
    sampling_method="wang_landau",
    importance_controls=None,
):
    """Estimate a broadened FGR rate with a frozen Wang--Landau measurement.

    ``fundamentals_cm`` and ``x_matrix_cm`` define the diagonal-anharmonic
    occupation-energy model used by the Wang--Landau DOS. The supplied
    ``model`` defines harmonic Duschinsky/FC/HT transition amplitudes unless
    ``transition_strength_callback`` supplies state-resolved squared couplings
    in Hartree squared. At zero temperature the electronic initial state is its vibrational ground state;
    each WL sample is a final-surface occupation. Finite-temperature initial
    populations are not implemented by this estimator.

    The returned rate integrates the product of the DOS, the microcanonical
    mean transition strength, and a Voigt energy-conservation profile. The
    energy-bin width must resolve the requested linewidth. Statistical and
    model diagnostics from the sampler are returned without claiming full
    anharmonic transition wavefunctions.

    ``harmonic_block_tolerance`` optionally truncates off-diagonal elements
    of the final-state Gaussian recurrence matrix before factoring independent
    mode blocks. Zero is exact. A positive value is an approximation; its
    removed matrix norm is reported and the rate must be checked for cutoff
    convergence.
    For built-in FC strengths, states with Voigt profile below
    ``profile_tail_relative_tolerance`` times its maximum are skipped. Their
    conservative omitted-rate bound is reported using the FC overlap bound
    ``|FC|^2 <= 1``. Set this tolerance to zero to evaluate every sampled state.
    ``sampling_profile_floor`` sets the minimum frozen-stage energy weight as
    a fraction of the peak Voigt profile; one selects uniform energy sampling.
    """
    from matrix_sentinel.quantum_rewl import sample_integer_windowed_wang_landau

    if sampling_method not in {"wang_landau", "occupation_importance"}:
        raise ValueError("sampling_method must be wang_landau or occupation_importance")
    if sampling_method == "occupation_importance" and profile_tail_relative_tolerance != 0:
        raise ValueError("occupation importance sampling requires the complete line profile")

    nu = np.asarray(fundamentals_cm, float)
    x = np.asarray(x_matrix_cm, float)
    coupling = np.asarray(coupling_hartree, complex)
    weights = np.ones(3) if channel_weights is None else np.asarray(channel_weights, float)
    if nu.ndim != 1 or not len(nu) or x.shape != (len(nu), len(nu)):
        raise ValueError("fundamentals and a square anharmonicity matrix are required")
    if coupling.shape != (3,) or weights.shape != (3,) or not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("three finite SOC channels and nonnegative channel weights are required")
    if not np.all(np.isfinite(nu)) or np.any(nu <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("finite positive fundamentals and finite anharmonicities required")
    if not np.allclose(x, x.T, rtol=1e-12, atol=1e-12):
        raise ValueError("the diagonal-anharmonicity matrix must be symmetric")
    if isinstance(maximum_integrals, bool) or not isinstance(maximum_integrals, (int, np.integer)) or maximum_integrals < 1:
        raise ValueError("maximum_integrals must be a positive integer")
    if not np.isfinite(harmonic_block_tolerance) or harmonic_block_tolerance < 0:
        raise ValueError("harmonic_block_tolerance must be finite and nonnegative")
    if (isinstance(profile_tail_relative_tolerance, bool)
            or not np.isfinite(profile_tail_relative_tolerance)
            or profile_tail_relative_tolerance < 0):
        raise ValueError("profile_tail_relative_tolerance must be finite and nonnegative")
    if (isinstance(sampling_profile_floor, bool) or not np.isfinite(sampling_profile_floor)
            or not 0 < sampling_profile_floor <= 1):
        raise ValueError("sampling_profile_floor must be finite and in (0, 1]")
    if (not np.isfinite(rate_relative_standard_error_tolerance)
            or rate_relative_standard_error_tolerance <= 0
            or not np.isfinite(profile_tail_rate_bound_tolerance)
            or profile_tail_rate_bound_tolerance < 0):
        raise ValueError("positive rate-error tolerance and nonnegative profile-tail tolerance required")
    if transition_strength_callback is not None and harmonic_block_tolerance != 0:
        raise ValueError("harmonic_block_tolerance applies only to the built-in harmonic recurrence")
    if not np.isfinite(gaussian_sigma_cm + lorentz_hwhm_cm) or min(gaussian_sigma_cm, lorentz_hwhm_cm) < 0 or gaussian_sigma_cm + lorentz_hwhm_cm <= 0:
        raise ValueError("a finite nonzero Voigt linewidth is required")
    if kind not in {"absorption", "emission"}:
        raise ValueError("kind must be absorption or emission")
    if kind != "emission":
        raise ValueError("the current WL transition callback samples final-state occupations for emission")

    oriented, _ = orient_harmonic_model(model, kind)
    if len(oriented.initial_frequencies_cm) != len(nu) or len(oriented.final_frequencies_cm) != len(nu):
        raise ValueError("WL energy model and harmonic transition model must have the same mode count")

    scale = np.sqrt(weights)
    oriented = type(oriented)(
        oriented.initial_frequencies_cm,
        oriented.final_frequencies_cm,
        oriented.mode_map,
        oriented.final_shift_dimensionless,
        oriented.electronic_gap_cm,
        coupling * scale,
        np.zeros_like(oriented.transition_gradient_dimensionless_au),
        {**oriented.provenance, "coupling": "explicit constant nonadiabatic matrix element"},
        np.zeros_like(oriented.transition_momentum_dimensionless_au),
    )
    origin = -oriented.origin_cm if vibronic_origin_cm is None else float(vibronic_origin_cm)
    if not np.isfinite(origin) or origin <= 0:
        raise ValueError("emission requires a finite positive vibronic origin")
    controls = {} if sampler_controls is None else dict(sampler_controls)
    if "inverse_temperatures" in controls or "observables" in controls:
        raise ValueError("inverse temperatures and observables are set by this rate routine")
    width = max(gaussian_sigma_cm, lorentz_hwhm_cm)
    controls.setdefault("energy_max_cm", origin + 8 * width)
    controls.setdefault("max_total_quanta", max(64, int(np.ceil(controls["energy_max_cm"] / float(np.min(nu)) * 1.25))))
    controls.setdefault("energy_bins", 4096)
    controls.setdefault("observable_stride", 64)
    profile_peak = float(voigt_profile(0.0, gaussian_sigma_cm, lorentz_hwhm_cm))
    if "frozen_energy_weights" not in controls:
        sample_centers = (np.arange(controls["energy_bins"]) + 0.5) * controls["energy_max_cm"] / controls["energy_bins"]
        target_profile = voigt_profile(origin - sample_centers, gaussian_sigma_cm, lorentz_hwhm_cm)
        controls["frozen_energy_weights"] = np.maximum(target_profile / profile_peak, sampling_profile_floor)
    from matrix_sentinel.quantum_wang_landau import _cubic_energy, _cubic_terms, _energy_cm
    y_tensor = controls.get("y_tensor_cm")
    cubic_terms = _cubic_terms(y_tensor)
    if transition_strength_callback is None:
        gaussian = harmonic_gaussian_generator(oriented)
        if gaussian["log_ground"] < np.log(np.finfo(float).tiny) + 4:
            raise FloatingPointError("harmonic ground overlap underflows; reduce the displacement or use a stable recurrence")
        # Sampled emission states depend only on the final/final block of the
        # Gaussian generator. A nonzero tolerance truncates weak cross-mode
        # edges so the remaining recurrence can be factored into components.
        final_matrix = np.asarray(gaussian["matrix"][len(nu):, len(nu):], float)
        truncated = final_matrix.copy()
        off_diagonal = ~np.eye(len(nu), dtype=bool)
        removed = off_diagonal & (np.abs(truncated) <= harmonic_block_tolerance)
        truncated[removed] = 0.0
        graph = truncated != 0.0
        np.fill_diagonal(graph, False)
        _, labels = connected_components(graph, directed=False)
        components = [np.flatnonzero(labels == label) for label in range(labels.max() + 1)]
        block_diagnostics = {
            "tolerance": float(harmonic_block_tolerance),
            "component_sizes": [int(len(component)) for component in components],
            "removed_off_diagonal_elements": int(np.count_nonzero(removed)),
            "removed_matrix_frobenius_norm": float(np.linalg.norm(final_matrix - truncated)),
            "removed_matrix_max_abs": float(np.max(np.abs(final_matrix - truncated), initial=0.0)),
        }
        state_cache = {}

        def raw_strength_by_state(states):
            occupations = np.asarray(states, dtype=np.int64)
            strengths = np.zeros(len(occupations))
            ground_overlap = np.exp(gaussian["log_ground"])
            coupling_strength = float(np.sum(np.abs(oriented.transition_dipole_au) ** 2))
            for row, occupation in enumerate(occupations):
                key = tuple(occupation.tolist())
                if key in state_cache:
                    strengths[row] = state_cache[key]
                    continue
                overlap = ground_overlap
                for component in components:
                    local_state = occupation[component][None, :]
                    local_matrix = truncated[np.ix_(component, component)]
                    local_linear = gaussian["linear"][len(nu) + component]
                    local_htlinear = gaussian["ht_linear"][:, len(nu) + component]
                    ids, values = _integral_dag(
                        local_state, local_matrix, local_linear, local_htlinear,
                        maximum_integrals, 1.0,
                    )
                    overlap *= values[ids[0, 0]]
                strength = coupling_strength * abs(overlap) ** 2
                strengths[row] = strength
                state_cache[key] = strength
                if len(state_cache) > 20000:
                    state_cache.pop(next(iter(state_cache)))
            return strengths

        strength_source = "harmonic FC overlap with Duschinsky rotation and displacement"
    else:
        if not callable(transition_strength_callback):
            raise ValueError("transition_strength_callback must be callable")

        def raw_strength_by_state(states):
            occupations = np.asarray(states, dtype=np.int64)
            values = np.asarray(transition_strength_callback(occupations.copy()), float)
            if values.shape != (len(occupations),) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("transition strength callback must return one finite nonnegative Hartree-squared value per occupation")
            return values

        strength_source = "caller-supplied state-resolved squared coupling"

    def strength_by_state(states):
        occupations = np.asarray(states, dtype=np.int64)
        energies = np.array([_energy_cm(row, nu, x) + _cubic_energy(row, cubic_terms) for row in occupations])
        profile_values = voigt_profile(origin - energies, gaussian_sigma_cm, lorentz_hwhm_cm)
        selected = np.ones(len(occupations), dtype=bool)
        if transition_strength_callback is None and profile_tail_relative_tolerance:
            selected = profile_values >= profile_tail_relative_tolerance * profile_peak
        values = np.zeros(len(occupations))
        if np.any(selected):
            values[selected] = raw_strength_by_state(occupations[selected]) * profile_values[selected]
        return values

    if sampling_method == "occupation_importance":
        return _occupation_importance_rate(
            oriented, nu, x, raw_strength_by_state, strength_by_state,
            controls=controls, importance_controls=importance_controls,
            rate_tolerance=rate_relative_standard_error_tolerance,
            gaussian_sigma_cm=gaussian_sigma_cm, lorentz_hwhm_cm=lorentz_hwhm_cm,
            origin=origin, strength_source=strength_source,
            block_diagnostics=block_diagnostics if transition_strength_callback is None else None,
        )

    wl = sample_integer_windowed_wang_landau(
        nu, x, inverse_temperatures=inverse_temperatures,
        observables={"profile_weighted_strength_au2_per_cm": strength_by_state}, **controls,
    )
    if not wl.diagnostics.get("measurement_complete", False):
        diagnostics = dict(wl.diagnostics)
        diagnostics.update(
            energy_bin_width_cm=float(wl.energy_bin_edges_cm[1] - wl.energy_bin_edges_cm[0]),
            transition_wavefunctions=("harmonic FC overlap with Duschinsky rotation and displacement"
                if transition_strength_callback is None else "caller-supplied state-resolved squared coupling"),
            vibronic_origin_cm=origin,
            rate_unavailable_reason="frozen Wang--Landau measurement did not complete",
        )
        if transition_strength_callback is None:
            diagnostics["harmonic_recurrence_blocks"] = block_diagnostics
        return dict(
            schema="matrix.sati.wang_landau_nonradiative_rate.v1",
            rate_s_1=None,
            coupling_density_hartree2_per_cm=None,
            gaussian_sigma_cm=float(gaussian_sigma_cm),
            lorentz_hwhm_cm=float(lorentz_hwhm_cm),
            diagnostics=diagnostics,
            wang_landau=wl,
            scope=("rate unavailable because the frozen Wang--Landau measurement is incomplete; "
                   "no estimate is inferred from adaptation data"),
            provenance=oriented.provenance,
        )
    edges = np.asarray(wl.energy_bin_edges_cm, float)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_width = float(edges[1] - edges[0])
    mismatch = origin - centers
    property_result = wl.microcanonical_properties["profile_weighted_strength_au2_per_cm"]
    sampled_strength = np.asarray(property_result["mean"])
    if np.any(np.isfinite(sampled_strength.imag) & (np.abs(sampled_strength.imag) > 1e-12 * np.maximum(1., np.abs(sampled_strength.real)))):
        raise FloatingPointError("Wang--Landau transition-strength observable acquired an imaginary component")
    mean_strength = sampled_strength.real
    dos = np.asarray(wl.density_of_states, float)
    profile = voigt_profile(mismatch, gaussian_sigma_cm, lorentz_hwhm_cm)
    ladder_dos = np.asarray(wl.diagnostics.get("density_of_states_by_ladder"), float)
    ladder_strength = np.asarray(property_result.get("mean_by_ladder"), complex)
    if (ladder_dos.ndim != 2 or ladder_dos.shape[1] != len(profile)
            or ladder_strength.shape != ladder_dos.shape):
        raise RuntimeError("Wang--Landau sampler did not return ladder-resolved strengths and densities")
    if np.any(np.isfinite(ladder_strength.imag) & (np.abs(ladder_strength.imag) > 1e-12 * np.maximum(1., np.abs(ladder_strength.real)))):
        raise FloatingPointError("ladder-resolved transition strengths acquired an imaginary component")
    ladder_strength = ladder_strength.real
    if np.any(np.isfinite(mean_strength.imag) & (np.abs(mean_strength.imag) > 1e-12 * np.maximum(1., np.abs(mean_strength.real)))):
        raise FloatingPointError("pooled transition strengths acquired an imaginary component")
    mean_strength = mean_strength.real
    bin_profile_upper = voigt_profile(np.maximum(0., np.abs(mismatch) - bin_width / 2),
                                      gaussian_sigma_cm, lorentz_hwhm_cm)
    effective_tail_tolerance = profile_tail_relative_tolerance if transition_strength_callback is None else 0.0
    tail_bins = bin_profile_upper < effective_tail_tolerance * profile_peak
    required_bins = ~tail_bins
    missing_pooled_bins = (dos > 0) & ~np.isfinite(mean_strength) & required_bins
    if np.any(missing_pooled_bins):
        raise RuntimeError(
            "important FGR bins lack pooled observable samples "
            f"({int(np.count_nonzero(missing_pooled_bins))} bins; "
            f"first indices {np.flatnonzero(missing_pooled_bins)[:12].tolist()}); "
            "reduce observable_stride or increase measurement_steps"
        )
    imputed_ladder_bins = (ladder_dos > 0) & ~np.isfinite(ladder_strength) & required_bins[None, :]
    ladder_strength = np.where(np.isfinite(ladder_strength), ladder_strength,
                               np.where(np.isfinite(mean_strength)[None, :], mean_strength[None, :], 0.0))
    ladder_valid = np.isfinite(ladder_dos)
    ladder_densities = np.sum(np.where(ladder_valid, ladder_dos * ladder_strength, 0.0), axis=1)
    ladder_density_mean = float(np.mean(ladder_densities))
    pooled_valid = np.isfinite(dos) & np.isfinite(mean_strength)
    density = float(np.sum(dos[pooled_valid] * mean_strength[pooled_valid]))
    if len(ladder_densities) > 1:
        density_se = float(np.std(ladder_densities, ddof=1) / np.sqrt(len(ladder_densities)))
        t95 = float(student_t.ppf(0.975, len(ladder_densities) - 1))
        density_ci95 = [max(0.0, density - t95 * density_se), density + t95 * density_se]
    else:
        density_se = float("nan")
        density_ci95 = [float("nan"), float("nan")]
    omitted_density_bound = 0.0
    if transition_strength_callback is None and np.any(tail_bins):
        coupling_strength = float(np.sum(np.abs(oriented.transition_dipole_au) ** 2))
        omitted_density_bound = float(coupling_strength * np.sum(dos[tail_bins] * bin_profile_upper[tail_bins]))
    relative_density_se = float(density_se / density) if density > 0 else float("inf")
    relative_tail_bound = float(omitted_density_bound / density) if density > 0 else float("inf")
    wl_converged = bool(wl.diagnostics.get("converged", False))
    rate_converged = bool(
        wl_converged
        and relative_density_se <= rate_relative_standard_error_tolerance
        and relative_tail_bound <= profile_tail_rate_bound_tolerance
        and not np.any(imputed_ladder_bins)
    )
    if not np.isfinite(density) or density < 0:
        raise FloatingPointError("Wang--Landau Golden Rule density must be finite and nonnegative")
    diagnostics = dict(wl.diagnostics)
    diagnostics.update(
        wang_landau_converged=wl_converged,
        transition_rate_converged=rate_converged,
        converged=rate_converged,
        rate_relative_standard_error_tolerance=float(rate_relative_standard_error_tolerance),
        profile_tail_rate_bound_tolerance=float(profile_tail_rate_bound_tolerance),
        energy_bin_width_cm=bin_width,
        bin_width_to_linewidth=float(bin_width / max(gaussian_sigma_cm, lorentz_hwhm_cm)),
        energy_conservation_profile="Voigt evaluated at energy-bin centers",
        transition_strength_samples=int(np.sum(property_result["sample_count"])),
        independent_ladder_coupling_densities_hartree2_per_cm=ladder_densities.tolist(),
        independent_ladder_mean_coupling_density_hartree2_per_cm=ladder_density_mean,
        pooled_mean_imputed_ladder_bins=int(np.count_nonzero(imputed_ladder_bins)),
        pooled_mean_imputed_bins_by_ladder=np.count_nonzero(imputed_ladder_bins,axis=1).tolist(),
        ladder_uncertainty_qualified=bool(not np.any(imputed_ladder_bins)),
        ladder_error_note=("between-ladder estimate uses pooled means for missing ladder bins; interval is descriptive"
                           if np.any(imputed_ladder_bins) else "independent ladder estimates use sampled bins"),
        coupling_density_standard_error_hartree2_per_cm=density_se,
        coupling_density_relative_standard_error=relative_density_se,
        omitted_profile_tail_relative_upper_bound=relative_tail_bound,
        coupling_density_ladder_t_interval_95_hartree2_per_cm=density_ci95,
        profile_tail_relative_tolerance=float(effective_tail_tolerance),
        omitted_profile_tail_upper_bound_hartree2_per_cm=omitted_density_bound,
        weighted_strength_observable="state strength multiplied by Voigt profile at the sampled anharmonic energy",
        dos_strength_sum_discrete=float(np.sum(dos[np.isfinite(mean_strength)] * mean_strength[np.isfinite(mean_strength)])),
        anharmonic_energy_model="diagonal occupation model using supplied fundamentals and x matrix",
        transition_wavefunctions=strength_source,
        finite_temperature_initial_population=False,
        vibronic_origin_cm=origin,
    )
    if transition_strength_callback is None:
        diagnostics["harmonic_recurrence_blocks"] = block_diagnostics
    return dict(
        schema="matrix.sati.wang_landau_nonradiative_rate.v1",
        rate_s_1=float(GOLDEN_RULE_FACTOR * density),
        rate_standard_error_s_1=float(GOLDEN_RULE_FACTOR * density_se),
        rate_ladder_t_interval_95_s_1=[float(GOLDEN_RULE_FACTOR * bound) for bound in density_ci95],
        coupling_density_hartree2_per_cm=density,
        gaussian_sigma_cm=float(gaussian_sigma_cm),
        lorentz_hwhm_cm=float(lorentz_hwhm_cm),
        diagnostics=diagnostics,
        wang_landau=wl,
        scope=("diagonal-anharmonic Wang--Landau energies and DOS with harmonic "
               "Duschinsky FC transition strengths; not a coupled anharmonic "
               "high-state wavefunction calculation"),
        provenance=oriented.provenance,
    )


def _occupation_importance_rate(oriented, nu, x, raw_strength, weighted_strength, *,
                               controls, importance_controls, rate_tolerance,
                               gaussian_sigma_cm, lorentz_hwhm_cm, origin,
                               strength_source, block_diagnostics):
    from matrix_sentinel.quantum_importance import sample_integer_occupation_importance

    settings = {} if importance_controls is None else dict(importance_controls)
    proposal_scale = settings.pop("proposal_scale", None)
    pilot_rounds = settings.pop("pilot_rounds", 2)
    pilot_samples = settings.pop("pilot_samples_per_replica", 1000)
    if isinstance(pilot_rounds, bool) or not isinstance(pilot_rounds, (int, np.integer)) or pilot_rounds < 0:
        raise ValueError("nonnegative integer pilot_rounds required")
    if "resume" in controls:
        raise ValueError("Wang--Landau checkpoints do not apply to independent importance sampling")
    qmap = (np.sqrt(oriented.final_frequencies_cm)[:, None] * oriented.mode_map
            / np.sqrt(oriented.initial_frequencies_cm)[None, :])
    pmap = np.linalg.inv(qmap).T
    occupation_means = np.maximum(0.0, (
        0.5 * np.sum(qmap * qmap, axis=1) + 0.5 * np.sum(pmap * pmap, axis=1)
        + oriented.final_shift_dimensionless ** 2 - 1.0
    ) / 2.0)
    if proposal_scale is None:
        from scipy.optimize import brentq

        floor = float(settings.get("mean_floor", 0.02))
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("positive finite proposal mean floor required")
        expected = lambda scale: float(nu @ np.maximum(occupation_means * scale, floor))
        high = 1.0
        if np.any(occupation_means > 0) and expected(0.0) < origin:
            while expected(high) < origin:
                high *= 2
            proposal_scale = float(brentq(lambda scale: expected(scale) - origin, 0.0, high))
        else:
            proposal_scale = 1.0
    proposal_scale = float(proposal_scale)
    if not np.isfinite(proposal_scale) or proposal_scale <= 0:
        raise ValueError("positive finite importance proposal scale required")
    settings.setdefault("seed", controls.get("seed", 61453))
    proposal_means = occupation_means * proposal_scale
    initial_proposal_means = proposal_means.copy()
    pilot_history = []
    for step in range(pilot_rounds):
        pilot_settings = dict(settings)
        pilot_settings.update(samples_per_replica=pilot_samples, replicas=4,
                              seed=int(settings["seed"]) + 104729 * (step + 1))
        pilot = sample_integer_occupation_importance(
            nu, x, proposal_means=proposal_means,
            observables={"profile_weighted_strength": weighted_strength},
            max_total_quanta=controls["max_total_quanta"], energy_max_cm=controls["energy_max_cm"],
            y_tensor_cm=controls.get("y_tensor_cm"), **pilot_settings,
        )
        pilot_estimate = pilot["estimates"]["profile_weighted_strength"]
        learned_means = pilot_estimate["weighted_occupation_means"]
        if learned_means is not None:
            proposal_means = 0.5 * pilot["diagnostics"]["proposal_means"] + 0.5 * learned_means
        pilot_history.append(dict(
            seed=pilot_settings["seed"], proposals=pilot["diagnostics"]["total_proposals"],
            effective_sample_size=pilot_estimate["effective_sample_size"],
            learned_occupation_means=learned_means,
        ))
    sample = sample_integer_occupation_importance(
        nu, x, proposal_means=proposal_means,
        observables={"profile_weighted_strength": weighted_strength, "strength": raw_strength},
        max_total_quanta=controls["max_total_quanta"], energy_max_cm=controls["energy_max_cm"],
        y_tensor_cm=controls.get("y_tensor_cm"), **settings,
        defensive_means=initial_proposal_means,
    )
    estimate = sample["estimates"]["profile_weighted_strength"]
    density = float(estimate["mean"])
    error = max(float(estimate["standard_error"]), float(estimate["iid_standard_error"]))
    relative_error = error / density if density > 0 else float("inf")
    replicas = sample["diagnostics"]["replicas"]
    t95 = float(student_t.ppf(0.975, replicas - 1))
    interval = [max(0.0, density - t95 * error), density + t95 * error]
    precision_passed = bool(relative_error <= rate_tolerance
                            and estimate["effective_sample_size"] >= 100
                            and estimate["maximum_weight_fraction"] <= 0.01)
    coupling = float(np.sum(np.abs(oriented.transition_dipole_au) ** 2))
    diagnostics = dict(sample["diagnostics"])
    diagnostics.update(
        converged=False, transition_rate_converged=False,
        finite_domain_precision_passed=precision_passed,
        finite_domain_extension_required=True,
        rate_relative_standard_error_tolerance=float(rate_tolerance),
        coupling_density_relative_standard_error=relative_error,
        rate_effective_sample_size=estimate["effective_sample_size"],
        rate_maximum_weight_fraction=estimate["maximum_weight_fraction"],
        harmonic_occupation_means=occupation_means,
        proposal_scale=proposal_scale,
        pilot_history=pilot_history, pilot_samples_used_in_rate=False,
        captured_coupling_strength_fraction=sample["estimates"]["strength"]["mean"] / coupling if coupling > 0 else None,
        transition_wavefunctions=strength_source, vibronic_origin_cm=float(origin),
        harmonic_recurrence_blocks=block_diagnostics,
        uncertainty_note="maximum of between-replica and pooled iid standard errors; rare unseen states remain possible",
    )
    return dict(
        schema="matrix.sati.occupation_importance_nonradiative_rate.v1",
        rate_s_1=float(GOLDEN_RULE_FACTOR * density),
        rate_standard_error_s_1=float(GOLDEN_RULE_FACTOR * error),
        rate_replica_t_interval_95_s_1=[float(GOLDEN_RULE_FACTOR * v) for v in interval],
        coupling_density_hartree2_per_cm=density,
        gaussian_sigma_cm=float(gaussian_sigma_cm), lorentz_hwhm_cm=float(lorentz_hwhm_cm),
        diagnostics=diagnostics, importance=sample,
        scope="independent importance estimate over a finite diagonal-anharmonic occupation domain with supplied transition strengths; domain extension and wavefunction approximations require separate qualification",
        provenance=oriented.provenance,
    )


def importance_nonradiative_rate(model, fundamentals_cm, x_matrix_cm, coupling_hartree, **kwargs):
    """Sample FGR strengths directly with exact independent-proposal weights.

    Proposal means follow the initial harmonic vacuum expressed in final modes,
    including displacement, frequency changes and Duschinsky mixing. The
    returned sum covers only the declared finite occupation/energy domain.
    """
    return wang_landau_nonradiative_rate(
        model, fundamentals_cm, x_matrix_cm, coupling_hartree,
        sampling_method="occupation_importance", **kwargs,
    )


__all__ = [
    "diagonal_anharmonicity_from_assigned_levels",
    "sam_vpt2_transition_strength_callback",
    "wang_landau_nonradiative_rate",
    "importance_nonradiative_rate",
]
