"""Probability- and energy-conserving cascades on resolved downward rate graphs."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix

from .gvpt2_rates import _readonly


@dataclass(frozen=True)
class RadiativeCascade:
    initial_rows: np.ndarray
    final_rows: np.ndarray
    photon_wavenumbers_cm: np.ndarray
    integrated_photon_yields: np.ndarray
    residence_times_s: np.ndarray
    terminal_population: np.ndarray
    times_s: np.ndarray
    mean_energies_cm: np.ndarray
    mean_absorption_time_s: float
    cooling_time_90_s: float | None
    diagnostics: dict

    def photon_spectrum(self, grid_cm, *, fwhm_cm):
        """Gaussian photon-number density, applied after discrete photon factors."""
        grid = np.asarray(grid_cm, float)
        if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid))
                or np.any(np.diff(grid) <= 0) or not np.isfinite(fwhm_cm) or fwhm_cm <= 0):
            raise ValueError('finite increasing wavenumbers and positive width required')
        sigma = fwhm_cm/np.sqrt(8*np.log(2))
        values = np.zeros(len(grid))
        # Keep memory linear in the presentation grid, independently of edges.
        for frequency, photons in zip(self.photon_wavenumbers_cm, self.integrated_photon_yields):
            if photons:
                values += photons*np.exp(-.5*((grid-frequency)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        return values


def solve_downward_cascade(energies_cm, pairs, rates_s_inverse, initial_population, *,
        times_s=None, relative_tolerance=1e-9, absolute_tolerance=1e-12):
    """Evolve a finite resolved rate graph and integrate its photons exactly.

    Every pair supplies one downward spontaneous rate, independent of row order.
    Integrated residence times follow the energy-ordered graph without time
    stepping. A sparse master equation supplies time-dependent mean energies.
    The radiative graph is the explicit supplied inventory: completeness, IVR,
    collisions and nonradiative pathways are not silently assumed.
    """
    energies, rates, population = (np.asarray(v, float) for v in
                                  (energies_cm, rates_s_inverse, initial_population))
    pairs = np.asarray(pairs)
    if (energies.ndim != 1 or len(energies) < 2 or not np.all(np.isfinite(energies))
            or population.shape != energies.shape or not np.all(np.isfinite(population))
            or np.any(population < 0) or not np.isclose(population.sum(), 1., rtol=0, atol=1e-12)
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
            or np.any(pairs < 0) or np.any(pairs >= len(energies))
            or np.any(pairs[:, 0] == pairs[:, 1])
            or len({tuple(sorted(p)) for p in pairs}) != len(pairs)
            or rates.shape != (len(pairs),) or not np.all(np.isfinite(rates)) or np.any(rates < 0)):
        raise ValueError('resolved finite energies, distinct pairs, nonnegative rates and normalized populations required')
    if any(not np.isfinite(t) or t <= 0 for t in (relative_tolerance, absolute_tolerance)):
        raise ValueError('positive finite propagation tolerances required')
    a, b = pairs.T
    high = np.where(energies[a] >= energies[b], a, b)
    low = np.where(high == a, b, a)
    gaps = energies[high]-energies[low]
    if np.any((gaps == 0) & (rates > 0)):
        raise ValueError('zero-gap spontaneous rates must vanish')
    gamma = np.bincount(high, weights=rates, minlength=len(energies))
    outgoing = [[] for _ in energies]
    for edge, upper in enumerate(high):
        if rates[edge] > 0:
            outgoing[upper].append(edge)
    visits, residence, photons = population.copy(), np.zeros(len(energies)), np.zeros(len(rates))
    for upper in np.argsort(energies, kind='stable')[::-1]:
        if gamma[upper] > 0:
            residence[upper] = visits[upper]/gamma[upper]
            edges = outgoing[upper]
            photons[edges] = residence[upper]*rates[edges]
            np.add.at(visits, low[edges], photons[edges])
    terminal = np.where(gamma == 0, visits, 0.)
    initial_energy, terminal_energy = float(population@energies), float(terminal@energies)
    emitted_energy = float(photons@gaps)
    conservation_error = emitted_energy-(initial_energy-terminal_energy)
    if not np.isclose(terminal.sum(), 1., rtol=0, atol=1e-10) or abs(conservation_error) > 1e-9*max(1., abs(emitted_energy)):
        raise ArithmeticError('downward cascade violates probability or photon-energy conservation')
    reached = (visits > 0) & (gamma > 0)
    if times_s is None:
        horizon = 20/np.min(gamma[reached]) if np.any(reached) else 1.
        times = np.r_[0., np.geomspace(horizon*1e-10, horizon, 240)]
    else:
        times = np.asarray(times_s, float)
        if (times.ndim != 1 or len(times) < 2 or times[0] != 0
                or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
            raise ValueError('finite increasing times beginning at zero required')
    if not np.all(np.isfinite(residence)) or not np.all(np.isfinite(times)):
        raise FloatingPointError('cascade time scale exceeds numerical range')
    n = len(energies)
    generator = coo_matrix((np.r_[rates, -gamma],
        (np.r_[low, np.arange(n)], np.r_[high, np.arange(n)])), shape=(n, n)).tocsc()
    target = terminal_energy+.1*(initial_energy-terminal_energy)

    def cooling_event(t, p):
        return float(p@energies-target)

    cooling_event.direction = -1
    event = cooling_event if initial_energy > terminal_energy else None
    trajectory = solve_ivp(lambda t, p: generator@p, (0., float(times[-1])), population,
        method='BDF', jac=generator, t_eval=times, events=event,
        rtol=relative_tolerance, atol=absolute_tolerance)
    if not trajectory.success:
        raise ArithmeticError('cascade master equation failed: '+trajectory.message)
    probability_error = float(abs(trajectory.y.sum(axis=0)-1.).max())
    minimum_population = float(trajectory.y.min())
    mean_energies = energies@trajectory.y
    if probability_error > 1e-7 or minimum_population < -1e-8:
        raise ArithmeticError('propagated cascade lost probability or positivity')
    cooling_time = (float(trajectory.t_events[0][0]) if event is not None
                    and len(trajectory.t_events[0]) else None)
    return RadiativeCascade(*[_readonly(v) for v in (high, low, gaps, photons, residence,
        terminal, times, mean_energies)], float(residence.sum()), cooling_time,
        dict(initial_energy_cm=initial_energy, terminal_energy_cm=terminal_energy,
            emitted_energy_cm=emitted_energy, energy_conservation_error_cm=conservation_error,
            terminal_probability_error=float(abs(terminal.sum()-1)),
            propagated_probability_max_error=probability_error, minimum_population=minimum_population,
            reached_transient_states=int(sum(reached)), listed_channels=len(rates),
            endpoint_completeness_asserted=False, generator_column_sum_max_error=float(abs(np.asarray(generator.sum(axis=0))).max()),
            relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance))
