"""Shared statistical convolution extracted from the resident TRINITY adapter.

No new spectral or statistical kernel: arbitrary qualified discrete LAM,
rotational, or joint LAM-rotation spectra use the same convolution and exact
canonical moments. Weights already include required statistical factors.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import logsumexp
from matrix_chem import Phy, get_physical_constants


def convolve_discrete_measure(values, bin_width_cm, energies_cm, weights, *, algorithm='fft'):
    """Convolve a zero-origin DOS or cumulative probability with discrete levels.

    The same linear spectral deposition is used for densities and N(E).
    No vibrational or thermochemical interpretation is imposed on the input.
    """
    if any(np.iscomplexobj(v) for v in [values,energies_cm,weights]):
        raise ValueError('real spectral measures and weights required')
    values=np.asarray(values,float);e=np.asarray(energies_cm,float);g=np.asarray(weights,float)
    if (values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values<0)
            or not np.isfinite(bin_width_cm) or bin_width_cm<=0 or e.ndim!=1 or not len(e)
            or g.shape!=e.shape or not np.all(np.isfinite(e)) or not np.all(np.isfinite(g))
            or np.any(e<0) or np.any(g<0) or not np.any(g>0)):
        raise ValueError('nonnegative measure, positive bin width and weighted discrete spectrum required')
    shifted=e/bin_width_cm;lower=np.floor(shifted).astype(int);fractions=shifted-lower
    if int(np.max(lower))+2>10_000_000:raise MemoryError('convolution grid exceeds ten million bins')
    if algorithm=='positive_direct':
        # N(E) can span hundreds of orders of magnitude. FFT roundoff creates
        # an artificial low-energy floor that dominates a low-T Boltzmann sum.
        # Positive spectral shifts preserve exact zeros and the dynamic range.
        combined=np.zeros(len(values)+int(np.max(lower))+1)
        # Deposit once before applying shifts: many rotational levels share
        # a grain. Grouping their positive weights retains the same linear
        # interpolation and avoids one full-array operation per rotor level.
        size=int(np.max(lower))+2
        shifts=np.bincount(lower,weights=g*(1-fractions),minlength=size)
        shifts+=np.bincount(lower+1,weights=g*fractions,minlength=size)
        for offset in np.flatnonzero(shifts):
            combined[offset:offset+len(values)]+=shifts[offset]*values
        return combined
    if algorithm!='fft':raise ValueError('algorithm must be fft or positive_direct')
    spectrum=np.zeros(int(np.max(lower))+2)
    np.add.at(spectrum,lower,g*(1-fractions));np.add.at(spectrum,lower+1,g*fractions)
    combined=fftconvolve(values,spectrum)
    if np.min(combined)<-1e-10*max(1.,float(np.max(combined))):
        raise FloatingPointError('spectral FFT convolution produced a negative measure')
    return np.maximum(combined,0.)


def convolve_discrete_levels(vibrational, rotational_energies_cm, rotational_weights, *, sector="rotation", zero_point_cm=None, algorithm="fft"):
    """Convolve a uniform vibrational DOS with weighted discrete rotor levels.

    Input weights contain all desired M/nuclear-spin/statistical weights.
    No additional symmetry number is divided out. Rotor energies are
    measured from its ground level. Histogram shifts use linear deposition
    and FFT convolution, preserving total mass and first moment. Canonical
    properties instead use exact discrete rotor energies and the sampled
    within-bin vibrational moments: no bin-center Boltzmann approximation.
    The rotor is assumed independent of vibrational occupation.
    """
    from types import SimpleNamespace
    from collections.abc import Mapping
    if isinstance(vibrational, Mapping):vibrational=SimpleNamespace(**vibrational)
    included=list(getattr(vibrational,'included_sectors',('SAM',)))
    additions=['LAM','rotation'] if sector=='LAM-rotation' else [sector]
    if any(item in included for item in additions):raise ValueError('statistical sector already included; double counting forbidden')
    if zero_point_cm is None:
        if sector!='rotation':raise ValueError('explicit sector ZPE required for LAM/SAM convolution')
        zero_point_cm=0.
    if not np.isfinite(zero_point_cm):raise ValueError('finite sector ZPE required')
    if sector not in {"SAM", "LAM", "rotation", "LAM-rotation"}:
        raise ValueError("declared statistical convolution sector required")
    e = np.asarray(rotational_energies_cm, float)
    g = np.asarray(rotational_weights, float)
    if (e.ndim != 1 or not len(e) or g.shape != e.shape or not np.all(np.isfinite(e))
            or not np.all(np.isfinite(g)) or np.any(e < 0) or np.any(g < 0) or not np.any(g > 0)):
        raise ValueError("finite nonnegative rotor energies and weights required")
    e=e+zero_point_cm
    if np.any(e<0):raise ValueError("nonnegative combined spectral energy required on convolution grid")
    edges = np.asarray(vibrational.energy_bin_edges_cm, float)
    density = np.asarray(vibrational.density_of_states, float)
    if (edges.shape != (len(density)+1,) or not np.all(np.isfinite(edges))
            or not np.all(np.isfinite(density)) or np.any(density < 0)
            or edges[0] != 0 or np.any(np.diff(edges) <= 0)
            or not np.allclose(np.diff(edges), edges[1]-edges[0], rtol=1e-10, atol=1e-12)):
        raise ValueError("finite uniform vibrational DOS with zero excitation origin required")
    delta = edges[1]-edges[0]
    combined = convolve_discrete_measure(density,delta,e,g,algorithm=algorithm)
    c = get_physical_constants()
    hc = c[Phy.PLANCK]*c[Phy.C_LIGHT]
    gas = c[Phy.AVOGADRO]*c[Phy.BOLTZMANN]
    temperatures = np.asarray(vibrational.temperatures_K, float)
    beta = hc/(c[Phy.BOLTZMANN]*temperatures)
    active = g > 0
    log_weights = np.log(g[active])[None,:]-beta[:,None]*e[active]
    log_q = logsumexp(log_weights, axis=1)
    population = np.exp(log_weights-log_q[:,None])
    mean = population@e[active]
    variance = population@(e[active]**2)-mean**2
    base_log_partition=getattr(vibrational,"log_partition_statistical",getattr(vibrational,"log_partition_vibrational",None))
    if base_log_partition is None:raise ValueError("source statistical partition required")
    return dict(schema="matrix.rovib.statistical_convolution.v1",
        energy_bin_edges_cm=np.arange(len(combined)+1)*delta,
        density_of_states=combined,
        expected_weighted_state_count=float(np.sum(density)*np.sum(g)),
        state_count_residual=float(np.sum(combined)-np.sum(density)*np.sum(g)),
        temperatures_K=temperatures,
        log_partition_rovibrational=base_log_partition+log_q,
        log_partition_statistical=base_log_partition+log_q,
        log_partition_spectral_factor=log_q,
        included_sectors=included+additions,sector_zero_point_cm=zero_point_cm,
        log_partition_rotational=log_q,
        internal_energy_kj_mol=vibrational.internal_energy_kj_mol+hc*c[Phy.AVOGADRO]*mean/1000,
        entropy_j_mol_K=vibrational.entropy_j_mol_K+gas*(log_q+beta*mean),
        heat_capacity_j_mol_K=vibrational.heat_capacity_j_mol_K+gas*beta**2*np.maximum(variance,0.),
        separable_rotor=sector=="rotation", separable_spectrum=True, extra_symmetry_division_applied=False,
        histogram_interpolation="linear spectral shifts; canonical moments use exact discrete levels",
        convolution_sector=sector,
        cumulative_number_of_states=np.cumsum(combined),
        density_per_cm=combined/delta)
