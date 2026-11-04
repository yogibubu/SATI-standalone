"""Shared complex FC/HT operator bank for electronic optical observables.

Every observable is a Hermitian quadratic form O_a W_ab O_b*. One Gaussian
factorization/cache serves all operators and signed interference observables.
Electronic response tensors are inputs, not inferred from oscillator strengths.
"""
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import numpy as np
from numba import njit
from scipy import fft
from threadpoolctl import threadpool_limits
from .vibronic_harmonic import (harmonic_gaussian_generator, harmonic_thermal_factors,
                               orient_harmonic_model, PHASE_PER_CM_FS)
from .vibronic_ti import _occupation_array, _integral_dag
from .vibronic_kernels import worker_count, parallel_rows


def prepare_operator_bank(model, constants, gradients, forms, kind):
    c = np.asarray(constants, complex)
    h = np.zeros((len(c), len(model.initial_frequencies_cm)), complex) if gradients is None else np.asarray(gradients, complex)
    w = np.asarray(forms, complex)
    if (c.ndim != 1 or not len(c) or h.shape != (len(c), len(model.initial_frequencies_cm))
            or w.ndim != 3 or w.shape[1:] != (len(c), len(c)) or not len(w)
            or not all(np.all(np.isfinite(x)) for x in (c, h, w))
            or not np.allclose(w, w.conj().swapaxes(1, 2), atol=1e-12, rtol=1e-12)):
        raise ValueError('finite compatible constants, gradients and Hermitian observable forms required')
    if kind == 'emission':
        transform = np.sqrt(model.initial_frequencies_cm)[:, None] * np.linalg.inv(model.mode_map)
        transform /= np.sqrt(model.final_frequencies_cm)[None, :]
        c = (c-h @ transform @ model.final_shift_dimensionless).conj()
        h = (h @ transform).conj()
    model, direction = orient_harmonic_model(model, kind)
    return model,direction,c,h,w


def _bank(model, constants, gradients, forms, kind):
    model,direction,c,h,w=prepare_operator_bank(model,constants,gradients,forms,kind)
    g = harmonic_gaussian_generator(model)
    wi, wf, j = model.initial_frequencies_cm, model.final_frequencies_cm, model.mode_map
    a = np.diag(wi)+(j.T*wf) @ j
    rhs = np.column_stack([-j.T @ (np.sqrt(wf)*model.final_shift_dimensionless),
                           np.sqrt(2)*np.column_stack([np.diag(np.sqrt(wi)), j.T*np.sqrt(wf)])])
    coordinates = np.sqrt(wi)[:, None]*np.linalg.solve(a, rhs)
    constant = c+h @ coordinates[:, 0]
    linear = h @ coordinates[:, 1:]
    return model, direction, c, h, w, g, constant, linear


@njit(cache=True, nogil=True)
def _operator_rows(start, stop, states, ids, values, constant, linear, output):
    for row in range(start, stop):
        for channel in range(len(constant)):
            value = constant[channel]*values[ids[row, 0]]
            for mode in range(states.shape[1]):
                if ids[row, mode+1] >= 0:
                    value += linear[channel, mode]*np.sqrt(states[row, mode])*values[ids[row, mode+1]]
            output[row, channel] = value


def harmonic_operator_spectrum(model, constants, forms, *, gradients=None, method='TI',
        kind='absorption', temperature_K=0., maximum_initial_quanta=None,
        maximum_final_quanta=None, maximum_class=None, maximum_states=100000,
        maximum_transitions=100000, maximum_integrals=500000, workers=0,
        time_step_fs=.25, time_points=2049, gaussian_sigma_cm=50., lorentz_hwhm_cm=0.,
        chunk_size=8, memory_budget_bytes=256*1024**2, fft_workers=0,
        selection="enumerate", target_fraction=.99, selection_batch_size=256,
        selection_maximum_per_mode_quanta=128, checkpoint_path=None, resume=False,
        checkpoint_interval_batches=10, checkpoint_interval_seconds=60.,selection_proposal=None,
        backend='python',native_executable=None,selection_energy_bias_per_cm=None,selection_channel_weights=None,
        selection_windows_cm=None,selection_profile_points=65,
        selection_profile_peak_tolerance=.02,selection_profile_L1_tolerance=.01,
        selection_profile_absolute_tolerance=0.,selection_window_rankings=True,
        selection_assignment_limit=50,selection_assignment_report_path=None):
    """Signed strength sticks or analytic thermal TD densities, in input units.

    Gradients are with respect to dimensionless initial-state normal positions.
    TD never enumerates vibronic states. Signed spectra are never clipped.
    TI omission bounds use the positive bank norm, not a canceling signed sum.
    Optional selection_energy_bias_per_cm exponentially tilts the proposal
    toward low (positive bias) or high (negative bias) oriented transition
    energies. It must leave the proposal normalizable. Positive per-channel
    selection_channel_weights balance operator ranking with operator_c2.
    Neither control reweights physical strengths or changes the sum rules.
    selection='targeted' builds a same-model TD reference for disjoint
    selection_windows_cm and stops on the requested fraction in EVERY window.
    A None energy bias chooses a normalizable ranking automatically. Returned
    global captured_fraction remains separate from targeted convergence.
    Targeted stopping additionally checks every observable profile (default
    relative peak/L1 tolerances .02/.01). Multiple windows use distinct
    rankings and one deduplicated transition set/cache by default. The
    assignment_report includes FC/HT decomposition for its displayed lines;
    selection_assignment_report_path optionally saves it atomically.

    """
    spectral_target=None
    if selection=='targeted':
        if method!='TI' or selection_windows_cm is None:
            raise ValueError('targeted selection requires TI and selection_windows_cm')
        from .vibronic_targeted import SpectralSelectionTarget
        # The identity form supplies the positive bank-density reference even
        # when all user observables are signed or indefinite quadratic forms.
        forms_array=np.asarray(forms,complex)
        reference=harmonic_operator_spectrum(model,constants,
            np.concatenate([forms_array,np.eye(len(constants))[None]],axis=0),
            gradients=gradients,method='TD',kind=kind,temperature_K=temperature_K,
            time_step_fs=time_step_fs,time_points=time_points,gaussian_sigma_cm=gaussian_sigma_cm,
            lorentz_hwhm_cm=lorentz_hwhm_cm,chunk_size=chunk_size,memory_budget_bytes=memory_budget_bytes,
            workers=workers,fft_workers=fft_workers,backend=backend,native_executable=native_executable)
        spectral_target=SpectralSelectionTarget(selection_windows_cm,reference['wavenumbers_cm'],
            reference['densities_per_cm'][:-1],reference['densities_per_cm'][-1],
            gaussian_sigma_cm=gaussian_sigma_cm,lorentz_hwhm_cm=lorentz_hwhm_cm,
            profile_points=selection_profile_points,profile_peak_tolerance=selection_profile_peak_tolerance,
            profile_L1_tolerance=selection_profile_L1_tolerance,profile_absolute_tolerance=selection_profile_absolute_tolerance,
            window_rankings=selection_window_rankings,assignment_limit=selection_assignment_limit)
        selection='adaptive'
        if selection_proposal is None:selection_proposal='operator_c2'
    elif selection_windows_cm is not None or selection_assignment_report_path is not None:
        raise ValueError('selection_windows_cm requires selection=targeted')
    if selection_proposal is None:selection_proposal='gaussian_c2'
    if selection_energy_bias_per_cm is None and spectral_target is None:selection_energy_bias_per_cm=0.
    if ((selection_energy_bias_per_cm not in (None,0) or selection_channel_weights is not None)
            and (method!='TI' or selection!='adaptive')):
        raise ValueError('selection bias and channel weights require adaptive TI')
    maximum_initial_quanta = (128 if selection == 'adaptive' else 2) if maximum_initial_quanta is None else maximum_initial_quanta
    maximum_final_quanta = (128 if selection == 'adaptive' else 10) if maximum_final_quanta is None else maximum_final_quanta
    if maximum_class is None and selection != 'adaptive': maximum_class = 2
    if backend not in ('python','cpp'):raise ValueError('backend must be python or cpp')
    if backend=='cpp':
        import os
        from .vibronic_native import native_harmonic_operator_spectrum
        executable=native_executable or os.environ.get('MATRIX_VIBRONIC_EXECUTABLE')
        if not executable:raise ValueError('Explicitly built C++ executable required')
        if method=='TI' and selection=='adaptive':
            from .vibronic_native import native_adaptive_operator_spectrum
            return native_adaptive_operator_spectrum(model,constants,forms,gradients=gradients,
                executable=executable,kind=kind,temperature_K=temperature_K,
                maximum_initial_quanta=maximum_initial_quanta,maximum_final_quanta=maximum_final_quanta,
                maximum_class=maximum_class,maximum_states=maximum_states,maximum_transitions=maximum_transitions,
                maximum_integrals=maximum_integrals,workers=workers,target_fraction=target_fraction,
                selection_batch_size=selection_batch_size,selection_maximum_per_mode_quanta=selection_maximum_per_mode_quanta,
                checkpoint_path=checkpoint_path,resume=resume,checkpoint_interval_batches=checkpoint_interval_batches,
                checkpoint_interval_seconds=checkpoint_interval_seconds,selection_proposal=selection_proposal,
                selection_energy_bias_per_cm=selection_energy_bias_per_cm,selection_channel_weights=selection_channel_weights,
                spectral_target=spectral_target,assignment_report_path=selection_assignment_report_path)
        if selection!='enumerate' or checkpoint_path is not None or resume:
            raise NotImplementedError('Native adaptive selection requires its C++ checkpoint/selector path')
        return native_harmonic_operator_spectrum(model,constants,forms,gradients=gradients,
            executable=executable,method=method,kind=kind,temperature_K=temperature_K,
            maximum_initial_quanta=maximum_initial_quanta,maximum_final_quanta=maximum_final_quanta,
            maximum_class=maximum_class,maximum_states=maximum_states,maximum_transitions=maximum_transitions,
            maximum_integrals=maximum_integrals,time_step_fs=time_step_fs,time_points=time_points,
            gaussian_sigma_cm=gaussian_sigma_cm,lorentz_hwhm_cm=lorentz_hwhm_cm,workers=workers)
    model, direction, c, h, w, g, constant, linear = _bank(model, constants, gradients, forms, kind)
    workers = worker_count(workers)
    q = harmonic_thermal_factors(model.initial_frequencies_cm, temperature_K)
    gram = np.outer(c, c.conj())+(h*((1+q)/(2*(1-q)))[None, :]) @ h.conj().T
    rules = np.einsum('jab,ab->j', w, gram).real
    norm_rule = float(np.trace(gram).real)
    common = dict(schema='matrix.sati.operator_spectrum.v1', method=method, kind=kind,
                  origin_cm=direction*model.origin_cm, sum_rules=rules, bank_norm_sum_rule=norm_rule)
    if method == 'TI' and selection == 'adaptive':
        from .vibronic_operator_adaptive import adaptive_operator_ti
        return adaptive_operator_ti(model, direction, c, h, w, g, constant, linear, common,
            temperature_K=temperature_K, maximum_initial_quanta=maximum_initial_quanta,
            maximum_final_quanta=maximum_final_quanta, maximum_class=maximum_class,
            maximum_states=maximum_states, maximum_transitions=maximum_transitions,
            maximum_integrals=maximum_integrals, workers=workers, target_fraction=target_fraction,
            selection_batch_size=selection_batch_size,
            selection_maximum_per_mode_quanta=selection_maximum_per_mode_quanta,
            checkpoint_path=checkpoint_path, resume=resume,
            checkpoint_interval_batches=checkpoint_interval_batches, checkpoint_interval_seconds=checkpoint_interval_seconds,
            selection_proposal=selection_proposal,selection_energy_bias_per_cm=selection_energy_bias_per_cm,
            selection_channel_weights=selection_channel_weights,spectral_target=spectral_target,assignment_report_path=selection_assignment_report_path)
    if selection_proposal!='gaussian_c2':
        raise ValueError('selection_proposal requires adaptive TI')
    if selection != 'enumerate' or checkpoint_path is not None or resume:
        raise ValueError('adaptive selection and checkpoints require method=TI, selection=adaptive')
    if method == 'TI':
        for value in (maximum_transitions, maximum_integrals):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('positive integer transition/integral budgets required')
        n = len(q)
        initial = _occupation_array(n, 0 if temperature_K == 0 else maximum_initial_quanta, maximum_class, maximum_states, workers)
        final = _occupation_array(n, maximum_final_quanta, maximum_class, maximum_states, workers)
        initial = initial[np.argsort(initial @ model.initial_frequencies_cm, kind='stable')]
        final = final[np.argsort(final @ model.final_frequencies_cm, kind='stable')]
        populations = np.prod(1-q)*np.prod(q[None, :]**initial, axis=1)
        live = np.flatnonzero(populations > 0)
        count = len(live)*len(final)
        if count > maximum_transitions:
            raise MemoryError('operator bank transition budget exceeded; reduce TI space or use TD')
        ii = np.repeat(live, len(final)); ff = np.tile(np.arange(len(final)), len(live))
        states = np.column_stack([initial[ii], final[ff]])
        if g['log_ground'] < np.log(np.finfo(float).tiny)+4:
            raise FloatingPointError('operator bank TI overlap underflows; use TD')
        ids, values = _integral_dag(states, g['matrix'], g['linear'], linear, maximum_integrals, np.exp(g['log_ground']))
        amplitudes = np.empty((count, len(c)), complex)
        parallel_rows(_operator_rows, count, workers, states, ids, values, constant, linear, amplitudes)
        strengths = np.einsum('ta,jab,tb->jt', amplitudes, w, amplitudes.conj(), optimize=True).real*populations[ii]
        norm = float(np.sum(abs(amplitudes)**2*populations[ii, None]))
        missing = max(0., norm_rule-norm)
        if norm > norm_rule+1e-8*max(norm_rule, 1.):
            raise RuntimeError('operator bank violates positive total-strength sum rule')
        bounds = np.max(abs(np.linalg.eigvalsh(w)), axis=1)*missing
        return dict(**common, wavenumbers_cm=direction*(model.origin_cm+final[ff] @ model.final_frequencies_cm-initial[ii] @ model.initial_frequencies_cm),
                    strengths=strengths, initial_quanta=initial[ii], final_quanta=final[ff],
                    diagnostics=dict(cached_integrals=len(values), evaluated_transitions=count,
                        initial_population_retained=float(populations.sum()), missing_bank_norm=missing,
                        observable_omission_bounds=bounds, workers=workers, finite_state_sum_used=True))
    if method != 'TD':
        raise ValueError('method must be TI or TD')
    if (isinstance(time_points, bool) or not isinstance(time_points, int) or not 3 <= time_points <= 131073
            or not np.isfinite(time_step_fs) or time_step_fs <= 0
            or not np.isfinite(gaussian_sigma_cm+lorentz_hwhm_cm)
            or min(gaussian_sigma_cm, lorentz_hwhm_cm) < 0 or gaussian_sigma_cm+lorentz_hwhm_cm <= 0):
        raise ValueError('positive time grid and nonnegative nonzero broadening required')
    for value in (chunk_size, memory_budget_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('positive integer chunk and memory budgets required')
    times = np.arange(time_points)*time_step_fs
    active = np.flatnonzero(np.r_[q, np.ones(len(q))] != 0)
    prefactors = np.r_[q, np.ones(len(q))][active]
    frequencies = np.r_[-model.initial_frequencies_cm, model.final_frequencies_cm][active]
    r = g['linear'][active]; matrix = g['matrix'][np.ix_(active, active)]
    linear = linear[:, active]; d = len(active); k = len(c)
    # Exact QR factorization, without singular-value cutoffs: tensor channels
    # never force more than d projected solves; pure FC needs just one RHS.
    if np.any(linear):
        projection, recovery = np.linalg.qr(linear.conj().T, mode='reduced')
    else:
        projection = np.empty((d, 0), complex); recovery = np.empty((0, k), complex)
    projected_form = np.einsum('jab,ad,rb->jdr', w, linear, recovery, optimize=True)
    rhs_count = 1+projection.shape[1]
    per_point = 16*(12*d*d+6*d*rhs_count+6*k+len(w))
    capacity = memory_budget_bytes//per_point
    if capacity < 1:
        raise MemoryError('operator bank Gaussian workspace exceeds memory budget')
    workers = min(workers, capacity, (time_points+chunk_size-1)//chunk_size)
    chunk_size = min(chunk_size, capacity//workers)
    correlations = np.empty((len(w), time_points), complex)
    def evaluate(start):
        stop = min(start+chunk_size, time_points)
        tau = direction*PHASE_PER_CM_FS*times[start:stop]
        z = prefactors[None, :]*np.exp(-1j*tau[:, None]*frequencies)
        product = (z[:, :, None]*matrix.conj()[None, :, :]*z[:, None, :]) @ matrix
        eigen = np.linalg.eigvals(product)
        if np.any(abs(eigen) > 1+1e-7) or np.min(abs(1-eigen)) < 1e-12:
            raise FloatingPointError('ill-conditioned thermal operator contraction')
        rhs = z*r.conj()+z*np.einsum('ij,tj->ti', matrix.conj(), z*r)
        solved = np.linalg.solve(np.eye(d)[None, :, :]-product,
                    np.concatenate([rhs[:, :, None], z[:, :, None]*projection[None, :, :]], axis=2))
        m1 = solved[:, :, 0]; m2 = z*r+z*np.einsum('ij,tj->ti', matrix, m1)
        base = np.exp(2*g['log_ground']+np.log1p(-q).sum()-.5*np.log1p(-eigen).sum(axis=1)
                      +.5*np.einsum('i,ti->t', r, m1)+.5*np.einsum('i,ti->t', r.conj(), m2))
        left = constant+np.einsum('ad,td->ta', linear, m1)
        right = constant.conj()+np.einsum('ad,td->ta', linear.conj(), m2)
        correlations[:, start:stop] = base[None, :]*(np.einsum('jab,ta,tb->jt', w, left, right, optimize=True)
                  +np.einsum('jdr,tdr->jt', projected_form, solved[:, :, 1:], optimize=True))
    with threadpool_limits(limits=1, user_api='blas'):
        if workers == 1:
            for start in range(0, time_points, chunk_size): evaluate(start)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                starts = iter(range(0, time_points, chunk_size))
                while batch := list(islice(starts, workers)):
                    for job in [pool.submit(evaluate, start) for start in batch]: job.result()
    if np.max(abs(correlations[:, 0]-rules)) > 1e-7*max(norm_rule, 1.):
        raise RuntimeError('operator bank zero-time sum rule failed')
    tau = PHASE_PER_CM_FS*times
    signal = correlations*np.exp(-lorentz_hwhm_cm*tau-.5*(gaussian_sigma_cm*tau)**2)
    length = 2*(time_points-1); spacing = PHASE_PER_CM_FS*time_step_fs
    density = fft.fftshift(fft.hfft(signal.conj(), n=length, axis=-1,
                   workers=min(worker_count(fft_workers), len(w))), axes=-1)*spacing/(2*np.pi)
    axis = fft.fftshift(2*np.pi*fft.fftfreq(length, d=spacing))+direction*model.origin_cm
    tail = np.max(abs(signal[:, -min(16, time_points):]), axis=1)/max(norm_rule, np.finfo(float).tiny)
    return dict(**common, wavenumbers_cm=axis, densities_per_cm=density, times_fs=times,
                correlations=correlations*np.exp(-1j*direction*model.origin_cm*tau),
                diagnostics=dict(finite_state_sum_used=False, workers=workers, linear_solve_rhs=rhs_count,
                    active_gaussian_dimension=d, estimated_parallel_workspace_bytes=workers*chunk_size*per_point,
                    time_window_tail_relative_bank_norm=tail, clipped_signed_density=False,
                    scope='tail diagnostic is not time-step or aliasing convergence'))


__all__ = ['harmonic_operator_spectrum']
