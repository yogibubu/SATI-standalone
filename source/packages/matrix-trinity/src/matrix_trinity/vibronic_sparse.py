"""Sparse bare Hamiltonians, positive filters and reusable spectral shortcuts.

A Golden Rule contribution is <v|delta_sigma(H-E)|v>. Applying an
approximation to sqrt(delta_sigma) avoids a complete final eigensystem and
keeps the computed contribution nonnegative. No harmonic basis rows or force
constants are selected by their coupling magnitude.
"""
from array import array
from dataclasses import dataclass
from math import comb
from functools import cached_property
import hashlib
import json
import sqlite3

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from scipy.fft import dct
from scipy.optimize import brentq
from scipy.sparse import csc_matrix, csr_matrix, issparse

from .curvilinear_vpt2 import (
    _canonical_cubic_terms, _canonical_quartic_terms, _compiled_operator_couplings,
)
from .vibronic_variational import _confinement
from .vibronic_vpt2 import SAMVPT2Field, _positive_integer


@dataclass(frozen=True)
class SparseVibrationalHamiltonian:
    basis: np.ndarray
    frequencies_cm: np.ndarray
    matrix_cm: csr_matrix
    diagnostics: dict

    def __post_init__(self):
        basis = np.asarray(self.basis)
        frequencies = np.asarray(self.frequencies_cm, float)
        if (basis.ndim != 2 or basis.dtype.kind not in 'iu' or np.any(basis < 0)
                or basis.shape[1] != len(frequencies) or not len(basis)
                or np.any(frequencies <= 0) or not np.all(np.isfinite(frequencies))
                or len(np.unique(basis, axis=0)) != len(basis)):
            raise ValueError('unique nonnegative occupation rows and positive frequencies required')
        matrix = _checked_matrix(self.matrix_cm)
        if matrix.shape[0] != len(basis):
            raise ValueError('Hamiltonian and basis dimensions differ')
        for name, value in (('basis', basis), ('frequencies_cm', frequencies)):
            value = value.copy(); value.setflags(write=False)
            object.__setattr__(self, name, value)
        matrix = matrix.copy()
        for value in (matrix.data, matrix.indices, matrix.indptr): value.setflags(write=False)
        object.__setattr__(self, 'matrix_cm', matrix)

    @cached_property
    def dense_matrix_view(self):
        """Read-only zero-copy view when the immutable CSR is explicitly full."""
        from matrix_core.sparse_layout import dense_csr_view
        return dense_csr_view(self.matrix_cm)

    @cached_property
    def spectral_interval_cm(self):
        """Scaled Gershgorin enclosure, prepared once for this immutable matrix."""
        return _spectral_interval(self.matrix_cm)

    @cached_property
    def fingerprint(self):
        from .vibronic_spectral_cache import hamiltonian_fingerprint
        return hamiltonian_fingerprint(self)


@njit(cache=True,nogil=True)
def _dense_hermitian_errors(matrix):
    """Visit each symmetric pair once in cache-sized tiles, with all checks."""
    n=len(matrix);scale=1.;error=0.
    for first in range(0,n,32):
        for second in range(first,n,32):
            for row in range(first,min(first+32,n)):
                for column in range(max(row,second),min(second+32,n)):
                    a=matrix[row,column];b=matrix[column,row]
                    if not np.isfinite(a) or not np.isfinite(b):return False,scale,error
                    scale=max(scale,abs(a),abs(b))
                    error=max(error,abs(a-np.conj(b)))
    return True,scale,error


def _checked_matrix(matrix):
    if not issparse(matrix) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError('nonempty square sparse Hermitian Hamiltonian required')
    matrix = csr_matrix(matrix)
    from matrix_core.sparse_layout import dense_csr_view
    dense = dense_csr_view(matrix)
    if dense is not None:
        # Check the identical full Hermitian contract without transpose/CSR
        # conversion and whole-matrix finite/absolute-value temporaries.
        finite,scale,error = _dense_hermitian_errors(dense)
        if not finite:raise ValueError('Hamiltonian contains nonfinite entries')
        if error>1e-12*scale:raise ValueError('Hamiltonian must be Hermitian')
        return matrix
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError('Hamiltonian contains nonfinite entries')
    difference = matrix-matrix.conj().T
    scale = max(float(np.max(abs(matrix.data), initial=0)), 1.)
    if np.max(abs(difference.data), initial=0) > 1e-12*scale:
        raise ValueError('Hamiltonian must be Hermitian')
    return matrix


def build_sparse_quartic_hamiltonian(field, *, maximum_total_quanta=24,
        maximum_harmonic_energy_cm=None, maximum_states=100000,
        maximum_nonzeros=10000000, maximum_sparse_bytes=512*1024**2,
        prepared_harmonic_diagonal_cm=None):
    """Build the complete declared finite HO Hamiltonian directly in sparse form.

    Same bare potential and confinement requirement as the dense solver.
    Bounds restrict HO excitation energy and shell number, not eigenenergies.
    The memory guard reserves 48 bytes per nonzero plus row/basis overhead for
    construction, validation and CSR conversion; it is not a process RSS limit.
    """
    if not isinstance(field, SAMVPT2Field): raise TypeError('SAMVPT2Field required')
    if np.any(field.metric_first_cm) or np.any(field.metric_second_cm):
        raise ValueError('sparse bare solver supports potential-only modal fields')
    for name, value in (('maximum_total_quanta', maximum_total_quanta),
                       ('maximum_states', maximum_states), ('maximum_nonzeros', maximum_nonzeros),
                       ('maximum_sparse_bytes', maximum_sparse_bytes)):
        _positive_integer(value, name)
    if maximum_harmonic_energy_cm is not None and (not np.isfinite(maximum_harmonic_energy_cm)
            or maximum_harmonic_energy_cm <= 0): raise ValueError('positive finite energy bound required')
    confinement = _confinement(field)
    n = len(field.frequencies_cm)
    if maximum_harmonic_energy_cm is None and comb(n+maximum_total_quanta,n) > maximum_states:
        raise MemoryError('sparse variational basis exceeds its declared state budget')
    rows = []
    def visit(prefix, mode, remaining, energy):
        if mode == n:
            rows.append(tuple(prefix))
            if len(rows) > maximum_states: raise MemoryError('sparse variational basis exceeds state budget')
            return
        limit = remaining
        if maximum_harmonic_energy_cm is not None:
            limit = min(limit, int(np.floor((maximum_harmonic_energy_cm-energy)/field.frequencies_cm[mode]+1e-12)))
        for occupation in range(limit+1):
            visit(prefix+[occupation],mode+1,remaining-occupation,
                  energy+occupation*field.frequencies_cm[mode])
    visit([],0,maximum_total_quanta,0.)
    rows.sort(key=lambda row:(sum(row),row))
    basis = np.asarray(rows,np.int64)
    harmonic_diagonal = None
    if prepared_harmonic_diagonal_cm is not None:
        harmonic_diagonal = np.asarray(prepared_harmonic_diagonal_cm,float)
        if harmonic_diagonal.shape!=(len(basis),) or not np.all(np.isfinite(harmonic_diagonal)):
            raise ValueError('one finite prepared harmonic energy per basis row required')
        reference = np.array([float((row+.5)@field.frequencies_cm+field.constant_shift_cm) for row in basis])
        if np.any(abs(harmonic_diagonal-reference)>64*np.finfo(float).eps*np.maximum(1.,abs(reference))):
            raise ValueError('prepared harmonic energies differ beyond arithmetic precision')
    lookup = {state:i for i,state in enumerate(rows)}
    overhead = len(basis)*(128+32*n)
    if overhead > maximum_sparse_bytes: raise MemoryError('sparse basis exceeds memory budget')
    terms = (_canonical_cubic_terms(field.cubic_cm,field.metric_first_cm,0.)
             +_canonical_quartic_terms(field.quartic_cm,field.metric_second_cm))
    values, indices, pointers = array('d'), array('q'), array('q',[0])
    for source,state in enumerate(rows):
        entries = _compiled_operator_couplings(state,terms)
        entries[state] = entries.get(state,0.)+(float(harmonic_diagonal[source]) if harmonic_diagonal is not None
            else float((basis[source]+.5)@field.frequencies_cm+field.constant_shift_cm))
        for target,value in entries.items():
            row = lookup.get(target)
            if row is None or value == 0: continue
            if complex(value).imag != 0: raise ValueError('potential-only Hamiltonian is not exactly real')
            if len(values)+1 > maximum_nonzeros or overhead+48*(len(values)+1) > maximum_sparse_bytes:
                raise MemoryError('sparse Hamiltonian exceeds declared nonzero or memory budget')
            indices.append(row); values.append(complex(value).real)
        pointers.append(len(values))
    matrix = csc_matrix((np.frombuffer(values,dtype=np.float64),np.frombuffer(indices,dtype=np.int64),
                         np.frombuffer(pointers,dtype=np.int64)),shape=(len(rows),)*2).tocsr()
    matrix.sort_indices()
    return SparseVibrationalHamiltonian(basis,field.frequencies_cm,matrix,dict(
        schema='matrix.trinity.sparse_quartic_hamiltonian.v1',hamiltonian='H0+V3+V4',
        confinement=confinement,maximum_total_quanta=maximum_total_quanta,
        maximum_harmonic_energy_cm=maximum_harmonic_energy_cm,basis_size=len(basis),
        nonzeros=matrix.nnz,sparse_storage_bytes=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes,
        dense_real_matrix_bytes=8*len(basis)**2,quartic_coverage=field.quartic_coverage,
        coefficient_pruning_used=False,hdc_completion_used=False,infinite_basis_convergence_certified=False))


def _spectral_interval(matrix, *, scaling_steps=128):
    """Enclose a Hermitian spectrum by positive diagonal similarities.

    For every positive vector w, Gershgorin applied to D^-1 H D gives
    min_i(H_ii - (|off(H)|w)_i/w_i) <= lambda <=
    max_i(H_ii + (|off(H)|w)_i/w_i). Power iteration only proposes weights;
    every iterate supplies a valid bound, independently of convergence.
    The best bounds include the original unscaled enclosure. Outward padding
    covers ordinary arithmetic rounding, not verified interval arithmetic.
    """
    diagonal = matrix.diagonal().real
    off = abs(matrix).tocsr()
    off.setdiag(0.)
    off.eliminate_zeros()
    radii = np.asarray(off.sum(axis=1)).ravel()
    lower,upper = float(np.min(diagonal-radii)),float(np.max(diagonal+radii))
    if off.nnz and scaling_steps:
        for sign in (1.,-1.):
            signed_diagonal = sign*diagonal
            shift = max(0.,float(-np.min(signed_diagonal)))+1.
            weights = np.ones(len(diagonal))
            best = upper if sign > 0 else -lower
            for _ in range(scaling_steps):
                product = off@weights + signed_diagonal*weights
                bound = float(np.max(product/weights))
                if np.isfinite(bound): best = min(best,bound)
                shifted = product+shift*weights
                scale = float(np.max(shifted))
                if not np.isfinite(scale) or scale <= 0: break
                weights = np.maximum(shifted/scale,1e-12)
            if sign > 0: upper = best
            else: lower = -best
    padding = 32*np.finfo(float).eps*max(float(np.max(abs(diagonal)+radii)),1.)
    return lower-padding,upper+padding


def _hamiltonian_interval(hamiltonian, matrix):
    return (hamiltonian.spectral_interval_cm if isinstance(hamiltonian,SparseVibrationalHamiltonian)
            else _spectral_interval(matrix))


def _log_chebyshev_error(degree, ratio, log_peak):
    # Bernstein ellipse: |Im(E)| <= radius*sinh(eta).  For sqrt(Gaussian)
    # M <= sqrt(peak)*exp((radius*sinh(eta))**2/(4*sigma**2)).
    # Degree-n Lobatto interpolation error <= 4*M*rho**(-n)/(rho-1).
    def derivative(eta):
        return ratio**2*np.sinh(eta)*np.cosh(eta)-degree-1/(-np.expm1(-eta))
    upper = 1.
    while derivative(upper) < 0 and upper < 20: upper *= 2
    eta = brentq(derivative,1e-14,upper)
    return np.log(4.)+log_peak+.5*(ratio*np.sinh(eta))**2-degree*eta-np.log(np.expm1(eta))



@njit(cache=True, parallel=True)
def _sparse_clenshaw_columns(data, indices, indptr, midpoint, radius, coefficients, source):
    """Share CSR reads among four scalar accumulators, preserving sum order."""
    count, columns = source.shape
    next1, next2, current = np.zeros_like(source), np.zeros_like(source), np.empty_like(source)
    for k in range(len(coefficients)-1,-1,-1):
        scale = 2. if k else 1.
        for row in prange(count):
            for group in range(columns//4):
                column = 4*group
                value0 = 0.*source[row,column+0]
                value1 = 0.*source[row,column+1]
                value2 = 0.*source[row,column+2]
                value3 = 0.*source[row,column+3]
                for entry in range(indptr[row],indptr[row+1]):
                    weight, neighbour = data[entry], indices[entry]
                    value0 += weight*next1[neighbour,column+0]
                    value1 += weight*next1[neighbour,column+1]
                    value2 += weight*next1[neighbour,column+2]
                    value3 += weight*next1[neighbour,column+3]
                current[row,column+0] = (scale*(value0-midpoint*next1[row,column+0])/radius
                    -next2[row,column+0]+coefficients[k,column+0]*source[row,column+0])
                current[row,column+1] = (scale*(value1-midpoint*next1[row,column+1])/radius
                    -next2[row,column+1]+coefficients[k,column+1]*source[row,column+1])
                current[row,column+2] = (scale*(value2-midpoint*next1[row,column+2])/radius
                    -next2[row,column+2]+coefficients[k,column+2]*source[row,column+2])
                current[row,column+3] = (scale*(value3-midpoint*next1[row,column+3])/radius
                    -next2[row,column+3]+coefficients[k,column+3]*source[row,column+3])
            remainder = columns%4
            column = 4*(columns//4)
            if remainder >= 2:
                value0 = 0.*source[row,column]
                value1 = 0.*source[row,column+1]
                value2 = 0.*source[row,column]
                for entry in range(indptr[row],indptr[row+1]):
                    weight, neighbour = data[entry], indices[entry]
                    value0 += weight*next1[neighbour,column]
                    value1 += weight*next1[neighbour,column+1]
                    if remainder == 3:
                        value2 += weight*next1[neighbour,column+2]
                current[row,column+0] = (scale*(value0-midpoint*next1[row,column+0])/radius
                    -next2[row,column+0]+coefficients[k,column+0]*source[row,column+0])
                current[row,column+1] = (scale*(value1-midpoint*next1[row,column+1])/radius
                    -next2[row,column+1]+coefficients[k,column+1]*source[row,column+1])
                if remainder == 3:
                    current[row,column+2] = (scale*(value2-midpoint*next1[row,column+2])/radius
                        -next2[row,column+2]+coefficients[k,column+2]*source[row,column+2])
            elif remainder == 1:
                value = 0.*source[row,column]
                for entry in range(indptr[row],indptr[row+1]):
                    value += data[entry]*next1[indices[entry],column]
                current[row,column] = (scale*(value-midpoint*next1[row,column])/radius
                    -next2[row,column]+coefficients[k,column]*source[row,column])
        next2,next1,current = next1,current,next2
    return next1


def _filter_columns(matrix, midpoint, radius, coefficients, source, workers):
    _positive_integer(workers,'workers')
    previous = get_num_threads()
    try:
        set_num_threads(workers)
        return _sparse_clenshaw_columns(matrix.data,matrix.indices,matrix.indptr,midpoint,radius,
            np.ascontiguousarray(coefficients),np.ascontiguousarray(source))
    finally:
        set_num_threads(previous)


def sparse_gaussian_spectral_density(hamiltonian, wavepacket, target_energy_cm, *,
        gaussian_sigma_cm, absolute_tolerance=1e-12, maximum_degree=32768,
        maximum_matvec_work=20000000000, workers=1, prepared_spectrum=None):
    """Evaluate <v|Gaussian(H-target)|v> without requiring final eigenstates.

    The nonnegative result is ||p(H)v||^2, where p interpolates sqrt(Gaussian).
    A Bernstein-ellipse bound controls the polynomial error over a Gershgorin
    enclosure of the *entire finite Hamiltonian*. It does not discard states
    outside the target window. The analytic bound excludes floating-point
    roundoff and all finite-HO-basis and PES errors. Only a Gaussian is supported.
    An explicit work budget is checked before polynomial matrix-vector products.
    An optional prepared complete-lowest eigenwindow replaces the recurrence
    only where its omitted Gaussian tails satisfy the same requested tolerance.
    """
    matrix = (hamiltonian.matrix_cm if isinstance(hamiltonian,SparseVibrationalHamiltonian)
              else _checked_matrix(hamiltonian))
    vector = np.asarray(wavepacket)
    if (vector.ndim != 1 or len(vector) != matrix.shape[0] or not np.all(np.isfinite(vector))
            or not np.isfinite(target_energy_cm) or not np.isfinite(gaussian_sigma_cm)
            or gaussian_sigma_cm <= 0 or not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0):
        raise ValueError('finite wavepacket, target, positive Gaussian sigma and absolute tolerance required')
    for name,value in (('maximum_degree',maximum_degree),('maximum_matvec_work',maximum_matvec_work),('workers',workers)):
        _positive_integer(value,name)
    if np.iscomplexobj(vector) and not np.any(vector.imag): vector = vector.real
    vector = vector.astype(np.result_type(matrix.dtype,vector.dtype,np.float64),copy=False)
    norm2 = float(np.vdot(vector,vector).real)
    if not np.isfinite(norm2): raise FloatingPointError('wavepacket norm overflow')
    if prepared_spectrum is not None and norm2:
        result = sparse_gaussian_spectral_density_batch(hamiltonian,vector[:,None],
            np.array([target_energy_cm]),gaussian_sigma_cm=gaussian_sigma_cm,
            density_tolerance_per_strength=absolute_tolerance/norm2,
            maximum_degree=maximum_degree,maximum_matvec_work=maximum_matvec_work,
            workers=workers,prepared_spectrum=prepared_spectrum)
        return dict(density_per_cm=float(result['densities_per_cm'][0]),
            absolute_error_bound_per_cm=float(result['absolute_error_bounds_per_cm'][0]),
            diagnostics=result['diagnostics'])
    lower,upper = _hamiltonian_interval(hamiltonian,matrix)
    midpoint,radius = (upper+lower)/2,(upper-lower)/2
    peak = 1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    distance = max(lower-target_energy_cm,target_energy_cm-upper,0.)
    zero_bound = norm2*peak*np.exp(-.5*(distance/gaussian_sigma_cm)**2)
    diagnostics = dict(schema='matrix.trinity.sparse_gaussian_density.v1',spectral_interval_cm=[lower,upper],
        input_norm_squared=norm2,polynomial_degree=0,matvec_count=0,estimated_nonzero_products=0,
        final_eigenvectors_computed=False,polynomial_error_bound_includes_roundoff=False,
        finite_basis_convergence_certified=False)
    if norm2 == 0 or zero_bound <= absolute_tolerance:
        return dict(density_per_cm=0.,absolute_error_bound_per_cm=zero_bound,diagnostics=diagnostics)
    if np.count_nonzero(matrix.data) == np.count_nonzero(matrix.diagonal()):
        density = float(np.sum(abs(vector)**2*peak*np.exp(-.5*((matrix.diagonal().real-target_energy_cm)/gaussian_sigma_cm)**2)))
        diagnostics['diagonal_evaluation_used'] = True
        return dict(density_per_cm=density,absolute_error_bound_per_cm=0.,diagnostics=diagnostics)
    # Solve norm2*epsilon*(2*sqrt(peak)+epsilon) <= tolerance stably.
    threshold = (absolute_tolerance/norm2)/(np.sqrt(peak+absolute_tolerance/norm2)+np.sqrt(peak))
    log_target = np.log(threshold)
    ratio = radius/(np.sqrt(2)*gaussian_sigma_cm)
    degree = min(32,maximum_degree)
    while _log_chebyshev_error(degree,ratio,.5*np.log(peak)) > log_target:
        if degree >= maximum_degree:
            raise MemoryError('Gaussian filter exceeds polynomial degree budget; no density inferred')
        degree = min(2*degree,maximum_degree)
    low,high = 1,degree
    while low < high:
        middle = (low+high)//2
        if _log_chebyshev_error(middle,ratio,.5*np.log(peak)) <= log_target: high = middle
        else: low = middle+1
    degree = high
    work = (degree+1)*matrix.nnz
    if work > maximum_matvec_work:
        raise MemoryError(f'Gaussian filter requires {degree} sparse matvecs ({work} nonzero products); work budget exceeded')
    nodes = np.cos(np.pi*np.arange(degree+1)/degree)
    samples = np.sqrt(peak)*np.exp(-.25*((midpoint+radius*nodes-target_energy_cm)/gaussian_sigma_cm)**2)
    coefficients = dct(samples,type=1)/degree
    coefficients[[0,-1]] *= .5
    filtered = _filter_columns(matrix,midpoint,radius,coefficients[:,None],vector[:,None],workers)[:,0]
    density = float(np.vdot(filtered,filtered).real)
    if not np.isfinite(density): raise FloatingPointError('nonfinite Gaussian spectral density')
    epsilon = np.exp(_log_chebyshev_error(degree,ratio,.5*np.log(peak)))
    error = norm2*epsilon*(2*np.sqrt(peak)+epsilon)
    diagnostics.update(polynomial_degree=degree,matvec_count=degree+1,
        estimated_nonzero_products=(degree+1)*matrix.nnz,
        polynomial_sqrt_filter_uniform_error=epsilon,requested_absolute_tolerance=absolute_tolerance)
    return dict(density_per_cm=density,absolute_error_bound_per_cm=error,diagnostics=diagnostics)


__all__ = ['SparseVibrationalHamiltonian','build_sparse_quartic_hamiltonian','sparse_gaussian_spectral_density']


class _ContractedSpectatorCache:
    """Bounded shared packets; never retain the much larger overlap matrix."""
    def __init__(self, owner, maximum_bytes):
        from collections import OrderedDict
        self.owner, self.maximum_bytes = owner, maximum_bytes
        self.values = OrderedDict()
        self.bytes = self.hits = self.evaluations = 0

    def evaluate(self, occupations):
        key = tuple(map(int, occupations))
        value = self.values.pop(key, None)
        if value is not None:
            self.values[key] = value
            self.hits += 1
            return value
        owner = self.owner
        packet, harmonic = owner._unshared_contracted_wavepacket(np.asarray(occupations, np.int64))
        packet.setflags(write=False); harmonic.setflags(write=False)
        value = (packet, harmonic)
        size = packet.nbytes + harmonic.nbytes
        self.evaluations += 1
        if size <= self.maximum_bytes:
            while self.values and self.bytes + size > self.maximum_bytes:
                _, removed = self.values.popitem(last=False)
                self.bytes -= sum(array.nbytes for array in removed)
            self.values[key] = value
            self.bytes += size
        return value


class SparseSpectatorObservable:
    """Gaussian Golden Rule callbacks with optional reusable spectral windows.

    ``initial`` is the emitting state for emission (the absorbing state for
    absorption), already embedded in the corresponding oriented harmonic basis.
    ``final_hamiltonian`` contains only the final active coordinates. All other
    final coordinates remain harmonic spectators; the full Gaussian generator
    retains Duschinsky rotations and the specified numerical threshold.

    Chebyshev is the default. The optional Lanczos route is experimental and
    returns successive-difference estimates rather than a global error bound.
    The Chebyshev filter tolerance is uniform per unit wavepacket strength. An analytic
    Mehler sum of that strength over *all* final HO occupations gives a global
    filter error bound, avoiding multiplication of a per-configuration tolerance
    by an unknown number of spectator states. It excludes Gaussian-threshold,
    roundoff, sampling, active-space, finite-HO-basis and potential-model errors.
    A prepared eigenwindow accelerates covered energies at this identical error
    contract; uncovered targets still use the bounded Chebyshev filter. Scalar
    cache identity is unchanged because Hamiltonian, observable and tolerance
    are unchanged. Diagnostics distinguish the actual evaluation backend.
    """
    def __init__(self, model, initial, final_hamiltonian, active_final_modes, coupling_hartree, *,
            kind='emission', gaussian_sigma_cm=21.23, total_filter_tolerance_s_1=1.,
            matrix_zero_tolerance=0., maximum_grid_entries=8000000,
            maximum_harmonic_pairs=2000000, cache_gaussian_components=False,
            maximum_moment_grid_entries=8000000, maximum_degree=32768,
            maximum_matvec_work=20000000000, maximum_cached_spectators=2048,
            spectral_method='chebyshev', lanczos_relative_tolerance=1e-6, maximum_lanczos_steps=8192, maximum_batch_size=8, workers=1, persistent_cache_path=None, prepared_gaussian_generator=None, prepared_initial_strength_moment=None, prepared_spectrum=None):
        from collections import OrderedDict
        from .vibronic_harmonic import orient_harmonic_model
        from .vibronic_variational import gaussian_number_moment
        from .vibronic_spectator_rates import _SpectatorOverlapKernel
        from .vibronic_rates import GOLDEN_RULE_FACTOR
        if spectral_method not in ('chebyshev','lanczos'): raise ValueError('spectral_method must be chebyshev or lanczos')
        if not np.isfinite(lanczos_relative_tolerance) or lanczos_relative_tolerance<=0:raise ValueError('positive Lanczos relative tolerance required')
        _positive_integer(maximum_lanczos_steps,'maximum_lanczos_steps')
        self.spectral_method,self.lanczos_relative_tolerance,self.maximum_lanczos_steps = spectral_method,lanczos_relative_tolerance,maximum_lanczos_steps
        _positive_integer(workers,'workers')
        self.workers = workers
        _positive_integer(maximum_batch_size,'maximum_batch_size')
        self.maximum_batch_size = maximum_batch_size
        self.total_matrix_passes = 0
        self.maximum_used_lanczos_steps = 0
        self.last_spectral_diagnostics = None
        if not isinstance(final_hamiltonian,SparseVibrationalHamiltonian):
            raise TypeError('SparseVibrationalHamiltonian required')
        if prepared_spectrum is not None:
            from .vibronic_spectral_cache import PreparedGaussianSpectrum,PreparedBlockGaussianSpectrum
            from .vibronic_energy_windows import PreparedEnergyWindows
            if (not isinstance(prepared_spectrum,(PreparedGaussianSpectrum,PreparedBlockGaussianSpectrum,PreparedEnergyWindows))
                    or prepared_spectrum.hamiltonian_sha256!=final_hamiltonian.fingerprint):
                raise ValueError('prepared spectrum must belong to the identical finite Hamiltonian')
            if spectral_method!='chebyshev':
                raise ValueError('prepared spectral acceleration requires the bounded Gaussian backend')
        self.prepared_spectrum = prepared_spectrum
        oriented,_ = orient_harmonic_model(model,kind)
        active = np.asarray(active_final_modes)
        n = len(oriented.final_frequencies_cm)
        if (active.ndim != 1 or active.dtype.kind not in 'iu' or not len(active)
                or np.any(active < 0) or np.any(active >= n) or len(np.unique(active)) != len(active)
                or not np.array_equal(initial.frequencies_cm,oriented.initial_frequencies_cm)
                or not np.array_equal(final_hamiltonian.frequencies_cm,oriented.final_frequencies_cm[active])):
            raise ValueError('oriented emitting/absorbing initial state and final active frequencies must match')
        coupling = np.asarray(coupling_hartree,complex)
        if coupling.ndim != 1 or not len(coupling) or not np.all(np.isfinite(coupling)):
            raise ValueError('finite constant coupling channels required')
        if (not np.isfinite(total_filter_tolerance_s_1) or total_filter_tolerance_s_1 <= 0
                or not np.isfinite(gaussian_sigma_cm) or gaussian_sigma_cm <= 0):
            raise ValueError('positive finite Gaussian sigma and total filter tolerance required')
        for name,value in (('maximum_cached_spectators',maximum_cached_spectators),
                           ('maximum_degree',maximum_degree),('maximum_matvec_work',maximum_matvec_work)):
            _positive_integer(value,name)
        self._shared_wavepackets = None
        self._shared_wavepacket_rows = None
        self.initial,self.final,self.oriented = initial,final_hamiltonian,oriented
        self.active_modes = active.copy()
        self.spectator_modes = np.setdiff1d(np.arange(n),active)
        self.frequencies_cm = oriented.final_frequencies_cm[self.spectator_modes]
        self.sigma = gaussian_sigma_cm
        self.coupling_strength = float(np.sum(abs(coupling)**2))
        self.factor = GOLDEN_RULE_FACTOR*self.coupling_strength
        self.maximum_degree,self.maximum_matvec_work = maximum_degree,maximum_matvec_work
        basis = np.zeros((len(final_hamiltonian.basis),n),np.int64)
        basis[:,active] = final_hamiltonian.basis
        self.kernel = _SpectatorOverlapKernel(oriented,initial.basis,basis,
            matrix_zero_tolerance=matrix_zero_tolerance,maximum_grid_entries=maximum_grid_entries,
            maximum_harmonic_pairs=maximum_harmonic_pairs,cache_components=cache_gaussian_components,
            prepared_generator=prepared_gaussian_generator)
        self.strength_moment = gaussian_number_moment(self.kernel.matrix,self.kernel.linear,
            np.log(self.kernel.ground),initial,maximum_grid_entries=maximum_moment_grid_entries)
        if prepared_initial_strength_moment is not None:
            prepared_moment = float(prepared_initial_strength_moment)
            actual_moment = self.strength_moment['moment']
            if (not np.isfinite(prepared_moment) or prepared_moment<=0
                    or abs(prepared_moment-actual_moment)>64*np.finfo(float).eps*max(1.,actual_moment)):
                raise ValueError('prepared initial strength differs beyond arithmetic precision')
            self.strength_moment = dict(self.strength_moment,moment=prepared_moment,
                portable_moment_roundoff_difference=actual_moment-prepared_moment)
        self.uniform_density_tolerance = (total_filter_tolerance_s_1/(self.factor*self.strength_moment['moment'])
                                          if self.factor else 1.)
        zero = np.flatnonzero(np.all(initial.basis == 0,axis=1))
        if len(zero) != 1: raise ValueError('initial basis must contain its harmonic vacuum')
        self.zero = int(zero[0])
        self.harmonic_final_energies = (basis+.5)@oriented.final_frequencies_cm
        self.harmonic_initial_energy = .5*initial.frequencies_cm.sum()
        self.spectator_zpe = .5*self.frequencies_cm.sum()
        self.cache = OrderedDict()
        self.maximum_cached_spectators = maximum_cached_spectators
        self.evaluated_spectators = self.cache_hits = self.maximum_used_degree = self.total_matvecs = 0
        self.total_filter_tolerance_s_1 = total_filter_tolerance_s_1
        self.persistent_cache = None
        self.persistent_cache_hits = self.persistent_cache_writes = 0
        digest = hashlib.sha256(b'matrix.trinity.sparse_spectator_values.v1')
        matrix = self.final.matrix_cm
        arrays = (self.kernel.matrix,self.kernel.linear,np.array([self.kernel.ground]),
                  initial.basis,initial.energies_cm,initial.eigenvectors,initial.frequencies_cm,
                  self.final.basis,self.final.frequencies_cm,matrix.data,matrix.indices,matrix.indptr,
                  self.active_modes,self.spectator_modes,
                  np.array([oriented.electronic_gap_cm,self.coupling_strength,self.sigma,
                            self.uniform_density_tolerance,self.lanczos_relative_tolerance,self.maximum_lanczos_steps]))
        digest.update(self.spectral_method.encode())
        for value in arrays:
            value = np.ascontiguousarray(value)
            digest.update(json.dumps((value.dtype.str,value.shape)).encode())
            digest.update(memoryview(value).cast('B') if value.size else b'')
        self.persistent_model_key = digest.hexdigest()
        # The optional spectrum is an accelerator of the same finite Gaussian
        # observable, accepted only at this same uniform error tolerance. It is
        # not part of the physical model/cache key, like worker and batch counts.
        if persistent_cache_path is not None:
            connection = sqlite3.connect(str(persistent_cache_path),timeout=30,isolation_level=None)
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=NORMAL')
            connection.execute('CREATE TABLE IF NOT EXISTS sparse_spectator_values_v1 '
                '(model TEXT NOT NULL, occupations BLOB NOT NULL, anharmonic REAL NOT NULL, '
                'harmonic REAL NOT NULL, filter_error REAL NOT NULL, PRIMARY KEY(model,occupations)) WITHOUT ROWID')
            self.persistent_cache = connection

    def _remember(self, key, value, *, persist=True):
        value = tuple(map(float,value))
        if not np.all(np.isfinite(value)) or min(value)<0:
            raise FloatingPointError('invalid sparse observable; cache write refused')
        self.cache[key] = value
        if len(self.cache)>self.maximum_cached_spectators:self.cache.popitem(last=False)
        if persist and self.persistent_cache is not None:
            changed = self.persistent_cache.execute(
                'INSERT OR IGNORE INTO sparse_spectator_values_v1 VALUES(?,?,?,?,?)',
                (self.persistent_model_key,np.asarray(key,dtype='<i8').tobytes(),*value)).rowcount
            self.persistent_cache_writes += int(changed>0)
        return value

    def _lookup(self, key):
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.persistent_cache is not None:
            row = self.persistent_cache.execute('SELECT anharmonic,harmonic,filter_error '
                'FROM sparse_spectator_values_v1 WHERE model=? AND occupations=?',
                (self.persistent_model_key,np.asarray(key,dtype='<i8').tobytes())).fetchone()
            if row is not None:
                self.persistent_cache_hits += 1
                return self._remember(key,row,persist=False)
        return None

    def close(self):
        if self.persistent_cache is not None:
            self.persistent_cache.close()
            self.persistent_cache = None

    def reuse_wavepackets_from(self, source, *, maximum_cache_bytes=64*1024**2):
        """Reuse exact common FC packets, including arbitrary subset/reordered rows.

        Hamiltonians and spectra may differ. Gaussian generators, initial
        coefficients, spectator axes and all retained final basis rows must
        match exactly. No amplitudes or couplings are screened for this reuse.
        Call before numerical sampling; observable cache identity is unchanged.
        """
        if not isinstance(source, SparseSpectatorObservable):
            raise TypeError('SparseSpectatorObservable source required')
        _positive_integer(maximum_cache_bytes, 'maximum_cache_bytes')
        cache = source._shared_wavepackets
        owner = source if cache is None else cache.owner
        same = (np.array_equal(self.spectator_modes, owner.spectator_modes)
                and np.array_equal(self.kernel.initial, owner.kernel.initial)
                and np.array_equal(self.initial.eigenvectors[:, 0], owner.initial.eigenvectors[:, 0])
                and self.zero == owner.zero
                and self.kernel.ground == owner.kernel.ground
                and np.array_equal(self.kernel.matrix, owner.kernel.matrix)
                and np.array_equal(self.kernel.linear, owner.kernel.linear))
        if not same:
            raise ValueError('Shared packets require identical Gaussian and initial-state definitions')
        if np.array_equal(self.kernel.final, owner.kernel.final):
            rows = slice(None)
        else:
            lookup = {row.tobytes(): index for index, row in enumerate(owner.kernel.final)}
            try: rows = np.array([lookup[row.tobytes()] for row in self.kernel.final], np.int64)
            except KeyError as error:
                raise ValueError('Shared source must contain every final basis row') from error
        if cache is None:
            cache = _ContractedSpectatorCache(owner, maximum_cache_bytes)
            source._shared_wavepackets = cache
            source._shared_wavepacket_rows = slice(None)
        elif cache.maximum_bytes != maximum_cache_bytes:
            raise ValueError('Existing shared packet cache has a different memory budget')
        self._shared_wavepackets, self._shared_wavepacket_rows = cache, rows

    def _contracted_wavepacket(self, occupations):
        if self._shared_wavepackets is not None:
            packet, harmonic = self._shared_wavepackets.evaluate(occupations)
            rows = self._shared_wavepacket_rows
            return packet[rows], harmonic[rows]
        return self._unshared_contracted_wavepacket(occupations)

    def conditional_doorways(self, states, *, maximum_workspace_bytes=256*1024**2):
        """Expose existing coupled conditional packets to the common TD engine.

        The same overlap contraction/cache and all initial coefficients are
        reused. Constant incoherent SOC channels share a spatial packet, so
        their squared norms are combined exactly. Spectator selection and
        any sampling weights belong to the caller. No scalar rate cache is
        evaluated or written by this operation.
        """
        states = np.asarray(states)
        _positive_integer(maximum_workspace_bytes, 'maximum_workspace_bytes')
        if (states.ndim != 2 or states.shape[1] != len(self.spectator_modes)
                or states.dtype.kind not in 'iu' or not len(states) or np.any(states < 0)):
            raise ValueError('nonnegative integer spectator occupation rows required')
        size = len(self.final.basis)*len(states)*np.dtype(complex).itemsize
        if size+4*len(self.final.basis)*np.dtype(complex).itemsize > maximum_workspace_bytes:
            raise MemoryError('conditional doorway batch exceeds declared workspace budget')
        packets = np.empty((len(self.final.basis), len(states)), complex)
        scale = np.sqrt(self.coupling_strength)
        for column, occupations in enumerate(states):
            packet, _ = self._contracted_wavepacket(occupations)
            packets[:, column] = scale*packet
        return dict(source_hartree=packets,
            spectator_energies_cm=(states+.5)@self.frequencies_cm,
            initial_energy_cm=float(self.initial.energies_cm[0]-self.oriented.electronic_gap_cm),
            source_coordinate_id=self.final.fingerprint,
            diagnostics=dict(initial_level=0, active_final_modes=self.active_modes.copy(),
                spectator_modes=self.spectator_modes.copy(),
                initial_coefficients_pruned=False, spatially_identical_channels_combined=True,
                scalar_observable_evaluated=False, spectator_weights_normalized=False))

    def _unshared_contracted_wavepacket(self, occupations):
        if not hasattr(self.kernel, 'prepare_contraction'):
            overlap = self.kernel.evaluate(self.spectator_modes, occupations)
            return overlap @ self.initial.eigenvectors[:, 0], overlap[:, self.zero].copy()
        contraction = getattr(self, '_initial_overlap_contraction', None)
        if contraction is None:
            coefficients = self.initial.eigenvectors[:, 0]
            if not np.any(coefficients.imag): coefficients = coefficients.real
            contraction = self.kernel.prepare_contraction(coefficients, self.zero)
            self._initial_overlap_contraction = contraction
        return contraction(self.spectator_modes, occupations)

    def _evaluate(self, occupations):
        key = tuple(map(int,occupations))
        cached = self._lookup(key)
        if cached is not None:return cached
        if self.factor == 0: return (0.,0.,0.)
        wavepacket, harmonic_amplitudes = self._contracted_wavepacket(occupations)
        norm2 = float(np.vdot(wavepacket,wavepacket).real)
        spectator_energy = float(self.frequencies_cm@occupations)
        target = (-self.oriented.electronic_gap_cm+self.initial.energies_cm[0]
                  -spectator_energy-self.spectator_zpe)
        if norm2:
            if self.spectral_method == 'chebyshev':
                density = sparse_gaussian_spectral_density(self.final,wavepacket,target,
                    gaussian_sigma_cm=self.sigma,absolute_tolerance=self.uniform_density_tolerance*norm2,
                    maximum_degree=self.maximum_degree,maximum_matvec_work=self.maximum_matvec_work,workers=self.workers,
                    prepared_spectrum=self.prepared_spectrum)
                error = density['absolute_error_bound_per_cm']
                self.maximum_used_degree = max(self.maximum_used_degree,density['diagnostics']['polynomial_degree'])
            else:
                density = lanczos_gaussian_spectral_density(self.final,wavepacket,target,
                    gaussian_sigma_cm=self.sigma,absolute_tolerance=self.uniform_density_tolerance*norm2,
                    relative_tolerance=self.lanczos_relative_tolerance,maximum_steps=self.maximum_lanczos_steps,
                    minimum_steps=min(128,self.maximum_lanczos_steps),maximum_matvec_work=self.maximum_matvec_work)
                error = density['absolute_error_estimate_per_cm']
                self.maximum_used_lanczos_steps = max(self.maximum_used_lanczos_steps,density['diagnostics']['matvec_count'])
            self.last_spectral_diagnostics = density['diagnostics']
            self.total_matvecs += density['diagnostics']['matvec_count']
            self.total_matrix_passes += density['diagnostics']['matvec_count']
            anharmonic = density['density_per_cm']
        else: anharmonic,error = 0.,0.
        mismatch = self.oriented.electronic_gap_cm+self.harmonic_final_energies-self.harmonic_initial_energy+spectator_energy
        harmonic = float(np.sum(abs(harmonic_amplitudes)**2
            *np.exp(-.5*(mismatch/self.sigma)**2)/(np.sqrt(2*np.pi)*self.sigma)))
        result = self.coupling_strength*np.array([anharmonic,harmonic,error])
        self.evaluated_spectators += 1
        return self._remember(key,result)

    def _evaluate_block(self, keys):
        packets,targets,harmonic_values = [],[],[]
        for key in keys:
            occupations = np.asarray(key,np.int64)
            packet, harmonic_amplitudes = self._contracted_wavepacket(occupations)
            packets.append(packet)
            spectator_energy = float(self.frequencies_cm@occupations)
            targets.append(-self.oriented.electronic_gap_cm+self.initial.energies_cm[0]-spectator_energy-self.spectator_zpe)
            mismatch = self.oriented.electronic_gap_cm+self.harmonic_final_energies-self.harmonic_initial_energy+spectator_energy
            harmonic_values.append(float(np.sum(abs(harmonic_amplitudes)**2
                *np.exp(-.5*(mismatch/self.sigma)**2)/(np.sqrt(2*np.pi)*self.sigma))))
        result = sparse_gaussian_spectral_density_batch(self.final,np.column_stack(packets),np.asarray(targets),
            gaussian_sigma_cm=self.sigma,density_tolerance_per_strength=self.uniform_density_tolerance,
            maximum_degree=self.maximum_degree,maximum_matvec_work=self.maximum_matvec_work*len(keys),workers=self.workers,
            prepared_spectrum=self.prepared_spectrum)
        diagnostics = result['diagnostics']
        self.last_spectral_diagnostics = diagnostics
        self.total_matrix_passes += diagnostics['matvec_count']
        self.total_matvecs += diagnostics['matvec_count']*diagnostics['filtered_wavepacket_count']
        self.maximum_used_degree = max(self.maximum_used_degree,diagnostics['polynomial_degree'])
        values = {}
        for key,anharmonic,harmonic,error in zip(keys,result['densities_per_cm'],harmonic_values,result['absolute_error_bounds_per_cm']):
            value = tuple(self.coupling_strength*np.array([anharmonic,harmonic,error]))
            values[key] = self._remember(key,value)
            self.evaluated_spectators += 1
        return values

    def _batch(self, states, index):
        states = np.asarray(states)
        if (states.ndim != 2 or states.shape[1] != len(self.spectator_modes)
                or states.dtype.kind not in 'iu' or np.any(states < 0)
                or np.any(states > np.iinfo(np.int32).max)):
            raise ValueError('nonnegative integer spectator occupation rows required')
        if self.spectral_method!='chebyshev' or len(states)<2 or self.factor==0:
            return np.array([self._evaluate(row)[index] for row in states])
        keys = [tuple(map(int,row)) for row in states]
        values,pending = {},[]
        for key in keys:
            if key in values: continue
            cached = self._lookup(key)
            if cached is not None:values[key] = cached
            elif key not in pending: pending.append(key)
        for start in range(0,len(pending),self.maximum_batch_size):
            values.update(self._evaluate_block(pending[start:start+self.maximum_batch_size]))
        return np.array([values[key][index] for key in keys])

    def anharmonic(self, states): return self._batch(states,0)
    def harmonic(self, states): return self._batch(states,1)
    def filter_error_bound(self, states):
        if self.spectral_method != 'chebyshev':raise ValueError('Lanczos successive differences are not certified error bounds')
        return self._batch(states,2)
    def filter_error_estimate(self, states): return self._batch(states,2)

    @property
    def diagnostics(self):
        # Prepared windows need no global enclosure unless a filter fallback
        # actually requested one. Reporting must not trigger 256 dense-CSR passes.
        interval = (self.final.__dict__.get('spectral_interval_cm')
                    if self.prepared_spectrum is not None else self.final.spectral_interval_cm)
        return dict(**self.kernel.diagnostics,hamiltonian=self.final.diagnostics,
            spectral_interval_cm=interval,active_final_modes=self.active_modes,
            spectator_modes=self.spectator_modes,evaluated_spectators=self.evaluated_spectators,
            cache_hits=self.cache_hits,persistent_model_key=self.persistent_model_key,
            shared_wavepackets_enabled=self._shared_wavepackets is not None,
            shared_wavepacket_cache=(None if self._shared_wavepackets is None else dict(
                evaluations=self._shared_wavepackets.evaluations,hits=self._shared_wavepackets.hits,
                bytes=self._shared_wavepackets.bytes,maximum_bytes=self._shared_wavepackets.maximum_bytes)),
            persistent_cache_hits=self.persistent_cache_hits,persistent_cache_writes=self.persistent_cache_writes,
            total_matvecs=self.total_matvecs,total_sparse_matrix_passes=self.total_matrix_passes,
            maximum_batch_size=self.maximum_batch_size,workers=self.workers,maximum_used_degree=self.maximum_used_degree,
            transformed_initial_strength_moment=self.strength_moment,
            uniform_density_tolerance_per_strength=self.uniform_density_tolerance,
            analytic_total_filter_rate_error_bound_s_1=self.total_filter_tolerance_s_1 if self.spectral_method=='chebyshev' else None,
            spectral_method=self.spectral_method,maximum_used_lanczos_steps=self.maximum_used_lanczos_steps,
            prepared_spectrum_enabled=self.prepared_spectrum is not None,
            last_spectral_diagnostics=self.last_spectral_diagnostics,
            filter_bound_includes_roundoff=False,final_eigenvectors_computed=False,
            transition_rate_converged=False,full_molecular_rate_qualified=False)


__all__.append('SparseSpectatorObservable')


def lanczos_gaussian_spectral_density(hamiltonian, wavepacket, target_energy_cm, *,
        gaussian_sigma_cm, absolute_tolerance=1e-12, relative_tolerance=1e-6,
        minimum_steps=128, maximum_steps=8192, maximum_matvec_work=20000000000):
    """Adaptive Gauss-Lanczos density with explicit, empirical convergence checks.

    Only a three-vector recurrence is stored. Checkpoints double until the
    bandwidth/linewidth minimum is resolved, then advance by half that
    resolution (at least minimum_steps). Only the tridiagonal eigenpairs near
    the requested energy are acquired. Omitted
    quadrature-node tails have a Gaussian bound; successive Krylov differences
    are *not* a bound on the full matrix-function error. Two consecutive stable
    checkpoints and a bandwidth/linewidth minimum are required. Breakdown of
    the finite Krylov subspace permits an earlier result. Loss of global
    orthogonality is not independently certified: use the Chebyshev filter or
    a complete finite eigensystem to qualify representative wavepackets.
    A failed convergence or work check returns no density.

    A QuantumDynamics instance reuses its Hamiltonian action and caller-supplied
    spectral enclosure. Its work guard charges N squared dense-equivalent
    products per action; this is not a bound on an arbitrary operator's actual
    cost. No sparse matrix or final eigenvectors are constructed.
    """
    from scipy.linalg import eigh_tridiagonal
    from .quantum_dynamics import QuantumDynamics
    action_only = isinstance(hamiltonian, QuantumDynamics)
    matrix = (hamiltonian.hamiltonian if action_only else
              hamiltonian.matrix_cm if isinstance(hamiltonian,SparseVibrationalHamiltonian)
              else _checked_matrix(hamiltonian))
    products_per_action = matrix.shape[0]**2 if action_only else matrix.nnz
    v = np.asarray(wavepacket)
    if (v.ndim != 1 or len(v) != matrix.shape[0] or not np.all(np.isfinite(v))
            or not np.isfinite(target_energy_cm) or not np.isfinite(gaussian_sigma_cm)
            or gaussian_sigma_cm <= 0 or not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0
            or not np.isfinite(relative_tolerance) or relative_tolerance <= 0):
        raise ValueError('finite wavepacket, target and positive linewidth/tolerances required')
    for name,value in (('minimum_steps',minimum_steps),('maximum_steps',maximum_steps),('maximum_matvec_work',maximum_matvec_work)):
        _positive_integer(value,name)
    if minimum_steps > maximum_steps: raise ValueError('minimum_steps exceeds maximum_steps')
    if np.iscomplexobj(v) and not np.any(v.imag): v = v.real
    v = v.astype(np.result_type(matrix.dtype,v.dtype,np.float64),copy=False)
    norm2 = float(np.vdot(v,v).real)
    if not np.isfinite(norm2): raise FloatingPointError('wavepacket norm overflow')
    lower,upper = (hamiltonian.bounds if action_only else _hamiltonian_interval(hamiltonian,matrix))
    peak = 1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    zero_bound = norm2*peak*np.exp(-.5*(max(lower-target_energy_cm,target_energy_cm-upper,0.)/gaussian_sigma_cm)**2)
    if zero_bound <= absolute_tolerance:
        return dict(density_per_cm=0.,absolute_error_estimate_per_cm=zero_bound,
            diagnostics=dict(matvec_count=0,estimated_nonzero_products=0,checkpoints=[],
                input_norm_squared=norm2,error_estimate_is_bound=True,final_eigenvectors_computed=False,
                numerical_convergence_observed=True,finite_basis_convergence_certified=False))
    # Prevent an early zero plateau from passing before the global bandwidth
    # has been resolved at the requested linewidth. This remains a heuristic.
    resolution_min = max(minimum_steps,int(np.ceil((upper-lower)/(4*gaussian_sigma_cm))))
    resolution_min = min(resolution_min,matrix.shape[0])
    halfwidth = gaussian_sigma_cm*np.sqrt(2*max(1.,np.log(norm2*peak/(.1*absolute_tolerance))))
    omitted_bound = norm2*peak*np.exp(-.5*(halfwidth/gaussian_sigma_cm)**2)
    q = v/np.sqrt(norm2)
    old = np.zeros_like(q)
    beta_previous = 0.
    diagonal,offdiagonal = [],[]
    checkpoints = []
    stable = 0
    next_check = min(32,maximum_steps)
    error_estimate = float('inf')
    for step in range(1,maximum_steps+1):
        if step*products_per_action > maximum_matvec_work:
            raise MemoryError('Lanczos matvec work budget exceeded; no density inferred')
        w = matrix@q-beta_previous*old
        alpha_complex = np.vdot(q,w)
        if abs(alpha_complex.imag) > 1e-10*max(1.,abs(alpha_complex.real)):
            raise ArithmeticError('Lanczos diagonal is not real')
        alpha = float(alpha_complex.real)
        w -= alpha*q
        beta = float(np.linalg.norm(w))
        diagonal.append(alpha)
        scale = max(abs(lower),abs(upper),1.)
        breakdown = beta <= 32*np.finfo(float).eps*scale
        if step == next_check or breakdown or step == maximum_steps:
            eigenvalues,eigenvectors = eigh_tridiagonal(np.asarray(diagonal),np.asarray(offdiagonal),
                select='v',select_range=(target_energy_cm-halfwidth,target_energy_cm+halfwidth),check_finite=False)
            weights = abs(eigenvectors[0])**2
            if np.sum(weights) > 1+1e-8: raise ArithmeticError('invalid Lanczos quadrature weights')
            density = float(norm2*np.sum(weights*peak*np.exp(-.5*((eigenvalues-target_energy_cm)/gaussian_sigma_cm)**2)))
            difference = abs(density-checkpoints[-1]['density_per_cm']) if checkpoints else float('inf')
            error_estimate = difference+omitted_bound
            checkpoints.append(dict(steps=step,density_per_cm=density,
                successive_difference_per_cm=None if not np.isfinite(difference) else difference,
                retained_quadrature_nodes=len(eigenvalues)))
            if step >= resolution_min and difference <= absolute_tolerance+relative_tolerance*abs(density): stable += 1
            else: stable = 0
            if breakdown or stable >= 2:
                if breakdown: error_estimate = omitted_bound
                return dict(density_per_cm=density,absolute_error_estimate_per_cm=error_estimate,
                    diagnostics=dict(matvec_count=step,estimated_nonzero_products=step*products_per_action,
                        hamiltonian_action_only=action_only,
                        work_estimate_is_dense_equivalent=action_only,
                        input_norm_squared=norm2,spectral_interval_cm=[lower,upper],checkpoints=checkpoints,
                        minimum_resolution_steps=resolution_min,numerical_convergence_observed=True,
                        finite_krylov_breakdown=bool(breakdown),gaussian_node_tail_bound_per_cm=omitted_bound,
                        error_estimate_is_bound=False,global_reorthogonalization_used=False,
                        final_eigenvectors_computed=False,finite_basis_convergence_certified=False))
            next_check = min(maximum_steps,
                step+max(minimum_steps, (resolution_min+1)//2)
                if step >= resolution_min else 2*next_check)
        if step < maximum_steps:
            offdiagonal.append(beta)
            old,q = q,w/beta
            beta_previous = beta
    raise RuntimeError(f'Lanczos Gaussian density did not converge within {maximum_steps} matvecs; last difference plus node-tail estimate {error_estimate:g}; no density inferred')


__all__.append('lanczos_gaussian_spectral_density')


def sparse_gaussian_spectral_density_batch(hamiltonian, wavepackets, target_energies_cm, *,
        gaussian_sigma_cm, density_tolerance_per_strength=1e-12,
        maximum_degree=32768, maximum_matvec_work=160000000000, workers=1, prepared_spectrum=None):
    """Apply different Gaussian filters to columns using common sparse passes.

    Each column can have its own target energy and norm. A uniform error
    tolerance per unit input strength makes the degree common to all columns.
    The same matrix-function and Bernstein bounds as the single-vector API
    apply, with no sum of absolute per-configuration tolerances. Work budget
    counts nonzero products times columns, not just shared matrix passes.
    """
    matrix = (hamiltonian.matrix_cm if isinstance(hamiltonian,SparseVibrationalHamiltonian)
              else _checked_matrix(hamiltonian))
    vectors,targets = np.asarray(wavepackets),np.asarray(target_energies_cm,float)
    if (vectors.ndim != 2 or vectors.shape[0] != matrix.shape[0] or targets.shape != (vectors.shape[1],)
            or not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(targets))
            or not np.isfinite(gaussian_sigma_cm) or gaussian_sigma_cm <= 0
            or not np.isfinite(density_tolerance_per_strength) or density_tolerance_per_strength <= 0):
        raise ValueError('finite wavepacket columns, one target per column, positive sigma and tolerance required')
    for name,value in (('maximum_degree',maximum_degree),('maximum_matvec_work',maximum_matvec_work),('workers',workers)):
        _positive_integer(value,name)
    if np.iscomplexobj(vectors) and not np.any(vectors.imag): vectors = vectors.real
    vectors = vectors.astype(np.result_type(matrix.dtype,vectors.dtype,np.float64),copy=False)
    norms = np.sum(abs(vectors)**2,axis=0)
    if not np.all(np.isfinite(norms)):raise FloatingPointError('wavepacket norms overflow')
    if prepared_spectrum is not None:
        from .vibronic_spectral_cache import PreparedGaussianSpectrum,PreparedBlockGaussianSpectrum
        from .vibronic_energy_windows import PreparedEnergyWindows
        if (not isinstance(hamiltonian,SparseVibrationalHamiltonian)
                or not isinstance(prepared_spectrum,(PreparedGaussianSpectrum,PreparedBlockGaussianSpectrum,PreparedEnergyWindows))
                or prepared_spectrum.hamiltonian_sha256!=hamiltonian.fingerprint):
            raise ValueError('prepared spectrum does not match the finite Hamiltonian')
        result = prepared_spectrum.evaluate(vectors,targets,sigma_cm=gaussian_sigma_cm,
            tolerance_per_strength=density_tolerance_per_strength,workers=workers)
        remaining = np.flatnonzero(~result.pop('accepted'))
        diagnostics = dict(schema='matrix.trinity.sparse_gaussian_density_batch.v1',
            wavepacket_count=len(targets),filtered_wavepacket_count=0,polynomial_degree=0,
            matvec_count=0,estimated_nonzero_products=0,final_eigenvectors_computed=False,
            prepared_eigenvectors_used=True,polynomial_error_bound_includes_roundoff=False,
            finite_basis_convergence_certified=False,**result['diagnostics'])
        if len(remaining):
            fallback = sparse_gaussian_spectral_density_batch(hamiltonian,vectors[:,remaining],
                targets[remaining],gaussian_sigma_cm=gaussian_sigma_cm,
                density_tolerance_per_strength=density_tolerance_per_strength,
                maximum_degree=maximum_degree,maximum_matvec_work=maximum_matvec_work,workers=workers)
            result['densities_per_cm'][remaining] = fallback['densities_per_cm']
            result['absolute_error_bounds_per_cm'][remaining] = fallback['absolute_error_bounds_per_cm']
            for name in ('filtered_wavepacket_count','polynomial_degree','matvec_count','estimated_nonzero_products'):
                diagnostics[name] = fallback['diagnostics'][name]
            diagnostics['spectral_backend'] = 'prepared_eigenwindow_with_chebyshev_fallback'
        diagnostics['prepared_wavepacket_count'] = len(targets)-len(remaining)
        diagnostics['fallback_wavepacket_count'] = len(remaining)
        return dict(result,diagnostics=diagnostics)
    lower,upper = _hamiltonian_interval(hamiltonian,matrix)
    midpoint,radius = (upper+lower)/2,(upper-lower)/2
    peak = 1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    distance = np.maximum(np.maximum(lower-targets,targets-upper),0.)
    zero_profile_bound = peak*np.exp(-.5*(distance/gaussian_sigma_cm)**2)
    use = np.flatnonzero((norms>0)&(zero_profile_bound>density_tolerance_per_strength))
    values,errors = np.zeros(len(targets)),norms*zero_profile_bound
    diagnostics = dict(schema='matrix.trinity.sparse_gaussian_density_batch.v1',
        wavepacket_count=len(targets),filtered_wavepacket_count=len(use),polynomial_degree=0,matvec_count=0,
        estimated_nonzero_products=0,final_eigenvectors_computed=False,
        polynomial_error_bound_includes_roundoff=False,finite_basis_convergence_certified=False)
    if not len(use):return dict(densities_per_cm=values,absolute_error_bounds_per_cm=errors,diagnostics=diagnostics)
    if np.count_nonzero(matrix.data)==np.count_nonzero(matrix.diagonal()):
        values = np.sum(abs(vectors)**2*peak*np.exp(-.5*((matrix.diagonal().real[:,None]-targets)/gaussian_sigma_cm)**2),axis=0)
        errors[:] = 0.
        diagnostics['diagonal_evaluation_used'] = True
        return dict(densities_per_cm=values,absolute_error_bounds_per_cm=errors,diagnostics=diagnostics)
    threshold = density_tolerance_per_strength/(np.sqrt(peak+density_tolerance_per_strength)+np.sqrt(peak))
    log_target = np.log(threshold)
    ratio = radius/(np.sqrt(2)*gaussian_sigma_cm)
    degree = min(32,maximum_degree)
    while _log_chebyshev_error(degree,ratio,.5*np.log(peak))>log_target:
        if degree>=maximum_degree:raise MemoryError('Gaussian batch exceeds polynomial degree budget; no density inferred')
        degree = min(2*degree,maximum_degree)
    low,high = 1,degree
    while low<high:
        middle = (low+high)//2
        if _log_chebyshev_error(middle,ratio,.5*np.log(peak))<=log_target:high=middle
        else:low=middle+1
    degree = high
    work = (degree+1)*matrix.nnz*len(use)
    if work>maximum_matvec_work:raise MemoryError('Gaussian batch exceeds nonzero-product work budget; no density inferred')
    nodes = np.cos(np.pi*np.arange(degree+1)/degree)
    samples = np.sqrt(peak)*np.exp(-.25*((midpoint+radius*nodes[:,None]-targets[use])/gaussian_sigma_cm)**2)
    coefficients = dct(samples,type=1,axis=0)/degree
    coefficients[[0,-1]] *= .5
    source = np.ascontiguousarray(vectors[:,use])
    filtered = _filter_columns(matrix,midpoint,radius,coefficients,source,workers)
    values[use] = np.sum(abs(filtered)**2,axis=0)
    if not np.all(np.isfinite(values)):raise FloatingPointError('nonfinite batch spectral densities')
    epsilon = np.exp(_log_chebyshev_error(degree,ratio,.5*np.log(peak)))
    errors[use] = norms[use]*epsilon*(2*np.sqrt(peak)+epsilon)
    diagnostics.update(polynomial_degree=degree,matvec_count=degree+1,estimated_nonzero_products=work,
        spectral_interval_cm=[lower,upper],polynomial_sqrt_filter_uniform_error=epsilon)
    return dict(densities_per_cm=values,absolute_error_bounds_per_cm=errors,diagnostics=diagnostics)


__all__.append('sparse_gaussian_spectral_density_batch')


def solve_sparse_banded_variational(hamiltonian, *, level_count=None,
        energy_window_cm=None, maximum_band_bytes=2*1024**3,
        maximum_workspace_bytes=8*1024**3, workers=1):
    """Acquire a reusable finite spectrum after an exact bandwidth reduction.

    A reverse Cuthill-McKee permutation changes only basis order. LAPACK solves
    the real symmetric/complex Hermitian band matrix without building a dense
    Hamiltonian. Its eigenvector transformation workspace can still be N-by-N;
    the explicit conservative workspace guard accounts for that allocation.
    This route amortizes spectral acquisition over many spectator callbacks;
    the direct positive Gaussian filter remains useful for a few probes.
    """
    from scipy.linalg import eig_banded
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    from threadpoolctl import threadpool_limits
    from .vibronic_variational import FiniteVibrationalStates
    if not isinstance(hamiltonian,SparseVibrationalHamiltonian):
        raise TypeError('SparseVibrationalHamiltonian required')
    for name,value in (('maximum_band_bytes',maximum_band_bytes),
                       ('maximum_workspace_bytes',maximum_workspace_bytes),('workers',workers)):
        _positive_integer(value,name)
    matrix = hamiltonian.matrix_cm
    size = matrix.shape[0]
    if level_count is not None:
        _positive_integer(level_count,'level_count')
        if level_count>size:raise ValueError('requested levels exceed Hamiltonian dimension')
    if energy_window_cm is not None:
        window = np.asarray(energy_window_cm,float)
        if level_count is not None or window.shape!=(2,) or not np.all(np.isfinite(window)) or window[0]>=window[1]:
            raise ValueError('choose ordered finite energy window or level count')
    else:window = None
    permutation = reverse_cuthill_mckee(matrix,symmetric_mode=True)
    ordered = matrix[permutation,:][:,permutation].tocoo()
    bandwidth = int(np.max(abs(ordered.row-ordered.col),initial=0))
    band_bytes = (bandwidth+1)*size*matrix.dtype.itemsize
    count_budget = size if level_count is None else level_count
    workspace_budget = 6*size*size*16+4*size*count_budget*16+2*band_bytes
    if band_bytes>maximum_band_bytes or workspace_budget>maximum_workspace_bytes:
        raise MemoryError(f'banded solve requires {band_bytes} band bytes and conservative workspace budget {workspace_budget}')
    band = np.zeros((bandwidth+1,size),dtype=matrix.dtype,order='F')
    keep = ordered.row>=ordered.col
    band[ordered.row[keep]-ordered.col[keep],ordered.col[keep]] = ordered.data[keep]
    del ordered
    if window is not None:controls=dict(select='v',select_range=tuple(window))
    elif level_count is not None:controls=dict(select='i',select_range=(0,level_count-1))
    else:controls=dict(select='a')
    with threadpool_limits(limits=workers,user_api='blas'):
        energies,vectors = eig_banded(band,lower=True,overwrite_a_band=True,check_finite=False,**controls)
        if not len(energies):raise ValueError('selected finite energy window contains no eigenlevels')
        restored = np.empty_like(vectors)
        restored[permutation,:] = vectors
        del vectors,band
        residual = np.empty(len(energies))
        for start in range(0,len(energies),64):
            end = min(start+64,len(energies))
            block = restored[:,start:end]
            residual[start:end] = np.linalg.norm(matrix@block-block*energies[start:end],axis=0)
        shell = hamiltonian.diagnostics['maximum_total_quanta']
        edge = hamiltonian.diagnostics['maximum_harmonic_energy_cm']
        boundary = hamiltonian.basis.sum(axis=1)>max(-1,shell-4)
        energy_boundary = np.zeros(size,bool) if edge is None else hamiltonian.basis@hamiltonian.frequencies_cm>edge-4*max(hamiltonian.frequencies_cm)
        diagnostics = {**hamiltonian.diagnostics,
            'schema':'matrix.trinity.sparse_banded_variational.v1',
            'numerical_arithmetic':'complex' if np.iscomplexobj(matrix.data) else 'real',
            'basis_permutation':'reverse Cuthill-McKee; all entries retained',
            'half_bandwidth':bandwidth,'band_storage_bytes':band_bytes,
            'conservative_workspace_budget_bytes':workspace_budget,
            'dense_transformation_workspace_possible':True,'dense_hamiltonian_allocated':False,
            'selected_energy_window_cm':None if window is None else window.tolist(),
            'selected_lowest_level_count':level_count,
            'residual_norms_cm':residual,
            'boundary_probabilities':np.sum(abs(restored[boundary])**2,axis=0),
            'energy_boundary_probabilities':np.sum(abs(restored[energy_boundary])**2,axis=0)}
        return FiniteVibrationalStates(hamiltonian.basis,energies,restored,hamiltonian.frequencies_cm,diagnostics)


__all__.append('solve_sparse_banded_variational')
