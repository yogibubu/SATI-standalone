"""Contract a finite coherent source before expanding active Gaussian pairs.

Wick contractions act on the source polynomial as exp(D_ij d_i d_j/2).
Truncated multivariate Horner substitution and a linear FFT convolution then
extract active Taylor coefficients. Truncation here is exact for the requested
occupation box: higher active powers cannot contribute to lower coefficients.
No source coefficients or couplings are thresholded.
"""
from math import prod
import numpy as np
from numba import njit, prange
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve
from scipy.special import gammaln
from .gaussian_hermite import (hermite_recurrence_plan, hermite_coefficients,
                               _hermite_recurrence_values)


def _grid(dimensions):
    size = prod(map(int, dimensions))
    strides = np.array([prod(map(int, dimensions[k+1:])) for k in range(len(dimensions))], np.int64)
    words = np.empty((size, len(dimensions)), np.int64)
    indices = np.arange(size)
    for k, (count, stride) in enumerate(zip(dimensions, strides)):
        words[:, k] = indices//stride % count
    return words, strides


@njit(cache=True, nogil=True)
def _wick_heat(original, covariance, dimensions, strides, words):
    result = original.copy()
    for i in range(len(dimensions)):
        for j in range(i, len(dimensions)):
            a = covariance[i, j]*(.5 if i == j else 1.)
            if a == 0:
                continue
            shift = strides[i]+strides[j]
            # Higher indices have not yet been updated by this commuting
            # differential exponential: the ascending in-place sweep is exact.
            for index in range(len(result)):
                p, q = words[index, i], words[index, j]
                limit = ((dimensions[i]-1-p)//2 if i == j else
                         min(dimensions[i]-1-p, dimensions[j]-1-q))
                factor = 1.+0j
                value = result[index]
                for order in range(1, limit+1):
                    if i == j:
                        factor *= a/order*(p+2*order-1)*(p+2*order)
                    else:
                        factor *= a/order*(p+order)*(q+order)
                    value += factor*result[index+order*shift]
                result[index] = value
    return result


@njit(cache=True, nogil=True)
def _multiply_affine_inplace(poly, constant, linear, lower, count):
    # Degree ordering keeps every lowered coefficient earlier in the vector.
    for index in range(count-1, -1, -1):
        value = constant*poly[index]
        for k in range(len(linear)):
            if lower[index, k] >= 0:
                value += linear[k]*poly[lower[index, k]]
        poly[index] = value


# Recursive LLVM symbols cannot be restored reliably from Numba disk caches.
@njit(cache=False, nogil=True)
def _horner_source(coefficients, support, dimensions, strides, constants, linear,
                   degree_bounds, degree_counts, lower, level, first):
    remaining = dimensions[level]*strides[level]
    bound = degree_bounds[level][first//remaining]
    if bound < 0:
        return np.zeros(0, np.complex128), -1
    bound = min(bound, len(degree_counts)-1)
    result = np.zeros(degree_counts[bound], np.complex128)
    highest = -1
    for power in range(dimensions[level]-1, -1, -1):
        start = first+power*strides[level]
        found = False
        for k in range(start, start+strides[level]):
            if support[k]:
                found = True
                break
        if found:
            highest = power
            break
    if highest < 0:
        return result, -1
    current_degree = -1
    for power in range(highest, -1, -1):
        if power != highest and current_degree >= 0:
            current_degree = min(current_degree+1, bound)
            _multiply_affine_inplace(result, constants[level], linear[level], lower,
                                     degree_counts[current_degree])
        start = first+power*strides[level]
        if level+1 == len(dimensions):
            result[0] += coefficients[start]
            current_degree = max(current_degree, 0)
        else:
            child, child_degree = _horner_source(coefficients, support, dimensions, strides, constants,
                linear, degree_bounds, degree_counts, lower, level+1, start)
            for index in range(len(child)):
                result[index] += child[index]
            current_degree = max(current_degree, child_degree)
    return result, bound


def _prepare_horner_support(self, lower=None):
    support = self.original.reshape(tuple(map(int, self.source_dimensions))) != 0
    # Downward closure is structural, with no amplitude threshold.
    for axis in range(len(self.source_dimensions)):
        support = np.flip(np.maximum.accumulate(np.flip(support, axis=axis), axis=axis), axis=axis)
    self.support = support.ravel().copy()
    degrees = self.active_words.sum(axis=1)
    self.degree_order = np.argsort(degrees, kind='stable')
    ranks = np.empty(len(self.active_words), np.int64)
    ranks[self.degree_order] = np.arange(len(self.active_words))
    self.degree_counts = np.cumsum(np.bincount(degrees))
    ordered_words = self.active_words[self.degree_order]
    self.lower = np.full(ordered_words.shape, -1, np.int64)
    for axis in range(ordered_words.shape[1]):
        present = ordered_words[:, axis] > 0
        parent = (self.degree_order[present]-self.active_strides[axis] if lower is None
                  else lower[self.degree_order[present], axis])
        self.lower[present, axis] = ranks[parent]
    self.degree_bounds = tuple(np.max(np.where(
        self.support.reshape(-1, int(self.source_dimensions[level])*int(self.source_strides[level])),
        self.source_words[:, level:].sum(axis=1).reshape(-1,
            int(self.source_dimensions[level])*int(self.source_strides[level])), -1), axis=1)
        for level in range(len(self.source_dimensions)))


class GaussianSourcePolynomial:
    """Resource-bounded source-first evaluator for the existing bath kernel."""
    def __init__(self, initial_rows, coefficients, active_basis, *,
                 maximum_integrals, maximum_workspace_bytes):
        initial_rows = np.asarray(initial_rows, np.int64)
        basis = np.asarray(active_basis, np.int64)
        ni, na = len(initial_rows), len(basis)
        self.initial_width, self.active_width = initial_rows.shape[1], basis.shape[1]
        source_dimensions = np.tile(initial_rows.max(axis=0)+1, 2)
        active_dimensions = np.tile(basis.max(axis=0)+1, 2)
        source_size, active_size = prod(map(int, source_dimensions)), prod(map(int, active_dimensions))
        fft_dimensions = tuple(next_fast_len(2*int(d)-1) for d in active_dimensions)
        fft_size = prod(fft_dimensions)
        if max(source_size, active_size, fft_size) > maximum_integrals:
            raise MemoryError('source-first polynomial/FFT grid exceeds declared entry budget')
        estimate = (16*(8*fft_size+8*source_size+(8+4*len(source_dimensions))*active_size)
                    +8*(source_size*len(source_dimensions)+active_size*len(active_dimensions))
                    +active_size*(96+48*len(active_dimensions))+64*na*na)
        if estimate > maximum_workspace_bytes:
            raise MemoryError('source-first polynomial exceeds numeric workspace budget')
        self.source_words, self.source_strides = _grid(source_dimensions)
        self.active_words, self.active_strides = _grid(active_dimensions)
        self.source_dimensions = np.asarray(source_dimensions, np.int64)
        self.active_dimensions = tuple(map(int, active_dimensions))
        self.plan = hermite_recurrence_plan(self.active_words, 0, maximum_integrals)
        self.original = np.zeros(source_size, complex)
        log_factorials = gammaln(initial_rows+1).sum(axis=1)
        scaled = np.asarray(coefficients)*np.exp(-.5*log_factorials)
        width = self.initial_width
        left = initial_rows@self.source_strides[:width]
        right = initial_rows@self.source_strides[width:]
        self.original[(left[:, None]+right[None, :]).ravel()] = np.outer(scaled.conj(), scaled).ravel()
        _prepare_horner_support(self)
        self.radii = np.sqrt(np.maximum(1, active_dimensions-1)).astype(float)
        self.gaussian_taylor_scale = np.exp(-.5*gammaln(self.active_words+1).sum(axis=1))
        a = basis@self.active_strides[:self.active_width]
        b = basis@self.active_strides[self.active_width:]
        self.take = a[:, None]+b[None, :]
        query = np.column_stack([np.repeat(basis, na, axis=0), np.tile(basis, (na, 1))])
        self.query_scale = np.exp(.5*gammaln(query+1).sum(axis=1)-query@np.log(self.radii)).reshape(na, na)
        mi, ma = self.initial_width, self.active_width
        self.source_indices = np.r_[np.arange(mi), np.arange(mi+ma, 2*mi+ma)]
        self.active_indices = np.r_[np.arange(mi, mi+ma), np.arange(2*mi+ma, 2*(mi+ma))]
        self.numeric_storage_bytes = sum(a.nbytes for a in (
            self.source_words, self.source_strides, self.active_words, self.active_strides,
            self.source_dimensions, self.original, self.support, self.radii,
            self.gaussian_taylor_scale, self.take, self.query_scale, self.source_indices,
            self.active_indices, self.degree_order, self.degree_counts, self.lower,
            *self.degree_bounds, *self.plan))
        self.evaluation_workspace_bytes = estimate
        self.diagnostics = dict(source_polynomial_entries=source_size,
            active_polynomial_entries=active_size, fft_workspace_entries=fft_size,
            initial_coefficients_pruned=False, active_power_truncation_is_exact=True,
            horner_uses_exact_total_degree_support=True,
            coefficient_scaling='sqrt(maximum occupation) per active variable',
            fft_roundoff_indicator_is_bound=False)

    def evaluate(self, matrix, linear, constant):
        x, y = self.source_indices, self.active_indices
        covariance = matrix[np.ix_(x, x)]
        polynomial = _wick_heat(self.original, covariance, self.source_dimensions,
                                self.source_strides, self.source_words)
        if len(x):
            compact, _ = _horner_source(polynomial, self.support, self.source_dimensions,
                self.source_strides, linear[x], matrix[np.ix_(x, y)]*self.radii[None, :],
                self.degree_bounds, self.degree_counts, self.lower, 0, 0)
            polynomial = np.zeros(len(self.active_words), complex)
            polynomial[self.degree_order[:len(compact)]] = compact
        else:
            polynomial = np.full(len(self.active_words), 0j)
            polynomial[0] = self.original[0]
        gaussian = hermite_coefficients(matrix[np.ix_(y, y)]*self.radii[:, None]*self.radii[None, :],
            linear[y]*self.radii, constant, self.plan)*self.gaussian_taylor_scale
        # Full linear convolution, then crop: circular aliasing is not allowed.
        if self.active_width:
            full = fftconvolve(polynomial.reshape(self.active_dimensions),
                               gaussian.reshape(self.active_dimensions), mode='full')
            cropped = full[tuple(slice(0, d) for d in self.active_dimensions)].ravel()
        else:
            cropped = polynomial*gaussian
        result = cropped[self.take]*self.query_scale
        self.diagnostics['last_fft_roundoff_indicator'] = float(64*np.finfo(float).eps
            *max(1., np.log2(self.diagnostics['fft_workspace_entries']))
            *np.sum(abs(polynomial))*np.sum(abs(gaussian))*np.max(self.query_scale))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('source-first Gaussian contraction exceeds numerical range')
        return result


@njit(cache=True, nogil=True)
def _sparse_source_convolution(polynomial, gaussian, offsets, targets, parents, factors, size):
    result = np.zeros(size, np.complex128)
    for term in range(len(polynomial)):
        for k in range(offsets[term], offsets[term+1]):
            result[targets[k]] += polynomial[term]*factors[k]*gaussian[parents[k]]
    return result


def _source_amplitudes_batch(matrix, linear, constants, heated, dimensions, strides,
        support, degree_bounds, degree_counts, lower, plan_rows, edge_offsets, edges,
        row_offsets, row_terms, row_parents, row_factors, width, size):
    result = np.empty((len(constants), size), np.complex128)
    for column in prange(len(constants)):
        if width:
            polynomial,_ = _horner_source(heated,support,dimensions,strides,
                linear[column,:width],matrix[:width,width:],
                degree_bounds,degree_counts,lower,0,0)
        else:
            polynomial = heated
        gaussian = _hermite_recurrence_values(matrix[width:,width:],
            linear[column,width:],constants[column],plan_rows,edge_offsets,edges)
        for row in range(size):
            value = 0.+0j
            for k in range(row_offsets[row],row_offsets[row+1]):
                term = row_terms[k]
                if term < len(polynomial):
                    value += polynomial[term]*row_factors[k]*gaussian[row_parents[k]]
            result[column,row] = value
    return result


_source_amplitudes_serial = njit(cache=False,nogil=True)(_source_amplitudes_batch)
_source_amplitudes_parallel = njit(cache=False,nogil=True,parallel=True)(_source_amplitudes_batch)


class GaussianSourceAmplitude:
    """Single coherent packets on the shared exact Gaussian lowering closure.

    Sparse polynomial convolution avoids both an active Cartesian box and
    active pairs. Every finite source coefficient and requested state remains.
    """
    def __init__(self, initial_rows, coefficients, basis, *,
                 maximum_integrals, maximum_workspace_bytes):
        from .gaussian_moment_gram import gaussian_lowering_closure
        initial_rows = np.asarray(initial_rows, np.int64)
        basis = np.asarray(basis, np.int64)
        self.initial_width, self.active_width = initial_rows.shape[1], basis.shape[1]
        self.source_dimensions = initial_rows.max(axis=0)+1
        source_size = prod(map(int, self.source_dimensions))
        if source_size > maximum_integrals:
            raise MemoryError('source amplitude polynomial exceeds declared entry budget')
        closure, lower, self.take = gaussian_lowering_closure(basis, maximum_states=maximum_integrals)
        degree = int(initial_rows.sum(axis=1).max())
        self.active_words = closure[closure.sum(axis=1) <= degree]
        # Closure ordering by degree makes the retained polynomial a prefix.
        self.active_strides = np.zeros(self.active_width, np.int64)
        polynomial_lower = lower[:len(self.active_words)]
        counts = np.array([np.all(basis >= row, axis=1).sum() for row in self.active_words])
        work = int(counts.sum())
        estimate = (source_size*(128+8*self.initial_width)
                    +len(closure)*(192+48*self.active_width)
                    +len(self.active_words)*(128+40*self.active_width)+72*work+64*len(basis))
        if work > maximum_integrals or estimate > maximum_workspace_bytes:
            raise MemoryError('source amplitude sparse convolution exceeds declared entry/workspace budget')
        self.source_words, self.source_strides = _grid(self.source_dimensions)
        self.plan = hermite_recurrence_plan(closure, 0, maximum_integrals)
        self.original = np.zeros(source_size, complex)
        self.original[initial_rows@self.source_strides] = coefficients*np.exp(-.5*gammaln(initial_rows+1).sum(axis=1))
        _prepare_horner_support(self, polynomial_lower)
        lookup = {tuple(row): i for i,row in enumerate(closure)}
        self.offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self.targets = np.empty(work, np.int64)
        self.parents = np.empty(work, np.int64)
        self.factors = np.empty(work)
        for term,row in enumerate(self.active_words):
            target = np.flatnonzero(np.all(basis >= row, axis=1))
            lowered = basis[target]-row
            a,b = self.offsets[term:term+2]
            self.targets[a:b] = target
            self.parents[a:b] = [lookup[tuple(word)] for word in lowered]
            self.factors[a:b] = np.exp(.5*(gammaln(basis[target]+1)-gammaln(lowered+1)).sum(axis=1))
        order = np.argsort(self.targets,kind='stable')
        index_dtype = np.int32 if max(work,len(closure),len(self.active_words)) <= np.iinfo(np.int32).max else np.int64
        self.row_offsets = np.r_[0,np.cumsum(np.bincount(self.targets,minlength=len(basis)))].astype(index_dtype)
        self.row_terms = np.repeat(np.arange(len(counts),dtype=index_dtype),counts)[order]
        gaussian_output = self.plan[4][:,0].astype(np.int64)
        self.row_parents = gaussian_output[self.parents[order]].astype(index_dtype)
        self.row_factors = self.factors[order]
        self.numeric_storage_bytes = sum(a.nbytes for a in (
            self.source_words,self.source_strides,self.source_dimensions,self.original,
            self.support,self.active_words,self.active_strides,self.degree_order,self.degree_counts,
            self.lower,self.take,self.offsets,self.targets,self.parents,self.factors,
            self.row_offsets,self.row_terms,self.row_parents,self.row_factors,
            *self.degree_bounds,*self.plan))
        self.evaluation_workspace_bytes = estimate
        self.diagnostics = dict(source_polynomial_entries=source_size,
            active_polynomial_entries=len(self.active_words), gaussian_lowering_closure_size=len(closure),
            sparse_convolution_products=work, initial_coefficients_pruned=False,
            active_power_truncation_is_exact=True, horner_uses_exact_total_degree_support=True,
            active_pairs_constructed=False, fft_used=False,
            batch_convolution_order='target rows, stable source terms')

    def evaluate_batch(self, matrix, linear, constants):
        """Compile complete packets in one call, with shared Wick contraction.

        Output rows label packets. The scalar evaluator remains an independent
        contraction reference. Main-thread batches use the configured Numba
        thread count; Python worker threads use the serial compiled kernel to
        avoid nesting another thread pool inside replica-level parallelism.
        """
        from threading import current_thread, main_thread
        linear,constants = np.asarray(linear),np.asarray(constants)
        width = self.initial_width
        heated = _wick_heat(self.original,matrix[:width,:width],self.source_dimensions,
                            self.source_strides,self.source_words)
        kernel = (_source_amplitudes_parallel if len(constants)>=32 and current_thread() is main_thread()
                  else _source_amplitudes_serial)
        value = kernel(matrix,linear,constants,heated,self.source_dimensions,
            self.source_strides,self.support,self.degree_bounds,self.degree_counts,self.lower,
            self.plan[0],self.plan[1],self.plan[2],self.row_offsets,self.row_terms,
            self.row_parents,self.row_factors,width,len(self.take))
        if not np.all(np.isfinite(value)):
            raise FloatingPointError('coherent Gaussian packet batch exceeds numerical range')
        return value

    def evaluate(self, matrix, linear, constant):
        width = self.initial_width
        polynomial = _wick_heat(self.original, matrix[:width,:width], self.source_dimensions,
                                self.source_strides,self.source_words)
        if width:
            polynomial,_ = _horner_source(polynomial,self.support,self.source_dimensions,
                self.source_strides,linear[:width],matrix[:width,width:],
                self.degree_bounds,self.degree_counts,self.lower,0,0)
        else:
            polynomial = self.original
        gaussian = hermite_coefficients(matrix[width:,width:],linear[width:],constant,self.plan)
        # Both polynomial support and degree_order have the same stable order.
        value = _sparse_source_convolution(polynomial,gaussian,self.offsets,
            self.targets,self.parents,self.factors,len(self.take))
        if not np.all(np.isfinite(value)):
            raise FloatingPointError('coherent Gaussian packet exceeds numerical range')
        return value
