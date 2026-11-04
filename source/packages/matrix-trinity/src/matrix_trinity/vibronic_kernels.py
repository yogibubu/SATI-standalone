"""Compiled, GIL-free kernels; fixed-order reductions and disjoint row writes."""
from __future__ import annotations
import numpy as np
from numba import njit
from .vibronic_parallel import worker_count,parallel_rows


@njit(cache=True, nogil=True)
def _unrank_combination(rank, size, count, binomial):
    result = np.empty(count, np.int64)
    start = 0
    for slot in range(count):
        for value in range(start, size):
            ways = binomial[size - value - 1, count - slot - 1]
            if rank < ways:
                result[slot] = value
                start = value + 1
                break
            rank -= ways
    return result


@njit(cache=True, nogil=True)
def fill_occupations(start, stop, output, totals, classes, offsets, binomial):
    modes = output.shape[1]
    for row in range(start, stop):
        if row == 0:
            continue
        task = np.searchsorted(offsets, row, side='right') - 1
        total = totals[task]
        count = classes[task]
        partitions = binomial[total - 1, count - 1]
        rank = row - offsets[task]
        active = _unrank_combination(rank // partitions, modes, count, binomial)
        cuts = _unrank_combination(rank % partitions, total - 1, count - 1, binomial)
        previous = -1
        for slot in range(count):
            cut = cuts[slot] if slot < count - 1 else total - 1
            output[row, active[slot]] = cut - previous
            previous = cut


@njit(cache=True, nogil=True)
def recurrence_values(offsets, parents, factors, ground):
    values = np.empty(len(offsets) - 1)
    values[0] = ground
    for row in range(1, len(values)):
        value = 0.
        for edge in range(offsets[row], offsets[row + 1]):
            value += factors[edge] * values[parents[edge]]
        values[row] = value
    return values


@njit(cache=True, nogil=True)
def transition_amplitudes(start, stop, states, ids, values, mu, ht0, htlinear, output):
    for row in range(start, stop):
        overlap = values[ids[row, 0]]
        fc_strength = 0.
        ht_strength = 0.
        cross = 0.
        combined = 0.
        for axis in range(3):
            fc = mu[axis] * overlap
            ht = ht0[axis] * overlap
            for mode in range(states.shape[1]):
                if ids[row, mode + 1] >= 0:
                    ht += htlinear[axis, mode] * np.sqrt(states[row, mode]) * values[ids[row, mode + 1]]
            fc_strength += fc.real**2 + fc.imag**2
            ht_strength += ht.real**2 + ht.imag**2
            cross += 2 * (fc.real * ht.real + fc.imag * ht.imag)
            combined += (fc.real + ht.real)**2 + (fc.imag + ht.imag)**2
        output[row, 0] = overlap
        output[row, 1] = fc_strength
        output[row, 2] = ht_strength
        output[row, 3] = cross
        output[row, 4] = combined
