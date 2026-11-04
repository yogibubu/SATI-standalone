"""Native occupation-masked PT2 properties from finite tensor-shift convolutions.

No virtual-state cutoff is used. The normalized wave operator and its direct
resonance exclusions are those of reduced_resonant_transitions. Scalar field
contractions are shared by all property components. Hashes accelerate lookup;
every candidate is checked against the full integer shift, so collisions cannot
alter a coefficient and there is no packed-word limit on the number of modes.
"""
from collections import namedtuple, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import os
import sys
from pathlib import Path
from threading import Lock, local

import numpy as np
from numba import njit, types
from numba.typed import Dict
from scipy.sparse.linalg import LinearOperator

from .reduced_resonant_transitions import PairReducedField, PairReducedProperty, _gf2_reduce, _parity_mask

_FieldCache = namedtuple('_FieldCache', 'kinds occupations shifts values statistics joins join_used join_limit join_statistics')

_Data = namedtuple('_Data', 'omega shifts hashes weights offsets terms modes annih creation coefficients mask_states mask_hashes mask_offsets mask_targets tolerance include_electrical')


@njit(cache=True)
def _hash(row, weights):
    value = np.uint64(0)
    for i in range(len(row)):
        value += np.uint64(row[i])*weights[i]
    return value


@njit(cache=True, inline='always')
def _same_ints(left, right):
    # Internal arrays have validated compatible lengths. Stop at the first
    # mismatch without allocating an elementwise comparison array.
    for i in range(len(right)):
        if left[i] != right[i]:
            return False
    return True


@njit(cache=True, inline='always')
def _same_target(target, n, shift):
    for i in range(len(n)):
        if target[i] != n[i]+shift[i]:
            return False
    return True


@njit(cache=True)
def _find(kind, shift, data):
    code = _hash(shift, data.weights)
    first, last = data.offsets[kind], data.offsets[kind+1]
    lo, hi = first, last
    while lo < hi:
        mid = (lo+hi)//2
        if data.hashes[mid] < code:
            lo = mid+1
        else:
            hi = mid
    while lo < last and data.hashes[lo] == code:
        if _same_ints(data.shifts[lo], shift):
            return lo
        lo += 1
    return -1


@njit(cache=True)
def _internal(n, shift, data):
    if not np.any(shift):
        return True
    code = _hash(n, data.weights)
    lo, hi = 0, len(data.mask_states)
    while lo < hi:
        mid = (lo+hi)//2
        if data.mask_hashes[mid] < code:
            lo = mid+1
        else:
            hi = mid
    while lo < len(data.mask_states) and data.mask_hashes[lo] == code:
        if _same_ints(data.mask_states[lo], n):
            for k in range(data.mask_offsets[lo], data.mask_offsets[lo+1]):
                if _same_target(data.mask_targets[k], n, shift):
                    return True
        lo += 1
    return False


@njit(cache=True, inline='always')
def _weight(term, n, data):
    weight = 1.
    for j in range(data.modes.shape[1]):
        mode = data.modes[term, j]
        if mode < 0:
            break
        occ = n[mode]
        a, b = data.creation[term, j], data.annih[term, j]
        if occ < b:
            return 0.
        for k in range(b):
            weight *= np.sqrt(occ-k)
        for k in range(a):
            weight *= np.sqrt(occ-b+k+1)
    return weight


@njit(cache=True)
def _bare(group, n, data):
    result = np.zeros(data.coefficients.shape[1], np.complex128)
    if group >= 0:
        for term in range(data.terms[group], data.terms[group+1]):
            weight = _weight(term, n, data)
            if weight:
                result += weight*data.coefficients[term]
    return result


@njit(cache=True, inline='always')
def _cache_slot(kind, n, shift, data, cache):
    if not len(cache.kinds):
        return -1
    code = (_hash(n, data.weights)*np.uint64(0x9e3779b97f4a7c15)
            ^ _hash(shift, data.weights) ^ np.uint64(kind*104729))
    return np.int64(code & np.uint64(len(cache.kinds)-1))


@njit(cache=True, inline='always')
def _remember(slot, kind, n, shift, value, cache):
    if slot >= 0:
        cache.kinds[slot] = kind
        cache.occupations[slot] = n
        cache.shifts[slot] = shift
        cache.values[slot] = value
    return value


@njit(cache=True, inline='always')
def _admissible(n, shift, degree):
    changed = 0
    for i in range(len(n)):
        if n[i] < 0 or n[i]+shift[i] < 0:
            return False
        changed += abs(shift[i])
        if changed > degree:
            return False
    return True


@njit(cache=True, inline='always')
def _valid_step(n, shift):
    for i in range(len(n)):
        if n[i]+shift[i] < 0:
            return False
    return True


# Products refer only to lower nodes except normalization's product node;
# that branch immediately descends to first-order wave operators.
_PRODUCTS = np.array([
    [8, 0, 5, 0, 1], [9, 5, 6, 0, 1], [13, 7, 5, 0, 1],
    [14, 2, 5, 2, 0], [15, 7, 2, 2, 1],
    [17, 7, 3, 3, 1], [18, 3, 5, 3, 0],
    [19, 11, 2, 2, 1], [20, 2, 10, 2, 0],
    [21, 7, 14, 0, 0], [22, 12, 2, 2, 1], [23, 2, 12, 2, 0]], dtype=np.int64)

# Exact tensor degrees, not excitation cutoffs: a product cannot change more
# oscillator quanta than the sum of the ranks of its factors.
_DEGREES = np.array([3, 4, 1, 2, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6,
                     4, 4, 4, 5, 5, 7, 7, 7, 7, 7, 7], dtype=np.int64)


_SUM_NODES = np.array([[3, 15, 14, -1, -1, -1, -1, -1],
                       [4, 17, 18, 19, 20, 21, 22, 23]], dtype=np.int64)


@njit(cache=True, inline='always')
def _difference_within_degree(total, part, degree):
    changed = 0
    for i in range(len(total)):
        changed += abs(total[i]-part[i])
        if changed > degree:
            return False
    return True


@njit
def _product_groups(support, other, shift, data, memo):
    """Exact primitive-support joins, independent of oscillator occupations.

    Each worker owns its bounded dictionary. Hash collisions are checked against
    the complete integer shift and can only evict an entry, never change a join.
    Candidate order is identical to the original primitive-group traversal.
    """
    code = (_hash(shift, data.weights)
            ^ np.uint64((support+1)*104729+(other+1)*1000003))
    if code in memo.joins:
        stored_shift, groups = memo.joins[code]
        if (stored_shift[-2] == support and stored_shift[-1] == other
                and _same_ints(stored_shift, shift)):
            memo.join_statistics[0] += 1
            return groups
    memo.join_statistics[1] += 1
    groups = np.empty(data.offsets[support+1]-data.offsets[support], np.int64)
    count = 0
    for group in range(data.offsets[support], data.offsets[support+1]):
        # A rank-r primitive has no shift with more than r oscillator quanta.
        # Check this exact degree bound before subtraction allocation and lookup.
        if not _difference_within_degree(shift, data.shifts[group], _DEGREES[other]):
            continue
        if _find(other, shift-data.shifts[group], data) >= 0:
            groups[count] = group
            count += 1
    groups = groups[:count].copy()
    # Include conservative container/array overhead in the declared byte budget.
    size = 256+8*(len(shift)+2+len(groups))
    if size <= memo.join_limit:
        if code in memo.joins or memo.join_used[0]+size > memo.join_limit:
            memo.joins.clear()
            memo.join_used[0] = 0
        key = np.empty(len(shift)+2, np.int64)
        key[:-2] = shift
        key[-2], key[-1] = support, other
        memo.joins[code] = (key, groups)
        memo.join_used[0] += size
    return groups


# Recursive dispatchers are intentionally compiled per process: persisting
# recursive LLVM addresses in a disk cache is unsafe across interpreters.
@njit
def _field_value(kind, n, shift, data, counter, maximum_work, memo):
    slot = _cache_slot(kind, n, shift, data, memo)
    if (slot >= 0 and memo.kinds[slot] == kind
            and _same_ints(memo.occupations[slot], n)
            and _same_ints(memo.shifts[slot], shift)):
        memo.statistics[0] += 1
        return memo.values[slot]
    memo.statistics[1] += 1
    counter[0] += 1
    if counter[0] > maximum_work:
        raise MemoryError('native resonant property work budget exceeded')
    if not _admissible(n, shift, _DEGREES[kind]):
        return _remember(slot, kind, n, shift, 0j, memo)
    if kind in (0, 1):
        group = _find(kind, shift, data)
        value = 0j
        if group >= 0:
            for term in range(data.terms[group], data.terms[group+1]):
                value += _weight(term, n, data)*data.coefficients[term, 0]
        return _remember(slot, kind, n, shift, value, memo)
    if kind in (5, 6):
        if _internal(n, shift, data) != (kind == 6):
            return _remember(slot, kind, n, shift, 0j, memo)
        value = _field_value(0, n, shift, data, counter, maximum_work, memo)
        if kind == 6:
            return _remember(slot, kind, n, shift, value, memo)
    elif kind in (7, 11):
        return _remember(slot, kind, n, shift, np.conjugate(_field_value(5 if kind == 7 else 10, n+shift, -shift, data, counter, maximum_work, memo)), memo)
    elif kind == 10:
        if _internal(n, shift, data):
            return _remember(slot, kind, n, shift, 0j, memo)
        value = (_field_value(1, n, shift, data, counter, maximum_work, memo)
                 +_field_value(8, n, shift, data, counter, maximum_work, memo)
                 -_field_value(9, n, shift, data, counter, maximum_work, memo))
    elif kind == 12:
        if not _internal(n, shift, data):
            return _remember(slot, kind, n, shift, 0j, memo)
        return _remember(slot, kind, n, shift, _field_value(13, n, shift, data, counter, maximum_work, memo), memo)
    else:
        left, right = (0, 5) if kind == 8 else ((5, 6) if kind == 9 else (7, 5))
        value = 0j
        for group in _product_groups(0, 0, shift, data, memo):
            dr = data.shifts[group]
            if not _valid_step(n, dr):
                continue
            dl = shift-dr
            middle = n+dr
            rv = _field_value(right, n, dr, data, counter, maximum_work, memo)
            if rv != 0:
                value += _field_value(left, middle, dl, data, counter, maximum_work, memo)*rv
        return _remember(slot, kind, n, shift, value, memo)
    if abs(value) <= 1e-12:
        return _remember(slot, kind, n, shift, 0j, memo)
    denominator = -np.dot(data.omega, shift.astype(np.float64))
    if abs(denominator) <= data.tolerance:
        raise ValueError('unresolved native resonant property denominator')
    return _remember(slot, kind, n, shift, value/denominator, memo)


@njit
def _value(kind, n, shift, data, counter, maximum_work, memo, scratch, depth):
    counter[0] += 1
    if counter[0] > maximum_work:
        raise MemoryError('native resonant property work budget exceeded')
    result = scratch[depth]
    result[:] = 0.
    if not _admissible(n, shift, _DEGREES[kind]):
        return
    if kind in (2, 3, 4):
        if kind in (3, 4) and not data.include_electrical:
            return
        group = _find(kind, shift, data)
        if group >= 0:
            for term in range(data.terms[group], data.terms[group+1]):
                weight = _weight(term, n, data)
                if weight:
                    for component in range(len(result)):
                        result[component] += weight*data.coefficients[term, component]
        return
    if kind in (16, 24):
        for node in _SUM_NODES[0 if kind == 16 else 1]:
            if node < 0:
                break
            _value(node, n, shift, data, counter, maximum_work, memo, scratch, depth+1)
            for component in range(len(result)):
                if kind == 24 and node in (22, 23):
                    result[component] -= .5*scratch[depth+1, component]
                else:
                    result[component] += scratch[depth+1, component]
        return
    for row in _PRODUCTS:
        if row[0] != kind:
            continue
        left, right, support, right_side = row[1:]
        left_field = left in (0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        other = left if right_side else right
        primitive = -1
        if other in (2, 3, 4):
            primitive = other
        elif other in (0, 5, 6, 7):
            primitive = 0
        groups = (_product_groups(support, primitive, shift, data, memo)
                  if primitive >= 0 else np.arange(data.offsets[support], data.offsets[support+1]))
        for group in groups:
            part = data.shifts[group]
            dr = part if right_side else shift-part
            if not _valid_step(n, dr):
                continue
            dl = shift-dr
            middle = n+dr
            # Scalar mechanical contractions are evaluated once, shared across
            # every property component; no field-vector temporaries are made.
            if left_field:
                if right_side:
                    _value(right, n, dr, data, counter, maximum_work, memo, scratch, depth+1)
                    property_value = scratch[depth+1]
                    if not np.any(property_value):
                        continue
                    scalar = _field_value(left, middle, dl, data, counter, maximum_work, memo)
                else:
                    scalar = _field_value(left, middle, dl, data, counter, maximum_work, memo)
                    if scalar == 0:
                        continue
                    _value(right, n, dr, data, counter, maximum_work, memo, scratch, depth+1)
                    property_value = scratch[depth+1]
            else:
                if right_side:
                    scalar = _field_value(right, n, dr, data, counter, maximum_work, memo)
                    if scalar == 0:
                        continue
                    _value(left, middle, dl, data, counter, maximum_work, memo, scratch, depth+1)
                    property_value = scratch[depth+1]
                else:
                    _value(left, middle, dl, data, counter, maximum_work, memo, scratch, depth+1)
                    property_value = scratch[depth+1]
                    if not np.any(property_value):
                        continue
                    scalar = _field_value(right, n, dr, data, counter, maximum_work, memo)
            for component in range(len(result)):
                result[component] += scalar*property_value[component]
        return
    raise ValueError('unknown native property node')


@njit(nogil=True)
def _orders(initials, finals, data, maximum_work, memo):
    values = np.zeros((len(initials), 3, data.coefficients.shape[1]), np.complex128)
    work = np.zeros(len(initials), np.int64)
    # Acyclic property nodes need at most one row per node. Each recursive
    # child owns the next row; siblings and successive pairs reuse storage.
    scratch = np.empty((len(_DEGREES), data.coefficients.shape[1]), np.complex128)
    for i in range(len(initials)):
        counter = np.zeros(1, np.int64)
        shift = finals[i]-initials[i]
        for rank, node in enumerate((2, 16, 24)):
            if (np.sum(shift)-rank-1) % 2 == 0:
                _value(node, initials[i], shift, data, counter, maximum_work, memo, scratch, 0)
                values[i, rank] = scratch[0]
        work[i] = counter[0]
    return values, work


def _pack(reduced):
    field = reduced.field
    rng = np.random.default_rng(9371)
    weights = rng.integers(1, np.iinfo(np.uint64).max, size=len(field.frequencies), dtype=np.uint64)
    def code(row):
        return int(np.asarray(row, np.int64).astype(np.uint64)@weights)
    tables = [field.base['c'], field.base['h'], *(reduced.base[f'p{k}'] for k in range(3))]
    shifts, hashes, offsets, terms = [], [], [0], [0]
    modes, annih, creation, coefficients = [], [], [], []
    for table in tables:
        for shift, polynomial in sorted(table.items(), key=lambda item: code(item[0])):
            shifts.append(shift); hashes.append(code(shift))
            for (u, v), value in polynomial.items():
                active = [i for i, (a, b) in enumerate(zip(u, v)) if a or b]
                modes.append(active); creation.append([u[i] for i in active]); annih.append([v[i] for i in active])
                coefficients.append(np.broadcast_to(value, (reduced.components,)).copy())
            terms.append(len(modes))
        offsets.append(len(shifts))
    width = max(map(len, modes), default=0)
    def padded(rows, fill):
        result = np.full((len(rows), width), fill, np.int64)
        for i, row in enumerate(rows):
            result[i, :len(row)] = row
        return result
    mask = {n: tuple(p for p in partners if p != n) for n, partners in field.exclusions.items()}
    mask = {n: partners for n, partners in mask.items() if partners}
    mask_states = sorted(mask, key=code)
    mask_offsets, targets = [0], []
    for n in mask_states:
        targets.extend(mask[n]); mask_offsets.append(len(targets))
    count = len(weights)
    return _Data(np.asarray(field.frequencies, float), np.asarray(shifts, np.int64).reshape(-1, count),
        np.asarray(hashes, np.uint64), weights, np.asarray(offsets, np.int64), np.asarray(terms, np.int64),
        padded(modes, -1), padded(annih, 0), padded(creation, 0),
        np.asarray(coefficients, complex).reshape(-1, reduced.components),
        np.asarray(mask_states, np.int64).reshape(-1, count), np.asarray([code(s) for s in mask_states], np.uint64),
        np.asarray(mask_offsets, np.int64), np.asarray(targets, np.int64).reshape(-1, count), field.tolerance, True)


class NativeResonantProperty:
    """Compiled masked tensor-convolution action, with arbitrary property axes.

    The primitive tensor data and masks are retained once. No transition matrix
    or virtual oscillator basis is stored. Work is bounded per endpoint and
    packet actions consider only exact nonzero input components. Completed
    field contractions use at most ``field_cache_bytes`` of numerical buffers
    per executing thread; complete complex orders share a bounded
    ``moment_cache_bytes`` cache across packet bases. Either cache can be
    disabled with zero. Primitive-support joins use at most ``structural_cache_bytes``
    per worker and retain exact integer shifts. Work counters count newly
    evaluated contraction nodes.
    """
    def __init__(self, field, first, second, third, *, momentum=False, maximum_work=100000000,
                 field_cache_bytes=8*1024**2, moment_cache_bytes=32*1024**2,
                 structural_cache_bytes=8*1024**2):
        if not isinstance(field, PairReducedField):
            raise TypeError('a reduced field with direct resonance exclusions is required')
        if isinstance(maximum_work, bool) or not isinstance(maximum_work, int) or maximum_work < 1:
            raise ValueError('positive integer work budget required')
        if (isinstance(field_cache_bytes, bool) or not isinstance(field_cache_bytes, int)
                or field_cache_bytes < 0):
            raise ValueError('nonnegative integer field cache budget required')
        if (isinstance(moment_cache_bytes, bool) or not isinstance(moment_cache_bytes, int)
                or moment_cache_bytes < 0):
            raise ValueError('nonnegative integer moment cache budget required')
        if (isinstance(structural_cache_bytes, bool) or not isinstance(structural_cache_bytes, int)
                or structural_cache_bytes < 0):
            raise ValueError('nonnegative integer structural cache budget required')
        self.structural_cache_bytes = structural_cache_bytes
        self.moment_cache_bytes = moment_cache_bytes
        self._moment_cache = OrderedDict()
        self._moment_cache_lock = Lock()
        self._moment_cache_size = 0
        self.field_cache_bytes = field_cache_bytes
        count = len(field.frequencies)
        arrays = tuple(np.asarray(v) for v in (first, second, third))
        components = arrays[0].shape[0] if arrays[0].ndim == 2 else 0
        if (not count or np.any(np.asarray(field.frequencies) <= 0)
                or not np.all(np.isfinite(field.frequencies)) or not components
                or any(a.shape != (components,)+(count,)*rank or not np.all(np.isfinite(a))
                       for rank, a in enumerate(arrays, 1))):
            raise ValueError('positive frequencies and complete finite property derivatives required')
        if any(n not in field.exclusions.get(p, ()) for n, partners in field.exclusions.items()
               for p in partners if p != n):
            raise ValueError('reciprocal direct resonance exclusions required')
        reduced = PairReducedProperty(field, first, second, third, momentum=momentum)
        self.data = _pack(reduced)
        self.components = reduced.components
        self.h_symmetry_basis = reduced.h_symmetry_basis
        self.allowed_property_cosets = reduced.allowed_property_cosets
        self.maximum_work = maximum_work
        self._local = local()
        self.last_work = np.array([], np.int64)
        self._cold_correction = None

    def _field_cache(self):
        # Each worker owns a bounded direct-mapped cache. Full integer keys
        # validate every hit; hash collisions only replace entries.
        stored = getattr(self._local, 'field_cache', None)
        if stored is None or stored[0] is not self.data:
            modes = len(self.data.omega)
            per_entry = (2*modes+1)*np.dtype(np.int64).itemsize+np.dtype(complex).itemsize
            entries = self.field_cache_bytes//per_entry
            entries = 0 if not entries else 1 << (entries.bit_length()-1)
            cache = _FieldCache(np.full(entries, -1, np.int64),
                np.empty((entries, modes), np.int64), np.empty((entries, modes), np.int64),
                np.empty(entries, complex), np.zeros(2, np.int64),
                Dict.empty(key_type=types.uint64,
                           value_type=types.Tuple((types.int64[::1], types.int64[::1]))),
                np.zeros(1, np.int64), self.structural_cache_bytes, np.zeros(2, np.int64))
            self._local.field_cache = self.data, cache
        return self._local.field_cache[1]

    @property
    def last_work(self):
        return getattr(self._local, 'work', np.array([], np.int64))

    @last_work.setter
    def last_work(self, value):
        self._local.work = value

    @classmethod
    def from_gvpt2(cls, result, field, derivatives, *, modalized=None, **controls):
        """Use the existing direct exclusions and pole-first cold fundamentals.

        The mode reference, normalized wave operator and rectilinear finite
        remainder are shared with the established selected-moment provider.
        """
        from .gvpt2_spectroscopy import (
            _modalize_sonic_field, _direct_polyad_exclusions,
            _rectilinear_dvpt2_fundamental_mechanical,
        )
        omega, _, f3, f4, g1, g2 = _modalize_sonic_field(field) if modalized is None else modalized
        if not np.allclose(result.frequencies_cm, omega, rtol=1e-9, atol=1e-6):
            raise ValueError('GVPT2 and property force-field frequencies disagree')
        tensors = (f3, f4, g1, g2)
        if not all(np.all(np.isfinite(t)) for t in tensors):
            raise ValueError('finite force and metric tensors required')
        if not all(np.allclose(t, t.swapaxes(a, b), atol=1e-10, rtol=1e-12)
                   for t, a, b in ((f3, 0, 1), (f3, 1, 2), (f4, 0, 1), (f4, 1, 2),
                                   (f4, 2, 3), (g1, 0, 1), (g2, 0, 1), (g2, 2, 3))):
            raise ValueError('symmetric force and metric tensors required')
        arrays = (derivatives.first, derivatives.second, derivatives.third)
        count = len(arrays[0])
        if any(np.asarray(t).shape != (count,)+(len(omega),)*rank or not np.all(np.isfinite(t))
               for rank, t in enumerate(arrays, 1)):
            raise ValueError('complete finite property derivatives through rank three required')
        exclusions = _direct_polyad_exclusions(result)
        obj = cls(PairReducedField(omega, *tensors, exclusions), *arrays, **controls)
        if not controls.get('momentum', False) and np.max(abs(g1)) <= 1e-14:
            potential = None
            corrections = {}
            # A multi-property calculation can already contain the pure
            # mechanical channel for the same first derivative. Reuse that
            # exact order instead of contracting it a second time.
            electrical_free = [j for j in range(count)
                               if not np.any(arrays[1][j]) and not np.any(arrays[2][j])]
            mechanical_rows = []
            for row in arrays[0]:
                mechanical_rows.append(next((j for j in electrical_free
                                             if np.array_equal(row, arrays[0][j])),
                                            -2 if not np.any(row) else -1))
            def cold(initial, final, values):
                nonlocal potential
                for i, (n, f) in enumerate(zip(initial, final)):
                    if np.sum(n) or np.sum(f) != 1 or len(exclusions.get(tuple(f), ())) <= 1:
                        continue
                    mode = int(np.argmax(f))
                    if mode not in corrections:
                        if not np.any(g2):
                            if -1 not in mechanical_rows:
                                generic = np.array([0j if j == -2 else values[i, 2, j]
                                                    for j in mechanical_rows])
                            else:
                                generic_values, calls = _orders(n[None, :], f[None, :],
                                    obj.data._replace(include_electrical=False), obj.maximum_work, obj._field_cache())
                                generic = generic_values[0, 2]
                                obj.last_work[i] += int(calls[0])
                        else:
                            if potential is None:
                                potential = cls(PairReducedField(omega, f3, f4, g1, np.zeros_like(g2), exclusions),
                                                arrays[0], np.zeros_like(arrays[1]), np.zeros_like(arrays[2]),
                                                maximum_work=obj.maximum_work)
                            generic = potential.moments(n[None, :], f[None, :])[0, 2]
                            obj.last_work[i] += int(potential.last_work.sum())
                        if obj.last_work[i] > obj.maximum_work:
                            raise MemoryError('native resonant property work budget exceeded including cold correction')
                        collected = _rectilinear_dvpt2_fundamental_mechanical(omega, f3, f4, arrays[0], mode, exclusions)
                        corrections[mode] = collected-generic
                    values[i, 2] += corrections[mode]
            obj._cold_correction = cold
        return obj

    def moments(self, initial, final):
        """Reuse completed complex orders across distinct packet bases.

        Keys contain both complete occupations. Entries are never inferred from
        rates, and the bounded cache belongs to this immutable molecular model.
        """
        if not self.moment_cache_bytes:
            return self._uncached_moments(initial, final)
        initial, final = np.asarray(initial), np.asarray(final)
        if (initial.ndim != 2 or initial.shape != final.shape or initial.shape[1] != len(self.data.omega)
                or initial.dtype.kind not in 'iu' or final.dtype.kind not in 'iu'
                or np.any(initial < 0) or np.any(final < 0)):
            raise ValueError('paired nonnegative integer occupations required')
        initial, final = initial.astype(np.int64), final.astype(np.int64)
        keys = [n.tobytes()+f.tobytes() for n, f in zip(initial, final)]
        values = np.zeros((len(initial), 3, self.components), complex)
        work = np.zeros(len(initial), np.int64)
        missing = []
        with self._moment_cache_lock:
            for i, key in enumerate(keys):
                if key in self._moment_cache:
                    values[i] = self._moment_cache[key]
                    self._moment_cache.move_to_end(key)
                else:
                    missing.append(i)
        if missing:
            fresh = self._uncached_moments(initial[missing], final[missing])
            values[missing] = fresh
            work[missing] = self.last_work
            with self._moment_cache_lock:
                for i, value in zip(missing, fresh):
                    key = keys[i]
                    if key in self._moment_cache:
                        continue
                    # Include ndarray and bytes object storage and a conservative
                    # allowance for the ordered dictionary entry.
                    value = value.copy()
                    size = sys.getsizeof(key)+sys.getsizeof(value)+256
                    if size > self.moment_cache_bytes:
                        continue
                    self._moment_cache[key] = value
                    self._moment_cache_size += size
                    while self._moment_cache_size > self.moment_cache_bytes:
                        old_key, old_value = self._moment_cache.popitem(last=False)
                        self._moment_cache_size -= sys.getsizeof(old_key)+sys.getsizeof(old_value)+256
        self.last_work = work
        return values

    def _uncached_moments(self, initial, final):
        initial, final = np.asarray(initial), np.asarray(final)
        if (initial.ndim != 2 or initial.shape != final.shape or initial.shape[1] != len(self.data.omega)
                or initial.dtype.kind not in 'iu' or final.dtype.kind not in 'iu'
                or np.any(initial < 0) or np.any(final < 0)):
            raise ValueError('paired nonnegative integer occupations required')
        selected = np.asarray([_gf2_reduce(_parity_mask(f-n), self.h_symmetry_basis)
                               in self.allowed_property_cosets for n, f in zip(initial, final)], bool)
        values = np.zeros((len(initial), 3, self.components), complex)
        self.last_work = np.zeros(len(initial), np.int64)
        if np.any(selected):
            values[selected], self.last_work[selected] = _orders(np.asarray(initial[selected], np.int64),
                np.asarray(final[selected], np.int64), self.data, self.maximum_work, self._field_cache())
        if self._cold_correction is not None:
            self._cold_correction(initial, final, values)
        return values

    def operators(self, basis, *, coordinate_id, cached_columns=32, maximum_column_work=1000000000,
                  maximum_workspace_bytes=256*1024**2, maximum_input_sources=128,
                  workers=1, cache_directory=None, progress=None,
                  known_moments=None, reference_id=None):
        basis = np.asarray(basis)
        if (basis.ndim != 2 or not len(basis) or basis.shape[1] != len(self.data.omega) or basis.dtype.kind not in 'iu'
                or np.any(basis < 0) or len(np.unique(basis, axis=0)) != len(basis)
                or not isinstance(coordinate_id, str) or not coordinate_id.strip()):
            raise ValueError('unique nonnegative basis and coordinate identity required')
        basis = basis.copy()
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
               (cached_columns, maximum_column_work, maximum_workspace_bytes, maximum_input_sources, workers)):
            raise ValueError('positive integer column cache and work budgets required')
        if progress is not None and not callable(progress):
            raise ValueError('callable column progress observer required')
        if known_moments is not None and (not isinstance(reference_id, str) or not reference_id.strip()):
            raise ValueError('a provenance identity is required for supplied bare reference moments')
        column_bytes = len(basis)*self.components*np.dtype(complex).itemsize
        if column_bytes*2 > maximum_workspace_bytes:
            raise MemoryError('native property source packet exceeds workspace budget')
        cached_columns = min(cached_columns, max(1, maximum_workspace_bytes//(2*column_bytes)))
        directory = None
        identity = hashlib.sha256()
        for value in (*self.data, basis, coordinate_id, self._cold_correction is not None, reference_id or ''):
            array = np.ascontiguousarray(value)
            identity.update(str((array.shape, array.dtype)).encode())
            identity.update(array.tobytes())
        for name in ('native_resonant_property.py', 'gvpt2_spectroscopy.py',
                     'reduced_transition_polynomials.py', 'reduced_resonant_transitions.py'):
            identity.update(Path(__file__).with_name(name).read_bytes())
        digest = identity.hexdigest()
        if cache_directory is not None:
            directory = Path(cache_directory)/digest
            directory.mkdir(parents=True, exist_ok=True)
        def save(path, values, completed, work):
            temporary = path.with_name(f'.{path.name}.{os.getpid()}.pending.npz')
            np.savez(temporary, identity=digest, values=values, completed=completed, work=work)
            temporary.replace(path)
        def restore(path):
            with np.load(path, allow_pickle=False) as stored:
                values = stored['values'].copy()
                completed, work = int(stored['completed']), int(stored['work'])
                if (str(stored['identity']) != digest or values.shape != (len(basis), self.components)
                        or not np.all(np.isfinite(values)) or not 0 <= completed <= len(basis) or work < 0):
                    raise ValueError('inconsistent native property column checkpoint')
            return values, completed, work
        cache, pending, lock = OrderedDict(), {}, Lock()
        def column(source, adjoint):
            key = source
            with lock:
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
                owner = key not in pending
                if owner:
                    pending[key] = Future()
                future = pending[key]
            if not owner:
                return future.result()
            try:
                values = np.zeros((len(basis), self.components), complex)
                work, completed = 0, 0
                final_path = None if directory is None else directory/f'column_{source:09d}.npz'
                partial_path = None if directory is None else directory/f'partial_{source:09d}.npz'
                if final_path is not None and final_path.exists():
                    values, completed, work = restore(final_path)
                    if completed != len(basis):
                        raise ValueError('incomplete final native property column')
                elif partial_path is not None and partial_path.exists():
                    values, completed, work = restore(partial_path)
                if work > maximum_column_work:
                    raise MemoryError('native dipole source-column work budget exceeded')
                if progress is not None:
                    progress(dict(source=int(source), completed=completed, total=len(basis), work=work,
                                  restored=completed > 0))
                for first in range(completed, len(basis), 16):
                    final = basis[first:first+16]
                    initial = np.repeat(basis[source:source+1], len(final), axis=0)
                    # The less excited endpoint convention is shared with the
                    # acquired Hermitian bare-moment archive, including the
                    # pole-first cold fundamental convention.
                    reverse = np.array([(sum(f), first+j) < (sum(n), source)
                                        for j, (n, f) in enumerate(zip(initial, final))])
                    n, f = initial.copy(), final.copy()
                    n[reverse], f[reverse] = final[reverse], initial[reverse]
                    block = np.zeros((len(final), self.components), complex)
                    missing = []
                    for j in range(len(final)):
                        target = first+j
                        saved = None if known_moments is None else known_moments.get((min(source, target), max(source, target)))
                        if saved is None:
                            missing.append(j)
                            continue
                        saved = np.asarray(saved)
                        if saved.shape != (3, self.components) or not np.all(np.isfinite(saved)):
                            raise ValueError('compatible finite reference moment orders required')
                        # Archive entries are indexed by increasing basis index;
                        # n/f above use the less-excited-endpoint convention.
                        value = saved.sum(axis=0)
                        native_source = target if reverse[j] else source
                        native_target = source if reverse[j] else target
                        block[j] = value if native_source <= native_target else value.conjugate()
                    if missing:
                        block[missing] = self.moments(n[missing], f[missing]).sum(axis=1)
                        work += int(self.last_work.sum())
                    if work > maximum_column_work:
                        raise MemoryError('native dipole source-column work budget exceeded')
                    block[reverse] = block[reverse].conjugate()
                    values[first:first+len(final)] = block
                    completed = first+len(final)
                    if partial_path is not None:
                        save(partial_path, values, completed, work)
                    if progress is not None:
                        progress(dict(source=int(source), completed=completed, total=len(basis), work=work,
                                      restored=False))
                if final_path is not None:
                    save(final_path, values, len(basis), work)
                    partial_path.unlink(missing_ok=True)
                # Canonical Hermitian completion is explicit; no probabilities
                # are converted back into phases.
                if adjoint:
                    values = values.copy()
                values.setflags(write=False)
                with lock:
                    cache[key] = values
                    while len(cache) > cached_columns:
                        cache.popitem(last=False)
                    pending.pop(key)
                    future.set_result(values)
                return values
            except BaseException as error:
                with lock:
                    pending.pop(key, None)
                    future.set_exception(error)
                raise
        packet_cache = {'input': None, 'value': None}
        packet_lock = Lock()
        def action(x, component, adjoint=False):
            x = np.asarray(x).reshape(len(basis), -1)
            sources = np.flatnonzero(np.any(x != 0, axis=1))
            if (len(sources) > maximum_input_sources
                    or x.nbytes*2+column_bytes*(x.shape[1]+1) > maximum_workspace_bytes//2):
                raise MemoryError('native property packet action exceeds source or workspace budget')
            with packet_lock:
                previous = packet_cache['input']
                if previous is not None and previous.shape == x.shape and np.array_equal(previous, x):
                    return packet_cache['value'][:, component, :]
            # Accumulate all property components together. Subsequent dipole
            # axes reuse this packet even when its input has more occupied
            # columns than the bounded column cache can retain.
            result = np.zeros((len(basis), self.components, x.shape[1]), complex)
            if workers == 1 or len(sources) < 2:
                for source in sources:
                    result += column(source, adjoint)[:, :, None]*x[source]
            else:
                # Bound queued/completed columns by available memory, allowing
                # free workers to advance while an earlier column is costly.
                # Accumulate in input order for reproducible reductions.
                buffered_columns = max(1, maximum_workspace_bytes//(4*column_bytes))
                active_workers = min(workers, buffered_columns)
                with ThreadPoolExecutor(max_workers=active_workers) as executor:
                    for first in range(0, len(sources), buffered_columns):
                        selected = sources[first:first+buffered_columns]
                        for source, values in zip(selected, executor.map(lambda s: column(s, adjoint), selected)):
                            result += values[:, :, None]*x[source]
            result.setflags(write=False)
            with packet_lock:
                packet_cache['input'] = x.copy()
                packet_cache['value'] = result
            return result[:, component, :]
        operators = []
        for component in range(self.components):
            op = LinearOperator((len(basis), len(basis)),
                matvec=lambda x, k=component: action(x, k)[:, 0],
                matmat=lambda x, k=component: action(x, k),
                rmatvec=lambda x, k=component: action(x, k, True)[:, 0], dtype=complex)
            op.coordinate_id = coordinate_id
            operators.append(op)
        return tuple(operators)
