"""Exact linear-operator prefix masses for bounded TI selection experiments.

All unassigned occupations are traced analytically. Prefix masses bound every
descendant line. A finite child cap leaves an explicitly accounted tail; it is
never silently renormalized. This experimental owner does not alter production
TI checkpoint formats or the default ranking strategy.
"""
import heapq
from time import perf_counter
import numpy as np
from numba import njit
from .gaussian_partial_trace import gaussian_partial_number_trace
from .gaussian_hermite import hermite_recurrence_plan, _hermite_recurrence_values


@njit(cache=True)
def _inserted_coefficients(matrix, linear, constant, left, right, intrinsic,
                           rows, edge_offsets, edges, offsets, terms):
    values = _hermite_recurrence_values(matrix, linear, constant, rows, edge_offsets, edges)
    result = np.zeros(len(offsets)-1, np.complex128)
    for row in range(len(result)):
        for t in range(offsets[row], offsets[row+1]):
            k, l = int(terms[t, 1]), int(terms[t, 2])
            coefficient = intrinsic if k < 0 else 0j
            for channel in range(len(left)):
                if k < 0:
                    coefficient += left[channel, 0]*right[channel, 0]
                elif l < 0:
                    coefficient += (left[channel, k+1]*right[channel, 0]
                                    + left[channel, 0]*right[channel, k+1])
                else:
                    coefficient += left[channel, k+1]*right[channel, l+1]
            result[row] += values[int(terms[t, 0])]*terms[t, 3]*coefficient
    return result


class GaussianOperatorPrefixMass:
    def __init__(self, matrix, linear, log_ground, constants, insertions, *,
                 order=None, maximum_integrals=200000):
        m, r = np.asarray(matrix), np.asarray(linear)
        c, h = np.asarray(constants, complex), np.asarray(insertions, complex)
        order = np.arange(len(r)) if order is None else np.asarray(order)
        if (order.dtype.kind not in 'iu' or order.shape != r.shape
                or not np.array_equal(np.sort(order), np.arange(len(r)))
                or c.ndim != 1 or not len(c) or h.shape != (len(c), len(r))
                or not np.all(np.isfinite(c)) or not np.all(np.isfinite(h))
                or not isinstance(maximum_integrals, int) or maximum_integrals < 1):
            raise ValueError('finite linear bank, complete mode permutation and integral budget required')
        # The existing partial-trace owner validates the Gaussian contract.
        gaussian_partial_number_trace(m, r, log_ground, np.arange(len(r)), np.ones(len(r)))
        self.matrix, self.linear = m[np.ix_(order, order)], r[order]
        self.constant, self.insertions = c, h[:, order]
        self.order, self.log_ground = order.copy(), float(log_ground)
        self.maximum_integrals = maximum_integrals
        self.reduced = {}
        self.evaluations = 0
        self.total = float(self.masses(np.empty((1, 0), dtype=int))[0])
        if not np.isfinite(self.total) or self.total <= 0:
            raise ValueError('positive finite operator-bank norm required')

    def _density(self, k):
        if k in self.reduced:
            return self.reduced[k]
        bath = np.arange(k, len(self.linear)); nb = len(bath)
        d = gaussian_partial_number_trace(self.matrix, self.linear, self.log_ground,
                                         bath, np.ones(nb))
        h, c = self.insertions, self.constant
        left = np.zeros((len(c), 1+2*k), complex); right = np.zeros_like(left)
        left[:, 0] = c.conj()+h[:, bath].conj() @ d['mean'][:nb]
        right[:, 0] = c+h[:, bath] @ d['mean'][nb:]
        left[:, 1:1+k] = h[:, :k].conj(); right[:, 1+k:] = h[:, :k]
        left[:, 1:] += h[:, bath].conj() @ d['covariance'][:nb] @ d['response']
        right[:, 1:] += h[:, bath] @ d['covariance'][nb:] @ d['response']
        intrinsic = np.einsum('ai,ij,aj->', h[:, bath].conj(),
                             d['covariance'][:nb, nb:], h[:, bath])
        self.reduced[k] = d, left, right, intrinsic
        return self.reduced[k]

    def masses(self, prefixes):
        prefixes = np.asarray(prefixes)
        if (prefixes.ndim != 2 or prefixes.dtype.kind not in 'iu'
                or prefixes.shape[1] > len(self.linear) or np.any(prefixes < 0)):
            raise ValueError('bounded nonnegative integer prefix rows required')
        d, left, right, intrinsic = self._density(prefixes.shape[1])
        words = np.concatenate([prefixes, prefixes], axis=1)
        plan = hermite_recurrence_plan(words, 2, self.maximum_integrals)
        values = _inserted_coefficients(d['matrix'], d['linear'], d['constant'],
                                       left, right, intrinsic, *plan)
        self.evaluations += len(prefixes)
        scale = max(float(np.max(abs(values), initial=0.)), getattr(self, 'total', 0.), 1e-300)
        if np.max(abs(values.imag), initial=0.) > 1e-9*scale or np.min(values.real, initial=0.) < -1e-9*scale:
            raise ArithmeticError('prefix masses lose positive real accuracy')
        return np.maximum(values.real, 0.)

    def best_first(self, *, maximum_lines=100, maximum_nodes=2000,
                   maximum_per_mode=12, maximum_seconds=60.):
        if (any(not isinstance(x, int) or x < 1 for x in
                (maximum_lines, maximum_nodes, maximum_per_mode))
                or not np.isfinite(maximum_seconds) or maximum_seconds <= 0):
            raise ValueError('positive explicit line, node, occupation and time budgets required')
        start = perf_counter(); heap = [(-self.total, ())]; lines = []; strengths = []
        tail = 0.; expanded = 0
        while heap and len(lines) < maximum_lines and expanded < maximum_nodes:
            if perf_counter()-start > maximum_seconds:
                break
            negative, prefix = heapq.heappop(heap); mass = -negative
            if len(prefix) == len(self.linear):
                lines.append(prefix); strengths.append(mass)
                continue
            children = np.array([prefix+(q,) for q in range(maximum_per_mode+1)], int)
            try:
                weights = self.masses(children)
            except MemoryError:
                heapq.heappush(heap, (negative, prefix))
                break
            if weights.sum() > mass+1e-9*self.total:
                raise ArithmeticError('child masses exceed parent mass')
            tail += max(0., mass-float(weights.sum())); expanded += 1
            for row, value in zip(children, weights):
                if value > 0:
                    heapq.heappush(heap, (-float(value), tuple(row)))
        states = np.array(lines, int).reshape(-1, len(self.linear))
        original = np.empty_like(states); original[:, self.order] = states
        frontier = sum(-x[0] for x in heap)
        return dict(states=original, strengths=np.array(strengths),
                    captured_fraction=sum(strengths)/self.total,
                    frontier_strength=frontier, capped_tail_strength=tail,
                    mass_balance_error=abs(sum(strengths)+frontier+tail-self.total),
                    expanded_nodes=expanded, seconds=perf_counter()-start,
                    exact_order_within_capped_domain=True,
                    globally_sorted_without_cap_certified=tail == 0.)
