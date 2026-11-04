"""Periodic rotor coupled to a complete frozen-metric harmonic reference.

For one periodic coordinate theta, the exact constant-metric unitary
exp(-i*m*c.q), c=G_ff^-1 G_f,theta, removes its mixed momentum terms. Fourier
potential blocks acquire exp(i*(m-n)*c.q); these factors retain the coupling
to every harmonic coordinate. No diagonal effective-mass approximation is
inserted. The fast harmonic reference may itself contain arbitrary mixing.

An active-preserving covariance factorization confines scalar-potential
dependence to a small fast-coordinate factor. A further orthogonal bath
rotation confines the gauge displacement to one bath oscillator. This makes
its exact infinite-oscillator displacement matrix sparse in a total-quanta
basis. Projected displacement matrices are not renormalized to be unitary.

The owner defines a frozen-metric nuclear model. It neither supplies variable
curvilinear metric fields nor identifies a molecular permutation symmetry.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr
from scipy.special import eval_genlaguerre, gammaln
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from .covariance_adapted_bath import covariance_adapted_frame, oscillator_annihilators
from .coupled_bath_galerkin import _TensorSumOperator


def periodic_momentum_frame(kinetic_cm, force_cm, periodic_index, active_fast_count):
    """Complete canonical reference with exact rotor momentum completion."""
    g, h = np.asarray(kinetic_cm, float), np.asarray(force_cm, float)
    if (g.ndim != 2 or h.shape != g.shape or g.shape[0] != g.shape[1]
            or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h))
            or not np.allclose(g, g.T) or not np.allclose(h, h.T)
            or min(np.linalg.eigvalsh(g)) <= 0):
        raise ValueError('Finite symmetric complete force and positive kinetic matrices required')
    if not isinstance(periodic_index, (int, np.integer)) or isinstance(periodic_index, bool) or not 0 <= periodic_index < len(g):
        raise ValueError('Explicit periodic coordinate index required')
    fast = np.delete(np.arange(len(g)), periodic_index)
    gf, hf = g[np.ix_(fast, fast)], h[np.ix_(fast, fast)]
    c = np.linalg.solve(gf, g[fast, periodic_index])
    reduced = float(g[periodic_index, periodic_index]-g[periodic_index, fast] @ c)
    if reduced <= 0:
        raise ValueError('Positive Schur rotor coefficient required')
    frame = covariance_adapted_frame(gf, hf, active_fast_count)
    d = active_fast_count
    phase = frame['coordinate_map'].T @ c
    rotation = np.eye(len(gf))
    if np.linalg.norm(phase[d:]) > 1e-14:
        rotation[d:, d:], _ = qr(phase[d:, None], mode='full')
    frame['coordinate_map'] = frame['coordinate_map'] @ rotation
    frame['momentum_map'] = frame['momentum_map'] @ rotation
    frame['number_force_cm'] = rotation.T @ frame['number_force_cm'] @ rotation
    frame['active_map'] = frame['coordinate_map'][:d, :d]
    phase = frame['coordinate_map'].T @ c
    leakage = float(np.max(abs(phase[d+1:]), initial=0.))
    if leakage > 1e-10:
        raise ValueError('Gauge bath rotation failed')
    frame.update(rotor_kinetic_cm=reduced, gauge_coordinate=c,
                 gauge_phase=phase, fast_indices=fast,
                 gauge_bath_leakage=leakage, periodic_index=periodic_index)
    return frame


def one_axis_displacement(occupations, phase, axis=0):
    """Exact <n|exp(i*phase*z_axis)|m> in a finite product occupation set."""
    states = tuple(tuple(int(v) for v in state) for state in occupations)
    if not states or not np.isfinite(phase) or not 0 <= axis < len(states[0]):
        raise ValueError('Finite oscillator displacement and occupations required')
    groups = {}
    for i, state in enumerate(states):
        key = state[:axis]+state[axis+1:]
        groups.setdefault(key, []).append((i, state[axis]))
    rows, columns, values = [], [], []
    for members in groups.values():
        for i, n in members:
            for j, m in members:
                lo, hi = min(n, m), max(n, m)
                value = (np.exp(-phase**2/4 + .5*(gammaln(lo+1)-gammaln(hi+1)))
                         *(1j*phase/np.sqrt(2))**(hi-lo)
                         *eval_genlaguerre(lo, hi-lo, phase**2/2))
                if abs(value) > 1e-15:
                    rows.append(i); columns.append(j); values.append(value)
    return csr_matrix((values, (rows, columns)), shape=(len(states), len(states)))


def oscillator_displacement(occupations, phases):
    """Dense small-factor matrix of exp(i*sum_j phases_j*z_j)."""
    states = np.asarray(occupations, int)
    phases = np.asarray(phases, float)
    if states.ndim != 2 or phases.shape != (states.shape[1],) or not np.all(np.isfinite(phases)):
        raise ValueError('Compatible oscillator states and finite phases required')
    result = np.ones((len(states), len(states)), complex)
    for axis, t in enumerate(phases):
        n, m = states[:, axis, None], states[None, :, axis]
        lo, hi = np.minimum(n, m), np.maximum(n, m)
        result *= (np.exp(-t*t/4 + .5*(gammaln(lo+1)-gammaln(hi+1)))
                   *(1j*t/np.sqrt(2))**(hi-lo)*eval_genlaguerre(lo, hi-lo, t*t/2))
    return result


@dataclass
class PeriodicCoupledBath:
    """Matrix-free Fourier blocks and full mixed harmonic oscillator factors."""
    modes: np.ndarray
    fast_terms: tuple
    potential_blocks: dict
    rotor_kinetic_cm: float
    active_dimension: int
    bath_dimension: int

    def operator(self):
        return fourier_tensor_operator(self.modes, self.active_dimension, self.bath_dimension,
            {k: ((a, b),) for k, (a, b) in self.potential_blocks.items()},
            fast_terms=self.fast_terms, rotor_kinetic_cm=self.rotor_kinetic_cm)


def fourier_tensor_operator(modes, active_dimension, bath_dimension, blocks, *,
                            average_blocks=None, product_blocks=None, fast_terms=(), rotor_kinetic_cm=0.):
    """Fourier tensor operator, optionally weighted by (m+n)/2 per block."""
    modes = np.asarray(modes, int)
    if len(modes) < 2 or len(np.unique(modes)) != len(modes):
        raise ValueError('At least two distinct Fourier modes required')
    na, nb = active_dimension, bath_dimension
    size = len(modes)*na*nb
    active_identity = np.eye(na)
    def compact(terms):
        # Pure bath contributions all carry a scalar active identity. Sum
        # their sparse bath matrices once, rather than repeating the same
        # dense active product for every q_i*q_j or p_i*p_j moment.
        combined = csr_matrix((nb, nb), dtype=complex)
        other = []
        for aa, bb in terms:
            scale = aa[0, 0]
            if np.array_equal(aa, scale*active_identity):
                combined += scale*bb
            else:
                other.append((aa, bb))
        if combined.nnz:
            other.append((active_identity, combined))
        return tuple(other)
    fast_terms = compact(fast_terms)
    blocks = {k: compact(v) for k, v in blocks.items()}
    average_blocks = None if average_blocks is None else {k: compact(v) for k, v in average_blocks.items()}
    product_blocks = None if product_blocks is None else {k: compact(v) for k, v in product_blocks.items()}
    pairs = {}
    for i, m in enumerate(modes):
        for j, n in enumerate(modes):
            k = int(m-n)
            if (k in blocks or (average_blocks is not None and k in average_blocks)
                    or (product_blocks is not None and k in product_blocks)):
                pairs.setdefault(k, ([], []))[0].append(i)
                pairs[k][1].append(j)
    def apply_factor(a, b, x):
        # One BLAS contraction and one sparse contraction for all retained
        # rotor modes at once; no repeated per-mode oscillator products.
        nm = len(x)
        right = (b @ x.transpose(2, 0, 1).reshape(nb, nm*na)).reshape(nb, nm, na).transpose(1, 2, 0)
        if a is active_identity:
            return right
        return (a @ right.transpose(1, 0, 2).reshape(na, nm*nb)).reshape(na, nm, nb).transpose(1, 0, 2)
    def action(vector):
        x = np.asarray(vector).reshape(len(modes), na, nb)
        y = (.5*rotor_kinetic_cm*modes**2)[:, None, None]*x
        y = np.array(y, dtype=complex)
        for a, b in fast_terms:
            y += apply_factor(a, b, x)
        for k, (target, source) in pairs.items():
            for a, b in blocks.get(k, ()):
                y[target] += apply_factor(a, b, x[source])
            if average_blocks is not None:
                average = .5*(modes[target]+modes[source])
                for a, b in average_blocks.get(k, ()):
                    y[target] += average[:, None, None]*apply_factor(a, b, x[source])
            if product_blocks is not None:
                product = modes[target]*modes[source]
                for a, b in product_blocks.get(k, ()):
                    y[target] += product[:, None, None]*apply_factor(a, b, x[source])
        return y.ravel()
    operator = LinearOperator((size, size), matvec=action, dtype=complex)
    base_diagonal = sum((np.outer(np.diag(aa), bb.diagonal()) for aa, bb in fast_terms),
                        start=np.zeros((na, nb), complex))
    diagonal = np.broadcast_to(base_diagonal, (len(modes), na, nb)).copy()
    diagonal += (.5*rotor_kinetic_cm*modes**2)[:, None, None]
    for aa, bb in blocks.get(0, ()):
        diagonal += np.outer(np.diag(aa), bb.diagonal())[None]
    if average_blocks is not None:
        for aa, bb in average_blocks.get(0, ()):
            diagonal += modes[:, None, None]*np.outer(np.diag(aa), bb.diagonal())[None]
    if product_blocks is not None:
        for aa, bb in product_blocks.get(0, ()):
            diagonal += modes[:, None, None]**2*np.outer(np.diag(aa), bb.diagonal())[None]
    operator.diagonal_values = diagonal.ravel()
    return operator


class DisplacedOscillatorPolynomials:
    """Exact projected fast Weyl polynomials including the gauge displacement.

Every factor uses an exterior one-quantum space on both sides of the exact
displacement. This retains intermediates absent from truncated operator
products. Coordinate products commute with the displacement; momentum
products are explicitly Weyl symmetrized. Coefficients may be complex Fourier
amplitudes; Hermiticity belongs to the conjugate pair of complete blocks.
"""
    def __init__(self, active_count, bath_count, active_total, bath_total, phases):
        self.active_count = active_count
        self.dimension = active_count+bath_count
        self.factors = []
        for dim, total, phase in ((active_count, active_total, np.asarray(phases[:active_count])),
                                  (bath_count, bath_total, np.asarray(phases[active_count:]))):
            retained, _ = oscillator_annihilators(dim, total)
            extended, ann = oscillator_annihilators(dim, total+1)
            nr, ne = len(retained), len(extended)
            identity = csr_matrix((np.ones(nr), (np.arange(nr), np.arange(nr))), shape=(ne, nr))
            if np.count_nonzero(abs(phase)>1e-12) <= 1:
                axis = int(np.argmax(abs(phase)))
                displacement = one_axis_displacement(extended, phase[axis], axis)
            else:
                if dim > active_count:
                    raise ValueError('Bath gauge must be aligned onto a single oscillator')
                displacement = oscillator_displacement(extended, phase)
            ops = {'I': identity}
            for i, op in enumerate(ann):
                ops['Q', i] = ((op+op.T)/np.sqrt(2))[:, :nr].tocsr()
                ops['P', i] = (1j*(op.T-op)/np.sqrt(2))[:, :nr].tocsr()
            self.factors.append((nr, displacement, ops))
        self.cache = {}

    def _factor(self, left=None, right=None):
        key = (left, right)
        if key in self.cache:
            return self.cache[key]
        result = []
        d = self.active_count
        for part, (_, displacement, ops) in enumerate(self.factors):
            def select(label):
                if label is None:
                    return ops['I']
                kind, axis = label
                if (axis < d) != (part == 0):
                    return ops['I']
                return ops[kind, axis if part == 0 else axis-d]
            l, r = select(left), select(right)
            if part == 0:
                value = np.asarray(l.conj().T @ (displacement @ r.toarray()))
            else:
                value = (l.conj().T @ displacement @ r).tocsr()
            result.append(value)
        self.cache[key] = tuple(result)
        return self.cache[key]

    def terms(self, *, constant=0., linear=None, second=None,
              momentum_linear=None, momentum_coordinate=None, momentum_second=None):
        n = self.dimension
        l = np.zeros(n) if linear is None else np.asarray(linear, complex)
        s = np.zeros((n, n)) if second is None else np.asarray(second, complex)
        p = np.zeros(n) if momentum_linear is None else np.asarray(momentum_linear, complex)
        w = np.zeros((n, n)) if momentum_coordinate is None else np.asarray(momentum_coordinate, complex)
        pp = np.zeros((n, n)) if momentum_second is None else np.asarray(momentum_second, complex)
        if (l.shape != (n,) or p.shape != (n,) or s.shape != (n, n) or w.shape != s.shape
                or pp.shape != s.shape or not all(np.all(np.isfinite(x)) for x in (l, s, p, w, pp))
                or not np.allclose(s, s.T) or not np.allclose(pp, pp.T) or not np.isfinite(constant)):
            raise ValueError('Finite compatible displaced Weyl coefficients required')
        terms = []
        def add(coefficient, left=None, right=None):
            if abs(coefficient) <= 1e-14:
                return
            aa, bb = self._factor(left, right)
            terms.append((coefficient*aa, bb))
        add(constant)
        for i in range(n):
            add(l[i], ('Q', i))
            add(p[i]/2, ('P', i)); add(p[i]/2, right=('P', i))
            for j in range(n):
                add(s[i, j]/2, ('Q', i), ('Q', j))
                add(pp[i, j]/2, ('P', i), ('P', j))
                add(w[i, j]/2, ('P', i), ('Q', j))
                add(w[i, j]/2, ('Q', j), ('P', i))
        if not terms:
            return ()
        na, nb = self.factors[0][0], self.factors[1][0]
        return _TensorSumOperator(terms, na, nb).terms


__all__ = ['periodic_momentum_frame', 'one_axis_displacement',
           'oscillator_displacement', 'PeriodicCoupledBath',
           'fourier_tensor_operator', 'DisplacedOscillatorPolynomials']
