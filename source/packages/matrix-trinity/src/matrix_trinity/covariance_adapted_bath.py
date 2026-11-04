"""Complete harmonic vacuum with an active-coordinate oscillator factorization.

The active coordinates are preserved as linear observables, not selected normal
modes. An orthogonal rotation of dimensionless full normal coordinates puts
these observables in the first factor. The resulting harmonic Hamiltonian is
number conserving, including its mixed active/bath terms. Its exact vacuum and
every one-quantum excitation therefore fit in a small product space even when
a statically relaxed coordinate partition has very strong kinetic mixing.

This local linear representation does not impose angular boundary conditions
or define a curvilinear metric away from its reference geometry.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix, eye
from .kinetic_bath_response import _normal
from .coupled_bath_galerkin import _occupations, _TensorSumOperator


def covariance_adapted_frame(kinetic_cm, force_cm, active_count):
    """Return a canonical oscillator frame preserving the first active rows."""
    w, coordinate, momentum = _normal(kinetic_cm, force_cm)
    n = len(w)
    if isinstance(active_count, bool) or not isinstance(active_count, (int, np.integer)):
        raise ValueError('Integer active dimension required')
    if not 0 < active_count < n:
        raise ValueError('Nonempty active and bath spaces required')
    if np.linalg.matrix_rank(coordinate[:active_count]) != active_count:
        raise ValueError('Independent active coordinates required')
    rotation, _ = qr(coordinate[:active_count].T, mode='full')
    qmap = coordinate @ rotation
    pmap = momentum @ rotation
    omega = (rotation.T * w) @ rotation
    canonical_error = float(np.max(abs(qmap.T @ pmap - np.eye(n))))
    leakage = float(np.max(abs(qmap[:active_count, active_count:])))
    if canonical_error > 1e-8 or leakage > 1e-10:
        raise ValueError('Active-preserving canonical factorization failed')
    return dict(frequencies_cm=w, coordinate_map=qmap, momentum_map=pmap,
                number_force_cm=omega, zero_point_cm=float(w.sum()/2),
                covariance=qmap @ qmap.T / 2,
                active_map=qmap[:active_count, :active_count],
                active_count=active_count, canonical_error=canonical_error,
                active_bath_coordinate_leakage=leakage)


def oscillator_annihilators(dimension, total_quanta):
    """Annihilation operators in a downward-closed total-quanta space."""
    if isinstance(total_quanta, bool) or not isinstance(total_quanta, (int, np.integer)) or total_quanta < 0:
        raise ValueError('Nonnegative integer oscillator cutoff required')
    states = _occupations(dimension, total_quanta)
    lookup = {s: i for i, s in enumerate(states)}
    result = []
    for axis in range(dimension):
        rows, columns, values = [], [], []
        for column, state in enumerate(states):
            if state[axis]:
                lower = list(state)
                lower[axis] -= 1
                rows.append(lookup[tuple(lower)])
                columns.append(column)
                values.append(np.sqrt(state[axis]))
        result.append(csr_matrix((values, (rows, columns)), shape=(len(states), len(states))))
    return states, result


def coupled_number_hamiltonian(frame, active_total, bath_total, *, active_potential_cm=None):
    """Build the full mixed number Hamiltonian, relative to the harmonic ZPE.

An explicitly supplied active matrix is added without any hidden subtraction.
The caller owns its quadrature, harmonic counterterm, and validity domain.
"""
    omega = np.asarray(frame['number_force_cm'], float)
    d = frame['active_count']
    astates, a = oscillator_annihilators(d, active_total)
    bstates, b = oscillator_annihilators(len(omega)-d, bath_total)
    na, nb = len(astates), len(bstates)
    ia, ib = eye(na, format='csr'), eye(nb, format='csr')
    ha, hb = csr_matrix((na, na)), csr_matrix((nb, nb))
    for i in range(d):
        for j in range(d):
            ha += omega[i, j] * (a[i].T @ a[j])
    for i in range(len(b)):
        for j in range(len(b)):
            hb += omega[d+i, d+j] * (b[i].T @ b[j])
    hactive = ha.toarray()
    if active_potential_cm is not None:
        v = np.asarray(active_potential_cm, float)
        if v.shape != (na, na) or not np.all(np.isfinite(v)) or not np.allclose(v, v.T):
            raise ValueError('Finite symmetric active potential matrix required')
        hactive += v
    terms = [(hactive, ib), (ia.toarray(), hb)]
    for i in range(d):
        for j in range(len(b)):
            terms.extend([(omega[i, d+j]*a[i].T.toarray(), b[j]),
                          (omega[i, d+j]*a[i].toarray(), b[j].T.tocsr())])
    return dict(hamiltonian=_TensorSumOperator(terms, na, nb),
                active_occupations=astates, bath_occupations=bstates,
                active_annihilators=a, bath_annihilators=b,
                active_dimension=na, bath_dimension=nb)


def quadratic_coordinate_terms(active_count, active_total, bath_total,
                               dimension, *, constant=0., linear=None,
                               second=None, momentum_coordinate=None):
    """Tensor factors for a Weyl polynomial in dimensionless oscillator axes.

Return constant + l.x + x.S.x/2 + sum_ij W_ij {p_i,x_j}/2.
Second moments include exterior one-quantum intermediates before restriction;
they are never products of truncated position/momentum matrices.
"""
    from .coupled_bath_galerkin import harmonic_bath_matrices
    d = active_count
    a = harmonic_bath_matrices(np.eye(d), np.eye(d), active_total)
    b = harmonic_bath_matrices(np.eye(dimension-d), np.eye(dimension-d), bath_total)
    na, nb = len(a['occupations']), len(b['occupations'])
    ia, ib = eye(na, format='csr'), eye(nb, format='csr')
    l = np.zeros(dimension) if linear is None else np.asarray(linear, float)
    s = np.zeros((dimension, dimension)) if second is None else np.asarray(second, float)
    w = np.zeros((dimension, dimension)) if momentum_coordinate is None else np.asarray(momentum_coordinate, float)
    if (not np.isfinite(constant) or l.shape != (dimension,) or s.shape != (dimension, dimension)
            or w.shape != s.shape or not np.all(np.isfinite(l))
            or not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)) or not np.allclose(s, s.T)):
        raise ValueError('Finite real Weyl polynomial coefficients required')
    terms = [(float(constant)*ia.toarray(), ib)]
    for i in range(dimension):
        if i < d:
            terms.append((l[i]*a['normal_coordinate'][i].toarray(), ib))
        else:
            terms.append((l[i]*ia.toarray(), b['normal_coordinate'][i-d]))
        for j in range(dimension):
            if i < d and j < d:
                qq = a['normal_coordinate_second'][i, j]
                pq = a['normal_momentum_coordinate'][i, j]
                terms.extend([(.5*s[i, j]*qq.toarray(), ib),
                              (.5*w[i, j]*(pq+pq.conj().T).toarray(), ib)])
            elif i >= d and j >= d:
                qq = b['normal_coordinate_second'][i-d, j-d]
                pq = b['normal_momentum_coordinate'][i-d, j-d]
                terms.extend([(.5*s[i, j]*ia.toarray(), qq),
                              (.5*w[i, j]*ia.toarray(), pq+pq.conj().T)])
            elif i < d:
                terms.extend([(.5*s[i, j]*a['normal_coordinate'][i].toarray(), b['normal_coordinate'][j-d]),
                              (w[i, j]*a['normal_momentum'][i].toarray(), b['normal_coordinate'][j-d])])
            else:
                terms.extend([(.5*s[i, j]*a['normal_coordinate'][j].toarray(), b['normal_coordinate'][i-d]),
                              (w[i, j]*a['normal_coordinate'][j].toarray(), b['normal_momentum'][i-d])])
    filtered = tuple((left, right) for left, right in terms
                     if np.max(abs(left)) > 1e-14 and right.nnz and np.max(abs(right.data)) > 1e-14)
    return _TensorSumOperator(filtered, na, nb).terms if filtered else ()


__all__ = ['covariance_adapted_frame', 'oscillator_annihilators',
           'coupled_number_hamiltonian', 'quadratic_coordinate_terms']
