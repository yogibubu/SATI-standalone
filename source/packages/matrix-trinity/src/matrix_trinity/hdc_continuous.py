"""Continuous operator completion on a fixed vibrational model space.

This is an operator-level extension, distinct from scalar pair-additive
HDCPT2. A smooth contact transformation retains every residual coupling.
There is no resonance list, gap snapping, or harmonic-degeneracy partition.
"""
from math import comb
import numpy as np
from scipy.linalg import eigh, expm
from threadpoolctl import threadpool_limits

from .hdc_hamiltonian import (HDCMatrixResult, SAMHDCResult, _hermitian,
                              _memory_guard, _operator_matrix, _shell_basis)
from .curvilinear_vpt2 import (_canonical_cubic_terms, _canonical_quartic_terms,
                               _compiled_operator_couplings)
from .hdc_polynomial import HDCPolynomialEnergy
from .vibronic_kernels import worker_count


def continuous_gap_filter(gap_cm, regularization_width_cm):
    """Entire odd filter (1-exp[-(d/sigma)^2])/d, with its value at zero."""
    if not np.isfinite(regularization_width_cm) or regularization_width_cm <= 0:
        raise ValueError("positive finite regularization_width_cm required")
    d = np.asarray(gap_cm, float)
    if not np.all(np.isfinite(d)):
        raise ValueError("finite harmonic gaps required")
    with np.errstate(over='ignore', under='ignore'):
        x = (d / regularization_width_cm)**2
        retained = np.exp(-x)
        inverse = np.divide(-np.expm1(-x), d, out=np.zeros_like(d), where=d != 0)
    return inverse, retained


def complete_continuous_hdc_normal_form(harmonic_energies_cm, cubic_cm, quartic_cm,
        *, regularization_width_cm=100., maximum_states=4096,
        maximum_dense_bytes=512*1024**2, workers=0, reconstruct_hamiltonian=True):
    """Joint continuous energies/functions/operators through mechanical order two.

    S=-f(ad_H0)V3, R=V3+[H0,S],
    K=H0+R+V4+[V3+R,S]/2, U=exp(S), H_eff=U K U^dagger.
    All residual interactions R remain in K and are diagonalized together.
    The positive smoothing width is fixed along the whole continuation and
    never classifies or discards an interaction.
    """
    e = np.asarray(harmonic_energies_cm, float)
    if e.ndim != 1 or not len(e) or not np.all(np.isfinite(e)):
        raise ValueError("finite nonempty harmonic energy vector required")
    n = len(e)
    _memory_guard(n, maximum_states, maximum_dense_bytes, copies=18)
    v = _hermitian(cubic_cm, n, 'cubic_cm')
    q = _hermitian(quartic_cm, n, 'quartic_cm')
    gaps = e[:, None]-e[None, :]
    inverse, retained = continuous_gap_filter(gaps, regularization_width_cm)
    s = -inverse*v
    residual = retained*v
    combined = v+residual
    k = np.diag(e)+residual+q+(combined@s-s@combined)/2
    k = (k+k.conj().T)/2
    workers = worker_count(workers)
    with threadpool_limits(limits=workers, user_api='blas'):
        levels, rotation = eigh(k.real if not np.any(k.imag) else k,
                                check_finite=False, driver='evr')
        transform = expm(s.real if not np.any(s.imag) else s)
        vectors = transform@rotation
        h = (vectors*levels)@vectors.conj().T if reconstruct_hamiltonian else None
    return HDCMatrixResult(h, levels, vectors, dict(
        schema='matrix.trinity.continuous_hdc_normal_form.v1',
        formulation='continuous_normal_form',
        conventional_scalar_hdcpt2_equivalence=False,
        resonance_selection=False, harmonic_degeneracy_partition=False,
        regularization_width_cm=float(regularization_width_cm), workers=workers,
        orthogonality_error=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(n))),
        retained_cubic_frobenius_cm=float(np.linalg.norm(residual)),
        fixed_model_space_size=n, effective_diagonal_cm=np.diag(k).real,
    ))


def sam_continuous_hdc_polynomial_energy(field, *, regularization_width_cm=100.):
    """Quadratic diagonal of the same continuous contact Hamiltonian.

    Virtual cubic actions precede projection. The commutator of cubic ladder
    polynomials has degree at most four, so its number-diagonal is quadratic.
    Residual off-diagonal interactions belong to the joint spectral solver.
    """
    n = len(field.frequencies_cm)
    cubic = _canonical_cubic_terms(field.cubic_cm, field.metric_first_cm, 0.)
    quartic = _canonical_quartic_terms(field.quartic_cm, field.metric_second_cm)
    zero = (0,)*n
    singles = [tuple(int(j == i) for j in range(n)) for i in range(n)]
    doubles = [tuple(2*int(j == i) for j in range(n)) for i in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i)]
    anchors = [zero, *singles, *doubles,
               *(tuple(int(k in (i,j)) for k in range(n)) for i,j in pairs)]
    def evaluate(state):
        result = (np.asarray(state)+.5)@field.frequencies_cm+field.constant_shift_cm
        result += _compiled_operator_couplings(state, quartic).get(state, 0.).real
        for target, amplitude in _compiled_operator_couplings(state, cubic).items():
            d = float((np.asarray(state)-target)@field.frequencies_cm)
            f, r = continuous_gap_filter(d, regularization_width_cm)
            result += float(f*(1+r))*abs(amplitude)**2
        return float(result)
    values = np.array([evaluate(state) for state in anchors])
    e0 = values[0]; one = values[1:1+n];x = np.zeros((n,n))
    x[np.diag_indices(n)] = (values[1+n:1+2*n]-2*one+e0)/2
    for value, (i,j) in zip(values[1+2*n:], pairs, strict=True):
        x[i,j] = x[j,i] = value-one[i]-one[j]+e0
    return HDCPolynomialEnergy(e0, one-e0, x, coordinate_id=field.basis_id)


def solve_sam_continuous_hdcpt2(field, *, maximum_total_quanta=8,
        maximum_states=4096, maximum_dense_bytes=512*1024**2,
        regularization_width_cm=100., mechanical_scale=1., workers=0):
    """Fixed global shell: model-space membership is independent of couplings."""
    if (isinstance(maximum_total_quanta, bool)
            or not isinstance(maximum_total_quanta, int) or maximum_total_quanta < 0):
        raise ValueError("nonnegative integer maximum_total_quanta required")
    if not np.isfinite(mechanical_scale) or mechanical_scale < 0:
        raise ValueError("finite nonnegative mechanical_scale required")
    n = len(field.frequencies_cm)
    _memory_guard(comb(n+maximum_total_quanta,n), maximum_states, maximum_dense_bytes, copies=18)
    basis = np.asarray(_shell_basis(n,maximum_total_quanta))
    cubic = _canonical_cubic_terms(field.cubic_cm,field.metric_first_cm,0.)
    quartic = _canonical_quartic_terms(field.quartic_cm,field.metric_second_cm)
    b = tuple(map(tuple,basis))
    r = complete_continuous_hdc_normal_form((basis+.5)@field.frequencies_cm,
        mechanical_scale*_operator_matrix(b,cubic),
        mechanical_scale**2*_operator_matrix(b,quartic),
        regularization_width_cm=regularization_width_cm,
        maximum_states=maximum_states, maximum_dense_bytes=maximum_dense_bytes,
        workers=workers, reconstruct_hamiltonian=False)
    boundary = basis.sum(axis=1) == maximum_total_quanta
    diagnostics = dict(r.diagnostics,
        boundary_weights=np.sum(abs(r.eigenvectors[boundary])**2,axis=0),
        polynomial_diagonal_role='joint Hamiltonian diagonal; residual couplings retained')
    return SAMHDCResult(basis, r.energies_cm+mechanical_scale**2*field.constant_shift_cm,
        r.eigenvectors,(tuple(range(n)),),diagnostics,field.frequencies_cm,field.basis_id)
