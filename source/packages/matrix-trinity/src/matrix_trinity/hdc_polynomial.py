"""Prescribed HDC occupation energies and an isospectral nuclear frame.

The energy polynomial is an input, never fitted to auxiliary eigenvalues.
The auxiliary operator defines wavefunctions only. No exact equivalence to
the original molecular Hamiltonian at finite coupling is implied.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from math import comb, factorial, prod

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from .hdc_hamiltonian import (
    SAMHDCResult, _canonical_cubic_terms, _canonical_quartic_terms,
    _memory_guard, _mode_components,
    _operator_matrix, _shell_basis,
)


@dataclass(frozen=True)
class HDCPolynomialEnergy:
    """E0 + nu*n + Xii*n_i*(n_i-1) + Xij*n_i*n_j + Y falling cubic.

    Y is a symmetric third derivative: its unique-index coefficient is
    Y_ijk / product(multiplicity factorials). Each repeated occupation is
    a falling factorial. Thus Y does not alter zero/one/two-quanta anchors.
    """
    zero_point_cm: float
    fundamentals_cm: np.ndarray
    x_matrix_cm: np.ndarray
    y_tensor_cm: np.ndarray | None = None
    coordinate_id: str = ""

    def __post_init__(self):
        nu = np.array(self.fundamentals_cm, dtype=float, copy=True)
        x = np.array(self.x_matrix_cm, dtype=float, copy=True)
        n = len(nu) if nu.ndim == 1 else 0
        if (not n or x.shape != (n, n) or not np.isfinite(self.zero_point_cm)
                or not np.all(np.isfinite(nu)) or np.any(nu <= 0)
                or not np.all(np.isfinite(x)) or not np.allclose(x, x.T, rtol=0, atol=1e-10)):
            raise ValueError("finite HDC polynomial with positive fundamentals and symmetric X required")
        y = None if self.y_tensor_cm is None else np.array(self.y_tensor_cm, dtype=float, copy=True)
        if y is not None and (y.shape != (n, n, n) or not np.all(np.isfinite(y))
                or not np.allclose(y, y.swapaxes(0, 1), rtol=0, atol=1e-10)
                or not np.allclose(y, y.swapaxes(1, 2), rtol=0, atol=1e-10)):
            raise ValueError("Y must be a finite symmetric rank-three tensor")
        for name, value in (("fundamentals_cm", nu), ("x_matrix_cm", x), ("y_tensor_cm", y)):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    def energies(self, occupations):
        a = np.asarray(occupations)
        if (a.ndim != 2 or a.shape[1] != len(self.fundamentals_cm)
                or a.dtype.kind not in "iu" or np.any(a < 0)):
            raise ValueError("nonnegative integer occupation rows required")
        a = a.astype(float)
        x = self.x_matrix_cm
        e = (self.zero_point_cm + a @ self.fundamentals_cm
             + (a * (a - 1)) @ np.diag(x)
             + .5 * np.einsum("bi,ij,bj->b", a, x - np.diag(np.diag(x)), a))
        if self.y_tensor_cm is not None:
            for key in combinations_with_replacement(range(a.shape[1]), 3):
                c = self.y_tensor_cm[key]
                if c:
                    term = np.full(len(a), c)
                    for i in set(key):
                        m = key.count(i)
                        for j in range(m):
                            term *= a[:, i] - j
                        term /= factorial(m)
                    e += term
        if not np.all(np.isfinite(e)):
            raise FloatingPointError("occupation energy exceeds floating-point range")
        return e


def sam_hdc_polynomial_energy(field, *, alpha=1., beta_cm2=5e5):
    """Reuse scalar HDC pair corrections at fixed 0/1/2 occupation anchors.

    Virtual polynomial actions are complete before projection. X is a declared
    effective energy model, not a least-squares fit of normal-form eigenvalues.
    """
    from ._hdc_anchor_kernel import pack_anchor_operators, evaluate_anchor_energies, diagonal_quartic_terms
    from .hdc_hamiltonian import _controls
    _controls(alpha, beta_cm2, None)
    n = len(field.frequencies_cm)
    c = _canonical_cubic_terms(field.cubic_cm, field.metric_first_cm, 0.)
    q = diagonal_quartic_terms(field)
    zero = (0,) * n
    singles = [tuple(int(j == i) for j in range(n)) for i in range(n)]
    overtones = [tuple(2*int(j == i) for j in range(n)) for i in range(n)]
    pairs = [(i,j) for i in range(n) for j in range(i)]
    pair_states = [tuple(int(k in (i,j)) for k in range(n)) for i,j in pairs]
    states = np.asarray([zero,*singles,*overtones,*pair_states],np.int64)
    packed = pack_anchor_operators(c,q,field.frequencies_cm)
    e = evaluate_anchor_energies(states,field.frequencies_cm,field.constant_shift_cm,*packed,alpha,beta_cm2)
    if not np.all(np.isfinite(e)):
        raise FloatingPointError("nonfinite HDC anchor energies")
    e0 = e[0]
    one = e[1:1+n]
    nu = one-e0
    x = np.zeros((n,n))
    x[np.diag_indices(n)] = (e[1+n:1+2*n]-2*one+e0)/2
    for value,(i,j) in zip(e[1+2*n:],pairs,strict=True):
        x[i,j] = x[j,i] = value-one[i]-one[j]+e0
    return HDCPolynomialEnergy(e0, nu, x, coordinate_id=field.basis_id)


def transition_state_hdc_polynomial_energy(normal_form, reactive_action, *, alpha=1., beta_cm2=5e5):
    """Scalar HDC X on the stable bath of a saddle QNF at fixed real action.

    Eliminated cubic self-energies are already in Z4. Only retained Z3 is
    regularized here; the unstable coordinate is never an occupation.
    """
    from matrix_vpt2_vci.hdcpt2 import _hybrid_cubic_term
    from .transition_state import TransitionStateNormalForm, _weyl_ladder, _real
    from .hdc_hamiltonian import _controls
    _controls(alpha, beta_cm2, None)
    if not isinstance(normal_form, TransitionStateNormalForm):
        raise TypeError("TransitionStateNormalForm required")
    if normal_form.normal_cubic is None or normal_form.normal_quartic is None:
        raise ValueError("separate Z3 and Z4 required to avoid duplicated self-energies")
    action = _real(reactive_action, "reactive_action")
    field = normal_form.field
    reactive = field.reaction_mode
    stable = [i for i in range(len(field.signed_frequencies_cm)) if i != reactive]
    frequencies = field.signed_frequencies_cm[stable]
    if not len(stable):
        raise ValueError("at least one stable transverse mode required for occupation sampling")

    def apply(poly, state):
        result = {}
        for key, coefficient in poly.items():
            a, b = key[2*reactive:2*reactive+2]
            if a != b or a > 2:
                raise ValueError("QNF must conserve reactive action through degree four")
            value = coefficient * (1. if a == 0 else action if a == 1 else action**2-.25)
            target = list(state)
            for j, mode in enumerate(stable):
                target[j], factor = _weyl_ladder(key[2*mode], key[2*mode+1], state[j])
                value *= factor
            if value:
                target = tuple(target)
                result[target] = result.get(target, 0.) + value
        return result

    def energy(state):
        value = (field.barrier_energy_cm + abs(field.signed_frequencies_cm[reactive])*action
                 + (np.asarray(state)+.5) @ frequencies
                 + apply(normal_form.normal_quartic, state).get(state, 0.))
        for other, coupling in apply(normal_form.normal_cubic, state).items():
            if other == state:
                value += coupling
            else:
                gap = (np.asarray(state)-other) @ frequencies
                value += _hybrid_cubic_term(abs(coupling), float(gap), alpha, beta_cm2)
        if abs(np.imag(value)) > 1e-8:
            raise ValueError("nonreal fixed-action occupation energy")
        return float(np.real(value))

    n = len(stable)
    e0 = energy((0,)*n)
    singles = [energy(tuple(int(k == i) for k in range(n))) for i in range(n)]
    x = np.zeros((n,n))
    for i in range(n):
        x[i,i] = (energy(tuple(2*int(k == i) for k in range(n))) - 2*singles[i]+e0)/2
        for j in range(i):
            x[i,j] = x[j,i] = energy(tuple(int(k in (i,j)) for k in range(n)))-singles[i]-singles[j]+e0
    return HDCPolynomialEnergy(e0, np.asarray(singles)-e0, x,
                               coordinate_id=f"fixed-reactive-action:{action:.17g}")


def hdc_isospectral_frame(harmonic_cm, cubic_cm, quartic_cm, prescribed_cm, *,
                         reference_scale_cm, workers=1, maximum_states=4096,
                         maximum_dense_bytes=512 * 1024**2):
    """Continuous bounded auxiliary frame, with exactly prescribed energies.

    D=diag(Epol-min(Epol)+scale)>0, B=D^-1/2 Voff D^-1/2,
    A=D^1/2 [I+B/sqrt(I+B^2)] D^1/2.
    Sorted eigenvectors of A define the frame; its eigenvalues are discarded.
    The analytic matrix function makes no exact/near degeneracy partition.
    At fixed nonzero gaps its correction to V is O(V^3), preserving second
    order states when Epol has the VPT2 second-order diagonal. Continuity of
    individual projectors requires an auxiliary gap; unresolved degeneracies
    carry only a subspace meaning. Continuity of input energy coefficients is
    an independent requirement, not inferred from this frame.
    """
    from .hdc_hamiltonian import HDCMatrixResult, _hermitian
    from .vibronic_kernels import worker_count
    e = np.asarray(prescribed_cm, float)
    h0 = np.asarray(harmonic_cm, float)
    n = len(e)
    _memory_guard(n, maximum_states, maximum_dense_bytes, copies=18)
    if (e.shape != (n,) or h0.shape != (n,) or not np.all(np.isfinite(e))
            or not np.all(np.isfinite(h0)) or not np.isfinite(reference_scale_cm)
            or reference_scale_cm <= 0):
        raise ValueError("finite energy vectors and positive auxiliary reference scale required")
    v = _hermitian(cubic_cm, n, "cubic_cm") + _hermitian(quartic_cm, n, "quartic_cm")
    np.fill_diagonal(v, 0.)
    d = e - np.min(e) + reference_scale_cm
    root = np.sqrt(d)
    workers = worker_count(workers)
    with threadpool_limits(limits=workers, user_api="blas"):
        b = v / root[:, None] / root[None, :]
        values, z = eigh(b, check_finite=False, driver="evr")
        bounded = (z * (values / np.hypot(1., values))) @ z.conj().T
        auxiliary = bounded * root[:, None] * root[None, :]
        auxiliary[np.diag_indices(n)] += d
        aux_values, vectors = eigh(auxiliary, check_finite=False, driver="evr")
        energies = np.sort(e, kind="stable")
        h = (vectors * energies) @ vectors.conj().T
    return HDCMatrixResult(h, energies, vectors, dict(
        schema="matrix.trinity.hdc_isospectral_frame.v1", energies_prescribed=True,
        auxiliary_eigenvalues_discarded=True, quartic_globally_diagonalized=False,
        exact_degeneracy_partition_used=False,
        auxiliary_reference_scale_cm=float(reference_scale_cm),
        minimum_auxiliary_gap_cm=float(np.min(np.diff(aux_values))) if n > 1 else None,
        orthogonality_error=float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(n))),
        continuity_scope="continuous input coefficients and isolated auxiliary spectral projectors",
        energy_preservation_error_cm=float(np.max(abs(energies - np.sort(e)))),
    ))


def solve_sam_hdc_polynomial(field, energy_model, *, maximum_total_quanta=7,
                             maximum_states=4096, maximum_dense_bytes=512*1024**2,
                             workers=1):
    """Separable SAM states with the supplied X/Y polynomial spectrum unchanged."""
    from matrix_sentinel.quantum_wang_landau import _certified_polynomial_cutoff
    if not isinstance(energy_model, HDCPolynomialEnergy):
        raise TypeError("HDCPolynomialEnergy required")
    n = len(field.frequencies_cm)
    if len(energy_model.fundamentals_cm) != n or energy_model.coordinate_id != field.basis_id:
        raise ValueError("energy polynomial and mechanical field need the same modal coordinate identity")
    if isinstance(maximum_total_quanta, bool) or not isinstance(maximum_total_quanta, int) or maximum_total_quanta < 1:
        raise ValueError("positive integer maximum_total_quanta required")
    cutoff, _ = _certified_polynomial_cutoff(energy_model.fundamentals_cm,
        energy_model.x_matrix_cm, energy_model.y_tensor_cm, maximum_total_quanta)
    if cutoff != maximum_total_quanta:
        raise ValueError(f"polynomial support exceeds monotone domain; maximum permitted shell is {cutoff}")
    cubic = _canonical_cubic_terms(field.cubic_cm, field.metric_first_cm, 0.)
    quartic = _canonical_quartic_terms(field.quartic_cm, field.metric_second_cm)
    graph_terms = list(cubic + quartic)
    for i in range(n):
        for j in range(i):
            if energy_model.x_matrix_cm[i, j]:
                graph_terms.append((((i, ()), (j, ())), 1.))
    if energy_model.y_tensor_cm is not None:
        for key in combinations_with_replacement(range(n), 3):
            if energy_model.y_tensor_cm[key]:
                graph_terms.append((tuple((i, ()) for i in set(key)), 1.))
    components = _mode_components(graph_terms, n)
    size = prod(comb(len(m)+maximum_total_quanta, maximum_total_quanta) for m in components)
    _memory_guard(size, maximum_states, maximum_dense_bytes, copies=18)
    bases = [_shell_basis(len(m), maximum_total_quanta) for m in components]
    energies = np.zeros(1)
    vectors = np.ones((1, 1), complex)
    diagnostics = []
    for modes, basis in zip(components, bases, strict=True):
        index = {m: i for i, m in enumerate(modes)}
        def restrict(terms):
            return tuple((tuple((index[m], ops) for m, ops in key), value)
                         for key, value in terms if all(m in index for m, _ in key))
        y = energy_model.y_tensor_cm
        local = HDCPolynomialEnergy(0., energy_model.fundamentals_cm[list(modes)],
            energy_model.x_matrix_cm[np.ix_(modes, modes)],
            None if y is None else y[np.ix_(modes, modes, modes)])
        r = hdc_isospectral_frame((np.asarray(basis)+.5) @ field.frequencies_cm[list(modes)],
            _operator_matrix(basis, restrict(cubic)), _operator_matrix(basis, restrict(quartic)),
            local.energies(np.asarray(basis)), reference_scale_cm=float(np.min(field.frequencies_cm[list(modes)])),
            workers=workers, maximum_states=maximum_states, maximum_dense_bytes=maximum_dense_bytes)
        energies = (energies[:, None] + r.energies_cm).ravel()
        vectors = np.kron(vectors, r.eigenvectors)
        diagnostics.append(r.diagnostics)
    states = []
    for selection in product(*bases):
        state = np.zeros(n, int)
        for modes, values in zip(components, selection, strict=True):
            state[list(modes)] = values
        states.append(state)
    order = np.argsort(energies, kind="stable")
    basis = np.asarray(states)
    boundary = np.zeros(len(basis), bool)
    for modes in components:
        boundary |= basis[:, list(modes)].sum(axis=1) == maximum_total_quanta
    return SAMHDCResult(basis, energies[order] + energy_model.zero_point_cm, vectors[:, order],
        components, dict(formulation="prescribed_hdc_polynomial", energies_prescribed=True,
            energy_degree=3 if energy_model.y_tensor_cm is not None else 2,
            component_modes=components, component_diagnostics=diagnostics,
            tensor_product_basis=True, maximum_total_quanta_per_component=maximum_total_quanta,
            boundary_weights=np.sum(abs(vectors[boundary][:, order])**2, axis=0),
            infinite_basis_convergence_certified=False), field.frequencies_cm, field.basis_id)
