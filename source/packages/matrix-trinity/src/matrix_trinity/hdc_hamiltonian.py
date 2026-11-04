"""Hermitian completion of regularized HDC-form pair energies.

This is an explicitly distinct model from pair-additive HDCPT2. The matrix
completion acts on ONE interacting component; the modal solver separates
exactly noninteracting components before applying the nonlinear map.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from math import comb, prod
import hashlib

import numpy as np
from scipy.linalg import eigh, svd, expm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

from .curvilinear_vpt2 import (
    _canonical_cubic_terms,
    _canonical_quartic_terms,
    _compiled_operator_couplings,
    _canonical_operator_key,
)
from .vibronic_kernels import worker_count
from .vibronic_vpt2 import SAMVPT2Field, _positive_integer


@dataclass(frozen=True)
class HDCMatrixResult:
    hamiltonian_cm: np.ndarray | None
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class SAMHDCResult:
    basis: np.ndarray
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    components: tuple
    diagnostics: dict
    frequencies_cm: np.ndarray
    basis_id: str = ""


def _controls(alpha, beta_cm2, mixing_weight):
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(beta_cm2) or beta_cm2 < 0:
        raise ValueError("HDC alpha must be positive and beta_cm2 nonnegative, both finite")
    if mixing_weight is not None and (
        not np.isfinite(mixing_weight) or not 0 <= mixing_weight <= 1
    ):
        raise ValueError("mixing_weight must be None or a finite number in [0,1]")


def hdc_coupling_moduli(gap_cm, coupling_cm, *, alpha=1.0, beta_cm2=5.0e5, mixing_weight=None):
    """Stable vectorized g(d,u), preserving phase separately; exact g(0,u)=u."""
    _controls(alpha, beta_cm2, mixing_weight)
    d, u = np.broadcast_arrays(np.asarray(gap_cm, float), np.asarray(coupling_cm, float))
    if np.any(d < 0) or np.any(u < 0) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(u)):
        raise ValueError("finite nonnegative gaps and coupling moduli required")
    result = u.copy()
    active = (d > 0) & (u > 0)
    if np.any(active):
        da, ua = d[active], u[active]
        # Entire computation is dimensionless; no u**2/d pole is evaluated.
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            x = da / ua
            weight = (
                (1 + np.tanh(alpha * (ua * (da / 2) - beta_cm2))) / 2
                if mixing_weight is None
                else np.full_like(ua, mixing_weight)
            )
        usable = np.isfinite(x) & (x > 0)
        factor = np.ones_like(ua)
        if np.any(usable):
            xx = x[usable]
            radius = np.hypot(xx, 2.0)
            b_over_u = 1 / (radius / 2 + xx / 2)
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                y = 2 / xx
                envelope = np.exp(-y * y)
            safe = envelope > 0
            t_over_u = np.zeros_like(xx)
            t_over_u[safe] = weight[usable][safe] * envelope[safe] * b_over_u[safe] ** 2 / xx[safe]
            factor[usable] = np.sqrt(1 + radius * t_over_u + t_over_u * t_over_u)
        result[active] = ua * factor
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("completed coupling is outside the floating-point range")
    return result


def _memory_guard(size, maximum_states, maximum_dense_bytes, copies=12):
    _positive_integer(maximum_states, "maximum_states")
    _positive_integer(maximum_dense_bytes, "maximum_dense_bytes")
    if size > maximum_states:
        raise MemoryError(f"HDC basis of {size} states exceeds maximum_states={maximum_states}")
    if copies * 16 * size * size > maximum_dense_bytes:
        raise MemoryError("HDC conservative dense working-set budget exceeded before allocation")


def _hermitian(value, n, name):
    a = np.array(value, complex, copy=True)
    if a.shape != (n, n) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite square matrix in the declared basis")
    scale = max(1.0, float(np.max(abs(a))))
    if np.max(abs(a - a.conj().T)) > 1.0e-11 * scale:
        raise ValueError(f"{name} must be Hermitian")
    return (a + a.conj().T) / 2


def _diagonalize(h, workers):
    """Exact zero connectivity blocks; compiled LAPACK with bounded BLAS threads."""
    n = len(h)
    graph = h != 0
    np.fill_diagonal(graph, False)
    count, labels = connected_components(csr_matrix(graph), directed=False)
    energies = np.empty(n)
    vectors = np.zeros((n, n), complex)
    cursor = 0
    largest = 0
    with threadpool_limits(limits=workers, user_api="blas"):
        for block in range(count):
            ids = np.flatnonzero(labels == block)
            largest = max(largest, len(ids))
            if len(ids) == 1:
                roots = np.array([h[ids[0], ids[0]].real])
                u = np.ones((1, 1))
            else:
                part = h[np.ix_(ids, ids)]
                if not np.any(part.imag):
                    part = part.real
                roots, u = eigh(part, check_finite=False, driver="evr")
            energies[cursor : cursor + len(ids)] = roots
            vectors[np.ix_(ids, np.arange(cursor, cursor + len(ids)))] = u
            cursor += len(ids)
    order = np.argsort(energies, kind="stable")
    energies = energies[order]
    vectors = vectors[:, order]
    pivots = vectors[np.argmax(abs(vectors), axis=0), np.arange(n)]
    vectors *= np.conj(pivots / abs(pivots))[None, :]
    return (
        energies,
        vectors,
        dict(diagonalization_blocks=count, largest_diagonalization_block=largest),
    )


def complete_hdc_hamiltonian(
    harmonic_energies_cm,
    cubic_cm,
    quartic_cm,
    *,
    alpha=1.0,
    beta_cm2=5.0e5,
    mixing_weight=None,
    maximum_states=4096,
    maximum_dense_bytes=512 * 1024**2,
    workers=0,
):
    """Complete one interacting finite component by exact spectral-projector blocks.

    Exact harmonic degeneracies use an SVD map, not elementwise rescaling.
    No near-degenerate energies are snapped or discarded. The exponential
    envelope makes the map finite at every gap. Input V4 is the bare quartic
    operator, WITHOUT cubic shifts or an effective X diagonal.
    """
    _controls(alpha, beta_cm2, mixing_weight)
    workers = worker_count(workers)
    e = np.asarray(harmonic_energies_cm, float)
    if e.ndim != 1 or not len(e) or not np.all(np.isfinite(e)):
        raise ValueError("finite nonempty harmonic energy vector required")
    n = len(e)
    _memory_guard(n, maximum_states, maximum_dense_bytes)
    v = _hermitian(cubic_cm, n, "cubic_cm")
    w = _hermitian(quartic_cm, n, "quartic_cm")
    groups = [np.flatnonzero(e == level) for level in np.unique(e)]
    completed = np.diag(e).astype(complex) + w
    scalar = np.array([a[0] for a in groups if len(a) == 1], int)
    if len(scalar):
        a = v[np.ix_(scalar, scalar)]
        d = abs(e[scalar, None] - e[None, scalar])
        u = abs(a)
        mapped = hdc_coupling_moduli(
            d, u, alpha=alpha, beta_cm2=beta_cm2, mixing_weight=mixing_weight
        )
        ratio = np.divide(mapped, u, out=np.ones_like(u), where=u > 0)
        completed[np.ix_(scalar, scalar)] += a * ratio
    degenerate_svd_count = 0
    with threadpool_limits(limits=workers, user_api="blas"):
        for a, ia in enumerate(groups):
            if len(ia) > 1:
                completed[np.ix_(ia, ia)] += v[np.ix_(ia, ia)]
            for ib in groups[:a]:
                if len(ia) == len(ib) == 1:
                    continue
                block = v[np.ix_(ia, ib)]
                if not np.any(block):
                    continue
                left, u, right = svd(block, full_matrices=False, check_finite=False)
                mapped = hdc_coupling_moduli(
                    abs(e[ia[0]] - e[ib[0]]),
                    u,
                    alpha=alpha,
                    beta_cm2=beta_cm2,
                    mixing_weight=mixing_weight,
                )
                block = (left * mapped) @ right
                completed[np.ix_(ia, ib)] += block
                completed[np.ix_(ib, ia)] += block.conj().T
                degenerate_svd_count += 1
    roots, u, diag = _diagonalize(completed, workers)
    change = completed - (np.diag(e) + v + w)
    return HDCMatrixResult(
        completed,
        roots,
        u,
        dict(
            diag,
            schema="matrix.trinity.hdc_completion.v1",
            model="regularized_hdc_pair_hamiltonian",
            conventional_hdcpt2_equivalence=False,
            exponential_envelope=True,
            harmonic_energy_blocks=len(groups),
            degenerate_svd_blocks=degenerate_svd_count,
            maximum_harmonic_degeneracy=max(map(len, groups)),
            completion_change_frobenius_cm=float(np.linalg.norm(change)),
            workers=workers,
            alpha=alpha,
            beta_cm2=beta_cm2,
            mixing_weight=mixing_weight,
            residual_norm_cm=float(np.linalg.norm(completed @ u - u * roots)),
            orthogonality_error=float(np.linalg.norm(u.conj().T @ u - np.eye(n))),
            model_space_complete=True,
            infinite_basis_convergence_certified=False,
        ),
    )


def _normal_form_stage(energies, groups, coupling, *, alpha, beta_cm2, mixing_weight):
    """Covariant bounded angular generator and its HDC pair self-energy blocks."""
    n = len(energies)
    generator = np.zeros((n, n), complex)
    shift = np.zeros((n, n), complex)
    scalar = np.array([ids[0] for ids in groups if len(ids) == 1], int)
    if len(scalar):
        a = coupling[np.ix_(scalar, scalar)]
        signed = energies[scalar, None] - energies[None, scalar]
        d = abs(signed)
        u = abs(a)
        g = hdc_coupling_moduli(d, u, alpha=alpha, beta_cm2=beta_cm2, mixing_weight=mixing_weight)
        radius = np.hypot(d, 2 * g)
        delta = np.divide(2 * g, d + radius, out=np.zeros_like(g), where=d + radius > 0) * g
        angle = np.arctan2(g, d + delta)
        phase = np.divide(a, u, out=np.zeros_like(a), where=u > 0)
        generator[np.ix_(scalar, scalar)] = -np.sign(signed) * phase * angle
        shift[scalar, scalar] = np.sum(np.sign(signed) * delta, axis=1)
    for i, ia in enumerate(groups):
        for ib in groups[:i]:
            if len(ia) == len(ib) == 1:
                continue
            a = coupling[np.ix_(ia, ib)]
            if not np.any(a):
                continue
            left, u, right = svd(a, full_matrices=False, check_finite=False)
            d = float(energies[ia[0]] - energies[ib[0]])
            g = hdc_coupling_moduli(
                d, u, alpha=alpha, beta_cm2=beta_cm2, mixing_weight=mixing_weight
            )
            delta = g * (2 * g / (d + np.hypot(d, 2 * g)))
            angle = np.arctan2(g, d + delta)
            block = -(left * angle) @ right
            generator[np.ix_(ia, ib)] = block
            generator[np.ix_(ib, ia)] = -block.conj().T
            shift[np.ix_(ia, ia)] += (left * delta) @ left.conj().T
            shift[np.ix_(ib, ib)] -= (right.conj().T * delta) @ right
    return generator, shift


def complete_hdc_normal_form(
    harmonic_energies_cm,
    cubic_cm,
    quartic_cm,
    *,
    alpha=1.0,
    beta_cm2=5.0e5,
    mixing_weight=None,
    maximum_states=4096,
    maximum_dense_bytes=512 * 1024**2,
    workers=0,
    reconstruct_hamiltonian=True,
):
    """Order-consistent unitary completion; V4 is NEVER globally diagonalized.

    U=exp(S1) exp(S2); S1 is the bounded HDC angular generator of V3.
    S2 uses W=V4+[V3_same,S1]+[V3_cross,S1]/2. HDC self-energy
    blocks define the normal-form energies. Exact harmonic manifolds are
    diagonalized collectively. Quartic/multicubic cross-manifold couplings
    enter the second generator and its higher-order pair self-energies, not
    a variational extrapolation of an unbounded local quartic potential.
    """
    _controls(alpha, beta_cm2, mixing_weight)
    workers = worker_count(workers)
    e = np.asarray(harmonic_energies_cm, float)
    if e.ndim != 1 or not len(e) or not np.all(np.isfinite(e)):
        raise ValueError("finite nonempty harmonic energy vector required")
    n = len(e)
    _memory_guard(n, maximum_states, maximum_dense_bytes, copies=16)
    v = _hermitian(cubic_cm, n, "cubic_cm")
    q = _hermitian(quartic_cm, n, "quartic_cm")
    groups = [np.flatnonzero(e == value) for value in np.unique(e)]
    controls = dict(alpha=alpha, beta_cm2=beta_cm2, mixing_weight=mixing_weight)
    with threadpool_limits(limits=workers, user_api="blas"):
        s1, shift1 = _normal_form_stage(e, groups, v, **controls)
        same = np.zeros_like(v)
        normal = np.diag(e).astype(complex) + shift1
        for ids in groups:
            same[np.ix_(ids, ids)] = v[np.ix_(ids, ids)]
            normal[np.ix_(ids, ids)] += v[np.ix_(ids, ids)] + q[np.ix_(ids, ids)]
        combined = v + same
        w = _hermitian(q + (combined @ s1 - s1 @ combined) / 2, n, "effective_order_two_operator")
        s2, shift2 = _normal_form_stage(e, groups, w, **controls)
        normal += shift2
        roots, rotation, diag = _diagonalize(normal, workers)
        # Real SAM fields retain the faster real compiled matrix exponential.
        a = s1.real if not np.any(s1.imag) else s1
        b = s2.real if not np.any(s2.imag) else s2
        vectors = expm(a) @ (expm(b) @ rotation)
        orth = float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(n)))
        h = (vectors * roots) @ vectors.conj().T if reconstruct_hamiltonian else None
    return HDCMatrixResult(
        h,
        roots,
        vectors,
        dict(
            diag,
            schema="matrix.trinity.hdc_normal_form.v1",
            model="regularized_hdc_unitary_normal_form",
            formulation="normal_form",
            conventional_hdcpt2_equivalence=False,
            quartic_globally_diagonalized=False,
            alpha=alpha,
            beta_cm2=beta_cm2,
            mixing_weight=mixing_weight,
            harmonic_energy_blocks=len(groups),
            maximum_harmonic_degeneracy=max(map(len, groups)),
            first_generator_frobenius=float(np.linalg.norm(s1)),
            second_generator_frobenius=float(np.linalg.norm(s2)),
            orthogonality_error=orth,
            maximum_column_norm_error=float(np.max(abs(np.sum(abs(vectors) ** 2, axis=0) - 1))),
            completion_change_frobenius_cm=None,
            self_energy_quartic_resummation=True,
            workers=workers,
            infinite_basis_convergence_certified=False,
        ),
    )


def _mode_components(terms, n):
    graph = np.eye(n, dtype=bool)
    for key, coefficient in terms:
        modes = [m for m, _ in key]
        if coefficient and modes:
            graph[np.ix_(modes, modes)] = True
    count, labels = connected_components(csr_matrix(graph), directed=False)
    return tuple(tuple(np.flatnonzero(labels == i).tolist()) for i in range(count))


def _shell_basis(n, cutoff):
    states = [(0,) * n]
    for total in range(1, cutoff + 1):
        for occupied in combinations_with_replacement(range(n), total):
            counts = np.bincount(occupied, minlength=n)
            states.append(tuple(counts.tolist()))
    return tuple(states)


def _operator_matrix(basis, terms):
    lookup = {state: i for i, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), complex)
    for source, state in enumerate(basis):
        for target, value in _compiled_operator_couplings(state, terms).items():
            row = lookup.get(target)
            if row is not None:
                matrix[row, source] = value
    return matrix


def solve_sam_hdcpt2(
    field,
    *,
    maximum_total_quanta=8,
    maximum_states=4096,
    maximum_dense_bytes=512 * 1024**2,
    mechanical_scale=1.0,
    alpha=1.0,
    beta_cm2=5.0e5,
    mixing_weight=None,
    allow_unbounded_quartic=False,
    formulation="normal_form",
    regularization_width_cm=100.0,
    workers=0,
):
    """Order-consistent native F3/F4/G1/G2 completion with automatic mode factors.

    Signed quartics are legitimate perturbative data in the default normal
    form. Only the explicit quartic_variational comparison globally
    diagonalizes the local polynomial and requires unbounded-potential opt-in.

    Each connected component gets complete excitation shells up to the cutoff;
    the TOTAL space is their tensor product. This is deliberately not a global
    excitation shell when components are independent. Budgets are checked before
    enumerating or allocating that product. Eigenlevel IDs are energy ordered,
    not presumed harmonic occupation labels.
    """
    if not isinstance(field, SAMVPT2Field):
        raise TypeError("SAMVPT2Field required")
    if formulation == "continuous_normal_form":
        from .hdc_continuous import solve_sam_continuous_hdcpt2
        return solve_sam_continuous_hdcpt2(field,
            maximum_total_quanta=maximum_total_quanta, maximum_states=maximum_states,
            maximum_dense_bytes=maximum_dense_bytes, mechanical_scale=mechanical_scale,
            regularization_width_cm=regularization_width_cm, workers=workers)
    if (
        isinstance(maximum_total_quanta, bool)
        or not isinstance(maximum_total_quanta, int)
        or maximum_total_quanta < 0
    ):
        raise ValueError("maximum_total_quanta must be a nonnegative integer")
    if not np.isfinite(mechanical_scale) or mechanical_scale < 0:
        raise ValueError("mechanical_scale must be finite and nonnegative")
    _controls(alpha, beta_cm2, mixing_weight)
    workers = worker_count(workers)
    if not isinstance(allow_unbounded_quartic, bool):
        raise ValueError("allow_unbounded_quartic must be boolean")
    unbounded_modes = [
        i
        for i in range(len(field.frequencies_cm))
        if mechanical_scale > 0
        and (
            field.quartic_cm[(i,) * 4] < 0
            or (field.quartic_cm[(i,) * 4] == 0 and field.cubic_cm[(i,) * 3] != 0)
        )
    ]
    if formulation not in {"normal_form", "quartic_variational"}:
        raise ValueError("formulation must be normal_form or quartic_variational")
    if unbounded_modes and formulation == "quartic_variational" and not allow_unbounded_quartic:
        raise ValueError(
            "bare local quartic potential is provably unbounded along modal axes; "
            "supply a confining potential or explicitly allow_unbounded_quartic for finite-local-model diagnostics"
        )
    cubic = _canonical_cubic_terms(field.cubic_cm, field.metric_first_cm, 0.0)
    quartic = _canonical_quartic_terms(field.quartic_cm, field.metric_second_cm)
    identity = hashlib.sha256(
        b"".join(
            np.asarray(getattr(field, name), dtype="<f8").tobytes()
            for name in (
                "frequencies_cm",
                "cubic_cm",
                "quartic_cm",
                "metric_first_cm",
                "metric_second_cm",
                "constant_shift_cm",
            )
        )
    ).hexdigest()
    components = _mode_components(cubic + quartic, len(field.frequencies_cm))
    sizes = [comb(len(modes) + maximum_total_quanta, len(modes)) for modes in components]
    _memory_guard(prod(sizes), maximum_states, maximum_dense_bytes)
    bases = []
    matrices = []
    component_diagnostics = []
    for modes in components:
        local = {m: i for i, m in enumerate(modes)}

        def restrict(terms):
            return tuple(
                (tuple((local[m], ops) for m, ops in key), value)
                for key, value in terms
                if all(m in local for m, _ in key)
            )

        basis = _shell_basis(len(modes), maximum_total_quanta)
        harmonic = (np.asarray(basis) + 0.5) @ field.frequencies_cm[list(modes)]
        solver = (
            complete_hdc_normal_form if formulation == "normal_form" else complete_hdc_hamiltonian
        )
        extra = {"reconstruct_hamiltonian": False} if formulation == "normal_form" else {}
        result = solver(
            harmonic,
            mechanical_scale * _operator_matrix(basis, restrict(cubic)),
            mechanical_scale**2 * _operator_matrix(basis, restrict(quartic)),
            alpha=alpha,
            beta_cm2=beta_cm2,
            mixing_weight=mixing_weight,
            maximum_states=maximum_states,
            maximum_dense_bytes=maximum_dense_bytes,
            workers=workers,
            **extra,
        )
        bases.append(basis)
        matrices.append(result)
        component_diagnostics.append(result.diagnostics)
    states = []
    for selection in product(*bases):
        state = np.zeros(len(field.frequencies_cm), int)
        for modes, values in zip(components, selection, strict=True):
            state[list(modes)] = values
        states.append(state)
    energies = np.zeros(1)
    vectors = np.ones((1, 1), complex)
    for result in matrices:
        energies = (energies[:, None] + result.energies_cm).ravel()
        vectors = np.kron(vectors, result.eigenvectors)
    energies += mechanical_scale**2 * field.constant_shift_cm
    order = np.argsort(energies, kind="stable")
    state_array = np.asarray(states, int)
    boundary = np.zeros(len(states), bool)
    for modes in components:
        boundary |= state_array[:, list(modes)].sum(axis=1) == maximum_total_quanta
    boundary_weights = np.sum(abs(vectors[boundary][:, order]) ** 2, axis=0)
    return SAMHDCResult(
        np.asarray(states, int),
        energies[order],
        vectors[:, order],
        components,
        dict(
            schema="matrix.trinity.sam_hdc_completion.v1",
            model="regularized_hdc_unitary_normal_form"
            if formulation == "normal_form"
            else "regularized_hdc_pair_hamiltonian",
            formulation=formulation,
            quartic_globally_diagonalized=formulation == "quartic_variational",
            conventional_hdcpt2_equivalence=False,
            automatic_component_rule=True,
            field_sha256=identity,
            quartic_coverage=field.quartic_coverage,
            quartic_model_approximation=field.quartic_coverage == "semidiagonal_model",
            second_order_limit="declared_polynomial_model",
            bare_potential_unbounded_below_proved=bool(unbounded_modes),
            unbounded_modal_axes=unbounded_modes,
            global_potential_confinement_certified=False,
            local_unbounded_quartic_explicitly_allowed=allow_unbounded_quartic,
            boundary_weights=boundary_weights,
            component_modes=components,
            component_diagnostics=component_diagnostics,
            basis_size=len(energies),
            maximum_total_quanta_per_component=maximum_total_quanta,
            tensor_product_basis=True,
            mechanical_scale=mechanical_scale,
            infinite_basis_convergence_certified=False,
            amplitude_pruning_used=False,
            completion_change_frobenius_component_bounds_cm=[
                r.diagnostics["completion_change_frobenius_cm"] for r in matrices
            ],
        ),
        field.frequencies_cm,
        field.basis_id,
    )


def sam_polynomial_operator(result, terms):
    """Bare modal polynomial: (coefficient, ordered ((q/p, mode),...)) terms.

    Coefficients may be complex scalars or equally shaped tensors (e.g. a
    dipole vector). Output shape is (n_basis,n_basis,*coefficient_shape).
    q and p are dimensionless. Arbitrary supplied Taylor orders are supported.
    """
    terms = list(terms)
    if not terms:
        raise ValueError("at least one property term required")
    shape = np.asarray(terms[0][0]).shape
    n = len(result.frequencies_cm)
    output = np.zeros((len(result.basis), len(result.basis)) + shape, complex)
    lookup = {tuple(state): i for i, state in enumerate(result.basis)}
    for coefficient, operators in terms:
        coefficient = np.asarray(coefficient, complex)
        if coefficient.shape != shape or not np.all(np.isfinite(coefficient)):
            raise ValueError("finite property coefficients with one common shape required")
        operators = tuple(tuple(item) for item in operators)
        if any(
            len(item) != 2
            or item[0] not in {"q", "p"}
            or isinstance(item[1], bool)
            or not isinstance(item[1], (int, np.integer))
            or not 0 <= item[1] < n
            for item in operators
        ):
            raise ValueError("property factors must be (q/p, valid zero-based mode)")
        key = _canonical_operator_key(operators)
        for col, state in enumerate(result.basis):
            for target, value in _compiled_operator_couplings(tuple(state), ((key, 1.0),)).items():
                row = lookup.get(target)
                if row is not None:
                    output[row, col] += coefficient * value
    return output


def hdc_transition_spectrum(
    initial,
    final,
    harmonic_operator,
    *,
    initial_levels=(0,),
    final_levels=None,
    initial_populations=None,
):
    """Two-surface U_f^dagger O U_i, also for same-surface vibrational/hot bands.

    Signed energy differences and absolute supplied populations are retained.
    Returned strengths are in the supplied operator's squared units; no optical
    prefactor or experimental completeness certificate is inferred.
    """

    def levels(value, size, name):
        a = np.asarray(value)
        if (
            a.ndim != 1
            or not len(a)
            or a.dtype.kind not in "iu"
            or np.any(a < 0)
            or np.any(a >= size)
            or len(np.unique(a)) != len(a)
        ):
            raise ValueError(f"{name} must contain unique valid eigenlevel indices")
        return a.astype(int)

    ii = levels(initial_levels, len(initial.energies_cm), "initial_levels")
    ff = levels(
        np.arange(len(final.energies_cm)) if final_levels is None else final_levels,
        len(final.energies_cm),
        "final_levels",
    )
    if initial_populations is None:
        if not np.array_equal(ii, [0]):
            raise ValueError("hot bands require explicit absolute initial populations")
        population = np.ones(1)
    else:
        population = np.asarray(initial_populations, float)
        if (
            population.shape != (len(ii),)
            or not np.all(np.isfinite(population))
            or np.any(population < 0)
            or population.sum() > 1 + 1.0e-12
        ):
            raise ValueError("absolute nonnegative populations summing to at most one required")
    bare = np.asarray(harmonic_operator, complex)
    if (
        bare.ndim < 2
        or bare.shape[:2] != (len(final.basis), len(initial.basis))
        or not np.all(np.isfinite(bare))
    ):
        raise ValueError("finite harmonic operator must have (final_basis,initial_basis,...) axes")
    components = bare.shape[2:]
    channels = bare.reshape(len(final.basis), len(initial.basis), -1)
    amplitude = np.einsum(
        "fm,fic,in->mnc",
        final.eigenvectors[:, ff].conj(),
        channels,
        initial.eigenvectors[:, ii],
        optimize=True,
    ).transpose(1, 0, 2)
    strengths = np.sum(abs(amplitude) ** 2, axis=2)
    weights = np.repeat(population, len(ff))
    return dict(
        schema="matrix.trinity.hdc_transition_spectrum.v1",
        initial_levels=np.repeat(ii, len(ff)),
        final_levels=np.tile(ff, len(ii)),
        wavenumbers_cm=(final.energies_cm[ff][None, :] - initial.energies_cm[ii][:, None]).ravel(),
        amplitudes=amplitude.reshape((len(ii) * len(ff),) + components),
        strengths=strengths.ravel(),
        populations=weights,
        weighted_strengths=weights * strengths.ravel(),
        diagnostics=dict(
            initial_population_retained=float(population.sum()),
            final_eigenbasis_complete=len(ff) == len(final.energies_cm),
            physical_omission_bound=None,
            infinite_basis_convergence_certified=False,
            no_renormalization=True,
            conventional_hdcpt2_equivalence=False,
        ),
    )


def sam_thermal_populations(result, temperature_K):
    """Normalized finite-model Boltzmann probabilities, NOT a thermal tail certificate."""
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("temperature_K must be finite and positive")
    from matrix_chem import Phy, get_physical_constants

    c = get_physical_constants()
    inverse = c[Phy.PLANCK] * c[Phy.C_LIGHT] / (c[Phy.BOLTZMANN] * temperature_K)
    weights = np.exp(-inverse * (result.energies_cm - result.energies_cm[0]))
    return weights / weights.sum()


__all__ = [
    "HDCMatrixResult",
    "SAMHDCResult",
    "hdc_coupling_moduli",
    "complete_hdc_hamiltonian",
    "solve_sam_hdcpt2",
    "sam_polynomial_operator",
    "hdc_transition_spectrum",
    "sam_thermal_populations",
]


def audit_sam_hdc_convergence(
    coarse,
    fine,
    *,
    levels,
    energy_tolerance_cm=0.1,
    minimum_subspace_overlap=0.999,
    maximum_boundary_weight=1.0e-5,
):
    """Practical nested-basis audit; never an infinite-space error certificate.

    A singular-value subspace comparison is independent of eigenvector phases
    and rotations inside degenerate target manifolds. Energy order is retained;
    new low-lying intruders are detected, not silently relabelled away.
    """
    ids = np.asarray(levels)
    if (
        ids.ndim != 1
        or not len(ids)
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= min(len(coarse.energies_cm), len(fine.energies_cm)))
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("unique common eigenlevel indices required for convergence audit")
    if (
        not np.isfinite(energy_tolerance_cm)
        or energy_tolerance_cm <= 0
        or not 0 < minimum_subspace_overlap <= 1
        or not 0 <= maximum_boundary_weight <= 1
    ):
        raise ValueError("invalid convergence audit tolerances")
    if (
        not np.array_equal(coarse.frequencies_cm, fine.frequencies_cm)
        or coarse.components != fine.components
        or coarse.basis_id != fine.basis_id
    ):
        raise ValueError("convergence requires the same modal gauge and component partition")
    for key in ("mechanical_scale", "field_sha256", "formulation"):
        if coarse.diagnostics[key] != fine.diagnostics[key]:
            raise ValueError("different Hamiltonians cannot certify basis convergence")
    for a, b in zip(
        coarse.diagnostics["component_diagnostics"],
        fine.diagnostics["component_diagnostics"],
        strict=True,
    ):
        if any(a[k] != b[k] for k in ("alpha", "beta_cm2", "mixing_weight")):
            raise ValueError("different completion parameters in basis audit")
    lookup = {tuple(state): i for i, state in enumerate(fine.basis)}
    if any(tuple(state) not in lookup for state in coarse.basis):
        raise ValueError("coarse basis must be nested in fine basis")
    rows = np.array([lookup[tuple(state)] for state in coarse.basis])
    overlap = coarse.eigenvectors[:, ids].conj().T @ fine.eigenvectors[rows][:, ids]
    least_overlap = float(np.min(svd(overlap, compute_uv=False)) ** 2)
    change = abs(fine.energies_cm[ids] - coarse.energies_cm[ids])
    boundary = float(np.max(fine.diagnostics["boundary_weights"][ids]))
    return dict(
        schema="matrix.trinity.sam_hdc_basis_audit.v1",
        levels=ids,
        energy_changes_cm=change,
        minimum_subspace_overlap=least_overlap,
        maximum_boundary_weight=boundary,
        energy_tolerance_cm=energy_tolerance_cm,
        required_subspace_overlap=minimum_subspace_overlap,
        allowed_boundary_weight=maximum_boundary_weight,
        passed=bool(
            np.max(change) <= energy_tolerance_cm
            and least_overlap >= minimum_subspace_overlap
            and boundary <= maximum_boundary_weight
        ),
        bare_potential_unbounded_below_proved=fine.diagnostics[
            "bare_potential_unbounded_below_proved"
        ],
        infinite_basis_convergence_certified=False,
    )


def sam_quadratic_energy_model(result, occupations, levels):
    """Fit E0 + sum_i nu_i*n_i + sum_i Xii*n_i*(n_i-1) + sum_i<j Xij*n_i*n_j.

    Level-to-occupation assignments are EXPLICIT, never inferred in resonant
    manifolds. This projection is useful for density-of-states work; residuals
    describe only the supplied calibration states, not arbitrary hot states.
    """
    from .vibronic_vpt2 import _occupations

    states = np.asarray(_occupations(occupations, len(result.frequencies_cm), "occupations"), float)
    ids = np.asarray(levels)
    if (
        ids.shape != (len(states),)
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= len(result.energies_cm))
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("one unique valid eigenlevel per occupation assignment required")
    n = states.shape[1]
    columns = [np.ones(len(states))] + [states[:, i] for i in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    columns += [
        states[:, i] * (states[:, i] - 1) if i == j else states[:, i] * states[:, j]
        for i, j in pairs
    ]
    design = np.column_stack(columns)
    energy = result.energies_cm[ids]
    coefficients, _, rank, singular = np.linalg.lstsq(design, energy, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("quadratic energy calibration is rank deficient")
    x = np.zeros((n, n))
    for value, (i, j) in zip(coefficients[1 + n :], pairs, strict=True):
        x[i, j] = x[j, i] = value
    residual = energy - design @ coefficients
    return dict(
        schema="matrix.trinity.sam_hdc_quadratic_projection.v1",
        zero_point_cm=float(coefficients[0]),
        fundamentals_cm=coefficients[1 : 1 + n],
        x_matrix_cm=x,
        occupations=states.astype(int),
        levels=ids,
        residuals_cm=residual,
        maximum_calibration_residual_cm=float(np.max(abs(residual))),
        condition_number=float(singular[0] / singular[-1]),
        quadratic_exactness_certified=False,
        scope="quadratic projection on explicit assigned levels; no off-calibration accuracy guarantee",
    )


__all__ += ["audit_sam_hdc_convergence", "sam_quadratic_energy_model"]

__all__.append("complete_hdc_normal_form")
