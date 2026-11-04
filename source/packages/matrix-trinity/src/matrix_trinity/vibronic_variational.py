"""Finite variational anharmonic states and exact two-surface FC contractions.

The Hamiltonian is the supplied bare quadratic/cubic/quartic potential, without
HDC completion or perturbative wavefunction coefficients. Harmonic spectators
can be embedded without dropping their contribution to the Duschinsky overlap.
"""
from dataclasses import dataclass
from math import comb, prod
from itertools import combinations_with_replacement

import numpy as np
from numba import njit
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import voigt_profile
from threadpoolctl import threadpool_limits

from .curvilinear_vpt2 import _canonical_cubic_terms, _canonical_quartic_terms
from .hdc_hamiltonian import _shell_basis, _operator_matrix
from .vibronic_harmonic import harmonic_gaussian_generator, orient_harmonic_model
from .vibronic_rates import GOLDEN_RULE_FACTOR
from .vibronic_vpt2 import SAMVPT2Field, _positive_integer


@dataclass(frozen=True)
class FiniteVibrationalStates:
    basis: np.ndarray
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    frequencies_cm: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        basis = np.asarray(self.basis)
        frequencies = np.asarray(self.frequencies_cm, float)
        energies = np.asarray(self.energies_cm, float)
        vectors = np.asarray(self.eigenvectors, complex)
        if (frequencies.ndim != 1 or not len(frequencies) or np.any(frequencies <= 0)
                or not np.all(np.isfinite(frequencies)) or basis.ndim != 2
                or basis.shape[1] != len(frequencies) or not len(basis)
                or basis.dtype.kind not in "iu" or np.any(basis < 0)
                or len(np.unique(basis, axis=0)) != len(basis)
                or energies.ndim != 1 or not len(energies)
                or not np.all(np.isfinite(energies)) or np.any(np.diff(energies) < 0)
                or vectors.shape != (len(basis), len(energies)) or not np.all(np.isfinite(vectors))):
            raise ValueError("compatible finite harmonic basis, ordered energies and eigenvectors required")
        orthogonality_vectors = vectors.real if not np.any(vectors.imag) else vectors
        if np.max(np.abs(orthogonality_vectors.conj().T @ orthogonality_vectors - np.eye(len(energies)))) > 1e-8:
            raise ValueError("variational eigenvectors must be orthonormal")
        for name, array in (("basis", basis), ("energies_cm", energies),
                            ("eigenvectors", vectors), ("frequencies_cm", frequencies)):
            array = np.array(array, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def project_anharmonic_potential(field, target_frequencies_cm, source_from_target, *,
                                 allow_rank_deficient=False):
    """Project F3/F4 under q_source=A q_target and replace the quadratic model.

    A may be rectangular for an explicitly selected active space. This is a
    tangent-model transport, not a Taylor translation between geometries.
    No source-coordinate semidiagonal truncation is reapplied after transport.
    An explicitly allowed rank-deficient map leaves its null directions harmonic;
    it does not discard the target quadratic coordinates.
    """
    if not isinstance(field, SAMVPT2Field):
        raise TypeError("SAMVPT2Field required")
    if np.any(field.metric_first_cm) or np.any(field.metric_second_cm):
        raise ValueError("potential projection requires zero metric derivatives")
    if not isinstance(allow_rank_deficient,(bool,np.bool_)):
        raise ValueError('explicit boolean anharmonic rank policy required')
    target = np.asarray(target_frequencies_cm, float)
    transform = np.asarray(source_from_target, float)
    if (target.ndim != 1 or not len(target) or transform.shape != (len(field.frequencies_cm), len(target))
            or not np.all(np.isfinite(transform))
            or (not allow_rank_deficient and np.linalg.matrix_rank(transform) != len(target))):
        raise ValueError("finite full-column-rank source-from-target coordinate map required")
    cubic = np.einsum("abc,ai,bj,ck->ijk", field.cubic_cm, transform, transform, transform, optimize=True)
    quartic = np.einsum("abcd,ai,bj,ck,dl->ijkl", field.quartic_cm,
                        transform, transform, transform, transform, optimize=True)
    return SAMVPT2Field(target, cubic, quartic, constant_shift_cm=field.constant_shift_cm,
                        quartic_coverage=field.quartic_coverage)


def _quartic_gram_certificate(f):
    """Seek a positive Gram representation without changing the quartic form.

    On z=(q_i**2, sqrt(2)*q_i*q_j), a shift of G[ii,jj] by p is
    canceled by a shift of G[ij,ij] by -p. Optimization only searches this
    polynomial null space; the returned eigenvalue is checked independently.
    Failure to find a certificate does not imply an unbounded potential.
    """
    n = len(f)
    pairs = list(combinations_with_replacement(range(n), 2))
    symmetric = np.zeros((n*n, len(pairs)))
    for column, (i, j) in enumerate(pairs):
        symmetric[i*n+j, column] = 1 if i == j else 1/np.sqrt(2)
        symmetric[j*n+i, column] = symmetric[i*n+j, column]
    gram = symmetric.T @ f.reshape(n*n, n*n) @ symmetric
    scale = max(float(np.linalg.norm(gram)), np.finfo(float).tiny)
    shifts = []
    for i in range(n):
        for j in range(i+1, n):
            delta = np.zeros_like(gram)
            a, b, c = pairs.index((i, i)), pairs.index((j, j)), pairs.index((i, j))
            delta[a, b] = delta[b, a] = 1
            delta[c, c] = -1
            shifts.append(delta)
    minimum = float(np.linalg.eigvalsh(gram)[0])
    if minimum > 1e-10*scale or n > 6:
        return gram, minimum
    shifts = np.asarray(shifts)
    normalized = gram/scale
    make = lambda p: normalized + np.einsum('k,kij->ij', p[:-1], shifts)
    start = np.r_[np.zeros(len(shifts)), minimum/scale]
    fit = minimize(lambda p: -p[-1], start, method='SLSQP',
        constraints={'type': 'ineq', 'fun': lambda p: np.linalg.eigvalsh(make(p))-p[-1]},
        options={'maxiter': 256, 'ftol': 1e-10})
    candidate = make(fit.x)*scale
    candidate_minimum = float(np.linalg.eigvalsh(candidate)[0])
    return (candidate, candidate_minimum) if candidate_minimum > minimum else (gram, minimum)


def audit_quartic_confinement(field):
    """Return a sufficient confinement certificate or an explicit negative ray.

    A negative leading quartic on any direction proves unboundedness. The
    direction search is diagnostic; failure to find one proves nothing.
    Neither the polynomial nor its coefficients are regularized.
    """
    if not isinstance(field, SAMVPT2Field):
        raise TypeError('SAMVPT2Field required')
    try:
        certificate = _confinement(field)
        report = dict(status='confinement certified', certificate=certificate,
                      potential_modified=False)
        if len(field.frequencies_cm) > 2 and np.any(field.quartic_cm):
            gram, minimum = _quartic_gram_certificate(field.quartic_cm)
            report.update(quartic_gram=gram, quartic_gram_minimum_eigenvalue_cm=minimum,
                gram_monomials='q_i**2 and sqrt(2)*q_i*q_j in combinations_with_replacement order')
        return report
    except ValueError:
        pass
    n = len(field.frequencies_cm)
    quartic = field.quartic_cm
    value = lambda x: float(np.einsum('ijkl,i,j,k,l', quartic, x, x, x, x))/(x@x)**2
    starts = np.r_[np.eye(n), np.random.default_rng(61453).normal(size=(32, n))]
    best = min((minimize(value, x, method='BFGS') for x in starts), key=lambda fit: fit.fun)
    direction = best.x/np.linalg.norm(best.x)
    leading = value(direction)
    negative = leading < -1e-10*max(float(np.linalg.norm(quartic)), 1)
    return dict(status='unbounded negative quartic ray' if negative else 'confinement uncertified',
                direction=direction, quartic_contraction_cm=leading,
                quadratic_coefficient_cm=float(np.sum(field.frequencies_cm*direction**2)/2),
                cubic_coefficient_cm=float(np.einsum('ijk,i,j,k', field.cubic_cm,
                                                    direction, direction, direction)/6),
                quartic_coefficient_cm=leading/24, potential_modified=False)


def _confinement(field):
    """Sufficient global confinement checks, exact for binary positive quartics."""
    f = field.quartic_cm
    if not np.any(f) and not np.any(field.cubic_cm):
        return "positive harmonic potential"
    n = len(field.frequencies_cm)
    # Exact uncoupled harmonic directions need no positive quartic term.
    # Removing only identically absent tensor rows preserves a separable sum
    # of a certified anharmonic block and positive harmonic oscillators.
    coupled = np.flatnonzero(np.any(field.cubic_cm != 0, axis=(1, 2))
                             | np.any(f != 0, axis=(1, 2, 3)))
    if 0 < len(coupled) < n:
        core = SAMVPT2Field(field.frequencies_cm[coupled],
            field.cubic_cm[np.ix_(coupled, coupled, coupled)],
            f[np.ix_(coupled, coupled, coupled, coupled)],
            quartic_coverage=field.quartic_coverage)
        return _confinement(core) + " plus independent positive harmonic directions"
    if n == 1 and f[0, 0, 0, 0] > 0:
        return "positive one-mode quartic leading term"
    if n == 2 and f[0, 0, 0, 0] > 0 and f[1, 1, 1, 1] > 0:
        # 24 V4(1,t): strictly positive for all real t, and at x=0.
        polynomial = np.array([f[0, 0, 0, 0], 4*f[0, 0, 0, 1], 6*f[0, 0, 1, 1],
                               4*f[0, 1, 1, 1], f[1, 1, 1, 1]])
        roots = np.polynomial.polynomial.polyroots(np.arange(1, 5) * polynomial[1:])
        real = roots.real[np.abs(roots.imag) < 1e-9 * (1 + np.abs(roots.real))]
        if np.min(np.polynomial.polynomial.polyval(real, polynomial)) > 1e-10 * np.max(abs(polynomial)):
            return "strictly positive binary quartic leading form"
    if n > 2:
        matrix, minimum = _quartic_gram_certificate(f)
        if minimum > 1e-10 * np.linalg.norm(matrix):
            return "positive-definite polynomial-equivalent quartic Gram form"
    raise ValueError("confinement of the supplied bare quartic potential is not certified; supply a confining potential")


def solve_quartic_variational(field, *, maximum_total_quanta=24, level_count=None,
                              maximum_states=4096, maximum_dense_bytes=512*1024**2,
                              workers=1, maximum_harmonic_energy_cm=None, real_arithmetic=False):
    """Diagonalize H0+V3+V4 in an explicitly bounded HO basis, with no HDC map."""
    if not isinstance(field, SAMVPT2Field):
        raise TypeError("SAMVPT2Field required")
    if not isinstance(real_arithmetic, bool):
        raise TypeError('real_arithmetic must be boolean')
    if np.any(field.metric_first_cm) or np.any(field.metric_second_cm):
        raise ValueError("this bare variational solver supports potential-only modal fields")
    for name, value in (("maximum_total_quanta", maximum_total_quanta),
                        ("maximum_states", maximum_states), ("maximum_dense_bytes", maximum_dense_bytes),
                        ("workers", workers)):
        _positive_integer(value, name)
    confinement = _confinement(field)
    n = len(field.frequencies_cm)
    if maximum_harmonic_energy_cm is not None and (
            not np.isfinite(maximum_harmonic_energy_cm) or maximum_harmonic_energy_cm <= 0):
        raise ValueError('positive finite harmonic excitation energy bound required')
    if maximum_harmonic_energy_cm is None:
        size = comb(n + maximum_total_quanta, n)
        basis = None
    else:
        # Enumerate only states inside both bounds, without allocating a larger
        # shell basis first. Bound applies to excitation energy, excluding ZPE.
        rows = []
        def visit(prefix, mode, remaining, energy):
            if mode == n:
                rows.append(tuple(prefix))
                if len(rows) > maximum_states:
                    raise MemoryError('variational basis exceeds its declared state budget')
                return
            limit = min(remaining, int(np.floor((maximum_harmonic_energy_cm-energy)
                                                /field.frequencies_cm[mode]+1e-12)))
            for occupation in range(limit+1):
                visit(prefix+[occupation], mode+1, remaining-occupation,
                      energy+occupation*field.frequencies_cm[mode])
        visit([], 0, maximum_total_quanta, 0.0)
        rows.sort(key=lambda row: (sum(row), row))
        basis = np.asarray(rows, np.int64)
        size = len(basis)
    if size > maximum_states or 6 * size * size * 16 > maximum_dense_bytes:
        raise MemoryError("variational Hamiltonian exceeds its declared state or memory budget")
    count = size if level_count is None else level_count
    _positive_integer(count, "level_count")
    if count > size:
        raise ValueError("level_count exceeds the variational basis dimension")
    if basis is None:
        basis = np.asarray(_shell_basis(n, maximum_total_quanta), np.int64)
    terms = (_canonical_cubic_terms(field.cubic_cm, field.metric_first_cm, 0.0)
             + _canonical_quartic_terms(field.quartic_cm, field.metric_second_cm))
    matrix = _operator_matrix(tuple(map(tuple, basis)), terms)
    matrix[np.diag_indices(size)] += (basis + 0.5) @ field.frequencies_cm + field.constant_shift_cm
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-10):
        raise ArithmeticError("bare variational Hamiltonian is not Hermitian")
    if real_arithmetic:
        if np.any(matrix.imag):
            raise ValueError('real arithmetic requires an exactly real Hamiltonian')
        matrix = np.array(matrix.real, copy=True)
    with threadpool_limits(limits=workers, user_api="blas"):
        energies, vectors = eigh(matrix, subset_by_index=(0, count-1), check_finite=False)
        residual = np.linalg.norm(matrix @ vectors - vectors * energies, axis=0)
    boundary = basis.sum(axis=1) > max(-1, maximum_total_quanta-4)
    energy_boundary = (np.zeros(size, bool) if maximum_harmonic_energy_cm is None else
        basis @ field.frequencies_cm > maximum_harmonic_energy_cm-4*max(field.frequencies_cm))
    return FiniteVibrationalStates(basis, energies, vectors, field.frequencies_cm, dict(
        schema="matrix.sati.bare_quartic_variational.v1", hamiltonian="H0+V3+V4",
        hdc_completion_used=False, confinement=confinement,
        numerical_arithmetic='real' if real_arithmetic else 'complex',
        maximum_total_quanta=maximum_total_quanta, basis_size=size,
        maximum_harmonic_energy_cm=maximum_harmonic_energy_cm,
        energy_boundary_probabilities=np.sum(abs(vectors[energy_boundary])**2, axis=0),
        residual_norms_cm=residual, boundary_probabilities=np.sum(abs(vectors[boundary])**2, axis=0),
        quartic_coverage=field.quartic_coverage, infinite_basis_convergence_certified=False,
    ))


def embed_variational_states(states, full_frequencies_cm, active_modes):
    """Embed an active block with all remaining harmonic modes in their vacuum."""
    frequencies = np.asarray(full_frequencies_cm, float)
    modes = np.asarray(active_modes)
    if (modes.ndim != 1 or modes.dtype.kind not in "iu" or len(modes) != len(states.frequencies_cm)
            or np.any(modes < 0) or np.any(modes >= len(frequencies))
            or len(np.unique(modes)) != len(modes)
            or not np.array_equal(frequencies[modes], states.frequencies_cm)):
        raise ValueError("active mode indices and harmonic frequencies must match exactly")
    basis = np.zeros((len(states.basis), len(frequencies)), np.int64)
    basis[:, modes] = states.basis
    spectators = np.ones(len(frequencies), bool)
    spectators[modes] = False
    return FiniteVibrationalStates(basis, states.energies_cm + 0.5*frequencies[spectators].sum(),
                                  states.eigenvectors, frequencies,
                                  {**states.diagnostics, "active_modes": modes.tolist(),
                                   "spectator_state": "harmonic vacuum"})


@njit(cache=True)
def _gaussian_overlap_grid(matrix, linear, dimensions, strides, ground):
    size = 1
    for d in dimensions:
        size *= d
    table = np.zeros(size)
    table[0] = ground
    counts = np.empty(len(dimensions), np.int64)
    for index in range(1, size):
        for k in range(len(dimensions)):
            counts[k] = (index // strides[k]) % dimensions[k]
        j = 0
        while counts[j] == 0:
            j += 1
        denominator = np.sqrt(counts[j])
        previous = index-strides[j]
        counts[j] -= 1
        value = linear[j]*table[previous]
        for k in range(len(dimensions)):
            if counts[k]:
                value += matrix[j, k]*np.sqrt(counts[k])*table[previous-strides[k]]
        table[index] = value/denominator
    return table


def harmonic_overlap_matrix(model, initial_basis, final_basis, *, maximum_grid_entries=5_000_000,
                             maximum_harmonic_pairs=2_000_000,backend='python',native_executable=None):
    """Exact finite-basis FC matrix with the entire two-surface Duschinsky map.

    Vacuum spectator integrals are retained in the full Gaussian generator.
    Only never-excited Gaussian variables are absent from the recurrence grid.
    There is no amplitude, coefficient or mode-map threshold.
    """
    n = len(model.initial_frequencies_cm)
    bases = []
    for value in (initial_basis, final_basis):
        value = np.asarray(value)
        if (value.ndim != 2 or not len(value) or value.shape[1] != n
                or value.dtype.kind not in "iu" or np.any(value < 0)
                or np.any(value > np.iinfo(np.int32).max)):
            raise ValueError("compatible nonnegative integer harmonic basis rows required")
        bases.append(np.asarray(value, np.int64))
    initial, final = bases
    _positive_integer(maximum_grid_entries, "maximum_grid_entries")
    _positive_integer(maximum_harmonic_pairs, "maximum_harmonic_pairs")
    if len(initial)*len(final) > maximum_harmonic_pairs:
        raise MemoryError("harmonic overlap pair budget exceeded")
    maximum = np.r_[initial.max(axis=0), final.max(axis=0)]
    active = np.flatnonzero(maximum)
    dimensions = maximum[active]+1
    entries = prod(int(d) for d in dimensions)
    if entries > maximum_grid_entries:
        raise MemoryError("exact Gaussian overlap grid exceeds its declared budget")
    strides = np.array([prod(int(d) for d in dimensions[k+1:]) for k in range(len(active))], np.int64)
    if backend not in ('python','cpp'):raise ValueError('backend must be python or cpp')
    library=None
    if backend=='cpp':
        if native_executable is None:raise ValueError('Explicit native_executable required for C++')
        from .vibronic_native_kernels import NativeKernels
        library=NativeKernels(str(native_executable)+'.kernels.so')
    generator = harmonic_gaussian_generator(model) if library is None else library.prepare_dipole(model,0.)
    if generator["log_ground"] < np.log(np.finfo(float).tiny)+4:
        raise FloatingPointError("overlap vacuum amplitude underflows")
    grid=_gaussian_overlap_grid if library is None else library.overlap_grid
    table = grid(generator["matrix"][np.ix_(active, active)],
                                   generator["linear"][active], dimensions, strides,
                                   np.exp(generator["log_ground"]))
    si = np.zeros(n, np.int64)
    sf = np.zeros(n, np.int64)
    for mode, stride in zip(active, strides):
        if mode < n:
            si[mode] = stride
        else:
            sf[mode-n] = stride
    matrix = table[(final @ sf)[:, None] + (initial @ si)[None, :]]
    return matrix, dict(gaussian_grid_entries=entries, harmonic_pairs=len(initial)*len(final),
                        full_duschinsky_retained=True, amplitude_pruning_used=False)


def gaussian_number_moment(matrix, linear, log_ground, initial_states, *,
                           final_log_weights=None, initial_level=0,
                           maximum_grid_entries=8_000_000):
    """Sum squared Gaussian overlap amplitudes with exponential final-number weights.

    The generating function has initial variables first, final variables last.
    A Mehler contraction sums *all* final harmonic occupations analytically;
    only the supplied finite initial wavefunction is projected. With spectator
    log weights t>=0 this gives a Chernoff numerator for occupation-tail bounds.
    The zero-weight result is the norm of the supplied Gaussian transformation,
    which need not be unitary after an explicitly applied matrix threshold.
    """
    if not isinstance(initial_states, FiniteVibrationalStates):
        raise TypeError('FiniteVibrationalStates required')
    if (np.iscomplexobj(matrix) or np.iscomplexobj(linear)
            or np.iscomplexobj(final_log_weights)):
        raise ValueError('the Gaussian number moment requires real generator data and weights')
    n = len(initial_states.frequencies_cm)
    m, l = np.asarray(matrix, float), np.asarray(linear, float)
    weights = np.zeros(n) if final_log_weights is None else np.asarray(final_log_weights, float)
    if (m.shape != (2*n, 2*n) or l.shape != (2*n,) or weights.shape != (n,)
            or not np.all(np.isfinite(m)) or not np.all(np.isfinite(l))
            or not np.all(np.isfinite(weights)) or not np.isfinite(log_ground)
            or not np.allclose(m, m.T, rtol=1e-12, atol=1e-12)):
        raise ValueError('finite symmetric real Gaussian generator and final log weights required')
    if isinstance(initial_level, bool) or not isinstance(initial_level, (int, np.integer)) or not 0 <= initial_level < len(initial_states.energies_cm):
        raise ValueError('initial level outside the finite state set')
    _positive_integer(maximum_grid_entries, 'maximum_grid_entries')
    scale = np.exp(weights/2)
    if not np.all(np.isfinite(scale)):
        raise FloatingPointError('number weights overflow')
    a = scale[:, None]*m[n:, n:]*scale[None, :]
    radius = float(np.max(abs(np.linalg.eigvalsh(a))))
    if radius >= 1-1e-12:
        raise ValueError('weighted Mehler contraction is not normalizable')
    b = scale[:, None]*m[n:, :n]
    lf = scale*l[n:]
    u = np.linalg.solve(np.eye(n)-a@a, np.eye(n))
    shift = np.linalg.solve(np.eye(n)-a, lf)
    q = m[:n, :n]+b.T@a@u@b
    r = b.T@u@b
    h = l[:n]+b.T@shift
    sign, determinant = np.linalg.slogdet(np.eye(n)-a@a)
    if sign <= 0:
        raise ArithmeticError('invalid weighted Mehler determinant')
    log_norm = 2*float(log_ground)-.5*determinant+lf@shift
    maxima = initial_states.basis.max(axis=0)
    active = np.flatnonzero(maxima)
    dimensions = np.tile(maxima[active]+1, 2)
    entries = prod(int(d) for d in dimensions)
    rows = initial_states.basis[:, active]
    if entries<=maximum_grid_entries:
        # Preserve established arithmetic/cache identities for existing small grids.
        joint=np.block([[q,r],[r.T,q]]);indices=np.r_[active,n+active]
        joint=joint[np.ix_(indices,indices)];joint=(joint+joint.T)/2
        strides=np.array([prod(int(d) for d in dimensions[k+1:]) for k in range(len(dimensions))],np.int64)
        table=_gaussian_overlap_grid(joint,np.tile(h,2)[indices],dimensions,strides,1.)
        width=len(active)
        gram=table[(rows@strides[:width])[:,None]+(rows@strides[width:])[None,:]]
        grid_diagnostics=dict(initial_moment_grid_entries=entries,moment_algorithm='Cartesian Gaussian recurrence')
    else:
        from .gaussian_moment_gram import gaussian_gram_on_lowering_closure
        gram,grid_diagnostics=gaussian_gram_on_lowering_closure(rows,
            ((q+q.T)/2)[np.ix_(active,active)],((r+r.T)/2)[np.ix_(active,active)],h[active],
            maximum_entries=maximum_grid_entries)
    coefficients = initial_states.eigenvectors[:, initial_level]
    contraction = float(np.vdot(coefficients, gram@coefficients).real)
    moment = float(np.exp(log_norm)*contraction)
    if not np.isfinite(moment) or moment <= 0:
        raise FloatingPointError('Gaussian number moment is nonpositive or unresolved')
    return dict(moment=moment, weighted_final_gaussian_spectral_radius=radius,
                **grid_diagnostics,cartesian_bounding_box_entries=entries,final_occupation_sum_analytic=True,
                final_log_weights=weights, gaussian_transform_assumed_unitary=False)


def variational_vibronic_transitions(model, initial, final, coupling_hartree, *,
                                    kind="emission", initial_levels=(0,), final_levels=None,
                                    gaussian_sigma_cm=21.23, lorentz_hwhm_cm=0.0,
                                    maximum_grid_entries=5_000_000,
                                    maximum_harmonic_pairs=2_000_000,backend='python',native_executable=None):
    """Compute U_f^dagger S(J,K) U_i between supplied finite variational states.

    Inputs refer to model.initial/model.final surfaces; emission swaps both.
    Eigenlevel IDs are energy ordered and are never interpreted as occupations.
    Returned rates sum ONLY the selected transitions and are not certified total
    nonradiative rates. Absolute vibrational energies include spectator ZPE.
    """
    if (not np.array_equal(initial.frequencies_cm, model.initial_frequencies_cm)
            or not np.array_equal(final.frequencies_cm, model.final_frequencies_cm)):
        raise ValueError("variational harmonic bases must match the vibronic model exactly")
    coupling = np.asarray(coupling_hartree, complex)
    if coupling.ndim != 1 or not len(coupling) or not np.all(np.isfinite(coupling)):
        raise ValueError("finite constant coupling channels in Hartree required")
    if (not np.isfinite(gaussian_sigma_cm+lorentz_hwhm_cm) or min(gaussian_sigma_cm, lorentz_hwhm_cm) < 0
            or gaussian_sigma_cm+lorentz_hwhm_cm <= 0):
        raise ValueError("finite nonzero Voigt linewidth required")
    oriented, direction = orient_harmonic_model(model, kind)
    if kind == "emission":
        initial, final = final, initial

    def selected(value, count):
        value = np.asarray(value)
        if (value.ndim != 1 or not len(value) or value.dtype.kind not in "iu"
                or np.any(value < 0) or np.any(value >= count) or len(np.unique(value)) != len(value)):
            raise ValueError("unique valid energy-ordered eigenlevel IDs required")
        return value

    ii = selected(initial_levels, len(initial.energies_cm))
    ff = selected(np.arange(len(final.energies_cm)) if final_levels is None else final_levels, len(final.energies_cm))
    overlap, diagnostics = harmonic_overlap_matrix(
        oriented, initial.basis, final.basis,
        maximum_grid_entries=maximum_grid_entries, maximum_harmonic_pairs=maximum_harmonic_pairs,
        backend=backend,native_executable=native_executable,
    )
    amplitude = final.eigenvectors[:, ff].conj().T @ overlap @ initial.eigenvectors[:, ii]
    strengths = abs(amplitude.T)**2 * float(np.sum(abs(coupling)**2))
    mismatch = direction*(oriented.electronic_gap_cm + final.energies_cm[ff][None, :]
                          - initial.energies_cm[ii][:, None])
    contributions = GOLDEN_RULE_FACTOR*strengths*voigt_profile(mismatch, gaussian_sigma_cm, lorentz_hwhm_cm)
    return dict(
        schema="matrix.sati.variational_vibronic_transitions.v1", kind=kind,
        initial_levels=ii, final_levels=ff, amplitudes=amplitude.T,
        strengths_hartree2=strengths, mismatch_cm=mismatch,
        rate_contributions_s_1=contributions, selected_rate_s_1=contributions.sum(axis=1),
        initial_energies_cm=initial.energies_cm[ii], final_energies_cm=final.energies_cm[ff],
        initial_solver_diagnostics=initial.diagnostics, final_solver_diagnostics=final.diagnostics,
        diagnostics={**diagnostics, "transition_rate_converged": False,
                     "wavefunctions": "finite bare quartic variational eigenstates",
                     "scope": "selected eigenlevels; harmonic spectator vacuum; no total-rate completeness claim"},
    )


__all__ = ["FiniteVibrationalStates", "audit_quartic_confinement", "project_anharmonic_potential", "solve_quartic_variational",
           "embed_variational_states", "harmonic_overlap_matrix", "gaussian_number_moment", "variational_vibronic_transitions"]
