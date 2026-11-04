"""Fixed two-stage composition of analytical two-state rotations.

Research prototype for small common supports, not a production occupation
solver. The scalar HDC energy model is neither fitted nor recomputed here.
There is no spectral solve or convergence loop. The residual stage is a
fixed-depth Jacobi-type approximation, not an exact collective closure.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _off_diagonal_norm(value):
    off = value.copy()
    np.fill_diagonal(off, 0)
    return float(np.linalg.norm(off))


def _finite_hermitian(value, name):
    a = np.array(value, dtype=complex, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite square matrix")
    scale = max(1., float(np.max(abs(a), initial=0.)))
    if np.max(abs(a-a.conj().T), initial=0.) > 1e-12*scale:
        raise ValueError(f"{name} must be Hermitian")
    return .5*a + .5*a.conj().T


def _dc_shift(gap, coupling):
    if coupling == 0:
        return 0.
    # s = g^2 / (sqrt((d/2)^2+g^2)+d/2), without forming g^2.
    radius = np.hypot(gap/2, coupling)
    with np.errstate(over="ignore", invalid="ignore"):
        value = coupling*(coupling/(radius+gap/2))
    if not np.isfinite(value):
        raise FloatingPointError("pair displacement exceeds floating-point range")
    return float(value)


def _palindrome(h, pairs, *, environment=False, collective=False):
    """Frozen pair angles; the second half reverses their fixed order.

    H is not updated while constructing the angles. Each gate is an explicit
    two-state rotation. This differs from an iterative Jacobi eigensolver.
    """
    # Angles are homogeneous in energy. Work in dimensionless variables so
    # environment Gram matrices and squared gaps cannot overflow. Subtracting
    # a common energy origin also makes its irrelevance explicit.
    normalized = h.copy()
    diagonal = h.diagonal().real-h[0, 0].real
    if not np.all(np.isfinite(diagonal)):
        raise FloatingPointError("reference span exceeds floating-point range")
    normalized[np.diag_indices_from(h)] = diagonal
    scale = float(np.max(abs(normalized), initial=0.))
    if scale:
        normalized /= scale
    rotations = []
    unresolved = []
    for a, b in pairs:
        z = normalized[b, a]
        gap = float(normalized[b, b].real-normalized[a, a].real)
        gram_a = gram_b = shift_a = shift_b = 0.
        gram_ab = 0j
        if environment:
            for k in range(len(h)):
                if k in (a, b):
                    continue
                x, y = normalized[a, k], normalized[b, k]
                aa, bb = abs(x)**2, abs(y)**2
                q2 = aa+bb
                gram_a += aa
                gram_b += bb
                gram_ab += x*y.conjugate()
                if q2 == 0:
                    continue
                da = float(normalized[a, a].real-normalized[k, k].real)
                db = float(normalized[b, b].real-normalized[k, k].real)
                fa = da/(da*da+4*q2)
                fb = db/(db*db+4*q2)
                shift_a += aa*fa
                shift_b += bb*fb
                z += .5*y*x.conjugate()*(fa+fb)
            gap += shift_b-shift_a
        modulus = abs(z)
        effective_norm = modulus
        if collective:
            # Largest eigenvalue of the TWO-by-TWO environment Gram matrix,
            # evaluated in closed form, not by a physical eigensolver.
            ell = .5*(gram_a+gram_b+np.hypot(gram_a-gram_b, 2*abs(gram_ab)))
            if ell > 0:
                radius = np.hypot(gap, 2*modulus)
                environment_gap = np.hypot(radius, 2*np.sqrt(ell))
                # eta=1-radius/environment_gap, written without cancellation.
                # It is twice the DC partner weight for this collective scale.
                eta = (2*np.sqrt(ell)/environment_gap)*(2*np.sqrt(ell)/(environment_gap+radius))
                effective_norm = np.hypot(modulus, np.sqrt(eta)*np.sqrt(ell))
        if not np.isfinite(gap):
            raise FloatingPointError("pair gap exceeds floating-point range")
        if effective_norm == 0 and gap == 0:
            # A frame convention only. This choice is NOT a continuity proof.
            theta = 0.
            phase = 1.+0j
            unresolved.append((a, b))
        else:
            phase = z/modulus if modulus else 1.+0j
            if effective_norm:
                theta = (modulus/effective_norm)*.5*np.arctan2(effective_norm, gap/2)
            else:
                theta = .5*np.arctan2(modulus, gap/2)
        rotations.append((a, b, float(theta), complex(phase)))

    u = np.eye(len(h), dtype=complex)
    for a, b, theta, phase in rotations+list(reversed(rotations)):
        c, s = np.cos(theta/2), np.sin(theta/2)
        left, right = u[:, a].copy(), u[:, b].copy()
        u[:, a] = c*left-phase*s*right
        u[:, b] = phase.conjugate()*s*left+c*right
    return u, tuple(unresolved)


def _regularized_residual_gate(current, a, b):
    """Phase-free gate with quadratic cancellation of regularization bias.

    For mu>0 the generator is z times a continuous real scalar, including
    at z=gap=0. The three extrapolation weights cancel the terms linear and
    quadratic in mu; they are fixed algebraic coefficients, not fitted ones.
    """
    z = current[b, a]
    g = abs(z)
    d = float(current[b, b].real-current[a, a].real)
    other = [k for k in range(len(current)) if k not in (a, b)]
    x, y = current[a, other], current[b, other]
    aa, bb = float(np.vdot(x, x).real), float(np.vdot(y, y).real)
    cross = x @ y.conjugate()
    ell = .5*(aa+bb+np.hypot(aa-bb, 2*abs(cross)))
    mu = 0.
    if ell > 0:
        radius = np.hypot(d, 2*g)
        total_radius = np.hypot(radius, 2*np.sqrt(ell))
        exposure = (2*np.sqrt(ell)/total_radius)*(2*np.sqrt(ell)/(total_radius+radius))
        mu = exposure*ell
    if mu > 0:
        def alpha(t):
            q = np.hypot(g, np.sqrt(t))
            return .5*np.arctan2(q, d/2)/q
        coefficient = alpha(mu)/3-2*alpha(mu/2)+(8/3)*alpha(mu/4)
        theta = g*coefficient
        # np.sinc(x)=sin(pi*x)/(pi*x). No arg(z) or division by |z|.
        off = z*coefficient*np.sinc(theta/np.pi)
        return float(np.cos(theta)), complex(off), False
    if g == 0 and d == 0:
        # Here ell=0 in exact arithmetic: a truly isolated degenerate pair.
        return 1., 0j, True
    theta = .5*np.arctan2(g, d/2)
    phase = z/g if g else 1.+0j
    return float(np.cos(theta)), complex(phase*np.sin(theta)), False


def _residual_feedback(h, pairs, *, regularized=False):
    """One fixed forward/backward sequence, with current two-state angles.

    Every gate updates the common residual before the next gate is evaluated.
    This is a predetermined finite Jacobi-type composition, not a convergence
    iteration. It contains exactly as many gates as the frozen palindrome.
    """
    current = h.copy()
    diagonal = h.diagonal().real-h[0, 0].real
    if not np.all(np.isfinite(diagonal)):
        raise FloatingPointError("residual span exceeds floating-point range")
    current[np.diag_indices_from(h)] = diagonal
    scale = float(np.max(abs(current), initial=0.))
    if scale:
        current /= scale
    u = np.eye(len(h), dtype=complex)
    unresolved = []
    for a, b in list(pairs)+list(reversed(pairs)):
        if regularized:
            c, off, degenerate = _regularized_residual_gate(current, a, b)
            if degenerate:
                unresolved.append((a, b))
        else:
            z = current[b, a]
            modulus = abs(z)
            gap = float(current[b, b].real-current[a, a].real)
            if modulus == 0 and gap == 0:
                theta, phase = 0., 1.+0j
                unresolved.append((a, b))
            else:
                theta = .5*np.arctan2(modulus, gap/2)
                phase = z/modulus if modulus else 1.+0j
            c, off = np.cos(theta), phase*np.sin(theta)
        # Right multiplication changes two columns; left multiplication by
        # the adjoint changes two rows. No dense elementary rotation is built.
        left, right = u[:, a].copy(), u[:, b].copy()
        u[:, a] = c*left-off*right
        u[:, b] = off.conjugate()*left+c*right
        left, right = current[:, a].copy(), current[:, b].copy()
        current[:, a] = c*left-off*right
        current[:, b] = off.conjugate()*left+c*right
        top, bottom = current[a, :].copy(), current[b, :].copy()
        current[a, :] = c*top-off.conjugate()*bottom
        current[b, :] = off*top+c*bottom
        # Only these rows/columns can acquire a Hermitian rounding defect.
        # Restoring them costs O(n), rather than copying the whole matrix at
        # every gate (which would turn the fixed sequence into O(n^4)).
        for index in (a, b):
            column = .5*current[:, index]+.5*current[index, :].conjugate()
            current[:, index] = column
            current[index, :] = column.conjugate()
    return u, tuple(unresolved)


def _columns(values, size):
    result = tuple(values)
    if (not result or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                          or i < 0 or i >= size for i in result)
            or len(set(result)) != len(result)):
        raise ValueError("nonempty distinct valid integer column indices required")
    return list(result)


@dataclass(frozen=True)
class PairCompositionResult:
    pair_hamiltonian_cm: np.ndarray
    first_rotation: np.ndarray
    first_residual_cm: np.ndarray
    residual_rotation: np.ndarray
    rotation: np.ndarray
    diagnostics: dict

    def projector(self, columns):
        """Projector of a supplied level or entire declared multiplet.

        Groups are explicit; this does not detect resonances or introduce an
        energy threshold. Internal phases/rotations do not change the result.
        """
        selected = self.rotation[:, _columns(columns, len(self.rotation))]
        return selected @ selected.conj().T

    def transition_strength(self, initial_columns, final_columns, operator, *, density_factor=None):
        """Per-component multiplet strength, with positive density rho=F F†.

        None means rho=I on the initial cluster: a SUM over initial states,
        not a degeneracy average. To average uniformly supply I/sqrt(rank).
        Nonuniform populations/coherences must be supplied in the selected
        initial-column basis and transformed with that basis.
        """
        initial = _columns(initial_columns, len(self.rotation))
        final = _columns(final_columns, len(self.rotation))
        reduced = self.transform_property(operator)[..., final, :][..., :, initial]
        if density_factor is None:
            amplitudes = reduced
        else:
            factor = np.asarray(density_factor, dtype=complex)
            if factor.ndim != 2 or factor.shape[0] != len(initial) or not np.all(np.isfinite(factor)):
                raise ValueError("finite density factor with one row per initial column required")
            amplitudes = reduced @ factor
        # Guaranteed nonnegative, with no positivity eigensolve or clipping.
        return np.sum(abs(amplitudes)**2, axis=(-2, -1))

    def transform_property(self, operator):
        """Transform scalar/vector/tensor components, including non-Hermitian O."""
        o = np.asarray(operator, dtype=complex)
        n = len(self.rotation)
        if o.ndim < 2 or o.shape[-2:] != (n, n) or not np.all(np.isfinite(o)):
            raise ValueError("finite property components with trailing shape (n,n) required")
        return self.rotation.conj().T @ o @ self.rotation

    def complete_prescribed_energies(self, energies_cm):
        """Embed supplied energies unchanged; report mismatch with the pair model.

        This constructs U diag(E) U^dagger. It does NOT establish that these
        energies equal eigenvalues or diagonal elements of the pair model.
        """
        e = np.asarray(energies_cm, dtype=float)
        if e.shape != (len(self.rotation),) or not np.all(np.isfinite(e)):
            raise ValueError("one finite real prescribed energy per column required")
        completed = (self.rotation*e) @ self.rotation.conj().T
        defect = float(np.linalg.norm(completed-self.pair_hamiltonian_cm))
        return completed, {"pair_model_difference_frobenius_cm": defect,
                           "energies_modified": False}


def compose_hdc_pairs(diagonal_cm, couplings_cm, *, pair_shifts_cm=None,
                      pair_order=None, blocks=None, ansatz="collective_feedback",
                      maximum_states=64):
    """Analytical pair completion and two fixed pair-composition stages.

    All pairs within each declared fixed block are retained, including induced
    ones. Blocks are supplied symmetry/noninteracting sectors, never inferred
    from a numerical cutoff or changed along a parameter scan.
    pair_shifts_cm supplies finite nonnegative symmetric pair displacements s;
    effective off-diagonal moduli are sqrt(s*(abs(d)+s)). The original coupling
    supplies its phase. A positive shift for an exactly zero coupling is
    rejected because that phase would be unspecified.

    None selects the DC limit as a reference model, NOT an implementation of
    the original hybrid HDC switching function. A chosen scalar HDC energy
    prescription must supply its own pair shifts. No resonance thresholds,
    empirical widths, or molecular constants are introduced.

    ansatz='collective_feedback' adds a regularized environmental pair
    Hamiltonian. The first stage also accounts for a collective bright-channel
    norm, weighted by its DC mixing exposure. The residual stage uses current
    common residual entries in one fixed forward/backward gate sequence.
    ansatz='regularized_feedback' is an experimental alternative with
    phase-free collective regularization and quadratic bias cancellation.
    It resolves embedded pair-gate singularities but is not a global
    continuity certificate. Both have the same fixed gate count.
    ansatz='collective' retains the previous frozen residual palindrome with
    environmental dressing as a comparison. Both have the same gate count.
    No extra stage is triggered by accuracy or molecule identity.
    ansatz='bare' reproduces the original two-stage control approximation.

    Exact unresolved pair degeneracies are flagged, not certified continuous.
    The result is an approximate frame on a common support; no assertion of
    global continuity for embedded degenerate pairs is made.
    """
    if isinstance(maximum_states, bool) or not isinstance(maximum_states, int) or maximum_states < 1:
        raise ValueError("maximum_states must be a positive integer")
    if ansatz not in ("bare", "collective", "collective_feedback", "regularized_feedback"):
        raise ValueError("unknown pair-composition ansatz")
    raw_e = np.asarray(diagonal_cm)
    if np.iscomplexobj(raw_e) and np.any(raw_e.imag != 0):
        raise ValueError("real reference diagonal required")
    e = np.asarray(raw_e.real, dtype=float)
    if e.ndim != 1 or not len(e) or not np.all(np.isfinite(e)):
        raise ValueError("nonempty finite reference diagonal required")
    if len(e) > maximum_states:
        raise MemoryError("small-support research prototype state limit exceeded")
    v = _finite_hermitian(couplings_cm, "couplings_cm")
    if v.shape != (len(e), len(e)):
        raise ValueError("coupling matrix and diagonal dimensions differ")
    if np.any(v.diagonal() != 0):
        raise ValueError("coupling diagonal must be zero; supply it in diagonal_cm")

    labels = (0,)*len(e) if blocks is None else tuple(blocks)
    if len(labels) != len(e) or any(not isinstance(label, (str, int)) for label in labels):
        raise ValueError("one fixed integer/string block label per reference state required")
    for a, b in combinations(range(len(e)), 2):
        if labels[a] != labels[b] and v[a, b] != 0:
            raise ValueError("couplings between declared distinct blocks are forbidden")
    all_pairs = tuple((a, b) for a, b in combinations(range(len(e)), 2)
                      if labels[a] == labels[b])
    if pair_order is None:
        pairs = all_pairs
    else:
        pairs = tuple(tuple(p) for p in pair_order)
        if len(pairs) != len(all_pairs) or set(pairs) != set(all_pairs):
            raise ValueError("pair_order must contain each canonical pair exactly once")

    shifts = None
    if pair_shifts_cm is not None:
        raw = np.asarray(pair_shifts_cm)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("real pair shifts required")
        shifts = np.array(raw.real, dtype=float, copy=True)
        if (shifts.shape != v.shape or not np.all(np.isfinite(shifts))
                or np.any(shifts < 0) or np.any(shifts.diagonal() != 0)
                or not np.array_equal(shifts, shifts.T)):
            raise ValueError("finite nonnegative symmetric pair shifts with zero diagonal required")
        for a, b in combinations(range(len(e)), 2):
            if labels[a] != labels[b] and shifts[a, b] != 0:
                raise ValueError("pair shifts between declared distinct blocks are forbidden")

    h = np.diag(e).astype(complex)
    for a, b in all_pairs:
        gap = abs(float(e[b]-e[a]))
        if not np.isfinite(gap):
            raise FloatingPointError("reference gap exceeds floating-point range")
        g = abs(v[b, a])
        s = _dc_shift(gap, g) if shifts is None else shifts[b, a]
        if g == 0:
            if s != 0:
                raise ValueError("a positive shift at zero coupling has no specified phase")
            continue
        # In the DC test limit no round trip is needed: the completed coupling
        # is algebraically the input coupling, so preserve it bit for bit.
        modulus = g if shifts is None else np.sqrt(s)*np.sqrt(gap+s)
        if not np.isfinite(modulus):
            raise FloatingPointError("effective coupling exceeds floating-point range")
        h[b, a] = modulus*(v[b, a]/g)
        h[a, b] = h[b, a].conjugate()

    dressed = ansatz != "bare"
    feedback = ansatz in ("collective_feedback", "regularized_feedback")
    u1, unresolved1 = _palindrome(h, pairs, environment=dressed, collective=dressed)
    h1 = u1.conj().T @ h @ u1
    h1 = .5*h1+.5*h1.conj().T
    if feedback:
        u2, unresolved2 = _residual_feedback(h1, pairs, regularized=ansatz == "regularized_feedback")
    else:
        u2, unresolved2 = _palindrome(h1, pairs, environment=dressed)
    u = u1 @ u2
    residual = u.conj().T @ h @ u
    diagnostics = {
        "method": "fixed_two_stage_pair_composition",
        "ansatz": ansatz,
        "environment_dressing": dressed,
        "collective_dc_exposure_first_stage": dressed,
        "residual_feedback": feedback,
        "residual_collective_regularization": ansatz == "regularized_feedback",
        "residual_regularization_bias_order": 3 if ansatz == "regularized_feedback" else None,
        "stages": 2,
        "elementary_pair_rotation_count": 4*len(pairs),
        "state_count": len(e),
        "pair_count": len(pairs),
        "pair_order": pairs,
        "blocks": labels,
        "pair_shift_source": "dc_reference_limit" if shifts is None else "supplied_pair_displacements",
        "spectral_solver_used": False,
        "convergence_loop_used": False,
        "production_qualified": False,
        "global_continuity": "not_certified",
        "unresolved_pair_frames_first_stage": unresolved1,
        "unresolved_pair_frames_residual_stage": unresolved2,
        "unitarity_frobenius": float(np.linalg.norm(u.conj().T @ u-np.eye(len(e)))),
        "off_diagonal_input_cm": _off_diagonal_norm(h),
        "off_diagonal_after_first_stage_cm": _off_diagonal_norm(h1),
        "off_diagonal_after_residual_stage_cm": _off_diagonal_norm(residual),
    }
    return PairCompositionResult(h, u1, h1, u2, u, diagnostics)
