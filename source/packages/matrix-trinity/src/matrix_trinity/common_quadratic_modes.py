"""Joint two-surface quadratic frame in an identified SMITH coordinate chart.

The potential objective is a weighted Frobenius off-diagonal norm in a
reference mass chart. A Jacobi sweep minimizes each plane exactly; deterministic
starts retain the best stationary solution, without claiming a global minimum.
Both state metrics, gradients and expansion points are transported exactly.
No polar projection of a Cartesian Duschinsky matrix is involved.
"""
from dataclasses import dataclass, replace
import hashlib

import numpy as np

from matrix_gf.harmonic import CM_PER_HARTREE, HESSIAN_EIGENVALUE_TO_CM, solve_wilson_gf
from .vibronic_harmonic import ESALCHarmonicState


@dataclass(frozen=True)
class CommonQuadraticFrame:
    """s = origin + transform @ x; states are expressed in the same x chart."""
    transform: np.ndarray
    origin: np.ndarray
    states: tuple[ESALCHarmonicState, ...]
    objective_history: tuple[float, ...]
    converged: bool
    relative_offdiagonal_norms: tuple[float, ...]


def _offnorm(matrices):
    off = matrices.copy()
    indices = np.arange(matrices.shape[-1])
    off[:, indices, indices] = 0
    return float(np.sum(off * off))


def joint_quadratic_modes(states, *, weights=None, reference_metric=None,
                         origin=None, tolerance=1e-12, maximum_sweeps=100):
    """Minimize sum_s weight_s ||offdiag(L.T F_s L)||_F^2.

    L = sqrt(G_reference) U, U orthogonal. Weights are explicit, nonnegative
    surface priorities, normalized to sum one; default equal. They do not
    normalize individual Hessians. Initial identity, every state eigenframe,
    and the mean eigenframe are tried. Unequal G_s are retained, never frozen.
    Inputs must already share a nonredundant coordinate chart and units.
    """
    states = tuple(states)
    if len(states) < 2 or any(not isinstance(s, ESALCHarmonicState) for s in states):
        raise ValueError('at least two ESALCHarmonicState inputs required')
    for state in states:
        state.modes()  # Reuse the existing F/G/stability validation owner.
    n = len(states[0].reference_esalc)
    if any(s.coordinate_identity != states[0].coordinate_identity
           or len(s.reference_esalc) != n for s in states):
        raise ValueError('transport states to the same identified chart first')
    if (not np.isfinite(tolerance) or tolerance <= 0
            or not isinstance(maximum_sweeps, int) or maximum_sweeps < 1):
        raise ValueError('positive tolerance and integer sweep budget required')
    w = np.ones(len(states)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(states),) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('finite nonnegative surface weights with positive sum required')
    w = w / w.sum()
    g = np.asarray(states[0].metric_inverse_amu if reference_metric is None else reference_metric, float)
    if g.shape != (n, n) or not np.all(np.isfinite(g)) or not np.allclose(g, g.T, rtol=1e-12, atol=1e-14):
        raise ValueError('finite symmetric reference metric required')
    eigen, vectors = np.linalg.eigh(g)
    if np.any(eigen <= 0):
        raise ValueError('positive definite reference metric required')
    root = (vectors * np.sqrt(eigen)) @ vectors.T
    matrices = np.array([root @ s.force_constants_hartree @ root for s in states])
    weighted = matrices * np.sqrt(w)[:, None, None]
    scale = float(np.sum(weighted * weighted))
    starts = [np.eye(n)] + [np.linalg.eigh(m)[1] for m in matrices]
    starts.append(np.linalg.eigh(np.einsum('s,sij->ij', w, matrices))[1])
    best = None
    for start in starts:
        rotation = start.copy()
        a = rotation.T @ weighted @ rotation
        history = [_offnorm(a)]
        converged = False
        for _ in range(maximum_sweeps):
            for p in range(n):
                for q in range(p + 1, n):
                    # offdiag after rotation = b cos(2t) + d sin(2t).
                    b = a[:, p, q]
                    d = (a[:, q, q] - a[:, p, p]) / 2
                    plane = np.array([[b @ b, b @ d], [b @ d, d @ d]])
                    _, v = np.linalg.eigh(plane)
                    angle = .5 * np.arctan2(v[1, 0], v[0, 0])
                    # Equivalent minimizer nearest identity avoids mode swaps.
                    angle = (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4
                    c, s = np.cos(angle), np.sin(angle)
                    r = np.array([[c, -s], [s, c]])
                    pair = [p, q]
                    a[:, :, pair] = a[:, :, pair] @ r
                    a[:, pair, :] = r.T @ a[:, pair, :]
                    rotation[:, pair] = rotation[:, pair] @ r
            value = _offnorm(a)
            history.append(value)
            if history[-2] - value <= tolerance * scale:
                converged = True
                break
        candidate = (history[-1], rotation, tuple(history), converged)
        if best is None or candidate[0] < best[0]:
            best = candidate
    _, rotation, history, converged = best
    transform = root @ rotation
    inverse = np.linalg.solve(transform, np.eye(n))
    center = np.array(states[0].reference_esalc if origin is None else origin, float)
    if center.shape != (n,) or not np.all(np.isfinite(center)):
        raise ValueError('finite common origin required')
    identity = hashlib.sha256(states[0].coordinate_identity.encode()
        + transform.astype('<f8').tobytes() + center.astype('<f8').tobytes()).hexdigest()
    transported = tuple(replace(s,
        force_constants_hartree=transform.T @ s.force_constants_hartree @ transform,
        metric_inverse_amu=inverse @ s.metric_inverse_amu @ inverse.T,
        reference_esalc=inverse @ (s.reference_esalc - center),
        gradient_hartree=None if s.gradient_hartree is None else transform.T @ s.gradient_hartree,
        coordinate_identity=identity) for s in states)
    residuals = tuple(np.sqrt(_offnorm(np.array([s.force_constants_hartree])))
                      / np.linalg.norm(s.force_constants_hartree) for s in transported)
    return CommonQuadraticFrame(transform, center, transported, history, converged, residuals)


def common_quadratic_coefficients(state, reference_frequencies_cm):
    """Shared dimensionless q-q, p-p, linear and scalar terms, in cm-1.

    The state already uses the optimized common mass chart. Coordinate length
    is x_i = length_i q_i. Off-diagonal kinetic terms are retained.
    """
    state.modes()
    omega = np.asarray(reference_frequencies_cm, float)
    n = len(state.reference_esalc)
    if omega.shape != (n,) or np.any(omega <= 0) or not np.all(np.isfinite(omega)):
        raise ValueError('positive common reference frequencies required')
    c = HESSIAN_EIGENVALUE_TO_CM / np.sqrt(CM_PER_HARTREE)
    f = np.asarray(state.force_constants_hartree)
    g = np.zeros(n) if state.gradient_hartree is None else np.asarray(state.gradient_hartree)
    displacement = np.asarray(state.reference_esalc)
    length = c / np.sqrt(omega)
    kq = f * length[:, None] * length * CM_PER_HARTREE
    kp = state.metric_inverse_amu * np.sqrt(omega[:, None] * omega)
    linear = length * (g - f @ displacement) * CM_PER_HARTREE
    constant = float((.5 * displacement @ f @ displacement - g @ displacement) * CM_PER_HARTREE)
    return kq, kp, linear, constant, length


def common_quadratic_model(state, occupations, reference_frequencies_cm, *,
                           maximum_states=512, **selection):
    """Project one transported surface onto a shared finite HO basis.

    Both electronic states must use identical occupations and reference omega.
    Retains displacement/gradient linear terms, scalar potential and both q-q
    and p-p residuals. Existing local_quadratic owns resonance selection, exact
    polyads and external PT2. No states outside the supplied basis are implied;
    its truncation must be qualified. Anharmonic terms are not added here.
    """
    from .curvilinear_vpt2 import _apply
    from .local_quadratic import build_local_quadratic_model

    state.modes()
    omega = np.asarray(reference_frequencies_cm, float)
    n = len(state.reference_esalc)
    basis = tuple(tuple(row) for row in occupations)
    if (omega.shape != (n,) or not np.all(np.isfinite(omega)) or np.any(omega <= 0)
            or not basis or len(basis) > maximum_states or len(set(basis)) != len(basis)
            or any(len(row) != n or any(not isinstance(x, (int, np.integer)) or x < 0 for x in row) for row in basis)):
        raise ValueError('positive shared frequencies and distinct bounded HO occupations required')
    kq, kp, linear, constant, length = common_quadratic_coefficients(state, omega)
    diagonal = np.full(len(basis), constant)
    index = {row: i for i, row in enumerate(basis)}
    edges = {}
    terms = [(linear[j], (('q', j),)) for j in range(n)]
    terms += [(.5 * matrix[j, k], ((kind, j), (kind, k)))
              for kind, matrix in (('q', kq), ('p', kp)) for j in range(n) for k in range(n)]
    for i, row in enumerate(basis):
        for coefficient, operators in terms:
            if coefficient == 0:
                continue
            for target, amplitude in _apply(row, operators).items():
                j = index.get(target)
                value = float((coefficient * amplitude).real)
                if j == i:
                    diagonal[i] += value
                elif j is not None and j > i:
                    edges[i, j] = edges.get((i, j), 0.) + value
    return build_local_quadratic_model(diagonal, {}, edges, **selection)


def common_chart_states_from_harmonic(model, *, coordinate_identity):
    """Recover a shared initial mass chart from an existing harmonic model.

    This is an exact affine representation of the supplied local model, not
    an internal-coordinate construction. An invertible, nonorthogonal mode
    map is retained exactly: the final metric is J^-1 J^-T, never replaced by
    identity. The electronic gap and transition properties remain owned by
    the input model. Physical Cartesian/internal chart transport must precede
    its construction; this adapter does not align geometries or fit a polar J.
    """
    from .vibronic_harmonic import HarmonicVibronicModel
    if not isinstance(model, HarmonicVibronicModel):
        raise TypeError('validated HarmonicVibronicModel required')
    if not isinstance(coordinate_identity, str) or not coordinate_identity.strip():
        raise ValueError('explicit initial mass-chart identity required')
    wi, wf = model.initial_frequencies_cm, model.final_frequencies_cm
    j = model.mode_map
    inverse = np.linalg.solve(j, np.eye(len(wi)))
    c = HESSIAN_EIGENVALUE_TO_CM / np.sqrt(CM_PER_HARTREE)
    initial = ESALCHarmonicState(np.diag((wi/HESSIAN_EIGENVALUE_TO_CM)**2),
        np.eye(len(wi)), np.zeros(len(wi)), coordinate_identity, label='initial')
    final = ESALCHarmonicState(j.T @ np.diag((wf/HESSIAN_EIGENVALUE_TO_CM)**2) @ j,
        inverse @ inverse.T, -c * inverse @ (model.final_shift_dimensionless/np.sqrt(wf)),
        coordinate_identity, label='final')
    return initial, final


def qualify_quadratic_frequency_partition(kq_cm,kp_cm,*,
        mixing_thresholds=(.01,.025,.05,.1),steps=(1/16,1/32),
        maximum_frequency_error_cm=1.,maximum_step_change_cm=.05):
    """Frequency-only check of variational/PT2 common-mode quadratic pairs.

    Each diagonal one-mode q/p squeeze is treated exactly. Exchange and pair
    vertices rank off-diagonal pairs; selected pairs remain exact in the base
    quadratic Hamiltonian. The others are scaled by lambda and their frequency
    response is expanded to second order, using two declared central steps.
    The full supplied quadratic frequencies independently qualify the result.
    This is not a wavefunction or optical-intensity PT2 qualification.
    """
    from scipy.optimize import linear_sum_assignment
    q=np.asarray(kq_cm,float);p=np.asarray(kp_cm,float)
    if (q.ndim!=2 or q.shape[0]!=q.shape[1] or not len(q) or p.shape!=q.shape
            or not all(np.all(np.isfinite(x)) and np.allclose(x,x.T,atol=1e-12,rtol=1e-12)
                       and np.min(np.linalg.eigvalsh(x))>0 for x in (q,p))):
        raise ValueError('positive complete symmetric quadratic q/p matrices required')
    if (len(steps)!=2 or not 0<steps[1]<steps[0]<1
            or not mixing_thresholds or any(not np.isfinite(x) or x<=0 for x in mixing_thresholds)
            or not np.isfinite(maximum_frequency_error_cm) or maximum_frequency_error_cm<=0
            or not np.isfinite(maximum_step_change_cm) or maximum_step_change_cm<=0):
        raise ValueError('positive frequency gates, mixing thresholds and two decreasing central steps required')
    omega=np.sqrt(np.diag(q)*np.diag(p));length=(np.diag(p)/np.diag(q))**.25
    pairs=[]
    for i in range(len(q)):
        for j in range(i+1,len(q)):
            gq=q[i,j]*length[i]*length[j];gp=p[i,j]/(length[i]*length[j])
            exchange=(gq+gp)/2;creation=(gq-gp)/2
            score=max(abs(exchange)/max(abs(omega[i]-omega[j]),1e-8),abs(creation)/(omega[i]+omega[j]))
            pairs.append(dict(modes_1based=[i+1,j+1],exchange_cm=float(exchange),pair_creation_cm=float(creation),mixing_score=float(score)))
    def spectrum(a,b):
        result=solve_wilson_gf(a,b,scale_to_cm=False)
        if np.any(result.frequencies_cm<=0):raise ValueError('unstable quadratic response path')
        return result.frequencies_cm,result.normal_modes
    full,_=spectrum(q,p);cases=[]
    for threshold in mixing_thresholds:
        var=[edge for edge in pairs if edge['mixing_score']>=threshold]
        q0=np.diag(np.diag(q));p0=np.diag(np.diag(p))
        for edge in var:
            i,j=(k-1 for k in edge['modes_1based'])
            q0[i,j]=q0[j,i]=q[i,j];p0[i,j]=p0[j,i]=p[i,j]
        row=dict(mixing_threshold=float(threshold),variational_pairs_1based=[x['modes_1based'] for x in var],
            variational_pair_count=len(var),perturbative_pair_count=len(pairs)-len(var))
        try:
            base,U=spectrum(q0,p0);approximations=[]
            def matched(lam):
                frequencies,vectors=spectrum(q0+lam*(q-q0),p0+lam*(p-p0))
                left,right=linear_sum_assignment(-abs(U.T@vectors)**2)
                order=np.empty(len(q),int);order[left]=right
                return frequencies[order]
            for h in steps:
                plus=matched(h);minus=matched(-h)
                approximations.append(base+(plus-minus)/(2*h)+(plus-2*base+minus)/(2*h*h))
            error=float(max(abs(np.sort(approximations[-1])-full)))
            change=float(max(abs(approximations[-1]-approximations[0])))
            zpe_error=float(.5*(approximations[-1].sum()-full.sum()))
            row.update(maximum_frequency_error_cm=error,maximum_central_step_change_cm=change,
                zero_point_energy_error_cm=zpe_error,
                frequencies_cm=np.sort(approximations[-1]).tolist(),
                passed=bool(error<=maximum_frequency_error_cm and abs(zpe_error)<=maximum_frequency_error_cm
                    and change<=maximum_step_change_cm))
        except ValueError as failure:
            row.update(passed=False,failure=str(failure))
        cases.append(row)
    accepted=[r for r in cases if r['passed']]
    selected=min(accepted,key=lambda r:(r['variational_pair_count'],r['maximum_frequency_error_cm'])) if accepted else None
    return dict(full_frequencies_cm=full.tolist(),pairs=pairs,cases=cases,selected=selected,
        maximum_frequency_error_tolerance_cm=maximum_frequency_error_cm,
        maximum_central_step_change_tolerance_cm=maximum_step_change_cm,
        perturbative_response_steps=list(steps),diagonal_squeezes_exact=True,
        scope='quadratic frequency-response diagnostic only; optical wavefunction dressing not qualified')
