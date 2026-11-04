"""Product-rule derivatives of native LAM surface operators and contractions."""

from __future__ import annotations
import numpy as np
from .lam_rotation import (
    _apply_momentum,
    _apply_lam_hamiltonian,
    solve_lam_basis,
    _rotational_operators,
    centrifugal_distortion_operators,
    _project_lam_values,
)
from .lam_fit import _lam_direction_action
from .lam_errors import LAMTruncationError

FIELDS = (
    "potential_MHz",
    "F_MHz",
    "rho",
    "R_MHz",
    "geometric_potential_MHz",
    "lam_quartic_kinetic_MHz",
    "p2_rotational_metric_MHz",
    "p_rotational_cubic_MHz",
    "p_rotational_anticommutator_MHz",
    "pi2_rotation_MHz",
    "quartic_distortion_MHz",
    "sextic_distortion_MHz",
)


def complete_surface_direction(surface, direction):
    result = {}
    for name in FIELDS:
        base = getattr(surface, name)
        delta = direction.get(name)
        if base is None:
            result[name] = None if delta is None else np.asarray(delta, float)
        else:
            result[name] = np.zeros_like(base) if delta is None else np.asarray(delta, float)
            if result[name].shape != np.shape(base):
                raise ValueError(f"surface derivative shape mismatch: {name}")
        if result[name] is not None and not np.all(np.isfinite(result[name])):
            raise ValueError(f"nonfinite surface derivative: {name}")
    result["spin_tensor_surfaces_MHz"] = {
        k: np.asarray(direction.get("spin_tensor_surfaces_MHz", {}).get(k, np.zeros_like(v)))
        for k, v in (surface.spin_tensor_surfaces_MHz or {}).items()
    }
    if any(np.any(v) for v in result["spin_tensor_surfaces_MHz"].values()):
        raise NotImplementedError(
            "structural LAM response does not include an electronic/spin Hamiltonian"
        )
    return result


def prepare_surface_response_workspace(surface, solved, *, extended=None):
    """Prepare point-local, read-only actions shared by all derivative columns.

    Strong references identify the exact surface and eigenbasis. This cache is
    confined to one derivative evaluation; changed coefficients or basis must
    construct a new workspace. No approximate cache key is used.
    """
    u = np.asarray(solved.basis.vectors, complex)
    p = lambda axis: (solved.basis.momentum_operators[axis] @ u
        if solved.basis.momentum_operators else _apply_momentum(surface, axis, u))
    potential = (np.asarray(surface.potential_MHz) +
        (0 if surface.geometric_potential_MHz is None else surface.geometric_potential_MHz)).reshape(-1)
    action = _apply_lam_hamiltonian(surface, u, potential=potential)
    momentum = tuple(p(axis) for axis in range(surface.dimension))
    for value in (action, *momentum):
        value.setflags(write=False)
    result = dict(surface=surface, basis=solved.basis, base_action=action, momentum=momentum)
    if extended is not None:
        result['extended'] = extended
    return result


def lam_span_response(surface, solved, direction, *, response_states=48, workspace=None):
    """Normal derivative of the retained LAM projector in a fixed primitive space."""
    direction = complete_surface_direction(surface, direction)
    u = np.asarray(solved.basis.vectors, complex)
    store = {} if workspace is None else workspace
    if isinstance(response_states,bool) or not isinstance(response_states,(int,np.integer)) or response_states<0:
        raise ValueError('nonnegative integer discarded response-state budget required')
    signature=(surface,solved.basis,int(response_states))
    if 'response_identity' in store:
        old=store['response_identity']
        if old[0] is not surface or old[1] is not solved.basis or old[2]!=response_states:
            raise ValueError('LAM response workspace belongs to a different physical point or state budget')
    store['response_identity']=signature
    action = _lam_direction_action(surface, direction, u)
    du = np.zeros_like(u)
    size = solved.basis.primitive_size or len(u)
    count = min(size, u.shape[1] + int(response_states))
    if count > u.shape[1] and np.any(action):
        if "extended" not in store:
            store["extended"] = solve_lam_basis(
                surface,
                state_count=count,
                primitive_basis=solved.basis.primitive_representation,
                fourier_cutoffs=solved.basis.fourier_cutoffs,
            )
        extended = store["extended"]
        q = extended.vectors[:, u.shape[1] :]
        e = extended.energies_MHz[u.shape[1] :]
        gap = solved.basis.energies_MHz[None, :] - e[:, None]
        resolution = max(
            1.0e-10,
            128 * np.spacing(max(1.0, float(np.max(abs(extended.energies_MHz), initial=0.0)))),
        )
        if np.any(abs(gap) < resolution):
            raise LAMTruncationError(
                "lam_states", "structural response cuts a degenerate LAM manifold"
            )
        du = q @ ((q.conj().T @ action) / gap)
    return du


def native_surface_response(
    surface,
    solved,
    direction,
    *,
    response_states=48,
    manifest=None,
    parameters=None,
    operator_direction=None,
    workspace=None,
    basis_tangent=None,
):
    """Differentiate the moving LAM span and all native rotation operator words.

    Only one discarded-space solve is shared across parameter directions at a
    given physical point. No displaced rovibrational eigensystem is evaluated.
    Internal rotations inside the retained space use the parallel-transport
    gauge; their pure basis rotations do not change physical eigenvalues.
    """
    direction = complete_surface_direction(surface, direction)
    store = {} if workspace is None else workspace
    if 'base_action' not in store:
        store.update(prepare_surface_response_workspace(surface, solved, extended=store.get('extended')))
    if store.get('surface') is not surface or store.get('basis') is not solved.basis:
        raise ValueError('surface response workspace belongs to different coefficients or eigenbasis')
    u = np.asarray(solved.basis.vectors, complex)
    if basis_tangent is None:
        du = lam_span_response(surface, solved, direction, response_states=response_states, workspace=store)
    else:
        du = np.asarray(basis_tangent, complex)
        if du.shape != u.shape or not np.all(np.isfinite(du)):
            raise ValueError('finite basis tangent matching the LAM span required')
        if np.max(abs(u.conj().T@du+du.conj().T@u),initial=0.) > 1e-9:
            raise ValueError('basis tangent must preserve orthonormality')
    action = _lam_direction_action(surface, direction, u)
    p = lambda axis, v: (
        solved.basis.momentum_operators[axis] @ v
        if solved.basis.momentum_operators
        else _apply_momentum(surface, axis, v)
    )
    fixed_span = not np.any(du)
    def project(au, dau):
        if fixed_span:
            return _project_lam_values(solved.basis, dau,
                preserve_sectors=surface.symmetry_sector_count > 1)
        return du.conj().T @ au + u.conj().T @ dau
    field = lambda f, df: project(
        np.asarray(f).reshape(-1, 1) * u,
        np.asarray(df).reshape(-1, 1) * u + np.asarray(f).reshape(-1, 1) * du,
    )
    n = len(u)
    d = surface.dimension
    f = np.asarray(surface.F_MHz).reshape(n, d, d)
    df = direction["F_MHz"].reshape(n, d, d)
    rho = np.asarray(surface.rho).reshape(n, 3, d)
    dr = direction["rho"].reshape(n, 3, d)
    r = np.asarray(surface.R_MHz).reshape(n, 3, 3)
    dR = direction["R_MHz"].reshape(n, 3, 3)
    au = store['base_action']
    dau = (
        _apply_lam_hamiltonian(
            surface,
            du,
            potential=(
                np.asarray(surface.potential_MHz)
                + (
                    0
                    if surface.geometric_potential_MHz is None
                    else surface.geometric_potential_MHz
                )
            ).reshape(-1),
        )
        + action
    )
    h0 = project(au, dau)
    linear = []
    quadratic = {}
    pu = store['momentum']
    pdu = [p(i, du) for i in range(d)]

    def sym_field_momentum(fv, dfv, axis, power=1):
        fv = np.asarray(fv).reshape(-1, 1)
        dfv = np.asarray(dfv).reshape(-1, 1)

        def power_action(a):
            for _ in range(power):
                a = p(axis, a)
            return a

        a = 0.5 * (power_action(fv * u) + fv * power_action(u))
        da = 0.5 * (power_action(dfv * u + fv * du) + dfv * power_action(u) + fv * power_action(du))
        return project(a, da)

    for a in range(3):
        au = np.zeros_like(u)
        dau = np.zeros_like(u)
        for mu in range(d):
            for nu in range(d):
                fv = f[:, mu, nu] * rho[:, a, nu]
                dv = df[:, mu, nu] * rho[:, a, nu] + f[:, mu, nu] * dr[:, a, nu]
                gu = f[:, mu, nu] * rho[:, a, mu]
                dg = df[:, mu, nu] * rho[:, a, mu] + f[:, mu, nu] * dr[:, a, mu]
                au -= p(mu, fv[:, None] * u) + gu[:, None] * pu[nu]
                dau -= (
                    p(mu, dv[:, None] * u + fv[:, None] * du)
                    + dg[:, None] * pu[nu]
                    + gu[:, None] * pdu[nu]
                )
        linear.append(project(au, dau))
        for b in range(3):
            fv = r[:, a, b] + np.einsum("ni,nij,nj->n", rho[:, a], f, rho[:, b])
            dv = (
                dR[:, a, b]
                + np.einsum("ni,nij,nj->n", dr[:, a], f, rho[:, b])
                + np.einsum("ni,nij,nj->n", rho[:, a], df, rho[:, b])
                + np.einsum("ni,nij,nj->n", rho[:, a], f, dr[:, b])
            )
            quadratic[a, b] = field(fv, dv)
    # All optional polynomial coefficient surfaces use their declared ordering.
    distortion = []
    for name, length in (("quartic_distortion_MHz", 5), ("sextic_distortion_MHz", 7)):
        base = getattr(surface, name)
        delta = direction[name]
        if base is None and delta is None:
            distortion.extend([None] * length)
        else:
            base = np.zeros(surface.grid_shape + (length,)) if base is None else base
            delta = np.zeros_like(base) if delta is None else delta
            distortion.extend(field(base[..., k], delta[..., k]) for k in range(length))
    extras = {}
    for name, power in (
        ("p2_rotational_metric_MHz", 2),
        ("p_rotational_cubic_MHz", 1),
        ("p_rotational_anticommutator_MHz", 1),
    ):
        base = getattr(surface, name)
        delta = direction[name]
        if base is None and delta is None:
            continue
        base = np.zeros_like(delta) if base is None else base
        delta = np.zeros_like(base) if delta is None else delta
        suffix = np.shape(base)[surface.dimension :]
        extras[name] = {
            idx: sym_field_momentum(base[(...,) + idx], delta[(...,) + idx], idx[0], power)
            for idx in np.ndindex(suffix)
        }
    pi = None
    if surface.pi2_rotation_MHz is not None or direction["pi2_rotation_MHz"] is not None:
        from .pi2_rotation import project_covariant_momentum_matrices

        basepi = project_covariant_momentum_matrices(u, rho, p)
        dp2 = np.array([project(p(mu, pu[mu]), p(mu, pdu[mu])) for mu in range(d)])
        dpr = np.array(
            [
                [2 * sym_field_momentum(rho[:, a, mu], dr[:, a, mu], mu) for a in range(3)]
                for mu in range(d)
            ]
        )
        drr = np.array(
            [
                [
                    [
                        field(
                            rho[:, a, mu] * rho[:, b, mu],
                            dr[:, a, mu] * rho[:, b, mu] + rho[:, a, mu] * dr[:, b, mu],
                        )
                        for b in range(3)
                    ]
                    for a in range(3)
                ]
                for mu in range(d)
            ]
        )
        pi = (basepi, (dp2, dpr, drr))
    result = {}
    for J in solved.blocks:
        rot = _rotational_operators(J, solved.representation)
        rot2 = sum(a @ a for a in rot)
        sym = lambda a, b: 0.5 * (rot[a] @ rot[b] + rot[b] @ rot[a])
        factors = (
            [(h0, np.eye(2 * J + 1))]
            + [(linear[a], rot[a]) for a in range(3)]
            + [(quadratic[a, b], sym(a, b)) for a in range(3) for b in range(3)]
        )
        ops = centrifugal_distortion_operators(
            J, surface.distortion_reduction, solved.representation
        )
        factors.extend(
            (v, o) for v, o in zip(distortion, ops[: len(distortion)], strict=True) if v is not None
        )
        for name, items in extras.items():
            for idx, mat in items.items():
                if name == "p2_rotational_metric_MHz":
                    operator = sym(idx[1], idx[2])
                elif name == "p_rotational_cubic_MHz":
                    operator = rot[idx[1]] @ rot2
                else:
                    operator = 0.5 * (
                        rot[idx[1]] @ rot[idx[1]] @ rot[idx[2]]
                        + rot[idx[2]] @ rot[idx[1]] @ rot[idx[1]]
                    )
                factors.append((mat, operator))
        if pi is not None:
            from .pi2_rotation import pi2_rotation_factors

            families, _ = pi2_rotation_factors(*pi[0], rot, np.ones(3))
            derivatives, _ = pi2_rotation_factors(*pi[1], rot, np.ones(3))
            coefficients = (
                np.zeros(3) if surface.pi2_rotation_MHz is None else surface.pi2_rotation_MHz
            )
            dc = (
                np.zeros(3)
                if direction["pi2_rotation_MHz"] is None
                else direction["pi2_rotation_MHz"]
            )
            for k in range(3):
                factors.extend((coefficients[k] * c * a, b) for c, a, b in derivatives[k])
                factors.extend((dc[k] * c * a, b) for c, a, b in families[k])
        if manifest is not None:
            from .general_lam_operator import (
                normalize_general_lam_operator_manifest,
                _fourier_field,
            )

            model = normalize_general_lam_operator_manifest(manifest)
            params = model.parameters_with(parameters)
            for term in model.terms:
                value = u
                dv = du
                for axis in reversed(term.momentum_axes):
                    value = p(axis, value)
                    dv = p(axis, dv)
                fv = _fourier_field(surface, term)[:, None]
                matrix = u.conj().T @ (fv * value)
                dm = project(fv * value, fv * dv)
                coefficient = term.prefactor * params[term.parameter]
                delta = term.prefactor * (operator_direction or {}).get(term.parameter, 0.0)
                mat = coefficient * dm + delta * matrix
                operator = np.eye(2 * J + 1, dtype=complex)
                for axis in term.rotation_word:
                    operator = operator @ rot[axis]
                factors.extend(((0.5 * mat, operator), (0.5 * mat.conj().T, operator.conj().T)))
        result[J] = tuple(factors)
    return result, du
