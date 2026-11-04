"""Variable quadratic-bath fibers and paired LAM rotational self energies.

All tensors use T=P.G.P/2, dimensionless canonical coordinates and cm^-1.
This is an adiabatic ground/one-phonon rotational model, not a full bath
anharmonic Hamiltonian. Kinetic transitions to omitted even bath sectors
are not diagonalized. The fiber derivative metric is retained in the
projected kinetic energy; changing gaps alone is not this model.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _root(matrix):
    values, vectors = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(values)) or np.min(values) <= 0:
        raise ValueError("quadratic bath matrices must be finite positive definite")
    return (vectors * np.sqrt(values)[..., None, :]) @ vectors.swapaxes(-1, -2)


def _root_derivative(root, direction):
    values, vectors = np.linalg.eigh(root)
    local = vectors.swapaxes(-1, -2) @ direction @ vectors
    return vectors @ (local / (values[..., :, None] + values[..., None, :])) @ vectors.swapaxes(-1, -2)


def quadratic_bath_fiber(kinetic, force, kinetic_derivatives, force_derivatives):
    """Normalized Gaussian ground/one-quantum fibers and derivative metrics.

    Arrays G,H have shape (grid,bath,bath); derivatives have shape
    (grid,LAM,bath,bath). B defines chi_i=sqrt(2)*(B*y)_i*chi_0 and
    chi_0 proportional to exp(-y.T*B.T*B*y/2). The one-quantum energy
    matrix is Omega=sqrt(sqrt(G)*H*sqrt(G)). No eigenvector tracking or
    mode-by-mode reassignment occurs along the path.
    """
    g, h = np.asarray(kinetic, float), np.asarray(force, float)
    dg, dh = np.asarray(kinetic_derivatives, float), np.asarray(force_derivatives, float)
    if (g.ndim != 3 or h.shape != g.shape or g.shape[1] != g.shape[2]
            or dg.ndim != 4 or dh.shape != dg.shape
            or dg.shape[:1] + dg.shape[2:] != g.shape):
        raise ValueError("incompatible quadratic bath field shapes")
    for value in (g, h, dg, dh):
        if not np.all(np.isfinite(value)) or not np.allclose(value, value.swapaxes(-1, -2), atol=1e-9, rtol=1e-10):
            raise ValueError("finite symmetric quadratic bath fields required")
    gh = _root(g)
    gi = np.linalg.inv(gh)
    omega = _root(gh @ h @ gh)
    oh = _root(omega)
    b = oh @ gi
    bi = np.linalg.inv(b)
    connection, squeezing = [], []
    for s in range(dg.shape[1]):
        dgh = _root_derivative(gh, dg[:, s])
        dgi = -gi @ dgh @ gi
        domega = _root_derivative(omega, dgh @ h @ gh + gh @ dh[:, s] @ gh + gh @ h @ dgh)
        doh = _root_derivative(oh, domega)
        db = doh @ gi + oh @ dgi
        k = db @ bi
        connection.append((k.swapaxes(-1, -2) - k) / 2)
        squeezing.append(k + k.swapaxes(-1, -2))
    connection = np.stack(connection, axis=1)
    squeezing = np.stack(squeezing, axis=1)
    ngrid, nb, _ = g.shape
    dim = dg.shape[1]
    e0 = np.empty((ngrid, dim, dim))
    e1 = np.empty((ngrid, dim, dim, nb, nb))
    for s in range(dim):
        for t in range(dim):
            e0[:, s, t] = np.einsum("gij,gji->g", squeezing[:, s], squeezing[:, t]) / 8
            e1[:, s, t] = (
                -connection[:, s] @ connection[:, t]
                + e0[:, s, t, None, None] * np.eye(nb)
                + squeezing[:, t] @ squeezing[:, s] / 4
            )
    return dict(B=b, inverse_B=bi, frequency_matrix_cm=omega,
                coordinate_covariance=bi @ bi.swapaxes(-1, -2) / 2,
                connection=connection, squeezing=squeezing,
                ground_derivative_metric=e0, one_quantum_derivative_metric=e1)


def harmonic_mixed_rotation_counterterm(active_kinetic, active_force, bath_kinetic,
                                       bath_force, a, c_derivative):
    """Harmonic counterpart of the same completed-square mixed vertex.

    a[s,b,axis] multiplies y_b*p_s; c_derivative[s,b,axis]
    multiplies q_s*p_b. General positive, non-diagonal oscillator metrics
    are reduced algebraically, without changing any molecular force field.
    """
    def normal(g, h):
        gh = _root(np.asarray(g, float))
        w2, u = np.linalg.eigh(gh @ np.asarray(h, float) @ gh)
        if np.min(w2) <= 0:
            raise ValueError("stable harmonic counterterm required")
        w = np.sqrt(w2)
        q = gh @ (u / np.sqrt(w)[None, :])
        p = np.linalg.inv(gh) @ (u * np.sqrt(w)[None, :])
        return w, q, p
    wa, qa, pa = normal(active_kinetic, active_force)
    wb, qb, pb = normal(bath_kinetic, bath_force)
    aa = np.einsum("sn,sbr,bm->nmr", pa, a, qb)
    cc = np.einsum("sn,sbr,bm->nmr", qa, c_derivative, pb)
    direct = (np.einsum("nma,nmb,n->ab", aa, aa, 1 / wa)
              + np.einsum("nma,nmb,m->ab", cc, cc, 1 / wb)) / 4
    virtual = -np.einsum("nma,nmb,nm->ab", aa + cc, aa + cc,
                         1 / (wa[:, None] + wb[None, :])) / 4
    return dict(direct_tensor_cm=direct, virtual_tensor_cm=virtual,
                paired_tensor_cm=direct + virtual,
                active_frequencies_cm=wa, bath_frequencies_cm=wb)


def dress_lam_variable_quadratic_bath_rotation(
    energies_cm, grid_vectors, grid_momenta, active_kinetic_cm,
    bath_kinetic_cm, bath_force_cm, bath_kinetic_derivatives,
    bath_force_derivatives, coupling_a_cm, coupling_a_derivatives_cm,
    coupling_c_cm, *, maximum_bytes=1024**3, lam_state=0,
):
    """Contract the variable adiabatic one-phonon resolvent and its completion.

    U(grid,LAM) is orthonormal in the flat DVR measure; pU(grid,LAMaxis,LAM)
    is the *unprojected* momentum action. a(grid,s,b,axis) and c(grid,b,axis)
    define the Hermitian mixed vertex sum_s {a_s*y,p_s}/2+c*p_b.
    da has shape (grid,derivative_axis,vertex_axis,bath,rotation_axis).
    The direct term is its matched positive kinetic completion. Return
    the paired tensor, never a prescription to add the virtual part alone.
    lam_state selects an isolated dressed LAM band below the one-phonon
    threshold. Default zero preserves the ground-band contraction. The
    historical ground_* output keys refer to the selected zero-bath fiber.
    The caller subtracts the same model's harmonic counterpart from VPT2.
    """
    e = np.asarray(energies_cm, float)
    u = np.asarray(grid_vectors, complex)
    pu = np.asarray(grid_momenta, complex)
    ga = np.asarray(active_kinetic_cm, float)
    gb = np.asarray(bath_kinetic_cm, float)
    hb = np.asarray(bath_force_cm, float)
    a = np.asarray(coupling_a_cm, float)
    da = np.asarray(coupling_a_derivatives_cm, float)
    c = np.asarray(coupling_c_cm, float)
    if u.ndim != 2 or e.shape != (u.shape[1],) or pu.ndim != 3:
        raise ValueError("compatible retained LAM vectors and unprojected momenta required")
    ng, nr = u.shape
    if not isinstance(lam_state, (int, np.integer)) or not 0 <= lam_state < nr:
        raise ValueError('retained isolated LAM state required')
    dim, nb = pu.shape[1], gb.shape[1]
    if (pu.shape != (ng, dim, nr) or ga.shape != (ng, dim, dim)
            or gb.shape != (ng, nb, nb) or hb.shape != gb.shape
            or a.shape != (ng, dim, nb, 3) or c.shape != (ng, nb, 3)
            or da.shape != (ng, dim, dim, nb, 3)
            or not all(np.all(np.isfinite(x)) for x in (e, u, pu, ga, gb, hb, a, da, c))
            or not np.allclose(u.conj().T @ u, np.eye(nr), atol=1e-8, rtol=1e-8)):
        raise ValueError("incompatible finite adiabatic rotational fields")
    if 6 * (nr * nb)**2 * np.dtype(complex).itemsize > maximum_bytes:
        raise MemoryError("one-phonon projected resolvent exceeds memory budget")
    if np.min(np.linalg.eigvalsh(ga)) <= 0:
        raise ValueError("positive active kinetic metric required")
    fiber = quadratic_bath_fiber(gb, hb, bath_kinetic_derivatives, bath_force_derivatives)
    d = fiber['connection']
    phi0 = np.einsum("gst,gst->g", ga, fiber['ground_derivative_metric']) / 2
    h0 = np.diag(e).astype(complex) + u.conj().T @ (phi0[:, None] * u)
    ev, vectors = np.linalg.eigh(h0)
    ground = vectors[:, lam_state]
    psi, ppsi = u @ ground, np.einsum("gsn,n->gs", pu, ground)

    def project(left, field, right):
        result = np.empty((nr, nb, nr, nb), complex)
        for i in range(nb):
            for j in range(nb):
                result[:, i, :, j] = left.conj().T @ (field[:, i, j, None] * right)
        return result.reshape(nr * nb, nr * nb)

    potential = fiber['frequency_matrix_cm'] + np.einsum(
        "gst,gstij->gij", ga, fiber['one_quantum_derivative_metric']) / 2
    h1 = np.kron(np.diag(e), np.eye(nb)).astype(complex) + project(u, potential, u)
    for s in range(dim):
        field = np.einsum("gt,gtij->gij", ga[:, s], d)
        first = project(pu[:, s], field, u)
        h1 += -.5j * (first - first.conj().T)
    hermitian_error = float(np.max(abs(h1 - h1.conj().T)))
    if hermitian_error > 1e-7:
        raise ValueError("projected one-phonon Hamiltonian is not Hermitian")
    h1 = (h1 + h1.conj().T) / 2
    q = fiber['inverse_B'] / np.sqrt(2)
    aa = np.einsum("gbk,gsba->gska", q, a)
    vertex = np.einsum("gska,gs->gka", aa, ppsi)
    vertex += 1j / np.sqrt(2) * np.einsum("gkb,gba,g->gka", fiber['B'], c, psi)
    for s in range(dim):
        connection_vertex = fiber['squeezing'][:, s] @ aa[:, s]
        connection_vertex -= np.einsum("gbk,gba->gka", q, da[:, s, s])
        vertex += .5j * connection_vertex * psi[:, None, None]
    w = np.einsum("gn,gba->nba", u.conj(), vertex).reshape(nr * nb, 3)
    resolvent = h1 - ev[lam_state] * np.eye(nr * nb)
    minimum_gap = float(np.linalg.eigvalsh(resolvent)[0])
    if minimum_gap <= 0:
        raise ValueError("one-phonon resolvent crosses the ground state")
    virtual = -np.real(w.conj().T @ cho_solve(cho_factor(resolvent, lower=True), w))
    direct_field = np.einsum(
        "gst,gsba,gbc,gtcd->gad", np.linalg.inv(ga), a,
        fiber['coordinate_covariance'], a, optimize=True) / 2
    direct_field += np.einsum("gba,gbc,gcd->gad", c, np.linalg.inv(gb), c) / 2
    direct = np.einsum("g,gab->ab", abs(psi)**2, direct_field)
    return dict(paired_tensor_cm=direct + virtual, direct_tensor_cm=direct,
                virtual_tensor_cm=virtual, ground_fiber_energy_shift_cm=float(ev[lam_state]-e[lam_state]),
                ground_fiber_scalar_potential_cm=phi0,
                ground_lam_coefficients=ground,
                minimum_one_phonon_gap_cm=minimum_gap,
                one_phonon_Hermiticity_error_cm=hermitian_error,
                retained_lam_states=nr, bath_dimension=nb, lam_state=int(lam_state),
                model='adiabatic_quadratic_bath_paired_rotational_resolvent',
                limitations='Quadratic local bath and leading-in-bath mixed vertex. Even-bath-sector nonadiabatic kinetic transitions are not variationally included. This is not a full anharmonic bath Hamiltonian.')


def dress_lam_fixed_quadratic_bath_rotation(
    energies_cm, grid_vectors, grid_momenta, active_kinetic_cm,
    bath_kinetic_cm, bath_force_cm, coupling_a_cm, coupling_c_cm,
    *, lam_state=0,
):
    """Exact fixed-coefficient specialization of the variable-bath resolvent.

    The bath normal modes diagonalize its constant one-phonon Hamiltonian.
    Only c(grid,bath,rotation) varies. Momenta must be unprojected actions
    in the same physical coordinates as a and the active kinetic tensor.
    The matched positive completion is retained along with the virtual term.
    """
    e = np.asarray(energies_cm, float)
    u = np.asarray(grid_vectors, complex)
    pu = np.asarray(grid_momenta, complex)
    ga, gb, hb = (np.asarray(x, float) for x in
                  (active_kinetic_cm, bath_kinetic_cm, bath_force_cm))
    a, c = np.asarray(coupling_a_cm, float), np.asarray(coupling_c_cm, float)
    if u.ndim != 2 or pu.ndim != 3 or gb.ndim != 2:
        raise ValueError('Grid vectors, physical momentum actions and square bath tensors required')
    ng, nr = u.shape
    dim, nb = pu.shape[1], gb.shape[0]
    if (e.shape != (nr,) or pu.shape != (ng, dim, nr)
            or ga.shape != (dim, dim) or gb.shape != (nb, nb)
            or hb.shape != gb.shape or a.shape != (dim, nb, 3)
            or c.shape != (ng, nb, 3)
            or not isinstance(lam_state, (int, np.integer)) or not 0 <= lam_state < nr
            or not all(np.all(np.isfinite(x)) for x in (e, u, pu, ga, gb, hb, a, c))
            or not np.allclose(u.conj().T @ u, np.eye(nr), atol=1e-8, rtol=1e-8)):
        raise ValueError('Incompatible finite fixed-bath rotational fields')
    for tensor in (ga, gb, hb):
        if not np.allclose(tensor, tensor.T, atol=1e-10) or np.min(np.linalg.eigvalsh(tensor)) <= 0:
            raise ValueError('Positive symmetric quadratic kinetic and force tensors required')
    gh = _root(gb)
    w2, modes = np.linalg.eigh(gh @ hb @ gh)
    wb = np.sqrt(w2)
    qb = gh @ (modes / np.sqrt(wb)[None, :])
    pb = np.linalg.inv(gh) @ (modes * np.sqrt(wb)[None, :])
    aa = np.einsum('sbr,bm->smr', a, qb) / np.sqrt(2)
    cc = np.einsum('gbr,bm->gmr', c, pb) / np.sqrt(2)
    vertex = np.einsum('smr,gs->gmr', aa, pu[:, :, lam_state])
    vertex += 1j * cc * u[:, lam_state, None, None]
    w = np.einsum('gn,gmr->nmr', u.conj(), vertex, optimize=True)
    gaps = e[:, None] - e[lam_state] + wb[None, :]
    if np.min(gaps) <= 0:
        raise ValueError('One-phonon resolvent crosses the selected LAM state')
    virtual = -np.real(np.einsum('nmr,nms,nm->rs', w.conj(), w, 1/gaps))
    direct = np.einsum('st,sbr,bc,tcd->rd', np.linalg.inv(ga), a,
                       qb @ qb.T / 2, a, optimize=True) / 2
    direct_field = np.einsum('gbr,bc,gcs->grs', c, np.linalg.inv(gb), c, optimize=True) / 2
    direct += np.einsum('g,grs->rs', abs(u[:, lam_state])**2, direct_field)
    return dict(paired_tensor_cm=direct+virtual, direct_tensor_cm=direct,
                virtual_tensor_cm=virtual, minimum_one_phonon_gap_cm=float(np.min(gaps)),
                retained_lam_states=nr, bath_dimension=nb, lam_state=int(lam_state),
                ground_fiber_energy_shift_cm=0.,
                model='fixed_quadratic_bath_paired_rotational_resolvent',
                limitations='Constant quadratic bath and leading mixed vertex; no variable-fiber derivative metric or higher bath anharmonicity.')


__all__ = ['quadratic_bath_fiber', 'harmonic_mixed_rotation_counterterm',
           'dress_lam_variable_quadratic_bath_rotation',
           'dress_lam_fixed_quadratic_bath_rotation']
