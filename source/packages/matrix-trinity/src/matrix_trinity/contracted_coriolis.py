"""Contract the modal GF mixed metric in the TRINITY product-state basis."""
from __future__ import annotations

import numpy as np


def dress_lam_harmonic_bath_rotation(
    frequencies_cm, metric_first_cm, active_modes, band_energies_cm,
    coordinate_elements, momentum_elements, coordinate_second_moments,
):
    """Dress the leading mixed LAM/bath rotational block with band states.

    The linear metric defines L_a = sum_sb (a_sba q_b p_s +
    b_sba q_s p_b), in T=P.G.P/2 normal coordinates. Bath modes remain
    harmonic ground states. Inputs q[n,s], p[n,s] are <n|q_s,p_s|0>
    in a common phase gauge; <q_s q_t> is evaluated before truncating
    intermediate band states. Energies and metric coefficients are cm^-1.

    Return the paired kinematic completion and virtual-state resolvent,
    their harmonic counterparts, and their difference. Only this paired
    difference may replace the same block of a full molecular VPT2 result.
    This is a leading-linear-metric model, not a full anharmonic bath or
    a determination of a deperturbed A from ground-state spectra alone.
    """
    w = np.asarray(frequencies_cm, float)
    g1 = np.asarray(metric_first_cm, float)
    active = tuple(active_modes)
    energy = np.asarray(band_energies_cm, float)
    q = np.asarray(coordinate_elements, complex)
    p = np.asarray(momentum_elements, complex)
    qq = np.asarray(coordinate_second_moments, float)
    n = len(w)
    if (w.ndim != 1 or np.any(w <= 0) or not active
            or any(isinstance(i, bool) or int(i) != i or not 0 <= i < n for i in active)
            or len(set(active)) != len(active)
            or g1.shape != (n + 3, n + 3, n)
            or energy.ndim != 1 or not len(energy)
            or np.any(energy < energy[0])
            or q.shape != (len(energy), len(active)) or p.shape != q.shape
            or qq.shape != (len(active), len(active))
            or not all(np.all(np.isfinite(a)) for a in (w, g1, energy, q, p, qq))):
        raise ValueError("compatible stable modal metric and phase-consistent band matrix elements required")
    if (not np.allclose(g1, g1.swapaxes(0, 1), rtol=1e-10, atol=1e-12)
            or not np.allclose(qq, qq.T, rtol=1e-10, atol=1e-12)
            or np.min(np.linalg.eigvalsh(qq)) < -1e-10):
        raise ValueError("symmetric metric and positive coordinate second moments required")
    active = tuple(map(int, active))
    bath = tuple(i for i in range(n) if i not in active)
    direct = np.zeros((3, 3))
    virtual = np.zeros((3, 3))
    harmonic_direct = np.zeros((3, 3))
    harmonic_virtual = np.zeros((3, 3))
    gaps = energy - energy[0]
    for b in bath:
        a = np.asarray([g1[s, n:, b] for s in active])
        c = np.asarray([g1[b, n:, s] for s in active])
        for k, s in enumerate(active):
            kinetic = np.outer(a[k], a[k]) / (4 * w[s])
            direct += kinetic
            harmonic_direct += kinetic + np.outer(c[k], c[k]) / (4 * w[b])
            harmonic_virtual -= np.outer(a[k] + c[k], a[k] + c[k]) / (4 * (w[s] + w[b]))
        direct += c.T @ qq @ c / (2 * w[b])
        amplitude = (p @ a + 1j * q @ c) / np.sqrt(2)
        virtual -= np.real(amplitude.conj().T @ (amplitude / (gaps + w[b])[:, None]))
    dressed = direct + virtual
    harmonic = harmonic_direct + harmonic_virtual
    return dict(
        dressed_tensor_cm=dressed,
        harmonic_counterterm_tensor_cm=harmonic,
        replacement_tensor_cm=dressed - harmonic,
        direct_kinematic_tensor_cm=direct,
        virtual_tensor_cm=virtual,
        harmonic_direct_tensor_cm=harmonic_direct,
        harmonic_virtual_tensor_cm=harmonic_virtual,
        active_modes=active,
        bath_modes=bath,
        definition="leading_linear_mixed_metric_with_paired_kinematic_completion",
    )


def contract_modal_coriolis(states, metric_first_cm, *, maximum_bytes=512*1024**2):
    """Return coefficients C[a,left,right] of C_a J_a in cm^-1.

    The producer metric uses T=1/2 P.G.P and dimensionless normal modes.
    Thus C_a=sum_ij G'[i,n+a,j] {p_i,q_j}/2. Matrix elements
    are evaluated analytically BEFORE projecting, including i=j; projecting
    q and p separately would lose intermediate states at a basis boundary.
    Axes and mode phases are those of SonicRovibrationalConstants.
    This is the linear mixed metric, not a replacement for higher derivatives.
    """
    occupations = np.asarray(states)
    if (occupations.ndim != 2 or not len(occupations) or not np.all(np.isfinite(occupations))
            or np.any(occupations < 0) or np.any(occupations != np.floor(occupations))
            or len({tuple(s) for s in occupations}) != len(occupations)):
        raise ValueError("unique nonnegative integer product states required")
    occupations = occupations.astype(int)
    n = occupations.shape[1]
    metric = np.asarray(metric_first_cm, float)
    if (metric.shape != (n + 3, n + 3, n) or not np.all(np.isfinite(metric))
            or not np.allclose(metric, metric.swapaxes(0, 1), rtol=1e-12, atol=1e-12)):
        raise ValueError("complete symmetric nonlinear-rotor metric derivative required")
    if 3*len(states)**2*np.dtype(complex).itemsize > maximum_bytes:
        raise MemoryError('contracted Coriolis matrices exceed memory budget')
    result = np.zeros((3, len(states), len(states)), complex)
    lookup = {tuple(state): index for index, state in enumerate(occupations)}
    pairs = np.argwhere(np.any(metric[:n, n:, :] != 0, axis=1))
    # Apply only nonzero ladder transitions, rather than testing every pair
    # of product states and every pair of modes.
    for right, ket in enumerate(occupations):
        for i, j in pairs:
            coefficients = metric[i, n:, j]
            steps = ((2, 0), (-2, 0)) if i == j else ((1, 1), (1, -1), (-1, 1), (-1, -1))
            for di, dj in steps:
                bra = ket.copy()
                bra[i] += di
                bra[j] += dj
                left = lookup.get(tuple(bra))
                if left is None:
                    continue
                if i == j:
                    # {p,q}/2 = i(a^dagger² - a²)/2.
                    v = (0.5j*np.sqrt((ket[i]+1)*(ket[i]+2)) if di == 2
                         else -0.5j*np.sqrt(ket[i]*(ket[i]-1)))
                else:
                    p = 1j*np.sqrt((ket[i]+1)/2) if di == 1 else -1j*np.sqrt(ket[i]/2)
                    q = np.sqrt((ket[j]+1)/2) if dj == 1 else np.sqrt(ket[j]/2)
                    v = p*q
                result[:, left, right] += coefficients*v
    return result
