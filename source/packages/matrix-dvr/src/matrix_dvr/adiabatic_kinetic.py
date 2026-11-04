"""Leading adiabatic kinetic partition with nonorthogonal fast coordinates.

Completing the fast momentum square leaves the Schur-complement slow metric.
The fast oscillator uses the original G_ff block, not the inverse covariant
fast block. Those agree only when slow and fast velocities are orthogonal.
This leading reduction does not supply derivative or nonadiabatic terms.
"""
import numpy as np


def adiabatic_kinetic_partition(kinetic_cm, fast_indices):
    """Partition a positive canonical kinetic matrix, retaining every cross term.

    Any unselected rows, including rotational momenta, are kept in original
    order. Returned matrices have the same kinetic units as the input.
    """
    g = np.asarray(kinetic_cm, float)
    raw = np.asarray(fast_indices)
    if (g.ndim < 2 or g.shape[-1] != g.shape[-2] or raw.ndim != 1
            or not len(raw) or np.any(raw != np.rint(raw))
            or len(set(raw.tolist())) != len(raw) or np.any(raw < 0)
            or np.any(raw >= g.shape[-1]) or len(raw) >= g.shape[-1]
            or not np.all(np.isfinite(g)) or not np.allclose(g, g.swapaxes(-1, -2))
            or np.min(np.linalg.eigvalsh(g)) <= 0):
        raise ValueError('Positive complete kinetic metric and a proper unique fast subset required')
    f = raw.astype(int)
    r = np.array([i for i in range(g.shape[-1]) if i not in f])
    ff = g[..., f[:, None], f[None, :]]
    rf = g[..., r[:, None], f[None, :]]
    rr = g[..., r[:, None], r[None, :]]
    reduced = rr-rf@np.linalg.solve(ff, rf.swapaxes(-1, -2))
    return dict(fast_indices=f, retained_indices=r, fast_kinetic_cm=ff,
                retained_kinetic_cm=(reduced+reduced.swapaxes(-1, -2))/2)


def adiabatic_fast_spectrum(kinetic_cm, fast_hessian_cm, fast_indices):
    """Fast harmonic spectrum of the same completed-square kinetic partition."""
    part = adiabatic_kinetic_partition(kinetic_cm, fast_indices)
    h = np.asarray(fast_hessian_cm, float)
    g = part['fast_kinetic_cm']
    if (h.shape != g.shape or not np.all(np.isfinite(h))
            or not np.allclose(h, h.swapaxes(-1, -2))
            or np.min(np.linalg.eigvalsh(h)) <= 0):
        raise ValueError('Matching positive fast Hessian in kinetic-compatible units required')
    values, vectors = np.linalg.eigh(g)
    root = (vectors*np.sqrt(values)[..., None, :])@vectors.swapaxes(-1, -2)
    force = root@h@root
    frequencies = np.sqrt(np.linalg.eigvalsh((force+force.swapaxes(-1, -2))/2))
    return dict(**part, frequencies_cm=frequencies,
                zero_point_cm=frequencies.sum(axis=-1)/2)
