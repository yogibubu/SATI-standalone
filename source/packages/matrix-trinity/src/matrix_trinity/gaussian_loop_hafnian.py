"""Optional loop-hafnian reference for the existing Bargmann amplitudes.

The Walrus remains an optional dependency, imported only on explicit use.
No coupling threshold or dense occupation tensor is introduced.
"""
import numpy as np
from scipy.special import gammaln


def loop_hafnian_coefficients(matrix, linear, log_ground, states):
    from thewalrus import loop_hafnian
    m, r, states = np.asarray(matrix), np.asarray(linear), np.asarray(states)
    if (m.shape != (len(r), len(r)) or not np.all(np.isfinite(m))
            or not np.all(np.isfinite(r)) or not np.isfinite(log_ground)
            or not np.allclose(m, m.T, rtol=1e-12, atol=1e-12)
            or states.ndim != 2 or states.shape[1] != len(r)
            or states.dtype.kind not in 'iu' or np.any(states < 0)):
        raise ValueError('finite symmetric Gaussian and nonnegative occupation rows required')
    output = np.empty(len(states), complex)
    for i, row in enumerate(states):
        active = np.flatnonzero(row)
        # Exact zero occupations are removed; weak nonzero couplings remain.
        value = loop_hafnian(m[np.ix_(active, active)], D=r[active], reps=row[active])
        output[i] = value*np.exp(log_ground-.5*gammaln(row+1).sum())
    return output


def loop_hafnian_operator_amplitudes(matrix, linear, log_ground, constants, insertions, states):
    """Evaluate (c+Lz)G with exact shared neighboring occupation requests."""
    states = np.asarray(states); c, h = np.asarray(constants), np.asarray(insertions)
    if (c.ndim != 1 or h.shape != (len(c), len(linear))
            or not np.all(np.isfinite(c)) or not np.all(np.isfinite(h))
            or states.ndim != 2 or states.shape[1] != len(linear)
            or states.dtype.kind not in 'iu' or np.any(states < 0)):
        raise ValueError('matching linear operator bank required')
    requested = {tuple(row) for row in states}
    for row in states:
        for k in np.flatnonzero(row * np.any(h != 0, axis=0)):
            lower = row.copy(); lower[k] -= 1; requested.add(tuple(lower))
    rows = np.array(sorted(requested), dtype=int).reshape(-1, len(linear))
    values = loop_hafnian_coefficients(matrix, linear, log_ground, rows)
    lookup = {tuple(row): value for row, value in zip(rows, values)}
    amplitudes = np.empty((len(states), len(c)), complex)
    for i, row in enumerate(states):
        amplitudes[i] = c*lookup[tuple(row)]
        for k in np.flatnonzero(row * np.any(h != 0, axis=0)):
            lower = row.copy(); lower[k] -= 1
            amplitudes[i] += h[:, k]*np.sqrt(row[k])*lookup[tuple(lower)]
    return amplitudes
