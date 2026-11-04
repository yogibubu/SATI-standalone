"""Complete scalar first/second jet pullback between internal charts."""
import numpy as np


def pullback_scalar_jet(gradient, hessian, coordinate_first, coordinate_second):
    """Transport all field components, retaining gradient times curvature.

    coordinate_first[i,a] = d old_i / d new_a, and coordinate_second[i,a,b]
    is its second derivative. No active/bath or isotope selection occurs here.
    """
    g = np.asarray(gradient, float); H = np.asarray(hessian, float)
    J = np.asarray(coordinate_first, float); Q = np.asarray(coordinate_second, float)
    if g.ndim != 1 or H.shape != (len(g), len(g)) or J.ndim != 2 or J.shape[0] != len(g) or Q.shape != (len(g), J.shape[1], J.shape[1]) or not all(np.all(np.isfinite(v)) for v in (g, H, J, Q)):
        raise ValueError('Compatible finite complete scalar and coordinate jets required')
    if not np.allclose(H, H.T, atol=1e-9) or not np.allclose(Q, Q.swapaxes(1, 2), atol=1e-9):
        raise ValueError('Symmetric Hessian and coordinate curvature required')
    return dict(gradient=J.T@g, hessian=J.T@H@J+np.einsum('i,iab->ab', g, Q))
