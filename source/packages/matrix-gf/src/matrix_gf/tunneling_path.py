"""Covariant reaction-path geometry in a nonredundant ESALC-I chart.

Potential derivatives use hartree; G is the contravariant metric in inverse
amu, with derivative indices last. Arc length is bohr*sqrt(amu). No Cartesian
reconstruction or differentiation of independently phased eigenvectors occurs.
"""
from dataclasses import dataclass
import numpy as np
from .harmonic import HESSIAN_EIGENVALUE_TO_CM
from .reaction_path import sonic_transverse_modes


def _array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or np.iscomplexobj(a) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name}: finite real tensor of shape {shape} required')
    return np.array(a, dtype=float, copy=True)


def internal_connection(metric_inverse_amu, metric_first_inverse_amu):
    """Levi-Civita Gamma[a,b,c] from G[i,j,k] = dG^ij/dq^k."""
    raw = np.asarray(metric_inverse_amu)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or len(raw) < 2:
        raise ValueError('nonredundant metric with at least two coordinates required')
    n = len(raw)
    G = _array(raw, (n,n), 'metric')
    dG = _array(metric_first_inverse_amu, (n,n,n), 'metric first derivative')
    if (not np.allclose(G, G.T, rtol=1e-11, atol=1e-13)
            or not np.allclose(dG, dG.swapaxes(0,1), rtol=1e-11, atol=1e-13)
            or np.min(np.linalg.eigvalsh(G)) <= 0):
        raise ValueError('symmetric positive metric and symmetric derivative required')
    g = np.linalg.inv(G)
    dg = -np.einsum('ai,ijk,jb->abk', g, dG, g)
    gamma = .5*np.einsum('ad,dbc->abc', G,
        dg + dg.swapaxes(1,2) - dg.transpose(2,0,1))
    return G, g, gamma


@dataclass(frozen=True)
class InternalPathGeometry:
    tangent: np.ndarray
    curvature: np.ndarray
    curvature_components: np.ndarray
    transverse_modes: np.ndarray
    frequencies_cm: np.ndarray
    curvature_norm: float
    gradient_transverse_fraction: float
    source: str
    transverse_mode_definition: str = 'curvilinear_surface'


def internal_path_geometry(gradient_hartree, hessian_hartree, metric_inverse_amu,
                           metric_first_inverse_amu, *, tangent=None,
                           cubic_hartree=None, path_tolerance=1e-6,
                           transverse_mode_definition='curvilinear_surface'):
    """Covariant IRC curvature; at a stationary saddle use the cubic TS limit.

    Away from the saddle k = P G H_cov t / (dV.t). At the saddle, in an
    orthonormal GF frame, k_j = -C_cov[j,r,r]/(lambda_j-2*lambda_r).
    The latter uses the SAME potential cubic and G1 as saddle VPT2/QNF.
    Potential Hessian/cubic inputs are ordinary partial derivatives, not
    already covariant tensors; applying the connection twice is forbidden.

    Transverse vibrations default to straight displacements in the supplied
    internal chart on the local surface t.T g delta_q = 0. Its physical image
    is curved: project the ordinary partial Hessian, including its coordinate
    curvature/gradient term, without subtracting Gamma.dV. The optional
    'covariant' modes project H_cov and retain the earlier Cartesian-equivalent
    definition. Physical IRC curvature ALWAYS uses H_cov for either choice.
    """
    if (not isinstance(transverse_mode_definition, str)
            or transverse_mode_definition not in {'curvilinear_surface', 'covariant'}):
        raise ValueError('transverse_mode_definition must be curvilinear_surface or covariant')
    if not np.isfinite(path_tolerance) or not 0 < path_tolerance < 1:
        raise ValueError('positive path tolerance below one required')
    G,g,gamma = internal_connection(metric_inverse_amu, metric_first_inverse_amu)
    n = len(G)
    dV = _array(gradient_hartree, (n,), 'gradient')
    H = _array(hessian_hartree, (n,n), 'Hessian')
    if not np.allclose(H,H.T,rtol=1e-11,atol=1e-13):
        raise ValueError('symmetric Hessian required')
    Hcov = H - np.einsum('kij,k->ij', gamma, dV)
    norm = float(np.sqrt(dV@G@dV))
    if norm == 0:
        C = _array(cubic_hartree, (n,n,n), 'stationary saddle cubic')
        if not (np.allclose(C,C.swapaxes(0,1)) and np.allclose(C,C.swapaxes(1,2))):
            raise ValueError('symmetric potential cubic required')
        ev, U = np.linalg.eigh(G)
        root = (U*np.sqrt(ev))@U.T
        lam, z = np.linalg.eigh(root@H@root)
        if np.count_nonzero(lam < 0) != 1 or np.any(lam == 0):
            raise ValueError('stationary path point must be an index-one saddle')
        B = root@z
        t = B[:,0]
        if tangent is not None:
            supplied = _array(tangent,(n,), 'tangent')
            supplied /= np.sqrt(supplied@g@supplied)
            if abs(supplied@g@t) < 1-path_tolerance:
                raise ValueError('saddle tangent must be the unique unstable GF mode')
            t *= np.sign(supplied@g@t)
        covC = (C - np.einsum('lij,lk->ijk',gamma,H)
                  - np.einsum('lik,jl->ijk',gamma,H)
                  - np.einsum('ljk,il->ijk',gamma,H))
        projected = np.einsum('ijk,ia,j,k->a',covC,B[:,1:],t,t)
        components = -projected/(lam[1:]-2*lam[0])
        modes = B[:,1:]
        k = modes@components
        frequencies = np.sqrt(lam[1:])*HESSIAN_EIGENVALUE_TO_CM
        transverse_fraction = 0.
        source = 'covariant_cubic_saddle_limit'
    else:
        t = G@dV/norm if tangent is None else _array(tangent,(n,), 'tangent')
        t /= np.sqrt(t@g@t)
        transverse_fraction = float(np.sqrt(max(0.,1-(dV@t/norm)**2)))
        if transverse_fraction > path_tolerance:
            raise ValueError('path gradient is not tangent to the declared IRC')
        slope = float(dV@t)
        k = G@Hcov@t/slope
        k -= t*(t@g@k)
        force = H if transverse_mode_definition == 'curvilinear_surface' else Hcov
        normal = sonic_transverse_modes(force,G,t,gradient=dV)
        if np.any(normal.frequencies_cm <= 0):
            raise ValueError('SCT requires stable path-transverse modes')
        modes = normal.sonic_modes
        frequencies = normal.frequencies_cm
        components = modes.T@g@k
        source = 'covariant_irc_gradient_hessian'
    return InternalPathGeometry(t,k,components,modes,frequencies,
        float(np.sqrt(k@g@k)),transverse_fraction,source,transverse_mode_definition)
