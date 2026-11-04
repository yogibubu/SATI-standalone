"""Harmonic restriction to a declared common internal-coordinate subspace."""
import hashlib
import numpy as np
from .vibronic_harmonic import ESALCHarmonicState


def project_esalc_harmonic_state(state, basis, *, reference_esalc=None):
    """Restrict force and covariant mass metrics to orthonormal chart columns.

    The inverse metric is not obtained by projecting G directly. The induced
    mass metric is B.T @ inverse(G) @ B, and is inverted only after restriction.
    Reference coordinates may be supplied after a caller-owned nonlinear path
    alignment. No geometry, force or displacement is inferred here.
    """
    b=np.asarray(basis,float);n=len(state.reference_esalc)
    if (b.ndim!=2 or b.shape[0]!=n or not 0<b.shape[1]<=n
            or not np.all(np.isfinite(b))
            or not np.allclose(b.T@b,np.eye(b.shape[1]),rtol=0,atol=1e-10)):
        raise ValueError('Finite orthonormal chart columns required')
    reference=np.asarray(state.reference_esalc if reference_esalc is None else reference_esalc,float)
    gradient=np.zeros(n) if state.gradient_hartree is None else np.asarray(state.gradient_hartree,float)
    f=np.asarray(state.force_constants_hartree,float);g=np.asarray(state.metric_inverse_amu,float)
    if (reference.shape!=(n,) or gradient.shape!=(n,) or f.shape!=(n,n) or g.shape!=(n,n)
            or not all(np.all(np.isfinite(x)) for x in (reference,gradient,f,g))
            or not np.allclose(f,f.T,atol=1e-12,rtol=1e-12)
            or not np.allclose(g,g.T,atol=1e-12,rtol=1e-12)):
        raise ValueError('Finite matching symmetric harmonic field required')
    np.linalg.cholesky(g)
    mass=b.T@np.linalg.solve(g,b)
    metric=np.linalg.solve(mass,np.eye(len(mass)))
    force=b.T@f@b
    fingerprint=hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()[:16]
    return ESALCHarmonicState((force+force.T)/2,(metric+metric.T)/2,
        b.T@reference,state.coordinate_identity+'-subspace-'+fingerprint,
        b.T@gradient,state.label)
