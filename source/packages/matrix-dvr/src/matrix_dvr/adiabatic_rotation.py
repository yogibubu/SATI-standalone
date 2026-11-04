"""Mechanical rotational observable for a declared adiabatic approximation.

Completing the internal momentum square separates the locked rotational
inertia from the coordinate-frame connection. Averaging this observable does
not retain a variational virtual/Coriolis response. Reference perturbative
corrections may be retained once through a matched observable replacement.
"""
import numpy as np


def mechanical_rotation_tensor(complete_kinetic,internal_count):
    """Schur complement of all internal momenta, in the original units.

    For a complete Cartesian metric this equals the inverse locked inertia.
    Dropping the off-diagonal block without this subtraction would retain its
    gauge-dependent quadratic completion in the direct rotational tensor.
    """
    g=np.asarray(complete_kinetic,float);n=int(internal_count)
    if (n<1 or g.shape[-2:]!=(n+3,n+3) or not np.all(np.isfinite(g))
            or not np.allclose(g,g.swapaxes(-1,-2),atol=1e-10)):
        raise ValueError('Complete finite symmetric internal-plus-rotation metric required')
    vv=g[...,:n,:n];vr=g[...,:n,n:]
    return g[...,n:,n:]-np.einsum('...ia,...ib->...ab',vr,np.linalg.solve(vv,vr))


__all__=['mechanical_rotation_tensor']
