"""Local nonlinear pullback of a stationary rectilinear vibrational field.

The supplied forward map u(q) has identity tangent. Its second and third
jets come from the physical coordinate provider, never from spectroscopy.
Potential and the divergence-form kinetic operator are transformed together.
"""

from dataclasses import replace
from itertools import permutations
from math import factorial
import numpy as np
from matrix_gf.harmonic import CM_PER_HARTREE
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM
from .curvilinear_vpt2 import _validate_field


def _symmetric_inputs(tensor):
    order = tensor.ndim - 1
    return sum(
        tensor.transpose((0,) + tuple(i + 1 for i in p)) for p in permutations(range(order))
    ) / factorial(order)


def rechart_rectilinear_field(field, forward_second, forward_third):
    """Pull back through q(u), including the flat-measure quantum correction.

    Input coordinates q are mass-orthonormal, with G(0)=I and G'(0)=0.
    forward_second[i,j,k] and forward_third[i,j,k,l] differentiate u_i(q).
    Existing G'' (e.g. Coriolis terms) is retained. All force-constant arrays
    are derivatives, including their factorial conventions, in native units.
    Missing quartics in an acquired semidiagonal polynomial remain missing;
    generated curvature terms are not new electronic-structure observations.
    """
    f2, f3, f4, g0, g1, g2 = _validate_field(field)
    n = len(f2)
    b = np.asarray(forward_second, float)
    c = np.asarray(forward_third, float)
    if (
        b.shape != (n,) * 3
        or c.shape != (n,) * 4
        or not np.all(np.isfinite(b))
        or not np.all(np.isfinite(c))
    ):
        raise ValueError("finite second/third coordinate jets of matching dimension required")
    if not np.allclose(g0, np.eye(n), atol=1e-10, rtol=1e-10) or np.max(abs(g1)) > 1e-10:
        raise ValueError("source must be a mass-orthonormal rectilinear field with G prime zero")
    if not np.allclose(b, b.swapaxes(1, 2), atol=1e-9, rtol=1e-7):
        raise ValueError("second coordinate jet is not symmetric")
    c = _symmetric_inputs(c)
    a = -b
    chain = np.einsum("iaj,akl->ijkl", b, b, optimize=True)
    d = -c + chain + chain.transpose(0, 2, 1, 3) + chain.transpose(0, 3, 2, 1)
    d = _symmetric_inputs(d)
    t = np.einsum("ai,ajk->ijk", f2, a, optimize=True)
    new3 = f3 + t + t.transpose(1, 0, 2) + t.transpose(1, 2, 0)
    t = np.einsum("akl,aij->ijkl", f3, a, optimize=True)
    cubic_curvature = (
        t
        + t.transpose(0, 2, 1, 3)
        + t.transpose(0, 3, 2, 1)
        + t.transpose(2, 1, 0, 3)
        + t.transpose(3, 1, 2, 0)
        + t.transpose(2, 3, 0, 1)
    )
    t = np.einsum("ai,ajkl->ijkl", f2, d, optimize=True)
    third_curvature = (
        t + t.transpose(1, 0, 2, 3) + t.transpose(1, 2, 0, 3) + t.transpose(1, 2, 3, 0)
    )
    t = np.einsum("ab,aij,bkl->ijkl", f2, a, a, optimize=True)
    second_curvature = t + t.transpose(0, 2, 1, 3) + t.transpose(0, 3, 2, 1)
    new4 = f4 + cubic_curvature + third_curvature + second_curvature
    new1 = b + b.swapaxes(0, 1)
    t = np.einsum("iak,jal->ijkl", b, b, optimize=True)
    new2 = (
        g2
        + c
        + c.swapaxes(0, 1)
        + t
        + t.swapaxes(2, 3)
        + np.einsum("ijm,mkl->ijkl", new1, a, optimize=True)
    )
    # The coordinate Jacobian changes the density of the wavefunction.
    # l = d log(det(dq/du)); Vg = div(G l)/4 + l.G.l/8 at the origin.
    trace = np.einsum("iik->k", b)
    log_jac_first = -trace
    log_jac_second = (
        -np.einsum("iikl->kl", c)
        + np.einsum("ijk,jil->kl", b, b, optimize=True)
        + np.einsum("m,mkl->kl", trace, b, optimize=True)
    )
    quantum_scale = HESSIAN_EIGENVALUE_TO_CM**2 / CM_PER_HARTREE
    scalar = quantum_scale * (
        (np.einsum("iji,j->", new1, log_jac_first) + np.trace(log_jac_second)) / 4
        + np.dot(log_jac_first, log_jac_first) / 8
    )
    result = replace(
        field,
        cubic_sonic=new3,
        quartic_sonic=new4,
        metric_first_sonic=new1,
        metric_second_sonic=new2,
        pseudopotential_cm=field.pseudopotential_cm + float(scalar),
    )
    _validate_field(result)
    return result, dict(
        inverse_second=a,
        inverse_third=d,
        pseudopotential_change_cm=float(scalar),
        quartic_curvature=cubic_curvature + third_curvature + second_curvature,
    )


def rechart_property_derivatives(first, second, third, inverse_second, inverse_third):
    """Transform any multiplicative property with arbitrary leading axes.

    All derivatives must use the native source coordinates (not dimensionless
    oscillator coordinates). The returned leading property axes are unchanged.
    """
    p1, p2, p3 = map(np.asarray, (first, second, third))
    a, d = np.asarray(inverse_second, float), np.asarray(inverse_third, float)
    if p1.ndim < 1:
        raise ValueError("property derivatives require a trailing coordinate axis")
    n = p1.shape[-1]
    if (
        a.shape != (n,) * 3
        or d.shape != (n,) * 4
        or p2.shape != p1.shape[:-1] + (n,) * 2
        or p3.shape != p1.shape[:-1] + (n,) * 3
        or any(not np.all(np.isfinite(t)) for t in (p1, p2, p3, a, d))
    ):
        raise ValueError("finite property derivatives with common leading axes required")
    t = np.einsum("...ai,ajk->...ijk", p2, a, optimize=True)
    return (
        p1.copy(),
        p2 + np.einsum("...a,aij->...ij", p1, a, optimize=True),
        p3
        + t
        + np.swapaxes(t, -3, -2)
        + np.swapaxes(t, -3, -1)
        + np.einsum("...a,aijk->...ijk", p1, d, optimize=True),
    )
