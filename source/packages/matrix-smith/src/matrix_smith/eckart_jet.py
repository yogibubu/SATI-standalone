"""Mass-specific Eckart transport of a fixed Cartesian path and its jets.

The physical path is never re-solved in a different internal-coordinate chart.
This distinction matters for frozen Cartesian SOOP coordinate definitions.
"""
from __future__ import annotations

import numpy as np


def mass_eckart_cartesian_jet(coordinates_angstrom, reference_angstrom,
                              masses_amu, tangents, second_derivatives=None):
    """Transport Cartesian values and their first/second parameter derivatives.

    Coordinates/reference have shape (atom,3), tangents (atom,3,parameter)
    and optional second derivatives (atom,3,parameter,parameter). The returned
    geometry is centered at its mass center. The proper mass-weighted Kabsch
    rotation is differentiated through the linearized Eckart condition, without
    numerical differentiation or eigenvector tracking. No electronic force
    tensor or intrinsic path parameter is changed.
    """
    x = np.asarray(coordinates_angstrom, float)
    ref = np.asarray(reference_angstrom, float)
    m = np.asarray(masses_amu, float)
    dx = np.asarray(tangents, float)
    if (x.ndim != 2 or x.shape[1] != 3 or ref.shape != x.shape
            or m.shape != (len(x),) or np.any(m <= 0)
            or dx.ndim != 3 or dx.shape[:2] != x.shape
            or not all(np.all(np.isfinite(v)) for v in (x, ref, m, dx))):
        raise ValueError('finite matching Cartesian geometry, masses and tangents required')
    k = dx.shape[2]
    dd = None if second_derivatives is None else np.asarray(second_derivatives, float)
    if dd is not None and (dd.shape != x.shape + (k, k)
                           or not np.all(np.isfinite(dd))
                           or not np.allclose(dd, dd.swapaxes(-1, -2), atol=1e-9, rtol=1e-9)):
        raise ValueError('symmetric Cartesian second derivatives required')
    x = x - np.average(x, axis=0, weights=m)
    ref = ref - np.average(ref, axis=0, weights=m)
    dx = dx - np.einsum('i,ias->as', m, dx)[None, :, :] / m.sum()
    left, _, right = np.linalg.svd((x*m[:, None]).T @ ref)
    rotation = left @ np.diag([1., 1., np.linalg.det(left @ right)]) @ right
    z = x @ rotation
    v = np.einsum('ias,ab->ibs', dx, rotation)
    cross_inertia = np.einsum('i,ia,ia->', m, ref, z)*np.eye(3) - np.einsum('i,ia,ib->ab', m, z, ref)
    if np.linalg.cond(cross_inertia) > 1e12:
        raise ValueError('singular Eckart orientation derivative')
    rhs = np.stack([np.einsum('i,ia->a', m, np.cross(ref, v[:, :, s])) for s in range(k)], axis=1)
    omega = np.linalg.solve(cross_inertia, rhs)
    dz = np.stack([v[:, :, s] + np.cross(z, omega[:, s]) for s in range(k)], axis=2)
    dz2 = None
    if dd is not None:
        dd = dd - np.einsum('i,iast->ast', m, dd)[None, :, :, :] / m.sum()
        raw = np.einsum('iast,ab->ibst', dd, rotation)
        dz2 = np.empty_like(raw)
        for s in range(k):
            for t in range(s, k):
                value = (raw[:, :, s, t] + np.cross(v[:, :, s], omega[:, t])
                         + np.cross(v[:, :, t], omega[:, s])
                         + .5*(np.cross(np.cross(z, omega[:, t]), omega[:, s])
                               + np.cross(np.cross(z, omega[:, s]), omega[:, t])))
                rhs2 = np.einsum('i,ia->a', m, np.cross(ref, value))
                value += np.cross(z, np.linalg.solve(cross_inertia, rhs2))
                dz2[:, :, s, t] = dz2[:, :, t, s] = value
    return dict(coordinates_angstrom=z, tangents=dz, second_derivatives=dz2,
                rotation=rotation,
                eckart_residual=float(np.max(abs(np.einsum('i,ia->a', m, np.cross(ref, z))))))


def mass_eckart_first_grid(coordinates_angstrom, reference_angstrom, masses_amu, tangents):
    """Batched first-jet version of the same analytic Eckart transport."""
    x=np.asarray(coordinates_angstrom,float);ref=np.asarray(reference_angstrom,float)
    m=np.asarray(masses_amu,float);D=np.asarray(tangents,float)
    if (x.ndim!=3 or x.shape[1:]!=(len(m),3) or ref.shape!=x.shape[1:]
            or D.ndim!=4 or D.shape[:3]!=x.shape or np.any(m<=0)
            or not all(np.all(np.isfinite(a)) for a in (x,ref,m,D))):
        raise ValueError('Matching finite batched Cartesian first jets required')
    x=x-np.einsum('i,pia->pa',m,x)[:,None,:]/m.sum()
    ref=ref-np.average(ref,axis=0,weights=m)
    D=D-np.einsum('i,pias->pas',m,D)[:,None,:,:]/m.sum()
    U,_,V=np.linalg.svd(np.einsum('i,pia,ib->pab',m,x,ref))
    signs=np.broadcast_to(np.eye(3),(len(x),3,3)).copy()
    signs[:,2,2]=np.linalg.det(U@V);R=U@signs@V
    z=x@R;v=np.einsum('pias,pab->pibs',D,R)
    A=(np.einsum('i,ia,pia->p',m,ref,z)[:,None,None]*np.eye(3)
       -np.einsum('i,pia,ib->pab',m,z,ref))
    if np.any(np.linalg.cond(A)>1e12):
        raise ValueError('Singular Eckart orientation derivative in grid')
    rhs=np.einsum('i,pisa->pas',m,np.cross(ref[None,:,None,:],v.swapaxes(-1,-2)))
    omega=np.linalg.solve(A,rhs)
    dz=v+np.cross(z[:,:,None,:],omega.swapaxes(-1,-2)[:,None,:,:]).swapaxes(-1,-2)
    return dict(coordinates_angstrom=z,tangents=dz,rotation=R,
        eckart_residual=np.max(abs(np.einsum('i,pia->pa',m,np.cross(ref[None],z))),axis=-1))


__all__ = ['mass_eckart_cartesian_jet', 'mass_eckart_first_grid']
