"""Dual-level force fields in a shared coordinate tangent.

Only the quadratic potential is replaced. Higher potential derivatives,
kinetic metric derivatives and property derivatives keep their declared
low-level coordinate frame. Subsequent modalization transforms them together.
"""
from dataclasses import replace

import numpy as np

from .curvilinear_vpt2 import SonicVPT2Input, _validate_field


def with_cartesian_harmonic_field(
    field: SonicVPT2Input,
    cartesian_hessian: np.ndarray,
    cartesian_jacobian: np.ndarray,
    *, symmetry_actions=None, symmetry_tolerance=1e-7, harmonic_projector=None,
) -> SonicVPT2Input:
    """Project an aligned high-level Hessian into the low-level field chart.

    ``cartesian_jacobian`` is d x / d s in bohr per native field coordinate;
    the Hessian is in hartree/bohr**2, in the SAME atom order and spatial frame.
    The high-level stationary quadratic model is identified with the shared
    tangent: this is not a Taylor translation of its entire PES to the
    low-level geometry. An optional metric-whitened orthogonal projector
    replaces only its common invariant harmonic sector, using
    B_low + P(B_high-B_low)P. Geometry alignment and this model choice belong to
    acquisition provenance. A curvilinear Hessian away from a stationary
    point additionally needs the gradient/coordinate-curvature contribution
    and is outside this Hessian-only interface.
    """
    f2,_,_,metric,_,_=_validate_field(field)
    hessian=np.asarray(cartesian_hessian,float)
    jacobian=np.asarray(cartesian_jacobian,float)
    if (hessian.ndim!=2 or hessian.shape[0]!=hessian.shape[1]
            or hessian.shape[0]%3 or jacobian.shape!=(len(hessian),len(f2))
            or not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(jacobian))):
        raise ValueError('finite compatible Cartesian Hessian and coordinate Jacobian required')
    scale=max(float(np.max(abs(hessian))),np.finfo(float).tiny)
    if np.max(abs(hessian-hessian.T))>1e-10*scale:
        raise ValueError('Cartesian Hessian must be symmetric in one common frame')
    if np.linalg.matrix_rank(jacobian)!=len(f2):
        raise ValueError('coordinate Jacobian must span the complete field tangent')
    quadratic=jacobian.T@((hessian+hessian.T)/2)@jacobian
    values,vectors=np.linalg.eigh(metric)
    root=(vectors*np.sqrt(values))@vectors.T
    if np.min(np.linalg.eigvalsh(root@quadratic@root))<=0:
        raise ValueError('replacement harmonic field is not a vibrational minimum')
    updates=dict(harmonic_sonic=(quadratic+quadratic.T)/2)
    if symmetry_actions is not None:
        from .field_symmetry import canonical_actions,invariant_tensor
        inverse_root=np.linalg.inv(root)
        actions=canonical_actions([inverse_root@a@root for a in symmetry_actions],
                                  tolerance=symmetry_tolerance)
        quadratic=invariant_tensor(root@quadratic@root,actions)
        updates.update(harmonic_sonic=inverse_root@quadratic@inverse_root,
            symmetry_actions=tuple(root@a@inverse_root for a in actions),
            symmetry_tolerance=symmetry_tolerance)
    if harmonic_projector is not None:
        projector=np.asarray(harmonic_projector,float)
        tolerance=symmetry_tolerance
        if (projector.shape!=f2.shape or not np.all(np.isfinite(projector))
                or np.max(abs(projector-projector.T))>tolerance
                or np.max(abs(projector@projector-projector))>tolerance):
            raise ValueError('orthogonal projector in metric-whitened coordinates required')
        low=root@f2@root
        high=root@updates['harmonic_sonic']@root
        for matrix in (low,high):
            scale=max(np.linalg.norm(matrix),np.finfo(float).tiny)
            if np.linalg.norm(projector@matrix-matrix@projector)>tolerance*scale:
                raise ValueError('selected harmonic sector must reduce both quadratic fields')
        if symmetry_actions is not None and any(np.max(abs(projector@a-a@projector))>tolerance for a in actions):
            raise ValueError('selected harmonic sector is not invariant under the common group')
        mixed=low+projector@(high-low)@projector
        inverse_root=np.linalg.inv(root)
        updates['harmonic_sonic']=inverse_root@mixed@inverse_root
    return replace(field,**updates)


def planar_mode_projector(
    geometry_angstrom, masses_amu, cartesian_jacobian, metric,
    *, planarity_tolerance_angstrom=1e-5, representation_tolerance=1e-7,
):
    """Return the in-plane sector in metric-whitened native coordinates.

    The +1 eigenspace of reflection in the molecular plane is selected.
    Plane orientation comes from geometry, not mode labels or frequencies.
    Linear/nonplanar references and non-invariant coordinate tangents fail
    explicitly. This is independent of whether the full group is abelian.
    """
    from .field_symmetry import canonical_actions
    geometry=np.asarray(geometry_angstrom,float)
    masses=np.asarray(masses_amu,float)
    jacobian=np.asarray(cartesian_jacobian,float)
    metric=np.asarray(metric,float)
    if (geometry.shape!=(len(masses),3) or len(masses)<3 or np.any(masses<=0)
            or metric.ndim!=2 or metric.shape[0]!=metric.shape[1]
            or jacobian.shape!=(3*len(masses),len(metric))
            or not all(np.all(np.isfinite(a)) for a in (geometry,masses,jacobian,metric))
            or not np.isfinite(planarity_tolerance_angstrom) or planarity_tolerance_angstrom<=0):
        raise ValueError('finite geometry, masses, metric, tangent and positive planarity tolerance required')
    centered=geometry-np.average(geometry,axis=0,weights=masses)
    _,singular,axes=np.linalg.svd(centered*np.sqrt(masses[:,None]),full_matrices=False)
    if singular[1]<=planarity_tolerance_angstrom*np.sqrt(sum(masses)):
        raise ValueError('linear geometry has no unique molecular plane')
    normal=axes[-1]
    normal*=1. if normal[np.argmax(abs(normal))]>=0 else -1.
    error=float(np.max(abs(centered@normal)))
    if error>planarity_tolerance_angstrom:
        raise ValueError('reference is not planar at the declared geometry precision')
    values,vectors=np.linalg.eigh(metric)
    if min(values)<=0:
        raise ValueError('positive coordinate metric required')
    root=(vectors*np.sqrt(values))@vectors.T
    tangent=jacobian@root
    weights=np.repeat(masses,3)
    if np.max(abs(tangent.T@(weights[:,None]*tangent)-np.eye(len(metric))))>representation_tolerance:
        raise ValueError('coordinate tangent and mass metric disagree')
    reflection=np.eye(3)-2*np.outer(normal,normal)
    reflected=np.einsum('ab,mbj->maj',reflection,tangent.reshape(len(masses),3,-1)).reshape(tangent.shape)
    action=tangent.T@(weights[:,None]*reflected)
    canonical=canonical_actions([np.eye(len(metric)),action],tolerance=representation_tolerance)
    projector=(np.eye(len(metric))+canonical[1])/2
    return projector,dict(plane_normal=normal.tolist(),maximum_planarity_error_angstrom=error,
        in_plane_dimension=int(round(np.trace(projector))),
        out_of_plane_dimension=int(round(len(metric)-np.trace(projector))))

