"""Transverse harmonic zero-point energies on a fixed electronic LAM path.

Electronic derivatives and Cartesian tangents describe a frozen internal
chart. Isotope masses enter only the metric. The rotational Schur complement
imposes zero total angular momentum on the transverse vibrational problem.
"""

from dataclasses import dataclass

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM


@dataclass(frozen=True)
class TransverseZeroPointEnergy:
    frequencies_cm: np.ndarray
    zero_point_cm: float
    reduced_mass_metric_amu_angstrom2: np.ndarray


@dataclass(frozen=True)
class HarmonicTransverseRelaxation:
    displacement: np.ndarray
    energy_change: float
    residual_gradient: np.ndarray


@dataclass(frozen=True)
class TransverseZeroPointEnergyGrid:
    frequencies_cm: np.ndarray
    zero_point_cm: np.ndarray
    reduced_mass_metric_amu_angstrom2: np.ndarray


def transverse_zero_point_energy_grid(coordinates_angstrom,masses_amu,
        transverse_cartesian_tangents_angstrom,transverse_hessian_hartree):
    """Batched form of the same zero-J transverse harmonic problem.

    Leading grid dimensions must match exactly. No eigenvalue clipping,
    coordinate interpolation, or electronic evaluation is performed.
    """
    xyz=np.asarray(coordinates_angstrom,float);m=np.asarray(masses_amu,float)
    tangent=np.asarray(transverse_cartesian_tangents_angstrom,float)
    H=np.asarray(transverse_hessian_hartree,float)
    if xyz.ndim<3 or xyz.shape[-1]!=3 or m.shape!=(xyz.shape[-2],) or np.any(m<=0):
        raise ValueError('Matching batched nonlinear coordinates and positive masses required')
    n=len(m);batch=xyz.shape[:-2]
    if (n<3 or tangent.ndim!=len(batch)+2 or tangent.shape[:-2]!=batch
            or tangent.shape[-2]!=3*n or not 0<tangent.shape[-1]<=3*n-6
            or H.shape!=batch+(tangent.shape[-1],)*2
            or not all(np.all(np.isfinite(a)) for a in (xyz,m,tangent,H))
            or not np.allclose(H,H.swapaxes(-1,-2),rtol=1e-8,atol=1e-12)):
        raise ValueError('Finite compatible batched transverse jets and symmetric Hessians required')
    xyz=xyz-np.einsum('i,...ia->...a',m,xyz)[...,None,:]/m.sum()
    D=tangent.reshape(batch+(n,3,tangent.shape[-1])).copy()
    D-=np.einsum('i,...iam->...am',m,D)[...,None,:,:]/m.sum()
    second=np.einsum('i,...ia,...ib->...ab',m,xyz,xyz)
    inertia=np.trace(second,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)-second
    if np.any(np.linalg.eigvalsh(inertia)<=0):
        raise ValueError('Transverse grid requires nonlinear rotational inertia')
    coupling=np.einsum('i,...ima->...am',m,np.cross(xyz[..., :,None,:],D.swapaxes(-1,-2)))
    metric=np.einsum('i,...iam,...ian->...mn',m,D,D)
    metric-=coupling.swapaxes(-1,-2)@np.linalg.solve(inertia,coupling)
    metric=.5*(metric+metric.swapaxes(-1,-2))
    eig,vec=np.linalg.eigh(metric)
    if np.any(eig<=0):raise ValueError('Nonpositive transverse zero-J mass metric')
    inverse_half=(vec/np.sqrt(eig)[...,None,:])@vec.swapaxes(-1,-2)
    force=inverse_half@H@inverse_half
    values=np.linalg.eigvalsh(.5*(force+force.swapaxes(-1,-2)))
    if np.any(values<=0):raise ValueError('Transverse vibrational instability in grid')
    frequencies=np.sqrt(values)*HESSIAN_EIGENVALUE_TO_CM*BOHR_TO_ANGSTROM
    return TransverseZeroPointEnergyGrid(frequencies,frequencies.sum(axis=-1)/2,metric)


def harmonic_transverse_relaxation(gradient, hessian) -> HarmonicTransverseRelaxation:
    """Minimize ``g.b + b.H.b/2`` in the supplied transverse chart.

    The caller owns the electronic chart, energy units, and any explicit
    reference-force convention. No clipping, imaginary-mode removal, or
    geometrical optimization is hidden in this local quadratic response.
    Positive transverse curvature is required; the returned displacement
    does not certify the accuracy of the quadratic electronic model.
    """
    force = np.asarray(gradient, dtype=float)
    curvature = np.asarray(hessian, dtype=float)
    if (force.ndim != 1 or not force.size
            or curvature.shape != (force.size, force.size)
            or not np.all(np.isfinite(force))
            or not np.all(np.isfinite(curvature))):
        raise ValueError("finite transverse gradient and compatible Hessian required")
    if not np.allclose(curvature, curvature.T, rtol=1e-8, atol=1e-12):
        raise ValueError("transverse Hessian must be symmetric")
    # A Cholesky factor both enforces stability and solves the same model.
    factor = np.linalg.cholesky(curvature)
    displacement = -np.linalg.solve(factor.T, np.linalg.solve(factor, force))
    residual = force + curvature @ displacement
    energy = float(force @ displacement + .5 * displacement @ curvature @ displacement)
    return HarmonicTransverseRelaxation(displacement, energy, residual)


def transverse_zero_point_energy(
    coordinates_angstrom,
    masses_amu,
    transverse_cartesian_tangents_angstrom,
    transverse_hessian_hartree,
) -> TransverseZeroPointEnergy:
    """Diagonalize the transverse Hessian with its zero-J mass metric.

    Tangent columns are Cartesian derivatives with respect to the retained
    transverse chart coordinates. The Hessian is in the same coordinate
    units and includes the chart-curvature/constraint term away from a
    stationary point. Its energy unit is hartree. This evaluates the local
    harmonic ZPE at the supplied geometry; it does not optimize a transverse
    gradient or introduce an implicit adiabatic-relaxation energy.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    tangents = np.asarray(transverse_cartesian_tangents_angstrom, dtype=float)
    hessian = np.asarray(transverse_hessian_hartree, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or masses.shape != (len(xyz),)
            or len(xyz) < 3 or np.any(masses <= 0)
            or tangents.ndim != 2 or tangents.shape[0] != xyz.size
            or not 0 < tangents.shape[1] <= xyz.size - 6
            or hessian.shape != (tangents.shape[1],) * 2
            or not all(np.all(np.isfinite(a)) for a in (xyz, masses, tangents, hessian))):
        raise ValueError("finite nonlinear geometry and compatible transverse derivatives required")
    if not np.allclose(hessian, hessian.T, rtol=1e-8, atol=1e-12):
        raise ValueError("transverse Hessian must be symmetric")
    xyz = xyz - np.average(xyz, axis=0, weights=masses)
    directions = tangents.reshape(len(xyz), 3, -1).copy()
    directions -= np.einsum("i,iam->am", masses, directions)[None, :, :] / masses.sum()
    inertia = sum(m * ((x @ x) * np.eye(3) - np.outer(x, x)) for m, x in zip(masses, xyz))
    if np.min(np.linalg.eigvalsh(inertia)) <= 0:
        raise ValueError("transverse ZPE requires a nonlinear rotational inertia")
    coupling = np.einsum(
        "i,ima->am", masses,
        np.cross(xyz[:, None, :], directions.swapaxes(1, 2)),
    )
    metric = np.einsum("i,iam,ian->mn", masses, directions, directions)
    metric -= coupling.T @ np.linalg.solve(inertia, coupling)
    metric = .5 * (metric + metric.T)
    values, vectors = np.linalg.eigh(metric)
    if np.min(values) <= 0:
        raise ValueError("transverse tangents have a nonpositive zero-J metric")
    inverse_half = (vectors / np.sqrt(values)) @ vectors.T
    eigenvalues = np.linalg.eigvalsh(inverse_half @ hessian @ inverse_half)
    if np.min(eigenvalues) <= 0:
        raise ValueError("transverse vibrational instability on the supplied LAM path")
    frequencies = np.sqrt(eigenvalues) * HESSIAN_EIGENVALUE_TO_CM * BOHR_TO_ANGSTROM
    return TransverseZeroPointEnergy(frequencies, float(frequencies.sum() / 2), metric)
