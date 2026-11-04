"""Smooth, invertible computational domains for confined molecular DVRs.

These maps change the integration coordinates, not the molecular coordinate
chart or the potential. Covariant metric tensors must be pulled back with
the returned Jacobian before constructing the kinetic operator and measure.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TriangularGaussianDomain:
    """Map the unit cube to a smoothly tapered domain.

    q_i = radius_i * t_i * exp(-sum_{j>i} a_ij*t_j**2)
          + sum_{j>i} shear_ij*t_j.

    A strictly upper triangular, nonnegative taper matrix gives an invertible
    map throughout the closed cube. Its determinant is the product of the
    positive diagonal derivatives. No grid point is masked or penalized.
    """

    axis_intercepts: np.ndarray
    taper_matrix: np.ndarray
    shear_matrix: np.ndarray | None = None

    def __post_init__(self):
        radii = np.array(self.axis_intercepts, dtype=float, copy=True)
        taper = np.array(self.taper_matrix, dtype=float, copy=True)
        shear = np.zeros_like(taper) if self.shear_matrix is None else np.array(self.shear_matrix, dtype=float, copy=True)
        if (radii.ndim != 1 or not 1 <= len(radii) <= 4
                or taper.shape != (len(radii), len(radii))
                or not np.all(np.isfinite(radii)) or np.any(radii <= 0)
                or not np.all(np.isfinite(taper)) or np.any(taper < 0)
                or np.any(np.tril(taper) != 0)
                or shear.shape != taper.shape or not np.all(np.isfinite(shear))
                or np.any(np.tril(shear) != 0)):
            raise ValueError('Positive radii and a finite nonnegative strictly upper triangular taper required')
        radii.flags.writeable = False
        taper.flags.writeable = False
        shear.flags.writeable = False
        object.__setattr__(self, 'axis_intercepts', radii)
        object.__setattr__(self, 'taper_matrix', taper)
        object.__setattr__(self, 'shear_matrix', shear)

    def evaluate(self, computational_points):
        """Return molecular coordinates q and analytic dq/dt."""
        t = np.asarray(computational_points, dtype=float)
        dimension = len(self.axis_intercepts)
        if (t.ndim < 1 or t.shape[-1] != dimension
                or not np.all(np.isfinite(t)) or np.any(abs(t) > 1 + 1e-12)):
            raise ValueError('Finite computational points in the closed unit cube required')
        scale = self.axis_intercepts * np.exp(-(t*t) @ self.taper_matrix.T)
        tapered = scale*t
        q = tapered + t @ self.shear_matrix.T
        jacobian = (scale[..., :, None]*np.eye(dimension)
                    - 2*tapered[..., :, None]*self.taper_matrix*t[..., None, :]
                    + self.shear_matrix)
        if (not np.all(np.isfinite(jacobian)) or np.any(scale <= 0)
                or not np.all(np.isfinite(q))):
            raise ValueError('Numerically singular domain map')
        return q, jacobian

    def to_dict(self):
        sheared = bool(np.any(self.shear_matrix != 0))
        result = dict(schema='matrix.dvr.triangular_gaussian_domain.v2' if sheared else 'matrix.dvr.triangular_gaussian_domain.v1',
                    taper_matrix=self.taper_matrix.tolist(),
                    computational_walls=[[-1., 1.]]*len(self.axis_intercepts),
                    jacobian='Analytic; molecular q rows, computational t columns',
                    determinant='Product of positive diagonal derivatives')
        if sheared:
            result.update(diagonal_scales_q=self.axis_intercepts.tolist(),
                          shear_matrix=self.shear_matrix.tolist())
        else:
            result['axis_intercepts_q'] = self.axis_intercepts.tolist()
        return result

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') == 'matrix.dvr.triangular_gaussian_domain.v1':
            if 'shear_matrix' in value:
                raise ValueError('Sheared domains require schema v2')
            return cls(value['axis_intercepts_q'], value['taper_matrix'])
        if value.get('schema') == 'matrix.dvr.triangular_gaussian_domain.v2':
            return cls(value['diagonal_scales_q'], value['taper_matrix'], value['shear_matrix'])
        raise ValueError('Unsupported computational domain schema')


def surface_from_cartesian_jet_grid(coordinates, potential_MHz, masses_amu,
                                    geometries_angstrom, tangents_angstrom,
                                    *, periodic, coordinate_jacobian=None,
                                    labels=(), dipole_au=None, metadata=None,
                                    include_geometric_potential=True):
    """Build a LOCK surface from exact fixed-frame Cartesian first jets.

    Tangents have trailing shape (atom, Cartesian, molecular coordinate).
    The optional Jacobian has molecular rows and computational columns.
    The Cartesian jet is pulled back before mass-metric contraction; the
    canonical metric owner constructs the flat-measure scalar potential.
    """
    from .lam_metric import _mass_metric, lam_rotation_surface_from_metric_tensors

    grids = tuple(np.asarray(q, dtype=float) for q in coordinates)
    shape = tuple(len(q) for q in grids)
    dimension = len(grids)
    masses = np.asarray(masses_amu, dtype=float)
    xyz = np.asarray(geometries_angstrom, dtype=float)
    derivative = np.asarray(tangents_angstrom, dtype=float)
    if (not 1 <= dimension <= 4 or masses.ndim != 1 or np.any(masses <= 0)
            or not np.all(np.isfinite(masses))
            or xyz.shape != shape + (len(masses), 3)
            or derivative.ndim != dimension + 3
            or derivative.shape[:-1] != xyz.shape
            or not np.all(np.isfinite(xyz))
            or not np.all(np.isfinite(derivative))):
        raise ValueError('Matching finite Cartesian geometry and first-jet grids required')
    if coordinate_jacobian is None:
        if derivative.shape[-1] != dimension:
            raise ValueError('A Jacobian is required for different coordinate dimensions')
    else:
        jacobian = np.asarray(coordinate_jacobian, dtype=float)
        if (jacobian.shape != shape + (derivative.shape[-1], dimension)
                or not np.all(np.isfinite(jacobian))):
            raise ValueError('Molecular-to-computational Jacobian shape mismatch')
        derivative = np.einsum('...iaq,...qt->...iat', derivative, jacobian)
    centered = xyz - np.einsum('i,...ia->...a', masses, xyz)[..., None, :] / masses.sum()
    derivative = derivative - np.einsum('i,...iat->...at', masses, derivative)[..., None, :, :] / masses.sum()
    inertia, coupling, internal = _mass_metric(masses, centered, derivative.swapaxes(-1, -2))
    return lam_rotation_surface_from_metric_tensors(
        grids, potential_MHz, inertia, coupling, internal,
        periodic=periodic, labels=labels, dipole_au=dipole_au,
        include_geometric_potential=include_geometric_potential,
        metadata={**(metadata or {}), 'cartesian_jet_owner': 'SMITH fixed-frame analytic first jet',
                  'metric_pullback': 'Cartesian tangent multiplied by dq/dt before contraction',
                  'measure_owner': 'matrix_dvr.lam_metric'},
    )


def physical_coordinate_momentum_actions(surface, coordinate_jacobian, vectors):
    """Hermitian physical momenta after a computational coordinate change.

    For J=dq/dt and M=J^-1, p_q[i] is sum_s {M[s,i],p_t[s]}/2
    in the flat DVR measure. Both terms act on the unprojected grid vectors;
    projecting momentum before this transformation loses derivative terms.
    """
    from matrix_rovib.lam_rotation import _apply_momentum

    shape = tuple(surface.grid_shape)
    dim = len(shape)
    jac = np.asarray(coordinate_jacobian, float)
    u = np.asarray(vectors, complex)
    if (jac.shape != shape + (dim, dim) or u.ndim != 2
            or u.shape[0] != int(np.prod(shape))
            or not np.all(np.isfinite(jac)) or not np.all(np.isfinite(u))
            or np.any(np.linalg.det(jac) <= 0)):
        raise ValueError('Finite grid vectors and an orientation-preserving square Jacobian required')
    inverse = np.linalg.inv(jac).reshape(-1, dim, dim)
    result = np.zeros((len(u), dim, u.shape[1]), complex)
    for s in range(dim):
        pu = _apply_momentum(surface, s, u)
        for i in range(dim):
            coefficient = inverse[:, s, i, None]
            result[:, i] += (coefficient * pu
                            + _apply_momentum(surface, s, coefficient * u)) / 2
    return result
