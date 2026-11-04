"""Nonlinear SMITH chart completing a supplied active molecular surface.

The same geometry function defines the energy, forces and full rovibrational
metric. Complementary coordinates are native internal displacements, rather
than Cartesian displacements added after evaluating a primary potential.
"""
from functools import lru_cache

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_smith import build_eckart_constrained_gic_chart, build_gic_b_matrix, evaluate_gic_values
from .curvilinear_vpt2 import sonic_rovibrational_metric


class NativeActiveBathChart:
    """Complete ``active_geometry(a)`` to 3N-6 coordinates in one native basis.

    Geometry is in Angstrom. Active coordinate units are those of the caller;
    complementary coordinates are Angstrom sqrt(amu). ``reference_bath`` has
    mass-weighted Cartesian orthonormal columns. Its projection fixes a smooth
    complementary gauge until a principal angle reaches 90 degrees, where a
    new chart is needed and an exception is raised. ``reference_projected``
    projects out rigid motion only, preserves C1 active-geometry regularity,
    and keeps mixed active/bath metric terms. No rank is silently lost.
    """

    def __init__(self, definition, masses_amu, active_geometry, reference_bath,
                 *, active_dimension, active_jacobian=None,
                 energy_gradient=None, geometry_step=1e-5,
                 minimum_frame_singular_value=1e-7, bath_gauge='orthogonal',
                 chart_tolerance=2e-12):
        self.definition = definition
        self.masses = np.asarray(masses_amu, float).copy()
        self.reference_bath = np.asarray(reference_bath, float).copy()
        self.active_dimension = active_dimension
        if (self.masses.ndim != 1 or len(self.masses)<3
                or not np.all(np.isfinite(self.masses)) or np.any(self.masses<=0)
                or isinstance(active_dimension,bool) or not isinstance(active_dimension,int)
                or not 0<active_dimension<3*len(self.masses)-6):
            raise ValueError('positive masses and a proper active dimension required')
        self.dimension = 3*len(self.masses)-6
        self.bath_dimension = self.dimension-active_dimension
        if (definition.rank != self.dimension
                or self.reference_bath.shape != (3*len(self.masses),self.bath_dimension)
                or not np.all(np.isfinite(self.reference_bath))
                or not np.allclose(self.reference_bath.T @ self.reference_bath,
                                   np.eye(self.bath_dimension),atol=1e-9,rtol=0)):
            raise ValueError('complete native definition and orthonormal reference bath required')
        if (not callable(active_geometry)
                or active_jacobian is not None and not callable(active_jacobian)
                or energy_gradient is not None and not callable(energy_gradient)
                or not np.isfinite(geometry_step) or geometry_step<=0
                or not np.isfinite(minimum_frame_singular_value)
                or not 0<minimum_frame_singular_value<1
                or bath_gauge not in {'orthogonal','reference_projected'}
                or not np.isfinite(chart_tolerance) or chart_tolerance<=0):
            raise ValueError('valid geometry/force callables and positive controls required')
        self.active_geometry = active_geometry
        self.active_jacobian = active_jacobian
        self.energy_gradient = energy_gradient
        self.step = float(geometry_step)
        self.minimum_singular_value = float(minimum_frame_singular_value)
        self.bath_gauge = bath_gauge
        self.chart_tolerance = float(chart_tolerance)
        self.roots = np.repeat(np.sqrt(self.masses),3)

    def _active_position(self, active):
        xyz = np.asarray(self.active_geometry(np.asarray(active,float)),float)
        if xyz.shape != (len(self.masses),3) or not np.all(np.isfinite(xyz)):
            raise ValueError('active geometry must return finite Cartesian Angstrom coordinates')
        return xyz-np.average(xyz,axis=0,weights=self.masses)

    def _active_tangent(self, active):
        if self.active_jacobian is None:
            tangent = np.column_stack([
                self._differentiate(self._active_position,active,axis).ravel()
                for axis in np.eye(self.active_dimension)])
        else:
            tangent = np.asarray(self.active_jacobian(np.asarray(active,float)),float)
        if tangent.shape != (3*len(self.masses),self.active_dimension) or not np.all(np.isfinite(tangent)):
            raise ValueError('active Jacobian has invalid dimensions or values')
        tangent = tangent.reshape(len(self.masses),3,self.active_dimension)
        tangent = tangent-np.einsum('iaj,i->aj',tangent,self.masses)[None,:,:]/self.masses.sum()
        return tangent.reshape(3*len(self.masses),self.active_dimension)

    def _differentiate(self, function, point, axis):
        # Average the two second-order one-sided derivatives. Unlike the
        # usual central two-point stencil this also cancels the O(h) error
        # at a C1 Hermite join with different second derivatives on each side.
        h = self.step
        return (-function(point+2*h*axis)+4*function(point+h*axis)
                -4*function(point-h*axis)+function(point-2*h*axis))/(4*h)

    @lru_cache(maxsize=128)
    def _frame(self, active):
        active = np.asarray(active,float)
        xyz = self._active_position(active)
        roots = np.sqrt(self.masses)[:,None]
        rigid = np.column_stack(
            [(roots*np.broadcast_to(axis,xyz.shape)).ravel() for axis in np.eye(3)]
            +[(roots*np.cross(axis,xyz)).ravel() for axis in np.eye(3)])
        weighted = self.roots[:,None]*self._active_tangent(active)
        constraints = np.column_stack([rigid,weighted])
        norms = np.linalg.norm(constraints,axis=0)
        if np.any(norms==0) or np.linalg.svd(constraints/norms,compute_uv=False)[-1]<1e-10:
            raise ValueError('active and rigid tangents are not independent')
        # A reference-projected gauge depends on position, not on derivatives
        # of the active tangent. It therefore retains the regularity of a C1
        # fitted active geometry. Mixed active/bath metric terms remain in G.
        removed = constraints if self.bath_gauge=='orthogonal' else rigid
        excluded = np.linalg.qr(removed,mode='reduced')[0]
        projected = self.reference_bath-excluded @ (excluded.T @ self.reference_bath)
        left,singular,right = np.linalg.svd(projected,full_matrices=False)
        if singular[-1] < self.minimum_singular_value:
            raise ValueError('complementary frame reaches a chart boundary; no mode may be deleted')
        bath = left @ right
        normalized = weighted/np.linalg.norm(weighted,axis=0)
        full_frame = np.column_stack([normalized,bath])
        if np.linalg.svd(full_frame,compute_uv=False)[-1]<self.minimum_singular_value:
            raise ValueError('active/bath chart loses rank; no mode may be duplicated')
        return xyz,bath,float(singular[-1])

    @lru_cache(maxsize=128)
    def _point(self, active):
        xyz,bath,smallest=self._frame(active)
        chart = build_eckart_constrained_gic_chart(self.definition,xyz,self.masses,
                                                   tolerance=self.chart_tolerance,max_iterations=24)
        native = np.asarray(build_gic_b_matrix(self.definition,coordinates_angstrom=xyz).rows)
        displacement = native @ (bath/self.roots[:,None])
        return xyz,bath,chart,displacement,smallest

    def _coordinates(self, coordinates):
        q = np.asarray(coordinates,float)
        if q.shape != (self.dimension,) or not np.all(np.isfinite(q)):
            raise ValueError('one finite complete active/bath coordinate vector required')
        return q

    def geometry(self, coordinates):
        q = self._coordinates(coordinates)
        _,_,chart,displacement,_ = self._point(tuple(q[:self.active_dimension]))
        return chart.geometry_at_displacement_angstrom(displacement @ q[self.active_dimension:])

    def jacobian(self, coordinates):
        q = self._coordinates(coordinates)
        _,_,chart,displacement,_ = self._point(tuple(q[:self.active_dimension]))
        xyz = self.geometry(q)
        native = np.asarray(build_gic_b_matrix(self.definition,coordinates_angstrom=xyz).rows)
        bath = np.linalg.solve(np.vstack([native,chart.rigid_constraint_rows]),
                               np.vstack([displacement,np.zeros((6,self.bath_dimension))]))
        # Changes of the moving bath and Eckart frame enter these derivatives.
        active = np.column_stack([
            self._differentiate(self.geometry,q,axis).ravel()
            for axis in np.eye(self.dimension)[:self.active_dimension]])
        return np.column_stack([active,bath])

    def energy_and_gradient(self, coordinates):
        if self.energy_gradient is None:
            raise ValueError('an unchanged analytic PES energy/gradient callback is required')
        q = self._coordinates(coordinates)
        energy, gradient = self.energy_gradient(self.geometry(q))
        gradient = np.asarray(gradient,float)
        if not np.isfinite(energy) or gradient.shape != (len(self.masses),3) or not np.all(np.isfinite(gradient)):
            raise ValueError('PES callback must return a finite energy and Cartesian gradient')
        return float(energy),self.jacobian(q).T @ gradient.ravel()

    def rovibrational_metric(self, coordinates):
        q = self._coordinates(coordinates)
        return sonic_rovibrational_metric(
            lambda z:self.geometry(z)/BOHR_TO_ANGSTROM,q,self.masses,
            tangent_at_s=lambda z:self.jacobian(z)/BOHR_TO_ANGSTROM)

    def hamiltonian_coefficients(self, coordinates):
        """Coefficients of the full weak-form nuclear Hamiltonian at one node.

        The metric includes vibration/rotation cross terms. The volume is from
        the determinant of the WHOLE covariant rovibrational metric; a caller
        must not replace it with the determinant of a vibrational sub-block.
        This is the curved-measure representation, with no separately added
        flat-measure Podolsky term. Energies follow the callback's energy unit;
        the metric uses the resident owner's bohr/amu convention.
        """
        energy, gradient = self.energy_and_gradient(coordinates)
        metric = self.rovibrational_metric(coordinates)
        sign, logdet = np.linalg.slogdet(metric)
        if sign<=0:
            raise ValueError('full rovibrational volume must be positive')
        return dict(potential=energy,potential_gradient=gradient,
            contravariant_rovibrational_metric=metric,
            log_volume_weight=-.5*float(logdet),
            vibrational_dimension=self.dimension,rotational_dimension=3,
            measure='complete_curved_rovibrational',
            separate_flat_measure_pseudopotential_included=False)

    def quadrature_geometry(self, coordinates):
        """Physical Cartesian integration map; no inverse GIC solve is needed.

        Quantum basis functions remain functions of the native GIC values.
        This map is only an integration parameterization, with its own complete
        metric and volume. It must not be combined with the inverse-chart G.
        """
        q=self._coordinates(coordinates)
        xyz,bath,_=self._frame(tuple(q[:self.active_dimension]))
        return xyz+((bath/self.roots[:,None]) @ q[self.active_dimension:]).reshape(xyz.shape)

    def quadrature_jacobian(self, coordinates):
        q=self._coordinates(coordinates)
        _,bath,_=self._frame(tuple(q[:self.active_dimension]))
        active=np.column_stack([self._differentiate(self.quadrature_geometry,q,axis).ravel()
            for axis in np.eye(self.dimension)[:self.active_dimension]])
        return np.column_stack([active,bath/self.roots[:,None]])

    def native_basis_coordinates(self, xyz, reference_active):
        """Evaluate native basis arguments and their Cartesian chain rule.

        All native coordinates are evaluated FORWARD at a physical geometry.
        Gaussian quadrature therefore cannot request an impossible inverse
        GIC displacement outside the chart image. This neither clips the
        native coordinates nor discards a quadrature sample or quantum state.
        """
        a=np.asarray(reference_active,float)
        if a.shape!=(self.active_dimension,) or not np.all(np.isfinite(a)):
            raise ValueError('finite complete active reference required')
        position=np.asarray(xyz,float)
        if position.shape!=(len(self.masses),3) or not np.all(np.isfinite(position)):
            raise ValueError('finite physical Cartesian geometry required')
        _,bath,chart,_,_=self._point(tuple(a))
        tangent=np.column_stack([self._active_tangent(a),bath/self.roots[:,None]])
        native=chart.frozen_b_per_angstrom @ tangent
        values=evaluate_gic_values(self.definition,coordinates_angstrom=position)
        b=np.asarray(build_gic_b_matrix(self.definition,coordinates_angstrom=position).rows)
        return np.linalg.solve(native,values-chart.reference_values),np.linalg.solve(native,b)

    def quadrature_coefficients(self, coordinates):
        """Same full Hamiltonian expressed in the physical integration map."""
        if self.energy_gradient is None:
            raise ValueError('an analytic PES callback is required')
        q=self._coordinates(coordinates);xyz=self.quadrature_geometry(q)
        energy,gradient=self.energy_gradient(xyz)
        jac=self.quadrature_jacobian(q)
        metric=sonic_rovibrational_metric(
            lambda z:self.quadrature_geometry(z)/BOHR_TO_ANGSTROM,q,self.masses,
            tangent_at_s=lambda _:jac/BOHR_TO_ANGSTROM)
        sign,logdet=np.linalg.slogdet(metric)
        if sign<=0 or not np.isfinite(energy):raise ValueError('invalid native quadrature coefficients')
        return dict(potential=float(energy),potential_gradient=jac.T @ np.asarray(gradient).ravel(),
            cartesian_geometry_angstrom=xyz,cartesian_jacobian_angstrom=jac,
            contravariant_rovibrational_metric=metric,log_volume_weight=-.5*float(logdet),
            measure='complete_curved_rovibrational',integration_map='physical_cartesian_pullback',
            quantum_basis_coordinates='native_GIC')

    def frame_control(self, active):
        a = np.asarray(active,float)
        if a.shape != (self.active_dimension,) or not np.all(np.isfinite(a)):
            raise ValueError('finite active coordinates required')
        xyz,bath,_,_,smallest = self._point(tuple(a))
        tangent = self.roots[:,None]*self._active_tangent(a)
        return dict(minimum_projection_singular_value=smallest,
            active_bath_overlap=float(np.max(abs(tangent.T @ bath))),
            bath_orthonormality_error=float(np.max(abs(bath.T @ bath-np.eye(self.bath_dimension)))),
            center_of_mass_error_A=float(np.max(abs(np.average(xyz,axis=0,weights=self.masses)))))
