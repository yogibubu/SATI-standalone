"""Matrix-free sine DVR with a complete weak-form curvilinear kinetic energy."""

import numpy as np
from scipy.fft import dstn
from scipy.sparse.linalg import LinearOperator, eigsh
from .sine_galerkin import sine_product_quadrature


class CurvilinearSineDVR(LinearOperator):
    """J=0 flat-measure representation with explicit physical-volume connection.

    The caller supplies G in cm^-1 times coordinate^2, V in cm^-1, and
    d log(rho)/dq for the full physical volume density. The kinetic form is
    1/2 (d-1/2 dlogrho)^dagger G (d-1/2 dlogrho). All mixed terms are retained.
    Existing sine values/derivatives are evaluated on an oversampled Gaussian
    rule; using a square collocation first derivative would introduce null
    modes. Coefficients/eigenvectors are in the orthogonal sine-DVR nodal basis.
    No electronic callback or potential fitting is performed by this solver.
    """

    def __init__(
        self, bounds, basis_counts, quadrature_counts, kinetic_cm, log_volume_gradient, potential_cm
    ):
        self.bounds = np.asarray(bounds, float)
        self.counts = tuple(basis_counts)
        self.qcounts = tuple(quadrature_counts)
        if (
            not self.counts
            or self.bounds.shape != (len(self.counts), 2)
            or len(self.qcounts) != len(self.counts)
            or any(
                isinstance(n, bool) or not isinstance(n, (int, np.integer))
                for n in (*self.counts, *self.qcounts)
            )
        ):
            raise ValueError("One physical interval and quadrature per coordinate required")
        self.axes = [
            sine_product_quadrature([wall], [n], [nq])
            for wall, n, nq in zip(self.bounds, self.counts, self.qcounts)
        ]
        nodes = [axis["points"][:, 0] for axis in self.axes]
        self.points = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(-1, len(nodes))
        self.values = [axis["values"] for axis in self.axes]
        self.derivatives = [(1j * axis["momenta"][:, 0]).real for axis in self.axes]

        def sampled(value):
            return np.asarray(value(self.points) if callable(value) else value, float)

        g = sampled(kinetic_cm)
        v = sampled(potential_cm)
        c = sampled(log_volume_gradient)
        d = len(nodes)
        ng = len(self.points)
        if (
            g.shape != (ng, d, d)
            or v.shape != (ng,)
            or c.shape != (ng, d)
            or not all(np.all(np.isfinite(a)) for a in (g, v, c))
            or not np.allclose(g, g.swapaxes(1, 2), atol=1e-12, rtol=1e-12)
            or np.any(np.linalg.eigvalsh(g) <= 0)
        ):
            raise ValueError("Finite positive complete kinetic metric and matching fields required")
        self.metric = g.reshape(self.qcounts + (d, d))
        self.connection = c.reshape(self.qcounts + (d,)) / 2
        self.potential = v.reshape(self.qcounts)
        size = int(np.prod(self.counts))
        super().__init__(dtype=np.dtype(float), shape=(size, size))

    @classmethod
    def from_affine_chart(cls, bounds, basis_counts, quadrature_counts, *,
            origin, transform, kinetic_cm, log_volume_gradient, potential_cm):
        """Use shared product modes x with original coordinates s=origin+A x.

        Full native potentials and variable metrics are evaluated in s. The
        inverse metric and physical-volume connection are transported exactly;
        the constant affine Jacobian adds no volume-gradient term. This is a
        basis/chart change, without a polynomial expansion or metric freezing.
        The caller must choose a physically valid finite x domain and qualify
        its boundaries independently of basis and quadrature convergence.
        """
        a = np.asarray(transform, float)
        o = np.asarray(origin, float)
        d = len(basis_counts)
        if (a.shape != (d, d) or o.shape != (d,)
                or not np.all(np.isfinite(a)) or not np.all(np.isfinite(o))
                or not all(callable(f) for f in (kinetic_cm, log_volume_gradient, potential_cm))):
            raise ValueError('finite invertible affine chart and callable native fields required')
        try:
            inverse = np.linalg.inv(a)
        except np.linalg.LinAlgError as error:
            raise ValueError('invertible affine chart required') from error
        def coordinates(x):
            return o + x @ a.T
        def metric(x):
            g = np.asarray(kinetic_cm(coordinates(x)), float)
            if (g.shape != (len(x), d, d) or not np.all(np.isfinite(g))
                    or not np.allclose(g, g.swapaxes(1, 2), atol=1e-12, rtol=1e-12)):
                raise ValueError('finite symmetric native kinetic metric required')
            transformed = np.einsum('ai,nij,bj->nab', inverse, g, inverse, optimize=True)
            # Exact congruence is symmetric; remove only floating-point
            # cancellation asymmetry after validating the native metric.
            return (transformed+transformed.swapaxes(1, 2))/2
        operator = cls(bounds, basis_counts, quadrature_counts, metric,
            lambda x: np.asarray(log_volume_gradient(coordinates(x))) @ a,
            lambda x: potential_cm(coordinates(x)))
        operator.affine_origin = o.copy()
        operator.affine_transform = a.copy()
        operator.original_points = coordinates(operator.points)
        return operator

    @staticmethod
    def _axis(values, axis, matrix):
        moved = np.moveaxis(values, axis, 0)
        shape = moved.shape
        product = matrix @ np.ascontiguousarray(moved.reshape(shape[0], -1))
        return np.moveaxis(product.reshape((len(matrix),) + shape[1:]), 0, axis)

    def _transform(self, values, derivative=None, adjoint=False):
        for axis in range(len(self.counts)):
            matrix = self.derivatives[axis] if axis == derivative else self.values[axis]
            values = self._axis(values, axis, matrix.T if adjoint else matrix)
        return values

    def quadrature_wavefunctions(self, vectors):
        """Return sqrt(weight) times flat-measure waves at this quadrature.

        The trailing axis labels states. Physical-volume factors are already
        included in the flat-measure representation and must not be added again.
        """
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != self.shape[0]:
            raise ValueError("Eigenvector columns must match the sine-DVR basis")
        coefficients = dstn(vectors.reshape(self.counts + (vectors.shape[1],)),
                            type=1, axes=tuple(range(len(self.counts))), norm="ortho")
        return self._transform(coefficients).reshape(len(self.points), vectors.shape[1])

    def _matvec(self, vector):
        # Orthogonal DST maps the nodal sine DVR to sine expansion coefficients.
        coefficient = dstn(np.asarray(vector).reshape(self.counts), type=1, norm="ortho")
        value = self._transform(coefficient)
        derivative = np.stack(
            [
                self._transform(coefficient, i) - self.connection[..., i] * value
                for i in range(len(self.counts))
            ],
            axis=-1,
        )
        flux = np.einsum("...ij,...j->...i", self.metric, derivative, optimize=True)
        out = self._transform(self.potential * value, adjoint=True)
        for i in range(len(self.counts)):
            out += 0.5 * (
                self._transform(flux[..., i], i, adjoint=True)
                - self._transform(self.connection[..., i] * flux[..., i], adjoint=True)
            )
        return dstn(out, type=1, norm="ortho").ravel()

    def solve(self, *, levels=8, tolerance=1e-10, maximum_iterations=5000):
        if (
            not isinstance(levels, int)
            or not 0 < levels < self.shape[0] - 1
            or not 0 < tolerance < 1
        ):
            raise ValueError("Bounded positive level count and eigensolver tolerance required")
        guess = np.random.default_rng(27311).normal(size=self.shape[0])
        energies, vectors = eigsh(
            self, k=levels, which="SA", v0=guess, tol=tolerance, maxiter=maximum_iterations
        )
        order = np.argsort(energies)
        energies = energies[order]
        vectors = vectors[:, order]
        axes = [np.linspace(lo, hi, n + 2)[1:-1] for (lo, hi), n in zip(self.bounds, self.counts)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))
        residuals = np.array(
            [
                np.linalg.norm(self @ vectors[:, i] - energies[i] * vectors[:, i])
                for i in range(levels)
            ]
        )
        return dict(
            energies_cm=energies,
            eigenvectors=vectors,
            dvr_points=points,
            residuals_cm=residuals,
            diagnostics=dict(
                representation="orthogonal sine DVR; oversampled Gaussian weak-form metric quadrature",
                basis_counts=self.counts,
                quadrature_counts=self.qcounts,
                complete_mixed_kinetic_terms=True,
                physical_volume_connection=True,
                dense_hamiltonian_constructed=False,
                new_electronic_evaluations=0,
            ),
        )
