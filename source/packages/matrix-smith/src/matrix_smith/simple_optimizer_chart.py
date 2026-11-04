"""Small-system full charts for simple redundant, delocalized, and mode coordinates.

The redundant variables remain the complete primitive list. SVD is used only
to remove the null space from maps and steps, not to relabel Cartesian steps
as internal coordinates. This explicit dense implementation is not LS.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from .models import GICPrimitive
from .evaluation import compiled_gic_primitive_csr


class _LazyOracleHessian:
    """Defer the expensive ANC reference until a dense audit requests it."""

    def __init__(self, source_path):
        self.source_path = source_path
        self._value = None

    def _get(self):
        if self._value is None:
            from matrix_oracle import build_initial_hessian

            self._value = build_initial_hessian(self.source_path)
        return self._value

    def __getattr__(self, name):
        return getattr(self._get(), name)


@dataclass
class SimpleOptimizerChart:
    kind: str
    reference: np.ndarray
    tangent: np.ndarray
    oracle_hessian: object
    source_path: object
    primitives: tuple = ()
    reference_values: np.ndarray | None = None
    primitive_transformation: np.ndarray | None = None

    def _uses_complete_vibrational_tangent(self):
        external_rank = int(self.reference.size - self.tangent.shape[1])
        return external_rank in {3, 5, 6}

    def _install_delocalized_basis(self, xyz, tangent):
        """Install a square DLC chart on the current symmetry/TR-free tangent."""

        self.reference = np.asarray(xyz, dtype=float).copy()
        self.tangent = np.asarray(tangent, dtype=float).copy()
        self.reference_values = self.evaluate(self.reference)[0]
        primitive_b = self.evaluate(self.reference)[1]
        source_differential = (
            primitive_b
            if self._uses_complete_vibrational_tangent()
            else primitive_b @ self.tangent
        )
        primitive_scaling = np.ones(source_differential.shape[0], dtype=float)
        if self._uses_complete_vibrational_tangent():
            row_norms = np.linalg.norm(source_differential, axis=1)
            finite_positive = row_norms[
                np.isfinite(row_norms) & (row_norms > np.finfo(float).tiny)
            ]
            if not finite_positive.size:
                raise ValueError('simple primitives have no finite differential support')
            # Delocalize a complete C1-like chart in a Wilson-normalized
            # primitive metric.  Raw improper and cyclic out-of-plane rows
            # can become arbitrarily large near a singular local plane;
            # letting those rows dominate the SVD produces formally
            # orthogonal DLCs whose finite changes rely on cancellation of
            # large primitive values.  Row normalization retains every
            # family and its rank while making the nonlinear chart local.
            row_floor = max(float(np.median(finite_positive)) * 1.0e-8, 1.0e-12)
            primitive_scaling[:] = 0.0
            supported = np.isfinite(row_norms) & (row_norms > row_floor)
            primitive_scaling[supported] = 1.0 / row_norms[supported]
        weighted_differential = primitive_scaling[:, None] * source_differential
        left, singular, _right = np.linalg.svd(
            weighted_differential,
            full_matrices=False,
        )
        cutoff = max(float(singular[0]) * 1.0e-10, 1.0e-12)
        rank = int(np.count_nonzero(singular > cutoff))
        if rank != self.tangent.shape[1]:
            raise ValueError(
                'simple primitives do not span the refreshed symmetry/TR-qualified tangent'
            )
        # Normalize the DLC eigenvectors by sqrt(lambda)=singular value.
        # U.T alone diagonalizes B B.T but leaves its potentially enormous
        # eigenvalue spread in the optimizer metric.  S^-1 U.T gives an
        # orthonormal differential on the certified symmetry/TR-free tangent,
        # which is the scale needed by a quasi-Newton optimizer.
        transformation = np.asarray(
            (left[:, :rank] / singular[:rank]).T
            * primitive_scaling[None, :],
            dtype=float,
        )
        for row in transformation:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                row *= -1.0
        self.primitive_transformation = transformation

    def differential_condition(self, xyz):
        """Return the condition number of the frozen DLC differential."""

        if self.kind != 'simple_delocalized':
            return 1.0
        _, primitive_b = self.evaluate(np.asarray(xyz, dtype=float))
        differential = self.primitive_transformation @ primitive_b
        if not self._uses_complete_vibrational_tangent():
            differential = differential @ self.tangent
        singular = np.linalg.svd(differential, compute_uv=False)
        if not singular.size or float(singular[-1]) <= np.finfo(float).tiny:
            return float('inf')
        return float(singular[0] / singular[-1])

    def rebase_delocalized(self, xyz):
        """Rebuild a DLC basis after its frozen tangent becomes ill-conditioned.

        The simple-delocalized route is an explicit small-system audit model,
        so materializing its symmetry/TR projector is permitted.  The
        production matrix-free routes remain untouched.
        """

        if self.kind != 'simple_delocalized':
            raise ValueError('only a simple-delocalized chart can be rebased')
        from matrix_chem import read_molecular_symmetry

        coordinates = np.asarray(xyz, dtype=float)
        symmetry = read_molecular_symmetry(self.source_path)
        dimension = coordinates.size
        natoms = coordinates.shape[0]
        centered = coordinates - np.mean(coordinates, axis=0)
        rigid = [np.tile(axis, natoms) for axis in np.eye(3)]
        rigid.extend(
            np.asarray([np.cross(axis, row) for row in centered]).reshape(-1)
            for axis in np.eye(3)
        )
        _u, singular, right = np.linalg.svd(np.asarray(rigid), full_matrices=False)
        rigid_rank = int(
            np.count_nonzero(
                singular > max(float(singular[0]) * 1.0e-10, 1.0e-12)
            )
        )
        external = np.asarray(right[:rigid_rank], dtype=float)
        internal_projector = np.eye(dimension) - external.T @ external
        operations = tuple(getattr(symmetry, 'operations', ()))
        symmetry_projector = np.zeros((dimension, dimension), dtype=float)
        if not operations:
            symmetry_projector[:] = np.eye(dimension)
        else:
            weight = 1.0 / len(operations)
            for operation in operations:
                rotation = np.asarray(operation.rotation, dtype=float)
                permutation = np.asarray(operation.permutation, dtype=int) - 1
                for source, target in enumerate(permutation):
                    symmetry_projector[
                        3 * source : 3 * source + 3,
                        3 * target : 3 * target + 3,
                    ] += weight * rotation
        dense_projector = (
            internal_projector @ symmetry_projector @ internal_projector
        )
        dense_projector = 0.5 * (dense_projector + dense_projector.T)
        eigenvalues, eigenvectors = np.linalg.eigh(dense_projector)
        selected = np.flatnonzero(eigenvalues > 0.5)
        if selected.size != self.tangent.shape[1]:
            raise ValueError('refreshed DLC projector has the wrong active rank')
        tangent = np.asarray(eigenvectors[:, selected], dtype=float)
        for column in range(tangent.shape[1]):
            pivot = int(np.argmax(np.abs(tangent[:, column])))
            if tangent[pivot, column] < 0.0:
                tangent[:, column] *= -1.0
        self._install_delocalized_basis(coordinates, tangent)
        condition = self.differential_condition(coordinates)
        if not np.isfinite(condition) or condition > 1.0e6:
            raise ValueError(
                f'refreshed DLC differential remains ill-conditioned: {condition:.6g}'
            )
        return condition

    def evaluate(self, xyz):
        result = compiled_gic_primitive_csr(self.primitives, np.asarray(xyz),
                                          reference_coords=self.reference)
        if result is None:
            raise RuntimeError('compiled simple-primitive evaluator is unavailable')
        values, data, indices, pointers = result
        b = csr_matrix((data, indices, pointers), shape=(len(values), self.reference.size))
        return np.asarray(values), b.toarray()

    def difference(self, left, right):
        delta = np.asarray(left) - np.asarray(right)
        periodic = np.array([p.function == 'D' for p in self.primitives])
        delta[periodic] = np.arctan2(np.sin(delta[periodic]), np.cos(delta[periodic]))
        return delta

    def actual_q(self, xyz):
        if self.kind == 'hessian_modes':
            return self.tangent.T @ (np.asarray(xyz)-self.reference).ravel()
        primitive_delta = self.difference(self.evaluate(xyz)[0], self.reference_values)
        if self.kind == 'simple_delocalized':
            return self.primitive_transformation @ primitive_delta
        return primitive_delta

    def maps(self, xyz):
        if self.kind == 'hessian_modes':
            return self.tangent, self.tangent.T
        _, b = self.evaluate(xyz)
        coordinate_b = (
            self.primitive_transformation @ b
            if self.kind == 'simple_delocalized'
            else b
        )
        if self.kind == 'simple_delocalized' and self._uses_complete_vibrational_tangent():
            u, s, vt = np.linalg.svd(coordinate_b, full_matrices=False)
            rank = np.count_nonzero(s > max(s[0]*1.e-10, 1.e-12))
            if rank != self.tangent.shape[1]:
                raise ValueError('simple internal chart lost the TR-qualified rank')
            j = (vt.T / s) @ u.T
            return j, coordinate_b
        a = coordinate_b @ self.tangent
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        rank = np.count_nonzero(s > max(s[0]*1.e-10, 1.e-12))
        if rank != self.tangent.shape[1]:
            raise ValueError('simple internal chart lost the symmetry/TR-qualified rank')
        j = self.tangent @ ((vt.T / s) @ u.T)
        return j, coordinate_b

    def _realize_delocalized_newton(self, q, start):
        """Back-transform a square DLC step with safeguarded Newton updates."""

        from matrix_chem.geometry_alignment import kabsch_align

        target = np.asarray(q, dtype=float)
        frame = np.asarray(start, dtype=float).copy()
        coordinates = frame.copy()
        tolerance = 1.0e-8
        maximum_iterations = 16
        accepted_scale = 1.0
        for iteration in range(1, maximum_iterations + 1):
            residual = target - self.actual_q(coordinates)
            residual_norm = float(np.linalg.norm(residual))
            if float(np.max(np.abs(residual), initial=0.0)) <= tolerance:
                self.maps(coordinates)
                return coordinates, {
                    'method': 'SIMPLE_DELOCALIZED_SAFEGUARDED_NEWTON',
                    'iterations': iteration - 1,
                    'converged': True,
                    'residual_norm': residual_norm,
                    'normal_residual': residual_norm,
                    'accepted_scale': accepted_scale,
                }
            jacobian, _coordinate_b = self.maps(coordinates)
            cartesian_step = (jacobian @ residual).reshape(coordinates.shape)
            rms_step = float(np.sqrt(np.mean(cartesian_step * cartesian_step)))
            if rms_step > 0.25:
                cartesian_step *= 0.25 / rms_step

            accepted = False
            for scale in (0.5**power for power in range(11)):
                candidate = kabsch_align(
                    coordinates + scale * cartesian_step,
                    frame,
                )
                candidate_residual = target - self.actual_q(candidate)
                candidate_norm = float(np.linalg.norm(candidate_residual))
                if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                    coordinates = candidate
                    accepted_scale = scale
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(
                    'delocalized Newton back-transformation did not reduce its residual'
                )
        residual = target - self.actual_q(coordinates)
        raise RuntimeError(
            'delocalized Newton back-transformation exceeded its iteration budget: '
            f'{float(np.linalg.norm(residual)):.3g}'
        )

    def realize(self, q, start=None):
        q = np.asarray(q, dtype=float)
        if self.kind == 'hessian_modes':
            return self.reference + (self.tangent @ q).reshape(self.reference.shape), {
                'method': 'HESSIAN_MODES_LINEAR', 'iterations': 0, 'converged': True}
        target = self.reference_values + q if self.kind == 'simple_redundant' else q
        start = self.reference if start is None else np.asarray(start)
        if (
            self.kind == 'simple_delocalized'
            and self._uses_complete_vibrational_tangent()
        ):
            return self._realize_delocalized_newton(target, start)
        z0 = self.tangent.T @ (start-self.reference).ravel()
        def coordinates(z):
            return self.reference + (self.tangent @ z).reshape(self.reference.shape)
        def residual(z):
            primitive_delta = self.difference(
                self.evaluate(coordinates(z))[0], self.reference_values
            )
            if self.kind == 'simple_delocalized':
                return self.primitive_transformation @ primitive_delta - target
            return primitive_delta - q
        def jacobian(z):
            primitive_b = self.evaluate(coordinates(z))[1]
            if self.kind == 'simple_delocalized':
                primitive_b = self.primitive_transformation @ primitive_b
            return primitive_b @ self.tangent
        fit = least_squares(residual, z0, jac=jacobian, method='trf',
                            ftol=1.e-12, xtol=1.e-12, gtol=1.e-10, max_nfev=100)
        # Redundant targets generally are incompatible at finite displacement;
        # DLC targets instead form a square local chart.  In both cases require
        # a stationary residual and certify the realized differential rank.
        normal_residual = float(np.linalg.norm(fit.jac.T @ fit.fun, ord=np.inf))
        if not fit.success or normal_residual > 1.e-7:
            raise RuntimeError(f'redundant back-transformation failed: {normal_residual:.3g}')
        xyz = coordinates(fit.x)
        self.maps(xyz)  # certify rank at the realized point
        method = (
            'SIMPLE_DELOCALIZED_NONLINEAR_BACKTRANSFORM'
            if self.kind == 'simple_delocalized'
            else 'SIMPLE_REDUNDANT_NONLINEAR_LEAST_SQUARES'
        )
        return xyz, {'method': method,
                     'iterations': fit.nfev, 'converged': True,
                     'residual_norm': float(np.linalg.norm(fit.fun)),
                     'normal_residual': normal_residual}

    def hessian(self, xyz, *, pair_error_tolerance=None):
        j, _ = self.maps(xyz)
        source = self.oracle_hessian
        if not np.array_equal(xyz, source.coordinates_angstrom) or pair_error_tolerance is not None:
            from matrix_oracle import build_initial_hessian
            source = build_initial_hessian(self.source_path, coordinates_angstrom=xyz,
                                            pair_error_tolerance=pair_error_tolerance)
        hj = np.column_stack([source.cartesian_action(v) for v in j.T])
        h = j.T @ hj
        return (h+h.T)*.5


def build_simple_optimizer_chart(path, kind, tangent):
    from matrix_chem import read_enriched_xyz
    from matrix_oracle import build_initial_hessian
    reference = np.asarray(read_enriched_xyz(path).coordinates_angstrom, dtype=float)
    if reference.size > 300:
        raise ValueError('full simple-chart validation route is limited to 100 atoms; not LS')
    if kind not in {'simple_redundant', 'simple_delocalized', 'hessian_modes'}:
        raise ValueError('unsupported simple optimizer chart')
    tangent = np.asarray(tangent, dtype=float)
    if (tangent.ndim != 2 or tangent.shape[0] != reference.size or tangent.shape[1] == 0
            or not np.isfinite(tangent).all()
            or not np.allclose(tangent.T @ tangent, np.eye(tangent.shape[1]), rtol=0., atol=1.e-7)):
        raise ValueError('simple chart requires an orthonormal symmetry/TR-qualified tangent')
    source = None
    if kind == 'hessian_modes':
        source = build_initial_hessian(path)
        ht = np.column_stack([source.cartesian_action(v) for v in tangent.T])
        h = tangent.T @ ht
        _, modes = np.linalg.eigh((h+h.T)*.5)
        return SimpleOptimizerChart(kind, reference, tangent @ modes, source, path)
    if kind == 'simple_delocalized':
        # Traditional DLCs use the frozen simple ORACLE pool, not the much
        # larger ddvopt measurement expansion used internally by the xTB ANC
        # Hessian.  This keeps the reference implementation practical while
        # retaining all bonds/pseudobonds, angles, linear bends, torsions and
        # out-of-plane primitives supplied by perception.
        from .oracle_contract_consumer import load_validated_oracle_candidate_pool
        from .contracts import GICForgeContractError

        try:
            pool = load_validated_oracle_candidate_pool(path)
        except GICForgeContractError:
            pool = None
        if pool is not None:
            source = _LazyOracleHessian(path)
            primitives = tuple(
                primitive
                for primitive in pool.candidates
                if primitive.function in {'R', 'A', 'L', 'D', 'U'}
            )
        else:
            # Compatibility for direct small-system validation inputs that
            # predate the frozen ORACLE handoff contract.
            source = build_initial_hessian(path)
            primitives = tuple(
                GICPrimitive(f'XTB{i+1:06d}', p.label, p.kind, p.function,
                             tuple(a+1 for a in p.atoms), mode=p.mode)
                for i, p in enumerate(source.simple_primitives())
            )
        if len(primitives) > 10000:
            raise ValueError('full delocalized audit route exceeds 10000 primitive rows')
    else:
        source = build_initial_hessian(path)
        primitives = tuple(GICPrimitive(f'XTB{i+1:06d}', p.label, p.kind, p.function,
                                        tuple(a+1 for a in p.atoms), mode=p.mode)
                           for i, p in enumerate(source.simple_primitives()))
        if len(primitives) > 3000:
            raise ValueError('full redundant validation route exceeds 3000 primitive rows')
    chart = SimpleOptimizerChart(kind, reference, tangent, source, path, primitives)
    chart.reference_values = chart.evaluate(reference)[0]
    if kind == 'simple_delocalized':
        chart._install_delocalized_basis(reference, tangent)
    chart.maps(reference)
    return chart
