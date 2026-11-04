"""Apply exact finite native quadrature by modal transforms, without a dense H."""
from math import prod
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator


class HarmonicExtensionReference(LinearOperator):
    """Immutable parent blocks plus the extra harmonic occupation energy."""
    def __init__(self, parent_matrix, blocks, extra_frequency_cm, size):
        self.parent = aslinearoperator(parent_matrix)
        self.blocks = tuple((int(s), int(e), int(o), np.asarray(t, dtype=np.int64))
                            for s, e, o, t in blocks)
        self.frequency = float(extra_frequency_cm)
        used = np.zeros(size, dtype=bool)
        if not np.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError('positive extra harmonic frequency required')
        for s, e, o, take in self.blocks:
            if (not 0 <= s < e <= size or len(take) != e-s or o < 0
                    or np.any(take < 0) or np.any(take >= self.parent.shape[0])
                    or len(np.unique(take)) != len(take) or np.any(used[s:e])):
                raise ValueError('disjoint valid parent blocks required')
            used[s:e] = True
        if not np.all(used):
            raise ValueError('harmonic reference must cover every basis row')
        super().__init__(dtype=np.result_type(self.parent.dtype, np.float64), shape=(size, size))

    def _matmat(self, x):
        result = np.zeros_like(x, dtype=np.result_type(x.dtype, self.dtype))
        for start, stop, occupation, take in self.blocks:
            full = np.zeros((self.parent.shape[0], x.shape[1]), dtype=result.dtype)
            full[take] = x[start:stop]
            result[start:stop] = self.parent.matmat(full)[take]
            result[start:stop] += (occupation+.5)*self.frequency*x[start:stop]
        return result

    def _matvec(self, x):
        return self._matmat(x[:, None])[:, 0]


class NativeTensorQuadratureOperator(LinearOperator):
    """The same projected residual F V F*, retaining every grid value/coupling.

    Function tables contain normalized modal functions times sqrt(weights).
    A reference operator is kept exactly (e.g. frozen parent plus harmonic
    extra mode). No new potential points, polynomial fit, SVD truncation or
    numerical coupling pruning is performed. Optional product backends may
    accelerate transforms; independent CPU qualification is then required.
    """
    def __init__(self, basis, function_tables, residual_cm, *, reference,
            product_backend=None, maximum_workspace_bytes=128*1024**3,
            maximum_batch_size=8):
        basis = np.asarray(basis)
        functions = tuple(np.asarray(f, dtype=np.float64) for f in function_tables)
        if (basis.ndim != 2 or basis.dtype.kind not in 'iu' or not len(basis)
                or basis.shape[1] != len(functions) or np.any(basis < 0)
                or len({tuple(row) for row in basis}) != len(basis)
                or any(f.ndim != 2 or not np.all(np.isfinite(f)) for f in functions)
                or maximum_batch_size < 1 or maximum_workspace_bytes < 1):
            raise ValueError('unique integer basis, finite function tables and positive resources required')
        if any(np.max(basis[:, i]) >= len(f) for i, f in enumerate(functions)):
            raise ValueError('function tables do not cover the retained basis')
        self.modal_shape = tuple(len(f) for f in functions)
        self.grid_shape = tuple(f.shape[1] for f in functions)
        residual = np.asarray(residual_cm, dtype=np.float64)
        if residual.size != prod(self.grid_shape) or not np.all(np.isfinite(residual)):
            raise ValueError('one finite residual value per original tensor node required')
        self.reference = aslinearoperator(reference)
        n = len(basis)
        if self.reference.shape != (n, n):
            raise ValueError('reference operator must use the identical retained basis')
        self.basis = basis.copy(); self.basis.setflags(write=False)
        self.functions = tuple(f.copy() for f in functions)
        for f in self.functions: f.setflags(write=False)
        self.residual = residual.reshape(self.grid_shape).copy(); self.residual.setflags(write=False)
        self.indices = tuple(self.basis.T)
        self.product_backend = product_backend
        self.approximate_products = bool(getattr(product_backend, 'approximate_products', False))
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self.maximum_batch_size = maximum_batch_size
        self.maximum_tensor_entries = prod(max(a, b) for a, b in zip(self.modal_shape, self.grid_shape))
        self.diagnostics = dict(operator='exact finite native tensor quadrature',
            dense_hamiltonian_constructed=False, numerical_couplings_pruned=False,
            potential_evaluations=0, approximate_products=self.approximate_products)
        super().__init__(dtype=np.result_type(self.reference.dtype, np.float64), shape=(n, n))

    def _transform(self, values, axis, matrix):
        moved = np.moveaxis(values, axis, 0)
        shape = moved.shape
        right = np.ascontiguousarray(moved.reshape(shape[0], -1))
        product = (matrix@right if self.product_backend is None
                   else self.product_backend.matmul(matrix, right))
        return np.moveaxis(product.reshape((len(matrix),)+shape[1:]), 0, axis)

    def evaluate_grid(self, coefficients):
        """Evaluate weighted modal expansions on the unchanged tensor rule."""
        coefficients=np.asarray(coefficients)
        if (coefficients.ndim!=2 or coefficients.shape[0]!=self.shape[0]
                or not np.all(np.isfinite(coefficients))):
            raise ValueError('finite basis coefficient columns required')
        dtype=np.result_type(self.dtype,coefficients.dtype)
        required=8*self.maximum_tensor_entries*coefficients.shape[1]*np.dtype(dtype).itemsize
        if required>self.maximum_workspace_bytes:raise MemoryError('modal grid evaluation exceeds workspace')
        values=np.zeros(self.modal_shape+(coefficients.shape[1],),dtype=dtype)
        values[self.indices]=coefficients
        for axis,f in enumerate(self.functions):values=self._transform(values,axis,f.T)
        return values.reshape(-1,coefficients.shape[1])

    def _matmat(self, x):
        x = np.asarray(x)
        if not np.all(np.isfinite(x)):
            raise ValueError('finite wavepackets required')
        dtype = np.result_type(self.dtype, x.dtype)
        reserved = x.nbytes+2*self.shape[0]*x.shape[1]*np.dtype(dtype).itemsize+self.residual.nbytes
        per_column = 8*self.maximum_tensor_entries*np.dtype(dtype).itemsize
        batch = min(self.maximum_batch_size, (self.maximum_workspace_bytes-reserved)//per_column)
        if batch < 1:
            raise MemoryError('native tensor action exceeds workspace budget')
        result = np.empty(x.shape, dtype=dtype)
        for first in range(0, x.shape[1], batch):
            last = min(first+batch, x.shape[1]); block = x[:, first:last]
            if self.product_backend is not None and hasattr(self.product_backend, 'tensor_quadrature_action'):
                correction = self.product_backend.tensor_quadrature_action(
                    self.functions, self.residual, self.basis, block)
                if self.product_backend.comparison == 'full':
                    expected = self._cpu_correction(block)
                    error = abs(correction-expected)
                    tolerance = self.product_backend.atol+self.product_backend.rtol*abs(expected)
                    if np.any(error > tolerance):
                        raise ArithmeticError('persistent Metal tensor action failed full CPU comparison')
                    self.product_backend.diagnostics['maximum_checked_error'] = max(
                        self.product_backend.diagnostics['maximum_checked_error'], float(error.max()))
                result[:, first:last] = correction+self.reference.matmat(block)
                continue
            values = self.evaluate_grid(block).reshape(self.grid_shape+(last-first,))
            values *= self.residual[..., None]
            for axis in reversed(range(len(self.functions))):
                values = self._transform(values, axis, self.functions[axis])
            result[:, first:last] = values[self.indices]+self.reference.matmat(block)
        return result

    def _cpu_correction(self, block):
        values = np.zeros(self.modal_shape+(block.shape[1],), dtype=np.result_type(block.dtype, self.dtype))
        values[self.indices] = block
        for axis, f in enumerate(self.functions):
            moved = np.moveaxis(values, axis, 0); shape = moved.shape
            values = np.moveaxis((f.T@moved.reshape(shape[0], -1)).reshape((f.shape[1],)+shape[1:]), 0, axis)
        values *= self.residual[..., None]
        for axis in reversed(range(len(self.functions))):
            f = self.functions[axis]; moved = np.moveaxis(values, axis, 0); shape = moved.shape
            values = np.moveaxis((f@moved.reshape(shape[0], -1)).reshape((len(f),)+shape[1:]), 0, axis)
        return values[self.indices]

    def _matvec(self, x):
        return self._matmat(np.asarray(x).reshape(-1, 1))[:, 0]

    def _rmatvec(self, x):
        return self._matvec(x)


def native_extension_tensor_operator(core, basis, *, extra_frequency_cm,
        quadrature_points_per_axis, residual_cm, product_backend=None,
        maximum_workspace_bytes=128*1024**3, maximum_batch_size=8):
    """Build an operator from a frozen parent and an acquired extension cache.

    The caller retains responsibility for verifying the residual-cache identity
    against its saved acquisition metadata. This factory performs no PES calls.
    """
    from scipy.special import roots_hermite
    basis = np.asarray(basis); counts = tuple(quadrature_points_per_axis)
    if (basis.ndim != 2 or basis.dtype.kind not in 'iu'
            or basis.shape[1] != core.basis.shape[1]+1 or len(counts) != basis.shape[1]
            or np.any(basis < 0) or not len(basis)
            or any(not isinstance(v, (int, np.integer)) or v < 2 for v in counts)
            or any(v != core.diagnostics['quadrature_points_per_axis'] for v in counts[:-1])):
        raise ValueError('extension basis and original parent tensor rule required')
    functions = []
    for mode, count in enumerate(counts):
        axis, weights = roots_hermite(count); maximum = int(basis[:, mode].max())
        if count <= maximum+1:
            raise ValueError('quadrature does not cover retained modal products')
        poly = np.empty((maximum+1, count)); poly[0] = np.pi**(-.25)
        if maximum: poly[1] = np.sqrt(2)*axis*poly[0]
        for order in range(2, maximum+1):
            poly[order] = np.sqrt(2/order)*axis*poly[order-1]-np.sqrt((order-1)/order)*poly[order-2]
        f = poly*np.sqrt(weights)
        if np.max(abs(f@f.T-np.eye(len(f)))) > 1e-10:
            raise ArithmeticError('modal quadrature orthogonality failed')
        functions.append(f)
    lookup = {tuple(row): i for i, row in enumerate(core.basis)}; blocks = []
    for occupation in np.unique(basis[:, -1]):
        take = np.flatnonzero(basis[:, -1] == occupation)
        if not np.array_equal(take, np.arange(take[0], take[-1]+1)):
            raise ValueError('extra occupations must use the original contiguous blocks')
        try: parent_rows = np.array([lookup[tuple(row)] for row in basis[take, :-1]])
        except KeyError as error:
            raise ValueError('extension contains a state absent from the frozen parent') from error
        blocks.append((int(take[0]), int(take[-1]+1), int(occupation), parent_rows))
    parent_matrix = getattr(core, 'dense_matrix_view', None)
    if parent_matrix is None: parent_matrix = core.matrix_cm
    reference = HarmonicExtensionReference(parent_matrix, blocks, extra_frequency_cm, len(basis))
    return NativeTensorQuadratureOperator(basis, functions, residual_cm,
        reference=reference, product_backend=product_backend,
        maximum_workspace_bytes=maximum_workspace_bytes, maximum_batch_size=maximum_batch_size)
