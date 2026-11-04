"""Declarative factorized operators shared by LOCK Hamiltonians.

The engine keeps physical terms independent from the representation used to
apply them.  The same registry can therefore assemble a dense reference
matrix, expose a matrix-free action, and return exact parameter derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import weakref
from typing import Mapping, Sequence

import numpy as np


LOCK_OPERATOR_MODEL_SCHEMA = "matrix.lock.operator_model.v1"


@dataclass(frozen=True)
class OperatorSpace:
    """One ordered Hilbert-space factor in a LOCK operator model."""

    name: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.name.strip() or isinstance(self.dimension, bool) or self.dimension < 1:
            raise ValueError("operator spaces require a name and positive dimension")


@dataclass(frozen=True)
class FactorizedOperatorTerm:
    """A coefficient multiplying a product of operators on named spaces.

    ``parameter`` is empty for a constant term.  A parameter derivative is
    obtained by setting its coefficient to one while retaining every factor.
    ``adjoint_symmetrized`` represents ``(O + O dagger)/2`` without requiring
    callers to duplicate the reverse-ordered contribution.
    """

    label: str
    factors: tuple[tuple[str, np.ndarray], ...]
    coefficient: complex = 1.0
    parameter: str = ""
    adjoint_symmetrized: bool = True
    symmetry: str = "totally-symmetric"

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.factors:
            raise ValueError("operator terms require a label and at least one factor")
        if len({name for name, _matrix in self.factors}) != len(self.factors):
            raise ValueError("an operator term may act only once on each named space")
        value = complex(self.coefficient)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("operator coefficients must be finite")
        if self.adjoint_symmetrized and abs(value.imag) > 0.0:
            raise ValueError("adjoint-symmetrized operator coefficients must be real")
        normalized = []
        for name, raw in self.factors:
            matrix = np.array(raw, dtype=complex, copy=True)
            if not name.strip() or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("operator factors require a name and square matrix")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("operator factors must be finite")
            matrix.setflags(write=False)
            normalized.append((name, matrix))
        object.__setattr__(self, "factors", tuple(normalized))
        object.__setattr__(self, "coefficient", value)


@dataclass(frozen=True)
class CompiledOperatorModel:
    """Validated factorized Hamiltonian and its exact derivative registry."""

    spaces: tuple[OperatorSpace, ...]
    terms: tuple[FactorizedOperatorTerm, ...]
    parameter_defaults: Mapping[str, float]
    schema: str = LOCK_OPERATOR_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LOCK_OPERATOR_MODEL_SCHEMA or not self.spaces:
            raise ValueError("unsupported or empty LOCK operator model")
        names = tuple(space.name for space in self.spaces)
        if len(names) != len(set(names)):
            raise ValueError("operator-space names must be unique")
        dimensions = {space.name: space.dimension for space in self.spaces}
        parameters = set()
        for term in self.terms:
            for name, matrix in term.factors:
                if name not in dimensions or matrix.shape != (dimensions[name], dimensions[name]):
                    raise ValueError(f"operator factor {name!r} has an incompatible dimension")
            if term.parameter:
                parameters.add(term.parameter)
        if parameters != set(self.parameter_defaults):
            missing = sorted(parameters - set(self.parameter_defaults))
            unused = sorted(set(self.parameter_defaults) - parameters)
            raise ValueError(
                f"operator parameter registry mismatch: missing={missing}, unused={unused}"
            )
        if not all(math.isfinite(float(value)) for value in self.parameter_defaults.values()):
            raise ValueError("operator parameter defaults must be finite")
        # Canonical factors let all execution backends reuse sparse forms and
        # partial contractions.  Factorized terms are immutable to callers, so
        # interning equal read-only arrays is safe inside the compiled model.
        # Clone the term records to preserve immutability if a caller reuses a
        # term while constructing more than one model.
        factors: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}
        interned_terms = []
        for term in self.terms:
            interned = []
            for name, matrix in term.factors:
                key = (matrix.shape, matrix.tobytes())
                shared = factors.setdefault(key, matrix)
                interned.append((name, shared))
            clone = object.__new__(FactorizedOperatorTerm)
            for field in (
                "label", "coefficient", "parameter",
                "adjoint_symmetrized", "symmetry",
            ):
                object.__setattr__(clone, field, getattr(term, field))
            object.__setattr__(clone, "factors", tuple(interned))
            interned_terms.append(clone)
        object.__setattr__(self, "terms", tuple(interned_terms))

    @property
    def dimension(self) -> int:
        return math.prod(space.dimension for space in self.spaces)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameter_defaults)


def _term_matrix(model: CompiledOperatorModel, term: FactorizedOperatorTerm) -> np.ndarray:
    factors = dict(term.factors)
    result = np.ones((1, 1), dtype=complex)
    for space in model.spaces:
        factor = factors.get(space.name)
        if factor is None:
            factor = np.eye(space.dimension, dtype=complex)
        result = np.kron(result, factor)
    if term.adjoint_symmetrized:
        result = 0.5 * (result + result.conj().T)
    return result


def assemble_operator(
    model: CompiledOperatorModel,
    parameters: Mapping[str, float] | None = None,
    *,
    maximum_bytes: int | None = None,
) -> np.ndarray:
    """Assemble a dense Hermitian matrix from the declarative registry."""

    required = model.dimension * model.dimension * np.dtype(complex).itemsize
    if maximum_bytes is not None and required > int(maximum_bytes):
        raise MemoryError(f"dense LOCK operator needs {required} bytes")
    values = dict(model.parameter_defaults)
    if parameters:
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown LOCK operator parameters: {sorted(unknown)}")
        values.update({name: float(value) for name, value in parameters.items()})
    matrix = np.zeros((model.dimension, model.dimension), dtype=complex)
    for term in model.terms:
        scale = term.coefficient * (values[term.parameter] if term.parameter else 1.0)
        factors = dict(term.factors)
        if len(model.spaces) == 2 and all(
            space.name in factors for space in model.spaces
        ):
            left = factors[model.spaces[0].name]
            right = factors[model.spaces[1].name]
            right_dimension = right.shape[0]
            rows, columns = np.nonzero(left)
            for row, column in zip(rows, columns, strict=True):
                row_slice = slice(row * right_dimension, (row + 1) * right_dimension)
                column_slice = slice(
                    column * right_dimension, (column + 1) * right_dimension
                )
                value = scale * left[row, column]
                if term.adjoint_symmetrized:
                    matrix[row_slice, column_slice] += 0.5 * value * right
                    reverse_row = slice(
                        column * right_dimension, (column + 1) * right_dimension
                    )
                    reverse_column = slice(
                        row * right_dimension, (row + 1) * right_dimension
                    )
                    matrix[reverse_row, reverse_column] += (
                        0.5 * value.conjugate() * right.conj().T
                    )
                else:
                    matrix[row_slice, column_slice] += value * right
        else:
            matrix += scale * _term_matrix(model, term)
    error = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1.0)
    if error > 2.0e-12 * scale:
        raise ValueError(f"assembled LOCK operator is not Hermitian: {error:.3e}")
    return 0.5 * (matrix + matrix.conj().T)


def operator_derivatives(model: CompiledOperatorModel) -> dict[str, np.ndarray]:
    """Return exact dense derivatives in declared parameter order."""

    derivatives = {
        name: np.zeros((model.dimension, model.dimension), dtype=complex)
        for name in model.parameter_names
    }
    for term in model.terms:
        if term.parameter:
            derivatives[term.parameter] += term.coefficient * _term_matrix(model, term)
    return derivatives


def apply_operator(
    model: CompiledOperatorModel,
    vectors: np.ndarray,
    parameters: Mapping[str, float] | None = None,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Apply the factorized model without materializing Kronecker matrices."""

    values = dict(model.parameter_defaults)
    if parameters:
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown LOCK operator parameters: {sorted(unknown)}")
        values.update({name: float(value) for name, value in parameters.items()})
    array = np.asarray(vectors, dtype=complex)
    one_vector = array.ndim == 1
    if one_vector:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != model.dimension:
        raise ValueError("LOCK operator vectors have an incompatible dimension")
    if backend not in {"auto", "python", "compiled"}:
        raise ValueError("factorized LOCK backend must be auto, python, or compiled")
    # The native entry point currently accepts one vector per call.  For a
    # whole eigenvector matrix NumPy's batched contractions are faster; use
    # native execution only when explicitly requested until the C++ ABI has a
    # batched kernel.
    use_native = backend == "compiled"
    if use_native:
        try:
            from . import _lock_native

            native_terms = _native_kronecker_terms(model, values)
            answer = np.asarray(_lock_native.kronecker_sum_matvec(native_terms, array))
            return answer[:, 0] if one_vector else answer
        except (ImportError, AttributeError):
            if backend == "compiled":
                raise RuntimeError("compiled LOCK Kronecker kernel is unavailable")
    native_terms = _native_kronecker_terms(model, values)
    answer = _apply_kronecker_terms(native_terms, array, backend="python")
    return answer[:, 0] if one_vector else answer


_SPARSE_FACTOR_CACHE: dict[
    int, tuple[weakref.ReferenceType[np.ndarray], object | None]
] = {}


def _cached_sparse_factor(factor: np.ndarray):
    """Return a cached CSR factor when sparse multiplication is profitable."""

    key = id(factor)
    cached = _SPARSE_FACTOR_CACHE.get(key)
    if cached is not None and cached[0]() is factor:
        return cached[1]
    density = np.count_nonzero(factor) / factor.size
    sparse = None
    if density <= 0.20:
        from scipy.sparse import csr_matrix

        sparse = csr_matrix(factor)

    def discard(reference, *, cache_key=key):
        current = _SPARSE_FACTOR_CACHE.get(cache_key)
        if current is not None and current[0] is reference:
            _SPARSE_FACTOR_CACHE.pop(cache_key, None)

    reference = weakref.ref(factor, discard)
    _SPARSE_FACTOR_CACHE[key] = (reference, sparse)
    return sparse


def _apply_factor_axis(
    factor: np.ndarray, tensor: np.ndarray, axis: int
) -> np.ndarray:
    moved = np.moveaxis(tensor, axis, 0)
    original_shape = moved.shape
    flattened = moved.reshape(original_shape[0], -1)
    sparse = _cached_sparse_factor(factor)
    product = (sparse @ flattened) if sparse is not None else (factor @ flattened)
    return np.moveaxis(np.asarray(product).reshape(original_shape), 0, axis)


def _native_kronecker_terms(
    model: CompiledOperatorModel,
    values: Mapping[str, float],
    *,
    parameter: str | None = None,
) -> tuple[tuple[complex, tuple[np.ndarray, ...]], ...]:
    result = []
    for term in model.terms:
        if parameter is not None and term.parameter != parameter:
            continue
        factors = dict(term.factors)
        ordered = tuple(
            factors.get(space.name)
            if factors.get(space.name) is not None
            else np.eye(space.dimension, dtype=complex)
            for space in model.spaces
        )
        scale = term.coefficient
        if parameter is None and term.parameter:
            scale *= values[term.parameter]
        if term.adjoint_symmetrized:
            result.append((0.5 * scale, ordered))
            result.append((0.5 * scale, tuple(value.conj().T for value in ordered)))
        else:
            result.append((scale, ordered))
    return tuple(result)


def hellmann_feynman_factorized_jacobian(
    model: CompiledOperatorModel,
    eigenvectors: np.ndarray,
    *,
    backend: str = "auto",
    parameter_combinations: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Evaluate exact level derivatives without dense derivative matrices.

    ``parameter_combinations`` maps independent fit variables to linear
    combinations of native operator parameters.  Applying those combinations
    before the matrix-vector products avoids evaluating unused or constrained
    native columns.
    """

    vectors = np.asarray(eigenvectors, dtype=complex)
    if vectors.ndim != 2 or vectors.shape[0] != model.dimension:
        raise ValueError("eigenvectors have an incompatible factorized dimension")
    if backend not in {"auto", "python", "compiled"}:
        raise ValueError("factorized LOCK backend must be auto, python, or compiled")
    if parameter_combinations is None:
        combinations = {
            name: {name: 1.0} for name in model.parameter_names
        }
    else:
        combinations = {
            str(name): {
                str(parameter): float(weight)
                for parameter, weight in weights.items()
                if float(weight) != 0.0
            }
            for name, weights in parameter_combinations.items()
        }
    names = tuple(combinations)
    jacobian = np.zeros((vectors.shape[1], len(names)), dtype=float)
    if len(model.spaces) == 2 and all(
        len(term.factors) == 2
        and {name for name, _factor in term.factors}
        == {space.name for space in model.spaces}
        for term in model.terms
    ):
        dimensions = tuple(space.dimension for space in model.spaces)
        tensor = vectors.reshape(dimensions + (vectors.shape[1],))
        conjugate = tensor.conj()
        destinations: dict[str, list[tuple[int, float]]] = {}
        for column, name in enumerate(names):
            for parameter, weight in combinations[name].items():
                destinations.setdefault(parameter, []).append((column, weight))
        by_right: dict[int, list[FactorizedOperatorTerm]] = {}
        for term in model.terms:
            if term.parameter in destinations:
                right = dict(term.factors)[model.spaces[1].name]
                by_right.setdefault(id(right), []).append(term)
        for terms in by_right.values():
            first_factors = dict(terms[0].factors)
            right_product = _apply_factor_axis(
                first_factors[model.spaces[1].name], tensor, 1
            )
            for term in terms:
                left = dict(term.factors)[model.spaces[0].name]
                # Contract all eigenvectors together using the same batched
                # dense/CSR action as matrix-free application. A Python loop
                # over each nonzero element is expensive for contracted DVR
                # factors, which are generally dense within symmetry blocks.
                left_product = _apply_factor_axis(left, right_product, 0)
                expectation = np.einsum(
                    "ars,ars->s", conjugate, left_product, optimize=True
                )
                if term.adjoint_symmetrized:
                    contribution = (term.coefficient * expectation.real).real
                else:
                    contribution = (term.coefficient * expectation).real
                for column, weight in destinations[term.parameter]:
                    jacobian[:, column] += weight * contribution
        return names, jacobian
    for column, name in enumerate(names):
        weights = combinations[name]
        derivative_terms = []
        for parameter, weight in weights.items():
            derivative_terms.extend(
                (weight * scale, factors)
                for scale, factors in _native_kronecker_terms(
                    model, {}, parameter=parameter
                )
            )
        products = _apply_kronecker_terms(derivative_terms, vectors, backend=backend)
        jacobian[:, column] = np.einsum(
            "ij,ij->j", vectors.conj(), products, optimize=True
        ).real
    return names, jacobian


def _apply_kronecker_terms(
    terms: tuple[tuple[complex, tuple[np.ndarray, ...]], ...],
    vectors: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    # The current native ABI accepts one vector per call.  NumPy's batched
    # contraction is faster for an eigenspace, so native execution remains an
    # explicit choice until the compiled kernel has a batched entry point.
    use_native = backend == "compiled"
    if use_native:
        try:
            from . import _lock_native

            return np.column_stack(
                [
                    np.asarray(_lock_native.kronecker_sum_matvec(terms, vectors[:, index]))
                    for index in range(vectors.shape[1])
                ]
            )
        except (ImportError, AttributeError):
            if backend == "compiled":
                raise RuntimeError("compiled LOCK Kronecker kernel is unavailable")
    dimensions = tuple(factor.shape[0] for factor in terms[0][1]) if terms else ()
    result = np.zeros_like(vectors)
    for scale, factors in terms:
        tensor = vectors.reshape(dimensions + (vectors.shape[1],))
        for axis, factor in enumerate(factors):
            tensor = _apply_factor_axis(factor, tensor, axis)
        result += scale * tensor.reshape(vectors.shape)
    return result


def hellmann_feynman_jacobian(
    eigenvectors: np.ndarray,
    derivatives: Mapping[str, np.ndarray] | Sequence[np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Evaluate exact level derivatives for an operator registry."""

    vectors = np.asarray(eigenvectors, dtype=complex)
    if vectors.ndim != 2:
        raise ValueError("eigenvectors must be a two-dimensional column matrix")
    if isinstance(derivatives, Mapping):
        names = tuple(derivatives)
        matrices = tuple(derivatives[name] for name in names)
    else:
        matrices = tuple(derivatives)
        names = tuple(str(index) for index in range(len(matrices)))
    jacobian = np.empty((vectors.shape[1], len(matrices)), dtype=float)
    for column, matrix in enumerate(matrices):
        operator = np.asarray(matrix, dtype=complex)
        if operator.shape != (vectors.shape[0], vectors.shape[0]):
            raise ValueError("Hellmann-Feynman operator has an incompatible shape")
        jacobian[:, column] = np.einsum(
            "ik,ij,jk->k", vectors.conj(), operator, vectors, optimize=True
        ).real
    return names, jacobian


__all__ = [
    "LOCK_OPERATOR_MODEL_SCHEMA",
    "CompiledOperatorModel",
    "FactorizedOperatorTerm",
    "OperatorSpace",
    "apply_operator",
    "assemble_operator",
    "hellmann_feynman_jacobian",
    "hellmann_feynman_factorized_jacobian",
    "operator_derivatives",
]
