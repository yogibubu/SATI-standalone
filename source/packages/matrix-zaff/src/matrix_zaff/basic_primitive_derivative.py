"""Analytic second derivatives for basic ZAFF valence primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PrimitiveSecondDerivative:
    """Sparse Hessian of one primitive, with global Cartesian indices."""

    entries: tuple[tuple[int, int, float], ...]
    analytic: bool = True


class _Jet2:
    __slots__ = ("val", "grad", "hess")

    def __init__(self, val: float, grad: np.ndarray, hess: np.ndarray):
        self.val = float(val)
        self.grad = np.asarray(grad, dtype=float)
        self.hess = np.asarray(hess, dtype=float)

    def _coerce(self, other: object) -> "_Jet2":
        if isinstance(other, _Jet2):
            return other
        return _Jet2(float(other), np.zeros_like(self.grad), np.zeros_like(self.hess))

    def __add__(self, other: object) -> "_Jet2":
        rhs = self._coerce(other)
        return _Jet2(self.val + rhs.val, self.grad + rhs.grad, self.hess + rhs.hess)

    __radd__ = __add__

    def __sub__(self, other: object) -> "_Jet2":
        rhs = self._coerce(other)
        return _Jet2(self.val - rhs.val, self.grad - rhs.grad, self.hess - rhs.hess)

    def __rsub__(self, other: object) -> "_Jet2":
        return self._coerce(other).__sub__(self)

    def __neg__(self) -> "_Jet2":
        return _Jet2(-self.val, -self.grad, -self.hess)

    def __mul__(self, other: object) -> "_Jet2":
        rhs = self._coerce(other)
        hess = (
            self.val * rhs.hess
            + rhs.val * self.hess
            + np.outer(self.grad, rhs.grad)
            + np.outer(rhs.grad, self.grad)
        )
        return _Jet2(self.val * rhs.val, self.val * rhs.grad + rhs.val * self.grad, hess)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "_Jet2":
        return self * _unary(self._coerce(other), lambda x: 1.0 / x, lambda x: -1.0 / x**2, lambda x: 2.0 / x**3)

    def __rtruediv__(self, other: object) -> "_Jet2":
        return self._coerce(other).__truediv__(self)


class _JetHv:
    """Second-order directional jet storing ``value``, ``grad`` and ``H v``."""

    __slots__ = ("val", "grad", "dot", "grad_dot")

    def __init__(
        self,
        val: float,
        grad: np.ndarray,
        dot: float,
        grad_dot: np.ndarray,
    ):
        self.val = float(val)
        self.grad = np.asarray(grad, dtype=float)
        self.dot = float(dot)
        self.grad_dot = np.asarray(grad_dot, dtype=float)

    def _coerce(self, other: object) -> "_JetHv":
        if isinstance(other, _JetHv):
            return other
        return _JetHv(
            float(other),
            np.zeros_like(self.grad),
            0.0,
            np.zeros_like(self.grad),
        )

    def __add__(self, other: object) -> "_JetHv":
        rhs = self._coerce(other)
        return _JetHv(
            self.val + rhs.val,
            self.grad + rhs.grad,
            self.dot + rhs.dot,
            self.grad_dot + rhs.grad_dot,
        )

    __radd__ = __add__

    def __sub__(self, other: object) -> "_JetHv":
        rhs = self._coerce(other)
        return _JetHv(
            self.val - rhs.val,
            self.grad - rhs.grad,
            self.dot - rhs.dot,
            self.grad_dot - rhs.grad_dot,
        )

    def __rsub__(self, other: object) -> "_JetHv":
        return self._coerce(other).__sub__(self)

    def __neg__(self) -> "_JetHv":
        return _JetHv(-self.val, -self.grad, -self.dot, -self.grad_dot)

    def __mul__(self, other: object) -> "_JetHv":
        rhs = self._coerce(other)
        return _JetHv(
            self.val * rhs.val,
            self.val * rhs.grad + rhs.val * self.grad,
            self.dot * rhs.val + self.val * rhs.dot,
            self.dot * rhs.grad
            + self.val * rhs.grad_dot
            + rhs.dot * self.grad
            + rhs.val * self.grad_dot,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "_JetHv":
        rhs = self._coerce(other)
        return self * _unary(
            rhs,
            lambda x: 1.0 / x,
            lambda x: -1.0 / x**2,
            lambda x: 2.0 / x**3,
        )

    def __rtruediv__(self, other: object) -> "_JetHv":
        return self._coerce(other).__truediv__(self)



def analytic_basic_primitive_second_derivative(
    primitive,
    coordinates_angstrom: np.ndarray,
    *,
    zero_tolerance: float = 1.0e-12,
) -> PrimitiveSecondDerivative:
    """Return the analytic sparse Hessian of an R, A, or D primitive."""

    coords = np.asarray(coordinates_angstrom, dtype=float)
    function = str(primitive.function).upper()
    if function not in {"R", "A", "D", "U"}:
        raise ValueError(f"unsupported basic ZAFF primitive: {function}")
    atoms = tuple(int(atom) for atom in primitive.atoms)
    ordered_atoms = tuple(sorted(set(atoms)))
    atom_position = {atom: index for index, atom in enumerate(ordered_atoms)}
    dimension = 3 * len(ordered_atoms)
    jets: dict[int, list[_Jet2]] = {}
    for atom in ordered_atoms:
        row: list[_Jet2] = []
        for axis in range(3):
            local = 3 * atom_position[atom] + axis
            gradient = np.zeros(dimension)
            gradient[local] = 1.0
            row.append(
                _Jet2(
                    coords[atom - 1, axis],
                    gradient,
                    np.zeros((dimension, dimension)),
                )
            )
        jets[atom] = row
    value = _basic_primitive_value_jet(function, atoms, jets)
    if not np.all(np.isfinite(value.hess)):
        raise FloatingPointError("non-finite analytic basic-primitive derivative")
    entries: list[tuple[int, int, float]] = []
    for local_a in range(dimension):
        atom_a, axis_a = ordered_atoms[local_a // 3], local_a % 3
        global_a = 3 * (atom_a - 1) + axis_a
        for local_b in range(dimension):
            element = float(value.hess[local_a, local_b])
            if abs(element) <= zero_tolerance:
                continue
            atom_b, axis_b = ordered_atoms[local_b // 3], local_b % 3
            global_b = 3 * (atom_b - 1) + axis_b
            entries.append((global_a, global_b, element))
    return PrimitiveSecondDerivative(tuple(entries))


def analytic_basic_primitive_hessian_vector(
    primitive,
    coordinates_angstrom: np.ndarray,
    vector_angstrom: np.ndarray,
) -> np.ndarray:
    """Apply one primitive-coordinate Hessian analytically without forming it."""

    coords = np.asarray(coordinates_angstrom, dtype=float)
    direction = np.asarray(vector_angstrom, dtype=float)
    if direction.size == coords.size:
        direction = direction.reshape(coords.shape)
    if direction.shape != coords.shape:
        raise ValueError("primitive Hessian-vector dimensions are inconsistent")
    function = str(primitive.function).upper()
    if function not in {"R", "A", "D", "U"}:
        raise ValueError(f"unsupported basic ZAFF primitive: {function}")
    atoms = tuple(int(atom) for atom in primitive.atoms)
    ordered_atoms = tuple(sorted(set(atoms)))
    atom_position = {atom: index for index, atom in enumerate(ordered_atoms)}
    dimension = 3 * len(ordered_atoms)
    jets: dict[int, list[_JetHv]] = {}
    for atom in ordered_atoms:
        row: list[_JetHv] = []
        for axis in range(3):
            local = 3 * atom_position[atom] + axis
            gradient = np.zeros(dimension)
            gradient[local] = 1.0
            row.append(
                _JetHv(
                    coords[atom - 1, axis],
                    gradient,
                    direction[atom - 1, axis],
                    np.zeros(dimension),
                )
            )
        jets[atom] = row
    value = _basic_primitive_value_jet(function, atoms, jets)
    if not np.all(np.isfinite(value.grad_dot)):
        raise FloatingPointError("non-finite analytic primitive Hessian-vector product")
    product = np.zeros(coords.size)
    for local, element in enumerate(value.grad_dot):
        atom, axis = ordered_atoms[local // 3], local % 3
        product[3 * (atom - 1) + axis] = element
    return product


def _basic_primitive_value_jet(function, atoms, jets):
    points = [jets[atom] for atom in atoms]
    if function == "R":
        return _norm(_sub(points[0], points[1]))
    if function == "A":
        left = _unit(_sub(points[0], points[1]))
        right = _unit(_sub(points[2], points[1]))
        return _acos(_dot(left, right))
    if function == "D":
        p0, p1, p2, p3 = points
        b0 = _neg(_sub(p1, p0))
        b1 = _unit(_sub(p2, p1))
        b2 = _sub(p3, p2)
        v = _sub(b0, _scale(b1, _dot(b0, b1)))
        w = _sub(b2, _scale(b1, _dot(b2, b1)))
        return _atan2(_dot(_cross(b1, v), w), _dot(v, w))
    if function == "U":
        vector = _sub(points[3], points[0])
        normal = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
        sine = _dot(_unit(vector), _unit(normal))
        return _unary(
            _asin(sine),
            lambda value: -value,
            lambda _value: -1.0,
            lambda _value: 0.0,
        )
    raise ValueError(f"unsupported basic ZAFF primitive: {function}")


def _unary(value: _Jet2 | _JetHv, f: Callable[[float], float], fp, fpp):
    if isinstance(value, _JetHv):
        first = fp(value.val)
        return _JetHv(
            f(value.val),
            first * value.grad,
            first * value.dot,
            first * value.grad_dot
            + fpp(value.val) * value.dot * value.grad,
        )
    return _Jet2(
        f(value.val),
        fp(value.val) * value.grad,
        fp(value.val) * value.hess + fpp(value.val) * np.outer(value.grad, value.grad),
    )


def _sqrt(value):
    if value.val <= 0.0:
        raise FloatingPointError("square root derivative is singular")
    return _unary(value, np.sqrt, lambda x: 0.5 / np.sqrt(x), lambda x: -0.25 / x**1.5)


def _acos(value):
    if abs(value.val) >= 1.0:
        raise FloatingPointError("acos derivative is singular")
    return _unary(value, np.arccos, lambda x: -1.0 / np.sqrt(1.0 - x * x), lambda x: -x / (1.0 - x * x) ** 1.5)


def _asin(value):
    if abs(value.val) >= 1.0:
        raise FloatingPointError("asin derivative is singular")
    return _unary(value, np.arcsin, lambda x: 1.0 / np.sqrt(1.0 - x * x), lambda x: x / (1.0 - x * x) ** 1.5)


def _atan2(y, x):
    radius2 = x * x + y * y
    if radius2.val <= 1.0e-20:
        raise FloatingPointError("atan2 derivative is singular")
    # Use the better-conditioned local chart.  The additive quadrant constant
    # has zero derivatives, so restoring atan2's value is sufficient.
    if abs(x.val) >= abs(y.val):
        ratio = y / x
        result = _atan(ratio)
    else:
        ratio = x / y
        result = -_atan(ratio)
    if isinstance(result, _JetHv):
        return _JetHv(
            float(np.arctan2(y.val, x.val)),
            result.grad,
            result.dot,
            result.grad_dot,
        )
    return _Jet2(float(np.arctan2(y.val, x.val)), result.grad, result.hess)


def _atan(value):
    return _unary(
        value,
        np.arctan,
        lambda z: 1.0 / (1.0 + z * z),
        lambda z: -2.0 * z / (1.0 + z * z) ** 2,
    )


def _constant_like(template, value):
    return _Jet2(value, np.zeros_like(template.grad), np.zeros_like(template.hess))


def _add(a, b):
    return [a[i] + b[i] for i in range(3)]


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _neg(a):
    return [-item for item in a]


def _scale(a, factor):
    return [factor * item for item in a]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return _sqrt(_dot(a, a))


def _unit(a):
    return _scale(a, 1.0 / _norm(a))


__all__ = [
    "PrimitiveSecondDerivative",
    "analytic_basic_primitive_hessian_vector",
    "analytic_basic_primitive_second_derivative",
]
