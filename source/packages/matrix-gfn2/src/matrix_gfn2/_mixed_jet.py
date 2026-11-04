"""Internal analytic mixed directional calculus (NumPy only, no steps)."""

from __future__ import annotations
import numpy as np
from math import erf


class MixedJet:
    """Value, coordinate derivatives, one tangent, and mixed derivatives."""

    __array_priority__ = 10000

    def __init__(self, value, first, tangent=None, mixed=None):
        self.v = np.asarray(value, dtype=float)
        self.a = np.asarray(first, dtype=float)
        self.b = np.zeros_like(self.v) if tangent is None else np.asarray(tangent, dtype=float)
        self.ab = np.zeros_like(self.a) if mixed is None else np.asarray(mixed, dtype=float)
        if (
            self.a.shape[1:] != self.v.shape
            or self.ab.shape != self.a.shape
            or self.b.shape != self.v.shape
        ):
            raise ValueError("incompatible mixed-jet shapes")

    @classmethod
    def constant(cls, value, dimension):
        value = np.asarray(value, dtype=float)
        return cls(value, np.zeros((dimension, *value.shape)))

    @property
    def n(self):
        return len(self.a)

    def coerce(self, other):
        if isinstance(other, MixedJet):
            if other.n != self.n:
                raise ValueError("mixed-jet dimensions differ")
            return other
        return self.constant(other, self.n)

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        return MixedJet(
            self.v[index],
            self.a[(slice(None), *index)],
            self.b[index],
            self.ab[(slice(None), *index)],
        )

    def __setitem__(self, index, other):
        other = self.coerce(other)
        index = index if isinstance(index, tuple) else (index,)
        self.v[index] = other.v
        self.b[index] = other.b
        derivative_index = (slice(None), *index)
        extra_axes = self.a[derivative_index].ndim - other.a.ndim
        shape = (self.n,) + (1,) * max(0, extra_axes) + other.v.shape
        self.a[derivative_index] = other.a.reshape(shape)
        self.ab[derivative_index] = other.ab.reshape(shape)

    def __add__(self, other):
        y = self.coerce(other)
        return MixedJet(
            self.v + y.v,
            np.stack([x + y.a[k] for k, x in enumerate(self.a)]),
            self.b + y.b,
            np.stack([x + y.ab[k] for k, x in enumerate(self.ab)]),
        )

    __radd__ = __add__

    def __neg__(self):
        return MixedJet(-self.v, -self.a, -self.b, -self.ab)

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        y = self.coerce(other)
        return MixedJet(
            self.v * y.v,
            np.stack([x * y.v + self.v * y.a[k] for k, x in enumerate(self.a)]),
            self.b * y.v + self.v * y.b,
            np.stack(
                [
                    x * y.v + self.a[k] * y.b + self.b * y.a[k] + self.v * y.ab[k]
                    for k, x in enumerate(self.ab)
                ]
            ),
        )

    __rmul__ = __mul__

    def compose(self, value, derivative, second):
        return MixedJet(
            value,
            self.a * derivative,
            self.b * derivative,
            self.ab * derivative + self.a * (self.b * second),
        )

    def __pow__(self, power):
        if power == 0:
            return self.constant(np.ones_like(self.v), self.n)
        if power == 1:
            return self
        return self.compose(
            self.v**power,
            power * self.v ** (power - 1),
            power * (power - 1) * self.v ** (power - 2),
        )

    def __truediv__(self, other):
        return self * self.coerce(other) ** -1

    def __rtruediv__(self, other):
        return self.coerce(other) * self**-1

    def exp(self):
        value = np.exp(self.v)
        return self.compose(value, value, value)

    def log(self):
        return self.compose(np.log(self.v), 1 / self.v, -1 / self.v**2)

    def tanh(self):
        value = np.tanh(self.v)
        derivative = 1 - value**2
        return self.compose(value, derivative, -2 * value * derivative)

    def erf(self):
        d = 2 / np.sqrt(np.pi) * np.exp(-(self.v**2))
        value = np.fromiter(
            (erf(float(x)) for x in self.v.flat), dtype=float, count=self.v.size
        ).reshape(self.v.shape)
        return self.compose(value, d, -2 * self.v * d)

    def sum(self, axis=None):
        axes = (
            tuple(range(self.v.ndim))
            if axis is None
            else ((axis,) if isinstance(axis, int) else tuple(axis))
        )
        axes = tuple(x % self.v.ndim for x in axes)
        return MixedJet(
            self.v.sum(axis=axes),
            self.a.sum(axis=tuple(x + 1 for x in axes)),
            self.b.sum(axis=axes),
            self.ab.sum(axis=tuple(x + 1 for x in axes)),
        )

    def reshape(self, *shape):
        return MixedJet(
            self.v.reshape(*shape),
            self.a.reshape(self.n, *shape),
            self.b.reshape(*shape),
            self.ab.reshape(self.n, *shape),
        )

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.v.ndim)))
        return MixedJet(
            self.v.transpose(*axes),
            self.a.transpose(0, *(x + 1 for x in axes)),
            self.b.transpose(*axes),
            self.ab.transpose(0, *(x + 1 for x in axes)),
        )

    @property
    def T(self):
        return self.transpose()

    def __matmul__(self, other):
        return contract("...ij,...jk->...ik", self, self.coerce(other))


def stack(values, axis=0):
    values = tuple(values)
    reference = next(x for x in values if isinstance(x, MixedJet))
    values = [reference.coerce(x) for x in values]
    axis = axis % (values[0].v.ndim + 1)
    return MixedJet(
        np.stack([x.v for x in values], axis),
        np.stack([x.a for x in values], axis + 1),
        np.stack([x.b for x in values], axis),
        np.stack([x.ab for x in values], axis + 1),
    )


def contract(expression, *values):
    reference = next(x for x in values if isinstance(x, MixedJet))
    values = [reference.coerce(x) for x in values]

    def term(replacements):
        return np.einsum(
            expression, *(replacements.get(i, x.v) for i, x in enumerate(values)), optimize=True
        )

    value = term({})
    first = []
    mixed = []
    tangent = sum(term({i: x.b}) for i, x in enumerate(values))
    for k in range(reference.n):
        first.append(sum(term({i: x.a[k]}) for i, x in enumerate(values)))
        mixed.append(
            sum(term({i: x.ab[k]}) for i, x in enumerate(values))
            + sum(
                term({i: x.a[k], j: y.b})
                for i, x in enumerate(values)
                for j, y in enumerate(values)
                if i != j
            )
        )
    return MixedJet(value, np.stack(first), tangent, np.stack(mixed))
