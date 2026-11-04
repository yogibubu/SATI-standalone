"""Sparse, fixed-topology Gaussian electrostatic exclusions for diagnostics.

No Exp-PE terms, self hardnesses or charge-flow channels are removed here.
Indices are zero-based. The pair list is frozen during an optimization; this
is not a reactive/dissociation exclusion policy.
"""

from dataclasses import replace
import operator

import numpy as np
from scipy.special import erf


def canonical_pairs(count, pairs):
    result = set()
    for pair in pairs:
        left, right = map(operator.index, pair)
        if left == right or min(left, right) < 0 or max(left, right) >= count:
            raise ValueError("invalid electrostatic exclusion pair")
        result.add((min(left, right), max(left, right)))
    return tuple(sorted(result))


def exclusions_12_13(count, bonds):
    """Graph distance one or two only; O(N d^2) work for maximum degree d."""
    neighbors = [set() for _ in range(count)]
    for left, right in canonical_pairs(count, bonds):
        neighbors[left].add(right)
        neighbors[right].add(left)
    pairs = set()
    for left, adjacent in enumerate(neighbors):
        reach = adjacent.copy()
        for middle in adjacent:
            reach.update(neighbors[middle])
        pairs.update((left, right) for right in reach if left < right)
    return tuple(sorted(pairs))


class GaussianExcludedPairs:
    """Analytic full Gaussian pair kernel, including long-range Coulomb.

    Stores only the excluded edges and their radial derivatives. No global
    matrix is allocated. Widths and all Cartesian quantities are in bohr.
    """

    def __init__(self, coordinates, widths, pairs):
        xyz = np.asarray(coordinates, dtype=float)
        widths = np.asarray(widths, dtype=float)
        self.count = len(widths)
        if (xyz.shape != (self.count, 3) or widths.shape != (self.count,)
                or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(widths))
                or np.any(widths <= 0)):
            raise ValueError("invalid Gaussian exclusion geometry or widths")
        self.pairs = canonical_pairs(self.count, pairs)
        indices = np.asarray(self.pairs, dtype=np.intp).reshape(-1, 2)
        self.left, self.right = indices.T
        self.delta = xyz[self.left] - xyz[self.right]
        r2 = np.einsum("ij,ij->i", self.delta, self.delta)
        beta2 = 1.0 / (2 * (widths[self.left]**2 + widths[self.right]**2))
        c = 2 * np.sqrt(beta2 / np.pi)
        t = beta2 * r2
        small = t < 1e-3
        self.phi, self.a, self.b = (np.empty(len(indices)) for _ in range(3))
        # phi; a=phi'/r; b=(phi''-phi'/r)/r^2, with regular limits at r=0.
        s = t[small]
        self.phi[small] = c[small] * (1 - s/3 + s**2/10 - s**3/42 + s**4/216)
        self.a[small] = c[small] * beta2[small] * (-2/3 + 2*s/5 - s**2/7 + s**3/27)
        self.b[small] = c[small] * beta2[small]**2 * (4/5 - 4*s/7 + 2*s**2/9)
        large = ~small
        self.phi[large] = erf(np.sqrt(t[large])) / np.sqrt(r2[large])
        exponential = c[large] * np.exp(-t[large])
        self.a[large] = (exponential - self.phi[large]) / r2[large]
        self.b[large] = (-2 * beta2[large] * exponential - 3*self.a[large]) / r2[large]

    def matrix_vector(self, charges, direction=None):
        q = np.asarray(charges, dtype=float).reshape(self.count)
        values = self.phi
        if direction is not None:
            v = np.asarray(direction, dtype=float).reshape(self.count, 3)
            values = self.a * np.einsum("ij,ij->i", self.delta, v[self.left] - v[self.right])
        result = np.zeros(self.count)
        np.add.at(result, self.left, values * q[self.right])
        np.add.at(result, self.right, values * q[self.left])
        return result

    def energy_gradient(self, charges):
        q = np.asarray(charges, dtype=float).reshape(self.count)
        weights = q[self.left] * q[self.right]
        pair_gradient = (weights * self.a)[:, None] * self.delta
        gradient = np.zeros((self.count, 3))
        np.add.at(gradient, self.left, pair_gradient)
        np.add.at(gradient, self.right, -pair_gradient)
        return float(weights @ self.phi), gradient

    def hessian_vector(self, charges, direction):
        q = np.asarray(charges, dtype=float).reshape(self.count)
        v = np.asarray(direction, dtype=float).reshape(self.count, 3)
        dv = v[self.left] - v[self.right]
        dot = np.einsum("ij,ij->i", self.delta, dv)
        pair = (q[self.left]*q[self.right])[:, None] * (
            self.a[:, None]*dv + (self.b*dot)[:, None]*self.delta)
        result = np.zeros_like(v)
        np.add.at(result, self.left, pair)
        np.add.at(result, self.right, -pair)
        return result


class ExcludedGaussianElectrostaticOperator:
    """Apply the same sparse mask to a resident fixed-charge E/G/Hv operator."""

    def __init__(self, baseline, pairs):
        self.baseline = baseline
        self.excluded_pairs = canonical_pairs(len(baseline.charges), pairs)

    def __getattr__(self, name):
        return getattr(self.baseline, name)

    def _kernel(self, xyz):
        return GaussianExcludedPairs(xyz, self.widths_bohr, self.excluded_pairs)

    def evaluate(self, xyz):
        value = self.baseline.evaluate(xyz)
        energy, gradient = self._kernel(xyz).energy_gradient(self.charges)
        return replace(value, energy_hartree=value.energy_hartree-energy,
                       gradient_hartree_per_bohr=value.gradient_hartree_per_bohr.reshape(-1)-gradient.ravel(),
                       backend=value.backend + "_EXCLUDED_PAIRS")

    def energy(self, xyz):
        return self.baseline.energy(xyz) - self._kernel(xyz).energy_gradient(self.charges)[0]

    def hessian_vector_product(self, xyz, direction):
        return (self.baseline.hessian_vector_product(xyz, direction).reshape(-1)
                - self._kernel(xyz).hessian_vector(self.charges, direction).ravel())
