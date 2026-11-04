"""Bonded ZAFF energy/gradient/Hv in local SONIC natural variables.

There is no Cartesian reconstruction or B matrix in this evaluator. The
caller still owns symmetry/constraint maps and the nonbonded geometry.
Existing physical potentials and their dissociation switches are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .sonic_variables import SonicVariableMap


class LocalSonicKernel(Protocol):
    def evaluate(self, values: np.ndarray, direction: np.ndarray | None):
        """Return energy, local gradient, and optional local Hessian product."""
        ...


@dataclass(frozen=True)
class ScalarSonicKernel:
    potential: object

    def evaluate(self, values, direction):
        if len(values) != 1:
            raise ValueError("scalar SONIC kernel requires one coordinate")
        result = self.potential.derivatives(float(values[0]))
        energy = result.energy if hasattr(result, "energy") else result.value
        return (float(energy), np.asarray([result.first]),
                None if direction is None else result.second * direction)


@dataclass(frozen=True)
class SmallBlockSonicKernel:
    """Existing fixed-arity angle/torsion kernels; never a global Hessian."""

    potential: object

    def evaluate(self, values, direction):
        if not 1 <= len(values) <= 8:
            raise ValueError("large SONIC terms require a matrix-free local kernel")
        result = self.potential.derivatives(*map(float, values))
        gradient = np.asarray(result.gradient, dtype=float)
        hessian = np.asarray(result.hessian, dtype=float)
        if hessian.shape != (len(values), len(values)) or not np.all(np.isfinite(hessian)):
            raise ValueError("invalid small-block SONIC Hessian")
        return result.energy, gradient, None if direction is None else hessian @ direction


@dataclass(frozen=True)
class FlapSonicKernel:
    """Ring bonds followed by flap components; sparse mixed couplings retained."""

    potential: object

    def evaluate(self, values, direction):
        bond_count = len(self.potential.ring_atoms)
        energy, gradient, product = self.potential.energy_gradient_hessian_vector(
            values[:bond_count], values[bond_count:], direction
        )
        return energy, gradient, None if direction is None else product


@dataclass(frozen=True)
class LocalSonicTerm:
    indices: tuple[int, ...]
    kernel: LocalSonicKernel


@dataclass(frozen=True)
class DirectSonicBondedValue:
    energy_hartree: float
    gradient_natural: np.ndarray
    hessian_vector_natural: np.ndarray | None
    execution: dict[str, object]


class DirectSonicBondedRuntime:
    """Accumulate local terms and apply the exact scalar-chart chain rule.

    Work/storage are O(number of variables + local kernel work). Linear
    scaling additionally requires a linear number of bounded-arity terms,
    or linear-storage ring kernels with sparse flap couplings. No chemical
    typing, Cartesian pair forces or LINK optimizer integration is implied.
    """

    def __init__(self, variables: SonicVariableMap, terms, *, energy_reference_hartree=0.0):
        self.variables = variables
        self.terms = tuple(terms)
        self.energy_reference = float(energy_reference_hartree)
        if not np.isfinite(self.energy_reference):
            raise ValueError("SONIC energy reference must be finite")
        self.indices = []
        for term in self.terms:
            raw = np.asarray(term.indices)
            if raw.ndim != 1 or raw.dtype.kind not in "iu" or len(raw) == 0:
                raise ValueError("local SONIC indices must be a nonempty integer vector")
            indices = raw.astype(np.intp, copy=True)
            if (len(set(indices)) != len(indices) or np.any(indices < 0)
                    or np.any(indices >= len(variables.variables))):
                raise ValueError("invalid or repeated local SONIC index")
            indices.setflags(write=False)
            self.indices.append(indices)
        self.indices = tuple(self.indices)

    def evaluate(self, natural_values, *, direction=None) -> DirectSonicBondedValue:
        physical, first, second = self.variables.inverse(natural_values)
        vector = None
        if direction is not None:
            vector = np.asarray(direction, dtype=float)
            if vector.shape != physical.shape or not np.all(np.isfinite(vector)):
                raise ValueError("invalid SONIC Hessian direction")
        physical_direction = None if vector is None else first * vector
        energy = self.energy_reference
        gradient = np.zeros_like(physical)
        product = None if vector is None else np.zeros_like(physical)
        for term, indices in zip(self.terms, self.indices, strict=True):
            local_energy, local_gradient, local_product = term.kernel.evaluate(
                physical[indices],
                None if physical_direction is None else physical_direction[indices],
            )
            local_gradient = np.asarray(local_gradient, dtype=float)
            if (not np.isfinite(local_energy) or local_gradient.shape != indices.shape
                    or not np.all(np.isfinite(local_gradient))):
                raise ValueError("invalid local SONIC energy/gradient")
            energy += local_energy
            gradient[indices] += local_gradient
            if product is not None:
                local_product = np.asarray(local_product, dtype=float)
                if local_product.shape != indices.shape or not np.all(np.isfinite(local_product)):
                    raise ValueError("invalid local SONIC Hessian product")
                product[indices] += local_product
        if product is not None:
            product = first * product + second * gradient * vector
        gradient = first * gradient
        if (not np.isfinite(energy) or not np.all(np.isfinite(gradient))
                or (product is not None and not np.all(np.isfinite(product)))):
            raise FloatingPointError("nonfinite natural-variable ZAFF result")
        return DirectSonicBondedValue(float(energy), gradient, product, {
            "coordinate_space": "LOCAL_SONIC_NATURAL",
            "term_count": len(self.terms), "cartesian_transformations": 0,
            "global_dense_matrices": 0, "analytic_hessian_vector": product is not None,
            "maximum_local_arity": max(map(len, self.indices), default=0),
        })


__all__ = ["DirectSonicBondedRuntime", "DirectSonicBondedValue", "LocalSonicTerm",
           "LocalSonicKernel", "ScalarSonicKernel", "SmallBlockSonicKernel", "FlapSonicKernel"]
