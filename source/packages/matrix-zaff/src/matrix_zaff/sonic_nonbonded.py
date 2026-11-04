"""Resident Exp-PE/electrostatics complement to direct bonded SONIC ZAFF.

Cartesian geometry is still necessary for pairs. Fixed Gaussian charges use
a fused local kernel and FMM point Coulomb; response models may instead
provide their complete variational E/G/Hv without freezing their charges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .native_kernels import sonic_local_pairs
from .nonbonded import (
    BOHR_TO_ANGSTROM, PersistentVerletNeighborList,
    _direct_coulomb, _direct_coulomb_hessian_vector,
    _fmm_available, _fmm_coulomb, _fmm_coulomb_hessian_vector,
)


@dataclass(frozen=True)
class SonicNonbondedValue:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_vector_hartree_per_bohr: np.ndarray | None
    execution: dict[str, object]


class SonicNonbondedRuntime:
    """One persistent local pair list; exact existing Exp-PE parameter convention.

    epsilon and half-rmin use the existing geometric/additive atomic mixing
    rule and its Exp-PE form, not D3. This API does not silently map a
    custom synthon pair library to that rule. Distances enter in bohr;
    half-rmin and the Exp-PE cutoff are specified in Angstrom.
    """

    def __init__(self, epsilon_hartree, rmin_half_angstrom, *, cutoff_angstrom,
                 charges=None, widths_bohr=None, backend="fmm",
                 charge_response_operator=None, fmm_precision=1e-10,
                 neighbor_skin_bohr=2.):
        self.epsilon = self._parameters(epsilon_hartree)
        self.natoms = len(self.epsilon)
        if not self.natoms:
            raise ValueError("nonbonded model requires at least one atom")
        self.rmin = self._parameters(rmin_half_angstrom, self.natoms)
        if np.any(self.epsilon <= 0) or np.any(self.rmin <= 0):
            raise ValueError("Exp-PE epsilon and radius must be positive")
        self.cutoff = float(cutoff_angstrom)
        self.precision = float(fmm_precision)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0 or not 0 < self.precision < 1:
            raise ValueError("invalid nonbonded cutoff or precision")
        if backend not in {"fmm", "direct_reference", "external_response"}:
            raise ValueError("nonbonded backend must be fmm, direct_reference or external_response")
        if backend == "direct_reference" and self.natoms > 128:
            raise ValueError("quadratic direct electrostatics is limited to small reference tests")
        if backend == "fmm" and not _fmm_available():
            raise RuntimeError("FMM required; no quadratic fallback is permitted")
        self.backend = backend
        self.response = charge_response_operator
        if backend == "external_response":
            if self.response is None or charges is not None or widths_bohr is not None:
                raise ValueError("external response requires one complete E/G/Hv operator, not fixed charges")
            if not all(callable(getattr(self.response, name, None))
                       for name in ("evaluate", "hessian_vector_product")):
                raise ValueError("external response requires a complete E/G/Hv operator")
            self.charges, self.widths = np.zeros(self.natoms), np.ones(self.natoms)
            local_cutoff = self.cutoff / BOHR_TO_ANGSTROM
        else:
            if self.response is not None or charges is None or widths_bohr is None:
                raise ValueError("fixed Gaussian mode requires explicit charges and widths only")
            self.charges = self._parameters(charges, self.natoms)
            self.widths = self._parameters(widths_bohr, self.natoms)
            if np.any(self.widths <= 0):
                raise ValueError("Gaussian widths must be positive")
            # beta*r < 8; 16*max(width) bounds every pair's penetration range.
            local_cutoff = max(self.cutoff / BOHR_TO_ANGSTROM,
                               16 * float(np.max(self.widths, initial=0)))
        self.neighbors = PersistentVerletNeighborList(
            cutoff_bohr=local_cutoff, skin_bohr=neighbor_skin_bohr)

    @staticmethod
    def _parameters(values, count=None):
        array = np.array(values, dtype=float, copy=True)
        if (array.ndim != 1 or (count is not None and len(array) != count)
                or not np.all(np.isfinite(array))):
            raise ValueError("invalid nonbonded parameter vector")
        array.setflags(write=False)
        return array

    def evaluate(self, coordinates_bohr, *, direction_bohr=None):
        xyz = np.asarray(coordinates_bohr, dtype=float)
        if xyz.shape != (self.natoms, 3) or not np.all(np.isfinite(xyz)):
            raise ValueError("invalid nonbonded geometry")
        direction = None if direction_bohr is None else np.asarray(direction_bohr, dtype=float)
        if direction is not None and (direction.shape != xyz.shape or not np.all(np.isfinite(direction))):
            raise ValueError("invalid nonbonded direction")
        candidates = self.neighbors.native_candidates(xyz)
        energy, gradient, product, nexp, ngauss = sonic_local_pairs(
            xyz, self.charges, self.widths, self.epsilon, self.rmin,
            candidates, self.cutoff, direction)
        if self.backend == "external_response":
            # The operator owns the full charge energy and response, not just
            # E_Coulomb(q(x)). No separately counted Gaussian fixed-charge term.
            response = self.response.evaluate(xyz)
            long_energy = response.energy_hartree
            long_gradient = np.asarray(response.gradient_hartree_per_bohr).reshape(xyz.shape)
            long_product = (None if direction is None else
                            self.response.hessian_vector_product(xyz, direction))
        elif self.backend == "fmm":
            long_energy, long_gradient = _fmm_coulomb(xyz, self.charges, self.precision)
            long_product = (None if direction is None else
                            _fmm_coulomb_hessian_vector(xyz, self.charges, direction, self.precision))
        else:
            long_energy, long_gradient = _direct_coulomb(xyz, self.charges)
            long_product = (None if direction is None else
                            _direct_coulomb_hessian_vector(xyz, self.charges, direction))
        energy += long_energy
        gradient = np.asarray(gradient).reshape(-1) + np.asarray(long_gradient).reshape(-1)
        if direction is not None:
            response_product = np.asarray(long_product, dtype=float)
            if response_product.size != xyz.size:
                raise ValueError("electrostatic response returned an invalid Hessian product")
            product = np.asarray(product).reshape(-1) + response_product.reshape(-1)
        if (not np.isfinite(energy) or not np.all(np.isfinite(gradient))
                or (product is not None and not np.all(np.isfinite(product)))):
            raise FloatingPointError("invalid nonbonded energy/gradient/Hessian product")
        return SonicNonbondedValue(float(energy), gradient, product, {
            "electrostatic_backend": self.backend,
            "charge_mode": "complete_response" if self.response is not None else "fixed_gaussian",
            "shared_local_radial_geometry": self.response is None,
            "exppe_pair_count": nexp, "penetration_pair_count": ngauss if self.response is None else 0,
            "neighbor_rebuild_count": self.neighbors.rebuild_count,
            "neighbor_reuse_count": self.neighbors.reuse_count,
            "owned_global_dense_matrices": 0,
            "global_dense_matrices": 0 if self.response is None else None,
            "linear_scaling_certified": False,
        })
