"""Exact sparse GFN-FF bonded and non-bonded repulsion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .native_kernels import (
    gfnff_repulsion_energy_gradient,
    gfnff_repulsion_hessian_vector,
    native_zaff_backend,
)
from .nonbonded import PersistentVerletNeighborList


@dataclass(frozen=True)
class GFNFFRepulsionResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    bonded_pair_count: int
    nonbonded_pair_count: int
    backend: str


@dataclass(frozen=True)
class GFNFFRepulsionModel:
    """Prepared molecular GFN-FF repulsion in native atomic units.

    ARCHITECT supplies the already typed bonded coefficients and per-atom
    non-bonded factors.  Only neighbor discovery and radial evaluation remain
    in the geometry loop.
    """

    bonded_pairs: np.ndarray
    bonded_amplitudes_hartree_bohr: np.ndarray
    bonded_exponents_per_bohr3over2: np.ndarray
    nonbonded_amplitude_factors: np.ndarray
    nonbonded_exponent_factors: np.ndarray
    nonbonded_class_codes: np.ndarray
    scale_override_pairs: np.ndarray
    scale_override_values: np.ndarray
    nonbonded_cutoff_bohr: float = 20.0

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "GFNFFRepulsionModel":
        if record.get("kind") != "GFNFF_REPULSION_R32":
            raise ValueError("unsupported GFN-FF repulsion record")
        return cls(
            bonded_pairs=np.asarray(record["bonded_pairs"], dtype=np.intp),
            bonded_amplitudes_hartree_bohr=np.asarray(
                record["bonded_amplitudes_hartree_bohr"], dtype=float
            ),
            bonded_exponents_per_bohr3over2=np.asarray(
                record["bonded_exponents_per_bohr3over2"], dtype=float
            ),
            nonbonded_amplitude_factors=np.asarray(
                record["nonbonded_amplitude_factors"], dtype=float
            ),
            nonbonded_exponent_factors=np.asarray(
                record["nonbonded_exponent_factors"], dtype=float
            ),
            nonbonded_class_codes=np.asarray(record["nonbonded_class_codes"], dtype=np.intp),
            scale_override_pairs=np.asarray(
                record.get("scale_override_pairs", ()), dtype=np.intp
            ).reshape(-1, 2),
            scale_override_values=np.asarray(record.get("scale_override_values", ()), dtype=float),
            nonbonded_cutoff_bohr=float(record.get("nonbonded_cutoff_bohr", 20.0)),
        )

    def __post_init__(self) -> None:
        arrays = {
            "bonded_pairs": np.asarray(self.bonded_pairs, dtype=np.intp).reshape(-1, 2),
            "bonded_amplitudes_hartree_bohr": np.asarray(
                self.bonded_amplitudes_hartree_bohr, dtype=float
            ).reshape(-1),
            "bonded_exponents_per_bohr3over2": np.asarray(
                self.bonded_exponents_per_bohr3over2, dtype=float
            ).reshape(-1),
            "nonbonded_amplitude_factors": np.asarray(
                self.nonbonded_amplitude_factors, dtype=float
            ).reshape(-1),
            "nonbonded_exponent_factors": np.asarray(
                self.nonbonded_exponent_factors, dtype=float
            ).reshape(-1),
            "nonbonded_class_codes": np.asarray(self.nonbonded_class_codes, dtype=np.intp).reshape(
                -1
            ),
            "scale_override_pairs": np.asarray(self.scale_override_pairs, dtype=np.intp).reshape(
                -1, 2
            ),
            "scale_override_values": np.asarray(self.scale_override_values, dtype=float).reshape(
                -1
            ),
        }
        natoms = len(arrays["nonbonded_amplitude_factors"])
        if natoms < 1 or any(
            len(arrays[name]) != natoms
            for name in ("nonbonded_exponent_factors", "nonbonded_class_codes")
        ):
            raise ValueError("GFN-FF repulsion atom arrays have inconsistent dimensions")
        if any(
            len(arrays[name]) != len(arrays["bonded_pairs"])
            for name in (
                "bonded_amplitudes_hartree_bohr",
                "bonded_exponents_per_bohr3over2",
            )
        ):
            raise ValueError("GFN-FF bonded-repulsion arrays have inconsistent dimensions")
        if len(arrays["scale_override_pairs"]) != len(arrays["scale_override_values"]):
            raise ValueError("GFN-FF repulsion scale overrides have inconsistent dimensions")
        pair_tables = (arrays["bonded_pairs"], arrays["scale_override_pairs"])
        if any(
            len(pairs)
            and (np.any(pairs < 0) or np.max(pairs) >= natoms or np.any(pairs[:, 0] == pairs[:, 1]))
            for pairs in pair_tables
        ):
            raise ValueError("GFN-FF repulsion pair index is invalid")
        if (
            np.any(~np.isfinite(arrays["bonded_amplitudes_hartree_bohr"]))
            or np.any(arrays["bonded_amplitudes_hartree_bohr"] <= 0.0)
            or np.any(~np.isfinite(arrays["bonded_exponents_per_bohr3over2"]))
            or np.any(arrays["bonded_exponents_per_bohr3over2"] <= 0.0)
            or np.any(~np.isfinite(arrays["nonbonded_amplitude_factors"]))
            or np.any(arrays["nonbonded_amplitude_factors"] <= 0.0)
            or np.any(~np.isfinite(arrays["nonbonded_exponent_factors"]))
            or np.any(arrays["nonbonded_exponent_factors"] <= 0.0)
            or np.any(~np.isfinite(arrays["scale_override_values"]))
            or np.any(arrays["scale_override_values"] <= 0.0)
            or not np.isfinite(self.nonbonded_cutoff_bohr)
            or self.nonbonded_cutoff_bohr <= 0.0
        ):
            raise ValueError("GFN-FF repulsion parameters are invalid")
        if np.any((arrays["nonbonded_class_codes"] < 0) | (arrays["nonbonded_class_codes"] > 4)):
            raise ValueError("GFN-FF repulsion class code is invalid")
        bonded_keys = [tuple(sorted(map(int, pair))) for pair in arrays["bonded_pairs"]]
        override_keys = [tuple(sorted(map(int, pair))) for pair in arrays["scale_override_pairs"]]
        if len(set(bonded_keys)) != len(bonded_keys) or len(set(override_keys)) != len(
            override_keys
        ):
            raise ValueError("GFN-FF repulsion pairs must be unique")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def natoms(self) -> int:
        return len(self.nonbonded_amplitude_factors)

    def nonbonded_parameters(
        self, candidate_pairs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pairs = np.asarray(candidate_pairs, dtype=np.intp).reshape(-1, 2)
        if not len(pairs):
            return pairs, np.empty(0), np.empty(0)
        bonded = {tuple(sorted(map(int, pair))) for pair in self.bonded_pairs}
        mask = np.fromiter(
            (tuple(sorted(map(int, pair))) not in bonded for pair in pairs),
            dtype=bool,
            count=len(pairs),
        )
        pairs = pairs[mask]
        left, right = pairs[:, 0], pairs[:, 1]
        amplitudes = (
            self.nonbonded_amplitude_factors[left] * self.nonbonded_amplitude_factors[right]
        )
        scales = _class_pair_scales(
            self.nonbonded_class_codes[left], self.nonbonded_class_codes[right]
        )
        overrides = {
            tuple(sorted(map(int, pair))): float(value)
            for pair, value in zip(
                self.scale_override_pairs, self.scale_override_values, strict=True
            )
        }
        if overrides:
            scales = np.fromiter(
                (
                    overrides.get(tuple(sorted(map(int, pair))), float(scale))
                    for pair, scale in zip(pairs, scales, strict=True)
                ),
                dtype=float,
                count=len(pairs),
            )
        exponents = (
            np.sqrt(self.nonbonded_exponent_factors[left] * self.nonbonded_exponent_factors[right])
            * scales
        )
        return pairs, amplitudes, exponents


class GFNFFRepulsionSession:
    """Persistent sparse repulsion evaluator for one compiled topology."""

    def __init__(
        self,
        model: GFNFFRepulsionModel,
        *,
        neighbor_skin_bohr: float = 2.0,
    ) -> None:
        self.model = model
        self.neighbor_list = PersistentVerletNeighborList(
            cutoff_bohr=model.nonbonded_cutoff_bohr,
            skin_bohr=float(neighbor_skin_bohr),
        )
        self.evaluation_count = 0
        self._prepared_candidates = None
        self._prepared_arrays = None
        self.parameter_rebuild_count = 0

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        neighbor_skin_bohr: float = 2.0,
    ) -> "GFNFFRepulsionSession":
        return cls(
            GFNFFRepulsionModel.from_record(record),
            neighbor_skin_bohr=neighbor_skin_bohr,
        )

    def _pair_arrays(
        self, coordinates_bohr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        xyz = _coordinates(coordinates_bohr, self.model.natoms)
        candidates = self.neighbor_list.native_candidates(xyz)
        if len(candidates):
            delta = xyz[candidates[:, 0]] - xyz[candidates[:, 1]]
            candidates = candidates[
                np.sum(delta * delta, axis=1) <= self.model.nonbonded_cutoff_bohr**2
            ]
        if self._prepared_candidates is not None and np.array_equal(
            candidates, self._prepared_candidates
        ):
            return self._prepared_arrays
        pairs, amplitudes, exponents = self.model.nonbonded_parameters(candidates)
        bonded_count = len(self.model.bonded_pairs)
        nonbonded_count = len(pairs)
        prepared = (
            np.concatenate((self.model.bonded_pairs, pairs), axis=0),
            np.concatenate((self.model.bonded_amplitudes_hartree_bohr, amplitudes)),
            np.concatenate((self.model.bonded_exponents_per_bohr3over2, exponents)),
            bonded_count,
            nonbonded_count,
        )
        self._prepared_candidates = candidates.copy()
        self._prepared_candidates.setflags(write=False)
        for array in prepared[:3]:
            array.setflags(write=False)
        self._prepared_arrays = prepared
        self.parameter_rebuild_count += 1
        return prepared

    def evaluate(self, coordinates_bohr: np.ndarray) -> GFNFFRepulsionResult:
        xyz = _coordinates(coordinates_bohr, self.model.natoms)
        pairs, amplitudes, exponents, bonded_count, nonbonded_count = self._pair_arrays(xyz)
        if native_zaff_backend(len(pairs)).accelerated:
            energy, gradient, _ = gfnff_repulsion_energy_gradient(xyz, pairs, amplitudes, exponents)
            backend = "CPP_SPARSE_VERLET"
        else:
            energy, gradient = _python_energy_gradient(xyz, pairs, amplitudes, exponents)
            backend = "PYTHON_SPARSE_VERLET"
        self.evaluation_count += 1
        return GFNFFRepulsionResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            bonded_pair_count=bonded_count,
            nonbonded_pair_count=nonbonded_count,
            backend=(
                backend
                + f"_R{self.neighbor_list.rebuild_count}"
                + f"_U{self.neighbor_list.reuse_count}"
            ),
        )

    def hessian_vector_product(
        self, coordinates_bohr: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        xyz = _coordinates(coordinates_bohr, self.model.natoms)
        direction = np.asarray(direction_bohr, dtype=float).reshape(xyz.shape)
        pairs, amplitudes, exponents, _, _ = self._pair_arrays(xyz)
        if native_zaff_backend(len(pairs)).accelerated:
            product, _ = gfnff_repulsion_hessian_vector(
                xyz, pairs, amplitudes, exponents, direction
            )
            return product.reshape(-1)
        return _python_hessian_vector(xyz, pairs, amplitudes, exponents, direction).reshape(-1)


def _class_pair_scales(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Codes are assigned by ARCHITECT: other, H, C, O, metal.
    scales = np.ones(len(left), dtype=float)
    hydrogen_left = left == 1
    hydrogen_right = right == 1
    scales[hydrogen_left & hydrogen_right] = 0.6290
    scales[(hydrogen_left & (right == 2)) | (hydrogen_right & (left == 2))] = 0.91
    scales[(hydrogen_left & (right == 3)) | (hydrogen_right & (left == 3))] = 1.04
    scales[(hydrogen_left & (right == 4)) | (hydrogen_right & (left == 4))] = 0.85
    return scales


def _coordinates(coordinates: np.ndarray, natoms: int) -> np.ndarray:
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape != (natoms, 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("GFN-FF repulsion coordinates have wrong dimensions")
    return xyz


def _radial_derivatives(
    distance: np.ndarray, amplitude: np.ndarray, exponent: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = np.sqrt(distance)
    r32 = distance * root
    energy = amplitude * np.exp(-exponent * r32) / distance
    log_first = -1.0 / distance - 1.5 * exponent * root
    log_second = 1.0 / distance**2 - 0.75 * exponent / root
    first = energy * log_first
    second = energy * (log_first**2 + log_second)
    return energy, first, second


def _python_energy_gradient(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    amplitudes: np.ndarray,
    exponents: np.ndarray,
) -> tuple[float, np.ndarray]:
    gradient = np.zeros_like(coordinates)
    if not len(pairs):
        return 0.0, gradient
    delta = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1.0e-14):
        raise ValueError("coincident atoms in GFN-FF repulsion pair")
    energy, first, _ = _radial_derivatives(distance, amplitudes, exponents)
    values = first[:, None] * delta / distance[:, None]
    np.add.at(gradient, pairs[:, 0], values)
    np.add.at(gradient, pairs[:, 1], -values)
    return float(np.sum(energy)), gradient


def _python_hessian_vector(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    amplitudes: np.ndarray,
    exponents: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    product = np.zeros_like(coordinates)
    if not len(pairs):
        return product
    delta = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1.0e-14):
        raise ValueError("coincident atoms in GFN-FF repulsion pair")
    unit = delta / distance[:, None]
    _, first, second = _radial_derivatives(distance, amplitudes, exponents)
    vector = direction[pairs[:, 0]] - direction[pairs[:, 1]]
    projection = np.sum(unit * vector, axis=1)
    values = (first / distance)[:, None] * vector + (second - first / distance)[
        :, None
    ] * projection[:, None] * unit
    np.add.at(product, pairs[:, 0], values)
    np.add.at(product, pairs[:, 1], -values)
    return product


__all__ = [
    "GFNFFRepulsionModel",
    "GFNFFRepulsionResult",
    "GFNFFRepulsionSession",
]
