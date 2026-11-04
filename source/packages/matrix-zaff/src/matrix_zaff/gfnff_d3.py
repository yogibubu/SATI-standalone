"""Sparse non-periodic GFN-FF two-body D3(BJ) contribution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping

import numpy as np

from .native_kernels import (
    gfnff_d3_reference_derivatives,
    native_zaff_backend,
)
from .nonbonded import PersistentVerletNeighborList


@dataclass(frozen=True)
class GFNFFD3Result:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    coordination_numbers: np.ndarray
    pair_count: int
    backend: str


@dataclass(frozen=True)
class _CoordinationState:
    values: np.ndarray
    raw: np.ndarray
    first_map: np.ndarray
    second_map: np.ndarray
    pair_distance: np.ndarray
    pair_unit: np.ndarray
    pair_first: np.ndarray
    pair_second: np.ndarray


@dataclass(frozen=True)
class GFNFFD3Model:
    """Prepared D3 layer using compact element-reference tables."""

    atomic_numbers: np.ndarray
    zeta_factors: np.ndarray
    sqrt_z_r4_over_r2: np.ndarray
    reference_cn_by_atomic_number: Mapping[int, np.ndarray]
    reference_c6_by_pair: Mapping[tuple[int, int], np.ndarray]
    coordination_pairs: np.ndarray
    coordination_radii_bohr: np.ndarray
    dispersion_pairs: np.ndarray
    cn_max: float = 4.4
    cn_steepness: float = 7.5
    weighting_factor: float = 4.0
    dispersion_scale: float = 1.0
    damping_a1: float = 0.58
    damping_a2: float = 4.80
    reference_counts: np.ndarray = field(init=False, repr=False, compare=False)
    reference_cn_padded: np.ndarray = field(init=False, repr=False, compare=False)
    pair_c6_padded: np.ndarray = field(init=False, repr=False, compare=False)

    @classmethod
    def from_records(
        cls,
        d3: Mapping[str, object],
        coordination: Mapping[str, object],
        coordination_pairs: np.ndarray,
        dispersion_pairs: np.ndarray,
    ) -> "GFNFFD3Model":
        if d3.get("kind") != "GFNFF_D3_BJ_TWO_BODY":
            raise ValueError("unsupported GFN-FF D3 record")
        if coordination.get("kind") != "GFNFF_LOG_ERF":
            raise ValueError("unsupported GFN-FF coordination record")
        references = {
            int(key): np.asarray(value, dtype=float)
            for key, value in dict(d3["reference_cn_by_atomic_number"]).items()
        }
        reference_c6: dict[tuple[int, int], np.ndarray] = {}
        for key, value in dict(d3["reference_c6_by_pair"]).items():
            left_text, right_text = str(key).split(":", maxsplit=1)
            reference_c6[(int(left_text), int(right_text))] = np.asarray(value, dtype=float)
        radii_angstrom = np.asarray(coordination["covalent_radii_angstrom"], dtype=float)
        bohr_to_angstrom = 0.52917721092
        cn_pairs = np.asarray(coordination_pairs, dtype=np.intp).reshape(-1, 2)
        return cls(
            atomic_numbers=np.asarray(d3["atomic_numbers"], dtype=np.intp),
            zeta_factors=np.asarray(d3["zeta_factors"], dtype=float),
            sqrt_z_r4_over_r2=np.asarray(d3["sqrt_z_r4_over_r2"], dtype=float),
            reference_cn_by_atomic_number=references,
            reference_c6_by_pair=reference_c6,
            coordination_pairs=cn_pairs,
            coordination_radii_bohr=(
                radii_angstrom[cn_pairs[:, 0]] + radii_angstrom[cn_pairs[:, 1]]
            )
            / bohr_to_angstrom,
            dispersion_pairs=np.asarray(dispersion_pairs, dtype=np.intp),
            cn_max=float(coordination.get("cn_max", 4.4)),
            cn_steepness=float(coordination.get("steepness", 7.5)),
            weighting_factor=float(d3.get("weighting_factor", 4.0)),
            dispersion_scale=float(d3.get("dispersion_scale", 1.0)),
            damping_a1=float(d3.get("damping_a1", 0.58)),
            damping_a2=float(d3.get("damping_a2", 4.80)),
        )

    def __post_init__(self) -> None:
        arrays = {
            "atomic_numbers": np.asarray(self.atomic_numbers, dtype=np.intp).reshape(-1),
            "zeta_factors": np.asarray(self.zeta_factors, dtype=float).reshape(-1),
            "sqrt_z_r4_over_r2": np.asarray(self.sqrt_z_r4_over_r2, dtype=float).reshape(-1),
            "coordination_pairs": np.asarray(self.coordination_pairs, dtype=np.intp).reshape(-1, 2),
            "coordination_radii_bohr": np.asarray(
                self.coordination_radii_bohr, dtype=float
            ).reshape(-1),
            "dispersion_pairs": np.asarray(self.dispersion_pairs, dtype=np.intp).reshape(-1, 2),
        }
        natoms = len(arrays["atomic_numbers"])
        if natoms < 1 or any(
            len(arrays[name]) != natoms for name in ("zeta_factors", "sqrt_z_r4_over_r2")
        ):
            raise ValueError("GFN-FF D3 atom arrays have inconsistent dimensions")
        if len(arrays["coordination_radii_bohr"]) != len(arrays["coordination_pairs"]):
            raise ValueError("GFN-FF D3 coordination arrays have inconsistent dimensions")
        if (
            np.any(arrays["atomic_numbers"] < 1)
            or np.any(~np.isfinite(arrays["zeta_factors"]))
            or np.any(arrays["zeta_factors"] < 0.0)
            or np.any(~np.isfinite(arrays["sqrt_z_r4_over_r2"]))
            or np.any(arrays["sqrt_z_r4_over_r2"] <= 0.0)
            or np.any(~np.isfinite(arrays["coordination_radii_bohr"]))
            or np.any(arrays["coordination_radii_bohr"] <= 0.0)
        ):
            raise ValueError("GFN-FF D3 atom parameters are invalid")
        for pairs in (arrays["coordination_pairs"], arrays["dispersion_pairs"]):
            if len(pairs) and (
                np.any(pairs < 0) or np.max(pairs) >= natoms or np.any(pairs[:, 0] == pairs[:, 1])
            ):
                raise ValueError("GFN-FF D3 pair index is invalid")
        unique_numbers = set(map(int, arrays["atomic_numbers"]))
        references: dict[int, np.ndarray] = {}
        for atomic_number in unique_numbers:
            if atomic_number not in self.reference_cn_by_atomic_number:
                raise ValueError("GFN-FF D3 reference CN table is incomplete")
            values = np.asarray(
                self.reference_cn_by_atomic_number[atomic_number], dtype=float
            ).reshape(-1)
            if not len(values) or np.any(~np.isfinite(values)):
                raise ValueError("GFN-FF D3 reference CN table is invalid")
            values.setflags(write=False)
            references[atomic_number] = values
        c6_tables: dict[tuple[int, int], np.ndarray] = {}
        for left in unique_numbers:
            for right in unique_numbers:
                if right < left:
                    continue
                key = (left, right)
                if key not in self.reference_c6_by_pair:
                    raise ValueError("GFN-FF D3 reference C6 table is incomplete")
                table = np.asarray(self.reference_c6_by_pair[key], dtype=float)
                expected = (len(references[left]), len(references[right]))
                if table.shape != expected or np.any(~np.isfinite(table)) or np.any(table <= 0.0):
                    raise ValueError("GFN-FF D3 reference C6 table is invalid")
                table.setflags(write=False)
                c6_tables[key] = table
        controls = (
            self.cn_max,
            self.cn_steepness,
            self.weighting_factor,
            self.dispersion_scale,
            self.damping_a1,
            self.damping_a2,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in controls):
            raise ValueError("GFN-FF D3 controls must be finite and positive")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "reference_cn_by_atomic_number", references)
        object.__setattr__(self, "reference_c6_by_pair", c6_tables)
        maximum_references = max(map(len, references.values()))
        reference_counts = np.empty(natoms, dtype=np.intp)
        reference_cn_padded = np.zeros((natoms, maximum_references))
        for number, atom_references in references.items():
            mask = arrays["atomic_numbers"] == number
            reference_counts[mask] = len(atom_references)
            reference_cn_padded[mask, : len(atom_references)] = atom_references
        pair_c6_padded = np.zeros(
            (len(arrays["dispersion_pairs"]), maximum_references, maximum_references)
        )
        # Each pair of element types shares one immutable C6 table. Scatter
        # by type in NumPy instead of a Python loop over every spatial pair.
        # The element alphabet is bounded independently of molecular size.
        pair_numbers = arrays["atomic_numbers"][arrays["dispersion_pairs"]]
        for (zi, zj), table in c6_tables.items():
            mask = (pair_numbers[:, 0] == zi) & (pair_numbers[:, 1] == zj)
            pair_c6_padded[mask, : table.shape[0], : table.shape[1]] = table
            if zi != zj:
                reverse = (pair_numbers[:, 0] == zj) & (pair_numbers[:, 1] == zi)
                pair_c6_padded[reverse, : table.shape[1], : table.shape[0]] = table.T
        for name, array in (
            ("reference_counts", reference_counts),
            ("reference_cn_padded", reference_cn_padded),
            ("pair_c6_padded", pair_c6_padded),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def natoms(self) -> int:
        return len(self.atomic_numbers)

    def evaluate(
        self, coordinates_bohr: np.ndarray, *, compute_gradient: bool = True
    ) -> GFNFFD3Result:
        xyz = self._coordinates(coordinates_bohr)
        state = self._coordination(xyz)
        return self._evaluate_with_coordination(xyz, state, compute_gradient=compute_gradient)

    def _evaluate_with_coordination(
        self, xyz: np.ndarray, state: _CoordinationState, *, compute_gradient: bool = True
    ) -> GFNFFD3Result:
        """Consume this geometry's CN state supplied by the complete runtime."""

        energy, gradient = self._energy_gradient(xyz, state, compute_gradient)
        return GFNFFD3Result(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            coordination_numbers=state.values,
            pair_count=len(self.dispersion_pairs),
            backend=(
                "CPP_SPARSE_CN_D3BJ"
                if native_zaff_backend(len(self.dispersion_pairs)).accelerated
                else "NUMPY_SPARSE_CN_D3BJ"
            ),
        )

    def hessian_vector_product(
        self, coordinates_bohr: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        xyz = self._coordinates(coordinates_bohr)
        direction = np.asarray(direction_bohr, dtype=float).reshape(xyz.shape)
        state = self._coordination(xyz)
        raw_dot, cn_dot = self._coordination_direction(state, direction)
        pairs = self.dispersion_pairs
        product = np.zeros_like(xyz)
        weights = np.zeros(self.natoms)
        weight_dot = np.zeros(self.natoms)
        if len(pairs):
            left, right = pairs[:, 0], pairs[:, 1]
            delta = xyz[left] - xyz[right]
            distance = np.linalg.norm(delta, axis=1)
            if np.any(distance <= 1.0e-14):
                raise ValueError("coincident atoms in GFN-FF D3 pair")
            unit = delta / distance[:, None]
            pair_vector = direction[left] - direction[right]
            rdot = np.sum(unit * pair_vector, axis=1)
            udot = (pair_vector - unit * rdot[:, None]) / distance[:, None]
            c6, c6_left, c6_right, c6_ll, c6_lr, c6_rr = self._pair_c6_derivatives(state.values)
            radial, radial_first, radial_second, scale = self._radial(distance)
            c6_dot = c6_left * cn_dot[left] + c6_right * cn_dot[right]
            pair_first = -scale * c6 * radial_first
            pair_first_dot = -scale * (c6_dot * radial_first + c6 * radial_second * rdot)
            values = pair_first_dot[:, None] * unit + pair_first[:, None] * udot
            np.add.at(product, left, values)
            np.add.at(product, right, -values)
            left_weight = -scale * radial * c6_left
            right_weight = -scale * radial * c6_right
            left_weight_dot = -scale * (
                radial_first * rdot * c6_left
                + radial * (c6_ll * cn_dot[left] + c6_lr * cn_dot[right])
            )
            right_weight_dot = -scale * (
                radial_first * rdot * c6_right
                + radial * (c6_lr * cn_dot[left] + c6_rr * cn_dot[right])
            )
            np.add.at(weights, left, left_weight)
            np.add.at(weights, right, right_weight)
            np.add.at(weight_dot, left, left_weight_dot)
            np.add.at(weight_dot, right, right_weight_dot)
        product += self._coordination_weighted_hvp(state, direction, raw_dot, weights, weight_dot)
        return product.reshape(-1)

    def _coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.shape != (self.natoms, 3) or np.any(~np.isfinite(xyz)):
            raise ValueError("GFN-FF D3 coordinates have wrong dimensions")
        return xyz

    def _energy_gradient(
        self, xyz: np.ndarray, state: _CoordinationState, compute_gradient: bool
    ) -> tuple[float, np.ndarray]:
        pairs = self.dispersion_pairs
        gradient = np.zeros_like(xyz)
        if not len(pairs):
            return 0.0, gradient
        left, right = pairs[:, 0], pairs[:, 1]
        delta = xyz[left] - xyz[right]
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 1.0e-14):
            raise ValueError("coincident atoms in GFN-FF D3 pair")
        c6, c6_left, c6_right, _, _, _ = self._pair_c6_derivatives(state.values)
        radial, radial_first, _, scale = self._radial(distance)
        energy = -scale * c6 * radial
        if compute_gradient:
            values = (-scale * c6 * radial_first)[:, None] * (delta / distance[:, None])
            np.add.at(gradient, left, values)
            np.add.at(gradient, right, -values)
            weights = np.zeros(self.natoms)
            np.add.at(weights, left, -scale * radial * c6_left)
            np.add.at(weights, right, -scale * radial * c6_right)
            gradient += self._coordination_weighted_gradient(state, weights)
        return float(np.sum(energy)), gradient

    def _radial(
        self, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pairs = self.dispersion_pairs
        left, right = pairs[:, 0], pairs[:, 1]
        r4_product = self.sqrt_z_r4_over_r2[left] * self.sqrt_z_r4_over_r2[right]
        damping = self.damping_a1 * np.sqrt(3.0 * r4_product) + self.damping_a2
        scale = self.dispersion_scale * self.zeta_factors[left] * self.zeta_factors[right]
        radial = np.zeros(len(distance))
        first = np.zeros(len(distance))
        second = np.zeros(len(distance))
        for power, coefficient in ((6, np.ones(len(distance))), (8, 6.0 * r4_product)):
            denominator = distance**power + damping**power
            inverse = 1.0 / denominator
            radial += coefficient * inverse
            first += -coefficient * power * distance ** (power - 1) * inverse**2
            second += coefficient * (
                -power * (power - 1) * distance ** (power - 2) * inverse**2
                + 2.0 * power**2 * distance ** (2 * power - 2) * inverse**3
            )
        return radial, first, second, scale

    def _reference_weights(
        self, atomic_number: int, coordination: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        references = self.reference_cn_by_atomic_number[int(atomic_number)]
        log_first = 2.0 * self.weighting_factor * (references - coordination)
        log_weight = -self.weighting_factor * (coordination - references) ** 2
        log_weight -= float(np.max(log_weight))
        weights = np.exp(log_weight)
        weights /= float(np.sum(weights))
        mean = float(weights @ log_first)
        first = weights * (log_first - mean)
        variance = float(weights @ (log_first - mean) ** 2)
        second = weights * ((log_first - mean) ** 2 - variance)
        return weights, first, second

    def _pair_c6_derivatives(
        self, coordination: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if native_zaff_backend(len(self.dispersion_pairs)).accelerated:
            return gfnff_d3_reference_derivatives(
                coordination,
                self.dispersion_pairs,
                self.reference_counts,
                self.reference_cn_padded,
                self.pair_c6_padded,
                self.weighting_factor,
            )
        weights = [
            self._reference_weights(int(z), float(cn))
            for z, cn in zip(self.atomic_numbers, coordination, strict=True)
        ]
        output = tuple(np.empty(len(self.dispersion_pairs)) for _ in range(6))
        for index, (left, right) in enumerate(self.dispersion_pairs):
            zi = int(self.atomic_numbers[left])
            zj = int(self.atomic_numbers[right])
            key = (min(zi, zj), max(zi, zj))
            table = self.reference_c6_by_pair[key]
            if zi > zj:
                table = table.T
            wi, dwi, d2wi = weights[int(left)]
            wj, dwj, d2wj = weights[int(right)]
            output[0][index] = wi @ table @ wj
            output[1][index] = dwi @ table @ wj
            output[2][index] = wi @ table @ dwj
            output[3][index] = d2wi @ table @ wj
            output[4][index] = dwi @ table @ dwj
            output[5][index] = wi @ table @ d2wj
        return output

    def _coordination(self, xyz: np.ndarray) -> _CoordinationState:
        pairs = self.coordination_pairs
        delta = xyz[pairs[:, 0]] - xyz[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 1.0e-14):
            raise ValueError("coincident atoms in GFN-FF coordination pair")
        unit = delta / distance[:, None]
        reduced = (distance - self.coordination_radii_bohr) / self.coordination_radii_bohr
        gaussian = np.exp(-((self.cn_steepness * reduced) ** 2))
        values = np.fromiter(
            (0.5 * (1.0 - math.erf(self.cn_steepness * value)) for value in reduced),
            dtype=float,
            count=len(reduced),
        )
        first = -self.cn_steepness * gaussian / (math.sqrt(math.pi) * self.coordination_radii_bohr)
        second = (
            2.0
            * self.cn_steepness**3
            * (distance - self.coordination_radii_bohr)
            * gaussian
            / (math.sqrt(math.pi) * self.coordination_radii_bohr**3)
        )
        raw = np.zeros(self.natoms)
        np.add.at(raw, pairs[:, 0], values)
        np.add.at(raw, pairs[:, 1], values)
        cn = np.logaddexp(0.0, self.cn_max) - np.logaddexp(0.0, self.cn_max - raw)
        first_map = np.exp(-np.logaddexp(0.0, raw - self.cn_max))
        second_map = -first_map * (1.0 - first_map)
        return _CoordinationState(cn, raw, first_map, second_map, distance, unit, first, second)

    def _coordination_direction(
        self, state: _CoordinationState, direction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pairs = self.coordination_pairs
        pair_vector = direction[pairs[:, 0]] - direction[pairs[:, 1]]
        rdot = np.sum(state.pair_unit * pair_vector, axis=1)
        raw_dot = np.zeros(self.natoms)
        contribution = state.pair_first * rdot
        np.add.at(raw_dot, pairs[:, 0], contribution)
        np.add.at(raw_dot, pairs[:, 1], contribution)
        return raw_dot, state.first_map * raw_dot

    def _coordination_weighted_gradient(
        self, state: _CoordinationState, weights: np.ndarray
    ) -> np.ndarray:
        pairs = self.coordination_pairs
        pair_weight = state.pair_first * (
            weights[pairs[:, 0]] * state.first_map[pairs[:, 0]]
            + weights[pairs[:, 1]] * state.first_map[pairs[:, 1]]
        )
        gradient = np.zeros((self.natoms, 3))
        values = pair_weight[:, None] * state.pair_unit
        np.add.at(gradient, pairs[:, 0], values)
        np.add.at(gradient, pairs[:, 1], -values)
        return gradient

    def _coordination_weighted_hvp(
        self,
        state: _CoordinationState,
        direction: np.ndarray,
        raw_dot: np.ndarray,
        weights: np.ndarray,
        weight_dot: np.ndarray,
    ) -> np.ndarray:
        pairs = self.coordination_pairs
        left, right = pairs[:, 0], pairs[:, 1]
        pair_vector = direction[left] - direction[right]
        rdot = np.sum(state.pair_unit * pair_vector, axis=1)
        udot = (pair_vector - state.pair_unit * rdot[:, None]) / state.pair_distance[:, None]
        saturated = weights[left] * state.first_map[left] + weights[right] * state.first_map[right]
        saturated_dot = (
            weight_dot[left] * state.first_map[left]
            + weights[left] * state.second_map[left] * raw_dot[left]
            + weight_dot[right] * state.first_map[right]
            + weights[right] * state.second_map[right] * raw_dot[right]
        )
        pair_weight = state.pair_first * saturated
        pair_weight_dot = state.pair_second * rdot * saturated + state.pair_first * saturated_dot
        values = pair_weight_dot[:, None] * state.pair_unit + pair_weight[:, None] * udot
        product = np.zeros((self.natoms, 3))
        np.add.at(product, left, values)
        np.add.at(product, right, -values)
        return product


class GFNFFD3Session:
    """Persistent two-list D3 session with no atom-pair matrices."""

    def __init__(
        self,
        d3_record: Mapping[str, object],
        coordination_record: Mapping[str, object],
        *,
        neighbor_skin_bohr: float = 2.0,
    ) -> None:
        self.d3_record = dict(d3_record)
        self.coordination_record = dict(coordination_record)
        bohr_to_angstrom = 0.52917721092
        self.coordination_cutoff_bohr = (
            float(coordination_record["cutoff_angstrom"]) / bohr_to_angstrom
        )
        self.dispersion_cutoff_bohr = float(d3_record["cutoff_bohr"])
        self.coordination_neighbors = PersistentVerletNeighborList(
            cutoff_bohr=self.coordination_cutoff_bohr,
            skin_bohr=float(neighbor_skin_bohr),
        )
        self.dispersion_neighbors = PersistentVerletNeighborList(
            cutoff_bohr=self.dispersion_cutoff_bohr,
            skin_bohr=float(neighbor_skin_bohr),
        )
        self.evaluation_count = 0

    @classmethod
    def from_model(
        cls, model: Mapping[str, object], *, neighbor_skin_bohr: float = 2.0
    ) -> "GFNFFD3Session":
        d3 = model.get("gfnff_d3_two_body")
        coordination = model.get("gfnff_coordination")
        if not isinstance(d3, Mapping) or not isinstance(coordination, Mapping):
            raise ValueError("compiled model lacks GFN-FF D3 or coordination data")
        return cls(d3, coordination, neighbor_skin_bohr=neighbor_skin_bohr)

    def _prepared_model(self, coordinates_bohr: np.ndarray) -> GFNFFD3Model:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        cn_pairs = self._filtered_pairs(
            self.coordination_neighbors, xyz, self.coordination_cutoff_bohr
        )
        d3_pairs = self._filtered_pairs(self.dispersion_neighbors, xyz, self.dispersion_cutoff_bohr)
        return GFNFFD3Model.from_records(
            self.d3_record, self.coordination_record, cn_pairs, d3_pairs
        )

    @staticmethod
    def _filtered_pairs(
        neighbors: PersistentVerletNeighborList,
        coordinates: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        candidates = neighbors.native_candidates(coordinates)
        if not len(candidates):
            return candidates
        delta = coordinates[candidates[:, 0]] - coordinates[candidates[:, 1]]
        return candidates[np.sum(delta * delta, axis=1) <= cutoff**2]

    def evaluate(self, coordinates_bohr: np.ndarray) -> GFNFFD3Result:
        result = self._prepared_model(coordinates_bohr).evaluate(coordinates_bohr)
        self.evaluation_count += 1
        return replace(
            result,
            backend=(
                result.backend
                + f"_CNR{self.coordination_neighbors.rebuild_count}"
                + f"_CNU{self.coordination_neighbors.reuse_count}"
                + f"_D3R{self.dispersion_neighbors.rebuild_count}"
                + f"_D3U{self.dispersion_neighbors.reuse_count}"
            ),
        )

    def hessian_vector_product(
        self, coordinates_bohr: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        return self._prepared_model(coordinates_bohr).hessian_vector_product(
            coordinates_bohr, direction_bohr
        )


__all__ = ["GFNFFD3Model", "GFNFFD3Result", "GFNFFD3Session"]
