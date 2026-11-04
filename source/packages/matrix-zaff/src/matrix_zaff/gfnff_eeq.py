"""Matrix-free non-periodic GFN-FF electronegativity equilibration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal, Mapping

import numpy as np

from .charge_response import ChargeResponseSite, GaussianChargeResponseOperator
from .native_kernels import (
    direct_gaussian_directional_potential,
    direct_gaussian_potential,
    native_zaff_backend,
)
from .nonbonded import (
    BOHR_TO_ANGSTROM,
    PersistentVerletNeighborList,
    electrostatic_energy_gradient,
    electrostatic_hessian_vector_product,
)


@dataclass(frozen=True)
class GFNFFEEQResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    charges: np.ndarray
    coordination_numbers: np.ndarray
    iterations: int
    residual_norm: float
    backend: str


class GFNFFEEQSession:
    """Resident EEQ session with a persistent sparse CN pair list."""

    def __init__(
        self,
        eeq_record: Mapping[str, object],
        coordination_record: Mapping[str, object],
        *,
        neighbor_skin_bohr: float = 2.0,
    ) -> None:
        if eeq_record.get("kind") != "GFNFF_EEQ_GAUSSIAN":
            raise ValueError("unsupported GFN-FF EEQ record")
        if coordination_record.get("kind") != "GFNFF_LOG_ERF":
            raise ValueError("unsupported GFN-FF coordination record")
        self.eeq_record = dict(eeq_record)
        self.coordination_record = dict(coordination_record)
        self.cutoff_bohr = (
            float(coordination_record["cutoff_angstrom"]) / BOHR_TO_ANGSTROM
        )
        self.neighbor_list = PersistentVerletNeighborList(
            cutoff_bohr=self.cutoff_bohr,
            skin_bohr=float(neighbor_skin_bohr),
        )
        self.evaluation_count = 0

    @classmethod
    def from_model(
        cls,
        model: Mapping[str, object],
        *,
        neighbor_skin_bohr: float = 2.0,
    ) -> "GFNFFEEQSession":
        eeq = model.get("gfnff_eeq")
        coordination = model.get("gfnff_coordination")
        if not isinstance(eeq, Mapping) or not isinstance(coordination, Mapping):
            raise ValueError("compiled model lacks GFN-FF EEQ or coordination data")
        return cls(eeq, coordination, neighbor_skin_bohr=neighbor_skin_bohr)

    def _prepared_model(self, coordinates_bohr: np.ndarray) -> "GFNFFEEQModel":
        xyz = np.asarray(coordinates_bohr, dtype=float)
        candidates = self.neighbor_list.native_candidates(xyz)
        if len(candidates):
            delta = xyz[candidates[:, 0]] - xyz[candidates[:, 1]]
            pairs = candidates[
                np.sum(delta * delta, axis=1) <= self.cutoff_bohr**2
            ]
        else:
            pairs = candidates
        return GFNFFEEQModel.from_records(
            self.eeq_record, self.coordination_record, pairs
        )

    def evaluate(
        self,
        coordinates_bohr: np.ndarray,
        *,
        compute_gradient: bool = True,
    ) -> GFNFFEEQResult:
        result = self._prepared_model(coordinates_bohr).evaluate(
            coordinates_bohr, compute_gradient=compute_gradient
        )
        self.evaluation_count += 1
        return replace(
            result,
            backend=(
                result.backend
                + f"_VERLET_R{self.neighbor_list.rebuild_count}"
                + f"_U{self.neighbor_list.reuse_count}"
            ),
        )

    def hessian_vector_product(
        self, coordinates_bohr: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        return self._prepared_model(coordinates_bohr).hessian_vector_product(
            coordinates_bohr, direction_bohr
        )


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


class _GaussianEEQOperator:
    def __init__(
        self,
        coordinates_bohr: np.ndarray,
        widths_bohr: np.ndarray,
        gamma_hartree: np.ndarray,
        *,
        backend: Literal["auto", "direct", "fmm"],
        fmm_precision: float,
        fmm_minimum_atoms: int,
    ) -> None:
        self.coordinates = np.asarray(coordinates_bohr, dtype=float)
        self.widths = np.asarray(widths_bohr, dtype=float)
        self.gamma = np.asarray(gamma_hartree, dtype=float)
        selected = backend
        if selected == "auto":
            try:
                import fmm3dpy  # noqa: F401
            except ImportError:
                available = False
            else:
                available = True
            selected = (
                "fmm"
                if available and len(self.coordinates) >= int(fmm_minimum_atoms)
                else "direct"
            )
        if selected not in {"direct", "fmm"}:
            raise ValueError("GFN-FF EEQ backend must be auto, direct, or fmm")
        self.selected = selected
        self.fmm_operator = None
        if selected == "fmm":
            sites = tuple(
                ChargeResponseSite(0.0, width, ((index, 1.0),), f"EEQ{index + 1}")
                for index, width in enumerate(self.widths)
            )
            self.fmm_operator = GaussianChargeResponseOperator.compile(
                self.coordinates,
                sites,
                backend="fmm",
                fmm_precision=float(fmm_precision),
                fmm_minimum_sites=0,
            )

    @property
    def diagonal(self) -> np.ndarray:
        return 1.0 / (math.sqrt(math.pi) * self.widths) + self.gamma

    def matrix_vector(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if self.fmm_operator is not None:
            return self.fmm_operator.matrix_vector(values) + self.gamma * values
        if native_zaff_backend(len(values)).accelerated:
            return direct_gaussian_potential(
                self.coordinates, values, self.widths
            ) + self.gamma * values
        result = self.diagonal * values
        for left in range(1, len(values)):
            for right in range(left):
                delta = self.coordinates[left] - self.coordinates[right]
                distance = float(np.linalg.norm(delta))
                if distance <= 1.0e-14:
                    raise ValueError("coincident atoms in GFN-FF EEQ")
                beta = 1.0 / math.sqrt(
                    2.0 * (self.widths[left] ** 2 + self.widths[right] ** 2)
                )
                kernel = math.erf(beta * distance) / distance
                result[left] += kernel * values[right]
                result[right] += kernel * values[left]
        return result

    def directional_matrix_vector(
        self, vector: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        direction = np.asarray(direction_bohr, dtype=float).reshape(-1, 3)
        if self.fmm_operator is not None:
            return self.fmm_operator.directional_matrix_vector(values, direction)
        if native_zaff_backend(len(values)).accelerated:
            return direct_gaussian_directional_potential(
                self.coordinates, values, self.widths, direction
            )
        result = np.zeros_like(values)
        for left in range(1, len(values)):
            for right in range(left):
                delta = self.coordinates[left] - self.coordinates[right]
                distance = float(np.linalg.norm(delta))
                if distance <= 1.0e-14:
                    raise ValueError("coincident atoms in GFN-FF EEQ")
                beta = 1.0 / math.sqrt(
                    2.0 * (self.widths[left] ** 2 + self.widths[right] ** 2)
                )
                argument = beta * distance
                first = (
                    2.0
                    * beta
                    * math.exp(-(argument**2))
                    / (math.sqrt(math.pi) * distance)
                    - math.erf(argument) / distance**2
                )
                rdot = float(
                    delta @ (direction[left] - direction[right]) / distance
                )
                result[left] += first * rdot * values[right]
                result[right] += first * rdot * values[left]
        return result


@dataclass(frozen=True)
class GFNFFEEQModel:
    """Prepared GFN-FF EEQ model with projected matrix-free charge solve."""

    electronegativities_hartree: np.ndarray
    cn_coefficients_hartree: np.ndarray
    gaussian_alpha_bohr2: np.ndarray
    hardness_hartree: np.ndarray
    fragment_ids: np.ndarray
    fragment_charges: np.ndarray
    coordination_pairs: np.ndarray
    coordination_radii_bohr: np.ndarray
    cn_max: float = 4.4
    steepness: float = 7.5
    tolerance: float = 1.0e-11
    maximum_iterations: int = 500
    backend: Literal["auto", "direct", "fmm"] = "auto"
    fmm_precision: float = 1.0e-10
    fmm_minimum_atoms: int = 256

    @classmethod
    def from_records(
        cls,
        eeq: Mapping[str, object],
        coordination: Mapping[str, object],
        coordination_pairs: np.ndarray,
    ) -> "GFNFFEEQModel":
        radii = np.asarray(coordination["covalent_radii_angstrom"], dtype=float)
        bohr_to_angstrom = 0.52917721092
        pairs = np.asarray(coordination_pairs, dtype=np.intp).reshape(-1, 2)
        return cls(
            electronegativities_hartree=np.asarray(
                eeq["electronegativities_hartree"], dtype=float
            ),
            cn_coefficients_hartree=np.asarray(
                eeq["cn_coefficients_hartree"], dtype=float
            ),
            gaussian_alpha_bohr2=np.asarray(eeq["gaussian_alpha_bohr2"], dtype=float),
            hardness_hartree=np.asarray(eeq["hardness_hartree"], dtype=float),
            fragment_ids=np.asarray(eeq["fragment_ids"], dtype=np.intp),
            fragment_charges=np.asarray(eeq["fragment_charges"], dtype=float),
            coordination_pairs=pairs,
            coordination_radii_bohr=(
                radii[pairs[:, 0]] + radii[pairs[:, 1]]
            )
            / bohr_to_angstrom,
            cn_max=float(coordination.get("cn_max", 4.4)),
            steepness=float(coordination.get("steepness", 7.5)),
            tolerance=float(eeq.get("tolerance", 1.0e-11)),
            maximum_iterations=int(eeq.get("maximum_iterations", 500)),
            backend=str(eeq.get("backend", "auto")),
            fmm_precision=float(eeq.get("fmm_precision", 1.0e-10)),
            fmm_minimum_atoms=int(eeq.get("fmm_minimum_atoms", 256)),
        )

    def __post_init__(self) -> None:
        arrays = {
            "electronegativities_hartree": np.asarray(
                self.electronegativities_hartree, dtype=float
            ).reshape(-1),
            "cn_coefficients_hartree": np.asarray(
                self.cn_coefficients_hartree, dtype=float
            ).reshape(-1),
            "gaussian_alpha_bohr2": np.asarray(
                self.gaussian_alpha_bohr2, dtype=float
            ).reshape(-1),
            "hardness_hartree": np.asarray(self.hardness_hartree, dtype=float).reshape(-1),
            "fragment_ids": np.asarray(self.fragment_ids, dtype=np.intp).reshape(-1),
            "fragment_charges": np.asarray(self.fragment_charges, dtype=float).reshape(-1),
            "coordination_pairs": np.asarray(
                self.coordination_pairs, dtype=np.intp
            ).reshape(-1, 2),
            "coordination_radii_bohr": np.asarray(
                self.coordination_radii_bohr, dtype=float
            ).reshape(-1),
        }
        natoms = len(arrays["electronegativities_hartree"])
        if any(len(arrays[name]) != natoms for name in (
            "cn_coefficients_hartree", "gaussian_alpha_bohr2",
            "hardness_hartree", "fragment_ids",
        )):
            raise ValueError("GFN-FF EEQ atom arrays have inconsistent dimensions")
        if len(arrays["coordination_radii_bohr"]) != len(arrays["coordination_pairs"]):
            raise ValueError("GFN-FF EEQ coordination arrays have inconsistent dimensions")
        if (
            np.any(~np.isfinite(arrays["electronegativities_hartree"]))
            or np.any(~np.isfinite(arrays["cn_coefficients_hartree"]))
            or np.any(~np.isfinite(arrays["gaussian_alpha_bohr2"]))
            or np.any(arrays["gaussian_alpha_bohr2"] <= 0.0)
            or np.any(~np.isfinite(arrays["hardness_hartree"]))
            or np.any(~np.isfinite(arrays["fragment_charges"]))
            or np.any(~np.isfinite(arrays["coordination_radii_bohr"]))
            or np.any(arrays["coordination_radii_bohr"] <= 0.0)
        ):
            raise ValueError("GFN-FF EEQ parameters are invalid")
        eeq_diagonal = (
            np.sqrt(2.0 / np.pi) / np.sqrt(arrays["gaussian_alpha_bohr2"])
            + arrays["hardness_hartree"]
        )
        if np.any(eeq_diagonal <= 0.0):
            raise ValueError("GFN-FF EEQ self-interaction diagonal must be positive")
        fragments = arrays["fragment_ids"]
        if np.any(fragments < 0) or set(fragments.tolist()) != set(range(len(arrays["fragment_charges"]))):
            raise ValueError("GFN-FF EEQ fragment identifiers must be contiguous")
        pairs = arrays["coordination_pairs"]
        if len(pairs) and (
            np.any(pairs < 0)
            or np.max(pairs) >= natoms
            or np.any(pairs[:, 0] == pairs[:, 1])
        ):
            raise ValueError("GFN-FF EEQ coordination pair is invalid")
        if self.backend not in {"auto", "direct", "fmm"}:
            raise ValueError("GFN-FF EEQ backend must be auto, direct, or fmm")
        if self.tolerance <= 0.0 or self.maximum_iterations < 1:
            raise ValueError("GFN-FF EEQ solver controls are invalid")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def evaluate(
        self,
        coordinates_bohr: np.ndarray,
        *,
        compute_gradient: bool = True,
    ) -> GFNFFEEQResult:
        xyz = self._coordinates(coordinates_bohr)
        state = self._coordination(xyz)
        return self._evaluate_with_coordination(xyz, state, compute_gradient=compute_gradient)

    def _evaluate_with_coordination(
        self, xyz: np.ndarray, state: _CoordinationState, *, compute_gradient: bool = True
    ) -> GFNFFEEQResult:
        """Consume this geometry's CN state supplied by the complete runtime."""

        chi = self.electronegativities_hartree + self.cn_coefficients_hartree * np.sqrt(
            state.values
        )
        operator = self._operator(xyz)
        charges, iterations, residual = self._solve(operator, chi, self.fragment_charges)
        hq = operator.matrix_vector(charges)
        energy = 0.5 * float(charges @ hq) - float(charges @ chi)
        gradient = np.zeros_like(xyz)
        if compute_gradient:
            gradient = self._electrostatic_gradient(xyz, charges, operator.selected)
            dchi = self.cn_coefficients_hartree / (
                2.0 * np.sqrt(state.values) + 1.0e-16
            )
            gradient += self._coordination_weighted_gradient(
                state, -charges * dchi
            )
        return GFNFFEEQResult(
            energy,
            gradient.reshape(-1),
            charges,
            state.values,
            iterations,
            residual,
            f"{operator.selected.upper()}_PROJECTED_PCG_GAUSSIAN_EEQ",
        )

    def hessian_vector_product(
        self, coordinates_bohr: np.ndarray, direction_bohr: np.ndarray
    ) -> np.ndarray:
        xyz = self._coordinates(coordinates_bohr)
        direction = np.asarray(direction_bohr, dtype=float).reshape(xyz.shape)
        state = self._coordination(xyz)
        raw_dot, cn_dot = self._coordination_direction(state, direction)
        sqrt_cn = np.sqrt(state.values)
        dchi = self.cn_coefficients_hartree / (2.0 * sqrt_cn + 1.0e-16)
        d2chi = -self.cn_coefficients_hartree / (
            4.0 * state.values * sqrt_cn + 1.0e-16
        )
        chi = self.electronegativities_hartree + self.cn_coefficients_hartree * sqrt_cn
        operator = self._operator(xyz)
        charges, _iterations, _residual = self._solve(
            operator, chi, self.fragment_charges
        )
        rhs_dot = dchi * cn_dot - operator.directional_matrix_vector(
            charges, direction
        )
        charge_dot, _iterations, _residual = self._solve(
            operator, rhs_dot, np.zeros_like(self.fragment_charges)
        )
        fixed = electrostatic_hessian_vector_product(
            xyz,
            charges,
            direction,
            electrostatic_gaussian_widths_bohr=np.sqrt(
                0.5 * self.gaussian_alpha_bohr2
            ),
            backend=operator.selected,
            fmm_precision=self.fmm_precision,
            fmm_minimum_atoms=0,
        ).reshape(xyz.shape)
        cross = (
            self._electrostatic_gradient(xyz, charges + charge_dot, operator.selected)
            - self._electrostatic_gradient(xyz, charges, operator.selected)
            - self._electrostatic_gradient(xyz, charge_dot, operator.selected)
        )
        weights = -charges * dchi
        weight_dot = -charge_dot * dchi - charges * d2chi * cn_dot
        cn_product = self._coordination_weighted_hvp(
            state, direction, raw_dot, weights, weight_dot
        )
        return (fixed + cross + cn_product).reshape(-1)

    def _coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.shape != (len(self.electronegativities_hartree), 3) or np.any(~np.isfinite(xyz)):
            raise ValueError("GFN-FF EEQ coordinates have wrong dimensions")
        return xyz

    def _operator(self, xyz: np.ndarray) -> _GaussianEEQOperator:
        return _GaussianEEQOperator(
            xyz,
            np.sqrt(0.5 * self.gaussian_alpha_bohr2),
            self.hardness_hartree,
            backend=self.backend,
            fmm_precision=self.fmm_precision,
            fmm_minimum_atoms=self.fmm_minimum_atoms,
        )

    def _project(self, vector: np.ndarray, counts: np.ndarray | None = None) -> np.ndarray:
        result = np.asarray(vector, dtype=float).copy()
        if counts is None:
            counts = np.bincount(self.fragment_ids, minlength=len(self.fragment_charges))
        sums = np.bincount(
            self.fragment_ids, weights=result, minlength=len(self.fragment_charges)
        )
        result -= (sums / counts)[self.fragment_ids]
        return result

    def _solve(
        self,
        operator: _GaussianEEQOperator,
        rhs: np.ndarray,
        fragment_charges: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        counts = np.bincount(self.fragment_ids, minlength=len(self.fragment_charges))
        affine = fragment_charges[self.fragment_ids] / counts[self.fragment_ids]
        residual = self._project(rhs - operator.matrix_vector(affine), counts)
        target = self.tolerance * max(1.0, float(np.linalg.norm(rhs)))
        if float(np.linalg.norm(residual)) <= target:
            return affine, 0, float(np.linalg.norm(residual))
        diagonal = operator.diagonal
        preconditioned = self._project(residual / diagonal, counts)
        direction = preconditioned.copy()
        rz = float(residual @ preconditioned)
        correction = np.zeros_like(rhs)
        for iteration in range(1, self.maximum_iterations + 1):
            product = self._project(operator.matrix_vector(direction), counts)
            denominator = float(direction @ product)
            if denominator <= 0.0 or not math.isfinite(denominator):
                raise ValueError("GFN-FF EEQ projected matrix is not positive definite")
            step = rz / denominator
            correction += step * direction
            residual -= step * product
            norm = float(np.linalg.norm(residual))
            if norm <= target:
                charges = affine + self._project(correction, counts)
                charge_error = np.bincount(
                    self.fragment_ids,
                    weights=charges,
                    minlength=len(fragment_charges),
                ) - fragment_charges
                if float(np.max(np.abs(charge_error), initial=0.0)) > 20.0 * self.tolerance:
                    raise RuntimeError("GFN-FF EEQ charge constraint was not preserved")
                return charges, iteration, norm
            next_preconditioned = self._project(residual / diagonal, counts)
            next_rz = float(residual @ next_preconditioned)
            direction = next_preconditioned + (next_rz / rz) * direction
            rz = next_rz
        raise RuntimeError("GFN-FF EEQ projected PCG did not converge")

    def _coordination(self, xyz: np.ndarray) -> _CoordinationState:
        pairs = self.coordination_pairs
        delta = xyz[pairs[:, 0]] - xyz[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 1.0e-14):
            raise ValueError("coincident atoms in GFN-FF coordination pair")
        unit = delta / distance[:, None]
        reduced = (distance - self.coordination_radii_bohr) / self.coordination_radii_bohr
        gaussian = np.exp(-(self.steepness * reduced) ** 2)
        values = np.fromiter(
            (0.5 * (1.0 - math.erf(self.steepness * value)) for value in reduced),
            dtype=float,
            count=len(reduced),
        )
        first = -self.steepness * gaussian / (
            math.sqrt(math.pi) * self.coordination_radii_bohr
        )
        second = (
            2.0
            * self.steepness**3
            * (distance - self.coordination_radii_bohr)
            * gaussian
            / (math.sqrt(math.pi) * self.coordination_radii_bohr**3)
        )
        raw = np.zeros(len(xyz))
        np.add.at(raw, pairs[:, 0], values)
        np.add.at(raw, pairs[:, 1], values)
        cn = np.logaddexp(0.0, self.cn_max) - np.logaddexp(0.0, self.cn_max - raw)
        first_map = np.exp(-np.logaddexp(0.0, raw - self.cn_max))
        second_map = -first_map * (1.0 - first_map)
        return _CoordinationState(
            cn, raw, first_map, second_map, distance, unit, first, second
        )

    def _coordination_direction(
        self, state: _CoordinationState, direction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pairs = self.coordination_pairs
        pair_vector = direction[pairs[:, 0]] - direction[pairs[:, 1]]
        rdot = np.sum(state.pair_unit * pair_vector, axis=1)
        raw_dot = np.zeros(len(state.values))
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
        gradient = np.zeros((len(weights), 3))
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
        udot = (
            pair_vector - state.pair_unit * rdot[:, None]
        ) / state.pair_distance[:, None]
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
        product = np.zeros((len(weights), 3))
        np.add.at(product, left, values)
        np.add.at(product, right, -values)
        return product

    def _electrostatic_gradient(
        self, xyz: np.ndarray, charges: np.ndarray, backend: str
    ) -> np.ndarray:
        return electrostatic_energy_gradient(
            xyz,
            charges,
            electrostatic_gaussian_widths_bohr=np.sqrt(
                0.5 * self.gaussian_alpha_bohr2
            ),
            backend=backend,
            fmm_precision=self.fmm_precision,
            fmm_minimum_atoms=0,
        ).gradient_hartree_per_bohr.reshape(xyz.shape)


__all__ = ["GFNFFEEQModel", "GFNFFEEQResult", "GFNFFEEQSession"]
