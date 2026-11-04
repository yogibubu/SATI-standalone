"""Size-consistent split-charge response on atomic and virtual sites."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Literal, Sequence, TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import erf

from .electrostatic_exclusions import GaussianExcludedPairs, canonical_pairs

if TYPE_CHECKING:
    from matrix_epoc.cpcm import CPCMReactionField


ZAFF_CHARGE_RESPONSE_SCHEMA = "matrix.zaff.split_charge_response.v1"
_SPARSE_INCIDENCE_MINIMUM_CHANNELS = 256


@dataclass(frozen=True)
class ChargeResponseSite:
    """Gaussian charge site with an exact affine atom-coordinate map."""

    reference_charge: float
    gaussian_width_bohr: float
    atom_weights: tuple[tuple[int, float], ...]
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.reference_charge)):
            raise ValueError("reference site charge must be finite")
        if not math.isfinite(float(self.gaussian_width_bohr)) or self.gaussian_width_bohr <= 0:
            raise ValueError("Gaussian site width must be finite and positive")
        if not self.atom_weights:
            raise ValueError("a charge-response site needs atom weights")
        indices = tuple(int(index) for index, _weight in self.atom_weights)
        weights = tuple(float(weight) for _index, weight in self.atom_weights)
        if min(indices) < 0 or len(indices) != len(set(indices)):
            raise ValueError("charge-response atom indices must be unique and nonnegative")
        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError("charge-response atom weights must be finite")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("charge-response affine weights must sum to one")


@dataclass(frozen=True)
class SplitChargeChannel:
    """One local, smoothly extinguished charge-flow degree of freedom."""

    left_site: int
    right_site: int
    hardness_hartree: float
    switch_radius_bohr: float
    switch_exponent: int = 6
    reference_bias_hartree: float = 0.0

    def __post_init__(self) -> None:
        if self.left_site == self.right_site or min(self.left_site, self.right_site) < 0:
            raise ValueError("split-charge channel endpoints must be distinct")
        if not math.isfinite(float(self.hardness_hartree)) or self.hardness_hartree <= 0:
            raise ValueError("split-charge hardness must be finite and positive")
        if not math.isfinite(float(self.switch_radius_bohr)) or self.switch_radius_bohr <= 0:
            raise ValueError("split-charge switch radius must be finite and positive")
        if int(self.switch_exponent) < 2 or int(self.switch_exponent) % 2:
            raise ValueError("split-charge switch exponent must be a positive even integer")
        if not math.isfinite(float(self.reference_bias_hartree)):
            raise ValueError("split-charge reference bias must be finite")


@dataclass(frozen=True)
class SplitChargeResponseResult:
    energy_correction_hartree: float
    gradient_correction_hartree_per_bohr: np.ndarray
    site_charges: np.ndarray
    charge_flows: np.ndarray
    iterations: int
    residual_norm: float
    backend: str = "DIRECT_DENSE_GAUSSIAN"
    penetration_pair_count: int = 0
    schema: str = ZAFF_CHARGE_RESPONSE_SCHEMA
    channel_count: int = 0
    flows_materialized: bool = True


@dataclass(frozen=True)
class PersistentSplitChargeResponse:
    """Resident static QEq/SQE contract for repeated geometry evaluations."""

    sites: tuple[ChargeResponseSite, ...]
    channels: tuple[SplitChargeChannel, ...]
    tolerance: float = 1.0e-11
    maximum_iterations: int = 500
    backend: Literal["auto", "direct", "fmm"] = "auto"
    fmm_precision: float = 1.0e-10
    fmm_minimum_sites: int = 256
    reaction_field: CPCMReactionField | None = None
    excluded_pairs: tuple[tuple[int, int], ...] = ()
    _channel_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = field(
        init=False, repr=False, compare=False
    )
    _reference: np.ndarray = field(init=False, repr=False, compare=False)
    _hardness: np.ndarray = field(init=False, repr=False, compare=False)
    _bias: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_pairs", canonical_pairs(len(self.sites), self.excluded_pairs))
        channel_data = _channel_arrays(self.channels)
        arrays = (
            *channel_data,
            np.asarray(
                [channel.hardness_hartree for channel in self.channels],
                dtype=float,
            ),
            np.asarray(
                [channel.reference_bias_hartree for channel in self.channels],
                dtype=float,
            ),
            np.asarray(
                [site.reference_charge for site in self.sites],
                dtype=float,
            ),
        )
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "_channel_data", arrays[:4])
        object.__setattr__(self, "_hardness", arrays[4])
        object.__setattr__(self, "_bias", arrays[5])
        object.__setattr__(self, "_reference", arrays[6])

    def solve(
        self,
        coordinates_bohr: np.ndarray,
        *,
        compute_gradient: bool = True,
    ) -> SplitChargeResponseResult:
        return solve_split_charge_response(
            coordinates_bohr,
            self.sites,
            self.channels,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations,
            backend=self.backend,
            fmm_precision=self.fmm_precision,
            fmm_minimum_sites=self.fmm_minimum_sites,
            reaction_field=self.reaction_field,
            compute_gradient=compute_gradient,
            _prepared=self,
        )

    def hessian_vector_product(
        self,
        coordinates_bohr: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        return split_charge_response_hessian_vector_product(
            coordinates_bohr,
            self.sites,
            self.channels,
            vector_bohr,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations,
            backend=self.backend,
            fmm_precision=self.fmm_precision,
            fmm_minimum_sites=self.fmm_minimum_sites,
            reaction_field=self.reaction_field,
            _prepared=self,
        )


@dataclass(frozen=True)
class PersistentAllPairSplitChargeResponse:
    """Exact all-pair SQE response with linear auxiliary storage.

    Pair hardnesses, switches and reference biases are generated in bounded
    chunks.  The variational flow variables are eliminated analytically and
    the resulting site-space system is solved matrix-free.
    """

    sites: tuple[ChargeResponseSite, ...]
    response_lengths: np.ndarray
    rmin_half_angstrom: np.ndarray
    reference_coordinates_bohr: np.ndarray
    tolerance: float = 1.0e-11
    maximum_iterations: int = 500
    backend: Literal["auto", "direct", "fmm"] = "auto"
    fmm_precision: float = 1.0e-10
    fmm_minimum_sites: int = 256
    reaction_field: CPCMReactionField | None = None
    switch_minimum_angstrom: float = 4.0
    switch_scale: float = 1.25
    switch_exponent: int = 6
    pair_chunk_size: int = 65536
    excluded_pairs: tuple[tuple[int, int], ...] = ()
    _reference: np.ndarray = field(init=False, repr=False, compare=False)
    _reference_site_xyz: np.ndarray = field(init=False, repr=False, compare=False)
    _reference_potential: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        count = len(self.sites)
        object.__setattr__(self, "excluded_pairs", canonical_pairs(count, self.excluded_pairs))
        response = np.asarray(self.response_lengths, dtype=float).reshape(-1)
        rmin = np.asarray(self.rmin_half_angstrom, dtype=float).reshape(-1)
        coordinates = np.asarray(self.reference_coordinates_bohr, dtype=float)
        if (
            response.shape != (count,)
            or rmin.shape != (count,)
            or np.any(~np.isfinite(response))
            or np.any(response <= 0.0)
            or np.any(~np.isfinite(rmin))
            or np.any(rmin <= 0.0)
            or int(self.switch_exponent) < 2
            or int(self.switch_exponent) % 2
            or int(self.pair_chunk_size) < 1
        ):
            raise ValueError("invalid implicit all-pair charge-response parameters")
        site_xyz, _weights = _site_geometry(coordinates, self.sites)
        reference = np.asarray(
            [site.reference_charge for site in self.sites], dtype=float
        )
        operator = GaussianChargeResponseOperator.compile(
            site_xyz,
            self.sites,
            excluded_pairs=self.excluded_pairs,
            backend=self.backend,
            fmm_precision=self.fmm_precision,
            fmm_minimum_sites=self.fmm_minimum_sites,
            reaction_field=self.reaction_field,
        )
        potential = operator.matrix_vector(reference)
        for array in (response, rmin, coordinates, site_xyz, reference, potential):
            array.setflags(write=False)
        object.__setattr__(self, "response_lengths", response)
        object.__setattr__(self, "rmin_half_angstrom", rmin)
        object.__setattr__(self, "reference_coordinates_bohr", coordinates)
        object.__setattr__(self, "_reference_site_xyz", site_xyz)
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_reference_potential", potential)

    @property
    def channel_count(self) -> int:
        count = len(self.sites)
        return count * (count - 1) // 2

    def solve(
        self,
        coordinates_bohr: np.ndarray,
        *,
        compute_gradient: bool = True,
    ) -> SplitChargeResponseResult:
        return _solve_implicit_all_pair_response(
            self, coordinates_bohr, compute_gradient=compute_gradient
        )

    def hessian_vector_product(
        self,
        coordinates_bohr: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        return _implicit_all_pair_hessian_vector_product(
            self, coordinates_bohr, vector_bohr
        )


@dataclass(frozen=True)
class PersistentLocalPairSplitChargeResponse(PersistentAllPairSplitChargeResponse):
    """Compact-support SQE response with bounded-density channel storage.

    The original rational response switch is multiplied by a C2 quintic
    taper before a fixed outer cutoff.  Consequently channels can enter or
    leave the resident Verlet list with continuous energy, gradient, and
    Hessian-vector products, while both stored pairs and work remain linear
    for bounded-density systems.
    """

    channel_cutoff_angstrom: float = 12.0
    taper_start_fraction: float = 0.75
    neighbor_skin_bohr: float = 2.0
    _channel_neighbors: Any = field(init=False, repr=False, compare=False)
    _reference_channel_count: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        cutoff = float(self.channel_cutoff_angstrom)
        fraction = float(self.taper_start_fraction)
        skin = float(self.neighbor_skin_bohr)
        maximum_radius = max(
            float(self.switch_minimum_angstrom),
            2.0 * float(self.switch_scale) * float(np.max(self.rmin_half_angstrom)),
        )
        if (
            not math.isfinite(cutoff)
            or cutoff <= maximum_radius
            or not math.isfinite(fraction)
            or not 0.0 < fraction < 1.0
            or fraction * cutoff <= maximum_radius
            or not math.isfinite(skin)
            or skin <= 0.0
        ):
            raise ValueError("invalid local split-charge cutoff/taper contract")
        super().__post_init__()
        from .nonbonded import PersistentVerletNeighborList

        neighbors = PersistentVerletNeighborList(
            cutoff_bohr=cutoff / 0.52917721092,
            skin_bohr=skin,
        )
        object.__setattr__(self, "_channel_neighbors", neighbors)
        object.__setattr__(
            self,
            "_reference_channel_count",
            len(self.active_pairs(self._reference_site_xyz)),
        )

    @property
    def channel_count(self) -> int:
        return self._reference_channel_count

    @property
    def linear_scaling_certified(self) -> bool:
        return True

    def active_pairs(self, site_xyz_bohr: np.ndarray) -> np.ndarray:
        xyz = np.asarray(site_xyz_bohr, dtype=float)
        candidates = self._channel_neighbors.native_candidates(xyz)
        if not len(candidates):
            return candidates
        delta = xyz[candidates[:, 0]] - xyz[candidates[:, 1]]
        cutoff_bohr = float(self.channel_cutoff_angstrom) / 0.52917721092
        return candidates[np.sum(delta * delta, axis=1) <= cutoff_bohr**2]


@dataclass(frozen=True)
class GaussianChargeResponseOperator:
    """Matrix-free Gaussian Coulomb plus optional CPCM response operator."""

    site_xyz_bohr: np.ndarray
    widths_bohr: np.ndarray
    diagonal: np.ndarray
    penetration_pairs: tuple[tuple[int, int], ...]
    backend: str
    fmm_precision: float
    penetration_pair_array: np.ndarray | None = None
    dense_matrix: np.ndarray | None = None
    reaction_field: CPCMReactionField | None = None
    reaction_operator: Any | None = None
    exclusions: GaussianExcludedPairs | None = None
    excluded_pair_set: frozenset = frozenset()

    @classmethod
    def compile(
        cls,
        site_xyz_bohr: np.ndarray,
        sites: Sequence[ChargeResponseSite],
        *,
        backend: Literal["auto", "direct", "fmm"] = "auto",
        fmm_precision: float = 1.0e-10,
        fmm_minimum_sites: int = 256,
        reaction_field: CPCMReactionField | None = None,
        excluded_pairs: tuple[tuple[int, int], ...] = (),
    ) -> "GaussianChargeResponseOperator":
        xyz = np.asarray(site_xyz_bohr, dtype=float)
        widths = np.asarray(
            [site.gaussian_width_bohr for site in sites], dtype=float
        )
        if xyz.shape != (len(sites), 3) or np.any(~np.isfinite(xyz)):
            raise ValueError("Gaussian response sites and coordinates are inconsistent")
        if backend not in {"auto", "direct", "fmm"}:
            raise ValueError("charge-response backend must be auto, direct, or fmm")
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
                if len(sites) >= int(fmm_minimum_sites) and available
                else "direct"
            )
        if selected == "fmm":
            try:
                import fmm3dpy  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "the FMM charge-response backend requires "
                    "matrix-zaff[fmm]"
                ) from exc
        diagonal = 1.0 / (math.sqrt(math.pi) * widths)
        excluded_pairs = canonical_pairs(len(sites), excluded_pairs)
        exclusions = GaussianExcludedPairs(xyz, widths, excluded_pairs) if excluded_pairs else None
        if selected == "direct":
            dense = _gaussian_coulomb_matrix(xyz, sites)
            if exclusions is not None:
                dense[exclusions.left, exclusions.right] = 0.0
                dense[exclusions.right, exclusions.left] = 0.0
            pairs: tuple[tuple[int, int], ...] = ()
        else:
            dense = None
            from .nonbonded import build_nonbonded_neighbor_list

            cutoff = 16.0 * float(np.max(widths))
            candidates = build_nonbonded_neighbor_list(xyz, cutoff_bohr=cutoff)
            pairs = tuple(
                (left, right)
                for left, right in candidates
                if _gaussian_beta(widths[left], widths[right])
                * float(np.linalg.norm(xyz[left] - xyz[right]))
                < 8.0
            )
        reaction_operator = (
            reaction_field.compile_persistent_operator(xyz)
            if reaction_field is not None
            and hasattr(reaction_field, "compile_persistent_operator")
            else None
        )
        return cls(
            site_xyz_bohr=xyz,
            widths_bohr=widths,
            diagonal=diagonal,
            penetration_pairs=pairs,
            penetration_pair_array=np.asarray(
                pairs, dtype=np.intp
            ).reshape(-1, 2),
            backend=f"{selected.upper()}_GAUSSIAN"
            + ("_WITH_CPCM" if reaction_field is not None else ""),
            fmm_precision=float(fmm_precision),
            dense_matrix=dense,
            reaction_field=reaction_field,
            reaction_operator=reaction_operator,
            exclusions=exclusions,
            excluded_pair_set=frozenset(excluded_pairs),
        )

    def matrix_vector(self, charges: np.ndarray) -> np.ndarray:
        q = np.asarray(charges, dtype=float).reshape(-1)
        if len(q) != len(self.site_xyz_bohr):
            raise ValueError("Gaussian response vector has the wrong dimension")
        if self.dense_matrix is not None:
            potential = self.dense_matrix @ q
        else:
            from fmm3dpy import lfmm3d
            from .nonbonded import FOUR_PI

            result = lfmm3d(
                eps=float(self.fmm_precision),
                sources=np.asfortranarray(self.site_xyz_bohr.T),
                charges=np.asfortranarray(q),
                pg=1,
            )
            potential = (
                FOUR_PI * np.asarray(result.pot, dtype=float).reshape(-1)
                + self.diagonal * q
            )
            from .native_kernels import (
                gaussian_correction_potential,
                native_zaff_backend,
            )

            if native_zaff_backend(len(q)).accelerated:
                correction, _count = gaussian_correction_potential(
                    self.site_xyz_bohr,
                    q,
                    self.widths_bohr,
                    (
                        self.penetration_pair_array
                        if self.penetration_pair_array is not None
                        else np.asarray(
                            self.penetration_pairs, dtype=np.intp
                        ).reshape(-1, 2)
                    ),
                )
                potential += correction
            else:
                for left, right in self.penetration_pairs:
                    distance = float(
                        np.linalg.norm(
                            self.site_xyz_bohr[left]
                            - self.site_xyz_bohr[right]
                        )
                    )
                    gaussian, _ = _gaussian_pair(
                        distance,
                        self.widths_bohr[left],
                        self.widths_bohr[right],
                    )
                    correction = gaussian - 1.0 / distance
                    potential[left] += correction * q[right]
                    potential[right] += correction * q[left]
        if self.reaction_field is not None:
            reaction = (
                self.reaction_operator.matrix_vector(q)
                if self.reaction_operator is not None
                else self.reaction_field.kernel_product(self.site_xyz_bohr, q)
            )
            potential = potential + reaction
        if self.exclusions is not None and self.dense_matrix is None:
            potential -= self.exclusions.matrix_vector(q)
        return potential

    def directional_matrix_vector(
        self, charges: np.ndarray, site_direction_bohr: np.ndarray
    ) -> np.ndarray:
        """Differentiate ``J(R) q`` for fixed charges along a site direction."""

        q = np.asarray(charges, dtype=float).reshape(-1)
        direction = np.asarray(site_direction_bohr, dtype=float)
        if direction.shape != self.site_xyz_bohr.shape or len(q) != len(direction):
            raise ValueError("Gaussian response directional dimensions are inconsistent")
        if self.dense_matrix is not None:
            derivative = _gaussian_coulomb_directional_derivative(
                self.site_xyz_bohr,
                direction,
                tuple(
                    ChargeResponseSite(0.0, width, ((0, 1.0),))
                    for width in self.widths_bohr
                ),
            ) @ q
        else:
            from fmm3dpy import lfmm3d
            from .nonbonded import FOUR_PI

            sources = np.asfortranarray(self.site_xyz_bohr.T)
            charge_result = lfmm3d(
                eps=float(self.fmm_precision),
                sources=sources,
                charges=np.asfortranarray(q),
                pg=2,
            )
            dipole_result = lfmm3d(
                eps=float(self.fmm_precision),
                sources=sources,
                dipvec=np.asfortranarray((q[:, None] * direction).T),
                pg=1,
            )
            gradient = (
                FOUR_PI
                * np.asarray(charge_result.grad, dtype=float).reshape(3, -1).T
            )
            derivative = np.einsum("nd,nd->n", gradient, direction)
            derivative += FOUR_PI * np.asarray(
                dipole_result.pot, dtype=float
            ).reshape(-1)
            for left, right in self.penetration_pairs:
                delta = self.site_xyz_bohr[left] - self.site_xyz_bohr[right]
                distance = float(np.linalg.norm(delta))
                distance_dot = float(
                    delta @ (direction[left] - direction[right]) / distance
                )
                _, gaussian_first = _gaussian_pair(
                    distance, self.widths_bohr[left], self.widths_bohr[right]
                )
                correction_dot = (
                    gaussian_first + 1.0 / distance**2
                ) * distance_dot
                derivative[left] += correction_dot * q[right]
                derivative[right] += correction_dot * q[left]
        if self.reaction_field is not None:
            derivative += (
                self.reaction_operator.directional_matrix_vector(q, direction)
                if self.reaction_operator is not None
                else self.reaction_field.kernel_directional_product(
                    self.site_xyz_bohr, q, direction
                )
            )
        if self.exclusions is not None:
            derivative -= self.exclusions.matrix_vector(q, direction)
        return derivative

    def channel_curvature(
        self, left: int, right: int, scale: float
    ) -> float:
        distance = float(
            np.linalg.norm(self.site_xyz_bohr[left] - self.site_xyz_bohr[right])
        )
        pair, _ = _gaussian_pair(
            distance, self.widths_bohr[left], self.widths_bohr[right]
        )
        if (min(left, right), max(left, right)) in self.excluded_pair_set:
            pair = 0.0
        value = float(scale) ** 2 * (
            self.diagonal[left] + self.diagonal[right] - 2.0 * pair
        )
        if self.reaction_field is not None:
            value += self.reaction_field.charge_flow_curvature(
                self.site_xyz_bohr, left, right, scale
            )
        return value

    def response_energy_gradient_difference(
        self, charges: np.ndarray, reference: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return ``E(q)-E(q0)`` and its fixed-stationary-charge gradient."""

        from .nonbonded import electrostatic_energy_gradient

        q = np.asarray(charges, dtype=float)
        q0 = np.asarray(reference, dtype=float)
        selected = "fmm" if self.dense_matrix is None else "direct"
        current = electrostatic_energy_gradient(
            self.site_xyz_bohr,
            q,
            electrostatic_gaussian_widths_bohr=self.widths_bohr,
            backend=selected,
            fmm_precision=self.fmm_precision,
            fmm_minimum_atoms=0,
        )
        baseline = electrostatic_energy_gradient(
            self.site_xyz_bohr,
            q0,
            electrostatic_gaussian_widths_bohr=self.widths_bohr,
            backend=selected,
            fmm_precision=self.fmm_precision,
            fmm_minimum_atoms=0,
        )
        self_energy = 0.5 * float(np.dot(self.diagonal, q**2 - q0**2))
        energy = current.energy_hartree - baseline.energy_hartree + self_energy
        gradient = (
            current.gradient_hartree_per_bohr
            - baseline.gradient_hartree_per_bohr
        ).reshape(-1, 3)
        if self.reaction_field is not None:
            reaction, reaction_gradient = self.reaction_field.reaction_energy_gradient(
                self.site_xyz_bohr, q
            )
            reference_reaction, reference_gradient = (
                self.reaction_field.reaction_energy_gradient(
                    self.site_xyz_bohr, q0
                )
            )
            energy += reaction - reference_reaction
            gradient += reaction_gradient - reference_gradient
        if self.exclusions is not None:
            excluded_energy, excluded_gradient = self.exclusions.energy_gradient(q)
            reference_energy, reference_gradient = self.exclusions.energy_gradient(q0)
            energy -= excluded_energy - reference_energy
            gradient -= excluded_gradient - reference_gradient
        return energy, gradient

    def relaxed_gradient_directional(
        self,
        charges: np.ndarray,
        reference: np.ndarray,
        charge_dot: np.ndarray,
        site_direction_bohr: np.ndarray,
    ) -> np.ndarray:
        """Differentiate the response gradient, including relaxed charges."""

        from .nonbonded import (
            electrostatic_energy_gradient,
            electrostatic_hessian_vector_product,
        )

        q = np.asarray(charges, dtype=float)
        q0 = np.asarray(reference, dtype=float)
        qdot = np.asarray(charge_dot, dtype=float)
        direction = np.asarray(site_direction_bohr, dtype=float)
        selected = "fmm" if self.dense_matrix is None else "direct"
        common = {
            "electrostatic_gaussian_widths_bohr": self.widths_bohr,
            "backend": selected,
            "fmm_precision": self.fmm_precision,
            "fmm_minimum_atoms": 0,
        }
        fixed = electrostatic_hessian_vector_product(
            self.site_xyz_bohr, q, direction, **common
        ).reshape(-1, 3)
        fixed -= electrostatic_hessian_vector_product(
            self.site_xyz_bohr, q0, direction, **common
        ).reshape(-1, 3)
        plus = electrostatic_energy_gradient(
            self.site_xyz_bohr, q + qdot, **common
        ).gradient_hartree_per_bohr.reshape(-1, 3)
        minus = electrostatic_energy_gradient(
            self.site_xyz_bohr, q - qdot, **common
        ).gradient_hartree_per_bohr.reshape(-1, 3)
        product = fixed + 0.5 * (plus - minus)
        if self.exclusions is not None:
            product -= self.exclusions.hessian_vector(q, direction)
            product += self.exclusions.hessian_vector(q0, direction)
            product -= 0.5 * (self.exclusions.energy_gradient(q + qdot)[1]
                              - self.exclusions.energy_gradient(q - qdot)[1])
        if self.reaction_field is not None:
            product += self.reaction_field.hessian_vector_product(
                self.site_xyz_bohr, q, direction
            ).reshape(-1, 3)
            product -= self.reaction_field.hessian_vector_product(
                self.site_xyz_bohr, q0, direction
            ).reshape(-1, 3)
            _, plus_gradient = self.reaction_field.reaction_energy_gradient(
                self.site_xyz_bohr, q + qdot
            )
            _, minus_gradient = self.reaction_field.reaction_energy_gradient(
                self.site_xyz_bohr, q - qdot
            )
            product += 0.5 * (plus_gradient - minus_gradient)
        return product


def solve_split_charge_response(
    coordinates_bohr: np.ndarray,
    sites: Sequence[ChargeResponseSite],
    channels: Sequence[SplitChargeChannel],
    *,
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 500,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_sites: int = 256,
    reaction_field: CPCMReactionField | None = None,
    compute_gradient: bool = True,
    excluded_pairs: tuple[tuple[int, int], ...] = (),
    _prepared: PersistentSplitChargeResponse | None = None,
) -> SplitChargeResponseResult:
    """Minimize the local QEq/SQE functional with exact charge conservation."""

    atom_xyz = np.asarray(coordinates_bohr, dtype=float)
    if atom_xyz.ndim != 2 or atom_xyz.shape[1] != 3 or np.any(~np.isfinite(atom_xyz)):
        raise ValueError("charge-response coordinates must have shape (natoms, 3)")
    site_xyz, weights = _site_geometry(atom_xyz, sites)
    reference = (
        _prepared._reference
        if _prepared is not None
        else np.asarray([site.reference_charge for site in sites], dtype=float)
    )
    if not channels:
        return SplitChargeResponseResult(
            0.0,
            np.zeros(atom_xyz.size),
            reference,
            np.zeros(0),
            0,
            0.0,
        )
    if max(max(channel.left_site, channel.right_site) for channel in channels) >= len(sites):
        raise ValueError("split-charge channel site index is out of range")
    operator = GaussianChargeResponseOperator.compile(
        site_xyz,
        sites,
        excluded_pairs=_prepared.excluded_pairs if _prepared is not None else excluded_pairs,
        backend=backend,
        fmm_precision=fmm_precision,
        fmm_minimum_sites=fmm_minimum_sites,
        reaction_field=reaction_field,
    )
    channel_data = _prepared._channel_data if _prepared is not None else None
    incidence, switches, switch_derivatives = _incidence_matrix(
        site_xyz, channels, channel_data
    )
    hardness = (
        _prepared._hardness
        if _prepared is not None
        else np.asarray([channel.hardness_hartree for channel in channels])
    )
    bias = (
        _prepared._bias
        if _prepared is not None
        else np.asarray([channel.reference_bias_hartree for channel in channels])
    )
    right_hand_side = bias - incidence.T @ operator.matrix_vector(reference)

    diagonal_inverse = 1.0 / np.maximum(
        hardness
        + _channel_curvatures(operator, channels, switches, channel_data),
        1.0e-12,
    )
    flows, iterations, residual, solver_backend = _solve_flow_system(
        operator,
        incidence,
        hardness,
        right_hand_side,
        diagonal_inverse,
        float(tolerance),
        int(maximum_iterations),
    )
    delta_charge = incidence @ flows
    charges = reference + delta_charge
    if compute_gradient:
        response_energy, site_gradient = operator.response_energy_gradient_difference(
            charges, reference
        )
    else:
        response_energy = 0.5 * float(
            charges @ operator.matrix_vector(charges)
            - reference @ operator.matrix_vector(reference)
        )
        site_gradient = np.zeros_like(site_xyz)
    energy = (
        response_energy
        + 0.5 * float(np.sum(hardness * flows**2))
        - float(bias @ flows)
    )
    if compute_gradient:
        potential = operator.matrix_vector(charges)
        left, right, _radii, _exponents = (
            channel_data if channel_data is not None else _channel_arrays(channels)
        )
        delta = site_xyz[left] - site_xyz[right]
        distances = np.linalg.norm(delta, axis=1)
        pair_gradient = (
            flows
            * (potential[left] - potential[right])
            * switch_derivatives
            / distances
        )[:, None] * delta
        np.add.at(site_gradient, left, pair_gradient)
        np.add.at(site_gradient, right, -pair_gradient)
    atom_gradient = weights.T @ site_gradient
    if not math.isclose(
        float(np.sum(charges)),
        float(np.sum(reference)),
        rel_tol=0.0,
        abs_tol=2.0e-12,
    ):
        raise RuntimeError("split-charge solve violated exact total-charge conservation")
    return SplitChargeResponseResult(
        energy,
        atom_gradient.reshape(-1),
        charges,
        flows,
        iterations,
        residual,
        (
            operator.backend
            if solver_backend == "FLOW_PCG"
            else f"{operator.backend}_{solver_backend}"
        ),
        len(operator.penetration_pairs),
        channel_count=len(channels),
    )


def split_charge_response_hessian_vector_product(
    coordinates_bohr: np.ndarray,
    sites: Sequence[ChargeResponseSite],
    channels: Sequence[SplitChargeChannel],
    vector_bohr: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 500,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_sites: int = 256,
    reaction_field: CPCMReactionField | None = None,
    excluded_pairs: tuple[tuple[int, int], ...] = (),
    _prepared: PersistentSplitChargeResponse | None = None,
) -> np.ndarray:
    """Apply the exact relaxed-response Hessian by implicit differentiation."""

    atom_xyz = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(vector_bohr, dtype=float)
    if direction.size == atom_xyz.size:
        direction = direction.reshape(atom_xyz.shape)
    if (
        atom_xyz.ndim != 2
        or atom_xyz.shape[1] != 3
        or direction.shape != atom_xyz.shape
        or np.any(~np.isfinite(atom_xyz))
        or np.any(~np.isfinite(direction))
    ):
        raise ValueError("charge-response Hessian-vector dimensions are inconsistent")
    site_xyz, weights = _site_geometry(atom_xyz, sites)
    site_direction = weights @ direction
    reference = (
        _prepared._reference
        if _prepared is not None
        else np.asarray([site.reference_charge for site in sites], dtype=float)
    )
    if not channels:
        return np.zeros(atom_xyz.size)
    operator = GaussianChargeResponseOperator.compile(
        site_xyz,
        sites,
        excluded_pairs=_prepared.excluded_pairs if _prepared is not None else excluded_pairs,
        backend=backend,
        fmm_precision=fmm_precision,
        fmm_minimum_sites=fmm_minimum_sites,
        reaction_field=reaction_field,
    )
    channel_data = _prepared._channel_data if _prepared is not None else None
    incidence, _switches, _switch_derivatives = _incidence_matrix(
        site_xyz, channels, channel_data
    )
    incidence_dot = _incidence_directional_derivative(
        site_xyz, site_direction, channels, channel_data
    )
    hardness = (
        _prepared._hardness
        if _prepared is not None
        else np.asarray([channel.hardness_hartree for channel in channels])
    )
    bias = (
        _prepared._bias
        if _prepared is not None
        else np.asarray([channel.reference_bias_hartree for channel in channels])
    )

    diagonal_inverse = 1.0 / np.maximum(
        hardness
        + _channel_curvatures(operator, channels, _switches, channel_data),
        1.0e-12,
    )
    right_hand_side = bias - incidence.T @ operator.matrix_vector(reference)
    flows, _iterations, _residual, _solver_backend = _solve_flow_system(
        operator,
        incidence,
        hardness,
        right_hand_side,
        diagonal_inverse,
        float(tolerance),
        int(maximum_iterations),
    )
    charges = reference + incidence @ flows
    fixed_flow_charge_dot = incidence_dot @ flows
    fixed_geometry_potential_dot = operator.directional_matrix_vector(
        charges, site_direction
    )
    stationarity_dot = (
        incidence_dot.T @ operator.matrix_vector(charges)
        + incidence.T @ fixed_geometry_potential_dot
        + incidence.T @ operator.matrix_vector(fixed_flow_charge_dot)
    )
    flow_dot, _response_iterations, _response_residual, _response_backend = (
        _solve_flow_system(
            operator,
            incidence,
            hardness,
            -stationarity_dot,
            diagonal_inverse,
            float(tolerance),
            int(maximum_iterations),
        )
    )
    charge_dot = fixed_flow_charge_dot + incidence @ flow_dot
    potential = operator.matrix_vector(charges)
    potential_dot = (
        fixed_geometry_potential_dot + operator.matrix_vector(charge_dot)
    )

    site_product = operator.relaxed_gradient_directional(
        charges,
        reference,
        charge_dot,
        site_direction,
    )
    left, right, radii, exponents = (
        channel_data if channel_data is not None else _channel_arrays(channels)
    )
    delta = site_xyz[left] - site_xyz[right]
    delta_dot = site_direction[left] - site_direction[right]
    distances = np.linalg.norm(delta, axis=1)
    unit = delta / distances[:, None]
    distance_dot = np.einsum("ij,ij->i", unit, delta_dot)
    unit_dot = (
        delta_dot - unit * distance_dot[:, None]
    ) / distances[:, None]
    switches = 1.0 / (1.0 + (distances / radii) ** exponents)
    first = -exponents * switches * (1.0 - switches) / distances
    second = (
        exponents
        * switches
        * (1.0 - switches)
        * (1.0 + exponents * (1.0 - 2.0 * switches))
        / distances**2
    )
    potential_difference = potential[left] - potential[right]
    potential_difference_dot = potential_dot[left] - potential_dot[right]
    longitudinal = (
        flow_dot * potential_difference * first
        + flows * potential_difference_dot * first
        + flows * potential_difference * second * distance_dot
    )
    pair_product = (
        longitudinal[:, None] * unit
        + (flows * potential_difference * first)[:, None] * unit_dot
    )
    np.add.at(site_product, left, pair_product)
    np.add.at(site_product, right, -pair_product)
    return (weights.T @ site_product).reshape(-1)


def _implicit_pair_chunks(
    site_count: int,
    chunk_size: int,
    pair_array: np.ndarray | None = None,
):
    """Yield complete-graph or compact local pairs in bounded chunks."""

    if pair_array is not None:
        pairs = np.asarray(pair_array, dtype=np.intp).reshape(-1, 2)
        for start in range(0, len(pairs), chunk_size):
            block = pairs[start : start + chunk_size]
            yield block[:, 0], block[:, 1]
        return
    for left in range(max(0, site_count - 1)):
        for start in range(left + 1, site_count, chunk_size):
            right = np.arange(
                start, min(site_count, start + chunk_size), dtype=np.intp
            )
            yield np.full(len(right), left, dtype=np.intp), right


def _implicit_pair_fields(
    runtime: PersistentAllPairSplitChargeResponse,
    site_xyz: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    delta = site_xyz[left] - site_xyz[right]
    distances = np.linalg.norm(delta, axis=1)
    if np.any(distances <= 1.0e-12):
        raise ValueError("coincident implicit split-charge channel sites")
    hardness = 0.5 * (
        runtime.response_lengths[left] + runtime.response_lengths[right]
    )
    radius_angstrom = np.maximum(
        float(runtime.switch_minimum_angstrom),
        float(runtime.switch_scale)
        * (
            runtime.rmin_half_angstrom[left]
            + runtime.rmin_half_angstrom[right]
        ),
    )
    radii = radius_angstrom / 0.52917721092
    exponent = int(runtime.switch_exponent)
    rational = 1.0 / (1.0 + (distances / radii) ** exponent)
    rational_first = -exponent * rational * (1.0 - rational) / distances
    rational_second = (
        exponent
        * rational
        * (1.0 - rational)
        * (1.0 + exponent * (1.0 - 2.0 * rational))
        / distances**2
    )
    if isinstance(runtime, PersistentLocalPairSplitChargeResponse):
        taper, taper_first, taper_second = _local_taper_fields(runtime, distances)
        switches = rational * taper
        first = rational_first * taper + rational * taper_first
        second = (
            rational_second * taper
            + 2.0 * rational_first * taper_first
            + rational * taper_second
        )
    else:
        switches = rational
        first = rational_first
        second = rational_second
    return delta, distances, hardness, switches, first, second


def _local_taper_fields(
    runtime: PersistentLocalPairSplitChargeResponse,
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = float(runtime.channel_cutoff_angstrom) / 0.52917721092
    inner = float(runtime.taper_start_fraction) * outer
    width = outer - inner
    scaled = np.clip((distances - inner) / width, 0.0, 1.0)
    active = (distances > inner) & (distances < outer)
    taper = np.where(
        distances <= inner,
        1.0,
        np.where(
            distances >= outer,
            0.0,
            1.0 - 10.0 * scaled**3 + 15.0 * scaled**4 - 6.0 * scaled**5,
        ),
    )
    first = np.zeros_like(distances)
    second = np.zeros_like(distances)
    first[active] = (
        -30.0 * scaled[active] ** 2
        + 60.0 * scaled[active] ** 3
        - 30.0 * scaled[active] ** 4
    ) / width
    second[active] = (
        -60.0 * scaled[active]
        + 180.0 * scaled[active] ** 2
        - 120.0 * scaled[active] ** 3
    ) / width**2
    return taper, first, second


def _implicit_reference_bias(
    runtime: PersistentAllPairSplitChargeResponse,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    delta = runtime._reference_site_xyz[left] - runtime._reference_site_xyz[right]
    distances = np.linalg.norm(delta, axis=1)
    radius_angstrom = np.maximum(
        float(runtime.switch_minimum_angstrom),
        float(runtime.switch_scale)
        * (
            runtime.rmin_half_angstrom[left]
            + runtime.rmin_half_angstrom[right]
        ),
    )
    radii = radius_angstrom / 0.52917721092
    reference_switch = 1.0 / (
        1.0 + (distances / radii) ** int(runtime.switch_exponent)
    )
    return reference_switch * (
        runtime._reference_potential[left]
        - runtime._reference_potential[right]
    )


def _implicit_bias_fields(
    runtime: PersistentAllPairSplitChargeResponse,
    distances: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_bias = _implicit_reference_bias(runtime, left, right)
    if not isinstance(runtime, PersistentLocalPairSplitChargeResponse):
        zeros = np.zeros_like(reference_bias)
        return reference_bias, zeros, zeros
    taper, first, second = _local_taper_fields(runtime, distances)
    return reference_bias * taper, reference_bias * first, reference_bias * second


def _implicit_laplacian_product(
    runtime: PersistentAllPairSplitChargeResponse,
    site_xyz: np.ndarray,
    vector: np.ndarray,
    *,
    directional: bool = False,
    site_direction: np.ndarray | None = None,
    pair_array: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``B H^-1 B.T`` or its directional derivative in O(N) memory."""

    values = np.asarray(vector, dtype=float).reshape(-1)
    product = np.zeros(len(site_xyz))
    for left, right in _implicit_pair_chunks(
        len(site_xyz), int(runtime.pair_chunk_size), pair_array
    ):
        delta, distances, hardness, switches, first, _second = (
            _implicit_pair_fields(runtime, site_xyz, left, right)
        )
        if not directional:
            weights = switches**2 / hardness
        else:
            if site_direction is None:
                raise ValueError("directional Laplacian requires site directions")
            distance_dot = np.einsum(
                "ij,ij->i",
                delta,
                site_direction[left] - site_direction[right],
            ) / distances
            weights = 2.0 * switches * first * distance_dot / hardness
        pair = weights * (values[left] - values[right])
        np.add.at(product, left, pair)
        np.add.at(product, right, -pair)
    return product


def _implicit_base_shift(
    runtime: PersistentAllPairSplitChargeResponse,
    site_xyz: np.ndarray,
    *,
    site_direction: np.ndarray | None = None,
    pair_array: np.ndarray | None = None,
) -> np.ndarray:
    shift = np.zeros(len(site_xyz))
    for left, right in _implicit_pair_chunks(
        len(site_xyz), int(runtime.pair_chunk_size), pair_array
    ):
        delta, distances, hardness, switches, first, _second = (
            _implicit_pair_fields(runtime, site_xyz, left, right)
        )
        bias, bias_first, _bias_second = _implicit_bias_fields(
            runtime, distances, left, right
        )
        if site_direction is None:
            coefficient = switches * bias / hardness
        else:
            distance_dot = np.einsum(
                "ij,ij->i",
                delta,
                site_direction[left] - site_direction[right],
            ) / distances
            coefficient = (
                first * bias + switches * bias_first
            ) * distance_dot / hardness
        np.add.at(shift, left, coefficient)
        np.add.at(shift, right, -coefficient)
    return shift


def _solve_implicit_site_system(
    runtime: PersistentAllPairSplitChargeResponse,
    operator: GaussianChargeResponseOperator,
    site_xyz: np.ndarray,
    right_hand_side: np.ndarray,
    pair_array: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    iterations = 0

    def product(charges: np.ndarray) -> np.ndarray:
        return charges + _implicit_laplacian_product(
            runtime,
            site_xyz,
            operator.matrix_vector(charges),
            pair_array=pair_array,
        )

    def count_iteration(_value) -> None:
        nonlocal iterations
        iterations += 1

    solution, status = gmres(
        LinearOperator(
            (len(site_xyz), len(site_xyz)),
            matvec=product,
            dtype=float,
        ),
        np.asarray(right_hand_side, dtype=float),
        rtol=float(runtime.tolerance),
        atol=float(runtime.tolerance),
        maxiter=int(runtime.maximum_iterations),
        callback=count_iteration,
        callback_type="legacy",
    )
    if status != 0:
        raise RuntimeError("implicit all-pair site response did not converge")
    residual = float(np.linalg.norm(product(solution) - right_hand_side))
    target = float(runtime.tolerance) * max(
        1.0, float(np.linalg.norm(right_hand_side))
    )
    if residual > 10.0 * target:
        raise RuntimeError("implicit all-pair response failed its residual gate")
    return np.asarray(solution), iterations, residual


def _solve_implicit_all_pair_response(
    runtime: PersistentAllPairSplitChargeResponse,
    coordinates_bohr: np.ndarray,
    *,
    compute_gradient: bool,
) -> SplitChargeResponseResult:
    atom_xyz = np.asarray(coordinates_bohr, dtype=float)
    if (
        atom_xyz.ndim != 2
        or atom_xyz.shape[1] != 3
        or np.any(~np.isfinite(atom_xyz))
    ):
        raise ValueError("charge-response coordinates must have shape (natoms, 3)")
    site_xyz, weights = _site_geometry(atom_xyz, runtime.sites)
    operator = GaussianChargeResponseOperator.compile(
        site_xyz,
        runtime.sites,
        excluded_pairs=runtime.excluded_pairs,
        backend=runtime.backend,
        fmm_precision=runtime.fmm_precision,
        fmm_minimum_sites=runtime.fmm_minimum_sites,
        reaction_field=runtime.reaction_field,
    )
    pair_array = (
        runtime.active_pairs(site_xyz)
        if isinstance(runtime, PersistentLocalPairSplitChargeResponse)
        else None
    )
    right_hand_side = runtime._reference + _implicit_base_shift(
        runtime, site_xyz, pair_array=pair_array
    )
    charges, iterations, residual = _solve_implicit_site_system(
        runtime, operator, site_xyz, right_hand_side, pair_array
    )
    if compute_gradient:
        response_energy, site_gradient = (
            operator.response_energy_gradient_difference(
                charges, runtime._reference
            )
        )
    else:
        response_energy = 0.5 * float(
            charges @ operator.matrix_vector(charges)
            - runtime._reference
            @ operator.matrix_vector(runtime._reference)
        )
        site_gradient = np.zeros_like(site_xyz)
    potential = operator.matrix_vector(charges)
    flow_energy = 0.0
    for left, right in _implicit_pair_chunks(
        len(site_xyz), int(runtime.pair_chunk_size), pair_array
    ):
        delta, distances, hardness, switches, first, _second = (
            _implicit_pair_fields(runtime, site_xyz, left, right)
        )
        bias, bias_first, _bias_second = _implicit_bias_fields(
            runtime, distances, left, right
        )
        difference = potential[left] - potential[right]
        flows = (bias - switches * difference) / hardness
        flow_energy += 0.5 * float(np.sum(hardness * flows**2))
        flow_energy -= float(np.dot(bias, flows))
        if compute_gradient:
            pair_gradient = (
                flows * (difference * first - bias_first) / distances
            )[:, None] * delta
            np.add.at(site_gradient, left, pair_gradient)
            np.add.at(site_gradient, right, -pair_gradient)
    atom_gradient = np.asarray(weights.T @ site_gradient).reshape(-1)
    if not math.isclose(
        float(np.sum(charges)),
        float(np.sum(runtime._reference)),
        rel_tol=0.0,
        abs_tol=2.0e-10,
    ):
        raise RuntimeError("implicit all-pair response violated charge conservation")
    return SplitChargeResponseResult(
        response_energy + flow_energy,
        atom_gradient,
        charges,
        np.zeros(0),
        iterations,
        residual,
        f"{operator.backend}_IMPLICIT_"
        + (
            "LOCAL_PAIR_SITE"
            if isinstance(runtime, PersistentLocalPairSplitChargeResponse)
            else "ALL_PAIR_SITE"
        ),
        len(operator.penetration_pairs),
        channel_count=(
            runtime.channel_count if pair_array is None else len(pair_array)
        ),
        flows_materialized=False,
    )


def _implicit_all_pair_hessian_vector_product(
    runtime: PersistentAllPairSplitChargeResponse,
    coordinates_bohr: np.ndarray,
    vector_bohr: np.ndarray,
) -> np.ndarray:
    atom_xyz = np.asarray(coordinates_bohr, dtype=float)
    direction = np.asarray(vector_bohr, dtype=float)
    if direction.size == atom_xyz.size:
        direction = direction.reshape(atom_xyz.shape)
    if (
        atom_xyz.ndim != 2
        or atom_xyz.shape[1] != 3
        or direction.shape != atom_xyz.shape
        or np.any(~np.isfinite(atom_xyz))
        or np.any(~np.isfinite(direction))
    ):
        raise ValueError("charge-response Hessian-vector dimensions are inconsistent")
    site_xyz, weights = _site_geometry(atom_xyz, runtime.sites)
    site_direction = np.asarray(weights @ direction)
    operator = GaussianChargeResponseOperator.compile(
        site_xyz,
        runtime.sites,
        excluded_pairs=runtime.excluded_pairs,
        backend=runtime.backend,
        fmm_precision=runtime.fmm_precision,
        fmm_minimum_sites=runtime.fmm_minimum_sites,
        reaction_field=runtime.reaction_field,
    )
    pair_array = (
        runtime.active_pairs(site_xyz)
        if isinstance(runtime, PersistentLocalPairSplitChargeResponse)
        else None
    )
    base = runtime._reference + _implicit_base_shift(
        runtime, site_xyz, pair_array=pair_array
    )
    charges, _iterations, _residual = _solve_implicit_site_system(
        runtime, operator, site_xyz, base, pair_array
    )
    potential = operator.matrix_vector(charges)
    geometry_potential_dot = operator.directional_matrix_vector(
        charges, site_direction
    )
    base_dot = _implicit_base_shift(
        runtime,
        site_xyz,
        site_direction=site_direction,
        pair_array=pair_array,
    )
    laplacian_dot_potential = _implicit_laplacian_product(
        runtime,
        site_xyz,
        potential,
        directional=True,
        site_direction=site_direction,
        pair_array=pair_array,
    )
    rhs_dot = (
        base_dot
        - laplacian_dot_potential
        - _implicit_laplacian_product(
            runtime,
            site_xyz,
            geometry_potential_dot,
            pair_array=pair_array,
        )
    )
    charge_dot, _dot_iterations, _dot_residual = (
        _solve_implicit_site_system(
            runtime, operator, site_xyz, rhs_dot, pair_array
        )
    )
    potential_dot = geometry_potential_dot + operator.matrix_vector(charge_dot)
    site_product = operator.relaxed_gradient_directional(
        charges,
        runtime._reference,
        charge_dot,
        site_direction,
    )
    for left, right in _implicit_pair_chunks(
        len(site_xyz), int(runtime.pair_chunk_size), pair_array
    ):
        delta, distances, hardness, switches, first, second = (
            _implicit_pair_fields(runtime, site_xyz, left, right)
        )
        delta_dot = site_direction[left] - site_direction[right]
        unit = delta / distances[:, None]
        distance_dot = np.einsum("ij,ij->i", unit, delta_dot)
        unit_dot = (
            delta_dot - unit * distance_dot[:, None]
        ) / distances[:, None]
        bias, bias_first, bias_second = _implicit_bias_fields(
            runtime, distances, left, right
        )
        difference = potential[left] - potential[right]
        difference_dot = potential_dot[left] - potential_dot[right]
        flows = (bias - switches * difference) / hardness
        flow_dot = (
            bias_first * distance_dot
            - first * distance_dot * difference
            - switches * difference_dot
        ) / hardness
        radial = difference * first - bias_first
        coefficient = flows * radial
        coefficient_dot = (
            flow_dot * radial
            + flows
            * (
                difference_dot * first
                + (difference * second - bias_second) * distance_dot
            )
        )
        pair_product = (
            coefficient_dot[:, None] * unit
            + coefficient[:, None] * unit_dot
        )
        np.add.at(site_product, left, pair_product)
        np.add.at(site_product, right, -pair_product)
    return np.asarray(weights.T @ site_product).reshape(-1)


def reference_channel_biases(
    coordinates_bohr: np.ndarray,
    sites: Sequence[ChargeResponseSite],
    channels: Sequence[SplitChargeChannel],
    *,
    excluded_pairs: tuple[tuple[int, int], ...] = (),
) -> tuple[float, ...]:
    """Return biases that make the supplied reference charges stationary."""

    xyz, _weights = _site_geometry(np.asarray(coordinates_bohr, dtype=float), sites)
    coulomb = _gaussian_coulomb_matrix(xyz, sites)
    for left, right in canonical_pairs(len(sites), excluded_pairs):
        coulomb[left, right] = coulomb[right, left] = 0.0
    incidence, _switches, _derivatives = _incidence_matrix(xyz, channels)
    charges = np.asarray([site.reference_charge for site in sites], dtype=float)
    return tuple(float(value) for value in incidence.T @ (coulomb @ charges))


def with_reference_channel_biases(
    coordinates_bohr: np.ndarray,
    sites: Sequence[ChargeResponseSite],
    channels: Sequence[SplitChargeChannel],
    *,
    excluded_pairs: tuple[tuple[int, int], ...] = (),
) -> tuple[SplitChargeChannel, ...]:
    biases = reference_channel_biases(coordinates_bohr, sites, channels, excluded_pairs=excluded_pairs)
    return tuple(
        SplitChargeChannel(
            channel.left_site,
            channel.right_site,
            channel.hardness_hartree,
            channel.switch_radius_bohr,
            channel.switch_exponent,
            bias,
        )
        for channel, bias in zip(channels, biases, strict=True)
    )


def _site_geometry(
    atom_xyz: np.ndarray,
    sites: Sequence[ChargeResponseSite],
) -> tuple[np.ndarray, csr_matrix]:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for site_index, site in enumerate(sites):
        for atom, weight in site.atom_weights:
            if atom >= len(atom_xyz):
                raise ValueError("charge-response site atom index is out of range")
            rows.append(site_index)
            columns.append(atom)
            values.append(weight)
    weights = csr_matrix(
        (values, (rows, columns)),
        shape=(len(sites), len(atom_xyz)),
        dtype=float,
    )
    return np.asarray(weights @ atom_xyz), weights






def _incidence_matrix(
    site_xyz: np.ndarray,
    channels: Sequence[SplitChargeChannel],
    channel_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
) -> tuple[np.ndarray | csc_matrix, np.ndarray, np.ndarray]:
    if not channels:
        return (
            np.zeros((len(site_xyz), 0)),
            np.zeros(0),
            np.zeros(0),
        )
    left, right, radii, exponents = (
        channel_data if channel_data is not None else _channel_arrays(channels)
    )
    distances = np.linalg.norm(site_xyz[left] - site_xyz[right], axis=1)
    if np.any(distances <= 1.0e-12):
        raise ValueError("coincident split-charge channel sites")
    ratios = (distances / radii) ** exponents
    switches = 1.0 / (1.0 + ratios)
    derivatives = -exponents * switches * (1.0 - switches) / distances
    columns = np.arange(len(channels), dtype=np.intp)
    if len(channels) >= _SPARSE_INCIDENCE_MINIMUM_CHANNELS:
        incidence = csc_matrix(
            (
                np.concatenate((switches, -switches)),
                (
                    np.concatenate((left, right)),
                    np.concatenate((columns, columns)),
                ),
            ),
            shape=(len(site_xyz), len(channels)),
        )
    else:
        incidence = np.zeros((len(site_xyz), len(channels)))
        incidence[left, columns] = switches
        incidence[right, columns] = -switches
    return incidence, switches, derivatives


def _gaussian_pair(
    distance: float,
    left_width: float,
    right_width: float,
) -> tuple[float, float]:
    beta = _gaussian_beta(left_width, right_width)
    argument = beta * distance
    exponential = math.exp(-(argument**2))
    value = math.erf(argument) / distance
    derivative = (
        2.0 * beta * exponential / (math.sqrt(math.pi) * distance)
        - math.erf(argument) / distance**2
    )
    return value, derivative


def _gaussian_beta(left_width: float, right_width: float) -> float:
    return 1.0 / math.sqrt(2.0 * (left_width**2 + right_width**2))




def _gaussian_coulomb_matrix(
    site_xyz: np.ndarray,
    sites: Sequence[ChargeResponseSite],
) -> np.ndarray:
    widths = np.asarray(
        [site.gaussian_width_bohr for site in sites], dtype=float
    )
    matrix = np.diag(1.0 / (math.sqrt(math.pi) * widths))
    if len(sites) < 2:
        return matrix
    left, right = np.triu_indices(len(sites), k=1)
    distances = np.linalg.norm(site_xyz[left] - site_xyz[right], axis=1)
    if np.any(distances <= 1.0e-12):
        raise ValueError("coincident Gaussian response sites")
    beta = 1.0 / np.sqrt(
        2.0 * (widths[left] ** 2 + widths[right] ** 2)
    )
    values = erf(beta * distances) / distances
    matrix[left, right] = values
    matrix[right, left] = values
    return matrix


def _channel_arrays(
    channels: Sequence[SplitChargeChannel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.fromiter(
            (channel.left_site for channel in channels),
            dtype=np.intp,
            count=len(channels),
        ),
        np.fromiter(
            (channel.right_site for channel in channels),
            dtype=np.intp,
            count=len(channels),
        ),
        np.fromiter(
            (channel.switch_radius_bohr for channel in channels),
            dtype=float,
            count=len(channels),
        ),
        np.fromiter(
            (channel.switch_exponent for channel in channels),
            dtype=float,
            count=len(channels),
        ),
    )


def _channel_curvatures(
    operator: GaussianChargeResponseOperator,
    channels: Sequence[SplitChargeChannel],
    switches: np.ndarray,
    channel_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
) -> np.ndarray:
    if not channels:
        return np.zeros(0)
    left, right, _radii, _exponents = (
        channel_data if channel_data is not None else _channel_arrays(channels)
    )
    distances = np.linalg.norm(
        operator.site_xyz_bohr[left] - operator.site_xyz_bohr[right],
        axis=1,
    )
    widths = operator.widths_bohr
    beta = 1.0 / np.sqrt(
        2.0 * (widths[left] ** 2 + widths[right] ** 2)
    )
    pair = erf(beta * distances) / distances
    if operator.excluded_pair_set:
        excluded = np.fromiter(
            ((min(a, b), max(a, b)) in operator.excluded_pair_set for a, b in zip(left, right)),
            dtype=bool, count=len(left))
        pair[excluded] = 0.0
    values = switches**2 * (
        operator.diagonal[left] + operator.diagonal[right] - 2.0 * pair
    )
    if operator.reaction_field is not None:
        values += np.asarray(
            [
                operator.reaction_field.charge_flow_curvature(
                    operator.site_xyz_bohr,
                    channel.left_site,
                    channel.right_site,
                    switches[index],
                )
                for index, channel in enumerate(channels)
            ]
        )
    return values


def _gaussian_coulomb_directional_derivative(
    site_xyz: np.ndarray,
    site_direction: np.ndarray,
    sites: Sequence[ChargeResponseSite],
) -> np.ndarray:
    derivative = np.zeros((len(sites), len(sites)))
    if len(sites) < 2:
        return derivative
    left, right = np.triu_indices(len(sites), k=1)
    delta = site_xyz[left] - site_xyz[right]
    distances = np.linalg.norm(delta, axis=1)
    distance_dot = np.einsum(
        "ij,ij->i",
        delta,
        site_direction[left] - site_direction[right],
    ) / distances
    widths = np.asarray(
        [site.gaussian_width_bohr for site in sites], dtype=float
    )
    beta = 1.0 / np.sqrt(
        2.0 * (widths[left] ** 2 + widths[right] ** 2)
    )
    arguments = beta * distances
    first = (
        2.0
        * beta
        * np.exp(-(arguments**2))
        / (math.sqrt(math.pi) * distances)
        - erf(arguments) / distances**2
    )
    values = first * distance_dot
    derivative[left, right] = values
    derivative[right, left] = values
    return derivative


def _incidence_directional_derivative(
    site_xyz: np.ndarray,
    site_direction: np.ndarray,
    channels: Sequence[SplitChargeChannel],
    channel_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
) -> np.ndarray | csc_matrix:
    if not channels:
        return np.zeros((len(site_xyz), 0))
    left, right, radii, exponents = (
        channel_data if channel_data is not None else _channel_arrays(channels)
    )
    delta = site_xyz[left] - site_xyz[right]
    distances = np.linalg.norm(delta, axis=1)
    direction_delta = site_direction[left] - site_direction[right]
    distance_dot = np.einsum("ij,ij->i", delta, direction_delta) / distances
    switches = 1.0 / (1.0 + (distances / radii) ** exponents)
    first = -exponents * switches * (1.0 - switches) / distances
    values = first * distance_dot
    columns = np.arange(len(channels), dtype=np.intp)
    if len(channels) >= _SPARSE_INCIDENCE_MINIMUM_CHANNELS:
        return csc_matrix(
            (
                np.concatenate((values, -values)),
                (
                    np.concatenate((left, right)),
                    np.concatenate((columns, columns)),
                ),
            ),
            shape=(len(site_xyz), len(channels)),
        )
    derivative = np.zeros((len(site_xyz), len(channels)))
    derivative[left, columns] = values
    derivative[right, columns] = -values
    return derivative




def _solve_flow_system(
    operator: GaussianChargeResponseOperator,
    incidence: np.ndarray | csc_matrix,
    hardness: np.ndarray,
    right_hand_side: np.ndarray,
    inverse_diagonal: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, int, float, str]:
    """Solve the SQE system in flow or exact site-Schur representation."""

    requested = os.environ.get("MATRIX_QEQ_SOLVER", "auto").strip().casefold()
    if requested not in {"auto", "flow", "site"}:
        raise ValueError("MATRIX_QEQ_SOLVER must be auto, flow, or site")
    use_site = requested == "site" or (
        requested == "auto"
        and len(right_hand_side) > 4 * len(operator.site_xyz_bohr)
    )

    def flow_product(flow: np.ndarray) -> np.ndarray:
        delta = np.asarray(incidence @ flow).reshape(-1)
        return hardness * flow + np.asarray(
            incidence.T @ operator.matrix_vector(delta)
        ).reshape(-1)

    if not use_site:
        solution, iterations, residual = _preconditioned_cg(
            flow_product,
            right_hand_side,
            inverse_diagonal,
            tolerance,
            maximum_iterations,
        )
        return solution, iterations, residual, "FLOW_PCG"

    inverse_hardness = 1.0 / hardness
    if isinstance(incidence, csc_matrix):
        weighted = incidence.multiply(inverse_hardness[None, :])
        charge_metric = np.asarray((weighted @ incidence.T).toarray())
    else:
        charge_metric = (
            np.asarray(incidence) * inverse_hardness[None, :]
        ) @ np.asarray(incidence).T
    site_right_hand_side = np.asarray(
        incidence @ (inverse_hardness * right_hand_side)
    ).reshape(-1)
    site_count = len(site_right_hand_side)
    iteration_count = 0
    if operator.dense_matrix is not None and operator.reaction_field is None:
        site_matrix = np.eye(site_count) + charge_metric @ operator.dense_matrix
        delta_charge = np.linalg.solve(site_matrix, site_right_hand_side)
        iteration_count = 1
    else:
        def site_product(delta: np.ndarray) -> np.ndarray:
            return delta + charge_metric @ operator.matrix_vector(delta)

        def count_iteration(_value) -> None:
            nonlocal iteration_count
            iteration_count += 1

        delta_charge, status = gmres(
            LinearOperator(
                (site_count, site_count),
                matvec=site_product,
                dtype=float,
            ),
            site_right_hand_side,
            rtol=tolerance,
            atol=tolerance,
            maxiter=maximum_iterations,
            callback=count_iteration,
            callback_type="legacy",
        )
        if status != 0:
            raise RuntimeError(
                "site-space split-charge response solve did not converge"
            )
    solution = inverse_hardness * (
        right_hand_side
        - np.asarray(
            incidence.T @ operator.matrix_vector(delta_charge)
        ).reshape(-1)
    )
    residual = float(np.linalg.norm(flow_product(solution) - right_hand_side))
    target = tolerance * max(1.0, float(np.linalg.norm(right_hand_side)))
    if residual > 10.0 * target:
        raise RuntimeError(
            "site-space split-charge response failed the flow-space residual gate"
        )
    return solution, iteration_count, residual, "SITE_SCHUR"


def _preconditioned_cg(
    matrix_vector,
    right_hand_side: np.ndarray,
    inverse_diagonal: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, int, float]:
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    target = tolerance * max(1.0, float(np.linalg.norm(right_hand_side)))
    if float(np.linalg.norm(residual)) <= target:
        return solution, 0, float(np.linalg.norm(residual))
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.copy()
    rz = float(residual @ preconditioned)
    for iteration in range(1, maximum_iterations + 1):
        product = matrix_vector(direction)
        denominator = float(direction @ product)
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError("split-charge response matrix is not positive definite")
        step = rz / denominator
        solution += step * direction
        residual -= step * product
        norm = float(np.linalg.norm(residual))
        if norm <= target:
            return solution, iteration, norm
        next_preconditioned = inverse_diagonal * residual
        next_rz = float(residual @ next_preconditioned)
        direction = next_preconditioned + (next_rz / rz) * direction
        rz = next_rz
    raise RuntimeError("split-charge response solve did not converge")


__all__ = [
    "ZAFF_CHARGE_RESPONSE_SCHEMA",
    "ChargeResponseSite",
    "GaussianChargeResponseOperator",
    "PersistentAllPairSplitChargeResponse",
    "PersistentLocalPairSplitChargeResponse",
    "PersistentSplitChargeResponse",
    "SplitChargeChannel",
    "SplitChargeResponseResult",
    "reference_channel_biases",
    "solve_split_charge_response",
    "split_charge_response_hessian_vector_product",
    "with_reference_channel_biases",
]
