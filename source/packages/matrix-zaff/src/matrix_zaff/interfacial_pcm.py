"""Exact planar dielectric reaction field for explicit--continuum interfaces.

The first INTERPHASES contract places every explicit charge site in the
positive, vacuum side of one plane.  The residual substrate occupies the
negative side and is represented by a scalar dielectric constant.  The
electrostatic reaction field is therefore the exact planar image solution for
the same normalized Gaussian charge densities used by ZAFF.  The Green
function is ``erf(beta*r)/r`` rather than a point-charge ``1/r`` shortcut.
Energy, Cartesian gradient, full Hessian, and Hessian--vector products are
analytic and continuous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import platform
import time
from typing import Any, Literal, Mapping

import numpy as np


ZAFF_INTERFACIAL_PCM_SCHEMA = "matrix.zaff.interfacial_pcm.v2"


@dataclass(frozen=True)
class PlanarDielectricInterface:
    """Frozen regular interface below an explicit molecular region.

    Coordinates are in bohr.  The unit normal points from the continuum
    substrate into the explicit region.  Version one deliberately keeps the
    explicit region at dielectric constant one because its molecular
    polarization is represented explicitly by the resident ZAFF Hamiltonian.
    """

    origin_bohr: np.ndarray
    normal: np.ndarray
    substrate_dielectric: float
    exclusion_gap_bohr: float = 0.0
    explicit_dielectric: float = 1.0

    @classmethod
    def from_angstrom(
        cls,
        origin_angstrom: np.ndarray,
        normal: np.ndarray,
        substrate_dielectric: float,
        *,
        exclusion_gap_angstrom: float = 0.0,
    ) -> "PlanarDielectricInterface":
        """Construct a ZAFF interface from a molecular-builder geometry."""

        from .nonbonded import BOHR_TO_ANGSTROM

        return cls(
            origin_bohr=np.asarray(origin_angstrom, dtype=float)
            / BOHR_TO_ANGSTROM,
            normal=normal,
            substrate_dielectric=substrate_dielectric,
            exclusion_gap_bohr=float(exclusion_gap_angstrom)
            / BOHR_TO_ANGSTROM,
        )

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_bohr, dtype=float).reshape(3)
        normal = np.asarray(self.normal, dtype=float).reshape(3)
        length = float(np.linalg.norm(normal))
        substrate = float(self.substrate_dielectric)
        explicit = float(self.explicit_dielectric)
        gap = float(self.exclusion_gap_bohr)
        if (
            np.any(~np.isfinite(origin))
            or np.any(~np.isfinite(normal))
            or not np.isfinite(length)
            or length <= 1.0e-14
        ):
            raise ValueError("interfacial PCM plane geometry is invalid")
        if not math.isinf(substrate) and (
            not np.isfinite(substrate) or substrate < 1.0
        ):
            raise ValueError(
                "interfacial PCM substrate dielectric must be at least one"
            )
        if not np.isclose(explicit, 1.0, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "INTERPHASES v1 requires an explicit molecular region at dielectric one"
            )
        if not np.isfinite(gap) or gap < 0.0:
            raise ValueError("interfacial PCM exclusion gap cannot be negative")
        normal = normal / length
        origin.setflags(write=False)
        normal.setflags(write=False)
        object.__setattr__(self, "origin_bohr", origin)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "substrate_dielectric", substrate)
        object.__setattr__(self, "explicit_dielectric", explicit)
        object.__setattr__(self, "exclusion_gap_bohr", gap)

    @property
    def image_factor(self) -> float:
        """Return the finite dielectric image-charge coefficient."""

        if math.isinf(self.substrate_dielectric):
            return -1.0
        return (1.0 - self.substrate_dielectric) / (
            1.0 + self.substrate_dielectric
        )

    @property
    def reflection(self) -> np.ndarray:
        return np.eye(3) - 2.0 * np.outer(self.normal, self.normal)

    def signed_distances(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or np.any(~np.isfinite(xyz)):
            raise ValueError("interfacial PCM coordinates must have shape (n, 3)")
        return (xyz - self.origin_bohr) @ self.normal

    def validate_explicit_coordinates(self, coordinates_bohr: np.ndarray) -> None:
        distances = self.signed_distances(coordinates_bohr)
        minimum = self.exclusion_gap_bohr + 1.0e-12
        if np.any(distances <= minimum):
            raise ValueError(
                "every explicit site must remain above the interfacial exclusion gap"
            )

    def reflected(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        distances = self.signed_distances(xyz)
        return xyz - 2.0 * distances[:, None] * self.normal


@dataclass(frozen=True)
class InterfacialPCMResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    backend: str
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ZAFF_INTERFACIAL_PCM_SCHEMA


@dataclass(frozen=True)
class InterfacialPCMSecondOrderResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    backend: str
    schema: str = ZAFF_INTERFACIAL_PCM_SCHEMA


@dataclass(frozen=True)
class PlanarSpectralCompression:
    """Accuracy and crossover policy for the planar spectral FMM.

    FMM3D represents well-separated image interactions with truncated
    spherical-harmonic spectra and evaluates the near field directly.  The
    same expansion is differentiated analytically for gradients and HVPs.
    """

    relative_tolerance: float = 1.0e-10
    minimum_sites: int = 512
    energy_gradient_minimum_sites: int | None = None
    hvp_minimum_sites: int | None = None
    batched_hvp_minimum_sites: int | None = None
    charge_response_minimum_sites: int | None = None
    hierarchical_minimum_sites: int = 512
    hierarchical_relative_tolerance: float = 1.0e-10
    hierarchical_max_rank: int = 128
    direct_block_size: int = 256
    validation_sites: int = 8
    validation_tolerance_multiplier: float = 50.0
    calibrated_platform: str = ""

    def __post_init__(self) -> None:
        tolerance = float(self.relative_tolerance)
        minimum = int(self.minimum_sites)
        operator_minima = {
            "energy_gradient_minimum_sites": self.energy_gradient_minimum_sites,
            "hvp_minimum_sites": self.hvp_minimum_sites,
            "batched_hvp_minimum_sites": self.batched_hvp_minimum_sites,
            "charge_response_minimum_sites": self.charge_response_minimum_sites,
        }
        block = int(self.direct_block_size)
        hierarchical_minimum = int(self.hierarchical_minimum_sites)
        hierarchical_tolerance = float(self.hierarchical_relative_tolerance)
        hierarchical_rank = int(self.hierarchical_max_rank)
        validation_sites = int(self.validation_sites)
        multiplier = float(self.validation_tolerance_multiplier)
        calibrated_platform = str(self.calibrated_platform)
        if not 0.0 < tolerance < 1.0:
            raise ValueError("spectral FMM tolerance must lie between zero and one")
        if minimum < 1:
            raise ValueError("spectral FMM crossover must be positive")
        for name, value in operator_minima.items():
            selected = minimum if value is None else int(value)
            if selected < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, selected)
        if block < 1:
            raise ValueError("direct image block size must be positive")
        if hierarchical_minimum < 1:
            raise ValueError("hierarchical crossover must be positive")
        if not 0.0 < hierarchical_tolerance < 1.0:
            raise ValueError("hierarchical tolerance must lie between zero and one")
        if hierarchical_rank < 1:
            raise ValueError("hierarchical maximum rank must be positive")
        if validation_sites < 0:
            raise ValueError("spectral validation site count cannot be negative")
        if not np.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError("spectral validation multiplier must be at least one")
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "minimum_sites", minimum)
        object.__setattr__(self, "direct_block_size", block)
        object.__setattr__(self, "hierarchical_minimum_sites", hierarchical_minimum)
        object.__setattr__(
            self, "hierarchical_relative_tolerance", hierarchical_tolerance
        )
        object.__setattr__(self, "hierarchical_max_rank", hierarchical_rank)
        object.__setattr__(self, "validation_sites", validation_sites)
        object.__setattr__(self, "validation_tolerance_multiplier", multiplier)
        object.__setattr__(self, "calibrated_platform", calibrated_platform)


def planar_spectral_platform_fingerprint() -> str:
    """Stable architecture label used to validate a measured crossover."""

    return "|".join(
        (
            platform.system() or "unknown-system",
            platform.machine() or "unknown-machine",
            platform.processor() or "unknown-processor",
            np.__version__,
        )
    )


def calibrate_planar_spectral_crossover(
    interface: PlanarDielectricInterface,
    *,
    candidate_site_counts: tuple[int, ...] = (
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ),
    repeats: int = 2,
    seed: int = 7823,
    relative_tolerance: float = 1.0e-10,
) -> PlanarSpectralCompression:
    """Measure distinct direct/FMM crossovers for the resident operators."""

    if repeats < 1 or not candidate_site_counts:
        raise ValueError("spectral crossover calibration needs candidates and repeats")
    candidates = tuple(sorted({int(value) for value in candidate_site_counts}))
    if candidates[0] < 1:
        raise ValueError("spectral crossover candidates must be positive")
    if not InterfacialPCMReactionField._fmm_available():
        raise RuntimeError("spectral crossover calibration requires matrix-zaff[fmm]")
    provisional = PlanarSpectralCompression(
        relative_tolerance=relative_tolerance,
        minimum_sites=max(candidates) + 1,
        calibrated_platform=planar_spectral_platform_fingerprint(),
    )
    field_model = InterfacialPCMReactionField(
        interface, provisional, gaussian_widths_bohr=1.0
    )
    rng = np.random.default_rng(int(seed))
    fallback = 4 * max(candidates)
    selected = {
        "energy_gradient": fallback,
        "hvp": fallback,
        "batched_hvp": fallback,
        "charge_response": fallback,
    }
    for count in candidates:
        tangent_a = np.asarray((1.0, 0.0, 0.0))
        if abs(float(tangent_a @ interface.normal)) > 0.8:
            tangent_a = np.asarray((0.0, 1.0, 0.0))
        tangent_a -= float(tangent_a @ interface.normal) * interface.normal
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(interface.normal, tangent_a)
        lateral = rng.normal(scale=8.0, size=(count, 2))
        heights = rng.uniform(
            interface.exclusion_gap_bohr + 1.0,
            interface.exclusion_gap_bohr + 7.0,
            size=count,
        )
        xyz = (
            interface.origin_bohr
            + lateral[:, :1] * tangent_a
            + lateral[:, 1:] * tangent_b
            + heights[:, None] * interface.normal
        )
        charges = rng.normal(scale=0.3, size=count)
        direction = rng.normal(size=(count, 3))
        directions = rng.normal(size=(4, count, 3))
        for operation in tuple(selected):
            if selected[operation] != fallback:
                continue
            timings: dict[str, float] = {}
            for backend in ("direct", "spectral"):
                best = math.inf
                for _ in range(int(repeats)):
                    start = time.perf_counter()
                    if operation == "energy_gradient":
                        field_model.reaction_energy_gradient(
                            xyz,
                            charges,
                            backend=backend,
                        )
                    elif operation == "hvp":
                        field_model.hessian_vector_product(
                            xyz,
                            charges,
                            direction,
                            backend=backend,
                        )
                    elif operation == "batched_hvp":
                        field_model.hessian_vector_products(
                            xyz,
                            charges,
                            directions,
                            backend=backend,
                        )
                    else:
                        field_model.kernel_product(
                            xyz,
                            charges,
                            backend=backend,
                        )
                    best = min(best, time.perf_counter() - start)
                timings[backend] = best
            if timings["spectral"] <= timings["direct"]:
                selected[operation] = count
        if all(value != fallback for value in selected.values()):
            break
    return PlanarSpectralCompression(
        relative_tolerance=relative_tolerance,
        minimum_sites=min(selected.values()),
        energy_gradient_minimum_sites=selected["energy_gradient"],
        hvp_minimum_sites=selected["hvp"],
        batched_hvp_minimum_sites=selected["batched_hvp"],
        charge_response_minimum_sites=selected["charge_response"],
        calibrated_platform=planar_spectral_platform_fingerprint(),
    )


@dataclass(frozen=True)
class PersistentPlanarImageOperator:
    """Fixed-geometry image kernel for iterative charge-response solves."""

    reaction_field: "InterfacialPCMReactionField"
    coordinates_bohr: np.ndarray
    gaussian_widths_bohr: np.ndarray
    selected_backend: Literal["direct", "fmm", "hierarchical"]
    dense_kernel: np.ndarray | None = None
    hierarchy: "PersistentPlanarHierarchy | None" = None
    hierarchical_kernel: "PersistentPlanarHMatrix | None" = None

    def matrix_vector(self, charges: np.ndarray) -> np.ndarray:
        q = np.asarray(charges, dtype=float).reshape(-1)
        if len(q) != len(self.coordinates_bohr):
            raise ValueError("persistent image charge vector has the wrong dimension")
        if self.dense_kernel is not None:
            return self.dense_kernel @ q
        if self.hierarchical_kernel is not None:
            return self.hierarchical_kernel.matrix_vector(q)
        return self.reaction_field.kernel_product(
            self.coordinates_bohr,
            q,
            gaussian_widths_bohr=self.gaussian_widths_bohr,
            backend="spectral",
        )

    def directional_matrix_vector(
        self,
        charges: np.ndarray,
        direction_bohr: np.ndarray,
    ) -> np.ndarray:
        return self.reaction_field.kernel_directional_product(
            self.coordinates_bohr,
            charges,
            direction_bohr,
            gaussian_widths_bohr=self.gaussian_widths_bohr,
            backend=(
                "direct"
                if self.selected_backend == "hierarchical"
                else self.selected_backend
            ),
        )


@dataclass(frozen=True)
class PlanarCluster:
    identifier: int
    indices: np.ndarray
    center_bohr: np.ndarray
    radius_bohr: float
    left: int | None = None
    right: int | None = None


@dataclass(frozen=True)
class PersistentPlanarHierarchy:
    """Geometry-only cluster tree and admissible block partition."""

    coordinates_bohr: np.ndarray
    clusters: tuple[PlanarCluster, ...]
    near_blocks: tuple[tuple[int, int], ...]
    far_blocks: tuple[tuple[int, int], ...]
    leaf_size: int
    admissibility: float

    @classmethod
    def build(
        cls,
        coordinates_bohr: np.ndarray,
        interface: PlanarDielectricInterface,
        *,
        leaf_size: int = 16,
        admissibility: float = 1.0,
    ) -> "PersistentPlanarHierarchy":
        xyz = np.asarray(coordinates_bohr, dtype=float)
        interface.validate_explicit_coordinates(xyz)
        leaf = max(2, int(leaf_size))
        eta = float(admissibility)
        if not np.isfinite(eta) or eta <= 0.0:
            raise ValueError("hierarchical admissibility must be positive")
        nodes: list[PlanarCluster] = []

        def build_node(indices: np.ndarray) -> int:
            points = xyz[indices]
            center = np.mean(points, axis=0)
            radius = float(
                np.max(np.linalg.norm(points - center, axis=1), initial=0.0)
            )
            identifier = len(nodes)
            nodes.append(
                PlanarCluster(identifier, indices, center, radius)
            )
            if len(indices) > leaf:
                spans = np.ptp(points, axis=0)
                axis = int(np.argmax(spans))
                order = indices[np.argsort(points[:, axis], kind="stable")]
                middle = len(order) // 2
                left = build_node(order[:middle])
                right = build_node(order[middle:])
                nodes[identifier] = PlanarCluster(
                    identifier,
                    indices,
                    center,
                    radius,
                    left,
                    right,
                )
            return identifier

        root = build_node(np.arange(len(xyz), dtype=int))
        near: list[tuple[int, int]] = []
        far: list[tuple[int, int]] = []

        def normalized(left: int, right: int) -> tuple[int, int]:
            return (left, right) if left <= right else (right, left)

        def partition(left_id: int, right_id: int) -> None:
            left_id, right_id = normalized(left_id, right_id)
            left_node = nodes[left_id]
            right_node = nodes[right_id]
            if left_id == right_id:
                if left_node.left is None:
                    near.append((left_id, right_id))
                else:
                    partition(left_node.left, left_node.left)
                    partition(left_node.left, left_node.right)
                    partition(left_node.right, left_node.right)
                return
            reflected_center = interface.reflected(
                right_node.center_bohr[None, :]
            )[0]
            separation = float(
                np.linalg.norm(left_node.center_bohr - reflected_center)
            )
            if separation > eta * (
                left_node.radius_bohr + right_node.radius_bohr
            ):
                far.append((left_id, right_id))
                return
            left_leaf = left_node.left is None
            right_leaf = right_node.left is None
            if left_leaf and right_leaf:
                near.append((left_id, right_id))
            elif right_leaf or (
                not left_leaf
                and left_node.radius_bohr >= right_node.radius_bohr
            ):
                partition(left_node.left, right_id)
                partition(left_node.right, right_id)
            else:
                partition(left_id, right_node.left)
                partition(left_id, right_node.right)

        partition(root, root)
        fixed_xyz = xyz.copy()
        fixed_xyz.setflags(write=False)
        fixed_nodes = []
        for node in nodes:
            indices = node.indices.copy()
            center = node.center_bohr.copy()
            indices.setflags(write=False)
            center.setflags(write=False)
            fixed_nodes.append(
                PlanarCluster(
                    node.identifier,
                    indices,
                    center,
                    node.radius_bohr,
                    node.left,
                    node.right,
                )
            )
        return cls(
            fixed_xyz,
            tuple(fixed_nodes),
            tuple(near),
            tuple(far),
            leaf,
            eta,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "cluster_count": len(self.clusters),
            "near_block_count": len(self.near_blocks),
            "far_block_count": len(self.far_blocks),
            "leaf_size": self.leaf_size,
            "admissibility": self.admissibility,
        }


@dataclass(frozen=True)
class PlanarDenseBlock:
    left_indices: np.ndarray
    right_indices: np.ndarray
    matrix: np.ndarray
    diagonal: bool


@dataclass(frozen=True)
class PlanarLowRankBlock:
    left_indices: np.ndarray
    right_indices: np.ndarray
    left_factor: np.ndarray
    right_factor: np.ndarray
    relative_truncation_error: float

    @property
    def rank(self) -> int:
        return self.left_factor.shape[1]


@dataclass(frozen=True)
class PersistentPlanarHMatrix:
    """Symmetric exact-near/low-rank-far image kernel."""

    site_count: int
    dense_blocks: tuple[PlanarDenseBlock, ...]
    low_rank_blocks: tuple[PlanarLowRankBlock, ...]
    requested_tolerance: float
    dense_element_equivalent: int

    @staticmethod
    def _kernel_block(
        coordinates: np.ndarray,
        reflected: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        factor: float,
        widths: np.ndarray,
    ) -> np.ndarray:
        from scipy.special import erf

        delta = (
            coordinates[left, None, :]
            - reflected[None, right, :]
        )
        distance = np.sqrt(np.einsum("ijd,ijd->ij", delta, delta))
        beta = 1.0 / np.sqrt(
            2.0
            * (
                widths[left, None] ** 2
                + widths[None, right] ** 2
            )
        )
        return factor * erf(beta * distance) / distance

    @classmethod
    def compile(
        cls,
        hierarchy: PersistentPlanarHierarchy,
        interface: PlanarDielectricInterface,
        gaussian_widths_bohr: np.ndarray,
        *,
        relative_tolerance: float,
        maximum_rank: int,
    ) -> "PersistentPlanarHMatrix":
        xyz = hierarchy.coordinates_bohr
        reflected = interface.reflected(xyz)
        dense_blocks: list[PlanarDenseBlock] = []
        low_rank_blocks: list[PlanarLowRankBlock] = []
        factor = interface.image_factor
        widths = np.asarray(gaussian_widths_bohr, dtype=float)

        def immutable(array: np.ndarray) -> np.ndarray:
            fixed = np.asarray(array).copy()
            fixed.setflags(write=False)
            return fixed

        for left_id, right_id in hierarchy.near_blocks:
            left = hierarchy.clusters[left_id].indices
            right = hierarchy.clusters[right_id].indices
            block = cls._kernel_block(
                xyz, reflected, left, right, factor, widths
            )
            if left_id == right_id:
                block = 0.5 * (block + block.T)
            dense_blocks.append(
                PlanarDenseBlock(
                    immutable(left),
                    immutable(right),
                    immutable(block),
                    left_id == right_id,
                )
            )
        for left_id, right_id in hierarchy.far_blocks:
            left = hierarchy.clusters[left_id].indices
            right = hierarchy.clusters[right_id].indices
            block = cls._kernel_block(
                xyz, reflected, left, right, factor, widths
            )
            u, singular, vt = np.linalg.svd(block, full_matrices=False)
            if len(singular) == 0 or singular[0] == 0.0:
                rank = 1
                truncation = 0.0
            else:
                rank = max(
                    1,
                    int(
                        np.count_nonzero(
                            singular
                            > float(relative_tolerance) * singular[0]
                        )
                    ),
                )
                rank = min(rank, int(maximum_rank), len(singular))
                truncation = (
                    0.0
                    if rank == len(singular)
                    else float(singular[rank] / singular[0])
                )
            compressed_elements = rank * (len(left) + len(right))
            if (
                truncation > float(relative_tolerance)
                or compressed_elements >= block.size
            ):
                dense_blocks.append(
                    PlanarDenseBlock(
                        immutable(left),
                        immutable(right),
                        immutable(block),
                        False,
                    )
                )
            else:
                low_rank_blocks.append(
                    PlanarLowRankBlock(
                        immutable(left),
                        immutable(right),
                        immutable(u[:, :rank] * singular[:rank]),
                        immutable(vt[:rank, :].T),
                        truncation,
                    )
                )
        return cls(
            len(xyz),
            tuple(dense_blocks),
            tuple(low_rank_blocks),
            float(relative_tolerance),
            len(xyz) ** 2,
        )

    def matrix_vector(self, charges: np.ndarray) -> np.ndarray:
        q = np.asarray(charges, dtype=float).reshape(-1)
        if len(q) != self.site_count:
            raise ValueError("hierarchical image vector has the wrong dimension")
        result = np.zeros_like(q)
        for block in self.dense_blocks:
            left = block.left_indices
            right = block.right_indices
            result[left] += block.matrix @ q[right]
            if not block.diagonal:
                result[right] += block.matrix.T @ q[left]
        for block in self.low_rank_blocks:
            left = block.left_indices
            right = block.right_indices
            result[left] += block.left_factor @ (
                block.right_factor.T @ q[right]
            )
            result[right] += block.right_factor @ (
                block.left_factor.T @ q[left]
            )
        return result

    def diagnostics(self) -> dict[str, Any]:
        stored = sum(block.matrix.size for block in self.dense_blocks)
        stored += sum(
            block.left_factor.size + block.right_factor.size
            for block in self.low_rank_blocks
        )
        ranks = [block.rank for block in self.low_rank_blocks]
        return {
            "schema": "matrix.zaff.interfacial_pcm.hmatrix.v1",
            "site_count": self.site_count,
            "dense_block_count": len(self.dense_blocks),
            "low_rank_block_count": len(self.low_rank_blocks),
            "maximum_rank": max(ranks, default=0),
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
            "stored_elements": stored,
            "dense_elements": self.dense_element_equivalent,
            "compression_ratio": (
                stored / self.dense_element_equivalent
                if self.dense_element_equivalent
                else 0.0
            ),
            "maximum_relative_truncation_error": max(
                (
                    block.relative_truncation_error
                    for block in self.low_rank_blocks
                ),
                default=0.0,
            ),
            "requested_tolerance": self.requested_tolerance,
        }


@dataclass(frozen=True)
class InterfacialPCMReactionField:
    """Exact image reaction field for one planar dielectric substrate."""

    interface: PlanarDielectricInterface
    spectral_compression: PlanarSpectralCompression = field(
        default_factory=PlanarSpectralCompression
    )
    gaussian_widths_bohr: np.ndarray | float | None = None

    def __post_init__(self) -> None:
        raw = self.gaussian_widths_bohr
        if raw is None:
            return
        widths = np.asarray(raw, dtype=float)
        if widths.ndim > 1 or np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
            raise ValueError("interfacial Gaussian widths must be finite and positive")
        fixed = widths.copy()
        fixed.setflags(write=False)
        object.__setattr__(self, "gaussian_widths_bohr", fixed)

    @property
    def modal_backend(self) -> str:
        """Compatibility label used by resident reaction-field consumers."""

        return "PLANAR_DIELECTRIC_GAUSSIAN_ERF_IMAGES_ANALYTIC"

    def _validated_widths(
        self,
        count: int,
        gaussian_widths_bohr: np.ndarray | float | None,
    ) -> np.ndarray:
        raw = (
            self.gaussian_widths_bohr
            if gaussian_widths_bohr is None
            else gaussian_widths_bohr
        )
        if raw is None:
            raise ValueError(
                "interfacial PCM requires Gaussian charge widths; point images "
                "are forbidden by the nano-Matrix electrostatics contract"
            )
        widths = np.asarray(raw, dtype=float)
        if widths.ndim == 0:
            widths = np.full(count, float(widths))
        if (
            widths.shape != (count,)
            or np.any(~np.isfinite(widths))
            or np.any(widths <= 0.0)
        ):
            raise ValueError(
                "interfacial Gaussian widths must have one positive value per site"
            )
        return widths

    @staticmethod
    def _gaussian_radial(
        delta: np.ndarray,
        target_widths: np.ndarray,
        source_widths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return f(r), f'(r)/r, and the radial Hessian coefficient."""

        from scipy.special import erf

        radius2 = np.einsum("...d,...d->...", delta, delta)
        radius = np.sqrt(radius2)
        beta = 1.0 / np.sqrt(
            2.0
            * (
                target_widths[:, None] ** 2
                + source_widths[None, :] ** 2
            )
        )
        argument = beta * radius
        exponential = np.exp(-(argument**2))
        error_function = erf(argument)
        root_pi = math.sqrt(math.pi)
        value = error_function / radius
        first = (
            2.0 * beta * exponential / (root_pi * radius)
            - error_function / radius2
        )
        second = (
            -4.0 * beta**3 * exponential / root_pi
            - 4.0 * beta * exponential / (root_pi * radius2)
            + 2.0 * error_function / radius**3
        )
        first_over_radius = first / radius
        outer_coefficient = (second - first_over_radius) / radius2
        return value, first_over_radius, outer_coefficient

    @staticmethod
    def _fmm_available() -> bool:
        try:
            import fmm3dpy  # noqa: F401
        except ImportError:
            return False
        return True

    def _selected_backend(
        self,
        backend: Literal["auto", "direct", "fmm", "spectral"],
        site_count: int,
        minimum_sites: int | None,
        *,
        operation: Literal[
            "energy_gradient",
            "hvp",
            "batched_hvp",
            "charge_response",
        ] = "energy_gradient",
    ) -> Literal["direct", "fmm"]:
        if backend not in {"auto", "direct", "fmm", "spectral"}:
            raise ValueError(
                "interfacial backend must be auto, direct, fmm, or spectral"
            )
        if backend in {"fmm", "spectral"}:
            if not self._fmm_available():
                raise RuntimeError(
                    "spectral interfacial PCM requires matrix-zaff[fmm]"
                )
            return "fmm"
        configured = {
            "energy_gradient": self.spectral_compression.energy_gradient_minimum_sites,
            "hvp": self.spectral_compression.hvp_minimum_sites,
            "batched_hvp": self.spectral_compression.batched_hvp_minimum_sites,
            "charge_response": self.spectral_compression.charge_response_minimum_sites,
        }[operation]
        crossover = int(configured if minimum_sites is None else minimum_sites)
        calibrated = self.spectral_compression.calibrated_platform
        if minimum_sites is None and not calibrated:
            from .native_kernels import native_zaff_backend

            if native_zaff_backend(site_count).accelerated:
                crossover = max(crossover, 32768)
        if (
            minimum_sites is None
            and calibrated
            and calibrated != planar_spectral_platform_fingerprint()
        ):
            crossover = max(crossover, 512)
        return (
            "fmm"
            if backend == "auto"
            and site_count >= crossover
            and self._fmm_available()
            else "direct"
        )

    def _precision(self, requested: float | None) -> float:
        precision = (
            self.spectral_compression.relative_tolerance
            if requested is None
            else float(requested)
        )
        if not 0.0 < precision < 1.0:
            raise ValueError("spectral FMM tolerance must lie between zero and one")
        return precision

    def _validation_indices(self, site_count: int) -> np.ndarray:
        count = min(self.spectral_compression.validation_sites, int(site_count))
        if count <= 0:
            return np.empty(0, dtype=int)
        return np.unique(
            np.linspace(0, int(site_count) - 1, count, dtype=int)
        )

    def _spectral_error_acceptable(
        self,
        approximate: np.ndarray,
        reference: np.ndarray,
        precision: float,
    ) -> bool:
        scale = max(1.0, float(np.max(np.abs(reference), initial=0.0)))
        error = float(np.max(np.abs(approximate - reference), initial=0.0))
        return error <= (
            self.spectral_compression.validation_tolerance_multiplier
            * precision
            * scale
        )

    @staticmethod
    def _spectral_error_value(
        approximate: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        return float(
            np.max(np.abs(approximate - reference), initial=0.0)
        )

    @staticmethod
    def _direct_image_targets(
        targets: np.ndarray,
        reflected_sources: np.ndarray,
        charges: np.ndarray,
        *,
        gradient: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = targets[:, None, :] - reflected_sources[None, :, :]
        radius2 = np.einsum("tsd,tsd->ts", delta, delta)
        potential = radius2**-0.5 @ charges
        field_gradient = (
            -np.einsum("ts,tsd,s->td", radius2**-1.5, delta, charges)
            if gradient
            else np.empty((0, 3), dtype=float)
        )
        return potential, field_gradient

    def _validated_system(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        gaussian_widths_bohr: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        q = np.asarray(charges, dtype=float).reshape(-1)
        if (
            xyz.shape != (len(q), 3)
            or np.any(~np.isfinite(xyz))
            or np.any(~np.isfinite(q))
        ):
            raise ValueError("interfacial PCM coordinates and charges are inconsistent")
        distances = self.interface.signed_distances(xyz)
        if np.any(distances <= self.interface.exclusion_gap_bohr + 1.0e-12):
            raise ValueError(
                "every explicit site must remain above the interfacial exclusion gap"
            )
        widths = self._validated_widths(len(q), gaussian_widths_bohr)
        return xyz, q, distances, widths

    def evaluate(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sources: int | None = None,
    ) -> InterfacialPCMResult:
        telemetry: dict[str, Any] = {
            "operation": "energy_gradient",
            "requested_backend": backend,
            "site_count": len(np.asarray(charges).reshape(-1)),
        }
        started = time.perf_counter()
        energy, gradient = self.reaction_energy_gradient(
            coordinates_bohr,
            charges,
            gaussian_widths_bohr=gaussian_widths_bohr,
            backend=backend,
            fmm_precision=fmm_precision,
            fmm_minimum_sources=fmm_minimum_sources,
            _telemetry=telemetry,
        )
        telemetry["elapsed_seconds"] = time.perf_counter() - started
        return InterfacialPCMResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            backend=self.modal_backend,
            telemetry=telemetry,
        )

    def reaction_energy(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
    ) -> float:
        xyz, q, _distances, widths = self._validated_system(
            coordinates_bohr, charges, gaussian_widths_bohr
        )
        potential = self._image_potential_gradient(
            xyz, q, widths, compute_gradient=False, backend=backend
        )[0]
        return 0.5 * self.interface.image_factor * float(np.dot(q, potential))

    def reaction_energy_gradient(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sources: int | None = None,
        _telemetry: dict[str, Any] | None = None,
    ) -> tuple[float, np.ndarray]:
        xyz, q, _distances, widths = self._validated_system(
            coordinates_bohr, charges, gaussian_widths_bohr
        )
        factor = self.interface.image_factor
        potential, potential_gradient = self._image_potential_gradient(
            xyz,
            q,
            widths,
            compute_gradient=True,
            backend=backend,
            fmm_precision=fmm_precision,
            fmm_minimum_sources=fmm_minimum_sources,
            telemetry=_telemetry,
        )
        energy = 0.5 * factor * float(np.dot(q, potential))
        gradient = factor * q[:, None] * potential_gradient
        return float(energy), gradient

    def _image_potential_gradient(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        gaussian_widths_bohr: np.ndarray,
        *,
        compute_gradient: bool,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sources: int | None = None,
        telemetry: dict[str, Any] | None = None,
        operation: Literal[
            "energy_gradient",
            "charge_response",
        ] = "energy_gradient",
    ) -> tuple[np.ndarray, np.ndarray]:
        reflected = self.interface.reflected(coordinates_bohr)
        selected = self._selected_backend(
            backend,
            len(coordinates_bohr),
            fmm_minimum_sources,
            operation=operation,
        )
        precision = self._precision(fmm_precision)
        if telemetry is not None:
            telemetry.update(
                {
                    "selected_backend": selected,
                    "actual_backend": selected,
                    "fallback_to_direct": False,
                    "requested_tolerance": precision,
                    "spectral_order": (
                        "FMM3D_AUTOMATIC" if selected == "fmm" else "DIRECT"
                    ),
                    "validation_sites": 0,
                    "validation_max_abs_error": 0.0,
                }
            )
        potential = np.empty(len(coordinates_bohr), dtype=float)
        gradient = (
            np.empty_like(coordinates_bohr)
            if compute_gradient
            else np.empty((0, 3), dtype=float)
        )
        block_size = self.spectral_compression.direct_block_size
        for start in range(0, len(coordinates_bohr), block_size):
            stop = min(len(coordinates_bohr), start + block_size)
            delta = coordinates_bohr[start:stop, None, :] - reflected[None, :, :]
            value, first_over_radius, _outer = self._gaussian_radial(
                delta,
                gaussian_widths_bohr[start:stop],
                gaussian_widths_bohr,
            )
            potential[start:stop] = value @ charges
            if compute_gradient:
                gradient[start:stop] = np.einsum(
                    "ts,tsd,s->td", first_over_radius, delta, charges
                )
        if telemetry is not None:
            telemetry["actual_backend"] = "blocked_gaussian_direct"
            telemetry["fallback_to_direct"] = selected == "fmm"
            telemetry["penetration_model"] = "ERF_GAUSSIAN_ALL_IMAGE_PAIRS"
        return potential, gradient

    def spectral_diagnostics(
        self,
        site_count: int,
        *,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        operation: Literal[
            "energy_gradient",
            "hvp",
            "batched_hvp",
            "charge_response",
        ] = "energy_gradient",
    ) -> dict[str, Any]:
        """Describe the compression selected without executing an interaction."""

        selected = self._selected_backend(
            backend,
            int(site_count),
            None,
            operation=operation,
        )
        return {
            "method": (
                "ADAPTIVE_SPHERICAL_HARMONIC_FMM"
                if selected == "fmm"
                else "BLOCKED_DIRECT_IMAGES"
            ),
            "selected_backend": selected,
            "relative_tolerance": self.spectral_compression.relative_tolerance,
            "minimum_sites": self.spectral_compression.minimum_sites,
            "operation": operation,
            "operator_minimum_sites": {
                "energy_gradient": (
                    self.spectral_compression.energy_gradient_minimum_sites
                ),
                "hvp": self.spectral_compression.hvp_minimum_sites,
                "batched_hvp": (
                    self.spectral_compression.batched_hvp_minimum_sites
                ),
                "charge_response": (
                    self.spectral_compression.charge_response_minimum_sites
                ),
            }[operation],
            "direct_block_size": self.spectral_compression.direct_block_size,
            "validation_sites": self.spectral_compression.validation_sites,
            "validation_tolerance_multiplier": (
                self.spectral_compression.validation_tolerance_multiplier
            ),
            "calibrated_platform": self.spectral_compression.calibrated_platform,
            "current_platform": planar_spectral_platform_fingerprint(),
            "calibration_matches_platform": (
                not self.spectral_compression.calibrated_platform
                or self.spectral_compression.calibrated_platform
                == planar_spectral_platform_fingerprint()
            ),
            "near_field": "EXACT",
            "derivatives": "ANALYTIC_E_G_H_HVP",
        }

    def compile_persistent_operator(
        self,
        coordinates_bohr: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
    ) -> PersistentPlanarImageOperator:
        """Compile a fixed-geometry operator for repeated charge vectors."""

        xyz = np.asarray(coordinates_bohr, dtype=float)
        self.interface.validate_explicit_coordinates(xyz)
        widths = self._validated_widths(len(xyz), gaussian_widths_bohr)
        selected = self._selected_backend(
            backend,
            len(xyz),
            None,
            operation="charge_response",
        )
        dense = None
        hierarchy = None
        hierarchical_kernel = None
        if len(xyz) >= self.spectral_compression.hierarchical_minimum_sites:
            hierarchy = PersistentPlanarHierarchy.build(
                xyz,
                self.interface,
                leaf_size=16,
                admissibility=0.5,
            )
            hierarchical_kernel = PersistentPlanarHMatrix.compile(
                hierarchy,
                self.interface,
                widths,
                relative_tolerance=(
                    self.spectral_compression.hierarchical_relative_tolerance
                ),
                maximum_rank=self.spectral_compression.hierarchical_max_rank,
            )
            selected = "hierarchical"
        elif selected == "direct":
            reflected = self.interface.reflected(xyz)
            dense = np.empty((len(xyz), len(xyz)), dtype=float)
            block_size = self.spectral_compression.direct_block_size
            for start in range(0, len(xyz), block_size):
                stop = min(len(xyz), start + block_size)
                delta = xyz[start:stop, None, :] - reflected[None, :, :]
                value, _first, _outer = self._gaussian_radial(
                    delta, widths[start:stop], widths
                )
                dense[start:stop] = self.interface.image_factor * value
            dense = 0.5 * (dense + dense.T)
            dense.setflags(write=False)
        else:
            hierarchy = PersistentPlanarHierarchy.build(
                xyz,
                self.interface,
            )
        fixed_xyz = xyz.copy()
        fixed_xyz.setflags(write=False)
        return PersistentPlanarImageOperator(
            reaction_field=self,
            coordinates_bohr=fixed_xyz,
            gaussian_widths_bohr=widths.copy(),
            selected_backend=selected,
            dense_kernel=dense,
            hierarchy=hierarchy,
            hierarchical_kernel=hierarchical_kernel,
        )

    def physical_diagnostics(
        self,
        coordinates_bohr: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        tolerance: float = 1.0e-11,
        maximum_sites: int = 256,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Audit reciprocity, sign, clearance and parallel invariance."""

        xyz = np.asarray(coordinates_bohr, dtype=float)
        self.interface.validate_explicit_coordinates(xyz)
        widths = self._validated_widths(len(xyz), gaussian_widths_bohr)
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("physical diagnostic tolerance must be positive")
        if int(maximum_sites) < 1:
            raise ValueError("physical diagnostic site limit must be positive")
        if len(xyz) > int(maximum_sites):
            indices = np.linspace(
                0, len(xyz) - 1, int(maximum_sites), dtype=int
            )
            audited_xyz = xyz[indices]
        else:
            audited_xyz = xyz
        operator = self.compile_persistent_operator(
            audited_xyz,
            gaussian_widths_bohr=(
                widths[indices] if len(xyz) > int(maximum_sites) else widths
            ),
            backend="direct",
        )
        kernel = np.asarray(operator.dense_kernel, dtype=float)
        reciprocity_error = float(np.max(np.abs(kernel - kernel.T), initial=0.0))
        eigenvalues = np.linalg.eigvalsh(kernel)
        largest_eigenvalue = float(np.max(eigenvalues, initial=0.0))
        smallest_eigenvalue = float(np.min(eigenvalues, initial=0.0))
        tangent = np.asarray((1.0, 0.0, 0.0))
        if abs(float(tangent @ self.interface.normal)) > 0.8:
            tangent = np.asarray((0.0, 1.0, 0.0))
        tangent -= float(tangent @ self.interface.normal) * self.interface.normal
        tangent /= np.linalg.norm(tangent)
        shifted = self.compile_persistent_operator(
            audited_xyz + 0.731 * tangent,
            gaussian_widths_bohr=(
                widths[indices] if len(xyz) > int(maximum_sites) else widths
            ),
            backend="direct",
        )
        translation_error = float(
            np.max(
                np.abs(kernel - np.asarray(shifted.dense_kernel)),
                initial=0.0,
            )
        )
        clearance = self.interface.signed_distances(audited_xyz)
        sign_ok = (
            largest_eigenvalue <= threshold
            if self.interface.image_factor <= 0.0
            else smallest_eigenvalue >= -threshold
        )
        passed = (
            reciprocity_error <= threshold
            and translation_error <= threshold
            and sign_ok
            and abs(self.interface.image_factor) <= 1.0 + threshold
        )
        diagnostics = {
            "schema": "matrix.zaff.interfacial_pcm.diagnostics.v1",
            "passed": passed,
            "audited_sites": len(audited_xyz),
            "total_sites": len(xyz),
            "reciprocity_max_abs_error": reciprocity_error,
            "parallel_translation_max_abs_error": translation_error,
            "smallest_kernel_eigenvalue": smallest_eigenvalue,
            "largest_kernel_eigenvalue": largest_eigenvalue,
            "minimum_clearance_bohr": float(np.min(clearance)),
            "image_factor": self.interface.image_factor,
            "negative_semidefinite_expected": self.interface.image_factor <= 0.0,
            "tolerance": threshold,
        }
        if strict and not passed:
            raise RuntimeError(
                "interfacial PCM physical diagnostics failed: "
                f"{diagnostics}"
            )
        return diagnostics

    def reaction_energy_gradient_hessian(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
    ) -> InterfacialPCMSecondOrderResult:
        xyz, q, _distances, widths = self._validated_system(
            coordinates_bohr, charges, gaussian_widths_bohr
        )
        energy, gradient = self.reaction_energy_gradient(
            xyz, q, gaussian_widths_bohr=widths
        )
        count = len(xyz)
        hessian = np.zeros((count, 3, count, 3), dtype=float)
        factor = self.interface.image_factor
        reflection = self.interface.reflection
        reflected = self.interface.reflected(xyz)
        identity = np.eye(3)
        # A dense Hessian necessarily has O(N^2) output, but each target row is
        # assembled by vectorized analytic blocks instead of Python pair loops:
        # H_ik = delta_ik a q_i sum_j q_j C_ij - a q_i q_k C_ik P.
        for target in range(count):
            delta = xyz[target] - reflected
            _value, first_over_radius, outer = self._gaussian_radial(
                delta[None, :, :], widths[target : target + 1], widths
            )
            curvature = (
                outer[0, :, None, None]
                * delta[:, :, None]
                * delta[:, None, :]
                + identity[None, :, :] * first_over_radius[0, :, None, None]
            )
            hessian[target, :, :, :] = np.moveaxis(
                -factor
                * q[target]
                * q[:, None, None]
                * np.einsum("nij,jk->nik", curvature, reflection),
                0,
                1,
            )
            hessian[target, :, target, :] += (
                factor
                * q[target]
                * np.einsum("n,nij->ij", q, curvature)
            )
        matrix = hessian.reshape(3 * count, 3 * count)
        return InterfacialPCMSecondOrderResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            hessian_hartree_per_bohr2=matrix,
            backend=self.modal_backend,
        )

    def hessian_vector_product(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        vector_bohr: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sites: int | None = None,
    ) -> np.ndarray:
        return self.hessian_vector_products(
            coordinates_bohr,
            charges,
            np.asarray(vector_bohr, dtype=float)[None, ...],
            gaussian_widths_bohr=gaussian_widths_bohr,
            backend=backend,
            fmm_precision=fmm_precision,
            fmm_minimum_sites=fmm_minimum_sites,
        )[0]

    def hessian_vector_products(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        vectors_bohr: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sites: int | None = None,
        _telemetry: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply the Cartesian Hessian to several directions in one traversal."""

        xyz, q, _distances, widths = self._validated_system(
            coordinates_bohr, charges, gaussian_widths_bohr
        )
        directions = np.asarray(vectors_bohr, dtype=float)
        if directions.ndim == 2 and directions.shape[1] == xyz.size:
            directions = directions.reshape(len(directions), *xyz.shape)
        elif directions.shape == xyz.shape:
            directions = directions[None, ...]
        if (
            directions.ndim != 3
            or directions.shape[1:] != xyz.shape
            or np.any(~np.isfinite(directions))
        ):
            raise ValueError(
                "interfacial PCM Hessian-vector batch dimensions are inconsistent"
            )
        product = np.zeros_like(directions)
        reflection = self.interface.reflection
        reflected = self.interface.reflected(xyz)
        reflected_directions = directions @ reflection
        identity = np.eye(3)
        block_size = self.spectral_compression.direct_block_size
        for start in range(0, len(xyz), block_size):
            stop = min(len(xyz), start + block_size)
            delta = xyz[start:stop, None, :] - reflected[None, :, :]
            _value, first_over_radius, outer = self._gaussian_radial(
                delta, widths[start:stop], widths
            )
            curvature = (
                outer[:, :, None, None]
                * delta[:, :, :, None]
                * delta[:, :, None, :]
                + identity[None, None, :, :]
                * first_over_radius[:, :, None, None]
            )
            effective = (
                directions[:, start:stop, None, :]
                - reflected_directions[:, None, :, :]
            )
            product[:, start:stop] = np.einsum(
                "s,tsij,mtsj->mti", q, curvature, effective
            )
        if _telemetry is not None:
            selected_backend = self._selected_backend(
                backend,
                len(xyz),
                fmm_minimum_sites,
                operation=("hvp" if len(directions) == 1 else "batched_hvp"),
            )
            _telemetry.update(
                {
                    "operation": "gaussian_erf_hvp",
                    "requested_backend": backend,
                    "selected_backend": selected_backend,
                    "actual_backend": "direct",
                    "fallback_to_direct": selected_backend == "fmm",
                    "penetration_model": "ERF_GAUSSIAN_ALL_IMAGE_PAIRS",
                    "site_count": len(xyz),
                    "direction_count": len(directions),
                }
            )
        return (
            self.interface.image_factor * q[None, :, None] * product
        ).reshape(len(directions), -1)

        # Legacy point-image implementations below are intentionally
        # unreachable; retained temporarily while compiled Gaussian kernels
        # are introduced behind the identical public contract.
        selected = self._selected_backend(
            backend,
            len(xyz),
            fmm_minimum_sites,
            operation=(
                "hvp" if len(directions) == 1 else "batched_hvp"
            ),
        )
        precision = self._precision(fmm_precision)
        if _telemetry is not None:
            _telemetry.update(
                {
                    "operation": (
                        "hvp" if len(directions) == 1 else "batched_hvp"
                    ),
                    "requested_backend": backend,
                    "selected_backend": selected,
                    "actual_backend": selected,
                    "fallback_to_direct": False,
                    "requested_tolerance": precision,
                    "spectral_order": (
                        "FMM3D_AUTOMATIC" if selected == "fmm" else "DIRECT"
                    ),
                    "validation_sites": 0,
                    "validation_max_abs_error": 0.0,
                    "site_count": len(xyz),
                    "direction_count": len(directions),
                }
            )
        if selected == "fmm":
            product = self._spectral_hessian_vector_products(
                xyz,
                q,
                directions,
                precision=precision,
            )
            audit = self._validation_indices(len(xyz))
            if len(audit):
                reflected = self.interface.reflected(xyz)
                reflected_directions = directions @ self.interface.reflection
                delta = xyz[audit, None, :] - reflected[None, :, :]
                radius2 = np.einsum("tsd,tsd->ts", delta, delta)
                curvature = (
                    3.0
                    * delta[:, :, :, None]
                    * delta[:, :, None, :]
                    * radius2[:, :, None, None] ** -2.5
                    - np.eye(3)[None, None, :, :]
                    * radius2[:, :, None, None] ** -1.5
                )
                effective = (
                    directions[:, audit, None, :]
                    - reflected_directions[:, None, :, :]
                )
                reference = np.einsum(
                    "s,tsij,mtsj->mti", q, curvature, effective
                )
                error = self._spectral_error_value(
                    product[:, audit], reference
                )
                if _telemetry is not None:
                    _telemetry["validation_sites"] = len(audit)
                    _telemetry["validation_max_abs_error"] = error
                if not self._spectral_error_acceptable(
                    product[:, audit], reference, precision
                ):
                    if _telemetry is not None:
                        _telemetry["fallback_to_direct"] = True
                        _telemetry["actual_backend"] = "direct"
                    return self.hessian_vector_products(
                        xyz,
                        q,
                        directions,
                        backend="direct",
                    )
            return (
                self.interface.image_factor
                * q[None, :, None]
                * product
            ).reshape(len(directions), -1)

        from .native_kernels import (
            native_zaff_backend,
            planar_image_hessian_vectors,
        )

        if native_zaff_backend(len(xyz)).accelerated:
            if _telemetry is not None:
                _telemetry["actual_backend"] = "native_direct"
            return planar_image_hessian_vectors(
                xyz,
                q,
                self.interface.origin_bohr,
                self.interface.normal,
                directions,
                self.interface.image_factor,
            ).reshape(len(directions), -1)

        product = np.zeros_like(directions)
        reflection = self.interface.reflection
        reflected = self.interface.reflected(xyz)
        reflected_directions = directions @ reflection
        identity = np.eye(3)
        block_size = self.spectral_compression.direct_block_size
        for start in range(0, len(xyz), block_size):
            stop = min(len(xyz), start + block_size)
            delta = xyz[start:stop, None, :] - reflected[None, :, :]
            radius2 = np.einsum("tsd,tsd->ts", delta, delta)
            curvature = (
                3.0
                * delta[:, :, :, None]
                * delta[:, :, None, :]
                * radius2[:, :, None, None] ** -2.5
                - identity[None, None, :, :]
                * radius2[:, :, None, None] ** -1.5
            )
            effective = (
                directions[:, start:stop, None, :]
                - reflected_directions[:, None, :, :]
            )
            product[:, start:stop] = np.einsum(
                "s,tsij,mtsj->mti", q, curvature, effective
            )
        return (
            self.interface.image_factor * q[None, :, None] * product
        ).reshape(len(directions), -1)

    def hessian_vector_products_with_telemetry(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        vectors_bohr: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sites: int | None = None,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        telemetry: dict[str, Any] = {}
        started = time.perf_counter()
        products = self.hessian_vector_products(
            coordinates_bohr,
            charges,
            vectors_bohr,
            gaussian_widths_bohr=gaussian_widths_bohr,
            backend=backend,
            fmm_precision=fmm_precision,
            fmm_minimum_sites=fmm_minimum_sites,
            _telemetry=telemetry,
        )
        telemetry["elapsed_seconds"] = time.perf_counter() - started
        return products, telemetry

    def _spectral_hessian_vector_products(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        directions_bohr: np.ndarray,
        *,
        precision: float,
    ) -> np.ndarray:
        """Differentiate the image field through FMM multipole spectra."""

        from fmm3dpy import lfmm3d
        from .nonbonded import FOUR_PI

        reflection = self.interface.reflection
        reflected = self.interface.reflected(coordinates_bohr)
        reflected_directions = directions_bohr @ reflection
        density_count = len(directions_bohr)
        sources = np.asfortranarray(reflected.T)
        targets = np.asfortranarray(coordinates_bohr.T)
        charge_result = lfmm3d(
            eps=precision,
            sources=sources,
            charges=np.asfortranarray(charges),
            targets=targets,
            pgt=3,
        )
        dipoles = np.moveaxis(
            charges[None, :, None] * reflected_directions,
            2,
            1,
        )
        dipole_result = lfmm3d(
            eps=precision,
            sources=sources,
            dipvec=np.asfortranarray(
                dipoles[0] if density_count == 1 else dipoles
            ),
            targets=targets,
            pgt=2,
            nd=density_count,
        )
        packed = FOUR_PI * np.asarray(
            charge_result.hesstarg, dtype=float
        ).reshape(6, -1)
        target_hessian = np.empty((len(charges), 3, 3), dtype=float)
        target_hessian[:, 0, 0] = packed[0]
        target_hessian[:, 1, 1] = packed[1]
        target_hessian[:, 2, 2] = packed[2]
        target_hessian[:, 0, 1] = target_hessian[:, 1, 0] = packed[3]
        target_hessian[:, 0, 2] = target_hessian[:, 2, 0] = packed[4]
        target_hessian[:, 1, 2] = target_hessian[:, 2, 1] = packed[5]
        dipole_gradient = FOUR_PI * np.asarray(
            dipole_result.gradtarg, dtype=float
        ).reshape(density_count, 3, -1).transpose(0, 2, 1)
        return (
            np.einsum("nij,mnj->mni", target_hessian, directions_bohr)
            + dipole_gradient
        )

    def reaction_potential(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        targets_bohr: np.ndarray | None = None,
        target_gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
    ) -> np.ndarray:
        xyz, q, _distances, widths = self._validated_system(
            coordinates_bohr, charges, gaussian_widths_bohr
        )
        targets = xyz if targets_bohr is None else np.asarray(targets_bohr, dtype=float)
        self.interface.validate_explicit_coordinates(targets)
        if targets_bohr is None:
            potential = self._image_potential_gradient(
                xyz,
                q,
                widths,
                compute_gradient=False,
                backend=backend,
                operation="charge_response",
            )[0]
            return self.interface.image_factor * potential

        reflected = self.interface.reflected(xyz)
        potential = np.zeros(len(targets), dtype=float)
        if target_gaussian_widths_bohr is None:
            raise ValueError(
                "Gaussian interfacial potentials at arbitrary targets require "
                "explicit target widths"
            )
        target_widths = np.asarray(target_gaussian_widths_bohr, dtype=float)
        if target_widths.ndim == 0:
            target_widths = np.full(len(targets), float(target_widths))
        if (
            target_widths.shape != (len(targets),)
            or np.any(~np.isfinite(target_widths))
            or np.any(target_widths <= 0.0)
        ):
            raise ValueError(
                "target Gaussian widths must have one positive value per target"
            )
        delta = targets[:, None, :] - reflected[None, :, :]
        value, _first, _outer = self._gaussian_radial(
            delta, target_widths, widths
        )
        potential[:] = value @ q
        return self.interface.image_factor * potential

    def kernel_product(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
    ) -> np.ndarray:
        return self.reaction_potential(
            coordinates_bohr,
            charges,
            gaussian_widths_bohr=gaussian_widths_bohr,
            backend=backend,
        )

    def kernel_product_with_telemetry(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        xyz, q, _distances, widths = self._validated_system(
            coordinates_bohr, charges, gaussian_widths_bohr
        )
        telemetry: dict[str, Any] = {
            "operation": "charge_response",
            "requested_backend": backend,
            "site_count": len(xyz),
        }
        started = time.perf_counter()
        potential = self._image_potential_gradient(
            xyz,
            q,
            widths,
            compute_gradient=False,
            backend=backend,
            telemetry=telemetry,
            operation="charge_response",
        )[0]
        telemetry["operation"] = "charge_response"
        telemetry["elapsed_seconds"] = time.perf_counter() - started
        return self.interface.image_factor * potential, telemetry

    def kernel_directional_product(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        vector_bohr: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sites: int | None = None,
        _telemetry: dict[str, Any] | None = None,
    ) -> np.ndarray:
        xyz, q, _distances, widths = self._validated_system(
            coordinates_bohr, charges, gaussian_widths_bohr
        )
        direction = np.asarray(vector_bohr, dtype=float)
        if direction.size == xyz.size:
            direction = direction.reshape(xyz.shape)
        if direction.shape != xyz.shape or np.any(~np.isfinite(direction)):
            raise ValueError(
                "interfacial PCM kernel directional vector has the wrong shape"
            )
        reflection = self.interface.reflection
        reflected = self.interface.reflected(xyz)
        reflected_direction = direction @ reflection
        selected = self._selected_backend(
            backend,
            len(xyz),
            fmm_minimum_sites,
            operation="charge_response",
        )
        precision = self._precision(fmm_precision)
        if _telemetry is not None:
            _telemetry.update(
                {
                    "operation": "kernel_directional_product",
                    "requested_backend": backend,
                    "selected_backend": selected,
                    "actual_backend": selected,
                    "fallback_to_direct": False,
                    "requested_tolerance": precision,
                    "spectral_order": (
                        "FMM3D_AUTOMATIC" if selected == "fmm" else "DIRECT"
                    ),
                    "validation_sites": 0,
                    "validation_max_abs_error": 0.0,
                    "site_count": len(xyz),
                }
            )
        result = np.zeros(len(xyz), dtype=float)
        block_size = self.spectral_compression.direct_block_size
        for start in range(0, len(xyz), block_size):
            stop = min(len(xyz), start + block_size)
            delta = xyz[start:stop, None, :] - reflected[None, :, :]
            _value, first_over_radius, _outer = self._gaussian_radial(
                delta, widths[start:stop], widths
            )
            effective = (
                direction[start:stop, None, :]
                - reflected_direction[None, :, :]
            )
            result[start:stop] = np.einsum(
                "s,ts,tsd,tsd->t",
                q,
                first_over_radius,
                delta,
                effective,
            )
        if _telemetry is not None:
            _telemetry["actual_backend"] = "blocked_gaussian_direct"
            _telemetry["fallback_to_direct"] = selected == "fmm"
        return self.interface.image_factor * result

        # Point-image implementation retained unreachable during the
        # transition to compiled Gaussian kernels.
        if selected == "fmm":
            from fmm3dpy import lfmm3d
            from .nonbonded import FOUR_PI

            sources = np.asfortranarray(reflected.T)
            targets = np.asfortranarray(xyz.T)
            charge_result = lfmm3d(
                eps=precision,
                sources=sources,
                charges=np.asfortranarray(q),
                targets=targets,
                pgt=2,
            )
            dipole_result = lfmm3d(
                eps=precision,
                sources=sources,
                dipvec=np.asfortranarray(
                    (q[:, None] * reflected_direction).T
                ),
                targets=targets,
                pgt=1,
            )
            gradient = FOUR_PI * np.asarray(
                charge_result.gradtarg, dtype=float
            ).reshape(3, -1).T
            result = np.einsum("nd,nd->n", gradient, direction)
            result += FOUR_PI * np.asarray(
                dipole_result.pottarg, dtype=float
            ).reshape(-1)
            audit = self._validation_indices(len(xyz))
            if len(audit):
                delta = xyz[audit, None, :] - reflected[None, :, :]
                radius2 = np.einsum("tsd,tsd->ts", delta, delta)
                effective = (
                    direction[audit, None, :]
                    - reflected_direction[None, :, :]
                )
                reference = -np.einsum(
                    "s,tsd,tsd,ts->t",
                    q,
                    delta,
                    effective,
                    radius2**-1.5,
                )
                error = self._spectral_error_value(
                    result[audit], reference
                )
                if _telemetry is not None:
                    _telemetry["validation_sites"] = len(audit)
                    _telemetry["validation_max_abs_error"] = error
                if not self._spectral_error_acceptable(
                    result[audit], reference, precision
                ):
                    if _telemetry is not None:
                        _telemetry["fallback_to_direct"] = True
                        _telemetry["actual_backend"] = "direct"
                    return self.kernel_directional_product(
                        xyz,
                        q,
                        direction,
                        backend="direct",
                    )
            return self.interface.image_factor * result

        result = np.zeros(len(xyz), dtype=float)
        block_size = self.spectral_compression.direct_block_size
        for start in range(0, len(xyz), block_size):
            stop = min(len(xyz), start + block_size)
            delta = xyz[start:stop, None, :] - reflected[None, :, :]
            radius2 = np.einsum("tsd,tsd->ts", delta, delta)
            effective = (
                direction[start:stop, None, :]
                - reflected_direction[None, :, :]
            )
            result[start:stop] = -np.einsum(
                "s,tsd,tsd,ts->t",
                q,
                delta,
                effective,
                radius2**-1.5,
            )
        return self.interface.image_factor * result

    def kernel_directional_product_with_telemetry(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        vector_bohr: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        fmm_precision: float | None = None,
        fmm_minimum_sites: int | None = None,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        telemetry: dict[str, Any] = {}
        started = time.perf_counter()
        product = self.kernel_directional_product(
            coordinates_bohr,
            charges,
            vector_bohr,
            gaussian_widths_bohr=gaussian_widths_bohr,
            backend=backend,
            fmm_precision=fmm_precision,
            fmm_minimum_sites=fmm_minimum_sites,
            _telemetry=telemetry,
        )
        telemetry["elapsed_seconds"] = time.perf_counter() - started
        return product, telemetry

    def reaction_batch_energies(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
        backend: Literal["auto", "direct", "fmm", "spectral"] = "auto",
        geometry_chunk_size: int | None = None,
    ) -> np.ndarray:
        geometries = np.asarray(coordinates_bohr, dtype=float)
        q = np.asarray(charges, dtype=float).reshape(-1)
        widths = self._validated_widths(len(q), gaussian_widths_bohr)
        if (
            geometries.ndim != 3
            or geometries.shape[1:] != (len(q), 3)
            or np.any(~np.isfinite(geometries))
        ):
            raise ValueError(
                "interfacial PCM batch must have shape (ngeometry, nsite, 3)"
            )
        if len(geometries) == 0:
            return np.zeros(0, dtype=float)
        distances = (
            geometries - self.interface.origin_bohr[None, None, :]
        ) @ self.interface.normal
        if np.any(
            distances <= self.interface.exclusion_gap_bohr + 1.0e-12
        ):
            raise ValueError(
                "every explicit site must remain above the interfacial exclusion gap"
            )
        selected = self._selected_backend(backend, len(q), None)
        if selected == "fmm":
            return np.asarray(
                [
                    self.reaction_energy(
                        geometry,
                        q,
                        gaussian_widths_bohr=widths,
                        backend="spectral",
                    )
                    for geometry in geometries
                ],
                dtype=float,
            )

        pair_count = max(1, len(q) ** 2)
        chunk = (
            max(1, 1_000_000 // pair_count)
            if geometry_chunk_size is None
            else max(1, int(geometry_chunk_size))
        )
        energies = np.empty(len(geometries), dtype=float)
        reflection = self.interface.reflection
        origin = self.interface.origin_bohr
        factor = self.interface.image_factor
        for start in range(0, len(geometries), chunk):
            stop = min(len(geometries), start + chunk)
            block = geometries[start:stop]
            reflected = origin + (block - origin) @ reflection
            delta = block[:, :, None, :] - reflected[:, None, :, :]
            from scipy.special import erf

            distance = np.sqrt(np.einsum("gijd,gijd->gij", delta, delta))
            beta = 1.0 / np.sqrt(
                2.0 * (widths[:, None] ** 2 + widths[None, :] ** 2)
            )
            inverse_distance = erf(beta[None, :, :] * distance) / distance
            energies[start:stop] = 0.5 * factor * np.einsum(
                "i,gij,j->g", q, inverse_distance, q
            )
        return energies

    def charge_flow_curvature(
        self,
        coordinates_bohr: np.ndarray,
        left_site: int,
        right_site: int,
        scale: float,
        *,
        gaussian_widths_bohr: np.ndarray | float | None = None,
    ) -> float:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        self.interface.validate_explicit_coordinates(xyz)
        if not (
            0 <= int(left_site) < len(xyz)
            and 0 <= int(right_site) < len(xyz)
            and int(left_site) != int(right_site)
        ):
            raise ValueError(
                "interfacial PCM charge-flow endpoints are inconsistent"
            )
        flow = np.zeros(len(xyz), dtype=float)
        flow[int(left_site)] = float(scale)
        flow[int(right_site)] = -float(scale)
        return float(
            np.dot(
                flow,
                self.kernel_product(
                    xyz, flow, gaussian_widths_bohr=gaussian_widths_bohr
                ),
            )
        )


def interfacial_pcm_to_record(
    interface: PlanarDielectricInterface,
    *,
    gaussian_widths_bohr: np.ndarray | float,
    spectral_compression: PlanarSpectralCompression | None = None,
) -> dict[str, Any]:
    compression = spectral_compression or PlanarSpectralCompression()
    widths = np.asarray(gaussian_widths_bohr, dtype=float)
    if widths.ndim > 1 or np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("interfacial Gaussian widths must be finite and positive")
    return {
        "schema": ZAFF_INTERFACIAL_PCM_SCHEMA,
        "boundary_contract": "FROZEN_PLANAR_INTERFACE",
        "origin_bohr": interface.origin_bohr.tolist(),
        "normal": interface.normal.tolist(),
        "explicit_side": "POSITIVE",
        "explicit_dielectric": 1.0,
        "substrate_dielectric": (
            "INFINITY"
            if math.isinf(interface.substrate_dielectric)
            else float(interface.substrate_dielectric)
        ),
        "exclusion_gap_bohr": float(interface.exclusion_gap_bohr),
        "backend": "ANALYTIC_PLANAR_GAUSSIAN_ERF_IMAGE_GREEN_FUNCTION",
        "electrostatics": "GAUSSIAN_ERF_PENETRATION",
        "gaussian_widths_bohr": widths.tolist(),
        "spectral_compression": {
            "method": "ADAPTIVE_SPHERICAL_HARMONIC_FMM",
            "relative_tolerance": compression.relative_tolerance,
            "minimum_sites": compression.minimum_sites,
            "energy_gradient_minimum_sites": (
                compression.energy_gradient_minimum_sites
            ),
            "hvp_minimum_sites": compression.hvp_minimum_sites,
            "batched_hvp_minimum_sites": (
                compression.batched_hvp_minimum_sites
            ),
            "charge_response_minimum_sites": (
                compression.charge_response_minimum_sites
            ),
            "hierarchical_minimum_sites": (
                compression.hierarchical_minimum_sites
            ),
            "hierarchical_relative_tolerance": (
                compression.hierarchical_relative_tolerance
            ),
            "hierarchical_max_rank": compression.hierarchical_max_rank,
            "direct_block_size": compression.direct_block_size,
            "validation_sites": compression.validation_sites,
            "validation_tolerance_multiplier": (
                compression.validation_tolerance_multiplier
            ),
            "calibrated_platform": compression.calibrated_platform,
        },
        "derivative_contract": "ANALYTIC_CONTINUOUS_E_G_H_HVP",
    }


def interfacial_pcm_from_record(
    record: Mapping[str, Any],
) -> PlanarDielectricInterface:
    if str(record.get("schema", "")) != ZAFF_INTERFACIAL_PCM_SCHEMA:
        raise ValueError("unsupported ZAFF interfacial PCM record")
    if str(record.get("explicit_side", "POSITIVE")).upper() != "POSITIVE":
        raise ValueError("INTERPHASES v1 supports the positive explicit side")
    raw_substrate = record.get("substrate_dielectric")
    substrate = (
        math.inf
        if str(raw_substrate).upper() == "INFINITY"
        else float(raw_substrate)
    )
    return PlanarDielectricInterface(
        origin_bohr=np.asarray(record["origin_bohr"], dtype=float),
        normal=np.asarray(record["normal"], dtype=float),
        substrate_dielectric=substrate,
        exclusion_gap_bohr=float(record.get("exclusion_gap_bohr", 0.0)),
        explicit_dielectric=float(record.get("explicit_dielectric", 1.0)),
    )


def interfacial_pcm_reaction_field_from_record(
    record: Mapping[str, Any],
) -> InterfacialPCMReactionField:
    if record.get("gaussian_widths_bohr") is None:
        raise ValueError(
            "ZAFF interfacial PCM v2 records require Gaussian charge widths"
        )
    raw = record.get("spectral_compression", {})
    compression = PlanarSpectralCompression(
        relative_tolerance=float(raw.get("relative_tolerance", 1.0e-10)),
        minimum_sites=int(raw.get("minimum_sites", 512)),
        energy_gradient_minimum_sites=int(
            raw.get(
                "energy_gradient_minimum_sites",
                raw.get("minimum_sites", 512),
            )
        ),
        hvp_minimum_sites=int(
            raw.get("hvp_minimum_sites", raw.get("minimum_sites", 512))
        ),
        batched_hvp_minimum_sites=int(
            raw.get(
                "batched_hvp_minimum_sites",
                raw.get("minimum_sites", 512),
            )
        ),
        charge_response_minimum_sites=int(
            raw.get(
                "charge_response_minimum_sites",
                raw.get("minimum_sites", 512),
            )
        ),
        hierarchical_minimum_sites=int(
            raw.get("hierarchical_minimum_sites", 512)
        ),
        hierarchical_relative_tolerance=float(
            raw.get("hierarchical_relative_tolerance", 1.0e-10)
        ),
        hierarchical_max_rank=int(raw.get("hierarchical_max_rank", 128)),
        direct_block_size=int(raw.get("direct_block_size", 256)),
        validation_sites=int(raw.get("validation_sites", 8)),
        validation_tolerance_multiplier=float(
            raw.get("validation_tolerance_multiplier", 50.0)
        ),
        calibrated_platform=str(raw.get("calibrated_platform", "")),
    )
    return InterfacialPCMReactionField(
        interfacial_pcm_from_record(record),
        spectral_compression=compression,
        gaussian_widths_bohr=record.get("gaussian_widths_bohr"),
    )


def attach_interfacial_pcm_reaction_field(
    model: Mapping[str, Any],
    interface: PlanarDielectricInterface,
    *,
    gaussian_widths_bohr: np.ndarray | float,
    spectral_compression: PlanarSpectralCompression | None = None,
) -> dict[str, Any]:
    if model.get("cpcm_reaction_field") is not None:
        raise ValueError(
            "homogeneous CPCM and interfacial PCM are alternative reaction fields"
        )
    payload = dict(model)
    payload["interfacial_pcm_reaction_field"] = interfacial_pcm_to_record(
        interface,
        spectral_compression=spectral_compression,
        gaussian_widths_bohr=gaussian_widths_bohr,
    )
    return payload


__all__ = [
    "ZAFF_INTERFACIAL_PCM_SCHEMA",
    "InterfacialPCMReactionField",
    "InterfacialPCMResult",
    "InterfacialPCMSecondOrderResult",
    "PlanarDielectricInterface",
    "PlanarCluster",
    "PlanarDenseBlock",
    "PlanarLowRankBlock",
    "PlanarSpectralCompression",
    "PersistentPlanarImageOperator",
    "PersistentPlanarHierarchy",
    "PersistentPlanarHMatrix",
    "calibrate_planar_spectral_crossover",
    "attach_interfacial_pcm_reaction_field",
    "interfacial_pcm_from_record",
    "interfacial_pcm_reaction_field_from_record",
    "interfacial_pcm_to_record",
    "planar_spectral_platform_fingerprint",
]
