"""Resident ZAFF nonbonded and long-range electrostatic engines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from matrix_chem.continuous_nonbonded import continuous_topology_pair_corrections
from matrix_chem.nonbonded import (
    NonbondedHessianComponents,
    continuous_electrostatic_pair_components,
    nonbonded_cartesian_hessian_components as _exact_nonbonded_components,
    nonbonded_cartesian_energy as architect_nonbonded_cartesian_energy,
)
from matrix_chem.spatial_regions import SpatialRegions, bounded_topological_distances
from matrix_chem.topology.covalent_radii import covalent_radius
from .native_kernels import (
    direct_gaussian_energy as _native_direct_gaussian_energy,
    direct_gaussian_energy_gradient as _native_direct_gaussian_energy_gradient,
    direct_gaussian_hessian_vector as _native_direct_gaussian_hessian_vector,
    gaussian_correction_energy as _native_gaussian_correction_energy,
    gaussian_correction_energy_gradient as _native_gaussian_correction_energy_gradient,
    gaussian_correction_hessian_vector as _native_gaussian_correction_hessian_vector,
    native_zaff_backend,
)


FOUR_PI = 4.0 * math.pi
BOHR_TO_ANGSTROM = 0.52917721092


@dataclass(frozen=True)
class ElectrostaticEnergyGradient:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    backend: str
    requested_precision: float
    pair_correction_count: int


@dataclass(frozen=True)
class LaplaceTargetEvaluation:
    """Potential and Cartesian gradient generated at arbitrary target points."""

    potential: np.ndarray
    gradient: np.ndarray
    backend: str
    requested_precision: float


@dataclass(frozen=True)
class MMRuntimePolicy:
    atom_count: int
    short_range_backend: str
    electrostatic_backend: str
    neighbor_pair_count: int
    cutoff_angstrom: float
    fmm_precision: float
    linear_scaling_required: bool = False
    linear_scaling_certified: bool = False

    def to_dict(self) -> dict[str, bool | float | int | str]:
        return {
            "atom_count": self.atom_count,
            "short_range_backend": self.short_range_backend,
            "electrostatic_backend": self.electrostatic_backend,
            "neighbor_pair_count": self.neighbor_pair_count,
            "cutoff_angstrom": self.cutoff_angstrom,
            "fmm_precision": self.fmm_precision,
            "linear_scaling_required": self.linear_scaling_required,
            "linear_scaling_certified": self.linear_scaling_certified,
        }


def select_mm_runtime_policy(
    coordinates_angstrom: np.ndarray,
    *,
    cutoff_angstrom: float = 12.0,
    neighbor_list_minimum_atoms: int = 64,
    fmm_minimum_atoms: int = 256,
    fmm_precision: float = 1.0e-10,
    materialize_neighbor_pairs: bool = True,
    require_linear_scaling: bool = False,
) -> MMRuntimePolicy:
    """Select the shared large-system MM spatial policy.

    Persistent runtimes may defer pair materialization to their resident
    Verlet list; stateless callers retain the historical exact pair count.
    """

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("MM runtime coordinates must have shape (natoms, 3)")
    count = len(coordinates)
    use_neighbors = count >= int(neighbor_list_minimum_atoms)
    pairs = (
        build_nonbonded_neighbor_list(
            coordinates / BOHR_TO_ANGSTROM,
            cutoff_bohr=float(cutoff_angstrom) / BOHR_TO_ANGSTROM,
        )
        if use_neighbors and materialize_neighbor_pairs
        else ()
    )
    large_system = count >= int(fmm_minimum_atoms)
    fmm_present = _fmm_available()
    if require_linear_scaling and large_system and not fmm_present:
        raise RuntimeError(
            "linear-scaling electrostatics requires matrix-zaff[fmm]; "
            "refusing the quadratic direct fallback"
        )
    electrostatic = "fmm" if large_system and fmm_present else "direct"
    return MMRuntimePolicy(
        atom_count=count,
        short_range_backend="spatial_neighbor_list" if use_neighbors else "direct_pairs",
        electrostatic_backend=electrostatic,
        neighbor_pair_count=(
            len(pairs) if materialize_neighbor_pairs or not use_neighbors else -1
        ),
        cutoff_angstrom=float(cutoff_angstrom),
        fmm_precision=float(fmm_precision),
        linear_scaling_required=bool(require_linear_scaling),
        linear_scaling_certified=bool(
            require_linear_scaling and (not large_system or electrostatic == "fmm")
        ),
    )


@dataclass(frozen=True)
class _RadialDerivatives:
    energy: float
    first: float
    second: float


def zaff_nonbonded_cartesian_hessian_components(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray | tuple[int, ...],
    topology_bonds: tuple[tuple[int, int], ...],
    **kwargs,
) -> NonbondedHessianComponents:
    """Return the exact analytic E/G/H reference used during compilation.

    This compatibility boundary makes ARCHITECT the owner of the force-field
    partition while retaining the validated low-level radial derivative
    kernel shared with older files.
    """

    return _exact_nonbonded_components(
        coordinates_bohr,
        atomic_numbers,
        topology_bonds,
        **kwargs,
    )


def electrostatic_energy_gradient(
    coordinates_bohr: np.ndarray,
    charges: np.ndarray,
    topology_bonds: tuple[tuple[int, int], ...] = (),
    *,
    atomic_numbers: np.ndarray | tuple[int, ...] | None = None,
    electrostatic_model: Literal[
        "gaussian_erf_all_pairs", "topology_scaled_point_charges"
    ] = "gaussian_erf_all_pairs",
    electrostatic_gaussian_width_scale: float = 1.0,
    electrostatic_gaussian_widths_bohr: np.ndarray | None = None,
    one_four_scale: float = 0.5,
    topology_scaling: Literal["continuous_erf_paths", "discrete_graph"] = "discrete_graph",
    topology_switch_alpha_per_angstrom: float = 8.0,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_atoms: int = 256,
) -> ElectrostaticEnergyGradient:
    """Evaluate all-pair electrostatics with an optional Laplace FMM.

    The default model represents every atom by a normalized Gaussian charge.
    No 1--2, 1--3 or 1--4 exclusions are applied. The FMM evaluates the point
    Coulomb limit and a bounded local ``-erfc(beta*r)/r`` correction restores
    charge penetration with analytic derivatives.
    """

    xyz = np.asarray(coordinates_bohr, dtype=float)
    q = np.asarray(charges, dtype=float).reshape(-1)
    if xyz.shape != (len(q), 3) or np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(q)):
        raise ValueError("electrostatic coordinates/charges are incomplete or non-finite")
    if backend not in {"auto", "direct", "fmm"}:
        raise ValueError("electrostatic backend must be auto, direct or fmm")
    precision = float(fmm_precision)
    if not np.isfinite(precision) or not 0.0 < precision < 1.0:
        raise ValueError("FMM precision must lie strictly between zero and one")
    selected = backend
    if selected == "auto":
        selected = "fmm" if len(q) >= int(fmm_minimum_atoms) and _fmm_available() else "direct"
    if selected == "direct" and electrostatic_model == "gaussian_erf_all_pairs":
        widths = _gaussian_widths(
            atomic_numbers,
            len(q),
            scale=electrostatic_gaussian_width_scale,
            explicit=electrostatic_gaussian_widths_bohr,
        )
        numerical = native_zaff_backend(len(q))
        if numerical.accelerated:
            energy, gradient, correction_count = (
                _native_direct_gaussian_energy_gradient(xyz, q, widths)
            )
            return ElectrostaticEnergyGradient(
                energy_hartree=energy,
                gradient_hartree_per_bohr=gradient.reshape(-1),
                backend=(
                    "DIRECT_NATIVE_LAPLACE_WITH_GAUSSIAN_ERF_ALL_PAIRS"
                ),
                requested_precision=precision,
                pair_correction_count=correction_count,
            )
    if selected == "fmm":
        energy, gradient = _fmm_coulomb(xyz, q, precision)
    else:
        energy, gradient = _direct_coulomb(xyz, q)

    if electrostatic_model == "gaussian_erf_all_pairs":
        widths = _gaussian_widths(
            atomic_numbers,
            len(q),
            scale=electrostatic_gaussian_width_scale,
            explicit=electrostatic_gaussian_widths_bohr,
        )
        numerical = native_zaff_backend(len(q))
        if numerical.accelerated:
            pairs = np.asarray(
                build_nonbonded_neighbor_list(
                    xyz,
                    cutoff_bohr=16.0 * float(np.max(widths)),
                ),
                dtype=np.intp,
            ).reshape(-1, 2)
            correction_energy, correction_gradient, correction_count = (
                _native_gaussian_correction_energy_gradient(
                    xyz, q, widths, pairs
                )
            )
            gradient += correction_gradient
        else:
            correction_energy, correction_count = (
                _apply_gaussian_penetration_gradient(
                    xyz,
                    q,
                    widths,
                    gradient,
                )
            )
        energy += correction_energy
        correction_label = "GAUSSIAN_ERF_ALL_PAIRS"
    elif electrostatic_model != "topology_scaled_point_charges":
        raise ValueError(
            "electrostatic model must be gaussian_erf_all_pairs or "
            "topology_scaled_point_charges"
        )
    elif topology_scaling == "continuous_erf_paths":
        numbers = _validated_atomic_numbers(atomic_numbers, len(q))
        correction_energy, correction_gradient, _product, correction_count = (
            continuous_topology_pair_corrections(
                xyz,
                numbers,
                lambda i, j, distance: _coulomb_radial(
                    distance,
                    q[i] * q[j],
                ),
                one_four_scale=float(one_four_scale),
                switch_alpha_per_angstrom=float(topology_switch_alpha_per_angstrom),
            )
        )
        energy += correction_energy
        gradient += correction_gradient.reshape(len(q), 3)
    elif topology_scaling == "discrete_graph":
        correction_count = _apply_discrete_energy_gradient_corrections(
            xyz,
            q,
            topology_bonds,
            float(one_four_scale),
            gradient,
        )
        energy += _discrete_energy_correction(
            xyz,
            q,
            topology_bonds,
            float(one_four_scale),
        )
    else:
        raise ValueError(
            "electrostatic topology scaling must be continuous_erf_paths or discrete_graph"
        )
    if electrostatic_model == "topology_scaled_point_charges":
        correction_label = topology_scaling.upper()
    return ElectrostaticEnergyGradient(
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=np.asarray(gradient, dtype=float).reshape(-1),
        backend=f"{selected.upper()}_LAPLACE_WITH_{correction_label}",
        requested_precision=precision,
        pair_correction_count=correction_count,
    )


def electrostatic_hessian_vector_product(
    coordinates_bohr: np.ndarray,
    charges: np.ndarray,
    vector: np.ndarray,
    topology_bonds: tuple[tuple[int, int], ...] = (),
    *,
    atomic_numbers: np.ndarray | tuple[int, ...] | None = None,
    electrostatic_model: Literal[
        "gaussian_erf_all_pairs", "topology_scaled_point_charges"
    ] = "gaussian_erf_all_pairs",
    electrostatic_gaussian_width_scale: float = 1.0,
    electrostatic_gaussian_widths_bohr: np.ndarray | None = None,
    one_four_scale: float = 0.5,
    topology_scaling: Literal["continuous_erf_paths", "discrete_graph"] = "discrete_graph",
    topology_switch_alpha_per_angstrom: float = 8.0,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_atoms: int = 256,
) -> np.ndarray:
    """Apply the electrostatic Cartesian Hessian without materializing it.

    For the FMM path, charge sources provide the target potential Hessians,
    while dipoles ``p_i=q_i v_i`` provide the source-displacement response.
    Their sum is the Coulomb Hessian action to the requested FMM tolerance.
    Sparse local ``-erfc(beta*r)/r`` blocks complete the default all-pair
    Gaussian-charge model without bonded exclusions.
    """

    xyz = np.asarray(coordinates_bohr, dtype=float)
    q = np.asarray(charges, dtype=float).reshape(-1)
    direction = np.asarray(vector, dtype=float)
    if direction.size == 3 * len(q):
        direction = direction.reshape(len(q), 3)
    if xyz.shape != (len(q), 3) or direction.shape != xyz.shape:
        raise ValueError("electrostatic Hessian-vector dimensions are inconsistent")
    selected = backend
    if selected == "auto":
        selected = "fmm" if len(q) >= int(fmm_minimum_atoms) and _fmm_available() else "direct"
    if selected == "direct" and electrostatic_model == "gaussian_erf_all_pairs":
        widths = _gaussian_widths(
            atomic_numbers,
            len(q),
            scale=electrostatic_gaussian_width_scale,
            explicit=electrostatic_gaussian_widths_bohr,
        )
        numerical = native_zaff_backend(len(q))
        if numerical.accelerated:
            return _native_direct_gaussian_hessian_vector(
                xyz, q, widths, direction
            ).reshape(-1)
    if selected == "fmm":
        product = _fmm_coulomb_hessian_vector(xyz, q, direction, float(fmm_precision))
    elif selected == "direct":
        product = _direct_coulomb_hessian_vector(xyz, q, direction)
    else:
        raise ValueError("electrostatic backend must be auto, direct or fmm")

    if electrostatic_model == "gaussian_erf_all_pairs":
        widths = _gaussian_widths(
            atomic_numbers,
            len(q),
            scale=electrostatic_gaussian_width_scale,
            explicit=electrostatic_gaussian_widths_bohr,
        )
        numerical = native_zaff_backend(len(q))
        if numerical.accelerated:
            pairs = np.asarray(
                build_nonbonded_neighbor_list(
                    xyz,
                    cutoff_bohr=16.0 * float(np.max(widths)),
                ),
                dtype=np.intp,
            ).reshape(-1, 2)
            correction_product, _count = (
                _native_gaussian_correction_hessian_vector(
                    xyz, q, widths, pairs, direction
                )
            )
            product += correction_product
        else:
            _apply_gaussian_penetration_hessian_vector(
                xyz,
                q,
                widths,
                direction,
                product,
            )
    elif electrostatic_model != "topology_scaled_point_charges":
        raise ValueError(
            "electrostatic model must be gaussian_erf_all_pairs or "
            "topology_scaled_point_charges"
        )
    elif topology_scaling == "continuous_erf_paths":
        numbers = _validated_atomic_numbers(atomic_numbers, len(q))
        _energy, _gradient, correction_product, _count = (
            continuous_topology_pair_corrections(
                xyz,
                numbers,
                lambda i, j, distance: _coulomb_radial(
                    distance,
                    q[i] * q[j],
                ),
                one_four_scale=float(one_four_scale),
                switch_alpha_per_angstrom=float(topology_switch_alpha_per_angstrom),
                vector=direction,
            )
        )
        if correction_product is not None:
            product += correction_product.reshape(len(q), 3)
    elif topology_scaling == "discrete_graph":
        _apply_discrete_hessian_vector_corrections(
            xyz,
            q,
            direction,
            topology_bonds,
            float(one_four_scale),
            product,
        )
    else:
        raise ValueError(
            "electrostatic topology scaling must be continuous_erf_paths or discrete_graph"
        )
    return np.asarray(product, dtype=float).reshape(-1)


def build_nonbonded_neighbor_list(
    coordinates_bohr: np.ndarray,
    *,
    cutoff_bohr: float,
) -> tuple[tuple[int, int], ...]:
    """Build a deterministic bounded-density pair list by spatial regions."""

    xyz = np.asarray(coordinates_bohr, dtype=float)
    cutoff = float(cutoff_bohr)
    regions = SpatialRegions.build(xyz, cell_size=cutoff)
    return tuple(regions.candidate_pairs(cutoff))


class PersistentVerletNeighborList:
    """Deterministic Verlet list reused until half of its skin is exhausted."""

    def __init__(self, *, cutoff_bohr: float, skin_bohr: float) -> None:
        if (
            not np.isfinite(cutoff_bohr)
            or not np.isfinite(skin_bohr)
            or float(cutoff_bohr) <= 0.0
            or float(skin_bohr) <= 0.0
        ):
            raise ValueError("Verlet cutoff and skin must be finite and positive")
        self.cutoff_bohr = float(cutoff_bohr)
        self.skin_bohr = float(skin_bohr)
        self.reference_coordinates_bohr: np.ndarray | None = None
        self.candidate_pairs: tuple[tuple[int, int], ...] = ()
        self.candidate_pair_array = np.empty((0, 2), dtype=np.intp)
        self.rebuild_count = 0
        self.reuse_count = 0

    def _needs_rebuild(self, coordinates: np.ndarray) -> bool:
        if (
            self.reference_coordinates_bohr is None
            or coordinates.shape != self.reference_coordinates_bohr.shape
        ):
            return True
        displacement = np.linalg.norm(
            coordinates - self.reference_coordinates_bohr,
            axis=1,
        )
        return float(np.max(displacement, initial=0.0)) > 0.5 * self.skin_bohr

    def maintain(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        """Refresh persistent candidates without filtering the exact cutoff."""

        coordinates = np.asarray(coordinates_bohr, dtype=float)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError("Verlet-list coordinates must have shape (atoms, 3)")
        if self._needs_rebuild(coordinates):
            self.candidate_pairs = build_nonbonded_neighbor_list(
                coordinates,
                cutoff_bohr=self.cutoff_bohr + self.skin_bohr,
            )
            self.candidate_pair_array = np.asarray(
                self.candidate_pairs, dtype=np.intp
            ).reshape(-1, 2)
            self.reference_coordinates_bohr = coordinates.copy()
            self.rebuild_count += 1
        else:
            self.reuse_count += 1
        return coordinates

    def native_candidates(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        """Return unfiltered Verlet candidates for a compiled radial kernel."""

        self.maintain(coordinates_bohr)
        return self.candidate_pair_array

    def pairs(self, coordinates_bohr: np.ndarray) -> tuple[tuple[int, int], ...]:
        """Return exact-cutoff pairs, rebuilding candidates only when required."""

        coordinates = self.maintain(coordinates_bohr)
        cutoff_squared = self.cutoff_bohr**2
        return tuple(
            (left, right)
            for left, right in self.candidate_pairs
            if float(
                np.dot(
                    coordinates[left] - coordinates[right],
                    coordinates[left] - coordinates[right],
                )
            )
            <= cutoff_squared
        )


class PersistentGaussianElectrostaticOperator:
    """Repeated Gaussian-charge E+G operator with FMM policy and a Verlet correction list.

    Moving FMM sources necessarily require fresh multipole coefficients at
    every MD force evaluation.  The operator, charge data, backend policy and
    bounded penetration candidate list persist across those evaluations.
    """

    def __init__(
        self,
        charges: np.ndarray,
        atomic_numbers: np.ndarray | tuple[int, ...],
        *,
        backend: Literal["auto", "direct", "fmm"] = "auto",
        fmm_precision: float = 1.0e-10,
        fmm_minimum_atoms: int = 256,
        width_scale: float = 1.0,
        neighbor_skin_bohr: float = 2.0,
    ) -> None:
        self.charges = np.asarray(charges, dtype=float).reshape(-1)
        self.atomic_numbers = _validated_atomic_numbers(
            atomic_numbers,
            len(self.charges),
        )
        self.widths_bohr = _gaussian_widths(
            self.atomic_numbers,
            len(self.charges),
            scale=float(width_scale),
            explicit=None,
        )
        if backend not in {"auto", "direct", "fmm"}:
            raise ValueError("electrostatic backend must be auto, direct or fmm")
        selected = backend
        if selected == "auto":
            selected = (
                "fmm"
                if len(self.charges) >= int(fmm_minimum_atoms) and _fmm_available()
                else "direct"
            )
        if selected == "fmm" and not _fmm_available():
            raise RuntimeError(
                "the FMM backend requires the optional matrix-zaff[fmm] dependency"
            )
        self.selected_backend = selected
        self.fmm_precision = float(fmm_precision)
        if not 0.0 < self.fmm_precision < 1.0:
            raise ValueError("FMM precision must lie strictly between zero and one")
        self.penetration_pairs = PersistentVerletNeighborList(
            cutoff_bohr=16.0 * float(np.max(self.widths_bohr)),
            skin_bohr=float(neighbor_skin_bohr),
        )
        self.evaluation_count = 0

    @property
    def persistent_fmm(self) -> bool:
        return self.selected_backend == "fmm"

    def evaluate(self, coordinates_bohr: np.ndarray) -> ElectrostaticEnergyGradient:
        coordinates = np.asarray(coordinates_bohr, dtype=float)
        if coordinates.shape != (len(self.charges), 3):
            raise ValueError("persistent electrostatic coordinates are inconsistent")
        numerical = native_zaff_backend(len(self.charges))
        if self.selected_backend == "direct" and numerical.accelerated:
            self.penetration_pairs.maintain(coordinates)
            energy, gradient, count = _native_direct_gaussian_energy_gradient(
                coordinates,
                self.charges,
                self.widths_bohr,
            )
            self.evaluation_count += 1
            return ElectrostaticEnergyGradient(
                energy_hartree=energy,
                gradient_hartree_per_bohr=gradient.reshape(-1),
                backend="DIRECT_NATIVE_WITH_PERSISTENT_GAUSSIAN_VERLET",
                requested_precision=self.fmm_precision,
                pair_correction_count=count,
            )
        numerical = native_zaff_backend(len(self.charges))
        if numerical.accelerated:
            candidates = self.penetration_pairs.native_candidates(coordinates)
        else:
            candidates = self.penetration_pairs.pairs(coordinates)
        if self.selected_backend == "fmm":
            energy, gradient = _fmm_coulomb(
                coordinates,
                self.charges,
                self.fmm_precision,
            )
        else:
            energy, gradient = _direct_coulomb(coordinates, self.charges)
        if numerical.accelerated:
            correction, correction_gradient, count = (
                _native_gaussian_correction_energy_gradient(
                    coordinates,
                    self.charges,
                    self.widths_bohr,
                    candidates,
                )
            )
            gradient += correction_gradient
        else:
            correction, count = _apply_gaussian_penetration_gradient(
                coordinates,
                self.charges,
                self.widths_bohr,
                gradient,
                pairs=candidates,
            )
        self.evaluation_count += 1
        return ElectrostaticEnergyGradient(
            energy_hartree=float(energy + correction),
            gradient_hartree_per_bohr=np.asarray(gradient, dtype=float).reshape(-1),
            backend=(
                f"{self.selected_backend.upper()}_LAPLACE_WITH_"
                "PERSISTENT_GAUSSIAN_VERLET"
            ),
            requested_precision=self.fmm_precision,
            pair_correction_count=count,
        )

    def energy(self, coordinates_bohr: np.ndarray) -> float:
        """Evaluate energy without allocating or evaluating Cartesian forces."""

        coordinates = np.asarray(coordinates_bohr, dtype=float)
        if coordinates.shape != (len(self.charges), 3):
            raise ValueError("persistent electrostatic coordinates are inconsistent")
        numerical = native_zaff_backend(len(self.charges))
        if self.selected_backend == "direct" and numerical.accelerated:
            self.penetration_pairs.maintain(coordinates)
            energy, _count = _native_direct_gaussian_energy(
                coordinates,
                self.charges,
                self.widths_bohr,
            )
            self.evaluation_count += 1
            return energy
        if numerical.accelerated:
            candidates = self.penetration_pairs.native_candidates(coordinates)
        else:
            candidates = self.penetration_pairs.pairs(coordinates)
        if self.selected_backend == "fmm":
            energy = _fmm_coulomb_energy(
                coordinates,
                self.charges,
                self.fmm_precision,
            )
        else:
            energy = _direct_coulomb_energy(coordinates, self.charges)
        if numerical.accelerated:
            correction, _count = _native_gaussian_correction_energy(
                coordinates,
                self.charges,
                self.widths_bohr,
                candidates,
            )
        else:
            correction, _count = _gaussian_penetration_energy(
                coordinates,
                self.charges,
                self.widths_bohr,
                pairs=candidates,
            )
        self.evaluation_count += 1
        return float(energy + correction)

    def hessian_vector_product(
        self,
        coordinates_bohr: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        """Apply the Gaussian electrostatic Hessian with resident pair state."""

        coordinates = np.asarray(coordinates_bohr, dtype=float)
        direction = np.asarray(vector_bohr, dtype=float)
        if direction.size == coordinates.size:
            direction = direction.reshape(coordinates.shape)
        if (
            coordinates.shape != (len(self.charges), 3)
            or direction.shape != coordinates.shape
        ):
            raise ValueError(
                "persistent electrostatic Hessian-vector dimensions are inconsistent"
            )
        numerical = native_zaff_backend(len(self.charges))
        if self.selected_backend == "direct" and numerical.accelerated:
            self.penetration_pairs.maintain(coordinates)
            product = _native_direct_gaussian_hessian_vector(
                coordinates,
                self.charges,
                self.widths_bohr,
                direction,
            )
            self.evaluation_count += 1
            return np.asarray(product, dtype=float).reshape(-1)
        candidates = (
            self.penetration_pairs.native_candidates(coordinates)
            if numerical.accelerated
            else self.penetration_pairs.pairs(coordinates)
        )
        if self.selected_backend == "fmm":
            product = _fmm_coulomb_hessian_vector(
                coordinates,
                self.charges,
                direction,
                self.fmm_precision,
            )
        else:
            product = _direct_coulomb_hessian_vector(
                coordinates,
                self.charges,
                direction,
            )
        if numerical.accelerated:
            correction, _count = _native_gaussian_correction_hessian_vector(
                coordinates,
                self.charges,
                self.widths_bohr,
                candidates,
                direction,
            )
            product += correction
        else:
            _apply_gaussian_penetration_hessian_vector(
                coordinates,
                self.charges,
                self.widths_bohr,
                direction,
                product,
                pairs=candidates,
            )
        self.evaluation_count += 1
        return np.asarray(product, dtype=float).reshape(-1)


class PersistentPointChargeElectrostaticOperator:
    """Repeated all-particle point-charge E+G with a fixed FMM policy."""

    def __init__(
        self,
        charges: np.ndarray,
        *,
        backend: Literal["auto", "direct", "fmm"] = "auto",
        fmm_precision: float = 1.0e-10,
        fmm_minimum_atoms: int = 256,
    ) -> None:
        self.charges = np.asarray(charges, dtype=float).reshape(-1)
        if np.any(~np.isfinite(self.charges)):
            raise ValueError("point-charge operator charges must be finite")
        if backend not in {"auto", "direct", "fmm"}:
            raise ValueError("electrostatic backend must be auto, direct or fmm")
        selected = backend
        if selected == "auto":
            selected = (
                "fmm"
                if len(self.charges) >= int(fmm_minimum_atoms) and _fmm_available()
                else "direct"
            )
        if selected == "fmm" and not _fmm_available():
            raise RuntimeError(
                "the FMM backend requires the optional matrix-zaff[fmm] dependency"
            )
        self.selected_backend = selected
        self.fmm_precision = float(fmm_precision)
        if not 0.0 < self.fmm_precision < 1.0:
            raise ValueError("FMM precision must lie strictly between zero and one")
        self.evaluation_count = 0

    @property
    def persistent_fmm(self) -> bool:
        return self.selected_backend == "fmm"

    def evaluate(self, coordinates_bohr: np.ndarray) -> ElectrostaticEnergyGradient:
        coordinates = np.asarray(coordinates_bohr, dtype=float)
        if (
            coordinates.shape != (len(self.charges), 3)
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError("point-charge operator coordinates are inconsistent")
        if self.selected_backend == "fmm":
            energy, gradient = _fmm_coulomb(
                coordinates,
                self.charges,
                self.fmm_precision,
            )
        else:
            energy, gradient = _direct_coulomb(coordinates, self.charges)
        self.evaluation_count += 1
        return ElectrostaticEnergyGradient(
            energy_hartree=float(energy),
            gradient_hartree_per_bohr=np.asarray(gradient, dtype=float).reshape(-1),
            backend=f"{self.selected_backend.upper()}_PERSISTENT_POINT_CHARGE",
            requested_precision=self.fmm_precision,
            pair_correction_count=0,
        )


def gaussian_electrostatic_pair_components(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray | tuple[int, ...],
    *,
    width_scale: float = 1.0,
) -> dict[tuple[int, int], tuple[float, np.ndarray, np.ndarray]]:
    """Return exact unit-charge E/G/H pair blocks for Gaussian electrostatics."""

    xyz = np.asarray(coordinates_bohr, dtype=float)
    numbers = _validated_atomic_numbers(atomic_numbers, len(xyz))
    widths = _gaussian_widths(numbers, len(xyz), scale=width_scale, explicit=None)
    size = 3 * len(xyz)
    components: dict[tuple[int, int], tuple[float, np.ndarray, np.ndarray]] = {}
    for left in range(len(xyz)):
        for right in range(left + 1, len(xyz)):
            delta = xyz[left] - xyz[right]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-12:
                raise ValueError("coincident atoms in electrostatic pair")
            radial = _gaussian_radial(
                distance,
                1.0,
                widths[left],
                widths[right],
            )
            gradient = np.zeros(size, dtype=float)
            hessian = np.zeros((size, size), dtype=float)
            _accumulate_radial_pair(
                gradient,
                hessian,
                left,
                right,
                delta,
                radial,
            )
            components[(left, right)] = (radial.energy, gradient, hessian)
    return components


def gaussian_cross_interaction_energies(
    left_coordinates_bohr: np.ndarray,
    left_charges: np.ndarray,
    left_widths_bohr: np.ndarray,
    right_coordinates_bohr: np.ndarray,
    right_charges: np.ndarray,
    right_widths_bohr: np.ndarray,
) -> np.ndarray:
    """Return exact Gaussian-erf cross energies for batched left geometries."""

    left = np.asarray(left_coordinates_bohr, dtype=float)
    right = np.asarray(right_coordinates_bohr, dtype=float)
    left_q = np.asarray(left_charges, dtype=float)
    right_q = np.asarray(right_charges, dtype=float)
    left_width = np.asarray(left_widths_bohr, dtype=float)
    right_width = np.asarray(right_widths_bohr, dtype=float)
    if left.ndim == 2:
        left = left[None, :, :]
    if (
        left.ndim != 3
        or left.shape[2] != 3
        or right.ndim != 2
        or right.shape[1] != 3
        or left_q.shape != (left.shape[1],)
        or left_width.shape != left_q.shape
        or right_q.shape != (len(right),)
        or right_width.shape != right_q.shape
        or np.any(left_width <= 0.0)
        or np.any(right_width <= 0.0)
    ):
        raise ValueError("Gaussian cross-interaction dimensions are inconsistent")
    delta = left[:, :, None, :] - right[None, None, :, :]
    distance = np.linalg.norm(delta, axis=3)
    if np.any(distance <= 1.0e-12):
        raise ValueError("coincident atoms in Gaussian cross interaction")
    beta = 1.0 / np.sqrt(
        2.0
        * (
            left_width[:, None] ** 2
            + right_width[None, :] ** 2
        )
    )
    try:
        from scipy.special import erf as vector_erf
    except ImportError:  # pragma: no cover
        vector_erf = np.vectorize(math.erf, otypes=[float])
    pair_energy = (
        left_q[None, :, None]
        * right_q[None, None, :]
        * vector_erf(beta[None, :, :] * distance)
        / distance
    )
    return np.sum(pair_energy, axis=(1, 2))


def _accumulate_radial_pair(
    gradient: np.ndarray,
    hessian: np.ndarray,
    left: int,
    right: int,
    delta: np.ndarray,
    radial: _RadialDerivatives,
) -> None:
    distance = float(np.linalg.norm(delta))
    unit = delta / distance
    block = (radial.second - radial.first / distance) * np.outer(unit, unit)
    block += (radial.first / distance) * np.eye(3)
    left_slice = slice(3 * left, 3 * left + 3)
    right_slice = slice(3 * right, 3 * right + 3)
    gradient[left_slice] += radial.first * unit
    gradient[right_slice] -= radial.first * unit
    hessian[left_slice, left_slice] += block
    hessian[right_slice, right_slice] += block
    hessian[left_slice, right_slice] -= block
    hessian[right_slice, left_slice] -= block


def _validated_atomic_numbers(
    atomic_numbers: np.ndarray | tuple[int, ...] | None,
    natoms: int,
) -> np.ndarray:
    if atomic_numbers is None:
        raise ValueError("this electrostatic model requires one atomic number per atom")
    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    if numbers.shape != (natoms,) or np.any(numbers <= 0):
        raise ValueError("electrostatic atomic numbers are incomplete or invalid")
    return numbers


def _gaussian_widths(
    atomic_numbers: np.ndarray | tuple[int, ...] | None,
    natoms: int,
    *,
    scale: float,
    explicit: np.ndarray | None,
) -> np.ndarray:
    if explicit is not None:
        widths = np.asarray(explicit, dtype=float).reshape(-1)
        if widths.shape != (natoms,) or np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
            raise ValueError("Gaussian electrostatic widths must be finite and positive")
        return widths
    factor = float(scale)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("Gaussian electrostatic width scale must be finite and positive")
    numbers = _validated_atomic_numbers(atomic_numbers, natoms)
    return np.asarray(
        [
            factor * float(covalent_radius(int(number)) or 0.75) / BOHR_TO_ANGSTROM
            for number in numbers
        ],
        dtype=float,
    )


def _gaussian_radial(
    distance: float,
    charge_product: float,
    width_i: float,
    width_j: float,
) -> _RadialDerivatives:
    r = float(distance)
    product = float(charge_product)
    beta = 1.0 / math.sqrt(2.0 * (float(width_i) ** 2 + float(width_j) ** 2))
    argument = beta * r
    gaussian = math.exp(-(argument**2))
    error_function = math.erf(argument)
    root_pi = math.sqrt(math.pi)
    first = 2.0 * beta * gaussian / (root_pi * r) - error_function / r**2
    second = (
        -4.0 * beta**3 * gaussian / root_pi
        - 4.0 * beta * gaussian / (root_pi * r**2)
        + 2.0 * error_function / r**3
    )
    return _RadialDerivatives(
        product * error_function / r,
        product * first,
        product * second,
    )


def _coulomb_radial(distance: float, charge_product: float) -> _RadialDerivatives:
    r = float(distance)
    product = float(charge_product)
    return _RadialDerivatives(product / r, -product / r**2, 2.0 * product / r**3)


def _gaussian_correction_radial(
    distance: float,
    charge_product: float,
    width_i: float,
    width_j: float,
) -> _RadialDerivatives:
    gaussian = _gaussian_radial(distance, charge_product, width_i, width_j)
    coulomb = _coulomb_radial(distance, charge_product)
    return _RadialDerivatives(
        gaussian.energy - coulomb.energy,
        gaussian.first - coulomb.first,
        gaussian.second - coulomb.second,
    )


def _gaussian_penetration_pairs(
    coordinates: np.ndarray,
    widths: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    # beta_ij*r >= 8 makes erfc(beta_ij*r) and its first two derivatives
    # negligible. beta is smallest for the two broadest Gaussian charges.
    cutoff = 16.0 * float(np.max(widths))
    regions = SpatialRegions.build(coordinates, cell_size=cutoff)
    selected = []
    for left, right in regions.candidate_pairs(cutoff):
        distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
        beta = 1.0 / math.sqrt(
            2.0 * (float(widths[left]) ** 2 + float(widths[right]) ** 2)
        )
        if beta * distance < 8.0:
            selected.append((left, right))
    return tuple(selected)


def _apply_gaussian_penetration_gradient(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    gradient: np.ndarray,
    *,
    pairs: tuple[tuple[int, int], ...] | None = None,
) -> tuple[float, int]:
    energy = 0.0
    selected_pairs = (
        _gaussian_penetration_pairs(coordinates, widths)
        if pairs is None
        else pairs
    )
    count = 0
    for left, right in selected_pairs:
        delta = coordinates[left] - coordinates[right]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            raise ValueError("coincident atoms in electrostatic pair")
        beta = 1.0 / math.sqrt(
            2.0 * (float(widths[left]) ** 2 + float(widths[right]) ** 2)
        )
        if beta * distance >= 8.0:
            continue
        radial = _gaussian_correction_radial(
            distance,
            charges[left] * charges[right],
            widths[left],
            widths[right],
        )
        pair_gradient = radial.first * delta / distance
        energy += radial.energy
        gradient[left] += pair_gradient
        gradient[right] -= pair_gradient
        count += 1
    return energy, count


def _gaussian_penetration_energy(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    *,
    pairs: tuple[tuple[int, int], ...] | None = None,
) -> tuple[float, int]:
    selected_pairs = (
        _gaussian_penetration_pairs(coordinates, widths)
        if pairs is None
        else pairs
    )
    energy = 0.0
    count = 0
    for left, right in selected_pairs:
        distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
        if distance <= 1.0e-12:
            raise ValueError("coincident atoms in electrostatic pair")
        beta = 1.0 / math.sqrt(
            2.0 * (float(widths[left]) ** 2 + float(widths[right]) ** 2)
        )
        if beta * distance >= 8.0:
            continue
        energy += _gaussian_correction_radial(
            distance,
            charges[left] * charges[right],
            widths[left],
            widths[right],
        ).energy
        count += 1
    return float(energy), count


def _apply_gaussian_penetration_hessian_vector(
    coordinates: np.ndarray,
    charges: np.ndarray,
    widths: np.ndarray,
    direction: np.ndarray,
    product: np.ndarray,
    *,
    pairs: tuple[tuple[int, int], ...] | np.ndarray | None = None,
) -> None:
    selected_pairs = (
        _gaussian_penetration_pairs(coordinates, widths)
        if pairs is None
        else pairs
    )
    for left, right in selected_pairs:
        delta = coordinates[left] - coordinates[right]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            raise ValueError("coincident atoms in electrostatic pair")
        unit = delta / distance
        radial = _gaussian_correction_radial(
            distance,
            charges[left] * charges[right],
            widths[left],
            widths[right],
        )
        block = (radial.second - radial.first / distance) * np.outer(unit, unit)
        block += (radial.first / distance) * np.eye(3)
        contribution = block @ (direction[left] - direction[right])
        product[left] += contribution
        product[right] -= contribution


def _discrete_corrections(
    natoms: int,
    topology_bonds: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], int]:
    return bounded_topological_distances(
        _zero_based_adjacency(natoms, topology_bonds),
        maximum_distance=3,
    )


def _discrete_energy_correction(
    coordinates: np.ndarray,
    charges: np.ndarray,
    topology_bonds: tuple[tuple[int, int], ...],
    one_four_scale: float,
) -> float:
    energy = 0.0
    for (left, right), distance in sorted(
        _discrete_corrections(len(charges), topology_bonds).items()
    ):
        scale = 0.0 if distance <= 2 else float(one_four_scale)
        pair_energy, _gradient = _coulomb_pair(
            coordinates[left],
            coordinates[right],
            charges[left] * charges[right],
        )
        energy += (scale - 1.0) * pair_energy
    return energy


def _apply_discrete_energy_gradient_corrections(
    coordinates: np.ndarray,
    charges: np.ndarray,
    topology_bonds: tuple[tuple[int, int], ...],
    one_four_scale: float,
    gradient: np.ndarray,
) -> int:
    count = 0
    for (left, right), distance in sorted(
        _discrete_corrections(len(charges), topology_bonds).items()
    ):
        scale = 0.0 if distance <= 2 else float(one_four_scale)
        _energy, pair_left = _coulomb_pair(
            coordinates[left],
            coordinates[right],
            charges[left] * charges[right],
        )
        gradient[left] += (scale - 1.0) * pair_left
        gradient[right] -= (scale - 1.0) * pair_left
        count += 1
    return count


def _apply_discrete_hessian_vector_corrections(
    coordinates: np.ndarray,
    charges: np.ndarray,
    direction: np.ndarray,
    topology_bonds: tuple[tuple[int, int], ...],
    one_four_scale: float,
    product: np.ndarray,
) -> None:
    for (left, right), distance in sorted(
        _discrete_corrections(len(charges), topology_bonds).items()
    ):
        scale = 0.0 if distance <= 2 else float(one_four_scale)
        block = _coulomb_pair_hessian(
            coordinates[left],
            coordinates[right],
            charges[left] * charges[right],
        )
        contribution = (scale - 1.0) * block @ (
            direction[left] - direction[right]
        )
        product[left] += contribution
        product[right] -= contribution


def _direct_coulomb(coordinates: np.ndarray, charges: np.ndarray) -> tuple[float, np.ndarray]:
    """Vectorized finite-system point-charge E+G kernel.

    This is the general non-periodic Coulomb path used by persistent MIFUNE
    evaluators.  It intentionally has no molecular or force-field assumptions.
    """

    xyz = np.asarray(coordinates, dtype=float)
    q = np.asarray(charges, dtype=float).reshape(-1)
    if xyz.shape != (len(q), 3) or np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(q)):
        raise ValueError("direct Coulomb coordinates and charges are inconsistent")
    left, right = np.triu_indices(len(q), k=1)
    delta = xyz[left] - xyz[right]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1.0e-12):
        raise ValueError("coincident atoms in electrostatic pair")
    products = q[left] * q[right]
    energy = float(np.sum(products / distance))
    pair_gradient = (-products / distance**3)[:, None] * delta
    gradient = np.zeros_like(xyz)
    np.add.at(gradient, left, pair_gradient)
    np.add.at(gradient, right, -pair_gradient)
    return energy, gradient


def point_charge_energy_gradient(
    coordinates_bohr: np.ndarray,
    charges: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate general non-periodic point-charge energy and gradient."""

    return _direct_coulomb(coordinates_bohr, charges)


def _direct_coulomb_energy(coordinates: np.ndarray, charges: np.ndarray) -> float:
    energy = 0.0
    for left in range(len(charges)):
        for right in range(left + 1, len(charges)):
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            if distance <= 1.0e-12:
                raise ValueError("coincident atoms in electrostatic pair")
            energy += charges[left] * charges[right] / distance
    return float(energy)


def _coulomb_pair(
    left: np.ndarray,
    right: np.ndarray,
    charge_product: float,
) -> tuple[float, np.ndarray]:
    delta = np.asarray(left) - np.asarray(right)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        raise ValueError("coincident atoms in electrostatic pair")
    product = float(charge_product)
    return product / distance, -product * delta / distance**3


def _coulomb_pair_hessian(
    left: np.ndarray,
    right: np.ndarray,
    charge_product: float,
) -> np.ndarray:
    delta = np.asarray(left) - np.asarray(right)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        raise ValueError("coincident atoms in electrostatic pair")
    return float(charge_product) * (
        3.0 * np.outer(delta, delta) / distance**5 - np.eye(3) / distance**3
    )


def _direct_coulomb_hessian_vector(
    coordinates: np.ndarray,
    charges: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    product = np.zeros_like(coordinates)
    for left in range(len(charges)):
        for right in range(left + 1, len(charges)):
            block = _coulomb_pair_hessian(
                coordinates[left],
                coordinates[right],
                charges[left] * charges[right],
            )
            contribution = block @ (direction[left] - direction[right])
            product[left] += contribution
            product[right] -= contribution
    return product


def _fmm_available() -> bool:
    try:
        import fmm3dpy  # noqa: F401
    except ImportError:
        return False
    return True


def fmm_available() -> bool:
    """Return whether the linear-scaling Laplace backend is importable."""

    return _fmm_available()


def laplace_potential_gradient_targets(
    sources_bohr: np.ndarray,
    charges: np.ndarray,
    targets_bohr: np.ndarray,
    *,
    backend: Literal["auto", "direct", "fmm"] = "auto",
    fmm_precision: float = 1.0e-10,
    fmm_minimum_sources: int = 256,
) -> LaplaceTargetEvaluation:
    """Evaluate ``sum_j q_j / |target-source_j|`` and its target gradient.

    Unlike the particle-particle electrostatic entry points, this operation
    keeps sources and targets distinct.  It is the shared primitive required
    by CPCM boundary projection and by matrix-free charge-response operators.
    """

    sources = np.asarray(sources_bohr, dtype=float)
    targets = np.asarray(targets_bohr, dtype=float)
    q = np.asarray(charges, dtype=float).reshape(-1)
    if (
        sources.shape != (len(q), 3)
        or targets.ndim != 2
        or targets.shape[1:] != (3,)
        or np.any(~np.isfinite(sources))
        or np.any(~np.isfinite(targets))
        or np.any(~np.isfinite(q))
    ):
        raise ValueError("Laplace sources, charges, and targets are inconsistent")
    if backend not in {"auto", "direct", "fmm"}:
        raise ValueError("Laplace target backend must be auto, direct, or fmm")
    selected = backend
    if selected == "auto":
        selected = (
            "fmm"
            if len(sources) >= int(fmm_minimum_sources) and _fmm_available()
            else "direct"
        )
    if selected == "fmm":
        try:
            from fmm3dpy import lfmm3d
        except ImportError as exc:
            raise RuntimeError(
                "the FMM backend requires the optional matrix-zaff[fmm] dependency"
            ) from exc
        result = lfmm3d(
            eps=float(fmm_precision),
            sources=np.asfortranarray(sources.T),
            charges=np.asfortranarray(q),
            targets=np.asfortranarray(targets.T),
            pgt=2,
        )
        potential = FOUR_PI * np.asarray(result.pottarg, dtype=float).reshape(-1)
        gradient = (
            FOUR_PI
            * np.asarray(result.gradtarg, dtype=float).reshape(3, -1).T
        )
    else:
        delta = targets[:, None, :] - sources[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        if np.any(distance <= 1.0e-12):
            raise ValueError("coincident Laplace source and target")
        potential = (1.0 / distance) @ q
        gradient = -np.einsum("ts,tsd,s->td", distance**-3, delta, q)
    return LaplaceTargetEvaluation(
        potential=np.asarray(potential, dtype=float),
        gradient=np.asarray(gradient, dtype=float),
        backend=str(selected),
        requested_precision=float(fmm_precision),
    )


def _fmm_coulomb(
    coordinates: np.ndarray,
    charges: np.ndarray,
    precision: float,
) -> tuple[float, np.ndarray]:
    try:
        from fmm3dpy import lfmm3d
    except ImportError as exc:
        raise RuntimeError(
            "the FMM backend requires the optional matrix-zaff[fmm] dependency"
        ) from exc
    result = lfmm3d(
        eps=float(precision),
        sources=np.asfortranarray(coordinates.T),
        charges=np.asfortranarray(charges),
        pg=2,
    )
    potential = FOUR_PI * np.asarray(result.pot, dtype=float).reshape(-1)
    field_gradient = FOUR_PI * np.asarray(result.grad, dtype=float).reshape(3, -1).T
    energy = 0.5 * float(np.dot(charges, potential))
    gradient = charges[:, None] * field_gradient
    return energy, gradient


def _fmm_coulomb_energy(
    coordinates: np.ndarray,
    charges: np.ndarray,
    precision: float,
) -> float:
    try:
        from fmm3dpy import lfmm3d
    except ImportError as exc:
        raise RuntimeError(
            "the FMM backend requires the optional matrix-zaff[fmm] dependency"
        ) from exc
    result = lfmm3d(
        eps=float(precision),
        sources=np.asfortranarray(coordinates.T),
        charges=np.asfortranarray(charges),
        pg=1,
    )
    potential = FOUR_PI * np.asarray(result.pot, dtype=float).reshape(-1)
    return 0.5 * float(np.dot(charges, potential))


def _fmm_coulomb_hessian_vector(
    coordinates: np.ndarray,
    charges: np.ndarray,
    direction: np.ndarray,
    precision: float,
) -> np.ndarray:
    try:
        from fmm3dpy import lfmm3d
    except ImportError as exc:
        raise RuntimeError(
            "the FMM backend requires the optional matrix-zaff[fmm] dependency"
        ) from exc
    sources = np.asfortranarray(coordinates.T)
    charge_result = lfmm3d(
        eps=float(precision),
        sources=sources,
        charges=np.asfortranarray(charges),
        pg=3,
    )
    dipole_result = lfmm3d(
        eps=float(precision),
        sources=sources,
        dipvec=np.asfortranarray((charges[:, None] * direction).T),
        pg=2,
    )
    packed = FOUR_PI * np.asarray(charge_result.hess, dtype=float).reshape(6, -1)
    target_hessian = np.empty((len(charges), 3, 3), dtype=float)
    target_hessian[:, 0, 0] = packed[0]
    target_hessian[:, 1, 1] = packed[1]
    target_hessian[:, 2, 2] = packed[2]
    target_hessian[:, 0, 1] = target_hessian[:, 1, 0] = packed[3]
    target_hessian[:, 0, 2] = target_hessian[:, 2, 0] = packed[4]
    target_hessian[:, 1, 2] = target_hessian[:, 2, 1] = packed[5]
    dipole_gradient = (
        FOUR_PI * np.asarray(dipole_result.grad, dtype=float).reshape(3, -1).T
    )
    return charges[:, None] * (
        np.einsum("nij,nj->ni", target_hessian, direction)
        + dipole_gradient
    )


def _zero_based_adjacency(
    natoms: int,
    topology_bonds: tuple[tuple[int, int], ...],
) -> list[set[int]]:
    adjacency = [set() for _ in range(natoms)]
    for raw_left, raw_right in topology_bonds:
        # Serialized topology is one based; accept an explicit zero index as a
        # zero-based in-memory contract for programmatic callers.
        if int(raw_left) == 0 or int(raw_right) == 0:
            left, right = int(raw_left), int(raw_right)
        else:
            left, right = int(raw_left) - 1, int(raw_right) - 1
        if 0 <= left < natoms and 0 <= right < natoms and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


architect_nonbonded_cartesian_hessian_components = (
    zaff_nonbonded_cartesian_hessian_components
)


__all__ = [
    "ElectrostaticEnergyGradient",
    "LaplaceTargetEvaluation",
    "MMRuntimePolicy",
    "PersistentVerletNeighborList",
    "PersistentGaussianElectrostaticOperator",
    "PersistentPointChargeElectrostaticOperator",
    "architect_nonbonded_cartesian_hessian_components",
    "architect_nonbonded_cartesian_energy",
    "zaff_nonbonded_cartesian_hessian_components",
    "build_nonbonded_neighbor_list",
    "continuous_electrostatic_pair_components",
    "electrostatic_energy_gradient",
    "point_charge_energy_gradient",
    "electrostatic_hessian_vector_product",
    "fmm_available",
    "gaussian_electrostatic_pair_components",
    "gaussian_cross_interaction_energies",
    "laplace_potential_gradient_targets",
    "select_mm_runtime_policy",
]
