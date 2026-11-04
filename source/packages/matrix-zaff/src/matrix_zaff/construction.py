"""Shared construction-time nonbonded scoring owned by ZAFF.

Construction engines supply candidate geometries and local neighbour lists;
ZAFF remains the single owner of the Gaussian-penetration electrostatics and
the undamped Exp-PE short-range Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os

import numpy as np

from .nonbonded import (
    BOHR_TO_ANGSTROM,
    laplace_potential_gradient_targets,
    select_mm_runtime_policy,
)


HARTREE_TO_KCAL_PER_MOL = 627.5094740631


def hydrogen_bond_radial_switch(
    distance_angstrom: float | np.ndarray,
    on_angstrom: float,
    off_angstrom: float,
) -> float | np.ndarray:
    """C2 radial cutoff shared by resident directional H-bond evaluators."""

    distance = np.asarray(distance_angstrom, dtype=float)
    if (
        not math.isfinite(on_angstrom)
        or not math.isfinite(off_angstrom)
        or on_angstrom <= 0.0
        or off_angstrom <= on_angstrom
        or np.any(~np.isfinite(distance))
    ):
        raise ValueError("invalid hydrogen-bond radial switch")
    t = np.clip((distance - on_angstrom) / (off_angstrom - on_angstrom), 0.0, 1.0)
    value = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
    return float(value) if value.ndim == 0 else value


@dataclass(frozen=True)
class ConstructionInteractionBatch:
    """Per-candidate interaction components in kcal mol-1."""

    electrostatic_kcal_per_mol: np.ndarray
    short_range_kcal_per_mol: np.ndarray
    directional_kcal_per_mol: np.ndarray
    execution: dict[str, object]

    @property
    def total_kcal_per_mol(self) -> np.ndarray:
        return (
            self.electrostatic_kcal_per_mol
            + self.short_range_kcal_per_mol
            + self.directional_kcal_per_mol
        )


@dataclass(frozen=True)
class ResolvedExpPEPairTable:
    """Pair-specific Exp-PE parameters compiled from an external source model."""

    epsilon_kcal_per_mol: np.ndarray
    r_min_angstrom: np.ndarray
    alpha: np.ndarray
    source: str

    def arrays(
        self, candidate_atoms: int, environment_atoms: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (int(candidate_atoms), int(environment_atoms))
        epsilon = np.asarray(self.epsilon_kcal_per_mol, dtype=float)
        r_min = np.asarray(self.r_min_angstrom, dtype=float)
        alpha = np.asarray(self.alpha, dtype=float)
        if epsilon.shape != shape or r_min.shape != shape or alpha.shape != shape:
            raise ValueError("resolved Exp-PE pair table has inconsistent dimensions")
        if (
            np.any(~np.isfinite(epsilon))
            or np.any(~np.isfinite(r_min))
            or np.any(~np.isfinite(alpha))
            or np.any(epsilon <= 0.0)
            or np.any(r_min <= 0.0)
            or np.any(alpha <= 4.0)
            or not str(self.source).strip()
        ):
            raise ValueError("resolved Exp-PE pair table is not physical")
        return epsilon, r_min, alpha

    def lookup_plan(self) -> "ExpPELookupPlan":
        """Preindex the immutable pair-alpha matrix for repeated GA batches."""

        return ExpPELookupPlan.compile(np.asarray(self.alpha, dtype=float))


@dataclass(frozen=True)
class DirectionalExpPEInteraction:
    """One resolved anchor--contact...terminal H-bond or XB interaction."""

    kind: str
    anchor_side: str
    anchor_index: int
    contact_side: str
    contact_index: int
    terminal_side: str
    terminal_index: int
    epsilon_kcal_per_mol: float
    radial_scale_angstrom: float
    alpha: float
    angular_power: int
    charge_beta: float = 0.0
    synthon_beta: float = 0.0
    charge_product: float = 0.0
    synthon_score: float = 0.0
    switch_on_angstrom: float | None = None
    switch_off_angstrom: float | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower().replace("-", "_")
        kind = {"hbond": "hydrogen_bond", "xbond": "halogen_bond", "x_bond": "halogen_bond"}.get(kind, kind)
        sides = (
            str(self.anchor_side).strip().lower(),
            str(self.contact_side).strip().lower(),
            str(self.terminal_side).strip().lower(),
        )
        if kind not in {
            "hydrogen_bond", "alkali_bond", "alkaline_earth_bond", "regium_bond",
            "spodium_bond", "triel_bond", "tetrel_bond", "pnictogen_bond",
            "chalcogen_bond", "halogen_bond", "aerogen_bond", "cation_pi",
            "anion_pi", "lone_pair_pi", "pi_pi", "orthogonal",
        }:
            raise ValueError("unsupported directional noncovalent interaction kind")
        if any(side not in {"candidate", "environment"} for side in sides):
            raise ValueError("directional interaction sides are invalid")
        if min(self.anchor_index, self.contact_index, self.terminal_index) < 0:
            raise ValueError("directional interaction indices cannot be negative")
        if (
            not math.isfinite(float(self.epsilon_kcal_per_mol))
            or float(self.epsilon_kcal_per_mol) <= 0.0
            or not math.isfinite(float(self.radial_scale_angstrom))
            or float(self.radial_scale_angstrom) <= 0.0
            or not math.isfinite(float(self.alpha))
            or float(self.alpha) <= 4.0
            or int(self.angular_power) != self.angular_power
            or not 1 <= int(self.angular_power) <= 16
        ):
            raise ValueError("directional Exp-PE parameters are not physical")
        if any(
            not math.isfinite(float(value))
            for value in (self.charge_beta, self.synthon_beta,
                          self.charge_product, self.synthon_score)
        ):
            raise ValueError("directional CM5/synthon parameters are invalid")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "anchor_side", sides[0])
        object.__setattr__(self, "contact_side", sides[1])
        object.__setattr__(self, "terminal_side", sides[2])
        object.__setattr__(self, "angular_power", int(self.angular_power))
        if kind == "hydrogen_bond":
            on = self.radial_scale_angstrom + 1.0 if self.switch_on_angstrom is None else float(self.switch_on_angstrom)
            off = self.radial_scale_angstrom + 2.0 if self.switch_off_angstrom is None else float(self.switch_off_angstrom)
            if not math.isfinite(on) or not math.isfinite(off) or not self.radial_scale_angstrom < on < off:
                raise ValueError("invalid hydrogen-bond switch interval")
            object.__setattr__(self, "switch_on_angstrom", on)
            object.__setattr__(self, "switch_off_angstrom", off)


@dataclass(frozen=True)
class ResolvedDirectionalExpPEContacts:
    """Small immutable directional contact list evaluated beside pair lists."""

    interactions: tuple[DirectionalExpPEInteraction, ...]
    source: str

    def __post_init__(self) -> None:
        interactions = tuple(self.interactions)
        if any(not isinstance(item, DirectionalExpPEInteraction) for item in interactions):
            raise TypeError("resolved directional contacts contain an invalid item")
        if not str(self.source).strip():
            raise ValueError("resolved directional contacts need provenance")
        object.__setattr__(self, "interactions", interactions)

    def energies(
        self,
        candidates_angstrom: np.ndarray,
        environment_angstrom: np.ndarray,
        *,
        radial_evaluation: str,
    ) -> np.ndarray:
        candidates = np.asarray(candidates_angstrom, dtype=float)
        environment = np.asarray(environment_angstrom, dtype=float)
        result = np.zeros(len(candidates), dtype=float)

        def coordinates(side: str, index: int) -> np.ndarray:
            if side == "candidate":
                if index >= candidates.shape[1]:
                    raise IndexError("directional candidate index is out of range")
                return candidates[:, index, :]
            if index >= len(environment):
                raise IndexError("directional environment index is out of range")
            return np.broadcast_to(environment[index], (len(candidates), 3))

        for interaction in self.interactions:
            anchor = coordinates(interaction.anchor_side, interaction.anchor_index)
            contact = coordinates(interaction.contact_side, interaction.contact_index)
            terminal = coordinates(interaction.terminal_side, interaction.terminal_index)
            first = anchor - contact
            second = terminal - contact
            first_norm2 = np.einsum("ij,ij->i", first, first)
            second_norm2 = np.einsum("ij,ij->i", second, second)
            if np.any(first_norm2 <= 1.0e-24) or np.any(second_norm2 <= 1.0e-24):
                raise ValueError("directional interaction contains coincident sites")
            cosine = np.einsum("ij,ij->i", first, second) / np.sqrt(
                first_norm2 * second_norm2
            )
            angular = np.maximum(0.0, -np.clip(cosine, -1.0, 1.0))
            angular = np.power(angular, interaction.angular_power)
            radial = interaction.epsilon_kcal_per_mol * (
                evaluate_exppe_dimensionless(
                    np.sqrt(second_norm2) / interaction.radial_scale_angstrom,
                    interaction.alpha,
                    radial_evaluation=radial_evaluation,
                )
            )
            if interaction.kind == "hydrogen_bond":
                radial *= hydrogen_bond_radial_switch(
                    np.sqrt(second_norm2),
                    interaction.switch_on_angstrom,
                    interaction.switch_off_angstrom,
                )
            modulation = math.exp(max(-4.0, min(4.0,
                interaction.charge_beta * interaction.charge_product
                + interaction.synthon_beta * interaction.synthon_score)))
            result += radial * angular * modulation
        return result

    def paired_energies(
        self,
        candidates_angstrom: np.ndarray,
        environments_angstrom: np.ndarray,
        *,
        radial_evaluation: str,
    ) -> np.ndarray:
        """Evaluate contacts for candidate-specific rigid environments."""

        candidates = np.asarray(candidates_angstrom, dtype=float)
        environments = np.asarray(environments_angstrom, dtype=float)
        if (
            candidates.ndim != 3
            or environments.ndim != 3
            or candidates.shape[0] != environments.shape[0]
            or candidates.shape[2] != 3
            or environments.shape[2] != 3
        ):
            raise ValueError("paired directional geometries have inconsistent shapes")
        result = np.zeros(len(candidates), dtype=float)

        def coordinates(side: str, index: int) -> np.ndarray:
            source = candidates if side == "candidate" else environments
            if index >= source.shape[1]:
                raise IndexError("paired directional index is out of range")
            return source[:, index, :]

        for interaction in self.interactions:
            anchor = coordinates(interaction.anchor_side, interaction.anchor_index)
            contact = coordinates(interaction.contact_side, interaction.contact_index)
            terminal = coordinates(interaction.terminal_side, interaction.terminal_index)
            first = anchor - contact
            second = terminal - contact
            first_norm2 = np.einsum("ij,ij->i", first, first)
            second_norm2 = np.einsum("ij,ij->i", second, second)
            if np.any(first_norm2 <= 1.0e-24) or np.any(second_norm2 <= 1.0e-24):
                raise ValueError("directional interaction contains coincident sites")
            cosine = np.einsum("ij,ij->i", first, second) / np.sqrt(
                first_norm2 * second_norm2
            )
            angular = np.power(
                np.maximum(0.0, -np.clip(cosine, -1.0, 1.0)),
                interaction.angular_power,
            )
            radial = interaction.epsilon_kcal_per_mol * (
                evaluate_exppe_dimensionless(
                    np.sqrt(second_norm2) / interaction.radial_scale_angstrom,
                    interaction.alpha,
                    radial_evaluation=radial_evaluation,
                )
            )
            if interaction.kind == "hydrogen_bond":
                radial *= hydrogen_bond_radial_switch(
                    np.sqrt(second_norm2),
                    interaction.switch_on_angstrom,
                    interaction.switch_off_angstrom,
                )
            modulation = math.exp(max(-4.0, min(4.0,
                interaction.charge_beta * interaction.charge_product
                + interaction.synthon_beta * interaction.synthon_score)))
            result += radial * angular * modulation
        return result


def compile_uff_exppe_pair_table(
    candidate_atomic_numbers: np.ndarray,
    environment_atomic_numbers: np.ndarray,
) -> ResolvedExpPEPairTable:
    """Compile UFF atom pairs into the same ZAFF Exp-PE table used by GFN-FF."""

    from .radial import exppe_from_minimum, mie_minimum_derivatives
    from matrix_chem import uff_pair_parameters

    candidate = np.asarray(candidate_atomic_numbers, dtype=int).reshape(-1)
    environment = np.asarray(environment_atomic_numbers, dtype=int).reshape(-1)
    epsilon = np.empty((len(candidate), len(environment)), dtype=float)
    r_min = np.empty_like(epsilon)
    alpha = np.empty_like(epsilon)
    for left, zi in enumerate(candidate):
        for right, zj in enumerate(environment):
            pair_r_min, pair_epsilon = uff_pair_parameters(int(zi), int(zj))
            potential = exppe_from_minimum(
                mie_minimum_derivatives(
                    pair_epsilon,
                    pair_r_min,
                    12.0,
                    6.0,
                )
            )
            epsilon[left, right] = potential.epsilon * HARTREE_TO_KCAL_PER_MOL
            r_min[left, right] = potential.r_min * BOHR_TO_ANGSTROM
            alpha[left, right] = potential.alpha
    return ResolvedExpPEPairTable(
        epsilon_kcal_per_mol=epsilon,
        r_min_angstrom=r_min,
        alpha=alpha,
        source="UFF compiled to ZAFF Exp-PE",
    )


@lru_cache(maxsize=128)
def all_pair_candidate_csr(
    candidate_count: int,
    environment_atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical CSR scheduling rows without truncating physics."""

    candidates = int(candidate_count)
    environment = int(environment_atom_count)
    if candidates < 0 or environment < 0:
        raise ValueError("ZAFF CSR dimensions cannot be negative")
    indices = np.tile(np.arange(environment, dtype=np.int64), candidates)
    offsets = np.arange(
        0,
        (candidates + 1) * environment,
        environment if environment else 1,
        dtype=np.int64,
    )
    if environment == 0:
        offsets = np.zeros(candidates + 1, dtype=np.int64)
    return indices, offsets


def batch_candidate_interactions(
    candidates_angstrom: np.ndarray,
    existing_coordinates_angstrom: np.ndarray,
    existing_epsilon_kcal_per_mol: np.ndarray,
    existing_rmin_half_angstrom: np.ndarray,
    existing_charges_e: np.ndarray,
    existing_charge_widths_angstrom: np.ndarray,
    candidate_epsilon_kcal_per_mol: np.ndarray,
    candidate_rmin_half_angstrom: np.ndarray,
    candidate_charges_e: np.ndarray,
    candidate_charge_widths_angstrom: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_offsets: np.ndarray,
    *,
    cutoff_angstrom: float,
    dielectric: float = 1.0,
    electrostatic_model: str = "gaussian_erf_all_pairs",
    radial_evaluation: str = "analytic",
    electrostatic_backend: str = "auto",
    fmm_minimum_sources: int = 256,
    fmm_precision: float = 1.0e-10,
    requested_backend: str = "auto",
    gpu_min_batch: int | None = None,
    allow_mixed_precision: bool = True,
    validate_accelerator: bool = True,
    resolved_short_range: ResolvedExpPEPairTable | None = None,
    resolved_directional: ResolvedDirectionalExpPEContacts | None = None,
) -> ConstructionInteractionBatch:
    """Evaluate local candidate/environment interactions with resident ZAFF models.

    Electrostatics uses normalized Gaussian charges by default.  Rigid
    construction searches may explicitly request ``point_charge_all_pairs``
    to omit charge penetration without changing the default Hamiltonian.  The
    short-range term is the resident undamped Exp-PE potential.
    """

    candidates = np.asarray(candidates_angstrom, dtype=float)
    existing_xyz = np.asarray(existing_coordinates_angstrom, dtype=float)
    existing_epsilon = np.asarray(existing_epsilon_kcal_per_mol, dtype=float)
    existing_rmin = np.asarray(existing_rmin_half_angstrom, dtype=float)
    existing_q = np.asarray(existing_charges_e, dtype=float)
    existing_width = np.asarray(existing_charge_widths_angstrom, dtype=float)
    candidate_epsilon = np.asarray(candidate_epsilon_kcal_per_mol, dtype=float)
    candidate_rmin = np.asarray(candidate_rmin_half_angstrom, dtype=float)
    candidate_q = np.asarray(candidate_charges_e, dtype=float)
    candidate_width = np.asarray(candidate_charge_widths_angstrom, dtype=float)
    electrostatics = str(electrostatic_model).strip().lower().replace("-", "_")
    if electrostatics not in {
        "gaussian_erf_all_pairs",
        "point_charge_all_pairs",
    }:
        raise ValueError(
            "ZAFF construction electrostatic_model must be "
            "gaussian_erf_all_pairs or point_charge_all_pairs"
        )
    radial_backend = str(radial_evaluation).strip().lower().replace("-", "_")
    if radial_backend not in {"analytic", "lookup"}:
        raise ValueError(
            "ZAFF construction radial_evaluation must be analytic or lookup"
        )
    long_range_backend = str(electrostatic_backend).strip().lower()
    if long_range_backend not in {"auto", "direct", "fmm"}:
        raise ValueError(
            "ZAFF construction electrostatic_backend must be auto, direct, or fmm"
        )
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    offsets = np.asarray(neighbor_offsets, dtype=np.int64)
    if candidates.ndim != 3 or candidates.shape[2] != 3:
        raise ValueError("candidate geometries must have shape (n, natom, 3)")
    candidate_atoms = candidates.shape[1]
    resolved_pair_arrays = (
        None
        if resolved_short_range is None
        else resolved_short_range.arrays(candidate_atoms, len(existing_xyz))
    )
    if (
        existing_xyz.shape != (len(existing_q), 3)
        or existing_epsilon.shape != existing_q.shape
        or existing_rmin.shape != existing_q.shape
        or existing_width.shape != existing_q.shape
        or candidate_epsilon.shape != (candidate_atoms,)
        or candidate_rmin.shape != (candidate_atoms,)
        or candidate_q.shape != (candidate_atoms,)
        or candidate_width.shape != (candidate_atoms,)
        or offsets.shape != (len(candidates) + 1,)
    ):
        raise ValueError("ZAFF construction parameters have inconsistent dimensions")
    if (
        np.any(~np.isfinite(candidates))
        or np.any(~np.isfinite(existing_xyz))
        or np.any(existing_epsilon < 0.0)
        or np.any(candidate_epsilon < 0.0)
        or np.any(existing_rmin <= 0.0)
        or np.any(candidate_rmin <= 0.0)
        or np.any(existing_width <= 0.0)
        or np.any(candidate_width <= 0.0)
        or not math.isfinite(float(dielectric))
        or float(dielectric) <= 0.0
    ):
        raise ValueError("ZAFF construction parameters must be finite and physical")
    if len(candidates) == 0:
        empty = np.empty(0, dtype=float)
        return ConstructionInteractionBatch(
            electrostatic_kcal_per_mol=empty,
            short_range_kcal_per_mol=empty.copy(),
            directional_kcal_per_mol=empty.copy(),
            execution={
                "owner": "matrix-zaff",
                "electrostatics": electrostatics,
                "radial_evaluation": radial_backend,
                "short_range": "exppe",
                "short_range_backend": "not-needed",
            },
        )

    if (
        len(indices)
        and (int(np.min(indices)) < 0 or int(np.max(indices)) >= len(existing_xyz))
    ) or np.any(np.diff(offsets) < 0):
        raise ValueError("ZAFF construction CSR indices are invalid")
    requested = str(requested_backend).strip().lower()
    del cutoff_angstrom
    pair_work = int(len(candidates) * candidate_atoms * len(existing_xyz))
    dense_short_range = _csr_covers_all_pairs(
        indices,
        offsets,
        len(candidates),
        len(existing_xyz),
    )
    short_range_pair_work = int(candidate_atoms * len(indices))
    electrostatic_override = None
    realized_electrostatic_backend = "direct"
    if electrostatics == "point_charge_all_pairs":
        policy = select_mm_runtime_policy(
            existing_xyz,
            fmm_minimum_atoms=int(fmm_minimum_sources),
            fmm_precision=float(fmm_precision),
            materialize_neighbor_pairs=False,
        )
        use_fmm = long_range_backend == "fmm" or (
            long_range_backend == "auto"
            and policy.electrostatic_backend == "fmm"
        )
        if use_fmm:
            targets = candidates.reshape(-1, 3)
            target_field = laplace_potential_gradient_targets(
                existing_xyz / BOHR_TO_ANGSTROM,
                existing_q,
                targets / BOHR_TO_ANGSTROM,
                backend="fmm",
                fmm_precision=float(fmm_precision),
                fmm_minimum_sources=int(fmm_minimum_sources),
            )
            electrostatic_override = (
                np.sum(
                    target_field.potential.reshape(
                        len(candidates), candidate_atoms
                    )
                    * candidate_q[None, :],
                    axis=1,
                )
                * HARTREE_TO_KCAL_PER_MOL
                / float(dielectric)
            )
            realized_electrostatic_backend = target_field.backend
    try:
        threshold = int(
            gpu_min_batch
            or os.environ.get(
                "MATRIX_ZAFF_CONSTRUCTION_GPU_MIN_PAIRS", "250000"
            )
        )
    except ValueError:
        threshold = 250000
    threshold = max(1, threshold)
    selected_device = _torch_construction_device(requested)
    device = selected_device
    use_gpu = (
        radial_backend == "analytic"
        and dense_short_range
        and electrostatic_override is None
        and selected_device is not None
        and (
        requested not in {"auto", "cpu", "numpy"} or pair_work >= threshold
        )
    )
    fallback_reason = ""
    validation_error = 0.0
    if use_gpu:
        try:
            electrostatic, short_range, precision = _torch_all_pair_batch(
                candidates,
                existing_xyz,
                existing_epsilon,
                existing_rmin,
                existing_q,
                existing_width,
                candidate_epsilon,
                candidate_rmin,
                candidate_q,
                candidate_width,
                dielectric=float(dielectric),
                electrostatic_model=electrostatics,
                resolved_pair_arrays=resolved_pair_arrays,
                device=device,
                allow_mixed_precision=allow_mixed_precision,
            )
            if validate_accelerator:
                sample = min(4, len(candidates))
                sample_indices, sample_offsets = all_pair_candidate_csr(
                    sample, len(existing_xyz)
                )
                reference = _numpy_all_pair_batch(
                    candidates[:sample],
                    existing_xyz,
                    existing_epsilon,
                    existing_rmin,
                    existing_q,
                    existing_width,
                    candidate_epsilon,
                    candidate_rmin,
                    candidate_q,
                    candidate_width,
                    sample_indices,
                    sample_offsets,
                    dielectric=float(dielectric),
                    electrostatic_model=electrostatics,
                    radial_evaluation=radial_backend,
                    electrostatic_override=electrostatic_override,
                    resolved_pair_arrays=resolved_pair_arrays,
                )
                validation_error = max(
                    float(
                        np.max(
                            np.abs(electrostatic[:sample] - reference[0])
                            / np.maximum(1.0, np.abs(reference[0]))
                        )
                    ),
                    float(
                        np.max(
                            np.abs(short_range[:sample] - reference[1])
                            / np.maximum(1.0, np.abs(reference[1]))
                        )
                    ),
                )
                tolerance = 2.0e-5 if precision == "float32" else 5.0e-11
                if validation_error > tolerance:
                    raise RuntimeError(
                        f"GPU/reference relative error {validation_error:.3e}"
                    )
            backend = "torch"
        except Exception as exc:
            fallback_reason = str(exc)
            electrostatic, short_range = _numpy_all_pair_batch(
                candidates,
                existing_xyz,
                existing_epsilon,
                existing_rmin,
                existing_q,
                existing_width,
                candidate_epsilon,
                candidate_rmin,
                candidate_q,
                candidate_width,
                indices,
                offsets,
                dielectric=float(dielectric),
                electrostatic_model=electrostatics,
                radial_evaluation=radial_backend,
                electrostatic_override=electrostatic_override,
                resolved_pair_arrays=resolved_pair_arrays,
            )
            backend, device, precision = "numpy-vectorized", "cpu", "float64"
    else:
        electrostatic, short_range = _numpy_all_pair_batch(
            candidates,
            existing_xyz,
            existing_epsilon,
            existing_rmin,
            existing_q,
            existing_width,
            candidate_epsilon,
            candidate_rmin,
            candidate_q,
            candidate_width,
            indices,
            offsets,
            dielectric=float(dielectric),
            electrostatic_model=electrostatics,
            radial_evaluation=radial_backend,
            electrostatic_override=electrostatic_override,
            resolved_pair_arrays=resolved_pair_arrays,
        )
        backend, device, precision = "numpy-vectorized", "cpu", "float64"
        if requested not in {"auto", "cpu", "numpy"} and selected_device is None:
            fallback_reason = "requested GPU backend is unavailable"

    directional = (
        np.zeros(len(candidates), dtype=float)
        if resolved_directional is None
        else resolved_directional.energies(
            candidates,
            existing_xyz,
            radial_evaluation=radial_backend,
        )
    )
    return ConstructionInteractionBatch(
        electrostatic_kcal_per_mol=electrostatic,
        short_range_kcal_per_mol=short_range,
        directional_kcal_per_mol=directional,
        execution={
            "owner": "matrix-zaff",
            "electrostatics": electrostatics,
            "electrostatic_backend": realized_electrostatic_backend,
            "radial_evaluation": radial_backend,
                "short_range": "exppe",
            "short_range_parameter_source": (
                "atomic_mixing"
                if resolved_short_range is None
                else str(resolved_short_range.source)
            ),
            "directional_source": (
                None if resolved_directional is None else resolved_directional.source
            ),
            "directional_contacts": (
                0
                if resolved_directional is None
                else len(resolved_directional.interactions)
            ),
            "short_range_backend": backend,
            "device": device,
            "precision": precision,
            "pair_work": pair_work,
            "short_range_pair_work": short_range_pair_work,
            "autotune_gpu_min_pairs": threshold,
            "accelerator_validation_relative_error": validation_error,
            "accelerator_fallback": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "csr_role": "scheduling_hint_only",
            "input_csr_entries": int(len(indices)),
            "asymptotic_pair_scope": "all_pairs",
            "short_range_pair_scope": (
                "all_pairs" if dense_short_range else "neighbor_csr"
            ),
        },
    )


def batch_paired_candidate_interactions(
    candidates_angstrom: np.ndarray,
    environments_angstrom: np.ndarray,
    existing_epsilon_kcal_per_mol: np.ndarray,
    existing_rmin_half_angstrom: np.ndarray,
    existing_charges_e: np.ndarray,
    existing_charge_widths_angstrom: np.ndarray,
    candidate_epsilon_kcal_per_mol: np.ndarray,
    candidate_rmin_half_angstrom: np.ndarray,
    candidate_charges_e: np.ndarray,
    candidate_charge_widths_angstrom: np.ndarray,
    *,
    dielectric: float = 1.0,
    electrostatic_model: str = "gaussian_erf_all_pairs",
    radial_evaluation: str = "analytic",
    resolved_short_range: ResolvedExpPEPairTable | None = None,
    resolved_directional: ResolvedDirectionalExpPEContacts | None = None,
) -> ConstructionInteractionBatch:
    """Evaluate candidate fragments against candidate-specific environments.

    This is the paired counterpart of :func:`batch_candidate_interactions`.
    It is intended for population searches in which both rigid fragments move
    between rows, such as solvent--solvent terms in a cluster GA.
    """

    candidates = np.asarray(candidates_angstrom, dtype=float)
    environments = np.asarray(environments_angstrom, dtype=float)
    existing_epsilon = np.asarray(existing_epsilon_kcal_per_mol, dtype=float)
    existing_rmin = np.asarray(existing_rmin_half_angstrom, dtype=float)
    existing_q = np.asarray(existing_charges_e, dtype=float)
    existing_width = np.asarray(existing_charge_widths_angstrom, dtype=float)
    candidate_epsilon = np.asarray(candidate_epsilon_kcal_per_mol, dtype=float)
    candidate_rmin = np.asarray(candidate_rmin_half_angstrom, dtype=float)
    candidate_q = np.asarray(candidate_charges_e, dtype=float)
    candidate_width = np.asarray(candidate_charge_widths_angstrom, dtype=float)
    if (
        candidates.ndim != 3
        or environments.ndim != 3
        or candidates.shape[0] != environments.shape[0]
        or candidates.shape[2] != 3
        or environments.shape[2] != 3
    ):
        raise ValueError("paired ZAFF geometries have inconsistent shapes")
    candidate_atoms = candidates.shape[1]
    environment_atoms = environments.shape[1]
    if (
        existing_epsilon.shape != (environment_atoms,)
        or existing_rmin.shape != (environment_atoms,)
        or existing_q.shape != (environment_atoms,)
        or existing_width.shape != (environment_atoms,)
        or candidate_epsilon.shape != (candidate_atoms,)
        or candidate_rmin.shape != (candidate_atoms,)
        or candidate_q.shape != (candidate_atoms,)
        or candidate_width.shape != (candidate_atoms,)
        or np.any(~np.isfinite(candidates))
        or np.any(~np.isfinite(environments))
        or np.any(existing_epsilon < 0.0)
        or np.any(candidate_epsilon < 0.0)
        or np.any(existing_rmin <= 0.0)
        or np.any(candidate_rmin <= 0.0)
        or np.any(existing_width <= 0.0)
        or np.any(candidate_width <= 0.0)
        or not math.isfinite(float(dielectric))
        or float(dielectric) <= 0.0
    ):
        raise ValueError("paired ZAFF parameters must be finite and physical")
    electrostatics = str(electrostatic_model).strip().lower().replace("-", "_")
    radial_backend = str(radial_evaluation).strip().lower().replace("-", "_")
    if electrostatics not in {"gaussian_erf_all_pairs", "point_charge_all_pairs"}:
        raise ValueError("unsupported paired ZAFF electrostatic model")
    if radial_backend not in {"analytic", "lookup"}:
        raise ValueError("unsupported paired ZAFF radial evaluation")
    if not len(candidates):
        empty = np.empty(0, dtype=float)
        return ConstructionInteractionBatch(empty, empty.copy(), empty.copy(), {
            "owner": "matrix-zaff", "paired_environments": True,
            "electrostatics": electrostatics, "radial_evaluation": radial_backend,
        })
    resolved_arrays = (
        None
        if resolved_short_range is None
        else resolved_short_range.arrays(candidate_atoms, environment_atoms)
    )
    beta = 1.0 / np.sqrt(
        2.0 * (candidate_width[:, None] ** 2 + existing_width[None, :] ** 2)
    )
    charge_product = candidate_q[:, None] * existing_q[None, :]
    if resolved_arrays is None:
        epsilon = np.sqrt(candidate_epsilon[:, None] * existing_epsilon[None, :])
        r_min = candidate_rmin[:, None] + existing_rmin[None, :]
        alpha: float | np.ndarray | ExpPELookupPlan = math.sqrt(160.0)
    else:
        epsilon, r_min, raw_alpha = resolved_arrays
        alpha = (
            ExpPELookupPlan.compile(np.asarray(raw_alpha, dtype=float))
            if radial_backend == "lookup"
            else raw_alpha
        )
    electrostatic = np.zeros(len(candidates), dtype=float)
    short_range = np.zeros(len(candidates), dtype=float)
    pair_plane = max(1, candidate_atoms * environment_atoms)
    chunk_size = max(1, min(len(candidates), 2_000_000 // pair_plane))
    for start in range(0, len(candidates), chunk_size):
        stop = min(len(candidates), start + chunk_size)
        delta = candidates[start:stop, :, None, :] - environments[
            start:stop, None, :, :
        ]
        distance = np.maximum(np.linalg.norm(delta, axis=3), 1.0e-12)
        distance_bohr = distance / BOHR_TO_ANGSTROM
        if electrostatics == "gaussian_erf_all_pairs":
            if radial_backend == "lookup":
                coulomb = _gaussian_coulomb_lookup(distance, beta)
            else:
                from scipy.special import erf
                coulomb = erf(
                    beta[None, :, :] * BOHR_TO_ANGSTROM * distance_bohr
                ) / distance_bohr
        else:
            coulomb = 1.0 / distance_bohr
        electrostatic[start:stop] = (
            np.sum(charge_product[None, :, :] * coulomb, axis=(1, 2))
            * HARTREE_TO_KCAL_PER_MOL / float(dielectric)
        )
        dimensionless = _damped_exppe_values(
            distance / r_min[None, :, :],
            radial_evaluation=radial_backend,
            alpha=alpha,
        )
        short_range[start:stop] = np.sum(
            epsilon[None, :, :] * dimensionless, axis=(1, 2)
        )
    directional = (
        np.zeros(len(candidates), dtype=float)
        if resolved_directional is None
        else resolved_directional.paired_energies(
            candidates, environments, radial_evaluation=radial_backend
        )
    )
    return ConstructionInteractionBatch(
        electrostatic_kcal_per_mol=electrostatic,
        short_range_kcal_per_mol=short_range,
        directional_kcal_per_mol=directional,
        execution={
            "owner": "matrix-zaff",
            "paired_environments": True,
            "electrostatics": electrostatics,
            "electrostatic_backend": "direct",
            "radial_evaluation": radial_backend,
            "short_range": "exppe",
            "short_range_backend": "numpy-vectorized",
            "pair_work": int(len(candidates) * candidate_atoms * environment_atoms),
        },
    )


def _numpy_all_pair_batch(
    candidates: np.ndarray,
    existing_xyz: np.ndarray,
    existing_epsilon: np.ndarray,
    existing_rmin: np.ndarray,
    existing_q: np.ndarray,
    existing_width: np.ndarray,
    candidate_epsilon: np.ndarray,
    candidate_rmin: np.ndarray,
    candidate_q: np.ndarray,
    candidate_width: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_offsets: np.ndarray,
    *,
    dielectric: float,
    electrostatic_model: str,
    radial_evaluation: str,
    electrostatic_override: np.ndarray | None,
    resolved_pair_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(candidates)
    electrostatic = (
        np.zeros(count, dtype=float)
        if electrostatic_override is None
        else np.asarray(electrostatic_override, dtype=float).copy()
    )
    short_range = np.zeros(count, dtype=float)
    pair_plane = max(1, candidates.shape[1] * len(existing_xyz))
    chunk_size = max(1, min(count, 2_000_000 // pair_plane))
    beta = 1.0 / np.sqrt(
        2.0
        * (
            candidate_width[:, None] ** 2
            + existing_width[None, :] ** 2
        )
    )
    charge_product = candidate_q[:, None] * existing_q[None, :]
    if resolved_pair_arrays is None:
        epsilon = np.sqrt(
            candidate_epsilon[:, None] * existing_epsilon[None, :]
        )
        r_min = candidate_rmin[:, None] + existing_rmin[None, :]
        alpha: float | np.ndarray = math.sqrt(160.0)
    else:
        epsilon, r_min, alpha = resolved_pair_arrays
    dense_short_range = _csr_covers_all_pairs(
        neighbor_indices,
        neighbor_offsets,
        count,
        len(existing_xyz),
    )
    dense_alpha: float | np.ndarray | ExpPELookupPlan = alpha
    if (
        dense_short_range
        and radial_evaluation == "lookup"
        and np.asarray(alpha).ndim > 0
    ):
        dense_alpha = ExpPELookupPlan.compile(np.asarray(alpha, dtype=float))
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        delta = (
            candidates[start:stop, :, None, :]
            - existing_xyz[None, None, :, :]
        )
        distance = np.maximum(np.linalg.norm(delta, axis=3), 1.0e-12)
        distance_bohr = distance / BOHR_TO_ANGSTROM
        width_beta_bohr = beta * BOHR_TO_ANGSTROM
        if electrostatic_override is None:
            if electrostatic_model == "gaussian_erf_all_pairs":
                if radial_evaluation == "lookup":
                    coulomb_kernel = _gaussian_coulomb_lookup(
                        distance,
                        beta,
                    )
                else:
                    from scipy.special import erf

                    coulomb_kernel = (
                        erf(width_beta_bohr[None, :, :] * distance_bohr)
                        / distance_bohr
                    )
            else:
                # The penetration-free path deliberately keeps exact 1/r.  For
                # large systems this is the same Laplace kernel used by ZAFF FMM.
                coulomb_kernel = 1.0 / distance_bohr
            electrostatic[start:stop] = (
                np.sum(
                    charge_product[None, :, :] * coulomb_kernel,
                    axis=(1, 2),
                )
                * HARTREE_TO_KCAL_PER_MOL
                / dielectric
            )
        if dense_short_range:
            x = distance / r_min[None, :, :]
            dimensionless = _damped_exppe_values(
                x,
                radial_evaluation=radial_evaluation,
                alpha=dense_alpha,
            )
            short_range[start:stop] = np.sum(
                epsilon[None, :, :] * dimensionless, axis=(1, 2)
            )
    if not dense_short_range:
        for candidate_index in range(count):
            selected = neighbor_indices[
                neighbor_offsets[candidate_index] : neighbor_offsets[candidate_index + 1]
            ]
            if len(selected) == 0:
                continue
            delta = (
                candidates[candidate_index, :, None, :]
                - existing_xyz[None, selected, :]
            )
            distance = np.maximum(np.linalg.norm(delta, axis=2), 1.0e-12)
            if resolved_pair_arrays is None:
                local_r_min = (
                    candidate_rmin[:, None] + existing_rmin[None, selected]
                )
                local_epsilon = np.sqrt(
                    candidate_epsilon[:, None] * existing_epsilon[None, selected]
                )
                local_alpha: float | np.ndarray = alpha
            else:
                local_epsilon = epsilon[:, selected]
                local_r_min = r_min[:, selected]
                local_alpha = np.asarray(alpha)[:, selected]
            dimensionless = _damped_exppe_values(
                distance / local_r_min,
                radial_evaluation=radial_evaluation,
                alpha=local_alpha,
            )
            short_range[candidate_index] = np.sum(
                local_epsilon * dimensionless
            )
    return electrostatic, short_range


def _csr_covers_all_pairs(
    indices: np.ndarray,
    offsets: np.ndarray,
    candidate_count: int,
    environment_count: int,
) -> bool:
    if len(indices) != candidate_count * environment_count:
        return False
    expected_offsets = np.arange(
        0,
        (candidate_count + 1) * environment_count,
        environment_count if environment_count else 1,
        dtype=np.int64,
    )
    if environment_count == 0:
        expected_offsets = np.zeros(candidate_count + 1, dtype=np.int64)
    if not np.array_equal(offsets, expected_offsets):
        return False
    return environment_count == 0 or np.array_equal(
        indices,
        np.tile(np.arange(environment_count, dtype=np.int64), candidate_count),
    )


def _damped_exppe_values(
    scaled_distance: np.ndarray,
    *,
    radial_evaluation: str,
    alpha: float | np.ndarray | "ExpPELookupPlan",
) -> np.ndarray:
    if radial_evaluation == "lookup":
        if isinstance(alpha, ExpPELookupPlan):
            return alpha.evaluate_squared(scaled_distance**2)
        return _damped_exppe_lookup(scaled_distance**2, alpha)
    exp_full = np.exp(alpha * (1.0 - scaled_distance))
    exp_half = np.exp(0.5 * alpha * (1.0 - scaled_distance))
    polynomial = scaled_distance**4 - 2.0 * scaled_distance**2 + 3.0
    return exp_full - polynomial * exp_half


def evaluate_exppe_dimensionless(
    scaled_distance: np.ndarray,
    alpha: float | np.ndarray,
    *,
    radial_evaluation: str = "analytic",
) -> np.ndarray:
    """Shared non-bonded/H-bond/XB Exp-PE radial kernel."""

    evaluation = str(radial_evaluation).strip().lower()
    if evaluation not in {"analytic", "lookup"}:
        raise ValueError("Exp-PE radial evaluation must be analytic or lookup")
    return _damped_exppe_values(
        np.asarray(scaled_distance, dtype=float),
        radial_evaluation=evaluation,
        alpha=alpha,
    )


# Legacy import name retained for serialized clients.
evaluate_damped_exppe_dimensionless = evaluate_exppe_dimensionless


_LOOKUP_MIN_R2 = 0.04
_VDW_LOOKUP_MAX_X2 = 36.0
_GAUSSIAN_LOOKUP_MAX_X2 = 64.0
_LOOKUP_STEP_R2 = 1.0e-4


@dataclass(frozen=True)
class ExpPELookupPlan:
    """Preindexed lookup groups for one immutable matrix of Exp-PE types."""

    alpha_keys: tuple[float, ...]
    pair_shape: tuple[int, ...]
    flat_groups: tuple[np.ndarray, ...]

    @classmethod
    def compile(cls, alpha: np.ndarray) -> "ExpPELookupPlan":
        values = np.asarray(alpha, dtype=float)
        if values.ndim == 0 or np.any(~np.isfinite(values)) or np.any(values <= 4.0):
            raise ValueError("an Exp-PE lookup plan needs a physical alpha matrix")
        rounded = np.round(values, decimals=10)
        keys, inverse = np.unique(rounded, return_inverse=True)
        groups = []
        for index in range(len(keys)):
            group = np.flatnonzero(inverse == index).astype(np.intp, copy=False)
            group.setflags(write=False)
            groups.append(group)
        return cls(
            alpha_keys=tuple(float(value) for value in keys),
            pair_shape=tuple(int(value) for value in values.shape),
            flat_groups=tuple(groups),
        )

    def evaluate_squared(self, squared_distance: np.ndarray) -> np.ndarray:
        values = np.asarray(squared_distance, dtype=float)
        if values.shape[-len(self.pair_shape) :] != self.pair_shape:
            raise ValueError("Exp-PE lookup values do not end in the compiled pair shape")
        leading = values.shape[: values.ndim - len(self.pair_shape)]
        flat = values.reshape((-1, int(np.prod(self.pair_shape))))
        output = np.empty_like(flat)
        for alpha, group in zip(self.alpha_keys, self.flat_groups, strict=True):
            output[:, group] = _damped_exppe_lookup_scalar(flat[:, group], alpha)
        return output.reshape((*leading, *self.pair_shape))


@lru_cache(maxsize=1)
def _gaussian_construction_lookup_table() -> np.ndarray:
    gaussian_x2 = _LOOKUP_STEP_R2 * np.arange(
        int(round(_GAUSSIAN_LOOKUP_MAX_X2 / _LOOKUP_STEP_R2)) + 1,
        dtype=float,
    )
    from scipy.special import erf

    gaussian_x = np.sqrt(gaussian_x2)
    gaussian_coulomb = np.empty_like(gaussian_x)
    gaussian_coulomb[0] = 2.0 / math.sqrt(math.pi)
    gaussian_coulomb[1:] = erf(gaussian_x[1:]) / gaussian_x[1:]
    gaussian_coulomb.setflags(write=False)
    return gaussian_coulomb


@lru_cache(maxsize=128)
def _damped_exppe_lookup_table(alpha_key: float) -> np.ndarray:
    vdw_x2 = _LOOKUP_MIN_R2 + _LOOKUP_STEP_R2 * np.arange(
        int(round((_VDW_LOOKUP_MAX_X2 - _LOOKUP_MIN_R2) / _LOOKUP_STEP_R2))
        + 1,
        dtype=float,
    )
    x = np.sqrt(vdw_x2)
    alpha = float(alpha_key)
    exp_full = np.exp(alpha * (1.0 - x))
    exp_half = np.exp(0.5 * alpha * (1.0 - x))
    polynomial = x**4 - 2.0 * x**2 + 3.0
    exppe = exp_full - polynomial * exp_half
    exppe.setflags(write=False)
    return exppe


def _uniform_squared_lookup(
    squared: np.ndarray,
    table: np.ndarray,
    *,
    maximum: float,
    exact,
) -> np.ndarray:
    values = np.asarray(squared, dtype=float)
    result = np.empty_like(values)
    inside = (values >= _LOOKUP_MIN_R2) & (values < maximum)
    if np.any(inside):
        position = (values[inside] - _LOOKUP_MIN_R2) / _LOOKUP_STEP_R2
        lower = np.floor(position).astype(np.intp)
        fraction = position - lower
        result[inside] = table[lower] + fraction * (
            table[lower + 1] - table[lower]
        )
    if np.any(~inside):
        result[~inside] = exact(values[~inside])
    return result


def _gaussian_coulomb_lookup(
    distance_angstrom: np.ndarray,
    beta_per_angstrom: np.ndarray,
) -> np.ndarray:
    gaussian_coulomb = _gaussian_construction_lookup_table()
    argument = beta_per_angstrom[None, :, :] * distance_angstrom
    argument_squared = argument**2
    dimensionless = np.empty_like(argument_squared)
    inside = argument_squared < _GAUSSIAN_LOOKUP_MAX_X2
    if np.any(inside):
        position = argument_squared[inside] / _LOOKUP_STEP_R2
        lower = np.floor(position).astype(np.intp)
        fraction = position - lower
        dimensionless[inside] = gaussian_coulomb[lower] + fraction * (
            gaussian_coulomb[lower + 1] - gaussian_coulomb[lower]
        )
    if np.any(~inside):
        from scipy.special import erf

        dimensionless[~inside] = (
            erf(argument[~inside]) / argument[~inside]
        )
    return (
        BOHR_TO_ANGSTROM
        * beta_per_angstrom[None, :, :]
        * dimensionless
    )


def _damped_exppe_lookup_scalar(
    selected_values: np.ndarray, selected_alpha: float
) -> np.ndarray:
    key = round(float(selected_alpha), 10)

    def exact(values: np.ndarray) -> np.ndarray:
        x = np.sqrt(values)
        exp_full = np.exp(key * (1.0 - x))
        exp_half = np.exp(0.5 * key * (1.0 - x))
        polynomial = x**4 - 2.0 * x**2 + 3.0
        return exp_full - polynomial * exp_half

    return _uniform_squared_lookup(
        selected_values,
        _damped_exppe_lookup_table(key),
        maximum=_VDW_LOOKUP_MAX_X2,
        exact=exact,
    )


def _damped_exppe_lookup(
    scaled_distance_squared: np.ndarray,
    alpha: float | np.ndarray = math.sqrt(160.0),
) -> np.ndarray:
    values = np.asarray(scaled_distance_squared, dtype=float)
    alpha_values = np.asarray(alpha, dtype=float)

    if alpha_values.ndim == 0:
        return _damped_exppe_lookup_scalar(values, float(alpha_values))
    values, broadcast_alpha = np.broadcast_arrays(values, alpha_values)
    return ExpPELookupPlan.compile(broadcast_alpha).evaluate_squared(values)


def _torch_construction_device(requested: str) -> str | None:
    if requested in {"cpu", "numpy"}:
        return None
    try:
        import torch
    except ImportError:
        return None
    if requested in {"cuda", "torch-cuda", "gpu"} and torch.cuda.is_available():
        return "cuda"
    if requested in {"mps", "torch-mps", "gpu"} and torch.backends.mps.is_available():
        return "mps"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return None


def _torch_all_pair_batch(
    candidates: np.ndarray,
    existing_xyz: np.ndarray,
    existing_epsilon: np.ndarray,
    existing_rmin: np.ndarray,
    existing_q: np.ndarray,
    existing_width: np.ndarray,
    candidate_epsilon: np.ndarray,
    candidate_rmin: np.ndarray,
    candidate_q: np.ndarray,
    candidate_width: np.ndarray,
    *,
    dielectric: float,
    electrostatic_model: str,
    resolved_pair_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    device: str,
    allow_mixed_precision: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    import torch

    dtype = (
        torch.float32
        if device == "mps" and allow_mixed_precision
        else torch.float64
    )
    def tensor(value):
        array = np.asarray(value)
        if not array.flags.writeable:
            array = array.copy()
        return torch.as_tensor(array, dtype=dtype, device=device)

    candidate_tensor = tensor(candidates)
    existing_tensor = tensor(existing_xyz)
    distance = torch.linalg.vector_norm(
        candidate_tensor[:, :, None, :] - existing_tensor[None, None, :, :],
        dim=3,
    ).clamp_min(1.0e-12)
    beta = 1.0 / torch.sqrt(
        2.0
        * (
            tensor(candidate_width)[:, None] ** 2
            + tensor(existing_width)[None, :] ** 2
        )
    )
    distance_bohr = distance / BOHR_TO_ANGSTROM
    coulomb_kernel = (
        torch.erf(beta[None, :, :] * BOHR_TO_ANGSTROM * distance_bohr)
        / distance_bohr
        if electrostatic_model == "gaussian_erf_all_pairs"
        else 1.0 / distance_bohr
    )
    electrostatic = torch.sum(
        tensor(candidate_q)[None, :, None]
        * tensor(existing_q)[None, None, :]
        * coulomb_kernel,
        dim=(1, 2),
    ) * (HARTREE_TO_KCAL_PER_MOL / dielectric)
    if resolved_pair_arrays is None:
        epsilon = torch.sqrt(
            tensor(candidate_epsilon)[:, None] * tensor(existing_epsilon)[None, :]
        )
        r_min = tensor(candidate_rmin)[:, None] + tensor(existing_rmin)[None, :]
        alpha = math.sqrt(160.0)
    else:
        epsilon = tensor(resolved_pair_arrays[0])
        r_min = tensor(resolved_pair_arrays[1])
        alpha = tensor(resolved_pair_arrays[2])
    x = distance / r_min[None, :, :]
    radial = epsilon[None, :, :] * (
        torch.exp(alpha * (1.0 - x))
        - (x**4 - 2.0 * x**2 + 3.0)
        * torch.exp(0.5 * alpha * (1.0 - x))
    )
    short_range = torch.sum(radial, dim=(1, 2))
    return (
        electrostatic.detach().cpu().numpy().astype(float),
        short_range.detach().cpu().numpy().astype(float),
        "float32" if dtype == torch.float32 else "float64",
    )


__all__ = [
    "ConstructionInteractionBatch",
    "DirectionalExpPEInteraction",
    "ExpPELookupPlan",
    "ResolvedDirectionalExpPEContacts",
    "ResolvedExpPEPairTable",
    "all_pair_candidate_csr",
    "batch_candidate_interactions",
    "batch_paired_candidate_interactions",
    "compile_uff_exppe_pair_table",
    "evaluate_damped_exppe_dimensionless",
    "evaluate_exppe_dimensionless",
]
