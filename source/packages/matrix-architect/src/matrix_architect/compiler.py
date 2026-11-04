"""ARCHITECT compiler for native PIC fields and zoomable ZAFF models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from matrix_chem import (
    BOHR_TO_ANGSTROM,
    CHEMICAL_COUPLING_RELATIONS,
    PrimitiveFunctionalPolicy,
    PrimitiveMMForceField,
    SynthonTypingThresholds,
    build_topology_objects,
    classify_primitive_coupling,
    derive_primitive_mm_force_field,
    primitive_b_matrix,
    primitive_force_field_bonded_cartesian_gradient,
    refit_primitive_mm_force_field_hessian,
    read_enriched_xyz,
    read_primitive_contract,
    topology_snapshot_from_xyzin,
    topology_bonds_from_xyzin,
    uff_pair_parameters,
)
from matrix_smith import (
    FrozenGIC,
    GICDefinition,
    GICPrimitive,
    build_gic_b_matrix,
    independent_zaff_analysis_definition,
)
from matrix_qm import (
    read_cartesian_hessian_section,
    read_qm_population_section,
)
from matrix_zaff.bmatrix_derivative import build_sparse_gic_b_matrix_derivative
from matrix_zaff.nonbonded import (
    architect_nonbonded_cartesian_hessian_components as nonbonded_cartesian_hessian_components,
    build_nonbonded_neighbor_list,
    continuous_electrostatic_pair_components,
    gaussian_electrostatic_pair_components,
)

from .acquisition import (
    ZaffAnharmonicAcquisitionConfig,
    build_zaff_anharmonic_plan,
    build_zaff_local_sonic_definition,
    redundant_zaff_field_contract,
)
from .model import ZaffForceField, ZaffZoomLevel
from .workflow import zaff_workflow_contract


EXACT_TERM_POLICY = "EXACT_QM_TERMS_VPT2_RANKED_TRUNCATION"
REFITTED_POLICY = "REFITTED_NUMERICAL_ORTHOGONAL_MATCHING_PURSUIT"
HYBRID_PROTECTED_POLICY = "PROTECTED_QM_CORE_REFITTED_SMALL_COUPLING_TAIL"
HESSIAN_EIGENVALUE_TO_CM = 5140.487143715055
GF_FREQUENCY_REGIONS_CM = {
    "low_frequency": (0.0, 500.0),
    "fingerprint": (500.0, 2000.0),
    "stretching": (2000.0, math.inf),
}
MODAL_REGION_PRIORITIES = {
    "low_frequency": 4.0,
    "fingerprint": 2.0,
    "stretching": 1.0,
}
MODAL_CURVATURE_FLOOR_CM = 50.0
MODE_SHAPE_ERROR_FLOOR = 0.05
MODE_SHAPE_TARGET_MULTIPLIER = 5.0
REGIONAL_CURVATURE_ERROR_FLOOR = 0.10
REGIONAL_CURVATURE_TARGET_MULTIPLIER = 5.0
MAX_INCREMENTAL_COUPLINGS_PER_COORDINATE = 6
DEFAULT_CHARGE_PRIOR_SIGMA_E = 0.05
DEFAULT_CHARGE_PRIOR_WEIGHT = 1.0e-3
DEFAULT_MAXIMUM_CHARGE_DISPLACEMENT_E = 0.15
DEFAULT_CHARGE_OPTIMIZATION_ITERATIONS = 2
DEFAULT_NONBONDED_PRIOR_WEIGHT = 1.0e-3
DEFAULT_NONBONDED_OPTIMIZATION_ITERATIONS = 2


@dataclass(frozen=True)
class ZaffBuildConfig:
    """Frozen choices for the first PIC-versus-local-SONIC ZAFF compiler."""

    subtract_electrostatic: bool = True
    subtract_uff_vdw: bool = True
    vdw_model: str = "exppe_from_uff"
    optimize_electrostatic_charges: bool = True
    charge_prior_sigma_e: float = DEFAULT_CHARGE_PRIOR_SIGMA_E
    charge_prior_weight: float = DEFAULT_CHARGE_PRIOR_WEIGHT
    maximum_charge_displacement_e: float = DEFAULT_MAXIMUM_CHARGE_DISPLACEMENT_E
    charge_optimization_iterations: int = DEFAULT_CHARGE_OPTIMIZATION_ITERATIONS
    optimize_nonbonded_parameters: bool = True
    nonbonded_prior_weight: float = DEFAULT_NONBONDED_PRIOR_WEIGHT
    vdw_well_depth_prior_sigma_log: float = 0.25
    vdw_radius_prior_sigma_log: float = 0.05
    vdw_damping_prior_sigma_log: float = 0.10
    nonbonded_optimization_iterations: int = DEFAULT_NONBONDED_OPTIMIZATION_ITERATIONS
    preoptimized_atomic_charges_e: tuple[float, ...] | None = None
    preoptimized_vdw_radius_scale: float | None = None
    preoptimized_vdw_well_depth_scale: float | None = None
    preoptimized_vdw_damping_scale: float | None = None
    nonbonded_14_scale: float = 1.0
    vdw_damping_scale: float = 0.80
    vdw_damping_exponent: int = 24
    electrostatic_model: str = "gaussian_erf_all_pairs"
    electrostatic_gaussian_width_scale: float = 1.0
    topology_scaling: str = "continuous_erf_paths"
    topology_switch_alpha_per_angstrom: float = 12.0
    nonbonded_neighbor_cutoff_angstrom: float = 12.0
    fmm_precision: float = 1.0e-10
    fmm_minimum_atoms: int = 256
    stretch_form: str = "inverse_power"
    angle_form: str = "cosine"
    morse_alpha_scale: float = 2.215
    use_transferable_seed_priors: bool = True
    gaff_parameter_path: Path | None = None
    allow_inexact_equilibrium: bool = False
    fit_full_physical_pic_hessian: bool = False
    select_physical_pic_hessian_couplings: bool = False
    coordinate_chart: str = "adaptive"
    local_salc: bool = True
    global_symmetrization: bool = False
    parallel_workers: int = 0
    coupling_policy: str = EXACT_TERM_POLICY
    protected_coupling_mixing_threshold: float = 1.0
    protected_coupling_second_order_threshold: float = 0.10
    protected_diagonal_gap_floor_fraction: float = 0.10
    comparison_error_targets: tuple[float, ...] = (0.30, 0.20, 0.10, 0.05, 0.02, 0.01)
    anharmonic_stretch_step_angstrom: float = 0.01
    anharmonic_angular_step_radian: float = 0.02
    anharmonic_stretch_model: str = "inverse_power"
    apoc_source: str | None = None
    apoc_format: str = "auto"
    apoc_grid_level: int = 4
    apoc_charge: int | None = None
    explicit_charge_model: str | None = None
    explicit_atomic_charges_e: tuple[float, ...] | None = None
    explicit_bond_order_model: str | None = None
    explicit_bond_orders: tuple[tuple[int, int, float], ...] | None = None


def _zaff_nonlinear_functional_contract(coupling_policy: str) -> dict[str, Any]:
    """Return the frozen analytic ZAFF v1 prior and damping contract."""

    return {
        "charges": "INDEPENDENT_COMPLETE_CM5_ELSE_ORACLE_ELECTRONEGATIVITY",
        "nonbonded": (
            "SEPARATE_COVALENT_AND_CONTINUOUSLY_WEIGHTED_EXPPE_NO_D3_NO_D4"
        ),
        "electrostatics": "ALL_PAIR_GAUSSIAN_ERF_NO_TOPOLOGICAL_EXCLUSIONS",
        "harmonic_reference": (
            "FULL_QM_HESSIAN_MINUS_GAUSSIAN_ELECTROSTATICS_AND_VDW_REPRODUCED_EXACTLY"
        ),
        "harmonic_couplings": (
            "ALL_REQUIRED_TERMS_RETAINED; SPARSE_SUPPORT_ALLOWED_ONLY_AT_ZERO_HESSIAN_ERROR"
        ),
        "bond_stretches": (
            "ONE_GENERALIZED_INVERSE_POWER_FAMILY_WITH_LIBRARY_DE; QM_G_AND_K2_SET_R0_AND_P; "
            "HIGHER_DERIVATIVES_ADD_ASYMPTOTICALLY_VANISHING_1_OVER_R_CORRECTIONS"
        ),
        "valence_angles": "THREE_HARMONIC_COSINE_SERIES_MATCHING_QM_K2_K3_TIMES_NORMALIZED_BO_IJ_BO_JK",
        "linear_valence_angles": "UFF_SIN2_PRIOR_TIMES_NORMALIZED_BO_IJ_BO_JK",
        "out_of_plane": "EVEN_SIN_CHI_POLYNOMIAL_SINGLE_OR_PHYSICAL_PLANAR_BARRIER_DOUBLE_WELL",
        "torsions": "UFF_FOURIER_TIMES_NORMALIZED_BO_IJ_BO_JK_BO_KL",
        "anharmonic_acquisition": (
            "OPTIONAL_ADAPTIVE_POINTS_FOR_CUBIC_AND_QUARTIC_1_OVER_R_COS_THETA_AND_FOURIER_TERMS"
        ),
        "bond_order_source": "INDEPENDENT_COMPLETE_MAYER_ELSE_ORACLE_PAULING",
        "typing_update": "RERUN_ORACLE_PERCEPTION_AT_EVERY_GEOMETRY",
        "topology_labels": "DIAGNOSTIC_ONLY_NEVER_A_PARAMETER_SWITCH",
        "nonbonded_topology_scaling": (
            "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_WITH_SEPARATE_COVALENT_FIELD"
        ),
        "architecture": ("TOPOLOGY_FREE_COLLECTIVE_FIELD_PLUS_LOCAL_SYMMETRY_AWARE_CORRECTIONS"),
        "selection_metric": (
            "CONSTRAINED_PARETO_OF_CARTESIAN_ERROR_LOCALITY_SPARSITY_AND_LOCAL_SONIC_GAIN"
        ),
        "special_coordinates": (
            "PROTECTED_RING_CHARM_AND_NO_CEILING_TRANSITION_METAL_COORDINATE_CONTRACT"
        ),
        "qm_update": "REFIT_REFERENCE_GRADIENT_AND_HESSIAN_AT_PL1",
        "couplings": coupling_policy,
    }


def _collective_switch_audit(
    source: _CompilationSource,
    config: ZaffBuildConfig,
) -> dict[str, Any]:
    """Classify radial-switch values by graph distance for diagnostics only.

    Graph labels never enter the energy.  They are used here to detect a
    switch transition that overlaps bonded, geminal or torsional distance
    ranges and would consequently burden the corresponding local correction.
    """

    if config.vdw_model == "exppe_from_uff":
        return {
            "role": "NOT_APPLICABLE_TO_UNDAMPED_EXPPE",
            "separation_contract": {"passed": True, "applicable": False},
        }

    natoms = len(source.atomic_numbers)
    adjacency = [set() for _ in range(natoms)]
    for left, right in topology_bonds_from_xyzin(source.path):
        i = int(left) - 1
        j = int(right) - 1
        if 0 <= i < natoms and 0 <= j < natoms and i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)

    distances: dict[tuple[int, int], int] = {}
    for start in range(natoms):
        frontier = {start}
        visited = {start}
        for depth in range(1, 4):
            frontier = {
                neighbor
                for atom in frontier
                for neighbor in adjacency[atom]
                if neighbor not in visited
            }
            if not frontier:
                break
            visited.update(frontier)
            for atom in frontier:
                distances[tuple(sorted((start, atom)))] = depth

    coords_bohr = np.asarray(source.coordinates_angstrom, dtype=float) / BOHR_TO_ANGSTROM
    scale = float(source.vdw_damping_scale)
    exponent = int(config.vdw_damping_exponent)
    grouped: dict[str, list[tuple[float, float, float]]] = {
        "1-2_stretch": [],
        "1-3_geminal_bend": [],
        "1-4_torsional": [],
        "beyond_1-4": [],
    }
    labels = {
        1: "1-2_stretch",
        2: "1-3_geminal_bend",
        3: "1-4_torsional",
    }
    for i in range(natoms):
        for j in range(i + 1, natoms):
            pair_distance = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
            minimum_distance, _well_depth = uff_pair_parameters(
                int(source.atomic_numbers[i]),
                int(source.atomic_numbers[j]),
            )
            minimum_distance *= float(source.vdw_radius_scale)
            ratio = (scale * minimum_distance / pair_distance) ** exponent
            switch = 1.0 / (1.0 + ratio)
            log_slope = exponent * switch * (1.0 - switch)
            label = labels.get(distances.get((i, j), 999999), "beyond_1-4")
            grouped[label].append((switch, log_slope, pair_distance / minimum_distance))

    records: dict[str, Any] = {}
    for label, values in grouped.items():
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            records[label] = {"pair_count": 0}
            continue
        records[label] = {
            "pair_count": int(len(values)),
            "switch_minimum": float(np.min(array[:, 0])),
            "switch_median": float(np.median(array[:, 0])),
            "switch_maximum": float(np.max(array[:, 0])),
            "transition_pair_count": int(np.sum((array[:, 0] > 0.05) & (array[:, 0] < 0.95))),
            "maximum_dimensionless_slope": float(np.max(array[:, 1])),
            "normalized_distance_minimum": float(np.min(array[:, 2])),
            "normalized_distance_maximum": float(np.max(array[:, 2])),
        }
    stretch = records["1-2_stretch"]
    geminal = records["1-3_geminal_bend"]
    separation_passed = (
        stretch.get("pair_count", 0) == 0 or float(stretch["switch_maximum"]) <= 0.05
    ) and (geminal.get("pair_count", 0) == 0 or float(geminal["switch_maximum"]) <= 0.05)
    return {
        "role": "DIAGNOSTIC_ONLY_NOT_USED_BY_COLLECTIVE_ENERGY",
        "switch_scale": scale,
        "switch_exponent": exponent,
        "separation_contract": {
            "maximum_1-2_switch": 0.05,
            "maximum_1-3_switch": 0.05,
            "passed": bool(separation_passed),
            "failure_policy": ("REPORT_NO_RADIAL_GAP_AND_VALIDATE_LOCAL_BEND_SCAN"),
        },
        "classes": records,
    }


@dataclass(frozen=True)
class _CompilationSource:
    path: Path
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    atomic_numbers: tuple[int, ...]
    masses_amu: np.ndarray
    primitives: tuple[Any, ...]
    primitive_schema: str
    primitive_b_sha256: str
    source_hessian_bohr: np.ndarray
    residual_hessian_bohr: np.ndarray
    nonbonded_hessian_bohr: np.ndarray
    electrostatic_hessian_bohr: np.ndarray
    uff_vdw_hessian_bohr: np.ndarray
    nonbonded_gradient_bohr: np.ndarray
    electrostatic_gradient_bohr: np.ndarray
    uff_vdw_gradient_bohr: np.ndarray
    atomic_charges: np.ndarray | None
    vdw_radius_scale: float
    vdw_well_depth_scale: float
    vdw_damping_scale: float
    nonbonded_14_scale: float
    charge_source: str
    bond_order_source: str
    charge_owner: str
    bond_order_owner: str
    charge_optimization_audit: Mapping[str, Any]
    synthons: Any
    ring_topology: Mapping[str, Any]
    nonbonded_neighbor_count: int


@dataclass(frozen=True)
class _ZoomCandidate:
    basis: str
    selection_policy: str
    coordinate_count: int
    retained_coupling_count: int
    parameter_count: int
    hessian_bohr: np.ndarray
    relative_l2_error: float
    maximum_absolute_error: float
    sign_mismatch_count: int
    gf_frequency_rms_error_cm: float
    gf_frequency_maximum_absolute_error_cm: float
    gf_imaginary_mode_mismatch_count: int
    gf_region_metrics: Mapping[str, Mapping[str, float | int]]
    gf_regional_curvature_metrics: Mapping[str, Mapping[str, float | int]]
    gf_mode_overlap_metrics: Mapping[str, Any]
    gf_mode_overlap_matrix: np.ndarray
    gf_mode_assignment: np.ndarray
    gf_degenerate_subspace_metrics: tuple[Mapping[str, Any], ...]
    retained_couplings: tuple["_SelectedCoupling", ...] = ()
    fit_diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _FrequencyAuditContext:
    """Common mass metric and reference spectrum for same-geometry Wilson GF audits."""

    inverse_sqrt_cartesian_masses: np.ndarray
    mass_weighted_vibrational_basis: np.ndarray
    reference_eigenvalues: np.ndarray
    reference_frequencies_cm: np.ndarray
    reference_modes: np.ndarray


@dataclass(frozen=True)
class _SelectedCoupling:
    left: int
    right: int
    left_label: str
    right_label: str
    force_constant: float
    normalized_magnitude: float
    spectral_error_reduction_at_selection: float
    relation: str
    origin: str
    qm_mixing_index: float | None = None
    qm_relative_second_order_shift: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_indices": [self.left, self.right],
            "coordinate_labels": [self.left_label, self.right_label],
            "force_constant": self.force_constant,
            "normalized_magnitude": self.normalized_magnitude,
            "spectral_error_reduction_at_selection": (self.spectral_error_reduction_at_selection),
            "relation": self.relation,
            "origin": self.origin,
            "qm_mixing_index": self.qm_mixing_index,
            "qm_relative_second_order_shift": self.qm_relative_second_order_shift,
        }


class _IncrementalQRFit:
    """Memory-bounded QR updates for greedy harmonic support construction."""

    def __init__(
        self,
        target: np.ndarray,
        *,
        maximum_columns: int,
        rcond: float = 1.0e-10,
    ) -> None:
        self.target = np.asarray(target, dtype=float).reshape(-1)
        self.rcond = float(rcond)
        self.maximum_columns = int(maximum_columns)
        if self.maximum_columns < 0:
            raise ValueError("maximum incremental QR column count must be non-negative")
        maximum_rank = min(self.target.size, self.maximum_columns)
        self._q = np.zeros((self.target.size, maximum_rank), dtype=float)
        self._r = np.zeros((maximum_rank, self.maximum_columns), dtype=float)
        self._q_target = np.zeros(maximum_rank, dtype=float)
        self.rank = 0
        self.column_count = 0

    def add_column(self, column: np.ndarray) -> None:
        value = np.asarray(column, dtype=float).reshape(-1)
        if value.shape != self.target.shape:
            raise ValueError("incremental QR column has the wrong observation count")
        if self.column_count >= self.maximum_columns:
            raise ValueError("incremental QR support exceeded its declared column ceiling")
        q_active = self._q[:, : self.rank]
        projection = q_active.T @ value
        orthogonal = value - q_active @ projection
        if self.rank:
            correction = q_active.T @ orthogonal
            projection = projection + correction
            orthogonal = orthogonal - q_active @ correction
        norm = float(np.linalg.norm(orthogonal))
        independent = norm > self.rcond * max(float(np.linalg.norm(value)), 1.0)
        old_rank = self.rank
        self._r[:old_rank, self.column_count] = projection
        if independent:
            new_q = orthogonal / norm
            self._q[:, old_rank] = new_q
            self._r[old_rank, self.column_count] = norm
            self._q_target[old_rank] = float(new_q @ self.target)
            self.rank += 1
        self.column_count += 1

    @property
    def residual(self) -> np.ndarray:
        return self.target - self._q[:, : self.rank] @ self._q_target[: self.rank]

    @property
    def coefficients(self) -> np.ndarray:
        if not self.column_count:
            return np.zeros(0, dtype=float)
        r_active = self._r[: self.rank, : self.column_count]
        q_target = self._q_target[: self.rank]
        if self.rank == self.column_count and np.all(np.abs(np.diag(r_active)) > self.rcond):
            return _back_substitute_upper(r_active, q_target)
        values, *_ = np.linalg.lstsq(r_active, q_target, rcond=self.rcond)
        return np.asarray(values, dtype=float)

    def diagnostics(self) -> dict[str, Any]:
        r_active = self._r[: self.rank, : self.column_count]
        diagonal = np.abs(np.diag(r_active[:, : self.rank]))
        active = diagonal[diagonal > self.rcond]
        condition = None if not active.size else float(np.max(active) / np.min(active))
        residual = self.residual
        return {
            "solver": "INCREMENTAL_REORTHOGONALIZED_QR",
            "observation_count": int(self.target.size),
            "coefficient_count": int(self.column_count),
            "numerical_rank": int(self.rank),
            "rank_deficient": bool(self.rank < self.column_count),
            "effective_condition_number_estimate": condition,
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
            "maximum_absolute_residual": float(np.max(np.abs(residual), initial=0.0)),
            "objective": "EQUAL_CARTESIAN_AND_REGION_WEIGHTED_MODAL_HESSIAN_BLOCKS",
            "modal_region_priorities": dict(MODAL_REGION_PRIORITIES),
        }


def _back_substitute_upper(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    upper = np.asarray(matrix, dtype=float)
    rhs = np.asarray(target, dtype=float)
    solution = np.zeros_like(rhs)
    for row in range(len(rhs) - 1, -1, -1):
        tail = float(upper[row, row + 1 :] @ solution[row + 1 :])
        solution[row] = (rhs[row] - tail) / upper[row, row]
    return solution


@dataclass(frozen=True)
class _HybridCoordinateSet:
    """Local ring SONIC rows plus the unreplaced ORACLE PIC source rows."""

    b_matrix: np.ndarray
    labels: tuple[str, ...]
    audit: Mapping[str, Any]


def derive_zaff_pic_force_field(
    path: Path | str,
    *,
    thresholds: SynthonTypingThresholds = SynthonTypingThresholds(),
    subtract_electrostatic: bool = True,
    subtract_uff_vdw: bool = True,
    vdw_model: str = "exppe_from_uff",
    nonbonded_14_scale: float = 1.0,
    topology_scaling: str = "continuous_erf_paths",
    topology_switch_alpha_per_angstrom: float = 12.0,
    allow_inexact_equilibrium: bool = False,
    stretch_form: str = "inverse_power",
    angle_form: str = "cosine",
    morse_alpha_scale: float = 2.215,
    fit_full_physical_pic_hessian: bool = False,
    select_physical_pic_hessian_couplings: bool = False,
) -> PrimitiveMMForceField:
    """Construct the native redundant-PIC field owned by ARCHITECT.

    All nonbonded subtraction and bonded construction live in ARCHITECT.
    """

    config = ZaffBuildConfig(
        subtract_electrostatic=subtract_electrostatic,
        subtract_uff_vdw=subtract_uff_vdw,
        vdw_model=vdw_model,
        nonbonded_14_scale=nonbonded_14_scale,
        topology_scaling=topology_scaling,
        topology_switch_alpha_per_angstrom=topology_switch_alpha_per_angstrom,
        stretch_form=stretch_form,
        angle_form=angle_form,
        morse_alpha_scale=morse_alpha_scale,
        fit_full_physical_pic_hessian=fit_full_physical_pic_hessian,
        select_physical_pic_hessian_couplings=select_physical_pic_hessian_couplings,
        allow_inexact_equilibrium=allow_inexact_equilibrium,
    )
    source = _load_source(path, config)
    return _fit_pic_field(source, thresholds, config)


def build_zaff_force_field(
    path: Path | str,
    *,
    config: ZaffBuildConfig = ZaffBuildConfig(),
) -> ZaffForceField:
    """Compile an evaluable ZAFF field after a PIC/SONIC/hybrid Pareto audit."""

    _validate_config(config)
    source = _load_source(path, config)
    pic_b = primitive_b_matrix(source.primitives, source.coordinates_angstrom)
    sonic_definition, sonic_conditioning = build_zaff_local_sonic_definition(source.path)
    if sonic_definition.symmetrize or config.global_symmetrization:
        raise ValueError("ZAFF SONIC comparison forbids global molecular symmetrization")
    raw_sonic_b = np.asarray(
        build_gic_b_matrix(
            sonic_definition,
            coordinates_angstrom=source.coordinates_angstrom,
        ).rows,
        dtype=float,
    )
    pic_labels = tuple(
        _pic_coordinate_label(primitive, index)
        for index, primitive in enumerate(source.primitives, start=1)
    )
    raw_sonic_labels = tuple(
        _sonic_coordinate_label(sonic_definition, gic) for gic in sonic_definition.gics
    )
    sonic_chart = _build_stretch_preserving_sonic_set(
        pic_b,
        pic_labels,
        raw_sonic_b,
        raw_sonic_labels,
        source,
        sonic_definition,
    )
    sonic_b = sonic_chart.b_matrix
    sonic_labels = sonic_chart.labels
    hybrid = _build_hybrid_coordinate_set(
        pic_b,
        pic_labels,
        raw_sonic_b,
        raw_sonic_labels,
        source,
        sonic_definition,
    )
    derivative_workers = _resolved_parallel_workers(config.parallel_workers)
    pic_derivative = build_sparse_gic_b_matrix_derivative(
        _pic_derivative_definition(source, sonic_definition),
        coordinates_angstrom=source.coordinates_angstrom,
        parallel_workers=derivative_workers,
    ).to_dense()
    raw_sonic_derivative = build_sparse_gic_b_matrix_derivative(
        sonic_definition,
        coordinates_angstrom=source.coordinates_angstrom,
        parallel_workers=derivative_workers,
    ).to_dense()
    raw_sonic_index = {label: index for index, label in enumerate(raw_sonic_labels)}
    sonic_derivative = np.asarray(
        [
            (
                pic_derivative[int(str(label).split(":", 1)[0][3:]) - 1]
                if str(label).split(":", 1)[0].startswith("PIC")
                else raw_sonic_derivative[raw_sonic_index[str(label)]]
            )
            for label in sonic_labels
        ],
        dtype=float,
    )
    reference_basis = _vibrational_cartesian_basis(source.coordinates_angstrom / BOHR_TO_ANGSTROM)
    reference_eigenvalues = _projected_eigenvalues(
        source.source_hessian_bohr,
        reference_basis,
    )
    frequency_audit = _frequency_audit_context(source)
    if config.optimize_electrostatic_charges and config.subtract_electrostatic:
        source = _optimize_zaff_electrostatic_charges(
            source,
            config,
            pic_b,
            pic_derivative,
            reference_basis,
            frequency_audit,
        )
    pic_field = _fit_pic_field(
        source,
        SynthonTypingThresholds(
            coupling_relative=0.0,
            coupling_absolute=0.0,
            coupling_relations=CHEMICAL_COUPLING_RELATIONS,
        ),
        config,
    )
    coordinate_curvatures = {
        "PIC": _coordinate_curvature_hessian_bohr(
            pic_b,
            pic_derivative,
            source.nonbonded_gradient_bohr,
        ),
        "SONIC_LOCAL": _coordinate_curvature_hessian_bohr(
            sonic_b,
            sonic_derivative,
            source.nonbonded_gradient_bohr,
        ),
    }
    if hybrid is not None:
        sonic_index = {label: index for index, label in enumerate(raw_sonic_labels)}
        hybrid_derivative_rows = []
        for label in hybrid.labels:
            head = str(label).split(":", 1)[0]
            if head.startswith("PIC") and head[3:].isdigit():
                hybrid_derivative_rows.append(pic_derivative[int(head[3:]) - 1])
            else:
                hybrid_derivative_rows.append(raw_sonic_derivative[sonic_index[str(label)]])
        coordinate_curvatures["HYBRID_LOCAL"] = _coordinate_curvature_hessian_bohr(
            hybrid.b_matrix,
            np.asarray(hybrid_derivative_rows, dtype=float),
            source.nonbonded_gradient_bohr,
        )
    families = {
        "PIC": _coordinate_candidates(
            "PIC",
            pic_b,
            source,
            reference_basis,
            reference_eigenvalues,
            coordinate_labels=pic_labels,
            minimum_error_target=min(config.comparison_error_targets),
            parallel_workers=config.parallel_workers,
            selection_policy=config.coupling_policy,
            protected_mixing_threshold=config.protected_coupling_mixing_threshold,
            protected_second_order_threshold=(config.protected_coupling_second_order_threshold),
            protected_gap_floor_fraction=config.protected_diagonal_gap_floor_fraction,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvatures["PIC"],
        ),
        "SONIC_LOCAL": _coordinate_candidates(
            "SONIC_LOCAL",
            sonic_b,
            source,
            reference_basis,
            reference_eigenvalues,
            coordinate_labels=sonic_labels,
            minimum_error_target=min(config.comparison_error_targets),
            parallel_workers=config.parallel_workers,
            selection_policy=config.coupling_policy,
            protected_mixing_threshold=config.protected_coupling_mixing_threshold,
            protected_second_order_threshold=(config.protected_coupling_second_order_threshold),
            protected_gap_floor_fraction=config.protected_diagonal_gap_floor_fraction,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvatures["SONIC_LOCAL"],
        ),
    }
    if hybrid is not None:
        families["HYBRID_LOCAL"] = _coordinate_candidates(
            "HYBRID_LOCAL",
            hybrid.b_matrix,
            source,
            reference_basis,
            reference_eigenvalues,
            coordinate_labels=hybrid.labels,
            minimum_error_target=min(config.comparison_error_targets),
            parallel_workers=config.parallel_workers,
            selection_policy=config.coupling_policy,
            protected_mixing_threshold=config.protected_coupling_mixing_threshold,
            protected_second_order_threshold=(config.protected_coupling_second_order_threshold),
            protected_gap_floor_fraction=config.protected_diagonal_gap_floor_fraction,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvatures["HYBRID_LOCAL"],
        )
    comparison = _compare_coordinate_families(
        families,
        config.comparison_error_targets,
    )
    comparison["gf_contract"] = (
        "same-geometry mass-weighted Wilson GF; identical masses and external-mode "
        "projector for QM and every reduced Hessian"
    )
    comparison["hessian_chain_rule"] = "B^T F B + sum_i(g_i B'_i) + H_nonbonded"
    comparison["b_prime_backend"] = "ARCHITECT_SPARSE_ANALYTIC_FIRST_B_DERIVATIVE"
    comparison["coordinate_curvature_rms_hartree_per_bohr2"] = {
        name: _matrix_rms(curvature) for name, curvature in coordinate_curvatures.items()
    }
    comparison["gf_reference_frequencies_cm"] = frequency_audit.reference_frequencies_cm.tolist()
    comparison["gf_frequency_regions_cm"] = {
        "low_frequency": {"minimum": 0.0, "maximum_exclusive": 500.0},
        "fingerprint": {"minimum": 500.0, "maximum_exclusive": 2000.0},
        "stretching": {"minimum": 2000.0, "maximum_exclusive": None},
    }
    comparison["gf_low_frequency_interpretation"] = (
        "diagnostic stability audit only; large-amplitude anharmonicity limits the "
        "physical meaning of a purely harmonic low-frequency error"
    )
    comparison["gf_mode_overlap_contract"] = (
        "absolute mass-weighted normal-mode overlaps with global maximum-overlap "
        "assignment; near-degenerate blocks are audited by principal overlaps"
    )
    comparison["harmonic_fit_objective"] = (
        "equal normalized Cartesian-Hessian and reference-normal-mode blocks; modal "
        "curvature/mixing weights use low:fingerprint:stretching priorities 4:2:1"
    )
    comparison["regional_mode_shape_contract"] = {
        "mode_shape_error": "maximum over regions of 1 - mean squared subspace-adjusted overlap",
        "limit": "max(0.05, 5 * projected-Hessian relative-error target)",
        "regions": ["low_frequency", "fingerprint", "stretching"],
    }
    comparison["regional_curvature_contract"] = {
        "metric": (
            "maximum over regions of the RMS overlap-assigned normal-mode force-constant error, "
            "normalized by max(abs(lambda_QM), lambda_50cm)"
        ),
        "limit": "max(0.10, 5 * projected-Hessian relative-error target)",
        "regions": ["low_frequency", "fingerprint", "stretching"],
    }
    comparison["electrostatic_charge_optimization"] = dict(source.charge_optimization_audit)
    comparison["nonbonded_optimization"] = dict(source.charge_optimization_audit)
    equilibrium_refit = _equilibrium_parameter_refit_audit(
        pic_field,
        source.primitives,
    )
    comparison["equilibrium_parameter_refit"] = equilibrium_refit
    transferable_prior_audit = _transferable_full_fit_prior_audit(
        source,
        pic_field,
        config,
    )
    comparison["transferable_library_priors"] = transferable_prior_audit
    selection_families = (
        {"PIC": families["PIC"]}
        if config.coordinate_chart == "primitive"
        else {"SONIC_LOCAL": families["SONIC_LOCAL"]}
        if config.coordinate_chart == "sonic"
        else families
    )
    zoom_candidates = _select_adaptive_zoom_candidates(selection_families)
    stationarity = {
        "PIC": _stationarity_residual(pic_b, source.nonbonded_gradient_bohr),
        "SONIC_LOCAL": _stationarity_residual(
            sonic_b,
            source.nonbonded_gradient_bohr,
        ),
    }
    if hybrid is not None:
        stationarity["HYBRID_LOCAL"] = _stationarity_residual(
            hybrid.b_matrix,
            source.nonbonded_gradient_bohr,
        )
    zoom_levels = tuple(
        _zoom_level(
            identifier,
            candidate,
            pic_field,
            *stationarity[candidate.basis],
            local_salc=config.local_salc,
        )
        for identifier, candidate in zip(
            ("H0", "H1", "H2", "H3"),
            zoom_candidates,
            strict=True,
        )
    )
    active = zoom_levels[-1]
    selected_basis = active.coordinate_basis
    anharmonic_definition = independent_zaff_analysis_definition(sonic_definition)
    zaff_morse_depth_priors = (
        _zaff_morse_depth_priors(source)
        if config.use_transferable_seed_priors
        and config.stretch_form in {"morse", "inverse_power"}
        else ()
    )
    depth_by_stretch = {
        tuple(sorted(int(atom) + 1 for atom in primitive.atoms)): float(depth)
        for primitive, depth in zip(
            source.primitives,
            zaff_morse_depth_priors,
            strict=True,
        )
        if primitive.kind in {"bond", "hbond_dist"} and float(depth) > 0.0
    }
    anharmonic_config = ZaffAnharmonicAcquisitionConfig(
        stretch_step_angstrom=config.anharmonic_stretch_step_angstrom,
        angular_step_radian=config.anharmonic_angular_step_radian,
        parallel_workers=config.parallel_workers,
        stretch_model=config.anharmonic_stretch_model,
        stretch_dissociation_depths=tuple(
            (str(primitive.identifier), float(depth))
            for primitive in anharmonic_definition.primitives
            if str(primitive.function).upper() == "R"
            and (
                depth := depth_by_stretch.get(
                    tuple(sorted(int(atom) for atom in primitive.atoms))
                )
            )
            is not None
        ),
    )
    anharmonic_plan = build_zaff_anharmonic_plan(
        anharmonic_definition,
        source.coordinates_angstrom,
        config=anharmonic_config,
    )
    anharmonic_plan = replace(
        anharmonic_plan,
        redundant_field_contract=redundant_zaff_field_contract(
            sonic_definition,
            anharmonic_definition,
            sonic_conditioning,
        ),
    )
    return ZaffForceField(
        atoms=source.atoms,
        reference_coordinates_angstrom=source.coordinates_angstrom,
        hessian_hartree_per_bohr2=active.hessian_hartree_per_bohr2,
        gradient_reference_hartree_per_bohr=(
            primitive_force_field_bonded_cartesian_gradient(
                pic_field, source.primitives, source.coordinates_angstrom
            ) * BOHR_TO_ANGSTROM + source.nonbonded_gradient_bohr
        ),
        coordinate_contract={
            "pic": {
                "schema": source.primitive_schema,
                "effective_point_group": "C1",
                "global_symmetrization": False,
                "b_matrix_sha256": source.primitive_b_sha256,
                "coordinate_count": int(pic_b.shape[0]),
                "rank": int(np.linalg.matrix_rank(pic_b, tol=1.0e-9)),
                "redundant": bool(pic_b.shape[0] > np.linalg.matrix_rank(pic_b, tol=1.0e-9)),
                "equilibrium_parameter_refit": equilibrium_refit,
            },
            "sonic": {
                "backend": sonic_definition.backend,
                "effective_point_group": "C1",
                "coordinate_count": int(sonic_b.shape[0]),
                "rank": int(np.linalg.matrix_rank(sonic_b, tol=1.0e-9)),
                "redundant": bool(
                    sonic_b.shape[0] > np.linalg.matrix_rank(sonic_b, tol=1.0e-9)
                ),
                "redundant_harmonic_gauge": (
                    "MOORE_PENROSE_MINIMUM_NORM_WITH_EXACT_CARTESIAN_RECONSTRUCTION"
                ),
                "analysis_chart": "INDEPENDENT_REDUCTION_DEFERRED_TO_MODES_AND_DISPLACEMENTS",
                "local_salc": bool(sonic_conditioning["local_salc_realized"]),
                "local_salc_requested": bool(config.local_salc),
                "conditioning": dict(sonic_conditioning),
                "stretch_preserving_chart": dict(sonic_chart.audit),
                "global_symmetrization": False,
                "point_group_used_for_projection": False,
                "rings": _rings_coordinate_contract(
                    sonic_definition,
                    sonic_conditioning,
                ),
            },
            "hybrid": None if hybrid is None else dict(hybrid.audit),
            "selected_basis": selected_basis,
            "requested_coordinate_chart": config.coordinate_chart,
            "stretch_coordinate_policy": "INDIVIDUAL_UNCOMBINED_PRIMITIVES_IN_ALL_CHARTS",
            "harmonic_diagnostic_policy": (
                "LOWEST_PARAMETER_PIC_SONIC_OR_LOCAL_RING_HYBRID_AT_FIXED_ERROR"
            ),
            "production_anharmonic_zoom_policy": (
                "COMPLETE_HARMONIC_BASE_THEN_PHYSICAL_DIAGONAL_SELECTED_COUPLINGS_FULL_QFF"
            ),
            "provider_workflow": zaff_workflow_contract(),
            "zoom_basis_sequence": [level.coordinate_basis for level in zoom_levels],
            "harmonic_selection_policy": config.coupling_policy,
            "protected_coupling_thresholds": {
                "vpt2_mixing_index": config.protected_coupling_mixing_threshold,
                "relative_second_order_shift": (config.protected_coupling_second_order_threshold),
                "diagonal_gap_floor_fraction": (config.protected_diagonal_gap_floor_fraction),
            },
            "nonlinear_functional_contract": _zaff_nonlinear_functional_contract(
                config.coupling_policy
            ),
            "anharmonic_basis": "SONIC_LOCAL",
            "anharmonic_schema": anharmonic_plan.schema,
        },
        metadata={
            "source_xyzin": str(source.path.resolve()),
            "source_sha256": _sha256(source.path),
            "compiler": "ARCHITECT_PIC_SONIC_HYBRID_PARETO_V3",
            "runtime_limit": "HARMONIC_EGH_WITH_SERIALIZED_ANHARMONIC_ACQUISITION_PLAN",
            "charge_source": source.charge_source,
            "bond_order_source": source.bond_order_source,
            "oracle_ring_topology": dict(source.ring_topology),
            "atomic_charges_e": (
                None if source.atomic_charges is None else source.atomic_charges.tolist()
            ),
            "nonbonded_parameters": {
                "vdw_model": config.vdw_model,
                "vdw_radius_scale": source.vdw_radius_scale,
                "vdw_well_depth_scale": source.vdw_well_depth_scale,
                "vdw_damping_active": False,
                "vdw_pair_policy": (
                    "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_SEPARATE_FROM_COVALENT"
                ),
                "topology_partition": (
                    "COLLECTIVE_FIELD_PLUS_LOCAL_STRETCH_BEND_TORSION_CORRECTIONS"
                ),
                "topological_distance_switch_audit": _collective_switch_audit(
                    source,
                    config,
                ),
                "legacy_topological_pair_scale_ignored": source.nonbonded_14_scale,
                "regional_neighbor_cutoff_angstrom": config.nonbonded_neighbor_cutoff_angstrom,
                "regional_neighbor_count": source.nonbonded_neighbor_count,
                "electrostatic_model": config.electrostatic_model,
                "electrostatic_gaussian_width_scale": (config.electrostatic_gaussian_width_scale),
                "electrostatic_pair_policy": "ALL_PAIRS_NO_BONDED_EXCLUSIONS",
                "electrostatic_large_system_backend": "LAPLACE_FMM_FIELD_AND_HESSIAN_VECTOR",
                "fmm_precision": config.fmm_precision,
                "fmm_minimum_atoms": config.fmm_minimum_atoms,
                "dense_hessian_policy": "EXACT_DIRECT_REFERENCE_ONLY",
            },
            "electrostatic_charge_optimization": dict(source.charge_optimization_audit),
            "nonbonded_optimization": dict(source.charge_optimization_audit),
            "equilibrium_parameter_refit": equilibrium_refit,
            "full_fit_priors": transferable_prior_audit,
            "parallel_workers": _resolved_parallel_workers(config.parallel_workers),
            "nonbonded_subtraction": [
                label
                for label, enabled in (
                    ("ELECTROSTATIC", config.subtract_electrostatic),
                    ("EXPPE_FROM_UFF", config.subtract_uff_vdw),
                )
                if enabled
            ],
            "basis_comparison": comparison,
            "harmonic_selection_policy": config.coupling_policy,
            "protected_coupling_thresholds": {
                "vpt2_mixing_index": config.protected_coupling_mixing_threshold,
                "relative_second_order_shift": (config.protected_coupling_second_order_threshold),
                "diagonal_gap_floor_fraction": (config.protected_diagonal_gap_floor_fraction),
            },
            "nonlinear_functional_contract": _zaff_nonlinear_functional_contract(
                config.coupling_policy
            ),
        },
        zoom_levels=zoom_levels,
        active_zoom_level=active.identifier,
        anharmonic_model=anharmonic_plan,
    )


def _transferable_full_fit_prior_audit(
    source: _CompilationSource,
    pic_field: PrimitiveMMForceField,
    config: ZaffBuildConfig,
) -> dict[str, Any]:
    """Build and compare transferable starting values against the full QM fit."""

    if not config.use_transferable_seed_priors:
        return {"status": "DISABLED"}
    from .seed_compiler import (
        ZaffSeedCoverageError,
        compile_zaff_seed_model,
        zaff_seed_full_fit_priors,
    )

    try:
        model = compile_zaff_seed_model(
            source.atoms,
            source.coordinates_angstrom,
            source.atomic_charges,
            mayer_bond_orders={
                tuple(sorted((int(left), int(right)))): float(
                    source.synthons.bond_order(left, right)
                )
                for left, right in source.synthons._external_bond_orders
            }
            if source.synthons._external_bond_orders
            else None,
            gaff_path=config.gaff_parameter_path,
            charge_source=source.charge_source,
            bond_order_source=source.bond_order_source,
            charge_owner=source.charge_owner,
            bond_order_owner=source.bond_order_owner,
        )
    except (OSError, ValueError, ZaffSeedCoverageError) as exc:
        return {
            "status": "PARTIAL",
            "reason": str(exc),
            "stretch_prior_status": "MM3_D_E_APPLIED_WHERE_COVERED",
            "gaff_bend_torsion_nonbonded_status": "UNAVAILABLE",
        }
    priors = zaff_seed_full_fit_priors(model)
    fitted = []
    for parameter in pic_field.functional_parameters:
        values = dict(parameter.parameters)
        fitted.append(
            {
                "primitive_type_id": parameter.primitive_type_id,
                "primitive_indices": list(parameter.primitive_indices),
                "kind": parameter.kind,
                "form": parameter.form,
                "fitted_parameters": values,
                "parameter_source": parameter.parameter_source,
                "used_transferable_start": (
                    "ZAFF_DE_PRIOR" in parameter.parameter_source
                    or "MM3_MORSE_D_E_PRIOR" in parameter.parameter_source
                ),
            }
        )
    return {
        "status": "APPLIED",
        "role": "INITIAL_VALUES_AND_REGULARIZED_PRIORS_NOT_HARD_CONSTRAINTS",
        "parameter_provenance": dict(model["parameter_provenance"]),
        "weights": dict(priors["weights"]),
        "initial_parameters": dict(priors["initial_parameters"]),
        "fitted_parameters": fitted,
        "fit_audit_contract": dict(priors["fit_audit_contract"]),
    }


def _load_source(path: Path | str, config: ZaffBuildConfig) -> _CompilationSource:
    source = Path(path)
    if config.apoc_source is not None:
        from matrix_apoc import analyze_source, attach_apoc_analysis

        analysis = analyze_source(
            config.apoc_source,
            source_format=config.apoc_format,
            grid_level=config.apoc_grid_level,
            charge=config.apoc_charge,
        )
        attach_apoc_analysis(source, analysis)
    geometry = read_enriched_xyz(source)
    contract = read_primitive_contract(source)
    topology_snapshot = topology_snapshot_from_xyzin(source)
    ring_basis = dict(topology_snapshot["ring_basis"])
    if not bool(ring_basis.get("complete", True)):
        raise ValueError("ARCHITECT requires a complete ORACLE ring-basis contract")
    hessian = read_cartesian_hessian_section(source)
    numbers = tuple(int(value) for value in hessian.atomic_numbers)
    if len(numbers) != len(geometry.atoms):
        raise ValueError("#CARTESIAN_HESSIAN atom count differs from the ORACLE geometry")
    hessian_xyz = np.asarray(hessian.cartesian_coordinates_bohr, dtype=float) * BOHR_TO_ANGSTROM
    if not np.allclose(
        hessian_xyz,
        geometry.coordinates_angstrom,
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise ValueError("#CARTESIAN_HESSIAN geometry differs from the ORACLE geometry")
    population = None
    population_source = ""
    try:
        population, population_source = read_qm_population_section(source)
    except (KeyError, ValueError):
        pass
    if population is not None and population.natoms != len(numbers):
        raise ValueError("#QM_POPULATION atom count differs from the ORACLE geometry")
    explicit_charges = config.explicit_atomic_charges_e
    explicit_orders = config.explicit_bond_orders
    if (explicit_charges is None) != (config.explicit_charge_model is None):
        raise ValueError("explicit ZAFF charges require an explicit charge-model label")
    if (explicit_orders is None) != (config.explicit_bond_order_model is None):
        raise ValueError("explicit ZAFF bond orders require an explicit model label")
    candidate_charges: dict[int, float] = {}
    candidate_orders: dict[tuple[int, int], float] = {}
    charge_source = "ORACLE electronegativity estimate"
    bond_order_source = "ORACLE Pauling estimate"
    charge_owner = "ORACLE"
    bond_order_owner = "ORACLE"
    if population is not None:
        from matrix_apoc import evaluate_cm5_quality

        cm5_quality = evaluate_cm5_quality(population, numbers)
        if cm5_quality.passed:
            candidate_charges = {
                index: float(value) for index, value in enumerate(population.cm5_charges)
            }
            charge_source = f"#QM_POPULATION APOC CM5 {population_source}"
            charge_owner = "APOC"
        candidate_orders = {
            (left, right): float(population.mayer_bond_orders[left, right])
            for left in range(len(numbers))
            for right in range(left + 1, len(numbers))
        }
        bond_order_source = f"APOC Mayer {population_source}"
        bond_order_owner = "APOC"
    if explicit_charges is not None:
        explicit_charge_values = np.asarray(explicit_charges, dtype=float)
        if explicit_charge_values.shape != (len(numbers),) or np.any(
            ~np.isfinite(explicit_charge_values)
        ):
            raise ValueError("explicit ZAFF charges must contain one finite value per atom")
        candidate_charges = {
            index: float(value) for index, value in enumerate(explicit_charge_values)
        }
        charge_source = f"EXPLICIT {config.explicit_charge_model}"
        charge_owner = "USER"
    if explicit_orders is not None:
        candidate_orders = _explicit_bond_order_matrix(explicit_orders, len(numbers))
        bond_order_source = f"EXPLICIT {config.explicit_bond_order_model}"
        bond_order_owner = "USER"
    _continuous, _discrete, perceived_rings, synthons, _aromaticity = build_topology_objects(
        geometry.coordinates_angstrom,
        numbers,
        bond_order_overrides=candidate_orders,
        external_charges=candidate_charges,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
    )
    charges = np.asarray(
        [float(synthons.charge(index)) for index in range(len(numbers))],
        dtype=float,
    )
    charge_source = str(synthons._charge_source)
    bond_order_source = str(synthons._bond_order_source)
    frozen_ring_sizes = tuple(sorted(int(item["size"]) for item in topology_snapshot["rings"]))
    perceived_ring_sizes = tuple(sorted(len(item.atoms) for item in perceived_rings.rings))
    if perceived_ring_sizes != frozen_ring_sizes:
        raise ValueError(
            "ARCHITECT topology update changed the frozen ORACLE ring basis; "
            "regenerate the ORACLE state before force-field compilation"
        )
    if config.preoptimized_atomic_charges_e is not None:
        override = np.asarray(config.preoptimized_atomic_charges_e, dtype=float)
        if override.shape != (len(numbers),) or np.any(~np.isfinite(override)):
            raise ValueError("preoptimized ZAFF charges must contain one finite value per atom")
        charges = override
        charge_source = "ZAFF_PREOPTIMIZED_NONBONDED_STATE"
        charge_owner = "ZAFF"
    radius_scale = (
        1.0
        if config.preoptimized_vdw_radius_scale is None
        else float(config.preoptimized_vdw_radius_scale)
    )
    well_depth_scale = (
        1.0
        if config.preoptimized_vdw_well_depth_scale is None
        else float(config.preoptimized_vdw_well_depth_scale)
    )
    damping_scale = (
        float(config.vdw_damping_scale)
        if config.preoptimized_vdw_damping_scale is None
        else float(config.preoptimized_vdw_damping_scale)
    )
    one_four_scale = float(config.nonbonded_14_scale)
    regional_pairs = build_nonbonded_neighbor_list(
        hessian.cartesian_coordinates_bohr,
        cutoff_bohr=(float(config.nonbonded_neighbor_cutoff_angstrom) / BOHR_TO_ANGSTROM),
    )
    correction = nonbonded_cartesian_hessian_components(
        hessian.cartesian_coordinates_bohr,
        np.asarray(numbers, dtype=int),
        topology_bonds_from_xyzin(source),
        charges=charges,
        charge_source=charge_source,
        electrostatic=config.subtract_electrostatic,
        uff_vdw=config.subtract_uff_vdw,
        vdw_model=config.vdw_model,
        one_four_scale=one_four_scale,
        vdw_radius_scale=radius_scale,
        vdw_well_depth_scale=well_depth_scale,
        vdw_damping_scale=damping_scale,
        vdw_damping_exponent=config.vdw_damping_exponent,
        electrostatic_model=config.electrostatic_model,
        electrostatic_gaussian_width_scale=config.electrostatic_gaussian_width_scale,
        topology_scaling=config.topology_scaling,
        topology_switch_alpha_per_angstrom=config.topology_switch_alpha_per_angstrom,
    )
    source_hessian = np.asarray(hessian.cartesian_hessian, dtype=float)
    residual = 0.5 * (source_hessian - correction.total + (source_hessian - correction.total).T)
    return _CompilationSource(
        path=source,
        atoms=tuple(geometry.atoms),
        coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        atomic_numbers=numbers,
        masses_amu=np.asarray(hessian.masses_amu, dtype=float),
        primitives=tuple(contract.primitives),
        primitive_schema=contract.schema,
        primitive_b_sha256=contract.b_matrix_sha256,
        source_hessian_bohr=source_hessian,
        residual_hessian_bohr=residual,
        nonbonded_hessian_bohr=correction.total,
        electrostatic_hessian_bohr=correction.electrostatic,
        uff_vdw_hessian_bohr=correction.uff_vdw,
        nonbonded_gradient_bohr=correction.total_gradient,
        electrostatic_gradient_bohr=correction.electrostatic_gradient,
        uff_vdw_gradient_bohr=correction.uff_vdw_gradient,
        atomic_charges=(None if charges is None else np.asarray(charges, dtype=float)),
        vdw_radius_scale=radius_scale,
        vdw_well_depth_scale=well_depth_scale,
        vdw_damping_scale=damping_scale,
        nonbonded_14_scale=one_four_scale,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
        charge_owner=charge_owner,
        bond_order_owner=bond_order_owner,
        charge_optimization_audit={
            "status": "NOT_REQUESTED",
            "reference_charge_source": charge_source,
        },
        synthons=synthons,
        ring_topology={
            "schema": "matrix.architect.oracle_ring_contract.v1",
            "basis": ring_basis,
            "ring_count": len(frozen_ring_sizes),
            "ring_sizes": frozen_ring_sizes,
            "consumer_policy": "ORACLE_PERCEPTION_SMITH_RING_COORDINATES",
        },
        nonbonded_neighbor_count=len(regional_pairs),
    )


def _explicit_bond_order_matrix(
    rows: tuple[tuple[int, int, float], ...],
    natoms: int,
) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for left, right, value in rows:
        i = int(left) - 1
        j = int(right) - 1
        if not (0 <= i < natoms and 0 <= j < natoms and i != j):
            raise ValueError("explicit bond-order indices must be distinct and one-based")
        pair = (i, j) if i < j else (j, i)
        if pair in values:
            raise ValueError(f"duplicate explicit bond-order pair: {left} {right}")
        if not np.isfinite(float(value)):
            raise ValueError("explicit bond orders must be finite")
        values[pair] = float(value)
    expected = natoms * (natoms - 1) // 2
    if len(values) != expected:
        raise ValueError(
            f"explicit bond-order model is incomplete: found {len(values)}, expected {expected}"
        )
    return values


def _optimize_zaff_electrostatic_charges(
    source: _CompilationSource,
    config: ZaffBuildConfig,
    pic_b: np.ndarray,
    pic_derivative: np.ndarray,
    vibrational_basis: np.ndarray,
    frequency_audit: _FrequencyAuditContext,
) -> _CompilationSource:
    """Relax reference charges under exact total-charge and bounded-prior constraints.

    Charge relaxation is performed once in the redundant-PIC reference gauge,
    before comparing PIC and SONIC supports.  The bonded PIC diagonal is
    variable-projected at every Gauss--Newton step, so charges are not forced to
    absorb an error that a standard diagonal bonded term can already represent.
    """

    if source.atomic_charges is None:
        return replace(
            source,
            charge_optimization_audit={
                "status": "SKIPPED_NO_CHARGES",
                "reference_charge_source": source.charge_source,
            },
        )
    reference_charges = np.asarray(source.atomic_charges, dtype=float)
    natoms = len(reference_charges)
    if natoms < 2:
        return source
    null_basis = np.linalg.qr(np.ones((natoms, 1), dtype=float), mode="complete")[0][:, 1:]
    target_vector = _mode_aware_hessian_vector(
        source.source_hessian_bohr,
        source.source_hessian_bohr,
        vibrational_basis,
        frequency_audit,
    )
    diagonal_columns, _pairs, _target = _mode_aware_fit_system(
        pic_b,
        source.source_hessian_bohr,
        vibrational_basis,
        frequency_audit,
    )
    diagonal_design = np.column_stack(diagonal_columns)
    vdw_total = source.uff_vdw_hessian_bohr + _coordinate_curvature_hessian_bohr(
        pic_b,
        pic_derivative,
        source.uff_vdw_gradient_bohr,
    )
    vdw_vector = _mode_aware_hessian_vector(
        vdw_total,
        source.source_hessian_bohr,
        vibrational_basis,
        frequency_audit,
    )
    charges = reference_charges.copy()
    electrostatic_pair_basis = (
        gaussian_electrostatic_pair_components(
            source.coordinates_angstrom / BOHR_TO_ANGSTROM,
            np.asarray(source.atomic_numbers, dtype=int),
            width_scale=config.electrostatic_gaussian_width_scale,
        )
        if config.electrostatic_model == "gaussian_erf_all_pairs"
        else continuous_electrostatic_pair_components(
            source.coordinates_angstrom / BOHR_TO_ANGSTROM,
            np.asarray(source.atomic_numbers, dtype=int),
            one_four_scale=config.nonbonded_14_scale,
            topology_switch_alpha_per_angstrom=config.topology_switch_alpha_per_angstrom,
        )
    )
    previous_objective = math.inf
    iterations = 0
    for iteration in range(max(int(config.charge_optimization_iterations), 1)):
        iterations = iteration + 1
        _energy, electrostatic_gradient, electrostatic_hessian = (
            _assemble_electrostatic_pair_components(electrostatic_pair_basis, charges)
        )
        electrostatic_total = electrostatic_hessian + _coordinate_curvature_hessian_bohr(
            pic_b,
            pic_derivative,
            electrostatic_gradient,
        )
        charge_response = _electrostatic_charge_response_columns(
            source,
            charges,
            electrostatic_pair_basis,
            pic_b,
            pic_derivative,
            vibrational_basis,
            frequency_audit,
        )
        reduced_response = charge_response @ null_basis
        current_reduced = null_basis.T @ (charges - reference_charges)
        electrostatic_vector = _mode_aware_hessian_vector(
            electrostatic_total,
            source.source_hessian_bohr,
            vibrational_basis,
            frequency_audit,
        )
        design = np.column_stack((diagonal_design, reduced_response))
        rhs = (
            target_vector - vdw_vector - electrostatic_vector + (reduced_response @ current_reduced)
        )
        sigma = float(config.charge_prior_sigma_e)
        regularizer = np.zeros((natoms - 1, design.shape[1]), dtype=float)
        regularizer[:, len(diagonal_columns) :] = (
            math.sqrt(float(config.charge_prior_weight)) * np.eye(natoms - 1) / sigma
        )
        augmented_design = np.vstack((design, regularizer))
        augmented_rhs = np.concatenate((rhs, np.zeros(natoms - 1, dtype=float)))
        solution, *_ = np.linalg.lstsq(augmented_design, augmented_rhs, rcond=1.0e-10)
        proposed = null_basis @ solution[len(diagonal_columns) :]
        maximum = float(config.maximum_charge_displacement_e)
        proposed = np.clip(proposed, -maximum, maximum)
        proposed -= np.mean(proposed)
        proposed = np.clip(proposed, -maximum, maximum)
        proposed -= np.mean(proposed)
        new_charges = reference_charges + proposed
        model_vector = design @ solution
        objective = float(np.linalg.norm(model_vector - (target_vector - vdw_vector)))
        charges = 0.5 * charges + 0.5 * new_charges
        charges += (np.sum(reference_charges) - np.sum(charges)) / natoms
        if (
            abs(previous_objective - objective) <= 1.0e-8
            or np.linalg.norm(new_charges - charges) <= 1.0e-6
        ):
            previous_objective = objective
            break
        previous_objective = objective

    optimized = _electrostatic_correction_for_charges(source, config, charges)
    nonbonded_hessian = optimized.electrostatic + source.uff_vdw_hessian_bohr
    nonbonded_gradient = optimized.electrostatic_gradient + source.uff_vdw_gradient_bohr
    residual = source.source_hessian_bohr - nonbonded_hessian
    residual = 0.5 * (residual + residual.T)
    updated = replace(
        source,
        residual_hessian_bohr=residual,
        nonbonded_hessian_bohr=nonbonded_hessian,
        electrostatic_hessian_bohr=optimized.electrostatic,
        nonbonded_gradient_bohr=nonbonded_gradient,
        electrostatic_gradient_bohr=optimized.electrostatic_gradient,
        atomic_charges=charges,
        charge_source=f"{source.charge_source}+ZAFF_CONSTRAINED_RELAXATION",
    )
    charge_only_source = updated
    if config.optimize_nonbonded_parameters:
        updated, nonbonded_parameter_audit = _optimize_zaff_global_nonbonded_parameters(
            updated,
            config,
            pic_b,
            pic_derivative,
            vibrational_basis,
            frequency_audit,
        )
    else:
        nonbonded_parameter_audit = {"status": "NOT_REQUESTED"}
    frozen_candidate = _pic_diagonal_candidate_for_nonbonded(
        source,
        pic_b,
        pic_derivative,
        vibrational_basis,
        frequency_audit,
    )
    charge_only_candidate = _pic_diagonal_candidate_for_nonbonded(
        charge_only_source,
        pic_b,
        pic_derivative,
        vibrational_basis,
        frequency_audit,
    )
    optimized_candidate = _pic_diagonal_candidate_for_nonbonded(
        updated,
        pic_b,
        pic_derivative,
        vibrational_basis,
        frequency_audit,
    )
    omitted = replace(
        source,
        residual_hessian_bohr=source.source_hessian_bohr,
        nonbonded_hessian_bohr=np.zeros_like(source.source_hessian_bohr),
        electrostatic_hessian_bohr=np.zeros_like(source.source_hessian_bohr),
        uff_vdw_hessian_bohr=np.zeros_like(source.source_hessian_bohr),
        nonbonded_gradient_bohr=np.zeros_like(source.nonbonded_gradient_bohr),
        electrostatic_gradient_bohr=np.zeros_like(source.electrostatic_gradient_bohr),
        uff_vdw_gradient_bohr=np.zeros_like(source.uff_vdw_gradient_bohr),
    )
    omitted_candidate = _pic_diagonal_candidate_for_nonbonded(
        omitted,
        pic_b,
        pic_derivative,
        vibrational_basis,
        frequency_audit,
    )
    vdw_form_audit = []
    for vdw_model in ("exppe_from_uff", "uff_12_6"):
        trial_source = _nonbonded_control_source(
            source,
            config,
            charges,
            one_four_scale=updated.nonbonded_14_scale,
            vdw_model=vdw_model,
            vdw_radius_scale=updated.vdw_radius_scale,
            vdw_well_depth_scale=updated.vdw_well_depth_scale,
            vdw_damping_scale=updated.vdw_damping_scale,
        )
        trial_candidate = _pic_diagonal_candidate_for_nonbonded(
            trial_source,
            pic_b,
            pic_derivative,
            vibrational_basis,
            frequency_audit,
        )
        vdw_form_audit.append(
            {
                "vdw_model": vdw_model,
                "pic_diagonal_control": _candidate_record(trial_candidate),
            }
        )
    audit = {
        "status": "OPTIMIZED",
        "reference_charge_source": source.charge_source,
        "constraint": (
            "sum(q)=sum(reference charges); bounded displacement; "
            "quadratic prior to reference charges"
        ),
        "reference_total_charge_e": float(np.sum(reference_charges)),
        "optimized_total_charge_e": float(np.sum(charges)),
        "prior_sigma_e": float(config.charge_prior_sigma_e),
        "prior_weight": float(config.charge_prior_weight),
        "maximum_displacement_e": float(config.maximum_charge_displacement_e),
        "maximum_realized_displacement_e": float(np.max(np.abs(charges - reference_charges))),
        "rms_realized_displacement_e": float(np.sqrt(np.mean((charges - reference_charges) ** 2))),
        "iterations": iterations,
        "reference_charges_e": reference_charges.tolist(),
        "optimized_charges_e": charges.tolist(),
        "pic_diagonal_controls": {
            "frozen_cm5_and_nonbonded": _candidate_record(
                frozen_candidate, include_overlap_matrix=True
            ),
            "optimized_charges_fixed_vdw": _candidate_record(
                charge_only_candidate, include_overlap_matrix=True
            ),
            "optimized_charges_and_nonbonded": _candidate_record(
                optimized_candidate, include_overlap_matrix=True
            ),
            "nonbonded_omitted_and_absorbed_by_pic": _candidate_record(
                omitted_candidate, include_overlap_matrix=True
            ),
        },
        "vdw_form_sensitivity": vdw_form_audit,
        "nonbonded_parameter_optimization": nonbonded_parameter_audit,
        "interpretation": (
            "the omitted-nonbonded control is diagnostic only; production uses the "
            "constrained optimized charges and retains the declared Exp-PE term"
        ),
    }
    return replace(updated, charge_optimization_audit=audit)


def _optimize_zaff_global_nonbonded_parameters(
    source: _CompilationSource,
    config: ZaffBuildConfig,
    pic_b: np.ndarray,
    pic_derivative: np.ndarray,
    vibrational_basis: np.ndarray,
    frequency_audit: _FrequencyAuditContext,
) -> tuple[_CompilationSource, dict[str, Any]]:
    """Relax the noncovalent Exp-PE layer with bonded projection."""

    if source.atomic_charges is None:
        return source, {"status": "SKIPPED_NO_CHARGES"}
    target_vector = _mode_aware_hessian_vector(
        source.source_hessian_bohr,
        source.source_hessian_bohr,
        vibrational_basis,
        frequency_audit,
    )
    diagonal_columns, _pairs, _target = _mode_aware_fit_system(
        pic_b,
        source.source_hessian_bohr,
        vibrational_basis,
        frequency_audit,
    )
    diagonal_design = np.column_stack(diagonal_columns)
    reference = np.zeros(2, dtype=float)
    parameters = reference.copy()
    prior_sigma = np.asarray(
        [
            config.vdw_well_depth_prior_sigma_log,
            config.vdw_radius_prior_sigma_log,
        ],
        dtype=float,
    )

    def trial(values: np.ndarray) -> _CompilationSource:
        return _nonbonded_control_source(
            source,
            config,
            np.asarray(source.atomic_charges, dtype=float),
            one_four_scale=float(source.nonbonded_14_scale),
            vdw_model=config.vdw_model,
            vdw_radius_scale=float(math.exp(values[1])),
            vdw_well_depth_scale=float(math.exp(values[0])),
            vdw_damping_scale=float(source.vdw_damping_scale),
        )

    def combined_vector(item: _CompilationSource) -> np.ndarray:
        curvature = _coordinate_curvature_hessian_bohr(
            pic_b,
            pic_derivative,
            item.nonbonded_gradient_bohr,
        )
        return _mode_aware_hessian_vector(
            item.nonbonded_hessian_bohr + curvature,
            source.source_hessian_bohr,
            vibrational_basis,
            frequency_audit,
        )

    def profiled_residual(item: _CompilationSource) -> float:
        rhs = target_vector - combined_vector(item)
        coefficients, *_ = np.linalg.lstsq(diagonal_design, rhs, rcond=1.0e-8)
        return float(np.linalg.norm(diagonal_design @ coefficients - rhs))

    initial_source = trial(reference)
    initial_residual = profiled_residual(initial_source)

    def regularized_objective(values: np.ndarray) -> float:
        candidate = trial(values)
        if not bool(_collective_switch_audit(candidate, config)["separation_contract"]["passed"]):
            return float("inf")
        residual = profiled_residual(candidate)
        prior = float(config.nonbonded_prior_weight) * float(
            np.sum(((values - reference) / prior_sigma) ** 2)
        )
        return residual * residual + prior

    current_objective = regularized_objective(parameters)
    iterations = 0
    for iteration in range(max(int(config.nonbonded_optimization_iterations), 1)):
        iterations = iteration + 1
        current = trial(parameters)
        current_vector = combined_vector(current)
        response_columns = []
        steps = (1.0e-3, 2.0e-4)
        for index, step in enumerate(steps):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            denominator = float(plus[index] - minus[index])
            response_columns.append(
                (combined_vector(trial(plus)) - combined_vector(trial(minus))) / denominator
            )
        response = np.column_stack(response_columns)
        design = np.column_stack((diagonal_design, response))
        rhs = target_vector - current_vector + response @ (parameters - reference)
        regularizer = np.zeros((2, design.shape[1]), dtype=float)
        regularizer[:, len(diagonal_columns) :] = math.sqrt(
            float(config.nonbonded_prior_weight)
        ) * np.diag(1.0 / prior_sigma)
        solution, *_ = np.linalg.lstsq(
            np.vstack((design, regularizer)),
            np.concatenate((rhs, np.zeros(2, dtype=float))),
            rcond=1.0e-8,
        )
        proposed = solution[len(diagonal_columns) :]
        proposed[0] = float(np.clip(proposed[0], math.log(0.5), math.log(2.0)))
        proposed[1] = float(np.clip(proposed[1], math.log(0.90), math.log(1.10)))
        accepted = None
        accepted_objective = current_objective
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = parameters + fraction * (proposed - parameters)
            candidate_objective = regularized_objective(candidate)
            if candidate_objective < accepted_objective - 1.0e-12:
                accepted = candidate
                accepted_objective = candidate_objective
                break
        if accepted is None:
            search_steps = np.asarray((0.15, 0.03), dtype=float) / (2.0**iteration)
            for index, step in enumerate(search_steps):
                for sign in (-1.0, 1.0):
                    candidate = parameters.copy()
                    candidate[index] += sign * step
                    candidate[0] = float(np.clip(candidate[0], math.log(0.5), math.log(2.0)))
                    candidate[1] = float(np.clip(candidate[1], math.log(0.90), math.log(1.10)))
                    candidate_objective = regularized_objective(candidate)
                    if candidate_objective < accepted_objective - 1.0e-12:
                        accepted = candidate
                        accepted_objective = candidate_objective
            if accepted is None:
                break
        if np.linalg.norm(accepted - parameters) <= 1.0e-6:
            parameters = accepted
            current_objective = accepted_objective
            break
        parameters = accepted
        current_objective = accepted_objective

    optimized = trial(parameters)
    final_residual = profiled_residual(optimized)
    audit = {
        "status": "OPTIMIZED",
        "parameterization": ("global log well-depth and radius scales"),
        "reference": {
            "vdw_well_depth_scale": 1.0,
            "vdw_radius_scale": 1.0,
        },
        "optimized": {
            "vdw_well_depth_scale": float(math.exp(parameters[0])),
            "vdw_radius_scale": float(math.exp(parameters[1])),
        },
        "bounds": {
            "vdw_well_depth_scale": [0.5, 2.0],
            "vdw_radius_scale": [0.90, 1.10],
        },
        "prior_sigma": {
            "log_vdw_well_depth": float(prior_sigma[0]),
            "log_vdw_radius": float(prior_sigma[1]),
        },
        "prior_weight": float(config.nonbonded_prior_weight),
        "profiled_joint_residual_before": initial_residual,
        "profiled_joint_residual_after": final_residual,
        "regularized_objective_after": current_objective,
        "iterations": iterations,
        "scope": (
            "low-dimensional first level; element/synthon-resolved parameters require "
            "the multi-structure ZAFF training corpus"
        ),
    }
    return optimized, audit


def _nonbonded_control_source(
    source: _CompilationSource,
    config: ZaffBuildConfig,
    charges: np.ndarray,
    *,
    one_four_scale: float,
    vdw_model: str,
    vdw_radius_scale: float = 1.0,
    vdw_well_depth_scale: float = 1.0,
    vdw_damping_scale: float | None = None,
) -> _CompilationSource:
    selected_damping_scale = (
        source.vdw_damping_scale if vdw_damping_scale is None else float(vdw_damping_scale)
    )
    correction = nonbonded_cartesian_hessian_components(
        source.coordinates_angstrom / BOHR_TO_ANGSTROM,
        np.asarray(source.atomic_numbers, dtype=int),
        topology_bonds_from_xyzin(source.path),
        charges=np.asarray(charges, dtype=float),
        charge_source=source.charge_source,
        electrostatic=True,
        uff_vdw=True,
        vdw_model=vdw_model,
        one_four_scale=one_four_scale,
        vdw_radius_scale=vdw_radius_scale,
        vdw_well_depth_scale=vdw_well_depth_scale,
        vdw_damping_scale=selected_damping_scale,
        vdw_damping_exponent=config.vdw_damping_exponent,
        electrostatic_model=config.electrostatic_model,
        electrostatic_gaussian_width_scale=config.electrostatic_gaussian_width_scale,
        topology_scaling=config.topology_scaling,
        topology_switch_alpha_per_angstrom=config.topology_switch_alpha_per_angstrom,
    )
    nonbonded_hessian = correction.total
    residual = source.source_hessian_bohr - nonbonded_hessian
    residual = 0.5 * (residual + residual.T)
    return replace(
        source,
        residual_hessian_bohr=residual,
        nonbonded_hessian_bohr=nonbonded_hessian,
        electrostatic_hessian_bohr=correction.electrostatic,
        uff_vdw_hessian_bohr=correction.uff_vdw,
        nonbonded_gradient_bohr=correction.total_gradient,
        electrostatic_gradient_bohr=correction.electrostatic_gradient,
        uff_vdw_gradient_bohr=correction.uff_vdw_gradient,
        atomic_charges=np.asarray(charges, dtype=float),
        vdw_radius_scale=float(vdw_radius_scale),
        vdw_well_depth_scale=float(vdw_well_depth_scale),
        vdw_damping_scale=selected_damping_scale,
        nonbonded_14_scale=float(one_four_scale),
    )


def _electrostatic_correction_for_charges(
    source: _CompilationSource,
    config: ZaffBuildConfig,
    charges: np.ndarray,
):
    return nonbonded_cartesian_hessian_components(
        source.coordinates_angstrom / BOHR_TO_ANGSTROM,
        np.asarray(source.atomic_numbers, dtype=int),
        topology_bonds_from_xyzin(source.path),
        charges=np.asarray(charges, dtype=float),
        charge_source=source.charge_source,
        electrostatic=True,
        uff_vdw=False,
        vdw_model=config.vdw_model,
        one_four_scale=config.nonbonded_14_scale,
        electrostatic_model=config.electrostatic_model,
        electrostatic_gaussian_width_scale=config.electrostatic_gaussian_width_scale,
        topology_scaling=config.topology_scaling,
        topology_switch_alpha_per_angstrom=config.topology_switch_alpha_per_angstrom,
    )


def _electrostatic_charge_response_columns(
    source: _CompilationSource,
    charges: np.ndarray,
    pair_components: Mapping[tuple[int, int], tuple[float, np.ndarray, np.ndarray]],
    pic_b: np.ndarray,
    pic_derivative: np.ndarray,
    vibrational_basis: np.ndarray,
    frequency_audit: _FrequencyAuditContext,
) -> np.ndarray:
    columns = []
    for index in range(len(charges)):
        derivative_gradient = np.zeros(3 * len(charges), dtype=float)
        derivative_hessian = np.zeros(
            (3 * len(charges), 3 * len(charges)),
            dtype=float,
        )
        for (left, right), (_energy, gradient, hessian) in pair_components.items():
            if index == left:
                factor = float(charges[right])
            elif index == right:
                factor = float(charges[left])
            else:
                continue
            derivative_gradient += factor * gradient
            derivative_hessian += factor * hessian
        derivative = derivative_hessian + _coordinate_curvature_hessian_bohr(
            pic_b,
            pic_derivative,
            derivative_gradient,
        )
        columns.append(
            _mode_aware_hessian_vector(
                derivative,
                source.source_hessian_bohr,
                vibrational_basis,
                frequency_audit,
            )
        )
    return np.column_stack(columns)


def _assemble_electrostatic_pair_components(
    pair_components: Mapping[tuple[int, int], tuple[float, np.ndarray, np.ndarray]],
    charges: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    charge_values = np.asarray(charges, dtype=float)
    size = 3 * len(charge_values)
    energy = 0.0
    gradient = np.zeros(size, dtype=float)
    hessian = np.zeros((size, size), dtype=float)
    for (left, right), (pair_energy, pair_gradient, pair_hessian) in pair_components.items():
        factor = float(charge_values[left] * charge_values[right])
        energy += factor * pair_energy
        gradient += factor * pair_gradient
        hessian += factor * pair_hessian
    return energy, gradient, 0.5 * (hessian + hessian.T)


def _pic_diagonal_candidate_for_nonbonded(
    source: _CompilationSource,
    pic_b: np.ndarray,
    pic_derivative: np.ndarray,
    vibrational_basis: np.ndarray,
    frequency_audit: _FrequencyAuditContext,
) -> _ZoomCandidate:
    curvature = _coordinate_curvature_hessian_bohr(
        pic_b,
        pic_derivative,
        source.nonbonded_gradient_bohr,
    )
    projected_b = np.asarray(pic_b, dtype=float) @ np.asarray(vibrational_basis, dtype=float)
    diagonal_columns = [BOHR_TO_ANGSTROM**2 * np.outer(row, row).reshape(-1) for row in projected_b]
    target = (
        vibrational_basis.T @ (source.residual_hessian_bohr - curvature) @ vibrational_basis
    ).reshape(-1)
    fit = _IncrementalQRFit(target, maximum_columns=len(diagonal_columns))
    for column in diagonal_columns:
        fit.add_column(column)
    coefficients = fit.coefficients
    return _candidate(
        "PIC",
        pic_b,
        np.diag(coefficients),
        (),
        source.nonbonded_hessian_bohr,
        vibrational_basis,
        _projected_eigenvalues(source.source_hessian_bohr, vibrational_basis),
        frequency_audit=frequency_audit,
        coordinate_curvature_bohr=curvature,
        selection_policy="PIC_DIAGONAL_NONBONDED_SENSITIVITY_CONTROL",
    )


def _fit_pic_field(
    source: _CompilationSource,
    thresholds: SynthonTypingThresholds,
    config: ZaffBuildConfig,
) -> PrimitiveMMForceField:
    morse_depth_priors = (
        _zaff_morse_depth_priors(source)
        if config.use_transferable_seed_priors
        and config.stretch_form in {"morse", "inverse_power"}
        else ()
    )
    result = derive_primitive_mm_force_field(
        source.primitives,
        source.coordinates_angstrom,
        source.source_hessian_bohr,
        source.atomic_numbers,
        hessian_unit="hartree/bohr^2",
        synthons=source.synthons,
        thresholds=thresholds,
        cartesian_hessian_correction=source.nonbonded_hessian_bohr,
        cartesian_gradient_correction=source.nonbonded_gradient_bohr,
        functional_policy=PrimitiveFunctionalPolicy(
            stretch_form=config.stretch_form,
            angle_form=config.angle_form,
            morse_alpha_scale=config.morse_alpha_scale,
            morse_depth_priors_hartree=morse_depth_priors,
        ),
    )
    if config.fit_full_physical_pic_hessian:
        result = refit_primitive_mm_force_field_hessian(
            result,
            source.primitives,
            source.coordinates_angstrom,
            source.residual_hessian_bohr / BOHR_TO_ANGSTROM**2,
            source.atomic_numbers,
            synthons=source.synthons,
            functional_policy=PrimitiveFunctionalPolicy(
                stretch_form=config.stretch_form,
                angle_form=config.angle_form,
                morse_alpha_scale=config.morse_alpha_scale,
                morse_depth_priors_hartree=morse_depth_priors,
            ),
        )
    if config.select_physical_pic_hessian_couplings:
        if not config.fit_full_physical_pic_hessian:
            raise ValueError("physical PIC selection requires the evaluated full Hessian fit")
        from .harmonic_selection import select_physical_pic_couplings

        result = select_physical_pic_couplings(
            result,
            source.primitives,
            source.coordinates_angstrom,
            source.source_hessian_bohr,
            source.nonbonded_hessian_bohr,
            source.masses_amu,
        ).field
    subtraction = tuple(
        label
        for label, enabled in (
            ("ELECTROSTATIC", config.subtract_electrostatic),
            ("EXPPE_FROM_UFF_NONCOVALENT", config.subtract_uff_vdw),
        )
        if enabled
    )
    result = replace(
        result,
        nonbonded_subtraction=subtraction,
        charge_source=source.charge_source,
        nonbonded_14_scale=float(source.nonbonded_14_scale),
        nonbonded_topology_scaling=config.topology_scaling.upper(),
        topology_switch_alpha_per_angstrom=float(config.topology_switch_alpha_per_angstrom),
        source_cartesian_hessian_rms=_matrix_rms(source.source_hessian_bohr),
        electrostatic_hessian_rms=_matrix_rms(source.electrostatic_hessian_bohr),
        uff_vdw_hessian_rms=_matrix_rms(source.uff_vdw_hessian_bohr),
        residual_cartesian_hessian_rms=_matrix_rms(source.residual_hessian_bohr),
        nonbonded_cartesian_gradient_rms=_vector_rms(source.nonbonded_gradient_bohr),
    )
    if result.equilibrium_fit_status != "PASS" and not config.allow_inexact_equilibrium:
        raise ValueError(
            "ZAFF PIC equilibrium refit did not cancel the total Cartesian gradient "
            f"(RMS={result.equilibrium_gradient_rms:.6g}, "
            f"max={result.equilibrium_gradient_max:.6g} hartree/angstrom); "
            "refine the typing/coupling thresholds or Fourier model. "
            "Use allow_inexact_equilibrium only for diagnostics."
        )
    return result


def _zaff_morse_depth_priors(source: _CompilationSource) -> tuple[float, ...]:
    """Return transferable ZAFF-library D_e values in frozen ORACLE PIC order."""

    from .mm3_morse import load_mm3_morse_catalog

    catalog = load_mm3_morse_catalog()
    values = np.zeros(len(source.primitives), dtype=float)
    stretches = {"bond", "hbond_dist"}
    for index, primitive in enumerate(source.primitives):
        if primitive.kind not in stretches or len(primitive.atoms) != 2:
            continue
        left, right = primitive.atoms
        distance = float(
            np.linalg.norm(
                source.coordinates_angstrom[int(left)] - source.coordinates_angstrom[int(right)]
            )
        )
        try:
            prior = catalog.interpolate_from_synthons(
                source.synthons,
                source.atomic_numbers,
                int(left),
                int(right),
                distance,
            )
        except ValueError:
            continue
        values[index] = prior.d_e_hartree
    return tuple(float(value) for value in values)


def _equilibrium_parameter_refit_audit(
    field: PrimitiveMMForceField,
    primitives: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Serialize the post-nonbonded R0/theta0/Fourier stationarity refit."""

    functional_rows = []
    for parameter in field.functional_parameters:
        values = dict(parameter.parameters)
        equilibrium_name = next(
            (name for name in ("R0", "THETA0", "Q0") if name in values),
            None,
        )
        equilibrium_value = (
            float(parameter.reference_value)
            if equilibrium_name is None
            else float(values[equilibrium_name])
        )
        functional_rows.append(
            {
                "primitive_type_id": parameter.primitive_type_id,
                "primitive_indices": [index + 1 for index in parameter.primitive_indices],
                "primitive_labels": (
                    []
                    if primitives is None
                    else [primitives[index].label for index in parameter.primitive_indices]
                ),
                "kind": parameter.kind,
                "form": parameter.form,
                "reference_value": float(parameter.reference_value),
                "equilibrium_parameter": equilibrium_name,
                "optimized_equilibrium_value": equilibrium_value,
                "equilibrium_shift": equilibrium_value - float(parameter.reference_value),
                "value_unit": (
                    "angstrom"
                    if parameter.kind in {"bond", "hbond_dist", "pseudo_bond"}
                    else "radian"
                ),
                "gradient_at_reference": float(parameter.gradient_at_qm),
                "curvature_at_reference": float(parameter.curvature_at_qm),
                "parameters": {name: float(value) for name, value in parameter.parameters},
                "parameter_source": parameter.parameter_source,
            }
        )
    torsion_rows = [
        {
            "primitive_type_id": parameter.primitive_type_id,
            "primitive_indices": [index + 1 for index in parameter.primitive_indices],
            "primitive_labels": (
                []
                if primitives is None
                else [primitives[index].label for index in parameter.primitive_indices]
            ),
            "reference_phase_radian": float(parameter.equilibrium_phase),
            "gradient_at_reference": float(parameter.gradient_at_qm),
            "curvature_at_reference": float(parameter.local_curvature),
            "periodicities": list(parameter.periodicities),
            "fit_method": parameter.fit_method,
            "gradient_residual": float(parameter.gradient_residual),
            "curvature_residual": float(parameter.curvature_residual),
        }
        for parameter in field.torsion_fourier_parameters
    ]
    return {
        "schema": "matrix.architect.equilibrium_parameter_refit.v1",
        "ordering": (
            "optimize constrained nonbonded parameters, recompute their analytic "
            "gradient, then refit bonded equilibrium parameters"
        ),
        "redundant_pic_gauge": (
            "minimum equilibrium-parameter displacement in chemical units using "
            "the retained coupled internal Hessian"
        ),
        "coupling_center": (
            "QM_REFERENCE_FOR_ALL_REFERENCE_CENTERED_FULL_HESSIAN_COUPLINGS"
            if field.coupling_reference_values is not None
            else "FITTED_EQUILIBRIUM_FOR_NONTORSIONAL_COORDINATES; "
            "QM_REFERENCE_FOR_TORSIONS_WITH_EXPLICIT_FOURIER_GRADIENT"
        ),
        "status": field.equilibrium_fit_status,
        "iterations": int(field.equilibrium_fit_iterations),
        "parameter_type_split_count": int(field.equilibrium_type_split_count),
        "total_cartesian_gradient_rms_hartree_per_angstrom": float(field.equilibrium_gradient_rms),
        "total_cartesian_gradient_max_hartree_per_angstrom": float(field.equilibrium_gradient_max),
        "functional_parameters": functional_rows,
        "torsion_fourier_parameters": torsion_rows,
    }


def _build_stretch_preserving_sonic_set(
    pic_b: np.ndarray,
    pic_labels: Sequence[str],
    sonic_b: np.ndarray,
    sonic_labels: Sequence[str],
    source: _CompilationSource,
    sonic_definition: Any,
) -> _HybridCoordinateSet:
    """Use SMITH SONICs except that every stretch remains one native PIC row."""

    primitive_by_id = {
        str(primitive.identifier): primitive for primitive in sonic_definition.primitives
    }
    combined_stretch_indices: list[int] = []
    retained_sonic_indices: list[int] = []
    for index, gic in enumerate(sonic_definition.gics):
        components = tuple(gic.coefficients) or ((str(gic.primitive_id), 1.0),)
        contains_stretch = any(
            str(primitive_by_id[str(identifier)].function).upper() == "R"
            for identifier, _coefficient in components
        )
        # A shared-proton coordinate is intentionally redundant with its two
        # native distances.  It belongs to the physical ZAFF field even though
        # it is assembled from stretch primitives; independence is imposed
        # only by the later analysis/modal chart.
        redundant_transfer = str(getattr(gic, "family", "")).upper() == "PROTON_TRANSFER"
        (
            combined_stretch_indices
            if contains_stretch and not redundant_transfer
            else retained_sonic_indices
        ).append(index)
    stretch_pic_indices = [
        index
        for index, primitive in enumerate(source.primitives)
        if str(primitive.kind).lower() == "bond" and len(primitive.atoms) == 2
    ]
    rows = [np.asarray(pic_b[index], dtype=float) for index in stretch_pic_indices]
    labels = [str(pic_labels[index]) for index in stretch_pic_indices]
    rows.extend(np.asarray(sonic_b[index], dtype=float) for index in retained_sonic_indices)
    labels.extend(str(sonic_labels[index]) for index in retained_sonic_indices)
    target_rank = int(np.linalg.matrix_rank(sonic_b, tol=1.0e-9))
    current_rank = _coordinate_row_rank(rows)
    restoration_indices: list[int] = []
    if current_rank < target_rank:
        for index, primitive in enumerate(source.primitives):
            if index in stretch_pic_indices:
                continue
            trial = rows + [np.asarray(pic_b[index], dtype=float)]
            trial_rank = _coordinate_row_rank(trial)
            if trial_rank <= current_rank:
                continue
            rows = trial
            labels.append(str(pic_labels[index]))
            restoration_indices.append(index)
            current_rank = trial_rank
            if current_rank >= target_rank:
                break
    if current_rank < target_rank:
        raise ValueError(
            "stretch-preserving SONIC chart cannot close the vibrational rank "
            f"without combined stretches: {current_rank}/{target_rank}"
        )
    matrix = np.vstack(rows)
    return _HybridCoordinateSet(
        matrix,
        tuple(labels),
        {
            "policy": "SMITH_SONIC_WITH_UNCOMBINED_ORACLE_PIC_STRETCHES_V1",
            "coordinate_count": int(matrix.shape[0]),
            "rank": int(current_rank),
            "target_rank": int(target_rank),
            "stretch_coordinate_count": len(stretch_pic_indices),
            "stretch_pic_labels": [str(pic_labels[index]) for index in stretch_pic_indices],
            "excluded_sonic_rows_containing_stretches": [
                str(sonic_labels[index]) for index in combined_stretch_indices
            ],
            "retained_sonic_coordinate_count": len(retained_sonic_indices),
            "redundant_field": bool(matrix.shape[0] > current_rank),
            "shared_proton_coordinate_count": sum(
                str(getattr(sonic_definition.gics[index], "family", "")).upper()
                == "PROTON_TRANSFER"
                for index in retained_sonic_indices
            ),
            "rank_restoration_pic_labels": [
                str(pic_labels[index]) for index in restoration_indices
            ],
            "stretch_contract": "EVERY_STRETCH_IS_ONE_UNCOMBINED_NATIVE_PRIMITIVE",
        },
    )


def _build_hybrid_coordinate_set(
    pic_b: np.ndarray,
    pic_labels: Sequence[str],
    sonic_b: np.ndarray,
    sonic_labels: Sequence[str],
    source: _CompilationSource,
    sonic_definition: Any,
) -> _HybridCoordinateSet | None:
    """Replace ring-local PIC source rows by SMITH's compact special rows.

    All bonds, non-ring valence angles, out-of-plane coordinates outside the
    selected ring blocks and exocyclic torsions remain native ORACLE PICs.  A
    deterministic rank-restoration pass may retain a covered PIC only when the
    local special block would otherwise fail to span the vibrational space.
    """

    special_indices = [
        index for index, gic in enumerate(sonic_definition.gics) if _is_local_ring_special(gic)
    ]
    if not special_indices:
        return None

    definition_primitives = {
        str(primitive.identifier): primitive for primitive in sonic_definition.primitives
    }
    special_primitive_identifiers = {
        str(primitive_identifier)
        for index in special_indices
        for primitive_identifier, _coefficient in sonic_definition.gics[index].coefficients
    }
    unknown = sorted(special_primitive_identifiers - set(definition_primitives))
    if unknown:
        raise ValueError(
            "SMITH local ring rows reference absent definition primitives: " + ", ".join(unknown)
        )
    special_signatures = {
        _definition_primitive_signature(definition_primitives[identifier])
        for identifier in special_primitive_identifiers
    }
    ring_four_atom_stencils = {
        tuple(sorted(int(atom) - 1 for atom in definition_primitives[identifier].atoms))
        for identifier in special_primitive_identifiers
        if str(definition_primitives[identifier].function).upper() == "U"
    }
    covered_source_indices = {
        index
        for index, primitive in enumerate(source.primitives)
        if _source_primitive_signature(primitive) in special_signatures
        or (
            str(primitive.kind) in {"dihedral", "out_of_plane"}
            and tuple(sorted(int(atom) for atom in primitive.atoms)) in ring_four_atom_stencils
        )
    }

    retained_pic_indices = [
        index
        for index, _primitive in enumerate(source.primitives)
        if index not in covered_source_indices
    ]
    rows = [np.asarray(sonic_b[index], dtype=float) for index in special_indices]
    labels = [str(sonic_labels[index]) for index in special_indices]
    rows.extend(np.asarray(pic_b[index], dtype=float) for index in retained_pic_indices)
    labels.extend(str(pic_labels[index]) for index in retained_pic_indices)

    target_rank = int(np.linalg.matrix_rank(sonic_b, tol=1.0e-9))
    current_rank = _coordinate_row_rank(rows)
    fallback_pic_indices: list[int] = []
    if current_rank < target_rank:
        for source_index in sorted(covered_source_indices):
            trial = rows + [np.asarray(pic_b[source_index], dtype=float)]
            trial_rank = _coordinate_row_rank(trial)
            if trial_rank <= current_rank:
                continue
            rows = trial
            labels.append(str(pic_labels[source_index]))
            fallback_pic_indices.append(source_index)
            current_rank = trial_rank
            if current_rank >= target_rank:
                break
    if current_rank < target_rank:
        raise ValueError(
            f"local ring hybrid failed to close the SONIC rank: {current_rank}/{target_rank}"
        )

    matrix = np.vstack(rows)
    special_records = [
        {
            "name": str(sonic_definition.gics[index].name),
            "family": str(sonic_definition.gics[index].family),
            "primitive_identifiers": [
                str(identifier)
                for identifier, _coefficient in sonic_definition.gics[index].coefficients
            ],
        }
        for index in special_indices
    ]
    audit = {
        "policy": "LOCAL_RING_SONIC_BLOCKS_PLUS_UNREPLACED_ORACLE_PICS_V1",
        "coordinate_count": int(matrix.shape[0]),
        "rank": int(current_rank),
        "target_rank": int(target_rank),
        "redundant": bool(matrix.shape[0] > current_rank),
        "local_salc": True,
        "global_symmetrization": False,
        "special_coordinate_count": len(special_indices),
        "special_coordinates": special_records,
        "replaced_pic_count": len(covered_source_indices),
        "replaced_pic_labels": [str(pic_labels[index]) for index in sorted(covered_source_indices)],
        "retained_native_pic_count": len(retained_pic_indices),
        "rank_restoration_pic_count": len(fallback_pic_indices),
        "rank_restoration_pic_labels": [str(pic_labels[index]) for index in fallback_pic_indices],
        "chemical_contract": (
            "retain all native PIC rows outside selected ring deformation/puckering "
            "source blocks; never replace special ring coordinates by exocyclic angles"
        ),
    }
    return _HybridCoordinateSet(matrix, tuple(labels), audit)


def _is_local_ring_special(gic: Any) -> bool:
    name = str(getattr(gic, "name", "")).upper()
    family = str(getattr(gic, "family", "")).upper()
    return name.startswith(("RDEF", "RPCK")) or family in {
        "CYCLIC_BEND",
        "RING_DEFORMATION",
        "RING_PUCKER_COMPONENT",
        "RING_PUCKERING",
    }


def _rings_coordinate_contract(
    definition: Any,
    conditioning: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Freeze the exact RINGS primitive chart consumed by ARCHITECT."""

    primitives = {str(primitive.identifier): primitive for primitive in definition.primitives}
    records: list[dict[str, Any]] = []
    for gic in definition.gics:
        if not _is_local_ring_special(gic):
            continue
        sources = []
        for identifier, coefficient in gic.coefficients:
            primitive = primitives[str(identifier)]
            sources.append(
                {
                    "identifier": str(identifier),
                    "function": str(primitive.function),
                    "atoms_one_based": [int(atom) for atom in primitive.atoms],
                    "mode": int(primitive.mode),
                    "coefficient": float(coefficient),
                }
            )
        records.append(
            {
                "name": str(gic.name),
                "family": str(gic.family),
                "sources": sources,
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    model = str(conditioning.get("ring_puckering_model", "triangular_flap"))
    return {
        "schema": "matrix.rings.curvilinear_coordinate_contract.v1",
        "model": model,
        "chart": ("BALANCED_TRIANGULAR_FLAP" if model == "triangular_flap" else model.upper()),
        "curvilinear": True,
        "derivatives": "ANALYTIC_SPARSE_B_AND_B_PRIME",
        "special_coordinate_count": len(records),
        "puckering_coordinate_count": sum(
            str(record["family"]).upper() in {"RING_PUCKER_COMPONENT", "RING_PUCKERING"}
            for record in records
        ),
        "cyclic_deformation_coordinate_count": sum(
            str(record["family"]).upper() in {"CYCLIC_BEND", "RING_DEFORMATION"}
            for record in records
        ),
        "definition_sha256": hashlib.sha256(encoded).hexdigest(),
        "coordinates": records,
    }


def _definition_primitive_signature(primitive: Any) -> tuple[Any, ...]:
    kind = {
        "R": "bond",
        "A": "angle",
        "L": "linear_bend",
        "D": "dihedral",
        "U": "out_of_plane",
        "H": "out_of_plane_height",
    }.get(str(primitive.function).upper(), str(primitive.function).lower())
    atoms = tuple(int(atom) - 1 for atom in primitive.atoms)
    return _normalized_primitive_signature(kind, atoms, int(primitive.mode))


def _source_primitive_signature(primitive: Any) -> tuple[Any, ...]:
    return _normalized_primitive_signature(
        str(primitive.kind),
        tuple(int(atom) for atom in primitive.atoms),
        int(primitive.mode),
    )


def _normalized_primitive_signature(
    kind: str,
    atoms: tuple[int, ...],
    mode: int,
) -> tuple[Any, ...]:
    if kind == "bond":
        normalized_atoms = tuple(sorted(atoms))
    elif kind in {"angle", "linear_bend"} and len(atoms) == 3:
        normalized_atoms = (min(atoms[0], atoms[2]), atoms[1], max(atoms[0], atoms[2]))
    elif kind == "dihedral":
        normalized_atoms = min(atoms, tuple(reversed(atoms)))
    elif kind in {"out_of_plane", "out_of_plane_height"}:
        normalized_atoms = (atoms[0], *tuple(sorted(atoms[1:])))
    else:
        normalized_atoms = atoms
    return kind, normalized_atoms, mode


def _coordinate_row_rank(rows: Sequence[np.ndarray]) -> int:
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.vstack(rows), tol=1.0e-9))


def _pic_derivative_definition(
    source: _CompilationSource,
    template: GICDefinition,
) -> GICDefinition:
    """Represent frozen ORACLE PIC rows in ARCHITECT's analytic B-prime contract."""

    function_by_kind = {
        "bond": "R",
        "angle": "A",
        "linear_bend": "L",
        "dihedral": "D",
        "out_of_plane": "U",
        "out_of_plane_height": "H",
    }
    primitives = []
    gics = []
    for index, primitive in enumerate(source.primitives, start=1):
        kind = str(primitive.kind)
        function = "R" if kind.endswith("_dist") else function_by_kind.get(kind)
        if function is None:
            raise ValueError(f"unsupported ORACLE primitive in B-prime audit: {kind}")
        identifier = f"PICP{index:04d}"
        name = f"PIC{index:03d}"
        converted = GICPrimitive(
            identifier=identifier,
            name=name,
            family=kind.upper(),
            function=function,
            atoms=tuple(int(atom) + 1 for atom in primitive.atoms),
            mode=int(getattr(primitive, "mode", 0)),
        )
        primitives.append(converted)
        gics.append(
            FrozenGIC(
                identifier=f"PICG{index:04d}",
                name=name,
                family=kind.upper(),
                irrep="A",
                primitive_id=identifier,
                gaussian_expression=converted.gaussian_expression(),
                coefficients=((identifier, 1.0),),
            )
        )
    return GICDefinition(
        backend=template.backend,
        point_group="C1",
        symmetrize=False,
        target_rank=len(gics),
        rank=len(gics),
        candidate_count=len(gics),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in source.coordinates_angstrom
        ),
        primitives=tuple(primitives),
        gics=tuple(gics),
        primitive_source="ORACLE_PIC_CONTRACT",
        primitive_source_schema=source.primitive_schema,
        primitive_b_matrix_sha256=source.primitive_b_sha256,
    )


def _coordinate_curvature_hessian_bohr(
    b_matrix: np.ndarray,
    b_derivative_tensor_per_angstrom2: np.ndarray,
    nonbonded_gradient_bohr: np.ndarray,
) -> np.ndarray:
    """Return the analytic ``sum_i g_i B'_i`` term for a frozen coordinate basis."""

    b = np.asarray(b_matrix, dtype=float)
    derivative = np.asarray(b_derivative_tensor_per_angstrom2, dtype=float)
    if derivative.shape != (b.shape[0], b.shape[1], b.shape[1]):
        raise ValueError("B-prime tensor does not match the coordinate basis")
    nonbonded_gradient_angstrom = (
        np.asarray(nonbonded_gradient_bohr, dtype=float) / BOHR_TO_ANGSTROM
    )
    internal_gradient, *_ = np.linalg.lstsq(
        b.T,
        -nonbonded_gradient_angstrom,
        rcond=1.0e-10,
    )
    curvature_angstrom2 = np.einsum(
        "i,iab->ab",
        internal_gradient,
        derivative,
        optimize=True,
    )
    curvature_angstrom2 = 0.5 * (curvature_angstrom2 + curvature_angstrom2.T)
    return curvature_angstrom2 * BOHR_TO_ANGSTROM**2


def _mode_aware_fit_system(
    b_matrix: np.ndarray,
    target_hessian_bohr: np.ndarray,
    cartesian_vibrational_basis: np.ndarray,
    frequency_audit: _FrequencyAuditContext,
) -> tuple[list[np.ndarray], dict[tuple[int, int], np.ndarray], np.ndarray]:
    """Build an equal-block Cartesian/modal least-squares system.

    The Cartesian block preserves force-constant information.  The second block
    expresses the same Hessian in the reference mass-weighted normal-mode basis,
    normalizes diagonal curvature and inter-mode mixing separately, and gives
    low, fingerprint and stretching regions priorities 4:2:1.  Consequently a
    small absolute error in a soft mode cannot be hidden by a large rigid-mode
    force constant.
    """

    b = np.asarray(b_matrix, dtype=float)
    target = np.asarray(target_hessian_bohr, dtype=float)
    cartesian_basis = np.asarray(cartesian_vibrational_basis, dtype=float)
    projected_b = b @ cartesian_basis
    cartesian_target = (cartesian_basis.T @ target @ cartesian_basis).reshape(-1)

    inverse_sqrt = frequency_audit.inverse_sqrt_cartesian_masses
    modal_b = (
        (b * inverse_sqrt[None, :])
        @ frequency_audit.mass_weighted_vibrational_basis
        @ frequency_audit.reference_modes
    )
    weighted_target = target * inverse_sqrt[:, None] * inverse_sqrt[None, :]
    reduced_target = (
        frequency_audit.mass_weighted_vibrational_basis.T
        @ weighted_target
        @ frequency_audit.mass_weighted_vibrational_basis
    )
    modal_target_matrix = (
        frequency_audit.reference_modes.T @ reduced_target @ frequency_audit.reference_modes
    )
    modal_weights = _modal_fit_weight_matrix(frequency_audit)
    modal_target = (modal_weights * modal_target_matrix).reshape(-1)

    cartesian_scale = 1.0 / max(np.linalg.norm(cartesian_target), 1.0e-30)
    modal_scale = 1.0 / max(np.linalg.norm(modal_target), 1.0e-30)
    block_scale = math.sqrt(0.5)
    target_vector = np.concatenate(
        (
            block_scale * cartesian_scale * cartesian_target,
            block_scale * modal_scale * modal_target,
        )
    )

    def combined_column(left: int, right: int) -> np.ndarray:
        if left == right:
            cartesian = BOHR_TO_ANGSTROM**2 * np.outer(projected_b[left], projected_b[left])
            modal = BOHR_TO_ANGSTROM**2 * np.outer(modal_b[left], modal_b[left])
        else:
            cartesian = BOHR_TO_ANGSTROM**2 * (
                np.outer(projected_b[left], projected_b[right])
                + np.outer(projected_b[right], projected_b[left])
            )
            modal = BOHR_TO_ANGSTROM**2 * (
                np.outer(modal_b[left], modal_b[right]) + np.outer(modal_b[right], modal_b[left])
            )
        return np.concatenate(
            (
                block_scale * cartesian_scale * cartesian.reshape(-1),
                block_scale * modal_scale * (modal_weights * modal).reshape(-1),
            )
        )

    diagonal_columns = [combined_column(index, index) for index in range(len(b))]
    pair_columns = {
        (left, right): combined_column(left, right)
        for left in range(len(b))
        for right in range(left + 1, len(b))
    }
    return diagonal_columns, pair_columns, target_vector


def _mode_aware_hessian_vector(
    hessian_bohr: np.ndarray,
    target_hessian_bohr: np.ndarray,
    cartesian_vibrational_basis: np.ndarray,
    frequency_audit: _FrequencyAuditContext,
) -> np.ndarray:
    """Project any Cartesian Hessian into the same joint fit metric as the target."""

    hessian = np.asarray(hessian_bohr, dtype=float)
    target = np.asarray(target_hessian_bohr, dtype=float)
    cartesian_basis = np.asarray(cartesian_vibrational_basis, dtype=float)
    inverse_sqrt = frequency_audit.inverse_sqrt_cartesian_masses
    modal_basis = frequency_audit.mass_weighted_vibrational_basis @ frequency_audit.reference_modes
    modal_weights = _modal_fit_weight_matrix(frequency_audit)

    def blocks(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cartesian = cartesian_basis.T @ matrix @ cartesian_basis
        weighted = matrix * inverse_sqrt[:, None] * inverse_sqrt[None, :]
        modal = modal_basis.T @ weighted @ modal_basis
        return cartesian, modal_weights * modal

    target_cartesian, target_modal = blocks(target)
    cartesian, modal = blocks(hessian)
    cartesian_scale = 1.0 / max(np.linalg.norm(target_cartesian), 1.0e-30)
    modal_scale = 1.0 / max(np.linalg.norm(target_modal), 1.0e-30)
    block_scale = math.sqrt(0.5)
    return np.concatenate(
        (
            block_scale * cartesian_scale * cartesian.reshape(-1),
            block_scale * modal_scale * modal.reshape(-1),
        )
    )


def _modal_fit_weight_matrix(context: _FrequencyAuditContext) -> np.ndarray:
    eigenvalues = np.asarray(context.reference_eigenvalues, dtype=float)
    frequencies = np.abs(np.asarray(context.reference_frequencies_cm, dtype=float))
    priority = np.where(
        frequencies < 500.0,
        MODAL_REGION_PRIORITIES["low_frequency"],
        np.where(
            frequencies < 2000.0,
            MODAL_REGION_PRIORITIES["fingerprint"],
            MODAL_REGION_PRIORITIES["stretching"],
        ),
    )
    floor = (MODAL_CURVATURE_FLOOR_CM / HESSIAN_EIGENVALUE_TO_CM) ** 2
    degenerate_group = np.full(eigenvalues.size, -1, dtype=int)
    for group, indices in enumerate(_near_degenerate_mode_blocks(frequencies)):
        degenerate_group[indices] = group
    weights = np.empty((eigenvalues.size, eigenvalues.size), dtype=float)
    for left in range(eigenvalues.size):
        for right in range(eigenvalues.size):
            left_scale = max(abs(eigenvalues[left]), floor)
            right_scale = max(abs(eigenvalues[right]), floor)
            if left == right:
                denominator = left_scale
            elif degenerate_group[left] >= 0 and degenerate_group[left] == degenerate_group[right]:
                denominator = math.sqrt(left_scale * right_scale)
            else:
                denominator = max(
                    abs(eigenvalues[left] - eigenvalues[right]),
                    0.05 * math.sqrt(left_scale * right_scale),
                    floor,
                )
            weights[left, right] = math.sqrt(priority[left] * priority[right]) / denominator
    return weights


def _coordinate_candidates(
    basis_name: str,
    b_matrix: np.ndarray,
    source: _CompilationSource,
    vibrational_basis: np.ndarray,
    reference_eigenvalues: np.ndarray,
    *,
    coordinate_labels: Sequence[str],
    minimum_error_target: float,
    parallel_workers: int,
    selection_policy: str,
    protected_mixing_threshold: float,
    protected_second_order_threshold: float,
    protected_gap_floor_fraction: float,
    frequency_audit: _FrequencyAuditContext,
    coordinate_curvature_bohr: np.ndarray,
) -> tuple[_ZoomCandidate, ...]:
    if selection_policy == EXACT_TERM_POLICY:
        return _coordinate_candidates_exact(
            basis_name,
            b_matrix,
            source,
            vibrational_basis,
            reference_eigenvalues,
            coordinate_labels=coordinate_labels,
            minimum_error_target=minimum_error_target,
            parallel_workers=parallel_workers,
            gap_floor_fraction=protected_gap_floor_fraction,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvature_bohr,
        )
    if selection_policy == REFITTED_POLICY:
        return _coordinate_candidates_refitted(
            basis_name,
            b_matrix,
            source,
            vibrational_basis,
            reference_eigenvalues,
            coordinate_labels=coordinate_labels,
            minimum_error_target=minimum_error_target,
            parallel_workers=parallel_workers,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvature_bohr,
        )
    if selection_policy == HYBRID_PROTECTED_POLICY:
        return _coordinate_candidates_hybrid(
            basis_name,
            b_matrix,
            source,
            vibrational_basis,
            reference_eigenvalues,
            coordinate_labels=coordinate_labels,
            minimum_error_target=minimum_error_target,
            parallel_workers=parallel_workers,
            protected_mixing_threshold=protected_mixing_threshold,
            protected_second_order_threshold=protected_second_order_threshold,
            protected_gap_floor_fraction=protected_gap_floor_fraction,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvature_bohr,
        )
    raise ValueError(f"unsupported ARCHITECT coupling policy: {selection_policy}")


def _canonical_internal_hessian(
    b_matrix: np.ndarray,
    residual_hessian_bohr: np.ndarray,
) -> np.ndarray:
    """Return the symmetric minimum-norm internal Hessian in a declared PIC gauge."""

    inverse = np.linalg.pinv(np.asarray(b_matrix, dtype=float), rcond=1.0e-10)
    residual_angstrom = np.asarray(residual_hessian_bohr, dtype=float) / BOHR_TO_ANGSTROM**2
    internal = inverse.T @ residual_angstrom @ inverse
    return 0.5 * (internal + internal.T)


def _coordinate_candidates_exact(
    basis_name: str,
    b_matrix: np.ndarray,
    source: _CompilationSource,
    vibrational_basis: np.ndarray,
    reference_eigenvalues: np.ndarray,
    *,
    coordinate_labels: Sequence[str],
    minimum_error_target: float,
    parallel_workers: int,
    gap_floor_fraction: float,
    frequency_audit: _FrequencyAuditContext,
    coordinate_curvature_bohr: np.ndarray,
) -> tuple[_ZoomCandidate, ...]:
    """Build a physical truncation path without fitting any retained QM term."""

    b = np.asarray(b_matrix, dtype=float)
    internal = _canonical_internal_hessian(
        b,
        source.residual_hessian_bohr - coordinate_curvature_bohr,
    )
    pairs = list(
        (
            (
                abs(float(internal[left, right]))
                / math.sqrt(
                    max(abs(float(internal[left, left] * internal[right, right])), 1.0e-30)
                ),
                abs(float(internal[left, right])),
                left,
                right,
            )
            for left in range(len(internal))
            for right in range(left + 1, len(internal))
            if abs(float(internal[left, right])) > 1.0e-14
        ),
    )
    diagonal = np.diag(internal)
    importance = {
        (pair[2], pair[3]): _vpt2_coupling_indices(
            diagonal,
            float(internal[pair[2], pair[3]]),
            pair[2],
            pair[3],
            gap_floor_fraction,
        )
        for pair in pairs
    }
    pairs.sort(
        key=lambda pair: (
            importance[(pair[2], pair[3])][1],
            importance[(pair[2], pair[3])][0],
            pair[0],
            pair[1],
        ),
        reverse=True,
    )
    diagonal_columns, all_pair_columns, fit_target = _mode_aware_fit_system(
        b,
        source.residual_hessian_bohr - coordinate_curvature_bohr,
        vibrational_basis,
        frequency_audit,
    )
    pair_columns = {
        (left, right): all_pair_columns[(left, right)]
        for _relative, _absolute, left, right in pairs
    }
    working = np.diag(np.diag(internal))
    selected_pairs: list[tuple[float, float, int, int]] = []
    selection_reductions: list[float] = []
    candidates = [
        _candidate(
            basis_name,
            b,
            working,
            (),
            source.nonbonded_hessian_bohr,
            vibrational_basis,
            reference_eigenvalues,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvature_bohr,
            selection_policy=EXACT_TERM_POLICY,
        )
    ]
    fit_residual = fit_target - sum(
        (float(internal[index, index]) * diagonal_columns[index] for index in range(len(b))),
        np.zeros_like(fit_target),
    )
    remaining = list(pairs)
    incremental_limit = min(
        len(remaining),
        MAX_INCREMENTAL_COUPLINGS_PER_COORDINATE * len(b),
    )
    workers = _resolved_parallel_workers(parallel_workers)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while (
            remaining
            and len(selected_pairs) < incremental_limit
            and not _candidate_satisfies_target(candidates[-1], minimum_error_target)
        ):
            previous_error = candidates[-1].relative_l2_error
            selected_score = _best_remaining_exact_pair(
                remaining,
                pair_columns,
                internal,
                fit_residual,
                executor,
                workers,
            )
            _neg_gain, _neg_relative, _neg_absolute, left, right, selected_index = selected_score
            selected_pair = remaining.pop(selected_index)
            selected_pairs.append(selected_pair)
            value = float(internal[left, right])
            working[left, right] = working[right, left] = value
            fit_residual = fit_residual - value * pair_columns[(left, right)]
            trial = _candidate(
                basis_name,
                b,
                working,
                (),
                source.nonbonded_hessian_bohr,
                vibrational_basis,
                reference_eigenvalues,
                frequency_audit=frequency_audit,
                coordinate_curvature_bohr=coordinate_curvature_bohr,
                selection_policy=EXACT_TERM_POLICY,
            )
            selection_reductions.append(float(previous_error - trial.relative_l2_error))
            candidates.append(
                replace(
                    trial,
                    retained_couplings=_exact_coupling_records(
                        basis_name,
                        source,
                        selected_pairs,
                        internal,
                        selection_reductions,
                        coordinate_labels,
                    ),
                    retained_coupling_count=len(selected_pairs),
                    parameter_count=len(b) + len(selected_pairs),
                )
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if (
        remaining
        and len(selected_pairs) >= incremental_limit
        and not _candidate_satisfies_target(candidates[-1], minimum_error_target)
    ):
        candidates[-1] = replace(
            candidates[-1],
            fit_diagnostics={
                **dict(candidates[-1].fit_diagnostics),
                "sparse_support_ceiling_reached": True,
                "maximum_incremental_couplings": incremental_limit,
                "interpretation": "coordinate chart is not sparse under the joint modal contract",
            },
        )
    if remaining:
        complete_pairs = selected_pairs + remaining
        candidates.append(
            _candidate(
                basis_name,
                b,
                internal,
                _exact_coupling_records(
                    basis_name,
                    source,
                    complete_pairs,
                    internal,
                    selection_reductions + [0.0] * len(remaining),
                    coordinate_labels,
                ),
                source.nonbonded_hessian_bohr,
                vibrational_basis,
                reference_eigenvalues,
                frequency_audit=frequency_audit,
                coordinate_curvature_bohr=coordinate_curvature_bohr,
                selection_policy=EXACT_TERM_POLICY,
            )
        )
    return tuple(candidates)


def _coordinate_candidates_refitted(
    basis_name: str,
    b_matrix: np.ndarray,
    source: _CompilationSource,
    vibrational_basis: np.ndarray,
    reference_eigenvalues: np.ndarray,
    *,
    coordinate_labels: Sequence[str],
    minimum_error_target: float,
    parallel_workers: int,
    frequency_audit: _FrequencyAuditContext,
    coordinate_curvature_bohr: np.ndarray,
) -> tuple[_ZoomCandidate, ...]:
    b = np.asarray(b_matrix, dtype=float)
    internal = _canonical_internal_hessian(
        b,
        source.residual_hessian_bohr - coordinate_curvature_bohr,
    )
    diagonal_columns, all_pair_columns, fit_target = _mode_aware_fit_system(
        b,
        source.residual_hessian_bohr - coordinate_curvature_bohr,
        vibrational_basis,
        frequency_audit,
    )
    pairs = sorted(
        (
            (
                abs(float(internal[left, right]))
                / math.sqrt(
                    max(abs(float(internal[left, left] * internal[right, right])), 1.0e-30)
                ),
                abs(float(internal[left, right])),
                left,
                right,
            )
            for left in range(len(internal))
            for right in range(left + 1, len(internal))
            if abs(float(internal[left, right])) > 1.0e-14
        ),
        reverse=True,
    )
    pair_columns = {
        (left, right): all_pair_columns[(left, right)]
        for _relative, _absolute, left, right in pairs
    }

    selected_pairs: list[tuple[float, float, int, int]] = []
    selection_reductions: list[float] = []
    incremental_limit = min(
        len(pairs),
        MAX_INCREMENTAL_COUPLINGS_PER_COORDINATE * len(b),
    )
    fit_state = _IncrementalQRFit(
        fit_target,
        maximum_columns=len(diagonal_columns) + incremental_limit,
    )
    for column in diagonal_columns:
        fit_state.add_column(column)
    fitted_values = fit_state.coefficients
    working = _internal_from_support_values(len(b), selected_pairs, fitted_values)
    projected_residual = fit_state.residual
    fit_diagnostics = fit_state.diagnostics()
    candidates = [
        _candidate(
            basis_name,
            b,
            working,
            (),
            source.nonbonded_hessian_bohr,
            vibrational_basis,
            reference_eigenvalues,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvature_bohr,
            selection_policy=REFITTED_POLICY,
            fit_diagnostics=fit_diagnostics,
        )
    ]
    remaining = list(pairs)
    workers = _resolved_parallel_workers(parallel_workers)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while (
            remaining
            and len(selected_pairs) < incremental_limit
            and not _candidate_satisfies_target(candidates[-1], minimum_error_target)
        ):
            previous_error = candidates[-1].relative_l2_error
            selected_score = _best_remaining_pair(
                remaining,
                pair_columns,
                projected_residual,
                executor,
                workers,
            )
            _neg_score, _neg_relative, _neg_absolute, left, right, selected_index = selected_score
            selected_pairs.append(remaining.pop(selected_index))
            fit_state.add_column(pair_columns[(left, right)])
            fitted_values = fit_state.coefficients
            working = _internal_from_support_values(
                len(b),
                selected_pairs,
                fitted_values,
            )
            projected_residual = fit_state.residual
            fit_diagnostics = fit_state.diagnostics()
            trial = _candidate(
                basis_name,
                b,
                working,
                _refitted_coupling_records(
                    basis_name,
                    source,
                    selected_pairs,
                    fitted_values,
                    selection_reductions + [0.0],
                    coordinate_labels,
                ),
                source.nonbonded_hessian_bohr,
                vibrational_basis,
                reference_eigenvalues,
                frequency_audit=frequency_audit,
                coordinate_curvature_bohr=coordinate_curvature_bohr,
                selection_policy=REFITTED_POLICY,
                fit_diagnostics=fit_diagnostics,
            )
            selection_reductions.append(float(previous_error - trial.relative_l2_error))
            trial = replace(
                trial,
                retained_couplings=_refitted_coupling_records(
                    basis_name,
                    source,
                    selected_pairs,
                    fitted_values,
                    selection_reductions,
                    coordinate_labels,
                ),
            )
            candidates.append(trial)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if (
        remaining
        and len(selected_pairs) >= incremental_limit
        and not _candidate_satisfies_target(candidates[-1], minimum_error_target)
    ):
        candidates[-1] = replace(
            candidates[-1],
            fit_diagnostics={
                **dict(candidates[-1].fit_diagnostics),
                "sparse_support_ceiling_reached": True,
                "maximum_incremental_couplings": incremental_limit,
                "interpretation": "coordinate chart is not sparse under the joint modal contract",
            },
        )
    if remaining:
        complete_pairs = selected_pairs + remaining
        complete_values = np.concatenate(
            (
                np.diag(internal),
                np.asarray([internal[left, right] for _r, _a, left, right in complete_pairs]),
            )
        )
        candidates.append(
            _candidate(
                basis_name,
                b,
                internal,
                _refitted_coupling_records(
                    basis_name,
                    source,
                    complete_pairs,
                    complete_values,
                    selection_reductions + [0.0] * len(remaining),
                    coordinate_labels,
                ),
                source.nonbonded_hessian_bohr,
                vibrational_basis,
                reference_eigenvalues,
                frequency_audit=frequency_audit,
                coordinate_curvature_bohr=coordinate_curvature_bohr,
                selection_policy=REFITTED_POLICY,
            )
        )
    return tuple(candidates)


def _coordinate_candidates_hybrid(
    basis_name: str,
    b_matrix: np.ndarray,
    source: _CompilationSource,
    vibrational_basis: np.ndarray,
    reference_eigenvalues: np.ndarray,
    *,
    coordinate_labels: Sequence[str],
    minimum_error_target: float,
    parallel_workers: int,
    protected_mixing_threshold: float,
    protected_second_order_threshold: float,
    protected_gap_floor_fraction: float,
    frequency_audit: _FrequencyAuditContext,
    coordinate_curvature_bohr: np.ndarray,
) -> tuple[_ZoomCandidate, ...]:
    """Protect physical QM terms and fit only a compressed small-coupling tail."""

    b = np.asarray(b_matrix, dtype=float)
    internal = _canonical_internal_hessian(
        b,
        source.residual_hessian_bohr - coordinate_curvature_bohr,
    )
    diagonal_columns, all_pair_columns, target_vector = _mode_aware_fit_system(
        b,
        source.residual_hessian_bohr - coordinate_curvature_bohr,
        vibrational_basis,
        frequency_audit,
    )
    pairs = sorted(
        (
            (
                abs(float(internal[left, right]))
                / math.sqrt(
                    max(abs(float(internal[left, left] * internal[right, right])), 1.0e-30)
                ),
                abs(float(internal[left, right])),
                left,
                right,
            )
            for left in range(len(internal))
            for right in range(left + 1, len(internal))
            if abs(float(internal[left, right])) > 1.0e-14
        ),
        reverse=True,
    )
    pair_columns = {
        (left, right): all_pair_columns[(left, right)]
        for _relative, _absolute, left, right in pairs
    }
    protected_pairs = []
    remaining = []
    diagonal = np.diag(internal)
    for pair in pairs:
        _relative, _absolute, left, right = pair
        mixing_index, relative_second_order_shift = _vpt2_coupling_indices(
            diagonal,
            float(internal[left, right]),
            left,
            right,
            protected_gap_floor_fraction,
        )
        if (
            mixing_index >= protected_mixing_threshold
            or relative_second_order_shift >= protected_second_order_threshold
        ):
            protected_pairs.append(pair)
        else:
            remaining.append(pair)

    fixed = np.diag(np.diag(internal))
    fixed_vector = sum(
        (float(internal[index, index]) * diagonal_columns[index] for index in range(len(b))),
        np.zeros_like(target_vector),
    )
    for _relative, _absolute, left, right in protected_pairs:
        fixed[left, right] = fixed[right, left] = float(internal[left, right])
        fixed_vector = fixed_vector + float(internal[left, right]) * pair_columns[(left, right)]
    tail_target = target_vector - fixed_vector
    incremental_limit = min(
        len(remaining),
        MAX_INCREMENTAL_COUPLINGS_PER_COORDINATE * len(b),
    )
    fit_state = _IncrementalQRFit(
        tail_target,
        maximum_columns=incremental_limit,
    )
    tail_residual = fit_state.residual
    selected_tail: list[tuple[float, float, int, int]] = []
    tail_values = np.asarray([], dtype=float)
    selection_reductions: list[float] = []
    candidates = [
        _candidate(
            basis_name,
            b,
            fixed,
            _hybrid_coupling_records(
                basis_name,
                source,
                protected_pairs,
                selected_tail,
                internal,
                tail_values,
                selection_reductions,
                coordinate_labels,
                protected_gap_floor_fraction,
            ),
            source.nonbonded_hessian_bohr,
            vibrational_basis,
            reference_eigenvalues,
            frequency_audit=frequency_audit,
            coordinate_curvature_bohr=coordinate_curvature_bohr,
            selection_policy=HYBRID_PROTECTED_POLICY,
        )
    ]
    workers = _resolved_parallel_workers(parallel_workers)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while (
            remaining
            and len(selected_tail) < incremental_limit
            and not _candidate_satisfies_target(candidates[-1], minimum_error_target)
        ):
            previous_error = candidates[-1].relative_l2_error
            selected_score = _best_remaining_pair(
                remaining,
                pair_columns,
                tail_residual,
                executor,
                workers,
            )
            _neg_score, _neg_relative, _neg_absolute, _left, _right, selected_index = selected_score
            selected_pair = remaining.pop(selected_index)
            selected_tail.append(selected_pair)
            fit_state.add_column(pair_columns[(selected_pair[2], selected_pair[3])])
            tail_values = fit_state.coefficients
            working = _tail_internal_from_values(
                fixed,
                selected_tail,
                tail_values,
            )
            tail_residual = fit_state.residual
            fit_diagnostics = fit_state.diagnostics()
            trial = _candidate(
                basis_name,
                b,
                working,
                (),
                source.nonbonded_hessian_bohr,
                vibrational_basis,
                reference_eigenvalues,
                frequency_audit=frequency_audit,
                coordinate_curvature_bohr=coordinate_curvature_bohr,
                selection_policy=HYBRID_PROTECTED_POLICY,
                fit_diagnostics=fit_diagnostics,
            )
            selection_reductions.append(float(previous_error - trial.relative_l2_error))
            candidates.append(
                replace(
                    trial,
                    retained_couplings=_hybrid_coupling_records(
                        basis_name,
                        source,
                        protected_pairs,
                        selected_tail,
                        internal,
                        tail_values,
                        selection_reductions,
                        coordinate_labels,
                        protected_gap_floor_fraction,
                    ),
                    retained_coupling_count=len(protected_pairs) + len(selected_tail),
                    parameter_count=len(b) + len(protected_pairs) + len(selected_tail),
                )
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if (
        remaining
        and len(selected_tail) >= incremental_limit
        and not _candidate_satisfies_target(candidates[-1], minimum_error_target)
    ):
        candidates[-1] = replace(
            candidates[-1],
            fit_diagnostics={
                **dict(candidates[-1].fit_diagnostics),
                "sparse_support_ceiling_reached": True,
                "maximum_incremental_couplings": incremental_limit,
                "interpretation": "coordinate chart is not sparse under the joint modal contract",
            },
        )
    if remaining:
        complete_pairs = protected_pairs + selected_tail + remaining
        candidates.append(
            _candidate(
                basis_name,
                b,
                internal,
                _exact_coupling_records(
                    basis_name,
                    source,
                    complete_pairs,
                    internal,
                    [0.0] * len(complete_pairs),
                    coordinate_labels,
                ),
                source.nonbonded_hessian_bohr,
                vibrational_basis,
                reference_eigenvalues,
                frequency_audit=frequency_audit,
                coordinate_curvature_bohr=coordinate_curvature_bohr,
                selection_policy=HYBRID_PROTECTED_POLICY,
            )
        )
    return tuple(candidates)


def _internal_from_support_values(
    coordinate_count: int,
    selected_pairs: Sequence[tuple[float, float, int, int]],
    fitted_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fitted_values, dtype=float)
    expected = coordinate_count + len(selected_pairs)
    if values.shape != (expected,):
        raise ValueError("fitted harmonic support has an inconsistent coefficient count")
    internal = np.diag(values[:coordinate_count])
    for value, (_relative, _absolute, left, right) in zip(
        values[coordinate_count:],
        selected_pairs,
        strict=True,
    ):
        internal[left, right] = internal[right, left] = float(value)
    return internal




def _tail_internal_from_values(
    fixed_internal: np.ndarray,
    selected_pairs: Sequence[tuple[float, float, int, int]],
    fitted_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fitted_values, dtype=float)
    if values.shape != (len(selected_pairs),):
        raise ValueError("fitted weak-tail support has an inconsistent coefficient count")
    internal = np.asarray(fixed_internal, dtype=float).copy()
    for value, (_relative, _absolute, left, right) in zip(
        values,
        selected_pairs,
        strict=True,
    ):
        internal[left, right] = internal[right, left] = float(value)
    return internal


def _best_remaining_pair(
    remaining: Sequence[tuple[float, float, int, int]],
    pair_columns: Mapping[tuple[int, int], np.ndarray],
    projected_residual: np.ndarray,
    executor: ThreadPoolExecutor | None,
    workers: int,
) -> tuple[float, float, float, int, int, int]:
    if executor is None or len(remaining) < 2 * workers:
        return _score_pair_range(
            remaining,
            pair_columns,
            projected_residual,
            0,
            len(remaining),
        )
    chunk = max(1, math.ceil(len(remaining) / workers))
    futures = [
        executor.submit(
            _score_pair_range,
            remaining,
            pair_columns,
            projected_residual,
            start,
            min(start + chunk, len(remaining)),
        )
        for start in range(0, len(remaining), chunk)
    ]
    return min(future.result() for future in futures)


def _best_remaining_exact_pair(
    remaining: Sequence[tuple[float, float, int, int]],
    pair_columns: Mapping[tuple[int, int], np.ndarray],
    internal_hessian: np.ndarray,
    fit_residual: np.ndarray,
    executor: ThreadPoolExecutor | None,
    workers: int,
) -> tuple[float, float, float, int, int, int]:
    if executor is None or len(remaining) < 2 * workers:
        return _score_exact_pair_range(
            remaining,
            pair_columns,
            internal_hessian,
            fit_residual,
            0,
            len(remaining),
        )
    chunk = max(1, math.ceil(len(remaining) / workers))
    futures = [
        executor.submit(
            _score_exact_pair_range,
            remaining,
            pair_columns,
            internal_hessian,
            fit_residual,
            start,
            min(start + chunk, len(remaining)),
        )
        for start in range(0, len(remaining), chunk)
    ]
    return min(future.result() for future in futures)


def _score_exact_pair_range(
    remaining: Sequence[tuple[float, float, int, int]],
    pair_columns: Mapping[tuple[int, int], np.ndarray],
    internal_hessian: np.ndarray,
    fit_residual: np.ndarray,
    start: int,
    stop: int,
) -> tuple[float, float, float, int, int, int]:
    residual_norm2 = float(fit_residual @ fit_residual)
    scores = []
    for index in range(start, stop):
        relative, absolute, left, right = remaining[index]
        contribution = float(internal_hessian[left, right]) * pair_columns[(left, right)]
        trial = fit_residual - contribution
        gain = residual_norm2 - float(trial @ trial)
        scores.append((-gain, -relative, -absolute, left, right, index))
    return min(scores)


def _score_pair_range(
    remaining: Sequence[tuple[float, float, int, int]],
    pair_columns: Mapping[tuple[int, int], np.ndarray],
    projected_residual: np.ndarray,
    start: int,
    stop: int,
) -> tuple[float, float, float, int, int, int]:
    scores = []
    for index in range(start, stop):
        relative, absolute, left, right = remaining[index]
        column = pair_columns[(left, right)]
        score = abs(float(column @ projected_residual)) / max(
            float(np.linalg.norm(column)),
            1.0e-30,
        )
        scores.append((-score, -relative, -absolute, left, right, index))
    return min(scores)


def _resolved_parallel_workers(requested: int) -> int:
    value = int(requested)
    if value < 0:
        raise ValueError("ARCHITECT parallel worker count must be non-negative")
    return value or max(1, min(int(os.cpu_count() or 1), 32))




def _refitted_coupling_records(
    basis_name: str,
    source: _CompilationSource,
    selected_pairs: Sequence[tuple[float, float, int, int]],
    fitted_values: np.ndarray,
    selection_reductions: Sequence[float],
    coordinate_labels: Sequence[str],
) -> tuple[_SelectedCoupling, ...]:
    coordinate_count = len(fitted_values) - len(selected_pairs)
    diagonal = fitted_values[:coordinate_count]
    records = []
    for offset, ((_relative, _absolute, left, right), reduction) in enumerate(
        zip(selected_pairs, selection_reductions, strict=True),
        start=coordinate_count,
    ):
        value = float(fitted_values[offset])
        normalized = abs(value) / math.sqrt(
            max(abs(float(diagonal[left] * diagonal[right])), 1.0e-30)
        )
        relation = _coupling_relation(
            basis_name,
            source,
            left,
            right,
            coordinate_labels,
        )
        records.append(
            _SelectedCoupling(
                left,
                right,
                str(coordinate_labels[left]),
                str(coordinate_labels[right]),
                value,
                normalized,
                float(reduction),
                relation,
                "NUMERICAL_REFIT",
            )
        )
    return tuple(records)


def _exact_coupling_records(
    basis_name: str,
    source: _CompilationSource,
    selected_pairs: Sequence[tuple[float, float, int, int]],
    internal_hessian: np.ndarray,
    selection_reductions: Sequence[float],
    coordinate_labels: Sequence[str],
) -> tuple[_SelectedCoupling, ...]:
    diagonal = np.diag(internal_hessian)
    records = []
    for (_relative, _absolute, left, right), reduction in zip(
        selected_pairs,
        selection_reductions,
        strict=True,
    ):
        value = float(internal_hessian[left, right])
        normalized = abs(value) / math.sqrt(
            max(abs(float(diagonal[left] * diagonal[right])), 1.0e-30)
        )
        records.append(
            _SelectedCoupling(
                left,
                right,
                str(coordinate_labels[left]),
                str(coordinate_labels[right]),
                value,
                normalized,
                float(reduction),
                _coupling_relation(
                    basis_name,
                    source,
                    left,
                    right,
                    coordinate_labels,
                ),
                "EXACT_QM",
            )
        )
    return tuple(records)


def _hybrid_coupling_records(
    basis_name: str,
    source: _CompilationSource,
    protected_pairs: Sequence[tuple[float, float, int, int]],
    selected_tail: Sequence[tuple[float, float, int, int]],
    internal_hessian: np.ndarray,
    tail_values: np.ndarray,
    selection_reductions: Sequence[float],
    coordinate_labels: Sequence[str],
    gap_floor_fraction: float,
) -> tuple[_SelectedCoupling, ...]:
    diagonal = np.diag(internal_hessian)
    records = []
    for _relative, _absolute, left, right in protected_pairs:
        value = float(internal_hessian[left, right])
        mixing, shift = _vpt2_coupling_indices(
            diagonal,
            value,
            left,
            right,
            gap_floor_fraction,
        )
        records.append(
            _SelectedCoupling(
                left,
                right,
                str(coordinate_labels[left]),
                str(coordinate_labels[right]),
                value,
                abs(value) / math.sqrt(max(abs(float(diagonal[left] * diagonal[right])), 1.0e-30)),
                0.0,
                _coupling_relation(
                    basis_name,
                    source,
                    left,
                    right,
                    coordinate_labels,
                ),
                "EXACT_QM_PROTECTED",
                mixing,
                shift,
            )
        )
    for value, (_pair, reduction) in zip(
        tail_values,
        zip(selected_tail, selection_reductions, strict=True),
        strict=True,
    ):
        _relative, _absolute, left, right = _pair
        scalar = float(value)
        mixing, shift = _vpt2_coupling_indices(
            diagonal,
            float(internal_hessian[left, right]),
            left,
            right,
            gap_floor_fraction,
        )
        records.append(
            _SelectedCoupling(
                left,
                right,
                str(coordinate_labels[left]),
                str(coordinate_labels[right]),
                scalar,
                abs(scalar) / math.sqrt(max(abs(float(diagonal[left] * diagonal[right])), 1.0e-30)),
                float(reduction),
                _coupling_relation(
                    basis_name,
                    source,
                    left,
                    right,
                    coordinate_labels,
                ),
                "NUMERICAL_COMPRESSED_TAIL",
                mixing,
                shift,
            )
        )
    return tuple(records)


def _vpt2_coupling_indices(
    diagonal: np.ndarray,
    coupling: float,
    left: int,
    right: int,
    gap_floor_fraction: float,
) -> tuple[float, float]:
    finite_diagonal = np.abs(diagonal[np.abs(diagonal) > 1.0e-14])
    global_scale = float(np.median(finite_diagonal)) if finite_diagonal.size else 1.0
    local_scale = max(
        math.sqrt(abs(float(diagonal[left] * diagonal[right]))),
        1.0e-8 * global_scale,
    )
    gap = max(
        abs(float(diagonal[left] - diagonal[right])),
        gap_floor_fraction * local_scale,
        1.0e-30,
    )
    magnitude = abs(float(coupling))
    return magnitude / gap, magnitude * magnitude / (gap * local_scale)


def _coupling_relation(
    basis_name: str,
    source: _CompilationSource,
    left: int,
    right: int,
    coordinate_labels: Sequence[str],
) -> str:
    if basis_name == "PIC":
        return classify_primitive_coupling(source.primitives[left], source.primitives[right])
    if basis_name == "HYBRID_LOCAL":
        return _hybrid_coupling_relation(
            source,
            str(coordinate_labels[left]),
            str(coordinate_labels[right]),
        )
    return _local_sonic_coupling_relation(
        str(coordinate_labels[left]),
        str(coordinate_labels[right]),
    )


def _sonic_coordinate_label(definition: Any, gic: Any) -> str:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    terms = []
    for primitive_id, coefficient in gic.coefficients:
        primitive = primitive_by_id[primitive_id]
        terms.append(f"{float(coefficient):.6g}*{primitive.gaussian_expression()}")
    return f"{gic.name}:" + "+".join(terms)


def _pic_coordinate_label(primitive: Any, index: int) -> str:
    function = {
        "bond": "R",
        "angle": "A",
        "linear_bend": "L",
        "dihedral": "D",
        "out_of_plane": "U",
        "out_of_plane_height": "H",
    }.get(str(primitive.kind), str(primitive.kind).upper())
    atoms = ",".join(str(int(atom) + 1) for atom in primitive.atoms)
    return f"PIC{index:03d}:{function}({atoms})"


def _hybrid_coupling_relation(
    source: _CompilationSource,
    left_label: str,
    right_label: str,
) -> str:
    """Preserve native PIC chemistry and label mixed ring/PIC couplings."""

    def pic_index(label: str) -> int | None:
        head = str(label).split(":", 1)[0]
        if not head.startswith("PIC") or not head[3:].isdigit():
            return None
        return int(head[3:]) - 1

    left_pic = pic_index(left_label)
    right_pic = pic_index(right_label)
    if left_pic is not None and right_pic is not None:
        return classify_primitive_coupling(
            source.primitives[left_pic], source.primitives[right_pic]
        )
    if left_pic is None and right_pic is None:
        return _local_sonic_coupling_relation(left_label, right_label)
    special = left_label if left_pic is None else right_label
    special_family = _local_sonic_label_family(special)
    return f"HYBRID_NATIVE_PIC_{special_family}"


def _local_sonic_coupling_relation(left_label: str, right_label: str) -> str:
    """Give a human-readable local chemical class to a SONIC coupling."""

    left_family = _local_sonic_label_family(left_label)
    right_family = _local_sonic_label_family(right_label)
    ordered = sorted((left_family, right_family))
    return f"LOCAL_SONIC_{ordered[0]}_{ordered[1]}"


def _local_sonic_label_family(label: str) -> str:
    name = str(label).split(":", 1)[0].upper()
    prefixes = (
        (("STRE",), "STRETCH"),
        (("XANG", "BEND", "ANGL", "LINC"), "BEND"),
        (("OUPL", "OUT"), "OUT_OF_PLANE"),
        (("TORS", "DIHE", "ONEDIH"), "TORSION"),
        (("RING", "RDEF", "RPCK", "PUCK", "CREM", "CP"), "RING"),
        (("FRAG", "POSE", "ROTA", "TRAN"), "FRAGMENT_POSE"),
    )
    for candidates, value in prefixes:
        if name.startswith(candidates):
            return value
    return "COMPOSITE"


def _candidate(
    basis_name: str,
    b_matrix: np.ndarray,
    internal_hessian: np.ndarray,
    retained_couplings: tuple[_SelectedCoupling, ...],
    nonbonded_hessian_bohr: np.ndarray,
    vibrational_basis: np.ndarray,
    reference_eigenvalues: np.ndarray,
    *,
    frequency_audit: _FrequencyAuditContext,
    coordinate_curvature_bohr: np.ndarray,
    selection_policy: str,
    fit_diagnostics: Mapping[str, Any] | None = None,
) -> _ZoomCandidate:
    model = (
        (b_matrix.T @ internal_hessian @ b_matrix) * BOHR_TO_ANGSTROM**2
        + coordinate_curvature_bohr
        + nonbonded_hessian_bohr
    )
    eigenvalues = _projected_eigenvalues(model, vibrational_basis)
    difference = eigenvalues - reference_eigenvalues
    frequencies, modes = _gf_modes(model, frequency_audit)
    frequency_difference = frequencies - frequency_audit.reference_frequencies_cm
    region_metrics = _gf_region_metrics(
        frequency_audit.reference_frequencies_cm,
        frequency_difference,
    )
    (
        mode_overlap_matrix,
        mode_assignment,
        mode_overlap_metrics,
        degenerate_subspace_metrics,
    ) = _gf_mode_overlap_audit(frequencies, modes, frequency_audit)
    regional_curvature_metrics = _gf_regional_curvature_metrics(
        frequencies,
        mode_assignment,
        frequency_audit,
    )
    scale = max(float(np.max(np.abs(reference_eigenvalues))), np.finfo(float).eps)
    resolved = np.abs(reference_eigenvalues) > 1.0e-8 * scale
    return _ZoomCandidate(
        basis=basis_name,
        selection_policy=selection_policy,
        coordinate_count=int(b_matrix.shape[0]),
        retained_coupling_count=len(retained_couplings),
        parameter_count=int(b_matrix.shape[0] + len(retained_couplings)),
        hessian_bohr=0.5 * (model + model.T),
        relative_l2_error=float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference_eigenvalues), 1.0e-30)
        ),
        maximum_absolute_error=float(np.max(np.abs(difference))),
        sign_mismatch_count=int(
            np.sum(np.sign(eigenvalues[resolved]) != np.sign(reference_eigenvalues[resolved]))
        ),
        gf_frequency_rms_error_cm=float(np.sqrt(np.mean(frequency_difference**2))),
        gf_frequency_maximum_absolute_error_cm=float(np.max(np.abs(frequency_difference))),
        gf_imaginary_mode_mismatch_count=int(
            np.sum((frequencies < 0.0) != (frequency_audit.reference_frequencies_cm < 0.0))
        ),
        gf_region_metrics=region_metrics,
        gf_regional_curvature_metrics=regional_curvature_metrics,
        gf_mode_overlap_metrics=mode_overlap_metrics,
        gf_mode_overlap_matrix=mode_overlap_matrix,
        gf_mode_assignment=mode_assignment,
        gf_degenerate_subspace_metrics=degenerate_subspace_metrics,
        retained_couplings=retained_couplings,
        fit_diagnostics={} if fit_diagnostics is None else dict(fit_diagnostics),
    )


def _compare_coordinate_families(
    families: Mapping[str, tuple[_ZoomCandidate, ...]],
    targets: Sequence[float],
) -> dict[str, Any]:
    target_rows = []
    wins = {name: 0 for name in families}
    parameter_sums = {name: 0 for name in families}
    for target in targets:
        row: dict[str, Any] = {"relative_l2_error_target": float(target)}
        available = {}
        for name, candidates in families.items():
            selected = _minimum_parameter_candidate(candidates, float(target))
            if selected is None:
                row[name] = None
                continue
            record = _candidate_record(selected, include_overlap_matrix=True)
            row[name] = record
            available[name] = selected.parameter_count
            parameter_sums[name] += selected.parameter_count
        if available:
            minimum = min(available.values())
            winners = sorted(name for name, count in available.items() if count == minimum)
            row["winner"] = "+".join(winners)
            for name in winners:
                wins[name] += 1
        else:
            row["winner"] = "NONE"
        target_rows.append(row)
    strictest = target_rows[-1]
    strict_counts = {
        name: int(strictest[name]["parameter_count"])
        for name in families
        if strictest.get(name) is not None
    }
    recommended = min(
        families,
        key=lambda name: (
            strict_counts.get(name, math.inf),
            -wins[name],
            parameter_sums[name],
            name,
        ),
    )
    return {
        "selection_policy": next(iter(families.values()))[0].selection_policy,
        "criterion": (
            "minimum independent harmonic scalars along a joint Cartesian/modal "
            "selection path at equal projected-Cartesian eigenvalue error and "
            "satisfying regional reference-mode curvature and mode-shape contracts; modal weights "
            "prevent the residual from concentrating in soft modes"
        ),
        "sonic_policy": "LOCAL_BLOCK_SALC_ONLY_NO_GLOBAL_POINT_GROUP_PROJECTION",
        "hybrid_policy": (
            "SMITH_RING_DEFORMATION_AND_PUCKERING_BLOCKS_PLUS_UNREPLACED_ORACLE_PICS"
        ),
        "recommended_basis": recommended,
        "wins_by_basis": wins,
        "summed_parameter_count_by_basis": parameter_sums,
        "targets": target_rows,
        "frontiers": {
            name: [_candidate_record(item) for item in _pareto_frontier(candidates)]
            for name, candidates in families.items()
        },
        "coupling_audit": {
            name: _basis_coupling_audit(candidates) for name, candidates in families.items()
        },
    }


def _basis_coupling_audit(candidates: Sequence[_ZoomCandidate]) -> dict[str, Any]:
    """Summarize which retained couplings buy accuracy and why they are readable."""

    diagonal = candidates[0]
    rows = []
    for target in (0.05, 0.02, 0.01):
        selected = _minimum_parameter_candidate(candidates, target)
        if selected is None:
            rows.append({"relative_l2_error_target": target, "model": None})
            continue
        relations: dict[str, int] = {}
        for coupling in selected.retained_couplings:
            relations[coupling.relation] = relations.get(coupling.relation, 0) + 1
        dominant = sorted(
            selected.retained_couplings,
            key=lambda item: (
                -item.spectral_error_reduction_at_selection,
                -item.normalized_magnitude,
                item.left,
                item.right,
            ),
        )[:10]
        rows.append(
            {
                "relative_l2_error_target": target,
                "model": _candidate_record(selected),
                "absolute_error_reduction_from_diagonal": (
                    diagonal.relative_l2_error - selected.relative_l2_error
                ),
                "fraction_of_diagonal_error_removed": (
                    0.0
                    if diagonal.relative_l2_error <= 0.0
                    else 1.0 - selected.relative_l2_error / diagonal.relative_l2_error
                ),
                "chemical_relation_counts": dict(sorted(relations.items())),
                "dominant_couplings_by_marginal_spectral_gain": [
                    coupling.to_dict() for coupling in dominant
                ],
            }
        )
    three = next(
        (candidate for candidate in candidates if candidate.retained_coupling_count == 3),
        None,
    )
    policy = candidates[0].selection_policy
    if policy == EXACT_TERM_POLICY:
        interpretation = (
            "QM diagonal and off-diagonal terms retain their canonical values; the greedy "
            "path changes only the retained support and labels each selected term chemically"
        )
    elif policy == HYBRID_PROTECTED_POLICY:
        interpretation = (
            "diagonal and threshold-large couplings retain canonical QM values; only the "
            "small-coupling tail is compressed by a labelled numerical active-support fit"
        )
    else:
        interpretation = (
            "couplings are selected numerically in projected Hessian space, refitted on "
            "the complete active support, then labelled by local chemical coordinate families"
        )
    return {
        "selection_policy": policy,
        "diagonal": _candidate_record(diagonal),
        "three_coupling_model": None if three is None else _candidate_record(three),
        "targets": rows,
        "interpretation": interpretation,
    }


def _pareto_frontier(candidates: Sequence[_ZoomCandidate]) -> tuple[_ZoomCandidate, ...]:
    frontier = []
    best_error = math.inf
    best_shape_error = math.inf
    for candidate in candidates:
        shape_error = _candidate_mode_shape_error(candidate)
        if (
            candidate.relative_l2_error < best_error - 1.0e-14
            or shape_error < best_shape_error - 1.0e-14
        ):
            frontier.append(candidate)
            best_error = min(best_error, candidate.relative_l2_error)
            best_shape_error = min(best_shape_error, shape_error)
    return tuple(frontier)


def _minimum_parameter_candidate(
    candidates: Sequence[_ZoomCandidate],
    target: float,
) -> _ZoomCandidate | None:
    eligible = [item for item in candidates if _candidate_satisfies_target(item, target)]
    return min(
        eligible,
        key=lambda item: (item.parameter_count, item.relative_l2_error),
        default=None,
    )


def _candidate_mode_shape_error(candidate: _ZoomCandidate) -> float:
    regions = candidate.gf_mode_overlap_metrics["regions"]
    errors = [
        1.0 - float(values["mean_squared_overlap"])
        for values in regions.values()
        if int(values["mode_count"]) > 0
    ]
    return max(errors, default=0.0)


def _candidate_regional_curvature_error(candidate: _ZoomCandidate) -> float:
    errors = [
        float(values["normalized_rms"])
        for values in candidate.gf_regional_curvature_metrics.values()
        if int(values["mode_count"]) > 0
    ]
    return max(errors, default=0.0)


def _candidate_satisfies_target(candidate: _ZoomCandidate, target: float) -> bool:
    """Require Hessian accuracy without concentrating error in one mode region."""

    mode_shape_limit = max(
        MODE_SHAPE_ERROR_FLOOR,
        MODE_SHAPE_TARGET_MULTIPLIER * float(target),
    )
    regional_curvature_limit = max(
        REGIONAL_CURVATURE_ERROR_FLOOR,
        REGIONAL_CURVATURE_TARGET_MULTIPLIER * float(target),
    )
    return (
        candidate.relative_l2_error <= float(target)
        and _candidate_mode_shape_error(candidate) <= mode_shape_limit
        and _candidate_regional_curvature_error(candidate) <= regional_curvature_limit
    )


def _select_adaptive_zoom_candidates(
    families: Mapping[str, tuple[_ZoomCandidate, ...]],
) -> tuple[_ZoomCandidate, _ZoomCandidate, _ZoomCandidate, _ZoomCandidate]:
    if not families or any(not candidates for candidates in families.values()):
        raise ValueError("ZAFF basis comparison produced no candidates")
    all_candidates = [candidate for candidates in families.values() for candidate in candidates]
    strict = [
        candidate for candidate in all_candidates if _candidate_satisfies_target(candidate, 0.01)
    ]
    if not strict:
        raise ValueError("no PIC, local-SONIC or hybrid model reaches 1%")
    z2 = min(strict, key=lambda item: (item.parameter_count, item.relative_l2_error, item.basis))

    def coarser(target: float, finer: _ZoomCandidate) -> _ZoomCandidate:
        eligible = [
            candidate
            for candidate in all_candidates
            if _candidate_satisfies_target(candidate, target)
            and candidate.parameter_count <= finer.parameter_count
            and candidate.relative_l2_error + 1.0e-12 >= finer.relative_l2_error
        ]
        if not eligible:
            return finer
        return min(
            eligible,
            key=lambda item: (item.parameter_count, -item.relative_l2_error, item.basis),
        )

    z1 = coarser(0.02, z2)
    z0 = coarser(0.05, z1)
    complete_eligible = [
        candidates[-1]
        for candidates in families.values()
        if candidates[-1].parameter_count >= z2.parameter_count
        and candidates[-1].relative_l2_error <= z2.relative_l2_error + 1.0e-12
    ]
    if not complete_eligible:
        raise ValueError("no complete ZAFF ceiling continues the selected zoom sequence")
    z3 = min(
        complete_eligible,
        key=lambda item: (item.parameter_count, item.relative_l2_error, item.basis),
    )
    return z0, z1, z2, z3


def _zoom_level(
    identifier: str,
    candidate: _ZoomCandidate,
    pic_field: PrimitiveMMForceField,
    gradient_rms: float,
    gradient_max: float,
    *,
    local_salc: bool,
) -> ZaffZoomLevel:
    descriptions = {
        "H0": "smallest PIC/local-SONIC/hybrid model targeting 5% spectral error",
        "H1": "smallest PIC/local-SONIC/hybrid model targeting 2% spectral error",
        "H2": "smallest PIC/local-SONIC/hybrid model targeting 1% spectral error",
        "H3": "smallest complete PIC/local-SONIC/hybrid harmonic coupling ceiling",
    }
    return ZaffZoomLevel(
        identifier=identifier,
        description=descriptions[identifier],
        hessian_hartree_per_bohr2=candidate.hessian_bohr,
        coordinate_basis=candidate.basis,
        parameter_count=candidate.parameter_count,
        atomic_type_count=len(pic_field.atomic_types),
        primitive_type_count=len(pic_field.primitive_types),
        coupling_count=candidate.retained_coupling_count,
        relative_l2_eigenvalue_error=candidate.relative_l2_error,
        maximum_absolute_eigenvalue_error_hartree_per_bohr2=(candidate.maximum_absolute_error),
        resolved_sign_mismatch_count=candidate.sign_mismatch_count,
        equilibrium_gradient_rms_hartree_per_angstrom=gradient_rms,
        equilibrium_gradient_max_hartree_per_angstrom=gradient_max,
        metadata={
            "role": (
                "PRODUCTION_COMPLETE_HARMONIC_BASE"
                if identifier == "H3"
                else "EXPLICIT_HARMONIC_TRUNCATION_DIAGNOSTIC"
            ),
            "zaff_anharmonic_zoom": False,
            "coordinate_count": candidate.coordinate_count,
            "local_salc": bool(local_salc and candidate.basis in {"SONIC_LOCAL", "HYBRID_LOCAL"}),
            "global_symmetrization": False,
            "coupling_selection_policy": candidate.selection_policy,
            "harmonic_selection_policy": candidate.selection_policy,
            "coupling_origin_counts": _coupling_origin_counts(candidate),
            "retained_couplings": [item.to_dict() for item in candidate.retained_couplings],
            "fit_diagnostics": dict(candidate.fit_diagnostics),
            "gf_frequency_rms_error_cm": candidate.gf_frequency_rms_error_cm,
            "gf_frequency_maximum_absolute_error_cm": (
                candidate.gf_frequency_maximum_absolute_error_cm
            ),
            "gf_imaginary_mode_mismatch_count": (candidate.gf_imaginary_mode_mismatch_count),
            "gf_region_metrics": {
                name: dict(values) for name, values in candidate.gf_region_metrics.items()
            },
            "gf_regional_curvature_metrics": {
                name: dict(values)
                for name, values in candidate.gf_regional_curvature_metrics.items()
            },
            "gf_regional_curvature_error": _candidate_regional_curvature_error(candidate),
            "gf_mode_overlap_metrics": dict(candidate.gf_mode_overlap_metrics),
            "gf_regional_mode_shape_error": _candidate_mode_shape_error(candidate),
            "gf_mode_overlap_matrix": candidate.gf_mode_overlap_matrix.tolist(),
            "gf_mode_assignment": (candidate.gf_mode_assignment + 1).tolist(),
            "gf_degenerate_subspace_metrics": [
                dict(values) for values in candidate.gf_degenerate_subspace_metrics
            ],
        },
    )


def _candidate_record(
    candidate: _ZoomCandidate,
    *,
    include_overlap_matrix: bool = False,
) -> dict[str, Any]:
    record = {
        "basis": candidate.basis,
        "selection_policy": candidate.selection_policy,
        "coupling_origin_counts": _coupling_origin_counts(candidate),
        "coordinate_count": candidate.coordinate_count,
        "retained_coupling_count": candidate.retained_coupling_count,
        "parameter_count": candidate.parameter_count,
        "relative_l2_eigenvalue_error": candidate.relative_l2_error,
        "relative_l2_eigenvalue_error_percent": 100.0 * candidate.relative_l2_error,
        "maximum_absolute_eigenvalue_error_hartree_per_bohr2": (candidate.maximum_absolute_error),
        "resolved_sign_mismatch_count": candidate.sign_mismatch_count,
        "gf_frequency_rms_error_cm": candidate.gf_frequency_rms_error_cm,
        "gf_frequency_maximum_absolute_error_cm": (
            candidate.gf_frequency_maximum_absolute_error_cm
        ),
        "gf_imaginary_mode_mismatch_count": candidate.gf_imaginary_mode_mismatch_count,
        "gf_region_metrics": {
            name: dict(values) for name, values in candidate.gf_region_metrics.items()
        },
        "gf_regional_curvature_metrics": {
            name: dict(values) for name, values in candidate.gf_regional_curvature_metrics.items()
        },
        "gf_regional_curvature_error": _candidate_regional_curvature_error(candidate),
        "gf_mode_overlap_metrics": dict(candidate.gf_mode_overlap_metrics),
        "gf_regional_mode_shape_error": _candidate_mode_shape_error(candidate),
        "gf_degenerate_subspace_metrics": [
            dict(values) for values in candidate.gf_degenerate_subspace_metrics
        ],
        "last_retained_coupling": (
            None if not candidate.retained_couplings else candidate.retained_couplings[-1].to_dict()
        ),
        "fit_diagnostics": dict(candidate.fit_diagnostics),
    }
    if include_overlap_matrix:
        record["gf_mode_overlap_matrix"] = candidate.gf_mode_overlap_matrix.tolist()
        record["gf_mode_assignment"] = (candidate.gf_mode_assignment + 1).tolist()
    return record


def _coupling_origin_counts(candidate: _ZoomCandidate) -> dict[str, int]:
    counts: dict[str, int] = {}
    for coupling in candidate.retained_couplings:
        counts[coupling.origin] = counts.get(coupling.origin, 0) + 1
    return dict(sorted(counts.items()))




def _stationarity_residual(
    b_matrix: np.ndarray,
    nonbonded_gradient_bohr: np.ndarray,
) -> tuple[float, float]:
    nonbonded_angstrom = np.asarray(nonbonded_gradient_bohr, dtype=float) / BOHR_TO_ANGSTROM
    coordinate_gradient, *_ = np.linalg.lstsq(
        np.asarray(b_matrix, dtype=float).T,
        -nonbonded_angstrom,
        rcond=1.0e-10,
    )
    residual = np.asarray(b_matrix, dtype=float).T @ coordinate_gradient + nonbonded_angstrom
    return float(np.sqrt(np.mean(residual**2))), float(np.max(np.abs(residual)))


def _frequency_audit_context(source: _CompilationSource) -> _FrequencyAuditContext:
    """Build the common mass metric used for every same-geometry GF comparison.

    In Cartesian coordinates the Wilson metric is the inverse atomic-mass
    matrix.  Projecting translations and rotations after mass weighting is
    therefore exactly the Cartesian representation of the Wilson GF problem,
    while avoiding any coordinate-basis dependence in the audit.
    """

    masses = np.asarray(source.masses_amu, dtype=float)
    if masses.shape != (len(source.atoms),) or np.any(masses <= 0.0):
        raise ValueError("ARCHITECT GF audit requires one positive mass per atom")
    coordinates_bohr = np.asarray(source.coordinates_angstrom, dtype=float) / BOHR_TO_ANGSTROM
    center = np.average(coordinates_bohr, axis=0, weights=masses)
    centered = coordinates_bohr - center
    sqrt_masses = np.sqrt(masses)
    external = []
    for axis in np.eye(3):
        external.append((sqrt_masses[:, None] * axis[None, :]).reshape(-1))
    for axis in np.eye(3):
        external.append((sqrt_masses[:, None] * np.cross(axis, centered)).reshape(-1))
    external_matrix = np.column_stack(external)
    _u, singular_values, vh = np.linalg.svd(external_matrix.T, full_matrices=True)
    rank = int(np.sum(singular_values > 1.0e-10 * singular_values[0]))
    vibrational_basis = vh[rank:, :].T
    inverse_sqrt = 1.0 / np.sqrt(np.repeat(masses, 3))
    provisional = _FrequencyAuditContext(
        inverse_sqrt_cartesian_masses=inverse_sqrt,
        mass_weighted_vibrational_basis=vibrational_basis,
        reference_eigenvalues=np.empty(vibrational_basis.shape[1], dtype=float),
        reference_frequencies_cm=np.empty(vibrational_basis.shape[1], dtype=float),
        reference_modes=np.empty((vibrational_basis.shape[1], vibrational_basis.shape[1])),
    )
    reference_frequencies, reference_modes = _gf_modes(
        source.source_hessian_bohr,
        provisional,
    )
    return replace(
        provisional,
        reference_eigenvalues=(
            np.sign(reference_frequencies)
            * (np.abs(reference_frequencies) / HESSIAN_EIGENVALUE_TO_CM) ** 2
        ),
        reference_frequencies_cm=reference_frequencies,
        reference_modes=reference_modes,
    )




def _gf_modes(
    hessian_hartree_per_bohr2: np.ndarray,
    context: _FrequencyAuditContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed frequencies and orthonormal mass-weighted normal modes."""

    weights = context.inverse_sqrt_cartesian_masses
    hessian = np.asarray(hessian_hartree_per_bohr2, dtype=float)
    weighted = hessian * weights[:, None] * weights[None, :]
    basis = context.mass_weighted_vibrational_basis
    reduced = basis.T @ (0.5 * (weighted + weighted.T)) @ basis
    eigenvalues, modes = np.linalg.eigh(0.5 * (reduced + reduced.T))
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * HESSIAN_EIGENVALUE_TO_CM
    return frequencies, modes


def _gf_mode_overlap_audit(
    model_frequencies_cm: np.ndarray,
    model_modes: np.ndarray,
    context: _FrequencyAuditContext,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Match normal modes and audit both individual and degenerate-subspace identity.

    The absolute mass-weighted scalar-product matrix is sign invariant.  A global
    maximum-overlap assignment supplies one model partner for each reference mode.
    Individual overlaps are not interpreted inside near-degenerate blocks; those
    blocks are compared through principal overlaps instead.
    """

    reference_modes = np.asarray(context.reference_modes, dtype=float)
    candidate_modes = np.asarray(model_modes, dtype=float)
    if reference_modes.shape != candidate_modes.shape:
        raise ValueError("GF normal-mode overlap requires equal reference/model dimensions")
    overlap = np.abs(reference_modes.T @ candidate_modes)
    assignment = _maximum_overlap_assignment(overlap)
    matched = overlap[np.arange(overlap.shape[0]), assignment]
    blocks = []
    subspace_adjusted_squared = matched**2
    model_frequency = np.asarray(model_frequencies_cm, dtype=float)
    for indices in _near_degenerate_mode_blocks(context.reference_frequencies_cm):
        model_indices = assignment[indices]
        block_overlap = reference_modes[:, indices].T @ candidate_modes[:, model_indices]
        principal = np.linalg.svd(block_overlap, compute_uv=False)
        subspace_adjusted_squared[indices] = float(np.mean(principal**2))
        blocks.append(
            {
                "reference_mode_indices": (indices + 1).tolist(),
                "model_mode_indices": (model_indices + 1).tolist(),
                "reference_frequencies_cm": context.reference_frequencies_cm[indices].tolist(),
                "model_frequencies_cm": model_frequency[model_indices].tolist(),
                "principal_overlaps": principal.tolist(),
                "minimum_principal_overlap": float(np.min(principal)),
                "mean_squared_principal_overlap": float(np.mean(principal**2)),
            }
        )
    adjusted = np.sqrt(np.clip(subspace_adjusted_squared, 0.0, 1.0))
    reference_frequency = np.abs(np.asarray(context.reference_frequencies_cm, dtype=float))
    region_metrics: dict[str, dict[str, float | int]] = {}
    for name, (lower, upper) in GF_FREQUENCY_REGIONS_CM.items():
        mask = (reference_frequency >= lower) & (reference_frequency < upper)
        region_metrics[name] = _mode_overlap_summary(adjusted[mask])

    metrics: dict[str, Any] = _mode_overlap_summary(adjusted)
    metrics["raw_assignment_mean_squared_overlap"] = float(np.mean(matched**2))
    metrics["regions"] = region_metrics
    metrics["assignment_contract"] = "global maximum sum of absolute mass-weighted mode overlaps"
    metrics["degenerate_block_contract"] = (
        "individual matched overlaps are replaced by mean squared principal overlap"
    )
    metrics["matrix_contract"] = (
        "absolute dot products; rows are reference modes and columns are model modes"
    )
    return overlap, assignment, metrics, tuple(blocks)


def _mode_overlap_summary(overlaps: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(overlaps, dtype=float)
    if not values.size:
        return {
            "mode_count": 0,
            "mean_absolute_overlap": 0.0,
            "minimum_absolute_overlap": 0.0,
            "mean_squared_overlap": 0.0,
            "modes_below_0_90": 0,
        }
    return {
        "mode_count": int(values.size),
        "mean_absolute_overlap": float(np.mean(values)),
        "minimum_absolute_overlap": float(np.min(values)),
        "mean_squared_overlap": float(np.mean(values**2)),
        "modes_below_0_90": int(np.sum(values < 0.90)),
    }


def _near_degenerate_mode_blocks(frequencies_cm: np.ndarray) -> tuple[np.ndarray, ...]:
    frequencies = np.asarray(frequencies_cm, dtype=float)
    if frequencies.ndim != 1:
        raise ValueError("GF reference frequencies must be one-dimensional")
    blocks: list[np.ndarray] = []
    start = 0
    for index in range(1, frequencies.size):
        scale = max(abs(frequencies[index - 1]), abs(frequencies[index]), 1.0)
        threshold = max(5.0, 0.005 * scale)
        if abs(frequencies[index] - frequencies[index - 1]) > threshold:
            if index - start > 1:
                blocks.append(np.arange(start, index, dtype=int))
            start = index
    if frequencies.size - start > 1:
        blocks.append(np.arange(start, frequencies.size, dtype=int))
    return tuple(blocks)


def _maximum_overlap_assignment(overlap: np.ndarray) -> np.ndarray:
    """Hungarian assignment, accelerated by SciPy when it is available."""

    values = np.asarray(overlap, dtype=float)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-values)
        assignment = np.empty(values.shape[0], dtype=int)
        assignment[np.asarray(rows, dtype=int)] = np.asarray(columns, dtype=int)
        return assignment
    except ImportError:
        return _maximum_overlap_assignment_numpy(values)


def _maximum_overlap_assignment_numpy(overlap: np.ndarray) -> np.ndarray:
    """Portable Hungarian fallback for a square overlap matrix."""

    values = np.asarray(overlap, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("normal-mode assignment requires a square overlap matrix")
    size = values.shape[0]
    cost = float(np.max(values)) - values
    row_potential = np.zeros(size + 1, dtype=float)
    column_potential = np.zeros(size + 1, dtype=float)
    matched_row = np.zeros(size + 1, dtype=int)
    predecessor = np.zeros(size + 1, dtype=int)
    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = np.full(size + 1, np.inf, dtype=float)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    cost[active_row - 1, candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum[candidate_column]:
                    minimum[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = np.empty(size, dtype=int)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


def _gf_region_metrics(
    reference_frequencies_cm: np.ndarray,
    frequency_difference_cm: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    reference = np.abs(np.asarray(reference_frequencies_cm, dtype=float))
    difference = np.asarray(frequency_difference_cm, dtype=float)
    metrics: dict[str, dict[str, float | int]] = {}
    for name, (lower, upper) in GF_FREQUENCY_REGIONS_CM.items():
        mask = (reference >= lower) & (reference < upper)
        selected = difference[mask]
        metrics[name] = {
            "mode_count": int(selected.size),
            "rms_error_cm": (0.0 if not selected.size else float(np.sqrt(np.mean(selected**2)))),
            "maximum_absolute_error_cm": (
                0.0 if not selected.size else float(np.max(np.abs(selected)))
            ),
        }
    return metrics


def _gf_regional_curvature_metrics(
    model_frequencies_cm: np.ndarray,
    mode_assignment: np.ndarray,
    context: _FrequencyAuditContext,
) -> dict[str, dict[str, float | int]]:
    """Audit assigned normal-mode force constants in each spectral region."""

    frequencies = np.asarray(model_frequencies_cm, dtype=float)
    assignment = np.asarray(mode_assignment, dtype=int)
    model_eigenvalues = np.sign(frequencies) * (np.abs(frequencies) / HESSIAN_EIGENVALUE_TO_CM) ** 2
    error = model_eigenvalues[assignment] - np.asarray(
        context.reference_eigenvalues,
        dtype=float,
    )
    floor = (MODAL_CURVATURE_FLOOR_CM / HESSIAN_EIGENVALUE_TO_CM) ** 2
    normalized = error / np.maximum(np.abs(context.reference_eigenvalues), floor)
    frequencies = np.abs(np.asarray(context.reference_frequencies_cm, dtype=float))
    metrics: dict[str, dict[str, float | int]] = {}
    for name, (lower, upper) in GF_FREQUENCY_REGIONS_CM.items():
        mask = (frequencies >= lower) & (frequencies < upper)
        selected = normalized[mask]
        metrics[name] = {
            "mode_count": int(selected.size),
            "normalized_rms": (0.0 if not selected.size else float(np.sqrt(np.mean(selected**2)))),
            "maximum_absolute_normalized_error": (
                0.0 if not selected.size else float(np.max(np.abs(selected)))
            ),
        }
    return metrics


def _vibrational_cartesian_basis(coordinates_bohr: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coordinates_bohr, dtype=float)
    centered = xyz - np.mean(xyz, axis=0)
    vectors = [np.tile(axis, (len(xyz), 1)).reshape(-1) for axis in np.eye(3)]
    vectors.extend(np.cross(axis, centered).reshape(-1) for axis in np.eye(3))
    external = np.column_stack(vectors)
    _u, singular_values, vh = np.linalg.svd(external.T, full_matrices=True)
    rank = int(np.sum(singular_values > 1.0e-10 * singular_values[0]))
    return vh[rank:, :].T


def _projected_eigenvalues(hessian: np.ndarray, basis: np.ndarray) -> np.ndarray:
    matrix = np.asarray(hessian, dtype=float)
    reduced = basis.T @ (0.5 * (matrix + matrix.T)) @ basis
    return np.linalg.eigvalsh(0.5 * (reduced + reduced.T))


def _validate_config(config: ZaffBuildConfig) -> None:
    if config.coordinate_chart not in {"adaptive", "primitive", "sonic"}:
        raise ValueError("ZAFF coordinate chart must be adaptive, primitive or sonic")
    if config.global_symmetrization:
        raise ValueError("ZAFF permits local SONIC SALCs but forbids global symmetrization")
    if not config.local_salc:
        raise ValueError("the ZAFF SONIC comparison requires local SALCs")
    if int(config.parallel_workers) < 0:
        raise ValueError("ARCHITECT parallel worker count must be non-negative")
    if (
        not math.isfinite(config.anharmonic_stretch_step_angstrom)
        or config.anharmonic_stretch_step_angstrom <= 0.0
        or not math.isfinite(config.anharmonic_angular_step_radian)
        or config.anharmonic_angular_step_radian <= 0.0
    ):
        raise ValueError("ZAFF anharmonic acquisition steps must be finite and positive")
    if config.anharmonic_stretch_model not in {"morse", "inverse_power", "spf"}:
        raise ValueError("ZAFF anharmonic stretch model must be morse, inverse_power or spf")
    if config.coupling_policy not in {
        EXACT_TERM_POLICY,
        REFITTED_POLICY,
        HYBRID_PROTECTED_POLICY,
    }:
        raise ValueError(
            "ARCHITECT coupling policy must be exact, hybrid protected-core or numerical"
        )
    for label, value in (
        ("mixing", config.protected_coupling_mixing_threshold),
        ("second-order", config.protected_coupling_second_order_threshold),
        ("gap-floor", config.protected_diagonal_gap_floor_fraction),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"protected coupling {label} threshold must be finite and positive")
    if not math.isclose(float(config.nonbonded_14_scale), 1.0, abs_tol=1.0e-12):
        raise ValueError("ZAFF no longer supports a 1-4 scale; the legacy field must be 1")
    if not math.isfinite(config.charge_prior_sigma_e) or config.charge_prior_sigma_e <= 0.0:
        raise ValueError("ZAFF charge prior sigma must be finite and positive")
    if not math.isfinite(config.charge_prior_weight) or config.charge_prior_weight < 0.0:
        raise ValueError("ZAFF charge prior weight must be finite and non-negative")
    if (
        not math.isfinite(config.maximum_charge_displacement_e)
        or config.maximum_charge_displacement_e <= 0.0
    ):
        raise ValueError("ZAFF maximum charge displacement must be finite and positive")
    if int(config.charge_optimization_iterations) <= 0:
        raise ValueError("ZAFF charge optimization iterations must be positive")
    if not math.isfinite(config.nonbonded_prior_weight) or config.nonbonded_prior_weight < 0.0:
        raise ValueError("ZAFF non-bonded prior weight must be finite and non-negative")
    for label, value in (
        ("well-depth", config.vdw_well_depth_prior_sigma_log),
        ("radius", config.vdw_radius_prior_sigma_log),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"ZAFF non-bonded {label} prior sigma must be positive")
    if int(config.nonbonded_optimization_iterations) <= 0:
        raise ValueError("ZAFF non-bonded optimization iterations must be positive")
    for label, value in (
        ("preoptimized radius", config.preoptimized_vdw_radius_scale),
        ("preoptimized well depth", config.preoptimized_vdw_well_depth_scale),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"ZAFF {label} scale must be finite and positive")
    if config.vdw_model != "exppe_from_uff":
        raise ValueError("ZAFF vdW is frozen to the all-pair undamped Exp-PE model")
    if config.preoptimized_vdw_damping_scale is not None:
        raise ValueError("ZAFF Exp-PE has no damping parameter to preoptimize")
    if config.vdw_damping_scale != 0.80 or config.vdw_damping_exponent != 24:
        raise ValueError("ZAFF Exp-PE has no damping parameters")
    if config.topology_scaling != "continuous_erf_paths":
        raise ValueError("ZAFF Exp-PE requires its certified continuous topology weights")
    if config.electrostatic_model != "gaussian_erf_all_pairs":
        raise ValueError("ZAFF electrostatics is frozen to all-pair Gaussian charge penetration")
    if (
        not math.isfinite(config.electrostatic_gaussian_width_scale)
        or config.electrostatic_gaussian_width_scale <= 0.0
    ):
        raise ValueError("ZAFF electrostatic Gaussian width scale must be positive")
    if (
        not math.isfinite(config.topology_switch_alpha_per_angstrom)
        or config.topology_switch_alpha_per_angstrom <= 0.0
    ):
        raise ValueError("ZAFF topology switch alpha must be finite and positive")
    if (
        not math.isfinite(config.nonbonded_neighbor_cutoff_angstrom)
        or config.nonbonded_neighbor_cutoff_angstrom <= 0.0
    ):
        raise ValueError("ARCHITECT regional non-bonded cutoff must be finite and positive")
    if not math.isfinite(config.fmm_precision) or not 0.0 < config.fmm_precision < 1.0:
        raise ValueError("ARCHITECT FMM precision must lie strictly between zero and one")
    if int(config.fmm_minimum_atoms) < 2:
        raise ValueError("ARCHITECT FMM atom threshold must be at least two")
    if config.morse_alpha_scale <= 0.0:
        raise ValueError("ZAFF Morse alpha scale must be positive")
    if not config.comparison_error_targets or any(
        not math.isfinite(value) or value <= 0.0 for value in config.comparison_error_targets
    ):
        raise ValueError("ZAFF comparison error targets must be finite and positive")


def _matrix_rms(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _vector_rms(vector: np.ndarray) -> float:
    values = np.asarray(vector, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
