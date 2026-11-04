from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

import numpy as np

from matrix_core import (
    CalculationResources,
    authorized_parent_plan_from_environment,
    atomic_json_write,
    require_authorized_descendant_calculation,
)
from matrix_chem.geometry_alignment import (
    aligned_cartesian_displacement,
    kabsch_rotation,
    rotate_cartesian_derivatives,
)
from matrix_chem.xyzin_geometry import read_xyzin_geometry
from matrix_qm.method_dialects import backend_method_name
from matrix_numerics import fit_taylor_derivatives
from .internal_coordinates import cartesian_from_internal_jacobian


BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
ORACLE_LINK_POINT_RESULT_SCHEMA = "oracle.link.point_result.v1"
ORACLE_LINK_SCAN_MANIFEST_SCHEMA = "oracle.link.scan_manifest.v1"
ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA = "oracle.link.finite_difference_derivatives.v1"
FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA = "matrix.link.full_precision_derivatives.v1"
FULL_PRECISION_DERIVATIVE_ARTIFACT_NAME = "matrix_derivatives.json"
COMPOSITE_TERM_RESTART_SCHEMA = "matrix.link.composite-term-restart.v1"
COMPOSITE_TERM_RESTART_NAME = "matrix_composite_term_result.json"

@dataclass(frozen=True)
class PESExplorationPolicy:
    """Symmetry/coordinate policy shared by scans, SENTINEL and PES fitting."""

    retained_group: str = "C1"
    pointwise_oracle_symmetry: bool = True
    pointwise_cartesian_symmetrization: bool = True
    separate_exocyclic_torsions: bool = True

    def __post_init__(self) -> None:
        group = str(self.retained_group).strip().upper() or "C1"
        object.__setattr__(self, "retained_group", group)
        if self.pointwise_cartesian_symmetrization and not self.pointwise_oracle_symmetry:
            raise ValueError("pointwise symmetrization requires ORACLE symmetry perception")
        if not self.separate_exocyclic_torsions:
            raise ValueError("PES exploration requires separate exocyclic torsions")

    def protocol_payload(self) -> dict[str, object]:
        return {
            "mode": "PES_EXPLORATION",
            "retained_group": self.retained_group,
            "sonic_symmetry": (
                "C1_UNSYMMETRIZED" if self.retained_group == "C1" else "RETAINED_GROUP_ONLY"
            ),
            "pointwise_oracle_symmetry": self.pointwise_oracle_symmetry,
            "pointwise_cartesian_symmetrization": self.pointwise_cartesian_symmetrization,
            "separate_exocyclic_torsions": self.separate_exocyclic_torsions,
        }

    @classmethod
    def monte_carlo(cls) -> "PESExplorationPolicy":
        """Fast C1 policy for independent GA/Monte Carlo candidates.

        Random intermolecular poses have no symmetry contract to recover.
        Disabling pointwise perception also prevents a geometry-only operation
        from scaling with the general molecular-symmetry machinery.
        """

        return cls(
            retained_group="C1",
            pointwise_oracle_symmetry=False,
            pointwise_cartesian_symmetrization=False,
            separate_exocyclic_torsions=True,
        )


@dataclass(frozen=True)
class PointSymmetryInfo:
    point_group: str
    operation_count: int
    projection_status: str
    projection_max_displacement_angstrom: float
    projection_rms_displacement_angstrom: float


@dataclass(frozen=True)
class CoordinateDirection:
    """Cartesian displacement basis vector for a LINK scan coordinate."""

    kind: str
    label: str
    vector_angstrom: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector_angstrom, dtype=float).reshape(-1)
        if vector.size == 0:
            raise ValueError("coordinate direction cannot be empty")
        if not np.all(np.isfinite(vector)):
            raise ValueError("coordinate direction contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("coordinate direction has zero norm")
        object.__setattr__(self, "vector_angstrom", vector)


@dataclass(frozen=True)
class ScanPoint:
    index: int
    displacement: float
    coordinates_angstrom: np.ndarray
    coordinate_key: str = ""
    side: str = "center"
    xyz_path: Path | None = None
    result_path: Path | None = None
    status: str = "prepared"
    symmetry_analyzed: bool = False
    point_group: str = ""
    symmetry_operation_count: int = 0
    symmetry_projection_status: str = "NOT_ANALYZED"
    symmetry_projection_max_displacement_angstrom: float = 0.0
    symmetry_projection_rms_displacement_angstrom: float = 0.0
    retained_group: str = "C1"

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates_angstrom, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("scan point coordinates must have shape natoms x 3")
        if not np.all(np.isfinite(coords)):
            raise ValueError("scan point coordinates contain non-finite values")
        object.__setattr__(self, "coordinates_angstrom", coords)
        if self.xyz_path is not None:
            object.__setattr__(self, "xyz_path", Path(self.xyz_path))
        if self.result_path is not None:
            object.__setattr__(self, "result_path", Path(self.result_path))
        object.__setattr__(self, "retained_group", str(self.retained_group).strip().upper() or "C1")


@dataclass(frozen=True)
class PointEvaluationResult:
    point_index: int
    displacement: float
    coordinate_key: str = ""
    side: str = "center"
    energy_hartree: float | None = None
    gradient_hartree_per_bohr: np.ndarray | None = None
    cm5_charges: np.ndarray | None = None
    hessian_hartree_per_bohr2: np.ndarray | None = None
    backend_coordinates_angstrom: np.ndarray | None = None
    frame_alignment_rms_angstrom: float = 0.0
    status: str = "completed"
    message: str = ""
    source: str = ""
    point_group: str = ""
    symmetry_projection_status: str = "NOT_ANALYZED"
    symmetry_projection_max_displacement_angstrom: float = 0.0
    symmetry_projection_rms_displacement_angstrom: float = 0.0
    execution: dict[str, object] = field(default_factory=dict)
    # In-process backends return the converged SCF state without serializing
    # large matrices into the point JSON.  The optimizer promotes it only when
    # the geometry is accepted; rejected trials and FD branches cannot pollute
    # the parent history.
    scf_state: object | None = field(default=None, repr=False, compare=False)
    schema: str = ORACLE_LINK_POINT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        coordinate_key = str(self.coordinate_key).strip()
        side = str(self.side).strip().casefold() or "unspecified"
        if side not in {"center", "plus", "minus", "unspecified"}:
            raise ValueError("finite-difference point side must be center, plus, or minus")
        displacement = float(self.displacement)
        if coordinate_key and side == "plus" and displacement <= 0.0:
            raise ValueError("plus finite-difference point must have positive displacement")
        if coordinate_key and side == "minus" and displacement >= 0.0:
            raise ValueError("minus finite-difference point must have negative displacement")
        if coordinate_key and side == "center" and displacement != 0.0:
            raise ValueError("center finite-difference point must have zero displacement")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "coordinate_key", coordinate_key)
        if self.energy_hartree is not None:
            object.__setattr__(self, "energy_hartree", float(self.energy_hartree))
        for name in ("gradient_hartree_per_bohr", "hessian_hartree_per_bohr2"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains non-finite values")
                object.__setattr__(self, name, array)
        if self.backend_coordinates_angstrom is not None:
            coordinates = np.asarray(self.backend_coordinates_angstrom, dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError("backend coordinates must have shape natoms x 3")
            object.__setattr__(self, "backend_coordinates_angstrom", coordinates)
        object.__setattr__(
            self, "frame_alignment_rms_angstrom", float(self.frame_alignment_rms_angstrom)
        )


@dataclass(frozen=True)
class FiniteDifferenceDerivatives:
    coordinate_label: str
    energy_derivatives_hartree: tuple[float, ...]
    gradient_derivatives_hartree_per_bohr: tuple[np.ndarray, ...] = ()
    hessian_derivatives_hartree_per_bohr2: tuple[np.ndarray, ...] = ()
    fit_rank: int = 0
    residual_norm: float = 0.0


@dataclass(frozen=True)
class QMBackendTerm:
    """One signed electronic-structure contribution to a composite surface."""

    label: str
    coefficient: float
    backend: "QMScanBackend"

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label:
            raise ValueError("composite backend terms require a label")
        coefficient = float(self.coefficient)
        if not math.isfinite(coefficient) or coefficient == 0.0:
            raise ValueError("composite backend coefficients must be finite and non-zero")
        if not isinstance(self.backend, QMScanBackend):
            raise TypeError("composite backend terms require a QMScanBackend")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class QMScanBackend:
    """Electronic-structure backend settings for LINK point scans."""

    name: str
    route: str = ""
    method: str = ""
    reference: str | None = None
    basis: str = ""
    dispersion_contract: str | None = None
    basis_file: Path | None = None
    turbomole_control_template: Path | None = None
    charge: int = 0
    multiplicity: int = 1
    executable: str | None = None
    launcher: Mapping[str, object] | None = None
    timeout: float | None = None
    env: dict[str, str] | None = None
    extra_args: tuple[str, ...] = ()
    oniom_high_atoms: tuple[int, ...] = ()
    oniom_atom_types: tuple[str, ...] = ()
    oniom_low_route: str = "#p UFF Force"
    oniom_link_atom_scale: float = 0.723
    gaussian_connectivity_bonds: tuple[tuple[int, int, float], ...] | None = None
    topology_bonds: tuple[tuple[int, int], ...] = ()
    processors: int = 1
    matrix_gfn2_solver_mode: str = "dense"
    matrix_gfn2_linear_scaling_threshold: int = 128
    matrix_gfn2_alpb_solvent: str | None = None
    memory_gb: int | None = None
    force_field: Path | None = None
    zaff_library: Path | None = None
    zaff_gaff_parameters: Path | None = None
    resolution: Mapping[str, object] | None = None
    properties: tuple[str, ...] = ()
    device: str | None = None
    zaff_zoom_level: str | None = None
    gradient_mode: str = "analytic"
    numerical_gradient_step_bohr: float = 1.0e-3
    numerical_gradient_stencil: str = "forward"
    electronic_state: int = 0
    excited_states: int | None = None
    state_spin: str = "singlet"
    freeze_core: bool = False
    state_tracking: str = "apoc"
    state_tracking_minimum_overlap: float = 0.70
    state_tracking_small_gap_minimum_overlap: float = 0.50
    state_tracking_ambiguity_margin: float = 0.05
    state_tracking_initial_roots: int | None = None
    state_tracking_root_increment: int = 2
    state_tracking_max_roots: int | None = None
    state_tracking_max_displacement_halvings: int = 3
    tda_spectral_range_ev: float = 7.0
    two_photon_spectral_range_ev: float = 20.0
    restart_artifact: Path | None = None
    initial_scf_density_artifact: Path | None = None
    restart_history_artifacts: tuple[Path, ...] = ()
    restart_projection: str | None = None
    gaussian_native_checkpoint_restart: bool = False
    scf_convergence: Mapping[str, float] | None = None
    restart_reuse_for_displacements: bool = True
    reuse_density_projection: bool = False
    gaussian_nosymm_authorized: bool = False
    gaussian_accuracy_tier: str = "protocol"
    # Scalar finite-difference energy points use a tighter SCF gate because
    # their differences can be smaller than the ordinary protocol tolerance.
    finite_difference_energy_scf_tight: bool = False
    pyfock_initial_guess: str = "core"
    composite_terms: tuple[QMBackendTerm, ...] = ()
    composite_parallel_workers: int = 1
    resume_completed_composite_terms: bool = False
    std2_contract: Mapping[str, object] | None = None
    std2_hessian_step_bohr: float = 0.01

    def __post_init__(self) -> None:
        # Every finite-difference scan is governed by the packaged, frozen
        # contract.  This prevents ad-hoc scripts (including KEYMAKER jobs)
        # from silently changing frames, units, or stencil conventions.
        if self.gaussian_nosymm_authorized:
            raise ValueError("Gaussian symmetry cannot be disabled")
        if not np.isfinite(self.tda_spectral_range_ev) or self.tda_spectral_range_ev <= 0:
            raise ValueError("TDA spectral range must be finite and positive")
        if float(self.two_photon_spectral_range_ev) <= 0.0:
            raise ValueError("two-photon spectral range must be positive")
        requested_gradient_mode = str(self.gradient_mode).strip().casefold()
        if requested_gradient_mode in {
            "energy",
            "energy-only",
            "numerical",
            "link-numerical",
            "sonic_one_sided_hybrid",
        }:
            from .protocol_manifest import load_link_optimizer_protocol

            protocol = load_link_optimizer_protocol()
            if protocol.payload["finite_derivative_contract"]["status"] != "frozen":
                raise ValueError("finite-derivative protocol is not frozen")
        terms = tuple(
            term
            if isinstance(term, QMBackendTerm)
            else QMBackendTerm(
                str(term.get("label", "")),
                float(term.get("coefficient", 0.0)),
                qm_scan_backend_from_dict(term.get("backend", {})),
            )
            for term in self.composite_terms
        )
        workers = int(self.composite_parallel_workers)
        if workers < 1:
            raise ValueError("composite_parallel_workers must be positive")
        backend_name = _normalized_backend_name(self.name)
        if backend_name == "linear_composite" and not terms:
            raise ValueError("linear_composite requires at least one composite term")
        if backend_name == "capped_subtractive":
            expected = ("low_real", "high_model", "low_model")
            if tuple(term.label for term in terms) != expected:
                raise ValueError(
                    "capped_subtractive terms must be low_real, high_model, low_model"
                )
            if self.electronic_state != 0:
                raise ValueError("QM/QM states must be specified separately on each component")
            if tuple(term.coefficient for term in terms) != (1.0, 1.0, -1.0):
                raise ValueError(
                    "capped_subtractive coefficients must be +1, +1, -1"
                )
            if not self.oniom_high_atoms or not self.topology_bonds:
                raise ValueError(
                    "capped_subtractive requires model atoms and covalent topology bonds"
                )
        if not 0.0 < float(self.oniom_link_atom_scale) < 1.0:
            raise ValueError("oniom_link_atom_scale must lie strictly between zero and one")
        matrix_gfn2_solver_mode = (
            str(self.matrix_gfn2_solver_mode).strip().casefold().replace("_", "-")
        )
        if matrix_gfn2_solver_mode == "linear-scaling":
            matrix_gfn2_solver_mode = "linear"
        if matrix_gfn2_solver_mode not in {"dense", "linear", "auto"}:
            raise ValueError("matrix_gfn2_solver_mode must be dense, linear, or auto")
        matrix_gfn2_threshold = int(self.matrix_gfn2_linear_scaling_threshold)
        if matrix_gfn2_threshold < 1:
            raise ValueError("matrix_gfn2_linear_scaling_threshold must be positive")
        alpb_solvent = str(self.matrix_gfn2_alpb_solvent or "").strip() or None
        if alpb_solvent is not None:
            if self.name.strip().casefold().replace("-", "_") != "matrix_gfn2":
                raise ValueError("ALPB solvent is supported only by MATRIX-GFN2")
            from matrix_gfn2.alpb import ALPBParameters

            alpb_solvent = ALPBParameters.gfn2(alpb_solvent).solvent
        nominal_overlap = float(self.state_tracking_minimum_overlap)
        small_gap_overlap = float(self.state_tracking_small_gap_minimum_overlap)
        if not 0.0 < nominal_overlap <= 1.0:
            raise ValueError("state_tracking_minimum_overlap must lie in (0, 1]")
        if not 0.0 < small_gap_overlap <= nominal_overlap:
            raise ValueError(
                "state_tracking_small_gap_minimum_overlap must lie in "
                "(0, state_tracking_minimum_overlap]"
            )
        object.__setattr__(self, "composite_terms", terms)
        if self.launcher is not None:
            object.__setattr__(self, "launcher", dict(self.launcher))
        object.__setattr__(self, "composite_parallel_workers", workers)
        object.__setattr__(self, "matrix_gfn2_solver_mode", matrix_gfn2_solver_mode)
        object.__setattr__(self, "matrix_gfn2_alpb_solvent", alpb_solvent)
        object.__setattr__(
            self, "matrix_gfn2_linear_scaling_threshold", matrix_gfn2_threshold
        )
        object.__setattr__(
            self,
            "topology_bonds",
            tuple(tuple(int(index) for index in pair) for pair in self.topology_bonds),
        )
        accuracy_tier = str(self.gaussian_accuracy_tier).strip().casefold()
        if accuracy_tier not in {
            "protocol",
            "coarse",
            "initial",
            "intermediate",
            "final",
        }:
            raise ValueError(
                "gaussian_accuracy_tier must be protocol, coarse, initial, "
                "intermediate, or final"
            )
        object.__setattr__(self, "gaussian_accuracy_tier", accuracy_tier)
        if self.finite_difference_energy_scf_tight and str(self.gradient_mode).strip().casefold() not in {
            "energy",
            "energy-only",
        }:
            raise ValueError(
                "finite_difference_energy_scf_tight is valid only for scalar-energy points"
            )
        object.__setattr__(
            self,
            "restart_history_artifacts",
            tuple(Path(item).expanduser() for item in self.restart_history_artifacts),
        )


def prepare_pes_exploration_geometry(
    xyzin_path: Path | str,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    policy: PESExplorationPolicy | None = None,
) -> tuple[np.ndarray, PointSymmetryInfo]:
    """Perceive and optionally project the instantaneous symmetry of one PES point."""

    from matrix_chem import MolecularGeometry, SymmetryThresholds, read_symmetry_thresholds
    from matrix_chem.symmetry import (
        analyze_molecular_symmetry,
        symmetry_projection_diagnostics,
        symmetrize_molecular_geometry,
    )

    resolved = policy or PESExplorationPolicy()
    coords = np.asarray(coordinates_angstrom, dtype=float)
    geometry = MolecularGeometry(
        atoms=tuple(str(atom) for atom in atoms),
        coordinates_angstrom=coords,
    )
    if not resolved.pointwise_oracle_symmetry:
        return coords.copy(), PointSymmetryInfo(
            point_group="NOT_ANALYZED",
            operation_count=0,
            projection_status="DISABLED",
            projection_max_displacement_angstrom=0.0,
            projection_rms_displacement_angstrom=0.0,
        )
    try:
        thresholds = read_symmetry_thresholds(Path(xyzin_path))
    except ValueError:
        # Plain XYZ inputs accepted by the low-level scan API have no frozen
        # ORACLE section yet; use the same documented ORACLE defaults.
        thresholds = SymmetryThresholds()
    symmetry = analyze_molecular_symmetry(
        geometry,
        distance_tolerance=thresholds.distance_angstrom,
        inertia_tolerance=thresholds.inertia_relative,
        max_rotation_order=thresholds.max_rotation_order,
    )
    projected = (
        symmetrize_molecular_geometry(geometry, symmetry)
        if resolved.pointwise_cartesian_symmetrization
        else geometry
    )
    diagnostics = symmetry_projection_diagnostics(geometry, projected, symmetry)
    return np.asarray(projected.coordinates_angstrom, dtype=float), PointSymmetryInfo(
        point_group=symmetry.point_group,
        operation_count=len(symmetry.operations),
        projection_status=(
            diagnostics.status
            if resolved.pointwise_cartesian_symmetrization
            else "PERCEIVED_NOT_PROJECTED"
        ),
        projection_max_displacement_angstrom=diagnostics.max_displacement_angstrom,
        projection_rms_displacement_angstrom=diagnostics.rms_displacement_angstrom,
    )


def _scan_point_with_pointwise_symmetry(
    xyzin_path: Path | str,
    atoms: Sequence[str],
    point: ScanPoint,
    *,
    policy: PESExplorationPolicy,
) -> ScanPoint:
    if point.symmetry_analyzed and point.retained_group == policy.retained_group:
        return point
    coordinates, symmetry = prepare_pes_exploration_geometry(
        xyzin_path,
        atoms,
        point.coordinates_angstrom,
        policy=policy,
    )
    return replace(
        point,
        coordinates_angstrom=coordinates,
        symmetry_analyzed=True,
        point_group=symmetry.point_group,
        symmetry_operation_count=symmetry.operation_count,
        symmetry_projection_status=symmetry.projection_status,
        symmetry_projection_max_displacement_angstrom=(
            symmetry.projection_max_displacement_angstrom
        ),
        symmetry_projection_rms_displacement_angstrom=(
            symmetry.projection_rms_displacement_angstrom
        ),
        retained_group=policy.retained_group,
    )


def coordinate_direction_from_gic(
    xyzin_path: Path | str,
    coordinate: str | int,
    *,
    reference_coordinates_angstrom: np.ndarray | None = None,
    rcond: float = 1.0e-8,
    definition: object | None = None,
) -> CoordinateDirection:
    """Build a linearized Cartesian displacement direction from a frozen SONIC/GIC row."""

    return coordinate_directions_from_gics(
        xyzin_path,
        (coordinate,),
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        rcond=rcond,
        definition=definition,
    )[0]


def coordinate_directions_from_gics(
    xyzin_path: Path | str,
    coordinates: Sequence[str | int],
    *,
    reference_coordinates_angstrom: np.ndarray | None = None,
    rcond: float = 1.0e-8,
    definition: object | None = None,
) -> tuple[CoordinateDirection, ...]:
    """Build several SONIC directions through one B evaluation and inverse."""

    from matrix_smith import build_gic_b_matrix, read_gic_definition_from_xyzin

    target = Path(xyzin_path)
    resolved_definition = (
        read_gic_definition_from_xyzin(target) if definition is None else definition
    )
    coords = (
        np.asarray(reference_coordinates_angstrom, dtype=float)
        if reference_coordinates_angstrom is not None
        else np.asarray(resolved_definition.reference_coordinates_angstrom, dtype=float)
    )
    b_matrix = np.asarray(
        build_gic_b_matrix(resolved_definition, coordinates_angstrom=coords).rows,
        dtype=float,
    )
    labels = tuple(gic.identifier for gic in resolved_definition.gics)
    names = tuple(gic.name for gic in resolved_definition.gics)
    indices = tuple(
        _coordinate_index(coordinate, labels, names) for coordinate in coordinates
    )
    from .fragment_backtransform import direct_fragment_rigid_tangent

    cartesian_from_q = cartesian_from_internal_jacobian(b_matrix, rcond=rcond)
    fragment_tangent = direct_fragment_rigid_tangent(
        resolved_definition,
        coords,
        b_matrix,
    )
    for handled_index in fragment_tangent.handled_indices:
        cartesian_from_q[:, handled_index] = fragment_tangent.cartesian_from_q[:, handled_index]
    source = (
        f"#GIC {target}"
        if definition is None
        else f"PES-exploration SONIC {target}"
    )
    return tuple(
        CoordinateDirection(
            kind="sonic",
            label=labels[index],
            vector_angstrom=cartesian_from_q[:, index],
            source=source,
        )
        for index in indices
    )


def coordinate_direction_from_pes_exploration_gic(
    xyzin_path: Path | str,
    coordinate: str | int,
    *,
    retained_group: str = "C1",
    rcond: float = 1.0e-8,
) -> CoordinateDirection:
    """Build a direction in the scan-safe transient SONIC contract."""

    from matrix_smith import build_pes_exploration_gic_definition_from_xyzin

    definition = build_pes_exploration_gic_definition_from_xyzin(
        Path(xyzin_path), retained_group=retained_group
    )
    return coordinate_direction_from_gic(
        xyzin_path,
        coordinate,
        rcond=rcond,
        definition=definition,
    )


def coordinate_direction_from_normal_mode(
    xyzin_path: Path | str,
    mode: int,
) -> CoordinateDirection:
    """Build a Cartesian displacement direction from a stored #NORMAL_MODES row."""

    from matrix_qm.sections import read_normal_modes_section

    modes = read_normal_modes_section(Path(xyzin_path))
    mode_index = int(mode)
    if mode_index < 1 or mode_index > modes.modes.shape[0]:
        raise ValueError(f"normal-mode index {mode_index} outside 1..{modes.modes.shape[0]}")
    vector = np.asarray(modes.modes[mode_index - 1], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"normal-mode {mode_index} has zero norm")
    return CoordinateDirection(
        kind="normal_mode",
        label=f"Q{mode_index}",
        vector_angstrom=vector / norm,
        source=f"#NORMAL_MODES {xyzin_path}",
    )


def coordinate_direction_from_cartesian_vector(
    vector: Sequence[float] | np.ndarray,
    *,
    label: str = "cartesian",
) -> CoordinateDirection:
    """Build a direction from an explicit Cartesian vector."""

    return CoordinateDirection(kind="cartesian", label=label, vector_angstrom=np.asarray(vector))


def prepare_coordinate_scan(
    xyzin_path: Path | str,
    direction: CoordinateDirection,
    displacements: Sequence[float],
    *,
    run_dir: Path | str | None = None,
    write_xyz: bool = True,
    exploration_policy: PESExplorationPolicy | None = None,
) -> tuple[ScanPoint, ...]:
    """Generate displaced geometries for a one-dimensional LINK scan."""

    geometry = read_xyzin_geometry(Path(xyzin_path))
    coords0 = np.asarray(geometry.coordinates_angstrom, dtype=float)
    vector = np.asarray(direction.vector_angstrom, dtype=float)
    if vector.size != coords0.size:
        raise ValueError(
            f"coordinate direction has {vector.size} components, expected {coords0.size}"
        )
    target_dir = Path(run_dir) if run_dir is not None else None
    if target_dir is not None and write_xyz:
        target_dir.mkdir(parents=True, exist_ok=True)
    policy = exploration_policy or PESExplorationPolicy()
    points: list[ScanPoint] = []
    for index, displacement in enumerate(displacements):
        value = float(displacement)
        raw_coords = coords0 + value * vector.reshape(coords0.shape)
        coords, symmetry = prepare_pes_exploration_geometry(
            xyzin_path,
            geometry.atoms,
            raw_coords,
            policy=policy,
        )
        xyz_path = None
        result_path = None
        if target_dir is not None:
            xyz_path = target_dir / f"point_{index:04d}.xyz"
            result_path = target_dir / f"point_{index:04d}.json"
            if write_xyz:
                _write_xyz(
                    xyz_path,
                    geometry.atoms,
                    coords,
                    f"{direction.label} {value:.12g}; ORACLE point group {symmetry.point_group}",
                )
        points.append(
            ScanPoint(
                index=index,
                displacement=value,
                coordinates_angstrom=coords,
                coordinate_key=str(direction.label),
                side=("plus" if value > 0.0 else "minus" if value < 0.0 else "center"),
                xyz_path=xyz_path,
                result_path=result_path,
                symmetry_analyzed=True,
                point_group=symmetry.point_group,
                symmetry_operation_count=symmetry.operation_count,
                symmetry_projection_status=symmetry.projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    symmetry.projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    symmetry.projection_rms_displacement_angstrom
                ),
                retained_group=policy.retained_group,
            )
        )
    return tuple(points)


def write_scan_manifest(
    path: Path | str,
    *,
    xyzin_path: Path | str,
    direction: CoordinateDirection,
    points: Sequence[ScanPoint],
    engine_command: str = "",
    external_protocol: str = "xyz-energy-gradient-json-v1",
    exploration_policy: PESExplorationPolicy | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    policy = exploration_policy or PESExplorationPolicy(
        retained_group=points[0].retained_group if points else "C1"
    )
    payload = {
        "schema": ORACLE_LINK_SCAN_MANIFEST_SCHEMA,
        "xyzin": str(xyzin_path),
        "coordinate": {
            "kind": direction.kind,
            "label": direction.label,
            "source": direction.source,
        },
        "backend": {
            "engine_command": engine_command,
            "external_protocol": external_protocol,
        },
        "pes_exploration": policy.protocol_payload(),
        "points": [
            {
                "index": point.index,
                "displacement": point.displacement,
                "xyz_path": str(point.xyz_path) if point.xyz_path is not None else None,
                "result_path": str(point.result_path) if point.result_path is not None else None,
                "status": point.status,
                "point_group": point.point_group,
                "symmetry_operation_count": point.symmetry_operation_count,
                "symmetry_projection_status": point.symmetry_projection_status,
                "symmetry_projection_max_displacement_angstrom": (
                    point.symmetry_projection_max_displacement_angstrom
                ),
                "symmetry_projection_rms_displacement_angstrom": (
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            }
            for point in points
        ],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def run_external_scan_points(
    points: Sequence[ScanPoint],
    *,
    engine_command: str,
    timeout: float | None = None,
) -> tuple[PointEvaluationResult, ...]:
    """Run an external point evaluator for prepared scan geometries.

    The command is a shell-like template.  The placeholders ``{xyz}``,
    ``{result}``, ``{index}``, ``{displacement}`` and ``{workdir}`` are expanded
    for each point.  The evaluator must write a JSON result file following
    ``oracle.link.point_result.v1``.
    """

    if not engine_command.strip():
        raise ValueError("external scan needs an engine command")
    results: list[PointEvaluationResult] = []
    for point in points:
        if point.xyz_path is None or point.result_path is None:
            raise ValueError("external scan points need xyz_path and result_path")
        workdir = point.xyz_path.parent
        result_path = point.result_path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = _format_command(
            engine_command,
            xyz=point.xyz_path,
            result=result_path,
            index=point.index,
            displacement=point.displacement,
            workdir=workdir,
        )
        require_authorized_descendant_calculation(
            backend="LINK/external-scan-evaluator",
            input_path=point.xyz_path,
            command=cmd,
            workdir=workdir,
        )
        completed = subprocess.run(
            cmd,
            cwd=workdir,
            check=False,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            results.append(
                PointEvaluationResult(
                    point_index=point.index,
                    displacement=point.displacement,
                    coordinate_key=point.coordinate_key,
                    side=point.side,
                    status="failed",
                    message=completed.stdout.strip(),
                    source=" ".join(cmd),
                    point_group=point.point_group,
                    symmetry_projection_status=point.symmetry_projection_status,
                    symmetry_projection_max_displacement_angstrom=(
                        point.symmetry_projection_max_displacement_angstrom
                    ),
                    symmetry_projection_rms_displacement_angstrom=(
                        point.symmetry_projection_rms_displacement_angstrom
                    ),
                )
            )
            continue
        results.append(read_point_result(result_path, point=point))
    return tuple(results)


def run_qm_scan_points(
    xyzin_path: Path | str,
    points: Sequence[ScanPoint],
    backend: QMScanBackend | str,
    *,
    run_dir: Path | str | None = None,
    exploration_policy: PESExplorationPolicy | None = None,
    scf_state_history: Sequence[object] = (),
    matrix_gfn2_session: object | None = None,
    gfnff_session: object | None = None,
    pyscf_session: object | None = None,
    pyfock_session: object | None = None,
    openmm_session: object | None = None,
    reuse_center_state: bool = False,
) -> tuple[PointEvaluationResult, ...]:
    """Run prepared scan points with a MATRIX electronic-structure adapter.

    This is the in-process equivalent of ``run_external_scan_points`` for
    supported QM backends.  Each point is evaluated in its own work directory,
    then written back to the standard LINK point-result JSON contract.
    """

    geometry = read_xyzin_geometry(Path(xyzin_path))
    spec = backend if isinstance(backend, QMScanBackend) else QMScanBackend(str(backend))
    name = _normalized_backend_name(spec.name)
    if name == "zaff":
        spec = _resolve_zaff_backend(
            spec,
            geometry.atoms,
            geometry.coordinates_angstrom,
        )
        name = _normalized_backend_name(spec.name)
    root = Path(run_dir) if run_dir is not None else _scan_root_from_points(points)
    root.mkdir(parents=True, exist_ok=True)
    prepared_points = tuple(
        input_point
        if exploration_policy is None
        else _scan_point_with_pointwise_symmetry(
            xyzin_path,
            geometry.atoms,
            input_point,
            policy=exploration_policy,
        )
        for input_point in points
    )
    if name == "zaff":
        try:
            zaff_results = _run_zaff_scan_batch(
                spec,
                geometry.atoms,
                prepared_points,
                xyzin_path=xyzin_path,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed point diagnostics
            zaff_results = tuple(
                PointEvaluationResult(
                    point_index=point.index,
                    displacement=point.displacement,
                    status="failed",
                    message=str(exc),
                    source=name,
                )
                for point in prepared_points
            )
        results: list[PointEvaluationResult] = []
        for point, raw_result in zip(prepared_points, zaff_results, strict=True):
            point_dir = root / f"point_{point.index:04d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            result = replace(
                raw_result,
                coordinate_key=point.coordinate_key,
                side=point.side,
                point_group=point.point_group,
                symmetry_projection_status=point.symmetry_projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    point.symmetry_projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            )
            if point.result_path is not None:
                write_point_result(point.result_path, result)
            write_point_result(point_dir / "point.json", result)
            results.append(result)
        return tuple(results)

    center_state_reuse_active = bool(
        reuse_center_state and name in {"gaussian", "gdv"}
    )
    evaluation_points = prepared_points
    active_spec = spec
    if center_state_reuse_active:
        centers = tuple(
            point for point in prepared_points if abs(float(point.displacement)) <= 1.0e-14
        )
        if len(centers) != 1:
            raise ValueError(
                "center-state reuse requires exactly one zero-displacement scan point"
            )
        center = centers[0]
        evaluation_points = (center,) + tuple(
            point for point in prepared_points if point is not center
        )
        # Evaluate the undisplaced geometry first, then force every finite-
        # difference point to start from that accepted electronic state.
        # This keeps symmetry-related +q and -q points on the same SCF branch
        # without changing the Gaussian convergence threshold.
        active_spec = replace(
            spec,
            gaussian_native_checkpoint_restart=True,
            restart_reuse_for_displacements=True,
        )

    results: list[PointEvaluationResult] = []
    center_restart: Path | None = None
    for point in evaluation_points:
        point_dir = root / f"point_{point.index:04d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        point_spec = active_spec
        try:
            if name == "matrix_gfn2" and matrix_gfn2_session is None:
                from .resident_calculators import create_resident_calculator

                matrix_gfn2_session = create_resident_calculator(
                    "matrix_gfn2",
                    geometry.atoms,
                    point.coordinates_angstrom,
                    charge=spec.charge,
                    multiplicity=spec.multiplicity,
                    method=spec.method or "MATRIX-GFN2",
                    processors=spec.processors,
                    solver_mode=spec.matrix_gfn2_solver_mode,
                    linear_scaling_threshold=spec.matrix_gfn2_linear_scaling_threshold,
                    alpb_solvent=spec.matrix_gfn2_alpb_solvent,
                )
            if name in {"gfnff", "gfnff_reference"} and gfnff_session is None:
                gfnff_session = _prepare_gfnff_session(
                    spec, geometry.atoms, point.coordinates_angstrom,
                )
            if (
                name == "pyscf"
                and pyscf_session is None
                and spec.dispersion_contract is None
                and "hessian"
                not in {str(item).strip().casefold() for item in spec.properties}
            ):
                from matrix_pyscf import PySCFResidentCalculator

                accelerator = (
                    str(spec.resolution.get("accelerator", "density_fitting"))
                    if isinstance(spec.resolution, Mapping)
                    else "density_fitting"
                )
                pyscf_session = PySCFResidentCalculator(
                    geometry.atoms,
                    point.coordinates_angstrom,
                    method=spec.method or spec.route or "HF",
                    reference=spec.reference,
                    basis=spec.basis or "sto-3g",
                    charge=spec.charge,
                    multiplicity=spec.multiplicity,
                    density_fitting=(
                        accelerator.strip().casefold() == "density_fitting"
                    ),
                    processors=spec.processors,
                )
            if name == "pyfock" and pyfock_session is None:
                from .pyfock_resident import PyFockResidentSession

                pyfock_session = PyFockResidentSession(
                    geometry.atoms,
                    point.coordinates_angstrom,
                    basis_name=spec.basis or "sto-3g",
                    method=spec.method or spec.route or "HF",
                    charge=spec.charge,
                    processors=spec.processors,
                )
            if name == "openmm" and openmm_session is None:
                from .openmm_resident import OpenMMResidentSession

                openmm_session = OpenMMResidentSession(
                    geometry.atoms,
                    point.coordinates_angstrom,
                    potential=spec.method or spec.route or "aimnet2",
                    charge=spec.charge,
                    multiplicity=spec.multiplicity,
                    bonds=spec.topology_bonds,
                    system_xml=spec.force_field,
                    platform=spec.device,
                )
            if center_state_reuse_active and center_restart is not None:
                point_spec = replace(
                    active_spec,
                    restart_artifact=center_restart,
                    restart_history_artifacts=(center_restart,),
                )
            result = _run_qm_scan_point(
                name,
                point_spec,
                geometry.atoms,
                point,
                point_dir,
                scf_state_history=scf_state_history,
                matrix_gfn2_session=matrix_gfn2_session,
                gfnff_session=gfnff_session,
                pyscf_session=pyscf_session,
                pyfock_session=pyfock_session,
                openmm_session=openmm_session,
            )
            result = _prefer_full_precision_derivative_artifact(result, point, point_dir)
            result = replace(
                result,
                coordinate_key=point.coordinate_key,
                side=point.side,
                point_group=point.point_group,
                symmetry_projection_status=point.symmetry_projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    point.symmetry_projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed point diagnostics
            result = PointEvaluationResult(
                point_index=point.index,
                displacement=point.displacement,
                coordinate_key=point.coordinate_key,
                side=point.side,
                status="failed",
                message=str(exc),
                source=name,
                point_group=point.point_group,
                symmetry_projection_status=point.symmetry_projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    point.symmetry_projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            )
        result = _attach_backend_resolution(result, point_spec)
        if point.result_path is not None:
            write_point_result(point.result_path, result)
        write_point_result(point_dir / "point.json", result)
        results.append(result)
        if center_state_reuse_active and abs(float(point.displacement)) <= 1.0e-14:
            artifact = result.execution.get("restart_artifact")
            if result.status != "completed" or artifact is None:
                raise RuntimeError(
                    "the central scan point did not produce a reusable electronic-state artifact"
                )
            center_restart = Path(str(artifact)).expanduser().resolve()
            if not center_restart.is_file():
                raise FileNotFoundError(
                    f"central electronic-state artifact not found: {center_restart}"
                )
    return tuple(sorted(results, key=lambda item: int(item.point_index)))


def write_full_precision_derivative_artifact(
    path: Path | str,
    *,
    energy_hartree: float,
    gradient_hartree_per_bohr: np.ndarray | None = None,
    hessian_hartree_per_bohr2: np.ndarray | None = None,
    coordinates_angstrom: np.ndarray | None = None,
) -> Path:
    """Write the backend-independent, float64-round-trippable derivative sidecar."""

    target = Path(path)
    payload: dict[str, object] = {
        "schema": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
        "energy_hartree": float(energy_hartree),
    }
    for key, value in (
        ("gradient_hartree_per_bohr", gradient_hartree_per_bohr),
        ("hessian_hartree_per_bohr2", hessian_hartree_per_bohr2),
        ("coordinates_angstrom", coordinates_angstrom),
    ):
        if value is not None:
            array = np.asarray(value, dtype=float)
            if np.any(~np.isfinite(array)):
                raise ValueError(f"full-precision derivative artifact contains non-finite {key}")
            payload[key] = array.tolist()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _prefer_full_precision_derivative_artifact(
    result: PointEvaluationResult,
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    """Apply a backend/wrapper derivative sidecar without decimal truncation."""

    artifact = point_dir / FULL_PRECISION_DERIVATIVE_ARTIFACT_NAME
    if not artifact.is_file():
        return result
    existing_artifact = result.execution.get("derivative_transport_artifact")
    if (
        bool(result.execution.get("derivative_transport_lossless_float64"))
        and existing_artifact is not None
        and Path(str(existing_artifact)).expanduser().resolve() == artifact.resolve()
    ):
        # The adapter already returned the float64 arrays persisted for audit;
        # decoding the same JSON immediately would only duplicate point I/O.
        return result
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("schema") != FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported full-precision derivative artifact schema")
    ncart = point.coordinates_angstrom.size
    gradient_value = payload.get("gradient_hartree_per_bohr")
    gradient = None if gradient_value is None else np.asarray(gradient_value, dtype=float).reshape(-1)
    if gradient is not None and gradient.size != ncart:
        raise ValueError("full-precision derivative gradient has the wrong dimension")
    hessian_value = payload.get("hessian_hartree_per_bohr2")
    hessian = None if hessian_value is None else np.asarray(hessian_value, dtype=float)
    if hessian is not None and hessian.shape != (ncart, ncart):
        raise ValueError("full-precision derivative Hessian has the wrong dimension")
    if gradient is not None and np.any(~np.isfinite(gradient)):
        raise ValueError("full-precision derivative gradient is non-finite")
    if hessian is not None and np.any(~np.isfinite(hessian)):
        raise ValueError("full-precision derivative Hessian is non-finite")
    source_coordinates_value = payload.get("coordinates_angstrom")
    if source_coordinates_value is not None:
        source_coordinates = np.asarray(source_coordinates_value, dtype=float)
        if source_coordinates.shape != point.coordinates_angstrom.shape:
            raise ValueError("full-precision derivative coordinates have the wrong dimension")
        rotation = kabsch_rotation(source_coordinates, point.coordinates_angstrom)
        rotation_gradient = np.zeros(ncart, dtype=float) if gradient is None else gradient
        gradient, hessian = rotate_cartesian_derivatives(
            rotation_gradient,
            rotation,
            hessian,
        )
        if gradient_value is None:
            gradient = None
    energy = float(payload["energy_hartree"])
    if not math.isfinite(energy):
        raise ValueError("full-precision derivative energy is non-finite")
    execution = dict(result.execution)
    source_transport = execution.get("derivative_transport")
    source_lossless = execution.get("derivative_transport_lossless_float64")
    execution.update(
        {
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": (
                "backend_full_precision_sidecar"
                if source_transport is None
                else f"backend_derivative_sidecar:{source_transport}"
            ),
            "derivative_transport_lossless_float64": (
                True if source_lossless is None else bool(source_lossless)
            ),
            "derivative_transport_artifact": str(artifact.resolve()),
        }
    )
    return replace(
        result,
        energy_hartree=energy,
        gradient_hartree_per_bohr=(
            result.gradient_hartree_per_bohr if gradient is None else gradient.reshape(-1)
        ),
        hessian_hartree_per_bohr2=(
            result.hessian_hartree_per_bohr2 if hessian is None else hessian
        ),
        execution=execution,
    )


def point_result_to_json(result: PointEvaluationResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": result.schema,
        "point_index": result.point_index,
        "displacement": result.displacement,
        "coordinate_key": result.coordinate_key,
        "side": result.side,
        "status": result.status,
        "message": result.message,
        "source": result.source,
        "point_group": result.point_group,
        "symmetry_projection_status": result.symmetry_projection_status,
        "symmetry_projection_max_displacement_angstrom": (
            result.symmetry_projection_max_displacement_angstrom
        ),
        "symmetry_projection_rms_displacement_angstrom": (
            result.symmetry_projection_rms_displacement_angstrom
        ),
        "execution": dict(result.execution),
    }
    if result.energy_hartree is not None:
        payload["energy_hartree"] = result.energy_hartree
    if result.gradient_hartree_per_bohr is not None:
        payload["gradient_hartree_per_bohr"] = (
            np.asarray(result.gradient_hartree_per_bohr, dtype=float).reshape(-1).tolist()
        )
    if result.cm5_charges is not None:
        payload["cm5_charges"] = (
            np.asarray(result.cm5_charges, dtype=float).reshape(-1).tolist()
        )
    if result.hessian_hartree_per_bohr2 is not None:
        payload["hessian_hartree_per_bohr2"] = np.asarray(
            result.hessian_hartree_per_bohr2, dtype=float
        ).tolist()
    if result.backend_coordinates_angstrom is not None:
        payload["backend_coordinates_angstrom"] = np.asarray(
            result.backend_coordinates_angstrom, dtype=float
        ).tolist()
        payload["frame_alignment_rms_angstrom"] = result.frame_alignment_rms_angstrom
    return payload


def write_point_result(path: Path | str, result: PointEvaluationResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(target, point_result_to_json(result))
    return target


def read_point_result(path: Path | str, *, point: ScanPoint | None = None) -> PointEvaluationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema", ORACLE_LINK_POINT_RESULT_SCHEMA)
    if schema != ORACLE_LINK_POINT_RESULT_SCHEMA:
        raise ValueError(f"unsupported LINK point-result schema: {schema}")
    point_index = int(payload.get("point_index", point.index if point is not None else 0))
    displacement = float(
        payload.get("displacement", point.displacement if point is not None else 0.0)
    )
    gradient = payload.get("gradient_hartree_per_bohr")
    cm5_charges = payload.get("cm5_charges")
    hessian = payload.get("hessian_hartree_per_bohr2")
    backend_coordinates = payload.get("backend_coordinates_angstrom")
    return PointEvaluationResult(
        point_index=point_index,
        displacement=displacement,
        coordinate_key=str(
            payload.get(
                "coordinate_key",
                point.coordinate_key if point is not None else "",
            )
        ),
        side=str(
            payload.get(
                "side",
                point.side if point is not None else "unspecified",
            )
        ),
        energy_hartree=payload.get("energy_hartree"),
        gradient_hartree_per_bohr=None if gradient is None else np.asarray(gradient, dtype=float),
        cm5_charges=(
            None if cm5_charges is None else np.asarray(cm5_charges, dtype=float)
        ),
        hessian_hartree_per_bohr2=None if hessian is None else np.asarray(hessian, dtype=float),
        backend_coordinates_angstrom=(
            None if backend_coordinates is None else np.asarray(backend_coordinates, dtype=float)
        ),
        frame_alignment_rms_angstrom=float(payload.get("frame_alignment_rms_angstrom", 0.0)),
        status=str(payload.get("status", "completed")),
        message=str(payload.get("message", "")),
        source=str(payload.get("source", path)),
        point_group=str(payload.get("point_group", point.point_group if point is not None else "")),
        symmetry_projection_status=str(
            payload.get(
                "symmetry_projection_status",
                point.symmetry_projection_status if point is not None else "NOT_ANALYZED",
            )
        ),
        symmetry_projection_max_displacement_angstrom=float(
            payload.get(
                "symmetry_projection_max_displacement_angstrom",
                (point.symmetry_projection_max_displacement_angstrom if point is not None else 0.0),
            )
        ),
        symmetry_projection_rms_displacement_angstrom=float(
            payload.get(
                "symmetry_projection_rms_displacement_angstrom",
                (point.symmetry_projection_rms_displacement_angstrom if point is not None else 0.0),
            )
        ),
        execution=dict(payload.get("execution", {})),
        schema=schema,
    )


def write_point_results_jsonl(path: Path | str, results: Iterable[PointEvaluationResult]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(point_result_to_json(result), sort_keys=True) + "\n")
    return target


def read_point_results_jsonl(path: Path | str) -> tuple[PointEvaluationResult, ...]:
    results: list[PointEvaluationResult] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        results.append(
            PointEvaluationResult(
                point_index=int(payload["point_index"]),
                displacement=float(payload["displacement"]),
                coordinate_key=str(payload.get("coordinate_key", "")),
                side=str(payload.get("side", "unspecified")),
                energy_hartree=payload.get("energy_hartree"),
                gradient_hartree_per_bohr=payload.get("gradient_hartree_per_bohr"),
                hessian_hartree_per_bohr2=payload.get("hessian_hartree_per_bohr2"),
                status=str(payload.get("status", "completed")),
                message=str(payload.get("message", "")),
                source=str(payload.get("source", "")),
                point_group=str(payload.get("point_group", "")),
                symmetry_projection_status=str(
                    payload.get("symmetry_projection_status", "NOT_ANALYZED")
                ),
                symmetry_projection_max_displacement_angstrom=float(
                    payload.get("symmetry_projection_max_displacement_angstrom", 0.0)
                ),
                symmetry_projection_rms_displacement_angstrom=float(
                    payload.get("symmetry_projection_rms_displacement_angstrom", 0.0)
                ),
                execution=dict(payload.get("execution", {})),
                schema=str(payload.get("schema", ORACLE_LINK_POINT_RESULT_SCHEMA)),
            )
        )
    return tuple(results)


def finite_difference_derivatives_to_json(
    derivatives: FiniteDifferenceDerivatives,
) -> dict[str, object]:
    return {
        "schema": ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA,
        "coordinate_label": derivatives.coordinate_label,
        "energy_derivatives_hartree": list(derivatives.energy_derivatives_hartree),
        "gradient_derivatives_hartree_per_bohr": [
            np.asarray(item, dtype=float).reshape(-1).tolist()
            for item in derivatives.gradient_derivatives_hartree_per_bohr
        ],
        "hessian_derivatives_hartree_per_bohr2": [
            np.asarray(item, dtype=float).tolist()
            for item in derivatives.hessian_derivatives_hartree_per_bohr2
        ],
        "fit_rank": int(derivatives.fit_rank),
        "residual_norm": float(derivatives.residual_norm),
    }


def write_finite_difference_derivatives(
    path: Path | str,
    derivatives: FiniteDifferenceDerivatives,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(finite_difference_derivatives_to_json(derivatives), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target


def read_finite_difference_derivatives(path: Path | str) -> FiniteDifferenceDerivatives:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema", ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA)
    if schema != ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA:
        raise ValueError(f"unsupported LINK finite-difference derivative schema: {schema}")
    return FiniteDifferenceDerivatives(
        coordinate_label=str(payload["coordinate_label"]),
        energy_derivatives_hartree=tuple(
            float(value) for value in payload.get("energy_derivatives_hartree", ())
        ),
        gradient_derivatives_hartree_per_bohr=tuple(
            np.asarray(item, dtype=float)
            for item in payload.get("gradient_derivatives_hartree_per_bohr", ())
        ),
        hessian_derivatives_hartree_per_bohr2=tuple(
            np.asarray(item, dtype=float)
            for item in payload.get("hessian_derivatives_hartree_per_bohr2", ())
        ),
        fit_rank=int(payload.get("fit_rank", 0)),
        residual_norm=float(payload.get("residual_norm", 0.0)),
    )


def finite_difference_derivatives(
    results: Sequence[PointEvaluationResult],
    *,
    coordinate_label: str,
    max_order: int = 4,
) -> FiniteDifferenceDerivatives:
    """Fit local derivatives with respect to the scan displacement."""

    completed = [item for item in results if item.status == "completed"]
    if len(completed) < 2:
        raise ValueError("at least two completed points are needed for finite differences")
    completed = list(
        _validated_finite_difference_stencil(completed, coordinate_label=coordinate_label)
    )
    x = np.asarray([item.displacement for item in completed], dtype=float)
    if len(set(float(value) for value in x)) != x.size:
        raise ValueError("scan displacements must be unique")
    energy_derivatives: tuple[float, ...] = ()
    rank = 0
    residual_norm = 0.0
    energies = [item.energy_hartree for item in completed]
    if all(value is not None for value in energies):
        fit = fit_taylor_derivatives(
            x, np.asarray(energies, dtype=float), max_order=max_order
        )
        energy_derivatives = tuple(float(value) for value in fit.derivatives)
        rank = fit.rank
        residual_norm = fit.residual_norm
    gradient_derivatives = _array_derivatives(
        completed,
        x,
        attr="gradient_hartree_per_bohr",
        max_order=max_order,
    )
    hessian_derivatives = _array_derivatives(
        completed,
        x,
        attr="hessian_hartree_per_bohr2",
        max_order=max_order,
    )
    return FiniteDifferenceDerivatives(
        coordinate_label=coordinate_label,
        energy_derivatives_hartree=energy_derivatives,
        gradient_derivatives_hartree_per_bohr=gradient_derivatives,
        hessian_derivatives_hartree_per_bohr2=hessian_derivatives,
        fit_rank=int(rank),
        residual_norm=float(residual_norm),
    )


def _validated_finite_difference_stencil(
    points: Sequence[PointEvaluationResult],
    *,
    coordinate_label: str,
) -> tuple[PointEvaluationResult, ...]:
    """Validate and deterministically order one complete signed stencil."""

    label = str(coordinate_label).strip()
    keys = {item.coordinate_key for item in points}
    if not label or keys != {label}:
        raise ValueError(
            "finite-difference points must carry one immutable coordinate_key matching "
            f"{coordinate_label!r}"
        )
    centers = [item for item in points if item.side == "center"]
    plus = {float(item.displacement): item for item in points if item.side == "plus"}
    minus = {abs(float(item.displacement)): item for item in points if item.side == "minus"}
    unspecified = [item for item in points if item.side == "unspecified"]
    if unspecified:
        raise ValueError("finite-difference stencil contains points with unspecified side")
    if len(centers) != 1:
        raise ValueError("finite-difference stencil requires exactly one center point")
    if len(plus) + len(minus) != len(points) - 1:
        raise ValueError("finite-difference stencil contains duplicate signed displacements")
    if plus and minus and set(plus) != set(minus):
        missing_plus = sorted(set(minus) - set(plus))
        missing_minus = sorted(set(plus) - set(minus))
        raise ValueError(
            "incomplete two-sided finite-difference stencil"
            f" (missing plus={missing_plus}, missing minus={missing_minus})"
        )
    if not plus and not minus:
        raise ValueError("finite-difference stencil has no displaced points")
    return tuple(sorted(points, key=lambda item: (float(item.displacement), item.point_index)))


def symmetric_displacements(step: float, points_each_side: int) -> tuple[float, ...]:
    count = int(points_each_side)
    if count <= 0:
        raise ValueError("points_each_side must be positive")
    h = float(step)
    if h <= 0.0:
        raise ValueError("step must be positive")
    values = [float(i * h) for i in range(-count, count + 1)]
    return tuple(values)


def _coordinate_index(
    coordinate: str | int, labels: tuple[str, ...], names: tuple[str, ...]
) -> int:
    if isinstance(coordinate, int):
        index = int(coordinate)
        if index < 1 or index > len(labels):
            raise ValueError(f"GIC index {index} outside 1..{len(labels)}")
        return index - 1
    wanted = str(coordinate)
    for index, label in enumerate(labels):
        if label == wanted:
            return index
    for index, name in enumerate(names):
        if name == wanted:
            return index
    raise ValueError(f"unknown GIC coordinate {wanted!r}")


def _write_xyz(path: Path, atoms: tuple[str, ...], coords: np.ndarray, comment: str) -> None:
    lines = [str(len(atoms)), comment]
    for atom, xyz in zip(atoms, coords):
        lines.append(f"{atom:2s} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan_root_from_points(points: Sequence[ScanPoint]) -> Path:
    for point in points:
        if point.result_path is not None:
            return Path(point.result_path).parent
        if point.xyz_path is not None:
            return Path(point.xyz_path).parent
    return Path.cwd()


def _normalized_backend_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "gaussian": "gaussian",
        "g16": "gaussian",
        "gdv": "gaussian",
        "gaussian16": "gaussian",
        "fragmentoniom": "fragment_oniom",
        "cappedsubtractive": "capped_subtractive",
        "residentoniom": "capped_subtractive",
        "cfour": "cfour",
        "orca": "orca",
        "nwchem": "nwchem",
        "turbomole": "turbomole",
        "tmole": "turbomole",
        "molpro": "molpro",
        "mrcc": "mrcc",
        "xtb": "xtb",
        "matrixgfn2": "matrix_gfn2",
        "gfn2xtb": "matrix_gfn2",
        "gfn2zaff": "gfn2_zaff_composite",
        "matrixgfn2zaff": "gfn2_zaff_composite",
        "gfn2zaffcomposite": "gfn2_zaff_composite",
        "gfn2zaffcompositetwomorse": "gfn2_zaff_composite",
        "gfn2zaffcompositetwomorseenergyxh": "gfn2_zaff_composite",
        "gfnff": "gfnff",
        "gfn2ff": "gfnff",
        "gfnffreference": "gfnff_reference",
        "libxtbreference": "gfnff_reference",
        "pyscf": "pyscf",
        "std2": "std2",
        "pyfock": "pyfock",
        "openmm": "openmm",
        "openmmml": "openmm",
        "et": "et",
        "zaff": "zaff",
        "ellipsoidalcpcm": "ellipsoidal_cpcm",
        "ellipsoidalasc": "ellipsoidal_cpcm",
        "cpcmellipsoid": "ellipsoidal_cpcm",
        "external": "external",
        "linearcomposite": "linear_composite",
        "composite": "linear_composite",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported QM scan backend: {name}") from exc


def _resolve_zaff_backend(
    spec: QMScanBackend,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    cm5_charges: Sequence[float] | None = None,
    mayer_bond_orders: Mapping[tuple[int, int], float] | None = None,
    charge_source: str = "",
    bond_order_source: str = "",
) -> QMScanBackend:
    """Resolve an existing ZAFF artifact without invoking ARCHITECT."""

    if _normalized_backend_name(spec.name) != "zaff" or spec.resolution is not None:
        return spec
    del cm5_charges, mayer_bond_orders, charge_source, bond_order_source
    from matrix_zaff import (
        explicit_zaff_resolution,
        resolve_zaff_force_field,
    )

    resolution = (
        explicit_zaff_resolution(spec.force_field)
        if spec.force_field is not None
        else resolve_zaff_force_field(
            atoms,
            coordinates_angstrom,
            charge=spec.charge,
            multiplicity=spec.multiplicity,
            library=spec.zaff_library,
        )
    )
    provenance = resolution.to_dict()
    if resolution.force_field_path is not None:
        return replace(
            spec,
            force_field=resolution.force_field_path,
            resolution=provenance,
        )
    raise ValueError(
        "ZAFF has no compiled model for this system; provide a ZAFF artifact "
        "or explicitly select native gfnff with a complete ARCHITECT force_field. "
        "Automatic fallback to libxtb or external xTB/GFN-FF is disabled."
    )


def _attach_backend_resolution(
    result: PointEvaluationResult,
    spec: QMScanBackend,
) -> PointEvaluationResult:
    if spec.resolution is None:
        return result
    execution = dict(result.execution)
    execution["force_field_resolution"] = dict(spec.resolution)
    source = result.source
    if spec.resolution.get("selection") == "GFN_FF_FALLBACK":
        source = f"{source} [ZAFF fallback: GFN-FF]"
    return replace(result, source=source, execution=execution)


def _run_qm_scan_point(
    name: str,
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
    *,
    scf_state_history: Sequence[object] = (),
    matrix_gfn2_session: object | None = None,
    gfnff_session: object | None = None,
    pyscf_session: object | None = None,
    pyfock_session: object | None = None,
    openmm_session: object | None = None,
) -> PointEvaluationResult:
    if name == "linear_composite":
        return _run_linear_composite_scan_point(spec, atoms, point, point_dir)
    if name == "gaussian":
        return _run_gaussian_scan_point(spec, atoms, point, point_dir)
    if name == "fragment_oniom":
        return _run_fragment_oniom_scan_point(spec, atoms, point, point_dir)
    if name == "orca":
        return _run_orca_scan_point(spec, atoms, point, point_dir)
    if name == "nwchem":
        return _run_nwchem_scan_point(spec, atoms, point, point_dir)
    if name == "turbomole":
        return _run_turbomole_scan_point(spec, atoms, point, point_dir)
    if name == "molpro":
        return _run_molpro_scan_point(spec, atoms, point, point_dir)
    if name == "mrcc":
        return _run_mrcc_scan_point(spec, atoms, point, point_dir)
    if name == "cfour":
        return _run_cfour_scan_point(spec, atoms, point, point_dir)
    if name == "xtb":
        return _run_xtb_scan_point(spec, atoms, point, point_dir)
    if name == "matrix_gfn2":
        return _run_matrix_gfn2_scan_point(spec, atoms, point, matrix_gfn2_session)
    if name in {"gfnff", "gfnff_reference"}:
        return _run_gfnff_scan_point(spec, atoms, point, gfnff_session)
    if name == "openmm":
        return _run_openmm_scan_point(spec, point, openmm_session)
    if name == "pyscf":
        if pyscf_session is not None and "hessian" not in {
            str(item).strip().casefold() for item in spec.properties
        }:
            return _run_resident_pyscf_scan_point(
                spec,
                atoms,
                point,
                pyscf_session,
                scf_state_history=scf_state_history,
            )
        return _run_pyscf_scan_point(
            spec, atoms, point, point_dir, scf_state_history=scf_state_history
        )
    if name == "pyfock":
        return _run_pyfock_scan_point(
            spec,
            atoms,
            point,
            point_dir,
            scf_state_history=scf_state_history,
            session=pyfock_session,
        )
    if name == "et":
        return _run_et_scan_point(spec, atoms, point, point_dir)
    if name == "zaff":
        return _run_zaff_scan_point(spec, atoms, point)
    if name == "ellipsoidal_cpcm":
        return _run_ellipsoidal_cpcm_scan_point(spec, atoms, point)
    raise ValueError(f"unsupported QM scan backend: {name}")


def _run_ellipsoidal_cpcm_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
) -> PointEvaluationResult:
    """Evaluate the analytic ellipsoidal ASC operator as an additive LINK term.

    The cavity and charge model are supplied through ``spec.resolution``.  This
    deliberately keeps the operator independent of the electronic-structure
    backend, so a Gaussian/GDV vacuum term can be combined with it through the
    ordinary LINK ``linear_composite`` surface.
    """
    cfg = dict(spec.resolution or {})
    if "configuration" in cfg:
        config_path = Path(str(cfg["configuration"])).expanduser().resolve()
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    required = ("center_angstrom", "semiaxes_angstrom", "charges")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError("ellipsoidal CPCM configuration lacks: " + ", ".join(missing))
    xyz = np.asarray(point.coordinates_angstrom, dtype=float)
    charges = np.asarray(cfg["charges"], dtype=float).reshape(-1)
    charge_model = str(cfg.get("charge_model", "CM5")).strip().upper()
    if charge_model not in {"CM5", "SCC", "EEQ", "GFN2_SCC_EEQ"}:
        raise ValueError(
            "ellipsoidal CPCM charge_model must be CM5 for advanced QM or "
            "SCC/EEQ for MATRIX-GFN2 and semiempirical QM/MM"
        )
    if charges.size != len(atoms):
        raise ValueError("ellipsoidal CPCM charges must match the LINK atom count")
    center = np.asarray(cfg["center_angstrom"], dtype=float) * ANGSTROM_TO_BOHR
    axes = np.asarray(cfg["semiaxes_angstrom"], dtype=float) * ANGSTROM_TO_BOHR
    rotation = np.asarray(cfg.get("rotation", np.eye(3)), dtype=float).reshape(3, 3)
    degree = int(cfg.get("maximum_degree", cfg.get("degree", 8)))
    polar_order = int(cfg.get("polar_order", 2 * degree + 4))
    azimuthal_order = int(cfg.get("azimuthal_order", 4 * degree + 8))
    field = _compiled_ellipsoidal_asc_field(
        tuple(center),
        tuple(axes),
        tuple(rotation.reshape(-1)),
        float(cfg.get("dielectric", 78.4)),
        degree,
        polar_order,
        azimuthal_order,
    )
    confinement = field.confinement
    exponents = cfg.get("gaussian_exponents_bohr2")
    result = field.evaluate(
        xyz * ANGSTROM_TO_BOHR,
        charges,
        gaussian_exponents_bohr2=(None if exponents is None else np.asarray(exponents, dtype=float)),
    )
    gradient = np.asarray(result.coordinate_gradient_hartree_per_bohr, dtype=float)
    hessian = None
    if "hessian" in {str(item).strip().casefold() for item in spec.properties}:
        ncart = xyz.size
        zero_shape = np.zeros(9, dtype=float)

        def hessian_column(column: int) -> np.ndarray:
            direction = np.zeros(ncart, dtype=float)
            direction[column] = 1.0
            hv, _ = field.hessian_vector_product(
                xyz * ANGSTROM_TO_BOHR,
                charges,
                direction,
                zero_shape,
                shape_parameters=field.shape_from_confinement(),
                gaussian_exponents_bohr2=(None if exponents is None else np.asarray(exponents, dtype=float)),
            )
            return np.asarray(hv, dtype=float).reshape(-1)

        hessian_workers = min(max(1, int(spec.processors)), ncart)
        if hessian_workers > 1:
            # Each HVP invokes dense SciPy/BLAS kernels.  Restrict the nested
            # BLAS pools to one thread while the independent Cartesian
            # columns are distributed by LINK; otherwise OpenBLAS can both
            # oversubscribe and corrupt its per-thread scratch workspace.
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1, user_api="blas"):
                with ThreadPoolExecutor(max_workers=hessian_workers) as pool:
                    columns = list(pool.map(hessian_column, range(ncart)))
        else:
            columns = [hessian_column(column) for column in range(ncart)]
        hessian = np.column_stack(columns)
        hessian = 0.5 * (hessian + hessian.T)
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        status="completed",
        energy_hartree=float(result.energy_hartree),
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        backend_coordinates_angstrom=xyz,
        source="ellipsoidal ASC CPCM",
        execution={
            "operator": "EllipsoidalASCReactionField",
            "cavity_shape": confinement.shape,
            "dielectric": confinement.dielectric,
            "maximum_degree": confinement.maximum_degree,
            "charge_model": charge_model,
            "boundary_residual_rms_hartree_per_e": result.boundary_residual_rms_hartree_per_e,
            "shape_gradient_available": True,
        },
    )


@lru_cache(maxsize=16)
def _compiled_ellipsoidal_asc_field(
    center_bohr: tuple[float, ...],
    semiaxes_bohr: tuple[float, ...],
    rotation_flat: tuple[float, ...],
    dielectric: float,
    maximum_degree: int,
    polar_order: int,
    azimuthal_order: int,
):
    """Cache the geometry-independent spectral factorization of a fixed cavity."""
    from matrix_epoc.cpcm import CPCMConfinement, EllipsoidalASCReactionField

    confinement = CPCMConfinement(
        center_bohr=np.asarray(center_bohr, dtype=float),
        semiaxes_bohr=np.asarray(semiaxes_bohr, dtype=float),
        rotation=np.asarray(rotation_flat, dtype=float).reshape(3, 3),
        dielectric=float(dielectric),
        maximum_degree=int(maximum_degree),
        polar_order=int(polar_order),
        azimuthal_order=int(azimuthal_order),
    )
    return EllipsoidalASCReactionField.compile(confinement)


def _run_linear_composite_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    """Evaluate and assemble a signed multi-backend energy surface.

    Analytic Cartesian derivatives are retained as a partial gradient.  Terms
    that only provide energies are *not* differentiated here: the optimizer
    differentiates their signed sum once, along the totally symmetric SONIC
    coordinates, using the frozen one-sided PCS2/PPCS2 contract.
    """

    requested_workers = min(
        max(1, int(spec.composite_parallel_workers)),
        len(spec.composite_terms),
    )
    parent = authorized_parent_plan_from_environment()
    available_processors = max(
        1,
        int(
            parent.resources.threads_per_job
            if parent is not None
            else max(term.backend.processors for term in spec.composite_terms)
        ),
    )
    available_memory_gb = (
        float(parent.resources.memory_per_job_gb)
        if parent is not None
        else max(
            (float(term.backend.memory_gb) for term in spec.composite_terms if term.backend.memory_gb),
            default=1.0,
        )
    )
    workers = min(
        requested_workers,
        available_processors,
        max(1, int(math.floor(available_memory_gb))),
    )
    processors_per_term = max(1, available_processors // workers)
    memory_per_term_gb = max(1, int(math.floor(available_memory_gb / workers)))

    def evaluate_term(term: QMBackendTerm) -> tuple[QMBackendTerm, PointEvaluationResult]:
        term_dir = point_dir / term.label
        result = (
            _read_completed_composite_term(term, atoms, point, term_dir)
            if spec.resume_completed_composite_terms
            else None
        )
        if result is None:
            term_backend = term.backend
            if workers > 1:
                term_backend = replace(
                    term_backend,
                    processors=processors_per_term,
                    memory_gb=memory_per_term_gb,
                )
            result = _run_qm_scan_point(
                _normalized_backend_name(term_backend.name),
                term_backend,
                atoms,
                point,
                term_dir,
            )
            if result.status == "completed":
                _write_completed_composite_term(term, atoms, point, term_dir, result)
        return term, result

    # Advanced-QM ASC terms consume CM5 populations produced at the same
    # geometry.  MATRIX-GFN2 and semiempirical QM/MM instead use their native
    # SCC/EEQ charges and must not trigger a Hirshfeld/CM5 grid calculation.
    has_asc = any(_normalized_backend_name(t.backend.name) == "ellipsoidal_cpcm" for t in spec.composite_terms)
    if has_asc:
        ordered = sorted(spec.composite_terms, key=lambda t: _normalized_backend_name(t.backend.name) == "ellipsoidal_cpcm")
        term_results = []
        cm5 = None
        for term in ordered:
            if _normalized_backend_name(term.backend.name) == "ellipsoidal_cpcm" and cm5 is not None:
                cfg_path = term.backend.resolution.get("configuration") if term.backend.resolution else None
                if cfg_path:
                    cfg = json.loads(Path(str(cfg_path)).read_text())
                    update_mode = str(cfg.get("charge_update_mode", "fixed")).casefold()
                    if update_mode in {"backend_cm5_each_point", "self_consistent_cm5"}:
                        cfg["charges"] = np.asarray(cm5, dtype=float).tolist()
                        cfg["charge_model"] = "CM5"
                        dyn = point_dir / term.label / "cm5_dynamic_configuration.json"
                        dyn.parent.mkdir(parents=True, exist_ok=True)
                        dyn.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                        term = QMBackendTerm(
                            term.label,
                            term.coefficient,
                            replace(
                                term.backend,
                                resolution={
                                    **dict(term.backend.resolution or {}),
                                    "configuration": str(dyn),
                                },
                            ),
                        )
            item = evaluate_term(term)
            term_results.append(item)
            if item[1].cm5_charges is not None:
                cm5 = item[1].cm5_charges
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            term_results = list(pool.map(evaluate_term, spec.composite_terms))
    else:
        term_results = [evaluate_term(term) for term in spec.composite_terms]
    if any(item.status != "completed" for _, item in term_results):
        failed = next(item for _, item in term_results if item.status != "completed")
        return PointEvaluationResult(
            point_index=point.index,
            displacement=point.displacement,
            status="failed",
            message=f"composite term failed: {failed.message}",
            source="linear_composite",
            execution={"composite_terms": [{"label": t.label, "coefficient": t.coefficient} for t, _ in term_results]},
        )
    if any(item.energy_hartree is None for _, item in term_results):
        raise ValueError("every composite term must provide an energy")
    energy = sum(term.coefficient * float(item.energy_hartree) for term, item in term_results)
    numerical_gradient_terms = [
        term.label
        for term, item in term_results
        if item.gradient_hartree_per_bohr is None
    ]
    analytic_terms = [
        (term, item)
        for term, item in term_results
        if item.gradient_hartree_per_bohr is not None
    ]
    analytic_partial = None
    if analytic_terms:
        analytic_partial = sum(
            term.coefficient * np.asarray(item.gradient_hartree_per_bohr, dtype=float)
            for term, item in analytic_terms
        )
    gradient = analytic_partial if not numerical_gradient_terms else None
    numerical_energy = sum(
        term.coefficient * float(item.energy_hartree)
        for term, item in term_results
        if item.gradient_hartree_per_bohr is None
    )
    execution = {
        "energy_evaluations": (
            sum(int(item.execution.get("energy_evaluations", 1)) for _, item in term_results)
        ),
        "gradient_evaluations": (
            sum(int(item.execution.get("gradient_evaluations", 0)) for _, item in term_results)
        ),
        "fd_displacements": 0,
        "composite_parallel_workers": workers,
        "composite_parallel_worker_allocation": {
            "requested": requested_workers,
            "available_processors_per_point": available_processors,
            "available_memory_gb_per_point": available_memory_gb,
            "processors_per_term": processors_per_term,
            "memory_gb_per_term": memory_per_term_gb,
            "source": "authorized-parent-plan" if parent is not None else "backend-specification",
        },
        "numerical_gradient_terms": numerical_gradient_terms,
        "composite_numerical_energy_hartree": numerical_energy,
        "composite_partial_analytic_gradient_hartree_per_bohr": (
            None if analytic_partial is None else np.asarray(analytic_partial).reshape(-1).tolist()
        ),
        "composite_gradient_assembly": (
            "analytic_linear_composite"
            if not numerical_gradient_terms
            else (
                "one_sided_totally_symmetric_SONIC_numerical_terms_plus_"
                "analytic_cartesian_terms"
            )
        ),
        "composite_terms": [
            {
                "label": term.label,
                "coefficient": term.coefficient,
                "source": item.source,
                "status": item.status,
                "energy_hartree": float(item.energy_hartree),
                "derivative_source": (
                    "analytic" if item.gradient_hartree_per_bohr is not None else "SONIC_one_sided"
                ),
                "restart_reused": bool(item.execution.get("restart_reused", False)),
                "restart_artifact": item.execution.get("restart_artifact"),
                "restart_projection": item.execution.get("restart_projection"),
                "restart_reuse_for_displacements": bool(
                    item.execution.get("restart_reuse_for_displacements", False)
                ),
            }
            for term, item in term_results
        ],
    }
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        backend_coordinates_angstrom=point.coordinates_angstrom,
        status="completed",
        source="linear_composite",
        execution=execution,
    )


def _composite_restart_json_value(value: object) -> object:
    if is_dataclass(value):
        return {
            item.name: _composite_restart_json_value(getattr(value, item.name))
            for item in fields(value)
            if item.name != "resume_completed_composite_terms"
        }
    if isinstance(value, Mapping):
        return {
            str(key): _composite_restart_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_composite_restart_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _composite_term_restart_contract(
    term: QMBackendTerm,
    atoms: tuple[str, ...],
    point: ScanPoint,
) -> dict[str, object]:
    backend_payload = _composite_restart_json_value(term.backend)
    backend_sha256 = hashlib.sha256(
        json.dumps(
            backend_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    coordinates = np.ascontiguousarray(
        np.asarray(point.coordinates_angstrom, dtype="<f8")
    )
    geometry_digest = hashlib.sha256()
    geometry_digest.update("\0".join(atoms).encode("utf-8"))
    geometry_digest.update(str(coordinates.shape).encode("ascii"))
    geometry_digest.update(coordinates.tobytes())
    return {
        "schema": COMPOSITE_TERM_RESTART_SCHEMA,
        "term_label": term.label,
        "term_coefficient": float(term.coefficient),
        "backend_sha256": backend_sha256,
        "geometry_sha256": geometry_digest.hexdigest(),
        "point_index": int(point.index),
        "point_displacement": float(point.displacement),
    }


def _write_completed_composite_term(
    term: QMBackendTerm,
    atoms: tuple[str, ...],
    point: ScanPoint,
    term_dir: Path,
    result: PointEvaluationResult,
) -> Path:
    target = term_dir / COMPOSITE_TERM_RESTART_NAME
    payload = point_result_to_json(result)
    payload["composite_term_restart"] = _composite_term_restart_contract(
        term,
        atoms,
        point,
    )
    atomic_json_write(target, payload)
    return target


def _read_completed_composite_term(
    term: QMBackendTerm,
    atoms: tuple[str, ...],
    point: ScanPoint,
    term_dir: Path,
) -> PointEvaluationResult | None:
    source = term_dir / COMPOSITE_TERM_RESTART_NAME
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid composite-term restart file: {source}") from exc
    expected = _composite_term_restart_contract(term, atoms, point)
    if payload.get("composite_term_restart") != expected:
        raise RuntimeError(
            f"composite-term restart identity changed for {term.label!r}: {source}"
        )
    result = read_point_result(source, point=point)
    if result.status != "completed" or result.energy_hartree is None:
        raise RuntimeError(f"composite-term restart is incomplete: {source}")
    return replace(
        result,
        execution={
            **result.execution,
            "restart_reused": True,
            "restart_source": str(source.resolve()),
            "process_launches": 0,
            "wall_time_seconds_backend": 0.0,
            "wall_time_seconds_total": 0.0,
        },
    )


def qm_scan_backend_from_dict(payload: Mapping[str, object]) -> QMScanBackend:
    """Reconstruct a backend, including recursively nested composite terms."""

    data = dict(payload)
    terms = data.pop("composite_terms", ())
    data["composite_terms"] = tuple(
        QMBackendTerm(
            str(item["label"]),
            float(item["coefficient"]),
            qm_scan_backend_from_dict(item["backend"]),
        )
        if isinstance(item, Mapping)
        else item
        for item in terms
    )
    for key in (
        "extra_args",
        "oniom_high_atoms",
        "oniom_atom_types",
        "restart_history_artifacts",
        "topology_bonds",
    ):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])
    if isinstance(data.get("properties"), list):
        data["properties"] = tuple(data["properties"])
    return QMScanBackend(**data)


def _run_zaff_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
) -> PointEvaluationResult:
    """Evaluate one point through the independent resident ZAFF backend."""

    return _run_zaff_scan_batch(spec, atoms, (point,))[0]


def _run_zaff_scan_batch(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    points: Sequence[ScanPoint],
    *,
    xyzin_path: Path | str | None = None,
) -> tuple[PointEvaluationResult, ...]:
    from matrix_engines import DerivativeOrder, PotentialSystem
    from matrix_zaff import ZaffBackend

    if spec.force_field is None:
        raise ValueError("ZAFF backend needs a compiled artifact path")
    properties = tuple(spec.properties or ("energy", "gradient"))
    requested = set(properties)
    unsupported = requested - {"energy", "gradient", "hessian"}
    if unsupported:
        raise ValueError(f"unsupported ZAFF properties: {sorted(unsupported)}")
    order = (
        DerivativeOrder.HESSIAN
        if "hessian" in requested
        else DerivativeOrder.GRADIENT
        if "gradient" in requested
        else DerivativeOrder.ENERGY
    )
    session = ZaffBackend().prepare(
        PotentialSystem(
            atoms=tuple(atoms),
            charge=spec.charge,
            multiplicity=spec.multiplicity,
        ),
        model=str(spec.force_field),
        options={
            "zoom_level": spec.zaff_zoom_level,
            "xyzin": xyzin_path,
        },
    )
    evaluations = session.evaluate_batch(
        tuple(point.coordinates_angstrom for point in points),
        derivative_order=order,
    )
    return tuple(
        _attach_backend_resolution(
            PointEvaluationResult(
                point_index=point.index,
                displacement=point.displacement,
                energy_hartree=(result.energy_hartree if "energy" in requested else None),
                gradient_hartree_per_bohr=(
                    result.gradient_hartree_per_bohr if "gradient" in requested else None
                ),
                hessian_hartree_per_bohr2=(
                    result.hessian_hartree_per_bohr2 if "hessian" in requested else None
                ),
                backend_coordinates_angstrom=point.coordinates_angstrom,
                source=f"ZAFF {Path(spec.force_field)}",
                execution={
                    **dict(result.execution),
                    "derivative_transport_contract": (
                        FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA
                    ),
                    "derivative_transport": "zaff_in_memory_float64",
                    "derivative_transport_lossless_float64": True,
                },
            ),
            spec,
        )
        for point, result in zip(points, evaluations, strict=True)
    )


def _run_gaussian_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_core import (
        CalculationResources,
        authorize_calculation_launch,
        authorized_parent_plan_from_environment,
        build_calculation_launch_plan,
    )
    from matrix_gaussian import (
        read_gaussian_log_geometry,
        read_gaussian_point_result,
        run_gaussian_job,
        write_gaussian_point_input,
        write_gaussian_oniom_point_input,
    )
    from matrix_gaussian.electronic import (
        GAUSSIAN_EOM_STATE_FINGERPRINT_REPRESENTATION,
        gaussian_state_fingerprint_data,
        gaussian_state_reference_energy_hartree,
        write_gaussian_state_fingerprint_archive,
    )

    point_adapter_started = time.perf_counter()
    requested_hessian = "hessian" in {str(x).lower() for x in spec.properties}
    cartesian_hessian = None
    route = _gaussian_route(spec)
    requested_mode = str(spec.gradient_mode).strip().casefold()
    ensure_force = requested_hessian or requested_mode not in {
        "energy",
        "numerical",
        "link-numerical",
    }
    guess_preparation = None
    guess_preparation_failure = ""
    guess_preparation_elapsed = 0.0
    initial_density_checkpoint = None
    history = tuple(Path(item).expanduser().resolve() for item in spec.restart_history_artifacts)
    if spec.restart_artifact is not None:
        point_restart = Path(spec.restart_artifact).expanduser().resolve()
        if point_restart not in history:
            history = (*history, point_restart)
    density_only_seed = False
    native_checkpoint = bool(spec.gaussian_native_checkpoint_restart)
    if (native_checkpoint or spec.reuse_density_projection or spec.initial_scf_density_artifact is not None) and not history and spec.initial_scf_density_artifact is not None:
        history = (Path(spec.initial_scf_density_artifact).expanduser().resolve(),)
        density_only_seed = True
    if density_only_seed and history:
        initial_density_checkpoint = history[-1]
        if not initial_density_checkpoint.is_file():
            raise FileNotFoundError(
                f"initial Gaussian density checkpoint not found: {initial_density_checkpoint}"
            )
        if re.search(r"\bguess\s*=", route, flags=re.IGNORECASE):
            raise ValueError(
                "Gaussian initial density projection owns Guess=; remove the user Guess keyword"
            )
        # Let the production job project the lower-rung occupied space and
        # rebuild the target-level Fock in one Gaussian invocation.
        route = f"{route} Guess=Read" if native_checkpoint else f"{route} Guess=(Read,NoFock,Translate,Project)"
    elif native_checkpoint and history:
        initial_density_checkpoint = history[-1]
        if not initial_density_checkpoint.is_file():
            raise FileNotFoundError(f"Gaussian restart checkpoint not found: {initial_density_checkpoint}")
        if re.search(r"\bguess\s*=", route, flags=re.IGNORECASE):
            raise ValueError("Gaussian native checkpoint restart owns Guess=; remove the user Guess keyword")
        route = f"{route} Guess=Read"
    elif spec.reuse_density_projection and history:
        guess_started = time.perf_counter()
        try:
            guess_preparation = _prepare_gaussian_joint_scf_guess(
                spec,
                atoms,
                point.coordinates_angstrom,
                history,
                point_dir / "scf_guess",
                density_only=density_only_seed,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            guess_preparation_failure = f"{type(exc).__name__}:{exc}"
        guess_preparation_elapsed = time.perf_counter() - guess_started
        if guess_preparation is not None:
            if re.search(r"\bguess\s*=", route, flags=re.IGNORECASE):
                raise ValueError(
                    "Gaussian joint density/Fock reuse owns Guess=; remove the user Guess keyword"
                )
            route = (
                f"{route} Guess=(Read,NoProject)"
                if density_only_seed
                else f"{route} Guess=(Read,Fock,NoProject)"
            )
    # Gaussian's SCF checkpoint restart and coupled-cluster amplitude restart
    # are separate contracts.  Link 913 IOp(77)=1 stores the converged
    # ground-state amplitudes.  At a subsequent geometry Link 804 IOp(24)=1
    # recovers and transforms those amplitudes; Link 804 then tells Link 913
    # that a genuine amplitude guess is available.  Merely setting the tens
    # digit of Link 913 IOp(77) bypasses that transformation and can therefore
    # produce a false-positive "Amplitudes recovered" message while starting
    # from the ordinary first-order amplitudes.  Limit this transport to native
    # checkpoint restarts of conventional ground-state CCSD/CCSD(T).
    cc_amplitude_checkpoint = bool(
        native_checkpoint
        and int(spec.electronic_state) == 0
        and re.search(r"\bccsd(?:\(t\))?\b", route, flags=re.IGNORECASE)
    )
    cc_amplitude_save_iop = None
    cc_amplitude_transform_iop = None
    if cc_amplitude_checkpoint:
        if re.search(
            r"\biop\s*\(\s*(?:8\s*/\s*24|9\s*/\s*77)",
            route,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "Gaussian CC amplitude restart owns IOp(8/24) and IOp(9/77); "
                "remove the user IOp"
            )
        cc_amplitude_save_iop = 1
        route = f"{route} IOp(9/77={cc_amplitude_save_iop})"
        if history:
            cc_amplitude_transform_iop = 1
            route = f"{route} IOp(8/24={cc_amplitude_transform_iop})"
    if ensure_force and not requested_hessian and not re.search(r"\bforce\b", route, flags=re.IGNORECASE):
        route = f"{route} Force"
    route_override = _gaussian_route_override(spec, route)
    writer = (
        write_gaussian_oniom_point_input if spec.oniom_high_atoms else write_gaussian_point_input
    )
    writer_kwargs = {}
    if spec.oniom_high_atoms:
        writer_kwargs = {
            "high_atoms": spec.oniom_high_atoms,
            "atom_types": spec.oniom_atom_types,
        }
    if spec.gaussian_connectivity_bonds is not None:
        writer_kwargs["connectivity_bonds"] = spec.gaussian_connectivity_bonds
    writer_kwargs["ensure_force"] = ensure_force and not requested_hessian
    input_path = writer(
        point_dir / "gauin.gjf",
        atoms,
        point.coordinates_angstrom,
        route=route,
        title=f"MATRIX LINK point {point.index} {point.displacement:.12g}",
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        link0=tuple(
            item
            for item in (
                "%Chk=gauin.chk",
                (
                    f"%OldChk={initial_density_checkpoint or guess_preparation.checkpoint}"
                    if initial_density_checkpoint is not None or guess_preparation is not None
                    else ""
                ),
                f"%NProcShared={int(spec.processors)}" if int(spec.processors) > 1 else "",
                f"%Mem={int(spec.memory_gb)}GB" if spec.memory_gb is not None else "",
            )
            if item
        ),
        basis_set_file=spec.basis_file,
        route_override=route_override,
        **writer_kwargs,
    )
    # The Gaussian writer normalizes the route (for example by adding ``#p``
    # and ``Force``).  Rebind the authorization to the exact route stored in
    # the input file; otherwise the final launcher check compares the digest
    # of the normalized route with the digest of the pre-normalized route.
    if route_override is not None:
        from matrix_gaussian import (
            create_gaussian_route_override,
            gaussian_input_routes,
            gaussian_route_violations,
        )

        normalized_routes = gaussian_input_routes(input_path)
        if len(normalized_routes) != 1:
            raise RuntimeError(
                "LINK Gaussian point input must contain exactly one route"
            )
        normalized_route = normalized_routes[0]
        normalized_violations = gaussian_route_violations(normalized_route)
        route_override = create_gaussian_route_override(
            normalized_route,
            allow_tight="tight" in normalized_violations
            and spec.finite_difference_energy_scf_tight,
            reason=route_override.reason,
            authorized_by=route_override.authorized_by,
        )
    direct_faf = point_dir / "gauin.faf"
    direct_faf.unlink(missing_ok=True)
    gaussian_env = dict(spec.env or {})
    gaussian_env["GAUSS_OFDEF"] = str(direct_faf.resolve())
    run_started = time.perf_counter()
    resources = CalculationResources(
        process_count=1,
        threads_per_process=max(1, int(spec.processors)),
        memory_per_job_gb=float(spec.memory_gb or 1),
        concurrent_jobs=1,
    )
    gaussian_executable = (
        spec.executable
        or os.environ.get("ORACLE_GAUSSIAN_EXE")
        or gaussian_env.get("ORACLE_GAUSSIAN_EXE")
        or "gdv"
    )
    parent_plan = authorized_parent_plan_from_environment()
    launch_plan = build_calculation_launch_plan(
        backend="Gaussian",
        host=None if parent_plan is None else parent_plan.host,
        input_path=input_path,
        command=(gaussian_executable, str(input_path.resolve())),
        resources=resources,
        workdir=point_dir,
    )
    launch_authorization = authorize_calculation_launch(
        launch_plan,
        approved_plan_sha256=launch_plan.sha256,
        authorized_by="LINK optimizer user request",
    )
    run = run_gaussian_job(
        point_dir,
        executable=gaussian_executable,
        # ``run_gaussian_job`` resolves input_path relative to point_dir;
        # passing the already-prefixed Path duplicates the directory.
        input_path=input_path.name,
        timeout=spec.timeout,
        env=gaussian_env,
        resources=resources,
        launch_authorization=launch_authorization,
        route_overrides=(() if route_override is None else (route_override,)),
    )
    backend_wall_time = time.perf_counter() - run_started
    if run.success is not True:
        raise RuntimeError(run.message)
    cc_amplitudes_recovered = False
    if cc_amplitude_checkpoint and history:
        gaussian_output = run.log_path.read_text(encoding="utf-8", errors="replace")
        cc_amplitudes_recovered = (
            "Recovering and transforming amplitudes from chk file." in gaussian_output
            and "Amplitudes recovered." in gaussian_output
        )
        if not cc_amplitudes_recovered:
            raise RuntimeError(
                "Gaussian checkpoint restart did not recover and transform "
                "coupled-cluster amplitudes through Link 804"
            )
    parsed = read_gaussian_point_result(
        run.log_path,
        require_gradient=ensure_force,
    )
    from matrix_gaussian.topology import read_gaussian_population

    population = read_gaussian_population(run.log_path)
    cm5_charges = None
    if population.cm5_charges:
        expected_indices = tuple(range(1, len(atoms) + 1))
        observed_indices = tuple(sorted(population.cm5_charges))
        if observed_indices != expected_indices:
            raise RuntimeError(
                "Gaussian CM5 population is incomplete: expected atom indices "
                f"{expected_indices}, found {observed_indices}"
            )
        cm5_charges = np.asarray(
            [population.cm5_charges[index] for index in expected_indices],
            dtype=float,
        )
    # Resolve the backend geometry once, after any full-precision derivative
    # artifact has had the opportunity to establish the input frame.
    returned = None
    execution = {
        "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
        "derivative_transport": "gaussian_restored_force_table",
        "derivative_transport_lossless_float64": False,
        "wall_time_seconds_backend": float(backend_wall_time),
        "wall_time_seconds_scf_guess_preparation": (
            float(guess_preparation_elapsed)
        ),
        "wall_time_seconds_total": float(
            backend_wall_time
            + guess_preparation_elapsed
        ),
        "gaussian_accuracy_tier": str(spec.gaussian_accuracy_tier),
        "gaussian_integration_grid": {
            "protocol": "route_or_backend_default",
            "coarse": "SG1Grid",
            "initial": "FineGrid",
            "intermediate": "FineGrid",
            "final": "UltraFineGrid",
        }[str(spec.gaussian_accuracy_tier)],
        "gaussian_scf_convergence_exponent": {
            "protocol": None,
            "coarse": 6,
            "initial": 6,
            "intermediate": 7,
            "final": 8,
        }[str(spec.gaussian_accuracy_tier)],
        "gaussian_derivative_scf_gate_exponent": {
            "protocol": None,
            "coarse": 6,
            "initial": 6,
            "intermediate": 7,
            "final": 8,
        }[str(spec.gaussian_accuracy_tier)],
        "scf_cycles": _gaussian_scf_cycles(run.log_path),
        "cc_iterations": _gaussian_cc_iterations(run.log_path),
        "cc_amplitude_checkpoint_iop_9_77": cc_amplitude_save_iop,
        "cc_amplitude_transform_iop_8_24": cc_amplitude_transform_iop,
        "cc_amplitude_restart_requested": bool(
            cc_amplitude_checkpoint and history
        ),
        "cc_amplitudes_recovered": cc_amplitudes_recovered,
        "gradient_electronic_state": (
            int(spec.electronic_state)
            if ensure_force and int(spec.electronic_state) > 0
            else 0
        ),
        "scf_state_reuse": (
            "lower_rung_density_seed_fallback"
            if density_only_seed and guess_preparation_failure
            else "lower_rung_density_seed_projected"
            if density_only_seed and initial_density_checkpoint is not None
            else "disabled"
            if not spec.reuse_density_projection
            else "standard_guess_fallback"
            if guess_preparation_failure
            else "standard_guess_no_history"
            if guess_preparation is None
            else guess_preparation.status
        ),
    }
    if guess_preparation_failure:
        execution["scf_state_fallback_reason"] = guess_preparation_failure
    if guess_preparation is not None:
        execution.update(
            {
                "scf_state_source_count": guess_preparation.source_state_count,
                "scf_state_projection_count": guess_preparation.projection_count,
                "scf_state_projection_residual_max": (
                    guess_preparation.projection_residual_max
                ),
                "scf_state_electron_count_error": guess_preparation.electron_count_error,
                "scf_state_events": list(guess_preparation.events),
                "scf_state_transport": (
                    "gaussian_native_mo_projection_density_only_float64"
                    if density_only_seed
                    else "gaussian_fixed_ao_density_fock_extrapolation_float64"
                ),
            }
        )
    elif initial_density_checkpoint is not None:
        execution.update(
            {
                "scf_state_source_count": 1,
                "scf_state_projection_count": 1,
                "scf_state_projection_residual_max": 0.0,
                "scf_state_electron_count_error": 0.0,
                "scf_state_events": [
                    "gaussian_inprocess_lower_rung_density_projection:"
                    "target_level_fock_rebuilt"
                ],
                "scf_state_transport": (
                    "gaussian_inprocess_mo_projection_density_only_float64"
                ),
            }
        )
    checkpoint = point_dir / "gauin.chk"
    derivative_extraction_started = time.perf_counter()
    if ensure_force and checkpoint.is_file():
        from matrix_gaussian import (
            export_gaussian_checkpoint_faf,
            formchk_checkpoint,
            read_gaussian_faf_point_result,
            read_gaussian_fchk_point_result,
        )

        cached_state_artifact = _gaussian_cached_faf_path(checkpoint)
        if requested_hessian:
            state_artifact = formchk_checkpoint(checkpoint, point_dir / "gauin.fchk", timeout=spec.timeout)
            checkpoint_point = read_gaussian_fchk_point_result(state_artifact)
            transport_name = "gaussian_checkpoint_fchk_float64"
            extraction_mode = "analytic_hessian_fchk"
        elif direct_faf.is_file():
            if spec.reuse_density_projection:
                cached_state_artifact.unlink(missing_ok=True)
                direct_faf.replace(cached_state_artifact)
                state_artifact = cached_state_artifact
            else:
                state_artifact = direct_faf
            checkpoint_point = read_gaussian_faf_point_result(state_artifact)
            transport_name = "gaussian_inprocess_faf_float64"
            extraction_mode = "inprocess_faf"
        elif spec.reuse_density_projection:
            state_artifact = cached_state_artifact
            export_gaussian_checkpoint_faf(
                checkpoint,
                state_artifact,
                gaussian_executable=spec.executable,
                timeout=spec.timeout,
            )
            checkpoint_point = read_gaussian_faf_point_result(state_artifact)
            transport_name = "gaussian_checkpoint_faf_float64"
            extraction_mode = "external_faf_fallback"
        else:
            state_artifact = formchk_checkpoint(
                checkpoint,
                point_dir / "gauin.fchk",
                timeout=spec.timeout,
            )
            checkpoint_point = read_gaussian_fchk_point_result(state_artifact)
            transport_name = "gaussian_checkpoint_fchk_float64"
            extraction_mode = "external_fchk_fallback"
        precise_gradient = np.asarray(
            checkpoint_point.cartesian_gradient_hartree_per_bohr, dtype=float
        ).reshape(-1)
        if precise_gradient.shape != (point.coordinates_angstrom.size,):
            raise RuntimeError("Gaussian checkpoint contains no complete Cartesian gradient")
        checkpoint_coordinates = (
            np.asarray(checkpoint_point.cartesian_coordinates_bohr, dtype=float)
            * BOHR_TO_ANGSTROM
        )
        rotation = kabsch_rotation(checkpoint_coordinates, point.coordinates_angstrom)
        transported_gradient, _unused_hessian = rotate_cartesian_derivatives(
            precise_gradient,
            rotation,
        )
        if requested_hessian:
            from matrix_gaussian import hessian_input_from_gaussian_fchk
            hessian_data = hessian_input_from_gaussian_fchk(state_artifact)
            _, cartesian_hessian = rotate_cartesian_derivatives(
                precise_gradient, rotation, np.asarray(hessian_data.cartesian_hessian, dtype=float))
        precise_energy = (
            parsed.energy_hartree
            if checkpoint_point.total_energy_hartree is None
            else float(checkpoint_point.total_energy_hartree)
        )
        parsed = replace(
            parsed,
            energy_hartree=precise_energy,
            gradient_hartree_per_bohr=transported_gradient.reshape(-1),
        )
        derivative_artifact = write_full_precision_derivative_artifact(
            point_dir / FULL_PRECISION_DERIVATIVE_ARTIFACT_NAME,
            energy_hartree=precise_energy,
            gradient_hartree_per_bohr=transported_gradient,
            hessian_hartree_per_bohr2=cartesian_hessian,
            coordinates_angstrom=point.coordinates_angstrom,
        )
        execution.update(
            {
                "derivative_transport": transport_name,
                "derivative_extraction_mode": extraction_mode,
                "derivative_transport_lossless_float64": True,
                "derivative_transport_artifact": str(derivative_artifact.resolve()),
                "scf_state_binary_artifact": (
                    str(state_artifact.resolve())
                    if spec.reuse_density_projection
                    else None
                ),
                "restart_artifact": str(checkpoint.resolve()),
                "restart_artifact_format": "gaussian_chk",
                "restart_projection": (
                    "joint_density_fock_float64"
                    if spec.reuse_density_projection
                    else "checkpoint_available_not_reused"
                ),
                "restart_reuse_for_displacements": True,
            }
        )
        # Checkpoint derivatives are restored to the input axes; use that
        # known frame and avoid reparsing the full log for production jobs.
        returned = np.asarray(point.coordinates_angstrom, dtype=float)
    if requested_hessian:
        if cartesian_hessian is None:
            raise RuntimeError("Gaussian analytic Hessian request returned no checkpoint force constants")
        execution.update(hessian_source="analytic_checkpoint_fchk", hessian_evaluations=1)
    if checkpoint.is_file():
        execution.setdefault("restart_artifact", str(checkpoint.resolve()))
        execution.setdefault("restart_artifact_format", "gaussian_chk")
        execution.setdefault(
            "restart_projection",
            "guess_read_cc_amplitudes" if cc_amplitude_checkpoint else "guess_read",
        )
        execution.setdefault("restart_reuse_for_displacements", True)
    execution["wall_time_seconds_derivative_extraction"] = float(
        time.perf_counter() - derivative_extraction_started
    )
    if returned is None:
        try:
            returned = read_gaussian_log_geometry(run.log_path).coordinates_angstrom
        except ValueError:
            # Minimal wrappers may return only energy and gradient.
            returned = None
    if spec.electronic_state > 0:
        state_data = gaussian_state_fingerprint_data(run.log_path)
        if state_data is None and str(spec.state_tracking).strip().casefold() == "apoc":
            raise RuntimeError(
                "Gaussian excited-state LINK point contains no supported amplitude "
                "manifold for APOC state following"
            )
        if state_data is not None:
            state_ids, excitations, _vectors, _representation = state_data
            reference_energy = gaussian_state_reference_energy_hartree(run.log_path)
            if reference_energy is None:
                raise RuntimeError(
                    "Gaussian excited-state point has no method-consistent reference energy"
                )
            execution.update(
                {
                    "energy_protocol_id": (
                        "gaussian.eom_ccsd.precise_manifold_energy.v1"
                        if _representation == GAUSSIAN_EOM_STATE_FINGERPRINT_REPRESENTATION
                        else "gaussian.tddft.tda.requested_root_energy.v1"
                    ),
                    "requested_state_energy_hartree": float(parsed.energy_hartree),
                    "state_fingerprint_representation": _representation,
                    "state_ids": [int(value) for value in state_ids],
                    "ground_energy_hartree": float(reference_energy),
                    "excitation_energies_hartree": [float(value) for value in excitations],
                }
            )
            if str(spec.state_tracking).strip().casefold() == "apoc":
                if _representation == "gaussian-td-configurations-v1":
                    from matrix_gaussian import formchk_checkpoint
                    from matrix_gaussian.td_tracking import write_td_ao_fingerprints, REPRESENTATION

                    if not checkpoint.is_file():
                        raise RuntimeError("Gaussian AO state tracking requires a checkpoint with orbitals")
                    tracking_fchk = point_dir / "gauin.fchk"
                    if not tracking_fchk.is_file():
                        formchk_checkpoint(checkpoint, tracking_fchk, timeout=spec.timeout)
                    archive = write_td_ao_fingerprints(
                        run.log_path, tracking_fchk, point_dir / "gaussian_state_fingerprints.npz")
                    execution["state_fingerprint_representation"] = REPRESENTATION
                    execution["state_fingerprint_scope"] = "printed_TD_amplitudes_transformed_with_checkpoint_MOs"
                    execution["state_fingerprint_frame"] = "kabsch_checkpoint_AO_rotation"
                else:
                    archive = write_gaussian_state_fingerprint_archive(
                        run.log_path, point_dir / "gaussian_state_fingerprints.npz")
                if archive is None:
                    raise RuntimeError("Gaussian state-fingerprint archive could not be written")
                execution["state_fingerprint_file"] = str(archive)
    # Gaussian restores its printed Cartesian forces to the input axes.  The
    # parsed archive geometry is retained only for alignment/provenance and
    # must not rotate those derivatives a second time.
    execution["wall_time_seconds_total"] = float(
        time.perf_counter() - point_adapter_started
    )
    result = _point_result_from_parsed(
        point,
        parsed,
        f"Gaussian {route}",
        returned,
        execution=execution,
        derivatives_in_point_frame=True,
    )
    return replace(result, cm5_charges=cm5_charges, hessian_hartree_per_bohr2=cartesian_hessian)


def _gaussian_route_override(spec: QMScanBackend, route: str):
    """Return an exact route-bound authorization owned by a frozen contract."""

    from matrix_gaussian import create_gaussian_route_override, gaussian_route_violations

    violations = gaussian_route_violations(route)
    if not violations:
        return None
    allow_tight = "tight" in violations and spec.finite_difference_energy_scf_tight
    if any(
        rule == "nosymm" or (rule == "tight" and not allow_tight)
        for rule in violations
    ):
        return None
    return create_gaussian_route_override(
        route,
        allow_tight=allow_tight,
        reason="User-authorized fixed LINK Gaussian scalar-energy finite-difference protocol",
        authorized_by="LINK Gaussian protocol",
    )


def _gaussian_cached_faf_path(checkpoint: Path) -> Path:
    """Return a content-addressed FAF cache beside an immutable point checkpoint."""

    with Path(checkpoint).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()[:16]
    source = Path(checkpoint)
    return source.with_name(f"{source.stem}.{digest}.scf.faf")


def _prepare_gaussian_joint_scf_guess(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    target_coordinates_angstrom: np.ndarray,
    history: Sequence[Path],
    workdir: Path,
    *,
    density_only: bool = False,
):
    """Extrapolate D/F in the fixed, atom-centred AO representation.

    Accepted optimization points have identical atom ordering and AO
    specification.  This is the same fixed-basis contract used by the
    in-process LINK adapters: AO rows follow their atoms as the geometry moves,
    so density and Fock matrices can be reused or extrapolated directly.  A
    native Gaussian projection remains mandatory for a real basis change (the
    lower-rung seed at the first point), but launching an extra projection job
    at every same-basis optimization point only adds process and checkpoint
    overhead without changing the represented AO space.
    """

    from matrix_gaussian import (
        GaussianProjectedSCFState,
        GaussianSCFGuessPreparation,
        export_gaussian_checkpoint_faf,
        extrapolate_projected_gaussian_states,
        gaussian_array_matrix,
        inject_gaussian_projected_state,
        overlap_from_complete_orbitals,
        read_gaussian_binary_arrays,
        read_gaussian_faf_point_result,
    )
    from matrix_chem.topology.elements import atomic_number

    started = time.perf_counter()
    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_states: list[GaussianProjectedSCFState] = []
    events: list[str] = []
    retained_history = tuple(history[-4:])
    latest_source_faf: Path | None = None
    for index, source_checkpoint in enumerate(retained_history):
        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"accepted Gaussian checkpoint not found: {source_checkpoint}")
        source_faf = _gaussian_cached_faf_path(source_checkpoint)
        cache_status = "cached"
        if not source_faf.is_file():
            source_faf = export_gaussian_checkpoint_faf(
                source_checkpoint,
                source_faf,
                gaussian_executable=spec.executable,
                timeout=spec.timeout,
            )
            cache_status = "exported"
        latest_source_faf = source_faf
        source_point = read_gaussian_faf_point_result(source_faf)
        expected_atomic_numbers = np.asarray([atomic_number(atom) for atom in atoms], dtype=int)
        if not np.array_equal(source_point.atomic_numbers, expected_atomic_numbers):
            raise ValueError("Gaussian SCF history changed atom identity or ordering")
        source_coordinates_angstrom = (
            np.asarray(source_point.cartesian_coordinates_bohr, dtype=float)
            * BOHR_TO_ANGSTROM
        )
        source_arrays = read_gaussian_binary_arrays(source_faf)
        source_orbitals_alpha = gaussian_array_matrix(
            source_arrays["ALPHA MO COEFFICIENTS"]
        )
        density_alpha = gaussian_array_matrix(
            source_arrays["ALPHA DENSITY MATRIX"]
        )
        fock_alpha = gaussian_array_matrix(source_arrays["ALPHA FOCK MATRIX"])
        density_beta = None
        fock_beta = None
        source_orbitals_beta = None
        beta_occupations = None
        if "BETA MO COEFFICIENTS" in source_arrays:
            source_orbitals_beta = gaussian_array_matrix(
                source_arrays["BETA MO COEFFICIENTS"]
            )
            density_beta = gaussian_array_matrix(
                source_arrays["BETA DENSITY MATRIX"]
            )
            fock_beta = gaussian_array_matrix(source_arrays["BETA FOCK MATRIX"])
        overlap = overlap_from_complete_orbitals(source_orbitals_alpha)
        if overlap is None:
            raise RuntimeError("Gaussian SCF state has no recoverable AO overlap")
        alpha_electrons = float(np.trace(density_alpha @ overlap))
        beta_electrons = (
            None
            if density_beta is None
            else float(np.trace(density_beta @ overlap))
        )
        measured_electrons = (
            2.0 * alpha_electrons
            if beta_electrons is None
            else alpha_electrons + beta_electrons
        )
        expected_electrons = float(np.sum(expected_atomic_numbers) - int(spec.charge))
        electron_error = abs(measured_electrons - expected_electrons)
        if electron_error > 1.0e-7:
            raise RuntimeError(
                "Gaussian projected density changed electron count by "
                f"{electron_error:.3e}"
            )
        if beta_electrons is None:
            if int(spec.multiplicity) != 1:
                raise RuntimeError(
                    "restricted Gaussian SCF state is incompatible with multiplicity"
                )
        elif abs(
            (alpha_electrons - beta_electrons) - (int(spec.multiplicity) - 1)
        ) > 1.0e-7:
            raise RuntimeError("Gaussian SCF density has an inconsistent spin count")
        occupation_array = source_arrays.get("SCF ORBITAL OCCUPATIONS")
        if occupation_array is not None:
            occupations = np.asarray(occupation_array.values, dtype=float)
            alpha_occupations = occupations[: source_orbitals_alpha.shape[1]]
            if source_orbitals_beta is not None:
                start = source_orbitals_alpha.shape[1]
                beta_occupations = occupations[
                    start : start + source_orbitals_beta.shape[1]
                ]
            occupation_status = "stored_occupations"
        else:
            alpha_inverse = np.linalg.pinv(source_orbitals_alpha, rcond=1.0e-11)
            alpha_occupations = np.clip(
                np.diag(alpha_inverse @ density_alpha @ alpha_inverse.T),
                0.0,
                1.0,
            )
            if source_orbitals_beta is not None and density_beta is not None:
                beta_inverse = np.linalg.pinv(source_orbitals_beta, rcond=1.0e-11)
                beta_occupations = np.clip(
                    np.diag(beta_inverse @ density_beta @ beta_inverse.T),
                    0.0,
                    1.0,
                )
            occupation_status = "density_trace_inferred_occupations"
        electron_count = expected_electrons
        source_states.append(
            GaussianProjectedSCFState(
                source_checkpoint=source_checkpoint,
                projected_checkpoint=source_checkpoint,
                source_coordinates_angstrom=source_coordinates_angstrom,
                target_coordinates_angstrom=source_coordinates_angstrom,
                density_alpha=density_alpha,
                density_beta=density_beta,
                fock_alpha=None if density_only else fock_alpha,
                fock_beta=None if density_only else fock_beta,
                orbitals_alpha=source_orbitals_alpha,
                orbitals_beta=source_orbitals_beta,
                occupations_alpha=alpha_occupations,
                occupations_beta=beta_occupations,
                overlap=overlap,
                electron_count=electron_count,
                projection_residual=0.0,
            )
        )
        events.append(
            f"state_{index + 1}:{cache_status}_binary_float64:"
            f"electron_error={electron_error:.3e}:{occupation_status}"
        )
    assert source_states
    source_state, status, _source_electron_error = extrapolate_projected_gaussian_states(
        source_states,
        target_coordinates_angstrom,
    )
    status = (
        "same_basis_reuse"
        if len(source_states) == 1
        else status.replace("projected", "same_basis", 1)
    )
    events.append(f"fixed_atom_centered_ao_frame:{status}")

    latest_checkpoint = retained_history[-1]
    assert latest_source_faf is not None
    target_state = GaussianProjectedSCFState(
        source_checkpoint=source_state.source_checkpoint,
        projected_checkpoint=latest_checkpoint,
        source_coordinates_angstrom=source_state.source_coordinates_angstrom,
        target_coordinates_angstrom=np.asarray(target_coordinates_angstrom, dtype=float),
        density_alpha=source_state.density_alpha,
        density_beta=source_state.density_beta,
        fock_alpha=source_state.fock_alpha,
        fock_beta=source_state.fock_beta,
        orbitals_alpha=source_state.orbitals_alpha,
        orbitals_beta=source_state.orbitals_beta,
        occupations_alpha=source_state.occupations_alpha,
        occupations_beta=source_state.occupations_beta,
        overlap=source_state.overlap,
        electron_count=source_state.electron_count,
        projection_residual=0.0,
    )
    electron_error = float(_source_electron_error)
    events.append(
        "same_basis_identity_transport:"
        f"electron_error={electron_error:.3e}"
    )
    output_checkpoint = inject_gaussian_projected_state(
        target_state,
        template_faf=latest_source_faf,
        output_faf=root / "joint_guess.faf",
        output_checkpoint=root / "joint_guess.chk",
        gaussian_executable=spec.executable,
        timeout=spec.timeout,
        require_fock=not density_only,
    )
    return GaussianSCFGuessPreparation(
        checkpoint=output_checkpoint,
        status=status,
        source_state_count=len(retained_history),
        projection_count=0,
        projection_residual_max=0.0,
        electron_count_error=electron_error,
        wall_time_seconds=time.perf_counter() - started,
        events=tuple(events),
    )


def _gaussian_scf_cycles(log_path: Path | str) -> int:
    """Return the total converged SCF cycles reported in one Gaussian log."""

    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    counts = [
        int(match.group(1))
        for match in re.finditer(
            r"SCF Done:.*?after\s+(\d+)\s+cycles?",
            text,
            flags=re.IGNORECASE,
        )
    ]
    return int(sum(counts))


def _gaussian_cc_iterations(log_path: Path | str) -> int:
    """Return the number of ground-state CC iterations in a Gaussian log."""

    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return len(
        re.findall(
            r"^\s*DE\(Corr\)\s*=",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )


def _gaussian_route(spec: QMScanBackend) -> str:
    """Compose a complete Gaussian route from model and extra keywords.

    ``QMScanBackend.route`` is also used for backend-specific additions such
    as empirical dispersion.  When those additions do not contain a
    method/basis model, retain the explicitly selected ``method`` and
    ``basis`` instead of silently falling back to HF/STO-3G.
    """

    route = str(spec.route).strip()
    method = str(spec.method).strip() or "HF"
    engine_method = backend_method_name(spec.name, method)
    basis = "Gen" if spec.basis_file is not None else (str(spec.basis).strip() or "STO-3G")
    normalized_method = re.sub(r"[^A-Z0-9]", "", method.upper())
    separate_basis_methods = {
        "REVDSDPBEP86D3",
        "REVDSDPBEP86D3BJ",
        "REVDSDPBEP86D4",
    }
    model = (
        f"{engine_method} {basis}"
        if normalized_method in separate_basis_methods
        else f"{engine_method}/{basis}"
    )
    if not route:
        assembled = f"#p {model}"
    else:
        tokens = route.lstrip("#").split()
        if any("/" in token for token in tokens):
            assembled = route
        else:
            prefix = "#p" if not route.startswith("#") else route.split(maxsplit=1)[0]
            extras = route.lstrip("#").removeprefix("p").removeprefix("P").strip()
            assembled = " ".join(item for item in (prefix, model, extras) if item)
    if int(spec.electronic_state) > 0:
        assembled = _gaussian_route_with_requested_excited_root(
            assembled,
            root=int(spec.electronic_state),
            nstates=max(int(spec.excited_states or 0), int(spec.electronic_state)),
        )
    if spec.finite_difference_energy_scf_tight:
        if re.search(r"\bscf\s*=", assembled, flags=re.IGNORECASE):
            raise ValueError(
                "finite-difference SCF=Tight is owned by the scalar-energy "
                "protocol; remove a competing SCF keyword"
            )
        assembled = f"{assembled} SCF=Tight"
    if not re.search(r"\bpop\s*=", assembled, flags=re.IGNORECASE):
        assembled = f"{assembled} Pop=CM5"
    accuracy_tier = str(spec.gaussian_accuracy_tier).strip().casefold()
    if accuracy_tier != "protocol":
        if re.search(
            r"\bscf\s*=|\b(?:integral|int)\s*=|\biop\s*\(\s*7\s*/\s*127\s*=",
            assembled,
            re.IGNORECASE,
        ):
            raise ValueError(
                "LINK Gaussian accuracy scheduling owns SCF=, Integral=, and the Link 701 "
                "SCF derivative gate; remove competing keywords from the base route"
            )
        scf_convergence, grid = {
            "coarse": (6, "SG1Grid"),
            "initial": (6, "FineGrid"),
            "intermediate": (7, "FineGrid"),
            "final": (8, "UltraFineGrid"),
        }[accuracy_tier]
        assembled = (
            f"{assembled} SCF=Conver={scf_convergence} "
            f"Integral={grid} IOp(7/127={scf_convergence})"
        )
    if "hessian" in {str(x).lower() for x in spec.properties}:
        frequency = re.search(r"\bfreq\s*=\s*(\([^)]*\)|[^\s]+)", assembled, flags=re.IGNORECASE)
        if frequency:
            options = frequency.group(1).strip("()").lower().split(",")
            if "analytic" not in options or any(x.strip().startswith(("num", "read")) for x in options):
                raise ValueError("LINK requires an analytic Hessian; competing Freq options are unsupported")
            assembled = re.sub(r"\bforce\b", "", assembled, flags=re.IGNORECASE).strip()
        else:
            assembled = re.sub(r"\b(?:force|freq)\b", "", assembled, flags=re.IGNORECASE)
            assembled = f"{assembled.strip()} Freq=Analytic"
    return assembled


def _gaussian_route_with_requested_excited_root(
    route: str,
    *,
    root: int,
    nstates: int,
    response_convergence_exponent: int | None = None,
) -> str:
    """Bind an explicit Gaussian TD/TDA route to LINK's requested state.

    APOC may follow one electronic character through a change in energetic
    root ordering. The Gaussian force must then be recomputed for that root;
    keeping ``Root`` embedded in the original route would return a gradient
    for the wrong state. Unrelated excited-state methods retain their own
    adapter semantics.
    """

    if root < 1 or nstates < root:
        raise ValueError("Gaussian excited-state route requires NStates >= Root >= 1")
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?P<name>TDA|TD)\s*(?:=\s*(?P<value>\([^)]*\)|[^\s]+))?",
        re.IGNORECASE,
    )
    match = pattern.search(route)
    if match is None:
        return route
    name = match.group("name")
    value = (match.group("value") or "").strip()
    if value.startswith("(") and value.endswith(")"):
        options = [item.strip() for item in value[1:-1].split(",") if item.strip()]
    elif value:
        options = [value]
    else:
        options = []
    options = [
        option
        for option in options
        if not re.match(r"^(?:root|nstates)\s*=", option, flags=re.IGNORECASE)
    ]
    if response_convergence_exponent is not None and not any(
        re.match(r"^conver\s*=", option, flags=re.IGNORECASE) for option in options
    ):
        options.append(f"Conver={int(response_convergence_exponent)}")
    options.extend((f"NStates={nstates}", f"Root={root}"))
    replacement = f"{name}=({','.join(options)})"
    return route[: match.start()] + replacement + route[match.end() :]


def _run_fragment_oniom_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    """Evaluate explicit subtractive E_L(real)+E_H(model)-E_L(model)."""

    high_atoms = tuple(int(index) for index in spec.oniom_high_atoms)
    if not high_atoms or min(high_atoms) < 0 or max(high_atoms) >= len(atoms):
        raise ValueError("fragment ONIOM backend needs valid oniom_high_atoms")
    full_low_spec = QMScanBackend(
        name="gaussian",
        route=spec.oniom_low_route,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        executable=spec.executable,
        timeout=spec.timeout,
        env=spec.env,
        gaussian_connectivity_bonds=spec.gaussian_connectivity_bonds,
    )
    fragment_high_spec = QMScanBackend(
        name="gaussian",
        route=spec.route,
        charge=0,
        multiplicity=1,
        executable=spec.executable,
        timeout=spec.timeout,
        env=spec.env,
    )
    fragment_low_spec = QMScanBackend(
        name="gaussian",
        route=spec.oniom_low_route,
        charge=0,
        multiplicity=1,
        executable=spec.executable,
        timeout=spec.timeout,
        env=spec.env,
        gaussian_connectivity_bonds=tuple(
            (high_atoms.index(left), high_atoms.index(right), order)
            for left, right, order in (spec.gaussian_connectivity_bonds or ())
            if left in high_atoms and right in high_atoms
        ),
    )
    low_full = _run_gaussian_scan_point(full_low_spec, atoms, point, point_dir / "low_full")
    local_coordinates = point.coordinates_angstrom[list(high_atoms)]
    local_atoms = tuple(atoms[index] for index in high_atoms)
    local_point = ScanPoint(
        index=point.index,
        displacement=point.displacement,
        coordinates_angstrom=local_coordinates,
    )
    high_fragment = _run_gaussian_scan_point(
        fragment_high_spec, local_atoms, local_point, point_dir / "high_fragment"
    )
    low_fragment = _run_gaussian_scan_point(
        fragment_low_spec, local_atoms, local_point, point_dir / "low_fragment"
    )
    components = (low_full, high_fragment, low_fragment)
    if any(
        item.energy_hartree is None or item.gradient_hartree_per_bohr is None for item in components
    ):
        raise RuntimeError("fragment ONIOM component lacks energy or analytic gradient")
    gradient = np.asarray(low_full.gradient_hartree_per_bohr, dtype=float).reshape((-1, 3))
    correction = (
        np.asarray(high_fragment.gradient_hartree_per_bohr, dtype=float)
        - np.asarray(low_fragment.gradient_hartree_per_bohr, dtype=float)
    ).reshape((-1, 3))
    gradient[list(high_atoms)] += correction
    energy = (
        float(low_full.energy_hartree)
        + float(high_fragment.energy_hartree)
        - float(low_fragment.energy_hartree)
    )
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient.reshape(-1),
        backend_coordinates_angstrom=point.coordinates_angstrom,
        status="completed",
        source=(
            f"fragment-ONIOM Gaussian: {spec.oniom_low_route} real + "
            f"{spec.route} model - {spec.oniom_low_route} model"
        ),
        message="three component energy/gradient evaluations",
    )


def _run_orca_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_orca import (
        convert_orca_gbw_to_molden,
        hessian_input_from_orca_output,
        orca_state_fingerprint_data,
        orca_transition_electric_dipoles,
        read_orca_output_geometry,
        read_orca_point_result,
        run_orca_job,
        write_orca_state_fingerprint_archive,
        write_orca_point_input,
    )

    route = _orca_route(spec)
    if spec.basis_file is not None:
        source_basis = Path(spec.basis_file).expanduser().resolve()
        if not source_basis.is_file():
            raise FileNotFoundError(f"ORCA orbital basis artifact does not exist: {source_basis}")
        local_basis = point_dir / "orbital-basis.bas"
        point_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_basis, local_basis)
        route += f'\n%basis\n  GTOName = "{local_basis.name}"\nend'
    elif str(spec.basis).strip().casefold() == "cc-pwcvtz":
        hydrogenic = tuple(
            element
            for element in ("H", "He")
            if any(atom.strip().casefold() == element.casefold() for atom in atoms)
        )
        if hydrogenic:
            overrides = "\n".join(
                f'  NewGTO {element} "cc-pVTZ" end' for element in hydrogenic
            )
            route += f"\n%basis\n{overrides}\nend"
    restart_orbitals = None
    if spec.restart_artifact is not None:
        source = Path(spec.restart_artifact).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ORCA restart artifact does not exist: {source}")
        if source.suffix.casefold() != ".gbw":
            raise ValueError("ORCA restart artifact must be a .gbw file")
        local_restart = point_dir / "central_reference.gbw"
        if source != local_restart.resolve():
            shutil.copy2(source, local_restart)
        restart_orbitals = local_restart.name
    input_path = write_orca_point_input(
        point_dir / "orca.inp",
        atoms,
        point.coordinates_angstrom,
        route=route,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        electronic_state=spec.electronic_state,
        excited_states=spec.excited_states,
        processors=spec.processors,
        memory_gb=spec.memory_gb,
        restart_orbitals=restart_orbitals,
        restart_projection=spec.restart_projection,
        scf_convergence=spec.scf_convergence,
        state_fingerprint_print_threshold=(
            1.0e-6
            if int(spec.electronic_state) > 0
            and str(spec.state_tracking).strip().casefold() == "apoc"
            else None
        ),
    )
    run = run_orca_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
        resources=CalculationResources(
            process_count=1,
            threads_per_process=max(1, int(spec.processors)),
            memory_per_job_gb=float(spec.memory_gb or 1),
            concurrent_jobs=1,
        ),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    requested_mode = str(spec.gradient_mode).strip().casefold()
    require_gradient = requested_mode not in {"energy", "numerical", "link-numerical"}
    parsed = read_orca_point_result(run.output_path, require_gradient=require_gradient)
    try:
        returned = read_orca_output_geometry(run.output_path).coordinates_angstrom
    except ValueError:
        returned = None
    # Keep the energy-read contract explicit and stable.  APOC may request a
    # rerun with a different selected root; LINK must be able to prove that
    # the rerun used exactly the same ORCA quantity and not silently fall back
    # to the SCF or excitation energy alone.
    excited = int(spec.electronic_state) > 0
    execution = {
        "energy_protocol_id": (
            "orca.tda.final_single_point_energy.v1"
            if excited
            else "orca.final_single_point_energy.v1"
        ),
        "requested_electronic_state": int(spec.electronic_state),
        "requested_state_energy_hartree": float(parsed.energy_hartree),
        "restart_reuse_for_displacements": bool(
            spec.restart_reuse_for_displacements
        ),
        "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
        "derivative_transport": (
            "orca_engrad_full_precision"
            if run.output_path.with_suffix(".engrad").is_file()
            else "orca_output_fallback"
        ),
        "derivative_transport_lossless_float64": (
            run.output_path.with_suffix(".engrad").is_file()
        ),
    }
    gbw_path = input_path.with_suffix(".gbw")
    if gbw_path.is_file():
        execution.update(
            {
                "restart_artifact": str(gbw_path.resolve()),
                "restart_artifact_format": "orca_gbw",
                "restart_projection": (
                    "CMatrix"
                    if "rohf" in str(spec.reference or "").casefold()
                    else "FMatrix"
                ),
            }
        )
    if spec.electronic_state > 0:
        state_data = orca_state_fingerprint_data(run.output_path)
        if state_data is None and str(spec.state_tracking).strip().casefold() == "apoc":
            raise RuntimeError("ORCA excited-state point contains no TDA state manifold")
        if state_data is not None:
            state_ids, excitations, _vectors, representation = state_data
            execution.update(
                {
                    "state_fingerprint_representation": representation,
                    "state_ids": [int(value) for value in state_ids],
                    # ORCA's .engrad energy belongs to the requested IRoot.
                    # Reconstruct the common reference with that root's
                    # excitation energy; subtracting root 1 unconditionally
                    # corrupts every IRoot > 1 APOC energy/gradient pair.
                    "ground_energy_hartree": _orca_excited_state_reference_energy(
                        float(parsed.energy_hartree),
                        state_ids,
                        excitations,
                        int(spec.electronic_state),
                    ),
                    "excitation_energies_hartree": [float(value) for value in excitations],
                }
            )
            transition_dipoles = orca_transition_electric_dipoles(run.output_path)
            if transition_dipoles:
                execution["state_transition_electric_dipoles_au"] = {
                    str(state_id): [float(value) for value in dipole]
                    for state_id, dipole in transition_dipoles.items()
                }
            if str(spec.state_tracking).strip().casefold() == "apoc":
                if not gbw_path.is_file():
                    raise RuntimeError("ORCA APOC tracking requires the point GBW artifact")
                converter = "orca_2mkl"
                if spec.executable is not None:
                    executable_path = Path(spec.executable).expanduser()
                    if executable_path.parent != Path("."):
                        converter = str(executable_path.with_name("orca_2mkl"))
                molden = convert_orca_gbw_to_molden(
                    gbw_path,
                    output=point_dir / "orca.molden.input",
                    executable=converter,
                    timeout=spec.timeout,
                ).molden_path
                archive = write_orca_state_fingerprint_archive(
                    run.output_path,
                    point_dir / "orca_state_fingerprints.npz",
                    molden=molden,
                )
                if archive is None:
                    raise RuntimeError("ORCA state-fingerprint archive could not be written")
                execution["state_fingerprint_file"] = str(archive)
                execution["state_fingerprint_representation"] = (
                    "orca-tda-transition-density-ao"
                )
                execution["state_fingerprint_source_file"] = str(molden.resolve())
    result = _point_result_from_parsed(
        point, parsed, f"ORCA {route}", returned, execution=execution
    )
    if "hessian" in set(spec.properties):
        hessian_data = hessian_input_from_orca_output(run.output_path)
        result = replace(
            result,
            hessian_hartree_per_bohr2=np.asarray(hessian_data.cartesian_hessian, dtype=float),
            execution={**result.execution, "hessian_source": "orca.hess"},
        )
    return result


def _run_nwchem_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from .nwchem import (
        read_nwchem_point_result,
        run_nwchem_job,
        write_nwchem_point_input,
    )

    requested_mode = str(spec.gradient_mode).strip().casefold()
    require_gradient = requested_mode not in {
        "energy",
        "energy-only",
        "numerical",
        "link-numerical",
    }
    database_path = point_dir / "matrix_link_point.db"
    restart_used = False
    if spec.restart_artifact is not None:
        source = Path(spec.restart_artifact).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"NWChem restart database not found: {source}")
        _copy_nwchem_restart_bundle(source, point_dir)
        restart_used = True
    input_path = write_nwchem_point_input(
        point_dir / "nwchem.nw",
        atoms,
        point.coordinates_angstrom,
        method=spec.method or "scf",
        basis=spec.basis or "sto-3g",
        route=spec.route,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        gradient=require_gradient,
        memory_gb=spec.memory_gb,
        restart=restart_used,
    )
    run = run_nwchem_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
        launcher=spec.launcher,
        resources=CalculationResources(
            process_count=max(1, int(spec.processors)),
            threads_per_process=1,
            memory_per_job_gb=float(spec.memory_gb or 1),
            concurrent_jobs=1,
        ),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_nwchem_point_result(run.output_path, require_gradient=require_gradient)
    returned_coordinates = (
        point.coordinates_angstrom
        if parsed.coordinates_bohr is None
        else np.asarray(parsed.coordinates_bohr, dtype=float) * 0.529177210903
    )
    execution: dict[str, object] = {
        "energy_protocol_id": "nwchem.total_energy.v1",
        "derivative_transport": "nwchem_output_atomic_units",
        "derivative_transport_lossless_float64": True,
        "processes": max(1, int(spec.processors)),
        "threads_per_process": 1,
        "memory_gb": int(spec.memory_gb or 1),
        "structured_launcher": bool(spec.launcher),
        "method_family": parsed.method_family,
        "restart_source_role": "accepted_point" if restart_used else "none",
    }
    if database_path.is_file():
        execution.update(
            {
                "restart_artifact": str(database_path.resolve()),
                "restart_artifact_format": "nwchem_db",
                "restart_reuse_for_displacements": True,
            }
        )
    return _point_result_from_parsed(
        point,
        parsed,
        f"NWChem {spec.method or 'SCF'}/{spec.basis or 'STO-3G'}",
        returned_coordinates,
        execution=execution,
        derivatives_in_point_frame=True,
    )


def _run_turbomole_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_turbomole import (
        run_turbomole_job,
        run_turbomole_ricc2_job,
        write_turbomole_control,
    )

    template = spec.turbomole_control_template or spec.basis_file
    if template is None and str(spec.route).strip():
        candidate = Path(str(spec.route).strip()).expanduser()
        if candidate.is_file():
            template = candidate
    if template is None:
        raise ValueError("TURBOMOLE LINK backend requires turbomole_control_template")
    control = write_turbomole_control(
        point_dir / "control", atoms, point.coordinates_angstrom, template=template
    )
    launcher_environment = {}
    if isinstance(spec.launcher, Mapping):
        raw_environment = spec.launcher.get("environment", {})
        if isinstance(raw_environment, Mapping):
            launcher_environment = {
                str(key): str(value) for key, value in raw_environment.items()
            }
    turbodir = (spec.env or {}).get("TURBODIR") or launcher_environment.get("TURBODIR")
    method = str(spec.method or "RIDFT").strip()
    ricc2_method = method.casefold().replace(" ", "")
    is_ricc2 = ricc2_method in {
        "mp2", "cc2", "adc(2)", "adc2", "cis(d)", "cis(d∞)", "cis(dinf)",
        "cis(d-infinity)", "ccs", "scs-cc2", "sos-cc2", "scs-adc(2)",
        "sos-adc(2)",
    }
    if is_ricc2:
        result = run_turbomole_ricc2_job(
            point_dir, control=control, turbodir=turbodir,
            method=method, timeout=spec.timeout,
        )
        protocol_id = "turbomole.ricc2.v1"
        derivative_transport = "turbomole_ricc2_gradient"
    else:
        if any(token in ricc2_method for token in ("ccsdf12", "ccsd(t)", "ccsdt", "pnoccsd")):
            raise ValueError(
                "TURBOMOLE ccsdf12/CCSD(T)/PNO-CCSD(T) is not exposed by Keymaker; "
                "select Molpro, MRCC, or ORCA"
            )
        result = run_turbomole_job(
            point_dir, control=control, turbodir=turbodir, timeout=spec.timeout
        )
        protocol_id = "turbomole.ridft.v1"
        derivative_transport = "turbomole_grad"
    requested = {str(item).strip().casefold() for item in spec.properties}
    gradient = result.gradient_hartree_per_bohr
    if "gradient" in requested and gradient is None:
        raise RuntimeError("TURBOMOLE did not produce a Cartesian gradient")
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=result.energy_hartree,
        gradient_hartree_per_bohr=gradient if "gradient" in requested else None,
        backend_coordinates_angstrom=point.coordinates_angstrom,
        source=f"TURBOMOLE {method}",
        execution={
            "engine": "turbomole",
            "energy_protocol_id": protocol_id,
            "derivative_transport": derivative_transport,
            "derivative_transport_lossless_float64": True,
        },
    )


def _copy_nwchem_restart_bundle(database: Path, target_directory: Path) -> None:
    """Copy the RTDB and its method-specific companion files into a point job."""

    target_directory.mkdir(parents=True, exist_ok=True)
    source_prefix = database.name.removesuffix(database.suffix)
    excluded_suffixes = {".nw", ".out", ".pid", ".json", ".jsonl"}
    copied_database = False
    for source in sorted(database.parent.glob(f"{source_prefix}.*")):
        if not source.is_file() or source.suffix.casefold() in excluded_suffixes:
            continue
        suffix = source.name[len(source_prefix) :]
        destination = target_directory / f"matrix_link_point{suffix}"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied_database = copied_database or destination.suffix.casefold() == ".db"
    if not copied_database:
        raise FileNotFoundError(f"NWChem restart bundle contains no RTDB file: {database}")


def _orca_excited_state_reference_energy(
    requested_total_energy_hartree: float,
    state_ids: np.ndarray,
    excitation_energies_hartree: np.ndarray,
    requested_root: int,
) -> float:
    """Recover the common ORCA reference from the requested IRoot energy."""

    ids = np.asarray(state_ids, dtype=int).reshape(-1)
    excitations = np.asarray(excitation_energies_hartree, dtype=float).reshape(-1)
    if ids.shape != excitations.shape:
        raise ValueError("ORCA state identifiers and excitation energies differ in size")
    matches = np.flatnonzero(ids == int(requested_root))
    if matches.size != 1:
        raise ValueError(f"ORCA manifold does not contain requested IRoot {requested_root}")
    return float(requested_total_energy_hartree - excitations[int(matches[0])])


def _orca_route(spec: QMScanBackend) -> str:
    """Compose ORCA model keywords with any route-only additions."""

    method = str(spec.method).strip() or "HF"
    # ORCA 6 names the available double hybrid as DSD-PBEP86 and takes D4 as
    # a separate keyword.  Accept the repository-level spelling as well.
    if method.casefold() in {"revsd-pbep86-d4", "revsd-pbep86"}:
        method = "DSD-PBEP86"
    basis = str(spec.basis).strip() or "STO-3G"
    route = str(spec.route).strip()
    if not route:
        route = "D4"
    tokens = route.split()
    normalized = {token.casefold() for token in tokens}
    requested_mode = str(spec.gradient_mode).strip().casefold()
    energy_only = requested_mode in {"energy", "numerical", "link-numerical"}
    additions = route if (energy_only or "engrad" in normalized) else f"{route} EnGrad"
    if spec.electronic_state > 0 and int(spec.electronic_state) >= 1:
        if "d4" not in normalized:
            additions = f"D4 {additions}"
        # Double hybrids require the corresponding RI-C auxiliary basis.
        if method.casefold() == "dsd-pbep86" and "/c" not in normalized:
            additions = f"{additions} {basis}/C"
    # A fully specified route supplied by a backend launcher is authoritative.
    # Do not prepend the generic HF/STO-3G defaults when the route already
    # contains a method/basis model such as ``B3LYP/G 6-31+G*``.
    # ORCA auxiliary-basis tokens (for example def2/J or def2-TZVP/C) contain
    # a slash but do not define the electronic method/orbital-basis model.
    # Only a non-auxiliary slash token makes the supplied route authoritative.
    route_has_model = any(
        "/" in token
        and not token.casefold().endswith(("/c", "/j", "/jk"))
        for token in tokens
    )
    if route_has_model or (method.casefold() in normalized and basis.casefold() in normalized):
        return additions
    if spec.basis_file is not None:
        return f"{method} {additions}"
    return f"{method} {basis} {additions}"


def _clone_or_copy_file(source: Path, target: Path) -> str:
    """Create a private restart artifact with copy-on-write when available."""

    target.parent.mkdir(parents=True, exist_ok=True)
    clonefile = getattr(os, "clonefile", None)
    if clonefile is not None:
        try:
            clonefile(source, target)
            shutil.copystat(source, target)
            return "clonefile"
        except OSError:
            target.unlink(missing_ok=True)
    try:
        import fcntl

        # Linux FICLONE: clone extents and copy them only when Molpro writes.
        with source.open("rb") as src, target.open("xb") as dst:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
        shutil.copystat(source, target)
        return "reflink"
    except (ImportError, OSError):
        target.unlink(missing_ok=True)
    shutil.copy2(source, target)
    return "copy"


def _run_molpro_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_core import CalculationResources, authorize_calculation_launch
    from matrix_engines import prepare_external_qm_launch_plan
    from matrix_molpro import (
        MOLPRO_SPEC,
        read_molpro_output_geometry,
        read_molpro_point_result,
        run_molpro_job,
        write_molpro_point_input,
    )

    method = spec.method or "hf"
    basis = spec.basis or "sto-3g"
    requested_mode = str(spec.gradient_mode).strip().casefold()
    require_gradient = requested_mode not in {"energy", "numerical", "link-numerical"}
    processes = max(1, int(spec.processors))
    total_memory_gb = float(spec.memory_gb or 1)
    # Molpro's memory directive is per MPI process and is expressed in
    # mega-words.  Keep the aggregate allocation inside the KeyMaker envelope.
    memory_megawords_per_process = max(
        1,
        int(total_memory_gb * (1024**3) // (8_000_000 * processes)),
    )
    restart_artifact = None
    restart_transfer = "none"
    # Molpro resolves FILE,2 names in its global WFU directory.  Numerical
    # displacements run concurrently, so the historical common basename
    # ``molpro.wfu`` causes independent jobs to overwrite one another.  Keep
    # the central-point restart contract, but give every displacement a
    # deterministic private WFU basename.
    point_token = hashlib.sha256(str(point_dir.resolve()).encode()).hexdigest()[:16]
    save_artifact = point_dir / f"molpro-{point_token}.wfu"
    if spec.restart_artifact is not None:
        source = Path(spec.restart_artifact).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Molpro restart artifact does not exist: {source}")
        restart_artifact = save_artifact.resolve()
        if source != restart_artifact:
            restart_transfer = _clone_or_copy_file(source, restart_artifact)
    input_path = write_molpro_point_input(
        point_dir / "molpro.com",
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        memory=f"{memory_megawords_per_process},m",
        gradient=require_gradient,
        basis_definition=spec.basis_file,
        restart_artifact=(save_artifact.name if restart_artifact is not None else None),
        # Molpro lowercases FILE paths.  A path relative to the point working
        # directory remains valid even when parent directories contain method
        # labels with uppercase characters (for example CCSD(T*)-F12b).
        save_artifact=save_artifact.name,
    )
    molpro_env = dict(spec.env or {})
    molpro_env.setdefault("OMP_NUM_THREADS", "1")
    molpro_env.setdefault("MKL_NUM_THREADS", "1")
    resources = CalculationResources(
        process_count=processes,
        threads_per_process=1,
        memory_per_job_gb=total_memory_gb,
        concurrent_jobs=1,
    )
    # A LINK invocation is itself the user's affirmative decision to launch
    # its exact point job.  Bind that decision to the generated Molpro input
    # and command so the external-backend fail-closed guard remains active.
    launch_plan = prepare_external_qm_launch_plan(
        point_dir,
        MOLPRO_SPEC,
        executable=spec.executable,
        input_path=input_path,
        env=molpro_env,
        extra_args=("-n", str(processes), *spec.extra_args),
        resources=resources,
    )
    launch_authorization = authorize_calculation_launch(
        launch_plan,
        approved_plan_sha256=launch_plan.sha256,
        authorized_by="LINK optimizer user request",
    )
    run = run_molpro_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=molpro_env,
        extra_args=("-n", str(processes), *spec.extra_args),
        resources=resources,
        launch_authorization=launch_authorization,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_molpro_point_result(run.output_path, require_gradient=require_gradient)
    try:
        returned = read_molpro_output_geometry(run.output_path).coordinates_angstrom
    except ValueError:
        returned = None
    return _point_result_from_parsed(
        point,
        parsed,
        f"Molpro {method}/{basis}",
        returned,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "molpro_output_fallback",
            "derivative_transport_lossless_float64": False,
            "energy_evaluations": 1,
            "gradient_evaluations": int(require_gradient),
            "fd_displacements": 0,
            "molpro_processes": processes,
            "molpro_memory_megawords_per_process": memory_megawords_per_process,
            "molpro_total_memory_limit_gb": total_memory_gb,
            "restart_artifact": str(save_artifact.resolve()) if save_artifact.is_file() else None,
            "restart_artifact_format": "molpro_wfu",
            "restart_projection": "start_2100.2",
            "restart_reuse_for_displacements": bool(
                spec.restart_reuse_for_displacements
            ),
            "restart_source_role": "central_point" if restart_artifact is not None else "none",
            "restart_transfer": restart_transfer,
        },
    )


def _run_mrcc_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_mrcc import (
        read_mrcc_output_geometry,
        read_mrcc_energy,
        read_mrcc_point_result,
        run_mrcc_job,
        write_mrcc_point_input,
    )

    method = spec.method or "HF"
    basis = spec.basis or "STO-3G"
    # MRCC does not list core-valence basis sets for H/He because these
    # elements have no core electrons.  For an all-H/He system the correct
    # specialization of cc-pwCVTZ is cc-pVTZ.
    basis_by_element = None
    if basis.casefold() == "cc-pwcvtz":
        elements = tuple(dict.fromkeys(atom.strip() for atom in atoms))
        if any(element.casefold() in {"h", "he"} for element in elements):
            basis_by_element = {
                element: (
                    "cc-pVTZ" if element.casefold() in {"h", "he"} else "cc-pwCVTZ"
                )
                for element in elements
            }
    input_path = write_mrcc_point_input(
        point_dir / "MINP",
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        # The high PCS2 term is energy-only and is differentiated by SONIC;
        # avoid MRCC's density/property phase for that calculation.
        density=0 if str(spec.gradient_mode).strip().casefold() in {"energy", "numerical", "link-numerical"} else 2,
        basis_by_element=basis_by_element,
    )
    if spec.basis_file is not None:
        source_basis = Path(spec.basis_file).expanduser().resolve()
        if not source_basis.is_file():
            raise FileNotFoundError(f"MRCC GENBAS artifact does not exist: {source_basis}")
        shutil.copy2(source_basis, point_dir / "GENBAS")
    # MRCC uses one keyword per line in MINP.  Preserve route-only controls
    # (for example ``core=0``) instead of silently discarding them.
    route_keywords = tuple(token for token in shlex.split(str(spec.route)) if "=" in token)
    if spec.memory_gb is not None and not any(
        token.partition("=")[0].strip().casefold() == "mem" for token in route_keywords
    ):
        memory = float(spec.memory_gb)
        memory_text = str(int(memory)) if memory.is_integer() else f"{memory:g}"
        route_keywords = (*route_keywords, f"mem={memory_text}gb")
    if route_keywords:
        with input_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(route_keywords) + "\n")
    mpi_tasks = 1
    for keyword in route_keywords:
        key, separator, value = keyword.partition("=")
        if separator and key.strip().casefold() == "mpitasks":
            try:
                mpi_tasks = max(1, int(value))
            except ValueError as exc:
                raise ValueError(f"invalid MRCC mpitasks value: {value}") from exc
    mrcc_env = dict(spec.env or {})
    mrcc_env.setdefault("OMP_NUM_THREADS", str(max(1, int(spec.processors))))
    run = run_mrcc_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=mrcc_env,
        extra_args=spec.extra_args,
        resources=CalculationResources(
            process_count=mpi_tasks,
            threads_per_process=max(1, int(spec.processors)),
            memory_per_job_gb=float(spec.memory_gb or 1),
            concurrent_jobs=1,
        ),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    energy_only = str(spec.gradient_mode).strip().casefold() in {
        "energy", "numerical", "link-numerical"
    }
    if energy_only:
        energy = read_mrcc_energy(run.output_path, method=method)
        gradient = None
        gradient_frame = "not_requested"
        gradient_source = "not_requested"
    else:
        parsed = read_mrcc_point_result(run.output_path, method=method)
        energy = parsed.energy_hartree
        gradient = parsed.gradient_hartree_per_bohr
        gradient_frame = getattr(parsed, "gradient_frame", "unknown")
        gradient_source = getattr(parsed, "gradient_source", "output")
    try:
        returned = read_mrcc_output_geometry(run.output_path).coordinates_angstrom
    except ValueError:
        returned = None
    full_precision_gradient = gradient_source == "GRAD"
    return _point_result_from_parsed(
        point,
        type("MRCCEnergyResult", (), {
            "energy_hartree": energy,
            "gradient_hartree_per_bohr": gradient,
        })(),
        f"MRCC {method}/{basis}",
        returned,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": (
                "mrcc_grad_fortran_unformatted"
                if full_precision_gradient
                else ("not_requested" if energy_only else "mrcc_output_fallback")
            ),
            "derivative_transport_lossless_float64": full_precision_gradient,
            "mrcc_gradient_frame": gradient_frame,
        },
        derivatives_in_point_frame=(
            gradient_frame == "input_orientation"
        ),
    )


def _run_cfour_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_cfour import (
        parse_cfour_grd,
        read_cfour_point_result,
        run_cfour_job,
        write_cfour_point_input,
    )

    method = spec.method or "HF"
    basis = spec.basis or "6-31G*"
    input_path = write_cfour_point_input(
        point_dir / "ZMAT",
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        route=spec.route,
        special_basis=spec.basis_file is not None,
    )
    _copy_cfour_genbas(point_dir, spec.env, basis_file=spec.basis_file)
    execution_env = dict(spec.env or {})
    executable_path = Path(spec.executable).expanduser() if spec.executable else None
    if executable_path is not None and executable_path.parent != Path("."):
        executable_parent = str(executable_path.resolve().parent)
        inherited_path = execution_env.get("PATH", os.environ.get("PATH", ""))
        path_entries = inherited_path.split(os.pathsep) if inherited_path else []
        if executable_parent not in path_entries:
            execution_env["PATH"] = os.pathsep.join((executable_parent, *path_entries))
    run = run_cfour_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=execution_env,
        extra_args=spec.extra_args,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_cfour_point_result(point_dir / "GRD", output=run.output_path)
    _numbers, returned_bohr = parse_cfour_grd(point_dir / "GRD")
    returned = np.asarray(returned_bohr, dtype=float) * BOHR_TO_ANGSTROM
    return _point_result_from_parsed(
        point,
        parsed,
        f"CFOUR {method}/{basis}",
        returned,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "cfour_grd_full_precision",
            "derivative_transport_lossless_float64": True,
            "energy_evaluations": 1,
            "gradient_evaluations": 1,
        },
    )


def _run_xtb_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_xtb import read_xtb_point_result, run_xtb_job, write_xtb_point_input

    input_path = write_xtb_point_input(
        point_dir / "xtb.xyz",
        atoms,
        point.coordinates_angstrom,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
    )
    options = list(shlex.split(spec.route)) if spec.route.strip() else ["--gfn", "2"]
    requested = {str(item).strip().lower() for item in spec.properties}
    # A Hessian request also requires the gradient channel for the LINK
    # evaluation contract and for downstream MEX projections.
    needs_gradient = not requested or bool(requested & {"gradient", "hessian"})
    if needs_gradient and "--grad" not in options:
        options.append("--grad")
    if "hessian" in requested and "--hess" not in options:
        options.append("--hess")
    if spec.charge and not any(item in options for item in ("--chrg", "--charge")):
        options.extend(("--chrg", str(spec.charge)))
    if spec.multiplicity > 1 and "--uhf" not in options:
        options.extend(("--uhf", str(spec.multiplicity - 1)))
    options.extend(spec.extra_args)
    resources = CalculationResources(
        process_count=1,
        threads_per_process=max(1, int(spec.processors)),
        memory_per_job_gb=float(spec.memory_gb or 1),
        concurrent_jobs=1,
    )
    run = run_xtb_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=tuple(options),
        resources=resources,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    hessian_run = run
    if "hessian" in requested:
        # xTB accepts --grad and --hess in the command line but only writes
        # the native Hessian bundle when the numerical-Hessian job is run as a
        # separate pass.  Keep the gradient pass for LINK's point contract,
        # then refresh the same work directory with --hess.
        hessian_options = [item for item in options if item != "--grad"]
        hessian_run = run_xtb_job(
            point_dir,
            executable=spec.executable,
            input_path=input_path,
            timeout=spec.timeout,
            env=spec.env,
            extra_args=tuple(hessian_options),
            resources=resources,
        )
        if hessian_run.success is not True:
            raise RuntimeError(hessian_run.message)
    parsed = read_xtb_point_result(
        hessian_run.output_path,
        geometry=input_path,
        require_gradient=needs_gradient,
    )
    if "hessian" in requested:
        from matrix_xtb import hessian_input_from_xtb_files

        hessian = hessian_input_from_xtb_files(
            point_dir / "hessian",
            geometry=input_path,
            spectrum=point_dir / "vibspectrum",
            output=run.output_path,
        ).cartesian_hessian
        parsed = replace(parsed, hessian_hartree_per_bohr2=hessian)
    return _point_result_from_parsed(
        point,
        parsed,
        f"xTB {' '.join(options)}",
        point.coordinates_angstrom,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "xtb_native_derivative_files",
            "derivative_transport_lossless_float64": True,
        },
    )


def _run_matrix_gfn2_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    session: object | None,
) -> PointEvaluationResult:
    requested_mode = str(spec.gradient_mode).strip().casefold()
    requested = (
        {str(item).strip().casefold() for item in spec.properties}
        if spec.properties
        else (
            {"energy"}
            if requested_mode in {"energy", "numerical", "link-numerical"}
            else {"energy", "gradient"}
        )
    )
    unsupported = requested - {"energy", "gradient", "hessian"}
    if unsupported:
        raise ValueError(
            "unsupported MATRIX-GFN2 point properties: "
            + ", ".join(sorted(unsupported))
        )
    include_gradient = "gradient" in requested
    if session is None:
        from .resident_calculators import create_resident_calculator

        session = create_resident_calculator(
            "matrix_gfn2",
            atoms,
            point.coordinates_angstrom,
            charge=spec.charge,
            multiplicity=spec.multiplicity,
            method=spec.method or "MATRIX-GFN2",
            processors=spec.processors,
            solver_mode=spec.matrix_gfn2_solver_mode,
            linear_scaling_threshold=spec.matrix_gfn2_linear_scaling_threshold,
            alpb_solvent=spec.matrix_gfn2_alpb_solvent,
        )
    evaluation = session.evaluate(
        point.coordinates_angstrom,
        **({"compute_hessian": True} if "hessian" in requested else {}),
    )
    statistics = getattr(session, "statistics", None)
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=evaluation.energy_hartree,
        gradient_hartree_per_bohr=(
            evaluation.gradient_hartree_per_bohr if include_gradient else None
        ),
        backend_coordinates_angstrom=point.coordinates_angstrom,
        source="MATRIX-GFN2 (resident)",
        hessian_hartree_per_bohr2=getattr(evaluation, "hessian_hartree_per_bohr2", None),
        execution={
            "engine": "matrix_gfn2",
            "method": spec.method or "MATRIX-GFN2",
            "process_launches": 0,
            "energy_evaluations": 1,
            "gradient_evaluations": int(include_gradient),
            "hessian_evaluations": int("hessian" in requested),
            "hessian_source": "analytic" if "hessian" in requested else None,
            "electronic_state_reuse": True,
            "threads_per_process": int(
                getattr(session, "effective_processors", spec.processors)
            ),
            "requested_threads_per_process": int(spec.processors),
            "parallel_runtime": "MATRIX_CPU_GPU_runtime",
            "solver_mode": spec.matrix_gfn2_solver_mode,
            "alpb_solvent": spec.matrix_gfn2_alpb_solvent,
            "linear_scaling_threshold": spec.matrix_gfn2_linear_scaling_threshold,
            "cache_hit": bool(getattr(evaluation, "cache_hit", False)),
            "resident_backend_evaluations": getattr(
                statistics, "backend_evaluations", None
            ),
            "resident_scc_retries": getattr(statistics, "scc_retries", None),
        },
        # MATRIX-GFN2 retains its native restart state inside the calculator. It
        # is not an AO density/Fock state and must not enter APOC's history.
        scf_state=None,
    )


def _prepare_gfnff_session(spec, atoms, coordinates_angstrom, *, topology_directory=None):
    from .resident_calculators import create_resident_calculator

    name = _normalized_backend_name(spec.name)
    return create_resident_calculator(
        name,
        atoms,
        coordinates_angstrom,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        processors=spec.processors,
        force_field=spec.force_field,
        topology_directory=topology_directory,
        allow_reference=name == "gfnff_reference",
    )


def _run_gfnff_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    session: object | None,
) -> PointEvaluationResult:
    requested_mode = str(spec.gradient_mode).strip().casefold()
    requested = (
        {str(item).strip().casefold() for item in spec.properties}
        if spec.properties
        else (
            {"energy"}
            if requested_mode in {"energy", "numerical", "link-numerical"}
            else {"energy", "gradient"}
        )
    )
    unsupported = requested - {"energy", "gradient"}
    if unsupported:
        raise ValueError(
            "unsupported resident GFN-FF point properties: "
            + ", ".join(sorted(unsupported))
        )
    if session is None:
        session = _prepare_gfnff_session(spec, atoms, point.coordinates_angstrom)
    is_reference = _normalized_backend_name(spec.name) == "gfnff_reference"
    if not is_reference and getattr(session, "engine", None) != "matrix_gfnff":
        raise ValueError("gfnff requires the native calculator; use gfnff_reference for libxtb")
    evaluation = session.evaluate(point.coordinates_angstrom)
    include_gradient = "gradient" in requested
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=evaluation.energy_hartree,
        gradient_hartree_per_bohr=(
            evaluation.gradient_hartree_per_bohr if include_gradient else None
        ),
        backend_coordinates_angstrom=point.coordinates_angstrom,
        source=("GFN-FF (resident libxtb reference)" if is_reference else session.source),
        execution={
            **dict(getattr(evaluation, "execution", {})),
            "engine": "libxtb" if is_reference else "matrix_gfnff",
            "method": "GFN-FF",
            "process_launches": 0,
            "energy_evaluations": 1,
            "gradient_evaluations": int(include_gradient),
            "topology_reuse": True,
            "reference_only": is_reference,
        },
        scf_state=None,
    )


def _run_openmm_scan_point(
    spec: QMScanBackend,
    point: ScanPoint,
    session: object | None,
) -> PointEvaluationResult:
    if session is None:
        raise RuntimeError("OpenMM point evaluation requires a resident session")
    requested = {
        str(item).strip().casefold()
        for item in (spec.properties or ("energy", "gradient"))
    }
    unsupported = requested - {"energy", "gradient"}
    if unsupported:
        raise ValueError(
            "unsupported OpenMM point properties: " + ", ".join(sorted(unsupported))
        )
    evaluation = session.evaluate(point.coordinates_angstrom)
    include_gradient = "gradient" in requested
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=evaluation.energy_hartree,
        gradient_hartree_per_bohr=(
            evaluation.gradient_hartree_per_bohr if include_gradient else None
        ),
        backend_coordinates_angstrom=point.coordinates_angstrom,
        source=f"OpenMM {evaluation.potential} (resident)",
        execution={
            "engine": "openmm",
            "potential": evaluation.potential,
            "process_launches": 0,
            "energy_evaluations": 1,
            "gradient_evaluations": int(include_gradient),
        },
    )


def _run_pyscf_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
    *,
    scf_state_history: Sequence[object] = (),
) -> PointEvaluationResult:
    from matrix_pyscf import (
        hessian_input_from_pyscf_output,
        read_pyscf_output_geometry,
        read_pyscf_point_result,
        run_pyscf_job,
        write_pyscf_hessian_input,
        write_pyscf_point_input,
    )

    method = spec.method or spec.route or "HF"
    basis = spec.basis or "sto-3g"
    requested_mode = str(spec.gradient_mode).strip().casefold()
    if spec.properties:
        requested = set(spec.properties)
    else:
        requested = {"energy"}
        if requested_mode not in {"energy", "numerical", "link-numerical"}:
            requested.add("gradient")
    unsupported = requested - {"energy", "gradient", "hessian"}
    if unsupported:
        raise ValueError(
            "unsupported PySCF point properties: " + ", ".join(sorted(unsupported))
        )
    include_hessian = "hessian" in requested
    include_gradient = "gradient" in requested or include_hessian
    writer = write_pyscf_hessian_input if include_hessian else write_pyscf_point_input
    writer_options = {
        "method": method,
        "reference": spec.reference,
        "basis": basis,
        "charge": spec.charge,
        "multiplicity": spec.multiplicity,
        "dispersion_contract": spec.dispersion_contract,
        "accelerator": (
            str(spec.resolution.get("accelerator", "density_fitting"))
            if isinstance(spec.resolution, Mapping)
            else "density_fitting"
        ),
    }
    if not include_hessian:
        writer_options["_include_gradient"] = include_gradient
    guess_status = "disabled" if not spec.reuse_density_projection else "standard_guess_no_history"
    guess_preparation_started = time.perf_counter()
    guess_path = None
    accepted_states = tuple(scf_state_history[-4:]) if spec.reuse_density_projection else ()
    if accepted_states:
        from matrix_apoc.state_reuse import (
            SCFBackendRequest,
            SCFState,
            extrapolate_same_basis_state,
            reuse_same_basis_state,
            validate_state_for_request,
        )
        from matrix_chem.topology.elements import atomic_number

        if any(
            not isinstance(state, SCFState) or state.backend.casefold() != "pyscf"
            for state in accepted_states
        ):
            raise RuntimeError("PySCF received an incompatible accepted SCF state")
        electron_count = (
            float(accepted_states[-1].electron_count)
            if accepted_states[-1].electron_count is not None
            else float(sum(int(atomic_number(atom) or 0) for atom in atoms) - spec.charge)
        )
        request = SCFBackendRequest(
            backend="pyscf",
            method=method,
            basis=basis,
            charge=int(spec.charge),
            multiplicity=int(spec.multiplicity),
            electronic_state=int(spec.electronic_state),
            coordinates=np.asarray(point.coordinates_angstrom, dtype=float),
            electron_count=electron_count,
        )
        for state in accepted_states:
            validate_state_for_request(state, request)
        try:
            guess = (
                extrapolate_same_basis_state(
                    accepted_states, request.coordinates, request.basis
                )
                if len(accepted_states) >= 2
                else reuse_same_basis_state(
                    accepted_states[-1], request.coordinates, request.basis
                )
            )
            guess_path = point_dir / "scf_guess.npz"
            guess_arrays = {
                name: np.asarray(value, dtype=float)
                for name, value in (
                    ("density_alpha", guess.density_alpha),
                    ("density_beta", guess.density_beta),
                    ("fock_alpha", guess.fock_alpha),
                    ("fock_beta", guess.fock_beta),
                )
                if value is not None
            }
            if "density_alpha" not in guess_arrays:
                raise RuntimeError("PySCF SCF reuse produced no density")
            np.savez(guess_path, **guess_arrays)
            guess_status = guess.status
        except (RuntimeError, ValueError) as exc:
            guess_status = f"standard_guess_fallback:{type(exc).__name__}:{exc}"
            guess_path = None
    state_path = point_dir / "scf_state.npz" if spec.reuse_density_projection else None
    writer_options["_scf_guess_path"] = guess_path
    writer_options["_scf_state_path"] = state_path
    guess_preparation_elapsed = time.perf_counter() - guess_preparation_started
    input_path = writer(
        point_dir / "pyscf_job.py",
        atoms,
        point.coordinates_angstrom,
        **writer_options,
    )
    backend_started = time.perf_counter()
    run = run_pyscf_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
        resources=CalculationResources(
            process_count=1,
            threads_per_process=max(1, int(spec.processors)),
            memory_per_job_gb=float(spec.memory_gb or 1),
            concurrent_jobs=1,
        ),
    )
    backend_elapsed = time.perf_counter() - backend_started
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_pyscf_point_result(run.output_path)
    if include_hessian:
        parsed = replace(
            parsed,
            hessian_hartree_per_bohr2=hessian_input_from_pyscf_output(
                run.output_path,
                input_path=input_path,
            ).cartesian_hessian,
        )
    returned = read_pyscf_output_geometry(run.output_path).coordinates_angstrom
    converged_state = None
    if state_path is not None:
        if not state_path.is_file():
            raise RuntimeError("PySCF did not return the requested binary SCF state")
        from matrix_apoc.state_reuse import SCFState
        with np.load(state_path, allow_pickle=False) as state_arrays:
            density_alpha = np.asarray(state_arrays["density_alpha"], dtype=float)
            density_beta = (
                np.asarray(state_arrays["density_beta"], dtype=float)
                if "density_beta" in state_arrays
                else None
            )
            fock_alpha = np.asarray(state_arrays["fock_alpha"], dtype=float)
            fock_beta = (
                np.asarray(state_arrays["fock_beta"], dtype=float)
                if "fock_beta" in state_arrays
                else None
            )
            overlap = np.asarray(state_arrays["overlap"], dtype=float)
            orbitals_raw = np.asarray(state_arrays["orbitals"], dtype=float)
            occupations = np.asarray(state_arrays["occupations"], dtype=float)
            electron_count = float(np.asarray(state_arrays["electron_count"]).reshape(-1)[0])
        converged_state = SCFState(
            backend="pyscf",
            method=method,
            basis=basis,
            charge=int(spec.charge),
            multiplicity=int(spec.multiplicity),
            electronic_state=int(spec.electronic_state),
            coordinates=np.asarray(returned, dtype=float),
            density_alpha=density_alpha,
            density_beta=density_beta,
            fock_alpha=0.5 * (fock_alpha + fock_alpha.T),
            fock_beta=(
                None if fock_beta is None else 0.5 * (fock_beta + fock_beta.T)
            ),
            orbitals=orbitals_raw if orbitals_raw.ndim == 2 else None,
            occupations=occupations,
            overlap=0.5 * (overlap + overlap.T),
            electron_count=electron_count,
            provenance={"artifact": str(state_path.resolve()), "atoms": tuple(atoms)},
        )
    dispersion_label = (
        "" if spec.dispersion_contract is None else f" + {spec.dispersion_contract}"
    )
    result = _point_result_from_parsed(
        point,
        parsed,
        f"PySCF {method}/{basis}{dispersion_label}",
        returned,
        execution={
            "energy_evaluations": 1,
            "gradient_evaluations": int(include_gradient),
            "hessian_evaluations": int(include_hessian),
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "pyscf_structured_float64",
            "derivative_transport_lossless_float64": True,
            "scf_state_reuse": guess_status,
            "scf_state_source_count": len(accepted_states),
            "scf_cycles": int(getattr(parsed, "scf_cycles", 0)),
            "wall_time_seconds_backend": backend_elapsed,
            "wall_time_seconds_scf_guess_preparation": guess_preparation_elapsed,
            "wall_time_seconds_total": backend_elapsed + guess_preparation_elapsed,
        },
    )
    return replace(result, scf_state=converged_state)


def _run_resident_pyscf_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    session: object,
    *,
    scf_state_history: Sequence[object] = (),
) -> PointEvaluationResult:
    from matrix_chem.topology.elements import atomic_number
    from matrix_apoc.scf_adapters import PySCFSCFAdapter
    from matrix_apoc.state_reuse import SCFBackendRequest, SCFState, SCFStateManager

    method = spec.method or spec.route or "HF"
    basis = spec.basis or "sto-3g"
    requested_mode = str(spec.gradient_mode).strip().casefold()
    requested = (
        {str(item).strip().casefold() for item in spec.properties}
        if spec.properties
        else (
            {"energy"}
            if requested_mode in {"energy", "numerical", "link-numerical"}
            else {"energy", "gradient"}
        )
    )
    unsupported = requested - {"energy", "gradient"}
    if unsupported:
        raise ValueError(
            "unsupported resident PySCF point properties: "
            + ", ".join(sorted(unsupported))
        )
    include_gradient = "gradient" in requested
    electron_count = float(
        sum(int(atomic_number(atom) or 0) for atom in atoms) - spec.charge
    )
    adapter = PySCFSCFAdapter(
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        electronic_state=spec.electronic_state,
        atoms=atoms,
    )
    manager = SCFStateManager(adapter)
    accepted_states = tuple(scf_state_history[-4:])
    if accepted_states:
        if any(
            not isinstance(state, SCFState) or state.backend.casefold() != "pyscf"
            for state in accepted_states
        ):
            raise RuntimeError("PySCF received an incompatible APOC state")
        manager.seed_history(accepted_states)
    guess = manager.prepare_guess(
        SCFBackendRequest(
            backend="pyscf",
            method=method,
            basis=basis,
            charge=spec.charge,
            multiplicity=spec.multiplicity,
            electronic_state=spec.electronic_state,
            coordinates=np.asarray(point.coordinates_angstrom, dtype=float),
            electron_count=electron_count,
        )
    )
    density_guess = None
    if guess.density_alpha is not None:
        density_guess = (
            guess.density_alpha
            if guess.density_beta is None
            else (guess.density_alpha, guess.density_beta)
        )
    started = time.perf_counter()
    evaluation = session.evaluate(
        point.coordinates_angstrom,
        density_guess=density_guess,
        include_gradient=include_gradient,
    )
    elapsed = time.perf_counter() - started
    density = np.asarray(evaluation.density, dtype=float)
    fock = np.asarray(evaluation.fock, dtype=float)
    density_alpha = density[0] if density.ndim == 3 else density
    density_beta = density[1] if density.ndim == 3 else None
    fock_alpha = fock[0] if fock.ndim == 3 else fock
    fock_beta = fock[1] if fock.ndim == 3 else None
    state = adapter.export_state(
        {
            "coordinates": np.asarray(point.coordinates_angstrom, dtype=float),
            "density_alpha": density_alpha,
            "density_beta": density_beta,
            "fock_alpha": 0.5 * (fock_alpha + fock_alpha.T),
            "fock_beta": (
                None if fock_beta is None else 0.5 * (fock_beta + fock_beta.T)
            ),
            "overlap": 0.5 * (evaluation.overlap + evaluation.overlap.T),
            "electron_count": electron_count,
            "provenance": {"calculator": "PySCF gradient scanner"},
        }
    )
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=evaluation.energy_hartree,
        gradient_hartree_per_bohr=evaluation.gradient_hartree_per_bohr,
        backend_coordinates_angstrom=point.coordinates_angstrom,
        source=f"PySCF {method}/{basis} (resident)",
        scf_state=state,
        execution={
            "backend": "pyscf",
            "calculator": "resident_gradient_scanner",
            "process_launches": 0,
            "energy_evaluations": 1,
            "gradient_evaluations": int(include_gradient),
            "scf_cycles": int(evaluation.scf_cycles),
            "electronic_state_reuse": True,
            "scf_state_owner": "APOC",
            "scf_state_reuse": guess.status,
            "scf_state_source_count": guess.source_state_count,
            "wall_time_seconds_backend": elapsed,
            "wall_time_seconds_total": elapsed,
        },
    )


def _run_pyfock_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
    *,
    scf_state_history: Sequence[object] = (),
    session: object | None = None,
) -> PointEvaluationResult:
    """Evaluate a LINK point with PyFock's in-process CPU implementation."""

    del point_dir
    requested_mode = str(spec.gradient_mode).strip().casefold()
    if requested_mode not in {"analytic", "energy", "numerical", "link-numerical"}:
        raise ValueError(
            "PyFock supports analytic gradients for DFT LDA/GGA/meta-GGA; use "
            "gradient_mode='analytic' or 'energy'"
        )
    if session is None:
        from .pyfock_resident import PyFockResidentSession

        session = PyFockResidentSession(
            atoms,
            point.coordinates_angstrom,
            basis_name=str(spec.basis or "sto-3g"),
            method=str(spec.method or spec.route or "HF"),
            charge=int(spec.charge),
            processors=max(1, int(spec.processors)),
        )
    DFT = session.DFT
    DFT_Grad = session.DFT_Grad
    DFT_NumGrad = session.DFT_NumGrad
    Mol = session.Mol
    overlap_mat_symm = session.overlap_mat_symm
    native_projection = True
    try:
        from pyfock.guess_projection import (
            project_density_between_bases, project_operator_between_bases,
        )
    except ImportError:
        source_root = str(
            (spec.env or {}).get("PYFOCK_SOURCE", "")
            or os.environ.get("PYFOCK_SOURCE", "")
        ).strip()
        if source_root:
            source_path = Path(source_root).expanduser()
            if source_path.is_dir() and str(source_path) not in sys.path:
                sys.path.insert(0, str(source_path))
            try:
                from pyfock.guess_projection import (
                    project_density_between_bases, project_operator_between_bases,
                )
            except ImportError:
                native_projection = False
        else:
            native_projection = False
    method = str(spec.method or spec.route or "HF").strip()
    basis_name = str(spec.basis or "sto-3g").strip()
    pyfock_atoms = [
        [symbol, *map(float, xyz)]
        for symbol, xyz in zip(atoms, point.coordinates_angstrom)
    ]
    molecule = session.molecule(point.coordinates_angstrom)
    basis = session.basis(molecule, basis_name)
    previous = tuple(scf_state_history[-4:])
    calculation = session.calculation(molecule, basis, method=method)
    calculation.conv_crit = 1.0e-7
    calculation.max_itr = 100
    density_guess = "core"
    pre_scf_iterations = 0
    initial_guess = str(spec.pyfock_initial_guess).strip().casefold()
    if initial_guess not in {"core", "pbe", "hf_sto3g", "lda_sto3g", "r2scan_sto3g"}:
        raise ValueError(
            "pyfock_initial_guess must be 'core', 'pbe', 'hf_sto3g', "
            "'lda_sto3g' or 'r2scan_sto3g'"
        )
    if not previous and initial_guess == "pbe" and method.casefold() != "pbe":
        pre_scf = DFT(molecule, basis, xc="PBE", ncores=max(1, int(spec.processors)))
        pre_scf.conv_crit = 1.0e-7
        pre_scf.max_itr = 100
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            _pre_energy, pre_density = pre_scf.scf()
        calculation.dmat = np.asarray(pre_density, dtype=float).copy()
        pre_scf_iterations = int(pre_scf.niter)
        density_guess = "pbe_pre_scf"
    sto3g_guess_methods = {
        "hf_sto3g": "HF",
        "lda_sto3g": "LDA",
        "r2scan_sto3g": "R2SCAN",
    }
    if not previous and initial_guess in sto3g_guess_methods:
        guess_method = sto3g_guess_methods[initial_guess]
        hf_molecule = Mol(atoms=pyfock_atoms, charge=int(spec.charge))
        hf_basis = session.basis(hf_molecule, "sto-3g")
        if method.casefold() != guess_method.casefold():
            guess_scf = DFT(
                hf_molecule,
                hf_basis,
                xc=guess_method,
                ncores=max(1, int(spec.processors)),
            )
            if guess_method == "HF":
                # PyFock's DF algorithm 1 has an auxiliary-shape defect for
                # the small HF path; algorithm 3 is the compatible RI-HF path.
                guess_scf.DF_algo = 3
            guess_scf.conv_crit = 1.0e-7
            guess_scf.max_itr = 100
            with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
                _pre_energy, guess_density = guess_scf.scf()
            if native_projection:
                calculation.dmat = project_density_between_bases(
                    guess_density, hf_basis, basis,
                    electron_count=float(molecule.nelectrons),
                )
                pre_scf_iterations = int(guess_scf.niter)
                density_guess = f"{guess_method.casefold()}_sto3g_projected"
            elif basis_name.casefold() == "sto-3g":
                # The AO basis is unchanged, so a direct copy is valid. No
                # cross-basis projection is claimed in this audited fallback.
                calculation.dmat = np.asarray(guess_density, dtype=float).copy()
                pre_scf_iterations = int(guess_scf.niter)
                density_guess = f"{guess_method.casefold()}_sto3g_same_basis_reuse"
            else:
                density_guess = f"{guess_method.casefold()}_sto3g_projection_unavailable_fallback"
    guess_preparation_started = time.perf_counter()
    if previous:
        from matrix_apoc.scf_adapters import PyFockSCFAdapter
        from matrix_apoc.state_reuse import SCFBackendRequest, SCFState, SCFStateManager

        def density_projector(state, _geometry, _target_basis):
            return project_density_between_bases(
                state.density_alpha, state.provenance["native_basis"], basis,
                electron_count=float(molecule.nelectrons),
            )

        def operator_projector(state, _geometry, _target_basis):
            if state.fock_alpha is None:
                return None
            return project_operator_between_bases(
                state.fock_alpha, state.provenance["native_basis"], basis,
            )

        adapter = PyFockSCFAdapter(
            method=method, basis=basis_name, atoms=atoms,
            density_projector=density_projector if native_projection else None,
            operator_projector=operator_projector if native_projection else None,
        )
        states = []
        for state in previous:
            if not isinstance(state, SCFState) or state.backend.casefold() != "pyfock":
                raise RuntimeError("PyFock received an incompatible accepted SCF state")
            states.append(state)
        manager = SCFStateManager(adapter)
        manager.seed_history(states)
        request = SCFBackendRequest(
            backend="pyfock", method=method, basis=basis_name,
            charge=int(spec.charge), multiplicity=int(spec.multiplicity),
            electronic_state=int(spec.electronic_state),
            coordinates=np.asarray(point.coordinates_angstrom, dtype=float),
            electron_count=float(molecule.nelectrons),
        )
        guess = manager.prepare_guess(request)
        if guess.density_alpha is not None:
            calculation.dmat = np.asarray(guess.density_alpha, dtype=float).copy()
            density_guess = guess.status
        if guess.fock_alpha is not None:
            # Re-diagonalize the projected Fock in the current AO basis.  A
            # raw old Fock is not a valid DIIS residual for the new density.
            overlap = overlap_mat_symm(basis)
            eigenvalues, eigenvectors = calculation.solve(
                guess.fock_alpha, overlap, orthogonalize=True
            )
            occupations = calculation.getOcc(molecule, eigenvalues, eigenvectors)
            calculation.dmat = calculation.gen_dm(eigenvectors, occupations)
            density_guess = f"{guess.status}_fock_diagonalization"
    guess_preparation_elapsed = time.perf_counter() - guess_preparation_started
    if method.casefold() == "hf":
        # PyFock's DF algorithm 1 has an auxiliary-shape defect for the
        # small HF path; algorithm 3 is the compatible RI-HF path.
        calculation.DF_algo = 3
    # PyFock 0.1.7 has no quiet flag on DFT.scf(); keep its diagnostics out of
    # LINK's protocol stream while preserving the calculation itself.
    scf_started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        energy, _density = calculation.scf()
    scf_elapsed = time.perf_counter() - scf_started
    gradient = None
    if requested_mode == "analytic":
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            gradient_result = DFT_Grad(calculation, verbose=False).calculate()
        gradient = np.asarray(gradient_result["gradient"], dtype=float)
    elif requested_mode in {"numerical", "link-numerical"}:
        # PyFock's HF path has no analytical nuclear gradient in 0.1.7. Keep
        # the distinction explicit: this channel is used by the optional
        # HF/STO-3G finite-difference Hessian seed.
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            gradient_result = DFT_NumGrad(
                calculation,
                step_size=float(spec.numerical_gradient_step_bohr),
                step_unit="bohr",
                method=str(spec.numerical_gradient_stencil),
                verbose=False,
            ).calculate()
        gradient = np.asarray(gradient_result["gradient"], dtype=float).reshape(-1)
    converged_fock = getattr(calculation, "fock_matrix", None)
    if converged_fock is not None:
        converged_fock = np.asarray(converged_fock, dtype=float)
        converged_fock = 0.5 * (converged_fock + converged_fock.T)
    from matrix_apoc.scf_adapters import PyFockSCFAdapter

    converged_state = PyFockSCFAdapter(
        method=method,
        basis=basis_name,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        electronic_state=spec.electronic_state,
        atoms=atoms,
    ).export_state(
        {
            "coordinates": np.asarray(point.coordinates_angstrom, dtype=float),
            "density_alpha": np.asarray(_density, dtype=float),
            "fock_alpha": converged_fock,
            "electron_count": float(molecule.nelectrons),
            "provenance": {"native_basis": basis},
        }
    )
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=gradient,
        backend_coordinates_angstrom=point.coordinates_angstrom,
        source=f"PyFock {method}/{basis_name}",
        scf_state=converged_state,
        execution={
            "backend": "pyfock",
            "calculator": "resident_basis_and_state_session",
            "process_launches": 0,
            "pyfock_version": "0.1.7",
            "derivative_contract": (
                "pyfock_analytic_gradient"
                if gradient is not None and requested_mode == "analytic"
                else "pyfock_numerical_gradient"
                if gradient is not None
                else "energy_only"
            ),
            "accelerator": "cpu",
            "scf_density_guess": density_guess,
            "scf_iterations": int(calculation.niter),
            "pre_scf_iterations": pre_scf_iterations,
            "native_state_projection": bool(native_projection),
            "scf_state_owner": "APOC",
            "scf_state_source_count": len(previous),
            "wall_time_seconds_scf_guess_preparation": guess_preparation_elapsed,
            "wall_time_seconds_scf": scf_elapsed,
        },
    )


def _run_et_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_et import evaluate_et_point

    method = spec.method or "HF"
    basis = spec.basis or "cc-pVDZ"
    requested_mode = str(spec.gradient_mode).strip().casefold()
    # LINK owns numerical differentiation in optimizer coordinates.  Asking
    # for ``numerical`` therefore makes eT an energy provider; the historical
    # Cartesian 3N finite difference remains available only as the explicit
    # ``cartesian-numerical`` diagnostic mode.
    et_mode = {
        "numerical": "energy",
        "link-numerical": "energy",
        "cartesian-numerical": "numerical",
    }.get(requested_mode, requested_mode)
    et_env = dict(spec.env or {})
    et_threads = str(max(1, int(spec.processors)))
    et_env.update(
        {
            "OMP_NUM_THREADS": et_threads,
            "MKL_NUM_THREADS": et_threads,
            "OPENBLAS_NUM_THREADS": et_threads,
        }
    )
    parsed = evaluate_et_point(
        point_dir,
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        executable=spec.executable,
        timeout=spec.timeout,
        env=et_env,
        extra_args=spec.extra_args,
        gradient_mode=et_mode,
        numerical_step_bohr=spec.numerical_gradient_step_bohr,
        numerical_stencil=spec.numerical_gradient_stencil,
        workers=spec.processors,
        memory_gb=spec.memory_gb or 8,
        electronic_state=spec.electronic_state,
        excited_states=spec.excited_states,
        state_spin=spec.state_spin,
        freeze_core=spec.freeze_core,
        capture_state_vectors=(
            spec.electronic_state > 0 and str(spec.state_tracking).strip().casefold() == "apoc"
        ),
    )
    state_label = (
        "S0"
        if spec.electronic_state == 0
        else f"{spec.state_spin[0].upper()}{spec.electronic_state}"
    )
    return _point_result_from_parsed(
        point,
        parsed,
        f"eT {method}/{basis} state={state_label} gradient={requested_mode}",
        point.coordinates_angstrom,
        # Preserve the backend execution metadata.  In particular, eT stores
        # the APOC state-manifold archive here; dropping it makes every
        # excited-state LINK point fail before state following can start.
        execution=parsed.execution,
    )


def _copy_cfour_genbas(
    point_dir: Path,
    env: dict[str, str] | None,
    *,
    basis_file: Path | None = None,
) -> None:
    if (point_dir / "GENBAS").exists():
        return
    if basis_file is not None:
        source = Path(basis_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"CFOUR GENBAS artifact not found: {source}")
        shutil.copy2(source, point_dir / "GENBAS")
        return
    cfour_home = None
    if env and env.get("MATRIX_CFOUR_HOME"):
        cfour_home = env["MATRIX_CFOUR_HOME"]
    elif os.environ.get("MATRIX_CFOUR_HOME"):
        cfour_home = os.environ["MATRIX_CFOUR_HOME"]
    if not cfour_home:
        return
    source = Path(cfour_home) / "basis" / "GENBAS"
    if source.is_file():
        shutil.copy2(source, point_dir / "GENBAS")


def _point_result_from_parsed(
    point: ScanPoint,
    parsed: object,
    source: str,
    backend_coordinates_angstrom: np.ndarray | None = None,
    execution: Mapping[str, object] | None = None,
    *,
    derivatives_in_point_frame: bool = False,
) -> PointEvaluationResult:
    normal = bool(getattr(parsed, "normal_termination", True))
    if not normal:
        raise RuntimeError(f"{source} did not terminate normally")
    raw_gradient = getattr(parsed, "gradient_hartree_per_bohr", None)
    gradient = None if raw_gradient is None else np.asarray(raw_gradient, dtype=float)
    hessian = getattr(parsed, "hessian_hartree_per_bohr2", None)
    returned = None
    alignment_rms = 0.0
    if backend_coordinates_angstrom is not None:
        returned = np.asarray(backend_coordinates_angstrom, dtype=float)
        if returned.shape != point.coordinates_angstrom.shape:
            raise RuntimeError(f"{source} returned incompatible Cartesian coordinates")
        rotation = kabsch_rotation(returned, point.coordinates_angstrom)
        if gradient is not None and not derivatives_in_point_frame:
            gradient, hessian = rotate_cartesian_derivatives(gradient, rotation, hessian)
        displacement = aligned_cartesian_displacement(point.coordinates_angstrom, returned)
        alignment_rms = float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=float(getattr(parsed, "energy_hartree")),
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        backend_coordinates_angstrom=returned,
        frame_alignment_rms_angstrom=alignment_rms,
        source=source,
        execution={**dict(getattr(parsed, "execution", {})), **dict(execution or {})},
    )


def _format_command(template: str, **values: object) -> list[str]:
    formatted = template.format(**{key: str(value) for key, value in values.items()})
    return shlex.split(formatted)


def _array_derivatives(
    results: Sequence[PointEvaluationResult],
    displacements: np.ndarray,
    *,
    attr: str,
    max_order: int,
) -> tuple[np.ndarray, ...]:
    arrays = [getattr(item, attr) for item in results]
    if any(array is None for array in arrays):
        return ()
    stacked = np.stack([np.asarray(array, dtype=float) for array in arrays])
    fit = fit_taylor_derivatives(displacements, stacked, max_order=max_order)
    return fit.derivatives
