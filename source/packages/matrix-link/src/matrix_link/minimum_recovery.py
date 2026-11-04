"""Exact-Hessian negative-mode recovery for LINK minimum searches.

The ordinary LINK optimizer owns one optimization on one coordinate contract.
This module owns the higher-level decision to branch from an independently
classified, non-minimum endpoint.  ORACLE and SMITH remain responsible for
perception and coordinate construction; the QM backend remains responsible
only for Cartesian electronic-structure quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .mode_symmetry import (
    ModeSymmetryAnalysis,
    analyze_mode_symmetry,
    classify_order_parameter_subspace,
    symmetry_inequivalent_order_parameter_directions,
    validate_detected_mode_subgroup,
)

if TYPE_CHECKING:
    from .optimizer import OptimizerResult
    from .scan import QMScanBackend


MINIMUM_RECOVERY_SCHEMA = "matrix.link.minimum_recovery.v3"
FINAL_CARTESIAN_HESSIAN_SCHEMA = "matrix.link.final_cartesian_hessian.v1"


@dataclass(frozen=True)
class MinimumRecoverySettings:
    """Explicit policy for one generation of negative-mode branches."""

    enabled: bool = False
    allow_symmetry_lowering: bool = False
    maximum_atom_displacement_angstrom: float = 0.05
    negative_frequency_cutoff_cm: float = -1.0e-6
    mode_symmetry_tolerance: float = 1.0e-5
    mode_degeneracy_tolerance_cm: float = 10.0
    directions: tuple[int, ...] = (-1, 1)

    def __post_init__(self) -> None:
        amplitude = float(self.maximum_atom_displacement_angstrom)
        cutoff = float(self.negative_frequency_cutoff_cm)
        symmetry_tolerance = float(self.mode_symmetry_tolerance)
        degeneracy_tolerance = float(self.mode_degeneracy_tolerance_cm)
        directions = tuple(int(value) for value in self.directions)
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError("minimum-recovery displacement must be positive")
        if not np.isfinite(cutoff) or cutoff >= 0.0:
            raise ValueError("minimum-recovery frequency cutoff must be negative")
        if not np.isfinite(symmetry_tolerance) or symmetry_tolerance <= 0.0:
            raise ValueError("minimum-recovery mode-symmetry tolerance must be positive")
        if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0.0:
            raise ValueError("minimum-recovery mode-degeneracy tolerance must be nonnegative")
        if not directions or any(value not in {-1, 1} for value in directions):
            raise ValueError("minimum-recovery directions must contain only -1 and +1")
        if len(set(directions)) != len(directions):
            raise ValueError("minimum-recovery directions must be unique")
        object.__setattr__(self, "maximum_atom_displacement_angstrom", amplitude)
        object.__setattr__(self, "negative_frequency_cutoff_cm", cutoff)
        object.__setattr__(self, "mode_symmetry_tolerance", symmetry_tolerance)
        object.__setattr__(self, "mode_degeneracy_tolerance_cm", degeneracy_tolerance)
        object.__setattr__(self, "directions", directions)


@dataclass(frozen=True)
class MinimumRecoveryBranch:
    """One immutable Cartesian seed displaced along an exact negative mode."""

    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    direction: int
    mode_index: int
    frequency_cm: float
    maximum_atom_displacement_angstrom: float
    mode_indices: tuple[int, ...] = ()
    order_parameter_index: int = 1
    order_parameter_coefficients: tuple[float, ...] = ()
    selection_policy: str = "individual_exact_hessian_mode"
    sign_equivalent: bool = False
    mode_symmetry: ModeSymmetryAnalysis | None = None
    selection_audit: dict[str, object] | None = None

    def __post_init__(self) -> None:
        atoms = tuple(str(atom) for atom in self.atoms)
        coordinates = np.asarray(self.coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("minimum-recovery branch must be a finite natoms x 3 geometry")
        if int(self.direction) not in {-1, 1}:
            raise ValueError("minimum-recovery branch direction must be -1 or +1")
        if int(self.mode_index) <= 0:
            raise ValueError("minimum-recovery mode index must be one-based and positive")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "coordinates_angstrom", coordinates.copy())
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(self, "mode_index", int(self.mode_index))
        object.__setattr__(self, "frequency_cm", float(self.frequency_cm))
        mode_indices = tuple(int(index) for index in (self.mode_indices or (self.mode_index,)))
        if any(index <= 0 for index in mode_indices):
            raise ValueError("minimum-recovery source mode indices must be positive")
        object.__setattr__(self, "mode_indices", mode_indices)
        object.__setattr__(self, "order_parameter_index", int(self.order_parameter_index))
        object.__setattr__(
            self,
            "order_parameter_coefficients",
            tuple(float(value) for value in self.order_parameter_coefficients),
        )
        object.__setattr__(
            self,
            "maximum_atom_displacement_angstrom",
            float(self.maximum_atom_displacement_angstrom),
        )


@dataclass(frozen=True)
class MinimumRecoveryAttempt:
    """Auditable outcome of one displaced minimum optimization."""

    branch: MinimumRecoveryBranch
    xyzin_path: Path
    run_dir: Path
    result: OptimizerResult | None = None
    error: str | None = None
    mode_symmetry: ModeSymmetryAnalysis | None = None
    retained_point_group: str | None = None

    @property
    def certified(self) -> bool:
        result = self.result
        return bool(
            result is not None
            and result.converged
            and result.exact_final_hessian
            and result.final_hessian_index == 0
        )


@dataclass(frozen=True)
class MinimumRecoveryResult:
    """Result of a single bidirectional negative-mode recovery generation."""

    source_result: OptimizerResult
    attempts: tuple[MinimumRecoveryAttempt, ...]
    selected_attempt: MinimumRecoveryAttempt | None
    summary_path: Path
    schema: str = MINIMUM_RECOVERY_SCHEMA

    @property
    def certified(self) -> bool:
        return self.selected_attempt is not None and self.selected_attempt.certified


def _geometry_distance_invariant(left: np.ndarray, right: np.ndarray) -> float:
    """Return an atom-order-preserving rigid-frame invariant distance."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape:
        return float("inf")
    first_distances = np.linalg.norm(first[:, None, :] - first[None, :, :], axis=2)
    second_distances = np.linalg.norm(second[:, None, :] - second[None, :, :], axis=2)
    return float(np.max(np.abs(first_distances - second_distances), initial=0.0))


def deduplicate_certified_products(
    attempts: Sequence[MinimumRecoveryAttempt],
    *,
    geometry_tolerance_angstrom: float = 1.0e-4,
    energy_tolerance_hartree: float = 1.0e-8,
) -> tuple[dict[str, object], ...]:
    """Group certified branch products that are physically the same minimum."""

    geometry_tolerance = float(geometry_tolerance_angstrom)
    energy_tolerance = float(energy_tolerance_hartree)
    if geometry_tolerance <= 0.0 or energy_tolerance < 0.0:
        raise ValueError("product deduplication tolerances are invalid")
    groups: list[list[MinimumRecoveryAttempt]] = []
    for attempt_index, attempt in enumerate(attempts):
        if not attempt.certified or attempt.result is None:
            continue
        assigned = False
        for group in groups:
            representative = group[0]
            if representative.result is None:
                continue
            attempt_coordinates = getattr(attempt.result, "final_coordinates_angstrom", None)
            representative_coordinates = getattr(
                representative.result, "final_coordinates_angstrom", None
            )
            if attempt_coordinates is None or representative_coordinates is None:
                continue
            same_geometry = _geometry_distance_invariant(
                attempt_coordinates,
                representative_coordinates,
            ) <= geometry_tolerance
            same_energy = abs(
                float(attempt.result.final_energy_hartree)
                - float(representative.result.final_energy_hartree)
            ) <= energy_tolerance
            if same_geometry and same_energy:
                group.append(attempt)
                assigned = True
                break
        if not assigned:
            groups.append([attempt])
    return tuple(
        {
            "representative": min(
                group,
                key=lambda item: float(item.result.final_energy_hartree),
            ).branch.order_parameter_index,
            "branches": [
                {
                    "attempt_index": next(
                        index for index, candidate in enumerate(attempts) if candidate is item
                    ),
                    "direction": item.branch.direction,
                    "order_parameter_index": item.branch.order_parameter_index,
                    "energy_hartree": float(item.result.final_energy_hartree),
                }
                for item in group
            ],
            "size": len(group),
        }
        for group in groups
    )


def minimum_mode_branches(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    cartesian_hessian_hartree_per_bohr2: np.ndarray,
    *,
    settings: MinimumRecoverySettings,
    symmetry=None,
) -> tuple[MinimumRecoveryBranch, ...]:
    """Return symmetry-inequivalent branches from the lowest unstable block.

    An isolated mode retains the historical bidirectional policy. A degenerate
    two-dimensional unstable irrep is treated as one invariant subspace:
    maximal-isotropy order-parameter rays are quotiented by the parent group,
    and signs related by a parent operation are generated only once.
    """

    if not settings.enabled:
        raise PermissionError("minimum recovery requires settings.enabled=True")
    if not settings.allow_symmetry_lowering:
        raise PermissionError(
            "negative-mode recovery requires explicit allow_symmetry_lowering=True"
        )
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number
    from matrix_qm import cartesian_normal_modes_from_hessian

    labels = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    hessian = np.asarray(cartesian_hessian_hartree_per_bohr2, dtype=float)
    expected = 3 * len(labels)
    if coordinates.shape != (len(labels), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("minimum-recovery source geometry must be finite natoms x 3")
    if hessian.shape != (expected, expected) or not np.all(np.isfinite(hessian)):
        raise ValueError("minimum-recovery Cartesian Hessian has the wrong shape or values")
    masses = np.asarray(
        [atomic_mass(int(atomic_number(atom) or 0)) for atom in labels], dtype=float
    )
    modes = cartesian_normal_modes_from_hessian(
        0.5 * (hessian + hessian.T),
        masses,
        coordinates * 1.8897261254578281,
        project_external=len(labels) > 1,
        source="LINK exact final minimum Hessian",
    )
    frequencies = np.asarray(modes.frequencies_cm, dtype=float)
    negative = np.flatnonzero(frequencies < settings.negative_frequency_cutoff_cm)
    if negative.size == 0:
        raise ValueError("minimum-recovery source Hessian has no qualifying negative mode")
    mode_index = int(negative[np.argmin(frequencies[negative])])
    cluster = np.asarray(
        [
            int(index)
            for index in negative
            if abs(float(frequencies[int(index)] - frequencies[mode_index]))
            <= settings.mode_degeneracy_tolerance_cm
        ],
        dtype=int,
    )
    amplitude = settings.maximum_atom_displacement_angstrom
    if cluster.size == 1:
        vector = np.asarray(modes.cartesian_directions[mode_index], dtype=float).reshape(
            (len(labels), 3)
        )
        vector -= np.sum(masses[:, None] * vector, axis=0) / np.sum(masses)
        maximum = float(np.max(np.linalg.norm(vector, axis=1), initial=0.0))
        if not np.isfinite(maximum) or maximum <= 1.0e-14:
            raise ValueError("selected minimum-recovery mode has zero Cartesian norm")
        unit_mode = vector / maximum
        analysis = (
            None
            if symmetry is None
            else analyze_mode_symmetry(
                symmetry,
                vector,
                tolerance=settings.mode_symmetry_tolerance,
            )
        )
        return tuple(
            MinimumRecoveryBranch(
                atoms=labels,
                coordinates_angstrom=coordinates + float(direction) * amplitude * unit_mode,
                direction=direction,
                mode_index=mode_index + 1,
                frequency_cm=float(frequencies[mode_index]),
                maximum_atom_displacement_angstrom=amplitude,
                mode_indices=(mode_index + 1,),
                order_parameter_coefficients=(1.0,),
                mode_symmetry=analysis,
            )
            for direction in settings.directions
        )
    if cluster.size != 2:
        raise ValueError(
            "minimum recovery currently requires an isolated mode or a "
            "two-dimensional degenerate unstable irrep"
        )
    if symmetry is None:
        raise ValueError("degenerate minimum recovery requires the frozen parent symmetry")
    basis = np.asarray(modes.mass_weighted_modes[list(cluster)], dtype=float).reshape(2, -1).T
    subspace_analysis = classify_order_parameter_subspace(
        symmetry,
        basis,
        tolerance=settings.mode_symmetry_tolerance,
    )
    if subspace_analysis.reducible:
        branches: list[MinimumRecoveryBranch] = []
        for index in cluster:
            vector = np.asarray(modes.cartesian_directions[int(index)], dtype=float).reshape(
                (len(labels), 3)
            )
            vector -= np.sum(masses[:, None] * vector, axis=0) / np.sum(masses)
            maximum = float(np.max(np.linalg.norm(vector, axis=1), initial=0.0))
            if maximum <= 1.0e-14:
                raise ValueError("accidentally degenerate mode has zero Cartesian norm")
            analysis = analyze_mode_symmetry(
                symmetry,
                vector,
                tolerance=settings.mode_symmetry_tolerance,
            )
            for direction in settings.directions:
                branches.append(
                    MinimumRecoveryBranch(
                        atoms=labels,
                        coordinates_angstrom=coordinates
                        + float(direction) * amplitude * (vector / maximum),
                        direction=direction,
                        mode_index=int(index) + 1,
                        frequency_cm=float(frequencies[int(index)]),
                        maximum_atom_displacement_angstrom=amplitude,
                        mode_indices=(int(index) + 1,),
                        order_parameter_coefficients=(1.0,),
                        selection_policy="accidental_degeneracy_individual_modes",
                        mode_symmetry=analysis,
                        selection_audit={
                            "policy": "accidental_degeneracy_individual_modes",
                            "cluster_indices": [int(value) + 1 for value in cluster],
                            "subspace_analysis": {
                                "character_norm": subspace_analysis.character_norm,
                                "representation_source": subspace_analysis.representation_source,
                            },
                        },
                    )
                )
        return tuple(branches)
    order_parameters, selection_audit = symmetry_inequivalent_order_parameter_directions(
        symmetry,
        basis,
        masses,
        tolerance=settings.mode_symmetry_tolerance,
    )
    eigenvalues = np.asarray(modes.eigenvalues, dtype=float)[list(cluster)]
    from matrix_qm.normal_modes import HESSIAN_EIGENVALUE_TO_CM

    branches: list[MinimumRecoveryBranch] = []
    for order_parameter_index, order_parameter in enumerate(order_parameters, start=1):
        vector = np.asarray(order_parameter.cartesian_direction, dtype=float).reshape(
            (len(labels), 3)
        )
        vector -= np.sum(masses[:, None] * vector, axis=0) / np.sum(masses)
        maximum = float(np.max(np.linalg.norm(vector, axis=1), initial=0.0))
        if not np.isfinite(maximum) or maximum <= 1.0e-14:
            raise ValueError("selected order-parameter direction has zero Cartesian norm")
        unit_mode = vector / maximum
        coefficients = np.asarray(order_parameter.coefficients, dtype=float)
        eigenvalue = float(coefficients @ (eigenvalues * coefficients))
        frequency = float(np.sign(eigenvalue) * np.sqrt(abs(eigenvalue)) * HESSIAN_EIGENVALUE_TO_CM)
        directions = (1,) if order_parameter.sign_equivalent else settings.directions
        for direction in directions:
            branches.append(
                MinimumRecoveryBranch(
                    atoms=labels,
                    coordinates_angstrom=(coordinates + float(direction) * amplitude * unit_mode),
                    direction=direction,
                    mode_index=int(cluster[0]) + 1,
                    frequency_cm=frequency,
                    maximum_atom_displacement_angstrom=amplitude,
                    mode_indices=tuple(int(index) + 1 for index in cluster),
                    order_parameter_index=order_parameter_index,
                    order_parameter_coefficients=order_parameter.coefficients,
                    selection_policy=order_parameter.selection_policy,
                    sign_equivalent=order_parameter.sign_equivalent,
                    mode_symmetry=order_parameter.mode_symmetry,
                    selection_audit=selection_audit,
                )
            )
    return tuple(branches)


def branches_from_optimizer_result(
    result: OptimizerResult,
    *,
    settings: MinimumRecoverySettings,
    symmetry=None,
) -> tuple[MinimumRecoveryBranch, ...]:
    """Load and validate LINK's exact final Hessian before branching."""

    if result.settings.stationary_point != "minimum":
        raise ValueError("minimum recovery requires a LINK minimum result")
    if not result.exact_final_hessian or result.final_cartesian_hessian_path is None:
        raise ValueError("minimum recovery requires an exact final Cartesian Hessian")
    source = Path(result.final_cartesian_hessian_path)
    if not source.is_file():
        raise FileNotFoundError(f"final Cartesian Hessian artifact does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != FINAL_CARTESIAN_HESSIAN_SCHEMA:
        raise ValueError("unsupported final Cartesian Hessian artifact")
    atoms = tuple(str(atom) for atom in payload.get("atoms", ()))
    coordinates = np.asarray(payload.get("coordinates_angstrom"), dtype=float)
    hessian = np.asarray(payload.get("hessian"), dtype=float)
    if atoms != tuple(result.atoms):
        raise ValueError("final Cartesian Hessian atom order differs from optimizer result")
    if coordinates.shape != np.asarray(result.final_coordinates_angstrom).shape or not np.allclose(
        coordinates,
        np.asarray(result.final_coordinates_angstrom, dtype=float),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("final Cartesian Hessian geometry differs from optimizer result")
    return minimum_mode_branches(
        atoms,
        coordinates,
        hessian,
        settings=settings,
        symmetry=symmetry,
    )


def recover_minimum(
    source_xyzin: Path | str,
    source_result: OptimizerResult,
    *,
    run_dir: Path | str,
    settings: MinimumRecoverySettings,
    backend: QMScanBackend | None = None,
    engine_command: str = "",
    timeout: float | None = None,
    parent_symmetry_group: str | None = None,
) -> MinimumRecoveryResult:
    """Run one bidirectional symmetry-aware recovery from a failed minimum.

    LINK derives the normal-mode isotropy subgroup from the parent operations;
    ORACLE must independently recognize the same subgroup at each displaced
    geometry before SMITH builds its complete all-irrep SYCART chart. Branches are optimized
    independently and are selectable only after a fresh exact Cartesian
    Hessian certifies index zero.
    """

    from matrix_chem import read_molecular_symmetry, restrict_molecular_symmetry
    from matrix_oracle import materialize_optimization_chart_artifact
    from matrix_smith import write_sycart_build_section

    from .optimizer import (
        coordinate_model_from_xyzin,
        optimize_geometry,
        optimizer_hessian_from_cartesian,
        optimizer_result_to_json,
    )

    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(source_xyzin).expanduser().resolve()
    parent_symmetry = read_molecular_symmetry(source)
    retained_parent_group = str(parent_symmetry_group or "").strip()
    if not retained_parent_group:
        try:
            from matrix_smith import read_gic_definition_from_xyzin

            retained_parent_group = str(read_gic_definition_from_xyzin(source).point_group).strip()
        except (OSError, RuntimeError, TypeError, ValueError):
            retained_parent_group = str(parent_symmetry.point_group)
    parent_symmetry = restrict_molecular_symmetry(
        parent_symmetry,
        retained_parent_group,
    )
    branches = branches_from_optimizer_result(
        source_result,
        settings=settings,
        symmetry=parent_symmetry,
    )
    source_hessian_payload = json.loads(
        Path(source_result.final_cartesian_hessian_path).read_text(encoding="utf-8")
    )
    source_hessian = np.asarray(source_hessian_payload["hessian"], dtype=float)
    optimizer_settings = replace(
        source_result.settings,
        stationary_point="minimum",
        compute_final_hessian=True,
        require_exact_final_hessian=True,
        symmetry_lowered_branch_all_irreps=True,
        resume=False,
    )
    attempts: list[MinimumRecoveryAttempt] = []
    for branch in branches:
        label = "minus" if branch.direction < 0 else "plus"
        branch_label = (
            f"branch-{label}"
            if branch.selection_policy == "individual_exact_hessian_mode"
            else f"order-{branch.order_parameter_index:02d}-{label}"
        )
        branch_root = root / branch_label
        branch_root.mkdir(parents=True, exist_ok=True)
        branch_xyzin = branch_root / "input.xyzin"
        mode_symmetry = branch.mode_symmetry or analyze_mode_symmetry(
            parent_symmetry,
            branch.coordinates_angstrom
            - np.asarray(source_result.final_coordinates_angstrom, dtype=float),
            tolerance=settings.mode_symmetry_tolerance,
        )
        retained_point_group: str | None = None
        try:
            materialize_optimization_chart_artifact(
                source,
                branch_xyzin,
                branch.atoms,
                branch.coordinates_angstrom,
                task_regime="MINIMUM",
            )
            branch_symmetry = read_molecular_symmetry(branch_xyzin)
            validate_detected_mode_subgroup(
                parent_symmetry,
                branch_symmetry,
                mode_symmetry,
            )
            retained_point_group = str(branch_symmetry.point_group)
            sycart = write_sycart_build_section(branch_xyzin)
            active_sycart_labels = tuple(
                f"SYC{index + 1:03d}" for index in range(len(sycart.vectors))
            )
            if not active_sycart_labels:
                raise ValueError("retained order-parameter subgroup has no SYCART modes")
            model = coordinate_model_from_xyzin(
                branch_xyzin,
                kind="sycart",
                coordinates=active_sycart_labels,
                task_regime="SYMMETRY_LOWERED_BRANCH",
            )
            initial_hessian = optimizer_hessian_from_cartesian(
                source_hessian,
                model,
                coordinates_angstrom=np.asarray(
                    source_result.final_coordinates_angstrom, dtype=float
                ),
            )
            result = optimize_geometry(
                branch_xyzin,
                run_dir=branch_root / "optimization",
                coordinate_model=model,
                engine_command=engine_command,
                backend=backend,
                timeout=timeout,
                settings=optimizer_settings,
                initial_hessian=initial_hessian,
                initial_hessian_source=("parent-exact-higher-index-cartesian-hessian"),
            )
            attempts.append(
                MinimumRecoveryAttempt(
                    branch=branch,
                    xyzin_path=branch_xyzin,
                    run_dir=branch_root,
                    result=result,
                    mode_symmetry=mode_symmetry,
                    retained_point_group=retained_point_group,
                )
            )
        except (FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as exc:
            attempts.append(
                MinimumRecoveryAttempt(
                    branch=branch,
                    xyzin_path=branch_xyzin,
                    run_dir=branch_root,
                    error=f"{type(exc).__name__}: {exc}",
                    mode_symmetry=mode_symmetry,
                    retained_point_group=retained_point_group,
                )
            )
    certified = [attempt for attempt in attempts if attempt.certified]
    product_groups = deduplicate_certified_products(certified)
    unique_certified = []
    for group in product_groups:
        group_indices = {item["attempt_index"] for item in group["branches"]}
        branches_in_group = [attempt for index, attempt in enumerate(certified) if index in group_indices]
        if branches_in_group:
            unique_certified.append(
                min(branches_in_group, key=lambda item: float(item.result.final_energy_hartree))
            )
    selected = (
        min(unique_certified, key=lambda attempt: float(attempt.result.final_energy_hartree))
        if unique_certified
        else None
    )
    summary_path = root / "minimum_recovery.json"
    payload = {
        "schema": MINIMUM_RECOVERY_SCHEMA,
        "source_summary": str(source_result.summary_path),
        "source_hessian_index": int(source_result.final_hessian_index),
        "source_frequencies_cm-1": list(source_result.final_frequencies_cm),
        "settings": {
            "maximum_atom_displacement_angstrom": settings.maximum_atom_displacement_angstrom,
            "negative_frequency_cutoff_cm": settings.negative_frequency_cutoff_cm,
            "mode_symmetry_tolerance": settings.mode_symmetry_tolerance,
            "mode_degeneracy_tolerance_cm-1": settings.mode_degeneracy_tolerance_cm,
            "directions": list(settings.directions),
            "allow_symmetry_lowering": settings.allow_symmetry_lowering,
        },
        "attempts": [
            {
                "direction": attempt.branch.direction,
                "mode_index": attempt.branch.mode_index,
                "mode_indices": list(attempt.branch.mode_indices),
                "frequency_cm-1": attempt.branch.frequency_cm,
                "order_parameter_index": attempt.branch.order_parameter_index,
                "order_parameter_coefficients": list(attempt.branch.order_parameter_coefficients),
                "order_parameter_selection_policy": attempt.branch.selection_policy,
                "sign_equivalent_under_parent_group": attempt.branch.sign_equivalent,
                "order_parameter_selection_audit": attempt.branch.selection_audit,
                "coordinate_kind": "sycart",
                "xyzin": str(attempt.xyzin_path),
                "run_dir": str(attempt.run_dir),
                "certified": attempt.certified,
                "error": attempt.error,
                "parent_point_group": (
                    None
                    if attempt.mode_symmetry is None
                    else attempt.mode_symmetry.parent_point_group
                ),
                "retained_point_group": attempt.retained_point_group,
                "mode_preserving_operations": (
                    []
                    if attempt.mode_symmetry is None
                    else list(attempt.mode_symmetry.preserving_operation_labels)
                ),
                "branch_exchanging_operations": (
                    []
                    if attempt.mode_symmetry is None
                    else list(attempt.mode_symmetry.branch_exchanging_operation_labels)
                ),
                "broken_operations": (
                    []
                    if attempt.mode_symmetry is None
                    else list(attempt.mode_symmetry.broken_operation_labels)
                ),
                "result": (
                    None if attempt.result is None else optimizer_result_to_json(attempt.result)
                ),
            }
            for attempt in attempts
        ],
        "certified_product_groups": list(product_groups),
        "certified": selected is not None,
        "selected_direction": None if selected is None else selected.branch.direction,
        "selected_summary": (
            None
            if selected is None or selected.result is None
            else str(selected.result.summary_path)
        ),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MinimumRecoveryResult(
        source_result=source_result,
        attempts=tuple(attempts),
        selected_attempt=selected,
        summary_path=summary_path,
    )


__all__ = [
    "FINAL_CARTESIAN_HESSIAN_SCHEMA",
    "MINIMUM_RECOVERY_SCHEMA",
    "MinimumRecoveryAttempt",
    "MinimumRecoveryBranch",
    "MinimumRecoveryResult",
    "MinimumRecoverySettings",
    "branches_from_optimizer_result",
    "minimum_mode_branches",
    "recover_minimum",
]
