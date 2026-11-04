"""Symmetry-aware branching of an exact index-two transition structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .mode_symmetry import (
    ModeSymmetryAnalysis,
    analyze_mode_symmetry,
    validate_detected_mode_subgroup,
)

if TYPE_CHECKING:
    from .optimizer import OptimizerResult
    from .scan import QMScanBackend


TRANSITION_STATE_BRANCHING_SCHEMA = "matrix.link.transition_state_branching.v2"
TRANSITION_STATE_BRANCHING_CHECKPOINT_SCHEMA = (
    "matrix.link.transition_state_branching.checkpoint.v1"
)
FINAL_CARTESIAN_HESSIAN_SCHEMA = "matrix.link.final_cartesian_hessian.v1"


@dataclass(frozen=True)
class TransitionStateBranchingSettings:
    enabled: bool = False
    allow_symmetry_lowering: bool = False
    maximum_atom_displacement_angstrom: float = 0.05
    negative_frequency_cutoff_cm: float = -1.0e-6
    mode_symmetry_tolerance: float = 1.0e-5
    mode_degeneracy_tolerance_cm: float = 10.0
    reaction_mode_overlap_threshold: float = 0.50
    directions: tuple[int, ...] = (-1, 1)
    resume: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_atom_displacement_angstrom) or (
            self.maximum_atom_displacement_angstrom <= 0.0
        ):
            raise ValueError("TS branch displacement must be positive")
        if not np.isfinite(self.negative_frequency_cutoff_cm) or (
            self.negative_frequency_cutoff_cm >= 0.0
        ):
            raise ValueError("TS branch frequency cutoff must be negative")
        if not np.isfinite(self.mode_symmetry_tolerance) or self.mode_symmetry_tolerance <= 0.0:
            raise ValueError("TS branch mode-symmetry tolerance must be positive")
        if (
            not np.isfinite(self.mode_degeneracy_tolerance_cm)
            or self.mode_degeneracy_tolerance_cm < 0.0
        ):
            raise ValueError("TS branch mode-degeneracy tolerance must be nonnegative")
        if not 0.0 <= self.reaction_mode_overlap_threshold <= 1.0:
            raise ValueError("TS branch reaction-mode overlap threshold must be in [0, 1]")
        directions = tuple(int(value) for value in self.directions)
        if not directions or any(value not in {-1, 1} for value in directions):
            raise ValueError("TS branch directions must contain only -1 and +1")
        if len(set(directions)) != len(directions):
            raise ValueError("TS branch directions must be unique")
        object.__setattr__(self, "directions", directions)


@dataclass(frozen=True)
class TransitionStateBranchSeed:
    atoms: tuple[str, ...]
    source_coordinates_angstrom: np.ndarray
    coordinates_angstrom: np.ndarray
    direction: int
    reaction_mode_index: int
    reaction_frequency_cm: float
    reaction_cartesian_mode: np.ndarray
    distortion_mode_index: int
    distortion_frequency_cm: float
    distortion_cartesian_mode: np.ndarray
    mode_symmetry: ModeSymmetryAnalysis


@dataclass(frozen=True)
class TransitionStateBranchPlan:
    source_hessian: np.ndarray
    source_gradient: np.ndarray | None
    seeds: tuple[TransitionStateBranchSeed, ...]
    parent_point_group: str
    mode_resolution: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionStateBranchAttempt:
    seed: TransitionStateBranchSeed
    xyzin_path: Path
    run_dir: Path
    retained_point_group: str | None = None
    result: OptimizerResult | None = None
    final_reaction_mode_overlap: float | None = None
    error: str | None = None
    overlap_threshold: float = 0.50
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        return bool(
            self.result is not None
            and self.result.converged
            and self.result.exact_final_hessian
            and self.result.final_hessian_index == 1
            and self.final_reaction_mode_overlap is not None
            and self.final_reaction_mode_overlap >= self.overlap_threshold
        )


@dataclass(frozen=True)
class TransitionStateBranchingResult:
    source_result: OptimizerResult
    attempts: tuple[TransitionStateBranchAttempt, ...]
    selected_attempt: TransitionStateBranchAttempt | None
    summary_path: Path

    @property
    def certified(self) -> bool:
        return self.selected_attempt is not None and self.selected_attempt.certified


def _load_exact_hessian(result: OptimizerResult):
    if result.settings.stationary_point != "transition_state":
        raise ValueError("TS branching requires a LINK transition-state result")
    if not result.exact_final_hessian or result.final_cartesian_hessian_path is None:
        raise ValueError("TS branching requires an exact final Cartesian Hessian")
    payload = json.loads(Path(result.final_cartesian_hessian_path).read_text(encoding="utf-8"))
    if payload.get("schema") != FINAL_CARTESIAN_HESSIAN_SCHEMA:
        raise ValueError("unsupported final Cartesian Hessian artifact")
    atoms = tuple(str(atom) for atom in payload["atoms"])
    coordinates = np.asarray(payload["coordinates_angstrom"], dtype=float)
    hessian = np.asarray(payload["hessian"], dtype=float)
    gradient_payload = payload.get("gradient_hartree_per_bohr")
    gradient = None if gradient_payload is None else np.asarray(gradient_payload, dtype=float)
    if atoms != tuple(result.atoms):
        raise ValueError("final Cartesian Hessian atom order differs from TS result")
    if not np.allclose(
        coordinates,
        np.asarray(result.final_coordinates_angstrom, dtype=float),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("final Cartesian Hessian geometry differs from TS result")
    return atoms, coordinates, hessian, gradient


def _normal_modes(atoms, coordinates, hessian):
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number
    from matrix_qm import cartesian_normal_modes_from_hessian

    masses = np.asarray([atomic_mass(int(atomic_number(atom) or 0)) for atom in atoms], dtype=float)
    return cartesian_normal_modes_from_hessian(
        0.5 * (hessian + hessian.T),
        masses,
        np.asarray(coordinates, dtype=float) * 1.8897261254578281,
        project_external=len(atoms) > 1,
        source="LINK exact index-two TS Hessian",
    )


def _stable_value(value: object) -> object:
    if is_dataclass(value):
        return _stable_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _identity_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _stable_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _artifact_record(path: Path | str, *, required: bool = True) -> dict[str, object] | None:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(f"required TS-branching artifact is missing: {resolved}")
        return None
    raw = resolved.read_bytes()
    return {
        "path": str(resolved),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    from matrix_core import atomic_json_write

    atomic_json_write(path, dict(payload), allow_nan=False)


def _mode_frequency(eigenvalue: float) -> float:
    from matrix_qm.normal_modes import HESSIAN_EIGENVALUE_TO_CM

    value = float(eigenvalue)
    return float(np.sign(value) * np.sqrt(abs(value)) * HESSIAN_EIGENVALUE_TO_CM)


def _resolve_imaginary_mode_subspace(
    atoms: tuple[str, ...],
    modes,
    imaginary: np.ndarray,
    symmetry,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float, ModeSymmetryAnalysis, dict[str, object]]:
    """Resolve A1 and transverse vectors invariantly inside a two-mode subspace."""

    from matrix_chem import atomic_mass, cartesian_operation_matrix
    from matrix_chem.topology.elements import atomic_number

    indices = tuple(int(index) for index in imaginary)
    basis = np.asarray(modes.mass_weighted_modes[list(indices)], dtype=float).reshape(2, -1).T
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(2), rtol=0.0, atol=1.0e-8):
        basis, _ = np.linalg.qr(basis)
    projector = np.zeros((2, 2), dtype=float)
    for operation in symmetry.operations:
        matrix = cartesian_operation_matrix(
            operation.rotation,
            tuple(int(atom) - 1 for atom in operation.permutation),
            natoms=len(atoms),
        )
        projector += basis.T @ matrix @ basis
    projector /= float(len(symmetry.operations))
    projector = 0.5 * (projector + projector.T)
    projector_values, projector_vectors = np.linalg.eigh(projector)
    rank_tolerance = max(100.0 * float(tolerance), 1.0e-7)
    if not (projector_values[-1] >= 1.0 - rank_tolerance and projector_values[0] <= rank_tolerance):
        raise ValueError(
            "the index-two imaginary subspace does not contain exactly one "
            "totally symmetric direction"
        )
    reaction_coefficients = projector_vectors[:, -1]
    transverse_coefficients = projector_vectors[:, 0]
    reaction_mw = basis @ reaction_coefficients
    transverse_mw = basis @ transverse_coefficients
    masses = np.repeat(
        np.asarray([atomic_mass(int(atomic_number(atom) or 0)) for atom in atoms], dtype=float),
        3,
    )
    reaction = reaction_mw / np.sqrt(masses)
    transverse = transverse_mw / np.sqrt(masses)
    reaction /= max(float(np.linalg.norm(reaction)), 1.0e-30)
    transverse_analysis = analyze_mode_symmetry(symmetry, transverse, tolerance=tolerance)
    reaction_analysis = analyze_mode_symmetry(symmetry, reaction, tolerance=tolerance)
    if not reaction_analysis.is_totally_symmetric or transverse_analysis.is_totally_symmetric:
        raise ValueError("symmetry-adapted imaginary subspace failed its irrep certificate")
    diagonal = np.asarray(modes.eigenvalues, dtype=float)[list(indices)]
    reaction_eigenvalue = float(reaction_coefficients @ (diagonal * reaction_coefficients))
    transverse_eigenvalue = float(transverse_coefficients @ (diagonal * transverse_coefficients))
    audit = {
        "policy": "group_projector_in_mass_weighted_index_two_subspace",
        "source_mode_indices": [index + 1 for index in indices],
        "projector_eigenvalues": projector_values.tolist(),
        "reaction_coefficients": reaction_coefficients.tolist(),
        "transverse_coefficients": transverse_coefficients.tolist(),
    }
    return (
        reaction,
        transverse,
        _mode_frequency(reaction_eigenvalue),
        _mode_frequency(transverse_eigenvalue),
        transverse_analysis,
        audit,
    )


def _parent_symmetry(source: Path, requested_group: str | None):
    from matrix_chem import read_molecular_symmetry, restrict_molecular_symmetry

    symmetry = read_molecular_symmetry(source)
    retained = str(requested_group or "").strip()
    if not retained:
        try:
            from matrix_smith import read_gic_definition_from_xyzin

            retained = str(read_gic_definition_from_xyzin(source).point_group).strip()
        except (OSError, RuntimeError, TypeError, ValueError):
            retained = str(symmetry.point_group)
    return restrict_molecular_symmetry(symmetry, retained)


def classify_transition_state_modes(
    frequencies_cm,
    analyses: dict[int, ModeSymmetryAnalysis],
    *,
    negative_frequency_cutoff_cm: float = -1.0e-6,
) -> tuple[int, int]:
    """Return ``(reaction, transverse)`` for a qualified index-two TS."""

    frequencies = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    imaginary = np.flatnonzero(frequencies < float(negative_frequency_cutoff_cm))
    if imaginary.size != 2:
        raise ValueError(
            "automatic TS branching requires an exact index-two Hessian; "
            f"found {imaginary.size} imaginary modes"
        )
    missing = [int(index) for index in imaginary if int(index) not in analyses]
    if missing:
        raise ValueError(f"missing mode-symmetry analyses for imaginary modes {missing}")
    reaction = tuple(index for index in imaginary if analyses[int(index)].is_totally_symmetric)
    transverse = tuple(
        index for index in imaginary if not analyses[int(index)].is_totally_symmetric
    )
    if len(reaction) != 1 or len(transverse) != 1:
        raise ValueError(
            "index-two TS branching requires one totally symmetric reaction mode "
            "and one symmetry-breaking imaginary mode"
        )
    return int(reaction[0]), int(transverse[0])


def plan_transition_state_branches(
    source_xyzin: Path | str,
    source_result: OptimizerResult,
    *,
    settings: TransitionStateBranchingSettings,
    parent_symmetry_group: str | None = None,
) -> TransitionStateBranchPlan:
    """Select the transverse imaginary mode and create its signed TS seeds."""

    if not settings.enabled:
        raise PermissionError("TS branching requires settings.enabled=True")
    if not settings.allow_symmetry_lowering:
        raise PermissionError("TS branching requires explicit symmetry-lowering consent")
    atoms, coordinates, hessian, gradient = _load_exact_hessian(source_result)
    modes = _normal_modes(atoms, coordinates, hessian)
    frequencies = np.asarray(modes.frequencies_cm, dtype=float)
    imaginary = np.flatnonzero(frequencies < settings.negative_frequency_cutoff_cm)
    if imaginary.size != 2:
        raise ValueError(
            "automatic TS branching requires an exact index-two Hessian; "
            f"found {imaginary.size} imaginary modes"
        )
    symmetry = _parent_symmetry(Path(source_xyzin), parent_symmetry_group)
    analyses = {
        int(index): analyze_mode_symmetry(
            symmetry,
            np.asarray(modes.cartesian_directions[int(index)], dtype=float),
            tolerance=settings.mode_symmetry_tolerance,
        )
        for index in imaginary
    }
    gap = abs(float(frequencies[int(imaginary[0])] - frequencies[int(imaginary[1])]))
    use_subspace = gap <= settings.mode_degeneracy_tolerance_cm
    if use_subspace:
        (
            reaction_mode,
            distortion_flat,
            reaction_frequency,
            distortion_frequency,
            distortion_analysis,
            mode_resolution,
        ) = _resolve_imaginary_mode_subspace(
            atoms,
            modes,
            imaginary,
            symmetry,
            tolerance=settings.mode_symmetry_tolerance,
        )
        reaction_index, distortion_index = (int(imaginary[0]), int(imaginary[1]))
        distortion_mode = distortion_flat.reshape((len(atoms), 3))
    else:
        reaction_index, distortion_index = classify_transition_state_modes(
            frequencies,
            analyses,
            negative_frequency_cutoff_cm=settings.negative_frequency_cutoff_cm,
        )
        reaction_mode = np.asarray(modes.cartesian_directions[reaction_index], dtype=float).reshape(
            -1
        )
        reaction_mode /= max(float(np.linalg.norm(reaction_mode)), 1.0e-30)
        distortion_mode = np.asarray(
            modes.cartesian_directions[distortion_index], dtype=float
        ).reshape((len(atoms), 3))
        reaction_frequency = float(frequencies[reaction_index])
        distortion_frequency = float(frequencies[distortion_index])
        distortion_analysis = analyses[distortion_index]
        mode_resolution = {
            "policy": "individual_exact_hessian_eigenmodes",
            "source_mode_indices": [reaction_index + 1, distortion_index + 1],
        }
    mode_resolution.update(
        {
            "imaginary_frequency_gap_cm-1": gap,
            "degeneracy_tolerance_cm-1": settings.mode_degeneracy_tolerance_cm,
        }
    )
    maximum = float(np.max(np.linalg.norm(distortion_mode, axis=1), initial=0.0))
    if not np.isfinite(maximum) or maximum <= 1.0e-14:
        raise ValueError("symmetry-breaking TS mode has zero Cartesian norm")
    scaled = distortion_mode / maximum
    seeds = tuple(
        TransitionStateBranchSeed(
            atoms=atoms,
            source_coordinates_angstrom=coordinates.copy(),
            coordinates_angstrom=(
                coordinates
                + float(direction) * settings.maximum_atom_displacement_angstrom * scaled
            ),
            direction=direction,
            reaction_mode_index=reaction_index + 1,
            reaction_frequency_cm=reaction_frequency,
            reaction_cartesian_mode=reaction_mode.copy(),
            distortion_mode_index=distortion_index + 1,
            distortion_frequency_cm=distortion_frequency,
            distortion_cartesian_mode=distortion_mode.reshape(-1).copy(),
            mode_symmetry=distortion_analysis,
        )
        for direction in settings.directions
    )
    return TransitionStateBranchPlan(
        source_hessian=hessian,
        source_gradient=gradient,
        seeds=seeds,
        parent_point_group=str(symmetry.point_group),
        mode_resolution=mode_resolution,
    )


def _reaction_mode_overlap(seed: TransitionStateBranchSeed, result: OptimizerResult) -> float:
    from matrix_chem.geometry_alignment import kabsch_rotation

    atoms, coordinates, hessian, _gradient = _load_exact_hessian(result)
    modes = _normal_modes(atoms, coordinates, hessian)
    frequencies = np.asarray(modes.frequencies_cm, dtype=float)
    negative = np.flatnonzero(frequencies < -1.0e-6)
    if negative.size != 1:
        return 0.0
    final_mode = np.asarray(modes.cartesian_directions[int(negative[0])], dtype=float).reshape(-1)
    final_mode /= max(float(np.linalg.norm(final_mode)), 1.0e-30)
    rotation = kabsch_rotation(seed.source_coordinates_angstrom, coordinates)
    parent_mode = (seed.reaction_cartesian_mode.reshape((-1, 3)) @ rotation).reshape(-1)
    parent_mode /= max(float(np.linalg.norm(parent_mode)), 1.0e-30)
    return abs(float(final_mode @ parent_mode))


def _reaction_kernel_identity(contract) -> tuple[tuple[tuple[int, int], str], ...]:
    """Return the invariant chemical identity of an ORACLE TS kernel."""

    return tuple(
        (tuple(int(atom) for atom in edge.atoms), str(edge.role))
        for edge in contract.reaction_kernel
    )


def _basin_identity(topology_sha256: str, reaction_kernel) -> str:
    return _identity_sha256(
        {
            "topology_sha256": str(topology_sha256),
            "reaction_kernel": _reaction_kernel_identity(reaction_kernel),
        }
    )


def _checkpoint_identity(
    *,
    source: Path,
    source_result,
    settings: TransitionStateBranchingSettings,
    parent_topology: Mapping[str, object],
    parent_contract,
    backend,
    engine_command: str,
) -> dict[str, object]:
    settings_identity = asdict(settings)
    settings_identity.pop("resume", None)
    payload = {
        "source": _artifact_record(source),
        "source_summary": _artifact_record(source_result.summary_path),
        "source_final_cartesian_hessian": _artifact_record(
            source_result.final_cartesian_hessian_path
        ),
        "settings": settings_identity,
        "parent_topology_sha256": parent_topology["topology_sha256"],
        "parent_reaction_kernel": _reaction_kernel_identity(parent_contract),
        "backend_sha256": _identity_sha256(backend),
        "engine_command_sha256": hashlib.sha256(engine_command.encode("utf-8")).hexdigest(),
    }
    return {"payload": payload, "sha256": _identity_sha256(payload)}


def branch_transition_state(
    source_xyzin: Path | str,
    source_result: OptimizerResult,
    *,
    run_dir: Path | str,
    settings: TransitionStateBranchingSettings,
    backend: QMScanBackend | None = None,
    engine_command: str = "",
    timeout: float | None = None,
    parent_symmetry_group: str | None = None,
    irc_target_geometries=(),
) -> TransitionStateBranchingResult:
    """Optimize signed symmetry-lowered TS branches and certify index one."""

    from matrix_chem import (
        read_molecular_symmetry,
        read_oracle_transition_state_geometry_contract,
        topology_snapshot_from_xyzin,
    )
    from matrix_oracle import materialize_optimization_chart_artifact
    from matrix_smith import write_sycart_build_section

    from .optimizer import (
        InitialSymmetryBreakingModeDisplacement,
        coordinate_model_from_xyzin,
        optimize_geometry,
        optimizer_hessian_from_cartesian,
        optimizer_result_to_json,
    )

    source = Path(source_xyzin).expanduser().resolve()
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = plan_transition_state_branches(
        source,
        source_result,
        settings=settings,
        parent_symmetry_group=parent_symmetry_group,
    )
    parent_symmetry = _parent_symmetry(source, parent_symmetry_group)
    parent_contract = read_oracle_transition_state_geometry_contract(source)
    parent_topology = topology_snapshot_from_xyzin(source)
    parent_basin = _basin_identity(str(parent_topology["topology_sha256"]), parent_contract)
    checkpoint_path = root / "transition_state_branching_checkpoint.json"
    checkpoint_identity = _checkpoint_identity(
        source=source,
        source_result=source_result,
        settings=settings,
        parent_topology=parent_topology,
        parent_contract=parent_contract,
        backend=backend,
        engine_command=engine_command,
    )
    resumed_from_checkpoint = False
    previous_attempts: dict[str, object] = {}
    if settings.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "TS branching resume requires transition_state_branching_checkpoint.json"
            )
        try:
            previous_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid TS branching checkpoint") from exc
        if previous_checkpoint.get("identity", {}).get("sha256") != checkpoint_identity["sha256"]:
            raise RuntimeError("TS branching checkpoint identity changed")
        stored_attempts = previous_checkpoint.get("attempts", {})
        if not isinstance(stored_attempts, dict):
            raise RuntimeError("invalid TS branching checkpoint attempts")
        previous_attempts = dict(stored_attempts)
        resumed_from_checkpoint = True
    checkpoint: dict[str, object] = {
        "schema": TRANSITION_STATE_BRANCHING_CHECKPOINT_SCHEMA,
        "identity": checkpoint_identity,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "attempts": previous_attempts,
    }
    _atomic_json(checkpoint_path, checkpoint)
    optimizer_settings = replace(
        source_result.settings,
        stationary_point="transition_state",
        compute_final_hessian=True,
        require_exact_final_hessian=True,
        transition_mode_overlap_threshold=settings.reaction_mode_overlap_threshold,
        resume=settings.resume,
    )
    attempts: list[TransitionStateBranchAttempt] = []
    for seed in plan.seeds:
        label = "minus" if seed.direction < 0 else "plus"
        branch_root = root / f"branch-{label}"
        branch_xyzin = branch_root / "input.xyzin"
        retained_group: str | None = None
        provenance: dict[str, object] = {}
        checkpoint_attempts = checkpoint["attempts"]
        assert isinstance(checkpoint_attempts, dict)
        checkpoint_attempts[label] = {"direction": seed.direction, "status": "running"}
        _atomic_json(checkpoint_path, checkpoint)
        try:
            materialize_optimization_chart_artifact(
                source,
                branch_xyzin,
                seed.atoms,
                seed.coordinates_angstrom,
                task_regime="TRANSITION_STATE",
            )
            branch_contract = read_oracle_transition_state_geometry_contract(branch_xyzin)
            if _reaction_kernel_identity(branch_contract) != _reaction_kernel_identity(
                parent_contract
            ):
                raise ValueError("symmetry-lowered TS branch changed the parent reaction kernel")
            seed_topology = topology_snapshot_from_xyzin(branch_xyzin)
            if seed_topology["topology_sha256"] != parent_topology["topology_sha256"]:
                raise ValueError("symmetry-lowered TS branch changed the parent connectivity")
            if (
                _basin_identity(str(seed_topology["topology_sha256"]), branch_contract)
                != parent_basin
            ):
                raise ValueError("symmetry-lowered TS branch changed the parent basin identity")
            branch_symmetry = read_molecular_symmetry(branch_xyzin)
            validate_detected_mode_subgroup(
                parent_symmetry,
                branch_symmetry,
                seed.mode_symmetry,
            )
            retained_group = str(branch_symmetry.point_group)
            # A TS displacement can make topology-derived internal coordinates
            # singular or chemically ambiguous.  SYCART remains defined from
            # geometry and symmetry alone and provides a non-redundant
            # full vibrational chart for every QM backend.  Keep every irrep:
            # both the reaction mode and the transverse symmetry-breaking mode
            # belong to the physical TS search space.
            sycart = write_sycart_build_section(branch_xyzin)
            chart_identity = {
                "point_group": str(sycart.point_group),
                "vectors": sycart.vectors,
                "irreps": sycart.irreps,
            }
            provenance.update(
                {
                    "seed_input": _artifact_record(branch_xyzin),
                    "seed_topology_sha256": seed_topology["topology_sha256"],
                    "seed_basin_sha256": parent_basin,
                    "sycart_chart_sha256": _identity_sha256(chart_identity),
                }
            )
            active_sycart_labels = tuple(
                f"SYC{index + 1:03d}" for index in range(len(sycart.vectors))
            )
            if not active_sycart_labels:
                raise ValueError("retained TS subgroup has no vibrational SYCART modes")
            model = coordinate_model_from_xyzin(
                branch_xyzin,
                kind="sycart",
                coordinates=active_sycart_labels,
                task_regime="TRANSITION_STATE",
            )
            initial_hessian = optimizer_hessian_from_cartesian(
                plan.source_hessian,
                model,
                coordinates_angstrom=seed.source_coordinates_angstrom,
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
                initial_hessian_source="parent-exact-index-two-cartesian-hessian",
                initial_symmetry_breaking_mode_displacement=(
                    InitialSymmetryBreakingModeDisplacement(
                        source_coordinates_angstrom=seed.source_coordinates_angstrom,
                        source_cartesian_mode=seed.distortion_cartesian_mode,
                        source_mode_index=seed.distortion_mode_index,
                        allow_symmetry_lowering=True,
                    )
                ),
                initial_transition_mode_cartesian=seed.reaction_cartesian_mode,
                irc_target_geometries=irc_target_geometries,
            )
            overlap = _reaction_mode_overlap(seed, result) if result.exact_final_hessian else 0.0
            final_xyzin = branch_root / "final.xyzin"
            materialize_optimization_chart_artifact(
                branch_xyzin,
                final_xyzin,
                tuple(result.atoms),
                np.asarray(result.final_coordinates_angstrom, dtype=float),
                task_regime="TRANSITION_STATE",
            )
            final_contract = read_oracle_transition_state_geometry_contract(final_xyzin)
            final_topology = topology_snapshot_from_xyzin(final_xyzin)
            if final_topology["topology_sha256"] != seed_topology["topology_sha256"]:
                raise ValueError("optimized TS branch changed connectivity")
            final_basin = _basin_identity(str(final_topology["topology_sha256"]), final_contract)
            if final_basin != parent_basin:
                raise ValueError("optimized TS branch left the parent reaction basin")
            cache_path = Path(result.cache_path)
            cache_contract_path = cache_path.parent / "optimizer_cache_contract.json"
            cache_identity = json.loads(cache_contract_path.read_text(encoding="utf-8"))
            provenance.update(
                {
                    "final_input": _artifact_record(final_xyzin),
                    "final_topology_sha256": final_topology["topology_sha256"],
                    "final_basin_sha256": final_basin,
                    "optimizer_summary": _artifact_record(result.summary_path),
                    "optimizer_cache": _artifact_record(cache_path),
                    "optimizer_cache_contract": _artifact_record(cache_contract_path),
                    "optimizer_cache_identity": cache_identity,
                    "runtime_method_manifest": (
                        None
                        if getattr(result, "runtime_method_manifest_path", None) is None
                        else _artifact_record(result.runtime_method_manifest_path)
                    ),
                    "trajectory": (
                        None
                        if getattr(result, "trajectory_path", None) is None
                        else _artifact_record(result.trajectory_path)
                    ),
                    "trace": (
                        None
                        if getattr(result, "trace_path", None) is None
                        else _artifact_record(result.trace_path)
                    ),
                    "final_optimizer_hessian": (
                        None
                        if getattr(result, "final_hessian_path", None) is None
                        else _artifact_record(result.final_hessian_path)
                    ),
                    "final_cartesian_hessian": (
                        None
                        if result.final_cartesian_hessian_path is None
                        else _artifact_record(result.final_cartesian_hessian_path)
                    ),
                }
            )
            attempts.append(
                TransitionStateBranchAttempt(
                    seed=seed,
                    xyzin_path=branch_xyzin,
                    run_dir=branch_root,
                    retained_point_group=retained_group,
                    result=result,
                    final_reaction_mode_overlap=overlap,
                    overlap_threshold=settings.reaction_mode_overlap_threshold,
                    provenance=provenance,
                )
            )
            checkpoint_attempts[label] = {
                "direction": seed.direction,
                "status": "completed",
                "certified": attempts[-1].certified,
                "provenance": provenance,
            }
            _atomic_json(checkpoint_path, checkpoint)
        except (FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as exc:
            attempts.append(
                TransitionStateBranchAttempt(
                    seed=seed,
                    xyzin_path=branch_xyzin,
                    run_dir=branch_root,
                    retained_point_group=retained_group,
                    error=f"{type(exc).__name__}: {exc}",
                    overlap_threshold=settings.reaction_mode_overlap_threshold,
                    provenance=provenance,
                )
            )
            checkpoint_attempts[label] = {
                "direction": seed.direction,
                "status": "failed",
                "error": attempts[-1].error,
                "provenance": provenance,
            }
            _atomic_json(checkpoint_path, checkpoint)
    certified = [attempt for attempt in attempts if attempt.certified]
    selected = (
        min(
            certified,
            key=lambda attempt: (
                float(attempt.result.final_energy_hartree),
                int(attempt.seed.direction),
            ),
        )
        if certified
        else None
    )
    summary_path = root / "transition_state_branching.json"
    payload = {
        "schema": TRANSITION_STATE_BRANCHING_SCHEMA,
        "source": {
            "input": _artifact_record(source),
            "summary": _artifact_record(source_result.summary_path),
            "final_cartesian_hessian": _artifact_record(source_result.final_cartesian_hessian_path),
            "topology_sha256": parent_topology["topology_sha256"],
            "basin_sha256": parent_basin,
        },
        "checkpoint": _artifact_record(checkpoint_path),
        "checkpoint_identity_sha256": checkpoint_identity["sha256"],
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "parent_point_group": plan.parent_point_group,
        "branch_coordinate_kind": "sycart",
        "branch_chart_policy": (
            "independent_full_nonredundant_SYCART_charts_compared_only_by_"
            "Cartesian_stationary_point_certification_and_energy"
        ),
        "mode_resolution": plan.mode_resolution,
        "source_hessian_index": int(source_result.final_hessian_index),
        "attempts": [
            {
                "direction": attempt.seed.direction,
                "reaction_mode_index": attempt.seed.reaction_mode_index,
                "reaction_frequency_cm-1": attempt.seed.reaction_frequency_cm,
                "distortion_mode_index": attempt.seed.distortion_mode_index,
                "distortion_frequency_cm-1": attempt.seed.distortion_frequency_cm,
                "mode_preserving_operations": list(
                    attempt.seed.mode_symmetry.preserving_operation_labels
                ),
                "branch_exchanging_operations": list(
                    attempt.seed.mode_symmetry.branch_exchanging_operation_labels
                ),
                "retained_point_group": attempt.retained_point_group,
                "final_reaction_mode_overlap": attempt.final_reaction_mode_overlap,
                "certified": attempt.certified,
                "error": attempt.error,
                "provenance": attempt.provenance,
                "result": (
                    None if attempt.result is None else optimizer_result_to_json(attempt.result)
                ),
            }
            for attempt in attempts
        ],
        "certified": selected is not None,
        "selected_direction": None if selected is None else selected.seed.direction,
        "selected_summary": (
            None
            if selected is None or selected.result is None
            else str(selected.result.summary_path)
        ),
    }
    _atomic_json(summary_path, payload)
    return TransitionStateBranchingResult(
        source_result=source_result,
        attempts=tuple(attempts),
        selected_attempt=selected,
        summary_path=summary_path,
    )


__all__ = [
    "TRANSITION_STATE_BRANCHING_SCHEMA",
    "TRANSITION_STATE_BRANCHING_CHECKPOINT_SCHEMA",
    "TransitionStateBranchAttempt",
    "TransitionStateBranchPlan",
    "TransitionStateBranchSeed",
    "TransitionStateBranchingResult",
    "TransitionStateBranchingSettings",
    "branch_transition_state",
    "classify_transition_state_modes",
    "plan_transition_state_branches",
]
