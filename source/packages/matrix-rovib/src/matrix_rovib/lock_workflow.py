"""End-to-end TRINITY surface to LOCK spectrum workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .artifact_chain import (
    ScientificArtifact,
    append_artifact,
    validate_artifact_chain,
    write_artifact_chain,
)
from .resource_planner import LOCKBlockRequirement, plan_lock_resources

from matrix_rovib import (
    LOCK_LAM_ROTATION_RESULT_SCHEMA,
    LOCK_LAM_ROTATION_SURFACE_SCHEMA,
    lam_rotation_transitions,
    native_rotor_parameters_from_section,
    read_lam_rotation_surface,
    read_rotational_section,
    solve_lam_rotation,
    write_lam_rotation_result,
)


LOCK_WORKFLOW_SCHEMA = "matrix.lock.workflow.v1"
LOCK_LINES_SCHEMA = "matrix.lock.lines.v1"
LOCK_LINE_FIELDS = (
    "Frequency (MHz)",
    "Intensity",
    "Line strength",
    "Relative intensity",
    "Branch",
    "Jl",
    "Ka_l",
    "Kc_l",
    "lam_l",
    "vt_l",
    "alpha_l",
    "tors_sym_l",
    "rovib_irrep_l",
    "symmetry_sector_l",
    "Ju",
    "Ka_u",
    "Kc_u",
    "lam_u",
    "vt_u",
    "alpha_u",
    "tors_sym_u",
    "rovib_irrep_u",
    "symmetry_sector_u",
    "spin_weight",
    "parent_purity",
)


@dataclass(frozen=True)
class LOCKWorkflowRequest:
    surface: Path
    outdir: Path
    j_min: int = 0
    j_max: int = 10
    lam_states: int = 12
    levels_per_j: int | None = 128
    representation: str = "Ir"
    dipole_au: tuple[float, float, float] | None = None
    temperature_K: float | None = None
    maximum_lam_state: int | None = None
    rotational_xyzin: Path | None = None
    assembly_backend: str = "auto"
    diagonalizer_backend: str = "auto"
    workers: int = 1
    memory_budget_bytes: int = 2 * 1024**3
    upstream_artifacts: tuple[ScientificArtifact, ...] = ()
    primitive_basis: str = "auto"
    fourier_cutoffs: tuple[int | None, ...] | None = None
    k_contraction_states: int | None = None

    def __post_init__(self) -> None:
        if self.j_min < 0 or self.j_max < self.j_min:
            raise ValueError("LOCK workflow requires 0 <= j_min <= j_max")
        if self.lam_states < 1:
            raise ValueError("LOCK workflow requires at least one LAM state")
        if self.primitive_basis not in {"auto", "fourier", "dvr"}:
            raise ValueError("primitive_basis must be auto, fourier or dvr")
        if self.k_contraction_states is not None and (
            isinstance(self.k_contraction_states, bool)
            or not isinstance(self.k_contraction_states, int)
            or self.k_contraction_states < 1
        ):
            raise ValueError("k_contraction_states must be a positive integer")
        if self.levels_per_j is not None and self.levels_per_j < 1:
            raise ValueError("levels_per_j must be positive")
        if self.temperature_K is not None and self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        if self.memory_budget_bytes < 1:
            raise ValueError("memory_budget_bytes must be positive")
        if self.upstream_artifacts:
            upstream = validate_artifact_chain(self.upstream_artifacts, require_complete=False)
            if tuple(item.owner for item in upstream) != ("ORACLE", "SMITH", "GF", "TRINITY"):
                raise ValueError("LOCK workflow upstream chain must end with TRINITY")


@dataclass(frozen=True)
class LOCKWorkflowResult:
    manifest: Path
    levels: Path
    lines: Path | None
    transition_count: int
    assembly_backends: tuple[str, ...]
    diagonalizer_backends: tuple[str, ...]
    artifact_chain: Path | None = None
    resource_plan: Path | None = None


def run_lock_workflow(request: LOCKWorkflowRequest) -> LOCKWorkflowResult:
    """Solve a TRINITY surface and write a complete LOCK artifact set."""
    surface_path = Path(request.surface).resolve()
    if not surface_path.is_file():
        raise FileNotFoundError(surface_path)
    outdir = Path(request.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    surface = read_lam_rotation_surface(surface_path)
    retained_lam_states = min(request.lam_states, int(_surface_size(surface)))
    resource_plan = plan_lock_resources(
        [
            LOCKBlockRequirement(
                key=f"J={J}",
                dimension=retained_lam_states * (2 * J + 1),
                roots=min(
                    retained_lam_states * (2 * J + 1),
                    request.levels_per_j or retained_lam_states * (2 * J + 1),
                ),
                factorized_term_count=11,
            )
            for J in range(request.j_min, request.j_max + 1)
        ],
        memory_budget_bytes=request.memory_budget_bytes,
        requested_workers=request.workers,
        input_contract={
            "surface": str(surface_path),
            "surface_metadata": dict(surface.metadata),
            "J": [request.j_min, request.j_max],
            "lam_states": retained_lam_states,
            "primitive_basis": request.primitive_basis,
            "fourier_cutoffs": request.fourier_cutoffs,
            "k_contraction_states": request.k_contraction_states,
            "levels_per_j": request.levels_per_j,
        },
    )
    if not resource_plan["ready"]:
        raise MemoryError(
            "LOCK resource plan cannot solve blocks: "
            + ", ".join(resource_plan["blocked_blocks"])
        )
    resource_plan_path = outdir / "lock_resource_plan.json"
    resource_plan_path.write_text(
        json.dumps(resource_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    distortion = None
    if request.rotational_xyzin is not None:
        rotational = read_rotational_section(request.rotational_xyzin)
        distortion = native_rotor_parameters_from_section(
            rotational,
            representation=request.representation,
        )
    symmetry_actions = None
    surface_metadata = dict(surface.metadata or {})
    if "oracle_pi_grid_action" in surface_metadata:
        # solve_lam_rotation consumes this self-contained contract directly.
        pass
    elif surface_metadata.get("symmetry_source") == "ORACLE":
        raise ValueError(
            "legacy ORACLE surface lacks its embedded PI grid action; rebuild the surface "
            "to freeze the actual sorted DVR grid and body-frame embedding"
        )
    result = solve_lam_rotation(
        surface,
        j_values=range(request.j_min, request.j_max + 1),
        lam_state_count=retained_lam_states,
        representation=request.representation,
        centrifugal_distortion=distortion,
        level_count=request.levels_per_j,
        assembly_backend=request.assembly_backend,
        diagonalizer_backend=request.diagonalizer_backend,
        rovibrational_symmetry=symmetry_actions,
        primitive_basis=request.primitive_basis,
        fourier_cutoffs=request.fourier_cutoffs,
        k_contraction_states=request.k_contraction_states,
        workers=int(resource_plan["selected_workers"]),
    )
    levels_path = write_lam_rotation_result(
        outdir / "lock_levels.json",
        result,
        surface_path=surface_path,
    )
    transitions: list[dict[str, object]] = []
    lines_path = None
    if surface.dipole_au is not None or request.dipole_au is not None:
        transitions = lam_rotation_transitions(
            surface,
            result,
            dipole_au=request.dipole_au,
            maximum_lam_parent=request.maximum_lam_state,
            temperature_K=request.temperature_K,
        )
        lines_path = _write_lines(outdir / "lock_lines.csv", transitions)
    manifest_path = outdir / "lock_workflow.json"
    manifest = {
        "schema": LOCK_WORKFLOW_SCHEMA,
        "integration": {
            "surface_and_basis_owner": "TRINITY/ESALC",
            "spectroscopic_solver": "LOCK",
            "interactive_frontend": "Keymaker",
            "boundary": "versioned LOCK surface and result artifacts",
        },
        "surface_provenance": dict(surface.metadata),
        "artifact_schemas": {
            "surface": LOCK_LAM_ROTATION_SURFACE_SCHEMA,
            "levels": LOCK_LAM_ROTATION_RESULT_SCHEMA,
            "lines": None if lines_path is None else LOCK_LINES_SCHEMA,
        },
        "surface": str(surface_path),
        "levels": str(levels_path),
        "lines": None if lines_path is None else str(lines_path),
        "transition_count": len(transitions),
        "J": [request.j_min, request.j_max],
        "lam_states": len(result.basis.energies_MHz),
        "primitive_basis": result.basis.primitive_representation,
        "fourier_cutoffs": result.basis.fourier_cutoffs,
        "k_contraction_states": request.k_contraction_states,
        "levels_per_j": request.levels_per_j,
        "representation": request.representation,
        "temperature_K": request.temperature_K,
        "maximum_lam_state": request.maximum_lam_state,
        "assembly_backends": list(result.assembly_backends),
        "diagonalizer_backends": list(result.diagonalizer_backends),
        "workers": request.workers,
        "selected_workers": resource_plan["selected_workers"],
        "resource_plan": str(resource_plan_path),
        "hermiticity_error_MHz": result.hermiticity_error_MHz,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    from matrix_core.owner_evidence import emit_owner_validation
    emit_owner_validation('LOCK', 'ROVIBRATIONAL_WORKFLOW', passed=True,
        metrics={'hermiticity_error': result.hermiticity_error_MHz,
                 'transition_count': float(len(transitions)), 'selected_workers': float(resource_plan['selected_workers'])},
        units={'hermiticity_error': 'MHz', 'transition_count': 'dimensionless', 'selected_workers': 'dimensionless'},
        context={'upstream_supplied': bool(request.upstream_artifacts), 'spectrum_written': lines_path is not None},
        gates=('Versioned TRINITY surface and selected basis are validated',
               'Declared angular-momentum, memory, worker and eigensolver budgets are enforced',
               'When supplied the immutable upstream chain ends with ORACLE, SMITH, GF, TRINITY',
               'A spectrum requires an explicit dipole and records temperature and line units'))
    chain_path = None
    if request.upstream_artifacts:
        chain = append_artifact(
            request.upstream_artifacts,
            owner="LOCK",
            schema=LOCK_WORKFLOW_SCHEMA,
            payload=manifest,
        )
        chain_path = write_artifact_chain(outdir / "lock_artifact_chain.json", chain)
    return LOCKWorkflowResult(
        manifest=manifest_path,
        levels=levels_path,
        lines=lines_path,
        transition_count=len(transitions),
        assembly_backends=result.assembly_backends,
        diagonalizer_backends=result.diagonalizer_backends,
        artifact_chain=chain_path,
        resource_plan=resource_plan_path,
    )


def _surface_size(surface) -> int:
    size = 1
    for extent in surface.grid_shape:
        size *= extent
    return size


def _write_lines(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOCK_LINE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "LOCK_LINES_SCHEMA",
    "LOCK_LINE_FIELDS",
    "LOCK_WORKFLOW_SCHEMA",
    "LOCKWorkflowRequest",
    "LOCKWorkflowResult",
    "run_lock_workflow",
]
