"""Versioned, GUI-independent workflow contracts for Keymaker.

Keymaker is an orchestrator: it selects and monitors scientific steps but never
implements their algorithms.  This module deliberately contains no imports from
``matrix_gui`` and no subprocess execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

from .atomic_io import atomic_json_write
from .tool_contracts import tool_contract
from .workspace import WorkspaceLayout, ensure_workspace


WORKFLOW_SCHEMA = "matrix.workflow.v1"
WORKFLOW_STATE_FILENAME = "matrix-workflow.json"

STEP_STATUSES = (
    "pending",
    "blocked",
    "ready",
    "awaiting_confirmation",
    "running",
    "completed",
    "failed",
    "cancelled",
)
PLAN_STATUSES = ("pending", "running", "completed", "failed", "cancelled")
CONFIRMATION_POLICIES = ("none", "user", "remote", "costly")
EXECUTION_MODES = ("auto", "internal", "local", "remote", "external")
GPU_POLICIES = ("auto", "never", "prefer", "require")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalized(values: Iterable[str], *, upper: bool = False) -> tuple[str, ...]:
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    if upper:
        cleaned = {value.upper() for value in cleaned}
    return tuple(sorted(cleaned))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9à-ÿ]+", text.casefold()))


@dataclass(frozen=True)
class WorkflowResources:
    processors: int | None = None
    memory_gb: float | None = None
    gpu_policy: str = "auto"
    walltime_minutes: int | None = None

    def __post_init__(self) -> None:
        if self.processors is not None and self.processors < 1:
            raise ValueError("processors must be positive")
        if self.memory_gb is not None and self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.walltime_minutes is not None and self.walltime_minutes < 1:
            raise ValueError("walltime_minutes must be positive")
        if self.gpu_policy not in GPU_POLICIES:
            raise ValueError(f"unknown GPU policy: {self.gpu_policy}")


@dataclass(frozen=True)
class WorkflowBackend:
    role: str = ""
    provider: str = "auto"
    execution: str = "auto"
    method: str = ""
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.execution not in EXECUTION_MODES:
            raise ValueError(f"unknown execution mode: {self.execution}")
        object.__setattr__(self, "keywords", tuple(self.keywords))


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    tool: str
    action: str
    title: str
    description: str = ""
    depends_on: tuple[str, ...] = ()
    required_sections: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    produced_sections: tuple[str, ...] = ()
    produced_artifacts: tuple[str, ...] = ()
    backend: WorkflowBackend = field(default_factory=WorkflowBackend)
    resources: WorkflowResources = field(default_factory=WorkflowResources)
    confirmation_policy: str = "none"
    status: str = "pending"
    confirmed: bool = False
    attempts: int = 0
    run_id: str = ""
    last_error: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.id):
            raise ValueError(f"invalid workflow step id: {self.id!r}")
        tool_contract(self.tool)
        if self.confirmation_policy not in CONFIRMATION_POLICIES:
            raise ValueError(f"unknown confirmation policy: {self.confirmation_policy}")
        if self.status not in STEP_STATUSES:
            raise ValueError(f"unknown step status: {self.status}")
        if self.attempts < 0:
            raise ValueError("attempts cannot be negative")


@dataclass(frozen=True)
class WorkflowRecipe:
    key: str
    title: str
    description: str
    keywords: tuple[str, ...]
    steps: tuple[WorkflowStep, ...]

    def __post_init__(self) -> None:
        _validate_step_graph(self.steps)


@dataclass(frozen=True)
class WorkflowRecommendation:
    recipe_key: str
    title: str
    score: int
    confidence: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowStepReadiness:
    step_id: str
    status: str
    missing_dependencies: tuple[str, ...]
    missing_sections: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    confirmation_required: bool

    @property
    def ready(self) -> bool:
        return not (
            self.missing_dependencies
            or self.missing_sections
            or self.missing_artifacts
            or self.confirmation_required
        )


@dataclass(frozen=True)
class WorkflowPlan:
    project_id: str
    recipe_key: str
    recipe_title: str
    objective: str
    steps: tuple[WorkflowStep, ...]
    observed_sections: tuple[str, ...] = ()
    observed_artifacts: tuple[str, ...] = ()
    status: str = "pending"
    schema: str = WORKFLOW_SCHEMA
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)
    revision: int = 1

    def __post_init__(self) -> None:
        if self.schema != WORKFLOW_SCHEMA:
            raise ValueError(f"unsupported workflow schema: {self.schema}")
        if self.status not in PLAN_STATUSES:
            raise ValueError(f"unknown plan status: {self.status}")
        if self.revision < 1:
            raise ValueError("revision must be positive")
        _validate_step_graph(self.steps)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "WorkflowPlan":
        step_records = data.get("steps", ())
        if not isinstance(step_records, (list, tuple)):
            raise ValueError("workflow steps must be an array")
        steps: list[WorkflowStep] = []
        for raw in step_records:
            if not isinstance(raw, Mapping):
                raise ValueError("workflow step must be an object")
            item = dict(raw)
            backend = item.get("backend", {})
            resources = item.get("resources", {})
            item["backend"] = WorkflowBackend(**dict(backend)) if isinstance(backend, Mapping) else WorkflowBackend()
            item["resources"] = WorkflowResources(**dict(resources)) if isinstance(resources, Mapping) else WorkflowResources()
            for name in (
                "depends_on", "required_sections", "required_artifacts",
                "produced_sections", "produced_artifacts",
            ):
                item[name] = tuple(item.get(name, ()))
            steps.append(WorkflowStep(**item))
        return cls(
            project_id=str(data["project_id"]),
            recipe_key=str(data["recipe_key"]),
            recipe_title=str(data["recipe_title"]),
            objective=str(data.get("objective", "")),
            steps=tuple(steps),
            observed_sections=tuple(data.get("observed_sections", ())),
            observed_artifacts=tuple(data.get("observed_artifacts", ())),
            status=str(data.get("status", "pending")),
            schema=str(data.get("schema", "")),
            created_utc=str(data.get("created_utc", "")),
            updated_utc=str(data.get("updated_utc", "")),
            revision=int(data.get("revision", 1)),
        )


def _step(
    id: str,
    tool: str,
    action: str,
    title: str,
    *,
    description: str = "",
    depends_on: tuple[str, ...] = (),
    required_sections: tuple[str, ...] = (),
    required_artifacts: tuple[str, ...] = (),
    produced_sections: tuple[str, ...] = (),
    produced_artifacts: tuple[str, ...] = (),
    backend: WorkflowBackend | None = None,
    confirmation_policy: str = "none",
) -> WorkflowStep:
    return WorkflowStep(
        id=id,
        tool=tool,
        action=action,
        title=title,
        description=description,
        depends_on=depends_on,
        required_sections=_normalized(required_sections, upper=True),
        required_artifacts=_normalized(required_artifacts),
        produced_sections=_normalized(produced_sections, upper=True),
        produced_artifacts=_normalized(produced_artifacts),
        backend=backend or WorkflowBackend(),
        confirmation_policy=confirmation_policy,
    )


_ORACLE = _step(
    "oracle_perception", "oracle", "analyze", "Perceive the molecular state",
    required_artifacts=("matrix.input.structure.v1",),
    produced_sections=("BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS", "PRIMITIVES"),
    produced_artifacts=("matrix.xyz.primitives.v2",),
    description="Normalize the structure and establish persistent topology, aromaticity, symmetry, synthons and PICs.",
)

_SMITH = _step(
    "smith_sonic", "gicforge", "build", "Build and inspect SONIC coordinates",
    depends_on=("oracle_perception",), required_sections=("PRIMITIVES",),
    required_artifacts=("matrix.xyz.primitives.v2",),
    produced_sections=("GIC", "SYCART"),
    produced_artifacts=("oracle.gic.definition.v1", "matrix.smith.sonic_diagnostics.v2"),
    description="Construct local-symmetry SONICs and human-readable coordinate diagnostics.",
)


def _validate_step_graph(steps: tuple[WorkflowStep, ...]) -> None:
    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("workflow step ids must be unique")
    known = set(ids)
    for step in steps:
        missing = set(step.depends_on) - known
        if missing:
            raise ValueError(f"step {step.id} has unknown dependencies: {sorted(missing)}")
        if step.id in step.depends_on:
            raise ValueError(f"step {step.id} depends on itself")
    visiting: set[str] = set()
    visited: set[str] = set()
    dependencies = {step.id: step.depends_on for step in steps}

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("workflow dependency graph contains a cycle")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in dependencies[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in ids:
        visit(step_id)


WORKFLOW_RECIPES: tuple[WorkflowRecipe, ...] = (
    WorkflowRecipe(
        "msr_semiexperimental_refinement",
        "MSR to reliable MORPHEUS structure",
        "Import a legacy MSR file and deliver a validated semiexperimental structure.",
        ("msr", "file msr", "input msr", "legacy msr"),
        (
            _step(
                "morpheus_import_msr",
                "morpheus",
                "import-msr",
                "Import and validate the MSR contract",
                required_artifacts=("matrix.input.msr.v1",),
                produced_sections=("BASIC", "ISOTOPOLOGUES"),
                produced_artifacts=("matrix.morpheus.observations.v1",),
            ),
            _step(
                "morpheus_refine",
                "morpheus",
                "fit",
                "Refine and validate the semiexperimental structure",
                depends_on=("morpheus_import_msr",),
                required_artifacts=("matrix.morpheus.observations.v1",),
                produced_sections=("MORPHEUS",),
                produced_artifacts=(
                    "matrix.morpheus.refined_structure.v1",
                    "matrix.morpheus.reliability.v1",
                    "matrix.morpheus.coauthor_latex.v1",
                    "matrix.morpheus.coauthor_pdf.v1",
                ),
            ),
        ),
    ),
    WorkflowRecipe(
        "structure_to_sonic", "Structure to SONIC", "Perceive a structure and construct auditable SONIC coordinates.",
        ("sonic", "coordinate", "coordinate interne", "smith", "topology", "topologia"),
        (_ORACLE, _SMITH,
         _step("smith_visualize", "gicforge", "visualize", "Visualize selected SONIC displacements",
               depends_on=("smith_sonic",), required_sections=("GIC",),
               required_artifacts=("matrix.smith.sonic_diagnostics.v2",),
               produced_artifacts=("matrix.smith.sonic_visualization.v1",))),
    ),
    WorkflowRecipe(
        "electronic_population_analysis",
        "APOC electronic analysis",
        "Normalize a QM wavefunction or density into portable CM5, Mayer and electronic-state contracts.",
        (
            "apoc", "cm5", "mayer", "population analysis", "analisi di popolazione",
            "density matrix", "matrice densità", "natural orbital",
        ),
        (
            _step(
                "apoc_analyze",
                "apoc",
                "analyze",
                "Normalize density-derived atomic and pair observables",
                required_artifacts=("matrix.input.qm_electronic_source.v1",),
                produced_sections=("QM_POPULATION",),
                produced_artifacts=("matrix.apoc.analysis.v1", "matrix.qm.population.v1"),
            ),
            _step(
                "oracle_reperceive_population",
                "oracle",
                "analyze",
                "Refresh topology and synthons from the APOC contract",
                depends_on=("apoc_analyze",),
                required_sections=("QM_POPULATION",),
                produced_sections=("TOPOLOGY", "SYNTHONS", "PRIMITIVES"),
                produced_artifacts=("matrix.apoc.perceived_structure.v1",),
            ),
        ),
    ),
    WorkflowRecipe(
        "fragment_assembly",
        "Overlap fragment assembly",
        "Assemble a target molecule from overlapping library fragments and validate its complete graph.",
        (
            "fragment assembly", "overlap assembly", "nano lego", "nano-lego",
            "assemblaggio frammenti", "lcb25", "lcb26",
        ),
        (
            _step(
                "fragments_assemble",
                "fragments",
                "assemble-overlap",
                "Assemble and jointly fit the overlapping fragments",
                required_artifacts=("matrix.input.fragment_assembly.v1",),
                produced_artifacts=(
                    "matrix.fragments.assembled_geometry.v1",
                    "matrix.fragments.assembly_manifest.v1",
                ),
            ),
            _step(
                "oracle_validate_assembly",
                "oracle",
                "analyze",
                "Perceive and validate the assembled molecular graph",
                depends_on=("fragments_assemble",),
                required_artifacts=("matrix.fragments.assembled_geometry.v1",),
                produced_sections=(
                    "BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY",
                    "SYNTHONS", "PRIMITIVES",
                ),
                produced_artifacts=("matrix.fragments.validated_assembly.v1",),
            ),
            _step(
                "fragments_freeze_contract",
                "fragments",
                "build",
                "Freeze fragment membership, frames and electronic states",
                depends_on=("oracle_validate_assembly",),
                required_sections=("TOPOLOGY", "SYNTHONS"),
                produced_sections=("FRAGMENTS",),
                produced_artifacts=("matrix.fragments.contract.v1",),
            ),
        ),
    ),
    WorkflowRecipe(
        "geometry_optimization", "Geometry optimization", "Optimize a molecular structure through LINK with an explicit E/G/H backend.",
        (
            "optimization", "optimize", "ottimizzazione", "ottimizzare", "ottimizza",
            "minimum", "minimo", "link",
        ),
        (_ORACLE, _SMITH,
         _step("link_optimize", "link", "optimize", "Optimize in internal coordinates",
               depends_on=("smith_sonic",), required_sections=("GIC",),
               produced_artifacts=("matrix.link.optimized_structure.v1",),
               backend=WorkflowBackend(role="energy-gradient-hessian", provider="user-select"),
               confirmation_policy="costly"),
         _step("oracle_reanalyze", "oracle", "analyze", "Re-perceive the optimized structure",
               depends_on=("link_optimize",), required_artifacts=("matrix.link.optimized_structure.v1",),
               produced_sections=("BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS", "PRIMITIVES"),
               produced_artifacts=("matrix.workflow.final_structure.v1",))),
    ),
    WorkflowRecipe(
        "ring_reconstruction",
        "Complete ring reconstruction",
        "Reconstruct isolated or coupled ring targets and validate the complete molecule.",
        (
            "ring reconstruction", "ricostruzione anello", "puckering target",
            "substituent", "sostituente", "cremer pople",
        ),
        (
            _ORACLE,
            _step(
                "smith_detach_substituents",
                "gicforge",
                "detach-ring-substituents",
                "Classify attachments and detach single-anchor components",
                depends_on=("oracle_perception",),
                required_sections=("TOPOLOGY",),
                produced_artifacts=("matrix.smith.ring_substituent_plan.v1",),
            ),
            _step(
                "smith_reconstruct_ring",
                "gicforge",
                "reconstruct-ring",
                "Solve the isolated or coupled ring target and reposition every component",
                depends_on=("smith_detach_substituents",),
                required_artifacts=("matrix.smith.ring_substituent_plan.v1",),
                produced_artifacts=("matrix.smith.complete_ring_reconstruction.v1",),
            ),
            _step(
                "oracle_validate_reconstruction",
                "oracle",
                "validate-ring-reconstruction",
                "Re-perceive the complete graph and enforce the acceptance gates",
                depends_on=("smith_reconstruct_ring",),
                required_artifacts=("matrix.smith.complete_ring_reconstruction.v1",),
                produced_artifacts=("matrix.keymaker.ring_reconstruction.v1",),
            ),
        ),
    ),
    WorkflowRecipe(
        "oniom_qmmm_optimization",
        "ONIOM QM/MM optimization",
        "Optimize a layered QM/MM structure with electrostatic embedding when no covalent bond is cut, or Gaussian-style ONIOM link atoms at covalent boundaries; use a user-selected QM backend, autonomous MATRIX-GFN2 for the semiempirical layer, and internal GFN2-FF for MM.",
        (
            "oniom", "oniom qm/mm", "oniom qmmm", "qm/mm", "qmmm", "link atom", "link atoms",
            "layered qm/mm", "qm mm",
        ),
        (
            _ORACLE,
            _SMITH,
            _step(
                "link_oniom_qmmm_optimize",
                "link",
                "optimize-oniom-qmmm",
                "Optimize the ONIOM QM/MM partition",
                description=(
                    "Select QM backend and method, include all MM charges in every QM "
                    "evaluation, and use Gaussian-style hydrogen link atoms only along "
                    "covalently cut bonds."
                ),
                depends_on=("smith_sonic",),
                required_sections=("GIC", "TOPOLOGY"),
                produced_artifacts=("matrix.link.oniom_qmmm.optimized_structure.v1",),
                backend=WorkflowBackend(
                    role="qm-mm-layered-energy-gradient-hessian",
                    provider="user-select-qm-resident-gfn2ff-mm",
                    method="QM backend selectable; semiempirical=MATRIX-GFN2; MM=internal GFN2-FF",
                    keywords=("qm-backend-selectable", "matrix-gfn2", "internal-gfn2-ff", "gaussian-link-atoms"),
                ),
                confirmation_policy="costly",
            ),
            _step(
                "oracle_reanalyze_oniom_qmmm",
                "oracle",
                "analyze",
                "Re-perceive the optimized ONIOM QM/MM structure",
                depends_on=("link_oniom_qmmm_optimize",),
                required_artifacts=("matrix.link.oniom_qmmm.optimized_structure.v1",),
                produced_sections=("BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS", "PRIMITIVES"),
                produced_artifacts=("matrix.workflow.final_structure.v1",),
            ),
        ),
    ),
    WorkflowRecipe(
        "qm1_qm2_optimization",
        "QM1/QM2 layered optimization",
        "Optimize a two-level QM1/QM2 partition with electrostatic embedding, optional Gaussian-style link atoms at covalent cuts, selectable quantum backends, and resident GFN2-FF for the molecular-mechanics environment.",
        (
            "qm1", "qm2", "qm1 qm2", "two-level qm", "multilayer qm", "qm/qm",
        ),
        (
            _ORACLE,
            _SMITH,
            _step(
                "link_qm1_qm2_optimize",
                "link",
                "optimize-qm1-qm2",
                "Optimize the QM1/QM2/MM partition",
                description=(
                    "Select independent QM1 and QM2 backends, include the full MM charge "
                    "field in each QM evaluation, retain MATRIX-GFN2 for a semiempirical layer, and "
                    "use internal GFN2-FF for MM."
                ),
                depends_on=("smith_sonic",),
                required_sections=("GIC", "TOPOLOGY"),
                produced_artifacts=("matrix.link.qm1_qm2.optimized_structure.v1",),
                backend=WorkflowBackend(
                    role="qm1-qm2-mm-layered-energy-gradient-hessian",
                    provider="user-select-qm1-qm2-resident-gfn2ff-mm",
                    method="QM1/QM2 backends selectable; semiempirical=MATRIX-GFN2; MM=internal GFN2-FF",
                    keywords=("qm1-backend-selectable", "qm2-backend-selectable", "matrix-gfn2", "internal-gfn2-ff"),
                ),
                confirmation_policy="costly",
            ),
            _step(
                "oracle_reanalyze_qm1_qm2",
                "oracle",
                "analyze",
                "Re-perceive the optimized QM1/QM2 structure",
                depends_on=("link_qm1_qm2_optimize",),
                required_artifacts=("matrix.link.qm1_qm2.optimized_structure.v1",),
                produced_sections=("BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS", "PRIMITIVES"),
                produced_artifacts=("matrix.workflow.final_structure.v1",),
            ),
        ),
    ),
    WorkflowRecipe(
        "pes_exploration", "PES scan or exploration", "Let SENTINEL select points and LINK realize and evaluate them.",
        ("pes", "scan", "sentinel", "genetic", "genetico", "monte carlo", "exploration", "esplorazione", "conformer"),
        (_ORACLE, _SMITH,
         _step("link_prepare_exploration", "link", "prepare-external", "Prepare the LINK–SENTINEL exchange",
               depends_on=("smith_sonic",), required_sections=("GIC",),
               produced_artifacts=("matrix.link.sentinel.protocol.v1",)),
         _step("sentinel_select", "sentinel", "run", "Select sequential, genetic or Monte Carlo points",
               depends_on=("link_prepare_exploration",),
               required_artifacts=("matrix.link.sentinel.protocol.v1",),
               produced_artifacts=("matrix.sentinel.point_selection.v1",),
               backend=WorkflowBackend(role="next-point", provider="sentinel", execution="external")),
         _step("link_evaluate_points", "link", "external-driver", "Realize and evaluate SENTINEL points",
               depends_on=("sentinel_select",),
               required_artifacts=("matrix.sentinel.point_selection.v1",),
               produced_artifacts=("matrix.link.pes_points.v1",),
               backend=WorkflowBackend(role="energy-gradient-hessian", provider="user-select"),
               confirmation_policy="costly")),
    ),
    WorkflowRecipe(
        "vibrational_analysis", "Vibrational analysis", "Acquire a Cartesian Hessian and solve the GF problem with TRINITY.",
        ("frequency", "frequencies", "frequenza", "frequenze", "vibration", "vibrazionale", "gf", "hessian"),
        (_ORACLE, _SMITH,
         _step("qm_hessian", "qm_adapters", "hessian", "Calculate and import the QM Hessian",
               depends_on=("smith_sonic",), required_sections=("BASIC",),
               produced_sections=("CARTESIAN_HESSIAN",),
               produced_artifacts=("matrix.qm.hessian.v1",),
               backend=WorkflowBackend(role="hessian", provider="user-select"),
               confirmation_policy="costly"),
         _step("trinity_gf", "trinity", "gf", "Compute harmonic modes and frequencies",
               depends_on=("qm_hessian",), required_sections=("BASIC", "CARTESIAN_HESSIAN"),
               produced_sections=("NORMAL_MODES", "VIBRATIONAL"),
               produced_artifacts=("matrix.trinity.gf.v1",))),
    ),
    WorkflowRecipe(
        "zaff_force_field", "ARCHITECT/ZAFF construction", "Build and validate a ZAFF force field from QM observables.",
        ("zaff", "architect", "force field", "campo di forza", "hessian", "hessiano", "parametri"),
        (_ORACLE, _SMITH,
         _step("qm_hessian", "qm_adapters", "hessian", "Calculate and import the QM Hessian",
               depends_on=("smith_sonic",), required_sections=("BASIC",),
               produced_sections=("CARTESIAN_HESSIAN",), produced_artifacts=("matrix.qm.hessian.v1",),
               backend=WorkflowBackend(role="hessian-and-properties", provider="user-select"),
               confirmation_policy="costly"),
         _step("architect_build", "architect", "build", "Construct and validate ZAFF",
               depends_on=("qm_hessian",),
               required_sections=("BASIC", "PRIMITIVES", "SYNTHONS", "CARTESIAN_HESSIAN"),
               produced_artifacts=("matrix.zaff.force_field.v1", "matrix.architect.derivative_validation.v1"))),
    ),
    WorkflowRecipe(
        "zaff_library_resolution",
        "Shared ZAFF library resolution",
        "Resolve a validated ZAFF field for a molecular state or record the explicit GFN-FF fallback.",
        (
            "zaff library", "shared force field", "gfn-ff fallback",
            "libreria zaff", "riuso campo di forza",
        ),
        (
            _ORACLE,
            _step(
                "architect_resolve_library",
                "architect",
                "library-resolve",
                "Resolve the molecular graph and electronic state in the ZAFF library",
                depends_on=("oracle_perception",),
                required_sections=("BASIC", "TOPOLOGY"),
                produced_artifacts=("matrix.zaff.library_resolution.v1",),
            ),
            _step(
                "architect_validate_resolved_field",
                "architect",
                "validate",
                "Validate the selected field or audit the GFN-FF fallback",
                depends_on=("architect_resolve_library",),
                required_artifacts=("matrix.zaff.library_resolution.v1",),
                produced_artifacts=("matrix.zaff.resolved_runtime.v1",),
            ),
        ),
    ),
    WorkflowRecipe(
        "semiexperimental_refinement", "MORPHEUS structural refinement", "Fit R0 while one parent L0 Freq=Anharm field is calculated, then obtain isotope-specific curvilinear SONIC DeltaBvib corrections, retain the Cartesian channel for intensities and validation, and refine against an independent high-level reference geometry.",
        ("morpheus", "isotopologue", "isotopologi", "semiexperimental", "semi-sperimentale", "rotational", "rotazionale"),
        (_step("morpheus_import_experiment", "morpheus", "import-microwave", "Extract and confirm isotopologues, B0 constants, uncertainties and literature provenance",
               produced_artifacts=("matrix.morpheus.microwave_observations.v1",), confirmation_policy="user"),
         _step("morpheus_r0", "morpheus", "fit-r0", "Fit raw ground-state constants immediately and diagnose rank, conditioning and unstable parameters",
               depends_on=("morpheus_import_experiment",), required_artifacts=("matrix.morpheus.microwave_observations.v1",),
               produced_artifacts=("matrix.morpheus.r0_structure.v1", "matrix.morpheus.r0_diagnostics.v1")),
         _step("oracle_l0", "oracle", "perceive", "Perceive the parent L0 geometry without replacing the independent structural reference",
               produced_sections=("BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS", "PRIMITIVES"),
               produced_artifacts=("matrix.morpheus.l0_perception.v1",)),
         _step("smith_l0_sonic", "gicforge", "build", "Construct the nonredundant SONIC basis used for the curvilinear rovibrational field",
               depends_on=("oracle_l0",), required_sections=("BASIC", "PRIMITIVES"),
               produced_sections=("GIC",), produced_artifacts=("matrix.morpheus.l0_sonic.v1",)),
         _step("qm_parent_anharm", "qm_adapters", "gaussian-freq-anharm", "Calculate the parent L0 Hessian, full cubic field and parent VPT2 quartic subset",
               produced_artifacts=("matrix.qm.parent_anharmonic_force_field.v1",),
               backend=WorkflowBackend(role="parent-anharmonic-force-field", provider="gaussian"), confirmation_policy="costly"),
         _step("morpheus_deltavib", "morpheus", "isotopic-deltavib", "Transform the parent field through nonredundant SONIC coordinates for each isotopologue, calculate curvilinear DeltaBvib, and compare with the retained Cartesian channel",
               depends_on=("qm_parent_anharm", "smith_l0_sonic", "morpheus_import_experiment"),
               required_artifacts=("matrix.qm.parent_anharmonic_force_field.v1", "matrix.morpheus.l0_sonic.v1", "matrix.morpheus.microwave_observations.v1"),
               produced_artifacts=("matrix.morpheus.isotopic_deltavib.v2", "matrix.trinity.vibrational-dual-representation.v1")),
         _step("oracle_reference", "oracle", "prepare-reference", "Prepare the independent PL1/PL2/L2 structural reference; refine L1 to PL1 when requested",
               produced_sections=("BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS", "PRIMITIVES"),
               produced_artifacts=("matrix.morpheus.structural_reference.v1",)),
         _step("smith_reference_sonic", "gicforge", "build", "Construct SONIC on the structural reference geometry",
               depends_on=("oracle_reference",), required_sections=("BASIC", "PRIMITIVES"),
               produced_sections=("GIC",), produced_artifacts=("matrix.smith.sonic.v1",)),
         _step("morpheus_refine", "morpheus", "fit", "Refine the final semiexperimental equilibrium structure",
               depends_on=("morpheus_r0", "morpheus_deltavib", "smith_reference_sonic"),
               required_artifacts=("matrix.morpheus.isotopic_deltavib.v2", "matrix.morpheus.structural_reference.v1"),
               produced_artifacts=("matrix.morpheus.refined_structure.v1",),
               confirmation_policy="user")),
    ),
    WorkflowRecipe(
        "structural_improvement", "ORACLE structural improvement", "Apply ORACLE posterior structural corrections and re-perceive the result.",
        ("structural improvement", "miglioramento strutturale", "core valence", "correction", "correzione", "pl1"),
        (_ORACLE,
         _step("oracle_improve", "oracle", "refine", "Apply structural improvement",
               depends_on=("oracle_perception",), required_sections=("PRIMITIVES", "SYNTHONS"),
               produced_sections=("ACCURACY_LADDER_REFINEMENT",),
               produced_artifacts=("oracle.xyz.accuracy_ladder_refinement.v1",))),
    ),
    WorkflowRecipe(
        "hybrid_anharmonic_spectroscopy",
        "Hybrid DVR/GVPT2 spectroscopy",
        "Deliver a validated spectrum by routing a large-amplitude path block to a "
        "variational solver and the transverse SONIC normal modes to reduced GVPT2.",
        (
            "hybrid", "dvr", "gvpt2", "vpt2", "large amplitude", "large-amplitude",
            "grande ampiezza", "spettro anarmonico", "anharmonic spectrum", "path",
        ),
        (
            _ORACLE,
            _SMITH,
            _step(
                "qm_stationary_hessian", "qm_adapters", "stationary-hessian",
                "Optimize the stationary structure and acquire its harmonic Hessian",
                depends_on=("smith_sonic",), required_sections=("BASIC", "GIC"),
                produced_sections=("CARTESIAN_HESSIAN", "NORMAL_MODES"),
                produced_artifacts=("matrix.qm.stationary_hessian.v1",),
                backend=WorkflowBackend(role="energy-gradient-hessian", provider="user-select"),
                confirmation_policy="costly",
            ),
            _step(
                "trinity_large_amplitude_path", "trinity", "large-amplitude-path",
                "Construct and validate the relaxed large-amplitude path in SONIC coordinates",
                depends_on=("qm_stationary_hessian",),
                required_artifacts=("matrix.qm.stationary_hessian.v1",),
                produced_artifacts=(
                    "matrix.trinity.large_amplitude_path.v1",
                    "matrix.trinity.path_adequacy.v1",
                ),
                backend=WorkflowBackend(role="path-energy-gradient", provider="user-select"),
                confirmation_policy="costly",
            ),
            _step(
                "hybrid_mode_partition", "vpt2_vci", "hybrid-partition",
                "Select the complete path-mode subspace and freeze the transverse block",
                depends_on=("trinity_large_amplitude_path",),
                required_artifacts=(
                    "matrix.qm.stationary_hessian.v1",
                    "matrix.trinity.large_amplitude_path.v1",
                ),
                produced_artifacts=("matrix.trinity.hybrid_partition.v1",),
            ),
            _step(
                "qm_transverse_qff", "qm_adapters", "selective-anharmonic",
                "Acquire diagonal and selected transverse anharmonic force constants",
                depends_on=("hybrid_mode_partition",),
                required_artifacts=("matrix.trinity.hybrid_partition.v1",),
                produced_artifacts=("matrix.qm.transverse_anharmonic.v1",),
                backend=WorkflowBackend(role="reduced-anharmonic-force-field", provider="user-select"),
                confirmation_policy="costly",
            ),
            _step(
                "dvr_variational", "dvr", "solve-path",
                "Solve the accepted large-amplitude Hamiltonian variationally",
                depends_on=("trinity_large_amplitude_path",),
                required_artifacts=("matrix.trinity.path_adequacy.v1",),
                produced_artifacts=("matrix.dvr.variational_levels.v1",),
            ),
            _step(
                "hybrid_assemble", "vpt2_vci", "hybrid-assemble",
                "Assemble, validate and report the variational and perturbative bands",
                depends_on=("qm_transverse_qff", "dvr_variational"),
                required_artifacts=(
                    "matrix.qm.transverse_anharmonic.v1",
                    "matrix.dvr.variational_levels.v1",
                    "matrix.trinity.hybrid_partition.v1",
                ),
                produced_sections=("VIBRATIONAL",),
                produced_artifacts=("matrix.trinity.hybrid_spectrum.v1",),
            ),
        ),
    ),
)


def workflow_recipes() -> tuple[WorkflowRecipe, ...]:
    return WORKFLOW_RECIPES


def workflow_recipe(key: str) -> WorkflowRecipe:
    normalized = key.strip().casefold()
    for recipe in WORKFLOW_RECIPES:
        if recipe.key.casefold() == normalized:
            return recipe
    raise KeyError(f"unknown MATRIX workflow recipe: {key}")


def recommend_workflow_recipe(objective: str, desired_result: str = "") -> WorkflowRecommendation:
    return rank_workflow_recipes(objective, desired_result)[0]


def rank_workflow_recipes(
    objective: str,
    desired_result: str = "",
) -> tuple[WorkflowRecommendation, ...]:
    """Return every workflow recommendation in deterministic score order."""

    objective_text = objective.casefold()
    context_text = desired_result.casefold()
    objective_words = _words(objective_text)
    context_words = _words(context_text)
    ranked: list[tuple[int, int, WorkflowRecipe, tuple[str, ...]]] = []
    for index, recipe in enumerate(WORKFLOW_RECIPES):
        matches: list[str] = []
        score = 0
        for keyword in recipe.keywords:
            term = keyword.casefold()
            in_objective = term in objective_text if " " in term else term in objective_words
            in_context = term in context_text if " " in term else term in context_words
            if in_objective or in_context:
                matches.append(keyword)
                base = 3 if " " in term else 1
                score += base * (2 if in_objective else 1)
        ranked.append((score, -index, recipe, tuple(matches)))
    ordered = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)
    recommendations: list[WorkflowRecommendation] = []
    for position, (score, _, recipe, matches) in enumerate(ordered):
        competitor = max(
            (item[0] for item in ordered if item[2].key != recipe.key),
            default=0,
        )
        if position > 0:
            competitor = ordered[0][0]
        confidence = (
            0.0
            if score == 0
            else min(1.0, 0.5 + 0.1 * score + 0.05 * (score - competitor))
        )
        recommendations.append(
            WorkflowRecommendation(
                recipe.key,
                recipe.title,
                score,
                round(max(0.0, confidence), 3),
                matches,
            )
        )
    return tuple(recommendations)


def build_workflow_plan(
    recipe: str | WorkflowRecipe,
    *,
    objective: str,
    project_id: str,
    present_sections: Iterable[str] = (),
    present_artifacts: Iterable[str] = (),
) -> WorkflowPlan:
    selected = workflow_recipe(recipe) if isinstance(recipe, str) else recipe
    plan = WorkflowPlan(
        project_id=project_id,
        recipe_key=selected.key,
        recipe_title=selected.title,
        objective=objective,
        steps=selected.steps,
        observed_sections=_normalized(present_sections, upper=True),
        observed_artifacts=_normalized(present_artifacts),
    )
    return refresh_workflow_plan(plan)


def _plan_status(steps: tuple[WorkflowStep, ...]) -> str:
    if steps and all(step.status == "completed" for step in steps):
        return "completed"
    if any(step.status == "running" for step in steps):
        return "running"
    if any(step.status == "failed" for step in steps):
        return "failed"
    if steps and all(step.status == "cancelled" for step in steps):
        return "cancelled"
    return "pending"


def refresh_workflow_plan(
    plan: WorkflowPlan,
    *,
    present_sections: Iterable[str] | None = None,
    present_artifacts: Iterable[str] | None = None,
) -> WorkflowPlan:
    sections = set(plan.observed_sections if present_sections is None else _normalized(present_sections, upper=True))
    artifacts = set(plan.observed_artifacts if present_artifacts is None else _normalized(present_artifacts))
    for step in plan.steps:
        if step.status == "completed":
            sections.update(step.produced_sections)
            artifacts.update(step.produced_artifacts)
    completed = {step.id for step in plan.steps if step.status == "completed"}
    refreshed: list[WorkflowStep] = []
    terminal_or_active = {"completed", "running", "failed", "cancelled"}
    for step in plan.steps:
        if step.status in terminal_or_active:
            refreshed.append(step)
            continue
        prerequisites = set(step.depends_on).issubset(completed)
        inputs = set(step.required_sections).issubset(sections) and set(step.required_artifacts).issubset(artifacts)
        if not prerequisites or not inputs:
            status = "blocked"
        elif step.confirmation_policy != "none" and not step.confirmed:
            status = "awaiting_confirmation"
        else:
            status = "ready"
        refreshed.append(replace(step, status=status))
    updated_steps = tuple(refreshed)
    return replace(
        plan,
        steps=updated_steps,
        observed_sections=_normalized(sections, upper=True),
        observed_artifacts=_normalized(artifacts),
        status=_plan_status(updated_steps),
        updated_utc=_utc_now(),
        revision=plan.revision + 1,
    )


def _replace_step(plan: WorkflowPlan, step_id: str, new_step: WorkflowStep) -> WorkflowPlan:
    if not any(step.id == step_id for step in plan.steps):
        raise KeyError(f"unknown workflow step: {step_id}")
    steps = tuple(new_step if step.id == step_id else step for step in plan.steps)
    return refresh_workflow_plan(replace(plan, steps=steps))


def workflow_step(plan: WorkflowPlan, step_id: str) -> WorkflowStep:
    for step in plan.steps:
        if step.id == step_id:
            return step
    raise KeyError(f"unknown workflow step: {step_id}")


def workflow_step_readiness(plan: WorkflowPlan, step_id: str) -> WorkflowStepReadiness:
    step = workflow_step(plan, step_id)
    completed = {candidate.id for candidate in plan.steps if candidate.status == "completed"}
    sections = set(plan.observed_sections)
    artifacts = set(plan.observed_artifacts)
    for candidate in plan.steps:
        if candidate.status == "completed":
            sections.update(candidate.produced_sections)
            artifacts.update(candidate.produced_artifacts)
    return WorkflowStepReadiness(
        step_id=step.id,
        status=step.status,
        missing_dependencies=_normalized(set(step.depends_on) - completed),
        missing_sections=_normalized(set(step.required_sections) - sections, upper=True),
        missing_artifacts=_normalized(set(step.required_artifacts) - artifacts),
        confirmation_required=step.confirmation_policy != "none" and not step.confirmed,
    )


def confirm_workflow_step(plan: WorkflowPlan, step_id: str) -> WorkflowPlan:
    step = workflow_step(plan, step_id)
    if step.status not in {"awaiting_confirmation", "blocked", "pending", "ready"}:
        raise ValueError(f"cannot confirm step {step_id} while {step.status}")
    return _replace_step(plan, step_id, replace(step, confirmed=True))


def start_workflow_step(plan: WorkflowPlan, step_id: str, *, run_id: str) -> WorkflowPlan:
    step = workflow_step(plan, step_id)
    if step.status != "ready":
        raise ValueError(f"step {step_id} is not ready (status={step.status})")
    if not run_id.strip():
        raise ValueError("run_id is required")
    return _replace_step(
        plan, step_id,
        replace(step, status="running", attempts=step.attempts + 1, run_id=run_id.strip(), last_error=""),
    )


def complete_workflow_step(
    plan: WorkflowPlan,
    step_id: str,
    *,
    produced_sections: Iterable[str] = (),
    produced_artifacts: Iterable[str] = (),
) -> WorkflowPlan:
    step = workflow_step(plan, step_id)
    if step.status != "running":
        raise ValueError(f"step {step_id} is not running (status={step.status})")
    verified_sections = _normalized((*step.produced_sections, *produced_sections), upper=True)
    verified_artifacts = _normalized((*step.produced_artifacts, *produced_artifacts))
    return _replace_step(
        plan, step_id,
        replace(step, status="completed", produced_sections=verified_sections,
                produced_artifacts=verified_artifacts, last_error=""),
    )


def fail_workflow_step(plan: WorkflowPlan, step_id: str, *, error: str) -> WorkflowPlan:
    step = workflow_step(plan, step_id)
    if step.status != "running":
        raise ValueError(f"step {step_id} is not running (status={step.status})")
    return _replace_step(plan, step_id, replace(step, status="failed", last_error=error.strip() or "unknown error"))


def restart_workflow_step(plan: WorkflowPlan, step_id: str) -> WorkflowPlan:
    step = workflow_step(plan, step_id)
    if step.status not in {"failed", "cancelled"}:
        raise ValueError(f"step {step_id} cannot be restarted from {step.status}")
    return _replace_step(plan, step_id, replace(step, status="pending", run_id="", last_error=""))


def cancel_workflow_step(plan: WorkflowPlan, step_id: str) -> WorkflowPlan:
    step = workflow_step(plan, step_id)
    if step.status in {"completed", "cancelled"}:
        raise ValueError(f"step {step_id} cannot be cancelled from {step.status}")
    return _replace_step(plan, step_id, replace(step, status="cancelled"))


def workflow_schema_path() -> Path:
    return Path(str(files("matrix_core").joinpath("schemas", "workflow-v1.schema.json")))


def workflow_plan_path(workspace: Path | WorkspaceLayout) -> Path:
    layout = workspace if isinstance(workspace, WorkspaceLayout) else ensure_workspace(Path(workspace))
    layout.ensure()
    return layout.state / WORKFLOW_STATE_FILENAME


def write_workflow_plan(plan: WorkflowPlan, workspace: Path | WorkspaceLayout) -> Path:
    path = workflow_plan_path(workspace)
    atomic_json_write(path, plan.to_dict())
    from .state_contract import sync_state_contract

    sync_state_contract(plan, workspace)
    return path


def read_workflow_plan(workspace: Path | WorkspaceLayout) -> WorkflowPlan:
    path = workflow_plan_path(workspace)
    return WorkflowPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def workflow_plan_lines(plan: WorkflowPlan) -> tuple[str, ...]:
    lines = [f"{plan.recipe_title}: {plan.status}", f"Objective: {plan.objective}"]
    for index, step in enumerate(plan.steps, 1):
        contract = tool_contract(step.tool)
        lines.append(f"{index}. [{step.status}] {contract.planned_name or contract.display_name}: {step.title}")
        if step.last_error:
            lines.append(f"   Error: {step.last_error}")
    return tuple(lines)


def workflow_dry_run(plan: WorkflowPlan) -> dict[str, object]:
    """Return a side-effect-free dependency/resource preview for Keymaker."""

    steps = []
    for step in plan.steps:
        steps.append({"id": step.id, "tool": step.tool, "title": step.title, "depends_on": list(step.depends_on), "resources": asdict(step.resources), "execution": step.backend.execution, "required_paths": list(step.required_artifacts), "produced_paths": list(step.produced_artifacts)})
    return {"schema": "matrix.keymaker.workflow_dry_run.v1", "project_id": plan.project_id, "recipe_key": plan.recipe_key, "step_count": len(steps), "steps": steps, "execution_side_effects": False}
