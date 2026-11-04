"""Canonical package and capability registry for the complete MATRIX suite."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping


PACKAGE_CAPABILITY_REGISTRY_SCHEMA = "matrix.package-capability-registry.v1"
PACKAGE_CAPABILITY_REGISTRY_SCHEMA_V2 = "matrix.package-capability-registry.v2"
KEYMAKER_MODES = ("direct", "orchestrator", "provider", "support")
SUPPORTED_SYSTEMS = ("linux", "macos")
SUPPORTED_ARCHITECTURES = ("x86_64", "arm64")


@dataclass(frozen=True)
class PackageCapability:
    """One installable package and the services it contributes to Keymaker."""

    package: str
    import_name: str
    layer: str
    role: str
    keymaker_mode: str
    capabilities: tuple[str, ...]
    commands: tuple[str, ...] = ()
    python_apis: tuple[str, ...] = ()
    tool_contracts: tuple[str, ...] = ()
    consumed_artifacts: tuple[str, ...] = ()
    produced_artifacts: tuple[str, ...] = ()
    provider_programs: tuple[str, ...] = ()
    native_backends: tuple[str, ...] = ()
    supported_systems: tuple[str, ...] = SUPPORTED_SYSTEMS
    supported_architectures: tuple[str, ...] = SUPPORTED_ARCHITECTURES
    portable_fallback: bool = True
    capability_requirements: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.package.startswith("matrix-"):
            raise ValueError("MATRIX package names must start with 'matrix-'")
        if not self.import_name.startswith("matrix_"):
            raise ValueError("MATRIX import names must start with 'matrix_'")
        if self.keymaker_mode not in KEYMAKER_MODES:
            raise ValueError(f"unsupported Keymaker mode: {self.keymaker_mode}")
        if not self.layer or not self.role or not self.capabilities:
            raise ValueError(f"incomplete package capability record: {self.package}")
        for capability in self.capabilities:
            if not capability or capability != capability.upper():
                raise ValueError(f"capability IDs must be uppercase: {capability!r}")
        unknown_requirements = set(self.capability_requirements) - set(self.capabilities)
        if unknown_requirements:
            raise ValueError(
                f"capability requirements reference unknown IDs: {sorted(unknown_requirements)}"
            )
        if self.keymaker_mode in {"direct", "orchestrator"} and not (
            self.commands or self.python_apis
        ):
            raise ValueError(f"{self.package} has no Keymaker executor")

    def to_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "import_name": self.import_name,
            "layer": self.layer,
            "role": self.role,
            "keymaker_mode": self.keymaker_mode,
            "capabilities": list(self.capabilities),
            "commands": list(self.commands),
            "python_apis": list(self.python_apis),
            "tool_contracts": list(self.tool_contracts),
            "consumed_artifacts": list(self.consumed_artifacts),
            "produced_artifacts": list(self.produced_artifacts),
            "provider_programs": list(self.provider_programs),
            "native_backends": list(self.native_backends),
            "supported_systems": list(self.supported_systems),
            "supported_architectures": list(self.supported_architectures),
            "portable_fallback": self.portable_fallback,
            "capability_requirements": {
                key: dict(value) for key, value in self.capability_requirements.items()
            },
        }


def _package(
    package: str,
    import_name: str,
    layer: str,
    role: str,
    keymaker_mode: str,
    capabilities: tuple[str, ...],
    **metadata: object,
) -> PackageCapability:
    return PackageCapability(
        package=package,
        import_name=import_name,
        layer=layer,
        role=role,
        keymaker_mode=keymaker_mode,
        capabilities=capabilities,
        **metadata,
    )


PACKAGE_CAPABILITIES: tuple[PackageCapability, ...] = (
    _package(
        "matrix-core",
        "matrix_core",
        "platform",
        "infrastructure",
        "support",
        ("WORKSPACE_MANAGEMENT", "MANIFEST_PROVENANCE", "WORKFLOW_STATE"),
        python_apis=("matrix_core.build_run_manifest", "matrix_core.build_workflow_plan"),
        produced_artifacts=("matrix.run-manifest.v1", "matrix.workflow.v1"),
    ),
    _package(
        "matrix-numerics",
        "matrix_numerics",
        "platform",
        "numerical-runtime",
        "support",
        ("NUMERICAL_KERNELS", "BACKEND_SELECTION", "HERMITIAN_DIAGONALIZATION"),
        python_apis=(
            "matrix_numerics.resolve_native_backend",
            "matrix_numerics.diagonalize_hermitian",
        ),
        native_backends=("numpy-scipy", "compiled-cpu", "gpu"),
    ),
    _package(
        "matrix-engines",
        "matrix_engines",
        "platform",
        "execution-runtime",
        "support",
        ("POTENTIAL_BACKEND_CONTRACT", "REMOTE_QM_TRANSPORT", "ENGINE_DISCOVERY"),
        python_apis=("matrix_engines.remote_qm", "matrix_engines.PotentialBackend"),
        produced_artifacts=("matrix.qm.remote-job.v1",),
        native_backends=("fortran", "compiled-cpu"),
    ),
    _package(
        "matrix-cli",
        "matrix_cli",
        "platform",
        "command-orchestrator",
        "orchestrator",
        ("CLI_ORCHESTRATION", "COMMAND_DISPATCH", "TYPED_ONIC_CLI"),
        commands=("matrix", "link", "smith"),
        python_apis=("matrix_cli.cli.matrix_main",),
    ),
    _package(
        "matrix-gui",
        "matrix_gui",
        "platform",
        "keymaker-orchestrator",
        "orchestrator",
        (
            "KEYMAKER_ORCHESTRATION",
            "DESKTOP_WORKFLOWS",
            "CAPABILITY_ROUTING",
            "TYPED_ONIC_KEYMAKER_ROUTING",
        ),
        commands=("matrix-gui", "matrix-keymaker", "matrix-keymaker-app"),
        python_apis=("matrix_gui.capability_coverage_report",),
        tool_contracts=("gui",),
        produced_artifacts=("matrix.keymaker.artifact_registry.v1",),
    ),
    _package(
        "matrix-switch",
        "matrix_switch",
        "input",
        "structure-seeding",
        "provider",
        ("INPUT_RESOLUTION", "SMILES_INTERPRETATION", "CARTESIAN_SEEDING"),
        commands=("matrix-switch",),
        python_apis=("matrix_switch.smiles_to_cartesian", "matrix_switch.name_to_cartesian"),
        tool_contracts=("import",),
        produced_artifacts=("matrix.geometry.seed.v1",),
    ),
    _package(
        "matrix-chem",
        "matrix_chem",
        "input",
        "perception-engine",
        "support",
        ("MOLECULAR_GEOMETRY", "CONTINUOUS_PERCEPTION", "TOPOLOGY", "SYMMETRY", "SYNTHONS"),
        python_apis=("matrix_chem.preprocess_to_enriched_xyz",),
        tool_contracts=("oracle",),
        produced_artifacts=("matrix.xyz.primitives.v2",),
        native_backends=("matrix_chem._chem_native",),
    ),
    _package(
        "matrix-oracle",
        "matrix_oracle",
        "input",
        "perception-service",
        "provider",
        (
            "ORACLE_ANALYSIS",
            "PIC_SOURCE",
            "GEOMETRY_CORRECTIONS",
            "TANK_LCB26_PERCEPTION",
            "TANK_LCB26_GEOMETRY",
        ),
        commands=("oracle", "matrix oracle"),
        python_apis=("matrix_oracle.analyze_structure",),
        tool_contracts=("oracle",),
        consumed_artifacts=("matrix.geometry.seed.v1",),
        produced_artifacts=(
            "matrix.xyz.primitives.v2",
            "oracle.xyz.accuracy_ladder_refinement.v1",
            "matrix.tank.perception_proposal.v1",
            "matrix.tank.geometry_proposal.v1",
        ),
    ),
    _package(
        "matrix-apoc",
        "matrix_apoc",
        "input",
        "electronic-analysis",
        "provider",
        (
            "ELECTRONIC_OUTPUT_NORMALIZATION",
            "HIRSHFELD_POPULATION",
            "CM5_CHARGES",
            "MAYER_BOND_ORDERS",
        ),
        commands=("apoc", "matrix apoc"),
        python_apis=("matrix_apoc.analyze_source",),
        tool_contracts=("apoc",),
        consumed_artifacts=("matrix.qm.result.v1",),
        produced_artifacts=("matrix.apoc.analysis.v1", "matrix.qm.population.v1"),
    ),
    _package(
        "matrix-fragments",
        "matrix_fragments",
        "input",
        "fragment-service",
        "direct",
        ("FRAGMENTATION", "FRAGMENT_ASSEMBLY", "FRAGMENT_HANDOFF"),
        commands=("matrix fragments",),
        python_apis=("matrix_fragments.write_fragment_build_section",),
        tool_contracts=("fragments",),
        consumed_artifacts=("matrix.xyz.primitives.v2",),
        produced_artifacts=("matrix.fragments.v1",),
    ),
    _package(
        "matrix-qm",
        "matrix_qm",
        "qm",
        "qm-contract",
        "support",
        ("QM_RESULT_CONTRACT", "QM_RESOURCE_RESOLUTION", "QM_PROVIDER_DISPATCH"),
        python_apis=("matrix_qm.resources.recommend_qm_backend",),
        tool_contracts=("qm_adapters",),
        produced_artifacts=("matrix.qm.result.v1", "matrix.qm.hessian.v1"),
    ),
    _package(
        "matrix-gfn2",
        "matrix_gfn2",
        "qm",
        "resident-semiempirical-runtime",
        "provider",
        (
            "RESIDENT_GFN2_ENERGY_GRADIENT",
            "SPARSE_LINEAR_SCALING_GFN2",
            "GFN2_POINT_CHARGE_EMBEDDING",
            "GFN2_SYMMETRY_ADAPTATION",
        ),
        python_apis=("matrix_gfn2.GFN2Calculator", "matrix_gfn2.LibraryCalculator"),
        tool_contracts=("qm_adapters",),
        produced_artifacts=("matrix.qm.result.v1",),
        native_backends=("compiled-cpu", "gpu"),
    ),
    _package(
        "matrix-composite",
        "matrix_composite",
        "qm",
        "composite-method-orchestration",
        "support",
        ("COMPOSITE_SURFACE_ASSEMBLY", "COMPOSITE_DERIVATIVE_ASSEMBLY"),
        python_apis=("matrix_composite.CompositeOrchestrator",),
        tool_contracts=("composite",),
        consumed_artifacts=("matrix.qm.result.v1",),
        produced_artifacts=("matrix.composite.result.v1",),
    ),
    _package(
        "matrix-gaussian",
        "matrix_gaussian",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_GAUSSIAN", "GAUSSIAN_RESULT_PROMOTION"),
        commands=("matrix gaussian",),
        python_apis=("matrix_gaussian",),
        tool_contracts=("qm_adapters",),
        provider_programs=("gdv", "g16"),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-molpro",
        "matrix_molpro",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_MOLPRO", "MOLPRO_RESULT_PROMOTION"),
        commands=("matrix molpro",),
        python_apis=("matrix_molpro",),
        tool_contracts=("qm_adapters",),
        provider_programs=("molpro",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-mrcc",
        "matrix_mrcc",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_MRCC", "MRCC_RESULT_PROMOTION"),
        commands=("matrix mrcc",),
        python_apis=("matrix_mrcc",),
        tool_contracts=("qm_adapters",),
        provider_programs=("mrcc",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-orca",
        "matrix_orca",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_ORCA", "ORCA_RESULT_PROMOTION"),
        commands=("matrix orca",),
        python_apis=("matrix_orca",),
        tool_contracts=("qm_adapters",),
        provider_programs=("orca",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-cfour",
        "matrix_cfour",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_CFOUR", "CFOUR_RESULT_PROMOTION"),
        commands=("matrix cfour",),
        python_apis=("matrix_cfour",),
        tool_contracts=("qm_adapters",),
        provider_programs=("cfour",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-turbomole",
        "matrix_turbomole",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_TURBOMOLE", "TURBOMOLE_RESULT_PROMOTION"),
        python_apis=("matrix_turbomole",),
        tool_contracts=("qm_adapters",),
        provider_programs=("turbomole",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-xtb",
        "matrix_xtb",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_XTB", "XTB_RESULT_PROMOTION"),
        commands=("matrix xtb",),
        python_apis=("matrix_xtb",),
        tool_contracts=("qm_adapters",),
        provider_programs=("xtb",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-pyscf",
        "matrix_pyscf",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_PYSCF", "PYSCF_RESULT_PROMOTION"),
        commands=("matrix pyscf",),
        python_apis=("matrix_pyscf",),
        tool_contracts=("qm_adapters",),
        provider_programs=("pyscf",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-et",
        "matrix_et",
        "qm",
        "qm-provider",
        "provider",
        ("QM_PROVIDER_ET", "ET_RESULT_PROMOTION"),
        commands=("matrix et",),
        python_apis=("matrix_et",),
        tool_contracts=("qm_adapters",),
        provider_programs=("et",),
        produced_artifacts=("matrix.qm.result.v1",),
    ),
    _package(
        "matrix-smith",
        "matrix_smith",
        "coordinates",
        "coordinate-authoring",
        "provider",
        (
            "SONIC_CONSTRUCTION",
            "SONIC_VISUALIZATION",
            "G16_COORDINATE_EXPORT",
            "TONIC_CONIC_SONIC_TAXONOMY",
            "TYPED_ONIC_BLOCK_CONSTRUCTION",
            "TYPED_ONIC_SELF_CONTAINED_ARTIFACT",
            "TYPED_ONIC_GENERAL_SPARSE_B_PRIME",
        ),
        commands=("smith", "matrix smith"),
        python_apis=(
            "matrix_smith.write_gicforge_build_sections",
            "matrix_smith.write_typed_onic_artifact",
        ),
        tool_contracts=("gicforge",),
        consumed_artifacts=("matrix.xyz.primitives.v2",),
        produced_artifacts=(
            "oracle.gic.definition.v1",
            "matrix.smith.sonic_diagnostics.v2",
            "matrix.smith.typed_onic_artifact.v1",
        ),
    ),
    _package(
        "matrix-architect",
        "matrix_architect",
        "coordinates",
        "field-authoring",
        "direct",
        (
            "ZAFF_FORCE_FIELD_CONSTRUCTION",
            "COUPLING_SELECTION",
            "B_MATRIX_FIRST_DERIVATIVE",
            "HESSIAN_COORDINATE_TRANSFORMATION",
        ),
        commands=("architect", "matrix architect"),
        python_apis=("matrix_architect.build_zaff_force_field",),
        tool_contracts=("architect",),
        consumed_artifacts=("matrix.xyz.primitives.v2", "oracle.gic.definition.v1"),
        produced_artifacts=(
            "matrix.zaff.force_field.v1",
            "matrix.architect.derivative_validation.v1",
        ),
    ),
    _package(
        "matrix-zaff",
        "matrix_zaff",
        "coordinates",
        "resident-field-runtime",
        "provider",
        ("ZAFF_EVALUATION", "ZAFF_ENERGY_GRADIENT_HESSIAN", "ZAFF_PERSISTENT_RUNTIME"),
        python_apis=("matrix_zaff.ZaffBackend", "matrix_zaff.evaluate_zaff_seed_model"),
        tool_contracts=("architect", "link"),
        consumed_artifacts=("matrix.zaff.force_field.v1",),
        produced_artifacts=("matrix.zaff.evaluation.v1",),
        native_backends=("matrix_zaff._zaff_native", "fmm3d"),
    ),
    _package(
        "matrix-epoc",
        "matrix_epoc",
        "coordinates",
        "boundary-field-runtime",
        "provider",
        (
            "ELLIPSOIDAL_BOUNDARY",
            "REACTION_FIELD",
            "FINITE_DOMAIN_CONFINEMENT",
            "ANALYTIC_BOUNDARY_EVALUATION",
        ),
        python_apis=(
            "matrix_epoc.build_boundary_request",
            "matrix_epoc.CPCMReactionField",
            "matrix_epoc.EllipsoidalVdwConfinement",
        ),
        tool_contracts=("link", "sentinel"),
        produced_artifacts=(
            "matrix.epoc.boundary.v1",
            "matrix.epoc.reaction_field.v1",
            "matrix.epoc.confinement.v1",
            "matrix.epoc.evaluation.v1",
        ),
        native_backends=("fmm3d",),
    ),
    _package(
        "matrix-link",
        "matrix_link",
        "coordinates",
        "realization-and-optimization",
        "provider",
        (
            "INTERNAL_TO_CARTESIAN_REALIZATION",
            "GEOMETRY_OPTIMIZATION",
            "TRANSITION_STATE_OPTIMIZATION",
            "POINT_EVALUATION",
            "MEX_SEAM_OPTIMIZATION",
            "TYPED_ONIC_RUNTIME",
            "TYPED_ONIC_OPTIMIZATION",
        ),
        commands=("link", "matrix link"),
        python_apis=(
            "matrix_link.GeometryEvaluationService",
            "matrix_link.optimize_geometry",
            "matrix_link.optimize_mex",
            "matrix_link.mex_surface_from_link_service",
            "matrix_link.TypedOnicRuntime",
        ),
        tool_contracts=("link", "import"),
        consumed_artifacts=(
            "oracle.gic.definition.v1",
            "matrix.smith.typed_onic_artifact.v1",
            "matrix.zaff.force_field.v1",
            "matrix.qm.result.v1",
        ),
        produced_artifacts=(
            "matrix.link.sentinel.request.v1",
            "matrix.link.sentinel.response.v1",
            "matrix.link.mex_result.v1",
        ),
    ),
    _package(
        "matrix-sentinel",
        "matrix_sentinel",
        "exploration",
        "sampling-service",
        "direct",
        (
            "SEQUENTIAL_SCAN_SELECTION",
            "GENETIC_PES_SELECTION",
            "MONTE_CARLO_PES_SELECTION",
            "SAMPLING_CHECKPOINT",
        ),
        commands=("matrix-sentinel", "matrix sentinel"),
        python_apis=("matrix_sentinel.GeneticStrategy", "matrix_sentinel.MonteCarloStrategy"),
        tool_contracts=("sentinel",),
        consumed_artifacts=("matrix.link.sentinel.request.v1",),
        produced_artifacts=(
            "matrix.link.sentinel.response.v1",
            "matrix.sentinel.point_selection.v1",
        ),
    ),
    _package(
        "matrix-mifune",
        "matrix_mifune",
        "exploration",
        "molecular-dynamics",
        "direct",
        (
            "MOLECULAR_DYNAMICS",
            "LANGEVIN_SAMPLING",
            "RIGID_BODY_DYNAMICS",
            "VARIABLE_CELL_NPT",
            "MULTIPLE_TIME_STEP_DYNAMICS",
            "GPU_DYNAMICS_ACCELERATION",
            "NEURAL_ENGINE_DYNAMICS_ACCELERATION",
            "DYNAMICS_CHECKPOINT",
        ),
        commands=("matrix-mifune", "matrix dynamics"),
        python_apis=(
            "matrix_mifune.run_dynamics",
            "matrix_mifune.EllipsoidVolumeShapeMove",
            "matrix_mifune.ReversibleRESPA",
            "matrix_mifune.ExactMinusSurrogate",
        ),
        consumed_artifacts=("matrix.zaff.evaluation.v1",),
        produced_artifacts=("matrix.mifune.trajectory.v1",),
        native_backends=("matrix_mifune._mifune_native",),
        capability_requirements={
            "GPU_DYNAMICS_ACCELERATION": {"gpu_count": 1},
            "NEURAL_ENGINE_DYNAMICS_ACCELERATION": {
                "systems": ["macos"],
                "architectures": ["arm64"],
                "neural_engine_count": 1,
            },
        },
    ),
    _package(
        "matrix-seraph",
        "matrix_seraph",
        "exploration",
        "solvent-environment",
        "direct",
        (
            "SOLVENT_ENVIRONMENT_CONSTRUCTION",
            "SURFACE_ENVIRONMENT_CONSTRUCTION",
            "SOLVATION_HANDOFF",
        ),
        commands=("matrix-seraph",),
        python_apis=("matrix_seraph.SeraphBuilder",),
        produced_artifacts=("matrix.seraph.environment.v1",),
    ),
    _package(
        "matrix-rama",
        "matrix_rama",
        "exploration",
        "reaction-path-ensemble",
        "direct",
        (
            "REACTION_HYPOTHESIS_GENERATION",
            "REACTION_PATH_ENSEMBLE",
            "TS_SEARCH_ORCHESTRATION",
        ),
        commands=("matrix-rama",),
        python_apis=(
            "matrix_rama.ReactionPathCandidate",
            "matrix_rama.ReactionPathEnsemble",
            "matrix_rama.ReactionMinimum",
            "matrix_rama.candidate_minimum_pairs",
            "matrix_rama.union_sonic_contracts",
            "matrix_rama.build_sentinel_request",
            "matrix_rama.run_sentinel_cycle",
            "matrix_rama.publish_link_transition_state",
        ),
        consumed_artifacts=(
            "oracle.gic.definition.v1",
            "matrix.seraph.environment.v1",
            "matrix.sentinel.point_selection.v1",
            "matrix.link.sentinel.response.v1",
        ),
        produced_artifacts=(
            "matrix.rama.reaction_path_ensemble.v1",
            "matrix.rama.cypher_handoff.v1",
            "matrix.link.sentinel.request.v1",
            "matrix.link.transition_state_handoff.v1",
        ),
    ),
    _package(
        "matrix-cypher",
        "matrix_cypher",
        "exploration",
        "ensemble-analysis",
        "direct",
        ("TRAJECTORY_ANALYSIS", "ENSEMBLE_CLUSTERING", "MEDOID_SELECTION"),
        commands=("matrix-cypher",),
        python_apis=("matrix_cypher.cluster_exploration_samples",),
        consumed_artifacts=("matrix.mifune.trajectory.v1", "matrix.sentinel.point_selection.v1"),
        produced_artifacts=("matrix.cypher.ensemble.v1",),
    ),
    _package(
        "matrix-niobe",
        "matrix_niobe",
        "inspection",
        "inspection-exploration",
        "direct",
        (
            "PROJECTION",
            "ARTIFACT_INSPECTION",
            "STATISTICAL_DIAGNOSTICS",
            "REPORT_GENERATION",
            "VISUALIZATION_MAPPING",
        ),
        commands=("niobe", "matrix-niobe"),
        python_apis=(
            "matrix_niobe.pca_report",
            "matrix_niobe.inspect_array",
            "matrix_niobe.build_workflow_plan",
        ),
        consumed_artifacts=(
            "matrix.cypher.ensemble.v1",
            "matrix.run-manifest.v1",
        ),
        produced_artifacts=(
            "matrix.niobe.pca.v1",
            "matrix.niobe.array_inspection.v1",
        ),
    ),
    _package(
        "matrix-gf",
        "matrix_gf",
        "observables",
        "harmonic-analysis",
        "direct",
        ("HARMONIC_GF_PED", "CURVATURE_IMPROVEMENT"),
        commands=("matrix gf",),
        python_apis=("matrix_gf",),
        tool_contracts=("gf", "trinity"),
        consumed_artifacts=("oracle.gic.definition.v1", "matrix.qm.hessian.v1"),
        produced_artifacts=("matrix.gf.ped.v1",),
    ),
    _package(
        "matrix-trinity",
        "matrix_trinity",
        "observables",
        "anharmonic-field",
        "direct",
        ("ANHARMONIC_FIELD_CONSTRUCTION", "ISOTOPOLOGUE_VIBRATION_ROTATION", "DELTA_B_VIB"),
        commands=("matrix trinity",),
        python_apis=("matrix_trinity",),
        tool_contracts=("trinity",),
        consumed_artifacts=(
            "oracle.gic.definition.v1",
            "matrix.qm.result.v1",
            "matrix.zaff.force_field.v1",
        ),
        produced_artifacts=("matrix.trinity.anharmonic-field.v1",),
    ),
    _package(
        "matrix-dvr",
        "matrix_dvr",
        "observables",
        "large-amplitude-solver",
        "direct",
        ("DVR", "LARGE_AMPLITUDE_LEVELS"),
        commands=("matrix dvr",),
        python_apis=("matrix_dvr",),
        tool_contracts=("dvr", "trinity"),
        produced_artifacts=("matrix.dvr.levels.v1",),
    ),
    _package(
        "matrix-vpt2-vci",
        "matrix_vpt2_vci",
        "observables",
        "anharmonic-solver",
        "direct",
        ("VPT2", "VCI", "DAVIDSON_EIGENSOLVER"),
        commands=("matrix vpt2-vci",),
        python_apis=("matrix_vpt2_vci",),
        tool_contracts=("vpt2_vci", "trinity"),
        consumed_artifacts=("matrix.trinity.anharmonic-field.v1",),
        produced_artifacts=("matrix.vpt2-vci.result.v1",),
    ),
    _package(
        "matrix-rovib",
        "matrix_rovib",
        "observables",
        "rovibrational-analysis",
        "direct",
        ("ROTATIONAL_SPECTROSCOPY", "VIBRATIONAL_SPECTROSCOPY", "ROVIBRATIONAL_DOS"),
        commands=("matrix rovib",),
        python_apis=("matrix_rovib",),
        tool_contracts=("rovib",),
        produced_artifacts=("matrix.rovib.result.v1",),
    ),
    _package(
        "matrix-thermo",
        "matrix_thermo",
        "observables",
        "thermochemistry",
        "direct",
        ("THERMOCHEMISTRY", "PARTITION_FUNCTIONS"),
        commands=("matrix thermo",),
        python_apis=("matrix_thermo",),
        tool_contracts=("thermo",),
        produced_artifacts=("matrix.thermo.result.v1",),
    ),
    _package(
        "matrix-kinetics",
        "matrix_kinetics",
        "observables",
        "kinetics",
        "direct",
        ("CANONICAL_TST", "MICROCANONICAL_RRKM", "KINETIC_NETWORK"),
        commands=("matrix kinetics",),
        python_apis=("matrix_kinetics",),
        tool_contracts=("kinetics",),
        consumed_artifacts=("matrix.thermo.result.v1",),
        produced_artifacts=("matrix.kinetics.network.v1",),
    ),
    _package(
        "matrix-morpheus",
        "matrix_morpheus",
        "observables",
        "semiexperimental-refinement",
        "direct",
        ("SEMIEXPERIMENTAL_REFINEMENT", "ISOTOPOLOGUE_FIT", "STRUCTURAL_UNCERTAINTY"),
        commands=("morpheus", "matrix semiexp"),
        python_apis=("matrix_morpheus.fit_semiexperimental_geometry",),
        tool_contracts=("morpheus",),
        consumed_artifacts=("matrix.trinity.anharmonic-field.v1",),
        produced_artifacts=("matrix.morpheus.refinement.v1",),
    ),
)


def package_capabilities() -> tuple[PackageCapability, ...]:
    return PACKAGE_CAPABILITIES


def package_capability(package: str) -> PackageCapability:
    normalized = str(package).strip().casefold().replace("_", "-")
    if not normalized.startswith("matrix-"):
        normalized = f"matrix-{normalized}"
    normalized = {"matrix-lock": "matrix-rovib", "matrix-sati": "matrix-trinity"}.get(
        normalized, normalized
    )
    for record in PACKAGE_CAPABILITIES:
        if record.package == normalized:
            return record
    raise KeyError(f"unknown MATRIX package: {package}")


def capability_providers(capability: str) -> tuple[PackageCapability, ...]:
    normalized = str(capability).strip().upper().replace("-", "_")
    return tuple(record for record in PACKAGE_CAPABILITIES if normalized in record.capabilities)


def package_registry_issues(
    expected_packages: Iterable[str] | None = None,
) -> tuple[str, ...]:
    issues: list[str] = []
    names = [record.package for record in PACKAGE_CAPABILITIES]
    if len(names) != len(set(names)):
        issues.append("package registry contains duplicate package names")
    imports = [record.import_name for record in PACKAGE_CAPABILITIES]
    if len(imports) != len(set(imports)):
        issues.append("package registry contains duplicate import names")
    for record in PACKAGE_CAPABILITIES:
        if len(record.capabilities) != len(set(record.capabilities)):
            issues.append(f"{record.package} contains duplicate capability IDs")
        if not record.supported_systems or not record.supported_architectures:
            issues.append(f"{record.package} has no platform support declaration")
    if expected_packages is not None:
        expected = {str(value).strip() for value in expected_packages}
        registered = set(names)
        for missing in sorted(expected - registered):
            issues.append(f"unregistered package: {missing}")
        for extra in sorted(registered - expected):
            issues.append(f"registry package absent from checkout: {extra}")
    return tuple(issues)


def package_registry_payload(
    *, schema: str = PACKAGE_CAPABILITY_REGISTRY_SCHEMA
) -> dict[str, object]:
    capability_index: dict[str, list[str]] = {}
    for record in PACKAGE_CAPABILITIES:
        for capability in record.capabilities:
            capability_index.setdefault(capability, []).append(record.package)
    return {
        "schema": str(schema),
        "package_count": len(PACKAGE_CAPABILITIES),
        "capability_count": len(capability_index),
        "packages": [record.to_dict() for record in PACKAGE_CAPABILITIES],
        "capability_index": {
            key: sorted(packages) for key, packages in sorted(capability_index.items())
        },
    }


def package_registry_v2_payload() -> dict[str, object]:
    """Return the capability-requirement-aware v2 contract."""

    return package_registry_payload(schema=PACKAGE_CAPABILITY_REGISTRY_SCHEMA_V2)


def package_registry_json(*, indent: int = 2) -> str:
    return json.dumps(package_registry_payload(), indent=indent, sort_keys=True)


def package_registry_lines() -> tuple[str, ...]:
    return tuple(
        "\t".join(
            (
                record.package,
                record.keymaker_mode,
                record.layer,
                ",".join(record.capabilities),
                ",".join(record.commands or record.python_apis),
            )
        )
        for record in PACKAGE_CAPABILITIES
    )


def package_registry_schema_path(version: int = 1) -> Path:
    if int(version) == 2:
        return Path(__file__).resolve().parent / "schemas" / "package-capability-registry-v2.schema.json"
    return (
        Path(__file__).resolve().parent / "schemas" / "package-capability-registry-v1.schema.json"
    )


__all__ = [
    "KEYMAKER_MODES",
    "PACKAGE_CAPABILITIES",
    "PACKAGE_CAPABILITY_REGISTRY_SCHEMA",
    "PACKAGE_CAPABILITY_REGISTRY_SCHEMA_V2",
    "SUPPORTED_ARCHITECTURES",
    "SUPPORTED_SYSTEMS",
    "PackageCapability",
    "capability_providers",
    "package_capabilities",
    "package_capability",
    "package_registry_issues",
    "package_registry_json",
    "package_registry_lines",
    "package_registry_payload",
    "package_registry_v2_payload",
    "package_registry_schema_path",
]
