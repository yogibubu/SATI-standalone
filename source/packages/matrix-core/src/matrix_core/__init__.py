"""Core MATRIX infrastructure with a lazy, stable public API."""

from __future__ import annotations

from importlib import import_module as _import_module


def _exports(module: str, names: str) -> dict[str, tuple[str, str]]:
    return {name: (f".{module}", name) for name in names.split()}


_EXPORTS: dict[str, tuple[str, str]] = {
    **_exports(
        "atomic_io",
        "atomic_copy atomic_json_write atomic_output_path atomic_write_bytes atomic_write_text",
    ),
    **_exports(
        "errors",
        "BackendError InputError OracleError ParseError ScientificValidationError",
    ),
    **_exports(
        "electronic_observables",
        "ELECTRONIC_OBSERVABLE_PROVENANCE_SCHEMA ElectronicObservableCompleteness ElectronicObservableKind ElectronicObservableOwner ElectronicObservableProvenance",
    ),
    **_exports(
        "environment",
        "DEFAULT_CONFIG_PATH ENVIRONMENT_SCHEMA RUNTIME_MACHINE_SCHEMA PROGRAM_DEFINITIONS PROGRAM_ENVIRONMENT ExternalProgram MachineLimits MatrixEnvironment ProgramLauncher RemoteMachine RuntimeMachine configured_remote_machine default_environment detect_external_programs detect_local_limits detect_runtime_machine environment_exports load_environment load_runtime_environment normalize_architecture normalize_operating_system refresh_runtime_environment resolve_program_path runtime_machine_schema_path write_environment",
    ),
    **_exports(
        "availability",
        "AVAILABILITY_SCHEMA BACKEND_ALIASES RESIDENT_BACKENDS AvailabilityInventory ExecutionCombination discover_execution_combinations",
    ),
    **_exports(
        "calculation_launch",
        "CALCULATION_AUTHORIZATION_BUNDLE_ENV CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA DELEGATED_CALCULATION_AUTHORIZATION_BUNDLE_SCHEMA CALCULATION_LAUNCH_AUTHORIZATION_SCHEMA CALCULATION_LAUNCH_PLAN_SCHEMA CalculationLaunchAuthorization CalculationLaunchError CalculationLaunchPlan CalculationResources ConcurrentJobEstimate authorize_calculation_launch authorized_parent_plan_from_environment build_calculation_launch_plan calculation_launch_plan_lines estimate_concurrent_jobs read_calculation_authorization_bundle require_calculation_launch_authorization require_calculation_launch_or_parent_authorization require_authorized_descendant_calculation write_calculation_authorization_bundle write_delegated_calculation_authorization_bundle",
    ),
    **_exports(
        "calculation_protocols",
        "CALCULATION_LIFECYCLE CALCULATION_PROTOCOL_ATLAS_ID CALCULATION_PROTOCOL_ATLAS_SCHEMA CALCULATION_PROTOCOL_ATLAS_VERSION CalculationExecutionDirective CalculationProtocol CalculationProtocolAtlas calculation_execution_directive calculation_protocol load_calculation_protocol_atlas validate_calculation_protocol_atlas",
    ),
    **_exports(
        "manifest",
        "MATRIX_MANIFEST_FRAMEWORK ORACLE_MANIFEST_SCHEMA RunManifest build_run_manifest file_checksums matrix_version sha256_file write_manifest",
    ),
    **_exports(
        "project_transaction",
        "PROJECT_LIFECYCLE_FILENAME PROJECT_LIFECYCLE_SCHEMA ProjectTransaction",
    ),
    **_exports(
        "package_registry",
        "KEYMAKER_MODES PACKAGE_CAPABILITIES PACKAGE_CAPABILITY_REGISTRY_SCHEMA SUPPORTED_ARCHITECTURES SUPPORTED_SYSTEMS PackageCapability capability_providers package_capabilities package_capability package_registry_issues package_registry_json package_registry_lines package_registry_payload package_registry_v2_payload package_registry_schema_path",
    ),
    **_exports(
        "operation_registry",
        "OPERATION_CONTRACTS OPERATION_REGISTRY_SCHEMA OperationContract operation_contract operation_contracts operation_registry_issues operation_registry_json operation_registry_payload operation_registry_schema_path",
    ),
    **_exports(
        "suite_registry",
        "SUITE_REGISTRY_SCHEMA suite_registry_issues suite_registry_json suite_registry_payload suite_registry_schema_path",
    ),
    **_exports(
        "host_capabilities",
        "HOST_CAPABILITY_SNAPSHOT_SCHEMA HOST_QUALIFICATION_SCHEMA ENVIRONMENT_HOST_QUALIFICATION_SCHEMA HostCapabilitySnapshot HostQualification build_host_capability_snapshot host_capability_snapshot_schema_path host_snapshot_json probe_remote_host_capabilities qualify_host_snapshot qualify_environment_hosts unreachable_host_capability_snapshot",
    ),
    **_exports(
        "handoff",
        "ATOMIC_HANDOFF_SCHEMA TOOL_COMPATIBILITY_SCHEMA VERIFIED_TRANSFER_SCHEMA commit_validated_handoff promote_verified_download rsync_download_command validate_tool_compatibility",
    ),
    **_exports(
        "provenance",
        "PROVENANCE_EVENT_TYPES PROVENANCE_FILENAME PROVENANCE_SCHEMA ProvenanceEvent ProvenanceLedger ProvenanceVerification artifact_reference provenance_path provenance_schema_path",
    ),
    **_exports(
        "keymaker_protocol",
        "FROZEN_KEYMAKER_STAGES frozen_keymaker_stage_labels validate_frozen_keymaker_stage_sequence",
    ),
    **_exports(
        "architect_zaff_reference",
        "ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID ARCHITECT_ZAFF_REFERENCE_SHA256 load_architect_zaff_reference_protocol",
    ),
    **_exports(
        "protocol_library",
        "ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID GFN2_ZAFF_FIT_PROTOCOL_IDS TWO_MORSE_GFN2_ZAFF_FIT_PROTOCOL_IDS KEYMAKER_PROTOCOL_CHANGE_POLICY KEYMAKER_PROTOCOL_ENTRYPOINT KEYMAKER_PROTOCOL_MANIFEST_SCHEMA KeymakerProtocolLibrary KeymakerProtocolManifest ProtocolArtifact ProtocolRoute resolve_keymaker_protocol load_keymaker_protocol_library validate_keymaker_protocol_manifest validate_keymaker_protocol_library keymaker_protocol_catalog",
    ),
    **_exports(
        "five_tool_keymaker",
        "FIVE_TOOL_DRY_RUN_SCHEMA FIVE_TOOL_PLAN_SCHEMA FIVE_TOOL_READINESS_SCHEMA FIVE_TOOL_ROUTE FIVE_TOOL_WORKFLOWS build_five_tool_keymaker_plan dry_run_five_tool_keymaker_plan five_tool_plan_schema_path five_tool_readiness_contracts five_tool_readiness_schema_path immutable_five_tool_artifact read_five_tool_keymaker_plan validate_five_tool_keymaker_plan write_five_tool_keymaker_plan",
    ),
    **_exports(
        "canonical_toolchain",
        "TOOLCHAIN_SCHEMA TOOLCHAIN_STAGES build_toolchain_plan record_toolchain_stage toolchain_artifact toolchain_readiness validate_toolchain_plan",
    ),
    **_exports(
        "reproducibility",
        "REPRODUCIBILITY_SCHEMA build_reproducibility_manifest write_reproducibility_manifest",
    ),
    **_exports(
        "scientific_report",
        "SCIENTIFIC_REPORT_SCHEMA build_molecule_scientific_report write_molecule_scientific_report",
    ),
    **_exports(
        "scientific_validation",
        "SCIENTIFIC_VALIDATION_SCHEMA ScientificValidationRecord",
    ),
    **_exports(
        "qm_backend_contract",
        "QM_BACKEND_PHASES QM_BACKENDS validate_qm_backend_fixture validate_qm_backend_matrix",
    ),
    **_exports(
        "remote_job_state",
        "REMOTE_JOB_STATE_SCHEMA REMOTE_JOB_STATUSES read_remote_job_state",
    ),
    **_exports(
        "library_contract",
        "validate_fragment_library_records",
    ),
    **_exports(
        "hessian_invariants",
        "validate_hessian_ped_invariants",
    ),
    **_exports(
        "intent",
        "DETERMINISTIC_COMPILER_ID DETERMINISTIC_LANGUAGE_COMPILER_ID GUARDED_LANGUAGE_COMPILER_ID INTENT_SCHEMA INTENT_STATE_FILENAME INTENT_STATUSES DeterministicIntentCompiler DeterministicLanguageIntentCompiler DeterministicPlanAssembler DeterministicResultNarrator IntentCandidate IntentCompilation IntentCompiler IntentIssue IntentRequest GuardedLanguageIntentCompiler LanguageProposalProvider ResultContext ScientificRequirements intent_compilation_path intent_schema_path read_intent_compilation result_context write_intent_compilation",
    ),
    **_exports(
        "intent_benchmark",
        "IntentBenchmarkCase IntentBenchmarkRecord IntentBenchmarkReport LanguageIntentBenchmarkCase LanguageIntentBenchmarkRecord LanguageIntentBenchmarkReport load_intent_benchmark load_language_intent_benchmark run_intent_benchmark run_language_intent_benchmark",
    ),
    **_exports(
        "online_help",
        "SECTION_COMPLETION_HINTS SectionCompletionHint ToolHelp missing_sections_guidance missing_sections_message online_help_json online_help_lines online_help_markdown online_help_records online_help_text section_completion_hint tool_help tool_help_lines",
    ),
    **_exports(
        "paths",
        "repo_root",
    ),
    **_exports(
        "library_paths",
        "matrix_library_paths",
    ),
    **_exports(
        "host_routing",
        "EXECUTION_ROUTING_SCHEMA WORKLOAD_CLASSES ExecutionRoutingDecision ExecutionRoutingPolicy classify_workload execution_routing_schema_path route_host",
    ),
    **_exports(
        "content_cache",
        "SCIENTIFIC_CACHE_ENVELOPE_SCHEMA SCIENTIFIC_STATE_SCHEMA ScientificStateManifest canonical_sha256 content_key get_json put_json scientific_cache_envelope scientific_state_key unwrap_scientific_cache",
    ),
    **_exports(
        "retry_routing",
        "run_with_host_fallback",
    ),
    **_exports(
        "resource_estimation",
        "estimate_workflow_resources",
    ),
    **_exports(
        "artifact_integrity",
        "artifact_digest verify_artifact",
    ),
    **_exports(
        "resource_calibration",
        "calibrate_walltime",
    ),
    **_exports(
        "path_security",
        "safe_workspace_path",
    ),
    **_exports(
        "backend_registry",
        "available_backend select_backend",
    ),
    **_exports(
        "sectioned_xyz",
        "has_section is_section_header_line read_sectioned_lines remove_section_from_lines replace_section replace_section_in_lines replace_xyz_block replace_xyz_block_in_lines section_content section_header write_sectioned_lines xyz_tail_start",
    ),
    **_exports(
        "sections",
        "ORACLE_XYZ_BASIC_SCHEMA SUPPORTED_BASIC_SCHEMAS BasicSection basic_section_lines key_value_section_lines normalize_key parse_basic_section parse_key_value_section read_basic_section write_basic_section",
    ),
    **_exports(
        "tool_contracts",
        "NANO_MATRIX_SCIENTIFIC_CONTRACT_SCHEMA NANO_MATRIX_SCIENTIFIC_INVARIANTS PLANNED_FRAMEWORK_EXPANSION PLANNED_FRAMEWORK_NAME TOOL_CONTRACTS ToolContract ToolReadiness ScientificInvariant nano_matrix_scientific_contract tool_contract tool_contract_lines tool_contract_markdown_table tool_contract_readiness tool_contract_readinesses tool_contracts tool_contracts_json tool_readiness_json tool_readiness_lines tool_readiness_markdown_table xyzin_section_names",
    ),
    **_exports(
        "workspace",
        "WORKSPACE_DIRS WorkspaceLayout ensure_workspace",
    ),
    **_exports(
        "state_contract",
        "ARTIFACT_STATUSES STATE_CONTRACT_FILENAME STATE_CONTRACT_SCHEMA STATE_OWNERS STATE_STATUSES MatrixStateContract OwnerState StateArtifact StateCheckpoint build_state_contract read_state_contract state_contract_path state_contract_schema_path sync_state_contract write_state_contract",
    ),
    **_exports(
        "unit_coherence",
        "UNIT_COHERENCE_SCHEMA UnitDescriptor convert_value describe_unit validate_unit_handoff",
    ),
    **_exports(
        "workflow",
        "CONFIRMATION_POLICIES EXECUTION_MODES GPU_POLICIES PLAN_STATUSES STEP_STATUSES WORKFLOW_RECIPES WORKFLOW_SCHEMA WORKFLOW_STATE_FILENAME WorkflowBackend WorkflowPlan WorkflowRecommendation WorkflowRecipe WorkflowResources WorkflowStep WorkflowStepReadiness build_workflow_plan cancel_workflow_step complete_workflow_step confirm_workflow_step fail_workflow_step read_workflow_plan rank_workflow_recipes recommend_workflow_recipe refresh_workflow_plan restart_workflow_step start_workflow_step workflow_plan_lines workflow_dry_run workflow_plan_path workflow_recipe workflow_recipes workflow_schema_path workflow_step workflow_step_readiness write_workflow_plan",
    ),
}

_PUBLIC_MODULES = (
    "atomic_io",
    "errors",
    "electronic_observables",
    "environment",
    "availability",
    "calculation_launch",
    "calculation_protocols",
    "manifest",
    "project_transaction",
    "package_registry",
    "operation_registry",
    "suite_registry",
    "host_capabilities",
    "handoff",
    "provenance",
    "keymaker_protocol",
    "protocol_library",
    "five_tool_keymaker",
    "canonical_toolchain",
    "reproducibility",
    "scientific_report",
    "scientific_validation",
    "qm_backend_contract",
    "remote_job_state",
    "library_contract",
    "hessian_invariants",
    "intent",
    "intent_benchmark",
    "online_help",
    "paths",
    "library_paths",
    "host_routing",
    "content_cache",
    "retry_routing",
    "resource_estimation",
    "artifact_integrity",
    "resource_calibration",
    "path_security",
    "backend_registry",
    "sectioned_xyz",
    "sections",
    "tool_contracts",
    "workspace",
    "state_contract",
    "unit_coherence",
    "workflow",
)
for _module in _PUBLIC_MODULES:
    _EXPORTS.setdefault(_module, (f".{_module}", ""))

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = _import_module(module_name, __name__)
    value = module if not attribute else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Preserve the historical function-over-module collision. Loading this small
# registry eagerly ensures that later ``matrix_core.tool_contracts`` imports do
# not replace the public ``tool_contracts()`` callable with the submodule.
tool_contracts = _import_module(".tool_contracts", __name__).tool_contracts
