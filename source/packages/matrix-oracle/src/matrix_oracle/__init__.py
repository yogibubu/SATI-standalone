"""Public ORACLE perception API with lazily loaded optional GUI clients."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._version import __version__
FACADE_EXPORTS = (
    "OracleAnalysis",
    "OracleAnalysisRequest",
    "analyze_structure",
    "analyze_structures",
    "write_oracle_analysis_reports",
)
def _exports(module: str, names: str, *, absolute: bool = False) -> dict[str, str]:
    qualified = module if absolute else f".{module}"
    return {name: qualified for name in names.split()}


_LAZY_EXPORTS = {
    **_exports("matrix_chem.permutation_inversion_scan", "extend_permutation_inversion_from_scan", absolute=True),
    **_exports(
        "matrix_chem.permutation_inversion_grid",
        "ORACLE_PI_GRID_SCHEMA PermutationInversionGridAction complete_permutation_inversion_from_geometry_grid derive_permutation_inversion_grid_action",
        absolute=True,
    ),
    **_exports(
        "matrix_chem.permutation_inversion",
        "ORACLE_PI_SCHEMA ORACLE_PI_SECTION InternalRotorPermutation OraclePermutationInversion PermutationInversionOperation PIExpansionLimit build_permutation_inversion_symmetry read_permutation_inversion_symmetry write_permutation_inversion_symmetry",
        absolute=True,
    ),
    **_exports(
        "primitive_classes",
        "ORACLE_PRIMITIVE_CLASS_SCHEMA PRIMITIVE_DOMAIN_INTER PRIMITIVE_DOMAIN_INTRA SYCART_CLASS_RANK PrimitiveClassAssignment assign_primitive_classes primitive_class",
    ),
    **_exports("selective_out_of_plane", "build_selective_out_of_plane_primitives"),
    **_exports("initial_hessian", "OracleInitialHessian build_initial_hessian build_simple_hessian"),
    **_exports("selective_topology", "SelectiveTopologyTracker"),
    **_exports(
        "topology_domains",
        "ORACLE_TOPOLOGY_DOMAIN_SCHEMA TopologyDomainAssessment assess_topology_domains",
    ),
    **_exports("force_field_topology", "OracleForceFieldTopology freeze_force_field_topology"),
    **_exports(
        "matrix_chem",
        "AccuracyLadderPlan AddedHydrogen BackTransformationResult HYDROGEN_COMPLETION_SCHEMA HydrogenCompletion Primitive PrimitiveCoordinateContract PrimitiveTarget RefinementLayer ValenceLevel apply_accuracy_ladder_plan backtransform_primitive_targets build_accuracy_ladder_plan build_l1_refinement_targets build_primitive_contract build_primitives core_valence_bond_shift complete_valence_hydrogens primitive_b_matrix read_primitive_contract target_values_from_plan validate_primitive_contract",
        absolute=True,
    ),
    **_exports(
        "api",
        "ORACLE_BATCH_SCHEMA ORACLE_REPORT_SCHEMA SUPPORTED_INPUT_FORMATS OracleAnalysis OracleAnalysisRequest analyze_structure analyze_structures oracle_human_report_lines oracle_version write_oracle_analysis_reports",
    ),
    **_exports(
        "config",
        "OracleConfig OraclePaths OracleSymmetryConfig load_oracle_config oracle_config_template write_oracle_config_template",
    ),
    **_exports(
        "scope",
        "DOWNSTREAM_OWNERSHIP ORACLE_EXCLUDED_CAPABILITIES ORACLE_OWNED_CAPABILITIES ORACLE_SCOPE_SCHEMA oracle_scope_contract",
    ),
    **_exports("layers", "ORACLE_LAYERS OracleLayer layer_contract"),
    **_exports(
        "validation",
        "ORACLE_VALIDATION_SCHEMA validate_analysis_report validate_artifact validate_xyzin_output",
    ),
    **_exports("dependencies", "dependency_status"),
    **_exports("cache", "cache_key cache_path read_cached read_cached_report write_cached"),
    **_exports(
        "remote",
        "local_qm_capabilities probe_remote_qm remote_qm_manifest write_capability_manifest",
    ),
    **_exports("batch", "pending_requests run_batch_safe"),
    **_exports("migrations", "migrate_analysis_report"),
    **_exports("minimal", "minimal_capabilities"),
    **_exports("gfn2_symmetry", "freeze_matrix_gfn2_symmetry"),
    **_exports(
        "rings",
        "ORACLE_COUPLED_RING_TOPOLOGY_SCHEMA ORACLE_RING_PERCEPTION_SCHEMA OracleCoupledRingComponent OracleCoupledRingTopology OracleRingCenter OracleRingPerception build_coupled_ring_topology perceive_coupled_ring_topology perceive_rings",
    ),
    **_exports(
        "lcb26",
        "LCB26_HARMONIC_RECORD_SCHEMA LCB26_L1_GEOMETRY_DATASET LCB26ReferenceError load_lcb26_harmonic_record load_lcb26_l1_geometry load_lcb26_reference query_lcb26 query_lcb26_l1_geometry",
    ),
    **_exports(
        "initial_structure",
        "INITIAL_STRUCTURE_SCHEMA InitialStructureError InitialStructurePreparation prepare_initial_structure weighted_l1_internal_closure",
    ),
    **_exports(
        "initial_geometry_quality",
        "INITIAL_GEOMETRY_QUALITY_SCHEMA InitialGeometryQuality assess_initial_geometry_quality",
    ),
    **_exports(
        "preoptimization_topology",
        "ORACLE_PREOPTIMIZATION_TOPOLOGY_SCHEMA PreoptimizationTopologyContract build_preoptimization_topology_contract",
    ),
    **_exports(
        "structure_refinement",
        "REFINE_STRUCTURE_SCHEMA RefinedStructure complete_refined_structure refine_structure",
    ),
    **_exports(
        "amino_acid_libraries",
        "AMINO_ACID_FRAGMENT_LIBRARY_SCHEMA AMINO_ACID_CONFORMERS AMINO_ACIDIC_RESIDUE_CONFORMERS GDV_POPULATION_KEYWORD SCIENTIFIC_POPULATION_LEVEL load_amino_acid_fragment_libraries",
    ),
    **_exports(
        "lcb26_exploitation",
        "LCB26_CATALOG_FAMILIES LCB26ExploitationCatalog LCB26ExploitationError",
    ),
    **_exports(
        "perception_proposal",
        "TANK_ELECTRONIC_STATUS TANK_PERCEPTION_SCHEMA propose_lcb26_perception",
    ),
    **_exports("tank", "TANK_GEOMETRY_POLICY TANK_GEOMETRY_SCHEMA propose_lcb26_geometry"),
    **_exports("lcb26_mapping", "L2AtomMapping L2MappingError compare_assembly_to_l2"),
    **_exports(
        "peptides",
        "PEPTIDE_BUILD_SCHEMA PEPTIDE_LIBRARY_SCHEMA AminoAcidDefinition PeptideBuild PeptideBuildError amino_acid_definitions build_peptide load_amino_acid_library parse_peptide_sequence query_amino_acid",
    ),
    **_exports(
        "hbond_charge_response",
        "ZAFF_HBOND_CHARGE_RESPONSE_SCHEMA WATER_CM5_REFERENCE WATER_HBOND_BOUNDARY_ALPHA WATER_HBOND_CHARGE_TRANSFER_E WATER_TIP3P_FB_REFERENCE HydrogenBondResponseCalibration EllipsoidalBoundaryResponse HydrogenBondChargeContact HydrogenBondChargeResponseResult HydrogenBondStrengthParameters WaterHydrogenBondResponseParameters evaluate_hydrogen_bond_charge_response evaluate_water_hydrogen_bond_charge_response fit_cm5_hydrogen_bond_response fit_hydrogen_bond_response hydrogen_bond_strength qmmm_mm_charge_response_contacts",
    ),
    **_exports(
        "hbond_training",
        "HydrogenBondTrainingComplex HydrogenBondTrainingGeometry HydrogenBondTrainingMolecule HydrogenBondGeometryAudit HydrogenBondResponseLibrary HydrogenBondResponseTemplate WaterCoordinationCluster build_hbond_training_geometry audit_hbond_training_geometry extended_hbond_transfer_complexes extended_hbond_training_molecules fit_training_geometry_cm5 formaldehyde_homodimer_control load_standard_hbond_response_library resident_hbond_response_contacts standard_hbond_training_complexes standard_hbond_training_molecules water_coordination_response_clusters",
    ),
    **_exports(
        "electrostatics",
        "ORACLE_ELECTROSTATICS_SCHEMA ORACLE_FRAGMENT_RECONSTRUCTION_SCHEMA ORACLE_REFERENCE_CHARGE_FLUCTUATION_SCHEMA ORACLE_XB_DESCRIPTOR_SCHEMA DEFAULT_POPULATION_HALO_DEPTH DEFAULT_XB_CM5_TRANSFER_E ChargeProjectionAudit ElectrostaticAtomType ElectrostaticEquivalenceThresholds NeutralChargeGroup OracleElectrostatics PerceivedHydrogenBond PerceivedHalogenBond PopulationAssembly PopulationFragmentPlan PopulationFragmentResult ReferenceChargeFluctuationContract assemble_overlapping_cm5_mayer plan_overlapping_population_fragments prepare_oracle_electrostatics perceive_halogen_bonds evaluate_halogen_bond_cm5_response standard_cm5_mayer_request",
    ),
    **_exports(
        "synthon_fingerprint",
        "ORACLE_SYNTHON_FINGERPRINT_SCHEMA SynthonFingerprint build_synthon_fingerprint synthon_fingerprint_similarity",
    ),
    **_exports(
        "coordinate_atlas",
        "MINIMUM_FRAGMENT_REPRESENTATION_AUTOMATIC MINIMUM_FRAGMENT_REPRESENTATION_PSEUDO_BONDS MINIMUM_FRAGMENT_REPRESENTATION_QUATERNION MINIMUM_FRAGMENT_REPRESENTATIONS ORACLE_COORDINATE_ATLAS_BUILDER ORACLE_COORDINATE_ATLAS_POLICY_ID ORACLE_COORDINATE_ATLAS_POLICY_VERSION build_exploration_coordinate_atlas_contract build_minimum_coordinate_atlas_contract build_transition_state_coordinate_atlas_contract coordinate_atlas_policy_manifest write_exploration_coordinate_atlas_contract write_minimum_coordinate_atlas_contract write_transition_state_coordinate_atlas_contract",
    ),
    **_exports(
        "transition_state_geometry",
        "ORACLE_TS_SINGLE_GEOMETRY_CATALOG ORACLE_TS_SINGLE_GEOMETRY_CATALOG_VERSION TRANSITION_STATE_SINGLE_GEOMETRY_CATALOG TransitionStateCatalogRule TransitionStateGeometryFeatures build_oracle_transition_state_geometry_contract_from_xyzin classify_transition_state_geometry_features write_oracle_transition_state_geometry_contract_from_xyzin",
    ),
    **_exports(
        "atom_classes",
        "SYNTHON_ATOM_CLASS_SCHEMA SynthonAtomClass SynthonAtomClassResult SynthonAtomClassThresholds classify_synthon_atoms",
    ),
    **_exports(
        "atom_typing",
        "ORACLE_GAFF_TRANSLATION_SCHEMA GaffAtomTypeTranslation assign_gaff_atom_types gaff_translation_from_snapshot",
    ),
    **_exports(
        "electronic_population",
        "ORACLE_ELECTRONEGATIVITY_PROVENANCE ORACLE_HBOND_TRAINING_SCHEMA ORACLE_PAULING_PROVENANCE ORACLE_POPULATION_CALCULATION_SCHEMA OracleElectronicPopulationWorkflow OracleHydrogenBondTrainingResult OraclePopulationBackend OraclePopulationCalculation PySCFOraclePopulationBackend create_portable_l0_population_workflow run_standard_hbond_training write_hbond_training_result",
    ),
    **_exports(
        "refinement",
        "ORACLE_REFINEMENT_SCHEMA OracleGeometryRefinement refine_l1_geometry",
    ),
    **_exports(
        "coordination_input",
        "ORACLE_COORDINATION_INPUT_SCHEMA coordination_input_json_schema load_coordination_input materialize_haptic_interaction_requests validate_coordination_input",
    ),
    **_exports(
        "auxiliary_contacts",
        "AUXILIARY_CONTACT_PROVIDER_SCHEMA DATIVE_PROVIDER STRUCTURAL_LIGAND_PROVIDER AuxiliaryContactEvidence AuxiliaryContactProviderSettings StructuralSiteContactRequest perceive_auxiliary_contact_evidence qualified_vdw_radius",
    ),
    **_exports(
        "contact_graph",
        "ClassifiedAuxiliaryContacts ORACLE_CONTACT_GRAPH_SCHEMA complete_and_classify_contact_orbits",
    ),
    **_exports(
        "multicenter_domains",
        "BRIDGE_PLANE_PROVIDER MAXIMUM_TRANSFERABLE_HYDROGEN_COVALENT_EXTENSION_ANGSTROM MULTICENTER_PROVIDER_SCHEMA SHARED_PROTON_PROVIDER STRUCTURAL_LIGAND_BRIDGE_PROVIDER TRANSFERABLE_HYDROGEN_PROVIDER perceive_multicenter_domains",
    ),
    **_exports(
        "local_perception",
        "ORACLE_LOCAL_PERCEPTION_PROVIDER ORACLE_LOCAL_PERCEPTION_PROVIDER_VERSION local_perception_settings_dict perceive_local_perception_domains read_frozen_effective_atomic_numbers",
    ),
    **_exports(
        "orbital_regions",
        "FragmentCutCandidate OrbitalRegion OrbitalRegionPlan QuasiNeutralFragmentProposal assemble_quasi_neutral_fragments perceive_orbital_regions",
    ),
    **_exports(
        "perception_robustness",
        "ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER ORACLE_PERCEPTION_ROBUSTNESS_PROVIDER_VERSION OraclePerceptionDecision OraclePerceptionFailure OraclePerceptionState PerceptionAuditPolicy audit_perception_robustness deterministic_cartesian_perturbations perceive_oracle_state",
    ),
    **_exports(
        "perception_history",
        "DEFAULT_HYSTERESIS_POLICIES HysteresisCalibration HysteresisCalibrationTrajectory HysteresisPolicy ORACLE_PERCEPTION_TRACKER_PROVIDER ORACLE_PERCEPTION_TRACKER_PROVIDER_VERSION PerceptionTracker TemporalDecisionEvidence calibrate_hysteresis_policy contact_temporal_evidence",
    ),
    **_exports(
        "perception_workflow",
        "FrozenPerceptionHandoff ORACLE_PERCEPTION_HANDOFF_SCHEMA ORACLE_PERCEPTION_WORKFLOW_SCHEMA PerceptionBasinPolicy PerceptionWorkflow PerceptionWorkflowEvent PerceptionWorkflowSnapshot",
    ),
    **_exports(
        "optimization_chart",
        "OPTIMIZATION_CHART_MINIMUM OPTIMIZATION_CHART_TRANSITION_STATE ORACLE_OPTIMIZATION_CHART_SCHEMA OptimizationChartAssessment OptimizationChartAssessor OptimizationChartIdentity materialize_optimization_chart_artifact optimization_chart_identity",
    ),
    **_exports(
        "perception_policy",
        "ChemicalPerceptionPolicy ORACLE_CHEMICAL_PERCEPTION_POLICIES ORACLE_CHEMICAL_PERCEPTION_POLICY_SCHEMA chemical_perception_policy_manifest validate_chemical_perception_policies",
    ),
    **_exports(
        "perception_reporting",
        "ORACLE_PERCEPTION_REPORT_SCHEMA attach_perception_robustness oracle_sonic_contract_sha256 perception_robustness_human_lines perception_robustness_report_document read_perception_robustness_report write_perception_decision_csv write_perception_robustness_report",
    ),
    **_exports(
        "robustness_dataset",
        "ORACLE_ROBUSTNESS_DATASET_SCHEMA ORACLE_ROBUSTNESS_DATASET_VERSION ORACLE_ROBUSTNESS_MANUSCRIPT_CLAIM RobustnessDatasetCase default_robustness_corpus generate_robustness_dataset write_robustness_dataset",
    ),
    **_exports(
        "sonic_contract_builder",
        "ORACLE_SONIC_CONTRACT_BUILDER ORACLE_SONIC_CONTRACT_BUILDER_VERSION build_oracle_sonic_contract_from_xyzin write_oracle_sonic_contract_from_xyzin",
    ),
}

_LAZY_ALIASES = {
    "LCB26_EXPLOITATION_CATALOG_SCHEMA": (".lcb26_exploitation", "CATALOG_SCHEMA"),
    "LCB26_EXPLOITATION_REQUEST_SCHEMA": (".lcb26_exploitation", "REQUEST_SCHEMA"),
}

_PUBLIC_LAZY_EXPORTS = {
    **_exports("matrix_chem.permutation_inversion_scan", "extend_permutation_inversion_from_scan", absolute=True),**_LAZY_EXPORTS, **_LAZY_ALIASES}

__all__ = [
    "AccuracyLadderPlan",
    "AddedHydrogen",
    "BackTransformationResult",
    "HYDROGEN_COMPLETION_SCHEMA",
    "HydrogenCompletion",
    "ORACLE_REPORT_SCHEMA",
    "ORACLE_BATCH_SCHEMA",
    "SUPPORTED_INPUT_FORMATS",
    "OracleAnalysis",
    "OracleAnalysisRequest",
    "OracleConfig",
    "OraclePaths",
    "OracleSymmetryConfig",
    "Primitive",
    "PrimitiveCoordinateContract",
    "PrimitiveTarget",
    "RefinementLayer",
    "ValenceLevel",
    "INITIAL_STRUCTURE_SCHEMA",
    "InitialStructureError",
    "InitialStructurePreparation",
    "prepare_initial_structure",
    "weighted_l1_internal_closure",
    "INITIAL_GEOMETRY_QUALITY_SCHEMA",
    "InitialGeometryQuality",
    "assess_initial_geometry_quality",
    "REFINE_STRUCTURE_SCHEMA",
    "RefinedStructure",
    "complete_refined_structure",
    "refine_structure",
    "AMINO_ACID_FRAGMENT_LIBRARY_SCHEMA",
    "AMINO_ACID_CONFORMERS",
    "AMINO_ACIDIC_RESIDUE_CONFORMERS",
    "GDV_POPULATION_KEYWORD",
    "SCIENTIFIC_POPULATION_LEVEL",
    "load_amino_acid_fragment_libraries",
    "TANK_ELECTRONIC_STATUS",
    "TANK_PERCEPTION_SCHEMA",
    "propose_lcb26_perception",
    "TANK_GEOMETRY_POLICY",
    "TANK_GEOMETRY_SCHEMA",
    "propose_lcb26_geometry",
    "PEPTIDE_BUILD_SCHEMA",
    "PEPTIDE_LIBRARY_SCHEMA",
    "AminoAcidDefinition",
    "PeptideBuild",
    "PeptideBuildError",
    "amino_acid_definitions",
    "build_peptide",
    "load_amino_acid_library",
    "parse_peptide_sequence",
    "query_amino_acid",
    "LCB26ReferenceError",
    "LCB26_HARMONIC_RECORD_SCHEMA",
    "LCB26_CATALOG_FAMILIES",
    "load_lcb26_reference",
    "load_lcb26_harmonic_record",
    "LCB26_L1_GEOMETRY_DATASET",
    "load_lcb26_l1_geometry",
    "query_lcb26",
    "query_lcb26_l1_geometry",
    "DOWNSTREAM_OWNERSHIP",
    "ORACLE_EXCLUDED_CAPABILITIES",
    "ORACLE_OWNED_CAPABILITIES",
    "ORACLE_SCOPE_SCHEMA",
    "ORACLE_RING_PERCEPTION_SCHEMA",
    "ORACLE_COUPLED_RING_TOPOLOGY_SCHEMA",
    "ZAFF_HBOND_CHARGE_RESPONSE_SCHEMA",
    "ORACLE_ELECTROSTATICS_SCHEMA",
    "ORACLE_FRAGMENT_RECONSTRUCTION_SCHEMA",
    "ORACLE_REFERENCE_CHARGE_FLUCTUATION_SCHEMA",
    "ORACLE_XB_DESCRIPTOR_SCHEMA",
    "DEFAULT_POPULATION_HALO_DEPTH",
    "DEFAULT_XB_CM5_TRANSFER_E",
    "ORACLE_SYNTHON_FINGERPRINT_SCHEMA",
    "WATER_CM5_REFERENCE",
    "WATER_HBOND_BOUNDARY_ALPHA",
    "WATER_HBOND_CHARGE_TRANSFER_E",
    "WATER_TIP3P_FB_REFERENCE",
    "HydrogenBondResponseCalibration",
    "EllipsoidalBoundaryResponse",
    "HydrogenBondChargeContact",
    "HydrogenBondChargeResponseResult",
    "HydrogenBondStrengthParameters",
    "HydrogenBondTrainingComplex",
    "HydrogenBondTrainingGeometry",
    "HydrogenBondTrainingMolecule",
    "HydrogenBondGeometryAudit",
    "HydrogenBondResponseLibrary",
    "HydrogenBondResponseTemplate",
    "WaterCoordinationCluster",
    "WaterHydrogenBondResponseParameters",
    "ChargeProjectionAudit",
    "ElectrostaticAtomType",
    "ElectrostaticEquivalenceThresholds",
    "NeutralChargeGroup",
    "OracleElectrostatics",
    "PerceivedHydrogenBond",
    "PerceivedHalogenBond",
    "PopulationAssembly",
    "PopulationFragmentPlan",
    "PopulationFragmentResult",
    "ReferenceChargeFluctuationContract",
    "SynthonFingerprint",
    "L2AtomMapping",
    "L2MappingError",
    "compare_assembly_to_l2",
    "OracleRingPerception",
    "OracleRingCenter",
    "OracleCoupledRingComponent",
    "OracleCoupledRingTopology",
    "analyze_structure",
    "analyze_structures",
    "backtransform_primitive_targets",
    "apply_accuracy_ladder_plan",
    "build_accuracy_ladder_plan",
    "build_l1_refinement_targets",
    "build_primitive_contract",
    "build_primitives",
    "core_valence_bond_shift",
    "complete_valence_hydrogens",
    "load_oracle_config",
    "oracle_config_template",
    "oracle_version",
    "oracle_human_report_lines",
    "oracle_scope_contract",
    "primitive_b_matrix",
    "perceive_rings",
    "build_coupled_ring_topology",
    "perceive_coupled_ring_topology",
    "evaluate_hydrogen_bond_charge_response",
    "evaluate_water_hydrogen_bond_charge_response",
    "fit_cm5_hydrogen_bond_response",
    "fit_hydrogen_bond_response",
    "hydrogen_bond_strength",
    "build_hbond_training_geometry",
    "audit_hbond_training_geometry",
    "extended_hbond_transfer_complexes",
    "extended_hbond_training_molecules",
    "fit_training_geometry_cm5",
    "formaldehyde_homodimer_control",
    "load_standard_hbond_response_library",
    "resident_hbond_response_contacts",
    "standard_hbond_training_complexes",
    "standard_hbond_training_molecules",
    "water_coordination_response_clusters",
    "qmmm_mm_charge_response_contacts",
    "prepare_oracle_electrostatics",
    "perceive_halogen_bonds",
    "evaluate_halogen_bond_cm5_response",
    "assemble_overlapping_cm5_mayer",
    "plan_overlapping_population_fragments",
    "standard_cm5_mayer_request",
    "build_synthon_fingerprint",
    "synthon_fingerprint_similarity",
    "read_primitive_contract",
    "target_values_from_plan",
    "validate_primitive_contract",
    "write_oracle_config_template",
    "write_oracle_analysis_reports",
    "LCB26_EXPLOITATION_CATALOG_SCHEMA",
    "LCB26_EXPLOITATION_REQUEST_SCHEMA",
    "LCB26ExploitationCatalog",
    "LCB26ExploitationError",
    "__version__",
]
__all__.extend([
    "ORACLE_LAYERS", "OracleLayer", "layer_contract", "ORACLE_VALIDATION_SCHEMA",
    "validate_analysis_report", "validate_artifact", "validate_xyzin_output", "dependency_status",
    "cache_key", "cache_path", "read_cached", "read_cached_report", "write_cached",
    "local_qm_capabilities", "probe_remote_qm", "remote_qm_manifest", "write_capability_manifest",
    "pending_requests", "run_batch_safe", "migrate_analysis_report", "minimal_capabilities",
])
for _name in FACADE_EXPORTS:
    if _name not in __all__:
        __all__.append(_name)

for _name in sorted(_PUBLIC_LAZY_EXPORTS):
    if _name not in __all__:
        __all__.append(_name)


def __getattr__(name: str) -> Any:
    alias = _LAZY_ALIASES.get(name)
    if alias is not None:
        module_name, attribute = alias
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
