"""Small, stable public API for application and library consumers."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "convolve_discrete_measure": (".statistical_convolution", "convolve_discrete_measure"),
    "convolve_discrete_levels": (".statistical_convolution", "convolve_discrete_levels"),
    'group_assignment_blends': ('.assignment_search', 'group_assignment_blends'),
    'remap_assignment_covariance': ('.assignment_covariance', 'remap_assignment_covariance'),
    'AssignmentExploration': ('.assignment_exploration', 'AssignmentExploration'),
    'build_assignment_exploration': ('.assignment_exploration', 'build_assignment_exploration'),
    'exploration_catalog': ('.assignment_exploration', 'exploration_catalog'),
    'propose_spectral_assignments': ('.assignment_search', 'propose_spectral_assignments'),
    'assignments_to_observations': ('.assignment_search', 'assignments_to_observations'),
    'transport_assignment_observations': ('.assignment_refinement', 'transport_assignment_observations'),
    'refine_exploratory_assignments': ('.assignment_refinement', 'refine_exploratory_assignments'),
    'ESALCHarmonicState': ('matrix_trinity.vibronic_harmonic', 'ESALCHarmonicState'),
    'HarmonicVibronicModel': ('matrix_trinity.vibronic_harmonic', 'HarmonicVibronicModel'),
    'harmonic_vibronic_from_esalc': ('matrix_trinity.vibronic_harmonic', 'harmonic_vibronic_from_esalc'),
    'harmonic_strength_sum_rule': ('matrix_trinity.vibronic_harmonic', 'harmonic_strength_sum_rule'),
    'harmonic_vibronic_ti': ('matrix_trinity.vibronic_ti', 'harmonic_vibronic_ti'),
    'harmonic_vibronic_correlation': ('matrix_trinity.vibronic_td', 'harmonic_vibronic_correlation'),
    'harmonic_vibronic_td': ('matrix_trinity.vibronic_td', 'harmonic_vibronic_td'),
    'run_esalc_vibronic_workflow': ('matrix_trinity.vibronic_workflow', 'run_esalc_vibronic_workflow'),
    'harmonic_resonance_raman': ('matrix_trinity.vibronic_raman', 'harmonic_resonance_raman'),
    'run_esalc_resonance_raman_workflow': ('matrix_trinity.vibronic_raman', 'run_esalc_resonance_raman_workflow'),
    'harmonic_radiative_emission_rate': ('matrix_trinity.vibronic_rates', 'harmonic_radiative_emission_rate'),
    'harmonic_nonradiative_rate': ('matrix_trinity.vibronic_rates', 'harmonic_nonradiative_rate'),
    'spin_orbit_mixed_electronic_states': ('matrix_trinity.vibronic_rates', 'spin_orbit_mixed_electronic_states'),

    'voigt_response': ('.spectral_profile', 'voigt_response'),
    'SpectralProfileContext': ('.spectral_profile', 'SpectralProfileContext'),
    'fit_native_spectral_profile': ('.spectral_profile', 'fit_native_spectral_profile'),
    'compare_assignment_hypotheses': ('.assignment_hypotheses', 'compare_assignment_hypotheses'),
    'build_assignment_review': ('.assignment_review', 'build_assignment_review'),
    'record_assignment_decision': ('.assignment_review', 'record_assignment_decision'),
    'reviewed_assignment_report': ('.assignment_review', 'reviewed_assignment_report'),
    'run_assignment_workflow': ('.assignment_workflow', 'run_assignment_workflow'),
    'spectroscopic_discrepancy_features': ('.spectral_discrepancy_intervals', 'spectroscopic_discrepancy_features'),
    'spectral_discrepancy_intervals': ('.spectral_discrepancy_intervals', 'spectral_discrepancy_intervals'),
    'lam_span_response': ('.lam_surface_response', 'lam_span_response'),
    'align_lam_span': ('.physical_spin_response', 'align_lam_span'),
    'physical_spin_matrices': ('.physical_spin_response', 'physical_spin_matrices'),
    'PhysicalSpinLAMProducer': ('.physical_spin_response', 'PhysicalSpinLAMProducer'),
    'SpinRequalificationRequired': ('.adaptive_spin_fit', 'SpinRequalificationRequired'),
    'AdaptiveNativeSpinEvaluator': ('.adaptive_spin_fit', 'AdaptiveNativeSpinEvaluator'),
    'spin_fit_information': ('.spin_prediction_uncertainty', 'spin_fit_information'),
    'spin_prediction_uncertainty': ('.spin_prediction_uncertainty', 'spin_prediction_uncertainty'),

    'reduce_native_spin_operator': ('.native_spin_effective', 'reduce_native_spin_operator'),
    'native_spin_dipole_operators': ('.spin_effective_observables', 'native_spin_dipole_operators'),
    'native_spin_effective_dipoles': ('.spin_effective_observables', 'native_spin_effective_dipoles'),
    'native_spin_effective_spectrum': ('.spin_effective_observables', 'native_spin_effective_spectrum'),
    'NativeSpinReductionQualification': ('.spin_reduction_control', 'NativeSpinReductionQualification'),
    'qualify_native_spin_reduction': ('.spin_reduction_control', 'qualify_native_spin_reduction'),
    'EffectiveSpinResponse': ('.spin_effective_response', 'EffectiveSpinResponse'),
    'canonical_spin_response': ('.spin_effective_response', 'canonical_spin_response'),
    'native_spin_effective_response': ('.spin_effective_response', 'native_spin_effective_response'),
    'effective_spin_dipole_response': ('.spin_effective_response', 'effective_spin_dipole_response'),
    'SpinReductionGapError': ('.spin_effective_response', 'SpinReductionGapError'),
    'native_spin_parameter_matrices': ('.spin_reduced_fit', 'native_spin_parameter_matrices'),
    'NativeSpinReducedEvaluator': ('.spin_reduced_fit', 'NativeSpinReducedEvaluator'),
    'fit_native_spin_reduced': ('.spin_reduced_fit', 'fit_native_spin_reduced'),

    'NativeSpinOperatorBlock': ('.native_spin_effective', 'NativeSpinOperatorBlock'),
    'NativeSpinEffectiveHamiltonian': ('.native_spin_effective', 'NativeSpinEffectiveHamiltonian'),
    'build_native_lam_spin_operator': ('.native_spin_effective', 'build_native_lam_spin_operator'),
    'native_spin_effective_hamiltonian': ('.native_spin_effective', 'native_spin_effective_hamiltonian'),
    'native_spin_n_values': ('.native_spin_effective', 'native_spin_n_values'),
    'TensorEffectiveHamiltonian': ('.tensor_downfold', 'TensorEffectiveHamiltonian'),
    'downfold_tensor_operator_model': ('.tensor_downfold', 'downfold_tensor_operator_model'),
    'retain_contracted_manifold': ('.contracted_rovib', 'retain_contracted_manifold'),
    'apply_physical_field_blocks': ('.adaptive_couplings', 'apply_physical_field_blocks'),
    'observable_qualification_policy': ('.qualification_policy', 'observable_qualification_policy'),
    'transport_spectral_subspaces': ('.subspace_transport', 'transport_spectral_subspaces'),
    'select_spectral_parameters': ('.parameter_selection', 'select_spectral_parameters'),
    'downfold_ordered_operators': ('.ordered_downfold', 'downfold_ordered_operators'),
    'lam_ordered_effective_hamiltonian': ('.ordered_downfold', 'lam_ordered_effective_hamiltonian'),
    'OrderedEffectiveHamiltonian': ('.ordered_downfold', 'OrderedEffectiveHamiltonian'),
    'LOCKTask': ('.resource_execution', 'LOCKTask'),
    'LOCKResourceExecutor': ('.resource_execution', 'LOCKResourceExecutor'),
    'map_lock_refits': ('.resource_execution', 'map_lock_refits'),
    'conditional_qc_noise': ('.qc_campaign', 'conditional_qc_noise'),
    'advance_qc_campaign': ('.qc_campaign', 'advance_qc_campaign'),
    'native_surface_response': ('.lam_surface_response', 'native_surface_response'),
    'cartesian_lam_surface_response': ('.physical_producer', 'cartesian_lam_surface_response'),
    'smith_chart_response': ('.physical_producer', 'smith_chart_response'),
    'build_physical_lam_producer': ('.physical_producer', 'build_physical_lam_producer'),
    'PhysicalLAMProducer': ('.physical_producer', 'PhysicalLAMProducer'),
    'trinity_qff_potential': ('.physical_producer', 'trinity_qff_potential'),
    'qualify_lam_accuracy': ('.lam_accuracy', 'qualify_lam_accuracy'),
    'prepare_qc_cycle': ('.qc_cycle', 'prepare_qc_cycle'),
    'collect_qc_cycle': ('.qc_cycle', 'collect_qc_cycle'),
    'refine_qc_pes_cycle': ('.qc_cycle', 'refine_qc_pes_cycle'),
    'apply_qc_prior': ('.qc_cycle', 'apply_qc_prior'),

    'lam_effective_manifold': ('.lam_manifold', 'lam_effective_manifold'),
    'effective_lam_manifold': ('.lam_manifold', 'effective_lam_manifold'),
    'LAMEffectiveManifold': ('.lam_manifold', 'LAMEffectiveManifold'),
    'cartesian_lam_surface': ('.structure_spectrum', 'cartesian_lam_surface'),
    'smith_lam_surface': ('.structure_spectrum', 'smith_lam_surface'),
    'fit_structure_pes_spectra': ('.structure_spectrum', 'fit_structure_pes_spectra'),
    'structured_model_discrepancy': ('.structured_discrepancy', 'structured_model_discrepancy'),
    'spectral_block_splits': ('.structured_discrepancy', 'spectral_block_splits'),
    'cross_validate_lam_blocks': ('.structured_discrepancy', 'cross_validate_lam_blocks'),
    'select_qc_candidates': ('.qc_acquisition', 'select_qc_candidates'),
    'periodic_pes_qc_candidates': ('.qc_acquisition', 'periodic_pes_qc_candidates'),

    'converge_lam_basis': ('.lam_qualification', 'converge_lam_basis'),
    'lam_rotational_observables': ('.lam_rotational_observables', 'lam_rotational_observables'),
    'build_lam_morpheus_handoff': ('.morpheus_handoff', 'build_lam_morpheus_handoff'),
    'prediction_uncertainty_budget': ('.prediction_budget', 'prediction_uncertainty_budget'),
    'plan_lam_fit_resources': ('.resource_planner', 'plan_lam_fit_resources'),

    'fit_isotopologue_tunneling_results': ('.lam_global_fit', 'fit_isotopologue_tunneling_results'),
    'fit_global_lam_spectra': ('.lam_global_fit', 'fit_global_lam_spectra'),
    'GLOBAL_LAM_FIT_SCHEMA': ('.lam_global_fit', 'GLOBAL_LAM_FIT_SCHEMA'),
    'LAMRotationSurface': ('.lam_rotation', 'LAMRotationSurface'),
    'solve_lam_rotation': ('.lam_rotation', 'solve_lam_rotation'),
    'fit_assigned_spectrum': ('.spectral_fit', 'fit_assigned_spectrum'),
    'ContractedRovibModel': ('.contracted_rovib', 'ContractedRovibModel'),
    'solve_contracted_rovib': ('.contracted_rovib', 'solve_contracted_rovib'),
    'solve_spin_hyperfine_rotor': ('.spin_hyperfine', 'solve_spin_hyperfine_rotor'),
    'read_lam_rotation_surface': ('.lam_rotation', 'read_lam_rotation_surface'),
    'write_lam_rotation_surface': ('.lam_rotation', 'write_lam_rotation_surface'),
    'LOCKWorkflowRequest': ('.lock_workflow', 'LOCKWorkflowRequest'),
    'run_lock_workflow': ('.lock_workflow', 'run_lock_workflow'),
    'ScientificArtifact': ('.artifact_chain', 'ScientificArtifact'),
    'append_artifact': ('.artifact_chain', 'append_artifact'),
    'validate_artifact_chain': ('.artifact_chain', 'validate_artifact_chain'),
    'profile_operator_model': ('.profiling', 'profile_operator_model'),
    'benchmark_lam_fit': ('.fit_benchmark', 'benchmark_lam_fit'),
    'fit_identifiable_qm_subspaces': ('.qm_subspace_fit', 'fit_identifiable_qm_subspaces'),
    'NModeCouplingBlock': ('.adaptive_couplings', 'NModeCouplingBlock'),
    'AdaptiveCouplingResult': ('.adaptive_couplings', 'AdaptiveCouplingResult'),
    'build_nmode_lam_surface': ('.adaptive_couplings', 'build_nmode_lam_surface'),
    'select_nmode_couplings': ('.adaptive_couplings', 'select_nmode_couplings'),
    'propagate_nmode_qm_uncertainty': ('.nmode_uncertainty', 'propagate_nmode_qm_uncertainty'),
    'qualify_qm_covariance': ('.qm_covariance_qualification', 'qualify_qm_covariance'),
    'contract_lam_tabulated_vscf_fields': ('.vscf_tensors', 'contract_lam_tabulated_vscf_fields'),
    'LAMModelCandidate': ('.lam_model_hierarchy', 'LAMModelCandidate'),
    'LAMModelHierarchyResult': ('.lam_model_hierarchy', 'LAMModelHierarchyResult'),
    'fit_lam_model_hierarchy': ('.lam_model_hierarchy', 'fit_lam_model_hierarchy'),
    'reduce_lam_kinetic_dependence': ('.kinetic_hierarchy', 'reduce_lam_kinetic_dependence'),
    'compare_lam_kinetic_models': ('.kinetic_hierarchy', 'compare_lam_kinetic_models'),
    'ModelCandidateEvaluation': ('.model_selection', 'ModelCandidateEvaluation'),
    'select_model_hierarchy': ('.model_selection', 'select_model_hierarchy'),
    'linearized_nested_model_selection': ('.model_selection', 'linearized_nested_model_selection'),
    'correlated_linearized_uncertainty': ('.uncertainty', 'correlated_linearized_uncertainty'),
    'sample_correlated_parameters': ('.uncertainty', 'sample_correlated_parameters'),
    'parametric_bootstrap': ('.uncertainty', 'parametric_bootstrap'),
    'profile_likelihood_intervals': ('.uncertainty', 'profile_likelihood_intervals'),
    'propagate_surface_model_uncertainty': ('.uncertainty', 'propagate_surface_model_uncertainty'),
    'qim_surface_parameter_distribution': ('.uncertainty', 'qim_surface_parameter_distribution'),
    'BenchmarkCase': ('.benchmark', 'BenchmarkCase'),
    'run_benchmark_suite': ('.benchmark', 'run_benchmark_suite'),
    'run_checkpointed_blocks': ('.checkpoint', 'run_checkpointed_blocks'),
    'GeneralLAMCoordinate': ('.multi_lam', 'GeneralLAMCoordinate'),
    'general_lam_surface': ('.multi_lam', 'general_lam_surface'),
    'curvilinear_tunneling_manifold_surface': ('.tunneling_coordinates', 'curvilinear_tunneling_manifold_surface'),
    'converge_curvilinear_manifold_splitting': ('.tunneling_coordinates', 'converge_curvilinear_manifold_splitting'),
    'rectilinear_tunneling_surface': ('.tunneling_coordinates', 'rectilinear_tunneling_surface'),
    'rectilinear_tunneling_surface_from_qim_result': ('.tunneling_coordinates', 'rectilinear_tunneling_surface_from_qim_result'),
    'isotopologue_tunneling_surfaces_from_qim_results': ('.tunneling_coordinates', 'isotopologue_tunneling_surfaces_from_qim_results'),
    'solve_isotopologue_tunneling_results': ('.tunneling_coordinates', 'solve_isotopologue_tunneling_results'),
    'AlignedEigenstates': ('.adaptive_solver', 'AlignedEigenstates'),
    'align_eigenstates_by_subspace': ('.adaptive_solver', 'align_eigenstates_by_subspace'),
    'create_staged_project': ('.staged_project', 'create_staged_project'),
    'record_stage_result': ('.staged_project', 'record_stage_result'),
    'promotion_assessment': ('.staged_project', 'promotion_assessment'),
    'promote_staged_project': ('.staged_project', 'promote_staged_project'),
    'read_staged_project': ('.staged_project', 'read_staged_project'),
    'write_staged_project': ('.staged_project', 'write_staged_project'),
    'build_qm_parameter_graph': ('.qm_parameter_graph', 'build_qm_parameter_graph'),
    'write_qm_parameter_graph': ('.qm_parameter_graph', 'write_qm_parameter_graph'),
    'select_registered_operator_hierarchy': ('.model_selection', 'select_registered_operator_hierarchy'),
    'plan_lock_resources': ('.resource_planner', 'plan_lock_resources'),
    'build_lock_studio_dashboard': ('.studio_contract', 'build_lock_studio_dashboard'),
}

__all__ = list(_EXPORTS)

def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, 'matrix_rovib') if module_name.startswith(".") else import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value

__all__.extend(('fit_global_lam_spectra', 'GLOBAL_LAM_FIT_SCHEMA'))

__all__.append('fit_isotopologue_tunneling_results')

__all__.extend(('converge_lam_basis', 'lam_rotational_observables', 'build_lam_morpheus_handoff', 'prediction_uncertainty_budget', 'plan_lam_fit_resources'))

__all__.extend(('lam_effective_manifold', 'effective_lam_manifold', 'LAMEffectiveManifold', 'cartesian_lam_surface', 'smith_lam_surface', 'fit_structure_pes_spectra', 'structured_model_discrepancy', 'spectral_block_splits', 'cross_validate_lam_blocks', 'select_qc_candidates', 'periodic_pes_qc_candidates'))

__all__.extend(('native_surface_response', 'cartesian_lam_surface_response', 'smith_chart_response', 'build_physical_lam_producer', 'PhysicalLAMProducer', 'trinity_qff_potential', 'qualify_lam_accuracy', 'prepare_qc_cycle', 'collect_qc_cycle', 'refine_qc_pes_cycle', 'apply_qc_prior'))

__all__.extend(('transport_spectral_subspaces', 'select_spectral_parameters', 'downfold_ordered_operators', 'lam_ordered_effective_hamiltonian', 'OrderedEffectiveHamiltonian', 'LOCKTask', 'LOCKResourceExecutor', 'map_lock_refits', 'conditional_qc_noise', 'advance_qc_campaign'))

if "convolve_discrete_levels" not in __all__:
    __all__.append("convolve_discrete_levels")
