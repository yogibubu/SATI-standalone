"""TRINITY molecular vibrational and anharmonic-field services."""

from importlib import import_module


TRINITY_LONG_NAME = (
    "Torsional and Rovibrational Internal-coordinate Network for Integrated "
    "Theoretical spectroscopY"
)


_LAZY_EXPORT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('nonadiabatic_operator', ('symmetry_forbidden_scalar', 'frozen_born_huang_operator', 'symmetry_project_cartesian_nac')),
    ('vibronic_gaussian_blocks', ('zero_temperature_gaussian_blocks', 'convolve_gaussian_block_lines', 'gaussian_block_line_strengths')),
    ("common_quadratic_correlation", ("common_quadratic_correlation", "common_quadratic_operator_correlation", "retain_common_quadratic_pairs", "screen_common_quadratic_pairs", "common_quadratic_dressed_correlation")),
    ("common_mode_rates", ("common_vci_lines", "common_vci_operator_lines", "common_vci_nonradiative_rate", "common_vci_radiative_emission_rate")),
    ("common_mode_vci", ("CommonModeVCI", "CommonVCIHamiltonian", "CommonVCIStates", "CommonVCIProperty", "solve_common_mode_vci", "common_vci_transition_orders")),
    ("common_quadratic_modes", ("CommonQuadraticFrame", "joint_quadratic_modes", "common_quadratic_model", "common_quadratic_coefficients", "common_chart_states_from_harmonic")),
    ("native_resonant_property", ("NativeResonantProperty",)),
    ("quantum_dynamics", ("QuantumDynamics", "LadderDynamicsOperator", "contact_dynamics_property", "sum_dynamics_operators", "correlation_emission_spectrum", "SecularRadiativeDynamics", "rotate_dynamics_operator", "gvpt2_quantum_dynamics", "gvpt2_emission_packets", "stationary_emission_correlation", "reduced_field_dynamics_properties")),
    ("statistical_partition", ("matched_partition_moments", "matched_occupation_partition")),
    ("mixed_statistical", ("MixedTransverseModel", "mixed_from_gvpt2", "mixed_transverse_partition")),
    ("mode_blocks", ("VibrationalModePartition", "BlockVibrationalModel", "BlockTransitionProperty", "solve_mode_blocks")),
    ("vibronic_inverse_iteration", ("DenseShiftedInverse", "refine_inverse_window")),
    ("vibronic_recycled_subspace", ("embed_recycled_vectors", "load_recycled_guesses", "RecycledSpectralGuesses")),
    ("vibronic_energy_windows", ("SpectralWindowCount", "QualifiedEigenwindow", "PreparedEnergyWindows", "count_spectral_window", "plan_gaussian_energy_windows", "save_prepared_energy_windows", "load_prepared_energy_windows")),
    ("native_tensor_operator", ("NativeTensorQuadratureOperator", "HarmonicExtensionReference", "native_extension_tensor_operator")),
    ("vibronic_metal_products", ("MetalCompensatedProducts",)),
    ("complementary_oniom", ("ComplementaryFrequencies", "complementary_oniom_frequencies")),
    ("nonequilibrium_solvation", ("ElectronicSolvationSample", "VerticalSolvationTransition",
                                  "vertical_solvation_transition", "NonequilibriumCartesianEnergy")),
    ("x_diagnostics", ("AnharmonicityContribution", "AnharmonicityDiagnostics", "sonic_vpt2_x_diagnostics")),
    ("raman_derivatives", ("RamanDerivativeLayout", "validate_cartesian_polarizability_jacobian", "decode_polarizability_derivatives", "polarizability_jacobian_from_gaussian_fchk")),
    ("lock_handoff", ("build_lock_surface_from_esalc_scan",)),
    ("native_pes_variational", ("NativeCartesianModalPotential", "native_modal_displacements_angstrom", "export_native_modal_grid", "run_native_pes_variational_protocol")),
    ("native_pes_hamiltonian", ("build_native_modal_hamiltonian",)),
    ("native_modal_couplings", ("sample_connected_modal_cuts",)),
    ("native_pes_extension", ("build_native_mode_extension", "native_core_quadrature_coordinates", "acquire_harmonic_extension_spectrum")),
    ("vibronic_sparse", ("solve_sparse_banded_variational", "sparse_gaussian_spectral_density_batch", "lanczos_gaussian_spectral_density", "SparseSpectatorObservable", "SparseVibrationalHamiltonian", "build_sparse_quartic_hamiltonian", "sparse_gaussian_spectral_density")),
    ("radiative_cascade", ("RadiativeCascade", "solve_downward_cascade")),
    ("cascade_reference", ("exact_cascade_frontier", "refine_exact_cascade_reference")),
    ("gvpt2_rates", ("GVPT2RateReference", "gvpt2_rate_reference")),
    ("gvpt2_rate_model", ("EffectiveQuantumRateModel", "fit_gvpt2_rate_model")),
    ("gvpt2_rate_selection", ("fit_selected_gvpt2_rate_model",)),
    ("gvpt2_conditional_rates", ("ConditionalQuantumRateModel", "fit_conditional_quantum_rates", "fit_transitionwise_quantum_rates")),
    ("gvpt2_hybrid_rates", ("PolynomialQuantumEnergyModel", "HybridQuantumRateModel", "fit_polynomial_quantum_energies", "fit_hybrid_quantum_rates", "with_exact_energy_window")),
    ("occupation_operators", ("LadderPolynomial", "ladder_polynomial_from_qp", "contact_property_orders", "OccupationTransitions", "OccupationTransitionOperator", "occupation_transition_lines", "occupation_dipole_emission", "occupation_strength_observable")),
    ("hdc_continuous", ("continuous_gap_filter", "complete_continuous_hdc_normal_form", "sam_continuous_hdc_polynomial_energy", "solve_sam_continuous_hdcpt2")),
    ("hdc_transport", ("HDCBranchFrame", "initialize_hdc_branches", "transport_hdc_branches", "transport_hdc_operator")),
    ("hdc_wavefunction_audit", ("audit_hdc_wavefunctions", "refine_hdc_wavefunctions", "assess_hdc_energy_frame")),
    ("hdc_polynomial", ("HDCPolynomialEnergy", "sam_hdc_polynomial_energy", "hdc_isospectral_frame", "solve_sam_hdc_polynomial", "transition_state_hdc_polynomial_energy")),
    ("hdc_statistical", ("sample_transition_state_wang_landau", "convolve_hdc_rotational_levels", "converge_hdc_thermal_domain", "sample_hdc_windowed_wang_landau")),
    ('transition_state', ('TransitionStateField', 'TransitionStateNormalForm', 'solve_transition_state_normal_form', 'transition_state_channel_matrix', 'transition_state_channels', 'transition_state_transmission', 'transition_state_transform_operator')),
    ('transition_state_workflow', ('run_transition_state_workflow',)),
    ('transition_state_hdc', ('TransitionStateHDCResult', 'solve_transition_state_hdc', 'transition_state_hdc_operator')),
    ('hdc_hamiltonian', ('HDCMatrixResult', 'SAMHDCResult', 'hdc_coupling_moduli', 'complete_hdc_normal_form', 'complete_hdc_hamiltonian', 'solve_sam_hdcpt2', 'sam_polynomial_operator', 'hdc_transition_spectrum', 'sam_thermal_populations', 'audit_sam_hdc_convergence', 'sam_quadratic_energy_model')),
    ('hdc_properties', ('HDCPropertyExpansion', 'hdc_property_matrix', 'hdc_property_spectrum', 'hdc_property_bilinear', 'hdc_harmonic_overlap')),
    ('vibronic_hdcpt2', ('hdc_vibronic_ti',)),
    ('vibronic_variational', ('FiniteVibrationalStates', 'audit_quartic_confinement', 'project_anharmonic_potential', 'solve_quartic_variational', 'embed_variational_states', 'harmonic_overlap_matrix', 'gaussian_number_moment', 'variational_vibronic_transitions')),
    ('vibronic_local_states', ('solve_local_quartic_ground_state', 'solve_local_potential_ground_state', 'truncate_local_ground_state_projection')),
    ('vibronic_spectator_rates', ('VariationalSpectatorObservable', 'MarginalSpectatorObservable', 'variational_spectator_nonradiative_rate')),
    ('vibronic_hdc_workflow', ('run_esalc_hdc_workflow',)),
    ('vibronic_vpt2', ('SAMVPT2Field', 'SAMVPT2Wavefunction', 'NonresonantVPT2Error', 'sam_vpt2_wavefunction', 'vpt2_vibronic_ti')),
    ('vibronic_vpt2_workflow', ('run_esalc_vpt2_vibronic_workflow',)),
    ('vibronic_ti_adaptive', ('harmonic_vibronic_ti_adaptive',)),
    ('vibronic_gaussian', ('read_gaussian_vibronic_data', 'align_gaussian_vibronic_data',
        'gaussian_force_field_in_esalc', 'gaussian_esalc_vibronic_model')),
    ('vibronic_assignment_report', ('targeted_assignment_report','save_targeted_assignment_report')),
    ('vibronic_targeted', ('SpectralSelectionTarget',)),
    ('vibronic_observables', ('harmonic_operator_spectrum',)),
    ('vibronic_optical', ('harmonic_circular_spectrum', 'harmonic_two_photon_spectrum', 'harmonic_mcd_spectrum', 'harmonic_mcd_manifold_spectrum')),
    ('vibronic_optical_workflow', ('run_esalc_optical_workflow',)),
    ('vibronic_workflow', ('run_esalc_vibronic_workflow',)),
    ('vibronic_raman', ('harmonic_resonance_raman','run_esalc_resonance_raman_workflow')),
    ('vibronic_rates', ('harmonic_radiative_emission_rate','harmonic_nonradiative_rate','sam_vpt2_nonradiative_rate','spin_orbit_mixed_electronic_states')),
    ('radiative_td', ('radiative_rate_from_td_density','harmonic_radiative_emission_rate_td')),
    ('vibronic_wang_landau', ('diagonal_anharmonicity_from_assigned_levels','sam_vpt2_transition_strength_callback','wang_landau_nonradiative_rate','importance_nonradiative_rate')),
    ('vibronic_harmonic', ('ESALCHarmonicState','HarmonicVibronicModel','harmonic_vibronic_from_esalc','harmonic_strength_sum_rule')),
    ('vibronic_ti', ('harmonic_vibronic_ti','harmonic_vibronic_occupations')),
    ('vibronic_preparation', ('NativePreparationCache',)),
    ('vibronic_storage', ('save_vibronic_columns',)),
    ('vibronic_td', ('harmonic_vibronic_correlation','harmonic_vibronic_td')),
    ('contracted_coriolis', ('contract_modal_coriolis', 'dress_lam_harmonic_bath_rotation')),
    (
        "algebraic_local_ir",
        (
            "LocalAlgebraicIRResult",
            "solve_local_algebraic_ir",
        ),
    ),
    (
        "local_polyad_ir",
        (
            "LocalPolyadIRResult",
            "solve_local_polyad_ir",
        ),
    ),
    (
        "keymaker_esalci_gvpt2",
        (
            "ESALCIGVPT2ProtocolResult",
            "run_esalci_gvpt2_protocol",
            "run_esalci_gvpt2_smith_metric_protocol",
            "run_esalci_gvpt2_smith_metric_eckart_protocol",
            "run_esalci_gvpt2_gdv_frame_protocol",
        ),
    ),
    (
        "algebraic_local_x",
        (
            "LocalAlgebraicXResult",
            "solve_local_algebraic_x",
        ),
    ),
    (
        "anharmonic_fields",
        (
            "build_normal_mode_anharmonic_field",
            "build_normal_mode_anharmonic_field_from_architect",
            "build_normal_mode_qff_from_calculator",
        ),
    ),
    (
        "anharmonic_sampling",
        (
            "SemiDiagonalCubicField",
            "semidiagonal_cubic_from_modal_gradient_stencil",
        ),
    ),
    (
        "curvilinear_vpt2",
        (
            "SonicVPT2Input",
            "SonicVPT2Result",
            "SonicGVPT2Result",
            "SonicAcquiredField",
            "acquire_sonic_vpt2_from_hessians",
            "acquire_sonic_gvpt2_from_hessians",
            "acquire_sonic_vpt2_from_cartesian_evaluator",
            "acquire_sonic_gvpt2_from_cartesian_evaluator",
            "sonic_rovibrational_metric",
            "wilson_metric_first_from_b_prime",
            "solve_sonic_vpt2",
            "sonic_vpt2_x_matrix",
            "solve_sonic_gvpt2",
            "solve_sonic_gvpt2_adaptive",
        ),
    ),
    (
        "coordinate_gauge",
        (
            "QuadraticChartConnection",
            "CommonGaugeVP2TD",
            "TransportedResonancePolyad",
            "quadratic_chart_connection",
            "transport_sonic_field_second_order",
            "transport_sonic_property_second_order",
            "transport_resonance_polyad",
            "transport_gvpt2_polyads",
            "solve_common_gauge_vp2td",
            "assess_chart_consistency",
            "assess_large_amplitude_dvr",
        ),
    ),
    (
        "local_quadratic",
        (
            "PolyadEdge",
            "CouplingSeries",
            "TransitionSeries",
            "VariationalTransition",
            "OperativeSpectrum",
            "LocalQuadraticModel",
            "build_local_quadratic_model",
            "transition_series",
        ),
    ),
    (
        "sonic_restart",
        (
            "SONIC_GVPT2_RESTART_SCHEMA",
            "write_sonic_gvpt2_restart",
            "read_sonic_gvpt2_restart",
        ),
    ),
    (
        "sonic_rovibrational",
        (
            "SonicRovibrationalConstants",
            "acquire_sonic_rovibrational_constants",
            "rovibrational_constants_from_modal_derivatives",
        ),
    ),
    (
        "gvpt2_spectroscopy",
        (
            "SonicPropertyDerivatives",
            "SonicDiagonalPropertyExpectation",
            "SonicMomentumPropertyDerivatives",
            "SonicROAPropertySurfaces",
            "SonicGVPT2Spectrum",
            "modal_property_derivatives_from_packed_mass_weighted",
            "project_sonic_property_derivatives",
            "property_derivatives_from_cartesian",
            "acquire_modal_property_derivatives_from_gradients",
            "acquire_modal_momentum_derivatives_from_values",
            "sonic_gvpt2_spectrum",
            "sonic_gvpt2_spectra",
            "sonic_gvpt2_spectrum_parallel",
            "resolve_sonic_property_workers",
            "SONIC_PROPERTY_WORKER_POLICY",
            "sonic_diagonal_property_expectation",
            "mix_sonic_polyad_property_moments",
        ),
    ),
    (
        "torsional_dvr",
        (
            "compare_periodic_property_perturbation_dvr",
        ),
    ),
    (
        "chiroptical",
        (
            "GaussianHarmonicChiroptical",
            "ROABackscattering",
            "gaussian_anharmonic_roa_surfaces_from_fchk",
            "gaussian_harmonic_chiroptical_from_fchk",
            "gaussian_roa_response_tensors",
            "roa_scp_backscattering",
            "vcd_rotatory_strength",
        ),
    ),
    (
        "sonic_thermochemistry",
        (
            "SonicHDCPT2Result",
            "SonicGVPT2ThermochemistryResult",
            "solve_sonic_hdcpt2",
            "solve_sonic_gvpt2_thermochemistry",
        ),
    ),
    (
        "quantum_wang_landau",
        (
            "QuantumWangLandauResult",
            "sample_hdcpt2_quantum_wang_landau",
        ),
    ),
    (
        "computed_cubic_job",
        (
            "COMPUTED_CUBIC_JOB_SCHEMA",
            "acquire_computed_cubic_job",
            "acquire_computed_cubic_job_from_hessians",
            "computed_cubic_job_from_dict",
            "computed_cubic_job_to_dict",
            "read_computed_cubic_job",
            "write_computed_cubic_job",
        ),
    ),
    (
        "curvilinear_rovib",
        (
            "CARTESIAN_GRADIENT_DELTABVIB_METHOD",
            "CARTESIAN_HESSIAN_DELTABVIB_METHOD",
            "CURVILINEAR_DELTABVIB_SCHEMA",
            "CallableDeltaBVibJob",
            "CurvilinearAlphaComponents",
            "CurvilinearDeltaBVibResult",
            "CurvilinearIsotopologueDefinition",
            "CurvilinearIsotopologueState",
            "SONIC_GRADIENT_DELTABVIB_METHOD",
            "SONIC_HESSIAN_DELTABVIB_METHOD",
            "SONIC_INTERNAL_DELTABVIB_METHOD",
            "TrinityDeltaBVibMethod",
            "TrinityDeltaBVibService",
            "curvilinear_alpha_from_internal_field",
            "curvilinear_qcent_from_sonic_state",
            "curvilinear_deltabvib_for_isotopologues",
            "curvilinear_deltabvib_from_alpha",
            "curvilinear_deltabvib_from_semidiagonal_cubic",
            "curvilinear_deltabvib_from_definitions",
            "curvilinear_deltabvib_from_dict",
            "curvilinear_deltabvib_from_internal_field",
            "curvilinear_deltabvib_to_dict",
            "curvilinear_isotopologue_state_from_internal_field",
            "read_curvilinear_deltabvib_results",
            "write_curvilinear_deltabvib_results",
            "write_curvilinear_deltabvib_to_xyzin",
        ),
    ),
    (
        "deltabvib_service",
        (
            "CurvilinearDeltaBVibJob",
            "TrinityDeltaBVibCorrection",
        ),
    ),
    (
        "dual_representation",
        (
            "CartesianSpectroscopicChannel",
            "CurvilinearSonicChannel",
            "DualRepresentationVibrationalField",
            "RepresentationSelection",
            "VIBRATIONAL_DUAL_REPRESENTATION_SCHEMA",
            "prepare_variational_handoff",
        ),
    ),
    (
        "isotopic_internal_qff",
        (
            "NonredundantInternalCubicField",
            "internal_cubic_from_parent_normal_field",
            "normal_cubic_from_internal_field",
        ),
    ),
    (
        "harmonic_field",
        (
            "HarmonicFieldTransform",
            "primitive_b_matrix_from_manifest",
            "transform_cartesian_hessian",
        ),
    ),
    (
        "property_spectroscopy",
        (
            "GAUSSIAN_IR_CONVERSION_KM_MOL",
            "TRINITY_IR_INTENSITY_SCHEMA",
            "TrinityIRIntensityResult",
            "dipole_surface_and_ir_from_gaussian_fchk",
            "harmonic_ir_intensities_from_property_surface",
        ),
    ),
)
_LAZY_EXPORTS = {
    name: (module_name, name) for module_name, names in _LAZY_EXPORT_GROUPS for name in names
}
_LAZY_EXPORTS["MethodAwareCurvilinearDeltaBVibJob"] = (
    "curvilinear_rovib",
    "CurvilinearDeltaBVibJob",
)
_LAZY_EXPORTS["PersistentTrinityDeltaBVibService"] = (
    "deltabvib_service",
    "TrinityDeltaBVibService",
)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "common_quadratic_correlation", "common_quadratic_operator_correlation", "retain_common_quadratic_pairs", "screen_common_quadratic_pairs", "common_quadratic_dressed_correlation",
    "common_vci_lines", "common_vci_operator_lines", "common_vci_nonradiative_rate", "common_vci_radiative_emission_rate",
    "CommonModeVCI", "CommonVCIHamiltonian", "CommonVCIStates", "CommonVCIProperty", "solve_common_mode_vci", "common_vci_transition_orders",
    "CommonQuadraticFrame", "joint_quadratic_modes", "common_quadratic_model", "common_quadratic_coefficients", "common_chart_states_from_harmonic",
    "ElectronicSolvationSample", "VerticalSolvationTransition",
    "vertical_solvation_transition", "NonequilibriumCartesianEnergy",
    "RadiativeCascade", "solve_downward_cascade",
    "exact_cascade_frontier", "refine_exact_cascade_reference",
    "GVPT2RateReference", "gvpt2_rate_reference",
    "EffectiveQuantumRateModel", "fit_gvpt2_rate_model",
    "fit_selected_gvpt2_rate_model",
    "ConditionalQuantumRateModel", "fit_conditional_quantum_rates", "fit_transitionwise_quantum_rates",
    "PolynomialQuantumEnergyModel", "HybridQuantumRateModel", "fit_polynomial_quantum_energies", "fit_hybrid_quantum_rates",
    "with_exact_energy_window",
    'contract_modal_coriolis',
    'dress_lam_harmonic_bath_rotation',
    "COMPUTED_CUBIC_JOB_SCHEMA",
    "CARTESIAN_GRADIENT_DELTABVIB_METHOD",
    "CARTESIAN_HESSIAN_DELTABVIB_METHOD",
    "CURVILINEAR_DELTABVIB_SCHEMA",
    "CallableDeltaBVibJob",
    "CartesianSpectroscopicChannel",
    "CurvilinearAlphaComponents",
    "CurvilinearDeltaBVibJob",
    "CurvilinearDeltaBVibResult",
    "CurvilinearIsotopologueDefinition",
    "CurvilinearIsotopologueState",
    "CurvilinearSonicChannel",
    "DualRepresentationVibrationalField",
    "GAUSSIAN_IR_CONVERSION_KM_MOL",
    "HarmonicFieldTransform",
    "LocalAlgebraicIRResult",
    "LocalAlgebraicXResult",
    "LocalPolyadIRResult",
    "ESALCIGVPT2ProtocolResult",
    "MethodAwareCurvilinearDeltaBVibJob",
    "NonredundantInternalCubicField",
    "QuadraticChartConnection",
    "CommonGaugeVP2TD",
    "transport_sonic_field_second_order",
    "transport_sonic_property_second_order",
    "PersistentTrinityDeltaBVibService",
    "RepresentationSelection",
    "SONIC_GRADIENT_DELTABVIB_METHOD",
    "SONIC_HESSIAN_DELTABVIB_METHOD",
    "SONIC_INTERNAL_DELTABVIB_METHOD",
    "SemiDiagonalCubicField",
    "SonicRovibrationalConstants",
    "SonicDiagonalPropertyExpectation",
    "TRINITY_IR_INTENSITY_SCHEMA",
    "TRINITY_LONG_NAME",
    "TrinityDeltaBVibCorrection",
    "TrinityDeltaBVibMethod",
    "TrinityDeltaBVibService",
    "TrinityIRIntensityResult",
    "TransportedResonancePolyad",
    "VIBRATIONAL_DUAL_REPRESENTATION_SCHEMA",
    "prepare_variational_handoff",
    "build_lock_surface_from_esalc_scan",
    "build_normal_mode_anharmonic_field",
    "build_normal_mode_anharmonic_field_from_architect",
    "build_normal_mode_qff_from_calculator",
    "acquire_computed_cubic_job",
    "acquire_computed_cubic_job_from_hessians",
    "acquire_sonic_rovibrational_constants",
    "computed_cubic_job_from_dict",
    "computed_cubic_job_to_dict",
    "compare_periodic_property_perturbation_dvr",
    "curvilinear_alpha_from_internal_field",
    "curvilinear_deltabvib_for_isotopologues",
    "curvilinear_deltabvib_from_alpha",
    "curvilinear_deltabvib_from_definitions",
    "curvilinear_deltabvib_from_dict",
    "curvilinear_deltabvib_from_internal_field",
    "curvilinear_deltabvib_to_dict",
    "curvilinear_isotopologue_state_from_internal_field",
    "dipole_surface_and_ir_from_gaussian_fchk",
    "harmonic_ir_intensities_from_property_surface",
    "internal_cubic_from_parent_normal_field",
    "normal_cubic_from_internal_field",
    "primitive_b_matrix_from_manifest",
    "quadratic_chart_connection",
    "solve_common_gauge_vp2td",
    "assess_chart_consistency",
    "assess_large_amplitude_dvr",
    "read_curvilinear_deltabvib_results",
    "read_computed_cubic_job",
    "rovibrational_constants_from_modal_derivatives",
    "semidiagonal_cubic_from_modal_gradient_stencil",
    "solve_local_algebraic_ir",
    "solve_local_algebraic_x",
    "solve_local_polyad_ir",
    "run_esalci_gvpt2_protocol",
    "run_esalci_gvpt2_smith_metric_protocol",
    "run_esalci_gvpt2_smith_metric_eckart_protocol",
    "run_esalci_gvpt2_gdv_frame_protocol",
    "sonic_vpt2_x_matrix",
    "sonic_diagonal_property_expectation",
    "transform_cartesian_hessian",
    "transport_resonance_polyad",
    "transport_gvpt2_polyads",
    "write_curvilinear_deltabvib_results",
    "write_computed_cubic_job",
    "write_curvilinear_deltabvib_to_xyzin",
]

__all__.extend(("ESALCHarmonicState","HarmonicVibronicModel","harmonic_vibronic_from_esalc","harmonic_strength_sum_rule","harmonic_vibronic_ti","harmonic_vibronic_occupations","harmonic_vibronic_correlation","harmonic_vibronic_td"))

__all__.append("run_esalc_vibronic_workflow")

__all__.extend(("harmonic_radiative_emission_rate","harmonic_nonradiative_rate","sam_vpt2_nonradiative_rate","spin_orbit_mixed_electronic_states"))

__all__.append("harmonic_resonance_raman")

__all__.append("run_esalc_resonance_raman_workflow")

__all__.extend(('harmonic_operator_spectrum', 'harmonic_circular_spectrum',
                'harmonic_two_photon_spectrum', 'harmonic_mcd_spectrum', 'harmonic_mcd_manifold_spectrum', 'run_esalc_optical_workflow'))

__all__.extend(('read_gaussian_vibronic_data', 'align_gaussian_vibronic_data',
                'gaussian_force_field_in_esalc', 'gaussian_esalc_vibronic_model'))

__all__.append('harmonic_vibronic_ti_adaptive')

__all__.extend(('SAMVPT2Field', 'SAMVPT2Wavefunction', 'NonresonantVPT2Error', 'sam_vpt2_wavefunction', 'vpt2_vibronic_ti', 'run_esalc_vpt2_vibronic_workflow'))

__all__.extend(('HDCMatrixResult', 'SAMHDCResult', 'hdc_coupling_moduli',
    'complete_hdc_hamiltonian', 'solve_sam_hdcpt2', 'sam_polynomial_operator',
    'hdc_transition_spectrum', 'sam_thermal_populations', 'audit_sam_hdc_convergence',
    'sam_quadratic_energy_model', 'hdc_vibronic_ti', 'run_esalc_hdc_workflow'))

__all__.append('complete_hdc_normal_form')

__all__.extend(('TransitionStateField', 'TransitionStateNormalForm', 'solve_transition_state_normal_form', 'transition_state_channel_matrix', 'transition_state_channels', 'transition_state_transmission', 'transition_state_transform_operator', 'run_transition_state_workflow'))

__all__.extend(('TransitionStateHDCResult', 'solve_transition_state_hdc', 'transition_state_hdc_operator'))

__all__.extend(('HDCPropertyExpansion', 'hdc_property_matrix', 'hdc_property_spectrum', 'hdc_property_bilinear', 'hdc_harmonic_overlap'))

__all__.extend(("LadderPolynomial", "ladder_polynomial_from_qp", "contact_property_orders",
    "OccupationTransitions", "OccupationTransitionOperator", "occupation_transition_lines",
    "occupation_dipole_emission", "occupation_strength_observable"))

__all__.extend(("diagonal_anharmonicity_from_assigned_levels", "sam_vpt2_transition_strength_callback",
    "wang_landau_nonradiative_rate", "importance_nonradiative_rate"))

__all__.extend(('FiniteVibrationalStates', 'project_anharmonic_potential', 'solve_quartic_variational',
    'embed_variational_states', 'harmonic_overlap_matrix', 'variational_vibronic_transitions'))
__all__.append('audit_quartic_confinement')
__all__.append('solve_local_quartic_ground_state')
__all__.extend(('gaussian_number_moment', 'truncate_local_ground_state_projection'))

__all__.extend(('VariationalSpectatorObservable', 'MarginalSpectatorObservable', 'variational_spectator_nonradiative_rate'))

__all__.extend(("solve_local_potential_ground_state", "NativeCartesianModalPotential", "native_modal_displacements_angstrom", "export_native_modal_grid", "run_native_pes_variational_protocol"))
__all__.append('build_native_modal_hamiltonian')
__all__.append('sample_connected_modal_cuts')
__all__.extend(('build_native_mode_extension', 'native_core_quadrature_coordinates', 'acquire_harmonic_extension_spectrum'))

__all__.extend((
    "AnharmonicityContribution", "AnharmonicityDiagnostics", "sonic_vpt2_x_diagnostics",
    "RamanDerivativeLayout", "validate_cartesian_polarizability_jacobian",
    "decode_polarizability_derivatives", "polarizability_jacobian_from_gaussian_fchk",
))

# Historical module names forward to their canonical implementation owners.
from matrix_core.module_aliases import register_module_aliases as _register_module_aliases

_register_module_aliases({'matrix_trinity.keymaker_isotope_acquisition': ('matrix_trinity.isotope_acquisition', False), 'matrix_trinity.survibfit': ('matrix_trinity.fitting_kernel', True), 'matrix_trinity.survibfit.auto_report_pipeline': ('matrix_smith.coordinate_kernel.auto_report_pipeline', False), 'matrix_trinity.survibfit.basis': ('matrix_trinity.fitting_kernel.basis', False), 'matrix_trinity.survibfit.fit': ('matrix_trinity.fitting_kernel.fit', False), 'matrix_trinity.survibfit.fragment_delta_correction': ('matrix_trinity.fragment_delta_correction', False), 'matrix_trinity.survibfit.io': ('matrix_trinity.fitting_kernel.io', False), 'matrix_trinity.survibfit.modify_geom': ('matrix_smith.coordinate_kernel.modify_geom', False), 'matrix_trinity.survibfit.pipeline': ('matrix_smith.coordinate_kernel.pipeline', False), 'matrix_trinity.survibfit.primitives': ('matrix_smith.coordinate_kernel.primitives', False), 'matrix_trinity.survibfit.symmetry_local': ('matrix_smith.coordinate_kernel.symmetry_local', False), 'matrix_trinity.survibfit.terms': ('matrix_trinity.fitting_kernel.terms', False), 'matrix_trinity.survibfit.topo_groups': ('matrix_smith.coordinate_kernel.topo_groups', False), 'matrix_trinity.survibfit.transforms': ('matrix_smith.coordinate_kernel.transforms', False), 'matrix_trinity.survibfit.vib_cli': ('matrix_trinity.fitting_kernel.vib_cli', False), 'matrix_trinity.survibfit.vibrational_internal': ('matrix_trinity.fitting_kernel.vibrational_internal', False), 'matrix_trinity.survibfit.weights': ('matrix_trinity.fitting_kernel.weights', False)})

__all__.extend(("ComplementaryFrequencies", "complementary_oniom_frequencies"))

__all__.extend(("matched_partition_moments", "matched_occupation_partition", "MixedTransverseModel", "mixed_from_gvpt2", "mixed_transverse_partition"))

__all__.extend(("QuantumDynamics", "LadderDynamicsOperator", "contact_dynamics_property",
                "sum_dynamics_operators", "correlation_emission_spectrum", "SecularRadiativeDynamics",
                "rotate_dynamics_operator", "gvpt2_quantum_dynamics", "gvpt2_emission_packets", "stationary_emission_correlation",
                "reduced_field_dynamics_properties"))

__all__.append("NativeResonantProperty")

__all__.extend(("VibrationalModePartition", "BlockVibrationalModel", "BlockTransitionProperty", "solve_mode_blocks"))

__all__.extend(("NativePreparationCache","save_vibronic_columns"))

__all__.extend(("radiative_rate_from_td_density", "harmonic_radiative_emission_rate_td"))

__all__.extend(('zero_temperature_gaussian_blocks', 'convolve_gaussian_block_lines', 'gaussian_block_line_strengths'))

__all__.extend(('symmetry_forbidden_scalar', 'frozen_born_huang_operator', 'symmetry_project_cartesian_nac'))
