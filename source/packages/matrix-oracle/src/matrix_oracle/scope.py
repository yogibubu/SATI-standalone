"""Frozen publication boundary for ORACLE inside MATRIX."""

from __future__ import annotations

from typing import Any


ORACLE_SCOPE_SCHEMA = "matrix.oracle.scope.v2"

ORACLE_OWNED_CAPABILITIES = (
    "geometry_import_and_normalization",
    "continuous_and_discrete_topology",
    "fragments_pseudobonds_rings_and_aromaticity",
    "persistent_aromatic_assignment_and_local_ring_planarity_projection",
    "orientation_covariant_symmetry_and_thresholded_projection",
    "continuous_sigma_pi_pi_pi_bond_indices",
    "effective_atomic_numbers_synthons_and_atom_classes",
    "redundant_primitive_internal_coordinates_and_wilson_b_matrix",
    "xtb_simple_primitive_initial_hessian_and_anc_conditioning",
    "primitive_space_structural_improvement_with_cv_conjugation_and_selected_pairs",
    "all_primitive_fragment_differential_geometry_correction",
)

DOWNSTREAM_OWNERSHIP = {
    "SONIC construction from the primitive Wilson B matrix": "SMITH",
    "SONIC SH realization from the ORACLE initial-Hessian operator": "SMITH",
    "internal-to-Cartesian realization, geometry optimization and scans": "LINK",
    "semiexperimental refinement without coordinate back-transformation": "MORPHEUS",
    "B-prime, QM/force-field Hessian transformation, force-field construction and ZAFF": "ARCHITECT",
    "PES-point proposal through the LINK protocol": "SENTINEL",
    "multilevel derivative assembly and vibrational perturbation theory": "TRINITY",
}

ORACLE_EXCLUDED_CAPABILITIES = (
    "force_field_parameter_fitting",
    "hessian_coupling_selection",
    "zaff_compilation_and_runtime",
    "global_or_local_geometry_optimization",
    "normal_mode_and_vibrational_property_analysis",
    "internal_to_cartesian_realization",
    "b_matrix_first_derivative",
)


def oracle_scope_contract() -> dict[str, Any]:
    """Return the machine-readable ownership boundary used by docs and reports."""

    return {
        "schema": ORACLE_SCOPE_SCHEMA,
        "owner": "ORACLE",
        "owned_capabilities": list(ORACLE_OWNED_CAPABILITIES),
        "excluded_capabilities": list(ORACLE_EXCLUDED_CAPABILITIES),
        "downstream_ownership": dict(DOWNSTREAM_OWNERSHIP),
        "scientific_model": {
            "observables_with_qm": "CM5 charges + Mayer bond orders",
            "observables_without_qm": (
                "electronegativity charges + Pauling bond orders"
            ),
            "availability_rule": (
                "the QM pair is used only when both complete vectors are available; "
                "otherwise the geometry-only pair is used"
            ),
            "synthon": "one canonical seven-component continuous descriptor",
            "aromaticity": (
                "geometry-first p-orbital eligibility with 4n+2, fused-system and "
                "metal-coordinated eta5 criteria"
            ),
            "aromatic_planarity": (
                "persistent local-ring plane projection with shared-atom averaging"
            ),
            "fragment_differential_geometry": (
                "q_target(low) + [q_fragment(accurate) - q_fragment(low)] for "
                "bonds, angles, dihedrals and out-of-plane coordinates"
            ),
            "cv_posterior": "radius-aware-period-line Gaussian",
            "descriptor_vdw_radii": "Merz-Kollman/Bondi",
            "uff_nonbonded_radii": "UFF only inside the UFF potential",
        },
        "compatibility_policy": (
            "legacy schemas are accepted only as migration input; every new ORACLE "
            "state emits one MATRIX schema and one scientific model, while capabilities "
            "owned by other tools are not exported by the standalone ORACLE public API"
        ),
    }
