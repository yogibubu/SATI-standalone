"""VPT2/VCI contracts with optional scientific adapters loaded lazily."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "DavidsonSettings",
    "DavidsonResult",
    "AnharmonicInput",
    "FCHKData",
    "ForceFieldSource",
    "HybridModeBand",
    "HybridVibrationalSpectrum",
    "HDCPT2Result",
    "NormalModeChemicalCharacter",
    "PathModeSelection",
    "PathModeSelectionSettings",
    "QuarticForceField",
    "ORACLE_XYZ_VPT2_VCI_SCHEMA",
    "ScientificValidationError",
    "VCIBlockInfo",
    "VCIOptions",
    "VCIRequest",
    "VCIResult",
    "VCIStateContribution",
    "VariationalBand",
    "VPT2VCIInventory",
    "VPT2VCIRun",
    "VPT2VCISection",
    "VPT2Result",
    "VPT2State",
    "VPT2VCIComparison",
    "VPT2VCIComparisonRow",
    "VPT2VCIFrequencyRow",
    "VPT2VCIModeContribution",
    "VPT2VCIOutputSnapshot",
    "VPT2VCIReport",
    "anharmonic_input_from_gaussian_fchk",
    "analyze_xh_bend_iijj_support",
    "assemble_hybrid_spectrum",
    "build_vci_hamiltonian",
    "collect_vpt2_vci_outputs",
    "collect_vpt2_vci_outputs_from_xyzin",
    "centered_path_tangent",
    "compare_vpt2_vci",
    "compare_vpt2_vci_from_anharmonic_input",
    "davidson_lowest",
    "detected_vpt2_vci_outputs",
    "force_field_from_anharmonic_input",
    "format_vpt2_vci_report",
    "generate_vibrational_basis",
    "inventory_vpt2_vci_backends",
    "load_force_field",
    "mass_weighted_align",
    "lower_to_symmetric",
    "normal_mode_internal_ped",
    "parse_vpt2_vci_section",
    "read_vpt2_vci_comparison",
    "read_vpt2_vci_frequencies",
    "read_vpt2_vci_mode_contributions",
    "read_vpt2_vci_section",
    "read_gaussian_fchk_qff",
    "read_indexed_qff_text",
    "refresh_vpt2_vci_section",
    "run_python_vci_from_gaussian_fchk",
    "run_vpt2_vci_report",
    "run_gfn2_zaff_hdcpt2",
    "select_path_mode_block",
    "solve_vci",
    "solve_hdcpt2",
    "solve_hdcpt2_from_anharmonic_input",
    "solve_vci_from_anharmonic_input",
    "solve_vpt2",
    "solve_vpt2_from_anharmonic_input",
    "validate_force_field",
    "vpt2_vci_output_summary_lines",
    "vpt2_vci_section_from_run",
    "vpt2_vci_section_lines",
    "vpt2_vci_csv_tables",
    "write_csv_tables",
    "write_vpt2_vci_manifest",
    "write_vpt2_vci_section",
    "zero_anharmonic_force_field",
    "zaff_normal_mode_chemical_characters",
    "ZAFF_NORMAL_QFF_SCHEMA",
    "ZAFF_IIJJ_SUPPORT_SCHEMA",
    "NormalModeQFFDiagnostics",
    "NormalModeQFFFitConfig",
    "NormalModeQFFFitResult",
    "dimensionless_normal_displacements",
    "compare_reduced_normal_qff",
    "fit_normal_mode_qff",
    "fit_normal_mode_qff_from_zaff",
    "full_cubic_indices",
    "semidiagonal_quartic_indices",
    "write_indexed_qff",
    "write_normal_mode_qff_result",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module("._api", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
