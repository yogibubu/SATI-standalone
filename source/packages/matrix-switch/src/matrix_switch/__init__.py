"""SWITCH: internal SMILES and name-to-geometry services for MATRIX.

The public surface is resolved lazily so importing SWITCH does not also load
depiction, numerical geometry, or optional third-party stacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "SWITCH_GEOMETRY_SCHEMA": ("geometry", "SWITCH_GEOMETRY_SCHEMA"),
    "build_cartesian_seed": ("geometry", "build_cartesian_seed"),
    "complete_graph_hydrogens": ("geometry", "complete_graph_hydrogens"),
    "SWITCH_AROMATICITY_SCHEMA": ("aromaticity", "SWITCH_AROMATICITY_SCHEMA"),
    "perceive_aromaticity": ("aromaticity", "perceive_aromaticity"),
    "SWITCH_DEPICTION_SCHEMA": ("depict", "SWITCH_DEPICTION_SCHEMA"),
    "MoleculeDepictionLayout": ("depict", "MoleculeDepictionLayout"),
    "build_molecule_depiction_layout": ("depict", "build_molecule_depiction_layout"),
    "render_molecule_png": ("depict", "render_molecule_png"),
    "render_molecule_svg": ("depict", "render_molecule_svg"),
    "SWITCH_GRAPH_SCHEMA": ("model", "SWITCH_GRAPH_SCHEMA"),
    "SwitchAtom": ("model", "SwitchAtom"),
    "SwitchBond": ("model", "SwitchBond"),
    "SwitchMolecularGraph": ("model", "SwitchMolecularGraph"),
    "graph_from_topology": ("model", "graph_from_topology"),
    "NameResolution": ("names", "NameResolution"),
    "NameResolutionError": ("names", "NameResolutionError"),
    "resolve_name": ("names", "resolve_name"),
    "CommonSubgraphMatch": ("matching", "CommonSubgraphMatch"),
    "find_substructure_matches": ("matching", "find_substructure_matches"),
    "maximum_common_connected_subgraphs": ("matching", "maximum_common_connected_subgraphs"),
    "SWITCH_CANONICAL_SCHEMA": ("canonical", "SWITCH_CANONICAL_SCHEMA"),
    "canonical_atom_ranks": ("canonical", "canonical_atom_ranks"),
    "canonical_graph_key": ("canonical", "canonical_graph_key"),
    "canonical_smiles": ("canonical", "canonical_smiles"),
    "SMARTS_SUBSET_SCHEMA": ("smarts", "SMARTS_SUBSET_SCHEMA"),
    "SmartsSubsetError": ("smarts", "SmartsSubsetError"),
    "parse_smarts": ("smarts", "parse_smarts"),
    "SWITCH_STEREO_ENUMERATION_SCHEMA": ("stereoisomers", "SWITCH_STEREO_ENUMERATION_SCHEMA"),
    "enumerate_stereoisomers": ("stereoisomers", "enumerate_stereoisomers"),
    "ALLOW_RDKIT_FALLBACK": ("backend_policy", "ALLOW_RDKIT_FALLBACK"),
    "STRICT_SWITCH": ("backend_policy", "STRICT_SWITCH"),
    "SWITCH_BACKEND_POLICY_SCHEMA": ("backend_policy", "SWITCH_BACKEND_POLICY_SCHEMA"),
    "resolve_switch_backend_policy": ("backend_policy", "resolve_switch_backend_policy"),
    "SmilesParseError": ("parser", "SmilesParseError"),
    "SwitchUnsupportedFeatureError": ("parser", "SwitchUnsupportedFeatureError"),
    "clear_smiles_parse_cache": ("parser", "clear_smiles_parse_cache"),
    "parse_smiles": ("parser", "parse_smiles"),
    "RDKIT_FALLBACK_SCHEMA": ("rdkit_fallback", "RDKIT_FALLBACK_SCHEMA"),
    "RDKitFallbackUnavailableError": ("rdkit_fallback", "RDKitFallbackUnavailableError"),
    "SwitchFallbackWarning": ("rdkit_fallback", "SwitchFallbackWarning"),
    "rdkit_fallback_available": ("rdkit_fallback", "rdkit_fallback_available"),
    "write_smiles": ("writer", "write_smiles"),
    "name_to_cartesian": ("workflow", "name_to_cartesian"),
    "smiles_to_cartesian": ("workflow", "smiles_to_cartesian"),
    "HbondCandidate": ("validation", "HbondCandidate"),
    "GeometryFileValidation": ("validation", "GeometryFileValidation"),
    "SwitchSeedComparison": ("validation", "SwitchSeedComparison"),
    "SwitchValidation": ("validation", "SwitchValidation"),
    "compare_switch_graphs": ("validation", "compare_switch_graphs"),
    "validate_switch_geometry": ("validation", "validate_switch_geometry"),
    "validate_geometry_file": ("validation", "validate_geometry_file"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
