"""APOC — Atomic and Pairwise Observables from the Charge density."""

from .dyson import APOC_DYSON_SCHEMA, DysonOrbital
from .gdv_dyson import read_gdv_ept_dysons
from .dyson_tracking import (
    ElectronicFrameAlignment, DysonMatch, reorient_electronic_state,
    reorient_dyson_orbital, cross_ao_overlap, follow_dyson_orbital,
)

from . import scf_adapters as _scf_adapters
from . import state_reuse as _state_reuse
from .solvation import ApocSolvationIntegrals, apoc_solvation_source, solvation_fock_ao
from .scf_adapters import *  # noqa: F401,F403
from .state_reuse import *  # noqa: F401,F403

from .service import (
    APOC_ANALYSIS_SCHEMA,
    ApocAnalysis,
    analyze_gaussian,
    analyze_gaussian_fchk,
    analyze_molden,
    analyze_orca,
    analyze_pyscf_output,
    analyze_source,
    attach_apoc_analysis,
    load_apoc_analysis,
    save_apoc_analysis,
)
from .population import (
    APOC_CM5_PROVENANCE,
    APOC_MAYER_PROVENANCE,
    mayer_bond_orders_from_electronic_state,
    mayer_spin_bond_orders_from_electronic_state,
    population_observables_from_electronic_state,
    pyscf_molecule_from_electronic_state,
)
from .quality import APOC_CM5_QUALITY_SCHEMA, APOCCM5Quality, evaluate_cm5_quality
from .state import (
    APOC_ELECTRONIC_STATE_SCHEMA,
    ElectronicState,
    ExcitedState,
    GaussianBasis,
    OrbitalSet,
    StateContinuityError,
    StateFingerprint,
    StateMatch,
    StateSubspaceMatch,
    cross_ao_overlap_from_molden,
    electronic_state_from_gaussian_fchk,
    electronic_state_from_molden,
    electronic_state_from_psi4,
    electronic_state_from_pyscf,
    gaussian_basis_from_pyscf,
    gaussian_basis_from_psi4,
    follow_excited_state,
    follow_state_fingerprint,
    follow_state_subspace,
    fingerprint_overlap,
    match_excited_states,
    match_state_fingerprints,
    natural_orbitals,
    recanonicalize_orbitals,
    transition_density_overlap,
)
from .trexio_adapter import read_electronic_state_trexio, write_electronic_state_trexio
from .std2_adapter import electronic_state_from_std2

__all__ = [
    "ElectronicFrameAlignment", "DysonMatch", "reorient_electronic_state",
    "reorient_dyson_orbital", "cross_ao_overlap", "follow_dyson_orbital",
    "APOC_DYSON_SCHEMA",
    "DysonOrbital",
    "read_gdv_ept_dysons",
    "ApocSolvationIntegrals",
    "apoc_solvation_source",
    "solvation_fock_ao",
    "APOC_CM5_PROVENANCE",
    "APOC_MAYER_PROVENANCE",
    "APOC_ANALYSIS_SCHEMA",
    "APOC_ELECTRONIC_STATE_SCHEMA",
    "APOC_CM5_QUALITY_SCHEMA",
    "APOCCM5Quality",
    "ApocAnalysis",
    "ElectronicState",
    "ExcitedState",
    "GaussianBasis",
    "OrbitalSet",
    "StateContinuityError",
    "StateFingerprint",
    "StateMatch",
    "StateSubspaceMatch",
    "analyze_gaussian",
    "analyze_gaussian_fchk",
    "analyze_molden",
    "analyze_orca",
    "analyze_pyscf_output",
    "analyze_source",
    "attach_apoc_analysis",
    "cross_ao_overlap_from_molden",
    "electronic_state_from_gaussian_fchk",
    "electronic_state_from_molden",
    "electronic_state_from_std2",
    "electronic_state_from_psi4",
    "electronic_state_from_pyscf",
    "evaluate_cm5_quality",
    "follow_excited_state",
    "follow_state_fingerprint",
    "follow_state_subspace",
    "fingerprint_overlap",
    "gaussian_basis_from_pyscf",
    "gaussian_basis_from_psi4",
    "load_apoc_analysis",
    "match_excited_states",
    "match_state_fingerprints",
    "mayer_bond_orders_from_electronic_state",
    "mayer_spin_bond_orders_from_electronic_state",
    "natural_orbitals",
    "population_observables_from_electronic_state",
    "pyscf_molecule_from_electronic_state",
    "read_electronic_state_trexio",
    "recanonicalize_orbitals",
    "save_apoc_analysis",
    "transition_density_overlap",
    "write_electronic_state_trexio",
]

__all__ += list(_state_reuse.__all__) + list(_scf_adapters.__all__)
