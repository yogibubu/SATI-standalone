"""Local occupied-orbital diagnostics for posthoc n/sigma source assignment.

This is an explicitly heuristic source analysis. It does not replace supplied
chemical annotations or change the reference state or any TDA operator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .electronic_state import GFN2OrbitalState
from .tda_angular_correction import _array
from .tda_three_channel_correction import nonbonding_source_weights


@dataclass(frozen=True)
class TDASourceCharacterDiagnostics:
    canonical_n_weights: np.ndarray
    localized_occupied_orbitals: np.ndarray
    atom_populations: np.ndarray
    nonbonding_candidate_mask: np.ndarray
    atomic_population_threshold: float
    density_error: float
    localization_sweeps: int
    maximum_sweeps: int

    def __post_init__(self):
        weights=_array(self.canonical_n_weights)
        orbitals=_array(self.localized_occupied_orbitals)
        populations=_array(self.atom_populations)
        mask=np.asarray(self.nonbonding_candidate_mask)
        if weights.ndim!=1 or np.any((weights<0)|(weights>1)) or orbitals.ndim!=2:
            raise ValueError('invalid source-character array dimensions or weights')
        if populations.ndim!=2 or populations.shape[1]!=orbitals.shape[1] or mask.shape!=(orbitals.shape[1],) or mask.dtype.kind!='b':
            raise ValueError('invalid localized-source populations or mask')
        if not 0.5<float(self.atomic_population_threshold)<=1 or not np.isfinite(self.density_error) or self.density_error<0:
            raise ValueError('invalid source-character controls or density error')
        if not 0<=self.localization_sweeps<=self.maximum_sweeps or self.maximum_sweeps<1:
            raise ValueError('invalid localization iteration counts')
        for name,value in [('canonical_n_weights',weights),('localized_occupied_orbitals',orbitals),
                           ('atom_populations',populations),('nonbonding_candidate_mask',_array(mask,dtype=bool))]:
            object.__setattr__(self,name,value)

    @property
    def localization_converged(self):
        # A conservative iteration-budget diagnostic. Reaching the last sweep
        # is not advertised as convergence even if its final gain was small.
        return self.localization_sweeps<self.maximum_sweeps


def analyze_tda_source_character(
    state: GFN2OrbitalState, *, atomic_population_threshold: float=0.80,
    maximum_sweeps: int=60,
) -> TDASourceCharacterDiagnostics:
    """Test a fixed one-centre criterion on Pipek--Mezey occupied orbitals.

    Candidate n orbitals have >=threshold Lowdin population on one atom.
    Polar bonds and multicentre lone pairs can defeat this criterion; returned
    weights are diagnostics, not a default automatic correction assignment.
    Localization uses existing resident code and preserves the SCF density.
    """
    if state.reference_kind!='restricted-closed-shell':
        raise ValueError('source diagnostics require a restricted closed shell')
    if state.nmo!=state.nao:
        raise ValueError('source localization requires a complete square MO basis')
    if isinstance(atomic_population_threshold,(bool,np.bool_)) or not np.isfinite(atomic_population_threshold) or not .5<atomic_population_threshold<=1:
        raise ValueError('atomic population threshold must lie in (0.5,1]')
    if isinstance(maximum_sweeps,(bool,np.bool_)) or not isinstance(maximum_sweeps,(int,np.integer)) or maximum_sweeps<1:
        raise ValueError('maximum_sweeps must be a positive integer')
    from .localized_molecular_orbitals import localize_molecular_orbitals
    occupations=state.alpha_occupations+state.beta_occupations
    if np.any((occupations>.01)&(occupations<1.99)):
        raise ValueError('source localization requires integer occupations')
    localized=localize_molecular_orbitals(state.coefficients,state.overlap,state.ao_atom,occupations,
        maximum_sweeps=maximum_sweeps,objective_tolerance=1e-10,backend='python',population_scheme='lowdin')
    indices=np.flatnonzero(occupations>1.99)
    coefficients=localized.coefficients[:,indices]
    populations=localized.atom_populations[:,indices]
    candidate=np.max(populations,axis=0)>=atomic_population_threshold
    weights=nonbonding_source_weights(state,coefficients[:,candidate])
    return TDASourceCharacterDiagnostics(weights,coefficients,populations,candidate,
        float(atomic_population_threshold),float(localized.density_error),int(localized.sweeps),maximum_sweeps)


__all__=['TDASourceCharacterDiagnostics','analyze_tda_source_character']
