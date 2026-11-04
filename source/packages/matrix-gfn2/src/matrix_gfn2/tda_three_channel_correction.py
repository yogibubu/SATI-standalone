"""Three explicit n/sigma/pi endpoints for vertical TDA postprocessing only."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .electronic_state import GFN2OrbitalState
from .stda import ClosedShellSTDAResult
from .sf_tda import HARTREE_TO_EV
from .tda_angular_correction import (
    TDAAngularCorrectionParameters, TDAAngularEnergyCorrection, _array,
    correct_tda_vertical_energies,
)
from .tda_pi_star import TDAPiStarDiagnostics, TDAPiStarPolicy, assess_tda_pi_star_acceptors


@dataclass(frozen=True)
class TDAThreeChannelCorrectionParameters:
    n_to_pi_star_ev: float
    sigma_to_pi_star_ev: float
    pi_to_pi_star_ev: float
    pairing: str = 'same_atom'
    minimum_p_pair_weight: float = 1e-12
    maximum_unresolved_state_weight: float = 1e-8

    def __post_init__(self):
        # Reuse the validation of the existing angular controls and endpoints.
        for endpoint in (self.n_to_pi_star_ev, self.sigma_to_pi_star_ev):
            TDAAngularCorrectionParameters(endpoint,self.pi_to_pi_star_ev,self.pairing,
                self.minimum_p_pair_weight,self.maximum_unresolved_state_weight)
        for name in ('n_to_pi_star_ev','sigma_to_pi_star_ev','pi_to_pi_star_ev',
                     'minimum_p_pair_weight','maximum_unresolved_state_weight'):
            object.__setattr__(self,name,float(getattr(self,name)))

    @property
    def angular_parameters(self):
        return TDAAngularCorrectionParameters(self.n_to_pi_star_ev,self.pi_to_pi_star_ev,self.pairing,
                    self.minimum_p_pair_weight,self.maximum_unresolved_state_weight)


@dataclass(frozen=True)
class TDAThreeChannelEnergyCorrection:
    parameters: TDAThreeChannelCorrectionParameters
    angular_diagnostics: TDAAngularEnergyCorrection
    channel_weights: np.ndarray  # columns n, sigma, pi
    corrections_ev: np.ndarray
    corrected_excitation_energies_hartree: np.ndarray

    def __post_init__(self):
        if not isinstance(self.parameters,TDAThreeChannelCorrectionParameters) or not isinstance(self.angular_diagnostics,TDAAngularEnergyCorrection):
            raise TypeError('three-channel correction requires explicit parameters and angular diagnostics')
        weights=_array(self.channel_weights)
        shifts=_array(self.corrections_ev)
        corrected=_array(self.corrected_excitation_energies_hartree)
        count=len(self.angular_diagnostics.state_indices)
        if weights.shape!=(count,3) or shifts.shape!=(count,) or corrected.shape!=(count,):
            raise ValueError('three-channel result dimensions are inconsistent')
        if np.any((weights<0)|(weights>1)) or not np.allclose(weights.sum(axis=1),1.,atol=1e-12,rtol=0):
            raise ValueError('channel weights must be normalized nonnegative fractions')
        expected=weights@np.array([self.parameters.n_to_pi_star_ev,self.parameters.sigma_to_pi_star_ev,self.parameters.pi_to_pi_star_ev])
        if not np.allclose(shifts,expected,atol=1e-12,rtol=0) or not np.allclose(
            corrected,self.angular_diagnostics.original_excitation_energies_hartree+shifts/HARTREE_TO_EV,atol=1e-12,rtol=0):
            raise ValueError('three-channel energies do not match the reported interpolation')
        object.__setattr__(self,'channel_weights',weights)
        object.__setattr__(self,'corrections_ev',shifts)
        object.__setattr__(self,'corrected_excitation_energies_hartree',corrected)

    @property
    def state_indices(self):
        return self.angular_diagnostics.state_indices

    @property
    def original_excitation_energies_hartree(self):
        return self.angular_diagnostics.original_excitation_energies_hartree

    @property
    def model_id(self):
        return 'TDA-LOCAL-P-THREE-CHANNEL-POSTHOC.v1'


@dataclass(frozen=True)
class TDAPiStarThreeChannelCorrection:
    diagnostics: TDAPiStarDiagnostics
    correction: TDAThreeChannelEnergyCorrection | None

    def __post_init__(self):
        if not isinstance(self.diagnostics,TDAPiStarDiagnostics):
            raise TypeError('pi-star diagnostics required')
        eligible=self.diagnostics.state_indices[self.diagnostics.eligible_states]
        if self.correction is None:
            if len(eligible):
                raise ValueError('eligible states require their correction')
        elif not isinstance(self.correction,TDAThreeChannelEnergyCorrection) or not np.array_equal(self.correction.state_indices,eligible):
            raise ValueError('corrected roots must match the eligible diagnostic roots')


def nonbonding_source_weights(
    state: GFN2OrbitalState, nonbonding_orbitals, *,
    occupied_orbitals=None, configuration_pi_pi_weights=None,
) -> np.ndarray:
    """Project canonical orbitals onto an explicitly identified localized n space.

    Input columns are S-orthonormal occupied nonbonding orbitals in the same
    AO basis. This function computes character weights; it does not identify
    lone pairs from localization alone. Without configuration arguments the
    result is the total n projection per canonical MO. Supplying both the TDA
    occupied indices and angular configuration scores returns conditional
    non-pi n weights: clip(eta_i/(1-c_ia),0,1). Pure-pi configurations use zero
    since their non-pi weight vanishes. This partition remains empirical.
    """
    if state.reference_kind!='restricted-closed-shell':
        raise ValueError('nonbonding source weights require a closed-shell reference')
    localized=np.asarray(nonbonding_orbitals,dtype=float)
    if localized.ndim!=2 or localized.shape[0]!=state.nao or not np.all(np.isfinite(localized)):
        raise ValueError('localized n orbitals must be a finite nao x nn matrix')
    if not np.allclose(localized.T@state.overlap@localized,np.eye(localized.shape[1]),atol=1e-8,rtol=0):
        raise ValueError('localized n orbitals must be S-orthonormal')
    occupied=state.coefficients[:,state.alpha_occupations+state.beta_occupations>1.99]
    residual=localized-occupied@(occupied.T@state.overlap@localized)
    if np.max(np.abs(residual),initial=0.)>1e-8:
        raise ValueError('localized n orbitals must belong to the occupied reference space')
    weights=np.sum(np.abs(localized.T@state.overlap@state.coefficients)**2,axis=0)
    weights=np.clip(weights,0.,1.)
    if occupied_orbitals is None and configuration_pi_pi_weights is None:
        return _array(weights)
    occupied=np.asarray(occupied_orbitals)
    pi=np.asarray(configuration_pi_pi_weights,dtype=float)
    if occupied.ndim!=1 or occupied.dtype.kind not in 'iu' or pi.shape!=occupied.shape or np.any(occupied<0) or np.any(occupied>=state.nmo):
        raise ValueError('configuration indices and angular scores must be matching vectors')
    if not np.all(np.isfinite(pi)) or np.any((pi<0)|(pi>1)):
        raise ValueError('angular configuration scores must lie in [0,1]')
    conditional=np.divide(weights[occupied],1-pi,out=np.zeros_like(pi),where=1-pi>1e-12)
    return _array(np.clip(conditional,0.,1.))


def correct_tda_three_channel_vertical_energies(
    state: GFN2OrbitalState, excited: ClosedShellSTDAResult, *,
    parameters: TDAThreeChannelCorrectionParameters,
    non_pi_n_weights,
    state_indices: Sequence[int] | None = None,
) -> TDAThreeChannelEnergyCorrection:
    """Correct n, sigma and pi channels with declared non-pi source character.

    ``non_pi_n_weights`` is a scalar, a vector per TDA configuration, or a
    matrix with the same shape as the full TDA amplitude array. Values in
    [0,1] specify the n share *conditional on the non-pi component*; sigma is
    its complement. This source assignment is explicit: angular alignment
    alone cannot distinguish a lone pair from a sigma orbital. Fractional
    inputs can come from a separately qualified source-orbital analysis.
    """
    if not isinstance(parameters,TDAThreeChannelCorrectionParameters):
        raise TypeError('three explicit channel parameters are required')
    angular=correct_tda_vertical_energies(state,excited,parameters=parameters.angular_parameters,state_indices=state_indices)
    source=np.asarray(non_pi_n_weights,dtype=float)
    amplitudes=np.asarray(excited.amplitudes)
    if source.ndim==0:
        source=np.broadcast_to(source,(len(angular.state_indices),amplitudes.shape[1]))
    elif source.shape==(amplitudes.shape[1],):
        source=np.broadcast_to(source,(len(angular.state_indices),amplitudes.shape[1]))
    elif source.shape==amplitudes.shape:
        source=source[angular.state_indices]
    else:
        raise ValueError('n weights must be a scalar, per configuration, or match the full amplitudes')
    if not np.all(np.isfinite(source)) or np.any((source<0)|(source>1)):
        raise ValueError('non-pi n weights must lie in [0,1]')
    weights=np.abs(amplitudes[angular.state_indices])**2
    weights[:,~angular.configuration_resolved]=0.
    weights/=weights.sum(axis=1)[:,None]
    pi=angular.configuration_pi_pi_weights
    channels=np.column_stack((np.sum(weights*(1-pi)*source,axis=1),
                              np.sum(weights*(1-pi)*(1-source),axis=1),weights@pi))
    channels=np.clip(channels,0.,1.)
    shifts=channels@np.array([parameters.n_to_pi_star_ev,parameters.sigma_to_pi_star_ev,parameters.pi_to_pi_star_ev])
    return TDAThreeChannelEnergyCorrection(parameters,angular,channels,shifts,
        angular.original_excitation_energies_hartree+shifts/HARTREE_TO_EV)


def correct_pi_star_three_channel_vertical_energies(
    state: GFN2OrbitalState, excited: ClosedShellSTDAResult, *,
    parameters: TDAThreeChannelCorrectionParameters, non_pi_n_weights,
    policy: TDAPiStarPolicy | None = None, state_indices: Sequence[int] | None = None,
) -> TDAPiStarThreeChannelCorrection:
    """Apply the three endpoints only to roots passing the pi-star diagnostic."""
    if not isinstance(parameters,TDAThreeChannelCorrectionParameters):
        raise TypeError('three explicit channel parameters are required')
    diagnostics=assess_tda_pi_star_acceptors(state,excited,policy=policy,state_indices=state_indices)
    selected=diagnostics.state_indices[diagnostics.eligible_states]
    result=(correct_tda_three_channel_vertical_energies(state,excited,parameters=parameters,
            non_pi_n_weights=non_pi_n_weights,state_indices=selected) if len(selected) else None)
    return TDAPiStarThreeChannelCorrection(diagnostics,result)


__all__=['TDAThreeChannelCorrectionParameters','TDAThreeChannelEnergyCorrection',
         'TDAPiStarThreeChannelCorrection','correct_tda_three_channel_vertical_energies',
         'correct_pi_star_three_channel_vertical_energies','nonbonding_source_weights']
